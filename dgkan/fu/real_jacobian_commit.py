"""Real parameter-Jacobian extraction and commit helpers for v22.16."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


EPS = 1.0e-12


@dataclass
class FlatParamSpec:
    names: list[str]
    shapes: list[torch.Size]
    sizes: list[int]

    @property
    def numel(self) -> int:
        return int(sum(self.sizes))


def select_named_parameters(model: torch.nn.Module, selector: str) -> list[tuple[str, torch.nn.Parameter]]:
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if selector == "basis":
        picked = [(n, p) for n, p in params if n.endswith("w1") or ".w1" in n or n == "w1"]
        return picked or params
    if selector == "readout":
        picked = [(n, p) for n, p in params if n.endswith("w2") or ".w2" in n or "readout" in n]
        return picked or params
    return params


def flatten_named_parameters(named_params: Iterable[tuple[str, torch.nn.Parameter]]) -> tuple[torch.Tensor, FlatParamSpec]:
    names: list[str] = []
    shapes: list[torch.Size] = []
    sizes: list[int] = []
    parts: list[torch.Tensor] = []
    for name, p in named_params:
        names.append(name)
        shapes.append(p.shape)
        sizes.append(int(p.numel()))
        parts.append(p.detach().reshape(-1).clone())
    flat = torch.cat(parts) if parts else torch.empty(0)
    return flat, FlatParamSpec(names, shapes, sizes)


def flatten_grad_list(grads: Iterable[torch.Tensor | None], params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for g, p in zip(grads, params):
        if g is None:
            parts.append(torch.zeros_like(p).reshape(-1))
        else:
            parts.append(g.detach().reshape(-1))
    return torch.cat(parts) if parts else torch.empty(0)


@torch.no_grad()
def add_flat_delta(named_params: Iterable[tuple[str, torch.nn.Parameter]], delta: torch.Tensor, *, scale: float = 1.0) -> None:
    offset = 0
    flat = delta.detach()
    for _name, p in named_params:
        n = int(p.numel())
        p.add_(flat[offset : offset + n].reshape_as(p), alpha=float(scale))
        offset += n


def output_jacobian(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    selector: str = "basis",
    max_output_rows: int = 64,
) -> tuple[torch.Tensor, FlatParamSpec, dict[str, Any]]:
    """Return d vec(f(x)) / d selected_params for real model parameters.

    This intentionally uses autograd row extraction. It is a correctness path
    for small basis audits, not the fast official kernel path.
    """
    named = select_named_parameters(model, selector)
    params = [p for _n, p in named]
    _flat, spec = flatten_named_parameters(named)
    logits = model(x).float()
    flat_out = logits.reshape(-1)
    rows = min(int(max_output_rows), int(flat_out.numel()))
    jac_rows: list[torch.Tensor] = []
    for idx in range(rows):
        grads = torch.autograd.grad(flat_out[idx], params, retain_graph=True, allow_unused=True)
        jac_rows.append(flatten_grad_list(grads, params).to(device=logits.device, dtype=logits.dtype))
    jac = torch.stack(jac_rows, dim=0) if jac_rows else torch.empty(0, spec.numel, device=logits.device)
    diag = {
        "selector": selector,
        "jacobian_rows": int(jac.shape[0]),
        "jacobian_cols": int(jac.shape[1]),
        "selected_param_count": len(spec.names),
        "selected_param_numel": spec.numel,
        "selected_param_names": ";".join(spec.names),
    }
    return jac, spec, diag


def solve_linearized_commit(
    jacobian: torch.Tensor,
    source: torch.Tensor,
    *,
    damping: float = 1.0e-3,
) -> tuple[torch.Tensor, dict[str, Any]]:
    j = jacobian.detach().float()
    z = source.detach().float().reshape(-1).to(j.device)[: j.shape[0]]
    if int(j.shape[0]) <= int(j.shape[1]):
        lhs = j @ j.T + float(damping) * torch.eye(j.shape[0], device=j.device, dtype=j.dtype)
        status = "dual_solve"
        try:
            coeff = torch.linalg.solve(lhs, z)
        except RuntimeError:
            coeff = torch.linalg.lstsq(lhs, z.unsqueeze(1)).solution.squeeze(1)
            status = "dual_lstsq"
        delta = j.T @ coeff
    else:
        lhs = j.T @ j + float(damping) * torch.eye(j.shape[1], device=j.device, dtype=j.dtype)
        rhs = j.T @ z
        status = "primal_solve"
        try:
            delta = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            delta = torch.linalg.lstsq(lhs, rhs.unsqueeze(1)).solution.squeeze(1)
            status = "primal_lstsq"
    effect = j @ delta
    residual = torch.linalg.vector_norm(effect - z).div(torch.linalg.vector_norm(z).clamp_min(EPS))
    denom = torch.linalg.vector_norm(effect).clamp_min(EPS) * torch.linalg.vector_norm(z).clamp_min(EPS)
    cosine = torch.dot(effect, z).div(denom)
    return delta, {
        "linearized_commit_status": status,
        "basis_operator_residual": float(residual.item()),
        "basis_projection_residual": float(residual.item()),
        "basis_projection_cosine": float(cosine.item()),
        "basis_update_norm": float(torch.linalg.vector_norm(delta).item()),
    }


def finite_difference_gradcheck(
    model: torch.nn.Module,
    x: torch.Tensor,
    delta: torch.Tensor,
    *,
    selector: str = "basis",
    eps: float = 1.0e-4,
    max_output_rows: int = 64,
) -> dict[str, Any]:
    named = select_named_parameters(model, selector)
    jac, _spec, diag = output_jacobian(model, x, selector=selector, max_output_rows=max_output_rows)
    d = delta.detach().float().to(jac.device)
    if d.numel() != jac.shape[1]:
        d = d[: jac.shape[1]] if d.numel() > jac.shape[1] else torch.nn.functional.pad(d, (0, jac.shape[1] - d.numel()))
    with torch.no_grad():
        base = model(x).float().reshape(-1)[: jac.shape[0]].detach().clone()
        add_flat_delta(named, d, scale=eps)
        plus = model(x).float().reshape(-1)[: jac.shape[0]].detach().clone()
        add_flat_delta(named, d, scale=-eps)
    fd = (plus - base) / float(eps)
    lin = jac @ d
    rel = torch.linalg.vector_norm(fd - lin).div(torch.linalg.vector_norm(fd).clamp_min(EPS))
    return {
        **diag,
        "basis_gradcheck_rel_error": float(rel.item()),
        "native_vs_autograd_gradcheck": int(float(rel.item()) <= 1.0e-3),
        "gradcheck_eps": float(eps),
        "gradcheck_path": "autograd_jacobian_vs_finite_difference_real_params",
    }


__all__ = [
    "FlatParamSpec",
    "add_flat_delta",
    "finite_difference_gradcheck",
    "flatten_named_parameters",
    "output_jacobian",
    "select_named_parameters",
    "solve_linearized_commit",
]
