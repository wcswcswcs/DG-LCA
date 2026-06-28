"""Downstream sensitivity pullback checks for KAN edge parameters."""

from __future__ import annotations

from typing import Any

import torch


def _functional_call(model: Any, params: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    try:
        from torch.func import functional_call

        return functional_call(model, params, (x,))
    except Exception:
        from torch.nn.utils.stateless import functional_call

        return functional_call(model, params, (x,))


def parameter_jvp_fd_error(
    model: Any,
    x: torch.Tensor,
    param_name: str = "w2",
    max_entries: int = 8,
    eps: float = 1.0e-4,
) -> dict[str, float]:
    params = {name: p.detach().clone().requires_grad_(True) for name, p in model.named_parameters()}
    if param_name not in params:
        param_name = next(iter(params))
    p0 = params[param_name]
    delta = torch.zeros_like(p0)
    flat = delta.reshape(-1)
    take = min(int(max_entries), int(flat.numel()))
    if take == 0:
        return {"edge_JVP_fd_error": 0.0, "edge_JVP_fd_abs_error": 0.0, "jvp_norm": 0.0, "fd_norm": 0.0}
    flat[:take] = torch.linspace(0.25, 1.0, take, dtype=delta.dtype, device=delta.device)
    delta = delta / delta.norm().clamp_min(1.0e-12)

    def fn(p: torch.Tensor) -> torch.Tensor:
        local = dict(params)
        local[param_name] = p
        return _functional_call(model, local, x).reshape(-1)

    with torch.no_grad():
        fd = (fn(p0 + float(eps) * delta) - fn(p0 - float(eps) * delta)) / (2.0 * float(eps))
    _out, jvp = torch.autograd.functional.jvp(fn, (p0,), (delta,), create_graph=False, strict=False)
    abs_err = torch.linalg.norm(jvp.detach() - fd.detach())
    rel_err = abs_err / torch.linalg.norm(fd.detach()).clamp_min(1.0e-12)
    return {
        "edge_JVP_fd_error": float(rel_err.detach().cpu().item()),
        "edge_JVP_fd_abs_error": float(abs_err.detach().cpu().item()),
        "jvp_norm": float(torch.linalg.norm(jvp.detach()).cpu().item()),
        "fd_norm": float(torch.linalg.norm(fd.detach()).cpu().item()),
    }


def downstream_sensitivity_gram(
    model: Any,
    x: torch.Tensor,
    param_name: str = "w2",
    max_cols: int = 12,
    ridge: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    params = {name: p.detach().clone().requires_grad_(True) for name, p in model.named_parameters()}
    if param_name not in params:
        param_name = next(iter(params))
    p0 = params[param_name]
    flat_n = int(p0.numel())
    cols = []

    def fn(p: torch.Tensor) -> torch.Tensor:
        local = dict(params)
        local[param_name] = p
        return _functional_call(model, local, x).reshape(-1)

    for idx in range(min(int(max_cols), flat_n)):
        delta = torch.zeros_like(p0).reshape(-1)
        delta[idx] = 1.0
        delta = delta.reshape_as(p0)
        _out, jvp = torch.autograd.functional.jvp(fn, (p0,), (delta,), create_graph=False, strict=False)
        cols.append(jvp.detach().reshape(-1).to(dtype=torch.float64))
    if not cols:
        gram = torch.zeros(0, 0, dtype=torch.float64, device=x.device)
    else:
        j = torch.stack(cols, dim=1)
        gram = j.transpose(0, 1) @ j / max(1, int(x.shape[0]))
    if int(gram.numel()) == 0:
        return gram, {
            "downstream_sensitivity_symmetry_error": 0.0,
            "downstream_sensitivity_PSD_min": 0.0,
            "downstream_sensitivity_trace": 0.0,
            "downstream_sensitivity_effective_rank": 0.0,
        }
    sym = 0.5 * (gram + gram.transpose(0, 1))
    eye = torch.eye(int(sym.shape[0]), dtype=sym.dtype, device=sym.device)
    evals = torch.linalg.eigvalsh(sym + float(ridge) * eye)
    vals = torch.clamp(evals, min=0.0)
    trace = vals.sum().clamp_min(1.0e-12)
    probs = vals / trace
    eff = torch.exp(-(probs * torch.log(probs.clamp_min(1.0e-12))).sum())
    return gram, {
        "downstream_sensitivity_symmetry_error": float(torch.linalg.norm(gram - gram.transpose(0, 1)).detach().cpu().item()),
        "downstream_sensitivity_PSD_min": float(evals.min().detach().cpu().item()),
        "downstream_sensitivity_trace": float(vals.sum().detach().cpu().item()),
        "downstream_sensitivity_effective_rank": float(eff.detach().cpu().item()),
    }
