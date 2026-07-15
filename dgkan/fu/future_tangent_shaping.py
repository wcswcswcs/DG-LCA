"""Future-tangent shaping helpers for v23.17 small-model audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from dgkan.fu.basis_covariant_nullspace import EPS, NullSolveResult, g_norm, null_ratio, select_soft_null_mu
from dgkan.fu.compositional_edge_tangent import apply_j, build_tangent_cache, flatten_coeffs, unflatten_coeffs


ResidualFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class FutureTangentCovector:
    flat: torch.Tensor
    scalar_value: float
    grad_norm: float
    u_detached_flag: int
    r_detached_flag: int
    nan_inf: int


@dataclass
class ShapingDirection:
    coeffs: list[torch.Tensor]
    flat: torch.Tensor
    raw_flat: torch.Tensor
    scale: float
    task_g_norm: float
    raw_g_norm: float
    shaping_g_norm: float
    current_output_ratio: float
    null_ratio: float
    mu: float
    feasible: int
    solve_residual: float


def detach_coeff_delta(delta: list[torch.Tensor]) -> list[torch.Tensor]:
    return [d.detach().clone().to(dtype=torch.float64) for d in delta]


def future_tangent_covector(
    model: Any,
    x_witness: torch.Tensor,
    y_witness: torch.Tensor,
    u_source: list[torch.Tensor],
    residual_fn: ResidualFn,
) -> FutureTangentCovector:
    """Compute grad_theta <r_W, J_W(theta) u_source> with r/u detached."""
    model.zero_grad(set_to_none=True)
    cache = build_tangent_cache(model, x_witness)
    residual = residual_fn(cache.logits, y_witness).detach()
    u_detached = detach_coeff_delta(u_source)
    pred = apply_j(model, cache, u_detached).to(dtype=torch.float64)
    scalar = (residual.to(dtype=torch.float64) * pred).sum() / float(max(1, int(residual.shape[0])))
    grads = torch.autograd.grad(scalar, list(model.coeffs), retain_graph=False, create_graph=False, allow_unused=True)
    dense_grads: list[torch.Tensor] = []
    nan_inf = 0
    for param, grad in zip(model.coeffs, grads):
        if grad is None:
            g = torch.zeros_like(param, dtype=torch.float64)
        else:
            g = grad.detach().to(dtype=torch.float64)
        nan_inf += int((~torch.isfinite(g)).sum().detach().cpu().item())
        dense_grads.append(g)
    flat = flatten_coeffs(dense_grads).to(device=cache.logits.device, dtype=torch.float64)
    return FutureTangentCovector(
        flat=flat,
        scalar_value=float(scalar.detach().cpu().item()),
        grad_norm=float(flat.norm().detach().cpu().item()),
        u_detached_flag=int(all(not d.requires_grad for d in u_detached)),
        r_detached_flag=int(not residual.requires_grad),
        nan_inf=int(nan_inf),
    )


def symmetric_future_tangent_covector(
    model: Any,
    x_source: torch.Tensor,
    y_source: torch.Tensor,
    u_source: list[torch.Tensor],
    x_witness: torch.Tensor,
    y_witness: torch.Tensor,
    u_witness: list[torch.Tensor],
    residual_fn: ResidualFn,
) -> tuple[torch.Tensor, dict[str, float]]:
    h_sw = future_tangent_covector(model, x_witness, y_witness, u_source, residual_fn)
    h_ws = future_tangent_covector(model, x_source, y_source, u_witness, residual_fn)
    flat = 0.5 * (h_sw.flat + h_ws.flat.to(device=h_sw.flat.device))
    cos = torch.tensor(0.0, device=flat.device, dtype=torch.float64)
    denom = h_sw.flat.norm() * h_ws.flat.norm().clamp_min(EPS)
    if float(denom.detach().cpu().item()) > EPS:
        cos = (h_sw.flat @ h_ws.flat.to(device=h_sw.flat.device)) / denom
    return flat, {
        "h_S_to_W_scalar": h_sw.scalar_value,
        "h_W_to_S_scalar": h_ws.scalar_value,
        "h_S_to_W_norm": h_sw.grad_norm,
        "h_W_to_S_norm": h_ws.grad_norm,
        "source_witness_h_FT_cosine": float(cos.detach().cpu().item()),
        "u_detached_flag": int(h_sw.u_detached_flag and h_ws.u_detached_flag),
        "r_detached_flag": int(h_sw.r_detached_flag and h_ws.r_detached_flag),
        "hvp_nan_inf": int(h_sw.nan_inf + h_ws.nan_inf),
    }


def scale_to_budget(
    raw: torch.Tensor,
    task_flat: torch.Tensor,
    g: torch.Tensor,
    j_sw: torch.Tensor,
    *,
    rho: float = 0.10,
    kappa: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    rr = raw.reshape(-1).to(device=g.device, dtype=torch.float64)
    uu = task_flat.reshape(-1).to(device=g.device, dtype=torch.float64)
    raw_norm = g_norm(rr, g).clamp_min(EPS)
    task_norm = g_norm(uu, g).clamp_min(EPS)
    unit = rr / raw_norm
    task_out = (j_sw.to(device=g.device, dtype=torch.float64) @ uu.reshape(-1, 1)).norm().clamp_min(EPS)
    raw_out = (j_sw.to(device=g.device, dtype=torch.float64) @ unit.reshape(-1, 1)).norm().clamp_min(EPS)
    g_budget = float(rho) * task_norm
    out_budget = float(kappa) * task_out / raw_out
    scale = torch.minimum(g_budget, out_budget)
    shaped = scale * unit
    shaped_norm = g_norm(shaped, g)
    current_output_ratio = float(((j_sw @ shaped.reshape(-1, 1)).norm() / task_out).detach().cpu().item())
    return shaped.reshape(-1), {
        "scale": float(scale.detach().cpu().item()),
        "task_g_norm": float(task_norm.detach().cpu().item()),
        "raw_g_norm": float(raw_norm.detach().cpu().item()),
        "shaping_g_norm": float(shaped_norm.detach().cpu().item()),
        "shaping_to_task_G_norm_ratio": float((shaped_norm / task_norm).detach().cpu().item()),
        "current_output_ratio": current_output_ratio,
    }


def build_shaping_direction(
    model: Any,
    g: torch.Tensor,
    j_sw: torch.Tensor,
    h: torch.Tensor,
    task_delta: list[torch.Tensor],
    *,
    tau: float = 0.05,
    rho: float = 0.10,
    kappa: float = 0.05,
    mu_min: float = 1.0,
    mu_max: float = 1.0e4,
    bisection_steps: int = 4,
) -> ShapingDirection:
    result: NullSolveResult = select_soft_null_mu(
        g,
        j_sw,
        h,
        tau=float(tau),
        mu_min=float(mu_min),
        mu_max=float(mu_max),
        bisection_steps=int(bisection_steps),
    )
    task_flat = flatten_coeffs(task_delta).to(device=g.device, dtype=torch.float64)
    shaped, scale_diag = scale_to_budget(result.vector, task_flat, g, j_sw, rho=float(rho), kappa=float(kappa))
    coeffs = unflatten_coeffs(shaped, [c.detach().to(dtype=torch.float64) for c in model.coeffs])
    return ShapingDirection(
        coeffs=coeffs,
        flat=shaped,
        raw_flat=result.vector,
        scale=float(scale_diag["scale"]),
        task_g_norm=float(scale_diag["task_g_norm"]),
        raw_g_norm=float(scale_diag["raw_g_norm"]),
        shaping_g_norm=float(scale_diag["shaping_g_norm"]),
        current_output_ratio=float(scale_diag["current_output_ratio"]),
        null_ratio=float(null_ratio(j_sw, shaped, g)),
        mu=float(result.mu),
        feasible=int(result.feasible),
        solve_residual=float(result.solve_residual),
    )

