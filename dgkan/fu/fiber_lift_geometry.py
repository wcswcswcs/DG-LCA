"""Fiber-lift geometry helpers for signal-channel audits.

The functions here are deliberately small and tensor-only.  They implement the
core geometry used by v22.98 without depending on experiment-runner state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from dgkan.fu.layer_composite_metric import EPS, sym


def _as_diag(g_shape: torch.Tensor) -> torch.Tensor:
    g = g_shape.detach().to(dtype=torch.float64).cpu()
    if g.ndim == 1:
        return g.reshape(-1).clamp_min(EPS)
    if g.ndim == 2:
        return torch.diag(g).reshape(-1).clamp_min(EPS)
    raise ValueError("G_shape must be a diagonal vector or a square matrix")


def _as_projector(p_sig: torch.Tensor) -> torch.Tensor:
    p = p_sig.detach().to(dtype=torch.float64).cpu()
    if p.ndim == 1:
        u = p.reshape(-1, 1)
        return sym(u @ u.T)
    if p.ndim != 2:
        raise ValueError("P must be a vector or matrix")
    return sym(p)


def compute_lift_operator(j: torch.Tensor, g_shape: torch.Tensor, p_sig: torch.Tensor) -> torch.Tensor:
    """Return B = G^{-1/2} J^T P J G^{-1/2}.

    ``g_shape`` may be a diagonal vector or a matrix whose diagonal defines the
    local shape cost.  The experiment runner uses a diagonal/block
    approximation, while this function supports full toy checks.
    """

    jj = j.detach().to(dtype=torch.float64).cpu()
    pp = _as_projector(p_sig)
    n_out = min(int(jj.shape[0]), int(pp.shape[0]))
    n_param = min(int(jj.shape[1]), int(_as_diag(g_shape).numel()))
    if n_out <= 0 or n_param <= 0:
        return torch.zeros((0, 0), dtype=torch.float64)
    g = _as_diag(g_shape)[:n_param]
    wj = jj[:n_out, :n_param] / g.sqrt().reshape(1, -1)
    return sym(wj.T @ pp[:n_out, :n_out] @ wj)


def lift_operator_diag(j: torch.Tensor, g_shape: torch.Tensor, p_sig: torch.Tensor) -> torch.Tensor:
    jj = j.detach().to(dtype=torch.float64).cpu()
    pp = _as_projector(p_sig)
    n_out = min(int(jj.shape[0]), int(pp.shape[0]))
    n_param = min(int(jj.shape[1]), int(_as_diag(g_shape).numel()))
    if n_out <= 0 or n_param <= 0:
        return torch.zeros(0, dtype=torch.float64)
    g = _as_diag(g_shape)[:n_param]
    weighted = pp[:n_out, :n_out] @ jj[:n_out, :n_param]
    return ((jj[:n_out, :n_param] * weighted).sum(dim=0) / g).clamp_min(0.0)


def minimum_norm_lift_cost(
    j: torch.Tensor,
    g_shape: torch.Tensor,
    output_direction: torch.Tensor,
    *,
    ridge: float = 1.0e-8,
) -> dict[str, float]:
    """Cost of the minimum-G-norm lift solving J delta ~= u."""

    jj = j.detach().to(dtype=torch.float64).cpu()
    u = output_direction.detach().to(dtype=torch.float64).cpu().reshape(-1)
    n_out = min(int(jj.shape[0]), int(u.numel()))
    n_param = min(int(jj.shape[1]), int(_as_diag(g_shape).numel()))
    if n_out <= 0 or n_param <= 0:
        return {"lift_cost": 0.0, "lift_residual_norm": 0.0, "lift_solution_norm": 0.0}
    jj = jj[:n_out, :n_param]
    u = u[:n_out]
    g = _as_diag(g_shape)[:n_param]
    jginv = jj / g.reshape(1, -1)
    gram = sym(jginv @ jj.T)
    gram = gram + float(ridge) * torch.eye(int(gram.shape[0]), dtype=torch.float64)
    alpha = torch.linalg.pinv(gram) @ u
    delta = (jj.T @ alpha) / g
    residual = jj @ delta - u
    cost = float((delta * g * delta).sum().clamp_min(0.0).item())
    return {
        "lift_cost": cost if math.isfinite(cost) else 0.0,
        "lift_residual_norm": float(residual.norm().item()),
        "lift_solution_norm": float(delta.norm().item()),
    }


def generalized_lift_eig(
    b_task: torch.Tensor,
    b_ctrl: torch.Tensor,
    *,
    ridge: float = 1.0e-6,
    rank: int | None = None,
) -> dict[str, Any]:
    """Solve a symmetric generalized lift-eigen problem."""

    bt = sym(b_task.detach().to(dtype=torch.float64).cpu())
    bc = sym(b_ctrl.detach().to(dtype=torch.float64).cpu())
    n = min(int(bt.shape[0]), int(bt.shape[1]), int(bc.shape[0]), int(bc.shape[1]))
    if n <= 0:
        return {"eigenvalues": torch.zeros(0, dtype=torch.float64), "vectors": torch.zeros((0, 0), dtype=torch.float64), "eigengap": 0.0}
    bt = bt[:n, :n]
    bc = bc[:n, :n] + float(ridge) * torch.eye(n, dtype=torch.float64)
    vals_c, vecs_c = torch.linalg.eigh(sym(bc))
    invsqrt = (vecs_c * vals_c.clamp_min(EPS).rsqrt().reshape(1, -1)) @ vecs_c.T
    rel = sym(invsqrt @ bt @ invsqrt)
    vals, vecs = torch.linalg.eigh(rel)
    order = torch.argsort(vals, descending=True)
    vals = vals[order]
    vecs = invsqrt @ vecs[:, order]
    if rank is not None:
        vals = vals[: int(rank)]
        vecs = vecs[:, : int(rank)]
    gap = float(vals[0].item() / max(float(vals[1].item()), EPS)) if int(vals.numel()) >= 2 else 0.0
    return {"eigenvalues": vals.clamp_min(0.0), "vectors": vecs, "eigengap": gap}


def source_guard_fiber_intersection(b_source: torch.Tensor, b_guard: torch.Tensor) -> dict[str, Any]:
    """PSD intersection proxy preserving common source/guard lift mass."""

    bs = sym(b_source.detach().to(dtype=torch.float64).cpu()).clamp_min(0.0)
    bg = sym(b_guard.detach().to(dtype=torch.float64).cpu()).clamp_min(0.0)
    n = min(int(bs.shape[0]), int(bs.shape[1]), int(bg.shape[0]), int(bg.shape[1]))
    if n <= 0:
        z = torch.zeros((0, 0), dtype=torch.float64)
        return {"intersection": z, "common_retention_ratio": 0.0, "source_guard_cosine": 0.0}
    bs = sym(bs[:n, :n])
    bg = sym(bg[:n, :n])
    bs = bs / torch.trace(bs).clamp_min(EPS)
    bg = bg / torch.trace(bg).clamp_min(EPS)
    common = sym(bs @ bg @ bs)
    vals, vecs = torch.linalg.eigh(common)
    common_psd = sym((vecs * vals.clamp_min(0.0).reshape(1, -1)) @ vecs.T)
    denom = bs.norm() * bg.norm()
    cos = float((bs.reshape(-1) @ bg.reshape(-1) / denom.clamp_min(EPS)).clamp(0.0, 1.0).item())
    return {
        "intersection": common_psd,
        "common_retention_ratio": float(torch.trace(common_psd).clamp_min(0.0).item()),
        "source_guard_cosine": cos,
    }


def spectrum_metrics(bmat: torch.Tensor, top_k: int = 5) -> dict[str, float | int]:
    b = sym(bmat.detach().to(dtype=torch.float64).cpu())
    if int(b.numel()) == 0:
        return {"trace": 0.0, "top_eigenvalue": 0.0, "top5_mass": 0.0, "effective_rank": 0.0, "positive_rank": 0}
    vals = torch.linalg.eigvalsh(b).clamp_min(0.0)
    vals = torch.sort(vals, descending=True).values
    trace = float(vals.sum().item())
    if trace <= EPS:
        return {"trace": 0.0, "top_eigenvalue": 0.0, "top5_mass": 0.0, "effective_rank": 0.0, "positive_rank": 0}
    probs = vals / trace
    eff = float(torch.exp(-(probs * probs.clamp_min(EPS).log()).sum()).item())
    return {
        "trace": trace,
        "top_eigenvalue": float(vals[0].item()),
        "top5_mass": float(vals[: int(top_k)].sum().item()) / max(trace, EPS),
        "effective_rank": eff,
        "positive_rank": int((vals > 1.0e-10).sum().item()),
    }


def kan_vs_mlp_controllability_metrics(b_kan: torch.Tensor, b_mlp: torch.Tensor, rank: int = 5) -> dict[str, float | int]:
    km = spectrum_metrics(b_kan, top_k=int(rank))
    mm = spectrum_metrics(b_mlp, top_k=int(rank))
    return {
        "kan_trace": km["trace"],
        "kan_top_eigenvalue": km["top_eigenvalue"],
        "kan_top5_mass": km["top5_mass"],
        "kan_effective_rank": km["effective_rank"],
        "mlp_trace": mm["trace"],
        "mlp_top_eigenvalue": mm["top_eigenvalue"],
        "mlp_top5_mass": mm["top5_mass"],
        "mlp_effective_rank": mm["effective_rank"],
        "adv_spec": float(km["trace"]) / max(float(mm["trace"]), EPS),
    }


def fiber_lift_toy_smoke_test() -> dict[str, float | int]:
    eye = torch.eye(3, dtype=torch.float64)
    b_eye = compute_lift_operator(eye, torch.ones(3, dtype=torch.float64), eye)
    p1 = torch.diag(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    b_p1 = compute_lift_operator(eye, torch.ones(3, dtype=torch.float64), p1)
    same = source_guard_fiber_intersection(p1, p1)
    orth = source_guard_fiber_intersection(p1, torch.diag(torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)))
    return {
        "fiber_lift_toy_identity_error": float((b_eye - eye).abs().max().item()),
        "fiber_lift_projection_rank_error": abs(int((torch.linalg.eigvalsh(b_p1) > 1.0e-8).sum().item()) - 1),
        "source_guard_identical_retention": float(same["common_retention_ratio"]),
        "source_guard_orthogonal_retention": float(orth["common_retention_ratio"]),
    }

