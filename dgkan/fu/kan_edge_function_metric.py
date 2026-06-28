"""KAN-native edge-function domain and pullback metric helpers.

The functions in this module operate on edge-function basis samples, not on
dataset labels or validation/test state.  They are intentionally small and
deterministic so v22.71 gates can audit finite metric construction before any
full-loop training is allowed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from dgkan.models.fc_purekan_primitives import _basis_derivative, _basis_eval


EPS = 1.0e-12


@dataclass(frozen=True)
class MetricStats:
    symmetry_error: float
    min_eigenvalue: float
    condition: float
    effective_rank: float
    trace: float


def default_centers_scales(k: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    kk = max(1, int(k))
    centers = torch.linspace(-1.0, 1.0, kk, device=device, dtype=dtype)
    scales = torch.tensor([max(0.2, 2.0 / max(1, kk - 1))], device=device, dtype=dtype)
    return centers, scales


def basis_eval(
    u: torch.Tensor,
    basis_name: str,
    k: int,
    centers: torch.Tensor | None = None,
    scales: torch.Tensor | None = None,
) -> torch.Tensor:
    work = u.detach().reshape(-1).to(dtype=torch.float64)
    if centers is None or scales is None:
        centers, scales = default_centers_scales(k, work.device, work.dtype)
    return _basis_eval(work, str(basis_name), int(k), centers.to(work), scales.to(work)).to(torch.float64)


def basis_derivative(
    u: torch.Tensor,
    basis_name: str,
    k: int,
    centers: torch.Tensor | None = None,
    scales: torch.Tensor | None = None,
) -> torch.Tensor:
    work = u.detach().reshape(-1).to(dtype=torch.float64)
    if centers is None or scales is None:
        centers, scales = default_centers_scales(k, work.device, work.dtype)
    return _basis_derivative(work, str(basis_name), int(k), centers.to(work), scales.to(work)).to(torch.float64)


def symmetrize(mat: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mat + mat.transpose(-1, -2))


def metric_stats(mat: torch.Tensor, ridge: float = 1.0e-8) -> MetricStats:
    if int(mat.numel()) == 0:
        return MetricStats(0.0, 0.0, 0.0, 0.0, 0.0)
    work = symmetrize(mat.detach().to(dtype=torch.float64))
    eye = torch.eye(int(work.shape[0]), dtype=work.dtype, device=work.device)
    ridged = work + float(ridge) * eye
    symmetry_error = float(torch.linalg.norm(work - work.transpose(0, 1)).detach().cpu().item())
    evals = torch.linalg.eigvalsh(ridged)
    min_eval = float(evals.min().detach().cpu().item())
    max_eval = float(evals.max().detach().cpu().item())
    condition = float(max_eval / max(min_eval, EPS)) if max_eval > 0.0 else 0.0
    vals = torch.clamp(evals, min=0.0)
    trace = float(vals.sum().detach().cpu().item())
    if trace <= EPS:
        eff_rank = 0.0
    else:
        probs = vals / vals.sum().clamp_min(EPS)
        eff_rank = float(torch.exp(-(probs * torch.log(probs.clamp_min(EPS))).sum()).detach().cpu().item())
    return MetricStats(symmetry_error, min_eval, condition, eff_rank, trace)


def domain_gram(
    u: torch.Tensor,
    basis_name: str,
    k: int,
    centers: torch.Tensor | None = None,
    scales: torch.Tensor | None = None,
    ridge: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    phi = basis_eval(u, basis_name, k, centers, scales)
    gram = phi.transpose(0, 1) @ phi / max(1, int(phi.shape[0]))
    stats = metric_stats(gram, ridge=ridge)
    return gram, {
        "domain_Gram_symmetry_error": stats.symmetry_error,
        "domain_Gram_min_eigenvalue": stats.min_eigenvalue,
        "domain_Gram_condition": stats.condition,
        "domain_Gram_effective_rank": stats.effective_rank,
        "domain_Gram_trace": stats.trace,
    }


def balanced_domain_gram(
    u: torch.Tensor,
    basis_name: str,
    k: int,
    centers: torch.Tensor | None = None,
    scales: torch.Tensor | None = None,
    ridge: float = 1.0e-5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train-only column-balanced domain Gram for ill-conditioned edge banks."""

    phi = basis_eval(u, basis_name, k, centers, scales)
    centered = phi - phi.mean(dim=0, keepdim=True)
    col_scale = centered.square().mean(dim=0).sqrt().clamp_min(1.0e-8)
    balanced = centered / col_scale
    gram = balanced.transpose(0, 1) @ balanced / max(1, int(balanced.shape[0]))
    stats = metric_stats(gram, ridge=ridge)
    return gram, {
        "domain_Gram_symmetry_error": stats.symmetry_error,
        "domain_Gram_min_eigenvalue": stats.min_eigenvalue,
        "domain_Gram_condition": stats.condition,
        "domain_Gram_effective_rank": stats.effective_rank,
        "domain_Gram_trace": stats.trace,
        "domain_Gram_balance_ridge": float(ridge),
    }


def wasserstein1_1d(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = torch.sort(a.detach().reshape(-1).to(dtype=torch.float64)).values
    bb = torch.sort(b.detach().reshape(-1).to(dtype=torch.float64)).values
    n = min(int(aa.numel()), int(bb.numel()))
    if n == 0:
        return 0.0
    if int(aa.numel()) != n:
        idx = torch.linspace(0, int(aa.numel()) - 1, n, device=aa.device).round().long()
        aa = aa.index_select(0, idx)
    if int(bb.numel()) != n:
        idx = torch.linspace(0, int(bb.numel()) - 1, n, device=bb.device).round().long()
        bb = bb.index_select(0, idx)
    return float((aa - bb).abs().mean().detach().cpu().item())


def support_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().reshape(-1).to(dtype=torch.float64)
    bb = b.detach().reshape(-1).to(dtype=torch.float64)
    if int(aa.numel()) == 0 or int(bb.numel()) == 0:
        return 0.0
    lo = max(float(aa.min().cpu().item()), float(bb.min().cpu().item()))
    hi = min(float(aa.max().cpu().item()), float(bb.max().cpu().item()))
    union_lo = min(float(aa.min().cpu().item()), float(bb.min().cpu().item()))
    union_hi = max(float(aa.max().cpu().item()), float(bb.max().cpu().item()))
    return max(0.0, hi - lo) / max(EPS, union_hi - union_lo)


def domain_drift_stats(old_u: torch.Tensor, new_u: torch.Tensor) -> dict[str, float]:
    old = old_u.detach().reshape(-1).to(dtype=torch.float64)
    new = new_u.detach().reshape(-1).to(dtype=torch.float64)
    if int(old.numel()) == 0 or int(new.numel()) == 0:
        return {
            "edge_domain_mean_shift": 0.0,
            "edge_domain_std_shift": 0.0,
            "edge_domain_wasserstein1": 0.0,
            "edge_domain_quantile_drift_p50": 0.0,
            "edge_domain_quantile_drift_p90": 0.0,
            "edge_domain_support_overlap": 0.0,
            "edge_extrapolation_rate": 0.0,
        }
    qs = torch.tensor([0.10, 0.50, 0.90], dtype=torch.float64, device=old.device)
    old_q = torch.quantile(old, qs)
    new_q = torch.quantile(new, qs)
    old_lo = float(old.min().detach().cpu().item())
    old_hi = float(old.max().detach().cpu().item())
    extra = ((new < old_lo) | (new > old_hi)).to(torch.float64).mean()
    return {
        "edge_domain_mean_shift": abs(float(new.mean().detach().cpu().item() - old.mean().detach().cpu().item())),
        "edge_domain_std_shift": abs(float(new.std(unbiased=False).detach().cpu().item() - old.std(unbiased=False).detach().cpu().item())),
        "edge_domain_wasserstein1": wasserstein1_1d(old, new),
        "edge_domain_quantile_drift_p50": abs(float((new_q[1] - old_q[1]).detach().cpu().item())),
        "edge_domain_quantile_drift_p90": abs(float((new_q[2] - old_q[2]).detach().cpu().item())),
        "edge_domain_support_overlap": support_overlap(old, new),
        "edge_extrapolation_rate": float(extra.detach().cpu().item()),
    }


def gram_relative_drift(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = symmetrize(a.detach().to(dtype=torch.float64))
    bb = symmetrize(b.detach().to(dtype=torch.float64))
    den = torch.linalg.norm(aa).clamp_min(EPS)
    return float((torch.linalg.norm(aa - bb) / den).detach().cpu().item())


def basis_high_frequency_energy(basis_name: str, gram_or_smooth: torch.Tensor) -> float:
    diag = torch.diag(gram_or_smooth.detach().to(dtype=torch.float64)).clamp_min(0.0)
    if int(diag.numel()) == 0:
        return 0.0
    if "fourier" in str(basis_name):
        weights = torch.arange(1, int(diag.numel()) + 1, dtype=diag.dtype, device=diag.device)
    else:
        weights = torch.arange(0, int(diag.numel()), dtype=diag.dtype, device=diag.device).pow(2.0) + 1.0
    weighted = diag * weights
    return float((weighted.sum() / diag.sum().clamp_min(EPS)).detach().cpu().item())


def edge_domain_metric_summary(old_u: torch.Tensor, new_u: torch.Tensor, basis_name: str, k: int) -> dict[str, float]:
    old_g, old_stats = domain_gram(old_u, basis_name, k)
    new_g, new_stats = domain_gram(new_u, basis_name, k)
    return {
        **domain_drift_stats(old_u, new_u),
        "edge_basis_Gram_condition": new_stats["domain_Gram_condition"],
        "edge_basis_effective_rank": new_stats["domain_Gram_effective_rank"],
        "edge_basis_high_frequency_energy": basis_high_frequency_energy(basis_name, new_g),
        "edge_domain_Gram_drift": gram_relative_drift(old_g, new_g),
        "edge_domain_old_Gram_condition": old_stats["domain_Gram_condition"],
    }


def finite_dict(row: dict[str, Any], keys: list[str]) -> bool:
    for key in keys:
        try:
            val = float(row.get(key, float("nan")))
        except Exception:
            return False
        if not math.isfinite(val):
            return False
    return True
