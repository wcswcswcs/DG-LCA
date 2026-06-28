"""Smoothness metrics for one-dimensional KAN edge-function bases."""

from __future__ import annotations

from typing import Any

import torch

from dgkan.fu.kan_edge_function_metric import basis_derivative, basis_eval, basis_high_frequency_energy, metric_stats


def smoothness_gram(
    basis_name: str,
    k: int,
    grid: torch.Tensor | None = None,
    ridge: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    if grid is None:
        grid = torch.linspace(-1.0, 1.0, 2049, dtype=torch.float64)
    work = grid.detach().reshape(-1).to(dtype=torch.float64)
    deriv = basis_derivative(work, basis_name, k)
    gram = deriv.transpose(0, 1) @ deriv / max(1, int(deriv.shape[0]))
    stats = metric_stats(gram, ridge=ridge)
    diag = [float(x) for x in torch.diag(gram).detach().cpu().tolist()]
    if "fourier" in str(basis_name):
        grouped = [diag[0]] if diag else []
        idx = 1
        while idx < len(diag):
            grouped.append(sum(diag[idx : min(idx + 2, len(diag))]) / max(1, min(2, len(diag) - idx)))
            idx += 2
        diag_for_rate = grouped
    else:
        diag_for_rate = diag
    monotone_pairs = 0
    total_pairs = max(1, len(diag_for_rate) - 1)
    for left, right in zip(diag_for_rate, diag_for_rate[1:]):
        if float(right) + 1.0e-10 >= float(left):
            monotone_pairs += 1
    monotone_rate = float(monotone_pairs / total_pairs)
    return gram, {
        "smoothness_PSD_min": stats.min_eigenvalue,
        "smoothness_condition": stats.condition,
        "smoothness_effective_rank": stats.effective_rank,
        "frequency_cost_monotone_rate": monotone_rate if "fourier" in str(basis_name) else 1.0,
        "degree_cost_monotone_rate": monotone_rate if str(basis_name) == "chebyshev" else 1.0,
        "edge_smoothness_energy": basis_high_frequency_energy(basis_name, gram),
    }


def smoothness_drift(basis_name: str, k: int, old_u: torch.Tensor, new_u: torch.Tensor) -> dict[str, float]:
    old_phi = basis_eval(old_u, basis_name, k)
    new_phi = basis_eval(new_u, basis_name, k)
    old_energy = old_phi.diff(dim=0).square().mean() if int(old_phi.shape[0]) > 1 else old_phi.new_tensor(0.0)
    new_energy = new_phi.diff(dim=0).square().mean() if int(new_phi.shape[0]) > 1 else new_phi.new_tensor(0.0)
    den = old_energy.abs().clamp_min(1.0e-12)
    return {
        "edge_smoothness_drift": float(((new_energy - old_energy).abs() / den).detach().cpu().item()),
        "edge_smoothness_energy_old": float(old_energy.detach().cpu().item()),
        "edge_smoothness_energy_new": float(new_energy.detach().cpu().item()),
    }
