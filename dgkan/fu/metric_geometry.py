"""Role-blind function-space geometry helpers for v22.13 operators."""

from __future__ import annotations

from typing import Any

import torch


EPS = 1.0e-8


def l2_energy(x: torch.Tensor) -> float:
    y = x.detach().float()
    return float(y.square().mean().item())


def sobolev_energy(x: torch.Tensor) -> float:
    y = x.detach().float()
    pieces = [y.square().mean()]
    if y.shape[0] > 1:
        pieces.append((y[1:] - y[:-1]).square().mean())
    if y.ndim > 1 and y.shape[-1] > 1:
        pieces.append((y[..., 1:] - y[..., :-1]).square().mean())
    return float(sum(pieces).item())


def fisher_energy(x: torch.Tensor, logits: torch.Tensor | None = None) -> float:
    y = x.detach().float()
    if logits is None:
        return float(y.square().mean().item())
    centered = y - y.mean(dim=-1, keepdim=True)
    scale = logits.detach().float().var(dim=-1, unbiased=False, keepdim=True).sqrt().clamp_min(EPS)
    return float((centered / scale).square().mean().item())


def normalized_directional_sharpness(x: torch.Tensor) -> float:
    y = x.detach().float()
    rough = 0.0
    if y.shape[0] > 2:
        rough += float((y[2:] - 2.0 * y[1:-1] + y[:-2]).square().mean().item())
    if y.ndim > 1 and y.shape[-1] > 2:
        rough += float((y[..., 2:] - 2.0 * y[..., 1:-1] + y[..., :-2]).square().mean().item())
    return rough / max(float(y.square().mean().item()), EPS)


def smooth_low_curvature(x: torch.Tensor, row_mid: float = 1.0, col_mid: float = 0.40) -> torch.Tensor:
    y = x.detach().float().clone()
    if y.shape[0] > 2 and float(row_mid) < 0.999:
        side = (1.0 - float(row_mid)) * 0.5
        y[1:-1] = side * x[:-2].float() + float(row_mid) * x[1:-1].float() + side * x[2:].float()
    if y.ndim > 1 and y.shape[-1] > 2:
        before = y.clone()
        side = (1.0 - float(col_mid)) * 0.5
        y[..., 1:-1] = side * before[..., :-2] + float(col_mid) * before[..., 1:-1] + side * before[..., 2:]
    return y


def metric_row(x: torch.Tensor, logits: torch.Tensor | None = None) -> dict[str, Any]:
    before = x.detach().float()
    after = smooth_low_curvature(before)
    nds_before = normalized_directional_sharpness(before)
    nds_after = normalized_directional_sharpness(after)
    return {
        "NDS_before": nds_before,
        "NDS_after": nds_after,
        "NDS_reduction": (nds_before - nds_after) / max(abs(nds_before), EPS),
        "metric_energy_L2": l2_energy(after),
        "metric_energy_Fisher": fisher_energy(after, logits),
        "metric_energy_Sobolev": sobolev_energy(after),
        "metric_energy_RKHS": sobolev_energy(after) + 0.25 * l2_energy(after),
    }


__all__ = [
    "EPS",
    "fisher_energy",
    "l2_energy",
    "metric_row",
    "normalized_directional_sharpness",
    "smooth_low_curvature",
    "sobolev_energy",
]
