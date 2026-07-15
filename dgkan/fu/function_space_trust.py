"""Function/output-space trust utilities for v23.15."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class TrustDecision:
    alpha: float
    accepted: int
    reject_reason: str
    source_after: dict[str, float]
    guard_after: dict[str, float]


def choose_alpha_by_function_metrics(
    *,
    model: Any,
    snapshot: dict[str, torch.Tensor],
    apply_update: Callable[[float], None],
    restore: Callable[[dict[str, torch.Tensor]], None],
    source_before: dict[str, float],
    guard_before: dict[str, float],
    source_metrics: Callable[[], dict[str, float]],
    guard_metrics: Callable[[], dict[str, float]],
    alphas: list[float],
    coverage_floor_tolerance: float = 0.0,
    loss_tolerance: float = 0.002,
) -> TrustDecision:
    best: TrustDecision | None = None
    reject_reason = "no_alpha_candidates"
    for alpha in alphas:
        restore(snapshot)
        apply_update(float(alpha))
        s_after = source_metrics()
        g_after = guard_metrics()
        coverage_ok = g_after.get("coverage", 0.0) + float(coverage_floor_tolerance) >= guard_before.get("coverage", 0.0)
        loss_ok = g_after.get("loss", 0.0) <= guard_before.get("loss", 0.0) + float(loss_tolerance)
        if coverage_ok or loss_ok:
            best = TrustDecision(float(alpha), 1, "accepted", s_after, g_after)
            break
        reject_reason = "function_space_trust_reject"
    if best is None:
        restore(snapshot)
        return TrustDecision(0.0, 0, reject_reason, source_before, guard_before)
    restore(snapshot)
    apply_update(float(best.alpha))
    return best


def g_norm(delta: torch.Tensor, metric: torch.Tensor) -> float:
    d = delta.detach().to(dtype=torch.float64)
    g = metric.to(device=d.device, dtype=torch.float64)
    val = torch.trace(d.T @ g @ d) if d.ndim == 2 else d.reshape(1, -1) @ g @ d.reshape(-1, 1)
    return float(torch.sqrt(val.clamp_min(0.0)).detach().cpu().item())


def clip_by_g_norm(delta: torch.Tensor, metric: torch.Tensor, max_norm: float) -> tuple[torch.Tensor, float]:
    norm = g_norm(delta, metric)
    if float(max_norm) <= 0.0 or norm <= float(max_norm):
        return delta, 1.0
    scale = float(max_norm) / max(norm, 1.0e-12)
    return delta * scale, scale

