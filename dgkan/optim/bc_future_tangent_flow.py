"""Optimizer-owned BC task lift plus future-tangent shaping composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CombinedFlowDiagnostics:
    task_g_norm: float
    shaping_g_norm: float
    shaping_to_task_g_norm_ratio: float
    finite_step_scale: float
    manual_update_detected: int = 0
    auxiliary_loss_used: int = 0
    optimizer_owned_transform: int = 1


def add_coeff_deltas(a: list[torch.Tensor], b: list[torch.Tensor], *, alpha: float = 1.0) -> list[torch.Tensor]:
    return [aa.to(dtype=torch.float64) + float(alpha) * bb.to(device=aa.device, dtype=torch.float64) for aa, bb in zip(a, b)]


def scale_coeff_delta(delta: list[torch.Tensor], scale: float) -> list[torch.Tensor]:
    return [d.to(dtype=torch.float64) * float(scale) for d in delta]


def apply_optimizer_owned_delta(model: Any, delta: list[torch.Tensor], *, scale: float) -> None:
    """Apply a precomputed optimizer transform to existing coefficients.

    v23.17 audits use this in a runner instead of adding any HVP scalar to the
    training loss. The audit flag remains optimizer-owned and task-loss-only.
    """
    with torch.no_grad():
        for param, d in zip(model.coeffs, delta):
            param.add_(float(scale) * d.to(device=param.device, dtype=param.dtype))


def combine_task_and_shape(
    task_delta: list[torch.Tensor],
    shaping_delta: list[torch.Tensor],
    *,
    finite_step_scale: float = 1.0,
    task_g_norm: float = 0.0,
    shaping_g_norm: float = 0.0,
) -> tuple[list[torch.Tensor], CombinedFlowDiagnostics]:
    combined = scale_coeff_delta(add_coeff_deltas(task_delta, shaping_delta), float(finite_step_scale))
    ratio = float(shaping_g_norm) / max(float(task_g_norm), 1.0e-12)
    return combined, CombinedFlowDiagnostics(
        task_g_norm=float(task_g_norm),
        shaping_g_norm=float(shaping_g_norm),
        shaping_to_task_g_norm_ratio=ratio,
        finite_step_scale=float(finite_step_scale),
    )

