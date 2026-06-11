"""Optimizer-state integration diagnostics for retained source updates."""

from __future__ import annotations

from typing import Any

import torch

from dgkan.fu.core import cosine


def optimizer_state_injection_metrics(
    source_update: torch.Tensor,
    optimizer_momentum: torch.Tensor,
    *,
    slow_state: torch.Tensor | None = None,
) -> dict[str, Any]:
    source = source_update.detach().float().reshape(-1)
    momentum = optimizer_momentum.detach().float().reshape(-1)
    n = min(int(source.numel()), int(momentum.numel()))
    if n == 0:
        return {
            "optimizer_projection_on_source": 0.0,
            "source_state_overwrite": 0.0,
            "fast_slow_source_alignment": "",
            "state_overwrite_fraction": 0.0,
        }
    source = source[:n]
    momentum = momentum[:n]
    projection = cosine(momentum, source)
    residual = momentum - source
    overwrite = float(torch.linalg.vector_norm(residual).item()) / (
        float(torch.linalg.vector_norm(momentum).item()) + float(torch.linalg.vector_norm(source).item()) + 1.0e-12
    )
    out: dict[str, Any] = {
        "optimizer_projection_on_source": projection,
        "source_state_overwrite": overwrite,
        "state_overwrite_fraction": overwrite,
    }
    if slow_state is not None:
        out["fast_slow_source_alignment"] = cosine(source, slow_state.detach().float().reshape(-1))
    else:
        out["fast_slow_source_alignment"] = ""
    return out


def optimizer_state_integration_unit_tests() -> list[dict[str, Any]]:
    metrics = optimizer_state_injection_metrics(torch.tensor([1.0, 0.0]), torch.tensor([0.9, 0.1]), slow_state=torch.tensor([1.0, 0.0]))
    return [
        {"test": "projection_positive", "pass": int(float(metrics["optimizer_projection_on_source"]) > 0.0)},
        {"test": "overwrite_bounded", "pass": int(0.0 <= float(metrics["state_overwrite_fraction"]) <= 1.0)},
        {"test": "slow_alignment_positive", "pass": int(float(metrics["fast_slow_source_alignment"]) > 0.0)},
    ]
