"""Stable LineC diagnostics entrypoint for v17.0.1.

This module is a compatibility shim over ``dgkan.metrics.linec`` so historical
and v17.0.1 runners can import LineC from one diagnostics namespace.
"""

from __future__ import annotations

from typing import Any

import torch

from dgkan.metrics.linec import (
    LineCMeasurementInvalid,
    LineCResult,
    linec_from_improvements,
    linec_model_update,
    run_linec_golden_tests,
)


def linec_metrics(
    model: torch.nn.Module,
    train_batch: tuple[torch.Tensor, torch.Tensor],
    probe_batch: tuple[torch.Tensor, torch.Tensor],
    *,
    seed: int = 1700,
    sketch_dim: int = 64,
    update: Any | None = None,
) -> LineCResult:
    """Measure LineC with explicit MeasurementInvalid behavior.

    ``seed`` and ``sketch_dim`` are accepted for API stability. If ``update`` is
    supplied it must be a pair of callables ``(apply_update, restore_state)``.
    Without an update, the measurement is a NoOp readback.
    """

    del seed, sketch_dim
    if update is None:
        def apply_update() -> None:
            return None

        def restore_state() -> None:
            return None
    else:
        apply_update, restore_state = update
    return linec_model_update(model, train_batch, probe_batch, apply_update, restore_state)
