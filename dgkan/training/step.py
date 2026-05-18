"""Manual training step schema."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StepResult:
    loss: float
    step_time_ms: float
    used_loss_backward: int = 0
