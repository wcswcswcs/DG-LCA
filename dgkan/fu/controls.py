"""Matched controls for v17 functional-update experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch

from dgkan.fu.core import UpdateTensor, normalized_like


@dataclass
class ControlSpec:
    name: str
    matched_property: str
    direction_source: str
    uses_audit_metrics: int = 0
    promotion_allowed: int = 0

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


CONTROL_SPECS = [
    ControlSpec("NoOpMatchedOverhead", "runtime_overhead", "train_stream_noop"),
    ControlSpec("RandomMatchedNorm", "update_norm", "train_stream_random"),
    ControlSpec("AdamWExtraStepsMatchedTime", "wall_time", "train_stream_adamw_control"),
    ControlSpec("SGDExtraStepsMatchedTime", "wall_time", "train_stream_sgd_control"),
    ControlSpec("RecoveryOnly", "recovery_schedule", "train_stream_recovery_control"),
    ControlSpec("SameActiveFractionRandom", "active_fraction", "train_stream_random_mask"),
    ControlSpec("SameProjectionRetentionRandom", "projection_retention", "train_stream_random_projection"),
]


def random_matched_update(reference: torch.Tensor, seed: int, mechanism: str = "RandomMatchedNorm") -> UpdateTensor:
    gen = torch.Generator(device=reference.device).manual_seed(int(seed))
    noise = torch.randn(reference.shape, device=reference.device, generator=gen)
    return UpdateTensor(normalized_like(noise, reference), "step", "subtract", "parameter", "train_stream_matched_random_control", mechanism, one_step_descent_claim=0)


def noop_update(reference: torch.Tensor, mechanism: str = "NoOpMatchedOverhead") -> UpdateTensor:
    return UpdateTensor(torch.zeros_like(reference), "step", "subtract", "parameter", "train_stream_noop_control", mechanism)

