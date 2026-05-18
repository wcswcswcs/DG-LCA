"""Functional update API for v9."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class FunctionalUpdate:
    delta: torch.Tensor
    event_score: float
    event_threshold: float
    event_triggered: int
    role_budget: float


@dataclass
class FunctionalUpdateRecord:
    event_score: float
    event_threshold: float
    event_triggered: int
    role_budget: float
    holdout_descent_ratio: float
    bad_step_flag: int
    functional_update_norm: float
    geometry_delta: float
    functional_update_time: float

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


class FunctionalController:
    def propose(self, params: torch.Tensor, grads: torch.Tensor, geometry: torch.Tensor, holdout_state: Any) -> FunctionalUpdate:
        raise NotImplementedError

    def apply(self, params: torch.Tensor, update: FunctionalUpdate) -> None:
        if update.event_triggered:
            params.add_(update.delta)

    def audit(self) -> Dict[str, Any]:
        return {}
