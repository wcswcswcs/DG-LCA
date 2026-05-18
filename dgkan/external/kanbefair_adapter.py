"""KANbeFair protocol adapter boundary.

This module deliberately does not run legacy experiments yet.  It defines the
baseline names and schema used by the v9 audit runner so future validation does
not scatter protocol metadata across experiment scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


BASELINES = (
    "KB-MLP",
    "KB-KAN",
    "DG-Transitional",
    "DG-FullEdge",
    "DG-FullEdge+Functional",
    "NoOp",
    "RandomFunc",
)


@dataclass(frozen=True)
class ExternalBaselineSpec:
    baseline_id: str
    params_counter: str = "not_run"
    flops_counter: str = "not_run"
    wall_clock_timer: str = "not_run"
    memory_meter: str = "not_run"

    def to_row(self) -> dict:
        return {
            "baseline_id": self.baseline_id,
            "params_counter": self.params_counter,
            "flops_counter": self.flops_counter,
            "wall_clock_timer": self.wall_clock_timer,
            "memory_meter": self.memory_meter,
        }


def baseline_registry_rows() -> List[dict]:
    return [ExternalBaselineSpec(name).to_row() for name in BASELINES]
