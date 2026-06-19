"""Train-stream source usefulness checks for MLP functional updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SourceUsefulness:
    source_loss_gain: float
    source_func: float
    source_age: int
    nds: float
    control_projection_fraction: float
    stale_source: int
    release_source: int
    boundary_distance: float
    risk_score: float

    def to_row(self) -> dict[str, Any]:
        return {
            "source_loss_gain": self.source_loss_gain,
            "source_func": self.source_func,
            "source_age": self.source_age,
            "NDS": self.nds,
            "control_projection_fraction": self.control_projection_fraction,
            "stale_source": self.stale_source,
            "release_source": self.release_source,
            "source_loss_boundary_distance": self.boundary_distance,
            "risk_score": self.risk_score,
        }


def source_boundary_distance(source_loss_gain: float, eps: float = 1.0e-8) -> float:
    return float(source_loss_gain) - float(eps)


def detect_stale_source(source_age: int, source_func: float, *, max_age: int = 800, min_func: float = 0.05) -> int:
    return int(int(source_age) >= int(max_age) or float(source_func) < float(min_func))


def source_risk_score(
    *,
    source_loss_gain: float,
    source_func: float,
    destructive_projection: float,
    source_age: int,
    nds: float,
) -> float:
    loss_bad = max(0.0, -float(source_loss_gain)) * 1000.0
    weak_func = max(0.0, 0.20 - float(source_func)) / 0.20
    age = min(1.0, max(0.0, float(source_age) / 1200.0))
    dense = min(1.0, max(0.0, float(nds)) * 200.0)
    risk = 0.45 * min(1.0, loss_bad) + 0.25 * min(1.0, weak_func) + 0.20 * max(0.0, float(destructive_projection)) + 0.07 * age + 0.03 * dense
    return max(0.0, min(1.0, float(risk)))


def decide_source_usefulness(
    *,
    source_loss_gain: float,
    source_func: float,
    destructive_projection: float,
    source_age: int,
    nds: float,
    control_projection_fraction: float = 0.0,
) -> SourceUsefulness:
    boundary = source_boundary_distance(source_loss_gain)
    stale = detect_stale_source(source_age, source_func)
    risk = source_risk_score(
        source_loss_gain=source_loss_gain,
        source_func=source_func,
        destructive_projection=destructive_projection,
        source_age=source_age,
        nds=nds,
    )
    release = int(boundary < 0.0 or stale or float(control_projection_fraction) > 0.35)
    return SourceUsefulness(
        source_loss_gain=float(source_loss_gain),
        source_func=float(source_func),
        source_age=int(source_age),
        nds=float(nds),
        control_projection_fraction=float(control_projection_fraction),
        stale_source=stale,
        release_source=release,
        boundary_distance=float(boundary),
        risk_score=float(risk),
    )


__all__ = [
    "SourceUsefulness",
    "decide_source_usefulness",
    "detect_stale_source",
    "source_boundary_distance",
    "source_risk_score",
]
