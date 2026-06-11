"""Source-state transport and release rules for adaptive FU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from dgkan.fu.adaptive_controller import cosine


@dataclass
class TransportedSourceState:
    fast: torch.Tensor
    slow: torch.Tensor
    mixed: torch.Tensor
    age: int
    refresh_count: int
    release_count: int
    stale_count: int

    def to_row(self) -> dict[str, Any]:
        return {
            "source_age": self.age,
            "source_refresh_count": self.refresh_count,
            "source_release_count": self.release_count,
            "stale_source_count": self.stale_count,
            "fast_slow_alignment": cosine(self.fast, self.slow),
        }


def init_source_state(source: torch.Tensor) -> TransportedSourceState:
    z = source.detach().float().clone()
    return TransportedSourceState(z, z.clone(), z.clone(), 0, 0, 0, 0)


def stale_source_detect(
    state: TransportedSourceState,
    candidate: torch.Tensor,
    *,
    source_loss_linear_gain: float,
    coherence_threshold: float = 0.05,
    loss_threshold: float = -1.0e-6,
) -> int:
    coherence = cosine(state.mixed, candidate)
    return int(coherence < coherence_threshold or float(source_loss_linear_gain) < float(loss_threshold))


def transport_source_state(
    previous: TransportedSourceState | None,
    candidate: torch.Tensor,
    *,
    washout_risk: float,
    source_loss_linear_gain: float,
    beta_fast: float = 0.80,
    beta_slow: float = 0.985,
    soft_release: bool = True,
) -> TransportedSourceState:
    cand = candidate.detach().float()
    if previous is None or previous.mixed.numel() != cand.numel():
        return init_source_state(cand)

    stale = stale_source_detect(previous, cand, source_loss_linear_gain=source_loss_linear_gain)
    release = int(float(source_loss_linear_gain) < -1.0e-6)
    if release and not soft_release:
        fresh = init_source_state(cand)
        fresh.release_count = previous.release_count + 1
        fresh.stale_count = previous.stale_count + stale
        return fresh

    bf = float(beta_fast)
    bs = float(beta_slow)
    if release and soft_release:
        bf = min(0.50, bf)
        bs = min(0.90, bs)
    fast = bf * previous.fast.to(cand.device) + (1.0 - bf) * cand
    slow = bs * previous.slow.to(cand.device) + (1.0 - bs) * cand
    risk = max(0.0, min(1.0, float(washout_risk)))
    mix_fast = max(0.0, min(1.0, 0.25 + 0.55 * risk - 0.0002 * previous.age - 0.25 * release))
    mixed = mix_fast * fast + (1.0 - mix_fast) * slow
    refresh = int(risk >= 0.70 or stale)
    return TransportedSourceState(
        fast=fast.detach(),
        slow=slow.detach(),
        mixed=mixed.detach(),
        age=0 if refresh else previous.age + 1,
        refresh_count=previous.refresh_count + refresh,
        release_count=previous.release_count + release,
        stale_count=previous.stale_count + stale,
    )


__all__ = [
    "TransportedSourceState",
    "init_source_state",
    "stale_source_detect",
    "transport_source_state",
]
