"""Source-state helpers for v21 source-retention experiments."""

from __future__ import annotations

import torch


def ema_source_state(previous: torch.Tensor | None, update: torch.Tensor, beta: float = 0.98) -> torch.Tensor:
    """Return an EMA source state without reading audit/validation signals."""
    if previous is None or previous.numel() != update.numel():
        return update.detach().clone()
    return float(beta) * previous.detach().to(device=update.device) + (1.0 - float(beta)) * update.detach()


def source_washout_rate(current: float, previous: float, eps: float = 1.0e-12) -> float | str:
    if previous <= 0.0:
        return ""
    return max(0.0, previous - current) / max(float(eps), previous)

