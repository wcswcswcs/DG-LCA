"""Slow-state FU helpers for v17.0.1."""

from __future__ import annotations

import torch


def update_slow_state(previous: torch.Tensor | None, signal: torch.Tensor, beta: float = 0.90) -> torch.Tensor:
    if previous is None or previous.numel() != signal.numel():
        return signal.detach().clone()
    return float(beta) * previous.detach() + (1.0 - float(beta)) * signal.detach()
