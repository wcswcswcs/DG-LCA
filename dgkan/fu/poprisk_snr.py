"""PopRisk/SNR signal-channel helpers."""

from __future__ import annotations

import torch


def snr_preconditioner(per_example_grads: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    if per_example_grads.numel() == 0:
        return per_example_grads
    mu = per_example_grads.mean(dim=0)
    var = per_example_grads.var(dim=0, unbiased=False)
    n = max(1, int(per_example_grads.shape[0]) - 1)
    snr = mu.square() / (var / n + float(eps))
    return mu * torch.sqrt(snr.clamp(max=100.0))
