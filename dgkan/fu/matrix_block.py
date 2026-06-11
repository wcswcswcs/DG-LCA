"""Matrix-block FU readback helpers."""

from __future__ import annotations

import torch


def block_energy(vector: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    flat = vector.detach().reshape(-1).float()
    if flat.numel() == 0:
        return torch.zeros(0)
    pad = (-flat.numel()) % int(block_size)
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    return flat.view(-1, int(block_size)).square().sum(dim=1)
