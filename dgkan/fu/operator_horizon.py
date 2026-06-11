"""Small generic horizon utilities for v22.13 runners."""

from __future__ import annotations

from typing import Any

import torch


HORIZONS = [100, 400, 800, 1600, 2400, 3200, 4000, 4800, 6400]


def safe_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().float().reshape(-1)
    y = b.detach().float().reshape(-1)
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom.item()) <= 1.0e-8:
        return 0.0
    return float((x @ y / denom).clamp(-1.0, 1.0).item())


def snapshot(model: torch.nn.Module, x: torch.Tensor, base_logits: torch.Tensor, target_delta: torch.Tensor, interface: Any, task_data: Any) -> dict[str, float]:
    with torch.no_grad():
        logits = model(x).detach().float()
        displacement = logits - base_logits.detach().float()
        loss_value = float(interface.value(logits, task_data).detach().item())
    target_norm = torch.linalg.vector_norm(target_delta).clamp_min(1.0e-8)
    residual_ratio = float(torch.linalg.vector_norm(displacement - target_delta).item() / target_norm.item())
    alignment = safe_cos(displacement, target_delta)
    return {
        "target_alignment": alignment,
        "target_residual_ratio": residual_ratio,
        "target_retention_score": alignment - residual_ratio,
        "loss_value": loss_value,
        "geometry_energy": float(displacement.square().mean().item()),
    }


__all__ = ["HORIZONS", "safe_cos", "snapshot"]

