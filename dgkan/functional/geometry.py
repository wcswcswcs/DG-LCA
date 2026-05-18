"""Geometry metric helpers."""

from __future__ import annotations

import torch


def coefficient_curvature(coefficients: torch.Tensor) -> torch.Tensor:
    if coefficients.shape[-1] < 3:
        return torch.zeros((), device=coefficients.device, dtype=coefficients.dtype)
    diff2 = coefficients[..., 2:] - 2.0 * coefficients[..., 1:-1] + coefficients[..., :-2]
    return diff2.square().mean()
