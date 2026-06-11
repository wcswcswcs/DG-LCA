"""Same-parameter MLP helpers for efficiency audits."""

from __future__ import annotations


def hidden_for_param_budget(input_dim: int, output_dim: int, target_params: int) -> int:
    best_h = 4
    best_delta = float("inf")
    for h in range(4, 257):
        params = input_dim * h + h * h + h * output_dim
        delta = abs(params - int(target_params))
        if delta < best_delta:
            best_h = h
            best_delta = delta
    return best_h
