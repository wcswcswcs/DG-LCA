"""Block-coordinate source-state diagnostics for v22.09."""

from __future__ import annotations

from typing import Any

import torch

from dgkan.fu.core import cosine


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except Exception:
        return default


def block_source_state_metrics(
    block_update: torch.Tensor,
    previous_state: torch.Tensor,
    optimizer_momentum: torch.Tensor | None = None,
) -> dict[str, Any]:
    update = block_update.detach().float().reshape(-1)
    state = previous_state.detach().float().reshape(-1)
    n = min(int(update.numel()), int(state.numel()))
    if n == 0:
        return {
            "block_source_energy": 0.0,
            "block_source_SNR": 0.0,
            "source_state_alignment_with_optimizer_momentum": "",
            "source_overwrite_fraction": 0.0,
        }
    update = update[:n]
    state = state[:n]
    residual = update - state
    energy = float(torch.linalg.vector_norm(update).item())
    noise = float(torch.linalg.vector_norm(residual).item())
    out = {
        "block_source_energy": energy,
        "block_source_SNR": energy / (noise + 1.0e-12),
        "source_overwrite_fraction": noise / (float(torch.linalg.vector_norm(state).item()) + energy + 1.0e-12),
    }
    if optimizer_momentum is not None:
        out["source_state_alignment_with_optimizer_momentum"] = cosine(update, optimizer_momentum.detach().float().reshape(-1))
    else:
        out["source_state_alignment_with_optimizer_momentum"] = ""
    return out


def block_source_state_score(row: dict[str, Any]) -> float:
    energy = finite_float(row.get("block_source_energy"), 0.0)
    snr = finite_float(row.get("block_source_SNR"), 0.0)
    age = finite_float(row.get("block_source_age"), 0.0)
    decay = finite_float(row.get("block_source_decay_rate"), 1.0)
    overwrite = finite_float(row.get("source_overwrite_fraction"), 1.0)
    angular = finite_float(row.get("row_angular_velocity"), 1.0)
    momentum = finite_float(row.get("source_state_alignment_with_optimizer_momentum"), 0.0)
    return 0.30 * energy + 0.35 * snr + 0.15 * age + 0.20 * momentum - 0.45 * overwrite - 0.25 * decay - 0.20 * angular


def source_state_dynamics_unit_tests() -> list[dict[str, Any]]:
    metrics = block_source_state_metrics(torch.tensor([1.0, 0.0]), torch.tensor([0.8, 0.2]), torch.tensor([1.0, 0.0]))
    score = block_source_state_score({**metrics, "block_source_age": 2, "block_source_decay_rate": 0.1, "row_angular_velocity": 0.1})
    return [
        {"test": "state_metrics_keys", "pass": int("block_source_SNR" in metrics and "source_overwrite_fraction" in metrics)},
        {"test": "momentum_alignment_positive", "pass": int(finite_float(metrics["source_state_alignment_with_optimizer_momentum"], -1.0) > 0.0)},
        {"test": "score_finite", "pass": int(score == score)},
    ]
