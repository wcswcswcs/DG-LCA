"""Continual-learning metric helpers for v22.18."""

from __future__ import annotations

from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in {float("inf"), float("-inf")}:
        return default
    return out


def average_forgetting(row: dict[str, Any]) -> float:
    vals = []
    for key in ["Forgetting_T0_after_T2", "Forgetting_T1_after_T2"]:
        if row.get(key) not in {"", None}:
            vals.append(safe_float(row.get(key)))
    return sum(vals) / len(vals) if vals else 0.0


def forgetting_reduction(base: dict[str, Any], fu: dict[str, Any]) -> dict[str, Any]:
    base_f = average_forgetting(base)
    fu_f = average_forgetting(fu)
    rel = (base_f - fu_f) / base_f if base_f > 1.0e-12 else 0.0
    return {
        "base_average_forgetting": base_f,
        "fu_average_forgetting": fu_f,
        "absolute_forgetting_reduction": base_f - fu_f,
        "relative_forgetting_reduction": rel,
        "average_accuracy_delta": safe_float(fu.get("average_accuracy")) - safe_float(base.get("average_accuracy")),
        "backward_transfer_delta": safe_float(fu.get("average_backward_transfer")) - safe_float(base.get("average_backward_transfer")),
    }
