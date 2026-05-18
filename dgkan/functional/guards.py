"""Task guard helpers."""

from __future__ import annotations


def bad_step_flag(holdout_descent_ratio: float, *, minimum_ratio: float = 0.95) -> int:
    return int(float(holdout_descent_ratio) < float(minimum_ratio))
