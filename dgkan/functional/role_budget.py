"""Role budget helpers."""

from __future__ import annotations


def clamp_role_budget(value: float, *, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(float(lo), min(float(hi), float(value)))
