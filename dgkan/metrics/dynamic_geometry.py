"""Dynamic-geometry debt accounting helpers for v17.1 audits."""

from __future__ import annotations

from typing import Any


def debt_recovery_rate(debt_peak: float, debt_final: float, eps: float = 1.0e-12) -> float:
    peak = float(debt_peak)
    final = float(debt_final)
    if peak <= 0:
        return 0.0
    return 1.0 - final / (peak + float(eps))


def debt_row(metric: str, peak: float, final: float) -> dict[str, Any]:
    return {
        "metric": metric,
        "debt_peak": float(peak),
        "debt_final": float(final),
        "debt_recovery_rate": debt_recovery_rate(peak, final),
    }
