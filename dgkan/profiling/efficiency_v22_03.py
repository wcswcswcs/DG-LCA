"""Compatibility helpers for v22.03/v22.04 efficiency audit imports."""

from __future__ import annotations

from dgkan.profiling.efficiency_v22_04 import classify_efficiency_v22_04


def classify_efficiency_v22_03(row):
    return classify_efficiency_v22_04(row)


__all__ = ["classify_efficiency_v22_03"]
