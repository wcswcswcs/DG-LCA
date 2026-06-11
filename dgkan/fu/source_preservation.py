"""v22.04 source-preservation audit helpers."""

from __future__ import annotations

from math import isfinite
from typing import Any


def _f(value: Any, default: float = float("nan")) -> float:
    try:
        if value in {"", None}:
            return default
        out = float(value)
    except Exception:
        return default
    return out if isfinite(out) else default


def source_preservation_delta(row: dict[str, Any]) -> dict[str, Any]:
    h800 = _f(row.get("h800"))
    h3200 = _f(row.get("h3200"))
    h4800 = _f(row.get("h4800"))
    return {
        "source_preservation_h3200_over_h800": h3200 / h800 if isfinite(h800) and abs(h800) > 1.0e-12 else "",
        "source_preservation_h4800_over_h3200": h4800 / h3200 if isfinite(h3200) and abs(h3200) > 1.0e-12 else "",
        "source_preservation_positive_terminal": int(isfinite(h4800) and h4800 >= 0.005),
    }


def source_preservation_unit_tests() -> list[dict[str, Any]]:
    got = source_preservation_delta({"h800": 1.0, "h3200": 0.8, "h4800": 0.41})
    return [
        {
            "case": "ratio",
            "pass": int(abs(float(got["source_preservation_h3200_over_h800"]) - 0.8) <= 1.0e-12 and got["source_preservation_positive_terminal"] == 1),
        }
    ]


__all__ = ["source_preservation_delta", "source_preservation_unit_tests"]
