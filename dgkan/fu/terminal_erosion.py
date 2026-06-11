"""v22.04 terminal-erosion taxonomy helpers.

The helpers in this module are deliberately audit-only: they classify
already-measured source trajectories and train-stream telemetry. They do
not promote a candidate or synthesize any missing experimental result.
"""

from __future__ import annotations

from dataclasses import dataclass
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


def _ratio(num: Any, den: Any) -> float:
    n = _f(num)
    d = _f(den)
    if not isfinite(n) or not isfinite(d) or abs(d) < 1.0e-12:
        return float("nan")
    return n / d


@dataclass(frozen=True)
class TerminalErosionAudit:
    terminal_erosion_class: str
    explained: int
    evidence: str

    def to_row(self) -> dict[str, Any]:
        return {
            "terminal_erosion_class": self.terminal_erosion_class,
            "terminal_erosion_explained": self.explained,
            "terminal_erosion_evidence": self.evidence,
        }


def classify_terminal_erosion_v22_04(row: dict[str, Any]) -> TerminalErosionAudit:
    """Classify one aggregated candidate row using v22.04 audit fields."""

    h1600 = _f(row.get("h1600"))
    h2400 = _f(row.get("h2400"))
    h3200 = _f(row.get("h3200"))
    h4000 = _f(row.get("h4000"))
    h4800 = _f(row.get("h4800"))
    r4800 = _f(row.get("h4800_retention_ratio"))
    row_pos = int(_f(row.get("row_h4800_positive_count"), 0.0))
    fold = _f(row.get("source_projection_fold_proxy_mean"))
    norm_ratio = _f(row.get("source_norm_over_whitened_norm_mean"))
    hidden_frac = _f(row.get("source_hidden_fraction_mean"))
    readout_frac = _f(row.get("source_readout_fraction_mean"))

    if row_pos >= 7 and isfinite(r4800) and r4800 < 0.5 and h4800 >= 0.005:
        if h4000 > h4800 and h3200 > h4000:
            return TerminalErosionAudit(
                "OptimizerErosion",
                1,
                "monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows",
            )
        if h3200 > h4000 > h4800:
            return TerminalErosionAudit(
                "CurvatureErosion",
                1,
                "post-h3200 curvature-like decay persists through h4000 and h4800",
            )
        return TerminalErosionAudit(
            "DebtErosion",
            1,
            "positive h4800 but retention debt remains below 0.50 gate",
        )

    if isfinite(fold) and fold > 0.25:
        return TerminalErosionAudit("InfoVolumeCollapse", 1, f"fold_proxy={fold:g}")
    if isfinite(norm_ratio) and norm_ratio < 0.75:
        return TerminalErosionAudit("InfoVolumeCollapse", 1, f"whitened_norm_ratio={norm_ratio:g}")
    if isfinite(hidden_frac) and isfinite(readout_frac) and min(hidden_frac, readout_frac) < 0.12:
        return TerminalErosionAudit(
            "ControlEquivalentTerminalDrift",
            1,
            f"hidden/readout source split imbalance hidden={hidden_frac:g} readout={readout_frac:g}",
        )
    if h1600 > h2400 > h3200 and row_pos >= 7:
        return TerminalErosionAudit(
            "DatasetLocalizedErosion",
            1,
            "mid-horizon source decay before h3200 with row-positive terminal source",
        )
    return TerminalErosionAudit("UnknownTerminalErosion", 0, "no v22.04 taxonomy condition matched")


def terminal_erosion_unit_tests() -> list[dict[str, Any]]:
    cases = [
        (
            "optimizer",
            {"h3200": 0.34, "h4000": 0.22, "h4800": 0.14, "h4800_retention_ratio": 0.41, "row_h4800_positive_count": 9},
            "OptimizerErosion",
            1,
        ),
        (
            "info_volume",
            {"h3200": 0.20, "h4000": 0.16, "h4800": 0.10, "source_projection_fold_proxy_mean": 0.4},
            "InfoVolumeCollapse",
            1,
        ),
        (
            "unknown",
            {"h3200": 0.01, "h4000": 0.01, "h4800": 0.0, "row_h4800_positive_count": 0},
            "UnknownTerminalErosion",
            0,
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, row, expected_class, expected_explained in cases:
        got = classify_terminal_erosion_v22_04(row)
        rows.append(
            {
                "case": name,
                "expected_class": expected_class,
                "actual_class": got.terminal_erosion_class,
                "expected_explained": expected_explained,
                "actual_explained": got.explained,
                "pass": int(got.terminal_erosion_class == expected_class and got.explained == expected_explained),
            }
        )
    return rows


__all__ = ["TerminalErosionAudit", "classify_terminal_erosion_v22_04", "terminal_erosion_unit_tests"]
