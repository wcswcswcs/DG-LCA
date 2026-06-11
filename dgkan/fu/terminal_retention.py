"""v22.03 terminal-retention gates and terminal-erosion taxonomy.

These helpers classify measured artifacts only. They do not choose training
directions and they do not inspect validation/test/future values during a run.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from dgkan.fu.source_chain import SOURCE_EPS, finite_float, retention_ratio


V22_03_RETENTION_RATIO_MIN = 0.50
V22_03_H4800_POSITIVE_ROWS_MIN = 7


@dataclass(frozen=True)
class V2203TerminalDecision:
    early_source_chain: int
    continuous_h3200_chain: int
    productive_h4800_chain: int
    terminal_erosion: int
    terminal_collapse: int
    h1600_retention_ratio: float | str
    h3200_retention_ratio: float | str
    h4800_retention_ratio: float | str
    h6400_retention_ratio: float | str
    blocker: str


def classify_v22_03_source_chain(
    h100: Any,
    h400: Any,
    h800: Any,
    h1600: Any,
    h2400: Any,
    h3200: Any,
    h4000: Any,
    h4800: Any,
    h6400: Any = "",
    *,
    control_equivalent: Any = 0,
    row_h4800_positive_count: Any = "",
    row_positive_min: int = V22_03_H4800_POSITIVE_ROWS_MIN,
) -> V2203TerminalDecision:
    s100 = finite_float(h100)
    s400 = finite_float(h400)
    s800 = finite_float(h800)
    s1600 = finite_float(h1600)
    s2400 = finite_float(h2400)
    s3200 = finite_float(h3200)
    s4000 = finite_float(h4000)
    s4800 = finite_float(h4800)
    s6400 = finite_float(h6400)
    control_eq = int(finite_float(control_equivalent, 0.0) != 0.0)
    row_pos = int(finite_float(row_h4800_positive_count, -1.0)) if str(row_h4800_positive_count) != "" else row_positive_min
    early = int(not control_eq and s100 >= SOURCE_EPS and s400 >= SOURCE_EPS and s800 >= SOURCE_EPS)
    r1600 = retention_ratio(s1600, s800)
    r3200 = retention_ratio(s3200, s1600)
    r4800 = retention_ratio(s4800, s3200)
    r6400 = retention_ratio(s6400, s4800)
    continuous = int(
        early
        and s1600 >= SOURCE_EPS
        and s2400 >= SOURCE_EPS
        and s3200 >= SOURCE_EPS
        and r1600 != ""
        and r3200 != ""
        and float(r1600) >= V22_03_RETENTION_RATIO_MIN
        and float(r3200) >= V22_03_RETENTION_RATIO_MIN
    )
    productive_h4800 = int(
        continuous
        and s4800 >= SOURCE_EPS
        and r4800 != ""
        and float(r4800) >= V22_03_RETENTION_RATIO_MIN
        and row_pos >= row_positive_min
    )
    terminal_collapse = int(continuous and (not math.isfinite(s4800) or s4800 < SOURCE_EPS))
    terminal_erosion = int(
        continuous
        and not terminal_collapse
        and s4800 >= SOURCE_EPS
        and r4800 != ""
        and float(r4800) < V22_03_RETENTION_RATIO_MIN
    )
    blockers: list[str] = []
    if control_eq:
        blockers.append("ControlEquivalent")
    if not early:
        blockers.append("EarlySourceChainMissing")
    if not continuous:
        blockers.append("ContinuousH3200Missing")
    if continuous and row_pos < row_positive_min:
        blockers.append("RowH4800PositiveInsufficient")
    if terminal_collapse:
        blockers.append("TerminalCollapse")
    if terminal_erosion:
        blockers.append("TerminalErosion")
    if continuous and not terminal_collapse and s4000 == s4000 and s4800 == s4800 and s4800 < s4000:
        blockers.append("PostH4000Erosion")
    return V2203TerminalDecision(
        early_source_chain=early,
        continuous_h3200_chain=continuous,
        productive_h4800_chain=productive_h4800,
        terminal_erosion=terminal_erosion,
        terminal_collapse=terminal_collapse,
        h1600_retention_ratio=r1600,
        h3200_retention_ratio=r3200,
        h4800_retention_ratio=r4800,
        h6400_retention_ratio=r6400,
        blocker="pass" if productive_h4800 else ";".join(dict.fromkeys(blockers)),
    )


def classify_terminal_erosion(row: dict[str, Any]) -> str:
    """Classify measured terminal erosion with conservative artifact-only labels."""

    h800 = finite_float(row.get("h800", row.get("source_h800_mean")))
    h1600 = finite_float(row.get("h1600", row.get("source_h1600_mean")))
    h2400 = finite_float(row.get("h2400", row.get("source_h2400_mean")))
    h3200 = finite_float(row.get("h3200", row.get("source_h3200_mean")))
    h4000 = finite_float(row.get("h4000", row.get("source_h4000_mean")))
    h4800 = finite_float(row.get("h4800", row.get("source_h4800_mean")))
    r4800 = finite_float(row.get("h4800_retention_ratio", row.get("retention_h4800_over_h3200")))
    row_pos = finite_float(row.get("row_h4800_positive_count"), -1.0)
    density = finite_float(row.get("source_state_consensus_density_h4000"), float("nan"))
    source_cos = finite_float(row.get("source_state_current_cos_h4000"), float("nan"))
    removed = finite_float(row.get("source_state_antiwashout_removed_norm_h4000"), 0.0)
    if all(v == v for v in [h800, h1600, h2400, h3200, h4000, h4800]) and h800 < h1600 and h1600 >= h2400 >= h3200 >= h4000 >= h4800:
        return "MeasuredMonotoneTerminalErosion"
    if h3200 == h3200 and h4000 == h4000 and h4800 == h4800 and h4000 < h3200 and h4800 < h4000:
        return "PostH3200PostH4000Erosion"
    if h4800 == h4800 and h4800 >= SOURCE_EPS and r4800 == r4800 and r4800 < V22_03_RETENTION_RATIO_MIN and row_pos >= 7:
        return "PositiveButNonproductiveRetention"
    if density == density and density < 0.05:
        return "TrainSourceSupportSparse"
    if source_cos == source_cos and source_cos < -0.02:
        return "OptimizerSourceConflictProxy"
    if removed > 0.0:
        return "AntiWashoutProjectionActive"
    return "MeasuredTerminalErosionUnresolved"


def terminal_retention_unit_tests() -> list[dict[str, Any]]:
    cases = [
        ("productive", (0.01, 0.02, 0.04, 0.035, 0.03, 0.025, 0.018, 0.014, 0.008), 1, 1, 1, 0),
        ("erosion", (0.01, 0.02, 0.04, 0.035, 0.03, 0.025, 0.014, 0.010, 0.006), 1, 1, 0, 1),
        ("collapse", (0.01, 0.02, 0.04, 0.035, 0.03, 0.025, 0.010, -0.001, -0.003), 1, 1, 0, 0),
        ("control", (0.02, 0.02, 0.04, 0.035, 0.03, 0.025, 0.018, 0.014, 0.008), 0, 0, 0, 0),
    ]
    rows: list[dict[str, Any]] = []
    for name, values, early, h3200, h4800, erosion in cases:
        got = classify_v22_03_source_chain(*values, control_equivalent=int(name == "control"), row_h4800_positive_count=9)
        rows.append(
            {
                "test": name,
                "expected_early": early,
                "actual_early": got.early_source_chain,
                "expected_h3200": h3200,
                "actual_h3200": got.continuous_h3200_chain,
                "expected_h4800": h4800,
                "actual_h4800": got.productive_h4800_chain,
                "expected_erosion": erosion,
                "actual_erosion": got.terminal_erosion,
                "pass": int(
                    got.early_source_chain == early
                    and got.continuous_h3200_chain == h3200
                    and got.productive_h4800_chain == h4800
                    and got.terminal_erosion == erosion
                ),
            }
        )
    return rows


__all__ = [
    "V22_03_H4800_POSITIVE_ROWS_MIN",
    "V22_03_RETENTION_RATIO_MIN",
    "V2203TerminalDecision",
    "classify_terminal_erosion",
    "classify_v22_03_source_chain",
    "terminal_retention_unit_tests",
]

