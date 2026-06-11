"""v22 source-chain metrics and route helpers.

These helpers are intentionally small and deterministic: they only classify
observed source readbacks and never generate FU directions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


SOURCE_EPS = 0.005
EARLY_ZERO = SOURCE_EPS
RETENTION_EPS = SOURCE_EPS
RETENTION_RATIO_MIN = 0.40


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in {"", None}:
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def retention_ratio(current: Any, previous: Any) -> float | str:
    prev = finite_float(previous)
    cur = finite_float(current)
    if not math.isfinite(prev) or prev <= 0.0:
        return ""
    if not math.isfinite(cur):
        return ""
    return max(0.0, cur) / prev


@dataclass(frozen=True)
class SourceChainDecision:
    early_source_chain: int
    continuous_retention_chain: int
    continuous_h4800_chain: int
    late_rebound: int
    terminal_collapse: int
    control_equivalent: int
    h1600_retention_ratio: float | str
    h3200_retention_ratio: float | str
    h4800_retention_ratio: float | str
    blocker: str


def classify_source_chain(
    h100: Any,
    h400: Any,
    h800: Any,
    h1600: Any,
    h3200: Any,
    h4800: Any = "",
    *,
    control_equivalent: Any = 0,
) -> SourceChainDecision:
    s100 = finite_float(h100)
    s400 = finite_float(h400)
    s800 = finite_float(h800)
    s1600 = finite_float(h1600)
    s3200 = finite_float(h3200)
    s4800 = finite_float(h4800)
    control_eq = int(finite_float(control_equivalent, 0.0) != 0.0)
    early = int(not control_eq and s100 >= SOURCE_EPS and s400 >= SOURCE_EPS and s800 >= SOURCE_EPS)
    r1600 = retention_ratio(s1600, s800)
    r3200 = retention_ratio(s3200, s1600)
    r4800 = retention_ratio(s4800, s3200)
    continuous = int(
        early
        and s800 >= RETENTION_EPS
        and s1600 >= RETENTION_EPS
        and s3200 >= RETENTION_EPS
        and r1600 != ""
        and r3200 != ""
        and float(r1600) >= RETENTION_RATIO_MIN
        and float(r3200) >= RETENTION_RATIO_MIN
    )
    continuous_h4800 = int(
        continuous
        and s4800 >= RETENTION_EPS
        and r4800 != ""
        and float(r4800) >= RETENTION_RATIO_MIN
    )
    late = int(
        not early
        and s800 < RETENTION_EPS
        and ((math.isfinite(s3200) and s3200 > RETENTION_EPS) or (math.isfinite(s4800) and s4800 > RETENTION_EPS))
    )
    terminal_collapse = int(continuous and (not math.isfinite(s4800) or s4800 < RETENTION_EPS))
    blockers: list[str] = []
    if control_eq:
        blockers.append("ControlEquivalent")
    if not early:
        blockers.append("EarlySourceChainMissing")
    if not continuous:
        blockers.append("ContinuousRetentionMissing")
    if terminal_collapse:
        blockers.append("TerminalCollapse")
    if late:
        blockers.append("LateReboundNotRetained")
    return SourceChainDecision(
        early_source_chain=early,
        continuous_retention_chain=continuous,
        continuous_h4800_chain=continuous_h4800,
        late_rebound=late,
        terminal_collapse=terminal_collapse,
        control_equivalent=control_eq,
        h1600_retention_ratio=r1600,
        h3200_retention_ratio=r3200,
        h4800_retention_ratio=r4800,
        blocker="pass" if early and continuous and not terminal_collapse else ";".join(blockers),
    )


def source_chain_row(row: dict[str, Any], prefix: str = "source_h") -> dict[str, Any]:
    decision = classify_source_chain(
        row.get(f"{prefix}100"),
        row.get(f"{prefix}400"),
        row.get(f"{prefix}800"),
        row.get(f"{prefix}1600"),
        row.get(f"{prefix}3200"),
        row.get(f"{prefix}4800"),
        control_equivalent=row.get("control_equivalent", row.get("control_equivalent_group", 0)),
    )
    return {
        "early_source_chain": decision.early_source_chain,
        "continuous_retention_chain": decision.continuous_retention_chain,
        "continuous_h4800_chain": decision.continuous_h4800_chain,
        "late_rebound_v22": decision.late_rebound,
        "terminal_collapse_v22": decision.terminal_collapse,
        "control_equivalent_v22": decision.control_equivalent,
        "h1600_retention_ratio_v22": decision.h1600_retention_ratio,
        "h3200_retention_ratio_v22": decision.h3200_retention_ratio,
        "h4800_retention_ratio_v22": decision.h4800_retention_ratio,
        "source_chain_blocker": decision.blocker,
    }


def source_chain_unit_tests() -> list[dict[str, Any]]:
    cases = [
        ("continuous_h4800", (0.01, 0.02, 0.03, 0.02, 0.01, 0.006, 0), 1, 1, 1, 0, 0, 0),
        ("all_zero_control", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1), 0, 0, 0, 0, 0, 1),
        ("zero_source_late_rebound", (0.0, 0.0, 0.0, 0.0, 0.02, 0.03, 0), 0, 0, 0, 1, 0, 0),
        ("control_explained_positive", (0.01, 0.02, 0.03, 0.02, 0.02, 0.02, 1), 0, 0, 0, 0, 0, 1),
        ("terminal_collapse", (0.01, 0.02, 0.03, 0.02, 0.01, -0.004, 0), 1, 1, 0, 0, 1, 0),
        ("h800_gap", (0.01, 0.01, -0.001, 0.02, 0.02, 0.02, 0), 0, 0, 0, 1, 0, 0),
        ("ratio_fail", (0.01, 0.02, 0.03, 0.005, 0.004, 0.004, 0), 1, 0, 0, 0, 0, 0),
    ]
    rows: list[dict[str, Any]] = []
    for name, values, early, continuous, h4800, late, terminal, control_eq in cases:
        got = classify_source_chain(*values[:6], control_equivalent=values[6])
        rows.append(
            {
                "test": name,
                "expected_early": early,
                "actual_early": got.early_source_chain,
                "expected_continuous": continuous,
                "actual_continuous": got.continuous_retention_chain,
                "expected_continuous_h4800": h4800,
                "actual_continuous_h4800": got.continuous_h4800_chain,
                "expected_late": late,
                "actual_late": got.late_rebound,
                "expected_terminal_collapse": terminal,
                "actual_terminal_collapse": got.terminal_collapse,
                "expected_control_equivalent": control_eq,
                "actual_control_equivalent": got.control_equivalent,
                "pass": int(
                    got.early_source_chain == early
                    and got.continuous_retention_chain == continuous
                    and got.continuous_h4800_chain == h4800
                    and got.late_rebound == late
                    and got.terminal_collapse == terminal
                    and got.control_equivalent == control_eq
                ),
            }
        )
    return rows


__all__ = [
    "SOURCE_EPS",
    "EARLY_ZERO",
    "RETENTION_EPS",
    "RETENTION_RATIO_MIN",
    "SourceChainDecision",
    "classify_source_chain",
    "finite_float",
    "retention_ratio",
    "source_chain_row",
    "source_chain_unit_tests",
]
