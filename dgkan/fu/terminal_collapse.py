"""v22.02 terminal-collapse source-chain and taxonomy helpers.

The helpers in this module only classify measured artifacts. They do not
generate functional directions and they do not inspect validation/test/future
outcomes for training-time decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from dgkan.fu.source_chain import SOURCE_EPS, finite_float, retention_ratio


V22_02_RETENTION_RATIO_MIN = 0.50


@dataclass(frozen=True)
class V2202SourceChainDecision:
    early_source_chain: int
    continuous_h3200_chain: int
    productive_h4800_chain: int
    terminal_collapse: int
    control_equivalent: int
    h1600_retention_ratio: float | str
    h3200_retention_ratio: float | str
    h4800_retention_ratio: float | str
    blocker: str


def classify_v22_02_source_chain(
    h100: Any,
    h400: Any,
    h800: Any,
    h1600: Any,
    h2400: Any,
    h3200: Any,
    h4800: Any,
    *,
    control_equivalent: Any = 0,
) -> V2202SourceChainDecision:
    s100 = finite_float(h100)
    s400 = finite_float(h400)
    s800 = finite_float(h800)
    s1600 = finite_float(h1600)
    s2400 = finite_float(h2400)
    s3200 = finite_float(h3200)
    s4800 = finite_float(h4800)
    control_eq = int(finite_float(control_equivalent, 0.0) != 0.0)
    early = int(not control_eq and s100 >= SOURCE_EPS and s400 >= SOURCE_EPS and s800 >= SOURCE_EPS)
    r1600 = retention_ratio(s1600, s800)
    r3200 = retention_ratio(s3200, s1600)
    r4800 = retention_ratio(s4800, s3200)
    continuous = int(
        early
        and s1600 >= SOURCE_EPS
        and s2400 >= SOURCE_EPS
        and s3200 >= SOURCE_EPS
        and r1600 != ""
        and r3200 != ""
        and float(r1600) >= V22_02_RETENTION_RATIO_MIN
        and float(r3200) >= V22_02_RETENTION_RATIO_MIN
    )
    productive_h4800 = int(
        continuous
        and s4800 >= SOURCE_EPS
        and r4800 != ""
        and float(r4800) >= V22_02_RETENTION_RATIO_MIN
    )
    terminal = int(continuous and (not math.isfinite(s4800) or s4800 < SOURCE_EPS))
    blockers: list[str] = []
    if control_eq:
        blockers.append("ControlEquivalent")
    if not early:
        blockers.append("EarlySourceChainMissing")
    if not continuous:
        blockers.append("ContinuousH3200Missing")
    if terminal:
        blockers.append("TerminalCollapse")
    return V2202SourceChainDecision(
        early_source_chain=early,
        continuous_h3200_chain=continuous,
        productive_h4800_chain=productive_h4800,
        terminal_collapse=terminal,
        control_equivalent=control_eq,
        h1600_retention_ratio=r1600,
        h3200_retention_ratio=r3200,
        h4800_retention_ratio=r4800,
        blocker="pass" if productive_h4800 else ";".join(blockers),
    )


def classify_terminal_collapse(row: dict[str, Any]) -> str:
    """Classify a measured collapsed group using v22.02 thresholds."""

    projection = finite_float(row.get("cumulative_optimizer_projection_on_source_h3200_to_h4800"))
    derivative = finite_float(row.get("source_derivative_h3200_to_h4800"))
    debt_growth = max(
        finite_float(row.get("tail_debt_growth_h3200_to_h4800"), 0.0),
        finite_float(row.get("LineC_channel_debt_growth_h3200_to_h4800"), 0.0),
        finite_float(row.get("ECE_debt_growth_h3200_to_h4800"), 0.0),
        finite_float(row.get("Brier_debt_growth_h3200_to_h4800"), 0.0),
    )
    stable_random_rate = finite_float(row.get("stable_random_h4800_source_rate"), 0.0)
    heterogeneity = finite_float(row.get("dataset_seed_heterogeneity_score"), 0.0)
    local_transfer = finite_float(row.get("B2_transfer_gain_h800"))
    source_projection_decay = finite_float(row.get("source_channel_projection_decay_h3200_h4800"))
    if math.isfinite(projection) and math.isfinite(derivative) and projection < -0.20 and derivative < -0.05:
        return "OptimizerWashout"
    if debt_growth >= 0.50 and math.isfinite(derivative) and derivative < 0.0:
        return "DebtExplosion"
    if stable_random_rate > 0.0:
        return "ControlEquivalentLateDrift"
    if heterogeneity >= 0.70:
        return "DatasetHeterogeneity"
    if math.isfinite(local_transfer) and local_transfer > 0.0 and math.isfinite(source_projection_decay) and source_projection_decay < 0.0:
        return "TargetMisalignment"
    return "UnknownTerminalCollapse"


def terminal_collapse_unit_tests() -> list[dict[str, Any]]:
    cases = [
        ("pass_h4800", (0.01, 0.02, 0.03, 0.03, 0.025, 0.02, 0.012), 1, 1, 1, 0),
        ("collapse", (0.01, 0.02, 0.03, 0.025, 0.02, 0.018, -0.01), 1, 1, 0, 1),
        ("missing_h2400", (0.01, 0.02, 0.03, 0.025, -0.01, 0.018, 0.012), 1, 0, 0, 0),
        ("control", (0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02), 0, 0, 0, 0),
    ]
    rows: list[dict[str, Any]] = []
    for name, values, early, h3200, h4800, terminal in cases:
        got = classify_v22_02_source_chain(*values, control_equivalent=int(name == "control"))
        rows.append(
            {
                "test": name,
                "expected_early": early,
                "actual_early": got.early_source_chain,
                "expected_h3200": h3200,
                "actual_h3200": got.continuous_h3200_chain,
                "expected_h4800": h4800,
                "actual_h4800": got.productive_h4800_chain,
                "expected_terminal": terminal,
                "actual_terminal": got.terminal_collapse,
                "pass": int(
                    got.early_source_chain == early
                    and got.continuous_h3200_chain == h3200
                    and got.productive_h4800_chain == h4800
                    and got.terminal_collapse == terminal
                ),
            }
        )
    return rows


__all__ = [
    "V22_02_RETENTION_RATIO_MIN",
    "V2202SourceChainDecision",
    "classify_terminal_collapse",
    "classify_v22_02_source_chain",
    "terminal_collapse_unit_tests",
]
