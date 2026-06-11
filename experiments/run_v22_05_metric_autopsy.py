#!/usr/bin/env python3
"""Autopsy for v22.05 metric-first FU failures.

The script only reads completed source-retention artifacts. It does not create
new experimental values; it classifies and correlates already logged metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_05_common import (  # noqa: E402
    PYTHON,
    append_exec,
    ensure_out,
    finite_float,
    mean,
    read_rows,
    simple_svg,
    write_json,
    write_rows,
)


METRIC_ATTEMPTS = [
    ("metric_projection", "metric_first_mlp_full"),
    ("source_memory_repair", "metric_source_memory_repair_full"),
]

METRIC_IDS = {
    "MLP-MF-G0-L2",
    "MLP-MF-G1-DiagFisher",
    "MLP-MF-G2-PopRiskDiag",
    "MLP-MF-G3-SobolevH1",
    "MLP-MF-G4-RKHSKNN",
    "MLP-MF-G5-FisherRKHS",
    "MLP-MF-G6-LowNDS",
    "MLP-MF-G8-Ensemble",
    "MLP-MFSM-G0-L2",
    "MLP-MFSM-G1-DiagFisher",
    "MLP-MFSM-G2-PopRiskDiag",
    "MLP-MFSM-G3-SobolevH1",
    "MLP-MFSM-G4-RKHSKNN",
    "MLP-MFSM-G5-FisherRKHS",
    "MLP-MFSM-G6-LowNDS",
    "MLP-MFSM-G8-Ensemble",
    "MLP-MFT-T0-LossCotangent",
    "MLP-MFT-T3-LowDegreeReadout",
    "MLP-MFT-T4-B1Transfer",
    "MLP-MFT-T5-SourceProjectedB3Null",
    "MLP-HC3-T1-SplitConsensus",
    "MLP-HC3-T2-PopRiskSNR",
    "MLP-HC3-T6-DiffeomorphicNoFold",
    "MLP-HC3-T7-RandomMatched",
    "MLP-HC3-T8-SignFlipped",
    "MLP-MFTW-T3-LowDegreeReadout-stop800",
    "MLP-MFTW-T4-B1Transfer-stop800",
    "MLP-MFTW-T5-SourceProjectedB3Null-stop800",
    "MLP-MFTW-T5-SourceProjectedB3Null-stop1600",
    "MLP-MFTC-T3-stop800-postM181",
    "MLP-MFTC-T5-stop800-postM181",
    "MLP-MFTC-T5-stop800-postM218",
    "MLP-MFTC-T5-stop1600-postM181",
    "MLP-HC2-NoProjection",
    "MLP-HC2-L2Projection",
    "MLP-HC2-HalfProjection",
    "MLP-HC2-H4000L2Projection",
    "MLP-HC2-H4000HalfProjection",
}

CORRELATION_FIELDS = [
    "metric_energy_L2",
    "metric_energy_Fisher",
    "metric_energy_PopRisk",
    "metric_energy_Sobolev",
    "metric_energy_RKHS",
    "metric_energy_mean",
    "NDS",
    "raw_gradient_NDS",
    "first_order_gain",
    "second_order_penalty",
    "optimizer_cumulative_projection_on_source_Gf",
    "negative_projection_fraction",
    "source_preservation_projection_norm",
    "source_state_projected_cos",
    "LineC_debt_final",
    "ECE_debt_final",
    "Brier_debt_final",
    "hidden_source_fraction",
    "readout_source_fraction",
]

TARGET_FIELDS = [
    "source_h4800_mean",
    "source_h3200_mean",
    "source_h100_mean",
    "source_decay_h3200_to_h4800",
    "R4800_over_3200",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument(
        "--attempt",
        action="append",
        default=[],
        help="Optional attempt as label:path. Defaults to v22.05 metric-only and source-memory repair dirs.",
    )
    return p


def _safe_ratio(num: Any, den: Any) -> float:
    n = finite_float(num)
    d = finite_float(den)
    if n == n and d == d and d > 0:
        return n / d
    return float("nan")


def _rank(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (vx * vy) ** 0.5


def _correlate(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    out = []
    for xfield in CORRELATION_FIELDS:
        for yfield in TARGET_FIELDS:
            pairs = []
            for row in rows:
                x = finite_float(row.get(xfield))
                y = finite_float(row.get(yfield))
                if x == x and y == y:
                    pairs.append((x, y))
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            pearson = _pearson(xs, ys)
            spearman = _pearson(_rank(xs), _rank(ys)) if len(xs) >= 3 else float("nan")
            out.append(
                {
                    "attempt_label": label,
                    "x_metric": xfield,
                    "y_metric": yfield,
                    "n": len(xs),
                    "pearson": pearson if pearson == pearson else "",
                    "spearman": spearman if spearman == spearman else "",
                }
            )
    return out


def _failure_mode(row: dict[str, Any]) -> str:
    h100 = finite_float(row.get("source_h100_mean"), -999.0)
    h800 = finite_float(row.get("source_h800_mean"), -999.0)
    h3200 = finite_float(row.get("source_h3200_mean"), -999.0)
    h4800 = finite_float(row.get("source_h4800_mean"), -999.0)
    r4800 = finite_float(row.get("R4800_over_3200"), float("nan"))
    control_equiv = int(finite_float(row.get("control_equivalent_fraction"), 0.0) >= 1.0)
    if h100 < 0.005 and h800 < 0.005:
        return "EarlySourceFormationFailed;ControlEquivalent" if control_equiv else "EarlySourceFormationFailed"
    if h3200 < 0.005:
        return "ContinuousH3200Missing"
    if h4800 < 0.005:
        return "TerminalSourceMissing"
    if r4800 == r4800 and r4800 < 0.50:
        return "TerminalRetentionRatioBelowGate"
    if control_equiv:
        return "ControlEquivalent"
    return "Unclassified"


def _horizon_mean(
    trace_group: list[dict[str, Any]],
    matrix_group: list[dict[str, Any]],
    key: str,
    horizon: int = 4800,
) -> float:
    vals = [
        finite_float(r.get(key))
        for r in trace_group
        if int(finite_float(r.get("step"), -1.0)) == horizon
    ]
    vals = [v for v in vals if v == v]
    if vals:
        return sum(vals) / len(vals)
    wide = mean(matrix_group, f"{key}_h{horizon}")
    if wide == wide:
        return wide
    return mean(matrix_group, key)


def _enrich_attempt(label: str, source_dir: Path) -> list[dict[str, Any]]:
    summary = read_rows(source_dir / "v21_01_mlp_source_retention_summary.csv") or read_rows(source_dir / "v21_01_source_retention_summary.csv")
    matrix = read_rows(source_dir / "v21_01_source_retention_matrix.csv")
    traces = read_rows(source_dir / "v21_01_source_retention_raw_traces.csv")
    by_id = {mid: [r for r in matrix if str(r.get("v21_id", "")) == mid] for mid in METRIC_IDS}
    traces_by_id = {mid: [r for r in traces if str(r.get("v21_id", "")) == mid] for mid in METRIC_IDS}
    controls = [r for r in summary if str(r.get("mechanism", "")).startswith("CTRL")]
    max_control_h4800 = max((finite_float(r.get("source_h4800_mean"), -999.0) for r in controls), default=-999.0)
    max_control_r4800 = max((finite_float(r.get("retention_h4800_over_h3200"), -999.0) for r in controls), default=-999.0)
    rows = []
    for row in summary:
        vid = str(row.get("v21_id", ""))
        if vid not in METRIC_IDS:
            continue
        group = by_id.get(vid, [])
        trace_group = traces_by_id.get(vid, [])
        h100 = finite_float(row.get("source_h100_mean"), -999.0)
        h400 = finite_float(row.get("source_h400_mean"), -999.0)
        h800 = finite_float(row.get("source_h800_mean"), -999.0)
        h3200 = finite_float(row.get("source_h3200_mean"), -999.0)
        h4800 = finite_float(row.get("source_h4800_mean"), -999.0)
        h6400 = finite_float(row.get("source_h6400_mean"), -999.0)
        item = {
            **row,
            "attempt_label": label,
            "source_dir": str(source_dir.relative_to(ROOT)) if source_dir.is_relative_to(ROOT) else str(source_dir),
            "R4800_over_3200": row.get("retention_h4800_over_h3200", ""),
            "R6400_over_4800": row.get("retention_h6400_over_h4800", ""),
            "row_h4800_positive_count": row.get("source_h4800_pass_count", ""),
            "control_equivalent_fraction": int(h4800 <= max_control_h4800 + 1.0e-12),
            "control_max_h4800": max_control_h4800,
            "control_max_R4800_over_3200": max_control_r4800,
            "source_decay_h3200_to_h4800": h4800 - h3200 if h3200 == h3200 and h4800 == h4800 else "",
            "source_decay_h100_to_h4800": h4800 - h100 if h100 == h100 and h4800 == h4800 else "",
            "source_rebound_h4800_to_h6400": h6400 - h4800 if h6400 == h6400 and h4800 == h4800 else "",
            "source_erosion_ratio_h4800_over_h3200": _safe_ratio(h4800, h3200),
            "early_source_chain_open": int(h100 >= 0.005 and h400 >= 0.005 and h800 >= 0.005),
            "terminal_retention_evaluable": int(h3200 >= 0.005),
            "productive_h4800_gate": int(h4800 >= 0.005 and _safe_ratio(h4800, h3200) >= 0.50 and finite_float(row.get("source_h4800_pass_count"), 0.0) >= 7),
            "metric_energy_L2": _horizon_mean(trace_group, group, "metric_energy_L2"),
            "metric_energy_Fisher": _horizon_mean(trace_group, group, "metric_energy_Fisher"),
            "metric_energy_PopRisk": _horizon_mean(trace_group, group, "metric_energy_PopRisk"),
            "metric_energy_Sobolev": _horizon_mean(trace_group, group, "metric_energy_Sobolev"),
            "metric_energy_RKHS": _horizon_mean(trace_group, group, "metric_energy_RKHS"),
            "metric_energy_mean": _horizon_mean(trace_group, group, "metric_energy_mean"),
            "NDS": _horizon_mean(trace_group, group, "NDS"),
            "raw_gradient_NDS": _horizon_mean(trace_group, group, "raw_gradient_NDS"),
            "first_order_gain": _horizon_mean(trace_group, group, "first_order_gain"),
            "second_order_penalty": _horizon_mean(trace_group, group, "second_order_penalty"),
            "metric_projection_cosine": _horizon_mean(trace_group, group, "metric_projection_cosine"),
            "optimizer_cumulative_projection_on_source_Gf": _horizon_mean(trace_group, group, "optimizer_cumulative_projection_on_source_Gf"),
            "negative_projection_fraction": _horizon_mean(trace_group, group, "negative_projection_fraction"),
            "source_preservation_projection_norm": _horizon_mean(trace_group, group, "source_preservation_projection_norm"),
            "source_state_projected_cos": _horizon_mean(trace_group, group, "source_state_projected_cos"),
            "LineC_debt_final": _horizon_mean(trace_group, group, "LineC_fast_loss"),
            "ECE_debt_final": _horizon_mean(trace_group, group, "ECE"),
            "Brier_debt_final": _horizon_mean(trace_group, group, "Brier"),
            "hidden_source_fraction": _horizon_mean(trace_group, group, "hidden_source_fraction"),
            "readout_source_fraction": _horizon_mean(trace_group, group, "readout_source_fraction"),
        }
        item["autopsy_failure_mode"] = _failure_mode(item)
        rows.append(item)
    return rows


def _best_correlation(corr_rows: list[dict[str, Any]], target: str) -> dict[str, Any]:
    candidates = []
    for row in corr_rows:
        if row.get("y_metric") != target:
            continue
        p = finite_float(row.get("pearson"))
        if p == p and int(finite_float(row.get("n"), 0.0)) >= 3:
            candidates.append((abs(p), row))
    if not candidates:
        return {}
    return sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    attempts: list[tuple[str, Path]] = []
    if args.attempt:
        for raw in args.attempt:
            label, path = raw.split(":", 1)
            attempts.append((label, Path(path)))
    else:
        attempts = [(label, out_dir / rel) for label, rel in METRIC_ATTEMPTS]
    rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    for label, source_dir in attempts:
        attempt_rows = _enrich_attempt(label, source_dir)
        rows.extend(attempt_rows)
        corr_rows.extend(_correlate(attempt_rows, label))
    corr_rows.extend(_correlate(rows, "combined"))
    best = sorted(rows, key=lambda r: finite_float(r.get("source_h4800_mean"), -999.0), reverse=True)[0] if rows else {}
    best_corr = _best_correlation(corr_rows, "source_h4800_mean")
    source_failed = sum(1 for r in rows if int(finite_float(r.get("early_source_chain_open"), 0.0)) == 0)
    evaluable = sum(1 for r in rows if int(finite_float(r.get("terminal_retention_evaluable"), 0.0)))
    control_equiv = sum(1 for r in rows if int(finite_float(r.get("control_equivalent_fraction"), 0.0)))
    productive = sum(1 for r in rows if int(finite_float(r.get("productive_h4800_gate"), 0.0)))
    dominant = "MetricNoEffect"
    if rows and source_failed == len(rows):
        dominant = "EarlySourceFormationFailed;ControlEquivalent;MetricNoEffect" if control_equiv == len(rows) else "EarlySourceFormationFailed;MetricNoEffect"
    elif evaluable and productive == 0:
        dominant = "TerminalErosionAfterH3200"
    verdict = {
        "attempt_labels": ";".join(label for label, _ in attempts),
        "metric_families_tried": "G0-L2;G1-DiagFisher;G2-PopRiskDiag;G3-SobolevH1;G4-RKHSKNN;G5-FisherRKHS;G6-LowNDS;G8-Ensemble;T0/T1/T2/T3/T4/T5/T6/T7/T8 metric-target;H-C2-h3200/h4000-projection",
        "metric_rows": len(rows),
        "source_formation_failed_rows": source_failed,
        "terminal_retention_evaluable_rows": evaluable,
        "control_equivalent_rows": control_equiv,
        "productive_h4800_rows": productive,
        "best_attempt_label": best.get("attempt_label", ""),
        "best_v21_id": best.get("v21_id", ""),
        "best_source_h100_mean": best.get("source_h100_mean", ""),
        "best_source_h3200_mean": best.get("source_h3200_mean", ""),
        "best_source_h4800_mean": best.get("source_h4800_mean", ""),
        "best_R4800_over_3200": best.get("R4800_over_3200", ""),
        "dominant_failure_mode": dominant,
        "strongest_logged_energy_correlation_to_h4800": best_corr.get("x_metric", ""),
        "strongest_logged_energy_correlation_pearson": best_corr.get("pearson", ""),
        "strongest_logged_energy_correlation_n": best_corr.get("n", ""),
        "next_metric_family_candidate": "explicit train-only loss-cotangent/source-channel metric target before terminal preservation",
    }
    write_rows(out_dir / "v22_05_metric_autopsy_summary.csv", rows)
    write_rows(out_dir / "v22_05_metric_energy_retention_correlations.csv", corr_rows)
    write_rows(out_dir / "v22_05_metric_failure_verdict.csv", [verdict])
    write_json(out_dir / "v22_05_metric_failure_verdict.json", verdict)
    simple_svg(out_dir / "figures/metric_autopsy_source_chain.svg", "v22.05 metric autopsy source chain", rows, "source_h4800_mean")
    simple_svg(out_dir / "figures/metric_autopsy_energy_correlation.svg", "v22.05 metric autopsy correlation", corr_rows, "pearson")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_05_metric_autopsy.py --out-dir {out_dir}",
        status="completed",
        note=f"rows={len(rows)} dominant={dominant} best={verdict['best_v21_id']}",
    )


if __name__ == "__main__":
    main()
