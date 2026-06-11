#!/usr/bin/env python3
"""Summarize v22.05 metric-first MLP FU source-retention rows."""

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
    V2204_OFFICIAL,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    mean,
    read_rows,
    simple_svg,
    write_json,
    write_rows,
)


METRIC_IDS = {
    "MLP-MF-G0-L2": "G0-L2",
    "MLP-MF-G1-DiagFisher": "G1-DiagFisher",
    "MLP-MF-G2-PopRiskDiag": "G2-PopRiskDiag",
    "MLP-MF-G3-SobolevH1": "G3-SobolevH1",
    "MLP-MF-G4-RKHSKNN": "G4-RKHS-KNN",
    "MLP-MF-G5-FisherRKHS": "G5-Fisher-RKHS",
    "MLP-MF-G6-LowNDS": "G6-LowNDS",
    "MLP-MF-G8-Ensemble": "G8-MetricEnsemble",
    "MLP-MFSM-G0-L2": "G0-L2-source-memory",
    "MLP-MFSM-G1-DiagFisher": "G1-DiagFisher-source-memory",
    "MLP-MFSM-G2-PopRiskDiag": "G2-PopRiskDiag-source-memory",
    "MLP-MFSM-G3-SobolevH1": "G3-SobolevH1-source-memory",
    "MLP-MFSM-G4-RKHSKNN": "G4-RKHS-KNN-source-memory",
    "MLP-MFSM-G5-FisherRKHS": "G5-Fisher-RKHS-source-memory",
    "MLP-MFSM-G6-LowNDS": "G6-LowNDS-source-memory",
    "MLP-MFSM-G8-Ensemble": "G8-MetricEnsemble-source-memory",
    "MLP-MFT-T0-LossCotangent": "T0-LossCotangentReadout-metric-target",
    "MLP-MFT-T3-LowDegreeReadout": "T3-LowDegreeReadout-metric-target",
    "MLP-MFT-T4-B1Transfer": "T4-B1LossTransfer-metric-target",
    "MLP-MFT-T5-SourceProjectedB3Null": "T5-SourceProjectedB3Null-metric-target",
    "MLP-HC3-T1-SplitConsensus": "T1-SplitConsensus-metric-target",
    "MLP-HC3-T2-PopRiskSNR": "T2-PopRiskSNR-metric-target",
    "MLP-HC3-T6-DiffeomorphicNoFold": "T6-DiffeomorphicNoFold-metric-target",
    "MLP-HC3-T7-RandomMatched": "T7-RandomMatchedActuation-control",
    "MLP-HC3-T8-SignFlipped": "T8-SignFlippedTarget-control",
    "MLP-HC4-T9-DebtCalibratedSource": "T9-DebtCalibratedSource-metric-target",
    "MLP-HC4-T9-stop1600-postM181": "T9-DebtCalibratedSource-stop1600-postM181",
    "MLP-HC4-T9-earlyburst-alt10": "T9-DebtCalibratedSource-earlyburst-alt10",
    "MLP-HC4-T9-earlyburst-alt10-stop1600-postM181": "T9-DebtCalibratedSource-earlyburst-alt10-stop1600-postM181",
    "MLP-HC4-T9-earlyburst-alt10-stop800-postM181": "T9-DebtCalibratedSource-earlyburst-alt10-stop800-postM181",
    "MLP-HC4-T9-earlyburst-alt10-stop400-postM181": "T9-DebtCalibratedSource-earlyburst-alt10-stop400-postM181",
    "MLP-HC4-T9-earlyburst-alt10-stop400-postM181-lr2": "T9-DebtCalibratedSource-earlyburst-alt10-stop400-postM181-lr2",
    "MLP-HC4-T9-earlyburst-alt10-stop400-postM218": "T9-DebtCalibratedSource-earlyburst-alt10-stop400-postM218",
    "MLP-HC5-T10-ObservableMidDebtSource": "T10-ObservableMidDebtSource-metric-target",
    "MLP-HC5-T10-earlyburst-alt10": "T10-ObservableMidDebtSource-earlyburst-alt10",
    "MLP-HC5-T10-earlyburst-alt10-stop200": "T10-ObservableMidDebtSource-earlyburst-alt10-stop200",
    "MLP-HC5-T10-earlyburst-alt10-stop400": "T10-ObservableMidDebtSource-earlyburst-alt10-stop400",
    "MLP-HC5-T10-earlyburst-alt10-stop800": "T10-ObservableMidDebtSource-earlyburst-alt10-stop800",
    "MLP-HC5-T10-earlyburst-alt10-stop400-postM181": "T10-ObservableMidDebtSource-earlyburst-alt10-stop400-postM181",
    "MLP-HC5-T10-earlyburst-alt10-stop400-postM218": "T10-ObservableMidDebtSource-earlyburst-alt10-stop400-postM218",
    "MLP-HC6-T10M181Bridge-stop400": "T10-ObservableMidDebtSource-M181Bridge-stop400",
    "MLP-HC6-T10M181Bridge-stop800": "T10-ObservableMidDebtSource-M181Bridge-stop800",
    "MLP-HC6-T10M181Bridge-stop800-lr025": "T10-ObservableMidDebtSource-M181Bridge-stop800-lr025",
    "MLP-HC6-T10M181Bridge-stop1200-lr025": "T10-ObservableMidDebtSource-M181Bridge-stop1200-lr025",
    "MLP-HC6-T10M181Bridge-stop1600-lr025": "T10-ObservableMidDebtSource-M181Bridge-stop1600-lr025",
    "MLP-HC7-T11-GradObservableMidDebt-stop400": "T11-GradientObservableMidDebtSource-stop400",
    "MLP-HC7-T11-GradObservableMidDebt-stop800": "T11-GradientObservableMidDebtSource-stop800",
    "MLP-HC7-T11-GradObservableMidDebt-stop1600": "T11-GradientObservableMidDebtSource-stop1600",
    "MLP-HC8-T12-PreH3200SourceAntiWashout-stop800": "T12-PreH3200SourceChannelAntiWashout-stop800",
    "MLP-HC8-T12-PreH3200SourceAntiWashout-stop1200": "T12-PreH3200SourceChannelAntiWashout-stop1200",
    "MLP-HC8-T12-PreH3200SourceAntiWashout-stop1600": "T12-PreH3200SourceChannelAntiWashout-stop1600",
    "MLP-HC8-T12-PreH3200SourceAntiWashout-stop800-lr100": "T12-PreH3200SourceChannelAntiWashout-stop800-lr100",
    "MLP-HC8-T12-PreH3200SourceAntiWashout-stop1200-lr100": "T12-PreH3200SourceChannelAntiWashout-stop1200-lr100",
    "MLP-HC8-T12-PreH3200SourceAntiWashout-stop1600-lr100": "T12-PreH3200SourceChannelAntiWashout-stop1600-lr100",
    "MLP-HC8-T12-PreH3200SourceAntiWashout-stop1600-lr200": "T12-PreH3200SourceChannelAntiWashout-stop1600-lr200",
    "MLP-HC8-T12-PreH3200SourceAntiWashout-stop2400-lr100": "T12-PreH3200SourceChannelAntiWashout-stop2400-lr100",
    "MLP-HC8-T12-PreH3200SourceAntiWashout-stop2400-lr200": "T12-PreH3200SourceChannelAntiWashout-stop2400-lr200",
    "MLP-HC8-T12-PreH3200SourceAntiWashout-stop3200-lr200": "T12-PreH3200SourceChannelAntiWashout-stop3200-lr200",
    "MLP-HC9-T13-RiskWeightedSplitConsensus-stop800-lr100": "T13-RiskWeightedSplitConsensusSource-stop800-lr100",
    "MLP-HC9-T13-RiskWeightedSplitConsensus-stop1200-lr100": "T13-RiskWeightedSplitConsensusSource-stop1200-lr100",
    "MLP-HC9-T13-RiskWeightedSplitConsensus-stop1600-lr100": "T13-RiskWeightedSplitConsensusSource-stop1600-lr100",
    "MLP-HC9-T13-RiskWeightedSplitConsensus-stop2400-lr100": "T13-RiskWeightedSplitConsensusSource-stop2400-lr100",
    "MLP-HC10-T14-ViewConsistentSourceCarry-stop800-lr100": "T14-ViewConsistentSourceCarry-stop800-lr100",
    "MLP-HC10-T14-ViewConsistentSourceCarry-stop1600-lr100": "T14-ViewConsistentSourceCarry-stop1600-lr100",
    "MLP-HC10-T14-ViewConsistentSourceCarry-stop2400-lr100": "T14-ViewConsistentSourceCarry-stop2400-lr100",
    "MLP-HC11-T15-ContinuousSourceCarry-stop1600-lr200": "T15-ContinuousSourceCarryAntiWashout-stop1600-lr200",
    "MLP-HC11-T15-ContinuousSourceCarry-stop2400-lr200": "T15-ContinuousSourceCarryAntiWashout-stop2400-lr200",
    "MLP-HC11-T15-ContinuousSourceCarry-stop2400-lr300": "T15-ContinuousSourceCarryAntiWashout-stop2400-lr300",
    "MLP-HC11-T15-ContinuousSourceCarry-stop2400-lr500": "T15-ContinuousSourceCarryAntiWashout-stop2400-lr500",
    "MLP-HC11-T15-ContinuousSourceCarry-stop3200-lr300": "T15-ContinuousSourceCarryAntiWashout-stop3200-lr300",
    "MLP-HC12-T16-NoiseOrthogonalCarry-stop2400-lr500": "T16-NoiseOrthogonalSourceCarry-stop2400-lr500",
    "MLP-HC12-T16-NoiseOrthogonalCarry-stop3200-lr300": "T16-NoiseOrthogonalSourceCarry-stop3200-lr300",
    "MLP-MFTW-T3-LowDegreeReadout-stop800": "T3-LowDegreeReadout-metric-target-stop800",
    "MLP-MFTW-T4-B1Transfer-stop800": "T4-B1LossTransfer-metric-target-stop800",
    "MLP-MFTW-T5-SourceProjectedB3Null-stop800": "T5-SourceProjectedB3Null-metric-target-stop800",
    "MLP-MFTW-T5-SourceProjectedB3Null-stop1600": "T5-SourceProjectedB3Null-metric-target-stop1600",
    "MLP-MFTC-T3-stop800-postM181": "T3-LowDegreeReadout-stop800-postM181",
    "MLP-MFTC-T5-stop800-postM181": "T5-SourceProjectedB3Null-stop800-postM181",
    "MLP-MFTC-T5-stop800-postM218": "T5-SourceProjectedB3Null-stop800-postM218",
    "MLP-MFTC-T5-stop1600-postM181": "T5-SourceProjectedB3Null-stop1600-postM181",
    "MLP-HC2-NoProjection": "HC2-H3200-no-projection",
    "MLP-HC2-L2Projection": "HC2-H3200-L2-projection",
    "MLP-HC2-HalfProjection": "HC2-H3200-half-projection",
    "MLP-HC2-H4000L2Projection": "HC2-H4000-L2-recompute-projection",
    "MLP-HC2-H4000HalfProjection": "HC2-H4000-half-recompute-projection",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", required=True)
    p.add_argument("--baseline-dir", default=str(V2204_OFFICIAL))
    p.add_argument("--label", default="metric-first MLP FU")
    return p


def _baseline_best(path: Path) -> dict[str, Any]:
    rows = []
    for name in [
        "v22_04_d1b_single_roworth_audit.csv",
        "v22_04_d1a_single_spp_lambda025_audit.csv",
        "v22_04_d1a_lambda_strength_wired3_audit.csv",
        "v22_04_d1_source_preserving_fu_audit.csv",
    ]:
        rows.extend(read_rows(path / name))
    candidates = [r for r in rows if str(r.get("v22_id", "")).startswith("MLP")]
    if not candidates:
        return {}
    return sorted(candidates, key=lambda r: finite_float(r.get("h4800_retention_ratio"), -999.0), reverse=True)[0]


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


def _metric_rows(
    summary: list[dict[str, Any]],
    matrix: list[dict[str, Any]],
    traces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {mid: [r for r in matrix if str(r.get("v21_id", "")) == mid] for mid in METRIC_IDS}
    traces_by_id = {mid: [r for r in traces if str(r.get("v21_id", "")) == mid] for mid in METRIC_IDS}
    out = []
    controls = [r for r in summary if str(r.get("mechanism", "")).startswith("CTRL")]
    max_control_h4800 = max((finite_float(r.get("source_h4800_mean"), -999.0) for r in controls), default=-999.0)
    max_control_r4800 = max((finite_float(r.get("retention_h4800_over_h3200"), -999.0) for r in controls), default=-999.0)
    for row in summary:
        vid = str(row.get("v21_id", ""))
        if vid not in METRIC_IDS:
            continue
        group = by_id.get(vid, [])
        trace_group = traces_by_id.get(vid, [])
        item = {
            **row,
            "metric_name": METRIC_IDS[vid],
            "R1600_over_800": row.get("retention_h1600_over_h800", ""),
            "R3200_over_1600": row.get("retention_h3200_over_h1600", ""),
            "R4800_over_3200": row.get("retention_h4800_over_h3200", ""),
            "R6400_over_4800": row.get("retention_h6400_over_h4800", ""),
            "row_h4800_positive_count": row.get("source_h4800_pass_count", ""),
            "control_equivalent_fraction": int(finite_float(row.get("source_h4800_mean"), -999.0) <= max_control_h4800 + 1.0e-12),
            "control_max_h4800": max_control_h4800,
            "control_max_R4800_over_3200": max_control_r4800,
            "metric_energy_L2": _horizon_mean(trace_group, group, "metric_energy_L2"),
            "metric_energy_Fisher": _horizon_mean(trace_group, group, "metric_energy_Fisher"),
            "metric_energy_Sobolev": _horizon_mean(trace_group, group, "metric_energy_Sobolev"),
            "metric_energy_RKHS": _horizon_mean(trace_group, group, "metric_energy_RKHS"),
            "metric_energy_mean": _horizon_mean(trace_group, group, "metric_energy_mean"),
            "NDS": _horizon_mean(trace_group, group, "NDS"),
            "raw_gradient_NDS": _horizon_mean(trace_group, group, "raw_gradient_NDS"),
            "first_order_gain": _horizon_mean(trace_group, group, "first_order_gain"),
            "second_order_penalty": _horizon_mean(trace_group, group, "second_order_penalty"),
            "optimizer_cumulative_projection_on_source_Gf": _horizon_mean(trace_group, group, "optimizer_cumulative_projection_on_source_Gf"),
            "negative_projection_fraction": _horizon_mean(trace_group, group, "negative_projection_fraction"),
            "source_preservation_projection_norm": _horizon_mean(trace_group, group, "source_preservation_projection_norm"),
            "source_state_projected_cos": _horizon_mean(trace_group, group, "source_state_projected_cos"),
            "source_state_signal_gain": _horizon_mean(trace_group, group, "source_state_signal_gain"),
            "source_state_corrupt_gain": _horizon_mean(trace_group, group, "source_state_corrupt_gain"),
            "source_state_gate_accept": _horizon_mean(trace_group, group, "source_state_gate_accept"),
            "source_state_balance_mean": _horizon_mean(trace_group, group, "source_state_balance_mean"),
            "source_state_consensus_density": _horizon_mean(trace_group, group, "source_state_consensus_density"),
            "PopRisk_SNR_score": _horizon_mean(trace_group, group, "PopRisk_SNR_score"),
            "PopRisk_snr_mean": _horizon_mean(trace_group, group, "PopRisk_snr_mean"),
            "PopRisk_snr_max": _horizon_mean(trace_group, group, "PopRisk_snr_max"),
            "PopRisk_mu_norm": _horizon_mean(trace_group, group, "PopRisk_mu_norm"),
            "PopRisk_var_mean": _horizon_mean(trace_group, group, "PopRisk_var_mean"),
            "target_gradient_conflict_cos_before": _horizon_mean(trace_group, group, "target_gradient_conflict_cos_before"),
            "target_gradient_conflict_cos_after": _horizon_mean(trace_group, group, "target_gradient_conflict_cos_after"),
            "target_gradient_conflict_removed_fraction": _horizon_mean(trace_group, group, "target_gradient_conflict_removed_fraction"),
            "source_observability_gate_accept": _horizon_mean(trace_group, group, "source_observability_gate_accept"),
            "LineC_debt_final": _horizon_mean(trace_group, group, "LineC_fast_loss"),
            "ECE_debt_final": _horizon_mean(trace_group, group, "ECE"),
            "Brier_debt_final": _horizon_mean(trace_group, group, "Brier"),
        }
        productive = int(
            finite_float(item.get("source_h4800_mean"), -999.0) >= 0.005
            and finite_float(item.get("R4800_over_3200"), 0.0) >= 0.50
            and int(float(item.get("row_h4800_positive_count") or 0)) >= 7
            and not int(item["control_equivalent_fraction"])
        )
        item["H_C1_exploration_S2_pass"] = productive
        if productive:
            item["failure_taxonomy"] = ""
        elif int(item["control_equivalent_fraction"]):
            item["failure_taxonomy"] = "ControlEquivalent"
        elif finite_float(item.get("NDS"), 0.0) > finite_float(item.get("raw_gradient_NDS"), float("inf")):
            item["failure_taxonomy"] = "HighNDSNoRetention"
        elif finite_float(item.get("metric_energy_Sobolev"), 0.0) > finite_float(item.get("metric_energy_L2"), float("inf")):
            item["failure_taxonomy"] = "SobolevHighEnergyNoRetention"
        elif finite_float(item.get("metric_energy_RKHS"), 0.0) > finite_float(item.get("metric_energy_L2"), float("inf")):
            item["failure_taxonomy"] = "RKHSNeighborhoodTear"
        elif finite_float(item.get("source_h4800_mean"), -999.0) < 0.005:
            item["failure_taxonomy"] = "MetricNoEffect"
        else:
            item["failure_taxonomy"] = "OptimizerErosion"
        out.append(item)
    return out


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir)
    baseline_dir = Path(args.baseline_dir)
    summary = read_rows(source_dir / "v21_01_mlp_source_retention_summary.csv") or read_rows(source_dir / "v21_01_source_retention_summary.csv")
    matrix = read_rows(source_dir / "v21_01_source_retention_matrix.csv")
    traces = read_rows(source_dir / "v21_01_source_retention_raw_traces.csv")
    metric_rows = _metric_rows(summary, matrix, traces)
    baseline = _baseline_best(baseline_dir)
    baseline_ratio = finite_float(baseline.get("h4800_retention_ratio"), float("nan"))
    def _best_key(row: dict[str, Any]) -> tuple[int, float, float]:
        ratio = finite_float(row.get("R4800_over_3200"))
        h4800 = finite_float(row.get("source_h4800_mean"), -999.0)
        h3200 = finite_float(row.get("source_h3200_mean"), -999.0)
        if ratio == ratio:
            return (1, ratio, h4800)
        return (0, h4800, h3200)

    best = sorted(metric_rows, key=_best_key, reverse=True)[0] if metric_rows else {}
    for row in metric_rows:
        row["baseline_best_v22_id"] = baseline.get("v22_id", "")
        row["baseline_best_R4800_over_3200"] = baseline_ratio
        row["R4800_improvement_vs_v2204_best"] = finite_float(row.get("R4800_over_3200")) - baseline_ratio if baseline_ratio == baseline_ratio else ""
    write_rows(out_dir / "v22_05_metric_first_mlp_summary.csv", metric_rows)
    write_rows(out_dir / "v22_05_metric_first_mlp_raw_matrix.csv", matrix)
    route = {
        "metric_rows": len(metric_rows),
        "baseline_best_v22_id": baseline.get("v22_id", ""),
        "baseline_best_R4800_over_3200": baseline_ratio if baseline_ratio == baseline_ratio else "",
        "best_metric_v21_id": best.get("v21_id", ""),
        "best_metric_name": best.get("metric_name", ""),
        "best_R4800_over_3200": best.get("R4800_over_3200", ""),
        "best_h4800": best.get("source_h4800_mean", ""),
        "best_row_h4800_positive_count": best.get("row_h4800_positive_count", ""),
        "productive_metric_rows": sum(int_flag(r.get("H_C1_exploration_S2_pass")) for r in metric_rows),
        "metric_improved_ge_0p03": int(any(finite_float(r.get("R4800_improvement_vs_v2204_best"), -999.0) >= 0.03 for r in metric_rows)),
    }
    if route["productive_metric_rows"]:
        route["functional_route"] = "F2-MLPProductiveTerminalSource"
    elif route["metric_improved_ge_0p03"]:
        route["functional_route"] = "F1-MetricRetentionImproved"
    else:
        route["functional_route"] = "F0-MetricNoEffect"
    write_json(out_dir / "v22_05_metric_first_route.json", route)
    simple_svg(out_dir / "figures/terminal_source_trajectory_h100_to_h6400.svg", "v22.05 terminal source trajectory", metric_rows, "source_h4800_mean")
    simple_svg(out_dir / "figures/R4800_over_3200_by_metric.svg", "v22.05 R4800/3200 by metric", metric_rows, "R4800_over_3200")
    simple_svg(out_dir / "figures/metric_energy_vs_retention_scatter.svg", "v22.05 metric energy vs retention", metric_rows, "metric_energy_mean")
    simple_svg(out_dir / "figures/NDS_vs_terminal_erosion_scatter.svg", "v22.05 NDS vs terminal erosion", metric_rows, "NDS")
    simple_svg(out_dir / "figures/Sobolev_energy_vs_h4800_scatter.svg", "v22.05 Sobolev energy vs h4800", metric_rows, "metric_energy_Sobolev")
    simple_svg(out_dir / "figures/RKHS_neighbor_preservation_vs_h4800.svg", "v22.05 RKHS energy vs h4800", metric_rows, "metric_energy_RKHS")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_05_metric_first_fu.py --source-dir {source_dir} --baseline-dir {baseline_dir} --out-dir {out_dir}",
        status="completed",
        note=f"metric_rows={len(metric_rows)} best={route['best_metric_v21_id']} route={route['functional_route']}",
    )


if __name__ == "__main__":
    main()
