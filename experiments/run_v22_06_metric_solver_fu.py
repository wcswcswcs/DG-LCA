#!/usr/bin/env python3
"""Summarize v22.06 metric-as-geometry solver source-retention rows."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_06_common import PYTHON, append_exec, ensure_out, finite_float, int_flag, mean, read_rows, simple_svg, write_json, write_rows  # noqa: E402


METRIC_SOLVER_IDS = {
    "MLP-V2206-S1-T0G0-ReadoutSolver": "S1-T0-G0-readout-exact",
    "MLP-V2206-S1-T1G0-SplitTransferSolver": "S1-T1-G0-split-transfer-readout-exact",
    "MLP-V2206-S1-T3G3-SignalSobolevSolver": "S1-T3-G3-signal-sobolev-readout-exact",
    "MLP-V2206-S1-T4G3-SmoothManifoldSolver": "S1-T4-G3-smooth-manifold-readout-exact",
    "MLP-V2206-S1-T5G6-LowNDSSolver": "S1-T5-G6-low-nds-readout-exact",
    "MLP-V2206-C4-T0G0-SourceStateCarry": "C4-I2-T0-G0-source-state-carry",
    "MLP-V2206-C4-T4G3-SourceStateCarry": "C4-I2-T4-G3-source-state-carry",
    "MLP-V2206-C4-T0G0-SourceStateCarry-lr50": "C4-I2-T0-G0-source-state-carry-lr50",
    "MLP-V2206-C4-T4G3-SourceStateCarry-lr50": "C4-I2-T4-G3-source-state-carry-lr50",
    "MLP-V2206-C4-T4G3-SourceStateCarry-lr100": "C4-I2-T4-G3-source-state-carry-lr100",
    "MLP-V2206-S1-T6G0-DualMemorySolver": "S1-T6-G0-dual-memory-readout-exact",
    "MLP-V2206-C4-T6G0-SourceStateCarry": "C4-I2-T6-G0-source-state-carry",
    "MLP-V2206-C4-T6G0-EarlyWarmCarry-stop800-lr50": "C4-I1-T6-G0-early-warm-carry-stop800-lr50",
    "MLP-V2206-C4-T6G0-EarlyWarmCarry-stop1200-lr50": "C4-I1-T6-G0-early-warm-carry-stop1200-lr50",
    "MLP-V2206-C4-T0G0-EarlyWarmCarry-stop800-lr50": "C4-I1-T0-G0-early-warm-carry-stop800-lr50",
    "MLP-V2206-C4-T4G3-EarlyWarmCarry-stop800-lr50": "C4-I1-T4-G3-early-warm-carry-stop800-lr50",
    "MLP-V2206-S2-T7G0-HiddenBlockSolver": "S2-T7-G0-hidden-readout-block",
    "MLP-V2206-C4-T7G0-SourceStateCarry": "C4-I2-T7-G0-hidden-block-source-state-carry",
    "MLP-V2206-C4-T7G0-EarlyWarmCarry-stop800-lr50": "C4-I1-T7-G0-hidden-block-early-warm-carry-stop800-lr50",
    "MLP-V2206-C4-T7G0-EarlyWarmCarry-stop1200-lr50": "C4-I1-T7-G0-hidden-block-early-warm-carry-stop1200-lr50",
    "MLP-V2206-S2-T8G0-AdaptiveHiddenBlockSolver": "S2-T8-G0-adaptive-hidden-readout-block",
    "MLP-V2206-C4-T8G0-SourceStateCarry": "C4-I2-T8-G0-adaptive-hidden-block-source-state-carry",
    "MLP-V2206-C4-T8G0-EarlyWarmCarry-stop800-lr50": "C4-I1-T8-G0-adaptive-hidden-block-early-warm-carry-stop800-lr50",
    "MLP-V2206-S2-T9G0-C3GatedHiddenBlockSolver": "S2-T9-G0-c3-gated-hidden-readout-block",
    "MLP-V2206-C4-T9G0-SourceStateCarry": "C4-I2-T9-G0-c3-gated-hidden-block-source-state-carry",
    "MLP-V2206-C4-T9G0-EarlyWarmCarry-stop800-lr50": "C4-I1-T9-G0-c3-gated-hidden-block-early-warm-carry-stop800-lr50",
    "MLP-V2206-S2-T10G0-CompensatedHiddenBlockSolver": "S2-T10-G0-compensated-hidden-readout-block",
    "MLP-V2206-C4-T10G0-SourceStateCarry": "C4-I2-T10-G0-compensated-hidden-block-source-state-carry",
    "MLP-V2206-C4-T10G0-EarlyWarmCarry-stop800-lr50": "C4-I1-T10-G0-compensated-hidden-block-early-warm-carry-stop800-lr50",
    "MLP-V2206-C4-T10G0-EarlyWarmCarry-stop1200-lr50": "C4-I1-T10-G0-compensated-hidden-block-early-warm-carry-stop1200-lr50",
    "MLP-V2206-C4-T10G0-PeriodicCarry-alt100-lr50": "C4-I3-T10-G0-compensated-hidden-block-periodic-carry-alt100-lr50",
    "MLP-V2206-C4-T10G0-PeriodicCarry-alt200-lr50": "C4-I3-T10-G0-compensated-hidden-block-periodic-carry-alt200-lr50",
    "MLP-V2206-C4-T10G0-SGDBootstrap400Carry-lr50": "C4-I1I2-T10-G0-sgd-bootstrap400-compensated-hidden-carry-lr50",
    "MLP-V2206-C4-T10G0-SGDBootstrap800Carry-lr50": "C4-I1I2-T10-G0-sgd-bootstrap800-compensated-hidden-carry-lr50",
    "MLP-V2206-C4-T10G0-OptTransport-lr50": "C4-I2-T10-G0-optimizer-state-transport-lr50",
    "MLP-V2206-C4-T10G0-OptTransport-lr100": "C4-I2-T10-G0-optimizer-state-transport-lr100",
    "MLP-V2206-C4-T10G0-OptTransport-lr150": "C4-I2-T10-G0-optimizer-state-transport-lr150",
    "MLP-V2206-C4-T10G0-OptTransport-lr300": "C4-I2-T10-G0-optimizer-state-transport-lr300",
    "MLP-V2206-C4-T10G0-EarlyWarm400ThenOptTransport-lr150": "C4-I1I2-T10-G0-early-warm400-then-optimizer-state-transport-lr150",
    "MLP-V2206-C4-T10G0-EarlyWarm800ThenOptTransport-lr150": "C4-I1I2-T10-G0-early-warm800-then-optimizer-state-transport-lr150",
    "MLP-V2206-C4-T10G0-AdamWBootstrap800OptTransport-lr150": "C4-I1I2-T10-G0-adamw-bootstrap800-optimizer-transport-lr150",
    "MLP-V2206-S2-T11G0-EarlyObservableSolver": "S2-T11-G0-early-observable-source-channel",
    "MLP-V2206-C4-T11G0-EarlyWarmCarry-stop800-lr50": "C4-I1-T11-G0-early-observable-warm-carry-stop800-lr50",
    "MLP-V2206-C4-T11G0-OptTransport-lr150": "C4-I2-T11-G0-early-observable-optimizer-transport-lr150",
    "MLP-V2206-C4-T11G0-ControlRelativeGate-stop800-lr50": "C4-I4-T11-G0-train-split-control-relative-gate-stop800-lr50",
    "MLP-V2206-C4-T11G0-AdamWControlGate-stop800-lr50": "C4-I4-T11-G0-adamw-relative-gate-stop800-lr50",
    "MLP-V2206-S2-T12G0-SoftCompensatedHiddenBlockSolver": "S2-T12-G0-soft-compensated-hidden-readout-block",
    "MLP-V2206-C4-T12G0-SourceStateCarry": "C4-I2-T12-G0-soft-compensated-hidden-block-source-state-carry",
    "MLP-V2206-C4-T12G0-OptTransport-lr150": "C4-I2-T12-G0-soft-compensated-hidden-block-optimizer-transport-lr150",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", required=True)
    p.add_argument("--label", default="v22.06 metric solver FU")
    return p


def _hmean(trace_group: list[dict[str, Any]], matrix_group: list[dict[str, Any]], key: str, horizon: int) -> float:
    vals = [finite_float(r.get(key)) for r in trace_group if int(finite_float(r.get("step"), -1.0)) == horizon]
    vals = [v for v in vals if v == v]
    if vals:
        return sum(vals) / len(vals)
    wide = mean(matrix_group, f"{key}_h{horizon}")
    if wide == wide:
        return wide
    return mean(matrix_group, key)


def _hfirst(trace_group: list[dict[str, Any]], matrix_group: list[dict[str, Any]], key: str, horizon: int) -> str:
    for r in trace_group:
        if int(finite_float(r.get("step"), -1.0)) == horizon and str(r.get(key, "")).strip():
            return str(r.get(key, ""))
    for r in matrix_group:
        value = r.get(f"{key}_h{horizon}", r.get(key, ""))
        if str(value).strip():
            return str(value)
    return ""


def _rows(summary: list[dict[str, Any]], matrix: list[dict[str, Any]], traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    matrix_by_id = {mid: [r for r in matrix if str(r.get("v21_id", "")) == mid] for mid in METRIC_SOLVER_IDS}
    traces_by_id = {mid: [r for r in traces if str(r.get("v21_id", "")) == mid] for mid in METRIC_SOLVER_IDS}
    for row in summary:
        vid = str(row.get("v21_id", ""))
        if vid not in METRIC_SOLVER_IDS:
            continue
        group = matrix_by_id.get(vid, [])
        tgroup = traces_by_id.get(vid, [])
        item: dict[str, Any] = {
            **row,
            "metric_solver_name": METRIC_SOLVER_IDS[vid],
            "solver_status": _hfirst(tgroup, group, "solver_status", 100),
            "solver_level": _hfirst(tgroup, group, "solver_level", 100),
            "target_family": _hfirst(tgroup, group, "target_family", 100),
            "metric_family": _hfirst(tgroup, group, "metric_family", 100),
            "ActuationR2_mean": _hmean(tgroup, group, "ActuationR2", 100),
            "projection_residual_Gf_mean": _hmean(tgroup, group, "projection_residual_Gf", 100),
            "B1_gain_mean": _hmean(tgroup, group, "B1_gain", 100),
            "B2_transfer_gain_mean": _hmean(tgroup, group, "B2_transfer_gain", 100),
            "B3_safety_gain_mean": _hmean(tgroup, group, "B3_safety_gain", 100),
            "solve_time_ms_mean": _hmean(tgroup, group, "solve_time_ms", 100),
            "source_to_reservoir_leakage_mean": _hmean(tgroup, group, "source_to_reservoir_leakage", 100),
            "source_h100_mean": row.get("source_h100_mean", ""),
            "source_h400_mean": row.get("source_h400_mean", ""),
            "source_h800_mean": row.get("source_h800_mean", ""),
            "source_h1600_mean": row.get("source_h1600_mean", ""),
            "source_h3200_mean": row.get("source_h3200_mean", ""),
            "row_h800_positive_count": row.get("source_h800_pass_count", ""),
            "row_h3200_positive_count": row.get("source_h3200_pass_count", ""),
            "R1600_over_800": row.get("retention_h1600_over_h800", ""),
            "R3200_over_1600": row.get("retention_h3200_over_h1600", ""),
            "early_source_warm_writer_active_mean": _hmean(tgroup, group, "early_source_warm_writer_active", 800),
            "early_source_warm_writer_lr_mean": _hmean(tgroup, group, "early_source_warm_writer_lr", 800),
            "source_state_carry_active_mean": _hmean(tgroup, group, "source_state_carry_active", 800),
            "direct_solver_commit_active_mean": _hmean(tgroup, group, "direct_solver_commit_active", 800),
            "periodic_solver_active_mean": _hmean(tgroup, group, "periodic_solver_active", 800),
            "sgd_bootstrap_writer_active_mean": _hmean(tgroup, group, "sgd_bootstrap_writer_active", 400),
            "adamw_bootstrap_writer_active_mean": _hmean(tgroup, group, "adamw_bootstrap_writer_active", 400),
            "adamw_bootstrap_writer_lr_mean": _hmean(tgroup, group, "adamw_bootstrap_writer_lr", 400),
            "optimizer_transport_active_mean": _hmean(tgroup, group, "optimizer_transport_active", 800),
            "optimizer_transport_strength_mean": _hmean(tgroup, group, "optimizer_transport_strength", 800),
            "optimizer_transport_projection_before_mean": _hmean(tgroup, group, "optimizer_transport_projection_before", 800),
            "optimizer_transport_projection_after_mean": _hmean(tgroup, group, "optimizer_transport_projection_after", 800),
            "optimizer_transport_removed_anti_source_mean": _hmean(tgroup, group, "optimizer_transport_removed_anti_source", 800),
            "optimizer_transport_grad_delta_norm_mean": _hmean(tgroup, group, "optimizer_transport_grad_delta_norm", 800),
            "train_split_control_gate_active_mean": _hmean(tgroup, group, "train_split_control_gate_active", 800),
            "train_split_control_gate_accept_mean": _hmean(tgroup, group, "train_split_control_gate_accept", 800),
            "train_split_control_gate_fu_signal_gain_mean": _hmean(tgroup, group, "train_split_control_gate_fu_signal_gain", 800),
            "train_split_control_gate_sgd_signal_gain_mean": _hmean(tgroup, group, "train_split_control_gate_sgd_signal_gain", 800),
            "train_split_control_gate_signal_margin_mean": _hmean(tgroup, group, "train_split_control_gate_signal_margin", 800),
            "adamw_control_gate_active_mean": _hmean(tgroup, group, "adamw_control_gate_active", 800),
            "adamw_control_gate_accept_mean": _hmean(tgroup, group, "adamw_control_gate_accept", 800),
            "adamw_control_gate_fu_signal_gain_mean": _hmean(tgroup, group, "adamw_control_gate_fu_signal_gain", 800),
            "adamw_control_gate_adamw_signal_gain_mean": _hmean(tgroup, group, "adamw_control_gate_adamw_signal_gain", 800),
            "adamw_control_gate_signal_margin_mean": _hmean(tgroup, group, "adamw_control_gate_signal_margin", 800),
            "adamw_control_gate_fu_b3_gain_mean": _hmean(tgroup, group, "adamw_control_gate_fu_b3_gain", 800),
            "adamw_control_gate_adamw_b3_gain_mean": _hmean(tgroup, group, "adamw_control_gate_adamw_b3_gain", 800),
            "adamw_control_gate_fu_corrupt_gain_mean": _hmean(tgroup, group, "adamw_control_gate_fu_corrupt_gain", 800),
            "adamw_control_gate_adamw_corrupt_gain_mean": _hmean(tgroup, group, "adamw_control_gate_adamw_corrupt_gain", 800),
            "block_source_hidden_residual_fraction_mean": _hmean(tgroup, group, "block_source_hidden_residual_fraction", 100),
            "block_source_hidden_function_cap_ratio_mean": _hmean(tgroup, group, "block_source_hidden_function_cap_ratio", 100),
            "block_source_hidden_compensation_norm_mean": _hmean(tgroup, group, "block_source_hidden_compensation_norm", 100),
            "block_source_hidden_compensation_scale_mean": _hmean(tgroup, group, "block_source_hidden_compensation_scale", 100),
            "block_source_hidden_compensation_mix_mean": _hmean(tgroup, group, "block_source_hidden_compensation_mix", 100),
            "block_source_hidden_compensation_residual_ratio_mean": _hmean(tgroup, group, "block_source_hidden_compensation_residual_ratio", 100),
            "target_early_observable_class_density_mean": _hmean(tgroup, group, "target_early_observable_class_density", 100),
            "target_early_observable_class_cos_mean": _hmean(tgroup, group, "target_early_observable_class_cos_mean", 100),
            "target_early_observable_mid_debt_mean": _hmean(tgroup, group, "target_early_observable_mid_debt_mean", 100),
            "target_early_observable_reference_agreement_mean": _hmean(tgroup, group, "target_early_observable_reference_agreement", 100),
        }
        item["C1_observability_pass"] = int(
            finite_float(item.get("B2_transfer_gain_mean"), -999.0) > 0.005
            and finite_float(item.get("B3_safety_gain_mean"), -999.0) > -0.001
            and finite_float(item.get("source_to_reservoir_leakage_mean"), 1.0) <= 0.35
        )
        item["C3_actuation_pass"] = int(
            finite_float(item.get("ActuationR2_mean"), -999.0) >= 0.50
            and finite_float(item.get("projection_residual_Gf_mean"), 999.0) <= 0.80
            and finite_float(item.get("B2_transfer_gain_mean"), -999.0) > 0.005
        )
        item["C4_early_source_pass"] = int(
            finite_float(item.get("source_h100_mean"), -999.0) >= 0.005
            and finite_float(item.get("source_h400_mean"), -999.0) >= 0.005
            and finite_float(item.get("source_h800_mean"), -999.0) >= 0.005
            and int(float(item.get("row_h800_positive_count") or 0)) >= max(1, int(float(item.get("rows") or 0)) * 2 // 3)
        )
        item["C4_h3200_source_pass"] = int(
            item["C4_early_source_pass"]
            and finite_float(item.get("source_h1600_mean"), -999.0) >= 0.005
            and finite_float(item.get("source_h3200_mean"), -999.0) >= 0.005
            and finite_float(item.get("R3200_over_1600"), 0.0) >= 0.50
        )
        if not item["C1_observability_pass"]:
            item["failure_taxonomy"] = "SourceObservableMissing"
        elif not item["C3_actuation_pass"]:
            item["failure_taxonomy"] = "SolverBlocked"
        elif not item["C4_early_source_pass"]:
            item["failure_taxonomy"] = "SourceFormationFailed"
        elif not item["C4_h3200_source_pass"]:
            item["failure_taxonomy"] = "OptimizerErosionOrDatasetLocalizedErosion"
        else:
            item["failure_taxonomy"] = ""
        out.append(item)
    return out


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir)
    summary = read_rows(source_dir / "v21_01_mlp_source_retention_summary.csv") or read_rows(source_dir / "v21_01_source_retention_summary.csv")
    matrix = read_rows(source_dir / "v21_01_source_retention_matrix.csv")
    traces = read_rows(source_dir / "v21_01_source_retention_raw_traces.csv")
    rows = _rows(summary, matrix, traces)
    best = (
        sorted(
            rows,
            key=lambda r: (
                int_flag(r.get("C4_h3200_source_pass")),
                int_flag(r.get("C4_early_source_pass")),
                int_flag(r.get("C3_actuation_pass")),
                int_flag(r.get("C1_observability_pass")),
                finite_float(r.get("source_h3200_mean"), -999.0),
                finite_float(r.get("source_h800_mean"), -999.0),
            ),
            reverse=True,
        )[0]
        if rows
        else {}
    )
    route = {
        "metric_solver_rows": len(rows),
        "C1_pass_rows": sum(int_flag(r.get("C1_observability_pass")) for r in rows),
        "C3_pass_rows": sum(int_flag(r.get("C3_actuation_pass")) for r in rows),
        "C4_early_source_pass_rows": sum(int_flag(r.get("C4_early_source_pass")) for r in rows),
        "C4_h3200_source_pass_rows": sum(int_flag(r.get("C4_h3200_source_pass")) for r in rows),
        "best_metric_v21_id": best.get("v21_id", ""),
        "best_metric_solver_name": best.get("metric_solver_name", ""),
        "best_source_h800": best.get("source_h800_mean", ""),
        "best_source_h3200": best.get("source_h3200_mean", ""),
        "functional_route": "C4-H3200SourceOpened" if any(int_flag(r.get("C4_h3200_source_pass")) for r in rows) else ("C3-ActuationOnly" if any(int_flag(r.get("C3_actuation_pass")) for r in rows) else "F0-MetricNoEffect"),
        "promotion_allowed": 0,
    }
    write_rows(out_dir / "v22_06_metric_solver_summary.csv", rows)
    write_rows(out_dir / "v22_06_metric_solver_raw_matrix.csv", matrix)
    write_json(out_dir / "v22_06_metric_solver_route.json", route)
    simple_svg(out_dir / "figures/projection_residual_vs_B2_transfer.svg", "v22.06 projection residual vs B2 transfer", rows, "projection_residual_Gf_mean")
    simple_svg(out_dir / "figures/source_trajectory_h100_to_h6400.svg", "v22.06 source trajectory", rows, "source_h3200_mean")
    simple_svg(out_dir / "figures/metric_energy_vs_h4800_retention.svg", "v22.06 metric energy vs source", rows, "ActuationR2_mean")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_06_metric_solver_fu.py --source-dir {source_dir} --out-dir {out_dir}",
        status="completed",
        note=f"rows={len(rows)} route={route['functional_route']} best={route['best_metric_v21_id']}",
    )


if __name__ == "__main__":
    main()
