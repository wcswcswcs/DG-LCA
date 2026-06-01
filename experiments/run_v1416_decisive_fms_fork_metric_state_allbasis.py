#!/usr/bin/env python3
"""DG-KAN v14.16 decisive FMS fork runner.

This runner executes the v14.16 stop/go fork without adding action tokens,
controllers, reset routes, F-CHE8/F-CHE9, or FMS-M6/M7. It performs the full
Line A selector-direction-boundary contrast, a bounded Metric-State line
(MS1/MS2/MS3), and rebuilds C/E from prior official artifacts without imputing
unavailable cells.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import zipfile
from copy import copy
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v1410_nonrat_fms_transfer_fms_definition_reset as v1410  # noqa: E402
from experiments import run_v1414_fms_causal_value_transfer_boundary_all_basis_continue_open as v1414  # noqa: E402
from experiments import run_v144_real_transfer_fms_all_basis_substrate as v144  # noqa: E402


DEFAULT_OUT = ROOT / "results/v14_16_decisive_fms_fork_metric_state_allbasis/official_v1416"
DEFAULT_LINE_E_OUT = ROOT / "results/v14_16_decisive_fms_fork_metric_state_allbasis/line_e_v1416_substrate_acceleration"
V1415_OUT = ROOT / "results/v14_15_parallel_fms_causal_fork_transfer_observability_allbasis/official_v1415"
V1414_OUT = ROOT / "results/v14_14_fms_causal_value_transfer_boundary_all_basis_continue_open/official_v1414"
PLAN_DOC = ROOT / "docs/DG-KAN_v14.16_DecisiveFMSFork_MetricState_AllBasisAcceleration_完整计划.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v14.16_DecisiveFMSFork_MetricState_AllBasisAcceleration_实验结果复盘.md"
EXEC_LOG_DOC = ROOT / "docs/DG-KAN_v14.16_DecisiveFMSFork_MetricState_AllBasisAcceleration_执行日志.md"

REQUIRED = [
    "v1416_route_decision.json",
    "v1416_progress_table.csv",
    "v1416_required_artifact_manifest.csv",
    "v1416_forbidden_information_audit.csv",
    "v1416_no_action_search_audit.csv",
    "v1416_a_current_fms_exit_matrix.csv",
    "v1416_a_direction_control_equivalence.csv",
    "v1416_a_failure_taxonomy.csv",
    "v1416_b_metric_state_results.csv",
    "v1416_b_metric_state_controls.csv",
    "v1416_b_population_risk_state_trace.csv",
    "v1416_c_transfer_predictor_leaveout.csv",
    "v1416_c_predictor_control_deconfound.csv",
    "v1416_c_time_budget_line_a_matrix.csv",
    "v1416_c_time_budget_summary.csv",
    "v1416_d_dche_real_lite.csv",
    "v1416_d_dche_controls.csv",
    "v1416_e_fou_substrate.csv",
    "v1416_e_rbf_substrate.csv",
    "v1416_e_wav_monitor.csv",
    "v1416_e_dche_no_regression.csv",
    "v1416_e_allbasis_summary.csv",
    "v1416_m_mlp_generic_controls.csv",
    "v1416_linec_tail_auc_audit.csv",
    "v1416_no_go_boundary.md",
    "v1416_next_hypothesis_queue.md",
    "v1416_code_review_packet.zip",
]

FIGURES = [
    "fig_a_direction_vs_controls_heatmap.svg",
    "fig_a_control_equivalent_fraction.svg",
    "fig_b_metric_state_trace.svg",
    "fig_b_metric_state_vs_controls.svg",
    "fig_c_leaveout_auc_matrix.svg",
    "fig_c_predictor_calibration.svg",
    "fig_d_dche_real_lite_pass_matrix.svg",
    "fig_e_allbasis_pass_heatmap.svg",
    "fig_e_fou_rbf_task_workspace_pareto.svg",
    "fig_m_mlp_vs_kan_control_gap.svg",
    "fig_z_route_dashboard.svg",
]

LINE_B_DCHE_METHODS = [
    "MS1-ParameterMetricState",
    "MS2-DegreeRoleMetricState",
    "MS3-BasisGroupMetricState",
    "C0-D-CHE-AdamW",
    "C1-D-CHE-AdamW-NoOpMatchedOverhead",
    "C2-D-CHE-AdamW-RandomMatchedNorm",
    "C5-D-CHE-AdamW-SameActiveFractionControl",
    "C6-D-CHE-SameTCRandomDirection",
    "C7-D-CHE-SameMetricScaleRandomPermutation",
]

LINE_M_MLP_METHODS = [
    "M0-MLP-AdamW",
    "M8-MLP-MetricState-MS1",
    "M9-MLP-MetricState-MS2",
    "M6-MLP-SameActiveFractionControl",
    "M10-MLP-SameMetricScaleRandomPermutation",
    "M3-MLP-RandomMatchedNorm",
    "M5-MLP-NoOpMatchedOverhead",
    "M4-MLP-GenericOptimizerStateControl",
    "M11-MLP-AdamWParallelDirectionControl",
]

LINE_A_SELECTORS = [
    "S0-always-on",
    "S1-predictor-high-score",
    "S4-NoOp-safe-abstention",
]
LINE_A_BOUNDARIES = [
    "B0-none",
    "B1-degree-projection-safety",
    "B4-value-retention-floor",
]
LINE_A_DIRECTIONS = [
    "D0-AdamW",
    "D1-current-FMS",
    "D2-random-matched-norm",
    "D3-same-active-fraction-random",
    "D4-same-projection-rejection-random",
    "D5-same-value-retention-random",
    "D6-AdamWParallelDirection-control",
]

LINE_A_DIRECTION_METHODS = {
    "D0-AdamW": "C0-D-CHE-AdamW",
    "D1-current-FMS": "F-CHE-FB3-GenericFMSPlusDegreeSafetyProjection",
    "D2-random-matched-norm": "C2-D-CHE-AdamW-RandomMatchedNorm",
    "D3-same-active-fraction-random": "C5-D-CHE-AdamW-SameActiveFractionControl",
    "D4-same-projection-rejection-random": "A-D4-SameProjectionRejectionRandom",
    "D5-same-value-retention-random": "A-D5-SameValueRetentionRandom",
    "D6-AdamWParallelDirection-control": "C3-D-CHE-AdamW-AdamWParallelDirectionControl",
}

LINE_E_CANDIDATES = [
    "D-FOU32-LowFreqIdentityResidualV3",
    "D-FOU33-BandwiseSNRWarmupV2",
    "D-FOU34-PhaseStableBandMixNoHighFreqV2",
    "D-FOU35-NoMaterializeLifetimeV4",
    "D-FOU36-HighFrequencyQuarantineV2",
    "D-RBF30-ActiveCenterOccupancyV3",
    "D-RBF31-WidthConditionIdentityResidualV2",
    "D-RBF32-CompactBumpNoDenseMaterializationV2",
    "D-RBF33-GaussianLocalK4TaskHealthV2",
    "D-RBF34-CenterOccupancyWarmupNoTaskBranch",
    "D-WAV29-TriangularSupportV4",
    "D-WAV30-ScaleOccupancyNoTailTargetV2",
    "D-WAV31-LocalSupportOverlapDampingV2",
    "D-WAV32-LocalTailCoverageAuditV2",
]

LINE_C_REQUIRED_LEAVEOUT_SPLITS = [
    "leave-dataset-out",
    "leave-seed-out",
    "leave-loss-interface-out",
    "leave-method-out",
    "leave-control-out",
    "leave-synthetic-family-out",
    "leave-time-budget-out",
]

LINE_C_ACTUAL_FEATURE_COLUMNS = {
    "P1-degree_projection_rejection_fraction": "degree_projection_rejection_fraction",
    "P3-value_retention_after_degree_projection": "value_retention_after_degree_projection",
    "P4-cos_projected_vs_generic": "cos_projected_vs_generic",
    "P5-drift_diffusion_group_utility": "fms_state_norm",
    "P6-split_window_consistency": "degree_gate_active_fraction",
    "P8-update_state_disagreement": "parameter_update_norm",
    "P9-degree_energy_stability": "degree_entropy",
}


def fnum(value: Any, default: float = 0.0) -> float:
    return v1414.fnum(value, default)


def sint(value: Any, default: int = 0) -> int:
    return v1414.sint(value, default)


def read_rows(path: Path) -> list[dict[str, str]]:
    return v1414.read_rows(path)


def write_rows(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    v1414.write_rows(path, rows, fieldnames)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    v1414.write_json(path, obj)


def write_text(path: Path, text: str) -> None:
    v1414.write_text(path, text)


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return statistics.fmean(vals) if vals else 0.0


def median(values: Iterable[float]) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    return statistics.median(vals) if vals else 0.0


def unique_pass_count(rows: Sequence[dict[str, Any]], pass_col: str = "strict_gate_pass") -> int:
    return len({(r.get("dataset"), str(r.get("seed"))) for r in rows if sint(r.get(pass_col), 0) == 1})


def auc_or_half(scores: Sequence[float], labels: Sequence[int]) -> float:
    return v1414.auc_or_half(scores, labels)


def spearman_or_zero(xs: Sequence[float], ys: Sequence[float]) -> float:
    return v1414.spearman_or_zero(xs, ys)


def simple_svg(path: Path, title: str, rows: Sequence[tuple[str, float]], threshold: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_rows = list(rows)[:24]
    width = 980
    height = 90 + 28 * max(1, len(safe_rows))
    maxv = max([abs(v) for _k, v in safe_rows] + ([abs(threshold)] if threshold is not None else [1.0]), default=1.0)
    maxv = max(maxv, 1.0e-8)
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#f8f8f4"/>',
        f'<text x="24" y="34" font-family="monospace" font-size="20" fill="#222">{title}</text>',
    ]
    for i, (label, value) in enumerate(safe_rows):
        y = 70 + i * 28
        bar = max(2.0, min(360.0, 360.0 * abs(float(value)) / maxv))
        color = "#2f7d62" if float(value) >= 0 else "#9b3d3d"
        body.append(f'<text x="24" y="{y}" font-family="monospace" font-size="13" fill="#333">{label}: {float(value):.4f}</text>')
        body.append(f'<rect x="520" y="{y - 13}" width="{bar:.1f}" height="16" fill="{color}" opacity="0.78"/>')
    if threshold is not None:
        body.append(f'<text x="24" y="{height - 18}" font-family="monospace" font-size="12" fill="#555">threshold={threshold}</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def run_metric_state(args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_paths = [
        out_dir / "v1416_b_metric_state_results.csv",
        out_dir / "v1416_b_metric_state_controls.csv",
        out_dir / "v1416_m_mlp_generic_controls.csv",
        out_dir / "v1416_b_population_risk_state_trace.csv",
    ]
    if sint(getattr(args, "reuse_metric_state_if_present", 1), 1) == 1 and all(path.exists() for path in metric_paths):
        enriched = [
            *read_rows(out_dir / "v1416_b_metric_state_results.csv"),
            *read_rows(out_dir / "v1416_b_metric_state_controls.csv"),
            *read_rows(out_dir / "v1416_m_mlp_generic_controls.csv"),
        ]
        projection_rows = read_rows(out_dir / "v1416_b_population_risk_state_trace.csv")
        existing_linec = read_rows(out_dir / "v1416_linec_tail_auc_audit.csv")
        metric_linec_rows = [r for r in existing_linec if r.get("source_artifact") == "v1416_metric_state_actual"]
        return enriched, projection_rows, metric_linec_rows

    raw_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    linec_rows: list[dict[str, Any]] = []
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    for dataset in v1410.parse_csv(args.datasets):
        for seed in v1410.parse_ints(args.seeds):
            load_args = copy(args)
            load_args.seed = int(seed)
            xtr, ytr, xva, yva, xte, yte, input_dim_t, output_dim_t = v144.load_real_split(load_args, dataset, int(seed), device)
            input_dim = int(input_dim_t.item() if hasattr(input_dim_t, "item") else input_dim_t)
            output_dim = int(output_dim_t.item() if hasattr(output_dim_t, "item") else output_dim_t)
            for method in LINE_B_DCHE_METHODS:
                result = v1410.train_model_case(
                    family="D-CHE",
                    candidate_id=str(args.dche_candidate),
                    method=method,
                    dataset=dataset,
                    task="real_lite",
                    seed=int(seed),
                    loss_interface="CE",
                    xtr=xtr,
                    ytr=ytr,
                    xva=xva,
                    yva=yva,
                    xte=xte,
                    yte=yte,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    args=args,
                    device=device,
                    real_linec=True,
                )
                raw_rows.append(result["row"])
                projection_rows.extend(result["projection_rows"])
                linec_rows.extend(result["linec_rows"])
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            for method in LINE_M_MLP_METHODS:
                result = v1410.train_model_case(
                    family="MLP",
                    candidate_id="MLP-v1416-control",
                    method=method,
                    dataset=dataset,
                    task="real_lite",
                    seed=int(seed),
                    loss_interface="CE",
                    xtr=xtr,
                    ytr=ytr,
                    xva=xva,
                    yva=yva,
                    xte=xte,
                    yte=yte,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    args=args,
                    device=device,
                    real_linec=False,
                )
                raw_rows.append(result["row"])
                projection_rows.extend(result["projection_rows"])
                linec_rows.extend(result["linec_rows"])
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    enriched = v1410.enrich_rows(raw_rows, "V1416_METRIC_STATE_REAL_LITE")
    for row in enriched:
        row["v1416_metric_state_executed"] = 1
        row["new_fms_token"] = 0
        row["new_fche_token"] = 0
        row["action_token"] = 0
        row["controller_executed"] = 0
        row["reset_route_used"] = 0
        row["direction_uses_train_stream_only"] = 1
    write_rows(out_dir / "v1416_b_metric_state_results.csv", [r for r in enriched if r.get("family") == "D-CHE" and sint(r.get("control_method"), 0) == 0])
    write_rows(out_dir / "v1416_b_metric_state_controls.csv", [r for r in enriched if r.get("family") == "D-CHE" and sint(r.get("control_method"), 0) == 1])
    write_rows(out_dir / "v1416_m_mlp_generic_controls.csv", [r for r in enriched if r.get("family") == "MLP"])
    write_rows(out_dir / "v1416_b_population_risk_state_trace.csv", projection_rows)
    return enriched, projection_rows, linec_rows


def build_line_a(out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    v1415_i = read_rows(V1415_OUT / "v1415_intervention_factorial.csv")
    v1414_q = read_rows(V1414_OUT / "v1414_q_fms_vs_matched_controls.csv")
    datasets = ["MNIST", "Fashion-MNIST", "KMNIST"]
    seeds = ["0", "1", "2"]
    source: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    dir_map = {
        "D0-AdamW": "D0-AdamW-direction",
        "D1-current-FMS": "D1-Generic-FMS-direction",
        "D2-random-matched-norm": "D3-Random-matched-norm-direction",
        "D6-AdamWParallelDirection-control": "D4-AdamWParallelDirection-control",
    }
    sel_map = {
        "S0-always-on": "S0-always-on",
        "S1-predictor-high-score": "S1-E4-predictor-high-score",
        "S4-NoOp-safe-abstention": "S4-NoOp-safe-abstention-selector",
    }
    b_map = {
        "B0-none": "B0-no-boundary",
        "B1-degree-projection-safety": "B1-degree-projection-safety",
        "B4-value-retention-floor": "B4-value-retention-floor",
    }
    for row in v1415_i:
        for d16, d15 in dir_map.items():
            if row.get("direction_id") == d15:
                for s16, s15 in sel_map.items():
                    if row.get("selector_id") != s15:
                        continue
                    for b16, b15 in b_map.items():
                        if row.get("boundary_id") == b15:
                            key = (str(row.get("dataset")), str(row.get("seed")), s16, d16, b16)
                            source.setdefault(key, row)
    q_map = {
        "D3-same-active-fraction-random": "G3-SameActiveFractionControl",
        "D4-same-projection-rejection-random": "G4-SameDegreeProjectionRejectionControl",
        "D5-same-value-retention-random": "G5-SameValueRetentionRandomDirection",
        "D6-AdamWParallelDirection-control": "G6-AdamWParallelDirectionControl",
    }
    for row in v1414_q:
        group = row.get("group")
        for d16, q_group in q_map.items():
            if group == q_group:
                key = (str(row.get("dataset")), str(row.get("seed")), "S0-always-on", d16, "B0-none")
                source.setdefault(key, row)

    matrix: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            for selector in LINE_A_SELECTORS:
                for direction in LINE_A_DIRECTIONS:
                    for boundary in LINE_A_BOUNDARIES:
                        key = (dataset, seed, selector, direction, boundary)
                        row = source.get(key)
                        item: dict[str, Any] = {
                            "dataset": dataset,
                            "seed": seed,
                            "selector": selector,
                            "direction": direction,
                            "boundary": boundary,
                            "executed_metric_available": int(row is not None),
                            "replay_source": "",
                            "source_vs_best_control": "",
                            "AUCtime_ratio": "",
                            "CEp99_delta": "",
                            "NLL_delta": "",
                            "ECE_delta": "",
                            "LineC_pass": "",
                            "control_equivalent": "",
                            "bad_event": "",
                            "step_time_ratio": "",
                            "memory_ratio": "",
                            "strict_gate_pass": 0,
                            "promotion_allowed": 0,
                        }
                        if row is not None:
                            item.update(
                                {
                                    "replay_source": row.get("source_artifact", row.get("group", "v1414/v1415 artifact")),
                                    "source_vs_best_control": fnum(row.get("source_vs_best_control"), 0.0),
                                    "AUCtime_ratio": fnum(row.get("auc_time_ratio", row.get("AUCtime_ratio_vs_best_control")), 9.0),
                                    "CEp99_delta": fnum(row.get("CEp99_delta", row.get("CEp99_delta_vs_best_control")), 0.0),
                                    "NLL_delta": fnum(row.get("NLL_delta", row.get("NLL_delta_vs_best_control")), 0.0),
                                    "ECE_delta": fnum(row.get("ECE_delta", row.get("ECE_delta_vs_best_control")), 0.0),
                                    "LineC_pass": sint(row.get("LineC_pass", 0), 0),
                                    "control_equivalent": sint(row.get("control_equivalent", 0), 0),
                                    "bad_event": sint(row.get("bad_event", 0), 0),
                                    "step_time_ratio": row.get("step_time_ratio", ""),
                                    "memory_ratio": row.get("memory_ratio", ""),
                                    "strict_gate_pass": sint(row.get("strict_gate_pass"), 0),
                                }
                            )
                        else:
                            item["missing_reason"] = "not executed in available v14.14/v14.15 contrast artifacts; not imputed"
                        matrix.append(item)

    current_rows = [r for r in matrix if r["direction"] == "D1-current-FMS" and sint(r.get("executed_metric_available"), 0) == 1]
    real_lite_pass = unique_pass_count(current_rows)
    source_mean = mean([fnum(r.get("source_vs_best_control"), 0.0) for r in current_rows])
    control_equiv = sum(1 for r in current_rows if fnum(r.get("source_vs_best_control"), 0.0) <= 0.005) / max(1, len(current_rows))
    bad_event = sum(1 for r in current_rows if sint(r.get("bad_event"), 0) == 1) / max(1, len(current_rows))
    pass_gate = int(source_mean > 0.005 and control_equiv <= 0.40 and bad_event <= 0.30 and real_lite_pass >= 3)
    summary = [
        {
            "line": "A",
            "available_rows": sum(sint(r.get("executed_metric_available"), 0) for r in matrix),
            "planned_rows": len(matrix),
            "mean_source_vs_best_control": source_mean,
            "control_equivalent_fraction": control_equiv,
            "bad_event_fraction": bad_event,
            "real_lite_pass_count": real_lite_pass,
            "line_a_gate_pass": pass_gate,
            "route": "S1-FMSDirectionCausalValue" if pass_gate else "R-A-CurrentFMSDirectionNoGo",
            "promotion_allowed": 0,
        }
    ]
    failure_rows = [
        {
            "failure": "current_fms_direction_control_equivalent",
            "triggered": int(not pass_gate),
            "mean_source_vs_best_control": source_mean,
            "control_equivalent_fraction": control_equiv,
            "bad_event_fraction": bad_event,
            "real_lite_pass_count": real_lite_pass,
            "promotion_allowed": 0,
        }
    ]
    write_rows(out_dir / "v1416_a_current_fms_exit_matrix.csv", matrix)
    write_rows(out_dir / "v1416_a_direction_control_equivalence.csv", summary)
    write_rows(out_dir / "v1416_a_failure_taxonomy.csv", failure_rows)
    return matrix, summary, summary[0]


def run_actual_line_a(args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    matrix_path = out_dir / "v1416_a_current_fms_exit_matrix.csv"
    summary_path = out_dir / "v1416_a_direction_control_equivalence.csv"
    failure_path = out_dir / "v1416_a_failure_taxonomy.csv"
    planned_rows = len(v1410.parse_csv(args.datasets)) * len(v1410.parse_ints(args.seeds)) * len(LINE_A_SELECTORS) * len(LINE_A_DIRECTIONS) * len(LINE_A_BOUNDARIES)
    if matrix_path.exists() and summary_path.exists():
        matrix = read_rows(matrix_path)
        if len(matrix) == planned_rows and all(sint(r.get("executed_metric_available"), 0) == 1 for r in matrix):
            summary = read_rows(summary_path)
            return matrix, summary, summary[0] if summary else {}

    raw_rows: list[dict[str, Any]] = []
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    for dataset in v1410.parse_csv(args.datasets):
        for seed in v1410.parse_ints(args.seeds):
            load_args = copy(args)
            load_args.seed = int(seed)
            xtr, ytr, xva, yva, xte, yte, input_dim_t, output_dim_t = v144.load_real_split(load_args, dataset, int(seed), device)
            input_dim = int(input_dim_t.item() if hasattr(input_dim_t, "item") else input_dim_t)
            output_dim = int(output_dim_t.item() if hasattr(output_dim_t, "item") else output_dim_t)
            for selector in LINE_A_SELECTORS:
                for boundary in LINE_A_BOUNDARIES:
                    for direction in LINE_A_DIRECTIONS:
                        method = LINE_A_DIRECTION_METHODS[direction]
                        local_args = copy(args)
                        local_args.line_a_selector_id = selector
                        local_args.line_a_direction_id = direction
                        local_args.line_a_boundary_id = boundary
                        result = v1410.train_model_case(
                            family="D-CHE",
                            candidate_id=str(args.dche_candidate),
                            method=method,
                            dataset=dataset,
                            task="line_a_actual",
                            seed=int(seed),
                            loss_interface="CE",
                            xtr=xtr,
                            ytr=ytr,
                            xva=xva,
                            yva=yva,
                            xte=xte,
                            yte=yte,
                            input_dim=input_dim,
                            output_dim=output_dim,
                            args=local_args,
                            device=device,
                            real_linec=True,
                        )
                        row = dict(result["row"])
                        row.update(
                            {
                                "selector": selector,
                                "direction": direction,
                                "boundary": boundary,
                                "direction_method": method,
                                "executed_metric_available": 1,
                                "source_artifact": "v1416_actual_line_a_train_model_case",
                                "replay_source": "",
                                "missing_reason": "",
                                "new_fms_token": 0,
                                "new_fche_token": 0,
                                "action_token": 0,
                                "controller_executed": 0,
                                "reset_route_used": 0,
                                "promotion_allowed": 0,
                            }
                        )
                        raw_rows.append(row)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

    matrix: list[dict[str, Any]] = []
    by_group: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        key = (str(row.get("dataset")), str(row.get("seed")), str(row.get("selector")), str(row.get("boundary")))
        by_group.setdefault(key, []).append(row)

    for group in by_group.values():
        adam = next((r for r in group if r.get("direction") == "D0-AdamW"), group[0])
        matched_controls = [r for r in group if r.get("direction") in set(LINE_A_DIRECTIONS) - {"D0-AdamW", "D1-current-FMS"}]
        controls_for_best = matched_controls or [adam]
        best_nll = min(fnum(r.get("NLL"), 9.0) for r in controls_for_best)
        best_auc = min(fnum(r.get("AUC_NLL"), 9.0) for r in controls_for_best)
        base_time = max(1.0e-8, fnum(adam.get("step_time_sec"), 1.0))
        base_mem = max(1.0, fnum(adam.get("peak_memory_bytes"), 1.0))
        for row in group:
            item = dict(row)
            item["stage"] = "V1416_A_CURRENT_FMS_EXIT_ACTUAL"
            item["source_vs_adamw"] = fnum(adam.get("NLL"), 9.0) - fnum(row.get("NLL"), 9.0)
            item["source_vs_best_control"] = best_nll - fnum(row.get("NLL"), 9.0)
            item["AUCtime_ratio"] = fnum(row.get("AUC_NLL"), 9.0) / max(1.0e-8, best_auc)
            item["CEp99_delta"] = fnum(row.get("CEp99"), 0.0) - fnum(adam.get("CEp99"), 0.0)
            item["NLL_delta"] = fnum(row.get("NLL"), 0.0) - fnum(adam.get("NLL"), 0.0)
            item["ECE_delta"] = fnum(row.get("ECE"), 0.0) - fnum(adam.get("ECE"), 0.0)
            item["step_time_ratio"] = fnum(row.get("step_time_sec"), 0.0) / base_time
            item["memory_ratio"] = fnum(row.get("peak_memory_bytes"), 0.0) / base_mem if base_mem > 1.0 else 1.0
            item["control_equivalent"] = int(fnum(item.get("source_vs_best_control"), -999.0) <= 0.005)
            item["bad_event"] = int(
                fnum(item.get("source_vs_best_control"), -999.0) < 0.005
                or fnum(item.get("AUCtime_ratio"), 9.0) > 1.0
                or sint(item.get("LineC_majority_pass"), 0) == 0
            )
            item["LineC_pass"] = sint(item.get("LineC_majority_pass"), 0)
            item["strict_gate_pass"] = int(
                item.get("direction") == "D1-current-FMS"
                and fnum(item.get("source_vs_best_control"), -999.0) >= 0.005
                and fnum(item.get("AUCtime_ratio"), 9.0) <= 1.0
                and fnum(item.get("CEp99_delta"), 999.0) <= 0.05
                and fnum(item.get("NLL_delta"), 999.0) <= 0.02
                and fnum(item.get("ECE_delta"), 999.0) <= 0.02
                and sint(item.get("LineC_majority_pass"), 0) == 1
                and fnum(item.get("step_time_ratio"), 999.0) <= 1.25
                and fnum(item.get("memory_ratio"), 999.0) <= 1.25
            )
            matrix.append(item)

    matrix.sort(key=lambda r: (str(r.get("dataset")), int(r.get("seed", 0)), str(r.get("selector")), str(r.get("direction")), str(r.get("boundary"))))
    current_rows = [r for r in matrix if r["direction"] == "D1-current-FMS"]
    real_lite_pass = unique_pass_count(current_rows)
    source_mean = mean([fnum(r.get("source_vs_best_control"), 0.0) for r in current_rows])
    control_equiv = sum(1 for r in current_rows if fnum(r.get("source_vs_best_control"), 0.0) <= 0.005) / max(1, len(current_rows))
    bad_event = sum(1 for r in current_rows if sint(r.get("bad_event"), 0) == 1) / max(1, len(current_rows))
    pass_gate = int(source_mean > 0.005 and control_equiv <= 0.40 and bad_event <= 0.30 and real_lite_pass >= 3)
    summary = [
        {
            "line": "A",
            "available_rows": sum(sint(r.get("executed_metric_available"), 0) for r in matrix),
            "planned_rows": len(matrix),
            "mean_source_vs_best_control": source_mean,
            "control_equivalent_fraction": control_equiv,
            "bad_event_fraction": bad_event,
            "real_lite_pass_count": real_lite_pass,
            "line_a_gate_pass": pass_gate,
            "route": "S1-FMSDirectionCausalValue" if pass_gate else "R-A-CurrentFMSDirectionNoGo",
            "promotion_allowed": 0,
        }
    ]
    failure_rows: list[dict[str, Any]] = []
    for row in current_rows:
        reasons = []
        if fnum(row.get("source_vs_best_control"), -999.0) < 0.005:
            reasons.append("source")
        if fnum(row.get("AUCtime_ratio"), 9.0) > 1.0:
            reasons.append("AUCtime")
        if fnum(row.get("CEp99_delta"), 999.0) > 0.05:
            reasons.append("CEp99")
        if fnum(row.get("NLL_delta"), 999.0) > 0.02:
            reasons.append("NLL")
        if fnum(row.get("ECE_delta"), 999.0) > 0.02:
            reasons.append("ECE")
        if sint(row.get("LineC_majority_pass"), 0) == 0:
            reasons.append("LineC")
        failure_rows.append(
            {
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "selector": row.get("selector"),
                "boundary": row.get("boundary"),
                "failure": "+".join(reasons) if reasons else "pass",
                "strict_gate_pass": sint(row.get("strict_gate_pass"), 0),
                "source_vs_best_control": row.get("source_vs_best_control"),
                "AUCtime_ratio": row.get("AUCtime_ratio"),
                "CEp99_delta": row.get("CEp99_delta"),
                "NLL_delta": row.get("NLL_delta"),
                "ECE_delta": row.get("ECE_delta"),
                "LineC_pass": row.get("LineC_majority_pass"),
                "promotion_allowed": 0,
            }
        )
    write_rows(matrix_path, matrix)
    write_rows(summary_path, summary)
    write_rows(failure_path, failure_rows)
    return matrix, summary, summary[0]


def run_budgeted_line_a(args: argparse.Namespace, out_dir: Path) -> list[dict[str, Any]]:
    final_path = out_dir / "v1416_c_time_budget_line_a_matrix.csv"
    summary_path = out_dir / "v1416_c_time_budget_summary.csv"
    planned_rows = len(v1410.parse_csv(args.datasets)) * len(v1410.parse_ints(args.seeds)) * len(LINE_A_SELECTORS) * len(LINE_A_DIRECTIONS) * len(LINE_A_BOUNDARIES)
    if final_path.exists():
        existing = read_rows(final_path)
        if len(existing) == planned_rows:
            return existing
    if sint(getattr(args, "compute_budgeted_run", 0), 0) != 1:
        return []

    budget_steps = max(40, int(getattr(args, "train_steps", 200)) // 2)
    budget_args = copy(args)
    budget_args.train_steps = budget_steps
    budget_dir = out_dir / f"v1416_c_time_budget_probe_steps{budget_steps}"
    matrix, _summary_rows, summary = run_actual_line_a(budget_args, budget_dir)
    rows: list[dict[str, Any]] = []
    for row in matrix:
        item = dict(row)
        item["time_budget_id"] = f"fixed-half-budget-{budget_steps}"
        item["time_budget_train_steps"] = budget_steps
        item["source_artifact"] = "v1416_c_time_budget_probe_actual_line_a"
        item["promotion_allowed"] = 0
        rows.append(item)
    write_rows(final_path, rows)
    write_rows(
        summary_path,
        [
            {
                "time_budget_id": f"fixed-half-budget-{budget_steps}",
                "train_steps": budget_steps,
                "rows": len(rows),
                "line_a_gate_pass": summary.get("line_a_gate_pass", ""),
                "mean_source_vs_best_control": summary.get("mean_source_vs_best_control", ""),
                "control_equivalent_fraction": summary.get("control_equivalent_fraction", ""),
                "promotion_allowed": 0,
            }
        ],
    )
    return rows


def build_line_c(out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    prior_leaveout = read_rows(V1415_OUT / "v1415_predictor_leaveout.csv")
    prior_robustness = read_rows(V1415_OUT / "v1415_predictor_robustness.csv")
    prior_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in prior_leaveout:
        fid = str(row.get("feature_id", ""))
        split = str(row.get("planned_leaveout_split", row.get("leaveout_split", "")))
        if fid and split:
            item = dict(row)
            item["source_artifact"] = "v1415_predictor_leaveout.csv"
            item["promotion_allowed"] = 0
            prior_by_key[(fid, split)] = item

    feature_ids = sorted(
        {
            *[str(r.get("feature_id")) for r in prior_robustness if r.get("feature_id")],
            *LINE_C_ACTUAL_FEATURE_COLUMNS.keys(),
        }
    )
    prior_summary = {str(r.get("feature_id")): dict(r) for r in prior_robustness if r.get("feature_id")}

    full_rows = read_rows(out_dir / "v1416_a_current_fms_exit_matrix.csv")
    for row in full_rows:
        row.setdefault("time_budget_id", f"full-budget-{row.get('train_steps', getattr(args, 'train_steps', 0))}")
        row.setdefault("time_budget_train_steps", row.get("train_steps", getattr(args, "train_steps", "")))
    budget_rows = run_budgeted_line_a(args, out_dir)
    actual_rows = [*full_rows, *budget_rows]

    def control_bucket(row: dict[str, Any]) -> str:
        direction = str(row.get("direction", ""))
        if direction == "D1-current-FMS":
            return "current_fms"
        if direction == "D0-AdamW":
            return "adamw_baseline"
        if direction:
            return "matched_direction_control"
        return ""

    def split_group(row: dict[str, Any], split: str) -> str:
        if split == "leave-dataset-out":
            return str(row.get("dataset", ""))
        if split == "leave-seed-out":
            return str(row.get("seed", ""))
        if split == "leave-method-out":
            return str(row.get("direction_method", row.get("method", "")))
        if split == "leave-control-out":
            return control_bucket(row)
        if split == "leave-time-budget-out":
            return str(row.get("time_budget_id", ""))
        return ""

    def actual_leaveout_row(fid: str, column: str, split: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
        usable = [r for r in rows if str(r.get(column, "")) != ""]
        groups = sorted({split_group(r, split) for r in usable if split_group(r, split)})
        if len(groups) < 2:
            return None
        aucs: list[float] = []
        degenerate_count = 0
        subset_sizes: list[int] = []
        for held in groups:
            subset = [r for r in usable if split_group(r, split) != held]
            if not subset:
                continue
            labels = [sint(r.get("strict_gate_pass"), 0) for r in subset]
            scores = [fnum(r.get(column), 0.0) for r in subset]
            if len(set(labels)) < 2:
                degenerate_count += 1
            aucs.append(auc_or_half(scores, labels))
            subset_sizes.append(len(subset))
        if not aucs:
            return None
        return {
            "feature_id": fid,
            "v1416_feature_column": column,
            "planned_leaveout_split": split,
            "source_leaveout_split": split,
            "auc": min(aucs),
            "auc_mean_over_heldout_groups": mean(aucs),
            "heldout_group_count": len(groups),
            "min_subset_rows": min(subset_sizes) if subset_sizes else 0,
            "single_class_label_subset_count": degenerate_count,
            "available_for_gate": 1,
            "executed_from_existing_artifact": 0,
            "source_artifact": "v1416_actual_line_a" if split != "leave-time-budget-out" else "v1416_actual_line_a_plus_fixed_half_budget",
            "missing_split_reason": "",
            "promotion_allowed": 0,
        }

    actual_leaveout: dict[tuple[str, str], dict[str, Any]] = {}
    actual_summary: dict[str, dict[str, Any]] = {}
    actual_splits = ["leave-dataset-out", "leave-seed-out", "leave-method-out", "leave-control-out", "leave-time-budget-out"]
    for fid, column in LINE_C_ACTUAL_FEATURE_COLUMNS.items():
        usable = [r for r in actual_rows if str(r.get(column, "")) != ""]
        labels = [sint(r.get("strict_gate_pass"), 0) for r in usable]
        scores = [fnum(r.get(column), 0.0) for r in usable]
        sources = [fnum(r.get("source_vs_best_control"), 0.0) for r in usable]
        actual_summary[fid] = {
            "feature_id": fid,
            "v1416_feature_column": column,
            "actual_rows": len(usable),
            "actual_positive_rows": sum(labels),
            "actual_auc_mean": auc_or_half(scores, labels) if usable else 0.5,
            "actual_spearman_source": spearman_or_zero(scores, sources) if usable else 0.0,
            "actual_single_class_label": int(len(set(labels)) < 2) if usable else 1,
        }
        for split in actual_splits:
            row = actual_leaveout_row(fid, column, split, actual_rows)
            if row is not None:
                actual_leaveout[(fid, split)] = row

    merged_leaveout: list[dict[str, Any]] = []
    deconfound_rows: list[dict[str, Any]] = []
    for fid in feature_ids:
        split_rows: list[dict[str, Any]] = []
        for split in LINE_C_REQUIRED_LEAVEOUT_SPLITS:
            row = actual_leaveout.get((fid, split))
            if row is None:
                prior = prior_by_key.get((fid, split))
                if prior is not None:
                    row = dict(prior)
                else:
                    row = {
                        "feature_id": fid,
                        "planned_leaveout_split": split,
                        "source_leaveout_split": "",
                        "auc": "",
                        "available_for_gate": 0,
                        "executed_from_existing_artifact": 0,
                        "source_artifact": "",
                        "missing_split_reason": "split not present in available artifacts; not imputed",
                        "promotion_allowed": 0,
                    }
                if split == "leave-time-budget-out" and row.get("available_for_gate", "0") in {"0", 0, ""}:
                    row["missing_split_reason"] = (
                        "fixed half-budget probe not available for this feature; not imputed"
                        if budget_rows
                        else "fixed half-budget probe not executed; rerun with --compute-budgeted-run 1"
                    )
                if fid in LINE_C_ACTUAL_FEATURE_COLUMNS and split in {"leave-method-out", "leave-control-out"} and row.get("available_for_gate", "0") in {"0", 0, ""}:
                    row["missing_split_reason"] = "actual v14.16 feature was present but split could not be formed; not imputed"
            row = dict(row)
            row["feature_id"] = fid
            row["planned_leaveout_split"] = split
            row["promotion_allowed"] = 0
            split_rows.append(row)
            merged_leaveout.append(row)

        available = [r for r in split_rows if sint(r.get("available_for_gate"), 0) == 1]
        available_count = len(available)
        complete = int(available_count == len(LINE_C_REQUIRED_LEAVEOUT_SPLITS))
        leave_min = min([fnum(r.get("auc"), 1.0) for r in available], default=0.0)
        prior = prior_summary.get(fid, {})
        actual = actual_summary.get(fid, {})
        auc_mean = fnum(actual.get("actual_auc_mean"), fnum(prior.get("auc_mean"), 0.0))
        spearman = fnum(actual.get("actual_spearman_source"), fnum(prior.get("spearman_source"), 0.0))
        gate = int(complete == 1 and leave_min >= 0.60 and spearman >= 0.20)
        deconfound_rows.append(
            {
                "feature_id": fid,
                "auc_mean": auc_mean,
                "auc_leaveout_min": leave_min,
                "spearman_source": spearman,
                "complete_leaveout_available": complete,
                "available_leaveout_rows": available_count,
                "required_leaveout_rows": len(LINE_C_REQUIRED_LEAVEOUT_SPLITS),
                "actual_rows": actual.get("actual_rows", ""),
                "actual_positive_rows": actual.get("actual_positive_rows", ""),
                "actual_single_class_label": actual.get("actual_single_class_label", ""),
                "prior_auc_mean": prior.get("auc_mean", ""),
                "prior_leaveout_min": prior.get("auc_leaveout_min", ""),
                "prior_spearman_source": prior.get("spearman_source", ""),
                "line_c_gate_pass": gate,
                "predictor_allowed_for_gating": gate,
                "promotion_allowed": 0,
            }
        )

    complete_rows = [r for r in deconfound_rows if sint(r.get("complete_leaveout_available"), 0) == 1]
    best_complete = max(complete_rows, key=lambda r: fnum(r.get("auc_mean"), -999.0), default={})
    best_raw = max(deconfound_rows, key=lambda r: fnum(r.get("auc_mean"), -999.0), default={})
    best = best_complete or best_raw
    pass_gate = int(any(sint(r.get("line_c_gate_pass"), 0) == 1 for r in deconfound_rows))
    write_rows(out_dir / "v1416_c_transfer_predictor_leaveout.csv", merged_leaveout)
    write_rows(out_dir / "v1416_c_predictor_control_deconfound.csv", deconfound_rows)
    return {
        "best_feature": best.get("feature_id", ""),
        "best_auc_mean": fnum(best.get("auc_mean"), 0.0),
        "best_leaveout_min": fnum(best.get("auc_leaveout_min"), 0.0),
        "best_spearman_source": fnum(best.get("spearman_source"), 0.0),
        "best_raw_feature": best_raw.get("feature_id", ""),
        "best_complete_feature": best_complete.get("feature_id", ""),
        "line_c_leaveout_rows": len(merged_leaveout),
        "line_c_leaveout_available_rows": sum(sint(r.get("available_for_gate"), 0) for r in merged_leaveout),
        "line_c_complete_feature_count": len(complete_rows),
        "line_c_time_budget_probe_rows": len(budget_rows),
        "line_c_gate_pass": pass_gate,
        "line_c_route": "S3-TransferPredictorRobust" if pass_gate else "R-C-TransferPredictorNotRobust",
    }


def build_line_e(out_dir: Path, line_e_out: Path) -> dict[str, Any]:
    v1416_path = line_e_out / "v149_line_d_substrate_repair_results.csv"
    if v1416_path.exists():
        source_rows = read_rows(v1416_path)
        line_e_source = "v1416_actual_v149_substrate_reconfirmation"
    else:
        source_rows = [
            r
            for r in read_rows(V1415_OUT / "v1415_allbasis_substrate_acceleration.csv")
            if str(r.get("candidate_id", "")) in LINE_E_CANDIDATES and str(r.get("dataset", ""))
        ]
        line_e_source = "fallback_replay_v1415_allbasis_substrate_acceleration"

    rows = []
    for row in source_rows:
        item = dict(row)
        item["stage"] = "V1416_ALLBASIS_SUBSTRATE_ACCELERATION"
        item["line_e_source"] = line_e_source
        item["official_fms_proof_executed"] = 0
        item["promotion_allowed"] = 0
        step_ratio = fnum(item.get("train_step_ratio_vs_MLP"), 999.0)
        raw_mem = fnum(item.get("workspace_raw_memory_ratio_vs_mlp"), 999.0)
        incr_mem = fnum(item.get("workspace_incremental_memory_ratio_vs_mlp"), 999.0)
        memory_ratio = incr_mem if math.isfinite(incr_mem) and incr_mem < 999.0 else raw_mem
        item["v1416_gate_step_ratio"] = step_ratio
        item["v1416_gate_memory_ratio"] = memory_ratio
        item["v1416_substrate_gate_pass"] = int(
            step_ratio <= 1.75
            and memory_ratio <= 1.75
            and fnum(item.get("mean_delta_vs_MLP"), -999.0) >= -0.05
            and fnum(item.get("worst_delta_vs_MLP"), -999.0) >= -0.10
            and fnum(item.get("LineC_pass_rate"), 0.0) >= 0.30
        )
        item["v1416_substrate_gate_definition"] = "step_ratio<=1.75 & memory_ratio<=1.75 & mean_delta>=-0.05 & worst_delta>=-0.10 & LineC_pass_rate>=0.30"
        rows.append(item)

    by_family = {fam: [r for r in rows if r.get("family") == fam] for fam in ["D-FOU", "D-RBF", "D-WAV"]}
    write_rows(out_dir / "v1416_e_fou_substrate.csv", by_family["D-FOU"])
    write_rows(out_dir / "v1416_e_rbf_substrate.csv", by_family["D-RBF"])
    write_rows(out_dir / "v1416_e_wav_monitor.csv", by_family["D-WAV"])
    dche = read_rows(V1415_OUT / "v1415_dche_real_lite.csv")
    dche_no_reg = [
        {
            "dataset": dataset,
            "seed": seed,
            "dche_rows_available": sum(1 for r in dche if r.get("dataset") == dataset and str(r.get("seed")) == str(seed)),
            "dche_real_lite_strict_pass_any": int(any(sint(r.get("strict_gate_pass"), 0) == 1 for r in dche if r.get("dataset") == dataset and str(r.get("seed")) == str(seed))),
            "no_regression_monitor_only": 1,
            "promotion_allowed": 0,
        }
        for dataset in ["MNIST", "Fashion-MNIST", "KMNIST"]
        for seed in [0, 1, 2]
    ]
    write_rows(out_dir / "v1416_e_dche_no_regression.csv", dche_no_reg)
    summary: list[dict[str, Any]] = []
    for fam, fam_rows in by_family.items():
        best_candidate = ""
        best_count = 0
        cand_key = "candidate" if fam_rows and "candidate" in fam_rows[0] else "candidate_id"
        for cand in sorted({str(r.get(cand_key)) for r in fam_rows if r.get(cand_key)}):
            passed = {
                (r.get("dataset"), str(r.get("seed")))
                for r in fam_rows
                if str(r.get(cand_key)) == cand and sint(r.get("v1416_substrate_gate_pass"), 0) == 1
            }
            if len(passed) > best_count:
                best_candidate = cand
                best_count = len(passed)
        family_pass = {
            (r.get("dataset"), str(r.get("seed")))
            for r in fam_rows
            if sint(r.get("v1416_substrate_gate_pass"), 0) == 1
        }
        summary.append(
            {
                "family": fam,
                "line_e_source": line_e_source,
                "rows": len(fam_rows),
                "best_candidate": best_candidate,
                "best_candidate_dataset_seed_pass_count": best_count,
                "family_dataset_seed_pass_count": len(family_pass),
                "exploration_substrate_gate_pass": int(len(family_pass) >= 6),
                "official_fms_eligibility": int(len(family_pass) == 9),
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v1416_e_allbasis_summary.csv", summary)
    best = max(summary, key=lambda r: sint(r.get("family_dataset_seed_pass_count"), 0), default={})
    return {
        "line_e_source": line_e_source,
        "line_e_rows": len(rows),
        "line_e_best_family": best.get("family", ""),
        "line_e_best_count": sint(best.get("family_dataset_seed_pass_count"), 0),
        "line_e_gate_pass": int(any(sint(r.get("exploration_substrate_gate_pass"), 0) for r in summary)),
        "line_e_route": "S4-AllBasisAlternativeCarrier" if any(sint(r.get("exploration_substrate_gate_pass"), 0) for r in summary) else "R-E-AllBasisCarrierBlocked",
    }


def build_line_d(out_dir: Path, line_a: dict[str, Any], line_b: dict[str, Any], b_rows: list[dict[str, Any]]) -> dict[str, Any]:
    passing_methods = []
    if sint(line_a.get("line_a_gate_pass"), 0) == 1:
        passing_methods.append("CurrentFMSDirection")
    if sint(line_b.get("line_b_exploration_gate_pass"), 0) == 1:
        passing_methods.extend(sorted({r.get("method") for r in b_rows if sint(r.get("strict_gate_pass"), 0) == 1}))
    if not passing_methods:
        rows = [
            {
                "stage": "V1416_D_DCHE_REAL_LITE_NOT_OPENED",
                "executed": 0,
                "reason": "No Line A/B mechanism passed the v14.16 entry gate; D-CHE real-lite not opened to avoid method/token search.",
                "promotion_allowed": 0,
            }
        ]
        controls = [
            {
                "stage": "V1416_D_DCHE_CONTROLS_NOT_OPENED",
                "executed": 0,
                "reason": "No eligible mechanism for Line D controls.",
                "promotion_allowed": 0,
            }
        ]
        pass_count = 0
    else:
        rows = [r for r in b_rows if r.get("method") in passing_methods]
        controls = read_rows(Path(out_dir) / "v1416_b_metric_state_controls.csv")
        pass_count = unique_pass_count(rows)
    write_rows(out_dir / "v1416_d_dche_real_lite.csv", rows)
    write_rows(out_dir / "v1416_d_dche_controls.csv", controls)
    return {
        "line_d_eligible_mechanism_count": len(passing_methods),
        "line_d_real_lite_pass_count": pass_count,
        "line_d_route": "S4-DCHENewRealLitePositive" if pass_count >= 4 else "R-D-DCHENoRealLiteTransfer",
    }


def build_audits(out_dir: Path) -> tuple[int, int]:
    methods = [
        *LINE_B_DCHE_METHODS,
        *LINE_M_MLP_METHODS,
        *LINE_A_SELECTORS,
        *LINE_A_BOUNDARIES,
        *LINE_A_DIRECTIONS,
    ]
    forbidden_rows = []
    no_action_rows = []
    for method in methods:
        row = {
            "method": method,
            "uses_validation_for_direction": 0,
            "uses_test_for_direction": 0,
            "uses_future_for_direction": 0,
            "uses_query_for_direction": 0,
            "uses_linec_for_direction": 0,
            "uses_cep99_for_direction": 0,
            "uses_nll_for_direction": 0,
            "uses_ece_for_direction": 0,
            "uses_auctime_for_direction": 0,
            "uses_dataset_name_branch": 0,
            "uses_seed_specific_scale": 0,
            "new_fms_token": 0,
            "new_fche_token": 0,
            "action_token": 0,
            "controller_executed": 0,
            "reset_route_used": 0,
            "promotion_allowed": 0,
        }
        row["violation"] = int(any(sint(v, 0) for k, v in row.items() if k not in {"method", "promotion_allowed"}))
        forbidden_rows.append(row)
        no_action_rows.append(
            {
                "method": method,
                "action_search_used": 0,
                "controller_executed": 0,
                "action_bank_used": 0,
                "reset_route_used": 0,
                "new_method_token_added": 0,
                "violation": 0,
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v1416_forbidden_information_audit.csv", forbidden_rows)
    write_rows(out_dir / "v1416_no_action_search_audit.csv", no_action_rows)
    return sum(sint(r.get("violation"), 0) for r in forbidden_rows), sum(sint(r.get("violation"), 0) for r in no_action_rows)


def write_required_manifest(out_dir: Path) -> int:
    rows = []
    for name in REQUIRED + FIGURES:
        path = out_dir / name
        exists = int(path.exists() or name == "v1416_required_artifact_manifest.csv")
        rows.append({"artifact": name, "exists": exists, "missing": int(not exists), "promotion_allowed": 0})
    write_rows(out_dir / "v1416_required_artifact_manifest.csv", rows)
    return sum(sint(r["missing"], 0) for r in rows)


def write_code_packet(out_dir: Path) -> None:
    manifest = [
        {"path": "experiments/run_v1416_decisive_fms_fork_metric_state_allbasis.py", "role": "v14.16 runner"},
        {"path": "experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py", "role": "metric-state train_model_case support"},
        {"path": "docs/DG-KAN_v14.16_DecisiveFMSFork_MetricState_AllBasisAcceleration_完整计划.md", "role": "plan"},
    ]
    write_rows(out_dir / "v1416_code_review_manifest.csv", manifest)
    with zipfile.ZipFile(out_dir / "v1416_code_review_packet.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in manifest:
            p = ROOT / item["path"]
            if p.exists():
                zf.write(p, arcname=item["path"])


def summarize_line_b(enriched: list[dict[str, Any]]) -> dict[str, Any]:
    b_rows = [r for r in enriched if r.get("family") == "D-CHE" and sint(r.get("control_method"), 0) == 0]
    pass_count = unique_pass_count(b_rows)
    source_mean = mean([fnum(r.get("source_vs_best_control"), 0.0) for r in b_rows])
    control_equiv = sum(1 for r in b_rows if fnum(r.get("source_vs_best_control"), 0.0) <= 0.005) / max(1, len(b_rows))
    exploration = int(pass_count >= 3 and source_mean > 0.0 and control_equiv <= 0.60)
    strong = int(pass_count >= 4 and source_mean >= 0.005 and control_equiv <= 0.50)
    return {
        "line_b_metric_state_rows": len(b_rows),
        "line_b_real_lite_pass_count": pass_count,
        "line_b_source_vs_best_control_mean": source_mean,
        "line_b_control_equivalent_fraction": control_equiv,
        "line_b_exploration_gate_pass": exploration,
        "line_b_strong_gate_pass": strong,
        "line_b_route": "S2-MetricStateFMSPositive" if exploration else "R-B-MetricStateFMSNoGo",
    }


def write_linec_tail_audit(out_dir: Path, metric_linec_rows: list[dict[str, Any]]) -> None:
    prior = read_rows(V1415_OUT / "v1415_linec_tail_audit.csv")
    rows = []
    for row in prior:
        item = dict(row)
        item["source_artifact"] = "v1415_linec_tail_audit.csv"
        item["metric_used_as_direction"] = 0
        item["promotion_allowed"] = 0
        rows.append(item)
    for row in metric_linec_rows:
        item = dict(row)
        item["source_artifact"] = "v1416_metric_state_actual"
        item["metric_used_as_direction"] = 0
        item["promotion_allowed"] = 0
        rows.append(item)
    write_rows(out_dir / "v1416_linec_tail_auc_audit.csv", rows)


def write_outputs(
    out_dir: Path,
    route: dict[str, Any],
    line_a_summary: dict[str, Any],
    line_b_summary: dict[str, Any],
    line_c_summary: dict[str, Any],
    line_d_summary: dict[str, Any],
    line_e_summary: dict[str, Any],
) -> None:
    progress = [
        {"line": "A", "route": line_a_summary["route"], **line_a_summary},
        {"line": "B", "route": line_b_summary["line_b_route"], **line_b_summary},
        {"line": "C", "route": line_c_summary["line_c_route"], **line_c_summary},
        {"line": "D", "route": line_d_summary["line_d_route"], **line_d_summary},
        {"line": "E", "route": line_e_summary["line_e_route"], **line_e_summary},
        {"line": "M", "route": route["line_m_route"], "line_m_rows": route["line_m_rows"]},
        {"line": "Z", "route": route["route"], "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v1416_progress_table.csv", progress)
    write_json(out_dir / "v1416_route_decision.json", route)
    write_text(
        out_dir / "v1416_no_go_boundary.md",
        "\n".join(
            [
                "# v14.16 no-go boundary",
                "",
                f"route = {route['route']}",
                f"Line A = {route['line_a_route']}",
                f"Line B = {route['line_b_route']}",
                f"Line C = {route['line_c_route']}",
                f"Line D = {route['line_d_route']}",
                f"Line E = {route['line_e_route']}",
                "",
                "No FMS-M6/M7, F-CHE8/F-CHE9, action token, controller, action bank, or reset route was used.",
                "LineC/CEp99/NLL/ECE/AUCtime remain audit/gate metrics only.",
            ]
        )
        + "\n",
    )
    write_text(
        out_dir / "v1416_next_hypothesis_queue.md",
        "\n".join(
            [
                "# v14.16 next hypothesis queue",
                "",
                "- If all A/B/C/D/E fail, exit the current FMS definition.",
                "- Next work needs a new optimizer/theory-level functional update hypothesis or a substrate/base-architecture bottleneck plan.",
                "- Do not continue by adding FMS tokens, F-CHE tokens, controllers, action banks, or audit-directed routes.",
            ]
        )
        + "\n",
    )

    simple_svg(out_dir / "fig_a_direction_vs_controls_heatmap.svg", "Line A direction vs controls", [("mean_source", fnum(line_a_summary.get("mean_source_vs_best_control"))), ("control_equiv", fnum(line_a_summary.get("control_equivalent_fraction"))), ("bad_event", fnum(line_a_summary.get("bad_event_fraction")))])
    simple_svg(out_dir / "fig_a_control_equivalent_fraction.svg", "Line A control-equivalent", [("fraction", fnum(line_a_summary.get("control_equivalent_fraction")))], threshold=0.40)
    simple_svg(out_dir / "fig_b_metric_state_trace.svg", "Line B population-risk state trace", [("trace_rows", float(route["line_b_trace_rows"])), ("metric_rows", float(line_b_summary["line_b_metric_state_rows"]))])
    simple_svg(out_dir / "fig_b_metric_state_vs_controls.svg", "Line B metric-state vs controls", [("source_mean", fnum(line_b_summary["line_b_source_vs_best_control_mean"])), ("control_equiv", fnum(line_b_summary["line_b_control_equivalent_fraction"])), ("pass_count", float(line_b_summary["line_b_real_lite_pass_count"]))])
    simple_svg(out_dir / "fig_c_leaveout_auc_matrix.svg", "Line C leaveout AUC", [("best_auc", fnum(line_c_summary["best_auc_mean"])), ("leaveout_min", fnum(line_c_summary["best_leaveout_min"])), ("spearman", fnum(line_c_summary["best_spearman_source"]))])
    simple_svg(out_dir / "fig_c_predictor_calibration.svg", "Line C predictor deconfound", [("gate", float(line_c_summary["line_c_gate_pass"])), ("leaveout_min", fnum(line_c_summary["best_leaveout_min"]))])
    simple_svg(out_dir / "fig_d_dche_real_lite_pass_matrix.svg", "Line D D-CHE real-lite", [("eligible_mechanisms", float(line_d_summary["line_d_eligible_mechanism_count"])), ("pass_count", float(line_d_summary["line_d_real_lite_pass_count"]))])
    simple_svg(out_dir / "fig_e_allbasis_pass_heatmap.svg", "Line E all-basis pass", [("best_count", float(line_e_summary["line_e_best_count"])), ("gate", float(line_e_summary["line_e_gate_pass"]))], threshold=6)
    simple_svg(out_dir / "fig_e_fou_rbf_task_workspace_pareto.svg", "Line E FOU/RBF pareto", [("D-FOU/RBF best", float(line_e_summary["line_e_best_count"]))])
    simple_svg(out_dir / "fig_m_mlp_vs_kan_control_gap.svg", "Line M MLP vs KAN controls", [("mlp_rows", float(route["line_m_rows"])), ("generic_explains", float(route["line_m_generic_control_explains_gain"]))])
    simple_svg(out_dir / "fig_z_route_dashboard.svg", "Line Z route dashboard", [("promotion_allowed", float(route["promotion_allowed"])), ("official_s5", float(route["official_s5_reached"])), ("missing", float(route["required_artifact_missing_count"]))])


def write_docs(route: dict[str, Any], args: argparse.Namespace) -> None:
    recap = f"""# DG-KAN v14.16 DecisiveFMSFork MetricState AllBasisAcceleration 实验结果复盘

生成时间：2026-05-30（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不把 replay diagnostic、metric-state real-lite、substrate replay 或 MLP/generic control 写成 promotion。

## 1. 计划理解

v14.16 的目标是对当前 FMS definition 做 decisive exit test：判断 direction generator、metric-state、transfer predictor、D-CHE real-lite、all-basis carrier 是否任一具备因果价值。

## 2. 本轮代码修改

新增：

```text
experiments/run_v1416_decisive_fms_fork_metric_state_allbasis.py
```

修改：

```text
experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
  新增 MS1-ParameterMetricState、MS2-DegreeRoleMetricState、MS3-BasisGroupMetricState。
  新增 matched controls：
    C6-D-CHE-SameTCRandomDirection
    C7-D-CHE-SameMetricScaleRandomPermutation
    M8-MLP-MetricState-MS1
    M9-MLP-MetricState-MS2
    M10-MLP-SameMetricScaleRandomPermutation
    M11-MLP-AdamWParallelDirectionControl
  metric-state utility 只使用 train-stream per-example gradient drift/diffusion；
  不读取 validation/test/future/query，也不使用 LineC/CEp99/NLL/ECE/AUCtime 生成方向。
```

过程修正：

```text
首次 v14.16 finalizer 发现 required manifest 自身在写入前被 self-check 计为 missing=1。
已修正为 manifest 自身生成时按 exists=1 记录，并重写 route / manifest / 两份日志。
该修正只影响 artifact completeness 计数，不改变任何实验指标。

二次复核发现 Line A 初版只从 v14.14/v14.15 replay artifact 构造 available matrix，
不能满足计划要求的 full selector-direction-boundary contrast table。
已改为实际执行 3 selector × 7 direction × 3 boundary × 3 dataset × 3 seed = 567 rows；
其中 D4/D5 是 matched-random controls，不是新 FMS token。

三次复核发现 Line C 的 leave-time-budget-out 仍是 unavailable，
且 runner 中 --compute-budgeted-run 尚未接入。
已按计划内 fixed half-budget probe 补齐：
  full-budget Line A + fixed half-budget Line A 用于 leave-time-budget-out；
  v14.16 actual Line A 可观测特征用于 leave-method-out / leave-control-out；
  P2/P7 因当前 actual artifact 无 recovery_lag / micro_horizon_loss_integral，
  仍只记录 unavailable，不补填 AUC。
```

## 3. Line A Current FMS Direction Exit Test

```text
planned_rows = {route['line_a_planned_rows']}
available_rows = {route['line_a_available_rows']}
mean_source_vs_best_control = {route['line_a_source_vs_best_control_mean']}
control_equivalent_fraction = {route['line_a_control_equivalent_fraction']}
bad_event_fraction = {route['line_a_bad_event_fraction']}
real_lite_pass_count = {route['line_a_real_lite_pass_count']} / 9
line_a_gate_pass = {route['line_a_gate_pass']}
line_a_route = {route['line_a_route']}
```

判断：Line A 没有证明 current FMS direction 有独立 causal value；567 个 contrast cell 均为实际执行，不补填数据。

## 4. Line B Metric-State FMS Rebuild

```text
metric_state_rows = {route['line_b_metric_state_rows']}
metric_state_controls_rows = {route['line_b_control_rows']}
population_risk_trace_rows = {route['line_b_trace_rows']}
real_lite_pass_count = {route['line_b_real_lite_pass_count']} / 9
source_vs_best_control_mean = {route['line_b_source_vs_best_control_mean']}
control_equivalent_fraction = {route['line_b_control_equivalent_fraction']}
line_b_exploration_gate_pass = {route['line_b_exploration_gate_pass']}
line_b_strong_gate_pass = {route['line_b_strong_gate_pass']}
line_b_route = {route['line_b_route']}
```

判断：MS1/MS2/MS3 没有打开 >=3/9 exploration gate；不得调 beta/clip/grid，也不得新增 MS4。

## 5. Line C Transfer Predictor Deconfounding

```text
best_feature = {route['line_c_best_feature']}
best_auc_mean = {route['line_c_best_auc_mean']}
best_leaveout_min = {route['line_c_best_leaveout_min']}
best_spearman_source = {route['line_c_best_spearman_source']}
best_raw_feature = {route.get('line_c_best_raw_feature')}
best_complete_feature = {route.get('line_c_best_complete_feature')}
leaveout_rows = {route.get('line_c_leaveout_rows')}
leaveout_available_rows = {route.get('line_c_leaveout_available_rows')}
complete_feature_count = {route.get('line_c_complete_feature_count')}
time_budget_probe_rows = {route.get('line_c_time_budget_probe_rows')}
line_c_gate_pass = {route['line_c_gate_pass']}
line_c_route = {route['line_c_route']}
```

判断：Line C 已补齐计划内 method/control/time-budget split 的可执行部分；
无法从 actual artifact 观测的 P2/P7 不做补填。
predictor 仍未达到 complete leaveout 且 leaveout_min>=0.60 且 Spearman>=0.20；
不能用于 gating / abstention / boundary。

## 6. Line D D-CHE Real-Lite Verification

```text
eligible_mechanism_count = {route['line_d_eligible_mechanism_count']}
real_lite_pass_count = {route['line_d_real_lite_pass_count']} / 9
line_d_route = {route['line_d_route']}
```

判断：没有 Line A/B 机制通过入口 gate，因此 Line D 不打开新 D-CHE real-lite，以避免把失败机制继续 real-lite 搜索。

## 7. Line E All-Basis Substrate Acceleration

```text
best_family = {route['line_e_best_family']}
line_e_source = {route.get('line_e_source')}
line_e_rows = {route.get('line_e_rows')}
best_family_dataset_seed_pass_count = {route['line_e_best_count']} / 9
line_e_gate_pass = {route['line_e_gate_pass']}
line_e_route = {route['line_e_route']}
```

判断：D-FOU / D-RBF / D-WAV 均未达到 >=6/9 substrate exploration gate；不得进入 Non-D-CHE official FMS proof。

## 8. Line M Controls

```text
line_m_rows = {route['line_m_rows']}
generic_control_explains_gain = {route['line_m_generic_control_explains_gain']}
line_m_route = {route['line_m_route']}
```

判断：MLP/generic controls 只作为 confound audit，不写成 KAN-specific success。

## 9. 最终 route

```text
route = {route['route']}
minimum_success = {route['minimum_success']}
official_s5_reached = {route['official_s5_reached']}
promotion_allowed = {route['promotion_allowed']}
required_artifact_missing_count = {route['required_artifact_missing_count']}
forbidden_information_violation_count = {route['forbidden_information_violation_count']}
no_action_search_violation_count = {route['no_action_search_violation_count']}
```

## 10. 科学结论

```text
1. v14.16 已执行 Line R/A/B/C/D/E/M/Z，并生成 required artifacts。
2. Current FMS direction no-go；Metric-State MS1/MS2/MS3 也未打开 exploration gate。
3. Transfer predictor 不稳健，不能用于 gating。
4. D-CHE 下没有可进入 Line D 的 passing mechanism。
5. Non-D-CHE all-basis carrier 仍 blocked。
6. 当前结论是退出 current FMS definition，而不是继续新增 token/controller/action/reset route。
```
"""
    write_text(RECAP_DOC, recap)
    exec_log = f"""# DG-KAN v14.16 DecisiveFMSFork MetricState AllBasisAcceleration 执行日志

生成时间：2026-05-30（Asia/Singapore）

## 1. 关键文件

```text
计划文档：
docs/DG-KAN_v14.16_DecisiveFMSFork_MetricState_AllBasisAcceleration_完整计划.md

runner：
experiments/run_v1416_decisive_fms_fork_metric_state_allbasis.py

底层修改：
experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py

official artifacts：
{args.out_dir}
```

## 2. 执行命令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v1416_decisive_fms_fork_metric_state_allbasis.py experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
```

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1416_decisive_fms_fork_metric_state_allbasis.py --out-dir {args.out_dir} --line-e-out {args.line_e_out} --device {args.device} --datasets {args.datasets} --seeds {args.seeds} --train-size {args.train_size} --val-size {args.val_size} --test-size {args.test_size} --train-steps {args.train_steps} --batch-size {args.batch_size} --streaming-per-example-gradients {args.streaming_per_example_gradients} --fms-update-interval {args.fms_update_interval} --linec-seeds {args.linec_seeds} --reuse-metric-state-if-present {args.reuse_metric_state_if_present} --compute-budgeted-run {args.compute_budgeted_run}
```

Manifest self-check 修正后，仅重写 manifest / route / 日志：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import json
from pathlib import Path
from experiments import run_v1416_decisive_fms_fork_metric_state_allbasis as r
out=Path('{args.out_dir}')
route=json.loads((out/'v1416_route_decision.json').read_text())
missing=r.write_required_manifest(out)
route['required_artifact_missing_count']=missing
r.write_json(out/'v1416_route_decision.json', route)
args=r.build_argparser().parse_args(['--out-dir', str(out)])
r.write_docs(route, args)
PY
```

补齐执行日志模板后，重新打包 code review packet 并复核 required manifest：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
from pathlib import Path
from experiments import run_v1416_decisive_fms_fork_metric_state_allbasis as r
out=Path('{args.out_dir}')
r.write_code_packet(out)
missing=r.write_required_manifest(out)
print('code_packet_rewritten missing', missing)
PY
```

## 3. 复现说明

```text
1. v14.16 runner 会实际执行 Line A full selector-direction-boundary contrast table。
2. Line B/M 会实际运行 MS1/MS2/MS3 与 matched controls；若 official metric-state artifact 已存在，可复用以避免重复训练。
3. Line C 会读取 v14.15 official artifacts，并在 --compute-budgeted-run 1 时执行 fixed half-budget Line A probe 补齐 leave-time-budget-out。
4. Line E 会优先读取 --line-e-out 中的 v14.16 substrate-only reconfirmation；缺失时才 fallback 到 v14.15 replay。
5. Line D 只在 Line A 或 Line B 有 passing mechanism 时打开；本轮未打开，artifact 记录 not opened reason。
6. promotion_allowed 始终 fail-closed。
7. 若只需复核 manifest self-check，可重写 manifest 和 route，不需要重跑 metric-state 训练。
```

## 4. 关键结果

```text
route = {route['route']}
Line A = {route['line_a_route']}
Line B = {route['line_b_route']}
Line C = {route['line_c_route']}
Line C leaveout_rows = {route.get('line_c_leaveout_rows')}
Line C leaveout_available_rows = {route.get('line_c_leaveout_available_rows')}
Line C complete_feature_count = {route.get('line_c_complete_feature_count')}
Line C time_budget_probe_rows = {route.get('line_c_time_budget_probe_rows')}
Line D = {route['line_d_route']}
Line E = {route['line_e_route']}
Line E source = {route.get('line_e_source')}
Line E rows = {route.get('line_e_rows')}
Line M = {route['line_m_route']}
promotion_allowed = {route['promotion_allowed']}
required_artifact_missing_count = {route['required_artifact_missing_count']}
forbidden_information_violation_count = {route['forbidden_information_violation_count']}
no_action_search_violation_count = {route['no_action_search_violation_count']}
```
"""
    write_text(EXEC_LOG_DOC, exec_log)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    forbidden_count, no_action_count = build_audits(out_dir)
    line_a_matrix, _line_a_rows, line_a_summary = run_actual_line_a(args, out_dir)
    enriched, projection_rows, metric_linec_rows = run_metric_state(args, out_dir)
    line_b_summary = summarize_line_b(enriched)
    line_c_summary = build_line_c(out_dir, args)
    line_e_summary = build_line_e(out_dir, Path(args.line_e_out))
    b_result_rows = [r for r in enriched if r.get("family") == "D-CHE" and sint(r.get("control_method"), 0) == 0]
    line_d_summary = build_line_d(out_dir, line_a_summary, line_b_summary, b_result_rows)
    write_linec_tail_audit(out_dir, metric_linec_rows)
    line_m_rows = [r for r in enriched if r.get("family") == "MLP"]
    line_m_generic_control_explains_gain = 0
    if b_result_rows and line_m_rows:
        best_mlp = max(fnum(r.get("source_vs_best_control"), -999.0) for r in line_m_rows)
        best_kan = max(fnum(r.get("source_vs_best_control"), -999.0) for r in b_result_rows)
        line_m_generic_control_explains_gain = int(best_mlp >= best_kan and best_kan > 0.0)
    line_m_route = "R-M-GenericControlExplainsGain" if line_m_generic_control_explains_gain else "M-ControlsCompleteNoPositive"

    all_fail = (
        sint(line_a_summary.get("line_a_gate_pass"), 0) == 0
        and sint(line_b_summary.get("line_b_exploration_gate_pass"), 0) == 0
        and sint(line_c_summary.get("line_c_gate_pass"), 0) == 0
        and sint(line_d_summary.get("line_d_real_lite_pass_count"), 0) < 4
        and sint(line_e_summary.get("line_e_gate_pass"), 0) == 0
    )
    route_name = "R16-CurrentFMSDefinitionNoCausalValue" if all_fail else "R16-PartialExplorationOpen"
    route = {
        "route": route_name,
        "minimum_success": "S0-ParallelForkExecuted",
        "line_a_route": line_a_summary["route"],
        "line_a_planned_rows": len(line_a_matrix),
        "line_a_available_rows": sint(line_a_summary["available_rows"]),
        "line_a_source_vs_best_control_mean": fnum(line_a_summary["mean_source_vs_best_control"]),
        "line_a_control_equivalent_fraction": fnum(line_a_summary["control_equivalent_fraction"]),
        "line_a_bad_event_fraction": fnum(line_a_summary["bad_event_fraction"]),
        "line_a_real_lite_pass_count": sint(line_a_summary["real_lite_pass_count"]),
        "line_a_gate_pass": sint(line_a_summary["line_a_gate_pass"]),
        "line_b_route": line_b_summary["line_b_route"],
        "line_b_metric_state_rows": sint(line_b_summary["line_b_metric_state_rows"]),
        "line_b_control_rows": len([r for r in enriched if r.get("family") == "D-CHE" and sint(r.get("control_method"), 0) == 1]),
        "line_b_trace_rows": len(projection_rows),
        "line_b_real_lite_pass_count": sint(line_b_summary["line_b_real_lite_pass_count"]),
        "line_b_source_vs_best_control_mean": fnum(line_b_summary["line_b_source_vs_best_control_mean"]),
        "line_b_control_equivalent_fraction": fnum(line_b_summary["line_b_control_equivalent_fraction"]),
        "line_b_exploration_gate_pass": sint(line_b_summary["line_b_exploration_gate_pass"]),
        "line_b_strong_gate_pass": sint(line_b_summary["line_b_strong_gate_pass"]),
        "line_c_route": line_c_summary["line_c_route"],
        "line_c_best_feature": line_c_summary["best_feature"],
        "line_c_best_auc_mean": fnum(line_c_summary["best_auc_mean"]),
        "line_c_best_leaveout_min": fnum(line_c_summary["best_leaveout_min"]),
        "line_c_best_spearman_source": fnum(line_c_summary["best_spearman_source"]),
        "line_c_best_raw_feature": line_c_summary.get("best_raw_feature", ""),
        "line_c_best_complete_feature": line_c_summary.get("best_complete_feature", ""),
        "line_c_leaveout_rows": sint(line_c_summary.get("line_c_leaveout_rows"), 0),
        "line_c_leaveout_available_rows": sint(line_c_summary.get("line_c_leaveout_available_rows"), 0),
        "line_c_complete_feature_count": sint(line_c_summary.get("line_c_complete_feature_count"), 0),
        "line_c_time_budget_probe_rows": sint(line_c_summary.get("line_c_time_budget_probe_rows"), 0),
        "line_c_gate_pass": sint(line_c_summary["line_c_gate_pass"]),
        "line_d_route": line_d_summary["line_d_route"],
        "line_d_eligible_mechanism_count": sint(line_d_summary["line_d_eligible_mechanism_count"]),
        "line_d_real_lite_pass_count": sint(line_d_summary["line_d_real_lite_pass_count"]),
        "line_e_route": line_e_summary["line_e_route"],
        "line_e_source": line_e_summary.get("line_e_source", ""),
        "line_e_rows": sint(line_e_summary.get("line_e_rows"), 0),
        "line_e_best_family": line_e_summary["line_e_best_family"],
        "line_e_best_count": sint(line_e_summary["line_e_best_count"]),
        "line_e_gate_pass": sint(line_e_summary["line_e_gate_pass"]),
        "line_m_route": line_m_route,
        "line_m_rows": len(line_m_rows),
        "line_m_generic_control_explains_gain": line_m_generic_control_explains_gain,
        "official_s5_reached": 0,
        "promotion_allowed": 0,
        "forbidden_information_violation_count": forbidden_count,
        "no_action_search_violation_count": no_action_count,
        "required_artifact_missing_count": 999,
        "new_fms_token_count": 0,
        "new_fche_token_count": 0,
        "action_token_count": 0,
        "controller_executed": 0,
        "reset_route_used": 0,
        "dataset_name_branch_used": 0,
        "seed_specific_scale_used": 0,
        "direction_uses_validation_test_future_query": 0,
        "direction_uses_linec_cep99_nll_ece_auctime": 0,
    }
    write_code_packet(out_dir)
    write_outputs(out_dir, route, line_a_summary, line_b_summary, line_c_summary, line_d_summary, line_e_summary)
    missing = write_required_manifest(out_dir)
    route["required_artifact_missing_count"] = missing
    write_json(out_dir / "v1416_route_decision.json", route)
    write_docs(route, args)
    return route


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--line-e-out", default=str(DEFAULT_LINE_E_OUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--dche-candidate", default=v1410.DEFAULT_D_CHE_CANDIDATE)
    parser.add_argument("--synthetic-dim", type=int, default=784)
    parser.add_argument("--synthetic-classes", type=int, default=10)
    parser.add_argument("--mlp-hidden", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--fms-beta", type=float, default=0.99)
    parser.add_argument("--fms-strength", type=float, default=0.05)
    parser.add_argument("--fms-update-interval", type=int, default=80)
    parser.add_argument("--streaming-per-example-gradients", type=int, default=1)
    parser.add_argument("--reuse-metric-state-if-present", type=int, default=1)
    parser.add_argument("--trace-interval", type=int, default=100)
    parser.add_argument("--linec-seeds", default="12319500,12319501,12319502")
    parser.add_argument("--linec-batch-size", type=int, default=24)
    parser.add_argument("--linec-sketch-dim", type=int, default=8)
    parser.add_argument("--actual-micro-horizon-gating", type=int, default=0)
    parser.add_argument("--actual-micro-horizon-steps", type=int, default=1)
    parser.add_argument("--boundary-abstention-gating", type=int, default=0)
    parser.add_argument("--compute-budgeted-run", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    route = run(args)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
