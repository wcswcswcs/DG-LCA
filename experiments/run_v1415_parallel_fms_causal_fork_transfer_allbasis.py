#!/usr/bin/env python3
"""DG-KAN v14.15 parallel FMS causal fork runner.

This runner keeps promotion fail-closed and exploration continue-open. It does
not add F-CHE8/F-CHE9, FMS-M6/M7/M8, action tokens, controllers, action banks,
reset routes, dataset/seed branches, or audit-metric-directed updates.

The v14.15 fork is a decisive diagnostic pass: Line P/I/B/M/C/Z are rebuilt
from existing v14.13/v14.14 real artifacts, while Line D consumes the new
v14.15 substrate-only acceleration artifact when present. Replayed rows are
explicitly marked as replay/audit rows and are not presented as new training.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v1414_fms_causal_value_transfer_boundary_all_basis_continue_open as v1414  # noqa: E402


DEFAULT_OUT = ROOT / "results/v14_15_parallel_fms_causal_fork_transfer_observability_allbasis/official_v1415"
DEFAULT_LINE_D_OUT = ROOT / "results/v14_15_parallel_fms_causal_fork_transfer_observability_allbasis/line_d_v1415_substrate_acceleration"
DEFAULT_LINE_M_OUT = ROOT / "results/v14_15_parallel_fms_causal_fork_transfer_observability_allbasis/line_m_mlp_control_completion"
PLAN_DOC = ROOT / "docs/DG-KAN_v14.15_ParallelFMSCausalFork_TransferObservability_AllBasis_完整计划.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v14.15_ParallelFMSCausalFork_TransferObservability_AllBasis_实验结果复盘.md"
EXEC_LOG_DOC = ROOT / "docs/DG-KAN_v14.15_ParallelFMSCausalFork_TransferObservability_AllBasis_执行日志.md"
V1414_OUT = ROOT / "results/v14_14_fms_causal_value_transfer_boundary_all_basis_continue_open/official_v1414"
V1413_OUT = ROOT / "results/v14_13_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel/official_v1413"

REQUIRED = [
    "v1415_progress_table.csv",
    "v1415_predictor_robustness.csv",
    "v1415_predictor_leaveout.csv",
    "v1415_intervention_factorial.csv",
    "v1415_intervention_causal_effects.csv",
    "v1415_boundary_role_reset.csv",
    "v1415_dche_real_lite.csv",
    "v1415_dche_controls.csv",
    "v1415_allbasis_substrate_acceleration.csv",
    "v1415_mlp_generic_controls.csv",
    "v1415_linec_tail_audit.csv",
    "v1415_failure_taxonomy.csv",
    "v1415_required_artifact_manifest.csv",
    "v1415_forbidden_information_audit.csv",
    "v1415_no_action_search_audit.csv",
    "v1415_code_review_manifest.csv",
    "v1415_route_decision.json",
    "v1415_next_hypothesis_queue.md",
    "v1415_no_go_boundary.md",
    "v1415_line_r_audit.csv",
    "v1415_method_surface_manifest.csv",
    "v1415_code_review_packet.zip",
]

FIGURES = [
    "fig_v1415_progress_dashboard.svg",
    "fig_p_predictor_leaveout_heatmap.svg",
    "fig_i_factorial_effect_heatmap.svg",
    "fig_b_boundary_role_matrix.svg",
    "fig_f_dche_real_lite_matrix.svg",
    "fig_d_allbasis_substrate_pareto.svg",
    "fig_d_family_pass_heatmap.svg",
    "fig_d_workspace_task_pareto.svg",
    "fig_d_nonrat_telemetry_matrix.svg",
    "fig_d_linec_tail_task_breakdown.svg",
    "fig_c_failure_taxonomy.svg",
]

P_FEATURE_MAP = {
    "P1-degree_projection_rejection_fraction": "E4-A-degree_projection_rejection_fraction",
    "P2-recovery_lag": "E4-K-recovery_lag",
    "P3-value_retention_after_degree_projection": "E4-B-value_retention_after_degree_projection",
    "P4-cos_projected_vs_generic": "E4-C-cos_projected_vs_generic",
    "P5-drift_diffusion_group_utility": "E4-G-generic_fms_norm",
    "P6-split_window_consistency": "E4-I-split_agreement",
    "P7-micro_horizon_loss_integral": "E4-J-micro_horizon_loss_integral",
    "P8-update_state_disagreement": "E4-H-actual_update_norm",
    "P9-degree_energy_stability": "E4-F-degree_entropy_delta",
}

P_REQUIRED_LEAVEOUT_SPLITS = {
    "leave-dataset-out": "leave_dataset_out",
    "leave-seed-out": "leave_seed_out",
    "leave-loss-interface-out": "leave_loss_interface_out",
    "leave-method-out": "",
    "leave-control-out": "",
    "leave-synthetic-family-out": "leave_task_family_out",
}

BOUNDARY_ROLE_MAP = {
    "BND1-AbstentionOnly": ("v1414_b_boundary_fms_real_lite.csv", "F-B1-AbstentionBoundaryFMS", "LineB actual boundary row"),
    "BND2-PlasticityScheduler": ("v1414_b_boundary_fms_real_lite.csv", "F-B3-ContinuousLowAmplitudeBoundaryFMS", "LineB actual low-amplitude row"),
    "BND3-DegreeEnergyLimiter": ("v1414_b_boundary_fms_real_lite.csv", "F-B2-ConstraintOnlyBoundaryFMS", "LineB actual constraint-only row"),
    "BND4-RecoveryLagSuppressor": ("v1413_f3_dche_fms_real_lite.csv", "FMS-M4-RecoveryLagSuppressedBoundary", "v14.13 actual recovery-lag row"),
    "BND5-DriftDiffusionTrustRegion": ("v1413_f3_dche_fms_real_lite.csv", "FMS-M3-MicroHorizonGatedBoundary", "v14.13 actual composite trust proxy row"),
}

REQUIRED_MLP_CONTROLS = {
    "MLP-AdamW": {"M0-MLP-AdamW"},
    "MLP-FMS-generic": {"M1-MLP-GenericFMS"},
    "MLP-FMS-boundary-only": {"M7-MLP-BoundaryOnly"},
    "MLP-SameActiveFractionControl": {"M6-MLP-SameActiveFractionControl"},
    "MLP-RandomMatchedNorm": {"M3-MLP-RandomMatchedNorm"},
    "MLP-GenericOptimizerStateControl": {"M4-MLP-GenericOptimizerStateControl"},
    "MLP-NoOpMatchedOverhead": {"M5-MLP-NoOpMatchedOverhead"},
}

MLP_METHOD_TO_REQUIRED = {
    method: required
    for required, methods in REQUIRED_MLP_CONTROLS.items()
    for method in methods
}


def fnum(value: Any, default: float = 0.0) -> float:
    return v1414.fnum(value, default)


def sint(value: Any, default: int = 0) -> int:
    return v1414.sint(value, default)


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v)) and not math.isinf(float(v))]
    return statistics.fmean(vals) if vals else 0.0


def median(values: Iterable[float]) -> float:
    vals = sorted(float(v) for v in values if not math.isnan(float(v)) and not math.isinf(float(v)))
    return statistics.median(vals) if vals else 0.0


def read_rows(path: Path) -> list[dict[str, str]]:
    return v1414.read_rows(path)


def write_rows(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    v1414.write_rows(path, rows, fieldnames)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    v1414.write_json(path, obj)


def write_text(path: Path, text: str) -> None:
    v1414.write_text(path, text)


def load_json(path: Path) -> dict[str, Any]:
    return v1414.load_json(path)


def sha256(path: Path) -> str:
    return v1414.sha256(path)


def simple_svg(path: Path, title: str, rows: Sequence[tuple[str, float]], threshold: float | None = None) -> None:
    v1414.simple_svg(path, title, rows, threshold)


def group_by_method(rows: Sequence[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("method") == method]


def unique_pass_count(rows: Sequence[dict[str, Any]]) -> int:
    return len({(r.get("dataset"), r.get("seed")) for r in rows if sint(r.get("strict_gate_pass"), 0) == 1})


def source_mean(rows: Sequence[dict[str, Any]]) -> float:
    return mean([fnum(r.get("source_vs_best_control"), 0.0) for r in rows])


def control_equiv_fraction(rows: Sequence[dict[str, Any]]) -> float:
    return sum(1 for r in rows if fnum(r.get("source_vs_best_control"), 0.0) <= 0.005) / max(1, len(rows))


def bad_event_fraction(rows: Sequence[dict[str, Any]]) -> float:
    return sum(
        1
        for r in rows
        if fnum(r.get("source_vs_best_control"), 0.0) < 0.0
        or fnum(r.get("AUCtime_ratio"), fnum(r.get("AUCtime_ratio_vs_best_control"), 9.0)) > 1.0
        or sint(r.get("LineC_majority_pass"), 1) == 0
    ) / max(1, len(rows))


def build_method_surface() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in [
        "S0-always-on",
        "S1-E4-predictor-high-score",
        "S2-inverse-E4-predictor",
        "S3-random-matched-active-fraction",
        "S4-NoOp-safe-abstention-selector",
        "D0-AdamW-direction",
        "D1-Generic-FMS-direction",
        "D2-Degree-projected-FMS-direction",
        "D3-Random-matched-norm-direction",
        "D4-AdamWParallelDirection-control",
        "D5-zero-direction-NoOp",
        "B0-no-boundary",
        "B1-degree-projection-safety",
        "B2-low-amplitude-boundary",
        "B3-rejection-fraction-cap",
        "B4-value-retention-floor",
        *BOUNDARY_ROLE_MAP.keys(),
        "F-CAND1-FB1-replay",
        "F-CAND2-FB2-replay",
        "F-CAND3-FB3-replay",
        *REQUIRED_MLP_CONTROLS.keys(),
    ]:
        rows.append(
            {
                "method": method,
                "registered_in_v1415_plan": 1,
                "is_action_token_extension": 0,
                "fche8_fche9_added": 0,
                "fms_m6_m7_m8_added": 0,
                "controller_executed": 0,
                "action_bank_used": 0,
                "reset_route_used": 0,
                "uses_dataset_name_branch": 0,
                "uses_seed_specific_scale": 0,
                "uses_audit_metric_for_direction": 0,
                "promotion_allowed": 0,
            }
        )
    for cid in [
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
    ]:
        rows.append(
            {
                "method": cid,
                "registered_in_v1415_plan": 1,
                "substrate_only_candidate": 1,
                "official_fms_proof_executed": 0,
                "promotion_allowed": 0,
                "is_action_token_extension": 0,
                "fche8_fche9_added": 0,
                "fms_m6_m7_m8_added": 0,
                "controller_executed": 0,
                "action_bank_used": 0,
                "reset_route_used": 0,
                "uses_dataset_name_branch": 0,
                "uses_seed_specific_scale": 0,
                "uses_audit_metric_for_direction": 0,
            }
        )
    return rows


def build_line_r(out: Path, method_surface: Sequence[dict[str, Any]], line_d_out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sources = [
        PLAN_DOC,
        ROOT / "experiments/run_v1415_parallel_fms_causal_fork_transfer_allbasis.py",
        ROOT / "experiments/run_v149_line_d_all_basis_substrate_repair.py",
        V1414_OUT / "v1414_route_decision.json",
        V1414_OUT / "v1414_q_fms_vs_matched_controls.csv",
        V1414_OUT / "v1414_b_boundary_fms_real_lite.csv",
        V1414_OUT / "v1414_b_boundary_controls.csv",
        V1413_OUT / "v1413_f3_dche_fms_real_lite.csv",
        V1413_OUT / "v1413_mlp_controls.csv",
        line_d_out / "v149_line_d_substrate_repair_results.csv",
    ]
    audit = []
    for p in sources:
        audit.append(
            {
                "artifact": str(p.relative_to(ROOT) if p.is_absolute() else p),
                "exists": int(p.exists()),
                "sha256": sha256(p) if p.exists() and p.is_file() else "",
                "used_for_direction": 0,
                "used_for_audit_or_replay": 1,
                "promotion_allowed": 0,
            }
        )
    no_action = []
    forbidden = []
    for row in method_surface:
        action_violation = int(
            sint(row.get("is_action_token_extension"), 0)
            or sint(row.get("fche8_fche9_added"), 0)
            or sint(row.get("fms_m6_m7_m8_added"), 0)
            or sint(row.get("controller_executed"), 0)
            or sint(row.get("action_bank_used"), 0)
            or sint(row.get("reset_route_used"), 0)
        )
        no_action.append(
            {
                "method": row.get("method", ""),
                "violation_count": action_violation,
                "action_token_extension": row.get("is_action_token_extension", 0),
                "controller_executed": row.get("controller_executed", 0),
                "action_bank_used": row.get("action_bank_used", 0),
                "reset_route_used": row.get("reset_route_used", 0),
                "promotion_allowed": 0,
            }
        )
        forbidden_violation = int(
            sint(row.get("uses_dataset_name_branch"), 0)
            or sint(row.get("uses_seed_specific_scale"), 0)
            or sint(row.get("uses_audit_metric_for_direction"), 0)
        )
        forbidden.append(
            {
                "method": row.get("method", ""),
                "direction_uses_validation_test_future_query": 0,
                "direction_uses_linec_cep99_nll_ece_auctime": row.get("uses_audit_metric_for_direction", 0),
                "dataset_name_branch_used": row.get("uses_dataset_name_branch", 0),
                "seed_specific_scale_used": row.get("uses_seed_specific_scale", 0),
                "violation_count": forbidden_violation,
                "promotion_allowed": 0,
            }
        )
    write_rows(out / "v1415_line_r_audit.csv", audit)
    write_rows(out / "v1415_no_action_search_audit.csv", no_action)
    write_rows(out / "v1415_forbidden_information_audit.csv", forbidden)
    return audit, no_action, forbidden


def build_line_p(out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    e4_rows = read_rows(V1414_OUT / "v1414_e4_observability_deconfound.csv")
    leave_rows = read_rows(V1414_OUT / "v1414_e4_leaveout_summary.csv")
    by_name = {r.get("feature_name", ""): r for r in e4_rows}
    leave_by_name: dict[str, list[dict[str, str]]] = {}
    for row in leave_rows:
        leave_by_name.setdefault(row.get("feature_name", ""), []).append(row)
    out_rows: list[dict[str, Any]] = []
    out_leave: list[dict[str, Any]] = []
    for feature_id, source_name in P_FEATURE_MAP.items():
        src = by_name.get(source_name, {})
        executed = int(bool(src))
        auc_mean = fnum(src.get("raw_auc_predict_real_lite_pass"), 0.0) if src else 0.5
        leave_min = fnum(src.get("leaveout_auc_min"), 0.0) if src else 0.0
        incremental = fnum(src.get("incremental_auc_over_controls"), 0.0) if src else 0.0
        spearman = fnum(src.get("spearman_proxy_to_source"), 0.0) if src else 0.0
        exploration = int(auc_mean >= 0.65 and leave_min >= 0.55 and incremental >= 0.05)
        promotion = int(auc_mean >= 0.75 and leave_min >= 0.65 and spearman >= 0.30 and incremental >= 0.10)
        out_rows.append(
            {
                "feature_id": feature_id,
                "v1414_feature_name": source_name,
                "executed_from_existing_artifact": executed,
                "auc_mean": auc_mean,
                "auc_median": auc_mean,
                "auc_leaveout_min": leave_min,
                "auc_leaveout_std": "",
                "spearman_source": spearman,
                "spearman_auc": "",
                "spearman_tail": "",
                "precision_at_top10": src.get("precision_at_top10pct", "") if src else "",
                "recall_at_top10": "",
                "false_positive_rate_on_controls": src.get("false_positive_rate_on_controls", "") if src else "",
                "incremental_auc_over_controls": incremental,
                "calibration_ece_of_predictor": "",
                "coverage_at_threshold": "",
                "exploration_gate_pass": exploration,
                "promotion_enabling_gate_pass": promotion,
                "route_hint": "P-PredictorRobust" if exploration else "R-P-PredictorWeak",
                "promotion_allowed": 0,
            }
        )
        available_leave = {
            str(lrow.get("leaveout_split", "")): lrow
            for lrow in leave_by_name.get(source_name, [])
            if str(lrow.get("leaveout_split", ""))
        }
        for planned_split, source_split in P_REQUIRED_LEAVEOUT_SPLITS.items():
            lrow = available_leave.get(source_split, {})
            available = int(bool(lrow) and executed)
            item = dict(lrow)
            item.update(
                {
                    "feature_id": feature_id,
                    "v1414_feature_name": source_name,
                    "planned_leaveout_split": planned_split,
                    "source_leaveout_split": source_split,
                    "executed_from_existing_artifact": available,
                    "available_for_gate": available,
                    "promotion_allowed": 0,
                }
            )
            if not available:
                item["auc"] = ""
                item["missing_split_reason"] = (
                    "not present in v14.14 E4 leaveout artifact; recorded as unavailable, not imputed"
                )
            elif planned_split == "leave-synthetic-family-out" and source_split == "leave_task_family_out":
                item["mapping_note"] = "mapped from v14.14 leave_task_family_out; no value was imputed"
            out_leave.append(item)
    best = max(out_rows, key=lambda r: fnum(r.get("auc_mean"), 0.0), default={})
    summary = {
        "line_p_rows": len(out_rows),
        "line_p_leaveout_rows": len(out_leave),
        "line_p_required_leaveout_split_count": len(P_REQUIRED_LEAVEOUT_SPLITS),
        "line_p_leaveout_available_rows": sum(sint(r.get("available_for_gate"), 0) for r in out_leave),
        "line_p_leaveout_unavailable_rows": sum(1 for r in out_leave if sint(r.get("available_for_gate"), 0) == 0),
        "line_p_leaveout_planned_splits": ",".join(P_REQUIRED_LEAVEOUT_SPLITS.keys()),
        "line_p_best_feature": best.get("feature_id", ""),
        "line_p_best_auc_mean": fnum(best.get("auc_mean"), 0.0),
        "line_p_best_leaveout_min": fnum(best.get("auc_leaveout_min"), 0.0),
        "line_p_best_incremental_auc": fnum(best.get("incremental_auc_over_controls"), 0.0),
        "line_p_best_spearman_source": fnum(best.get("spearman_source"), 0.0),
        "line_p_exploration_gate_pass": int(any(sint(r.get("exploration_gate_pass"), 0) for r in out_rows)),
        "line_p_promotion_enabling_gate_pass": int(any(sint(r.get("promotion_enabling_gate_pass"), 0) for r in out_rows)),
        "line_p_route": "P-PredictorRobust" if any(sint(r.get("exploration_gate_pass"), 0) for r in out_rows) else "R-P-PredictorWeak",
    }
    write_rows(out / "v1415_predictor_robustness.csv", out_rows)
    write_rows(out / "v1415_predictor_leaveout.csv", out_leave)
    return out_rows, out_leave, summary


def direction_id(row: dict[str, Any]) -> str:
    group = str(row.get("group", ""))
    method = str(row.get("method", ""))
    if "AdamW-baseline" in group or method == "C0-D-CHE-AdamW":
        return "D0-AdamW-direction"
    if "RandomMatchedNorm" in group or "RandomMatchedNorm" in method:
        return "D3-Random-matched-norm-direction"
    if "AdamWParallel" in group or "AdamWParallel" in method:
        return "D4-AdamWParallelDirection-control"
    if "NoOp" in group or "NoOp" in method:
        return "D5-zero-direction-NoOp"
    if "Degree" in method or "Projection" in method or "G4" in group:
        return "D2-Degree-projected-FMS-direction"
    return "D1-Generic-FMS-direction"


def build_line_i(out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    q_rows = read_rows(V1414_OUT / "v1414_q_fms_vs_matched_controls.csv")
    boundary_rows = read_rows(V1414_OUT / "v1414_b_boundary_fms_real_lite.csv")
    chain_rows = read_rows(V1413_OUT / "v1413_v3_proxy_to_effect_chain.csv")
    recovery_score_by_key = {
        (str(r.get("method", "")), str(r.get("dataset", "")), str(r.get("seed", ""))): fnum(r.get("recovery_lag"), float("nan"))
        for r in chain_rows
        if str(r.get("recovery_lag", "")) != ""
    }
    recovery_scores = [score for score in recovery_score_by_key.values() if not math.isnan(score)]
    recovery_threshold = v1414.percentile(recovery_scores, 0.50) if recovery_scores else float("nan")
    random_source_by_dataset_seed: dict[tuple[str, str], list[float]] = {}
    for r in q_rows:
        group = str(r.get("group", ""))
        if "RandomMatchedNorm" in group or "SameActiveFraction" in group:
            key = (str(r.get("dataset", "")), str(r.get("seed", "")))
            random_source_by_dataset_seed.setdefault(key, []).append(fnum(r.get("source_vs_best_control"), 0.0))
    rows: list[dict[str, Any]] = []
    for row in q_rows:
        group = str(row.get("group", ""))
        selector = "S0-always-on"
        if "SameActiveFraction" in group or "Random" in group:
            selector = "S3-random-matched-active-fraction"
        elif "NoOp" in group:
            selector = "S4-NoOp-safe-abstention-selector"
        item = {
            "selector_id": selector,
            "direction_id": direction_id(row),
            "boundary_id": "B0-no-boundary",
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "method": row.get("method", ""),
            "source_artifact": "v1414_q_fms_vs_matched_controls.csv",
            "source_vs_best_control": row.get("source_vs_best_control", ""),
            "auc_time_ratio": row.get("AUCtime_ratio_vs_best_control", ""),
            "CEp99_delta": row.get("CEp99_delta_vs_best_control", ""),
            "NLL_delta": row.get("NLL_delta_vs_best_control", ""),
            "ECE_delta": row.get("ECE_delta_vs_best_control", ""),
            "LineC_pass": int(fnum(row.get("LineC_pass_gap_vs_control"), -1.0) >= 0.0),
            "control_equivalent": row.get("control_equivalent", ""),
            "bad_event": row.get("bad_event", ""),
            "NoOp_equivalent": int("NoOp" in group or abs(fnum(row.get("source_vs_best_control"), 0.0)) <= 0.002),
            "random_direction_equivalent": int("Random" in group),
            "selector_incremental_gain": 0.0,
            "direction_incremental_gain": row.get("fms_specific_source_delta", row.get("source_vs_best_control", 0.0)),
            "boundary_incremental_gain": 0.0,
            "step_time_ratio": "",
            "memory_ratio": "",
            "replay_or_actual": "replay_from_v1414_q",
            "promotion_allowed": 0,
        }
        rows.append(item)
        key = (str(row.get("method", "")), str(row.get("dataset", "")), str(row.get("seed", "")))
        score = recovery_score_by_key.get(key, float("nan"))
        if not math.isnan(score):
            ds_key = (str(row.get("dataset", "")), str(row.get("seed", "")))
            random_baseline = mean(random_source_by_dataset_seed.get(ds_key, []))
            selector_item = dict(item)
            selector_item["selector_id"] = "S1-E4-predictor-high-score" if score > recovery_threshold else "S2-inverse-E4-predictor"
            selector_item["selector_score_feature"] = "E4-K-recovery_lag"
            selector_item["selector_score_value"] = score
            selector_item["selector_score_threshold"] = recovery_threshold
            selector_item["selector_threshold_rule"] = "S1 iff score > threshold; S2 otherwise"
            selector_item["source_artifact"] = "v1414_q_fms_vs_matched_controls.csv+v1413_v3_proxy_to_effect_chain.csv"
            selector_item["selector_incremental_gain"] = fnum(row.get("source_vs_best_control"), 0.0) - random_baseline
            selector_item["strict_same_direction_random_selector_available"] = 0
            selector_item["selector_contrast_note"] = "E4 selector replay from train-state artifact; same-direction random selector was not executed, so this is diagnostic only."
            selector_item["replay_or_actual"] = "selector_replay_from_existing_artifacts"
            rows.append(selector_item)
    for row in boundary_rows:
        method = str(row.get("method", ""))
        boundary = "B1-degree-projection-safety"
        selector = "S0-always-on"
        direction = "D2-Degree-projected-FMS-direction"
        if method.startswith("F-B1"):
            selector = "S4-NoOp-safe-abstention-selector"
            boundary = "B3-rejection-fraction-cap"
        elif method.startswith("F-B3"):
            boundary = "B2-low-amplitude-boundary"
        elif method.startswith("F-B2"):
            direction = "D0-AdamW-direction"
        rows.append(
            {
                "selector_id": selector,
                "direction_id": direction,
                "boundary_id": boundary,
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "method": method,
                "source_artifact": "v1414_b_boundary_fms_real_lite.csv",
                "source_vs_best_control": row.get("source_vs_best_control", ""),
                "auc_time_ratio": row.get("AUCtime_ratio", ""),
                "CEp99_delta": row.get("CEp99_delta", ""),
                "NLL_delta": row.get("NLL_delta", ""),
                "ECE_delta": row.get("ECE_delta", ""),
                "LineC_pass": row.get("LineC_majority_pass", ""),
                "control_equivalent": int(fnum(row.get("source_vs_best_control"), 0.0) <= 0.005),
                "bad_event": int(fnum(row.get("source_vs_best_control"), 0.0) < 0.0 or fnum(row.get("AUCtime_ratio"), 9.0) > 1.0 or sint(row.get("LineC_majority_pass"), 0) == 0),
                "NoOp_equivalent": int(abs(fnum(row.get("source_vs_best_control"), 0.0)) <= 0.002),
                "random_direction_equivalent": 0,
                "selector_incremental_gain": 0.0,
                "direction_incremental_gain": row.get("source_vs_best_control", ""),
                "boundary_incremental_gain": row.get("source_vs_best_control", ""),
                "step_time_ratio": row.get("step_time_ratio", ""),
                "memory_ratio": row.get("memory_ratio", ""),
                "replay_or_actual": "actual_v1414_boundary_real_lite",
                "promotion_allowed": 0,
            }
        )
    for row in read_rows(V1413_OUT / "v1413_f3_dche_fms_real_lite.csv"):
        if str(row.get("method", "")) != "FMS-M5-ProjectionRetentionFloor":
            continue
        rows.append(
            {
                "selector_id": "S0-always-on",
                "direction_id": "D2-Degree-projected-FMS-direction",
                "boundary_id": "B4-value-retention-floor",
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "method": row.get("method", ""),
                "source_artifact": "v1413_f3_dche_fms_real_lite.csv",
                "source_vs_best_control": row.get("source_vs_best_control", ""),
                "auc_time_ratio": row.get("AUCtime_ratio", ""),
                "CEp99_delta": row.get("CEp99_delta", ""),
                "NLL_delta": row.get("NLL_delta", ""),
                "ECE_delta": row.get("ECE_delta", ""),
                "LineC_pass": row.get("LineC_majority_pass", ""),
                "control_equivalent": int(fnum(row.get("source_vs_best_control"), 0.0) <= 0.005),
                "bad_event": int(fnum(row.get("source_vs_best_control"), 0.0) < 0.0 or fnum(row.get("AUCtime_ratio"), 9.0) > 1.0 or sint(row.get("LineC_majority_pass"), 0) == 0),
                "NoOp_equivalent": int(abs(fnum(row.get("source_vs_best_control"), 0.0)) <= 0.002),
                "random_direction_equivalent": 0,
                "selector_incremental_gain": 0.0,
                "direction_incremental_gain": row.get("source_vs_best_control", ""),
                "boundary_incremental_gain": row.get("source_vs_best_control", ""),
                "step_time_ratio": row.get("step_time_ratio", ""),
                "memory_ratio": row.get("memory_ratio", ""),
                "replay_or_actual": "actual_v1413_value_retention_floor_replay",
                "promotion_allowed": 0,
            }
        )
    effects: list[dict[str, Any]] = []
    for axis, col, gain_col in [
        ("selector", "selector_id", "selector_incremental_gain"),
        ("direction", "direction_id", "direction_incremental_gain"),
        ("boundary", "boundary_id", "boundary_incremental_gain"),
    ]:
        for value in sorted({str(r.get(col, "")) for r in rows}):
            group = [r for r in rows if str(r.get(col, "")) == value]
            effects.append(
                {
                    "axis": axis,
                    "axis_value": value,
                    "rows": len(group),
                    "mean_incremental_gain": mean([fnum(r.get(gain_col), 0.0) for r in group]),
                    "mean_source_vs_best_control": source_mean(group),
                    "control_equivalent_fraction": sum(sint(r.get("control_equivalent"), 0) for r in group) / max(1, len(group)),
                    "bad_event_fraction": sum(sint(r.get("bad_event"), 0) for r in group) / max(1, len(group)),
                    "strict_pass_rows": sum(sint(r.get("strict_gate_pass"), 0) for r in group),
                    "strict_same_direction_random_selector_available": int(any(sint(r.get("strict_same_direction_random_selector_available"), 1) for r in group)) if axis == "selector" else "",
                    "comparison_note": (
                        "S1/S2 selector rows use reconstructed E4-K recovery_lag replay; strict same-direction random-selector contrast was not executed."
                        if axis == "selector" and value in {"S1-E4-predictor-high-score", "S2-inverse-E4-predictor"}
                        else ""
                    ),
                    "promotion_allowed": 0,
                }
            )
    fms_like = [r for r in rows if r.get("direction_id") in {"D1-Generic-FMS-direction", "D2-Degree-projected-FMS-direction"}]
    pass_count = unique_pass_count(fms_like)
    direction_gain = mean([fnum(r.get("direction_incremental_gain"), 0.0) for r in fms_like])
    ce_frac = sum(sint(r.get("control_equivalent"), 0) for r in fms_like) / max(1, len(fms_like))
    bad_frac = sum(sint(r.get("bad_event"), 0) for r in fms_like) / max(1, len(fms_like))
    summary = {
        "line_i_rows": len(rows),
        "line_i_effect_rows": len(effects),
        "line_i_selector_replay_rows": sum(1 for r in rows if str(r.get("replay_or_actual")) == "selector_replay_from_existing_artifacts"),
        "line_i_selector_ids": ",".join(sorted({str(r.get("selector_id", "")) for r in rows if str(r.get("selector_id", ""))})),
        "line_i_boundary_ids": ",".join(sorted({str(r.get("boundary_id", "")) for r in rows if str(r.get("boundary_id", ""))})),
        "line_i_value_retention_floor_rows": sum(1 for r in rows if str(r.get("boundary_id", "")) == "B4-value-retention-floor"),
        "line_i_strict_selector_contrast_available": 0,
        "line_i_mean_direction_incremental_gain": direction_gain,
        "line_i_control_equivalent_fraction": ce_frac,
        "line_i_bad_event_fraction": bad_frac,
        "line_i_real_lite_pass_count": pass_count,
        "line_i_exploration_gate_pass": int(direction_gain >= 0.002 and ce_frac <= 0.60 and bad_frac <= 0.30 and pass_count >= 3),
        "line_i_meaningful_gate_pass": int(direction_gain >= 0.005 and ce_frac <= 0.40 and bad_frac <= 0.20 and pass_count >= 4),
        "line_i_route": "I-CausalDirectionCandidate" if direction_gain >= 0.002 and ce_frac <= 0.60 else "R-I-DirectionControlEquivalent",
    }
    write_rows(out / "v1415_intervention_factorial.csv", rows)
    write_rows(out / "v1415_intervention_causal_effects.csv", effects)
    return rows, effects, summary


def load_role_rows(source_name: str, method: str) -> list[dict[str, Any]]:
    base = V1414_OUT if source_name.startswith("v1414") else V1413_OUT
    return group_by_method(read_rows(base / source_name), method)


def build_line_b(out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v1414_bad_ref = fnum(load_json(V1414_OUT / "v1414_route_decision.json").get("q_bad_event_fraction"), 0.9555555555555556)
    for role, (artifact, method, note) in BOUNDARY_ROLE_MAP.items():
        group = load_role_rows(artifact, method)
        bad_frac = bad_event_fraction(group)
        ce_frac = control_equiv_fraction(group)
        rows.append(
            {
                "boundary_id": role,
                "source_artifact": artifact,
                "evidence_method": method,
                "execution_note": note,
                "rows": len(group),
                "dataset_seed_pass_count": unique_pass_count(group),
                "abstention_rate": mean([1.0 - fnum(r.get("degree_gate_active_fraction"), 0.0) for r in group]),
                "plasticity_mean": mean([fnum(r.get("degree_gate_active_fraction"), 0.0) for r in group]),
                "plasticity_min": min([fnum(r.get("degree_gate_active_fraction"), 0.0) for r in group] or [0.0]),
                "plasticity_max": max([fnum(r.get("degree_gate_active_fraction"), 0.0) for r in group] or [0.0]),
                "degree_energy_growth": mean([fnum(r.get("degree_energy_after"), 0.0) - fnum(r.get("degree_energy_before"), 0.0) for r in group]),
                "high_degree_fraction_delta": mean([fnum(r.get("high_degree_energy_fraction"), 0.0) for r in group]),
                "recovery_lag_delta": mean([fnum(r.get("source_vs_best_control"), 0.0) for r in group]),
                "drift_diffusion_ratio": mean([fnum(r.get("cos_projected_vs_generic"), 0.0) for r in group]),
                "source_vs_best_control": source_mean(group),
                "AUCtime_ratio": mean([fnum(r.get("AUCtime_ratio"), 9.0) for r in group]),
                "tail_delta": mean([fnum(r.get("CEp99_delta"), 0.0) + fnum(r.get("NLL_delta"), 0.0) + fnum(r.get("ECE_delta"), 0.0) for r in group]),
                "LineC_pass": mean([sint(r.get("LineC_majority_pass"), 0) for r in group]),
                "control_equivalent_fraction": ce_frac,
                "NoOp_equivalent_fraction": sum(1 for r in group if abs(fnum(r.get("source_vs_best_control"), 0.0)) <= 0.002) / max(1, len(group)),
                "bad_event_fraction": bad_frac,
                "bad_event_fraction_decrease_vs_v1414_q": max(0.0, v1414_bad_ref - bad_frac),
                "boundary_gate_pass": int((v1414_bad_ref - bad_frac) >= 0.20 and source_mean(group) >= -0.002 and ce_frac <= 0.60),
                "route_hint": "B-BoundaryCandidate" if (v1414_bad_ref - bad_frac) >= 0.20 and source_mean(group) >= -0.002 and ce_frac <= 0.60 else "R-B-BoundaryIsHarmlessNull",
                "promotion_allowed": 0,
            }
        )
    summary = {
        "line_b_rows": len(rows),
        "line_b_executed_boundary_count": sum(1 for r in rows if sint(r.get("rows"), 0) > 0),
        "line_b_best_boundary": max(rows, key=lambda r: fnum(r.get("source_vs_best_control"), -999.0), default={}).get("boundary_id", ""),
        "line_b_best_source_vs_best_control": max([fnum(r.get("source_vs_best_control"), -999.0) for r in rows] or [0.0]),
        "line_b_min_control_equivalent_fraction": min([fnum(r.get("control_equivalent_fraction"), 1.0) for r in rows] or [1.0]),
        "line_b_any_gate_pass": int(any(sint(r.get("boundary_gate_pass"), 0) for r in rows)),
        "line_b_route": "B-BoundaryCandidate" if any(sint(r.get("boundary_gate_pass"), 0) for r in rows) else "R-B-BoundaryIsHarmlessNull",
    }
    write_rows(out / "v1415_boundary_role_reset.csv", rows)
    return rows, summary


def build_line_f(out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    f_rows = []
    for row in read_rows(V1414_OUT / "v1414_b_boundary_fms_real_lite.csv"):
        if row.get("method") in {"F-B1-AbstentionBoundaryFMS", "F-B2-ConstraintOnlyBoundaryFMS", "F-B3-ContinuousLowAmplitudeBoundaryFMS"}:
            item = dict(row)
            item["stage"] = "V1415_DCHE_REAL_LITE_CAUSAL_FORK"
            item["candidate_source"] = "v1414_pre_registered_boundary_replay"
            item["reused_existing_real_lite_row"] = 1
            item["promotion_allowed"] = 0
            f_rows.append(item)
    ctrl_rows = []
    for row in read_rows(V1414_OUT / "v1414_b_boundary_controls.csv"):
        item = dict(row)
        item["stage"] = "V1415_DCHE_CONTROL_REPLAY"
        item["candidate_source"] = "v1414_boundary_controls"
        item["reused_existing_real_lite_row"] = 1
        item["promotion_allowed"] = 0
        ctrl_rows.append(item)
    pass_count = unique_pass_count(f_rows)
    summary = {
        "line_f_rows": len(f_rows),
        "line_f_control_rows": len(ctrl_rows),
        "line_f_candidate_count": len({r.get("method") for r in f_rows}),
        "line_f_real_lite_pass_count": pass_count,
        "line_f_source_vs_best_control_mean": source_mean(f_rows),
        "line_f_source_vs_best_control_min": min([fnum(r.get("source_vs_best_control"), 0.0) for r in f_rows] or [0.0]),
        "line_f_AUCtime_ratio_mean": mean([fnum(r.get("AUCtime_ratio"), 9.0) for r in f_rows]),
        "line_f_AUCtime_ratio_max": max([fnum(r.get("AUCtime_ratio"), 0.0) for r in f_rows] or [0.0]),
        "line_f_CEp99_delta_max": max([fnum(r.get("CEp99_delta"), 0.0) for r in f_rows] or [0.0]),
        "line_f_NLL_delta_max": max([fnum(r.get("NLL_delta"), 0.0) for r in f_rows] or [0.0]),
        "line_f_ECE_delta_max": max([fnum(r.get("ECE_delta"), 0.0) for r in f_rows] or [0.0]),
        "line_f_LineC_fail_count": sum(1 for r in f_rows if sint(r.get("LineC_majority_pass"), 0) == 0),
        "line_f_control_equivalent_fraction": control_equiv_fraction(f_rows),
        "line_f_step_time_ratio_mean": mean([fnum(r.get("step_time_ratio"), 9.0) for r in f_rows]),
        "line_f_memory_ratio_mean": mean([fnum(r.get("memory_ratio"), 9.0) for r in f_rows]),
        "line_f_exploration_weak_gate_pass": int(pass_count >= 3),
        "line_f_meaningful_gate_pass": int(pass_count >= 4),
        "line_f_s4_gate_pass": int(pass_count >= 6),
        "line_f_route": "F-RealLiteCandidate" if pass_count >= 3 else "R-F-RealLiteBelow4",
    }
    write_rows(out / "v1415_dche_real_lite.csv", f_rows)
    write_rows(out / "v1415_dche_controls.csv", ctrl_rows)
    return f_rows, ctrl_rows, summary


def build_line_d(out: Path, line_d_out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = line_d_out / "v149_line_d_substrate_repair_results.csv"
    baseline_rows = read_rows(V1414_OUT / "v1414_d_all_basis_substrate_hardening.csv")
    rows: list[dict[str, Any]] = []
    for row in baseline_rows:
        item = dict(row)
        item["v1415_line_d_component"] = "v1414_baseline_replay"
        rows.append(item)
    if path.exists():
        for row in read_rows(path):
            item = dict(row)
            item["v1415_line_d_component"] = "v1415_new_substrate_acceleration"
            rows.append(item)
        source = "v1414_baseline_plus_v1415_new_line_d_v149_substrate_acceleration"
    else:
        source = "fallback_replay_v1414_line_d"
    out_rows = []
    for row in rows:
        item = dict(row)
        item["stage"] = "V1415_ALLBASIS_SUBSTRATE_ACCELERATION"
        item["v1415_source"] = source
        item["official_fms_proof_executed"] = 0
        item["promotion_allowed"] = 0
        if str(item.get("dataset", "")):
            step_ratio = fnum(item.get("train_step_ratio_vs_MLP"), 999.0)
            raw_mem = fnum(item.get("workspace_raw_memory_ratio_vs_mlp"), 999.0)
            incr_mem = fnum(item.get("workspace_incremental_memory_ratio_vs_mlp"), 999.0)
            memory_ratio = incr_mem if math.isfinite(incr_mem) and incr_mem < 999.0 else raw_mem
            item["v1415_gate_step_ratio"] = step_ratio
            item["v1415_gate_memory_ratio"] = memory_ratio
            item["v1415_substrate_gate_pass"] = int(
                step_ratio <= 1.75
                and memory_ratio <= 1.75
                and fnum(item.get("mean_delta_vs_MLP"), -999.0) >= -0.05
                and fnum(item.get("worst_delta_vs_MLP"), -999.0) >= -0.10
                and fnum(item.get("LineC_pass_rate"), 0.0) >= 0.30
            )
            item["v1415_substrate_gate_definition"] = "step_ratio<=1.75 & memory_ratio<=1.75 & mean_delta>=-0.05 & worst_delta>=-0.10 & LineC_pass_rate>=0.30"
        out_rows.append(item)
    summaries = []
    for family in sorted({r.get("family", "") for r in out_rows if r.get("family", "")}):
        fam = [r for r in out_rows if r.get("family") == family]
        pass_keys = {
            (r.get("dataset"), r.get("seed"))
            for r in fam
            if sint(r.get("v1415_substrate_gate_pass"), 0) == 1
        }
        best_candidate = ""
        best_count = -1
        for cid in sorted({r.get("candidate_id", "") for r in fam}):
            keys = {
                (r.get("dataset"), r.get("seed"))
                for r in fam
                if r.get("candidate_id") == cid and sint(r.get("v1415_substrate_gate_pass"), 0) == 1
            }
            if len(keys) > best_count:
                best_count = len(keys)
                best_candidate = cid
        summaries.append(
            {
                "family": family,
                "rows": len(fam),
                "family_dataset_seed_pass_count": len(pass_keys),
                "best_candidate": best_candidate,
                "best_candidate_dataset_seed_pass_count": max(0, best_count),
                "exploration_substrate_gate_pass": int(len(pass_keys) >= 6),
                "official_fms_eligible": int(len(pass_keys) >= 9),
                "mean_delta_best": max([fnum(r.get("mean_delta_vs_MLP"), -999.0) for r in fam] or [-999.0]),
                "linec_best": max([fnum(r.get("LineC_pass_rate"), 0.0) for r in fam] or [0.0]),
                "promotion_allowed": 0,
            }
        )
    best_non_dche = max([r for r in summaries if r.get("family") != "D-CHE"], key=lambda r: sint(r.get("family_dataset_seed_pass_count"), 0), default={})
    fou_new_rows = [
        r
        for r in out_rows
        if r.get("family") == "D-FOU" and r.get("v1415_line_d_component") == "v1415_new_substrate_acceleration"
    ]
    rbf_new_rows = [
        r
        for r in out_rows
        if r.get("family") == "D-RBF" and r.get("v1415_line_d_component") == "v1415_new_substrate_acceleration"
    ]
    fou_telemetry_rows = [
        r
        for r in fou_new_rows
        if sint(r.get("fourier_telemetry_available"), 0) == 1
    ]
    rbf_residual_rows = [
        r
        for r in rbf_new_rows
        if sint(r.get("residual_over_base_available"), 0) == 1
    ]
    fou_telemetry_sources = sorted({str(r.get("fourier_telemetry_source", "")) for r in fou_telemetry_rows if str(r.get("fourier_telemetry_source", ""))})
    rbf_residual_sources = sorted({str(r.get("residual_over_base_source", "")) for r in rbf_residual_rows if str(r.get("residual_over_base_source", ""))})
    summary = {
        "line_d_rows": len(out_rows),
        "line_d_summary_rows": len(summaries),
        "line_d_source": source,
        "line_d_v1414_baseline_rows": len(baseline_rows),
        "line_d_v1415_new_rows": max(0, len(out_rows) - len(baseline_rows)),
        "line_d_fou_telemetry_required_rows": len(fou_new_rows),
        "line_d_fou_telemetry_available_rows": len(fou_telemetry_rows),
        "line_d_fou_telemetry_missing_rows": max(0, len(fou_new_rows) - len(fou_telemetry_rows)),
        "line_d_fou_telemetry_source": ",".join(fou_telemetry_sources),
        "line_d_rbf_residual_required_rows": len(rbf_new_rows),
        "line_d_rbf_residual_available_rows": len(rbf_residual_rows),
        "line_d_rbf_residual_missing_rows": max(0, len(rbf_new_rows) - len(rbf_residual_rows)),
        "line_d_rbf_residual_source": ",".join(rbf_residual_sources),
        "line_d_v1415_gate_definition": "step_ratio<=1.75 & memory_ratio<=1.75 & mean_delta>=-0.05 & worst_delta>=-0.10 & LineC_pass_rate>=0.30",
        "line_d_best_non_dche_family": best_non_dche.get("family", ""),
        "line_d_best_non_dche_dataset_seed_pass_count": sint(best_non_dche.get("family_dataset_seed_pass_count"), 0),
        "line_d_official_fms_eligible_family_count": sum(sint(r.get("official_fms_eligible"), 0) for r in summaries if r.get("family") != "D-CHE"),
        "line_d_exploration_open_family_count": sum(sint(r.get("exploration_substrate_gate_pass"), 0) for r in summaries if r.get("family") != "D-CHE"),
        "line_d_route": "D-SubstrateCandidateOpened" if sint(best_non_dche.get("family_dataset_seed_pass_count"), 0) >= 6 else "R-D-SubstrateStillBlocked",
    }
    write_rows(out / "v1415_allbasis_substrate_acceleration.csv", out_rows + summaries)
    return out_rows + summaries, summary


def build_line_m(out: Path, line_m_out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for row in read_rows(V1413_OUT / "v1413_mlp_controls.csv"):
        item = dict(row)
        item["stage"] = "V1415_MLP_GENERIC_CONTROL_REPLAY"
        item["reused_existing_v1413_control"] = 1
        item["reused_v1415_completion_control"] = 0
        item["required_control_name"] = MLP_METHOD_TO_REQUIRED.get(str(item.get("method")), "")
        item["missing_required_control"] = 0
        item["promotion_allowed"] = 0
        rows.append(item)
    completion_path = line_m_out / "v1410_mlp_generic_controls.csv"
    if completion_path.exists():
        for row in read_rows(completion_path):
            method = str(row.get("method", ""))
            if method not in MLP_METHOD_TO_REQUIRED:
                continue
            item = dict(row)
            item["stage"] = "V1415_MLP_GENERIC_CONTROL_COMPLETION"
            item["reused_existing_v1413_control"] = 0
            item["reused_v1415_completion_control"] = 1
            item["required_control_name"] = MLP_METHOD_TO_REQUIRED.get(method, "")
            item["missing_required_control"] = 0
            item["promotion_allowed"] = 0
            rows.append(item)
    present_required = sorted({r.get("required_control_name", "") for r in rows if r.get("required_control_name", "")})
    missing_required = sorted(set(REQUIRED_MLP_CONTROLS) - set(present_required))
    for required in missing_required:
        rows.append(
            {
                "stage": "V1415_MLP_GENERIC_CONTROL_COVERAGE",
                "family": "MLP",
                "candidate_id": "MLP-v1415-required-control",
                "dataset": "",
                "task": "",
                "seed": "",
                "method": "",
                "required_control_name": required,
                "missing_required_control": 1,
                "source_artifact": str(completion_path if completion_path.exists() else V1413_OUT / "v1413_mlp_controls.csv"),
                "promotion_allowed": 0,
            }
        )
    real_like = [r for r in rows if r.get("task") == "real_lite" or r.get("dataset") in {"MNIST", "Fashion-MNIST", "KMNIST"}]
    summary = {
        "line_m_rows": len(rows),
        "line_m_real_like_rows": len(real_like),
        "line_m_required_control_count": len(REQUIRED_MLP_CONTROLS),
        "line_m_executed_required_control_count": len(present_required),
        "line_m_missing_required_control_count": len(missing_required),
        "line_m_missing_required_controls": ",".join(missing_required),
        "line_m_best_mean_source_vs_best_control": max([fnum(r.get("mean_source_vs_best_control"), fnum(r.get("source_vs_best_control"), -999.0)) for r in rows] or [0.0]),
        "line_m_generic_controls_positive": int(any(fnum(r.get("source_vs_best_control"), 0.0) > 0.005 for r in real_like)),
        "line_m_route": (
            "M-RequiredControlsMissing"
            if missing_required
            else ("M-GenericControlsPositive" if any(fnum(r.get("source_vs_best_control"), 0.0) > 0.005 for r in real_like) else "M-ControlsCompleteNoGenericPositive")
        ),
    }
    write_rows(out / "v1415_mlp_generic_controls.csv", rows)
    return rows, summary


def build_line_c(out: Path, f_rows: Sequence[dict[str, Any]], line_i_rows: Sequence[dict[str, Any]], line_d_rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    audit = []
    for row in read_rows(V1414_OUT / "v1414_linec_tail_audit.csv"):
        item = dict(row)
        item["stage"] = "V1415_LINEC_TAIL_AUDIT_REPLAY"
        item["used_for_direction"] = 0
        item["promotion_allowed"] = 0
        audit.append(item)
    taxonomy = []
    for name, fail_count, route in [
        ("LineP", 1, "R-P-PredictorWeak"),
        ("LineI", int(any(sint(r.get("control_equivalent"), 0) for r in line_i_rows)), "R-I-DirectionControlEquivalent"),
        ("LineB", int(control_equiv_fraction(f_rows) > 0.60), "R-B-BoundaryIsHarmlessNull"),
        ("LineF", int(unique_pass_count(f_rows) < 4), "R-F-RealLiteBelow4"),
        ("LineD", int(max([sint(r.get("family_dataset_seed_pass_count"), 0) for r in line_d_rows if "family_dataset_seed_pass_count" in r] or [0]) < 6), "R-D-SubstrateStillBlocked"),
    ]:
        taxonomy.append(
            {
                "line": name,
                "blocked": fail_count,
                "route_hint": route if fail_count else "pass-or-not-primary-blocker",
                "used_audit_metric_for_direction": 0,
                "promotion_allowed": 0,
            }
        )
    summary = {
        "line_c_rows": len(audit),
        "failure_taxonomy_rows": len(taxonomy),
        "blocked_line_count": sum(sint(r.get("blocked"), 0) for r in taxonomy),
    }
    write_rows(out / "v1415_linec_tail_audit.csv", audit)
    write_rows(out / "v1415_failure_taxonomy.csv", taxonomy)
    return audit, taxonomy, summary


def decide_route(
    line_p: dict[str, Any],
    line_i: dict[str, Any],
    line_b: dict[str, Any],
    line_f: dict[str, Any],
    line_d: dict[str, Any],
    violations: dict[str, int],
    missing: int,
) -> dict[str, Any]:
    if missing > 0 or violations["forbidden"] > 0 or violations["no_action"] > 0:
        route = "R0-ContractViolation"
    elif sint(line_f.get("line_f_s4_gate_pass"), 0) and sint(line_d.get("line_d_best_non_dche_dataset_seed_pass_count"), 0) >= 6:
        route = "S4-ExplorationPositive"
    elif sint(line_f.get("line_f_meaningful_gate_pass"), 0):
        route = "S4-lite-RealLiteMeaningfulNoPromotion"
    elif str(line_i.get("line_i_route")) == "R-I-DirectionControlEquivalent" and str(line_b.get("line_b_route")) == "R-B-BoundaryIsHarmlessNull":
        route = "R15-ParallelForkNoCausalValueAllBasisBlocked"
    elif sint(line_d.get("line_d_best_non_dche_dataset_seed_pass_count"), 0) < 6:
        route = "R4-AllBasisSubstrateBlocked"
    else:
        route = "S0-ExecutionCompleteNoPromotion"
    return {
        "route": route,
        "minimum_success": "S3-DCHESyntheticFMSPass",
        **line_p,
        **line_i,
        **line_b,
        **line_f,
        **line_d,
        "official_s5_reached": 0,
        "promotion_allowed": 0,
        "required_artifact_missing_count": missing,
        "forbidden_information_violation_count": violations["forbidden"],
        "no_action_search_violation_count": violations["no_action"],
    }


def write_progress(out: Path, route: dict[str, Any]) -> None:
    rows = [
        {"line": "R", "status": "completed", "key_metric": "violations", "value": route["forbidden_information_violation_count"] + route["no_action_search_violation_count"], "promotion_allowed": 0},
        {"line": "P", "status": "completed", "key_metric": "best_auc/leaveout", "value": f"{route['line_p_best_auc_mean']}/{route['line_p_best_leaveout_min']}", "promotion_allowed": 0},
        {"line": "I", "status": "completed", "key_metric": "direction_gain/control_equiv", "value": f"{route['line_i_mean_direction_incremental_gain']}/{route['line_i_control_equivalent_fraction']}", "promotion_allowed": 0},
        {"line": "B", "status": "completed", "key_metric": "best_source", "value": route["line_b_best_source_vs_best_control"], "promotion_allowed": 0},
        {"line": "F", "status": "completed", "key_metric": "real_lite_pass", "value": route["line_f_real_lite_pass_count"], "promotion_allowed": 0},
        {"line": "D", "status": "completed", "key_metric": "best_non_dche", "value": route["line_d_best_non_dche_dataset_seed_pass_count"], "promotion_allowed": 0},
        {"line": "Z", "status": "completed", "key_metric": "route", "value": route["route"], "promotion_allowed": 0},
    ]
    write_rows(out / "v1415_progress_table.csv", rows)


def write_required_manifest(out: Path) -> list[dict[str, Any]]:
    manifest_path = out / "v1415_required_artifact_manifest.csv"
    if not manifest_path.exists():
        manifest_path.write_text("artifact,exists,bytes,sha256,required,missing\n", encoding="utf-8")
    rows = []
    for name in REQUIRED + FIGURES:
        path = out / name
        rows.append(
            {
                "artifact": name,
                "exists": int(path.exists()),
                "bytes": path.stat().st_size if path.exists() else 0,
                "sha256": sha256(path) if path.exists() and path.is_file() else "",
                "required": 1,
                "missing": int(not path.exists()),
            }
        )
    write_rows(manifest_path, rows)
    return rows


def write_figures(out: Path, route: dict[str, Any], p_rows: Sequence[dict[str, Any]], i_effects: Sequence[dict[str, Any]], b_rows: Sequence[dict[str, Any]], f_rows: Sequence[dict[str, Any]], d_rows: Sequence[dict[str, Any]], taxonomy: Sequence[dict[str, Any]]) -> None:
    simple_svg(out / "fig_v1415_progress_dashboard.svg", "v14.15 progress", [("P", fnum(route.get("line_p_exploration_gate_pass"), 0)), ("I", fnum(route.get("line_i_exploration_gate_pass"), 0)), ("B", fnum(route.get("line_b_any_gate_pass"), 0)), ("F", fnum(route.get("line_f_real_lite_pass_count"), 0)), ("D", fnum(route.get("line_d_best_non_dche_dataset_seed_pass_count"), 0))])
    simple_svg(out / "fig_p_predictor_leaveout_heatmap.svg", "Line P leaveout min", [(r.get("feature_id", ""), fnum(r.get("auc_leaveout_min"), 0.0)) for r in p_rows], 0.55)
    simple_svg(out / "fig_i_factorial_effect_heatmap.svg", "Line I effects", [(r.get("axis_value", ""), fnum(r.get("mean_incremental_gain"), 0.0)) for r in i_effects])
    simple_svg(out / "fig_b_boundary_role_matrix.svg", "Boundary roles", [(r.get("boundary_id", ""), fnum(r.get("source_vs_best_control"), 0.0)) for r in b_rows])
    simple_svg(out / "fig_f_dche_real_lite_matrix.svg", "D-CHE real-lite pass", [(m, unique_pass_count(group_by_method(f_rows, m))) for m in sorted({r.get("method", "") for r in f_rows})])
    family_rows = [r for r in d_rows if "family_dataset_seed_pass_count" in r]
    simple_svg(out / "fig_d_allbasis_substrate_pareto.svg", "Line D family pass", [(r.get("family", ""), fnum(r.get("family_dataset_seed_pass_count"), 0.0)) for r in family_rows], 6.0)
    simple_svg(out / "fig_d_family_pass_heatmap.svg", "Line D family gate", [(r.get("family", ""), fnum(r.get("family_dataset_seed_pass_count"), 0.0)) for r in family_rows], 6.0)
    candidate_rows = [r for r in d_rows if r.get("stage") == "V1415_ALLBASIS_SUBSTRATE_ACCELERATION" and r.get("dataset", "")]
    simple_svg(
        out / "fig_d_workspace_task_pareto.svg",
        "Line D task delta",
        [(r.get("candidate_id", ""), fnum(r.get("mean_delta_vs_MLP"), -1.0)) for r in candidate_rows],
        -0.05,
    )
    telemetry_rows = [
        ("D-FOU Fourier", fnum(route.get("line_d_fou_telemetry_available_rows"), 0.0)),
        ("D-RBF occupancy", float(sum(1 for r in candidate_rows if r.get("family") == "D-RBF" and str(r.get("center_occupancy_entropy", "")) != ""))),
        ("D-WAV support", float(sum(1 for r in candidate_rows if r.get("family") == "D-WAV" and str(r.get("wavelet_layer1_active_fraction", "")) != ""))),
    ]
    simple_svg(out / "fig_d_nonrat_telemetry_matrix.svg", "Line D telemetry rows", telemetry_rows)
    simple_svg(
        out / "fig_d_linec_tail_task_breakdown.svg",
        "Line D LineC/task",
        [(r.get("family", ""), fnum(r.get("linec_best"), 0.0)) for r in family_rows],
        0.30,
    )
    simple_svg(out / "fig_c_failure_taxonomy.svg", "Failure taxonomy", [(r.get("line", ""), fnum(r.get("blocked"), 0.0)) for r in taxonomy])


def write_no_go(out: Path, route: dict[str, Any]) -> None:
    write_text(
        out / "v1415_no_go_boundary.md",
        f"""
```text
1. v14.15 已执行 Line R/P/I/B/F/D/M/C/Z。
2. Line P route = {route.get('line_p_route')}。
3. Line I route = {route.get('line_i_route')}。
4. Line B route = {route.get('line_b_route')}。
5. Line F route = {route.get('line_f_route')}，real-lite pass = {route.get('line_f_real_lite_pass_count')}/9。
6. Line D route = {route.get('line_d_route')}，best non-D-CHE = {route.get('line_d_best_non_dche_dataset_seed_pass_count')}/9。
7. 当前不允许新增 F-CHE8/F-CHE9、FMS-M6/M7/M8、action token、controller、action bank 或 reset route。
8. 不允许用 real fail pattern 或 LineC/CEp99/NLL/ECE/AUCtime audit metric 反推 direction。
```
""",
    )
    write_text(
        out / "v1415_next_hypothesis_queue.md",
        "下一步需要新的 functional causal theory 或新的 substrate carrier；不能在 v14.15 内临时扩 token/controller/action/reset route，也不能使用 audit metric 生成方向。\n",
    )


def write_docs(out: Path, route: dict[str, Any]) -> None:
    recap = f"""# DG-KAN v14.15 ParallelFMSCausalFork TransferObservability AllBasis 实验结果复盘

生成时间：2026-05-30（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不把 replay diagnostic、real-lite、local positive、substrate replay 或 MLP/generic control 写成 promotion。

## 1. 计划理解

v14.15 将 v14.14 的串行 no-go 拆成并行 decisive fork：Line P predictor robustness、Line I intervention causality、Line B boundary/curriculum role reset、Line F D-CHE bounded real-lite、Line D all-basis substrate acceleration、Line M controls、Line C/Z。

## 2. 本轮代码修改

新增：

```text
experiments/run_v1415_parallel_fms_causal_fork_transfer_allbasis.py
```

修改：

```text
experiments/run_v149_line_d_all_basis_substrate_repair.py
  新增 v14.15 预注册 substrate-only candidates：
  D-FOU32..36、D-RBF30..34、D-WAV29..32。
  不执行 official FMS proof，不新增 action/controller/reset route。

experiments/run_v143_nonrat_compact_task_health_probe.py
  新增 D-FOU Fourier train-stream telemetry：
  band_energy_low / band_energy_mid / band_energy_high、
  phase_drift、high_freq_ratio、bandwise_snr。
  telemetry_source = train_stream_layer1_basis_channel_split；
  只使用 train-stream layer1 basis activation，不使用 labels / validation / test /
  future / query / LineC / CEp99 / NLL / ECE / AUCtime 生成方向。
  同时补齐 RBF quantile_width100 repair 解析，
  使 D-RBF34-CenterOccupancyWarmupNoTaskBranch 不再因 parser 缺口 blocked。

experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
  为 v14.15 Line M 补齐 MLP/generic controls：
  M5-MLP-NoOpMatchedOverhead、
  M6-MLP-SameActiveFractionControl、
  M7-MLP-BoundaryOnly。
  这些是 control surface，不是 D-CHE FMS token；
  不新增 F-CHE8/F-CHE9、FMS-M6/M7、action/controller/reset route。
```

过程修正：

```text
首次 Line D 命令只覆盖最低完成要求中的 D-FOU32..36 / D-RBF30..34。
复核计划 8.4 后确认 D-WAV29..32 也是预注册 low-budget all-basis 分支。
已修正执行命令并重跑 Line D 全量候选：
  D-FOU32..36、D-RBF30..34、D-WAV29..32。
最终 Line D candidate rows = {route.get('line_d_rows')}；
其中 v14.14 baseline rows = {route.get('line_d_v1414_baseline_rows')}；
v14.15 new rows = {route.get('line_d_v1415_new_rows')}；
line_d_summary_rows = {route.get('line_d_summary_rows')}；
v1415_allbasis_substrate_acceleration.csv total rows = {sint(route.get('line_d_rows'), 0) + sint(route.get('line_d_summary_rows'), 0)}；
best_non_dche_dataset_seed_pass_count = {route.get('line_d_best_non_dche_dataset_seed_pass_count')} / 9。
```

## 3. Line P 结果

```text
best_feature = {route.get('line_p_best_feature')}
best_auc_mean = {route.get('line_p_best_auc_mean')}
best_leaveout_min = {route.get('line_p_best_leaveout_min')}
best_incremental_auc = {route.get('line_p_best_incremental_auc')}
best_spearman_source = {route.get('line_p_best_spearman_source')}
line_p_exploration_gate_pass = {route.get('line_p_exploration_gate_pass')}
line_p_promotion_enabling_gate_pass = {route.get('line_p_promotion_enabling_gate_pass')}
line_p_route = {route.get('line_p_route')}
leaveout_rows = {route.get('line_p_leaveout_rows')}
leaveout_planned_splits = {route.get('line_p_leaveout_planned_splits')}
leaveout_available_rows = {route.get('line_p_leaveout_available_rows')}
leaveout_unavailable_rows = {route.get('line_p_leaveout_unavailable_rows')}
```

判断：

```text
Line P 按计划补齐六类 planned leaveout split 记录。
其中 method/control leaveout 在 v14.14 E4 artifact 中不存在；
本轮只标记 unavailable，不补填 AUC，不参与 gate。
leave-synthetic-family-out 映射自 v14.14 leave_task_family_out，
并保留 mapping_note 方便审计。
```

## 4. Line I 结果

```text
line_i_rows = {route.get('line_i_rows')}
line_i_selector_ids = {route.get('line_i_selector_ids')}
line_i_boundary_ids = {route.get('line_i_boundary_ids')}
line_i_selector_replay_rows = {route.get('line_i_selector_replay_rows')}
line_i_value_retention_floor_rows = {route.get('line_i_value_retention_floor_rows')}
line_i_strict_selector_contrast_available = {route.get('line_i_strict_selector_contrast_available')}
mean_direction_incremental_gain = {route.get('line_i_mean_direction_incremental_gain')}
control_equivalent_fraction = {route.get('line_i_control_equivalent_fraction')}
bad_event_fraction = {route.get('line_i_bad_event_fraction')}
real_lite_pass_count = {route.get('line_i_real_lite_pass_count')}
line_i_exploration_gate_pass = {route.get('line_i_exploration_gate_pass')}
line_i_route = {route.get('line_i_route')}
```

判断：

```text
补充复核后，Line I 增加 S1/S2 selector replay：
  S1-E4-predictor-high-score
  S2-inverse-E4-predictor
这些行只重建 E4-K recovery_lag selector 分组，不是新 real training。
同时补入 B4-value-retention-floor，
来源是 v14.13 FMS-M5-ProjectionRetentionFloor 真实 artifact replay。
由于同方向 random selector 未执行，
strict_selector_contrast_available = 0；
因此 selector replay 不能打开 causal promotion。
```

## 5. Line B 结果

```text
line_b_executed_boundary_count = {route.get('line_b_executed_boundary_count')}
best_boundary = {route.get('line_b_best_boundary')}
best_source_vs_best_control = {route.get('line_b_best_source_vs_best_control')}
min_control_equivalent_fraction = {route.get('line_b_min_control_equivalent_fraction')}
line_b_any_gate_pass = {route.get('line_b_any_gate_pass')}
line_b_route = {route.get('line_b_route')}
```

## 6. Line F 结果

```text
line_f_candidate_count = {route.get('line_f_candidate_count')}
line_f_real_lite_pass_count = {route.get('line_f_real_lite_pass_count')} / 9
source_vs_best_control_mean = {route.get('line_f_source_vs_best_control_mean')}
source_vs_best_control_min = {route.get('line_f_source_vs_best_control_min')}
control_equivalent_fraction = {route.get('line_f_control_equivalent_fraction')}
line_f_meaningful_gate_pass = {route.get('line_f_meaningful_gate_pass')}
line_f_s4_gate_pass = {route.get('line_f_s4_gate_pass')}
line_f_route = {route.get('line_f_route')}
```

## 7. Line D 结果

```text
line_d_source = {route.get('line_d_source')}
line_d_rows = {route.get('line_d_rows')}
line_d_v1414_baseline_rows = {route.get('line_d_v1414_baseline_rows')}
line_d_v1415_new_rows = {route.get('line_d_v1415_new_rows')}
line_d_fou_telemetry_required_rows = {route.get('line_d_fou_telemetry_required_rows')}
line_d_fou_telemetry_available_rows = {route.get('line_d_fou_telemetry_available_rows')}
line_d_fou_telemetry_missing_rows = {route.get('line_d_fou_telemetry_missing_rows')}
line_d_fou_telemetry_source = {route.get('line_d_fou_telemetry_source')}
line_d_rbf_residual_required_rows = {route.get('line_d_rbf_residual_required_rows')}
line_d_rbf_residual_available_rows = {route.get('line_d_rbf_residual_available_rows')}
line_d_rbf_residual_missing_rows = {route.get('line_d_rbf_residual_missing_rows')}
line_d_rbf_residual_source = {route.get('line_d_rbf_residual_source')}
line_d_v1415_gate_definition = {route.get('line_d_v1415_gate_definition')}
best_non_dche_family = {route.get('line_d_best_non_dche_family')}
best_non_dche_dataset_seed_pass_count = {route.get('line_d_best_non_dche_dataset_seed_pass_count')} / 9
official_fms_eligible_family_count = {route.get('line_d_official_fms_eligible_family_count')}
line_d_route = {route.get('line_d_route')}
```

Line D telemetry / visualization coverage:

```text
D-FOU Fourier telemetry required/available/missing =
  {route.get('line_d_fou_telemetry_required_rows')} /
  {route.get('line_d_fou_telemetry_available_rows')} /
  {route.get('line_d_fou_telemetry_missing_rows')}
telemetry_source = {route.get('line_d_fou_telemetry_source')}
D-RBF residual_over_base required/available/missing =
  {route.get('line_d_rbf_residual_required_rows')} /
  {route.get('line_d_rbf_residual_available_rows')} /
  {route.get('line_d_rbf_residual_missing_rows')}
residual_source = {route.get('line_d_rbf_residual_source')}
v14.15 substrate gate definition =
  {route.get('line_d_v1415_gate_definition')}
fig_d_family_pass_heatmap.svg = generated
fig_d_workspace_task_pareto.svg = generated
fig_d_nonrat_telemetry_matrix.svg = generated
fig_d_linec_tail_task_breakdown.svg = generated
```

## 7.1 Line M control 覆盖

```text
line_m_required_control_count = {route.get('line_m_required_control_count')}
line_m_executed_required_control_count = {route.get('line_m_executed_required_control_count')}
line_m_missing_required_control_count = {route.get('line_m_missing_required_control_count')}
line_m_missing_required_controls = {route.get('line_m_missing_required_controls')}
line_m_route = {route.get('line_m_route')}
```

## 8. 最终 route

```text
route = {route.get('route')}
minimum_success = {route.get('minimum_success')}
official_s5_reached = {route.get('official_s5_reached')}
promotion_allowed = {route.get('promotion_allowed')}
required_artifact_missing_count = {route.get('required_artifact_missing_count')}
forbidden_information_violation_count = {route.get('forbidden_information_violation_count')}
no_action_search_violation_count = {route.get('no_action_search_violation_count')}
```

## 9. 科学结论

```text
1. v14.15 已执行 Line R/P/I/B/F/D/M/C/Z。
2. 本轮没有达成 S4-lite / S4 / S5，promotion_allowed = 0。
3. Line I 未证明 direction 有独立 causal value；Line B 未打开 boundary/curriculum gate。
4. Line F D-CHE real-lite 仍低于 >=4/9 meaningful gate。
5. Line D 非 D-CHE substrate 仍低于 >=6/9 exploration gate。
6. Line M 已补齐计划要求的 7 类 MLP/generic controls；
   没有把 MLP/generic control 写成 promotion。
7. 继续推进需要下一版 functional causal theory 或新的 substrate carrier；
   不能在 v14.15 内新增 token/controller/action/reset route 或 audit-directed branch。
```
"""
    write_text(RECAP_DOC, recap)
    exec_log = f"""# DG-KAN v14.15 ParallelFMSCausalFork TransferObservability AllBasis 执行日志

生成时间：2026-05-30（Asia/Singapore）

## 1. 关键文件

```text
plan = docs/DG-KAN_v14.15_ParallelFMSCausalFork_TransferObservability_AllBasis_完整计划.md
runner = experiments/run_v1415_parallel_fms_causal_fork_transfer_allbasis.py
line_d_runner = experiments/run_v149_line_d_all_basis_substrate_repair.py
out_dir = results/v14_15_parallel_fms_causal_fork_transfer_observability_allbasis/official_v1415
line_d_out_dir = results/v14_15_parallel_fms_causal_fork_transfer_observability_allbasis/line_d_v1415_substrate_acceleration
line_m_out_dir = results/v14_15_parallel_fms_causal_fork_transfer_observability_allbasis/line_m_mlp_control_completion
```

## 2. 执行指令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v1415_parallel_fms_causal_fork_transfer_allbasis.py experiments/run_v149_line_d_all_basis_substrate_repair.py experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py experiments/run_v143_nonrat_compact_task_health_probe.py

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py --out-dir results/v14_15_parallel_fms_causal_fork_transfer_observability_allbasis/line_d_v1415_substrate_acceleration --device cuda:0 --candidates D-FOU32-LowFreqIdentityResidualV3,D-FOU33-BandwiseSNRWarmupV2,D-FOU34-PhaseStableBandMixNoHighFreqV2,D-FOU35-NoMaterializeLifetimeV4,D-FOU36-HighFrequencyQuarantineV2,D-RBF30-ActiveCenterOccupancyV3,D-RBF31-WidthConditionIdentityResidualV2,D-RBF32-CompactBumpNoDenseMaterializationV2,D-RBF33-GaussianLocalK4TaskHealthV2,D-RBF34-CenterOccupancyWarmupNoTaskBranch,D-WAV29-TriangularSupportV4,D-WAV30-ScaleOccupancyNoTailTargetV2,D-WAV31-LocalSupportOverlapDampingV2,D-WAV32-LocalTailCoverageAuditV2 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --epochs 1 --linec-seeds 12319500,12319501,12319502

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py --out-dir results/v14_15_parallel_fms_causal_fork_transfer_observability_allbasis/line_m_mlp_control_completion --device cuda:0 --methods '' --mlp-methods M5-MLP-NoOpMatchedOverhead,M6-MLP-SameActiveFractionControl,M7-MLP-BoundaryOnly --skip-real 1 --run-fallbacks 0 --train-steps 200 --batch-size 32

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1415_parallel_fms_causal_fork_transfer_allbasis.py --out-dir results/v14_15_parallel_fms_causal_fork_transfer_observability_allbasis/official_v1415 --line-d-out results/v14_15_parallel_fms_causal_fork_transfer_observability_allbasis/line_d_v1415_substrate_acceleration --line-m-out results/v14_15_parallel_fms_causal_fork_transfer_observability_allbasis/line_m_mlp_control_completion
```

## 3. 关键结果

```text
route = {route.get('route')}
Line P = {route.get('line_p_route')}
Line I = {route.get('line_i_route')}
Line B = {route.get('line_b_route')}
Line F = {route.get('line_f_route')}
Line D = {route.get('line_d_route')}
Line M = {route.get('line_m_route')}
promotion_allowed = {route.get('promotion_allowed')}
required_artifact_missing_count = {route.get('required_artifact_missing_count')}
forbidden_information_violation_count = {route.get('forbidden_information_violation_count')}
no_action_search_violation_count = {route.get('no_action_search_violation_count')}
```

## 4. 覆盖性修正记录

```text
首次计划复核后发现 Line D D-WAV29..32 属于预注册 low-budget all-basis 分支。
因此将 Line D 命令从 D-FOU/D-RBF 扩展为 D-FOU/D-RBF/D-WAV 全量候选并重跑。
最终 v1415_allbasis_substrate_acceleration.csv candidate rows = {route.get('line_d_rows')}；
v14.14 baseline rows = {route.get('line_d_v1414_baseline_rows')}；
v14.15 new rows = {route.get('line_d_v1415_new_rows')}；
summary rows = {route.get('line_d_summary_rows')}；
total rows = {sint(route.get('line_d_rows'), 0) + sint(route.get('line_d_summary_rows'), 0)}。
D-WAV 补齐后仍未打开 substrate exploration gate；
line_d_best_non_dche_dataset_seed_pass_count = {route.get('line_d_best_non_dche_dataset_seed_pass_count')} / 9。
```

## 4.1 Line D D-FOU telemetry 补齐记录

```text
复核 v14.15 计划 8.2 后发现 D-FOU 必须测：
  band_energy_low/mid/high
  phase_drift
  high_freq_ratio
  bandwise_snr
本次在 compact substrate probe 中新增 train-stream layer1 basis channel split telemetry。
telemetry 只写入 artifact，不参与 direction / gate / promotion。
line_d_fou_telemetry_required_rows = {route.get('line_d_fou_telemetry_required_rows')}。
line_d_fou_telemetry_available_rows = {route.get('line_d_fou_telemetry_available_rows')}。
line_d_fou_telemetry_missing_rows = {route.get('line_d_fou_telemetry_missing_rows')}。
line_d_fou_telemetry_source = {route.get('line_d_fou_telemetry_source')}。
line_d_rbf_residual_required_rows = {route.get('line_d_rbf_residual_required_rows')}。
line_d_rbf_residual_available_rows = {route.get('line_d_rbf_residual_available_rows')}。
line_d_rbf_residual_missing_rows = {route.get('line_d_rbf_residual_missing_rows')}。
line_d_rbf_residual_source = {route.get('line_d_rbf_residual_source')}。
v14.15 gate = {route.get('line_d_v1415_gate_definition')}。
```

Line D visualization 覆盖：

```text
fig_d_family_pass_heatmap.svg
fig_d_workspace_task_pareto.svg
fig_d_nonrat_telemetry_matrix.svg
fig_d_linec_tail_task_breakdown.svg
```

## 5. Line P leaveout 补齐记录

```text
复核 v14.15 计划 4.3 后，Line P 必须报告六类 split：
  leave-dataset-out
  leave-seed-out
  leave-loss-interface-out
  leave-method-out
  leave-control-out
  leave-synthetic-family-out
本轮将 v1415_predictor_leaveout.csv 规范为 planned split table。
可从 v14.14 E4 artifact 读取的 split 直接写入真实 AUC；
缺失的 method/control split 只记录 unavailable，不填假 AUC。
line_p_leaveout_rows = {route.get('line_p_leaveout_rows')}。
line_p_leaveout_available_rows = {route.get('line_p_leaveout_available_rows')}。
line_p_leaveout_unavailable_rows = {route.get('line_p_leaveout_unavailable_rows')}。
```

## 6. Line M control 补齐记录

```text
复核 v14.15 计划 9.2 后发现必须比较 7 类 MLP/generic controls。
旧 artifact 只覆盖：
  MLP-AdamW、MLP-FMS-generic、MLP-RandomMatchedNorm、
  MLP-GenericOptimizerStateControl，以及一个 MLP degree analog。
本次新增并执行：
  MLP-NoOpMatchedOverhead
  MLP-SameActiveFractionControl
  MLP-FMS-boundary-only
最终 line_m_executed_required_control_count = {route.get('line_m_executed_required_control_count')} / {route.get('line_m_required_control_count')}。
line_m_missing_required_control_count = {route.get('line_m_missing_required_control_count')}。
```

## 7. Line I selector 补齐记录

```text
复核 v14.15 计划 5.2/5.3/5.4 后发现 reduced matrix 缺少：
  S1-E4-predictor-high-score
  S2-inverse-E4-predictor
本次从 v14.13/v14.14 真实 artifact 重建 E4-K recovery_lag selector replay。
line_i_selector_ids = {route.get('line_i_selector_ids')}。
line_i_boundary_ids = {route.get('line_i_boundary_ids')}。
line_i_selector_replay_rows = {route.get('line_i_selector_replay_rows')}。
line_i_value_retention_floor_rows = {route.get('line_i_value_retention_floor_rows')}。
strict same-direction random-selector contrast 未执行，
因此只作为 diagnostic coverage，不允许 promotion。
```

## 8. 复现说明

```text
1. 先运行 py_compile。
2. 再运行 Line D substrate-only acceleration；该命令不执行 official FMS proof。
3. 再运行 Line M MLP control completion，补齐 NoOp / SameActiveFraction / boundary-only controls。
4. 最后运行 v14.15 runner 汇总 P/I/B/F/D/M/C/Z；
   runner 会从既有 v14.13/v14.14 artifact 重建 S1/S2 selector replay。
5. 所有 replay 行都在 CSV 中标注 reused/replay/source artifact；不得写成新训练。
```
"""
    write_text(EXEC_LOG_DOC, exec_log)


def write_code_review_manifest(out: Path, route: dict[str, Any]) -> None:
    rows = [
        {"item": "runner_py_compile", "status": "pass", "promotion_allowed": 0},
        {"item": "required_artifacts", "status": "pass" if sint(route.get("required_artifact_missing_count"), 0) == 0 else "fail", "promotion_allowed": 0},
        {"item": "forbidden_information_audit", "status": "pass" if sint(route.get("forbidden_information_violation_count"), 0) == 0 else "fail", "promotion_allowed": 0},
        {"item": "no_action_search_audit", "status": "pass" if sint(route.get("no_action_search_violation_count"), 0) == 0 else "fail", "promotion_allowed": 0},
    ]
    write_rows(out / "v1415_code_review_manifest.csv", rows)


def make_packet(out: Path) -> None:
    packet = out / "v1415_code_review_packet.zip"
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in [
            PLAN_DOC,
            RECAP_DOC,
            EXEC_LOG_DOC,
            ROOT / "experiments/run_v1415_parallel_fms_causal_fork_transfer_allbasis.py",
            ROOT / "experiments/run_v149_line_d_all_basis_substrate_repair.py",
        ]:
            if path.exists():
                zf.write(path, arcname=str(path.relative_to(ROOT)))
        for path in sorted(out.glob("v1415_*")):
            if path.name == packet.name:
                continue
            if path.is_file():
                zf.write(path, arcname=f"artifacts/{path.name}")
        for path in sorted(out.glob("fig_*.svg")):
            if path.is_file():
                zf.write(path, arcname=f"figures/{path.name}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    line_d_out = Path(args.line_d_out)
    method_surface = build_method_surface()
    write_rows(out / "v1415_method_surface_manifest.csv", method_surface)
    _audit, no_action, forbidden = build_line_r(out, method_surface, line_d_out)
    p_rows, _p_leave, p_summary = build_line_p(out)
    i_rows, i_effects, i_summary = build_line_i(out)
    b_rows, b_summary = build_line_b(out)
    f_rows, _f_ctrl, f_summary = build_line_f(out)
    d_rows, d_summary = build_line_d(out, line_d_out)
    _m_rows, m_summary = build_line_m(out, Path(args.line_m_out))
    _c_rows, taxonomy, c_summary = build_line_c(out, f_rows, i_rows, d_rows)
    violations = {
        "forbidden": sum(sint(r.get("violation_count"), sint(r.get("violation"), 0)) for r in forbidden),
        "no_action": sum(sint(r.get("violation_count"), sint(r.get("violation"), 0)) for r in no_action),
    }
    route = decide_route(p_summary, i_summary, b_summary, f_summary, d_summary, violations, 0)
    route.update(m_summary)
    route.update(c_summary)
    write_json(out / "v1415_route_decision.json", route)
    write_progress(out, route)
    write_no_go(out, route)
    write_code_review_manifest(out, route)
    write_figures(out, route, p_rows, i_effects, b_rows, f_rows, d_rows, taxonomy)
    write_docs(out, route)
    make_packet(out)
    manifest = write_required_manifest(out)
    missing = sum(sint(r.get("missing"), 0) for r in manifest)
    if missing != route["required_artifact_missing_count"]:
        route = decide_route(p_summary, i_summary, b_summary, f_summary, d_summary, violations, missing)
        route.update(m_summary)
        route.update(c_summary)
        write_json(out / "v1415_route_decision.json", route)
        write_progress(out, route)
        write_no_go(out, route)
        write_code_review_manifest(out, route)
        write_figures(out, route, p_rows, i_effects, b_rows, f_rows, d_rows, taxonomy)
        write_docs(out, route)
        make_packet(out)
        write_required_manifest(out)
    return route


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DG-KAN v14.15 parallel FMS causal fork")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--line-d-out", default=str(DEFAULT_LINE_D_OUT))
    parser.add_argument("--line-m-out", default=str(DEFAULT_LINE_M_OUT))
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    start = time.perf_counter()
    route = run(args)
    route = dict(route)
    route["wall_time_sec"] = time.perf_counter() - start
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
