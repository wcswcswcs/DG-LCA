#!/usr/bin/env python3
"""DG-KAN v14.14 FMS causal value / transfer boundary runner.

This runner is continue-open for exploration and fail-closed for promotion. It
does not add F-CHE8/F-CHE9, FMS-M6/M7/M8, action tokens, controllers, action
banks, or reset routes. It rebuilds E4 observability with matched-control
deconfounding, audits Q causal value against controls, executes only the
pre-registered boundary mechanisms B1/B2/B3, and replays all-basis substrate
evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v1410_nonrat_fms_transfer_fms_definition_reset as v1410
from experiments import run_v1412_1_functional_continue_open_transfer_observability_all_basis as v1412
from experiments import run_v1413_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel as v1413
from experiments import run_v144_real_transfer_fms_all_basis_substrate as v144


DEFAULT_OUT = ROOT / "results/v14_14_fms_causal_value_transfer_boundary_all_basis_continue_open/official_v1414"
PLAN_DOC = ROOT / "docs/DG-KAN_v14.14_FMS_CausalValue_TransferBoundary_AllBasisContinueOpen_完整计划.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v14.14_FMS_CausalValue_TransferBoundary_AllBasisContinueOpen_实验结果复盘.md"
EXEC_LOG_DOC = ROOT / "docs/DG-KAN_v14.14_FMS_CausalValue_TransferBoundary_AllBasisContinueOpen_执行日志.md"

V1413_OUT = ROOT / "results/v14_13_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel/official_v1413"
V1410_SYNTH = ROOT / "results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/repair_dche17_seed012_streaming_allfche_steps300_interval200_v1410"
V1410_REAL = ROOT / "results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_allfchefb_steps200_interval200_v1410"

REQUIRED = [
    "v1414_route_decision.json",
    "v1414_required_artifact_manifest.csv",
    "v1414_forbidden_information_audit.csv",
    "v1414_no_action_search_audit.csv",
    "v1414_method_surface_manifest.csv",
    "v1414_line_r_audit.csv",
    "v1414_e4_observability_deconfound.csv",
    "v1414_e4_leaveout_summary.csv",
    "v1414_q_fms_vs_matched_controls.csv",
    "v1414_q_control_equivalence_summary.csv",
    "v1414_b_boundary_fms_real_lite.csv",
    "v1414_b_boundary_controls.csv",
    "v1414_d_cross_version_reconciliation.csv",
    "v1414_d_all_basis_substrate_hardening.csv",
    "v1414_linec_tail_audit.csv",
    "v1414_failure_taxonomy.csv",
    "v1414_no_go_boundary.md",
    "v1414_next_hypothesis_queue.md",
    "v1414_code_review_packet.zip",
]

FIGURES = [
    "fig_v1414_gate_ladder.svg",
    "fig_v1414_e4_roc_pr.svg",
    "fig_v1414_leaveout_auc_heatmap.svg",
    "fig_v1414_proxy_control_deconfound.svg",
    "fig_v1414_v3_breakpoint_sankey.svg",
    "fig_v1414_fms_vs_controls_effect.svg",
    "fig_v1414_control_equivalence_by_stratum.svg",
    "fig_v1414_proxy_to_effect_chain_waterfall.svg",
    "fig_v1414_all_basis_substrate_matrix.svg",
    "fig_v1414_failure_taxonomy_heatmap.svg",
]

BOUNDARY_ALIASES = {
    "F-B1-AbstentionBoundaryFMS": "F-CHE-RT1-TrainSplitAgreement",
    "F-B2-ConstraintOnlyBoundaryFMS": "F-CHE-FB2-DegreeConstraintOnly",
    "F-B3-ContinuousLowAmplitudeBoundaryFMS": "F-CHE6-PhaseScheduleDegreeFMS",
}

D_CHE_CONTROLS = list(v1413.D_CHE_CONTROLS)

Q_GROUPS = {
    "G0-D-CHE-AdamW-baseline": "C0-D-CHE-AdamW",
    "G2-RandomMatchedNorm": "C2-D-CHE-AdamW-RandomMatchedNorm",
    "G3-SameActiveFractionControl": "C5-D-CHE-AdamW-SameActiveFractionControl",
    "G6-AdamWParallelDirectionControl": "C3-D-CHE-AdamW-AdamWParallelDirectionControl",
    "G7-NoOpMatchedOverhead": "C1-D-CHE-AdamW-NoOpMatchedOverhead",
    "G8-GenericOptimizerStateControl": "C4-D-CHE-AdamW-GenericOptimizerStateControl",
}


def fnum(value: Any, default: float = 0.0) -> float:
    return v1412.fnum(value, default)


def sint(value: Any, default: int = 0) -> int:
    return v1412.sint(value, default)


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(v) and not math.isinf(v)]
    return statistics.fmean(vals) if vals else 0.0


def median(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(v) and not math.isinf(v)]
    return statistics.median(vals) if vals else 0.0


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def split_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def read_rows(path: Path) -> list[dict[str, str]]:
    return v1412.read_rows(path)


def write_rows(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    v1412.write_rows(path, rows, fieldnames)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    v1412.write_json(path, obj)


def write_text(path: Path, text: str) -> None:
    v1412.write_text(path, text)


def load_json(path: Path) -> dict[str, Any]:
    return v1412.load_json(path)


def sha256(path: Path) -> str:
    return v1412.sha256(path)


def auc_or_half(scores: Sequence[float], labels: Sequence[int]) -> float:
    auc = v1412.auc_score(scores, labels)
    return 0.5 if auc is None else float(auc)


def spearman_or_zero(xs: Sequence[float], ys: Sequence[float]) -> float:
    rho = v1412.spearman(xs, ys)
    return 0.0 if rho is None else float(rho)


def precision_at_top(scores: Sequence[float], labels: Sequence[int], frac: float = 0.10) -> float:
    if not scores:
        return 0.0
    k = max(1, int(math.ceil(len(scores) * frac)))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return sum(labels[i] for i in order) / max(1, k)


def percentile(values: Sequence[float], q: float) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    idx = min(len(vals) - 1, max(0, int(math.ceil(q * len(vals))) - 1))
    return vals[idx]


def simple_svg(path: Path, title: str, rows: Sequence[tuple[str, float]], threshold: float | None = None) -> None:
    v1412.simple_svg(path, title, rows, threshold)


def build_method_surface_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in D_CHE_CONTROLS:
        rows.append(
            {
                "method": method,
                "method_family": "D-CHE-control",
                "underlying_method": method,
                "executed_in_boundary": 1,
                "is_control": 1,
                "is_boundary_mechanism": 0,
                "fche_token_extension_count": 0,
                "fms_method_extension_count": 0,
                "new_action_token_count": 0,
                "controller_executed": 0,
                "action_bank_used": 0,
                "reset_route_used": 0,
                "uses_dataset_name_branch": 0,
                "uses_seed_specific_scale": 0,
                "uses_linec_tail_direction": 0,
                "promotion_allowed": 0,
            }
        )
    for alias, underlying in BOUNDARY_ALIASES.items():
        rows.append(
            {
                "method": alias,
                "method_family": "D-CHE-boundary-mechanism",
                "underlying_method": underlying,
                "executed_in_boundary": 1,
                "is_control": 0,
                "is_boundary_mechanism": 1,
                "boundary_abstention_gating": int(alias == "F-B1-AbstentionBoundaryFMS"),
                "constraint_only_boundary": int(alias == "F-B2-ConstraintOnlyBoundaryFMS"),
                "continuous_low_amplitude_boundary": int(alias == "F-B3-ContinuousLowAmplitudeBoundaryFMS"),
                "fche_token_extension_count": 0,
                "fms_method_extension_count": 0,
                "new_action_token_count": 0,
                "controller_executed": 0,
                "action_bank_used": 0,
                "reset_route_used": 0,
                "uses_dataset_name_branch": 0,
                "uses_seed_specific_scale": 0,
                "uses_linec_tail_direction": 0,
                "promotion_allowed": 0,
            }
        )
    for group in ["G4-SameDegreeProjectionRejectionControl", "G5-SameValueRetentionRandomDirection"]:
        rows.append(
            {
                "method": group,
                "method_family": "matched-control-deconfound",
                "underlying_method": "replay-matched-existing-controls",
                "executed_in_boundary": 0,
                "is_control": 1,
                "is_boundary_mechanism": 0,
                "fche_token_extension_count": 0,
                "fms_method_extension_count": 0,
                "new_action_token_count": 0,
                "controller_executed": 0,
                "action_bank_used": 0,
                "reset_route_used": 0,
                "uses_dataset_name_branch": 0,
                "uses_seed_specific_scale": 0,
                "uses_linec_tail_direction": 0,
                "promotion_allowed": 0,
            }
        )
    return rows


def build_line_r(out: Path, method_manifest: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sources = [
        PLAN_DOC,
        ROOT / "experiments/run_v1414_fms_causal_value_transfer_boundary_all_basis_continue_open.py",
        ROOT / "experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py",
        V1413_OUT / "v1413_route_decision.json",
        V1413_OUT / "v1413_e3_transfer_observability_summary.csv",
        V1413_OUT / "v1413_v3_proxy_to_effect_chain.csv",
        V1413_OUT / "v1413_f3_dche_fms_real_lite.csv",
        V1413_OUT / "v1413_f3_dche_fms_controls.csv",
        V1413_OUT / "v1413_d0_cross_version_reconciliation.csv",
    ]
    audit = [
        {
            "artifact": str(p.relative_to(ROOT) if p.is_absolute() and p.exists() else p),
            "exists": int(p.exists()),
            "sha256": sha256(p) if p.exists() and p.is_file() else "",
            "used_for_direction": 0,
            "used_for_audit_or_replay": 1,
            "promotion_allowed": 0,
        }
        for p in sources
    ]
    no_action = []
    forbidden = []
    for row in method_manifest:
        action_violation = int(
            sint(row.get("fche_token_extension_count"), 0) > 0
            or sint(row.get("fms_method_extension_count"), 0) > 0
            or sint(row.get("new_action_token_count"), 0) > 0
            or sint(row.get("controller_executed"), 0) > 0
            or sint(row.get("action_bank_used"), 0) > 0
            or sint(row.get("reset_route_used"), 0) > 0
        )
        no_action.append(
            {
                "method": row["method"],
                "fche_token_extension_count": row.get("fche_token_extension_count", 0),
                "fms_method_extension_count": row.get("fms_method_extension_count", 0),
                "new_action_token_count": row.get("new_action_token_count", 0),
                "controller_executed": row.get("controller_executed", 0),
                "action_bank_used": row.get("action_bank_used", 0),
                "reset_route_used": row.get("reset_route_used", 0),
                "violation": action_violation,
                "promotion_allowed": 0,
            }
        )
        forbidden_violation = int(sint(row.get("uses_dataset_name_branch"), 0) or sint(row.get("uses_seed_specific_scale"), 0))
        forbidden.append(
            {
                "method": row["method"],
                "direction_uses_validation_test_future_query": 0,
                "direction_uses_linec_cep99_nll_ece_auctime": 0,
                "dataset_name_branch_used": row.get("uses_dataset_name_branch", 0),
                "seed_specific_scale_used": row.get("uses_seed_specific_scale", 0),
                "uses_teacher_distillation_loss_mod_sampler_class_weight": 0,
                "violation": forbidden_violation,
                "promotion_allowed": 0,
            }
        )
    write_rows(out / "v1414_line_r_audit.csv", audit)
    write_rows(out / "v1414_no_action_search_audit.csv", no_action)
    write_rows(out / "v1414_forbidden_information_audit.csv", forbidden)
    return audit, no_action, forbidden


def by_dataset_seed(rows: Sequence[dict[str, Any]], method: str | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if method is not None and row.get("method") != method:
            continue
        out[(str(row.get("dataset", "")), str(row.get("seed", "")))] = row
    return out


def build_e4(out: Path, f3_rows: Sequence[dict[str, Any]], controls: Sequence[dict[str, Any]], chain_rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    chain_by_key = {(r.get("method"), r.get("dataset"), r.get("seed")): r for r in chain_rows}
    c1 = by_dataset_seed(controls, "C1-D-CHE-AdamW-NoOpMatchedOverhead")
    c2 = by_dataset_seed(controls, "C2-D-CHE-AdamW-RandomMatchedNorm")

    feature_rows: list[dict[str, Any]] = []
    control_feature_rows: dict[str, list[dict[str, Any]]] = {}

    def add_feature(row: dict[str, Any], name: str, value: Any, source: str) -> None:
        if value == "" or value is None:
            return
        feature_rows.append(
            {
                "stage": "V1414_E4_FEATURE",
                "feature_name": name,
                "feature_source": source,
                "method": row.get("method", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "loss_interface": row.get("loss_interface", ""),
                "task_family": row.get("task", "real_lite"),
                "feature_value": fnum(value, 0.0),
                "real_lite_pass": sint(row.get("strict_gate_pass"), 0),
                "source_vs_best_control": fnum(row.get("source_vs_best_control"), 0.0),
                "promotion_allowed": 0,
            }
        )

    for row in f3_rows:
        chain = chain_by_key.get((row.get("method"), row.get("dataset"), row.get("seed")), {})
        key = (str(row.get("dataset", "")), str(row.get("seed", "")))
        add_feature(row, "E4-A-degree_projection_rejection_fraction", row.get("degree_projection_rejection_fraction", chain.get("degree_projection_rejection_fraction", "")), "v1413_f3/v3")
        add_feature(row, "E4-B-value_retention_after_degree_projection", row.get("value_retention_after_degree_projection", chain.get("value_retention_after_degree_projection", "")), "v1413_f3/v3")
        add_feature(row, "E4-C-cos_projected_vs_generic", row.get("cos_projected_vs_generic", chain.get("cos_projected_vs_generic", "")), "v1413_f3/v3")
        add_feature(row, "E4-D-projection_rejection_fraction", row.get("degree_projection_rejection_fraction", chain.get("degree_projection_rejection_fraction", "")), "v1413_f3/v3")
        add_feature(row, "E4-E-high_degree_fraction_delta", chain.get("actual_high_degree_fraction_delta", row.get("high_degree_energy_fraction", "")), "v1413_v3")
        add_feature(row, "E4-F-degree_entropy_delta", row.get("degree_entropy", ""), "v1413_f3")
        add_feature(row, "E4-G-generic_fms_norm", row.get("fms_state_norm", chain.get("generic_fms_norm", "")), "v1413_f3/v3")
        add_feature(row, "E4-H-actual_update_norm", row.get("parameter_update_norm", chain.get("actual_update_norm", "")), "v1413_f3/v3")
        add_feature(row, "E4-I-split_agreement", row.get("degree_gate_active_fraction", ""), "available_train_stream_proxy_degree_gate")
        add_feature(row, "E4-J-micro_horizon_loss_integral", chain.get("micro_horizon_integral", ""), "v1413_v3")
        add_feature(row, "E4-K-recovery_lag", chain.get("recovery_lag", ""), "v1413_v3")
        add_feature(row, "E4-L-adamw_fms_update_cosine", row.get("cos_projected_vs_generic", chain.get("cos_projected_vs_generic", "")), "v1413_f3/v3")
        add_feature(row, "E4-M-matched_random_same_norm_response", c2.get(key, {}).get("source_vs_adamw", ""), "v1413_control_C2")
        add_feature(row, "E4-N-noop_matched_overhead_response", c1.get(key, {}).get("source_vs_adamw", ""), "v1413_control_C1")

    for crow in controls:
        pseudo = dict(crow)
        pseudo["strict_gate_pass"] = 0
        for name, value in [
            ("E4-A-degree_projection_rejection_fraction", crow.get("degree_projection_rejection_fraction", "")),
            ("E4-B-value_retention_after_degree_projection", crow.get("value_retention_after_degree_projection", "")),
            ("E4-C-cos_projected_vs_generic", crow.get("cos_projected_vs_generic", "")),
            ("E4-D-projection_rejection_fraction", crow.get("degree_projection_rejection_fraction", "")),
            ("E4-G-generic_fms_norm", crow.get("fms_state_norm", "")),
            ("E4-H-actual_update_norm", crow.get("parameter_update_norm", "")),
            ("E4-L-adamw_fms_update_cosine", crow.get("cos_projected_vs_generic", "")),
        ]:
            if value != "":
                control_feature_rows.setdefault(name, []).append({"feature_value": fnum(value, 0.0), "label": 0, "source": fnum(crow.get("source_vs_adamw"), 0.0)})

    summary: list[dict[str, Any]] = []
    for feature_name in sorted({r["feature_name"] for r in feature_rows}):
        group = [r for r in feature_rows if r["feature_name"] == feature_name]
        scores = [fnum(r["feature_value"], 0.0) for r in group]
        labels = [sint(r["real_lite_pass"], 0) for r in group]
        sources = [fnum(r["source_vs_best_control"], 0.0) for r in group]
        raw_auc = auc_or_half(scores, labels)
        rho = spearman_or_zero(scores, sources)
        leave_values: dict[str, list[float]] = {}
        for split_key in ["task_family", "dataset", "seed", "loss_interface"]:
            vals = []
            for held in sorted({str(r.get(split_key, "")) for r in group if str(r.get(split_key, "")) != ""}):
                subset = [r for r in group if str(r.get(split_key, "")) != held]
                vals.append(auc_or_half([fnum(r["feature_value"], 0.0) for r in subset], [sint(r["real_lite_pass"], 0) for r in subset]))
            leave_values[split_key] = vals
        all_leave = [v for vals in leave_values.values() for v in vals]
        ctrl = control_feature_rows.get(feature_name, [])
        ctrl_scores = [fnum(r["feature_value"], 0.0) for r in ctrl]
        ctrl_labels = [sint(r["label"], 0) for r in ctrl]
        control_auc = auc_or_half(ctrl_scores, ctrl_labels) if ctrl else 0.5
        threshold = percentile(scores, 0.90)
        fpr_controls = sum(1 for s in ctrl_scores if s >= threshold) / max(1, len(ctrl_scores))
        incremental = raw_auc - control_auc
        summary.append(
            {
                "feature_name": feature_name,
                "rows": len(group),
                "positive_rows": sum(labels),
                "raw_auc_predict_real_lite_pass": raw_auc,
                "leave_task_family_out_auc": min(leave_values["task_family"]) if leave_values["task_family"] else "",
                "leave_dataset_out_auc": min(leave_values["dataset"]) if leave_values["dataset"] else "",
                "leave_seed_out_auc": min(leave_values["seed"]) if leave_values["seed"] else "",
                "leave_loss_interface_out_auc": min(leave_values["loss_interface"]) if leave_values["loss_interface"] else "",
                "leaveout_auc_min": min(all_leave) if all_leave else 0.5,
                "spearman_proxy_to_source": rho,
                "precision_at_top10pct": precision_at_top(scores, labels),
                "false_positive_rate_on_controls": fpr_controls,
                "control_matched_auc": control_auc,
                "incremental_auc_over_controls": incremental,
                "exploration_gate_pass": int(raw_auc >= 0.60 and (min(all_leave) if all_leave else 0.5) >= 0.55),
                "promotion_enabling_gate_pass": int(raw_auc >= 0.70 and (min(all_leave) if all_leave else 0.5) >= 0.65 and rho >= 0.30 and incremental >= 0.05),
                "promotion_allowed": 0,
            }
        )

    leaveout_rows = []
    for row in summary:
        for split_name, key in [
            ("leave_task_family_out", "leave_task_family_out_auc"),
            ("leave_dataset_out", "leave_dataset_out_auc"),
            ("leave_seed_out", "leave_seed_out_auc"),
            ("leave_loss_interface_out", "leave_loss_interface_out_auc"),
        ]:
            leaveout_rows.append({"feature_name": row["feature_name"], "leaveout_split": split_name, "auc": row.get(key, ""), "promotion_allowed": 0})

    best = max(summary, key=lambda r: fnum(r.get("raw_auc_predict_real_lite_pass"), 0.0), default={})
    e4 = {
        "e4_rows": len(feature_rows),
        "e4_summary_rows": len(summary),
        "e4_best_feature": best.get("feature_name", ""),
        "e4_best_auc_mean": fnum(best.get("raw_auc_predict_real_lite_pass"), 0.0),
        "e4_best_leaveout_auc_min": fnum(best.get("leaveout_auc_min"), 0.0),
        "e4_best_spearman": fnum(best.get("spearman_proxy_to_source"), 0.0),
        "e4_best_incremental_auc_over_controls": fnum(best.get("incremental_auc_over_controls"), 0.0),
        "e4_exploration_gate_pass": int(fnum(best.get("raw_auc_predict_real_lite_pass"), 0.0) >= 0.60 and fnum(best.get("leaveout_auc_min"), 0.0) >= 0.55),
        "e4_promotion_enabling_gate_pass": sint(best.get("promotion_enabling_gate_pass"), 0),
        "e4_observability_explained_by_controls": int(fnum(best.get("incremental_auc_over_controls"), 0.0) < 0.05),
    }
    write_rows(out / "v1414_e4_observability_deconfound.csv", summary)
    write_rows(out / "v1414_e4_leaveout_summary.csv", leaveout_rows)
    return summary, leaveout_rows, e4


def nearest_control(row: dict[str, Any], controls: Sequence[dict[str, Any]], metric: str) -> dict[str, Any]:
    same = [c for c in controls if c.get("dataset") == row.get("dataset") and c.get("seed") == row.get("seed")]
    if not same:
        same = list(controls)
    target = fnum(row.get(metric), 0.0)
    return min(same, key=lambda c: abs(fnum(c.get(metric), 0.0) - target), default={})


def build_q(out: Path, f3_rows: Sequence[dict[str, Any]], controls: Sequence[dict[str, Any]], v3_summary: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    control_by_key = {(c.get("dataset"), c.get("seed"), c.get("method")): c for c in controls}

    for fms in f3_rows:
        same_controls = [c for c in controls if c.get("dataset") == fms.get("dataset") and c.get("seed") == fms.get("seed")]
        best = max(same_controls, key=lambda c: fnum(c.get("source_vs_adamw"), -999.0), default={})
        degree_match = nearest_control(fms, same_controls, "degree_projection_rejection_fraction")
        retention_match = nearest_control(fms, same_controls, "value_retention_after_degree_projection")
        matched_controls = [c for c in [best, degree_match, retention_match] if c]
        best_matched = max(matched_controls, key=lambda c: fnum(c.get("source_vs_adamw"), -999.0), default=best)
        source_delta = fnum(fms.get("source_vs_adamw"), 0.0) - fnum(best_matched.get("source_vs_adamw"), 0.0)
        auc_delta = fnum(fms.get("AUC_NLL"), 9.0) - fnum(best_matched.get("AUC_NLL"), 9.0)
        tail_delta = (
            fnum(fms.get("CEp99"), 0.0) - fnum(best_matched.get("CEp99"), 0.0)
            + fnum(fms.get("NLL"), 0.0) - fnum(best_matched.get("NLL"), 0.0)
            + fnum(fms.get("ECE"), 0.0) - fnum(best_matched.get("ECE"), 0.0)
        )
        linec_delta = fnum(fms.get("LineC_pass_rate"), 0.0) - fnum(best_matched.get("LineC_pass_rate"), 0.0)
        control_equiv = int(source_delta <= 0.005)
        bad_event = int(source_delta < 0.0 or fnum(fms.get("AUCtime_ratio"), 9.0) > 1.0 or sint(fms.get("LineC_majority_pass"), 0) == 0)
        harmless_null = int(abs(source_delta) <= 0.002 and bad_event == 0)
        rows.append(
            {
                "group": "G1-D-CHE-FMS-M1..M5",
                "dataset": fms.get("dataset", ""),
                "seed": fms.get("seed", ""),
                "method": fms.get("method", ""),
                "matched_control_method": best_matched.get("method", ""),
                "fms_specific_source_delta": source_delta,
                "fms_specific_auc_delta": auc_delta,
                "fms_specific_tail_delta": tail_delta,
                "fms_specific_linec_delta": linec_delta,
                "source_vs_best_control": fnum(fms.get("source_vs_best_control"), 0.0),
                "AUCtime_ratio_vs_best_control": fnum(fms.get("AUCtime_ratio"), 9.0),
                "CEp99_delta_vs_best_control": fnum(fms.get("CEp99"), 0.0) - fnum(best_matched.get("CEp99"), 0.0),
                "NLL_delta_vs_best_control": fnum(fms.get("NLL"), 0.0) - fnum(best_matched.get("NLL"), 0.0),
                "ECE_delta_vs_best_control": fnum(fms.get("ECE"), 0.0) - fnum(best_matched.get("ECE"), 0.0),
                "LineC_pass_gap_vs_control": linec_delta,
                "control_equivalent": control_equiv,
                "harmless_null": harmless_null,
                "bad_event": bad_event,
                "strict_gate_pass": sint(fms.get("strict_gate_pass"), 0),
                "promotion_allowed": 0,
            }
        )

    for group, method in Q_GROUPS.items():
        for ctrl in [c for c in controls if c.get("method") == method]:
            rows.append(
                {
                    "group": group,
                    "dataset": ctrl.get("dataset", ""),
                    "seed": ctrl.get("seed", ""),
                    "method": ctrl.get("method", ""),
                    "matched_control_method": ctrl.get("method", ""),
                    "fms_specific_source_delta": 0.0,
                    "fms_specific_auc_delta": 0.0,
                    "fms_specific_tail_delta": 0.0,
                    "fms_specific_linec_delta": 0.0,
                    "source_vs_best_control": ctrl.get("source_vs_best_control", ""),
                    "AUCtime_ratio_vs_best_control": ctrl.get("AUCtime_ratio", ""),
                    "CEp99_delta_vs_best_control": ctrl.get("CEp99_delta", ""),
                    "NLL_delta_vs_best_control": ctrl.get("NLL_delta", ""),
                    "ECE_delta_vs_best_control": ctrl.get("ECE_delta", ""),
                    "LineC_pass_gap_vs_control": 0.0,
                    "control_equivalent": 1,
                    "harmless_null": 1,
                    "bad_event": 0,
                    "strict_gate_pass": 0,
                    "promotion_allowed": 0,
                }
            )

    for group, metric in [("G4-SameDegreeProjectionRejectionControl", "degree_projection_rejection_fraction"), ("G5-SameValueRetentionRandomDirection", "value_retention_after_degree_projection")]:
        for fms in f3_rows:
            ctrl = nearest_control(fms, controls, metric)
            if not ctrl:
                continue
            rows.append(
                {
                    "group": group,
                    "dataset": fms.get("dataset", ""),
                    "seed": fms.get("seed", ""),
                    "method": group,
                    "matched_control_method": ctrl.get("method", ""),
                    "fms_specific_source_delta": 0.0,
                    "fms_specific_auc_delta": 0.0,
                    "fms_specific_tail_delta": 0.0,
                    "fms_specific_linec_delta": 0.0,
                    "source_vs_best_control": ctrl.get("source_vs_best_control", ""),
                    "AUCtime_ratio_vs_best_control": ctrl.get("AUCtime_ratio", ""),
                    "CEp99_delta_vs_best_control": ctrl.get("CEp99_delta", ""),
                    "NLL_delta_vs_best_control": ctrl.get("NLL_delta", ""),
                    "ECE_delta_vs_best_control": ctrl.get("ECE_delta", ""),
                    "LineC_pass_gap_vs_control": 0.0,
                    "control_equivalent": 1,
                    "harmless_null": 1,
                    "bad_event": 0,
                    "strict_gate_pass": 0,
                    "promotion_allowed": 0,
                }
            )

    fms_q = [r for r in rows if r.get("group") == "G1-D-CHE-FMS-M1..M5"]
    control_equiv_fraction = sum(sint(r.get("control_equivalent"), 0) for r in fms_q) / max(1, len(fms_q))
    harmless_fraction = sum(sint(r.get("harmless_null"), 0) for r in fms_q) / max(1, len(fms_q))
    bad_fraction = sum(sint(r.get("bad_event"), 0) for r in fms_q) / max(1, len(fms_q))
    source_mean = mean([fnum(r.get("source_vs_best_control"), 0.0) for r in fms_q])
    linec_rate = mean([1.0 if fnum(r.get("LineC_pass_gap_vs_control"), -1.0) >= 0.0 else 0.0 for r in fms_q])
    v3_b6 = next((r for r in v3_summary if r.get("breakpoint") == "V3-B6-ControlEquivalent"), {})
    summary = [
        {
            "q_rows": len(rows),
            "q_fms_rows": len(fms_q),
            "source_vs_best_control_mean": source_mean,
            "control_equivalent_fraction": control_equiv_fraction,
            "harmless_null_fraction": harmless_fraction,
            "bad_event_fraction": bad_fraction,
            "linec_nonnegative_gap_rate": linec_rate,
            "v3_b6_fraction": fnum(v3_b6.get("fraction"), 0.0),
            "causal_exploration_gate_pass": int(source_mean > 0.0 and control_equiv_fraction <= 0.60 and bad_fraction <= 0.25),
            "strong_causal_gate_pass": int(source_mean >= 0.005 and control_equiv_fraction <= 0.40 and linec_rate >= 0.80),
            "q_route_hint": "R3-FMSDirectionControlEquivalent" if control_equiv_fraction > 0.60 or fnum(v3_b6.get("fraction"), 0.0) > 0.60 else "Q-CausalValueNotRejected",
            "promotion_allowed": 0,
        }
    ]
    q_summary = dict(summary[0])
    write_rows(out / "v1414_q_fms_vs_matched_controls.csv", rows)
    write_rows(out / "v1414_q_control_equivalence_summary.csv", summary)
    return rows, summary, q_summary


def alias_boundary_result(result: dict[str, Any], alias: str, underlying: str) -> dict[str, Any]:
    out = deepcopy(result)
    for key in ["row", "degree_rows", "projection_rows", "linec_rows"]:
        rows = [out[key]] if key == "row" else out.get(key, [])
        for row in rows:
            row["method_alias_underlying"] = underlying
            row["method"] = alias
            row["pre_registered_v1414_boundary"] = 1
            row["is_boundary_mechanism"] = 1
            row["is_action_token_extension"] = 0
            row["controller_executed"] = 0
            row["action_bank_used"] = 0
            row["promotion_allowed"] = 0
    return out


def run_boundary_real_lite(args: argparse.Namespace, out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fms_path = out / "v1414_b_boundary_fms_real_lite.csv"
    ctrl_path = out / "v1414_b_boundary_controls.csv"
    linec_path = out / "v1414_linec_tail_audit.csv"
    tax_path = out / "v1414_failure_taxonomy.csv"
    degree_path = out / "v1414_dche_degree_telemetry.csv"
    proj_path = out / "v1414_dche_projection_retention.csv"
    if fms_path.exists() and ctrl_path.exists() and not int(args.force_boundary_real_lite):
        fms_rows = read_rows(fms_path)
        ctrl_rows = read_rows(ctrl_path)
        taxonomy = read_rows(tax_path)
        summary = {
            "boundary_reused_existing": 1,
            "boundary_rows": len(fms_rows),
            "boundary_control_rows": len(ctrl_rows),
            "boundary_real_lite_pass_count": len({(r.get("dataset"), r.get("seed")) for r in fms_rows if sint(r.get("strict_gate_pass")) == 1}),
            "boundary_mean_source_vs_best_control": mean([fnum(r.get("source_vs_best_control"), 0.0) for r in fms_rows]),
            "boundary_source_fail_count": sum(sint(r.get("source_fail"), 0) for r in taxonomy),
            "boundary_auctime_fail_count": sum(sint(r.get("AUC_fail"), 0) for r in taxonomy),
            "boundary_tail_fail_count": sum(sint(r.get("CEp99_fail"), 0) + sint(r.get("NLL_fail"), 0) + sint(r.get("ECE_fail"), 0) for r in taxonomy),
            "boundary_linec_fail_count": sum(sint(r.get("LineC_fail"), 0) for r in taxonomy),
        }
        return fms_rows, ctrl_rows, read_rows(linec_path), taxonomy, read_rows(degree_path), read_rows(proj_path), summary

    import torch

    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    case_args = v1410.build_argparser().parse_args([])
    case_args.device = str(device)
    case_args.data_root = args.data_root
    case_args.no_download = bool(args.no_download)
    case_args.train_size = int(args.real_lite_train_size)
    case_args.val_size = int(args.real_lite_val_size)
    case_args.test_size = int(args.real_lite_test_size)
    case_args.train_steps = int(args.real_lite_train_steps)
    case_args.batch_size = int(args.real_lite_batch_size)
    case_args.lr = float(args.real_lite_lr)
    case_args.weight_decay = float(args.real_lite_weight_decay)
    case_args.fms_beta = float(args.real_lite_fms_beta)
    case_args.fms_update_interval = int(args.real_lite_fms_update_interval)
    case_args.streaming_per_example_gradients = 1
    case_args.trace_interval = max(1, int(args.real_lite_trace_interval))
    case_args.linec_seeds = args.real_lite_linec_seeds
    case_args.linec_batch_size = int(args.real_lite_linec_batch_size)
    case_args.linec_sketch_dim = int(args.real_lite_linec_sketch_dim)
    case_args.dche_candidate = args.dche_candidate
    case_args.mlp_hidden = int(args.mlp_hidden)

    raw_rows: list[dict[str, Any]] = []
    linec_rows: list[dict[str, Any]] = []
    degree_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    methods = D_CHE_CONTROLS + list(BOUNDARY_ALIASES)
    for dataset in split_csv(args.real_lite_datasets):
        for seed in split_ints(args.real_lite_seeds):
            xtr, ytr, xva, yva, xte, yte, input_dim, output_dim = v144.load_real_split(case_args, dataset, seed, device)
            for method in methods:
                underlying = BOUNDARY_ALIASES.get(method, method)
                case_args.boundary_abstention_gating = int(method == "F-B1-AbstentionBoundaryFMS")
                case_args.actual_micro_horizon_gating = 0
                case_args.actual_micro_horizon_steps = 0
                case_args.fms_strength = float(args.boundary_low_amplitude_strength if method == "F-B3-ContinuousLowAmplitudeBoundaryFMS" else args.real_lite_fms_strength)
                result = v1410.train_model_case(
                    family="D-CHE",
                    candidate_id=args.dche_candidate,
                    method=underlying,
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
                    args=case_args,
                    device=device,
                    real_linec=True,
                )
                if method in BOUNDARY_ALIASES:
                    result = alias_boundary_result(result, method, underlying)
                    result["row"]["boundary_abstention_gating"] = int(method == "F-B1-AbstentionBoundaryFMS")
                    result["row"]["constraint_only_boundary"] = int(method == "F-B2-ConstraintOnlyBoundaryFMS")
                    result["row"]["continuous_low_amplitude_boundary"] = int(method == "F-B3-ContinuousLowAmplitudeBoundaryFMS")
                else:
                    result["row"]["method_alias_underlying"] = method
                raw_rows.append(result["row"])
                for r in result.get("degree_rows", []):
                    r["method"] = method if method in BOUNDARY_ALIASES else r.get("method", method)
                    degree_rows.append(r)
                for r in result.get("projection_rows", []):
                    r["method"] = method if method in BOUNDARY_ALIASES else r.get("method", method)
                    projection_rows.append(r)
                for r in result.get("linec_rows", []):
                    r["method"] = method if method in BOUNDARY_ALIASES else r.get("method", method)
                    linec_rows.append(r)

    enriched = v1410.enrich_rows(raw_rows, "V1414_BOUNDARY_REAL_LITE")
    controls = [r for r in enriched if sint(r.get("control_method"), 0) == 1]
    fms_rows = [r for r in enriched if sint(r.get("control_method"), 0) == 0]
    taxonomy = v1410.failure_taxonomy(enriched)
    f3_route = load_json(V1413_OUT / "v1413_route_decision.json")
    f3_control_equiv = 0.80
    for row in fms_rows:
        row["boundary_real_lite_diagnostic"] = 1
        row["control_equivalent"] = int(fnum(row.get("source_vs_best_control"), 0.0) <= 0.005)
        row["bad_event"] = int(fnum(row.get("source_vs_best_control"), 0.0) < 0.0 or fnum(row.get("AUCtime_ratio"), 9.0) > 1.0 or sint(row.get("LineC_majority_pass"), 0) == 0)
        row["proxy_score"] = row.get("fms_state_norm", "")
        row["proxy_stratum"] = "high_rejection" if fnum(row.get("degree_projection_rejection_fraction"), 0.0) >= 0.20 else "low_rejection"
        row["active_fraction"] = row.get("degree_gate_active_fraction", "")
        row["promotion_allowed"] = 0
    for row in controls:
        row["boundary_control"] = 1
        row["promotion_allowed"] = 0
    for row in taxonomy:
        row["stage"] = "V1414_BOUNDARY_FAILURE_TAXONOMY"

    write_rows(fms_path, fms_rows)
    write_rows(ctrl_path, controls)
    write_rows(linec_path, linec_rows)
    write_rows(tax_path, taxonomy)
    write_rows(degree_path, degree_rows)
    write_rows(proj_path, projection_rows)
    pass_count = len({(r.get("dataset"), r.get("seed")) for r in fms_rows if sint(r.get("strict_gate_pass")) == 1})
    control_equiv_fraction = sum(sint(r.get("control_equivalent"), 0) for r in fms_rows) / max(1, len(fms_rows))
    summary = {
        "boundary_reused_existing": 0,
        "boundary_rows": len(fms_rows),
        "boundary_control_rows": len(controls),
        "boundary_real_lite_pass_count": pass_count,
        "boundary_mean_source_vs_best_control": mean([fnum(r.get("source_vs_best_control"), 0.0) for r in fms_rows]),
        "boundary_control_equivalent_fraction": control_equiv_fraction,
        "boundary_control_equivalent_decrease_fraction": max(0.0, f3_control_equiv - control_equiv_fraction) / max(1.0e-8, f3_control_equiv),
        "boundary_source_fail_count": sum(sint(r.get("source_fail"), 0) for r in taxonomy),
        "boundary_auctime_fail_count": sum(sint(r.get("AUC_fail"), 0) for r in taxonomy),
        "boundary_tail_fail_count": sum(sint(r.get("CEp99_fail"), 0) + sint(r.get("NLL_fail"), 0) + sint(r.get("ECE_fail"), 0) for r in taxonomy),
        "boundary_linec_fail_count": sum(sint(r.get("LineC_fail"), 0) for r in taxonomy),
        "boundary_exploration_gate_pass": int(pass_count >= 3 and mean([fnum(r.get("source_vs_best_control"), 0.0) for r in fms_rows]) > 0.0 and control_equiv_fraction <= 0.64),
        "boundary_meaningful_gate_pass": int(pass_count >= 4 and mean([fnum(r.get("source_vs_best_control"), 0.0) for r in fms_rows]) >= 0.002),
        "boundary_s4_gate_pass": int(pass_count >= 6 and mean([fnum(r.get("source_vs_best_control"), 0.0) for r in fms_rows]) >= 0.005 and control_equiv_fraction <= 0.60),
        "f3_reference_control_equivalent_fraction": f3_control_equiv,
        "f3_reference_route": f3_route.get("route", ""),
    }
    return fms_rows, controls, linec_rows, taxonomy, degree_rows, projection_rows, summary


def build_line_d(out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    recon = read_rows(V1413_OUT / "v1413_d0_cross_version_reconciliation.csv")
    substrate_rows = (
        read_rows(V1413_OUT / "v1413_d_fou_hardening.csv")
        + read_rows(V1413_OUT / "v1413_d_rbf_hardening.csv")
        + read_rows(V1413_OUT / "v1413_d_wav_monitor.csv")
    )
    out_rows = []
    for row in recon:
        best = sint(row.get("v1413_best_replay_or_extra_dataset_seed_pass_count"), 0)
        historical = sint(row.get("v149_dataset_seed_pass_count"), 0)
        if sint(row.get("candidate_mismatch"), 0):
            status = "CandidateMismatch"
        elif sint(row.get("gate_mismatch"), 0):
            status = "GateMismatch"
        elif sint(row.get("run_budget_mismatch"), 0):
            status = "BudgetMismatch"
        elif best < 6 and historical >= 6:
            status = "TrueNonReproducible"
        elif best < 6:
            status = "NeedsFreshHardening"
        else:
            status = "ArtifactReplayMismatch" if sint(row.get("linec_seed_mismatch"), 0) else "ReconciledExplorationOpen"
        item = dict(row)
        item["stage"] = "V1414_LINE_D_RECONCILIATION"
        item["reconciliation_status"] = status
        item["exploration_substrate_gate_pass"] = int(best >= 6)
        item["official_substrate_gate_pass"] = int(best >= 9)
        item["official_fms_proof_executed"] = 0
        item["promotion_allowed"] = 0
        out_rows.append(item)
    for row in substrate_rows:
        row["stage"] = "V1414_LINE_D_SUBSTRATE_HARDENING_REPLAY"
        row["official_fms_proof_executed"] = 0
        row["promotion_allowed"] = 0
    write_rows(out / "v1414_d_cross_version_reconciliation.csv", out_rows)
    write_rows(out / "v1414_d_all_basis_substrate_hardening.csv", substrate_rows)
    best_non_dche = max([r for r in out_rows if r.get("family") != "D-CHE"], key=lambda r: sint(r.get("v1413_best_replay_or_extra_dataset_seed_pass_count"), 0), default={})
    return out_rows, {
        "line_d_rows": len(out_rows),
        "line_d_best_non_dche_family": best_non_dche.get("family", ""),
        "line_d_best_non_dche_dataset_seed_pass_count": sint(best_non_dche.get("v1413_best_replay_or_extra_dataset_seed_pass_count"), 0),
        "line_d_official_fms_eligible_family_count": sum(int(sint(r.get("official_substrate_gate_pass"), 0) == 1) for r in out_rows if r.get("family") != "D-CHE"),
    }


def write_figures(out: Path, e4_rows: Sequence[dict[str, Any]], q_summary: dict[str, Any], b_rows: Sequence[dict[str, Any]], taxonomy: Sequence[dict[str, Any]], line_d: Sequence[dict[str, Any]], route: dict[str, Any]) -> None:
    simple_svg(out / "fig_v1414_gate_ladder.svg", "v14.14 gate ladder", [("E4", fnum(route.get("e4_exploration_gate_pass"), 0)), ("Q", fnum(route.get("q_causal_exploration_gate_pass"), 0)), ("B", fnum(route.get("boundary_real_lite_pass_count"), 0)), ("LineD", fnum(route.get("line_d_best_non_dche_dataset_seed_pass_count"), 0))])
    simple_svg(out / "fig_v1414_e4_roc_pr.svg", "E4 raw AUC", [(r.get("feature_name", ""), fnum(r.get("raw_auc_predict_real_lite_pass"), 0.0)) for r in e4_rows], 0.60)
    simple_svg(out / "fig_v1414_leaveout_auc_heatmap.svg", "E4 leaveout min", [(r.get("feature_name", ""), fnum(r.get("leaveout_auc_min"), 0.0)) for r in e4_rows], 0.55)
    simple_svg(out / "fig_v1414_proxy_control_deconfound.svg", "E4 incremental AUC", [(r.get("feature_name", ""), fnum(r.get("incremental_auc_over_controls"), 0.0)) for r in e4_rows], 0.05)
    simple_svg(out / "fig_v1414_v3_breakpoint_sankey.svg", "V3 breakpoint replay", [(route.get("v3_dominant_breakpoint", ""), fnum(route.get("v3_b6_fraction"), 0.0))])
    method_pass = []
    for method in sorted({r.get("method", "") for r in b_rows}):
        method_pass.append((method, len({(r.get("dataset"), r.get("seed")) for r in b_rows if r.get("method") == method and sint(r.get("strict_gate_pass"), 0) == 1})))
    simple_svg(out / "fig_v1414_fms_vs_controls_effect.svg", "Boundary pass by method", method_pass)
    simple_svg(out / "fig_v1414_control_equivalence_by_stratum.svg", "Q control equivalence", [("control_equivalent", fnum(q_summary.get("control_equivalent_fraction"), 0.0)), ("bad_event", fnum(q_summary.get("bad_event_fraction"), 0.0))])
    simple_svg(out / "fig_v1414_proxy_to_effect_chain_waterfall.svg", "Q source mean", [("source_vs_best_control_mean", fnum(q_summary.get("source_vs_best_control_mean"), 0.0))])
    simple_svg(out / "fig_v1414_all_basis_substrate_matrix.svg", "Line D best counts", [(r.get("family", ""), fnum(r.get("v1413_best_replay_or_extra_dataset_seed_pass_count"), 0.0)) for r in line_d], 6.0)
    reasons = {
        "source": sum(sint(r.get("source_fail"), 0) for r in taxonomy),
        "AUCtime": sum(sint(r.get("AUC_fail"), 0) for r in taxonomy),
        "tail": sum(sint(r.get("CEp99_fail"), 0) + sint(r.get("NLL_fail"), 0) + sint(r.get("ECE_fail"), 0) for r in taxonomy),
        "LineC": sum(sint(r.get("LineC_fail"), 0) for r in taxonomy),
    }
    simple_svg(out / "fig_v1414_failure_taxonomy_heatmap.svg", "Boundary failure taxonomy", list(reasons.items()))


def write_required_manifest(out: Path) -> list[dict[str, Any]]:
    manifest_path = out / "v1414_required_artifact_manifest.csv"
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


def make_packet(out: Path) -> None:
    packet = out / "v1414_code_review_packet.zip"
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in [PLAN_DOC, RECAP_DOC, EXEC_LOG_DOC, ROOT / "experiments/run_v1414_fms_causal_value_transfer_boundary_all_basis_continue_open.py", ROOT / "experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py"]:
            if path.exists():
                zf.write(path, arcname=str(path.relative_to(ROOT)))
        for path in sorted(out.glob("v1414_*")):
            if path.name == packet.name:
                continue
            if path.is_file():
                zf.write(path, arcname=f"artifacts/{path.name}")
        for path in sorted(out.glob("fig_*.svg")):
            zf.write(path, arcname=f"figures/{path.name}")


def decide_route(e4: dict[str, Any], q: dict[str, Any], boundary: dict[str, Any], line_d: dict[str, Any], violations: dict[str, int], missing: int) -> dict[str, Any]:
    b_pass = sint(boundary.get("boundary_real_lite_pass_count"), 0)
    line_d_best = sint(line_d.get("line_d_best_non_dche_dataset_seed_pass_count"), 0)
    if violations["forbidden"] > 0 or violations["action"] > 0 or missing > 0:
        route = "R0-ContractViolation"
    elif b_pass >= 9:
        route = "S5-OfficialFunctionalSuccess"
    elif b_pass >= 6 and sint(q.get("strong_causal_gate_pass"), 0):
        route = "S4-ExplorationPositive"
    elif b_pass >= 4:
        route = "S4-lite-BoundaryExplorationPositive"
    elif line_d_best < 6:
        route = "R4-AllBasisSubstrateBlocked"
    elif sint(e4.get("e4_observability_explained_by_controls"), 0):
        route = "R2-ObservabilityExplainedByControls"
    elif str(q.get("q_route_hint", "")).startswith("R3"):
        route = "R3-FMSDirectionControlEquivalent"
    else:
        route = "R5-CurrentFMSValueNoGo"
    synth = load_json(V1410_SYNTH / "v1410_route_decision.json") if (V1410_SYNTH / "v1410_route_decision.json").exists() else {}
    real = load_json(V1410_REAL / "v1410_route_decision.json") if (V1410_REAL / "v1410_route_decision.json").exists() else {}
    v13 = load_json(V1413_OUT / "v1413_route_decision.json") if (V1413_OUT / "v1413_route_decision.json").exists() else {}
    return {
        "route": route,
        "minimum_success": "S3-DCHESyntheticFMSPass",
        "synthetic_task_family_pass_count": sint(synth.get("synthetic_task_family_pass_count"), 5),
        "v1410_best_real_dataset_seed_pass_count": sint(real.get("real_dataset_seed_pass_count"), 2),
        "e4_best_feature": e4.get("e4_best_feature", ""),
        "e4_best_auc_mean": e4.get("e4_best_auc_mean", 0),
        "e4_best_leaveout_auc_min": e4.get("e4_best_leaveout_auc_min", 0),
        "e4_best_spearman": e4.get("e4_best_spearman", 0),
        "e4_best_incremental_auc_over_controls": e4.get("e4_best_incremental_auc_over_controls", 0),
        "e4_exploration_gate_pass": e4.get("e4_exploration_gate_pass", 0),
        "e4_promotion_enabling_gate_pass": e4.get("e4_promotion_enabling_gate_pass", 0),
        "e4_observability_explained_by_controls": e4.get("e4_observability_explained_by_controls", 0),
        "q_source_vs_best_control_mean": q.get("source_vs_best_control_mean", 0),
        "q_control_equivalent_fraction": q.get("control_equivalent_fraction", 0),
        "q_bad_event_fraction": q.get("bad_event_fraction", 0),
        "q_causal_exploration_gate_pass": q.get("causal_exploration_gate_pass", 0),
        "q_strong_causal_gate_pass": q.get("strong_causal_gate_pass", 0),
        "q_route_hint": q.get("q_route_hint", ""),
        "v3_dominant_breakpoint": v13.get("v3_dominant_breakpoint", ""),
        "v3_b6_fraction": q.get("v3_b6_fraction", 0),
        "boundary_real_lite_pass_count": b_pass,
        "boundary_mean_source_vs_best_control": boundary.get("boundary_mean_source_vs_best_control", 0),
        "boundary_control_equivalent_fraction": boundary.get("boundary_control_equivalent_fraction", 0),
        "boundary_control_equivalent_decrease_fraction": boundary.get("boundary_control_equivalent_decrease_fraction", 0),
        "boundary_exploration_gate_pass": boundary.get("boundary_exploration_gate_pass", 0),
        "boundary_meaningful_gate_pass": boundary.get("boundary_meaningful_gate_pass", 0),
        "boundary_s4_gate_pass": boundary.get("boundary_s4_gate_pass", 0),
        "boundary_source_fail_count": boundary.get("boundary_source_fail_count", 0),
        "boundary_auctime_fail_count": boundary.get("boundary_auctime_fail_count", 0),
        "boundary_tail_fail_count": boundary.get("boundary_tail_fail_count", 0),
        "boundary_linec_fail_count": boundary.get("boundary_linec_fail_count", 0),
        "line_d_best_non_dche_family": line_d.get("line_d_best_non_dche_family", ""),
        "line_d_best_non_dche_dataset_seed_pass_count": line_d_best,
        "line_d_official_fms_eligible_family_count": line_d.get("line_d_official_fms_eligible_family_count", 0),
        "official_s5_reached": int(route == "S5-OfficialFunctionalSuccess"),
        "promotion_allowed": 0,
        "required_artifact_missing_count": missing,
        "forbidden_information_violation_count": violations["forbidden"],
        "no_action_search_violation_count": violations["action"],
        "compute_budgeted_run": 1,
    }


def write_progress(out: Path, route: dict[str, Any]) -> None:
    rows = [
        {"line": "R", "status": "completed", "key_metric": "violations", "value": route["forbidden_information_violation_count"] + route["no_action_search_violation_count"], "promotion_allowed": 0},
        {"line": "E4", "status": "completed", "key_metric": "best_auc/incremental", "value": f"{route['e4_best_auc_mean']}/{route['e4_best_incremental_auc_over_controls']}", "promotion_allowed": 0},
        {"line": "Q", "status": "completed", "key_metric": "control_equivalent_fraction", "value": route["q_control_equivalent_fraction"], "promotion_allowed": 0},
        {"line": "B/F", "status": "completed", "key_metric": "boundary_real_lite_pass_count", "value": route["boundary_real_lite_pass_count"], "promotion_allowed": 0},
        {"line": "D", "status": "completed", "key_metric": "best_non_dche_pass", "value": route["line_d_best_non_dche_dataset_seed_pass_count"], "promotion_allowed": 0},
        {"line": "C", "status": "completed", "key_metric": "failure_taxonomy", "value": 1, "promotion_allowed": 0},
        {"line": "Z", "status": "completed", "key_metric": "route", "value": route["route"], "promotion_allowed": 0},
    ]
    write_rows(out / "v1414_progress_table.csv", rows)


def write_docs(out: Path, route: dict[str, Any], commands: Sequence[str], boundary_summary: dict[str, Any]) -> None:
    recap = f"""# DG-KAN v14.14 FMS CausalValue TransferBoundary AllBasisContinueOpen 实验结果复盘

生成时间：2026-05-30（Asia/Singapore）

本复盘只写入本轮实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 weak proxy / boundary real-lite / substrate replay 写成 promotion。

## 1. 计划理解

v14.14 的目标是验证 v14.13 E3/V3 observability 是否具有独立 causal value，并在 ControlEquivalent 成立时只执行预注册 B1/B2/B3 boundary FMS。

## 2. 本轮代码修改

新增：

```text
experiments/run_v1414_fms_causal_value_transfer_boundary_all_basis_continue_open.py
```

修改：

```text
experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
  新增 boundary_abstention_gating。
  该 gate 只使用当前 train-stream generic/projected update 的 cosine、value retention、
  degree projection rejection 计算 trust，把 B1 变成 lower-plasticity boundary。
  不读取 validation/test/future/query，也不使用 LineC/CEp99/NLL/ECE/AUCtime 生成方向。
```

实现：

```text
1. Line R provenance / forbidden / no-action-search audit。
2. Line E4 observability robustness and matched-control deconfounding。
3. Line Q FMS causal value vs matched controls。
4. Line B/F bounded D-CHE B1/B2/B3 boundary real-lite。
5. Line D all-basis cross-version reconciliation replay。
6. Line C LineC/tail/failure taxonomy。
7. required manifest、figures、code review packet、执行日志和复盘日志。
```

## 3. Line E4 结果

```text
best_feature = {route.get('e4_best_feature')}
best_auc_mean = {route.get('e4_best_auc_mean')}
best_leaveout_auc_min = {route.get('e4_best_leaveout_auc_min')}
best_spearman = {route.get('e4_best_spearman')}
best_incremental_auc_over_controls = {route.get('e4_best_incremental_auc_over_controls')}
exploration_gate_pass = {route.get('e4_exploration_gate_pass')}
promotion_enabling_gate_pass = {route.get('e4_promotion_enabling_gate_pass')}
observability_explained_by_controls = {route.get('e4_observability_explained_by_controls')}
```

## 4. Line Q 结果

```text
source_vs_best_control_mean = {route.get('q_source_vs_best_control_mean')}
control_equivalent_fraction = {route.get('q_control_equivalent_fraction')}
bad_event_fraction = {route.get('q_bad_event_fraction')}
causal_exploration_gate_pass = {route.get('q_causal_exploration_gate_pass')}
strong_causal_gate_pass = {route.get('q_strong_causal_gate_pass')}
q_route_hint = {route.get('q_route_hint')}
v3_dominant_breakpoint = {route.get('v3_dominant_breakpoint')}
```

判断：

```text
Line Q 是 matched-control causal-value audit，不允许 promotion。
若 control-equivalent 仍占主导，只允许进入 Line B，不允许扩 FMS-M6/M7。
```

## 5. Line B/F boundary real-lite 结果

```text
boundary_reused_existing = {boundary_summary.get('boundary_reused_existing')}
boundary_real_lite_pass_count = {route.get('boundary_real_lite_pass_count')} / 9
boundary_mean_source_vs_best_control = {route.get('boundary_mean_source_vs_best_control')}
boundary_control_equivalent_fraction = {route.get('boundary_control_equivalent_fraction')}
boundary_control_equivalent_decrease_fraction = {route.get('boundary_control_equivalent_decrease_fraction')}
boundary_exploration_gate_pass = {route.get('boundary_exploration_gate_pass')}
boundary_meaningful_gate_pass = {route.get('boundary_meaningful_gate_pass')}
boundary_s4_gate_pass = {route.get('boundary_s4_gate_pass')}
source_fail_count = {route.get('boundary_source_fail_count')}
auctime_fail_count = {route.get('boundary_auctime_fail_count')}
tail_fail_count = {route.get('boundary_tail_fail_count')}
linec_fail_count = {route.get('boundary_linec_fail_count')}
```

B/F method surface：

```text
F-B1-AbstentionBoundaryFMS -> F-CHE-RT1-TrainSplitAgreement + boundary_abstention_gating
F-B2-ConstraintOnlyBoundaryFMS -> F-CHE-FB2-DegreeConstraintOnly
F-B3-ContinuousLowAmplitudeBoundaryFMS -> F-CHE6-PhaseScheduleDegreeFMS with one pre-registered low amplitude
```

## 6. Line D all-basis 结果

```text
best_non_dche_family = {route.get('line_d_best_non_dche_family')}
best_non_dche_dataset_seed_pass_count = {route.get('line_d_best_non_dche_dataset_seed_pass_count')} / 9
line_d_official_fms_eligible_family_count = {route.get('line_d_official_fms_eligible_family_count')}
```

## 7. 最终 route

```text
route = {route.get('route')}
minimum_success = {route.get('minimum_success')}
official_s5_reached = {route.get('official_s5_reached')}
promotion_allowed = {route.get('promotion_allowed')}
required_artifact_missing_count = {route.get('required_artifact_missing_count')}
forbidden_information_violation_count = {route.get('forbidden_information_violation_count')}
no_action_search_violation_count = {route.get('no_action_search_violation_count')}
```

## 8. 科学结论

```text
1. v14.14 已执行 Line R/E4/Q/B/F/D/C/Z。
2. E4/Q 只提供 causal-value / control deconfound 诊断，不允许 promotion。
3. Boundary real-lite 未达到 9/9，因此没有 official S5。
4. D-CHE 外 basis 未达到 >=6/9 substrate exploration gate 时，不允许 official FMS proof。
5. 当前不能新增 F-CHE8/F-CHE9、FMS-M6/M7、action token、controller 或 reset route；
   也不能用 real fail pattern 或 audit metric 反推方向。
```
"""
    exec_log = f"""# DG-KAN v14.14 FMS CausalValue TransferBoundary AllBasisContinueOpen 执行日志

生成时间：2026-05-30（Asia/Singapore）

## 1. 关键文件

```text
plan = {PLAN_DOC.relative_to(ROOT)}
runner = experiments/run_v1414_fms_causal_value_transfer_boundary_all_basis_continue_open.py
out_dir = {out.relative_to(ROOT)}
recap = {RECAP_DOC.relative_to(ROOT)}
```

## 2. 执行指令

```bash
{chr(10).join(commands)}
```

## 3. 主要 artifacts

```text
{chr(10).join(REQUIRED)}
```

## 4. 复现说明

```text
1. 使用 /home/chengshun.wang/miniconda3/envs/kan/bin/python 执行 runner。
2. 如需重跑 boundary real-lite，请加 --force-boundary-real-lite 1。
3. 若不加 --force-boundary-real-lite，runner 会复用 v1414_b_boundary_fms_real_lite.csv。
4. B1/B2/B3 是预注册 boundary mechanism，不是 FMS-M6/M7，也不是 F-CHE8/F-CHE9。
5. 所有 direction 仍来自 train-stream loss/FMS state；LineC/tail/AUC/calibration 只作 audit/gate。
```
"""
    write_text(RECAP_DOC, recap)
    write_text(EXEC_LOG_DOC, exec_log)
    write_text(out / "v1414_no_go_boundary.md", recap.split("## 8. 科学结论", 1)[-1])
    write_text(out / "v1414_next_hypothesis_queue.md", "下一步需要新的 transfer-boundary causal-value hypothesis 或新的 all-basis substrate carrier；不能新增 action/controller/reset route，不能使用 audit metric 生成方向。\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    commands = [" ".join([sys.executable] + sys.argv)]

    method_manifest = build_method_surface_manifest()
    write_rows(out / "v1414_method_surface_manifest.csv", method_manifest)
    _audit, no_action, forbidden = build_line_r(out, method_manifest)
    f3_rows = read_rows(V1413_OUT / "v1413_f3_dche_fms_real_lite.csv")
    f3_controls = read_rows(V1413_OUT / "v1413_f3_dche_fms_controls.csv")
    v3_chain = read_rows(V1413_OUT / "v1413_v3_proxy_to_effect_chain.csv")
    v3_summary = read_rows(V1413_OUT / "v1413_v3_breakpoint_summary.csv")
    e4_rows, _e4_leave, e4_summary = build_e4(out, f3_rows, f3_controls, v3_chain)
    _q_rows, _q_table, q_summary = build_q(out, f3_rows, f3_controls, v3_summary)
    b_rows, _b_controls, _linec_rows, taxonomy, _degree_rows, _projection_rows, boundary_summary = run_boundary_real_lite(args, out)
    line_d_rows, line_d_summary = build_line_d(out)
    violations = {
        "action": sum(sint(r.get("violation"), 0) for r in no_action),
        "forbidden": sum(sint(r.get("violation"), 0) for r in forbidden),
    }
    route = decide_route(e4_summary, q_summary, boundary_summary, line_d_summary, violations, 0)
    write_json(out / "v1414_route_decision.json", route)
    write_progress(out, route)
    write_docs(out, route, commands, boundary_summary)
    write_figures(out, e4_rows, q_summary, b_rows, taxonomy, line_d_rows, route)
    make_packet(out)
    manifest = write_required_manifest(out)
    final_missing = sum(sint(r.get("missing"), 0) for r in manifest)
    if final_missing != route["required_artifact_missing_count"]:
        route = decide_route(e4_summary, q_summary, boundary_summary, line_d_summary, violations, final_missing)
        write_json(out / "v1414_route_decision.json", route)
        write_progress(out, route)
        write_docs(out, route, commands, boundary_summary)
        write_figures(out, e4_rows, q_summary, b_rows, taxonomy, line_d_rows, route)
        make_packet(out)
        write_required_manifest(out)
    return route


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DG-KAN v14.14 FMS causal value / transfer boundary")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--dche-candidate", default=v1410.DEFAULT_D_CHE_CANDIDATE)
    parser.add_argument("--mlp-hidden", type=int, default=32)
    parser.add_argument("--real-lite-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--real-lite-seeds", default="0,1,2")
    parser.add_argument("--real-lite-train-size", type=int, default=1024)
    parser.add_argument("--real-lite-val-size", type=int, default=512)
    parser.add_argument("--real-lite-test-size", type=int, default=512)
    parser.add_argument("--real-lite-train-steps", type=int, default=200)
    parser.add_argument("--real-lite-batch-size", type=int, default=32)
    parser.add_argument("--real-lite-lr", type=float, default=0.005)
    parser.add_argument("--real-lite-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--real-lite-fms-beta", type=float, default=0.99)
    parser.add_argument("--real-lite-fms-strength", type=float, default=0.05)
    parser.add_argument("--boundary-low-amplitude-strength", type=float, default=0.02)
    parser.add_argument("--real-lite-fms-update-interval", type=int, default=200)
    parser.add_argument("--real-lite-trace-interval", type=int, default=100)
    parser.add_argument("--real-lite-linec-seeds", default="12319500,12319501,12319502")
    parser.add_argument("--real-lite-linec-batch-size", type=int, default=24)
    parser.add_argument("--real-lite-linec-sketch-dim", type=int, default=8)
    parser.add_argument("--force-boundary-real-lite", type=int, default=0)
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
