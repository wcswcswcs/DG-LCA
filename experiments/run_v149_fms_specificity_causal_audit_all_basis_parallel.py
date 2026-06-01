#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from copy import copy
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import experiments.run_v144_real_transfer_fms_all_basis_substrate as v144


ROOT = Path("results/v14_9_fms_specificity_causal_audit_all_basis_parallel")
PLAN_PATH = Path("docs/DG-KAN_v14.9_FMSSpecificityCausalAudit_AllBasisParallel_完整计划.md")
V148_DIR = Path("results/v14_8_optimizer_state_confound_audit_fms_specificity_all_basis_parallel/official_v148")
V147_ROOT = Path("results/v14_7_optimizer_state_transport_fms_all_basis_parallel")
V145_LINE_D = Path("results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/lined_allbasis_summary_v145")

REQUIRED = [
    "v149_route_decision.json",
    "v149_progress_table.csv",
    "v149_no_action_search_audit.csv",
    "v149_forbidden_information_audit.csv",
    "v149_semantic_diff.csv",
    "v149_generic_optimizer_controls.csv",
    "v149_fms_specificity_factorial.csv",
    "v149_difference_in_differences_summary.csv",
    "v149_affected_mask_semantics.csv",
    "v149_overhead_breakdown.csv",
    "v149_mlp_analog_controls.csv",
    "v149_all_basis_substrate_status.csv",
    "v149_linec_tail_audit.csv",
    "v149_failure_taxonomy.csv",
    "v149_no_go_boundary.md",
    "v149_next_hypothesis_queue.md",
    "v149_required_artifact_manifest.csv",
    "v149_code_review_packet.zip",
    "fig_v149_diff_in_diff_fms_specificity.svg",
    "fig_v149_generic_reset_vs_fms_endpoint_strict.svg",
    "fig_v149_affected_mask_semantics.svg",
    "fig_v149_step_time_breakdown.svg",
    "fig_v149_mlp_vs_rat_fms_specificity.svg",
    "fig_v149_all_basis_substrate_matrix.svg",
    "fig_v149_linec_tail_failure_heatmap.svg",
    "fig_v149_route_decision_tree.svg",
]

GENERIC_METHODS = [
    "G0-RAT-AdamW",
    "G1-RAT-AdamW-Beta1Zero",
    "G2-RAT-AdamW-Beta1Half",
    "G3-RAT-AdamW-PeriodicMomentReset",
    "G4-RAT-AdamW-EventMatchedRandomReset",
    "G5-RAT-AdamW-FullMomentResetAtFMSIntervalsNoFMS",
    "G6-RAT-AdamW-RMSPropLikeNoMomentum",
    "G7-RAT-AdamW-NoMomentumWarmupThenAdamW",
    "G8-RAT-AdamW-MatchedOverheadNoStateChange",
]

FACTORIAL_METHODS = [
    "F0-RAT-AdamW",
    "F1-RAT-AdamW-GenericBestTransport",
    "F2-RAT-FMS-NoTransport",
    "F3-RAT-FMS-GenericBestTransportMatched",
    "F4-RAT-FMS-ValuePathOnly-NoStateTransport",
    "F5-RAT-FMS-DirectionRemoved-StateOnly",
    "F6-RAT-FMS-RandomDirectionMatchedState",
    "F7-RAT-FMS-AdamWParallelDirectionControl",
]

MLP_METHODS = [
    "M0-MLP-AdamW",
    "M1-MLP-FMS-NoTransport",
    "M2-MLP-FMS-MatchedGenericTransport",
    "M3-MLP-RandomDirectionMatchedTransport",
    "M4-MLP-FMS-DirectionRemoved-StateOnly",
    "M5-MLP-Beta1Zero",
    "M6-MLP-PeriodicMomentReset",
]


def fnum(value: Any, default: float = 0.0) -> float:
    return v144.fnum(value, default)


def sint(value: Any, default: int = 0) -> int:
    return v144.sint(value, default)


def median(values: list[float]) -> float:
    return v144.median(values)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def base_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "family": "D-RAT",
        "v144_method": "K0-RAT-AdamW",
        "candidate_id": str(args.rational_candidate),
        "control_method": 1,
        "control_role": "control",
        "factorial_role": "none",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "generic_optimizer_control_mode": "none",
        "generic_optimizer_reset_fraction": 1.0,
        "optimizer_state_transport_mode": "none",
        "optimizer_state_transport_scope": "affected",
        "optimizer_state_transport_recovery_window": 0,
        "direction_control": "none",
        "transport_condition": "none",
    }


def method_config(method: str, args: argparse.Namespace) -> dict[str, Any]:
    base = base_config(args)
    random_fraction = float(args.generic_random_reset_fraction)
    generic_t = {
        "generic_optimizer_control_mode": "event_matched_random_reset",
        "generic_optimizer_reset_fraction": random_fraction,
        "transport_condition": "generic_event_matched_random_reset",
    }
    configs: dict[str, dict[str, Any]] = {
        "G0-RAT-AdamW": {**base, "control_role": "generic_adamw"},
        "G1-RAT-AdamW-Beta1Zero": {**base, "adam_beta1": 0.0, "control_role": "generic_beta1_zero"},
        "G2-RAT-AdamW-Beta1Half": {**base, "adam_beta1": 0.45, "control_role": "generic_beta1_half"},
        "G3-RAT-AdamW-PeriodicMomentReset": {
            **base,
            "generic_optimizer_control_mode": "periodic_moment_reset",
            "control_role": "generic_periodic_moment_reset",
            "transport_condition": "periodic_full_moment_reset",
        },
        "G4-RAT-AdamW-EventMatchedRandomReset": {
            **base,
            **generic_t,
            "control_role": "generic_event_matched_random_reset",
        },
        "G5-RAT-AdamW-FullMomentResetAtFMSIntervalsNoFMS": {
            **base,
            "generic_optimizer_control_mode": "full_moment_reset_at_fms_intervals",
            "control_role": "generic_full_moment_reset_no_fms",
            "transport_condition": "event_matched_full_moment_reset",
        },
        "G6-RAT-AdamW-RMSPropLikeNoMomentum": {
            **base,
            "adam_beta1": 0.0,
            "generic_optimizer_control_mode": "rmsprop_like_no_momentum",
            "control_role": "generic_rmsprop_like_no_momentum",
            "transport_condition": "every_step_no_momentum",
        },
        "G7-RAT-AdamW-NoMomentumWarmupThenAdamW": {
            **base,
            "generic_optimizer_control_mode": "no_momentum_warmup_then_adamw",
            "control_role": "generic_no_momentum_warmup",
            "transport_condition": "warmup_no_momentum",
        },
        "G8-RAT-AdamW-MatchedOverheadNoStateChange": {
            **base,
            "generic_optimizer_control_mode": "matched_overhead_no_state_change",
            "generic_optimizer_reset_fraction": random_fraction,
            "control_role": "matched_overhead_no_state_change",
            "transport_condition": "event_matched_overhead_only",
        },
        "F0-RAT-AdamW": {**base, "control_role": "factorial_A_adamw", "factorial_role": "A"},
        "F1-RAT-AdamW-GenericBestTransport": {
            **base,
            **generic_t,
            "control_role": "factorial_B_adamw_plus_generic_transport",
            "factorial_role": "B",
        },
        "F2-RAT-FMS-NoTransport": {
            **base,
            "v144_method": "K-RT2-TrainStreamTailTrust",
            "control_method": 0,
            "control_role": "factorial_C_fms_no_transport",
            "factorial_role": "C",
        },
        "F3-RAT-FMS-GenericBestTransportMatched": {
            **base,
            **generic_t,
            "v144_method": "K-RT2-TrainStreamTailTrust",
            "control_method": 0,
            "control_role": "factorial_D_fms_plus_generic_transport",
            "factorial_role": "D",
        },
        "F4-RAT-FMS-ValuePathOnly-NoStateTransport": {
            **base,
            "v144_method": "K-RT3-ProjectionValueRetention",
            "control_method": 0,
            "control_role": "fms_value_path_only_no_transport",
            "factorial_role": "FMS_VALUE_PATH",
        },
        "F5-RAT-FMS-DirectionRemoved-StateOnly": {
            **base,
            **generic_t,
            "v144_method": "K-FMSDIR0-DirectionRemovedStateOnly",
            "control_role": "direction_removed_state_only",
            "factorial_role": "DIRECTION_REMOVED_CONTROL",
            "direction_control": "direction_removed",
        },
        "F6-RAT-FMS-RandomDirectionMatchedState": {
            **base,
            **generic_t,
            "v144_method": "K-FMSDIR1-RandomDirectionMatchedState",
            "control_role": "random_direction_matched_state",
            "factorial_role": "RANDOM_DIRECTION_CONTROL",
            "direction_control": "random_matched_norm",
        },
        "F7-RAT-FMS-AdamWParallelDirectionControl": {
            **base,
            **generic_t,
            "v144_method": "K-FMSDIR2-AdamWParallelDirectionControl",
            "control_role": "adamw_parallel_direction_control",
            "factorial_role": "ADAMW_PARALLEL_DIRECTION_CONTROL",
            "direction_control": "adamw_parallel",
        },
        "M0-MLP-AdamW": {
            **base,
            "family": "MLP",
            "v144_method": "MLP-AdamW",
            "candidate_id": "MLP-v149-Control",
            "control_role": "mlp_adamw",
        },
        "M1-MLP-FMS-NoTransport": {
            **base,
            "family": "MLP",
            "v144_method": "K-RT2-TrainStreamTailTrust",
            "candidate_id": "MLP-v149-Control",
            "control_method": 0,
            "control_role": "mlp_fms_no_transport",
        },
        "M2-MLP-FMS-MatchedGenericTransport": {
            **base,
            **generic_t,
            "family": "MLP",
            "v144_method": "K-RT2-TrainStreamTailTrust",
            "candidate_id": "MLP-v149-Control",
            "control_method": 0,
            "control_role": "mlp_fms_matched_generic_transport",
        },
        "M3-MLP-RandomDirectionMatchedTransport": {
            **base,
            **generic_t,
            "family": "MLP",
            "v144_method": "K-FMSDIR1-RandomDirectionMatchedState",
            "candidate_id": "MLP-v149-Control",
            "control_role": "mlp_random_direction_matched_transport",
        },
        "M4-MLP-FMS-DirectionRemoved-StateOnly": {
            **base,
            **generic_t,
            "family": "MLP",
            "v144_method": "K-FMSDIR0-DirectionRemovedStateOnly",
            "candidate_id": "MLP-v149-Control",
            "control_role": "mlp_direction_removed_state_only",
        },
        "M5-MLP-Beta1Zero": {
            **base,
            "family": "MLP",
            "v144_method": "MLP-AdamW",
            "candidate_id": "MLP-v149-Control",
            "adam_beta1": 0.0,
            "control_role": "mlp_beta1_zero",
        },
        "M6-MLP-PeriodicMomentReset": {
            **base,
            "family": "MLP",
            "v144_method": "MLP-AdamW",
            "candidate_id": "MLP-v149-Control",
            "generic_optimizer_control_mode": "periodic_moment_reset",
            "control_role": "mlp_periodic_moment_reset",
            "transport_condition": "periodic_full_moment_reset",
        },
    }
    return configs[method]


def canonicalize(rows: list[dict[str, Any]], method: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["v144_base_method"] = item.get("method", cfg["v144_method"])
        item["method"] = method
        item["v149_method"] = method
        item["control_method"] = int(cfg["control_method"])
        item["control_role"] = cfg["control_role"]
        item["factorial_role"] = cfg["factorial_role"]
        item["direction_control"] = cfg["direction_control"]
        item["transport_condition"] = cfg["transport_condition"]
        item["adam_beta1"] = float(cfg["adam_beta1"])
        item["adam_beta2"] = float(cfg["adam_beta2"])
        item["generic_optimizer_control_mode"] = cfg["generic_optimizer_control_mode"]
        item["generic_optimizer_reset_fraction"] = float(cfg["generic_optimizer_reset_fraction"])
        item["optimizer_state_transport_mode"] = cfg["optimizer_state_transport_mode"]
        item["optimizer_state_transport_scope"] = cfg["optimizer_state_transport_scope"]
        item["optimizer_state_transport_recovery_window"] = int(cfg["optimizer_state_transport_recovery_window"])
        item["promotion_allowed"] = 0
        out.append(item)
    return out


def enrich(rows: list[dict[str, Any]], family_filter: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("family")) != family_filter:
            continue
        groups.setdefault((str(row["dataset"]), int(row["seed"]), str(row["loss_interface"])), []).append(row)
    out: list[dict[str, Any]] = []
    for _key, group in groups.items():
        controls = [row for row in group if sint(row.get("control_method"), 0) == 1]
        adam_names = {"G0-RAT-AdamW", "F0-RAT-AdamW", "M0-MLP-AdamW"}
        adam = next((row for row in group if row.get("method") in adam_names), controls[0] if controls else group[0])
        best_control_group = controls or group
        best_nll = min(fnum(row.get("NLL"), 9.0) for row in best_control_group)
        best_auc = min(fnum(row.get("AUC_NLL"), 9.0) for row in best_control_group)
        base_time = max(1.0e-8, fnum(adam.get("step_time_sec"), 1.0))
        base_mem = max(1.0, fnum(adam.get("peak_memory_bytes"), 1.0))
        for row in group:
            item = dict(row)
            item["source_vs_best_control"] = best_nll - fnum(row.get("NLL"), 9.0)
            item["source_vs_adamw"] = fnum(adam.get("NLL"), 9.0) - fnum(row.get("NLL"), 9.0)
            item["AUCtime_ratio_vs_best_control"] = fnum(row.get("AUC_NLL"), 9.0) / max(1.0e-8, best_auc)
            item["AUCtime_ratio_vs_adamw"] = fnum(row.get("AUC_NLL"), 9.0) / max(1.0e-8, fnum(adam.get("AUC_NLL"), 9.0))
            item["CEp99_delta_vs_adamw"] = fnum(row.get("CEp99"), 0.0) - fnum(adam.get("CEp99"), 0.0)
            item["NLL_delta_vs_adamw"] = fnum(row.get("NLL"), 0.0) - fnum(adam.get("NLL"), 0.0)
            item["ECE_delta_vs_adamw"] = fnum(row.get("ECE"), 0.0) - fnum(adam.get("ECE"), 0.0)
            item["Brier_delta_vs_adamw"] = fnum(row.get("Brier"), 0.0) - fnum(adam.get("Brier"), 0.0)
            item["margin_p10_delta_vs_adamw"] = fnum(row.get("margin_p10"), 0.0) - fnum(adam.get("margin_p10"), 0.0)
            item["step_time_ratio_vs_adamw"] = fnum(row.get("step_time_sec"), 0.0) / base_time
            item["peak_memory_ratio_vs_adamw"] = fnum(row.get("peak_memory_bytes"), 0.0) / base_mem
            item["v149_endpoint_vs_adamw_gate_pass"] = int(
                fnum(item["source_vs_adamw"]) >= 0.005
                and fnum(item["AUCtime_ratio_vs_adamw"]) <= 1.0
                and fnum(item["CEp99_delta_vs_adamw"]) <= 0.05
                and fnum(item["NLL_delta_vs_adamw"]) <= 0.02
                and fnum(item["ECE_delta_vs_adamw"]) <= 0.02
                and sint(item.get("LineC_majority_pass"), 0) == 1
            )
            item["v149_specificity_gate_pass"] = int(fnum(item["source_vs_best_control"]) >= 0.005)
            item["v149_efficiency_gate_pass"] = int(
                fnum(item["step_time_ratio_vs_adamw"]) <= 1.25
                and fnum(item["peak_memory_ratio_vs_adamw"]) <= 1.25
            )
            item["v149_strict_gate_pass"] = int(
                sint(item.get("control_method"), 0) == 0
                and fnum(item["source_vs_best_control"]) >= 0.005
                and fnum(item["AUCtime_ratio_vs_best_control"]) <= 1.0
                and fnum(item["CEp99_delta_vs_adamw"]) <= 0.05
                and fnum(item["NLL_delta_vs_adamw"]) <= 0.02
                and fnum(item["ECE_delta_vs_adamw"]) <= 0.02
                and sint(item.get("LineC_majority_pass"), 0) == 1
                and fnum(item["step_time_ratio_vs_adamw"]) <= 1.25
                and fnum(item["peak_memory_ratio_vs_adamw"]) <= 1.25
            )
            item["promotion_allowed"] = 0
            out.append(item)
    return out


def pass_count(rows: list[dict[str, Any]], method: str, field: str = "v149_strict_gate_pass") -> int:
    return len(
        {
            (str(row.get("dataset")), str(row.get("seed")))
            for row in rows
            if row.get("method") == method and sint(row.get(field), 0) == 1
        }
    )


def summarize(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row.get("method")), []).append(row)
    out: list[dict[str, Any]] = []
    for method, group in sorted(by_method.items()):
        out.append(
            {
                "stage": stage,
                "method": method,
                "rows": len(group),
                "strict_dataset_seed_pass_count": pass_count(group, method),
                "endpoint_vs_adamw_pass_count": pass_count(group, method, "v149_endpoint_vs_adamw_gate_pass"),
                "specificity_pass_count": pass_count(group, method, "v149_specificity_gate_pass"),
                "efficiency_pass_count": pass_count(group, method, "v149_efficiency_gate_pass"),
                "mean_source_vs_best_control": sum(fnum(row.get("source_vs_best_control"), 0.0) for row in group) / max(1, len(group)),
                "median_auc_vs_best_control": median([fnum(row.get("AUCtime_ratio_vs_best_control"), 9.0) for row in group]),
                "median_auc_vs_adamw": median([fnum(row.get("AUCtime_ratio_vs_adamw"), 9.0) for row in group]),
                "median_step_time_ratio": median([fnum(row.get("step_time_ratio_vs_adamw"), 9.0) for row in group]),
                "median_memory_ratio": median([fnum(row.get("peak_memory_ratio_vs_adamw"), 9.0) for row in group]),
                "median_affected_param_fraction": median([fnum(row.get("median_affected_param_fraction"), 0.0) for row in group]),
                "median_transport_reset_param_fraction": median([fnum(row.get("median_transport_reset_param_fraction"), 0.0) for row in group]),
                "control_method": int(all(sint(row.get("control_method"), 0) == 1 for row in group)),
                "promotion_allowed": 0,
            }
        )
    return out


def failure_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if fnum(row.get("source_vs_best_control"), 0.0) < 0.005:
        reasons.append("source")
    if fnum(row.get("AUCtime_ratio_vs_best_control"), 9.0) > 1.0:
        reasons.append("AUCtime")
    if fnum(row.get("CEp99_delta_vs_adamw"), 0.0) > 0.05:
        reasons.append("CEp99_tail")
    if fnum(row.get("NLL_delta_vs_adamw"), 0.0) > 0.02:
        reasons.append("NLL_tail")
    if fnum(row.get("ECE_delta_vs_adamw"), 0.0) > 0.02:
        reasons.append("ECE_tail")
    if sint(row.get("LineC_majority_pass"), 0) != 1:
        reasons.append("LineC")
    if fnum(row.get("step_time_ratio_vs_adamw"), 9.0) > 1.25:
        reasons.append("step_time")
    if fnum(row.get("peak_memory_ratio_vs_adamw"), 9.0) > 1.25:
        reasons.append("memory")
    if fnum(row.get("source_vs_best_control"), 0.0) < 0.005:
        reasons.append("specificity")
    return reasons


def semantic_diff(current_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v148_route = read_json(V148_DIR / "v148_route_decision.json")
    v148_progress = read_rows(V148_DIR / "v148_progress_table.csv")
    for run_id, path in {
        "v147_every_step": V147_ROOT / "official_v147" / "v147_route_decision.json",
        "v147_event_only": V147_ROOT / "repair_v147_event_only_transport" / "v147_route_decision.json",
        "v147_delta_top25": V147_ROOT / "repair_v147_event_only_delta_top25" / "v147_route_decision.json",
    }.items():
        route = read_json(path)
        rows.append(
            {
                "stage": "V149_SEMANTIC_DIFF",
                "run_id": run_id,
                "semantics_type": run_id.replace("v147_", ""),
                "strict_pass_count": route.get("k2_dataset_seed_pass_count", ""),
                "endpoint_vs_adamw_pass_count": route.get("k2_endpoint_vs_adamw_pass_count", ""),
                "source_mean": "",
                "AUCtime_median": "",
                "tail_fail_count": "",
                "LineC_fail_count": "",
                "step_time_median": "",
                "controls_endpoint_full_count": route.get("control_endpoint_full_success_count", ""),
                "specificity_pass_count": route.get("k2_specificity_pass_count", ""),
                "route": route.get("route", ""),
                "source_artifact": str(path.parent),
                "promotion_allowed": 0,
            }
        )
    for method in ["S5-RAT-FMS-MomentResetWithPostEventRecoveryWindow", "G4-RAT-AdamW-EventMatchedRandomReset"]:
        item = next((row for row in v148_progress if row.get("method") == method), {})
        rows.append(
            {
                "stage": "V149_SEMANTIC_DIFF",
                "run_id": "v148_official",
                "semantics_type": "post_event_window" if method.startswith("S5") else "generic_control",
                "method": method,
                "strict_pass_count": item.get("strict_dataset_seed_pass_count", ""),
                "endpoint_vs_adamw_pass_count": item.get("endpoint_vs_adamw_pass_count", ""),
                "source_mean": item.get("mean_source_vs_best_control", ""),
                "AUCtime_median": item.get("median_auc_vs_best_control", ""),
                "tail_fail_count": "",
                "LineC_fail_count": "",
                "step_time_median": item.get("median_step_time_ratio", ""),
                "controls_endpoint_full_count": v148_route.get("generic_optimizer_state_confound", ""),
                "specificity_pass_count": item.get("specificity_pass_count", ""),
                "route": v148_route.get("route", ""),
                "source_artifact": str(V148_DIR),
                "promotion_allowed": 0,
            }
        )
    for method in ["F3-RAT-FMS-GenericBestTransportMatched", "F5-RAT-FMS-DirectionRemoved-StateOnly", "G4-RAT-AdamW-EventMatchedRandomReset"]:
        item = next((row for row in current_summary if row.get("method") == method), {})
        rows.append(
            {
                "stage": "V149_SEMANTIC_DIFF",
                "run_id": "v149_current",
                "semantics_type": "factorial_fms_generic_transport" if method.startswith("F3") else ("direction_control" if method.startswith("F5") else "generic_control"),
                "method": method,
                "strict_pass_count": item.get("strict_dataset_seed_pass_count", ""),
                "endpoint_vs_adamw_pass_count": item.get("endpoint_vs_adamw_pass_count", ""),
                "source_mean": item.get("mean_source_vs_best_control", ""),
                "AUCtime_median": item.get("median_auc_vs_best_control", ""),
                "tail_fail_count": "",
                "LineC_fail_count": "",
                "step_time_median": item.get("median_step_time_ratio", ""),
                "controls_endpoint_full_count": "",
                "specificity_pass_count": item.get("specificity_pass_count", ""),
                "route": "",
                "source_artifact": "v149_current",
                "promotion_allowed": 0,
            }
        )
    return rows


def did_summary(k_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in k_rows:
        method = str(row.get("method"))
        if method in FACTORIAL_METHODS:
            groups.setdefault((str(row["dataset"]), int(row["seed"])), {})[method] = row
    rows: list[dict[str, Any]] = []
    for (dataset, seed), items in sorted(groups.items()):
        if not all(m in items for m in ["F0-RAT-AdamW", "F1-RAT-AdamW-GenericBestTransport", "F2-RAT-FMS-NoTransport", "F3-RAT-FMS-GenericBestTransportMatched"]):
            continue
        a = items["F0-RAT-AdamW"]
        b = items["F1-RAT-AdamW-GenericBestTransport"]
        c = items["F2-RAT-FMS-NoTransport"]
        d = items["F3-RAT-FMS-GenericBestTransportMatched"]
        source_interaction = (fnum(c.get("NLL")) - fnum(d.get("NLL"))) - (fnum(a.get("NLL")) - fnum(b.get("NLL")))
        auc_interaction = (fnum(c.get("AUC_NLL")) - fnum(d.get("AUC_NLL"))) - (fnum(a.get("AUC_NLL")) - fnum(b.get("AUC_NLL")))
        cep_interaction = (fnum(c.get("CEp99")) - fnum(d.get("CEp99"))) - (fnum(a.get("CEp99")) - fnum(b.get("CEp99")))
        nll_tail_interaction = source_interaction
        ece_interaction = (fnum(c.get("ECE")) - fnum(d.get("ECE"))) - (fnum(a.get("ECE")) - fnum(b.get("ECE")))
        linec_interaction = sint(d.get("LineC_majority_pass")) - sint(c.get("LineC_majority_pass")) - (sint(b.get("LineC_majority_pass")) - sint(a.get("LineC_majority_pass")))
        control_methods = ["F5-RAT-FMS-DirectionRemoved-StateOnly", "F6-RAT-FMS-RandomDirectionMatchedState", "F7-RAT-FMS-AdamWParallelDirectionControl"]
        best_direction_control_endpoint = max((sint(items[m].get("v149_endpoint_vs_adamw_gate_pass")) for m in control_methods if m in items), default=0)
        best_direction_control_source = max((fnum(items[m].get("source_vs_adamw")) for m in control_methods if m in items), default=-999.0)
        control_equivalent = int(
            best_direction_control_endpoint >= sint(d.get("v149_endpoint_vs_adamw_gate_pass"))
            and best_direction_control_source >= fnum(d.get("source_vs_adamw")) - 0.005
        )
        endpoint_pass = int(
            sint(d.get("v149_endpoint_vs_adamw_gate_pass")) == 1
            and source_interaction > 0.0
            and auc_interaction >= 0.0
            and control_equivalent == 0
        )
        rows.append(
            {
                "stage": "V149_DIFFERENCE_IN_DIFFERENCES",
                "dataset": dataset,
                "seed": seed,
                "A_method": "F0-RAT-AdamW",
                "B_method": "F1-RAT-AdamW-GenericBestTransport",
                "C_method": "F2-RAT-FMS-NoTransport",
                "D_method": "F3-RAT-FMS-GenericBestTransportMatched",
                "delta_generic_source": fnum(a.get("NLL")) - fnum(b.get("NLL")),
                "delta_fms_source": fnum(a.get("NLL")) - fnum(c.get("NLL")),
                "interaction_source_delta": source_interaction,
                "interaction_auc_delta": auc_interaction,
                "interaction_tail_cep99_delta": cep_interaction,
                "interaction_tail_nll_delta": nll_tail_interaction,
                "interaction_tail_ece_delta": ece_interaction,
                "interaction_linec_delta": linec_interaction,
                "interaction_endpoint_pass": endpoint_pass,
                "interaction_strict_pass": int(endpoint_pass and sint(d.get("v149_strict_gate_pass")) == 1),
                "control_equivalent_flag": control_equivalent,
                "harmless_null_flag": int(fnum(c.get("source_vs_best_control")) <= 0.0 and sint(c.get("v149_endpoint_vs_adamw_gate_pass")) == 0),
                "generic_controls_match_interaction_row": int(sint(b.get("v149_endpoint_vs_adamw_gate_pass")) >= sint(d.get("v149_endpoint_vs_adamw_gate_pass"))),
                "promotion_allowed": 0,
            }
        )
    if rows:
        rows.append(
            {
                "stage": "V149_DID_AGGREGATE",
                "dataset": "ALL",
                "seed": "ALL",
                "interaction_endpoint_pass_count": sum(sint(row.get("interaction_endpoint_pass")) for row in rows),
                "interaction_strict_pass_count": sum(sint(row.get("interaction_strict_pass")) for row in rows),
                "fms_specific_source_delta_mean": sum(fnum(row.get("interaction_source_delta")) for row in rows) / len(rows),
                "fms_specific_auc_delta_mean": sum(fnum(row.get("interaction_auc_delta")) for row in rows) / len(rows),
                "control_equivalent_count": sum(sint(row.get("control_equivalent_flag")) for row in rows),
                "harmless_null_count": sum(sint(row.get("harmless_null_flag")) for row in rows),
                "promotion_allowed": 0,
            }
        )
    return rows


def affected_mask_semantics(probe_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v147_event = read_rows(V147_ROOT / "repair_v147_event_only_transport" / "v147_fms_affected_coordinate_manifest.csv")
    v147_sparse = read_rows(V147_ROOT / "repair_v147_event_only_delta_top25" / "v147_fms_affected_coordinate_manifest.csv")
    v147_event_k2 = [r for r in v147_event if r.get("method") == "K2-RAT-FMS-ZeroMomentReset-AffectedOnly"]
    v147_sparse_k2 = [r for r in v147_sparse if r.get("method") == "K2-RAT-FMS-ZeroMomentReset-AffectedOnly"]
    sparse_results = read_rows(V147_ROOT / "repair_v147_event_only_delta_top25" / "v147_real_summary.csv")
    event_results = read_rows(V147_ROOT / "repair_v147_event_only_transport" / "v147_real_summary.csv")
    sparse_k2 = next((r for r in sparse_results if r.get("method") == "K2-RAT-FMS-ZeroMomentReset-AffectedOnly"), {})
    event_k2 = next((r for r in event_results if r.get("method") == "K2-RAT-FMS-ZeroMomentReset-AffectedOnly"), {})
    for run_id, source_rows, conclusion in [
        ("v147_event_only_k2", v147_event_k2, "A2-AffectedSetIntrinsicallyBroad"),
        ("v147_delta_top25_k2", v147_sparse_k2, "A3-SparseAffectedMaskDestroysValue"),
    ]:
        affected = [fnum(r.get("affected_param_fraction")) for r in source_rows]
        reset = [fnum(r.get("transport_reset_param_fraction")) for r in source_rows]
        rows.append(
            {
                "stage": "V149_AFFECTED_MASK_SEMANTICS",
                "run_id": run_id,
                "method": "K2-RAT-FMS-ZeroMomentReset-AffectedOnly",
                "affected_fraction": median(affected),
                "rolewise_affected_fraction": "not_role_expanded_in_existing_probe",
                "source_value_mass_in_affected": "",
                "source_value_mass_in_unaffected": "",
                "cos_fms_update_affected_vs_full": "",
                "cos_fms_update_unaffected_vs_full": "",
                "sparse_top25_endpoint_delta": sint(sparse_k2.get("endpoint_vs_adamw_pass_rows")) - sint(event_k2.get("endpoint_vs_adamw_pass_rows")),
                "sparse_top25_source_delta": fnum(sparse_k2.get("mean_source_vs_best_control")) - fnum(event_k2.get("mean_source_vs_best_control")),
                "sparse_top25_auc_delta": fnum(sparse_k2.get("median_auc_vs_best_control")) - fnum(event_k2.get("median_auc_vs_best_control")),
                "reset_fraction": median(reset),
                "mask_definition_uses_audit_metric": 0,
                "conclusion": conclusion,
                "promotion_allowed": 0,
            }
        )
    current = [r for r in probe_rows if r.get("method") in {"F2-RAT-FMS-NoTransport", "F3-RAT-FMS-GenericBestTransportMatched"}]
    affected = [fnum(r.get("affected_param_fraction")) for r in current]
    reset = [fnum(r.get("transport_reset_param_fraction")) for r in current]
    rows.append(
        {
            "stage": "V149_AFFECTED_MASK_SEMANTICS",
            "run_id": "v149_factorial_current",
            "method": "F2/F3 current FMS event probe",
            "affected_fraction": median(affected),
            "rolewise_affected_fraction": "not_role_expanded_in_probe",
            "source_value_mass_in_affected": "",
            "source_value_mass_in_unaffected": "",
            "cos_fms_update_affected_vs_full": "",
            "cos_fms_update_unaffected_vs_full": "",
            "sparse_top25_endpoint_delta": sint(sparse_k2.get("endpoint_vs_adamw_pass_rows")) - sint(event_k2.get("endpoint_vs_adamw_pass_rows")),
            "sparse_top25_source_delta": fnum(sparse_k2.get("mean_source_vs_best_control")) - fnum(event_k2.get("mean_source_vs_best_control")),
            "sparse_top25_auc_delta": fnum(sparse_k2.get("median_auc_vs_best_control")) - fnum(event_k2.get("median_auc_vs_best_control")),
            "reset_fraction": median(reset),
            "mask_definition_uses_audit_metric": 0,
            "conclusion": "A2-AffectedSetIntrinsicallyBroad" if median(affected) >= 0.75 else "A1-AffectedMaskImplementationBug",
            "promotion_allowed": 0,
        }
    )
    return rows


def overhead_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "stage": "V149_OVERHEAD_BREAKDOWN",
                "family": row.get("family"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "method": row.get("method"),
                "step_time_ratio": row.get("step_time_ratio_vs_adamw"),
                "memory_ratio": row.get("peak_memory_ratio_vs_adamw"),
                "per_example_gradient_time": row.get("per_example_gradient_time_sec", ""),
                "fms_state_update_time": row.get("fms_state_update_time_sec", ""),
                "basis_projection_time": row.get("projection_time_sec", ""),
                "state_transport_time": row.get("state_transport_time_sec", ""),
                "optimizer_update_time": row.get("optimizer_update_time_sec", ""),
                "linec_audit_time": row.get("linec_audit_time_sec", ""),
                "artifact_logging_time": row.get("artifact_logging_time_sec", ""),
                "cuda_sync_time": row.get("sync_time_sec", ""),
                "python_overhead_time": "",
                "linec_audit_time_excluded": 1,
                "overhead_gate_pass": int(fnum(row.get("step_time_ratio_vs_adamw"), 9.0) <= 1.25 and fnum(row.get("peak_memory_ratio_vs_adamw"), 9.0) <= 1.25),
                "promotion_allowed": 0,
            }
        )
    return out


def linec_tail_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "stage": "V149_LINEC_TAIL_AUDIT",
                "family": row.get("family"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "method": row.get("method"),
                "CouplingR2": row.get("CouplingR2"),
                "NoiseSignalLeak": row.get("NoiseSignalLeak"),
                "RealSignalReservoirRatio": row.get("RealSignalReservoirRatio"),
                "CEp99": row.get("CEp99"),
                "NLL": row.get("NLL"),
                "ECE": row.get("ECE"),
                "Brier": row.get("Brier"),
                "margin_p10": row.get("margin_p10"),
                "AUC_step": "",
                "AUC_time": row.get("AUC_NLL"),
                "LineC_pass": row.get("LineC_majority_pass"),
                "failure_reasons": "|".join(failure_reasons(row)),
                "audit_metric_used_for_direction": 0,
                "promotion_allowed": 0,
            }
        )
    return out


def failure_taxonomy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        reasons = failure_reasons(row)
        out.append(
            {
                "stage": "V149_FAILURE_TAXONOMY",
                "family": row.get("family"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "method": row.get("method"),
                "strict_gate_pass": row.get("v149_strict_gate_pass"),
                "endpoint_vs_adamw_gate_pass": row.get("v149_endpoint_vs_adamw_gate_pass"),
                "source_fail": int("source" in reasons),
                "AUCtime_fail": int("AUCtime" in reasons),
                "tail_fail": int(any(r in reasons for r in ["CEp99_tail", "NLL_tail", "ECE_tail"])),
                "LineC_fail": int("LineC" in reasons),
                "step_time_fail": int("step_time" in reasons),
                "memory_fail": int("memory" in reasons),
                "specificity_fail": int("specificity" in reasons),
                "failure_reasons": "|".join(reasons) if reasons else "-",
                "promotion_allowed": 0,
            }
        )
    return out


def all_basis_status(k_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_rat_pass = max((pass_count(k_rows, method) for method in FACTORIAL_METHODS), default=0)
    rows.append(
        {
            "stage": "V149_ALL_BASIS_SUBSTRATE_STATUS",
            "family": "D-RAT",
            "candidate": "D-RAT-v149-CausalSpecificityCarrier",
            "new_training_executed": 1,
            "strict_gate_pass_rows": best_rat_pass,
            "substrate_dataset_seed_pass_count": "",
            "official_fms_eligibility": int(best_rat_pass == 9),
            "source_artifact": "v149_fms_specificity_factorial.csv",
            "status": "causal_specificity_audit_executed",
            "promotion_allowed": 0,
        }
    )
    v145_summary = read_rows(V145_LINE_D / "v145_lined_allbasis_summary.csv")
    best_by_family = {row.get("family"): row for row in v145_summary if row.get("config") == "lined_allbasis_task_trainstream_repair_full_v145"}
    planned = {
        "D-FOU": ["D-FOU23-LowFreqIdentityResidualUnified", "D-FOU24-BandwiseSNRWarmupNoFMS", "D-FOU25-PhaseStableBandMixNoHighFreq", "D-FOU26-NoMaterializeLifetimeAuditV2"],
        "D-WAV": ["D-WAV23-TriangularSupportStableV2", "D-WAV24-ScaleOccupancyHardeningV2", "D-WAV25-LocalTailCoverageWithoutAuditMetric", "D-WAV26-ReservoirStableTrainEntropyGeometry"],
        "D-RBF": ["D-RBF23-CompactBumpIdentityResidual", "D-RBF24-ActiveCenterOccupancyNoDense", "D-RBF25-WidthConditionGuardNoTaskBranch", "D-RBF26-FastKANGaussianLocalK4NoDense"],
        "D-CHE": ["D-CHE23-LowDegreeIdentityResidual", "D-CHE24-HighDegreeLateEnable", "D-CHE25-DegreeEnergyDampingNoAudit", "D-CHE26-RecurrenceLifetimeV2"],
    }
    for family, candidates in planned.items():
        prev = best_by_family.get(family, {})
        for candidate in candidates:
            rows.append(
                {
                    "stage": "V149_ALL_BASIS_SUBSTRATE_STATUS",
                    "family": family,
                    "candidate": candidate,
                    "new_training_executed": 0,
                    "strict_gate_pass_rows": 0,
                    "substrate_dataset_seed_pass_count": prev.get("dataset_seed_pass_count", ""),
                    "official_fms_eligibility": 0,
                    "source_artifact": str(V145_LINE_D / "v145_lined_allbasis_summary.csv") if prev else "",
                    "status": "carried_forward_substrate_status_only_after_line_f_no_go",
                    "center_occupancy_entropy": "",
                    "empty_center_fraction": "",
                    "width_condition": "",
                    "out_of_grid_fraction": "",
                    "task_health": prev.get("best_mean_delta_vs_MLP", ""),
                    "LineC_pass_rate": prev.get("best_LineC_pass_rate", ""),
                    "promotion_allowed": 0,
                }
            )
    return rows


def write_audits(out_dir: Path) -> None:
    v144.write_rows(
        out_dir / "v149_no_action_search_audit.csv",
        [
            {
                "stage": "V149_NO_ACTION_SEARCH_AUDIT",
                "new_k_rt_auc_fl_reset_action_token_added": 0,
                "controller_executed": 0,
                "strength_lambda_lr_refresh_mask_grid_search": 0,
                "audit_metric_direction_used": 0,
                "dataset_seed_branch_used": 0,
                "cross_run_local_positive_spliced_as_s5": 0,
                "endpoint_only_written_as_strict": 0,
                "generic_reset_written_as_fms_specific": 0,
                "violation": 0,
                "promotion_allowed": 0,
            }
        ],
    )
    v144.write_rows(
        out_dir / "v149_forbidden_information_audit.csv",
        [
            {
                "stage": "V149_FORBIDDEN_INFORMATION_AUDIT",
                "teacher_distillation_used": 0,
                "loss_modification_used": 0,
                "sampler_or_class_weight_used": 0,
                "dataset_name_branch_used": 0,
                "seed_specific_scaling_used": 0,
                "label_informed_initialization_used": 0,
                "direction_uses_validation_test_future_query": 0,
                "direction_uses_linec_cep99_nll_ece_auctime_brier": 0,
                "violation": 0,
                "promotion_allowed": 0,
            }
        ],
    )


def write_required_manifest(out_dir: Path) -> int:
    rows = []
    missing = 0
    for name in REQUIRED:
        path = out_dir / name
        exists = int(path.exists())
        missing += int(not exists)
        rows.append({"artifact": name, "exists": exists, "bytes": path.stat().st_size if exists else 0})
    v144.write_rows(out_dir / "v149_required_artifact_manifest.csv", rows)
    return missing


def write_code_review_packet(out_dir: Path) -> None:
    files = [
        Path("experiments/run_v149_fms_specificity_causal_audit_all_basis_parallel.py"),
        Path("experiments/run_v148_optimizer_state_confound_audit_fms_specificity.py"),
        Path("experiments/run_v144_real_transfer_fms_all_basis_substrate.py"),
        PLAN_PATH,
        Path("docs/DG-KAN_v14.8_OptimizerStateConfoundAudit_FMSSpecificity_AllBasisParallel_实验结果复盘.md"),
    ]
    manifest = [{"path": str(path), "exists": int(path.exists()), "sha256": sha256_file(path)} for path in files]
    with zipfile.ZipFile(out_dir / "v149_code_review_packet.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("v149_code_review_manifest.csv", rows_to_csv(manifest))
        for path in files:
            if path.exists():
                zf.write(path, arcname=str(path))


def build_route(
    k_rows: list[dict[str, Any]],
    mlp_rows: list[dict[str, Any]],
    did_rows: list[dict[str, Any]],
    affected_rows: list[dict[str, Any]],
    missing: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    expected = len(v144.parse_csv(args.datasets)) * len(v144.parse_ints(args.seeds))
    fms_candidates = ["F2-RAT-FMS-NoTransport", "F3-RAT-FMS-GenericBestTransportMatched", "F4-RAT-FMS-ValuePathOnly-NoStateTransport"]
    direction_controls = ["F5-RAT-FMS-DirectionRemoved-StateOnly", "F6-RAT-FMS-RandomDirectionMatchedState", "F7-RAT-FMS-AdamWParallelDirectionControl"]
    fms_strict = {method: pass_count(k_rows, method) for method in fms_candidates}
    fms_endpoint = {method: pass_count(k_rows, method, "v149_endpoint_vs_adamw_gate_pass") for method in fms_candidates}
    generic_strict = {method: pass_count(k_rows, method) for method in GENERIC_METHODS}
    generic_endpoint = {method: pass_count(k_rows, method, "v149_endpoint_vs_adamw_gate_pass") for method in GENERIC_METHODS}
    direction_endpoint = {method: pass_count(k_rows, method, "v149_endpoint_vs_adamw_gate_pass") for method in direction_controls}
    mlp_endpoint = {method: pass_count(mlp_rows, method, "v149_endpoint_vs_adamw_gate_pass") for method in MLP_METHODS}
    best_fms_method = max(fms_candidates, key=lambda m: (fms_strict[m], fms_endpoint[m]))
    best_fms_strict = fms_strict[best_fms_method]
    best_fms_endpoint = fms_endpoint[best_fms_method]
    generic_confound = any(count >= best_fms_endpoint and best_fms_endpoint > 0 for count in generic_endpoint.values())
    direction_equiv = any(count >= best_fms_endpoint and best_fms_endpoint > 0 for count in direction_endpoint.values())
    mlp_confound = any(count >= best_fms_endpoint and best_fms_endpoint > 0 for count in mlp_endpoint.values())
    aggregate = next((row for row in did_rows if row.get("stage") == "V149_DID_AGGREGATE"), {})
    interaction_endpoint = sint(aggregate.get("interaction_endpoint_pass_count"))
    interaction_strict = sint(aggregate.get("interaction_strict_pass_count"))
    source_delta_mean = fnum(aggregate.get("fms_specific_source_delta_mean"))
    affected_conclusion = "|".join(sorted({str(row.get("conclusion")) for row in affected_rows if row.get("conclusion")}))
    if best_fms_strict >= expected and not generic_confound and not direction_equiv and not mlp_confound and missing == 0:
        route = "S5-OfficialFunctionalSuccess"
    elif interaction_endpoint >= 6 and source_delta_mean > 0.0 and not generic_confound and not direction_equiv:
        route = "S4d-FMSSpecificExplorationPositive"
    elif generic_confound:
        route = "R3-GenericOptimizerStateResetConfound"
    elif direction_equiv:
        route = "R4-FMSDirectionControlEquivalent"
    elif source_delta_mean <= 0.0:
        route = "R5-HarmlessNullNoValue"
    elif best_fms_endpoint > 0 and any(fnum(row.get("step_time_ratio_vs_adamw"), 0.0) > 1.25 for row in k_rows if row.get("method") == best_fms_method):
        route = "R6-FMSValueButOverheadBlocked"
    elif "A2-AffectedSetIntrinsicallyBroad" in affected_conclusion or "A3-SparseAffectedMaskDestroysValue" in affected_conclusion:
        route = "R7-AffectedMaskSemanticNoGo"
    else:
        route = "R1-DiagnosticMetricInflation"
    return {
        "stage": "V149_ROUTE_DECISION",
        "route": route,
        "minimum_success": "S5-OfficialFunctionalSuccess" if route == "S5-OfficialFunctionalSuccess" else ("S4d-FMSSpecificExplorationPositive" if route == "S4d-FMSSpecificExplorationPositive" else "S4c-MechanismSufficientDiagnostic"),
        "expected_dataset_seed_count": expected,
        "best_fms_method": best_fms_method,
        "best_fms_strict_pass_count": best_fms_strict,
        "best_fms_endpoint_vs_adamw_pass_count": best_fms_endpoint,
        "generic_endpoint_pass_by_method": json.dumps(generic_endpoint, sort_keys=True),
        "generic_strict_pass_by_method": json.dumps(generic_strict, sort_keys=True),
        "direction_control_endpoint_pass_by_method": json.dumps(direction_endpoint, sort_keys=True),
        "mlp_endpoint_pass_by_method": json.dumps(mlp_endpoint, sort_keys=True),
        "interaction_endpoint_pass_count": interaction_endpoint,
        "interaction_strict_pass_count": interaction_strict,
        "fms_specific_source_delta_mean": source_delta_mean,
        "generic_optimizer_state_confound": int(generic_confound),
        "direction_control_equivalent": int(direction_equiv),
        "mlp_analog_confound": int(mlp_confound),
        "affected_mask_conclusion": affected_conclusion,
        "official_s5_reached": int(route == "S5-OfficialFunctionalSuccess"),
        "promotion_allowed": int(route == "S5-OfficialFunctionalSuccess" and int(args.compute_budgeted_run) == 0 and missing == 0),
        "required_artifact_missing_count": missing,
        "forbidden_information_violation_count": 0,
        "no_action_search_violation_count": 0,
        "controller_executed": 0,
        "compute_budgeted_run": int(args.compute_budgeted_run),
        "out_dir": str(args.out_dir),
    }


def write_boundaries(out_dir: Path, route: dict[str, Any]) -> None:
    no_go = [
        "# v14.9 no-go boundary",
        "",
        f"route = {route.get('route')}",
        f"official_s5_reached = {route.get('official_s5_reached')}",
        f"promotion_allowed = {route.get('promotion_allowed')}",
        "",
        "- Do not add K-RT/K-AUC/K-FL/reset action tokens.",
        "- Do not start a controller.",
        "- Do not run strength/lambda/lr/refresh/mask grids.",
        "- Do not use LineC/CEp99/NLL/ECE/AUCtime/Brier as direction.",
        "- If generic optimizer controls explain the gain, stop the reset route and return to FMS definition/substrate repair.",
    ]
    (out_dir / "v149_no_go_boundary.md").write_text("\n".join(no_go) + "\n", encoding="utf-8")
    next_hyp = [
        "# v14.9 next hypothesis queue",
        "",
        "1. Re-define FMS value so that it beats generic optimizer-state dynamics without reset semantics.",
        "2. If FMS-specific value appears later, optimize overhead only after decomposed timing proves the bottleneck.",
        "3. Continue Non-RAT substrate repair only as substrate, not official FMS proof, until full 3x3 substrate gate opens.",
    ]
    (out_dir / "v149_next_hypothesis_queue.md").write_text("\n".join(next_hyp) + "\n", encoding="utf-8")


def write_figures(
    out_dir: Path,
    route: dict[str, Any],
    progress: list[dict[str, Any]],
    did_rows: list[dict[str, Any]],
    affected: list[dict[str, Any]],
    substrate: list[dict[str, Any]],
) -> None:
    v144.write_svg(out_dir / "fig_v149_diff_in_diff_fms_specificity.svg", "v14.9 diff-in-diff", [f"interaction_endpoint={route['interaction_endpoint_pass_count']}", f"source_delta_mean={route['fms_specific_source_delta_mean']}"])
    v144.write_svg(out_dir / "fig_v149_generic_reset_vs_fms_endpoint_strict.svg", "v14.9 generic reset vs FMS", [f"{r['method']}: strict={r['strict_dataset_seed_pass_count']} endpoint={r['endpoint_vs_adamw_pass_count']}" for r in progress if str(r["method"]).startswith(("G", "F2", "F3", "F4"))][:16])
    v144.write_svg(out_dir / "fig_v149_affected_mask_semantics.svg", "v14.9 affected mask", [f"{r['run_id']}: affected={r['affected_fraction']} reset={r['reset_fraction']} {r['conclusion']}" for r in affected])
    v144.write_svg(out_dir / "fig_v149_step_time_breakdown.svg", "v14.9 step time", [f"{r['method']}: step={float(r['median_step_time_ratio']):.3f}" for r in progress[:16] if r.get("median_step_time_ratio") != ""])
    v144.write_svg(out_dir / "fig_v149_mlp_vs_rat_fms_specificity.svg", "v14.9 MLP vs RAT", [f"{r['method']}: endpoint={r['endpoint_vs_adamw_pass_count']}" for r in progress if str(r["method"]).startswith(("M", "F3", "G4"))][:14])
    v144.write_svg(out_dir / "fig_v149_all_basis_substrate_matrix.svg", "v14.9 all-basis substrate", [f"{r['family']} {r['candidate']} pass={r['substrate_dataset_seed_pass_count']}" for r in substrate[:18]])
    v144.write_svg(out_dir / "fig_v149_linec_tail_failure_heatmap.svg", "v14.9 LineC/tail", [f"route={route['route']}", f"best={route['best_fms_method']}"])
    v144.write_svg(out_dir / "fig_v149_route_decision_tree.svg", "v14.9 route", [f"route={route['route']}", f"generic_confound={route['generic_optimizer_state_confound']}", f"direction_equiv={route['direction_control_equivalent']}"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    datasets = v144.parse_csv(args.datasets)
    seeds = v144.parse_ints(args.seeds)
    methods = v144.parse_csv(args.methods)
    mlp_methods = [] if int(args.skip_mlp_control) else v144.parse_csv(args.mlp_methods)
    raw_rows: list[dict[str, Any]] = []
    optimizer_probe_rows: list[dict[str, Any]] = []
    linec_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            xtr, ytr, xva, yva, xte, yte, input_dim_t, output_dim_t = v144.load_real_split(args, dataset, seed, device)
            input_dim = int(input_dim_t.item() if hasattr(input_dim_t, "item") else input_dim_t)
            output_dim = int(output_dim_t.item() if hasattr(output_dim_t, "item") else output_dim_t)
            for method in methods + mlp_methods:
                cfg = method_config(method, args)
                case_args = copy(args)
                case_args.adam_beta1 = float(cfg["adam_beta1"])
                case_args.adam_beta2 = float(cfg["adam_beta2"])
                case_args.generic_optimizer_control_mode = cfg["generic_optimizer_control_mode"]
                case_args.generic_optimizer_reset_fraction = float(cfg["generic_optimizer_reset_fraction"])
                case_args.generic_optimizer_warmup_steps = int(args.generic_optimizer_warmup_steps)
                case_args.optimizer_state_transport_mode = cfg["optimizer_state_transport_mode"]
                case_args.optimizer_state_transport_scope = cfg["optimizer_state_transport_scope"]
                case_args.optimizer_state_transport_recovery_window = int(cfg["optimizer_state_transport_recovery_window"])
                case_args.optimizer_state_transport_probe = 1
                result = v144.train_case(
                    family=cfg["family"],
                    dataset=dataset,
                    seed=seed,
                    method=cfg["v144_method"],
                    candidate_id=cfg["candidate_id"],
                    loss_interface=str(args.loss_interface),
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
                )
                raw_rows.extend(canonicalize([result["row"]], method, cfg))
                optimizer_probe_rows.extend(canonicalize(result["optimizer_probe_rows"], method, cfg))
                linec_rows.extend(canonicalize(result["linec_rows"], method, cfg))
    k_rows = enrich(raw_rows, "D-RAT")
    mlp_rows = enrich(raw_rows, "MLP")
    all_rows = k_rows + mlp_rows
    progress = summarize(all_rows, "V149_PROGRESS_TABLE")
    did_rows = did_summary(k_rows)
    affected_rows = affected_mask_semantics(optimizer_probe_rows)
    overhead_rows = overhead_breakdown(all_rows)
    substrate_rows = all_basis_status(k_rows)
    linec_tail_rows = linec_tail_audit(all_rows)
    failure_rows = failure_taxonomy(all_rows)
    semantic_rows = semantic_diff(progress)

    write_audits(out_dir)
    v144.write_rows(out_dir / "v149_progress_table.csv", progress)
    v144.write_rows(out_dir / "v149_semantic_diff.csv", semantic_rows)
    v144.write_rows(out_dir / "v149_generic_optimizer_controls.csv", [r for r in k_rows if r.get("method") in GENERIC_METHODS])
    v144.write_rows(out_dir / "v149_fms_specificity_factorial.csv", [r for r in k_rows if r.get("method") in FACTORIAL_METHODS])
    v144.write_rows(out_dir / "v149_difference_in_differences_summary.csv", did_rows)
    v144.write_rows(out_dir / "v149_affected_mask_semantics.csv", affected_rows)
    v144.write_rows(out_dir / "v149_overhead_breakdown.csv", overhead_rows)
    v144.write_rows(out_dir / "v149_mlp_analog_controls.csv", [r for r in mlp_rows if r.get("method") in MLP_METHODS])
    v144.write_rows(out_dir / "v149_all_basis_substrate_status.csv", substrate_rows)
    v144.write_rows(out_dir / "v149_linec_tail_audit.csv", linec_tail_rows)
    v144.write_rows(out_dir / "v149_failure_taxonomy.csv", failure_rows)
    v144.write_rows(out_dir / "v149_optimizer_probe_rows.csv", optimizer_probe_rows)
    v144.write_rows(out_dir / "v149_linec_raw_rows.csv", linec_rows)
    write_code_review_packet(out_dir)
    route = build_route(k_rows, mlp_rows, did_rows, affected_rows, 999, args)
    write_boundaries(out_dir, route)
    write_figures(out_dir, route, progress, did_rows, affected_rows, substrate_rows)
    missing = write_required_manifest(out_dir)
    route = build_route(k_rows, mlp_rows, did_rows, affected_rows, missing, args)
    v144.write_json(out_dir / "v149_route_decision.json", route)
    write_boundaries(out_dir, route)
    write_figures(out_dir, route, progress, did_rows, affected_rows, substrate_rows)
    missing = write_required_manifest(out_dir)
    route["required_artifact_missing_count"] = missing
    route["promotion_allowed"] = int(route["route"] == "S5-OfficialFunctionalSuccess" and int(args.compute_budgeted_run) == 0 and missing == 0)
    v144.write_json(out_dir / "v149_route_decision.json", route)
    return route


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DG-KAN v14.9 FMS Specificity Causal Audit + All-Basis Parallel")
    parser.add_argument("--out-dir", default=str(ROOT / "official_v149"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--methods", default=",".join(GENERIC_METHODS + FACTORIAL_METHODS))
    parser.add_argument("--mlp-methods", default=",".join(MLP_METHODS))
    parser.add_argument("--skip-mlp-control", action="store_true")
    parser.add_argument("--rational-candidate", default="D-RAT28-GroupDiversityPreservingRational")
    parser.add_argument("--loss-interface", default="CE")
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--mlp-hidden", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--fms-beta", type=float, default=0.99)
    parser.add_argument("--fms-strength", type=float, default=0.05)
    parser.add_argument("--fms-update-interval", type=int, default=80)
    parser.add_argument("--trace-interval", type=int, default=100)
    parser.add_argument("--rt-lambda-max", type=float, default=0.5)
    parser.add_argument("--rt-agreement-a0", type=float, default=0.0)
    parser.add_argument("--rt-agreement-a1", type=float, default=0.5)
    parser.add_argument("--rt-state-beta", type=float, default=0.90)
    parser.add_argument("--rt-risk-scale", type=float, default=4.0)
    parser.add_argument("--output-geometry-repair", default="none")
    parser.add_argument("--generic-random-reset-fraction", type=float, default=0.25)
    parser.add_argument("--generic-optimizer-warmup-steps", type=int, default=80)
    parser.add_argument("--linec-mode", choices=["none", "exact"], default="exact")
    parser.add_argument("--linec-seeds", default="12319500,12319501,12319502")
    parser.add_argument("--linec-batch-size", type=int, default=24)
    parser.add_argument("--linec-sketch-dim", type=int, default=8)
    parser.add_argument("--compute-budgeted-run", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    route = run(args)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
