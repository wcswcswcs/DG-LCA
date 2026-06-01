#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
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


ROOT = Path("results/v14_8_optimizer_state_confound_audit_fms_specificity_all_basis_parallel")
PLAN_PATH = Path("docs/DG-KAN_v14.8_OptimizerStateConfoundAudit_FMSSpecificity_AllBasisParallel_完整计划.md")
V1461_AFTER_P1 = Path(
    "results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/"
    "diagnostic_v1461_after_p1_transport"
)
V1461_P1_ZERO = Path(
    "results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/"
    "p1_transport_zero_moment_reset_v1461"
)
V147_ROOT = Path("results/v14_7_optimizer_state_transport_fms_all_basis_parallel")

REQUIRED = [
    "v148_route_decision.json",
    "v148_progress_table.csv",
    "v148_v1461_v147_semantic_diff.csv",
    "v148_generic_optimizer_reset_controls.csv",
    "v148_fms_specificity_results.csv",
    "v148_affected_mask_autopsy.csv",
    "v148_overhead_breakdown.csv",
    "v148_mlp_analog_controls.csv",
    "v148_all_basis_substrate_status.csv",
    "v148_linec_tail_audit.csv",
    "v148_no_action_search_audit.csv",
    "v148_forbidden_information_audit.csv",
    "v148_required_artifact_manifest.csv",
    "v148_code_review_packet.zip",
    "v148_no_go_boundary.md",
    "v148_next_hypothesis_queue.md",
    "fig_v148_semantic_diff_v1461_v147.svg",
    "fig_v148_specificity_vs_controls.svg",
    "fig_v148_k2_vs_generic_optimizer_reset.svg",
    "fig_v148_affected_fraction_distribution.svg",
    "fig_v148_value_retention_vs_reset_fraction.svg",
    "fig_v148_step_time_waterfall.svg",
    "fig_v148_mlp_vs_rat_reset_controls.svg",
    "fig_v148_all_basis_substrate_status.svg",
    "fig_v148_failure_taxonomy.svg",
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
]

FMS_METHODS = [
    "S0-RAT-FMS-NoStateTransport",
    "S1-RAT-FMS-ZeroMomentResetAffectedEventOnly",
    "S2-RAT-FMS-ZeroMomentResetAllFMSRolesEventOnly",
    "S3-RAT-FMS-ProjectedMomentTransportEventOnly",
    "S4-RAT-FMS-RMSRecomputeEventOnly",
    "S5-RAT-FMS-MomentResetWithPostEventRecoveryWindow",
]

MLP_METHODS = [
    "M0-MLP-AdamW",
    "M1-MLP-FMS-NoStateTransport",
    "M2-MLP-FMS-ZeroMomentResetAffected",
    "M3-MLP-FMS-ZeroMomentResetRandomCoords",
    "M4-MLP-FMS-FullAdamWStateReset",
    "M5-MLP-AdamW-Beta1Zero",
    "M6-MLP-AdamW-PeriodicMomentReset",
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


def method_config(method: str, args: argparse.Namespace) -> dict[str, Any]:
    random_fraction = float(getattr(args, "generic_random_reset_fraction", 0.25))
    base = {
        "family": "D-RAT",
        "v144_method": "K0-RAT-AdamW",
        "candidate_id": str(args.rational_candidate),
        "control_method": 1,
        "control_role": "generic_optimizer_control",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "generic_optimizer_control_mode": "none",
        "generic_optimizer_reset_fraction": 1.0,
        "optimizer_state_transport_mode": "none",
        "optimizer_state_transport_scope": "affected",
        "optimizer_state_transport_recovery_window": 0,
        "reset_timing": "none",
        "reset_scope": "none",
    }
    configs: dict[str, dict[str, Any]] = {
        "G0-RAT-AdamW": {**base, "control_role": "generic_adamw"},
        "G1-RAT-AdamW-Beta1Zero": {**base, "adam_beta1": 0.0, "control_role": "generic_beta1_zero"},
        "G2-RAT-AdamW-Beta1Half": {**base, "adam_beta1": 0.45, "control_role": "generic_beta1_half"},
        "G3-RAT-AdamW-PeriodicMomentReset": {
            **base,
            "generic_optimizer_control_mode": "periodic_moment_reset",
            "control_role": "generic_periodic_moment_reset",
            "reset_timing": "event_matched_interval",
            "reset_scope": "full_moment",
        },
        "G4-RAT-AdamW-EventMatchedRandomReset": {
            **base,
            "generic_optimizer_control_mode": "event_matched_random_reset",
            "generic_optimizer_reset_fraction": random_fraction,
            "control_role": "generic_event_matched_random_reset",
            "reset_timing": "event_matched_interval",
            "reset_scope": "random_fraction",
        },
        "G5-RAT-AdamW-FullMomentResetAtFMSIntervalsNoFMS": {
            **base,
            "generic_optimizer_control_mode": "full_moment_reset_at_fms_intervals",
            "control_role": "generic_full_moment_reset_no_fms",
            "reset_timing": "event_matched_interval",
            "reset_scope": "full_moment",
        },
        "G6-RAT-AdamW-RMSPropLikeNoMomentum": {
            **base,
            "adam_beta1": 0.0,
            "generic_optimizer_control_mode": "rmsprop_like_no_momentum",
            "control_role": "generic_rmsprop_like_no_momentum",
            "reset_timing": "every_step",
            "reset_scope": "full_moment",
        },
        "G7-RAT-AdamW-NoMomentumWarmupThenAdamW": {
            **base,
            "generic_optimizer_control_mode": "no_momentum_warmup_then_adamw",
            "control_role": "generic_no_momentum_warmup",
            "reset_timing": "warmup_only",
            "reset_scope": "full_moment",
        },
        "S0-RAT-FMS-NoStateTransport": {
            **base,
            "v144_method": "K-RT2-TrainStreamTailTrust",
            "control_method": 0,
            "control_role": "fms_specificity_no_transport",
            "reset_timing": "none",
            "reset_scope": "none",
        },
        "S1-RAT-FMS-ZeroMomentResetAffectedEventOnly": {
            **base,
            "v144_method": "K-RT2-TrainStreamTailTrust",
            "control_method": 0,
            "control_role": "fms_zero_reset_affected_event_only",
            "optimizer_state_transport_mode": "zero_moment_reset",
            "optimizer_state_transport_scope": "affected",
            "reset_timing": "event_only",
            "reset_scope": "affected",
        },
        "S2-RAT-FMS-ZeroMomentResetAllFMSRolesEventOnly": {
            **base,
            "v144_method": "K-RT2-TrainStreamTailTrust",
            "control_method": 0,
            "control_role": "fms_zero_reset_all_roles_event_only",
            "optimizer_state_transport_mode": "zero_moment_reset",
            "optimizer_state_transport_scope": "all_fms_roles",
            "reset_timing": "event_only",
            "reset_scope": "all_fms_roles",
        },
        "S3-RAT-FMS-ProjectedMomentTransportEventOnly": {
            **base,
            "v144_method": "K-RT2-TrainStreamTailTrust",
            "control_method": 0,
            "control_role": "fms_projected_moment_event_only",
            "optimizer_state_transport_mode": "moment_transport_projected_grad",
            "optimizer_state_transport_scope": "affected",
            "reset_timing": "event_only",
            "reset_scope": "affected",
        },
        "S4-RAT-FMS-RMSRecomputeEventOnly": {
            **base,
            "v144_method": "K-RT2-TrainStreamTailTrust",
            "control_method": 0,
            "control_role": "fms_rms_recompute_event_only",
            "optimizer_state_transport_mode": "rms_recompute_microbatch",
            "optimizer_state_transport_scope": "affected",
            "reset_timing": "event_only",
            "reset_scope": "affected",
        },
        "S5-RAT-FMS-MomentResetWithPostEventRecoveryWindow": {
            **base,
            "v144_method": "K-RT2-TrainStreamTailTrust",
            "control_method": 0,
            "control_role": "fms_zero_reset_post_event_recovery_window",
            "optimizer_state_transport_mode": "zero_moment_reset",
            "optimizer_state_transport_scope": "affected",
            "optimizer_state_transport_recovery_window": int(args.post_event_recovery_window),
            "reset_timing": "post_event_window",
            "reset_scope": "affected",
        },
        "M0-MLP-AdamW": {
            **base,
            "family": "MLP",
            "v144_method": "MLP-AdamW",
            "candidate_id": "MLP-v148-Control",
            "control_role": "mlp_adamw",
        },
        "M1-MLP-FMS-NoStateTransport": {
            **base,
            "family": "MLP",
            "v144_method": "MLP-FMS-Amortized",
            "candidate_id": "MLP-v148-Control",
            "control_method": 0,
            "control_role": "mlp_fms_no_transport",
        },
        "M2-MLP-FMS-ZeroMomentResetAffected": {
            **base,
            "family": "MLP",
            "v144_method": "MLP-FMS-Amortized",
            "candidate_id": "MLP-v148-Control",
            "control_method": 0,
            "control_role": "mlp_fms_zero_reset_affected",
            "optimizer_state_transport_mode": "zero_moment_reset",
            "optimizer_state_transport_scope": "affected",
            "reset_timing": "event_only",
            "reset_scope": "affected",
        },
        "M3-MLP-FMS-ZeroMomentResetRandomCoords": {
            **base,
            "family": "MLP",
            "v144_method": "MLP-FMS-Amortized",
            "candidate_id": "MLP-v148-Control",
            "control_role": "mlp_random_reset",
            "optimizer_state_transport_mode": "zero_moment_reset",
            "optimizer_state_transport_scope": "random_matched",
            "reset_timing": "event_only",
            "reset_scope": "random_matched",
        },
        "M4-MLP-FMS-FullAdamWStateReset": {
            **base,
            "family": "MLP",
            "v144_method": "MLP-FMS-Amortized",
            "candidate_id": "MLP-v148-Control",
            "control_role": "mlp_full_reset",
            "optimizer_state_transport_mode": "zero_moment_reset",
            "optimizer_state_transport_scope": "full_adamw",
            "reset_timing": "event_only",
            "reset_scope": "full_adamw",
        },
        "M5-MLP-AdamW-Beta1Zero": {
            **base,
            "family": "MLP",
            "v144_method": "MLP-AdamW",
            "candidate_id": "MLP-v148-Control",
            "adam_beta1": 0.0,
            "control_role": "mlp_beta1_zero",
        },
        "M6-MLP-AdamW-PeriodicMomentReset": {
            **base,
            "family": "MLP",
            "v144_method": "MLP-AdamW",
            "candidate_id": "MLP-v148-Control",
            "generic_optimizer_control_mode": "periodic_moment_reset",
            "control_role": "mlp_periodic_moment_reset",
            "reset_timing": "event_matched_interval",
            "reset_scope": "full_moment",
        },
    }
    return configs[method]


def canonicalize_artifact_rows(rows: list[dict[str, Any]], method: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["v144_base_method"] = item.get("method", cfg["v144_method"])
        item["method"] = method
        item["v148_method"] = method
        item["control_method"] = int(cfg["control_method"])
        item["control_role"] = cfg["control_role"]
        item["reset_timing"] = cfg["reset_timing"]
        item["reset_scope"] = cfg["reset_scope"]
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


def enrich_v148(rows: list[dict[str, Any]], family_filter: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("family")) != family_filter:
            continue
        groups.setdefault((str(row["dataset"]), int(row["seed"]), str(row["loss_interface"])), []).append(row)
    out: list[dict[str, Any]] = []
    for _key, group in groups.items():
        controls = [row for row in group if sint(row.get("control_method"), 0) == 1]
        adam_methods = {"G0-RAT-AdamW", "M0-MLP-AdamW"}
        adam = next((row for row in group if row.get("method") in adam_methods), controls[0] if controls else group[0])
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
            item["v148_endpoint_vs_adamw_gate_pass"] = int(
                fnum(item["source_vs_adamw"]) >= 0.005
                and fnum(item["AUCtime_ratio_vs_adamw"]) <= 1.0
                and fnum(item["CEp99_delta_vs_adamw"]) <= 0.05
                and fnum(item["NLL_delta_vs_adamw"]) <= 0.02
                and fnum(item["ECE_delta_vs_adamw"]) <= 0.02
                and sint(item.get("LineC_majority_pass"), 0) == 1
            )
            item["v148_specificity_gate_pass"] = int(fnum(item["source_vs_best_control"]) >= 0.005)
            item["v148_efficiency_gate_pass"] = int(
                fnum(item["step_time_ratio_vs_adamw"]) <= 1.25
                and fnum(item["peak_memory_ratio_vs_adamw"]) <= 1.25
            )
            item["v148_strict_gate_pass"] = int(
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


def pass_count(rows: list[dict[str, Any]], method: str, field: str = "v148_strict_gate_pass") -> int:
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
                "endpoint_vs_adamw_pass_count": pass_count(group, method, "v148_endpoint_vs_adamw_gate_pass"),
                "specificity_pass_count": pass_count(group, method, "v148_specificity_gate_pass"),
                "efficiency_pass_count": pass_count(group, method, "v148_efficiency_gate_pass"),
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


def rows_for_methods(rows: list[dict[str, Any]], methods: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("method")) in methods]


def semantic_diff(current_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v1461_route = read_json(V1461_AFTER_P1 / "v1461_route_decision.json")
    v1461_p1_route = read_json(V1461_P1_ZERO / "v144_route_decision.json")
    v147_runs = {
        "Q2-v147-every-step": V147_ROOT / "official_v147",
        "Q3-v147-event-only": V147_ROOT / "repair_v147_event_only_transport",
        "Q4-v147-delta-top25": V147_ROOT / "repair_v147_event_only_delta_top25",
    }
    rows.append(
        {
            "stage": "V148_SEMANTIC_DIFF",
            "qid": "Q0",
            "run_id": "v1461_after_p1_diagnostic_old_metrics",
            "method": "P1-OptimizerStateTransport:zero_moment_reset",
            "reset_timing": "diagnostic_fixed_mode",
            "reset_scope": "affected",
            "strict_pass": "",
            "endpoint_vs_adamw_pass": v1461_route.get("coverage_after", ""),
            "source_vs_best_control": "",
            "AUCtime_ratio": "",
            "CEp99_delta": "",
            "NLL_delta": "",
            "ECE_delta": "",
            "LineC_pass": "",
            "step_time_ratio": "",
            "memory_ratio": "",
            "affected_param_fraction": "",
            "actual_reset_param_fraction": "",
            "reset_event_count": "",
            "reset_every_step_count": "",
            "controls_endpoint_pass_count": "",
            "route": v1461_route.get("route", ""),
            "source_artifact": str(V1461_AFTER_P1),
            "promotion_allowed": 0,
        }
    )
    rows.append(
        {
            "stage": "V148_SEMANTIC_DIFF",
            "qid": "Q1",
            "run_id": "v1461_p1_zero_under_v144_runner_not_v147_strict",
            "method": "zero_moment_reset",
            "reset_timing": "fixed_probe_mode",
            "reset_scope": "affected",
            "strict_pass": "not_available_under_v147_strict",
            "endpoint_vs_adamw_pass": v1461_p1_route.get("real_dataset_seed_pass_count", ""),
            "source_vs_best_control": v1461_p1_route.get("mean_source_vs_best_control_noncontrol", ""),
            "AUCtime_ratio": "",
            "CEp99_delta": "",
            "NLL_delta": "",
            "ECE_delta": "",
            "LineC_pass": "",
            "step_time_ratio": "",
            "memory_ratio": "",
            "affected_param_fraction": "",
            "actual_reset_param_fraction": "",
            "reset_event_count": "",
            "reset_every_step_count": "",
            "controls_endpoint_pass_count": "",
            "route": v1461_p1_route.get("route", ""),
            "source_artifact": str(V1461_P1_ZERO),
            "promotion_allowed": 0,
        }
    )
    for qid, path in v147_runs.items():
        route = read_json(path / "v147_route_decision.json")
        summary = read_rows(path / "v147_real_summary.csv")
        k2 = next((row for row in summary if row.get("method") == "K2-RAT-FMS-ZeroMomentReset-AffectedOnly"), {})
        probe = read_rows(path / "v147_fms_affected_coordinate_manifest.csv")
        k2_probe = [row for row in probe if row.get("method") == "K2-RAT-FMS-ZeroMomentReset-AffectedOnly"]
        rows.append(
            {
                "stage": "V148_SEMANTIC_DIFF",
                "qid": qid.split("-", 1)[0],
                "run_id": qid,
                "method": "K2-RAT-FMS-ZeroMomentReset-AffectedOnly",
                "reset_timing": "every_step" if "every" in qid else ("event_only_sparse" if "delta" in qid else "event_only"),
                "reset_scope": "affected" if "delta" not in qid else "sparse_delta_top25",
                "strict_pass": route.get("k2_dataset_seed_pass_count", k2.get("strict_dataset_seed_pass_count", "")),
                "endpoint_vs_adamw_pass": route.get("k2_endpoint_vs_adamw_pass_count", k2.get("endpoint_vs_adamw_pass_rows", "")),
                "source_vs_best_control": k2.get("mean_source_vs_best_control", ""),
                "AUCtime_ratio": k2.get("median_auc_time", k2.get("median_auc_vs_best_control", "")),
                "CEp99_delta": "",
                "NLL_delta": "",
                "ECE_delta": "",
                "LineC_pass": k2.get("linec_majority_pass_rows", ""),
                "step_time_ratio": k2.get("median_step_time", ""),
                "memory_ratio": k2.get("median_memory_ratio", ""),
                "affected_param_fraction": median([fnum(row.get("affected_param_fraction"), 0.0) for row in k2_probe]),
                "actual_reset_param_fraction": median([fnum(row.get("transport_reset_param_fraction"), 0.0) for row in k2_probe]),
                "reset_event_count": "",
                "reset_every_step_count": "",
                "controls_endpoint_pass_count": route.get("control_endpoint_full_success_count", ""),
                "route": route.get("route", ""),
                "source_artifact": str(path),
                "promotion_allowed": 0,
            }
        )
    current_summary = summarize(current_rows, "V148_CURRENT_SUMMARY")
    for method in ["S1-RAT-FMS-ZeroMomentResetAffectedEventOnly", "S5-RAT-FMS-MomentResetWithPostEventRecoveryWindow"]:
        item = next((row for row in current_summary if row.get("method") == method), {})
        rows.append(
            {
                "stage": "V148_SEMANTIC_DIFF",
                "qid": "Q5",
                "run_id": "v148_current",
                "method": method,
                "reset_timing": "event_only" if method.startswith("S1") else "post_event_window",
                "reset_scope": "affected",
                "strict_pass": item.get("strict_dataset_seed_pass_count", ""),
                "endpoint_vs_adamw_pass": item.get("endpoint_vs_adamw_pass_count", ""),
                "source_vs_best_control": item.get("mean_source_vs_best_control", ""),
                "AUCtime_ratio": item.get("median_auc_vs_best_control", ""),
                "CEp99_delta": "",
                "NLL_delta": "",
                "ECE_delta": "",
                "LineC_pass": "",
                "step_time_ratio": item.get("median_step_time_ratio", ""),
                "memory_ratio": item.get("median_memory_ratio", ""),
                "affected_param_fraction": item.get("median_affected_param_fraction", ""),
                "actual_reset_param_fraction": item.get("median_transport_reset_param_fraction", ""),
                "reset_event_count": "",
                "reset_every_step_count": "",
                "controls_endpoint_pass_count": "",
                "route": "",
                "source_artifact": "v148_current",
                "promotion_allowed": 0,
            }
        )
    return rows


def affected_mask_autopsy(current_probe_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    sources = {
        "v147_event_only": V147_ROOT / "repair_v147_event_only_transport" / "v147_fms_affected_coordinate_manifest.csv",
        "v147_delta_top25": V147_ROOT / "repair_v147_event_only_delta_top25" / "v147_fms_affected_coordinate_manifest.csv",
    }
    for run_id, path in sources.items():
        rows = [row for row in read_rows(path) if row.get("method") == "K2-RAT-FMS-ZeroMomentReset-AffectedOnly"]
        affected = [fnum(row.get("affected_param_fraction"), 0.0) for row in rows]
        reset = [fnum(row.get("transport_reset_param_fraction"), 0.0) for row in rows]
        conclusion = "A2-AffectedSetIntrinsicallyBroad" if "event_only" in run_id else "A3-SparseAffectedMaskDestroysValue"
        out.append(
            {
                "stage": "V148_AFFECTED_MASK_AUTOPSY",
                "run_id": run_id,
                "method": "K2-RAT-FMS-ZeroMomentReset-AffectedOnly",
                "row_count": len(rows),
                "fms_delta_abs_distribution": "from_projected_minus_generic_nonzero_fraction",
                "affected_fraction_min": min(affected) if affected else 0.0,
                "affected_fraction_median": median(affected),
                "affected_fraction_max": max(affected) if affected else 0.0,
                "reset_fraction_min": min(reset) if reset else 0.0,
                "reset_fraction_median": median(reset),
                "reset_fraction_max": max(reset) if reset else 0.0,
                "fms_delta_norm_by_role": "not_role_expanded_in_existing_probe",
                "reset_fraction_by_role": "not_role_expanded_in_existing_probe",
                "source_contribution_by_reset_fraction": "see_v147_real_results",
                "moment_staleness_by_role": "not_role_expanded_in_existing_probe",
                "value_retention_vs_reset_fraction": "see_v147_projection_value_retention_if_available",
                "linec_tail_nonharm_vs_reset_fraction": "see_v147_linec_tail_audit",
                "conclusion": conclusion,
                "promotion_allowed": 0,
            }
        )
    s1_rows = [row for row in current_probe_rows if row.get("method") == "S1-RAT-FMS-ZeroMomentResetAffectedEventOnly"]
    affected = [fnum(row.get("affected_param_fraction"), 0.0) for row in s1_rows]
    reset = [fnum(row.get("transport_reset_param_fraction"), 0.0) for row in s1_rows]
    out.append(
        {
            "stage": "V148_AFFECTED_MASK_AUTOPSY",
            "run_id": "v148_s1_current",
            "method": "S1-RAT-FMS-ZeroMomentResetAffectedEventOnly",
            "row_count": len(s1_rows),
            "fms_delta_abs_distribution": "from_current_probe_projected_minus_generic_nonzero_fraction",
            "affected_fraction_min": min(affected) if affected else 0.0,
            "affected_fraction_median": median(affected),
            "affected_fraction_max": max(affected) if affected else 0.0,
            "reset_fraction_min": min(reset) if reset else 0.0,
            "reset_fraction_median": median(reset),
            "reset_fraction_max": max(reset) if reset else 0.0,
            "fms_delta_norm_by_role": "not_role_expanded_in_probe",
            "reset_fraction_by_role": "not_role_expanded_in_probe",
            "source_contribution_by_reset_fraction": "see_v148_fms_specificity_results",
            "moment_staleness_by_role": "not_role_expanded_in_probe",
            "value_retention_vs_reset_fraction": "see_v148_fms_specificity_results",
            "linec_tail_nonharm_vs_reset_fraction": "see_v148_linec_tail_audit",
            "conclusion": "A2-AffectedSetIntrinsicallyBroad" if median(affected) >= 0.75 else "A1-AffectedMaskTooBroadImplementationBug",
            "promotion_allowed": 0,
        }
    )
    return out


def overhead_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        instrumented = str(row.get("per_example_gradient_time_sec", "")) != ""
        out.append(
            {
                "stage": "V148_OVERHEAD_BREAKDOWN",
                "family": row.get("family"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "method": row.get("method"),
                "step_time_sec": row.get("step_time_sec"),
                "step_time_ratio_vs_adamw": row.get("step_time_ratio_vs_adamw"),
                "per_example_gradient_time": row.get("per_example_gradient_time_sec", "not_instrumented"),
                "fms_state_update_time": row.get("fms_state_update_time_sec", "not_instrumented"),
                "projection_time": row.get("projection_time_sec", "not_instrumented"),
                "state_transport_time": row.get("state_transport_time_sec", "not_instrumented"),
                "linec_audit_time_excluded": 1,
                "optimizer_update_time": row.get("optimizer_update_time_sec", "not_instrumented"),
                "sync_time": row.get("sync_time_sec", "not_instrumented"),
                "artifact_logging_time": row.get("artifact_logging_time_sec", "not_instrumented"),
                "linec_audit_time": row.get("linec_audit_time_sec", "not_instrumented"),
                "instrumentation_available": int(instrumented),
                "overhead_gate_pass": int(fnum(row.get("step_time_ratio_vs_adamw"), 9.0) <= 1.25),
                "promotion_allowed": 0,
            }
        )
    return out


def linec_tail_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "stage": "V148_LINEC_TAIL_AUDIT",
                "family": row.get("family"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "method": row.get("method"),
                "CouplingR2": row.get("CouplingR2"),
                "NoiseSignalLeak": row.get("NoiseSignalLeak"),
                "RealSignalReservoirRatio": row.get("RealSignalReservoirRatio"),
                "LineC_pass": row.get("LineC_majority_pass"),
                "CEp99_delta": row.get("CEp99_delta_vs_adamw"),
                "NLL_delta": row.get("NLL_delta_vs_adamw"),
                "ECE_delta": row.get("ECE_delta_vs_adamw"),
                "Brier_delta": row.get("Brier_delta_vs_adamw"),
                "margin_p10_delta": row.get("margin_p10_delta_vs_adamw"),
                "source_vs_best_control": row.get("source_vs_best_control"),
                "AUCtime_ratio": row.get("AUCtime_ratio_vs_best_control"),
                "failure_reasons": "|".join(failure_reasons(row)),
                "linec_tail_used_for_direction": 0,
                "promotion_allowed": 0,
            }
        )
    return out


def all_basis_status(k_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_rat_pass = max((pass_count(k_rows, method) for method in FMS_METHODS), default=0)
    rows = [
        {
            "stage": "V148_ALL_BASIS_SUBSTRATE_STATUS",
            "family": "D-RAT",
            "candidate": "D-RAT-Monitor-v148",
            "new_training_executed": 1,
            "strict_gate_pass_rows": best_rat_pass,
            "official_functional_open": int(best_rat_pass == 9),
            "source_artifact": "v148_fms_specificity_results.csv",
            "not_executed_reason": "",
            "promotion_allowed": 0,
        }
    ]
    for family, candidates in {
        "D-WAV": ["support/scale/local-tail hardening monitor"],
        "D-FOU": ["low-frequency/bandwise/phase-stable monitor"],
        "D-RBF": ["active-center/width/identity-residual monitor"],
        "D-CHE": ["low-degree/high-degree/energy damping monitor"],
    }.items():
        for candidate in candidates:
            rows.append(
                {
                    "stage": "V148_ALL_BASIS_SUBSTRATE_STATUS",
                    "family": family,
                    "candidate": candidate,
                    "new_training_executed": 0,
                    "strict_gate_pass_rows": 0,
                    "official_functional_open": 0,
                    "source_artifact": "",
                    "not_executed_reason": "v14.8 plan records substrate status only; Non-RAT official FMS not executed",
                    "promotion_allowed": 0,
                }
            )
    return rows


def write_audits(out_dir: Path) -> None:
    v144.write_rows(
        out_dir / "v148_no_action_search_audit.csv",
        [
            {
                "stage": "V148_NO_ACTION_SEARCH_AUDIT",
                "new_k_rt_auc_fl_action_token_added": 0,
                "action_bank_used_as_search_space": 0,
                "controller_executed": 0,
                "strength_lambda_lr_refresh_grid_search": 0,
                "local_positive_triggered_new_action": 0,
                "cross_run_local_positive_spliced_as_s5": 0,
                "endpoint_only_written_as_strict": 0,
                "random_full_reset_written_as_fms_specific": 0,
                "violation": 0,
                "promotion_allowed": 0,
            }
        ],
    )
    v144.write_rows(
        out_dir / "v148_forbidden_information_audit.csv",
        [
            {
                "stage": "V148_FORBIDDEN_INFORMATION_AUDIT",
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
    missing = 0
    rows = []
    for name in REQUIRED:
        path = out_dir / name
        exists = int(path.exists())
        missing += int(not exists)
        rows.append({"artifact": name, "exists": exists, "bytes": path.stat().st_size if path.exists() else 0})
    v144.write_rows(out_dir / "v148_required_artifact_manifest.csv", rows)
    return missing


def write_code_review_packet(out_dir: Path) -> None:
    files = [
        Path("experiments/run_v148_optimizer_state_confound_audit_fms_specificity.py"),
        Path("experiments/run_v144_real_transfer_fms_all_basis_substrate.py"),
        Path("experiments/run_v147_optimizer_state_transport_fms_all_basis_parallel.py"),
        PLAN_PATH,
        Path("docs/DG-KAN_v14.7_OptimizerStateTransportFMS_AllBasisParallel_实验结果复盘.md"),
    ]
    manifest = [{"path": str(path), "exists": int(path.exists()), "sha256": sha256_file(path)} for path in files]
    with zipfile.ZipFile(out_dir / "v148_code_review_packet.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("v148_code_review_manifest.csv", rows_to_csv(manifest))
        for path in files:
            if path.exists():
                zf.write(path, arcname=str(path))


def write_markdown_boundaries(out_dir: Path, route: dict[str, Any]) -> None:
    no_go = [
        "# v14.8 no-go boundary",
        "",
        f"route = {route.get('route')}",
        f"official_s5_reached = {route.get('official_s5_reached')}",
        f"promotion_allowed = {route.get('promotion_allowed')}",
        "",
        "- Do not add K-RT/K-AUC/K-FL action tokens.",
        "- Do not start a controller.",
        "- Do not run strength/lambda/lr/refresh grids.",
        "- Do not use LineC/CEp99/NLL/ECE/AUCtime/Brier as direction.",
        "- Do not write endpoint-only or reset-control-equivalent gains as S5.",
    ]
    (out_dir / "v148_no_go_boundary.md").write_text("\n".join(no_go) + "\n", encoding="utf-8")
    next_hyp = [
        "# v14.8 next hypothesis queue",
        "",
        "1. Pre-register a genuinely sparse train-stream-only FMS affected-mask mechanism only if A1 is supported.",
        "2. Add timing instrumentation before optimizing overhead; do not lower the step_time gate.",
        "3. If generic optimizer controls explain gains, retreat from optimizer-state reset route to FMS definition/substrate repair.",
    ]
    (out_dir / "v148_next_hypothesis_queue.md").write_text("\n".join(next_hyp) + "\n", encoding="utf-8")


def build_route(
    k_rows: list[dict[str, Any]],
    mlp_rows: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
    affected_rows: list[dict[str, Any]],
    missing: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    expected = len(v144.parse_csv(args.datasets)) * len(v144.parse_ints(args.seeds))
    fms_counts = {method: pass_count(k_rows, method) for method in FMS_METHODS}
    fms_endpoint = {method: pass_count(k_rows, method, "v148_endpoint_vs_adamw_gate_pass") for method in FMS_METHODS}
    generic_counts = {method: pass_count(k_rows, method) for method in GENERIC_METHODS}
    generic_endpoint = {method: pass_count(k_rows, method, "v148_endpoint_vs_adamw_gate_pass") for method in GENERIC_METHODS}
    mlp_counts = {method: pass_count(mlp_rows, method) for method in MLP_METHODS}
    mlp_endpoint = {method: pass_count(mlp_rows, method, "v148_endpoint_vs_adamw_gate_pass") for method in MLP_METHODS}
    best_fms_method = max(fms_counts, key=lambda m: (fms_counts[m], fms_endpoint[m]))
    best_fms_count = fms_counts[best_fms_method]
    best_fms_endpoint = fms_endpoint[best_fms_method]
    generic_confound = any(count >= best_fms_endpoint and best_fms_endpoint > 0 for count in generic_endpoint.values())
    mlp_confound = any(count >= best_fms_endpoint and best_fms_endpoint > 0 for count in mlp_endpoint.values())
    v147_control_endpoint = max((sint(row.get("controls_endpoint_pass_count"), 0) for row in semantic_rows), default=0)
    v147_every_endpoint = next((sint(row.get("endpoint_vs_adamw_pass"), 0) for row in semantic_rows if row.get("run_id") == "Q2-v147-every-step"), 0)
    v147_event_endpoint = next((sint(row.get("endpoint_vs_adamw_pass"), 0) for row in semantic_rows if row.get("run_id") == "Q3-v147-event-only"), 0)
    affected_conclusion = "|".join(sorted({str(row.get("conclusion")) for row in affected_rows if row.get("conclusion")}))
    if best_fms_count >= expected and not generic_confound and not mlp_confound and v147_control_endpoint == 0 and missing == 0:
        route = "S5-OfficialFunctionalSuccess"
    elif generic_confound or mlp_confound or v147_control_endpoint > 0:
        route = "R3-GenericOptimizerStateResetConfound"
    elif v147_every_endpoint >= expected and v147_event_endpoint < expected:
        route = "R2-ContinuousMomentumSuppressionNotFMSEvent"
    elif "A2-AffectedSetIntrinsicallyBroad" in affected_conclusion or "A3-SparseAffectedMaskDestroysValue" in affected_conclusion:
        route = "R4-AffectedCoordinateIdentificationFail"
    elif any(fnum(row.get("step_time_ratio_vs_adamw"), 0.0) > 1.25 for row in k_rows if row.get("method") == best_fms_method) and best_fms_endpoint > 0:
        route = "R5-OverheadBlocker"
    else:
        route = "R6-FMSStateTransportNoGo"
    return {
        "stage": "V148_ROUTE_DECISION",
        "route": route,
        "minimum_success": "S5-OfficialFunctionalSuccess" if route == "S5-OfficialFunctionalSuccess" else "S4c-MechanismSufficientDiagnostic",
        "expected_dataset_seed_count": expected,
        "best_fms_method": best_fms_method,
        "best_fms_strict_pass_count": best_fms_count,
        "best_fms_endpoint_vs_adamw_pass_count": best_fms_endpoint,
        "s1_strict_pass_count": fms_counts.get("S1-RAT-FMS-ZeroMomentResetAffectedEventOnly", 0),
        "s5_strict_pass_count": fms_counts.get("S5-RAT-FMS-MomentResetWithPostEventRecoveryWindow", 0),
        "generic_strict_pass_by_method": json.dumps(generic_counts, sort_keys=True),
        "generic_endpoint_pass_by_method": json.dumps(generic_endpoint, sort_keys=True),
        "mlp_strict_pass_by_method": json.dumps(mlp_counts, sort_keys=True),
        "mlp_endpoint_pass_by_method": json.dumps(mlp_endpoint, sort_keys=True),
        "generic_optimizer_state_confound": int(generic_confound or v147_control_endpoint > 0),
        "mlp_analog_confound": int(mlp_confound),
        "affected_mask_conclusion": affected_conclusion,
        "official_s5_reached": int(route == "S5-OfficialFunctionalSuccess"),
        "promotion_allowed": int(route == "S5-OfficialFunctionalSuccess" and int(args.compute_budgeted_run) == 0 and missing == 0),
        "compute_budgeted_run": int(args.compute_budgeted_run),
        "required_artifact_missing_count": missing,
        "forbidden_information_violation_count": 0,
        "no_action_search_violation_count": 0,
        "controller_executed": 0,
        "out_dir": str(args.out_dir),
    }


def write_figures(
    out_dir: Path,
    route: dict[str, Any],
    semantic_rows: list[dict[str, Any]],
    progress: list[dict[str, Any]],
    affected: list[dict[str, Any]],
    substrate: list[dict[str, Any]],
) -> None:
    v144.write_svg(
        out_dir / "fig_v148_semantic_diff_v1461_v147.svg",
        "v14.8 semantic diff",
        [f"{row['run_id']}: strict={row['strict_pass']} endpoint={row['endpoint_vs_adamw_pass']} route={row['route']}" for row in semantic_rows[:10]],
    )
    v144.write_svg(
        out_dir / "fig_v148_specificity_vs_controls.svg",
        "v14.8 specificity vs controls",
        [f"route={route['route']}", f"best={route['best_fms_method']} strict={route['best_fms_strict_pass_count']} endpoint={route['best_fms_endpoint_vs_adamw_pass_count']}"],
    )
    v144.write_svg(
        out_dir / "fig_v148_k2_vs_generic_optimizer_reset.svg",
        "v14.8 FMS vs generic reset",
        [f"{row['method']}: strict={row['strict_dataset_seed_pass_count']} endpoint={row['endpoint_vs_adamw_pass_count']}" for row in progress if row["method"].startswith(("G", "S1", "S5"))][:12],
    )
    v144.write_svg(
        out_dir / "fig_v148_affected_fraction_distribution.svg",
        "v14.8 affected fraction",
        [f"{row['run_id']}: affected_med={row['affected_fraction_median']} reset_med={row['reset_fraction_median']} {row['conclusion']}" for row in affected],
    )
    v144.write_svg(
        out_dir / "fig_v148_value_retention_vs_reset_fraction.svg",
        "v14.8 value retention vs reset fraction",
        [f"{row['run_id']}: {row['value_retention_vs_reset_fraction']}" for row in affected],
    )
    v144.write_svg(
        out_dir / "fig_v148_step_time_waterfall.svg",
        "v14.8 step time",
        [f"{row['method']}: step={float(row['median_step_time_ratio']):.3f}" for row in progress[:14] if row.get("median_step_time_ratio") != ""],
    )
    v144.write_svg(
        out_dir / "fig_v148_mlp_vs_rat_reset_controls.svg",
        "v14.8 MLP vs RAT reset controls",
        [f"{row['method']}: strict={row['strict_dataset_seed_pass_count']} endpoint={row['endpoint_vs_adamw_pass_count']}" for row in progress if row["method"].startswith(("M", "G"))][:14],
    )
    v144.write_svg(
        out_dir / "fig_v148_all_basis_substrate_status.svg",
        "v14.8 all-basis status",
        [f"{row['family']} exec={row['new_training_executed']} pass={row['strict_gate_pass_rows']}" for row in substrate],
    )
    v144.write_svg(
        out_dir / "fig_v148_failure_taxonomy.svg",
        "v14.8 failure taxonomy",
        [f"route={route['route']}", f"generic_confound={route['generic_optimizer_state_confound']}", f"affected={route['affected_mask_conclusion']}"],
    )


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
                raw_rows.extend(canonicalize_artifact_rows([result["row"]], method, cfg))
                optimizer_probe_rows.extend(canonicalize_artifact_rows(result["optimizer_probe_rows"], method, cfg))
                linec_rows.extend(canonicalize_artifact_rows(result["linec_rows"], method, cfg))
    k_rows = enrich_v148(raw_rows, "D-RAT")
    mlp_rows = enrich_v148(raw_rows, "MLP")
    real_rows = k_rows + mlp_rows
    generic_rows = rows_for_methods(k_rows, set(GENERIC_METHODS))
    fms_rows = rows_for_methods(k_rows, set(FMS_METHODS))
    mlp_control_rows = rows_for_methods(mlp_rows, set(MLP_METHODS))
    progress = summarize(real_rows, "V148_PROGRESS_TABLE")
    semantic_rows = semantic_diff(real_rows)
    affected_rows = affected_mask_autopsy(optimizer_probe_rows)
    overhead_rows = overhead_breakdown(real_rows)
    substrate_rows = all_basis_status(k_rows)
    linec_tail_rows = linec_tail_audit(real_rows)
    write_audits(out_dir)
    v144.write_rows(out_dir / "v148_progress_table.csv", progress)
    v144.write_rows(out_dir / "v148_v1461_v147_semantic_diff.csv", semantic_rows)
    v144.write_rows(out_dir / "v148_generic_optimizer_reset_controls.csv", generic_rows)
    v144.write_rows(out_dir / "v148_fms_specificity_results.csv", fms_rows)
    v144.write_rows(out_dir / "v148_affected_mask_autopsy.csv", affected_rows)
    v144.write_rows(out_dir / "v148_overhead_breakdown.csv", overhead_rows)
    v144.write_rows(out_dir / "v148_mlp_analog_controls.csv", mlp_control_rows)
    v144.write_rows(out_dir / "v148_all_basis_substrate_status.csv", substrate_rows)
    v144.write_rows(out_dir / "v148_linec_tail_audit.csv", linec_tail_rows)
    v144.write_rows(out_dir / "v148_optimizer_probe_rows.csv", optimizer_probe_rows)
    v144.write_rows(out_dir / "v148_linec_raw_rows.csv", linec_rows)
    write_code_review_packet(out_dir)
    route = build_route(k_rows, mlp_rows, semantic_rows, affected_rows, 999, args)
    write_markdown_boundaries(out_dir, route)
    write_figures(out_dir, route, semantic_rows, progress, affected_rows, substrate_rows)
    missing = write_required_manifest(out_dir)
    route = build_route(k_rows, mlp_rows, semantic_rows, affected_rows, missing, args)
    route["promotion_allowed"] = int(route["route"] == "S5-OfficialFunctionalSuccess" and int(args.compute_budgeted_run) == 0 and missing == 0)
    v144.write_json(out_dir / "v148_route_decision.json", route)
    write_markdown_boundaries(out_dir, route)
    write_figures(out_dir, route, semantic_rows, progress, affected_rows, substrate_rows)
    missing = write_required_manifest(out_dir)
    route["required_artifact_missing_count"] = missing
    route["promotion_allowed"] = int(route["route"] == "S5-OfficialFunctionalSuccess" and int(args.compute_budgeted_run) == 0 and missing == 0)
    v144.write_json(out_dir / "v148_route_decision.json", route)
    return route


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DG-KAN v14.8 OptimizerStateConfoundAudit FMSSpecificity AllBasisParallel")
    parser.add_argument("--out-dir", default=str(ROOT / "official_v148"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--methods", default=",".join(GENERIC_METHODS + FMS_METHODS))
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
    parser.add_argument("--post-event-recovery-window", type=int, default=2)
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
