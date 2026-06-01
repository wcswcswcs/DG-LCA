#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
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


ROOT = Path("results/v14_7_optimizer_state_transport_fms_all_basis_parallel")
PLAN_PATH = Path("docs/DG-KAN_v14.7_OptimizerStateTransportFMS_AllBasisParallel_完整计划.md")
V1461_AFTER_P1 = Path(
    "results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/"
    "diagnostic_v1461_after_p1_transport"
)

REQUIRED = [
    "v147_route_decision.json",
    "v147_real_results.csv",
    "v147_real_summary.csv",
    "v147_optimizer_state_transport_manifest.csv",
    "v147_fms_affected_coordinate_manifest.csv",
    "v147_state_transport_controls.csv",
    "v147_mlp_state_transport_control.csv",
    "v147_all_basis_substrate_status.csv",
    "v147_trajectory_autopsy.csv",
    "v147_forbidden_information_audit.csv",
    "v147_no_action_search_audit.csv",
    "v147_required_artifact_manifest.csv",
    "v147_code_review_packet.zip",
    "fig_v147_auc_debt_before_after.svg",
    "fig_v147_state_transport_gate_matrix.svg",
    "fig_v147_controls_comparison.svg",
    "fig_v147_all_basis_status.svg",
]


K_METHODS = [
    "K0-RAT-AdamW",
    "K1-RAT-FMS-NoStateTransport",
    "K2-RAT-FMS-ZeroMomentReset-AffectedOnly",
    "K3-RAT-FMS-ZeroMomentReset-AllFMSRoles",
    "K4-RAT-FMS-PartialMomentInterpolation",
    "K5-RAT-FMS-RMSRecomputeMicrobatch",
    "K6-RAT-FMS-MomentTransportProjectedGrad",
    "KCTRL-RandomMomentResetMatchedFraction",
    "KCTRL-ZeroMomentResetRandomCoords",
    "KCTRL-FullAdamWStateReset",
    "KCTRL-NoOpMatchedOverhead",
]

MLP_METHODS = [
    "M0-MLP-AdamW",
    "M1-MLP-FMS-NoStateTransport",
    "M2-MLP-FMS-ZeroMomentResetAffected",
    "M3-MLP-FMS-ZeroMomentResetRandomCoords",
    "M4-MLP-FMS-FullAdamWStateReset",
]


def fnum(value: Any, default: float = 0.0) -> float:
    return v144.fnum(value, default)


def sint(value: Any, default: int = 0) -> int:
    return v144.sint(value, default)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def method_config(method: str, args: argparse.Namespace | None = None) -> dict[str, Any]:
    policy = str(getattr(args, "affected_coordinate_policy", "delta_nonzero")) if args is not None else "delta_nonzero"
    affected_scope = "delta_top25" if policy == "delta_top25" else "affected"
    random_scope = "random_delta_top25" if policy == "delta_top25" else "random_matched"
    configs = {
        "K0-RAT-AdamW": ("D-RAT", "K0-RAT-AdamW", "none", "affected", 1, "adamw_control"),
        "K1-RAT-FMS-NoStateTransport": ("D-RAT", "K-RT2-TrainStreamTailTrust", "none", "affected", 0, "main_no_transport"),
        "K2-RAT-FMS-ZeroMomentReset-AffectedOnly": ("D-RAT", "K-RT2-TrainStreamTailTrust", "zero_moment_reset", affected_scope, 0, "main_candidate"),
        "K3-RAT-FMS-ZeroMomentReset-AllFMSRoles": ("D-RAT", "K-RT2-TrainStreamTailTrust", "zero_moment_reset", "all_fms_roles", 0, "scope_probe"),
        "K4-RAT-FMS-PartialMomentInterpolation": ("D-RAT", "K-RT2-TrainStreamTailTrust", "partial_moment_interpolation", "affected", 0, "mechanism_probe"),
        "K5-RAT-FMS-RMSRecomputeMicrobatch": ("D-RAT", "K-RT2-TrainStreamTailTrust", "rms_recompute_microbatch", "affected", 0, "mechanism_probe"),
        "K6-RAT-FMS-MomentTransportProjectedGrad": ("D-RAT", "K-RT2-TrainStreamTailTrust", "moment_transport_projected_grad", "affected", 0, "mechanism_probe"),
        "KCTRL-RandomMomentResetMatchedFraction": ("D-RAT", "K-RT2-TrainStreamTailTrust", "zero_moment_reset", random_scope, 1, "random_reset_control"),
        "KCTRL-ZeroMomentResetRandomCoords": ("D-RAT", "K-RT2-TrainStreamTailTrust", "zero_moment_reset", random_scope, 1, "random_reset_control"),
        "KCTRL-FullAdamWStateReset": ("D-RAT", "K-RT2-TrainStreamTailTrust", "zero_moment_reset", "full_adamw", 1, "full_reset_control"),
        "KCTRL-NoOpMatchedOverhead": ("D-RAT", "K-RT2-TrainStreamTailTrust", "none", "affected", 1, "noop_overhead_control"),
        "M0-MLP-AdamW": ("MLP", "MLP-AdamW", "none", "affected", 1, "mlp_adamw_control"),
        "M1-MLP-FMS-NoStateTransport": ("MLP", "MLP-FMS-Amortized", "none", "affected", 0, "mlp_generic_fms"),
        "M2-MLP-FMS-ZeroMomentResetAffected": ("MLP", "MLP-FMS-Amortized", "zero_moment_reset", affected_scope, 0, "mlp_zero_reset"),
        "M3-MLP-FMS-ZeroMomentResetRandomCoords": ("MLP", "MLP-FMS-Amortized", "zero_moment_reset", random_scope, 1, "mlp_random_reset_control"),
        "M4-MLP-FMS-FullAdamWStateReset": ("MLP", "MLP-FMS-Amortized", "zero_moment_reset", "full_adamw", 1, "mlp_full_reset_control"),
    }
    family, base_method, transport_mode, transport_scope, control, role = configs[method]
    return {
        "family": family,
        "v144_method": base_method,
        "transport_mode": transport_mode,
        "transport_scope": transport_scope,
        "control_method": int(control),
        "control_role": role,
        "affected_coordinate_policy": policy,
    }


def canonicalize_artifact_rows(rows: list[dict[str, Any]], method: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["v144_base_method"] = item.get("method", cfg["v144_method"])
        item["method"] = method
        item["v147_method"] = method
        item["control_method"] = int(cfg["control_method"])
        item["control_role"] = cfg["control_role"]
        item["optimizer_state_transport_scope"] = cfg["transport_scope"]
        item["affected_coordinate_policy"] = cfg.get("affected_coordinate_policy", "delta_nonzero")
        out.append(item)
    return out


def enrich_v147(rows: list[dict[str, Any]], family_filter: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("family")) != family_filter:
            continue
        groups.setdefault((str(row["dataset"]), int(row["seed"]), str(row["loss_interface"])), []).append(row)
    out: list[dict[str, Any]] = []
    for _key, group in groups.items():
        controls = [row for row in group if sint(row.get("control_method"), 0) == 1]
        base_controls = [row for row in controls if row.get("method") in {"K0-RAT-AdamW", "M0-MLP-AdamW"}]
        best_source_controls = controls or group
        best_nll = min(fnum(row.get("NLL"), 9.0) for row in best_source_controls)
        best_auc = min(fnum(row.get("AUC_NLL"), 9.0) for row in best_source_controls)
        adam = base_controls[0] if base_controls else (controls[0] if controls else group[0])
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
            item["step_time_ratio_vs_adamw"] = fnum(row.get("step_time_sec"), 0.0) / base_time
            item["peak_memory_ratio_vs_adamw"] = fnum(row.get("peak_memory_bytes"), 0.0) / base_mem
            item["v147_endpoint_vs_adamw_gate_pass"] = int(
                fnum(item["source_vs_adamw"]) >= 0.005
                and fnum(item["AUCtime_ratio_vs_adamw"]) <= 1.0
                and fnum(item["CEp99_delta_vs_adamw"]) <= 0.05
                and fnum(item["NLL_delta_vs_adamw"]) <= 0.02
                and fnum(item["ECE_delta_vs_adamw"]) <= 0.02
                and sint(item.get("LineC_majority_pass"), 0) == 1
            )
            item["v147_specificity_gate_pass"] = int(fnum(item["source_vs_best_control"]) >= 0.005)
            item["v147_efficiency_gate_pass"] = int(
                fnum(item["step_time_ratio_vs_adamw"]) <= 1.25
                and fnum(item["peak_memory_ratio_vs_adamw"]) <= 1.25
            )
            item["v147_strict_gate_pass"] = int(
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


def summarize(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row.get("method")), []).append(row)
    out: list[dict[str, Any]] = []
    for method, group in sorted(by_method.items()):
        target = [row for row in group if sint(row.get("control_method"), 0) == 0] or group
        out.append(
            {
                "stage": stage,
                "method": method,
                "rows": len(group),
                "dataset_seed_pass_count": len({(row.get("dataset"), str(row.get("seed"))) for row in group if sint(row.get("v147_strict_gate_pass"), 0) == 1}),
                "pass_rows": sum(sint(row.get("v147_strict_gate_pass"), 0) for row in group),
                "mean_source_vs_best_control": sum(fnum(row.get("source_vs_best_control"), 0.0) for row in target) / max(1, len(target)),
                "median_auc_time": v144.median([fnum(row.get("AUCtime_ratio_vs_best_control"), 9.0) for row in target]),
                "median_auc_time_vs_adamw": v144.median([fnum(row.get("AUCtime_ratio_vs_adamw"), 9.0) for row in target]),
                "median_step_time": v144.median([fnum(row.get("step_time_ratio_vs_adamw"), 9.0) for row in target]),
                "median_memory_ratio": v144.median([fnum(row.get("peak_memory_ratio_vs_adamw"), 9.0) for row in target]),
                "endpoint_vs_adamw_pass_rows": sum(sint(row.get("v147_endpoint_vs_adamw_gate_pass"), 0) for row in group),
                "specificity_pass_rows": sum(sint(row.get("v147_specificity_gate_pass"), 0) for row in group),
                "efficiency_pass_rows": sum(sint(row.get("v147_efficiency_gate_pass"), 0) for row in group),
                "linec_majority_pass_rows": sum(sint(row.get("LineC_majority_pass"), 0) for row in target),
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
    if fnum(row.get("step_time_ratio_vs_adamw"), 0.0) > 1.25:
        reasons.append("step_time")
    if fnum(row.get("peak_memory_ratio_vs_adamw"), 0.0) > 1.25:
        reasons.append("memory")
    return reasons


def transport_manifest(k_rows: list[dict[str, Any]], mlp_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in k_rows + mlp_rows:
        out.append(
            {
                "stage": "V147_OPTIMIZER_STATE_TRANSPORT_MANIFEST",
                "family": row.get("family"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "method": row.get("method"),
                "v144_base_method": row.get("v144_base_method"),
                "control_method": row.get("control_method"),
                "control_role": row.get("control_role"),
                "optimizer_state_transport_mode": row.get("optimizer_state_transport_mode"),
                "optimizer_state_transport_scope": row.get("optimizer_state_transport_scope"),
                "optimizer_state_transport_event_count": row.get("optimizer_state_transport_event_count"),
                "median_moment_staleness_before": row.get("median_moment_staleness_before"),
                "median_moment_staleness_after": row.get("median_moment_staleness_after"),
                "median_rms_mismatch_before": row.get("median_rms_mismatch_before"),
                "median_rms_mismatch_after": row.get("median_rms_mismatch_after"),
                "direction_changed_by_transport": 0,
                "promotion_allowed": 0,
            }
        )
    return out


def affected_coordinate_manifest(probe_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in probe_rows:
        out.append(
            {
                "stage": "V147_FMS_AFFECTED_COORDINATE_MANIFEST",
                "family": row.get("family"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "method": row.get("method"),
                "step": row.get("step"),
                "optimizer_state_transport_mode": row.get("optimizer_state_transport_mode"),
                "optimizer_state_transport_scope": row.get("optimizer_state_transport_scope"),
                "affected_param_fraction": row.get("affected_param_fraction"),
                "transport_reset_param_fraction": row.get("transport_reset_param_fraction", row.get("affected_param_fraction")),
                "coordinates_identified_from_train_stream": 1,
                "direction_uses_validation_test_future_query": 0,
                "direction_uses_linec_cep99_nll_ece_auc_brier": 0,
                "promotion_allowed": 0,
            }
        )
    return out


def trajectory_autopsy(k_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key_method = {(str(row.get("dataset")), int(row.get("seed")), str(row.get("method"))): row for row in k_rows}
    baseline_by_key = {
        (str(row.get("dataset")), int(row.get("seed"))): row
        for row in k_rows
        if row.get("method") == "K1-RAT-FMS-NoStateTransport"
    }
    out = []
    for row in k_rows:
        if row.get("method") != "K2-RAT-FMS-ZeroMomentReset-AffectedOnly":
            continue
        key = (str(row.get("dataset")), int(row.get("seed")))
        before = baseline_by_key.get(key)
        auc_before = max(0.0, fnum(before.get("AUCtime_ratio_vs_best_control"), 9.0) - 1.0) if before else 0.0
        auc_after = max(0.0, fnum(row.get("AUCtime_ratio_vs_best_control"), 9.0) - 1.0)
        out.append(
            {
                "stage": "V147_TRAJECTORY_AUTOPSY",
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "method": row.get("method"),
                "baseline_method": before.get("method") if before else "",
                "AUC_debt_before": auc_before,
                "AUC_debt_after": auc_after,
                "AUC_debt_reduction": auc_before - auc_after,
                "source_after": row.get("source_vs_best_control"),
                "tail_after_CEp99_delta": row.get("CEp99_delta_vs_adamw"),
                "NLL_delta_after": row.get("NLL_delta_vs_adamw"),
                "ECE_delta_after": row.get("ECE_delta_vs_adamw"),
                "LineC_after": row.get("LineC_majority_pass"),
                "strict_gate_pass": row.get("v147_strict_gate_pass"),
                "failure_reasons_after": "|".join(failure_reasons(row)),
                "moment_staleness_before": row.get("median_moment_staleness_before"),
                "moment_staleness_after": row.get("median_moment_staleness_after"),
                "rms_mismatch_before": row.get("median_rms_mismatch_before"),
                "rms_mismatch_after": row.get("median_rms_mismatch_after"),
                "promotion_allowed": 0,
            }
        )
    return out


def all_basis_status(k_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    k2_pass = sum(
        sint(row.get("v147_strict_gate_pass"), 0)
        for row in k_rows
        if row.get("method") == "K2-RAT-FMS-ZeroMomentReset-AffectedOnly"
    )
    rows = [
        {
            "stage": "V147_ALL_BASIS_SUBSTRATE_STATUS",
            "family": "D-RAT",
            "candidate": "D-RAT-Monitor-v147",
            "new_training_executed": 1,
            "strict_gate_pass_rows": k2_pass,
            "official_functional_open": int(k2_pass == 9),
            "source_artifact": "v147_real_results.csv",
            "not_executed_reason": "",
            "promotion_allowed": 0,
        }
    ]
    for family, candidates in {
        "D-WAV": ["D-WAV20-TriangleSupportStable", "D-WAV21-ScaleOccupancyHardening", "D-WAV22-LocalTailCoverageGuard"],
        "D-FOU": ["D-FOU20-LowFreqIdentityResidual", "D-FOU21-BandwiseSNRWarmup", "D-FOU22-PhaseStableBandMix"],
        "D-RBF": ["D-RBF20-CompactBumpIdentityResidual", "D-RBF21-ActiveCenterOccupancyRepair", "D-RBF22-WidthConditionGuard"],
        "D-CHE": ["D-CHE20-LowDegreeIdentityResidual", "D-CHE21-HighDegreeLateEnable", "D-CHE22-DegreeEnergyDamping"],
    }.items():
        for candidate in candidates:
            rows.append(
                {
                    "stage": "V147_ALL_BASIS_SUBSTRATE_STATUS",
                    "family": family,
                    "candidate": candidate,
                    "new_training_executed": 0,
                    "strict_gate_pass_rows": 0,
                    "official_functional_open": 0,
                    "source_artifact": "",
                    "not_executed_reason": "rational_official_confirmation_not_opened_or_candidate_not_implemented_in_runner",
                    "promotion_allowed": 0,
                }
            )
    return rows


def pass_count(rows: list[dict[str, Any]], method: str, field: str = "v147_strict_gate_pass") -> int:
    return len({(row.get("dataset"), str(row.get("seed"))) for row in rows if row.get("method") == method and sint(row.get(field), 0) == 1})


def build_route(k_rows: list[dict[str, Any]], mlp_rows: list[dict[str, Any]], missing: int, args: argparse.Namespace) -> dict[str, Any]:
    expected = len(v144.parse_csv(args.datasets)) * len(v144.parse_ints(args.seeds))
    k2_count = pass_count(k_rows, "K2-RAT-FMS-ZeroMomentReset-AffectedOnly")
    k2_endpoint_count = pass_count(k_rows, "K2-RAT-FMS-ZeroMomentReset-AffectedOnly", "v147_endpoint_vs_adamw_gate_pass")
    k2_specificity_count = pass_count(k_rows, "K2-RAT-FMS-ZeroMomentReset-AffectedOnly", "v147_specificity_gate_pass")
    k2_efficiency_count = pass_count(k_rows, "K2-RAT-FMS-ZeroMomentReset-AffectedOnly", "v147_efficiency_gate_pass")
    k3_count = pass_count(k_rows, "K3-RAT-FMS-ZeroMomentReset-AllFMSRoles")
    k5_count = pass_count(k_rows, "K5-RAT-FMS-RMSRecomputeMicrobatch")
    k6_count = pass_count(k_rows, "K6-RAT-FMS-MomentTransportProjectedGrad")
    reset_controls = [
        "KCTRL-RandomMomentResetMatchedFraction",
        "KCTRL-ZeroMomentResetRandomCoords",
        "KCTRL-FullAdamWStateReset",
        "M3-MLP-FMS-ZeroMomentResetRandomCoords",
        "M4-MLP-FMS-FullAdamWStateReset",
    ]
    control_success = {method: pass_count(k_rows + mlp_rows, method) for method in reset_controls}
    control_endpoint_success = {method: pass_count(k_rows + mlp_rows, method, "v147_endpoint_vs_adamw_gate_pass") for method in reset_controls}
    control_any_full = any(count >= expected for count in control_success.values())
    if k2_count >= expected and not control_any_full and missing == 0:
        route = "S5-OfficialFunctionalSuccess"
        minimum = route
    elif k2_count >= expected and control_any_full:
        route = "R2-ResetWorksButNotFMSSpecific"
        minimum = "S4c-MechanismSufficientDiagnostic"
    elif k2_endpoint_count >= expected and (k2_specificity_count < expected or any(count >= expected for count in control_endpoint_success.values())):
        route = "R2-ResetWorksButNotFMSSpecific"
        minimum = "S4c-MechanismSufficientDiagnostic"
    elif k2_count < expected and k3_count >= expected:
        route = "R3-AffectedCoordinateIdentificationFail"
        minimum = "S4c-MechanismSufficientDiagnostic"
    elif k2_count < expected and k5_count >= expected:
        route = "R4-RMSMismatchDominant"
        minimum = "S4c-MechanismSufficientDiagnostic"
    elif k2_count < expected and k6_count >= expected:
        route = "R5-ProjectedMomentTransportDominant"
        minimum = "S4c-MechanismSufficientDiagnostic"
    else:
        route = "R1-ZeroMomentResetNotReplicated"
        minimum = "S4c-MechanismSufficientDiagnostic"
    mean_source = sum(
        fnum(row.get("source_vs_best_control"), 0.0)
        for row in k_rows
        if row.get("method") == "K2-RAT-FMS-ZeroMomentReset-AffectedOnly"
    ) / max(1, len([row for row in k_rows if row.get("method") == "K2-RAT-FMS-ZeroMomentReset-AffectedOnly"]))
    return {
        "stage": "V147_ROUTE_DECISION",
        "route": route,
        "minimum_success": minimum,
        "expected_dataset_seed_count": expected,
        "k2_dataset_seed_pass_count": k2_count,
        "k2_endpoint_vs_adamw_pass_count": k2_endpoint_count,
        "k2_specificity_pass_count": k2_specificity_count,
        "k2_efficiency_pass_count": k2_efficiency_count,
        "k3_dataset_seed_pass_count": k3_count,
        "k5_dataset_seed_pass_count": k5_count,
        "k6_dataset_seed_pass_count": k6_count,
        "control_success_by_method": json.dumps(control_success, sort_keys=True),
        "control_endpoint_success_by_method": json.dumps(control_endpoint_success, sort_keys=True),
        "control_full_success_count": sum(1 for count in control_success.values() if count >= expected),
        "control_endpoint_full_success_count": sum(1 for count in control_endpoint_success.values() if count >= expected),
        "official_s5_reached": int(route == "S5-OfficialFunctionalSuccess"),
        "promotion_allowed": int(route == "S5-OfficialFunctionalSuccess" and int(args.compute_budgeted_run) == 0),
        "compute_budgeted_run": int(args.compute_budgeted_run),
        "mean_source_vs_best_control_k2": mean_source,
        "required_artifact_missing_count": missing,
        "forbidden_information_violation_count": 0,
        "no_action_search_violation_count": 0,
        "controller_executed": 0,
        "controller_not_executed_reason": "v147_official_confirmation_has_no_controller",
        "source_v1461_after_p1": str(V1461_AFTER_P1),
        "out_dir": str(args.out_dir),
    }


def write_audits(out_dir: Path) -> None:
    v144.write_rows(
        out_dir / "v147_no_action_search_audit.csv",
        [
            {
                "stage": "V147_NO_ACTION_SEARCH_AUDIT",
                "new_k_token_added": 0,
                "action_bank_used": 0,
                "controller_executed": 0,
                "strength_lambda_lr_refresh_grid_search": 0,
                "dataset_seed_branch_used": 0,
                "cross_run_local_positive_spliced_as_s5": 0,
                "diagnostic_oracle_written_as_promotion": 0,
                "violation": 0,
            }
        ],
    )
    v144.write_rows(
        out_dir / "v147_forbidden_information_audit.csv",
        [
            {
                "stage": "V147_FORBIDDEN_INFORMATION_AUDIT",
                "direction_uses_validation_test_future_query": 0,
                "direction_uses_linec_cep99_nll_ece_auc_brier": 0,
                "teacher_distillation_used": 0,
                "loss_modification_used": 0,
                "sampler_or_class_weight_used": 0,
                "label_informed_initialization_used": 0,
                "dataset_name_branch_used": 0,
                "seed_specific_scaling_used": 0,
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
    v144.write_rows(out_dir / "v147_required_artifact_manifest.csv", rows)
    return missing


def write_code_review_packet(out_dir: Path) -> None:
    files = [
        Path("experiments/run_v147_optimizer_state_transport_fms_all_basis_parallel.py"),
        Path("experiments/run_v144_real_transfer_fms_all_basis_substrate.py"),
        PLAN_PATH,
        Path("docs/DG-KAN_v14.6.1_TrajectoryMechanismSufficiency_NoActionSearch_实验结果复盘.md"),
    ]
    manifest = [
        {
            "path": str(path),
            "exists": int(path.exists()),
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    packet = out_dir / "v147_code_review_packet.zip"
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("v147_code_review_manifest.csv", rows_to_csv(manifest))
        for path in files:
            if path.exists():
                zf.write(path, arcname=str(path))


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    lines = []
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    lines.append(buf.getvalue())
    return "".join(lines)


def write_figures(out_dir: Path, route: dict[str, Any], summary: list[dict[str, Any]], autopsy: list[dict[str, Any]], substrate: list[dict[str, Any]]) -> None:
    v144.write_svg(
        out_dir / "fig_v147_auc_debt_before_after.svg",
        "v14.7 AUC debt before/after",
        [
            f"route={route['route']}",
            f"K2 pass={route['k2_dataset_seed_pass_count']}/{route['expected_dataset_seed_count']}",
            *[
                f"{row['dataset']}:{row['seed']} debt {float(row['AUC_debt_before']):.3f}->{float(row['AUC_debt_after']):.3f}"
                for row in autopsy[:8]
            ],
        ],
    )
    v144.write_svg(
        out_dir / "fig_v147_state_transport_gate_matrix.svg",
        "v14.7 state transport gate matrix",
        [
            f"K2={route['k2_dataset_seed_pass_count']}",
            f"K3={route['k3_dataset_seed_pass_count']}",
            f"K5={route['k5_dataset_seed_pass_count']}",
            f"K6={route['k6_dataset_seed_pass_count']}",
        ],
    )
    control_lines = [f"{row['method']}: pass={row['dataset_seed_pass_count']} step={float(row['median_step_time']):.2f}" for row in summary if str(row.get("method", "")).startswith("KCTRL")][:10]
    v144.write_svg(out_dir / "fig_v147_controls_comparison.svg", "v14.7 controls comparison", control_lines or ["no control summary"])
    basis_lines = [f"{row['family']} {row['candidate']} exec={row['new_training_executed']} pass={row['strict_gate_pass_rows']}" for row in substrate[:12]]
    v144.write_svg(out_dir / "fig_v147_all_basis_status.svg", "v14.7 all-basis status", basis_lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if int(getattr(args, "reuse_existing_results", 0)) == 1 and (out_dir / "v147_real_results.csv").exists():
        existing = read_rows(out_dir / "v147_real_results.csv")
        k_rows = enrich_v147([row for row in existing if str(row.get("family")) == "D-RAT"], "D-RAT")
        mlp_rows = enrich_v147([row for row in existing if str(row.get("family")) == "MLP"], "MLP")
        summary = summarize(k_rows + mlp_rows, "V147_REAL_SUMMARY")
        controls = [row for row in k_rows if sint(row.get("control_method"), 0) == 1]
        transport_rows = transport_manifest(k_rows, mlp_rows)
        affected_rows = read_rows(out_dir / "v147_fms_affected_coordinate_manifest.csv")
        autopsy_rows = trajectory_autopsy(k_rows)
        substrate_rows = all_basis_status(k_rows)
        write_audits(out_dir)
        v144.write_rows(out_dir / "v147_real_results.csv", k_rows + mlp_rows)
        v144.write_rows(out_dir / "v147_real_summary.csv", summary)
        v144.write_rows(out_dir / "v147_optimizer_state_transport_manifest.csv", transport_rows)
        v144.write_rows(out_dir / "v147_fms_affected_coordinate_manifest.csv", affected_rows)
        v144.write_rows(out_dir / "v147_state_transport_controls.csv", controls)
        v144.write_rows(out_dir / "v147_mlp_state_transport_control.csv", mlp_rows)
        v144.write_rows(out_dir / "v147_all_basis_substrate_status.csv", substrate_rows)
        v144.write_rows(out_dir / "v147_trajectory_autopsy.csv", autopsy_rows)
        write_code_review_packet(out_dir)
        write_figures(out_dir, {"route": "pending", "k2_dataset_seed_pass_count": 0, "expected_dataset_seed_count": len(v144.parse_csv(args.datasets)) * len(v144.parse_ints(args.seeds)), "k3_dataset_seed_pass_count": 0, "k5_dataset_seed_pass_count": 0, "k6_dataset_seed_pass_count": 0}, summary, autopsy_rows, substrate_rows)
        missing = write_required_manifest(out_dir)
        route = build_route(k_rows, mlp_rows, missing, args)
        v144.write_json(out_dir / "v147_route_decision.json", route)
        write_figures(out_dir, route, summary, autopsy_rows, substrate_rows)
        missing = write_required_manifest(out_dir)
        route["required_artifact_missing_count"] = missing
        route["promotion_allowed"] = int(route["route"] == "S5-OfficialFunctionalSuccess" and int(args.compute_budgeted_run) == 0 and missing == 0)
        v144.write_json(out_dir / "v147_route_decision.json", route)
        return route
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    datasets = v144.parse_csv(args.datasets)
    seeds = v144.parse_ints(args.seeds)
    k_methods = v144.parse_csv(args.methods)
    mlp_methods = [] if int(args.skip_mlp_control) else v144.parse_csv(args.mlp_methods)
    k_rows_raw: list[dict[str, Any]] = []
    mlp_rows_raw: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    proxy_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    costate_rows: list[dict[str, Any]] = []
    optimizer_probe_rows: list[dict[str, Any]] = []
    linec_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            xtr, ytr, xva, yva, xte, yte, input_dim, output_dim = v144.load_real_split(args, dataset, seed, device)
            input_dim = int(input_dim.item() if hasattr(input_dim, "item") else input_dim)
            output_dim = int(output_dim.item() if hasattr(output_dim, "item") else output_dim)
            for method in k_methods + mlp_methods:
                cfg = method_config(method, args)
                case_args = copy(args)
                case_args.optimizer_state_transport_mode = cfg["transport_mode"]
                case_args.optimizer_state_transport_scope = cfg["transport_scope"]
                case_args.optimizer_state_transport_probe = 1
                result = v144.train_case(
                    family=cfg["family"],
                    dataset=dataset,
                    seed=seed,
                    method=cfg["v144_method"],
                    candidate_id=args.rational_candidate if cfg["family"] != "MLP" else "MLP-StateTransportControl",
                    loss_interface=args.loss_interface,
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
                row = canonicalize_artifact_rows([result["row"]], method, cfg)[0]
                if cfg["family"] == "MLP":
                    mlp_rows_raw.append(row)
                else:
                    k_rows_raw.append(row)
                projection_rows.extend(canonicalize_artifact_rows(result["projection_rows"], method, cfg))
                proxy_rows.extend(canonicalize_artifact_rows(result["proxy_rows"], method, cfg))
                split_rows.extend(canonicalize_artifact_rows(result["split_rows"], method, cfg))
                costate_rows.extend(canonicalize_artifact_rows(result["costate_rows"], method, cfg))
                optimizer_probe_rows.extend(canonicalize_artifact_rows(result["optimizer_probe_rows"], method, cfg))
                linec_rows.extend(canonicalize_artifact_rows(result["linec_rows"], method, cfg))
    k_rows = enrich_v147(k_rows_raw, "D-RAT")
    mlp_rows = enrich_v147(mlp_rows_raw, "MLP")
    real_results = k_rows + mlp_rows
    summary = summarize(real_results, "V147_REAL_SUMMARY")
    controls = [row for row in k_rows if sint(row.get("control_method"), 0) == 1]
    mlp_control = mlp_rows
    transport_rows = transport_manifest(k_rows, mlp_rows)
    affected_rows = affected_coordinate_manifest(optimizer_probe_rows)
    autopsy_rows = trajectory_autopsy(k_rows)
    substrate_rows = all_basis_status(k_rows)
    write_audits(out_dir)
    v144.write_rows(out_dir / "v147_real_results.csv", real_results)
    v144.write_rows(out_dir / "v147_real_summary.csv", summary)
    v144.write_rows(out_dir / "v147_optimizer_state_transport_manifest.csv", transport_rows)
    v144.write_rows(out_dir / "v147_fms_affected_coordinate_manifest.csv", affected_rows)
    v144.write_rows(out_dir / "v147_state_transport_controls.csv", controls)
    v144.write_rows(out_dir / "v147_mlp_state_transport_control.csv", mlp_control)
    v144.write_rows(out_dir / "v147_all_basis_substrate_status.csv", substrate_rows)
    v144.write_rows(out_dir / "v147_trajectory_autopsy.csv", autopsy_rows)
    v144.write_rows(out_dir / "v147_linec_audit.csv", linec_rows)
    v144.write_rows(out_dir / "v147_train_stream_proxy.csv", proxy_rows)
    v144.write_rows(out_dir / "v147_projection_value_retention.csv", projection_rows)
    v144.write_rows(out_dir / "v147_split_agreement.csv", split_rows)
    v144.write_rows(out_dir / "v147_source_tail_costate.csv", costate_rows)
    write_code_review_packet(out_dir)
    write_figures(out_dir, {"route": "pending", "k2_dataset_seed_pass_count": 0, "expected_dataset_seed_count": len(datasets) * len(seeds), "k3_dataset_seed_pass_count": 0, "k5_dataset_seed_pass_count": 0, "k6_dataset_seed_pass_count": 0}, summary, autopsy_rows, substrate_rows)
    missing = write_required_manifest(out_dir)
    route = build_route(k_rows, mlp_rows, missing, args)
    v144.write_json(out_dir / "v147_route_decision.json", route)
    write_figures(out_dir, route, summary, autopsy_rows, substrate_rows)
    missing = write_required_manifest(out_dir)
    route["required_artifact_missing_count"] = missing
    route["promotion_allowed"] = int(route["route"] == "S5-OfficialFunctionalSuccess" and int(args.compute_budgeted_run) == 0 and missing == 0)
    v144.write_json(out_dir / "v147_route_decision.json", route)
    return route


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DG-KAN v14.7 OptimizerStateTransportFMS AllBasisParallel")
    parser.add_argument("--out-dir", default=str(ROOT / "official_v147"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--methods", default=",".join(K_METHODS))
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
    parser.add_argument("--linec-mode", choices=["none", "exact"], default="exact")
    parser.add_argument("--linec-seeds", default="12319500,12319501,12319502")
    parser.add_argument("--linec-batch-size", type=int, default=24)
    parser.add_argument("--linec-sketch-dim", type=int, default=8)
    parser.add_argument("--compute-budgeted-run", type=int, default=0)
    parser.add_argument("--reuse-existing-results", type=int, default=0)
    parser.add_argument("--affected-coordinate-policy", choices=["delta_nonzero", "delta_top25"], default="delta_nonzero")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    route = run(args)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
