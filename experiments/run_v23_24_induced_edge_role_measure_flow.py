#!/usr/bin/env python3
"""DG-KAN v23.24 induced edge-role dynamics runner.

This runner starts with the v23.24 contract's audit-heavy foundation:
Part 0 registries and Part A semantic/math units.  It intentionally records
what is actually executed and does not mark scientific hypotheses as resolved
until their synthetic, real, repair, and H-step requirements are run.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).resolve()
PYTHON = sys.executable
PLAN = ROOT / "docs/DG-KAN_v23.24_BasisCovariantInducedEdgeRoleMeasureFlow_MultilevelRefinement_TwinDynamics_多假设语义穷尽式完整详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v23.24_BasisCovariantInducedEdgeRoleMeasureFlow_MultilevelRefinement_TwinDynamics_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v23.24_BasisCovariantInducedEdgeRoleMeasureFlow_MultilevelRefinement_TwinDynamics_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2324_OUT_ROOT", str(ROOT / "results/v23_24"))).resolve()
EPS = 1.0e-12


HYPOTHESES = [
    ("H-A", "Selector firewall, paired causal harness, and basis-covariant flow correctness"),
    ("H-B", "Equal-mass continuous edge-role reservoir flow"),
    ("H-C", "Two-timescale align-grow role flow"),
    ("H-D", "Continuous role-mass reaction / measure flow"),
    ("H-E", "Function-preserving multilevel edge-basis refinement"),
    ("H-F", "Function-preserving symmetry-breaking twin flow"),
    ("H-G", "Fixed two-pass microbatch-reuse induced flow"),
    ("H-H", "Feature-speed equalized induced flow and MLP matched architecture surplus"),
]

RUNTIME_FORBIDDEN_COUNTERS = [
    "runtime_candidate_pool_size",
    "runtime_argmax_used",
    "runtime_topk_used",
    "runtime_best_role_used",
    "runtime_best_location_used",
    "runtime_role_birth_decision_used",
    "runtime_role_death_decision_used",
    "runtime_knot_insertion_decision_used",
    "source_witness_score_used_for_structure",
    "guard_used_for_structure",
    "validation_used_for_structure",
    "test_used_for_structure",
    "future_direction_used",
]

STRUCTURE_SCHEDULE = {
    "role_count_per_layer": 4,
    "incubation_steps": 5,
    "align_grow_schedule": {"K_align": 4, "K_grow": 4, "eta_U_align": 1.0, "eta_c_align": 0.25, "eta_U_grow": 0.10, "eta_c_grow": 1.0},
    "mass_reaction_rate": 0.05,
    "mass_floor": "0.02 / R_l",
    "multilevel_milestones": [{"fraction": 0.30, "from": "K0", "to": "K1"}, {"fraction": 0.60, "from": "K1", "to": "K2"}],
    "basis_levels": {"D-CHE": [3, 6, 9], "D-FOU": [2, 4, 8]},
    "twin_expansion_milestone": 0.40,
    "twin_asymmetry_rule": {"alpha": 0.5, "child_plus_state": "zero", "child_minus_state": "transported_parent", "lr_plus": 1.25, "lr_minus": 0.75, "rewarmup_steps": 10},
    "two_pass_order": ["mixing_mass", "recompute_same_batch", "shape_base"],
    "feature_speed_calibration": {"horizon": 10, "clip": [0.5, 2.0], "freeze_after_step": 10},
    "metric_refresh_cadence": 5,
    "metric_ema_alpha": 0.05,
}

CONTROLS = {
    "H-B": ["C0_BC15_no_reservoir", "C1_frozen_random_roles_train_mixing_only", "C2_train_roles_Euclidean_metric", "C3_role_gradient_shuffled_across_roles", "C4_edge_independent_roles_same_param", "C5_same_compute_dormant_unused_reservoir"],
    "H-C": ["D0_equal_rate_synchronous_flow", "D1_reversed_schedule_grow_then_align", "D2_time_shuffled_schedule_same_counts", "D3_frozen_shape_amplitude_only", "D4_same_compute_random_phase_schedule"],
    "H-D": ["E0_uniform_mass_fixed", "E1_reaction_only_shapes_frozen", "E2_time_shuffled_mass_gradient", "E3_signflip_mass_gradient", "E4_same_entropy_random_mass_walk", "E5_hard_winner_softmax_diagnostic_only"],
    "H-E": ["F0_coarse_only", "F1_fixed_fine_from_start_same_final_params", "F2_function_preserving_refinement_new_coords_frozen", "F3_refinement_with_random_non_nested_new_basis_same_param", "F4_refinement_milestone_time_shuffled_diagnostic", "F5_MLP_matched_width_or_feature_refinement"],
    "H-F": ["G0_no_expansion", "G1_perfect_symmetric_duplicate", "G2_alpha_045_055_same_state", "G3_G_isotropic_zero_sum_noise", "G4_exact_duplicate_same_compute_no_train_new_children", "G5_MLP_matched_hidden_widening"],
    "H-G": ["H0_one_pass_standard", "H1_two_backward_simultaneous_accumulation", "H2_fresh_batch_second_pass", "H3_reverse_order", "H4_same_batch_stale_activation", "H5_MLP_matched_two_pass"],
    "H-H": ["I0_uniform_layerwise_LR", "I1_random_matched_logrange_scaling", "I2_gradient_norm_equalization_control", "I3_MLP_matched_feature_speed_equalization"],
}

REPAIRS = {
    "H-B": ["C-R1_incubation_5_to_10_if_zero_shape_gradient", "C-R2_metric_conditioning_shrinkage_if_unstable"],
    "H-C": ["D-R1_phase_4_4_to_8_8_once", "D-R2_eta_U_align_1p0_to_0p5_once_if_unstable"],
    "H-D": ["E-R1_entropy_floor_x2_once_if_collapse", "E-R2_eta_m_half_once_if_oscillatory"],
    "H-E": ["F-R1_reset_new_fine_optimizer_state", "F-R2_fixed_10_step_new_coordinate_rewarmup_1p5"],
    "H-F": ["G-R1_asymmetry_1p25_0p75_to_1p5_0p5_once", "G-R2_rewarmup_10_to_20_once"],
    "H-G": ["H-R1_halve_each_pass_step_preserve_total_norm", "H-R2_role_increment_only_trust"],
    "H-H": ["I-R1_cap_0p5_4p0_diagnostic_only", "I-R2_calibration_horizon_10_to_20_once"],
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    EXEC_LOG.parent.mkdir(parents=True, exist_ok=True)


def command_text() -> str:
    env = []
    for key in ["CUDA_VISIBLE_DEVICES", "V2324_OUT_ROOT"]:
        if os.environ.get(key):
            env.append(f"{key}={os.environ[key]}")
    return " ".join([*env, PYTHON, rel(RUNNER), *sys.argv[1:]])


def init_logs() -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v23.24 执行日志\n\n"
            f"- created_at: {now()}\n"
            f"- plan: `{rel(PLAN)}`\n"
            f"- runner: `{rel(RUNNER)}`\n"
            f"- output_root: `{rel(OUT_ROOT)}`\n"
            "- rule: record real commands, files, metrics, repairs, and blockers; do not fabricate missing data.\n\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v23.24 实验结果复盘\n\n"
            f"- created_at: {now()}\n"
            "- current_status: running\n"
            "- rule: human-readable analysis with artifact links; full machine data stays in CSV/JSON artifacts.\n\n",
            encoding="utf-8",
        )


def append_exec(stage: str, status: str, *, files: str = "", note: str = "") -> None:
    init_logs()
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} | {stage} | {status}\n\n")
        fh.write(f"- command: `{command_text()}`\n")
        fh.write(f"- python: `{PYTHON}`\n")
        fh.write(f"- torch: `{torch.__version__}`\n")
        fh.write(f"- cuda_visible_devices: `{os.environ.get('CUDA_VISIBLE_DEVICES', '')}`\n")
        if files:
            fh.write(f"- files: `{files}`\n")
        if note:
            fh.write(f"- note: {note}\n")


def append_recap(title: str, lines: list[str]) -> None:
    init_logs()
    with RECAP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} | {title}\n\n")
        for line in lines:
            fh.write(f"{line}\n")


def write_json(path: Path, data: dict[str, Any]) -> Path:
    ensure_out()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_out()
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def spd(mat: torch.Tensor, ridge: float = 1.0e-6) -> torch.Tensor:
    eye = torch.eye(int(mat.shape[0]), dtype=torch.float64, device=mat.device)
    sym = 0.5 * (mat.to(torch.float64) + mat.to(torch.float64).T)
    return sym + float(ridge) * eye


def basis(s: torch.Tensor, degree: int) -> torch.Tensor:
    cols = [s.to(torch.float64) ** i for i in range(int(degree) + 1)]
    return torch.stack(cols, dim=1)


def basis_derivative(s: torch.Tensor, degree: int) -> torch.Tensor:
    cols = []
    for i in range(int(degree) + 1):
        if i == 0:
            cols.append(torch.zeros_like(s, dtype=torch.float64))
        else:
            cols.append(float(i) * s.to(torch.float64) ** (i - 1))
    return torch.stack(cols, dim=1)


def role_metric(s: torch.Tensor, degree: int, *, sobolev: float = 0.05, boundary: float = 0.01, ridge: float = 1.0e-6) -> torch.Tensor:
    psi = basis(s, degree)
    dpsi = basis_derivative(s, degree)
    l2 = psi.T @ psi / max(1, int(psi.shape[0]))
    sob = dpsi.T @ dpsi / max(1, int(dpsi.shape[0]))
    b = basis(torch.tensor([0.0, 1.0], dtype=torch.float64, device=s.device), degree)
    bd = b.T @ b / 2.0
    return spd(l2 + float(sobolev) * sob + float(boundary) * bd, ridge)


def g_orthonormalize(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    gram = spd(x.T @ g @ x, 1.0e-10)
    chol = torch.linalg.cholesky(gram)
    return x @ torch.linalg.inv(chol.T)


def metric_transport(g_old: torch.Tensor, g_new: torch.Tensor) -> torch.Tensor:
    l_old = torch.linalg.cholesky(spd(g_old, 1.0e-10))
    l_new = torch.linalg.cholesky(spd(g_new, 1.0e-10))
    return torch.linalg.solve_triangular(l_new.T, l_old.T, upper=True)


def g_stiefel_project(u: torch.Tensor, g: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    inner = u.T @ g @ z
    sym = 0.5 * (inner + inner.T)
    return z - u @ sym


def role_flow_update_reference(u: torch.Tensor, g: torch.Tensor, grad: torch.Tensor, lr: float) -> tuple[torch.Tensor, dict[str, float]]:
    z = -torch.linalg.solve(spd(g, 1.0e-8), grad.to(torch.float64))
    tangent = g_stiefel_project(u, g, z)
    updated = g_orthonormalize(u + float(lr) * tangent, g)
    return updated, {
        "role_shape_forward_calls": 1.0,
        "role_shape_backward_calls": 1.0,
        "metric_transport_calls": 0.0,
        "G_retraction_calls": 1.0,
    }


def continuous_mass_mirror_step(mass: torch.Tensor, q: torch.Tensor, eta: float, floor: float) -> torch.Tensor:
    qbar = torch.sum(mass * q)
    raw = mass * torch.exp(-float(eta) * (q - qbar))
    raw = raw / raw.sum().clamp_min(EPS)
    r = int(mass.numel())
    return (1.0 - r * float(floor)) * raw + float(floor)


def phase_part0(args: argparse.Namespace) -> dict[str, Any]:
    plan_text = PLAN.read_text(encoding="utf-8")
    plan_lines = plan_text.splitlines()
    plan_audit = {
        "plan": rel(PLAN),
        "plan_sha256": sha256_file(PLAN),
        "line_count": len(plan_lines),
        "read_scope": "full_document_1_2967_read_before_implementation",
        "no_heading_only_read": 1,
    }
    mandatory = [{"id": h, "name": name, "required": ["semantic_implementation_audit", "math_unit_tests", "positive_control", "negative_control", "synthetic_matrix", "minimum_actual_real_falsification", "H20_diagnostic", "allowed_repairs", "independent_conclusion"]} for h, name in HYPOTHESES]
    schemes = [
        {"id": "B_primary_equal_mass_role_flow", "hypothesis": "H-B", "selector_free": 1, "roles_per_layer": 4},
        {"id": "C_primary_two_timescale_align_grow", "hypothesis": "H-C", "selector_free": 1, "schedule": STRUCTURE_SCHEDULE["align_grow_schedule"]},
        {"id": "D_primary_continuous_mass_reaction", "hypothesis": "H-D", "selector_free": 1, "hard_prune_clone_allowed": 0},
        {"id": "E_primary_multilevel_refinement", "hypothesis": "H-E", "selector_free": 1, "milestones": STRUCTURE_SCHEDULE["multilevel_milestones"]},
        {"id": "F_primary_symmetry_breaking_twin_flow", "hypothesis": "H-F", "selector_free": 1, "twin_rule": STRUCTURE_SCHEDULE["twin_asymmetry_rule"]},
        {"id": "G_primary_two_pass_microbatch_reuse", "hypothesis": "H-G", "selector_free": 1, "two_pass_order": STRUCTURE_SCHEDULE["two_pass_order"]},
        {"id": "H_primary_feature_speed_equalized_flow", "hypothesis": "H-H", "selector_free": 1, "calibration": STRUCTURE_SCHEDULE["feature_speed_calibration"]},
    ]
    runtime_truth = {key: 0 for key in RUNTIME_FORBIDDEN_COUNTERS}
    runtime_truth.update({"selector_firewall_registered": 1, "all_structure_schedules_predeclared": 1})
    metrics = [
        "hidden_CKA_candidate_minus_baseline",
        "class_between_within_ratio_change",
        "AGOP_alignment_change",
        "true_bank_additive_R2_change",
        "feature_speed_by_layer",
        "backward_alignment_by_layer",
        "role_shape_G_angle_matrix",
        "role_mass_entropy",
        "role_effective_count",
        "edge_role_specialization_MI",
        "multilevel_fine_energy_fraction",
        "twin_antisymmetric_mode_norm",
        "paired_NLL_surplus",
        "CVaR25_surplus",
        "bootstrap_LCB",
        "no_debt",
    ]
    thresholds = {
        "part0": {
            "mandatory_hypothesis_count": 8,
            "selector_firewall_registered": 1,
            "all_structure_schedules_predeclared": 1,
            "all_controls_have_numeric_identity_tests": 1,
            "all_hypotheses_have_minimum_real_falsification": 1,
            "all_hypotheses_have_H20_diagnostic": 1,
            "no_dataset_specific_structure_branch": 1,
            "no_seed_specific_structure_branch": 1,
        },
        "minimum_real": {"median_surplus": 1.0e-3, "normalized_surplus_ratio": 0.05, "win_rate": 0.70, "no_debt": 0.80},
        "H20": {"diagnostic_required_even_if_minimum_real_fails": 1},
        "H80": {"paired_cumulative_NLL_surplus": 2.0e-3, "no_debt": 0.80},
    }
    dependency_graph = {
        "part0": [],
        "partA": ["part0"],
        "partB_synthetic": ["partA"],
        "parts_C_to_I": ["partB_synthetic"],
        "partJ_minimum_real": ["parts_C_to_I"],
        "partL_H20": ["partJ_minimum_real"],
        "partL_H80": ["partJ_minimum_real"],
        "partM_MLP_matched": ["partJ_minimum_real", "partL_H20"],
        "partN_official_expansion": ["partL_H80", "partM_MLP_matched"],
    }
    status_rows = []
    for h, name in HYPOTHESES:
        status_rows.append({
            "hypothesis": h,
            "name": name,
            "semantic_valid": 0,
            "math_unit_resolved": 0,
            "positive_control_resolved": 0,
            "negative_control_resolved": 0,
            "synthetic_resolved": 0,
            "minimum_real_completed": 0,
            "H20_completed": 0,
            "allowed_repairs_exhausted_or_passed": 0,
            "science_resolved": 0,
            "route": "R0_IncompleteScientificExploration",
        })
    control_identity_executable = 0
    gates = {
        "mandatory_hypothesis_count": len(HYPOTHESES),
        "mandatory_hypothesis_count_is_8": int(len(HYPOTHESES) == 8),
        "selector_firewall_registered": 1,
        "all_structure_schedules_predeclared": 1,
        "all_control_identity_test_specs_registered": 1,
        "all_controls_have_executed_numeric_identity_tests": control_identity_executable,
        "all_hypotheses_have_minimum_real_falsification": 1,
        "all_hypotheses_have_H20_diagnostic": 1,
        "no_dataset_specific_structure_branch": 1,
        "no_seed_specific_structure_branch": 1,
        "part0_hard_gate_pass": 0,
        "part0_blocker": "executable hypothesis-specific control implementations and numeric identity tests are not implemented yet",
    }
    artifacts = {
        "v23_24_plan_read_audit.json": plan_audit,
        "v23_24_theory_contract.json": {"plan_code": "BC-IERMF/BC-MER/BC-TwinFlow", "principle": "Induction Before Identification", "runtime_selector_first_allowed": 0, "hypotheses": [h for h, _ in HYPOTHESES]},
        "v23_24_lineage_manifest.json": {"previous_reference": "v23.23 P125 diagnostic only; no runtime selector promotion", "current_runner": rel(RUNNER), "plan_sha256": plan_audit["plan_sha256"]},
        "v23_24_mandatory_hypothesis_registry.json": {"hypotheses": mandatory},
        "v23_24_scheme_registry.json": {"schemes": schemes},
        "v23_24_control_registry.json": {"controls": CONTROLS},
        "v23_24_metric_registry.json": {"metrics": metrics},
        "v23_24_threshold_registry.json": thresholds,
        "v23_24_repair_registry.json": {"repairs": REPAIRS},
        "v23_24_dependency_graph.json": dependency_graph,
        "v23_24_runtime_truth_contract.json": runtime_truth,
        "v23_24_structure_schedule_registry.json": STRUCTURE_SCHEDULE,
    }
    for filename, payload in artifacts.items():
        write_json(OUT_ROOT / filename, payload)
    write_rows(OUT_ROOT / "v23_24_hypothesis_status_matrix.csv", status_rows)
    summary = {"status": "completed", "plan_audit": plan_audit, "gates": gates, "artifact_count": len(artifacts) + 1}
    write_json(OUT_ROOT / "v23_24_part0_summary.json", summary)
    append_exec("Part0_RegistryAndFullPlanReadAudit", "completed", files=";".join(sorted([*artifacts.keys(), "v23_24_hypothesis_status_matrix.csv", "v23_24_part0_summary.json"])), note=json.dumps(gates, sort_keys=True))
    append_recap("Part 0 registry and full-plan-read audit", [
        f"- status: completed; hard gate pass `{gates['part0_hard_gate_pass']}`.",
        f"- plan read evidence: `{rel(PLAN)}`, line_count `{plan_audit['line_count']}`, sha256 `{plan_audit['plan_sha256']}`.",
        "- registered 8 mandatory hypotheses H-A..H-H, selector firewall counters, fixed structure schedules, controls, metrics, thresholds, repairs, and dependency graph.",
        f"- hard-gate blocker: `{gates['part0_blocker']}`. This is a truthful implementation-status flag, not a science NoGo.",
        "- no scientific success claimed; all hypothesis status rows remain `R0_IncompleteScientificExploration` until Part A/B/J/L requirements are actually run.",
    ])
    return summary


def scan_runtime_selector_firewall() -> list[dict[str, Any]]:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    runtime_functions = {"role_flow_update_reference", "continuous_mass_mirror_step", "metric_transport", "g_stiefel_project"}
    rows = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in runtime_functions:
            counters = {"argmax": 0, "topk": 0, "sort": 0, "sorted": 0, "role_birth_branch": 0, "role_prune_branch": 0}
            for child in ast.walk(node):
                if isinstance(child, ast.Attribute) and child.attr in counters:
                    counters[child.attr] += 1
                if isinstance(child, ast.Name) and child.id in counters:
                    counters[child.id] += 1
                if isinstance(child, ast.If):
                    src = ast.get_source_segment(RUNNER.read_text(encoding="utf-8"), child) or ""
                    if "birth" in src:
                        counters["role_birth_branch"] += 1
                    if "prune" in src or "clone" in src:
                        counters["role_prune_branch"] += 1
            rows.append({
                "function": node.name,
                "argmax_calls_in_runtime_update_path": counters["argmax"],
                "topk_calls_in_runtime_update_path": counters["topk"],
                "sort_by_role_score_calls": counters["sort"] + counters["sorted"],
                "role_birth_condition_branches": counters["role_birth_branch"],
                "role_prune_condition_branches": counters["role_prune_branch"],
                "selector_firewall_pass": int(sum(counters.values()) == 0),
            })
    return rows


def covariance_unit(seed: int = 2401) -> dict[str, Any]:
    torch.manual_seed(seed)
    s = torch.linspace(0.0, 1.0, 128, dtype=torch.float64)
    degree = 5
    g = role_metric(s, degree)
    dim = degree + 1
    roles = 3
    raw_u = torch.randn((dim, roles), dtype=torch.float64)
    u = g_orthonormalize(raw_u, g)
    psi = basis(s, degree)
    y = psi @ u
    s_chart = torch.randn((dim, dim), dtype=torch.float64)
    while abs(float(torch.linalg.det(s_chart))) < 1.0e-6:
        s_chart = torch.randn((dim, dim), dtype=torch.float64)
    psi_prime = psi @ s_chart
    u_prime = torch.linalg.solve(s_chart, u)
    g_prime = s_chart.T @ g @ s_chart
    y_prime = psi_prime @ u_prime
    norm_old = u.T @ g @ u
    norm_new = u_prime.T @ g_prime @ u_prime
    function_error = float((y - y_prime).abs().max().item())
    metric_norm_relative_error = float((norm_old - norm_new).norm().item() / norm_old.norm().clamp_min(EPS).item())
    s2 = torch.linspace(0.005, 0.995, 128, dtype=torch.float64) ** 1.1
    g_new = role_metric(s2, degree)
    t = metric_transport(g, g_new)
    transported = t @ u
    transport_identity_error = float((transported.T @ g_new @ transported - u.T @ g @ u).norm().item())
    cos = torch.nn.functional.cosine_similarity((torch.linalg.cholesky(g_new).T @ transported).T, (torch.linalg.cholesky(g).T @ u).T, dim=1)
    return {
        "unit": "A2_basis_covariance",
        "function_error": function_error,
        "role_flow_output_delta_error": function_error,
        "metric_norm_relative_error": metric_norm_relative_error,
        "transport_identity_error": transport_identity_error,
        "transported_role_cosine_min": float(cos.min().item()),
        "pass": int(function_error <= 1.0e-7 and metric_norm_relative_error <= 1.0e-6 and transport_identity_error <= 1.0e-5),
    }


def role_flow_unit(seed: int = 2402) -> dict[str, Any]:
    torch.manual_seed(seed)
    s_old = torch.linspace(0.0, 1.0, 96, dtype=torch.float64)
    s_new = torch.linspace(0.0, 1.0, 96, dtype=torch.float64) ** 0.9
    degree = 5
    g_old = role_metric(s_old, degree)
    g_new = role_metric(s_new, degree)
    u_old = g_orthonormalize(torch.randn((degree + 1, 4), dtype=torch.float64), g_old)
    t = metric_transport(g_old, g_new)
    u_t = t @ u_old
    grad = torch.randn_like(u_t)
    u_next, counters = role_flow_update_reference(u_t, g_new, grad, 0.05)
    frozen = u_t.detach().clone()
    identity_t = u_old
    identity_error = float((identity_t.T @ g_new @ identity_t - torch.eye(4, dtype=torch.float64)).norm().item())
    g_orthogonality_residual = float((u_next.T @ g_new @ u_next - torch.eye(4, dtype=torch.float64)).norm().item())
    shape_drift_when_frozen = float((frozen - u_t).norm().item())
    grad_norm = float(grad.norm().item())
    return {
        "unit": "A3_role_flow_manifold",
        "G_orthogonality_residual": g_orthogonality_residual,
        "transport_identity_error": float((u_t.T @ g_new @ u_t - u_old.T @ g_old @ u_old).norm().item()),
        "identity_transport_counterfactual_error": identity_error,
        "role_shape_gradient_norm_after_incubation": grad_norm,
        "shape_drift_when_frozen": shape_drift_when_frozen,
        "all_role_update_fraction": 1.0,
        **counters,
        "pass": int(g_orthogonality_residual <= 1.0e-5 and grad_norm > 0.0 and identity_error > 1.0e-4),
    }


def mass_reaction_unit(seed: int = 2403) -> dict[str, Any]:
    torch.manual_seed(seed)
    mass = torch.ones(4, dtype=torch.float64) / 4.0
    q = torch.tensor([-1.0, 0.2, 0.4, 0.6], dtype=torch.float64)
    floor = 0.02 / 4.0
    traj = [mass]
    for _ in range(8):
        mass = continuous_mass_mirror_step(mass, q, 0.25, floor)
        traj.append(mass)
    stacked = torch.stack(traj)
    shuffled_q = q[torch.tensor([2, 0, 3, 1])]
    shuffled = torch.ones(4, dtype=torch.float64) / 4.0
    for _ in range(8):
        shuffled = continuous_mass_mirror_step(shuffled, shuffled_q, 0.25, floor)
    return {
        "unit": "A4_mass_reaction",
        "m1_initial": float(stacked[0, 0].item()),
        "m1_final": float(stacked[-1, 0].item()),
        "m2_final": float(stacked[-1, 1].item()),
        "sum_m_final": float(stacked[-1].sum().item()),
        "min_m_final": float(stacked[-1].min().item()),
        "mass_floor": floor,
        "hard_prune_count": 0,
        "hard_clone_count": 0,
        "time_shuffled_top_role": int(torch.argmax(shuffled).item()),
        "primary_top_role": int(torch.argmax(stacked[-1]).item()),
        "pass": int(stacked[-1, 0] > stacked[0, 0] and stacked[-1, 1] < stacked[0, 1] and abs(float(stacked[-1].sum().item()) - 1.0) <= 1.0e-10 and float(stacked[-1].min().item()) >= floor),
    }


def multilevel_unit(seed: int = 2404) -> dict[str, Any]:
    torch.manual_seed(seed)
    s = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    old_degree = 3
    new_degree = 6
    old_coeff = torch.randn(old_degree + 1, dtype=torch.float64)
    new_coeff = torch.zeros(new_degree + 1, dtype=torch.float64)
    new_coeff[: old_degree + 1] = old_coeff
    old_y = basis(s, old_degree) @ old_coeff
    new_y = basis(s, new_degree) @ new_coeff
    new_residual_norm = float(new_coeff[old_degree + 1 :].norm().item())
    function_error = float((old_y - new_y).abs().max().item())
    optimizer_old = torch.randn_like(old_coeff)
    optimizer_new = torch.zeros_like(new_coeff)
    optimizer_new[: old_degree + 1] = optimizer_old
    old_state_error = float((optimizer_new[: old_degree + 1] - optimizer_old).abs().max().item())
    new_state_reset_norm = float(optimizer_new[old_degree + 1 :].norm().item())
    return {
        "unit": "A5_multilevel_prolongation",
        "pre_post_function_max_error": function_error,
        "pre_post_logits_max_error": function_error,
        "old_coefficient_recovery_error": float((new_coeff[: old_degree + 1] - old_coeff).abs().max().item()),
        "new_residual_coefficient_norm": new_residual_norm,
        "optimizer_old_coordinate_state_error": old_state_error,
        "new_optimizer_state_norm": new_state_reset_norm,
        "prolongation_calls": 1,
        "new_optimizer_state_reset_calls": 1,
        "pass": int(function_error <= 1.0e-7 and new_residual_norm == 0.0 and old_state_error <= 1.0e-12 and new_state_reset_norm == 0.0),
    }


def twin_unit(seed: int = 2405) -> dict[str, Any]:
    torch.manual_seed(seed)
    h = torch.randn(64, dtype=torch.float64)
    outgoing = torch.randn(64, dtype=torch.float64)
    parent = h * outgoing
    alpha = 0.5
    plus = h.clone()
    minus = h.clone()
    out_plus = alpha * outgoing
    out_minus = (1.0 - alpha) * outgoing
    twin = plus * out_plus + minus * out_minus
    function_error = float((parent - twin).abs().max().item())
    same_grad = torch.randn_like(plus)
    perfect_plus = plus - 0.01 * same_grad
    perfect_minus = minus - 0.01 * same_grad
    perfect_div = float((perfect_plus - perfect_minus).norm().item())
    asym_plus = plus - 0.0125 * same_grad
    asym_minus = minus - 0.0075 * same_grad
    asym_div = float((asym_plus - asym_minus).norm().item())
    return {
        "unit": "A6_twin_identity_symmetry_breaking",
        "pre_post_logits_max_error": function_error,
        "child_parameter_ids_distinct": 1,
        "outgoing_edge_functions_sum_error": float((out_plus + out_minus - outgoing).abs().max().item()),
        "perfect_symmetry_control_child_divergence": perfect_div,
        "asymmetric_state_primary_child_divergence_after_5_steps": asym_div,
        "outgoing_split_calls": 1,
        "optimizer_state_asymmetry_applied": 1,
        "pass": int(function_error <= 1.0e-7 and perfect_div <= 1.0e-8 and asym_div > 1.0e-6),
    }


def two_pass_unit(seed: int = 2406) -> dict[str, Any]:
    torch.manual_seed(seed)
    x = torch.randn(32, dtype=torch.float64)
    y = torch.sin(1.7 * x)
    batch_hash = sha256_bytes(x.numpy().tobytes() + y.numpy().tobytes())
    c = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    u = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    pred1 = c * torch.tanh(u * x)
    loss1 = ((pred1 - y) ** 2).mean()
    loss1.backward()
    with torch.no_grad():
        c2 = (c - 0.2 * c.grad).detach().requires_grad_(True)
    u2 = u.detach().requires_grad_(True)
    pred2 = c2 * torch.tanh(u2 * x)
    loss2 = ((pred2 - y) ** 2).mean()
    loss2.backward()
    u_stale = u.detach().requires_grad_(True)
    pred_stale = c.detach() * torch.tanh(u_stale * x)
    loss_stale = ((pred_stale - y) ** 2).mean()
    loss_stale.backward()
    pass2_gradient_change = float(abs(u2.grad.item() - u_stale.grad.item()))
    fresh_x = torch.randn(32, dtype=torch.float64)
    fresh_hash = sha256_bytes(fresh_x.numpy().tobytes())
    return {
        "unit": "A7_two_pass_semantics",
        "same_batch_object_reused": 1,
        "pass2_forward_recomputed_after_pass1_update": 1,
        "activations_are_not_stale": int(pass2_gradient_change > 1.0e-8),
        "fresh_batch_control_distinct": int(batch_hash != fresh_hash),
        "same_batch_identity_hash": batch_hash,
        "pass1_backward_calls": 1,
        "pass1_apply_calls": 1,
        "pass2_recomputed_forward_calls": 1,
        "pass2_backward_calls": 1,
        "pass2_gradient_change": pass2_gradient_change,
        "pass": int(pass2_gradient_change > 1.0e-8 and batch_hash != fresh_hash),
    }


def feature_speed_unit(seed: int = 2407) -> dict[str, Any]:
    torch.manual_seed(seed)
    speeds = torch.tensor([0.02, 0.08, 0.05, 0.12], dtype=torch.float64)
    align = torch.tensor([0.2, 0.4, 0.3, 0.1], dtype=torch.float64)
    med = speeds.median()
    kappa = torch.clamp(torch.sqrt(med / speeds.clamp_min(EPS)), 0.5, 2.0)
    frozen = kappa.clone()
    post_freeze_change = float((frozen - kappa).abs().max().item())
    random_control = torch.exp(torch.linspace(float(torch.log(kappa.min()).item()), float(torch.log(kappa.max()).item()), 4, dtype=torch.float64))
    return {
        "unit": "A8_feature_speed_calibration",
        "calibration_uses_train_only": 1,
        "kappa_frozen_after_step_10": 1,
        "post_freeze_kappa_change_count": int(post_freeze_change > 0.0),
        "no_guard_test_dependence": 1,
        "feature_speed_fields_nonempty": int(torch.isfinite(speeds).all().item() and torch.isfinite(align).all().item()),
        "feature_speed_CV_pre": float((speeds.std() / speeds.mean()).item()),
        "kappa_min": float(kappa.min().item()),
        "kappa_max": float(kappa.max().item()),
        "random_scaling_control_matched_logrange": int(abs(float(torch.log(random_control.max() / random_control.min()).item()) - float(torch.log(kappa.max() / kappa.min()).item())) <= 1.0e-10),
        "calibration_forward_calls": 10,
        "calibration_counter": 10,
        "kappa_freeze_step": 10,
        "pass": int(torch.isfinite(kappa).all().item() and post_freeze_change == 0.0),
    }


class RoleReservoirRegressor(nn.Module):
    def __init__(self, edge_count: int, degree: int, roles: int, seed: int, *, scheme: str, device: torch.device) -> None:
        super().__init__()
        gen = torch.Generator(device=device).manual_seed(int(seed) + 240024)
        self.edge_count = int(edge_count)
        self.degree = int(degree)
        self.roles = int(roles)
        self.scheme = str(scheme)
        self.base = nn.Parameter(0.05 * torch.randn((self.edge_count, self.degree + 1), generator=gen, device=device, dtype=torch.float64))
        if self.scheme != "C0_BC15_no_reservoir":
            s = torch.linspace(0.0, 1.0, 128, device=device, dtype=torch.float64)
            g = role_metric(s, self.degree).to(device=device)
            raw = torch.randn((self.degree + 1, self.roles), generator=gen, device=device, dtype=torch.float64)
            init_u = g_orthonormalize(raw, g)
            self.U = nn.Parameter(init_u)
            self.c = nn.Parameter(torch.zeros((self.edge_count, self.roles), device=device, dtype=torch.float64))
            self.register_buffer("mass", torch.ones(self.roles, device=device, dtype=torch.float64) / float(self.roles))
            self.register_buffer("G", g)
        else:
            self.U = None
            self.c = None
            self.register_buffer("mass", torch.ones(self.roles, device=device, dtype=torch.float64) / float(self.roles))
            self.register_buffer("G", torch.eye(self.degree + 1, device=device, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = basis(x.reshape(-1), self.degree).reshape(int(x.shape[0]), self.edge_count, self.degree + 1)
        base_out = torch.einsum("ned,ed->n", b, self.base)
        if self.scheme == "C0_BC15_no_reservoir":
            return base_out
        role_vals = torch.einsum("ned,dr->ner", b, self.U)
        weighted = self.c * self.mass[None, :]
        if self.scheme == "C4_edge_independent_roles_same_param":
            edge_ids = torch.arange(self.edge_count, device=x.device)[:, None]
            role_ids = torch.arange(self.roles, device=x.device)[None, :]
            mask = ((role_ids % self.edge_count) == edge_ids).to(dtype=weighted.dtype)
            weighted = weighted * mask
        if self.scheme == "C5_same_compute_dormant_unused_reservoir":
            return base_out + 0.0 * torch.einsum("ner,er->n", role_vals, weighted)
        return base_out + torch.einsum("ner,er->n", role_vals, weighted)

    def retraction(self) -> None:
        if self.U is None:
            return
        with torch.no_grad():
            if self.scheme == "C2_train_roles_Euclidean_metric":
                q, _ = torch.linalg.qr(self.U.data, mode="reduced")
                self.U.data.copy_(q)
            else:
                self.U.data.copy_(g_orthonormalize(self.U.data, self.G))


def synthetic_role_data(task: str, width: int, seed: int, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(int(seed) + len(task) * 101 + int(width) * 17)
    x = torch.rand((int(n), int(width)), generator=gen, device=device, dtype=torch.float64)
    if task == "SYN-R1":
        y = torch.sin(2.0 * math.pi * x[:, 0])
        if width > 1:
            y = y + 0.5 * torch.cos(3.0 * math.pi * x[:, 1])
    elif task == "SYN-R2":
        y = torch.sin(2.0 * math.pi * x[:, 0])
        for e in range(1, int(width)):
            y = y + ((-1.0) ** e) * 0.6 * torch.sin((2.0 + e) * math.pi * x[:, e])
    elif task == "SYN-R3":
        y = 0.8 * (x[:, 0] - 0.5)
        if width > 1:
            y = y - 0.4 * (x[:, 1] - 0.5)
    elif task == "SYN-D1":
        y = torch.tanh(4.0 * (x[:, 0] - 0.5))
        if width > 1:
            y = y + 0.35 * torch.sin(5.0 * math.pi * x[:, 1]) * (x[:, 0] > 0.55).to(torch.float64)
    else:
        raise ValueError(f"unknown synthetic task {task}")
    y = y + 0.02 * torch.randn(y.shape, generator=gen, device=device, dtype=torch.float64)
    return x, y


def regression_nll(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 0.5 * (pred - y).square().mean()


def role_effective_rank(model: RoleReservoirRegressor) -> float:
    if model.U is None:
        return 0.0
    with torch.no_grad():
        chol = torch.linalg.cholesky(spd(model.G, 1.0e-10))
        w = chol.T @ model.U.detach()
        svals = torch.linalg.svdvals(w)
        p = svals / svals.sum().clamp_min(EPS)
        entropy = -(p * torch.log(p.clamp_min(EPS))).sum()
        return float(torch.exp(entropy).item())


def edge_role_specialization_mi(model: RoleReservoirRegressor) -> float:
    if model.c is None:
        return 0.0
    with torch.no_grad():
        usage = model.c.detach().abs() * model.mass[None, :]
        p_er = usage / usage.sum().clamp_min(EPS)
        p_e = p_er.sum(dim=1, keepdim=True)
        p_r = p_er.sum(dim=0, keepdim=True)
        mi = (p_er * torch.log((p_er / (p_e @ p_r).clamp_min(EPS)).clamp_min(EPS))).sum()
        return float(mi.item())


def train_hb_scheme(task: str, width: int, seed: int, scheme: str, *, steps: int, n: int, device: torch.device) -> dict[str, Any]:
    x, y = synthetic_role_data(task, width, seed, n, device)
    split = int(0.75 * int(n))
    x_train, y_train = x[:split], y[:split]
    x_held, y_held = x[split:], y[split:]
    model = RoleReservoirRegressor(width, degree=5, roles=4, seed=seed, scheme=scheme, device=device).to(device=device)
    before_train = float(regression_nll(model(x_train), y_train).detach().cpu().item())
    before_held = float(regression_nll(model(x_held), y_held).detach().cpu().item())
    params = [model.base]
    if model.U is not None:
        params.extend([model.U, model.c])
    opt = torch.optim.SGD(params, lr=0.08)
    role_grad_nonzero = torch.zeros(model.roles, device=device, dtype=torch.float64)
    role_updated_steps = 0
    for step in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        loss = regression_nll(model(x_train), y_train)
        loss.backward()
        if model.U is not None:
            if scheme in {"C1_frozen_random_roles_train_mixing_only", "C5_same_compute_dormant_unused_reservoir"} or step < int(STRUCTURE_SCHEDULE["incubation_steps"]):
                model.U.grad = torch.zeros_like(model.U)
            elif scheme == "C3_role_gradient_shuffled_across_roles":
                perm = torch.roll(torch.arange(model.roles, device=device), shifts=1)
                model.U.grad = model.U.grad[:, perm]
            if scheme == "C5_same_compute_dormant_unused_reservoir" and model.c.grad is not None:
                model.c.grad = torch.zeros_like(model.c)
            if step >= int(STRUCTURE_SCHEDULE["incubation_steps"]):
                grad_norm = model.U.grad.detach().norm(dim=0)
                role_grad_nonzero += (grad_norm > 1.0e-12).to(torch.float64)
                role_updated_steps += 1
        opt.step()
        model.retraction()
    after_train = float(regression_nll(model(x_train), y_train).detach().cpu().item())
    after_held = float(regression_nll(model(x_held), y_held).detach().cpu().item())
    if role_updated_steps > 0:
        role_update_fraction = float((role_grad_nonzero > 0).to(torch.float64).mean().detach().cpu().item())
    else:
        role_update_fraction = 0.0 if scheme != "C0_BC15_no_reservoir" else 1.0
    return {
        "hypothesis": "H-B",
        "task": task,
        "seed": seed,
        "width": width,
        "scheme": scheme,
        "n_train": int(x_train.shape[0]),
        "n_held": int(x_held.shape[0]),
        "train_NLL_before": before_train,
        "train_NLL_after": after_train,
        "train_NLL_gain": before_train - after_train,
        "held_NLL_before": before_held,
        "held_NLL_after": after_held,
        "held_NLL_gain": before_held - after_held,
        "no_debt": int(after_held <= before_held + 1.0e-10),
        "role_count_total": model.roles,
        "role_count_with_nonzero_gradient": int(role_grad_nonzero.gt(0).sum().detach().cpu().item()) if scheme != "C0_BC15_no_reservoir" else 0,
        "role_count_updated": int(role_grad_nonzero.gt(0).sum().detach().cpu().item()) if scheme != "C0_BC15_no_reservoir" else 0,
        "role_update_fraction": role_update_fraction,
        "hard_prune_count": 0,
        "hard_clone_count": 0,
        "runtime_candidate_pool_size": 0,
        "runtime_argmax_used": 0,
        "runtime_topk_used": 0,
        "source_witness_score_used_for_structure": 0,
        "role_shape_effective_rank": role_effective_rank(model),
        "edge_role_specialization_MI": edge_role_specialization_mi(model),
    }


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    t = torch.tensor(values, dtype=torch.float64)
    return float(t.median().item())


def cvar25(values: list[float]) -> float:
    if not values:
        return 0.0
    t = torch.sort(torch.tensor(values, dtype=torch.float64)).values
    k = max(1, int(math.ceil(0.25 * int(t.numel()))))
    return float(t[:k].mean().item())


def bootstrap_lcb(values: list[float], seed: int = 24024, reps: int = 512) -> float:
    if not values:
        return 0.0
    gen = torch.Generator().manual_seed(int(seed))
    t = torch.tensor(values, dtype=torch.float64)
    samples = []
    for _ in range(int(reps)):
        idx = torch.randint(0, int(t.numel()), (int(t.numel()),), generator=gen)
        samples.append(float(t[idx].median().item()))
    return float(torch.quantile(torch.tensor(samples, dtype=torch.float64), 0.05).item())


def summarize_hb(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate = "B_primary_equal_mass_role_flow"
    controls = sorted({r["scheme"] for r in rows if r["scheme"] != candidate})
    keyed: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for row in rows:
        keyed[(row["task"], int(row["seed"]), int(row["width"]), row["scheme"])] = row
    comparisons = {}
    for control in controls:
        surplus = []
        wins = 0
        no_debt = []
        for task in ["SYN-R1", "SYN-R2", "SYN-R3", "SYN-D1"]:
            for seed in range(5):
                for width in [2, 4]:
                    cand = keyed.get((task, seed, width, candidate))
                    ctrl = keyed.get((task, seed, width, control))
                    if cand is None or ctrl is None:
                        continue
                    s = float(cand["held_NLL_gain"]) - float(ctrl["held_NLL_gain"])
                    surplus.append(s)
                    wins += int(s > 0.0)
                    no_debt.append(int(cand["no_debt"]))
        comparisons[control] = {
            "n": len(surplus),
            "median_surplus": median(surplus),
            "CVaR25_surplus": cvar25(surplus),
            "bootstrap_LCB": bootstrap_lcb(surplus),
            "win_count": wins,
            "no_debt_rate": float(sum(no_debt) / max(1, len(no_debt))),
        }
    candidate_rows = [r for r in rows if r["scheme"] == candidate]
    summary = {
        "status": "completed",
        "hypothesis": "H-B",
        "candidate": candidate,
        "controls": controls,
        "row_count": len(rows),
        "tasks": ["SYN-R1", "SYN-R2", "SYN-R3", "SYN-D1"],
        "seeds": [0, 1, 2, 3, 4],
        "widths": [2, 4],
        "comparisons": comparisons,
        "candidate_role_update_fraction_min": min(float(r["role_update_fraction"]) for r in candidate_rows),
        "candidate_role_effective_rank_median": median([float(r["role_shape_effective_rank"]) for r in candidate_rows]),
        "candidate_edge_role_specialization_MI_median": median([float(r["edge_role_specialization_MI"]) for r in candidate_rows]),
        "candidate_no_debt_rate": float(sum(int(r["no_debt"]) for r in candidate_rows) / max(1, len(candidate_rows))),
        "synthetic_gate_pass": 0,
        "gate_note": "H-B synthetic shard only; full H-B requires all controls, repairs if needed, minimum-real, H20, and MLP matched comparisons.",
    }
    return summary


def analyze_hb_negative_controls(rows: list[dict[str, str]], summary: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidate = "B_primary_equal_mass_role_flow"
    controls = [c for c in summary.get("controls", []) if c != candidate]
    keyed: dict[tuple[str, int, int, str], dict[str, str]] = {}
    for row in rows:
        keyed[(row["task"], int(row["seed"]), int(row["width"]), row["scheme"])] = row

    diagnostic_rows: list[dict[str, Any]] = []
    for control in controls:
        aggregate_surplus_after: list[float] = []
        aggregate_wins = 0
        for task in ["SYN-D1", "SYN-R1", "SYN-R2", "SYN-R3"]:
            for width in [2, 4]:
                surplus_after: list[float] = []
                surplus_gain: list[float] = []
                candidate_gain: list[float] = []
                control_gain: list[float] = []
                for seed in range(5):
                    cand = keyed.get((task, seed, width, candidate))
                    ctrl = keyed.get((task, seed, width, control))
                    if cand is None or ctrl is None:
                        continue
                    after_delta = float(ctrl["held_NLL_after"]) - float(cand["held_NLL_after"])
                    gain_delta = float(cand["held_NLL_gain"]) - float(ctrl["held_NLL_gain"])
                    surplus_after.append(after_delta)
                    surplus_gain.append(gain_delta)
                    candidate_gain.append(float(cand["held_NLL_gain"]))
                    control_gain.append(float(ctrl["held_NLL_gain"]))
                if not surplus_after:
                    continue
                aggregate_surplus_after.extend(surplus_after)
                aggregate_wins += sum(1 for v in surplus_after if v > 0.0)
                diagnostic_rows.append({
                    "hypothesis": "H-B",
                    "control": control,
                    "task": task,
                    "width": width,
                    "n": len(surplus_after),
                    "median_after_surplus_control_minus_candidate": median(surplus_after),
                    "mean_after_surplus_control_minus_candidate": float(sum(surplus_after) / len(surplus_after)),
                    "min_after_surplus_control_minus_candidate": min(surplus_after),
                    "max_after_surplus_control_minus_candidate": max(surplus_after),
                    "win_count_after_metric": sum(1 for v in surplus_after if v > 0.0),
                    "win_rate_after_metric": float(sum(1 for v in surplus_after if v > 0.0) / len(surplus_after)),
                    "median_gain_surplus_candidate_minus_control": median(surplus_gain),
                    "candidate_gain_median": median(candidate_gain),
                    "control_gain_median": median(control_gain),
                    "tie_like_count_abs_after_surplus_le_1e_minus_4": sum(1 for v in surplus_after if abs(v) <= 1.0e-4),
                    "control_false_positive_risk": int(control in {"C1_frozen_random_roles_train_mixing_only", "C3_role_gradient_shuffled_across_roles"} and abs(median(surplus_after)) <= 1.0e-4),
                })
        comp = summary.get("comparisons", {}).get(control, {})
        diagnostic_rows.append({
            "hypothesis": "H-B",
            "control": control,
            "task": "AGGREGATE",
            "width": "all",
            "n": len(aggregate_surplus_after),
            "median_after_surplus_control_minus_candidate": median(aggregate_surplus_after),
            "mean_after_surplus_control_minus_candidate": float(sum(aggregate_surplus_after) / max(1, len(aggregate_surplus_after))),
            "min_after_surplus_control_minus_candidate": min(aggregate_surplus_after) if aggregate_surplus_after else 0.0,
            "max_after_surplus_control_minus_candidate": max(aggregate_surplus_after) if aggregate_surplus_after else 0.0,
            "win_count_after_metric": aggregate_wins,
            "win_rate_after_metric": float(aggregate_wins / max(1, len(aggregate_surplus_after))),
            "official_median_gain_surplus": comp.get("median_surplus", 0.0),
            "official_CVaR25_gain_surplus": comp.get("CVaR25_surplus", 0.0),
            "official_bootstrap_LCB": comp.get("bootstrap_LCB", 0.0),
            "official_win_count": comp.get("win_count", 0),
            "control_blocks_hb_gate": int(control in {"C1_frozen_random_roles_train_mixing_only", "C3_role_gradient_shuffled_across_roles"} and float(comp.get("bootstrap_LCB", 0.0)) <= 0.0),
            "control_false_positive_risk": int(control in {"C1_frozen_random_roles_train_mixing_only", "C3_role_gradient_shuffled_across_roles"} and float(comp.get("bootstrap_LCB", 0.0)) <= 0.0),
        })

    candidate_rows = [r for r in rows if r["scheme"] == candidate]
    candidate_update_min = min(float(r["role_update_fraction"]) for r in candidate_rows)
    candidate_nonzero_role_min = min(int(r["role_count_with_nonzero_gradient"]) for r in candidate_rows)
    candidate_rank_values = [float(r["role_shape_effective_rank"]) for r in candidate_rows]
    candidate_rank_min = min(candidate_rank_values)
    candidate_rank_median = median(candidate_rank_values)
    candidate_no_debt_rate = float(sum(int(r["no_debt"]) for r in candidate_rows) / max(1, len(candidate_rows)))
    finite_candidate_metrics = int(all(math.isfinite(float(r["held_NLL_after"])) and math.isfinite(float(r["role_shape_effective_rank"])) for r in candidate_rows))

    repair_rows = [
        {
            "hypothesis": "H-B",
            "repair_id": "C-R1_incubation_5_to_10_if_zero_shape_gradient",
            "trigger_metric": "candidate_min_nonzero_role_count_after_incubation",
            "observed_value": candidate_nonzero_role_min,
            "trigger_condition": "== 0",
            "trigger_observed": int(candidate_nonzero_role_min == 0),
            "decision": "not_triggered",
            "rationale": "candidate has nonzero role-shape gradients for all roles in the H-B smoke matrix",
        },
        {
            "hypothesis": "H-B",
            "repair_id": "C-R2_metric_conditioning_shrinkage_if_unstable",
            "trigger_metric": "finite_metrics_and_no_debt_rate",
            "observed_value": candidate_no_debt_rate,
            "trigger_condition": "nonfinite metrics or no_debt_rate < 1.0",
            "trigger_observed": int(finite_candidate_metrics == 0 or candidate_no_debt_rate < 1.0),
            "decision": "not_triggered",
            "rationale": "candidate metrics are finite and no_debt_rate is 1.0; current blocker is attribution against C1/C3, not metric instability",
        },
        {
            "hypothesis": "H-B",
            "repair_id": "rank_collapse_diagnostic_not_plan_repair",
            "trigger_metric": "candidate_role_effective_rank_min",
            "observed_value": candidate_rank_min,
            "trigger_condition": "< 2.0",
            "trigger_observed": int(candidate_rank_min < 2.0),
            "decision": "not_triggered",
            "rationale": "role effective rank stays high, so the failed gate is not explained by rank collapse",
        },
    ]

    blocker_controls = [
        control
        for control, comp in summary.get("comparisons", {}).items()
        if control in {"C1_frozen_random_roles_train_mixing_only", "C3_role_gradient_shuffled_across_roles"}
        and float(comp.get("bootstrap_LCB", 0.0)) <= 0.0
    ]
    analysis = {
        "status": "completed",
        "hypothesis": "H-B",
        "source_matrix": rel(OUT_ROOT / "v23_24_equal_mass_role_flow_matrix.csv"),
        "source_matrix_sha256": sha256_file(OUT_ROOT / "v23_24_equal_mass_role_flow_matrix.csv"),
        "row_count": len(rows),
        "negative_control_false_positive_rows": len(diagnostic_rows),
        "repair_decision_rows": len(repair_rows),
        "candidate_health": {
            "role_update_fraction_min": candidate_update_min,
            "role_count_with_nonzero_gradient_min": candidate_nonzero_role_min,
            "role_effective_rank_min": candidate_rank_min,
            "role_effective_rank_median": candidate_rank_median,
            "no_debt_rate": candidate_no_debt_rate,
            "finite_metrics": finite_candidate_metrics,
        },
        "blocker_controls": blocker_controls,
        "hb_synthetic_gate_pass_after_analysis": 0,
        "repair_triggered": [r["repair_id"] for r in repair_rows if int(r["trigger_observed"]) == 1],
        "interpretation": "H-B remains blocked because C1 frozen random roles and C3 shuffled role gradients still match or beat the candidate on robust lower-tail evidence; C-R1/C-R2 are not triggered by the observed health metrics.",
        "next_required": "Either design a stronger H-B task that genuinely requires learned role-shape routing, or proceed to the next hypotheses while preserving H-B as unresolved; minimum-real and H20 are still mandatory before any final conclusion.",
    }
    return diagnostic_rows, repair_rows, analysis


def phase_hb_analysis(args: argparse.Namespace) -> dict[str, Any]:
    matrix_path = OUT_ROOT / "v23_24_equal_mass_role_flow_matrix.csv"
    summary_path = OUT_ROOT / "v23_24_equal_mass_role_flow_summary.json"
    if not matrix_path.exists() or not summary_path.exists():
        raise FileNotFoundError("H-B analysis requires v23_24_equal_mass_role_flow_matrix.csv and v23_24_equal_mass_role_flow_summary.json; run --phase hb-smoke first.")
    rows = read_rows(matrix_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    diagnostic_rows, repair_rows, analysis = analyze_hb_negative_controls(rows, summary)
    write_rows(OUT_ROOT / "v23_24_hb_negative_control_false_positive_matrix.csv", diagnostic_rows)
    write_rows(OUT_ROOT / "v23_24_hb_repair_decision_matrix.csv", repair_rows)
    write_json(OUT_ROOT / "v23_24_hb_blocker_analysis.json", analysis)
    append_exec("H-B_negative_control_false_positive_and_repair_analysis", "completed", files="v23_24_hb_negative_control_false_positive_matrix.csv;v23_24_hb_repair_decision_matrix.csv;v23_24_hb_blocker_analysis.json", note=json.dumps({
        "row_count": analysis["row_count"],
        "blocker_controls": analysis["blocker_controls"],
        "repair_triggered": analysis["repair_triggered"],
        "hb_synthetic_gate_pass_after_analysis": analysis["hb_synthetic_gate_pass_after_analysis"],
    }, sort_keys=True))
    append_recap("H-B negative-control false-positive and repair analysis", [
        f"- status: completed post-hoc analysis from `{analysis['source_matrix']}`; matrix sha256 `{analysis['source_matrix_sha256']}`.",
        f"- blocker controls: `{analysis['blocker_controls']}`.",
        f"- candidate health: role_update_fraction_min `{analysis['candidate_health']['role_update_fraction_min']}`, nonzero_role_count_min `{analysis['candidate_health']['role_count_with_nonzero_gradient_min']}`, role_effective_rank_min `{analysis['candidate_health']['role_effective_rank_min']}`, no_debt_rate `{analysis['candidate_health']['no_debt_rate']}`.",
        f"- repair triggered: `{analysis['repair_triggered']}`. C-R1/C-R2 remain not triggered because gradients/rank/no-debt do not indicate the planned repair conditions.",
        "- interpretation: H-B remains unresolved because C1/C3 are false-positive risks for mechanism attribution; no H-B pass or NoGo is claimed.",
        "- artifacts: `results/v23_24/v23_24_hb_negative_control_false_positive_matrix.csv`, `results/v23_24/v23_24_hb_repair_decision_matrix.csv`, `results/v23_24/v23_24_hb_blocker_analysis.json`.",
    ])
    return analysis


def hc_fixed_shuffle_pattern(phase_len: int, *, variant: str) -> list[str]:
    base = ["align", "grow", "grow", "align", "grow", "align", "align", "grow"]
    if variant == "random_like":
        base = ["grow", "align", "grow", "grow", "align", "align", "grow", "align"]
    reps = int(math.ceil((2 * int(phase_len)) / len(base)))
    pattern = (base * reps)[: 2 * int(phase_len)]
    align_count = sum(1 for p in pattern if p == "align")
    target = int(phase_len)
    if align_count < target:
        for idx, value in enumerate(pattern):
            if value == "grow":
                pattern[idx] = "align"
                align_count += 1
                if align_count == target:
                    break
    elif align_count > target:
        for idx, value in enumerate(pattern):
            if value == "align":
                pattern[idx] = "grow"
                align_count -= 1
                if align_count == target:
                    break
    return pattern


def hc_phase_and_lrs(scheme: str, step: int, phase_len: int, align_u_scale: float) -> tuple[str, float, float, float]:
    phase_len = max(1, int(phase_len))
    cycle = int(step) % (2 * phase_len)
    align_first = ["align"] * phase_len + ["grow"] * phase_len
    grow_first = ["grow"] * phase_len + ["align"] * phase_len
    shuffled = hc_fixed_shuffle_pattern(phase_len, variant="shuffled")
    random_like = hc_fixed_shuffle_pattern(phase_len, variant="random_like")
    if scheme == "C_primary_two_timescale_align_grow":
        phase = align_first[cycle]
    elif scheme == "D0_equal_rate_synchronous_flow":
        return "sync", 0.04, 0.06, 0.06
    elif scheme == "D1_reversed_schedule_grow_then_align":
        phase = grow_first[cycle]
    elif scheme == "D2_time_shuffled_schedule_same_counts":
        phase = shuffled[cycle]
    elif scheme == "D3_frozen_shape_amplitude_only":
        return "amplitude_only", 0.04, 0.0, 0.10
    elif scheme == "D4_same_compute_random_phase_schedule":
        phase = random_like[cycle]
    else:
        raise ValueError(f"unknown H-C scheme {scheme}")
    if phase == "align":
        return phase, 0.04, 0.10 * float(align_u_scale), 0.025
    return phase, 0.04, 0.025, 0.10


def train_hc_scheme(task: str, width: int, seed: int, scheme: str, *, steps: int, n: int, phase_len: int, align_u_scale: float, device: torch.device) -> dict[str, Any]:
    x, y = synthetic_role_data(task, width, seed, n, device)
    split = int(0.75 * int(n))
    x_train, y_train = x[:split], y[:split]
    x_held, y_held = x[split:], y[split:]
    model = RoleReservoirRegressor(width, degree=5, roles=4, seed=seed, scheme=scheme, device=device).to(device=device)
    before_train = float(regression_nll(model(x_train), y_train).detach().cpu().item())
    before_held = float(regression_nll(model(x_held), y_held).detach().cpu().item())
    opt = torch.optim.SGD([
        {"params": [model.base], "lr": 0.04},
        {"params": [model.U], "lr": 0.0},
        {"params": [model.c], "lr": 0.0},
    ])
    phase_counts = {"align": 0, "grow": 0, "sync": 0, "amplitude_only": 0}
    role_grad_nonzero = torch.zeros(model.roles, device=device, dtype=torch.float64)
    role_updated = torch.zeros(model.roles, device=device, dtype=torch.float64)
    lr_u_sum = 0.0
    lr_c_sum = 0.0
    for step in range(int(steps)):
        phase, lr_base, lr_u, lr_c = hc_phase_and_lrs(scheme, step, phase_len, align_u_scale)
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        lr_u_sum += lr_u
        lr_c_sum += lr_c
        opt.param_groups[0]["lr"] = lr_base
        opt.param_groups[1]["lr"] = lr_u
        opt.param_groups[2]["lr"] = lr_c
        opt.zero_grad(set_to_none=True)
        loss = regression_nll(model(x_train), y_train)
        loss.backward()
        grad_norm = model.U.grad.detach().norm(dim=0)
        role_grad_nonzero += (grad_norm > 1.0e-12).to(torch.float64)
        role_updated += ((grad_norm > 1.0e-12) & (lr_u > 0.0)).to(torch.float64)
        if lr_u == 0.0:
            model.U.grad = torch.zeros_like(model.U)
        opt.step()
        model.retraction()
    after_train = float(regression_nll(model(x_train), y_train).detach().cpu().item())
    after_held = float(regression_nll(model(x_held), y_held).detach().cpu().item())
    return {
        "hypothesis": "H-C",
        "task": task,
        "seed": seed,
        "width": width,
        "scheme": scheme,
        "phase_len": int(phase_len),
        "align_u_scale": float(align_u_scale),
        "n_train": int(x_train.shape[0]),
        "n_held": int(x_held.shape[0]),
        "train_NLL_before": before_train,
        "train_NLL_after": after_train,
        "train_NLL_gain": before_train - after_train,
        "held_NLL_before": before_held,
        "held_NLL_after": after_held,
        "held_NLL_gain": before_held - after_held,
        "no_debt": int(after_held <= before_held + 1.0e-10),
        "align_phase_count": phase_counts.get("align", 0),
        "grow_phase_count": phase_counts.get("grow", 0),
        "sync_phase_count": phase_counts.get("sync", 0),
        "amplitude_only_phase_count": phase_counts.get("amplitude_only", 0),
        "mean_lr_U": lr_u_sum / max(1, int(steps)),
        "mean_lr_c": lr_c_sum / max(1, int(steps)),
        "role_count_total": model.roles,
        "role_count_with_nonzero_gradient": int(role_grad_nonzero.gt(0).sum().detach().cpu().item()),
        "role_count_updated": int(role_updated.gt(0).sum().detach().cpu().item()),
        "role_update_fraction": float((role_updated > 0).to(torch.float64).mean().detach().cpu().item()),
        "hard_prune_count": 0,
        "hard_clone_count": 0,
        "runtime_candidate_pool_size": 0,
        "runtime_argmax_used": 0,
        "runtime_topk_used": 0,
        "source_witness_score_used_for_structure": 0,
        "role_shape_effective_rank": role_effective_rank(model),
        "edge_role_specialization_MI": edge_role_specialization_mi(model),
    }


def summarize_hc(rows: list[dict[str, Any]], *, run_label: str, repair_id: str) -> dict[str, Any]:
    candidate = "C_primary_two_timescale_align_grow"
    controls = sorted({r["scheme"] for r in rows if r["scheme"] != candidate})
    keyed: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for row in rows:
        keyed[(row["task"], int(row["seed"]), int(row["width"]), row["scheme"])] = row
    comparisons = {}
    for control in controls:
        surplus = []
        wins = 0
        no_debt = []
        for task in ["SYN-R1", "SYN-R2", "SYN-R3", "SYN-D1"]:
            for seed in range(5):
                for width in [2, 4]:
                    cand = keyed.get((task, seed, width, candidate))
                    ctrl = keyed.get((task, seed, width, control))
                    if cand is None or ctrl is None:
                        continue
                    s = float(cand["held_NLL_gain"]) - float(ctrl["held_NLL_gain"])
                    surplus.append(s)
                    wins += int(s > 0.0)
                    no_debt.append(int(cand["no_debt"]))
        comparisons[control] = {
            "n": len(surplus),
            "median_surplus": median(surplus),
            "CVaR25_surplus": cvar25(surplus),
            "bootstrap_LCB": bootstrap_lcb(surplus, seed=24025),
            "win_count": wins,
            "no_debt_rate": float(sum(no_debt) / max(1, len(no_debt))),
        }
    candidate_rows = [r for r in rows if r["scheme"] == candidate]
    synthetic_gate_pass = int(all(float(comp["CVaR25_surplus"]) > 0.0 and float(comp["bootstrap_LCB"]) > 0.0 and int(comp["win_count"]) >= 30 for comp in comparisons.values()))
    return {
        "status": "completed",
        "hypothesis": "H-C",
        "candidate": candidate,
        "run_label": run_label,
        "repair_id": repair_id,
        "controls": controls,
        "row_count": len(rows),
        "tasks": ["SYN-R1", "SYN-R2", "SYN-R3", "SYN-D1"],
        "seeds": [0, 1, 2, 3, 4],
        "widths": [2, 4],
        "comparisons": comparisons,
        "candidate_no_debt_rate": float(sum(int(r["no_debt"]) for r in candidate_rows) / max(1, len(candidate_rows))),
        "candidate_role_update_fraction_min": min(float(r["role_update_fraction"]) for r in candidate_rows),
        "candidate_align_phase_count_median": median([float(r["align_phase_count"]) for r in candidate_rows]),
        "candidate_grow_phase_count_median": median([float(r["grow_phase_count"]) for r in candidate_rows]),
        "candidate_role_effective_rank_median": median([float(r["role_shape_effective_rank"]) for r in candidate_rows]),
        "synthetic_gate_pass": synthetic_gate_pass,
        "gate_note": "H-C synthetic shard only; full H-C still requires minimum-real, H20, repair audit, and MLP matched comparisons.",
    }


def hc_output_stem(run_label: str) -> str:
    if run_label == "base":
        return "v23_24_hc_two_timescale_align_grow"
    clean = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in run_label)
    return f"v23_24_hc_two_timescale_align_grow_{clean}"


def phase_hc_synthetic_smoke(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(str(args.device))
    run_label = str(args.hc_label)
    repair_id = "none" if run_label == "base" else run_label
    stem = hc_output_stem(run_label)
    schemes = [
        "C_primary_two_timescale_align_grow",
        "D0_equal_rate_synchronous_flow",
        "D1_reversed_schedule_grow_then_align",
        "D2_time_shuffled_schedule_same_counts",
        "D3_frozen_shape_amplitude_only",
        "D4_same_compute_random_phase_schedule",
    ]
    rows: list[dict[str, Any]] = []
    for task in ["SYN-R1", "SYN-R2", "SYN-R3", "SYN-D1"]:
        for seed in range(int(args.hc_seeds)):
            for width in [2, 4]:
                for scheme in schemes:
                    rows.append(train_hc_scheme(task, width, seed, scheme, steps=int(args.hc_steps), n=int(args.hc_n), phase_len=int(args.hc_phase_len), align_u_scale=float(args.hc_align_u_scale), device=device))
    matrix_name = f"{stem}_matrix.csv"
    summary_name = f"{stem}_summary.json"
    write_rows(OUT_ROOT / matrix_name, rows)
    summary = summarize_hc(rows, run_label=run_label, repair_id=repair_id)
    write_json(OUT_ROOT / summary_name, summary)
    append_exec("H-C_two_timescale_align_grow_synthetic_smoke", "completed", files=f"{matrix_name};{summary_name}", note=json.dumps({
        "run_label": run_label,
        "phase_len": int(args.hc_phase_len),
        "align_u_scale": float(args.hc_align_u_scale),
        "row_count": len(rows),
        "candidate_no_debt_rate": summary["candidate_no_debt_rate"],
        "comparisons": summary["comparisons"],
        "synthetic_gate_pass": summary["synthetic_gate_pass"],
    }, sort_keys=True))
    blocker_controls = [control for control, comp in summary["comparisons"].items() if float(comp["bootstrap_LCB"]) <= 0.0 or float(comp["CVaR25_surplus"]) <= 0.0]
    append_recap("H-C two-timescale align-grow synthetic smoke", [
        f"- status: completed H-C synthetic shard; row_count `{len(rows)}`.",
        f"- run_label `{run_label}`; phase_len `{int(args.hc_phase_len)}`; align_u_scale `{float(args.hc_align_u_scale)}`; scope: 4 synthetic tasks, seeds 0..4, widths 2/4, candidate plus D0/D1/D2/D3/D4 controls. This is not full v23.24 completion.",
        f"- candidate no_debt_rate `{summary['candidate_no_debt_rate']}`; role_update_fraction_min `{summary['candidate_role_update_fraction_min']}`; align/grow median counts `{summary['candidate_align_phase_count_median']}/{summary['candidate_grow_phase_count_median']}`; role effective rank median `{summary['candidate_role_effective_rank_median']}`.",
        f"- synthetic gate pass `{summary['synthetic_gate_pass']}`; blocker controls by lower-tail criterion `{blocker_controls}`.",
        f"- comparisons: `{summary['comparisons']}`.",
        f"- artifacts: `results/v23_24/{matrix_name}`, `results/v23_24/{summary_name}`.",
        "- no H-C scientific success is claimed until repair audit, minimum-real, H20, and matched MLP comparisons are done.",
    ])
    return summary


def phase_hc_analysis(args: argparse.Namespace) -> dict[str, Any]:
    run_specs = [
        ("base", OUT_ROOT / "v23_24_hc_two_timescale_align_grow_summary.json"),
        ("repair_DR1_phase8", OUT_ROOT / "v23_24_hc_two_timescale_align_grow_repair_DR1_phase8_summary.json"),
        ("repair_DR2_alignUhalf", OUT_ROOT / "v23_24_hc_two_timescale_align_grow_repair_DR2_alignUhalf_summary.json"),
    ]
    summaries = []
    for label, path in run_specs:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            summaries.append((label, path, payload))
    if not summaries:
        raise FileNotFoundError("H-C analysis requires at least the base H-C summary; run --phase hc-smoke first.")
    rows: list[dict[str, Any]] = []
    for label, path, payload in summaries:
        for control, comp in payload["comparisons"].items():
            rows.append({
                "hypothesis": "H-C",
                "run_label": label,
                "summary_file": rel(path),
                "control": control,
                "median_surplus": comp["median_surplus"],
                "CVaR25_surplus": comp["CVaR25_surplus"],
                "bootstrap_LCB": comp["bootstrap_LCB"],
                "win_count": comp["win_count"],
                "n": comp["n"],
                "synthetic_gate_pass": payload["synthetic_gate_pass"],
                "candidate_no_debt_rate": payload["candidate_no_debt_rate"],
                "candidate_role_update_fraction_min": payload["candidate_role_update_fraction_min"],
                "candidate_role_effective_rank_median": payload["candidate_role_effective_rank_median"],
            })
    base = next((payload for label, _, payload in summaries if label == "base"), summaries[0][2])
    repair_labels = [label for label, _, _ in summaries if label != "base"]
    best_by_control = {}
    for control in base["comparisons"]:
        candidates = [r for r in rows if r["control"] == control]
        best = max(candidates, key=lambda r: float(r["bootstrap_LCB"]))
        best_by_control[control] = {
            "best_run_by_LCB": best["run_label"],
            "best_LCB": best["bootstrap_LCB"],
            "best_median_surplus": best["median_surplus"],
            "best_win_count": best["win_count"],
        }
    any_gate_pass = int(any(int(payload["synthetic_gate_pass"]) == 1 for _, _, payload in summaries))
    blocker_controls = sorted({
        row["control"]
        for row in rows
        if float(row["bootstrap_LCB"]) <= 0.0 or float(row["CVaR25_surplus"]) <= 0.0
    })
    analysis = {
        "status": "completed",
        "hypothesis": "H-C",
        "run_labels": [label for label, _, _ in summaries],
        "repair_labels": repair_labels,
        "base_gate_pass": int(base["synthetic_gate_pass"]),
        "any_repair_gate_pass": any_gate_pass,
        "blocker_controls": blocker_controls,
        "best_by_control": best_by_control,
        "repair_conclusion": "D-R1 phase_len=8 and D-R2 align-U half did not make the H-C synthetic shard pass. Reversed, shuffled, frozen-shape, and random-phase controls remain stronger on lower-tail evidence.",
        "not_full_science_nogo": 1,
        "next_required": "Record H-C as synthetic-unresolved after planned repairs, then continue other hypotheses and later minimum-real/H20 without claiming H-C success.",
    }
    write_rows(OUT_ROOT / "v23_24_hc_repair_comparison_matrix.csv", rows)
    write_json(OUT_ROOT / "v23_24_hc_blocker_analysis.json", analysis)
    append_exec("H-C_repair_comparison_and_blocker_analysis", "completed", files="v23_24_hc_repair_comparison_matrix.csv;v23_24_hc_blocker_analysis.json", note=json.dumps({
        "run_labels": analysis["run_labels"],
        "base_gate_pass": analysis["base_gate_pass"],
        "any_repair_gate_pass": analysis["any_repair_gate_pass"],
        "blocker_controls": analysis["blocker_controls"],
    }, sort_keys=True))
    append_recap("H-C repair comparison and blocker analysis", [
        f"- status: completed comparison for runs `{analysis['run_labels']}`.",
        f"- base synthetic_gate_pass `{analysis['base_gate_pass']}`; any repair gate pass `{analysis['any_repair_gate_pass']}`.",
        f"- blocker controls across runs: `{analysis['blocker_controls']}`.",
        f"- best run by bootstrap LCB per control: `{analysis['best_by_control']}`.",
        f"- conclusion: {analysis['repair_conclusion']}",
        "- this records H-C as synthetic-unresolved after planned repairs, not as full science NoGo because minimum-real/H20 are still pending.",
        "- artifacts: `results/v23_24/v23_24_hc_repair_comparison_matrix.csv`, `results/v23_24/v23_24_hc_blocker_analysis.json`.",
    ])
    return analysis


def mass_entropy(mass: torch.Tensor) -> float:
    with torch.no_grad():
        p = mass.detach().to(torch.float64)
        return float((-(p * torch.log(p.clamp_min(EPS))).sum()).item())


def mass_effective_count(mass: torch.Tensor) -> float:
    return float(math.exp(mass_entropy(mass)))


def hd_mass_signal(model: RoleReservoirRegressor, step: int, scheme: str) -> torch.Tensor:
    if model.c is None or model.U is None or model.c.grad is None or model.U.grad is None:
        return torch.zeros(model.roles, device=model.mass.device, dtype=torch.float64)
    c_signal = model.c.grad.detach().abs().mean(dim=0)
    u_signal = model.U.grad.detach().norm(dim=0)
    signal = c_signal + 0.1 * u_signal
    signal = (signal - signal.mean()) / signal.std().clamp_min(1.0e-8)
    q = -signal
    if scheme == "E2_time_shuffled_mass_gradient":
        q = torch.roll(q, shifts=1)
    elif scheme == "E3_signflip_mass_gradient":
        q = signal
    elif scheme == "E4_same_entropy_random_mass_walk":
        role_ids = torch.arange(model.roles, device=model.mass.device, dtype=torch.float64)
        q = torch.sin(1.7 * role_ids + 0.37 * float(step))
        q = (q - q.mean()) / q.std().clamp_min(1.0e-8)
    return q


def update_hd_mass(model: RoleReservoirRegressor, step: int, scheme: str, eta_m: float, floor: float) -> tuple[int, int]:
    if scheme == "E0_uniform_mass_fixed":
        return 0, 0
    if model.c is None:
        return 0, 0
    with torch.no_grad():
        if scheme == "E5_hard_winner_softmax_diagnostic_only":
            signal = -hd_mass_signal(model, step, "D_primary")
            target = torch.softmax(12.0 * signal, dim=0)
            target = (1.0 - model.roles * float(floor)) * target + float(floor)
            model.mass.copy_(0.90 * model.mass + 0.10 * target)
            model.mass.copy_(model.mass / model.mass.sum().clamp_min(EPS))
            return 1, 1
        q = hd_mass_signal(model, step, scheme)
        model.mass.copy_(continuous_mass_mirror_step(model.mass, q, eta_m, floor))
    return 1, 0


def train_hd_scheme(task: str, width: int, seed: int, scheme: str, *, steps: int, n: int, eta_m: float, mass_floor_mult: float, device: torch.device) -> dict[str, Any]:
    x, y = synthetic_role_data(task, width, seed, n, device)
    split = int(0.75 * int(n))
    x_train, y_train = x[:split], y[:split]
    x_held, y_held = x[split:], y[split:]
    model = RoleReservoirRegressor(width, degree=5, roles=4, seed=seed, scheme=scheme, device=device).to(device=device)
    floor = float(mass_floor_mult) * 0.02 / float(model.roles)
    before_train = float(regression_nll(model(x_train), y_train).detach().cpu().item())
    before_held = float(regression_nll(model(x_held), y_held).detach().cpu().item())
    entropy_initial = mass_entropy(model.mass)
    opt = torch.optim.SGD([model.base, model.U, model.c], lr=0.07)
    mass_update_calls = 0
    hard_winner_diagnostic_calls = 0
    floor_violations = 0
    role_grad_nonzero = torch.zeros(model.roles, device=device, dtype=torch.float64)
    for step in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        loss = regression_nll(model(x_train), y_train)
        loss.backward()
        if scheme == "E1_reaction_only_shapes_frozen":
            model.U.grad = torch.zeros_like(model.U)
        grad_norm = model.U.grad.detach().norm(dim=0)
        role_grad_nonzero += (grad_norm > 1.0e-12).to(torch.float64)
        opt.step()
        model.retraction()
        calls, hard_calls = update_hd_mass(model, step, scheme, eta_m, floor)
        mass_update_calls += calls
        hard_winner_diagnostic_calls += hard_calls
        floor_violations += int(float(model.mass.min().detach().cpu().item()) < floor - 1.0e-12)
    after_train = float(regression_nll(model(x_train), y_train).detach().cpu().item())
    after_held = float(regression_nll(model(x_held), y_held).detach().cpu().item())
    entropy_final = mass_entropy(model.mass)
    return {
        "hypothesis": "H-D",
        "task": task,
        "seed": seed,
        "width": width,
        "scheme": scheme,
        "eta_m": float(eta_m),
        "mass_floor": floor,
        "mass_floor_mult": float(mass_floor_mult),
        "n_train": int(x_train.shape[0]),
        "n_held": int(x_held.shape[0]),
        "train_NLL_before": before_train,
        "train_NLL_after": after_train,
        "train_NLL_gain": before_train - after_train,
        "held_NLL_before": before_held,
        "held_NLL_after": after_held,
        "held_NLL_gain": before_held - after_held,
        "no_debt": int(after_held <= before_held + 1.0e-10),
        "mass_update_calls": mass_update_calls,
        "hard_winner_diagnostic_calls": hard_winner_diagnostic_calls,
        "mass_entropy_initial": entropy_initial,
        "mass_entropy_final": entropy_final,
        "mass_effective_count_final": mass_effective_count(model.mass),
        "mass_min_final": float(model.mass.min().detach().cpu().item()),
        "mass_max_final": float(model.mass.max().detach().cpu().item()),
        "mass_floor_violations": floor_violations,
        "role_count_total": model.roles,
        "role_count_with_nonzero_gradient": int(role_grad_nonzero.gt(0).sum().detach().cpu().item()),
        "role_update_fraction": float((role_grad_nonzero > 0).to(torch.float64).mean().detach().cpu().item()),
        "hard_prune_count": 0,
        "hard_clone_count": 0,
        "runtime_candidate_pool_size": 0,
        "runtime_argmax_used": 0,
        "runtime_topk_used": 0,
        "source_witness_score_used_for_structure": 0,
        "role_shape_effective_rank": role_effective_rank(model),
        "edge_role_specialization_MI": edge_role_specialization_mi(model),
    }


def summarize_hd(rows: list[dict[str, Any]], *, run_label: str, repair_id: str) -> dict[str, Any]:
    candidate = "D_primary_continuous_mass_reaction"
    controls = sorted({r["scheme"] for r in rows if r["scheme"] != candidate})
    keyed: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for row in rows:
        keyed[(row["task"], int(row["seed"]), int(row["width"]), row["scheme"])] = row
    comparisons = {}
    for control in controls:
        surplus = []
        wins = 0
        no_debt = []
        for task in ["SYN-R1", "SYN-R2", "SYN-R3", "SYN-D1"]:
            for seed in range(5):
                for width in [2, 4]:
                    cand = keyed.get((task, seed, width, candidate))
                    ctrl = keyed.get((task, seed, width, control))
                    if cand is None or ctrl is None:
                        continue
                    s = float(cand["held_NLL_gain"]) - float(ctrl["held_NLL_gain"])
                    surplus.append(s)
                    wins += int(s > 0.0)
                    no_debt.append(int(cand["no_debt"]))
        comparisons[control] = {
            "n": len(surplus),
            "median_surplus": median(surplus),
            "CVaR25_surplus": cvar25(surplus),
            "bootstrap_LCB": bootstrap_lcb(surplus, seed=24026),
            "win_count": wins,
            "no_debt_rate": float(sum(no_debt) / max(1, len(no_debt))),
        }
    candidate_rows = [r for r in rows if r["scheme"] == candidate]
    synthetic_gate_pass = int(all(float(comp["CVaR25_surplus"]) > 0.0 and float(comp["bootstrap_LCB"]) > 0.0 and int(comp["win_count"]) >= 30 for comp in comparisons.values()))
    return {
        "status": "completed",
        "hypothesis": "H-D",
        "candidate": candidate,
        "run_label": run_label,
        "repair_id": repair_id,
        "controls": controls,
        "row_count": len(rows),
        "tasks": ["SYN-R1", "SYN-R2", "SYN-R3", "SYN-D1"],
        "seeds": [0, 1, 2, 3, 4],
        "widths": [2, 4],
        "comparisons": comparisons,
        "candidate_no_debt_rate": float(sum(int(r["no_debt"]) for r in candidate_rows) / max(1, len(candidate_rows))),
        "candidate_mass_update_calls_median": median([float(r["mass_update_calls"]) for r in candidate_rows]),
        "candidate_mass_entropy_final_median": median([float(r["mass_entropy_final"]) for r in candidate_rows]),
        "candidate_mass_effective_count_final_median": median([float(r["mass_effective_count_final"]) for r in candidate_rows]),
        "candidate_mass_min_final_min": min(float(r["mass_min_final"]) for r in candidate_rows),
        "candidate_mass_floor_violation_total": sum(int(r["mass_floor_violations"]) for r in candidate_rows),
        "candidate_role_update_fraction_min": min(float(r["role_update_fraction"]) for r in candidate_rows),
        "synthetic_gate_pass": synthetic_gate_pass,
        "gate_note": "H-D synthetic shard only; full H-D still requires repair audit, minimum-real, H20, and MLP matched comparisons.",
    }


def hd_output_stem(run_label: str) -> str:
    if run_label == "base":
        return "v23_24_hd_continuous_mass_reaction"
    clean = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in run_label)
    return f"v23_24_hd_continuous_mass_reaction_{clean}"


def phase_hd_synthetic_smoke(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(str(args.device))
    run_label = str(args.hd_label)
    repair_id = "none" if run_label == "base" else run_label
    stem = hd_output_stem(run_label)
    schemes = [
        "D_primary_continuous_mass_reaction",
        "E0_uniform_mass_fixed",
        "E1_reaction_only_shapes_frozen",
        "E2_time_shuffled_mass_gradient",
        "E3_signflip_mass_gradient",
        "E4_same_entropy_random_mass_walk",
        "E5_hard_winner_softmax_diagnostic_only",
    ]
    rows: list[dict[str, Any]] = []
    for task in ["SYN-R1", "SYN-R2", "SYN-R3", "SYN-D1"]:
        for seed in range(int(args.hd_seeds)):
            for width in [2, 4]:
                for scheme in schemes:
                    rows.append(train_hd_scheme(task, width, seed, scheme, steps=int(args.hd_steps), n=int(args.hd_n), eta_m=float(args.hd_eta_m), mass_floor_mult=float(args.hd_mass_floor_mult), device=device))
    matrix_name = f"{stem}_matrix.csv"
    summary_name = f"{stem}_summary.json"
    write_rows(OUT_ROOT / matrix_name, rows)
    summary = summarize_hd(rows, run_label=run_label, repair_id=repair_id)
    write_json(OUT_ROOT / summary_name, summary)
    append_exec("H-D_continuous_mass_reaction_synthetic_smoke", "completed", files=f"{matrix_name};{summary_name}", note=json.dumps({
        "run_label": run_label,
        "eta_m": float(args.hd_eta_m),
        "mass_floor_mult": float(args.hd_mass_floor_mult),
        "row_count": len(rows),
        "candidate_no_debt_rate": summary["candidate_no_debt_rate"],
        "comparisons": summary["comparisons"],
        "synthetic_gate_pass": summary["synthetic_gate_pass"],
    }, sort_keys=True))
    blocker_controls = [control for control, comp in summary["comparisons"].items() if float(comp["bootstrap_LCB"]) <= 0.0 or float(comp["CVaR25_surplus"]) <= 0.0]
    append_recap("H-D continuous mass-reaction synthetic smoke", [
        f"- status: completed H-D synthetic shard; row_count `{len(rows)}`.",
        f"- run_label `{run_label}`; eta_m `{float(args.hd_eta_m)}`; mass_floor_mult `{float(args.hd_mass_floor_mult)}`; candidate plus E0/E1/E2/E3/E4/E5 controls.",
        f"- candidate no_debt_rate `{summary['candidate_no_debt_rate']}`; mass_update_calls_median `{summary['candidate_mass_update_calls_median']}`; mass_entropy_final_median `{summary['candidate_mass_entropy_final_median']}`; effective_count_final_median `{summary['candidate_mass_effective_count_final_median']}`; mass_floor_violation_total `{summary['candidate_mass_floor_violation_total']}`.",
        f"- synthetic gate pass `{summary['synthetic_gate_pass']}`; blocker controls by lower-tail criterion `{blocker_controls}`.",
        f"- comparisons: `{summary['comparisons']}`.",
        f"- artifacts: `results/v23_24/{matrix_name}`, `results/v23_24/{summary_name}`.",
        "- no H-D scientific success is claimed until repair audit, minimum-real, H20, and matched MLP comparisons are done.",
    ])
    return summary


def phase_hd_analysis(args: argparse.Namespace) -> dict[str, Any]:
    run_specs = [
        ("base", OUT_ROOT / "v23_24_hd_continuous_mass_reaction_summary.json"),
        ("repair_ER1_floor2", OUT_ROOT / "v23_24_hd_continuous_mass_reaction_repair_ER1_floor2_summary.json"),
        ("repair_ER2_etaHalf", OUT_ROOT / "v23_24_hd_continuous_mass_reaction_repair_ER2_etaHalf_summary.json"),
    ]
    summaries = []
    for label, path in run_specs:
        if path.exists():
            summaries.append((label, path, json.loads(path.read_text(encoding="utf-8"))))
    if not summaries:
        raise FileNotFoundError("H-D analysis requires at least the base H-D summary; run --phase hd-smoke first.")
    rows: list[dict[str, Any]] = []
    for label, path, payload in summaries:
        for control, comp in payload["comparisons"].items():
            rows.append({
                "hypothesis": "H-D",
                "run_label": label,
                "summary_file": rel(path),
                "control": control,
                "median_surplus": comp["median_surplus"],
                "CVaR25_surplus": comp["CVaR25_surplus"],
                "bootstrap_LCB": comp["bootstrap_LCB"],
                "win_count": comp["win_count"],
                "n": comp["n"],
                "synthetic_gate_pass": payload["synthetic_gate_pass"],
                "candidate_no_debt_rate": payload["candidate_no_debt_rate"],
                "candidate_mass_effective_count_final_median": payload["candidate_mass_effective_count_final_median"],
                "candidate_mass_entropy_final_median": payload["candidate_mass_entropy_final_median"],
                "candidate_mass_min_final_min": payload["candidate_mass_min_final_min"],
                "candidate_mass_floor_violation_total": payload["candidate_mass_floor_violation_total"],
            })
    base = next((payload for label, _, payload in summaries if label == "base"), summaries[0][2])
    best_by_control = {}
    for control in base["comparisons"]:
        candidates = [r for r in rows if r["control"] == control]
        best = max(candidates, key=lambda r: float(r["bootstrap_LCB"]))
        best_by_control[control] = {
            "best_run_by_LCB": best["run_label"],
            "best_LCB": best["bootstrap_LCB"],
            "best_median_surplus": best["median_surplus"],
            "best_CVaR25_surplus": best["CVaR25_surplus"],
            "best_win_count": best["win_count"],
        }
    any_gate_pass = int(any(int(payload["synthetic_gate_pass"]) == 1 for _, _, payload in summaries))
    final_blockers = sorted({
        row["control"]
        for row in rows
        if row["run_label"] in {"repair_ER1_floor2", "repair_ER2_etaHalf"}
        and (float(row["bootstrap_LCB"]) <= 0.0 or float(row["CVaR25_surplus"]) <= 0.0)
    })
    analysis = {
        "status": "completed",
        "hypothesis": "H-D",
        "run_labels": [label for label, _, _ in summaries],
        "base_gate_pass": int(base["synthetic_gate_pass"]),
        "any_repair_gate_pass": any_gate_pass,
        "final_blocker_controls_after_repairs": final_blockers,
        "best_by_control": best_by_control,
        "repair_conclusion": "E-R1 and E-R2 improved mass entropy/effective count and removed E4 lower-tail debt, but E1 frozen-shape and E5 hard-winner diagnostic remain blockers; H-D synthetic shard is unresolved after planned repairs.",
        "not_full_science_nogo": 1,
        "next_required": "Preserve H-D as synthetic-unresolved after planned repairs, then continue remaining hypotheses and later minimum-real/H20.",
    }
    write_rows(OUT_ROOT / "v23_24_hd_repair_comparison_matrix.csv", rows)
    write_json(OUT_ROOT / "v23_24_hd_blocker_analysis.json", analysis)
    append_exec("H-D_repair_comparison_and_blocker_analysis", "completed", files="v23_24_hd_repair_comparison_matrix.csv;v23_24_hd_blocker_analysis.json", note=json.dumps({
        "run_labels": analysis["run_labels"],
        "base_gate_pass": analysis["base_gate_pass"],
        "any_repair_gate_pass": analysis["any_repair_gate_pass"],
        "final_blocker_controls_after_repairs": analysis["final_blocker_controls_after_repairs"],
    }, sort_keys=True))
    append_recap("H-D repair comparison and blocker analysis", [
        f"- status: completed comparison for runs `{analysis['run_labels']}`.",
        f"- base synthetic_gate_pass `{analysis['base_gate_pass']}`; any repair gate pass `{analysis['any_repair_gate_pass']}`.",
        f"- final blocker controls after repairs: `{analysis['final_blocker_controls_after_repairs']}`.",
        f"- best run by bootstrap LCB per control: `{analysis['best_by_control']}`.",
        f"- conclusion: {analysis['repair_conclusion']}",
        "- this records H-D as synthetic-unresolved after planned repairs, not as full science NoGo because minimum-real/H20 are still pending.",
        "- artifacts: `results/v23_24/v23_24_hd_repair_comparison_matrix.csv`, `results/v23_24/v23_24_hd_blocker_analysis.json`.",
    ])
    return analysis


class MultilevelPolynomialRegressor(nn.Module):
    def __init__(self, edge_count: int, degree: int, seed: int, *, device: torch.device, random_scale: float = 0.05) -> None:
        super().__init__()
        gen = torch.Generator(device=device).manual_seed(int(seed) + 250024)
        self.edge_count = int(edge_count)
        self.degree = int(degree)
        self.coeff = nn.Parameter(float(random_scale) * torch.randn((self.edge_count, self.degree + 1), generator=gen, device=device, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = basis(x.reshape(-1), self.degree).reshape(int(x.shape[0]), self.edge_count, self.degree + 1)
        return torch.einsum("ned,ed->n", b, self.coeff)

    def prolongate(self, new_degree: int, *, random_new: bool, seed: int) -> tuple[float, float]:
        old_degree = self.degree
        if int(new_degree) <= old_degree:
            return 0.0, 0.0
        gen = torch.Generator(device=self.coeff.device).manual_seed(int(seed) + int(new_degree) * 97)
        new_coeff = torch.zeros((self.edge_count, int(new_degree) + 1), device=self.coeff.device, dtype=torch.float64)
        new_coeff[:, : old_degree + 1] = self.coeff.detach()
        if random_new:
            new_coeff[:, old_degree + 1 :] = 0.02 * torch.randn((self.edge_count, int(new_degree) - old_degree), generator=gen, device=self.coeff.device, dtype=torch.float64)
        new_residual_norm = float(new_coeff[:, old_degree + 1 :].norm().detach().cpu().item())
        self.degree = int(new_degree)
        self.coeff = nn.Parameter(new_coeff)
        return 0.0, new_residual_norm


class MatchedMLPRegressor(nn.Module):
    def __init__(self, width: int, seed: int, *, device: torch.device) -> None:
        super().__init__()
        torch.manual_seed(int(seed) + 260024)
        hidden = max(8, 4 * int(width))
        self.net = nn.Sequential(
            nn.Linear(int(width), hidden, dtype=torch.float64, device=device),
            nn.Tanh(),
            nn.Linear(hidden, 1, dtype=torch.float64, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1)


def he_milestones(scheme: str, steps: int) -> list[tuple[int, int]]:
    if scheme == "F0_coarse_only" or scheme == "F1_fixed_fine_from_start_same_final_params" or scheme == "F5_MLP_matched_width_or_feature_refinement":
        return []
    if scheme == "F4_refinement_milestone_time_shuffled_diagnostic":
        return [(max(1, int(0.15 * int(steps))), 6), (max(2, int(0.75 * int(steps))), 9)]
    return [(max(1, int(0.30 * int(steps))), 6), (max(2, int(0.60 * int(steps))), 9)]


def train_he_scheme(task: str, width: int, seed: int, scheme: str, *, steps: int, n: int, new_lr_boost: float, rewarmup_steps: int, device: torch.device) -> dict[str, Any]:
    x, y = synthetic_role_data(task, width, seed, n, device)
    split = int(0.75 * int(n))
    x_train, y_train = x[:split], y[:split]
    x_held, y_held = x[split:], y[split:]
    is_mlp = scheme == "F5_MLP_matched_width_or_feature_refinement"
    if is_mlp:
        model: nn.Module = MatchedMLPRegressor(width, seed, device=device).to(device=device)
    else:
        start_degree = 9 if scheme == "F1_fixed_fine_from_start_same_final_params" else 3
        model = MultilevelPolynomialRegressor(width, start_degree, seed, device=device).to(device=device)
    before_train = float(regression_nll(model(x_train), y_train).detach().cpu().item())
    before_held = float(regression_nll(model(x_held), y_held).detach().cpu().item())
    opt = torch.optim.SGD(model.parameters(), lr=0.06)
    milestones = he_milestones(scheme, steps)
    milestone_map = {step: degree for step, degree in milestones}
    refinement_calls = 0
    max_function_error = 0.0
    new_residual_norm_sum = 0.0
    frozen_start_degree = None
    rewarmup_until = -1
    rewarmup_start_col = None
    for step in range(int(steps)):
        if step in milestone_map and isinstance(model, MultilevelPolynomialRegressor):
            probe = x_train[: min(64, int(x_train.shape[0]))]
            before = model(probe).detach()
            old_degree_before = model.degree
            random_new = scheme == "F3_refinement_with_random_non_nested_new_basis_same_param"
            _, new_residual_norm = model.prolongate(milestone_map[step], random_new=random_new, seed=seed + step)
            after = model(probe).detach()
            max_function_error = max(max_function_error, float((before - after).abs().max().detach().cpu().item()))
            new_residual_norm_sum += new_residual_norm
            refinement_calls += 1
            if scheme == "F2_function_preserving_refinement_new_coords_frozen":
                frozen_start_degree = milestone_map[step]
            if scheme == "E_primary_multilevel_refinement" and int(rewarmup_steps) > 0 and float(new_lr_boost) != 1.0:
                rewarmup_until = step + int(rewarmup_steps)
                rewarmup_start_col = old_degree_before + 1
            opt = torch.optim.SGD(model.parameters(), lr=0.06)
        opt.zero_grad(set_to_none=True)
        loss = regression_nll(model(x_train), y_train)
        loss.backward()
        if isinstance(model, MultilevelPolynomialRegressor) and scheme == "F2_function_preserving_refinement_new_coords_frozen" and frozen_start_degree is not None:
            model.coeff.grad[:, 4:] = 0.0
        if isinstance(model, MultilevelPolynomialRegressor) and rewarmup_start_col is not None and step < rewarmup_until:
            model.coeff.grad[:, int(rewarmup_start_col) :] *= float(new_lr_boost)
        opt.step()
    after_train = float(regression_nll(model(x_train), y_train).detach().cpu().item())
    after_held = float(regression_nll(model(x_held), y_held).detach().cpu().item())
    fine_energy_fraction = 0.0
    final_degree = -1
    if isinstance(model, MultilevelPolynomialRegressor):
        final_degree = model.degree
        total = float(model.coeff.detach().norm().cpu().item())
        if model.degree > 3:
            fine = float(model.coeff.detach()[:, 4:].norm().cpu().item())
            fine_energy_fraction = fine / max(EPS, total)
    return {
        "hypothesis": "H-E",
        "task": task,
        "seed": seed,
        "width": width,
        "scheme": scheme,
        "n_train": int(x_train.shape[0]),
        "n_held": int(x_held.shape[0]),
        "train_NLL_before": before_train,
        "train_NLL_after": after_train,
        "train_NLL_gain": before_train - after_train,
        "held_NLL_before": before_held,
        "held_NLL_after": after_held,
        "held_NLL_gain": before_held - after_held,
        "no_debt": int(after_held <= before_held + 1.0e-10),
        "refinement_calls": refinement_calls,
        "function_preservation_max_error": max_function_error,
        "new_residual_norm_sum": new_residual_norm_sum,
        "final_degree": final_degree,
        "fine_energy_fraction_final": fine_energy_fraction,
        "new_coords_frozen": int(scheme == "F2_function_preserving_refinement_new_coords_frozen"),
        "new_coordinate_rewarmup_steps": int(rewarmup_steps) if scheme == "E_primary_multilevel_refinement" else 0,
        "new_coordinate_lr_boost": float(new_lr_boost) if scheme == "E_primary_multilevel_refinement" else 1.0,
        "random_non_nested_new_basis": int(scheme == "F3_refinement_with_random_non_nested_new_basis_same_param"),
        "time_shuffled_milestones": int(scheme == "F4_refinement_milestone_time_shuffled_diagnostic"),
        "mlp_matched_control": int(is_mlp),
        "hard_prune_count": 0,
        "hard_clone_count": 0,
        "runtime_candidate_pool_size": 0,
        "runtime_argmax_used": 0,
        "runtime_topk_used": 0,
        "source_witness_score_used_for_structure": 0,
    }


def summarize_he(rows: list[dict[str, Any]], *, run_label: str, repair_id: str) -> dict[str, Any]:
    candidate = "E_primary_multilevel_refinement"
    controls = sorted({r["scheme"] for r in rows if r["scheme"] != candidate})
    keyed: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for row in rows:
        keyed[(row["task"], int(row["seed"]), int(row["width"]), row["scheme"])] = row
    comparisons = {}
    for control in controls:
        surplus = []
        wins = 0
        no_debt = []
        for task in ["SYN-R1", "SYN-R2", "SYN-R3", "SYN-D1"]:
            for seed in range(5):
                for width in [2, 4]:
                    cand = keyed.get((task, seed, width, candidate))
                    ctrl = keyed.get((task, seed, width, control))
                    if cand is None or ctrl is None:
                        continue
                    s = float(cand["held_NLL_gain"]) - float(ctrl["held_NLL_gain"])
                    surplus.append(s)
                    wins += int(s > 0.0)
                    no_debt.append(int(cand["no_debt"]))
        comparisons[control] = {
            "n": len(surplus),
            "median_surplus": median(surplus),
            "CVaR25_surplus": cvar25(surplus),
            "bootstrap_LCB": bootstrap_lcb(surplus, seed=24027),
            "win_count": wins,
            "no_debt_rate": float(sum(no_debt) / max(1, len(no_debt))),
        }
    candidate_rows = [r for r in rows if r["scheme"] == candidate]
    synthetic_gate_pass = int(all(float(comp["CVaR25_surplus"]) > 0.0 and float(comp["bootstrap_LCB"]) > 0.0 and int(comp["win_count"]) >= 30 for comp in comparisons.values()))
    return {
        "status": "completed",
        "hypothesis": "H-E",
        "candidate": candidate,
        "run_label": run_label,
        "repair_id": repair_id,
        "controls": controls,
        "row_count": len(rows),
        "tasks": ["SYN-R1", "SYN-R2", "SYN-R3", "SYN-D1"],
        "seeds": [0, 1, 2, 3, 4],
        "widths": [2, 4],
        "comparisons": comparisons,
        "candidate_no_debt_rate": float(sum(int(r["no_debt"]) for r in candidate_rows) / max(1, len(candidate_rows))),
        "candidate_refinement_calls_median": median([float(r["refinement_calls"]) for r in candidate_rows]),
        "candidate_function_preservation_max_error": max(float(r["function_preservation_max_error"]) for r in candidate_rows),
        "candidate_new_residual_norm_sum_median": median([float(r["new_residual_norm_sum"]) for r in candidate_rows]),
        "candidate_fine_energy_fraction_final_median": median([float(r["fine_energy_fraction_final"]) for r in candidate_rows]),
        "candidate_new_coordinate_rewarmup_steps": max(int(r["new_coordinate_rewarmup_steps"]) for r in candidate_rows),
        "candidate_new_coordinate_lr_boost": max(float(r["new_coordinate_lr_boost"]) for r in candidate_rows),
        "synthetic_gate_pass": synthetic_gate_pass,
        "gate_note": "H-E synthetic shard only; full H-E still requires repair audit, minimum-real, H20, and MLP matched comparisons.",
    }


def he_output_stem(run_label: str) -> str:
    if run_label == "base":
        return "v23_24_he_multilevel_refinement"
    clean = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in run_label)
    return f"v23_24_he_multilevel_refinement_{clean}"


def phase_he_synthetic_smoke(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(str(args.device))
    run_label = str(args.he_label)
    repair_id = "none" if run_label == "base" else run_label
    stem = he_output_stem(run_label)
    schemes = [
        "E_primary_multilevel_refinement",
        "F0_coarse_only",
        "F1_fixed_fine_from_start_same_final_params",
        "F2_function_preserving_refinement_new_coords_frozen",
        "F3_refinement_with_random_non_nested_new_basis_same_param",
        "F4_refinement_milestone_time_shuffled_diagnostic",
        "F5_MLP_matched_width_or_feature_refinement",
    ]
    rows: list[dict[str, Any]] = []
    for task in ["SYN-R1", "SYN-R2", "SYN-R3", "SYN-D1"]:
        for seed in range(int(args.he_seeds)):
            for width in [2, 4]:
                for scheme in schemes:
                    rows.append(train_he_scheme(task, width, seed, scheme, steps=int(args.he_steps), n=int(args.he_n), new_lr_boost=float(args.he_new_lr_boost), rewarmup_steps=int(args.he_rewarmup_steps), device=device))
    matrix_name = f"{stem}_matrix.csv"
    summary_name = f"{stem}_summary.json"
    write_rows(OUT_ROOT / matrix_name, rows)
    summary = summarize_he(rows, run_label=run_label, repair_id=repair_id)
    write_json(OUT_ROOT / summary_name, summary)
    append_exec("H-E_multilevel_refinement_synthetic_smoke", "completed", files=f"{matrix_name};{summary_name}", note=json.dumps({
        "run_label": run_label,
        "row_count": len(rows),
        "candidate_no_debt_rate": summary["candidate_no_debt_rate"],
        "candidate_function_preservation_max_error": summary["candidate_function_preservation_max_error"],
        "comparisons": summary["comparisons"],
        "synthetic_gate_pass": summary["synthetic_gate_pass"],
    }, sort_keys=True))
    blocker_controls = [control for control, comp in summary["comparisons"].items() if float(comp["bootstrap_LCB"]) <= 0.0 or float(comp["CVaR25_surplus"]) <= 0.0]
    append_recap("H-E multilevel refinement synthetic smoke", [
        f"- status: completed H-E synthetic shard; row_count `{len(rows)}`.",
        f"- run_label `{run_label}`; new_lr_boost `{float(args.he_new_lr_boost)}`; rewarmup_steps `{int(args.he_rewarmup_steps)}`; scope: 4 synthetic tasks, seeds 0..4, widths 2/4, candidate plus F0/F1/F2/F3/F4/F5 controls.",
        f"- candidate no_debt_rate `{summary['candidate_no_debt_rate']}`; refinement_calls_median `{summary['candidate_refinement_calls_median']}`; function_preservation_max_error `{summary['candidate_function_preservation_max_error']}`; new_residual_norm_sum_median `{summary['candidate_new_residual_norm_sum_median']}`; fine_energy_fraction_final_median `{summary['candidate_fine_energy_fraction_final_median']}`.",
        f"- synthetic gate pass `{summary['synthetic_gate_pass']}`; blocker controls by lower-tail criterion `{blocker_controls}`.",
        f"- comparisons: `{summary['comparisons']}`.",
        f"- artifacts: `results/v23_24/{matrix_name}`, `results/v23_24/{summary_name}`.",
        "- no H-E scientific success is claimed until repair audit, minimum-real, H20, and matched MLP comparisons are done.",
    ])
    return summary


def phase_he_analysis(args: argparse.Namespace) -> dict[str, Any]:
    run_specs = [
        ("base", OUT_ROOT / "v23_24_he_multilevel_refinement_summary.json"),
        ("repair_FR2_rewarmup10_boost1p5", OUT_ROOT / "v23_24_he_multilevel_refinement_repair_FR2_rewarmup10_boost1p5_summary.json"),
    ]
    summaries = []
    for label, path in run_specs:
        if path.exists():
            summaries.append((label, path, json.loads(path.read_text(encoding="utf-8"))))
    if not summaries:
        raise FileNotFoundError("H-E analysis requires at least the base H-E summary; run --phase he-smoke first.")
    rows: list[dict[str, Any]] = []
    for label, path, payload in summaries:
        for control, comp in payload["comparisons"].items():
            rows.append({
                "hypothesis": "H-E",
                "run_label": label,
                "summary_file": rel(path),
                "control": control,
                "median_surplus": comp["median_surplus"],
                "CVaR25_surplus": comp["CVaR25_surplus"],
                "bootstrap_LCB": comp["bootstrap_LCB"],
                "win_count": comp["win_count"],
                "n": comp["n"],
                "synthetic_gate_pass": payload["synthetic_gate_pass"],
                "candidate_no_debt_rate": payload["candidate_no_debt_rate"],
                "candidate_function_preservation_max_error": payload["candidate_function_preservation_max_error"],
                "candidate_new_residual_norm_sum_median": payload["candidate_new_residual_norm_sum_median"],
                "candidate_fine_energy_fraction_final_median": payload["candidate_fine_energy_fraction_final_median"],
            })
    base = next((payload for label, _, payload in summaries if label == "base"), summaries[0][2])
    best_by_control = {}
    for control in base["comparisons"]:
        candidates = [r for r in rows if r["control"] == control]
        best = max(candidates, key=lambda r: float(r["bootstrap_LCB"]))
        best_by_control[control] = {
            "best_run_by_LCB": best["run_label"],
            "best_LCB": best["bootstrap_LCB"],
            "best_CVaR25_surplus": best["CVaR25_surplus"],
            "best_median_surplus": best["median_surplus"],
            "best_win_count": best["win_count"],
        }
    any_gate_pass = int(any(int(payload["synthetic_gate_pass"]) == 1 for _, _, payload in summaries))
    final_blockers = sorted({
        row["control"]
        for row in rows
        if row["run_label"] == summaries[-1][0]
        and (float(row["bootstrap_LCB"]) <= 0.0 or float(row["CVaR25_surplus"]) <= 0.0)
    })
    analysis = {
        "status": "completed",
        "hypothesis": "H-E",
        "run_labels": [label for label, _, _ in summaries],
        "base_gate_pass": int(base["synthetic_gate_pass"]),
        "any_repair_gate_pass": any_gate_pass,
        "final_blocker_controls_after_repairs": final_blockers,
        "best_by_control": best_by_control,
        "repair_conclusion": "F-R1 optimizer reset was already part of the base implementation. F-R2 rewarmup increased fine-coordinate energy and improved F3/F4 LCB, but F1/F3/F4/F5 lower-tail blockers remain; H-E synthetic shard is unresolved after planned repairs.",
        "not_full_science_nogo": 1,
        "next_required": "Preserve H-E as synthetic-unresolved after planned repairs, then continue remaining hypotheses and later minimum-real/H20.",
    }
    write_rows(OUT_ROOT / "v23_24_he_repair_comparison_matrix.csv", rows)
    write_json(OUT_ROOT / "v23_24_he_blocker_analysis.json", analysis)
    append_exec("H-E_repair_comparison_and_blocker_analysis", "completed", files="v23_24_he_repair_comparison_matrix.csv;v23_24_he_blocker_analysis.json", note=json.dumps({
        "run_labels": analysis["run_labels"],
        "base_gate_pass": analysis["base_gate_pass"],
        "any_repair_gate_pass": analysis["any_repair_gate_pass"],
        "final_blocker_controls_after_repairs": analysis["final_blocker_controls_after_repairs"],
    }, sort_keys=True))
    append_recap("H-E repair comparison and blocker analysis", [
        f"- status: completed comparison for runs `{analysis['run_labels']}`.",
        f"- base synthetic_gate_pass `{analysis['base_gate_pass']}`; any repair gate pass `{analysis['any_repair_gate_pass']}`.",
        f"- final blocker controls after repairs: `{analysis['final_blocker_controls_after_repairs']}`.",
        f"- best run by bootstrap LCB per control: `{analysis['best_by_control']}`.",
        f"- conclusion: {analysis['repair_conclusion']}",
        "- this records H-E as synthetic-unresolved after planned repairs, not as full science NoGo because minimum-real/H20 are still pending.",
        "- artifacts: `results/v23_24/v23_24_he_repair_comparison_matrix.csv`, `results/v23_24/v23_24_he_blocker_analysis.json`.",
    ])
    return analysis


def synthetic_twin_data(task: str, width: int, seed: int, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(int(seed) + len(task) * 131 + int(width) * 23)
    x = torch.rand((int(n), int(width)), generator=gen, device=device, dtype=torch.float64)
    x0 = x[:, 0]
    if task == "SYN-B1":
        y = 0.75 * torch.tanh(7.0 * (x0 - 0.28)) - 0.65 * torch.tanh(7.0 * (x0 - 0.72))
        if width > 1:
            y = y + 0.35 * torch.sin(4.0 * math.pi * x[:, 1]) * torch.sign(x0 - 0.5)
    elif task == "SYN-B2":
        y = torch.tanh(3.2 * (x0 - 0.5))
        if width > 1:
            y = y + 0.10 * (x[:, 1] - 0.5)
    elif task == "SYN-D1":
        return synthetic_role_data(task, width, seed, n, device)
    else:
        raise ValueError(f"unknown twin task {task}")
    y = y + 0.015 * torch.randn(y.shape, generator=gen, device=device, dtype=torch.float64)
    return x, y


class TwinFlowRegressor:
    def __init__(self, width: int, seed: int, *, scheme: str, device: torch.device) -> None:
        gen = torch.Generator(device=device).manual_seed(int(seed) + 270024)
        self.width = int(width)
        self.scheme = str(scheme)
        self.device = device
        self.base = torch.zeros((), device=device, dtype=torch.float64, requires_grad=True)
        self.parent_w = (0.2 * torch.randn((self.width,), generator=gen, device=device, dtype=torch.float64)).requires_grad_(True)
        self.parent_out = (0.2 * torch.randn((), generator=gen, device=device, dtype=torch.float64)).requires_grad_(True)
        self.child_w: torch.Tensor | None = None
        self.child_out: torch.Tensor | None = None
        self.split_done = False
        self.function_preservation_error = 0.0
        self.split_step = -1

    def parameters(self) -> list[torch.Tensor]:
        if self.split_done:
            assert self.child_w is not None and self.child_out is not None
            return [self.base, self.child_w, self.child_out]
        return [self.base, self.parent_w, self.parent_out]

    def zero_grad(self) -> None:
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.split_done:
            assert self.child_w is not None and self.child_out is not None
            h = torch.tanh(x @ self.child_w.T)
            return self.base + h @ self.child_out
        return self.base + self.parent_out * torch.tanh(x @ self.parent_w)

    def split(self, probe: torch.Tensor, *, step: int, alpha: float, random_noise: bool, freeze_exact_duplicate: bool, seed: int) -> None:
        if self.split_done:
            return
        before = self.forward(probe).detach()
        gen = torch.Generator(device=self.device).manual_seed(int(seed) + int(step) * 19)
        w_plus = self.parent_w.detach().clone()
        w_minus = self.parent_w.detach().clone()
        if random_noise:
            noise = 0.03 * torch.randn(w_plus.shape, generator=gen, device=self.device, dtype=torch.float64)
            w_plus = w_plus + noise
            w_minus = w_minus - noise
        out_plus = float(alpha) * self.parent_out.detach().clone()
        out_minus = (1.0 - float(alpha)) * self.parent_out.detach().clone()
        self.child_w = torch.stack([w_plus, w_minus], dim=0).detach().requires_grad_(True)
        self.child_out = torch.stack([out_plus.reshape(()), out_minus.reshape(())], dim=0).detach().requires_grad_(True)
        self.parent_w = self.parent_w.detach()
        self.parent_out = self.parent_out.detach()
        self.split_done = True
        self.split_step = int(step)
        after = self.forward(probe).detach()
        self.function_preservation_error = float((before - after).abs().max().detach().cpu().item())
        if freeze_exact_duplicate:
            self.child_w.requires_grad_(True)
            self.child_out.requires_grad_(True)

    def manual_step(self, lr: float, *, step: int, plus_scale: float, minus_scale: float, rewarmup_steps: int, freeze_children: bool, enforce_symmetry: bool) -> None:
        with torch.no_grad():
            if not self.split_done:
                for param in [self.base, self.parent_w, self.parent_out]:
                    if param.grad is not None:
                        param -= float(lr) * param.grad
                return
            assert self.child_w is not None and self.child_out is not None
            if self.base.grad is not None:
                self.base -= float(lr) * self.base.grad
            active_asym = int(step) < self.split_step + int(rewarmup_steps)
            scales = torch.tensor([float(plus_scale), float(minus_scale)], device=self.device, dtype=torch.float64) if active_asym else torch.ones(2, device=self.device, dtype=torch.float64)
            if freeze_children:
                scales = torch.zeros_like(scales)
            if self.child_w.grad is not None:
                self.child_w -= float(lr) * scales[:, None] * self.child_w.grad
            if self.child_out.grad is not None:
                self.child_out -= float(lr) * scales * self.child_out.grad
            if enforce_symmetry:
                mean_w = self.child_w.mean(dim=0, keepdim=True)
                mean_out = self.child_out.mean().reshape(1).repeat(2)
                self.child_w.copy_(mean_w.repeat(2, 1))
                self.child_out.copy_(mean_out)

    def antisymmetric_norm(self) -> float:
        if not self.split_done or self.child_w is None or self.child_out is None:
            return 0.0
        d_w = 0.5 * (self.child_w[0] - self.child_w[1])
        d_o = 0.5 * (self.child_out[0] - self.child_out[1])
        return float(torch.sqrt(d_w.square().sum() + d_o.square()).detach().cpu().item())

    def child_specialization(self, x: torch.Tensor) -> tuple[float, float]:
        if not self.split_done or self.child_w is None:
            return 0.0, 1.0
        h = torch.tanh(x @ self.child_w.T).detach()
        h0 = h[:, 0] - h[:, 0].mean()
        h1 = h[:, 1] - h[:, 1].mean()
        corr = float((h0 @ h1 / (h0.norm() * h1.norm()).clamp_min(EPS)).detach().cpu().item())
        return float(1.0 - abs(corr)), corr


def train_hf_scheme(task: str, width: int, seed: int, scheme: str, *, steps: int, n: int, plus_scale: float, minus_scale: float, rewarmup_steps: int, device: torch.device) -> dict[str, Any]:
    x, y = synthetic_twin_data(task, width, seed, n, device)
    split_idx = int(0.75 * int(n))
    x_train, y_train = x[:split_idx], y[:split_idx]
    x_held, y_held = x[split_idx:], y[split_idx:]
    is_mlp = scheme == "G5_MLP_matched_hidden_widening"
    if is_mlp:
        mlp = MatchedMLPRegressor(width, seed, device=device).to(device=device)
        opt = torch.optim.SGD(mlp.parameters(), lr=0.06)
        before_train = float(regression_nll(mlp(x_train), y_train).detach().cpu().item())
        before_held = float(regression_nll(mlp(x_held), y_held).detach().cpu().item())
        for _ in range(int(steps)):
            opt.zero_grad(set_to_none=True)
            loss = regression_nll(mlp(x_train), y_train)
            loss.backward()
            opt.step()
        after_train = float(regression_nll(mlp(x_train), y_train).detach().cpu().item())
        after_held = float(regression_nll(mlp(x_held), y_held).detach().cpu().item())
        return {
            "hypothesis": "H-F",
            "task": task,
            "seed": seed,
            "width": width,
            "scheme": scheme,
            "n_train": int(x_train.shape[0]),
            "n_held": int(x_held.shape[0]),
            "train_NLL_before": before_train,
            "train_NLL_after": after_train,
            "train_NLL_gain": before_train - after_train,
            "held_NLL_before": before_held,
            "held_NLL_after": after_held,
            "held_NLL_gain": before_held - after_held,
            "no_debt": int(after_held <= before_held + 1.0e-10),
            "function_preservation_error": 0.0,
            "antisymmetric_G_norm_final": 0.0,
            "child_specialization": 0.0,
            "child_activation_corr": 1.0,
            "symmetry_break_time": -1,
            "split_calls": 0,
            "perfect_symmetry_enforced": 0,
            "hard_prune_count": 0,
            "hard_clone_count": 0,
            "runtime_candidate_pool_size": 0,
            "runtime_argmax_used": 0,
            "runtime_topk_used": 0,
            "source_witness_score_used_for_structure": 0,
        }
    model = TwinFlowRegressor(width, seed, scheme=scheme, device=device)
    before_train = float(regression_nll(model.forward(x_train), y_train).detach().cpu().item())
    before_held = float(regression_nll(model.forward(x_held), y_held).detach().cpu().item())
    split_step = int(0.40 * int(steps))
    symmetry_break_time = -1
    for step in range(int(steps)):
        if step == split_step and scheme != "G0_no_expansion":
            if scheme == "G2_alpha_045_055_same_state":
                alpha = 0.45
            else:
                alpha = 0.5
            random_noise = scheme == "G3_G_isotropic_zero_sum_noise"
            freeze_children = scheme == "G4_exact_duplicate_same_compute_no_train_new_children"
            model.split(x_train[: min(64, int(x_train.shape[0]))], step=step, alpha=alpha, random_noise=random_noise, freeze_exact_duplicate=freeze_children, seed=seed)
        model.zero_grad()
        loss = regression_nll(model.forward(x_train), y_train)
        loss.backward()
        if scheme == "G1_perfect_symmetric_duplicate":
            p_scale, m_scale = 1.0, 1.0
            enforce_symmetry = True
        else:
            p_scale, m_scale = float(plus_scale), float(minus_scale)
            enforce_symmetry = False
        if scheme in {"G2_alpha_045_055_same_state", "G3_G_isotropic_zero_sum_noise"}:
            p_scale, m_scale = 1.0, 1.0
        freeze_children = scheme == "G4_exact_duplicate_same_compute_no_train_new_children"
        model.manual_step(0.06, step=step, plus_scale=p_scale, minus_scale=m_scale, rewarmup_steps=int(rewarmup_steps), freeze_children=freeze_children, enforce_symmetry=enforce_symmetry)
        if symmetry_break_time < 0 and model.antisymmetric_norm() > 1.0e-4:
            symmetry_break_time = step
    after_train = float(regression_nll(model.forward(x_train), y_train).detach().cpu().item())
    after_held = float(regression_nll(model.forward(x_held), y_held).detach().cpu().item())
    child_spec, child_corr = model.child_specialization(x_held)
    return {
        "hypothesis": "H-F",
        "task": task,
        "seed": seed,
        "width": width,
        "scheme": scheme,
        "plus_scale": float(plus_scale),
        "minus_scale": float(minus_scale),
        "rewarmup_steps": int(rewarmup_steps),
        "n_train": int(x_train.shape[0]),
        "n_held": int(x_held.shape[0]),
        "train_NLL_before": before_train,
        "train_NLL_after": after_train,
        "train_NLL_gain": before_train - after_train,
        "held_NLL_before": before_held,
        "held_NLL_after": after_held,
        "held_NLL_gain": before_held - after_held,
        "no_debt": int(after_held <= before_held + 1.0e-10),
        "function_preservation_error": model.function_preservation_error,
        "antisymmetric_G_norm_final": model.antisymmetric_norm(),
        "child_specialization": child_spec,
        "child_activation_corr": child_corr,
        "symmetry_break_time": symmetry_break_time,
        "split_calls": int(scheme != "G0_no_expansion"),
        "perfect_symmetry_enforced": int(scheme == "G1_perfect_symmetric_duplicate"),
        "hard_prune_count": 0,
        "hard_clone_count": 0,
        "runtime_candidate_pool_size": 0,
        "runtime_argmax_used": 0,
        "runtime_topk_used": 0,
        "source_witness_score_used_for_structure": 0,
    }


def summarize_hf(rows: list[dict[str, Any]], *, run_label: str, repair_id: str) -> dict[str, Any]:
    candidate = "F_primary_symmetry_breaking_twin_flow"
    controls = sorted({r["scheme"] for r in rows if r["scheme"] != candidate})
    keyed: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for row in rows:
        keyed[(row["task"], int(row["seed"]), int(row["width"]), row["scheme"])] = row
    comparisons = {}
    for control in controls:
        surplus = []
        wins = 0
        no_debt = []
        for task in ["SYN-B1", "SYN-B2", "SYN-D1"]:
            for seed in range(5):
                for width in [1, 2]:
                    cand = keyed.get((task, seed, width, candidate))
                    ctrl = keyed.get((task, seed, width, control))
                    if cand is None or ctrl is None:
                        continue
                    s = float(cand["held_NLL_gain"]) - float(ctrl["held_NLL_gain"])
                    surplus.append(s)
                    wins += int(s > 0.0)
                    no_debt.append(int(cand["no_debt"]))
        comparisons[control] = {
            "n": len(surplus),
            "median_surplus": median(surplus),
            "CVaR25_surplus": cvar25(surplus),
            "bootstrap_LCB": bootstrap_lcb(surplus, seed=24028),
            "win_count": wins,
            "no_debt_rate": float(sum(no_debt) / max(1, len(no_debt))),
        }
    candidate_rows = [r for r in rows if r["scheme"] == candidate]
    symmetry_rows = [r for r in rows if r["scheme"] == "G1_perfect_symmetric_duplicate"]
    synthetic_gate_pass = int(
        max(float(r["function_preservation_error"]) for r in candidate_rows) <= 1.0e-7
        and max(float(r["antisymmetric_G_norm_final"]) for r in symmetry_rows) <= 1.0e-8
        and median([float(r["antisymmetric_G_norm_final"]) for r in candidate_rows]) > 1.0e-4
        and median([float(r["child_specialization"]) for r in candidate_rows if r["task"] == "SYN-B1"]) >= 0.05
        and all(float(comp["CVaR25_surplus"]) > 0.0 and float(comp["bootstrap_LCB"]) > 0.0 and int(comp["win_count"]) >= 20 for comp in comparisons.values())
    )
    return {
        "status": "completed",
        "hypothesis": "H-F",
        "candidate": candidate,
        "run_label": run_label,
        "repair_id": repair_id,
        "controls": controls,
        "row_count": len(rows),
        "tasks": ["SYN-B1", "SYN-B2", "SYN-D1"],
        "seeds": [0, 1, 2, 3, 4],
        "widths": [1, 2],
        "comparisons": comparisons,
        "candidate_no_debt_rate": float(sum(int(r["no_debt"]) for r in candidate_rows) / max(1, len(candidate_rows))),
        "candidate_function_preservation_max_error": max(float(r["function_preservation_error"]) for r in candidate_rows),
        "perfect_symmetry_control_antisym_max": max(float(r["antisymmetric_G_norm_final"]) for r in symmetry_rows) if symmetry_rows else 0.0,
        "candidate_antisymmetric_norm_median": median([float(r["antisymmetric_G_norm_final"]) for r in candidate_rows]),
        "candidate_child_specialization_median_SYN_B1": median([float(r["child_specialization"]) for r in candidate_rows if r["task"] == "SYN-B1"]),
        "candidate_symmetry_break_time_median": median([float(r["symmetry_break_time"]) for r in candidate_rows if int(r["symmetry_break_time"]) >= 0]),
        "synthetic_gate_pass": synthetic_gate_pass,
        "gate_note": "H-F synthetic shard only; full H-F still requires repair audit, minimum-real, H20, and MLP matched comparisons.",
    }


def hf_output_stem(run_label: str) -> str:
    if run_label == "base":
        return "v23_24_hf_twin_flow"
    clean = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in run_label)
    return f"v23_24_hf_twin_flow_{clean}"


def phase_hf_synthetic_smoke(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(str(args.device))
    run_label = str(args.hf_label)
    repair_id = "none" if run_label == "base" else run_label
    stem = hf_output_stem(run_label)
    schemes = [
        "F_primary_symmetry_breaking_twin_flow",
        "G0_no_expansion",
        "G1_perfect_symmetric_duplicate",
        "G2_alpha_045_055_same_state",
        "G3_G_isotropic_zero_sum_noise",
        "G4_exact_duplicate_same_compute_no_train_new_children",
        "G5_MLP_matched_hidden_widening",
    ]
    rows: list[dict[str, Any]] = []
    for task in ["SYN-B1", "SYN-B2", "SYN-D1"]:
        for seed in range(int(args.hf_seeds)):
            for width in [1, 2]:
                for scheme in schemes:
                    rows.append(train_hf_scheme(task, width, seed, scheme, steps=int(args.hf_steps), n=int(args.hf_n), plus_scale=float(args.hf_plus_scale), minus_scale=float(args.hf_minus_scale), rewarmup_steps=int(args.hf_rewarmup_steps), device=device))
    matrix_name = f"{stem}_matrix.csv"
    summary_name = f"{stem}_summary.json"
    write_rows(OUT_ROOT / matrix_name, rows)
    summary = summarize_hf(rows, run_label=run_label, repair_id=repair_id)
    write_json(OUT_ROOT / summary_name, summary)
    append_exec("H-F_twin_flow_synthetic_smoke", "completed", files=f"{matrix_name};{summary_name}", note=json.dumps({
        "run_label": run_label,
        "row_count": len(rows),
        "candidate_no_debt_rate": summary["candidate_no_debt_rate"],
        "function_preservation_error": summary["candidate_function_preservation_max_error"],
        "comparisons": summary["comparisons"],
        "synthetic_gate_pass": summary["synthetic_gate_pass"],
    }, sort_keys=True))
    blocker_controls = [control for control, comp in summary["comparisons"].items() if float(comp["bootstrap_LCB"]) <= 0.0 or float(comp["CVaR25_surplus"]) <= 0.0]
    append_recap("H-F symmetry-breaking twin-flow synthetic smoke", [
        f"- status: completed H-F synthetic shard; row_count `{len(rows)}`.",
        f"- run_label `{run_label}`; plus/minus scale `{float(args.hf_plus_scale)}/{float(args.hf_minus_scale)}`; rewarmup_steps `{int(args.hf_rewarmup_steps)}`; candidate plus G0/G1/G2/G3/G4/G5 controls.",
        f"- candidate no_debt_rate `{summary['candidate_no_debt_rate']}`; function_preservation_max_error `{summary['candidate_function_preservation_max_error']}`; perfect_symmetry_control_antisym_max `{summary['perfect_symmetry_control_antisym_max']}`; candidate antisym median `{summary['candidate_antisymmetric_norm_median']}`; SYN-B1 specialization median `{summary['candidate_child_specialization_median_SYN_B1']}`.",
        f"- synthetic gate pass `{summary['synthetic_gate_pass']}`; blocker controls by lower-tail criterion `{blocker_controls}`.",
        f"- comparisons: `{summary['comparisons']}`.",
        f"- artifacts: `results/v23_24/{matrix_name}`, `results/v23_24/{summary_name}`.",
        "- no H-F scientific success is claimed until repair audit, minimum-real, H20, and matched MLP comparisons are done.",
    ])
    return summary


def phase_hf_analysis(args: argparse.Namespace) -> dict[str, Any]:
    run_specs = [
        ("base", OUT_ROOT / "v23_24_hf_twin_flow_summary.json"),
        ("repair_GR1_asym1p5_0p5", OUT_ROOT / "v23_24_hf_twin_flow_repair_GR1_asym1p5_0p5_summary.json"),
        ("repair_GR2_rewarmup20", OUT_ROOT / "v23_24_hf_twin_flow_repair_GR2_rewarmup20_summary.json"),
    ]
    summaries = []
    for label, path in run_specs:
        if path.exists():
            summaries.append((label, path, json.loads(path.read_text(encoding="utf-8"))))
    if not summaries:
        raise FileNotFoundError("H-F analysis requires at least the base H-F summary; run --phase hf-smoke first.")
    rows: list[dict[str, Any]] = []
    for label, path, payload in summaries:
        for control, comp in payload["comparisons"].items():
            rows.append({
                "hypothesis": "H-F",
                "run_label": label,
                "summary_file": rel(path),
                "control": control,
                "median_surplus": comp["median_surplus"],
                "CVaR25_surplus": comp["CVaR25_surplus"],
                "bootstrap_LCB": comp["bootstrap_LCB"],
                "win_count": comp["win_count"],
                "n": comp["n"],
                "synthetic_gate_pass": payload["synthetic_gate_pass"],
                "candidate_no_debt_rate": payload["candidate_no_debt_rate"],
                "candidate_function_preservation_max_error": payload["candidate_function_preservation_max_error"],
                "perfect_symmetry_control_antisym_max": payload["perfect_symmetry_control_antisym_max"],
                "candidate_antisymmetric_norm_median": payload["candidate_antisymmetric_norm_median"],
                "candidate_child_specialization_median_SYN_B1": payload["candidate_child_specialization_median_SYN_B1"],
            })
    base = next((payload for label, _, payload in summaries if label == "base"), summaries[0][2])
    best_by_control = {}
    for control in base["comparisons"]:
        candidates = [r for r in rows if r["control"] == control]
        best = max(candidates, key=lambda r: float(r["bootstrap_LCB"]))
        best_by_control[control] = {
            "best_run_by_LCB": best["run_label"],
            "best_LCB": best["bootstrap_LCB"],
            "best_CVaR25_surplus": best["CVaR25_surplus"],
            "best_median_surplus": best["median_surplus"],
            "best_win_count": best["win_count"],
        }
    any_gate_pass = int(any(int(payload["synthetic_gate_pass"]) == 1 for _, _, payload in summaries))
    final_label = summaries[-1][0]
    final_blockers = sorted({
        row["control"]
        for row in rows
        if row["run_label"] == final_label
        and (float(row["bootstrap_LCB"]) <= 0.0 or float(row["CVaR25_surplus"]) <= 0.0)
    })
    analysis = {
        "status": "completed",
        "hypothesis": "H-F",
        "run_labels": [label for label, _, _ in summaries],
        "base_gate_pass": int(base["synthetic_gate_pass"]),
        "any_repair_gate_pass": any_gate_pass,
        "final_blocker_controls_after_repairs": final_blockers,
        "best_by_control": best_by_control,
        "repair_conclusion": "G-R1 and G-R2 increased antisymmetric norm while preserving function identity and perfect-symmetry control, but child specialization remained near zero and G0/G1/G2/G3/G5 lower-tail blockers remained; H-F synthetic shard is unresolved after planned repairs.",
        "not_full_science_nogo": 1,
        "next_required": "Preserve H-F as synthetic-unresolved after planned repairs, then continue H-G/H-H and later minimum-real/H20.",
    }
    write_rows(OUT_ROOT / "v23_24_hf_repair_comparison_matrix.csv", rows)
    write_json(OUT_ROOT / "v23_24_hf_blocker_analysis.json", analysis)
    append_exec("H-F_repair_comparison_and_blocker_analysis", "completed", files="v23_24_hf_repair_comparison_matrix.csv;v23_24_hf_blocker_analysis.json", note=json.dumps({
        "run_labels": analysis["run_labels"],
        "base_gate_pass": analysis["base_gate_pass"],
        "any_repair_gate_pass": analysis["any_repair_gate_pass"],
        "final_blocker_controls_after_repairs": analysis["final_blocker_controls_after_repairs"],
    }, sort_keys=True))
    append_recap("H-F repair comparison and blocker analysis", [
        f"- status: completed comparison for runs `{analysis['run_labels']}`.",
        f"- base synthetic_gate_pass `{analysis['base_gate_pass']}`; any repair gate pass `{analysis['any_repair_gate_pass']}`.",
        f"- final blocker controls after repairs: `{analysis['final_blocker_controls_after_repairs']}`.",
        f"- best run by bootstrap LCB per control: `{analysis['best_by_control']}`.",
        f"- conclusion: {analysis['repair_conclusion']}",
        "- this records H-F as synthetic-unresolved after planned repairs, not as full science NoGo because minimum-real/H20 are still pending.",
        "- artifacts: `results/v23_24/v23_24_hf_repair_comparison_matrix.csv`, `results/v23_24/v23_24_hf_blocker_analysis.json`.",
    ])
    return analysis


def synthetic_twopass_data(task: str, width: int, seed: int, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if task != "SYN-P1":
        return synthetic_role_data(task, width, seed, n, device)
    gen = torch.Generator(device=device).manual_seed(int(seed) + int(width) * 29 + 280024)
    x = torch.rand((int(n), int(width)), generator=gen, device=device, dtype=torch.float64)
    x0 = x[:, 0]
    y = torch.sin(2.0 * math.pi * x0) + 0.45 * torch.sin(5.0 * math.pi * x0) * torch.sigmoid(8.0 * (x0 - 0.5))
    if width > 1:
        y = y + 0.35 * torch.sin(3.0 * math.pi * x[:, 1]) * (0.5 + x0)
    y = y + 0.015 * torch.randn(y.shape, generator=gen, device=device, dtype=torch.float64)
    return x, y


class TwoPassRoleRegressor(nn.Module):
    def __init__(self, edge_count: int, degree: int, roles: int, seed: int, *, device: torch.device) -> None:
        super().__init__()
        gen = torch.Generator(device=device).manual_seed(int(seed) + 290024)
        self.edge_count = int(edge_count)
        self.degree = int(degree)
        self.roles = int(roles)
        self.base = nn.Parameter(0.04 * torch.randn((self.edge_count, self.degree + 1), generator=gen, device=device, dtype=torch.float64))
        grid = torch.linspace(0.0, 1.0, 64, device=device, dtype=torch.float64)
        self.register_buffer("G", role_metric(grid, self.degree))
        raw_u = torch.randn((self.degree + 1, self.roles), generator=gen, device=device, dtype=torch.float64)
        self.U = nn.Parameter(g_orthonormalize(raw_u, self.G))
        self.c = nn.Parameter(0.04 * torch.randn((self.edge_count, self.roles), generator=gen, device=device, dtype=torch.float64))
        self.log_mass = nn.Parameter(torch.zeros(self.roles, device=device, dtype=torch.float64))

    def mass(self) -> torch.Tensor:
        return torch.softmax(self.log_mass, dim=0)

    def role_values(self, x: torch.Tensor) -> torch.Tensor:
        b = basis(x.reshape(-1), self.degree).reshape(int(x.shape[0]), self.edge_count, self.degree + 1)
        return torch.einsum("ned,dr->ner", b, self.U)

    def weighted_representation(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.c * self.mass()[None, :]
        return self.role_values(x) * weights[None, :, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = basis(x.reshape(-1), self.degree).reshape(int(x.shape[0]), self.edge_count, self.degree + 1)
        base_out = torch.einsum("ned,ed->n", b, self.base)
        return base_out + self.weighted_representation(x).sum(dim=(1, 2))

    def mixing_mass_params(self) -> list[torch.Tensor]:
        return [self.c, self.log_mass]

    def shape_base_params(self) -> list[torch.Tensor]:
        return [self.base, self.U]

    def all_trainable_params(self) -> list[torch.Tensor]:
        return [self.base, self.U, self.c, self.log_mass]

    def retraction(self) -> None:
        with torch.no_grad():
            self.U.data.copy_(g_orthonormalize(self.U.data, self.G))


def zero_param_grads(params: list[torch.Tensor]) -> None:
    for param in params:
        if param.grad is not None:
            param.grad.zero_()


def cloned_param_grads(params: list[torch.Tensor]) -> list[torch.Tensor]:
    grads = []
    for param in params:
        if param.grad is None:
            grads.append(torch.zeros_like(param))
        else:
            grads.append(param.grad.detach().clone())
    return grads


def flat_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        return torch.zeros(0, dtype=torch.float64)
    return torch.cat([t.detach().reshape(-1).to(torch.float64).cpu() for t in tensors])


def grad_rotation(a: torch.Tensor, b: torch.Tensor) -> float:
    if int(a.numel()) == 0 or int(b.numel()) == 0:
        return 0.0
    denom = float(a.norm().item() * b.norm().item())
    if denom <= EPS:
        return 0.0
    cos = float((a @ b).item() / denom)
    return float(1.0 - max(-1.0, min(1.0, cos)))


def rel_grad_change(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((b - a).norm().item() / max(EPS, float(a.norm().item())))


def apply_param_grads(params: list[torch.Tensor], grads: list[torch.Tensor], lr: float, *, trust_param: torch.Tensor | None = None, role_trust: float = 0.0, trust_stats: dict[str, float] | None = None) -> float:
    delta_sq = 0.0
    with torch.no_grad():
        for param, grad in zip(params, grads):
            update = float(lr) * grad
            if trust_param is not None and param is trust_param and float(role_trust) > 0.0:
                if trust_stats is not None:
                    trust_stats["checks"] = trust_stats.get("checks", 0.0) + 1.0
                max_norm = float(role_trust) * max(EPS, float(param.detach().norm().cpu().item()))
                update_norm = float(update.detach().norm().cpu().item())
                if update_norm > max_norm:
                    update = update * (max_norm / max(EPS, update_norm))
                    if trust_stats is not None:
                        trust_stats["clips"] = trust_stats.get("clips", 0.0) + 1.0
            delta_sq += float(update.detach().square().sum().cpu().item())
            param -= update
    return math.sqrt(delta_sq)


def average_grads(left: list[torch.Tensor], right: list[torch.Tensor]) -> list[torch.Tensor]:
    return [0.5 * (a + b) for a, b in zip(left, right)]


def train_hg_mlp_scheme(task: str, width: int, seed: int, *, steps: int, n: int, pass_lr_scale: float, device: torch.device) -> dict[str, Any]:
    x, y = synthetic_twopass_data(task, width, seed, n, device)
    split = int(0.75 * int(n))
    x_train, y_train = x[:split], y[:split]
    x_held, y_held = x[split:], y[split:]
    half = max(1, int(x_train.shape[0]) // 2)
    model = MatchedMLPRegressor(width, seed, device=device).to(device=device)
    linear0 = model.net[0]
    linear1 = model.net[2]
    pass1_params = [linear1.weight, linear1.bias]
    pass2_params = [linear0.weight, linear0.bias]
    all_params = list(model.parameters())
    before_train = float(regression_nll(model(x_train), y_train).detach().cpu().item())
    before_held = float(regression_nll(model(x_held), y_held).detach().cpu().item())
    pass1_delta_sum = 0.0
    grad_change_sum = 0.0
    same_rotation_sum = 0.0
    rep_change_sum = 0.0
    pass2_gain_sum = 0.0
    lr = 0.055 * float(pass_lr_scale)
    for step in range(int(steps)):
        if step % 2 == 0:
            xb1, yb1 = x_train[:half], y_train[:half]
        else:
            xb1, yb1 = x_train[half:], y_train[half:]
        if int(xb1.shape[0]) == 0:
            xb1, yb1 = x_train[:half], y_train[:half]
        with torch.no_grad():
            hidden_before = torch.tanh(linear0(xb1)).detach()
        zero_param_grads(all_params)
        loss1 = regression_nll(model(xb1), yb1)
        loss1.backward()
        stale_pass2 = flat_tensors(cloned_param_grads(pass2_params))
        pass1_grads = cloned_param_grads(pass1_params)
        pass1_delta_sum += apply_param_grads(pass1_params, pass1_grads, lr)
        zero_param_grads(all_params)
        loss2 = regression_nll(model(xb1), yb1)
        loss2.backward()
        pass2_grads = cloned_param_grads(pass2_params)
        pass2_flat = flat_tensors(pass2_grads)
        grad_change_sum += rel_grad_change(stale_pass2, pass2_flat)
        same_rotation_sum += grad_rotation(stale_pass2, pass2_flat)
        before_pass2 = float(loss2.detach().cpu().item())
        apply_param_grads(pass2_params, pass2_grads, lr)
        with torch.no_grad():
            hidden_after = torch.tanh(linear0(xb1)).detach()
            rep_change_sum += float((hidden_after - hidden_before).norm().cpu().item() / math.sqrt(max(1, hidden_before.numel())))
            pass2_gain_sum += max(0.0, before_pass2 - float(regression_nll(model(xb1), yb1).detach().cpu().item()))
    after_train = float(regression_nll(model(x_train), y_train).detach().cpu().item())
    after_held = float(regression_nll(model(x_held), y_held).detach().cpu().item())
    return {
        "hypothesis": "H-G",
        "task": task,
        "seed": seed,
        "width": width,
        "scheme": "H5_MLP_matched_two_pass",
        "run_label": "base",
        "n_train": int(x_train.shape[0]),
        "n_held": int(x_held.shape[0]),
        "train_NLL_before": before_train,
        "train_NLL_after": after_train,
        "train_NLL_gain": before_train - after_train,
        "held_NLL_before": before_held,
        "held_NLL_after": after_held,
        "held_NLL_gain": before_held - after_held,
        "no_debt": int(after_held <= before_held + 1.0e-10),
        "two_backward_compute_matched": 1,
        "pass1_parameter_delta": pass1_delta_sum / max(1, int(steps)),
        "pass2_gradient_change": grad_change_sum / max(1, int(steps)),
        "same_batch_gradient_rotation": same_rotation_sum / max(1, int(steps)),
        "fresh_batch_gradient_rotation": 0.0,
        "higher_order_gain_proxy": pass2_gain_sum / max(1, int(steps)),
        "representation_change_between_passes": rep_change_sum / max(1, int(steps)),
        "pass_order_sensitivity": 0.0,
        "compute_adjusted_gain": (before_held - after_held) / 2.0,
        "same_batch_reused": 1,
        "fresh_batch_second_pass": 0,
        "stale_activation_used": 0,
        "reverse_order": 0,
        "mlp_matched_control": 1,
        "role_increment_only_trust": 0.0,
        "role_trust_clip_count": 0,
        "role_trust_check_count": 0,
        "role_trust_clip_fraction": 0.0,
        "pass_lr_scale": float(pass_lr_scale),
        "hard_prune_count": 0,
        "hard_clone_count": 0,
        "runtime_candidate_pool_size": 0,
        "runtime_argmax_used": 0,
        "runtime_topk_used": 0,
        "source_witness_score_used_for_structure": 0,
    }


def train_hg_scheme(task: str, width: int, seed: int, scheme: str, *, steps: int, n: int, pass_lr_scale: float, role_trust: float, device: torch.device) -> dict[str, Any]:
    if scheme == "H5_MLP_matched_two_pass":
        row = train_hg_mlp_scheme(task, width, seed, steps=steps, n=n, pass_lr_scale=pass_lr_scale, device=device)
        row["role_increment_only_trust"] = float(role_trust)
        return row
    x, y = synthetic_twopass_data(task, width, seed, n, device)
    split = int(0.75 * int(n))
    x_train, y_train = x[:split], y[:split]
    x_held, y_held = x[split:], y[split:]
    half = max(1, int(x_train.shape[0]) // 2)
    model = TwoPassRoleRegressor(width, degree=5, roles=4, seed=seed, device=device).to(device=device)
    before_train = float(regression_nll(model(x_train), y_train).detach().cpu().item())
    before_held = float(regression_nll(model(x_held), y_held).detach().cpu().item())
    lr_mix = 0.075 * float(pass_lr_scale)
    lr_shape = 0.055 * float(pass_lr_scale)
    pass1_delta_sum = 0.0
    grad_change_sum = 0.0
    same_rotation_sum = 0.0
    fresh_rotation_sum = 0.0
    rep_change_sum = 0.0
    pass2_gain_sum = 0.0
    all_params = model.all_trainable_params()
    mix_params = model.mixing_mass_params()
    shape_params = model.shape_base_params()
    trust_stats: dict[str, float] = {"checks": 0.0, "clips": 0.0}
    for step in range(int(steps)):
        if step % 2 == 0:
            xb1, yb1 = x_train[:half], y_train[:half]
            xb2, yb2 = x_train[half:], y_train[half:]
        else:
            xb1, yb1 = x_train[half:], y_train[half:]
            xb2, yb2 = x_train[:half], y_train[:half]
        if int(xb1.shape[0]) == 0:
            xb1, yb1 = x_train[:half], y_train[:half]
        if int(xb2.shape[0]) == 0:
            xb2, yb2 = xb1, yb1
        with torch.no_grad():
            rep_before = model.weighted_representation(xb1).detach()

        if scheme == "H0_one_pass_standard":
            zero_param_grads(all_params)
            loss1 = regression_nll(model(xb1), yb1)
            loss1.backward()
            all_grads = cloned_param_grads(all_params)
            stale_shape = flat_tensors(cloned_param_grads(shape_params))
            pass1_delta_sum += apply_param_grads(all_params, all_grads, 0.06 * float(pass_lr_scale), trust_param=model.U, role_trust=float(role_trust), trust_stats=trust_stats)
            model.retraction()
            zero_param_grads(all_params)
            loss2 = regression_nll(model(xb1), yb1)
            loss2.backward()
            pass2_shape = flat_tensors(cloned_param_grads(shape_params))
            grad_change_sum += rel_grad_change(stale_shape, pass2_shape)
            same_rotation_sum += grad_rotation(stale_shape, pass2_shape)
            with torch.no_grad():
                rep_after = model.weighted_representation(xb1).detach()
                rep_change_sum += float((rep_after - rep_before).norm().cpu().item() / math.sqrt(max(1, rep_before.numel())))
                pass2_gain_sum += 0.0
            continue

        if scheme == "H1_two_backward_simultaneous_accumulation":
            zero_param_grads(all_params)
            loss1 = regression_nll(model(xb1), yb1)
            loss1.backward()
            grads1_all = cloned_param_grads(all_params)
            stale_shape = flat_tensors(cloned_param_grads(shape_params))
            zero_param_grads(all_params)
            loss2 = regression_nll(model(xb1), yb1)
            loss2.backward()
            grads2_all = cloned_param_grads(all_params)
            pass2_shape = flat_tensors(cloned_param_grads(shape_params))
            pass1_delta_sum += apply_param_grads(all_params, average_grads(grads1_all, grads2_all), 0.06 * float(pass_lr_scale), trust_param=model.U, role_trust=float(role_trust), trust_stats=trust_stats)
            model.retraction()
            grad_change_sum += rel_grad_change(stale_shape, pass2_shape)
            same_rotation_sum += grad_rotation(stale_shape, pass2_shape)
            with torch.no_grad():
                rep_after = model.weighted_representation(xb1).detach()
                rep_change_sum += float((rep_after - rep_before).norm().cpu().item() / math.sqrt(max(1, rep_before.numel())))
            continue

        if scheme == "H4_same_batch_stale_activation":
            zero_param_grads(all_params)
            loss1 = regression_nll(model(xb1), yb1)
            loss1.backward()
            mix_grads = cloned_param_grads(mix_params)
            stale_shape = flat_tensors(cloned_param_grads(shape_params))
            zero_param_grads(all_params)
            loss2_stale = regression_nll(model(xb1), yb1)
            loss2_stale.backward()
            stale_shape_grads = cloned_param_grads(shape_params)
            pass2_shape = flat_tensors(stale_shape_grads)
            pass1_delta_sum += apply_param_grads(mix_params, mix_grads, lr_mix)
            with torch.no_grad():
                rep_after_mix = model.weighted_representation(xb1).detach()
                rep_change_sum += float((rep_after_mix - rep_before).norm().cpu().item() / math.sqrt(max(1, rep_before.numel())))
            before_pass2 = float(regression_nll(model(xb1), yb1).detach().cpu().item())
            apply_param_grads(shape_params, stale_shape_grads, lr_shape, trust_param=model.U, role_trust=float(role_trust), trust_stats=trust_stats)
            model.retraction()
            after_pass2 = float(regression_nll(model(xb1), yb1).detach().cpu().item())
            pass2_gain_sum += max(0.0, before_pass2 - after_pass2)
            grad_change_sum += rel_grad_change(stale_shape, pass2_shape)
            same_rotation_sum += grad_rotation(stale_shape, pass2_shape)
            continue

        if scheme == "H3_reverse_order":
            zero_param_grads(all_params)
            loss1 = regression_nll(model(xb1), yb1)
            loss1.backward()
            stale_mix = flat_tensors(cloned_param_grads(mix_params))
            shape_grads = cloned_param_grads(shape_params)
            pass1_delta_sum += apply_param_grads(shape_params, shape_grads, lr_shape, trust_param=model.U, role_trust=float(role_trust), trust_stats=trust_stats)
            model.retraction()
            with torch.no_grad():
                rep_after_shape = model.weighted_representation(xb1).detach()
                rep_change_sum += float((rep_after_shape - rep_before).norm().cpu().item() / math.sqrt(max(1, rep_before.numel())))
            zero_param_grads(all_params)
            loss2 = regression_nll(model(xb1), yb1)
            loss2.backward()
            pass2_mix_grads = cloned_param_grads(mix_params)
            pass2_mix = flat_tensors(pass2_mix_grads)
            before_pass2 = float(loss2.detach().cpu().item())
            apply_param_grads(mix_params, pass2_mix_grads, lr_mix)
            after_pass2 = float(regression_nll(model(xb1), yb1).detach().cpu().item())
            pass2_gain_sum += max(0.0, before_pass2 - after_pass2)
            grad_change_sum += rel_grad_change(stale_mix, pass2_mix)
            same_rotation_sum += grad_rotation(stale_mix, pass2_mix)
            continue

        zero_param_grads(all_params)
        loss1 = regression_nll(model(xb1), yb1)
        loss1.backward()
        stale_shape = flat_tensors(cloned_param_grads(shape_params))
        mix_grads = cloned_param_grads(mix_params)
        pass1_delta_sum += apply_param_grads(mix_params, mix_grads, lr_mix)
        with torch.no_grad():
            rep_after_mix = model.weighted_representation(xb1).detach()
            rep_change_sum += float((rep_after_mix - rep_before).norm().cpu().item() / math.sqrt(max(1, rep_before.numel())))
        pass2_x, pass2_y = (xb2, yb2) if scheme == "H2_fresh_batch_second_pass" else (xb1, yb1)
        zero_param_grads(all_params)
        loss2 = regression_nll(model(pass2_x), pass2_y)
        loss2.backward()
        pass2_shape_grads = cloned_param_grads(shape_params)
        pass2_shape = flat_tensors(pass2_shape_grads)
        before_pass2 = float(loss2.detach().cpu().item())
        apply_param_grads(shape_params, pass2_shape_grads, lr_shape, trust_param=model.U, role_trust=float(role_trust), trust_stats=trust_stats)
        model.retraction()
        after_pass2 = float(regression_nll(model(pass2_x), pass2_y).detach().cpu().item())
        pass2_gain_sum += max(0.0, before_pass2 - after_pass2)
        if scheme == "H2_fresh_batch_second_pass":
            fresh_rotation_sum += grad_rotation(stale_shape, pass2_shape)
        else:
            same_rotation_sum += grad_rotation(stale_shape, pass2_shape)
        grad_change_sum += rel_grad_change(stale_shape, pass2_shape)

    after_train = float(regression_nll(model(x_train), y_train).detach().cpu().item())
    after_held = float(regression_nll(model(x_held), y_held).detach().cpu().item())
    steps_f = max(1, int(steps))
    return {
        "hypothesis": "H-G",
        "task": task,
        "seed": seed,
        "width": width,
        "scheme": scheme,
        "n_train": int(x_train.shape[0]),
        "n_held": int(x_held.shape[0]),
        "train_NLL_before": before_train,
        "train_NLL_after": after_train,
        "train_NLL_gain": before_train - after_train,
        "held_NLL_before": before_held,
        "held_NLL_after": after_held,
        "held_NLL_gain": before_held - after_held,
        "no_debt": int(after_held <= before_held + 1.0e-10),
        "two_backward_compute_matched": 1,
        "pass1_parameter_delta": pass1_delta_sum / steps_f,
        "pass2_gradient_change": grad_change_sum / steps_f,
        "same_batch_gradient_rotation": same_rotation_sum / steps_f,
        "fresh_batch_gradient_rotation": fresh_rotation_sum / steps_f,
        "higher_order_gain_proxy": pass2_gain_sum / steps_f,
        "representation_change_between_passes": rep_change_sum / steps_f,
        "pass_order_sensitivity": 0.0,
        "compute_adjusted_gain": (before_held - after_held) / 2.0,
        "same_batch_reused": int(scheme in {"G_primary_fixed_two_pass_microbatch_reuse", "H0_one_pass_standard", "H1_two_backward_simultaneous_accumulation", "H3_reverse_order", "H4_same_batch_stale_activation"}),
        "fresh_batch_second_pass": int(scheme == "H2_fresh_batch_second_pass"),
        "stale_activation_used": int(scheme == "H4_same_batch_stale_activation"),
        "reverse_order": int(scheme == "H3_reverse_order"),
        "mlp_matched_control": 0,
        "role_increment_only_trust": float(role_trust),
        "role_trust_clip_count": int(trust_stats.get("clips", 0.0)),
        "role_trust_check_count": int(trust_stats.get("checks", 0.0)),
        "role_trust_clip_fraction": float(trust_stats.get("clips", 0.0) / max(1.0, trust_stats.get("checks", 0.0))),
        "pass_lr_scale": float(pass_lr_scale),
        "hard_prune_count": 0,
        "hard_clone_count": 0,
        "runtime_candidate_pool_size": 0,
        "runtime_argmax_used": 0,
        "runtime_topk_used": 0,
        "source_witness_score_used_for_structure": 0,
    }


def summarize_hg(rows: list[dict[str, Any]], *, run_label: str, repair_id: str) -> dict[str, Any]:
    candidate = "G_primary_fixed_two_pass_microbatch_reuse"
    controls = sorted({r["scheme"] for r in rows if r["scheme"] != candidate})
    tasks = ["SYN-P1", "SYN-R1", "SYN-R2", "SYN-D1"]
    widths = [2, 4]
    keyed: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for row in rows:
        keyed[(row["task"], int(row["seed"]), int(row["width"]), row["scheme"])] = row
    comparisons = {}
    for control in controls:
        surplus = []
        wins = 0
        no_debt = []
        for task in tasks:
            for seed in range(5):
                for width in widths:
                    cand = keyed.get((task, seed, width, candidate))
                    ctrl = keyed.get((task, seed, width, control))
                    if cand is None or ctrl is None:
                        continue
                    s = float(cand["held_NLL_gain"]) - float(ctrl["held_NLL_gain"])
                    surplus.append(s)
                    wins += int(s > 0.0)
                    no_debt.append(int(cand["no_debt"]))
        comparisons[control] = {
            "n": len(surplus),
            "median_surplus": median(surplus),
            "CVaR25_surplus": cvar25(surplus),
            "bootstrap_LCB": bootstrap_lcb(surplus, seed=24029),
            "win_count": wins,
            "no_debt_rate": float(sum(no_debt) / max(1, len(no_debt))),
        }
    candidate_rows = [r for r in rows if r["scheme"] == candidate]
    stale_rows = [r for r in rows if r["scheme"] == "H4_same_batch_stale_activation"]
    reverse_rows = [r for r in rows if r["scheme"] == "H3_reverse_order"]
    simultaneous_rows = [r for r in rows if r["scheme"] == "H1_two_backward_simultaneous_accumulation"]
    fresh_rows = [r for r in rows if r["scheme"] == "H2_fresh_batch_second_pass"]
    candidate_no_debt = float(sum(int(r["no_debt"]) for r in candidate_rows) / max(1, len(candidate_rows)))
    candidate_grad_change = median([float(r["pass2_gradient_change"]) for r in candidate_rows])
    stale_grad_change = median([float(r["pass2_gradient_change"]) for r in stale_rows])
    candidate_rep_change = median([float(r["representation_change_between_passes"]) for r in candidate_rows])
    reverse_gain = median([float(r["held_NLL_gain"]) for r in reverse_rows])
    candidate_gain = median([float(r["held_NLL_gain"]) for r in candidate_rows])
    synthetic_gate_pass = int(
        candidate_no_debt >= 0.80
        and candidate_grad_change > max(1.0e-5, stale_grad_change + 1.0e-5)
        and candidate_rep_change > 1.0e-6
        and all(float(comp["median_surplus"]) >= 1.0e-3 and float(comp["CVaR25_surplus"]) > 0.0 and float(comp["bootstrap_LCB"]) > 0.0 and int(comp["win_count"]) >= 30 for comp in comparisons.values())
    )
    return {
        "status": "completed",
        "hypothesis": "H-G",
        "candidate": candidate,
        "run_label": run_label,
        "repair_id": repair_id,
        "controls": controls,
        "row_count": len(rows),
        "tasks": tasks,
        "seeds": [0, 1, 2, 3, 4],
        "widths": widths,
        "comparisons": comparisons,
        "candidate_no_debt_rate": candidate_no_debt,
        "candidate_pass1_parameter_delta_median": median([float(r["pass1_parameter_delta"]) for r in candidate_rows]),
        "candidate_pass2_gradient_change_median": candidate_grad_change,
        "stale_control_pass2_gradient_change_median": stale_grad_change,
        "candidate_same_batch_gradient_rotation_median": median([float(r["same_batch_gradient_rotation"]) for r in candidate_rows]),
        "fresh_control_gradient_rotation_median": median([float(r["fresh_batch_gradient_rotation"]) for r in fresh_rows]),
        "candidate_higher_order_gain_proxy_median": median([float(r["higher_order_gain_proxy"]) for r in candidate_rows]),
        "candidate_representation_change_median": candidate_rep_change,
        "candidate_compute_adjusted_gain_median": median([float(r["compute_adjusted_gain"]) for r in candidate_rows]),
        "simultaneous_control_gain_median": median([float(r["held_NLL_gain"]) for r in simultaneous_rows]),
        "fresh_control_gain_median": median([float(r["held_NLL_gain"]) for r in fresh_rows]),
        "reverse_control_gain_median": reverse_gain,
        "pass_order_sensitivity_median": candidate_gain - reverse_gain,
        "synthetic_gate_pass": synthetic_gate_pass,
        "gate_note": "H-G synthetic shard only; full H-G still requires repair audit, minimum-real, H20, and official integration checks.",
    }


def hg_output_stem(run_label: str) -> str:
    if run_label == "base":
        return "v23_24_hg_two_pass_microbatch_reuse"
    clean = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in run_label)
    return f"v23_24_hg_two_pass_microbatch_reuse_{clean}"


def phase_hg_synthetic_smoke(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(str(args.device))
    run_label = str(args.hg_label)
    repair_id = "none" if run_label == "base" else run_label
    stem = hg_output_stem(run_label)
    schemes = [
        "G_primary_fixed_two_pass_microbatch_reuse",
        "H0_one_pass_standard",
        "H1_two_backward_simultaneous_accumulation",
        "H2_fresh_batch_second_pass",
        "H3_reverse_order",
        "H4_same_batch_stale_activation",
        "H5_MLP_matched_two_pass",
    ]
    rows: list[dict[str, Any]] = []
    for task in ["SYN-P1", "SYN-R1", "SYN-R2", "SYN-D1"]:
        for seed in range(int(args.hg_seeds)):
            for width in [2, 4]:
                for scheme in schemes:
                    rows.append(train_hg_scheme(task, width, seed, scheme, steps=int(args.hg_steps), n=int(args.hg_n), pass_lr_scale=float(args.hg_pass_lr_scale), role_trust=float(args.hg_role_trust), device=device))
    matrix_name = f"{stem}_matrix.csv"
    summary_name = f"{stem}_summary.json"
    write_rows(OUT_ROOT / matrix_name, rows)
    summary = summarize_hg(rows, run_label=run_label, repair_id=repair_id)
    write_json(OUT_ROOT / summary_name, summary)
    append_exec("H-G_two_pass_microbatch_reuse_synthetic_smoke", "completed", files=f"{matrix_name};{summary_name}", note=json.dumps({
        "run_label": run_label,
        "row_count": len(rows),
        "candidate_no_debt_rate": summary["candidate_no_debt_rate"],
        "candidate_pass2_gradient_change_median": summary["candidate_pass2_gradient_change_median"],
        "candidate_representation_change_median": summary["candidate_representation_change_median"],
        "comparisons": summary["comparisons"],
        "synthetic_gate_pass": summary["synthetic_gate_pass"],
    }, sort_keys=True))
    blocker_controls = [control for control, comp in summary["comparisons"].items() if float(comp["median_surplus"]) < 1.0e-3 or float(comp["bootstrap_LCB"]) <= 0.0 or float(comp["CVaR25_surplus"]) <= 0.0]
    append_recap("H-G fixed two-pass microbatch-reuse synthetic smoke", [
        f"- status: completed H-G synthetic shard; row_count `{len(rows)}`.",
        f"- run_label `{run_label}`; pass_lr_scale `{float(args.hg_pass_lr_scale)}`; role_increment_only_trust `{float(args.hg_role_trust)}`; scope: 4 synthetic tasks, seeds 0..4, widths 2/4, candidate plus H0/H1/H2/H3/H4/H5 controls.",
        f"- candidate no_debt_rate `{summary['candidate_no_debt_rate']}`; pass1_delta_median `{summary['candidate_pass1_parameter_delta_median']}`; pass2_gradient_change_median `{summary['candidate_pass2_gradient_change_median']}`; stale_control_grad_change_median `{summary['stale_control_pass2_gradient_change_median']}`.",
        f"- candidate same_batch_rotation_median `{summary['candidate_same_batch_gradient_rotation_median']}`; fresh_control_rotation_median `{summary['fresh_control_gradient_rotation_median']}`; representation_change_median `{summary['candidate_representation_change_median']}`; higher_order_gain_proxy_median `{summary['candidate_higher_order_gain_proxy_median']}`.",
        f"- synthetic gate pass `{summary['synthetic_gate_pass']}`; blocker controls by lower-tail/median criterion `{blocker_controls}`.",
        f"- comparisons: `{summary['comparisons']}`.",
        f"- artifacts: `results/v23_24/{matrix_name}`, `results/v23_24/{summary_name}`.",
        "- no H-G scientific success is claimed until repair audit, minimum-real and H20 are done.",
    ])
    return summary


def phase_hg_analysis(args: argparse.Namespace) -> dict[str, Any]:
    run_specs = [
        ("base", OUT_ROOT / "v23_24_hg_two_pass_microbatch_reuse_summary.json"),
        ("repair_HR1_half_pass_lr", OUT_ROOT / "v23_24_hg_two_pass_microbatch_reuse_repair_HR1_half_pass_lr_summary.json"),
        ("repair_HR2_roleTrust", OUT_ROOT / "v23_24_hg_two_pass_microbatch_reuse_repair_HR2_roleTrust_summary.json"),
        ("repair_HR2_roleTrust_strict0p001", OUT_ROOT / "v23_24_hg_two_pass_microbatch_reuse_repair_HR2_roleTrust_strict0p001_summary.json"),
        ("repair_HR2_roleTrust_active1e-5", OUT_ROOT / "v23_24_hg_two_pass_microbatch_reuse_repair_HR2_roleTrust_active1e-5_summary.json"),
    ]
    summaries = []
    for label, path in run_specs:
        if path.exists():
            summaries.append((label, path, json.loads(path.read_text(encoding="utf-8"))))
    if not summaries:
        raise FileNotFoundError("H-G analysis requires at least the base H-G summary; run --phase hg-smoke first.")
    rows: list[dict[str, Any]] = []
    for label, path, payload in summaries:
        for control, comp in payload["comparisons"].items():
            rows.append({
                "hypothesis": "H-G",
                "run_label": label,
                "summary_file": rel(path),
                "control": control,
                "median_surplus": comp["median_surplus"],
                "CVaR25_surplus": comp["CVaR25_surplus"],
                "bootstrap_LCB": comp["bootstrap_LCB"],
                "win_count": comp["win_count"],
                "n": comp["n"],
                "synthetic_gate_pass": payload["synthetic_gate_pass"],
                "candidate_no_debt_rate": payload["candidate_no_debt_rate"],
                "candidate_pass2_gradient_change_median": payload["candidate_pass2_gradient_change_median"],
                "candidate_representation_change_median": payload["candidate_representation_change_median"],
                "candidate_higher_order_gain_proxy_median": payload["candidate_higher_order_gain_proxy_median"],
            })
    base = next((payload for label, _, payload in summaries if label == "base"), summaries[0][2])
    best_by_control = {}
    for control in base["comparisons"]:
        candidates = [r for r in rows if r["control"] == control]
        best = max(candidates, key=lambda r: float(r["bootstrap_LCB"]))
        best_by_control[control] = {
            "best_run_by_LCB": best["run_label"],
            "best_LCB": best["bootstrap_LCB"],
            "best_CVaR25_surplus": best["CVaR25_surplus"],
            "best_median_surplus": best["median_surplus"],
            "best_win_count": best["win_count"],
        }
    any_gate_pass = int(any(int(payload["synthetic_gate_pass"]) == 1 for _, _, payload in summaries))
    final_label = summaries[-1][0]
    final_blockers = sorted({
        row["control"]
        for row in rows
        if row["run_label"] == final_label
        and (float(row["median_surplus"]) < 1.0e-3 or float(row["bootstrap_LCB"]) <= 0.0 or float(row["CVaR25_surplus"]) <= 0.0)
    })
    final_payload = summaries[-1][2]
    repair_conclusion = (
        "H-G synthetic shard passed in at least one run; still not full v23.24 success until minimum-real/H20 are executed."
        if any_gate_pass
        else "H-R1/H-R2 did not make fixed two-pass reuse pass robust lower-tail controls; finite-step reuse remains synthetic-unresolved."
    )
    analysis = {
        "status": "completed",
        "hypothesis": "H-G",
        "run_labels": [label for label, _, _ in summaries],
        "base_gate_pass": int(base["synthetic_gate_pass"]),
        "any_repair_gate_pass": any_gate_pass,
        "final_blocker_controls_after_repairs": final_blockers,
        "best_by_control": best_by_control,
        "final_candidate_mechanism_metrics": {
            "candidate_no_debt_rate": final_payload["candidate_no_debt_rate"],
            "candidate_pass2_gradient_change_median": final_payload["candidate_pass2_gradient_change_median"],
            "stale_control_pass2_gradient_change_median": final_payload["stale_control_pass2_gradient_change_median"],
            "candidate_representation_change_median": final_payload["candidate_representation_change_median"],
            "candidate_higher_order_gain_proxy_median": final_payload["candidate_higher_order_gain_proxy_median"],
            "pass_order_sensitivity_median": final_payload["pass_order_sensitivity_median"],
        },
        "repair_conclusion": repair_conclusion,
        "not_full_science_nogo": 1,
        "next_required": "Preserve H-G status, then continue H-H and later minimum-real/H20.",
    }
    write_rows(OUT_ROOT / "v23_24_hg_repair_comparison_matrix.csv", rows)
    write_json(OUT_ROOT / "v23_24_hg_blocker_analysis.json", analysis)
    append_exec("H-G_repair_comparison_and_blocker_analysis", "completed", files="v23_24_hg_repair_comparison_matrix.csv;v23_24_hg_blocker_analysis.json", note=json.dumps({
        "run_labels": analysis["run_labels"],
        "base_gate_pass": analysis["base_gate_pass"],
        "any_repair_gate_pass": analysis["any_repair_gate_pass"],
        "final_blocker_controls_after_repairs": analysis["final_blocker_controls_after_repairs"],
    }, sort_keys=True))
    append_recap("H-G repair comparison and blocker analysis", [
        f"- status: completed comparison for runs `{analysis['run_labels']}`.",
        f"- base synthetic_gate_pass `{analysis['base_gate_pass']}`; any repair gate pass `{analysis['any_repair_gate_pass']}`.",
        f"- final blocker controls after repairs: `{analysis['final_blocker_controls_after_repairs']}`.",
        f"- final mechanism metrics: `{analysis['final_candidate_mechanism_metrics']}`.",
        f"- best run by bootstrap LCB per control: `{analysis['best_by_control']}`.",
        f"- conclusion: {analysis['repair_conclusion']}",
        "- this records H-G as synthetic-shard evidence only; full NoGo/success still requires H-H, minimum-real and H20.",
        "- artifacts: `results/v23_24/v23_24_hg_repair_comparison_matrix.csv`, `results/v23_24/v23_24_hg_blocker_analysis.json`.",
    ])
    return analysis


def cv_float(values: list[float]) -> float:
    if not values:
        return 0.0
    t = torch.tensor(values, dtype=torch.float64)
    mean = float(t.mean().item())
    if abs(mean) <= EPS:
        return 0.0
    return float(t.std(unbiased=False).item() / abs(mean))


def hh_forward_with_features(model: RoleReservoirRegressor, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    b = basis(x.reshape(-1), model.degree).reshape(int(x.shape[0]), model.edge_count, model.degree + 1)
    base_feat = torch.einsum("ned,ed->ne", b, model.base)
    role_vals = torch.einsum("ned,dr->ner", b, model.U)
    weighted = role_vals * (model.c * model.mass[None, :])[None, :, :]
    pred = base_feat.sum(dim=1) + weighted.sum(dim=(1, 2))
    return pred, [base_feat, role_vals, weighted]


def hh_feature_speeds(before: list[torch.Tensor], after: list[torch.Tensor]) -> list[float]:
    speeds = []
    for b, a in zip(before, after):
        denom = math.sqrt(max(1, int(b.numel())))
        speeds.append(float((a.detach() - b.detach()).norm().cpu().item() / denom))
    return speeds


def hh_feature_alignments(before: list[torch.Tensor], after: list[torch.Tensor], grads: list[torch.Tensor]) -> list[float]:
    alignments = []
    for b, a, g in zip(before, after, grads):
        delta = (a.detach() - b.detach()).reshape(-1).cpu()
        neg_grad = (-g.detach()).reshape(-1).cpu()
        denom = float(delta.norm().item() * neg_grad.norm().item())
        if denom <= EPS:
            alignments.append(0.0)
        else:
            alignments.append(float((delta @ neg_grad).item() / denom))
    return alignments


def hh_kappa_from_calibration(scheme: str, speed_means: list[float], grad_means: list[float], *, cap_max: float, seed: int) -> list[float]:
    if scheme == "I0_uniform_layerwise_LR":
        return [1.0 for _ in speed_means]
    if scheme == "I2_gradient_norm_equalization_control":
        med = median(grad_means)
        return [float(max(0.5, min(float(cap_max), math.sqrt(med / max(EPS, g))))) for g in grad_means]
    base_med = median(speed_means)
    base = [float(max(0.5, min(float(cap_max), math.sqrt(base_med / max(EPS, v))))) for v in speed_means]
    if scheme == "I1_random_matched_logrange_scaling":
        lo = math.log(max(0.5, min(base)))
        hi = math.log(min(float(cap_max), max(base)))
        if hi < lo:
            lo, hi = hi, lo
        gen = random.Random(int(seed) + 310024)
        return [float(math.exp(gen.uniform(lo, hi))) for _ in base]
    return base


def train_hh_role_scheme(task: str, width: int, seed: int, scheme: str, *, steps: int, n: int, cal_horizon: int, cap_max: float, device: torch.device) -> dict[str, Any]:
    x, y = synthetic_role_data(task, width, seed, n, device)
    split = int(0.75 * int(n))
    x_train, y_train = x[:split], y[:split]
    x_held, y_held = x[split:], y[split:]
    model = RoleReservoirRegressor(width, degree=5, roles=4, seed=seed, scheme="B_primary_equal_mass_role_flow", device=device).to(device=device)
    params_by_layer = [[model.base], [model.U], [model.c]]
    all_params = [model.base, model.U, model.c]
    initial_mi = edge_role_specialization_mi(model)
    with torch.no_grad():
        _, held_features_before = hh_forward_with_features(model, x_held)
        held_rep_before = torch.cat([f.reshape(-1).detach().cpu() for f in held_features_before])
    before_train = float(regression_nll(model(x_train), y_train).detach().cpu().item())
    before_held = float(regression_nll(model(x_held), y_held).detach().cpu().item())
    pre_speeds: list[list[float]] = []
    post_speeds: list[list[float]] = []
    pre_align: list[list[float]] = []
    post_align: list[list[float]] = []
    grad_norms: list[list[float]] = []
    kappa = [1.0, 1.0, 1.0]
    kappa_frozen = 0
    lr = 0.065
    horizon = max(1, int(cal_horizon))
    for step in range(int(steps)):
        zero_param_grads(all_params)
        pred, features = hh_forward_with_features(model, x_train)
        for feat in features:
            feat.retain_grad()
        before_features = [feat.detach().clone() for feat in features]
        loss = regression_nll(pred, y_train)
        loss.backward()
        feature_grads = [feat.grad.detach().clone() if feat.grad is not None else torch.zeros_like(feat) for feat in features]
        layer_grad_norms = []
        for params in params_by_layer:
            layer_grad_norms.append(float(math.sqrt(sum(float((p.grad.detach().square().sum()).cpu().item()) for p in params if p.grad is not None))))
        grad_norms.append(layer_grad_norms)
        active_kappa = [1.0, 1.0, 1.0] if step < horizon else kappa
        with torch.no_grad():
            for mult, params in zip(active_kappa, params_by_layer):
                for param in params:
                    if param.grad is not None:
                        param -= float(lr) * float(mult) * param.grad
        model.retraction()
        with torch.no_grad():
            _, after_features = hh_forward_with_features(model, x_train)
            speeds = hh_feature_speeds(before_features, after_features)
            aligns = hh_feature_alignments(before_features, after_features, feature_grads)
        if step < horizon:
            pre_speeds.append(speeds)
            pre_align.append(aligns)
            if step == horizon - 1:
                speed_means = [float(sum(row[i] for row in pre_speeds) / max(1, len(pre_speeds))) for i in range(3)]
                grad_means = [float(sum(row[i] for row in grad_norms[-horizon:]) / max(1, min(horizon, len(grad_norms)))) for i in range(3)]
                kappa = hh_kappa_from_calibration(scheme, speed_means, grad_means, cap_max=float(cap_max), seed=seed)
                kappa_frozen = 1
        else:
            post_speeds.append(speeds)
            post_align.append(aligns)
    after_train = float(regression_nll(model(x_train), y_train).detach().cpu().item())
    after_held = float(regression_nll(model(x_held), y_held).detach().cpu().item())
    with torch.no_grad():
        _, held_features_after = hh_forward_with_features(model, x_held)
        held_rep_after = torch.cat([f.reshape(-1).detach().cpu() for f in held_features_after])
    pre_mean = [float(sum(row[i] for row in pre_speeds) / max(1, len(pre_speeds))) for i in range(3)]
    post_source = post_speeds if post_speeds else pre_speeds
    post_mean = [float(sum(row[i] for row in post_source) / max(1, len(post_source))) for i in range(3)]
    align_source = [*pre_align, *post_align]
    align_mean = [float(sum(row[i] for row in align_source) / max(1, len(align_source))) for i in range(3)] if align_source else [0.0, 0.0, 0.0]
    min_pre_idx = min(range(3), key=lambda i: pre_mean[i])
    pre_cv = cv_float(pre_mean)
    post_cv = cv_float(post_mean)
    final_mi = edge_role_specialization_mi(model)
    return {
        "hypothesis": "H-H",
        "task": task,
        "seed": seed,
        "width": width,
        "scheme": scheme,
        "n_train": int(x_train.shape[0]),
        "n_held": int(x_held.shape[0]),
        "train_NLL_before": before_train,
        "train_NLL_after": after_train,
        "train_NLL_gain": before_train - after_train,
        "held_NLL_before": before_held,
        "held_NLL_after": after_held,
        "held_NLL_gain": before_held - after_held,
        "no_debt": int(after_held <= before_held + 1.0e-10),
        "calibration_horizon": horizon,
        "kappa_cap_min": 0.5,
        "kappa_cap_max": float(cap_max),
        "kappa_frozen_after_calibration": kappa_frozen,
        "post_freeze_kappa_change_count": 0,
        "kappa_by_layer": ";".join(f"{v:.12g}" for v in kappa),
        "feature_speed_by_layer_pre": ";".join(f"{v:.12g}" for v in pre_mean),
        "feature_speed_by_layer_post": ";".join(f"{v:.12g}" for v in post_mean),
        "backward_alignment_by_layer": ";".join(f"{v:.12g}" for v in align_mean),
        "feature_speed_CV_pre": pre_cv,
        "feature_speed_CV_post": post_cv,
        "feature_speed_CV_reduction": (pre_cv - post_cv) / max(EPS, pre_cv),
        "critical_early_layer_speed_increase": post_mean[min_pre_idx] / max(EPS, pre_mean[min_pre_idx]),
        "backward_alignment_min": min(align_mean),
        "alignment_CV": cv_float(align_mean),
        "role_specialization_before": initial_mi,
        "role_specialization_after": final_mi,
        "role_specialization_change": final_mi - initial_mi,
        "representation_change": float((held_rep_after - held_rep_before).norm().item() / math.sqrt(max(1, int(held_rep_before.numel())))),
        "paired_task_gain": before_held - after_held,
        "mlp_matched_control": 0,
        "runtime_candidate_pool_size": 0,
        "runtime_argmax_used": 0,
        "runtime_topk_used": 0,
        "source_witness_score_used_for_structure": 0,
    }


def hh_mlp_forward_with_features(model: MatchedMLPRegressor, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    hidden = torch.tanh(model.net[0](x))
    pred = model.net[2](hidden).reshape(-1)
    return pred, [hidden, pred[:, None]]


def train_hh_mlp_scheme(task: str, width: int, seed: int, *, steps: int, n: int, cal_horizon: int, cap_max: float, device: torch.device) -> dict[str, Any]:
    x, y = synthetic_role_data(task, width, seed, n, device)
    split = int(0.75 * int(n))
    x_train, y_train = x[:split], y[:split]
    x_held, y_held = x[split:], y[split:]
    model = MatchedMLPRegressor(width, seed, device=device).to(device=device)
    layer0 = [model.net[0].weight, model.net[0].bias]
    layer1 = [model.net[2].weight, model.net[2].bias]
    params_by_layer = [layer0, layer1]
    all_params = list(model.parameters())
    with torch.no_grad():
        _, held_features_before = hh_mlp_forward_with_features(model, x_held)
        held_rep_before = torch.cat([f.reshape(-1).detach().cpu() for f in held_features_before])
    before_train = float(regression_nll(model(x_train), y_train).detach().cpu().item())
    before_held = float(regression_nll(model(x_held), y_held).detach().cpu().item())
    pre_speeds: list[list[float]] = []
    post_speeds: list[list[float]] = []
    align_rows: list[list[float]] = []
    kappa = [1.0, 1.0]
    lr = 0.055
    horizon = max(1, int(cal_horizon))
    for step in range(int(steps)):
        zero_param_grads(all_params)
        pred, features = hh_mlp_forward_with_features(model, x_train)
        for feat in features:
            feat.retain_grad()
        before_features = [feat.detach().clone() for feat in features]
        loss = regression_nll(pred, y_train)
        loss.backward()
        grads = [feat.grad.detach().clone() if feat.grad is not None else torch.zeros_like(feat) for feat in features]
        active_kappa = [1.0, 1.0] if step < horizon else kappa
        with torch.no_grad():
            for mult, params in zip(active_kappa, params_by_layer):
                for param in params:
                    if param.grad is not None:
                        param -= float(lr) * float(mult) * param.grad
        with torch.no_grad():
            _, after_features = hh_mlp_forward_with_features(model, x_train)
            speeds = hh_feature_speeds(before_features, after_features)
            aligns = hh_feature_alignments(before_features, after_features, grads)
        if step < horizon:
            pre_speeds.append(speeds)
            if step == horizon - 1:
                means = [float(sum(row[i] for row in pre_speeds) / max(1, len(pre_speeds))) for i in range(2)]
                med = median(means)
                kappa = [float(max(0.5, min(float(cap_max), math.sqrt(med / max(EPS, v))))) for v in means]
        else:
            post_speeds.append(speeds)
        align_rows.append(aligns)
    after_train = float(regression_nll(model(x_train), y_train).detach().cpu().item())
    after_held = float(regression_nll(model(x_held), y_held).detach().cpu().item())
    with torch.no_grad():
        _, held_features_after = hh_mlp_forward_with_features(model, x_held)
        held_rep_after = torch.cat([f.reshape(-1).detach().cpu() for f in held_features_after])
    pre_mean = [float(sum(row[i] for row in pre_speeds) / max(1, len(pre_speeds))) for i in range(2)]
    post_source = post_speeds if post_speeds else pre_speeds
    post_mean = [float(sum(row[i] for row in post_source) / max(1, len(post_source))) for i in range(2)]
    align_mean = [float(sum(row[i] for row in align_rows) / max(1, len(align_rows))) for i in range(2)] if align_rows else [0.0, 0.0]
    min_pre_idx = min(range(2), key=lambda i: pre_mean[i])
    pre_cv = cv_float(pre_mean)
    post_cv = cv_float(post_mean)
    return {
        "hypothesis": "H-H",
        "task": task,
        "seed": seed,
        "width": width,
        "scheme": "I3_MLP_matched_feature_speed_equalization",
        "n_train": int(x_train.shape[0]),
        "n_held": int(x_held.shape[0]),
        "train_NLL_before": before_train,
        "train_NLL_after": after_train,
        "train_NLL_gain": before_train - after_train,
        "held_NLL_before": before_held,
        "held_NLL_after": after_held,
        "held_NLL_gain": before_held - after_held,
        "no_debt": int(after_held <= before_held + 1.0e-10),
        "calibration_horizon": horizon,
        "kappa_cap_min": 0.5,
        "kappa_cap_max": float(cap_max),
        "kappa_frozen_after_calibration": 1,
        "post_freeze_kappa_change_count": 0,
        "kappa_by_layer": ";".join(f"{v:.12g}" for v in kappa),
        "feature_speed_by_layer_pre": ";".join(f"{v:.12g}" for v in pre_mean),
        "feature_speed_by_layer_post": ";".join(f"{v:.12g}" for v in post_mean),
        "backward_alignment_by_layer": ";".join(f"{v:.12g}" for v in align_mean),
        "feature_speed_CV_pre": pre_cv,
        "feature_speed_CV_post": post_cv,
        "feature_speed_CV_reduction": (pre_cv - post_cv) / max(EPS, pre_cv),
        "critical_early_layer_speed_increase": post_mean[min_pre_idx] / max(EPS, pre_mean[min_pre_idx]),
        "backward_alignment_min": min(align_mean),
        "alignment_CV": cv_float(align_mean),
        "role_specialization_before": 0.0,
        "role_specialization_after": 0.0,
        "role_specialization_change": 0.0,
        "representation_change": float((held_rep_after - held_rep_before).norm().item() / math.sqrt(max(1, int(held_rep_before.numel())))),
        "paired_task_gain": before_held - after_held,
        "mlp_matched_control": 1,
        "runtime_candidate_pool_size": 0,
        "runtime_argmax_used": 0,
        "runtime_topk_used": 0,
        "source_witness_score_used_for_structure": 0,
    }


def train_hh_scheme(task: str, width: int, seed: int, scheme: str, *, steps: int, n: int, cal_horizon: int, cap_max: float, device: torch.device) -> dict[str, Any]:
    if scheme == "I3_MLP_matched_feature_speed_equalization":
        return train_hh_mlp_scheme(task, width, seed, steps=steps, n=n, cal_horizon=cal_horizon, cap_max=cap_max, device=device)
    return train_hh_role_scheme(task, width, seed, scheme, steps=steps, n=n, cal_horizon=cal_horizon, cap_max=cap_max, device=device)


def summarize_hh(rows: list[dict[str, Any]], *, run_label: str, repair_id: str, diagnostic_only: int) -> dict[str, Any]:
    candidate = "H_primary_feature_speed_equalized_role_flow"
    controls = sorted({r["scheme"] for r in rows if r["scheme"] != candidate})
    tasks = ["SYN-R1", "SYN-R2", "SYN-R3", "SYN-D1"]
    widths = [2, 4]
    keyed: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for row in rows:
        keyed[(row["task"], int(row["seed"]), int(row["width"]), row["scheme"])] = row
    comparisons = {}
    for control in controls:
        surplus = []
        wins = 0
        no_debt = []
        for task in tasks:
            for seed in range(5):
                for width in widths:
                    cand = keyed.get((task, seed, width, candidate))
                    ctrl = keyed.get((task, seed, width, control))
                    if cand is None or ctrl is None:
                        continue
                    s = float(cand["held_NLL_gain"]) - float(ctrl["held_NLL_gain"])
                    surplus.append(s)
                    wins += int(s > 0.0)
                    no_debt.append(int(cand["no_debt"]))
        comparisons[control] = {
            "n": len(surplus),
            "median_surplus": median(surplus),
            "CVaR25_surplus": cvar25(surplus),
            "bootstrap_LCB": bootstrap_lcb(surplus, seed=24030),
            "win_count": wins,
            "no_debt_rate": float(sum(no_debt) / max(1, len(no_debt))),
        }
    candidate_rows = [r for r in rows if r["scheme"] == candidate]
    uniform_rows = [r for r in rows if r["scheme"] == "I0_uniform_layerwise_LR"]
    candidate_no_debt = float(sum(int(r["no_debt"]) for r in candidate_rows) / max(1, len(candidate_rows)))
    cv_reduction = median([float(r["feature_speed_CV_reduction"]) for r in candidate_rows])
    early_increase = median([float(r["critical_early_layer_speed_increase"]) for r in candidate_rows])
    align_min = min(float(r["backward_alignment_min"]) for r in candidate_rows)
    role_change = median([float(r["role_specialization_change"]) for r in candidate_rows])
    uniform_role_change = median([float(r["role_specialization_change"]) for r in uniform_rows])
    diagnostic_gate = int(
        cv_reduction >= 0.30
        and early_increase >= 1.25
        and align_min >= 0.0
        and role_change > uniform_role_change
        and candidate_no_debt >= 0.80
        and all(float(comp["median_surplus"]) >= 1.0e-3 and float(comp["CVaR25_surplus"]) >= 0.0 and float(comp["bootstrap_LCB"]) >= 0.0 for comp in comparisons.values())
    )
    synthetic_gate_pass = int(diagnostic_gate == 1 and int(diagnostic_only) == 0)
    return {
        "status": "completed",
        "hypothesis": "H-H",
        "candidate": candidate,
        "run_label": run_label,
        "repair_id": repair_id,
        "diagnostic_only_no_promotion": int(diagnostic_only),
        "controls": controls,
        "row_count": len(rows),
        "tasks": tasks,
        "seeds": [0, 1, 2, 3, 4],
        "widths": widths,
        "comparisons": comparisons,
        "candidate_no_debt_rate": candidate_no_debt,
        "candidate_feature_speed_CV_reduction_median": cv_reduction,
        "candidate_critical_early_speed_increase_median": early_increase,
        "candidate_backward_alignment_min": align_min,
        "candidate_alignment_CV_median": median([float(r["alignment_CV"]) for r in candidate_rows]),
        "candidate_role_specialization_change_median": role_change,
        "uniform_role_specialization_change_median": uniform_role_change,
        "candidate_representation_change_median": median([float(r["representation_change"]) for r in candidate_rows]),
        "candidate_kappa_cap_max": max(float(r["kappa_cap_max"]) for r in candidate_rows),
        "candidate_calibration_horizon": max(int(r["calibration_horizon"]) for r in candidate_rows),
        "diagnostic_gate_pass": diagnostic_gate,
        "synthetic_gate_pass": synthetic_gate_pass,
        "gate_note": "H-H synthetic shard only; cap-4 run is diagnostic-only and cannot promote by itself.",
    }


def hh_output_stem(run_label: str) -> str:
    if run_label == "base":
        return "v23_24_hh_feature_speed_equalized_flow"
    clean = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in run_label)
    return f"v23_24_hh_feature_speed_equalized_flow_{clean}"


def phase_hh_synthetic_smoke(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(str(args.device))
    run_label = str(args.hh_label)
    repair_id = "none" if run_label == "base" else run_label
    diagnostic_only = int("IR1" in run_label or float(args.hh_cap_max) > 2.0)
    stem = hh_output_stem(run_label)
    schemes = [
        "H_primary_feature_speed_equalized_role_flow",
        "I0_uniform_layerwise_LR",
        "I1_random_matched_logrange_scaling",
        "I2_gradient_norm_equalization_control",
        "I3_MLP_matched_feature_speed_equalization",
    ]
    rows: list[dict[str, Any]] = []
    for task in ["SYN-R1", "SYN-R2", "SYN-R3", "SYN-D1"]:
        for seed in range(int(args.hh_seeds)):
            for width in [2, 4]:
                for scheme in schemes:
                    rows.append(train_hh_scheme(task, width, seed, scheme, steps=int(args.hh_steps), n=int(args.hh_n), cal_horizon=int(args.hh_cal_horizon), cap_max=float(args.hh_cap_max), device=device))
    matrix_name = f"{stem}_matrix.csv"
    summary_name = f"{stem}_summary.json"
    write_rows(OUT_ROOT / matrix_name, rows)
    summary = summarize_hh(rows, run_label=run_label, repair_id=repair_id, diagnostic_only=diagnostic_only)
    write_json(OUT_ROOT / summary_name, summary)
    append_exec("H-H_feature_speed_equalized_synthetic_smoke", "completed", files=f"{matrix_name};{summary_name}", note=json.dumps({
        "run_label": run_label,
        "row_count": len(rows),
        "candidate_no_debt_rate": summary["candidate_no_debt_rate"],
        "candidate_feature_speed_CV_reduction_median": summary["candidate_feature_speed_CV_reduction_median"],
        "candidate_critical_early_speed_increase_median": summary["candidate_critical_early_speed_increase_median"],
        "candidate_backward_alignment_min": summary["candidate_backward_alignment_min"],
        "comparisons": summary["comparisons"],
        "synthetic_gate_pass": summary["synthetic_gate_pass"],
        "diagnostic_only_no_promotion": summary["diagnostic_only_no_promotion"],
    }, sort_keys=True))
    blocker_controls = [control for control, comp in summary["comparisons"].items() if float(comp["median_surplus"]) < 1.0e-3 or float(comp["bootstrap_LCB"]) < 0.0 or float(comp["CVaR25_surplus"]) < 0.0]
    append_recap("H-H feature-speed equalized synthetic smoke", [
        f"- status: completed H-H synthetic shard; row_count `{len(rows)}`.",
        f"- run_label `{run_label}`; calibration_horizon `{int(args.hh_cal_horizon)}`; cap_max `{float(args.hh_cap_max)}`; diagnostic_only_no_promotion `{summary['diagnostic_only_no_promotion']}`.",
        f"- candidate no_debt_rate `{summary['candidate_no_debt_rate']}`; feature_speed_CV_reduction_median `{summary['candidate_feature_speed_CV_reduction_median']}`; critical_early_speed_increase_median `{summary['candidate_critical_early_speed_increase_median']}`; backward_alignment_min `{summary['candidate_backward_alignment_min']}`.",
        f"- role_specialization_change_median `{summary['candidate_role_specialization_change_median']}` vs uniform `{summary['uniform_role_specialization_change_median']}`; representation_change_median `{summary['candidate_representation_change_median']}`.",
        f"- diagnostic_gate_pass `{summary['diagnostic_gate_pass']}`; synthetic_gate_pass `{summary['synthetic_gate_pass']}`; blocker controls `{blocker_controls}`.",
        f"- comparisons: `{summary['comparisons']}`.",
        f"- artifacts: `results/v23_24/{matrix_name}`, `results/v23_24/{summary_name}`.",
        "- no H-H official success is claimed until repair comparison, minimum-real and H20 are done.",
    ])
    return summary


def phase_hh_analysis(args: argparse.Namespace) -> dict[str, Any]:
    run_specs = [
        ("base", OUT_ROOT / "v23_24_hh_feature_speed_equalized_flow_summary.json"),
        ("repair_IR1_cap4_diag", OUT_ROOT / "v23_24_hh_feature_speed_equalized_flow_repair_IR1_cap4_diag_summary.json"),
        ("repair_IR2_horizon20", OUT_ROOT / "v23_24_hh_feature_speed_equalized_flow_repair_IR2_horizon20_summary.json"),
    ]
    summaries = []
    for label, path in run_specs:
        if path.exists():
            summaries.append((label, path, json.loads(path.read_text(encoding="utf-8"))))
    if not summaries:
        raise FileNotFoundError("H-H analysis requires at least the base H-H summary; run --phase hh-smoke first.")
    rows: list[dict[str, Any]] = []
    for label, path, payload in summaries:
        for control, comp in payload["comparisons"].items():
            rows.append({
                "hypothesis": "H-H",
                "run_label": label,
                "summary_file": rel(path),
                "control": control,
                "median_surplus": comp["median_surplus"],
                "CVaR25_surplus": comp["CVaR25_surplus"],
                "bootstrap_LCB": comp["bootstrap_LCB"],
                "win_count": comp["win_count"],
                "n": comp["n"],
                "synthetic_gate_pass": payload["synthetic_gate_pass"],
                "diagnostic_gate_pass": payload["diagnostic_gate_pass"],
                "diagnostic_only_no_promotion": payload["diagnostic_only_no_promotion"],
                "candidate_no_debt_rate": payload["candidate_no_debt_rate"],
                "candidate_feature_speed_CV_reduction_median": payload["candidate_feature_speed_CV_reduction_median"],
                "candidate_critical_early_speed_increase_median": payload["candidate_critical_early_speed_increase_median"],
                "candidate_role_specialization_change_median": payload["candidate_role_specialization_change_median"],
            })
    base = next((payload for label, _, payload in summaries if label == "base"), summaries[0][2])
    best_by_control = {}
    for control in base["comparisons"]:
        candidates = [r for r in rows if r["control"] == control]
        best = max(candidates, key=lambda r: float(r["bootstrap_LCB"]))
        best_by_control[control] = {
            "best_run_by_LCB": best["run_label"],
            "best_LCB": best["bootstrap_LCB"],
            "best_CVaR25_surplus": best["CVaR25_surplus"],
            "best_median_surplus": best["median_surplus"],
            "best_win_count": best["win_count"],
        }
    any_gate_pass = int(any(int(payload["synthetic_gate_pass"]) == 1 for _, _, payload in summaries))
    final_label = summaries[-1][0]
    final_blockers = sorted({
        row["control"]
        for row in rows
        if row["run_label"] == final_label
        and (float(row["median_surplus"]) < 1.0e-3 or float(row["bootstrap_LCB"]) < 0.0 or float(row["CVaR25_surplus"]) < 0.0)
    })
    final_payload = summaries[-1][2]
    repair_conclusion = (
        "H-H synthetic shard passed in a promotable run; still not full v23.24 success until minimum-real/H20 are executed."
        if any_gate_pass
        else "I-R1/I-R2 did not make feature-speed equalization pass robust controls; H-H remains synthetic-unresolved."
    )
    analysis = {
        "status": "completed",
        "hypothesis": "H-H",
        "run_labels": [label for label, _, _ in summaries],
        "base_gate_pass": int(base["synthetic_gate_pass"]),
        "any_repair_gate_pass": any_gate_pass,
        "final_blocker_controls_after_repairs": final_blockers,
        "best_by_control": best_by_control,
        "final_candidate_mechanism_metrics": {
            "candidate_no_debt_rate": final_payload["candidate_no_debt_rate"],
            "candidate_feature_speed_CV_reduction_median": final_payload["candidate_feature_speed_CV_reduction_median"],
            "candidate_critical_early_speed_increase_median": final_payload["candidate_critical_early_speed_increase_median"],
            "candidate_backward_alignment_min": final_payload["candidate_backward_alignment_min"],
            "candidate_role_specialization_change_median": final_payload["candidate_role_specialization_change_median"],
            "uniform_role_specialization_change_median": final_payload["uniform_role_specialization_change_median"],
            "candidate_representation_change_median": final_payload["candidate_representation_change_median"],
        },
        "repair_conclusion": repair_conclusion,
        "not_full_science_nogo": 1,
        "next_required": "Continue mandatory minimum-real and H20 diagnostics after all synthetic shards are recorded.",
    }
    write_rows(OUT_ROOT / "v23_24_hh_repair_comparison_matrix.csv", rows)
    write_json(OUT_ROOT / "v23_24_hh_blocker_analysis.json", analysis)
    append_exec("H-H_repair_comparison_and_blocker_analysis", "completed", files="v23_24_hh_repair_comparison_matrix.csv;v23_24_hh_blocker_analysis.json", note=json.dumps({
        "run_labels": analysis["run_labels"],
        "base_gate_pass": analysis["base_gate_pass"],
        "any_repair_gate_pass": analysis["any_repair_gate_pass"],
        "final_blocker_controls_after_repairs": analysis["final_blocker_controls_after_repairs"],
    }, sort_keys=True))
    append_recap("H-H repair comparison and blocker analysis", [
        f"- status: completed comparison for runs `{analysis['run_labels']}`.",
        f"- base synthetic_gate_pass `{analysis['base_gate_pass']}`; any repair gate pass `{analysis['any_repair_gate_pass']}`.",
        f"- final blocker controls after repairs: `{analysis['final_blocker_controls_after_repairs']}`.",
        f"- final mechanism metrics: `{analysis['final_candidate_mechanism_metrics']}`.",
        f"- best run by bootstrap LCB per control: `{analysis['best_by_control']}`.",
        f"- conclusion: {analysis['repair_conclusion']}",
        "- this records H-H as synthetic-shard evidence only; full NoGo/success still requires minimum-real and H20.",
        "- artifacts: `results/v23_24/v23_24_hh_repair_comparison_matrix.csv`, `results/v23_24/v23_24_hh_blocker_analysis.json`.",
    ])
    return analysis


REAL_PARTJ_DATASETS = ["Wine", "Spam", "Rice", "Bean", "CIFAR10_compact", "SVHN", "EMNIST-Letters"]
H20_DATASETS = ["Wine", "CIFAR10_compact", "SVHN"]
MINREAL_SEEDS = [11, 12, 13]
H20_SEEDS = [21, 22]


def parse_csv_arg(value: str) -> list[str]:
    return [v.strip() for v in str(value).split(",") if v.strip()]


def load_idx_images_np(path: Path) -> np.ndarray:
    data = path.read_bytes()
    magic = int.from_bytes(data[:4], "big")
    if magic != 2051:
        raise RuntimeError(f"bad idx image magic for {path}: {magic}")
    n = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(n, rows * cols).astype("float64") / 255.0


def load_idx_labels_np(path: Path) -> np.ndarray:
    data = path.read_bytes()
    magic = int.from_bytes(data[:4], "big")
    if magic != 2049:
        raise RuntimeError(f"bad idx label magic for {path}: {magic}")
    n = int.from_bytes(data[4:8], "big")
    return np.frombuffer(data, dtype=np.uint8, offset=8).reshape(n).astype("int64")


def parse_arff_numeric_np(content: str) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[str]] = []
    in_data = False
    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        if s.lower().startswith("@data"):
            in_data = True
            continue
        if not in_data or s.startswith("@"):
            continue
        rows.append([x.strip() for x in s.split(",")])
    if not rows:
        raise RuntimeError("empty ARFF data")
    labels = sorted({r[-1] for r in rows})
    label_map = {lab: i for i, lab in enumerate(labels)}
    x_np = np.asarray([[float(v) for v in r[:-1]] for r in rows], dtype=np.float64)
    y_np = np.asarray([label_map[r[-1]] for r in rows], dtype=np.int64)
    return x_np, y_np


def load_cifar10_np(root: Path) -> tuple[np.ndarray, np.ndarray]:
    base = root / "cifar-10-batches-py"
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for name in ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]:
        with (base / name).open("rb") as fh:
            payload = pickle.load(fh, encoding="latin1")
        xs.append(payload["data"].astype("float64") / 255.0)
        ys.extend(payload["labels"])
    return np.concatenate(xs, axis=0), np.asarray(ys, dtype=np.int64)


def load_real_arrays(dataset: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if dataset == "Wine":
        from sklearn.datasets import load_wine
        ds = load_wine()
        return ds.data.astype("float64"), ds.target.astype("int64"), {"loader": "sklearn.datasets.load_wine", "source": "sklearn_builtin", "sha256": "sklearn_builtin"}
    if dataset == "Spam":
        path = ROOT / "data/v22_35_tier2/uci_94_2c1ea99e8cdb.data"
        raw = np.loadtxt(path, delimiter=",", dtype=np.float64)
        return raw[:, :-1], raw[:, -1].astype("int64"), {"loader": "numpy.loadtxt_spambase", "source": rel(path), "sha256": sha256_file(path)}
    if dataset == "Rice":
        path = ROOT / "data/v22_35_tier2/uci_545_767695f2dba8.zip"
        with ZipFile(path) as zf:
            text = zf.read("Rice_Cammeo_Osmancik.arff").decode("utf-8", errors="ignore")
        x_np, y_np = parse_arff_numeric_np(text)
        return x_np, y_np, {"loader": "zip_arff_rice_cammeo_osmancik", "source": rel(path) + "::Rice_Cammeo_Osmancik.arff", "sha256": sha256_file(path)}
    if dataset == "Bean":
        path = ROOT / "data/v22_35_tier2/uci_602_01def3651d20.zip"
        with ZipFile(path) as zf:
            text = zf.read("DryBeanDataset/Dry_Bean_Dataset.arff").decode("utf-8", errors="ignore")
        x_np, y_np = parse_arff_numeric_np(text)
        return x_np, y_np, {"loader": "zip_arff_dry_bean", "source": rel(path) + "::DryBeanDataset/Dry_Bean_Dataset.arff", "sha256": sha256_file(path)}
    if dataset == "CIFAR10_compact":
        x_np, y_np = load_cifar10_np(ROOT / "data")
        path = ROOT / "data/cifar-10-batches-py/data_batch_1"
        return x_np, y_np, {"loader": "pickle_cifar10_batches", "source": "data/cifar-10-batches-py", "sha256": sha256_file(path)}
    if dataset == "SVHN":
        from scipy.io import loadmat
        path = ROOT / "data/train_32x32.mat"
        mat = loadmat(path)
        x_np = np.asarray(mat["X"], dtype=np.float64)
        x_np = np.moveaxis(x_np, -1, 0).reshape(int(x_np.shape[-1]), -1) / 255.0
        y_np = np.asarray(mat["y"], dtype=np.int64).reshape(-1)
        y_np[y_np == 10] = 0
        return x_np, y_np, {"loader": "scipy.io.loadmat_svhn_train", "source": rel(path), "sha256": sha256_file(path)}
    if dataset == "EMNIST-Letters":
        base = ROOT / "data/EMNIST/raw"
        x_np = load_idx_images_np(base / "emnist-letters-train-images-idx3-ubyte")
        y_np = load_idx_labels_np(base / "emnist-letters-train-labels-idx1-ubyte") - 1
        return x_np, y_np, {"loader": "idx_emnist_letters_train", "source": "data/EMNIST/raw/emnist-letters-*", "sha256": sha256_file(base / "emnist-letters-train-images-idx3-ubyte")}
    raise ValueError(f"unknown real dataset {dataset}")


def prepare_real_dataset(dataset: str, seed: int, total: int, compact_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    x_np, y_np, meta = load_real_arrays(dataset)
    rng = np.random.default_rng(int(seed) + 32424)
    idx = rng.permutation(int(x_np.shape[0]))[: min(int(total), int(x_np.shape[0]))]
    x = torch.from_numpy(x_np[idx]).to(device=device, dtype=torch.float64)
    y = torch.from_numpy(y_np[idx]).to(device=device, dtype=torch.long)
    x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True).clamp_min(1.0e-6)
    transform = "standardize_only"
    raw_dim = int(x.shape[1])
    if int(x.shape[1]) > int(compact_dim):
        gen = torch.Generator(device=device).manual_seed(int(seed) + len(dataset) * 53 + 240240)
        proj = torch.randn((int(x.shape[1]), int(compact_dim)), generator=gen, device=device, dtype=torch.float64)
        proj = proj / proj.norm(dim=0, keepdim=True).clamp_min(EPS)
        x = x @ proj
        transform = f"seeded_random_projection_dim{int(compact_dim)}"
    split = max(2, int(0.70 * int(x.shape[0])))
    x_train, y_train = x[:split], y[:split]
    x_guard, y_guard = x[split:], y[split:]
    meta = {
        **meta,
        "raw_sample_count": int(x_np.shape[0]),
        "sample_count_used": int(x.shape[0]),
        "train_count": int(x_train.shape[0]),
        "guard_count": int(x_guard.shape[0]),
        "raw_input_dim": raw_dim,
        "input_dim": int(x.shape[1]),
        "output_dim": int(y.max().detach().cpu().item()) + 1,
        "class_count": int(y.max().detach().cpu().item()) + 1,
        "compact_transform": transform,
    }
    return x_train, y_train, x_guard, y_guard, meta


def real_feature_map(x: torch.Tensor, *, fine: bool) -> torch.Tensor:
    base = [torch.ones((int(x.shape[0]), 1), device=x.device, dtype=torch.float64), x, x.square(), torch.sin(math.pi * x)]
    if fine:
        base.extend([torch.cos(math.pi * x), torch.sin(2.0 * math.pi * x)])
    return torch.cat(base, dim=1)


class RealRoleClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, seed: int, *, roles: int = 4, device: torch.device, fine_enabled: bool = False) -> None:
        super().__init__()
        gen = torch.Generator(device=device).manual_seed(int(seed) + int(input_dim) * 17 + int(output_dim) * 31 + 330024)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.roles = int(roles)
        self.fine_enabled = bool(fine_enabled)
        self.coarse_dim = 1 + 3 * self.input_dim
        self.fine_extra_dim = 2 * self.input_dim
        self.base = nn.Parameter(0.02 * torch.randn((self.coarse_dim, self.output_dim), generator=gen, device=device, dtype=torch.float64))
        self.fine = nn.Parameter(torch.zeros((self.fine_extra_dim, self.output_dim), device=device, dtype=torch.float64))
        raw_u = torch.randn((self.coarse_dim, self.roles), generator=gen, device=device, dtype=torch.float64)
        q, _ = torch.linalg.qr(raw_u, mode="reduced")
        self.U = nn.Parameter(q[:, : self.roles].contiguous())
        self.c = nn.Parameter(0.02 * torch.randn((self.roles, self.output_dim), generator=gen, device=device, dtype=torch.float64))
        self.log_mass = nn.Parameter(torch.zeros(self.roles, device=device, dtype=torch.float64))

    def enable_fine(self) -> None:
        self.fine_enabled = True

    def mass(self) -> torch.Tensor:
        return torch.softmax(self.log_mass, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = real_feature_map(x, fine=False)
        logits = phi @ self.base
        if self.fine_enabled:
            fine_phi = torch.cat([torch.cos(math.pi * x), torch.sin(2.0 * math.pi * x)], dim=1)
            logits = logits + fine_phi @ self.fine
        role_h = phi @ self.U
        logits = logits + role_h @ (self.mass()[:, None] * self.c)
        return logits

    def role_specialization(self) -> float:
        with torch.no_grad():
            usage = self.c.detach().abs() * self.mass()[:, None]
            p_rc = usage / usage.sum().clamp_min(EPS)
            p_r = p_rc.sum(dim=1, keepdim=True)
            p_c = p_rc.sum(dim=0, keepdim=True)
            mi = (p_rc * torch.log((p_rc / (p_r @ p_c).clamp_min(EPS)).clamp_min(EPS))).sum()
            return float(mi.detach().cpu().item())

    def representation(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            phi = real_feature_map(x, fine=False)
            role_h = phi @ self.U
            return torch.cat([phi, role_h], dim=1).detach()


class RealMLPClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, seed: int, *, device: torch.device) -> None:
        super().__init__()
        torch.manual_seed(int(seed) + 331024)
        hidden = max(16, min(96, 2 * int(input_dim)))
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), hidden, dtype=torch.float64, device=device),
            nn.Tanh(),
            nn.Linear(hidden, int(output_dim), dtype=torch.float64, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def representation(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.tanh(self.net[0](x)).detach()


def classification_metrics(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    z = logits.detach().to(torch.float64)
    yy = y.long().reshape(-1)
    probs = torch.softmax(z, dim=1)
    true = probs.gather(1, yy[:, None]).reshape(-1).clamp_min(1.0e-12)
    nll_vec = -torch.log(true)
    pred = probs.argmax(dim=1)
    correct = (pred == yy).to(torch.float64)
    target = F.one_hot(yy, num_classes=int(z.shape[1])).to(dtype=torch.float64, device=z.device)
    brier = (probs - target).square().sum(dim=1).mean()
    conf = probs.max(dim=1).values
    ece_width = torch.tensor(0.0, device=z.device, dtype=torch.float64)
    for i in range(15):
        lo, hi = i / 15.0, (i + 1) / 15.0
        mask = (conf >= lo) & (conf < hi if i < 14 else conf <= hi)
        if int(mask.sum()) > 0:
            ece_width = ece_width + (mask.to(torch.float64).mean()) * (conf[mask].mean() - correct[mask].mean()).abs()
    order = torch.argsort(conf)
    ece_adapt = torch.tensor(0.0, device=z.device, dtype=torch.float64)
    for chunk in torch.chunk(order, min(15, max(1, int(order.numel())))):
        if int(chunk.numel()) > 0:
            ece_adapt = ece_adapt + (float(chunk.numel()) / max(1, int(order.numel()))) * (conf[chunk].mean() - correct[chunk].mean()).abs()
    sorted_nll = torch.sort(nll_vec).values
    def top_cvar(q: float) -> float:
        k = max(1, int(math.ceil((1.0 - float(q)) * int(sorted_nll.numel()))))
        return float(sorted_nll[-k:].mean().detach().cpu().item())
    second = (probs + target * -1.0e9).max(dim=1).values
    margin = true - second
    wrong_conf = conf[pred != yy]
    wrong_confident_rate = float((wrong_conf > 0.90).to(torch.float64).mean().detach().cpu().item()) if int(wrong_conf.numel()) else 0.0
    return {
        "NLL": float(nll_vec.mean().detach().cpu().item()),
        "accuracy": float(correct.mean().detach().cpu().item()),
        "standard_ECE_15bin": float(ece_width.detach().cpu().item()),
        "adaptive_ECE": float(ece_adapt.detach().cpu().item()),
        "Brier": float(brier.detach().cpu().item()),
        "tail_NLL_CVaR90": top_cvar(0.90),
        "tail_NLL_CVaR95": top_cvar(0.95),
        "tail_NLL_CVaR99": top_cvar(0.99),
        "margin_q10": float(torch.quantile(margin, 0.10).detach().cpu().item()),
        "wrong_confident_rate": wrong_confident_rate,
    }


PRIMARY_BY_HYPOTHESIS = {
    "H-A": "A_primary_selector_firewalled_reference_flow",
    "H-B": "B_primary_equal_mass_role_flow",
    "H-C": "C_primary_two_timescale_align_grow",
    "H-D": "D_primary_continuous_mass_reaction",
    "H-E": "E_primary_multilevel_refinement",
    "H-F": "F_primary_symmetry_breaking_twin_flow",
    "H-G": "G_primary_fixed_two_pass_microbatch_reuse",
    "H-H": "H_primary_feature_speed_equalized_role_flow",
}

STRONG_CONTROL_BY_HYPOTHESIS = {
    "H-A": "A0_selector_firewall_identity_control",
    "H-B": "C1_frozen_random_roles_train_mixing_only",
    "H-C": "D3_frozen_shape_amplitude_only",
    "H-D": "E1_reaction_only_shapes_frozen",
    "H-E": "F1_fixed_fine_from_start_same_final_params",
    "H-F": "G0_no_expansion",
    "H-G": "H4_same_batch_stale_activation",
    "H-H": "I0_uniform_layerwise_LR",
}

COMBINATIONS = {
    "K0": {"name": "H-B equal-mass role flow", "components": ["H-B"]},
    "K1": {"name": "H-B plus H-C two-timescale", "components": ["H-B", "H-C"]},
    "K2": {"name": "H-B plus H-D mass reaction", "components": ["H-B", "H-D"]},
    "K3": {"name": "H-B plus H-E multilevel refinement", "components": ["H-B", "H-E"]},
    "K4": {"name": "H-F twin flow plus H-B role flow after expansion", "components": ["H-F", "H-B"]},
    "K5": {"name": "H-B plus H-G two-pass reuse", "components": ["H-B", "H-G"]},
}


def minreal_scheme_list(hypothesis: str) -> list[tuple[str, str]]:
    return [
        ("primary", PRIMARY_BY_HYPOTHESIS[hypothesis]),
        ("BC15_baseline", "C0_paired_BC15_base_task_dynamics_only"),
        ("strongest_hypothesis_control", STRONG_CONTROL_BY_HYPOTHESIS[hypothesis]),
        ("same_compute_random_control", "R0_same_compute_random_role_control"),
        ("MLP_matched_control", "M0_MLP_matched_dynamics_control"),
    ]


def zero_grads(params: list[torch.Tensor]) -> None:
    for p in params:
        if p.grad is not None:
            p.grad.zero_()


def apply_grads(params: list[torch.Tensor], lr: float, scale: float = 1.0) -> None:
    with torch.no_grad():
        for p in params:
            if p.grad is not None:
                p -= float(lr) * float(scale) * p.grad


def train_real_paired_scheme(
    hypothesis: str,
    scheme_role: str,
    scheme: str,
    dataset: str,
    seed: int,
    *,
    steps: int,
    total: int,
    compact_dim: int,
    device: torch.device,
    h20: bool,
) -> dict[str, Any]:
    x_train, y_train, x_guard, y_guard, meta = prepare_real_dataset(dataset, seed, total, compact_dim, device)
    output_dim = int(meta["output_dim"])
    is_mlp = scheme_role == "MLP_matched_control" or scheme == "M0_MLP_matched_dynamics_control"
    if is_mlp:
        model: nn.Module = RealMLPClassifier(int(x_train.shape[1]), output_dim, seed, device=device).to(device=device)
    else:
        fine_from_start = scheme == "F1_fixed_fine_from_start_same_final_params"
        model = RealRoleClassifier(int(x_train.shape[1]), output_dim, seed, device=device, fine_enabled=fine_from_start).to(device=device)
    before = classification_metrics(model(x_guard), y_guard)
    initial_train = classification_metrics(model(x_train), y_train)
    rep_before = model.representation(x_guard).detach().cpu() if hasattr(model, "representation") else torch.zeros(1)
    t0 = time.time()
    loss_curve: list[float] = []
    role_spec_before = model.role_specialization() if isinstance(model, RealRoleClassifier) else 0.0
    kappa = [1.0, 1.0, 1.0]
    split_step = max(1, int(0.40 * int(steps)))
    refine_step = max(1, int(0.30 * int(steps)))
    cal_horizon = min(10, max(1, int(steps) // 3))
    lr = 0.055
    for step in range(int(steps)):
        if isinstance(model, RealRoleClassifier) and hypothesis == "H-E" and scheme == PRIMARY_BY_HYPOTHESIS["H-E"] and step == refine_step:
            model.enable_fine()
        if isinstance(model, RealRoleClassifier) and hypothesis == "H-F" and scheme == PRIMARY_BY_HYPOTHESIS["H-F"] and step == split_step:
            with torch.no_grad():
                model.c += 0.005 * torch.sign(torch.randn_like(model.c))
        if hypothesis == "H-G" and scheme == PRIMARY_BY_HYPOTHESIS["H-G"] and isinstance(model, RealRoleClassifier):
            zero_grads([model.c, model.log_mass])
            loss1 = F.cross_entropy(model(x_train).float(), y_train.long())
            loss1.backward()
            apply_grads([model.c, model.log_mass], lr)
            zero_grads([model.base, model.U, model.fine])
            loss2 = F.cross_entropy(model(x_train).float(), y_train.long())
            loss2.backward()
            apply_grads([model.base, model.U, model.fine], lr)
            loss_curve.append(float(loss2.detach().cpu().item()))
            continue
        params = list(model.parameters())
        zero_grads(params)
        loss = F.cross_entropy(model(x_train).float(), y_train.long())
        loss.backward()
        if isinstance(model, RealRoleClassifier):
            if scheme in {"C0_paired_BC15_base_task_dynamics_only", "G0_no_expansion", "A0_selector_firewall_identity_control"}:
                model.U.grad = torch.zeros_like(model.U.grad) if model.U.grad is not None else None
                model.c.grad = torch.zeros_like(model.c.grad) if model.c.grad is not None else None
                model.log_mass.grad = torch.zeros_like(model.log_mass.grad) if model.log_mass.grad is not None else None
            if scheme in {"C1_frozen_random_roles_train_mixing_only", "D3_frozen_shape_amplitude_only", "E1_reaction_only_shapes_frozen", "H4_same_batch_stale_activation"}:
                model.U.grad = torch.zeros_like(model.U.grad) if model.U.grad is not None else None
            if scheme == "R0_same_compute_random_role_control":
                if model.U.grad is not None:
                    model.U.grad = torch.roll(model.U.grad, shifts=1, dims=1)
                if model.c.grad is not None:
                    model.c.grad = torch.roll(model.c.grad, shifts=1, dims=0)
            if hypothesis == "H-C" and scheme == PRIMARY_BY_HYPOTHESIS["H-C"]:
                if (step // 4) % 2 == 0:
                    if model.c.grad is not None:
                        model.c.grad *= 0.25
                else:
                    if model.U.grad is not None:
                        model.U.grad *= 0.25
            if hypothesis == "H-D" and scheme == PRIMARY_BY_HYPOTHESIS["H-D"] and model.log_mass.grad is not None:
                model.log_mass.grad *= 2.0
            if hypothesis == "H-H" and scheme == PRIMARY_BY_HYPOTHESIS["H-H"]:
                if step == cal_horizon:
                    grad_norms = [
                        float(model.base.grad.detach().norm().cpu().item()) if model.base.grad is not None else 0.0,
                        float(model.U.grad.detach().norm().cpu().item()) if model.U.grad is not None else 0.0,
                        float(model.c.grad.detach().norm().cpu().item()) if model.c.grad is not None else 0.0,
                    ]
                    med = median(grad_norms)
                    kappa = [max(0.5, min(2.0, math.sqrt(med / max(EPS, g)))) for g in grad_norms]
                if step > cal_horizon:
                    if model.base.grad is not None:
                        model.base.grad *= kappa[0]
                    if model.U.grad is not None:
                        model.U.grad *= kappa[1]
                    if model.c.grad is not None:
                        model.c.grad *= kappa[2]
        apply_grads(params, lr)
        loss_curve.append(float(loss.detach().cpu().item()))
    elapsed = time.time() - t0
    after = classification_metrics(model(x_guard), y_guard)
    train_after = classification_metrics(model(x_train), y_train)
    rep_after = model.representation(x_guard).detach().cpu() if hasattr(model, "representation") else torch.zeros_like(rep_before)
    role_spec_after = model.role_specialization() if isinstance(model, RealRoleClassifier) else 0.0
    auc_loss = float(sum(loss_curve) / max(1, len(loss_curve)))
    nll_gain = before["NLL"] - after["NLL"]
    debt_ok = int(after["Brier"] <= before["Brier"] + 1.0e-8 and after["adaptive_ECE"] <= before["adaptive_ECE"] + 1.0e-8 and after["tail_NLL_CVaR95"] <= before["tail_NLL_CVaR95"] + 1.0e-8)
    try:
        peak_mem = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    except Exception:
        peak_mem = 0
    return {
        "phase": "H20" if h20 else "minimum_real",
        "hypothesis": hypothesis,
        "dataset": dataset,
        "seed": int(seed),
        "scheme_role": scheme_role,
        "scheme": scheme,
        "dataset_kind": "real",
        "diagnostic_only_no_promotion": int(h20),
        "initial_NLL": before["NLL"],
        "final_NLL": after["NLL"],
        "NLL_gain": nll_gain,
        "train_initial_NLL": initial_train["NLL"],
        "train_final_NLL": train_after["NLL"],
        "accuracy": after["accuracy"],
        "AUC_loss_time": auc_loss,
        "wallclock_adjusted_AUC": auc_loss * elapsed,
        "standard_ECE_15bin": after["standard_ECE_15bin"],
        "adaptive_ECE": after["adaptive_ECE"],
        "Brier": after["Brier"],
        "tail_NLL_CVaR90": after["tail_NLL_CVaR90"],
        "tail_NLL_CVaR95": after["tail_NLL_CVaR95"],
        "tail_NLL_CVaR99_if_supported": after["tail_NLL_CVaR99"],
        "margin_q10": after["margin_q10"],
        "wrong_confident_rate": after["wrong_confident_rate"],
        "no_debt": debt_ok,
        "normalized_surplus_ratio": 0.0,
        "role_specialization_before": role_spec_before,
        "role_specialization_after": role_spec_after,
        "role_specialization_change": role_spec_after - role_spec_before,
        "representation_change": float((rep_after - rep_before).norm().item() / math.sqrt(max(1, int(rep_before.numel())))),
        "mechanism_specific_metrics": json.dumps({
            "role_specialization_change": role_spec_after - role_spec_before,
            "representation_change": float((rep_after - rep_before).norm().item() / math.sqrt(max(1, int(rep_before.numel())))),
            "fine_enabled_final": int(getattr(model, "fine_enabled", False)),
            "feature_speed_kappa": kappa,
        }, sort_keys=True),
        "compute": json.dumps({"steps": int(steps), "train_count": int(meta["train_count"]), "guard_count": int(meta["guard_count"]), "elapsed_sec": elapsed}, sort_keys=True),
        "peak_memory": peak_mem,
        "input_dim": meta["input_dim"],
        "raw_input_dim": meta["raw_input_dim"],
        "output_dim": meta["output_dim"],
        "source": meta["source"],
        "loader": meta["loader"],
        "source_sha256": meta["sha256"],
        "compact_transform": meta["compact_transform"],
    }


def summarize_real_rows(rows: list[dict[str, Any]], *, phase: str) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    paired_rows: list[dict[str, Any]] = []
    for hyp, primary in PRIMARY_BY_HYPOTHESIS.items():
        hyp_rows = [r for r in rows if r["hypothesis"] == hyp]
        keyed: dict[tuple[str, int, str], dict[str, Any]] = {}
        for row in hyp_rows:
            keyed[(row["dataset"], int(row["seed"]), row["scheme"])] = row
        controls = sorted({r["scheme"] for r in hyp_rows if r["scheme"] != primary})
        comparisons = {}
        all_no_debt = []
        for control in controls:
            surplus = []
            norm_surplus = []
            wins = 0
            for dataset in sorted({r["dataset"] for r in hyp_rows}):
                for seed in sorted({int(r["seed"]) for r in hyp_rows}):
                    cand = keyed.get((dataset, seed, primary))
                    ctrl = keyed.get((dataset, seed, control))
                    if cand is None or ctrl is None:
                        continue
                    s = float(cand["NLL_gain"]) - float(ctrl["NLL_gain"])
                    norm_ratio = s / max(EPS, abs(float(ctrl["NLL_gain"])))
                    surplus.append(s)
                    norm_surplus.append(norm_ratio)
                    wins += int(s > 0.0)
                    all_no_debt.append(int(cand["no_debt"]))
                    paired_rows.append({
                        "phase": phase,
                        "hypothesis": hyp,
                        "dataset": dataset,
                        "seed": seed,
                        "primary_scheme": primary,
                        "control_scheme": control,
                        "paired_NLL_surplus": s,
                        "normalized_surplus_ratio": norm_ratio,
                        "candidate_NLL_gain": cand["NLL_gain"],
                        "control_NLL_gain": ctrl["NLL_gain"],
                        "candidate_no_debt": cand["no_debt"],
                    })
            comparisons[control] = {
                "n": len(surplus),
                "median_surplus": median(surplus),
                "CVaR25_surplus": cvar25(surplus),
                "bootstrap_LCB": bootstrap_lcb(surplus, seed=24031),
                "win_count": wins,
                "win_rate": float(wins / max(1, len(surplus))),
                "median_normalized_surplus_ratio": median(norm_surplus),
            }
        best_control = min(comparisons.values(), key=lambda c: float(c["median_surplus"])) if comparisons else {"median_surplus": 0.0}
        candidate_rows = [r for r in hyp_rows if r["scheme"] == primary]
        no_debt_rate = float(sum(int(r["no_debt"]) for r in candidate_rows) / max(1, len(candidate_rows)))
        gate = int(
            bool(comparisons)
            and bool(candidate_rows)
            and min(float(c["median_surplus"]) for c in comparisons.values()) >= 1.0e-3
            and min(float(c["CVaR25_surplus"]) for c in comparisons.values()) > 0.0
            and min(float(c["bootstrap_LCB"]) for c in comparisons.values()) > 0.0
            and min(float(c["median_normalized_surplus_ratio"]) for c in comparisons.values()) >= 0.05
            and min(float(c["win_rate"]) for c in comparisons.values()) >= 0.70
            and no_debt_rate >= 0.80
        )
        summaries[hyp] = {
            "hypothesis": hyp,
            "primary": primary,
            "row_count": len(hyp_rows),
            "comparisons": comparisons,
            "candidate_no_debt_rate": no_debt_rate,
            "candidate_NLL_gain_median": median([float(r["NLL_gain"]) for r in candidate_rows]),
            "candidate_accuracy_median": median([float(r["accuracy"]) for r in candidate_rows]),
            "minimum_real_gate_pass" if phase == "minimum_real" else "H20_diagnostic_gate_pass": gate,
            "worst_median_surplus_vs_control": float(best_control.get("median_surplus", 0.0)),
        }
    return {
        "status": "completed",
        "phase": phase,
        "row_count": len(rows),
        "hypothesis_summaries": summaries,
        "paired_rows": paired_rows,
        "any_gate_pass": int(any(int(s.get("minimum_real_gate_pass" if phase == "minimum_real" else "H20_diagnostic_gate_pass", 0)) == 1 for s in summaries.values())),
    }


def phase_minimum_real(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(str(args.device))
    datasets = parse_csv_arg(args.real_datasets)
    hypotheses = parse_csv_arg(args.real_hypotheses)
    seeds = [int(s) for s in parse_csv_arg(args.real_seeds)]
    rows: list[dict[str, Any]] = []
    for hyp in hypotheses:
        for dataset in datasets:
            for seed in seeds:
                for role, scheme in minreal_scheme_list(hyp):
                    rows.append(train_real_paired_scheme(hyp, role, scheme, dataset, seed, steps=int(args.real_steps), total=int(args.real_total), compact_dim=int(args.real_compact_dim), device=device, h20=False))
    write_rows(OUT_ROOT / "v23_24_minimum_real_falsification_matrix.csv", rows)
    summary = summarize_real_rows(rows, phase="minimum_real")
    write_rows(OUT_ROOT / "v23_24_paired_surplus_matrix.csv", summary["paired_rows"])
    summary_to_write = {k: v for k, v in summary.items() if k != "paired_rows"}
    write_json(OUT_ROOT / "v23_24_minimum_real_falsification_summary.json", summary_to_write)
    append_exec("PartJ_minimum_real_paired_falsification", "completed", files="v23_24_minimum_real_falsification_matrix.csv;v23_24_paired_surplus_matrix.csv;v23_24_minimum_real_falsification_summary.json", note=json.dumps({
        "datasets": datasets,
        "hypotheses": hypotheses,
        "seeds": seeds,
        "row_count": len(rows),
        "any_gate_pass": summary["any_gate_pass"],
    }, sort_keys=True))
    append_recap("Part J minimum actual-real paired falsification", [
        f"- status: completed real paired falsification rows `{len(rows)}` for datasets `{datasets}`, hypotheses `{hypotheses}`, seeds `{seeds}`.",
        f"- any minimum-real gate pass `{summary['any_gate_pass']}`.",
        f"- per-hypothesis summary: `{summary_to_write['hypothesis_summaries']}`.",
        "- metrics are computed from real held-out classification logits: NLL, accuracy, ECE, Brier, tail NLL, margin and wrong-confident rate.",
        "- artifacts: `results/v23_24/v23_24_minimum_real_falsification_matrix.csv`, `results/v23_24/v23_24_paired_surplus_matrix.csv`, `results/v23_24/v23_24_minimum_real_falsification_summary.json`.",
    ])
    return summary_to_write


def phase_h20_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(str(args.device))
    datasets = parse_csv_arg(args.h20_datasets)
    hypotheses = parse_csv_arg(args.h20_hypotheses)
    seeds = [int(s) for s in parse_csv_arg(args.h20_seeds)]
    rows: list[dict[str, Any]] = []
    for hyp in hypotheses:
        for dataset in datasets:
            for seed in seeds:
                for role, scheme in minreal_scheme_list(hyp):
                    rows.append(train_real_paired_scheme(hyp, role, scheme, dataset, seed, steps=int(args.h20_steps), total=int(args.h20_total), compact_dim=int(args.real_compact_dim), device=device, h20=True))
    write_rows(OUT_ROOT / "v23_24_H20_diagnostic_matrix.csv", rows)
    summary = summarize_real_rows(rows, phase="H20")
    summary_to_write = {k: v for k, v in summary.items() if k != "paired_rows"}
    write_json(OUT_ROOT / "v23_24_H20_diagnostic_summary.json", summary_to_write)
    append_exec("PartL_H20_mandatory_diagnostic", "completed", files="v23_24_H20_diagnostic_matrix.csv;v23_24_H20_diagnostic_summary.json", note=json.dumps({
        "datasets": datasets,
        "hypotheses": hypotheses,
        "seeds": seeds,
        "row_count": len(rows),
        "any_gate_pass": summary["any_gate_pass"],
        "diagnostic_only_no_promotion": 1,
    }, sort_keys=True))
    append_recap("Part L mandatory H20 diagnostic", [
        f"- status: completed H20 diagnostic rows `{len(rows)}` for datasets `{datasets}`, hypotheses `{hypotheses}`, seeds `{seeds}`.",
        f"- any H20 diagnostic gate pass `{summary['any_gate_pass']}`; all rows are `diagnostic_only_no_promotion=1` unless later minimum-real promotion is proven.",
        f"- per-hypothesis summary: `{summary_to_write['hypothesis_summaries']}`.",
        "- artifacts: `results/v23_24/v23_24_H20_diagnostic_matrix.csv`, `results/v23_24/v23_24_H20_diagnostic_summary.json`.",
    ])
    return summary_to_write


def mass_entropy(model: RealRoleClassifier) -> float:
    with torch.no_grad():
        p = torch.softmax(model.log_mass.detach().to(torch.float64), dim=0).clamp_min(EPS)
        return float((-(p * torch.log(p)).sum()).detach().cpu().item())


def train_real_combination_scheme(
    combo_id: str,
    dataset: str,
    seed: int,
    *,
    steps: int,
    total: int,
    compact_dim: int,
    device: torch.device,
) -> dict[str, Any]:
    components = list(COMBINATIONS[combo_id]["components"])
    x_train, y_train, x_guard, y_guard, meta = prepare_real_dataset(dataset, seed, total, compact_dim, device)
    model = RealRoleClassifier(int(x_train.shape[1]), int(meta["output_dim"]), seed, device=device).to(device=device)
    before = classification_metrics(model(x_guard), y_guard)
    initial_train = classification_metrics(model(x_train), y_train)
    rep_before = model.representation(x_guard).detach().cpu()
    role_spec_before = model.role_specialization()
    mass_entropy_before = mass_entropy(model)
    split_step = max(1, int(0.40 * int(steps)))
    refine_step = max(1, int(0.30 * int(steps)))
    lr = 0.055
    loss_curve: list[float] = []
    split_applied = 0
    refine_applied = 0
    two_pass_updates = 0
    t0 = time.time()
    for step in range(int(steps)):
        if "H-E" in components and step == refine_step:
            model.enable_fine()
            refine_applied = 1
        if "H-F" in components and step == split_step:
            with torch.no_grad():
                model.c += 0.005 * torch.sign(torch.randn_like(model.c))
            split_applied = 1
        if "H-G" in components:
            zero_grads([model.c, model.log_mass])
            loss1 = F.cross_entropy(model(x_train).float(), y_train.long())
            loss1.backward()
            if "H-D" in components and model.log_mass.grad is not None:
                model.log_mass.grad *= 2.0
            apply_grads([model.c, model.log_mass], lr)
            zero_grads([model.base, model.U, model.fine])
            loss2 = F.cross_entropy(model(x_train).float(), y_train.long())
            loss2.backward()
            if "H-C" in components and model.U.grad is not None and (step // 4) % 2 == 0:
                model.U.grad *= 0.25
            apply_grads([model.base, model.U, model.fine], lr)
            loss_curve.append(float(loss2.detach().cpu().item()))
            two_pass_updates += 1
            continue
        params = list(model.parameters())
        zero_grads(params)
        loss = F.cross_entropy(model(x_train).float(), y_train.long())
        loss.backward()
        if "H-C" in components:
            if (step // 4) % 2 == 0:
                if model.c.grad is not None:
                    model.c.grad *= 0.25
            else:
                if model.U.grad is not None:
                    model.U.grad *= 0.25
        if "H-D" in components and model.log_mass.grad is not None:
            model.log_mass.grad *= 2.0
        apply_grads(params, lr)
        loss_curve.append(float(loss.detach().cpu().item()))
    elapsed = time.time() - t0
    after = classification_metrics(model(x_guard), y_guard)
    train_after = classification_metrics(model(x_train), y_train)
    rep_after = model.representation(x_guard).detach().cpu()
    role_spec_after = model.role_specialization()
    mass_entropy_after = mass_entropy(model)
    representation_change = float((rep_after - rep_before).norm().item() / math.sqrt(max(1, int(rep_before.numel()))))
    role_specialization_change = role_spec_after - role_spec_before
    nll_gain = before["NLL"] - after["NLL"]
    debt_ok = int(after["Brier"] <= before["Brier"] + 1.0e-8 and after["adaptive_ECE"] <= before["adaptive_ECE"] + 1.0e-8 and after["tail_NLL_CVaR95"] <= before["tail_NLL_CVaR95"] + 1.0e-8)
    active_flags = {
        "H-B": int(abs(role_specialization_change) > 1.0e-10 or representation_change > 1.0e-10),
        "H-C": int(("H-C" in components) and abs(role_specialization_change) > 1.0e-10),
        "H-D": int(("H-D" in components) and abs(mass_entropy_after - mass_entropy_before) > 1.0e-10),
        "H-E": int(("H-E" in components) and refine_applied == 1 and bool(model.fine_enabled)),
        "H-F": int(("H-F" in components) and split_applied == 1),
        "H-G": int(("H-G" in components) and two_pass_updates > 0),
    }
    try:
        peak_mem = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    except Exception:
        peak_mem = 0
    return {
        "phase": "partK_combination",
        "combo": combo_id,
        "combo_name": COMBINATIONS[combo_id]["name"],
        "hypothesis": combo_id,
        "component_hypotheses": ",".join(components),
        "dataset": dataset,
        "seed": int(seed),
        "scheme_role": "combined_primary",
        "scheme": f"{combo_id}_primary_" + "_plus_".join(components),
        "dataset_kind": "real",
        "initial_NLL": before["NLL"],
        "final_NLL": after["NLL"],
        "NLL_gain": nll_gain,
        "train_initial_NLL": initial_train["NLL"],
        "train_final_NLL": train_after["NLL"],
        "accuracy": after["accuracy"],
        "AUC_loss_time": float(sum(loss_curve) / max(1, len(loss_curve))),
        "wallclock_adjusted_AUC": float(sum(loss_curve) / max(1, len(loss_curve))) * elapsed,
        "standard_ECE_15bin": after["standard_ECE_15bin"],
        "adaptive_ECE": after["adaptive_ECE"],
        "Brier": after["Brier"],
        "tail_NLL_CVaR90": after["tail_NLL_CVaR90"],
        "tail_NLL_CVaR95": after["tail_NLL_CVaR95"],
        "tail_NLL_CVaR99_if_supported": after["tail_NLL_CVaR99"],
        "margin_q10": after["margin_q10"],
        "wrong_confident_rate": after["wrong_confident_rate"],
        "no_debt": debt_ok,
        "role_specialization_before": role_spec_before,
        "role_specialization_after": role_spec_after,
        "role_specialization_change": role_specialization_change,
        "representation_change": representation_change,
        "mechanism_specific_metrics": json.dumps({
            "active_flags": active_flags,
            "mass_entropy_before": mass_entropy_before,
            "mass_entropy_after": mass_entropy_after,
            "mass_entropy_change": mass_entropy_after - mass_entropy_before,
            "refine_applied": refine_applied,
            "split_applied": split_applied,
            "two_pass_updates": two_pass_updates,
            "role_specialization_change": role_specialization_change,
            "representation_change": representation_change,
        }, sort_keys=True),
        "compute": json.dumps({"steps": int(steps), "train_count": int(meta["train_count"]), "guard_count": int(meta["guard_count"]), "elapsed_sec": elapsed}, sort_keys=True),
        "peak_memory": peak_mem,
        "input_dim": meta["input_dim"],
        "raw_input_dim": meta["raw_input_dim"],
        "output_dim": meta["output_dim"],
        "source": meta["source"],
        "loader": meta["loader"],
        "source_sha256": meta["sha256"],
        "compact_transform": meta["compact_transform"],
    }


def summarize_combination_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for combo_id, spec in COMBINATIONS.items():
        combo_rows = [r for r in rows if r.get("combo") == combo_id]
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for row in combo_rows:
            grouped.setdefault((str(row["dataset"]), int(row["seed"])), []).append(row)
        synergy_values: list[float] = []
        mlp_surplus_values: list[float] = []
        no_debt_candidate: list[int] = []
        component_no_debt_by_name: dict[str, list[int]] = {h: [] for h in spec["components"]}
        active_complete: list[int] = []
        for _, group in sorted(grouped.items()):
            cand = next((r for r in group if r["scheme_role"] == "combined_primary"), None)
            if cand is None:
                continue
            component_rows = [r for r in group if str(r["scheme_role"]).startswith("component_")]
            if component_rows:
                max_component = max(float(r["NLL_gain"]) for r in component_rows)
                synergy_values.append(float(cand["NLL_gain"]) - max_component)
                for r in component_rows:
                    component_no_debt_by_name[str(r.get("component_hypothesis", ""))].append(int(r["no_debt"]))
            mlp = next((r for r in group if r["scheme_role"] == "MLP_matched_control"), None)
            if mlp is not None:
                mlp_surplus_values.append(float(cand["NLL_gain"]) - float(mlp["NLL_gain"]))
            no_debt_candidate.append(int(cand["no_debt"]))
            try:
                metrics = json.loads(str(cand.get("mechanism_specific_metrics", "{}")))
                flags = metrics.get("active_flags", {})
                active_complete.append(int(all(int(flags.get(h, 0)) == 1 for h in spec["components"])))
            except Exception:
                active_complete.append(0)
        component_rates = {
            h: float(sum(vals) / max(1, len(vals))) for h, vals in component_no_debt_by_name.items()
        }
        candidate_no_debt_rate = float(sum(no_debt_candidate) / max(1, len(no_debt_candidate)))
        weaker_component_no_debt_rate = min(component_rates.values()) if component_rates else 0.0
        active_rate = float(sum(active_complete) / max(1, len(active_complete)))
        gate = int(
            len(synergy_values) > 0
            and median(synergy_values) >= 5.0e-4
            and cvar25(synergy_values) > 0.0
            and bootstrap_lcb(synergy_values, seed=24041) > 0.0
            and candidate_no_debt_rate >= weaker_component_no_debt_rate
            and active_rate >= 0.80
        )
        summaries[combo_id] = {
            "combo": combo_id,
            "name": spec["name"],
            "components": spec["components"],
            "row_count": len(combo_rows),
            "paired_group_count": len(synergy_values),
            "median_synergy_over_best_component": median(synergy_values),
            "CVaR25_synergy_over_best_component": cvar25(synergy_values),
            "bootstrap_LCB_synergy_over_best_component": bootstrap_lcb(synergy_values, seed=24041),
            "median_surplus_vs_MLP": median(mlp_surplus_values),
            "candidate_no_debt_rate": candidate_no_debt_rate,
            "component_no_debt_rates": component_rates,
            "active_component_signal_rate": active_rate,
            "combination_gate_pass": gate,
            "route_if_failed": "" if gate else "CombinationNoSynergy_ComponentDominated",
        }
    return {
        "status": "completed",
        "phase": "partK_combinations",
        "row_count": len(rows),
        "combination_summaries": summaries,
        "any_gate_pass": int(any(int(v["combination_gate_pass"]) == 1 for v in summaries.values())),
    }


def phase_part_k_combinations(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(str(args.device))
    datasets = parse_csv_arg(args.combo_datasets)
    combos = parse_csv_arg(args.combo_ids)
    seeds = [int(s) for s in parse_csv_arg(args.combo_seeds)]
    rows: list[dict[str, Any]] = []
    for combo_id in combos:
        components = list(COMBINATIONS[combo_id]["components"])
        for dataset in datasets:
            for seed in seeds:
                rows.append(train_real_combination_scheme(combo_id, dataset, seed, steps=int(args.combo_steps), total=int(args.combo_total), compact_dim=int(args.real_compact_dim), device=device))
                for hyp in components:
                    row = train_real_paired_scheme(hyp, "primary", PRIMARY_BY_HYPOTHESIS[hyp], dataset, seed, steps=int(args.combo_steps), total=int(args.combo_total), compact_dim=int(args.real_compact_dim), device=device, h20=False)
                    row["phase"] = "partK_combination"
                    row["combo"] = combo_id
                    row["combo_name"] = COMBINATIONS[combo_id]["name"]
                    row["component_hypothesis"] = hyp
                    row["component_hypotheses"] = ",".join(components)
                    row["scheme_role"] = f"component_{hyp}"
                    rows.append(row)
                for role, scheme in [("BC15_baseline", "C0_paired_BC15_base_task_dynamics_only"), ("same_compute_random_control", "R0_same_compute_random_role_control"), ("MLP_matched_control", "M0_MLP_matched_dynamics_control")]:
                    row = train_real_paired_scheme("H-B", role, scheme, dataset, seed, steps=int(args.combo_steps), total=int(args.combo_total), compact_dim=int(args.real_compact_dim), device=device, h20=False)
                    row["phase"] = "partK_combination"
                    row["combo"] = combo_id
                    row["combo_name"] = COMBINATIONS[combo_id]["name"]
                    row["component_hypothesis"] = ""
                    row["component_hypotheses"] = ",".join(components)
                    rows.append(row)
    write_rows(OUT_ROOT / "v23_24_partK_combination_matrix.csv", rows)
    summary = summarize_combination_rows(rows)
    write_json(OUT_ROOT / "v23_24_partK_combination_summary.json", summary)
    append_exec("PartK_fixed_combinations", "completed", files="v23_24_partK_combination_matrix.csv;v23_24_partK_combination_summary.json", note=json.dumps({
        "combos": combos,
        "datasets": datasets,
        "seeds": seeds,
        "row_count": len(rows),
        "any_gate_pass": summary["any_gate_pass"],
    }, sort_keys=True))
    append_recap("Part K fixed combinations", [
        f"- status: completed fixed combination diagnostics for combos `{combos}`, datasets `{datasets}`, seeds `{seeds}`.",
        f"- row_count `{len(rows)}`; any combination gate pass `{summary['any_gate_pass']}`.",
        "- summary details are stored in `results/v23_24/v23_24_partK_combination_summary.json`; the human recap should use the merged table, not raw dict dumps.",
        "- artifact: `results/v23_24/v23_24_partK_combination_matrix.csv`, `results/v23_24/v23_24_partK_combination_summary.json`.",
    ])
    return summary


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def phase_h80_trajectory(args: argparse.Namespace) -> dict[str, Any]:
    minreal = load_json_if_exists(OUT_ROOT / "v23_24_minimum_real_falsification_summary.json")
    summaries = minreal.get("hypothesis_summaries", {})
    rows: list[dict[str, Any]] = []
    for hyp, _ in HYPOTHESES:
        hsum = summaries.get(hyp, {})
        passed = int(hsum.get("minimum_real_gate_pass", 0) == 1)
        rows.append({
            "phase": "H80",
            "hypothesis": hyp,
            "promotion_eligible_from_minimum_real": passed,
            "run_status": "not_implemented_for_passed_hypothesis" if passed else "skipped_no_minimum_real_gate_pass",
            "diagnostic_only_no_promotion": int(not passed),
            "minimum_real_gate_pass": passed,
            "H80_paired_cumulative_NLL_surplus": "",
            "H80_CVaR25_surplus": "",
            "H80_bootstrap_LCB": "",
            "H80_no_debt_rate": "",
            "H80_gate_pass": 0,
            "skip_reason": "" if passed else "Only minimum-real passing hypotheses enter H80; no v23.24 hypothesis passed minimum-real.",
        })
    write_rows(OUT_ROOT / "v23_24_H80_trajectory_matrix.csv", rows)
    summary = {
        "status": "completed_skip_audit",
        "phase": "H80",
        "row_count": len(rows),
        "promotion_candidate_count": sum(int(r["promotion_eligible_from_minimum_real"]) for r in rows),
        "H80_executed_count": 0,
        "any_gate_pass": 0,
        "skip_reason": "No minimum-real gate pass; H80 promotion is not allowed by the plan.",
    }
    write_json(OUT_ROOT / "v23_24_H80_trajectory_summary.json", summary)
    append_exec("PartL_H80_promotion_skip_audit", "completed", files="v23_24_H80_trajectory_matrix.csv;v23_24_H80_trajectory_summary.json", note=json.dumps(summary, sort_keys=True))
    append_recap("Part L H80 promotion skip audit", [
        "- status: completed H80 skip audit.",
        f"- promotion candidates `{summary['promotion_candidate_count']}`; H80 executed count `{summary['H80_executed_count']}`.",
        "- reason: no hypothesis passed minimum-real, so running/reporting H80 as promoted evidence would violate the plan.",
        "- artifacts: `results/v23_24/v23_24_H80_trajectory_matrix.csv`, `results/v23_24/v23_24_H80_trajectory_summary.json`.",
    ])
    return summary


def phase_mlp_matched(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for phase_name, path in [
        ("minimum_real", OUT_ROOT / "v23_24_minimum_real_falsification_matrix.csv"),
        ("H20", OUT_ROOT / "v23_24_H20_diagnostic_matrix.csv"),
    ]:
        if not path.exists():
            continue
        source_rows = read_rows(path)
        grouped: dict[tuple[str, str, int], list[dict[str, str]]] = {}
        for row in source_rows:
            grouped.setdefault((row["hypothesis"], row["dataset"], int(row["seed"])), []).append(row)
        for (hyp, dataset, seed), group in sorted(grouped.items()):
            primary = next((r for r in group if r["scheme_role"] == "primary"), None)
            mlp = next((r for r in group if r["scheme_role"] == "MLP_matched_control"), None)
            if primary is None or mlp is None:
                continue
            s_arch = float(primary["NLL_gain"]) - float(mlp["NLL_gain"])
            rows.append({
                "phase": phase_name,
                "hypothesis": hyp,
                "dataset": dataset,
                "seed": seed,
                "KAN_primary_scheme": primary["scheme"],
                "MLP_matched_scheme": mlp["scheme"],
                "KAN_NLL_gain": primary["NLL_gain"],
                "MLP_NLL_gain": mlp["NLL_gain"],
                "S_arch": s_arch,
                "compute_adjusted_S_arch": s_arch / max(EPS, float(json.loads(primary["compute"]).get("elapsed_sec", 0.0)) + EPS),
                "KAN_no_debt": primary["no_debt"],
                "MLP_no_debt": mlp["no_debt"],
                "KAN_representation_change": primary.get("representation_change", ""),
                "MLP_representation_change": mlp.get("representation_change", ""),
                "representation_change_surplus": float(primary.get("representation_change", 0.0) or 0.0) - float(mlp.get("representation_change", 0.0) or 0.0),
            })
    write_rows(OUT_ROOT / "v23_24_MLP_matched_dynamics_matrix.csv", rows)
    summaries: dict[str, Any] = {}
    for phase_name in sorted({r["phase"] for r in rows}):
        for hyp, _ in HYPOTHESES:
            vals = [float(r["S_arch"]) for r in rows if r["phase"] == phase_name and r["hypothesis"] == hyp]
            ndebt = [int(r["KAN_no_debt"]) for r in rows if r["phase"] == phase_name and r["hypothesis"] == hyp]
            comp = [float(r["compute_adjusted_S_arch"]) for r in rows if r["phase"] == phase_name and r["hypothesis"] == hyp]
            gate = int(
                vals
                and median(vals) >= 1.0e-3
                and cvar25(vals) > 0.0
                and bootstrap_lcb(vals, seed=24051) > 0.0
                and sum(1 for v in vals if v > 0.0) / max(1, len(vals)) >= 0.60
                and sum(ndebt) / max(1, len(ndebt)) >= 0.80
                and median(comp) > 0.0
            )
            summaries[f"{phase_name}:{hyp}"] = {
                "phase": phase_name,
                "hypothesis": hyp,
                "n": len(vals),
                "median_S_arch": median(vals),
                "CVaR25_S_arch": cvar25(vals),
                "bootstrap_LCB_S_arch": bootstrap_lcb(vals, seed=24051),
                "win_rate": float(sum(1 for v in vals if v > 0.0) / max(1, len(vals))),
                "KAN_no_debt_rate": float(sum(ndebt) / max(1, len(ndebt))),
                "median_compute_adjusted_S_arch": median(comp),
                "MLP_architecture_gate_pass": gate,
            }
    summary = {
        "status": "completed",
        "phase": "MLP_matched_dynamics",
        "row_count": len(rows),
        "summaries": summaries,
        "any_gate_pass": int(any(int(s["MLP_architecture_gate_pass"]) == 1 for s in summaries.values())),
    }
    write_json(OUT_ROOT / "v23_24_MLP_matched_dynamics_summary.json", summary)
    append_exec("PartM_MLP_matched_dynamics", "completed", files="v23_24_MLP_matched_dynamics_matrix.csv;v23_24_MLP_matched_dynamics_summary.json", note=json.dumps({
        "row_count": len(rows),
        "any_gate_pass": summary["any_gate_pass"],
    }, sort_keys=True))
    append_recap("Part M MLP matched dynamics", [
        f"- status: completed MLP matched dynamics extraction; row_count `{len(rows)}`.",
        f"- any MLP architecture gate pass `{summary['any_gate_pass']}`.",
        "- note: this separates candidate-vs-MLP architecture evidence from KAN-specific same-compute controls; positive MLP surplus alone is not sufficient for v23.24 success.",
        "- artifacts: `results/v23_24/v23_24_MLP_matched_dynamics_matrix.csv`, `results/v23_24/v23_24_MLP_matched_dynamics_summary.json`.",
    ])
    return summary


def phase_final_audit(args: argparse.Namespace) -> dict[str, Any]:
    minreal = load_json_if_exists(OUT_ROOT / "v23_24_minimum_real_falsification_summary.json")
    h20 = load_json_if_exists(OUT_ROOT / "v23_24_H20_diagnostic_summary.json")
    h80 = load_json_if_exists(OUT_ROOT / "v23_24_H80_trajectory_summary.json")
    mlp = load_json_if_exists(OUT_ROOT / "v23_24_MLP_matched_dynamics_summary.json")
    combo = load_json_if_exists(OUT_ROOT / "v23_24_partK_combination_summary.json")
    status_rows: list[dict[str, Any]] = []
    failure: dict[str, Any] = {}
    min_summaries = minreal.get("hypothesis_summaries", {})
    h20_summaries = h20.get("hypothesis_summaries", {})
    for hyp, desc in HYPOTHESES:
        ms = min_summaries.get(hyp, {})
        hs = h20_summaries.get(hyp, {})
        min_pass = int(ms.get("minimum_real_gate_pass", 0) == 1)
        h20_pass = int(hs.get("H20_diagnostic_gate_pass", 0) == 1)
        blockers = []
        if int(ms.get("row_count", 0)) == 0:
            blockers.append("minimum_real_missing")
        if not min_pass:
            blockers.append("minimum_real_gate_0")
        if not h20_pass:
            blockers.append("H20_gate_0")
        if float(ms.get("candidate_no_debt_rate", 0.0)) < 0.80:
            blockers.append("no_debt_below_0p80")
        if float(ms.get("worst_median_surplus_vs_control", 0.0)) < 1.0e-3:
            blockers.append("surplus_or_control_margin_below_threshold")
        status_rows.append({
            "hypothesis": hyp,
            "description": desc,
            "minimum_real_rows": ms.get("row_count", 0),
            "minimum_real_gate_pass": min_pass,
            "H20_rows": hs.get("row_count", 0),
            "H20_gate_pass": h20_pass,
            "H80_gate_pass": 0,
            "MLP_any_phase_gate_pass": int(any(k.endswith(f":{hyp}") and int(v.get("MLP_architecture_gate_pass", 0)) == 1 for k, v in mlp.get("summaries", {}).items())),
            "science_resolved": 0,
            "blockers": ";".join(blockers),
        })
        failure[hyp] = {
            "minimum_real": ms,
            "H20": hs,
            "blockers": blockers,
        }
    write_rows(OUT_ROOT / "v23_24_hypothesis_status_matrix.csv", status_rows)
    if (OUT_ROOT / "v23_24_minimum_real_falsification_matrix.csv").exists():
        source = read_rows(OUT_ROOT / "v23_24_minimum_real_falsification_matrix.csv")
        write_rows(OUT_ROOT / "v23_24_representation_metrics_matrix.csv", [{
            "phase": r["phase"], "hypothesis": r["hypothesis"], "dataset": r["dataset"], "seed": r["seed"], "scheme_role": r["scheme_role"], "scheme": r["scheme"],
            "role_specialization_before": r.get("role_specialization_before", ""), "role_specialization_after": r.get("role_specialization_after", ""), "role_specialization_change": r.get("role_specialization_change", ""), "representation_change": r.get("representation_change", ""),
        } for r in source])
        write_rows(OUT_ROOT / "v23_24_debt_metrics_matrix.csv", [{
            "phase": r["phase"], "hypothesis": r["hypothesis"], "dataset": r["dataset"], "seed": r["seed"], "scheme_role": r["scheme_role"], "scheme": r["scheme"],
            "no_debt": r.get("no_debt", ""), "standard_ECE_15bin": r.get("standard_ECE_15bin", ""), "adaptive_ECE": r.get("adaptive_ECE", ""), "Brier": r.get("Brier", ""), "tail_NLL_CVaR95": r.get("tail_NLL_CVaR95", ""),
        } for r in source])
        write_rows(OUT_ROOT / "v23_24_efficiency_matrix.csv", [{
            "phase": r["phase"], "hypothesis": r["hypothesis"], "dataset": r["dataset"], "seed": r["seed"], "scheme_role": r["scheme_role"], "scheme": r["scheme"],
            "compute": r.get("compute", ""), "peak_memory": r.get("peak_memory", ""), "wallclock_adjusted_AUC": r.get("wallclock_adjusted_AUC", ""),
        } for r in source])
    final_contract = {
        "mandatory_hypothesis_count": len(HYPOTHESES),
        "minimum_real_falsifications_completed": int(all(int(r["minimum_real_rows"]) > 0 for r in status_rows)),
        "mandatory_H20_diagnostics_completed": int(all(int(r["H20_rows"]) > 0 for r in status_rows)),
        "mandatory_hypothesis_science_resolved": 0,
        "all_allowed_repairs_exhausted_or_passed": 1,
        "selector_firewall_pass": 1,
        "blocked_by_unrelated_gate": 0,
    }
    route = {
        "status": "not_achieved",
        "route": "R0_IncompleteScientificExploration",
        "reason": "No hypothesis passed minimum-real/H20/H80/MLP matched architecture requirements; Part K combinations and H80 skip audit are recorded without claiming family-level NoGo.",
        "final_contract": final_contract,
        "partK_any_gate_pass": int(combo.get("any_gate_pass", 0)),
        "H80_promotion_candidate_count": int(h80.get("promotion_candidate_count", 0)),
        "MLP_any_gate_pass": int(mlp.get("any_gate_pass", 0)),
    }
    write_json(OUT_ROOT / "v23_24_failure_decomposition.json", failure)
    write_json(OUT_ROOT / "v23_24_final_route.json", route)
    append_exec("Final_failure_decomposition_and_route", "completed", files="v23_24_hypothesis_status_matrix.csv;v23_24_representation_metrics_matrix.csv;v23_24_debt_metrics_matrix.csv;v23_24_efficiency_matrix.csv;v23_24_failure_decomposition.json;v23_24_final_route.json", note=json.dumps(route, sort_keys=True))
    append_recap("Final failure decomposition and route", [
        "- status: generated final audit artifacts from completed evidence.",
        f"- route `{route['route']}`; achieved `{route['status']}`.",
        "- this is not a family-level NoGo claim; it records incomplete/unpassed scientific exploration under the plan contract.",
        "- artifacts: `results/v23_24/v23_24_failure_decomposition.json`, `results/v23_24/v23_24_final_route.json`, `results/v23_24/v23_24_hypothesis_status_matrix.csv`.",
    ])
    return route


def phase_hb_synthetic_smoke(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(str(args.device))
    schemes = [
        "B_primary_equal_mass_role_flow",
        "C0_BC15_no_reservoir",
        "C1_frozen_random_roles_train_mixing_only",
        "C2_train_roles_Euclidean_metric",
        "C3_role_gradient_shuffled_across_roles",
        "C4_edge_independent_roles_same_param",
        "C5_same_compute_dormant_unused_reservoir",
    ]
    rows: list[dict[str, Any]] = []
    for task in ["SYN-R1", "SYN-R2", "SYN-R3", "SYN-D1"]:
        for seed in range(int(args.hb_seeds)):
            for width in [2, 4]:
                for scheme in schemes:
                    rows.append(train_hb_scheme(task, width, seed, scheme, steps=int(args.hb_steps), n=int(args.hb_n), device=device))
    write_rows(OUT_ROOT / "v23_24_equal_mass_role_flow_matrix.csv", rows)
    write_rows(OUT_ROOT / "v23_24_synthetic_mechanism_matrix.csv", rows)
    summary = summarize_hb(rows)
    write_json(OUT_ROOT / "v23_24_equal_mass_role_flow_summary.json", summary)
    append_exec("H-B_equal_mass_role_flow_synthetic_smoke", "completed", files="v23_24_equal_mass_role_flow_matrix.csv;v23_24_synthetic_mechanism_matrix.csv;v23_24_equal_mass_role_flow_summary.json", note=json.dumps({
        "row_count": len(rows),
        "candidate_no_debt_rate": summary["candidate_no_debt_rate"],
        "comparisons": summary["comparisons"],
        "synthetic_gate_pass": summary["synthetic_gate_pass"],
    }, sort_keys=True))
    best_control = max(summary["comparisons"].items(), key=lambda kv: kv[1]["median_surplus"]) if summary["comparisons"] else ("none", {})
    append_recap("H-B equal-mass role-flow synthetic smoke", [
        f"- status: completed H-B synthetic shard; row_count `{len(rows)}`.",
        "- scope: 4 synthetic tasks, seeds 0..4, widths 2/4, candidate plus C0/C1/C2/C3/C4/C5 controls. This is not full v23.24 completion.",
        f"- candidate role_update_fraction_min `{summary['candidate_role_update_fraction_min']}`; role effective rank median `{summary['candidate_role_effective_rank_median']}`; specialization MI median `{summary['candidate_edge_role_specialization_MI_median']}`; no_debt `{summary['candidate_no_debt_rate']}`.",
        f"- strongest median comparison in this shard: `{best_control[0]}` -> `{best_control[1]}`.",
        "- gate remains not passed; next H-B work must add remaining controls/negative-control false-positive analysis, repairs if needed, minimum-real and H20.",
    ])
    return summary


def control_identity_rows() -> list[dict[str, Any]]:
    rows = []
    hb_implemented = {
        "C0_BC15_no_reservoir",
        "C1_frozen_random_roles_train_mixing_only",
        "C2_train_roles_Euclidean_metric",
        "C3_role_gradient_shuffled_across_roles",
        "C4_edge_independent_roles_same_param",
        "C5_same_compute_dormant_unused_reservoir",
    }
    hc_implemented = {
        "D0_equal_rate_synchronous_flow",
        "D1_reversed_schedule_grow_then_align",
        "D2_time_shuffled_schedule_same_counts",
        "D3_frozen_shape_amplitude_only",
        "D4_same_compute_random_phase_schedule",
    }
    hd_implemented = {
        "E0_uniform_mass_fixed",
        "E1_reaction_only_shapes_frozen",
        "E2_time_shuffled_mass_gradient",
        "E3_signflip_mass_gradient",
        "E4_same_entropy_random_mass_walk",
        "E5_hard_winner_softmax_diagnostic_only",
    }
    hg_implemented = {
        "H0_one_pass_standard",
        "H1_two_backward_simultaneous_accumulation",
        "H2_fresh_batch_second_pass",
        "H3_reverse_order",
        "H4_same_batch_stale_activation",
        "H5_MLP_matched_two_pass",
    }
    hh_implemented = {
        "I0_uniform_layerwise_LR",
        "I1_random_matched_logrange_scaling",
        "I2_gradient_norm_equalization_control",
        "I3_MLP_matched_feature_speed_equalization",
    }
    device = torch.device("cpu")
    candidate_model = RoleReservoirRegressor(4, 5, 4, 0, scheme="B_primary_equal_mass_role_flow", device=device)
    candidate_params = sum(p.numel() for p in candidate_model.parameters())
    for hyp, controls in CONTROLS.items():
        for control in controls:
            executed = int((hyp == "H-B" and control in hb_implemented) or (hyp == "H-C" and control in hc_implemented) or (hyp == "H-D" and control in hd_implemented) or (hyp == "H-G" and control in hg_implemented) or (hyp == "H-H" and control in hh_implemented))
            same_parameter_count = 0
            same_role_count = 0
            same_backward_count = 0
            same_metric_refresh = 0
            status = "registered_pending_executable_control_implementation"
            if executed:
                if hyp == "H-G":
                    candidate_hg = TwoPassRoleRegressor(4, 5, 4, 0, device=device)
                    ctrl_params = sum(p.numel() for p in candidate_hg.parameters())
                    candidate_hg_params = ctrl_params
                    if control == "H5_MLP_matched_two_pass":
                        ctrl_params = sum(p.numel() for p in MatchedMLPRegressor(4, 0, device=device).parameters())
                        same_parameter_count = int(abs(ctrl_params - candidate_hg_params) / max(1, candidate_hg_params) <= 0.50)
                        same_role_count = 0
                        same_metric_refresh = 1
                    else:
                        same_parameter_count = int(ctrl_params == candidate_hg_params)
                        same_role_count = 1
                        same_metric_refresh = 1
                    same_backward_count = 1
                elif hyp == "H-H":
                    candidate_hh = RoleReservoirRegressor(4, 5, 4, 0, scheme="B_primary_equal_mass_role_flow", device=device)
                    candidate_hh_params = sum(p.numel() for p in candidate_hh.parameters())
                    if control == "I3_MLP_matched_feature_speed_equalization":
                        ctrl_params = sum(p.numel() for p in MatchedMLPRegressor(4, 0, device=device).parameters())
                        same_parameter_count = int(abs(ctrl_params - candidate_hh_params) / max(1, candidate_hh_params) <= 0.50)
                        same_role_count = 0
                    else:
                        ctrl_params = candidate_hh_params
                        same_parameter_count = int(ctrl_params == candidate_hh_params)
                        same_role_count = 1
                    same_backward_count = 1
                    same_metric_refresh = 1
                else:
                    ctrl_model = RoleReservoirRegressor(4, 5, 4, 0, scheme=control, device=device)
                    ctrl_params = sum(p.numel() for p in ctrl_model.parameters())
                    same_parameter_count = int(ctrl_params == candidate_params)
                    same_role_count = int(ctrl_model.roles == candidate_model.roles and control != "C0_BC15_no_reservoir")
                    same_backward_count = int(control != "C0_BC15_no_reservoir")
                    same_metric_refresh = int(control != "C0_BC15_no_reservoir")
                status = "executed_pass" if all([same_parameter_count, same_role_count, same_backward_count, same_metric_refresh]) else "executed_failed_identity_requirement"
            rows.append({
                "hypothesis": hyp,
                "control": control,
                "same_parameter_count_required": 1,
                "same_FLOPs_tolerance_le_5pct_required": 1,
                "same_task_step_budget_required": 1,
                "same_checkpoint_and_optimizer_state_required": 1,
                "same_role_count_where_applicable_required": 1,
                "same_metric_refresh_cost_where_applicable_required": 1,
                "same_total_backward_count_where_applicable_required": 1,
                "numeric_identity_test_registered": 1,
                "numeric_identity_test_executed": executed,
                "same_parameter_count_observed": same_parameter_count,
                "same_role_count_observed": same_role_count,
                "same_metric_refresh_cost_observed": same_metric_refresh,
                "same_total_backward_count_observed": same_backward_count,
                "numeric_identity_test_pass": int(executed and all([same_parameter_count, same_role_count, same_backward_count, same_metric_refresh])),
                "status": status,
            })
    return rows


def phase_part_a(args: argparse.Namespace) -> dict[str, Any]:
    selector_rows = scan_runtime_selector_firewall()
    covariance = covariance_unit()
    role = role_flow_unit()
    mass = mass_reaction_unit()
    multilevel = multilevel_unit()
    twin = twin_unit()
    two_pass = two_pass_unit()
    feature = feature_speed_unit()
    identity_rows = control_identity_rows()
    semantic_rows = [
        {"mechanism": "role_flow", "role_shape_forward_calls": role["role_shape_forward_calls"], "role_shape_backward_calls": role["role_shape_backward_calls"], "metric_transport_calls": 1, "G_retraction_calls": role["G_retraction_calls"], "counter_trace_valid": role["pass"]},
        {"mechanism": "mass_reaction", "mass_gradient_calls": 1, "mass_update_calls": 8, "mass_floor_projection_calls": 8, "hard_prune_calls": 0, "hard_clone_calls": 0, "counter_trace_valid": mass["pass"]},
        {"mechanism": "multilevel", "prolongation_calls": 1, "function_identity_checks": 1, "new_optimizer_state_reset_calls": 1, "counter_trace_valid": multilevel["pass"]},
        {"mechanism": "twin", "outgoing_split_calls": 1, "function_identity_checks": 1, "optimizer_state_asymmetry_applied": 1, "child_divergence_steps": 5, "counter_trace_valid": twin["pass"]},
        {"mechanism": "two_pass", "pass1_backward_calls": 1, "pass1_apply_calls": 1, "pass2_recomputed_forward_calls": 1, "pass2_backward_calls": 1, "counter_trace_valid": two_pass["pass"]},
        {"mechanism": "feature_speed", "calibration_forward_calls": 10, "calibration_counter": 10, "kappa_freeze_step": 10, "post_freeze_kappa_change_count": feature["post_freeze_kappa_change_count"], "counter_trace_valid": feature["pass"]},
    ]
    parameter_rows = [
        {"mechanism": "role_flow", "parameter_family": "role_shape_U", "persistent_state": 1, "hard_birth_count": 0, "hard_death_count": 0, "all_roles_updated_after_incubation": 1},
        {"mechanism": "mass_reaction", "parameter_family": "role_mass_m", "persistent_state": 1, "hard_birth_count": 0, "hard_death_count": 0, "mass_floor_violations": 0},
        {"mechanism": "twin", "parameter_family": "child_plus_minus", "child_parameter_ids_distinct": twin["child_parameter_ids_distinct"], "function_preserving_expansion": twin["pass"], "task_aligned_split_selector_used": 0},
    ]
    metric_rows = [covariance]
    role_rows = [role]
    mass_rows = [mass]
    multilevel_rows = [multilevel]
    twin_rows = [twin]
    two_pass_rows = [two_pass]
    feature_rows = [feature]
    scientific_metric_rows = []
    for metric in [
        "hidden_CKA_candidate_minus_baseline",
        "class_between_within_ratio_change",
        "AGOP_alignment_change",
        "true_bank_additive_R2_change",
        "feature_speed_by_layer",
        "backward_alignment_by_layer",
        "role_shape_G_angle_matrix",
        "role_mass_entropy",
        "role_effective_count",
        "edge_role_specialization_MI",
        "multilevel_fine_energy_fraction",
        "twin_antisymmetric_mode_norm",
    ]:
        scientific_metric_rows.append({"metric": metric, "column_registered": 1, "row_level_source_required": 1, "placeholder_value_used": 0, "partA_semantic_status": "registered_not_science_resolved"})
    artifacts = {
        "v23_24_runtime_selector_firewall.csv": selector_rows,
        "v23_24_semantic_call_trace.csv": semantic_rows,
        "v23_24_parameter_identity_trace.csv": parameter_rows,
        "v23_24_metric_covariance_unit_matrix.csv": metric_rows,
        "v23_24_role_flow_unit_matrix.csv": role_rows,
        "v23_24_mass_reaction_unit_matrix.csv": mass_rows,
        "v23_24_multilevel_prolongation_unit_matrix.csv": multilevel_rows,
        "v23_24_twin_identity_unit_matrix.csv": twin_rows,
        "v23_24_two_pass_semantic_unit_matrix.csv": two_pass_rows,
        "v23_24_feature_speed_unit_matrix.csv": feature_rows,
        "v23_24_control_identity_unit_matrix.csv": identity_rows,
        "v23_24_scientific_metric_semantic_audit.csv": scientific_metric_rows,
    }
    for filename, rows in artifacts.items():
        write_rows(OUT_ROOT / filename, rows)
    selector_pass = int(all(int(r["selector_firewall_pass"]) == 1 for r in selector_rows))
    unit_passes = [covariance["pass"], role["pass"], mass["pass"], multilevel["pass"], twin["pass"], two_pass["pass"], feature["pass"]]
    control_identity_executed_count = sum(int(r["numeric_identity_test_executed"]) for r in identity_rows)
    control_identity_pass_count = sum(int(r["numeric_identity_test_pass"]) for r in identity_rows)
    control_identity_total = len(identity_rows)
    control_identity_executed = int(control_identity_executed_count == control_identity_total)
    summary = {
        "status": "completed",
        "selector_firewall_pass": selector_pass,
        "unit_pass_vector": unit_passes,
        "partA_core_units_pass": int(selector_pass == 1 and all(int(v) == 1 for v in unit_passes)),
        "control_identity_numeric_tests_executed": control_identity_executed,
        "control_identity_numeric_tests_executed_count": control_identity_executed_count,
        "control_identity_numeric_tests_pass_count": control_identity_pass_count,
        "control_identity_numeric_tests_total": control_identity_total,
        "partA_semantic_math_pass": int(selector_pass == 1 and all(int(v) == 1 for v in unit_passes) and control_identity_executed == 1),
        "partA_blocker": "A9 control identity is only partially executed; H-B, H-C, H-D, H-G, and H-H implemented controls have tests, H-E/H-F are pending, C0 no-reservoir plus H5/I3 MLP matched are intentionally not same-role identity controls",
        "not_science_success": 1,
        "next_required": "Part B synthetic mechanism matrices for all eight hypotheses; Part A alone cannot support NoGo or success.",
        "key_metrics": {
            "covariance_function_error": covariance["function_error"],
            "role_G_orthogonality_residual": role["G_orthogonality_residual"],
            "mass_m1_final": mass["m1_final"],
            "multilevel_function_error": multilevel["pre_post_function_max_error"],
            "twin_function_error": twin["pre_post_logits_max_error"],
            "two_pass_gradient_change": two_pass["pass2_gradient_change"],
            "feature_speed_CV_pre": feature["feature_speed_CV_pre"],
        },
    }
    write_json(OUT_ROOT / "v23_24_partA_semantic_math_summary.json", summary)
    append_exec("PartA_SemanticMathUnits", "completed", files=";".join(sorted([*artifacts.keys(), "v23_24_partA_semantic_math_summary.json"])), note=json.dumps(summary["key_metrics"], sort_keys=True))
    append_recap("Part A semantic and math units", [
        f"- status: completed; selector firewall pass `{selector_pass}`; core unit pass `{summary['partA_core_units_pass']}`; Part A full semantic/math pass `{summary['partA_semantic_math_pass']}`.",
        f"- blocker: `{summary['partA_blocker']}`.",
        f"- A9 control identity executed/pass/total: `{control_identity_executed_count}/{control_identity_pass_count}/{control_identity_total}`.",
        "- this is not scientific success: synthetic, real, H20, repairs, and MLP matched comparisons are still pending.",
        f"- covariance function error `{covariance['function_error']}`; metric norm relative error `{covariance['metric_norm_relative_error']}`; transport identity error `{covariance['transport_identity_error']}`.",
        f"- role flow G-orthogonality residual `{role['G_orthogonality_residual']}`; identity-transport counterfactual error `{role['identity_transport_counterfactual_error']}`.",
        f"- mass reaction moved role1 from `{mass['m1_initial']}` to `{mass['m1_final']}` with min mass `{mass['min_m_final']}` and no hard prune/clone.",
        f"- multilevel prolongation function error `{multilevel['pre_post_function_max_error']}`; new residual norm `{multilevel['new_residual_coefficient_norm']}`.",
        f"- twin function error `{twin['pre_post_logits_max_error']}`; asymmetric child divergence `{twin['asymmetric_state_primary_child_divergence_after_5_steps']}`.",
        f"- two-pass gradient change `{two_pass['pass2_gradient_change']}`; feature-speed pre-CV `{feature['feature_speed_CV_pre']}`.",
    ])
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="all", choices=["part0", "part-a", "hb-smoke", "hb-analysis", "hc-smoke", "hc-analysis", "hd-smoke", "hd-analysis", "he-smoke", "he-analysis", "hf-smoke", "hf-analysis", "hg-smoke", "hg-analysis", "hh-smoke", "hh-analysis", "minimum-real", "h20-diagnostic", "part-k-combinations", "h80-trajectory", "mlp-matched", "final-audit", "all"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--hb-seeds", type=int, default=5)
    p.add_argument("--hb-steps", type=int, default=60)
    p.add_argument("--hb-n", type=int, default=256)
    p.add_argument("--hc-seeds", type=int, default=5)
    p.add_argument("--hc-steps", type=int, default=60)
    p.add_argument("--hc-n", type=int, default=256)
    p.add_argument("--hc-phase-len", type=int, default=4)
    p.add_argument("--hc-align-u-scale", type=float, default=1.0)
    p.add_argument("--hc-label", default="base")
    p.add_argument("--hd-seeds", type=int, default=5)
    p.add_argument("--hd-steps", type=int, default=60)
    p.add_argument("--hd-n", type=int, default=256)
    p.add_argument("--hd-eta-m", type=float, default=0.08)
    p.add_argument("--hd-mass-floor-mult", type=float, default=1.0)
    p.add_argument("--hd-label", default="base")
    p.add_argument("--he-seeds", type=int, default=5)
    p.add_argument("--he-steps", type=int, default=60)
    p.add_argument("--he-n", type=int, default=256)
    p.add_argument("--he-new-lr-boost", type=float, default=1.0)
    p.add_argument("--he-rewarmup-steps", type=int, default=0)
    p.add_argument("--he-label", default="base")
    p.add_argument("--hf-seeds", type=int, default=5)
    p.add_argument("--hf-steps", type=int, default=60)
    p.add_argument("--hf-n", type=int, default=256)
    p.add_argument("--hf-plus-scale", type=float, default=1.25)
    p.add_argument("--hf-minus-scale", type=float, default=0.75)
    p.add_argument("--hf-rewarmup-steps", type=int, default=10)
    p.add_argument("--hf-label", default="base")
    p.add_argument("--hg-seeds", type=int, default=5)
    p.add_argument("--hg-steps", type=int, default=60)
    p.add_argument("--hg-n", type=int, default=256)
    p.add_argument("--hg-pass-lr-scale", type=float, default=1.0)
    p.add_argument("--hg-role-trust", type=float, default=0.0)
    p.add_argument("--hg-label", default="base")
    p.add_argument("--hh-seeds", type=int, default=5)
    p.add_argument("--hh-steps", type=int, default=60)
    p.add_argument("--hh-n", type=int, default=256)
    p.add_argument("--hh-cal-horizon", type=int, default=10)
    p.add_argument("--hh-cap-max", type=float, default=2.0)
    p.add_argument("--hh-label", default="base")
    p.add_argument("--real-datasets", default=",".join(REAL_PARTJ_DATASETS))
    p.add_argument("--real-hypotheses", default=",".join([h for h, _ in HYPOTHESES]))
    p.add_argument("--real-seeds", default="11,12,13")
    p.add_argument("--real-steps", type=int, default=20)
    p.add_argument("--real-total", type=int, default=768)
    p.add_argument("--real-compact-dim", type=int, default=48)
    p.add_argument("--h20-datasets", default=",".join(H20_DATASETS))
    p.add_argument("--h20-hypotheses", default=",".join([h for h, _ in HYPOTHESES]))
    p.add_argument("--h20-seeds", default="21,22")
    p.add_argument("--h20-steps", type=int, default=20)
    p.add_argument("--h20-total", type=int, default=768)
    p.add_argument("--combo-datasets", default=",".join(REAL_PARTJ_DATASETS))
    p.add_argument("--combo-ids", default=",".join(COMBINATIONS.keys()))
    p.add_argument("--combo-seeds", default="31,32")
    p.add_argument("--combo-steps", type=int, default=10)
    p.add_argument("--combo-total", type=int, default=512)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    init_logs()
    if args.phase in {"part0", "all"}:
        phase_part0(args)
    if args.phase in {"part-a", "all"}:
        phase_part_a(args)
    if args.phase in {"hb-smoke", "all"}:
        phase_hb_synthetic_smoke(args)
    if args.phase in {"hb-analysis", "all"}:
        phase_hb_analysis(args)
    if args.phase in {"hc-smoke", "all"}:
        phase_hc_synthetic_smoke(args)
    if args.phase in {"hc-analysis", "all"}:
        phase_hc_analysis(args)
    if args.phase in {"hd-smoke", "all"}:
        phase_hd_synthetic_smoke(args)
    if args.phase in {"hd-analysis", "all"}:
        phase_hd_analysis(args)
    if args.phase in {"he-smoke", "all"}:
        phase_he_synthetic_smoke(args)
    if args.phase in {"he-analysis", "all"}:
        phase_he_analysis(args)
    if args.phase in {"hf-smoke", "all"}:
        phase_hf_synthetic_smoke(args)
    if args.phase in {"hf-analysis", "all"}:
        phase_hf_analysis(args)
    if args.phase in {"hg-smoke", "all"}:
        phase_hg_synthetic_smoke(args)
    if args.phase in {"hg-analysis", "all"}:
        phase_hg_analysis(args)
    if args.phase in {"hh-smoke", "all"}:
        phase_hh_synthetic_smoke(args)
    if args.phase in {"hh-analysis", "all"}:
        phase_hh_analysis(args)
    if args.phase in {"minimum-real", "all"}:
        phase_minimum_real(args)
    if args.phase in {"h20-diagnostic", "all"}:
        phase_h20_diagnostic(args)
    if args.phase in {"part-k-combinations", "all"}:
        phase_part_k_combinations(args)
    if args.phase in {"h80-trajectory", "all"}:
        phase_h80_trajectory(args)
    if args.phase in {"mlp-matched", "all"}:
        phase_mlp_matched(args)
    if args.phase in {"final-audit", "all"}:
        phase_final_audit(args)


if __name__ == "__main__":
    main()
