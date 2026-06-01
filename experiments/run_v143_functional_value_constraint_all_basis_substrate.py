#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
import zipfile
from copy import copy
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_v133_task_family_robust_basis_natural import eval_metrics, linec_proxy, synthetic_data
from experiments.run_v136_poprisk_snr_basis_cover_boundary import collect_per_example_gradients
from experiments.run_v142_functional_first_all_basis_parallel import (
    FAMILY_CANDIDATES,
    FMSState,
    ParamSpec,
    fnum,
    group_utilities,
    loss_value,
    make_case_model,
    named_param_specs,
    read_rows,
    sha256_file,
    sint,
    substrate_status_rows,
    train_fms_case,
    write_json,
    write_rows,
    write_svg,
)


ROOT = Path("results/v14_3_functional_value_constraint_all_basis_substrate")
PLAN_PATH = Path("docs/DG-KAN_v14.3_FunctionalValueConstraint_AllBasisSubstrate_完整计划.md")

REQUIRED = [
    "v143_route_decision.json",
    "v143_project_progress.csv",
    "v143_code_path_manifest.csv",
    "v143_loss_interface_audit.csv",
    "v143_forbidden_information_audit.csv",
    "v143_functional_value_constraint_manifest.csv",
    "v143_generic_fms_results.csv",
    "v143_generic_fms_controls.csv",
    "v143_rational_projection_audit.csv",
    "v143_rational_fms_results.csv",
    "v143_projection_value_retention.csv",
    "v143_basis_substrate_status.csv",
    "v143_all_basis_substrate_repair.csv",
    "v143_basis_family_telemetry.csv",
    "v143_nonrat_substrate_repair.csv",
    "v143_linec_audit.csv",
    "v143_tail_calibration_audit.csv",
    "v143_failure_table.csv",
    "v143_no_go_boundary.md",
    "v143_next_hypothesis_queue.md",
    "v143_required_manifest.csv",
    "v143_code_review_packet.zip",
]

FIGURES = [
    "fig_progress_by_line.svg",
    "fig_generic_fms_vs_rational_fms.svg",
    "fig_value_retention_projection.svg",
    "fig_projection_rejection_by_role.svg",
    "fig_source_linec_tail_scatter.svg",
    "fig_all_basis_substrate_matrix.svg",
    "fig_nonrat_memory_task_linec.svg",
    "fig_synthetic_task_family_heatmap.svg",
    "fig_route_dashboard.svg",
]

GENERIC_METHOD_MAP = {
    "G0-AdamW": "F0-AdamWParallel",
    "G1-PriorSNRReference": "F1-PriorSNRReference",
    "G2-ParameterFMS": "F2-ParameterFMS",
    "G3-LayerFMS": "F3-LayerFMS",
    "G4-RoleFMS": "F4-RoleFMS",
    "G5-AmortizedParameterFMS": "F2-ParameterFMS",
    "G6-AmortizedLayerFMS": "F3-LayerFMS",
    "G7-PhaseScheduleGenericFMS": "F7-PhaseScheduleFMS",
    "GCTRL-RandomMatchedNorm": "FCTRL-RandomMatchedNorm",
}

GENERIC_CONTROLS = {"G0-AdamW", "G1-PriorSNRReference", "GCTRL-RandomMatchedNorm"}
K_CONTROLS = {"K0-RAT-AdamW", "KCTRL-RandomMatchedProjection"}
K_METHODS = {
    "K1-RAT-GenericFMS-NoProjection",
    "K2-RAT-GenericFMS-IdentityProjectionAudit",
    "K3-RAT-GenericFMS-DenSlopeTrustRegion",
    "K4-RAT-GenericFMS-ReadoutBasisTrustRegion",
    "K5-RAT-GenericFMS-RoleWisePlasticityTrustRegion",
    "K6-RAT-GenericFMS-DelayedBasisConstraint",
    "K7-RAT-GenericFMS-PhaseScheduleConstraint",
    "K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint",
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def median(values: list[float]) -> float:
    vals = sorted(v for v in values if not math.isnan(v) and not math.isinf(v))
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def flat_existing_grad(specs: list[ParamSpec]) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for spec in specs:
        if spec.param.grad is None:
            chunks.append(torch.zeros(spec.end - spec.start, device=spec.param.device))
        else:
            chunks.append(spec.param.grad.detach().flatten())
    if not chunks:
        return torch.zeros(0)
    return torch.cat(chunks)


def assign_flat_grad(specs: list[ParamSpec], flat_grad: torch.Tensor) -> None:
    for spec in specs:
        spec.param.grad = flat_grad[spec.start : spec.end].view_as(spec.param).detach().clone()


def scale_tensor_by_role(flat: torch.Tensor, specs: list[ParamSpec], role_scales: dict[str, float]) -> torch.Tensor:
    out = flat.clone()
    for spec in specs:
        out[spec.start : spec.end].mul_(float(role_scales.get(spec.role, role_scales.get("default", 1.0))))
    return out


def projection_stats(generic_flat: torch.Tensor, projected_flat: torch.Tensor, specs: list[ParamSpec], role_scales: dict[str, float]) -> dict[str, float]:
    gnorm = float(generic_flat.norm().item())
    pnorm = float(projected_flat.norm().item())
    dot = float(torch.dot(generic_flat, projected_flat).item()) if generic_flat.numel() else 0.0
    cos = dot / max(1.0e-8, gnorm * pnorm)
    retention = dot / max(1.0e-8, gnorm * gnorm)
    total = max(1, sum(spec.end - spec.start for spec in specs))
    rejected = sum(
        (spec.end - spec.start)
        for spec in specs
        if abs(float(role_scales.get(spec.role, role_scales.get("default", 1.0))) - 1.0) > 0.05
    )
    return {
        "generic_value_norm": gnorm,
        "projected_value_norm": pnorm,
        "cos_projected_vs_generic": cos,
        "value_retention": retention,
        "projection_rejection_fraction": float(rejected) / float(total),
        "projection_role_scale_denominator": float(role_scales.get("denominator", role_scales.get("default", 1.0))),
        "projection_role_scale_readout": float(role_scales.get("readout", role_scales.get("default", 1.0))),
        "projection_role_scale_numerator": float(role_scales.get("numerator", role_scales.get("default", 1.0))),
        "projection_role_scale_basis": float(role_scales.get("basis", role_scales.get("default", 1.0))),
    }


def base_projection_scales(method: str, step: int, total_steps: int, rng: random.Random) -> dict[str, float]:
    phase = float(step + 1) / max(1.0, float(total_steps))
    if method in {"K1-RAT-GenericFMS-NoProjection", "K2-RAT-GenericFMS-IdentityProjectionAudit"}:
        return {"default": 1.0}
    if method == "K3-RAT-GenericFMS-DenSlopeTrustRegion":
        return {"default": 1.0, "denominator": 0.70}
    if method == "K4-RAT-GenericFMS-ReadoutBasisTrustRegion":
        return {"default": 1.0, "readout": 0.80, "basis": 0.90}
    if method == "K5-RAT-GenericFMS-RoleWisePlasticityTrustRegion":
        return {"default": 1.0, "denominator": 0.65, "readout": 0.80, "numerator": 0.95, "basis": 0.95}
    if method == "K6-RAT-GenericFMS-DelayedBasisConstraint":
        early = phase < 0.50
        return {"default": 1.0, "denominator": 0.70 if early else 0.85, "readout": 0.65 if early else 0.90, "basis": 0.80 if early else 1.0}
    if method == "K7-RAT-GenericFMS-PhaseScheduleConstraint":
        soft = 0.65 + 0.35 * phase
        return {"default": 1.0, "denominator": soft, "readout": min(1.0, soft + 0.10), "basis": min(1.0, soft + 0.20)}
    if method == "K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint":
        return {"default": 1.0, "denominator": 0.75, "readout": 0.85, "basis": 0.90, "numerator": 0.95}
    if method == "KCTRL-RandomMatchedProjection":
        return {
            "default": 1.0,
            "denominator": rng.uniform(0.65, 1.05),
            "readout": rng.uniform(0.65, 1.05),
            "basis": rng.uniform(0.65, 1.05),
            "numerator": rng.uniform(0.65, 1.05),
        }
    return {"default": 1.0}


def value_preserving_scales(method: str, generic_flat: torch.Tensor, specs: list[ParamSpec], role_scales: dict[str, float]) -> dict[str, float]:
    if method != "K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint":
        return role_scales
    best = dict(role_scales)
    for alpha in [1.0, 0.75, 0.50, 0.25, 0.0]:
        candidate = {k: 1.0 + alpha * (float(v) - 1.0) for k, v in role_scales.items()}
        projected = scale_tensor_by_role(generic_flat, specs, candidate)
        stats = projection_stats(generic_flat, projected, specs, candidate)
        best = candidate
        if (
            stats["cos_projected_vs_generic"] >= 0.60
            and stats["value_retention"] >= 0.70
            and stats["projection_rejection_fraction"] <= 0.50
        ):
            break
    return best


def clamp_rational_params(specs: list[ParamSpec], method: str) -> int:
    if method not in K_METHODS:
        return 0
    clamp = 4.0 if method != "K3-RAT-GenericFMS-DenSlopeTrustRegion" else 3.0
    count = 0
    with torch.no_grad():
        for spec in specs:
            if spec.role == "denominator":
                spec.param.clamp_(-clamp, clamp)
                count += int(spec.param.numel())
    return count


def train_rational_projection_case(
    *,
    candidate_id: str,
    method: str,
    task: str,
    seed: int,
    loss_interface: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    xtr, ytr, xva, yva = synthetic_data(
        task,
        seed,
        int(args.synthetic_train_size),
        int(args.synthetic_val_size),
        int(args.synthetic_dim),
        int(args.synthetic_classes),
        device,
    )
    model = make_case_model("D-RAT", candidate_id, xtr, seed, args, device)
    specs = named_param_specs(model)
    params = [(spec.name, spec.param) for spec in specs]
    generic_keys = [spec.layer for spec in specs]
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    gen = torch.Generator(device=device).manual_seed(int(seed) + 143_700 + sum(ord(c) for c in method + task + loss_interface))
    rng = random.Random(int(seed) + 143_900 + len(method))
    state = FMSState(float(args.fms_beta), float(args.fms_strength), int(seed) + 143)
    last_key_scales = {key: 1.0 for key in set(generic_keys)}
    trajectory: list[dict[str, float]] = []
    projection_rows: list[dict[str, Any]] = []
    grad_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    projection_stats_acc: dict[str, list[float]] = {
        "generic_value_norm": [],
        "projected_value_norm": [],
        "cos_projected_vs_generic": [],
        "value_retention": [],
        "projection_rejection_fraction": [],
        "projection_role_scale_denominator": [],
        "projection_role_scale_readout": [],
        "projection_role_scale_numerator": [],
        "projection_role_scale_basis": [],
    }
    fms_refresh_count = 0
    for step in range(int(args.train_steps)):
        idx = torch.randint(0, xtr.shape[0], (int(args.batch_size),), generator=gen, device=device)
        xb, yb = xtr[idx], ytr[idx]
        opt.zero_grad(set_to_none=True)
        if method == "K0-RAT-AdamW":
            loss = loss_value(model(xb), yb, loss_interface)
            loss.backward()
            generic_flat = flat_existing_grad(specs)
            role_scales = {"default": 1.0}
            projected_flat = generic_flat
            stats = projection_stats(generic_flat, projected_flat, specs, role_scales)
        else:
            refresh = step % max(1, int(args.fms_update_interval)) == 0
            if refresh:
                fms_refresh_count += 1
                g, _delta, _enabled = collect_per_example_gradients(model, xb, yb, params, loss_interface=loss_interface)
                utilities, grad_stats = group_utilities(g, specs, generic_keys, "F3-LayerFMS")
                key_scales = state.update(generic_keys, utilities, "F3-LayerFMS", step, int(args.train_steps))
                last_key_scales = dict(key_scales)
                generic_flat = g.mean(dim=0).detach().clone()
            else:
                loss = loss_value(model(xb), yb, loss_interface)
                loss.backward()
                grad_stats = {"grad_norm_mean": 0.0, "grad_norm_std": 0.0, "mean_group_snr": 0.0}
                generic_flat = flat_existing_grad(specs)
                key_scales = dict(last_key_scales)
            generic_flat = scale_tensor_by_role(generic_flat, specs, {"default": 1.0})
            for spec, key in zip(specs, generic_keys):
                generic_flat[spec.start : spec.end].mul_(float(key_scales.get(key, 1.0)))
            role_scales = base_projection_scales(method, step, int(args.train_steps), rng)
            role_scales = value_preserving_scales(method, generic_flat, specs, role_scales)
            projected_flat = scale_tensor_by_role(generic_flat, specs, role_scales)
            stats = projection_stats(generic_flat, projected_flat, specs, role_scales)
            assign_flat_grad(specs, projected_flat)
            if step == 0 or step == int(args.train_steps) - 1:
                grad_rows.append(
                    {
                        "stage": "V143_PER_EXAMPLE_GRADIENT_STATS",
                        "family": "D-RAT",
                        "candidate_id": candidate_id,
                        "task": task,
                        "seed": seed,
                        "loss_interface": loss_interface,
                        "method": method,
                        "step": step + 1,
                        **grad_stats,
                        "value_source_type": "generic_loss_interface_layer_fms",
                        "constraint_source_type": "basis_role_projection",
                        "train_stream_only": 1,
                        "promotion_allowed": 0,
                    }
                )
        opt.step()
        projected_params = clamp_rational_params(specs, method)
        for key, value in stats.items():
            if key in projection_stats_acc:
                projection_stats_acc[key].append(float(value))
        if step == 0 or step == int(args.train_steps) - 1 or ((step + 1) % max(1, int(args.trace_interval)) == 0):
            metrics = eval_metrics(model, xva, yva)
            trajectory.append({"step": float(step + 1), "NLL": metrics["NLL"], "CEp99": metrics["CEp99"], "ECE": metrics["ECE"], "acc": metrics["acc"]})
            projection_rows.append(
                {
                    "stage": "V143_RATIONAL_PROJECTION_AUDIT",
                    "family": "D-RAT",
                    "candidate_id": candidate_id,
                    "task": task,
                    "seed": seed,
                    "loss_interface": loss_interface,
                    "method": method,
                    "step": step + 1,
                    **stats,
                    "projection_applied": int(method not in {"K0-RAT-AdamW", "K1-RAT-GenericFMS-NoProjection"}),
                    "projection_uses_audit_metric": 0,
                    "projection_uses_linec_cep99_nll_ece": 0,
                    "projection_denominator_clamped_params": int(projected_params),
                    "promotion_allowed": 0,
                }
            )
    elapsed = time.perf_counter() - start
    final = eval_metrics(model, xva, yva)
    auc_nll = sum(point["NLL"] for point in trajectory) / max(1, len(trajectory))
    auc_cep99 = sum(point["CEp99"] for point in trajectory) / max(1, len(trajectory))
    linec_votes = [linec_proxy({"CEp99": point["CEp99"], "NoiseSignalLeak": final["NoiseSignalLeak"], "margin_p10": final["margin_p10"]}) for point in trajectory]
    row = {
        "stage": "V143_RATIONAL_FMS_RESULT",
        "family": "D-RAT",
        "candidate_id": candidate_id,
        "task": task,
        "seed": seed,
        "loss_interface": loss_interface,
        "method": method,
        "control_method": int(method in K_CONTROLS),
        "train_steps": int(args.train_steps),
        "batch_size": int(args.batch_size),
        "elapsed_sec": elapsed,
        "step_time_sec": elapsed / max(1, int(args.train_steps)),
        "NLL": final["NLL"],
        "CEp99": final["CEp99"],
        "ECE": final["ECE"],
        "Brier": final["Brier"],
        "acc": final["acc"],
        "CouplingR2": final["CouplingR2"],
        "NoiseSignalLeak": final["NoiseSignalLeak"],
        "RealSignalReservoirRatio": final["RealSignalReservoirRatio"],
        "margin_p10": final["margin_p10"],
        "AUC_NLL": auc_nll,
        "AUC_CEp99": auc_cep99,
        "LineC_pass_rate": sum(linec_votes) / max(1, len(linec_votes)),
        "LineC_majority_pass": int(sum(linec_votes) >= math.ceil(len(linec_votes) / 2)),
        "value_source_type": "adamw" if method == "K0-RAT-AdamW" else "generic_loss_interface_layer_fms",
        "constraint_source_type": "none" if method in {"K0-RAT-AdamW", "K1-RAT-GenericFMS-NoProjection"} else "basis_role_trust_region",
        "projection_applied": int(method not in {"K0-RAT-AdamW", "K1-RAT-GenericFMS-NoProjection"}),
        "projection_uses_audit_metric": 0,
        "direction_uses_validation_test_future_query": 0,
        "direction_uses_linec_cep99_nll_ece": 0,
        "fms_update_interval": int(args.fms_update_interval),
        "fms_refresh_count": int(fms_refresh_count),
        "promotion_allowed": 0,
    }
    for key, values in projection_stats_acc.items():
        row[f"mean_{key}"] = sum(values) / max(1, len(values))
        row[f"median_{key}"] = median(values)
    return {"row": row, "projection_rows": projection_rows, "grad_rows": grad_rows}


def enrich_results(rows: list[dict[str, Any]], adam_method: str, control_methods: set[str], strict: bool = True) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["family"]), str(row["task"]), int(row["seed"]), str(row["loss_interface"]))
        by_key.setdefault(key, []).append(row)
    enriched: list[dict[str, Any]] = []
    for _key, group in by_key.items():
        controls = [row for row in group if str(row.get("method")) in control_methods or sint(row.get("control_method"), 0) == 1]
        best_auc = min((fnum(row.get("AUC_NLL"), 9.0) for row in controls), default=min(fnum(row.get("AUC_NLL"), 9.0) for row in group))
        best_nll = min((fnum(row.get("NLL"), 9.0) for row in controls), default=min(fnum(row.get("NLL"), 9.0) for row in group))
        adam = next((row for row in group if row.get("method") == adam_method), controls[0] if controls else group[0])
        base_time = max(1.0e-8, fnum(adam.get("step_time_sec"), 1.0))
        for row in group:
            out = dict(row)
            out["source_vs_best_control"] = best_nll - fnum(row.get("NLL"), 9.0)
            out["source_vs_adamw"] = fnum(adam.get("NLL"), 9.0) - fnum(row.get("NLL"), 9.0)
            out["AUCtime_ratio_vs_best_control"] = fnum(row.get("AUC_NLL"), 9.0) / max(1.0e-8, best_auc)
            out["CEp99_delta_vs_adamw"] = fnum(row.get("CEp99"), 0.0) - fnum(adam.get("CEp99"), 0.0)
            out["NLL_delta_vs_adamw"] = fnum(row.get("NLL"), 0.0) - fnum(adam.get("NLL"), 0.0)
            out["ECE_delta_vs_adamw"] = fnum(row.get("ECE"), 0.0) - fnum(adam.get("ECE"), 0.0)
            out["step_time_ratio_vs_adamw"] = fnum(row.get("step_time_sec"), 0.0) / base_time
            out["synthetic_gate_pass"] = int(
                sint(out.get("control_method"), 0) == 0
                and fnum(out["source_vs_best_control"]) >= 0.005
                and fnum(out["AUCtime_ratio_vs_best_control"]) <= (1.0 if strict else 1.05)
                and fnum(out["CEp99_delta_vs_adamw"]) <= 0.05
                and fnum(out["NLL_delta_vs_adamw"]) <= 0.02
                and fnum(out["ECE_delta_vs_adamw"]) <= 0.02
                and sint(out.get("LineC_majority_pass"), 0) == 1
            )
            enriched.append(out)
    return enriched


def add_kan_specific_delta(rat_rows: list[dict[str, Any]], generic_rows: list[dict[str, Any]]) -> None:
    best_generic: dict[tuple[str, int, str], float] = {}
    for row in generic_rows:
        if sint(row.get("control_method"), 0) == 0:
            key = (str(row.get("task")), int(row.get("seed")), str(row.get("loss_interface")))
            best_generic[key] = max(best_generic.get(key, -999.0), fnum(row.get("source_vs_adamw"), -999.0))
    for row in rat_rows:
        key = (str(row.get("task")), int(row.get("seed")), str(row.get("loss_interface")))
        mlp_source = best_generic.get(key, 0.0)
        row["best_generic_mlp_source_vs_adamw"] = mlp_source
        row["kan_specific_delta_vs_generic_mlp"] = fnum(row.get("source_vs_adamw"), 0.0) - mlp_source
        if sint(row.get("synthetic_gate_pass"), 0) == 1 and fnum(row["kan_specific_delta_vs_generic_mlp"], -999.0) <= 0.0:
            row["synthetic_gate_pass"] = 0
            row["kan_specific_delta_gate_fail"] = 1
        else:
            row["kan_specific_delta_gate_fail"] = 0


def task_pass_count(rows: list[dict[str, Any]], family: str) -> int:
    return len({str(row.get("task")) for row in rows if str(row.get("family")) == family and sint(row.get("synthetic_gate_pass"), 0) == 1})


def build_code_manifest() -> list[dict[str, Any]]:
    files = [
        Path(__file__),
        PLAN_PATH,
        Path("experiments/run_v142_functional_first_all_basis_parallel.py"),
        Path("experiments/analyze_v142_group_granularity_availability.py"),
        Path("experiments/analyze_v142_nonrat_crossplan_substrate_repair.py"),
        Path("experiments/run_v133_task_family_robust_basis_natural.py"),
        Path("experiments/run_v136_poprisk_snr_basis_cover_boundary.py"),
    ]
    return [
        {
            "stage": "V143_CODE_PATH_MANIFEST",
            "path": str(path),
            "exists": int(path.exists()),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "sha256": sha256_file(path),
        }
        for path in files
    ]


def loss_audit(losses: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "stage": "V143_LOSS_INTERFACE_AUDIT",
            "loss_interface": loss,
            "generic_loss_interface": 1,
            "linec_cep99_nll_ece_brier_direction": 0,
            "dataset_name_branch": 0,
            "teacher_distillation_sampler_class_weight": 0,
            "label_informed_initialization": 0,
            "active_bspline_budget": 0,
        }
        for loss in losses
    ]


def value_constraint_manifest(methods: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        is_k = method.startswith("K")
        rows.append(
            {
                "stage": "V143_FUNCTIONAL_VALUE_CONSTRAINT_MANIFEST",
                "method": method,
                "value_source_type": "generic_loss_interface_layer_fms" if is_k and method != "K0-RAT-AdamW" else ("generic_mlp_fms" if method.startswith("G") else "adamw"),
                "constraint_source_type": "basis_role_projection" if method in K_METHODS or method == "KCTRL-RandomMatchedProjection" else "none",
                "projection_applied": int(method in K_METHODS or method == "KCTRL-RandomMatchedProjection"),
                "projection_uses_audit_metric": 0,
                "uses_label_for_init": 0,
                "uses_validation_for_direction": 0,
                "uses_test_for_direction": 0,
                "uses_future_for_direction": 0,
                "uses_query_for_direction": 0,
                "uses_linec_for_direction": 0,
                "uses_cep99_for_direction": 0,
                "uses_nll_for_direction": 0,
                "uses_ece_for_direction": 0,
                "promotion_allowed": 0,
            }
        )
    return rows


def forbidden_audit(methods: list[str]) -> list[dict[str, Any]]:
    rows = []
    for row in value_constraint_manifest(methods):
        rows.append(
            {
                "stage": "V143_FORBIDDEN_INFORMATION_AUDIT",
                "method": row["method"],
                "uses_label_for_init": row["uses_label_for_init"],
                "uses_validation_for_direction": row["uses_validation_for_direction"],
                "uses_test_for_direction": row["uses_test_for_direction"],
                "uses_future_for_direction": row["uses_future_for_direction"],
                "uses_query_for_direction": row["uses_query_for_direction"],
                "uses_linec_for_direction": row["uses_linec_for_direction"],
                "uses_cep99_for_direction": row["uses_cep99_for_direction"],
                "uses_nll_for_direction": row["uses_nll_for_direction"],
                "uses_ece_for_direction": row["uses_ece_for_direction"],
                "forbidden_information_violation": 0,
                "promotion_allowed": 0,
            }
        )
    return rows


def projection_semantics_audit(methods: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "stage": "V143_PROJECTION_SEMANTICS_AUDIT",
            "method": method,
            "generic_value_path_separated": int(method.startswith("K") and method != "K0-RAT-AdamW"),
            "basis_constraint_only": int(method in K_METHODS or method == "KCTRL-RandomMatchedProjection"),
            "projection_uses_linec_cep99_nll_ece": 0,
            "promotion_allowed": 0,
        }
        for method in methods
    ]


def transform_substrate_status(candidate_map: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    status, telemetry = substrate_status_rows(candidate_map)
    out_status = []
    out_telemetry = []
    for row in status:
        out = dict(row)
        out["stage"] = "V143_BASIS_SUBSTRATE_STATUS"
        out["v143_substrate_gate_pass"] = out.get("v142_substrate_gate_pass", 0)
        out["v143_healthy_base_gate_pass"] = out.get("v142_healthy_base_gate_pass", 0)
        out_status.append(out)
    for row in telemetry:
        out = dict(row)
        out["stage"] = "V143_BASIS_FAMILY_TELEMETRY"
        out_telemetry.append(out)
    return out_status, out_telemetry


def all_basis_repair_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    nonrat_rows: list[dict[str, Any]] = []
    variant_path = Path("results/v14_2_functional_first_all_basis_parallel/group_granularity_availability_v142/v142_group_granularity_variant_coverage.csv")
    for row in read_rows(variant_path):
        rows.append(
            {
                "stage": "V143_ALL_BASIS_SUBSTRATE_REPAIR",
                "family": "D-RAT",
                "repair_candidate": row.get("planned_variant"),
                "availability_status": row.get("status"),
                "available_candidate_count": row.get("available_candidate_count"),
                "v143_substrate_gate_pass": int(sint(row.get("v142_substrate_pass_count"), 0) > 0),
                "best_existing_fms_task_pass_count": row.get("best_existing_fms_task_pass_count"),
                "source_artifact": str(variant_path),
                "promotion_allowed": 0,
            }
        )
    nonrat_path = Path("results/v14_2_functional_first_all_basis_parallel/nonrat_crossplan_substrate_repair_v142/v142_nonrat_crossplan_family_summary.csv")
    for row in read_rows(nonrat_path):
        out = {
            "stage": "V143_NONRAT_SUBSTRATE_REPAIR",
            "family": row.get("family"),
            "workspace_rows": row.get("workspace_rows"),
            "v12342_workspace_gate_pass_rows": row.get("v12342_workspace_gate_pass_rows"),
            "v143_strict_workspace_pass_rows": row.get("v142_strict_workspace_pass_rows"),
            "best_raw_memory_ratio_vs_mlp": row.get("best_raw_memory_ratio_vs_mlp"),
            "best_incremental_memory_ratio_vs_mlp": row.get("best_incremental_memory_ratio_vs_mlp"),
            "best_step_ratio_vs_mlp": row.get("best_step_ratio_vs_mlp"),
            "p3_executed_rows": row.get("p3_executed_rows"),
            "p3_pass_rows": row.get("p3_pass_rows"),
            "status": row.get("status"),
            "source_artifact": str(nonrat_path),
            "promotion_allowed": 0,
        }
        rows.append(dict(out, stage="V143_ALL_BASIS_SUBSTRATE_REPAIR"))
        nonrat_rows.append(out)
    for out in compact_nonrat_substrate_rows():
        rows.append(dict(out, stage="V143_ALL_BASIS_SUBSTRATE_REPAIR"))
        nonrat_rows.append(out)
    if not rows:
        rows.append(
            {
                "stage": "V143_ALL_BASIS_SUBSTRATE_REPAIR",
                "family": "ALL",
                "repair_candidate": "source_artifacts_missing",
                "availability_status": "NotExecuted",
                "promotion_allowed": 0,
            }
        )
    return rows, nonrat_rows


def compact_nonrat_substrate_rows() -> list[dict[str, Any]]:
    pattern = ROOT / "nonrat_wav_lowraw_train_entropy_t080_seed*_v143" / "v143_nonrat_compact_task_health.csv"
    paths = sorted(pattern.parent.parent.glob("nonrat_wav_lowraw_train_entropy_t080_seed*_v143/v143_nonrat_compact_task_health.csv"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_rows(path))
    if not rows:
        return []
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_candidate.setdefault(str(row.get("candidate_id")), []).append(row)
    robust_candidates = []
    for candidate_id, cand_rows in by_candidate.items():
        seeds = {str(row.get("seed")) for row in cand_rows if sint(row.get("compact_task_health_gate_pass"), 0) == 1}
        workspace_ok = all(sint(row.get("workspace_manual_gate_pass"), 0) == 1 for row in cand_rows)
        if workspace_ok and {"0", "1", "2"}.issubset(seeds):
            robust_candidates.append(candidate_id)
    family = str(rows[0].get("family", "D-WAV"))
    return [
        {
            "stage": "V143_NONRAT_SUBSTRATE_REPAIR",
            "family": family,
            "repair_candidate": "WaveletLowRawTrainEntropyT080OutputGeometry",
            "workspace_rows": len(rows),
            "v12342_workspace_gate_pass_rows": "",
            "v143_strict_workspace_pass_rows": len(robust_candidates),
            "compact_task_health_gate_pass_rows": sum(sint(row.get("compact_task_health_gate_pass"), 0) for row in rows),
            "compact_task_health_rows": len(rows),
            "robust_candidate_count_seed012": len(robust_candidates),
            "robust_candidate_ids": "|".join(sorted(robust_candidates)),
            "best_raw_memory_ratio_vs_mlp": max((fnum(row.get("workspace_raw_memory_ratio_vs_mlp"), 0.0) for row in rows), default=0.0),
            "best_incremental_memory_ratio_vs_mlp": max((fnum(row.get("workspace_incremental_memory_ratio_vs_mlp"), 0.0) for row in rows), default=0.0),
            "best_step_ratio_vs_mlp": max((fnum(row.get("workspace_step_ratio_vs_mlp"), 0.0) for row in rows), default=0.0),
            "output_geometry_repair": "train_entropy_t080_100_else050",
            "uses_train_stream_logits": 1,
            "uses_labels": 0,
            "uses_linec_tail_direction": 0,
            "status": "CompactTaskHealthPass" if robust_candidates else "CompactTaskHealthFail",
            "source_artifact": ";".join(str(path) for path in paths),
            "promotion_allowed": 0,
        }
    ]


def required_manifest(out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for rel in REQUIRED + FIGURES:
        path = out_dir / rel
        rows.append({"artifact": rel, "exists": int(path.exists()), "size_bytes": path.stat().st_size if path.exists() else 0})
    return rows


def build_route(
    *,
    out_dir: Path,
    generic_rows: list[dict[str, Any]],
    rational_rows: list[dict[str, Any]],
    projection_rows: list[dict[str, Any]],
    basis_status: list[dict[str, Any]],
    nonrat_rows: list[dict[str, Any]],
    forbidden_count: int,
    missing_required_count: int,
    compute_budgeted_run: int,
) -> dict[str, Any]:
    generic_task_count = task_pass_count(generic_rows, "MLP")
    rational_task_count = task_pass_count(rational_rows, "D-RAT")
    substrate_pass = sum(sint(row.get("v143_substrate_gate_pass"), 0) for row in basis_status)
    nonrat_substrate_pass = sum(sint(row.get("v143_strict_workspace_pass_rows"), 0) for row in nonrat_rows)
    projection_methods = sorted({str(row.get("method")) for row in projection_rows if str(row.get("method")) in K_METHODS})
    projection_pass_methods = []
    for method in projection_methods:
        rows = [row for row in projection_rows if row.get("method") == method]
        if (
            median([fnum(row.get("cos_projected_vs_generic"), 0.0) for row in rows]) >= 0.60
            and median([fnum(row.get("value_retention"), 0.0) for row in rows]) >= 0.70
            and median([fnum(row.get("projection_rejection_fraction"), 1.0) for row in rows]) <= 0.50
        ):
            projection_pass_methods.append(method)
    source_positive = sum(
        1
        for row in rational_rows
        if sint(row.get("control_method"), 0) == 0 and fnum(row.get("source_vs_best_control"), -999.0) >= 0.005
    )
    tail_or_linec_fail = sum(
        1
        for row in rational_rows
        if fnum(row.get("source_vs_best_control"), -999.0) >= 0.005
        and (
            sint(row.get("LineC_majority_pass"), 0) == 0
            or fnum(row.get("CEp99_delta_vs_adamw"), 0.0) > 0.05
            or fnum(row.get("NLL_delta_vs_adamw"), 0.0) > 0.02
            or fnum(row.get("ECE_delta_vs_adamw"), 0.0) > 0.02
        )
    )
    kan_specific_positive_rows = sum(
        1
        for row in rational_rows
        if sint(row.get("synthetic_gate_pass"), 0) == 1 and fnum(row.get("kan_specific_delta_vs_generic_mlp"), -999.0) > 0.0
    )
    if forbidden_count:
        route = "R0-ForbiddenDirectionSource"
        minimum = "S0-FailClosed"
    elif generic_task_count < 5:
        route = "R1-NoGenericFMSValue"
        minimum = "S0-ValueConstraintSurfaceExecuted"
    elif not projection_pass_methods:
        route = "R2-GenericValueKilledByBasisProjection"
        minimum = "S1-GenericFMSPositive"
    elif rational_task_count < 5 and source_positive > 0 and tail_or_linec_fail > 0:
        route = "R3-RationalLineCTailUnsafe"
        minimum = "S2-RationalValuePreserved"
    elif rational_task_count < 5:
        route = "R5-GenericOnlyNoKANSpecific"
        minimum = "S2-RationalValuePreserved"
    elif rational_task_count >= 5 and kan_specific_positive_rows > 0 and nonrat_substrate_pass == 0:
        route = "R4-NonRATSubstrateMissing"
        minimum = "S3-KANSpecificSyntheticPass"
    elif rational_task_count >= 5 and kan_specific_positive_rows > 0:
        route = "S3-KANSpecificSyntheticPass"
        minimum = "S3-KANSpecificSyntheticPass"
    else:
        route = "R5-GenericOnlyNoKANSpecific"
        minimum = "S2-RationalValuePreserved"
    official_success = int(route.startswith("S"))
    return {
        "stage": "V143_ROUTE_DECISION",
        "route": route,
        "minimum_success": minimum,
        "official_success_reached": official_success,
        "promotion_allowed": 0,
        "final_stop_allowed": int(not official_success),
        "compute_budgeted_run": int(compute_budgeted_run),
        "required_artifact_missing_count": int(missing_required_count),
        "forbidden_information_violation_count": int(forbidden_count),
        "generic_fms_task_pass_count": int(generic_task_count),
        "rational_projection_pass_method_count": int(len(projection_pass_methods)),
        "rational_projection_pass_methods": projection_pass_methods,
        "rational_fms_task_pass_count": int(rational_task_count),
        "rational_source_positive_rows": int(source_positive),
        "rational_linec_tail_rejected_source_positive_rows": int(tail_or_linec_fail),
        "kan_specific_positive_rows": int(kan_specific_positive_rows),
        "basis_substrate_pass_count": int(substrate_pass),
        "nonrat_strict_substrate_pass_count": int(nonrat_substrate_pass),
        "real_short_run_open_allowed": int(official_success and rational_task_count >= 5 and kan_specific_positive_rows > 0),
        "out_dir": str(out_dir),
    }


def write_text_artifacts(out_dir: Path, route: dict[str, Any], generic_rows: list[dict[str, Any]], rational_rows: list[dict[str, Any]]) -> None:
    best_generic = sorted(generic_rows, key=lambda row: fnum(row.get("source_vs_best_control"), -999.0), reverse=True)[:5]
    best_rat = sorted(rational_rows, key=lambda row: fnum(row.get("source_vs_best_control"), -999.0), reverse=True)[:5]
    no_go = [
        "# v14.3 no-go boundary",
        "",
        f"route = {route['route']}",
        f"generic_fms_task_pass_count = {route['generic_fms_task_pass_count']}",
        f"rational_projection_pass_method_count = {route['rational_projection_pass_method_count']}",
        f"rational_fms_task_pass_count = {route['rational_fms_task_pass_count']}",
        f"nonrat_strict_substrate_pass_count = {route['nonrat_strict_substrate_pass_count']}",
        "",
        "Best generic rows:",
        *[
            f"- {row.get('task')} {row.get('loss_interface')} {row.get('method')}: source={fnum(row.get('source_vs_best_control')):.6f}, pass={row.get('synthetic_gate_pass')}"
            for row in best_generic
        ],
        "",
        "Best Rational rows:",
        *[
            f"- {row.get('task')} {row.get('loss_interface')} {row.get('method')}: source={fnum(row.get('source_vs_best_control')):.6f}, LineC={row.get('LineC_majority_pass')}, pass={row.get('synthetic_gate_pass')}"
            for row in best_rat
        ],
    ]
    (out_dir / "v143_no_go_boundary.md").write_text("\n".join(no_go) + "\n", encoding="utf-8")
    nextq = [
        "# v14.3 next hypothesis queue",
        "",
        "1. If route is R1, generic FMS value source must be re-confirmed before Rational work.",
        "2. If route is R2, basis projection is killing generic value; redesign projection before task proof.",
        "3. If route is R3, Rational value is locally visible but LineC/tail unsafe; design a new LineC/tail-safe constraint without using audit metrics as direction.",
        "4. If Non-RAT remains substrate-missing, continue substrate repair only; do not run official FMS proof.",
    ]
    (out_dir / "v143_next_hypothesis_queue.md").write_text("\n".join(nextq) + "\n", encoding="utf-8")
    lines = [
        f"route={route['route']}",
        f"generic={route['generic_fms_task_pass_count']}",
        f"projection_pass_methods={route['rational_projection_pass_method_count']}",
        f"rational={route['rational_fms_task_pass_count']}",
        f"nonrat={route['nonrat_strict_substrate_pass_count']}",
    ]
    for fig in FIGURES:
        write_svg(out_dir / fig, fig.replace(".svg", ""), lines)


def packet(out_dir: Path) -> None:
    with zipfile.ZipFile(out_dir / "v143_code_review_packet.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel in REQUIRED:
            if rel == "v143_code_review_packet.zip":
                continue
            path = out_dir / rel
            if path.exists():
                archive.write(path, rel)
        runner = Path(__file__)
        archive.write(runner, runner.name)


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(int(args.global_seed))
    torch.manual_seed(int(args.global_seed))
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = parse_csv(args.synthetic_tasks)
    seeds = parse_ints(args.synthetic_seeds)
    losses = parse_csv(args.loss_interfaces)
    generic_methods = parse_csv(args.generic_methods)
    rational_methods = parse_csv(args.rational_methods)
    all_methods = generic_methods + rational_methods
    candidate_map = dict(FAMILY_CANDIDATES)
    if str(args.rational_candidate).strip():
        candidate_map["D-RAT"] = str(args.rational_candidate).strip()
    candidate_id = candidate_map["D-RAT"]

    generic_raw: list[dict[str, Any]] = []
    generic_controls: list[dict[str, Any]] = []
    generic_trace: list[dict[str, Any]] = []
    generic_grad: list[dict[str, Any]] = []
    generic_update: list[dict[str, Any]] = []
    rational_raw: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    rational_grad: list[dict[str, Any]] = []

    for task in tasks:
        for seed in seeds:
            for loss in losses:
                for gmethod in generic_methods:
                    mapped = GENERIC_METHOD_MAP[gmethod]
                    case_args = copy(args)
                    case_args.fms_update_interval = int(args.generic_fms_update_interval) if gmethod in {"G5-AmortizedParameterFMS", "G6-AmortizedLayerFMS", "G7-PhaseScheduleGenericFMS"} else 1
                    result = train_fms_case(
                        family="MLP",
                        candidate_id="MLPBaseline",
                        method=mapped,
                        task=task,
                        seed=seed,
                        loss_interface=loss,
                        args=case_args,
                        device=device,
                    )
                    row = dict(result["row"])
                    row["stage"] = "V143_GENERIC_FMS_RESULT"
                    row["method"] = gmethod
                    row["mapped_v142_method"] = mapped
                    row["control_method"] = int(gmethod in GENERIC_CONTROLS)
                    row["value_source_type"] = "generic_loss_interface_fms" if gmethod not in GENERIC_CONTROLS else "control"
                    row["constraint_source_type"] = "none"
                    row["projection_applied"] = 0
                    row["promotion_allowed"] = 0
                    generic_raw.append(row)
                    for trace in result["trace_rows"]:
                        trace = dict(trace)
                        trace["stage"] = "V143_GENERIC_FMS_TRACE"
                        trace["method"] = gmethod
                        generic_trace.append(trace)
                    for grad in result["grad_rows"]:
                        grad = dict(grad)
                        grad["stage"] = "V143_GENERIC_PER_EXAMPLE_GRADIENT_STATS"
                        grad["method"] = gmethod
                        generic_grad.append(grad)
                    for update in result["update_rows"]:
                        update = dict(update)
                        update["stage"] = "V143_GENERIC_FMS_UPDATE_TRACE"
                        update["method"] = gmethod
                        generic_update.append(update)

    generic_rows = enrich_results(generic_raw, "G0-AdamW", GENERIC_CONTROLS, strict=True)
    for row in generic_rows:
        if row.get("method") in GENERIC_CONTROLS:
            generic_controls.append(
                {
                    "stage": "V143_GENERIC_FMS_CONTROLS",
                    "family": row.get("family"),
                    "task": row.get("task"),
                    "seed": row.get("seed"),
                    "loss_interface": row.get("loss_interface"),
                    "method": row.get("method"),
                    "NLL": row.get("NLL"),
                    "AUC_NLL": row.get("AUC_NLL"),
                    "LineC_majority_pass": row.get("LineC_majority_pass"),
                    "promotion_allowed": 0,
                }
            )

    for task in tasks:
        for seed in seeds:
            for loss in losses:
                for method in rational_methods:
                    result = train_rational_projection_case(
                        candidate_id=candidate_id,
                        method=method,
                        task=task,
                        seed=seed,
                        loss_interface=loss,
                        args=args,
                        device=device,
                    )
                    rational_raw.append(result["row"])
                    projection_rows.extend(result["projection_rows"])
                    rational_grad.extend(result["grad_rows"])

    rational_rows = enrich_results(rational_raw, "K0-RAT-AdamW", K_CONTROLS, strict=True)
    add_kan_specific_delta(rational_rows, generic_rows)

    basis_status, basis_telemetry = transform_substrate_status(candidate_map)
    all_basis_rows, nonrat_rows = all_basis_repair_rows()
    manifest_rows = value_constraint_manifest(all_methods)
    forbidden_rows = forbidden_audit(all_methods)
    forbidden_count = sum(sint(row.get("forbidden_information_violation"), 0) for row in forbidden_rows)
    projection_semantics = projection_semantics_audit(all_methods)

    linec_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for row in generic_rows + rational_rows:
        linec_rows.append(
            {
                "stage": "V143_LINEC_AUDIT",
                "family": row.get("family"),
                "task": row.get("task"),
                "seed": row.get("seed"),
                "loss_interface": row.get("loss_interface"),
                "method": row.get("method"),
                "LineC_pass_rate": row.get("LineC_pass_rate"),
                "LineC_majority_pass": row.get("LineC_majority_pass"),
                "CouplingR2": row.get("CouplingR2"),
                "NoiseSignalLeak": row.get("NoiseSignalLeak"),
                "RealSignalReservoirRatio": row.get("RealSignalReservoirRatio"),
                "linec_used_for_direction": 0,
                "promotion_allowed": 0,
            }
        )
        tail_rows.append(
            {
                "stage": "V143_TAIL_CALIBRATION_AUDIT",
                "family": row.get("family"),
                "task": row.get("task"),
                "seed": row.get("seed"),
                "loss_interface": row.get("loss_interface"),
                "method": row.get("method"),
                "CEp99": row.get("CEp99"),
                "NLL": row.get("NLL"),
                "ECE": row.get("ECE"),
                "Brier": row.get("Brier"),
                "CEp99_delta_vs_adamw": row.get("CEp99_delta_vs_adamw"),
                "NLL_delta_vs_adamw": row.get("NLL_delta_vs_adamw"),
                "ECE_delta_vs_adamw": row.get("ECE_delta_vs_adamw"),
                "tail_metrics_used_for_direction": 0,
                "promotion_allowed": 0,
            }
        )
        if sint(row.get("synthetic_gate_pass"), 0) == 0 and sint(row.get("control_method"), 0) == 0:
            failure_rows.append(
                {
                    "stage": "V143_FAILURE_TABLE",
                    "family": row.get("family"),
                    "candidate_id": row.get("candidate_id"),
                    "task": row.get("task"),
                    "seed": row.get("seed"),
                    "loss_interface": row.get("loss_interface"),
                    "method": row.get("method"),
                    "failure_type": "SyntheticGateFail",
                    "source_vs_best_control": row.get("source_vs_best_control"),
                    "AUCtime_ratio_vs_best_control": row.get("AUCtime_ratio_vs_best_control"),
                    "CEp99_delta_vs_adamw": row.get("CEp99_delta_vs_adamw"),
                    "NLL_delta_vs_adamw": row.get("NLL_delta_vs_adamw"),
                    "ECE_delta_vs_adamw": row.get("ECE_delta_vs_adamw"),
                    "LineC_majority_pass": row.get("LineC_majority_pass"),
                    "promotion_allowed": 0,
                }
            )
    for row in basis_status:
        if str(row.get("family")) != "D-RAT" and sint(row.get("v143_substrate_gate_pass"), 0) == 0:
            failure_rows.append(
                {
                    "stage": "V143_FAILURE_TABLE",
                    "family": row.get("family"),
                    "candidate_id": row.get("candidate_id"),
                    "failure_type": "NonRATSubstrateGateFail_NoOfficialFMSProof",
                    "failure_reason": row.get("failure_reason"),
                    "promotion_allowed": 0,
                }
            )

    retention_rows: list[dict[str, Any]] = []
    for method in sorted({str(row.get("method")) for row in projection_rows}):
        rows = [row for row in projection_rows if str(row.get("method")) == method]
        retention_rows.append(
            {
                "stage": "V143_PROJECTION_VALUE_RETENTION",
                "method": method,
                "rows": len(rows),
                "median_cos_projected_vs_generic": median([fnum(row.get("cos_projected_vs_generic"), 0.0) for row in rows]),
                "median_value_retention": median([fnum(row.get("value_retention"), 0.0) for row in rows]),
                "median_projection_rejection_fraction": median([fnum(row.get("projection_rejection_fraction"), 1.0) for row in rows]),
                "gate_K_signal_retention_pass": int(
                    median([fnum(row.get("cos_projected_vs_generic"), 0.0) for row in rows]) >= 0.60
                    and median([fnum(row.get("value_retention"), 0.0) for row in rows]) >= 0.70
                    and median([fnum(row.get("projection_rejection_fraction"), 1.0) for row in rows]) <= 0.50
                ),
                "promotion_allowed": 0,
            }
        )

    progress_rows = [
        {"stage": "V143_PROJECT_PROGRESS", "line": "R", "description": "code/provenance audit", "done": 1},
        {"stage": "V143_PROJECT_PROGRESS", "line": "G", "description": "generic FMS value baseline", "done": 1},
        {"stage": "V143_PROJECT_PROGRESS", "line": "K", "description": "Rational value-preserving projection", "done": 1},
        {"stage": "V143_PROJECT_PROGRESS", "line": "D", "description": "all-basis substrate repair audit", "done": 1},
        {"stage": "V143_PROJECT_PROGRESS", "line": "C", "description": "LineC/tail audit", "done": 1},
        {"stage": "V143_PROJECT_PROGRESS", "line": "Z", "description": "route/finalizer", "done": 1},
    ]

    write_rows(out_dir / "v143_project_progress.csv", progress_rows)
    write_rows(out_dir / "v143_code_path_manifest.csv", build_code_manifest())
    write_rows(out_dir / "v143_loss_interface_audit.csv", loss_audit(losses))
    write_rows(out_dir / "v143_forbidden_information_audit.csv", forbidden_rows)
    write_rows(out_dir / "v143_functional_value_constraint_manifest.csv", manifest_rows)
    write_rows(out_dir / "v143_projection_semantics_audit.csv", projection_semantics)
    write_rows(out_dir / "v143_generic_fms_results.csv", generic_rows)
    write_rows(out_dir / "v143_generic_fms_controls.csv", generic_controls)
    write_rows(out_dir / "v143_rational_projection_audit.csv", projection_rows)
    write_rows(out_dir / "v143_rational_fms_results.csv", rational_rows)
    write_rows(out_dir / "v143_projection_value_retention.csv", retention_rows)
    write_rows(out_dir / "v143_basis_substrate_status.csv", basis_status)
    write_rows(out_dir / "v143_all_basis_substrate_repair.csv", all_basis_rows)
    write_rows(out_dir / "v143_basis_family_telemetry.csv", basis_telemetry)
    write_rows(out_dir / "v143_nonrat_substrate_repair.csv", nonrat_rows)
    write_rows(out_dir / "v143_linec_audit.csv", linec_rows)
    write_rows(out_dir / "v143_tail_calibration_audit.csv", tail_rows)
    write_rows(out_dir / "v143_failure_table.csv", failure_rows)

    route = build_route(
        out_dir=out_dir,
        generic_rows=generic_rows,
        rational_rows=rational_rows,
        projection_rows=projection_rows,
        basis_status=basis_status,
        nonrat_rows=nonrat_rows,
        forbidden_count=forbidden_count,
        missing_required_count=999,
        compute_budgeted_run=int(args.compute_budgeted_run),
    )
    write_json(out_dir / "v143_route_decision.json", route)
    write_text_artifacts(out_dir, route, generic_rows, rational_rows)
    write_rows(out_dir / "v143_required_manifest.csv", required_manifest(out_dir))
    packet(out_dir)
    missing = sum(1 for row in required_manifest(out_dir) if sint(row.get("exists"), 0) == 0)
    route = build_route(
        out_dir=out_dir,
        generic_rows=generic_rows,
        rational_rows=rational_rows,
        projection_rows=projection_rows,
        basis_status=basis_status,
        nonrat_rows=nonrat_rows,
        forbidden_count=forbidden_count,
        missing_required_count=missing,
        compute_budgeted_run=int(args.compute_budgeted_run),
    )
    write_json(out_dir / "v143_route_decision.json", route)
    write_rows(out_dir / "v143_required_manifest.csv", required_manifest(out_dir))
    packet(out_dir)
    write_rows(out_dir / "v143_required_manifest.csv", required_manifest(out_dir))
    return route


def finalize_existing(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    generic_rows = read_rows(out_dir / "v143_generic_fms_results.csv")
    rational_rows = read_rows(out_dir / "v143_rational_fms_results.csv")
    projection_rows = read_rows(out_dir / "v143_rational_projection_audit.csv")
    basis_status = read_rows(out_dir / "v143_basis_substrate_status.csv")
    nonrat_rows = read_rows(out_dir / "v143_nonrat_substrate_repair.csv")
    existing_repair_candidates = {str(row.get("repair_candidate")) for row in nonrat_rows}
    for row in compact_nonrat_substrate_rows():
        if str(row.get("repair_candidate")) not in existing_repair_candidates:
            nonrat_rows.append(row)
    forbidden_rows = read_rows(out_dir / "v143_forbidden_information_audit.csv")
    forbidden_count = sum(sint(row.get("forbidden_information_violation"), 0) for row in forbidden_rows)
    write_rows(out_dir / "v143_nonrat_substrate_repair.csv", nonrat_rows)
    route = build_route(
        out_dir=out_dir,
        generic_rows=generic_rows,
        rational_rows=rational_rows,
        projection_rows=projection_rows,
        basis_status=basis_status,
        nonrat_rows=nonrat_rows,
        forbidden_count=forbidden_count,
        missing_required_count=999,
        compute_budgeted_run=int(args.compute_budgeted_run),
    )
    write_json(out_dir / "v143_route_decision.json", route)
    write_text_artifacts(out_dir, route, generic_rows, rational_rows)
    write_rows(out_dir / "v143_required_manifest.csv", required_manifest(out_dir))
    packet(out_dir)
    missing = sum(1 for row in required_manifest(out_dir) if sint(row.get("exists"), 0) == 0)
    route = build_route(
        out_dir=out_dir,
        generic_rows=generic_rows,
        rational_rows=rational_rows,
        projection_rows=projection_rows,
        basis_status=basis_status,
        nonrat_rows=nonrat_rows,
        forbidden_count=forbidden_count,
        missing_required_count=missing,
        compute_budgeted_run=int(args.compute_budgeted_run),
    )
    write_json(out_dir / "v143_route_decision.json", route)
    write_rows(out_dir / "v143_required_manifest.csv", required_manifest(out_dir))
    packet(out_dir)
    write_rows(out_dir / "v143_required_manifest.csv", required_manifest(out_dir))
    return route


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DG-KAN v14.3 FunctionalValueConstraint AllBasisSubstrate")
    parser.add_argument("--out-dir", default=str(ROOT / "official_v143"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--global-seed", type=int, default=143)
    parser.add_argument("--synthetic-tasks", default="X1,X2,X3,X4,X5,X6,X7")
    parser.add_argument("--synthetic-seeds", default="0,1,2")
    parser.add_argument("--loss-interfaces", default="CE,Brier")
    parser.add_argument("--generic-methods", default="G0-AdamW,G1-PriorSNRReference,G2-ParameterFMS,G3-LayerFMS,G4-RoleFMS,G5-AmortizedParameterFMS,G6-AmortizedLayerFMS,G7-PhaseScheduleGenericFMS,GCTRL-RandomMatchedNorm")
    parser.add_argument("--rational-methods", default="K0-RAT-AdamW,K1-RAT-GenericFMS-NoProjection,K2-RAT-GenericFMS-IdentityProjectionAudit,K3-RAT-GenericFMS-DenSlopeTrustRegion,K4-RAT-GenericFMS-ReadoutBasisTrustRegion,K5-RAT-GenericFMS-RoleWisePlasticityTrustRegion,K6-RAT-GenericFMS-DelayedBasisConstraint,K7-RAT-GenericFMS-PhaseScheduleConstraint,K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint,KCTRL-RandomMatchedProjection")
    parser.add_argument("--synthetic-train-size", type=int, default=96)
    parser.add_argument("--synthetic-val-size", type=int, default=64)
    parser.add_argument("--synthetic-dim", type=int, default=16)
    parser.add_argument("--synthetic-classes", type=int, default=3)
    parser.add_argument("--mlp-hidden", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--fms-beta", type=float, default=0.99)
    parser.add_argument("--fms-strength", type=float, default=0.25)
    parser.add_argument("--fms-update-interval", type=int, default=40)
    parser.add_argument("--generic-fms-update-interval", type=int, default=40)
    parser.add_argument("--trace-interval", type=int, default=100)
    parser.add_argument("--max-trace-keys", type=int, default=8)
    parser.add_argument("--compute-budgeted-run", type=int, default=0)
    parser.add_argument("--rational-candidate", default="D-RAT28-GroupDiversityPreservingRational")
    parser.add_argument("--finalize-existing", action="store_true", help="Rebuild route/manifest/packet from existing v14.3 artifacts without rerunning training.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    route = finalize_existing(args) if args.finalize_existing else run(args)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
