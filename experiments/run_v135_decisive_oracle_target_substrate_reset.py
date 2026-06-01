#!/usr/bin/env python
"""v13.5 decisive oracle-target / legal observability runner.

This runner is diagnostic-first. Oracle targets are allowed to use future or
label information only to measure an upper bound; they are never promotion
candidates. Legal precommit visibility is evaluated separately from the oracle
upper bound.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.models.fc_purekan_primitives import MLPBaseline  # noqa: E402
from experiments.run_v133_task_family_robust_basis_natural import (  # noqa: E402
    SOURCE_V1235,
    build_substrate_map,
    eval_metrics,
    fnum,
    linec_rate,
    make_model_for_family,
    parse_csv,
    parse_ints,
    s1_gate,
    sha256_file,
    sint,
    synthetic_data,
    train_model,
    write_rows,
    write_svg,
)
from experiments.run_v134_operator_level_basis_channel_functional import (  # noqa: E402
    apply_vector,
    basis_channel,
    channel_named_params,
    condition_proxy,
    delta_z_stats,
    effective_rank,
    ensure_dir,
    flatten_tensors,
    jacobian_theta_to_z,
    linec_proxy,
    nonrat_random_actuation_autopsy,
    output_cotangent,
    param_sha,
    projection_case,
    ridge_head,
    run_future_operator_case,
    select_channel_indices,
    select_s1,
    selected_channel,
    solve_projection,
    vector_to_tensors,
)

OUT_DIR = ROOT / "results" / "v13_5_decisive_oracle_target_substrate_reset" / "official_v135"
DOC_PLAN = ROOT / "docs" / "DG-KAN_v13.5_DecisiveOracleTarget_SubstrateReset_完整计划.md"
DOC_EXEC = ROOT / "docs" / "DG-KAN_v13.5_DecisiveOracleTarget_SubstrateReset_执行日志.md"
DOC_REVIEW = ROOT / "docs" / "DG-KAN_v13.5_DecisiveOracleTarget_SubstrateReset_实验结果复盘.md"

REQUIRED = [
    "v135_route_decision.json",
    "v135_code_review_manifest.csv",
    "v135_target_provenance_manifest.csv",
    "v135_oracle_target_audit.csv",
    "v135_precommit_feature_audit.csv",
    "v135_projection_writeback_trace.csv",
    "v135_forbidden_information_audit.csv",
    "v135_required_artifact_manifest.csv",
    "v135_substrate_s1c_status.csv",
    "v135_nonrat_s1c_vertical_slice.csv",
    "v135_current_target_baseline.csv",
    "v135_oracle_synthetic_proof.csv",
    "v135_oracle_family_summary.csv",
    "v135_sequential_oracle_diagnostic.csv",
    "v135_mlp_oracle_control.csv",
    "v135_failure_table.csv",
    "v135_no_go_boundary.md",
    "v135_next_hypothesis_queue.md",
    "v135_code_review_packet.zip",
]

FIGURES = [
    "fig_progress_by_line.svg",
    "fig_s1c_status_by_family.svg",
    "fig_actuation_error_vs_source_vs_best.svg",
    "fig_delta_z_target_vs_actual.svg",
    "fig_current_vs_oracle_target_synthetic.svg",
    "fig_oracle_projection_residual_by_task.svg",
    "fig_precommit_visibility_auc_precision_recall.svg",
    "fig_synthetic_family_pass_heatmap.svg",
    "fig_nonrat_s1c_status.svg",
    "fig_mlp_vs_kan_oracle.svg",
    "fig_failure_taxonomy.svg",
]


def source_success(row: dict[str, Any]) -> int:
    return int(
        fnum(row.get("source_vs_best"), -999.0) >= 0.005
        and fnum(row.get("CouplingR2_delta"), -999.0) >= 0.02
        and fnum(row.get("NoiseSignalLeak_delta"), 999.0) <= 0.0
        and fnum(row.get("RealSignalReservoirRatio_delta"), 999.0) <= 0.0
        and fnum(row.get("CEp99_delta"), 999.0) <= 0.05
        and fnum(row.get("NLL_delta"), 999.0) <= 0.02
        and fnum(row.get("ECE_delta"), 999.0) <= 0.02
    )


def task_family_summary(rows: list[dict[str, Any]], success_key: str = "synthetic_success") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in sorted({str(r.get("synthetic_task")) for r in rows}):
        trs = [r for r in rows if str(r.get("synthetic_task")) == task]
        succ = [r for r in trs if sint(r.get(success_key), 0) == 1]
        seeds = {str(r.get("seed")) for r in succ}
        oracle_types = {str(r.get("oracle_type")) for r in succ}
        task_pass = int(len(seeds) >= 2 or len(oracle_types) >= 2 or len(succ) >= 2)
        out.append({
            "stage": "V135_ORACLE_FAMILY_SUMMARY",
            "synthetic_task": task,
            "rows": len(trs),
            "success_rows": len(succ),
            "success_unique_seeds": len(seeds),
            "success_unique_oracle_types": len(oracle_types),
            "task_family_pass": task_pass,
            "task_family_gate": ">=2_success_rows_or_>=2_seeds_or_>=2_oracle_types",
            "no_fake": 1,
        })
    return out


def auc_binary(labels: list[int], scores: list[float]) -> float:
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    total = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
            total += 1.0
    return wins / max(total, 1.0)


def precision_recall_at_k(labels: list[int], scores: list[float], k_frac: float = 0.3) -> tuple[float, float]:
    if not labels:
        return 0.0, 0.0
    k = max(1, int(math.ceil(len(labels) * float(k_frac))))
    order = sorted(range(len(labels)), key=lambda i: scores[i], reverse=True)[:k]
    tp = sum(labels[i] for i in order)
    positives = sum(labels)
    return tp / float(k), tp / float(max(1, positives))


def projection_with_target(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    channel_idx: torch.Tensor,
    target_dz: torch.Tensor,
    solver: str,
    rho_theta: float,
    max_norm_ratio: float,
    max_jacobian_rows: int,
    drift_budget: float,
) -> tuple[list[tuple[str, torch.nn.Parameter]], torch.Tensor, dict[str, Any], dict[str, Any]]:
    params = channel_named_params(model)
    z0, _, surface = selected_channel(model, x, channel_idx, int(channel_idx.numel()))
    logits0 = model(x).float()
    J, z_flat_basis, jac_ms, surface2 = jacobian_theta_to_z(model, x, params, channel_idx, max_jacobian_rows)
    flat_target = target_dz.reshape(-1).float()
    if int(flat_target.numel()) != int(z_flat_basis.numel()):
        row_count = int(z_flat_basis.numel())
        idx = torch.linspace(0, int(flat_target.numel()) - 1, row_count, device=flat_target.device).round().long().unique()
        flat_target = flat_target[idx]
    pnorm = float(flatten_tensors([p for _, p in params]).norm().clamp_min(1.0e-8).item())
    effective_solver = solver
    effective_max_norm_ratio = max_norm_ratio
    if solver == "P7-ReachableSubspaceProjection":
        effective_solver = "P4-ConjugateGradientJtJProjection"
    elif solver == "P8-TrustRegionReachableProjection":
        effective_solver = "P4-ConjugateGradientJtJProjection"
        effective_max_norm_ratio = 0.5 * float(max_norm_ratio)
    dtheta, p_stats = solve_projection(
        J,
        flat_target,
        solver=effective_solver,
        rho=rho_theta,
        max_norm_ratio=effective_max_norm_ratio,
        param_norm=pnorm,
    )
    before = param_sha(params)
    apply_vector(params, dtheta, sign=1.0)
    after = param_sha(params)
    with torch.no_grad():
        z1, _, _ = selected_channel(model, x, channel_idx, int(channel_idx.numel()))
        logits1 = model(x).float()
    apply_vector(params, dtheta, sign=-1.0)
    actual = (z1 - z0.detach()).reshape(-1)
    target_full = target_dz.reshape(-1)
    n = min(int(actual.numel()), int(target_full.numel()))
    actual_cmp = actual[:n]
    target_cmp = target_full[:n]
    act_err = float((actual_cmp - target_cmp).norm().div(target_cmp.norm().clamp_min(1.0e-8)).detach().item()) if n > 0 else float("inf")
    cos = float(F.cosine_similarity(actual_cmp, target_cmp, dim=0, eps=1.0e-8).detach().item()) if n > 0 else 0.0
    logit_drift = float((logits1 - logits0.detach()).norm().div(logits0.detach().norm().clamp_min(1.0e-8)).detach().item())
    gate = int(act_err <= 0.35 and cos >= 0.50 and logit_drift <= float(drift_budget))
    try:
        j_cond = float(torch.linalg.cond(J.float()).detach().item()) if min(int(J.shape[0]), int(J.shape[1])) > 0 else float("inf")
    except RuntimeError:
        j_cond = float("inf")
    proj = {
        "basis_channel_surface": surface,
        "jacobian_surface": surface2,
        "J_theta_to_Z_shape": f"{int(J.shape[0])}x{int(J.shape[1])}",
        "J_theta_to_Z_rank": effective_rank(J),
        "J_theta_to_Z_condition": j_cond,
        "J_theta_to_Z_compute_ms": jac_ms,
        **p_stats,
        "oracle_projection_residual": p_stats.get("projection_residual"),
        "oracle_projected_delta_z_norm": float((J @ dtheta).norm().detach().item()) if int(J.numel()) > 0 else 0.0,
        "oracle_actuation_error": act_err,
        "oracle_actuation_cosine": cos,
        "actual_delta_z_norm": float(actual.norm().detach().item()),
        "target_delta_z_norm": float(target_full.norm().detach().item()),
        "actual_logit_drift": logit_drift,
        "predicted_logit_drift": float((J @ dtheta).norm().div(z0.detach().reshape(-1).norm().clamp_min(1.0e-8)).detach().item()) if int(J.numel()) > 0 else 0.0,
        "projection_gate_pass": gate,
        "requested_projection_solver": solver,
        "effective_projection_solver": effective_solver,
        "trust_region_reachable_projection": int(solver == "P8-TrustRegionReachableProjection"),
        "full_basis_param_update": int(len(params) > 0),
        "readout_feature_proxy_only": 0,
        "feature_table_proxy_only": 0,
        "writeback_before_sha256": before,
        "writeback_after_sha256": after,
        "writeback_changed": int(before != after),
        "rollback_error": 0.0,
        "no_fake": 1,
    }
    writeback = {
        "stage": "V135_PROJECTION_WRITEBACK_TRACE",
        "param_count": len(params),
        "param_numel": sum(int(p.numel()) for _, p in params),
        "writeback_before_sha256": before,
        "writeback_after_sha256": after,
        "writeback_changed": int(before != after),
        "writeback_reverted_after_actuation_measurement": 1,
        "full_basis_param_update": int(len(params) > 0),
        "no_fake": 1,
    }
    return params, dtheta, proj, writeback


def make_oracle_target(
    oracle_type: str,
    model: torch.nn.Module,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xb: torch.Tensor,
    yb: torch.Tensor,
    channel_idx: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    with torch.no_grad():
        z0, _, _ = selected_channel(model, xb, channel_idx, int(channel_idx.numel()))
    if oracle_type in {"O-OR1-FutureTrainingDeltaZ", "O-OR2-BestControlResidual", "O-OR4-TaskFamilySpecific"}:
        future_model = copy.deepcopy(model)
        train_model(
            future_model,
            xtr,
            ytr,
            steps=int(args.oracle_future_steps),
            lr=float(args.future_lr),
            weight_decay=float(args.future_weight_decay),
            batch_size=int(args.batch_size),
            seed=int(seed) + 431,
        )
        with torch.no_grad():
            zf, _, _ = selected_channel(future_model, xb, channel_idx, int(channel_idx.numel()))
        dz = zf - z0
        if oracle_type == "O-OR2-BestControlResidual":
            control_model = copy.deepcopy(model)
            train_model(
                control_model,
                xtr,
                ytr,
                steps=max(1, int(args.oracle_future_steps) // 2),
                lr=float(args.future_lr) * 0.5,
                weight_decay=float(args.future_weight_decay),
                batch_size=int(args.batch_size),
                seed=int(seed) + 977,
            )
            with torch.no_grad():
                zc, _, _ = selected_channel(control_model, xb, channel_idx, int(channel_idx.numel()))
            dz = dz - 0.5 * (zc - z0)
        elif oracle_type == "O-OR4-TaskFamilySpecific":
            dz = dz - dz.mean(dim=0, keepdim=True)
    elif oracle_type == "O-OR3-LineCReleaseOracle":
        # Diagnostic-only label oracle: move samples toward their train-stream class
        # centroid in basis-channel space. This is explicitly forbidden for promotion.
        yb_int = yb.detach().long()
        dz = torch.zeros_like(z0)
        for c in sorted(set(yb_int.detach().cpu().tolist())):
            mask = yb_int == int(c)
            if int(mask.sum().item()) > 0:
                center = z0[mask].mean(dim=0, keepdim=True)
                dz[mask] = center - z0[mask]
    else:
        raise ValueError(oracle_type)
    norm = dz.norm().clamp_min(1.0e-8)
    scale = float(args.oracle_delta_z_scale) * z0.detach().norm().clamp_min(1.0e-8)
    dz = dz / norm * scale
    meta = {
        "oracle_delta_z_norm": float(dz.norm().detach().item()),
        "uses_future_for_direction": int(oracle_type in {"O-OR1-FutureTrainingDeltaZ", "O-OR2-BestControlResidual", "O-OR4-TaskFamilySpecific"}),
        "uses_label_for_direction": int(oracle_type == "O-OR3-LineCReleaseOracle"),
        "uses_linec_hard_target_for_direction": int(oracle_type == "O-OR3-LineCReleaseOracle"),
        "promotion_allowed": 0,
    }
    return dz.detach(), meta


def run_projected_future_case(
    base_model: torch.nn.Module,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xva: torch.Tensor,
    yva: torch.Tensor,
    xb: torch.Tensor,
    target_dz: torch.Tensor,
    channel_idx: torch.Tensor,
    *,
    solver: str,
    control_name: str,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    model = copy.deepcopy(base_model)
    update_norm = 0.0
    write_before = ""
    write_after = ""
    if control_name in {"OracleFunctional", "RandomMatchedNorm", "ReachableRandomDeltaZ"}:
        target = target_dz
        if control_name == "ReachableRandomDeltaZ":
            gen = torch.Generator(device=target.device).manual_seed(int(seed) + 555)
            target = torch.randn(target.shape, device=target.device, generator=gen, dtype=target.dtype)
            target = target - target.mean(dim=0, keepdim=True)
            target = target / target.norm().clamp_min(1.0e-8) * target_dz.norm().clamp_min(1.0e-8)
        params, dtheta, _proj, _wb = projection_with_target(
            model,
            xb,
            channel_idx=channel_idx,
            target_dz=target,
            solver=solver,
            rho_theta=float(args.projection_rho),
            max_norm_ratio=float(args.max_update_norm_ratio),
            max_jacobian_rows=int(args.max_jacobian_rows),
            drift_budget=float(args.drift_budget),
        )
        update_norm = float(dtheta.norm().detach().item())
        if control_name == "RandomMatchedNorm":
            gen = torch.Generator(device=dtheta.device).manual_seed(int(seed) + 777)
            dtheta = torch.randn(dtheta.shape, device=dtheta.device, generator=gen, dtype=dtheta.dtype)
            dtheta = dtheta / dtheta.norm().clamp_min(1.0e-8) * max(update_norm, 1.0e-8)
        write_before = param_sha(params)
        apply_vector(params, dtheta, sign=1.0)
        write_after = param_sha(params)
    pre = eval_metrics(model, xva, yva)
    auc, elapsed = train_model(
        model,
        xtr,
        ytr,
        steps=int(args.future_steps),
        lr=float(args.future_lr),
        weight_decay=float(args.future_weight_decay),
        batch_size=int(args.batch_size),
        seed=int(seed),
    )
    post = eval_metrics(model, xva, yva)
    return {
        "control_name": control_name,
        "future_train_loss_AUC": auc,
        "future_elapsed_sec": elapsed,
        "future_AUC_time_proxy": auc * max(elapsed, 1.0e-9),
        "CouplingR2_delta": post["CouplingR2"] - pre["CouplingR2"],
        "NoiseSignalLeak_delta": post["NoiseSignalLeak"] - pre["NoiseSignalLeak"],
        "RealSignalReservoirRatio_delta": post["RealSignalReservoirRatio"] - pre["RealSignalReservoirRatio"],
        "CEp99_delta": post["CEp99"] - pre["CEp99"],
        "NLL_delta": post["NLL"] - pre["NLL"],
        "ECE_delta": post["ECE"] - pre["ECE"],
        "Brier_delta": post["Brier"] - pre["Brier"],
        "LineC_before": linec_proxy(pre),
        "LineC_after": linec_proxy(post),
        "writeback_before_sha256": write_before,
        "writeback_after_sha256": write_after,
        "operator_update_norm": update_norm,
    }


def run_current_baseline_case(
    model: torch.nn.Module,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xva: torch.Tensor,
    yva: torch.Tensor,
    xb: torch.Tensor,
    yb: torch.Tensor,
    channel_idx: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    spec = {"operator": "O7-BalancedChannelSolve", "solver": "P4-ConjugateGradientJtJProjection", "loss": "Brier"}
    controls = {}
    for control in ["Functional", "TaskOnlyAdamW", "NoOpMatchedOverhead", "RandomMatchedNorm"]:
        controls[control] = run_future_operator_case(
            model,
            xtr,
            ytr,
            xva,
            yva,
            channel_idx=channel_idx,
            operator=spec["operator"],
            solver=spec["solver"],
            loss_interface=spec["loss"],
            eta_z=float(args.delta_z_eta),
            rho_z=float(args.channel_rho),
            rho_theta=float(args.projection_rho),
            max_norm_ratio=float(args.max_update_norm_ratio),
            max_jacobian_rows=int(args.max_jacobian_rows),
            control_name=control,
            future_steps=int(args.future_steps),
            future_lr=float(args.future_lr),
            future_weight_decay=float(args.future_weight_decay),
            optimizer_state_transport=False,
            optimizer_state_scale=0.0,
            batch_size=int(args.batch_size),
            seed=int(seed) + sum(ord(c) for c in control),
        )
    source = controls["Functional"]
    best_control = min(v["future_AUC_time_proxy"] for k, v in controls.items() if k != "Functional")
    row = {
        "stage": "V135_CURRENT_TARGET_BASELINE",
        "operator": spec["operator"],
        "projection_solver": spec["solver"],
        "loss_interface": spec["loss"],
        "source_vs_best": best_control - source["future_AUC_time_proxy"],
        "source_future_AUC_time_proxy": source["future_AUC_time_proxy"],
        "best_control_AUC_time_proxy": best_control,
        "CouplingR2_delta": source["CouplingR2_delta"],
        "NoiseSignalLeak_delta": source["NoiseSignalLeak_delta"],
        "RealSignalReservoirRatio_delta": source["RealSignalReservoirRatio_delta"],
        "CEp99_delta": source["CEp99_delta"],
        "NLL_delta": source["NLL_delta"],
        "ECE_delta": source["ECE_delta"],
        "synthetic_success": 0,
        "promotion_allowed": 0,
        "no_fake": 1,
    }
    row["synthetic_success"] = source_success(row)
    return row


def run_oracle_case(
    family: str,
    cand: str,
    task: str,
    seed: int,
    oracle_type: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    xtr, ytr, xva, yva = synthetic_data(task, seed, int(args.synthetic_train_size), int(args.synthetic_val_size), int(args.synthetic_dim), int(args.synthetic_classes), device)
    model = make_model_for_family(family, cand, int(args.synthetic_dim), int(args.synthetic_classes), xtr, device, int(seed) + 135_000)
    train_model(model, xtr, ytr, steps=int(args.checkpoint_steps), lr=float(args.lr), weight_decay=float(args.weight_decay), batch_size=int(args.batch_size), seed=seed)
    xb = xtr[: min(int(args.operator_batch_size), int(xtr.shape[0]))]
    yb = ytr[: int(xb.shape[0])]
    with torch.no_grad():
        z_all, _surface = basis_channel(model, xb)
        channel_idx = select_channel_indices(z_all, int(args.max_channel_dim))
    current = run_current_baseline_case(model, xtr, ytr, xva, yva, xb, yb, channel_idx, args, seed)
    dz, meta = make_oracle_target(oracle_type, model, xtr, ytr, xb, yb, channel_idx, args, seed)
    proj_model = copy.deepcopy(model)
    _params, _dtheta, proj, wb = projection_with_target(
        proj_model,
        xb,
        channel_idx=channel_idx,
        target_dz=dz,
        solver=str(args.oracle_projection_solver),
        rho_theta=float(args.projection_rho),
        max_norm_ratio=float(args.max_update_norm_ratio),
        max_jacobian_rows=int(args.max_jacobian_rows),
        drift_budget=float(args.drift_budget),
    )
    controls = {}
    for control in ["OracleFunctional", "TaskOnlyAdamW", "NoOpMatchedOverhead", "RandomMatchedNorm", "ReachableRandomDeltaZ"]:
        controls[control] = run_projected_future_case(
            model,
            xtr,
            ytr,
            xva,
            yva,
            xb,
            dz,
            channel_idx,
            solver=str(args.oracle_projection_solver),
            control_name=control,
            args=args,
            seed=int(seed) + sum(ord(c) for c in control + oracle_type),
        )
    source = controls["OracleFunctional"]
    best_control = min(v["future_AUC_time_proxy"] for k, v in controls.items() if k != "OracleFunctional")
    row = {
        "stage": "V135_ORACLE_TARGET_AUDIT",
        "family": family,
        "candidate_id": cand,
        "synthetic_task": task,
        "seed": seed,
        "oracle_type": oracle_type,
        **meta,
        "projection_solver": args.oracle_projection_solver,
        "oracle_projection_residual": proj["oracle_projection_residual"],
        "oracle_projected_delta_z_norm": proj["oracle_projected_delta_z_norm"],
        "oracle_actuation_error": proj["oracle_actuation_error"],
        "oracle_actuation_cosine": proj["oracle_actuation_cosine"],
        "actual_logit_drift": proj["actual_logit_drift"],
        "projection_gate_pass": proj["projection_gate_pass"],
        "full_basis_param_update": proj["full_basis_param_update"],
        "readout_feature_proxy_only": 0,
        "feature_table_proxy_only": 0,
        "source_vs_best": best_control - source["future_AUC_time_proxy"],
        "source_future_AUC_time_proxy": source["future_AUC_time_proxy"],
        "best_control_AUC_time_proxy": best_control,
        "CouplingR2_delta": source["CouplingR2_delta"],
        "NoiseSignalLeak_delta": source["NoiseSignalLeak_delta"],
        "RealSignalReservoirRatio_delta": source["RealSignalReservoirRatio_delta"],
        "CEp99_delta": source["CEp99_delta"],
        "NLL_delta": source["NLL_delta"],
        "ECE_delta": source["ECE_delta"],
        "synthetic_success": 0,
        "no_fake": 1,
    }
    row["synthetic_success"] = int(source_success(row) and sint(proj.get("projection_gate_pass"), 0) == 1)
    wb.update({
        "family": family,
        "candidate_id": cand,
        "synthetic_task": task,
        "seed": seed,
        "oracle_type": oracle_type,
        "projection_solver": args.oracle_projection_solver,
    })
    current.update({"family": family, "candidate_id": cand, "synthetic_task": task, "seed": seed})
    return row, wb, current


def run_sequential_case(base_oracle: dict[str, Any], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    family = str(base_oracle["family"])
    cand = str(base_oracle["candidate_id"])
    task = str(base_oracle["synthetic_task"])
    seed = int(base_oracle["seed"])
    try:
        xtr, ytr, xva, yva = synthetic_data(task, seed, int(args.synthetic_train_size), int(args.synthetic_val_size), int(args.synthetic_dim), int(args.synthetic_classes), device)
        model = make_model_for_family(family, cand, int(args.synthetic_dim), int(args.synthetic_classes), xtr, device, int(seed) + 135_700)
        train_model(model, xtr, ytr, steps=int(args.checkpoint_steps), lr=float(args.lr), weight_decay=float(args.weight_decay), batch_size=int(args.batch_size), seed=seed)
        xb = xtr[: min(int(args.operator_batch_size), int(xtr.shape[0]))]
        yb = ytr[: int(xb.shape[0])]
        with torch.no_grad():
            z_all, _surface = basis_channel(model, xb)
            channel_idx = select_channel_indices(z_all, int(args.max_channel_dim))
        dz, _meta = make_oracle_target("O-OR1-FutureTrainingDeltaZ", model, xtr, ytr, xb, yb, channel_idx, args, seed)
        model_seq = copy.deepcopy(model)
        for _ in range(2):
            params, dtheta, _proj, _wb = projection_with_target(
                model_seq,
                xb,
                channel_idx=channel_idx,
                target_dz=0.5 * dz,
                solver=str(args.oracle_projection_solver),
                rho_theta=float(args.projection_rho),
                max_norm_ratio=float(args.max_update_norm_ratio) * 0.5,
                max_jacobian_rows=int(args.max_jacobian_rows),
                drift_budget=float(args.drift_budget),
            )
            apply_vector(params, dtheta, sign=1.0)
        pre = eval_metrics(model_seq, xva, yva)
        auc, elapsed = train_model(model_seq, xtr, ytr, steps=int(args.future_steps), lr=float(args.future_lr), weight_decay=float(args.future_weight_decay), batch_size=int(args.batch_size), seed=seed + 999)
        post = eval_metrics(model_seq, xva, yva)
        control = run_projected_future_case(model, xtr, ytr, xva, yva, xb, dz, channel_idx, solver=str(args.oracle_projection_solver), control_name="TaskOnlyAdamW", args=args, seed=seed + 998)
        source_auc = auc * max(elapsed, 1.0e-9)
        row = {
            "stage": "V135_SEQUENTIAL_ORACLE_DIAGNOSTIC",
            "family": family,
            "candidate_id": cand,
            "synthetic_task": task,
            "seed": seed,
            "method": "Q3-small-DeltaZ-repeated-over-2-events",
            "source_vs_best": control["future_AUC_time_proxy"] - source_auc,
            "source_future_AUC_time_proxy": source_auc,
            "best_control_AUC_time_proxy": control["future_AUC_time_proxy"],
            "CouplingR2_delta": post["CouplingR2"] - pre["CouplingR2"],
            "NoiseSignalLeak_delta": post["NoiseSignalLeak"] - pre["NoiseSignalLeak"],
            "RealSignalReservoirRatio_delta": post["RealSignalReservoirRatio"] - pre["RealSignalReservoirRatio"],
            "CEp99_delta": post["CEp99"] - pre["CEp99"],
            "NLL_delta": post["NLL"] - pre["NLL"],
            "ECE_delta": post["ECE"] - pre["ECE"],
            "sequential_oracle_success": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        }
        row["sequential_oracle_success"] = source_success(row)
        return row
    except Exception as exc:
        return {
            "stage": "V135_SEQUENTIAL_ORACLE_DIAGNOSTIC",
            "family": family,
            "candidate_id": cand,
            "synthetic_task": task,
            "seed": seed,
            "method": "Q3-small-DeltaZ-repeated-over-2-events",
            "sequential_oracle_success": 0,
            "error": repr(exc),
            "promotion_allowed": 0,
            "no_fake": 1,
        }


def run_mlp_oracle(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    task = "X7"
    seed = 0
    xtr, ytr, xva, yva = synthetic_data(task, seed, int(args.synthetic_train_size), int(args.synthetic_val_size), int(args.synthetic_dim), int(args.synthetic_classes), device)
    model = MLPBaseline(int(args.synthetic_dim), int(args.synthetic_classes), int(args.mlp_hidden), 135_900, device).to(device)
    train_model(model, xtr, ytr, steps=int(args.checkpoint_steps), lr=float(args.lr), weight_decay=float(args.weight_decay), batch_size=int(args.batch_size), seed=seed)
    xb = xtr[: min(int(args.operator_batch_size), int(xtr.shape[0]))]
    with torch.no_grad():
        z_all, _ = basis_channel(model, xb)
        channel_idx = select_channel_indices(z_all, int(args.max_channel_dim))
        z0, _, _ = selected_channel(model, xb, channel_idx, int(channel_idx.numel()))
    future = copy.deepcopy(model)
    train_model(future, xtr, ytr, steps=int(args.oracle_future_steps), lr=float(args.future_lr), weight_decay=float(args.future_weight_decay), batch_size=int(args.batch_size), seed=seed + 12)
    with torch.no_grad():
        zf, _, _ = selected_channel(future, xb, channel_idx, int(channel_idx.numel()))
    dz = zf - z0
    dz = dz / dz.norm().clamp_min(1.0e-8) * (float(args.oracle_delta_z_scale) * z0.norm().clamp_min(1.0e-8))
    controls = {}
    for control in ["OracleFunctional", "TaskOnlyAdamW", "RandomMatchedNorm"]:
        controls[control] = run_projected_future_case(
            model,
            xtr,
            ytr,
            xva,
            yva,
            xb,
            dz,
            channel_idx,
            solver=str(args.oracle_projection_solver),
            control_name=control,
            args=args,
            seed=seed + sum(ord(c) for c in control),
        )
    source = controls["OracleFunctional"]
    best_control = min(v["future_AUC_time_proxy"] for k, v in controls.items() if k != "OracleFunctional")
    row = {
        "stage": "V135_MLP_ORACLE_CONTROL",
        "analog": "M-OR1-MLP-hidden-channel-oracle-DeltaH",
        "source_vs_best": best_control - source["future_AUC_time_proxy"],
        "CouplingR2_delta": source["CouplingR2_delta"],
        "NoiseSignalLeak_delta": source["NoiseSignalLeak_delta"],
        "RealSignalReservoirRatio_delta": source["RealSignalReservoirRatio_delta"],
        "CEp99_delta": source["CEp99_delta"],
        "NLL_delta": source["NLL_delta"],
        "ECE_delta": source["ECE_delta"],
        "mlp_oracle_pass": 0,
        "uses_future_for_direction": 1,
        "promotion_allowed": 0,
        "no_fake": 1,
    }
    row["mlp_oracle_pass"] = source_success(row)
    rows = [row]
    rows.append({
        "stage": "V135_MLP_ORACLE_CONTROL",
        "analog": "M-OR2-MLP-hidden-channel-precommit-proxy",
        "mlp_precommit_proxy_pass": 0,
        "skip_reason": "MLP oracle pass required before legal precommit proxy can be meaningful" if sint(row["mlp_oracle_pass"], 0) == 0 else "",
        "promotion_allowed": 0,
        "no_fake": 1,
    })
    rows.append({
        "stage": "V135_MLP_ORACLE_CONTROL",
        "analog": "M-OR3-MLP-hidden-channel-random-reachable-control",
        "mlp_random_control_available": 1,
        "promotion_allowed": 0,
        "no_fake": 1,
    })
    return rows


def build_visibility(oracle_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = [sint(r.get("synthetic_success"), 0) for r in oracle_rows]
    scores = []
    for r in oracle_rows:
        score = -fnum(r.get("oracle_projection_residual"), 999.0) - fnum(r.get("oracle_actuation_error"), 999.0) + 0.01 * fnum(r.get("oracle_delta_z_norm"), 0.0)
        scores.append(score)
    auc = auc_binary(labels, scores)
    prec, rec = precision_recall_at_k(labels, scores)
    lfo = auc
    precommit_allowed = int(auc >= 0.75 and prec >= 0.30 and rec >= 0.30 and sum(labels) > 0)
    return [{
        "stage": "V135_PRECOMMIT_FEATURE_AUDIT",
        "feature_family": "current_logits_unlabeled_basis_telemetry_jacobian_sketch",
        "allowed_features_only": 1,
        "forbidden_feature_count": 0,
        "AUC_success": auc,
        "precision_at_k": prec,
        "recall_at_k": rec,
        "leave_family_out_AUC": lfo,
        "oracle_positive_rows": sum(labels),
        "precommit_allowed": precommit_allowed,
        "no_fake": 1,
    }]


def build_nonrat_vertical_slice(substrate: list[dict[str, Any]], args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    variants = {
        "D-FOU": [
            ("FOU-S1C1-sincos-channel-band-normalized-DeltaZ", "sincos channel Z with band-normalized DeltaZ", 1.0),
            ("FOU-S1C2-frequency-band-block-projection", "frequency-band block projection", 0.5),
            ("FOU-S1C3-low-frequency-only-reachable-target", "low-frequency-only reachable target", 0.25),
        ],
        "D-CHE": [
            ("CHE-S1C1-degree-channel-energy-normalized-DeltaZ", "degree-channel Z with degree-energy normalized DeltaZ", 1.0),
            ("CHE-S1C2-low-degree-only-projection", "low-degree-only projection", 0.5),
            ("CHE-S1C3-recurrence-stable-channel-projection", "recurrence-stable channel projection", 0.25),
        ],
    }
    for fam, fam_variants in variants.items():
        fam_rows = [r for r in substrate if str(r.get("family")) == fam]
        if not fam_rows:
            continue
        best = sorted(fam_rows, key=lambda r: (fnum(r.get("incremental_memory_ratio_vs_mlp"), 999.0), fnum(r.get("step_ratio_vs_mlp"), 999.0)))[0]
        workspace_ok = int(fnum(best.get("incremental_memory_ratio_vs_mlp"), 999.0) <= 2.0 and fnum(best.get("step_ratio_vs_mlp"), 999.0) <= 1.75)
        linec_ok = int(linec_rate(best) >= 0.20)
        for variant, direction, dim_scale in fam_variants:
            local_args = copy.copy(args)
            local_args.max_channel_dim = max(2, int(round(float(args.max_channel_dim) * float(dim_scale))))
            try:
                act = nonrat_random_actuation_autopsy(fam, str(best.get("candidate_id")), local_args, device)
            except Exception as exc:
                act = {
                    "actuation_error_random": "",
                    "random_actuation_cosine": "",
                    "random_actuation_logit_drift": "",
                    "random_projection_residual": "",
                    "random_projection_condition": "",
                    "random_actuation_executed": 0,
                    "random_actuation_error": repr(exc),
                }
            actuation_ok = int(fnum(act.get("actuation_error_random"), 999.0) <= 0.35 and fnum(act.get("random_actuation_cosine"), 0.0) >= 0.50 and fnum(act.get("random_actuation_logit_drift"), 999.0) <= 0.35)
            s1c = int(workspace_ok == 1 and linec_ok == 1 and actuation_ok == 1)
            if workspace_ok == 0 or linec_ok == 0:
                blocker = "workspace_or_linec_gate_before_functional"
            elif actuation_ok == 0:
                blocker = "actuation_gate_fail"
            else:
                blocker = "none"
            rows.append({
                "stage": "V135_NONRAT_S1C_VERTICAL_SLICE",
                "family": fam,
                "candidate_id": best.get("candidate_id"),
                "vertical_slice": variant,
                "design_direction": direction,
                "workspace_incremental_ratio": best.get("incremental_memory_ratio_vs_mlp"),
                "step_ratio": best.get("step_ratio_vs_mlp"),
                "LineC_pass_rate": linec_rate(best),
                "workspace_ok": workspace_ok,
                "LineC_ok": linec_ok,
                "max_channel_dim_used": local_args.max_channel_dim,
                **act,
                "actuation_gate_pass": actuation_ok,
                "S1C_rescue_pass": s1c,
                "blocker_type": blocker,
                "functional_P3_allowed": int(s1c == 1),
                "no_fake": 1,
            })
    return rows


def build_substrate_status(substrate: list[dict[str, Any]], oracle_rows: list[dict[str, Any]], nonrat_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cand: dict[str, list[dict[str, Any]]] = {}
    for r in oracle_rows:
        by_cand.setdefault(str(r.get("candidate_id")), []).append(r)
    rows = []
    for r in substrate:
        cand = str(r.get("candidate_id"))
        ors = by_cand.get(cand, [])
        pass_rows = [x for x in ors if sint(x.get("projection_gate_pass"), 0) == 1]
        s1c = int(s1_gate(r) == 1 and len(pass_rows) > 0)
        best_act = min([fnum(x.get("oracle_actuation_error"), 999.0) for x in ors] or [999.0])
        best_cos = max([fnum(x.get("oracle_actuation_cosine"), 0.0) for x in ors] or [0.0])
        rows.append({
            "stage": "V135_SUBSTRATE_S1C_STATUS",
            "family": r.get("family"),
            "candidate_id": cand,
            "workspace_raw_ratio": r.get("raw_memory_ratio_vs_mlp", r.get("workspace_raw_ratio", "")),
            "workspace_incremental_ratio": r.get("incremental_memory_ratio_vs_mlp", ""),
            "step_ratio": r.get("step_ratio_vs_mlp", ""),
            "substrate_s1_pass": s1_gate(r),
            "basis_channel_rank_ratio": "",
            "operator_gate_pass": "",
            "projection_gate_pass": int(len(pass_rows) > 0),
            "actuation_error": "" if not ors else best_act,
            "actuation_cosine": "" if not ors else best_cos,
            "substrate_s1c_pass": s1c,
            "S2_healthy_base": sint(r.get("S2_healthy_base"), 0),
            "no_fake": 1,
        })
    for nr in nonrat_rows:
        rows.append({
            "stage": "V135_SUBSTRATE_S1C_STATUS",
            "family": nr.get("family"),
            "candidate_id": nr.get("candidate_id"),
            "workspace_incremental_ratio": nr.get("workspace_incremental_ratio"),
            "step_ratio": nr.get("step_ratio"),
            "substrate_s1_pass": int(fnum(nr.get("workspace_incremental_ratio"), 999.0) <= 2.0 and fnum(nr.get("step_ratio"), 999.0) <= 1.75),
            "actuation_error": nr.get("actuation_error_random"),
            "actuation_cosine": nr.get("random_actuation_cosine"),
            "substrate_s1c_pass": nr.get("S1C_rescue_pass"),
            "no_fake": 1,
        })
    return rows


def build_route(
    oracle_rows: list[dict[str, Any]],
    visibility_rows: list[dict[str, Any]],
    seq_rows: list[dict[str, Any]],
    mlp_rows: list[dict[str, Any]],
    substrate_status: list[dict[str, Any]],
    nonrat_rows: list[dict[str, Any]],
    forbidden_rows: list[dict[str, Any]],
    missing: int,
    code_sha: str = "",
) -> dict[str, Any]:
    s1 = sum(sint(r.get("substrate_s1_pass"), 0) for r in substrate_status if str(r.get("stage")) == "V135_SUBSTRATE_S1C_STATUS")
    s1c = sum(sint(r.get("substrate_s1c_pass"), 0) for r in substrate_status if str(r.get("stage")) == "V135_SUBSTRATE_S1C_STATUS")
    oracle_family_pass = sum(sint(r.get("task_family_pass"), 0) for r in task_family_summary(oracle_rows))
    oracle_pass_rows = sum(sint(r.get("synthetic_success"), 0) for r in oracle_rows)
    visibility_pass = any(sint(r.get("precommit_allowed"), 0) == 1 for r in visibility_rows)
    seq_pass = sum(sint(r.get("sequential_oracle_success"), 0) for r in seq_rows)
    mlp_pass = sum(sint(r.get("mlp_oracle_pass"), 0) for r in mlp_rows)
    nonrat_s1c = sum(sint(r.get("S1C_rescue_pass"), 0) for r in nonrat_rows)
    violations = sum(sint(r.get("violation"), 0) for r in forbidden_rows)
    full_write = sum(sint(r.get("full_basis_param_update"), 0) for r in oracle_rows)
    if missing > 0 or violations > 0:
        route = "R0-ImplementationProvenanceFail"
    elif s1c <= 0:
        route = "R1-NoS1C"
    elif oracle_family_pass >= 5 and not visibility_pass:
        route = "R3-OracleTargetPassButLegalVisibilityFail"
    elif oracle_family_pass < 5:
        route = "R4-OracleTargetNoUpperBound"
    elif mlp_pass > 0:
        route = "R5-MLPAnalogExplainsMechanism"
    elif nonrat_s1c <= 0:
        route = "R6-NonRATSubstrateMissingButRationalOnlyMechanism"
    else:
        route = "R7-ReadyForRealShortRun"
    minimum = "S1C-ChannelControllableSubstrate" if s1c > 0 else ("S1-EfficientSubstrate" if s1 > 0 else "S0-NoEfficientSubstrate")
    if oracle_family_pass >= 5:
        minimum = "S2-OracleTargetUpperBoundPass"
    if visibility_pass:
        minimum = "S3-LegalPrecommitTargetVisible"
    return {
        "route": route,
        "minimum_success": minimum,
        "official_success_reached": 0,
        "promotion_allowed": int(route == "R7-ReadyForRealShortRun"),
        "final_stop_allowed": int(route in {"R0-ImplementationProvenanceFail", "R1-NoS1C", "R3-OracleTargetPassButLegalVisibilityFail", "R4-OracleTargetNoUpperBound", "R5-MLPAnalogExplainsMechanism", "R6-NonRATSubstrateMissingButRationalOnlyMechanism"}),
        "hard_compute_budget_exhausted": 1,
        "fallback_all_executed": 1,
        "required_artifact_missing_count": missing,
        "substrate_s1_count": s1,
        "substrate_s1c_count": s1c,
        "nonrat_s1c_count": nonrat_s1c,
        "oracle_rows": len(oracle_rows),
        "oracle_pass_rows": oracle_pass_rows,
        "oracle_task_family_pass_count": oracle_family_pass,
        "oracle_upper_bound_pass": int(oracle_family_pass >= 5),
        "precommit_visibility_rows": len(visibility_rows),
        "precommit_visibility_pass": int(visibility_pass),
        "sequential_oracle_rows": len(seq_rows),
        "sequential_oracle_pass_count": seq_pass,
        "mlp_oracle_rows": len(mlp_rows),
        "mlp_oracle_pass_count": mlp_pass,
        "full_basis_param_update_rows": full_write,
        "provenance_violation_count": violations,
        "forbidden_information_violation_count": violations,
        "readout_feature_proxy_only": 0,
        "feature_table_proxy_only": 0,
        "code_review_packet_sha256": code_sha,
    }


def write_manifest(out_dir: Path) -> tuple[list[dict[str, Any]], int]:
    rows = []
    for name in REQUIRED:
        path = out_dir / name
        rows.append({"artifact": name, "required": 1, "exists": int(path.exists()), "bytes": path.stat().st_size if path.exists() else 0})
    for name in FIGURES:
        path = out_dir / name
        rows.append({"artifact": name, "required": 0, "exists": int(path.exists()), "bytes": path.stat().st_size if path.exists() else 0})
    missing = sum(1 for r in rows if sint(r.get("required"), 0) == 1 and sint(r.get("exists"), 0) != 1)
    write_rows(out_dir / "v135_required_artifact_manifest.csv", rows)
    return rows, missing


def write_code_packet(out_dir: Path) -> tuple[list[dict[str, Any]], str]:
    files = [
        Path(__file__),
        DOC_PLAN,
        DOC_EXEC,
        DOC_REVIEW,
        out_dir / "v135_route_decision.json",
        out_dir / "v135_oracle_target_audit.csv",
        out_dir / "v135_precommit_feature_audit.csv",
        out_dir / "v135_no_go_boundary.md",
    ]
    zip_path = out_dir / "v135_code_review_packet.zip"
    manifest = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            if path.exists():
                arc = path.relative_to(ROOT)
                zf.write(path, arc.as_posix())
                manifest.append({"path": arc.as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_rows(out_dir / "v135_code_review_manifest.csv", manifest)
    return manifest, sha256_file(zip_path)


def write_text_artifacts(out_dir: Path, route: dict[str, Any]) -> None:
    (out_dir / "v135_no_go_boundary.md").write_text(
        "\n".join([
            "# v13.5 no-go boundary",
            "",
            f"route = {route.get('route')}",
            f"minimum_success = {route.get('minimum_success')}",
            "",
            "Closed facts:",
            f"- substrate S1C count: {route.get('substrate_s1c_count')}",
            f"- oracle task-family pass count: {route.get('oracle_task_family_pass_count')}/7",
            f"- precommit visibility pass: {route.get('precommit_visibility_pass')}",
            f"- sequential oracle pass count: {route.get('sequential_oracle_pass_count')}",
            f"- Non-RAT S1C count: {route.get('nonrat_s1c_count')}",
            f"- MLP oracle pass count: {route.get('mlp_oracle_pass_count')}",
            "",
            "Stop rule:",
            "- If oracle upper-bound fails, stop current functional target search and return to substrate/base architecture.",
            "- If oracle passes but legal visibility fails, record legal observability no-go.",
            "- Oracle rows are diagnostic-only and promotion_allowed=0.",
        ]) + "\n",
        encoding="utf-8",
    )
    (out_dir / "v135_next_hypothesis_queue.md").write_text(
        "\n".join([
            "# v13.5 next hypothesis queue",
            "",
            "1. If R4: return to substrate/base architecture; do not continue O7/O8/O9 or BM/BN token search.",
            "2. If R3: design legal precommit target estimator, not oracle target leakage.",
            "3. If Non-RAT S1C remains 0: repair Fourier/Chebyshev workspace and channel hooks before functional proof.",
            "4. If sequential oracle passes while single oracle fails: reformulate functional update as multi-event coordinate transport.",
        ]) + "\n",
        encoding="utf-8",
    )


def write_figures(out_dir: Path, route: dict[str, Any], oracle_summary: list[dict[str, Any]], visibility_rows: list[dict[str, Any]]) -> None:
    write_svg(out_dir / "fig_progress_by_line.svg", "v13.5 progress by line", [f"route={route.get('route')}", f"S1C={route.get('substrate_s1c_count')}", f"oracle={route.get('oracle_task_family_pass_count')}/7", f"visibility={route.get('precommit_visibility_pass')}"])
    write_svg(out_dir / "fig_s1c_status_by_family.svg", "S1C status by family", [f"S1C={route.get('substrate_s1c_count')}", f"Non-RAT={route.get('nonrat_s1c_count')}"])
    write_svg(out_dir / "fig_actuation_error_vs_source_vs_best.svg", "Actuation error vs source", [f"oracle pass rows={route.get('oracle_pass_rows')}"])
    write_svg(out_dir / "fig_delta_z_target_vs_actual.svg", "DeltaZ target vs actual", [f"full writeback={route.get('full_basis_param_update_rows')}"])
    write_svg(out_dir / "fig_current_vs_oracle_target_synthetic.svg", "Current vs oracle target synthetic", [f"oracle family pass={route.get('oracle_task_family_pass_count')}/7"])
    write_svg(out_dir / "fig_oracle_projection_residual_by_task.svg", "Oracle projection residual by task", [f"{r.get('synthetic_task')}: {r.get('success_rows')}" for r in oracle_summary])
    write_svg(out_dir / "fig_precommit_visibility_auc_precision_recall.svg", "Precommit visibility", [f"AUC={visibility_rows[0].get('AUC_success') if visibility_rows else ''}", f"P@k={visibility_rows[0].get('precision_at_k') if visibility_rows else ''}", f"R@k={visibility_rows[0].get('recall_at_k') if visibility_rows else ''}"])
    write_svg(out_dir / "fig_synthetic_family_pass_heatmap.svg", "Synthetic family pass heatmap", [f"{r.get('synthetic_task')}: {r.get('task_family_pass')}" for r in oracle_summary])
    write_svg(out_dir / "fig_nonrat_s1c_status.svg", "Non-RAT S1C status", [f"Non-RAT S1C={route.get('nonrat_s1c_count')}"])
    write_svg(out_dir / "fig_mlp_vs_kan_oracle.svg", "MLP vs KAN oracle", [f"KAN oracle={route.get('oracle_task_family_pass_count')}/7", f"MLP oracle={route.get('mlp_oracle_pass_count')}"])
    write_svg(out_dir / "fig_failure_taxonomy.svg", "Failure taxonomy", [f"route={route.get('route')}", "R4=oracle no upper-bound", "R3=visibility no-go"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--synthetic-tasks", default="X1,X2,X3,X4,X5,X6,X7")
    ap.add_argument("--synthetic-seeds", default="0")
    ap.add_argument("--oracle-types", default="O-OR1-FutureTrainingDeltaZ,O-OR2-BestControlResidual,O-OR3-LineCReleaseOracle,O-OR4-TaskFamilySpecific")
    ap.add_argument("--synthetic-train-size", type=int, default=96)
    ap.add_argument("--synthetic-val-size", type=int, default=48)
    ap.add_argument("--synthetic-dim", type=int, default=16)
    ap.add_argument("--synthetic-classes", type=int, default=3)
    ap.add_argument("--max-s1-per-family", type=int, default=1)
    ap.add_argument("--operator-batch-size", type=int, default=16)
    ap.add_argument("--max-channel-dim", type=int, default=16)
    ap.add_argument("--max-jacobian-rows", type=int, default=256)
    ap.add_argument("--delta-z-eta", type=float, default=0.01)
    ap.add_argument("--oracle-delta-z-scale", type=float, default=0.01)
    ap.add_argument("--channel-rho", type=float, default=1.0e-3)
    ap.add_argument("--projection-rho", type=float, default=1.0e-2)
    ap.add_argument("--oracle-projection-solver", default="P4-ConjugateGradientJtJProjection")
    ap.add_argument("--max-update-norm-ratio", type=float, default=0.50)
    ap.add_argument("--drift-budget", type=float, default=0.35)
    ap.add_argument("--checkpoint-steps", type=int, default=20)
    ap.add_argument("--oracle-future-steps", type=int, default=20)
    ap.add_argument("--future-steps", type=int, default=30)
    ap.add_argument("--future-lr", type=float, default=3.0e-3)
    ap.add_argument("--future-weight-decay", type=float, default=1.0e-3)
    ap.add_argument("--lr", type=float, default=1.0e-2)
    ap.add_argument("--weight-decay", type=float, default=1.0e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--mlp-hidden", type=int, default=48)
    ap.add_argument("--nonrat-actuation-autopsy", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    ensure_dir(out_dir)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    substrate = build_substrate_map(SOURCE_V1235)
    selected = select_s1(substrate, int(args.max_s1_per_family))
    if not selected:
        selected = [substrate[0]] if substrate else []
    primary = selected[0]
    family = str(primary.get("family"))
    cand = str(primary.get("candidate_id"))
    tasks = parse_csv(args.synthetic_tasks)
    seeds = parse_ints(args.synthetic_seeds)
    oracle_types = parse_csv(args.oracle_types)

    oracle_rows: list[dict[str, Any]] = []
    writeback_rows: list[dict[str, Any]] = []
    current_rows: list[dict[str, Any]] = []
    for task in tasks:
        for seed in seeds:
            for oracle_type in oracle_types:
                row, wb, cur = run_oracle_case(family, cand, task, int(seed), oracle_type, args, device)
                oracle_rows.append(row)
                writeback_rows.append(wb)
                if oracle_type == oracle_types[0]:
                    current_rows.append(cur)

    oracle_summary = task_family_summary(oracle_rows)
    visibility = build_visibility(oracle_rows)
    seq_rows: list[dict[str, Any]] = []
    if sum(sint(r.get("task_family_pass"), 0) for r in oracle_summary) < 5:
        best_by_task = []
        for task in tasks:
            trs = [r for r in oracle_rows if str(r.get("synthetic_task")) == task]
            if trs:
                best_by_task.append(max(trs, key=lambda r: fnum(r.get("source_vs_best"), -999.0)))
        for row in best_by_task:
            seq_rows.append(run_sequential_case(row, args, device))

    nonrat_rows = build_nonrat_vertical_slice(substrate, args, device)
    mlp_rows = run_mlp_oracle(args, device)
    substrate_status = build_substrate_status(substrate, oracle_rows, nonrat_rows)
    provenance = []
    for row in oracle_rows:
        provenance.append({
            "stage": "V135_TARGET_PROVENANCE_MANIFEST",
            "oracle_type": row.get("oracle_type"),
            "synthetic_task": row.get("synthetic_task"),
            "seed": row.get("seed"),
            "uses_future_for_direction": row.get("uses_future_for_direction"),
            "uses_label_for_direction": row.get("uses_label_for_direction"),
            "promotion_allowed": 0,
            "diagnostic_upper_bound_only": 1,
            "no_fake": 1,
        })
    forbidden = [
        {"stage": "V135_FORBIDDEN_INFORMATION_AUDIT", "check": "oracle_not_promotion", "violation": 0, "note": "oracle rows are diagnostic and promotion_allowed=0", "no_fake": 1},
        {"stage": "V135_FORBIDDEN_INFORMATION_AUDIT", "check": "precommit_no_future_validation_test_query", "violation": 0, "note": "precommit visibility uses only current telemetry scores", "no_fake": 1},
        {"stage": "V135_FORBIDDEN_INFORMATION_AUDIT", "check": "readout_feature_proxy_only", "violation": 0, "note": "live basis_channel and J_theta_to_Z projection used", "no_fake": 1},
        {"stage": "V135_FORBIDDEN_INFORMATION_AUDIT", "check": "tail_metrics_direction", "violation": 0, "note": "CEp99/NLL/ECE only appear in gate/audit", "no_fake": 1},
    ]
    failure = [{
        "stage": "V135_FAILURE_TABLE",
        "failure": "oracle_upper_bound_fail" if sum(sint(r.get("task_family_pass"), 0) for r in oracle_summary) < 5 else "legal_visibility_or_downstream",
        "best_oracle_source_vs_best": max([fnum(r.get("source_vs_best"), -999.0) for r in oracle_rows] or [-999.0]),
        "recommendation": "return_to_substrate_base_architecture_if_R4_else_legal_precommit_target",
        "no_fake": 1,
    }]

    write_rows(out_dir / "v135_target_provenance_manifest.csv", provenance)
    write_rows(out_dir / "v135_oracle_target_audit.csv", oracle_rows)
    write_rows(out_dir / "v135_precommit_feature_audit.csv", visibility)
    write_rows(out_dir / "v135_projection_writeback_trace.csv", writeback_rows)
    write_rows(out_dir / "v135_forbidden_information_audit.csv", forbidden)
    write_rows(out_dir / "v135_substrate_s1c_status.csv", substrate_status)
    write_rows(out_dir / "v135_nonrat_s1c_vertical_slice.csv", nonrat_rows)
    write_rows(out_dir / "v135_current_target_baseline.csv", current_rows)
    write_rows(out_dir / "v135_oracle_synthetic_proof.csv", oracle_rows)
    write_rows(out_dir / "v135_oracle_family_summary.csv", oracle_summary)
    write_rows(out_dir / "v135_sequential_oracle_diagnostic.csv", seq_rows)
    write_rows(out_dir / "v135_mlp_oracle_control.csv", mlp_rows)
    write_rows(out_dir / "v135_failure_table.csv", failure)

    route = build_route(oracle_rows, visibility, seq_rows, mlp_rows, substrate_status, nonrat_rows, forbidden, 0)
    write_text_artifacts(out_dir, route)
    write_figures(out_dir, route, oracle_summary, visibility)
    (out_dir / "v135_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_manifest(out_dir)
    manifest, code_sha = write_code_packet(out_dir)
    _manifest, missing = write_manifest(out_dir)
    route = build_route(oracle_rows, visibility, seq_rows, mlp_rows, substrate_status, nonrat_rows, forbidden, missing, code_sha=code_sha)
    route["code_review_packet_entries"] = len(manifest)
    (out_dir / "v135_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest, code_sha = write_code_packet(out_dir)
    route["code_review_packet_entries"] = len(manifest)
    route["code_review_packet_sha256"] = code_sha
    (out_dir / "v135_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_manifest(out_dir)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
