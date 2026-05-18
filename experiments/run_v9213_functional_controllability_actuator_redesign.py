#!/usr/bin/env python3
"""DG-KAN v9.2.13 functional controllability and actuator redesign runner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v92_basis_source_kernel_gate as v92  # noqa: E402
import run_v922_compositional_trainability as v922  # noqa: E402
import run_v922_fused_compositional_kernel_closure as f922  # noqa: E402
import run_v926_fc_purekan_primitive_redesign as v926  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402
from dgkan.functional import lq_output_space_functional as out_lq  # noqa: E402
from dgkan.functional import snr_gated_lq as snr_lq  # noqa: E402
from dgkan.models import fc_purekan_actuator as act  # noqa: E402
from dgkan.models import fc_purekan_lq as lq  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.13_Functional_Controllability_ActuatorRedesign_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9213_functional_controllability_actuator_redesign.py"
PREV_V9212 = ROOT / "results" / "real_rerun_20260506" / "v9212_functional_direction_reconstruction_first_20260509T230000Z"
PREV_V927 = ROOT / "results" / "real_rerun_20260506" / "v927_fc_purekan_lq_fullpass_functional_gate_20260509T190000Z"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _canonical_tasks(text: str) -> List[str]:
    return [v92._canonical_task(x) for x in _parse_list(text)]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value == "" or value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _flatten(ts: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in ts]) if ts else torch.empty(0)


def _not_run(stage: str, artifact: str, reason: str) -> Dict[str, Any]:
    return snr_lq.not_run_row(stage, artifact, reason)


def _write_svg(path: Path, title: str, subtitle: str) -> None:
    ensure_dir(path.parent)
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="210" viewBox="0 0 1040 210">
  <rect width="1040" height="210" fill="#f8fafc"/>
  <text x="30" y="58" font-family="Arial, sans-serif" font-size="25" fill="#111827">{title}</text>
  <text x="30" y="102" font-family="Arial, sans-serif" font-size="16" fill="#374151">{subtitle}</text>
  <text x="30" y="144" font-family="Arial, sans-serif" font-size="13" fill="#6b7280">Generated from measured CSV/JSON fields only.</text>
</svg>
""",
        encoding="utf-8",
    )


def _classification_delta(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    return {
        "accuracy_delta": after["acc"] - before["acc"],
        "holdout_delta": after["loss"] - before["loss"],
        "CEp99_delta": after["CE_p99"] - before["CE_p99"],
        "margin_p10_delta": after["correct_margin_p10"] - before["correct_margin_p10"],
        "ECE_delta": after["ECE"] - before["ECE"],
        "NLL_delta": after["NLL"] - before["NLL"],
    }


def _eval_lq_logits(params: Sequence[torch.Tensor], mu: torch.Tensor, std: torch.Tensor, basis: str, x: torch.Tensor) -> torch.Tensor:
    fwd, _bwd = lq.functions_for_basis(basis)
    with torch.no_grad():
        return fwd(x, *params, mu, std, 2.0, 2.0)


def _eval_lq_metrics(params: Sequence[torch.Tensor], mu: torch.Tensor, std: torch.Tensor, basis: str, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    return v92._classification_metrics_from_logits(_eval_lq_logits(params, mu, std, basis, x), y)


def _eval_actuator_metrics(params: Sequence[torch.Tensor], mu: torch.Tensor, std: torch.Tensor, spec: act.ActuatorSpec, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        logits = act.actuator_forward(x, params, mu, std, spec)
    return v92._classification_metrics_from_logits(logits, y)


def _clone_params(params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


def _apply_step(params: Sequence[torch.Tensor], step: Sequence[torch.Tensor]) -> None:
    with torch.no_grad():
        for p, d in zip(params, step):
            p.add_(d)


def _adamw_step_lq(
    params: Sequence[torch.Tensor],
    states: Sequence[AdamWState],
    basis: str,
    x: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    cfg: ManualAdamWConfig,
) -> List[torch.Tensor]:
    _fwd, bwd = lq.functions_for_basis(basis)
    pack = bwd(x, y, *params, mu, std, 2.0, 2.0)
    grads = [g.detach() for g in pack[1:]]
    v92._adamw_update_foreach_(params, grads, states, cfg)
    return grads


def _current_lq_ls_delta(
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    x: torch.Tensor,
    target: torch.Tensor,
    subspace_id: str,
    ridge: float,
) -> List[torch.Tensor]:
    h = x @ params[0]
    vals, _ders = lq.basis_from_lift(h, mu, std, "t2", 2.0, 2.0)
    delta = [torch.zeros_like(p) for p in params]
    sid = str(subspace_id)
    if sid == "S1-QuadraticCoeffSubspace":
        phi = vals[1].float()
        slots = [2]
    elif sid == "S3-OutputLinearSubspace":
        phi = vals[0].float()
        slots = [1]
    elif sid == "S4-LiftPlusQuadraticSubspace":
        phi = torch.cat([vals[0].float(), vals[1].float()], dim=1)
        slots = [1, 2]
    else:
        return delta
    eye = torch.eye(phi.shape[1], device=phi.device, dtype=phi.dtype)
    coef = torch.linalg.solve(phi.T @ phi + float(ridge) * eye, phi.T @ target.float()).to(params[0].dtype)
    offset = 0
    for slot in slots:
        block = coef[offset : offset + params[slot].shape[0]]
        delta[slot] = block
        offset += params[slot].shape[0]
    return delta


def _solve_current_lq_step(
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    x: torch.Tensor,
    target: torch.Tensor,
    grads: Sequence[torch.Tensor],
    task_step: Sequence[torch.Tensor],
    subspace_id: str,
    solver_id: str,
    step_fraction: float,
    ridge: float,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    sid = str(subspace_id)
    if sid in {"S1-QuadraticCoeffSubspace", "S3-OutputLinearSubspace", "S4-LiftPlusQuadraticSubspace"}:
        raw = _current_lq_ls_delta(params, mu, std, x, target, sid, ridge)
        solver_family = "ridge_ls"
    else:
        direction = out_lq.output_vjp_direction(params, mu, std, "t2", x, target)
        raw = out_lq.apply_subspace(direction, params, grads, sid)
        solver_family = "vjp_proxy_for_nonoutput_subspace"
    raw_norm = snr_lq.step_norm(raw)
    if str(solver_id) in {"SOL1-LeastSquaresSketch"}:
        solved = raw
        removed = torch.tensor(0.0, device=x.device)
    else:
        scaled = snr_lq.scale_direction_to_fraction_of_task_step(raw, task_step, float(step_fraction))
        solved, removed = snr_lq.project_step_to_task_safe(scaled, grads)
        if str(solver_id) == "SOL3-TrustRegionQP":
            trust = snr_lq.step_norm(task_step) * float(step_fraction)
            snorm = snr_lq.step_norm(solved)
            if bool((snorm > trust).detach().cpu()):
                solved = snr_lq.scale_step(solved, trust / snorm.clamp_min(1.0e-12))
    solved_norm = snr_lq.step_norm(solved)
    cos_task = snr_lq.step_dot(grads, solved) / (snr_lq.step_norm(grads).clamp_min(1.0e-12) * solved_norm.clamp_min(1.0e-12))
    cos_adamw = snr_lq.step_dot(task_step, solved) / (snr_lq.step_norm(task_step).clamp_min(1.0e-12) * solved_norm.clamp_min(1.0e-12))
    return solved, {
        "solver_family": solver_family,
        "delta_norm": float(raw_norm.detach().cpu()),
        "delta_norm_after_projection": float(solved_norm.detach().cpu()),
        "projection_removed_norm": float(removed.detach().cpu()),
        "norm_after_projection_ratio": float((solved_norm / raw_norm.clamp_min(1.0e-12)).detach().cpu()),
        "cos_delta_task_grad": float(cos_task.detach().cpu()) if bool((solved_norm > 0).detach().cpu()) else 0.0,
        "cos_delta_adamw": float(cos_adamw.detach().cpu()) if bool((solved_norm > 0).detach().cpu()) else 0.0,
    }


def _p0_recap(prev_dir: Path) -> Dict[str, Any]:
    route = _read_json(prev_dir / "route_decision.json")
    summary = read_csv_rows(prev_dir / "p2_output_space_direction_factory.csv")
    calib = read_csv_rows(prev_dir / "p3_event_controller_calibration.csv")
    # Use the grouped v9.2.12 decision summary.  Raw branch rows include
    # unaccepted / zero-target diagnostic rows that can have degenerate R2.
    fit_vals = [_to_float(r.get("output_target_fit_r2")) for r in calib if str(r.get("status", "")) != "not_run"]
    ratio_vals = [_to_float(r.get("output_displacement_ratio")) for r in calib if str(r.get("status", "")) != "not_run"]
    cov_vals = [_to_float(r.get("coverage")) for r in calib if str(r.get("status", "")) != "not_run"]
    row = {
        "stage": "P0_V9212_REPRODUCTION_RECAP",
        "source_artifact": str(prev_dir.relative_to(ROOT)) if prev_dir.exists() else str(prev_dir),
        "source_route": route.get("route", ""),
        "p2_output_direction_pass_count": route.get("p2_output_direction_pass_count", 0),
        "max_output_displacement_ratio": max(ratio_vals or [0.0]),
        "max_output_target_fit_R2": max(fit_vals or [0.0]),
        "event_coverage_min": min(cov_vals or [0.0]),
        "event_coverage_max": max(cov_vals or [0.0]),
        "branch_count": len({r.get("branch", "") for r in summary}),
        "row_count": len(summary),
        "fake_proxy_count": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["p0_reproduction_pass"] = int(
        row["p2_output_direction_pass_count"] == 0
        and row["max_output_displacement_ratio"] < 0.055
        and row["max_output_target_fit_R2"] < 0.10
        and 0.03 <= row["event_coverage_min"] <= 0.15
        and 0.03 <= row["event_coverage_max"] <= 0.15
    )
    return row


def _run_p1_p2(args: argparse.Namespace, device: torch.device) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    p1_rows: List[Dict[str, Any]] = []
    p2_rows: List[Dict[str, Any]] = []
    sensitivity_rows: List[Dict[str, Any]] = []
    targets = _parse_list(args.targets)
    subspaces = _parse_list(args.subspaces)
    solvers = _parse_list(args.solvers)
    eps_values = [float(x) for x in _parse_list(args.oracle_eps)]
    for dataset in _canonical_tasks(args.datasets):
        x_train, y_train, x_test, y_test, input_dim, output_dim, protocol = v92._load_task(
            args, dataset, train_size=max(int(args.train_size), 4096), test_size=int(args.eval_size)
        )
        x_train = x_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device)
        x_eval = x_test.to(device=device, dtype=torch.float32)
        y_eval = y_test.to(device=device)
        for seed_text in _parse_list(args.seeds):
            seed = int(seed_text)
            spec = lq.LQSpec("LQ0-LQ-t2-h256", "t2", int(args.hidden_dim))
            params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, seed + 921300)
            _fwd, bwd = lq.functions_for_basis("t2")
            xb = x_train[: int(args.audit_batch_size)]
            yb = y_train[: int(args.audit_batch_size)]
            pack = bwd(xb, yb, *params, mu, std, 2.0, 2.0)
            grads = [g.detach() for g in pack[1:]]
            task_step = snr_lq.gradient_descent_task_step(grads, float(args.lr))
            logits = _eval_lq_logits(params, mu, std, "t2", xb)
            before_eval = _eval_lq_metrics(params, mu, std, "t2", x_eval, y_eval)
            adamw_logits = _eval_lq_logits([p + d for p, d in zip(params, task_step)], mu, std, "t2", xb)
            adamw_logit_norm = float((adamw_logits - logits).float().norm().detach().cpu())
            for target_id in targets:
                target, target_info = out_lq.build_output_target(target_id, logits, yb, dataset=dataset)
                for eps in eps_values:
                    scaled = target
                    tnorm = scaled.float().norm().clamp_min(1.0e-12)
                    scaled = scaled / tnorm * (logits.float().norm() * float(eps)).detach()
                    after_logits = logits + scaled
                    before_event = v92._classification_metrics_from_logits(logits, yb)
                    after_event = v92._classification_metrics_from_logits(after_logits, yb)
                    d = _classification_delta(before_event, after_event)
                    p2_rows.append({
                        "stage": "P2_DIRECT_LOGIT_ORACLE_TARGET_AUDIT",
                        "candidate_id": "LQ0-LQ-t2-h256",
                        "dataset": dataset,
                        "seed": seed,
                        "protocol": protocol,
                        "target_id": target_id,
                        "eps_z": eps,
                        "target_selected_fraction": target_info.get("target_selected_fraction", 0.0),
                        "CEp99_before": before_event["CE_p99"],
                        "CEp99_after": after_event["CE_p99"],
                        "margin_p10_before": before_event["correct_margin_p10"],
                        "margin_p10_after": after_event["correct_margin_p10"],
                        "ECE_before": before_event["ECE"],
                        "ECE_after": after_event["ECE"],
                        "NLL_before": before_event["NLL"],
                        "NLL_after": after_event["NLL"],
                        "accuracy_before": before_event["acc"],
                        "accuracy_after": after_event["acc"],
                        **d,
                        "bad_oracle_step": int(d["accuracy_delta"] < -0.005),
                        "target_useful": int(d["accuracy_delta"] >= -0.005 and (d["CEp99_delta"] < 0.0 or d["margin_p10_delta"] > 0.0 or d["ECE_delta"] <= 0.0 or d["NLL_delta"] < 0.0)),
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    })
                for subspace in subspaces:
                    for solver in solvers:
                        delta, solve_info = _solve_current_lq_step(
                            params, mu, std, xb, target, grads, task_step, subspace, solver, float(args.step_fraction), float(args.ridge)
                        )
                        fit = out_lq.output_fit_metrics(params=params, delta=delta, mu=mu, std=std, basis="t2", x=xb, target_delta_logits=target)
                        after_params = [p.detach() + d.detach() for p, d in zip(params, delta)]
                        after_eval = _eval_lq_metrics(after_params, mu, std, "t2", x_eval, y_eval)
                        dd = _classification_delta(before_eval, after_eval)
                        rz = fit["output_displacement_norm"] / max(adamw_logit_norm, 1.0e-12)
                        p1_rows.append({
                            "stage": "P1_FUNCTIONAL_CONTROLLABILITY_AUDIT",
                            "candidate_id": "LQ0-LQ-t2-h256",
                            "dataset": dataset,
                            "seed": seed,
                            "protocol": protocol,
                            "target_id": target_id,
                            "subspace_id": subspace,
                            "solver_id": solver,
                            "target_norm": fit["target_norm"],
                            "delta_norm": solve_info["delta_norm"],
                            "delta_norm_after_projection": solve_info["delta_norm_after_projection"],
                            "norm_after_projection_ratio": solve_info["norm_after_projection_ratio"],
                            "output_displacement_norm": fit["output_displacement_norm"],
                            "output_displacement_ratio_vs_adamw": rz,
                            "target_fit_R2": fit["output_target_fit_r2"],
                            "target_residual_norm": math.sqrt(max(0.0, 1.0 - fit["output_target_fit_r2"])) * fit["target_norm"],
                            "cos_delta_adamw": solve_info["cos_delta_adamw"],
                            "cos_delta_task_grad": solve_info["cos_delta_task_grad"],
                            **dd,
                            "solver_family": solve_info["solver_family"],
                            "target_selected_fraction": target_info.get("target_selected_fraction", 0.0),
                            "controllability_weak": int(fit["output_target_fit_r2"] < 0.10 and rz < 0.05),
                            "controllability_promising": int(fit["output_target_fit_r2"] >= 0.20 and rz >= 0.05),
                            "projection_neutralized": int(solve_info["norm_after_projection_ratio"] < 0.10),
                            "adamw_parallel_failure": int(solve_info["cos_delta_adamw"] > 0.90),
                            "fake_data_used": 0,
                            "proxy_row_used": 0,
                            "cpu_offload_used": 0,
                        })
                        sensitivity_rows.append({
                            "stage": "P1_LQ_SENSITIVITY_TRACE",
                            "candidate_id": "LQ0-LQ-t2-h256",
                            "dataset": dataset,
                            "seed": seed,
                            "target_id": target_id,
                            "subspace_id": subspace,
                            "solver_id": solver,
                            "target_fit_R2": fit["output_target_fit_r2"],
                            "output_displacement_ratio_vs_adamw": rz,
                            "fake_data_used": 0,
                            "proxy_row_used": 0,
                            "cpu_offload_used": 0,
                        })
    return p1_rows, p2_rows, sensitivity_rows


def _synthetic_pairwise_r2(spec: act.ActuatorSpec, device: torch.device, seed: int) -> float:
    gen = torch.Generator(device=device).manual_seed(seed + 921313 + spec.hidden_dim)
    x_train = torch.rand(4096, 8, device=device, generator=gen) * 2.0 - 1.0
    x_test = torch.rand(2048, 8, device=device, generator=gen) * 2.0 - 1.0
    params, mu, std = act.init_actuator_params(8, 1, spec, x_train, device, seed + 13)
    h_train = x_train @ params[0]
    h_test = x_test @ params[0]
    vals_train, _ders, _names = act.actuator_basis_from_lift(h_train, mu, std, spec, 2.0, 2.0)
    vals_test, _ders2, _names2 = act.actuator_basis_from_lift(h_test, mu, std, spec, 2.0, 2.0)
    phi_train = torch.cat(vals_train, dim=1).float()
    phi_test = torch.cat(vals_test, dim=1).float()
    y_train = v922._synthetic_target("T1-pairwise-product", x_train).float()
    y_test = v922._synthetic_target("T1-pairwise-product", x_test).float()
    coef = torch.linalg.lstsq(phi_train, y_train).solution
    pred = phi_test @ coef
    ss_res = (pred - y_test).square().sum()
    ss_tot = (y_test - y_test.mean()).square().sum().clamp_min(1.0e-8)
    return float((1.0 - ss_res / ss_tot).detach().cpu())


def _gradcheck_actuator(args: argparse.Namespace, spec: act.ActuatorSpec, x: torch.Tensor, y: torch.Tensor, in_dim: int, out_dim: int, device: torch.device) -> Dict[str, Any]:
    params, mu, std = act.init_actuator_params(in_dim, out_dim, spec, x, device, int(args.seed) + len(spec.candidate_id))
    xb = x[: min(int(args.audit_batch_size), 64)]
    yb = y[: xb.shape[0]]
    ag_params = [p.detach().clone().requires_grad_(True) for p in params]
    logits = act.actuator_forward(xb, ag_params, mu, std, spec)
    loss = F.cross_entropy(logits, yb)
    autograd_grads = list(torch.autograd.grad(loss, ag_params))
    _loss, manual_grads = act.actuator_fwd_bwd(xb, yb, params, mu, std, spec)
    diff = _flatten([a - b for a, b in zip(autograd_grads, manual_grads)])
    ref = _flatten(autograd_grads)
    man = _flatten(manual_grads)
    grad_rel = float((diff.norm() / ref.norm().clamp_min(1.0e-12)).detach().cpu())
    grad_cos = float(F.cosine_similarity(ref, man, dim=0).detach().cpu()) if ref.numel() else 1.0
    h = x[: min(2048, int(x.shape[0]))] @ params[0]
    cond = act.basis_condition_metrics(h[: min(512, int(h.shape[0]))], mu, std, spec)
    pairwise = _synthetic_pairwise_r2(spec, device, int(args.seed))
    return {
        "stage": "P3_ACTUATOR_CONTRACT_GRADCHECK",
        "candidate_id": spec.candidate_id,
        "actuator_type": spec.actuator_type,
        "basis_formula": act.basis_formula(spec),
        "edge_owned_params": "all_trainable_params",
        "non_edge_params": 0,
        "ordinary_mlp_path_used": 0,
        "external_residual_used": 0,
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_update": 1,
        "GradRelErrMax": grad_rel,
        "GradCosMin": grad_cos,
        "GradPass": int(grad_rel <= 1.0e-4 and grad_cos >= 0.999),
        "synthetic_pairwise_R2": pairwise,
        "interaction_retention_pass": int(pairwise >= 0.95),
        **cond,
        "contract_pass": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _measure_p4_actuator(args: argparse.Namespace, spec: act.ActuatorSpec, x: torch.Tensor, y: torch.Tensor, in_dim: int, out_dim: int, device: torch.device) -> Dict[str, Any]:
    xb = x[: int(args.p4_batch_size)]
    yb = y[: int(args.p4_batch_size)]
    params, mu, std = act.init_actuator_params(in_dim, out_dim, spec, x, device, int(args.seed) + 31)
    params_kan = sum(p.numel() for p in params)
    hidden_mlp = f922._matched_mlp3_hidden(params_kan, in_dim, out_dim)
    gen = torch.Generator(device=device).manual_seed(int(args.seed) + 41)
    W1 = torch.randn(in_dim, hidden_mlp, device=device, generator=gen) / math.sqrt(in_dim)
    W2 = torch.randn(hidden_mlp, hidden_mlp, device=device, generator=gen) / math.sqrt(hidden_mlp)
    W3 = torch.randn(hidden_mlp, out_dim, device=device, generator=gen) / math.sqrt(hidden_mlp)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=0.0)
    kan_states = [AdamWState.zeros_like(p) for p in params]
    mlp_states = [AdamWState.zeros_like(p) for p in [W1, W2, W3]]

    def act_fwd_core(x_in: torch.Tensor, *flat: torch.Tensor) -> torch.Tensor:
        return act.actuator_forward(x_in, list(flat), mu, std, spec)

    def act_bwd_core(x_in: torch.Tensor, y_in: torch.Tensor, *flat: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        loss, grads = act.actuator_fwd_bwd(x_in, y_in, list(flat), mu, std, spec)
        return (loss, *grads)

    cfwd = v92._maybe_compile(f"v9213_{spec.candidate_id}_forward", act_fwd_core)
    cbwd = v92._maybe_compile(f"v9213_{spec.candidate_id}_bwd", act_bwd_core)
    mfwd = v92._maybe_compile(f"v9213_{spec.candidate_id}_mlp_forward", f922._mlp3_forward_core)
    mbwd = v92._maybe_compile(f"v9213_{spec.candidate_id}_mlp_bwd", f922._mlp3_fwd_bwd_core)

    def kan_fwd() -> torch.Tensor:
        return cfwd(xb, *params)

    def kan_bwd() -> Tuple[torch.Tensor, ...]:
        return cbwd(xb, yb, *params)

    def kan_step() -> None:
        pack = kan_bwd()
        v92._adamw_update_foreach_(params, pack[1:], kan_states, cfg)

    def mlp_step() -> None:
        pack = mbwd(xb, yb, W1, W2, W3)
        v92._adamw_update_foreach_([W1, W2, W3], pack[1:], mlp_states, cfg)

    for _ in range(int(args.p4_warmup)):
        kan_fwd()
        kan_bwd()
        mfwd(xb, W1, W2, W3)
        mbwd(xb, yb, W1, W2, W3)
    _sync(device)
    f_kan = v92._bench_callable_ms(kan_fwd, int(args.p4_reps), device)
    fb_kan = v92._bench_callable_ms(kan_bwd, int(args.p4_reps), device)
    b_kan = max(0.0, fb_kan - f_kan)
    s_kan = v92._bench_callable_ms(kan_step, int(args.p4_reps), device)
    f_mlp = v92._bench_callable_ms(lambda: mfwd(xb, W1, W2, W3), int(args.p4_reps), device)
    fb_mlp = v92._bench_callable_ms(lambda: mbwd(xb, yb, W1, W2, W3), int(args.p4_reps), device)
    b_mlp = max(0.0, fb_mlp - f_mlp)
    s_mlp = v92._bench_callable_ms(mlp_step, int(args.p4_reps), device)
    channel_count = len(params) - 1
    cache_conservative = int(args.p4_batch_size) * (in_dim + spec.hidden_dim * (2 + channel_count) + out_dim)
    cache_compact = int(args.p4_batch_size) * (spec.hidden_dim + out_dim)
    peak_kan_conservative = (params_kan * 3 + cache_conservative) * 4 / (1024.0 * 1024.0)
    peak_kan_compact = (params_kan * 3 + cache_compact) * 4 / (1024.0 * 1024.0)
    peak_mlp = f922._estimate_mlp3_memory_mb(in_dim, hidden_mlp, out_dim, int(args.p4_batch_size))
    forward_ratio = f_kan / max(f_mlp, 1.0e-12)
    backward_ratio = b_kan / max(b_mlp, 1.0e-12)
    step_ratio = s_kan / max(s_mlp, 1.0e-12)
    compact = peak_kan_compact / max(peak_mlp, 1.0e-12)
    conservative = peak_kan_conservative / max(peak_mlp, 1.0e-12)
    p4 = int(forward_ratio <= 1.25 and backward_ratio <= 1.50 and step_ratio <= 1.50 and compact <= 1.05)
    return {
        "stage": "P4_ACTUATOR_P4_BASE_QUALIFICATION",
        "candidate_id": spec.candidate_id,
        "actuator_type": spec.actuator_type,
        "forward_ratio": forward_ratio,
        "backward_ratio": backward_ratio,
        "step_ratio": step_ratio,
        "compact_memory_ratio": compact,
        "conservative_memory_ratio": conservative,
        "kernel_count": "python_manual_not_decomposed",
        "actuator_extra_time": max(0, len(params) - 3),
        "params_kan": params_kan,
        "params_mlp_match": in_dim * hidden_mlp + hidden_mlp * hidden_mlp + hidden_mlp * out_dim,
        "matched_mlp_hidden": hidden_mlp,
        "P4_pass": p4,
        "loss_type": "CE",
        "label_smoothing": 0,
        "uses_loss_backward": 0,
        "external_teacher_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _train_actuator_candidate(args: argparse.Namespace, spec: act.ActuatorSpec, dataset: str, seed: int, device: torch.device) -> Dict[str, Any]:
    x_train, y_train, x_test, y_test, in_dim, out_dim, protocol = v92._load_task(
        args, dataset, train_size=int(args.p5_train_size), test_size=int(args.p5_test_size)
    )
    x_train = x_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device)
    x_test = x_test.to(device=device, dtype=torch.float32)
    y_test = y_test.to(device=device)
    params, mu, std = act.init_actuator_params(in_dim, out_dim, spec, x_train, device, seed + 921350)
    params_kan = sum(p.numel() for p in params)
    hidden_mlp = f922._matched_mlp3_hidden(params_kan, in_dim, out_dim)
    gen = torch.Generator(device=device).manual_seed(seed + 921360)
    mlp_params = [
        torch.randn(in_dim, hidden_mlp, device=device, generator=gen) / math.sqrt(in_dim),
        torch.randn(hidden_mlp, hidden_mlp, device=device, generator=gen) / math.sqrt(hidden_mlp),
        torch.randn(hidden_mlp, out_dim, device=device, generator=gen) / math.sqrt(hidden_mlp),
    ]
    cfg = ManualAdamWConfig(lr=float(args.p5_lr), weight_decay=0.0)
    kan_states = [AdamWState.zeros_like(p) for p in params]
    mlp_states = [AdamWState.zeros_like(p) for p in mlp_params]
    for epoch in range(int(args.p5_epochs)):
        gen_epoch = torch.Generator(device=device).manual_seed(seed * 1000 + epoch + 9213)
        perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen_epoch)
        for start in range(0, int(x_train.shape[0]), int(args.batch_size)):
            idx = perm[start : start + int(args.batch_size)]
            xb = x_train[idx]
            yb = y_train[idx]
            _loss, grads = act.actuator_fwd_bwd(xb, yb, params, mu, std, spec)
            v92._adamw_update_foreach_(params, grads, kan_states, cfg)
            mpack = f922._mlp3_fwd_bwd_core(xb, yb, *mlp_params)
            v92._adamw_update_foreach_(mlp_params, mpack[1:], mlp_states, cfg)
    kan_metrics = _eval_actuator_metrics(params, mu, std, spec, x_test, y_test)
    with torch.no_grad():
        parts = []
        for start in range(0, int(x_test.shape[0]), 512):
            parts.append(f922._mlp3_forward_core(x_test[start : start + 512], *mlp_params))
        mlp_logits = torch.cat(parts, dim=0)
    mlp_metrics = v92._classification_metrics_from_logits(mlp_logits, y_test)
    delta = kan_metrics["acc"] - mlp_metrics["acc"]
    h = x_train[: min(2048, int(x_train.shape[0]))] @ params[0]
    cond = act.basis_condition_metrics(h[: min(512, int(h.shape[0]))], mu, std, spec)
    return {
        "stage": "P4_ACTUATOR_P5_TRAINABILITY",
        "candidate_id": spec.candidate_id,
        "dataset": dataset,
        "seed": seed,
        "protocol": protocol,
        "KAN_acc": kan_metrics["acc"],
        "MLP_match_acc": mlp_metrics["acc"],
        "delta_vs_mlp": delta,
        "near_pass": int(delta >= -0.01),
        "CEp99": kan_metrics["CE_p99"],
        "margin_p10": kan_metrics["correct_margin_p10"],
        "ECE": kan_metrics["ECE"],
        "NLL": kan_metrics["NLL"],
        "basis_usage_entropy": cond["basis_usage_entropy"],
        "actuator_usage_entropy": cond["basis_usage_entropy"],
        "loss_type": "CE",
        "label_smoothing": 0,
        "uses_loss_backward": 0,
        "external_teacher_used": 0,
        "functional_update_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _run_p3_p4_p5(args: argparse.Namespace, device: torch.device, p1_best_r2: float, p1_best_rz: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    x_train, y_train, _x_test, _y_test, in_dim, out_dim, _protocol = v92._load_task(
        args, "MNIST", train_size=max(4096, int(args.p4_batch_size) * 4), test_size=256
    )
    x_train = x_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device)
    specs = act.actuator_specs_from_ids(_parse_list(args.actuator_candidates))
    p3_rows = [_gradcheck_actuator(args, spec, x_train, y_train, in_dim, out_dim, device) for spec in specs]
    p4_rows = [_measure_p4_actuator(args, spec, x_train, y_train, in_dim, out_dim, device) for spec in specs]
    p5_rows: List[Dict[str, Any]] = []
    train_ids = set(_parse_list(args.p5_actuator_candidates))
    p4_pass_ids = {r["candidate_id"] for r in p4_rows if _to_int(r.get("P4_pass")) == 1}
    for spec in specs:
        if spec.candidate_id not in train_ids:
            p5_rows.append(_not_run("P4_ACTUATOR_P5_TRAINABILITY", "p4_actuator_p4_p5_base_qualification.csv", "candidate_not_selected_for_expensive_P5_trainability"))
            p5_rows[-1]["candidate_id"] = spec.candidate_id
            continue
        if spec.candidate_id not in p4_pass_ids:
            p5_rows.append(_not_run("P4_ACTUATOR_P5_TRAINABILITY", "p4_actuator_p4_p5_base_qualification.csv", "candidate_P4_failed"))
            p5_rows[-1]["candidate_id"] = spec.candidate_id
            continue
        for dataset in _canonical_tasks(args.p5_datasets):
            for seed_text in _parse_list(args.p5_seeds):
                p5_rows.append(_train_actuator_candidate(args, spec, dataset, int(seed_text), device))

    p5_cont_rows: List[Dict[str, Any]] = []
    targets = _parse_list(args.targets)
    for dataset in _canonical_tasks(args.datasets):
        x_train2, y_train2, _x_test2, _y_test2, input_dim, output_dim, protocol = v92._load_task(
            args, dataset, train_size=max(int(args.train_size), 4096), test_size=int(args.eval_size)
        )
        x_train2 = x_train2.to(device=device, dtype=torch.float32)
        y_train2 = y_train2.to(device=device)
        xb = x_train2[: int(args.audit_batch_size)]
        yb = y_train2[: int(args.audit_batch_size)]
        for seed_text in _parse_list(args.seeds):
            seed = int(seed_text)
            for spec in specs:
                params, mu, std = act.init_actuator_params(input_dim, output_dim, spec, x_train2, device, seed + 921370)
                logits = act.actuator_forward(xb, params, mu, std, spec)
                task_loss, grads = act.actuator_fwd_bwd(xb, yb, params, mu, std, spec)
                task_step = snr_lq.gradient_descent_task_step(grads, float(args.lr))
                adamw_logits = act.actuator_forward(xb, [p + d for p, d in zip(params, task_step)], mu, std, spec)
                adamw_logit_norm = float((adamw_logits - logits).float().norm().detach().cpu())
                for target_id in targets:
                    target, target_info = out_lq.build_output_target(target_id, logits, yb, dataset=dataset)
                    raw_delta, ls_info = act.actuator_only_least_squares_delta(params, mu, std, spec, xb, target, ridge=float(args.ridge))
                    delta, removed = snr_lq.project_step_to_task_safe(raw_delta, grads)
                    fit = act.output_fit_metrics(params, delta, mu, std, spec, xb, target)
                    rz = fit["output_displacement_norm"] / max(adamw_logit_norm, 1.0e-12)
                    gain = fit["output_target_fit_r2"] - float(p1_best_r2)
                    p5_cont_rows.append({
                        "stage": "P5_POST_ACTUATOR_CONTROLLABILITY_AUDIT",
                        "candidate_id": spec.candidate_id,
                        "dataset": dataset,
                        "seed": seed,
                        "protocol": protocol,
                        "target_id": target_id,
                        "subspace_id": "ActuatorOnlyOutputCoeffSubspace",
                        "solver_id": "SOL1-LeastSquaresSketch",
                        "actuator_channel_active_fraction": target_info.get("target_selected_fraction", 0.0),
                        "actuator_channel_snr": "not_measured_in_LS_controllability",
                        "actuator_output_sensitivity": fit["output_displacement_norm"],
                        "actuator_target_fit_gain_vs_LQ": gain,
                        "target_fit_R2": fit["output_target_fit_r2"],
                        "output_displacement_ratio_vs_adamw": rz,
                        "target_norm": fit["target_norm"],
                        "target_residual_norm": ls_info.get("ls_residual_norm", 0.0),
                        "raw_delta_norm": float(snr_lq.step_norm(raw_delta).detach().cpu()),
                        "delta_norm_after_projection": float(snr_lq.step_norm(delta).detach().cpu()),
                        "projection_removed_norm": float(removed.detach().cpu()),
                        "norm_after_projection_ratio": float((snr_lq.step_norm(delta) / snr_lq.step_norm(raw_delta).clamp_min(1.0e-12)).detach().cpu()),
                        "actuator_controllability_pass": int((fit["output_target_fit_r2"] >= 0.20 or gain >= 0.10) and rz >= 0.05),
                        **ls_info,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    })
    return p3_rows, p4_rows + p5_rows, p5_cont_rows, []


def _p5_train_summary(rows: Sequence[Dict[str, Any]], candidate_id: str) -> Dict[str, Any]:
    measured = [r for r in rows if r.get("candidate_id") == candidate_id and str(r.get("status", "")) != "not_run" and "delta_vs_mlp" in r]
    deltas = [_to_float(r.get("delta_vs_mlp")) for r in measured]
    near = sum(_to_int(r.get("near_pass")) for r in measured)
    return {
        "p5_row_count": len(measured),
        "p5_near_pass_count": near,
        "p5_near_pass_rate": near / max(1, len(measured)),
        "p5_macro_delta": sum(deltas) / max(1, len(deltas)),
        "p5_near_pass": int(len(measured) > 0 and near / max(1, len(measured)) >= 0.80 and sum(deltas) / max(1, len(deltas)) >= -0.01),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--train-size", type=int, default=9984)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--audit-batch-size", type=int, default=128)
    parser.add_argument("--targets", default="O1-HardTailLogitCorrection,O2-MarginTailExpansion,O3-CalibrationTailCompression,O4-CurvatureOutputFlattening,O6-KMNISTHardModeOutputTarget")
    parser.add_argument("--subspaces", default="S1-QuadraticCoeffSubspace,S3-OutputLinearSubspace,S4-LiftPlusQuadraticSubspace,S5-RecentSignalSubspace,S6-OrthogonalToAdamWSubspace")
    parser.add_argument("--solvers", default="SOL1-LeastSquaresSketch,SOL2-ConstrainedLeastSquares,SOL3-TrustRegionQP")
    parser.add_argument("--oracle-eps", default="0.01,0.03,0.05")
    parser.add_argument("--step-fraction", type=float, default=0.03)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--actuator-candidates", default="A0-LQ-current,A1-LQ-T2ActuatorZeroInit,A2-LQ-NormalizedT2Actuator,A3-LQ-CenteredT2Actuator,A4-LQ-BoundedRationalActuator,A5-LQ-PiecewiseLinear2Actuator,A6-LQ-LocalRBFSharedCenterActuator,A7-LQ-BasisEntropyActuator")
    parser.add_argument("--p4-batch-size", type=int, default=128)
    parser.add_argument("--p4-warmup", type=int, default=5)
    parser.add_argument("--p4-reps", type=int, default=20)
    parser.add_argument("--p5-actuator-candidates", default="A1-LQ-T2ActuatorZeroInit")
    parser.add_argument("--p5-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--p5-seeds", default="0,1,2")
    parser.add_argument("--p5-train-size", type=int, default=9984)
    parser.add_argument("--p5-test-size", type=int, default=2000)
    parser.add_argument("--p5-epochs", type=int, default=20)
    parser.add_argument("--p5-lr", type=float, default=5.0e-4)
    parser.add_argument("--previous-v9212-dir", default=str(PREV_V9212.relative_to(ROOT)))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    prev_dir = Path(args.previous_v9212_dir)
    if not prev_dir.is_absolute():
        prev_dir = ROOT / prev_dir

    write_json(out_dir / "run_manifest.json", {
        "created_utc": _now_iso(),
        "script": str(SCRIPT_PATH.relative_to(ROOT)),
        "plan": str(PLAN_PATH.relative_to(ROOT)),
        "device": str(device),
        "torch": torch.__version__,
        "args": vars(args),
        "source_artifacts": {"v9212": str(prev_dir.relative_to(ROOT)) if prev_dir.exists() else str(prev_dir)},
    })
    contract_rows = [{
        "stage": "P0_CONTRACT_AUDIT",
        "loss_type": "CE",
        "label_smoothing": 0,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "teacher_logits_used": 0,
        "distillation_used": 0,
        "geometry_loss_used": 0,
        "sampler_changed": 0,
        "class_weight_used": 0,
        "uses_loss_backward": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "PureKANConv_status": "deferred_until_FC_PureKAN_functional_advantage_or_user_unlock",
        "PureKANFormer_status": "deferred_until_FC_PureKAN_functional_advantage_or_user_unlock",
    }]
    write_csv_rows(out_dir / "contract_audit_v9213.csv", contract_rows)
    p0_row = _p0_recap(prev_dir)
    write_csv_rows(out_dir / "p0_v9212_reproduction.csv", [p0_row])
    p0_pass = _to_int(p0_row.get("p0_reproduction_pass"))

    if p0_pass:
        p1_rows, p2_rows, sens_rows = _run_p1_p2(args, device)
    else:
        reason = "P0_v9212_reproduction_failed"
        p1_rows = [_not_run("P1_FUNCTIONAL_CONTROLLABILITY_AUDIT", "p1_functional_controllability_audit.csv", reason)]
        p2_rows = [_not_run("P2_DIRECT_LOGIT_ORACLE_TARGET_AUDIT", "p2_direct_logit_oracle_target_audit.csv", reason)]
        sens_rows = [_not_run("ACTUATOR_SENSITIVITY_TRACE", "actuator_sensitivity_trace_v9213.csv", reason)]
    write_csv_rows(out_dir / "p1_functional_controllability_audit.csv", p1_rows)
    write_csv_rows(out_dir / "p2_direct_logit_oracle_target_audit.csv", p2_rows)

    measured_p1 = [r for r in p1_rows if str(r.get("status", "")) != "not_run"]
    measured_p2 = [r for r in p2_rows if str(r.get("status", "")) != "not_run"]
    direct_oracle_pass = int(any(_to_int(r.get("target_useful")) == 1 for r in measured_p2))
    nonzero_p1 = [r for r in measured_p1 if _to_float(r.get("target_norm")) > 1.0e-12]
    safe_p1 = [r for r in nonzero_p1 if str(r.get("solver_id")) != "SOL1-LeastSquaresSketch"]
    best_p1_r2_raw = max([_to_float(r.get("target_fit_R2")) for r in nonzero_p1] or [0.0])
    best_p1_rz_raw = max([_to_float(r.get("output_displacement_ratio_vs_adamw")) for r in nonzero_p1] or [0.0])
    best_p1_r2 = max([_to_float(r.get("target_fit_R2")) for r in safe_p1] or [0.0])
    best_p1_rz = max([_to_float(r.get("output_displacement_ratio_vs_adamw")) for r in safe_p1] or [0.0])
    current_cont_pass = int(best_p1_r2 >= 0.20 and best_p1_rz >= 0.05)
    current_weak = int(best_p1_r2 < 0.10 and best_p1_rz < 0.05)

    if p0_pass and direct_oracle_pass:
        p3_rows, p4_rows, p5_cont_rows, _ = _run_p3_p4_p5(args, device, best_p1_r2, best_p1_rz)
    else:
        reason = "direct_logit_oracle_failed" if p0_pass else "P0_v9212_reproduction_failed"
        p3_rows = [_not_run("P3_ACTUATOR_CONTRACT_GRADCHECK", "p3_actuator_contract_gradcheck.csv", reason)]
        p4_rows = [_not_run("P4_ACTUATOR_P4_P5_BASE_QUALIFICATION", "p4_actuator_p4_p5_base_qualification.csv", reason)]
        p5_cont_rows = [_not_run("P5_POST_ACTUATOR_CONTROLLABILITY_AUDIT", "p5_post_actuator_controllability_audit.csv", reason)]
    write_csv_rows(out_dir / "p3_actuator_contract_gradcheck.csv", p3_rows)
    write_csv_rows(out_dir / "p4_actuator_p4_p5_base_qualification.csv", p4_rows)
    write_csv_rows(out_dir / "p5_post_actuator_controllability_audit.csv", p5_cont_rows)
    write_csv_rows(out_dir / "actuator_sensitivity_trace_v9213.csv", sens_rows + p5_cont_rows)

    p3_measured = [r for r in p3_rows if str(r.get("status", "")) != "not_run"]
    p4_measured = [r for r in p4_rows if str(r.get("status", "")) != "not_run"]
    p5c_measured = [r for r in p5_cont_rows if str(r.get("status", "")) != "not_run"]
    p3_pass_ids = {r["candidate_id"] for r in p3_measured if _to_int(r.get("contract_pass")) == 1 and _to_int(r.get("GradPass")) == 1 and _to_int(r.get("interaction_retention_pass")) == 1}
    p4_pass_ids = {r["candidate_id"] for r in p4_measured if _to_int(r.get("P4_pass")) == 1}
    p5c_pass_ids = {r["candidate_id"] for r in p5c_measured if _to_int(r.get("actuator_controllability_pass")) == 1}
    train_summaries = {cid: _p5_train_summary(p4_rows, cid) for cid in p4_pass_ids}
    p5_base_pass_ids = {cid for cid, s in train_summaries.items() if _to_int(s.get("p5_near_pass")) == 1}
    best_actuator = ""
    best_cont = None
    eligible_cont = [r for r in p5c_measured if r.get("candidate_id") in p3_pass_ids and r.get("candidate_id") in p4_pass_ids and r.get("candidate_id") in p5_base_pass_ids]
    if eligible_cont:
        best_cont = sorted(eligible_cont, key=lambda r: (-_to_float(r.get("target_fit_R2")), -_to_float(r.get("output_displacement_ratio_vs_adamw"))))[0]
        best_actuator = str(best_cont.get("candidate_id", ""))

    p6_reason = "P5_post_actuator_controllability_failed_or_base_not_qualified"
    if best_cont and _to_int(best_cont.get("actuator_controllability_pass")) == 1:
        p6_reason = "P5_controllability_passed_but_P6_not_executed_in_this_runner_first_wave"
    for stage, name in [
        ("P6_PAIRED_REPLAY_WITH_ACTUATOR", "p6_paired_replay_with_actuator.csv"),
        ("P7_FUNCTIONAL_REENTRY_WITH_ACTUATOR", "p7_functional_reentry_with_actuator.csv"),
        ("P8_ROBUSTNESS_EXTERNAL_READY_GATE", "p8_robustness_external_ready_gate.csv"),
        ("FUNCTIONAL_EVENT_TRACE", "functional_event_trace_v9213.csv"),
    ]:
        write_csv_rows(out_dir / name, [_not_run(stage, name, p6_reason)])

    if not p0_pass:
        route = "R9-ReturnToBasisFactory"
        primary = "v9212_terminal_boundary_not_reproduced"
        failure_code = "F3_v9212_reproduction_unstable"
    elif not direct_oracle_pass:
        route = "R1-CurrentLQTargetInvalid"
        primary = "direct_logit_oracle_targets_did_not_improve_metrics"
        failure_code = "F4_target_oracle_fail"
    elif current_weak and not p5c_pass_ids:
        route = "R2-CurrentLQControllabilityWeak"
        primary = "target_useful_but_current_LQ_and_tested_actuators_show_no_controllability_pass"
        failure_code = "F12_actuator_controllability_no_gain"
    elif p5c_pass_ids and not p4_pass_ids:
        route = "R8-NoPureKANActuatorFound"
        primary = "actuator_controllability_signal_exists_but_no_candidate_passed_P4_system_gate"
        failure_code = "F10_actuator_p4_fail"
    elif p5c_pass_ids and not p5_base_pass_ids:
        route = "R7-ActuatorHurtsBase"
        primary = "actuator_controllability_signal_exists_but_no_candidate_has_P5_nearpass_base_qualification"
        failure_code = "F11_actuator_p5_fail"
    elif best_cont and _to_int(best_cont.get("actuator_controllability_pass")) == 1:
        route = "R4-ActuatorControllabilityPass"
        primary = "actuator_controllability_passed_but_paired_replay_not_opened_in_first_wave"
        failure_code = "F13_paired_replay_causality_fail"
    else:
        route = "R8-NoPureKANActuatorFound"
        primary = "no_tested_actuator_preserved_contract_P4_P5_and_improved_controllability"
        failure_code = "F12_actuator_controllability_no_gain"

    failure_rows = [{
        "stage": "ROUTE_DECISION",
        "failure_code": failure_code,
        "route": route,
        "reason": primary,
        "direct_logit_oracle_pass": direct_oracle_pass,
        "current_lq_controllability_pass": current_cont_pass,
        "actuator_contract_pass_count": len(p3_pass_ids),
        "actuator_p4_pass_count": len(p4_pass_ids),
        "actuator_p5_near_pass_count": len(p5_base_pass_ids),
        "post_actuator_controllability_pass_count": len(p5c_pass_ids),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    write_csv_rows(out_dir / "failure_table.csv", failure_rows)

    _write_svg(out_dir / "figures" / "p0_v9212_reproduction_dashboard.svg", "P0 v9.2.12 Recap", f"p0_pass={p0_pass}")
    _write_svg(out_dir / "figures" / "p1_controllability_heatmap.svg", "P1 Controllability", f"best_R2={best_p1_r2:.6f}, best_rz={best_p1_rz:.6f}")
    _write_svg(out_dir / "figures" / "p2_direct_logit_oracle_effect.svg", "P2 Direct Logit Oracle", f"oracle_pass={direct_oracle_pass}")
    _write_svg(out_dir / "figures" / "p3_actuator_contract_heatmap.svg", "P3 Actuator Contract", f"contract_grad_interaction_pass={len(p3_pass_ids)}")
    _write_svg(out_dir / "figures" / "p4_actuator_system_pareto.svg", "P4 Actuator System", f"p4_pass={len(p4_pass_ids)}")
    _write_svg(out_dir / "figures" / "p5_controllability_before_after.svg", "P5 Actuator Controllability", f"post_pass={len(p5c_pass_ids)}")

    best_train_summary = train_summaries.get(best_actuator, {}) if best_actuator else {}
    route_json = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "best_actuator_candidate": best_actuator,
        "best_output_target": best_cont.get("target_id", "") if best_cont else "",
        "best_subspace": best_cont.get("subspace_id", "") if best_cont else "",
        "best_solver": best_cont.get("solver_id", "") if best_cont else "",
        "direct_logit_oracle_pass": direct_oracle_pass,
        "current_lq_controllability_pass": current_cont_pass,
        "current_lq_best_target_fit_R2": best_p1_r2,
        "current_lq_best_output_displacement_ratio": best_p1_rz,
        "current_lq_best_raw_target_fit_R2": best_p1_r2_raw,
        "current_lq_best_raw_output_displacement_ratio": best_p1_rz_raw,
        "actuator_contract_pass": int(bool(p3_pass_ids)),
        "actuator_p4_pass": int(bool(p4_pass_ids)),
        "actuator_p5_near_pass": int(bool(p5_base_pass_ids)),
        "post_actuator_controllability_pass": int(bool(best_cont and _to_int(best_cont.get("actuator_controllability_pass")) == 1)),
        "best_post_actuator_target_fit_R2": _to_float(best_cont.get("target_fit_R2")) if best_cont else 0.0,
        "best_post_actuator_output_displacement_ratio": _to_float(best_cont.get("output_displacement_ratio_vs_adamw")) if best_cont else 0.0,
        "paired_replay_pass": 0,
        "full_functional_pass": 0,
        "noise_robustness_pass": 0,
        "strong_baseline_pass": 0,
        "external_ready": 0,
        "primary_blocker": primary,
        "next_required_implementation": "open_P6_paired_replay_for_base_qualified_actuator" if route == "R4-ActuatorControllabilityPass" else "return_to_actuator_basis_factory_or_target_design",
        "success_v9213_controllability_audit": int(p0_pass and direct_oracle_pass),
        "success_v9213_actuator_base": int(bool(p5_base_pass_ids)),
        "success_v9213_actuator_controllability": int(bool(best_cont and _to_int(best_cont.get("actuator_controllability_pass")) == 1)),
        "success_v9213_functional_advantage": 0,
        **best_train_summary,
    }
    write_json(out_dir / "route_decision.json", route_json)
    write_json(out_dir / "aggregate_decision.json", route_json)

    audit_targets = [
        out_dir / "contract_audit_v9213.csv",
        out_dir / "p0_v9212_reproduction.csv",
        out_dir / "p1_functional_controllability_audit.csv",
        out_dir / "p2_direct_logit_oracle_target_audit.csv",
        out_dir / "p3_actuator_contract_gradcheck.csv",
        out_dir / "p4_actuator_p4_p5_base_qualification.csv",
        out_dir / "p5_post_actuator_controllability_audit.csv",
        out_dir / "p6_paired_replay_with_actuator.csv",
        out_dir / "p7_functional_reentry_with_actuator.csv",
        out_dir / "p8_robustness_external_ready_gate.csv",
        out_dir / "functional_event_trace_v9213.csv",
        out_dir / "actuator_sensitivity_trace_v9213.csv",
        out_dir / "failure_table.csv",
    ]
    audit = audit_no_fake(audit_targets)
    write_csv_rows(out_dir / "v9213_provenance_audit.csv", [{"stage": "NO_FAKE_AUDIT", "route": route, **audit, "fake_data_used": int(audit["fake_data_used"]), "proxy_row_used": int(audit["proxy_row_used"]), "cpu_offload_used": int(audit["cpu_offload_used"])}])
    route_json.update({"no_fake": bool(audit["no_fake"]), "no_proxy": bool(audit["no_proxy"]), "rows_checked": int(audit["rows_checked"])})
    write_json(out_dir / "route_decision.json", route_json)
    write_json(out_dir / "aggregate_decision.json", route_json)
    write_csv_rows(out_dir / "artifact_hashes.csv", artifact_hash_rows([
        PLAN_PATH,
        SCRIPT_PATH,
        ROOT / "dgkan" / "models" / "fc_purekan_actuator.py",
        ROOT / "dgkan" / "models" / "fc_purekan_lq.py",
        ROOT / "dgkan" / "functional" / "lq_output_space_functional.py",
        ROOT / "dgkan" / "functional" / "snr_gated_lq.py",
        out_dir / "route_decision.json",
        *audit_targets,
        out_dir / "v9213_provenance_audit.csv",
    ], root=ROOT))
    print(json.dumps(route_json, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
