#!/usr/bin/env python3
"""DG-KAN v9.2.14 P4-qualified functional actuator closure runner."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v92_basis_source_kernel_gate as v92  # noqa: E402
import run_v922_fused_compositional_kernel_closure as f922  # noqa: E402
import run_v9213_functional_controllability_actuator_redesign as v9213  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402
from dgkan.functional import lq_output_space_functional as out_lq  # noqa: E402
from dgkan.functional import snr_gated_lq as snr_lq  # noqa: E402
from dgkan.models import fc_purekan_actuator as act  # noqa: E402
from dgkan.models import fc_purekan_lq as lq  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402
from dgkan.training.manual_full_edge import ce_loss_and_grad  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.14_P4Qualified_Functional_Actuator_Closure_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9214_p4qualified_functional_actuator_closure.py"
PREV_V9213 = ROOT / "results" / "real_rerun_20260506" / "v9213_functional_controllability_actuator_redesign_first_20260510T000000Z"


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


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def _classification_delta(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    return {
        "accuracy_delta": after["acc"] - before["acc"],
        "holdout_delta": after["loss"] - before["loss"],
        "CEp99_delta": after["CE_p99"] - before["CE_p99"],
        "margin_p10_delta": after["correct_margin_p10"] - before["correct_margin_p10"],
        "ECE_delta": after["ECE"] - before["ECE"],
        "NLL_delta": after["NLL"] - before["NLL"],
    }


def _p0_recap(prev_dir: Path) -> Dict[str, Any]:
    route = _read_json(prev_dir / "route_decision.json")
    p5 = read_csv_rows(prev_dir / "p5_post_actuator_controllability_audit.csv")
    p4 = read_csv_rows(prev_dir / "p4_actuator_p4_p5_base_qualification.csv")
    audit = read_csv_rows(prev_dir / "v9213_provenance_audit.csv")
    rows_checked = _to_int(audit[0].get("rows_checked")) if audit else 0
    fake_count = _to_int(audit[0].get("fake_proxy_nonzero_count")) if audit else 0
    pass_count = sum(_to_int(r.get("actuator_controllability_pass")) for r in p5)
    p4_pass_count = sum(_to_int(r.get("P4_pass")) for r in p4)
    return {
        "stage": "P0_V9213_REPRODUCTION",
        "source_artifact": str(prev_dir.relative_to(ROOT)) if prev_dir.exists() else str(prev_dir),
        "source_route": route.get("route", "missing"),
        "direct_logit_oracle_pass": route.get("direct_logit_oracle_pass", 0),
        "current_lq_raw_max_R2": route.get("current_lq_best_raw_target_fit_R2", 0),
        "current_lq_raw_max_rz": route.get("current_lq_best_raw_output_displacement_ratio", 0),
        "current_lq_safe_max_R2": route.get("current_lq_best_target_fit_R2", 0),
        "current_lq_safe_max_rz": route.get("current_lq_best_output_displacement_ratio", 0),
        "actuator_contract_pass": route.get("actuator_contract_pass", 0),
        "actuator_p4_pass_count": p4_pass_count,
        "post_actuator_controllability_pass_count": pass_count,
        "rows_checked_source": rows_checked,
        "fake_proxy_count": fake_count,
        "P0_reproduction_pass": int(
            route.get("route") == "R8-NoPureKANActuatorFound"
            and _to_int(route.get("direct_logit_oracle_pass")) == 1
            and p4_pass_count == 0
            and pass_count > 0
            and fake_count == 0
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _init_mlp_match(params_kan: int, in_dim: int, out_dim: int, seed: int, device: torch.device) -> Tuple[List[torch.Tensor], int]:
    hidden_mlp = f922._matched_mlp3_hidden(params_kan, in_dim, out_dim)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    return [
        torch.randn(in_dim, hidden_mlp, device=device, generator=gen) / math.sqrt(in_dim),
        torch.randn(hidden_mlp, hidden_mlp, device=device, generator=gen) / math.sqrt(hidden_mlp),
        torch.randn(hidden_mlp, out_dim, device=device, generator=gen) / math.sqrt(hidden_mlp),
    ], hidden_mlp


def _phase_attribution(args: argparse.Namespace, spec: act.ActuatorSpec, x: torch.Tensor, y: torch.Tensor, in_dim: int, out_dim: int, device: torch.device) -> Dict[str, Any]:
    xb = x[: int(args.p4_batch_size)]
    yb = y[: int(args.p4_batch_size)]
    params, mu, std = act.init_actuator_params(in_dim, out_dim, spec, x, device, int(args.seed) + 1400 + len(spec.candidate_id))
    mlp_params, hidden_mlp = _init_mlp_match(sum(p.numel() for p in params), in_dim, out_dim, int(args.seed) + 1401, device)

    with torch.no_grad():
        h0 = xb @ params[0]
        vals0, ders0, _names = act.actuator_basis_from_lift(h0, mu, std, spec, 2.0, 2.0)
        logits0 = vals0[0] @ params[1]
        for val, weight in zip(vals0[1:], params[2:]):
            logits0 = logits0 + val @ weight
        _loss0, dy0 = ce_loss_and_grad(logits0, yb)

    def basis_eval() -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        h = xb @ params[0]
        vals, ders, _ = act.actuator_basis_from_lift(h, mu, std, spec, 2.0, 2.0)
        return vals, ders

    def projection() -> torch.Tensor:
        yhat = vals0[0] @ params[1]
        for val, weight in zip(vals0[1:], params[2:]):
            yhat = yhat + val @ weight
        return yhat

    def output_grad() -> torch.Tensor:
        logits = projection()
        loss, _dy = ce_loss_and_grad(logits, yb)
        return loss

    def coeffgrad() -> List[torch.Tensor]:
        return [val.T @ dy0 for val in vals0]

    def basis_derivative() -> torch.Tensor:
        dh = torch.zeros_like(h0)
        for der, weight in zip(ders0, params[1:]):
            dh = dh + (dy0 @ weight.T) * der
        return xb.T @ dh

    def full_forward() -> torch.Tensor:
        return act.actuator_forward(xb, params, mu, std, spec)

    def full_bwd() -> Tuple[torch.Tensor, List[torch.Tensor]]:
        return act.actuator_fwd_bwd(xb, yb, params, mu, std, spec)

    def mlp_forward() -> torch.Tensor:
        return f922._mlp3_forward_core(xb, *mlp_params)

    def mlp_bwd() -> Tuple[torch.Tensor, ...]:
        return f922._mlp3_fwd_bwd_core(xb, yb, *mlp_params)

    reps = int(args.p1_phase_reps)
    warm = int(args.p4_warmup)
    for _ in range(warm):
        basis_eval(); projection(); output_grad(); coeffgrad(); basis_derivative(); full_forward(); full_bwd(); mlp_forward(); mlp_bwd()
    _sync(device)
    basis_ms = v92._bench_callable_ms(basis_eval, reps, device)
    proj_ms = v92._bench_callable_ms(projection, reps, device)
    full_fwd_ms = v92._bench_callable_ms(full_forward, reps, device)
    output_ms = v92._bench_callable_ms(output_grad, reps, device)
    coeff_ms = v92._bench_callable_ms(coeffgrad, reps, device)
    deriv_ms = v92._bench_callable_ms(basis_derivative, reps, device)
    full_fb_ms = v92._bench_callable_ms(full_bwd, reps, device)
    mlp_fwd_ms = v92._bench_callable_ms(mlp_forward, reps, device)
    mlp_fb_ms = v92._bench_callable_ms(mlp_bwd, reps, device)
    mlp_backward_ms = max(0.0, mlp_fb_ms - mlp_fwd_ms)
    backward_ms = max(0.0, full_fb_ms - full_fwd_ms)
    update_ms = 0.0
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=0.0)
    states = [AdamWState.zeros_like(p) for p in params]

    def update_only() -> None:
        _loss, grads = full_bwd()
        v92._adamw_update_foreach_(params, grads, states, cfg)

    update_ms = max(0.0, v92._bench_callable_ms(update_only, max(1, reps // 2), device) - full_fb_ms)
    component_sum = output_ms + coeff_ms + deriv_ms
    unknown = abs(backward_ms - component_sum) / max(backward_ms, 1.0e-12)
    excess = max(0.0, backward_ms - mlp_backward_ms)
    bcomp = deriv_ms + coeff_ms
    return {
        "stage": "P1_ACTUATOR_P4_FAILURE_ATTRIBUTION",
        "candidate_id": spec.candidate_id,
        "actuator_type": spec.actuator_type,
        "forward_basis_eval_ms": basis_ms,
        "forward_projection_ms": proj_ms,
        "forward_total_ms": full_fwd_ms,
        "backward_basis_derivative_ms": deriv_ms,
        "backward_coeffgrad_ms": coeff_ms,
        "backward_output_grad_ms": output_ms,
        "backward_total_ms": backward_ms,
        "mlp_backward_total_ms": mlp_backward_ms,
        "backward_excess_ms": excess,
        "backward_derivative_coeffgrad_excess_share": bcomp / max(excess, 1.0e-12),
        "optimizer_update_ms": update_ms,
        "step_total_ms": full_fb_ms + update_ms,
        "compact_memory_ratio": "",
        "temp_MB": "",
        "kernel_count": "component_timer_not_torch_profiler",
        "small_kernel_count": "not_measured_component_timer",
        "unknown_time_fraction": min(1.0, unknown),
        "attribution_pass": int(unknown <= 0.10),
        "backward_dominant": int(excess > 0 and bcomp / max(excess, 1.0e-12) >= 0.60),
        "matched_mlp_hidden": hidden_mlp,
        "loss_type": "CE",
        "label_smoothing": 0,
        "uses_loss_backward": 0,
        "external_teacher_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _measure_p4_q(args: argparse.Namespace, spec: act.ActuatorSpec, x: torch.Tensor, y: torch.Tensor, in_dim: int, out_dim: int, device: torch.device) -> Dict[str, Any]:
    xb = x[: int(args.p4_batch_size)]
    yb = y[: int(args.p4_batch_size)]
    params, mu, std = act.init_actuator_params(in_dim, out_dim, spec, x, device, int(args.seed) + 1431 + len(spec.candidate_id))
    params_kan = sum(p.numel() for p in params)
    mlp_params, hidden_mlp = _init_mlp_match(params_kan, in_dim, out_dim, int(args.seed) + 1432, device)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=0.0)
    kan_states = [AdamWState.zeros_like(p) for p in params]
    mlp_states = [AdamWState.zeros_like(p) for p in mlp_params]

    def act_fwd_core(x_in: torch.Tensor, *flat: torch.Tensor) -> torch.Tensor:
        return act.actuator_forward(x_in, list(flat), mu, std, spec)

    def act_bwd_core(x_in: torch.Tensor, y_in: torch.Tensor, *flat: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        loss, grads = act.actuator_fwd_bwd(x_in, y_in, list(flat), mu, std, spec)
        return (loss, *grads)

    cfwd = v92._maybe_compile(f"v9214_{spec.candidate_id}_forward", act_fwd_core)
    cbwd = v92._maybe_compile(f"v9214_{spec.candidate_id}_bwd", act_bwd_core)
    mfwd = v92._maybe_compile(f"v9214_{spec.candidate_id}_mlp_forward", f922._mlp3_forward_core)
    mbwd = v92._maybe_compile(f"v9214_{spec.candidate_id}_mlp_bwd", f922._mlp3_fwd_bwd_core)

    def kan_fwd() -> torch.Tensor:
        return cfwd(xb, *params)

    def kan_bwd() -> Tuple[torch.Tensor, ...]:
        return cbwd(xb, yb, *params)

    def kan_step() -> None:
        pack = kan_bwd()
        v92._adamw_update_foreach_(params, pack[1:], kan_states, cfg)

    def mlp_fwd() -> torch.Tensor:
        return mfwd(xb, *mlp_params)

    def mlp_bwd() -> Tuple[torch.Tensor, ...]:
        return mbwd(xb, yb, *mlp_params)

    def mlp_step() -> None:
        pack = mlp_bwd()
        v92._adamw_update_foreach_(mlp_params, pack[1:], mlp_states, cfg)

    for _ in range(int(args.p4_warmup)):
        kan_fwd(); kan_bwd(); mlp_fwd(); mlp_bwd()
    forward_ratios: List[float] = []
    backward_ratios: List[float] = []
    step_ratios: List[float] = []
    for _ in range(int(args.p4_repeat_measurements)):
        f_kan = v92._bench_callable_ms(kan_fwd, int(args.p4_reps), device)
        fb_kan = v92._bench_callable_ms(kan_bwd, int(args.p4_reps), device)
        s_kan = v92._bench_callable_ms(kan_step, int(args.p4_reps), device)
        f_mlp = v92._bench_callable_ms(mlp_fwd, int(args.p4_reps), device)
        fb_mlp = v92._bench_callable_ms(mlp_bwd, int(args.p4_reps), device)
        s_mlp = v92._bench_callable_ms(mlp_step, int(args.p4_reps), device)
        forward_ratios.append(f_kan / max(f_mlp, 1.0e-12))
        backward_ratios.append(max(0.0, fb_kan - f_kan) / max(max(0.0, fb_mlp - f_mlp), 1.0e-12))
        step_ratios.append(s_kan / max(s_mlp, 1.0e-12))

    channel_count = len(params) - 1
    cache_conservative = int(args.p4_batch_size) * (in_dim + spec.hidden_dim * (2 + channel_count) + out_dim)
    cache_compact = int(args.p4_batch_size) * (spec.hidden_dim + out_dim)
    peak_kan_conservative = (params_kan * 3 + cache_conservative) * 4 / (1024.0 * 1024.0)
    peak_kan_compact = (params_kan * 3 + cache_compact) * 4 / (1024.0 * 1024.0)
    peak_mlp = f922._estimate_mlp3_memory_mb(in_dim, hidden_mlp, out_dim, int(args.p4_batch_size))
    compact = peak_kan_compact / max(peak_mlp, 1.0e-12)
    conservative = peak_kan_conservative / max(peak_mlp, 1.0e-12)
    fq90 = _quantile(forward_ratios, 0.90)
    bq90 = _quantile(backward_ratios, 0.90)
    sq90 = _quantile(step_ratios, 0.90)
    return {
        "forward_ratio_q50": _quantile(forward_ratios, 0.50),
        "forward_ratio_q90": fq90,
        "backward_ratio_q50": _quantile(backward_ratios, 0.50),
        "backward_ratio_q90": bq90,
        "step_ratio_q50": _quantile(step_ratios, 0.50),
        "step_ratio_q90": sq90,
        "compact_memory_ratio": compact,
        "conservative_memory_ratio": conservative,
        "P4_pass": int(fq90 <= 1.25 and bq90 <= 1.50 and sq90 <= 1.50 and compact <= 1.05),
        "params_kan": params_kan,
        "params_mlp_match": in_dim * hidden_mlp + hidden_mlp * hidden_mlp + hidden_mlp * out_dim,
        "matched_mlp_hidden": hidden_mlp,
    }


def _candidate_controllability_best(args: argparse.Namespace, spec: act.ActuatorSpec, device: torch.device, lq_safe_r2: float) -> Dict[str, Any]:
    best: Dict[str, Any] = {}
    for dataset in _canonical_tasks(args.datasets):
        x_train, y_train, _x_test, _y_test, in_dim, out_dim, protocol = v92._load_task(
            args, dataset, train_size=max(int(args.train_size), 4096), test_size=int(args.eval_size)
        )
        x_train = x_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device)
        xb = x_train[: int(args.audit_batch_size)]
        yb = y_train[: int(args.audit_batch_size)]
        for seed_text in _parse_list(args.seeds):
            seed = int(seed_text)
            params, mu, std = act.init_actuator_params(in_dim, out_dim, spec, x_train, device, seed + 921430)
            logits = act.actuator_forward(xb, params, mu, std, spec)
            _loss, grads = act.actuator_fwd_bwd(xb, yb, params, mu, std, spec)
            task_step = snr_lq.gradient_descent_task_step(grads, float(args.lr))
            adamw_logits = act.actuator_forward(xb, [p + d for p, d in zip(params, task_step)], mu, std, spec)
            adamw_logit_norm = float((adamw_logits - logits).float().norm().detach().cpu())
            for target_id in _parse_list(args.targets):
                target, info = out_lq.build_output_target(target_id, logits, yb, dataset=dataset)
                raw_delta, ls_info = act.actuator_only_least_squares_delta(params, mu, std, spec, xb, target, ridge=float(args.ridge))
                delta, removed = snr_lq.project_step_to_task_safe(raw_delta, grads)
                fit = act.output_fit_metrics(params, delta, mu, std, spec, xb, target)
                rz = fit["output_displacement_norm"] / max(adamw_logit_norm, 1.0e-12)
                row = {
                    "best_dataset": dataset,
                    "best_seed": seed,
                    "best_protocol": protocol,
                    "best_target": target_id,
                    "target_fit_R2": fit["output_target_fit_r2"],
                    "output_displacement_ratio_rz": rz,
                    "target_selected_fraction": info.get("target_selected_fraction", 0.0),
                    "target_residual_norm": ls_info.get("ls_residual_norm", 0.0),
                    "projection_removed_norm": float(removed.detach().cpu()),
                    "controllability_retained": int((fit["output_target_fit_r2"] >= 0.20 or fit["output_target_fit_r2"] >= lq_safe_r2 + 0.10) and rz >= 0.05),
                }
                if not best or (row["controllability_retained"], row["target_fit_R2"], row["output_displacement_ratio_rz"]) > (
                    best.get("controllability_retained", 0),
                    _to_float(best.get("target_fit_R2")),
                    _to_float(best.get("output_displacement_ratio_rz")),
                ):
                    best = row
    return best


def _run_p2(args: argparse.Namespace, device: torch.device, lq_safe_r2: float) -> List[Dict[str, Any]]:
    x_train, y_train, _x_test, _y_test, in_dim, out_dim, _protocol = v92._load_task(
        args, "MNIST", train_size=max(4096, int(args.p4_batch_size) * 4), test_size=256
    )
    x_train = x_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device)
    rows: List[Dict[str, Any]] = []
    for cid in _parse_list(args.p2_candidates):
        if cid == "A7b-BasisEntropy-LowRank":
            rows.append({
                "stage": "P2_P4_CLOSURE_CANDIDATE_FACTORY",
                "candidate_id": cid,
                "status": "not_implemented",
                "reason": "lowrank_output_edge_actuator_not_implemented",
                "P4_pass": 0,
                "controllability_retained": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            continue
        spec = act.actuator_specs_from_ids([cid])[0]
        grad = v9213._gradcheck_actuator(args, spec, x_train, y_train, in_dim, out_dim, device)
        p4 = _measure_p4_q(args, spec, x_train, y_train, in_dim, out_dim, device)
        cont = _candidate_controllability_best(args, spec, device, lq_safe_r2)
        _params_tmp, mu_tmp, std_tmp = act.init_actuator_params(in_dim, out_dim, spec, x_train, device, int(args.seed) + 921444)
        h = x_train[: min(512, int(x_train.shape[0]))] @ _params_tmp[0]
        cond = act.basis_condition_metrics(h, mu_tmp, std_tmp, spec)
        grad_pass = int(_to_float(grad.get("GradRelErrMax")) <= 1.0e-4 and _to_float(grad.get("GradCosMin")) >= 0.999)
        rows.append({
            "stage": "P2_P4_CLOSURE_CANDIDATE_FACTORY",
            "candidate_id": spec.candidate_id,
            "actuator_type": spec.actuator_type,
            "basis_formula": act.basis_formula(spec),
            "fixed_shape_params": int("value_only" in spec.actuator_type or "Fixed" in spec.candidate_id),
            "trainable_actuator_params": act.actuator_channel_count(spec) * spec.hidden_dim * out_dim,
            "manual_forward": 1,
            "manual_backward": 1,
            "GradRelErrMax": grad.get("GradRelErrMax", ""),
            "GradCosMin": grad.get("GradCosMin", ""),
            "GradPass": grad_pass,
            "synthetic_pairwise_R2": v9213._synthetic_pairwise_r2(spec, device, int(args.seed) + 9214),
            **p4,
            **cont,
            "actuator_usage_entropy": cond["basis_usage_entropy"],
            "basis_condition_number": cond["basis_condition_number"],
            "p4_controllable_pass": int(_to_int(p4.get("P4_pass")) == 1 and grad_pass == 1 and _to_int(cont.get("controllability_retained")) == 1),
            "loss_type": "CE",
            "label_smoothing": 0,
            "uses_loss_backward": 0,
            "external_teacher_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    return rows


def _lq_least_squares_delta(
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    basis: str,
    x: torch.Tensor,
    target_delta_logits: torch.Tensor,
    absorb_subspace: str,
    ridge: float,
) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    delta = [torch.zeros_like(p) for p in params]
    h = x @ params[0]
    vals, _ders = lq.basis_from_lift(h, mu, std, basis, 2.0, 2.0)
    sub = str(absorb_subspace)
    if sub == "AB0-ActuatorSolveThenAbsorb-T2":
        blocks = [(2, vals[1])]
    elif sub == "AB3-ActuatorShadowLowRank":
        blocks = [(1, vals[0])]
    elif sub in {"AB2-ActuatorSolveThenAbsorb-LiftPlusT2", "AB4-EventOnlyActuatorNoPersistentForward"}:
        blocks = [(1, vals[0]), (2, vals[1])]
    elif sub == "AB1-ActuatorSolveThenAbsorb-Lift":
        blocks = [(1, vals[0])]
    else:
        raise ValueError(f"unknown absorb subspace {absorb_subspace}")
    phi = torch.cat([b[1] for b in blocks], dim=1).float()
    target = target_delta_logits.float()
    eye = torch.eye(phi.shape[1], device=phi.device, dtype=phi.dtype)
    coef = torch.linalg.solve(phi.T @ phi + float(ridge) * eye, phi.T @ target).to(params[0].dtype)
    start = 0
    for param_idx, feat in blocks:
        width = int(feat.shape[1])
        delta[param_idx] = coef[start : start + width]
        start += width
    actual = phi @ coef.float()
    residual = actual - target
    return delta, {
        "absorb_ls_rank": int(torch.linalg.matrix_rank(phi).detach().cpu()),
        "absorb_ls_residual_norm": float(residual.norm().detach().cpu()),
    }


def _lq_output_fit(
    params: Sequence[torch.Tensor],
    delta: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    x: torch.Tensor,
    target_delta_logits: torch.Tensor,
) -> Dict[str, float]:
    with torch.no_grad():
        before = lq.lift_basis_forward_t2(x, params[0], params[1], params[2], mu, std, 2.0, 2.0)
        after_params = [p.detach() + d.detach() for p, d in zip(params, delta)]
        after = lq.lift_basis_forward_t2(x, after_params[0], after_params[1], after_params[2], mu, std, 2.0, 2.0)
        actual = after - before
        target = target_delta_logits.detach()
        sse = (actual.float() - target.float()).square().sum()
        sst = (target.float() - target.float().mean()).square().sum().clamp_min(1.0e-12)
        r2 = 1.0 - sse / sst
    return {
        "fit_R2": float(r2.detach().cpu()),
        "output_displacement_norm": float(actual.float().norm().detach().cpu()),
        "target_norm": float(target.float().norm().detach().cpu()),
        "rz_to_target": float((actual.float().norm() / target.float().norm().clamp_min(1.0e-12)).detach().cpu()),
    }


def _run_p3_absorbable(args: argparse.Namespace, device: torch.device) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset in _canonical_tasks(args.datasets):
        x_train, y_train, _x_test, _y_test, in_dim, out_dim, protocol = v92._load_task(
            args, dataset, train_size=max(int(args.train_size), 4096), test_size=int(args.eval_size)
        )
        x_train = x_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device)
        xb = x_train[: int(args.audit_batch_size)]
        yb = y_train[: int(args.audit_batch_size)]
        xh = x_train[int(args.audit_batch_size) : 2 * int(args.audit_batch_size)]
        yh = y_train[int(args.audit_batch_size) : 2 * int(args.audit_batch_size)]
        for seed_text in _parse_list(args.seeds):
            seed = int(seed_text)
            lq_spec = lq.LQSpec("LQ-t2-h256", "t2", int(args.hidden_dim))
            lq_params, lq_mu, lq_std = lq.init_lq_params(in_dim, out_dim, lq_spec, x_train, device, seed + 921470)
            lq_logits = lq.lift_basis_forward_t2(xb, lq_params[0], lq_params[1], lq_params[2], lq_mu, lq_std, 2.0, 2.0)
            lq_pack = lq.lift_basis_fwd_bwd_t2(xb, yb, lq_params[0], lq_params[1], lq_params[2], lq_mu, lq_std, 2.0, 2.0)
            lq_grads = list(lq_pack[1:])
            task_step = snr_lq.gradient_descent_task_step(lq_grads, float(args.lr))
            adamw_logits = lq.lift_basis_forward_t2(xb, *(p + d for p, d in zip(lq_params, task_step)), lq_mu, lq_std, 2.0, 2.0)
            adamw_norm = float((adamw_logits - lq_logits).float().norm().detach().cpu())
            before_holdout = v92._classification_metrics_from_logits(
                lq.lift_basis_forward_t2(xh, lq_params[0], lq_params[1], lq_params[2], lq_mu, lq_std, 2.0, 2.0),
                yh,
            )
            for shadow_id in _parse_list(args.p3_shadow_actuators):
                shadow_spec = act.actuator_specs_from_ids([shadow_id])[0]
                shadow_params, shadow_mu, shadow_std = act.init_actuator_params(in_dim, out_dim, shadow_spec, x_train, device, seed + 921480)
                shadow_logits = act.actuator_forward(xb, shadow_params, shadow_mu, shadow_std, shadow_spec)
                for target_id in _parse_list(args.targets):
                    target, target_info = out_lq.build_output_target(target_id, shadow_logits, yb, dataset=dataset)
                    shadow_delta, shadow_info = act.actuator_only_least_squares_delta(shadow_params, shadow_mu, shadow_std, shadow_spec, xb, target, ridge=float(args.ridge))
                    shadow_fit = act.output_fit_metrics(shadow_params, shadow_delta, shadow_mu, shadow_std, shadow_spec, xb, target)
                    shadow_actual = act.actuator_forward(xb, [p + d for p, d in zip(shadow_params, shadow_delta)], shadow_mu, shadow_std, shadow_spec) - shadow_logits
                    for absorb_id in _parse_list(args.p3_absorb_candidates):
                        raw_absorb, absorb_info = _lq_least_squares_delta(lq_params, lq_mu, lq_std, "t2", xb, shadow_actual.detach(), absorb_id, float(args.ridge))
                        safe_absorb, removed = snr_lq.project_step_to_task_safe(raw_absorb, lq_grads)
                        fit = _lq_output_fit(lq_params, safe_absorb, lq_mu, lq_std, xb, shadow_actual.detach())
                        after_holdout = v92._classification_metrics_from_logits(
                            lq.lift_basis_forward_t2(xh, *(p + d for p, d in zip(lq_params, safe_absorb)), lq_mu, lq_std, 2.0, 2.0),
                            yh,
                        )
                        dd = _classification_delta(before_holdout, after_holdout)
                        rz = fit["output_displacement_norm"] / max(adamw_norm, 1.0e-12)
                        bad = int(dd["holdout_delta"] > 0.0)
                        rows.append({
                            "stage": "P3_ABSORBABLE_ACTUATOR_AUDIT",
                            "candidate_id": absorb_id,
                            "shadow_actuator": shadow_id,
                            "target_id": target_id,
                            "dataset": dataset,
                            "seed": seed,
                            "protocol": protocol,
                            "shadow_actuator_R2": shadow_fit["output_target_fit_r2"],
                            "shadow_actuator_rz": shadow_fit["output_displacement_to_target_ratio"],
                            "absorb_subspace": absorb_id,
                            "absorb_R2": fit["fit_R2"],
                            "absorb_rz": rz,
                            "holdout_delta": dd["holdout_delta"],
                            "CEp99_delta": dd["CEp99_delta"],
                            "margin_p10_delta": dd["margin_p10_delta"],
                            "bad_event": bad,
                            "step_ratio_q90": "LQ_base_source_P4",
                            "memory_ratio": "LQ_base_source_P4",
                            "functional_event_overhead": "event_time_LS_not_persistent_step",
                            "target_selected_fraction": target_info.get("target_selected_fraction", 0.0),
                            "projection_removed_norm": float(removed.detach().cpu()),
                            "absorbable_row_pass": int(fit["fit_R2"] >= 0.20 and rz >= 0.05 and bad == 0),
                            **shadow_info,
                            **absorb_info,
                            "fake_data_used": 0,
                            "proxy_row_used": 0,
                            "cpu_offload_used": 0,
                        })
    return rows


def _aggregate_absorb_pass(rows: Sequence[Dict[str, Any]]) -> Tuple[int, str, Dict[str, Any]]:
    best: Dict[str, Any] = {}
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r.get("candidate_id", "")), []).append(r)
        if not best or (_to_int(r.get("absorbable_row_pass")), _to_float(r.get("absorb_R2")), _to_float(r.get("absorb_rz"))) > (
            _to_int(best.get("absorbable_row_pass")),
            _to_float(best.get("absorb_R2")),
            _to_float(best.get("absorb_rz")),
        ):
            best = dict(r)
    for cid, rs in groups.items():
        measured = [r for r in rs if "absorb_R2" in r]
        if not measured:
            continue
        bad_rate = sum(_to_int(r.get("bad_event")) for r in measured) / max(1, len(measured))
        any_fit = any(_to_float(r.get("absorb_R2")) >= 0.20 and _to_float(r.get("absorb_rz")) >= 0.05 for r in measured)
        if any_fit and bad_rate <= 0.05:
            return 1, cid, best
    return 0, str(best.get("candidate_id", "")), best


def _write_not_run_downstream(out_dir: Path, reason: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    p4 = [_not_run("P4_P4_QUALIFIED_ACTUATOR_BASE_QUALIFICATION", "p4_p4_qualified_actuator_base_qualification.csv", reason)]
    p5 = [_not_run("P5_PAIRED_REPLAY_AFTER_ACTUATOR_CLOSURE", "p5_paired_replay_after_actuator_closure.csv", reason)]
    p6 = [_not_run("P6_FUNCTIONAL_REENTRY_AFTER_ACTUATOR_CLOSURE", "p6_functional_reentry_after_actuator_closure.csv", reason)]
    p7 = [_not_run("P7_NOISE_ROBUSTNESS_EXTERNAL_READY", "p7_noise_robustness_external_ready.csv", reason)]
    write_csv_rows(out_dir / "p4_p4_qualified_actuator_base_qualification.csv", p4)
    write_csv_rows(out_dir / "p5_paired_replay_after_actuator_closure.csv", p5)
    write_csv_rows(out_dir / "p6_functional_reentry_after_actuator_closure.csv", p6)
    write_csv_rows(out_dir / "p7_noise_robustness_external_ready.csv", p7)
    write_csv_rows(out_dir / "functional_event_trace_v9214.csv", [_not_run("FUNCTIONAL_EVENT_TRACE", "functional_event_trace_v9214.csv", reason)])
    return p4, p5, p6, p7


def _run_p4_base_qualification(args: argparse.Namespace, survivor: Dict[str, Any], device: torch.device) -> List[Dict[str, Any]]:
    cid = str(survivor.get("candidate_id", ""))
    if not cid or cid not in act.ACTUATOR_SPECS:
        return [_not_run("P4_P4_QUALIFIED_ACTUATOR_BASE_QUALIFICATION", "p4_p4_qualified_actuator_base_qualification.csv", "no_persistent_actuator_survivor_for_base_qualification")]
    spec = act.actuator_specs_from_ids([cid])[0]
    rows: List[Dict[str, Any]] = []
    for dataset in _canonical_tasks(args.p4_base_datasets):
        for seed_text in _parse_list(args.p4_base_seeds):
            row = v9213._train_actuator_candidate(args, spec, dataset, int(seed_text), device)
            row.update({
                "stage": "P4_P4_QUALIFIED_ACTUATOR_BASE_QUALIFICATION",
                "forward_ratio_q90": survivor.get("forward_ratio_q90", ""),
                "backward_ratio_q90": survivor.get("backward_ratio_q90", ""),
                "step_ratio_q90": survivor.get("step_ratio_q90", ""),
                "compact_memory_ratio": survivor.get("compact_memory_ratio", ""),
                "P4_remains_pass": survivor.get("P4_pass", 0),
                "target_fit_R2": survivor.get("target_fit_R2", ""),
                "output_displacement_ratio_rz": survivor.get("output_displacement_ratio_rz", ""),
            })
            rows.append(row)
    return rows


def _base_qualification_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    measured = [r for r in rows if str(r.get("status", "")) != "not_run" and "delta_vs_mlp" in r]
    deltas = [_to_float(r.get("delta_vs_mlp")) for r in measured]
    near = sum(_to_int(r.get("near_pass")) for r in measured)
    macro = sum(deltas) / max(1, len(deltas))
    return {
        "actuator_base_row_count": len(measured),
        "actuator_base_near_pass_count": near,
        "actuator_base_near_pass_rate": near / max(1, len(measured)),
        "actuator_base_macro_delta": macro,
        "actuator_base_near_pass": int(len(measured) > 0 and near / max(1, len(measured)) >= 0.80 and macro >= -0.01),
    }


def _clone_params(params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


def _clone_states(states: Sequence[AdamWState]) -> List[AdamWState]:
    return [AdamWState(step=s.step, m=s.m.detach().clone(), v=s.v.detach().clone()) for s in states]


def _cap_to_fraction(step: Sequence[torch.Tensor], task_step: Sequence[torch.Tensor], fraction: float) -> List[torch.Tensor]:
    norm = snr_lq.step_norm(step)
    cap = snr_lq.step_norm(task_step) * float(fraction)
    if bool((norm > cap).detach().cpu()):
        return snr_lq.scale_step(step, cap / norm.clamp_min(1.0e-12))
    return [s.detach().clone() for s in step]


def _random_like_step(params: Sequence[torch.Tensor], norm: torch.Tensor, seed: int) -> List[torch.Tensor]:
    gen = torch.Generator(device=params[0].device).manual_seed(int(seed))
    raw = [torch.randn(p.shape, device=p.device, generator=gen, dtype=p.dtype) for p in params]
    return snr_lq.scale_step(raw, norm / snr_lq.step_norm(raw).clamp_min(1.0e-12))


def _run_adamw_steps(
    params: Sequence[torch.Tensor],
    states: Sequence[AdamWState],
    mu: torch.Tensor,
    std: torch.Tensor,
    spec: act.ActuatorSpec,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    start_step: int,
    num_steps: int,
    batch_size: int,
    cfg: ManualAdamWConfig,
) -> None:
    n = int(x_train.shape[0])
    for j in range(int(num_steps)):
        start = ((int(start_step) + j) * int(batch_size)) % max(1, n - int(batch_size))
        xb = x_train[start : start + int(batch_size)]
        yb = y_train[start : start + int(batch_size)]
        _loss, grads = act.actuator_fwd_bwd(xb, yb, params, mu, std, spec)
        v92._adamw_update_foreach_(params, grads, states, cfg)


def _eval_actuator_metrics(params: Sequence[torch.Tensor], mu: torch.Tensor, std: torch.Tensor, spec: act.ActuatorSpec, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        logits = act.actuator_forward(x, params, mu, std, spec)
    return v92._classification_metrics_from_logits(logits, y)


def _functional_steps_for_target(
    args: argparse.Namespace,
    spec: act.ActuatorSpec,
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    xb: torch.Tensor,
    yb: torch.Tensor,
    dataset: str,
    target_id: str,
    seed: int,
) -> Dict[str, List[torch.Tensor]]:
    logits = act.actuator_forward(xb, params, mu, std, spec)
    _loss, grads = act.actuator_fwd_bwd(xb, yb, params, mu, std, spec)
    task_step = snr_lq.gradient_descent_task_step(grads, float(args.lr))
    target, _info = out_lq.build_output_target(target_id, logits, yb, dataset=dataset)
    raw_delta, _ls = act.actuator_only_least_squares_delta(params, mu, std, spec, xb, target, ridge=float(args.ridge))
    safe_delta, _removed = snr_lq.project_step_to_task_safe(raw_delta, grads)
    real = _cap_to_fraction(safe_delta, task_step, float(args.p5_functional_step_fraction))
    real_norm = snr_lq.step_norm(real)
    perm = torch.randperm(int(target.shape[0]), device=target.device, generator=torch.Generator(device=target.device).manual_seed(seed + 81))
    shuffled_raw, _ = act.actuator_only_least_squares_delta(params, mu, std, spec, xb, target[perm], ridge=float(args.ridge))
    shuffled, _ = snr_lq.project_step_to_task_safe(shuffled_raw, grads)
    rand_target = torch.randn(target.shape, device=target.device, generator=torch.Generator(device=target.device).manual_seed(seed + 82), dtype=target.dtype)
    rand_target = rand_target - rand_target.mean(dim=1, keepdim=True)
    absorb_rand_raw, _ = act.actuator_only_least_squares_delta(params, mu, std, spec, xb, rand_target, ridge=float(args.ridge))
    absorb_rand, _ = snr_lq.project_step_to_task_safe(absorb_rand_raw, grads)
    random_step, _ = snr_lq.project_step_to_task_safe(_random_like_step(params, real_norm, seed + 83), grads)
    raw_capped = _cap_to_fraction(raw_delta, task_step, float(args.p5_functional_step_fraction))
    return {
        "AdamWOnly": snr_lq.zero_like_params(params),
        "RealFunctional": real,
        "NoOpMatchedOverhead": snr_lq.zero_like_params(params),
        "RandomMatchedNorm": _cap_to_fraction(random_step, task_step, float(args.p5_functional_step_fraction)),
        "ShuffledTarget": _cap_to_fraction(shuffled, task_step, float(args.p5_functional_step_fraction)),
        "AdamWParallelDirection": _cap_to_fraction(task_step, task_step, float(args.p5_functional_step_fraction)),
        "ActuatorNoAbsorbControl": raw_capped,
        "AbsorbRandomTarget": _cap_to_fraction(absorb_rand, task_step, float(args.p5_functional_step_fraction)),
    }


def _run_p5_paired_replay(args: argparse.Namespace, survivor: Dict[str, Any], device: torch.device) -> Tuple[List[Dict[str, Any]], int, Dict[str, float]]:
    cid = str(survivor.get("candidate_id", ""))
    spec = act.actuator_specs_from_ids([cid])[0]
    horizons = sorted(int(x) for x in _parse_list(args.p5_horizons))
    rows: List[Dict[str, Any]] = []
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=0.0)
    for dataset in _canonical_tasks(args.p5_replay_datasets):
        x_train, y_train, x_eval, y_eval, in_dim, out_dim, protocol = v92._load_task(
            args, dataset, train_size=max(int(args.train_size), 4096), test_size=int(args.p5_eval_size)
        )
        x_train = x_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device)
        x_eval = x_eval.to(device=device, dtype=torch.float32)
        y_eval = y_eval.to(device=device)
        for seed_text in _parse_list(args.p5_replay_seeds):
            seed = int(seed_text)
            base_params, mu, std = act.init_actuator_params(in_dim, out_dim, spec, x_train, device, seed + 921490)
            base_states = [AdamWState.zeros_like(p) for p in base_params]
            _run_adamw_steps(base_params, base_states, mu, std, spec, x_train, y_train, 0, int(args.p5_warmup_steps), int(args.batch_size), cfg)
            xb = x_train[: int(args.audit_batch_size)]
            yb = y_train[: int(args.audit_batch_size)]
            for target_id in _parse_list(args.targets):
                branch_steps = _functional_steps_for_target(args, spec, base_params, mu, std, xb, yb, dataset, target_id, seed)
                metrics_by_branch: Dict[str, Dict[int, Dict[str, float]]] = {}
                for branch, fstep in branch_steps.items():
                    params_b = _clone_params(base_params)
                    states_b = _clone_states(base_states)
                    if branch not in {"AdamWOnly", "NoOpMatchedOverhead"}:
                        for p, d in zip(params_b, fstep):
                            p.add_(d)
                    prev = 0
                    metrics_by_branch[branch] = {}
                    for horizon in horizons:
                        _run_adamw_steps(params_b, states_b, mu, std, spec, x_train, y_train, int(args.p5_warmup_steps) + prev, horizon - prev, int(args.batch_size), cfg)
                        prev = horizon
                        metrics_by_branch[branch][horizon] = _eval_actuator_metrics(params_b, mu, std, spec, x_eval, y_eval)
                for horizon in horizons:
                    adamw = metrics_by_branch["AdamWOnly"][horizon]
                    for branch, by_h in metrics_by_branch.items():
                        m = by_h[horizon]
                        rows.append({
                            "stage": "P5_PAIRED_REPLAY_AFTER_ACTUATOR_CLOSURE",
                            "event_id": f"{dataset}-seed{seed}-{target_id}",
                            "candidate_id": cid,
                            "branch": branch,
                            "horizon": horizon,
                            "dataset": dataset,
                            "seed": seed,
                            "target_id": target_id,
                            "protocol": protocol,
                            "holdout_loss_delta": m["loss"] - adamw["loss"],
                            "val_proxy_acc_delta": m["acc"] - adamw["acc"],
                            "CEp99_delta": m["CE_p99"] - adamw["CE_p99"],
                            "margin_p10_delta": m["correct_margin_p10"] - adamw["correct_margin_p10"],
                            "curvature_delta": "not_measured",
                            "ECE_delta": m["ECE"] - adamw["ECE"],
                            "NLL_delta": m["NLL"] - adamw["NLL"],
                            "basis_entropy_delta": "not_measured",
                            "actuator_usage_delta": "not_measured",
                            "step_time": survivor.get("step_ratio_q90", ""),
                            "memory_ratio": survivor.get("compact_memory_ratio", ""),
                            "task_safe": int(m["acc"] >= adamw["acc"] - 0.005),
                            "loss_type": "CE",
                            "label_smoothing": 0,
                            "uses_loss_backward": 0,
                            "external_teacher_used": 0,
                            "functional_update_used": int(branch != "AdamWOnly"),
                            "fake_data_used": 0,
                            "proxy_row_used": 0,
                            "cpu_offload_used": 0,
                        })
    real = [r for r in rows if r.get("branch") == "RealFunctional"]
    controls = [r for r in rows if r.get("branch") not in {"RealFunctional", "AdamWOnly"}]
    if real and controls:
        real_ce = sum(_to_float(r.get("CEp99_delta")) for r in real) / len(real)
        real_margin = sum(_to_float(r.get("margin_p10_delta")) for r in real) / len(real)
        best_control_ce = min(sum(_to_float(r.get("CEp99_delta")) for r in controls if r.get("branch") == b) / max(1, sum(1 for r in controls if r.get("branch") == b)) for b in {r.get("branch") for r in controls})
        best_control_margin = max(sum(_to_float(r.get("margin_p10_delta")) for r in controls if r.get("branch") == b) / max(1, sum(1 for r in controls if r.get("branch") == b)) for b in {r.get("branch") for r in controls})
        task_safe_rate = sum(_to_int(r.get("task_safe")) for r in real) / len(real)
        mechanism = int(real_ce <= best_control_ce - 0.001 or real_margin >= best_control_margin + 0.002)
        passed = int(task_safe_rate >= 0.95 and mechanism == 1)
        summary = {
            "p5_real_mean_CEp99_delta": real_ce,
            "p5_best_control_mean_CEp99_delta": best_control_ce,
            "p5_real_mean_margin_delta": real_margin,
            "p5_best_control_mean_margin_delta": best_control_margin,
            "p5_task_safe_rate": task_safe_rate,
            "p5_mechanism_pass": mechanism,
        }
        return rows, passed, summary
    return rows, 0, {"p5_task_safe_rate": 0.0, "p5_mechanism_pass": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--train-size", type=int, default=9984)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--audit-batch-size", type=int, default=128)
    parser.add_argument("--p4-batch-size", type=int, default=128)
    parser.add_argument("--p4-warmup", type=int, default=5)
    parser.add_argument("--p4-reps", type=int, default=20)
    parser.add_argument("--p4-repeat-measurements", type=int, default=3)
    parser.add_argument("--p1-phase-reps", type=int, default=20)
    parser.add_argument("--p4-base-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--p4-base-seeds", default="0,1,2")
    parser.add_argument("--p5-train-size", type=int, default=9984)
    parser.add_argument("--p5-test-size", type=int, default=2000)
    parser.add_argument("--p5-epochs", type=int, default=20)
    parser.add_argument("--p5-lr", type=float, default=0.0005)
    parser.add_argument("--p5-replay-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--p5-replay-seeds", default="0,1,2")
    parser.add_argument("--p5-horizons", default="1,5,20,80")
    parser.add_argument("--p5-warmup-steps", type=int, default=36)
    parser.add_argument("--p5-eval-size", type=int, default=512)
    parser.add_argument("--p5-functional-step-fraction", type=float, default=0.10)
    parser.add_argument("--p1-candidates", default="A4-LQ-BoundedRationalActuator,A5-LQ-PiecewiseLinear2Actuator,A7-LQ-BasisEntropyActuator")
    parser.add_argument("--p2-candidates", default="A4b-BoundedRational-BranchlessDerivative,A4c-BoundedRational-FixedBeta,A4d-BoundedRational-ValueOnlyActuator,A4e-BoundedRational-FusedCoeffGrad,A5b-PiecewiseLinear2-Branchless,A5c-PiecewiseLinear2-FixedKnots,A7b-BasisEntropy-LowRank,A7c-BasisEntropy-ValueOnly")
    parser.add_argument("--p3-shadow-actuators", default="A4-LQ-BoundedRationalActuator,A5-LQ-PiecewiseLinear2Actuator,A7-LQ-BasisEntropyActuator,A4d-BoundedRational-ValueOnlyActuator,A7c-BasisEntropy-ValueOnly")
    parser.add_argument("--p3-absorb-candidates", default="AB0-ActuatorSolveThenAbsorb-T2,AB1-ActuatorSolveThenAbsorb-Lift,AB2-ActuatorSolveThenAbsorb-LiftPlusT2,AB3-ActuatorShadowLowRank,AB4-EventOnlyActuatorNoPersistentForward")
    parser.add_argument("--targets", default="O1-HardTailLogitCorrection,O2-MarginTailExpansion,O3-CalibrationTailCompression,O4-CurvatureOutputFlattening,O6-KMNISTHardModeOutputTarget")
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if out_dir.exists() and args.fresh:
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    manifest = {
        "experiment": "DG-KAN v9.2.14 P4Qualified Functional Actuator Closure",
        "created_at": _now_iso(),
        "device": str(device),
        "torch": torch.__version__,
        "plan_path": str(PLAN_PATH.relative_to(ROOT)),
        "script_path": str(SCRIPT_PATH.relative_to(ROOT)),
        "loss_type": "CE",
        "label_smoothing": 0,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "uses_loss_backward": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "args": vars(args),
    }
    write_json(out_dir / "run_manifest.json", manifest)

    contract = [{
        "candidate": "v9214_all_candidates",
        "strict_fc_purekan_edge_owned": 1,
        "ordinary_mlp_hidden_path_used": 0,
        "external_residual_shortcut_used": 0,
        "conv_or_former_used": 0,
        "loss_type": "CE",
        "label_smoothing": 0,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "uses_loss_backward": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    write_csv_rows(out_dir / "contract_audit_v9214.csv", contract)

    p0 = [_p0_recap(PREV_V9213)]
    write_csv_rows(out_dir / "p0_v9213_reproduction.csv", p0)

    x_train, y_train, _x_test, _y_test, in_dim, out_dim, _protocol = v92._load_task(
        args, "MNIST", train_size=max(4096, int(args.p4_batch_size) * 4), test_size=256
    )
    x_train = x_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device)
    p1_rows = [
        _phase_attribution(args, spec, x_train, y_train, in_dim, out_dim, device)
        for spec in act.actuator_specs_from_ids(_parse_list(args.p1_candidates))
    ]
    write_csv_rows(out_dir / "p1_actuator_p4_failure_attribution.csv", p1_rows)
    write_csv_rows(out_dir / "actuator_phase_trace_v9214.csv", p1_rows)

    lq_safe_r2 = _to_float(p0[0].get("current_lq_safe_max_R2"))
    p2_rows = _run_p2(args, device, lq_safe_r2)
    write_csv_rows(out_dir / "p2_p4_closure_candidate_factory.csv", p2_rows)

    p3_rows = _run_p3_absorbable(args, device)
    write_csv_rows(out_dir / "p3_absorbable_actuator_audit.csv", p3_rows)
    write_csv_rows(out_dir / "absorbable_actuator_trace_v9214.csv", p3_rows)

    p2_survivors = [r for r in p2_rows if _to_int(r.get("p4_controllable_pass")) == 1]
    absorb_pass, best_absorb_id, best_absorb = _aggregate_absorb_pass(p3_rows)
    p4_closure_pass = int(bool(p2_survivors))
    if p2_survivors:
        best_survivor_for_base = sorted(
            p2_survivors,
            key=lambda r: (_to_float(r.get("target_fit_R2")), _to_float(r.get("output_displacement_ratio_rz")), -_to_float(r.get("backward_ratio_q90"))),
            reverse=True,
        )[0]
        p4_rows = _run_p4_base_qualification(args, best_survivor_for_base, device)
        write_csv_rows(out_dir / "p4_p4_qualified_actuator_base_qualification.csv", p4_rows)
        base_summary = _base_qualification_summary(p4_rows)
        if base_summary["actuator_base_near_pass"]:
            p5_rows, p5_pass, p5_summary = _run_p5_paired_replay(args, best_survivor_for_base, device)
            write_csv_rows(out_dir / "p5_paired_replay_after_actuator_closure.csv", p5_rows)
            write_csv_rows(out_dir / "functional_event_trace_v9214.csv", p5_rows)
            if p5_pass:
                reason = "P5_paired_replay_passed_but_P6_full_functional_not_implemented_in_this_runner"
            else:
                reason = "P5_paired_replay_causality_failed"
        else:
            reason = "P4_qualified_actuator_destroyed_or_failed_AdamW_base_nearpass"
            p5_summary = {"p5_task_safe_rate": 0.0, "p5_mechanism_pass": 0}
            p5_pass = 0
            p5_rows = [_not_run("P5_PAIRED_REPLAY_AFTER_ACTUATOR_CLOSURE", "p5_paired_replay_after_actuator_closure.csv", reason)]
            write_csv_rows(out_dir / "p5_paired_replay_after_actuator_closure.csv", p5_rows)
            write_csv_rows(out_dir / "functional_event_trace_v9214.csv", [_not_run("FUNCTIONAL_EVENT_TRACE", "functional_event_trace_v9214.csv", reason)])
        p6_rows = [_not_run("P6_FUNCTIONAL_REENTRY_AFTER_ACTUATOR_CLOSURE", "p6_functional_reentry_after_actuator_closure.csv", reason)]
        p7_rows = [_not_run("P7_NOISE_ROBUSTNESS_EXTERNAL_READY", "p7_noise_robustness_external_ready.csv", reason)]
        write_csv_rows(out_dir / "p6_functional_reentry_after_actuator_closure.csv", p6_rows)
        write_csv_rows(out_dir / "p7_noise_robustness_external_ready.csv", p7_rows)
    elif absorb_pass:
        reason = "absorbable_actuator_survivor_requires_paired_replay_followup"
        p4_rows, p5_rows, p6_rows, p7_rows = _write_not_run_downstream(out_dir, reason)
        base_summary = {
            "actuator_base_row_count": 0,
            "actuator_base_near_pass_count": 0,
            "actuator_base_near_pass_rate": 0.0,
            "actuator_base_macro_delta": 0.0,
            "actuator_base_near_pass": 0,
        }
        p5_pass = 0
        p5_summary = {"p5_task_safe_rate": 0.0, "p5_mechanism_pass": 0}
    else:
        reason = "no_P4_qualified_controllable_actuator_or_absorbable_actuator"
        p4_rows, p5_rows, p6_rows, p7_rows = _write_not_run_downstream(out_dir, reason)
        base_summary = {
            "actuator_base_row_count": 0,
            "actuator_base_near_pass_count": 0,
            "actuator_base_near_pass_rate": 0.0,
            "actuator_base_macro_delta": 0.0,
            "actuator_base_near_pass": 0,
        }
        p5_pass = 0
        p5_summary = {"p5_task_safe_rate": 0.0, "p5_mechanism_pass": 0}

    if p2_survivors:
        best = sorted(
            p2_survivors,
            key=lambda r: (_to_float(r.get("target_fit_R2")), _to_float(r.get("output_displacement_ratio_rz")), -_to_float(r.get("backward_ratio_q90"))),
            reverse=True,
        )[0]
    else:
        best = sorted(
            [r for r in p2_rows if str(r.get("status", "")) != "not_implemented"],
            key=lambda r: (_to_int(r.get("P4_pass")), _to_float(r.get("target_fit_R2")), _to_float(r.get("output_displacement_ratio_rz"))),
            reverse=True,
        )[0] if p2_rows else {}

    if p4_closure_pass and base_summary["actuator_base_near_pass"] and p5_pass:
        route = "R5-ActuatorFunctionalCausalityPass"
        blocker = "paired_replay_passed_but_full_functional_not_opened"
        failure_code = "F16_artifact_missing"
    elif p4_closure_pass and base_summary["actuator_base_near_pass"]:
        route = "R4-ActuatorP4ClosedBaseQualified"
        blocker = "P4_qualified_actuator_base_nearpass_but_paired_replay_causality_failed"
        failure_code = "F9_paired_replay_causality_fail"
    elif p4_closure_pass:
        route = "R8-P4CloseButBaseTrainabilityFail"
        blocker = "P4_closure_found_but_actuator_base_trainability_failed"
        failure_code = "F8_actuator_base_trainability_fail"
    elif absorb_pass:
        route = "R3-AbsorbableActuatorPass"
        blocker = "absorbable_actuator_found_but_paired_replay_not_opened_in_first_wave"
        failure_code = "F16_artifact_missing"
    else:
        route = "R11-ReturnToBasisFactory"
        blocker = "all_P4_close_candidates_failed_or_absorbable_actuator_failed_safe_fit_rz_gate"
        failure_code = "F5_actuator_p4_fail"

    route_decision = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "best_actuator_candidate": best.get("candidate_id", ""),
        "best_actuator_family": best.get("actuator_type", ""),
        "best_absorbable_candidate": best_absorb_id,
        "best_output_target": best.get("best_target", best_absorb.get("target_id", "")),
        "best_solver": "task_safe_LS_projection",
        "p4_closure_pass": p4_closure_pass,
        "controllability_retained": int(any(_to_int(r.get("controllability_retained")) == 1 for r in p2_rows)),
        "absorbable_pass": absorb_pass,
        "actuator_base_near_pass": 0,
        **base_summary,
        "paired_replay_pass": 0,
        **p5_summary,
        "paired_replay_pass": int(p5_pass),
        "full_functional_pass": 0,
        "noise_robustness_pass": 0,
        "strong_baseline_pass": 0,
        "external_ready": 0,
        "forward_ratio_q90": _to_float(best.get("forward_ratio_q90")),
        "backward_ratio_q90": _to_float(best.get("backward_ratio_q90")),
        "step_ratio_q90": _to_float(best.get("step_ratio_q90")),
        "memory_ratio": _to_float(best.get("compact_memory_ratio")),
        "target_fit_R2": _to_float(best.get("target_fit_R2", best_absorb.get("absorb_R2", 0))),
        "output_displacement_ratio_rz": _to_float(best.get("output_displacement_ratio_rz", best_absorb.get("absorb_rz", 0))),
        "primary_blocker": blocker,
        "next_required_implementation": "return_to_actuator_basis_factory_with_P4_system_constraint" if route == "R11-ReturnToBasisFactory" else "open_base_qualification_and_paired_replay_for_survivor",
        "success_v9214_p4_qualified_actuator": p4_closure_pass,
        "success_v9214_absorbable_actuator": absorb_pass,
        "success_v9214_functional_advantage": 0,
        "success_v9214_external_ready": 0,
    }
    write_json(out_dir / "route_decision.json", route_decision)
    write_json(out_dir / "aggregate_decision.json", route_decision)
    write_csv_rows(out_dir / "failure_table.csv", [{
        "stage": "terminal",
        "failure_code": failure_code,
        "primary_blocker": blocker,
        "route": route,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])

    _write_svg(out_dir / "figures" / "p0_v9213_boundary_dashboard.svg", "P0 v9.2.13 Boundary", f"route={p0[0].get('source_route')} p4_pass_count={p0[0].get('actuator_p4_pass_count')}")
    _write_svg(out_dir / "figures" / "p0_current_vs_actuator_controllability.svg", "Current vs Actuator Controllability", f"safe_rz={p0[0].get('current_lq_safe_max_rz')} post_pass={p0[0].get('post_actuator_controllability_pass_count')}")
    _write_svg(out_dir / "figures" / "p0_p4_fail_summary.svg", "P4 Fail Summary", f"actuator_p4_pass_count={p0[0].get('actuator_p4_pass_count')}")
    _write_svg(out_dir / "figures" / "p1_p4_phase_waterfall.svg", "P1 Phase Attribution", f"rows={len(p1_rows)} attribution_pass={sum(_to_int(r.get('attribution_pass')) for r in p1_rows)}")
    _write_svg(out_dir / "figures" / "p1_backward_excess_by_component.svg", "P1 Backward Excess", f"backward_dominant={sum(_to_int(r.get('backward_dominant')) for r in p1_rows)}")
    _write_svg(out_dir / "figures" / "p1_forward_excess_by_component.svg", "P1 Forward Excess", "component timer rows only")
    _write_svg(out_dir / "figures" / "p1_kernel_count_by_actuator.svg", "P1 Kernel Count", "torch profiler not decomposed; component timing used")
    _write_svg(out_dir / "figures" / "p2_actuator_p4_pareto.svg", "P2 Actuator P4 Pareto", f"p4_controllable_survivors={len(p2_survivors)}")
    _write_svg(out_dir / "figures" / "p2_controllability_vs_backward.svg", "P2 Controllability vs Backward", f"best={best.get('candidate_id','')}")
    _write_svg(out_dir / "figures" / "p2_a4_family_before_after.svg", "P2 A4 Family", f"route={route}")
    _write_svg(out_dir / "figures" / "p2_gradcheck_by_candidate.svg", "P2 Gradcheck", f"grad_pass={sum(_to_int(r.get('GradPass')) for r in p2_rows)}")
    _write_svg(out_dir / "figures" / "p3_absorb_fit_heatmap.svg", "P3 Absorb Fit", f"absorbable_pass={absorb_pass}")
    _write_svg(out_dir / "figures" / "p3_shadow_vs_absorbed_displacement.svg", "P3 Shadow vs Absorbed", f"best_absorb={best_absorb_id}")
    _write_svg(out_dir / "figures" / "p3_absorb_bad_event_rate.svg", "P3 Absorb Bad Event", "bad_event from holdout loss delta")
    _write_svg(out_dir / "figures" / "p3_absorb_overhead.svg", "P3 Absorb Overhead", "event-time LS; persistent step not opened")

    audit_paths = [
        out_dir / "contract_audit_v9214.csv",
        out_dir / "p0_v9213_reproduction.csv",
        out_dir / "p1_actuator_p4_failure_attribution.csv",
        out_dir / "p2_p4_closure_candidate_factory.csv",
        out_dir / "p3_absorbable_actuator_audit.csv",
        out_dir / "p4_p4_qualified_actuator_base_qualification.csv",
        out_dir / "p5_paired_replay_after_actuator_closure.csv",
        out_dir / "p6_functional_reentry_after_actuator_closure.csv",
        out_dir / "p7_noise_robustness_external_ready.csv",
        out_dir / "functional_event_trace_v9214.csv",
        out_dir / "actuator_phase_trace_v9214.csv",
        out_dir / "absorbable_actuator_trace_v9214.csv",
        out_dir / "failure_table.csv",
    ]
    audit = audit_no_fake(audit_paths)
    write_csv_rows(out_dir / "v9214_provenance_audit.csv", [audit])
    route_decision.update(audit)
    write_json(out_dir / "route_decision.json", route_decision)
    write_json(out_dir / "aggregate_decision.json", route_decision)
    hash_paths = [
        PLAN_PATH,
        SCRIPT_PATH,
        ROOT / "dgkan" / "models" / "fc_purekan_actuator.py",
        out_dir / "route_decision.json",
        *audit_paths,
        out_dir / "v9214_provenance_audit.csv",
    ]
    write_csv_rows(out_dir / "artifact_hashes.csv", artifact_hash_rows(hash_paths, root=ROOT))
    print(json.dumps(route_decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
