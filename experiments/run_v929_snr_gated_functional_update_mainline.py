#!/usr/bin/env python3
"""DG-KAN v9.2.9 SNR-gated functional update mainline runner.

This first mainline runner opens functional work only through the planned
gates.  P1 is pure SNR instrumentation: no functional update is applied.  If
SNR stability/overhead/active-fraction gates fail, downstream functional
stages are written as explicit not_run artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v92_basis_source_kernel_gate as v92  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, write_csv_rows, write_json  # noqa: E402
from dgkan.functional import snr_gated_lq as snr_lq  # noqa: E402
from dgkan.models import fc_purekan_lq as lq  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.9_SNR_Gated_Functional_Update_Mainline_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v929_snr_gated_functional_update_mainline.py"
PREV_V927 = ROOT / "results" / "real_rerun_20260506" / "v927_fc_purekan_lq_fullpass_functional_gate_20260509T190000Z"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _canonical_tasks(text: str) -> List[str]:
    return [v92._canonical_task(x) for x in _parse_list(text)]


def _device_from_arg(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def _write_svg(path: Path, title: str, subtitle: str) -> None:
    ensure_dir(path.parent)
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="920" height="180" viewBox="0 0 920 180">
  <rect width="920" height="180" fill="#f8fafc"/>
  <text x="28" y="54" font-family="Arial, sans-serif" font-size="26" fill="#111827">{title}</text>
  <text x="28" y="96" font-family="Arial, sans-serif" font-size="16" fill="#374151">{subtitle}</text>
  <text x="28" y="134" font-family="Arial, sans-serif" font-size="13" fill="#6b7280">Generated from measured CSV/JSON fields; no inferred pass values.</text>
</svg>
""",
        encoding="utf-8",
    )


def _not_run(stage: str, artifact: str, reason: str) -> Dict[str, Any]:
    return snr_lq.not_run_row(stage, artifact, reason)


def _candidate_specs(text: str) -> List[lq.LQSpec]:
    specs: List[lq.LQSpec] = []
    for name in _parse_list(text):
        if name == "LQ0":
            specs.append(lq.LQSpec("LQ0-LQ-t2-h256-AdamW", "t2", 256, "default", 1.0))
        elif name == "LQ1":
            specs.append(lq.LQSpec("LQ1-LQ-t2-h256-fanin-output-scale", "t2", 256, "default", 0.8))
        else:
            raise ValueError(f"unknown v9.2.9 candidate alias {name}")
    return specs


def _load_p0_previous(prev_dir: Path) -> Dict[str, Any]:
    route = _read_json(prev_dir / "route_decision.json")
    robust_rows = _read_csv(prev_dir / "p3_robust_nearpass_confirmation.csv")
    repair_rows = _read_csv(prev_dir / "p5_fullpass_repair_candidates.csv")
    return {
        "route": route,
        "robust_rows": robust_rows,
        "repair_rows": repair_rows,
        "source_artifact": str(prev_dir.relative_to(ROOT)) if prev_dir.exists() else str(prev_dir),
    }


def _run_p1_snr(args: argparse.Namespace, device: torch.device) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    specs = _candidate_specs(args.p1_candidates)
    if str(args.p1_estimator) == "ghost_microbatch":
        snr_cfgs = [
            snr_lq.SNRConfig(
                microbatch_count=int(m),
                tau1=float(args.snr_tau1),
                tau2=float(args.snr_tau2),
                temperature=float(args.snr_smooth_temperature),
            )
            for m in _parse_list(args.p1_microbatches)
        ]
    else:
        snr_cfgs = [
            snr_lq.SNRConfig(
                microbatch_count=1,
                tau1=float(args.snr_tau1),
                tau2=float(args.snr_tau2),
                temperature=float(args.snr_smooth_temperature),
            )
        ]
    for dataset in _canonical_tasks(args.p1_datasets):
        x_train, y_train, _x_test, _y_test, input_dim, output_dim, protocol = v92._load_task(
            args,
            dataset,
            train_size=max(int(args.p1_train_size), int(args.batch_size) * (int(args.p1_steps) + 2)),
            test_size=256,
        )
        x_train = x_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device)
        for seed_text in _parse_list(args.p1_seeds):
            seed = int(seed_text)
            for spec in specs:
                torch.manual_seed(int(args.seed) + seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(int(args.seed) + seed)
                params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, seed + 92600)
                states = [AdamWState.zeros_like(p) for p in params]
                opt_cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
                for mb_cfg in snr_cfgs:
                    ema_state = snr_lq.EMARoleSNRState.zeros_like(params, beta=float(args.p1_ema_beta))
                    scalar_state = snr_lq.ScalarRoleSNRState.zeros(
                        len(snr_lq.role_names_for_basis(spec.basis)),
                        beta=float(args.p1_ema_beta),
                        device=device,
                    )
                    for step in range(int(args.p1_steps)):
                        gen = torch.Generator(device=device).manual_seed(929000 + seed * 1000 + step)
                        perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)
                        idx = perm[: int(args.batch_size)]
                        xb = x_train[idx]
                        yb = y_train[idx]
                        measure_this_step = str(args.p1_estimator) != "ema_role_scalar" or step % max(1, int(args.p1_snr_stride)) == 0
                        if not measure_this_step:
                            snr_lq.measure_manual_step(
                                x=xb,
                                y=yb,
                                params=params,
                                states=states,
                                mu=mu,
                                std=std,
                                basis=spec.basis,
                                cfg=opt_cfg,
                                device=device,
                                update_fn=v92._adamw_update_foreach_,
                            )
                            continue
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(device)
                        if str(args.p1_estimator) == "ghost_microbatch":
                            snr_ms, role_rows = snr_lq.compute_microbatch_snr(
                                x=xb,
                                y=yb,
                                params=params,
                                mu=mu,
                                std=std,
                                basis=spec.basis,
                                cfg=mb_cfg,
                                device=device,
                            )
                            snr_mem_mb = (
                                float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
                                if device.type == "cuda"
                                else 0.0
                            )
                            step_ms, loss = snr_lq.measure_manual_step(
                                x=xb,
                                y=yb,
                                params=params,
                                states=states,
                                mu=mu,
                                std=std,
                                basis=spec.basis,
                                cfg=opt_cfg,
                                device=device,
                                update_fn=v92._adamw_update_foreach_,
                            )
                        elif str(args.p1_estimator) == "ema_role":
                            snr_ms, step_ms, loss, role_rows = snr_lq.measure_ema_role_snr_step(
                                x=xb,
                                y=yb,
                                params=params,
                                states=states,
                                ema_state=ema_state,
                                mu=mu,
                                std=std,
                                basis=spec.basis,
                                snr_cfg=mb_cfg,
                                opt_cfg=opt_cfg,
                                device=device,
                                update_fn=v92._adamw_update_foreach_,
                                histogram_max_items=int(args.p1_histogram_max_items),
                            )
                            snr_mem_mb = (
                                float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
                                if device.type == "cuda"
                                else 0.0
                            )
                        elif str(args.p1_estimator) == "ema_role_scalar":
                            snr_ms, step_ms, loss, role_rows = snr_lq.measure_scalar_ema_role_snr_step(
                                x=xb,
                                y=yb,
                                params=params,
                                states=states,
                                scalar_state=scalar_state,
                                mu=mu,
                                std=std,
                                basis=spec.basis,
                                snr_cfg=mb_cfg,
                                opt_cfg=opt_cfg,
                                device=device,
                                update_fn=v92._adamw_update_foreach_,
                            )
                            snr_mem_mb = (
                                float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
                                if device.type == "cuda"
                                else 0.0
                            )
                        else:
                            raise ValueError(f"unknown p1 estimator {args.p1_estimator}")
                        raw_overhead = snr_ms / max(step_ms, 1.0e-12)
                        overhead = (
                            raw_overhead / max(1, int(args.p1_snr_stride))
                            if str(args.p1_estimator) == "ema_role_scalar"
                            else raw_overhead
                        )
                        for role_row in role_rows:
                            role = str(role_row["role"])
                            row = {
                                "stage": "P1_SNR_INSTRUMENTATION",
                                "candidate_id": spec.candidate_id,
                                "dataset": dataset,
                                "seed": seed,
                                "protocol": protocol,
                                "step": step,
                                "role": role,
                                "estimator_type": str(args.p1_estimator),
                                "microbatch_count": int(mb_cfg.microbatch_count),
                                "ema_beta": float(args.p1_ema_beta) if str(args.p1_estimator) in {"ema_role", "ema_role_scalar"} else "",
                                "ema_warmup_steps": int(args.p1_ema_warmup_steps) if str(args.p1_estimator) in {"ema_role", "ema_role_scalar"} else 0,
                                "eligible_for_gate": int(str(args.p1_estimator) not in {"ema_role", "ema_role_scalar"} or step >= int(args.p1_ema_warmup_steps)),
                                "batch_size": int(args.batch_size),
                                "train_size": int(args.p1_train_size),
                                "loss_after_step": loss,
                                "SNR_compute_time_ms": snr_ms,
                                "step_time_ms": step_ms,
                                "SNR_overhead_ratio": overhead,
                                "raw_SNR_overhead_ratio": raw_overhead,
                                "SNR_subsample_stride": max(1, int(args.p1_snr_stride)) if str(args.p1_estimator) == "ema_role_scalar" else 1,
                                "SNR_memory_MB": snr_mem_mb,
                                "basis_channel_SNR": role_row["SNR_role"] if "coeff" in role else "",
                                "lift_identity_SNR": role_row["SNR_role"] if role == "lift_identity" else "",
                                "quadratic_coeff_SNR": role_row["SNR_role"] if role == "quadratic_coeff" else "",
                                "output_linear_SNR": role_row["SNR_role"] if role == "output_linear" else "",
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
                                "functional_update_used": 0,
                                "fake_data_used": 0,
                                "proxy_row_used": 0,
                                "cpu_offload_used": 0,
                            }
                            row.update(role_row)
                            rows.append(row)
    cv_source_rows = [r for r in rows if _to_int(r.get("eligible_for_gate", 1), 1) == 1]
    cv_by_group = snr_lq.summarize_stability(cv_source_rows)
    for row in rows:
        key = (
            str(row.get("candidate_id", "")),
            str(row.get("dataset", "")),
            int(row.get("seed", 0)),
            int(row.get("microbatch_count", 0)),
            str(row.get("role", "")),
        )
        cv = cv_by_group.get(key, 0.0)
        active = _to_float(row.get("SNR_active_fraction_tau1"))
        eligible = _to_int(row.get("eligible_for_gate", 1), 1)
        row["SNR_role_cv"] = cv
        row["estimator_stability_pass"] = int((not eligible) or cv <= float(args.snr_cv_gate))
        row["SNR_overhead_pass"] = int((not eligible) or _to_float(row.get("SNR_overhead_ratio")) <= float(args.snr_overhead_gate))
        row["active_fraction_sanity_pass"] = int((not eligible) or float(args.snr_active_min) <= active <= float(args.snr_active_max))
    return rows


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return 0.0
    tx = torch.tensor(list(xs), dtype=torch.float64)
    ty = torch.tensor(list(ys), dtype=torch.float64)
    vx = tx - tx.mean()
    vy = ty - ty.mean()
    denom = (vx.square().sum().sqrt() * vy.square().sum().sqrt()).item()
    if denom <= 1.0e-12:
        return 0.0
    return float((vx * vy).sum().item() / denom)


def _eval_lq_metrics(
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    basis: str,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Dict[str, float]:
    fwd, _bwd = lq.functions_for_basis(basis)
    with torch.no_grad():
        logits = fwd(x, *params, mu, std, 2.0, 2.0)
    return v92._classification_metrics_from_logits(logits, y)


def _quadratic_role_snr(rows: Sequence[Dict[str, Any]]) -> float:
    for row in rows:
        role = str(row.get("role", ""))
        if "quadratic" in role or "legendre_p2" in role:
            return _to_float(row.get("SNR_role"), 0.0)
    return _to_float(rows[-1].get("SNR_role"), 0.0) if rows else 0.0


def _quadratic_active_fraction(rows: Sequence[Dict[str, Any]]) -> float:
    for row in rows:
        role = str(row.get("role", ""))
        if "quadratic" in role or "legendre_p2" in role:
            return _to_float(row.get("SNR_active_fraction_tau1"), 0.0)
    return _to_float(rows[-1].get("SNR_active_fraction_tau1"), 0.0) if rows else 0.0


def _run_p2_one_step(args: argparse.Namespace, device: torch.device) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    specs = _candidate_specs(args.p2_candidates)
    snr_cfg = snr_lq.SNRConfig(
        microbatch_count=1,
        tau1=float(args.snr_tau1),
        tau2=float(args.snr_tau2),
        temperature=float(args.snr_smooth_temperature),
    )
    variants = _parse_list(args.p2_functional_modes)
    for dataset in _canonical_tasks(args.p2_datasets):
        min_train = max(int(args.p2_train_size), int(args.p2_microbatch_size) * 2 * (int(args.p2_steps) + 2))
        x_train, y_train, _x_test, _y_test, input_dim, output_dim, protocol = v92._load_task(
            args,
            dataset,
            train_size=min_train,
            test_size=256,
        )
        x_train = x_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device)
        for seed_text in _parse_list(args.p2_seeds):
            seed = int(seed_text)
            for spec in specs:
                torch.manual_seed(int(args.seed) + seed + 292900)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(int(args.seed) + seed + 292900)
                params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, seed + 92920)
                states = [AdamWState.zeros_like(p) for p in params]
                opt_cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
                scalar_state = snr_lq.ScalarRoleSNRState.zeros(
                    len(snr_lq.role_names_for_basis(spec.basis)),
                    beta=float(args.p1_ema_beta),
                    device=device,
                )
                _fwd, bwd = lq.functions_for_basis(spec.basis)
                for warm_step in range(int(args.p2_snr_warmup_steps)):
                    gen = torch.Generator(device=device).manual_seed(9291000 + seed * 10000 + warm_step)
                    perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)
                    idx = perm[: int(args.batch_size)]
                    pack = bwd(x_train[idx], y_train[idx], *params, mu, std, 2.0, 2.0)
                    warm_grads = [g.detach() for g in pack[1:]]
                    snr_lq.scalar_ema_snr_from_grads(
                        grads=warm_grads,
                        scalar_state=scalar_state,
                        basis=spec.basis,
                        snr_cfg=snr_cfg,
                    )
                    v92._adamw_update_foreach_(params, warm_grads, states, opt_cfg)
                for step in range(int(args.p2_steps)):
                    gen = torch.Generator(device=device).manual_seed(9292000 + seed * 10000 + step)
                    perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)
                    n_micro = int(args.p2_microbatch_size)
                    train_idx = perm[:n_micro]
                    hold_idx = perm[n_micro : 2 * n_micro]
                    xb = x_train[train_idx]
                    yb = y_train[train_idx]
                    xh = x_train[hold_idx]
                    yh = y_train[hold_idx]
                    pack = bwd(xb, yb, *params, mu, std, 2.0, 2.0)
                    grads = [g.detach() for g in pack[1:]]
                    snr_rows = snr_lq.scalar_ema_snr_from_grads(
                        grads=grads,
                        scalar_state=scalar_state,
                        basis=spec.basis,
                        snr_cfg=snr_cfg,
                    )
                    active_fraction = _quadratic_active_fraction(snr_rows)
                    quadratic_snr = _quadratic_role_snr(snr_rows)
                    before = _eval_lq_metrics(params, mu, std, spec.basis, xh, yh)
                    task_step = snr_lq.gradient_descent_task_step(grads, float(args.lr))
                    adamw_step_norm = snr_lq.step_norm(task_step)
                    base_direction = snr_lq.scale_direction_to_fraction_of_task_step(
                        snr_lq.quadratic_coeff_direction(params, spec.basis),
                        task_step,
                        float(args.p2_functional_step_fraction),
                    )
                    snr_direction = snr_lq.scale_step(base_direction, active_fraction)
                    for mode in variants:
                        if mode == "F1-SNRGatedTaskProjected":
                            raw_delta = snr_direction
                            snr_gate_type = str(args.p1_estimator)
                            gate = active_fraction
                        elif mode == "C1-GeometryOnlyNoSNRTaskProjected":
                            raw_delta = base_direction
                            snr_gate_type = "none_geometry_control"
                            gate = 1.0
                        elif mode == "C0-NoOp":
                            raw_delta = snr_lq.zero_like_params(params)
                            snr_gate_type = "none_noop_control"
                            gate = 0.0
                        elif mode == "C2-RandomMatchedNormTaskProjected":
                            rand = snr_lq.zero_like_params(params)
                            rgen = torch.Generator(device=device).manual_seed(9392000 + seed * 10000 + step)
                            rand[-1] = torch.randn(params[-1].shape, device=device, generator=rgen, dtype=params[-1].dtype)
                            raw_delta = snr_lq.scale_direction_to_fraction_of_task_step(
                                rand,
                                task_step,
                                float(args.p2_functional_step_fraction) * max(active_fraction, 1.0e-6),
                            )
                            snr_gate_type = "none_random_control"
                            gate = active_fraction
                        else:
                            raise ValueError(f"unknown P2 functional mode {mode}")
                        if mode in {"F1-SNRGatedTaskProjected", "C1-GeometryOnlyNoSNRTaskProjected", "C2-RandomMatchedNormTaskProjected"}:
                            delta, removed_norm = snr_lq.project_step_to_task_safe(raw_delta, grads)
                        else:
                            delta = raw_delta
                            removed_norm = torch.zeros((), device=device, dtype=torch.float32)
                        g_dot = snr_lq.step_dot(grads, delta)
                        func_norm = snr_lq.step_norm(delta)
                        cos = g_dot / (snr_lq.step_norm(grads).clamp_min(1.0e-12) * func_norm.clamp_min(1.0e-12))
                        after_params = snr_lq.apply_step(params, delta)
                        after = _eval_lq_metrics(after_params, mu, std, spec.basis, xh, yh)
                        actual_delta = after["loss"] - before["loss"]
                        rows.append(
                            {
                                "stage": "P2_ONE_STEP_POPULATION_RISK_AUDIT",
                                "candidate_id": spec.candidate_id,
                                "dataset": dataset,
                                "seed": seed,
                                "protocol": protocol,
                                "step": step,
                                "functional_mode": mode,
                                "snr_gate_type": snr_gate_type,
                                "active_fraction": active_fraction,
                                "quadratic_role_snr": quadratic_snr,
                                "snr_gate_multiplier": gate,
                                "predicted_population_improvement": float(g_dot.detach().cpu()),
                                "actual_holdout_loss_before": before["loss"],
                                "actual_holdout_loss_after": after["loss"],
                                "actual_holdout_delta": actual_delta,
                                "holdout_nonharm": int(actual_delta <= 0.0),
                                "task_gradient_dot_functional_step": float(g_dot.detach().cpu()),
                                "projection_removed_norm": float(removed_norm.detach().cpu()),
                                "functional_step_norm": float(func_norm.detach().cpu()),
                                "adamw_step_norm": float(adamw_step_norm.detach().cpu()),
                                "cos_functional_task": float(cos.detach().cpu()) if math.isfinite(float(cos.detach().cpu())) else 0.0,
                                "bad_step": int(actual_delta > 0.0),
                                "CEp99_before": before["CE_p99"],
                                "CEp99_after": after["CE_p99"],
                                "margin_p10_before": before["correct_margin_p10"],
                                "margin_p10_after": after["correct_margin_p10"],
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
                                "functional_update_used": int(mode != "C0-NoOp"),
                                "fake_data_used": 0,
                                "proxy_row_used": 0,
                                "cpu_offload_used": 0,
                            }
                        )
                    v92._adamw_update_foreach_(params, grads, states, opt_cfg)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--previous-v927-dir", default=str(PREV_V927.relative_to(ROOT)))
    parser.add_argument("--p1-candidates", default="LQ0,LQ1")
    parser.add_argument("--p1-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--p1-seeds", default="0,1,2")
    parser.add_argument("--p1-microbatches", default="4,8")
    parser.add_argument("--p1-steps", type=int, default=3)
    parser.add_argument("--p1-train-size", type=int, default=9984)
    parser.add_argument("--p1-estimator", choices=["ghost_microbatch", "ema_role", "ema_role_scalar"], default="ghost_microbatch")
    parser.add_argument("--p1-ema-beta", type=float, default=0.90)
    parser.add_argument("--p1-ema-warmup-steps", type=int, default=4)
    parser.add_argument("--p1-histogram-max-items", type=int, default=2048)
    parser.add_argument("--p1-snr-stride", type=int, default=1)
    parser.add_argument("--snr-tau1", type=float, default=1.0)
    parser.add_argument("--snr-tau2", type=float, default=2.0)
    parser.add_argument("--snr-smooth-temperature", type=float, default=0.0)
    parser.add_argument("--snr-cv-gate", type=float, default=0.50)
    parser.add_argument("--snr-overhead-gate", type=float, default=0.20)
    parser.add_argument("--snr-active-min", type=float, default=0.05)
    parser.add_argument("--snr-active-max", type=float, default=0.80)
    parser.add_argument("--run-p2-one-step", action="store_true")
    parser.add_argument("--p2-candidates", default="LQ0,LQ1")
    parser.add_argument("--p2-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--p2-seeds", default="0,1,2")
    parser.add_argument("--p2-steps", type=int, default=8)
    parser.add_argument("--p2-train-size", type=int, default=9984)
    parser.add_argument("--p2-microbatch-size", type=int, default=64)
    parser.add_argument("--p2-functional-step-fraction", type=float, default=0.10)
    parser.add_argument("--p2-snr-warmup-steps", type=int, default=36)
    parser.add_argument(
        "--p2-functional-modes",
        default="F1-SNRGatedTaskProjected,C1-GeometryOnlyNoSNRTaskProjected,C0-NoOp,C2-RandomMatchedNormTaskProjected",
    )
    parser.add_argument("--p2-corr-gate", type=float, default=0.30)
    parser.add_argument("--p2-bad-step-gate", type=float, default=0.05)
    parser.add_argument("--p2-nonharm-gate", type=float, default=0.70)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    device = _device_from_arg(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    prev_dir = Path(args.previous_v927_dir)
    if not prev_dir.is_absolute():
        prev_dir = ROOT / prev_dir
    p0_prev = _load_p0_previous(prev_dir)
    prev_route = p0_prev["route"]

    write_json(
        out_dir / "run_manifest.json",
        {
            "created_utc": _now_iso(),
            "script": str(SCRIPT_PATH.relative_to(ROOT)),
            "plan": str(PLAN_PATH.relative_to(ROOT)),
            "device": str(device),
            "torch": torch.__version__,
            "args": vars(args),
            "source_artifacts": {"v927": p0_prev["source_artifact"]},
            "contract": {
                "loss_type": "CE",
                "label_smoothing": 0,
                "external_teacher_used": 0,
                "self_teacher_used": 0,
                "teacher_logits_used": 0,
                "distillation_used": 0,
                "geometry_loss_used": 0,
                "sampler_changed": 0,
                "class_weight_used": 0,
                "cpu_offload_used": 0,
                "uses_loss_backward": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
            },
        },
    )

    p0_purekan = int(_to_int(prev_route.get("success_v927_purekan_equivalence", prev_route.get("purekan_equivalence_pass", 0))) == 1)
    p0_p4 = int(_to_int(prev_route.get("success_v927_p4", prev_route.get("p4_pass", 0))) == 1)
    p0_near = int(_to_int(prev_route.get("success_v927_p5_nearpass", prev_route.get("p5_near_pass", 0))) == 1)
    p0_full = int(_to_int(prev_route.get("success_v927_p5_fullpass", prev_route.get("p5_pass", 0))) == 1)
    p0_contract = 1
    p0_can_open = int(p0_purekan and p0_p4 and p0_near and p0_contract)

    contract_rows = [
        {
            "stage": "P0_CONTRACT",
            "candidate_id": "LQ-t2-h256",
            "candidate_family": "LinearLiftQuadraticEdgeBasis",
            "source_artifact": p0_prev["source_artifact"],
            "purekan_equivalence_pass": p0_purekan,
            "p4_pass": p0_p4,
            "p5_near_pass": p0_near,
            "p5_full_pass": p0_full,
            "loss_type": "CE",
            "label_smoothing": 0,
            "teacher_used": 0,
            "external_teacher_used": 0,
            "self_teacher_used": 0,
            "teacher_logits_used": 0,
            "distillation_used": 0,
            "geometry_loss_used": 0,
            "uses_loss_backward": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
            "purekanconv_status": "deferred_until_FC_PureKAN_P5_fullpass_or_user_unlock",
            "purekanformer_status": "deferred_until_FC_PureKAN_P5_fullpass_or_user_unlock",
        }
    ]
    open_rows = [
        {
            "stage": "P0_FUNCTIONAL_OPEN_GATE",
            "candidate_id": "LQ-t2-h256",
            "source_artifact": p0_prev["source_artifact"],
            "functional_diagnostic_can_open": p0_can_open,
            "functional_update_used": 0,
            "reason": "P0_gate_passed_for_gated_diagnostic" if p0_can_open else "P0_gate_failed",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    ]
    write_csv_rows(out_dir / "contract_audit_v929.csv", contract_rows)
    write_csv_rows(out_dir / "functional_open_gate_v929.csv", open_rows)

    _write_svg(out_dir / "figures" / "p0_functional_open_gate_dashboard.svg", "P0 Functional Open Gate", f"open={p0_can_open}, source={p0_prev['source_artifact']}")
    _write_svg(out_dir / "figures" / "p0_contract_heatmap.svg", "P0 Contract Heatmap", "CE-only, no teacher, no geometry loss, no fake/proxy rows")
    _write_svg(out_dir / "figures" / "p0_route_position_diagram.svg", "P0 Route Position", "LQ near-pass base; functional diagnostics gated by P1")

    if p0_can_open:
        p1_rows = _run_p1_snr(args, device)
    else:
        p1_rows = [_not_run("P1_SNR_INSTRUMENTATION", "snr_instrumentation_v929.csv", "P0_functional_open_gate_failed")]
    write_csv_rows(out_dir / "snr_instrumentation_v929.csv", p1_rows)

    measured_p1_all = [r for r in p1_rows if str(r.get("status", "")) != "not_run"]
    measured_p1 = [r for r in measured_p1_all if _to_int(r.get("eligible_for_gate", 1), 1) == 1]
    p1_stable = bool(measured_p1) and all(_to_int(r.get("estimator_stability_pass")) == 1 for r in measured_p1)
    p1_overhead = bool(measured_p1) and all(_to_int(r.get("SNR_overhead_pass")) == 1 for r in measured_p1)
    p1_active = bool(measured_p1) and all(_to_int(r.get("active_fraction_sanity_pass")) == 1 for r in measured_p1)
    p1_pass = int(p1_stable and p1_overhead and p1_active)
    max_overhead = max((_to_float(r.get("SNR_overhead_ratio")) for r in measured_p1), default=0.0)
    mean_overhead = sum(_to_float(r.get("SNR_overhead_ratio")) for r in measured_p1) / max(1, len(measured_p1))
    min_active = min((_to_float(r.get("SNR_active_fraction_tau1")) for r in measured_p1), default=0.0)
    max_active = max((_to_float(r.get("SNR_active_fraction_tau1")) for r in measured_p1), default=0.0)
    max_cv = max((_to_float(r.get("SNR_role_cv")) for r in measured_p1), default=0.0)

    _write_svg(out_dir / "figures" / "p1_snr_by_role_violin.svg", "P1 SNR By Role", f"max_cv={max_cv:.4f}, measured_rows={len(measured_p1)}")
    _write_svg(out_dir / "figures" / "p1_snr_active_fraction_by_dataset.svg", "P1 Active Fraction", f"min={min_active:.4f}, max={max_active:.4f}")
    _write_svg(out_dir / "figures" / "p1_snr_overhead_bar.svg", "P1 SNR Overhead", f"mean={mean_overhead:.4f}, max={max_overhead:.4f}, gate=0.20")
    _write_svg(out_dir / "figures" / "p1_snr_vs_margin_scatter.svg", "P1 SNR vs Margin", "margin scatter not opened before P2; see SNR rows")
    _write_svg(out_dir / "figures" / "p1_basis_channel_snr_heatmap.svg", "P1 Basis Channel SNR", "role-level SNR measured for lift/output/quadratic")

    if p1_pass and bool(args.run_p2_one_step):
        p2_rows = _run_p2_one_step(args, device)
    elif p1_pass:
        p2_rows = [_not_run("P2_ONE_STEP_POPULATION_RISK_AUDIT", "one_step_population_risk_audit_v929.csv", "P1_passed_but_P2_not_requested")]
    else:
        p2_rows = [_not_run("P2_ONE_STEP_POPULATION_RISK_AUDIT", "one_step_population_risk_audit_v929.csv", "P1_snr_instrumentation_failed_so_functional_update_not_opened")]
    write_csv_rows(out_dir / "one_step_population_risk_audit_v929.csv", p2_rows)

    measured_p2 = [r for r in p2_rows if str(r.get("status", "")) != "not_run"]
    snr_p2 = [r for r in measured_p2 if str(r.get("functional_mode")) == "F1-SNRGatedTaskProjected"]
    geo_p2 = [r for r in measured_p2 if str(r.get("functional_mode")) == "C1-GeometryOnlyNoSNRTaskProjected"]
    p2_corr = _pearson(
        [_to_float(r.get("predicted_population_improvement")) for r in snr_p2],
        [_to_float(r.get("actual_holdout_delta")) for r in snr_p2],
    )
    p2_bad = sum(_to_int(r.get("bad_step")) for r in snr_p2) / max(1, len(snr_p2))
    p2_nonharm = sum(_to_int(r.get("holdout_nonharm")) for r in snr_p2) / max(1, len(snr_p2))
    snr_delta_mean = sum(_to_float(r.get("actual_holdout_delta")) for r in snr_p2) / max(1, len(snr_p2))
    geo_delta_mean = sum(_to_float(r.get("actual_holdout_delta")) for r in geo_p2) / max(1, len(geo_p2))
    geo_bad = sum(_to_int(r.get("bad_step")) for r in geo_p2) / max(1, len(geo_p2))
    p2_corr_pass = bool(snr_p2) and p2_corr >= float(args.p2_corr_gate)
    p2_bad_pass = bool(snr_p2) and p2_bad <= float(args.p2_bad_step_gate)
    p2_nonharm_pass = bool(snr_p2) and p2_nonharm >= float(args.p2_nonharm_gate)
    p2_useful = bool(snr_p2 and geo_p2) and (p2_bad < geo_bad or snr_delta_mean < geo_delta_mean)
    p2_pass = int(p2_corr_pass and p2_bad_pass and p2_nonharm_pass and p2_useful)
    _write_svg(out_dir / "figures" / "p2_predicted_vs_holdout_delta_scatter.svg", "P2 Predicted vs Holdout", f"corr={p2_corr:.4f}, rows={len(snr_p2)}")
    _write_svg(out_dir / "figures" / "p2_bad_step_rate_by_mode.svg", "P2 Bad Step Rate", f"SNR={p2_bad:.4f}, noSNR={geo_bad:.4f}")
    _write_svg(out_dir / "figures" / "p2_holdout_delta_by_mode.svg", "P2 Holdout Delta", f"SNR={snr_delta_mean:.6f}, noSNR={geo_delta_mean:.6f}")

    if p2_pass:
        downstream_reason = "P2_passed_but_P3_to_P6_not_enabled_in_this_terminal_run"
    elif measured_p2:
        downstream_reason = "P2_one_step_population_risk_audit_failed_so_P3_P6_not_opened"
    elif p1_pass:
        downstream_reason = "P1_passed_but_P2_not_requested"
    else:
        downstream_reason = "P1_snr_instrumentation_failed_so_functional_update_not_opened"
    downstream_specs = [
        ("P3_SHORT_RUN_FUNCTIONAL_SAFETY", "short_run_functional_safety_v929.csv"),
        ("P4_FULL_FUNCTIONAL_REENTRY_10SEED", "full_functional_reentry_10seed_v929.csv"),
        ("P5_STRONG_BASELINE_CHALLENGE", "strong_baseline_challenge_v929.csv"),
        ("P6_ROBUSTNESS_NOISE_DIAGNOSTIC", "robustness_noise_diagnostic_v929.csv"),
        ("FUNCTIONAL_EVENT_TRACE", "functional_event_trace_v929.csv"),
    ]
    for stage, name in downstream_specs:
        write_csv_rows(out_dir / name, [_not_run(stage, name, downstream_reason)])

    if not p0_can_open:
        route = "R3-FunctionalUnsafe"
        primary_blocker = "P0_functional_open_gate_failed"
        next_required = "restore_LQ_P4_and_P5_nearpass_before_functional"
        failure_code = "F3_functional_open_gate_fail"
    elif not p1_overhead:
        route = "R6-FunctionalSystemOverheadFail"
        primary_blocker = "P1_snr_compute_overhead_exceeds_20_percent_step_gate"
        next_required = "implement_low_overhead_running_or_subsampled_SNR_estimator_before_P2"
        failure_code = "F5_snr_overhead_fail"
    elif not p1_active or not p1_stable:
        route = "R4-SNRGateNotUseful"
        primary_blocker = "P1_snr_active_fraction_or_stability_gate_failed"
        next_required = "calibrate_tau_temperature_or_rolewise_SNR_before_functional_update"
        failure_code = "F4_snr_estimator_unstable"
    elif not bool(args.run_p2_one_step):
        route = "R4-SNRGateNotUseful"
        primary_blocker = "P2_not_requested_after_P1_pass"
        next_required = "run_P2_one_step_population_risk_audit"
        failure_code = "F16_artifact_missing"
    elif measured_p2 and not p2_corr_pass:
        route = "R4-SNRGateNotUseful"
        primary_blocker = "P2_predicted_population_improvement_did_not_correlate_with_holdout_delta"
        next_required = "repair_functional_predictor_or_SNR_gate_before_short_run"
        failure_code = "F7_predicted_holdout_corr_fail"
    elif measured_p2 and (not p2_bad_pass or not p2_nonharm_pass):
        route = "R3-FunctionalUnsafe"
        primary_blocker = "P2_one_step_functional_update_harmed_holdout_too_often"
        next_required = "strengthen_task_projection_or_reduce_functional_step_before_P3"
        failure_code = "F6_bad_step_rate_fail"
    elif measured_p2 and not p2_useful:
        route = "R4-SNRGateNotUseful"
        primary_blocker = "P2_snr_gated_functional_did_not_beat_noSNR_geometry_control"
        next_required = "redesign_SNR_gate_or_role_mapping_before_P3"
        failure_code = "F8_snr_usefulness_fail"
    else:
        route = "R4-SNRGateNotUseful"
        primary_blocker = "P2_passed_but_P3_to_P6_not_enabled_in_this_runner"
        next_required = "open_P3_short_run_functional_safety"
        failure_code = "F16_artifact_missing"

    failure_rows = [
        {
            "stage": "P2" if measured_p2 else "P1",
            "failure_code": failure_code,
            "candidate_id": "LQ0/LQ1",
            "route": route,
            "reason": primary_blocker,
            "max_snr_overhead_ratio": max_overhead,
            "mean_snr_overhead_ratio": mean_overhead,
            "min_active_fraction_tau1": min_active,
            "max_active_fraction_tau1": max_active,
            "max_snr_role_cv": max_cv,
            "p2_prediction_corr": p2_corr,
            "p2_bad_step_rate": p2_bad,
            "p2_holdout_nonharm_rate": p2_nonharm,
            "p2_snr_delta_mean": snr_delta_mean,
            "p2_noSNR_delta_mean": geo_delta_mean,
            "p2_snr_usefulness_pass": int(p2_useful),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    ]
    write_csv_rows(out_dir / "failure_table.csv", failure_rows)

    route_json = {
        "route": route,
        "best_functional_candidate": "",
        "base_candidate": "LQ-t2-h256",
        "p4_pass": p0_p4,
        "p5_near_pass": p0_near,
        "p5_full_pass": p0_full,
        "p1_snr_row_count": len(measured_p1),
        "p1_snr_stability_pass": int(p1_stable),
        "p1_snr_overhead_pass": int(p1_overhead),
        "p1_snr_active_fraction_pass": int(p1_active),
        "p1_snr_pass": p1_pass,
        "max_snr_overhead_ratio": max_overhead,
        "mean_snr_overhead_ratio": mean_overhead,
        "min_active_fraction_tau1": min_active,
        "max_active_fraction_tau1": max_active,
        "max_snr_role_cv": max_cv,
        "p2_one_step_row_count": len(measured_p2),
        "p2_snr_row_count": len(snr_p2),
        "p2_prediction_corr": p2_corr,
        "p2_prediction_corr_pass": int(p2_corr_pass),
        "p2_bad_step_rate": p2_bad,
        "p2_bad_step_pass": int(p2_bad_pass),
        "p2_holdout_nonharm_rate": p2_nonharm,
        "p2_holdout_nonharm_pass": int(p2_nonharm_pass),
        "p2_snr_delta_mean": snr_delta_mean,
        "p2_noSNR_delta_mean": geo_delta_mean,
        "p2_noSNR_bad_step_rate": geo_bad,
        "p2_snr_gate_usefulness_pass": int(p2_useful),
        "p2_one_step_population_risk_pass": p2_pass,
        "functional_task_safe": 0,
        "functional_geometry_pass": 0,
        "functional_calibration_pass": 0,
        "functional_kmnist_repair_pass": 0,
        "functional_control_pass": 0,
        "functional_system_pass": 0,
        "quadratic_baseline_challenge_pass": 0,
        "noise_robustness_pass": 0,
        "external_fair_ready": 0,
        "primary_blocker": primary_blocker,
        "next_required_implementation": next_required,
        "success_v929_p1_snr_gate": p1_pass,
        "success_v929_functional_opened": 0,
        "success_v929_functional_advantage": 0,
        "success_v929_external_ready": 0,
    }
    write_json(out_dir / "route_decision.json", route_json)
    write_json(out_dir / "aggregate_decision.json", route_json)

    audit_targets = [
        out_dir / "contract_audit_v929.csv",
        out_dir / "functional_open_gate_v929.csv",
        out_dir / "snr_instrumentation_v929.csv",
        out_dir / "one_step_population_risk_audit_v929.csv",
        out_dir / "short_run_functional_safety_v929.csv",
        out_dir / "full_functional_reentry_10seed_v929.csv",
        out_dir / "strong_baseline_challenge_v929.csv",
        out_dir / "robustness_noise_diagnostic_v929.csv",
        out_dir / "functional_event_trace_v929.csv",
        out_dir / "failure_table.csv",
    ]
    audit = audit_no_fake(audit_targets)
    write_csv_rows(
        out_dir / "v929_provenance_audit.csv",
        [
            {
                "stage": "NO_FAKE_AUDIT",
                "route": route,
                **audit,
                "fake_data_used": int(audit["fake_data_used"]),
                "proxy_row_used": int(audit["proxy_row_used"]),
                "cpu_offload_used": int(audit["cpu_offload_used"]),
            }
        ],
    )
    route_json.update({"no_fake": bool(audit["no_fake"]), "no_proxy": bool(audit["no_proxy"]), "rows_checked": int(audit["rows_checked"])})
    write_json(out_dir / "route_decision.json", route_json)
    write_json(out_dir / "aggregate_decision.json", route_json)
    hash_rows = artifact_hash_rows(
        [
            PLAN_PATH,
            SCRIPT_PATH,
            ROOT / "dgkan" / "functional" / "snr_gated_lq.py",
            ROOT / "dgkan" / "models" / "fc_purekan_lq.py",
            out_dir / "route_decision.json",
            out_dir / "contract_audit_v929.csv",
            out_dir / "functional_open_gate_v929.csv",
            out_dir / "snr_instrumentation_v929.csv",
            out_dir / "one_step_population_risk_audit_v929.csv",
            out_dir / "short_run_functional_safety_v929.csv",
            out_dir / "full_functional_reentry_10seed_v929.csv",
            out_dir / "strong_baseline_challenge_v929.csv",
            out_dir / "robustness_noise_diagnostic_v929.csv",
            out_dir / "functional_event_trace_v929.csv",
            out_dir / "failure_table.csv",
            out_dir / "v929_provenance_audit.csv",
        ],
        root=ROOT,
    )
    write_csv_rows(out_dir / "artifact_hashes.csv", hash_rows)
    print(json.dumps(route_json, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
