#!/usr/bin/env python3
"""DG-KAN v9.2.11 functional causality controller runner.

The runner keeps reusable model and direction math in dgkan.functional /
dgkan.models modules.  This file orchestrates artifacts and gate decisions for
P0-P2 first-wave causality replay, then opens later stages only when gates pass.
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
from typing import Any, Dict, List, Sequence, Tuple

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
from dgkan.functional import lq_functional_predictor as fp_lq  # noqa: E402
from dgkan.functional import snr_gated_lq as snr_lq  # noqa: E402
from dgkan.models import fc_purekan_lq as lq  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.11_Functional_Causality_Controller_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9211_functional_causality_controller.py"
PREV_V9210 = ROOT / "results" / "real_rerun_20260506" / "v9210_functional_predictor_repair_p3p4_20260509T210000Z"


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


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _write_svg(path: Path, title: str, subtitle: str) -> None:
    ensure_dir(path.parent)
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="980" height="190" viewBox="0 0 980 190">
  <rect width="980" height="190" fill="#f8fafc"/>
  <text x="28" y="54" font-family="Arial, sans-serif" font-size="25" fill="#111827">{title}</text>
  <text x="28" y="96" font-family="Arial, sans-serif" font-size="16" fill="#374151">{subtitle}</text>
  <text x="28" y="136" font-family="Arial, sans-serif" font-size="13" fill="#6b7280">Generated from measured CSV/JSON fields; no inferred pass values.</text>
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
            raise ValueError(f"unknown candidate alias {name}")
    return specs


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


def _quadratic_active_fraction(rows: Sequence[Dict[str, Any]]) -> float:
    for row in rows:
        role = str(row.get("role", ""))
        if "quadratic" in role or "legendre_p2" in role:
            return _to_float(row.get("SNR_active_fraction_tau1"), 0.0)
    return _to_float(rows[-1].get("SNR_active_fraction_tau1"), 0.0) if rows else 0.0


def _quadratic_snr(rows: Sequence[Dict[str, Any]]) -> float:
    for row in rows:
        role = str(row.get("role", ""))
        if "quadratic" in role or "legendre_p2" in role:
            return _to_float(row.get("SNR_role"), 0.0)
    return _to_float(rows[-1].get("SNR_role"), 0.0) if rows else 0.0


def _gate_multiplier(gate_id: str, active_fraction: float, quadratic_snr: float, high_tau: float) -> float:
    if gate_id == "G1-RoleSNR":
        return float(active_fraction)
    if gate_id == "G2-HighConfidenceRoleSNR":
        return float(active_fraction) if float(quadratic_snr) >= float(high_tau) else 0.0
    if gate_id == "G4-PredictedNonharm":
        return float(active_fraction)
    raise ValueError(f"unknown gate_id {gate_id}")


def _make_delta(
    *,
    direction_id: str,
    branch_id: str,
    params: Sequence[torch.Tensor],
    grads: Sequence[torch.Tensor],
    basis: str,
    task_step: Sequence[torch.Tensor],
    step_fraction: float,
    gate_multiplier: float,
    device: torch.device,
    seed: int,
) -> List[torch.Tensor]:
    if branch_id in {"AdamWOnly", "C0-NoOpMatchedOverhead"}:
        return snr_lq.zero_like_params(params)
    if branch_id == "C1-RandomMatchedNorm":
        raw = fp_lq.random_matched_direction(params, task_step, seed=seed, device=device)
        return snr_lq.scale_step(raw, float(step_fraction) * float(gate_multiplier))
    if branch_id == "C4-GeometryD1Control":
        raw = fp_lq.direction_for_id("D1-QuadraticCoeffDamping", params, grads, basis)
    else:
        raw = fp_lq.direction_for_id(direction_id, params, grads, basis)
    scaled = snr_lq.scale_direction_to_fraction_of_task_step(raw, task_step, float(step_fraction))
    return snr_lq.scale_step(scaled, float(gate_multiplier))


def _clone_params_states(params: Sequence[torch.Tensor]) -> Tuple[List[torch.Tensor], List[AdamWState]]:
    cloned = [p.detach().clone() for p in params]
    return cloned, [AdamWState.zeros_like(p) for p in cloned]


def _copy_states(states: Sequence[AdamWState]) -> List[AdamWState]:
    return [AdamWState(step=s.step, m=s.m.detach().clone(), v=s.v.detach().clone()) for s in states]


def _apply_in_place(params: Sequence[torch.Tensor], step: Sequence[torch.Tensor]) -> None:
    with torch.no_grad():
        for p, d in zip(params, step):
            p.add_(d)


def _p0_source_recap(prev_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    route = _read_json(prev_dir / "route_decision.json")
    p2 = _read_csv(prev_dir / "p2_direction_factory_summary.csv")
    p4 = _read_csv(prev_dir / "p4_short_run_functional_safety.csv")
    d9 = [
        r
        for r in p2
        if r.get("direction_id") == "D9-SignalChannelProjection"
        and r.get("gate_id") == "G1-RoleSNROnly"
        and abs(_to_float(r.get("step_fraction")) - 0.01) < 1.0e-12
    ]
    best_rows = [r for r in p4 if r.get("mode_id") == "BestFunctional"]
    step_ratios = [_to_float(r.get("step_time_ratio_vs_adamw")) for r in best_rows]
    p0 = [
        {
            "stage": "P0_V9210_REPRODUCTION_SOURCE_RECAP",
            "source_artifact": str(prev_dir.relative_to(ROOT)) if prev_dir.exists() else str(prev_dir),
            "source_route": route.get("route", ""),
            "source_best_direction": route.get("best_direction", ""),
            "source_best_gate": route.get("best_gate", ""),
            "source_best_step_fraction": route.get("best_step_fraction", ""),
            "d9_bad_step_rate": _to_float(d9[0].get("bad_step_rate_accepted")) if d9 else "",
            "d9_nonharm": _to_float(d9[0].get("holdout_nonharm_fraction_accepted")) if d9 else "",
            "d9_prediction_corr": _to_float(d9[0].get("prediction_corr_all")) if d9 else "",
            "d9_overhead": _to_float(d9[0].get("amortized_overhead_ratio")) if d9 else "",
            "p4_best_step_ratio_mean": sum(step_ratios) / max(1, len(step_ratios)),
            "p4_best_step_ratio_q90": sorted(step_ratios)[min(len(step_ratios) - 1, int(0.90 * len(step_ratios)))] if step_ratios else "",
            "p0_reproduction_pass": int(bool(d9) and abs(_to_float(d9[0].get("bad_step_rate_accepted")) - 0.027778) <= 0.05 and abs(_to_float(d9[0].get("prediction_corr_all")) - 0.649843) <= 0.15),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    ]
    return p0, p4


def _p1_failure_attribution(p4_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for horizon in sorted({r.get("horizon_steps", "") for r in p4_rows if r.get("horizon_steps", "")}):
        best = [r for r in p4_rows if r.get("mode_id") == "BestFunctional" and r.get("horizon_steps") == horizon]
        controls = [r for r in p4_rows if r.get("mode_id") in {"GeometryD1Control", "RandomMatchedControl"} and r.get("horizon_steps") == horizon]
        if not best:
            continue
        mean = lambda rs, k: sum(_to_float(r.get(k)) for r in rs) / max(1, len(rs))
        best_step = mean(best, "step_time_ratio_vs_adamw")
        best_ce = mean(best, "CEp99_delta_vs_adamw")
        best_margin = mean(best, "margin_p10_delta_vs_adamw")
        best_curv = mean(best, "curvature_delta_vs_adamw")
        control_ce = min([mean([r for r in controls if r.get("mode_id") == m], "CEp99_delta_vs_adamw") for m in {"GeometryD1Control", "RandomMatchedControl"}] or [0.0])
        control_margin = max([mean([r for r in controls if r.get("mode_id") == m], "margin_p10_delta_vs_adamw") for m in {"GeometryD1Control", "RandomMatchedControl"}] or [0.0])
        attribution = []
        if best_step > 1.20:
            attribution.append("F1-overhead_dominant")
        if abs(best_ce) < 0.02 and abs(best_margin) < 0.02 and abs(best_curv) < 1.0e-4:
            attribution.append("F2-mechanism_weak")
        if best_ce >= control_ce and best_margin <= control_margin:
            attribution.append("F3-control_equivalent")
        out.append(
            {
                "stage": "P1_P4_FAILURE_ATTRIBUTION",
                "horizon_steps": horizon,
                "best_functional_step_ratio_mean": best_step,
                "best_functional_step_ratio_q90": sorted([_to_float(r.get("step_time_ratio_vs_adamw")) for r in best])[min(len(best) - 1, int(0.90 * len(best)))],
                "best_functional_CEp99_delta": best_ce,
                "best_control_CEp99_delta": control_ce,
                "best_functional_margin_delta": best_margin,
                "best_control_margin_delta": control_margin,
                "best_functional_curvature_delta": best_curv,
                "cos_with_adamw_step": "not_measured_in_v9210_p4_source",
                "primary_factors": ",".join(attribution) if attribution else "unattributed",
                "p1_attribution_pass": int(bool(attribution)),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    return out or [_not_run("P1_P4_FAILURE_ATTRIBUTION", "p1_p4_failure_attribution.csv", "v9210_p4_rows_missing")]


def _advance_adamw(
    params: Sequence[torch.Tensor],
    states: Sequence[AdamWState],
    bwd,
    x: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    opt_cfg: ManualAdamWConfig,
) -> List[torch.Tensor]:
    pack = bwd(x, y, *params, mu, std, 2.0, 2.0)
    grads = [g.detach() for g in pack[1:]]
    v92._adamw_update_foreach_(params, grads, states, opt_cfg)
    return grads


def _run_paired_event_replay(args: argparse.Namespace, device: torch.device) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    specs = _candidate_specs(args.candidates)
    horizons = [int(x) for x in _parse_list(args.p2_horizons)]
    max_horizon = max(horizons)
    directions = _parse_list(args.p2_directions)
    events = _parse_list(args.p2_event_types)
    branches = ["AdamWOnly", "RealFunctional", "C0-NoOpMatchedOverhead", "C1-RandomMatchedNorm", "C4-GeometryD1Control"]
    snr_cfg = snr_lq.SNRConfig(tau1=float(args.snr_tau1), tau2=float(args.snr_tau2), temperature=float(args.snr_smooth_temperature))
    for dataset in _canonical_tasks(args.datasets):
        min_train = max(int(args.train_size), int(args.batch_size) * (int(args.snr_warmup_steps) + int(args.p2_event_steps) + max_horizon + 4))
        x_train, y_train, x_test, y_test, input_dim, output_dim, protocol = v92._load_task(args, dataset, train_size=min_train, test_size=int(args.p2_eval_size))
        x_train = x_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device)
        x_eval = x_test.to(device=device, dtype=torch.float32)
        y_eval = y_test.to(device=device)
        for seed_text in _parse_list(args.seeds):
            seed = int(seed_text)
            for spec in specs:
                params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, seed + 92920)
                states = [AdamWState.zeros_like(p) for p in params]
                opt_cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
                scalar_state = snr_lq.ScalarRoleSNRState.zeros(len(snr_lq.role_names_for_basis(spec.basis)), beta=float(args.snr_ema_beta), device=device)
                _fwd, bwd = lq.functions_for_basis(spec.basis)
                for warm_step in range(int(args.snr_warmup_steps)):
                    gen = torch.Generator(device=device).manual_seed(9291000 + seed * 10000 + warm_step)
                    idx = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)[: int(args.batch_size)]
                    grads = _advance_adamw(params, states, bwd, x_train[idx], y_train[idx], mu, std, opt_cfg)
                    snr_lq.scalar_ema_snr_from_grads(grads=grads, scalar_state=scalar_state, basis=spec.basis, snr_cfg=snr_cfg)
                for event_step in range(int(args.p2_event_steps)):
                    gen = torch.Generator(device=device).manual_seed(9511000 + seed * 10000 + event_step)
                    perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)
                    event_idx = perm[: int(args.batch_size)]
                    future_batches = []
                    for hstep in range(1, max_horizon):
                        fgen = torch.Generator(device=device).manual_seed(9512000 + seed * 10000 + event_step * 100 + hstep)
                        future_batches.append(torch.randperm(int(x_train.shape[0]), device=device, generator=fgen)[: int(args.batch_size)])
                    xb = x_train[event_idx]
                    yb = y_train[event_idx]
                    pack = bwd(xb, yb, *params, mu, std, 2.0, 2.0)
                    grads = [g.detach() for g in pack[1:]]
                    snr_rows = snr_lq.scalar_ema_snr_from_grads(grads=grads, scalar_state=scalar_state, basis=spec.basis, snr_cfg=snr_cfg)
                    active_fraction = _quadratic_active_fraction(snr_rows)
                    quadratic_snr = _quadratic_snr(snr_rows)
                    before_metrics = _eval_lq_metrics(params, mu, std, spec.basis, x_eval, y_eval)
                    before_mech = fp_lq.mechanism_metrics(params, mu, std, spec.basis, x_eval[: min(256, int(x_eval.shape[0]))])
                    task_step = snr_lq.gradient_descent_task_step(grads, float(args.lr))
                    for direction_id in directions:
                        for event_type in events:
                            accepts = fp_lq.event_accepts(
                                event_type,
                                metrics=before_metrics,
                                mechanism=before_mech,
                                quadratic_snr=quadratic_snr,
                                step=event_step,
                                stride=int(args.event_stride),
                                cep99_tau=float(args.event_cep99_tau),
                                margin_tau=float(args.event_margin_tau),
                                curvature_tau=float(args.event_curvature_tau),
                                entropy_tau=float(args.event_entropy_tau),
                                snr_tau=float(args.high_snr_tau),
                            )
                            branch_results: Dict[str, Dict[int, Dict[str, Any]]] = {}
                            for branch_id in branches:
                                bparams, bstates = _clone_params_states(params)
                                bstates = _copy_states(states)
                                multiplier = _gate_multiplier("G1-RoleSNR", active_fraction, quadratic_snr, float(args.high_snr_tau)) if accepts else 0.0
                                raw_delta = _make_delta(
                                    direction_id=direction_id,
                                    branch_id=branch_id,
                                    params=bparams,
                                    grads=grads,
                                    basis=spec.basis,
                                    task_step=task_step,
                                    step_fraction=float(args.step_fraction),
                                    gate_multiplier=multiplier,
                                    device=device,
                                    seed=9611000 + seed * 10000 + event_step,
                                )
                                projected_delta, removed_norm = snr_lq.project_step_to_task_safe(raw_delta, grads)
                                event_accepted = int(accepts and branch_id not in {"AdamWOnly", "C0-NoOpMatchedOverhead"} and snr_lq.step_norm(projected_delta).detach().cpu() > 0)
                                if device.type == "cuda":
                                    torch.cuda.reset_peak_memory_stats(device)
                                _sync(device)
                                t0 = time.perf_counter()
                                bgrads = _advance_adamw(bparams, bstates, bwd, xb, yb, mu, std, opt_cfg)
                                if event_accepted:
                                    _apply_in_place(bparams, projected_delta)
                                branch_results[branch_id] = {}
                                for horizon in horizons:
                                    if horizon == 1:
                                        eval_params = [p.detach().clone() for p in bparams]
                                        eval_states = _copy_states(bstates)
                                    else:
                                        eval_params = [p.detach().clone() for p in bparams]
                                        eval_states = _copy_states(bstates)
                                        for fidx in future_batches[: horizon - 1]:
                                            _advance_adamw(eval_params, eval_states, bwd, x_train[fidx], y_train[fidx], mu, std, opt_cfg)
                                    metrics = _eval_lq_metrics(eval_params, mu, std, spec.basis, x_eval, y_eval)
                                    mech = fp_lq.mechanism_metrics(eval_params, mu, std, spec.basis, x_eval[: min(256, int(x_eval.shape[0]))])
                                    branch_results[branch_id][horizon] = {"metrics": metrics, "mech": mech}
                                _sync(device)
                                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                                peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == "cuda" else 0.0
                                for horizon in horizons:
                                    am = branch_results["AdamWOnly"][horizon]["metrics"] if "AdamWOnly" in branch_results else before_metrics
                                    ach = branch_results["AdamWOnly"][horizon]["mech"] if "AdamWOnly" in branch_results else before_mech
                                    bm = branch_results[branch_id][horizon]["metrics"]
                                    bmech = branch_results[branch_id][horizon]["mech"]
                                    rows.append(
                                        {
                                            "stage": "P2_PAIRED_EVENT_REPLAY_CAUSALITY",
                                            "candidate_id": spec.candidate_id,
                                            "dataset": dataset,
                                            "seed": seed,
                                            "protocol": protocol,
                                            "event_id": f"{dataset}-s{seed}-{spec.candidate_id}-e{event_step}-{direction_id}-{event_type}",
                                            "event_step": event_step,
                                            "event_type": event_type,
                                            "direction_id": direction_id,
                                            "gate_id": "G1-RoleSNR",
                                            "branch": branch_id,
                                            "horizon": horizon,
                                            "event_triggered": int(accepts),
                                            "event_accepted": event_accepted,
                                            "active_fraction": active_fraction,
                                            "quadratic_snr": quadratic_snr,
                                            "functional_norm": float(snr_lq.step_norm(projected_delta).detach().cpu()),
                                            "projection_removed_norm": float(removed_norm.detach().cpu()),
                                            "cos_with_task_gradient": float((snr_lq.step_dot(grads, projected_delta) / (snr_lq.step_norm(grads).clamp_min(1.0e-12) * snr_lq.step_norm(projected_delta).clamp_min(1.0e-12))).detach().cpu()) if event_accepted else 0.0,
                                            "acc": bm["acc"],
                                            "adamw_acc": am["acc"],
                                            "acc_delta_vs_adamw": bm["acc"] - am["acc"],
                                            "loss": bm["loss"],
                                            "adamw_loss": am["loss"],
                                            "holdout_loss_delta_vs_adamw": bm["loss"] - am["loss"],
                                            "CEp99": bm["CE_p99"],
                                            "adamw_CEp99": am["CE_p99"],
                                            "CEp99_delta_vs_adamw": bm["CE_p99"] - am["CE_p99"],
                                            "margin_p10": bm["correct_margin_p10"],
                                            "adamw_margin_p10": am["correct_margin_p10"],
                                            "margin_p10_delta_vs_adamw": bm["correct_margin_p10"] - am["correct_margin_p10"],
                                            "curvature_proxy": bmech["curvature_proxy"],
                                            "adamw_curvature_proxy": ach["curvature_proxy"],
                                            "curvature_delta_vs_adamw": bmech["curvature_proxy"] - ach["curvature_proxy"],
                                            "local_lipschitz_delta_vs_adamw": bmech["local_lipschitz_proxy"] - ach["local_lipschitz_proxy"],
                                            "basis_entropy_delta_vs_adamw": bmech["basis_usage_entropy"] - ach["basis_usage_entropy"],
                                            "step_time_ms": elapsed_ms,
                                            "peak_memory_mb": peak_mb,
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
                                            "functional_update_used": int(branch_id not in {"AdamWOnly", "C0-NoOpMatchedOverhead"} and event_accepted),
                                            "fake_data_used": 0,
                                            "proxy_row_used": 0,
                                            "cpu_offload_used": 0,
                                        }
                                    )
                                trace_rows.append(
                                    {
                                        "stage": "PAIRED_REPLAY_BRANCH_TRACE",
                                        "event_id": f"{dataset}-s{seed}-{spec.candidate_id}-e{event_step}-{direction_id}-{event_type}",
                                        "branch": branch_id,
                                        "event_triggered": int(accepts),
                                        "event_accepted": event_accepted,
                                        "elapsed_ms": elapsed_ms,
                                        "peak_memory_mb": peak_mb,
                                        "fake_data_used": 0,
                                        "proxy_row_used": 0,
                                        "cpu_offload_used": 0,
                                    }
                                )
                    # Advance the real base one AdamW step after auditing this event checkpoint.
                    v92._adamw_update_foreach_(params, grads, states, opt_cfg)
    return rows, trace_rows


def _summarize_paired(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("branch") == "RealFunctional":
            key = (str(r.get("direction_id")), str(r.get("event_type")), str(r.get("gate_id")), str(r.get("horizon")))
            groups.setdefault(key, []).append(r)
    out: List[Dict[str, Any]] = []
    for key, real_rows in groups.items():
        direction_id, event_type, gate_id, horizon = key
        peer_controls = [
            r
            for r in rows
            if str(r.get("direction_id")) == direction_id
            and str(r.get("event_type")) == event_type
            and str(r.get("gate_id")) == gate_id
            and str(r.get("horizon")) == horizon
            and str(r.get("branch")) in {"C1-RandomMatchedNorm", "C4-GeometryD1Control", "C0-NoOpMatchedOverhead"}
        ]
        mean = lambda rs, k: sum(_to_float(r.get(k)) for r in rs) / max(1, len(rs))
        control_by_event: Dict[str, List[Dict[str, Any]]] = {}
        for c in peer_controls:
            control_by_event.setdefault(str(c.get("event_id")), []).append(c)
        causal_events = 0
        task_safe_events = 0
        triggered = sum(_to_int(r.get("event_triggered")) for r in real_rows)
        accepted = sum(_to_int(r.get("event_accepted")) for r in real_rows)
        for r in real_rows:
            controls = control_by_event.get(str(r.get("event_id")), [])
            if not controls:
                continue
            best_control_ce = min(_to_float(c.get("CEp99_delta_vs_adamw")) for c in controls)
            best_control_margin = max(_to_float(c.get("margin_p10_delta_vs_adamw")) for c in controls)
            best_control_curv = min(_to_float(c.get("curvature_proxy")) for c in controls)
            adamw_ce = abs(_to_float(r.get("adamw_CEp99")))
            ce_pass = _to_float(r.get("CEp99_delta_vs_adamw")) <= best_control_ce - 0.05 * adamw_ce
            margin_pass = _to_float(r.get("margin_p10_delta_vs_adamw")) >= best_control_margin + 0.02
            curv_pass = _to_float(r.get("curvature_proxy")) <= 0.90 * best_control_curv
            task_safe = _to_float(r.get("acc_delta_vs_adamw")) >= -0.005
            task_safe_events += int(task_safe)
            causal_events += int(task_safe and (ce_pass or margin_pass or curv_pass))
        event_count = len(real_rows)
        coverage = accepted / max(1, event_count)
        causality_rate = causal_events / max(1, event_count)
        task_safe_rate = task_safe_events / max(1, event_count)
        real_ce = mean(real_rows, "CEp99_delta_vs_adamw")
        real_margin = mean(real_rows, "margin_p10_delta_vs_adamw")
        real_curv = mean(real_rows, "curvature_delta_vs_adamw")
        best_control_ce_mean = min([mean([r for r in peer_controls if r.get("branch") == b], "CEp99_delta_vs_adamw") for b in {"C0-NoOpMatchedOverhead", "C1-RandomMatchedNorm", "C4-GeometryD1Control"}] or [0.0])
        best_control_margin_mean = max([mean([r for r in peer_controls if r.get("branch") == b], "margin_p10_delta_vs_adamw") for b in {"C0-NoOpMatchedOverhead", "C1-RandomMatchedNorm", "C4-GeometryD1Control"}] or [0.0])
        out.append(
            {
                "stage": "P2_PAIRED_EVENT_REPLAY_SUMMARY",
                "direction_id": direction_id,
                "event_type": event_type,
                "gate_id": gate_id,
                "horizon": horizon,
                "rows": event_count,
                "triggered_rows": triggered,
                "accepted_rows": accepted,
                "event_coverage": coverage,
                "task_safe_rate": task_safe_rate,
                "causal_event_rate": causality_rate,
                "real_CEp99_delta": real_ce,
                "best_control_CEp99_delta": best_control_ce_mean,
                "real_margin_delta": real_margin,
                "best_control_margin_delta": best_control_margin_mean,
                "real_curvature_delta": real_curv,
                "real_acc_delta": mean(real_rows, "acc_delta_vs_adamw"),
                "mean_step_time_ms": mean(real_rows, "step_time_ms"),
                "event_causality_pass": int(str(horizon) in {"5", "20"} and coverage >= 0.03 and task_safe_rate >= 0.95 and causality_rate >= 0.50),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    return out


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
    parser.add_argument("--previous-v9210-dir", default=str(PREV_V9210.relative_to(ROOT)))
    parser.add_argument("--candidates", default="LQ0,LQ1")
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--train-size", type=int, default=9984)
    parser.add_argument("--snr-ema-beta", type=float, default=0.97)
    parser.add_argument("--snr-warmup-steps", type=int, default=36)
    parser.add_argument("--snr-tau1", type=float, default=20.0)
    parser.add_argument("--snr-tau2", type=float, default=40.0)
    parser.add_argument("--snr-smooth-temperature", type=float, default=1.0)
    parser.add_argument("--high-snr-tau", type=float, default=20.0)
    parser.add_argument("--event-stride", type=int, default=1)
    parser.add_argument("--event-cep99-tau", type=float, default=4.0)
    parser.add_argument("--event-margin-tau", type=float, default=0.0)
    parser.add_argument("--event-curvature-tau", type=float, default=0.08)
    parser.add_argument("--event-entropy-tau", type=float, default=0.68)
    parser.add_argument("--step-fraction", type=float, default=0.01)
    parser.add_argument("--p2-event-steps", type=int, default=4)
    parser.add_argument("--p2-horizons", default="1,5,20")
    parser.add_argument("--p2-directions", default="D9-SignalChannelProjection,D10-OrthogonalSignalGeometry,D11-SignalSubspaceCurvature,D13-LiftConditionCorrection,D14-QuadraticBasisEntropyCorrection,D15-OutputScaleTailCorrection,D16-KMNISTHardModeGeometry")
    parser.add_argument("--p2-event-types", default="E0-uniform-stride,E2-margin-tail,E5-high-confidence-SNR")
    parser.add_argument("--p2-eval-size", type=int, default=512)
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

    prev_dir = Path(args.previous_v9210_dir)
    if not prev_dir.is_absolute():
        prev_dir = ROOT / prev_dir
    p0_rows, prev_p4_rows = _p0_source_recap(prev_dir)
    p1_rows = _p1_failure_attribution(prev_p4_rows)
    write_csv_rows(out_dir / "contract_audit_v9211.csv", [{
        "stage": "P0_CONTRACT",
        "source_artifact": str(prev_dir.relative_to(ROOT)) if prev_dir.exists() else str(prev_dir),
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
    }])
    write_json(out_dir / "run_manifest.json", {
        "created_utc": _now_iso(),
        "script": str(SCRIPT_PATH.relative_to(ROOT)),
        "plan": str(PLAN_PATH.relative_to(ROOT)),
        "device": str(device),
        "torch": torch.__version__,
        "args": vars(args),
        "source_artifacts": {"v9210": str(prev_dir.relative_to(ROOT)) if prev_dir.exists() else str(prev_dir)},
    })
    write_csv_rows(out_dir / "p0_v9210_reproduction.csv", p0_rows)
    write_csv_rows(out_dir / "p1_p4_failure_attribution.csv", p1_rows)

    p0_pass = _to_int(p0_rows[0].get("p0_reproduction_pass"))
    p1_pass = int(all(_to_int(r.get("p1_attribution_pass")) == 1 for r in p1_rows if str(r.get("status", "")) != "not_run"))
    if p0_pass and p1_pass:
        p2_rows, trace_rows = _run_paired_event_replay(args, device)
    else:
        reason = "P0_or_P1_failed"
        p2_rows = [_not_run("P2_PAIRED_EVENT_REPLAY_CAUSALITY", "p2_paired_event_replay_causality.csv", reason)]
        trace_rows = [_not_run("PAIRED_REPLAY_BRANCH_TRACE", "paired_replay_branch_trace_v9211.csv", reason)]
    write_csv_rows(out_dir / "p2_paired_event_replay_causality.csv", p2_rows)
    write_csv_rows(out_dir / "paired_replay_branch_trace_v9211.csv", trace_rows)
    measured_p2 = [r for r in p2_rows if str(r.get("status", "")) != "not_run"]
    p2_summary = _summarize_paired(measured_p2) if measured_p2 else [_not_run("P2_PAIRED_EVENT_REPLAY_SUMMARY", "p3_event_controller_direction_selection.csv", "P2_not_measured")]
    write_csv_rows(out_dir / "p3_event_controller_direction_selection.csv", p2_summary)

    p2_pass_rows = [r for r in p2_summary if _to_int(r.get("event_causality_pass")) == 1]
    best = None
    if p2_pass_rows:
        best = sorted(p2_pass_rows, key=lambda r: (-_to_float(r.get("causal_event_rate")), -_to_float(r.get("event_coverage"))))[0]

    downstream_reason = "P2_event_causality_failed"
    if not p0_pass:
        downstream_reason = "P0_v9210_reproduction_failed"
    elif not p1_pass:
        downstream_reason = "P1_failure_attribution_unclear"
    elif best:
        downstream_reason = "P2_event_causality_passed_but_P4_P7_not_executed_in_this_runner"
    for stage, name in [
        ("P4_SHORT_RUN_CONTROLLER_VALIDATION", "p4_short_run_controller_validation.csv"),
        ("P5_FULL_FUNCTIONAL_REENTRY_10SEED", "p5_full_functional_reentry_10seed.csv"),
        ("P6_ROBUSTNESS_NOISY_SIGNAL_VALIDATION", "p6_robustness_noisy_signal_validation.csv"),
        ("P7_STRONG_BASELINE_EXTERNAL_READY", "p7_strong_baseline_external_ready.csv"),
        ("FUNCTIONAL_EVENT_TRACE", "functional_event_trace_v9211.csv"),
    ]:
        write_csv_rows(out_dir / name, [_not_run(stage, name, downstream_reason)])

    if not p0_pass:
        route = "R7-EventControllerNoSurvivor"
        failure_code = "F3_v9210_reproduction_unstable"
        primary = "v9210_D9_one_step_survivor_or_P4_failure_not_reproduced"
        next_required = "repair_measurement_before_controller_search"
    elif not p1_pass:
        route = "R7-EventControllerNoSurvivor"
        failure_code = "F4_p4_failure_unattributed"
        primary = "v9210_P4_failure_attribution_unclear"
        next_required = "improve_failure_attribution_before_paired_replay"
    elif best:
        route = "R1-EventCausalityEstablished"
        failure_code = "F17_artifact_missing"
        primary = "P2_passed_but_P4_P7_not_executed_in_this_runner"
        next_required = "open_P4_short_run_controller_validation"
    else:
        # If real branches are task safe but never beat controls, choose control-equivalent route.
        real_rows = [r for r in measured_p2 if r.get("branch") == "RealFunctional"]
        task_safe = sum(int(_to_float(r.get("acc_delta_vs_adamw")) >= -0.005) for r in real_rows) / max(1, len(real_rows))
        route = "R5-FunctionalControlEquivalent" if task_safe >= 0.95 else "R7-EventControllerNoSurvivor"
        failure_code = "F6_control_equivalence_fail" if route == "R5-FunctionalControlEquivalent" else "F5_event_causality_fail"
        primary = "paired_replay_real_functional_did_not_beat_matched_controls"
        next_required = "redesign_functional_direction_or_event_controller"

    failure_rows = [{
        "stage": "P2",
        "failure_code": failure_code,
        "route": route,
        "reason": primary,
        "p0_pass": p0_pass,
        "p1_pass": p1_pass,
        "p2_event_causality_pass_count": len(p2_pass_rows),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    write_csv_rows(out_dir / "failure_table.csv", failure_rows)

    _write_svg(out_dir / "figures" / "p0_v9210_reproduction_dashboard.svg", "P0 v9.2.10 Reproduction", f"p0_pass={p0_pass}")
    _write_svg(out_dir / "figures" / "p1_failure_factor_matrix.svg", "P1 Failure Attribution", f"p1_pass={p1_pass}, factors={';'.join(str(r.get('primary_factors','')) for r in p1_rows[:2])}")
    _write_svg(out_dir / "figures" / "p2_paired_replay_branch_curves.svg", "P2 Paired Replay", f"route={route}, pass_groups={len(p2_pass_rows)}")
    _write_svg(out_dir / "figures" / "p2_real_vs_best_control_delta.svg", "P2 Real vs Controls", f"measured_rows={len(measured_p2)}")

    route_json = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "best_functional_candidate": best.get("direction_id", "") if best else "",
        "best_direction": best.get("direction_id", "") if best else "",
        "best_event_type": best.get("event_type", "") if best else "",
        "best_gate": best.get("gate_id", "") if best else "",
        "best_system_variant": "S1-cached-snr",
        "p0_reproduction_pass": p0_pass,
        "p1_failure_attribution_pass": p1_pass,
        "p2_event_causality_pass": int(bool(best)),
        "p2_event_causality_pass_count": len(p2_pass_rows),
        "p4_short_run_pass": 0,
        "p5_full_reentry_pass": 0,
        "functional_task_safe": int(route in {"R5-FunctionalControlEquivalent", "R1-EventCausalityEstablished"}),
        "functional_mechanism_pass": int(bool(best)),
        "functional_control_pass": int(bool(best)),
        "functional_system_broad_pass": 0,
        "functional_system_strong_pass": 0,
        "functional_kmnist_repair_pass": 0,
        "noise_robustness_pass": 0,
        "strong_baseline_challenge_pass": 0,
        "external_ready": 0,
        "primary_blocker": primary,
        "next_required_implementation": next_required,
        "success_v9211_event_causality": int(bool(best)),
        "success_v9211_short_run": 0,
        "success_v9211_full_functional": 0,
        "success_v9211_external_ready": 0,
    }
    write_json(out_dir / "route_decision.json", route_json)
    write_json(out_dir / "aggregate_decision.json", route_json)

    audit_targets = [
        out_dir / "contract_audit_v9211.csv",
        out_dir / "p0_v9210_reproduction.csv",
        out_dir / "p1_p4_failure_attribution.csv",
        out_dir / "p2_paired_event_replay_causality.csv",
        out_dir / "p3_event_controller_direction_selection.csv",
        out_dir / "p4_short_run_controller_validation.csv",
        out_dir / "p5_full_functional_reentry_10seed.csv",
        out_dir / "p6_robustness_noisy_signal_validation.csv",
        out_dir / "p7_strong_baseline_external_ready.csv",
        out_dir / "functional_event_trace_v9211.csv",
        out_dir / "paired_replay_branch_trace_v9211.csv",
        out_dir / "failure_table.csv",
    ]
    audit = audit_no_fake(audit_targets)
    write_csv_rows(out_dir / "v9211_provenance_audit.csv", [{"stage": "NO_FAKE_AUDIT", "route": route, **audit, "fake_data_used": int(audit["fake_data_used"]), "proxy_row_used": int(audit["proxy_row_used"]), "cpu_offload_used": int(audit["cpu_offload_used"])}])
    route_json.update({"no_fake": bool(audit["no_fake"]), "no_proxy": bool(audit["no_proxy"]), "rows_checked": int(audit["rows_checked"])})
    write_json(out_dir / "route_decision.json", route_json)
    write_json(out_dir / "aggregate_decision.json", route_json)
    write_csv_rows(
        out_dir / "artifact_hashes.csv",
        artifact_hash_rows(
            [
                PLAN_PATH,
                SCRIPT_PATH,
                ROOT / "dgkan" / "functional" / "lq_functional_predictor.py",
                ROOT / "dgkan" / "functional" / "snr_gated_lq.py",
                ROOT / "dgkan" / "models" / "fc_purekan_lq.py",
                out_dir / "route_decision.json",
                *audit_targets,
                out_dir / "v9211_provenance_audit.csv",
            ],
            root=ROOT,
        ),
    )
    print(json.dumps(route_json, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
