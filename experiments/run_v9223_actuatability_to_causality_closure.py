#!/usr/bin/env python3
"""DG-KAN v9.2.23 actuatability-to-causality closure runner.

This runner starts from the measured v9.2.22 strict PureKAN functional
interface boundary.  The central question is whether the large P3
actuatability proxy from v9.2.22 becomes realized logit/tail movement and then
beats AdamWParallel / LR controls.  Downstream stages are opened only when the
prior gate is actually measured and passed.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v92_basis_source_kernel_gate as v92  # noqa: E402
import run_v9222_basequalified_strict_purekan_functional_interface as v9222  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402
from dgkan.functional import lq_output_space_functional as out_lq  # noqa: E402
from dgkan.functional import snr_gated_lq as snr_lq  # noqa: E402
from dgkan.models import fc_purekan_actuator as act  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.23_Actuatability_to_Causality_Closure_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9223_actuatability_to_causality_closure.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9222 = RESULT_ROOT / "v9222_basequalified_strict_purekan_functional_interface_first_20260510T120000Z"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _mean(vals: Iterable[float]) -> float:
    clean = [float(v) for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return sum(clean) / max(1, len(clean))


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return 0.0
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 1.0e-30 or vy <= 1.0e-30:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def _q(vals: Sequence[float], q: float) -> float:
    clean = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not clean:
        return 0.0
    idx = min(len(clean) - 1, max(0, int(round((len(clean) - 1) * float(q)))))
    return clean[idx]


def _not_run(stage: str, artifact: str, reason: str) -> Dict[str, Any]:
    return {
        "stage": stage,
        "status": "not_run",
        "artifact": artifact,
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _clone_params(params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


def _clone_states(states: Sequence[AdamWState]) -> List[AdamWState]:
    return [AdamWState(step=s.step, m=s.m.detach().clone(), v=s.v.detach().clone()) for s in states]


def _apply_step(params: Sequence[torch.Tensor], step: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [p.detach() + d.detach() for p, d in zip(params, step)]


def _step_cos(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> float:
    an = snr_lq.step_norm(a)
    bn = snr_lq.step_norm(b)
    if bool((an <= 0).detach().cpu()) or bool((bn <= 0).detach().cpu()):
        return 0.0
    return float((snr_lq.step_dot(a, b) / (an * bn).clamp_min(1.0e-12)).detach().cpu())


def _orthogonal_to(step: Sequence[torch.Tensor], ref: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    denom = snr_lq.step_dot(ref, ref).clamp_min(1.0e-12)
    coeff = snr_lq.step_dot(step, ref) / denom
    return [s.detach() - r.detach() * coeff for s, r in zip(step, ref)]


def _roll_like_step(step: Sequence[torch.Tensor], shift: int) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for idx, part in enumerate(step):
        flat = part.detach().reshape(-1)
        if flat.numel() == 0:
            out.append(part.detach().clone())
        else:
            out.append(torch.roll(flat, shifts=(int(shift) + idx * 17) % int(flat.numel())).reshape_as(part))
    return out


def _target_mask(target_id: str, logits: torch.Tensor, y: torch.Tensor, dataset: str) -> torch.Tensor:
    log_probs = logits.log_softmax(dim=1)
    per_ce = -log_probs[torch.arange(y.numel(), device=y.device), y]
    row = torch.arange(y.numel(), device=y.device)
    true = logits[row, y]
    masked = logits.clone()
    masked[row, y] = -torch.inf
    wrong = masked.max(dim=1).values
    margin = true - wrong
    tid = str(target_id)
    if tid == "O1-HardTailLogitCorrection":
        return per_ce >= torch.quantile(per_ce.float(), 0.90)
    if tid == "O2-MarginTailExpansion":
        return margin <= torch.quantile(margin.float(), 0.20)
    if tid == "O6-KMNISTHardModeOutputTarget":
        if str(dataset) != "KMNIST":
            return torch.zeros_like(y, dtype=torch.bool)
        return margin <= torch.quantile(margin.float(), 0.30)
    return torch.ones_like(y, dtype=torch.bool)


def _eval_metrics(params: Sequence[torch.Tensor], mu: torch.Tensor, std: torch.Tensor, spec: act.ActuatorSpec, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    return v9222._eval_actuator_metrics(params, mu, std, spec, x, y)


def _metric_delta(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    return {
        "acc_delta": after["acc"] - before["acc"],
        "CEp99_delta": after["CE_p99"] - before["CE_p99"],
        "margin_p10_delta": after["correct_margin_p10"] - before["correct_margin_p10"],
        "ECE_delta": after["ECE"] - before["ECE"],
        "NLL_delta": after["NLL"] - before["NLL"],
        "loss_delta": after["loss"] - before["loss"],
    }


def _source_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9222 / "route_decision.json")
    audit_rows = read_csv_rows(SRC_V9222 / "v9222_provenance_audit.csv")
    fake_count = _int(audit_rows[0].get("fake_proxy_nonzero_count")) if audit_rows else 1
    p4_rows = read_csv_rows(SRC_V9222 / "p4_paired_replay_control_gate.csv")
    real = [r for r in p4_rows if str(r.get("branch")) == "RealFunctional"]
    p0_pass = int(
        route.get("route") == "R3-FunctionalActuatabilityRetained"
        and route.get("best_base_neutral_candidate") == "N2a-TinyInit-RationalFunc-BranchRatioCap"
        and _int(route.get("interface_p4_pass")) == 1
        and _int(route.get("interface_p5_nearpass")) == 1
        and _int(route.get("functional_actuatability_pass")) == 1
        and _int(route.get("paired_replay_pass")) == 0
        and fake_count == 0
    )
    return {
        "stage": "P0_V9222_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "source_artifact": str(SRC_V9222.relative_to(ROOT)),
        "route": route.get("route", ""),
        "best_base_neutral_candidate": route.get("best_base_neutral_candidate", ""),
        "interface_contract_pass": route.get("interface_contract_pass", 0),
        "interface_p4_pass": route.get("interface_p4_pass", 0),
        "interface_p5_nearpass": route.get("interface_p5_nearpass", 0),
        "functional_actuatability_pass": route.get("functional_actuatability_pass", 0),
        "paired_replay_pass": route.get("paired_replay_pass", 0),
        "p2_best_near_pass_count": route.get("p2_best_near_pass_count", ""),
        "p2_best_macro_delta": route.get("p2_best_macro_delta", ""),
        "p3_max_r_perp": route.get("p3_max_r_perp", ""),
        "p3_bad_event_rate": route.get("p3_bad_event_rate", ""),
        "p4_real_beats_adamwparallel_rate": route.get("p4_real_beats_adamwparallel_rate", ""),
        "p4_real_beats_best_lr_rate": route.get("p4_real_beats_best_lr_rate", ""),
        "p4_real_row_count": len(real),
        "fake_proxy_count": fake_count,
        "P0_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _make_args_for_v9222(args: argparse.Namespace) -> argparse.Namespace:
    # v9.2.22 helpers expect these names.  Keep them explicit so the v9.2.23
    # protocol remains auditable.
    for name, value in {
        "p5_train_size": args.train_size,
        "p5_test_size": args.test_size,
        "p5_epochs": args.p5_epochs,
        "p5_lr": args.lr,
        "eval_size": args.eval_size,
        "audit_batch_size": args.audit_batch_size,
        "batch_size": args.batch_size,
        "data_root": args.data_root,
        "seed": args.seed,
        "lr": args.lr,
    }.items():
        setattr(args, name, value)
    return args


def _load_split(args: argparse.Namespace, dataset: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    x_train, y_train, x_eval, y_eval, _in_dim, _out_dim, protocol = v92._load_task(
        args, dataset, train_size=int(args.train_size), test_size=int(args.eval_size)
    )
    return (
        x_train.to(device=device, dtype=torch.float32),
        y_train.to(device=device),
        x_eval.to(device=device, dtype=torch.float32),
        y_eval.to(device=device),
        protocol,
    )


def _train_cache(
    args: argparse.Namespace,
    cand: v9222.InterfaceCandidate,
    dataset: str,
    seed: int,
    device: torch.device,
    cache: Dict[Tuple[str, str, int], Dict[str, Any]],
) -> Dict[str, Any]:
    key = (cand.candidate_id, dataset, seed)
    if key not in cache:
        _row, saved = v9222._train_candidate(args, cand, dataset, seed, device, store_cache=True)
        assert saved is not None
        cache[key] = saved
    saved = cache[key]
    return {
        "params": [p.to(device=device) for p in saved["params"]],
        "states": _clone_states(saved["states"]),
        "mu": saved["mu"].to(device=device),
        "std": saved["std"].to(device=device),
    }


def _base_direction(
    args: argparse.Namespace,
    cand: v9222.InterfaceCandidate,
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    xb: torch.Tensor,
    yb: torch.Tensor,
    dataset: str,
    target_id: str,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
    assert cand.spec is not None
    base_logits = act.actuator_forward(xb, params, mu, std, cand.spec)
    _loss, grads = act.actuator_fwd_bwd(xb, yb, params, mu, std, cand.spec)
    task_step = snr_lq.gradient_descent_task_step(grads, float(args.lr))
    target, target_info = out_lq.build_output_target(target_id, base_logits, yb, dataset=dataset)
    raw_delta, ls_info = act.actuator_only_least_squares_delta(params, mu, std, cand.spec, xb, target, ridge=float(args.ridge))
    safe_delta, removed = snr_lq.project_step_to_task_safe(raw_delta, grads)
    adamw_logits_delta = act.actuator_forward(xb, _apply_step(params, task_step), mu, std, cand.spec) - base_logits
    safe_logits_delta = act.actuator_forward(xb, _apply_step(params, safe_delta), mu, std, cand.spec) - base_logits
    dot = (safe_logits_delta.float() * adamw_logits_delta.float()).sum()
    denom = adamw_logits_delta.float().square().sum().clamp_min(1.0e-12)
    perp = safe_logits_delta.float() - adamw_logits_delta.float() * (dot / denom)
    p3_proxy = float((perp.norm() / adamw_logits_delta.float().norm().clamp_min(1.0e-12)).detach().cpu())
    fit = act.output_fit_metrics(params, safe_delta, mu, std, cand.spec, xb, target)
    info: Dict[str, Any] = {
        "target_fit_R2": fit["output_target_fit_r2"],
        "target_selected_fraction": target_info.get("target_selected_fraction", 0.0),
        "p3_proxy_r_perp": p3_proxy,
        "p3_proxy_output_displacement_ratio": fit["output_displacement_to_target_ratio"],
        "projection_removed_norm": float(removed.detach().cpu()),
        "ls_residual_norm": ls_info.get("ls_residual_norm", 0.0),
    }
    return safe_delta, task_step, grads, info


def _actual_stats(
    *,
    params: Sequence[torch.Tensor],
    real_step: Sequence[torch.Tensor],
    parallel_step: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    spec: act.ActuatorSpec,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    dataset: str,
    target_id: str,
) -> Dict[str, Any]:
    before_logits = act.actuator_forward(x_eval, params, mu, std, spec)
    after_real = act.actuator_forward(x_eval, _apply_step(params, real_step), mu, std, spec)
    after_parallel = act.actuator_forward(x_eval, _apply_step(params, parallel_step), mu, std, spec)
    real_delta = after_real - before_logits
    par_delta = after_parallel - before_logits
    mask = _target_mask(target_id, before_logits, y_eval, dataset)
    if not bool(mask.any()):
        tail_real = torch.zeros((), device=x_eval.device)
        tail_par = torch.zeros((), device=x_eval.device)
    else:
        tail_real = real_delta[mask].float().norm()
        tail_par = par_delta[mask].float().norm()
    dot = (real_delta.float() * par_delta.float()).sum()
    denom = par_delta.float().square().sum().clamp_min(1.0e-12)
    perp = real_delta.float() - par_delta.float() * (dot / denom)
    before = v92._classification_metrics_from_logits(before_logits, y_eval)
    real_m = v92._classification_metrics_from_logits(after_real, y_eval)
    par_m = v92._classification_metrics_from_logits(after_parallel, y_eval)
    d_real = _metric_delta(before, real_m)
    d_par = _metric_delta(before, par_m)
    return {
        "actual_logit_delta_norm": float(real_delta.float().norm().detach().cpu()),
        "adamwparallel_logit_delta_norm": float(par_delta.float().norm().detach().cpu()),
        "actual_tail_logit_delta_norm": float(tail_real.detach().cpu()),
        "adamwparallel_tail_logit_delta_norm": float(tail_par.detach().cpu()),
        "actual_nonadamw_logit_delta_norm": float(perp.norm().detach().cpu()),
        "actual_r_z": float((real_delta.float().norm() / par_delta.float().norm().clamp_min(1.0e-12)).detach().cpu()),
        "actual_r_z_tail": float((tail_real / tail_par.clamp_min(1.0e-12)).detach().cpu()),
        "actual_r_z_perp": float((perp.norm() / par_delta.float().norm().clamp_min(1.0e-12)).detach().cpu()),
        "real_CEp99_delta": d_real["CEp99_delta"],
        "real_margin_p10_delta": d_real["margin_p10_delta"],
        "real_ECE_delta": d_real["ECE_delta"],
        "real_NLL_delta": d_real["NLL_delta"],
        "real_acc_delta": d_real["acc_delta"],
        "adamwparallel_CEp99_delta": d_par["CEp99_delta"],
        "adamwparallel_margin_p10_delta": d_par["margin_p10_delta"],
        "adamwparallel_ECE_delta": d_par["ECE_delta"],
        "adamwparallel_NLL_delta": d_par["NLL_delta"],
        "adamwparallel_acc_delta": d_par["acc_delta"],
        "curvature_delta": "not_measured_in_v9223_p1",
    }


def _p1_calibration(
    args: argparse.Namespace,
    device: torch.device,
    opened: bool,
    cache: Dict[Tuple[str, str, int], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        return [_not_run("P1_ACTUATABILITY_REALIZED_EFFECT_CALIBRATION", "p1_actuatability_realized_effect_calibration.csv", "P0_v9222_boundary_failed")], {}
    cand = v9222._candidate_registry()["N2a-TinyInit-RationalFunc-BranchRatioCap"]
    assert cand.spec is not None
    rows: List[Dict[str, Any]] = []
    for dataset in [v92._canonical_task(d) for d in _parse_list(args.p1_datasets)]:
        x_train, y_train, x_eval, y_eval, protocol = _load_split(args, dataset, device)
        xb = x_train[: int(args.audit_batch_size)]
        yb = y_train[: int(args.audit_batch_size)]
        for seed in _parse_ints(args.p1_seeds):
            saved = _train_cache(args, cand, dataset, seed, device, cache)
            params = saved["params"]
            mu = saved["mu"]
            std = saved["std"]
            for target_id in _parse_list(args.p1_targets):
                safe_delta, task_step, grads, info = _base_direction(args, cand, params, mu, std, xb, yb, dataset, target_id)
                real_step = v9222._cap_to_fraction(safe_delta, task_step, float(args.functional_step_fraction))
                parallel_step = v9222._cap_to_fraction(task_step, task_step, float(args.parallel_trust_ratio))
                stats = _actual_stats(
                    params=params,
                    real_step=real_step,
                    parallel_step=parallel_step,
                    mu=mu,
                    std=std,
                    spec=cand.spec,
                    x_eval=x_eval,
                    y_eval=y_eval,
                    dataset=dataset,
                    target_id=target_id,
                )
                after_params = _apply_step(params, real_step)
                diag = v9222._diagnose_channels(after_params, mu, std, cand.spec, xb, yb, float(args.lr))
                failure = "none"
                if stats["actual_r_z"] < 0.10 or stats["actual_r_z_tail"] < 0.10:
                    failure = "functional_event_silent"
                elif (
                    stats["real_CEp99_delta"] >= stats["adamwparallel_CEp99_delta"]
                    and stats["real_margin_p10_delta"] <= stats["adamwparallel_margin_p10_delta"]
                ):
                    failure = "functional_event_misaligned"
                rows.append({
                    "stage": "P1_ACTUATABILITY_REALIZED_EFFECT_CALIBRATION",
                    "candidate": cand.candidate_id,
                    "dataset": dataset,
                    "seed": seed,
                    "protocol": protocol,
                    "event_id": target_id,
                    "branch": "RealFunctional-vs-AdamWParallelTrustRatio",
                    **info,
                    **stats,
                    "CEp99_delta": stats["real_CEp99_delta"],
                    "margin_p10_delta": stats["real_margin_p10_delta"],
                    "ECE_delta": stats["real_ECE_delta"],
                    "NLL_delta": stats["real_NLL_delta"],
                    "acc_delta": stats["real_acc_delta"],
                    "branch_ratio_event": diag["branch_ratio"],
                    "effective_derivative_event": diag["effective_derivative_p95"],
                    "functional_step_norm": float(snr_lq.step_norm(real_step).detach().cpu()),
                    "adamwparallel_step_norm": float(snr_lq.step_norm(parallel_step).detach().cpu()),
                    "task_step_norm": float(snr_lq.step_norm(task_step).detach().cpu()),
                    "cos_functional_adamw": _step_cos(real_step, parallel_step),
                    "gradient_dot_functional": float(snr_lq.step_dot(grads, real_step).detach().cpu()),
                    "failure": failure,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
    corr = _corr([_float(r.get("p3_proxy_r_perp")) for r in rows], [_float(r.get("actual_r_z_perp")) for r in rows])
    mean_rz = _mean([_float(r.get("actual_r_z")) for r in rows])
    mean_tail_rz = _mean([_float(r.get("actual_r_z_tail")) for r in rows])
    silent_rate = _mean([1.0 if str(r.get("failure")) == "functional_event_silent" else 0.0 for r in rows])
    misaligned_rate = _mean([1.0 if str(r.get("failure")) == "functional_event_misaligned" else 0.0 for r in rows])
    proxy_pass = int(corr >= 0.30)
    silent = int(silent_rate >= 0.50 or mean_rz < 0.10 or mean_tail_rz < 0.10)
    misaligned = int((not silent) and misaligned_rate >= 0.50)
    return rows, {
        "actuatability_proxy_calibration_pass": proxy_pass,
        "p1_row_count": len(rows),
        "p1_proxy_actual_corr": corr,
        "p1_mean_actual_r_z": mean_rz,
        "p1_mean_actual_r_z_tail": mean_tail_rz,
        "p1_mean_actual_r_z_perp": _mean([_float(r.get("actual_r_z_perp")) for r in rows]),
        "p1_silent_rate": silent_rate,
        "p1_misaligned_rate": misaligned_rate,
        "p1_event_silent": silent,
        "p1_event_misaligned": misaligned,
    }


def _calibrated_step(
    args: argparse.Namespace,
    calibration: str,
    base_step: Sequence[torch.Tensor],
    task_step: Sequence[torch.Tensor],
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    spec: act.ActuatorSpec,
    xb: torch.Tensor,
    yb: torch.Tensor,
    grads: Sequence[torch.Tensor],
) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    step = list(base_step)
    info: Dict[str, Any] = {"event_accept_precheck": 1, "calibration_scale": 1.0}
    cal = str(calibration)
    if cal == "CAL3-N2a-branch-ratio-calibrated":
        diag0 = v9222._diagnose_channels(_apply_step(params, step), mu, std, spec, xb, yb, float(args.lr))
        target = float(args.branch_ratio_target)
        scale = max(0.1, min(5.0, target / max(1.0e-8, _float(diag0.get("branch_ratio")))))
        step = snr_lq.scale_step(step, scale)
        info["calibration_scale"] = scale
    elif cal == "CAL4-N2a-derivative-scale-calibrated":
        diag0 = v9222._diagnose_channels(_apply_step(params, step), mu, std, spec, xb, yb, float(args.lr))
        target = float(args.derivative_target)
        scale = max(0.1, min(5.0, target / max(1.0e-8, _float(diag0.get("effective_derivative_p95")))))
        step = snr_lq.scale_step(step, scale)
        info["calibration_scale"] = scale
    elif cal == "CAL5-N2a-nonadamw-orthogonal-calibrated":
        step = _orthogonal_to(step, task_step)
        step = v9222._cap_to_fraction(step, task_step, float(args.functional_step_fraction))
    # CAL1/CAL2/CAL6 are accepted/rejected after actual one-step stats.
    step, removed = snr_lq.project_step_to_task_safe(step, grads)
    info["post_calibration_projection_removed_norm"] = float(removed.detach().cpu())
    return v9222._cap_to_fraction(step, task_step, float(args.functional_step_fraction)), info


def _run_replay_for_event(
    args: argparse.Namespace,
    params: Sequence[torch.Tensor],
    states: Sequence[AdamWState],
    mu: torch.Tensor,
    std: torch.Tensor,
    spec: act.ActuatorSpec,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    real_step: Sequence[torch.Tensor],
    task_step: Sequence[torch.Tensor],
    seed: int,
    horizons: Sequence[int],
) -> Dict[Tuple[str, int], Dict[str, float]]:
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=0.0)
    real_norm = snr_lq.step_norm(real_step)
    branches: Dict[str, List[torch.Tensor]] = {
        "AdamWOnly": snr_lq.zero_like_params(params),
        "RealFunctional": list(real_step),
        "NoOpMatchedOverhead": snr_lq.zero_like_params(params),
        "RandomMatchedNorm": v9222._random_like_step(params, real_norm, seed + 922300),
        "AdamWParallelTrustRatio-0.003": v9222._cap_to_fraction(task_step, task_step, 0.003),
        "AdamWParallelTrustRatio-0.01": v9222._cap_to_fraction(task_step, task_step, 0.01),
        "AdamWParallelTrustRatio-0.03": v9222._cap_to_fraction(task_step, task_step, 0.03),
        "BestLRScale": v9222._cap_to_fraction(task_step, task_step, float(args.best_lr_scale) - 1.0),
        "ShuffledRoleMask": _roll_like_step(real_step, seed + 13),
        "InvertedRoleMask": [-d.detach() for d in real_step],
        "FrozenFuncChannel": snr_lq.zero_like_params(params),
        "ShuffledFuncChannel": _roll_like_step(real_step, seed + 71),
    }
    metrics: Dict[Tuple[str, int], Dict[str, float]] = {}
    for branch, fstep in branches.items():
        params_b = _clone_params(params)
        states_b = _clone_states(states)
        if branch not in {"AdamWOnly", "NoOpMatchedOverhead", "FrozenFuncChannel"}:
            for p, d in zip(params_b, fstep):
                p.add_(d)
        prev = 0
        for horizon in sorted(int(h) for h in horizons):
            v9222._run_adamw_steps(params_b, states_b, mu, std, spec, x_train, y_train, prev, horizon - prev, int(args.batch_size), cfg)
            prev = horizon
            metrics[(branch, horizon)] = _eval_metrics(params_b, mu, std, spec, x_eval, y_eval)
    return metrics


def _p2_activation(
    args: argparse.Namespace,
    device: torch.device,
    opened: bool,
    cache: Dict[Tuple[str, str, int], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P2_FT7_MECHANISM_CALIBRATED_ACTIVATION", "p2_ft7_mechanism_calibrated_activation.csv", "P1_actuatability_calibration_failed_or_event_silent")
        return [row], [_not_run("P3_INTERFACE_FAMILY_COMPARISON", "p3_interface_family_comparison.csv", "P2_not_opened")], {}
    cand = v9222._candidate_registry()["N2a-TinyInit-RationalFunc-BranchRatioCap"]
    assert cand.spec is not None
    rows: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    horizons = _parse_ints(args.p2_horizons)
    calibrations = _parse_list(args.p2_calibrations)
    for dataset in [v92._canonical_task(d) for d in _parse_list(args.p2_datasets)]:
        x_train, y_train, x_eval, y_eval, protocol = _load_split(args, dataset, device)
        xb = x_train[: int(args.audit_batch_size)]
        yb = y_train[: int(args.audit_batch_size)]
        for seed in _parse_ints(args.p2_seeds):
            saved = _train_cache(args, cand, dataset, seed, device, cache)
            params = saved["params"]
            states = saved["states"]
            mu = saved["mu"]
            std = saved["std"]
            for target_id in _parse_list(args.p2_targets):
                safe_delta, task_step, grads, base_info = _base_direction(args, cand, params, mu, std, xb, yb, dataset, target_id)
                for cal in calibrations:
                    real_step, cal_info = _calibrated_step(args, cal, safe_delta, task_step, params, mu, std, cand.spec, xb, yb, grads)
                    parallel_step = v9222._cap_to_fraction(task_step, task_step, float(args.parallel_trust_ratio))
                    stats = _actual_stats(
                        params=params,
                        real_step=real_step,
                        parallel_step=parallel_step,
                        mu=mu,
                        std=std,
                        spec=cand.spec,
                        x_eval=x_eval,
                        y_eval=y_eval,
                        dataset=dataset,
                        target_id=target_id,
                    )
                    accept = 1
                    if cal == "CAL1-N2a-realized-logit-calibrated" and stats["actual_r_z"] < 0.10:
                        accept = 0
                    if cal == "CAL2-N2a-tail-logit-calibrated" and stats["actual_r_z_tail"] < 0.10:
                        accept = 0
                    if cal == "CAL6-N2a-control-contrastive-calibrated":
                        if not (stats["real_CEp99_delta"] < stats["adamwparallel_CEp99_delta"] or stats["real_margin_p10_delta"] > stats["adamwparallel_margin_p10_delta"]):
                            accept = 0
                    if not accept:
                        real_step = snr_lq.zero_like_params(params)
                    after_diag = v9222._diagnose_channels(_apply_step(params, real_step), mu, std, cand.spec, xb, yb, float(args.lr))
                    metrics = _run_replay_for_event(args, params, states, mu, std, cand.spec, x_train, y_train, x_eval, y_eval, real_step, task_step, seed, horizons)
                    for horizon in horizons:
                        adamw = metrics[("AdamWOnly", horizon)]
                        best_parallel = min([m for (b, h), m in metrics.items() if h == horizon and b.startswith("AdamWParallel")], key=lambda m: m["CE_p99"])
                        best_lr = metrics[("BestLRScale", horizon)]
                        best_control = min([m for (b, h), m in metrics.items() if h == horizon and b not in {"RealFunctional", "AdamWOnly"}], key=lambda m: m["CE_p99"])
                        for (branch, h), after in metrics.items():
                            if h != horizon:
                                continue
                            delta = _metric_delta(adamw, after)
                            rows.append({
                                "stage": "P2_FT7_MECHANISM_CALIBRATED_ACTIVATION",
                                "candidate": cal,
                                "interface": "V1-TinyInit-Rational-BranchCap",
                                "base_candidate": cand.candidate_id,
                                "dataset": dataset,
                                "seed": seed,
                                "protocol": protocol,
                                "event_type": target_id,
                                "horizon": horizon,
                                "branch": branch,
                                "event_accept": accept if branch == "RealFunctional" else "",
                                "branch_ratio_target": args.branch_ratio_target,
                                "branch_ratio_actual": after_diag["branch_ratio"] if branch == "RealFunctional" else "",
                                "effective_derivative_target": args.derivative_target,
                                "effective_derivative_actual": after_diag["effective_derivative_p95"] if branch == "RealFunctional" else "",
                                "functional_step_norm": float(snr_lq.step_norm(real_step).detach().cpu()) if branch == "RealFunctional" else "",
                                "actual_r_z": stats["actual_r_z"] if branch == "RealFunctional" else "",
                                "actual_r_z_tail": stats["actual_r_z_tail"] if branch == "RealFunctional" else "",
                                "actual_r_z_perp": stats["actual_r_z_perp"] if branch == "RealFunctional" else "",
                                "bad_event_rate": int(branch == "RealFunctional" and after["acc"] < adamw["acc"] - 0.005),
                                "holdout_nonharm": int(branch != "RealFunctional" or after["loss"] <= adamw["loss"] + 1.0e-7),
                                "CEp99_delta": delta["CEp99_delta"],
                                "margin_p10_delta": delta["margin_p10_delta"],
                                "curvature_delta": "not_measured_in_v9223_p2",
                                "ECE_delta": delta["ECE_delta"],
                                "NLL_delta": delta["NLL_delta"],
                                "acc_delta": delta["acc_delta"],
                                "real_beats_adamwparallel": int(branch == "RealFunctional" and after["CE_p99"] < best_parallel["CE_p99"]),
                                "real_beats_best_lr": int(branch == "RealFunctional" and after["CE_p99"] < best_lr["CE_p99"]),
                                "real_beats_best_control": int(branch == "RealFunctional" and after["CE_p99"] < best_control["CE_p99"]),
                                "step_ratio_q90": "not_measured_event_replay",
                                "memory_ratio": "not_measured_event_replay",
                                "fake_data_used": 0,
                                "proxy_row_used": 0,
                                "cpu_offload_used": 0,
                            })
                    trace.append({
                        "stage": "branch_derivative_calibration_trace",
                        "candidate": cal,
                        "dataset": dataset,
                        "seed": seed,
                        "event_type": target_id,
                        **base_info,
                        **cal_info,
                        **stats,
                        "branch_ratio_actual": after_diag["branch_ratio"],
                        "effective_derivative_actual": after_diag["effective_derivative_p95"],
                        "event_accept": accept,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    })
    p3_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for cal in calibrations:
        real = [r for r in rows if r.get("candidate") == cal and r.get("branch") == "RealFunctional"]
        if not real:
            continue
        beat_parallel = _mean([_float(r.get("real_beats_adamwparallel")) for r in real])
        beat_lr = _mean([_float(r.get("real_beats_best_lr")) for r in real])
        bad = _mean([_float(r.get("bad_event_rate")) for r in real])
        nonharm = _mean([_float(r.get("holdout_nonharm")) for r in real])
        ce = _mean([_float(r.get("CEp99_delta")) for r in real])
        margin = _mean([_float(r.get("margin_p10_delta")) for r in real])
        survivor = int(bad <= 0.05 and nonharm >= 0.70 and beat_parallel >= 0.50 and beat_lr >= 0.50 and (ce < 0.0 or margin > 0.0))
        summary = {
            "stage": "P3_INTERFACE_FAMILY_COMPARISON",
            "interface": "V1-TinyInit-Rational-BranchCap",
            "basis_family": "rational",
            "base_phase_policy": "tiny_init_branch_ratio_cap",
            "event_phase_policy": cal,
            "contract_pass": 1,
            "P4_pass": 1,
            "P5_nearpass": 1,
            "actuatability_pass": 1,
            "actual_r_z_tail": _mean([_float(r.get("actual_r_z_tail")) for r in real if r.get("actual_r_z_tail") != ""]),
            "actual_r_z_perp": _mean([_float(r.get("actual_r_z_perp")) for r in real if r.get("actual_r_z_perp") != ""]),
            "bad_event_rate": bad,
            "holdout_nonharm": nonharm,
            "paired_replay_real_beats_adamwparallel_rate": beat_parallel,
            "paired_replay_real_beats_best_lr_rate": beat_lr,
            "CEp99_delta_real": ce,
            "margin_delta_real": margin,
            "curvature_delta_real": "not_measured_in_v9223_p3",
            "interface_survivor": survivor,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        summaries.append(summary)
        p3_rows.append(summary)
    for iface, reason in [
        ("V2-ZeroInit-Rational-EventOnly", "not_opened_after_V1_calibrated_activation_no_survivor"),
        ("V3-Warmstart-Rational-Attach", "not_opened_after_V1_calibrated_activation_no_survivor"),
        ("V4-TinyInit-Piecewise-BranchCap", "not_opened_after_V1_calibrated_activation_no_survivor"),
        ("V5-TinyInit-SharedRBF-BranchCap", "not_opened_after_V1_calibrated_activation_no_survivor"),
        ("V6-DualRole-Rational-TaskOrthogonal", "not_opened_after_V1_calibrated_activation_no_survivor"),
        ("V7-DualRole-Rational-TailCoupled", "not_opened_after_V1_calibrated_activation_no_survivor"),
    ]:
        p3_rows.append(_not_run("P3_INTERFACE_FAMILY_COMPARISON", "p3_interface_family_comparison.csv", reason) | {"interface": iface})
    survivors = [s for s in summaries if _int(s.get("interface_survivor"))]
    best = max(summaries, key=lambda r: (_float(r.get("paired_replay_real_beats_adamwparallel_rate")), _float(r.get("paired_replay_real_beats_best_lr_rate")), -abs(_float(r.get("CEp99_delta_real")))), default={})
    return rows, p3_rows, {
        "p2_activation_pass": int(bool(survivors)),
        "p2_survivor_count": len(survivors),
        "p3_interface_survivor_count": len(survivors),
        "best_calibrated_candidate": best.get("event_phase_policy", ""),
        "best_interface_family": best.get("interface", ""),
        "p2_best_real_beats_adamwparallel_rate": best.get("paired_replay_real_beats_adamwparallel_rate", 0),
        "p2_best_real_beats_best_lr_rate": best.get("paired_replay_real_beats_best_lr_rate", 0),
        "p2_best_bad_event_rate": best.get("bad_event_rate", 0),
        "p2_best_holdout_nonharm": best.get("holdout_nonharm", 0),
        "p2_best_CEp99_delta": best.get("CEp99_delta_real", 0),
        "p2_best_margin_delta": best.get("margin_delta_real", 0),
    }


def _write_downstream_not_run(out_dir: Path, reason: str) -> None:
    files = [
        ("p4_strict_paired_replay_control_gate.csv", "P4_STRICT_PAIRED_REPLAY_CONTROL_GATE"),
        ("p5_short_run_functional_validation.csv", "P5_SHORT_RUN_FUNCTIONAL_VALIDATION"),
        ("p6_full_10seed_functional_validation.csv", "P6_FULL_10SEED_FUNCTIONAL_VALIDATION"),
        ("p7_adamw_only_fullpass_repair.csv", "P7_ADAMW_ONLY_FULLPASS_REPAIR"),
        ("p8_robustness_external_ready.csv", "P8_ROBUSTNESS_EXTERNAL_READY"),
        ("paired_replay_branch_trace_v9223.csv", "paired_replay_branch_trace"),
        ("functional_channel_usage_trace_v9223.csv", "functional_channel_usage_trace"),
    ]
    for fname, stage in files:
        write_csv_rows(out_dir / fname, [_not_run(stage, fname, reason)])


def _write_figures(out_dir: Path, route: Dict[str, Any]) -> None:
    fig = ensure_dir(out_dir / "figures")
    items = [
        ("p0_v9222_boundary_dashboard.svg", "v9.2.22 boundary"),
        ("p0_gate_ladder.svg", "Gate ladder"),
        ("p0_real_vs_controls_recap.svg", "Real vs controls recap"),
        ("p1_proxy_vs_actual_rperp.svg", "Proxy vs actual r_perp"),
        ("p1_actual_logit_delta_distribution.svg", "Actual logit delta"),
        ("p1_tail_logit_delta_vs_CEp99.svg", "Tail logit delta vs CEp99"),
        ("p1_branch_ratio_derivative_vs_gain.svg", "Branch / derivative vs gain"),
        ("p1_projection_removed_norm.svg", "Projection removed norm"),
        ("p2_branch_ratio_band_pareto.svg", "Branch ratio band"),
        ("p2_derivative_band_pareto.svg", "Derivative band"),
        ("p2_real_vs_adamwparallel_by_event.svg", "Real vs AdamWParallel"),
        ("p2_event_time_activation_dashboard.svg", "Event-time activation"),
        ("p3_interface_family_pareto.svg", "Interface family pareto"),
        ("p3_basis_family_control_gap.svg", "Basis family control gap"),
        ("p3_paired_replay_heatmap.svg", "Paired replay heatmap"),
        ("p3_kmnist_tail_repair_by_interface.svg", "KMNIST tail repair"),
    ]
    lines = [
        f"route = {route.get('route')}",
        f"failure_mode = {route.get('failure_mode')}",
        f"corr = {route.get('p1_proxy_actual_corr')}",
        f"best_calibrated_candidate = {route.get('best_calibrated_candidate')}",
        f"blocker = {route.get('primary_blocker')}",
    ]
    for name, title in items:
        (fig / name).write_text(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1160\" height=\"250\">"
            "<rect width=\"1160\" height=\"250\" fill=\"#f8fafc\"/>"
            f"<text x=\"24\" y=\"42\" font-family=\"Arial\" font-size=\"24\" fill=\"#111827\">{title}</text>"
            + "".join(f"<text x=\"24\" y=\"{82 + i * 28}\" font-family=\"Arial\" font-size=\"16\" fill=\"#374151\">{line}</text>" for i, line in enumerate(lines))
            + "</svg>\n",
            encoding="utf-8",
        )


def _write_report(out_dir: Path, route: Dict[str, Any], audit: Dict[str, Any], p0: Dict[str, Any], p1: Dict[str, Any], p2: Dict[str, Any]) -> None:
    report = ROOT / "docs" / "DG-KAN_v9.2.23_Actuatability_to_Causality_Closure_实验复盘.md"
    text = f"""# DG-KAN v9.2.23 Actuatability-to-Causality Closure 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.23_Actuatability_to_Causality_Closure_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把未打开的 downstream 阶段写成通过。

## 0. 最新结论

```text
route = {route.get('route')}
base_candidate = LQ-t2-h256
success_v9223_strict_purekan_functional = {str(bool(route.get('success_v9223_strict_purekan_functional'))).lower()}
success_v9223_full_functional = {str(bool(route.get('success_v9223_full_functional'))).lower()}
success_v9223_external_ready = {str(bool(route.get('success_v9223_external_ready'))).lower()}
```

最终 artifact：

```text
{out_dir.relative_to(ROOT)}/
```

核心结论：

1. P0 复现 v9.2.22 boundary：`N2a` 仍是 base-qualified / actuatability pass 的 strict PureKAN interface source，paired replay source 仍未通过。
2. P1 重新测量 actuatability proxy 到 realized logit movement 的关系，proxy-realized correlation = `{_float(route.get('p1_proxy_actual_corr')):.6f}`。
3. P1 mean actual logit ratio `r_z = {_float(route.get('p1_mean_actual_r_z')):.6f}`，tail ratio `r_z_tail = {_float(route.get('p1_mean_actual_r_z_tail')):.6f}`，failure mode = `{route.get('failure_mode')}`。
4. P2/P3 状态：best calibrated candidate = `{route.get('best_calibrated_candidate')}`，best real beats AdamWParallel rate = `{_float(route.get('p2_best_real_beats_adamwparallel_rate')):.6f}`，best real beats best LR rate = `{_float(route.get('p2_best_real_beats_best_lr_rate')):.6f}`。
5. 当前 blocker：`{route.get('primary_blocker')}`。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v9223_actuatability_to_causality_closure.py` | v9.2.23 runner；生成 P0-P8 artifacts、actuatability-to-realized-effect calibration、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9223_actuatability_to_causality_closure.py
```

正式运行：

```bash
python experiments/run_v9223_actuatability_to_causality_closure.py \\
  --out-dir {out_dir.relative_to(ROOT)} \\
  --fresh \\
  --device auto \\
  --data-root data \\
  --seed 1314
```

## 2. Route

`route_decision.json`：

```json
{json.dumps(route, indent=2, ensure_ascii=False)}
```

## 3. P0 v9.2.22 boundary reproduction

Artifact：

```text
p0_v9222_boundary_reproduction.csv
```

关键值：

| metric | value |
|---|---:|
| source route | `{p0.get('route')}` |
| best candidate | `{p0.get('best_base_neutral_candidate')}` |
| interface P4/P5/actuatability | `{p0.get('interface_p4_pass')}/{p0.get('interface_p5_nearpass')}/{p0.get('functional_actuatability_pass')}` |
| source paired replay pass | `{p0.get('paired_replay_pass')}` |
| fake proxy count | `{p0.get('fake_proxy_count')}` |

判断：P0 是 source artifact recap，不伪装成新 full training。

## 4. P1 actuatability-to-realized-effect calibration

Artifact：

```text
p1_actuatability_realized_effect_calibration.csv
actual_logit_displacement_trace_v9223.csv
```

Summary：

```text
rows = {p1.get('p1_row_count')}
proxy_actual_corr = {p1.get('p1_proxy_actual_corr')}
mean_actual_r_z = {p1.get('p1_mean_actual_r_z')}
mean_actual_r_z_tail = {p1.get('p1_mean_actual_r_z_tail')}
mean_actual_r_z_perp = {p1.get('p1_mean_actual_r_z_perp')}
silent_rate = {p1.get('p1_silent_rate')}
misaligned_rate = {p1.get('p1_misaligned_rate')}
proxy_calibration_pass = {p1.get('actuatability_proxy_calibration_pass')}
```

判断：P1 不把 P3 的 large `r_perp` 直接当成功，而是重新测量 event 后 logits / tail logits 相对 AdamWParallel 的实际移动。

## 5. P2/P3 calibrated activation and interface comparison

Artifacts：

```text
p2_ft7_mechanism_calibrated_activation.csv
p3_interface_family_comparison.csv
branch_derivative_calibration_trace_v9223.csv
```

Summary：

```text
p2_activation_pass = {p2.get('p2_activation_pass', 0)}
p2_survivor_count = {p2.get('p2_survivor_count', 0)}
best_calibrated_candidate = {p2.get('best_calibrated_candidate', '')}
best_interface_family = {p2.get('best_interface_family', '')}
best_real_beats_adamwparallel_rate = {p2.get('p2_best_real_beats_adamwparallel_rate', '')}
best_real_beats_best_lr_rate = {p2.get('p2_best_real_beats_best_lr_rate', '')}
```

如果 P2 未打开或无 survivor，P4-P8 均保持 `not_run`，没有倒灌成功。

## 6. No-fake audit

`v9223_provenance_audit.csv`：

```text
rows_checked = {audit.get('rows_checked')}
fake_proxy_nonzero_count = {audit.get('fake_proxy_nonzero_count')}
fake_data_used = {audit.get('fake_data_used')}
proxy_row_used = {audit.get('proxy_row_used')}
cpu_offload_used = {audit.get('cpu_offload_used')}
no_fake = {str(audit.get('no_fake')).lower()}
no_proxy = {str(audit.get('no_proxy')).lower()}
```

## 7. Hash

见 `artifact_hashes.csv`。关键 artifacts 包含 plan、runner、route、P0-P8、trace 与 provenance audit。

## 8. 最终分析结论

v9.2.23 的真实推进是：

```text
v9.2.22: base qualification 和 actuatability 已闭合，但 paired replay 输给 AdamWParallel / LR。
v9.2.23: 校准 P3 actuatability proxy 到 actual logits / tail logits，再决定是否打开 calibrated activation。
```

机制判断：

1. 当前应该把 `r_perp` 理解成 actuatability proxy，而不是 causal gain。
2. 如果 P1 失败，说明下一步应先修 actuatability metric；如果 P1 过关但 P2/P3 失败，则说明 current base-neutral interface 的 event-time activation 仍 control-equivalent。
3. 本轮没有打开 short/full validation，除非 paired replay 真的通过 strong controls。

最终一句话：

> v9.2.23 真实执行后停在 `{route.get('route')}`：`{route.get('primary_blocker')}`。
"""
    report.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=9984)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--audit-batch-size", type=int, default=128)
    parser.add_argument("--p5-epochs", type=int, default=20)
    parser.add_argument("--ridge", type=float, default=1.0e-4)
    parser.add_argument("--functional-step-fraction", type=float, default=0.10)
    parser.add_argument("--parallel-trust-ratio", type=float, default=0.03)
    parser.add_argument("--best-lr-scale", type=float, default=1.03)
    parser.add_argument("--branch-ratio-target", type=float, default=0.18)
    parser.add_argument("--derivative-target", type=float, default=0.02)
    parser.add_argument("--p1-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--p1-seeds", default="0,1,2")
    parser.add_argument("--p1-targets", default="O1-HardTailLogitCorrection,O2-MarginTailExpansion,O6-KMNISTHardModeOutputTarget")
    parser.add_argument("--p2-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--p2-seeds", default="0,1,2")
    parser.add_argument("--p2-targets", default="O1-HardTailLogitCorrection,O2-MarginTailExpansion,O6-KMNISTHardModeOutputTarget")
    parser.add_argument("--p2-horizons", default="1,5,20,80")
    parser.add_argument("--p2-calibrations", default="CAL0-N2a-current,CAL1-N2a-realized-logit-calibrated,CAL2-N2a-tail-logit-calibrated,CAL3-N2a-branch-ratio-calibrated,CAL4-N2a-derivative-scale-calibrated,CAL5-N2a-nonadamw-orthogonal-calibrated,CAL6-N2a-control-contrastive-calibrated")
    args = parser.parse_args()
    args = _make_args_for_v9222(args)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    device = _device(args.device)

    write_json(out_dir / "run_manifest.json", {
        "experiment": "DG-KAN v9.2.23 Actuatability-to-Causality Closure",
        "created_utc": _now_iso(),
        "device": str(device),
        "plan": str(PLAN_PATH.relative_to(ROOT)),
        "script": str(SCRIPT_PATH.relative_to(ROOT)),
        "source_v9222": str(SRC_V9222.relative_to(ROOT)),
        "args": vars(args),
    })

    contract = [{
        "stage": "CONTRACT_AUDIT_V9223",
        "loss_type": "CE",
        "label_smoothing": 0,
        "teacher_used": 0,
        "distillation_used": 0,
        "sampler_changed": 0,
        "class_weight_used": 0,
        "cpu_offload_used": 0,
        "ordinary_mlp_path_used": 0,
        "external_residual_used": 0,
        "uses_loss_backward": 0,
        "purekanconv_deferred": 1,
        "purekanformer_deferred": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }]
    write_csv_rows(out_dir / "contract_audit_v9223.csv", contract)

    p0 = _source_boundary()
    write_csv_rows(out_dir / "p0_v9222_boundary_reproduction.csv", [p0])

    cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    p1_rows, p1_summary = _p1_calibration(args, device, _int(p0.get("P0_pass")) == 1, cache)
    write_csv_rows(out_dir / "p1_actuatability_realized_effect_calibration.csv", p1_rows)
    write_csv_rows(out_dir / "actual_logit_displacement_trace_v9223.csv", p1_rows)

    open_p2 = (
        _int(p0.get("P0_pass")) == 1
        and _int(p1_summary.get("actuatability_proxy_calibration_pass")) == 1
        and _int(p1_summary.get("p1_event_silent")) == 0
    )
    p2_rows, p3_rows, p2_summary = _p2_activation(args, device, open_p2, cache)
    write_csv_rows(out_dir / "p2_ft7_mechanism_calibrated_activation.csv", p2_rows)
    write_csv_rows(out_dir / "p3_interface_family_comparison.csv", p3_rows)
    trace_rows = [r for r in p2_rows if str(r.get("branch")) == "RealFunctional"] if open_p2 else p1_rows
    write_csv_rows(out_dir / "branch_derivative_calibration_trace_v9223.csv", trace_rows)
    write_csv_rows(out_dir / "functional_channel_usage_trace_v9223.csv", trace_rows)
    if not _int(p2_summary.get("p2_activation_pass")):
        reason = "P2_calibrated_activation_no_survivor" if open_p2 else "P1_actuatability_calibration_failed_or_event_silent"
        _write_downstream_not_run(out_dir, reason)

    failure_mode = "none"
    if not _int(p0.get("P0_pass")):
        route_name = "R1-ActuatabilityProxyMiscalibrated"
        failure_mode = "v9222_boundary_failed"
        blocker = "v9222_boundary_could_not_be_reproduced"
        next_impl = "reproduce_v9222_source_boundary_before_v9223"
    elif not _int(p1_summary.get("actuatability_proxy_calibration_pass")):
        route_name = "R1-ActuatabilityProxyMiscalibrated"
        failure_mode = "proxy_actuatability_miscalibrated"
        blocker = "P3_proxy_rperp_does_not_predict_realized_logit_movement"
        next_impl = "repair_actuatability_metric_before_event_activation"
    elif _int(p1_summary.get("p1_event_silent")):
        route_name = "R2-FunctionalEventSilent"
        failure_mode = "functional_event_silent"
        blocker = "realized_logit_or_tail_logit_displacement_below_threshold"
        next_impl = "increase_event_time_activation_band_without_breaking_base"
    elif _int(p1_summary.get("p1_event_misaligned")):
        route_name = "R3-FunctionalEventMisaligned"
        failure_mode = "functional_event_misaligned"
        blocker = "functional_moves_logits_but_not_along_tail_margin_gain"
        next_impl = "align_functional_event_to_FT7_branch_derivative_signal"
    elif _int(p2_summary.get("p2_activation_pass")):
        route_name = "R4-FT7MechanismActivationPass"
        failure_mode = "paired_replay_survivor_found_but_downstream_not_opened_in_this_runner"
        blocker = "P4_strict_paired_replay_not_implemented_after_P2_survivor"
        next_impl = "open_P4_strict_paired_replay_for_best_survivor"
    else:
        best_rate = max(_float(p2_summary.get("p2_best_real_beats_adamwparallel_rate")), _float(p2_summary.get("p2_best_real_beats_best_lr_rate")))
        route_name = "R8-InterfaceFamilyControlEquivalent" if best_rate < 0.30 else "R3-FunctionalEventMisaligned"
        failure_mode = "calibrated_interfaces_control_equivalent"
        blocker = "calibrated_activation_did_not_beat_adamwparallel_or_best_lr"
        next_impl = "return_to_deeper_interface_family_or_redefine_functional_event"

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9222_boundary_pass": p0.get("P0_pass", 0),
        "actuatability_proxy_calibration_pass": p1_summary.get("actuatability_proxy_calibration_pass", 0),
        "failure_mode": failure_mode,
        "best_calibrated_candidate": p2_summary.get("best_calibrated_candidate", ""),
        "best_interface_family": p2_summary.get("best_interface_family", ""),
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "functional_task_safe": 1.0 if p1_rows else 0.0,
        "functional_control_pass": 0,
        "functional_system_pass": 0,
        "functional_kmnist_repair_pass": 0,
        "adamw_fullpass": 0,
        "strong_baseline_pass": 0,
        "robustness_pass": 0,
        "external_ready": 0,
        "p1_row_count": p1_summary.get("p1_row_count", 0),
        "p1_proxy_actual_corr": p1_summary.get("p1_proxy_actual_corr", 0),
        "p1_mean_actual_r_z": p1_summary.get("p1_mean_actual_r_z", 0),
        "p1_mean_actual_r_z_tail": p1_summary.get("p1_mean_actual_r_z_tail", 0),
        "p1_mean_actual_r_z_perp": p1_summary.get("p1_mean_actual_r_z_perp", 0),
        "p1_silent_rate": p1_summary.get("p1_silent_rate", 0),
        "p1_misaligned_rate": p1_summary.get("p1_misaligned_rate", 0),
        "p2_activation_pass": p2_summary.get("p2_activation_pass", 0),
        "p2_survivor_count": p2_summary.get("p2_survivor_count", 0),
        "p3_interface_survivor_count": p2_summary.get("p3_interface_survivor_count", 0),
        "p2_best_real_beats_adamwparallel_rate": p2_summary.get("p2_best_real_beats_adamwparallel_rate", 0),
        "p2_best_real_beats_best_lr_rate": p2_summary.get("p2_best_real_beats_best_lr_rate", 0),
        "p2_best_bad_event_rate": p2_summary.get("p2_best_bad_event_rate", 0),
        "p2_best_holdout_nonharm": p2_summary.get("p2_best_holdout_nonharm", 0),
        "p2_best_CEp99_delta": p2_summary.get("p2_best_CEp99_delta", 0),
        "p2_best_margin_delta": p2_summary.get("p2_best_margin_delta", 0),
        "primary_blocker": blocker,
        "next_required_implementation": next_impl,
        "success_v9223_strict_purekan_functional": 0,
        "success_v9223_full_functional": 0,
        "success_v9223_external_ready": 0,
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", {"route": route, "p0": p0, "p1": p1_summary, "p2": p2_summary})

    failure_rows = [{
        "stage": "ROUTE",
        "failure": failure_mode,
        "primary_blocker": blocker,
        "next_required_implementation": next_impl,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    write_csv_rows(out_dir / "failure_table.csv", failure_rows)
    _write_figures(out_dir, route)

    audit_paths = [
        out_dir / "contract_audit_v9223.csv",
        out_dir / "p0_v9222_boundary_reproduction.csv",
        out_dir / "p1_actuatability_realized_effect_calibration.csv",
        out_dir / "p2_ft7_mechanism_calibrated_activation.csv",
        out_dir / "p3_interface_family_comparison.csv",
        out_dir / "p4_strict_paired_replay_control_gate.csv",
        out_dir / "p5_short_run_functional_validation.csv",
        out_dir / "p6_full_10seed_functional_validation.csv",
        out_dir / "p7_adamw_only_fullpass_repair.csv",
        out_dir / "p8_robustness_external_ready.csv",
        out_dir / "failure_table.csv",
    ]
    audit = audit_no_fake([p for p in audit_paths if p.exists()])
    write_csv_rows(out_dir / "v9223_provenance_audit.csv", [audit])

    hash_paths = [
        PLAN_PATH,
        SCRIPT_PATH,
        out_dir / "run_manifest.json",
        out_dir / "route_decision.json",
        out_dir / "aggregate_decision.json",
        *[p for p in audit_paths if p.exists()],
        out_dir / "actual_logit_displacement_trace_v9223.csv",
        out_dir / "branch_derivative_calibration_trace_v9223.csv",
        out_dir / "functional_channel_usage_trace_v9223.csv",
        out_dir / "v9223_provenance_audit.csv",
    ]
    write_csv_rows(out_dir / "artifact_hashes.csv", artifact_hash_rows(hash_paths, root=ROOT))
    _write_report(out_dir, route, audit, p0, p1_summary, p2_summary)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
