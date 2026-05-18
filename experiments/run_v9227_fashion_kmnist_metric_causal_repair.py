#!/usr/bin/env python3
"""DG-KAN v9.2.27 Fashion delayed + KMNIST metric-causal repair."""

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

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v92_basis_source_kernel_gate as v92  # noqa: E402
import run_v9222_basequalified_strict_purekan_functional_interface as v9222  # noqa: E402
import run_v9223_actuatability_to_causality_closure as v9223  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402
from dgkan.functional import snr_gated_lq as snr_lq  # noqa: E402


SCRIPT_PATH = ROOT / "experiments" / "run_v9227_fashion_kmnist_metric_causal_repair.py"
SRC_V9226 = ROOT / "results" / "real_rerun_20260506" / "v9226_fashion_delayed_controller_kmnist_diagnosis_first_20260510T160000Z"
SRC_V9222 = ROOT / "results" / "real_rerun_20260506" / "v9222_basequalified_strict_purekan_functional_interface_first_20260510T120000Z"


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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
    clean = [float(v) for v in vals if math.isfinite(float(v))]
    return sum(clean) / max(1, len(clean))


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    x = [float(v) for v in xs if math.isfinite(float(v))]
    y = [float(v) for v in ys if math.isfinite(float(v))]
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 1.0e-20 or vy <= 1.0e-20:
        return 0.0
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def _make_args(args: argparse.Namespace) -> argparse.Namespace:
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
        "best_lr_scale": args.best_lr_scale,
        "ridge": args.ridge,
    }.items():
        setattr(args, name, value)
    return args


def _source_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9226 / "route_decision.json")
    audit = read_csv_rows(SRC_V9226 / "v9226_provenance_audit.csv")
    fake = _int(audit[0].get("fake_proxy_nonzero_count")) if audit else 1
    p0_pass = int(
        route.get("route") == "R2-FashionPartialDelayedSignalKMNISTPrimitiveBlocker"
        and _int(route.get("fashion_ablation_pass")) == 0
        and route.get("kmnist_diagnosis_blocker") == "KMNIST_realized_effect_not_metric_causal"
        and _float(route.get("kmnist_actual_effect_pass_rate")) == 0.0
        and fake == 0
    )
    return {
        "stage": "P0_V9226_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "source_artifact": str(SRC_V9226.relative_to(ROOT)),
        "route": route.get("route", ""),
        "source_route_v9225": route.get("source_route", ""),
        "best_fashion_candidate": route.get("fashion_best_candidate", ""),
        "best_fashion_scope": route.get("fashion_best_scope", ""),
        "fashion_best_CEp99_delta": route.get("fashion_best_CEp99_delta", ""),
        "fashion_best_margin_delta": route.get("fashion_best_margin_delta", ""),
        "fashion_best_beats_adamwparallel": route.get("fashion_best_beats_adamwparallel", ""),
        "fashion_best_beats_best_lr": route.get("fashion_best_beats_best_lr", ""),
        "fashion_ablation_pass": route.get("fashion_ablation_pass", ""),
        "fashion_all_scope_pass_count": route.get("fashion_all_scope_pass_count", ""),
        "kmnist_target_oracle_useful_rate": route.get("kmnist_target_oracle_useful_rate", ""),
        "kmnist_raw_fit_pass_rate": route.get("kmnist_raw_fit_pass_rate", ""),
        "kmnist_safe_fit_pass_rate": route.get("kmnist_safe_fit_pass_rate", ""),
        "kmnist_actual_effect_pass_rate": route.get("kmnist_actual_effect_pass_rate", ""),
        "kmnist_max_cap_rz": route.get("kmnist_max_cap_rz", ""),
        "kmnist_max_raw_R2": route.get("kmnist_max_raw_R2", ""),
        "kmnist_max_safe_R2": route.get("kmnist_max_safe_R2", ""),
        "kmnist_mean_actual_CEp99_delta": route.get("kmnist_mean_actual_CEp99_delta", ""),
        "kmnist_mean_actual_margin_delta": route.get("kmnist_mean_actual_margin_delta", ""),
        "kmnist_diagnosis_blocker": route.get("kmnist_diagnosis_blocker", ""),
        "fake_proxy_count": fake,
        "P0_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _zero_like(params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [torch.zeros_like(p) for p in params]


def _combine_steps(steps: Sequence[Sequence[torch.Tensor]]) -> List[torch.Tensor]:
    if not steps:
        raise ValueError("empty step list")
    out = [torch.zeros_like(p) for p in steps[0]]
    for step in steps:
        for dst, src in zip(out, step):
            dst.add_(src)
    scale = 1.0 / float(len(steps))
    return [p * scale for p in out]


def _build_target(dataset: str, target_id: str, logits: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any], List[str]]:
    mapping = {
        "K1-KMNIST-O2-MarginTail": ["O2-MarginTailExpansion"],
        "K2-KMNIST-O1-CEp99Tail": ["O1-HardTailLogitCorrection"],
        "K3-KMNIST-O6-HardMode": ["O6-KMNISTHardModeOutputTarget"],
        "K4-KMNIST-ConfusionTail": ["O1-HardTailLogitCorrection"],
        "K5-KMNIST-CurvatureTail": ["O4-CurvatureOutputFlattening"],
        "K6-KMNIST-MetricCausalTarget": ["O1-HardTailLogitCorrection", "O2-MarginTailExpansion"],
        "K7-KMNIST-AbstainUnlessActualEffect": ["O6-KMNISTHardModeOutputTarget"],
        "K8-KMNIST-ClassModeBucketTarget": ["O1-HardTailLogitCorrection", "O6-KMNISTHardModeOutputTarget"],
        "O1-HardTailLogitCorrection": ["O1-HardTailLogitCorrection"],
        "O2-MarginTailExpansion": ["O2-MarginTailExpansion"],
        "O6-KMNISTHardModeOutputTarget": ["O6-KMNISTHardModeOutputTarget"],
    }
    parts = mapping.get(target_id, [target_id])
    targets: List[torch.Tensor] = []
    fracs: List[float] = []
    for tid in parts:
        target, info = v9223.out_lq.build_output_target(tid, logits, y, dataset=dataset)
        targets.append(target)
        fracs.append(_float(info.get("target_selected_fraction")))
    combined = sum(targets) / max(1, len(targets))
    return combined, {"target_selected_fraction": max(fracs or [0.0]), "target_parts": ",".join(parts)}, parts


def _step_for_targets(
    args: argparse.Namespace,
    cand: v9222.InterfaceCandidate,
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    xb: torch.Tensor,
    yb: torch.Tensor,
    dataset: str,
    targets: Sequence[str],
    fraction: float,
    mode: str,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
    assert cand.spec is not None
    base_logits = v9223.act.actuator_forward(xb, params, mu, std, cand.spec)
    _loss, grads = v9223.act.actuator_fwd_bwd(xb, yb, params, mu, std, cand.spec)
    task_step = snr_lq.gradient_descent_task_step(grads, float(args.lr))
    raw_steps: List[List[torch.Tensor]] = []
    infos: List[Dict[str, Any]] = []
    for target_id in targets:
        target, target_info, _parts = _build_target(dataset, target_id, base_logits, yb)
        raw_delta, ls_info = v9223.act.actuator_only_least_squares_delta(params, mu, std, cand.spec, xb, target, ridge=float(args.ridge))
        safe_delta, removed = snr_lq.project_step_to_task_safe(raw_delta, grads)
        raw_steps.append(safe_delta)
        fit = v9223.act.output_fit_metrics(params, safe_delta, mu, std, cand.spec, xb, target)
        infos.append({
            "target": target_id,
            "target_selected_fraction": target_info.get("target_selected_fraction", 0.0),
            "target_fit_R2": fit["output_target_fit_r2"],
            "target_rz": fit["output_displacement_to_target_ratio"],
            "projection_removed_norm": float(removed.detach().cpu()),
            "ls_residual_norm": ls_info.get("ls_residual_norm", 0.0),
        })
    step = _combine_steps(raw_steps)
    if mode == "orthogonal":
        step = v9223._orthogonal_to(step, task_step)
    elif mode == "inverse":
        step = [-d for d in step]
    step = v9222._cap_to_fraction(step, task_step, float(fraction))
    after_diag = v9222._diagnose_channels(v9223._apply_step(params, step), mu, std, cand.spec, xb, yb, float(args.lr))
    info = {
        "target_ids": ",".join(targets),
        "target_selected_fraction": max([_float(i.get("target_selected_fraction")) for i in infos] or [0.0]),
        "target_fit_R2": _mean([_float(i.get("target_fit_R2")) for i in infos]),
        "target_rz": _mean([_float(i.get("target_rz")) for i in infos]),
        "projection_removed_norm": _mean([_float(i.get("projection_removed_norm")) for i in infos]),
        "ls_residual_norm": _mean([_float(i.get("ls_residual_norm")) for i in infos]),
        "branch_ratio": after_diag["branch_ratio"],
        "effective_derivative": after_diag["effective_derivative_p95"],
    }
    return step, task_step, grads, info


def _best_controls(metrics: Dict[Tuple[str, int], Dict[str, float]], horizon: int) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    parallel = min([m for (b, h), m in metrics.items() if h == horizon and b.startswith("AdamWParallel")], key=lambda m: m["CE_p99"])
    lr = metrics[("BestLRScale", horizon)]
    controls = [m for (b, h), m in metrics.items() if h == horizon and b != "RealFunctional"]
    best = min(controls, key=lambda m: m["CE_p99"])
    return parallel, lr, best


def _fashion_controller_specs() -> List[Dict[str, Any]]:
    return [
        {"candidate": "F0-Fashion-NoFunctional", "targets": [], "fraction": 0.0, "mode": "zero", "implemented": 1},
        {"candidate": "F3-Fashion-O2-h80", "targets": ["O2-MarginTailExpansion"], "fraction": 0.50, "mode": "safe", "implemented": 1},
        {"candidate": "F6-Fashion-O1O2-BalancedDelayed", "targets": ["O1-HardTailLogitCorrection", "O2-MarginTailExpansion"], "fraction": 0.50, "mode": "safe", "implemented": 1},
        {"candidate": "F7-Fashion-O2-BranchBand", "targets": ["O2-MarginTailExpansion"], "fraction": 0.25, "mode": "safe", "implemented": 1},
        {"candidate": "F8-Fashion-O2-DerivativeBand", "targets": ["O2-MarginTailExpansion"], "fraction": 0.75, "mode": "safe", "implemented": 1},
        {"candidate": "F12-Fashion-O2-MarginTailHybrid", "targets": ["O1-HardTailLogitCorrection", "O2-MarginTailExpansion"], "fraction": 0.35, "mode": "safe", "implemented": 1},
        {"candidate": "F1-Fashion-O2-h20", "targets": [], "fraction": 0.0, "mode": "not_implemented", "implemented": 0},
        {"candidate": "F2-Fashion-O2-h50", "targets": [], "fraction": 0.0, "mode": "not_implemented", "implemented": 0},
        {"candidate": "F4-Fashion-O2-h160", "targets": [], "fraction": 0.0, "mode": "not_implemented", "implemented": 0},
        {"candidate": "F5-Fashion-O2-h240", "targets": [], "fraction": 0.0, "mode": "not_implemented", "implemented": 0},
        {"candidate": "F9-Fashion-O2-AbstainUnlessControlBeat", "targets": [], "fraction": 0.0, "mode": "not_implemented", "implemented": 0},
        {"candidate": "F10-Fashion-O2-MultiHorizonVote", "targets": [], "fraction": 0.0, "mode": "not_implemented", "implemented": 0},
        {"candidate": "F11-Fashion-O2-TailOnlyEvent", "targets": [], "fraction": 0.0, "mode": "not_implemented", "implemented": 0},
    ]


def _run_fashion(
    args: argparse.Namespace,
    device: torch.device,
    opened: bool,
    cache: Dict[Tuple[str, str, int], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P1_FASHION_DELAYED_SIGNAL_VERIFICATION", "p1_fashion_delayed_signal_verification.csv", "P0_boundary_failed")
        return [row], [row], [row], {}
    cand = v9222._candidate_registry()["N2a-TinyInit-RationalFunc-BranchRatioCap"]
    assert cand.spec is not None
    dataset = "Fashion-MNIST"
    x_train, y_train, x_eval, y_eval, protocol = v9223._load_split(args, dataset, device)
    xb = x_train[: int(args.audit_batch_size)]
    yb = y_train[: int(args.audit_batch_size)]
    horizons = _parse_ints(args.p1_horizons)
    rows: List[Dict[str, Any]] = []
    for spec in _fashion_controller_specs():
        if not spec["implemented"]:
            rows.append({
                "stage": "P1_FASHION_DELAYED_SIGNAL_VERIFICATION",
                "status": "not_implemented",
                "candidate": spec["candidate"],
                "reason": "controller_policy_not_implemented_in_v9227_first_wave",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            continue
        for seed in _parse_ints(args.p1_seeds):
            saved = v9223._train_cache(args, cand, dataset, seed, device, cache)
            params, states, mu, std = saved["params"], saved["states"], saved["mu"], saved["std"]
            if spec["mode"] == "zero":
                _loss, grads = v9223.act.actuator_fwd_bwd(xb, yb, params, mu, std, cand.spec)
                task_step = snr_lq.gradient_descent_task_step(grads, float(args.lr))
                real_step = _zero_like(params)
                info = {"target_ids": "none", "branch_ratio": 0.0, "effective_derivative": 0.0, "target_fit_R2": 0.0, "target_rz": 0.0}
            else:
                real_step, task_step, _grads, info = _step_for_targets(args, cand, params, mu, std, xb, yb, dataset, spec["targets"], spec["fraction"], spec["mode"])
            metrics = v9223._run_replay_for_event(args, params, states, mu, std, cand.spec, x_train, y_train, x_eval, y_eval, real_step, task_step, seed, horizons)
            for horizon in horizons:
                adamw = metrics[("AdamWOnly", horizon)]
                best_parallel, best_lr, best_control = _best_controls(metrics, horizon)
                best_parallel_delta = v9223._metric_delta(adamw, best_parallel)
                best_lr_delta = v9223._metric_delta(adamw, best_lr)
                best_control_delta = v9223._metric_delta(adamw, best_control)
                for (branch, h), after in metrics.items():
                    if h != horizon:
                        continue
                    delta = v9223._metric_delta(adamw, after)
                    rows.append({
                        "stage": "P1_FASHION_DELAYED_SIGNAL_VERIFICATION",
                        "status": "measured",
                        "candidate": spec["candidate"],
                        "dataset": dataset,
                        "seed": seed,
                        "protocol": protocol,
                        "horizon": horizon,
                        "branch": branch,
                        "selected_target": info.get("target_ids"),
                        "event_count": int(branch == "RealFunctional" and spec["mode"] != "zero"),
                        "event_coverage": 1.0 / max(1, int(args.audit_batch_size)) if branch == "RealFunctional" and spec["mode"] != "zero" else 0.0,
                        "CEp99_delta": delta["CEp99_delta"],
                        "margin_p10_delta": delta["margin_p10_delta"],
                        "ECE_delta": delta["ECE_delta"],
                        "NLL_delta": delta["NLL_delta"],
                        "acc_delta": delta["acc_delta"],
                        "curvature_delta": "not_measured_v9227_p1",
                        "task_safe": int(branch != "RealFunctional" or after["acc"] >= adamw["acc"] - 0.005),
                        "bad_event_rate": int(branch == "RealFunctional" and after["acc"] < adamw["acc"] - 0.005),
                        "real_beats_adamwparallel": int(branch == "RealFunctional" and after["CE_p99"] < best_parallel["CE_p99"]),
                        "real_beats_best_lr": int(branch == "RealFunctional" and after["CE_p99"] < best_lr["CE_p99"]),
                        "best_control_CEp99_delta": best_control_delta["CEp99_delta"],
                        "best_control_margin_delta": best_control_delta["margin_p10_delta"],
                        "best_adamwparallel_CEp99_delta": best_parallel_delta["CEp99_delta"],
                        "best_lr_CEp99_delta": best_lr_delta["CEp99_delta"],
                        "branch_ratio": info.get("branch_ratio", ""),
                        "effective_derivative": info.get("effective_derivative", ""),
                        "actual_logit_delta": "",
                        "actual_tail_logit_delta": "",
                        "target_fit_R2": info.get("target_fit_R2", ""),
                        "target_rz": info.get("target_rz", ""),
                        "step_ratio_q90": "not_measured_in_replay",
                        "memory_ratio": "not_measured_in_replay",
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    })
    real = [r for r in rows if r.get("status") == "measured" and r.get("branch") == "RealFunctional"]
    summary: List[Dict[str, Any]] = []
    for cand_id in sorted({str(r.get("candidate")) for r in real}):
        group = [r for r in real if str(r.get("candidate")) == cand_id]
        beat_p = _mean([_float(r.get("real_beats_adamwparallel")) for r in group])
        beat_lr = _mean([_float(r.get("real_beats_best_lr")) for r in group])
        task_safe = _mean([_float(r.get("task_safe")) for r in group])
        ce = _mean([_float(r.get("CEp99_delta")) for r in group])
        margin = _mean([_float(r.get("margin_p10_delta")) for r in group])
        best_ce = _mean([_float(r.get("best_control_CEp99_delta")) for r in group])
        best_margin = _mean([_float(r.get("best_control_margin_delta")) for r in group])
        mech = int((ce <= best_ce - 0.0005) or (margin >= best_margin + 0.001))
        fashion_pass = int(beat_p >= 0.75 and beat_lr >= 0.75 and task_safe >= 0.95 and mech)
        summary.append({
            "stage": "P1_FASHION_DELAYED_SIGNAL_SUMMARY",
            "candidate": cand_id,
            "rows": len(group),
            "real_beats_adamwparallel_rate": beat_p,
            "real_beats_best_lr_rate": beat_lr,
            "task_safe_rate": task_safe,
            "CEp99_delta": ce,
            "margin_p10_delta": margin,
            "best_control_CEp99_delta": best_ce,
            "best_control_margin_delta": best_margin,
            "mechanism_magnitude_pass": mech,
            "fashion_local_survivor": fashion_pass,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    mech_rows: List[Dict[str, Any]] = []
    for cand_id in sorted({str(r.get("candidate")) for r in real}):
        group = [r for r in real if str(r.get("candidate")) == cand_id]
        mech_rows.append({
            "stage": "P2_FASHION_MECHANISM_DIAGNOSIS",
            "candidate": cand_id,
            "corr_horizon_neg_CEp99": _corr([_float(r.get("horizon")) for r in group], [-_float(r.get("CEp99_delta")) for r in group]),
            "corr_horizon_margin": _corr([_float(r.get("horizon")) for r in group], [_float(r.get("margin_p10_delta")) for r in group]),
            "corr_branch_neg_CEp99": _corr([_float(r.get("branch_ratio")) for r in group], [-_float(r.get("CEp99_delta")) for r in group]),
            "corr_derivative_margin": _corr([_float(r.get("effective_derivative")) for r in group], [_float(r.get("margin_p10_delta")) for r in group]),
            "delayed_alignment_pass": int(
                _corr([_float(r.get("horizon")) for r in group], [-_float(r.get("CEp99_delta")) for r in group]) >= 0.30
                or _corr([_float(r.get("horizon")) for r in group], [_float(r.get("margin_p10_delta")) for r in group]) >= 0.30
            ),
            "branch_mechanism_pass": int(
                _corr([_float(r.get("branch_ratio")) for r in group], [-_float(r.get("CEp99_delta")) for r in group]) >= 0.30
                or _corr([_float(r.get("effective_derivative")) for r in group], [_float(r.get("margin_p10_delta")) for r in group]) >= 0.30
            ),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    survivors = [r for r in summary if _int(r.get("fashion_local_survivor"))]
    best = sorted(summary, key=lambda r: (-_int(r.get("fashion_local_survivor")), -_float(r.get("real_beats_adamwparallel_rate")), -_float(r.get("real_beats_best_lr_rate")), _float(r.get("CEp99_delta"))))[0] if summary else {}
    decision = {
        "fashion_local_survivor_count": len(survivors),
        "fashion_local_survivor": int(bool(survivors)),
        "fashion_best_candidate": best.get("candidate", ""),
        "fashion_best_beats_adamwparallel": _float(best.get("real_beats_adamwparallel_rate")) if best else 0.0,
        "fashion_best_beats_best_lr": _float(best.get("real_beats_best_lr_rate")) if best else 0.0,
        "fashion_best_CEp99_delta": _float(best.get("CEp99_delta")) if best else 0.0,
        "fashion_best_margin_delta": _float(best.get("margin_p10_delta")) if best else 0.0,
        "fashion_best_task_safe": _float(best.get("task_safe_rate")) if best else 0.0,
        "fashion_mechanism_pass": int(any(_int(r.get("delayed_alignment_pass")) or _int(r.get("branch_mechanism_pass")) for r in mech_rows)),
    }
    return rows, summary, mech_rows, decision


def _direct_oracle(logits: torch.Tensor, y: torch.Tensor, dataset: str, target_id: str, eps: float) -> Dict[str, Any]:
    target, info, _parts = _build_target(dataset, target_id, logits, y)
    before = v92._classification_metrics_from_logits(logits, y)
    scaled = target.float() / target.float().norm().clamp_min(1.0e-12) * (logits.float().norm() * float(eps)).detach()
    after = v92._classification_metrics_from_logits(logits + scaled, y)
    delta = v9223._metric_delta(before, after)
    return {
        "oracle_CEp99_delta": delta["CEp99_delta"],
        "oracle_margin_delta": delta["margin_p10_delta"],
        "oracle_ECE_delta": delta["ECE_delta"],
        "oracle_NLL_delta": delta["NLL_delta"],
        "oracle_acc_delta": delta["acc_delta"],
        "oracle_useful": int(delta["acc_delta"] >= -0.005 and (delta["CEp99_delta"] < 0 or delta["margin_p10_delta"] > 0)),
        "target_selected_fraction": info.get("target_selected_fraction", 0.0),
    }


def _run_kmnist_targets(
    args: argparse.Namespace,
    device: torch.device,
    opened: bool,
    cache: Dict[Tuple[str, str, int], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P3_KMNIST_TARGET_TO_REALIZED_EFFECT_DIAGNOSIS", "p3_kmnist_target_to_realized_effect.csv", "P0_boundary_failed")
        return [row], [row], {}
    cand = v9222._candidate_registry()["N2a-TinyInit-RationalFunc-BranchRatioCap"]
    assert cand.spec is not None
    dataset = "KMNIST"
    x_train, y_train, x_eval, y_eval, protocol = v9223._load_split(args, dataset, device)
    xb = x_train[: int(args.audit_batch_size)]
    yb = y_train[: int(args.audit_batch_size)]
    horizons = _parse_ints(args.p3_horizons)
    rows: List[Dict[str, Any]] = []
    for seed in _parse_ints(args.p3_seeds):
        saved = v9223._train_cache(args, cand, dataset, seed, device, cache)
        params, states, mu, std = saved["params"], saved["states"], saved["mu"], saved["std"]
        logits = v9223.act.actuator_forward(xb, params, mu, std, cand.spec)
        for target_id in _parse_list(args.p3_targets):
            target, target_info, _parts = _build_target(dataset, target_id, logits, yb)
            oracle = _direct_oracle(logits, yb, dataset, target_id, float(args.oracle_eps))
            real_step, task_step, _grads, info = _step_for_targets(args, cand, params, mu, std, xb, yb, dataset, [target_id], float(args.functional_step_fraction), "safe")
            raw_delta, _ls = v9223.act.actuator_only_least_squares_delta(params, mu, std, cand.spec, xb, target, ridge=float(args.ridge))
            raw_fit = v9223.act.output_fit_metrics(params, raw_delta, mu, std, cand.spec, xb, target)
            metrics = v9223._run_replay_for_event(args, params, states, mu, std, cand.spec, x_train, y_train, x_eval, y_eval, real_step, task_step, seed, horizons)
            for horizon in horizons:
                adamw = metrics[("AdamWOnly", horizon)]
                best_parallel, _best_lr, _best_control = _best_controls(metrics, horizon)
                best_parallel_delta = v9223._metric_delta(adamw, best_parallel)
                after = metrics[("RealFunctional", horizon)]
                delta = v9223._metric_delta(adamw, after)
                actual = v9223._actual_stats(
                    params=params,
                    real_step=real_step,
                    parallel_step=v9222._cap_to_fraction(task_step, task_step, float(args.parallel_trust_ratio)),
                    mu=mu,
                    std=std,
                    spec=cand.spec,
                    x_eval=x_eval,
                    y_eval=y_eval,
                    dataset=dataset,
                    target_id=target_id,
                )
                actual_pass = int(
                    after["acc"] >= adamw["acc"] - 0.005
                    and (after["CE_p99"] < best_parallel["CE_p99"] or after["correct_margin_p10"] > best_parallel["correct_margin_p10"])
                )
                rows.append({
                    "stage": "P3_KMNIST_TARGET_TO_REALIZED_EFFECT_DIAGNOSIS",
                    "status": "measured",
                    "candidate": cand.candidate_id,
                    "dataset": dataset,
                    "seed": seed,
                    "protocol": protocol,
                    "target": target_id,
                    "horizon": horizon,
                    "event_id": target_id,
                    **oracle,
                    "safe_fit_R2": info.get("target_fit_R2"),
                    "safe_fit_rz": info.get("target_rz"),
                    "raw_fit_R2": raw_fit["output_target_fit_r2"],
                    "raw_fit_rz": raw_fit["output_displacement_to_target_ratio"],
                    "actual_logit_delta_norm": actual["actual_logit_delta_norm"],
                    "actual_tail_logit_delta_norm": actual["actual_tail_logit_delta_norm"],
                    "actual_nonadamw_delta_norm": actual["actual_nonadamw_logit_delta_norm"],
                    "actual_r_z": actual["actual_r_z"],
                    "actual_r_z_tail": actual["actual_r_z_tail"],
                    "actual_CEp99_delta": delta["CEp99_delta"],
                    "actual_margin_delta": delta["margin_p10_delta"],
                    "actual_ECE_delta": delta["ECE_delta"],
                    "actual_NLL_delta": delta["NLL_delta"],
                    "actual_curvature_delta": "not_measured_v9227_p3",
                    "wrong_confidence_p95_delta": "not_measured_v9227_p3",
                    "top2_margin_delta": "not_measured_v9227_p3",
                    "confusion_pair_id": "not_bucketed_v9227_p3",
                    "class_or_mode_bucket": "not_bucketed_v9227_p3",
                    "best_adamwparallel_CEp99_delta": best_parallel_delta["CEp99_delta"],
                    "best_adamwparallel_margin_delta": best_parallel_delta["margin_p10_delta"],
                    "task_safe": int(after["acc"] >= adamw["acc"] - 0.005),
                    "bad_event": int(after["acc"] < adamw["acc"] - 0.005),
                    "actual_effect_pass": actual_pass,
                    "silent_failure": int(actual["actual_r_z"] < 0.10 or actual["actual_r_z_tail"] < 0.10),
                    "misaligned_failure": int(
                        actual["actual_r_z"] >= 0.10
                        and after["CE_p99"] >= best_parallel["CE_p99"]
                        and after["correct_margin_p10"] <= best_parallel["correct_margin_p10"]
                    ),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
    summary: List[Dict[str, Any]] = []
    for target in sorted({str(r.get("target")) for r in rows if r.get("status") == "measured"}):
        group = [r for r in rows if str(r.get("target")) == target and r.get("status") == "measured"]
        rate = _mean([_float(r.get("actual_effect_pass")) for r in group])
        summary.append({
            "stage": "P3_KMNIST_TARGET_SUMMARY",
            "target": target,
            "rows": len(group),
            "oracle_useful_rate": _mean([_float(r.get("oracle_useful")) for r in group]),
            "safe_fit_pass_rate": _mean([1.0 if _float(r.get("safe_fit_R2")) >= 0.20 and _float(r.get("safe_fit_rz")) >= 0.05 else 0.0 for r in group]),
            "actual_effect_pass_rate": rate,
            "silent_failure_rate": _mean([_float(r.get("silent_failure")) for r in group]),
            "misaligned_failure_rate": _mean([_float(r.get("misaligned_failure")) for r in group]),
            "mean_CEp99_delta": _mean([_float(r.get("actual_CEp99_delta")) for r in group]),
            "mean_margin_delta": _mean([_float(r.get("actual_margin_delta")) for r in group]),
            "kmnist_metric_causal_target_survivor": int(rate >= 0.50 and _mean([_float(r.get("task_safe")) for r in group]) >= 0.95),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    survivors = [r for r in summary if _int(r.get("kmnist_metric_causal_target_survivor"))]
    best = sorted(summary, key=lambda r: (-_float(r.get("actual_effect_pass_rate")), _float(r.get("mean_CEp99_delta"))))[0] if summary else {}
    decision = {
        "kmnist_target_survivor_count": len(survivors),
        "kmnist_metric_causal_target_survivor": int(bool(survivors)),
        "kmnist_best_target": best.get("target", ""),
        "kmnist_best_actual_effect_rate": _float(best.get("actual_effect_pass_rate")) if best else 0.0,
        "kmnist_best_CEp99_delta": _float(best.get("mean_CEp99_delta")) if best else 0.0,
        "kmnist_best_margin_delta": _float(best.get("mean_margin_delta")) if best else 0.0,
        "kmnist_failure_type": "target_survivor_found" if survivors else "realized_effect_still_not_metric_causal",
    }
    return rows, summary, decision


def _source_candidate_gate(candidate: str) -> Dict[str, Any]:
    rows = read_csv_rows(SRC_V9222 / "p2_base_neutral_interface_factory.csv")
    summaries = [r for r in rows if r.get("candidate") == candidate and r.get("status") == "candidate_summary"]
    p4s = [r for r in rows if r.get("candidate") == candidate and r.get("status") == "measured_P4"]
    s = summaries[0] if summaries else {}
    p4 = p4s[0] if p4s else {}
    return {
        "source_P4_pass": _int(p4.get("P4_pass")),
        "source_P5_nearpass": _int(s.get("P5_nearpass")),
        "source_forward_q90": p4.get("forward_q90", ""),
        "source_backward_q90": p4.get("backward_q90", ""),
        "source_step_q90": p4.get("step_q90", ""),
        "source_memory_ratio": p4.get("memory_ratio", ""),
        "source_near_pass_count": s.get("near_pass_count", ""),
        "source_macro_delta": s.get("macro_delta", ""),
    }


def _run_kmnist_primitives(
    args: argparse.Namespace,
    device: torch.device,
    opened: bool,
    cache: Dict[Tuple[str, str, int], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P4_KMNIST_PRIMITIVE_INTERFACE_REPAIR", "p4_kmnist_primitive_repair.csv", "P0_boundary_failed")
        return [row], [row], {}
    registry = v9222._candidate_registry()
    dataset = "KMNIST"
    x_train, y_train, x_eval, y_eval, protocol = v9223._load_split(args, dataset, device)
    xb = x_train[: int(args.audit_batch_size)]
    yb = y_train[: int(args.audit_batch_size)]
    horizons = _parse_ints(args.p4_horizons)
    rows: List[Dict[str, Any]] = []
    for cand_id in _parse_list(args.p4_primitives):
        cand = registry[cand_id]
        if cand.spec is None:
            rows.append({"stage": "P4_KMNIST_PRIMITIVE_INTERFACE_REPAIR", "status": "not_implemented", "primitive": cand_id, "reason": "no_edge_owned_spec", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
            continue
        gate = _source_candidate_gate(cand_id)
        for seed in _parse_ints(args.p4_seeds):
            saved = v9223._train_cache(args, cand, dataset, seed, device, cache)
            params, states, mu, std = saved["params"], saved["states"], saved["mu"], saved["std"]
            for target in _parse_list(args.p4_targets):
                real_step, task_step, _grads, info = _step_for_targets(args, cand, params, mu, std, xb, yb, dataset, [target], float(args.functional_step_fraction), "safe")
                metrics = v9223._run_replay_for_event(args, params, states, mu, std, cand.spec, x_train, y_train, x_eval, y_eval, real_step, task_step, seed, horizons)
                for horizon in horizons:
                    adamw = metrics[("AdamWOnly", horizon)]
                    best_parallel, _best_lr, _best_control = _best_controls(metrics, horizon)
                    after = metrics[("RealFunctional", horizon)]
                    d = v9223._metric_delta(adamw, after)
                    actual_pass = int(after["acc"] >= adamw["acc"] - 0.005 and (after["CE_p99"] < best_parallel["CE_p99"] or after["correct_margin_p10"] > best_parallel["correct_margin_p10"]))
                    rows.append({
                        "stage": "P4_KMNIST_PRIMITIVE_INTERFACE_REPAIR",
                        "status": "measured",
                        "primitive": cand_id,
                        "interface_family": cand.interface_family,
                        "target": target,
                        "dataset": dataset,
                        "seed": seed,
                        "protocol": protocol,
                        "horizon": horizon,
                        **gate,
                        "contract_pass": int(cand.spec is not None),
                        "grad_pass": "source_recap_v9222",
                        "P4_pass": gate["source_P4_pass"],
                        "P5_nearpass": gate["source_P5_nearpass"],
                        "safe_fit_R2": info.get("target_fit_R2"),
                        "safe_fit_rz": info.get("target_rz"),
                        "actual_effect_pass": actual_pass,
                        "actual_CEp99_delta": d["CEp99_delta"],
                        "actual_margin_delta": d["margin_p10_delta"],
                        "actual_curvature_delta": "not_measured_v9227_p4",
                        "task_safe": int(after["acc"] >= adamw["acc"] - 0.005),
                        "branch_ratio": info.get("branch_ratio"),
                        "effective_derivative": info.get("effective_derivative"),
                        "functional_channel_entropy": "source_recap_v9222",
                        "dominant_basis_fraction": "source_recap_v9222",
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    })
    summary: List[Dict[str, Any]] = []
    measured = [r for r in rows if r.get("status") == "measured"]
    for primitive in sorted({str(r.get("primitive")) for r in measured}):
        group = [r for r in measured if str(r.get("primitive")) == primitive]
        rate = _mean([_float(r.get("actual_effect_pass")) for r in group])
        task_safe = _mean([_float(r.get("task_safe")) for r in group])
        p4 = max([_int(r.get("P4_pass")) for r in group] or [0])
        p5 = max([_int(r.get("P5_nearpass")) for r in group] or [0])
        survivor = int(p4 and p5 and rate >= 0.50 and task_safe >= 0.95)
        summary.append({
            "stage": "P4_KMNIST_PRIMITIVE_SUMMARY",
            "primitive": primitive,
            "rows": len(group),
            "P4_pass": p4,
            "P5_nearpass": p5,
            "actual_effect_pass_rate": rate,
            "task_safe_rate": task_safe,
            "mean_CEp99_delta": _mean([_float(r.get("actual_CEp99_delta")) for r in group]),
            "mean_margin_delta": _mean([_float(r.get("actual_margin_delta")) for r in group]),
            "kmnist_primitive_survivor": survivor,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    survivors = [r for r in summary if _int(r.get("kmnist_primitive_survivor"))]
    best = sorted(summary, key=lambda r: (-_float(r.get("actual_effect_pass_rate")), _float(r.get("mean_CEp99_delta"))))[0] if summary else {}
    decision = {
        "kmnist_primitive_survivor_count": len(survivors),
        "kmnist_primitive_survivor": int(bool(survivors)),
        "kmnist_best_primitive": best.get("primitive", ""),
        "kmnist_best_primitive_actual_effect_rate": _float(best.get("actual_effect_pass_rate")) if best else 0.0,
        "kmnist_best_primitive_CEp99_delta": _float(best.get("mean_CEp99_delta")) if best else 0.0,
        "kmnist_best_primitive_margin_delta": _float(best.get("mean_margin_delta")) if best else 0.0,
    }
    return rows, summary, decision


def _write_downstream(out_dir: Path, reason: str) -> None:
    for fname, stage in [
        ("p5_unified_routed_controller.csv", "P5_UNIFIED_ROUTED_CONTROLLER_CONSTRUCTION"),
        ("p6_short_run_functional_validation.csv", "P6_SHORT_RUN_FUNCTIONAL_VALIDATION"),
        ("p7_full_10seed_validation.csv", "P7_FULL_10SEED_VALIDATION"),
        ("p8_adamw_only_fullpass_repair.csv", "P8_ADAMW_ONLY_FULLPASS_REPAIR"),
        ("p9_robustness_external_ready.csv", "P9_ROBUSTNESS_EXTERNAL_READY"),
    ]:
        write_csv_rows(out_dir / fname, [_not_run(stage, fname, reason)])


def _write_figures(out_dir: Path, route: Dict[str, Any]) -> None:
    fig = ensure_dir(out_dir / "figures")
    lines = [
        f"route = {route.get('route')}",
        f"fashion_best = {route.get('fashion_best_candidate')}",
        f"kmnist_best_target = {route.get('kmnist_best_target')}",
        f"kmnist_best_primitive = {route.get('kmnist_best_primitive')}",
        f"blocker = {route.get('primary_blocker')}",
    ]
    for name in [
        "p0_boundary_dashboard.svg",
        "p1_fashion_horizon_effect.svg",
        "p2_fashion_time_to_effect.svg",
        "p3_kmnist_oracle_vs_actual.svg",
        "p4_kmnist_primitive_pareto.svg",
        "p5_gate_ladder.svg",
    ]:
        (fig / name).write_text(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1180\" height=\"260\">"
            "<rect width=\"1180\" height=\"260\" fill=\"#f8fafc\"/>"
            f"<text x=\"24\" y=\"42\" font-family=\"Arial\" font-size=\"24\" fill=\"#111827\">{name}</text>"
            + "".join(f"<text x=\"24\" y=\"{82 + i * 28}\" font-family=\"Arial\" font-size=\"16\" fill=\"#374151\">{line}</text>" for i, line in enumerate(lines))
            + "</svg>\n",
            encoding="utf-8",
        )


def _write_report(out_dir: Path, route: Dict[str, Any], p0: Dict[str, Any], p1_summary: List[Dict[str, Any]], p3_summary: List[Dict[str, Any]], p4_summary: List[Dict[str, Any]], audit: Dict[str, Any]) -> None:
    report = ROOT / "docs" / "DG-KAN_v9.2.27_FashionDelayed_KMNISTMetricCausalRepair_实验复盘.md"
    lines = [
        "# DG-KAN v9.2.27 Fashion Delayed 与 KMNIST Metric-Causal Repair 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.2.27_完整重审_FashionDelayed_KMNISTMetricCausalRepair_实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把未打开的 P5-P9 写成通过。",
        "",
        "## 0. 最新结论",
        "",
        "```text",
        f"route = {route.get('route')}",
        "base_candidate = LQ-t2-h256",
        f"success_v9227_fashion_survivor = {bool(route.get('success_v9227_fashion_survivor'))}",
        f"success_v9227_kmnist_metric_causal = {bool(route.get('success_v9227_kmnist_metric_causal'))}",
        f"success_v9227_strict_purekan_functional = {bool(route.get('success_v9227_strict_purekan_functional'))}",
        "```",
        "",
        "最终 artifact：",
        "",
        "```text",
        str(out_dir.relative_to(ROOT)) + "/",
        "```",
        "",
        "核心结论：",
        "",
        f"1. P0 复现 v9.2.26 boundary：source route = `{p0.get('route')}`，fake/proxy = `{p0.get('fake_proxy_count')}`。",
        f"2. Fashion local survivor count = `{route.get('fashion_local_survivor_count')}`，best = `{route.get('fashion_best_candidate')}`，beats AdamWParallel = `{route.get('fashion_best_beats_adamwparallel')}`，beats best LR = `{route.get('fashion_best_beats_best_lr')}`。",
        f"3. KMNIST target survivor count = `{route.get('kmnist_target_survivor_count')}`，best target = `{route.get('kmnist_best_target')}`，actual effect rate = `{route.get('kmnist_best_actual_effect_rate')}`。",
        f"4. KMNIST primitive survivor count = `{route.get('kmnist_primitive_survivor_count')}`，best primitive = `{route.get('kmnist_best_primitive')}`，actual effect rate = `{route.get('kmnist_best_primitive_actual_effect_rate')}`。",
        f"5. 当前 blocker：`{route.get('primary_blocker')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/run_v9227_fashion_kmnist_metric_causal_repair.py` | v9.2.27 runner；生成 P0-P9 artifacts、route、failure/no-fake audit |",
        "",
        "代码检查：",
        "",
        "```text",
        "python -m py_compile experiments/run_v9227_fashion_kmnist_metric_causal_repair.py",
        "```",
        "",
        "正式运行：",
        "",
        "```bash",
        "python experiments/run_v9227_fashion_kmnist_metric_causal_repair.py \\",
        f"  --out-dir {out_dir.relative_to(ROOT)} \\",
        "  --fresh \\",
        "  --device auto \\",
        "  --data-root data \\",
        "  --seed 1314",
        "```",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, indent=2, sort_keys=True),
        "```",
        "",
        "## 3. P1 Fashion delayed verification",
        "",
        "Artifact：`p1_fashion_delayed_signal_verification.csv` / `p1_fashion_delayed_signal_summary.csv`",
        "",
        "| candidate | rows | beats AdamWParallel | beats best LR | task safe | CEp99 delta | margin delta | survivor |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in p1_summary:
        lines.append(
            f"| {r.get('candidate')} | `{r.get('rows')}` | `{_float(r.get('real_beats_adamwparallel_rate')):.6f}` | `{_float(r.get('real_beats_best_lr_rate')):.6f}` | `{_float(r.get('task_safe_rate')):.6f}` | `{_float(r.get('CEp99_delta')):.6f}` | `{_float(r.get('margin_p10_delta')):.6f}` | `{_int(r.get('fashion_local_survivor'))}` |"
        )
    lines.extend([
        "",
        "## 4. P3 KMNIST target-to-effect diagnosis",
        "",
        "Artifact：`p3_kmnist_target_to_realized_effect.csv` / `p3_kmnist_target_summary.csv`",
        "",
        "| target | rows | oracle useful | safe fit pass | actual effect | silent | misaligned | CEp99 delta | margin delta | survivor |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in p3_summary:
        lines.append(
            f"| {r.get('target')} | `{r.get('rows')}` | `{_float(r.get('oracle_useful_rate')):.6f}` | `{_float(r.get('safe_fit_pass_rate')):.6f}` | `{_float(r.get('actual_effect_pass_rate')):.6f}` | `{_float(r.get('silent_failure_rate')):.6f}` | `{_float(r.get('misaligned_failure_rate')):.6f}` | `{_float(r.get('mean_CEp99_delta')):.6f}` | `{_float(r.get('mean_margin_delta')):.6f}` | `{_int(r.get('kmnist_metric_causal_target_survivor'))}` |"
        )
    lines.extend([
        "",
        "## 5. P4 KMNIST primitive repair",
        "",
        "Artifact：`p4_kmnist_primitive_repair.csv` / `p4_kmnist_primitive_summary.csv`",
        "",
        "| primitive | rows | P4 | P5 near | actual effect | task safe | CEp99 delta | margin delta | survivor |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in p4_summary:
        lines.append(
            f"| {r.get('primitive')} | `{r.get('rows')}` | `{_int(r.get('P4_pass'))}` | `{_int(r.get('P5_nearpass'))}` | `{_float(r.get('actual_effect_pass_rate')):.6f}` | `{_float(r.get('task_safe_rate')):.6f}` | `{_float(r.get('mean_CEp99_delta')):.6f}` | `{_float(r.get('mean_margin_delta')):.6f}` | `{_int(r.get('kmnist_primitive_survivor'))}` |"
        )
    lines.extend([
        "",
        "## 6. Downstream boundary",
        "",
        "P5-P9 只有在 Fashion survivor 与 KMNIST target/primitive survivor 后打开。本轮未打开阶段均以 `not_run` row 落盘。",
        "",
        "## 7. No-fake audit",
        "",
        "```text",
        f"rows_checked = {audit.get('rows_checked')}",
        f"fake_proxy_nonzero_count = {audit.get('fake_proxy_nonzero_count')}",
        f"fake_data_used = {audit.get('fake_data_used')}",
        f"proxy_row_used = {audit.get('proxy_row_used')}",
        f"cpu_offload_used = {audit.get('cpu_offload_used')}",
        f"no_fake = {audit.get('no_fake')}",
        f"no_proxy = {audit.get('no_proxy')}",
        "```",
        "",
        "## 8. 最终分析结论",
        "",
        "v9.2.27 的真实推进是：",
        "",
        "```text",
        "Fashion delayed signal 经过更多 horizon/controller/seed 重审；",
        "KMNIST 从 target-to-effect 与 primitive/interface 两层重审 metric-causal repair。",
        "```",
        "",
        "机制判断：",
        "",
        "1. Fashion 只有在满足 strong controls 与机制幅度 gate 后才能进入 routed controller；否则仍是 partial diagnostic。",
        "2. KMNIST 若 oracle/safe fit 仍高但 actual effect 低，说明问题在 metric-causal transfer，而不是继续放大 cap。",
        "3. P4 primitive repair 若没有 survivor，当前 N2a/local/RBF family 不能把 KMNIST target 转成 metric gain。",
        "",
        "最终一句话：",
        "",
        f"> v9.2.27 真实执行后停在 `{route.get('route')}`：`{route.get('primary_blocker')}`。",
    ])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_downstream(out_dir: Path, reason: str) -> None:
    for fname, stage in [
        ("p5_unified_routed_controller.csv", "P5_UNIFIED_ROUTED_CONTROLLER_CONSTRUCTION"),
        ("p6_short_run_functional_validation.csv", "P6_SHORT_RUN_FUNCTIONAL_VALIDATION"),
        ("p7_full_10seed_validation.csv", "P7_FULL_10SEED_VALIDATION"),
        ("p8_adamw_only_fullpass_repair.csv", "P8_ADAMW_ONLY_FULLPASS_REPAIR"),
        ("p9_robustness_external_ready.csv", "P9_ROBUSTNESS_EXTERNAL_READY"),
    ]:
        write_csv_rows(out_dir / fname, [_not_run(stage, fname, reason)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=9984)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--audit-batch-size", type=int, default=128)
    parser.add_argument("--p5-epochs", type=int, default=20)
    parser.add_argument("--ridge", type=float, default=1.0e-4)
    parser.add_argument("--best-lr-scale", type=float, default=1.03)
    parser.add_argument("--parallel-trust-ratio", type=float, default=0.03)
    parser.add_argument("--functional-step-fraction", type=float, default=0.50)
    parser.add_argument("--oracle-eps", type=float, default=0.03)
    parser.add_argument("--p1-seeds", default="0,1,2,3,4")
    parser.add_argument("--p1-horizons", default="20,50,80,160,240")
    parser.add_argument("--p3-seeds", default="0,1,2,3,4")
    parser.add_argument("--p3-horizons", default="1,5,20,80,240")
    parser.add_argument("--p3-targets", default="K1-KMNIST-O2-MarginTail,K2-KMNIST-O1-CEp99Tail,K3-KMNIST-O6-HardMode,K5-KMNIST-CurvatureTail,K6-KMNIST-MetricCausalTarget,K8-KMNIST-ClassModeBucketTarget")
    parser.add_argument("--p4-seeds", default="0,1,2")
    parser.add_argument("--p4-horizons", default="20,80,240")
    parser.add_argument("--p4-targets", default="K1-KMNIST-O2-MarginTail,K2-KMNIST-O1-CEp99Tail,K3-KMNIST-O6-HardMode")
    parser.add_argument("--p4-primitives", default="N2a-TinyInit-RationalFunc-BranchRatioCap,N2b-TinyInit-PiecewiseFunc-BranchRatioCap,N2c-TinyInit-SharedRBFFunc-BranchRatioCap,N3c-SharedRBFFunc-DerivativeBand")
    args = _make_args(parser.parse_args())

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    device = _device(args.device)
    torch.manual_seed(int(args.seed))

    write_json(out_dir / "run_manifest.json", {
        "stage": "run_manifest",
        "script": str(SCRIPT_PATH.relative_to(ROOT)),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "args": vars(args),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    write_csv_rows(out_dir / "contract_audit_v9227.csv", [{
        "loss_type": "CE",
        "label_smoothing": 0,
        "teacher_used": 0,
        "distillation_used": 0,
        "sampler_changed": 0,
        "class_weight_used": 0,
        "uses_loss_backward": 0,
        "purekan_conv_measured": 0,
        "purekan_former_measured": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])

    cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    p0 = _source_boundary()
    write_csv_rows(out_dir / "p0_v9226_boundary_reproduction.csv", [p0])
    opened = bool(_int(p0.get("P0_pass")))

    p1_rows, p1_summary, p2_rows, fashion_decision = _run_fashion(args, device, opened, cache)
    write_csv_rows(out_dir / "p1_fashion_delayed_signal_verification.csv", p1_rows)
    write_csv_rows(out_dir / "p1_fashion_delayed_signal_summary.csv", p1_summary)
    write_csv_rows(out_dir / "p2_fashion_mechanism_diagnosis.csv", p2_rows)

    p3_rows, p3_summary, km_target_decision = _run_kmnist_targets(args, device, opened, cache)
    write_csv_rows(out_dir / "p3_kmnist_target_to_realized_effect.csv", p3_rows)
    write_csv_rows(out_dir / "p3_kmnist_target_summary.csv", p3_summary)

    p4_rows, p4_summary, km_primitive_decision = _run_kmnist_primitives(args, device, opened, cache)
    write_csv_rows(out_dir / "p4_kmnist_primitive_repair.csv", p4_rows)
    write_csv_rows(out_dir / "p4_kmnist_primitive_summary.csv", p4_summary)

    fashion_pass = _int(fashion_decision.get("fashion_local_survivor"))
    km_pass = int(_int(km_target_decision.get("kmnist_metric_causal_target_survivor")) or _int(km_primitive_decision.get("kmnist_primitive_survivor")))
    if fashion_pass and km_pass:
        route_name = "R1-FashionAndKMNISTSurvivorsReadyForRouting"
        primary = "P5_routed_controller_not_executed_in_this_runner"
        next_impl = "open_P5_unified_routed_controller"
    elif fashion_pass:
        route_name = "R3-FashionSurvivorOnlyKMNISTStillBlocked"
        primary = "kmnist_metric_causal_repair_failed"
        next_impl = "redesign_kmnist_metric_causal_primitive"
    elif km_pass:
        route_name = "R4-KMNISTRepairOnlyFashionStillPartial"
        primary = "fashion_delayed_signal_not_stable"
        next_impl = "redesign_fashion_delayed_controller"
    else:
        route_name = "R8-DelayedSignalNotGeneralizedPrimitiveReset"
        primary = "fashion_not_stable_and_kmnist_metric_causal_repair_failed"
        next_impl = "return_to_deeper_functional_interface_or_primitive_design"

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "source_route": p0.get("route", ""),
        **fashion_decision,
        **km_target_decision,
        **km_primitive_decision,
        "primary_blocker": primary,
        "next_required_implementation": next_impl,
        "success_v9227_fashion_survivor": fashion_pass,
        "success_v9227_kmnist_metric_causal": km_pass,
        "success_v9227_strict_purekan_functional": 0,
        "success_v9227_external_ready": 0,
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    write_csv_rows(out_dir / "failure_table.csv", [{
        "stage": "ROUTE",
        "failure": route_name,
        "primary_blocker": primary,
        "next_required_implementation": next_impl,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    _write_downstream(out_dir, primary)
    _write_figures(out_dir, route)

    audit_paths = [
        out_dir / "contract_audit_v9227.csv",
        out_dir / "p0_v9226_boundary_reproduction.csv",
        out_dir / "p1_fashion_delayed_signal_verification.csv",
        out_dir / "p1_fashion_delayed_signal_summary.csv",
        out_dir / "p2_fashion_mechanism_diagnosis.csv",
        out_dir / "p3_kmnist_target_to_realized_effect.csv",
        out_dir / "p3_kmnist_target_summary.csv",
        out_dir / "p4_kmnist_primitive_repair.csv",
        out_dir / "p4_kmnist_primitive_summary.csv",
        out_dir / "p5_unified_routed_controller.csv",
        out_dir / "p6_short_run_functional_validation.csv",
        out_dir / "p7_full_10seed_validation.csv",
        out_dir / "p8_adamw_only_fullpass_repair.csv",
        out_dir / "p9_robustness_external_ready.csv",
        out_dir / "failure_table.csv",
    ]
    audit = audit_no_fake(audit_paths)
    write_csv_rows(out_dir / "v9227_provenance_audit.csv", [audit])
    hash_paths = [SCRIPT_PATH, out_dir / "route_decision.json", *audit_paths, out_dir / "v9227_provenance_audit.csv"]
    write_csv_rows(out_dir / "artifact_hashes.csv", artifact_hash_rows(hash_paths, root=ROOT))
    _write_report(out_dir, route, p0, p1_summary, p3_summary, p4_summary, audit)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
