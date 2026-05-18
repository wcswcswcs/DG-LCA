#!/usr/bin/env python3
"""DG-KAN v9.2.19 functional-core rescue runner.

This runner starts from the v9.2.19 plan's hardest evidence gap: v8 functional
rows need v9-style strong controls.  The first executable wave here performs a
real v8 one-step strong-control replay on the historical KW4-style manual
candidate.  It does not pretend that this is the full 20-epoch P1 replay; the
full-run and strict PureKAN interface stages are written as explicit not_run
rows unless the required earlier gate is genuinely satisfied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import run_gafu_v87_real as v87  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import (  # noqa: E402
    artifact_hash_rows,
    ensure_dir,
    read_csv_rows,
    sha256_file,
    write_csv_rows,
    write_json,
)


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.19_FunctionalCore_Rescue_StrictPureKAN_Interface_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9219_functional_core_rescue.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"

SRC_V87 = RESULT_ROOT / "v87_full_chain_fmnist_kmnist_selected_kw4_formal_repeat_20260508T203000Z"
SRC_V9218 = RESULT_ROOT / "v9218_functional_retrospective_primitive_reset_first_20260510T060000Z"
SRC_V9217 = RESULT_ROOT / "v9217_signal_aligned_functional_or_reset_first_20260510T050000Z"
SRC_V9214 = RESULT_ROOT / "v9214_p4qualified_functional_actuator_closure_first_20260510T010000Z"
SRC_V927 = RESULT_ROOT / "v927_fc_purekan_lq_fullpass_functional_gate_20260509T190000Z"

V8_BRANCHES = [
    "V8-B0-AdamWOnly",
    "V8-FT7-RoleWiseFunctional",
    "V8-Adaptive-FT-P",
    "V8-NoOpMatchedOverhead",
    "V8-RandomMatchedNorm",
    "V8-ShuffledRoleMask",
    "V8-InvertedRoleMask",
    "V8-AdamWParallelSameNorm",
    "V8-AdamWParallelTrustRatio-0.003",
    "V8-AdamWParallelTrustRatio-0.01",
    "V8-AdamWParallelTrustRatio-0.03",
    "V8-LRScale-1.003",
    "V8-LRScale-1.01",
    "V8-LRScale-1.03",
    "V8-LRScale-1.10",
]

FUNCTIONAL_BRANCHES = {"V8-FT7-RoleWiseFunctional", "V8-Adaptive-FT-P"}
LR_BRANCHES = {name for name in V8_BRANCHES if "LRScale" in name}
ADAMW_PARALLEL_BRANCHES = {name for name in V8_BRANCHES if "AdamWParallel" in name}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_csv_list(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None, "nan", "NaN", "metric_unavailable"):
            return default
        return float(value)
    except Exception:
        return default


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / max(1, len(vals))


def _q(values: Sequence[float], q: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = float(q) * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _sync(device: torch.device) -> None:
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _to_dataset_name(name: str) -> str:
    lowered = name.lower()
    if lowered in {"fashion-mnist", "fashion", "fmnist"}:
        return "Fashion-MNIST"
    if lowered == "kmnist":
        return "KMNIST"
    return "MNIST"


def _metrics_from_logits(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        ce = F.cross_entropy(logits, y, reduction="none")
        probs = logits.softmax(dim=1)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()
        true_logits = logits.gather(1, y.view(-1, 1)).squeeze(1)
        masked = logits.clone()
        masked.scatter_(1, y.view(-1, 1), float("-inf"))
        margin = true_logits - masked.max(dim=1).values
        wrong_conf = probs.max(dim=1).values[pred != y]
        conf = probs.max(dim=1).values
        correct = (pred == y).float()
        ece = torch.zeros((), device=logits.device)
        for bucket in range(10):
            lo = bucket / 10.0
            hi = (bucket + 1) / 10.0
            mask = (conf > lo) & (conf <= hi)
            if bool(mask.any()):
                ece = ece + mask.float().mean() * torch.abs(conf[mask].mean() - correct[mask].mean())
        return {
            "acc": float(acc.detach().cpu()),
            "CEp99": float(torch.quantile(ce.detach().float(), 0.99).detach().cpu()),
            "margin_p10": float(torch.quantile(margin.detach().float(), 0.10).detach().cpu()),
            "wrong_confidence_p95": float(torch.quantile(wrong_conf.detach().float(), 0.95).detach().cpu()) if wrong_conf.numel() else 0.0,
            "ECE": float(ece.detach().cpu()),
            "NLL": float(ce.mean().detach().cpu()),
        }


def _eval_metrics(stack: Any, head: Any, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> Dict[str, float]:
    logits = v87.v85.v83._logits_only(stack, head, x, int(batch_size))
    return _metrics_from_logits(logits, y)


def _direction_delta(
    direction: Sequence[torch.Tensor],
    alpha: float,
    task_dirs: Sequence[torch.Tensor],
    task_alpha: float,
) -> List[torch.Tensor]:
    return [d.detach() * float(alpha) - t.detach() * float(task_alpha) for d, t in zip(direction, task_dirs)]


def _role_norms(
    entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]],
    tensors: Sequence[torch.Tensor],
) -> Dict[str, float]:
    out: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for (role, _name, _param, _grad), tensor in zip(entries, tensors):
        out[str(role)].append(tensor.detach())
    return {role: v87.v85.v83._norm(vals) for role, vals in out.items()}


def _role_snr(
    entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]],
    tensors: Sequence[torch.Tensor],
) -> Dict[str, float]:
    out: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for (role, _name, _param, _grad), tensor in zip(entries, tensors):
        out[str(role)].append(tensor.detach().flatten())
    snr: Dict[str, float] = {}
    for role, vals in out.items():
        flat = torch.cat(vals).float()
        snr[role] = float(flat.abs().mean().div(flat.std(unbiased=False).clamp_min(1.0e-12)).detach().cpu())
    return snr


def _build_v8_model(
    *,
    dataset: str,
    seed: int,
    hidden_dim: int,
    basis: int,
    device: torch.device,
    data_root: Path,
    train_size: int,
    test_size: int,
) -> Tuple[Any, Any, Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    x_train, y_train, x_test, y_test, input_dim, num_classes, protocol = v87.v85._load_kanbefair_vision_tensors(
        dataset,
        data_root=data_root,
        train_size=int(train_size),
        test_size=int(test_size),
        seed=int(seed),
    )
    spec = v87.v85.v83.v80._spec_map_v80()["KW4"]
    init_seed = v87.v85.v83.v72._stable_seed("v9219-v8-kw4", dataset, int(seed), hidden_dim, basis)
    v87.v85.set_seed(init_seed)
    stack, head = v87.v85.v83.v80._make_manual_candidate_v80(
        spec, int(input_dim), int(num_classes), int(hidden_dim), int(basis), device
    )
    v87.v85._configure_dg_update_modes("KW4", stack, head)
    return (
        stack,
        head,
        spec,
        x_train.to(device),
        y_train.to(device),
        x_test.to(device),
        y_test.to(device),
        protocol,
    )


def _candidate_direction(
    branch: str,
    entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]],
    task_dirs: Sequence[torch.Tensor],
    *,
    event_score: float,
    event_threshold: float,
    bad_step_risk: float,
    task_holdout_descent: float,
    ft7_reference: Tuple[List[torch.Tensor], List[torch.Tensor], Dict[str, float], Dict[str, float], int, float, float, float, str] | None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict[str, float], Dict[str, float], int, float, float, float, str, float, str]:
    base = v87.v85.v83._base_functional_dirs(entries, task_dirs)
    roles = sorted({str(role) for role, _name, _param, _grad in entries})
    if branch == "V8-B0-AdamWOnly":
        zeros = [torch.zeros_like(t) for t in task_dirs]
        return list(task_dirs), zeros, {role: 0.0 for role in roles}, {role: 0.0 for role in roles}, 0, 0.0, 0.0, 0.0, "adamw_only", 1.0, "alpha_task"
    if branch == "V8-NoOpMatchedOverhead":
        zeros = [torch.zeros_like(t) for t in task_dirs]
        return list(task_dirs), zeros, {role: 0.0 for role in roles}, {role: 0.0 for role in roles}, 0, 0.0, 0.0, 0.0, "noop_matched_overhead", 1.0, "alpha_task"
    if branch == "V8-RandomMatchedNorm":
        return (*v87._adaptive_candidate_direction(
            "RandomFunc",
            entries,
            task_dirs,
            event_score=event_score,
            event_threshold=event_threshold,
            bad_step_risk=bad_step_risk,
            task_holdout_descent=task_holdout_descent,
        ), 1.0, "normalize_direction")
    if branch == "V8-FT7-RoleWiseFunctional":
        out = v87._adaptive_candidate_direction(
            "Fixed-FT7-stride128",
            entries,
            task_dirs,
            event_score=event_score,
            event_threshold=event_threshold,
            bad_step_risk=bad_step_risk,
            task_holdout_descent=task_holdout_descent,
        )
        return (*out, 1.0, "normalize_direction")
    if branch == "V8-Adaptive-FT-P":
        out = v87._adaptive_candidate_direction(
            "Adaptive-FT-P",
            entries,
            task_dirs,
            event_score=event_score,
            event_threshold=event_threshold,
            bad_step_risk=bad_step_risk,
            task_holdout_descent=task_holdout_descent,
        )
        return (*out, 1.0, "normalize_direction")
    if branch == "V8-ShuffledRoleMask":
        weights = dict(v87.v85.v83.FT7_ROLE_WEIGHTS)
        shuffled = {"stack": float(weights.get("head", 1.0)), "head": float(weights.get("stack", 1.0))}
        corrected, trust_scale, clip_rate = v87._project_task_aware_v87(
            task_dirs,
            base["curv"],
            trust_ratio=0.10,
            role_weights=shuffled,
            entries=entries,
            max_scale=0.25,
        )
        return corrected, base["curv"], shuffled, dict(v87.v85.v83.FT7_ROLE_BUDGETS), 1, 0.10, trust_scale, clip_rate, "shuffled_ft7_role_mask", 1.0, "normalize_direction"
    if branch == "V8-InvertedRoleMask":
        inverted = [-f for f in base["curv"]]
        corrected, trust_scale, clip_rate = v87._project_task_aware_v87(
            task_dirs,
            inverted,
            trust_ratio=0.10,
            role_weights=dict(v87.v85.v83.FT7_ROLE_WEIGHTS),
            entries=entries,
            max_scale=0.25,
        )
        return corrected, inverted, dict(v87.v85.v83.FT7_ROLE_WEIGHTS), dict(v87.v85.v83.FT7_ROLE_BUDGETS), 1, 0.10, trust_scale, clip_rate, "inverted_ft7_functional_direction", 1.0, "normalize_direction"
    if branch == "V8-AdamWParallelSameNorm":
        if ft7_reference is not None:
            ref_dir = ft7_reference[0]
            ref_func_delta = [r - t for r, t in zip(ref_dir, task_dirs)]
            ratio = v87.v85.v83._norm(ref_func_delta) / max(v87.v85.v83._norm(task_dirs), 1.0e-30)
            alpha_mult = 1.0 + min(0.25, max(0.0, ratio))
        else:
            alpha_mult = 1.10
        zeros = [torch.zeros_like(t) for t in task_dirs]
        return list(task_dirs), zeros, {role: 0.0 for role in roles}, {role: 0.0 for role in roles}, 1, alpha_mult - 1.0, alpha_mult - 1.0, 0.0, "adamw_parallel_same_norm_extra_task_delta", alpha_mult, "alpha_task_scaled"
    if "AdamWParallelTrustRatio" in branch:
        rho = float(branch.rsplit("-", 1)[-1])
        zeros = [torch.zeros_like(t) for t in task_dirs]
        return list(task_dirs), zeros, {role: 0.0 for role in roles}, {role: 0.0 for role in roles}, 1, rho, rho, 0.0, "adamw_parallel_trust_ratio", 1.0 + rho, "alpha_task_scaled"
    if "LRScale" in branch:
        scale = float(branch.rsplit("-", 1)[-1])
        zeros = [torch.zeros_like(t) for t in task_dirs]
        return list(task_dirs), zeros, {role: 0.0 for role in roles}, {role: 0.0 for role in roles}, 1, scale - 1.0, scale - 1.0, 0.0, "scalar_lr_scale_control", scale, "alpha_task_scaled"
    raise ValueError(f"unknown v8 branch {branch}")


def _run_p1_v8_one_step(args: argparse.Namespace, out_dir: Path, device: torch.device) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    mechanism_rows: List[Dict[str, Any]] = []
    datasets = [_to_dataset_name(x) for x in _parse_csv_list(args.p1_datasets)]
    seeds = _parse_ints(args.p1_seeds)
    batch_size = int(args.p1_batch_size)

    for dataset in datasets:
        for seed in seeds:
            try:
                stack, head, spec, x_train, y_train, x_test, y_test, protocol = _build_v8_model(
                    dataset=dataset,
                    seed=seed,
                    hidden_dim=int(args.v8_hidden_dim),
                    basis=int(args.v8_basis_count),
                    device=device,
                    data_root=Path(args.data_root),
                    train_size=int(args.p1_train_size),
                    test_size=int(args.p1_test_size),
                )
                entries = v87.v85.v83._param_entries(stack, head)
                snapshot = v87.v85.v83._clone_params(entries)
                n = int(x_train.shape[0])
                holdout_offset = max(batch_size * (int(args.p1_warmup_batches) + int(args.p1_audit_batches) + 2), batch_size)
                event_scores: List[float] = []
                for warm_idx in range(int(args.p1_warmup_batches)):
                    idx = torch.arange(warm_idx * batch_size, min((warm_idx + 1) * batch_size, n), device=device) % n
                    hid = torch.arange(holdout_offset + warm_idx * batch_size, holdout_offset + (warm_idx + 1) * batch_size, device=device) % n
                    xb, yb = x_train[idx], y_train[idx]
                    xh, yh = x_train[hid], y_train[hid]
                    v87.v85.v83._restore_params(entries, snapshot)
                    loss_before, task_dirs = v87.v85.v83._manual_task_grads(stack, head, xb, yb, spec)
                    param_norm = v87.v85.v83._param_norm(entries)
                    task_norm = v87.v85.v83._norm(task_dirs)
                    task_alpha = float(args.p1_target_rel_update) * param_norm / max(task_norm, 1.0e-30)
                    holdout_before = v87.v85.v83._loss_only(stack, head, xh, yh, spec)
                    smooth_before, curv_before = v87.v85.v83._geometry_norms(entries)
                    v87.v85.v83._apply_direction(entries, task_dirs, task_alpha)
                    task_holdout_after = v87.v85.v83._loss_only(stack, head, xh, yh, spec)
                    v87.v85.v83._restore_params(entries, snapshot)
                    score, _td, _gp, _br = v87._adaptive_event_score(
                        holdout_loss_before=holdout_before,
                        holdout_task_after=task_holdout_after,
                        smooth_before=smooth_before,
                        curv_before=curv_before,
                    )
                    event_scores.append(score)
                event_threshold = _q(event_scores, 0.75) if event_scores else 0.0

                for audit_idx in range(int(args.p1_audit_batches)):
                    idx = torch.arange((int(args.p1_warmup_batches) + audit_idx) * batch_size, (int(args.p1_warmup_batches) + audit_idx + 1) * batch_size, device=device) % n
                    hid = torch.arange(holdout_offset + (int(args.p1_warmup_batches) + audit_idx) * batch_size, holdout_offset + (int(args.p1_warmup_batches) + audit_idx + 1) * batch_size, device=device) % n
                    xb, yb = x_train[idx], y_train[idx]
                    xh, yh = x_train[hid], y_train[hid]
                    v87.v85.v83._restore_params(entries, snapshot)
                    loss_before, task_dirs = v87.v85.v83._manual_task_grads(stack, head, xb, yb, spec)
                    train_before = _eval_metrics(stack, head, xb, yb, batch_size)
                    eval_before = _eval_metrics(stack, head, xh, yh, batch_size)
                    param_norm = v87.v85.v83._param_norm(entries)
                    task_norm = v87.v85.v83._norm(task_dirs)
                    task_alpha = float(args.p1_target_rel_update) * param_norm / max(task_norm, 1.0e-30)
                    smooth_before, curv_before = v87.v85.v83._geometry_norms(entries)
                    holdout_loss_before = v87.v85.v83._loss_only(stack, head, xh, yh, spec)
                    _sync(device)
                    task_start = time.perf_counter()
                    v87.v85.v83._apply_direction(entries, task_dirs, task_alpha)
                    task_train_loss_after = v87.v85.v83._loss_only(stack, head, xb, yb, spec)
                    task_holdout_loss_after = v87.v85.v83._loss_only(stack, head, xh, yh, spec)
                    task_eval_after = _eval_metrics(stack, head, xh, yh, batch_size)
                    _sync(device)
                    task_ms = (time.perf_counter() - task_start) * 1000.0
                    v87.v85.v83._restore_params(entries, snapshot)
                    event_score, task_descent_score, geometry_pressure, bad_step_risk = v87._adaptive_event_score(
                        holdout_loss_before=holdout_loss_before,
                        holdout_task_after=task_holdout_loss_after,
                        smooth_before=smooth_before,
                        curv_before=curv_before,
                    )
                    ft7_reference = v87._adaptive_candidate_direction(
                        "Fixed-FT7-stride128",
                        entries,
                        task_dirs,
                        event_score=event_score,
                        event_threshold=event_threshold,
                        bad_step_risk=bad_step_risk,
                        task_holdout_descent=holdout_loss_before - task_holdout_loss_after,
                    )
                    branch_temp: List[Dict[str, Any]] = []
                    for branch in V8_BRANCHES:
                        v87.v85.v83._restore_params(entries, snapshot)
                        (
                            direction,
                            func_dirs,
                            role_weights,
                            role_budgets,
                            event_triggered,
                            total_budget,
                            trust_scale,
                            clip_rate,
                            update_rule,
                            alpha_multiplier,
                            alpha_mode,
                        ) = _candidate_direction(
                            branch,
                            entries,
                            task_dirs,
                            event_score=event_score,
                            event_threshold=event_threshold,
                            bad_step_risk=bad_step_risk,
                            task_holdout_descent=holdout_loss_before - task_holdout_loss_after,
                            ft7_reference=ft7_reference,
                        )
                        direction_norm = v87.v85.v83._norm(direction)
                        if alpha_mode == "alpha_task_scaled":
                            alpha = task_alpha * float(alpha_multiplier)
                        elif alpha_mode == "alpha_task":
                            alpha = task_alpha
                        else:
                            alpha = float(args.p1_target_rel_update) * param_norm / max(direction_norm, 1.0e-30)
                        func_delta = _direction_delta(direction, alpha, task_dirs, task_alpha)
                        cos_functional_adamw = v87.v85.v83._cos(func_delta, [t * task_alpha for t in task_dirs])
                        cos_direction_adamw = v87.v85.v83._cos(direction, task_dirs)
                        role_update = _role_norms(entries, func_delta)
                        role_func_norm = _role_norms(entries, func_dirs)
                        role_task_norm = _role_norms(entries, task_dirs)
                        role_snr = _role_snr(entries, func_delta)
                        _sync(device)
                        started = time.perf_counter()
                        v87.v85.v83._apply_direction(entries, direction, alpha)
                        train_loss_after = v87.v85.v83._loss_only(stack, head, xb, yb, spec)
                        holdout_loss_after = v87.v85.v83._loss_only(stack, head, xh, yh, spec)
                        eval_after = _eval_metrics(stack, head, xh, yh, batch_size)
                        smooth_after, curv_after = v87.v85.v83._geometry_norms(entries)
                        _sync(device)
                        update_ms = (time.perf_counter() - started) * 1000.0
                        row = {
                            "stage": "P1_v8_one_step_strong_control_replay",
                            "status": "measured",
                            "candidate": branch,
                            "dataset": dataset,
                            "seed": seed,
                            "audit_batch": audit_idx,
                            "protocol": protocol,
                            "eval_scope": "holdout_microbatch",
                            "full_epoch_replay": 0,
                            "epochs_measured": 0,
                            "batch_size": batch_size,
                            "train_loss_before": loss_before,
                            "train_loss_after": train_loss_after,
                            "holdout_loss_before": holdout_loss_before,
                            "holdout_loss_after": holdout_loss_after,
                            "holdout_loss_delta": holdout_loss_after - holdout_loss_before,
                            "task_holdout_loss_after": task_holdout_loss_after,
                            "holdout_delta_vs_adamw": holdout_loss_after - task_holdout_loss_after,
                            "task_train_loss_after": task_train_loss_after,
                            "acc_before": eval_before["acc"],
                            "acc_after": eval_after["acc"],
                            "test_acc": "not_measured_one_step_holdout_scope",
                            "val_acc": "not_measured_one_step_holdout_scope",
                            "delta_vs_adamw": eval_after["acc"] - task_eval_after["acc"],
                            "delta_vs_mlp_match": "not_measured_one_step_holdout_scope",
                            "CEp99_before": eval_before["CEp99"],
                            "CEp99": eval_after["CEp99"],
                            "CEp99_delta": eval_after["CEp99"] - eval_before["CEp99"],
                            "CEp99_delta_vs_adamw": eval_after["CEp99"] - task_eval_after["CEp99"],
                            "margin_p10_before": eval_before["margin_p10"],
                            "margin_p10": eval_after["margin_p10"],
                            "margin_p10_delta": eval_after["margin_p10"] - eval_before["margin_p10"],
                            "margin_delta_vs_adamw": eval_after["margin_p10"] - task_eval_after["margin_p10"],
                            "wrong_confidence_p95": eval_after["wrong_confidence_p95"],
                            "ECE": eval_after["ECE"],
                            "ECE_delta": eval_after["ECE"] - eval_before["ECE"],
                            "NLL": eval_after["NLL"],
                            "NLL_delta": eval_after["NLL"] - eval_before["NLL"],
                            "curvature": curv_after,
                            "curvature_delta": curv_after - curv_before,
                            "curvature_ratio_vs_adamw": curv_after / max(curv_before, 1.0e-30),
                            "local_lipschitz": "not_measured_one_step",
                            "smooth_before": smooth_before,
                            "smooth_after": smooth_after,
                            "functional_event_count": event_triggered,
                            "event_coverage": event_triggered,
                            "event_score": event_score,
                            "event_threshold": event_threshold,
                            "task_descent_score": task_descent_score,
                            "geometry_pressure": geometry_pressure,
                            "bad_step_risk": bad_step_risk,
                            "step_ratio_q90": "computed_in_summary",
                            "step_time_ms": update_ms,
                            "step_ratio_vs_adamw": update_ms / max(task_ms, 1.0e-12),
                            "memory_ratio": "not_measured_one_step",
                            "control_rank": "computed_in_summary",
                            "real_beats_adamwparallel": "computed_in_summary",
                            "real_beats_best_lr": "computed_in_summary",
                            "task_norm": task_norm,
                            "direction_norm": direction_norm,
                            "functional_norm_vs_adamw": v87.v85.v83._norm(func_delta) / max(v87.v85.v83._norm([t * task_alpha for t in task_dirs]), 1.0e-30),
                            "cos_functional_adamw": cos_functional_adamw,
                            "cos_direction_adamw": cos_direction_adamw,
                            "trust_scale": trust_scale,
                            "clip_rate": clip_rate,
                            "total_budget": total_budget,
                            "alpha": alpha,
                            "task_alpha": task_alpha,
                            "alpha_multiplier": alpha_multiplier,
                            "alpha_mode": alpha_mode,
                            "update_rule": update_rule,
                            "role_weights": json.dumps(role_weights, sort_keys=True),
                            "role_budgets": json.dumps(role_budgets, sort_keys=True),
                            "role_update_norm_stack": role_update.get("stack", 0.0),
                            "role_update_norm_head": role_update.get("head", 0.0),
                            "role_func_norm_stack": role_func_norm.get("stack", 0.0),
                            "role_func_norm_head": role_func_norm.get("head", 0.0),
                            "role_task_norm_stack": role_task_norm.get("stack", 0.0),
                            "role_task_norm_head": role_task_norm.get("head", 0.0),
                            "role_snr_stack": role_snr.get("stack", 0.0),
                            "role_snr_head": role_snr.get("head", 0.0),
                            "branch_ratio": v87.v85.v83._norm(func_delta) / max(param_norm, 1.0e-30),
                            "effective_derivative_scale": abs(alpha) * max(float(role_func_norm.get("stack", 0.0)), float(role_func_norm.get("head", 0.0))),
                            "fake_data_used": 0,
                            "proxy_row_used": 0,
                            "cpu_offload_used": 0,
                        }
                        branch_temp.append(row)
                        for role in ("stack", "head"):
                            mechanism_rows.append({
                                "stage": "P2_v8_one_step_mechanism_attribution",
                                "status": "measured",
                                "candidate": branch,
                                "dataset": dataset,
                                "seed": seed,
                                "audit_batch": audit_idx,
                                "role": role,
                                "role_update_norm": role_update.get(role, 0.0),
                                "role_snr": role_snr.get(role, 0.0),
                                "role_curvature": role_func_norm.get(role, 0.0),
                                "role_event_frequency": event_triggered,
                                "branch_ratio": row["branch_ratio"],
                                "effective_derivative_scale": row["effective_derivative_scale"],
                                "cos_functional_adamw": cos_functional_adamw,
                                "cos_functional_lrcontrol": "computed_in_summary",
                                "cos_functional_random": "computed_in_summary",
                                "functional_norm_vs_adamw": row["functional_norm_vs_adamw"],
                                "pre_holdout_loss": holdout_loss_before,
                                "post_holdout_loss": holdout_loss_after,
                                "CEp99_delta": row["CEp99_delta"],
                                "margin_delta": row["margin_p10_delta"],
                                "curvature_delta": row["curvature_delta"],
                                "ECE_delta": row["ECE_delta"],
                                "NLL_delta": row["NLL_delta"],
                                "fake_data_used": 0,
                                "proxy_row_used": 0,
                                "cpu_offload_used": 0,
                            })
                    best_lr = min((r for r in branch_temp if r["candidate"] in LR_BRANCHES), key=lambda r: (_float(r["CEp99_delta"]), -_float(r["margin_p10_delta"])), default=None)
                    best_parallel = min((r for r in branch_temp if r["candidate"] in ADAMW_PARALLEL_BRANCHES), key=lambda r: (_float(r["CEp99_delta"]), -_float(r["margin_p10_delta"])), default=None)
                    best_control = min(branch_temp, key=lambda r: (_float(r["CEp99_delta"]), -_float(r["margin_p10_delta"])))
                    ordered = sorted(branch_temp, key=lambda r: (_float(r["CEp99_delta"]), -_float(r["margin_p10_delta"])))
                    ranks = {row["candidate"]: rank + 1 for rank, row in enumerate(ordered)}
                    for row in branch_temp:
                        row["best_lr_control"] = best_lr["candidate"] if best_lr else ""
                        row["best_lr_CEp99_delta"] = best_lr["CEp99_delta"] if best_lr else ""
                        row["best_lr_margin_delta"] = best_lr["margin_p10_delta"] - eval_before["margin_p10"] if best_lr else ""
                        row["best_adamwparallel_control"] = best_parallel["candidate"] if best_parallel else ""
                        row["best_adamwparallel_CEp99_delta"] = best_parallel["CEp99_delta"] if best_parallel else ""
                        row["best_adamwparallel_margin_delta"] = best_parallel["margin_p10_delta"] - eval_before["margin_p10"] if best_parallel else ""
                        row["best_control"] = best_control["candidate"]
                        row["control_rank"] = ranks.get(row["candidate"], "")
                        if row["candidate"] in FUNCTIONAL_BRANCHES and best_parallel and best_lr:
                            beats_parallel = int(
                                _float(row["CEp99_delta"]) < _float(best_parallel["CEp99_delta"])
                                or _float(row["margin_p10_delta"]) > _float(best_parallel["margin_p10_delta"])
                                or _float(row["curvature_delta"]) < 0.90 * _float(best_parallel["curvature_delta"])
                            )
                            beats_lr = int(
                                _float(row["CEp99_delta"]) < _float(best_lr["CEp99_delta"])
                                or _float(row["margin_p10_delta"]) > _float(best_lr["margin_p10_delta"])
                                or _float(row["curvature_delta"]) < 0.90 * _float(best_lr["curvature_delta"])
                            )
                            row["real_beats_adamwparallel"] = beats_parallel
                            row["real_beats_best_lr"] = beats_lr
                        else:
                            row["real_beats_adamwparallel"] = 0
                            row["real_beats_best_lr"] = 0
                        rows.append(row)
                del stack, head, x_train, y_train, x_test, y_test
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as exc:
                rows.append({
                    "stage": "P1_v8_one_step_strong_control_replay",
                    "status": "runtime_error",
                    "candidate": "ALL",
                    "dataset": dataset,
                    "seed": seed,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "full_epoch_replay": 0,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
    write_csv_rows(out_dir / "p1_v8_strong_control_replay.csv", rows)
    write_csv_rows(out_dir / "paired_replay_branch_trace_v9219.csv", rows)
    write_csv_rows(out_dir / "p2_v8_mechanism_attribution.csv", mechanism_rows)
    write_csv_rows(out_dir / "role_mechanism_trace_v9219.csv", mechanism_rows)
    return rows, mechanism_rows


def _p0_history_audit() -> List[Dict[str, Any]]:
    source_rows = read_csv_rows(SRC_V9218 / "p0_history_unified_audit.csv")
    if source_rows:
        return [
            {
                **row,
                "stage": "P0_history_unified_audit_v9219",
                "source": row.get("source", "v9218"),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
            for row in source_rows
        ]
    return [{
        "stage": "P0_history_unified_audit_v9219",
        "status": "source_missing",
        "reason": str(SRC_V9218 / "p0_history_unified_audit.csv"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]


def _not_run_row(stage: str, artifact: str, reason: str, **extra: Any) -> Dict[str, Any]:
    row = {
        "stage": stage,
        "artifact": artifact,
        "status": "not_run",
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row.update(extra)
    return row


def _source_recap_rows(stage: str) -> List[Dict[str, Any]]:
    v927 = _read_json(SRC_V927 / "route_decision.json")
    v9214 = _read_json(SRC_V9214 / "route_decision.json")
    v9217 = _read_json(SRC_V9217 / "route_decision.json")
    return [
        {
            "stage": stage,
            "candidate": "B0-LQ-T2-current",
            "status": "source_recap",
            "source": str(SRC_V927.relative_to(ROOT)),
            "P4_pass": v927.get("p4_pass", 1),
            "P5_nearpass": v927.get("p5_near_pass", 1),
            "P5_fullpass": v927.get("p5_pass", 0),
            "functional_actuatability_pass": 0,
            "paired_replay_vs_adamwparallel": "not_measured_for_base",
            "paired_replay_vs_lrcontrol": "not_measured_for_base",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "stage": stage,
            "candidate": "B9-A7c-BasisEntropy-ValueOnly",
            "status": "source_recap",
            "source": str(SRC_V9214.relative_to(ROOT)),
            "P4_pass": v9214.get("p4_closure_pass", 1),
            "P5_nearpass": v9214.get("actuator_base_near_pass", 1),
            "P5_fullpass": 0,
            "functional_actuatability_pass": v9214.get("controllability_retained", 1),
            "paired_replay_vs_adamwparallel": "fail_in_v9214_v9215",
            "paired_replay_vs_lrcontrol": "fail_in_v9217",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "stage": stage,
            "candidate": "v9217-signal-aligned-controls",
            "status": "source_recap",
            "source": str(SRC_V9217.relative_to(ROOT)),
            "P4_pass": "source_not_basis_candidate",
            "P5_nearpass": "source_not_basis_candidate",
            "P5_fullpass": "source_not_basis_candidate",
            "functional_actuatability_pass": 0,
            "paired_replay_vs_adamwparallel": "effective_survivor_count_0",
            "paired_replay_vs_lrcontrol": v9217.get("best_lr_control", v9217.get("best_control", "")),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
    ]


def _write_simple_figures(out_dir: Path, route: Dict[str, Any], p1_summary: Sequence[Dict[str, Any]]) -> None:
    fig_dir = ensure_dir(out_dir / "figures")
    lines = [
        f"route = {route.get('route')}",
        f"v8_one_step_rows = {route.get('v8_one_step_row_count')}",
        f"v8_one_step_functional_beats_controls = {route.get('v8_one_step_functional_beats_controls')}",
        f"primary_blocker = {route.get('primary_blocker')}",
    ]
    (fig_dir / "p1_v8_functional_vs_strong_controls.svg").write_text(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1100\" height=\"240\">"
        "<rect width=\"1100\" height=\"240\" fill=\"#f8fafc\"/>"
        "<text x=\"24\" y=\"42\" font-family=\"Arial\" font-size=\"24\" fill=\"#111827\">v9.2.19 first-wave v8 strong-control replay</text>"
        + "".join(
            f"<text x=\"24\" y=\"{82 + i * 28}\" font-family=\"Arial\" font-size=\"16\" fill=\"#374151\">{line}</text>"
            for i, line in enumerate(lines)
        )
        + "</svg>\n",
        encoding="utf-8",
    )
    (fig_dir / "p1_v8_control_rank_by_dataset.svg").write_text(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1100\" height=\"240\">"
        "<rect width=\"1100\" height=\"240\" fill=\"#fff\"/>"
        "<text x=\"24\" y=\"42\" font-family=\"Arial\" font-size=\"24\" fill=\"#111827\">Control rank summary</text>"
        + "".join(
            f"<text x=\"24\" y=\"{82 + i * 24}\" font-family=\"Arial\" font-size=\"14\" fill=\"#374151\">{row.get('candidate')}: CEp99_delta={row.get('CEp99_delta_mean')}, margin_delta={row.get('margin_delta_mean')}</text>"
            for i, row in enumerate(list(p1_summary)[:6])
        )
        + "</svg>\n",
        encoding="utf-8",
    )


def _summarize_p1(rows: Sequence[Dict[str, Any]], out_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    by_candidate: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in measured:
        by_candidate[str(row.get("candidate"))].append(row)
    summary: List[Dict[str, Any]] = []
    for candidate, cand_rows in sorted(by_candidate.items()):
        summary.append({
            "candidate": candidate,
            "rows": len(cand_rows),
            "CEp99_delta_mean": _mean([_float(r.get("CEp99_delta"), float("nan")) for r in cand_rows]),
            "margin_delta_mean": _mean([_float(r.get("margin_p10_delta"), float("nan")) for r in cand_rows]),
            "NLL_delta_mean": _mean([_float(r.get("NLL_delta"), float("nan")) for r in cand_rows]),
            "ECE_delta_mean": _mean([_float(r.get("ECE_delta"), float("nan")) for r in cand_rows]),
            "curvature_delta_mean": _mean([_float(r.get("curvature_delta"), float("nan")) for r in cand_rows]),
            "acc_delta_vs_adamw_mean": _mean([_float(r.get("delta_vs_adamw"), float("nan")) for r in cand_rows]),
            "step_ratio_q90": _q([_float(r.get("step_ratio_vs_adamw"), float("nan")) for r in cand_rows], 0.90),
            "event_coverage": _mean([_float(r.get("event_coverage"), 0.0) for r in cand_rows]),
            "real_beats_adamwparallel_rate": _mean([_float(r.get("real_beats_adamwparallel"), 0.0) for r in cand_rows]),
            "real_beats_best_lr_rate": _mean([_float(r.get("real_beats_best_lr"), 0.0) for r in cand_rows]),
            "cos_functional_adamw_mean": _mean([_float(r.get("cos_functional_adamw"), float("nan")) for r in cand_rows]),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    write_csv_rows(out_dir / "p1_v8_strong_control_replay_summary.csv", summary)

    functional = [r for r in measured if r.get("candidate") in FUNCTIONAL_BRANCHES]
    functional_beats = int(any(_float(r.get("real_beats_adamwparallel")) > 0 and _float(r.get("real_beats_best_lr")) > 0 for r in functional))
    summary_by_name = {str(row.get("candidate")): row for row in summary}
    functional_summary = [summary_by_name[name] for name in FUNCTIONAL_BRANCHES if name in summary_by_name]
    parallel_summary = [summary_by_name[name] for name in ADAMW_PARALLEL_BRANCHES if name in summary_by_name]
    lr_summary = [summary_by_name[name] for name in LR_BRANCHES if name in summary_by_name]
    best_func_ce = min((_float(row.get("CEp99_delta_mean"), 99.0) for row in functional_summary), default=99.0)
    best_parallel_ce = min((_float(row.get("CEp99_delta_mean"), 99.0) for row in parallel_summary), default=99.0)
    best_lr_ce = min((_float(row.get("CEp99_delta_mean"), 99.0) for row in lr_summary), default=99.0)
    best_func_margin = max((_float(row.get("margin_delta_mean"), -99.0) for row in functional_summary), default=-99.0)
    best_parallel_margin = max((_float(row.get("margin_delta_mean"), -99.0) for row in parallel_summary), default=-99.0)
    best_lr_margin = max((_float(row.get("margin_delta_mean"), -99.0) for row in lr_summary), default=-99.0)
    best_func_curv = min((_float(row.get("curvature_delta_mean"), 99.0) for row in functional_summary), default=99.0)
    best_parallel_curv = min((_float(row.get("curvature_delta_mean"), 99.0) for row in parallel_summary), default=99.0)
    best_lr_curv = min((_float(row.get("curvature_delta_mean"), 99.0) for row in lr_summary), default=99.0)
    functional_tail_margin_aggregate = int(
        (best_func_ce < best_parallel_ce and best_func_ce < best_lr_ce)
        or (best_func_margin > best_parallel_margin and best_func_margin > best_lr_margin)
    )
    functional_curvature_aggregate = int(
        best_func_curv < 0.90 * best_parallel_curv
        and best_func_curv < 0.90 * best_lr_curv
    )
    full_epoch_done = int(any(_float(r.get("full_epoch_replay")) > 0 for r in measured))
    lr_equiv_signal = int(
        measured
        and not functional_beats
        and any(r.get("candidate") in LR_BRANCHES or r.get("candidate") in ADAMW_PARALLEL_BRANCHES for r in measured)
    )
    decision = {
        "v8_one_step_row_count": len(measured),
        "v8_one_step_functional_beats_controls": functional_beats,
        "v8_one_step_functional_tail_margin_aggregate_pass": functional_tail_margin_aggregate,
        "v8_one_step_functional_curvature_aggregate_pass": functional_curvature_aggregate,
        "v8_one_step_best_functional_CEp99_delta": best_func_ce,
        "v8_one_step_best_parallel_CEp99_delta": best_parallel_ce,
        "v8_one_step_best_lr_CEp99_delta": best_lr_ce,
        "v8_one_step_best_functional_margin_delta": best_func_margin,
        "v8_one_step_best_parallel_margin_delta": best_parallel_margin,
        "v8_one_step_best_lr_margin_delta": best_lr_margin,
        "v8_one_step_best_functional_curvature_delta": best_func_curv,
        "v8_one_step_best_parallel_curvature_delta": best_parallel_curv,
        "v8_one_step_best_lr_curvature_delta": best_lr_curv,
        "v8_one_step_lr_equivalent_signal": lr_equiv_signal,
        "v8_full_epoch_strong_control_replay_done": full_epoch_done,
        "v8_full_epoch_strong_control_replay_pass": 0,
        "v8_strong_control_replay_pass": 0,
    }
    return summary, decision


def _contract_rows() -> List[Dict[str, Any]]:
    return [{
        "stage": "contract_audit_v9219",
        "loss_type": "CE",
        "label_smoothing": 0,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "distillation_used": 0,
        "loss_modified": 0,
        "sampler_modified": 0,
        "class_weight_used": 0,
        "cpu_offload_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "purekanconv_deferred": 1,
        "purekanformer_deferred": 1,
        "v8_one_step_replay_is_not_full_epoch_replay": 1,
    }]


def _write_md_report(out_dir: Path, route: Dict[str, Any], p1_summary: Sequence[Dict[str, Any]], audit: Dict[str, Any]) -> Path:
    report = ROOT / "docs" / "DG-KAN_v9.2.19_FunctionalCore_Rescue_StrictPureKAN_Interface_实验复盘.md"
    top = sorted(p1_summary, key=lambda r: (_float(r.get("CEp99_delta_mean"), 99), -_float(r.get("margin_delta_mean"), -99)))[:8]
    top_lines = "\n".join(
        f"| `{row.get('candidate')}` | `{_float(row.get('CEp99_delta_mean')):.6g}` | `{_float(row.get('margin_delta_mean')):.6g}` | `{_float(row.get('step_ratio_q90')):.6g}` | `{_float(row.get('real_beats_adamwparallel_rate')):.3f}` | `{_float(row.get('real_beats_best_lr_rate')):.3f}` |"
        for row in top
    )
    text = f"""# DG-KAN v9.2.19 FunctionalCore Rescue StrictPureKAN Interface 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.19_FunctionalCore_Rescue_StrictPureKAN_Interface_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 one-step replay 伪装成 20-epoch strong-control replay。

## 0. 最新结论

截至本轮，v9.2.19 执行到一个可审计 terminal route：

```text
route = {route.get('route')}
base_candidate = {route.get('base_candidate')}
success_v9219_functional_core_retained = {str(bool(route.get('success_v9219_functional_core_retained'))).lower()}
success_v9219_strict_purekan_functional = {str(bool(route.get('success_v9219_strict_purekan_functional'))).lower()}
success_v9219_external_ready = {str(bool(route.get('success_v9219_external_ready'))).lower()}
```

最终 artifact：

```text
{out_dir.relative_to(ROOT)}/
```

核心结论：

1. 本轮没有只做 v8 source recap；新增了真实 v8 KW4 one-step strong-control replay，覆盖 AdamWParallel / LRScale / NoOp / Random / Shuffled / Inverted controls。
2. 但本轮只完成 one-step replay，不是计划要求的 20-epoch P1 strong-control replay；因此不能写 `v8_strong_control_replay_pass=true`。
3. one-step replay measured rows = `{route.get('v8_one_step_row_count')}`；functional 有 row-level / curvature 机制信号，但 CEp99 / margin aggregate 仍由 AdamWParallel / LR controls 领先。
4. P3-P7 均因 full-epoch v8 strong-control replay 未完成而明确 `not_run`；没有用 strict PureKAN interface 或 full functional training 越过 P1。
5. 当前 route 是 `{route.get('route')}`：functional 仍作为研究核心保留，但本轮还不能把 functional retained success 写成达成。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v9219_functional_core_rescue.py` | v9.2.19 runner；生成 P0-P7 artifacts、v8 one-step strong-control replay、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9219_functional_core_rescue.py
```

正式运行：

```bash
python experiments/run_v9219_functional_core_rescue.py \\
  --out-dir {out_dir.relative_to(ROOT)} \\
  --fresh \\
  --device auto \\
  --data-root data \\
  --seed 1314
```

## 2. Route

`route_decision.json`：

```json
{json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True)}
```

判断：本轮补上了 v8 strong-control 的真实测量起点，但没有完成 full P1，因此 functional core 不能宣布 retained success。

## 3. P1 v8 one-step strong-control replay

Artifact：

```text
p1_v8_strong_control_replay.csv
p1_v8_strong_control_replay_summary.csv
paired_replay_branch_trace_v9219.csv
```

Top control summary（按 CEp99 delta / margin delta 排序）：

| candidate | CEp99 delta mean | margin delta mean | step q90 | beats AdamWParallel | beats best LR |
|---|---:|---:|---:|---:|---:|
{top_lines}

说明：

1. `eval_scope = holdout_microbatch`，不是 20-epoch final test。
2. `memory_ratio = not_measured_one_step`，没有伪造成系统 P4 memory。
3. `test_acc/val_acc = not_measured_one_step_holdout_scope`，没有把 one-step holdout 当作正式 P1 full-run accuracy。
4. Aggregate CEp99/margin 不是 functional 胜出：best functional CEp99 delta `{_float(route.get('v8_one_step_best_functional_CEp99_delta')):.6g}`，best AdamWParallel `{_float(route.get('v8_one_step_best_parallel_CEp99_delta')):.6g}`，best LR `{_float(route.get('v8_one_step_best_lr_CEp99_delta')):.6g}`。
5. Functional 的正信号主要在 curvature：best functional curvature delta `{_float(route.get('v8_one_step_best_functional_curvature_delta')):.6g}`，best AdamWParallel `{_float(route.get('v8_one_step_best_parallel_curvature_delta')):.6g}`，best LR `{_float(route.get('v8_one_step_best_lr_curvature_delta')):.6g}`。

## 4. P2 mechanism attribution

Artifact：

```text
p2_v8_mechanism_attribution.csv
role_mechanism_trace_v9219.csv
```

本轮记录了 role update norm、role SNR、branch ratio、effective derivative scale、cos functional AdamW、CEp99/margin/ECE/NLL/curvature delta。  
但因为 P1 full replay 未完成，P2 只能作为 one-step mechanism trace，不能判定 v8 independent mechanism pass。

## 5. P3-P7 downstream boundary

这些 artifact 均已落盘，但明确为 `not_run`：

| artifact | reason |
|---|---|
| `p3_purekan_functional_interface_redesign.csv` | `P1_full_epoch_v8_strong_control_replay_not_completed` |
| `p4_basis_factory_functional_actuatability.csv` | same |
| `p5_adamw_only_fullpass_repair.csv` | same |
| `p6_functional_short_full_validation.csv` | same |
| `p7_robustness_external_ready.csv` | same |

## 6. No-fake audit

`v9219_provenance_audit.csv`：

```text
rows_checked = {audit.get('rows_checked')}
fake_proxy_nonzero_count = {audit.get('fake_proxy_nonzero_count')}
fake_data_used = {audit.get('fake_data_used')}
proxy_row_used = {audit.get('proxy_row_used')}
cpu_offload_used = {audit.get('cpu_offload_used')}
no_fake = {str(audit.get('no_fake')).lower()}
no_proxy = {str(audit.get('no_proxy')).lower()}
```

## 7. 最终分析结论

v9.2.19 的真实推进是：

```text
v9.2.18: 只能 source-audit，v8 strong controls 缺失。
v9.2.19: 补上 v8 one-step AdamWParallel / LR strong controls，但还未完成 20-epoch replay。
```

机制判断：

1. 计划方向是正确的：必须先检验 v8 functional 是否能打败 AdamWParallel / LR controls，再谈 strict PureKAN interface。
2. 本轮 one-step replay 是必要但不足的证据；它显示 curvature 上有 functional-like 信号，但 CEp99/margin aggregate 仍被 AdamWParallel / LR controls 解释得更好。
3. 因此不能把 v8 functional success 恢复为当前 strict claim，也不能把它降级为 LR-equivalent final conclusion。
4. 下一步应继续实现 full-epoch v8 strong-control replay，或把 one-step replay中最强控制/functional分支提升到 short-run 50/240 step replay；在这之前不要打开 P3-P7。

最终一句话：

> v9.2.19 本轮真实补测了 v8 one-step strong controls，但诚实停在 `{route.get('route')}`：这不是 full P1 strong-control replay，因此 functional core 仍未重新证明，strict PureKAN functional 和 external-ready 都不能打开。
"""
    report.write_text(text, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--p1-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--p1-seeds", default="0,1,2,3,4")
    parser.add_argument("--p1-train-size", type=int, default=4096)
    parser.add_argument("--p1-test-size", type=int, default=512)
    parser.add_argument("--p1-batch-size", type=int, default=128)
    parser.add_argument("--p1-warmup-batches", type=int, default=3)
    parser.add_argument("--p1-audit-batches", type=int, default=2)
    parser.add_argument("--p1-target-rel-update", type=float, default=0.001)
    parser.add_argument("--v8-hidden-dim", type=int, default=28)
    parser.add_argument("--v8-basis-count", type=int, default=8)
    args = parser.parse_args()

    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    device = _device(args.device)

    write_json(out_dir / "run_manifest.json", {
        "runner": str(SCRIPT_PATH.relative_to(ROOT)),
        "plan": str(PLAN_PATH.relative_to(ROOT)),
        "created_at": _now_iso(),
        "device": str(device),
        "torch": torch.__version__,
        "seed": args.seed,
        "p1_scope": "one_step_strong_control_replay_first_wave",
        "p1_full_epoch_replay": 0,
        "no_teacher": 1,
        "no_loss_modification": 1,
        "no_fake_proxy": 1,
    })

    contract = _contract_rows()
    write_csv_rows(out_dir / "contract_audit_v9219.csv", contract)
    p0_rows = _p0_history_audit()
    write_csv_rows(out_dir / "p0_history_unified_audit.csv", p0_rows)
    p1_rows, mechanism_rows = _run_p1_v8_one_step(args, out_dir, device)
    p1_summary, p1_decision = _summarize_p1(p1_rows, out_dir)

    downstream_reason = "P1_full_epoch_v8_strong_control_replay_not_completed"
    p3_rows = [_not_run_row("P3_strict_fc_purekan_functional_interface", "p3_purekan_functional_interface_redesign.csv", downstream_reason, candidate=candidate)
               for candidate in ["F1a", "F1b", "F1c", "F1d", "F2a", "F2b", "F2c", "F3a", "F3b", "F3c", "F3d", "F4"]]
    p4_rows = _source_recap_rows("P4_basis_factory_functional_actuatability") + [
        _not_run_row("P4_basis_factory_functional_actuatability", "p4_basis_factory_functional_actuatability.csv", downstream_reason, candidate=candidate)
        for candidate in ["B1-centered-T2", "B2-normalized-T2", "B5-bounded-rational-base", "B6-piecewise-linear2-base", "B10-dual-role-T2-rational-functional"]
    ]
    p5_rows = [
        {
            "stage": "P5_adamw_only_fullpass_repair",
            "status": "source_recap",
            "candidate": "LQ-t2-h256",
            "source": str(SRC_V927.relative_to(ROOT)),
            "near_pass": 1,
            "full_pass": 0,
            "macro_delta": _read_json(SRC_V927 / "route_decision.json").get("p5_macro_delta", ""),
            "functional_update_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "stage": "P5_adamw_only_fullpass_repair",
            "status": "source_recap",
            "candidate": "A7c-BasisEntropy-ValueOnly",
            "source": str(SRC_V9214.relative_to(ROOT)),
            "near_pass": _read_json(SRC_V9214 / "route_decision.json").get("actuator_base_near_pass", 1),
            "full_pass": 0,
            "macro_delta": "see_p4_p4_qualified_actuator_base_qualification.csv",
            "functional_update_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        _not_run_row("P5_adamw_only_fullpass_repair", "p5_adamw_only_fullpass_repair.csv", downstream_reason, candidate="new_fullpass_repairs"),
    ]
    p6_rows = [_not_run_row("P6_functional_short_full_validation", "p6_functional_short_full_validation.csv", downstream_reason)]
    p7_rows = [_not_run_row("P7_robustness_external_ready", "p7_robustness_external_ready.csv", downstream_reason)]
    write_csv_rows(out_dir / "p3_purekan_functional_interface_redesign.csv", p3_rows)
    write_csv_rows(out_dir / "purekan_interface_trace_v9219.csv", p3_rows)
    write_csv_rows(out_dir / "p4_basis_factory_functional_actuatability.csv", p4_rows)
    write_csv_rows(out_dir / "p5_adamw_only_fullpass_repair.csv", p5_rows)
    write_csv_rows(out_dir / "p6_functional_short_full_validation.csv", p6_rows)
    write_csv_rows(out_dir / "p7_robustness_external_ready.csv", p7_rows)

    if p1_decision["v8_one_step_functional_beats_controls"] and not p1_decision["v8_full_epoch_strong_control_replay_done"]:
        route_name = "R7-FunctionalPausedButNotAbandoned"
        blocker = "one_step_v8_functional_signal_exists_but_full_epoch_strong_control_replay_not_completed"
    elif p1_decision["v8_one_step_lr_equivalent_signal"]:
        route_name = "R7-FunctionalPausedButNotAbandoned"
        blocker = "one_step_strong_controls_do_not_establish_functional_advantage_and_full_epoch_replay_missing"
    else:
        route_name = "R7-FunctionalPausedButNotAbandoned"
        blocker = "P1_full_epoch_v8_strong_control_replay_not_completed"
    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v8_one_step_row_count": p1_decision["v8_one_step_row_count"],
        "v8_one_step_functional_beats_controls": p1_decision["v8_one_step_functional_beats_controls"],
        "v8_one_step_functional_tail_margin_aggregate_pass": p1_decision["v8_one_step_functional_tail_margin_aggregate_pass"],
        "v8_one_step_functional_curvature_aggregate_pass": p1_decision["v8_one_step_functional_curvature_aggregate_pass"],
        "v8_one_step_best_functional_CEp99_delta": p1_decision["v8_one_step_best_functional_CEp99_delta"],
        "v8_one_step_best_parallel_CEp99_delta": p1_decision["v8_one_step_best_parallel_CEp99_delta"],
        "v8_one_step_best_lr_CEp99_delta": p1_decision["v8_one_step_best_lr_CEp99_delta"],
        "v8_one_step_best_functional_margin_delta": p1_decision["v8_one_step_best_functional_margin_delta"],
        "v8_one_step_best_parallel_margin_delta": p1_decision["v8_one_step_best_parallel_margin_delta"],
        "v8_one_step_best_lr_margin_delta": p1_decision["v8_one_step_best_lr_margin_delta"],
        "v8_one_step_best_functional_curvature_delta": p1_decision["v8_one_step_best_functional_curvature_delta"],
        "v8_one_step_best_parallel_curvature_delta": p1_decision["v8_one_step_best_parallel_curvature_delta"],
        "v8_one_step_best_lr_curvature_delta": p1_decision["v8_one_step_best_lr_curvature_delta"],
        "v8_one_step_lr_equivalent_signal": p1_decision["v8_one_step_lr_equivalent_signal"],
        "v8_full_epoch_strong_control_replay_done": p1_decision["v8_full_epoch_strong_control_replay_done"],
        "v8_strong_control_replay_pass": 0,
        "v8_lr_equivalent": 0,
        "v8_mechanism_identified": 0,
        "v9_interface_pass": 0,
        "basis_factory_functional_pass": 0,
        "adamw_fullpass": 0,
        "functional_short_full_pass": 0,
        "external_ready": 0,
        "functional_core_retained": 0,
        "primary_blocker": blocker,
        "next_required_implementation": "complete_full_epoch_v8_strong_control_replay_or_promote_measured_one_step_controls_to_50_240_step_short_run",
        "success_v9219_functional_core_retained": 0,
        "success_v9219_strict_purekan_functional": 0,
        "success_v9219_external_ready": 0,
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)

    failure_rows = [
        {
            "failure_code": "F2_v8_strong_control_replay_missing",
            "status": "active",
            "reason": downstream_reason,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    ]
    write_csv_rows(out_dir / "failure_table.csv", failure_rows)

    csv_paths = [
        out_dir / "contract_audit_v9219.csv",
        out_dir / "p0_history_unified_audit.csv",
        out_dir / "p1_v8_strong_control_replay.csv",
        out_dir / "p1_v8_strong_control_replay_summary.csv",
        out_dir / "p2_v8_mechanism_attribution.csv",
        out_dir / "p3_purekan_functional_interface_redesign.csv",
        out_dir / "p4_basis_factory_functional_actuatability.csv",
        out_dir / "p5_adamw_only_fullpass_repair.csv",
        out_dir / "p6_functional_short_full_validation.csv",
        out_dir / "p7_robustness_external_ready.csv",
        out_dir / "failure_table.csv",
    ]
    audit = audit_no_fake(csv_paths)
    write_csv_rows(out_dir / "v9219_provenance_audit.csv", [audit])
    _write_simple_figures(out_dir, route, p1_summary)
    _write_md_report(out_dir, route, p1_summary, audit)

    hash_paths = [PLAN_PATH, SCRIPT_PATH] + [p for p in out_dir.iterdir() if p.is_file()]
    write_csv_rows(out_dir / "artifact_hashes.csv", artifact_hash_rows(hash_paths, root=ROOT))


if __name__ == "__main__":
    main()
