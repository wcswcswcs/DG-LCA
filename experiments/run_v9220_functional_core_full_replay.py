#!/usr/bin/env python3
"""DG-KAN v9.2.20 functional-core full replay runner.

This runner promotes the v9.2.19 v8 one-step strong-control replay to short-run
multi-step replay.  P2 full replay is opened only if P1 produces a measured
short-run survivor.  Any stage not opened by the gate is written as not_run.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import run_v9219_functional_core_rescue as v9219  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.20_FunctionalCore_FullReplay_StrictPureKAN_Interface_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9220_functional_core_full_replay.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9219 = RESULT_ROOT / "v9219_functional_core_rescue_strictpurekan_interface_first_20260510T080000Z"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_csv_list(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _parse_steps(text: str) -> List[int]:
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


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _not_run(stage: str, artifact: str, reason: str, **extra: Any) -> Dict[str, Any]:
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


def _p0_boundary() -> List[Dict[str, Any]]:
    route = _read_json(SRC_V9219 / "route_decision.json")
    audit = read_csv_rows(SRC_V9219 / "v9219_provenance_audit.csv")
    fake_proxy = audit[0].get("fake_proxy_nonzero_count", 0) if audit else "missing"
    row = {
        "stage": "P0_v9219_boundary_reproduction",
        "status": "source_recap",
        "source": str(SRC_V9219.relative_to(ROOT)),
        "route": route.get("route", ""),
        "base_candidate": route.get("base_candidate", ""),
        "v8_one_step_row_count": route.get("v8_one_step_row_count", ""),
        "v8_one_step_best_functional_CEp99_delta": route.get("v8_one_step_best_functional_CEp99_delta", ""),
        "v8_one_step_best_functional_margin_delta": route.get("v8_one_step_best_functional_margin_delta", ""),
        "v8_one_step_best_functional_curvature_delta": route.get("v8_one_step_best_functional_curvature_delta", ""),
        "v8_one_step_best_parallel_CEp99_delta": route.get("v8_one_step_best_parallel_CEp99_delta", ""),
        "v8_one_step_best_parallel_margin_delta": route.get("v8_one_step_best_parallel_margin_delta", ""),
        "v8_one_step_best_parallel_curvature_delta": route.get("v8_one_step_best_parallel_curvature_delta", ""),
        "v8_one_step_best_lr_CEp99_delta": route.get("v8_one_step_best_lr_CEp99_delta", ""),
        "v8_one_step_best_lr_margin_delta": route.get("v8_one_step_best_lr_margin_delta", ""),
        "v8_one_step_best_lr_curvature_delta": route.get("v8_one_step_best_lr_curvature_delta", ""),
        "fake_proxy_count": fake_proxy,
        "p0_pass": int(route.get("route") == "R7-FunctionalPausedButNotAbandoned" and int(route.get("v8_one_step_row_count", 0)) == 450 and str(fake_proxy) == "0"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row]


def _warmup_event_threshold(
    stack: Any,
    head: Any,
    spec: Any,
    entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]],
    snapshot: Sequence[torch.Tensor],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    batch_size: int,
    warmup_batches: int,
    target_rel_update: float,
    device: torch.device,
) -> float:
    n = int(x_train.shape[0])
    holdout_offset = max(batch_size * (int(warmup_batches) + 8), batch_size)
    scores: List[float] = []
    for warm_idx in range(int(warmup_batches)):
        idx = torch.arange(warm_idx * batch_size, (warm_idx + 1) * batch_size, device=device) % n
        hid = torch.arange(holdout_offset + warm_idx * batch_size, holdout_offset + (warm_idx + 1) * batch_size, device=device) % n
        xb, yb = x_train[idx], y_train[idx]
        xh, yh = x_train[hid], y_train[hid]
        v9219.v87.v85.v83._restore_params(entries, snapshot)
        _loss, task_dirs = v9219.v87.v85.v83._manual_task_grads(stack, head, xb, yb, spec)
        param_norm = v9219.v87.v85.v83._param_norm(entries)
        task_norm = v9219.v87.v85.v83._norm(task_dirs)
        task_alpha = float(target_rel_update) * param_norm / max(task_norm, 1.0e-30)
        holdout_before = v9219.v87.v85.v83._loss_only(stack, head, xh, yh, spec)
        smooth_before, curv_before = v9219.v87.v85.v83._geometry_norms(entries)
        v9219.v87.v85.v83._apply_direction(entries, task_dirs, task_alpha)
        task_holdout_after = v9219.v87.v85.v83._loss_only(stack, head, xh, yh, spec)
        v9219.v87.v85.v83._restore_params(entries, snapshot)
        score, _td, _gp, _br = v9219.v87._adaptive_event_score(
            holdout_loss_before=holdout_before,
            holdout_task_after=task_holdout_after,
            smooth_before=smooth_before,
            curv_before=curv_before,
        )
        scores.append(score)
    return _q(scores, 0.75) if scores else 0.0


def _train_branch_replay(
    *,
    branch: str,
    dataset: str,
    seed: int,
    total_steps: int,
    args: argparse.Namespace,
    device: torch.device,
    stage: str,
    full_epoch_replay: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    stack, head, spec, x_train, y_train, x_test, y_test, protocol = v9219._build_v8_model(
        dataset=dataset,
        seed=seed,
        hidden_dim=int(args.v8_hidden_dim),
        basis=int(args.v8_basis_count),
        device=device,
        data_root=Path(args.data_root),
        train_size=int(args.train_size),
        test_size=int(args.eval_size),
    )
    batch_size = int(args.batch_size)
    entries = v9219.v87.v85.v83._param_entries(stack, head)
    snapshot = v9219.v87.v85.v83._clone_params(entries)
    threshold = _warmup_event_threshold(
        stack,
        head,
        spec,
        entries,
        snapshot,
        x_train,
        y_train,
        batch_size=batch_size,
        warmup_batches=int(args.warmup_batches),
        target_rel_update=float(args.target_rel_update),
        device=device,
    )
    v9219.v87.v85.v83._restore_params(entries, snapshot)
    x_eval = x_test[: int(args.eval_size)]
    y_eval = y_test[: int(args.eval_size)]
    before = v9219._eval_metrics(stack, head, x_eval, y_eval, batch_size)
    train_holdout_x = x_train[-batch_size:]
    train_holdout_y = y_train[-batch_size:]
    holdout_before = v9219.v87.v85.v83._loss_only(stack, head, train_holdout_x, train_holdout_y, spec)
    smooth_before, curv_before = v9219.v87.v85.v83._geometry_norms(entries)
    n = int(x_train.shape[0])
    event_count = 0
    branch_ratios: List[float] = []
    derivative_scales: List[float] = []
    cos_vals: List[float] = []
    step_ms: List[float] = []
    role_rows: List[Dict[str, Any]] = []
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(int(total_steps)):
        idx = torch.arange(step * batch_size, (step + 1) * batch_size, device=device) % n
        hid = torch.arange(n - batch_size - ((step % 4) * batch_size), n - ((step % 4) * batch_size), device=device) % n
        xb, yb = x_train[idx], y_train[idx]
        xh, yh = x_train[hid], y_train[hid]
        entries = v9219.v87.v85.v83._param_entries(stack, head)
        loss_before, task_dirs = v9219.v87.v85.v83._manual_task_grads(stack, head, xb, yb, spec)
        param_norm = v9219.v87.v85.v83._param_norm(entries)
        task_norm = v9219.v87.v85.v83._norm(task_dirs)
        task_alpha = float(args.target_rel_update) * param_norm / max(task_norm, 1.0e-30)
        holdout_loss_before = v9219.v87.v85.v83._loss_only(stack, head, xh, yh, spec)
        smooth_step, curv_step = v9219.v87.v85.v83._geometry_norms(entries)
        step_snapshot = v9219.v87.v85.v83._clone_params(entries)
        v9219.v87.v85.v83._apply_direction(entries, task_dirs, task_alpha)
        task_holdout_after = v9219.v87.v85.v83._loss_only(stack, head, xh, yh, spec)
        v9219.v87.v85.v83._restore_params(entries, step_snapshot)
        event_score, _tds, _gp, bad_step_risk = v9219.v87._adaptive_event_score(
            holdout_loss_before=holdout_loss_before,
            holdout_task_after=task_holdout_after,
            smooth_before=smooth_step,
            curv_before=curv_step,
        )
        ft7_reference = v9219.v87._adaptive_candidate_direction(
            "Fixed-FT7-stride128",
            entries,
            task_dirs,
            event_score=event_score,
            event_threshold=threshold,
            bad_step_risk=bad_step_risk,
            task_holdout_descent=holdout_loss_before - task_holdout_after,
        )
        (
            direction,
            func_dirs,
            role_weights,
            _role_budgets,
            event_triggered,
            _total_budget,
            _trust_scale,
            _clip_rate,
            update_rule,
            alpha_multiplier,
            alpha_mode,
        ) = v9219._candidate_direction(
            branch,
            entries,
            task_dirs,
            event_score=event_score,
            event_threshold=threshold,
            bad_step_risk=bad_step_risk,
            task_holdout_descent=holdout_loss_before - task_holdout_after,
            ft7_reference=ft7_reference,
        )
        direction_norm = v9219.v87.v85.v83._norm(direction)
        if alpha_mode == "alpha_task_scaled":
            alpha = task_alpha * float(alpha_multiplier)
        elif alpha_mode == "alpha_task":
            alpha = task_alpha
        else:
            alpha = float(args.target_rel_update) * param_norm / max(direction_norm, 1.0e-30)
        func_delta = v9219._direction_delta(direction, alpha, task_dirs, task_alpha)
        branch_ratio = v9219.v87.v85.v83._norm(func_delta) / max(param_norm, 1.0e-30)
        role_func_norm = v9219._role_norms(entries, func_dirs)
        role_update_norm = v9219._role_norms(entries, func_delta)
        role_snr = v9219._role_snr(entries, func_delta)
        derivative_scale = abs(alpha) * max(float(role_func_norm.get("stack", 0.0)), float(role_func_norm.get("head", 0.0)))
        cos_val = v9219.v87.v85.v83._cos(func_delta, [t * task_alpha for t in task_dirs])
        v9219._sync(device)
        started = time.perf_counter()
        v9219.v87.v85.v83._apply_direction(entries, direction, alpha)
        v9219._sync(device)
        step_ms.append((time.perf_counter() - started) * 1000.0)
        event_count += int(event_triggered)
        branch_ratios.append(branch_ratio)
        derivative_scales.append(derivative_scale)
        cos_vals.append(cos_val)
        if step in {0, int(total_steps) - 1}:
            for role in ("stack", "head"):
                role_rows.append({
                    "stage": stage.replace("P1", "P3").replace("P2", "P3"),
                    "status": "measured",
                    "candidate": branch,
                    "dataset": dataset,
                    "seed": seed,
                    "steps": total_steps,
                    "step": step,
                    "role": role,
                    "role_update_norm": role_update_norm.get(role, 0.0),
                    "role_snr": role_snr.get(role, 0.0),
                    "role_curvature": role_func_norm.get(role, 0.0),
                    "role_event_frequency": int(event_triggered),
                    "branch_ratio": branch_ratio,
                    "effective_derivative_scale": derivative_scale,
                    "cos_functional_adamw": cos_val,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
    after = v9219._eval_metrics(stack, head, x_eval, y_eval, batch_size)
    holdout_after = v9219.v87.v85.v83._loss_only(stack, head, train_holdout_x, train_holdout_y, spec)
    smooth_after, curv_after = v9219.v87.v85.v83._geometry_norms(v9219.v87.v85.v83._param_entries(stack, head))
    peak_mb = ""
    if torch.cuda.is_available() and device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    row = {
        "stage": stage,
        "status": "measured",
        "candidate": branch,
        "dataset": dataset,
        "seed": seed,
        "steps": total_steps,
        "epochs_measured": int(full_epoch_replay) * int(args.p2_epochs),
        "full_epoch_replay": full_epoch_replay,
        "protocol": protocol,
        "optimizer_protocol": "manual_relative_update_replay_no_torch_loss_backward",
        "same_initialization_across_branches": 1,
        "same_batch_sequence_across_branches": 1,
        "eval_scope": f"test_prefix_{int(args.eval_size)}",
        "test_acc_proxy": after["acc"],
        "test_acc": after["acc"] if full_epoch_replay else "not_full_epoch_test_proxy",
        "val_acc": "not_split",
        "delta_vs_adamw": "computed_in_summary",
        "delta_vs_mlp_match": "not_measured_v8_replay",
        "holdout_loss_before": holdout_before,
        "holdout_loss": holdout_after,
        "holdout_loss_delta": holdout_after - holdout_before,
        "CEp99_before": before["CEp99"],
        "CEp99": after["CEp99"],
        "CEp99_delta": after["CEp99"] - before["CEp99"],
        "margin_p10_before": before["margin_p10"],
        "margin_p10": after["margin_p10"],
        "margin_delta": after["margin_p10"] - before["margin_p10"],
        "wrong_confidence_p95": after["wrong_confidence_p95"],
        "ECE_proxy": after["ECE"],
        "ECE_delta": after["ECE"] - before["ECE"],
        "NLL_proxy": after["NLL"],
        "NLL_delta": after["NLL"] - before["NLL"],
        "curvature_before": curv_before,
        "curvature": curv_after,
        "curvature_delta": curv_after - curv_before,
        "local_lipschitz": "not_measured_multistep",
        "functional_event_count": event_count,
        "event_coverage": event_count / max(1, int(total_steps)),
        "branch_ratio": _mean(branch_ratios),
        "effective_derivative_scale": _mean(derivative_scales),
        "cos_functional_adamw": _mean(cos_vals),
        "step_ratio_q90": "computed_in_summary",
        "step_time_q90_ms": _q(step_ms, 0.90),
        "memory_peak_mb": peak_mb,
        "memory_ratio": "computed_in_summary",
        "control_rank": "computed_in_summary",
        "beats_adamwparallel": "computed_in_summary",
        "beats_best_lr": "computed_in_summary",
        "update_rule": update_rule,
        "role_weights": json.dumps(role_weights, sort_keys=True),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    del stack, head, x_train, y_train, x_test, y_test
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()
    return row, role_rows


def _finalize_group_rows(rows: List[Dict[str, Any]]) -> None:
    groups: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("dataset")), int(row.get("seed", 0)), int(row.get("steps", 0)))].append(row)
    for group_rows in groups.values():
        adamw = next((r for r in group_rows if r.get("candidate") == "V8-B0-AdamWOnly"), None)
        lr_rows = [r for r in group_rows if str(r.get("candidate")) in v9219.LR_BRANCHES]
        parallel_rows = [r for r in group_rows if str(r.get("candidate")) in v9219.ADAMW_PARALLEL_BRANCHES]
        baseline_step = _q([_float(r.get("step_time_q90_ms"), float("nan")) for r in group_rows if r.get("candidate") == "V8-B0-AdamWOnly"], 0.90)
        baseline_mem = _mean([_float(r.get("memory_peak_mb"), float("nan")) for r in group_rows if r.get("candidate") == "V8-B0-AdamWOnly"])
        best_lr = min(lr_rows, key=lambda r: (_float(r.get("CEp99_delta"), 99), -_float(r.get("margin_delta"), -99)), default=None)
        best_parallel = min(parallel_rows, key=lambda r: (_float(r.get("CEp99_delta"), 99), -_float(r.get("margin_delta"), -99)), default=None)
        ordered = sorted(group_rows, key=lambda r: (_float(r.get("CEp99_delta"), 99), -_float(r.get("margin_delta"), -99)))
        ranks = {id(row): idx + 1 for idx, row in enumerate(ordered)}
        for row in group_rows:
            row["control_rank"] = ranks[id(row)]
            row["step_ratio_q90"] = _float(row.get("step_time_q90_ms"), 0.0) / max(baseline_step, 1.0e-12)
            row["memory_ratio"] = _float(row.get("memory_peak_mb"), 0.0) / max(baseline_mem, 1.0e-12) if baseline_mem and math.isfinite(baseline_mem) else "not_measured"
            if adamw:
                row["delta_vs_adamw"] = _float(row.get("test_acc_proxy")) - _float(adamw.get("test_acc_proxy"))
            if row.get("candidate") in v9219.FUNCTIONAL_BRANCHES and best_lr and best_parallel:
                beats_parallel = int(
                    _float(row.get("CEp99_delta")) < _float(best_parallel.get("CEp99_delta"))
                    or _float(row.get("margin_delta")) > _float(best_parallel.get("margin_delta"))
                    or _float(row.get("curvature_delta")) < 0.90 * _float(best_parallel.get("curvature_delta"))
                )
                beats_lr = int(
                    _float(row.get("CEp99_delta")) < _float(best_lr.get("CEp99_delta"))
                    or _float(row.get("margin_delta")) > _float(best_lr.get("margin_delta"))
                    or _float(row.get("curvature_delta")) < 0.90 * _float(best_lr.get("curvature_delta"))
                )
                row["beats_adamwparallel"] = beats_parallel
                row["beats_best_lr"] = beats_lr
            else:
                row["beats_adamwparallel"] = 0
                row["beats_best_lr"] = 0
            row["best_lr_control"] = best_lr.get("candidate") if best_lr else ""
            row["best_adamwparallel_control"] = best_parallel.get("candidate") if best_parallel else ""


def _run_short_replay(args: argparse.Namespace, out_dir: Path, device: torch.device) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    role_rows: List[Dict[str, Any]] = []
    branches = v9219.V8_BRANCHES
    for dataset in [v9219._to_dataset_name(x) for x in _parse_csv_list(args.datasets)]:
        for seed in _parse_ints(args.seeds):
            for steps in _parse_steps(args.p1_steps_list):
                for branch in branches:
                    row, rr = _train_branch_replay(
                        branch=branch,
                        dataset=dataset,
                        seed=seed,
                        total_steps=steps,
                        args=args,
                        device=device,
                        stage="P1_v8_short_run_strong_control_replay",
                        full_epoch_replay=0,
                    )
                    rows.append(row)
                    role_rows.extend(rr)
    _finalize_group_rows(rows)
    write_csv_rows(out_dir / "p1_v8_short_run_strong_control_replay.csv", rows)
    write_csv_rows(out_dir / "paired_replay_branch_trace_v9220.csv", rows)
    write_csv_rows(out_dir / "role_mechanism_trace_v9220.csv", role_rows)
    summary = _summarize_replay(rows, out_dir / "p1_v8_short_run_summary.csv")
    decision = _replay_decision(rows, summary, require_full=False)
    return rows, role_rows, decision


def _summarize_replay(rows: List[Dict[str, Any]], path: Path) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(str(row.get("candidate")), int(row.get("steps", 0)), str(row.get("stage")))].append(row)
    summary: List[Dict[str, Any]] = []
    for (candidate, steps, stage), vals in sorted(by_key.items()):
        summary.append({
            "stage": stage + "_summary",
            "candidate": candidate,
            "steps": steps,
            "rows": len(vals),
            "test_acc_proxy_mean": _mean([_float(r.get("test_acc_proxy"), float("nan")) for r in vals]),
            "delta_vs_adamw_mean": _mean([_float(r.get("delta_vs_adamw"), float("nan")) for r in vals]),
            "CEp99_delta_mean": _mean([_float(r.get("CEp99_delta"), float("nan")) for r in vals]),
            "margin_delta_mean": _mean([_float(r.get("margin_delta"), float("nan")) for r in vals]),
            "curvature_delta_mean": _mean([_float(r.get("curvature_delta"), float("nan")) for r in vals]),
            "ECE_delta_mean": _mean([_float(r.get("ECE_delta"), float("nan")) for r in vals]),
            "NLL_delta_mean": _mean([_float(r.get("NLL_delta"), float("nan")) for r in vals]),
            "event_coverage_mean": _mean([_float(r.get("event_coverage"), float("nan")) for r in vals]),
            "branch_ratio_mean": _mean([_float(r.get("branch_ratio"), float("nan")) for r in vals]),
            "cos_functional_adamw_mean": _mean([_float(r.get("cos_functional_adamw"), float("nan")) for r in vals]),
            "step_ratio_q90": _q([_float(r.get("step_ratio_q90"), float("nan")) for r in vals], 0.90),
            "memory_ratio_q90": _q([_float(r.get("memory_ratio"), float("nan")) for r in vals], 0.90),
            "beats_adamwparallel_rate": _mean([_float(r.get("beats_adamwparallel"), 0.0) for r in vals]),
            "beats_best_lr_rate": _mean([_float(r.get("beats_best_lr"), 0.0) for r in vals]),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    write_csv_rows(path, summary)
    return summary


def _replay_decision(rows: List[Dict[str, Any]], summary: List[Dict[str, Any]], *, require_full: bool) -> Dict[str, Any]:
    functional_rows = [r for r in rows if r.get("candidate") in v9219.FUNCTIONAL_BRANCHES]
    if not functional_rows:
        return {"pass": 0, "survivor_count": 0}
    pass_rows = [
        r for r in functional_rows
        if _float(r.get("delta_vs_adamw"), -99.0) >= -0.005
        and _float(r.get("beats_adamwparallel"), 0.0) > 0
        and _float(r.get("beats_best_lr"), 0.0) > 0
        and _float(r.get("step_ratio_q90"), 99.0) <= 1.50
        and _float(r.get("memory_ratio"), 99.0) <= 1.05
    ]
    survivor_keys = {(r.get("candidate"), r.get("steps")) for r in pass_rows}
    # Aggregate task/tail gate is intentionally stricter than row-level
    # curvature-only wins.
    aggregate_pass = 0
    for row in summary:
        if row.get("candidate") in v9219.FUNCTIONAL_BRANCHES:
            if (
                _float(row.get("delta_vs_adamw_mean"), -99.0) >= -0.005
                and _float(row.get("beats_adamwparallel_rate"), 0.0) >= 0.80
                and _float(row.get("beats_best_lr_rate"), 0.0) >= 0.80
                and _float(row.get("step_ratio_q90"), 99.0) <= 1.50
                and _float(row.get("memory_ratio_q90"), 99.0) <= 1.05
            ):
                aggregate_pass = 1
    return {
        "pass": int(bool(pass_rows) and aggregate_pass),
        "survivor_count": len(survivor_keys),
        "row_pass_count": len(pass_rows),
        "aggregate_pass": aggregate_pass,
    }


def _run_full_replay(args: argparse.Namespace, out_dir: Path, device: torch.device, p1_decision: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if not p1_decision.get("pass"):
        rows = [_not_run("P2_v8_full_epoch_strong_control_replay", "p2_v8_full_epoch_strong_control_replay.csv", "P1_short_run_strong_control_replay_failed")]
        role_rows: List[Dict[str, Any]] = []
        write_csv_rows(out_dir / "p2_v8_full_epoch_strong_control_replay.csv", rows)
        return rows, role_rows, {"pass": 0, "survivor_count": 0, "row_pass_count": 0, "aggregate_pass": 0}
    total_steps = math.ceil(int(args.train_size) / int(args.batch_size)) * int(args.p2_epochs)
    rows = []
    role_rows = []
    for dataset in [v9219._to_dataset_name(x) for x in _parse_csv_list(args.datasets)]:
        for seed in _parse_ints(args.seeds):
            for branch in v9219.V8_BRANCHES:
                row, rr = _train_branch_replay(
                    branch=branch,
                    dataset=dataset,
                    seed=seed,
                    total_steps=total_steps,
                    args=args,
                    device=device,
                    stage="P2_v8_full_epoch_strong_control_replay",
                    full_epoch_replay=1,
                )
                rows.append(row)
                role_rows.extend(rr)
    _finalize_group_rows(rows)
    write_csv_rows(out_dir / "p2_v8_full_epoch_strong_control_replay.csv", rows)
    summary = _summarize_replay(rows, out_dir / "p2_v8_full_epoch_summary.csv")
    return rows, role_rows, _replay_decision(rows, summary, require_full=True)


def _write_figures(out_dir: Path, route: Dict[str, Any]) -> None:
    fig_dir = ensure_dir(out_dir / "figures")
    for name, title in [
        ("p0_v9219_boundary_dashboard.svg", "v9.2.19 boundary recap"),
        ("p1_short_run_functional_vs_controls.svg", "v8 short-run strong controls"),
        ("p1_ce_margin_curvature_by_step.svg", "CE / margin / curvature by step"),
    ]:
        lines = [
            f"route = {route.get('route')}",
            f"v8_short_run_pass = {route.get('v8_short_run_pass')}",
            f"v8_full_replay_pass = {route.get('v8_full_replay_pass')}",
            f"primary_blocker = {route.get('primary_blocker')}",
        ]
        (fig_dir / name).write_text(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1100\" height=\"240\">"
            "<rect width=\"1100\" height=\"240\" fill=\"#f8fafc\"/>"
            f"<text x=\"24\" y=\"42\" font-family=\"Arial\" font-size=\"24\" fill=\"#111827\">{title}</text>"
            + "".join(
                f"<text x=\"24\" y=\"{82 + i * 28}\" font-family=\"Arial\" font-size=\"16\" fill=\"#374151\">{line}</text>"
                for i, line in enumerate(lines)
            )
            + "</svg>\n",
            encoding="utf-8",
        )


def _write_report(out_dir: Path, route: Dict[str, Any], audit: Dict[str, Any]) -> None:
    report = ROOT / "docs" / "DG-KAN_v9.2.20_FunctionalCore_FullReplay_StrictPureKAN_Interface_实验复盘.md"
    p1_summary = read_csv_rows(out_dir / "p1_v8_short_run_summary.csv")
    top = sorted(p1_summary, key=lambda r: (_float(r.get("CEp99_delta_mean"), 99), -_float(r.get("margin_delta_mean"), -99)))[:10]
    table = "\n".join(
        f"| `{r.get('candidate')}` | `{r.get('steps')}` | `{_float(r.get('CEp99_delta_mean')):.6g}` | `{_float(r.get('margin_delta_mean')):.6g}` | `{_float(r.get('curvature_delta_mean')):.6g}` | `{_float(r.get('step_ratio_q90')):.6g}` | `{_float(r.get('beats_adamwparallel_rate')):.3f}` | `{_float(r.get('beats_best_lr_rate')):.3f}` |"
        for r in top
    )
    text = f"""# DG-KAN v9.2.20 FunctionalCore FullReplay StrictPureKAN Interface 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.20_FunctionalCore_FullReplay_StrictPureKAN_Interface_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把未打开的 P2-P8 写成通过。

## 0. 最新结论

截至本轮，v9.2.20 执行到一个可审计 terminal route：

```text
route = {route.get('route')}
base_candidate = LQ-t2-h256
success_v9220_functional_core_retained = {str(bool(route.get('success_v9220_functional_core_retained'))).lower()}
success_v9220_strict_purekan_functional = {str(bool(route.get('success_v9220_strict_purekan_functional'))).lower()}
success_v9220_external_ready = {str(bool(route.get('success_v9220_external_ready'))).lower()}
```

最终 artifact：

```text
{out_dir.relative_to(ROOT)}/
```

核心结论：

1. P0 复现 v9.2.19 boundary：one-step 有 curvature signal，但不是 full replay。
2. P1 已真实执行 v8 50/240-step strong-control replay，覆盖 15 个 branch/control、3 个任务、指定 seeds。
3. P1 route gate 结果：`v8_short_run_pass = {route.get('v8_short_run_pass')}`，row survivor count = `{route.get('v8_short_run_row_pass_count')}`。
4. P2 full replay 只有在 P1 pass 时打开；本轮 P2 状态为 `{route.get('p2_status')}`。
5. P4-P8 没有越过 P1/P2 gate；未打开部分明确 `not_run`。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v9220_functional_core_full_replay.py` | v9.2.20 runner；生成 P0-P8 artifacts、P1 short-run replay、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9220_functional_core_full_replay.py
```

正式运行：

```bash
python experiments/run_v9220_functional_core_full_replay.py \\
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

## 3. P1 v8 50/240-step strong-control replay

Artifacts：

```text
p1_v8_short_run_strong_control_replay.csv
p1_v8_short_run_summary.csv
paired_replay_branch_trace_v9220.csv
```

Top rows（按 CEp99 delta / margin delta 排序）：

| candidate | steps | CEp99 delta | margin delta | curvature delta | step q90 | beats AdamWParallel | beats best LR |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

说明：本轮 P1 使用 `manual_relative_update_replay_no_torch_loss_backward`，同一初始化、同一 batch sequence；`test_acc_proxy` 是 test prefix evaluation，不伪装成完整 external final test。

## 4. Downstream boundary

| artifact | status / reason |
|---|---|
| `p2_v8_full_epoch_strong_control_replay.csv` | `{route.get('p2_status')}` |
| `p4_strict_purekan_functional_interface_design.csv` | gate-blocked unless P1/P2 retained |
| `p5_basis_factory_functional_actuatability.csv` | gate-blocked unless interface reset opened |
| `p7_functional_survivor_full_validation.csv` | gate-blocked unless survivor exists |
| `p8_robustness_external_ready.csv` | gate-blocked |

## 5. No-fake audit

`v9220_provenance_audit.csv`：

```text
rows_checked = {audit.get('rows_checked')}
fake_proxy_nonzero_count = {audit.get('fake_proxy_nonzero_count')}
fake_data_used = {audit.get('fake_data_used')}
proxy_row_used = {audit.get('proxy_row_used')}
cpu_offload_used = {audit.get('cpu_offload_used')}
no_fake = {str(audit.get('no_fake')).lower()}
no_proxy = {str(audit.get('no_proxy')).lower()}
```

## 6. 最终分析结论

v9.2.20 的真实推进是：

```text
v9.2.19: v8 one-step strong controls completed.
v9.2.20: v8 short-run strong controls completed; full replay only opens if short-run survivor exists.
```

机制判断：

1. 这轮真正检查了 one-step curvature signal 能不能在 50/240-step 中延续。
2. 如果 P1 没有 pass，则不能继续把 v8 functional 写成 core retained，也不能提前打开 strict PureKAN interface。
3. 如果 P1 出现 curvature-only 但 tail/margin/control 不成立，应归类为 geometry-only diagnostic，而不是 task functional success。
4. 下一步必须按 route blocker 继续，不应回到 output target / SNR threshold 小修。

最终一句话：

> v9.2.20 真实执行后停在 `{route.get('route')}`：P1 short-run replay 已落盘，但 functional core 是否恢复取决于 strong-control gate；未过 gate 的下游阶段均保持 `not_run`。
"""
    report.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--target-rel-update", type=float, default=0.001)
    parser.add_argument("--p1-steps-list", default="50,240")
    parser.add_argument("--p2-epochs", type=int, default=20)
    parser.add_argument("--v8-hidden-dim", type=int, default=28)
    parser.add_argument("--v8-basis-count", type=int, default=8)
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
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
        "seed": int(args.seed),
        "no_teacher": 1,
        "no_loss_modification": 1,
        "no_fake_proxy": 1,
        "purekanconv_deferred": 1,
        "purekanformer_deferred": 1,
    })
    contract = [{
        "stage": "contract_audit_v9220",
        "loss_type": "CE",
        "label_smoothing": 0,
        "teacher_used": 0,
        "loss_modified": 0,
        "sampler_modified": 0,
        "class_weight_used": 0,
        "uses_torch_loss_backward_graph": 0,
        "optimizer_protocol": "manual_relative_update_replay_no_torch_loss_backward",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    write_csv_rows(out_dir / "contract_audit_v9220.csv", contract)
    p0_rows = _p0_boundary()
    write_csv_rows(out_dir / "p0_v9219_boundary_reproduction.csv", p0_rows)

    p1_rows, role_rows, p1_decision = _run_short_replay(args, out_dir, device)
    p2_rows, p2_role_rows, p2_decision = _run_full_replay(args, out_dir, device, p1_decision)
    all_role_rows = role_rows + p2_role_rows
    write_csv_rows(out_dir / "p3_v8_mechanism_extraction.csv", all_role_rows if all_role_rows else [_not_run("P3_v8_mechanism_extraction", "p3_v8_mechanism_extraction.csv", "no_measured_role_rows")])

    if p2_decision.get("pass"):
        gate_reason = "not_implemented_after_v8_full_replay_pass_in_this_runner"
    else:
        gate_reason = "P1_or_P2_v8_functional_retained_gate_failed"
    for name, stage in [
        ("p4_strict_purekan_functional_interface_design.csv", "P4_strict_purekan_functional_interface_design"),
        ("p5_basis_factory_functional_actuatability.csv", "P5_basis_factory_functional_actuatability"),
        ("p6_adamw_only_fullpass_repair.csv", "P6_adamw_only_fullpass_repair"),
        ("p7_functional_survivor_full_validation.csv", "P7_functional_survivor_full_validation"),
        ("p8_robustness_external_ready.csv", "P8_robustness_external_ready"),
    ]:
        write_csv_rows(out_dir / name, [_not_run(stage, name, gate_reason)])
    write_csv_rows(out_dir / "purekan_interface_trace_v9220.csv", [_not_run("purekan_interface_trace", "purekan_interface_trace_v9220.csv", gate_reason)])
    write_csv_rows(out_dir / "basis_functional_actuatability_trace_v9220.csv", [_not_run("basis_functional_actuatability_trace", "basis_functional_actuatability_trace_v9220.csv", gate_reason)])

    if p1_decision.get("pass") and p2_decision.get("pass"):
        route_name = "R2-v8FunctionalFullReplayPass"
        blocker = "strict_purekan_interface_not_yet_implemented_after_v8_full_replay_pass"
        retained = 1
    elif p1_decision.get("pass"):
        route_name = "R1-v8FunctionalShortRunPass"
        blocker = "P2_full_replay_not_passed_or_not_opened"
        retained = 0
    else:
        route_name = "R8-FunctionalPausedButNotAbandoned"
        blocker = "v8_short_run_strong_control_replay_failed_or_control_equivalent"
        retained = 0
    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v8_short_run_pass": int(p1_decision.get("pass", 0)),
        "v8_short_run_survivor_count": int(p1_decision.get("survivor_count", 0)),
        "v8_short_run_row_pass_count": int(p1_decision.get("row_pass_count", 0)),
        "v8_full_replay_pass": int(p2_decision.get("pass", 0)),
        "v8_full_replay_survivor_count": int(p2_decision.get("survivor_count", 0)),
        "v8_lr_equivalent": 0,
        "v8_mechanism_identified": int(bool(p1_decision.get("pass") or p2_decision.get("pass"))),
        "v9_interface_pass": 0,
        "basis_factory_functional_pass": 0,
        "adamw_fullpass": 0,
        "functional_short_full_pass": 0,
        "external_ready": 0,
        "functional_core_retained": retained,
        "p2_status": p2_rows[0].get("status") if p2_rows else "missing",
        "primary_blocker": blocker,
        "next_required_implementation": "if_short_run_failed_stop_target_patching_else_complete_full_replay_and_strict_purekan_interface",
        "success_v9220_functional_core_retained": retained,
        "success_v9220_strict_purekan_functional": 0,
        "success_v9220_external_ready": 0,
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    failure_code = "F3_v8_short_run_control_equivalent" if not p1_decision.get("pass") else "F4_v8_full_replay_control_equivalent"
    write_csv_rows(out_dir / "failure_table.csv", [{
        "failure_code": failure_code,
        "status": "active",
        "reason": blocker,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    csv_paths = [
        out_dir / "contract_audit_v9220.csv",
        out_dir / "p0_v9219_boundary_reproduction.csv",
        out_dir / "p1_v8_short_run_strong_control_replay.csv",
        out_dir / "p1_v8_short_run_summary.csv",
        out_dir / "p2_v8_full_epoch_strong_control_replay.csv",
        out_dir / "p3_v8_mechanism_extraction.csv",
        out_dir / "p4_strict_purekan_functional_interface_design.csv",
        out_dir / "p5_basis_factory_functional_actuatability.csv",
        out_dir / "p6_adamw_only_fullpass_repair.csv",
        out_dir / "p7_functional_survivor_full_validation.csv",
        out_dir / "p8_robustness_external_ready.csv",
        out_dir / "failure_table.csv",
    ]
    audit = audit_no_fake(csv_paths)
    write_csv_rows(out_dir / "v9220_provenance_audit.csv", [audit])
    _write_figures(out_dir, route)
    _write_report(out_dir, route, audit)
    write_csv_rows(out_dir / "artifact_hashes.csv", artifact_hash_rows([PLAN_PATH, SCRIPT_PATH] + [p for p in out_dir.iterdir() if p.is_file()], root=ROOT))


if __name__ == "__main__":
    main()
