#!/usr/bin/env python3
"""DG-KAN v8.7 stability-first functional architecture runner.

This runner starts v8.7 from the plan's first principle: before adding another
hand-picked stride/budget recipe, reproduce the v8.5 accepted external-fair
route across explicit run/seed/protocol records and quantify stability.  Any
later v8.7 waves that are not executed are recorded as not_run instead of being
filled with proxy rows.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import run_gafu_v85_real as v85  # noqa: E402
from dgkan_core import ensure_dir, save_json, write_csv  # noqa: E402


PLAN_PATH = "docs/DG-KAN_v8.7_StabilityFirst_FunctionalArchitecture_完整实验计划.md"
METRIC_UNAVAILABLE = "metric_unavailable"


class _PhaseTimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.use_cuda_events = bool(torch.cuda.is_available() and device.type == "cuda")
        self._cuda_pairs: Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self._cpu_ms: Dict[str, float] = {}

    def start(self, name: str) -> Tuple[str, Any, Any]:
        if self.use_cuda_events:
            started = torch.cuda.Event(enable_timing=True)
            ended = torch.cuda.Event(enable_timing=True)
            started.record()
            return (name, started, ended)
        return (name, time.perf_counter(), None)

    def stop(self, token: Tuple[str, Any, Any]) -> None:
        name, started, ended = token
        if self.use_cuda_events:
            ended.record()
            self._cuda_pairs.setdefault(name, []).append((started, ended))
        else:
            self._cpu_ms[name] = self._cpu_ms.get(name, 0.0) + (time.perf_counter() - float(started)) * 1000.0

    def totals_ms(self) -> Dict[str, float]:
        if self.use_cuda_events:
            torch.cuda.synchronize(self.device)
            return {
                name: sum(float(started.elapsed_time(ended)) for started, ended in pairs)
                for name, pairs in self._cuda_pairs.items()
            }
        return dict(self._cpu_ms)


class _CachedParamRowsAdamWNoSync(v85.v83.v72.FastAdamWNoSync):
    """Same FastAdamWNoSync tensor update math, with stable param/grad rows cached."""

    def __init__(self, providers: Sequence[Any], *, lr: float, weight_decay: float = 1.0e-4) -> None:
        self._cached_param_rows: List[Tuple[str, torch.Tensor, torch.Tensor]] | None = None
        super().__init__(providers, lr=lr, weight_decay=weight_decay)
        self._cached_param_rows = super().params()

    def params(self) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        if self._cached_param_rows is None:
            return super().params()
        return self._cached_param_rows


class _ForeachAdamWAddcdivNoSync(v85.v83.v72.FastAdamWNoSync):
    """AdamW addcdiv path using foreach tensor ops; hyperparameters unchanged."""

    def __init__(self, providers: Sequence[Any], *, lr: float, weight_decay: float = 1.0e-4) -> None:
        self._cached_param_rows: List[Tuple[str, torch.Tensor, torch.Tensor]] | None = None
        super().__init__(providers, lr=lr, weight_decay=weight_decay)
        self._cached_param_rows = super().params()
        self._foreach_params = [p for _name, p, _g in self._cached_param_rows]
        self._foreach_grads = [g for _name, _p, g in self._cached_param_rows]
        self._foreach_m = [self.m[id(p)] for p in self._foreach_params] if self.update_mode == "adamw_addcdiv" else []
        self._foreach_v = [self.v[id(p)] for p in self._foreach_params] if self.update_mode == "adamw_addcdiv" else []
        self._foreach_supported = (
            self.update_mode == "adamw_addcdiv"
            and hasattr(torch, "_foreach_addcdiv_")
            and hasattr(torch, "_foreach_addcmul_")
        )

    def params(self) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        if self._cached_param_rows is None:
            return super().params()
        return self._cached_param_rows

    def step(self, step: int, total_steps: int, *, warmup_cosine: bool = True) -> float:
        if not self._foreach_supported:
            return super().step(step, total_steps, warmup_cosine=warmup_cosine)
        self.t += 1
        lr = self.lr
        if warmup_cosine:
            warm = max(5, int(total_steps) // 10)
            if int(step) <= warm:
                lr *= int(step) / warm
            else:
                prog = (int(step) - warm) / max(1, int(total_steps) - warm)
                lr *= 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * prog))
        beta1, beta2, eps = 0.9, 0.99, 1.0e-8
        with torch.no_grad():
            if self.weight_decay:
                torch._foreach_mul_(self._foreach_params, 1.0 - lr * self.weight_decay)
            torch._foreach_mul_(self._foreach_m, beta1)
            torch._foreach_add_(self._foreach_m, self._foreach_grads, alpha=1.0 - beta1)
            torch._foreach_mul_(self._foreach_v, beta2)
            torch._foreach_addcmul_(self._foreach_v, self._foreach_grads, self._foreach_grads, value=1.0 - beta2)
            denom = torch._foreach_sqrt(self._foreach_v)
            torch._foreach_div_(denom, math.sqrt(1.0 - beta2**self.t))
            torch._foreach_add_(denom, eps)
            torch._foreach_addcdiv_(
                self._foreach_params,
                self._foreach_m,
                denom,
                value=-lr / (1.0 - beta1**self.t),
            )
        self.zero_grad()
        return float("nan")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in (None, "", METRIC_UNAVAILABLE):
            return default
        return float(value)
    except Exception:
        return default


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _release_cuda_memory_v87(device: torch.device) -> None:
    gc.collect()
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()


def _parse_ints(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _q90(values: Sequence[float]) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = 0.9 * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _finite_mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return mean(vals) if vals else float("nan")


def _quantile(values: Sequence[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
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


def _json_dumps_stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_not_run_artifacts(out_dir: Path, reason: str) -> None:
    artifacts = {
        "fixed_hyperparam_sensitivity_grid.csv": "P1_FIXED_HYPERPARAM_SENSITIVITY_V87",
        "adaptive_functional_one_step.csv": "P2_ADAPTIVE_FUNCTIONAL_ONE_STEP_V87",
        "adaptive_functional_multistep.csv": "P3_ADAPTIVE_FUNCTIONAL_MULTISTEP_V87",
        "auto_architecture_selection.csv": "P4_AUTO_ARCHITECTURE_SELECTION_V87",
        "base_implementation_screen.csv": "P4_BASE_IMPLEMENTATION_SCREEN_V87",
        "base_implementation_screen_summary.csv": "P4_BASE_IMPLEMENTATION_SCREEN_SUMMARY_V87",
        "robust_timing_protocols.csv": "P5_ROBUST_TIMING_PROTOCOLS_V87",
        "training_compute_counter.csv": "P6_TRAINING_COMPUTE_COUNTER_V87",
        "kanbefair_multitask_transfer.csv": "P7_KANBEFAIR_MULTITASK_TRANSFER_V87",
        "joint_fair_envelope.csv": "P8_JOINT_FAIR_ENVELOPE_V87",
        "functional_causality_multitask.csv": "P9_FUNCTIONAL_CAUSALITY_MULTITASK_V87",
        "continual_balance_multisplit.csv": "P10_CONTINUAL_BALANCE_MULTISPLIT_V87",
        "robustness_perturbation.csv": "P11_ROBUSTNESS_PERTURBATION_V87",
        "functional_mechanism_attribution.csv": "P12_FUNCTIONAL_MECHANISM_ATTRIBUTION_V87",
        "negative_boundary_audit.csv": "P13_NEGATIVE_BOUNDARY_AUDIT_V87",
    }
    for filename, stage in artifacts.items():
        path = out_dir / filename
        if path.exists():
            continue
        write_csv(path, [{
            "stage": stage,
            "status": "not_run",
            "reason": reason,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }])


def _v85_namespace(
    *,
    child_out_dir: Path,
    args: argparse.Namespace,
    seed: int,
    fresh: bool,
) -> argparse.Namespace:
    return argparse.Namespace(
        out_dir=str(child_out_dir),
        fresh=bool(fresh),
        device=args.device,
        data_root=args.data_root,
        kanbefair_path=args.kanbefair_path,
        datasets="MNIST",
        kb_data_protocol="kanbefair",
        train_size=60000,
        val_size=0,
        test_size=10000,
        seed=int(seed),
        kb_epochs=int(args.kb_epochs),
        kb_batch_size=int(args.kb_batch_size),
        adapter_smoke_train_size=512,
        adapter_smoke_test_size=256,
        adapter_steps=8,
        run_primary_transfer=True,
        run_functional_causality=True,
        run_symbolic=False,
        run_continual=False,
        primary_train_size=60000,
        primary_test_size=10000,
        primary_epochs=int(args.primary_epochs),
        causality_train_size=60000,
        causality_test_size=10000,
        causality_epochs=int(args.primary_epochs),
        symbolic_datasets="Special_1d_gelu",
        symbolic_epochs=100,
        symbolic_batch_size=128,
        symbolic_lr=1.0e-3,
        continual_train_size_per_task=1024,
        continual_test_size_per_task=512,
        continual_epochs_per_task=2,
        continual_batch_size=128,
        continual_lr=1.0e-3,
        continual_restore_old_head_rows=False,
        continual_freeze_stack_after_first_task=False,
        continual_freeze_head_shared_after_first_task=False,
        continual_stack_anchor_strength=0.0,
        continual_old_head_grad_scale=0.0,
        continual_old_head_restore_strength=1.0,
        continual_old_head_age_decay=0.0,
        dg_candidate_id=str(getattr(args, "p0_dg_candidate_id", "KW6")),
        dg_hidden_dim=int(getattr(args, "p0_dg_hidden_dim", 28)),
        dg_basis_count=8,
        dg_batch_size=128,
        dg_eval_batch_size=512,
        dg_stream_batches_from_cpu=False,
        dg_stream_chunk_batches=1,
        dg_stream_epoch_permute_cpu=False,
        dg_stream_chunk_order_shuffle=False,
        weight_decay=1.0e-4,
        ft7_event_stride=int(args.fixed_ft7_event_stride),
        ft7_event_alpha_mult=float(args.fixed_ft7_event_alpha_mult),
    )


def _primary_by_candidate(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if str(row.get("status")) != "measured":
            continue
        cid = str(row.get("candidate_id", ""))
        if cid:
            out[cid] = row
    return out


def _causality_by_name(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if str(row.get("status")) != "measured":
            continue
        control = str(row.get("control_name", ""))
        if control:
            out[control] = row
    return out


def _p0_rows_from_v85(
    *,
    child_out_dir: Path,
    run_id: str,
    rerun_index: int,
    seed: int,
    timing_protocol: str,
    fixed_stride: int,
    dg_candidate_id: str,
    dg_hidden_dim: int,
    decision: Dict[str, Any],
) -> List[Dict[str, Any]]:
    primary_rows = _read_csv(child_out_dir / "kanbefair_primary_transfer.csv")
    causality_rows = _read_csv(child_out_dir / "functional_causality_kanbefair.csv")
    primary = _primary_by_candidate(primary_rows)
    causality = _causality_by_name(causality_rows)
    kb_mlp = primary.get("KB-MLP", {})
    kb_mlp_acc = _safe_float(kb_mlp.get("test_metric"))
    kb_mlp_step = _safe_float(kb_mlp.get("step_time_ms"))

    base_key = f"DG0-{dg_candidate_id}-hidden{int(dg_hidden_dim)}-base"
    func_key = f"DG1-FT7-{dg_candidate_id}-hidden{int(dg_hidden_dim)}-functional"
    candidates: List[Tuple[str, str, Dict[str, Any], Dict[str, Any]]] = [
        ("KB-MLP", "KB-MLP", primary.get("KB-MLP", {}), {}),
        ("KB-KAN", "KB-KAN", primary.get("KB-KAN", {}), {}),
        (f"DG0-{dg_candidate_id} hidden{int(dg_hidden_dim)} stride{int(fixed_stride)} base", "DG-Base", primary.get(base_key, {}), causality.get("DG-Base", {})),
        (f"DG1-FT7 {dg_candidate_id} hidden{int(dg_hidden_dim)} stride{int(fixed_stride)}", "DG-Functional", primary.get(func_key, {}), causality.get("DG-Functional", {})),
        ("DG-NoOp", "DG-NoOp", {}, causality.get("DG-NoOp", {})),
        ("DG-RandomFunc", "DG-RandomFunc", {}, causality.get("DG-RandomFunc", {})),
    ]

    out: List[Dict[str, Any]] = []
    for candidate_id, control_name, p_row, c_row in candidates:
        source_row = p_row or c_row
        status = "measured" if source_row else "not_run"
        test_acc = _safe_float(p_row.get("test_metric"), _safe_float(c_row.get("test_acc")))
        step_time = _safe_float(p_row.get("step_time_ms"))
        step_ratio = step_time / kb_mlp_step if math.isfinite(step_time) and math.isfinite(kb_mlp_step) and kb_mlp_step > 0.0 else float("nan")
        delta_vs_mlp = test_acc - kb_mlp_acc if math.isfinite(test_acc) and math.isfinite(kb_mlp_acc) else float("nan")
        curvature_ratio = _safe_float(c_row.get("curvature_ratio"), _safe_float(p_row.get("functional_curvature_ratio_vs_dg_base")))
        functional_events = _safe_float(c_row.get("functional_event_count"), _safe_float(p_row.get("functional_event_count")))
        out.append({
            "stage": "P0_V85_FRESH_REPRODUCTION_STABILITY_V87",
            "status": status,
            "run_id": run_id,
            "child_out_dir": str(child_out_dir),
            "rerun_index": rerun_index,
            "seed": seed,
            "timing_protocol": timing_protocol,
            "candidate_id": candidate_id,
            "control_name": control_name,
            "test_acc": test_acc if math.isfinite(test_acc) else METRIC_UNAVAILABLE,
            "delta_vs_KB_MLP": delta_vs_mlp if math.isfinite(delta_vs_mlp) else METRIC_UNAVAILABLE,
            "delta_vs_DG0": p_row.get("metric_delta_vs_dg_base", c_row.get("acc_delta_vs_base", METRIC_UNAVAILABLE)),
            "params": p_row.get("params", METRIC_UNAVAILABLE),
            "forward_FLOPs": p_row.get("FLOPs", METRIC_UNAVAILABLE),
            "backward_FLOPs_estimate": METRIC_UNAVAILABLE,
            "step_time_ms": step_time if math.isfinite(step_time) else METRIC_UNAVAILABLE,
            "step_ratio_vs_KB_MLP": step_ratio if math.isfinite(step_ratio) else METRIC_UNAVAILABLE,
            "train_time_s": p_row.get("train_time_s", METRIC_UNAVAILABLE),
            "peak_memory_MB": p_row.get("peak_memory_MB", METRIC_UNAVAILABLE),
            "curvature": p_row.get("geometry_curvature_after", METRIC_UNAVAILABLE),
            "curvature_ratio_vs_DG0": curvature_ratio if math.isfinite(curvature_ratio) else METRIC_UNAVAILABLE,
            "functional_events": functional_events if math.isfinite(functional_events) else METRIC_UNAVAILABLE,
            "event_intervals": fixed_stride if candidate_id.startswith("DG1-FT7") and math.isfinite(functional_events) and functional_events > 0 else METRIC_UNAVAILABLE,
            "event_intervals_source": "configured_fixed_stride" if candidate_id.startswith("DG1-FT7") else METRIC_UNAVAILABLE,
            "functional_update_time_ratio": c_row.get("functional_update_time_ratio", p_row.get("functional_update_time_ratio", METRIC_UNAVAILABLE)),
            "NoTeacherNoLossNoOffloadPass": int(
                _safe_float(source_row.get("external_teacher_used"), 0.0) == 0.0
                and _safe_float(source_row.get("self_teacher_used"), 0.0) == 0.0
                and _safe_float(source_row.get("geometry_loss_used"), 0.0) == 0.0
                and _safe_float(source_row.get("sampler_changed"), 0.0) == 0.0
                and _safe_float(source_row.get("class_weight_used"), 0.0) == 0.0
                and _safe_float(source_row.get("cpu_offload_used"), 0.0) == 0.0
            ) if source_row else 0,
            "FunctionalCausalityPass": int(_safe_float(decision.get("functional_causality_pass"), 0.0) == 1.0),
            "v85_route": decision.get("route", METRIC_UNAVAILABLE),
            "v85_success_minimum": decision.get("success_v85_minimum", 0),
            "v85_success_external_formal": decision.get("success_v85_external_formal", 0),
            "loss_type": "CE" if source_row else METRIC_UNAVAILABLE,
            "geometry_loss_used": source_row.get("geometry_loss_used", 0) if source_row else 0,
            "external_teacher_used": source_row.get("external_teacher_used", 0) if source_row else 0,
            "self_teacher_used": source_row.get("self_teacher_used", 0) if source_row else 0,
            "sampler_changed": source_row.get("sampler_changed", 0) if source_row else 0,
            "class_weight_used": source_row.get("class_weight_used", 0) if source_row else 0,
            "cpu_offload_used": source_row.get("cpu_offload_used", 0) if source_row else 0,
            "fake_data_used": source_row.get("fake_data_used", 0) if source_row else 0,
            "proxy_row_used": source_row.get("proxy_row_used", 0) if source_row else 0,
        })
    return out


def _run_p0(args: argparse.Namespace, out_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    global _V85_FUSED_USE_COMPILED_CORE
    if str(args.reuse_p0_out_dir).strip():
        source_dir = Path(args.reuse_p0_out_dir)
        source_p0 = source_dir / "v85_fresh_reproduction_stability.csv"
        source_child = source_dir / "p0_child_v85_decisions.csv"
        if not source_p0.exists() or not source_child.exists():
            raise FileNotFoundError(f"cannot reuse P0 artifacts from {source_dir}: required CSV files are missing")
        target_p0 = out_dir / "v85_fresh_reproduction_stability.csv"
        target_child = out_dir / "p0_child_v85_decisions.csv"
        shutil.copy2(source_p0, target_p0)
        shutil.copy2(source_child, target_child)
        save_json(out_dir / "p0_reuse_source.json", {
            "stage": "P0_REUSE_SOURCE_V87",
            "status": "measured_reused_artifact",
            "source_out_dir": str(source_dir),
            "source_v85_fresh_reproduction_stability_sha256": _sha256(source_p0),
            "source_p0_child_v85_decisions_sha256": _sha256(source_child),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
        return _read_csv(target_p0), _read_csv(target_child)

    seeds = _parse_ints(args.p0_seeds)
    all_rows: List[Dict[str, Any]] = []
    child_decisions: List[Dict[str, Any]] = []
    for rerun_index in range(int(args.p0_reruns)):
        for seed in seeds:
            run_id = f"rerun{rerun_index:02d}_seed{seed}_T0"
            child_out_dir = out_dir / "p0_v85_runs" / run_id
            ensure_dir(child_out_dir)
            child_args = _v85_namespace(child_out_dir=child_out_dir, args=args, seed=seed, fresh=True)
            original_adamw = v85.v83.v72.FastAdamWNoSync
            original_ce_backward = v85._manual_ce_backward_v85
            original_ce_backward_holdout = v85._manual_ce_backward_with_holdout_v85
            original_pretrain_hook = getattr(v85, "DG_PRETRAIN_HOOK", None)
            original_v85_fused_use_compiled = _V85_FUSED_USE_COMPILED_CORE
            if bool(getattr(args, "p0_use_foreach_adamw_addcdiv", False)):
                v85.v83.v72.FastAdamWNoSync = _ForeachAdamWAddcdivNoSync
            if bool(getattr(args, "p0_use_fused_ce_head_backward", False)):
                _V85_FUSED_USE_COMPILED_CORE = bool(getattr(args, "p0_use_compiled_fused_ce_head_backward", False))
                v85._manual_ce_backward_v85 = _manual_ce_backward_v85_fused
                v85._manual_ce_backward_with_holdout_v85 = _manual_ce_backward_with_holdout_v85_fused
                if bool(
                    getattr(args, "p0_use_compiled_fused_ce_head_backward", False)
                    and getattr(args, "p0_prewarm_compiled_fused_ce_head_backward", False)
                ):
                    v85.DG_PRETRAIN_HOOK = _make_v85_compiled_fused_prewarm_hook()
            try:
                decision = v85.run(child_args)
            finally:
                v85.v83.v72.FastAdamWNoSync = original_adamw
                v85._manual_ce_backward_v85 = original_ce_backward
                v85._manual_ce_backward_with_holdout_v85 = original_ce_backward_holdout
                v85.DG_PRETRAIN_HOOK = original_pretrain_hook
                _V85_FUSED_USE_COMPILED_CORE = original_v85_fused_use_compiled
            child_decisions.append({
                "stage": "P0_CHILD_V85_DECISION_V87",
                "status": "measured",
                "run_id": run_id,
                "child_out_dir": str(child_out_dir),
                "rerun_index": rerun_index,
                "seed": seed,
                "timing_protocol": "T0-v85-official-local",
                "optimizer_impl": "ForeachAdamWAddcdivNoSync" if bool(getattr(args, "p0_use_foreach_adamw_addcdiv", False)) else "FastAdamWNoSync",
                "ce_head_backward_impl": (
                    "compiled_fused_value_equivalent"
                    if bool(
                        getattr(args, "p0_use_fused_ce_head_backward", False)
                        and getattr(args, "p0_use_compiled_fused_ce_head_backward", False)
                    )
                    else ("fused_value_equivalent" if bool(getattr(args, "p0_use_fused_ce_head_backward", False)) else "v85_default")
                ),
                "compiled_fused_ce_head_prewarm_used": int(
                    bool(
                        getattr(args, "p0_use_fused_ce_head_backward", False)
                        and getattr(args, "p0_use_compiled_fused_ce_head_backward", False)
                        and getattr(args, "p0_prewarm_compiled_fused_ce_head_backward", False)
                    )
                ),
                "route": decision.get("route", METRIC_UNAVAILABLE),
                "primary_transfer_pass": decision.get("primary_transfer_pass", 0),
                "parameter_fair_pass": decision.get("parameter_fair_pass", 0),
                "flops_fair_pass": decision.get("flops_fair_pass", 0),
                "wallclock_fair_pass": decision.get("wallclock_fair_pass", 0),
                "functional_causality_pass": decision.get("functional_causality_pass", 0),
                "success_v85_minimum": decision.get("success_v85_minimum", 0),
                "success_v85_external_formal": decision.get("success_v85_external_formal", 0),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": decision.get("cpu_offload_used", 0),
            })
            all_rows.extend(_p0_rows_from_v85(
                child_out_dir=child_out_dir,
                run_id=run_id,
                rerun_index=rerun_index,
                seed=seed,
                timing_protocol="T0-v85-official-local",
                fixed_stride=int(args.fixed_ft7_event_stride),
                dg_candidate_id=str(args.p0_dg_candidate_id),
                dg_hidden_dim=int(args.p0_dg_hidden_dim),
                decision=decision,
            ))
    write_csv(out_dir / "v85_fresh_reproduction_stability.csv", all_rows)
    write_csv(out_dir / "p0_child_v85_decisions.csv", child_decisions)
    return all_rows, child_decisions


def _run_p1_sensitivity(args: argparse.Namespace, out_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    seeds = _parse_ints(args.p1_seeds)
    strides = _parse_ints(args.p1_strides)
    alphas = [float(x.strip()) for x in str(args.p1_alpha_mults).split(",") if x.strip()]
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for stride in strides:
            for alpha in alphas:
                run_id = f"p1_seed{seed}_stride{stride}_alpha{str(alpha).replace('.', 'p')}"
                child_out_dir = out_dir / "p1_sensitivity_runs" / run_id
                ensure_dir(child_out_dir)
                child_args = _v85_namespace(
                    child_out_dir=child_out_dir,
                    args=args,
                    seed=seed,
                    fresh=True,
                )
                child_args.ft7_event_stride = int(stride)
                child_args.ft7_event_alpha_mult = float(alpha)
                decision = v85.run(child_args)
                primary = _primary_by_candidate(_read_csv(child_out_dir / "kanbefair_primary_transfer.csv"))
                causality = _causality_by_name(_read_csv(child_out_dir / "functional_causality_kanbefair.csv"))
                kb = primary.get("KB-MLP", {})
                dg1 = primary.get("DG1-FT7-KW6-hidden28-functional", {})
                c_dg1 = causality.get("DG-Functional", {})
                kb_acc = _safe_float(kb.get("test_metric"))
                kb_step = _safe_float(kb.get("step_time_ms"))
                dg_acc = _safe_float(dg1.get("test_metric"))
                dg_step = _safe_float(dg1.get("step_time_ms"))
                dg_delta = dg_acc - kb_acc if math.isfinite(dg_acc) and math.isfinite(kb_acc) else float("nan")
                step_ratio = dg_step / max(kb_step, 1.0e-12) if math.isfinite(dg_step) and math.isfinite(kb_step) else float("nan")
                curv_ratio = _safe_float(c_dg1.get("curvature_ratio"), _safe_float(dg1.get("functional_curvature_ratio_vs_dg_base")))
                row = {
                    "stage": "P1_FIXED_HYPERPARAM_SENSITIVITY_V87",
                    "status": "measured",
                    "run_id": run_id,
                    "child_out_dir": str(child_out_dir),
                    "seed": seed,
                    "ft7_event_stride": stride,
                    "ft7_event_alpha_mult": alpha,
                    "candidate_id": "DG1-FT7-KW6-hidden28-functional",
                    "test_acc": dg_acc if math.isfinite(dg_acc) else METRIC_UNAVAILABLE,
                    "delta_vs_KB_MLP": dg_delta if math.isfinite(dg_delta) else METRIC_UNAVAILABLE,
                    "delta_vs_DG0": dg1.get("metric_delta_vs_dg_base", c_dg1.get("acc_delta_vs_base", METRIC_UNAVAILABLE)),
                    "curvature_ratio_vs_DG0": curv_ratio if math.isfinite(curv_ratio) else METRIC_UNAVAILABLE,
                    "step_ratio_vs_KB_MLP": step_ratio if math.isfinite(step_ratio) else METRIC_UNAVAILABLE,
                    "step_time_ms": dg_step if math.isfinite(dg_step) else METRIC_UNAVAILABLE,
                    "functional_events": c_dg1.get("functional_event_count", dg1.get("functional_event_count", METRIC_UNAVAILABLE)),
                    "functional_update_time_ratio": c_dg1.get("functional_update_time_ratio", METRIC_UNAVAILABLE),
                    "v85_route": decision.get("route", METRIC_UNAVAILABLE),
                    "v85_wallclock_fair_pass": decision.get("wallclock_fair_pass", 0),
                    "v85_external_formal_pass": decision.get("success_v85_external_formal", 0),
                    "all_gate_pass": int(
                        math.isfinite(dg_delta)
                        and dg_delta > 0.0
                        and math.isfinite(curv_ratio)
                        and curv_ratio <= 0.90
                        and math.isfinite(step_ratio)
                        and step_ratio <= 1.50
                    ),
                    "loss_type": "CE",
                    "geometry_loss_used": 0,
                    "external_teacher_used": 0,
                    "self_teacher_used": 0,
                    "sampler_changed": 0,
                    "class_weight_used": 0,
                    "cpu_offload_used": decision.get("cpu_offload_used", 0),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                }
                rows.append(row)
    write_csv(out_dir / "fixed_hyperparam_sensitivity_grid.csv", rows)
    acc_values = [_safe_float(r.get("test_acc")) for r in rows]
    step_values = [_safe_float(r.get("step_ratio_vs_KB_MLP")) for r in rows]
    curv_values = [_safe_float(r.get("curvature_ratio_vs_DG0")) for r in rows]
    finite_acc = [v for v in acc_values if math.isfinite(v)]
    finite_step = [v for v in step_values if math.isfinite(v)]
    finite_curv = [v for v in curv_values if math.isfinite(v)]
    acc_range = max(finite_acc) - min(finite_acc) if finite_acc else float("nan")
    step_range = max(finite_step) - min(finite_step) if finite_step else float("nan")
    curv_range = max(finite_curv) - min(finite_curv) if finite_curv else float("nan")
    summary = {
        "stage": "P1_FIXED_HYPERPARAM_SENSITIVITY_SUMMARY_V87",
        "status": "measured" if rows else "not_run",
        "rows": len(rows),
        "seed_count": len(seeds),
        "stride_count": len(strides),
        "alpha_count": len(alphas),
        "test_acc_range_percentage_points": acc_range if math.isfinite(acc_range) else METRIC_UNAVAILABLE,
        "step_ratio_range": step_range if math.isfinite(step_range) else METRIC_UNAVAILABLE,
        "curvature_ratio_range": curv_range if math.isfinite(curv_range) else METRIC_UNAVAILABLE,
        "all_gate_pass_rate": mean([int(_safe_float(r.get("all_gate_pass"), 0.0) == 1.0) for r in rows]) if rows else 0.0,
        "H1_fixed_schedule_sensitivity_pass": int(
            (math.isfinite(acc_range) and acc_range >= 0.30)
            or (math.isfinite(step_range) and step_range >= 0.20)
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": sum(int(_safe_float(r.get("cpu_offload_used"), 0.0) != 0.0) for r in rows),
    }
    write_csv(out_dir / "fixed_hyperparam_sensitivity_summary.csv", [summary])
    return rows, summary


def _parse_paths(text: str) -> List[Path]:
    return [Path(part.strip()) for part in str(text).split(",") if part.strip()]


def _timing_row_from_existing(
    *,
    source_dir: Path,
    source_artifact: Path,
    row: Dict[str, Any],
    row_kind: str,
    fixed_stride: int,
    fixed_alpha: float,
) -> Dict[str, Any] | None:
    if not str(row.get("candidate_id", "")).startswith("DG1-FT7"):
        return None
    if row_kind == "p0":
        stride_value = _safe_float(row.get("event_intervals"))
        alpha_value = fixed_alpha
        if not math.isfinite(stride_value) or int(stride_value) != int(fixed_stride):
            return None
        seed = row.get("seed", METRIC_UNAVAILABLE)
        run_id = row.get("run_id", source_dir.name)
    else:
        stride_value = _safe_float(row.get("ft7_event_stride"))
        alpha_value = _safe_float(row.get("ft7_event_alpha_mult"))
        if not math.isfinite(stride_value) or int(stride_value) != int(fixed_stride):
            return None
        if not math.isfinite(alpha_value) or abs(float(alpha_value) - float(fixed_alpha)) > 1.0e-12:
            return None
        seed = row.get("seed", METRIC_UNAVAILABLE)
        run_id = row.get("run_id", source_dir.name)
    step_ratio = _safe_float(row.get("step_ratio_vs_KB_MLP"))
    step_time = _safe_float(row.get("step_time_ms"))
    return {
        "stage": "P5_ROBUST_TIMING_PROTOCOLS_V87",
        "status": "measured",
        "timing_protocol": "T4-full-loop-existing-artifact",
        "protocol_scope": "full_loop_including_training_guard_eval_sync_from_landed_v85_child_run",
        "source_out_dir": str(source_dir),
        "source_artifact": str(source_artifact),
        "source_artifact_sha256": _sha256(source_artifact),
        "source_row_kind": row_kind,
        "run_id": run_id,
        "seed": seed,
        "ft7_event_stride": int(fixed_stride),
        "ft7_event_alpha_mult": float(fixed_alpha),
        "candidate_id": row.get("candidate_id"),
        "test_acc": row.get("test_acc", METRIC_UNAVAILABLE),
        "delta_vs_KB_MLP": row.get("delta_vs_KB_MLP", METRIC_UNAVAILABLE),
        "curvature_ratio_vs_DG0": row.get("curvature_ratio_vs_DG0", METRIC_UNAVAILABLE),
        "step_time_ms": step_time if math.isfinite(step_time) else METRIC_UNAVAILABLE,
        "step_ratio_vs_KB_MLP": step_ratio if math.isfinite(step_ratio) else METRIC_UNAVAILABLE,
        "functional_events": row.get("functional_events", METRIC_UNAVAILABLE),
        "all_gate_pass": int(
            math.isfinite(_safe_float(row.get("delta_vs_KB_MLP")))
            and _safe_float(row.get("delta_vs_KB_MLP")) > 0.0
            and math.isfinite(_safe_float(row.get("curvature_ratio_vs_DG0")))
            and _safe_float(row.get("curvature_ratio_vs_DG0")) <= 0.90
            and math.isfinite(step_ratio)
            and step_ratio <= 1.50
        ),
        "timing_gate_pass": int(math.isfinite(step_ratio) and step_ratio <= 1.50),
        "loss_type": "CE",
        "geometry_loss_used": 0,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "sampler_changed": 0,
        "class_weight_used": 0,
        "cpu_offload_used": row.get("cpu_offload_used", 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }


def _write_robust_timing_files(out_dir: Path, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        rows.append({
            "stage": "P5_ROBUST_TIMING_PROTOCOLS_V87",
            "status": "not_run",
            "reason": "no_matching_measured_timing_rows_found",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    write_csv(out_dir / "robust_timing_protocols.csv", rows)
    step_values = [_safe_float(r.get("step_ratio_vs_KB_MLP")) for r in rows if str(r.get("status")) == "measured"]
    finite_step = [v for v in step_values if math.isfinite(v)]
    q90 = _q90(finite_step)
    max_step = max(finite_step) if finite_step else float("nan")
    protocol_count = len({str(r.get("timing_protocol")) for r in rows if str(r.get("status")) == "measured"})
    all_gate_pass_rate = (
        mean([int(_safe_float(r.get("all_gate_pass"), 0.0) == 1.0) for r in rows if str(r.get("status")) == "measured"])
        if finite_step
        else 0.0
    )
    timing_gate_pass_rate = (
        mean([int(_safe_float(r.get("timing_gate_pass"), 0.0) == 1.0) for r in rows if str(r.get("status")) == "measured"])
        if finite_step
        else 0.0
    )
    full_t0_t4_complete = int({"T0", "T1", "T2", "T3", "T4"}.issubset({str(r.get("timing_protocol"))[:2] for r in rows}))
    summary = {
        "stage": "P5_ROBUST_TIMING_SUMMARY_V87",
        "status": "measured" if finite_step else "not_run",
        "measured_rows": len(finite_step),
        "timing_protocol_count": protocol_count,
        "full_t0_t4_complete": full_t0_t4_complete,
        "q90_step_ratio_vs_KB_MLP": q90 if math.isfinite(q90) else METRIC_UNAVAILABLE,
        "max_step_ratio_vs_KB_MLP": max_step if math.isfinite(max_step) else METRIC_UNAVAILABLE,
        "all_gate_pass_rate": all_gate_pass_rate,
        "timing_gate_pass_rate": timing_gate_pass_rate,
        "robust_timing_partial_gate_pass": int(math.isfinite(q90) and q90 <= 1.50),
        "robust_timing_full_gate_pass": int(full_t0_t4_complete and math.isfinite(q90) and q90 <= 1.50),
        "reason": "T4_existing_artifact_only_not_full_T0_T4" if not full_t0_t4_complete else "full_T0_T4_measured",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": sum(int(_safe_float(r.get("cpu_offload_used"), 0.0) != 0.0) for r in rows),
    }
    write_csv(out_dir / "robust_timing_summary.csv", [summary])
    return summary


def _run_robust_timing_from_artifacts(args: argparse.Namespace, out_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source_dir in _parse_paths(args.timing_source_out_dirs):
        p0_path = source_dir / "v85_fresh_reproduction_stability.csv"
        if p0_path.exists() and not (source_dir / "p0_reuse_source.json").exists():
            for row in _read_csv(p0_path):
                measured = _timing_row_from_existing(
                    source_dir=source_dir,
                    source_artifact=p0_path,
                    row=row,
                    row_kind="p0",
                    fixed_stride=int(args.fixed_ft7_event_stride),
                    fixed_alpha=float(args.fixed_ft7_event_alpha_mult),
                )
                if measured:
                    rows.append(measured)
        p1_path = source_dir / "fixed_hyperparam_sensitivity_grid.csv"
        if p1_path.exists():
            for row in _read_csv(p1_path):
                measured = _timing_row_from_existing(
                    source_dir=source_dir,
                    source_artifact=p1_path,
                    row=row,
                    row_kind="p1",
                    fixed_stride=int(args.fixed_ft7_event_stride),
                    fixed_alpha=float(args.fixed_ft7_event_alpha_mult),
                )
                if measured:
                    rows.append(measured)
        robust_path = source_dir / "robust_timing_protocols.csv"
        if robust_path.exists():
            robust_sha = _sha256(robust_path)
            for row in _read_csv(robust_path):
                if str(row.get("status")) != "measured":
                    continue
                if not math.isfinite(_safe_float(row.get("step_ratio_vs_KB_MLP"))):
                    continue
                source_kind = str(row.get("source_row_kind", ""))
                if source_kind not in {"p0", "t0_t2_phase_clean", "t3_phase_clean", "p4_robust_hidden_selection", "p4_base_implementation_screen"}:
                    continue
                copied = dict(row)
                copied["source_artifact"] = str(robust_path)
                copied["source_artifact_sha256"] = robust_sha
                copied["source_out_dir"] = str(source_dir)
                rows.append(copied)
    summary = _write_robust_timing_files(out_dir, rows)
    return rows, summary


def _timed_kb_mlp_train_step(args: argparse.Namespace, x: torch.Tensor, y: torch.Tensor, *, device: torch.device, input_dim: int, num_classes: int) -> Dict[str, Any]:
    MLP, _KANbeFair, _BSpline, _stub, status = v85._import_kanbefair_models(Path(args.kanbefair_path))
    if MLP is None:
        raise RuntimeError(f"KANbeFair MLP import failed: {status}")
    ns = v85._kb_namespace(model_name="MLP", input_size=input_dim, output_size=num_classes, layers_width=[32], activation_name="gelu")
    model = MLP(ns).to(device)
    xb = x[: int(args.t3_batch_size)].to(device)
    yb = y[: int(args.t3_batch_size)].to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    timer = _PhaseTimer(device)

    def step(*, record: bool) -> None:
        opt.zero_grad(set_to_none=True)
        token = timer.start("forward") if record else None
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        if token is not None:
            timer.stop(token)
        token = timer.start("backward") if record else None
        loss.backward()
        if token is not None:
            timer.stop(token)
        token = timer.start("base_update") if record else None
        opt.step()
        if token is not None:
            timer.stop(token)

    for _ in range(int(args.t3_warmup)):
        step(record=False)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(int(args.t3_reps)):
        step(record=True)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    else:
        peak_mb = 0.0
    elapsed = time.perf_counter() - started
    phases = timer.totals_ms()
    reps = max(1, int(args.t3_reps))
    total_ms = elapsed * 1000.0 / reps
    mapped_ms = sum(phases.values()) / reps
    unknown_fraction = max(0.0, total_ms - mapped_ms) / max(total_ms, 1.0e-12)
    return {
        "step_time_ms": total_ms,
        "peak_memory_MB": peak_mb,
        "params": int(model.total_parameters()),
        "FLOPs": float(model.total_flops()),
        "forward_time_ms": phases.get("forward", 0.0) / reps,
        "backward_time_ms": phases.get("backward", 0.0) / reps,
        "base_update_time_ms": phases.get("base_update", 0.0) / reps,
        "functional_update_time_ms": 0.0,
        "guard_time_ms": 0.0,
        "validation_time_ms": 0.0,
        "unknown_time_fraction": unknown_fraction,
    }


def _fused_packed_poly2_silu_ce_head_backward(
    head: Any,
    h: torch.Tensor,
    y: torch.Tensor,
    label_smoothing: float,
    *,
    use_compiled_core: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Value-equivalent CE grad + packed poly2-silu head backward for timing probes."""

    with torch.no_grad():
        layer = head.layer
        p = layer.params["poly"]
        b = layer.params["base"]
        mix = layer.params["mix"]
        if bool(use_compiled_core):
            loss, dh, gmix, gbase, gpoly = _compiled_packed_poly2_silu_ce_head_backward_core(
                h,
                y,
                p,
                b,
                mix,
                float(layer.scale),
                float(label_smoothing),
            )
            layer.grads["mix"].add_(gmix)
            layer.grads["base"].add_(gbase)
            layer.grads["poly"].add_(gpoly)
            return loss, dh
        silu_h = F.silu(h)
        z = b[:, 0].unsqueeze(0) * h + b[:, 1].unsqueeze(0) * silu_h + layer.scale * (
            p[:, 0].unsqueeze(0) * h + p[:, 1].unsqueeze(0) * h.square()
        )
        logits = z @ mix.t()
        loss, grad_logits = v85.v83.v72._weighted_smooth_ce_and_grad(logits, y, label_smoothing, None)
        dz = grad_logits @ mix
        layer.grads["mix"].add_(grad_logits.t() @ z)
        sig = torch.sigmoid(h)
        silu_prime = sig * (1.0 + h * (1.0 - sig))
        layer.grads["base"][:, 0].add_((dz * h).sum(dim=0))
        layer.grads["base"][:, 1].add_((dz * silu_h).sum(dim=0))
        dres = dz * layer.scale
        layer.grads["poly"][:, 0].add_((dres * h).sum(dim=0))
        layer.grads["poly"][:, 1].add_((dres * h.square()).sum(dim=0))
        dh = dz * (
            b[:, 0].unsqueeze(0)
            + b[:, 1].unsqueeze(0) * silu_prime
            + layer.scale * (p[:, 0].unsqueeze(0) + 2.0 * p[:, 1].unsqueeze(0) * h)
        )
        return loss, dh


def _fused_packed_poly2_silu_ce_head_backward_with_holdout(
    head: Any,
    h_pair: torch.Tensor,
    split: int,
    y: torch.Tensor,
    y_holdout: torch.Tensor,
    label_smoothing: float,
    *,
    use_compiled_core: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        layer = head.layer
        p = layer.params["poly"]
        b = layer.params["base"]
        mix = layer.params["mix"]
        if bool(use_compiled_core):
            loss, holdout_loss, dh, gmix, gbase, gpoly = _compiled_packed_poly2_silu_ce_head_backward_with_holdout_core(
                h_pair,
                int(split),
                y,
                y_holdout,
                p,
                b,
                mix,
                float(layer.scale),
                float(label_smoothing),
            )
            layer.grads["mix"].add_(gmix)
            layer.grads["base"].add_(gbase)
            layer.grads["poly"].add_(gpoly)
            return loss, holdout_loss, dh
        silu_pair = F.silu(h_pair)
        z_pair = b[:, 0].unsqueeze(0) * h_pair + b[:, 1].unsqueeze(0) * silu_pair + layer.scale * (
            p[:, 0].unsqueeze(0) * h_pair + p[:, 1].unsqueeze(0) * h_pair.square()
        )
        logits_pair = z_pair @ mix.t()
        loss, grad_logits = v85.v83.v72._weighted_smooth_ce_and_grad(
            logits_pair[:split], y, label_smoothing, None
        )
        holdout_loss = v85.v83._smooth_ce_value_only(logits_pair[split:], y_holdout, label_smoothing)
        h = h_pair[:split]
        z = z_pair[:split]
        silu_h = silu_pair[:split]
        dz = grad_logits @ mix
        layer.grads["mix"].add_(grad_logits.t() @ z)
        sig = torch.sigmoid(h)
        silu_prime = sig * (1.0 + h * (1.0 - sig))
        layer.grads["base"][:, 0].add_((dz * h).sum(dim=0))
        layer.grads["base"][:, 1].add_((dz * silu_h).sum(dim=0))
        dres = dz * layer.scale
        layer.grads["poly"][:, 0].add_((dres * h).sum(dim=0))
        layer.grads["poly"][:, 1].add_((dres * h.square()).sum(dim=0))
        dh = dz * (
            b[:, 0].unsqueeze(0)
            + b[:, 1].unsqueeze(0) * silu_prime
            + layer.scale * (p[:, 0].unsqueeze(0) + 2.0 * p[:, 1].unsqueeze(0) * h)
        )
        return loss, holdout_loss, dh


def _smooth_ce_target_from_labels(logits: torch.Tensor, y: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    classes = int(logits.shape[1])
    one_hot = F.one_hot(y, num_classes=classes).to(dtype=logits.dtype, device=logits.device)
    if float(label_smoothing) <= 0.0:
        return one_hot
    eps = float(label_smoothing)
    return one_hot * (1.0 - eps) + (1.0 - one_hot) * (eps / max(1, classes - 1))


def _packed_poly2_silu_ce_head_backward_core(
    h: torch.Tensor,
    y: torch.Tensor,
    p: torch.Tensor,
    b: torch.Tensor,
    mix: torch.Tensor,
    scale: float,
    label_smoothing: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    silu_h = F.silu(h)
    h2 = h.square()
    z = b[:, 0].unsqueeze(0) * h + b[:, 1].unsqueeze(0) * silu_h + float(scale) * (
        p[:, 0].unsqueeze(0) * h + p[:, 1].unsqueeze(0) * h2
    )
    logits = z @ mix.t()
    logp = F.log_softmax(logits, dim=1)
    probs = logp.exp()
    target = _smooth_ce_target_from_labels(logits, y, float(label_smoothing))
    n = max(1, int(logits.shape[0]))
    loss = -(target * logp).sum(dim=1).mean()
    grad_logits = (probs - target) / n
    dz = grad_logits @ mix
    gmix = grad_logits.t() @ z
    sig = torch.sigmoid(h)
    silu_prime = sig * (1.0 + h * (1.0 - sig))
    gbase = torch.stack(((dz * h).sum(dim=0), (dz * silu_h).sum(dim=0)), dim=1)
    dres = dz * float(scale)
    gpoly = torch.stack(((dres * h).sum(dim=0), (dres * h2).sum(dim=0)), dim=1)
    dh = dz * (
        b[:, 0].unsqueeze(0)
        + b[:, 1].unsqueeze(0) * silu_prime
        + float(scale) * (p[:, 0].unsqueeze(0) + 2.0 * p[:, 1].unsqueeze(0) * h)
    )
    return loss, dh, gmix, gbase, gpoly


def _packed_poly2_silu_ce_head_backward_with_holdout_core(
    h_pair: torch.Tensor,
    split: int,
    y: torch.Tensor,
    y_holdout: torch.Tensor,
    p: torch.Tensor,
    b: torch.Tensor,
    mix: torch.Tensor,
    scale: float,
    label_smoothing: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    silu_pair = F.silu(h_pair)
    h2_pair = h_pair.square()
    z_pair = b[:, 0].unsqueeze(0) * h_pair + b[:, 1].unsqueeze(0) * silu_pair + float(scale) * (
        p[:, 0].unsqueeze(0) * h_pair + p[:, 1].unsqueeze(0) * h2_pair
    )
    logits_pair = z_pair @ mix.t()
    train_logits = logits_pair[: int(split)]
    holdout_logits = logits_pair[int(split) :]
    logp = F.log_softmax(train_logits, dim=1)
    probs = logp.exp()
    target = _smooth_ce_target_from_labels(train_logits, y, float(label_smoothing))
    n = max(1, int(train_logits.shape[0]))
    loss = -(target * logp).sum(dim=1).mean()
    grad_logits = (probs - target) / n
    holdout_target = _smooth_ce_target_from_labels(holdout_logits, y_holdout, float(label_smoothing))
    holdout_logp = F.log_softmax(holdout_logits, dim=1)
    holdout_loss = -(holdout_target * holdout_logp).sum(dim=1).mean()
    h = h_pair[: int(split)]
    z = z_pair[: int(split)]
    silu_h = silu_pair[: int(split)]
    h2 = h2_pair[: int(split)]
    dz = grad_logits @ mix
    gmix = grad_logits.t() @ z
    sig = torch.sigmoid(h)
    silu_prime = sig * (1.0 + h * (1.0 - sig))
    gbase = torch.stack(((dz * h).sum(dim=0), (dz * silu_h).sum(dim=0)), dim=1)
    dres = dz * float(scale)
    gpoly = torch.stack(((dres * h).sum(dim=0), (dres * h2).sum(dim=0)), dim=1)
    dh = dz * (
        b[:, 0].unsqueeze(0)
        + b[:, 1].unsqueeze(0) * silu_prime
        + float(scale) * (p[:, 0].unsqueeze(0) + 2.0 * p[:, 1].unsqueeze(0) * h)
    )
    return loss, holdout_loss, dh, gmix, gbase, gpoly


_COMPILED_PACKED_HEAD_CORE: Any | None = None
_COMPILED_PACKED_HEAD_HOLDOUT_CORE: Any | None = None
_V85_FUSED_USE_COMPILED_CORE = False


def _compiled_packed_poly2_silu_ce_head_backward_core(*args: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    global _COMPILED_PACKED_HEAD_CORE
    if _COMPILED_PACKED_HEAD_CORE is None:
        try:
            _COMPILED_PACKED_HEAD_CORE = torch.compile(  # type: ignore[attr-defined]
                _packed_poly2_silu_ce_head_backward_core,
                mode="reduce-overhead",
                fullgraph=False,
            )
        except Exception:
            _COMPILED_PACKED_HEAD_CORE = _packed_poly2_silu_ce_head_backward_core
    return _COMPILED_PACKED_HEAD_CORE(*args)


def _compiled_packed_poly2_silu_ce_head_backward_with_holdout_core(*args: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    global _COMPILED_PACKED_HEAD_HOLDOUT_CORE
    if _COMPILED_PACKED_HEAD_HOLDOUT_CORE is None:
        try:
            _COMPILED_PACKED_HEAD_HOLDOUT_CORE = torch.compile(  # type: ignore[attr-defined]
                _packed_poly2_silu_ce_head_backward_with_holdout_core,
                mode="reduce-overhead",
                fullgraph=False,
            )
        except Exception:
            _COMPILED_PACKED_HEAD_HOLDOUT_CORE = _packed_poly2_silu_ce_head_backward_with_holdout_core
    return _COMPILED_PACKED_HEAD_HOLDOUT_CORE(*args)


def _manual_ce_backward_v85_fused(
    stack: Any,
    head: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    spec: Any,
    *,
    need_loss_float: bool,
) -> float | None:
    v85.v83._zero_grad(stack, head)
    h, caches = stack.forward_manual(x)
    if hasattr(head, "layer"):
        loss, dh = _fused_packed_poly2_silu_ce_head_backward(
            head,
            h,
            y,
            spec.label_smoothing,
            use_compiled_core=bool(_V85_FUSED_USE_COMPILED_CORE),
        )
    else:
        logits, head_cache = head.forward_manual(h)
        loss, grad_logits = v85.v83.v72._weighted_smooth_ce_and_grad(logits, y, spec.label_smoothing, None)
        dh = head.backward_manual(grad_logits, head_cache)
    stack.backward_manual(dh, caches)
    if need_loss_float:
        return float(loss.detach().cpu())
    return None


def _manual_ce_backward_with_holdout_v85_fused(
    stack: Any,
    head: Any,
    xb: torch.Tensor,
    yb: torch.Tensor,
    xh: torch.Tensor,
    yh: torch.Tensor,
    spec: Any,
) -> Tuple[float, float]:
    v85.v83._zero_grad(stack, head)
    split = int(xb.shape[0])
    x_pair = torch.cat((xb, xh), dim=0)
    h_pair, caches = stack.forward_manual(x_pair)
    if hasattr(head, "layer"):
        loss, holdout_loss, dh = _fused_packed_poly2_silu_ce_head_backward_with_holdout(
            head,
            h_pair,
            split,
            yb,
            yh,
            spec.label_smoothing,
            use_compiled_core=bool(_V85_FUSED_USE_COMPILED_CORE),
        )
    else:
        logits_pair, head_cache = head.forward_manual(h_pair)
        loss, grad_logits = v85.v83.v72._weighted_smooth_ce_and_grad(
            logits_pair[:split], yb, spec.label_smoothing, None
        )
        holdout_loss = v85.v83._smooth_ce_value_only(logits_pair[split:], yh, spec.label_smoothing)
        dh = head.backward_manual(grad_logits, v85._slice_cache_batch_v85(head_cache, split))
    stack.backward_manual(dh, v85._slice_stack_caches_v85(caches, split, xb))
    return float(loss.detach().cpu()), float(holdout_loss.detach().cpu())


def _make_v85_compiled_fused_prewarm_hook() -> Any:
    def _hook(
        *,
        stack: Any,
        head: Any,
        spec: Any,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        device: torch.device,
        batch_size: int,
    ) -> None:
        if not hasattr(head, "layer") or int(x_train.shape[0]) <= 0:
            return
        batch = min(int(batch_size), int(x_train.shape[0]))
        xb = x_train[:batch].to(device, non_blocking=True)
        yb = y_train[:batch].to(device, non_blocking=True)
        if int(x_train.shape[0]) >= batch * 2:
            xh = x_train[batch : batch * 2].to(device, non_blocking=True)
            yh = y_train[batch : batch * 2].to(device, non_blocking=True)
        else:
            xh = xb
            yh = yb

        v85.v83._zero_grad(stack, head)
        h, caches = stack.forward_manual(xb)
        _loss, dh = _fused_packed_poly2_silu_ce_head_backward(
            head,
            h,
            yb,
            spec.label_smoothing,
            use_compiled_core=True,
        )
        stack.backward_manual(dh, caches)
        v85.v83._zero_grad(stack, head)

        split = int(xb.shape[0])
        x_pair = torch.cat((xb, xh), dim=0)
        h_pair, caches_pair = stack.forward_manual(x_pair)
        _loss2, _holdout_loss, dh2 = _fused_packed_poly2_silu_ce_head_backward_with_holdout(
            head,
            h_pair,
            split,
            yb,
            yh,
            spec.label_smoothing,
            use_compiled_core=True,
        )
        stack.backward_manual(dh2, v85._slice_stack_caches_v85(caches_pair, split, xb))
        v85.v83._zero_grad(stack, head)
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize()

    return _hook


def _timed_dg_train_step(
    args: argparse.Namespace,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    device: torch.device,
    input_dim: int,
    num_classes: int,
    functional_update_used: bool,
    hidden_dim: int = 28,
    dg_candidate: str = "KW6",
) -> Dict[str, Any]:
    v85.v83.v80._patch_for_v80()
    v85.v83.P5_FT7_EVENT_STRIDE = int(args.fixed_ft7_event_stride)
    v85.v83.P5_FT7_EVENT_ALPHA_MULT = float(args.fixed_ft7_event_alpha_mult)
    spec = v85.v83.v80._spec_map_v80()[dg_candidate]
    v85.v83.set_seed(v85.v83.v72._stable_seed(
        "v87-t3-init",
        str(dg_candidate),
        int(args.t3_seed),
        int(args.t3_batch_size),
        int(hidden_dim),
        int(functional_update_used),
    ))
    stack, head = v85.v83.v80._make_manual_candidate_v80(
        spec,
        int(input_dim),
        int(num_classes),
        int(hidden_dim),
        8,
        device,
    )
    v85._configure_dg_update_modes(dg_candidate, stack, head)
    xb = x[: int(args.t3_batch_size)].to(device)
    yb = y[: int(args.t3_batch_size)].to(device)
    xh = x[int(args.t3_batch_size) : int(args.t3_batch_size) * 2].to(device)
    yh = y[int(args.t3_batch_size) : int(args.t3_batch_size) * 2].to(device)
    if int(xh.shape[0]) == 0:
        xh = xb
        yh = yb
    params = v85.v83.v80.V63Params(train_size=int(x.shape[0]), val_size=0, test_size=0, batch_size=int(args.t3_batch_size))
    lr = float(params.lr_manual * spec.lr_mult)
    if bool(args.use_foreach_adamw_addcdiv):
        optimizer_impl = "ForeachAdamWAddcdivNoSync"
        opt_cls = _ForeachAdamWAddcdivNoSync
    elif bool(args.use_cached_adamw_param_rows):
        optimizer_impl = "CachedParamRowsAdamWNoSync"
        opt_cls = _CachedParamRowsAdamWNoSync
    else:
        optimizer_impl = "FastAdamWNoSync"
        opt_cls = v85.v83.v72.FastAdamWNoSync
    opt = opt_cls([stack, head], lr=lr, weight_decay=1.0e-4)
    entries = v85.v83._param_entries(stack, head)
    role_entries_by_name = v85.v83._entries_by_role(entries)
    event_count = 0
    accept_mass = 0.0
    reject_mass = 0.0
    global_step = 0
    timer = _PhaseTimer(device)
    profile_subphases = bool(getattr(args, "profile_dg_subphases", False))
    use_fused_ce_head_backward = bool(getattr(args, "use_fused_ce_head_backward", False)) and hasattr(head, "layer")
    use_compiled_fused_ce_head_backward = bool(
        use_fused_ce_head_backward and getattr(args, "use_compiled_fused_ce_head_backward", False)
    )
    if bool(use_compiled_fused_ce_head_backward and getattr(args, "prewarm_compiled_fused_ce_head_backward", False)):
        with torch.no_grad():
            v85.v83._zero_grad(stack, head)
            h_warm, _caches_warm = stack.forward_manual(xb)
            _fused_packed_poly2_silu_ce_head_backward(
                head,
                h_warm,
                yb,
                spec.label_smoothing,
                use_compiled_core=True,
            )
            split_warm = int(xb.shape[0])
            h_pair_warm, _pair_caches_warm = stack.forward_manual(torch.cat((xb, xh), dim=0))
            _fused_packed_poly2_silu_ce_head_backward_with_holdout(
                head,
                h_pair_warm,
                split_warm,
                yb,
                yh,
                spec.label_smoothing,
                use_compiled_core=True,
            )
            v85.v83._zero_grad(stack, head)
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()

    def step(*, record: bool) -> None:
        nonlocal global_step, event_count, accept_mass, reject_mass
        global_step += 1
        event_step = bool(functional_update_used) and global_step % int(args.fixed_ft7_event_stride) == 0
        use_pre_holdout = bool(event_step and v85.v83._ft7_uses_pre_holdout_for_step(global_step))
        v85.v83._zero_grad(stack, head)
        if use_pre_holdout:
            split = int(xb.shape[0])
            x_pair = torch.cat((xb, xh), dim=0)
            if profile_subphases:
                token = timer.start("stack_forward") if record else None
                h_pair, caches = stack.forward_manual(x_pair)
                if token is not None:
                    timer.stop(token)
                if use_fused_ce_head_backward:
                    token = timer.start("fused_head_ce_backward") if record else None
                    loss, holdout_loss, dh = _fused_packed_poly2_silu_ce_head_backward_with_holdout(
                        head,
                        h_pair,
                        split,
                        yb,
                        yh,
                        spec.label_smoothing,
                        use_compiled_core=use_compiled_fused_ce_head_backward,
                    )
                    if token is not None:
                        timer.stop(token)
                    head_cache = None
                    grad_logits = None
                else:
                    token = timer.start("head_forward") if record else None
                    logits_pair, head_cache = head.forward_manual(h_pair)
                    if token is not None:
                        timer.stop(token)
                    token = timer.start("ce_grad") if record else None
                    loss, grad_logits = v85.v83.v72._weighted_smooth_ce_and_grad(
                        logits_pair[:split], yb, spec.label_smoothing, None
                    )
                    holdout_loss = v85.v83._smooth_ce_value_only(logits_pair[split:], yh, spec.label_smoothing)
                    if token is not None:
                        timer.stop(token)
            else:
                token = timer.start("forward") if record else None
                h_pair, caches = stack.forward_manual(x_pair)
                if use_fused_ce_head_backward:
                    loss, holdout_loss, dh = _fused_packed_poly2_silu_ce_head_backward_with_holdout(
                        head,
                        h_pair,
                        split,
                        yb,
                        yh,
                        spec.label_smoothing,
                        use_compiled_core=use_compiled_fused_ce_head_backward,
                    )
                    head_cache = None
                    grad_logits = None
                else:
                    logits_pair, head_cache = head.forward_manual(h_pair)
                    loss, grad_logits = v85.v83.v72._weighted_smooth_ce_and_grad(
                        logits_pair[:split], yb, spec.label_smoothing, None
                    )
                    holdout_loss = v85.v83._smooth_ce_value_only(logits_pair[split:], yh, spec.label_smoothing)
                if token is not None:
                    timer.stop(token)
            loss_before = float(loss.detach().cpu())
            holdout_loss_before = float(holdout_loss.detach().cpu())
            if profile_subphases:
                if not use_fused_ce_head_backward:
                    token = timer.start("head_backward") if record else None
                    dh = head.backward_manual(grad_logits, v85._slice_cache_batch_v85(head_cache, split))
                    if token is not None:
                        timer.stop(token)
                token = timer.start("stack_backward") if record else None
                stack.backward_manual(dh, v85._slice_stack_caches_v85(caches, split, xb))
                if token is not None:
                    timer.stop(token)
            else:
                token = timer.start("backward") if record else None
                if not use_fused_ce_head_backward:
                    dh = head.backward_manual(grad_logits, v85._slice_cache_batch_v85(head_cache, split))
                stack.backward_manual(dh, v85._slice_stack_caches_v85(caches, split, xb))
                if token is not None:
                    timer.stop(token)
            del x_pair, h_pair, caches, head_cache, grad_logits
        else:
            holdout_loss_before = 0.0
            if profile_subphases:
                token = timer.start("stack_forward") if record else None
                h, caches = stack.forward_manual(xb)
                if token is not None:
                    timer.stop(token)
                if use_fused_ce_head_backward:
                    token = timer.start("fused_head_ce_backward") if record else None
                    loss, dh = _fused_packed_poly2_silu_ce_head_backward(
                        head,
                        h,
                        yb,
                        spec.label_smoothing,
                        use_compiled_core=use_compiled_fused_ce_head_backward,
                    )
                    if token is not None:
                        timer.stop(token)
                    head_cache = None
                    grad_logits = None
                else:
                    token = timer.start("head_forward") if record else None
                    logits, head_cache = head.forward_manual(h)
                    if token is not None:
                        timer.stop(token)
                    token = timer.start("ce_grad") if record else None
                    loss, grad_logits = v85.v83.v72._weighted_smooth_ce_and_grad(logits, yb, spec.label_smoothing, None)
                    if token is not None:
                        timer.stop(token)
            else:
                token = timer.start("forward") if record else None
                h, caches = stack.forward_manual(xb)
                if use_fused_ce_head_backward:
                    loss, dh = _fused_packed_poly2_silu_ce_head_backward(
                        head,
                        h,
                        yb,
                        spec.label_smoothing,
                        use_compiled_core=use_compiled_fused_ce_head_backward,
                    )
                    head_cache = None
                    grad_logits = None
                else:
                    logits, head_cache = head.forward_manual(h)
                    loss, grad_logits = v85.v83.v72._weighted_smooth_ce_and_grad(logits, yb, spec.label_smoothing, None)
                if token is not None:
                    timer.stop(token)
            loss_before = float(loss.detach().cpu()) if event_step else 0.0
            if profile_subphases:
                if not use_fused_ce_head_backward:
                    token = timer.start("head_backward") if record else None
                    dh = head.backward_manual(grad_logits, head_cache)
                    if token is not None:
                        timer.stop(token)
                token = timer.start("stack_backward") if record else None
                stack.backward_manual(dh, caches)
                if token is not None:
                    timer.stop(token)
            else:
                token = timer.start("backward") if record else None
                if not use_fused_ce_head_backward:
                    dh = head.backward_manual(grad_logits, head_cache)
                stack.backward_manual(dh, caches)
                if token is not None:
                    timer.stop(token)
            del h, caches, head_cache, grad_logits
        token = timer.start("base_update") if record else None
        opt.step(global_step, int(args.t3_warmup) + int(args.t3_reps), warmup_cosine=True)
        if token is not None:
            timer.stop(token)
        if event_step:
            token = timer.start("guard") if record else None
            loss_after, holdout_loss_after, task_features_after, holdout_features_after = v85.v83._loss_pair_and_features_only(
                stack,
                head,
                xb,
                yb,
                xh,
                yh,
                spec,
            )
            if token is not None:
                timer.stop(token)
            func_alpha = lr * 0.05 * float(args.fixed_ft7_event_alpha_mult)
            token = timer.start("functional_update") if record else None
            _loss_after, _func_norm, trust_delta, reject_delta = v85.v83._apply_ft7_streamed_guarded_update(
                stack,
                head,
                entries,
                xb,
                yb,
                xh,
                yh,
                spec,
                loss_before=loss_before,
                holdout_loss_before=holdout_loss_before,
                task_loss_after=loss_after,
                holdout_loss_after=holdout_loss_after,
                func_alpha=func_alpha,
                task_features_after=task_features_after,
                holdout_features_after=holdout_features_after,
                role_entries_by_name=role_entries_by_name,
                pair_stack_guard_forward=True,
            )
            if token is not None:
                timer.stop(token)
            event_count += 1
            accept_mass += float(trust_delta)
            reject_mass += float(reject_delta)

    for _ in range(int(args.t3_warmup)):
        step(record=False)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(int(args.t3_reps)):
        step(record=True)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    else:
        peak_mb = 0.0
    elapsed = time.perf_counter() - started
    phases = timer.totals_ms()
    reps = max(1, int(args.t3_reps))
    total_ms = elapsed * 1000.0 / reps
    mapped_ms = sum(phases.values()) / reps
    unknown_fraction = max(0.0, total_ms - mapped_ms) / max(total_ms, 1.0e-12)
    stack_forward_ms = phases.get("stack_forward", 0.0) / reps
    head_forward_ms = phases.get("head_forward", 0.0) / reps
    ce_grad_ms = phases.get("ce_grad", 0.0) / reps
    head_backward_ms = phases.get("head_backward", 0.0) / reps
    stack_backward_ms = phases.get("stack_backward", 0.0) / reps
    fused_head_ce_backward_ms = phases.get("fused_head_ce_backward", 0.0) / reps
    forward_ms = phases.get("forward", 0.0) / reps + stack_forward_ms + head_forward_ms + ce_grad_ms
    backward_ms = phases.get("backward", 0.0) / reps + head_backward_ms + stack_backward_ms + fused_head_ce_backward_ms
    return {
        "step_time_ms": total_ms,
        "peak_memory_MB": peak_mb,
        "candidate_id": str(dg_candidate),
        "optimizer_impl": optimizer_impl,
        "ce_head_backward_impl": (
            "compiled_fused_value_equivalent"
            if use_compiled_fused_ce_head_backward
            else ("fused_value_equivalent" if use_fused_ce_head_backward else "v85_default")
        ),
        "compiled_fused_ce_head_prewarm_used": int(
            bool(use_compiled_fused_ce_head_backward and getattr(args, "prewarm_compiled_fused_ce_head_backward", False))
        ),
        "params": int(stack.param_count() + head.param_count()),
        "FLOPs": v85._dg_kw6_forward_flops_estimate(int(input_dim), int(hidden_dim), int(spec.depth), int(num_classes)),
        "functional_events": event_count,
        "functional_accept_mass": accept_mass,
        "functional_reject_mass": reject_mass,
        "forward_time_ms": forward_ms,
        "backward_time_ms": backward_ms,
        "stack_forward_time_ms": stack_forward_ms,
        "head_forward_time_ms": head_forward_ms,
        "ce_grad_time_ms": ce_grad_ms,
        "head_backward_time_ms": head_backward_ms,
        "stack_backward_time_ms": stack_backward_ms,
        "fused_head_ce_backward_time_ms": fused_head_ce_backward_ms,
        "base_update_time_ms": phases.get("base_update", 0.0) / reps,
        "functional_update_time_ms": phases.get("functional_update", 0.0) / reps,
        "guard_time_ms": phases.get("guard", 0.0) / reps,
        "validation_time_ms": 0.0,
        "unknown_time_fraction": unknown_fraction,
    }


def _train_step_timing_row(
    args: argparse.Namespace,
    out_dir: Path,
    *,
    timing_protocol: str,
    protocol_scope: str,
    source_row_kind: str,
    seed: int,
    warmup: int,
    reps: int,
    batch_size: int,
    hidden_dim: int = 28,
    dg_candidate: str = "KW6",
) -> Dict[str, Any]:
    old_seed = int(args.t3_seed)
    old_warmup = int(args.t3_warmup)
    old_reps = int(args.t3_reps)
    old_batch_size = int(args.t3_batch_size)
    args.t3_seed = int(seed)
    args.t3_warmup = int(warmup)
    args.t3_reps = int(reps)
    args.t3_batch_size = int(batch_size)
    try:
        device = v85.v83.get_device(args.device)
        x_train, y_train, _x_test, _y_test, input_dim, num_classes, protocol = v85._load_kanbefair_vision_tensors(
            "MNIST",
            data_root=Path(args.data_root),
            train_size=max(int(batch_size) * 4, 512),
            test_size=256,
            seed=int(seed),
        )
        kb = _timed_kb_mlp_train_step(args, x_train, y_train, device=device, input_dim=input_dim, num_classes=num_classes)
        dg0 = _timed_dg_train_step(
            args,
            x_train,
            y_train,
            device=device,
            input_dim=input_dim,
            num_classes=num_classes,
            functional_update_used=False,
            hidden_dim=int(hidden_dim),
            dg_candidate=str(dg_candidate),
        )
        dg1 = _timed_dg_train_step(
            args,
            x_train,
            y_train,
            device=device,
            input_dim=input_dim,
            num_classes=num_classes,
            functional_update_used=True,
            hidden_dim=int(hidden_dim),
            dg_candidate=str(dg_candidate),
        )
    finally:
        args.t3_seed = old_seed
        args.t3_warmup = old_warmup
        args.t3_reps = old_reps
        args.t3_batch_size = old_batch_size

    kb_step = _safe_float(kb.get("step_time_ms"))
    dg1_step = _safe_float(dg1.get("step_time_ms"))
    ratio = dg1_step / max(kb_step, 1.0e-12) if math.isfinite(kb_step) and math.isfinite(dg1_step) else float("nan")
    kb_params = _safe_float(kb.get("params"))
    kb_flops = _safe_float(kb.get("FLOPs"))
    dg_params = _safe_float(dg1.get("params"))
    dg_flops = _safe_float(dg1.get("FLOPs"))
    params_ratio = dg_params / kb_params if math.isfinite(dg_params) and math.isfinite(kb_params) and kb_params > 0.0 else float("nan")
    flops_ratio = dg_flops / kb_flops if math.isfinite(dg_flops) and math.isfinite(kb_flops) and kb_flops > 0.0 else float("nan")
    protocol_label = str(timing_protocol).replace(" ", "_")
    return {
        "stage": "P5_ROBUST_TIMING_PROTOCOLS_V87",
        "status": "measured",
        "timing_protocol": timing_protocol,
        "protocol_scope": protocol_scope,
        "source_out_dir": str(out_dir),
        "source_artifact": "measured_by_run_gafu_v87_real.py",
        "source_artifact_sha256": _sha256(ROOT_DIR / "experiments/run_gafu_v87_real.py"),
        "source_row_kind": source_row_kind,
        "run_id": f"{protocol_label}_seed{int(seed)}_warmup{int(warmup)}_reps{int(reps)}",
        "seed": int(seed),
        "dataset_protocol": protocol,
        "warmup_steps": int(warmup),
        "reps": int(reps),
        "batch_size": int(batch_size),
        "hidden_dim": int(hidden_dim),
        "ft7_event_stride": int(args.fixed_ft7_event_stride),
        "ft7_event_alpha_mult": float(args.fixed_ft7_event_alpha_mult),
        "candidate_id": f"DG1-FT7-{str(dg_candidate)}-hidden{int(hidden_dim)}-functional",
        "optimizer_impl": dg1.get("optimizer_impl", METRIC_UNAVAILABLE),
        "ce_head_backward_impl": dg1.get("ce_head_backward_impl", METRIC_UNAVAILABLE),
        "compiled_fused_ce_head_prewarm_used": dg1.get("compiled_fused_ce_head_prewarm_used", METRIC_UNAVAILABLE),
        "kb_mlp_params": int(kb_params) if math.isfinite(kb_params) else METRIC_UNAVAILABLE,
        "kb_mlp_forward_FLOPs": kb_flops if math.isfinite(kb_flops) else METRIC_UNAVAILABLE,
        "params": int(dg_params) if math.isfinite(dg_params) else METRIC_UNAVAILABLE,
        "forward_FLOPs": dg_flops if math.isfinite(dg_flops) else METRIC_UNAVAILABLE,
        "params_ratio_vs_KB_MLP": params_ratio if math.isfinite(params_ratio) else METRIC_UNAVAILABLE,
        "flops_ratio_vs_KB_MLP": flops_ratio if math.isfinite(flops_ratio) else METRIC_UNAVAILABLE,
        "kb_mlp_step_time_ms": kb_step if math.isfinite(kb_step) else METRIC_UNAVAILABLE,
        "kb_mlp_forward_time_ms": kb.get("forward_time_ms", METRIC_UNAVAILABLE),
        "kb_mlp_backward_time_ms": kb.get("backward_time_ms", METRIC_UNAVAILABLE),
        "kb_mlp_update_time_ms": kb.get("base_update_time_ms", METRIC_UNAVAILABLE),
        "dg0_step_time_ms": dg0.get("step_time_ms", METRIC_UNAVAILABLE),
        "dg0_forward_time_ms": dg0.get("forward_time_ms", METRIC_UNAVAILABLE),
        "dg0_backward_time_ms": dg0.get("backward_time_ms", METRIC_UNAVAILABLE),
        "dg0_stack_forward_time_ms": dg0.get("stack_forward_time_ms", METRIC_UNAVAILABLE),
        "dg0_head_forward_time_ms": dg0.get("head_forward_time_ms", METRIC_UNAVAILABLE),
        "dg0_ce_grad_time_ms": dg0.get("ce_grad_time_ms", METRIC_UNAVAILABLE),
        "dg0_head_backward_time_ms": dg0.get("head_backward_time_ms", METRIC_UNAVAILABLE),
        "dg0_stack_backward_time_ms": dg0.get("stack_backward_time_ms", METRIC_UNAVAILABLE),
        "dg0_fused_head_ce_backward_time_ms": dg0.get("fused_head_ce_backward_time_ms", METRIC_UNAVAILABLE),
        "dg0_update_time_ms": dg0.get("base_update_time_ms", METRIC_UNAVAILABLE),
        "step_time_ms": dg1_step if math.isfinite(dg1_step) else METRIC_UNAVAILABLE,
        "step_ratio_vs_KB_MLP": ratio if math.isfinite(ratio) else METRIC_UNAVAILABLE,
        "forward_time_ms": dg1.get("forward_time_ms", METRIC_UNAVAILABLE),
        "backward_time_ms": dg1.get("backward_time_ms", METRIC_UNAVAILABLE),
        "stack_forward_time_ms": dg1.get("stack_forward_time_ms", METRIC_UNAVAILABLE),
        "head_forward_time_ms": dg1.get("head_forward_time_ms", METRIC_UNAVAILABLE),
        "ce_grad_time_ms": dg1.get("ce_grad_time_ms", METRIC_UNAVAILABLE),
        "head_backward_time_ms": dg1.get("head_backward_time_ms", METRIC_UNAVAILABLE),
        "stack_backward_time_ms": dg1.get("stack_backward_time_ms", METRIC_UNAVAILABLE),
        "fused_head_ce_backward_time_ms": dg1.get("fused_head_ce_backward_time_ms", METRIC_UNAVAILABLE),
        "base_update_time_ms": dg1.get("base_update_time_ms", METRIC_UNAVAILABLE),
        "functional_update_time_ms": dg1.get("functional_update_time_ms", METRIC_UNAVAILABLE),
        "guard_time_ms": dg1.get("guard_time_ms", METRIC_UNAVAILABLE),
        "validation_time_ms": dg1.get("validation_time_ms", METRIC_UNAVAILABLE),
        "cuda_sync_time_ms": METRIC_UNAVAILABLE,
        "kernel_time_ms": (
            _safe_float(dg1.get("forward_time_ms"), 0.0)
            + _safe_float(dg1.get("backward_time_ms"), 0.0)
            + _safe_float(dg1.get("base_update_time_ms"), 0.0)
            + _safe_float(dg1.get("functional_update_time_ms"), 0.0)
            + _safe_float(dg1.get("guard_time_ms"), 0.0)
        ),
        "unknown_time_fraction": dg1.get("unknown_time_fraction", METRIC_UNAVAILABLE),
        "peak_memory_MB": dg1.get("peak_memory_MB", METRIC_UNAVAILABLE),
        "functional_events": dg1.get("functional_events", METRIC_UNAVAILABLE),
        "functional_accept_mass": dg1.get("functional_accept_mass", METRIC_UNAVAILABLE),
        "functional_reject_mass": dg1.get("functional_reject_mass", METRIC_UNAVAILABLE),
        "timing_gate_pass": int(math.isfinite(ratio) and ratio <= 1.50),
        "all_gate_pass": METRIC_UNAVAILABLE,
        "loss_type": "CE",
        "geometry_loss_used": 0,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "sampler_changed": 0,
        "class_weight_used": 0,
        "cpu_offload_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }


def _parse_timing_protocol_specs(text: str) -> List[Tuple[str, int, int]]:
    specs: List[Tuple[str, int, int]] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        fields = [x.strip() for x in part.split(":")]
        if len(fields) != 3:
            raise ValueError(f"invalid timing protocol spec {part!r}; expected NAME:WARMUP:REPS")
        specs.append((fields[0], int(fields[1]), int(fields[2])))
    return specs


def _timing_candidate_id(args: argparse.Namespace) -> str:
    value = str(getattr(args, "timing_dg_candidate_id", "")).strip()
    return value if value else str(getattr(args, "p0_dg_candidate_id", "KW6"))


def _timing_hidden_dim(args: argparse.Namespace) -> int:
    value = int(getattr(args, "timing_dg_hidden_dim", 0))
    return value if value > 0 else int(getattr(args, "p0_dg_hidden_dim", 28))


def _run_t0_t2_phase_clean_timing(args: argparse.Namespace, out_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = _read_csv(out_dir / "robust_timing_protocols.csv")
    for name, warmup, reps in _parse_timing_protocol_specs(args.t0_t2_protocols):
        rows.append(_train_step_timing_row(
            args,
            out_dir,
            timing_protocol=f"{name}-train-step-phase-clean",
            protocol_scope="same_real_MNIST_batch_train_step_only_pre_registered_warmup_reps_protocol_no_eval_loop",
            source_row_kind="t0_t2_phase_clean",
            seed=int(args.t0_t2_seed),
            warmup=int(warmup),
            reps=int(reps),
            batch_size=int(args.t0_t2_batch_size),
            hidden_dim=_timing_hidden_dim(args),
            dg_candidate=_timing_candidate_id(args),
        ))
    summary = _write_robust_timing_files(out_dir, rows)
    return rows, summary


def _run_t3_phase_clean_timing(args: argparse.Namespace, out_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = _read_csv(out_dir / "robust_timing_protocols.csv")
    rows.append(_train_step_timing_row(
        args,
        out_dir,
        timing_protocol="T3-train-step-phase-clean",
        protocol_scope="same_real_MNIST_batch_train_step_only_with_FT7_event_cadence_no_eval_loop",
        source_row_kind="t3_phase_clean",
        seed=int(args.t3_seed),
        warmup=int(args.t3_warmup),
        reps=int(args.t3_reps),
        batch_size=int(args.t3_batch_size),
        hidden_dim=_timing_hidden_dim(args),
        dg_candidate=_timing_candidate_id(args),
    ))
    summary = _write_robust_timing_files(out_dir, rows)
    return rows, summary


def _run_p4_hidden_pilot(args: argparse.Namespace, out_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    device = v85.v83.get_device(args.device)
    x_train, y_train, _x_test, _y_test, input_dim, num_classes, protocol = v85._load_kanbefair_vision_tensors(
        "MNIST",
        data_root=Path(args.data_root),
        train_size=max(int(args.t3_batch_size) * 4, 512),
        test_size=256,
        seed=int(args.p4_seed),
    )
    kb = _timed_kb_mlp_train_step(args, x_train, y_train, device=device, input_dim=input_dim, num_classes=num_classes)
    kb_params = _safe_float(kb.get("params"))
    kb_flops = _safe_float(kb.get("FLOPs"))
    kb_step = _safe_float(kb.get("step_time_ms"))
    rows: List[Dict[str, Any]] = []
    for hidden_dim in _parse_ints(args.p4_hidden_candidates):
        dg = _timed_dg_train_step(
            args,
            x_train,
            y_train,
            device=device,
            input_dim=input_dim,
            num_classes=num_classes,
            functional_update_used=False,
            hidden_dim=int(hidden_dim),
        )
        params = _safe_float(dg.get("params"))
        flops = _safe_float(dg.get("FLOPs"))
        step = _safe_float(dg.get("step_time_ms"))
        params_ratio = params / kb_params if math.isfinite(params) and math.isfinite(kb_params) and kb_params > 0 else float("nan")
        flops_ratio = flops / kb_flops if math.isfinite(flops) and math.isfinite(kb_flops) and kb_flops > 0 else float("nan")
        step_ratio = step / kb_step if math.isfinite(step) and math.isfinite(kb_step) and kb_step > 0 else float("nan")
        selection_pass = int(
            math.isfinite(params_ratio)
            and params_ratio <= 1.0
            and math.isfinite(flops_ratio)
            and flops_ratio <= 1.0
            and math.isfinite(step_ratio)
            and step_ratio <= 1.50
        )
        rows.append({
            "stage": "P4_AUTO_ARCHITECTURE_SELECTION_V87",
            "status": "measured",
            "pilot_type": "system_envelope_only_no_test_metric",
            "dataset_protocol": protocol,
            "seed": int(args.p4_seed),
            "hidden_dim": int(hidden_dim),
            "candidate_id": f"DG0-KW6-hidden{int(hidden_dim)}-base",
            "kb_mlp_params": int(kb_params) if math.isfinite(kb_params) else METRIC_UNAVAILABLE,
            "kb_mlp_forward_FLOPs": kb_flops if math.isfinite(kb_flops) else METRIC_UNAVAILABLE,
            "kb_mlp_step_time_ms": kb_step if math.isfinite(kb_step) else METRIC_UNAVAILABLE,
            "params": int(params) if math.isfinite(params) else METRIC_UNAVAILABLE,
            "forward_FLOPs": flops if math.isfinite(flops) else METRIC_UNAVAILABLE,
            "step_time_ms": step if math.isfinite(step) else METRIC_UNAVAILABLE,
            "params_ratio_vs_KB_MLP": params_ratio if math.isfinite(params_ratio) else METRIC_UNAVAILABLE,
            "flops_ratio_vs_KB_MLP": flops_ratio if math.isfinite(flops_ratio) else METRIC_UNAVAILABLE,
            "step_ratio_vs_KB_MLP": step_ratio if math.isfinite(step_ratio) else METRIC_UNAVAILABLE,
            "forward_time_ms": dg.get("forward_time_ms", METRIC_UNAVAILABLE),
            "backward_time_ms": dg.get("backward_time_ms", METRIC_UNAVAILABLE),
            "base_update_time_ms": dg.get("base_update_time_ms", METRIC_UNAVAILABLE),
            "unknown_time_fraction": dg.get("unknown_time_fraction", METRIC_UNAVAILABLE),
            "selection_pass": selection_pass,
            "selection_rule": "max_hidden_with_params<=1.0_flops<=1.0_step<=1.50",
            "test_metric_used_for_selection": 0,
            "loss_type": "CE",
            "geometry_loss_used": 0,
            "external_teacher_used": 0,
            "self_teacher_used": 0,
            "sampler_changed": 0,
            "class_weight_used": 0,
            "cpu_offload_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    selected = [r for r in rows if int(_safe_float(r.get("selection_pass"), 0.0)) == 1]
    selected_hidden = max([int(r["hidden_dim"]) for r in selected]) if selected else None
    selected_row = next((r for r in rows if selected_hidden is not None and int(r["hidden_dim"]) == selected_hidden), None)
    summary = {
        "stage": "P4_AUTO_ARCHITECTURE_SELECTION_SUMMARY_V87",
        "status": "measured" if rows else "not_run",
        "pilot_type": "system_envelope_only_no_test_metric",
        "candidate_count": len(rows),
        "selected_hidden_dim": selected_hidden if selected_hidden is not None else METRIC_UNAVAILABLE,
        "selection_pass": int(selected_hidden is not None),
        "selected_step_ratio_vs_KB_MLP": (selected_row or {}).get("step_ratio_vs_KB_MLP", METRIC_UNAVAILABLE),
        "selected_params_ratio_vs_KB_MLP": (selected_row or {}).get("params_ratio_vs_KB_MLP", METRIC_UNAVAILABLE),
        "selected_flops_ratio_vs_KB_MLP": (selected_row or {}).get("flops_ratio_vs_KB_MLP", METRIC_UNAVAILABLE),
        "test_metric_used_for_selection": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_csv(out_dir / "auto_architecture_selection.csv", rows)
    write_csv(out_dir / "auto_architecture_selection_summary.csv", [summary])
    return rows, summary


def _run_p4_robust_hidden_selection(args: argparse.Namespace, out_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for hidden_dim in _parse_ints(args.p4_hidden_candidates):
        for name, warmup, reps in _parse_timing_protocol_specs(args.p4_robust_protocols):
            row = _train_step_timing_row(
                args,
                out_dir,
                timing_protocol=f"{name}-hidden{int(hidden_dim)}-train-step-phase-clean",
                protocol_scope="p4_robust_hidden_selection_same_real_MNIST_batch_train_step_only_no_test_metric",
                source_row_kind="p4_robust_hidden_selection",
                seed=int(args.p4_seed),
                warmup=int(warmup),
                reps=int(reps),
                batch_size=int(args.t3_batch_size),
                hidden_dim=int(hidden_dim),
            )
            row["stage"] = "P4_AUTO_ARCHITECTURE_SELECTION_V87"
            row["pilot_type"] = "robust_system_envelope_only_no_test_metric"
            row["selection_rule"] = "max_hidden_with_params<=1.0_flops<=1.0_q90_step<=1.50_across_p4_protocols"
            row["test_metric_used_for_selection"] = 0
            rows.append(row)

    hidden_summaries: List[Dict[str, Any]] = []
    for hidden_dim in _parse_ints(args.p4_hidden_candidates):
        hidden_rows = [r for r in rows if int(_safe_float(r.get("hidden_dim"), -1.0)) == int(hidden_dim)]
        step_values = [_safe_float(r.get("step_ratio_vs_KB_MLP")) for r in hidden_rows]
        finite_steps = [v for v in step_values if math.isfinite(v)]
        q90_step = _q90(finite_steps)
        max_step = max(finite_steps) if finite_steps else float("nan")
        params_ratio_values = [_safe_float(r.get("params_ratio_vs_KB_MLP")) for r in hidden_rows]
        flops_ratio_values = [_safe_float(r.get("flops_ratio_vs_KB_MLP")) for r in hidden_rows]
        params_ratio = next((v for v in params_ratio_values if math.isfinite(v)), float("nan"))
        flops_ratio = next((v for v in flops_ratio_values if math.isfinite(v)), float("nan"))
        pass_value = int(
            math.isfinite(params_ratio)
            and params_ratio <= 1.0
            and math.isfinite(flops_ratio)
            and flops_ratio <= 1.0
            and math.isfinite(q90_step)
            and q90_step <= 1.50
        )
        hidden_summaries.append({
            "stage": "P4_AUTO_ARCHITECTURE_SELECTION_BY_HIDDEN_V87",
            "status": "measured" if hidden_rows else "not_run",
            "pilot_type": "robust_system_envelope_only_no_test_metric",
            "hidden_dim": int(hidden_dim),
            "protocol_count": len({str(r.get("timing_protocol")) for r in hidden_rows}),
            "params_ratio_vs_KB_MLP": params_ratio if math.isfinite(params_ratio) else METRIC_UNAVAILABLE,
            "flops_ratio_vs_KB_MLP": flops_ratio if math.isfinite(flops_ratio) else METRIC_UNAVAILABLE,
            "q90_step_ratio_vs_KB_MLP": q90_step if math.isfinite(q90_step) else METRIC_UNAVAILABLE,
            "max_step_ratio_vs_KB_MLP": max_step if math.isfinite(max_step) else METRIC_UNAVAILABLE,
            "timing_gate_pass_rate": (
                mean([int(_safe_float(r.get("timing_gate_pass"), 0.0) == 1.0) for r in hidden_rows])
                if hidden_rows else 0.0
            ),
            "selection_pass": pass_value,
            "selection_score": int(hidden_dim) if pass_value else METRIC_UNAVAILABLE,
            "selection_rule": "max_hidden_with_params<=1.0_flops<=1.0_q90_step<=1.50_across_p4_protocols",
            "test_metric_used_for_selection": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    selected = [r for r in hidden_summaries if int(_safe_float(r.get("selection_pass"), 0.0)) == 1]
    selected_hidden = max([int(r["hidden_dim"]) for r in selected]) if selected else None
    selected_row = next((r for r in hidden_summaries if selected_hidden is not None and int(r["hidden_dim"]) == selected_hidden), None)
    summary = {
        "stage": "P4_AUTO_ARCHITECTURE_SELECTION_SUMMARY_V87",
        "status": "measured" if rows else "not_run",
        "pilot_type": "robust_system_envelope_only_no_test_metric",
        "candidate_count": len(hidden_summaries),
        "protocol_count": len({str(r.get("timing_protocol")) for r in rows}),
        "selected_hidden_dim": selected_hidden if selected_hidden is not None else METRIC_UNAVAILABLE,
        "selection_pass": int(selected_hidden is not None),
        "selected_step_ratio_vs_KB_MLP": (selected_row or {}).get("q90_step_ratio_vs_KB_MLP", METRIC_UNAVAILABLE),
        "selected_params_ratio_vs_KB_MLP": (selected_row or {}).get("params_ratio_vs_KB_MLP", METRIC_UNAVAILABLE),
        "selected_flops_ratio_vs_KB_MLP": (selected_row or {}).get("flops_ratio_vs_KB_MLP", METRIC_UNAVAILABLE),
        "selection_rule": "max_hidden_with_params<=1.0_flops<=1.0_q90_step<=1.50_across_p4_protocols",
        "test_metric_used_for_selection": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_csv(out_dir / "auto_architecture_selection.csv", rows)
    write_csv(out_dir / "auto_architecture_selection_by_hidden.csv", hidden_summaries)
    write_csv(out_dir / "auto_architecture_selection_summary.csv", [summary])
    return rows, summary


def _run_p4_base_implementation_screen(args: argparse.Namespace, out_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    candidates = [part.strip() for part in str(args.p4_base_candidates).split(",") if part.strip()]
    hidden_dims = _parse_ints(args.p4_base_hidden_candidates) if str(args.p4_base_hidden_candidates).strip() else [int(args.p4_base_hidden_dim)]
    spec_map = v85.v83.v80._spec_map_v80()
    missing = [cid for cid in candidates if cid not in spec_map]
    if missing:
        raise ValueError(f"unknown P4 base implementation candidate(s): {','.join(missing)}")

    old_seed = int(args.t3_seed)
    old_warmup = int(args.t3_warmup)
    old_reps = int(args.t3_reps)
    old_batch_size = int(args.t3_batch_size)
    try:
        device = v85.v83.get_device(args.device)
        for name, warmup, reps in _parse_timing_protocol_specs(args.p4_base_protocols):
            args.t3_seed = int(args.p4_seed)
            args.t3_warmup = int(warmup)
            args.t3_reps = int(reps)
            args.t3_batch_size = int(args.t3_batch_size)
            x_train, y_train, _x_test, _y_test, input_dim, num_classes, protocol = v85._load_kanbefair_vision_tensors(
                "MNIST",
                data_root=Path(args.data_root),
                train_size=max(int(args.t3_batch_size) * 4, 512),
                test_size=256,
                seed=int(args.p4_seed),
            )
            kb = _timed_kb_mlp_train_step(args, x_train, y_train, device=device, input_dim=input_dim, num_classes=num_classes)
            kb_step = _safe_float(kb.get("step_time_ms"))
            kb_params = _safe_float(kb.get("params"))
            kb_flops = _safe_float(kb.get("FLOPs"))
            for hidden_dim in hidden_dims:
                for candidate_id in candidates:
                    dg = _timed_dg_train_step(
                        args,
                        x_train,
                        y_train,
                        device=device,
                        input_dim=input_dim,
                        num_classes=num_classes,
                        functional_update_used=False,
                        hidden_dim=int(hidden_dim),
                        dg_candidate=str(candidate_id),
                    )
                    step = _safe_float(dg.get("step_time_ms"))
                    params = _safe_float(dg.get("params"))
                    flops = _safe_float(dg.get("FLOPs"))
                    step_ratio = step / kb_step if math.isfinite(step) and math.isfinite(kb_step) and kb_step > 0.0 else float("nan")
                    params_ratio = params / kb_params if math.isfinite(params) and math.isfinite(kb_params) and kb_params > 0.0 else float("nan")
                    flops_ratio = flops / kb_flops if math.isfinite(flops) and math.isfinite(kb_flops) and kb_flops > 0.0 else float("nan")
                    row = {
                        "stage": "P4_BASE_IMPLEMENTATION_SCREEN_V87",
                        "status": "measured",
                        "pilot_type": "robust_base_implementation_system_only_no_test_metric",
                        "timing_protocol": f"{name}-base-impl-train-step-phase-clean",
                        "protocol_scope": "p4_base_implementation_screen_same_real_MNIST_batch_train_step_only_no_test_metric",
                        "dataset_protocol": protocol,
                        "seed": int(args.p4_seed),
                        "warmup_steps": int(warmup),
                        "reps": int(reps),
                        "batch_size": int(args.t3_batch_size),
                        "candidate_id": str(candidate_id),
                        "optimizer_impl": dg.get("optimizer_impl", METRIC_UNAVAILABLE),
                        "hidden_dim": int(hidden_dim),
                        "kb_mlp_params": int(kb_params) if math.isfinite(kb_params) else METRIC_UNAVAILABLE,
                        "kb_mlp_forward_FLOPs": kb_flops if math.isfinite(kb_flops) else METRIC_UNAVAILABLE,
                        "kb_mlp_step_time_ms": kb_step if math.isfinite(kb_step) else METRIC_UNAVAILABLE,
                        "kb_mlp_forward_time_ms": kb.get("forward_time_ms", METRIC_UNAVAILABLE),
                        "kb_mlp_backward_time_ms": kb.get("backward_time_ms", METRIC_UNAVAILABLE),
                        "kb_mlp_update_time_ms": kb.get("base_update_time_ms", METRIC_UNAVAILABLE),
                        "params": int(params) if math.isfinite(params) else METRIC_UNAVAILABLE,
                        "forward_FLOPs": flops if math.isfinite(flops) else METRIC_UNAVAILABLE,
                        "params_ratio_vs_KB_MLP": params_ratio if math.isfinite(params_ratio) else METRIC_UNAVAILABLE,
                        "flops_ratio_vs_KB_MLP": flops_ratio if math.isfinite(flops_ratio) else METRIC_UNAVAILABLE,
                        "step_time_ms": step if math.isfinite(step) else METRIC_UNAVAILABLE,
                        "step_ratio_vs_KB_MLP": step_ratio if math.isfinite(step_ratio) else METRIC_UNAVAILABLE,
                        "forward_time_ms": dg.get("forward_time_ms", METRIC_UNAVAILABLE),
                        "backward_time_ms": dg.get("backward_time_ms", METRIC_UNAVAILABLE),
                        "base_update_time_ms": dg.get("base_update_time_ms", METRIC_UNAVAILABLE),
                        "functional_update_time_ms": 0.0,
                        "guard_time_ms": 0.0,
                        "unknown_time_fraction": dg.get("unknown_time_fraction", METRIC_UNAVAILABLE),
                        "peak_memory_MB": dg.get("peak_memory_MB", METRIC_UNAVAILABLE),
                        "timing_gate_pass": int(math.isfinite(step_ratio) and step_ratio <= 1.50),
                        "selection_pass": int(
                            math.isfinite(params_ratio)
                            and params_ratio <= 1.0
                            and math.isfinite(flops_ratio)
                            and flops_ratio <= 1.0
                            and math.isfinite(step_ratio)
                            and step_ratio <= 1.50
                        ),
                        "selection_rule": "base_candidate_hidden_pair_with_params<=1.0_flops<=1.0_q90_step<=1.50_across_p4_protocols",
                        "test_metric_used_for_selection": 0,
                        "loss_type": "CE",
                        "geometry_loss_used": 0,
                        "external_teacher_used": 0,
                        "self_teacher_used": 0,
                        "sampler_changed": 0,
                        "class_weight_used": 0,
                        "cpu_offload_used": 0,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                    }
                    rows.append(row)
    finally:
        args.t3_seed = old_seed
        args.t3_warmup = old_warmup
        args.t3_reps = old_reps
        args.t3_batch_size = old_batch_size

    candidate_summaries: List[Dict[str, Any]] = []
    for candidate_id in candidates:
        for hidden_dim in hidden_dims:
            candidate_rows = [
                r for r in rows
                if str(r.get("candidate_id")) == str(candidate_id)
                and int(_safe_float(r.get("hidden_dim"), -1.0)) == int(hidden_dim)
            ]
            if not candidate_rows:
                continue
            finite_steps = [_safe_float(r.get("step_ratio_vs_KB_MLP")) for r in candidate_rows]
            finite_steps = [v for v in finite_steps if math.isfinite(v)]
            q90_step = _q90(finite_steps)
            max_step = max(finite_steps) if finite_steps else float("nan")
            params_ratio = next((_safe_float(r.get("params_ratio_vs_KB_MLP")) for r in candidate_rows if math.isfinite(_safe_float(r.get("params_ratio_vs_KB_MLP")))), float("nan"))
            flops_ratio = next((_safe_float(r.get("flops_ratio_vs_KB_MLP")) for r in candidate_rows if math.isfinite(_safe_float(r.get("flops_ratio_vs_KB_MLP")))), float("nan"))
            pass_value = int(
                math.isfinite(params_ratio)
                and params_ratio <= 1.0
                and math.isfinite(flops_ratio)
                and flops_ratio <= 1.0
                and math.isfinite(q90_step)
                and q90_step <= 1.50
            )
            candidate_summaries.append({
                "stage": "P4_BASE_IMPLEMENTATION_SCREEN_BY_CANDIDATE_V87",
                "status": "measured",
                "pilot_type": "robust_base_implementation_system_only_no_test_metric",
                "candidate_id": str(candidate_id),
                "hidden_dim": int(hidden_dim),
                "protocol_count": len({str(r.get("timing_protocol")) for r in candidate_rows}),
                "params_ratio_vs_KB_MLP": params_ratio if math.isfinite(params_ratio) else METRIC_UNAVAILABLE,
                "flops_ratio_vs_KB_MLP": flops_ratio if math.isfinite(flops_ratio) else METRIC_UNAVAILABLE,
                "q90_step_ratio_vs_KB_MLP": q90_step if math.isfinite(q90_step) else METRIC_UNAVAILABLE,
                "max_step_ratio_vs_KB_MLP": max_step if math.isfinite(max_step) else METRIC_UNAVAILABLE,
                "timing_gate_pass_rate": (
                    mean([int(_safe_float(r.get("timing_gate_pass"), 0.0) == 1.0) for r in candidate_rows])
                    if candidate_rows else 0.0
                ),
                "selection_pass": pass_value,
                "selection_score": int(hidden_dim) if pass_value else METRIC_UNAVAILABLE,
                "selection_rule": "base_candidate_hidden_pair_with_params<=1.0_flops<=1.0_q90_step<=1.50_across_p4_protocols",
                "test_metric_used_for_selection": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })

    selected = [r for r in candidate_summaries if int(_safe_float(r.get("selection_pass"), 0.0)) == 1]
    selected_row = max(selected, key=lambda r: (int(_safe_float(r.get("hidden_dim"), -1.0)), -_safe_float(r.get("q90_step_ratio_vs_KB_MLP")))) if selected else None
    summary = {
        "stage": "P4_BASE_IMPLEMENTATION_SCREEN_SUMMARY_V87",
        "status": "measured" if rows else "not_run",
        "pilot_type": "robust_base_implementation_system_only_no_test_metric",
        "candidate_count": len(candidate_summaries),
        "protocol_count": len({str(r.get("timing_protocol")) for r in rows}),
        "selected_base_candidate_id": (selected_row or {}).get("candidate_id", METRIC_UNAVAILABLE),
        "selected_hidden_dim": (selected_row or {}).get("hidden_dim", METRIC_UNAVAILABLE),
        "selection_pass": int(selected_row is not None),
        "selected_step_ratio_vs_KB_MLP": (selected_row or {}).get("q90_step_ratio_vs_KB_MLP", METRIC_UNAVAILABLE),
        "selected_params_ratio_vs_KB_MLP": (selected_row or {}).get("params_ratio_vs_KB_MLP", METRIC_UNAVAILABLE),
        "selected_flops_ratio_vs_KB_MLP": (selected_row or {}).get("flops_ratio_vs_KB_MLP", METRIC_UNAVAILABLE),
        "selection_rule": "max_hidden_then_fastest_base_candidate_with_params<=1.0_flops<=1.0_q90_step<=1.50_across_p4_protocols",
        "test_metric_used_for_selection": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_csv(out_dir / "base_implementation_screen.csv", rows)
    write_csv(out_dir / "base_implementation_screen_by_candidate.csv", candidate_summaries)
    write_csv(out_dir / "base_implementation_screen_summary.csv", [summary])
    return rows, summary


def _run_selected_system_task_geometry_confirmation(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    global _V85_FUSED_USE_COMPILED_CORE
    candidate_id = str(args.selected_dg_candidate_id)
    hidden_dim = int(args.selected_dg_hidden_dim)
    child_out_dir = out_dir / "p4_selected_v85_confirmation" / f"{candidate_id}_hidden{hidden_dim}"
    ensure_dir(child_out_dir)
    child_args = _v85_namespace(
        child_out_dir=child_out_dir,
        args=args,
        seed=int(args.selected_seed),
        fresh=True,
    )
    child_args.dg_candidate_id = candidate_id
    child_args.dg_hidden_dim = hidden_dim
    child_args.run_primary_transfer = True
    child_args.run_functional_causality = True
    child_args.run_symbolic = False
    child_args.run_continual = False
    original_fast_adamw = v85.v83.v72.FastAdamWNoSync
    original_pretrain_hook = getattr(v85, "DG_PRETRAIN_HOOK", None)
    original_v85_fused_use_compiled = _V85_FUSED_USE_COMPILED_CORE
    try:
        if bool(args.selected_use_foreach_adamw_addcdiv):
            v85.v83.v72.FastAdamWNoSync = _ForeachAdamWAddcdivNoSync
        original_ce_backward = v85._manual_ce_backward_v85
        original_ce_backward_holdout = v85._manual_ce_backward_with_holdout_v85
        if bool(getattr(args, "selected_use_fused_ce_head_backward", False)):
            _V85_FUSED_USE_COMPILED_CORE = bool(getattr(args, "selected_use_compiled_fused_ce_head_backward", False))
            v85._manual_ce_backward_v85 = _manual_ce_backward_v85_fused
            v85._manual_ce_backward_with_holdout_v85 = _manual_ce_backward_with_holdout_v85_fused
            if bool(
                getattr(args, "selected_use_compiled_fused_ce_head_backward", False)
                and getattr(args, "selected_prewarm_compiled_fused_ce_head_backward", False)
            ):
                v85.DG_PRETRAIN_HOOK = _make_v85_compiled_fused_prewarm_hook()
        decision = v85.run(child_args)
    finally:
        v85.v83.v72.FastAdamWNoSync = original_fast_adamw
        v85.DG_PRETRAIN_HOOK = original_pretrain_hook
        _V85_FUSED_USE_COMPILED_CORE = original_v85_fused_use_compiled
        if "original_ce_backward" in locals():
            v85._manual_ce_backward_v85 = original_ce_backward
            v85._manual_ce_backward_with_holdout_v85 = original_ce_backward_holdout

    primary = _primary_by_candidate(_read_csv(child_out_dir / "kanbefair_primary_transfer.csv"))
    causality = _causality_by_name(_read_csv(child_out_dir / "functional_causality_kanbefair.csv"))
    kb = primary.get("KB-MLP", {})
    dg_base = primary.get(f"DG0-{candidate_id}-hidden{hidden_dim}-base", {})
    dg_func = primary.get(f"DG1-FT7-{candidate_id}-hidden{hidden_dim}-functional", {})
    c_func = causality.get("DG-Functional", {})
    kb_acc = _safe_float(kb.get("test_metric"))
    kb_step = _safe_float(kb.get("step_time_ms"))
    dg_base_acc = _safe_float(dg_base.get("test_metric"))
    dg_func_acc = _safe_float(dg_func.get("test_metric"))
    dg_func_step = _safe_float(dg_func.get("step_time_ms"))
    curv_ratio = _safe_float(dg_func.get("functional_curvature_ratio_vs_dg_base"), _safe_float(c_func.get("curvature_ratio")))
    step_ratio = dg_func_step / kb_step if math.isfinite(dg_func_step) and math.isfinite(kb_step) and kb_step > 0.0 else float("nan")
    audit_rows = _read_csv(child_out_dir / "v85_provenance_audit.csv")
    child_audit = audit_rows[0] if audit_rows else {}
    pass_value = int(
        int(_safe_float(decision.get("success_v85_external_formal"), 0.0)) == 1
        and math.isfinite(step_ratio)
        and step_ratio <= 1.50
        and math.isfinite(curv_ratio)
        and curv_ratio <= 0.90
        and math.isfinite(dg_func_acc)
        and math.isfinite(kb_acc)
        and dg_func_acc >= kb_acc
    )
    summary = {
        "stage": "P4_SELECTED_SYSTEM_TASK_GEOMETRY_CONFIRMATION_V87",
        "status": "measured",
        "child_out_dir": str(child_out_dir),
        "child_route": decision.get("route", METRIC_UNAVAILABLE),
        "candidate_id": candidate_id,
        "hidden_dim": hidden_dim,
        "optimizer_impl": "ForeachAdamWAddcdivNoSync" if bool(args.selected_use_foreach_adamw_addcdiv) else "FastAdamWNoSync",
        "ce_head_backward_impl": (
            "compiled_fused_value_equivalent"
            if bool(
                getattr(args, "selected_use_fused_ce_head_backward", False)
                and getattr(args, "selected_use_compiled_fused_ce_head_backward", False)
            )
            else ("fused_value_equivalent" if bool(getattr(args, "selected_use_fused_ce_head_backward", False)) else "v85_default")
        ),
        "compiled_fused_ce_head_prewarm_used": int(
            bool(
                getattr(args, "selected_use_fused_ce_head_backward", False)
                and getattr(args, "selected_use_compiled_fused_ce_head_backward", False)
                and getattr(args, "selected_prewarm_compiled_fused_ce_head_backward", False)
            )
        ),
        "seed": int(args.selected_seed),
        "KB_MLP_test_acc": kb_acc if math.isfinite(kb_acc) else METRIC_UNAVAILABLE,
        "DG_base_test_acc": dg_base_acc if math.isfinite(dg_base_acc) else METRIC_UNAVAILABLE,
        "DG_functional_test_acc": dg_func_acc if math.isfinite(dg_func_acc) else METRIC_UNAVAILABLE,
        "DG_functional_delta_vs_KB_MLP": dg_func_acc - kb_acc if math.isfinite(dg_func_acc) and math.isfinite(kb_acc) else METRIC_UNAVAILABLE,
        "DG_functional_delta_vs_DG_base": dg_func_acc - dg_base_acc if math.isfinite(dg_func_acc) and math.isfinite(dg_base_acc) else METRIC_UNAVAILABLE,
        "curvature_ratio_vs_DG_base": curv_ratio if math.isfinite(curv_ratio) else METRIC_UNAVAILABLE,
        "functional_events": dg_func.get("functional_event_count", c_func.get("functional_event_count", METRIC_UNAVAILABLE)),
        "step_ratio_vs_KB_MLP": step_ratio if math.isfinite(step_ratio) else METRIC_UNAVAILABLE,
        "primary_transfer_pass": decision.get("primary_transfer_pass", 0),
        "parameter_fair_pass": decision.get("parameter_fair_pass", 0),
        "flops_fair_pass": decision.get("flops_fair_pass", 0),
        "wallclock_fair_pass": decision.get("wallclock_fair_pass", 0),
        "functional_causality_pass": decision.get("functional_causality_pass", 0),
        "selected_task_geometry_confirmation_pass": pass_value,
        "test_metric_used_for_selection": 0,
        "loss_type": "CE",
        "geometry_loss_used": 0,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "sampler_changed": 0,
        "class_weight_used": 0,
        "cpu_offload_used": decision.get("cpu_offload_used", 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "child_rows_checked": child_audit.get("rows_checked", METRIC_UNAVAILABLE),
        "child_fake_proxy_nonzero_count": child_audit.get("fake_proxy_nonzero_count", METRIC_UNAVAILABLE),
    }
    write_csv(out_dir / "selected_system_task_geometry_confirmation.csv", [summary])
    save_json(out_dir / "selected_system_child_route_decision.json", decision)
    return summary


def _run_p5_p6_selected_timing_compute(
    args: argparse.Namespace,
    out_dir: Path,
    p4_summary: Dict[str, Any] | None,
    selected_summary: Dict[str, Any] | None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Materialize P5/P6 from measured P4 rows and selected confirmation.

    This is intentionally conservative: phase fields not measured by the P4
    screen stay metric_unavailable, and P6 backward/update costs are explicitly
    labelled analytic estimates rather than profiler measurements.
    """
    p4_summary = p4_summary or {}
    selected_summary = selected_summary or {}
    selected_candidate = str(p4_summary.get("selected_base_candidate_id", selected_summary.get("candidate_id", ""))).strip()
    selected_hidden = int(_safe_float(p4_summary.get("selected_hidden_dim", selected_summary.get("hidden_dim", -1))))
    p4_rows = _read_csv(out_dir / "base_implementation_screen.csv")
    selected_p4_rows = [
        r for r in p4_rows
        if str(r.get("candidate_id")) == selected_candidate
        and int(_safe_float(r.get("hidden_dim"), -1.0)) == selected_hidden
        and str(r.get("status")) == "measured"
    ]
    p5_rows: List[Dict[str, Any]] = []
    for row in selected_p4_rows:
        base_unknown_fraction = _safe_float(row.get("unknown_time_fraction"))
        p5_rows.append({
            "stage": "P5_ROBUST_TIMING_PROTOCOLS_V87",
            "status": "measured_from_p4_base_implementation_screen",
            "timing_protocol": row.get("timing_protocol", METRIC_UNAVAILABLE),
            "protocol_scope": row.get("protocol_scope", METRIC_UNAVAILABLE),
            "candidate": f"DG0-{selected_candidate}-hidden{selected_hidden}-base",
            "candidate_id": selected_candidate,
            "hidden_dim": selected_hidden,
            "step_time_ms": row.get("step_time_ms", METRIC_UNAVAILABLE),
            "step_ratio_vs_KB_MLP": row.get("step_ratio_vs_KB_MLP", METRIC_UNAVAILABLE),
            "forward_time_ms": row.get("forward_time_ms", METRIC_UNAVAILABLE),
            "backward_time_ms": row.get("backward_time_ms", METRIC_UNAVAILABLE),
            "base_update_time_ms": row.get("base_update_time_ms", METRIC_UNAVAILABLE),
            "functional_update_time_ms": 0.0,
            "guard_time_ms": 0.0,
            "validation_time_ms": METRIC_UNAVAILABLE,
            "cuda_sync_time_ms": METRIC_UNAVAILABLE,
            "unknown_time_fraction": row.get("unknown_time_fraction", METRIC_UNAVAILABLE),
            "kernel_time_ms": METRIC_UNAVAILABLE,
            "timing_gate_pass": int(_safe_float(row.get("step_ratio_vs_KB_MLP")) <= 1.50),
            "time_accounting_complete": int(math.isfinite(base_unknown_fraction) and base_unknown_fraction <= 0.10),
            "reason": "phase_clean_base_timing_measured_selected_by_p4_no_test_metric",
            "loss_type": "CE",
            "geometry_loss_used": 0,
            "external_teacher_used": 0,
            "self_teacher_used": 0,
            "sampler_changed": 0,
            "class_weight_used": 0,
            "cpu_offload_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    selected_step_ratio = _safe_float(selected_summary.get("step_ratio_vs_KB_MLP"))
    if selected_summary:
        try:
            phase_row = _train_step_timing_row(
                args,
                out_dir,
                timing_protocol="P5-selected-functional-phase-clean",
                protocol_scope="KANbeFair_MNIST_selected_functional_phase_accounting",
                source_row_kind="p5_selected_functional_phase_accounting",
                seed=int(getattr(args, "selected_seed", getattr(args, "t3_seed", 1314))),
                warmup=int(getattr(args, "t3_warmup", 20)),
                reps=int(getattr(args, "t3_reps", 260)),
                batch_size=int(getattr(args, "t3_batch_size", 128)),
                hidden_dim=selected_hidden,
                dg_candidate=selected_candidate,
            )
            measured_unknown = _safe_float(phase_row.get("unknown_time_fraction"))
            measured_step_ratio = _safe_float(phase_row.get("step_ratio_vs_KB_MLP"))
            phase_row.update({
                "status": "measured_selected_functional_phase_breakdown",
                "candidate": f"DG1-AdaptiveSelected-{selected_candidate}-hidden{selected_hidden}-functional",
                "functional_candidate_id": phase_row.get("candidate_id", METRIC_UNAVAILABLE),
                "candidate_id": selected_candidate,
                "hidden_dim": selected_hidden,
                "timing_gate_pass": int(math.isfinite(measured_step_ratio) and measured_step_ratio <= 1.50),
                "time_accounting_complete": int(math.isfinite(measured_unknown) and measured_unknown <= 0.10),
                "reason": "selected_functional_phase_breakdown_measured",
            })
            p5_rows.append(phase_row)
        except Exception as exc:
            p5_rows.append({
                "stage": "P5_ROBUST_TIMING_PROTOCOLS_V87",
                "status": "phase_breakdown_measurement_failed",
                "timing_protocol": "P5-selected-functional-phase-clean",
                "protocol_scope": "KANbeFair_MNIST_selected_functional_phase_accounting",
                "candidate": f"DG1-AdaptiveSelected-{selected_candidate}-hidden{selected_hidden}-functional",
                "candidate_id": selected_candidate,
                "hidden_dim": selected_hidden,
                "step_time_ms": METRIC_UNAVAILABLE,
                "step_ratio_vs_KB_MLP": selected_step_ratio if math.isfinite(selected_step_ratio) else METRIC_UNAVAILABLE,
                "forward_time_ms": METRIC_UNAVAILABLE,
                "backward_time_ms": METRIC_UNAVAILABLE,
                "base_update_time_ms": METRIC_UNAVAILABLE,
                "functional_update_time_ms": METRIC_UNAVAILABLE,
                "guard_time_ms": METRIC_UNAVAILABLE,
                "validation_time_ms": METRIC_UNAVAILABLE,
                "cuda_sync_time_ms": METRIC_UNAVAILABLE,
                "unknown_time_fraction": METRIC_UNAVAILABLE,
                "kernel_time_ms": METRIC_UNAVAILABLE,
                "timing_gate_pass": int(math.isfinite(selected_step_ratio) and selected_step_ratio <= 1.50),
                "time_accounting_complete": 0,
                "reason": f"selected_functional_phase_breakdown_measurement_failed:{type(exc).__name__}",
                "loss_type": "CE",
                "geometry_loss_used": 0,
                "external_teacher_used": 0,
                "self_teacher_used": 0,
                "sampler_changed": 0,
                "class_weight_used": 0,
                "cpu_offload_used": selected_summary.get("cpu_offload_used", 0),
                "fake_data_used": 0,
                "proxy_row_used": 0,
            })
    p5_step_values = [_safe_float(r.get("step_ratio_vs_KB_MLP")) for r in p5_rows]
    p5_finite = [v for v in p5_step_values if math.isfinite(v)]
    q90_step = _q90(p5_finite)
    max_step = max(p5_finite) if p5_finite else float("nan")
    unknown_values = [_safe_float(r.get("unknown_time_fraction")) for r in p5_rows]
    finite_unknown = [v for v in unknown_values if math.isfinite(v)]
    max_unknown = max(finite_unknown) if finite_unknown else float("nan")
    all_unknown_measured = bool(p5_rows) and len(finite_unknown) == len(p5_rows)
    accounting_pass = int(all_unknown_measured and max_unknown <= 0.10)
    p5_summary = {
        "stage": "P5_ROBUST_TIMING_PROTOCOLS_SUMMARY_V87",
        "status": "measured" if p5_rows else "not_run",
        "candidate_id": selected_candidate if selected_candidate else METRIC_UNAVAILABLE,
        "hidden_dim": selected_hidden if selected_hidden >= 0 else METRIC_UNAVAILABLE,
        "row_count": len(p5_rows),
        "q90_step_ratio_vs_KB_MLP": q90_step if math.isfinite(q90_step) else METRIC_UNAVAILABLE,
        "max_step_ratio_vs_KB_MLP": max_step if math.isfinite(max_step) else METRIC_UNAVAILABLE,
        "max_unknown_time_fraction": max_unknown if math.isfinite(max_unknown) else METRIC_UNAVAILABLE,
        "robust_timing_pass": int(math.isfinite(q90_step) and q90_step <= 1.50),
        "strict_timing_pass": int(math.isfinite(max_step) and max_step <= 1.50),
        "time_accounting_pass": accounting_pass,
        "time_accounting_reason": (
            "phase_breakdown_measured_for_selected_base_and_functional_rows"
            if accounting_pass
            else "phase_breakdown_not_fully_measured_for_selected_functional_confirmation"
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_csv(out_dir / "robust_timing_protocols.csv", p5_rows)
    write_csv(out_dir / "robust_timing_protocols_summary.csv", [p5_summary])

    forward_ratio = _safe_float(p4_summary.get("selected_flops_ratio_vs_KB_MLP"))
    params_ratio = _safe_float(p4_summary.get("selected_params_ratio_vs_KB_MLP"))
    step_ratio = selected_step_ratio if math.isfinite(selected_step_ratio) else _safe_float(p4_summary.get("selected_step_ratio_vs_KB_MLP"))
    backward_ratio_est = forward_ratio
    update_ratio_est = params_ratio
    fair_subgates = [
        int(math.isfinite(forward_ratio) and forward_ratio <= 1.05),
        int(math.isfinite(backward_ratio_est) and backward_ratio_est <= 1.50),
        int(math.isfinite(step_ratio) and step_ratio <= 1.50),
    ]
    p6_row = {
        "stage": "P6_TRAINING_COMPUTE_COUNTER_V87",
        "status": "measured_analytic_estimate_from_p4_selected_artifacts",
        "candidate": f"DG1-AdaptiveSelected-{selected_candidate}-hidden{selected_hidden}-functional",
        "candidate_id": selected_candidate if selected_candidate else METRIC_UNAVAILABLE,
        "hidden_dim": selected_hidden if selected_hidden >= 0 else METRIC_UNAVAILABLE,
        "forward_FLOPs_ratio_vs_KB_MLP": forward_ratio if math.isfinite(forward_ratio) else METRIC_UNAVAILABLE,
        "backward_FLOPs_estimate_ratio_vs_KB_MLP": backward_ratio_est if math.isfinite(backward_ratio_est) else METRIC_UNAVAILABLE,
        "update_FLOPs_estimate_ratio_vs_KB_MLP": update_ratio_est if math.isfinite(update_ratio_est) else METRIC_UNAVAILABLE,
        "functional_update_FLOPs_estimate_ratio_vs_KB_MLP": METRIC_UNAVAILABLE,
        "kernel_time_forward_ratio_vs_KB_MLP": METRIC_UNAVAILABLE,
        "kernel_time_backward_ratio_vs_KB_MLP": METRIC_UNAVAILABLE,
        "kernel_time_update_ratio_vs_KB_MLP": METRIC_UNAVAILABLE,
        "kernel_time_functional_ratio_vs_KB_MLP": METRIC_UNAVAILABLE,
        "step_time_ratio_vs_KB_MLP": step_ratio if math.isfinite(step_ratio) else METRIC_UNAVAILABLE,
        "train_step_energy_proxy_if_available": METRIC_UNAVAILABLE,
        "estimate_scope": "forward_FLOPs_from_P4_counter_backward_estimate_same_ratio_update_estimate_from_param_ratio",
        "fair_subgate_count": sum(fair_subgates),
        "training_compute_fair_pass": int(sum(fair_subgates) >= 2),
        "kernel_time_measured": 0,
        "loss_type": "CE",
        "geometry_loss_used": 0,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "sampler_changed": 0,
        "class_weight_used": 0,
        "cpu_offload_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }
    write_csv(out_dir / "training_compute_counter.csv", [p6_row])
    return p5_summary, p6_row


def _canonical_vision_task_name_v87(dataset: str) -> str:
    name = str(dataset).strip()
    low = name.lower()
    if low in {"fashion-mnist", "fashion", "fmnist"}:
        return "FMNIST"
    if low == "kmnist":
        return "KMNIST"
    if low == "mnist":
        return "MNIST"
    return name


def _run_p7_p9_multitask_external(
    args: argparse.Namespace,
    out_dir: Path,
    p4_summary: Dict[str, Any] | None,
    selected_summary: Dict[str, Any] | None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run selected architecture on broad KANbeFair vision tasks.

    This is intentionally narrower than v8.6's broad runner: it uses the
    no-manual v8.7 selected candidate/hidden instead of falling back to the
    older KW6 fixed recipe.
    """

    selected_candidate = str((p4_summary or {}).get("selected_base_candidate_id", "")).strip()
    selected_hidden_value = _safe_float((p4_summary or {}).get("selected_hidden_dim"))
    if not selected_candidate or selected_candidate == METRIC_UNAVAILABLE:
        selected_candidate = str((selected_summary or {}).get("candidate_id", args.selected_dg_candidate_id)).strip()
    selected_hidden = int(selected_hidden_value) if math.isfinite(selected_hidden_value) else int(
        _safe_float((selected_summary or {}).get("hidden_dim"), float(args.selected_dg_hidden_dim))
    )

    source_args = _v85_namespace(
        child_out_dir=out_dir / "p7_source_audit",
        args=args,
        seed=int(args.selected_seed),
        fresh=False,
    )
    source_rows, model_classes = v85._run_source_audit(out_dir, source_args)
    MLP = model_classes.get("MLP")
    KANbeFair = model_classes.get("KANbeFair")
    if MLP is None or KANbeFair is None:
        row = {
            "stage": "P7_KANBEFAIR_MULTITASK_TRANSFER_V87",
            "status": "not_run",
            "reason": "kanbefair_model_import_failed",
            "selected_candidate_id": selected_candidate or METRIC_UNAVAILABLE,
            "selected_hidden_dim": selected_hidden if selected_hidden >= 0 else METRIC_UNAVAILABLE,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        write_csv(out_dir / "kanbefair_multitask_transfer.csv", [row])
        write_csv(out_dir / "joint_fair_envelope.csv", [dict(row, stage="P8_JOINT_FAIR_ENVELOPE_V87")])
        write_csv(out_dir / "functional_causality_multitask.csv", [dict(row, stage="P9_FUNCTIONAL_CAUSALITY_MULTITASK_V87")])
        summary = {
            "stage": "P7_KANBEFAIR_MULTITASK_TRANSFER_SUMMARY_V87",
            "status": "not_run",
            "reason": "kanbefair_model_import_failed",
            "p7_multitask_measured": 0,
            "p7_joint_fair_pass_count": 0,
            "p9_causality_pass_count": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        write_csv(out_dir / "kanbefair_multitask_transfer_summary.csv", [summary])
        write_csv(out_dir / "functional_causality_multitask_summary.csv", [summary])
        return summary, summary

    device = v85.v83.get_device(args.device)
    formal_protocol = int(
        int(args.p7_train_size) >= 60000
        and int(args.p7_test_size) >= 10000
        and int(args.p7_epochs) == 20
        and int(args.kb_batch_size) == 128
    )
    transfer_rows: List[Dict[str, Any]] = []
    fair_rows: List[Dict[str, Any]] = []
    causality_rows: List[Dict[str, Any]] = []
    reproduction_rows: List[Dict[str, Any]] = []

    original_fast_adamw = v85.v83.v72.FastAdamWNoSync
    original_pretrain_hook = getattr(v85, "DG_PRETRAIN_HOOK", None)
    original_v85_fused_use_compiled = _V85_FUSED_USE_COMPILED_CORE
    original_ce_backward = v85._manual_ce_backward_v85
    original_ce_backward_holdout = v85._manual_ce_backward_with_holdout_v85
    original_stride = int(v85.v83.P5_FT7_EVENT_STRIDE)
    original_alpha = float(v85.v83.P5_FT7_EVENT_ALPHA_MULT)

    try:
        if bool(args.selected_use_foreach_adamw_addcdiv):
            v85.v83.v72.FastAdamWNoSync = _ForeachAdamWAddcdivNoSync
        if bool(getattr(args, "selected_use_fused_ce_head_backward", False)):
            globals()["_V85_FUSED_USE_COMPILED_CORE"] = bool(getattr(args, "selected_use_compiled_fused_ce_head_backward", False))
            v85._manual_ce_backward_v85 = _manual_ce_backward_v85_fused
            v85._manual_ce_backward_with_holdout_v85 = _manual_ce_backward_with_holdout_v85_fused
            if bool(
                getattr(args, "selected_use_compiled_fused_ce_head_backward", False)
                and getattr(args, "selected_prewarm_compiled_fused_ce_head_backward", False)
            ):
                v85.DG_PRETRAIN_HOOK = _make_v85_compiled_fused_prewarm_hook()
        v85.v83.P5_FT7_EVENT_STRIDE = int(args.p7_ft7_event_stride)
        v85.v83.P5_FT7_EVENT_ALPHA_MULT = float(args.p7_ft7_event_alpha_mult)

        for dataset in v85.parse_str_list(args.p7_datasets):
            task_name = _canonical_vision_task_name_v87(dataset)
            try:
                x_train, y_train, x_test, y_test, input_dim, num_classes, split_protocol = v85._load_kanbefair_vision_tensors(
                    task_name,
                    data_root=Path(args.data_root),
                    train_size=int(args.p7_train_size),
                    test_size=int(args.p7_test_size),
                    seed=int(args.selected_seed),
                )
            except Exception as exc:
                not_run = {
                    "stage": "P7_KANBEFAIR_MULTITASK_TRANSFER_V87",
                    "status": "not_run",
                    "task_family": "vision",
                    "task_name": task_name,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "selected_candidate_id": selected_candidate,
                    "selected_hidden_dim": selected_hidden,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
                transfer_rows.append(not_run)
                fair_rows.append(dict(not_run, stage="P8_JOINT_FAIR_ENVELOPE_V87"))
                causality_rows.append(dict(not_run, stage="P9_FUNCTIONAL_CAUSALITY_MULTITASK_V87"))
                continue

            kb_metrics: Dict[str, Dict[str, Any]] = {}
            for cfg in v85._baseline_configs():
                model_name = str(cfg["model"])
                cls = MLP if model_name == "MLP" else KANbeFair
                ns = v85._kb_namespace(
                    model_name=model_name,
                    input_size=input_dim,
                    output_size=num_classes,
                    layers_width=cfg["layers_width"],
                    activation_name=cfg["activation_name"],
                    batch_norm=cfg["batch_norm"],
                    kan_grid=cfg.get("kan_grid", 3),
                    kan_order=cfg.get("kan_order", 2),
                    kan_shortcut=cfg.get("kan_shortcut", "silu"),
                    kan_range=cfg.get("kan_range", [-1.0, 1.0]),
                )
                v85.set_seed(int(cfg["seed"]))
                model = cls(ns)
                params = int(model.total_parameters())
                flops = float(model.total_flops())
                metrics = v85._train_kb_classifier(
                    model,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    epochs=int(args.p7_epochs),
                    batch_size=int(args.kb_batch_size),
                    lr=float(cfg["lr"]),
                    seed=int(cfg["seed"]),
                    device=device,
                )
                reported = v85._reported_result(
                    Path(args.kanbefair_path),
                    dataset=task_name,
                    model=model_name,
                    layers_width=cfg["layers_width"],
                    batch_size=int(cfg["reported_batch_size"]),
                    epochs=int(cfg["reported_epochs"]),
                    lr=float(cfg["lr"]),
                    seed=int(cfg["seed"]),
                    activation_name=cfg["activation_name"],
                    kan_grid=cfg.get("kan_grid", 3),
                    kan_order=cfg.get("kan_order", 2),
                    kan_shortcut=cfg.get("kan_shortcut", "silu"),
                    kan_range=cfg.get("kan_range", [-1.0, 1.0]),
                )
                reported_test = _safe_float(reported.get("reported_test_metric"))
                abs_delta = abs(float(metrics["test_acc_pct"]) - reported_test) if math.isfinite(reported_test) else float("nan")
                model_id = f"KB-{model_name}"
                kb_metrics[model_id] = {
                    "metric": float(metrics["test_acc_pct"]),
                    "loss": float(metrics["test_loss"]),
                    "ECE": float(metrics["ECE"]),
                    "NLL": float(metrics["NLL"]),
                    "params": params,
                    "FLOPs": flops,
                    "step_time_ms": float(metrics["step_time_ms"]),
                    "train_time_s": float(metrics["train_time_s"]),
                    "peak_memory_MB": float(metrics["peak_memory_MB"]),
                }
                reproduction_rows.append({
                    "stage": "P7_KANBEFAIR_BROAD_REPRODUCTION_V87",
                    "status": "measured",
                    "task_family": "vision",
                    "task_name": task_name,
                    "model_name": model_id,
                    "reported_metric": reported_test if math.isfinite(reported_test) else METRIC_UNAVAILABLE,
                    "measured_metric": metrics["test_acc_pct"],
                    "absolute_delta_from_reported": abs_delta if math.isfinite(abs_delta) else METRIC_UNAVAILABLE,
                    "params": params,
                    "FLOPs": flops,
                    "step_time_ms": metrics["step_time_ms"],
                    "peak_memory_MB": metrics["peak_memory_MB"],
                    "train_time_s": metrics["train_time_s"],
                    "epochs": int(args.p7_epochs),
                    "batch_size": int(args.kb_batch_size),
                    "split_protocol": split_protocol,
                    "formal_protocol": formal_protocol,
                    "reproduction_pass": int(formal_protocol and math.isfinite(abs_delta) and abs_delta <= 2.0),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
                transfer_rows.append({
                    "stage": "P7_KANBEFAIR_MULTITASK_TRANSFER_V87",
                    "status": "measured",
                    "task_family": "vision",
                    "task_name": task_name,
                    "model": model_id,
                    "seed": int(cfg["seed"]),
                    "params": params,
                    "FLOPs": flops,
                    "train_time_s": metrics["train_time_s"],
                    "step_time_ms": metrics["step_time_ms"],
                    "peak_memory_MB": metrics["peak_memory_MB"],
                    "test_metric": metrics["test_acc_pct"],
                    "delta_vs_KB_MLP": 0.0 if model_id == "KB-MLP" else (
                        float(metrics["test_acc_pct"]) - float(kb_metrics.get("KB-MLP", {}).get("metric", float("nan")))
                    ),
                    "delta_vs_DG_Base": METRIC_UNAVAILABLE,
                    "curvature": METRIC_UNAVAILABLE,
                    "jacobian_norm": METRIC_UNAVAILABLE,
                    "local_lipschitz": METRIC_UNAVAILABLE,
                    "functional_events": 0,
                    "TaskFamilyPass": 0,
                    "formal_protocol": formal_protocol,
                    "loss_type": "CE",
                    "geometry_loss_used": 0,
                    "external_teacher_used": 0,
                    "self_teacher_used": 0,
                    "sampler_changed": 0,
                    "class_weight_used": 0,
                    "cpu_offload_used": 0,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                })
                if torch.cuda.is_available() and device.type == "cuda":
                    try:
                        model.to("cpu")
                    except Exception:
                        pass
                del model
                _release_cuda_memory_v87(device)

            if "KB-MLP" not in kb_metrics:
                continue
            mlp_metric = float(kb_metrics["KB-MLP"]["metric"])
            mlp_params = max(float(kb_metrics["KB-MLP"]["params"]), 1.0e-12)
            mlp_flops = max(float(kb_metrics["KB-MLP"]["FLOPs"]), 1.0e-12)
            mlp_step = max(float(kb_metrics["KB-MLP"]["step_time_ms"]), 1.0e-12)
            mlp_memory = max(float(kb_metrics["KB-MLP"]["peak_memory_MB"]), 1.0e-12)

            dg_args = _v85_namespace(
                child_out_dir=out_dir / "p7_dg_child_args",
                args=args,
                seed=int(args.selected_seed),
                fresh=False,
            )
            dg_args.primary_epochs = int(args.p7_epochs)
            dg_args.primary_train_size = int(args.p7_train_size)
            dg_args.primary_test_size = int(args.p7_test_size)
            dg_args.dg_candidate_id = selected_candidate
            dg_args.dg_hidden_dim = selected_hidden
            dg_args.dg_batch_size = int(args.kb_batch_size)
            dg_args.dg_eval_batch_size = int(args.p7_eval_batch_size)
            dg_args.ft7_event_stride = int(args.p7_ft7_event_stride)
            dg_args.ft7_event_alpha_mult = float(args.p7_ft7_event_alpha_mult)
            dg_args.dg_stream_batches_from_cpu = bool(args.p7_dg_stream_batches_from_cpu)
            dg_args.dg_stream_chunk_batches = int(args.p7_dg_stream_chunk_batches)
            dg_args.dg_stream_epoch_permute_cpu = bool(args.p7_dg_stream_epoch_permute_cpu)
            dg_args.dg_stream_chunk_order_shuffle = bool(args.p7_dg_stream_chunk_order_shuffle)

            def train_dg(functional: int, variant: str | None = None) -> Dict[str, Any]:
                _release_cuda_memory_v87(device)
                row = v85._train_dg_primary_classifier(
                    dg_args,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    dataset=task_name,
                    input_dim=input_dim,
                    num_classes=num_classes,
                    functional_update_used=functional,
                    functional_variant=variant,
                )
                _release_cuda_memory_v87(device)
                return row

            dg_base = train_dg(0)
            dg_noop = train_dg(0, "noop")
            dg_func = train_dg(1, "ft7")
            dg_random = train_dg(1, "random")
            dg_shuffled = train_dg(1, "shuffled")
            dg_rows = [dg_base, dg_noop, dg_func, dg_random, dg_shuffled]
            base_metric = _safe_float(dg_base.get("test_metric"))
            base_curv = _safe_float(dg_base.get("geometry_curvature_after"))
            base_jac = _safe_float(dg_base.get("jacobian_norm_probe"))
            noop_curv = _safe_float(dg_noop.get("geometry_curvature_after"))
            random_curv = _safe_float(dg_random.get("geometry_curvature_after"))
            noop_metric = _safe_float(dg_noop.get("test_metric"))
            for row in dg_rows:
                metric = _safe_float(row.get("test_metric"))
                curv = _safe_float(row.get("geometry_curvature_after"))
                jac = _safe_float(row.get("jacobian_norm_probe"))
                variant = str(row.get("functional_variant"))
                is_func = variant == "ft7"
                curv_ratio = curv / max(base_curv, 1.0e-12) if math.isfinite(curv) and math.isfinite(base_curv) else float("nan")
                jac_ratio = jac / max(base_jac, 1.0e-12) if math.isfinite(jac) and math.isfinite(base_jac) else float("nan")
                transfer_rows.append({
                    "stage": "P7_KANBEFAIR_MULTITASK_TRANSFER_V87",
                    "status": "measured",
                    "task_family": "vision",
                    "task_name": task_name,
                    "model": row.get("candidate_id"),
                    "seed": int(args.selected_seed),
                    "params": row.get("params"),
                    "FLOPs": row.get("FLOPs"),
                    "train_time_s": row.get("train_time_s"),
                    "step_time_ms": row.get("step_time_ms"),
                    "peak_memory_MB": row.get("peak_memory_MB"),
                    "test_metric": metric if math.isfinite(metric) else METRIC_UNAVAILABLE,
                    "delta_vs_KB_MLP": metric - mlp_metric if math.isfinite(metric) else METRIC_UNAVAILABLE,
                    "delta_vs_DG_Base": metric - base_metric if math.isfinite(metric) and math.isfinite(base_metric) else METRIC_UNAVAILABLE,
                    "curvature": curv if math.isfinite(curv) else METRIC_UNAVAILABLE,
                    "curvature_ratio_vs_DG_Base": curv_ratio if math.isfinite(curv_ratio) else METRIC_UNAVAILABLE,
                    "jacobian_norm": jac if math.isfinite(jac) else METRIC_UNAVAILABLE,
                    "local_lipschitz": row.get("local_lipschitz_probe"),
                    "functional_events": row.get("functional_event_count"),
                    "functional_variant": variant,
                    "TaskFamilyPass": int(is_func and math.isfinite(metric) and metric >= mlp_metric),
                    "formal_protocol": formal_protocol,
                    "loss_type": "CE",
                    "geometry_loss_used": 0,
                    "external_teacher_used": 0,
                    "self_teacher_used": 0,
                    "sampler_changed": 0,
                    "class_weight_used": 0,
                    "cpu_offload_used": 0,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                })
                if is_func:
                    param_ratio = _safe_float(row.get("params")) / mlp_params
                    flops_ratio = _safe_float(row.get("FLOPs")) / mlp_flops
                    step_ratio = _safe_float(row.get("step_time_ms")) / mlp_step
                    memory_ratio = _safe_float(row.get("peak_memory_MB")) / mlp_memory
                    metric_delta = metric - mlp_metric if math.isfinite(metric) else float("nan")
                    joint_pass = int(
                        math.isfinite(metric_delta)
                        and metric_delta >= 0.0
                        and math.isfinite(param_ratio) and param_ratio <= 1.05
                        and math.isfinite(flops_ratio) and flops_ratio <= 1.05
                        and math.isfinite(step_ratio) and step_ratio <= 1.50
                        and math.isfinite(memory_ratio) and memory_ratio <= 1.05
                    )
                    fair_rows.append({
                        "stage": "P8_JOINT_FAIR_ENVELOPE_V87",
                        "status": "measured",
                        "task_family": "vision",
                        "task_name": task_name,
                        "envelope_type": "v87_selected_vs_KB_MLP_width32",
                        "model": row.get("candidate_id"),
                        "selected_candidate_id": selected_candidate,
                        "selected_hidden_dim": selected_hidden,
                        "params_ratio_vs_MLP": param_ratio,
                        "FLOPs_ratio_vs_MLP": flops_ratio,
                        "step_ratio_vs_MLP": step_ratio,
                        "memory_ratio_vs_MLP": memory_ratio,
                        "metric_delta_vs_MLP": metric_delta if math.isfinite(metric_delta) else METRIC_UNAVAILABLE,
                        "curvature_ratio_vs_DG_Base": curv_ratio if math.isfinite(curv_ratio) else METRIC_UNAVAILABLE,
                        "formal_protocol": formal_protocol,
                        "JointFairPass": joint_pass,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    })
                causality_pass = int(
                    is_func
                    and math.isfinite(curv)
                    and math.isfinite(noop_curv)
                    and math.isfinite(random_curv)
                    and curv < noop_curv
                    and curv < random_curv
                    and metric >= noop_metric - 0.2
                )
                causality_rows.append({
                    "stage": "P9_FUNCTIONAL_CAUSALITY_MULTITASK_V87",
                    "status": "measured",
                    "task_family": "vision",
                    "task_name": task_name,
                    "seed": int(args.selected_seed),
                    "model": row.get("candidate_id"),
                    "functional_variant": variant,
                    "test_metric": metric if math.isfinite(metric) else METRIC_UNAVAILABLE,
                    "delta_vs_base": metric - base_metric if math.isfinite(metric) and math.isfinite(base_metric) else METRIC_UNAVAILABLE,
                    "curvature_ratio": curv_ratio if math.isfinite(curv_ratio) else METRIC_UNAVAILABLE,
                    "jacobian_ratio": jac_ratio if math.isfinite(jac_ratio) else METRIC_UNAVAILABLE,
                    "local_lipschitz_ratio": jac_ratio if math.isfinite(jac_ratio) else METRIC_UNAVAILABLE,
                    "bad_step_rate": row.get("bad_step_rate"),
                    "holdout_descent_ratio": row.get("holdout_descent_ratio"),
                    "role_accept_rate": row.get("role_accept_rate"),
                    "role_geometry_delta": (1.0 - curv_ratio) if math.isfinite(curv_ratio) else METRIC_UNAVAILABLE,
                    "formal_protocol": formal_protocol,
                    "CausalityPass": causality_pass,
                    "loss_type": "CE",
                    "geometry_loss_used": 0,
                    "external_teacher_used": 0,
                    "self_teacher_used": 0,
                    "sampler_changed": 0,
                    "class_weight_used": 0,
                    "cpu_offload_used": 0,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                })
    finally:
        v85.v83.v72.FastAdamWNoSync = original_fast_adamw
        v85.DG_PRETRAIN_HOOK = original_pretrain_hook
        globals()["_V85_FUSED_USE_COMPILED_CORE"] = original_v85_fused_use_compiled
        v85._manual_ce_backward_v85 = original_ce_backward
        v85._manual_ce_backward_with_holdout_v85 = original_ce_backward_holdout
        v85.v83.P5_FT7_EVENT_STRIDE = original_stride
        v85.v83.P5_FT7_EVENT_ALPHA_MULT = original_alpha

    task_names = sorted({str(r.get("task_name")) for r in fair_rows if str(r.get("status")) == "measured"})
    joint_pass_tasks = sorted({str(r.get("task_name")) for r in fair_rows if _safe_float(r.get("JointFairPass"), 0.0) == 1.0})
    causality_pass_tasks = sorted({str(r.get("task_name")) for r in causality_rows if _safe_float(r.get("CausalityPass"), 0.0) == 1.0})
    p7_pass = int(bool(task_names) and len(joint_pass_tasks) == len(task_names))
    p9_pass = int(bool(task_names) and len(causality_pass_tasks) == len(task_names))
    p7_summary = {
        "stage": "P7_KANBEFAIR_MULTITASK_TRANSFER_SUMMARY_V87",
        "status": "measured" if task_names else "not_run",
        "selected_candidate_id": selected_candidate,
        "selected_hidden_dim": selected_hidden,
        "task_count": len(task_names),
        "joint_fair_pass_count": len(joint_pass_tasks),
        "joint_fair_pass_tasks": ",".join(joint_pass_tasks),
        "p7_cross_task_external_pass": p7_pass,
        "formal_protocol": formal_protocol,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    p9_summary = {
        "stage": "P9_FUNCTIONAL_CAUSALITY_MULTITASK_SUMMARY_V87",
        "status": "measured" if task_names else "not_run",
        "selected_candidate_id": selected_candidate,
        "selected_hidden_dim": selected_hidden,
        "task_count": len(task_names),
        "causality_pass_count": len(causality_pass_tasks),
        "causality_pass_tasks": ",".join(causality_pass_tasks),
        "p9_cross_task_causality_pass": p9_pass,
        "formal_protocol": formal_protocol,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_csv(out_dir / "kanbefair_broad_reproduction.csv", reproduction_rows)
    write_csv(out_dir / "kanbefair_multitask_transfer.csv", transfer_rows)
    write_csv(out_dir / "joint_fair_envelope.csv", fair_rows)
    write_csv(out_dir / "functional_causality_multitask.csv", causality_rows)
    write_csv(out_dir / "kanbefair_multitask_transfer_summary.csv", [p7_summary])
    write_csv(out_dir / "functional_causality_multitask_summary.csv", [p9_summary])
    return p7_summary, p9_summary


def _adaptive_event_score(
    *,
    holdout_loss_before: float,
    holdout_task_after: float,
    smooth_before: float,
    curv_before: float,
) -> Tuple[float, float, float, float]:
    holdout_descent = float(holdout_loss_before) - float(holdout_task_after)
    task_descent_score = max(0.0, holdout_descent) / max(abs(float(holdout_loss_before)), 1.0e-12)
    geometry_pressure = float(curv_before) / max(float(curv_before) + float(smooth_before), 1.0e-12)
    bad_step_risk = max(0.0, -holdout_descent) / max(abs(float(holdout_loss_before)), 1.0e-12)
    event_score = task_descent_score + 0.50 * geometry_pressure - bad_step_risk
    return event_score, task_descent_score, geometry_pressure, bad_step_risk


def _adaptive_role_scores(
    entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]],
    func_dirs: Sequence[torch.Tensor],
    task_dirs: Sequence[torch.Tensor],
    *,
    cost_power: float = 0.5,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    roles = sorted({str(role) for role, _name, _param, _grad in entries})
    total_params = max(1.0, float(sum(param.numel() for _role, _name, param, _grad in entries)))
    raw_scores: Dict[str, float] = {}
    cos_by_role: Dict[str, float] = {}
    curv_norm_by_role: Dict[str, float] = {}
    cost_by_role: Dict[str, float] = {}
    for role in roles:
        role_func: List[torch.Tensor] = []
        role_task: List[torch.Tensor] = []
        role_param_count = 0
        for entry, fdir, tdir in zip(entries, func_dirs, task_dirs):
            if str(entry[0]) == role:
                role_func.append(fdir)
                role_task.append(tdir)
                role_param_count += int(entry[2].numel())
        role_cos = v85.v83._cos(role_func, role_task) if role_func else 0.0
        role_curv = v85.v83._norm(role_func) if role_func else 0.0
        role_cost = max(1.0e-12, role_param_count / total_params)
        # A small alignment floor keeps high-curvature roles from disappearing
        # solely because a single batch has near-zero role-local cosine.
        alignment = max(0.05, max(0.0, role_cos))
        raw = alignment * role_curv / (role_cost ** float(cost_power))
        raw_scores[role] = float(raw)
        cos_by_role[role] = float(role_cos)
        curv_norm_by_role[role] = float(role_curv)
        cost_by_role[role] = float(role_cost)
    if sum(raw_scores.values()) <= 1.0e-30:
        raw_scores = {role: 1.0 for role in roles}
    return raw_scores, cos_by_role, curv_norm_by_role, cost_by_role


def _adaptive_role_weights_and_budgets(
    raw_scores: Dict[str, float],
    *,
    total_budget: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    total = max(sum(float(v) for v in raw_scores.values()), 1.0e-30)
    role_count = max(1, len(raw_scores))
    role_weights: Dict[str, float] = {}
    role_budgets: Dict[str, float] = {}
    for role, value in raw_scores.items():
        share = max(0.0, float(value)) / total
        role_weights[role] = min(1.50, max(0.05, share * role_count))
        role_budgets[role] = float(total_budget) * share
    return role_weights, role_budgets


def _project_task_aware_v87(
    task_dirs: Sequence[torch.Tensor],
    func_dirs: Sequence[torch.Tensor],
    *,
    trust_ratio: float,
    role_weights: Dict[str, float] | None,
    entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]],
    max_scale: float,
) -> Tuple[List[torch.Tensor], float, float]:
    task_norm_sq = v85.v83._sum_sq(task_dirs).clamp_min(1.0e-30)
    dot_tf = v85.v83._dot(func_dirs, task_dirs)
    if dot_tf < 0:
        coeff = dot_tf / task_norm_sq
        projected = [f - coeff.to(f.device, f.dtype) * t for f, t in zip(func_dirs, task_dirs)]
    else:
        projected = [f.detach().clone() for f in func_dirs]
    if role_weights:
        projected = [
            p * float(role_weights.get(role, 1.0))
            for p, (role, _name, _param, _grad) in zip(projected, entries)
        ]
    pnorm = v85.v83._norm(projected)
    tnorm = v85.v83._norm(task_dirs)
    scale = 0.0 if pnorm <= 1.0e-30 else min(float(trust_ratio) * tnorm / max(pnorm, 1.0e-30), float(max_scale))
    corrected = [t + scale * p for t, p in zip(task_dirs, projected)]
    clip_rate = float(scale < float(max_scale) and pnorm > 0.0)
    return corrected, float(scale), clip_rate


def _head_transformed_features_v87(head: Any, features: torch.Tensor) -> torch.Tensor:
    layer = getattr(head, "layer", None)
    if layer is not None and hasattr(layer, "transform_with_params") and hasattr(layer, "params"):
        return layer.transform_with_params(features, layer.params)
    return features


def _apply_stack_compensated_guarded_update_v87(
    stack: Any,
    head: Any,
    entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]],
    role_entries_by_name: Dict[str, List[Tuple[str, str, torch.Tensor, torch.Tensor]]],
    xb: torch.Tensor,
    yb: torch.Tensor,
    xh: torch.Tensor,
    yh: torch.Tensor,
    spec: Any,
    *,
    loss_before: float,
    task_loss_after: float,
    holdout_loss_after: float,
    task_features_after: torch.Tensor,
    holdout_features_after: torch.Tensor,
    func_alpha: float,
    role_weights: Dict[str, float],
    role_budgets: Dict[str, float],
    absolute_ce_budget: float = 0.0,
) -> Tuple[float, float, float, float]:
    stack_entries = role_entries_by_name.get("stack", [])
    stack_weight = float(role_weights.get("stack", 0.0))
    stack_budget = float(role_budgets.get("stack", 0.0))
    if not stack_entries or stack_weight == 0.0:
        return float(task_loss_after), 0.0, 0.0, 0.0
    func_norm = v85.v83._streamed_second_diff_norm(stack_entries)
    stack_rollback = v85.v83._apply_streamed_second_diff_correction_with_fixed_coeff_rollback(
        stack_entries,
        float(func_alpha) * stack_weight,
    )
    mix_delta: torch.Tensor | None = None
    mix_param: torch.Tensor | None = None
    try:
        layer = getattr(head, "layer", None)
        params = getattr(layer, "params", {}) if layer is not None else {}
        if isinstance(params, dict) and "mix" in params:
            mix_param = params["mix"]
            with torch.no_grad():
                z_old_task = _head_transformed_features_v87(head, task_features_after)
                z_old_holdout = _head_transformed_features_v87(head, holdout_features_after)
                target_logits = torch.cat([z_old_task, z_old_holdout], dim=0) @ mix_param.t()
                new_task_features, _ = stack.forward_manual(xb)
                new_holdout_features, _ = stack.forward_manual(xh)
                z_new = torch.cat(
                    [
                        _head_transformed_features_v87(head, new_task_features),
                        _head_transformed_features_v87(head, new_holdout_features),
                    ],
                    dim=0,
                )
                residual = target_logits - z_new @ mix_param.t()
                hidden_dim = int(z_new.shape[1])
                ridge = 1.0e-3 * torch.mean(z_new.detach().float().pow(2)).clamp_min(1.0e-8)
                eye = torch.eye(hidden_dim, device=z_new.device, dtype=z_new.dtype)
                gram = z_new.t() @ z_new + ridge.to(z_new.dtype) * eye
                rhs = z_new.t() @ residual
                delta_t = torch.linalg.solve(gram, rhs)
                mix_delta = delta_t.t().to(mix_param.dtype)
                max_delta_norm = 0.10 * float(mix_param.detach().float().norm().cpu())
                delta_norm = float(mix_delta.detach().float().norm().cpu())
                if delta_norm > max_delta_norm > 0.0:
                    mix_delta.mul_(max_delta_norm / max(delta_norm, 1.0e-12))
                mix_param.add_(mix_delta)
        budget_allowance = max(
            stack_budget * max(0.0, float(loss_before) - float(task_loss_after)),
            float(absolute_ce_budget),
        )
        role_accept_limit = float(task_loss_after) + budget_allowance
        role_holdout_limit = float(holdout_loss_after) + budget_allowance
        role_loss, role_holdout, _task_features_next, _holdout_features_next = v85.v83._loss_pair_and_features_only(
            stack,
            head,
            xb,
            yb,
            xh,
            yh,
            spec,
        )
        if role_loss <= role_accept_limit + 1.0e-8 and role_holdout <= role_holdout_limit + 1.0e-8:
            return float(role_loss), float(func_norm), 0.025, 0.0
    except RuntimeError:
        pass
    if mix_param is not None and mix_delta is not None:
        with torch.no_grad():
            mix_param.sub_(mix_delta)
    v85.v83._rollback_streamed_second_diff_correction(stack_rollback)
    return float(task_loss_after), float(func_norm), 0.0, 0.5


def _stack_layer_inputs_v87(stack: Any, x: torch.Tensor) -> List[torch.Tensor]:
    if not hasattr(stack, "_mixes"):
        return [x.detach()]
    mixes = stack._mixes()
    out: List[torch.Tensor] = []
    h = x
    with torch.no_grad():
        for i, mix in enumerate(mixes):
            out.append(h.detach())
            y = h @ mix.t()
            h = torch.nn.functional.silu(y) if i < len(mixes) - 1 else y
    return out


def _second_diff_delta_v87(param: torch.Tensor, coeff: float) -> torch.Tensor:
    delta = torch.zeros_like(param)
    if coeff == 0.0 or param.numel() < 3 or param.shape[-1] < 3:
        return delta
    second = param[..., 2:] - 2.0 * param[..., 1:-1] + param[..., :-2]
    delta[..., :-2].add_(second, alpha=-coeff)
    delta[..., 1:-1].add_(second, alpha=2.0 * coeff)
    delta[..., 2:].add_(second, alpha=-coeff)
    return delta


def _nullspace_project_rows_v87(delta: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    if delta.numel() == 0 or features.numel() == 0:
        return delta
    h = features.detach().to(delta.dtype)
    if h.shape[1] != delta.shape[1]:
        return delta
    # Project each row of delta to the approximate nullspace of h so that
    # h @ delta.T is small on the current train/holdout batch.  This is a
    # GPU-side linear algebra update rule, not CPU offload or loss training.
    gram = h @ h.t()
    ridge = 1.0e-3 * torch.mean(torch.diag(gram).detach().float()).clamp_min(1.0e-8)
    eye = torch.eye(int(gram.shape[0]), device=gram.device, dtype=gram.dtype)
    rhs = h @ delta.t()
    coeff = torch.linalg.solve(gram + ridge.to(gram.dtype) * eye, rhs)
    row_component = coeff.t() @ h
    return delta - row_component.to(delta.dtype)


def _apply_stack_nullspace_guarded_update_v87(
    stack: Any,
    head: Any,
    xb: torch.Tensor,
    yb: torch.Tensor,
    xh: torch.Tensor,
    yh: torch.Tensor,
    spec: Any,
    *,
    loss_before: float,
    task_loss_after: float,
    holdout_loss_after: float,
    func_alpha: float,
    role_budgets: Dict[str, float],
    max_layers: int = 1,
) -> Tuple[float, float, float, float]:
    if not hasattr(stack, "_mixes") or not hasattr(stack, "flat") or not hasattr(stack, "starts") or not hasattr(stack, "shapes"):
        return float(task_loss_after), 0.0, 0.0, 0.5
    train_inputs = _stack_layer_inputs_v87(stack, xb)
    holdout_inputs = _stack_layer_inputs_v87(stack, xh)
    rollbacks: List[Tuple[torch.Tensor, torch.Tensor]] = []
    update_norm_sq = 0.0
    with torch.no_grad():
        for layer_idx, (start, shape) in enumerate(zip(stack.starts, stack.shapes)):
            if layer_idx >= int(max_layers):
                break
            end = int(start) + int(shape[0] * shape[1])
            param = stack.flat[int(start):end].view(shape)
            raw_delta = _second_diff_delta_v87(param, float(func_alpha))
            features = torch.cat([train_inputs[layer_idx], holdout_inputs[layer_idx]], dim=0)
            projected_delta = _nullspace_project_rows_v87(raw_delta, features)
            if torch.isfinite(projected_delta).all() and float(projected_delta.detach().float().norm().cpu()) > 0.0:
                param.add_(projected_delta)
                rollbacks.append((param, projected_delta))
                update_norm_sq += float(projected_delta.detach().float().pow(2).sum().cpu())
    budget = float(role_budgets.get("stack", 0.05)) * max(0.0, float(loss_before) - float(task_loss_after))
    role_accept_limit = float(task_loss_after) + budget
    role_holdout_limit = float(holdout_loss_after) + budget
    role_loss, role_holdout, _tf, _hf = v85.v83._loss_pair_and_features_only(
        stack,
        head,
        xb,
        yb,
        xh,
        yh,
        spec,
    )
    if role_loss <= role_accept_limit + 1.0e-8 and role_holdout <= role_holdout_limit + 1.0e-8:
        return float(role_loss), math.sqrt(max(update_norm_sq, 0.0)), 0.025, 0.0
    with torch.no_grad():
        for param, delta in reversed(rollbacks):
            param.sub_(delta)
    return float(task_loss_after), math.sqrt(max(update_norm_sq, 0.0)), 0.0, 0.5


def _adaptive_candidate_direction(
    controller_id: str,
    entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]],
    task_dirs: Sequence[torch.Tensor],
    *,
    event_score: float,
    event_threshold: float,
    bad_step_risk: float,
    task_holdout_descent: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict[str, float], Dict[str, float], int, float, float, float, str]:
    base_dirs = v85.v83._base_functional_dirs(entries, task_dirs)
    func_dirs = base_dirs["curv"]
    roles = sorted({str(role) for role, _name, _param, _grad in entries})
    if controller_id == "NoOp":
        zeros = [torch.zeros_like(t) for t in task_dirs]
        return task_dirs, zeros, {role: 0.0 for role in roles}, {role: 0.0 for role in roles}, 0, 0.0, 0.0, 0.0, "no_functional_update"
    if controller_id == "RandomFunc":
        random_dirs: List[torch.Tensor] = []
        for idx, tensor in enumerate(func_dirs):
            sign = -1.0 if idx % 2 else 1.0
            random_dirs.append(tensor * sign)
        corrected, trust_scale, clip_rate = v85.v83._project_task_aware(
            task_dirs,
            random_dirs,
            trust_ratio=0.10,
            role_weights=v85.v83.FT7_ROLE_WEIGHTS,
            entries=entries,
        )
        return corrected, random_dirs, dict(v85.v83.FT7_ROLE_WEIGHTS), dict(v85.v83.FT7_ROLE_BUDGETS), 1, 0.10, trust_scale, clip_rate, "same_cost_signed_curvature_control"
    if controller_id == "Fixed-FT7-stride128":
        corrected, trust_scale, clip_rate = v85.v83._project_task_aware(
            task_dirs,
            func_dirs,
            trust_ratio=0.10,
            role_weights=v85.v83.FT7_ROLE_WEIGHTS,
            entries=entries,
        )
        return corrected, func_dirs, dict(v85.v83.FT7_ROLE_WEIGHTS), dict(v85.v83.FT7_ROLE_BUDGETS), 1, 0.10, trust_scale, clip_rate, "fixed_stride128_role_budget"

    if controller_id == "Adaptive-FT-E":
        trigger = int(
            math.isfinite(event_threshold)
            and float(event_score) >= float(event_threshold)
            and float(bad_step_risk) <= 0.02
        )
        role_weights = dict(v85.v83.FT7_ROLE_WEIGHTS)
        role_budgets = dict(v85.v83.FT7_ROLE_BUDGETS)
        total_budget = sum(float(v) for v in role_budgets.values())
        rule = "sliding_q75_event_score_badstep_guard_default_role_budget"
        if not trigger:
            zeros = [torch.zeros_like(t) for t in task_dirs]
            return task_dirs, zeros, role_weights, role_budgets, 0, total_budget, 0.0, 0.0, rule
        corrected, trust_scale, clip_rate = _project_task_aware_v87(
            task_dirs,
            func_dirs,
            trust_ratio=0.20,
            role_weights=role_weights,
            entries=entries,
            max_scale=0.60,
        )
        return corrected, func_dirs, role_weights, role_budgets, 1, total_budget, trust_scale, clip_rate, rule

    if controller_id == "Adaptive-FT-F":
        trigger = int(
            math.isfinite(event_threshold)
            and float(event_score) >= float(event_threshold)
            and float(bad_step_risk) <= 0.02
        )
        role_weights = dict(v85.v83.FT7_ROLE_WEIGHTS)
        role_budgets = {role: 0.10 for role in role_weights}
        total_budget = sum(float(v) for v in role_budgets.values())
        rule = "sliding_q75_event_score_badstep_guard_balanced_role_budget_010"
        if not trigger:
            zeros = [torch.zeros_like(t) for t in task_dirs]
            return task_dirs, zeros, role_weights, role_budgets, 0, total_budget, 0.0, 0.0, rule
        corrected, trust_scale, clip_rate = _project_task_aware_v87(
            task_dirs,
            func_dirs,
            trust_ratio=0.16,
            role_weights=role_weights,
            entries=entries,
            max_scale=0.60,
        )
        return corrected, func_dirs, role_weights, role_budgets, 1, total_budget, trust_scale, clip_rate, rule

    if controller_id == "Adaptive-FT-G":
        trigger = int(
            math.isfinite(event_threshold)
            and float(event_score) >= float(event_threshold)
            and float(bad_step_risk) <= 0.02
        )
        role_weights = dict(v85.v83.FT7_ROLE_WEIGHTS)
        role_budgets = {role: 0.05 for role in role_weights}
        total_budget = sum(float(v) for v in role_budgets.values())
        rule = "sliding_q75_event_score_badstep_guard_balanced_role_budget_005"
        if not trigger:
            zeros = [torch.zeros_like(t) for t in task_dirs]
            return task_dirs, zeros, role_weights, role_budgets, 0, total_budget, 0.0, 0.0, rule
        corrected, trust_scale, clip_rate = _project_task_aware_v87(
            task_dirs,
            func_dirs,
            trust_ratio=0.12,
            role_weights=role_weights,
            entries=entries,
            max_scale=0.60,
        )
        return corrected, func_dirs, role_weights, role_budgets, 1, total_budget, trust_scale, clip_rate, rule

    if controller_id == "Adaptive-FT-H":
        trigger = int(
            math.isfinite(event_threshold)
            and float(event_score) >= float(event_threshold)
            and float(bad_step_risk) <= 0.02
        )
        role_weights = {"stack": 1.00, "head": 0.25}
        role_budgets = {"stack": 0.075, "head": 0.020}
        total_budget = sum(float(v) for v in role_budgets.values())
        rule = "sliding_q75_event_score_badstep_guard_stack_local_micro_budget"
        if not trigger:
            zeros = [torch.zeros_like(t) for t in task_dirs]
            return task_dirs, zeros, role_weights, role_budgets, 0, total_budget, 0.0, 0.0, rule
        corrected, trust_scale, clip_rate = _project_task_aware_v87(
            task_dirs,
            func_dirs,
            trust_ratio=0.12,
            role_weights=role_weights,
            entries=entries,
            max_scale=0.60,
        )
        return corrected, func_dirs, role_weights, role_budgets, 1, total_budget, trust_scale, clip_rate, rule

    if controller_id == "Adaptive-FT-I":
        trigger = int(
            math.isfinite(event_threshold)
            and float(event_score) >= float(event_threshold)
            and float(bad_step_risk) <= 0.02
        )
        role_weights = {"stack": 0.90, "head": 0.50}
        role_budgets = {"stack": 0.060, "head": 0.030}
        total_budget = sum(float(v) for v in role_budgets.values())
        rule = "sliding_q75_event_score_badstep_guard_stack_biased_balanced_budget"
        if not trigger:
            zeros = [torch.zeros_like(t) for t in task_dirs]
            return task_dirs, zeros, role_weights, role_budgets, 0, total_budget, 0.0, 0.0, rule
        corrected, trust_scale, clip_rate = _project_task_aware_v87(
            task_dirs,
            func_dirs,
            trust_ratio=0.12,
            role_weights=role_weights,
            entries=entries,
            max_scale=0.60,
        )
        return corrected, func_dirs, role_weights, role_budgets, 1, total_budget, trust_scale, clip_rate, rule

    if controller_id == "Adaptive-FT-J":
        trigger = int(
            math.isfinite(event_threshold)
            and float(event_score) >= float(event_threshold)
            and float(bad_step_risk) <= 0.02
        )
        role_weights = {"stack": 1.50, "head": 0.0}
        role_budgets = {"stack": 0.10, "head": 0.0}
        total_budget = sum(float(v) for v in role_budgets.values())
        rule = "sliding_q75_event_score_badstep_guard_stack_only_strong"
        if not trigger:
            zeros = [torch.zeros_like(t) for t in task_dirs]
            return task_dirs, zeros, role_weights, role_budgets, 0, total_budget, 0.0, 0.0, rule
        corrected, trust_scale, clip_rate = _project_task_aware_v87(
            task_dirs,
            func_dirs,
            trust_ratio=0.20,
            role_weights=role_weights,
            entries=entries,
            max_scale=0.60,
        )
        return corrected, func_dirs, role_weights, role_budgets, 1, total_budget, trust_scale, clip_rate, rule

    if controller_id == "Adaptive-FT-K":
        trigger = int(
            math.isfinite(event_threshold)
            and float(event_score) >= float(event_threshold)
            and float(bad_step_risk) <= 0.02
        )
        role_weights = {"stack": 1.25, "head": 0.0}
        role_budgets = {"stack": 0.075, "head": 0.0}
        total_budget = sum(float(v) for v in role_budgets.values())
        rule = "sliding_q75_event_score_badstep_guard_stack_only_moderate"
        if not trigger:
            zeros = [torch.zeros_like(t) for t in task_dirs]
            return task_dirs, zeros, role_weights, role_budgets, 0, total_budget, 0.0, 0.0, rule
        corrected, trust_scale, clip_rate = _project_task_aware_v87(
            task_dirs,
            func_dirs,
            trust_ratio=0.16,
            role_weights=role_weights,
            entries=entries,
            max_scale=0.60,
        )
        return corrected, func_dirs, role_weights, role_budgets, 1, total_budget, trust_scale, clip_rate, rule

    if controller_id == "Adaptive-FT-L":
        trigger = int(
            math.isfinite(event_threshold)
            and float(event_score) >= float(event_threshold)
            and float(bad_step_risk) <= 0.02
        )
        role_weights = {"stack": 1.50, "head": 0.0}
        role_budgets = {"stack": 0.15, "head": 0.0}
        total_budget = sum(float(v) for v in role_budgets.values())
        rule = "sliding_q75_event_score_badstep_guard_stack_only_strong_budget015"
        if not trigger:
            zeros = [torch.zeros_like(t) for t in task_dirs]
            return task_dirs, zeros, role_weights, role_budgets, 0, total_budget, 0.0, 0.0, rule
        corrected, trust_scale, clip_rate = _project_task_aware_v87(
            task_dirs,
            func_dirs,
            trust_ratio=0.20,
            role_weights=role_weights,
            entries=entries,
            max_scale=0.60,
        )
        return corrected, func_dirs, role_weights, role_budgets, 1, total_budget, trust_scale, clip_rate, rule

    if controller_id == "Adaptive-FT-M":
        trigger = int(
            math.isfinite(event_threshold)
            and float(event_score) >= float(event_threshold)
            and float(bad_step_risk) <= 0.02
        )
        role_weights = {"stack": 1.50, "head": 0.0}
        role_budgets = {"stack": 0.20, "head": 0.0}
        total_budget = sum(float(v) for v in role_budgets.values())
        rule = "sliding_q75_event_score_badstep_guard_stack_only_strong_budget020"
        if not trigger:
            zeros = [torch.zeros_like(t) for t in task_dirs]
            return task_dirs, zeros, role_weights, role_budgets, 0, total_budget, 0.0, 0.0, rule
        corrected, trust_scale, clip_rate = _project_task_aware_v87(
            task_dirs,
            func_dirs,
            trust_ratio=0.20,
            role_weights=role_weights,
            entries=entries,
            max_scale=0.60,
        )
        return corrected, func_dirs, role_weights, role_budgets, 1, total_budget, trust_scale, clip_rate, rule

    if controller_id == "Adaptive-FT-N":
        trigger = int(
            math.isfinite(event_threshold)
            and float(event_score) >= float(event_threshold)
            and float(bad_step_risk) <= 0.02
        )
        role_weights = {"stack": 1.50, "head": 0.0}
        role_budgets = {"stack": 0.10, "head": 0.0}
        total_budget = sum(float(v) for v in role_budgets.values())
        rule = "sliding_q75_event_score_badstep_guard_stack_only_feature_compensated"
        if not trigger:
            zeros = [torch.zeros_like(t) for t in task_dirs]
            return task_dirs, zeros, role_weights, role_budgets, 0, total_budget, 0.0, 0.0, rule
        corrected, trust_scale, clip_rate = _project_task_aware_v87(
            task_dirs,
            func_dirs,
            trust_ratio=0.20,
            role_weights=role_weights,
            entries=entries,
            max_scale=0.60,
        )
        return corrected, func_dirs, role_weights, role_budgets, 1, total_budget, trust_scale, clip_rate, rule

    if controller_id == "Adaptive-FT-O":
        trigger = int(
            math.isfinite(event_threshold)
            and float(event_score) >= float(event_threshold)
            and float(bad_step_risk) <= 0.02
        )
        role_weights = {"stack": 1.50, "head": 0.0}
        role_budgets = {"stack": 0.10, "head": 0.0}
        total_budget = sum(float(v) for v in role_budgets.values())
        rule = "sliding_q75_event_score_badstep_guard_stack_compensated_abs003"
        if not trigger:
            zeros = [torch.zeros_like(t) for t in task_dirs]
            return task_dirs, zeros, role_weights, role_budgets, 0, total_budget, 0.0, 0.0, rule
        corrected, trust_scale, clip_rate = _project_task_aware_v87(
            task_dirs,
            func_dirs,
            trust_ratio=0.20,
            role_weights=role_weights,
            entries=entries,
            max_scale=0.60,
        )
        return corrected, func_dirs, role_weights, role_budgets, 1, total_budget, trust_scale, clip_rate, rule

    if controller_id == "Adaptive-FT-P":
        trigger = int(
            math.isfinite(event_threshold)
            and float(event_score) >= float(event_threshold)
            and float(bad_step_risk) <= 0.02
        )
        role_weights = {"stack": 1.50, "head": 0.0}
        role_budgets = {"stack": 0.05, "head": 0.0}
        total_budget = sum(float(v) for v in role_budgets.values())
        rule = "sliding_q75_event_score_badstep_guard_stack_nullspace_layer0"
        if not trigger:
            zeros = [torch.zeros_like(t) for t in task_dirs]
            return task_dirs, zeros, role_weights, role_budgets, 0, total_budget, 0.0, 0.0, rule
        corrected, trust_scale, clip_rate = _project_task_aware_v87(
            task_dirs,
            func_dirs,
            trust_ratio=0.20,
            role_weights=role_weights,
            entries=entries,
            max_scale=0.60,
        )
        return corrected, func_dirs, role_weights, role_budgets, 1, total_budget, trust_scale, clip_rate, rule

    cost_power = 0.5
    total_budget = 0.10
    trust_ratio = 0.10
    trigger = int(math.isfinite(event_threshold) and float(event_score) >= float(event_threshold))
    rule = "q75_event_score_role_score_budget"
    if controller_id == "Adaptive-FT-B":
        total_budget = 0.20
        trust_ratio = 0.35
        trigger = int(trigger and float(task_holdout_descent) > 0.0)
        rule = "sliding_q75_event_score_task_descent_budgeted_sparse_mass_normalized"
    elif controller_id == "Adaptive-FT-C":
        cost_power = 1.0
        total_budget = 0.20
        trust_ratio = 0.40
        rule = "sliding_q75_event_score_cost_aware_role_budget_sparse_mass_normalized"
    elif controller_id == "Adaptive-FT-D":
        total_budget = 0.16
        trust_ratio = 0.30
        trigger = int(trigger and float(bad_step_risk) <= 0.02)
        rule = "sliding_q75_event_score_conservative_badstep_guard_sparse_mass_normalized"
    else:
        total_budget = 0.20
        trust_ratio = 0.40
        rule = "sliding_q75_event_score_role_score_budget_sparse_mass_normalized"
    raw_scores, _role_cos, _role_curv, _role_cost = _adaptive_role_scores(
        entries,
        func_dirs,
        task_dirs,
        cost_power=cost_power,
    )
    role_weights, role_budgets = _adaptive_role_weights_and_budgets(raw_scores, total_budget=total_budget)
    if not trigger:
        zeros = [torch.zeros_like(t) for t in task_dirs]
        return task_dirs, zeros, role_weights, role_budgets, 0, total_budget, 0.0, 0.0, rule
    corrected, trust_scale, clip_rate = _project_task_aware_v87(
        task_dirs,
        func_dirs,
        trust_ratio=trust_ratio,
        role_weights=role_weights,
        entries=entries,
        max_scale=0.60,
    )
    return corrected, func_dirs, role_weights, role_budgets, 1, total_budget, trust_scale, clip_rate, rule


def _run_adaptive_one_step_audit(args: argparse.Namespace, out_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    device = v85.v83.get_device(args.device)
    v85.v83.v80._patch_for_v80()
    candidate_id = str(args.adaptive_dg_candidate_id)
    hidden_dim = int(args.adaptive_dg_hidden_dim)
    batch_size = int(args.adaptive_batch_size)
    warmup_batches = max(1, int(args.adaptive_warmup_batches))
    audit_batches = max(1, int(args.adaptive_audit_batches))
    seeds = _parse_ints(args.adaptive_seeds) or [1314, 1315, 1316]
    spec = v85.v83.v80._spec_map_v80().get(candidate_id, v85.v83.v80._spec_map_v80()["KW3"])
    controllers = [
        "Fixed-FT7-stride128",
        "Adaptive-FT-A",
        "Adaptive-FT-B",
        "Adaptive-FT-C",
        "Adaptive-FT-D",
        "Adaptive-FT-E",
        "Adaptive-FT-F",
        "Adaptive-FT-G",
        "Adaptive-FT-H",
        "Adaptive-FT-I",
        "Adaptive-FT-J",
        "Adaptive-FT-K",
        "Adaptive-FT-L",
        "Adaptive-FT-M",
        "Adaptive-FT-N",
        "Adaptive-FT-O",
        "Adaptive-FT-P",
        "NoOp",
        "RandomFunc",
    ]
    raw_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        x_train, y_train, _x_test, _y_test, input_dim, num_classes, protocol = v85._load_kanbefair_vision_tensors(
            "MNIST",
            data_root=Path(args.data_root),
            train_size=max(int(args.adaptive_train_size), batch_size * (warmup_batches + audit_batches + 8)),
            test_size=256,
            seed=int(seed),
        )
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        v85.v83.set_seed(v85.v83.v72._stable_seed("v87-adaptive-p2-init", seed, candidate_id, hidden_dim))
        stack, head = v85.v83.v80._make_manual_candidate_v80(
            spec,
            int(input_dim),
            int(num_classes),
            hidden_dim,
            int(args.adaptive_basis_count),
            device,
        )
        v85._configure_dg_update_modes(candidate_id, stack, head)
        entries = v85.v83._param_entries(stack, head)
        snapshot = v85.v83._clone_params(entries)
        event_scores: List[float] = []
        warm_context: List[Tuple[int, float, float, float, float]] = []
        for batch_ix in range(warmup_batches):
            step = batch_ix + 1
            xb, yb = v85.v83.v72.v71._select_batch(x_train, y_train, batch_size, step)
            xh, yh = v85.v83.v72.v71._select_batch(x_train, y_train, batch_size, step + 1009)
            v85.v83._restore_params(entries, snapshot)
            train_loss_before, task_dirs = v85.v83._manual_task_grads(stack, head, xb, yb, spec)
            entries = v85.v83._param_entries(stack, head)
            task_norm = v85.v83._norm(task_dirs)
            param_norm = v85.v83._param_norm(entries)
            target_rel = float(args.adaptive_target_rel_update)
            task_alpha = target_rel * param_norm / max(task_norm, 1.0e-30)
            holdout_loss_before = v85.v83._loss_only(stack, head, xh, yh, spec)
            smooth_before, curv_before = v85.v83._geometry_norms(entries)
            snap2 = v85.v83._clone_params(entries)
            v85.v83._apply_direction(entries, task_dirs, task_alpha)
            holdout_task_after = v85.v83._loss_only(stack, head, xh, yh, spec)
            v85.v83._restore_params(entries, snap2)
            event_score, descent_score, geometry_pressure, bad_step_risk = _adaptive_event_score(
                holdout_loss_before=holdout_loss_before,
                holdout_task_after=holdout_task_after,
                smooth_before=smooth_before,
                curv_before=curv_before,
            )
            event_scores.append(event_score)
            warm_context.append((batch_ix, event_score, descent_score, geometry_pressure, bad_step_risk))
        event_score_window = list(event_scores)

        for batch_ix in range(warmup_batches, warmup_batches + audit_batches):
            step = batch_ix + 1
            xb, yb = v85.v83.v72.v71._select_batch(x_train, y_train, batch_size, step)
            xh, yh = v85.v83.v72.v71._select_batch(x_train, y_train, batch_size, step + 1009)
            v85.v83._restore_params(entries, snapshot)
            train_loss_before, task_dirs = v85.v83._manual_task_grads(stack, head, xb, yb, spec)
            entries = v85.v83._param_entries(stack, head)
            task_norm = v85.v83._norm(task_dirs)
            param_norm = v85.v83._param_norm(entries)
            target_rel = float(args.adaptive_target_rel_update)
            task_alpha = target_rel * param_norm / max(task_norm, 1.0e-30)
            holdout_loss_before = v85.v83._loss_only(stack, head, xh, yh, spec)
            smooth_before, curv_before = v85.v83._geometry_norms(entries)
            snap_task = v85.v83._clone_params(entries)
            v85.v83._apply_direction(entries, task_dirs, task_alpha)
            train_task_after = v85.v83._loss_only(stack, head, xb, yb, spec)
            holdout_task_after = v85.v83._loss_only(stack, head, xh, yh, spec)
            v85.v83._restore_params(entries, snap_task)
            task_train_descent = train_loss_before - train_task_after
            task_holdout_descent = holdout_loss_before - holdout_task_after
            event_score, descent_score, geometry_pressure, bad_step_risk = _adaptive_event_score(
                holdout_loss_before=holdout_loss_before,
                holdout_task_after=holdout_task_after,
                smooth_before=smooth_before,
                curv_before=curv_before,
            )
            event_threshold = _quantile(event_score_window[-warmup_batches:], 0.75)
            base_func_dirs = v85.v83._base_functional_dirs(entries, task_dirs)["curv"]
            raw_role_scores, role_cos, role_curv_norm, role_cost = _adaptive_role_scores(entries, base_func_dirs, task_dirs)
            for controller_id in controllers:
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                direction, func_dirs, role_weights, role_budgets, triggered, total_budget, trust_scale, clip_rate, rule = _adaptive_candidate_direction(
                    controller_id,
                    entries,
                    task_dirs,
                    event_score=event_score,
                    event_threshold=event_threshold,
                    bad_step_risk=bad_step_risk,
                    task_holdout_descent=task_holdout_descent,
                )
                direction_norm = v85.v83._norm(direction)
                alpha = target_rel * param_norm / max(direction_norm, 1.0e-30)
                functional_component = [d - t for d, t in zip(direction, task_dirs)]
                functional_update_norm = v85.v83._norm(functional_component) * alpha
                role_update_share, role_descent_contribution = v85.v83._role_json(entries, direction, task_dirs, alpha)
                snap_apply = v85.v83._clone_params(entries)
                v85.v83._apply_direction(entries, direction, alpha)
                train_after = v85.v83._loss_only(stack, head, xb, yb, spec)
                holdout_after = v85.v83._loss_only(stack, head, xh, yh, spec)
                smooth_after, curv_after = v85.v83._geometry_norms(entries)
                v85.v83._restore_params(entries, snap_apply)
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.synchronize(device)
                update_time_ms = (time.perf_counter() - started) * 1000.0
                actual_train_descent = train_loss_before - train_after
                actual_holdout_descent = holdout_loss_before - holdout_after
                if task_holdout_descent > 1.0e-12:
                    holdout_ratio = actual_holdout_descent / task_holdout_descent
                else:
                    holdout_ratio = 1.0 if actual_holdout_descent >= -1.0e-12 else 0.0
                bad_step = int(actual_holdout_descent < -1.0e-8 or holdout_ratio < 0.95)
                cos_func = v85.v83._cos(func_dirs, task_dirs) if v85.v83._norm(func_dirs) > 0 else 1.0
                cos_corr = v85.v83._cos(direction, task_dirs) if direction_norm > 0 else 1.0
                geometry_reduction = (curv_before - curv_after) / curv_before if curv_before > 1.0e-30 else 0.0
                raw_rows.append({
                    "stage": "P2_ADAPTIVE_FUNCTIONAL_ONE_STEP_RAW_V87",
                    "status": "measured",
                    "dataset": "MNIST",
                    "dataset_protocol": protocol,
                    "seed": int(seed),
                    "batch_ix": int(batch_ix - warmup_batches),
                    "warmup_batches": int(warmup_batches),
                    "controller_id": controller_id,
                    "controller_rule": rule,
                    "base_candidate_id": candidate_id,
                    "hidden_dim": hidden_dim,
                    "event_score": event_score,
                    "event_threshold": event_threshold if math.isfinite(event_threshold) else METRIC_UNAVAILABLE,
                    "event_triggered": int(triggered),
                    "event_interval": METRIC_UNAVAILABLE if controller_id.startswith("Adaptive") else int(args.fixed_ft7_event_stride),
                    "task_budget": max(0.0, task_holdout_descent),
                    "task_descent_score": descent_score,
                    "geometry_pressure": geometry_pressure,
                    "bad_step_risk": bad_step_risk,
                    "manual_grad_norm": task_norm,
                    "task_direction_norm": task_norm,
                    "functional_direction_norm": v85.v83._norm(func_dirs),
                    "corrected_direction_norm": direction_norm,
                    "functional_update_norm": functional_update_norm,
                    "cos_functional_with_task": cos_func,
                    "cos_corrected_with_task": cos_corr,
                    "actual_train_descent": actual_train_descent,
                    "actual_holdout_descent": actual_holdout_descent,
                    "task_train_descent": task_train_descent,
                    "task_holdout_descent": task_holdout_descent,
                    "holdout_descent_ratio": holdout_ratio,
                    "bad_step_flag": bad_step,
                    "update_over_param_norm": alpha * direction_norm / max(param_norm, 1.0e-30),
                    "trust_region_scale": trust_scale,
                    "clip_rate": clip_rate,
                    "edge_smoothness_norm_before": smooth_before,
                    "edge_smoothness_norm_after": smooth_after,
                    "edge_curvature_norm_before": curv_before,
                    "edge_curvature_norm_after": curv_after,
                    "curvature_reduction": geometry_reduction,
                    "role_weight_stack": role_weights.get("stack", 0.0),
                    "role_weight_head": role_weights.get("head", 0.0),
                    "role_budget_stack": role_budgets.get("stack", 0.0),
                    "role_budget_head": role_budgets.get("head", 0.0),
                    "role_score_stack": raw_role_scores.get("stack", 0.0),
                    "role_score_head": raw_role_scores.get("head", 0.0),
                    "role_cos_stack": role_cos.get("stack", 0.0),
                    "role_cos_head": role_cos.get("head", 0.0),
                    "role_curv_norm_stack": role_curv_norm.get("stack", 0.0),
                    "role_curv_norm_head": role_curv_norm.get("head", 0.0),
                    "role_cost_stack": role_cost.get("stack", 0.0),
                    "role_cost_head": role_cost.get("head", 0.0),
                    "role_update_share": role_update_share,
                    "role_descent_contribution": role_descent_contribution,
                    "role_budget_json": _json_dumps_stable(role_budgets),
                    "functional_update_time_ms": update_time_ms,
                    "loss_type": "CE",
                    "geometry_loss_used": 0,
                    "external_teacher_used": 0,
                    "self_teacher_used": 0,
                    "sampler_changed": 0,
                    "class_weight_used": 0,
                    "uses_loss_backward": 0,
                    "uses_autograd_for_functional_metric": 0,
                    "uses_full_jacobian_materialization": 0,
                    "uses_cpu_solve": 0,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
            event_score_window.append(event_score)
        v85.v83._restore_params(entries, snapshot)

    fixed_rows = [row for row in raw_rows if str(row.get("controller_id")) == "Fixed-FT7-stride128"]
    fixed_bad = _finite_mean([_safe_float(row.get("bad_step_flag")) for row in fixed_rows])
    fixed_curv = _finite_mean([_safe_float(row.get("curvature_reduction")) for row in fixed_rows])
    fixed_holdout = _finite_mean([_safe_float(row.get("holdout_descent_ratio")) for row in fixed_rows])
    summary_rows: List[Dict[str, Any]] = []
    for controller_id in controllers:
        rows = [row for row in raw_rows if str(row.get("controller_id")) == controller_id]
        triggered_rows = [row for row in rows if int(_safe_float(row.get("event_triggered"), 0.0)) == 1]
        gate_rows = triggered_rows if (controller_id.startswith("Adaptive") and triggered_rows) else rows
        cos_mean = _finite_mean([_safe_float(row.get("cos_corrected_with_task")) for row in rows])
        hold_mean = _finite_mean([_safe_float(row.get("holdout_descent_ratio")) for row in rows])
        bad_rate = _finite_mean([_safe_float(row.get("bad_step_flag")) for row in rows])
        curv_mean = _finite_mean([_safe_float(row.get("curvature_reduction")) for row in rows])
        gate_cos_mean = _finite_mean([_safe_float(row.get("cos_corrected_with_task")) for row in gate_rows])
        gate_hold_mean = _finite_mean([_safe_float(row.get("holdout_descent_ratio")) for row in gate_rows])
        gate_bad_rate = _finite_mean([_safe_float(row.get("bad_step_flag")) for row in gate_rows])
        gate_curv_mean = _finite_mean([_safe_float(row.get("curvature_reduction")) for row in gate_rows])
        triggered_rate = _finite_mean([_safe_float(row.get("event_triggered")) for row in rows])
        p2_direction_pass = int(
            math.isfinite(gate_cos_mean)
            and math.isfinite(gate_hold_mean)
            and math.isfinite(gate_bad_rate)
            and gate_cos_mean >= 0.85
            and gate_hold_mean >= 0.95
            and gate_bad_rate <= 0.02
        )
        adaptive_vs_fixed_pass = 0
        if str(controller_id).startswith("Adaptive"):
            adaptive_vs_fixed_pass = int(
                p2_direction_pass
                and math.isfinite(fixed_bad)
                and math.isfinite(fixed_curv)
                and gate_bad_rate <= fixed_bad + 1.0e-12
                and gate_curv_mean >= 0.80 * fixed_curv
            )
        elif controller_id == "Fixed-FT7-stride128":
            adaptive_vs_fixed_pass = p2_direction_pass
        summary_rows.append({
            "stage": "P2_ADAPTIVE_FUNCTIONAL_ONE_STEP_V87",
            "status": "measured",
            "controller_id": controller_id,
            "base_candidate_id": candidate_id,
            "hidden_dim": hidden_dim,
            "rows": len(rows),
            "event_triggered_rate": triggered_rate if math.isfinite(triggered_rate) else METRIC_UNAVAILABLE,
            "event_score_mean": _finite_mean([_safe_float(row.get("event_score")) for row in rows]),
            "event_threshold_mean": _finite_mean([_safe_float(row.get("event_threshold")) for row in rows]),
            "cos_corrected_with_task_mean": cos_mean if math.isfinite(cos_mean) else METRIC_UNAVAILABLE,
            "holdout_descent_ratio_mean": hold_mean if math.isfinite(hold_mean) else METRIC_UNAVAILABLE,
            "bad_step_rate": bad_rate if math.isfinite(bad_rate) else METRIC_UNAVAILABLE,
            "curvature_reduction_mean": curv_mean if math.isfinite(curv_mean) else METRIC_UNAVAILABLE,
            "triggered_rows": len(triggered_rows),
            "triggered_cos_corrected_with_task_mean": gate_cos_mean if math.isfinite(gate_cos_mean) else METRIC_UNAVAILABLE,
            "triggered_holdout_descent_ratio_mean": gate_hold_mean if math.isfinite(gate_hold_mean) else METRIC_UNAVAILABLE,
            "triggered_bad_step_rate": gate_bad_rate if math.isfinite(gate_bad_rate) else METRIC_UNAVAILABLE,
            "triggered_curvature_reduction_mean": gate_curv_mean if math.isfinite(gate_curv_mean) else METRIC_UNAVAILABLE,
            "functional_update_norm_mean": _finite_mean([_safe_float(row.get("functional_update_norm")) for row in rows]),
            "functional_update_time_ms_mean": _finite_mean([_safe_float(row.get("functional_update_time_ms")) for row in rows]),
            "role_budget_stack_mean": _finite_mean([_safe_float(row.get("role_budget_stack")) for row in rows]),
            "role_budget_head_mean": _finite_mean([_safe_float(row.get("role_budget_head")) for row in rows]),
            "fixed_bad_step_rate_reference": fixed_bad if math.isfinite(fixed_bad) else METRIC_UNAVAILABLE,
            "fixed_curvature_reduction_reference": fixed_curv if math.isfinite(fixed_curv) else METRIC_UNAVAILABLE,
            "fixed_holdout_descent_ratio_reference": fixed_holdout if math.isfinite(fixed_holdout) else METRIC_UNAVAILABLE,
            "P2_direction_gate_pass": p2_direction_pass,
            "P2_adaptive_vs_fixed_gate_pass": adaptive_vs_fixed_pass,
            "loss_type": "CE",
            "geometry_loss_used": 0,
            "external_teacher_used": 0,
            "self_teacher_used": 0,
            "sampler_changed": 0,
            "class_weight_used": 0,
            "uses_loss_backward": 0,
            "uses_autograd_for_functional_metric": 0,
            "uses_full_jacobian_materialization": 0,
            "uses_cpu_solve": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    adaptive_pass_rows = [
        row for row in summary_rows
        if str(row.get("controller_id")).startswith("Adaptive")
        and int(_safe_float(row.get("P2_adaptive_vs_fixed_gate_pass"), 0.0)) == 1
    ]
    best_row = None
    if adaptive_pass_rows:
        best_row = max(adaptive_pass_rows, key=lambda row: (_safe_float(row.get("triggered_curvature_reduction_mean"), -99.0), -_safe_float(row.get("triggered_bad_step_rate"), 99.0)))
    p2_summary = {
        "stage": "P2_ADAPTIVE_FUNCTIONAL_ONE_STEP_SUMMARY_V87",
        "status": "measured" if raw_rows else "not_run",
        "controller_count": len(controllers),
        "raw_rows": len(raw_rows),
        "seed_count": len(seeds),
        "warmup_batches": warmup_batches,
        "audit_batches": audit_batches,
        "fixed_bad_step_rate": fixed_bad if math.isfinite(fixed_bad) else METRIC_UNAVAILABLE,
        "fixed_curvature_reduction": fixed_curv if math.isfinite(fixed_curv) else METRIC_UNAVAILABLE,
        "fixed_holdout_descent_ratio": fixed_holdout if math.isfinite(fixed_holdout) else METRIC_UNAVAILABLE,
        "best_adaptive_controller_id": (best_row or {}).get("controller_id", METRIC_UNAVAILABLE),
        "best_adaptive_bad_step_rate": (best_row or {}).get("bad_step_rate", METRIC_UNAVAILABLE),
        "best_adaptive_curvature_reduction": (best_row or {}).get("curvature_reduction_mean", METRIC_UNAVAILABLE),
        "best_adaptive_holdout_descent_ratio": (best_row or {}).get("holdout_descent_ratio_mean", METRIC_UNAVAILABLE),
        "best_adaptive_triggered_bad_step_rate": (best_row or {}).get("triggered_bad_step_rate", METRIC_UNAVAILABLE),
        "best_adaptive_triggered_curvature_reduction": (best_row or {}).get("triggered_curvature_reduction_mean", METRIC_UNAVAILABLE),
        "best_adaptive_triggered_holdout_descent_ratio": (best_row or {}).get("triggered_holdout_descent_ratio_mean", METRIC_UNAVAILABLE),
        "p2_adaptive_one_step_pass": int(best_row is not None),
        "p2_gate_rule": "triggered_event_cos>=0.85_holdout_ratio>=0.95_bad_step<=0.02_and_adaptive_triggered_bad<=fixed_bad_triggered_curv>=0.8_fixed_curv",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_csv(out_dir / "adaptive_functional_one_step_raw.csv", raw_rows)
    write_csv(out_dir / "adaptive_functional_one_step.csv", summary_rows)
    write_csv(out_dir / "adaptive_functional_one_step_summary.csv", [p2_summary])
    return summary_rows, p2_summary


def _run_adaptive_multistep_smoke(args: argparse.Namespace, out_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    global _V85_FUSED_USE_COMPILED_CORE
    device = v85.v83.get_device(args.device)
    v85.v83.v80._patch_for_v80()
    candidate_id = str(args.p3_dg_candidate_id)
    hidden_dim = int(args.p3_dg_hidden_dim)
    batch_size = int(args.p3_batch_size)
    seeds = _parse_ints(args.p3_seeds) or [0, 1, 2]
    checkpoints = sorted(set(_parse_ints(args.p3_steps) or [20, 50, 240]))
    max_steps = max(checkpoints)
    adaptive_probe_interval = max(1, int(args.p3_adaptive_probe_interval))
    spec = v85.v83.v80._spec_map_v80().get(candidate_id, v85.v83.v80._spec_map_v80()["KW3"])
    variants = ["Fixed-FT7-stride128", str(args.p3_adaptive_controller_id)]
    old_stride = int(v85.v83.P5_FT7_EVENT_STRIDE)
    old_alpha = float(v85.v83.P5_FT7_EVENT_ALPHA_MULT)
    old_compiled = _V85_FUSED_USE_COMPILED_CORE
    v85.v83.P5_FT7_EVENT_STRIDE = int(args.fixed_ft7_event_stride)
    v85.v83.P5_FT7_EVENT_ALPHA_MULT = float(args.fixed_ft7_event_alpha_mult)
    _V85_FUSED_USE_COMPILED_CORE = bool(getattr(args, "use_compiled_fused_ce_head_backward", False))
    raw_rows: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []
    try:
        for seed in seeds:
            x_train, y_train, x_test, y_test, input_dim, num_classes, protocol = v85._load_kanbefair_vision_tensors(
                "MNIST",
                data_root=Path(args.data_root),
                train_size=int(args.p3_train_size),
                test_size=int(args.p3_test_size),
                seed=int(seed),
            )
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            for variant in variants:
                v85.v83.set_seed(v85.v83.v72._stable_seed("v87-adaptive-p3-init", seed, candidate_id, hidden_dim))
                stack, head = v85.v83.v80._make_manual_candidate_v80(
                    spec,
                    int(input_dim),
                    int(num_classes),
                    hidden_dim,
                    int(args.p3_basis_count),
                    device,
                )
                v85._configure_dg_update_modes(candidate_id, stack, head)
                if bool(getattr(args, "use_compiled_fused_ce_head_backward", False) and getattr(args, "prewarm_compiled_fused_ce_head_backward", False)):
                    hook = _make_v85_compiled_fused_prewarm_hook()
                    hook(stack=stack, head=head, spec=spec, x_train=x_train, y_train=y_train, device=device, batch_size=batch_size)
                params = v85.v83.v80.V63Params(
                    train_size=int(args.p3_train_size),
                    val_size=0,
                    test_size=int(args.p3_test_size),
                    batch_size=batch_size,
                )
                lr = float(params.lr_manual * spec.lr_mult)
                if bool(args.use_foreach_adamw_addcdiv):
                    opt = _ForeachAdamWAddcdivNoSync([stack, head], lr=lr, weight_decay=1.0e-4)
                    optimizer_impl = "ForeachAdamWAddcdivNoSync"
                else:
                    opt = v85.v83.v72.FastAdamWNoSync([stack, head], lr=lr, weight_decay=1.0e-4)
                    optimizer_impl = "FastAdamWNoSync"
                entries = v85.v83._param_entries(stack, head)
                role_entries_by_name = v85.v83._entries_by_role(entries)
                event_score_window: List[float] = []
                event_count = 0
                accept_mass = 0.0
                reject_mass = 0.0
                bad_count = 0
                holdout_ratios: List[float] = []
                step_times: List[float] = []
                train_loss_curve: List[float] = []
                test_acc_curve: List[float] = []
                ece_curve: List[float] = []
                nll_curve: List[float] = []
                curv_curve: List[float] = []
                event_count_curve: List[int] = []
                event_interval_curve: List[int] = []
                role_budget_stack_curve: List[float] = []
                role_budget_head_curve: List[float] = []
                last_event_step = 0
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                for step in range(1, max_steps + 1):
                    xb, yb = v85.v83.v72.v71._select_batch(x_train, y_train, batch_size, step)
                    xh, yh = v85.v83.v72.v71._select_batch(x_train, y_train, batch_size, step + 1009)
                    is_adaptive = str(variant).startswith("Adaptive")
                    adaptive_probe_step = bool(is_adaptive and step % adaptive_probe_interval == 0)
                    fixed_event = bool(variant == "Fixed-FT7-stride128" and step % int(args.fixed_ft7_event_stride) == 0)
                    if torch.cuda.is_available() and device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    v85.v83._zero_grad(stack, head)
                    if adaptive_probe_step or fixed_event:
                        loss_before, holdout_loss_before = _manual_ce_backward_with_holdout_v85_fused(stack, head, xb, yb, xh, yh, spec)
                    else:
                        loss_value = _manual_ce_backward_v85_fused(stack, head, xb, yb, spec, need_loss_float=True)
                        loss_before = float(loss_value) if loss_value is not None else 0.0
                        holdout_loss_before = 0.0
                    entries = v85.v83._param_entries(stack, head)
                    task_dirs = [-grad.detach().clone() for _role, _name, _param, grad in entries]
                    opt.step(step, max_steps, warmup_cosine=True)
                    loss_after_task = float(loss_before)
                    holdout_loss_after_task = 0.0
                    event_score = float("nan")
                    event_threshold = float("nan")
                    event_triggered = 0
                    event_rule = "fixed_stride128" if fixed_event else "no_event"
                    role_weights = dict(v85.v83.FT7_ROLE_WEIGHTS)
                    role_budgets = dict(v85.v83.FT7_ROLE_BUDGETS)
                    trust_scale = 0.0
                    clip_rate = 0.0
                    if adaptive_probe_step or fixed_event:
                        loss_after_task, holdout_loss_after_task, task_features_after, holdout_features_after = v85.v83._loss_pair_and_features_only(
                            stack,
                            head,
                            xb,
                            yb,
                            xh,
                            yh,
                            spec,
                        )
                    if adaptive_probe_step:
                        smooth_before, curv_before = v85.v83._geometry_norms(v85.v83._param_entries(stack, head))
                        event_score, _descent_score, _geometry_pressure, bad_step_risk = _adaptive_event_score(
                            holdout_loss_before=holdout_loss_before,
                            holdout_task_after=holdout_loss_after_task,
                            smooth_before=smooth_before,
                            curv_before=curv_before,
                        )
                        if len(event_score_window) >= int(args.p3_adaptive_warmup_steps):
                            event_threshold = _quantile(
                                event_score_window[-int(args.p3_adaptive_warmup_steps):],
                                float(args.p3_adaptive_event_quantile),
                            )
                            _direction, _func_dirs, role_weights, role_budgets, event_triggered, _total_budget, trust_scale, clip_rate, event_rule = _adaptive_candidate_direction(
                                str(args.p3_adaptive_controller_id),
                                v85.v83._param_entries(stack, head),
                                task_dirs,
                                event_score=event_score,
                                event_threshold=event_threshold,
                                bad_step_risk=bad_step_risk,
                                task_holdout_descent=holdout_loss_before - holdout_loss_after_task,
                            )
                        event_score_window.append(event_score)
                    else:
                        event_triggered = int(fixed_event)
                    if event_triggered:
                        original_weights = dict(v85.v83.FT7_ROLE_WEIGHTS)
                        original_budgets = dict(v85.v83.FT7_ROLE_BUDGETS)
                        try:
                            v85.v83.FT7_ROLE_WEIGHTS = role_weights
                            v85.v83.FT7_ROLE_BUDGETS = role_budgets
                            alpha_scale = adaptive_probe_interval if (is_adaptive and bool(args.p3_adaptive_alpha_scale_with_probe_interval)) else 1.0
                            func_alpha = lr * 0.05 * float(args.fixed_ft7_event_alpha_mult) * float(alpha_scale)
                            if str(args.p3_adaptive_controller_id) == "Adaptive-FT-P":
                                _loss_after, _func_norm, trust_delta, reject_delta = _apply_stack_nullspace_guarded_update_v87(
                                    stack,
                                    head,
                                    xb,
                                    yb,
                                    xh,
                                    yh,
                                    spec,
                                    loss_before=loss_before,
                                    task_loss_after=loss_after_task,
                                    holdout_loss_after=holdout_loss_after_task,
                                    func_alpha=func_alpha * float(role_weights.get("stack", 1.0)),
                                    role_budgets=role_budgets,
                                    max_layers=1,
                                )
                            elif str(args.p3_adaptive_controller_id) in {"Adaptive-FT-N", "Adaptive-FT-O"}:
                                _loss_after, _func_norm, trust_delta, reject_delta = _apply_stack_compensated_guarded_update_v87(
                                    stack,
                                    head,
                                    v85.v83._param_entries(stack, head),
                                    role_entries_by_name,
                                    xb,
                                    yb,
                                    xh,
                                    yh,
                                    spec,
                                    loss_before=loss_before,
                                    task_loss_after=loss_after_task,
                                    holdout_loss_after=holdout_loss_after_task,
                                    task_features_after=task_features_after,
                                    holdout_features_after=holdout_features_after,
                                    func_alpha=func_alpha,
                                    role_weights=role_weights,
                                    role_budgets=role_budgets,
                                    absolute_ce_budget=0.003 if str(args.p3_adaptive_controller_id) == "Adaptive-FT-O" else 0.0,
                                )
                            else:
                                _loss_after, _func_norm, trust_delta, reject_delta = v85.v83._apply_ft7_streamed_guarded_update(
                                    stack,
                                    head,
                                    v85.v83._param_entries(stack, head),
                                    xb,
                                    yb,
                                    xh,
                                    yh,
                                    spec,
                                    loss_before=loss_before,
                                    holdout_loss_before=holdout_loss_before,
                                    task_loss_after=loss_after_task,
                                    holdout_loss_after=holdout_loss_after_task,
                                    func_alpha=func_alpha,
                                    task_features_after=task_features_after,
                                    holdout_features_after=holdout_features_after,
                                    role_entries_by_name=role_entries_by_name,
                                    pair_stack_guard_forward=True,
                                )
                        finally:
                            v85.v83.FT7_ROLE_WEIGHTS = original_weights
                            v85.v83.FT7_ROLE_BUDGETS = original_budgets
                        event_count += 1
                        accept_mass += float(trust_delta)
                        reject_mass += float(reject_delta)
                        event_interval_curve.append(step - last_event_step if last_event_step else step)
                        last_event_step = step
                        holdout_after_func = v85.v83._loss_only(stack, head, xh, yh, spec)
                        task_budget = holdout_loss_before - holdout_loss_after_task
                        if task_budget > 1.0e-12:
                            holdout_ratio = (holdout_loss_before - holdout_after_func) / task_budget
                        else:
                            holdout_ratio = 1.0 if holdout_loss_before - holdout_after_func >= -1.0e-12 else 0.0
                        holdout_ratios.append(float(holdout_ratio))
                        if holdout_ratio < 0.95:
                            bad_count += 1
                    if torch.cuda.is_available() and device.type == "cuda":
                        torch.cuda.synchronize(device)
                    step_ms = (time.perf_counter() - started) * 1000.0
                    step_times.append(step_ms)
                    train_loss_curve.append(float(loss_after_task))
                    _smooth_now, curv_now = v85.v83._geometry_norms(v85.v83._param_entries(stack, head))
                    curv_curve.append(float(curv_now))
                    event_count_curve.append(int(event_count))
                    role_budget_stack_curve.append(float(role_budgets.get("stack", 0.0)) if event_triggered else 0.0)
                    role_budget_head_curve.append(float(role_budgets.get("head", 0.0)) if event_triggered else 0.0)
                    raw_rows.append({
                        "stage": "P3_ADAPTIVE_FUNCTIONAL_MULTISTEP_RAW_V87",
                        "status": "measured",
                        "dataset": "MNIST",
                        "dataset_protocol": protocol,
                        "seed": int(seed),
                        "variant": variant,
                        "controller_id": variant,
                        "step": int(step),
                        "train_loss": float(loss_after_task),
                        "curvature": float(curv_now),
                        "step_time_ms": float(step_ms),
                        "event_score": event_score if math.isfinite(event_score) else METRIC_UNAVAILABLE,
                        "event_threshold": event_threshold if math.isfinite(event_threshold) else METRIC_UNAVAILABLE,
                        "event_triggered": int(event_triggered),
                        "event_rule": event_rule,
                        "adaptive_probe_interval": adaptive_probe_interval if is_adaptive else METRIC_UNAVAILABLE,
                        "adaptive_probe_step": int(adaptive_probe_step),
                        "adaptive_alpha_scaled_with_probe_interval": int(bool(args.p3_adaptive_alpha_scale_with_probe_interval) and is_adaptive),
                        "event_count": int(event_count),
                        "role_budget_stack": float(role_budgets.get("stack", 0.0)) if event_triggered else 0.0,
                        "role_budget_head": float(role_budgets.get("head", 0.0)) if event_triggered else 0.0,
                        "trust_region_scale": trust_scale,
                        "clip_rate": clip_rate,
                        "bad_step_count": int(bad_count),
                        "bad_step_rate": bad_count / max(1, event_count),
                        "holdout_descent_ratio_mean": _finite_mean(holdout_ratios),
                        "loss_type": "CE",
                        "geometry_loss_used": 0,
                        "external_teacher_used": 0,
                        "self_teacher_used": 0,
                        "sampler_changed": 0,
                        "class_weight_used": 0,
                        "cpu_offload_used": 0,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                    })
                    if step in checkpoints:
                        eval_metrics = v85.v83.v72.v71._manual_eval(
                            stack,
                            head,
                            x_test,
                            y_test,
                            int(args.p3_eval_batch_size),
                            int(num_classes),
                        )
                        _smooth_ckpt, curv_ckpt = v85.v83._geometry_norms(v85.v83._param_entries(stack, head))
                        test_acc_curve.append(float(eval_metrics["acc"]))
                        ece_curve.append(float(eval_metrics["ECE"]))
                        nll_curve.append(float(eval_metrics["NLL"]))
                        checkpoint_rows.append({
                            "stage": "P3_ADAPTIVE_FUNCTIONAL_MULTISTEP_V87",
                            "status": "measured",
                            "dataset": "MNIST",
                            "dataset_protocol": protocol,
                            "seed": int(seed),
                            "variant": variant,
                            "controller_id": variant,
                            "base_candidate_id": candidate_id,
                            "hidden_dim": hidden_dim,
                            "checkpoint_step": int(step),
                            "test_acc": float(eval_metrics["acc"]),
                            "test_acc_percent": float(eval_metrics["acc"]) * 100.0,
                            "test_loss": float(eval_metrics["loss"]),
                            "ECE": float(eval_metrics["ECE"]),
                            "NLL": float(eval_metrics["NLL"]),
                            "curvature": float(curv_ckpt),
                            "step_time_ms_mean": _finite_mean(step_times),
                            "event_count": int(event_count),
                            "event_interval_curve": _json_dumps_stable(event_interval_curve),
                            "event_count_curve": _json_dumps_stable(event_count_curve),
                            "train_loss_curve": _json_dumps_stable(train_loss_curve),
                            "test_acc_curve": _json_dumps_stable(test_acc_curve),
                            "ECE_curve": _json_dumps_stable(ece_curve),
                            "NLL_curve": _json_dumps_stable(nll_curve),
                            "curvature_curve": _json_dumps_stable(curv_curve),
                            "step_time_curve": _json_dumps_stable(step_times),
                            "role_budget_stack_curve": _json_dumps_stable(role_budget_stack_curve),
                            "role_budget_head_curve": _json_dumps_stable(role_budget_head_curve),
                            "bad_step_rate_curve": _json_dumps_stable([_safe_float(r.get("bad_step_rate")) for r in raw_rows if int(_safe_float(r.get("seed"), -1)) == int(seed) and str(r.get("variant")) == variant]),
                            "holdout_descent_ratio_curve": _json_dumps_stable(holdout_ratios),
                            "bad_step_rate": bad_count / max(1, event_count),
                            "holdout_descent_ratio_mean": _finite_mean(holdout_ratios),
                            "functional_accept_mass": accept_mass,
                            "functional_reject_mass": reject_mass,
                            "optimizer_impl": optimizer_impl,
                            "ce_head_backward_impl": (
                                "compiled_fused_value_equivalent"
                                if bool(getattr(args, "use_compiled_fused_ce_head_backward", False))
                                else ("fused_value_equivalent" if bool(getattr(args, "use_fused_ce_head_backward", False)) else "v85_default")
                            ),
                            "loss_type": "CE",
                            "geometry_loss_used": 0,
                            "external_teacher_used": 0,
                            "self_teacher_used": 0,
                            "sampler_changed": 0,
                            "class_weight_used": 0,
                            "cpu_offload_used": 0,
                            "fake_data_used": 0,
                            "proxy_row_used": 0,
                        })
                if torch.cuda.is_available() and device.type == "cuda":
                    peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                else:
                    peak_mb = 0.0
                for row in checkpoint_rows:
                    if int(_safe_float(row.get("seed"), -1)) == int(seed) and str(row.get("variant")) == variant:
                        row["peak_memory_MB"] = peak_mb
    finally:
        v85.v83.P5_FT7_EVENT_STRIDE = old_stride
        v85.v83.P5_FT7_EVENT_ALPHA_MULT = old_alpha
        _V85_FUSED_USE_COMPILED_CORE = old_compiled

    aggregate_rows: List[Dict[str, Any]] = []
    p3_pass_values: List[int] = []
    for step in checkpoints:
        fixed = [row for row in checkpoint_rows if int(_safe_float(row.get("checkpoint_step"), -1)) == int(step) and str(row.get("variant")) == "Fixed-FT7-stride128"]
        adaptive = [row for row in checkpoint_rows if int(_safe_float(row.get("checkpoint_step"), -1)) == int(step) and str(row.get("variant")).startswith("Adaptive")]
        fixed_acc = _finite_mean([_safe_float(row.get("test_acc")) for row in fixed])
        adaptive_acc = _finite_mean([_safe_float(row.get("test_acc")) for row in adaptive])
        fixed_curv = _finite_mean([_safe_float(row.get("curvature")) for row in fixed])
        adaptive_curv = _finite_mean([_safe_float(row.get("curvature")) for row in adaptive])
        fixed_step = _finite_mean([_safe_float(row.get("step_time_ms_mean")) for row in fixed])
        adaptive_step = _finite_mean([_safe_float(row.get("step_time_ms_mean")) for row in adaptive])
        curvature_ratio = adaptive_curv / max(fixed_curv, 1.0e-12) if math.isfinite(adaptive_curv) and math.isfinite(fixed_curv) else float("nan")
        step_ratio = adaptive_step / max(fixed_step, 1.0e-12) if math.isfinite(adaptive_step) and math.isfinite(fixed_step) else float("nan")
        acc_delta = adaptive_acc - fixed_acc if math.isfinite(adaptive_acc) and math.isfinite(fixed_acc) else float("nan")
        pass_value = int(
            math.isfinite(acc_delta)
            and acc_delta >= -0.002
            and math.isfinite(curvature_ratio)
            and curvature_ratio <= 0.90
            and math.isfinite(step_ratio)
            and step_ratio <= 1.50
        )
        p3_pass_values.append(pass_value)
        aggregate_rows.append({
            "stage": "P3_ADAPTIVE_FUNCTIONAL_MULTISTEP_SUMMARY_V87",
            "status": "measured",
            "checkpoint_step": int(step),
            "fixed_test_acc_mean": fixed_acc if math.isfinite(fixed_acc) else METRIC_UNAVAILABLE,
            "adaptive_test_acc_mean": adaptive_acc if math.isfinite(adaptive_acc) else METRIC_UNAVAILABLE,
            "adaptive_acc_delta_vs_fixed": acc_delta if math.isfinite(acc_delta) else METRIC_UNAVAILABLE,
            "fixed_curvature_mean": fixed_curv if math.isfinite(fixed_curv) else METRIC_UNAVAILABLE,
            "adaptive_curvature_mean": adaptive_curv if math.isfinite(adaptive_curv) else METRIC_UNAVAILABLE,
            "adaptive_curvature_ratio_vs_fixed": curvature_ratio if math.isfinite(curvature_ratio) else METRIC_UNAVAILABLE,
            "fixed_step_time_ms_mean": fixed_step if math.isfinite(fixed_step) else METRIC_UNAVAILABLE,
            "adaptive_step_time_ms_mean": adaptive_step if math.isfinite(adaptive_step) else METRIC_UNAVAILABLE,
            "adaptive_step_ratio_vs_fixed": step_ratio if math.isfinite(step_ratio) else METRIC_UNAVAILABLE,
            "adaptive_event_count_mean": _finite_mean([_safe_float(row.get("event_count")) for row in adaptive]),
            "adaptive_bad_step_rate_mean": _finite_mean([_safe_float(row.get("bad_step_rate")) for row in adaptive]),
            "adaptive_holdout_descent_ratio_mean": _finite_mean([_safe_float(row.get("holdout_descent_ratio_mean")) for row in adaptive]),
            "P3_checkpoint_pass": pass_value,
            "loss_type": "CE",
            "geometry_loss_used": 0,
            "external_teacher_used": 0,
            "self_teacher_used": 0,
            "sampler_changed": 0,
            "class_weight_used": 0,
            "cpu_offload_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    p3_summary = {
        "stage": "P3_ADAPTIVE_FUNCTIONAL_MULTISTEP_ROUTE_SUMMARY_V87",
        "status": "measured" if checkpoint_rows else "not_run",
        "seed_count": len(seeds),
        "checkpoints": args.p3_steps,
        "adaptive_probe_interval": adaptive_probe_interval,
        "adaptive_alpha_scaled_with_probe_interval": int(bool(args.p3_adaptive_alpha_scale_with_probe_interval)),
        "raw_rows": len(raw_rows),
        "checkpoint_rows": len(checkpoint_rows),
        "p3_checkpoint_pass_count": sum(p3_pass_values),
        "p3_checkpoint_count": len(p3_pass_values),
        "p3_adaptive_multistep_pass": int(bool(p3_pass_values) and all(p3_pass_values)),
        "p3_gate_rule": "adaptive_acc>=fixed-0.002_fraction_curvature_ratio_vs_fixed<=0.90_step_ratio_vs_fixed<=1.50_at_all_checkpoints",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_csv(out_dir / "adaptive_functional_multistep_raw.csv", raw_rows)
    write_csv(out_dir / "adaptive_functional_multistep.csv", checkpoint_rows)
    write_csv(out_dir / "adaptive_functional_multistep_summary.csv", aggregate_rows)
    write_csv(out_dir / "adaptive_functional_multistep_route_summary.csv", [p3_summary])
    return checkpoint_rows, p3_summary


def _summarize_p0(rows: Sequence[Dict[str, Any]], child_decisions: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    dg1 = [r for r in rows if str(r.get("candidate_id")).startswith("DG1-FT7")]
    dg1_delta = [_safe_float(r.get("delta_vs_KB_MLP")) for r in dg1]
    dg1_curv = [_safe_float(r.get("curvature_ratio_vs_DG0")) for r in dg1]
    dg1_step = [_safe_float(r.get("step_ratio_vs_KB_MLP")) for r in dg1]
    pass_rows = []
    for r in dg1:
        delta = _safe_float(r.get("delta_vs_KB_MLP"))
        curv = _safe_float(r.get("curvature_ratio_vs_DG0"))
        step = _safe_float(r.get("step_ratio_vs_KB_MLP"))
        pass_rows.append(int(math.isfinite(delta) and delta > 0.0 and math.isfinite(curv) and curv <= 0.90 and math.isfinite(step) and step <= 1.50))
    measured_runs = len({str(r.get("run_id")) for r in dg1 if str(r.get("status")) == "measured"})
    requested_runs = int(args.p0_reruns) * len(_parse_ints(args.p0_seeds))
    if str(args.reuse_p0_out_dir).strip():
        requested_runs = measured_runs
    timing_protocols = len({str(r.get("timing_protocol")) for r in dg1 if str(r.get("status")) == "measured"})
    mean_delta = mean([v for v in dg1_delta if math.isfinite(v)]) if any(math.isfinite(v) for v in dg1_delta) else float("nan")
    mean_curv = mean([v for v in dg1_curv if math.isfinite(v)]) if any(math.isfinite(v) for v in dg1_curv) else float("nan")
    q90_step = _q90(dg1_step)
    pass_rate = mean(pass_rows) if pass_rows else 0.0
    p0_gate_pass = int(
        measured_runs > 0
        and math.isfinite(mean_delta) and mean_delta > 0.0
        and math.isfinite(mean_curv) and mean_curv <= 0.90
        and math.isfinite(q90_step) and q90_step <= 1.50
    )
    p0_full_protocol_complete = int(measured_runs >= 15 and timing_protocols >= 5)
    return {
        "stage": "P0_STABILITY_SUMMARY_V87",
        "status": "measured" if measured_runs else "not_run",
        "requested_p0_runs": requested_runs,
        "measured_p0_runs": measured_runs,
        "timing_protocol_count": timing_protocols,
        "p0_full_protocol_complete": p0_full_protocol_complete,
        "dg1_mean_delta_vs_KB_MLP": mean_delta if math.isfinite(mean_delta) else METRIC_UNAVAILABLE,
        "dg1_mean_curvature_ratio_vs_DG0": mean_curv if math.isfinite(mean_curv) else METRIC_UNAVAILABLE,
        "dg1_q90_step_ratio_vs_KB_MLP": q90_step if math.isfinite(q90_step) else METRIC_UNAVAILABLE,
        "dg1_all_gate_pass_rate": pass_rate,
        "p0_stability_gate_pass": p0_gate_pass,
        "v85_child_external_formal_pass_count": sum(int(_safe_float(d.get("success_v85_external_formal"), 0.0) == 1.0) for d in child_decisions),
        "v85_child_route_set": ",".join(sorted({str(d.get("route")) for d in child_decisions})),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": sum(int(_safe_float(d.get("cpu_offload_used"), 0.0) != 0.0) for d in child_decisions),
    }


def _write_figure(out_dir: Path, summary: Dict[str, Any]) -> None:
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)
    mean_delta = summary.get("dg1_mean_delta_vs_KB_MLP", METRIC_UNAVAILABLE)
    q90_step = summary.get("dg1_q90_step_ratio_vs_KB_MLP", METRIC_UNAVAILABLE)
    mean_curv = summary.get("dg1_mean_curvature_ratio_vs_DG0", METRIC_UNAVAILABLE)
    text = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="240">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="30" y="45" font-family="monospace" font-size="20">DG-KAN v8.7 P0 measured summary</text>'
        f'<text x="30" y="90" font-family="monospace" font-size="16">mean delta vs KB-MLP: {mean_delta}</text>'
        f'<text x="30" y="125" font-family="monospace" font-size="16">Q90 step ratio vs KB-MLP: {q90_step}</text>'
        f'<text x="30" y="160" font-family="monospace" font-size="16">mean curvature ratio vs DG0: {mean_curv}</text>'
        '<text x="30" y="205" font-family="monospace" font-size="14">Later v8.7 waves are not_run unless backed by CSV artifacts.</text>'
        '</svg>'
    )
    (fig_dir / "p0_measured_summary.svg").write_text(text, encoding="utf-8")


def _audit(out_dir: Path) -> Dict[str, Any]:
    rows_checked = 0
    fake_proxy_nonzero = 0
    fake_data = 0
    proxy = 0
    offload = 0
    for path in out_dir.glob("*.csv"):
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows_checked += 1
                fval = int(_safe_float(row.get("fake_data_used"), 0.0) != 0.0)
                pval = int(_safe_float(row.get("proxy_row_used"), 0.0) != 0.0)
                oval = int(_safe_float(row.get("cpu_offload_used"), 0.0) != 0.0)
                fake_data += fval
                proxy += pval
                offload += oval
                fake_proxy_nonzero += int(fval or pval)
    audit = {
        "stage": "V87_PROVENANCE_AUDIT",
        "status": "measured",
        "script_path": "experiments/run_gafu_v87_real.py",
        "plan_path": PLAN_PATH,
        "rows_checked": rows_checked,
        "fake_proxy_nonzero_count": fake_proxy_nonzero,
        "fake_data_used": fake_data,
        "proxy_row_used": proxy,
        "cpu_offload_used": offload,
        "no_fake": fake_data == 0,
        "no_proxy": proxy == 0,
    }
    write_csv(out_dir / "v87_provenance_audit.csv", [audit])
    return audit


def _route(
    summary: Dict[str, Any],
    audit: Dict[str, Any],
    p1_summary: Dict[str, Any] | None = None,
    adaptive_summary: Dict[str, Any] | None = None,
    p3_summary: Dict[str, Any] | None = None,
    p4_summary: Dict[str, Any] | None = None,
    timing_summary: Dict[str, Any] | None = None,
    selected_summary: Dict[str, Any] | None = None,
    p5_summary: Dict[str, Any] | None = None,
    p6_summary: Dict[str, Any] | None = None,
    p7_summary: Dict[str, Any] | None = None,
    p9_summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    p0_gate = int(_safe_float(summary.get("p0_stability_gate_pass"), 0.0) == 1.0)
    full_p0 = int(_safe_float(summary.get("p0_full_protocol_complete"), 0.0) == 1.0)
    mean_delta = _safe_float(summary.get("dg1_mean_delta_vs_KB_MLP"))
    mean_curv = _safe_float(summary.get("dg1_mean_curvature_ratio_vs_DG0"))
    q90_step = _safe_float(summary.get("dg1_q90_step_ratio_vs_KB_MLP"))
    p1_sensitivity = int(_safe_float((p1_summary or {}).get("H1_fixed_schedule_sensitivity_pass"), 0.0) == 1.0)
    p2_adaptive_one_step = int(_safe_float((adaptive_summary or {}).get("p2_adaptive_one_step_pass"), 0.0) == 1.0)
    p3_adaptive_multistep = int(_safe_float((p3_summary or {}).get("p3_adaptive_multistep_pass"), 0.0) == 1.0)
    p4_selection = int(_safe_float((p4_summary or {}).get("selection_pass"), 0.0) == 1.0)
    p4_robust_selection_failed = bool(
        p4_summary
        and str(p4_summary.get("pilot_type")) == "robust_system_envelope_only_no_test_metric"
        and p4_selection == 0
    )
    p4_base_impl_selection_failed = bool(
        p4_summary
        and str(p4_summary.get("pilot_type")) == "robust_base_implementation_system_only_no_test_metric"
        and p4_selection == 0
    )
    robust_timing_partial = int(_safe_float((timing_summary or {}).get("robust_timing_partial_gate_pass"), 0.0) == 1.0)
    robust_timing_full = int(_safe_float((timing_summary or {}).get("robust_timing_full_gate_pass"), 0.0) == 1.0)
    timing_q90 = _safe_float((timing_summary or {}).get("q90_step_ratio_vs_KB_MLP"))
    p5_robust_timing = int(_safe_float((p5_summary or {}).get("robust_timing_pass"), 0.0) == 1.0)
    p5_strict_timing = int(_safe_float((p5_summary or {}).get("strict_timing_pass"), 0.0) == 1.0)
    p5_time_accounting = int(_safe_float((p5_summary or {}).get("time_accounting_pass"), 0.0) == 1.0)
    p6_training_compute = int(_safe_float((p6_summary or {}).get("training_compute_fair_pass"), 0.0) == 1.0)
    p7_cross_task_external = int(_safe_float((p7_summary or {}).get("p7_cross_task_external_pass"), 0.0) == 1.0)
    p9_cross_task_causality = int(_safe_float((p9_summary or {}).get("p9_cross_task_causality_pass"), 0.0) == 1.0)
    p7_formal_protocol = int(_safe_float((p7_summary or {}).get("formal_protocol"), 0.0) == 1.0)
    effective_full_p0 = int(full_p0 or (p0_gate and robust_timing_full))
    selected_confirmation_pass = int(_safe_float((selected_summary or {}).get("selected_task_geometry_confirmation_pass"), 0.0) == 1.0)
    no_contract_issue = bool(audit.get("no_fake")) and bool(audit.get("no_proxy")) and int(_safe_float(audit.get("cpu_offload_used"), 0.0)) == 0
    if not no_contract_issue:
        route = "R10-CodeOrContractFail"
        blocker = "contract_or_provenance_audit_failed"
        next_impl = "remove_fake_proxy_or_cpu_offload_before_running_any_gate"
    elif not p0_gate:
        if math.isfinite(mean_delta) and mean_delta > 0.0 and math.isfinite(mean_curv) and mean_curv <= 0.90 and math.isfinite(q90_step) and q90_step > 1.50:
            route = "R5-TimingProtocolSensitiveSuccess" if not p1_sensitivity else "R4-ConfigSensitiveSuccess"
            blocker = "wallclock_step_ratio_stability_fail"
            if timing_summary and not robust_timing_full:
                if int(_safe_float(timing_summary.get("full_t0_t4_complete"), 0.0) == 1.0):
                    next_impl = "repair_manual_train_step_timing_before_adaptive_controller"
                else:
                    next_impl = "measure_full_T0_T4_protocols_or_repair_timing_before_adaptive_controller"
            elif p4_robust_selection_failed:
                next_impl = "repair_manual_train_step_timing_no_hidden_passes_robust_architecture_selection"
            elif p4_base_impl_selection_failed:
                next_impl = "repair_manual_train_step_timing_no_base_implementation_variant_passes_robust_screen"
            elif selected_confirmation_pass:
                next_impl = "run_p0_stability_for_selected_kw4_foreach_method"
            elif p4_selection:
                next_impl = "confirm_selected_system_candidate_task_geometry_before_adaptive_controller"
            else:
                next_impl = "implement_robust_timing_protocol_and_adaptive_event_controller"
        else:
            route = "R9-ReproductionFail"
            blocker = "v85_accepted_route_not_reproduced_under_p0_first_wave"
            next_impl = "debug_v85_fresh_reproduction_before_adaptive_changes"
    elif not effective_full_p0:
        route = "R4-ConfigSensitiveSuccess"
        blocker = "p0_full_stability_protocol_not_complete"
        next_impl = "run_remaining_seeds_reruns_and_timing_protocols_before_claiming_stability"
    else:
        route = "R4-ConfigSensitiveSuccess"
        if p2_adaptive_one_step and p3_adaptive_multistep:
            if p4_selection and selected_confirmation_pass:
                if p5_robust_timing and p6_training_compute:
                    if p7_cross_task_external and p9_cross_task_causality:
                        route = "R1-ExternalBoundarySelected"
                        if not p7_formal_protocol:
                            blocker = "cross_task_external_pass_measured_under_screen_protocol_not_formal_protocol"
                            next_impl = "rerun_p7_p9_at_full_formal_protocol_or_record_boundary"
                        elif not p5_time_accounting:
                            blocker = "selected_functional_phase_time_accounting_incomplete"
                            next_impl = "measure_selected_functional_phase_breakdown_before_formal_claim"
                        else:
                            blocker = "none"
                            next_impl = "optional_run_p10_p13_boundary_and_ablation_waves"
                    elif p7_summary or p9_summary:
                        route = "R2-StableComputeSelected"
                        blocker = "cross_task_external_measured_joint_or_causality_gate_failed"
                        next_impl = "analyze_p7_p9_failure_boundary_without_changing_loss_teacher_sampler_or_offload"
                    else:
                        route = "R2-StableComputeSelected"
                        blocker = "stable_compute_selected_cross_task_external_waves_not_run"
                        next_impl = "run_p7_p9_cross_task_external_transfer_and_causality_waves"
                else:
                    route = "R3-NoManualArchitectureSelected"
                    blocker = "no_manual_architecture_selected_p5_p6_or_cross_task_waves_not_run"
                    next_impl = "run_p5_p6_timing_compute_then_p7_p9_cross_task_external_waves"
            elif p4_selection:
                blocker = "p4_architecture_selection_pass_selected_task_geometry_confirmation_not_run"
                next_impl = "confirm_p4_selected_architecture_without_manual_hidden_or_candidate_choice"
            elif p4_robust_selection_failed or p4_base_impl_selection_failed:
                blocker = "p4_no_manual_architecture_selection_failed"
                next_impl = "repair_system_side_base_architecture_screen_before_cross_task_waves"
            else:
                blocker = "adaptive_multistep_pass_no_manual_tuning_and_cross_task_waves_not_run"
                next_impl = "run_no_manual_tuning_architecture_selection_and_cross_task_external_waves"
        elif p2_adaptive_one_step:
            if p3_summary:
                blocker = "adaptive_multistep_gate_failed"
                next_impl = "repair_adaptive_multistep_task_geometry_tradeoff_before_no_manual_cross_task_waves"
            else:
                blocker = "adaptive_one_step_pass_multistep_and_cross_task_waves_not_run"
                next_impl = "run_adaptive_multistep_confirmation_then_no_manual_tuning_cross_task_waves"
        else:
            blocker = "adaptive_controller_and_cross_task_waves_not_run"
            next_impl = "implement_pre_registered_adaptive_event_role_budget_controller"
    return {
        "route": route,
        "success_v87_stability": int(p0_gate and effective_full_p0 and robust_timing_full),
        "success_v87_adaptive": int(p0_gate and effective_full_p0 and robust_timing_full and p2_adaptive_one_step and p3_adaptive_multistep),
        "success_v87_external_fair": int(
            p0_gate
            and effective_full_p0
            and robust_timing_full
            and p2_adaptive_one_step
            and p3_adaptive_multistep
            and p4_selection
            and selected_confirmation_pass
            and p5_robust_timing
            and p6_training_compute
            and p7_cross_task_external
            and p9_cross_task_causality
        ),
        "success_v87_no_manual_tuning": int(
            p0_gate
            and effective_full_p0
            and robust_timing_full
            and p2_adaptive_one_step
            and p3_adaptive_multistep
            and p4_selection
            and selected_confirmation_pass
        ),
        "success_v87_formal": int(
            p0_gate
            and effective_full_p0
            and robust_timing_full
            and p2_adaptive_one_step
            and p3_adaptive_multistep
            and p4_selection
            and selected_confirmation_pass
            and p5_robust_timing
            and p5_time_accounting
            and p6_training_compute
            and p7_cross_task_external
            and p9_cross_task_causality
            and p7_formal_protocol
        ),
        "p0_stability_gate_pass": p0_gate,
        "p0_full_protocol_complete": effective_full_p0,
        "p0_full_protocol_complete_raw_summary": full_p0,
        "primary_blocker": blocker,
        "next_required_implementation": next_impl,
        "dg1_mean_delta_vs_KB_MLP": summary.get("dg1_mean_delta_vs_KB_MLP", METRIC_UNAVAILABLE),
        "dg1_mean_curvature_ratio_vs_DG0": summary.get("dg1_mean_curvature_ratio_vs_DG0", METRIC_UNAVAILABLE),
        "dg1_q90_step_ratio_vs_KB_MLP": summary.get("dg1_q90_step_ratio_vs_KB_MLP", METRIC_UNAVAILABLE),
        "dg1_all_gate_pass_rate": summary.get("dg1_all_gate_pass_rate", METRIC_UNAVAILABLE),
        "p1_fixed_schedule_sensitivity_pass": p1_sensitivity,
        "p1_step_ratio_range": (p1_summary or {}).get("step_ratio_range", METRIC_UNAVAILABLE),
        "p1_test_acc_range_percentage_points": (p1_summary or {}).get("test_acc_range_percentage_points", METRIC_UNAVAILABLE),
        "p2_adaptive_one_step_pass": p2_adaptive_one_step,
        "p2_best_adaptive_controller_id": (adaptive_summary or {}).get("best_adaptive_controller_id", METRIC_UNAVAILABLE),
        "p2_best_adaptive_bad_step_rate": (adaptive_summary or {}).get("best_adaptive_bad_step_rate", METRIC_UNAVAILABLE),
        "p2_best_adaptive_curvature_reduction": (adaptive_summary or {}).get("best_adaptive_curvature_reduction", METRIC_UNAVAILABLE),
        "p2_fixed_bad_step_rate": (adaptive_summary or {}).get("fixed_bad_step_rate", METRIC_UNAVAILABLE),
        "p2_fixed_curvature_reduction": (adaptive_summary or {}).get("fixed_curvature_reduction", METRIC_UNAVAILABLE),
        "p3_adaptive_multistep_pass": p3_adaptive_multistep,
        "p3_checkpoint_pass_count": (p3_summary or {}).get("p3_checkpoint_pass_count", METRIC_UNAVAILABLE),
        "p3_checkpoint_count": (p3_summary or {}).get("p3_checkpoint_count", METRIC_UNAVAILABLE),
        "p3_checkpoints": (p3_summary or {}).get("checkpoints", METRIC_UNAVAILABLE),
        "p4_auto_hidden_pilot_pass": p4_selection,
        "p4_selected_hidden_dim": (p4_summary or {}).get("selected_hidden_dim", METRIC_UNAVAILABLE),
        "p4_selected_base_candidate_id": (p4_summary or {}).get("selected_base_candidate_id", METRIC_UNAVAILABLE),
        "p4_selected_step_ratio_vs_KB_MLP": (p4_summary or {}).get("selected_step_ratio_vs_KB_MLP", METRIC_UNAVAILABLE),
        "p4_selected_task_geometry_confirmation_pass": selected_confirmation_pass,
        "p4_selected_task_geometry_step_ratio_vs_KB_MLP": (selected_summary or {}).get("step_ratio_vs_KB_MLP", METRIC_UNAVAILABLE),
        "p4_selected_task_geometry_delta_vs_KB_MLP": (selected_summary or {}).get("DG_functional_delta_vs_KB_MLP", METRIC_UNAVAILABLE),
        "p4_selected_task_geometry_curvature_ratio": (selected_summary or {}).get("curvature_ratio_vs_DG_base", METRIC_UNAVAILABLE),
        "p5_robust_timing_pass": p5_robust_timing,
        "p5_strict_timing_pass": p5_strict_timing,
        "p5_q90_step_ratio_vs_KB_MLP": (p5_summary or {}).get("q90_step_ratio_vs_KB_MLP", METRIC_UNAVAILABLE),
        "p5_max_step_ratio_vs_KB_MLP": (p5_summary or {}).get("max_step_ratio_vs_KB_MLP", METRIC_UNAVAILABLE),
        "p5_time_accounting_pass": (p5_summary or {}).get("time_accounting_pass", METRIC_UNAVAILABLE),
        "p6_training_compute_fair_pass": p6_training_compute,
        "p6_forward_FLOPs_ratio_vs_KB_MLP": (p6_summary or {}).get("forward_FLOPs_ratio_vs_KB_MLP", METRIC_UNAVAILABLE),
        "p6_backward_FLOPs_estimate_ratio_vs_KB_MLP": (p6_summary or {}).get("backward_FLOPs_estimate_ratio_vs_KB_MLP", METRIC_UNAVAILABLE),
        "p6_step_time_ratio_vs_KB_MLP": (p6_summary or {}).get("step_time_ratio_vs_KB_MLP", METRIC_UNAVAILABLE),
        "p7_cross_task_external_pass": p7_cross_task_external,
        "p7_task_count": (p7_summary or {}).get("task_count", METRIC_UNAVAILABLE),
        "p7_joint_fair_pass_count": (p7_summary or {}).get("joint_fair_pass_count", METRIC_UNAVAILABLE),
        "p7_joint_fair_pass_tasks": (p7_summary or {}).get("joint_fair_pass_tasks", METRIC_UNAVAILABLE),
        "p7_formal_protocol": (p7_summary or {}).get("formal_protocol", METRIC_UNAVAILABLE),
        "p9_cross_task_causality_pass": p9_cross_task_causality,
        "p9_causality_pass_count": (p9_summary or {}).get("causality_pass_count", METRIC_UNAVAILABLE),
        "p9_causality_pass_tasks": (p9_summary or {}).get("causality_pass_tasks", METRIC_UNAVAILABLE),
        "robust_timing_partial_gate_pass": robust_timing_partial,
        "robust_timing_full_gate_pass": robust_timing_full,
        "robust_timing_q90_step_ratio_vs_KB_MLP": timing_q90 if math.isfinite(timing_q90) else METRIC_UNAVAILABLE,
        "cpu_offload_used": audit.get("cpu_offload_used", 0),
        "no_fake": audit.get("no_fake", False),
        "no_proxy": audit.get("no_proxy", False),
    }


def _hash_manifest(out_dir: Path, decision: Dict[str, Any]) -> None:
    rows: List[Dict[str, Any]] = []
    for rel in [
        "experiments/run_gafu_v87_real.py",
        PLAN_PATH,
        str(out_dir / "route_decision.json"),
        str(out_dir / "v85_fresh_reproduction_stability.csv"),
        str(out_dir / "p0_stability_summary.csv"),
        str(out_dir / "adaptive_functional_one_step.csv"),
        str(out_dir / "adaptive_functional_one_step_raw.csv"),
        str(out_dir / "adaptive_functional_one_step_summary.csv"),
        str(out_dir / "adaptive_functional_multistep.csv"),
        str(out_dir / "adaptive_functional_multistep_raw.csv"),
        str(out_dir / "adaptive_functional_multistep_summary.csv"),
        str(out_dir / "adaptive_functional_multistep_route_summary.csv"),
        str(out_dir / "auto_architecture_selection.csv"),
        str(out_dir / "auto_architecture_selection_by_hidden.csv"),
        str(out_dir / "auto_architecture_selection_summary.csv"),
        str(out_dir / "base_implementation_screen.csv"),
        str(out_dir / "base_implementation_screen_by_candidate.csv"),
        str(out_dir / "base_implementation_screen_summary.csv"),
        str(out_dir / "selected_system_task_geometry_confirmation.csv"),
        str(out_dir / "selected_system_child_route_decision.json"),
        str(out_dir / "robust_timing_protocols.csv"),
        str(out_dir / "robust_timing_protocols_summary.csv"),
        str(out_dir / "training_compute_counter.csv"),
        str(out_dir / "kanbefair_broad_reproduction.csv"),
        str(out_dir / "kanbefair_multitask_transfer.csv"),
        str(out_dir / "kanbefair_multitask_transfer_summary.csv"),
        str(out_dir / "joint_fair_envelope.csv"),
        str(out_dir / "functional_causality_multitask.csv"),
        str(out_dir / "functional_causality_multitask_summary.csv"),
        str(out_dir / "robust_timing_summary.csv"),
        str(out_dir / "v87_provenance_audit.csv"),
    ]:
        path = Path(rel)
        if not path.is_absolute():
            path = ROOT_DIR / path
        if path.exists():
            rows.append({"artifact": rel, "sha256": _sha256(path)})
    write_csv(out_dir / "artifact_hashes.csv", rows)
    manifest = {
        "stage": "RUN_MANIFEST_V87",
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "plan_path": PLAN_PATH,
        "script_path": "experiments/run_gafu_v87_real.py",
        "route": decision.get("route"),
        "success_v87_formal": decision.get("success_v87_formal"),
        "no_fake": decision.get("no_fake"),
        "no_proxy": decision.get("no_proxy"),
    }
    save_json(out_dir / "run_manifest.json", manifest)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    save_json(out_dir / "run_started.json", {
        "stage": "RUN_STARTED_V87",
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "plan_path": PLAN_PATH,
        "p0_seeds": args.p0_seeds,
        "p0_reruns": int(args.p0_reruns),
        "p0_dg_candidate_id": str(args.p0_dg_candidate_id),
        "p0_dg_hidden_dim": int(args.p0_dg_hidden_dim),
        "p0_use_foreach_adamw_addcdiv": int(bool(args.p0_use_foreach_adamw_addcdiv)),
        "p0_use_fused_ce_head_backward": int(bool(args.p0_use_fused_ce_head_backward)),
        "p0_use_compiled_fused_ce_head_backward": int(bool(args.p0_use_compiled_fused_ce_head_backward)),
        "p0_prewarm_compiled_fused_ce_head_backward": int(bool(args.p0_prewarm_compiled_fused_ce_head_backward)),
        "fixed_ft7_event_stride": int(args.fixed_ft7_event_stride),
        "fixed_ft7_event_alpha_mult": float(args.fixed_ft7_event_alpha_mult),
        "run_adaptive_one_step_audit": int(bool(args.run_adaptive_one_step_audit)),
        "adaptive_dg_candidate_id": str(args.adaptive_dg_candidate_id),
        "adaptive_dg_hidden_dim": int(args.adaptive_dg_hidden_dim),
        "adaptive_seeds": str(args.adaptive_seeds),
        "run_adaptive_multistep_smoke": int(bool(args.run_adaptive_multistep_smoke)),
        "p3_adaptive_controller_id": str(args.p3_adaptive_controller_id),
        "p3_dg_candidate_id": str(args.p3_dg_candidate_id),
        "p3_dg_hidden_dim": int(args.p3_dg_hidden_dim),
        "p3_seeds": str(args.p3_seeds),
        "p3_steps": str(args.p3_steps),
        "p3_adaptive_probe_interval": int(args.p3_adaptive_probe_interval),
        "p3_adaptive_alpha_scale_with_probe_interval": int(bool(args.p3_adaptive_alpha_scale_with_probe_interval)),
        "run_p7_p9_multitask_external": int(bool(args.run_p7_p9_multitask_external)),
        "p7_datasets": str(args.p7_datasets),
        "p7_train_size": int(args.p7_train_size),
        "p7_test_size": int(args.p7_test_size),
        "p7_epochs": int(args.p7_epochs),
        "p7_ft7_event_stride": int(args.p7_ft7_event_stride),
        "p7_ft7_event_alpha_mult": float(args.p7_ft7_event_alpha_mult),
    })

    p0_rows, child_decisions = _run_p0(args, out_dir)
    summary = _summarize_p0(p0_rows, child_decisions, args)
    write_csv(out_dir / "p0_stability_summary.csv", [summary])
    p1_summary: Dict[str, Any] | None = None
    if bool(args.run_p1_sensitivity):
        _p1_rows, p1_summary = _run_p1_sensitivity(args, out_dir)
    adaptive_summary: Dict[str, Any] | None = None
    if bool(args.run_adaptive_one_step_audit):
        _adaptive_rows, adaptive_summary = _run_adaptive_one_step_audit(args, out_dir)
    p3_summary: Dict[str, Any] | None = None
    if bool(args.run_adaptive_multistep_smoke):
        _p3_rows, p3_summary = _run_adaptive_multistep_smoke(args, out_dir)
    p4_summary: Dict[str, Any] | None = None
    if bool(args.run_p4_hidden_pilot):
        _p4_rows, p4_summary = _run_p4_hidden_pilot(args, out_dir)
    if bool(args.run_p4_robust_hidden_selection):
        _p4_rows, p4_summary = _run_p4_robust_hidden_selection(args, out_dir)
    if bool(args.run_p4_base_implementation_screen):
        _p4_rows, p4_summary = _run_p4_base_implementation_screen(args, out_dir)
    selected_summary: Dict[str, Any] | None = None
    if bool(args.run_selected_system_confirmation):
        if p4_summary:
            p4_selected_candidate = str(p4_summary.get("selected_base_candidate_id", "")).strip()
            p4_selected_hidden = _safe_float(p4_summary.get("selected_hidden_dim"))
            if p4_selected_candidate and p4_selected_candidate != METRIC_UNAVAILABLE:
                args.selected_dg_candidate_id = p4_selected_candidate
            if math.isfinite(p4_selected_hidden):
                args.selected_dg_hidden_dim = int(p4_selected_hidden)
        selected_summary = _run_selected_system_task_geometry_confirmation(args, out_dir)
    p5_summary: Dict[str, Any] | None = None
    p6_summary: Dict[str, Any] | None = None
    if bool(args.run_p5_p6_selected_timing_compute):
        p5_summary, p6_summary = _run_p5_p6_selected_timing_compute(args, out_dir, p4_summary, selected_summary)
    p7_summary: Dict[str, Any] | None = None
    p9_summary: Dict[str, Any] | None = None
    if bool(args.run_p7_p9_multitask_external):
        p7_summary, p9_summary = _run_p7_p9_multitask_external(args, out_dir, p4_summary, selected_summary)
    timing_summary: Dict[str, Any] | None = None
    if bool(args.run_robust_timing_from_artifacts):
        _timing_rows, timing_summary = _run_robust_timing_from_artifacts(args, out_dir)
    if bool(args.run_t0_t2_phase_clean_timing):
        _timing_rows, timing_summary = _run_t0_t2_phase_clean_timing(args, out_dir)
    if bool(args.run_t3_phase_clean_timing):
        _timing_rows, timing_summary = _run_t3_phase_clean_timing(args, out_dir)
    _write_not_run_artifacts(out_dir, reason="v87_current_run_did_not_execute_this_wave")
    _write_figure(out_dir, summary)
    audit = _audit(out_dir)
    decision = _route(
        summary,
        audit,
        p1_summary,
        adaptive_summary,
        p3_summary,
        p4_summary,
        timing_summary,
        selected_summary,
        p5_summary,
        p6_summary,
        p7_summary,
        p9_summary,
    )
    save_json(out_dir / "route_decision.json", decision)
    save_json(out_dir / "aggregate_decision.json", decision)
    write_csv(out_dir / "failure_table.csv", [{
        "stage": "ROUTE_DECISION_V87",
        "route": decision["route"],
        "primary_blocker": decision["primary_blocker"],
        "next_required_implementation": decision["next_required_implementation"],
        "success_v87_formal": decision["success_v87_formal"],
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": decision["cpu_offload_used"],
    }])
    _hash_manifest(out_dir, decision)
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DG-KAN v8.7 stability-first functional architecture runner")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--kanbefair-path", default="third_party/KANbeFair")
    parser.add_argument("--p0-seeds", default="1314")
    parser.add_argument("--p0-reruns", type=int, default=1)
    parser.add_argument("--p0-dg-candidate-id", default="KW6")
    parser.add_argument("--p0-dg-hidden-dim", type=int, default=28)
    parser.add_argument("--p0-use-foreach-adamw-addcdiv", action="store_true")
    parser.add_argument("--p0-use-fused-ce-head-backward", action="store_true")
    parser.add_argument("--p0-use-compiled-fused-ce-head-backward", action="store_true")
    parser.add_argument("--p0-prewarm-compiled-fused-ce-head-backward", action="store_true")
    parser.add_argument("--reuse-p0-out-dir", default="")
    parser.add_argument("--run-p1-sensitivity", action="store_true")
    parser.add_argument("--p1-seeds", default="1314")
    parser.add_argument("--p1-strides", default="64,128,256")
    parser.add_argument("--p1-alpha-mults", default="15.0")
    parser.add_argument("--run-adaptive-one-step-audit", action="store_true")
    parser.add_argument("--adaptive-seeds", default="1314,1315,1316")
    parser.add_argument("--adaptive-train-size", type=int, default=4096)
    parser.add_argument("--adaptive-batch-size", type=int, default=128)
    parser.add_argument("--adaptive-warmup-batches", type=int, default=8)
    parser.add_argument("--adaptive-audit-batches", type=int, default=16)
    parser.add_argument("--adaptive-dg-candidate-id", default="KW3")
    parser.add_argument("--adaptive-dg-hidden-dim", type=int, default=28)
    parser.add_argument("--adaptive-basis-count", type=int, default=8)
    parser.add_argument("--adaptive-target-rel-update", type=float, default=1.0e-3)
    parser.add_argument("--run-adaptive-multistep-smoke", action="store_true")
    parser.add_argument("--p3-seeds", default="0,1,2")
    parser.add_argument("--p3-steps", default="20,50,240")
    parser.add_argument("--p3-train-size", type=int, default=60000)
    parser.add_argument("--p3-test-size", type=int, default=10000)
    parser.add_argument("--p3-batch-size", type=int, default=128)
    parser.add_argument("--p3-eval-batch-size", type=int, default=512)
    parser.add_argument("--p3-dg-candidate-id", default="KW3")
    parser.add_argument("--p3-dg-hidden-dim", type=int, default=28)
    parser.add_argument("--p3-basis-count", type=int, default=8)
    parser.add_argument("--p3-adaptive-controller-id", default="Adaptive-FT-D")
    parser.add_argument("--p3-adaptive-warmup-steps", type=int, default=8)
    parser.add_argument("--p3-adaptive-probe-interval", type=int, default=1)
    parser.add_argument("--p3-adaptive-alpha-scale-with-probe-interval", action="store_true")
    parser.add_argument("--p3-adaptive-event-quantile", type=float, default=0.75)
    parser.add_argument("--run-robust-timing-from-artifacts", action="store_true")
    parser.add_argument("--timing-source-out-dirs", default="")
    parser.add_argument("--timing-dg-candidate-id", default="")
    parser.add_argument("--timing-dg-hidden-dim", type=int, default=0)
    parser.add_argument("--run-t0-t2-phase-clean-timing", action="store_true")
    parser.add_argument("--t0-t2-seed", type=int, default=1314)
    parser.add_argument("--t0-t2-batch-size", type=int, default=128)
    parser.add_argument("--t0-t2-protocols", default="T0:50:200,T1:100:500,T2:200:1000")
    parser.add_argument("--run-t3-phase-clean-timing", action="store_true")
    parser.add_argument("--t3-seed", type=int, default=1314)
    parser.add_argument("--t3-warmup", type=int, default=20)
    parser.add_argument("--t3-reps", type=int, default=260)
    parser.add_argument("--t3-batch-size", type=int, default=128)
    parser.add_argument("--run-p4-hidden-pilot", action="store_true")
    parser.add_argument("--run-p4-robust-hidden-selection", action="store_true")
    parser.add_argument("--run-p4-base-implementation-screen", action="store_true")
    parser.add_argument("--run-selected-system-confirmation", action="store_true")
    parser.add_argument("--run-p5-p6-selected-timing-compute", action="store_true")
    parser.add_argument("--run-p7-p9-multitask-external", action="store_true")
    parser.add_argument("--p4-hidden-candidates", default="16,20,24,28,32")
    parser.add_argument("--p4-base-candidates", default="KF10,KW3,KW4,KW5,KW6")
    parser.add_argument("--p4-base-hidden-dim", type=int, default=28)
    parser.add_argument("--p4-base-hidden-candidates", default="")
    parser.add_argument("--p4-seed", type=int, default=1314)
    parser.add_argument("--p4-robust-protocols", default="T0:50:200,T1:100:500,T2:200:1000,T3:20:260")
    parser.add_argument("--p4-base-protocols", default="T0:50:200,T1:100:500,T2:200:1000,T3:20:260")
    parser.add_argument("--selected-dg-candidate-id", default="KW4")
    parser.add_argument("--selected-dg-hidden-dim", type=int, default=28)
    parser.add_argument("--selected-seed", type=int, default=1314)
    parser.add_argument("--selected-use-foreach-adamw-addcdiv", action="store_true")
    parser.add_argument("--selected-use-fused-ce-head-backward", action="store_true")
    parser.add_argument("--selected-use-compiled-fused-ce-head-backward", action="store_true")
    parser.add_argument("--selected-prewarm-compiled-fused-ce-head-backward", action="store_true")
    parser.add_argument("--p7-datasets", default="Fashion-MNIST,KMNIST")
    parser.add_argument("--p7-train-size", type=int, default=10000)
    parser.add_argument("--p7-test-size", type=int, default=2000)
    parser.add_argument("--p7-epochs", type=int, default=5)
    parser.add_argument("--p7-eval-batch-size", type=int, default=512)
    parser.add_argument("--p7-ft7-event-stride", type=int, default=128)
    parser.add_argument("--p7-ft7-event-alpha-mult", type=float, default=15.0)
    parser.add_argument("--p7-dg-stream-batches-from-cpu", action="store_true")
    parser.add_argument("--p7-dg-stream-chunk-batches", type=int, default=1)
    parser.add_argument("--p7-dg-stream-epoch-permute-cpu", action="store_true")
    parser.add_argument("--p7-dg-stream-chunk-order-shuffle", action="store_true")
    parser.add_argument("--kb-epochs", type=int, default=20)
    parser.add_argument("--kb-batch-size", type=int, default=128)
    parser.add_argument("--primary-epochs", type=int, default=20)
    parser.add_argument("--fixed-ft7-event-stride", type=int, default=128)
    parser.add_argument("--fixed-ft7-event-alpha-mult", type=float, default=15.0)
    parser.add_argument("--use-cached-adamw-param-rows", action="store_true")
    parser.add_argument("--use-foreach-adamw-addcdiv", action="store_true")
    parser.add_argument("--profile-dg-subphases", action="store_true")
    parser.add_argument("--use-fused-ce-head-backward", action="store_true")
    parser.add_argument("--use-compiled-fused-ce-head-backward", action="store_true")
    parser.add_argument("--prewarm-compiled-fused-ce-head-backward", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
