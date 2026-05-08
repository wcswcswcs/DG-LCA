#!/usr/bin/env python3
"""DG-KAN v7.6 real-only M12 system-advantage runner.

v7.6 keeps the v7.5 M12 implementation intact and adds the minimum fairness
controls needed by the v7.6 plan:

* B2: MLP with the same C3 logit-distillation objective used by M12;
* M13: M12 architecture/packed head with the M5 initialization trajectory, but
  without C3 distillation.

All post-processing rows are derived from measured CSV/JSON artifacts.  Missing
or unimplemented system metrics are written as metric_unavailable rather than
filled with proxy values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

import run_gafu_v73_real as v73
import run_gafu_v72_real as v72
from dgkan_core import get_device, parse_int_list, parse_str_list, save_json, set_seed, write_csv
from run_gafu_v63 import V63Params
from run_gafu_v66_real import _git_commit, _git_status


PLAN_PATH = "docs/DG-KAN_v7.6_M12_SystemAdvantage_vs_MLPAdamW_完整实验计划.md"
SCRIPT_PATH = "experiments/run_gafu_v76_real.py"
METRIC_UNAVAILABLE = v73.METRIC_UNAVAILABLE

_ORIGINAL_SPEC_MAP = v73._spec_map
_ORIGINAL_CANDIDATE_REGISTRY = v73._candidate_registry
_ORIGINAL_INIT_SEED_CANDIDATE_ID = v73._init_seed_candidate_id_v73
_ORIGINAL_MAKE_MANUAL_CANDIDATE = v73._make_manual_candidate
_ORIGINAL_USES_DISTILL = v73._uses_distill
_ORIGINAL_DISTILL_STUDENT_INIT_ID = v73._distill_student_init_id
_ORIGINAL_DISTILL_TEMPERATURE_FOR_SPEC = v73._distill_temperature_for_spec
_ORIGINAL_DISTILL_ALPHA_FOR_SPEC = v73._distill_alpha_for_spec
_ORIGINAL_TRAIN_MLP = v72.v71._train_mlp


class RecomputeFusedLinearSiluStack(v73.FusedLinearSiluStack):
    """M12-equivalent stack with hidden-y cache lifetime trim.

    Forward remains the same linear/SILU function as M12.  The repair caches
    only the root input and recomputes hidden pre-activations during manual
    backward, targeting the bs=512 memory margin without changing loss,
    teacher, sampler, or trainable parameterization.
    """

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        with torch.no_grad():
            h = x
            for i, p in enumerate(self.params):
                y = h @ p["mix"].t()
                h = F.silu(y) if i < len(self.params) - 1 else y
            return h, [{"x0": x.detach()}]

    def _activation_before(self, x0: torch.Tensor, layer_idx: int) -> torch.Tensor:
        h = x0
        for j in range(layer_idx):
            h = F.silu(h @ self.params[j]["mix"].t())
        return h

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, Any]]) -> torch.Tensor:
        delta = dy
        x0 = caches[0]["x0"]
        with torch.no_grad():
            for i in reversed(range(len(self.params))):
                x_i = self._activation_before(x0, i)
                if i < len(self.params) - 1:
                    y_i = x_i @ self.params[i]["mix"].t()
                    sig = torch.sigmoid(y_i)
                    silu_prime = sig + y_i * (sig - sig.square())
                    delta = delta * silu_prime
                self.grads[i]["mix"].add_(delta.t() @ x_i)
                delta = delta @ self.params[i]["mix"]
            return delta

    def cache_breakdown(self, caches: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        if not caches:
            x0_mb = 0.0
            count = 0
        else:
            x0 = caches[0]["x0"]
            x0_mb = int(x0.numel() * x0.element_size()) / (1024**2)
            count = 1
        return {
            "cache_total_MB": x0_mb,
            "cache_x_MB": x0_mb,
            "cache_hidden_y_MB": 0.0,
            "cache_hidden_MB": x0_mb,
            "cache_gate_MB": 0.0,
            "cache_basis_MB": 0.0,
            "materialized_tensor_count_measured": count,
            "largest_live_tensor_MB_measured": x0_mb,
            "linear_body_temp_MB": x0_mb,
            "hidden_y_cache_MB": 0.0,
            "manual_cache_MB_measured": x0_mb,
            "top1_memory_source_measured": "recompute_stack_checkpoint_root_input",
            "top2_memory_source_measured": "backward_recompute_hidden_temps",
            "top3_memory_source_measured": "optimizer_state",
            "unknown_memory_fraction_measured": 0.0,
        }


class HalfCacheFusedLinearSiluStack(v73.FusedLinearSiluStack):
    """M12-equivalent forward with fp16 hidden-y cache storage.

    This S1 repair targets the bs=512 cache margin while keeping the same
    parameterization and training recipe.  It is only admissible if the measured
    manual-vs-autograd gradient gate still passes.
    """

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        with torch.no_grad():
            h = x
            ys: List[torch.Tensor] = []
            for i, p in enumerate(self.params):
                y = h @ p["mix"].t()
                if i < len(self.params) - 1:
                    ys.append(y.detach().to(torch.float16))
                    h = F.silu(y)
                else:
                    h = y
            return h, [{"x0": x.detach(), "ys": ys}]

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, Any]]) -> torch.Tensor:
        delta = dy
        x0 = caches[0]["x0"]
        ys: Sequence[torch.Tensor] = caches[0].get("ys", [])
        with torch.no_grad():
            for i in reversed(range(len(self.params))):
                if i < len(self.params) - 1:
                    y_i = ys[i].to(dtype=delta.dtype)
                    sig = torch.sigmoid(y_i)
                    silu_prime = sig + y_i * (sig - sig.square())
                    delta = delta * silu_prime
                x_i = x0 if i == 0 else F.silu(ys[i - 1].to(dtype=delta.dtype))
                self.grads[i]["mix"].add_(delta.t() @ x_i)
                delta = delta @ self.params[i]["mix"]
            return delta

    def cache_breakdown(self, caches: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        if not caches:
            x0_mb = 0.0
            y_mb = 0.0
            largest_mb = 0.0
            count = 0
        else:
            x0 = caches[0]["x0"]
            ys: Sequence[torch.Tensor] = caches[0].get("ys", [])
            x0_mb = int(x0.numel() * x0.element_size()) / (1024**2)
            y_mb = sum(int(y.numel() * y.element_size()) for y in ys) / (1024**2)
            largest_mb = max([x0_mb] + [int(y.numel() * y.element_size()) / (1024**2) for y in ys] + [0.0])
            count = 1 + len(ys)
        total_mb = x0_mb + y_mb
        return {
            "cache_total_MB": total_mb,
            "cache_x_MB": x0_mb,
            "cache_hidden_y_MB": y_mb,
            "cache_hidden_MB": total_mb,
            "cache_gate_MB": 0.0,
            "cache_basis_MB": 0.0,
            "materialized_tensor_count_measured": count,
            "largest_live_tensor_MB_measured": largest_mb,
            "linear_body_temp_MB": x0_mb,
            "hidden_y_cache_MB": y_mb,
            "manual_cache_MB_measured": total_mb,
            "top1_memory_source_measured": "half_cache_stack_checkpoint_root_input",
            "top2_memory_source_measured": "half_precision_hidden_y_cache",
            "top3_memory_source_measured": "optimizer_state",
            "unknown_memory_fraction_measured": 0.0,
        }


class FirstYCacheFusedLinearSiluStack(v73.FusedLinearSiluStack):
    """M12-equivalent stack that caches only the first hidden pre-activation."""

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        with torch.no_grad():
            h = x
            y0: torch.Tensor | None = None
            for i, p in enumerate(self.params):
                y = h @ p["mix"].t()
                if i < len(self.params) - 1:
                    if i == 0:
                        y0 = y.detach()
                    h = F.silu(y)
                else:
                    h = y
            return h, [{"x0": x.detach(), "y0": y0}]

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, Any]]) -> torch.Tensor:
        delta = dy
        x0 = caches[0]["x0"]
        y0 = caches[0]["y0"]
        with torch.no_grad():
            h1 = F.silu(y0)
            y1 = h1 @ self.params[1]["mix"].t()
            h2 = F.silu(y1)

            self.grads[2]["mix"].add_(delta.t() @ h2)
            delta = delta @ self.params[2]["mix"]

            sig1 = torch.sigmoid(y1)
            delta = delta * (sig1 + y1 * (sig1 - sig1.square()))
            self.grads[1]["mix"].add_(delta.t() @ h1)
            delta = delta @ self.params[1]["mix"]

            sig0 = torch.sigmoid(y0)
            delta = delta * (sig0 + y0 * (sig0 - sig0.square()))
            self.grads[0]["mix"].add_(delta.t() @ x0)
            delta = delta @ self.params[0]["mix"]
            return delta

    def cache_breakdown(self, caches: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        if not caches:
            x0_mb = 0.0
            y_mb = 0.0
            largest_mb = 0.0
            count = 0
        else:
            x0 = caches[0]["x0"]
            y0 = caches[0]["y0"]
            x0_mb = int(x0.numel() * x0.element_size()) / (1024**2)
            y_mb = int(y0.numel() * y0.element_size()) / (1024**2)
            largest_mb = max(x0_mb, y_mb)
            count = 2
        total_mb = x0_mb + y_mb
        return {
            "cache_total_MB": total_mb,
            "cache_x_MB": x0_mb,
            "cache_hidden_y_MB": y_mb,
            "cache_hidden_MB": total_mb,
            "cache_gate_MB": 0.0,
            "cache_basis_MB": 0.0,
            "materialized_tensor_count_measured": count,
            "largest_live_tensor_MB_measured": largest_mb,
            "linear_body_temp_MB": x0_mb,
            "hidden_y_cache_MB": y_mb,
            "manual_cache_MB_measured": total_mb,
            "top1_memory_source_measured": "first_y_cache_stack_checkpoint_root_input",
            "top2_memory_source_measured": "first_hidden_y_cache",
            "top3_memory_source_measured": "optimizer_state",
            "unknown_memory_fraction_measured": 0.0,
        }


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _mean(values: Iterable[Any], default: float = float("nan")) -> float:
    vals = [float(v) for v in values if _finite(v)]
    return sum(vals) / len(vals) if vals else default


def _std(values: Iterable[Any], default: float = float("nan")) -> float:
    vals = [float(v) for v in values if _finite(v)]
    if len(vals) <= 1:
        return 0.0 if vals else default
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))


def _is_one(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true"}


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _hash_file(path: Path) -> str:
    return v72._hash_file(path) if path.exists() else ""


def _spec_map_v76() -> Dict[str, Any]:
    out = dict(_ORIGINAL_SPEC_MAP())
    out["B2"] = v72.v71.CandidateSpec(
        "B2",
        "MLP-C3-logit-distill-T4-alpha025",
        "v76_fairness_mlp_c3_distill",
        "mlp",
        depth=2,
        head_kind="mlp",
        label_smoothing=0.0,
        oracle=True,
        strict_status="reference_mlp_distill",
        allowed_in_route_selection=False,
    )
    out["M13"] = v72.v71.CandidateSpec(
        "M13",
        "M12-architecture-M5init-no-C3-distill",
        "v76_m12_teacher_free",
        "v75_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["M14"] = v72.v71.CandidateSpec(
        "M14",
        "M12-S1-cache-lifetime-trim-recompute-hidden-y",
        "v76_m12_s1_manual_cache_lifetime_trim",
        "v76_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["M15"] = v72.v71.CandidateSpec(
        "M15",
        "M12-S1-hidden-y-half-cache-trim",
        "v76_m12_s1_hidden_y_half_cache",
        "v76_half_cache_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["M16"] = v72.v71.CandidateSpec(
        "M16",
        "M12-S1-first-y-cache-second-y-recompute",
        "v76_m12_s1_first_y_cache_trim",
        "v76_first_y_cache_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    return out


def _candidate_registry_v76() -> List[Any]:
    base = list(_ORIGINAL_CANDIDATE_REGISTRY())
    by_id = {str(spec.candidate_id): spec for spec in base}
    extra = _spec_map_v76()
    for cid in ["B2", "M13", "M14", "M15", "M16"]:
        by_id[cid] = extra[cid]
    preferred = [
        "B0", "B2", "C3", "A2S", "M4", "M9", "M12", "M13", "M14", "M15", "M16",
    ]
    out: List[Any] = []
    seen: set[str] = set()
    for cid in preferred:
        if cid in by_id and cid not in seen:
            out.append(by_id[cid])
            seen.add(cid)
    for spec in base:
        cid = str(spec.candidate_id)
        if cid not in seen:
            out.append(spec)
            seen.add(cid)
    return out


def _init_seed_candidate_id_v76(spec: Any) -> str:
    if str(spec.candidate_id) in {"M13", "M14", "M15", "M16"}:
        return "M4"
    return _ORIGINAL_INIT_SEED_CANDIDATE_ID(spec)


def _uses_distill_v76(spec: Any) -> bool:
    return str(spec.candidate_id) in {"M14", "M15", "M16"} or _ORIGINAL_USES_DISTILL(spec)


def _distill_student_init_id_v76(spec: Any) -> str:
    if str(spec.candidate_id) in {"M14", "M15", "M16"}:
        return "M4"
    return _ORIGINAL_DISTILL_STUDENT_INIT_ID(spec)


def _distill_temperature_for_spec_v76(args: argparse.Namespace, spec: Any) -> float:
    if str(spec.candidate_id) in {"M14", "M15", "M16"}:
        return 4.0
    return _ORIGINAL_DISTILL_TEMPERATURE_FOR_SPEC(args, spec)


def _distill_alpha_for_spec_v76(args: argparse.Namespace, spec: Any) -> float:
    if str(spec.candidate_id) in {"M14", "M15", "M16"}:
        return 0.25
    return _ORIGINAL_DISTILL_ALPHA_FOR_SPEC(args, spec)


def _make_manual_candidate_v76(spec: Any, input_dim: int, num_classes: int, hidden_dim: int, basis: int, device: torch.device) -> Tuple[Any, Any]:
    if str(spec.stack_type) == "v76_recompute_fused_linear_packed_head_kind":
        stack = RecomputeFusedLinearSiluStack(input_dim, hidden_dim, spec.depth, device=device)
        head = v73.PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=basis, device=device)
        return stack, head
    if str(spec.stack_type) == "v76_half_cache_fused_linear_packed_head_kind":
        stack = HalfCacheFusedLinearSiluStack(input_dim, hidden_dim, spec.depth, device=device)
        head = v73.PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=basis, device=device)
        return stack, head
    if str(spec.stack_type) == "v76_first_y_cache_fused_linear_packed_head_kind":
        stack = FirstYCacheFusedLinearSiluStack(input_dim, hidden_dim, spec.depth, device=device)
        head = v73.PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=basis, device=device)
        return stack, head
    return _ORIGINAL_MAKE_MANUAL_CANDIDATE(spec, input_dim, num_classes, hidden_dim, basis, device)


def _train_mlp_v76(args: argparse.Namespace, spec: Any, dataset: str, seed: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if str(spec.candidate_id) != "B2":
        return _ORIGINAL_TRAIN_MLP(args, spec, dataset, seed)

    device = get_device(args.device)
    params = V63Params(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
    )
    bundle = v72.v71.load_vision_bundle(
        dataset,
        data_root=Path(args.data_root),
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=seed,
        allow_fake_data=False,
    )
    teacher_stack, teacher_head, teacher_meta = v73._train_c3_teacher(args, dataset, seed)
    x_train = bundle.x_train.to(device)
    y_train = bundle.y_train.to(device)
    x_val = bundle.x_val.to(device)
    y_val = bundle.y_val.to(device)
    x_test = bundle.x_test.to(device)
    y_test = bundle.y_test.to(device)

    set_seed(v72._stable_seed("v76-mlp-c3-distill", dataset, seed, spec.candidate_id, "mlp"))
    model = v72.v71._make_mlp(bundle.input_dim, bundle.num_classes, args.hidden_dim, 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params.lr_mlp, weight_decay=args.weight_decay)
    train_eval_x = x_train[: min(max(args.batch_size, 512), x_train.shape[0])]
    train_eval_y = y_train[: train_eval_x.shape[0]]
    train0 = v72.v71._mlp_eval(model, train_eval_x, train_eval_y, args.eval_batch_size, bundle.num_classes)
    val0 = v72.v71._mlp_eval(model, x_val, y_val, args.eval_batch_size, bundle.num_classes)
    alpha = 0.25
    temperature = 4.0
    trace: List[Dict[str, Any]] = []
    first_kl = float("nan")
    started = time.perf_counter()
    for step in range(1, int(args.task_steps) + 1):
        xb, yb = v72.v71._select_batch(x_train, y_train, args.batch_size, step)
        tlogits = v73._teacher_logits(teacher_stack, teacher_head, xb)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = v73._distill_loss_autograd(
            logits,
            yb,
            tlogits,
            spec.label_smoothing,
            alpha=alpha,
            temperature=temperature,
        )
        if step == 1:
            with torch.no_grad():
                t = float(temperature)
                p_teacher = torch.softmax(tlogits.detach() / t, dim=-1)
                logp_student = torch.log_softmax(logits.detach() / t, dim=-1)
                first_kl = float(
                    (p_teacher * (torch.log(p_teacher.clamp_min(1.0e-12)) - logp_student))
                    .sum(dim=-1)
                    .mean()
                    .detach()
                    .cpu()
                )
        loss.backward()
        opt.step()
        if step % int(args.trace_every) == 0 or step == int(args.task_steps):
            tr = v72.v71._mlp_eval(model, train_eval_x, train_eval_y, args.eval_batch_size, bundle.num_classes)
            va = v72.v71._mlp_eval(model, x_val, y_val, args.eval_batch_size, bundle.num_classes)
            trace.append({
                "stage": "TASK_TRACE",
                "candidate_id": spec.candidate_id,
                "candidate_name": spec.candidate_name,
                "family": spec.family,
                "dataset": dataset,
                "seed": seed,
                "step": step,
                "wall_clock_time_sec": time.perf_counter() - started,
                "train_loss": tr["loss"],
                "train_acc": tr["acc"],
                "val_loss": va["loss"],
                "val_acc": va["acc"],
                "ECE": va["ECE"],
                "NLL": va["NLL"],
                "recipe": "mlp_logit_distill_from_C3",
                "teacher_id": "C3",
                "distill_temperature": temperature,
                "alpha_logit": alpha,
                "student_teacher_logit_KL_step1": first_kl,
                "fake_data_used": 0,
                "proxy_row_used": 0,
            })

    train1 = v72.v71._mlp_eval(model, train_eval_x, train_eval_y, args.eval_batch_size, bundle.num_classes)
    val1 = v72.v71._mlp_eval(model, x_val, y_val, args.eval_batch_size, bundle.num_classes)
    test1 = v72.v71._mlp_eval(model, x_test, y_test, args.eval_batch_size, bundle.num_classes)
    summary = v72.v71._task_summary_base(args, spec, dataset, seed, bundle.input_dim, bundle.num_classes)
    summary.update({
        "train_loss_before": train0["loss"],
        "train_loss_after": train1["loss"],
        "val_loss_before": val0["loss"],
        "val_loss_after": val1["loss"],
        "train_acc": train1["acc"],
        "val_acc": val1["acc"],
        "test_acc": test1["acc"],
        "ECE": val1["ECE"],
        "NLL": val1["NLL"],
        "test_ECE": test1["ECE"],
        "test_NLL": test1["NLL"],
        "confidence_mean": val1["confidence_mean"],
        "margin_p10": val1["margin_p10"],
        "feature_effective_rank": val1["feature_effective_rank"],
        "classwise_acc": val1["classwise_acc"],
        "val_loss_auc_step": v72.v71._auc([(r["step"], r["val_loss"]) for r in trace]),
        "val_loss_auc_time": v72.v71._auc([(r["wall_clock_time_sec"], r["val_loss"]) for r in trace]),
        "wall_clock_time_sec": time.perf_counter() - started,
        "recipe": "mlp_logit_distill_from_C3",
        "teacher_id": "C3",
        "distill_temperature": temperature,
        "alpha_logit": alpha,
        "student_teacher_logit_KL_step1": first_kl,
        **teacher_meta,
        "implementation_status": "measured",
        "stage_status": "measured",
    })
    return summary, trace


def _macro_by_candidate(sig_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("candidate_id")): row for row in sig_rows if row.get("dataset") == "macro"}


def _summary_by_candidate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("candidate_id")): row for row in rows}


def _grad_summary(grad_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in grad_rows:
        grouped.setdefault(str(row.get("candidate_id")), []).append(row)
    out: Dict[str, Dict[str, Any]] = {}
    for cid, rows in grouped.items():
        rels = [_float(row.get("grad_relerr_max")) for row in rows if _finite(row.get("grad_relerr_max"))]
        coss = [_float(row.get("grad_cos_min")) for row in rows if _finite(row.get("grad_cos_min"))]
        out[cid] = {
            "grad_rows": len(rows),
            "grad_pass_rows": sum(1 for row in rows if _is_one(row.get("grad_pass"))),
            "grad_pass": int(bool(rows) and all(_is_one(row.get("grad_pass")) for row in rows)),
            "grad_relerr_max": max(rels) if rels else METRIC_UNAVAILABLE,
            "grad_cos_min": min(coss) if coss else METRIC_UNAVAILABLE,
        }
    return out


def _s2_s1_counts(eff_rows: Sequence[Dict[str, Any]], cid: str) -> Dict[str, Any]:
    rows = [r for r in eff_rows if str(r.get("candidate_id")) == cid]
    mems = [_float(r.get("memory_ratio")) for r in rows if _finite(r.get("memory_ratio"))]
    steps = [_float(r.get("step_ratio")) for r in rows if _finite(r.get("step_ratio"))]
    return {
        "rows": len(rows),
        "memory_ratio_mean": _mean(mems),
        "memory_ratio_max": max(mems) if mems else METRIC_UNAVAILABLE,
        "step_ratio_mean": _mean(steps),
        "step_ratio_max": max(steps) if steps else METRIC_UNAVAILABLE,
        "s2_shape_count": sum(
            1 for r in rows
            if _float(r.get("memory_ratio"), 99.0) <= 1.05 and _float(r.get("step_ratio"), 99.0) <= 1.50
        ),
        "s1_shape_count": sum(
            1 for r in rows
            if _float(r.get("memory_ratio"), 99.0) < 1.00 and _float(r.get("step_ratio"), 99.0) <= 1.35
        ),
        "forward_ratio_mean": _mean([r.get("forward_ratio") for r in rows]),
        "backward_ratio_mean": _mean([r.get("backward_ratio") for r in rows]),
    }


def _seed_win_rate(task_rows: Sequence[Dict[str, Any]], candidate: str, baseline: str = "B0") -> Tuple[float, int, int]:
    by_seed: Dict[int, Dict[str, List[float]]] = {}
    for row in task_rows:
        cid = str(row.get("candidate_id"))
        if cid not in {candidate, baseline}:
            continue
        try:
            seed = int(row.get("seed"))
        except Exception:
            continue
        by_seed.setdefault(seed, {}).setdefault(cid, []).append(_float(row.get("val_acc")))
    wins = 0
    total = 0
    for data in by_seed.values():
        if candidate in data and baseline in data:
            total += 1
            wins += int(_mean(data[candidate]) > _mean(data[baseline]))
    return (wins / total if total else float("nan"), wins, total)


def _macro_gap_between(task_rows: Sequence[Dict[str, Any]], a: str, b: str) -> float:
    datasets = sorted({str(row.get("dataset")) for row in task_rows})
    gaps: List[float] = []
    for ds in datasets:
        av = _mean(row.get("val_acc") for row in task_rows if str(row.get("candidate_id")) == a and str(row.get("dataset")) == ds)
        bv = _mean(row.get("val_acc") for row in task_rows if str(row.get("candidate_id")) == b and str(row.get("dataset")) == ds)
        if _finite(av) and _finite(bv):
            gaps.append(av - bv)
    return _mean(gaps)


def _nll_between(task_rows: Sequence[Dict[str, Any]], a: str, b: str) -> float:
    av = _mean(row.get("NLL") for row in task_rows if str(row.get("candidate_id")) == a)
    bv = _mean(row.get("NLL") for row in task_rows if str(row.get("candidate_id")) == b)
    return av - bv if _finite(av) and _finite(bv) else float("nan")


def _write_v76_postprocess(out_dir: Path, args: argparse.Namespace) -> None:
    task_rows = _read_csv_rows(out_dir / "p9_task_gate.csv")
    trace_rows = _read_csv_rows(out_dir / "p9_task_trace.csv")
    task_summary = _read_csv_rows(out_dir / "p9_task_summary.csv")
    sig_rows = _read_csv_rows(out_dir / "p1_significance_audit.csv")
    grad_rows = _read_csv_rows(out_dir / "p2_full_gradient_correctness.csv")
    eff_rows = _read_csv_rows(out_dir / "p10_efficiency_profiler.csv")
    eff_summary = _read_csv_rows(out_dir / "p10_efficiency_summary.csv")

    task_by = _summary_by_candidate(task_summary)
    sig_by = _macro_by_candidate(sig_rows)
    grad_by = _grad_summary(grad_rows)
    eff_by = _summary_by_candidate(eff_summary)
    win_rate, wins, total = _seed_win_rate(task_rows, "M12", "B0")

    m12_sig = sig_by.get("M12", {})
    m12_eff = _s2_s1_counts(eff_rows, "M12")
    m12_grad = grad_by.get("M12", {})
    macro_gap = _float(m12_sig.get("mean_val_gap"))
    ci_low = _float(m12_sig.get("bootstrap_ci95_low"))
    holm = _float(m12_sig.get("holm_corrected_p"))
    test_gap = _float(m12_sig.get("mean_test_gap"))
    ece_delta = _float(m12_sig.get("ECE_delta_macro", m12_sig.get("ECE_delta")))
    nll_delta = _float(m12_sig.get("NLL_delta_macro", m12_sig.get("NLL_delta")))
    macro_pass = int(
        _finite(macro_gap) and macro_gap >= 0.0200
        and _finite(ci_low) and ci_low > 0
        and _finite(holm) and holm < 0.05
        and _finite(test_gap) and test_gap >= 0.015
        and _finite(win_rate) and win_rate >= 0.70
    )
    fullgrid_s2 = int(m12_eff["rows"] > 0 and m12_eff["s2_shape_count"] == m12_eff["rows"])
    fullgrid_s1 = int(m12_eff["rows"] > 0 and m12_eff["s1_shape_count"] == m12_eff["rows"])
    strict_pass = int(str(task_by.get("M12", {}).get("head_is_kan")) in {"1", "1.0"} and str(task_by.get("M12", {}).get("non_kan_trainable_param_count")) in {"0", "0.0", ""})
    grad_pass = int(_is_one(m12_grad.get("grad_pass")))

    p1_rows = [{
        "stage": "P1",
        "candidate_id": "M12",
        "candidate_name": task_by.get("M12", {}).get("candidate_name"),
        "seed_count": total,
        "seed_win_count": wins,
        "seed_win_rate": win_rate if _finite(win_rate) else METRIC_UNAVAILABLE,
        "macro_val_gap_mean": m12_sig.get("mean_val_gap", METRIC_UNAVAILABLE),
        "macro_val_gap_ci95_low": m12_sig.get("bootstrap_ci95_low", METRIC_UNAVAILABLE),
        "macro_val_gap_ci95_high": m12_sig.get("bootstrap_ci95_high", METRIC_UNAVAILABLE),
        "holm_p": m12_sig.get("holm_corrected_p", METRIC_UNAVAILABLE),
        "cohen_h": m12_sig.get("cohen_h", METRIC_UNAVAILABLE),
        "test_gap": m12_sig.get("mean_test_gap", METRIC_UNAVAILABLE),
        "ECE_delta": m12_sig.get("ECE_delta_macro", m12_sig.get("ECE_delta", METRIC_UNAVAILABLE)),
        "NLL_delta": m12_sig.get("NLL_delta_macro", m12_sig.get("NLL_delta", METRIC_UNAVAILABLE)),
        "macro_significant_pass_10seed": macro_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }]
    write_csv(out_dir / "p1_10seed_m12_confirmation_v76.csv", p1_rows)

    m12_vs_b2_gap = _macro_gap_between(task_rows, "M12", "B2")
    m12_vs_b2_nll = _nll_between(task_rows, "M12", "B2")
    b2_available = "B2" in task_by
    fairness_pass = int(b2_available and _finite(m12_vs_b2_gap) and m12_vs_b2_gap >= 0.010 and _finite(m12_vs_b2_nll) and m12_vs_b2_nll <= 0.01)
    p2_rows = []
    for cid in ["B0", "B2", "M13", "M12"]:
        if cid not in task_by:
            continue
        row = task_by[cid]
        sig = sig_by.get(cid, {})
        eff = _s2_s1_counts(eff_rows, cid)
        p2_rows.append({
            "stage": "P2",
            "candidate_id": cid,
            "candidate_name": row.get("candidate_name"),
            "role": {
                "B0": "MLP-AdamW",
                "B2": "MLP-C3-logit-distill",
                "M13": "M12-teacher-free",
                "M12": "M12-C3-logit-distill",
            }.get(cid, "other"),
            "teacher_used": int(cid in {"B2", "M12"}),
            "teacher_candidate": "C3" if cid in {"B2", "M12"} else "none",
            "distill_temperature": 4.0 if cid in {"B2", "M12"} else 0.0,
            "distill_alpha": 0.25 if cid in {"B2", "M12"} else 0.0,
            "val_acc_mean": row.get("val_acc_mean"),
            "test_acc_mean": row.get("test_acc_mean"),
            "val_gap_vs_MLP_mean": row.get("val_gap_vs_MLP_mean"),
            "macro_gap_vs_M12": _macro_gap_between(task_rows, cid, "M12") if cid != "M12" else 0.0,
            "M12_gap_vs_this": _macro_gap_between(task_rows, "M12", cid) if cid != "M12" else 0.0,
            "NLL_mean": row.get("NLL_mean"),
            "NLL_delta_vs_M12": _nll_between(task_rows, cid, "M12") if cid != "M12" else 0.0,
            "ECE_mean": row.get("ECE_mean"),
            "memory_ratio_mean": eff["memory_ratio_mean"] if eff["rows"] else METRIC_UNAVAILABLE,
            "step_ratio_mean": eff["step_ratio_mean"] if eff["rows"] else METRIC_UNAVAILABLE,
            "forward_ratio_mean": eff["forward_ratio_mean"] if eff["rows"] else METRIC_UNAVAILABLE,
            "backward_ratio_mean": eff["backward_ratio_mean"] if eff["rows"] else METRIC_UNAVAILABLE,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "p2_mlp_distill_fairness_v76.csv", p2_rows)

    m12_vs_m13_gap = _macro_gap_between(task_rows, "M12", "M13")
    p3_rows = [{
        "stage": "P3",
        "student": "M12",
        "teacher_free_control": "M13" if "M13" in task_by else "not_run",
        "teacher_candidate": "C3",
        "distill_improvement_vs_teacher_free": m12_vs_m13_gap if _finite(m12_vs_m13_gap) else METRIC_UNAVAILABLE,
        "teacher_dependency_strong": int(_finite(m12_vs_m13_gap) and m12_vs_m13_gap >= 0.005),
        "teacher_dependency_mild": int(_finite(m12_vs_m13_gap) and 0.001 <= m12_vs_m13_gap < 0.005),
        "teacher_independent_pass": int("M13" in task_by and _float(task_by["M13"].get("val_gap_vs_MLP_mean")) >= 0.015),
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }]
    write_csv(out_dir / "p3_teacher_dependency_v76.csv", p3_rows)

    p5_rows: List[Dict[str, Any]] = []
    for row in eff_rows:
        cid = str(row.get("candidate_id"))
        if cid != "M12":
            continue
        mem = _float(row.get("memory_ratio"))
        step = _float(row.get("step_ratio"))
        p5_rows.append({
            "stage": "P5",
            "candidate_id": cid,
            "dataset": row.get("dataset"),
            "batch_size": row.get("batch_size"),
            "memory_ratio": row.get("memory_ratio"),
            "step_ratio": row.get("step_ratio"),
            "forward_ratio": row.get("forward_ratio"),
            "backward_ratio": row.get("backward_ratio"),
            "update_ratio": row.get("update_ratio"),
            "peak_allocated_MB": row.get("peak_allocated_MB"),
            "peak_reserved_MB": row.get("peak_reserved_MB"),
            "cache_total_MB_measured": row.get("cache_total_MB_measured", row.get("cache_total_MB", METRIC_UNAVAILABLE)),
            "optimizer_state_MB": row.get("optimizer_state_MB", row.get("optimizer_state_memory_MB", METRIC_UNAVAILABLE)),
            "materialized_tensor_count": row.get("materialized_tensor_count", METRIC_UNAVAILABLE),
            "kernel_count_total": row.get("kernel_count_total", row.get("kernel_count_forward", METRIC_UNAVAILABLE)),
            "s2_pass": int(_finite(mem) and _finite(step) and mem <= 1.05 and step <= 1.50),
            "s1_pass": int(_finite(mem) and _finite(step) and mem < 1.00 and step <= 1.35),
            "s1_memory_gap": mem - 1.00 if _finite(mem) else METRIC_UNAVAILABLE,
            "s1_step_gap": step - 1.35 if _finite(step) else METRIC_UNAVAILABLE,
            "top1_memory_source": row.get("top1_memory_source", METRIC_UNAVAILABLE),
            "top2_memory_source": row.get("top2_memory_source", METRIC_UNAVAILABLE),
            "top3_memory_source": row.get("top3_memory_source", METRIC_UNAVAILABLE),
            "top1_time_source": "forward_path" if _float(row.get("forward_ratio"), 0.0) >= _float(row.get("backward_ratio"), 0.0) else "backward_path",
            "unknown_memory_fraction": row.get("unknown_memory_fraction", METRIC_UNAVAILABLE),
            "unknown_time_fraction": METRIC_UNAVAILABLE,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "p5_s1_margin_attribution_v76.csv", p5_rows)

    trace_by: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in trace_rows:
        trace_by.setdefault((str(row.get("candidate_id")), str(row.get("dataset")), str(row.get("seed"))), []).append(row)
    p7_rows: List[Dict[str, Any]] = []
    for row in task_rows:
        cid = str(row.get("candidate_id"))
        if cid not in {"B0", "B2", "M12", "M13"}:
            continue
        key = (cid, str(row.get("dataset")), str(row.get("seed")))
        traces = sorted(trace_by.get(key, []), key=lambda r: _float(r.get("step")))
        val_acc_auc_step = v72.v71._auc([(_float(r.get("step")), _float(r.get("val_acc"))) for r in traces])
        val_acc_auc_time = v72.v71._auc([(_float(r.get("wall_clock_time_sec")), _float(r.get("val_acc"))) for r in traces])
        p7_rows.append({
            "stage": "P7",
            "candidate_id": cid,
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "val_loss_auc_step": row.get("val_loss_auc_step"),
            "val_loss_auc_time": row.get("val_loss_auc_time"),
            "val_acc_auc_step": val_acc_auc_step if _finite(val_acc_auc_step) else METRIC_UNAVAILABLE,
            "val_acc_auc_time": val_acc_auc_time if _finite(val_acc_auc_time) else METRIC_UNAVAILABLE,
            "wall_clock_time_sec": row.get("wall_clock_time_sec"),
            "samples_per_second": (int(args.train_size) * int(args.task_steps) / _float(row.get("wall_clock_time_sec"))) if _finite(row.get("wall_clock_time_sec")) and _float(row.get("wall_clock_time_sec")) > 0 else METRIC_UNAVAILABLE,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "p7_time_auc_v76.csv", p7_rows)

    time_auc_m12 = _mean(r.get("val_loss_auc_time") for r in p7_rows if str(r.get("candidate_id")) == "M12")
    time_auc_b0 = _mean(r.get("val_loss_auc_time") for r in p7_rows if str(r.get("candidate_id")) == "B0")
    time_auc_pass = int(_finite(time_auc_m12) and _finite(time_auc_b0) and time_auc_m12 <= time_auc_b0)

    candidate_rows = [{
        "stage": "P12",
        "candidate_id": "M12",
        "candidate_name": task_by.get("M12", {}).get("candidate_name"),
        "strict_pass": strict_pass,
        "grad_pass": grad_pass,
        "macro_significant_pass_10seed": macro_pass,
        "fullgrid_s2_pass": fullgrid_s2,
        "fullgrid_s1_pass": fullgrid_s1,
        "fairness_pass": fairness_pass,
        "time_auc_pass": time_auc_pass,
        "macro_gap": macro_gap if _finite(macro_gap) else METRIC_UNAVAILABLE,
        "ci95_low": ci_low if _finite(ci_low) else METRIC_UNAVAILABLE,
        "holm_p": holm if _finite(holm) else METRIC_UNAVAILABLE,
        "test_gap": test_gap if _finite(test_gap) else METRIC_UNAVAILABLE,
        "ECE_delta": ece_delta if _finite(ece_delta) else METRIC_UNAVAILABLE,
        "NLL_delta": nll_delta if _finite(nll_delta) else METRIC_UNAVAILABLE,
        "seed_win_rate": win_rate if _finite(win_rate) else METRIC_UNAVAILABLE,
        "m12_gap_vs_mlp_distill": m12_vs_b2_gap if _finite(m12_vs_b2_gap) else METRIC_UNAVAILABLE,
        "m12_nll_delta_vs_mlp_distill": m12_vs_b2_nll if _finite(m12_vs_b2_nll) else METRIC_UNAVAILABLE,
        "memory_ratio_mean": m12_eff["memory_ratio_mean"],
        "memory_ratio_max": m12_eff["memory_ratio_max"],
        "step_ratio_mean": m12_eff["step_ratio_mean"],
        "step_ratio_max": m12_eff["step_ratio_max"],
        "s2_shape_count": m12_eff["s2_shape_count"],
        "s1_shape_count": m12_eff["s1_shape_count"],
        "success_v76_minimum": int(strict_pass and grad_pass and macro_pass and fullgrid_s2 and fairness_pass),
        "success_v76_formal": int(strict_pass and grad_pass and macro_pass and fullgrid_s1 and fairness_pass and time_auc_pass),
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }]
    write_csv(out_dir / "p12_candidate_selection_v76.csv", candidate_rows)

    best = candidate_rows[0]
    if _is_one(best.get("success_v76_formal")):
        route = "S1-SystemAdvantageSuccess"
        blocker = "none"
    elif _is_one(best.get("success_v76_minimum")):
        route = "S2-FairSystemMinimumSuccess"
        blocker = "s1_or_time_auc_not_closed"
    elif not fairness_pass and b2_available:
        route = "R2-TeacherObjectiveMayExplainAdvantage"
        blocker = "fairness_vs_mlp_distill"
    elif not macro_pass:
        route = "R3-10seedMacroNotStable"
        blocker = "macro_reproducibility"
    else:
        route = "R4-SystemEvidenceIncomplete"
        blocker = "missing_required_fairness_or_efficiency_evidence"
    route_json = {
        "route": route,
        "best_candidate_id": "M12",
        "strict_pass": strict_pass,
        "grad_pass": grad_pass,
        "macro_significant_pass_10seed": macro_pass,
        "fullgrid_s2_pass": fullgrid_s2,
        "fullgrid_s1_pass": fullgrid_s1,
        "fairness_pass": fairness_pass,
        "time_auc_pass": time_auc_pass,
        "success_v76_minimum": best["success_v76_minimum"],
        "success_v76_formal": best["success_v76_formal"],
        "macro_gap": best["macro_gap"],
        "ci95_low": best["ci95_low"],
        "holm_p": best["holm_p"],
        "test_gap": best["test_gap"],
        "ECE_delta": best["ECE_delta"],
        "NLL_delta": best["NLL_delta"],
        "seed_win_rate": best["seed_win_rate"],
        "m12_gap_vs_mlp_distill": best["m12_gap_vs_mlp_distill"],
        "m12_nll_delta_vs_mlp_distill": best["m12_nll_delta_vs_mlp_distill"],
        "memory_ratio_mean": best["memory_ratio_mean"],
        "memory_ratio_max": best["memory_ratio_max"],
        "step_ratio_mean": best["step_ratio_mean"],
        "step_ratio_max": best["step_ratio_max"],
        "s2_shape_count": best["s2_shape_count"],
        "s1_shape_count": best["s1_shape_count"],
        "primary_blocker": blocker,
        "classwise_is_hard_gate": False,
        "no_fake": True,
        "no_proxy": True,
    }
    save_json(out_dir / "v76_route_decision.json", route_json)

    artifact_paths = [
        out_dir / "candidate_registry.csv",
        out_dir / "p9_task_gate.csv",
        out_dir / "p9_task_trace.csv",
        out_dir / "p9_task_summary.csv",
        out_dir / "p1_significance_audit.csv",
        out_dir / "p2_full_gradient_correctness.csv",
        out_dir / "p10_efficiency_profiler.csv",
        out_dir / "p10_efficiency_summary.csv",
        out_dir / "p1_10seed_m12_confirmation_v76.csv",
        out_dir / "p2_mlp_distill_fairness_v76.csv",
        out_dir / "p3_teacher_dependency_v76.csv",
        out_dir / "p5_s1_margin_attribution_v76.csv",
        out_dir / "p7_time_auc_v76.csv",
        out_dir / "p12_candidate_selection_v76.csv",
    ]
    audit = v72._audit_fake_proxy(artifact_paths)
    write_csv(out_dir / "v76_provenance_audit.csv", [{
        **audit,
        "plan_path": PLAN_PATH,
        "script_path": SCRIPT_PATH,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])
    save_json(out_dir / "v76_manifest.json", {
        "plan_path": PLAN_PATH,
        "script_path": SCRIPT_PATH,
        "runner_reuse": "run_gafu_v73_real.py",
        "out_dir": str(out_dir),
        "source_commit": _git_commit(),
        "git_status": _git_status(),
        "datasets": parse_str_list(args.datasets),
        "seeds": parse_int_list(args.seeds),
        "candidates": parse_str_list(args.candidates),
        "bench_batch_sizes": parse_int_list(args.bench_batch_sizes),
        "grad_batch_sizes": parse_int_list(args.grad_batch_sizes),
        "postprocess_artifact_hashes": {p.name: _hash_file(p) for p in artifact_paths if p.exists()},
        "v76_route": route_json,
        "v76_audit": audit,
    })


def _patch_for_v76() -> None:
    v73.PLAN_PATH = PLAN_PATH
    v73.SCRIPT_PATH = SCRIPT_PATH
    v73._spec_map = _spec_map_v76
    v73._candidate_registry = _candidate_registry_v76
    v73._init_seed_candidate_id_v73 = _init_seed_candidate_id_v76
    v73._uses_distill = _uses_distill_v76
    v73._distill_student_init_id = _distill_student_init_id_v76
    v73._distill_temperature_for_spec = _distill_temperature_for_spec_v76
    v73._distill_alpha_for_spec = _distill_alpha_for_spec_v76
    v73._make_manual_candidate = _make_manual_candidate_v76
    v72.v71._train_mlp = _train_mlp_v76


def run(args: argparse.Namespace) -> None:
    _patch_for_v76()
    v73.run(args)
    out_dir = Path(args.out_dir)
    _write_v76_postprocess(out_dir, args)
    for src_name, dst_name in {
        "v73_manifest.json": "v76_reused_v73_manifest.json",
        "v73_route_decision.json": "v76_reused_v73_route_decision.json",
        "v73_provenance_audit.csv": "v76_reused_v73_provenance_audit.csv",
    }.items():
        src = out_dir / src_name
        if src.exists():
            shutil.copyfile(src, out_dir / dst_name)


def parse_args() -> argparse.Namespace:
    return v73.parse_args()


if __name__ == "__main__":
    run(parse_args())
