#!/usr/bin/env python3
"""DG-KAN v8.1 CE-only teacher-free architecture-matrix runner.

This runner intentionally narrows the v8.0 code-native path.  It reuses the
real measured manual-training candidates from ``run_gafu_v80_real`` but writes
v8.1 artifacts with a stricter official contract:

* no external teacher
* no self teacher
* CE loss only
* no special loss / sampler / class weight / label smoothing
* strict manual PureKAN forward/backward/update for official candidates

Rows that are not implemented or not run are recorded as such and never counted
as pass.  No fake/proxy rows are emitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import torch.nn.functional as F

import run_gafu_v72_real as v72
import run_gafu_v80_real as v80
from dgkan_core import get_device, parse_int_list, parse_str_list, save_json, set_seed, write_csv
from run_gafu_v66_real import _git_commit, _git_status


PLAN_PATH = "docs/DG-KAN_v8.1_CEOnly_TeacherFree_ArchitectureMatrix_完整实验计划.md"
SCRIPT_PATH = "experiments/run_gafu_v81_real.py"
METRIC_UNAVAILABLE = v80.METRIC_UNAVAILABLE


@dataclass
class CandidateSpecV3:
    candidate_id: str
    candidate_name: str
    route_role: str
    architecture_alias: str
    dense_cls_name: str
    model_family: str
    hidden_dim: Any
    basis_count: Any
    depth: Any
    external_teacher_used: int
    external_teacher_logits_used: int
    external_teacher_forward_used: int
    self_teacher_used: int
    self_teacher_logits_used: int
    self_teacher_forward_used: int
    loss_type: str
    special_loss_used: int
    distill_loss_used: int
    margin_loss_used: int
    calibration_loss_used: int
    nll_balanced_loss_used: int
    label_smoothing_used: int
    focal_loss_used: int
    geometry_loss_used: int
    sampler_changed: int
    class_weight_used: int
    optimizer_family: str
    optimizer_hparams_locked: int
    expected_nonkan_count: Any
    expected_manual_forward: int
    expected_manual_backward: int
    expected_manual_update: int
    expected_uses_loss_backward: int
    expected_uses_torch_autograd_graph: int
    strict_expected: int
    no_teacher_no_loss_expected: int
    architecture_only_eligible: int
    official_eligible: int
    implementation_status: str
    kernel_path: str


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "" or str(value).lower() in {"nan", "none", "metric_unavailable", "not_run", "not_implemented"}:
            return default
        return float(value)
    except Exception:
        return default


def _mean(values: Iterable[Any], default: float = float("nan")) -> float:
    vals = [_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else default


def _is_one(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def _hash_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _base_spec_from_v80() -> Dict[str, Any]:
    return v80._spec_v2_registry(argparse.Namespace(hidden_dim=64, basis_count=8))


def _spec_v3_registry(args: argparse.Namespace) -> Dict[str, CandidateSpecV3]:
    v80_specs = _base_spec_from_v80()

    def mk(
        cid: str,
        *,
        alias: str,
        role: str = "official_architecture",
        dense: str | None = None,
        family: str | None = None,
        implemented: str | None = None,
        architecture_only: int = 1,
        external_teacher: int = 0,
        self_teacher: int = 0,
        distill_loss: int = 0,
        special_loss: int = 0,
        uses_loss_backward: int = 0,
        uses_autograd: int = 0,
        manual: int = 1,
        nonkan: Any = 0,
        kernel_path: str = "manual_python",
    ) -> CandidateSpecV3:
        old = v80_specs.get(cid)
        name = old.candidate_name if old else cid
        impl = implemented if implemented is not None else (old.implementation_status if old else "not_implemented")
        dense_name = dense if dense is not None else (old.dense_cls_name if old else "metric_unavailable")
        fam = family if family is not None else (old.model_family if old else "purekan_manual")
        no_teacher_no_loss = int(
            external_teacher == 0
            and self_teacher == 0
            and distill_loss == 0
            and special_loss == 0
        )
        strict_expected = int(manual == 1 and str(nonkan) in {"0", "0.0"} and uses_loss_backward == 0)
        official = int(architecture_only and no_teacher_no_loss and strict_expected and impl == "implemented")
        return CandidateSpecV3(
            candidate_id=cid,
            candidate_name=name,
            route_role=role,
            architecture_alias=alias,
            dense_cls_name=dense_name,
            model_family=fam,
            hidden_dim=getattr(args, "hidden_dim", 64),
            basis_count=getattr(args, "basis_count", 8),
            depth=old.depth if old else "runtime_spec",
            external_teacher_used=external_teacher,
            external_teacher_logits_used=external_teacher,
            external_teacher_forward_used=external_teacher,
            self_teacher_used=self_teacher,
            self_teacher_logits_used=self_teacher,
            self_teacher_forward_used=self_teacher,
            loss_type="CE",
            special_loss_used=special_loss,
            distill_loss_used=distill_loss,
            margin_loss_used=0,
            calibration_loss_used=0,
            nll_balanced_loss_used=0,
            label_smoothing_used=0,
            focal_loss_used=0,
            geometry_loss_used=0,
            sampler_changed=0,
            class_weight_used=0,
            optimizer_family="manual_adamw_family" if manual else "torch_adamw",
            optimizer_hparams_locked=1,
            expected_nonkan_count=nonkan,
            expected_manual_forward=manual,
            expected_manual_backward=manual,
            expected_manual_update=manual,
            expected_uses_loss_backward=uses_loss_backward,
            expected_uses_torch_autograd_graph=uses_autograd,
            strict_expected=strict_expected,
            no_teacher_no_loss_expected=no_teacher_no_loss,
            architecture_only_eligible=architecture_only,
            official_eligible=official,
            implementation_status=impl,
            kernel_path=kernel_path,
        )

    rows: Dict[str, CandidateSpecV3] = {
        "B0": mk(
            "B0",
            alias="B0",
            role="baseline_mlp",
            family="mlp_reference",
            architecture_only=0,
            manual=0,
            nonkan=METRIC_UNAVAILABLE,
            uses_loss_backward=1,
            uses_autograd=1,
            kernel_path="torch_autograd_mlp",
        ),
        "B2": mk("B2", alias="EXCLUDED_B2", role="excluded_external_teacher_mlp", architecture_only=0, external_teacher=1, distill_loss=1, manual=0, nonkan=METRIC_UNAVAILABLE, uses_loss_backward=1, uses_autograd=1),
        "M12": mk("M12", alias="EXCLUDED_M12", role="excluded_external_teacher_purekan", architecture_only=0, external_teacher=1, distill_loss=1),
        "M13": mk("M13", alias="ARCH0", kernel_path="manual_linear_stack_poly2_silu_head"),
        "A2S": mk("A2S", alias="ARCH1", kernel_path="manual_cached_linear_stack_poly2_silu_head"),
        "M9": mk("M9", alias="ARCH2", kernel_path="manual_cached_linear_stack_poly2_silu_head"),
        "RR3": mk("RR3", alias="ARCH5", kernel_path="manual_dwm2lite_rbf_residual"),
        "RR8": mk("RR8", alias="ARCH5_STREAM", kernel_path="manual_dwm2lite_rbf_streaming"),
        "RR9": mk("RR9", alias="ARCH5_BASIS2", kernel_path="manual_dwm2lite_rbf_basis2_unrolled"),
        "RR10": mk("RR10", alias="ARCH5_BASIS2_RECOMPUTE", kernel_path="manual_dwm2lite_rbf_basis2_recompute_z"),
        "RR11": mk("RR11", alias="ARCH5_BASIS2_COMPILED", kernel_path="torch_compile_dwm2lite_basis2"),
        "RR12": mk("RR12", alias="ARCH5_BASIS2_COMPILED_FORWARD", kernel_path="torch_compile_forward_dwm2lite_basis2"),
        "RR13": mk("RR13", alias="ARCH5_BASIS2_TRITON_FORWARD", kernel_path="triton_forward_transform"),
        "RR14": mk("RR14", alias="ARCH5_BASIS2_TRITON_BACKWARD", kernel_path="triton_backward_transform"),
        "KC1": mk("KC1", alias="ARCH_SYS1", role="official_kernel_native_s2_repair", kernel_path="manual_recompute_cache_trim"),
        "KC2": mk("KC2", alias="ARCH_SYS2", role="official_kernel_native_s2_repair", kernel_path="manual_recompute_cache_trim_adamw_v_bfloat"),
        "KC3": mk("KC3", alias="ARCH_SYS3", role="official_kernel_native_s2_repair", kernel_path="manual_packed_recompute_adamw_v_bfloat"),
        "KC4": mk("KC4", alias="ARCH_SYS4", role="official_kernel_native_s2_repair", kernel_path="manual_packed_cache_y_adamw_v_bfloat"),
        "KC5": mk("KC5", alias="ARCH_SYS5", role="official_kernel_native_s2_repair", kernel_path="manual_packed_prefix_recompute_adamw_v_bfloat"),
        "KC6": mk("KC6", alias="ARCH_SYS6", role="official_kernel_native_s2_repair", kernel_path="manual_packed_prefix_recompute"),
        "KC7": mk("KC7", alias="ARCH_SYS7", role="official_kernel_native_s2_repair", kernel_path="manual_packed_prefix_recompute_adamw_addcdiv"),
        "KC8": mk("KC8", alias="ARCH_SYS8", role="official_kernel_native_s2_repair", kernel_path="manual_packed_prefix_recompute_adamw_addcdiv_head_cache_release"),
        "KF1": mk("KF1", alias="ARCH_FUSED1", role="official_kernel_native_fused_backward", kernel_path="triton_inplace_silu_backward"),
        "KF2": mk("KF2", alias="ARCH_FUSED2", role="official_kernel_native_fused_backward", kernel_path="triton_inplace_silu_backward_adamw_addcdiv"),
        "KF3": mk("KF3", alias="ARCH_FUSED3", role="official_kernel_native_fused_backward", kernel_path="aten_silu_backward"),
        "KF4": mk("KF4", alias="ARCH_FUSED4", role="official_kernel_native_fused_backward", kernel_path="aten_silu_backward_adamw_addcdiv"),
        "KF5": mk("KF5", alias="ARCH_FUSED5", role="official_kernel_native_fused_backward", kernel_path="explicit_inplace_silu_backward"),
        "KF6": mk("KF6", alias="ARCH_FUSED6", role="official_kernel_native_fused_backward", kernel_path="explicit_inplace_silu_backward_adamw_addcdiv"),
        "KF7": mk("KF7", alias="ARCH_FUSED7", role="official_kernel_native_fused_backward", kernel_path="aten_upper_only_silu_backward"),
        "KF8": mk("KF8", alias="ARCH_FUSED8", role="official_kernel_native_fused_backward", kernel_path="aten_upper_only_silu_backward_adamw_addcdiv"),
        "KF9": mk("KF9", alias="ARCH_FUSED9", role="official_kernel_native_fused_backward", kernel_path="aten_lower_only_silu_backward"),
        "KF10": mk("KF10", alias="ARCH_FUSED10", role="official_kernel_native_fused_backward", kernel_path="aten_lower_only_silu_backward_adamw_addcdiv"),
        "KW1": mk("KW1", alias="ARCH_WORKSPACE1", role="official_stack_workspace_trim", kernel_path="no_input_grad_recompute"),
        "KW2": mk("KW2", alias="ARCH_WORKSPACE2", role="official_stack_workspace_trim", kernel_path="no_input_grad_recompute_adamw_addcdiv"),
        "KW3": mk("KW3", alias="ARCH_WORKSPACE3", role="official_stack_workspace_trim", kernel_path="no_input_grad_aten_lower_only_adamw_addcdiv"),
        "KW4": mk("KW4", alias="ARCH_WORKSPACE4", role="official_stack_step_repair", kernel_path="cached_no_input_grad_aten_lower_only_adamw_addcdiv"),
        "KW5": mk("KW5", alias="ARCH_WORKSPACE5", role="official_stack_step_repair", kernel_path="compiled_explicit_no_input_grad_aten_lower_only_adamw_addcdiv"),
        "KW6": mk("KW6", alias="ARCH_WORKSPACE6", role="official_stack_step_repair", kernel_path="fast_mix_view_no_input_grad_aten_lower_only_adamw_addcdiv"),
        "TF7": mk("TF7", alias="EXCLUDED_TF7", role="excluded_training_recipe_swa", architecture_only=0, special_loss=0),
        "TF8": mk("TF8", alias="EXCLUDED_TF8", role="excluded_training_recipe_ema", architecture_only=0, special_loss=0),
    }
    for cid in ["ARCH3", "ARCH4", "ARCH6", "ARCH7", "ARCH8", "ARCH9", "ARCH10", "ARCH11"]:
        rows[cid] = mk(
            cid,
            alias=cid,
            role="planned_architecture_not_implemented",
            dense=cid.lower(),
            family="planned_core_primitive",
            implemented="not_implemented",
            architecture_only=1,
            kernel_path="not_implemented",
        )
    return rows


def _eff_summary(eff_rows: List[Dict[str, Any]], cid: str) -> Dict[str, Any]:
    items = [r for r in eff_rows if str(r.get("candidate_id")) == cid]
    mem = [_float(r.get("memory_ratio")) for r in items if math.isfinite(_float(r.get("memory_ratio")))]
    step = [_float(r.get("step_ratio")) for r in items if math.isfinite(_float(r.get("step_ratio")))]
    datasets = sorted({str(r.get("dataset")) for r in items if r.get("dataset") not in (None, "")})
    batches = sorted({int(_float(r.get("batch_size"), -1)) for r in items if _float(r.get("batch_size"), -1) > 0})
    fullgrid = int(set(datasets) >= {"MNIST", "Fashion-MNIST", "KMNIST"} and {128, 256, 512}.issubset(set(batches)))
    return {
        "eff_shape_count": len(items),
        "fullgrid_shape_complete": fullgrid,
        "memory_ratio_mean": _mean(mem),
        "memory_ratio_max": max(mem) if mem else METRIC_UNAVAILABLE,
        "step_ratio_mean": _mean(step),
        "step_ratio_max": max(step) if step else METRIC_UNAVAILABLE,
        "forward_ratio_mean": _mean(r.get("forward_ratio") for r in items),
        "backward_ratio_mean": _mean(r.get("backward_ratio") for r in items),
        "S2_shape_count": sum(1 for r in items if _float(r.get("memory_ratio"), 99) <= 1.05 and _float(r.get("step_ratio"), 99) <= 1.50),
        "S1_shape_count": sum(1 for r in items if _float(r.get("memory_ratio"), 99) < 1.00 and _float(r.get("step_ratio"), 99) <= 1.35),
        "FullGridS2Pass": int(bool(items) and fullgrid and all(_float(r.get("memory_ratio"), 99) <= 1.05 and _float(r.get("step_ratio"), 99) <= 1.50 for r in items)),
        "FullGridS1Pass": int(bool(items) and fullgrid and all(_float(r.get("memory_ratio"), 99) < 1.00 and _float(r.get("step_ratio"), 99) <= 1.35 for r in items)),
    }


def _grad_summary(grad_rows: List[Dict[str, Any]], cid: str) -> Dict[str, Any]:
    items = [r for r in grad_rows if str(r.get("candidate_id")) == cid]
    rels = [_float(r.get("grad_relerr_max")) for r in items if math.isfinite(_float(r.get("grad_relerr_max")))]
    coss = [_float(r.get("grad_cos_min")) for r in items if math.isfinite(_float(r.get("grad_cos_min")))]
    return {
        "GradPass": int(bool(items) and all(_is_one(r.get("grad_pass")) for r in items)),
        "grad_relerr_max": max(rels) if rels else METRIC_UNAVAILABLE,
        "grad_cos_min": min(coss) if coss else METRIC_UNAVAILABLE,
        "grad_rows": len(items),
    }


def _time_auc_summary(out_dir: Path, task_rows: List[Dict[str, Any]], cid: str) -> Dict[str, Any]:
    p7 = _read_csv_rows(out_dir / "p7_time_auc_v76.csv")
    source = p7 if p7 else task_rows
    c_step = _mean(r.get("val_loss_auc_step") for r in source if str(r.get("candidate_id")) == cid)
    b_step = _mean(r.get("val_loss_auc_step") for r in source if str(r.get("candidate_id")) == "B0")
    c_time = _mean(r.get("val_loss_auc_time") for r in source if str(r.get("candidate_id")) == cid)
    b_time = _mean(r.get("val_loss_auc_time") for r in source if str(r.get("candidate_id")) == "B0")
    return {
        "ValLossAUC_step": c_step if math.isfinite(c_step) else METRIC_UNAVAILABLE,
        "ValLossAUC_time": c_time if math.isfinite(c_time) else METRIC_UNAVAILABLE,
        "ValLossAUC_step_ratio_vs_B0": c_step / b_step if math.isfinite(c_step) and math.isfinite(b_step) and b_step else METRIC_UNAVAILABLE,
        "ValLossAUC_time_ratio_vs_B0": c_time / b_time if math.isfinite(c_time) and math.isfinite(b_time) and b_time else METRIC_UNAVAILABLE,
        "TimeAUCPass": int(math.isfinite(c_time) and math.isfinite(b_time) and c_time <= b_time),
    }


def _current_allocated_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return float("nan")
    return float(torch.cuda.memory_allocated(device) / (1024**2))


def _current_reserved_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return float("nan")
    return float(torch.cuda.memory_reserved(device) / (1024**2))


def _tensor_mb(t: torch.Tensor) -> float:
    return float(t.numel() * t.element_size() / (1024**2))


def _optimizer_state_mb(opt: Any) -> float:
    total = 0.0
    for attr in ("m", "v"):
        states = getattr(opt, attr, {})
        if isinstance(states, dict):
            for tensor in states.values():
                if isinstance(tensor, torch.Tensor):
                    total += _tensor_mb(tensor)
    return total


def _measure_phase(device: torch.device, fn: Any) -> Dict[str, Any]:
    v72.v71._sync(device)
    start_alloc = _current_allocated_mb(device)
    start_reserved = _current_reserved_mb(device)
    v72.v71._reset_peak(device)
    t0 = time.perf_counter()
    result = fn()
    v72.v71._sync(device)
    t1 = time.perf_counter()
    return {
        "result": result,
        "time_ms": (t1 - t0) * 1000.0,
        "start_allocated_MB": start_alloc,
        "start_reserved_MB": start_reserved,
        "peak_allocated_MB": v72.v71._peak_allocated_mb(device),
        "peak_reserved_MB": v72.v71._peak_reserved_mb(device),
        "end_allocated_MB": _current_allocated_mb(device),
        "end_reserved_MB": _current_reserved_mb(device),
    }


def _manual_live_allocator_row(args: argparse.Namespace, cid: str, dataset: str, batch_size: int) -> Dict[str, Any]:
    device = get_device(args.device)
    if device.type != "cuda":
        return {
            "stage": "P11_GPU_LIVE_ALLOCATOR_V81",
            "candidate_id": cid,
            "dataset": dataset,
            "batch_size": batch_size,
            "status": "cuda_unavailable",
            "fake_data_used": 0,
            "proxy_row_used": 0,
        }
    spec = v80._spec_map_v80().get(cid)
    if spec is None or cid == "B0":
        return _mlp_live_allocator_row(args, dataset, batch_size)

    bundle = v72.v71.load_vision_bundle(
        dataset,
        data_root=Path(args.data_root),
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=0,
        allow_fake_data=False,
    )
    x = bundle.x_train[:batch_size].to(device)
    y = bundle.y_train[:batch_size].to(device)
    set_seed(v72._stable_seed("v81-gpu-live", dataset, batch_size, cid))
    stack, head = v80._make_manual_candidate_v80(spec, bundle.input_dim, bundle.num_classes, args.hidden_dim, args.basis_count, device)
    params = v80.V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    opt = v72.v71.FastAdamW([stack, head], lr=params.lr_manual * spec.lr_mult, weight_decay=args.weight_decay)
    opt_state_mb = _optimizer_state_mb(opt)

    holders: Dict[str, Any] = {}
    fwd = _measure_phase(device, lambda: _manual_forward_for_live(stack, head, x, y, spec.label_smoothing))
    holders.update(fwd["result"])
    head_bwd = _measure_phase(device, lambda: head.backward_manual(holders["grad_logits"], holders["head_cache"]))
    holders["dh"] = head_bwd["result"]
    stack_bwd = _measure_phase(device, lambda: stack.backward_manual(holders["dh"], holders["caches"]))
    update = _measure_phase(device, lambda: opt.step(1, 1, warmup_cosine=False))
    cache_details = dict(stack.cache_breakdown(holders["caches"])) if hasattr(stack, "cache_breakdown") else {}
    top_phase, top_peak = max(
        [
            ("forward", fwd["peak_allocated_MB"]),
            ("head_backward", head_bwd["peak_allocated_MB"]),
            ("stack_backward", stack_bwd["peak_allocated_MB"]),
            ("update", update["peak_allocated_MB"]),
        ],
        key=lambda item: _float(item[1], -1.0),
    )
    return {
        "stage": "P11_GPU_LIVE_ALLOCATOR_V81",
        "candidate_id": cid,
        "candidate_name": getattr(spec, "candidate_name", cid),
        "dataset": dataset,
        "batch_size": batch_size,
        "status": "measured",
        "device": str(device),
        "cpu_offload_used": 0,
        "triton_silu_backward_available": int(bool(getattr(v80, "_TRITON_AVAILABLE", False))),
        "triton_silu_backward_calls": int(getattr(stack, "triton_silu_backward_calls", 0)),
        "triton_silu_backward_fallbacks": int(getattr(stack, "triton_silu_backward_fallbacks", 0)),
        "aten_silu_backward_calls": int(getattr(stack, "aten_silu_backward_calls", 0)),
        "explicit_inplace_silu_backward_calls": int(getattr(stack, "explicit_inplace_silu_backward_calls", 0)),
        "compiled_explicit_silu_backward_available": int(bool(getattr(v80, "_EXPLICIT_SILU_BACKWARD_COMPILED_AVAILABLE", False))),
        "compiled_explicit_silu_backward_calls": int(getattr(stack, "compiled_explicit_silu_backward_calls", 0)),
        "compiled_explicit_silu_backward_fallbacks": int(getattr(stack, "compiled_explicit_silu_backward_fallbacks", 0)),
        "compiled_explicit_silu_backward_global_failures": int(getattr(v80, "_EXPLICIT_SILU_BACKWARD_COMPILED_FAILURES", 0)),
        "forward_peak_allocated_MB": fwd["peak_allocated_MB"],
        "head_backward_peak_allocated_MB": head_bwd["peak_allocated_MB"],
        "stack_backward_peak_allocated_MB": stack_bwd["peak_allocated_MB"],
        "update_peak_allocated_MB": update["peak_allocated_MB"],
        "forward_start_allocated_MB": fwd["start_allocated_MB"],
        "head_backward_start_allocated_MB": head_bwd["start_allocated_MB"],
        "stack_backward_start_allocated_MB": stack_bwd["start_allocated_MB"],
        "update_start_allocated_MB": update["start_allocated_MB"],
        "top_peak_phase": top_phase,
        "top_peak_allocated_MB": top_peak,
        "forward_time_ms": fwd["time_ms"],
        "head_backward_time_ms": head_bwd["time_ms"],
        "stack_backward_time_ms": stack_bwd["time_ms"],
        "update_time_ms": update["time_ms"],
        "root_input_cache_MB": cache_details.get("cache_x_MB", METRIC_UNAVAILABLE),
        "manual_cache_MB": cache_details.get("manual_cache_MB_measured", cache_details.get("cache_total_MB", METRIC_UNAVAILABLE)),
        "hidden_y_cache_MB": cache_details.get("hidden_y_cache_MB", METRIC_UNAVAILABLE),
        "optimizer_state_MB": opt_state_mb,
        "top1_memory_source_measured": cache_details.get("top1_memory_source_measured", METRIC_UNAVAILABLE),
        "top2_memory_source_measured": cache_details.get("top2_memory_source_measured", METRIC_UNAVAILABLE),
        "top3_memory_source_measured": cache_details.get("top3_memory_source_measured", METRIC_UNAVAILABLE),
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }


def _manual_forward_for_live(stack: Any, head: Any, x: torch.Tensor, y: torch.Tensor, label_smoothing: float) -> Dict[str, Any]:
    h, caches = stack.forward_manual(x)
    logits, head_cache = head.forward_manual(h)
    loss, grad_logits = v72._weighted_smooth_ce_and_grad(logits, y, label_smoothing, None)
    return {
        "h": h,
        "caches": caches,
        "logits": logits,
        "head_cache": head_cache,
        "loss": loss,
        "grad_logits": grad_logits,
    }


def _mlp_live_allocator_row(args: argparse.Namespace, dataset: str, batch_size: int) -> Dict[str, Any]:
    device = get_device(args.device)
    if device.type != "cuda":
        return {
            "stage": "P11_GPU_LIVE_ALLOCATOR_V81",
            "candidate_id": "B0",
            "dataset": dataset,
            "batch_size": batch_size,
            "status": "cuda_unavailable",
            "fake_data_used": 0,
            "proxy_row_used": 0,
        }
    bundle = v72.v71.load_vision_bundle(
        dataset,
        data_root=Path(args.data_root),
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=0,
        allow_fake_data=False,
    )
    x = bundle.x_train[:batch_size].to(device)
    y = bundle.y_train[:batch_size].to(device)
    set_seed(v72.v71._stable_seed("v81-gpu-live", dataset, batch_size, "mlp"))
    model = v72.v71._make_mlp(bundle.input_dim, bundle.num_classes, args.hidden_dim, 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=args.weight_decay)
    holders: Dict[str, Any] = {}
    fwd = _measure_phase(device, lambda: _mlp_forward_for_live(model, x, y))
    holders.update(fwd["result"])
    bwd = _measure_phase(device, lambda: holders["loss"].backward())
    update = _measure_phase(device, lambda: opt.step())
    top_phase, top_peak = max(
        [
            ("forward", fwd["peak_allocated_MB"]),
            ("backward", bwd["peak_allocated_MB"]),
            ("update", update["peak_allocated_MB"]),
        ],
        key=lambda item: _float(item[1], -1.0),
    )
    return {
        "stage": "P11_GPU_LIVE_ALLOCATOR_V81",
        "candidate_id": "B0",
        "candidate_name": "MLP-AdamW-reference",
        "dataset": dataset,
        "batch_size": batch_size,
        "status": "measured",
        "device": str(device),
        "cpu_offload_used": 0,
        "forward_peak_allocated_MB": fwd["peak_allocated_MB"],
        "head_backward_peak_allocated_MB": 0.0,
        "stack_backward_peak_allocated_MB": bwd["peak_allocated_MB"],
        "update_peak_allocated_MB": update["peak_allocated_MB"],
        "forward_start_allocated_MB": fwd["start_allocated_MB"],
        "head_backward_start_allocated_MB": 0.0,
        "stack_backward_start_allocated_MB": bwd["start_allocated_MB"],
        "update_start_allocated_MB": update["start_allocated_MB"],
        "top_peak_phase": top_phase,
        "top_peak_allocated_MB": top_peak,
        "forward_time_ms": fwd["time_ms"],
        "head_backward_time_ms": 0.0,
        "stack_backward_time_ms": bwd["time_ms"],
        "update_time_ms": update["time_ms"],
        "root_input_cache_MB": 0.0,
        "manual_cache_MB": 0.0,
        "hidden_y_cache_MB": 0.0,
        "optimizer_state_MB": sum(_tensor_mb(t) for group in opt.state.values() for t in group.values() if isinstance(t, torch.Tensor)),
        "top1_memory_source_measured": "torch_autograd_graph_or_optimizer_state",
        "top2_memory_source_measured": "metric_unavailable",
        "top3_memory_source_measured": "metric_unavailable",
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }


def _mlp_forward_for_live(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
    opt_logits = model(x)
    loss = F.cross_entropy(opt_logits, y)
    return {"logits": opt_logits, "loss": loss}


def _write_gpu_live_allocator_v81(out_dir: Path, args: argparse.Namespace, measured_ids: List[str]) -> None:
    supported = {"B0", "KC6", "KC7", "KF1", "KF2", "KF3", "KF4", "KF5", "KF6", "KF7", "KF8", "KF9", "KF10", "KW1", "KW2", "KW3", "KW4", "KW5", "KW6"}
    ids = [cid for cid in parse_str_list(args.candidates) if cid in supported and cid in measured_ids]
    rows: List[Dict[str, Any]] = []
    for cid in ids:
        for dataset in parse_str_list(args.datasets):
            for batch_size in parse_int_list(args.bench_batch_sizes):
                try:
                    rows.append(_manual_live_allocator_row(args, cid, dataset, int(batch_size)))
                except Exception as exc:
                    rows.append({
                        "stage": "P11_GPU_LIVE_ALLOCATOR_V81",
                        "candidate_id": cid,
                        "dataset": dataset,
                        "batch_size": batch_size,
                        "status": "measurement_error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:240],
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                    })
    if not rows:
        rows.append({
            "stage": "P11_GPU_LIVE_ALLOCATOR_V81",
            "status": "not_run",
            "reason": "no_supported_gpu_live_allocator_candidate_in_measured_set",
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "p11_gpu_live_allocator_v81.csv", rows)


def _write_v81_postprocess(out_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    specs = _spec_v3_registry(args)
    task_rows = _read_csv_rows(out_dir / "p9_task_summary.csv")
    macro_rows = _read_csv_rows(out_dir / "p1_significance_audit.csv")
    grad_rows = _read_csv_rows(out_dir / "p2_full_gradient_correctness.csv")
    eff_rows = _read_csv_rows(out_dir / "p10_efficiency_profiler.csv")
    measured_ids = sorted({str(r.get("candidate_id")) for r in task_rows})
    task_by = {str(r.get("candidate_id")): r for r in task_rows}
    macro_by = {str(r.get("candidate_id")): r for r in macro_rows if str(r.get("dataset")) == "macro"}

    registry_rows = []
    for cid, spec in specs.items():
        registry_rows.append({
            **asdict(spec),
            "stage": "P0_CANDIDATE_SPEC_V3",
            "measured_in_run": int(cid in measured_ids),
            "constructed_from_spec": int(spec.implementation_status == "implemented"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "candidate_registry_v3.csv", registry_rows)
    write_csv(out_dir / "implementation_status_registry_v81.csv", registry_rows)

    factory_rows = []
    contract_rows = []
    teacher_rows = []
    loss_rows = []
    strict_rows = []
    score_rows = []
    for cid in measured_ids:
        spec = specs.get(cid)
        task = task_by.get(cid, {})
        macro = macro_by.get(cid, {})
        grad = _grad_summary(grad_rows, cid)
        eff = _eff_summary(eff_rows, cid)
        time_auc = _time_auc_summary(out_dir, task_rows, cid)
        metadata_present = int(spec is not None)
        no_teacher_no_loss = int(bool(spec and spec.no_teacher_no_loss_expected and spec.official_eligible))
        strict = int(
            bool(spec and spec.official_eligible)
            and _is_one(task.get("head_is_kan"))
            and str(task.get("non_kan_trainable_param_count")) in {"0", "0.0"}
            and _is_one(task.get("manual_forward"))
            and _is_one(task.get("manual_backward"))
            and _is_one(task.get("manual_update"))
            and spec.expected_uses_loss_backward == 0
        )
        macro_gap = _float(macro.get("mean_val_gap"), 0.0)
        ci95_low = macro.get("ci95_low", macro.get("bootstrap_ci95_low", METRIC_UNAVAILABLE))
        ci95_high = macro.get("ci95_high", macro.get("bootstrap_ci95_high", METRIC_UNAVAILABLE))
        holm_p = macro.get("holm_p", macro.get("holm_corrected_p", METRIC_UNAVAILABLE))
        test_gap = macro.get("mean_test_gap", METRIC_UNAVAILABLE)
        teacher_free_ce_macro_pass = int(
            bool(spec and spec.official_eligible)
            and macro_gap >= 0.0200
            and _float(ci95_low, -1.0) > 0.0
            and _float(holm_p, 1.0) < 0.05
            and _float(test_gap, 0.0) >= 0.015
        )
        row = {
            "stage": "P6_OFFICIAL_SCORE_INPUT_V81",
            "candidate_id": cid,
            "candidate_name": task.get("candidate_name", spec.candidate_name if spec else cid),
            "architecture_alias": spec.architecture_alias if spec else METRIC_UNAVAILABLE,
            "route_role": spec.route_role if spec else "missing_metadata",
            "implementation_status": spec.implementation_status if spec else "missing_metadata",
            "metadata_present": metadata_present,
            "constructed_from_spec": int(bool(spec and spec.implementation_status == "implemented")),
            "external_teacher_used": spec.external_teacher_used if spec else METRIC_UNAVAILABLE,
            "self_teacher_used": spec.self_teacher_used if spec else METRIC_UNAVAILABLE,
            "loss_type": spec.loss_type if spec else METRIC_UNAVAILABLE,
            "special_loss_used": spec.special_loss_used if spec else METRIC_UNAVAILABLE,
            "distill_loss_used": spec.distill_loss_used if spec else METRIC_UNAVAILABLE,
            "label_smoothing_used": spec.label_smoothing_used if spec else METRIC_UNAVAILABLE,
            "sampler_changed": spec.sampler_changed if spec else METRIC_UNAVAILABLE,
            "class_weight_used": spec.class_weight_used if spec else METRIC_UNAVAILABLE,
            "official_eligible": spec.official_eligible if spec else 0,
            "NoTeacherNoLossModificationPass": no_teacher_no_loss,
            "StrictPass": strict,
            **grad,
            "TeacherFreeCEMacroPass": teacher_free_ce_macro_pass,
            "macro_gap": macro.get("mean_val_gap", 0.0),
            "ci95_low": ci95_low,
            "ci95_high": ci95_high,
            "Holm_p": holm_p,
            "test_gap": test_gap,
            "ECE_delta": macro.get("ECE_delta", macro.get("ECE_delta_macro", METRIC_UNAVAILABLE)),
            "NLL_delta": macro.get("NLL_delta", macro.get("NLL_delta_macro", METRIC_UNAVAILABLE)),
            "val_acc_mean": task.get("val_acc_mean", METRIC_UNAVAILABLE),
            "test_acc_mean": task.get("test_acc_mean", METRIC_UNAVAILABLE),
            "ECE": task.get("ECE_mean", METRIC_UNAVAILABLE),
            "NLL": task.get("NLL_mean", METRIC_UNAVAILABLE),
            **eff,
            **time_auc,
            "ProfilerPass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        }
        score_rows.append(row)
        factory_rows.append({
            "stage": "P0_CANDIDATE_FACTORY_AUDIT_V81",
            "candidate_id": cid,
            "metadata_present": metadata_present,
            "constructed_from_spec": row["constructed_from_spec"],
            "dense_cls_name": spec.dense_cls_name if spec else METRIC_UNAVAILABLE,
            "implementation_status": row["implementation_status"],
            "measured_task_rows": task.get("rows", 0),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
        teacher_rows.append({
            "stage": "P0_TEACHER_CONTRACT_NO_TEACHER",
            "candidate_id": cid,
            "external_teacher_used": row["external_teacher_used"],
            "external_teacher_logits_used": spec.external_teacher_logits_used if spec else METRIC_UNAVAILABLE,
            "external_teacher_forward_used": spec.external_teacher_forward_used if spec else METRIC_UNAVAILABLE,
            "self_teacher_used": row["self_teacher_used"],
            "self_teacher_logits_used": spec.self_teacher_logits_used if spec else METRIC_UNAVAILABLE,
            "self_teacher_forward_used": spec.self_teacher_forward_used if spec else METRIC_UNAVAILABLE,
            "official_eligible": row["official_eligible"],
            "teacher_contract_pass": int((not spec or not spec.official_eligible) or (spec.external_teacher_used == 0 and spec.self_teacher_used == 0)),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
        loss_rows.append({
            "stage": "P0_LOSS_CONTRACT_CE_ONLY",
            "candidate_id": cid,
            "loss_type": row["loss_type"],
            "special_loss_used": row["special_loss_used"],
            "distill_loss_used": row["distill_loss_used"],
            "margin_loss_used": spec.margin_loss_used if spec else METRIC_UNAVAILABLE,
            "calibration_loss_used": spec.calibration_loss_used if spec else METRIC_UNAVAILABLE,
            "nll_balanced_loss_used": spec.nll_balanced_loss_used if spec else METRIC_UNAVAILABLE,
            "label_smoothing_used": row["label_smoothing_used"],
            "focal_loss_used": spec.focal_loss_used if spec else METRIC_UNAVAILABLE,
            "geometry_loss_used": spec.geometry_loss_used if spec else METRIC_UNAVAILABLE,
            "sampler_changed": row["sampler_changed"],
            "class_weight_used": row["class_weight_used"],
            "official_eligible": row["official_eligible"],
            "loss_contract_pass": int((not spec or not spec.official_eligible) or spec.no_teacher_no_loss_expected == 1),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
        strict_rows.append({
            "stage": "P0_STRICT_PUREKAN_CONTRACT",
            "candidate_id": cid,
            "manual_forward": task.get("manual_forward", METRIC_UNAVAILABLE),
            "manual_backward": task.get("manual_backward", METRIC_UNAVAILABLE),
            "manual_update": task.get("manual_update", METRIC_UNAVAILABLE),
            "uses_loss_backward": spec.expected_uses_loss_backward if spec else METRIC_UNAVAILABLE,
            "uses_torch_autograd_graph": spec.expected_uses_torch_autograd_graph if spec else METRIC_UNAVAILABLE,
            "nonKAN_param_count": task.get("non_kan_trainable_param_count", METRIC_UNAVAILABLE),
            "StrictPass": strict,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
        contract_rows.append({
            "stage": "P0_CONTRACT_VALIDATOR_V81",
            "candidate_id": cid,
            "NoTeacherNoLossModificationPass": no_teacher_no_loss,
            "StrictPass": strict,
            "GradPass": grad["GradPass"],
            "official_eligible": row["official_eligible"],
            "contract_pass_for_official_route": int(no_teacher_no_loss and strict and grad["GradPass"]),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })

    write_csv(out_dir / "candidate_factory_audit_v81.csv", factory_rows)
    write_csv(out_dir / "teacher_contract_no_teacher.csv", teacher_rows)
    write_csv(out_dir / "loss_contract_ce_only.csv", loss_rows)
    write_csv(out_dir / "strict_purekan_contract.csv", strict_rows)
    write_csv(out_dir / "contract_validator_v81.csv", contract_rows)
    join_coverage = sum(1 for row in factory_rows if row["metadata_present"]) / max(1, len(factory_rows))
    write_csv(out_dir / "artifact_join_coverage_v81.csv", [{
        "stage": "P0_ARTIFACT_JOIN_COVERAGE_V81",
        "measured_candidate_count": len(measured_ids),
        "registry_joined_count": sum(1 for row in factory_rows if row["metadata_present"]),
        "join_coverage": join_coverage,
        "candidate_factory_pass": int(join_coverage == 1.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])

    p1 = [row for row in score_rows if row["candidate_id"] in {"B0", "M13", "A2S", "M9"}]
    p3 = []
    for cid, spec in specs.items():
        if spec.architecture_alias.startswith("ARCH") and spec.architecture_alias not in {"ARCH0", "ARCH1", "ARCH2"}:
            measured = next((row for row in score_rows if row["candidate_id"] == cid), None)
            if measured:
                p3.append(measured | {"stage": "P3_ARCHITECTURE_IMPLEMENTATION_MATRIX"})
            else:
                p3.append({
                    "stage": "P3_ARCHITECTURE_IMPLEMENTATION_MATRIX",
                    "candidate_id": cid,
                    "architecture_alias": spec.architecture_alias,
                    "implementation_status": spec.implementation_status,
                    "forward_smoke_pass": "not_run" if spec.implementation_status == "implemented" else "not_implemented",
                    "backward_smoke_pass": "not_run" if spec.implementation_status == "implemented" else "not_implemented",
                    "GradPass": "not_run" if spec.implementation_status == "implemented" else "not_implemented",
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                })
    p4 = [row | {"stage": "P4_ARCHITECTURE_TASK_MATRIX"} for row in score_rows if row["candidate_id"] not in {"B0", "B2", "M12"}]
    write_csv(out_dir / "p1_ce_only_reproduction.csv", p1)
    write_csv(out_dir / "p2_validation_robustness.csv", [{
        "stage": "P2_VALIDATION_ROBUSTNESS",
        "status": "not_run",
        "reason": "validation_shard_runner_not_implemented_in_this_execution_slice",
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])
    write_csv(out_dir / "p3_architecture_implementation_matrix.csv", p3)
    write_csv(out_dir / "p4_architecture_task_matrix.csv", p4)
    write_csv(out_dir / "p5_representation_hardsample_attribution.csv", [{
        "stage": "P5_REPRESENTATION_HARDSAMPLE_ATTRIBUTION",
        "status": "not_run",
        "reason": "feature_rank_margin_hardsample_metrics_not_implemented_in_this_execution_slice",
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])
    write_csv(out_dir / "p6_official_coselection.csv", score_rows)
    write_csv(out_dir / "p7_phase_clean_fullgrid.csv", [{
        "stage": "P7_PHASE_CLEAN_FULLGRID",
        "status": "not_run",
        "reason": "phase_clean_train_step_only_profiler_not_implemented",
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])
    write_csv(out_dir / "p8_time_accounting.csv", [{
        "stage": "P8_TIME_ACCOUNTING",
        "status": "not_run",
        "reason": "time_accounting_not_implemented",
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])
    write_csv(out_dir / "p9_phase_mapped_kernel_profiler.csv", [{
        "stage": "P9_PHASE_MAPPED_KERNEL_PROFILER",
        "status": "not_run",
        "reason": "phase_mapped_kernel_profiler_not_implemented",
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])
    write_csv(out_dir / "p10_s1_memory_system_package.csv", [{
        "stage": "P10_S1_MEMORY_SYSTEM_PACKAGE",
        "status": "not_run",
        "reason": "no_CE_only_teacher_free_macro_S2_candidate_selected_for_system_repair",
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])
    _write_gpu_live_allocator_v81(out_dir, args, measured_ids)

    official = [row for row in score_rows if _is_one(row.get("official_eligible")) and _is_one(row.get("StrictPass")) and _is_one(row.get("GradPass"))]
    official.sort(key=lambda row: (
        int(_is_one(row.get("TeacherFreeCEMacroPass"))),
        int(_is_one(row.get("FullGridS2Pass"))),
        _float(row.get("macro_gap"), -99.0),
    ), reverse=True)
    best = official[0] if official else {}
    code_native_pass = int(join_coverage == 1.0 and all(row["constructed_from_spec"] for row in factory_rows))
    no_teacher_no_loss_pass = int(bool(best) and _is_one(best.get("NoTeacherNoLossModificationPass")))
    strict_pass = int(bool(best) and _is_one(best.get("StrictPass")))
    grad_pass = int(bool(best) and _is_one(best.get("GradPass")))
    macro_pass = int(bool(best) and _is_one(best.get("TeacherFreeCEMacroPass")))
    s2_pass = int(bool(best) and _is_one(best.get("FullGridS2Pass")))
    s1_pass = int(bool(best) and _is_one(best.get("FullGridS1Pass")))
    time_auc_pass = int(bool(best) and _is_one(best.get("TimeAUCPass")))
    profiler_pass = 0
    if macro_pass and s1_pass and time_auc_pass and profiler_pass:
        route = "R1-CEOnlyTeacherFreeFullSystemAdvantage"
    elif macro_pass and s2_pass and time_auc_pass:
        route = "R2-CEOnlyTeacherFreeS2TimeAUC"
    elif macro_pass and s2_pass:
        route = "R3-CEOnlyTeacherFreeS2TimeAUCFail"
    elif macro_pass:
        route = "R4-CEOnlyTeacherFreeMacroButS2Fail"
    elif any(_float(row.get("macro_gap"), 0.0) >= 0.0195 for row in official):
        route = "R5-CEOnlyTeacherFreeNearPass"
    else:
        route = "R7-NoCEOnlyArchitectureReproduction"
    success_min = int(code_native_pass and no_teacher_no_loss_pass and strict_pass and grad_pass and macro_pass and s2_pass)
    success_formal = int(success_min and s1_pass and time_auc_pass and profiler_pass)
    route_json = {
        "route": route,
        "best_official_candidate_id": best.get("candidate_id", METRIC_UNAVAILABLE),
        "architecture_alias": best.get("architecture_alias", METRIC_UNAVAILABLE),
        "code_native_pass": code_native_pass,
        "no_teacher_no_loss_pass": no_teacher_no_loss_pass,
        "strict_pass": strict_pass,
        "grad_pass": grad_pass,
        "teacher_free_ce_macro_pass": macro_pass,
        "fullgrid_s2_pass": s2_pass,
        "fullgrid_s1_pass": s1_pass,
        "time_auc_pass": time_auc_pass,
        "profiler_pass": profiler_pass,
        "success_v81_minimum": success_min,
        "success_v81_formal": success_formal,
        "macro_gap": best.get("macro_gap", METRIC_UNAVAILABLE),
        "ci95_low": best.get("ci95_low", METRIC_UNAVAILABLE),
        "holm_p": best.get("Holm_p", METRIC_UNAVAILABLE),
        "test_gap": best.get("test_gap", METRIC_UNAVAILABLE),
        "memory_ratio_max": best.get("memory_ratio_max", METRIC_UNAVAILABLE),
        "step_ratio_max": best.get("step_ratio_max", METRIC_UNAVAILABLE),
        "S2_shape_count": best.get("S2_shape_count", METRIC_UNAVAILABLE),
        "S1_shape_count": best.get("S1_shape_count", METRIC_UNAVAILABLE),
        "primary_blocker": (
            "teacher_free_ce_macro_gap_not_closed"
            if not macro_pass
            else ("fullgrid_s2_not_closed" if not s2_pass else "s1_time_auc_profiler_not_closed")
        ),
        "next_required_implementation": (
            "CE_only_architecture_primitive_or_validation_shard_runner"
            if not macro_pass
            else ("S2_system_kernel_repair" if not s2_pass else "S1_TimeAUC_Profiler")
        ),
        "no_fake": True,
        "no_proxy": True,
    }
    save_json(out_dir / "v81_route_decision.json", route_json)
    save_json(out_dir / "aggregate_decision_v81.json", route_json)
    write_csv(out_dir / "failure_table_v81.csv", [
        {"failure_type": f, "count": 1, "fake_data_used": 0, "proxy_row_used": 0}
        for f, flag in [
            ("F1_code_contract_fail", not code_native_pass),
            ("F2_teacher_or_loss_contract_fail", not no_teacher_no_loss_pass),
            ("F3_teacher_free_ce_macro_fail", not macro_pass),
            ("F4_fullgrid_s2_fail", not s2_pass),
            ("F5_time_auc_fail", not time_auc_pass),
            ("F6_profiler_incomplete", not profiler_pass),
        ]
        if flag
    ] or [{"failure_type": "none", "count": 0, "fake_data_used": 0, "proxy_row_used": 0}])

    artifact_paths = [
        out_dir / "candidate_registry_v3.csv",
        out_dir / "candidate_factory_audit_v81.csv",
        out_dir / "teacher_contract_no_teacher.csv",
        out_dir / "loss_contract_ce_only.csv",
        out_dir / "strict_purekan_contract.csv",
        out_dir / "contract_validator_v81.csv",
        out_dir / "artifact_join_coverage_v81.csv",
        out_dir / "p1_ce_only_reproduction.csv",
        out_dir / "p2_validation_robustness.csv",
        out_dir / "p3_architecture_implementation_matrix.csv",
        out_dir / "p4_architecture_task_matrix.csv",
        out_dir / "p5_representation_hardsample_attribution.csv",
        out_dir / "p6_official_coselection.csv",
        out_dir / "p7_phase_clean_fullgrid.csv",
        out_dir / "p8_time_accounting.csv",
        out_dir / "p9_phase_mapped_kernel_profiler.csv",
        out_dir / "p10_s1_memory_system_package.csv",
        out_dir / "p11_gpu_live_allocator_v81.csv",
        out_dir / "v81_route_decision.json",
        out_dir / "failure_table_v81.csv",
        out_dir / "p9_task_summary.csv",
        out_dir / "p1_significance_audit.csv",
        out_dir / "p2_full_gradient_correctness.csv",
        out_dir / "p10_efficiency_profiler.csv",
    ]
    audit = v72._audit_fake_proxy(artifact_paths)
    write_csv(out_dir / "v81_provenance_audit.csv", [{
        **audit,
        "plan_path": PLAN_PATH,
        "script_path": SCRIPT_PATH,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])
    save_json(out_dir / "v81_manifest.json", {
        "plan_path": PLAN_PATH,
        "script_path": SCRIPT_PATH,
        "runner_reuse": "run_gafu_v80_real.py measured path + v8.1 CE-only contract postprocess",
        "out_dir": str(out_dir),
        "source_commit": _git_commit(),
        "git_status": _git_status(),
        "datasets": parse_str_list(args.datasets),
        "seeds": parse_int_list(args.seeds),
        "candidates": parse_str_list(args.candidates),
        "bench_batch_sizes": parse_int_list(args.bench_batch_sizes),
        "grad_batch_sizes": parse_int_list(args.grad_batch_sizes),
        "artifact_hashes": {p.name: _hash_file(p) for p in artifact_paths if p.exists()},
        "route_decision": route_json,
        "provenance_audit": audit,
    })
    return route_json


def run(args: argparse.Namespace) -> None:
    v80.PLAN_PATH = PLAN_PATH
    v80.SCRIPT_PATH = SCRIPT_PATH
    v80.run(args)
    out_dir = Path(args.out_dir)
    _write_v81_postprocess(out_dir, args)
    for src_name, dst_name in {
        "v80_manifest.json": "v81_reused_v80_manifest.json",
        "v80_route_decision.json": "v81_reused_v80_route_decision.json",
        "v80_provenance_audit.csv": "v81_reused_v80_provenance_audit.csv",
    }.items():
        src = out_dir / src_name
        if src.exists():
            shutil.copyfile(src, out_dir / dst_name)


def parse_args() -> argparse.Namespace:
    return v80.parse_args()


if __name__ == "__main__":
    run(parse_args())
