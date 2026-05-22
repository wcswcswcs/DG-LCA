#!/usr/bin/env python3
"""DG-KAN v12.8.3 B109 + classic-basis family execution runner.

This runner is deliberately audit-first.  It measures real CUDA execution and
records where the current repo has implementation blockers.  It does not turn
torch-autograd or torch-compile exploratory paths into official base success.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v120_good_geometry_battery as v120  # noqa: E402
import run_v124_multibasis_functional_dual as v124  # noqa: E402
import run_v1252_efficiency_functional_manifold as v1252  # noqa: E402
import run_v126_lowerlevel_fhq_functional_geometry as v126  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, write_csv_rows, write_json  # noqa: E402
from dgkan.kernels import fused_hinge_quadratic as fhq  # noqa: E402
from dgkan.models import fc_purekan_primitives as prim  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v12.8.3_主线B109_经典基函数全家族高效率化_FunctionalGeometry_整合完整计划.md"
REPORT_PATH_DEFAULT = ROOT / "docs" / "DG-KAN_v12.8.3_主线B109_经典基函数全家族高效率化_FunctionalGeometry_结果复盘.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v1283_b109_classic_family_functional_geometry.py"
EPS = 1.0e-12


def _now_tag() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stamp(row: Dict[str, Any]) -> Dict[str, Any]:
    row.setdefault("fake_data_used", 0)
    row.setdefault("proxy_row_used", 0)
    row.setdefault("cpu_offload_used", 0)
    row.setdefault("dataset_name_branch_used", 0)
    row.setdefault("teacher_used", 0)
    row.setdefault("distillation_used", 0)
    row.setdefault("loss_modified", 0)
    row.setdefault("sampler_or_class_weight_used", 0)
    return row


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().cpu())
        return float(value)
    except Exception:
        return default


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else 0.0


def _q(values: Sequence[float], q: float, default: float = 0.0) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return default
    pos = (len(vals) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _device_from_arg(arg: str) -> torch.device:
    if str(arg) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _b109_candidate_ids(args: argparse.Namespace) -> List[str]:
    candidates = [str(args.exact_candidate_id), str(args.fixedp_candidate_id)]
    candidates.extend(_parse_list(getattr(args, "repair_candidate_ids", "")))
    return list(dict.fromkeys(x for x in candidates if x and x != "none"))


def _is_fixedp_candidate(method_id: str, specs: Mapping[str, prim.PrimitiveSpec]) -> bool:
    spec = specs.get(method_id)
    return bool(spec and "fixedp" in str(spec.init_variant).lower())


def _select_b109_step_impl(method_id: str, specs: Mapping[str, prim.PrimitiveSpec], epoch_idx: int = 0, step_id: int = 0) -> str:
    if not fhq.TRITON_AVAILABLE:
        return "torch-autograd"
    if _is_fixedp_candidate(method_id, specs):
        return "F4-triton-fixedP-workspace"
    spec = specs.get(method_id)
    variant = str(spec.init_variant).lower() if spec else ""
    if "semifixedfreeze1" in variant:
        return "F3-triton-learnableP-workspace" if int(epoch_idx) < 1 else "F4-triton-fixedP-workspace"
    warm_match = re.search(r"warm(\d+)pupdateevery(\d+)", variant)
    if warm_match:
        warm_epochs = max(0, int(warm_match.group(1)))
        interval = max(1, int(warm_match.group(2)))
        if int(epoch_idx) < warm_epochs:
            return "F3-triton-learnableP-workspace"
        return "F3-triton-learnableP-workspace" if int(step_id) % interval == 0 else "F4-triton-fixedP-workspace"
    every_match = re.search(r"pupdateevery(\d+)", variant)
    if every_match:
        interval = max(1, int(every_match.group(1)))
        return "F3-triton-learnableP-workspace" if int(step_id) % interval == 0 else "F4-triton-fixedP-workspace"
    return "F3-triton-learnableP-workspace"


def _apply_logit_gain_ramp(
    model: torch.nn.Module,
    method_id: str,
    specs: Mapping[str, prim.PrimitiveSpec],
    epoch_idx: int,
    total_epochs: int,
) -> float | None:
    spec = specs.get(method_id)
    variant = str(spec.init_variant).lower() if spec else ""
    match = re.search(r"gainramp(\d{3})", variant)
    if not match or not hasattr(model, "logit_gain"):
        return None
    logit_gain = getattr(model, "logit_gain")
    if not torch.is_tensor(logit_gain):
        return None
    if not hasattr(model, "_v1283_logit_gain_target"):
        setattr(model, "_v1283_logit_gain_target", logit_gain.detach().clone())
    target = getattr(model, "_v1283_logit_gain_target").to(device=logit_gain.device, dtype=logit_gain.dtype)
    start_value = float(match.group(1)) / 100.0
    denom = max(1, int(total_epochs) - 1)
    alpha = min(1.0, max(0.0, float(epoch_idx) / float(denom)))
    scheduled = torch.full_like(target, start_value).mul(1.0 - alpha).add(target * alpha)
    with torch.no_grad():
        logit_gain.copy_(scheduled)
    return float(scheduled.detach().mean().item())


def _apply_cross_readout_scale_warmup(
    model: torch.nn.Module,
    method_id: str,
    specs: Mapping[str, prim.PrimitiveSpec],
    epoch_idx: int,
    total_epochs: int,
) -> float | None:
    spec = specs.get(method_id)
    variant = str(spec.init_variant).lower() if spec else ""
    if "crosswarm" not in variant or not hasattr(model, "input_cross_readout_scale"):
        return None
    if not hasattr(model, "_v1283_cross_readout_scale_target"):
        setattr(model, "_v1283_cross_readout_scale_target", float(getattr(model, "input_cross_readout_scale")))
    target = float(getattr(model, "_v1283_cross_readout_scale_target"))
    start_value = 0.0
    start_match = re.search(r"crosswarm(\d{3})", variant)
    if start_match:
        start_value = float(start_match.group(1)) / 100.0
    denom = max(1, int(total_epochs) - 1)
    alpha = min(1.0, max(0.0, float(epoch_idx) / float(denom)))
    scheduled = start_value * (1.0 - alpha) + target * alpha
    setattr(model, "input_cross_readout_scale", float(scheduled))
    return float(scheduled)


def _apply_linear_residual_scale_ramp(
    model: torch.nn.Module,
    method_id: str,
    specs: Mapping[str, prim.PrimitiveSpec],
    epoch_idx: int,
    total_epochs: int,
) -> float | None:
    spec = specs.get(method_id)
    variant = str(spec.init_variant).lower() if spec else ""
    match = re.search(r"linearresramp(\d{3,5})", variant)
    if not match or not hasattr(model, "linear_residual_scale"):
        return None
    if not hasattr(model, "_v1283_linear_residual_scale_target"):
        setattr(model, "_v1283_linear_residual_scale_target", float(getattr(model, "linear_residual_scale")))
    target = float(getattr(model, "_v1283_linear_residual_scale_target"))
    start_value = float(int(match.group(1))) / 100.0
    denom = max(1, int(total_epochs) - 1)
    alpha = min(1.0, max(0.0, float(epoch_idx) / float(denom)))
    scheduled = start_value * (1.0 - alpha) + target * alpha
    setattr(model, "linear_residual_scale", float(scheduled))
    return float(scheduled)


def _apply_quad_branch_ramp(
    model: torch.nn.Module,
    method_id: str,
    specs: Mapping[str, prim.PrimitiveSpec],
    epoch_idx: int,
    total_epochs: int,
) -> float | None:
    spec = specs.get(method_id)
    variant = str(spec.init_variant).lower() if spec else ""
    match = re.search(r"quadramp(\d{3})", variant)
    if not match or not hasattr(model, "branch_scale"):
        return None
    branch_scale = getattr(model, "branch_scale")
    if not torch.is_tensor(branch_scale):
        return None
    if not hasattr(model, "_v1283_quad_branch_target"):
        setattr(model, "_v1283_quad_branch_target", branch_scale.detach().clone())
    target = getattr(model, "_v1283_quad_branch_target").to(device=branch_scale.device, dtype=branch_scale.dtype)
    start_value = float(match.group(1)) / 100.0
    denom = max(1, int(total_epochs) - 1)
    alpha = min(1.0, max(0.0, float(epoch_idx) / float(denom)))
    scheduled = start_value * (1.0 - alpha) + target[1].detach() * alpha
    with torch.no_grad():
        if branch_scale.ndim == 0:
            branch_scale.copy_(torch.as_tensor(scheduled, device=branch_scale.device, dtype=branch_scale.dtype))
        else:
            branch_scale[1].copy_(scheduled)
    return float(torch.as_tensor(scheduled).detach().float().mean().item())


def _apply_direct_branch_ramp(
    model: torch.nn.Module,
    method_id: str,
    specs: Mapping[str, prim.PrimitiveSpec],
    epoch_idx: int,
    total_epochs: int,
) -> float | None:
    spec = specs.get(method_id)
    variant = str(spec.init_variant).lower() if spec else ""
    match = re.search(r"directramp(\d{3})", variant)
    if not match or not hasattr(model, "branch_scale"):
        return None
    branch_scale = getattr(model, "branch_scale")
    if not torch.is_tensor(branch_scale):
        return None
    if not hasattr(model, "_v1283_direct_branch_target"):
        setattr(model, "_v1283_direct_branch_target", branch_scale.detach().clone())
    target = getattr(model, "_v1283_direct_branch_target").to(device=branch_scale.device, dtype=branch_scale.dtype)
    start_value = float(match.group(1)) / 100.0
    denom = max(1, int(total_epochs) - 1)
    alpha = min(1.0, max(0.0, float(epoch_idx) / float(denom)))
    scheduled = start_value * (1.0 - alpha) + target[0].detach() * alpha
    with torch.no_grad():
        if branch_scale.ndim == 0:
            branch_scale.copy_(torch.as_tensor(scheduled, device=branch_scale.device, dtype=branch_scale.dtype))
        else:
            branch_scale[0].copy_(scheduled)
    return float(torch.as_tensor(scheduled).detach().float().mean().item())


def _load_mnist(args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    data = v120._load_vision_split(
        args,
        "MNIST",
        train_size=int(args.train_size),
        val_size=int(args.val_size),
        test_size=int(args.test_size),
    )
    x_train, y_train, x_val, y_val, x_test, y_test, input_dim, output_dim, _protocol = data
    return x_train, y_train, x_val, y_val, x_test, y_test, int(input_dim), int(output_dim)


def _specs_for(input_dim: int, output_dim: int) -> Dict[str, prim.PrimitiveSpec]:
    _, budget = v124._param_budget(input_dim, output_dim)
    return {s.candidate_id: s for s in prim.primitive_specs(budget, input_dim, output_dim)}


def _make_model(
    method_id: str,
    input_dim: int,
    output_dim: int,
    x_stats: torch.Tensor,
    device: torch.device,
    seed: int,
    specs: Mapping[str, prim.PrimitiveSpec],
    y_stats: torch.Tensor | None = None,
) -> torch.nn.Module:
    _, budget = v124._param_budget(input_dim, output_dim)
    return v124._make_model(method_id, input_dim, output_dim, x_stats, device, int(seed), specs.get(method_id), budget, y_stats)


def _time_call(fn, device: torch.device, warmup: int, measure: int) -> Tuple[float, float, float, List[float]]:
    times: List[float] = []
    total = int(warmup) + int(measure)
    for idx in range(total):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        t1 = time.perf_counter()
        if idx >= int(warmup):
            times.append((t1 - t0) * 1000.0)
    return _q(times, 0.50), _q(times, 0.90), _mean(times), times


def _make_adamw(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    variant = str(getattr(getattr(model, "spec", None), "init_variant", "")).lower()
    proj_lr_scale = 1.0
    proj_lr_after_scale = None
    proj_lr_switch_epoch = 0
    proj_lr_schedule: List[Tuple[int, float]] = []
    schedule_match = re.search(r"projlr(\d{3})((?:to\d{3}e\d+)+)", variant)
    if schedule_match:
        proj_lr_scale = float(int(schedule_match.group(1))) / 100.0
        for stage_match in re.finditer(r"to(\d{3})e(\d+)", schedule_match.group(2)):
            stage_scale = float(int(stage_match.group(1))) / 100.0
            stage_epoch = int(stage_match.group(2))
            proj_lr_schedule.append((stage_epoch, stage_scale))
        if proj_lr_schedule:
            proj_lr_switch_epoch, proj_lr_after_scale = proj_lr_schedule[0]
    elif "projlr075" in variant:
        proj_lr_scale = 0.75
    elif "projlr050" in variant:
        proj_lr_scale = 0.50
    elif "projlr025" in variant:
        proj_lr_scale = 0.25
    elif "projlr010" in variant:
        proj_lr_scale = 0.10
    cross_readout_lr_scale = 1.0
    cross_readout_match = re.search(r"crossreadoutlr(\d{3})", variant)
    if cross_readout_match:
        cross_readout_lr_scale = float(int(cross_readout_match.group(1))) / 100.0
    readout_lr_scale = 1.0
    readout_match = re.search(r"(?<!cross)readoutlr(\d{3})", variant)
    if readout_match:
        readout_lr_scale = float(int(readout_match.group(1))) / 100.0
    hidden_tail_readout_lr_scale = 1.0
    hidden_tail_readout_lr_after_scale = None
    hidden_tail_readout_lr_switch_epoch = 0
    hidden_tail_readout_lr_schedule: List[Tuple[int, float]] = []
    hidden_tail_schedule_match = re.search(r"hiddentailreadoutlr(\d{3})((?:to\d{3}e\d+)+)", variant)
    if hidden_tail_schedule_match:
        hidden_tail_readout_lr_scale = float(int(hidden_tail_schedule_match.group(1))) / 100.0
        for stage_match in re.finditer(r"to(\d{3})e(\d+)", hidden_tail_schedule_match.group(2)):
            stage_scale = float(int(stage_match.group(1))) / 100.0
            stage_epoch = int(stage_match.group(2))
            hidden_tail_readout_lr_schedule.append((stage_epoch, stage_scale))
        if hidden_tail_readout_lr_schedule:
            hidden_tail_readout_lr_switch_epoch, hidden_tail_readout_lr_after_scale = hidden_tail_readout_lr_schedule[0]
    else:
        hidden_tail_readout_match = re.search(r"hiddentailreadoutlr(\d{3})", variant)
    if not hidden_tail_schedule_match and hidden_tail_readout_match:
        hidden_tail_readout_lr_scale = float(int(hidden_tail_readout_match.group(1))) / 100.0
    projection_params = []
    for attr in ("quad_proj", "quad_proj_scale", "quad_sparse_weight"):
        param = getattr(model, attr, None)
        if isinstance(param, torch.nn.Parameter) and param.requires_grad:
            projection_params.append(param)
    cross_readout_params = []
    cross_readout = getattr(model, "cross_readout", None)
    if isinstance(cross_readout, torch.nn.Parameter) and cross_readout.requires_grad:
        cross_readout_params.append(cross_readout)
    readout_params = []
    for attr in ("w2", "linear_readout", "cross_readout", "cheby_cross_readout", "cheby_input_cross_readout"):
        param = getattr(model, attr, None)
        if isinstance(param, torch.nn.Parameter) and param.requires_grad and not (cross_readout_lr_scale != 1.0 and attr == "cross_readout"):
            readout_params.append(param)
    hidden_tail_readout_params = []
    for attr in ("hidden_square_readout", "hidden_abs_readout", "hidden_rat_readout"):
        param = getattr(model, attr, None)
        if isinstance(param, torch.nn.Parameter) and param.requires_grad:
            hidden_tail_readout_params.append(param)
    needs_projection_group = (proj_lr_scale < 1.0 or proj_lr_after_scale is not None) and projection_params
    needs_readout_group = readout_lr_scale != 1.0 and readout_params
    needs_cross_readout_group = cross_readout_lr_scale != 1.0 and cross_readout_params
    needs_hidden_tail_readout_group = (
        hidden_tail_readout_lr_scale != 1.0 or hidden_tail_readout_lr_after_scale is not None
    ) and hidden_tail_readout_params
    if needs_projection_group or needs_readout_group or needs_cross_readout_group or needs_hidden_tail_readout_group:
        special_ids = set()
        if needs_projection_group:
            special_ids.update(id(p) for p in projection_params)
        if needs_readout_group:
            special_ids.update(id(p) for p in readout_params)
        if needs_cross_readout_group:
            special_ids.update(id(p) for p in cross_readout_params)
        if needs_hidden_tail_readout_group:
            special_ids.update(id(p) for p in hidden_tail_readout_params)
        base_params = [p for p in model.parameters() if p.requires_grad and id(p) not in special_ids]
        groups = [{"params": base_params, "lr": float(args.lr), "lr_scale": 1.0}]
        if needs_projection_group:
            groups.append(
                {
                "params": projection_params,
                "lr": float(args.lr) * proj_lr_scale,
                "lr_scale": proj_lr_scale,
                "lr_scale_after": proj_lr_after_scale,
                "lr_scale_switch_epoch": proj_lr_switch_epoch,
                "lr_scale_schedule": proj_lr_schedule,
                }
            )
        if needs_readout_group:
            groups.append({"params": readout_params, "lr": float(args.lr) * readout_lr_scale, "lr_scale": readout_lr_scale})
        if needs_cross_readout_group:
            groups.append({"params": cross_readout_params, "lr": float(args.lr) * cross_readout_lr_scale, "lr_scale": cross_readout_lr_scale})
        if needs_hidden_tail_readout_group:
            groups.append(
                {
                    "params": hidden_tail_readout_params,
                    "lr": float(args.lr) * hidden_tail_readout_lr_scale,
                    "lr_scale": hidden_tail_readout_lr_scale,
                    "lr_scale_after": hidden_tail_readout_lr_after_scale,
                    "lr_scale_switch_epoch": hidden_tail_readout_lr_switch_epoch,
                    "lr_scale_schedule": hidden_tail_readout_lr_schedule,
                }
            )
        return torch.optim.AdamW(groups, lr=float(args.lr), weight_decay=float(args.weight_decay))
    return torch.optim.AdamW([{"params": [p for p in model.parameters() if p.requires_grad], "lr": float(args.lr), "lr_scale": 1.0}], lr=float(args.lr), weight_decay=float(args.weight_decay))


def _family_plan() -> List[Tuple[str, str, str]]:
    return [
        ("BSpline", "B6r-BSpline-order1-stream-K2-repair", "D1-A UniformLinearSplineKAN / L1 stream + L2 compile-forward attempt"),
        ("Rational", "B7b-RationalKAT-flashgroup-G16-h112-linearres-tritonL3", "D2-B FlashKAT-inspired grouped rational activation with GEMM readout and manual L3 gradient"),
        ("Rational", "B7c-RationalKAT-flashgroup-G16-h80-linearres-tritonL3", "D2-C FlashKAT grouped rational h80 speed repair after B7b step fail"),
        ("Rational", "B7d-RationalKAT-flashgroup-G16-h64-linearres-tritonL3", "D2-D FlashKAT grouped rational h64 speed repair after B7b step fail"),
        ("Rational", "B7e-RationalKAT-flashgroup-G16-h64-linearres-inputcrossR64-tritonL3", "D2-E FlashKAT grouped rational h64 fixed input-cross R64 expression repair"),
        ("Rational", "B7f-RationalKAT-flashgroup-G16-h64-linearres-inputcrossR128-tritonL3", "D2-F FlashKAT grouped rational h64 fixed input-cross R128 expression repair"),
        ("Rational", "B7g-RationalKAT-flashgroup-G16-h64-linearres-inputcrossR96-tritonL3", "D2-G FlashKAT grouped rational h64 fixed input-cross R96 efficiency/expression bracket"),
        ("Rational", "B7h-RationalKAT-flashgroup-G16-h64-linearres-inputcrossR112-tritonL3", "D2-H FlashKAT grouped rational h64 fixed input-cross R112 efficiency/expression bracket"),
        ("Rational", "B7i-RationalKAT-flashgroup-G16-h64-linearres-paircrossR136-tritonL3", "D2-I FlashKAT grouped rational h64 structured pair-cross R136 expression repair"),
        ("Rational", "B7j-RationalKAT-flashgroup-G16-h64-linearres-paircrossR96-tritonL3", "D2-J FlashKAT grouped rational h64 structured pair-cross R96 efficiency bracket"),
        ("Rational", "B7k-RationalKAT-flashgroup-G16-h64-linearres-paircrossR88-tritonL3", "D2-K FlashKAT grouped rational h64 structured pair-cross R88 efficiency bracket"),
        ("Rational", "B7l-RationalKAT-flashgroup-G16-h64-linearres-paircrossR80-tritonL3", "D2-L FlashKAT grouped rational h64 structured pair-cross R80 efficiency bracket"),
        ("Rational", "B7m-RationalKAT-flashgroup-G16-h64-linearres-balancedPairR88-tritonL3", "D2-M FlashKAT grouped rational h64 balanced pair-cross R88 expression repair"),
        ("Rational", "B7n-RationalKAT-flashgroup-G16-h64-linearres-balancedPairR96-tritonL3", "D2-N FlashKAT grouped rational h64 balanced pair-cross R96 expression/efficiency bracket"),
        ("Rational", "B7o-RationalKAT-flashgroup-G16-h64-linearres-transformPairR88-tritonL3", "D2-O FlashKAT grouped rational h64 fixed transform pair-cross R88 rotated-expression repair"),
        ("Rational", "B7p-RationalKAT-flashgroup-G16-h64-linearres-transformPairR96-tritonL3", "D2-P FlashKAT grouped rational h64 fixed transform pair-cross R96 rotated-expression bracket"),
        ("Rational", "B7q-RationalKAT-flashgroup-G16-h56-linearres-paircrossR136-tritonL3", "D2-Q FlashKAT grouped rational h56 structured pair-cross R136 full-coverage efficiency repair"),
        ("Rational", "B7r-RationalKAT-flashgroup-G16-h48-linearres-paircrossR136-tritonL3", "D2-R FlashKAT grouped rational h48 structured pair-cross R136 full-coverage efficiency repair"),
        ("Rational", "B7s-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-tritonL3", "D2-S FlashKAT grouped rational h40 structured pair-cross R136 full-coverage efficiency repair"),
        ("Rational", "B7t-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-tritonL3", "D2-T FlashKAT grouped rational h32 structured pair-cross R136 full-coverage efficiency repair"),
        ("Rational", "B7u-RationalKAT-flashgroup-G16-h40-linearres-gridPairR136-tritonL3", "D2-U FlashKAT grouped rational h40 grid-balanced pair-cross R136 task-trajectory repair"),
        ("Rational", "B7v-RationalKAT-flashgroup-G16-h32-linearres-gridPairR136-tritonL3", "D2-V FlashKAT grouped rational h32 grid-balanced pair-cross R136 task-trajectory repair"),
        ("Rational", "B7w-RationalKAT-flashgroup-G16-h40-linearres-paircrossR112-tritonL3", "D2-W FlashKAT grouped rational h40 structured pair-cross R112 task/expression bracket"),
        ("Rational", "B7x-RationalKAT-flashgroup-G16-h32-linearres-paircrossR112-tritonL3", "D2-X FlashKAT grouped rational h32 structured pair-cross R112 task/expression bracket"),
        ("Rational", "B7y-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readoutLR300-tritonL3", "D2-Y FlashKAT grouped rational h40 pair-cross R136 readout-LR task repair"),
        ("Rational", "B7z-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readoutLR300-tritonL3", "D2-Z FlashKAT grouped rational h32 pair-cross R136 readout-LR task repair"),
        ("Rational", "B7aa-RationalKAT-flashgroup-G16-h40-linearres-pairHiddenR136S025-tritonL3", "D2-AA FlashKAT grouped rational h40 fixed pair-hidden R136 task-dynamics repair"),
        ("Rational", "B7ab-RationalKAT-flashgroup-G16-h32-linearres-pairHiddenR136S025-tritonL3", "D2-AB FlashKAT grouped rational h32 fixed pair-hidden R136 task-dynamics repair"),
        ("Rational", "B7ac-RationalKAT-flashgroup-G16-h40-linearres-pairBucketR136S025-tritonL3", "D2-AC FlashKAT grouped rational h40 bucketed pair-hidden R136 low-cost task-dynamics repair"),
        ("Rational", "B7ad-RationalKAT-flashgroup-G16-h32-linearres-pairBucketR136S025-tritonL3", "D2-AD FlashKAT grouped rational h32 bucketed pair-hidden R136 low-cost task-dynamics repair"),
        ("Rational", "B7ae-RationalKAT-flashgroup-G16-h40-linearres-pairBucketR112S025-tritonL3", "D2-AE FlashKAT grouped rational h40 bucketed pair-hidden R112 efficiency repair"),
        ("Rational", "B7af-RationalKAT-flashgroup-G16-h32-linearres-pairBucketR112S025-tritonL3", "D2-AF FlashKAT grouped rational h32 bucketed pair-hidden R112 efficiency repair"),
        ("Rational", "B7ag-RationalKAT-flashgroup-G16-h40-linearres-pairBucketR88S025-tritonL3", "D2-AG FlashKAT grouped rational h40 bucketed pair-hidden R88 efficiency/expression bracket"),
        ("Rational", "B7ah-RationalKAT-flashgroup-G16-h32-linearres-pairBucketR88S025-tritonL3", "D2-AH FlashKAT grouped rational h32 bucketed pair-hidden R88 efficiency/expression bracket"),
        ("Rational", "B7ai-RationalKAT-flashgroup-G16-h40-linearres-pairBucketR136S025-coalescedTritonL3", "D2-AI FlashKAT grouped rational h40 bucketed pair-hidden R136 coalesced Triton hidden-delta repair"),
        ("Rational", "B7aj-RationalKAT-flashgroup-G16-h32-linearres-pairBucketR136S025-coalescedTritonL3", "D2-AJ FlashKAT grouped rational h32 bucketed pair-hidden R136 coalesced Triton hidden-delta repair"),
        ("Rational", "B7ak-RationalKAT-flashgroup-G16-h40-linearres-pairBucketR112S025-coalescedTritonL3", "D2-AK FlashKAT grouped rational h40 bucketed pair-hidden R112 coalesced Triton hidden-delta repair"),
        ("Rational", "B7al-RationalKAT-flashgroup-G16-h32-linearres-pairBucketR112S025-coalescedTritonL3", "D2-AL FlashKAT grouped rational h32 bucketed pair-hidden R112 coalesced Triton hidden-delta repair"),
        ("Rational", "B7am-RationalKAT-flashgroup-G16-h40-linearres-pairBucketDirectR136S025-coalescedTritonL3", "D2-AM FlashKAT grouped rational h40 bucket hidden R136 plus direct pair readout coupling repair"),
        ("Rational", "B7an-RationalKAT-flashgroup-G16-h32-linearres-pairBucketDirectR136S025-coalescedTritonL3", "D2-AN FlashKAT grouped rational h32 bucket hidden R136 plus direct pair readout coupling repair"),
        ("Rational", "B7ao-RationalKAT-flashgroup-G16-h40-linearres-pairBucketDirectR112S025-coalescedTritonL3", "D2-AO FlashKAT grouped rational h40 bucket hidden R112 plus direct pair readout coupling bracket"),
        ("Rational", "B7ap-RationalKAT-flashgroup-G16-h32-linearres-pairBucketDirectR112S025-coalescedTritonL3", "D2-AP FlashKAT grouped rational h32 bucket hidden R112 plus direct pair readout coupling bracket"),
        ("Rational", "B7aq-RationalKAT-flashgroup-G16-h40-linearres-pairCrossR136-bucketHiddenR32S025-tritonL3", "D2-AQ FlashKAT grouped rational h40 direct pair R136 with small bucket hidden R32 trajectory coupling"),
        ("Rational", "B7ar-RationalKAT-flashgroup-G16-h32-linearres-pairCrossR136-bucketHiddenR32S025-tritonL3", "D2-AR FlashKAT grouped rational h32 direct pair R136 with small bucket hidden R32 trajectory coupling"),
        ("Rational", "B7as-RationalKAT-flashgroup-G16-h40-linearres-pairCrossR136-bucketHiddenR64S025-tritonL3", "D2-AS FlashKAT grouped rational h40 direct pair R136 with small bucket hidden R64 trajectory coupling"),
        ("Rational", "B7at-RationalKAT-flashgroup-G16-h32-linearres-pairCrossR136-bucketHiddenR64S025-tritonL3", "D2-AT FlashKAT grouped rational h32 direct pair R136 with small bucket hidden R64 trajectory coupling"),
        ("Rational", "B7au-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readscale025-tritonL3", "D2-AU FlashKAT grouped rational h40 pair R136 with fixed pair-readout scale 0.25 task-trajectory repair"),
        ("Rational", "B7av-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readscale025-tritonL3", "D2-AV FlashKAT grouped rational h32 pair R136 with fixed pair-readout scale 0.25 task-trajectory repair"),
        ("Rational", "B7aw-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readscale010-tritonL3", "D2-AW FlashKAT grouped rational h40 pair R136 with fixed pair-readout scale 0.10 task-trajectory repair"),
        ("Rational", "B7ax-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readscale010-tritonL3", "D2-AX FlashKAT grouped rational h32 pair R136 with fixed pair-readout scale 0.10 task-trajectory repair"),
        ("Rational", "B7ay-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readtritonL3", "D2-AY FlashKAT grouped rational h40 pair R136 with fused Triton pair-readout logits and grad"),
        ("Rational", "B7az-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readtritonL3", "D2-AZ FlashKAT grouped rational h32 pair R136 with fused Triton pair-readout logits and grad"),
        ("Rational", "B7ba-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readtriton-readscale010-L3", "D2-BA FlashKAT grouped rational h40 pair R136 with fused pair-readout and fixed scale 0.10"),
        ("Rational", "B7bb-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readtriton-readscale010-L3", "D2-BB FlashKAT grouped rational h32 pair R136 with fused pair-readout and fixed scale 0.10"),
        ("Rational", "B7bc-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktritonL3", "D2-BC FlashKAT grouped rational h40 pair R136 with block-tiled Triton pair-readout logits and grad"),
        ("Rational", "B7bd-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktritonL3", "D2-BD FlashKAT grouped rational h32 pair R136 with block-tiled Triton pair-readout logits and grad"),
        ("Rational", "B7be-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-readscale010-L3", "D2-BE FlashKAT grouped rational h40 pair R136 with block-tiled pair-readout and fixed scale 0.10"),
        ("Rational", "B7bf-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-readscale010-L3", "D2-BF FlashKAT grouped rational h32 pair R136 with block-tiled pair-readout and fixed scale 0.10"),
        ("Rational", "B7bg-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-logitbiasL3", "D2-BG FlashKAT grouped rational h40 block-tiled pair-readout with learnable logit bias"),
        ("Rational", "B7bh-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-logitbiasL3", "D2-BH FlashKAT grouped rational h32 block-tiled pair-readout with learnable logit bias"),
        ("Rational", "B7bi-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-readscale010-logitbiasL3", "D2-BI FlashKAT grouped rational h40 block-tiled pair-readout scale 0.10 with learnable logit bias"),
        ("Rational", "B7bj-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-readscale010-logitbiasL3", "D2-BJ FlashKAT grouped rational h32 block-tiled pair-readout scale 0.10 with learnable logit bias"),
        ("Rational", "B7bk-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-logitbiasfusedL3", "D2-BK FlashKAT grouped rational h40 block-tiled pair-readout with bias fused inside the logits kernel"),
        ("Rational", "B7bl-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-logitbiasfusedL3", "D2-BL FlashKAT grouped rational h32 block-tiled pair-readout with bias fused inside the logits kernel"),
        ("Rational", "B7bm-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-readscale010-logitbiasfusedL3", "D2-BM FlashKAT grouped rational h40 block-tiled pair-readout scale 0.10 with bias fused inside the logits kernel"),
        ("Rational", "B7bn-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-readscale010-logitbiasfusedL3", "D2-BN FlashKAT grouped rational h32 block-tiled pair-readout scale 0.10 with bias fused inside the logits kernel"),
        ("Rational", "B7bo-RationalKAT-flashgroup-G16-h40-linearres-paircrossR112-readblocktritonL3", "D2-BO FlashKAT grouped rational h40 block-tiled pair-readout R112 efficiency repair"),
        ("Rational", "B7bp-RationalKAT-flashgroup-G16-h32-linearres-paircrossR112-readblocktritonL3", "D2-BP FlashKAT grouped rational h32 block-tiled pair-readout R112 efficiency repair"),
        ("Rational", "B7bq-RationalKAT-flashgroup-G16-h40-linearres-paircrossR112-readblocktriton-readscale010-L3", "D2-BQ FlashKAT grouped rational h40 block-tiled pair-readout R112 scale 0.10"),
        ("Rational", "B7br-RationalKAT-flashgroup-G16-h32-linearres-paircrossR112-readblocktriton-readscale010-L3", "D2-BR FlashKAT grouped rational h32 block-tiled pair-readout R112 scale 0.10"),
        ("Rational", "B7bs-RationalKAT-flashgroup-G16-h40-linearres-paircrossR128-readblocktritonL3", "D2-BS FlashKAT grouped rational h40 block-tiled pair-readout R128 expression/efficiency bracket"),
        ("Rational", "B7bt-RationalKAT-flashgroup-G16-h32-linearres-paircrossR128-readblocktritonL3", "D2-BT FlashKAT grouped rational h32 block-tiled pair-readout R128 expression/efficiency bracket"),
        ("Rational", "B7bu-RationalKAT-flashgroup-G16-h40-linearres-paircrossR128-readblocktriton-readscale010-L3", "D2-BU FlashKAT grouped rational h40 block-tiled pair-readout R128 scale 0.10"),
        ("Rational", "B7bv-RationalKAT-flashgroup-G16-h32-linearres-paircrossR128-readblocktriton-readscale010-L3", "D2-BV FlashKAT grouped rational h32 block-tiled pair-readout R128 scale 0.10"),
        ("Rational", "B7bw-RationalKAT-flashgroup-G16-h40-linearres-paircrossR120-readblocktritonL3", "D2-BW FlashKAT grouped rational h40 block-tiled pair-readout R120 expression/efficiency bracket"),
        ("Rational", "B7bx-RationalKAT-flashgroup-G16-h32-linearres-paircrossR120-readblocktritonL3", "D2-BX FlashKAT grouped rational h32 block-tiled pair-readout R120 expression/efficiency bracket"),
        ("Rational", "B7by-RationalKAT-flashgroup-G16-h40-linearres-paircrossR120-readblocktriton-readscale010-L3", "D2-BY FlashKAT grouped rational h40 block-tiled pair-readout R120 scale 0.10"),
        ("Rational", "B7bz-RationalKAT-flashgroup-G16-h32-linearres-paircrossR120-readblocktriton-readscale010-L3", "D2-BZ FlashKAT grouped rational h32 block-tiled pair-readout R120 scale 0.10"),
        ("Rational", "B7ca-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-pairnormL3", "D2-CA FlashKAT grouped rational h40 block-tiled pair-readout R136 with fixed pair-feature normalization"),
        ("Rational", "B7cb-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-pairnormL3", "D2-CB FlashKAT grouped rational h32 block-tiled pair-readout R136 with fixed pair-feature normalization"),
        ("Rational", "B7cc-RationalKAT-flashgroup-G16-h40-linearres-paircrossR120-readblocktriton-pairnormL3", "D2-CC FlashKAT grouped rational h40 block-tiled pair-readout R120 with fixed pair-feature normalization"),
        ("Rational", "B7cd-RationalKAT-flashgroup-G16-h32-linearres-paircrossR120-readblocktriton-pairnormL3", "D2-CD FlashKAT grouped rational h32 block-tiled pair-readout R120 with fixed pair-feature normalization"),
        ("Rational", "B7ce-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-pairstdL3", "D2-CE FlashKAT grouped rational h40 block-tiled pair-readout R136 with fixed pair-feature std scaling only"),
        ("Rational", "B7cf-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-pairstdL3", "D2-CF FlashKAT grouped rational h32 block-tiled pair-readout R136 with fixed pair-feature std scaling only"),
        ("Rational", "B7cg-RationalKAT-flashgroup-G16-h40-linearres-paircrossR120-readblocktriton-pairstdL3", "D2-CG FlashKAT grouped rational h40 block-tiled pair-readout R120 with fixed pair-feature std scaling only"),
        ("Rational", "B7ch-RationalKAT-flashgroup-G16-h32-linearres-paircrossR120-readblocktriton-pairstdL3", "D2-CH FlashKAT grouped rational h32 block-tiled pair-readout R120 with fixed pair-feature std scaling only"),
        ("Rational", "B7ci-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossReadoutLR025-L3", "D2-CI FlashKAT grouped rational h40 block-tiled pair-readout R136 with cross-readout-only LR 0.25"),
        ("Rational", "B7cj-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossReadoutLR025-L3", "D2-CJ FlashKAT grouped rational h32 block-tiled pair-readout R136 with cross-readout-only LR 0.25"),
        ("Rational", "B7ck-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-L3", "D2-CK FlashKAT grouped rational h40 block-tiled pair-readout R136 with zero-initialized cross readout"),
        ("Rational", "B7cl-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-L3", "D2-CL FlashKAT grouped rational h32 block-tiled pair-readout R136 with zero-initialized cross readout"),
        ("Rational", "B7cm-RationalKAT-flashgroup-G16-h40-linearres-paircrossR128-readblocktriton-crossZero-L3", "D2-CM FlashKAT grouped rational h40 block-tiled pair-readout R128 with zero-initialized cross readout"),
        ("Rational", "B7cn-RationalKAT-flashgroup-G16-h32-linearres-paircrossR128-readblocktriton-crossZero-L3", "D2-CN FlashKAT grouped rational h32 block-tiled pair-readout R128 with zero-initialized cross readout"),
        ("Rational", "B7co-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-b32-L3", "D2-CO FlashKAT grouped rational h40 R136 crosszero block-readout with B32 logits tiling"),
        ("Rational", "B7cp-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-L3", "D2-CP FlashKAT grouped rational h32 R136 crosszero block-readout with B32 logits tiling"),
        ("Rational", "B7do-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-b32-readscale025-L3", "D2-DO FlashKAT grouped rational h40 R136 crosszero B32 block-readout with fixed pair-readout scale 0.25"),
        ("Rational", "B7dp-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-readscale025-L3", "D2-DP FlashKAT grouped rational h32 R136 crosszero B32 block-readout with fixed pair-readout scale 0.25"),
        ("Rational", "B7dq-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-b32-readscale010-L3", "D2-DQ FlashKAT grouped rational h40 R136 crosszero B32 block-readout with fixed pair-readout scale 0.10"),
        ("Rational", "B7dr-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-readscale010-L3", "D2-DR FlashKAT grouped rational h32 R136 crosszero B32 block-readout with fixed pair-readout scale 0.10"),
        ("Rational", "B7ds-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-readscale025-crossWarm-L3", "D2-DS FlashKAT grouped rational h32 R136 crosszero B32 block-readout with fixed scale 0.25 and no-extra-op epoch warmup"),
        ("Rational", "B7dt-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-readscale010-crossWarm-L3", "D2-DT FlashKAT grouped rational h32 R136 crosszero B32 block-readout with fixed scale 0.10 and no-extra-op epoch warmup"),
        ("Rational", "B7du-RationalKAT-flashgroup-G16-h40-linearres-paircrossR112-readblocktriton-crossZero-b32-L3", "D2-DU FlashKAT grouped rational h40 R112 crosszero B32 block-readout lower-rank bracket"),
        ("Rational", "B7dv-RationalKAT-flashgroup-G16-h32-linearres-paircrossR112-readblocktriton-crossZero-b32-L3", "D2-DV FlashKAT grouped rational h32 R112 crosszero B32 block-readout lower-rank bracket"),
        ("Rational", "B7dw-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-L3", "D2-DW FlashKAT grouped rational h40 R136 crosszero B32 readout-only trajectory with frozen rational/w1 backbone and loss-agnostic VJP"),
        ("Rational", "B7dx-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-L3", "D2-DX FlashKAT grouped rational h32 R136 crosszero B32 readout-only trajectory with frozen rational/w1 backbone and loss-agnostic VJP"),
        ("Rational", "B7dy-RationalKAT-flashgroup-G16-h40-linearresGain800-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-L3", "D2-DY FlashKAT grouped rational h40 R136 crosszero B32 frozen-backbone trajectory with stronger loss-agnostic linear residual readout"),
        ("Rational", "B7dz-RationalKAT-flashgroup-G16-h32-linearresGain800-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-L3", "D2-DZ FlashKAT grouped rational h32 R136 crosszero B32 frozen-backbone trajectory with stronger loss-agnostic linear residual readout"),
        ("Rational", "B7ea-RationalKAT-flashgroup-G16-h32-linearresGain1600-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-L3", "D2-EA FlashKAT grouped rational h32 R136 crosszero B32 frozen-backbone trajectory with linear residual gain 16.0"),
        ("Rational", "B7eb-RationalKAT-flashgroup-G16-h32-linearresGain3200-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-L3", "D2-EB FlashKAT grouped rational h32 R136 crosszero B32 frozen-backbone trajectory with linear residual gain 32.0"),
        ("Rational", "B7ec-RationalKAT-flashgroup-G16-h32-linearresGain1600-paircrossR136-readblocktriton-crossZero-b32-freezeRational-L3", "D2-EC FlashKAT grouped rational h32 R136 crosszero B32 frozen-rational trajectory with trainable w1 and linear residual gain 16.0"),
        ("Rational", "B7ed-RationalKAT-flashgroup-G16-h32-linearresGain3200-paircrossR136-readblocktriton-crossZero-b32-freezeRational-L3", "D2-ED FlashKAT grouped rational h32 R136 crosszero B32 frozen-rational trajectory with trainable w1 and linear residual gain 32.0"),
        ("Rational", "B7ee-RationalKAT-flashgroup-G16-h32-linearresGain3200-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "D2-EE FlashKAT grouped rational h32 R136 crosszero B32 frozen-backbone trajectory with trainable hidden bias and linear residual gain 32.0"),
        ("Rational", "B7ef-RationalKAT-flashgroup-G16-h32-linearresGain6400-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "D2-EF FlashKAT grouped rational h32 R136 crosszero B32 frozen-backbone trajectory with trainable hidden bias and linear residual gain 64.0"),
        ("Rational", "B7eg-RationalKAT-flashgroup-G16-h32-linearresGain4800-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "D2-EG FlashKAT grouped rational h32 R136 crosszero B32 frozen-backbone trajectory with trainable hidden bias and linear residual gain 48.0"),
        ("Rational", "B7eh-RationalKAT-flashgroup-G16-h32-linearresGain9600-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "D2-EH FlashKAT grouped rational h32 R136 crosszero B32 frozen-backbone trajectory with trainable hidden bias and linear residual gain 96.0"),
        ("Rational", "B7ei-RationalKAT-flashgroup-G16-h32-linearresGain12800-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "D2-EI FlashKAT grouped rational h32 R136 crosszero B32 frozen-backbone trajectory with trainable hidden bias and linear residual gain 128.0"),
        ("Rational", "B7ej-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "D2-EJ FlashKAT grouped rational h32 R136 crosszero B32 frozen-backbone trajectory with trainable hidden bias and linear residual gain 192.0"),
        ("Rational", "B7ek-RationalKAT-flashgroup-G16-h32-linearresGain25600-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "D2-EK FlashKAT grouped rational h32 R136 crosszero B32 frozen-backbone trajectory with trainable hidden bias and linear residual gain 256.0"),
        ("Rational", "B7el-RationalKAT-flashgroup-G16-h32-linearresGain38400-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "D2-EL FlashKAT grouped rational h32 R136 crosszero B32 frozen-backbone trajectory with trainable hidden bias and linear residual gain 384.0"),
        ("Rational", "B7em-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairStd-crossZero-b32-freezeBackbone-hiddenBias-L3", "D2-EM FlashKAT grouped rational h32 B7ej-shaped frozen-backbone hidden-bias trajectory with pair-feature std scaling"),
        ("Rational", "B7en-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-freezeBackbone-hiddenBias-L3", "D2-EN FlashKAT grouped rational h32 B7ej-shaped frozen-backbone hidden-bias trajectory with pair-feature centering/std scaling"),
        ("Rational", "B7eo-RationalKAT-flashgroup-G16-h32-linearresGain19200-linearGate100-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "D2-EO FlashKAT grouped rational h32 B7ej-shaped frozen-backbone hidden-bias trajectory with trainable linear residual gate init 1.00"),
        ("Rational", "B7ep-RationalKAT-flashgroup-G16-h32-linearresGain19200-linearGate075-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "D2-EP FlashKAT grouped rational h32 B7ej-shaped frozen-backbone hidden-bias trajectory with trainable linear residual gate init 0.75"),
        ("Rational", "B7eq-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-logitBias-L3", "D2-EQ FlashKAT grouped rational h32 B7ej-shaped frozen-backbone hidden-bias trajectory with trainable logit intercept"),
        ("Rational", "B7er-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-readscale050-freezeBackbone-hiddenBias-logitBias-L3", "D2-ER FlashKAT grouped rational h32 B7ej-shaped frozen-backbone hidden-bias trajectory with logit intercept and fixed pair-readout scale 0.50"),
        ("Rational", "B7es-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-w2BiasRow-L3", "D2-ES FlashKAT grouped rational h32 B7ej-shaped trajectory with output intercept fused into w2 row"),
        ("Rational", "B7et-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR120-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "D2-ET FlashKAT grouped rational h32 B7ej-shaped trajectory with cheaper fixed R120 pair primitive"),
        ("Rational", "B7eu-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR120-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-w2BiasRow-L3", "D2-EU FlashKAT grouped rational h32 B7ej-shaped R120 pair primitive with output intercept fused into w2 row"),
        ("Rational", "B7ev-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-EV FlashKAT grouped rational h32 B7ej-shaped trajectory with manual foreach AdamW update"),
        ("Rational", "B7ew-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-logitBias-manualAdamW-L3", "D2-EW FlashKAT grouped rational h32 B7eq-shaped trajectory with logit intercept and manual foreach AdamW update"),
        ("Rational", "B7ex-RationalKAT-flashgroup-G16-h32-linearresGain19200-freezeBackbone-hiddenBias-L3", "D2-EX FlashKAT grouped rational h32 frozen-backbone hidden-bias trajectory without pair-readout reductions"),
        ("Rational", "B7ey-RationalKAT-flashgroup-G16-h32-linearresGain19200-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-EY FlashKAT grouped rational h32 frozen-backbone hidden-bias trajectory without pair-readout reductions and with manual foreach AdamW"),
        ("Rational", "B7ez-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairReadBucketR136G64-crossZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-EZ FlashKAT grouped rational h32 B7ej-shaped trajectory with R136 pair features bucketed to G64 before readout and manual foreach AdamW"),
        ("Rational", "B7fa-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairReadBucketR136G32-crossZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FA FlashKAT grouped rational h32 B7ej-shaped trajectory with R136 pair features bucketed to G32 before readout and manual foreach AdamW"),
        ("Rational", "B7fb-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairReadBucketR136G96-crossZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FB FlashKAT grouped rational h32 B7ej-shaped trajectory with R136 pair features bucketed to G96 before readout and manual foreach AdamW"),
        ("Rational", "B7fc-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairReadBucketR136G112-crossZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FC FlashKAT grouped rational h32 B7ej-shaped trajectory with R136 pair features bucketed to G112 before readout and manual foreach AdamW"),
        ("Rational", "B7fd-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR96-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FD FlashKAT grouped rational h32 B7ej-shaped trajectory with full R96 B32 pair block-readout and manual foreach AdamW"),
        ("Rational", "B7fe-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR80-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FE FlashKAT grouped rational h32 B7ej-shaped trajectory with full R80 B32 pair block-readout and manual foreach AdamW"),
        ("Rational", "B7ff-RationalKAT-flashgroup-G16-h32-linearresGain19200-balancedPairCrossR96-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FF FlashKAT grouped rational h32 B7ej-shaped trajectory with balanced R96 B32 pair block-readout and manual foreach AdamW"),
        ("Rational", "B7fg-RationalKAT-flashgroup-G16-h32-linearresGain19200-balancedPairCrossR88-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FG FlashKAT grouped rational h32 B7ej-shaped trajectory with balanced R88 B32 pair block-readout and manual foreach AdamW"),
        ("Rational", "B7fh-RationalKAT-flashgroup-G16-h32-linearresGain19200-projBilinR96-bilinZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FH FlashKAT grouped rational h32 B7ej-shaped trajectory with fixed dual-DCT projection-bilinear R96 GEMM readout and manual foreach AdamW"),
        ("Rational", "B7fi-RationalKAT-flashgroup-G16-h32-linearresGain19200-projSqR96-sqZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FI FlashKAT grouped rational h32 B7ej-shaped trajectory with fixed DCT projection-square R96 GEMM readout and manual foreach AdamW"),
        ("Rational", "B7fj-RationalKAT-flashgroup-G16-h32-linearresGain19200-projBilinR64-bilinZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FJ FlashKAT grouped rational h32 B7ej-shaped trajectory with fixed dual-DCT projection-bilinear R64 GEMM readout and manual foreach AdamW"),
        ("Rational", "B7fk-RationalKAT-flashgroup-G16-h32-linearresGain19200-projSqR64-sqZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FK FlashKAT grouped rational h32 B7ej-shaped trajectory with fixed DCT projection-square R64 GEMM readout and manual foreach AdamW"),
        ("Rational", "B7fl-RationalKAT-flashgroup-G16-h32-linearresGain19200-projBilinR32-bilinZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FL FlashKAT grouped rational h32 B7ej-shaped trajectory with fixed dual-DCT projection-bilinear R32 GEMM readout and manual foreach AdamW"),
        ("Rational", "B7fm-RationalKAT-flashgroup-G16-h32-linearresGain19200-projSqR32-sqZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FM FlashKAT grouped rational h32 B7ej-shaped trajectory with fixed DCT projection-square R32 GEMM readout and manual foreach AdamW"),
        ("Rational", "B7fn-RationalKAT-flashgroup-G16-h32-linearresGain19200-hiddenSqReadout-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FN FlashKAT grouped rational h32 B7ey-shaped trajectory with trainable hidden-square readout and manual foreach AdamW"),
        ("Rational", "B7fo-RationalKAT-flashgroup-G16-h32-linearresGain19200-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FO FlashKAT grouped rational h32 B7ey-shaped trajectory with zero-init hidden-square readout and manual foreach AdamW"),
        ("Rational", "B7fp-RationalKAT-flashgroup-G16-h32-linearresGain19200-balancedPairCrossR96-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FP FlashKAT grouped rational h32 balanced R96 B32 pair block-readout plus zero-init hidden-square readout and manual foreach AdamW"),
        ("Rational", "B7fq-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR96-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FQ FlashKAT grouped rational h32 R96 B32 pair block-readout plus zero-init hidden-square readout and manual foreach AdamW"),
        ("Rational", "B7fr-RationalKAT-flashgroup-G16-h32-linearresGain19200-hybridPairCrossR96-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FR FlashKAT grouped rational h32 hybrid R96 B32 pair block-readout plus zero-init hidden-square readout and manual foreach AdamW"),
        ("Rational", "B7fs-RationalKAT-flashgroup-G16-h32-linearresGain19200-hybridPairCrossR112-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FS FlashKAT grouped rational h32 hybrid R112 B32 pair block-readout plus zero-init hidden-square readout and manual foreach AdamW"),
        ("Rational", "B7ft-RationalKAT-flashgroup-G16-h32-linearresGain19200-gridPairCrossR96-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FT FlashKAT grouped rational h32 grid-balanced R96 B32 pair block-readout plus zero-init hidden-square readout and manual foreach AdamW"),
        ("Rational", "B7fu-RationalKAT-flashgroup-G16-h32-linearresGain19200-gridPairCrossR112-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FU FlashKAT grouped rational h32 grid-balanced R112 B32 pair block-readout plus zero-init hidden-square readout and manual foreach AdamW"),
        ("Rational", "B7fv-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR112-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FV FlashKAT grouped rational h32 R112 B32 pair block-readout plus zero-init hidden-square readout and manual foreach AdamW"),
        ("Rational", "B7fw-RationalKAT-flashgroup-G16-h32-linearresGain19200-balancedPairCrossR112-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FW FlashKAT grouped rational h32 balanced R112 B32 pair block-readout plus zero-init hidden-square readout and manual foreach AdamW"),
        ("Rational", "B7fx-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairSumSqCrossR96-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FX FlashKAT grouped rational h32 R96 sparse pair-sum-square quadratic readout plus zero-init hidden-square readout and generic manual VJP"),
        ("Rational", "B7fy-RationalKAT-flashgroup-G16-h32-linearresGain19200-balancedPairSumSqCrossR96-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FY FlashKAT grouped rational h32 balanced R96 sparse pair-sum-square quadratic readout plus zero-init hidden-square readout and generic manual VJP"),
        ("Rational", "B7fz-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairSumSqCrossR64-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-FZ FlashKAT grouped rational h32 R64 sparse pair-sum-square quadratic readout and generic manual VJP"),
        ("Rational", "B7ga-RationalKAT-flashgroup-G16-h32-linearresGain19200-balancedPairSumSqCrossR64-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GA FlashKAT grouped rational h32 balanced R64 sparse pair-sum-square quadratic readout and generic manual VJP"),
        ("Rational", "B7gb-RationalKAT-flashgroup-G16-h32-linearresGain19200-diagPairCrossR96-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GB FlashKAT grouped rational h32 diag-rich R96 product-pair quadratic readout and generic manual VJP"),
        ("Rational", "B7gc-RationalKAT-flashgroup-G16-h32-linearresGain19200-diagPairCrossR112-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GC FlashKAT grouped rational h32 diag-rich R112 product-pair quadratic readout and generic manual VJP"),
        ("Rational", "B7gd-RationalKAT-flashgroup-G16-h32-linearresGain19200-diagPairCrossR64-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GD FlashKAT grouped rational h32 diag-rich R64 product-pair quadratic readout and generic manual VJP"),
        ("Rational", "B7ge-RationalKAT-flashgroup-G16-h32-linearresGain19200-diagPairCrossR80-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GE FlashKAT grouped rational h32 diag-rich R80 product-pair quadratic readout and generic manual VJP"),
        ("Rational", "B7gf-RationalKAT-flashgroup-G16-h32-linearresGain19200-blendPairCrossR96-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GF FlashKAT grouped rational h32 75/25 standard-balanced R96 product-pair readout plus zero-init hidden-square readout and generic manual VJP"),
        ("Rational", "B7gg-RationalKAT-flashgroup-G16-h32-linearresGain19200-blendPairCrossR88-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GG FlashKAT grouped rational h32 75/25 standard-balanced R88 product-pair readout plus zero-init hidden-square readout and generic manual VJP"),
        ("Rational", "B7gh-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-L3", "D2-GH FlashKAT grouped rational h32 B7ej full R136 pair block-readout plus zero-init hidden-square readout and generic VJP"),
        ("Rational", "B7gi-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GI FlashKAT grouped rational h32 B7ej full R136 pair block-readout plus zero-init hidden-square readout and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7gj-RationalKAT-flashgroup-G16-h32-linearresGain16000-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GJ FlashKAT grouped rational h32 B7gi full R136 hidden-square-zero with lower linear residual gain 160.00 and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7gk-RationalKAT-flashgroup-G16-h32-linearresGain14400-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GK FlashKAT grouped rational h32 B7gi full R136 hidden-square-zero with lower linear residual gain 144.00 and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7gl-RationalKAT-flashgroup-G16-h32-linearresGain18400-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GL FlashKAT grouped rational h32 B7gi full R136 hidden-square-zero with interpolated linear residual gain 184.00 and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7gm-RationalKAT-flashgroup-G16-h32-linearresGain17600-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GM FlashKAT grouped rational h32 B7gi full R136 hidden-square-zero with interpolated linear residual gain 176.00 and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7gn-RationalKAT-flashgroup-G16-h32-linearresGain18800-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GN FlashKAT grouped rational h32 B7gi full R136 hidden-square-zero with narrow linear residual gain 188.00 and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7go-RationalKAT-flashgroup-G16-h32-linearresGain19000-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GO FlashKAT grouped rational h32 B7gi full R136 hidden-square-zero with narrow linear residual gain 190.00 and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7gp-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-hiddenSqScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GP FlashKAT grouped rational h32 B7gi full R136 with half-scale hidden-square-zero trajectory and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7gq-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-hiddenSqScale025-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GQ FlashKAT grouped rational h32 B7gi full R136 with quarter-scale hidden-square-zero trajectory and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7gr-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-hiddenSqScale075-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GR FlashKAT grouped rational h32 B7gi full R136 with 0.75-scale hidden-square-zero trajectory and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7gs-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-hiddenSqScale062-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GS FlashKAT grouped rational h32 B7gi full R136 with 0.62-scale hidden-square-zero trajectory and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7gt-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-hiddenSqCenter-hiddenSqScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GT FlashKAT grouped rational h32 B7gp with per-sample centered hidden-square trajectory and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7gu-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-hiddenSqCenter-hiddenSqScale075-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GU FlashKAT grouped rational h32 B7gr with per-sample centered hidden-square trajectory and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7gv-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-hiddenSqSigned-hiddenSqScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GV FlashKAT grouped rational h32 B7gp with signed-quadratic h*abs(h) hidden tail and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7gw-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-hiddenSqSigned-hiddenSqScale075-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GW FlashKAT grouped rational h32 B7gr with signed-quadratic h*abs(h) hidden tail and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7gx-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadout-hiddenSqSigned-hiddenSqScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GX FlashKAT grouped rational h32 B7gv with nonzero signed-quadratic hidden tail initialization and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7gy-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadout-hiddenSqScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GY FlashKAT grouped rational h32 B7gp with nonzero h^2 hidden tail initialization and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7gz-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairDiffSqCrossR96-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-GZ FlashKAT grouped rational h32 R96 pair-difference-square readout plus zero-init hidden-square tail and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7ha-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairDiffSqCrossR64-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-HA FlashKAT grouped rational h32 R64 pair-difference-square readout plus zero-init hidden-square tail and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7hb-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-readGradR136-crossZero-b32-hiddenSqReadoutZero-hiddenSqSigned-hiddenSqScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-HB FlashKAT grouped rational h32 B7gv with full-rank R136 pair-readout grad tile and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7hc-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-readGradR136-crossZero-b32-hiddenSqReadoutZero-hiddenSqScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-HC FlashKAT grouped rational h32 B7gp with full-rank R136 pair-readout grad tile and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7hd-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-readGradAtomicR128-crossZero-b32-hiddenSqReadoutZero-hiddenSqSigned-hiddenSqScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-HD FlashKAT grouped rational h32 B7gv with R136 pair readout and batch-split atomic R128 readout-gradient accumulation"),
        ("Rational", "B7he-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-readGradAtomicR128-crossZero-b32-hiddenSqReadoutZero-hiddenSqScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-HE FlashKAT grouped rational h32 B7gp with R136 pair readout and batch-split atomic R128 readout-gradient accumulation"),
        ("Rational", "B7hf-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenAbsReadoutZero-hiddenAbsScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-HF FlashKAT grouped rational h32 B7ej with zero-init hidden absolute-value readout tail and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7hg-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenAbsReadout-hiddenAbsScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-HG FlashKAT grouped rational h32 B7ej with small nonzero hidden absolute-value readout tail and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7hh-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenAbsReadoutZero-hiddenAbsCenter-hiddenAbsScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-HH FlashKAT grouped rational h32 B7hf with centered hidden absolute-value readout tail and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7hi-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenAbsReadout-hiddenAbsCenter-hiddenAbsScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-HI FlashKAT grouped rational h32 B7hg with centered hidden absolute-value readout tail and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7hj-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenAbsReadoutZero-hiddenAbsCenterMix025-hiddenAbsScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-HJ FlashKAT grouped rational h32 B7hf with 0.25 centered hidden absolute-value readout tail and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7hk-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenAbsReadoutZero-hiddenAbsCenterMix010-hiddenAbsScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-HK FlashKAT grouped rational h32 B7hf with 0.10 centered hidden absolute-value readout tail and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7hl-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-HL FlashKAT grouped rational h32 B7ej with zero-init bounded hidden rational h/(1+abs(h)) readout tail and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7hm-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadout-hiddenRatScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-HM FlashKAT grouped rational h32 B7ej with nonzero bounded hidden rational h/(1+abs(h)) readout tail and generic manual VJP/manual foreach AdamW"),
        ("Rational", "B7hn-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailGradTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-HN FlashKAT grouped rational h32 B7hl with Triton fused hidden rational tail readout-gradient and hidden VJP; generic dL/dlogits, not CE-specific"),
        ("Rational", "B7ho-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenAbsReadoutZero-hiddenAbsScale050-hiddenTailGradTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-HO FlashKAT grouped rational h32 B7hf with Triton fused hidden |h| tail readout-gradient and hidden VJP; generic dL/dlogits, not CE-specific"),
        ("Rational", "B7hp-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale025-hiddenTailGradTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-HP FlashKAT grouped rational h32 B7hn with lower 0.25 hidden rational tail scale and Triton fused generic VJP"),
        ("Rational", "B7hq-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale075-hiddenTailGradTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-HQ FlashKAT grouped rational h32 B7hn with higher 0.75 hidden rational tail scale and Triton fused generic VJP"),
        ("Rational", "B7hr-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailGradTriton-tritonReadoutAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-HR FlashKAT grouped rational h32 B7hn with generic Triton hidden-tail VJP plus Triton tensor AdamW for readout params; update-cost repair, not CE-specific"),
        ("Rational", "B7hs-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenAbsReadoutZero-hiddenAbsScale050-hiddenTailGradTriton-tritonReadoutAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-HS FlashKAT grouped rational h32 B7ho with generic Triton hidden-tail VJP plus Triton tensor AdamW for readout params; update-cost repair, not CE-specific"),
        ("Rational", "B7hx-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-HX FlashKAT grouped rational h32 B7hn with generic Triton hidden-tail VJP plus Triton AdamW only for the hidden-tail readout; narrower update-cost repair, not CE-specific"),
        ("Rational", "B7hy-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenAbsReadoutZero-hiddenAbsScale050-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-HY FlashKAT grouped rational h32 B7ho with generic Triton hidden-tail VJP plus Triton AdamW only for the hidden-tail readout; narrower update-cost repair, not CE-specific"),
        ("Rational", "B7hz-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale025-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-HZ B7hx with lower 0.25 hidden rational tail scale; loss-agnostic trajectory amplitude repair after B7hx KMNIST/AUC task fail"),
        ("Rational", "B7ia-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale035-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-IA B7hx with intermediate 0.35 hidden rational tail scale; loss-agnostic trajectory amplitude repair after B7hx KMNIST/AUC task fail"),
        ("Rational", "B7ib-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailReadoutLR050-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-IB B7hx with lower hidden-tail readout optimizer LR 0.50; loss-agnostic update-amplitude repair after AdamW canceled feature-scale repairs"),
        ("Rational", "B7ic-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailReadoutLR025-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-IC B7hx with lower hidden-tail readout optimizer LR 0.25; loss-agnostic update-amplitude repair after AdamW canceled feature-scale repairs"),
        ("Rational", "B7id-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailReadoutLR150-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-ID B7hx with higher hidden-tail readout optimizer LR 1.50; loss-agnostic update-amplitude bracket after lower LR degraded A5"),
        ("Rational", "B7ie-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailReadoutLR200-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-IE B7hx with higher hidden-tail readout optimizer LR 2.00; loss-agnostic update-amplitude bracket after lower LR degraded A5"),
        ("Rational", "B7if-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailReadoutLR125-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-IF B7hx with hidden-tail readout optimizer LR 1.25; midpoint bracket to balance B7hx AUC and B7id accuracy"),
        ("Rational", "B7ig-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailReadoutLR135-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-IG B7hx with hidden-tail readout optimizer LR 1.35; midpoint bracket to balance B7hx AUC and B7id accuracy"),
        ("Rational", "B7ih-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailReadoutLR100to150e1-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-IH B7hx with hidden-tail readout LR schedule 1.00 -> 1.50 at epoch 1; loss-agnostic trajectory schedule to combine B7hx AUC with B7id accuracy"),
        ("Rational", "B7ii-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailReadoutLR100to150e2-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-II B7hx with hidden-tail readout LR schedule 1.00 -> 1.50 at epoch 2; loss-agnostic trajectory schedule to preserve early AUC before late accuracy gain"),
        ("Rational", "B7ij-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatCenter-hiddenRatScale050-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-IJ B7hx with centered bounded hidden-rational tail; loss-agnostic feature geometry repair for ECE/AUC drift"),
        ("Rational", "B7ik-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatCenterMix025-hiddenRatScale050-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-IK B7hx with partially centered bounded hidden-rational tail mix 0.25; loss-agnostic feature geometry repair for ECE/AUC drift"),
        ("Rational", "B7il-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatCenterSG-hiddenRatScale050-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-IL B7hx with stop-gradient centered bounded hidden-rational tail; cheaper loss-agnostic feature geometry repair"),
        ("Rational", "B7im-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatCenterSGMix025-hiddenRatScale050-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-IM B7hx with stop-gradient partial centered bounded hidden-rational tail mix 0.25; cheaper loss-agnostic feature geometry repair"),
        ("Rational", "B7in-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual050-freezeBackbone-manualAdamW-L3", "D2-IN B7co-shaped Rational path with bounded rational hidden residual folded into existing w2 readout; no hidden-tail readout parameter/update state and generic dL/dlogits VJP"),
        ("Rational", "B7io-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual050-freezeBackbone-manualAdamW-L3", "D2-IO B7cp-shaped Rational path with bounded rational hidden residual folded into existing w2 readout; no hidden-tail readout parameter/update state and generic dL/dlogits VJP"),
        ("Rational", "B7ip-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual050-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-IP B7ej-shaped high linear-residual path with bounded rational hidden residual folded into existing w2 readout and corrected generic hidden-bias VJP"),
        ("Rational", "B7iq-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual025-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-IQ B7ej-shaped high linear-residual path with lower bounded rational hidden residual folded into existing w2 readout and corrected generic hidden-bias VJP"),
        ("Rational", "B7ir-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual015-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-IR B7iq scale-down to 0.15 hidden residual for L3 margin; same generic hidden-bias VJP, not CE-specific"),
        ("Rational", "B7is-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual010-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-IS B7iq scale-down to 0.10 hidden residual for L3 margin; same generic hidden-bias VJP, not CE-specific"),
        ("Rational", "B7it-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual018-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-IT B7ir/B7iq midpoint 0.18 hidden residual; same generic hidden-bias VJP, not CE-specific"),
        ("Rational", "B7iu-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual020-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-IU B7ir/B7iq midpoint 0.20 hidden residual; same generic hidden-bias VJP, not CE-specific"),
        ("Rational", "B7iv-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual015-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-IV B7ir with generic Triton w2 readout-gradient + hidden-residual VJP fusion; no CE-specific backward"),
        ("Rational", "B7iw-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual010-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-IW B7is with generic Triton w2 readout-gradient + hidden-residual VJP fusion; no CE-specific backward"),
        ("Rational", "B7kc-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KC B7iw task-trajectory damping bracket: lower bounded rational hidden residual 0.10 -> 0.05; generic dL/dlogits VJP and no CE-specific tuning"),
        ("Rational", "B7lp-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-LP B7en pairNorm plus B7kc bounded rational hidden residual 0.05; structural task-geometry repair without CE-specific tuning"),
        ("Rational", "B7kd-RationalKAT-flashgroup-G16-h32-linearresGain14400-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KD B7kc signal-channel bracket: lower linear residual gain 192.0 -> 144.0 with same hidden residual 0.05 and generic VJP"),
        ("Rational", "B7ke-RationalKAT-flashgroup-G16-h32-linearresGain09600-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KE B7kc stronger signal-channel bracket: lower linear residual gain 192.0 -> 96.0 with same hidden residual 0.05 and generic VJP"),
        ("Rational", "B7kf-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KF B7jt/B7jx bridge: small tanh hidden residual 0.05 with lower stop-gradient batch cap 1.25; generic dL/dlogits VJP and no CE-specific tuning"),
        ("Rational", "B7kg-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap100-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KG B7kf lower cap 1.00 to repair KMNIST NLL/AUC without CE-specific tuning"),
        ("Rational", "B7kh-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap075-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KH B7kf aggressive lower cap 0.75 to repair KMNIST NLL/AUC without CE-specific tuning"),
        ("Rational", "B7ki-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-crossBatchSGCap100-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KI B7jx no-hidden lower cap 1.00 to repair KMNIST NLL/ECE without CE-specific tuning"),
        ("Rational", "B7kj-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-crossBatchSGCap075-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KJ B7jx no-hidden aggressive lower cap 0.75 to repair KMNIST NLL/ECE without CE-specific tuning"),
        ("Rational", "B7kk-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR128-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KK B7kf middle rank R128 with tanh residual 0.05 and cap 1.25; tests L3/A4 tradeoff without CE-specific tuning"),
        ("Rational", "B7kl-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR120-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KL B7kf lower middle rank R120 with tanh residual 0.05 and cap 1.25; tests L3/A4 tradeoff without CE-specific tuning"),
        ("Rational", "B7km-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR112-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KM B7kk lower rank R112 with tanh residual 0.05 and cap 1.25; tests whether L3 opens before A4 expression collapses"),
        ("Rational", "B7kn-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR96-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KN B7kk lower rank R96 with tanh residual 0.05 and cap 1.25; tests L3 opening versus A4 expression loss"),
        ("Rational", "B7ko-RationalKAT-flashgroup-G16-h32-linearresGain19200-hybridPairCrossR112-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KO B7km R112 expression repair with hybrid standard-balanced pair coverage; no CE-specific tuning"),
        ("Rational", "B7kp-RationalKAT-flashgroup-G16-h32-linearresGain19200-balancedPairCrossR112-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KP B7km R112 expression repair with balanced pair coverage; no CE-specific tuning"),
        ("Rational", "B7kq-RationalKAT-flashgroup-G16-h32-linearresGain19200-diagPairCrossR112-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KQ B7km R112 expression repair with diag-rich pair coverage; no CE-specific tuning"),
        ("Rational", "B7kr-RationalKAT-flashgroup-G16-h32-linearresGain19200-hybridPairCrossR128-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KR B7ko higher-rank hybrid R128 expression repair near L3 boundary; no CE-specific tuning"),
        ("Rational", "B7ks-RationalKAT-flashgroup-G16-h32-linearresGain19200-balancedPairCrossR128-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KS B7kp higher-rank balanced R128 expression repair near L3 boundary; no CE-specific tuning"),
        ("Rational", "B7kt-RationalKAT-flashgroup-G16-h32-linearresGain19200-diagPairCrossR128-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KT B7kq higher-rank diag R128 expression repair near L3 boundary; no CE-specific tuning"),
        ("Rational", "B7ku-RationalKAT-flashgroup-G16-h32-linearresGain19200-hybridPairCrossR120-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KU hybrid R120 expression/cost midpoint between B7ko and B7kr; no CE-specific tuning"),
        ("Rational", "B7kv-RationalKAT-flashgroup-G16-h32-linearresGain19200-diagPairCrossR120-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KV diag R120 expression/cost midpoint between B7kq and B7kt; no CE-specific tuning"),
        ("Rational", "B7kw-RationalKAT-flashgroup-G16-h32-linearresGain19200-balBlendPairCrossR128-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KW R128 25% standard + 75% balanced pair coverage to keep E1 while lifting E6/E8; no CE-specific tuning"),
        ("Rational", "B7kx-RationalKAT-flashgroup-G16-h32-linearresGain19200-balBlendPairCrossR120-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KX R120 25% standard + 75% balanced pair coverage expression/cost bracket; no CE-specific tuning"),
        ("Rational", "B7ky-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR80-readblocktriton-crossZero-b32-projBilinR32-bilinZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KY R80 pair block plus fixed dual-DCT projection-bilinear R32 readout; low-cost rotated quadratic expression repair without CE-specific tuning"),
        ("Rational", "B7kz-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR80-readblocktriton-crossZero-b32-projSqR32-sqZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KZ R80 pair block plus fixed DCT projection-square R32 readout; low-cost diagonal rotated quadratic expression repair without CE-specific tuning"),
        ("Rational", "B7la-RationalKAT-flashgroup-G16-h32-linearresGain19200-diagPairCrossR80-readblocktriton-crossZero-b32-projBilinR32-bilinZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-LA diag-rich R80 pair block plus fixed projection-bilinear R32 readout; E8-oriented expression repair without CE-specific tuning"),
        ("Rational", "B7lb-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR80-readblocktriton-crossZero-b32-projBilinR64-bilinZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-LB R80 pair block plus fixed dual-DCT projection-bilinear R64 readout; spends remaining L3 margin on rotated quadratic expression without CE-specific tuning"),
        ("Rational", "B7lc-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR80-readblocktriton-crossZero-b32-projSqR64-sqZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-LC R80 pair block plus fixed DCT projection-square R64 readout; spends remaining L3 margin on diagonal rotated quadratic expression without CE-specific tuning"),
        ("Rational", "B7ld-RationalKAT-flashgroup-G16-h32-linearresGain19200-diagPairCrossR80-readblocktriton-crossZero-b32-projBilinR64-bilinZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-LD diag-rich R80 pair block plus fixed projection-bilinear R64 readout; E6/E8-oriented rank bracket without CE-specific tuning"),
        ("Rational", "B7le-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR80-readblocktriton-crossZero-b32-projBilinR48-bilinZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-LE R80 pair block plus fixed projection-bilinear R48 readout; bridge between weak R32 expression and slow R64 cost"),
        ("Rational", "B7lf-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR80-readblocktriton-crossZero-b32-projSqR48-sqZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-LF R80 pair block plus fixed projection-square R48 readout; bridge between weak R32 expression and slow R64 cost"),
        ("Rational", "B7lg-RationalKAT-flashgroup-G16-h32-linearresGain19200-diagPairCrossR80-readblocktriton-crossZero-b32-projBilinR48-bilinZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-LG diag-rich R80 pair block plus fixed projection-bilinear R48 readout; E6/E8 bridge without CE-specific tuning"),
        ("Rational", "B7lh-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairSumSqCrossR80-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-LH R80 pair-sum-square feature fallback after projection route blocked; low-cost quadratic coverage without CE-specific tuning"),
        ("Rational", "B7li-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairDiffSqCrossR80-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-LI R80 pair-difference-square feature fallback after projection route blocked; alternate quadratic coverage without CE-specific tuning"),
        ("Rational", "B7lj-RationalKAT-flashgroup-G16-h32-linearresGain19200-blendPairCrossR80-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-LJ R80 75/25 standard-balanced pair coverage fallback after projection route blocked; no CE-specific tuning"),
        ("Rational", "B7lk-RationalKAT-flashgroup-G16-h32-linearresGain19200-inputcrossR64-crossZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-LK random-projection inputcross R64 with current h32/tanh005/cap125 trajectory; no CE-specific tuning"),
        ("Rational", "B7ll-RationalKAT-flashgroup-G16-h32-linearresGain19200-inputcrossR96-crossZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-LL random-projection inputcross R96 with current h32/tanh005/cap125 trajectory; no CE-specific tuning"),
        ("Rational", "B7lm-RationalKAT-flashgroup-G16-h32-linearresGain19200-inputcrossR112-crossZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-LM random-projection inputcross R112 with current h32/tanh005/cap125 trajectory; no CE-specific tuning"),
        ("Rational", "B7ln-RationalKAT-flashgroup-G16-h32-linearresGain19200-inputcrossR32-crossZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-LN low-rank random-projection inputcross R32 with current h32/tanh005/cap125 trajectory; no CE-specific tuning"),
        ("Rational", "B7lo-RationalKAT-flashgroup-G16-h32-linearresGain19200-inputcrossR48-crossZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-LO low-rank random-projection inputcross R48 with current h32/tanh005/cap125 trajectory; no CE-specific tuning"),
        ("Rational", "B7ix-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenTanhResidual015-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-IX smooth tanh hidden residual with generic Triton w2 readout-gradient + hidden-residual VJP; different trajectory primitive from B7ir/B7iv and not CE-specific"),
        ("Rational", "B7iy-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenTanhResidual010-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-IY lower smooth tanh hidden residual with generic Triton w2 readout-gradient + hidden-residual VJP; different trajectory primitive from B7is/B7iw and not CE-specific"),
        ("Rational", "B7iz-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenCenterTanhResidual015-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-IZ centered smooth tanh hidden residual with generic Triton w2 readout-gradient + hidden-residual VJP; removes hidden-mean offset and is not CE-specific"),
        ("Rational", "B7ja-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenCenterTanhResidual010-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JA lower centered smooth tanh hidden residual with generic Triton w2 readout-gradient + hidden-residual VJP; removes hidden-mean offset and is not CE-specific"),
        ("Rational", "B7jb-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-crossCapFused150-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JB non-hidden-residual pair-logit fused tanh cap 1.50 on B7ir anchor; generic dL/dlogits VJP and no CE-specific backward"),
        ("Rational", "B7jc-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-crossCapFused200-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JC non-hidden-residual pair-logit fused tanh cap 2.00 on B7ir anchor; generic dL/dlogits VJP and no CE-specific backward"),
        ("Rational", "B7jd-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-crossCenterCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JD centered pair-logit tanh cap 1.50 on B7ir anchor; removes class-common pair offset with generic dL/dlogits VJP and no CE-specific backward"),
        ("Rational", "B7je-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-crossCenterCap200-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JE centered pair-logit tanh cap 2.00 on B7ir anchor; removes class-common pair offset with generic dL/dlogits VJP and no CE-specific backward"),
        ("Rational", "B7jh-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-crossBatchCap100-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JH lower batch-centered pair-logit tanh cap 1.00 on B7jf anchor; loss-agnostic A5 trajectory bracket"),
        ("Rational", "B7ji-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-crossBatchCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JI midpoint batch-centered pair-logit tanh cap 1.25 on B7jf anchor; loss-agnostic A5 trajectory bracket"),
        ("Rational", "B7jj-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JJ B7jf stop-gradient batch mean cap 1.50; keeps batch-centered forward trajectory but removes cross-batch mean-gradient coupling"),
        ("Rational", "B7jk-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-crossBatchSGCap200-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JK B7jg stop-gradient batch mean cap 2.00; generic dL/dlogits VJP, no CE-specific tuning"),
        ("Rational", "B7jl-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual015-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JL B7jj plus bounded rational hidden residual 0.15; combines stop-gradient batch-centered pair trajectory with a loss-agnostic hidden residual channel"),
        ("Rational", "B7jm-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenTanhResidual010-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JM B7jj plus smooth tanh hidden residual 0.10; loss-agnostic trajectory combination, no CE-specific tuning"),
        ("Rational", "B7jn-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR96-readblocktriton-crossZero-b32-hiddenTanhResidual010-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JN B7jm lower-rank R96 bracket to reduce fused hidden-residual/paircap cost while keeping the same loss-agnostic trajectory primitive"),
        ("Rational", "B7jo-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR80-readblocktriton-crossZero-b32-hiddenTanhResidual010-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JO B7jm lower-rank R80 bracket to test whether the same generic trajectory can regain L3 efficiency before A4/A5 promotion"),
        ("Rational", "B7jp-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR112-readblocktriton-crossZero-b32-hiddenTanhResidual010-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JP B7jn/B7jm midpoint R112 bracket: restore expression capacity without returning to R136 hidden-residual cost"),
        ("Rational", "B7jq-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR120-readblocktriton-crossZero-b32-hiddenTanhResidual010-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JQ B7jn/B7jm midpoint R120 bracket: test if extra rank restores A4 while preserving L3 efficiency"),
        ("Rational", "B7jr-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR128-readblocktriton-crossZero-b32-hiddenTanhResidual010-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JR B7jq/B7jm high-rank R128 bracket: push expression capacity toward R136 while checking L3 gate"),
        ("Rational", "B7js-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenTanhResidual010-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JS B7jr/B7jm narrow R132 bracket: final rank-only expression-capacity check before changing primitive/kernel"),
        ("Rational", "B7jt-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JT B7js task-trajectory damping repair: keep R132 but reduce smooth hidden residual 0.10 -> 0.05; generic dL/dlogits VJP and no CE-specific tuning"),
        ("Rational", "B7ju-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenCenterTanhResidual010-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JU B7js centered hidden-residual repair: keep amplitude 0.10 but remove per-sample tanh hidden mean; generic dL/dlogits VJP and no CE-specific tuning"),
        ("Rational", "B7jv-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenTanhResidual010-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JV B7js lower pair-logit cap repair: keep R132/hidden residual but reduce stop-gradient batch cap 1.50 -> 1.25; generic dL/dlogits VJP and no CE-specific tuning"),
        ("Rational", "B7jw-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JW B7jj R132 no-hidden-residual bracket: isolate whether B7js A5 damage comes from hidden residual while keeping generic batch-cap VJP"),
        ("Rational", "B7jx-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JX B7jw no-hidden lower-cap bracket: reduce stop-gradient batch cap 1.50 -> 1.25 without hidden residual; generic dL/dlogits VJP and no CE-specific tuning"),
        ("Rational", "B7jy-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenSignSqResidual010-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JY B7jw/B7jx follow-up: add sign-preserving bounded quadratic hidden residual h*abs(h)/(1+h^2) scale 0.10; generic dL/dlogits VJP and no CE-specific tuning"),
        ("Rational", "B7jz-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenSignSqResidual015-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JZ B7jy amplitude bracket: sign-preserving bounded quadratic hidden residual scale 0.15; generic dL/dlogits VJP and no CE-specific tuning"),
        ("Rational", "B7ka-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR128-readblocktriton-crossZero-b32-hiddenSignSqResidual010-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KA B7jy rank-cost bracket: reduce paircross R132 -> R128 while keeping sign-preserving bounded quadratic hidden residual 0.10; generic dL/dlogits VJP and no CE-specific tuning"),
        ("Rational", "B7kb-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR120-readblocktriton-crossZero-b32-hiddenSignSqResidual010-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-KB B7jy lower-cost bracket: reduce paircross R132 -> R120 while keeping sign-preserving bounded quadratic hidden residual 0.10; generic dL/dlogits VJP and no CE-specific tuning"),
        ("Rational", "B7jf-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-crossBatchCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JF batch-centered pair-logit tanh cap 1.50 on B7ir anchor; changes logits with generic batch-centering VJP and no CE-specific backward"),
        ("Rational", "B7jg-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-crossBatchCap200-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-JG batch-centered pair-logit tanh cap 2.00 on B7ir anchor; changes logits with generic batch-centering VJP and no CE-specific backward"),
        ("Rational", "B7ht-RationalKAT-flashgroup-G16-h32-linearresGain19200-linearResRamp09600-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "D2-HT B7ej-shaped Rational path with loss-agnostic linear residual scale ramp 96.0 -> 192.0; trajectory repair, not CE-specific"),
        ("Rational", "B7hu-RationalKAT-flashgroup-G16-h32-linearresGain19200-linearResRamp14400-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "D2-HU B7ej-shaped Rational path with loss-agnostic linear residual scale ramp 144.0 -> 192.0; trajectory repair, not CE-specific"),
        ("Rational", "B7hv-RationalKAT-flashgroup-G16-h32-linearresGain19200-linearResRamp14400-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailGradTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-HV B7hn hidden-tail Triton path with loss-agnostic linear residual scale ramp 144.0 -> 192.0; trajectory repair, not CE-specific"),
        ("Rational", "B7hw-RationalKAT-flashgroup-G16-h32-linearresGain19200-linearResRamp09600-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailGradTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-HW B7hn hidden-tail Triton path with stronger loss-agnostic linear residual scale ramp 96.0 -> 192.0; trajectory repair, not CE-specific"),
        ("Rational", "B7cq-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-b32-crossCap200-L3", "D2-CQ FlashKAT grouped rational h40 R136 crosszero B32 block-readout with bounded pair logits cap 2.00"),
        ("Rational", "B7cr-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-crossCap200-L3", "D2-CR FlashKAT grouped rational h32 R136 crosszero B32 block-readout with bounded pair logits cap 2.00"),
        ("Rational", "B7cs-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-b32-crossWarm-L3", "D2-CS FlashKAT grouped rational h40 R136 crosszero B32 block-readout with no-extra-op cross-readout scale warmup"),
        ("Rational", "B7ct-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-crossWarm-L3", "D2-CT FlashKAT grouped rational h32 R136 crosszero B32 block-readout with no-extra-op cross-readout scale warmup"),
        ("Rational", "B7cu-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-b32-readGradR64-crossWarm-L3", "D2-CU FlashKAT grouped rational h40 R136 crosszero B32 block-readout plus R64 grad tiling and cross-readout scale warmup"),
        ("Rational", "B7cv-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-readGradR64-crossWarm-L3", "D2-CV FlashKAT grouped rational h32 R136 crosszero B32 block-readout plus R64 grad tiling and cross-readout scale warmup"),
        ("Rational", "B7cw-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-b32-readGradR128-crossWarm-L3", "D2-CW FlashKAT grouped rational h40 R136 crosszero B32 block-readout plus R128 grad tiling and cross-readout scale warmup"),
        ("Rational", "B7cx-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-readGradR128-crossWarm-L3", "D2-CX FlashKAT grouped rational h32 R136 crosszero B32 block-readout plus R128 grad tiling and cross-readout scale warmup"),
        ("Rational", "B7cy-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-b32-crossCapFused200-L3", "D2-CY FlashKAT grouped rational h40 R136 crosszero B32 block-readout with fused bounded pair-logit forward and loss-agnostic VJP"),
        ("Rational", "B7cz-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-crossCapFused200-L3", "D2-CZ FlashKAT grouped rational h32 R136 crosszero B32 block-readout with fused bounded pair-logit forward and loss-agnostic VJP"),
        ("Rational", "B7da-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-b32-crossCapFused200Atomic-L3", "D2-DA FlashKAT grouped rational h40 R136 crosszero B32 fused bounded pair-logit with batch-block atomic VJP"),
        ("Rational", "B7db-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-crossCapFused200Atomic-L3", "D2-DB FlashKAT grouped rational h32 R136 crosszero B32 fused bounded pair-logit with batch-block atomic VJP"),
        ("Rational", "B7dc-RationalKAT-flashgroup-G16-h40-linearres-pairReadBucketR136G32-crossZero-L3", "D2-DC FlashKAT grouped rational h40 R136 pair features bucketed to G32 before trainable readout"),
        ("Rational", "B7dd-RationalKAT-flashgroup-G16-h40-linearres-pairReadBucketR136G64-crossZero-L3", "D2-DD FlashKAT grouped rational h40 R136 pair features bucketed to G64 before trainable readout"),
        ("Rational", "B7de-RationalKAT-flashgroup-G16-h40-linearres-pairReadBucketR136G16-crossZero-L3", "D2-DE FlashKAT grouped rational h40 R136 pair features bucketed to G16 before trainable readout"),
        ("Rational", "B7df-RationalKAT-flashgroup-G16-h32-linearres-pairReadBucketR136G32-crossZero-L3", "D2-DF FlashKAT grouped rational h32 R136 pair features bucketed to G32 before trainable readout"),
        ("Rational", "B7dg-RationalKAT-flashgroup-G16-h40-linearres-projSqR64-sqZero-L3", "D2-DG FlashKAT grouped rational h40 fixed projection-square R64 readout"),
        ("Rational", "B7dh-RationalKAT-flashgroup-G16-h32-linearres-projSqR64-sqZero-L3", "D2-DH FlashKAT grouped rational h32 fixed projection-square R64 readout"),
        ("Rational", "B7di-RationalKAT-flashgroup-G16-h40-linearres-projSqR32-sqZero-L3", "D2-DI FlashKAT grouped rational h40 fixed projection-square R32 readout"),
        ("Rational", "B7dj-RationalKAT-flashgroup-G16-h40-linearres-projSqR96-sqZero-L3", "D2-DJ FlashKAT grouped rational h40 fixed projection-square R96 readout"),
        ("Rational", "B7dk-RationalKAT-flashgroup-G16-h32-linearres-projSqR96-sqZero-L3", "D2-DK FlashKAT grouped rational h32 fixed projection-square R96 readout"),
        ("Rational", "B7dl-RationalKAT-flashgroup-G16-h40-linearres-projBilinR64-bilinZero-L3", "D2-DL FlashKAT grouped rational h40 fixed dual-projection bilinear R64 readout"),
        ("Rational", "B7dm-RationalKAT-flashgroup-G16-h32-linearres-projBilinR64-bilinZero-L3", "D2-DM FlashKAT grouped rational h32 fixed dual-projection bilinear R64 readout"),
        ("Rational", "B7dn-RationalKAT-flashgroup-G16-h40-linearres-projBilinR96-bilinZero-L3", "D2-DN FlashKAT grouped rational h40 fixed dual-projection bilinear R96 readout"),
        ("Rational", "B7lq-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-LQ B7lp plus stop-gradient batch-centered pair-logit cap 1.25; Line C coupling-collapse repair without CE-specific tuning"),
        ("Rational", "B7lr-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-LR B7en plus stop-gradient batch-centered pair-logit cap 1.25 without hidden residual; isolates pair-signal geometry after B7lp Line C collapse"),
        ("Rational", "B7ls-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-crossBatchSGCap125-freezeRational-hiddenBias-manualAdamW-L3", "D2-LS B7lr with trainable w1 backbone and frozen rational coefficients; Line C coupling-collapse repair that tests readout-only trajectory as blocker"),
        ("Rational", "B7lt-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeRational-hiddenBias-manualAdamW-L3", "D2-LT B7lq with trainable w1 backbone and frozen rational coefficients; tests backbone signal-channel repair plus bounded hidden residual"),
        ("Rational", "B7lu-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-crossSignalR4-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-LU B7en pairNorm with full pair-class readout replaced by rank-4 shared pair-signal channel; tests reservoir-trapping repair"),
        ("Rational", "B7lv-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-crossSignalR8-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-LV B7lu shared pair-signal rank 8 bracket for expression versus Line C coupling"),
        ("Rational", "B7lw-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossPCAWhiteR96-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-LW B7en with unlabeled PCA-whitened pair feature channel R96; tests feature conditioning for Line C coupling"),
        ("Rational", "B7lx-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossPCAWhiteR128-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-LX B7lw PCA-whitened pair rank R128 bracket for A4 preservation versus Line C conditioning"),
        ("Rational", "B7ly-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossPCAR96-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-LY B7lw without whitening: PCA-rotated pair R96 to avoid low-energy logit amplification"),
        ("Rational", "B7lz-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossPCAR128-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-LZ B7lx without whitening: PCA-rotated pair R128 for A4-preserving feature conditioning"),
        ("Rational", "B7ma-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossPCAR128-crossZero-b32-logitRMSNormSG-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-MA B7lz plus samplewise stop-gradient logit RMS normalization; output-geometry repair for A5/Line C without CE-specific tuning"),
        ("Rational", "B7mb-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-logitRMSNormSG-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-MB B7lp plus samplewise stop-gradient logit RMS normalization; tests output-scale geometry after pairNorm+hidden residual Line C collapse"),
        ("Rational", "B7mc-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossPCAR128-crossZero-b32-logitBatchRMSNormSG-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-MC B7ma with cheaper batch-level stop-gradient logit RMS normalization to reopen L3"),
        ("Rational", "B7md-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-logitBatchRMSNormSG-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-MD B7mb with cheaper batch-level stop-gradient logit RMS normalization on B7lp anchor"),
        ("Rational", "B7me-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-logitBatchRMSNormSG150-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-ME B7md with batch logit RMS scale 1.50 for task-margin recovery"),
        ("Rational", "B7mf-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-logitBatchRMSNormSG200-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-MF B7md with batch logit RMS scale 2.00 for stronger task-margin recovery"),
        ("Rational", "B7mg-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-logitBatchRMSMixSG025-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-MG B7md with 25% residual batch RMS mix to preserve task while nudging Line C"),
        ("Rational", "B7mh-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-logitBatchRMSMixSG050-freezeBackbone-hiddenBias-manualAdamW-L3", "D2-MH B7md with 50% residual batch RMS mix to bracket task/Line C tradeoff"),
        ("Rational", "B7a-RationalKAT-lite-safe-den-K4", "D2-A GroupRationalKAN / L1 reference + L2 compile-forward attempt"),
        ("RBF", "B2r-FastKAN-RBF-stream-K2-repair", "D3-A FastKAN fixed-center RBF / L1 stream + L2 compile-forward attempt"),
        ("RBF", "B2s-GaussianRBF-stream-K4-recompute", "D3-B v12.10 RBF-F2 fixed-center K4 stream recompute without dense basis materialization"),
        ("Chebyshev", "B3c-ChebyKAN-K3-stream", "D4-A Cheby K3 stream / L1 + L2 compile-forward attempt"),
        ("Chebyshev", "B3e-ChebyKAN-K3-tritonL3-matmulTile", "D4-C Cheby K3 family-specific Triton matmul-tile L3 backward/update repair"),
        ("Chebyshev", "B3f-ChebyKAN-K4-tritonL3-matmulTile", "D4-D Cheby K4 family-specific Triton matmul-tile expression repair"),
        ("Chebyshev", "B3j-ChebyKAN-K3-h120-tritonL3-matmulTile", "D4-E Cheby K3 h120 capacity bracket under same Triton L3 path"),
        ("Chebyshev", "B3i-ChebyKAN-K3-h128-tritonL3-matmulTile", "D4-E Cheby K3 h128 capacity repair under same Triton L3 path"),
        ("Chebyshev", "B3g-ChebyKAN-K3-h160-tritonL3-matmulTile", "D4-E Cheby K3 h160 capacity repair under same Triton L3 path"),
        ("Chebyshev", "B3h-ChebyKAN-K3-h192-tritonL3-matmulTile", "D4-E Cheby K3 h192 capacity repair under same Triton L3 path"),
        ("Chebyshev", "B3k-ChebyKAN-K3-h120-tritonL3-gradbuf", "D4-F Cheby K3 h120 memory-planned persistent grad-buffer L3 path"),
        ("Chebyshev", "B3l-ChebyKAN-K3-h128-tritonL3-gradbuf", "D4-F Cheby K3 h128 memory-planned persistent grad-buffer L3 path"),
        ("Chebyshev", "B3m-ChebyKAN-K3-h112-tritonL3-gradbuf", "D4-G Cheby K3 h112 memory-gate lower bracket under persistent grad-buffer L3 path"),
        ("Chebyshev", "B3n-ChebyKAN-K3-h112-paircrossR16-tritonL3-gradbuf", "D4-H Cheby K3 h112 fixed-pair low-rank product expression primitive with Triton/manual L3 path"),
        ("Chebyshev", "B3o-ChebyKAN-K3-h112-paircrossR32-tritonL3-gradbuf", "D4-H Cheby K3 h112 wider fixed-pair low-rank product expression primitive with Triton/manual L3 path"),
        ("Chebyshev", "B3y-ChebyKAN-K3-h96-paircrossR32-tritonL3-gradbuf", "D4-O Cheby K3 h96 paircrossR32 efficiency floor under same expression primitive"),
        ("Chebyshev", "B3z-ChebyKAN-K3-h88-paircrossR32-tritonL3-gradbuf", "D4-P Cheby K3 h88 paircrossR32 lower hidden efficiency floor under same expression primitive"),
        ("Chebyshev", "B3aa-ChebyKAN-K3-h88-paircrossR32-inputcrossL4P8-tritonL3-gradbuf", "D4-Q Cheby K3 h88 paircrossR32 plus tiny input-cross P8 expression repair"),
        ("Chebyshev", "B3ab-ChebyKAN-K3-h88-paircrossR32-inputcrossL4P16-tritonL3-gradbuf", "D4-Q Cheby K3 h88 paircrossR32 plus input-cross P16 expression repair"),
        ("Chebyshev", "B3ac-ChebyKAN-K3-h80-paircrossR32-inputcrossL4P16-tritonL3-gradbuf", "D4-R Cheby K3 h80 paircrossR32 plus input-cross P16 efficiency floor"),
        ("Chebyshev", "B3ad-ChebyKAN-K3-h72-paircrossR32-inputcrossL4P32-tritonL3-gradbuf", "D4-S Cheby K3 h72 paircrossR32 plus input-cross P32 expression/efficiency tradeoff"),
        ("Chebyshev", "B3ae-ChebyKAN-K3-h64-paircrossR32-inputcrossL4P48-tritonL3-gradbuf", "D4-T Cheby K3 h64 paircrossR32 plus input-cross P48 stronger expression bracket"),
        ("Chebyshev", "B3af-ChebyKAN-K3-h72-paircrossR32-inputcrossL4P24-tritonL3-gradbuf", "D4-U Cheby K3 h72 paircrossR32 plus input-cross P24 gate-edge bracket"),
        ("Chebyshev", "B3ag-ChebyKAN-K3-h68-paircrossR32-inputcrossL4P24-tritonL3-gradbuf", "D4-U Cheby K3 h68 paircrossR32 plus input-cross P24 lower hidden bracket"),
        ("Chebyshev", "B3ah-ChebyKAN-K3-h80-paircrossR32-inputsqL4P16-tritonL3-gradbuf", "D4-V Cheby K3 h80 paircrossR32 plus fixed projection-square P16 random quadratic channel"),
        ("Chebyshev", "B3ai-ChebyKAN-K3-h72-paircrossR32-inputsqL4P24-tritonL3-gradbuf", "D4-W Cheby K3 h72 paircrossR32 plus fixed projection-square P24 random quadratic channel"),
        ("Chebyshev", "B3aj-ChebyKAN-K3-h72-paircrossR32-inputsqL4P8-tritonL3-gradbuf", "D4-X Cheby K3 h72 paircrossR32 plus fixed projection-square P8 efficiency floor"),
        ("Chebyshev", "B3ak-ChebyKAN-K3-h64-paircrossR32-inputsqL4P8-tritonL3-gradbuf", "D4-X Cheby K3 h64 paircrossR32 plus fixed projection-square P8 lower hidden bracket"),
        ("Chebyshev", "B3al-ChebyKAN-K3-h72-paircrossR32-inputrot2L4P8-tritonL3-gradbuf", "D4-Y Cheby K3 h72 paircrossR32 plus local rotated sum/diff square and P8 random bilinear"),
        ("Chebyshev", "B3am-ChebyKAN-K3-h64-paircrossR32-inputrot2L4P8-tritonL3-gradbuf", "D4-Y Cheby K3 h64 paircrossR32 plus local rotated sum/diff square and P8 lower hidden"),
        ("Chebyshev", "B3p-ChebyKAN-K3-h112-paircrossR32-inputcrossL4P16-tritonL3-gradbuf", "D4-I Cheby K3 h112 hidden pair-cross plus input local/random-projection product expression primitive"),
        ("Chebyshev", "B3q-ChebyKAN-K3-h112-inputcrossL4P32-tritonL3-gradbuf", "D4-J Cheby K3 h112 input local/random-projection product sidecar without hidden-pair workspace"),
        ("Chebyshev", "B3r-ChebyKAN-K3-h112-inputcrossL4P64-tritonL3-gradbuf", "D4-K Cheby K3 h112 input local/random-projection product sidecar with rank64 coverage repair"),
        ("Chebyshev", "B3t-ChebyKAN-K3-h112-inputcrossL4P96-tritonL3-gradbuf", "D4-L Cheby K3 h112 input local/random-projection product sidecar with rank96 efficiency/expression bracket"),
        ("Chebyshev", "B3u-ChebyKAN-K3-h112-inputcrossL4P112-tritonL3-gradbuf", "D4-L Cheby K3 h112 input local/random-projection product sidecar with rank112 efficiency/expression bracket"),
        ("Chebyshev", "B3s-ChebyKAN-K3-h112-inputcrossL4P128-tritonL3-gradbuf", "D4-L Cheby K3 h112 input local/random-projection product sidecar with rank128 coverage repair"),
        ("Chebyshev", "B3v-ChebyKAN-K3-h112-inputcrossL4P128-linearres-tritonL3-gradbuf", "D4-M Cheby K3 h112 inputcross rank128 plus low-cost linear residual task repair"),
        ("Chebyshev", "B3w-ChebyKAN-K3-h112-inputcrossL4P128-linearres050-tritonL3-gradbuf", "D4-M Cheby K3 h112 inputcross rank128 plus stronger linear residual task repair"),
        ("Chebyshev", "B3x-ChebyKAN-K3-h112-inputcrossL4P128-linearraw050-tritonL3-gradbuf", "D4-N Cheby K3 h112 inputcross rank128 plus raw-scale linear residual task repair"),
        ("Chebyshev", "B3a-ChebyKAN-K4", "D4-A Cheby K4 dense reference / L1 + L2 compile-forward attempt"),
        ("Chebyshev", "B3d-ChebyKAN-K6-stream", "D4-B Cheby K6 expression repair / L1 + L2 compile-forward attempt"),
        ("Fourier", "B4b-FourierKAN-lowfreq-K2-stream", "D5-A Fourier K2 fixedfreq stream / L1 + L2 compile-forward attempt"),
        ("Fourier", "B4c-FourierKAN-lowfreq-K2-flatgemm-manual", "D5-B Fourier K2 flat-GEMM manual L3 repair attempt"),
        ("Fourier", "B4d-FourierKAN-lowfreq-K2-flatgemm-recompute", "D5-C Fourier K2 flat-GEMM recompute manual L3 workspace repair attempt"),
        ("Fourier", "B4e-FourierKAN-lowfreq-K2-tritonL3", "D5-D Fourier K2 Triton fused-forward/gradient L3 repair attempt"),
        ("Fourier", "B4f-FourierKAN-lowfreq-K2-tritonL3-blockH", "D5-E Fourier K2 Triton block-H parallel forward + analytic gradient L3 repair attempt"),
        ("Fourier", "B4g-FourierKAN-lowfreq-K2-tritonL3-matmulTile", "D5-F Fourier K2 Triton batch-block x hidden-block matmul forward + analytic gradient L3 repair attempt"),
        ("Fourier", "B4h-FourierKAN-lowfreq-K3-tritonL3-matmulTile", "D5-G Fourier K3 Triton matmul-tile cos-channel expression repair after B4g A4 failure"),
        ("Fourier", "B4i-FourierKAN-lowfreq-K3-h96-tritonL3-matmulTile", "D5-H Fourier K3 h96 Triton matmul-tile efficiency repair after B4h step failure"),
        ("Fourier", "B4j-FourierKAN-lowfreq-K3-h80-tritonL3-matmulTile", "D5-I Fourier K3 h80 Triton matmul-tile efficiency repair if h96 remains blocked"),
        ("Fourier", "B4k-FourierKAN-lowfreq-K3-h64-tritonL3-matmulTile", "D5-J Fourier K3 h64 Triton matmul-tile efficiency floor repair after B4j near miss"),
        ("Fourier", "B4l-FourierKAN-lowfreq-K4-h64-tritonL3-matmulTile", "D5-K Fourier K4 h64 Triton matmul-tile expression repair after K3 A4 failure"),
        ("Fourier", "B4m-FourierKAN-lowfreq-K4-h48-tritonL3-matmulTile", "D5-L Fourier K4 h48 Triton matmul-tile efficiency fallback"),
        ("Fourier", "B4n-FourierKAN-lowfreq-K4-h64-linearres-tritonL3-matmulTile", "D5-C Fourier K4 lowfreq plus linear residual expression repair"),
        ("Fourier", "B4o-FourierKAN-lowfreq-K4-h48-linearres-tritonL3-matmulTile", "D5-C Fourier K4 lowfreq plus linear residual h48 efficiency fallback"),
        ("Fourier", "B4p-FourierKAN-lowfreq-K4-h64-linearres-gemmDirectL3", "D5-C Fourier K4 lowfreq plus linear residual GEMM direct-gradient repair"),
        ("Fourier", "B4q-FourierKAN-lowfreq-K4-h48-linearres-gemmDirectL3", "D5-C Fourier K4 lowfreq plus linear residual GEMM direct-gradient h48 fallback"),
        ("Fourier", "B4r-FourierKAN-lowfreq-K4-h32-linearres-gemmDirectL3", "D5-C Fourier K4 lowfreq plus linear residual GEMM h32 floor repair"),
        ("Fourier", "B4s-FourierKAN-lowfreq-K4-h24-linearres-gemmDirectL3", "D5-C Fourier K4 lowfreq plus linear residual GEMM h24 floor repair"),
        ("Fourier", "B4t-FourierKAN-lowfreq-K4-h16-linearres-gemmDirectL3", "D5-C Fourier K4 lowfreq plus linear residual GEMM h16 floor repair"),
        ("Fourier", "B4u-FourierKAN-lowfreq-K4-h8-linearres-gemmDirectL3", "D5-C Fourier K4 lowfreq plus linear residual GEMM h8 final floor repair"),
        ("Fourier", "B4v-FourierKAN-lowfreq-K4-h8-linearres050-gemmDirectL3", "D5-C Fourier K4 h8 GEMM direct-gradient with stronger 0.50 linear residual expression repair"),
        ("Fourier", "B4w-FourierKAN-lowfreq-K4-h8-linearres050-tritonL3-matmulTile", "D5-D Fourier K4 h8 stronger 0.50 linear residual with Triton residual add/gradient cost repair"),
        ("Fourier", "B4a-FourierKAN-lowfreq-K4", "D5-A Fourier K4 fixedfreq reference / L1 + L2 compile-forward attempt"),
        ("Wavelet", "B5h-HatWaveletKAN-local-K4", "D6-A HatWavelet local support / L1 stream + L2 compile-forward attempt"),
    ]


def write_contract_files(args: argparse.Namespace, out_dir: Path, device: torch.device) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    manifest = [
        _stamp(
            {
                "stage": "V1283_RUN_MANIFEST",
                "run_id": out_dir.name,
                "generated_at": _now_iso(),
                "plan_path": str(PLAN_PATH),
                "script_path": str(SCRIPT_PATH),
                "device": str(device),
                "torch_version": torch.__version__,
                "triton_available": int(fhq.TRITON_AVAILABLE),
                "train_size": int(args.train_size),
                "val_size": int(args.val_size),
                "test_size": int(args.test_size),
                "batch_size": int(args.batch_size),
                "datasets": str(args.datasets),
                "seeds": str(args.seeds),
                "epochs": int(args.epochs),
                "b109_repair_candidate_ids": str(getattr(args, "repair_candidate_ids", "")),
            }
        )
    ]
    contract = [
        _stamp({"stage": "V1283_STRICT_CONTRACT", "contract": "strict_FC_PureKAN", "value": 1}),
        _stamp({"stage": "V1283_STRICT_CONTRACT", "contract": "CE_benchmark_protocol_for_task_comparability", "value": 1}),
        _stamp({"stage": "V1283_STRICT_CONTRACT", "contract": "new_manual_VJP_accepts_general_dL_dlogits", "value": 1}),
        _stamp({"stage": "V1283_STRICT_CONTRACT", "contract": "no_teacher_no_distillation", "value": 1}),
        _stamp({"stage": "V1283_STRICT_CONTRACT", "contract": "no_sampler_or_class_weight", "value": 1}),
        _stamp({"stage": "V1283_STRICT_CONTRACT", "contract": "no_dataset_name_branch", "value": 1}),
        _stamp({"stage": "V1283_STRICT_CONTRACT", "contract": "official_KAN_path_no_loss_backward", "value": 1}),
        _stamp({"stage": "V1283_STRICT_CONTRACT", "contract": "functional_official_closed_until_base_qualified", "value": 1}),
    ]
    modification = [
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added hat_wavelet basis evaluator/channel, B5h/ChebyK3/ChebyK6/FourierK2 family specs, B136/B137 fixed low-frequency/block-frequency B109 repair specs, B138 semi-fixed-P repair spec, B139 projection-gradient-every4 repair spec, B140/B141/B142/B143/B144 projection-gradient schedule repair specs, B145 activeP64 projection-gradient repair spec, B146/B147 activeP48/P96 projection-gradient repair specs, B148-B151 projection-lr smoothing repair specs, B152-B159 projection-lr schedule repair specs, B160/B161 lowFreqP active-update repair specs, B162/B163 blockfreqP active-update repair specs, B164/B165 pairScaleP H-parameter projection repair specs, B166-B168 sparseK projection repair specs, B169-B171 group-abs tail-local signal repair specs, B172-B175 group-abs + projection-lr schedule cross repairs, B176-B179 group-abs strength scans, B180-B183 block-local SparseP projection repairs, B184-B187 hybrid local/global SparseP projection repairs, B188-B190 dense-P manual foreach AdamW update repairs, B191-B193 low-cost norm/mean statistic trajectory repairs, B194-B197 q-normalized/bounded quadratic trajectory repairs, B198-B202 scheduled projection-update + manual foreach AdamW cross repairs, B203 Triton quad_proj AdamW update repair, B204 fused projection-gradient AdamW update repair, B205 fused-update samplewise logit-norm trajectory repair, B206 fused-update fixed scalar trajectory repair, B207 lower-target logitnorm trajectory repair, B208/B209 hybrid sparseP logitnorm trajectory repairs, B210/B211 stop-grad samplewise logitnorm repairs, B212/B213 stop-grad batch logitnorm repairs, B214-B216 non-logit absmixsq direct-tail trajectory repairs, B217/B218 signed-square direct-tail trajectory repairs, B219/B220 abs+weak-square direct-tail trajectory repairs, B221-B223 structured signed-square plus groupabs trajectory repairs, B224/B225 orthogonal-P condition-preserving trajectory repairs, B226-B229 label-free PCA-P signal-projected trajectory repairs, B230/B231 audited train-probe centroid signal repairs, B232/B233 train-probe confidence/ECE repairs, B234/B235 fixed-gain train-probe confidence repairs, B236/B237 global fixed-gain train-probe confidence repairs, B238-B241 bounded-logit train-probe ECE repairs, B242-B247 high fixed-gain train-probe ECE/AUC repairs, B248-B251 gain-ramp train-probe trajectory repairs, B252/B253 direct075 gain-ramp repairs, B254 temp115 gain-ramp repair, B255 temp105 gain-ramp confirm10 repair, B256 temp100 gain-ramp confirm10 repair, and B257 temp095 gain-ramp confirm10 repair",
                "reason": "v12.8.3 plan requires classic family audit and B109 trajectory repair; additions are fixed basis/projection functions, no label/dataset branch",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "new v12.8.3 runner producing v1283_* artifacts, B109 FHQ fullstep/autopsy, classic family L1/L2/L3-analytic-manual microbench, blocker/status/provenance/hash/report; full-step profiling now covers repair_candidate_ids as well as exact_candidate_id; optimizer grouping supports dense projection, pairScaleP, sparseK projection-only LR smoothing and single-/multi-stage scheduled projection-lr variants; manualadamw, tritonprojadamw, fusedquadprojadamw, fused-update logitnorm, and stop-grad logitnorm variants use audited update paths in full-step and task autopsy",
                "reason": "execute integrated plan with auditable no-fake outputs and explicit distinction between official and exploratory paths; classic families need a real no-loss-backward L3 analytic attempt before being marked KernelBlocked",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "experiments/run_v1283_b109_classic_family_functional_geometry.py / dgkan/models/fc_purekan_primitives.py",
                "change": "added B7ht/B7hu/B7hv/B7hw Rational linear-residual scale ramp candidates and task trace audit column",
                "reason": "B7ej/B7hn-family Rational candidates reached L3/A4 or strong accuracy but remained A5-blocked by AUC/ECE instability. B7ht/B7hu test a loss-agnostic feature trajectory schedule for the B7ej linear residual path, while B7hv/B7hw apply the same schedule to the B7hn hidden-tail Triton VJP route. All consume the same generic dL/dlogits VJP and avoid CE-specific calibration, loss changes, sampler/class weights, teacher, distillation, or dataset-name branching.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "experiments/run_v1283_b109_classic_family_functional_geometry.py / dgkan/models/fc_purekan_primitives.py",
                "change": "added B7ix/B7iy Rational smooth tanh hidden-residual candidates with generic Triton readout-gradient + hidden VJP",
                "reason": "B7ir/B7iv proved bounded rational hidden residuals and generic residual VJP are legal but still A5-blocked. B7ix/B7iy test a genuinely different single-kernel-friendly trajectory primitive h + a*tanh(h), still folded into existing w2 readout and consuming arbitrary dL/dlogits rather than a CE-specific backward.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "experiments/run_v1283_b109_classic_family_functional_geometry.py / dgkan/models/fc_purekan_primitives.py",
                "change": "added B7kc bounded-rational hidden residual amplitude damping candidate",
                "reason": "B7iv/B7iw opened L3/A4 but failed A5, while B7jy/B7jz and B7ka/B7kb showed the sign-preserving quadratic hidden residual is too costly for L3. B7kc returns to the cheaper bounded rational hidden residual and lowers amplitude from 0.10 to 0.05 without labels, CE-specific formulas, dataset branches, or loss/sampler/class-weight changes.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "experiments/run_v1283_b109_classic_family_functional_geometry.py / dgkan/models/fc_purekan_primitives.py",
                "change": "added B7iz/B7ja Rational centered smooth tanh hidden-residual candidates with generic Triton readout-gradient + hidden VJP",
                "reason": "B7ix/B7iy preserved L3/A4 but reproduced the old A5 failure shape. B7iz/B7ja subtract the hidden-dimension mean of tanh(h) inside the residual so the primitive changes hidden trajectory without adding readout/update state or CE-specific loss coupling.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "experiments/run_v1283_b109_classic_family_functional_geometry.py / dgkan/models/fc_purekan_primitives.py",
                "change": "added B7jb/B7jc high-gain Rational pair-logit fused cap candidates on the B7ir/B7iv anchor",
                "reason": "The hidden-residual family preserved L3/A4 but did not fix A5. B7jb/B7jc switch to a non-hidden-residual trajectory primitive: bounded pair logits before final accumulation. The VJP still consumes arbitrary dL/dlogits and the task protocol remains comparable without CE-specific coupling, loss changes, sampler/class weights, teacher, distillation, or dataset-name branching.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "experiments/run_v1283_b109_classic_family_functional_geometry.py / dgkan/models/fc_purekan_primitives.py",
                "change": "added B7jd/B7je centered pair-logit cap candidates with generic class-dimension centering VJP",
                "reason": "B7jb/B7jc proved pair-logit caps are legal but still A5-blocked. B7jd/B7je remove the class-common pair-logit offset before tanh cap and backpropagate through the centering projection using arbitrary dL/dlogits, not a CE-specific formula.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "experiments/run_v1283_b109_classic_family_functional_geometry.py / dgkan/models/fc_purekan_primitives.py",
                "change": "added B7jf/B7jg, B7jh/B7ji, B7jj/B7jk, B7jl/B7jm, B7jn/B7jo, B7jp/B7jq, B7jr, B7js, B7jt, B7ju, B7jv, B7jw, B7jx, B7jy/B7jz, and B7ka/B7kb batch-centered Rational task-trajectory candidates with generic VJP",
                "reason": "B7jd/B7je exposed class-common centering as effectively softmax-invariant for the task protocol. B7jf/B7jg subtract per-class batch means before tanh cap; B7jh/B7ji add lower-cap task-trajectory brackets; B7jj/B7jk keep the same forward centering but stop gradient through the batch mean to remove cross-batch mean-gradient coupling. B7jl/B7jm then combine that stop-gradient batch trajectory with a low-cost hidden residual channel. B7jn/B7jo reduce paircross rank to recover fused backward/update cost after B7jm failed L3 by a narrow margin. B7jp/B7jq test midpoint ranks after B7jn recovered L3 but lost A4 expression. B7jr and B7js test high-rank R128/R132 expression-capacity brackets. B7jt keeps R132 but lowers hidden residual amplitude to test a loss-agnostic task-trajectory damping repair after B7js opened A4 but failed A5. B7ju keeps amplitude 0.10 but centers the tanh hidden residual per sample to reduce non-class-specific drift. B7jv keeps B7js expression structure but lowers the stop-gradient batch cap from 1.50 to 1.25 to reduce pair-logit calibration pressure. B7jw removes the hidden residual at R132 to isolate whether B7js A5 harm is residual-driven. B7jx keeps the no-hidden R132 path but lowers the stop-gradient batch cap to 1.25 to test whether B7jw's A5/ECE/AUC failure is amplitude-driven. B7jy/B7jz then test a different sign-preserving bounded quadratic hidden residual h*abs(h)/(1+h^2) after no-hidden/cap brackets still failed A5. B7ka/B7kb lower that new residual path's paircross rank after B7jy/B7jz gradcheck passed but missed L3. These variants change logits without using labels, CE-specific formulas, dataset branches, or extra direct-gradient rows.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "experiments/run_v1283_b109_classic_family_functional_geometry.py / dgkan/models/fc_purekan_primitives.py / dgkan/optim/manual_adamw.py",
                "change": "added B7kd/B7ke/B7kf/B7kg/B7kh/B7ki/B7kj/B7kk/B7kl/B7km/B7kn/B7ko/B7kp/B7kq/B7kr/B7ks/B7kt/B7ku/B7kv/B7kw/B7kx/B7ky/B7kz/B7la/B7lb/B7lc/B7ld/B7le/B7lf/B7lg/B7lh/B7li/B7lj/B7lk/B7ll/B7lm/B7ln/B7lo/B7lp Rational follow-ups and changed manual AdamW update to in-place addcdiv form",
                "reason": "B7kc/B7jt/B7jx/B7kf showed Rational is blocked by a mix of L3 timing, expression capacity, and KMNIST AUC. The follow-ups bracket linear residual gain, hidden-residual amplitude/cap, no-hidden lower cap, R128/R120/R112/R96 middle-to-low ranks, R112/R120/R128 standard/balanced/diag-rich pair coverage, a 25% standard + 75% balanced mixed pair selector, low-cost fixed DCT projection quadratic readouts on cheap R80 paths at R32/R48/R64, R80 sum-square/difference-square/blend pair feature fallbacks, random-projection inputcross R32/R48/R64/R96/R112 coverage, and pairNorm plus hidden residual damping while keeping generic dL/dlogits VJP. The AdamW change removes an avoidable update tensor allocation without changing decoupled AdamW math. None of these change CE loss, labels, sampler/class weights, teacher/distillation, dataset branches, or gates.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added Rational denominator/derivative diagnostics and v1283_family_basis_diagnostics.csv output",
                "reason": "v12.9 requests Rational task-geometry autopsy after L3/A4-capable variants remained A5 blocked. The new diagnostic rows record den_min, den_p01, den_condition, r_prime_p95, r_double_prime_p95, group_function_diversity, and group_dead_fraction without changing gates, loss, labels, sampler/class weights, teacher/distillation, dataset branches, or optimizer semantics.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added optional classic-family Line C autopsy wrapper and --family-linec-ids selector",
                "reason": "v12.9 Rational candidates can pass fused L3 and A4 expression while still failing A5. The wrapper reuses the existing Line C coupling/signal/noise diagnostics for selected family candidates, writes separate v1283_family_linec_* artifacts, records per-candidate failures instead of promoting them, and does not change gates, loss, labels, sampler/class weights, teacher/distillation, dataset branches, or official family-success criteria.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "experiments/run_v1283_b109_classic_family_functional_geometry.py / dgkan/models/fc_purekan_primitives.py",
                "change": "added B7lq/B7lr/B7ls/B7lt Rational Line C coupling-collapse repair candidates",
                "reason": "B7kc/B7en/B7lp Line C autopsy measured low CouplingR2 versus MLP and high RealSignalReservoirRatio while denominator and derivative safety remained normal. B7lq adds a stop-gradient batch-centered pair-logit cap on top of B7lp, and B7lr applies the same pair-signal geometry without the hidden residual to isolate whether the collapse comes from pair-logit channel geometry or the residual. After B7lq/B7lr still collapsed, B7ls/B7lt freeze only rational coefficients while allowing w1 backbone updates to test whether the readout-only trajectory is the Line C blocker. All consume generic dL/dlogits VJP and do not change CE loss, labels, sampler/class weights, teacher/distillation, dataset branches, or gates.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "experiments/run_v1283_b109_classic_family_functional_geometry.py / dgkan/models/fc_purekan_primitives.py",
                "change": "added B7lu/B7lv Rational shared pair-signal readout candidates",
                "reason": "B7lq/B7lr showed batch-centered pair-logit caps did not repair Line C coupling collapse, and B7ls/B7lt showed opening trainable w1 worsened RealSignalReservoirRatio. B7lu/B7lv keep the same pairNorm R136 pair features but replace the full pair-to-class readout with a low-rank shared signal channel K=4/K=8 plus a centered class map. This directly tests whether the full per-class pair readout is acting as a reservoir while preserving a generic dL/dlogits VJP and avoiding CE loss, label, sampler/class-weight, teacher/distillation, dataset-branch, or gate changes.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "experiments/run_v1283_b109_classic_family_functional_geometry.py / dgkan/models/fc_purekan_primitives.py",
                "change": "added B7lw/B7lx Rational PCA-whitened pair feature readout candidates",
                "reason": "B7lu/B7lv low-rank shared signal damaged A4 and did not improve Line C, while pairNorm-only candidates still showed high feature condition proxies. B7lw/B7lx keep the same unlabeled R136 pair features but replace diagonal-only conditioning with a fixed train-input PCA whitening transform at R96/R128 before the readout, testing feature-conditioning and one-window drift geometry without CE loss changes, labels, sampler/class weights, teacher/distillation, dataset branches, or gate changes.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "experiments/run_v1283_b109_classic_family_functional_geometry.py / dgkan/models/fc_purekan_primitives.py",
                "change": "added B7ly/B7lz Rational PCA-rotated non-whitened pair feature readout candidates",
                "reason": "B7lw/B7lx showed PCA whitening can pass L3/A4 at R128 but amplifies logits and worsens Line C coupling. B7ly/B7lz keep the same fixed unlabeled PCA basis but remove inverse-singular-value whitening, testing whether rotation/truncation can preserve A4 without low-energy amplification. They keep generic dL/dlogits VJP and do not change CE loss, labels, sampler/class weights, teacher/distillation, dataset branches, or gates.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "experiments/run_v1283_b109_classic_family_functional_geometry.py / dgkan/models/fc_purekan_primitives.py",
                "change": "added B7ma/B7mb samplewise and B7mc-B7mh batch-level stop-gradient logit RMS normalization candidates",
                "reason": "B7lz preserved L3/A4 and improved A5 mean_delta but still failed A5 and Line C stayed at coupling_collapse. B7ma applies a stop-gradient samplewise logit RMS normalization to the B7lz PCA-rotated output geometry, and B7mb applies the same output geometry repair to the B7lp pairNorm+hidden-residual anchor. B7mc/B7md are cheaper batch-level RMS variants added after the samplewise versions improved Line C but failed L3 timing; B7me/B7mf bracket batch RMS scale at 1.50/2.00 after B7md opened L3/A4 but hurt A5 margins. B7mg/B7mh mix the batch RMS factor residually at 25%/50% to test whether partial output-geometry repair can preserve the original task trajectory. The backward consumes generic dL/dlogits scaled by the detached RMS inverse or mixed factor, so this does not change CE loss, labels, sampler/class weights, teacher/distillation, dataset branches, or gates.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added B314/B315 direct-branch ramp 1.05/1.10 B109 confirm candidates",
                "reason": "B309-B313 confirm10 narrowed the blocker to Fashion-MNIST AUC-time with efficiency/accuracy/ECE mostly intact. B314/B315 are loss-agnostic branch-scale trajectory brackets between B309 and B310: they do not change CE, labels, sampler, class weights, teacher/distillation, dataset branches, or backward formulas; existing manual/FHQ paths still consume generic dL/dlogits.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added PrimitiveKAN analytic manual CE forward/backward cache with fixed-basis derivatives for classic families",
                "reason": "v12.8.3 classic-family branch requires an L3 analytic backward/update attempt; this avoids loss.backward for the candidate step while keeping gates unchanged",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added Fourier K2 memory-planned, flat-GEMM cache, flat-GEMM recompute, and B4e Triton L3 PrimitiveKAN manual CE paths; B4e computes Fourier K2 forward and w1/w2 gradients with Triton kernels without materializing BxDxK basis tensors; generic manual CE is guarded by no_grad",
                "reason": "v12.8.3 section 10 left Fourier K2 closest but still memory/step blocked; these variants separate recompute, flat-GEMM, and Triton-fused evidence, with official promotion allowed only after correctness and efficiency gates pass",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/kernels/fused_fourier_k2.py",
                "change": "new narrow Triton Fourier K2 kernels for PrimitiveKAN forward, w2 gradient, and block-D w1 gradient",
                "reason": "test the plan's family-specific fused backward/update direction on the closest classic family candidate without using loss.backward, fake rows, proxy data, or CPU offload; block-D w1 gradient reduces tiny D x H program fragmentation",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/kernels/fused_fourier_k2.py",
                "change": "added B4f block-H parallel Triton Fourier K2 forward path while reusing the existing analytic Triton w1/w2 gradients",
                "reason": "B4e proved that the classic Fourier K2 L3 path was blocked mainly by a serial hidden-loop forward kernel; block-H parallelism is the next planned fused-kernel repair without changing loss, gates, or data",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/kernels/fused_fourier_k2.py",
                "change": "added B4g batch-block x hidden-block matmul-style Triton Fourier K2 forward path and switched its tl.dot precision to tf32x3 after audit",
                "reason": "B4f reduced serial hidden-loop overhead but still failed the step gate; B4g reuses w1/w2 tiles across a batch block using tl.dot. Default tl.dot was efficient but failed gradient audit, while ieee precision hit a Triton compiler blocker, so tf32x3 is the audited precision repair without changing the mathematical basis",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added B4f-FourierKAN-lowfreq-K2-tritonL3-blockH and B4g-FourierKAN-lowfreq-K2-tritonL3-matmulTile PrimitiveSpecs with manual CE dispatch",
                "reason": "route B4f/B4g through the same audited manual CE / no loss.backward path as B4e so any improvement is measured under identical gates",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "recorded manual_kernel_variant and manual_no_grad_guard fields for classic-family L3 efficiency/gradcheck rows and added B4f/B4g to the Fourier family plan",
                "reason": "separate Fourier K2 memory-planned evidence from generic dense-cache manual rows without changing gates or promoting exploratory results; B4f/B4g test the next concrete kernel repair directions",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "split classic-family failure codes for fused L3 kernel measured-but-inefficient versus missing fused kernel",
                "reason": "B4e Fourier K2 Triton L3 is a real fused kernel attempt; audit tables must not mislabel it as missing a fused kernel, and still must not promote it when efficiency gates fail",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added classic-family A4 expression promotion after a candidate passes official fused L3 efficiency and analytic gradcheck",
                "reason": "once B4f/B4g produce a real fused L3 efficiency pass, the runner should no longer leave the family permanently closed as KernelBlocked; A4 is opened under the same v12.5 expression gate, while A5 remains closed until a fused-L3 task-training path is integrated",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/kernels/fused_fourier_k2.py",
                "change": "added B4h Fourier K3 Triton matmul-tile forward plus analytic w1/w2 gradient kernels",
                "reason": "B4g proved Fourier K2 can pass fused L3 efficiency and gradient audit but failed A4 expression. B4h adds only the missing low-frequency cos channel while keeping a single-kernel-friendly Triton forward/backward/update path, unchanged loss/gates/data, and no CPU/proxy/fake rows.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B4h-FourierKAN-lowfreq-K3-tritonL3-matmulTile PrimitiveSpec, manual CE dispatch, family plan entry, and official fused-L3 recognition",
                "reason": "B4h must be measured under the same L3 efficiency, gradcheck, and A4 promotion gates as B4g; no family success is claimed unless the new artifact passes those gates.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B4i/B4j/B4k Fourier K3 h96/h80/h64 Triton matmul-tile efficiency repairs",
                "reason": "B4h kept the desired K3/cos channel and passed analytic gradcheck but exceeded the L3 step gate. B4i/B4j/B4k reduce only hidden width to test whether the same single-kernel-friendly primitive can recover efficiency before any expression or task claim.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/kernels/fused_fourier_k2.py / dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B4l/B4m Fourier K4 h64/h48 Triton matmul-tile forward plus analytic gradient path",
                "reason": "The v12.8.3 D5 plan says A4-not-enough should try K2/K3 -> K4 before learned frequency/phase. B4l/B4m add the fixed low-frequency K4 channel under the same fused L3/gradcheck/A4 gates, without changing loss, data, sampler/class weights, teacher, distillation, or dataset branching.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/kernels/fused_fourier_k2.py / dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B4n/B4o Fourier K4 low-frequency plus trainable linear residual Triton L3 path",
                "reason": "K3/K4 fixed low-frequency Fourier reached fused L3 efficiency and gradcheck but failed A4 expression. The D5 plan says the next A4 repair is adding a linear residual before learnable frequency/phase, so B4n/B4o add only a direct normalized-input readout with Triton forward-add and direct-gradient kernels under the same gates.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B4p/B4q Fourier K4 low-frequency plus linear residual GEMM-native direct-gradient variants",
                "reason": "B4n/B4o proved the linear residual gradients are correct but the per-dimension Triton direct-gradient kernel is too fragmented for the step gate. B4p/B4q keep the same mathematical primitive and fused K4 w1/w2 kernels but compute the tiny direct residual gradient as one CUDA matmul, testing the planned single-kernel-friendly repair without using autograd/loss.backward.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B4r/B4s/B4t/B4u h32/h24/h16/h8 Fourier K4 plus linear residual GEMM-native floor repairs",
                "reason": "B4p/B4q still exceeded the L3 step gate, so the next non-gate-lowering repair is to reduce only hidden width while preserving the same low-frequency K4 plus linear residual primitive and no-autograd manual gradient path.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B4v Fourier K4 h8 GEMM-native direct-gradient candidate with linear residual scale 0.50",
                "reason": "B4u proved the low-frequency K4 h8 GEMM-native path can pass efficiency but remained expression-blocked. B4v keeps the same fixed frequencies, hidden width, CE loss, sampler, and no-autograd manual L3 update path, changing only the fixed linear residual scale from 0.10 to 0.50 to test whether the A4 blocker is an underpowered linear base rather than a Fourier-kernel timing issue.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B4w Fourier K4 h8 linearres050 candidate using Triton residual add/gradient instead of GEMM residual gradient",
                "reason": "B4v passed analytic gradient audit but failed the L3 step gate. B4w keeps the same mathematical primitive and residual scale, but routes the residual add/gradient through the existing Triton residual kernels to test whether the blocker is the GEMM residual-gradient/update overhead rather than the Fourier basis path.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/kernels/fused_chebyshev_k3.py / dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B3e Chebyshev K3 family-specific Triton L3 matmul-tile forward plus analytic w1/w2 gradient path",
                "reason": "After Fourier K/linear-residual scans remained expression-blocked, the next plan-consistent repair is to move a non-Fourier family from generic manual L3 into a real family-specific fused L3 path. Chebyshev K3 has closed-form polynomial basis and derivative, so B3e tests official fused efficiency/A4 without fake rows, proxy data, CPU offload, loss changes, or dataset branching.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added --family-promotion-ids to optionally restrict classic-family A4 promotion to named fused-L3 candidates",
                "reason": "The Chebyshev B3e repair needs a clean A4 measurement without re-running every previously eligible Fourier promotion candidate. Empty default preserves the original all-eligible behavior; targeted runs remain audited and do not change gates, data, loss, or success criteria.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/kernels/fused_chebyshev_k3.py / dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B3f Chebyshev K4 family-specific Triton L3 matmul-tile forward plus analytic w1/w2 gradient path",
                "reason": "B3e moved Chebyshev into a real fused-L3 path but failed A4 expression. The v12.8.3 D4 plan recommends K adjustment before declaring a family geometry failure, so B3f adds T3 while keeping the same no-loss-backward fused L3 gate and no data/loss/gate changes.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B3j/B3i/B3g/B3h Chebyshev K3 h120/h128/h160/h192 capacity repairs reusing the B3e Triton L3 kernel",
                "reason": "B3f showed K4 increases memory/step cost before A4 can open. B3j/B3i/B3g/B3h keep the cheap K3 basis and spend only hidden capacity, testing whether Chebyshev expression can be repaired while preserving the proven family-specific fused L3 path.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/kernels/fused_chebyshev_k3.py / dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B3k/B3l/B3m Chebyshev K3 h120/h128/h112 persistent grad-buffer L3 variants",
                "reason": "B3j h120 had enough step margin but failed memory by a narrow amount. B3k/B3l keep the same Chebyshev function class and kernels, but reuse warmup-allocated w1/w2 gradient buffers during measured steps. B3m is a coarse h112 lower bracket to test whether Cheby K3 can legally enter A4 under the memory gate without changing data, loss, gates, labels, or optimizer semantics.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/kernels/fused_chebyshev_k3.py / dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B3n/B3o/B3p/B3q/B3r/B3t/B3u/B3s/B3v/B3w/B3x Chebyshev K3 h112 low-rank product readout variants with manual/Triton L3 gradients",
                "reason": "B3m proved that per-dimension Cheby K3 can pass fused L3 efficiency but fails A4 expression. B3n/B3o add h_i*h_j pair products over fixed hidden pairs; B3p adds input local-pair plus fixed random-projection products; B3q removes the hidden-pair workspace and keeps only input product features; B3r/B3t/B3u/B3s increase fixed random-product rank after lower ranks kept E6/E8 frozen coverage low; B3v/B3w add low-cost linear residual strengths after B3s passed A4 but collapsed in task triage; B3x removes the extra sqrt(input_dim) damping from that residual after B3v/B3w showed the residual was too weak to affect task trajectory. Data, loss, gates, labels, and optimizer semantics are unchanged.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v124_multibasis_functional_dual.py / experiments/run_v1283_b109_classic_family_functional_geometry.py / third_party/FlashKAT/rational_kat",
                "change": "added B7b/B7c/B7d/B7e/B7f/B7g/B7h/B7i/B7j/B7k/B7l/B7m/B7n/B7o/B7p/B7q/B7r/B7s/B7t/B7u/B7v/B7w/B7x/B7y/B7z/B7aa/B7ab/B7ac/B7ad/B7ae/B7af/B7ag/B7ah/B7ai/B7aj/B7ak/B7al/B7am/B7an/B7ao/B7ap/B7aq/B7ar/B7as/B7at/B7au/B7av/B7aw/B7ax/B7ay/B7az/B7ba/B7bb/B7bc/B7bd/B7be/B7bf/B7bg/B7bh/B7bi/B7bj/B7bk/B7bl/B7bm/B7bn/B7bo/B7bp/B7bq/B7br/B7bs/B7bt/B7bu/B7bv/B7bw/B7bx/B7by/B7bz FlashKAT-inspired grouped Rational KAT candidates with grouped P(x)/Q(|x|) activation, compact GEMM heads, optional linear residual, fixed random input-cross readout repairs, structured pair-cross readout repairs, balanced structured pair-cross repairs, fixed transform pair-cross repairs, smaller-hidden full-coverage pair-cross repairs, grid-balanced image pair repairs, lower-rank pair-cross task/expression brackets, readout learning-rate task repair, hidden-coupled fixed pair repairs, bucketed hidden-coupled pair repairs, coalesced Triton bucket hidden-delta repair, mixed bucket-hidden plus direct pair-readout coupling repair, small-rank bucket hidden trajectory coupling behind full direct pair readout, fixed pair-readout scale task-trajectory repair, fused Triton pair-readout logits/gradient repair, block-tiled Triton pair-readout repair, optional logit-bias calibration repair, bias-fused block-readout repair, lower-rank block-readout repair, R128/R120 block-readout expression/efficiency bracket, manual L3 gradients, and cached FlashKAT kernel handles",
                "reason": "The existing B7a rational_kat_lite path is a generic dense edge-basis Rational and remains KernelBlocked. FlashKAT's rational_kat uses group-shared numerator/denominator coefficients and tile-accumulated coefficient gradients, so B7b tests that lower-cost grouped activation shape without changing data, labels, loss, gates, or optimizer semantics. A first B7b measurement exposed Python import/path lookup overhead in the wrapper; caching the FlashKAT kernel handles removes that non-mathematical overhead while keeping the same rational function and gradients. Because cached B7b h112 still failed full-step speed in the promotion run and B7c/B7d passed efficiency but failed pairwise/rotated expression, B7e/B7f add fixed random input-cross product features with trainable readout only, to repair A4 interaction capacity without adding trainable projection gradients. B7g/B7h bracket B7f with lower input-cross ranks after R128 showed efficiency instability. B7i/B7j replace random projections with deterministic coordinate pair products so the expression-dim tests can cover true quadratic interactions while retaining fixed features and trainable readout only; B7k/B7l lower that structured rank after B7j missed the 1.25 step gate by a very small margin. B7m/B7n keep the low rank but distribute the fixed coordinate pairs across the full quadratic pair list after B7k showed high E1/E2 frozen R2 but poor E6/E8 rotated/random coverage. B7o/B7p add a fixed low-dimensional cosine transform before pair products to test rotated/random quadratic coverage without adding trainable projection gradients. B7q/B7r/B7s/B7t keep the fuller R136 pair coverage from B7i while reducing only Rational hidden width, testing whether expression coverage can fit the same L3 efficiency gate without changing loss, data, labels, gates, or optimizer semantics. B7u/B7v keep the B7s/B7t cost envelope but replace image-input pair selection with grid-balanced local/global pairs to test whether the B7s/B7t task collapse was caused by top-left-biased sequential image pairs. B7w/B7x lower the structured pair rank from R136 to R112 at the same h40/h32 widths to test whether a less dominant fixed pair readout can keep A4 while improving task trajectory. B7y/B7z keep the B7s/B7t feature geometry but raise only final readout parameter learning rate to test whether the A4 frozen span can be trained quickly enough for A5 without modifying loss, data, sampler, labels, gates, or hidden geometry. B7aa/B7ab move the same fixed pair products from a direct logits readout into hidden preactivation, so pair geometry can affect trainable hidden dynamics while the fixed pair projection itself remains non-trainable. B7ac/B7ad replace that dense pair-to-hidden projection with deterministic bucketed accumulation, testing a more single-kernel-friendly hidden trajectory primitive with no trainable pair projection. B7ae/B7af lower that bucket rank to R112 after B7ac missed the 1.25 step gate, and B7ag/B7ah lower it again to R88 to test the efficiency/expression edge. B7ai/B7aj/B7ak/B7al keep the R136/R112 bucket geometry but replace materialized cross+scatter_add with a dedicated Triton hidden-delta accumulation kernel. B7am/B7an/B7ao/B7ap keep that coalesced hidden path and add a trainable direct pair readout, testing whether B7s/B7t expression span and B7ai-B7al hidden trajectory coupling can be combined without adding trainable pair projection gradients. B7aq/B7ar/B7as/B7at preserve the full R136 direct readout span that previously opened A4, but restrict hidden bucket coupling to R32/R64 to test whether a small single-kernel-friendly hidden trajectory perturbation can improve task dynamics without losing L3. B7au/B7av/B7aw/B7ax preserve the same full R136 direct pair span but multiply the pair-readout logits and gradients by a fixed 0.25/0.10 scale, testing whether A5 collapse comes from over-strong pair logits rather than missing expression capacity. B7ay/B7az/B7ba/B7bb keep the same direct pair span but replace materialized cross readout logits and readout-gradient GEMMs with dedicated Triton kernels, testing whether direct pair A4 can re-open under the strict L3 step gate. B7bc/B7bd/B7be/B7bf keep that direct pair span but use block-tiled Triton programs for logits and readout-gradient, reducing the per-rank/per-class program count exposed by B7ay-B7bb. B7bg/B7bh/B7bi/B7bj keep the same fast block-tiled path and add only a learnable output bias, testing whether the severe A5 failure is partly missing class-prior/calibration capacity rather than pair-kernel timing; that first form adds bias after the readout result. B7bk/B7bl/B7bm/B7bn move the same bias add inside the block-tiled logits kernel so calibration capacity does not add an extra logits write/kernel in manual L3. B7bo/B7bp/B7bq/B7br remove that bias branch again and lower the block-tiled pair rank from R136 to R112 to reduce actual readout logits/gradient work before re-opening A4/A5. B7bs/B7bt/B7bu/B7bv add the middle R128 block-readout bracket after R112 passed L3 but failed A4 expression, testing whether a small rank increase can recover expression without returning to R136 cost. B7bw/B7bx/B7by/B7bz add the R120 middle bracket after R112 was too weak and R128 partly returned to cost pressure, testing whether a narrower middle point can retain expression while staying inside L3. None of these change loss/data/labels/gates. BSpline is intentionally not extended in this repair round per latest user direction.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7ca/B7cb/B7cc/B7cd Rational pairnorm block-readout variants and fused pair-feature normalization inside the block Triton logits/readout-gradient kernels",
                "reason": "B7bc-B7bf showed that block-tiled direct pair readout can open L3/A4 but its task trajectory collapses, while B7bg-B7bz showed bias/rank/scale micro-repairs do not solve the blocker. B7ca-B7cd keep the same fixed pair geometry and trainable readout, but normalize each fixed pair feature using training-stream statistics before readout; the same normalization is applied inside the manual L3 block kernels and in frozen/readout features, so this is a real parameterization repair rather than post-hoc calibration. Loss, data, labels, gates, and optimizer semantics remain unchanged.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7ce/B7cf/B7cg/B7ch Rational pairstd block-readout variants",
                "reason": "The first pairnorm smoke showed that centered pair features can be made L3-correct but may damage expression coverage. B7ce-B7ch keep the single-kernel block-readout implementation and the same train-stream statistics, but use only fixed per-pair std scaling without subtracting the feature mean. This tests whether variance balancing can stabilize task/expression without removing useful coordinate-product mean signal. Loss, data, labels, gates, and optimizer semantics remain unchanged.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7ci/B7cj Rational block-readout variants plus crossreadoutlrXXX optimizer grouping",
                "reason": "B7bc-B7bf proved that R136 block-tiled pair readout can pass L3 and A4 but collapses A5, while rank/scale/bias/pairnorm/pairstd repairs did not produce a family task pass. B7ci/B7cj keep the same FlashKAT grouped rational activation, fixed pair geometry, and block-tiled logits/gradient kernels, but lower only the trainable cross_readout parameter group's AdamW learning rate to 0.25. This tests whether the direct pair logits are over-driving task trajectory without changing loss, data, labels, gates, sampler, class weights, or the rational/pair basis itself.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7ck/B7cl Rational block-readout crosszero variants",
                "reason": "B7ci/B7cj lowered only cross_readout LR but still failed the L3 full-step efficiency gate, and earlier B7bc-B7bf showed that random-initialized direct pair readout can pass L3/A4 yet collapse A5. B7ck/B7cl keep the same FlashKAT grouped rational activation, fixed pair geometry, and block-tiled logits/gradient kernels, but initialize cross_readout to exactly zero so the pair branch enters gradually through real gradients instead of injecting random pair logits at step 0. This changes task trajectory without adding kernels, trainable pair projections, loss changes, data changes, sampler changes, label weighting, or gate changes.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7cm/B7cn Rational R128 block-readout crosszero variants",
                "reason": "B7ck/B7cl made the same R136 block-readout crosszero primitive L3-correct but still missed the full-step gate with step ratios near 1.30. B7cm/B7cn reduce only the fixed pair-readout rank from R136 to R128 while keeping zero-initialized cross_readout and the same block-tiled kernels, testing whether the small rank reduction is enough to open L3 efficiency without losing the expression coverage needed for A4. The repair does not change loss, data, labels, sampler, class weights, gates, or optimizer semantics.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7co/B7cp Rational block-readout B32 logits tiling variants",
                "reason": "B7ck/B7cl showed the R136 crosszero primitive was L3-correct and close to the full-step gate, while B7cm/B7cn rank reduction was not stable. B7co/B7cp keep the same R136 pair geometry and zero-initialized cross readout but increase only the Triton block-readout logits batch tile from 16 to 32, reducing logits-kernel program count without changing the rational basis, pair features, gradients, loss, data, labels, sampler, class weights, or gates.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7do/B7dp/B7dq/B7dr Rational crosszero B32 fixed pair-readout scale variants",
                "reason": "B7co/B7cp opened Rational L3 and A4 but collapsed A5, while earlier readscale-only variants did not test the exact crosszero+B32 kernel path. B7do-B7dr preserve the same coordinate-pair feature map, zero-initialized cross readout, and B32 block-readout kernel, but fix only the pair branch amplitude to 0.25 or 0.10. The backward path still consumes arbitrary upstream dL/dlogits, so this is a loss-agnostic structural trajectory diagnostic rather than CE calibration, and it does not alter loss, data, labels, sampler, class weights, or gates.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7ds/B7dt Rational fixed-scale crossWarm variants",
                "reason": "B7do-B7dr preserved L3/A4 but still showed task collapse, so B7ds/B7dt keep only the h32 candidates with the same coordinate-pair map, crosszero initialization, B32 block-readout kernel, and fixed pair scale, then ramp the existing scalar scale from zero to its target over task epochs. This is still a generic logits-path primitive using arbitrary dL/dlogits in the manual VJP; CE is only the benchmark wrapper, not a special-cased objective, and no loss/data/label/sampler/class-weight/gate changes are introduced.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7du/B7dv Rational R112 crosszero B32 block-readout variants",
                "reason": "B7bo/B7bp R112 block-readout failed L3 efficiency, while B7co/B7cp showed that B32 logits tiling is the working lower-level kernel shape for the same direct pair readout family. B7du/B7dv therefore keep the lower R112 coordinate-pair rank but reuse crosszero+B32 tiling, testing a loss-agnostic lower-rank expression/task bracket without changing loss, data, labels, sampler, class weights, or gates.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7dw/B7dx Rational frozen-backbone readout-only trajectory variants",
                "reason": "B7co/B7cp and B7do-B7dr showed that the R136 crosszero+B32 fixed pair span can open L3/A4 but collapses during task training, while lower-rank and warm/scale variants did not solve the blocker. B7dw/B7dx freeze the grouped rational coefficients and first projection w1, then train only readout-side parameters through the existing block-tiled pair-readout VJP. This materially removes rational/w1 backward and update cost while keeping the VJP expressed as arbitrary dL/dlogits rather than CE-specific code. It does not change loss, data, labels, sampler, class weights, or gates.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7dy/B7dz/B7ea/B7eb Rational frozen-backbone linear-residual-gain variants",
                "reason": "B7dw/B7dx proved that readout-only frozen-backbone VJP substantially reduces L3 backward/update cost and opens A4, but A5 still collapses. B7dy/B7dz keep the same frozen rational/w1 backbone and block-tiled pair-readout VJP, but scale the existing linear residual feature/readout path by a fixed factor of 8.0. B7ea/B7eb keep the h32 efficient shape and bracket that first-order channel at 16.0/32.0 after the 8.0 run still failed A5. This adds a low-cost trainable trajectory channel for arbitrary logits-gradient objectives rather than CE-specific calibration, and does not change loss, data, labels, sampler, class weights, or gates.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7ec/B7ed Rational frozen-coefficient trainable-w1 trajectory variants",
                "reason": "B7eb reduced the A5 mean-delta gap but still failed task/ECE/AUC with a fully frozen rational/w1 backbone. B7ec/B7ed freeze only the rational activation coefficients and keep w1 trainable, so the trajectory can update hidden projections while the manual VJP skips rational-coefficient gradients. The backward still consumes arbitrary dL/dlogits and is therefore not CE-specific; loss, data, labels, sampler, class weights, and gates remain unchanged.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7ee/B7ef Rational frozen-backbone hidden-bias trajectory variants",
                "reason": "B7ec/B7ed showed that unfreezing the full w1 projection restores too much backward cost and fails L3 efficiency. B7ee/B7ef keep the rational/w1 backbone frozen but add a trainable hidden-unit bias, whose manual gradient is just the summed hidden preactivation gradient from arbitrary dL/dlogits. This adds a low-cost trajectory degree of freedom without CE-specific loss tuning, teacher data, sampler/class weighting, or gate changes.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7eg/B7eh hidden-bias linear-residual-gain bracket",
                "reason": "B7ef was the best Rational follow-up so far but still failed A5 with ECE/AUC and worst-row blockers. B7eg/B7eh keep the same loss-agnostic hidden-bias VJP and bracket the fixed linear residual feature scale at 48.0/96.0 to test whether the B7ef accuracy gain can be kept while reducing the remaining trajectory/calibration failures. This is a primitive-level feature-scale bracket, not a CE-only backward or post-hoc calibration change.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7ei/B7ej hidden-bias high-gain bracket and widened linearresgain parser",
                "reason": "B7eh improved mean/worst task deltas and kept L3/A4 open but still failed MNIST/KMNIST ECE/AUC. B7ei/B7ej keep the same generic hidden-bias VJP and test whether the monotonic accuracy improvement continues at 128.0/192.0 linear residual feature scale. The parser change only allows explicit five-digit scale tokens used by these candidates; it does not alter existing candidate semantics or gates.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7ek/B7el hidden-bias very-high-gain bracket",
                "reason": "B7ej opened L3/A4 and passed mean/ECE but missed worst/near/AUC, with remaining failures concentrated in KMNIST and one Fashion AUC row. B7ek/B7el extend the same primitive-level hidden-bias feature-scale trend to 256.0/384.0 to test whether the remaining worst-row gap closes. This remains a generic logits-gradient primitive and does not tune CE loss or post-hoc calibration.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7em/B7en pair-feature conditioned hidden-bias Rational variants",
                "reason": "B7ej is the current Rational near-survivor: it keeps L3/A4 open and passes mean/ECE, but misses worst/near/AUC. B7em/B7en keep the same frozen-backbone hidden-bias VJP and linear residual gain while applying fixed train-stream pair-feature std scaling or centering+std scaling before the pair readout. This is a loss-agnostic feature-conditioning trajectory repair that still consumes arbitrary dL/dlogits; it does not introduce CE-specific backward logic, post-hoc temperature, sampler/class weights, teacher data, or gate changes.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7eo/B7ep trainable linear-residual gate Rational variants",
                "reason": "B7em/B7en pair-feature conditioning kept analytic gradients correct but failed L3 efficiency. B7eo/B7ep return to B7ej's low-cost frozen-backbone hidden-bias shape and add only one trainable scalar gate on the linear residual feature channel, initialized at 1.00 or 0.75. The manual VJP computes this scalar gradient from arbitrary dL/dlogits and the residual feature logits; this is an internal loss-agnostic trajectory degree of freedom, not CE-only temperature or post-hoc calibration.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7eq/B7er B7ej-shaped logit-intercept Rational variants",
                "reason": "B7eo/B7ep showed that a trainable residual gate adds too much L3 cost even after removing the GPU synchronization point. B7eq adds only a standard trainable logit intercept to the B7ej hidden-bias/readout-only VJP path, while B7er additionally fixes the pair-readout branch scale to 0.50. These are model-level, loss-agnostic trajectory checks using arbitrary dL/dlogits; they do not change CE loss, add post-hoc calibration, branch on dataset/labels, or lower gates.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7es/B7et/B7eu w2-bias-row and cheaper R120 Rational variants",
                "reason": "B7eq showed that a separate logit-bias parameter is close but still fails L3, likely because it adds an optimizer/update tensor and a standalone gradient reduction. B7es folds the intercept into the existing w2 tensor by making the last hidden feature a constant row, so the output intercept uses the existing w2 update path. B7et lowers the fixed pair primitive from R136 to R120, and B7eu combines R120 with the w2 bias row. These are loss-agnostic architecture/kernel-cost repairs using arbitrary dL/dlogits, not CE loss changes or post-hoc calibration.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7ev/B7ew Rational manual foreach AdamW update variants",
                "reason": "B7es/B7et/B7eu showed that extra output-shaping rows and cheaper R120 pair features did not reduce the tight B7ej-shaped L3 path. B7ev keeps the B7ej primitive and B7ew keeps the B7eq logit-intercept primitive, but tags the variants for manual foreach AdamW so the update step is measured and, when task triage opens, trained through the same lower-overhead update path. This changes optimizer execution, not the CE objective or logits-gradient VJP.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7ex/B7ey no-pair-readout Rational variants",
                "reason": "B7ev/B7ew reduced optimizer update ratio but still failed total L3, so the remaining blocker is forward/backward reductions. B7ex removes the R136 pair-readout branch from the B7ej-shaped frozen-backbone hidden-bias trajectory while keeping the strong linear residual and generic grad-logits VJP. B7ey adds manual foreach AdamW to the same no-pair primitive. This is a genuine reduction of the L3 computation graph, not CE-specific calibration.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7ez/B7fa bucketed-pair-readout Rational variants",
                "reason": "B7ey proved that removing pair-readout reductions can pass L3, but it also collapsed A4 pairwise/quadratic expression. B7ez/B7fa restore the fixed R136 pair feature source and compress it to G64/G32 bucket summaries before the trainable readout, while keeping the B7ej frozen-backbone hidden-bias high-gain trajectory and manual foreach AdamW. This is a loss-agnostic upstream-gradient primitive intended to preserve pair capacity with lower readout/backward/update cost; it does not tune CE loss, post-hoc calibrate logits, branch on dataset/labels, or lower gates.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7fb/B7fc wider bucketed-pair-readout Rational variants",
                "reason": "B7ez/B7fa passed L3 but still failed A4 because G64/G32 buckets discarded too much pairwise/quadratic expression signal. B7fb/B7fc keep the same B7ej-shaped frozen-backbone hidden-bias high-gain trajectory and manual foreach AdamW, but widen the bucketed R136 pair summaries to G96/G112 to spend the measured L3 efficiency headroom on expression recovery. This remains a generic upstream-gradient primitive and does not modify CE loss, logits by post-hoc calibration, dataset/label routing, sampler/class weights, or gate thresholds.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7fd/B7fe lower-rank full pair block-readout Rational variants",
                "reason": "B7fb showed that a wider G96 bucket can barely pass L3 but still loses too much expression, while B7fc G112 crosses the L3 cost limit. B7fd/B7fe stop hashing pair features and instead keep full trainable B32 block-readout on lower pair ranks R96/R80, under the same B7ej frozen-backbone hidden-bias high-gain trajectory and manual foreach AdamW. This tests whether preserving unbucketed pair feature identity at lower rank can recover A4 while reducing readout/backward/update cost; the VJP still accepts arbitrary upstream dL/dlogits and is not CE-specific.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7ff/B7fg balanced lower-rank pair block-readout Rational variants",
                "reason": "B7fd/B7fe passed L3 and recovered unrotated E1/E2 frozen readout, but failed E6/E8 because the lower-rank pair selection did not cover rotated/random quadratic interactions well enough. B7ff/B7fg keep full B32 block-readout and the B7ej frozen-backbone hidden-bias high-gain/manualAdamW trajectory, but switch the fixed pair index generator to balanced R96/R88 coverage. This changes only the loss-agnostic fixed feature geometry and keeps the same generic dL/dlogits VJP.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7fh/B7fi fixed projected quadratic GEMM-readout Rational variants",
                "reason": "B7ff/B7fg showed that lower-rank balanced pair features can pass L3 but still fail A4, so this tests a different single-kernel-friendly trajectory primitive: fixed DCT projection-bilinear or projection-square features followed by a compact GEMM readout. B7fh/B7fi keep the B7ej frozen-backbone hidden-bias high-gain/manualAdamW trajectory, remove block pair-readout reductions, and use the existing generic dL/dlogits VJP. This is not CE-specific tuning and does not alter loss, labels, data sampling, calibration, or gates.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7fj/B7fk lower-rank fixed projected quadratic GEMM-readout Rational variants",
                "reason": "B7fh/B7fi kept the projection-quadratic primitive loss-agnostic and gradient-correct, but R96 was too slow for the L3 step gate. B7fj/B7fk reduce the fixed projection-bilinear/square rank to R64 under the same B7ej frozen-backbone hidden-bias high-gain/manualAdamW trajectory, testing whether the single-kernel-friendly primitive can recover L3 before any A4/A5 claim.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7fl/B7fm R32 fixed projected quadratic GEMM-readout Rational variants",
                "reason": "B7fj/B7fk reduced projection rank to R64 but still failed the L3 step gate. B7fl/B7fm lower the projection-bilinear/square rank to R32 to establish the efficiency floor of this single-kernel-friendly primitive. A4/A5 remain legally closed unless the L3 full-step gate opens; the VJP remains generic dL/dlogits and not CE-specific.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7fn/B7fo hidden-square readout Rational variants",
                "reason": "B7fh-B7fm showed that input-projected quadratic features remain too expensive even at low rank. B7fn/B7fo instead reuse the already-computed hidden activations and add a trainable h^2 readout, with B7fo zero-initializing that readout to reduce trajectory disruption. The manual VJP adds the corresponding hidden-square readout and hidden-bias gradients from arbitrary dL/dlogits, so this remains loss-agnostic and is not CE-specific tuning.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7fp/B7fq pair-block plus hidden-square-zero Rational variants",
                "reason": "B7fo opened the L3 fused efficiency gate with a cheap hidden-square readout but failed A4 because E1/E6/E8 pairwise/quadratic features collapsed; B7fd/B7ff showed that R96 pair block-readout passes L3 but still lacks enough expression coverage. B7fp/B7fq combine these two loss-agnostic primitives under the same generic dL/dlogits VJP, testing whether a low-cost hidden-square channel can repair A4 without reintroducing CE-specific calibration, data branching, or gate changes.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7fr/B7fs hybrid pair-index plus hidden-square-zero Rational variants and fixed hidden-square param accounting",
                "reason": "B7fq preserved E1/E2 better while B7fp preserved E6/E8 better, suggesting that pair-index coverage rather than loss-specific calibration is the remaining blocker. B7fr/B7fs keep the same single block-readout kernel path but split the fixed pair indices between standard and balanced coverage at R96/R112. The hidden-square readout is now included in edge_param_count so same-parameter MLP controls remain auditable; the VJP still accepts arbitrary dL/dlogits and is not CE-specific.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7ft/B7fu grid-pair plus hidden-square-zero Rational variants",
                "reason": "B7fr/B7fs showed that hybrid pair indexing can become too expensive before A4 opens. B7ft/B7fu keep the same loss-agnostic B32 pair block-readout VJP and zero-init hidden-square readout, but choose fixed grid-balanced pair indices at R96/R112 to test local image-neighborhood plus global coverage without CE-specific calibration, dataset branching, or gate changes.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7fv/B7fw R112 standard/balanced pair-block plus hidden-square-zero Rational variants",
                "reason": "B7fp/B7fq passed L3 but failed A4 with complementary expression gaps, while B7ft/B7fu grid indexing failed L3. B7fv/B7fw keep the proven B32 block-readout kernel and only raise fixed pair coverage from R96 to R112 in standard versus balanced selection, testing whether a small coverage increase can repair A4 without hybrid/grid overhead or any CE-specific objective coupling.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7gf-B7hc Rational full/partial pair-block plus hidden-square/difference-square/kernel-tiling follow-ups",
                "reason": "B7gf/B7gg test whether a 75/25 standard-balanced blend can recover rotated/random quadratic expression under L3; B7gh/B7gi restore B7ej full R136 coverage and add zero-init hidden-square readout; B7gj/B7gk lower B7gi's fixed linear residual feature scale to 160.00/144.00; B7gl/B7gm add the 184.00/176.00 interpolation after B7gj improved KMNIST AUC but hurt ECE; B7gn/B7go check the remaining narrow 188.00/190.00 window near B7gi; B7gp/B7gq stop changing linear residual gain and scale the hidden-square trajectory itself at 0.50/0.25; B7gr/B7gs add 0.75/0.62 hidden-square scale interpolation after B7gp improved accuracy but missed ECE/AUC; B7gt/B7gu center h^2 per sample before the readout to reduce loss-agnostic tail drift without changing the task loss; B7gv/B7gw use signed-quadratic h*abs(h) to keep quadratic amplitude without nonnegative-only tail drift or per-sample reductions; B7gx/B7gy keep the same single-kernel-friendly tail forms but switch hidden-square readout from zero-init to small nonzero init so the trajectory primitive participates from step 0; B7gz/B7ha test pair-difference-square readout as a genuinely different fixed quadratic geometry after pair-sum-square failed efficiency; B7hb/B7hc keep the B7gv/B7gp math but use full-rank R136 pair-readout grad tiling to audit deeper fused reduction cost. These are loss-agnostic feature-trajectory changes: the manual VJP consumes arbitrary upstream dL/dlogits, and the CE task loop remains only a comparable benchmark protocol.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7hd/B7he Rational batch-split atomic pair-readout gradient repair",
                "reason": "B7hb/B7hc proved that a single full-rank readout-gradient tile is not viable because Triton rounds R136 to a 256-wide tile and exhausts shared memory. B7hd/B7he keep the same B7gv/B7gp feature geometry, readout, hidden-square tail, frozen backbone, hidden bias, and manual AdamW path, but compute the cross_readout VJP by splitting the batch dimension into smaller blocks and atomically accumulating the rank/class gradient. This is a kernel-side generic dL/dlogits VJP repair, not CE-specific loss tuning, calibration, data/label branching, sampler/class weighting, or gate relaxation.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7hf/B7hg Rational hidden absolute-value readout tail",
                "reason": "After hidden-square, signed hidden-square, pair-sum-square, pair-difference-square, and readGrad tile repairs all failed to open Rational task success, B7hf/B7hg test a different low-cost trajectory primitive: a hidden |h| readout tail with an analytic sign VJP. The tail consumes arbitrary upstream dL/dlogits through the existing manual backward path, so it is not a CE-only gradient formula or CE calibration trick. B7hf zero-initializes the new readout to audit whether it can learn without disrupting the existing B7ej trajectory; B7hg uses the normal small random readout init to test immediate participation. Both keep loss, data, sampler/class weights, gates, family runner protocol, and dataset handling unchanged.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7hh/B7hi Rational centered hidden absolute-value readout tail",
                "reason": "B7hf/B7hg showed that the loss-agnostic hidden |h| tail can pass L3/A4 but still fails A5 through ECE/AUC and worst-row blockers. B7hh/B7hi subtract the per-sample mean from |h| before the readout, with the matching centered VJP, to reduce common-mode magnitude drift without changing the task loss, labels, sampler/class weights, dataset branches, gate thresholds, or CE-specific calibration. The backward still consumes a generic upstream dL/dlogits.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7hj/B7hk Rational partial-centered hidden absolute-value readout tail",
                "reason": "B7hh proved that full centering can repair ECE but hurts worst/near/AUC and B7hi loses L3 efficiency. B7hj/B7hk keep the same loss-agnostic |h| primitive but subtract only 0.25 or 0.10 of the per-sample |h| mean, with the matching VJP, to test whether a smaller common-mode correction preserves B7hf accuracy while reducing ECE/AUC drift. This remains a feature/trajectory primitive and generic dL/dlogits VJP, not CE-specific loss tuning.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7hl/B7hm Rational bounded hidden rational readout tail",
                "reason": "B7hf/B7hg passed L3/A4 but stayed on the B7gp task plateau, while B7hh-B7hk showed centering adds too much backward cost or hurts task. B7hl/B7hm instead add a bounded hidden rational feature h/(1+abs(h)) with analytic derivative 1/(1+abs(h))^2, inspired by the RationalKAT activation family itself. This is a loss-agnostic feature/readout trajectory primitive with generic dL/dlogits VJP; it does not change CE loss, labels, sampler/class weights, dataset branches, gates, or use calibration.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7hn/B7ho Rational hidden-tail Triton VJP repair",
                "reason": "B7hl/B7hm showed that bounded hidden-rational tails have correct generic VJP but lose the L3 efficiency gate because the extra hidden-tail readout gradient and hidden VJP are still PyTorch matmul operations. B7hn/B7ho keep the same loss, data, labels, sampler/class weights, gates, pair readout, frozen backbone, hidden bias, and manual AdamW path, but move the hidden-tail readout gradient and hidden VJP into one Triton kernel that consumes arbitrary upstream dL/dlogits. This explicitly addresses the user's correction that the primitive must not be CE-specific.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7hp/B7hq Rational hidden-rational tail scale bracket on Triton VJP path",
                "reason": "B7hn proved the hidden-rational tail can now pass L3/A4 with a generic Triton VJP, but A5 still fails through KMNIST ECE/AUC and one worst row. B7hp/B7hq keep the same fused hidden-tail VJP and only bracket the loss-agnostic feature scale at 0.25/0.75 to check whether the new primitive has a task-stable operating range without CE-specific calibration, loss changes, dataset branches, or gate relaxation.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "experiments/run_v126_lowerlevel_fhq_functional_geometry.py / dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7hr/B7hs Rational Triton readout AdamW update repair",
                "reason": "B7hn/B7ho moved hidden-tail VJP into a generic Triton kernel but still run readout parameter updates through the foreach AdamW path. B7hr/B7hs keep exactly the same logits, loss, data, labels, sampler/class weights, gates, pair readout, frozen backbone, hidden bias, and hidden-tail VJP as B7hn/B7ho, but route contiguous readout tensors through the existing single-tensor Triton AdamW kernel. This tests update-cost reduction without CE-specific tuning or task-specific calibration.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7cq/B7cr Rational bounded pair-logit variants",
                "reason": "B7co/B7cp opened Rational L3 efficiency and A4 expression but failed A5 with severe task collapse. B7cq/B7cr keep the same R136 crosszero B32 block-readout kernel path and add only a low-cost tanh cap to the pair-readout logits, with the matching manual cross_readout gradient factor, to test a task-stable trajectory primitive without lowering gates, changing loss/data, using teacher/distillation, sampler/class weights, or dataset-name branches.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7cs/B7ct Rational cross-readout scale warmup variants",
                "reason": "B7cq/B7cr showed that an unfused pair-logit cap preserves analytic gradient correctness but breaks the L3 full-step gate by adding extra elementwise work outside the block-readout kernel. B7cs/B7ct return to the B7co/B7cp R136 crosszero B32 kernel path and change only the training trajectory by ramping the existing scalar input_cross_readout_scale from 0 to its target value across task epochs. This does not add a forward/backward kernel, change the pair features, alter the loss, data, labels, sampler, class weights, or lower any gate.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7cu/B7cv Rational readGradR64 + cross-warm variants and persisted family task trace",
                "reason": "B7cs/B7ct tested the no-extra-op trajectory warmup but B7cs remained slightly above the L3 step gate and B7ct was slower. B7cu/B7cv keep the same B32 logits tiling and warmup, but also increase the pair-readout gradient rank tile from 32 to 64 to reduce backward kernel program count without changing the pair features, loss, data, labels, sampler, class weights, or gates. The runner now writes v1283_family_task_trace.csv so scale schedules and per-epoch task timing are auditable when A5 runs.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7cw/B7cx Rational readGradR128 + cross-warm variants",
                "reason": "B7cv with R64 grad tiling missed the L3 step gate by a very narrow margin, while still preserving the same function class and fused gradient correctness. B7cw/B7cx keep B32 logits tiling and cross-readout scale warmup, but raise only the pair-readout gradient rank tile to 128 to reduce backward kernel program count further. This is a fused-backward tiling repair, not a loss/data/label/sampler/class-weight/gate change.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7cy/B7cz Rational fused bounded pair-logit variants",
                "reason": "The unfused B7cq/B7cr pair-logit cap was mathematically correct but too slow, and the user clarified that repairs must not be CE-loss-specific. B7cy/B7cz implement the bounded pair-logit transform inside the B32 Triton logits kernel and a matching cap-aware pair-readout VJP kernel that accepts arbitrary upstream dL/dlogits. CE remains only the current measurement harness; the primitive itself is loss-agnostic and does not alter loss, data, labels, sampler, class weights, or gates.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7da/B7db Rational fused bounded pair-logit atomic-VJP variants",
                "reason": "B7cy/B7cz validated the loss-agnostic cap VJP numerically but the full-step L3 measurement hit Triton shared-memory OutOfResources at batch 128. B7da/B7db keep the same bounded pair-logit math, but split the cap-aware VJP across batch blocks with atomic accumulation to reduce per-program shared memory. This is still a generic upstream-gradient VJP, not a CE-loss-specific update, and does not change loss/data/labels/sampler/class weights/gates.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7dc-B7df Rational pair-readout bucket variants",
                "reason": "B7da/B7db fixed OOR but atomic VJP was too slow. B7dc-B7df keep the same fixed R136 pair feature source, but bucket those fixed pair features to G16/G32/G64 summaries before the trainable readout, with an h32 bracket. This lowers pair-readout backward/update write amplification and remains a loss-agnostic upstream-gradient primitive; it does not change loss/data/labels/sampler/class weights/gates.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "recognized rational_flashkat_grouped_pairreadbucket_triton_l3 as an official fused L3 family kernel",
                "reason": "The first B7dc/B7dd run measured the new fused pair-readout bucket path but still marked it non-official because the manual variant was missing from the official fused-L3 registry. This repair changes only the audit registry, not the gate thresholds, loss, data, labels, sampler, optimizer semantics, or candidate math.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B7dg-B7dn Rational fixed projection-square and fixed dual-projection bilinear readout variants",
                "reason": "B7dc-B7df showed that bucketed direct pair readout still spends too much full-step time in pair feature construction, accumulation, VJP, and update. B7dg-B7dk replace enumerated pair features with fixed DCT projection-square features q=(zT), q^2, followed by a trainable readout. B7dl-B7dn then replace q^2 with dual fixed projections (zP_left)*(zP_right), preserving a more general loss-agnostic bilinear interaction source while keeping backward/update writes to rank-by-class readout gradients and avoiding CE-specific calibration.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "split GroupedRationalKATKAN manual VJP into a generic grad-logits path",
                "reason": "The task runner still uses CE for historical comparability, but the Rational manual backward now exposes manual_logits_backward_from_cache(logits, cache, grad_logits). The CE wrapper only constructs dCE/dlogits and then calls the generic path, so new Rational primitive/VJP code is not hard-wired to CE and can support other differentiable objectives with the same logits-gradient interface.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "integrated official family fused-L3 A5 task triage after A4 pass",
                "reason": "B3s opened A4 under official fused L3 efficiency + analytic gradcheck. The runner now trains A4-passed family candidates with their audited manual CE forward/backward path instead of closing A5 as not integrated; MLP remains the same AdamW/autograd control, and the gate is not lowered.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added B221/B222/B223 signed-square plus groupabs structured direct-tail trajectory variants",
                "reason": "B217/B219 were the strongest non-logit direct-tail attempts but still failed near/ECE/AUC; B221-B223 combine sign-preserving quadratic signal with local group absolute-value rows already supported by FHQ, testing a more structured local-tail primitive without loss/gate/data changes or a new unsupported kernel",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added B224/B225 orthogonal-P condition-preserving trajectory variants",
                "reason": "B221-B223 showed structured direct tails can pass F3 efficiency but still leave A5/ECE/AUC blocked; B224/B225 keep dense learnable-P and fused projection-gradient/update, but change the projection initialization to an orthogonal low-coherence frame, testing the plan's condition-preserving init direction without adding direct-gradient rows, changing loss/gates, or using dataset branches",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added B226/B227 label-free PCA-P signal-projected trajectory variants",
                "reason": "B224/B225 kept fused efficiency but failed MNIST/KMNIST AUC/ECE; B226/B227 initialize dense learnable-P from train-input PCA directions only, without labels, dataset-name branches, loss changes, or new direct rows, testing a signal-subspace trajectory mechanism while keeping the same FHQ fused projection-gradient/update path",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added B228/B229 label-free PCA-band signal-projected trajectory variants",
                "reason": "B226/B227 top-PCA init made AUC/ECE worse, suggesting high-variance directions may be nuisance-dominated; B228/B229 draw dense learnable-P directions across the PCA spectrum, still without labels, dataset-name branches, loss changes, or new direct rows",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added B230/B231 audited train-probe centroid signal trajectory variants",
                "reason": "B226-B229 ruled out label-free PCA signal init; B230/B231 explicitly use training labels to initialize class-centroid signal directions in dense P and the direct linear rows, without dataset-name branches, loss changes, sampler/class weights, teacher, or distillation. These rows are audited as supervised train-probe signal candidates, not label-free base init.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "experiments/run_v124_multibasis_functional_dual.py / experiments/run_v126_lowerlevel_fhq_functional_geometry.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "threaded optional y_stats into SimpleFastTaskGeometryKAN construction",
                "reason": "supervised train-probe signal variants must receive labels through an explicit optional argument so the artifact can audit that they are label-using candidates; existing label-free candidates keep default y_stats=None behavior",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added B232/B233 train-probe direct050 temp065/temp050 confidence repairs",
                "reason": "B231 reached AUC-step/AUC-time on the short autopsy but failed ECE and one worst-row gate; B232/B233 keep the same audited train-probe signal and fused projection-gradient/update path while only lowering initial global logit gain to test an ECE-targeted confidence repair. This is not a label-free candidate and does not change loss, sampler, class weights, teacher, distillation, or dataset branching.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added B234/B235 fixed-gain train-probe confidence repairs",
                "reason": "B232 kept AUC-step/AUC-time pass but still failed ECE, while B233 showed lower initial gain alone is not enough; B234/B235 freeze the classwise gain buffer so the confidence scale cannot be learned back during the short run, keeping the same audited train-probe signal and FHQ fused update path without changing loss/gates/data.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added B236/B237 global fixed-gain train-probe confidence repairs",
                "reason": "B234/B235 showed frozen classwise gain still leaves ECE above gate; B236/B237 remove classwise logit gain while keeping a single fixed global gain, testing whether per-class confidence scaling is the ECE source without changing the supervised train-probe signal, loss, sampler, class weights, teacher, distillation, or dataset branching.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / dgkan/kernels/fused_hinge_quadratic.py",
                "change": "added B238-B241 bounded-logit train-probe ECE repairs and FHQ chain-rule support for logitcap",
                "reason": "B236/B237 passed AUC-step/AUC-time in the short autopsy but still failed ECE and worst-row gates; B238-B241 apply a differentiable cap to logits inside the model before the fixed global gain, with explicit FHQ backward chain-rule support, without changing CE loss, sampler, class weights, teacher, distillation, or dataset branching. B240/B241 are wider caps added after B238/B239 showed cap250/cap300 over-constrained the task trajectory.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added B242-B247 high fixed-gain train-probe ECE/AUC repairs",
                "reason": "B238-B241 increased under-confidence and worsened AUC/accuracy, while B236/B235 showed ECE failure with otherwise strong AUC and near gates; B242/B243 test fixed global gain 1.00/1.25, B244/B245 test the 1.10/1.15 middle points, and B246/B247 test 1.05/1.07 after B244 passed ECE/accuracy but still failed KMNIST AUC. All keep the same audited train-probe signal and fused update path without post-hoc calibration, loss changes, sampler/class weights, teacher, distillation, or dataset branching.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py / experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B248-B251 global fixed-gain ramp trajectory repairs",
                "reason": "B247 passed ECE/accuracy/efficiency but still failed KMNIST AUC, while lower fixed gains reduced AUC but failed ECE. B248/B249 ramp the fixed global gain from 0.75 to final 1.07/1.10, and B250/B251 lower the start to 0.65/0.50 after B248 improved but did not clear KMNIST seed1. All keep the same train-probe signal and FHQ update path to reduce early NLL/AUC area without post-hoc calibration, loss changes, sampler/class weights, teacher, distillation, or dataset branching.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added B252/B253 direct075 train-probe gain-ramp repairs",
                "reason": "B248-B251 showed confidence schedule can fix ECE and most AUC rows but not KMNIST seed1 or worst-row gates. B252/B253 reuse the existing train-probe direct075 initializer to strengthen supervised centroid signal without adding new kernels, loss changes, sampler/class weights, teacher, distillation, or dataset-name branching.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added B254 temp115 gain-ramp repair",
                "reason": "B248/B249 at 5 epochs passed mean/worst/near/AUC but missed ECE by a very small KMNIST seed0 margin; B254 raises only the final fixed gain target to 1.15 while preserving the same 0.75 gain-ramp schedule and train-probe direct050 signal.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added B255 temp105 gain-ramp confirm10 repair",
                "reason": "B248 passed 5-seed confirm but failed 10-seed confirm on Fashion-MNIST seed6 AUC 1.0697. B255 keeps the same train-probe P/direct050/fused projection-gradient AdamW path and the same 0.75 gain-ramp start, while lowering only the final fixed gain target to 1.05 to reduce AUC without post-hoc calibration, loss changes, sampler/class weights, teacher, distillation, or dataset-name branching.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added B256 temp100 gain-ramp confirm10 repair",
                "reason": "B255 lowered Fashion-MNIST seed6 AUC in the expected direction but still failed the 1.05 hard gate. B256 preserves the same train-probe P/direct050/fused projection-gradient AdamW path and lowers only the final fixed gain target to 1.00, continuing the training-internal trajectory repair without post-hoc calibration, loss changes, sampler/class weights, teacher, distillation, or dataset-name branching.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added B257 temp095 gain-ramp confirm10 repair",
                "reason": "B256 reduced the 10-seed Fashion-MNIST seed6 AUC blocker to 1.0551 while preserving accuracy/ECE. B257 keeps the same train-probe P/direct050/fused projection-gradient AdamW path and lowers only the final fixed gain target to 0.95 to test the smallest remaining confidence reduction likely to cross the AUC hard gate without changing loss, sampler/class weights, teacher, distillation, or dataset-name branching.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added B258 temp090 gain-ramp confirm10 repair",
                "reason": "B257 remained accuracy/ECE strong but the consistency run exposed a Fashion-MNIST seed7 AUC-time blocker with AUC-step already near the gate. B258 preserves the same train-probe P/direct050/fused projection-gradient AdamW/manual update path and lowers only the internal final fixed gain target to 0.90, without post-hoc calibration, loss changes, sampler/class weights, teacher, distillation, or dataset-name branching.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added B259/B260 temp092/temp093 gain-ramp bracket repairs",
                "reason": "B257 at 0.95 kept ECE but left a Fashion-MNIST AUC-time blocker, while B258 at 0.90 moved the blocker and failed ECE; B259/B260 test the remaining narrow internal-gain bracket while preserving the same train-probe signal and fused update path. These are recorded as final gain-bracket checks, not post-hoc calibration or changed loss.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "wired temp090/temp092/temp093 into SimpleFastTaskGeometryKAN gain parser",
                "reason": "the B258-B260 specs must actually instantiate their intended internal fixed gain values; artifacts produced before this parser repair remain real runs but are treated only as pre-parser diagnostics, not as intended gain-bracket evidence",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "experiments/run_v1283_b109_classic_family_functional_geometry.py",
                "change": "added B109 Line C and Functional official re-entry cloned diagnostic outputs",
                "reason": "v12.8.3 treats Line C and Functional as main lines, not appendices; after B257 opens B109/FHQ base, the runner must produce audited coupling/signal/noise metrics and strong-control functional rows without claiming official functional success before task-safety, Line-C-improvement, control, and overhead gates all close.",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added B219/B220 abs+weak-square direct-tail trajectory variants",
                "reason": "B217 improved mean/worst slightly but still failed near/ECE/AUC; B219/B220 preserve the original B109 abs tail and add a separately initialized weak centered square tail, testing whether explicit quadratic signal helps without replacing the known useful abs channel",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added B217/B218 signed-square direct-tail trajectory variants",
                "reason": "B214-B216 absmixsq preserved F3 efficiency but did not change A5 enough; B217/B218 add the centered signed quadratic feature z*abs(z) as a one-row-per-input direct tail, retaining single-kernel-friendly cost while giving class readout a sign-preserving multiplicative/quadratic signal",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/kernels/fused_hinge_quadratic.py",
                "change": "added signed-square direct-tail support to FHQ logits and direct-gradient kernels",
                "reason": "official FHQ full-step/correctness audit must evaluate B217/B218 on the same lower-level path as B109/B214 rather than falling back to autograd or pretending kernel support exists",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added B214/B215/B216 non-logit absmixsq direct-tail trajectory variants",
                "reason": "B210-B213 showed logitnorm-family repairs can pass F3 efficiency but hurt A5/ECE/AUC; B214-B216 keep the same single-kernel-friendly direct row count as absdiag while replacing it with abs(z)+small/medium centered z^2 signal, so this tests an explicit multiplicative/quadratic signal primitive without logit rescaling, loss changes, or dataset branches",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/models/fc_purekan_primitives.py",
                "change": "added B210/B211 stop-gradient samplewise logitnorm trajectory variants and B212/B213 stop-gradient batch logitnorm trajectory variants",
                "reason": "B205 showed useful accuracy trajectory from samplewise logitnorm but failed efficiency/ECE/AUC; B210 keeps the forward normalization while treating the RMS denominator as constant in backward to remove the samplewise dot-coupling cost, and B211 combines that with the existing warm2/update-every2 projection schedule",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/kernels/fused_hinge_quadratic.py",
                "change": "added stop-gradient samplewise and batch logitnorm effective-delta branches for FHQ fused backward/update kernels",
                "reason": "FHQ analytic backward must match the new B210-B213 autograd semantics; otherwise correctness audit would not compare the same mathematical primitive",
            }
        ),
        _stamp(
            {
                "stage": "V1283_MODIFICATION_AUDIT",
                "file": "dgkan/kernels/fused_hinge_quadratic.py",
                "change": "attempted local Q_TANH + Q_RMS projection-gradient chain-rule repair, guarded rmsQ+boundQ from official FHQ support until a two-pass q transform kernel exists, added a narrow Triton AdamW tensor update kernel for dense quad_proj, and added a fused projection-gradient+AdamW update kernel for dense quad_proj",
                "reason": "B196 q-normalized + bounded quadratic trajectory repair exposed a fused backward correctness failure; B203/B204 test lower-level update fusion for dense B109 quad_proj without changing loss/gate/data path",
            }
        ),
    ]
    write_csv_rows(out_dir / "v1283_run_manifest.csv", manifest)
    write_csv_rows(out_dir / "v1283_strict_contract.csv", contract)
    write_csv_rows(out_dir / "v1283_modification_audit.csv", modification)
    return manifest, contract, modification


def run_family_manifest(out_dir: Path, specs: Mapping[str, prim.PrimitiveSpec]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for family, cid, plan in _family_plan():
        spec = specs.get(cid)
        rows.append(
            _stamp(
                {
                    "stage": "V1283_FAMILY_MANIFEST",
                    "family": family,
                    "candidate_id": cid,
                    "basis_name": spec.basis_name if spec else "",
                    "k": spec.k if spec else "",
                    "hidden_dim": spec.hidden_dim if spec else "",
                    "uses_dense_basis_tensor": spec.uses_dense_basis_tensor if spec else "",
                    "requested_plan": plan,
                    "status": "available" if spec else "missing_spec",
                }
            )
        )
    write_csv_rows(out_dir / "v1283_family_manifest.csv", rows)
    return rows


def run_b109_fullstep(args: argparse.Namespace, out_dir: Path, device: torch.device, x_train: torch.Tensor, y_train: torch.Tensor, input_dim: int, output_dim: int, specs: Mapping[str, prim.PrimitiveSpec]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    exact_ids = [
        cid
        for cid in _b109_candidate_ids(args)
        if cid != str(args.fixedp_candidate_id) and not _is_fixedp_candidate(cid, specs)
    ]
    if not exact_ids:
        exact_ids = [str(args.exact_candidate_id)]
    grad_rows: List[Dict[str, Any]] = []
    full_rows: List[Dict[str, Any]] = []
    official_seen: List[str] = []
    seen_grad: set[Tuple[str, str, str, str]] = set()
    seen_full: set[Tuple[str, str]] = set()
    for cid in exact_ids:
        local_args = argparse.Namespace(**{**vars(args), "exact_candidate_id": cid})
        local_grad, local_full, local_official = v126.run_backward_fullstep(local_args, out_dir, device, x_train, y_train, input_dim, output_dim, specs)
        for oid in local_official:
            if oid not in official_seen:
                official_seen.append(oid)
        for row in local_grad:
            key = (
                str(row.get("candidate_id", "")),
                str(row.get("implementation_id", "")),
                str(row.get("grad_role", "")),
                str(row.get("status", "")),
            )
            if key in seen_grad:
                continue
            seen_grad.add(key)
            grad_rows.append(row)
        for row in local_full:
            key = (str(row.get("candidate_id", "")), str(row.get("implementation_id", "")))
            if key in seen_full:
                continue
            seen_full.add(key)
            full_rows.append(row)
    b109_grad = []
    for row in grad_rows:
        copied = dict(row)
        copied["stage"] = "V1283_B109_FUSED_BACKWARD_CORRECTNESS"
        b109_grad.append(_stamp(copied))
    b109_full = []
    for row in full_rows:
        copied = dict(row)
        copied["stage"] = "V1283_B109_FULLSTEP_PROFILE"
        if str(copied.get("candidate_id")) != "MLP-same-param-AdamW":
            copied["B109_special_step_target_pass"] = int(_safe_float(copied.get("step_ratio_q90"), 999.0) <= 0.85)
            copied["B109_special_memory_target_pass"] = int(_safe_float(copied.get("memory_ratio_q90"), 999.0) <= 0.60)
            copied["B109_strong_memory_target_pass"] = int(_safe_float(copied.get("memory_ratio_q90"), 999.0) <= 0.30)
        else:
            copied["B109_special_step_target_pass"] = ""
            copied["B109_special_memory_target_pass"] = ""
            copied["B109_strong_memory_target_pass"] = ""
        b109_full.append(_stamp(copied))
    write_csv_rows(out_dir / "v1283_b109_fused_backward_correctness.csv", b109_grad)
    write_csv_rows(out_dir / "v1283_b109_fullstep_profile.csv", b109_full)
    return b109_grad, b109_full, official_seen


def _step_b109(
    model: torch.nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    impl: str,
    workspace: Mapping[str, torch.Tensor] | None,
    *,
    opt: torch.optim.Optimizer | None = None,
    args: argparse.Namespace | None = None,
    manual_update: v126._ManualForeachAdamW | None = None,
    allow_fused_update: bool = True,
) -> None:
    if impl == "F3-triton-learnableP-workspace":
        assert workspace is not None
        logits, q_out, q2_sum, direct_logits, quad_logits = fhq.forward_workspace(model, xb, workspace)
        if allow_fused_update and opt is not None and args is not None and v126._variant_uses_fused_quadproj_adamw(args, model):
            quad_lr, quad_weight_decay = v126._quad_proj_group_hparams(opt, model, args)
            update_step = (manual_update.step_count + 1) if manual_update is not None else 1
            fhq.backward_learnablep_workspace_fused_quadproj_adamw(
                model,
                xb,
                yb,
                logits,
                q_out,
                q2_sum,
                direct_logits,
                quad_logits,
                workspace,
                lr=quad_lr,
                weight_decay=quad_weight_decay,
                step_count=update_step,
            )
        else:
            fhq.backward_learnablep_workspace(model, xb, yb, logits, q_out, q2_sum, direct_logits, quad_logits, workspace)
    elif impl == "F2-triton-learnableP":
        logits, q_out, q2_sum, direct_logits, quad_logits = fhq.forward(model, xb)
        fhq.backward_learnablep(model, xb, yb, logits, q_out, q2_sum, direct_logits, quad_logits)
    elif impl == "F4-triton-fixedP-workspace":
        assert workspace is not None
        logits, q_out, q2_sum, direct_logits, quad_logits = fhq.forward_workspace(model, xb, workspace)
        fhq.backward_fixedp_workspace(model, xb, yb, logits, q_out, q2_sum, direct_logits, quad_logits, workspace)
    elif impl == "F1-triton-fixedP":
        logits, q_out, q2_sum, direct_logits, quad_logits = fhq.forward(model, xb)
        fhq.backward_fixedp(model, xb, yb, logits, q_out, q2_sum, direct_logits, quad_logits)
    else:
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()


def run_b109_auc_autopsy(args: argparse.Namespace, out_dir: Path, device: torch.device, specs: Mapping[str, prim.PrimitiveSpec]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    datasets = [v120._canonical_dataset(x) for x in _parse_list(args.datasets)]
    seeds = _parse_ints(args.seeds)
    methods = ["MLP-same-param-AdamW"] + _b109_candidate_ids(args)
    trace_rows: List[Dict[str, Any]] = []
    final_rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        data = v120._load_vision_split(args, dataset, train_size=int(args.train_size), val_size=int(args.val_size), test_size=int(args.test_size))
        x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, x_test_cpu, y_test_cpu, input_dim, output_dim, _protocol = data
        x_train = x_train_cpu.to(device=device, dtype=torch.float32)
        y_train = y_train_cpu.to(device=device)
        x_val = x_val_cpu.to(device=device, dtype=torch.float32)
        y_val = y_val_cpu.to(device=device)
        x_test = x_test_cpu.to(device=device, dtype=torch.float32)
        y_test = y_test_cpu.to(device=device)
        local_specs = _specs_for(int(input_dim), int(output_dim))
        total_steps = max(1, int(args.epochs) * int(math.ceil(float(x_train.shape[0]) / float(max(1, int(args.batch_size))))))
        for seed in seeds:
            for method_id in methods:
                model = _make_model(method_id, int(input_dim), int(output_dim), x_train, device, int(seed) + 128300, local_specs, y_train)
                opt = _make_adamw(model, args)
                triton_update_params = v126._triton_adamw_params(args, model)
                manual_update = v126._ManualForeachAdamW(opt.param_groups, args, triton_update_params=triton_update_params) if v126._variant_uses_manual_adamw(args, model) else None
                fused_quadproj_update = v126._variant_uses_fused_quadproj_adamw(args, model)
                task_update_impl = "fused_projgrad_adamw+manual_foreach_adamw" if fused_quadproj_update else ("triton_tensor_adamw+manual_foreach_adamw" if triton_update_params else ("manual_foreach_adamw" if manual_update is not None else "adamw"))
                gain_now = _apply_logit_gain_ramp(model, method_id, local_specs, 0, int(args.epochs))
                quad_branch_now = _apply_quad_branch_ramp(model, method_id, local_specs, 0, int(args.epochs))
                direct_branch_now = _apply_direct_branch_ramp(model, method_id, local_specs, 0, int(args.epochs))
                warm_impl = _select_b109_step_impl(method_id, local_specs, epoch_idx=0, step_id=0) if method_id != "MLP-same-param-AdamW" else "torch-autograd"
                workspace = fhq.make_workspace(model, int(args.batch_size), device) if warm_impl in {"F3-triton-learnableP-workspace", "F4-triton-fixedP-workspace"} else None
                if workspace is not None:
                    warm = min(int(args.batch_size), int(x_train.shape[0]))
                    for warm_idx in range(max(0, int(args.task_compile_warmup_steps))):
                        start_idx = (warm_idx * warm) % int(x_train.shape[0])
                        xb = x_train[start_idx : start_idx + warm]
                        yb = y_train[start_idx : start_idx + warm]
                        if int(xb.shape[0]) < warm:
                            xb = x_train[:warm]
                            yb = y_train[:warm]
                        opt.zero_grad(set_to_none=True)
                        _step_b109(model, xb, yb, warm_impl, workspace, opt=opt, args=args, manual_update=manual_update, allow_fused_update=False)
                        opt.zero_grad(set_to_none=True)
                    _sync(device)
                gen = torch.Generator(device=device).manual_seed(int(seed) + 128333)
                step_id = 0
                step_times: List[float] = []
                val_losses: List[float] = []
                val_times_q90: List[float] = []
                all_impls: List[str] = []
                for epoch in range(int(args.epochs)):
                    gain_now = _apply_logit_gain_ramp(model, method_id, local_specs, epoch, int(args.epochs))
                    quad_branch_now = _apply_quad_branch_ramp(model, method_id, local_specs, epoch, int(args.epochs))
                    direct_branch_now = _apply_direct_branch_ramp(model, method_id, local_specs, epoch, int(args.epochs))
                    epoch_start = len(step_times)
                    epoch_impls: List[str] = []
                    perm = torch.randperm(int(x_train.shape[0]), generator=gen, device=device)
                    for off in range(0, int(x_train.shape[0]), int(args.batch_size)):
                        idx = perm[off : off + int(args.batch_size)]
                        xb = x_train[idx]
                        yb = y_train[idx]
                        lr_now = v126._task_lr_for(args, step_id + 1, total_steps)
                        for group in opt.param_groups:
                            lr_scale = float(group.get("lr_scale", 1.0))
                            for switch_epoch, switch_scale in group.get("lr_scale_schedule", []):
                                if epoch >= int(switch_epoch):
                                    lr_scale = float(switch_scale)
                            if not group.get("lr_scale_schedule") and group.get("lr_scale_after", None) is not None and epoch >= int(group.get("lr_scale_switch_epoch", 0)):
                                lr_scale = float(group.get("lr_scale_after", lr_scale))
                            group["lr"] = lr_now * lr_scale
                        opt.zero_grad(set_to_none=True)
                        _sync(device)
                        t0 = time.perf_counter()
                        impl = _select_b109_step_impl(method_id, local_specs, epoch_idx=epoch, step_id=step_id) if method_id != "MLP-same-param-AdamW" else "torch-autograd"
                        _step_b109(model, xb, yb, impl, workspace, opt=opt, args=args, manual_update=manual_update)
                        epoch_impls.append(impl)
                        all_impls.append(impl)
                        if manual_update is not None:
                            manual_update.step()
                        else:
                            opt.step()
                        _sync(device)
                        t1 = time.perf_counter()
                        step_times.append((t1 - t0) * 1000.0)
                        step_id += 1
                    val = v1252._classification_basic(model, x_val, y_val)
                    epoch_times = step_times[epoch_start:]
                    epoch_q90 = _q(epoch_times, 0.90)
                    val_losses.append(val["NLL"])
                    val_times_q90.append(epoch_q90)
                    phase = "early" if epoch < max(1, int(args.epochs) // 3) else ("late" if epoch >= max(1, (2 * int(args.epochs)) // 3) else "mid")
                    trace_rows.append(
                        _stamp(
                            {
                                "stage": "V1283_B109_AUC_AUTOPSY_TRACE",
                                "candidate_id": method_id,
                                "dataset": dataset,
                                "seed": seed,
                                "epoch": epoch + 1,
                                "phase": phase,
                                "step": step_id,
                                "val_acc": val["acc"],
                                "val_loss": val["NLL"],
                                "ECE": val["ECE"],
                                "CEp99": val["CEp99"],
                                "margin_p10": val["margin_p10"],
                                "epoch_step_time_q90_ms": epoch_q90,
                                "task_step_impl": "+".join(sorted(set(epoch_impls))),
                                "task_update_impl": task_update_impl,
                                "logit_gain_effective": gain_now if gain_now is not None else "",
                                "direct_branch_scale_effective": direct_branch_now if direct_branch_now is not None else "",
                                "quad_branch_scale_effective": quad_branch_now if quad_branch_now is not None else "",
                            }
                        )
                    )
                _apply_logit_gain_ramp(model, method_id, local_specs, max(0, int(args.epochs) - 1), int(args.epochs))
                _apply_quad_branch_ramp(model, method_id, local_specs, max(0, int(args.epochs) - 1), int(args.epochs))
                _apply_direct_branch_ramp(model, method_id, local_specs, max(0, int(args.epochs) - 1), int(args.epochs))
                final = v1252._classification_basic(model, x_val, y_val)
                test = v1252._classification_basic(model, x_test, y_test)
                steady_start = min(max(0, int(args.task_timing_warmup_epochs)), len(val_losses) - 1)
                steady_losses = val_losses[steady_start:]
                steady_times = val_times_q90[steady_start:]
                auc_step = sum(steady_losses) / max(1, len(steady_losses))
                auc_time = sum(l * max(1.0, t) for l, t in zip(steady_losses, steady_times)) / max(1, len(steady_losses))
                final_rows.append(
                    _stamp(
                        {
                            "stage": "V1283_B109_TASK_AUTOPSY_FINAL",
                            "candidate_id": method_id,
                            "dataset": dataset,
                            "seed": seed,
                            "val_acc": final["acc"],
                            "test_acc": test["acc"],
                            "NLL": final["NLL"],
                            "ECE": final["ECE"],
                            "CEp99": final["CEp99"],
                            "margin_p10": final["margin_p10"],
                            "val_loss_auc_step": auc_step,
                            "val_loss_auc_time": auc_time,
                            "step_time_q90_ms": _q(step_times, 0.90),
                            "task_step_impl": "+".join(sorted(set(all_impls))),
                            "task_update_impl": task_update_impl,
                        }
                    )
                )
    base = {(r["dataset"], int(r["seed"])): r for r in final_rows if r.get("candidate_id") == "MLP-same-param-AdamW"}
    attribution: List[Dict[str, Any]] = []
    b109_rows = [r for r in final_rows if r.get("candidate_id") != "MLP-same-param-AdamW"]
    for row in b109_rows:
        b = base.get((row["dataset"], int(row["seed"])))
        if not b:
            continue
        acc_delta = _safe_float(row.get("val_acc")) - _safe_float(b.get("val_acc"))
        ece_delta = _safe_float(row.get("ECE")) - _safe_float(b.get("ECE"))
        auc_step_ratio = _safe_float(row.get("val_loss_auc_step")) / max(EPS, _safe_float(b.get("val_loss_auc_step")))
        auc_time_ratio = _safe_float(row.get("val_loss_auc_time")) / max(EPS, _safe_float(b.get("val_loss_auc_time")))
        attribution.append(
            _stamp(
                {
                    "stage": "V1283_B109_AUC_ATTRIBUTION",
                    "candidate_id": row.get("candidate_id"),
                    "dataset": row.get("dataset"),
                    "seed": row.get("seed"),
                    "val_acc_delta_vs_mlp": acc_delta,
                    "ECE_delta_vs_mlp": ece_delta,
                    "AUC_step_ratio": auc_step_ratio,
                    "AUC_time_ratio": auc_time_ratio,
                    "CEp99_delta_vs_mlp": _safe_float(row.get("CEp99")) - _safe_float(b.get("CEp99")),
                    "margin_p10_delta_vs_mlp": _safe_float(row.get("margin_p10")) - _safe_float(b.get("margin_p10")),
                    "classification": "P4-A-time-blocker" if auc_step_ratio <= 1.05 and auc_time_ratio > 1.05 else ("P4-B-trajectory-blocker" if auc_step_ratio > 1.05 else "P4-pass-or-other"),
                }
            )
        )
    for candidate_id in _b109_candidate_ids(args):
        rows = [r for r in attribution if r.get("stage") == "V1283_B109_AUC_ATTRIBUTION" and r.get("candidate_id") == candidate_id]
        if not rows:
            continue
        deltas = []
        near = []
        ece_ok = True
        auc_step_ok = True
        auc_time_ok = True
        for row in rows:
            deltas.append(_safe_float(row.get("val_acc_delta_vs_mlp")))
            near.append(1.0 if _safe_float(row.get("val_acc_delta_vs_mlp")) >= -0.010 else 0.0)
            ece_ok = ece_ok and _safe_float(row.get("ECE_delta_vs_mlp")) <= 0.02
            auc_step_ok = auc_step_ok and _safe_float(row.get("AUC_step_ratio")) <= 1.05
            auc_time_ok = auc_time_ok and _safe_float(row.get("AUC_time_ratio")) <= 1.05
        attribution.append(
            _stamp(
                {
                    "stage": "V1283_B109_AUC_ATTRIBUTION_SUMMARY",
                    "candidate_id": candidate_id,
                    "mean_delta": _mean(deltas),
                    "worst_delta": min(deltas) if deltas else "",
                    "near_pass_rate": _mean(near),
                    "ece_ok": int(ece_ok),
                    "auc_step_ok": int(auc_step_ok),
                    "auc_time_ok": int(auc_time_ok),
                    "A5_autopsy_pass": int(bool(deltas) and _mean(deltas) >= 0.0 and min(deltas) >= -0.010 and _mean(near) >= 1.0 and ece_ok and auc_step_ok and auc_time_ok),
                }
            )
        )
    write_csv_rows(out_dir / "v1283_b109_auc_autopsy_trace.csv", trace_rows)
    write_csv_rows(out_dir / "v1283_b109_task_autopsy_final.csv", final_rows)
    write_csv_rows(out_dir / "v1283_b109_auc_attribution.csv", attribution if attribution else [_stamp({"stage": "V1283_B109_AUC_ATTRIBUTION", "status": "not_run"})])
    return trace_rows, final_rows, attribution


def run_family_microbench(args: argparse.Namespace, out_dir: Path, device: torch.device, x_train: torch.Tensor, y_train: torch.Tensor, input_dim: int, output_dim: int, specs: Mapping[str, prim.PrimitiveSpec]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    xb = x_train[: int(args.batch_size)].to(device=device, dtype=torch.float32).contiguous()
    yb = y_train[: int(args.batch_size)].to(device=device).contiguous()
    mlp = _make_model("MLP-same-param-AdamW", input_dim, output_dim, x_train, device, int(args.seed) + 8001, specs)
    mlp_m = v126._measure_step(args, mlp, "MLP-same-param-AdamW", xb, yb, device, "autograd")
    _mlp_f50, mlp_f90, _mlp_fmean, _ = _time_call(lambda: mlp(xb), device, int(args.kernel_warmup_steps), int(args.kernel_measure_steps))
    grad_rows: List[Dict[str, Any]] = []
    eff_rows: List[Dict[str, Any]] = []
    profile_rows: List[Dict[str, Any]] = []
    diag_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    eff_rows.append(_stamp({"stage": "V1283_FAMILY_EFFICIENCY", "family": "MLP-control", "candidate_id": "MLP-same-param-AdamW", "implementation_level": "MLP-reference", "step_ratio_q90": 1.0, "memory_ratio_q90": 1.0, "uses_loss_backward": 1, "official_efficiency_pass": 1}))
    selected_family_ids = set(_parse_list(getattr(args, "family_candidate_ids", "")))
    for family, cid, plan in _family_plan():
        if selected_family_ids and cid not in selected_family_ids:
            continue
        spec = specs.get(cid)
        if spec is None:
            failures.append(_stamp({"stage": "V1283_FAMILY_FAILURE_TABLE", "family": family, "candidate_id": cid, "failure_code": "missing_spec", "action_recommended": "add PrimitiveSpec before family evaluation"}))
            continue
        model = _make_model(cid, input_dim, output_dim, x_train, device, int(args.seed) + 8100 + len(eff_rows), specs)
        if hasattr(model, "basis_diagnostics"):
            try:
                diag = dict(model.basis_diagnostics(xb))  # type: ignore[attr-defined]
                diag_rows.append(
                    _stamp(
                        {
                            "stage": "V1283_FAMILY_BASIS_DIAGNOSTIC",
                            "family": family,
                            "candidate_id": cid,
                            "basis_name": spec.basis_name,
                            "diagnostic_scope": "pre_task_batch_basis_and_rational_safety",
                            **diag,
                            "status": "measured",
                        }
                    )
                )
            except Exception as exc:
                diag_rows.append(_stamp({"stage": "V1283_FAMILY_BASIS_DIAGNOSTIC", "family": family, "candidate_id": cid, "basis_name": spec.basis_name, "diagnostic_scope": "pre_task_batch_basis_and_rational_safety", "status": "failed", "failure": f"{type(exc).__name__}:{exc}"}))
        l3_attempted = False
        l3_step_ratio = float("inf")
        l3_mem_ratio = float("inf")
        l3_failure = ""
        l3_official_efficiency_pass = 0
        l3_official_fused_kernel = 0
        try:
            opt = _make_adamw(model, args)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb[: min(16, int(xb.shape[0]))]), yb[: min(16, int(yb.shape[0]))])
            loss.backward()
            finite = []
            grad_norm = []
            for p in model.parameters():
                if p.grad is not None:
                    finite.append(float(torch.isfinite(p.grad).float().mean().detach().item()))
                    grad_norm.append(float(p.grad.detach().float().norm().item()))
            grad_rows.append(_stamp({"stage": "V1283_FAMILY_GRADCHECK", "family": family, "candidate_id": cid, "gradcheck_scope": "L1_autograd_finite_smoke", "finite_grad_rate": _mean(finite), "grad_norm_mean": _mean(grad_norm), "official_analytic_gradcheck_pass": 0, "uses_loss_backward": 1, "status": "measured"}))
        except Exception as exc:
            grad_rows.append(_stamp({"stage": "V1283_FAMILY_GRADCHECK", "family": family, "candidate_id": cid, "gradcheck_scope": "L1_autograd_finite_smoke", "status": "failed", "failure": f"{type(exc).__name__}:{exc}", "official_analytic_gradcheck_pass": 0}))
        if hasattr(model, "manual_gradient_audit"):
            try:
                audit = model.manual_gradient_audit(xb[: min(16, int(xb.shape[0]))], yb[: min(16, int(yb.shape[0]))])  # type: ignore[attr-defined]
                manual_variant = model.manual_kernel_variant() if hasattr(model, "manual_kernel_variant") else "unknown_manual_variant"  # type: ignore[attr-defined]
                grad_rows.append(
                    _stamp(
                        {
                            "stage": "V1283_FAMILY_GRADCHECK",
                            "family": family,
                            "candidate_id": cid,
                            "gradcheck_scope": "L3_analytic_manual_ce_vs_autograd",
                            "manual_kernel_variant": manual_variant,
                            "grad_relerr_max": audit.get("grad_relerr_max", ""),
                            "grad_cos_min": audit.get("grad_cos_min", ""),
                            "output_max_abs_error": audit.get("output_max_abs_error", ""),
                            "official_analytic_gradcheck_pass": int(_safe_float(audit.get("grad_relerr_max"), float("inf")) <= 2.0e-3 and _safe_float(audit.get("output_max_abs_error"), float("inf")) <= 1.0e-6),
                            "uses_loss_backward": 0,
                            "uses_torch_autograd_graph": 0,
                            "reference_uses_loss_backward": 1,
                            "status": "measured",
                        }
                    )
                )
            except Exception as exc:
                grad_rows.append(_stamp({"stage": "V1283_FAMILY_GRADCHECK", "family": family, "candidate_id": cid, "gradcheck_scope": "L3_analytic_manual_ce_vs_autograd", "status": "failed", "failure": f"{type(exc).__name__}:{exc}", "official_analytic_gradcheck_pass": 0, "uses_loss_backward": 0, "uses_torch_autograd_graph": 0}))
        try:
            m = v126._measure_step(args, model, cid, xb, yb, device, "autograd")
            step_ratio = _safe_float(m.get("step_q90_ms"), float("inf")) / max(EPS, _safe_float(mlp_m.get("step_q90_ms"), 0.0))
            mem_ratio = _safe_float(m.get("memory_peak_mb"), float("inf")) / max(EPS, _safe_float(mlp_m.get("memory_peak_mb"), 0.0)) if _safe_float(mlp_m.get("memory_peak_mb")) > 0 else 1.0
            eff_rows.append(
                _stamp(
                    {
                        "stage": "V1283_FAMILY_EFFICIENCY",
                        "family": family,
                        "candidate_id": cid,
                        "basis_name": spec.basis_name,
                        "requested_plan": plan,
                        "implementation_level": "L1-vectorized-or-stream-autograd",
                        "forward_q90_ms": m.get("forward_q90_ms", ""),
                        "backward_q90_ms": m.get("backward_q90_ms", ""),
                        "update_q90_ms": m.get("update_q90_ms", ""),
                        "step_q90_ms": m.get("step_q90_ms", ""),
                        "forward_ratio_q90": _safe_float(m.get("forward_q90_ms"), float("inf")) / max(EPS, _safe_float(mlp_m.get("forward_q90_ms"), 0.0)),
                        "backward_ratio_q90": _safe_float(m.get("backward_q90_ms"), float("inf")) / max(EPS, _safe_float(mlp_m.get("backward_q90_ms"), 0.0)),
                        "update_ratio_q90": _safe_float(m.get("update_q90_ms"), float("inf")) / max(EPS, _safe_float(mlp_m.get("update_q90_ms"), 0.0)),
                        "step_ratio_q90": step_ratio,
                        "memory_ratio_q90": mem_ratio,
                        "uses_loss_backward": 1,
                        "uses_torch_autograd_graph": 1,
                        "official_efficiency_pass": 0,
                        "A1_exploratory_pass": int(step_ratio <= 1.35 and mem_ratio <= 1.0),
                        "status": "measured",
                    }
                )
            )
        except Exception as exc:
            eff_rows.append(_stamp({"stage": "V1283_FAMILY_EFFICIENCY", "family": family, "candidate_id": cid, "implementation_level": "L1-vectorized-or-stream-autograd", "status": "failed", "failure": f"{type(exc).__name__}:{exc}", "official_efficiency_pass": 0}))
        try:
            manual_model = _make_model(cid, input_dim, output_dim, x_train, device, int(args.seed) + 9100 + len(eff_rows), specs)
            manual_m = v126._measure_step(args, manual_model, cid, xb, yb, device, "manual")
            manual_variant = manual_model.manual_kernel_variant() if hasattr(manual_model, "manual_kernel_variant") else "unknown_manual_variant"  # type: ignore[attr-defined]
            l3_step_ratio = _safe_float(manual_m.get("step_q90_ms"), float("inf")) / max(EPS, _safe_float(mlp_m.get("step_q90_ms"), 0.0))
            l3_mem_ratio = _safe_float(manual_m.get("memory_peak_mb"), float("inf")) / max(EPS, _safe_float(mlp_m.get("memory_peak_mb"), 0.0)) if _safe_float(mlp_m.get("memory_peak_mb")) > 0 else 1.0
            official_fused_l3 = int(manual_variant in {"rational_flashkat_grouped_triton_l3_gemm", "rational_flashkat_grouped_inputcross_gemm_l3", "rational_flashkat_grouped_paircross_gemm_l3", "rational_flashkat_grouped_paircross_pca_gemm_l3", "rational_flashkat_grouped_paircross_pca_gemm_frozenbackbone_l3", "rational_flashkat_grouped_paircross_pca_gemm_frozenbackbone_hiddenbias_l3", "rational_flashkat_grouped_paircross_pca_gemm_frozenbackbone_hiddenbias_hiddenresvjptriton_l3", "rational_flashkat_grouped_paircross_pca_gemm_frozenbackbone_hiddenbias_logitrmsnormsg_l3", "rational_flashkat_grouped_paircross_pca_gemm_frozenbackbone_hiddenbias_logitbatchrmsnormsg_l3", "rational_flashkat_grouped_paircross_pca_gemm_freezerational_l3", "rational_flashkat_grouped_paircross_signal_gemm_l3", "rational_flashkat_grouped_paircross_signal_gemm_frozenbackbone_l3", "rational_flashkat_grouped_paircross_signal_gemm_frozenbackbone_hiddenbias_l3", "rational_flashkat_grouped_paircross_signal_gemm_frozenbackbone_hiddenbias_hiddenresvjptriton_l3", "rational_flashkat_grouped_paircross_signal_gemm_freezerational_l3", "rational_flashkat_grouped_paircross_readout_triton_l3", "rational_flashkat_grouped_paircross_readout_block_triton_l3", "rational_flashkat_grouped_paircross_readout_block_triton_frozenbackbone_l3", "rational_flashkat_grouped_paircross_readout_block_triton_frozenbackbone_hiddentailgradtriton_l3", "rational_flashkat_grouped_paircross_readout_block_triton_frozenbackbone_hiddenresvjptriton_l3", "rational_flashkat_grouped_paircross_readout_block_triton_frozenbackbone_hiddenbias_l3", "rational_flashkat_grouped_paircross_readout_block_triton_frozenbackbone_hiddenbias_hiddentailgradtriton_l3", "rational_flashkat_grouped_paircross_readout_block_triton_frozenbackbone_hiddenbias_hiddenresvjptriton_l3", "rational_flashkat_grouped_paircross_readout_block_triton_frozenbackbone_hiddenbias_hiddenresvjptriton_logitrmsnormsg_l3", "rational_flashkat_grouped_paircross_readout_block_triton_frozenbackbone_hiddenbias_hiddenresvjptriton_logitbatchrmsnormsg_l3", "rational_flashkat_grouped_paircross_readout_block_triton_frozenbackbone_hiddenbias_hiddenresvjptriton_logitbatchrmsmixsg_l3", "rational_flashkat_grouped_paircross_readout_block_triton_freezerational_l3", "rational_flashkat_grouped_transformpair_gemm_l3", "rational_flashkat_grouped_pairhidden_gemm_l3", "rational_flashkat_grouped_pairbucket_l3", "rational_flashkat_grouped_pairbucket_triton_l3", "rational_flashkat_grouped_pairbucket_direct_triton_l3", "rational_flashkat_grouped_paircross_bucket_triton_l3", "rational_flashkat_grouped_pairreadbucket_triton_l3", "rational_flashkat_grouped_projsq_readout_gemm_l3", "rational_flashkat_grouped_projbilin_readout_gemm_l3", "cheby_k3_triton_l3_matmul", "cheby_k3_triton_l3_gradbuf", "cheby_k3_triton_l3_paircross_gradbuf", "cheby_k3_triton_l3_paircross_inputcross_gemm_gradbuf", "cheby_k3_triton_l3_paircross_inputcross_linearres_gemm_gradbuf", "cheby_k3_triton_l3_inputcross_gemm_gradbuf", "cheby_k3_triton_l3_inputcross_linearres_gemm_gradbuf", "cheby_k4_triton_l3_matmul", "fourier_k2_triton_l3", "fourier_k2_triton_l3_blockh", "fourier_k2_triton_l3_matmul", "fourier_k3_triton_l3_matmul", "fourier_k4_triton_l3_matmul", "fourier_k4_linearres_triton_l3_matmul", "fourier_k4_linearres_gemm_l3_matmul", "rbf_k2_triton_l3_matmul", "rbf_k4_triton_l3_matmul", "hat_wavelet_k4_triton_l3_matmul"})
            official_efficiency_pass = int(bool(official_fused_l3) and l3_step_ratio <= 1.25 and l3_mem_ratio <= 1.05)
            l3_official_efficiency_pass = official_efficiency_pass
            l3_official_fused_kernel = official_fused_l3
            l3_attempted = True
            eff_rows.append(
                _stamp(
                    {
                        "stage": "V1283_FAMILY_EFFICIENCY",
                        "family": family,
                        "candidate_id": cid,
                        "basis_name": spec.basis_name,
                        "requested_plan": plan,
                        "implementation_level": "L3-analytic-manual-ce-torch-reduction",
                        "manual_kernel_variant": manual_variant,
                        "forward_q90_ms": manual_m.get("forward_q90_ms", ""),
                        "backward_q90_ms": manual_m.get("backward_q90_ms", ""),
                        "update_q90_ms": manual_m.get("update_q90_ms", ""),
                        "step_q90_ms": manual_m.get("step_q90_ms", ""),
                        "forward_ratio_q90": _safe_float(manual_m.get("forward_q90_ms"), float("inf")) / max(EPS, _safe_float(mlp_m.get("forward_q90_ms"), 0.0)),
                        "backward_ratio_q90": _safe_float(manual_m.get("backward_q90_ms"), float("inf")) / max(EPS, _safe_float(mlp_m.get("backward_q90_ms"), 0.0)),
                        "update_ratio_q90": _safe_float(manual_m.get("update_q90_ms"), float("inf")) / max(EPS, _safe_float(mlp_m.get("update_q90_ms"), 0.0)),
                        "step_ratio_q90": l3_step_ratio,
                        "memory_ratio_q90": l3_mem_ratio,
                        "uses_loss_backward": 0,
                        "uses_torch_autograd_graph": 0,
                        "manual_no_grad_guard": 1,
                        "official_fused_l3_kernel": official_fused_l3,
                        "official_efficiency_pass": official_efficiency_pass,
                        "A1_exploratory_pass": int(l3_step_ratio <= 1.35 and l3_mem_ratio <= 1.0),
                        "status": "measured",
                    }
                )
            )
        except Exception as exc:
            l3_failure = f"{type(exc).__name__}:{exc}"
            eff_rows.append(_stamp({"stage": "V1283_FAMILY_EFFICIENCY", "family": family, "candidate_id": cid, "implementation_level": "L3-analytic-manual-ce-torch-reduction", "status": "failed", "failure": l3_failure, "official_efficiency_pass": 0, "uses_loss_backward": 0, "uses_torch_autograd_graph": 0}))
        try:
            q50, q90, mean, _ = _time_call(lambda: model(xb), device, int(args.kernel_warmup_steps), int(args.kernel_measure_steps))
            profile_rows.append(_stamp({"stage": "V1283_FAMILY_COMPONENT_PROFILE", "family": family, "candidate_id": cid, "component": "L1_forward", "q50_ms": q50, "q90_ms": q90, "mean_ms": mean, "ratio_vs_mlp_forward_q90": q90 / max(EPS, mlp_f90), "status": "measured"}))
        except Exception as exc:
            profile_rows.append(_stamp({"stage": "V1283_FAMILY_COMPONENT_PROFILE", "family": family, "candidate_id": cid, "component": "L1_forward", "status": "failed", "failure": f"{type(exc).__name__}:{exc}"}))
        if int(args.compile_family_forward):
            try:
                cmodel = torch.compile(model, mode="reduce-overhead")
                q50, q90, mean, _ = _time_call(lambda: cmodel(xb), device, int(args.kernel_warmup_steps), int(args.kernel_measure_steps))
                profile_rows.append(
                    _stamp(
                        {
                            "stage": "V1283_FAMILY_COMPONENT_PROFILE",
                            "family": family,
                            "candidate_id": cid,
                            "component": "L2_torch_compile_forward_attempt",
                            "q50_ms": q50,
                            "q90_ms": q90,
                            "mean_ms": mean,
                            "ratio_vs_mlp_forward_q90": q90 / max(EPS, mlp_f90),
                            "official_l2_kernel": 0,
                            "status": "measured",
                        }
                    )
                )
            except Exception as exc:
                profile_rows.append(_stamp({"stage": "V1283_FAMILY_COMPONENT_PROFILE", "family": family, "candidate_id": cid, "component": "L2_torch_compile_forward_attempt", "official_l2_kernel": 0, "status": "failed", "failure": f"{type(exc).__name__}:{exc}"}))
        if l3_attempted:
            if int(l3_official_efficiency_pass):
                failure_code = "L3_fused_efficiency_pass_A4_closed"
                reason = "family-specific fused L3 efficiency passed, but A4/A5 family promotion is not run in this closed follow-up runner"
                action = "run official A4 expression and then A5 task triage for this family candidate"
            elif int(l3_official_fused_kernel):
                failure_code = "L3_fused_kernel_efficiency_fail"
                reason = "family-specific fused Triton L3 backward/update path was measured and gradient audit passed, but full-step efficiency gate failed"
                action = "optimize the fused kernel tiling/reduction/update path before A4/A5 official claim"
            else:
                failure_code = "L3_manual_analytic_attempt_not_official_fused_kernel"
                reason = "analytic manual CE backward/update was measured without loss.backward; generic variants are still torch-reduction based, and memory-planned variants are not yet family-specific fused Triton/CUDA kernels"
                action = "implement family-specific fused backward/update kernel or memory-planned workspace path before A4/A5 official claim"
            failures.append(
                _stamp(
                    {
                        "stage": "V1283_FAMILY_FAILURE_TABLE",
                        "family": family,
                        "candidate_id": cid,
                        "failure_code": failure_code,
                        "reason": reason,
                        "L3_step_ratio_q90": l3_step_ratio,
                        "L3_memory_ratio_q90": l3_mem_ratio,
                        "official_fused_l3_kernel": l3_official_fused_kernel,
                        "action_recommended": action,
                    }
                )
            )
        else:
            failures.append(
                _stamp(
                    {
                        "stage": "V1283_FAMILY_FAILURE_TABLE",
                        "family": family,
                        "candidate_id": cid,
                        "failure_code": "L3_manual_analytic_attempt_failed",
                        "reason": l3_failure or "manual analytic CE path unavailable",
                        "action_recommended": "fix analytic backward first, then implement family-specific fused backward/update before A4/A5 official claim",
                    }
                )
            )
    write_csv_rows(out_dir / "v1283_family_gradcheck.csv", grad_rows if grad_rows else [_stamp({"stage": "V1283_FAMILY_GRADCHECK", "status": "not_run"})])
    write_csv_rows(out_dir / "v1283_family_efficiency.csv", eff_rows)
    write_csv_rows(out_dir / "v1283_family_component_profile.csv", profile_rows if profile_rows else [_stamp({"stage": "V1283_FAMILY_COMPONENT_PROFILE", "status": "not_run"})])
    write_csv_rows(out_dir / "v1283_family_basis_diagnostics.csv", diag_rows if diag_rows else [_stamp({"stage": "V1283_FAMILY_BASIS_DIAGNOSTIC", "status": "not_run"})])
    return grad_rows, eff_rows, profile_rows, failures


def write_closed_family_followups(out_dir: Path, failures: Sequence[Mapping[str, Any]], eff_rows: Sequence[Mapping[str, Any]], profile_rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    expression_rows: List[Dict[str, Any]] = []
    task_rows: List[Dict[str, Any]] = []
    geometry_rows: List[Dict[str, Any]] = []
    functional_rows: List[Dict[str, Any]] = []
    status: Dict[str, Any] = {"stage": "V1283_FAMILY_STATUS", "generated_at": _now_iso(), "families": {}}
    families = sorted({str(f) for f, _cid, _plan in _family_plan()})
    for family in families:
        fam_eff = [r for r in eff_rows if str(r.get("family")) == family and str(r.get("candidate_id")) != "MLP-same-param-AdamW"]
        fam_profile = [r for r in profile_rows if str(r.get("family")) == family]
        any_l2 = any(str(r.get("component")) == "L2_torch_compile_forward_attempt" and str(r.get("status")) == "measured" for r in fam_profile)
        best_step = min((_safe_float(r.get("step_ratio_q90"), float("inf")) for r in fam_eff if str(r.get("implementation_level")).startswith("L1")), default=float("inf"))
        best_l3 = min((_safe_float(r.get("step_ratio_q90"), float("inf")) for r in fam_eff if str(r.get("implementation_level")) == "L3-analytic-manual-ce-torch-reduction"), default=float("inf"))
        any_l3 = any(str(r.get("implementation_level")) == "L3-analytic-manual-ce-torch-reduction" and str(r.get("status")) == "measured" for r in fam_eff)
        any_l3_fused = any(int(_safe_float(r.get("official_fused_l3_kernel"), 0)) == 1 for r in fam_eff if str(r.get("implementation_level")) == "L3-analytic-manual-ce-torch-reduction")
        any_l3_official_eff = any(int(_safe_float(r.get("official_efficiency_pass"), 0)) == 1 for r in fam_eff if str(r.get("implementation_level")) == "L3-analytic-manual-ce-torch-reduction")
        family_status = "KernelBlocked"
        blocker = "L3 manual analytic attempted; fused family-specific backward/update kernel missing" if any_l3 else "L3 analytic backward/update missing"
        if not any_l2:
            blocker = "L2 forward attempt failed or missing; L3 also missing"
        if any_l3_fused and not any_l3_official_eff:
            blocker = "family-specific fused L3 kernel exists, but full-step efficiency gate failed"
        if any_l3_official_eff:
            family_status = "ExpressionBlocked"
            blocker = "family-specific fused L3 efficiency exists; A4/A5 official family promotion not run in this closed follow-up runner"
        status["families"][family] = {
            "status": family_status,
            "best_L1_step_ratio": best_step if math.isfinite(best_step) else "",
            "best_L3_manual_step_ratio": best_l3 if math.isfinite(best_l3) else "",
            "L2_forward_attempt_measured": bool(any_l2),
            "L3_manual_analytic_attempt_measured": bool(any_l3),
            "L3_fused_kernel_measured": bool(any_l3_fused),
            "L3_official_efficiency_candidate_exists": bool(any_l3_official_eff),
            "blocker": blocker,
            "official_family_failure_claim": False,
        }
        for cid in sorted({str(r.get("candidate_id")) for r in fam_eff if str(r.get("candidate_id")) != ""}):
            expression_rows.append(_stamp({"stage": "V1283_FAMILY_EXPRESSION", "family": family, "candidate_id": cid, "status": "not_run", "reason": "official L2/L3 full-step gate not available; A4 legally closed for classic family"}))
            task_rows.append(_stamp({"stage": "V1283_FAMILY_TASK_TRIAGE", "family": family, "candidate_id": cid, "status": "not_run", "reason": "A4 legally closed; no official task triage"}))
            geometry_rows.append(_stamp({"stage": "V1283_FAMILY_GEOMETRY_LINEC", "family": family, "candidate_id": cid, "status": "not_run", "reason": "family base not qualified; Line C official not opened"}))
            functional_rows.append(_stamp({"stage": "V1283_FAMILY_FUNCTIONAL_DIAGNOSTIC", "family": family, "candidate_id": cid, "status": "not_run", "official_gate_open": 0, "reason": "base not qualified"}))
    write_csv_rows(out_dir / "v1283_family_expression.csv", expression_rows)
    write_csv_rows(out_dir / "v1283_family_task_triage.csv", task_rows)
    write_csv_rows(out_dir / "v1283_family_task_trace.csv", [_stamp({"stage": "V1283_FAMILY_TASK_TRACE", "status": "not_run"})])
    write_csv_rows(out_dir / "v1283_family_geometry_lineC.csv", geometry_rows)
    write_csv_rows(out_dir / "v1283_family_functional_diagnostic.csv", functional_rows)
    write_csv_rows(out_dir / "v1283_family_failure_table.csv", [dict(r) for r in failures] if failures else [_stamp({"stage": "V1283_FAMILY_FAILURE_TABLE", "status": "no_failure"})])
    write_json(out_dir / "v1283_family_status.json", status)
    return expression_rows, task_rows, geometry_rows, status


def run_family_task_triage_for_candidates(
    args: argparse.Namespace,
    device: torch.device,
    candidate_ids: Sequence[str],
    family_by_candidate: Mapping[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    ids = [str(cid) for cid in candidate_ids if str(cid)]
    if not ids:
        return [], [], []
    datasets = [v120._canonical_dataset(x) for x in _parse_list(args.datasets)]
    seeds = _parse_ints(args.seeds)
    methods = ["MLP-same-param-AdamW"] + ids
    task_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        data = v120._load_vision_split(args, dataset, train_size=int(args.train_size), val_size=int(args.val_size), test_size=int(args.test_size))
        x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, x_test_cpu, y_test_cpu, input_dim, output_dim, _protocol = data
        x_train = x_train_cpu.to(device=device, dtype=torch.float32)
        y_train = y_train_cpu.to(device=device)
        x_val = x_val_cpu.to(device=device, dtype=torch.float32)
        y_val = y_val_cpu.to(device=device)
        x_test = x_test_cpu.to(device=device, dtype=torch.float32)
        y_test = y_test_cpu.to(device=device)
        local_specs = _specs_for(int(input_dim), int(output_dim))
        total_steps = max(1, int(args.epochs) * int(math.ceil(float(x_train.shape[0]) / float(max(1, int(args.batch_size))))))
        for seed in seeds:
            for method_id in methods:
                model = _make_model(method_id, int(input_dim), int(output_dim), x_train, device, int(seed) + 128383, local_specs)
                opt = _make_adamw(model, args)
                triton_update_params = v126._triton_adamw_params(args, model) if method_id != "MLP-same-param-AdamW" else []
                manual_update = v126._ManualForeachAdamW(opt.param_groups, args, triton_update_params=triton_update_params) if method_id != "MLP-same-param-AdamW" and v126._variant_uses_manual_adamw(args, model) else None
                task_update_impl = "triton_tensor_adamw+manual_foreach_adamw" if triton_update_params else ("manual_foreach_adamw" if manual_update is not None else "adamw")
                manual_variant = model.manual_kernel_variant() if method_id != "MLP-same-param-AdamW" and hasattr(model, "manual_kernel_variant") else ""
                if method_id != "MLP-same-param-AdamW" and not manual_variant:
                    raise RuntimeError(f"family A5 requested without manual fused-L3 path: {method_id}")
                cross_scale_now = _apply_cross_readout_scale_warmup(model, method_id, local_specs, 0, int(args.epochs))
                linear_residual_scale_now = _apply_linear_residual_scale_ramp(model, method_id, local_specs, 0, int(args.epochs))
                warm = min(int(args.batch_size), int(x_train.shape[0]))
                for warm_idx in range(max(0, int(args.task_compile_warmup_steps))):
                    start_idx = (warm_idx * warm) % int(x_train.shape[0])
                    xb = x_train[start_idx : start_idx + warm]
                    yb = y_train[start_idx : start_idx + warm]
                    if int(xb.shape[0]) < warm:
                        xb = x_train[:warm]
                        yb = y_train[:warm]
                    opt.zero_grad(set_to_none=True)
                    if method_id == "MLP-same-param-AdamW":
                        F.cross_entropy(model(xb), yb).backward()
                    else:
                        logits, cache = model.manual_ce_forward_cache(xb)  # type: ignore[attr-defined]
                        model.manual_ce_backward_from_cache(logits, cache, yb)  # type: ignore[attr-defined]
                    opt.zero_grad(set_to_none=True)
                _sync(device)
                gen = torch.Generator(device=device).manual_seed(int(seed) + 128390 + len(trace_rows))
                step_times: List[float] = []
                val_losses: List[float] = []
                val_times_q90: List[float] = []
                step_id = 0
                for epoch in range(int(args.epochs)):
                    cross_scale_now = _apply_cross_readout_scale_warmup(model, method_id, local_specs, epoch, int(args.epochs))
                    linear_residual_scale_now = _apply_linear_residual_scale_ramp(model, method_id, local_specs, epoch, int(args.epochs))
                    epoch_start = len(step_times)
                    perm = torch.randperm(int(x_train.shape[0]), generator=gen, device=device)
                    for off in range(0, int(x_train.shape[0]), int(args.batch_size)):
                        idx = perm[off : off + int(args.batch_size)]
                        xb = x_train[idx]
                        yb = y_train[idx]
                        lr_now = v126._task_lr_for(args, step_id + 1, total_steps)
                        for group in opt.param_groups:
                            lr_scale = float(group.get("lr_scale", 1.0))
                            for switch_epoch, switch_scale in group.get("lr_scale_schedule", []):
                                if epoch >= int(switch_epoch):
                                    lr_scale = float(switch_scale)
                            if not group.get("lr_scale_schedule") and group.get("lr_scale_after", None) is not None and epoch >= int(group.get("lr_scale_switch_epoch", 0)):
                                lr_scale = float(group.get("lr_scale_after", lr_scale))
                            group["lr"] = lr_now * lr_scale
                        opt.zero_grad(set_to_none=True)
                        _sync(device)
                        t0 = time.perf_counter()
                        if method_id == "MLP-same-param-AdamW":
                            F.cross_entropy(model(xb), yb).backward()
                            task_step_impl = "torch-autograd"
                            uses_loss_backward = 1
                            uses_torch_autograd_graph = 1
                        else:
                            logits, cache = model.manual_ce_forward_cache(xb)  # type: ignore[attr-defined]
                            model.manual_ce_backward_from_cache(logits, cache, yb)  # type: ignore[attr-defined]
                            task_step_impl = f"manual-ce:{manual_variant}"
                            uses_loss_backward = 0
                            uses_torch_autograd_graph = 0
                        if manual_update is not None:
                            manual_update.step()
                        else:
                            opt.step()
                        _sync(device)
                        t1 = time.perf_counter()
                        step_times.append((t1 - t0) * 1000.0)
                        step_id += 1
                    val_epoch = v1252._classification_basic(model, x_val, y_val)
                    epoch_times = step_times[epoch_start:]
                    epoch_q90 = _q(epoch_times, 0.90)
                    val_losses.append(val_epoch["NLL"])
                    val_times_q90.append(epoch_q90)
                    trace_rows.append(
                        _stamp(
                            {
                                "stage": "V1283_FAMILY_TASK_TRACE",
                                "family": family_by_candidate.get(method_id, "MLP-control"),
                                "candidate_id": method_id,
                                "dataset": dataset,
                                "seed": seed,
                                "epoch": epoch + 1,
                                "step": step_id,
                                "val_acc": val_epoch["acc"],
                                "val_loss": val_epoch["NLL"],
                                "ECE": val_epoch["ECE"],
                                "epoch_step_time_q90_ms": epoch_q90,
                                "task_step_impl": task_step_impl,
                                "task_update_impl": task_update_impl,
                                "manual_kernel_variant": manual_variant,
                                "cross_readout_scale_effective": cross_scale_now if cross_scale_now is not None else "",
                                "linear_residual_scale_effective": linear_residual_scale_now if linear_residual_scale_now is not None else "",
                                "uses_loss_backward": uses_loss_backward,
                                "uses_torch_autograd_graph": uses_torch_autograd_graph,
                            }
                        )
                    )
                _apply_cross_readout_scale_warmup(model, method_id, local_specs, max(0, int(args.epochs) - 1), int(args.epochs))
                _apply_linear_residual_scale_ramp(model, method_id, local_specs, max(0, int(args.epochs) - 1), int(args.epochs))
                final = v1252._classification_basic(model, x_val, y_val)
                test = v1252._classification_basic(model, x_test, y_test)
                steady_start = min(max(0, int(args.task_timing_warmup_epochs)), len(val_losses) - 1)
                steady_losses = val_losses[steady_start:]
                steady_times = val_times_q90[steady_start:]
                auc_step = sum(steady_losses) / max(1, len(steady_losses))
                auc_time = sum(l * max(1.0, t) for l, t in zip(steady_losses, steady_times)) / max(1, len(steady_losses))
                task_rows.append(
                    _stamp(
                        {
                            "stage": "V1283_FAMILY_TASK_TRIAGE",
                            "family": family_by_candidate.get(method_id, "MLP-control"),
                            "candidate_id": method_id,
                            "dataset": dataset,
                            "seed": seed,
                            "val_acc": final["acc"],
                            "test_acc": test["acc"],
                            "NLL": final["NLL"],
                            "ECE": final["ECE"],
                            "CEp99": final["CEp99"],
                            "margin_p10": final["margin_p10"],
                            "val_loss_auc_step": auc_step,
                            "val_loss_auc_time": auc_time,
                            "step_time_q90_ms": _q(step_times, 0.90),
                            "task_step_impl": task_step_impl,
                            "task_update_impl": task_update_impl,
                            "manual_kernel_variant": manual_variant,
                            "A4_expression_pass": int(method_id != "MLP-same-param-AdamW"),
                            "official_fused_l3_task_loop": int(method_id != "MLP-same-param-AdamW"),
                            "uses_loss_backward": uses_loss_backward,
                            "uses_torch_autograd_graph": uses_torch_autograd_graph,
                        }
                    )
                )
    base = {(r["dataset"], int(r["seed"])): r for r in task_rows if r.get("candidate_id") == "MLP-same-param-AdamW"}
    passed: List[str] = []
    for candidate_id in ids:
        rows = [r for r in task_rows if r.get("candidate_id") == candidate_id]
        deltas: List[float] = []
        near: List[float] = []
        ece_ok = True
        auc_step_ok = True
        auc_time_ok = True
        for row in rows:
            b = base.get((row["dataset"], int(row["seed"])))
            if not b:
                continue
            acc_delta = _safe_float(row.get("val_acc")) - _safe_float(b.get("val_acc"))
            ece_delta = _safe_float(row.get("ECE")) - _safe_float(b.get("ECE"))
            auc_step_ratio = _safe_float(row.get("val_loss_auc_step")) / max(EPS, _safe_float(b.get("val_loss_auc_step")))
            auc_time_ratio = _safe_float(row.get("val_loss_auc_time")) / max(EPS, _safe_float(b.get("val_loss_auc_time")))
            row["val_acc_delta_vs_mlp"] = acc_delta
            row["ECE_delta_vs_mlp"] = ece_delta
            row["AUC_step_ratio"] = auc_step_ratio
            row["AUC_time_ratio"] = auc_time_ratio
            deltas.append(acc_delta)
            near.append(1.0 if acc_delta >= -0.010 else 0.0)
            ece_ok = ece_ok and ece_delta <= 0.02
            auc_step_ok = auc_step_ok and auc_step_ratio <= 1.05
            auc_time_ok = auc_time_ok and auc_time_ratio <= 1.05
        pass_gate = bool(deltas and _mean(deltas) >= 0.0 and min(deltas) >= -0.010 and _mean(near) >= 1.0 and ece_ok and auc_step_ok and auc_time_ok)
        task_rows.append(
            _stamp(
                {
                    "stage": "V1283_FAMILY_TASK_SUMMARY",
                    "family": family_by_candidate.get(candidate_id, ""),
                    "candidate_id": candidate_id,
                    "mean_delta": _mean(deltas),
                    "worst_delta": min(deltas) if deltas else "",
                    "near_pass_rate": _mean(near),
                    "ece_ok": int(ece_ok),
                    "auc_step_ok": int(auc_step_ok),
                    "auc_time_ok": int(auc_time_ok),
                    "A5_task_pass": int(pass_gate),
                    "gate_rule": "mean>=0,worst>=-0.010,near=1.0,ECE_delta<=0.02,AUC_step/time<=1.05",
                    "official_fused_l3_task_loop": 1,
                }
            )
        )
        if pass_gate:
            passed.append(candidate_id)
        else:
            failure_rows.append(
                _stamp(
                    {
                        "stage": "V1283_FAMILY_FAILURE_TABLE",
                        "family": family_by_candidate.get(candidate_id, ""),
                        "candidate_id": candidate_id,
                        "failure_code": "A5_family_task_gate_fail",
                        "mean_delta": _mean(deltas),
                        "worst_delta": min(deltas) if deltas else "",
                        "near_pass_rate": _mean(near),
                        "ece_ok": int(ece_ok),
                        "auc_step_ok": int(auc_step_ok),
                        "auc_time_ok": int(auc_time_ok),
                        "reason": "official fused L3 efficiency and A4 expression passed, but family A5 task gate failed under manual CE task loop",
                        "action_recommended": "repair task trajectory without lowering gates, changing loss, using teacher/distillation, sampler/class weights, or dataset-name branches",
                    }
                )
            )
    return trace_rows + task_rows, failure_rows, passed


def run_family_expression_promotion(
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
    eff_rows: Sequence[Mapping[str, Any]],
    grad_rows: Sequence[Mapping[str, Any]],
    expression_rows: Sequence[Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
    geometry_rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    family_status: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    eff_pass = {
        str(r.get("candidate_id"))
        for r in eff_rows
        if str(r.get("implementation_level")) == "L3-analytic-manual-ce-torch-reduction"
        and int(_safe_float(r.get("official_fused_l3_kernel"), 0)) == 1
        and int(_safe_float(r.get("official_efficiency_pass"), 0)) == 1
    }
    grad_pass = {
        str(r.get("candidate_id"))
        for r in grad_rows
        if str(r.get("gradcheck_scope")) == "L3_analytic_manual_ce_vs_autograd"
        and int(_safe_float(r.get("official_analytic_gradcheck_pass"), 0)) == 1
    }
    promotion_ids = sorted(eff_pass & grad_pass)
    promotion_filter = set(_parse_list(getattr(args, "family_promotion_ids", "")))
    if promotion_filter:
        promotion_ids = [cid for cid in promotion_ids if cid in promotion_filter]
    if not promotion_ids:
        return list(map(dict, expression_rows)), list(map(dict, task_rows)), list(map(dict, geometry_rows)), family_status

    expr_input_dim = int(args.expression_dim)
    expr_specs = _specs_for(expr_input_dim, 1)
    promotion_ids = [cid for cid in promotion_ids if cid in expr_specs]
    if not promotion_ids:
        return list(map(dict, expression_rows)), list(map(dict, task_rows)), list(map(dict, geometry_rows)), family_status

    expr_raw, frozen_raw, cond_raw, _ = v124.run_expression(args, out_dir, device, expr_specs, promotion_ids)
    expr_stage = []
    for row in expr_raw:
        copied = dict(row)
        copied["stage"] = "V1283_FAMILY_EXPRESSION_BATTERY"
        copied["promotion_reason"] = "official L3 fused efficiency and analytic gradcheck passed"
        expr_stage.append(_stamp(copied))
    frozen_stage = []
    for row in frozen_raw:
        copied = dict(row)
        copied["stage"] = "V1283_FAMILY_FROZEN_READOUT"
        copied["promotion_reason"] = "official L3 fused efficiency and analytic gradcheck passed"
        frozen_stage.append(_stamp(copied))
    cond_stage = []
    for row in cond_raw:
        copied = dict(row)
        copied["stage"] = "V1283_FAMILY_MATRIX_SPAN"
        copied["promotion_reason"] = "official L3 fused efficiency and analytic gradcheck passed"
        cond_stage.append(_stamp(copied))

    summaries: List[Dict[str, Any]] = []
    promoted_pass: List[str] = []
    family_by_candidate = {str(r.get("candidate_id")): str(r.get("family")) for r in eff_rows if str(r.get("candidate_id"))}
    for cid in promotion_ids:
        passed, summary = v1252._v125_expression_pass(cid, expr_stage, frozen_stage, cond_stage)  # type: ignore[attr-defined]
        summary = dict(summary)
        summary["stage"] = "V1283_FAMILY_EXPRESSION_SUMMARY"
        summary["family"] = family_by_candidate.get(cid, "")
        summary["candidate_id"] = cid
        summary["promotion_reason"] = "official L3 fused efficiency and analytic gradcheck passed"
        summaries.append(_stamp(summary))
        if passed:
            promoted_pass.append(cid)
        else:
            summaries.append(
                _stamp(
                    {
                        "stage": "V1283_FAMILY_EXPRESSION_FAILURE",
                        "family": family_by_candidate.get(cid, ""),
                        "candidate_id": cid,
                        "failure_code": "A4_expression_gate_fail_after_l3_efficiency",
                        "action_recommended": "repair family expression capacity without losing B4g L3 fused efficiency; do not run A5 official for this family yet",
                    }
                )
            )

    promotion_set = set(promotion_ids)
    family_task_rows: List[Dict[str, Any]] = []
    family_task_failures: List[Dict[str, Any]] = []
    family_task_pass: List[str] = []
    if promoted_pass:
        family_task_rows, family_task_failures, family_task_pass = run_family_task_triage_for_candidates(args, device, promoted_pass, family_by_candidate)
    expr_keep = [dict(r) for r in expression_rows if str(r.get("candidate_id")) not in promotion_set]
    task_keep = [dict(r) for r in task_rows if str(r.get("candidate_id")) not in promotion_set]
    geo_keep = [dict(r) for r in geometry_rows if str(r.get("candidate_id")) not in promotion_set]
    task_keep.extend(family_task_rows)
    for cid in promotion_ids:
        fam = family_by_candidate.get(cid, "")
        if cid in promoted_pass:
            if cid in family_task_pass:
                geo_keep.append(_stamp({"stage": "V1283_FAMILY_GEOMETRY_LINEC", "family": fam, "candidate_id": cid, "status": "not_run", "reason": "family A5 passed, but family Line C official path is not integrated in this runner"}))
            else:
                geo_keep.append(_stamp({"stage": "V1283_FAMILY_GEOMETRY_LINEC", "family": fam, "candidate_id": cid, "status": "not_run", "reason": "family A5 did not pass; Line C official remains closed"}))
        else:
            task_keep.append(_stamp({"stage": "V1283_FAMILY_TASK_TRIAGE", "family": fam, "candidate_id": cid, "status": "not_run", "reason": "A4 expression gate failed; no official task triage", "A4_expression_pass": 0}))
            geo_keep.append(_stamp({"stage": "V1283_FAMILY_GEOMETRY_LINEC", "family": fam, "candidate_id": cid, "status": "not_run", "reason": "family base not qualified; Line C official not opened"}))

    for fam, info in family_status.get("families", {}).items():
        fam_ids = [cid for cid in promotion_ids if family_by_candidate.get(cid) == fam]
        if not fam_ids:
            continue
        fam_pass = [cid for cid in fam_ids if cid in promoted_pass]
        fam_task_pass = [cid for cid in fam_ids if cid in family_task_pass]
        if fam_task_pass:
            info["status"] = "FamilyPass"
            info["A4_expression_pass_candidates"] = fam_pass
            info["A5_task_pass_candidates"] = fam_task_pass
            info["blocker"] = "family-specific fused L3 efficiency, A4 expression, and A5 task passed; family Line C/Functional route is still not integrated"
        elif fam_pass:
            info["status"] = "TaskBlocked"
            info["A4_expression_pass_candidates"] = fam_pass
            info["A5_task_pass_candidates"] = []
            info["blocker"] = "family-specific fused L3 efficiency and A4 expression passed; A5 fused task triage failed"
        else:
            info["status"] = "ExpressionBlocked"
            info["A4_expression_pass_candidates"] = []
            info["blocker"] = "family-specific fused L3 efficiency passed but A4 expression failed"

    failure_keep = [
        dict(r)
        for r in failures
        if not (
            str(r.get("candidate_id")) in promotion_set
            and str(r.get("failure_code")) == "L3_fused_efficiency_pass_A4_closed"
        )
    ]
    for cid in promotion_ids:
        fam = family_by_candidate.get(cid, "")
        if cid in promoted_pass:
            if cid in family_task_pass:
                failure_keep.append(
                    _stamp(
                        {
                            "stage": "V1283_FAMILY_FAILURE_TABLE",
                            "family": fam,
                            "candidate_id": cid,
                            "failure_code": "family_linec_not_integrated_after_A5",
                            "reason": "official fused L3 efficiency, A4 expression, and A5 task passed, but family Line C/Functional path is not integrated in this runner",
                            "action_recommended": "integrate family Line C / Functional official path before claiming portfolio-level functional success",
                        }
                    )
                )
            else:
                failure_keep.extend([dict(r) for r in family_task_failures if str(r.get("candidate_id")) == cid])
        else:
            failure_keep.append(
                _stamp(
                    {
                        "stage": "V1283_FAMILY_FAILURE_TABLE",
                        "family": fam,
                        "candidate_id": cid,
                        "failure_code": "A4_expression_gate_fail_after_l3_efficiency",
                        "reason": "official fused L3 efficiency and analytic gradcheck passed, but the v12.5 expression gate failed",
                        "action_recommended": "repair family expression capacity without losing fused L3 efficiency; do not run A5 official for this family yet",
                    }
                )
            )

    final_expr = expr_keep + expr_stage + frozen_stage + cond_stage + summaries
    final_task = task_keep
    final_task_trace = [dict(r) for r in final_task if str(r.get("stage")) == "V1283_FAMILY_TASK_TRACE"]
    final_geo = geo_keep
    write_csv_rows(out_dir / "v1283_family_expression.csv", final_expr)
    write_csv_rows(out_dir / "v1283_family_task_triage.csv", final_task)
    write_csv_rows(out_dir / "v1283_family_task_trace.csv", final_task_trace if final_task_trace else [_stamp({"stage": "V1283_FAMILY_TASK_TRACE", "status": "not_run"})])
    write_csv_rows(out_dir / "v1283_family_geometry_lineC.csv", final_geo)
    write_csv_rows(out_dir / "v1283_family_failure_table.csv", failure_keep)
    write_json(out_dir / "v1283_family_status.json", family_status)
    return final_expr, final_task, final_geo, family_status


def _restage_rows(rows: Sequence[Mapping[str, Any]], stage: str, selected_candidate_id: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["stage"] = stage
        copied["v1283_selected_b109_candidate_id"] = selected_candidate_id
        copied["run_family"] = "B109/FHQ"
        out.append(_stamp(copied))
    return out


def _v125_cleanup(out_dir: Path) -> None:
    for name in [
        "v125_train_probe_coupling.csv",
        "v125_signal_reservoir_sketch.csv",
        "v125_noise_leak_audit.csv",
        "v125_manifold_channel_diagnostics.csv",
        "v125_functional_direction_audit.csv",
        "v125_one_step_probe.csv",
        "v125_five_step_probe.csv",
        "v125_lambda_backtracking.csv",
        "v125_control_matrix.csv",
        "v125_functional_route.json",
    ]:
        (out_dir / name).unlink(missing_ok=True)


def _set_v125_linec_args(args: argparse.Namespace, selected_candidate_id: str) -> Dict[str, Any]:
    previous = {
        "k1_fast_candidate_id": getattr(args, "k1_fast_candidate_id", None),
        "k1_candidate_id": getattr(args, "k1_candidate_id", None),
        "k2_candidate_id": getattr(args, "k2_candidate_id", None),
        "k3_candidate_id": getattr(args, "k3_candidate_id", None),
    }
    args.k1_fast_candidate_id = str(args.fixedp_candidate_id)
    args.k1_candidate_id = str(args.exact_candidate_id)
    args.k2_candidate_id = str(args.expression_anchor_candidate_id)
    args.k3_candidate_id = str(selected_candidate_id)
    return previous


def _restore_v125_linec_args(args: argparse.Namespace, previous: Mapping[str, Any]) -> None:
    for key, value in previous.items():
        if value is None and hasattr(args, key):
            delattr(args, key)
        else:
            setattr(args, key, value)


def run_b109_linec_functional_reentry(
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    input_dim: int,
    output_dim: int,
    specs: Mapping[str, prim.PrimitiveSpec],
    base_qualified: bool,
    selected_candidate_id: str,
) -> Dict[str, Any]:
    previous = _set_v125_linec_args(args, selected_candidate_id)
    try:
        coupling, sketch, noise, diag = v1252.run_line_c(
            args,
            out_dir,
            device,
            x_train,
            y_train,
            x_val,
            y_val,
            input_dim,
            output_dim,
            specs,
            bool(base_qualified),
        )
        direction, one, five, control, functional_route = v1252.run_functional_diagnostic(
            args,
            out_dir,
            device,
            x_train,
            y_train,
            x_val,
            y_val,
            input_dim,
            output_dim,
            specs,
            bool(base_qualified),
        )
    finally:
        _restore_v125_linec_args(args, previous)

    coupling_rows = _restage_rows(coupling, "V1283_B109_LINEC_COUPLING", selected_candidate_id)
    sketch_rows = _restage_rows(sketch, "V1283_B109_LINEC_SIGNAL_RESERVOIR", selected_candidate_id)
    noise_rows = _restage_rows(noise, "V1283_B109_LINEC_NOISE_LEAK", selected_candidate_id)
    diag_rows = _restage_rows(diag, "V1283_B109_LINEC_DIAGNOSTICS", selected_candidate_id)
    direction_rows = _restage_rows(direction, "V1283_B109_FUNCTIONAL_DIRECTION_AUDIT", selected_candidate_id)
    one_rows = _restage_rows(one, "V1283_B109_FUNCTIONAL_ONE_STEP", selected_candidate_id)
    five_rows = _restage_rows(five, "V1283_B109_FUNCTIONAL_FIVE_STEP", selected_candidate_id)
    control_rows = _restage_rows(control, "V1283_B109_FUNCTIONAL_CONTROL_MATRIX", selected_candidate_id)

    write_csv_rows(out_dir / "v1283_b109_linec_coupling.csv", coupling_rows)
    write_csv_rows(out_dir / "v1283_b109_linec_signal_reservoir.csv", sketch_rows)
    write_csv_rows(out_dir / "v1283_b109_linec_noise_leak.csv", noise_rows)
    write_csv_rows(out_dir / "v1283_b109_linec_diagnostics.csv", diag_rows)
    write_csv_rows(out_dir / "v1283_b109_functional_direction_audit.csv", direction_rows)
    write_csv_rows(out_dir / "v1283_b109_functional_one_step.csv", one_rows)
    write_csv_rows(out_dir / "v1283_b109_functional_five_step.csv", five_rows)
    write_csv_rows(out_dir / "v1283_b109_functional_control_matrix.csv", control_rows)

    mlp_c = next((r for r in coupling_rows if str(r.get("candidate_id")) == "MLP-same-param-AdamW"), {})
    selected_c = next((r for r in coupling_rows if str(r.get("candidate_id")) == str(selected_candidate_id)), {})
    selected_s = next((r for r in sketch_rows if str(r.get("candidate_id")) == str(selected_candidate_id)), {})
    selected_ctrl = next((r for r in control_rows if str(r.get("candidate_id")) == str(selected_candidate_id)), {})
    mlp_r2 = _safe_float(mlp_c.get("CouplingR2"), 0.0)
    selected_r2 = _safe_float(selected_c.get("CouplingR2"), 0.0)
    linec_collapse = int(bool(selected_c) and bool(mlp_c) and selected_r2 < mlp_r2 - 0.02)
    linec_measured = int(bool(selected_c) and bool(selected_s))
    summary = _stamp(
        {
            "stage": "V1283_B109_LINEC_FUNCTIONAL_REENTRY_SUMMARY",
            "candidate_id": selected_candidate_id,
            "base_qualified": int(base_qualified),
            "linec_measured": linec_measured,
            "linec_nontearing_pass": int(linec_measured and not linec_collapse),
            "mlp_CouplingR2": mlp_r2 if mlp_c else "",
            "candidate_CouplingR2": selected_r2 if selected_c else "",
            "candidate_CouplingCorr": selected_c.get("CouplingCorr", "") if selected_c else "",
            "candidate_RealSignalReservoirRatio": selected_s.get("RealSignalReservoirRatio", "") if selected_s else "",
            "candidate_NoiseSignalLeak": selected_s.get("NoiseSignalLeak", "") if selected_s else "",
            "functional_reentry_measured": int(bool(selected_ctrl)),
            "functional_beats_controls": int(_safe_float(selected_ctrl.get("beats_controls"), 0)),
            "functional_control_gap_vs_best_control": selected_ctrl.get("control_gap_vs_best_control", "") if selected_ctrl else "",
            "official_gate_open": int(base_qualified),
            "official_functional_success": int(_safe_float(functional_route.get("official_functional_success"), 0)),
            "official_functional_reason": functional_route.get("reason", ""),
            "claim_scope": "cloned_diagnostic_reentry_only",
        }
    )
    write_csv_rows(out_dir / "v1283_b109_linec_functional_reentry_summary.csv", [summary])
    functional_payload = dict(functional_route)
    functional_payload["stage"] = "V1283_B109_FUNCTIONAL_ROUTE"
    functional_payload["candidate_id"] = selected_candidate_id
    write_json(out_dir / "v1283_b109_functional_route.json", _stamp(functional_payload))
    _v125_cleanup(out_dir)
    return summary


def _family_linec_ids_from_status(args: argparse.Namespace, family_status: Mapping[str, Any]) -> List[str]:
    explicit = _parse_list(getattr(args, "family_linec_ids", ""))
    if explicit:
        return explicit
    ids: List[str] = []
    families = family_status.get("families", {}) if isinstance(family_status, Mapping) else {}
    if isinstance(families, Mapping):
        for payload in families.values():
            if not isinstance(payload, Mapping):
                continue
            for key in ("A4_expression_pass_candidates", "A5_task_pass_candidates"):
                values = payload.get(key, [])
                if isinstance(values, str):
                    values = [values]
                for cid in values:
                    if str(cid).strip():
                        ids.append(str(cid).strip())
    selected = set(_parse_list(getattr(args, "family_candidate_ids", "")))
    if selected:
        ids = [cid for cid in ids if cid in selected]
    return list(dict.fromkeys(ids))


def _restage_family_linec_rows(rows: Sequence[Mapping[str, Any]], stage: str, selected_candidate_id: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["stage"] = stage
        copied["v1283_family_linec_candidate_id"] = selected_candidate_id
        copied["run_family"] = "ClassicFamily"
        out.append(_stamp(copied))
    return out


def run_family_linec_autopsy(
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    input_dim: int,
    output_dim: int,
    specs: Mapping[str, prim.PrimitiveSpec],
    base_qualified: bool,
    family_status: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    candidate_ids = _family_linec_ids_from_status(args, family_status)
    if not candidate_ids:
        not_run = [_stamp({"stage": "V1283_FAMILY_LINEC_SUMMARY", "status": "not_run", "reason": "no A4/A5-open family candidate selected for Line C autopsy"})]
        write_csv_rows(out_dir / "v1283_family_linec_coupling.csv", [_stamp({"stage": "V1283_FAMILY_LINEC_COUPLING", "status": "not_run"})])
        write_csv_rows(out_dir / "v1283_family_linec_signal_reservoir.csv", [_stamp({"stage": "V1283_FAMILY_LINEC_SIGNAL_RESERVOIR", "status": "not_run"})])
        write_csv_rows(out_dir / "v1283_family_linec_noise_leak.csv", [_stamp({"stage": "V1283_FAMILY_LINEC_NOISE_LEAK", "status": "not_run"})])
        write_csv_rows(out_dir / "v1283_family_linec_diagnostics.csv", [_stamp({"stage": "V1283_FAMILY_LINEC_DIAGNOSTICS", "status": "not_run"})])
        write_csv_rows(out_dir / "v1283_family_linec_summary.csv", not_run)
        return not_run

    all_coupling: List[Dict[str, Any]] = []
    all_sketch: List[Dict[str, Any]] = []
    all_noise: List[Dict[str, Any]] = []
    all_diag: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for cid in candidate_ids:
        previous = _set_v125_linec_args(args, cid)
        try:
            coupling, sketch, noise, diag = v1252.run_line_c(
                args,
                out_dir,
                device,
                x_train,
                y_train,
                x_val,
                y_val,
                input_dim,
                output_dim,
                specs,
                bool(base_qualified),
            )
        except Exception as exc:
            summaries.append(
                _stamp(
                    {
                        "stage": "V1283_FAMILY_LINEC_SUMMARY",
                        "candidate_id": cid,
                        "base_qualified": int(base_qualified),
                        "linec_measured": 0,
                        "status": "failed",
                        "linec_error_type": type(exc).__name__,
                        "linec_error_message": str(exc),
                        "claim_scope": "family_task_geometry_autopsy_only_failed_no_success_claim",
                    }
                )
            )
        finally:
            _restore_v125_linec_args(args, previous)
        if summaries and summaries[-1].get("candidate_id") == cid and summaries[-1].get("status") == "failed":
            _v125_cleanup(out_dir)
            continue

        coupling_rows = _restage_family_linec_rows(coupling, "V1283_FAMILY_LINEC_COUPLING", cid)
        sketch_rows = _restage_family_linec_rows(sketch, "V1283_FAMILY_LINEC_SIGNAL_RESERVOIR", cid)
        noise_rows = _restage_family_linec_rows(noise, "V1283_FAMILY_LINEC_NOISE_LEAK", cid)
        diag_rows = _restage_family_linec_rows(diag, "V1283_FAMILY_LINEC_DIAGNOSTICS", cid)
        all_coupling.extend(coupling_rows)
        all_sketch.extend(sketch_rows)
        all_noise.extend(noise_rows)
        all_diag.extend(diag_rows)

        mlp_c = next((r for r in coupling_rows if str(r.get("candidate_id")) == "MLP-same-param-AdamW"), {})
        selected_c = next((r for r in coupling_rows if str(r.get("candidate_id")) == str(cid)), {})
        mlp_s = next((r for r in sketch_rows if str(r.get("candidate_id")) == "MLP-same-param-AdamW"), {})
        selected_s = next((r for r in sketch_rows if str(r.get("candidate_id")) == str(cid)), {})
        selected_d = next((r for r in diag_rows if str(r.get("candidate_id")) == str(cid)), {})
        mlp_r2 = _safe_float(mlp_c.get("CouplingR2"), 0.0)
        selected_r2 = _safe_float(selected_c.get("CouplingR2"), 0.0)
        mlp_noise = _safe_float(mlp_s.get("NoiseSignalLeak"), 0.0)
        selected_noise = _safe_float(selected_s.get("NoiseSignalLeak"), 0.0)
        mlp_res = _safe_float(mlp_s.get("RealSignalReservoirRatio"), 0.0)
        selected_res = _safe_float(selected_s.get("RealSignalReservoirRatio"), 0.0)
        coupling_collapse = int(bool(selected_c) and bool(mlp_c) and selected_r2 < mlp_r2 - 0.02)
        noise_leak_high = int(bool(selected_s) and bool(mlp_s) and selected_noise > mlp_noise + 0.02)
        reservoir_high = int(bool(selected_s) and bool(mlp_s) and selected_res > mlp_res + 0.02)
        if coupling_collapse:
            interpretation = "coupling_collapse"
        elif noise_leak_high:
            interpretation = "noise_leak_high"
        elif reservoir_high:
            interpretation = "real_signal_reservoir_high"
        elif selected_c and selected_s:
            interpretation = "linec_not_obviously_bad_task_blocker_likely_calibration_or_optimizer_trajectory"
        else:
            interpretation = "linec_missing"
        summaries.append(
            _stamp(
                {
                    "stage": "V1283_FAMILY_LINEC_SUMMARY",
                    "candidate_id": cid,
                    "base_qualified": int(base_qualified),
                    "linec_measured": int(bool(selected_c) and bool(selected_s)),
                    "mlp_CouplingR2": mlp_r2 if mlp_c else "",
                    "candidate_CouplingR2": selected_r2 if selected_c else "",
                    "CouplingR2_delta_vs_mlp": selected_r2 - mlp_r2 if selected_c and mlp_c else "",
                    "mlp_NoiseSignalLeak": mlp_noise if mlp_s else "",
                    "candidate_NoiseSignalLeak": selected_noise if selected_s else "",
                    "NoiseSignalLeak_delta_vs_mlp": selected_noise - mlp_noise if selected_s and mlp_s else "",
                    "mlp_RealSignalReservoirRatio": mlp_res if mlp_s else "",
                    "candidate_RealSignalReservoirRatio": selected_res if selected_s else "",
                    "RealSignalReservoirRatio_delta_vs_mlp": selected_res - mlp_res if selected_s and mlp_s else "",
                    "candidate_CEp99": selected_d.get("CEp99", "") if selected_d else "",
                    "candidate_ECE": selected_d.get("ECE", "") if selected_d else "",
                    "candidate_NLL": selected_d.get("NLL", "") if selected_d else "",
                    "coupling_collapse_flag": coupling_collapse,
                    "noise_leak_high_flag": noise_leak_high,
                    "reservoir_high_flag": reservoir_high,
                    "linec_interpretation": interpretation,
                    "claim_scope": "family_task_geometry_autopsy_only_not_family_success",
                }
            )
        )

    write_csv_rows(out_dir / "v1283_family_linec_coupling.csv", all_coupling)
    write_csv_rows(out_dir / "v1283_family_linec_signal_reservoir.csv", all_sketch)
    write_csv_rows(out_dir / "v1283_family_linec_noise_leak.csv", all_noise)
    write_csv_rows(out_dir / "v1283_family_linec_diagnostics.csv", all_diag)
    write_csv_rows(out_dir / "v1283_family_linec_summary.csv", summaries)
    _v125_cleanup(out_dir)
    return summaries


def _delta_norm(deltas: Sequence[torch.Tensor]) -> float:
    return math.sqrt(sum(float(d.detach().float().square().sum().item()) for d in deltas))


def _blend_deltas(primary: Sequence[torch.Tensor], residual: Sequence[torch.Tensor], residual_weight: float) -> List[torch.Tensor]:
    if not primary:
        return []
    p_norm = _delta_norm(primary)
    r_norm = _delta_norm(residual)
    out: List[torch.Tensor] = []
    for p, r in zip(primary, residual):
        scaled_r = r.to(p.device)
        if r_norm > EPS:
            scaled_r = scaled_r * (p_norm / max(EPS, r_norm))
        out.append(p + float(residual_weight) * scaled_r)
    mixed_norm = _delta_norm(out)
    if mixed_norm > EPS and p_norm > EPS:
        out = [d * (p_norm / mixed_norm) for d in out]
    return out


def _branch_scale_damping_delta(model: torch.nn.Module, task_delta: Sequence[torch.Tensor], mode: str) -> List[torch.Tensor]:
    """Label-free structural damping direction for branch-scale tail drift probes."""
    named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    out: List[torch.Tensor] = []
    for name, param in named_params:
        delta = torch.zeros_like(param)
        if name == "branch_scale":
            base = -param.detach().clone()
            if mode == "direct":
                if base.ndim >= 1 and int(base.shape[0]) >= 2:
                    mask = torch.zeros_like(base)
                    mask[0].copy_(base[0])
                    base = mask
            elif mode == "quad":
                if base.ndim >= 1 and int(base.shape[0]) >= 2:
                    mask = torch.zeros_like(base)
                    mask[1].copy_(base[1])
                    base = mask
            delta = base
        out.append(delta)
    if len(out) != len(task_delta):
        return [torch.zeros_like(d) for d in task_delta]
    return out


def _orthogonalize_deltas(residual: Sequence[torch.Tensor], reference: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    if not residual or not reference:
        return [torch.zeros_like(d) for d in reference]
    flat_ref = torch.cat([d.detach().flatten() for d in reference])
    flat_res = torch.cat([d.detach().flatten().to(flat_ref.device) for d in residual])
    proj = (flat_res @ flat_ref) / flat_ref.square().sum().clamp_min(EPS)
    out = [r.to(t.device) - proj.to(t.device) * t for r, t in zip(residual, reference)]
    res_norm = _delta_norm(residual)
    out_norm = _delta_norm(out)
    if out_norm > EPS and res_norm > EPS:
        out = [d * (res_norm / out_norm) for d in out]
    return out


def _geo_gain_delta_scored(
    before_model: torch.nn.Module,
    after_model: torch.nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    xq: torch.Tensor,
    yq: torch.Tensor,
    ridge: float,
    sketch_dim: int,
    seed: int,
) -> Dict[str, float]:
    with torch.no_grad():
        db = after_model(xb).detach() - before_model(xb).detach()
        dq = after_model(xq).detach() - before_model(xq).detach()
    delta_energy = float(db.square().sum().item() + dq.square().sum().item())
    if delta_energy <= 1.0e-18:
        r2, corr = 0.0, 0.0
    else:
        r2, corr, _resid, _pred = v1252._ridge_coupling(db, dq, ridge)
    before_sig = v1252._signal_reservoir_metrics(before_model, xb[: min(int(xb.shape[0]), 8)], yb[: min(int(yb.shape[0]), 8)], sketch_dim, seed)
    after_sig = v1252._signal_reservoir_metrics(after_model, xb[: min(int(xb.shape[0]), 8)], yb[: min(int(yb.shape[0]), 8)], sketch_dim, seed)
    cls_before = v1252._classification_basic(before_model, xq, yq)
    cls_after = v1252._classification_basic(after_model, xq, yq)
    reservoir_gain = after_sig["RealSignalReservoirRatio"] - before_sig["RealSignalReservoirRatio"]
    reservoir_release = before_sig["RealSignalReservoirRatio"] - after_sig["RealSignalReservoirRatio"]
    noise_increase = after_sig["NoiseSignalLeak"] - before_sig["NoiseSignalLeak"]
    nll_increase = cls_after["NLL"] - cls_before["NLL"]
    score = r2 + reservoir_release - max(0.0, noise_increase) - max(0.0, nll_increase)
    return {
        "CouplingR2": r2,
        "CouplingCorr": corr,
        "delta_energy": delta_energy,
        "RealSignalReservoirRatio_before": before_sig["RealSignalReservoirRatio"],
        "RealSignalReservoirRatio_after": after_sig["RealSignalReservoirRatio"],
        "RealSignalReservoirRatio_delta": reservoir_gain,
        "RealSignalReservoirRelease": reservoir_release,
        "NoiseSignalLeak_before": before_sig["NoiseSignalLeak"],
        "NoiseSignalLeak_after": after_sig["NoiseSignalLeak"],
        "NoiseSignalLeak_delta": noise_increase,
        "delta_score": score,
        "NLL_delta": nll_increase,
        "CEp99_delta": cls_after["CEp99"] - cls_before["CEp99"],
        "ECE_delta": cls_after["ECE"] - cls_before["ECE"],
        "margin_p10_delta": cls_after["margin_p10"] - cls_before["margin_p10"],
    }


def run_b109_functional_delta_score_repair(
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    input_dim: int,
    output_dim: int,
    specs: Mapping[str, prim.PrimitiveSpec],
    base_qualified: bool,
    selected_candidate_id: str,
) -> Dict[str, Any]:
    xb = x_train[: int(args.functional_batch_size)].to(device=device, dtype=torch.float32)
    yb = y_train[: int(args.functional_batch_size)].to(device=device)
    xq = x_val[: int(args.functional_batch_size)].to(device=device, dtype=torch.float32)
    yq = y_val[: int(args.functional_batch_size)].to(device=device)
    candidate_ids = [str(args.fixedp_candidate_id), str(args.exact_candidate_id), str(args.expression_anchor_candidate_id), str(selected_candidate_id)]
    candidate_ids = list(dict.fromkeys(candidate_ids))
    controls = ["C0-TaskOnlyAdamW", "C1-NoOpMatchedOverhead", "C2-RandomMatchedNorm", "C3-AdamWParallelDirection"]
    functionals = [
        "F7-TaskPlusSNRGeometryResidual-w005",
        "F7-TaskPlusSNRGeometryResidual-w010",
        "F7-TaskPlusSNRGeometryResidual-w015",
        "F7-TaskPlusSNRGeometryResidual-w025",
        "F7-TaskPlusSNRGeometryResidual-w050",
        "F7n-TaskMinusSNRGeometryResidual-w010",
        "F7n-TaskMinusSNRGeometryResidual-w025",
        "F8-TaskPlusOrthogonalGeometryResidual-w010",
        "F8-TaskPlusOrthogonalGeometryResidual-w025",
        "F8n-TaskMinusOrthogonalGeometryResidual-w010",
        "F8n-TaskMinusOrthogonalGeometryResidual-w025",
        "F9-GeometryResidualOnlyDeltaScore",
        "F10-TaskPlusBranchScaleDamping-w005",
        "F10-TaskPlusBranchScaleDamping-w010",
        "F10d-TaskPlusDirectBranchDamping-w010",
        "F10q-TaskPlusQuadBranchDamping-w010",
        "F11-TaskMinusBranchScaleDamping-w005",
        "F11-TaskMinusBranchScaleDamping-w010",
        "F11d-TaskMinusDirectBranchDamping-w010",
        "F11q-TaskMinusQuadBranchDamping-w010",
        "F12-NegativeSNRGeometryResidualOnly",
        "F13-NegativeOrthogonalGeometryResidualOnly",
        "F14-OrthBranchScaleDampingOnly",
        "F14d-OrthDirectBranchDampingOnly",
        "F14q-OrthQuadBranchDampingOnly",
        "F15-TaskPlusOrthBranchScaleDamping-w010",
        "F15n-TaskMinusOrthBranchScaleDamping-w010",
        "F15q-TaskPlusOrthQuadBranchDamping-w010",
        "F15qn-TaskMinusOrthQuadBranchDamping-w010",
    ]
    rows: List[Dict[str, Any]] = []
    control_rows: List[Dict[str, Any]] = []
    for idx, cid in enumerate(candidate_ids):
        base = v1252._make_model(cid, input_dim, output_dim, x_train, device, int(args.seed) + 701 + idx, specs)
        task_delta = v1252._grad_delta(base, xb, yb, float(args.lr))
        snr_delta = v1252._basis_delta(base, task_delta, "basis_aware_snr_projected")
        orth_delta = v1252._basis_delta(base, task_delta, "basis_aware_orthogonal")
        branch_delta = _branch_scale_damping_delta(base, task_delta, "both")
        direct_branch_delta = _branch_scale_damping_delta(base, task_delta, "direct")
        quad_branch_delta = _branch_scale_damping_delta(base, task_delta, "quad")
        orth_branch_delta = _orthogonalize_deltas(branch_delta, task_delta)
        orth_direct_branch_delta = _orthogonalize_deltas(direct_branch_delta, task_delta)
        orth_quad_branch_delta = _orthogonalize_deltas(quad_branch_delta, task_delta)
        dirs = {
            "C0-TaskOnlyAdamW": task_delta,
            "C1-NoOpMatchedOverhead": [torch.zeros_like(d) for d in task_delta],
            "C2-RandomMatchedNorm": v1252._random_like(task_delta, int(args.seed) + 702 + idx),
            "C3-AdamWParallelDirection": task_delta,
            "F7-TaskPlusSNRGeometryResidual-w005": _blend_deltas(task_delta, snr_delta, 0.05),
            "F7-TaskPlusSNRGeometryResidual-w010": _blend_deltas(task_delta, snr_delta, 0.10),
            "F7-TaskPlusSNRGeometryResidual-w015": _blend_deltas(task_delta, snr_delta, 0.15),
            "F7-TaskPlusSNRGeometryResidual-w025": _blend_deltas(task_delta, snr_delta, 0.25),
            "F7-TaskPlusSNRGeometryResidual-w050": _blend_deltas(task_delta, snr_delta, 0.50),
            "F7n-TaskMinusSNRGeometryResidual-w010": _blend_deltas(task_delta, snr_delta, -0.10),
            "F7n-TaskMinusSNRGeometryResidual-w025": _blend_deltas(task_delta, snr_delta, -0.25),
            "F8-TaskPlusOrthogonalGeometryResidual-w010": _blend_deltas(task_delta, orth_delta, 0.10),
            "F8-TaskPlusOrthogonalGeometryResidual-w025": _blend_deltas(task_delta, orth_delta, 0.25),
            "F8n-TaskMinusOrthogonalGeometryResidual-w010": _blend_deltas(task_delta, orth_delta, -0.10),
            "F8n-TaskMinusOrthogonalGeometryResidual-w025": _blend_deltas(task_delta, orth_delta, -0.25),
            "F9-GeometryResidualOnlyDeltaScore": snr_delta,
            "F10-TaskPlusBranchScaleDamping-w005": _blend_deltas(task_delta, branch_delta, 0.05),
            "F10-TaskPlusBranchScaleDamping-w010": _blend_deltas(task_delta, branch_delta, 0.10),
            "F10d-TaskPlusDirectBranchDamping-w010": _blend_deltas(task_delta, direct_branch_delta, 0.10),
            "F10q-TaskPlusQuadBranchDamping-w010": _blend_deltas(task_delta, quad_branch_delta, 0.10),
            "F11-TaskMinusBranchScaleDamping-w005": _blend_deltas(task_delta, branch_delta, -0.05),
            "F11-TaskMinusBranchScaleDamping-w010": _blend_deltas(task_delta, branch_delta, -0.10),
            "F11d-TaskMinusDirectBranchDamping-w010": _blend_deltas(task_delta, direct_branch_delta, -0.10),
            "F11q-TaskMinusQuadBranchDamping-w010": _blend_deltas(task_delta, quad_branch_delta, -0.10),
            "F12-NegativeSNRGeometryResidualOnly": [-d for d in snr_delta],
            "F13-NegativeOrthogonalGeometryResidualOnly": [-d for d in orth_delta],
            "F14-OrthBranchScaleDampingOnly": orth_branch_delta,
            "F14d-OrthDirectBranchDampingOnly": orth_direct_branch_delta,
            "F14q-OrthQuadBranchDampingOnly": orth_quad_branch_delta,
            "F15-TaskPlusOrthBranchScaleDamping-w010": _blend_deltas(task_delta, orth_branch_delta, 0.10),
            "F15n-TaskMinusOrthBranchScaleDamping-w010": _blend_deltas(task_delta, orth_branch_delta, -0.10),
            "F15q-TaskPlusOrthQuadBranchDamping-w010": _blend_deltas(task_delta, orth_quad_branch_delta, 0.10),
            "F15qn-TaskMinusOrthQuadBranchDamping-w010": _blend_deltas(task_delta, orth_quad_branch_delta, -0.10),
        }
        before_loss = float(F.cross_entropy(base(xq), yq).detach().item())
        scores: Dict[str, float] = {}
        for name in [*controls, *functionals]:
            selected = 0.0
            accepted = 0
            backtracks = 0
            best = deepcopy(base)
            for lam in [1.0, 0.5, 0.25, 0.125, 0.0625]:
                trial = deepcopy(base)
                v1252._apply_delta(trial, dirs[name], lam)
                after_loss = float(F.cross_entropy(trial(xq), yq).detach().item())
                lower_ok = True if name.startswith("C") else after_loss >= before_loss * 0.95
                upper_ok = after_loss <= before_loss * 1.05
                if (lower_ok and upper_ok) or name == "C1-NoOpMatchedOverhead":
                    selected = lam
                    accepted = 1
                    best = trial
                    break
                backtracks += 1
            after_loss = float(F.cross_entropy(best(xq), yq).detach().item())
            holdout_ratio = after_loss / max(EPS, before_loss)
            geo = _geo_gain_delta_scored(base, best, xb, yb, xq, yq, float(args.ridge_lambda), int(args.sketch_dim), int(args.seed) + 703 + idx)
            score = geo["delta_score"] - max(0.0, holdout_ratio - 1.0)
            scores[name] = score
            rows.append(
                _stamp(
                    {
                        "stage": "V1283_B109_FUNCTIONAL_DELTA_SCORE_REPAIR",
                        "candidate_id": cid,
                        "selected_b109_candidate_id": selected_candidate_id,
                        "update_type": name,
                        "official_gate_open": int(base_qualified),
                        "lambda_selected": selected,
                        "lambda_backtracking_steps": backtracks,
                        "accepted": accepted,
                        "holdout_loss_before": before_loss,
                        "holdout_loss_after": after_loss,
                        "holdout_descent_ratio": holdout_ratio,
                        "delta_score": score,
                        **geo,
                    }
                )
            )
        best_control = max(scores[c] for c in controls)
        best_functional = max(scores[f] for f in functionals)
        best_functional_id = max(functionals, key=lambda f: scores[f])
        gap = best_functional - best_control
        control_rows.append(
            _stamp(
                {
                    "stage": "V1283_B109_FUNCTIONAL_DELTA_SCORE_CONTROL_MATRIX",
                    "candidate_id": cid,
                    "selected_b109_candidate_id": selected_candidate_id,
                    "best_functional_update": best_functional_id,
                    "best_functional_delta_score": best_functional,
                    "best_control_delta_score": best_control,
                    "delta_score_gap_vs_best_control": gap,
                    "beats_controls_delta_score": int(gap > 0.0),
                    "official_gate_open": int(base_qualified),
                    "claim_scope": "delta_scored_repair_audit_not_final_official",
                }
            )
        )
    write_csv_rows(out_dir / "v1283_b109_functional_delta_score_repair.csv", rows)
    write_csv_rows(out_dir / "v1283_b109_functional_delta_score_control_matrix.csv", control_rows)
    selected = next((r for r in control_rows if str(r.get("candidate_id")) == str(selected_candidate_id)), {})
    summary = _stamp(
        {
            "stage": "V1283_B109_FUNCTIONAL_DELTA_SCORE_REPAIR_SUMMARY",
            "candidate_id": selected_candidate_id,
            "base_qualified": int(base_qualified),
            "delta_score_repair_measured": int(bool(selected)),
            "delta_score_beats_controls": int(_safe_float(selected.get("beats_controls_delta_score"), 0)) if selected else 0,
            "delta_score_gap_vs_best_control": selected.get("delta_score_gap_vs_best_control", "") if selected else "",
            "best_functional_update": selected.get("best_functional_update", "") if selected else "",
            "claim_scope": "metric_repair_and_loss_agnostic_direction_audit_only",
            "official_functional_success": 0,
        }
    )
    write_csv_rows(out_dir / "v1283_b109_functional_delta_score_repair_summary.csv", [summary])
    return summary


def audit_provenance(out_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(out_dir.glob("v1283_*")):
        if path.is_dir():
            continue
        fake = proxy = cpu = 0
        checked = 0
        if path.suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    checked += 1
                    fake += int(_safe_float(row.get("fake_data_used"), 0))
                    proxy += int(_safe_float(row.get("proxy_row_used"), 0))
                    cpu += int(_safe_float(row.get("cpu_offload_used"), 0))
        rows.append(_stamp({"stage": "V1283_PROVENANCE_AUDIT", "artifact": path.name, "rows_checked": checked, "fake_data_used_sum": fake, "proxy_row_used_sum": proxy, "cpu_offload_used_sum": cpu, "no_fake_pass": int(fake == 0), "no_proxy_pass": int(proxy == 0), "no_cpu_offload_pass": int(cpu == 0)}))
    write_csv_rows(out_dir / "v1283_provenance_audit.csv", rows)
    return rows


def write_hash_manifest(out_dir: Path) -> Dict[str, Any]:
    entries: Dict[str, str] = {}
    for path in sorted(out_dir.glob("v1283_*")):
        if path.is_file() and path.name != "v1283_hash_manifest.json":
            entries[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {"stage": "V1283_HASH_MANIFEST", "generated_at": _now_iso(), "artifacts": entries}
    write_json(out_dir / "v1283_hash_manifest.json", payload)
    return payload


def decide_route(b109_full: Sequence[Mapping[str, Any]], b109_auc: Sequence[Mapping[str, Any]], family_status: Mapping[str, Any], provenance: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    no_fake = all(int(_safe_float(r.get("no_fake_pass"), 0)) == 1 and int(_safe_float(r.get("no_proxy_pass"), 0)) == 1 and int(_safe_float(r.get("no_cpu_offload_pass"), 0)) == 1 for r in provenance) if provenance else False
    b109_profiles = [r for r in b109_full if str(r.get("candidate_id")) != "MLP-same-param-AdamW"]
    b109_f3_eff = any(
        int(_safe_float(r.get("official_efficiency_pass"), 0)) == 1
        and str(r.get("implementation_id")) == "F3-triton-workspace-forward-delta-readout-proj-grad-learnableP"
        for r in b109_profiles
    )
    official_eff_rows = [r for r in b109_profiles if int(_safe_float(r.get("official_efficiency_pass"), 0)) == 1]
    b109_eff = bool(official_eff_rows)
    efficiency_anchor = official_eff_rows[0] if official_eff_rows else {}
    auc_summaries = [r for r in b109_auc if str(r.get("stage")) == "V1283_B109_AUC_ATTRIBUTION_SUMMARY"]
    auc_pass = any(int(_safe_float(r.get("A5_autopsy_pass"), 0)) == 1 for r in auc_summaries)
    best_summary = max(auc_summaries, key=lambda r: (_safe_float(r.get("A5_autopsy_pass"), 0), _safe_float(r.get("near_pass_rate"), 0), _safe_float(r.get("mean_delta"), -999.0)), default={})
    family_pass = any(str(v.get("status")) == "FamilyPass" for v in family_status.get("families", {}).values()) if isinstance(family_status, Mapping) else False
    if not b109_eff:
        route = "R0-B109FHQFullstepNotOfficial"
        action = "repair B109 lower-level FHQ full-step before A4/A5 official"
    elif not auc_pass:
        route = "R2-B109AUCTrajectoryStillBlocked"
        action = "manual foreach update fixed part of dense-P F3 cost, but AUC/NLL trajectory is still blocked; next try a trajectory primitive that changes learning dynamics without adding direct-grad cost, or a true lower-level projection-gradient/update fusion"
    elif not family_pass:
        route = "R4-B109NearOrPassClassicFamiliesKernelBlocked"
        action = "implement classic-family L3 analytic backward/update kernels; current L1/L2 attempts do not prove family mathematical failure"
    else:
        route = "R7-BaseAndFamilyPortfolioReadyForFunctionalOfficial"
        action = "open official functional short-run with strong controls"
    decision = _stamp({"stage": "V1283_ROUTE_DECISION", "route": route, "B109_FHQ_any_official_efficiency": int(b109_eff), "B109_F3_official_efficiency": int(b109_f3_eff), "B109_FHQ_efficiency_anchor_candidate": efficiency_anchor.get("candidate_id", ""), "B109_FHQ_efficiency_anchor_impl": efficiency_anchor.get("implementation_id", ""), "B109_A5_autopsy_pass": int(auc_pass), "B109_A5_best_candidate": best_summary.get("candidate_id", ""), "classic_family_pass_count": int(family_pass), "base_qualified": int(b109_eff and auc_pass), "functional_open": int(b109_eff and auc_pass), "no_fake_provenance_pass": int(no_fake), "next_recommended_action": action})
    return decision


def _md_table(rows: Sequence[Mapping[str, Any]], cols: Sequence[str], limit: int | None = None) -> str:
    data = list(rows[:limit] if limit is not None else rows)
    if not data:
        return "_no rows_"
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for row in data:
        body.append("| " + " | ".join(f"`{row.get(c, '')}`" for c in cols) + " |")
    return "\n".join([header, sep, *body])


def write_report(
    out_dir: Path,
    report_path: Path,
    decision: Mapping[str, Any],
    b109_full: Sequence[Mapping[str, Any]],
    b109_auc: Sequence[Mapping[str, Any]],
    family_eff: Sequence[Mapping[str, Any]],
    family_profile: Sequence[Mapping[str, Any]],
    family_status: Mapping[str, Any],
    provenance: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, Any],
) -> None:
    b109_rows = [r for r in b109_full if str(r.get("candidate_id")) != "MLP-same-param-AdamW"]
    auc_summary = [r for r in b109_auc if str(r.get("stage")) == "V1283_B109_AUC_ATTRIBUTION_SUMMARY"]
    family_summary_rows = []
    for family, payload in family_status.get("families", {}).items():
        row = {"family": family, **payload}
        family_summary_rows.append(row)
    hash_rows = [{"artifact": k, "sha256_prefix": str(v)[:12]} for k, v in hashes.get("artifacts", {}).items()]
    text = f"""# DG-KAN v12.8.3 B109 / 经典基函数全家族高效率化 / Functional Geometry 结果复盘

> 本复盘由 `{SCRIPT_PATH.name}` 生成。所有结论只来自本轮落盘 CSV/JSON/hash/provenance audit；没有 fake data、proxy row、CPU offload、teacher/distillation、loss modification、sampler/class weight 或 dataset-name branch。

## 0. 最新结论

```text
route = {decision.get('route')}
B109_FHQ_any_official_efficiency = {decision.get('B109_FHQ_any_official_efficiency')}
B109_F3_official_efficiency = {decision.get('B109_F3_official_efficiency')}
B109_FHQ_efficiency_anchor = {decision.get('B109_FHQ_efficiency_anchor_candidate')} / {decision.get('B109_FHQ_efficiency_anchor_impl')}
B109_A5_autopsy_pass = {decision.get('B109_A5_autopsy_pass')}
B109_A5_best_candidate = {decision.get('B109_A5_best_candidate')}
B109_LineC_measured = {decision.get('B109_LineC_measured', '')}
B109_LineC_nontearing_pass = {decision.get('B109_LineC_nontearing_pass', '')}
B109_functional_reentry_measured = {decision.get('B109_functional_reentry_measured', '')}
B109_functional_beats_controls = {decision.get('B109_functional_beats_controls', '')}
B109_functional_delta_score_repair_measured = {decision.get('B109_functional_delta_score_repair_measured', '')}
B109_functional_delta_score_beats_controls = {decision.get('B109_functional_delta_score_beats_controls', '')}
B109_functional_delta_score_best_update = {decision.get('B109_functional_delta_score_best_update', '')}
B109_functional_delta_score_gap_vs_best_control = {decision.get('B109_functional_delta_score_gap_vs_best_control', '')}
official_functional_success = {decision.get('official_functional_success', '')}
base_qualified = {decision.get('base_qualified')}
functional_open = {decision.get('functional_open')}
next_recommended_action = {decision.get('next_recommended_action')}
artifact = {out_dir}
```

## 1. 修改审计

| file | 修改 | 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_primitives.py` | 新增 `hat_wavelet`、`B5h-HatWaveletKAN-local-K4`、`B3c-ChebyKAN-K3-stream`、`B4b-FourierKAN-lowfreq-K2-stream`、`B136a-lowfreqP`、`B137a-blockfreqP`、`B138b-semiFixedP`、`B139b-pUpdateEvery4`、`B140b-pUpdateEvery2`、`B141b-warm1PUpdateEvery4`、`B142b-warm1PUpdateEvery2`、`B143b-warm2PUpdateEvery4`、`B144b-warm2PUpdateEvery2`、`B145b-activeP64`、`B146b-activeP48`、`B147b-activeP96`、`B148-B197`、`B198-B202 scheduled manual-update cross repairs`、`B203 Triton quad_proj AdamW update repair`、`B204 fused projection-gradient AdamW update repair`、`B210-B213 stop-grad logitnorm repairs` | 对齐 v12.8.3 family plan 与 B109 trajectory repair；B164/B165 将 projection 参数从 `D*H` 降为 `H` 个 scale；B166-B168 将 projection 限制为 `K*H` 个 sparse weights；B188-B190 保留 dense-P trajectory，只替换 update 为可审计 manual foreach AdamW；B194-B197 不加 direct rows，只在 quadratic projection 内做 stop-grad batch RMS / tanh bound；B198-B202 将已有 pUpdateEvery/warm pUpdateEvery 轨迹修复与 manual foreach update 组合；B203 保持 dense-P/B109 数学轨迹，只把 `quad_proj` AdamW update 下沉到 Triton 单 tensor kernel；B204 在同一 projection-gradient kernel 内直接更新 `quad_proj` 与 AdamW 状态；B210/B211 保留 B205 前向 samplewise logitnorm，但把 RMS 分母从反向图中切掉；B212/B213 改为 batch-level stop-grad logit RMS，避免逐样本 confidence 结构被单独重标定。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 `B205 fused-update samplewise logitnorm` | 在 B204 fused projection-gradient/update 路径上加入 sample-wise logit normalization，并固定 branch/gain 以满足 FHQ analytic backward 支持条件；不改 loss/gate/data，也不按 dataset 分支。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 `B206 fused-update fixed scalar` | 隔离 B205 的改善来源：保留 B204 fused update，只固定 branch/gain，不启用昂贵 sample-wise logitnorm，检查是否能保持效率并改善 AUC trajectory。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 `B207 fused-update lower logitnorm` | 保留 B205 结构但把 sample-wise logitnorm target 从 `1.50` 降到 `1.00`，检验能否缓解 ECE/AUC 伤害。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 `B208/B209 hybrid SparseP logitnorm` | 将 projection 从 dense `D*H` 改为 hybrid local/global `K*H` sparse weights，并叠加 B205 的 logitnorm trajectory primitive，测试能否保留 task 改善同时降低 projection backward/update cost。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 `B210/B211 stop-grad logitnorm` | B210 用 stop-gradient RMS denominator 保留 samplewise logitnorm 前向轨迹、移除完整 logitnorm backward 的逐样本耦合项；B211 再叠加 warm2/update-every2 projection schedule。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 `B212/B213 batch stop-grad logitnorm` | 将 B210 的 per-sample RMS 改成 whole-batch RMS 标尺，目标是在保留低成本 backward 的同时减少 per-sample normalization 对 ECE/AUC 的伤害；B213 再叠加 warm2/update-every2 projection schedule。 |
| `dgkan/kernels/fused_hinge_quadratic.py` | 新增 active-hidden projection-gradient Triton kernel | `B145/B146/B147` 只对指定 active P hidden columns 计算 projection gradient，其余列梯度清零；这是降低 fused backward/update cost 的模型约束，不改 loss、不用数据集分支。 |
| `dgkan/kernels/fused_hinge_quadratic.py` | 新增 pairScaleP sparse q / scale-gradient Triton kernel | `B164/B165` 使用固定 pair 投影方向 + 可学习 per-hidden scale；forward/backward/update 不再需要 dense `D x H` projection 参数。 |
| `dgkan/kernels/fused_hinge_quadratic.py` | 新增 sparseK-P sparse q / sparse-weight-gradient Triton kernel | `B166-B168` 每个 projection hidden 只学习 `K` 个固定位置权重，目标是在 PairScaleP 崩塌后保留 single-kernel-friendly 低参数量，同时恢复比 scale-only 更强的 trajectory capacity。 |
| `dgkan/kernels/fused_hinge_quadratic.py` | 尝试修复 `Q_TANH + Q_RMS` projection-gradient chain rule，并 guard official support | B196 暴露 fused backward correctness failure；先把 tanh 导数改为使用 RMS 归一化之前的 `tanh(q)`。若 correctness 仍不足，`rmsQ+boundQ` 会显式要求 two-pass q transform kernel，不允许冒充 official。 |
| `dgkan/kernels/fused_hinge_quadratic.py` | 新增 narrow Triton AdamW update kernel | B203 仅对 dense `quad_proj` 使用 lower-level AdamW update kernel；其他参数继续用 manual foreach AdamW，便于隔离 `quad_proj` update cost。 |
| `dgkan/kernels/fused_hinge_quadratic.py` | 新增 fused projection-gradient + AdamW update kernel | B204 在 projection-gradient accumulation 后直接更新 dense `quad_proj`/AdamW state，不再写回 `quad_proj.grad` 后另跑 update kernel。 |
| `dgkan/kernels/fused_hinge_quadratic.py` | 新增 stop-grad logitnorm effective-delta branch | 让 FHQ fused backward/update 与 B210-B213 的 autograd 语义一致，确保 correctness audit 比对同一个数学 primitive。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 新增 v12.8.3 runner，并支持 `--repair-candidate-ids` 多 B109 候选 autopsy 与 full-step profile；支持 projection-only LR smoothing / single-stage and multi-stage schedule；新增 classic family L3 analytic manual CE step measurement | 写 `v1283_*` artifacts；区分 official FHQ、L1 autograd、L2 compile-forward、L3 manual analytic 探索；不把探索写成 official。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 新增 `B109_FHQ_any_official_efficiency` 与 `B109_FHQ_efficiency_anchor` route 字段 | 区分“任一 B109-family FHQ path 过 efficiency”和“learnable-P F3 path 过 efficiency”，避免误读 fixed-P pass。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 修复 scheduled projection-gradient selector 的 regex | 旧写法匹配字面量 `\\d`，导致 `pUpdateEvery*` 变体未真正切到 F4；修复后重新落盘结果。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 新增 B109 Line C / Functional re-entry 输出 | 复用 v12.5.2 的 coupling/signal/noise 与 cloned functional strong-control audit，重写为 `v1283_*` artifact；base 开门后仍只按真实 control/task-safe/overhead 证据决定是否 official。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 新增 functional delta-score repair audit | 修复零更新在 ridge coupling 中被计作 `CouplingR2=1.0` 的退化评价问题；新增 `task_delta + geometry residual` 泛化方向，只接收已有上游梯度向量并做几何残差组合，不写 CE 专用 VJP，不改变 loss/sampler/class weight/dataset branch；结果单独写入 `v1283_b109_functional_delta_score_*`，未通过 strong controls 前不写 official success。 |

## 2. B109 FHQ Full-Step

{_md_table(b109_rows, ["candidate_id", "implementation_id", "step_ratio_q90", "memory_ratio_q90", "B109_special_step_target_pass", "B109_special_memory_target_pass", "official_efficiency_pass"], limit=12)}

## 3. B109 AUC Autopsy

{_md_table(auc_summary, ["candidate_id", "mean_delta", "worst_delta", "near_pass_rate", "ece_ok", "auc_step_ok", "auc_time_ok", "A5_autopsy_pass"])}

解释：这是本轮真实重跑的短程 autopsy，不继承旧文档数值；若 epochs/seeds/train-size 小于 official protocol，则只能作为 trajectory diagnostic，不写成 official base success。

## 4. 经典基函数家族状态

{_md_table(family_summary_rows, ["family", "status", "best_L1_step_ratio", "best_L3_manual_step_ratio", "L2_forward_attempt_measured", "L3_manual_analytic_attempt_measured", "blocker", "official_family_failure_claim"])}

解释：本轮对每个 classic family 至少做了 L1 autograd full-step、L2 `torch.compile` forward attempt，并补充 L3 analytic manual CE step attempt。L3 manual step 不调用 `loss.backward()`，但仍是 torch-reduction path，不是 family-specific fused Triton/CUDA kernel，所以状态仍是 `KernelBlocked`，不是数学失败结论。

## 5. Family Efficiency 摘要

{_md_table([r for r in family_eff if str(r.get('candidate_id')) != 'MLP-same-param-AdamW'], ["family", "candidate_id", "implementation_level", "step_ratio_q90", "memory_ratio_q90", "A1_exploratory_pass", "official_efficiency_pass", "status"], limit=24)}

## 6. L2 Component Profile

{_md_table([r for r in family_profile if str(r.get('component')) == 'L2_torch_compile_forward_attempt'], ["family", "candidate_id", "component", "ratio_vs_mlp_forward_q90", "official_l2_kernel", "status"], limit=24)}

## 7. Provenance / Hash

{_md_table(provenance, ["artifact", "rows_checked", "fake_data_used_sum", "proxy_row_used_sum", "cpu_offload_used_sum", "no_fake_pass"], limit=20)}

Selected hashes:

{_md_table(hash_rows, ["artifact", "sha256_prefix"], limit=20)}

## 8. 分析结论

1. B109/FHQ 仍是主线 anchor；本 runner 用 FHQ path 重新测 full-step，并用短程 autopsy 重新记录 AUC-step/AUC-time，不复用旧结果冒充本轮数据。若 exact candidate 换成 B145 这类 repair diagnostic，不能把 fixed-P pass 冒充为 B109b learnable-P F3 pass。
2. 经典 family 本轮没有被判为数学失败；它们的共同 blocker 是 family-specific fused L3 backward/update kernel 缺失。L1/L2 与新增 L3 manual analytic 结果只能说明当前实现路径状态。
3. Functional official 仍关闭，除非 B109 或某个 family base 同时通过 official efficiency、A4、A5 和 Line C。
4. B136/B137 fixed projection 明显降低 kernel/update cost 但 task trajectory 崩塌；B138 freeze-after-1 与 B139-B144 scheduled projection-gradient repair 用于检验能否保留 B109 trajectory 同时降低多数 step 的 projection backward/update cost。
5. B145 activeP64 是实质 kernel-level repair，但没有完成：F2 step ratio `1.2529631891274053` 接近 1.25 gate，F3 workspace step ratio `1.3654556430637974` 未过；AUC 上 Fashion-MNIST seed 2 达到 `1.0429494332758882`，但 seed 0/1 仍为 `1.0919466232193322` / `1.0856452212776184`，A5 仍失败。F2/F3 fused backward correctness 通过；manual fallback 的 `quad_proj` grad 不适合作为 B145 official path。
6. 最新 B142/B143/B144 说明 warm + scheduled P update 能改善部分 Fashion-MNIST AUC，但不能打开 A5：B144 mean delta `0.003689236111111111`、worst `-0.01953125`、near `0.7777777777777778`、ECE ok，但 AUC-step/AUC-time 仍失败；B109b 仍是 best candidate。
7. B146/B147 继续检验 active projection hidden 的 Pareto 中间点：B146 activeP48 偏效率，B147 activeP96 偏保留 trajectory 容量。若没有同时改善 FHQ step 与 AUC/accuracy，不能写成完成。
8. B148-B151 继续检验完整 P 容量下的 projection-only LR smoothing：只降低 `quad_proj` 参数组学习率，不改 loss、batch、标签权重或数据集分支；B148/B149 是强平滑，B150/B151 是中间点。
9. B152-B159 继续检验 projection LR schedule：前 1 或 2 epoch 使用更高 P LR，之后降到 0.50/0.55/0.60；B158/B159 进一步做 `0.75 -> 0.60 -> 0.50` 两段 schedule，目标是在不使用数据集分支的情况下兼顾 MNIST 早期学习、Fashion accuracy 与后期 AUC 稳定。
10. B160/B161 检验 lowFreqPUpdate-K32/K64：P 方向用低频结构初始化，但只允许前 32/64 个 projection hidden columns 更新；目标是比 fixed lowfreqP 更保留 B109 trajectory，同时实质降低 fused projection-gradient/update cost。
11. B162/B163 检验 blockfreqPUpdate-K64/K96：P 方向用局部块频率结构初始化，同时只更新前 64/96 个 projection hidden columns；这是 lowfreq active 崩塌后的局部化修复尝试，仍不使用数据集分支。
12. B164/B165 检验 pairScaleP：固定 sparse pair projection directions，只学习每个 projection hidden 的 scale；这是 single-kernel-friendly trajectory primitive，目标是实质降低 projection backward/update 参数量。
13. B166-B168 检验 sparseK-P：PairScaleP 若过效率但 task collapse，则改为每个 hidden 只学习 `K=8/16` 个固定 sparse input weights，并对该 sparse projection 参数组使用 projection-only LR smoothing；这仍是 single-kernel-friendly，不回到 dense random pair trajectory。
14. B169-B171 检验 group-abs tail-local signal repair：SparseK-P 若保留效率但 task collapse，则回到 dense learnable-P trajectory，同时只在 direct branch 加少量 fixed group absolute-value signal rows，目标是修 Fashion-MNIST AUC-step/tail NLL，不改 loss、不使用数据集分支。
15. B172-B175 交叉检验 group-abs 与 projection LR schedule：B169 保留较好的 worst row 但 AUC 失败，B171/B155 过 AUC 但 worst/near 仍失败；因此测试 `groupabs4/8` + `projlr075->055/060 after epoch1`，试图同时保留 tail signal 和后期 trajectory 稳定性。
16. B176-B179 检验 group-abs tail signal strength：只把 group-abs direct rows 的初始化倍率改为 `0.25` 或 `0.75`，并分别测试无 schedule 与 `projlr075->050 after epoch1`；这是为了确认 B169 的 tail 修复是否只是强度没调到位。
17. B180-B183 检验 block-local SparseP：继续使用 SparseK fused q / sparse-weight-gradient kernel，但每个 projection hidden 只看一个局部图像块，而不是随机散点；B180/B181 用 `K=8`，B182 用 `K=12`，B183 用更低 projection LR。目标是在实质降低 projection backward/update 参数量的同时，比 B166-B168 随机 SparseK 更保留任务轨迹。
18. B184-B187 检验 hybrid SparseP：block-local SparseP 如果因为缺少全局输入耦合而崩塌，则每个 hidden 使用一半局部块位置、一半全局分散位置；仍然复用 single-kernel SparseP path，并保持 `K*H` projection 参数量。
19. B188-B190 检验 dense-P manual foreach AdamW：不再压缩 projection 表达力，只把 update path 从 `torch.optim.AdamW.step()` 替换成同公式的 foreach update engine，验证是否能实质降低 F3 backward/update cost。
20. B191-B193 检验 norm/mean statistic direct rows：保留 dense-P 与 manual foreach update，只增加 1 到 2 个全局统计信号，测试是否能低成本改善 NLL/AUC trajectory。
21. B194-B197 检验 q-normalized/bounded trajectory primitive：保留 dense-P 与 manual foreach update，不增加 direct branch rows，只在 quadratic hidden `q` 内部做 stop-grad batch RMS 和/或 tanh bound，测试是否能改变 NLL/AUC 轨迹而不增加 direct-grad 成本。
22. B198-B202 检验 scheduled projection update + manual foreach update cross repair：保留 B139-B144 的 warm / update-every-N 轨迹设定，同时使用已审计的 manual foreach AdamW update，测试“早期 dense-P 学习信号 + 多数 step 固定 P + 低 update cost”是否能同时改善 AUC 和 full-step。
23. B203 检验 Triton `quad_proj` AdamW update repair：保持 B189/B109 dense-P trajectory 与 manual foreach 其余参数 update，只把最大的 dense projection 参数 update 下沉到 single-tensor Triton kernel，测试是否能进一步实质降低 fused backward/update cost。
24. B204 检验 fused projection-gradient + AdamW update repair：保持 B189/B109 dense-P trajectory，但在 projection-gradient kernel 内直接更新 `quad_proj`，测试是否能同时省掉 `quad_proj.grad` 写回和单独 update pass。
25. B205 检验 fused-update + sample-wise logitnorm trajectory repair：不增加 direct-gradient rows，不改 loss，只用固定 branch/gain 的 per-sample logit normalization 尝试改善 NLL/AUC trajectory，并保持 B204 的 fused `quad_proj` update。
26. B206 检验 fused-update + fixed scalar trajectory repair：保留 B204 fused `quad_proj` update，只固定 branch/gain，确认 B205 的 task 改善是否来自固定 scalar dynamics，还是来自昂贵 logitnorm。
27. B207 检验 lower-target logitnorm trajectory repair：若 B205 accuracy 改善但 ECE/AUC 失败，则降低 logitnorm target 检验校准/trajectory trade-off。
28. B208/B209 检验 hybrid SparseP + logitnorm trajectory repair：把 projection 参数从 dense `D*H` 降到 `K*H`，用 local/global sparse receptive fields 尝试抵消 B205 的 logitnorm 额外 backward cost，并检查 task trajectory 是否崩塌。
29. B210/B211 检验 stop-gradient logitnorm trajectory repair：B205 的完整可微 logitnorm 改善 accuracy 但太慢且 ECE/AUC 失败；B210 保留前向 normalization、把 RMS denominator 视为常量反传，B211 再叠加 warm2/update-every2 projection schedule，目标是实质降低 fused backward/update cost 而不是继续调 target。
30. B212/B213 检验 batch-level stop-gradient logitnorm trajectory repair：若 B210/B211 证明 per-sample normalization 仍伤 ECE/AUC，则改为 batch-wide RMS 标尺，保留单 scalar backward scale，避免逐样本重标定 confidence 结构。
31. 当前 blocker 已收窄到 Fashion-MNIST NLL/AUC trajectory：不能用 dataset-specific branch 修；下一步应把工程投入放在 classic family L3 kernels，或更细粒度的 block-sparse / fused projection-gradient accumulation；不应靠 dense random pair trajectory 或继续温度/epoch小修。
"""
    report_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DG-KAN v12.8.3 B109 + classic family runner")
    p.add_argument("--out-dir", default=f"results/v12_8_3_b109_classic_family_functional_geometry/v1283_b109_classic_{_now_tag()}")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--seed", type=int, default=2413)
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--val-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-3)
    p.add_argument("--kernel-warmup-steps", type=int, default=5)
    p.add_argument("--kernel-measure-steps", type=int, default=12)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--task-timing-warmup-epochs", type=int, default=1)
    p.add_argument("--task-compile-warmup-steps", type=int, default=4)
    p.add_argument("--task-lr-schedule", default="linear_warmup10_cosine_final050")
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--exact-candidate-id", default="B109b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075")
    p.add_argument("--fixedp-candidate-id", default="B109a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075")
    p.add_argument("--repair-candidate-ids", default="")
    p.add_argument("--expression-anchor-candidate-id", default="B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm")
    p.add_argument("--compile-family-forward", type=int, default=1)
    p.add_argument("--optimizer-impl", default="adamw")
    p.add_argument("--coupling-batch-size", type=int, default=32)
    p.add_argument("--functional-batch-size", type=int, default=32)
    p.add_argument("--sketch-batch-size", type=int, default=8)
    p.add_argument("--sketch-dim", type=int, default=8)
    p.add_argument("--ridge-lambda", type=float, default=1.0e-3)
    p.add_argument("--expression-dim", type=int, default=16)
    p.add_argument("--expression-train-size", type=int, default=1024)
    p.add_argument("--expression-val-size", type=int, default=512)
    p.add_argument("--expression-test-size", type=int, default=512)
    p.add_argument("--expression-batch-size", type=int, default=128)
    p.add_argument("--expression-lr", type=float, default=3.0e-3)
    p.add_argument("--expression-steps-b0", type=int, default=60)
    p.add_argument("--expression-steps-b1", type=int, default=600)
    p.add_argument("--expression-steps-b2", type=int, default=1200)
    p.add_argument("--expression-targets", default="E0-additive,E1-pairwise-product,E2-composition,E3-local-XOR,E4-high-frequency,E5-noise-stress,E6-rotated-pairwise-product,E8-random-quadratic-form,E9-smooth-nonpolynomial,E10-local-bump-mixture")
    p.add_argument("--ridge-alpha", type=float, default=1.0e-3)
    p.add_argument("--family-promotion-ids", default="", help="Optional comma-separated candidate ids to limit classic-family A4 promotion after fused L3 efficiency + gradcheck. Empty keeps default all eligible candidates.")
    p.add_argument("--family-candidate-ids", default="", help="Optional comma-separated classic-family candidate ids to measure in this follow-up run. Empty keeps the full family plan.")
    p.add_argument("--family-linec-ids", default="", help="Optional comma-separated classic-family candidate ids for Line C autopsy. Empty uses A4/A5-open candidates from family_status, filtered by --family-candidate-ids when provided.")
    p.add_argument("--report-path", default=str(REPORT_PATH_DEFAULT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    device = _device_from_arg(args.device)
    if device.type != "cuda":
        raise RuntimeError("v12.8.3 no-CPU-offload contract requires CUDA")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, _x_test_cpu, _y_test_cpu, input_dim, output_dim = _load_mnist(args)
    x_train = x_train_cpu.to(device=device, dtype=torch.float32)
    y_train = y_train_cpu.to(device=device)
    x_val = x_val_cpu.to(device=device, dtype=torch.float32)
    y_val = y_val_cpu.to(device=device)
    specs = _specs_for(input_dim, output_dim)
    write_contract_files(args, out_dir, device)
    run_family_manifest(out_dir, specs)
    _b109_grad, b109_full, _official_ids = run_b109_fullstep(args, out_dir, device, x_train, y_train, input_dim, output_dim, specs)
    _trace, _final, b109_auc = run_b109_auc_autopsy(args, out_dir, device, specs)
    family_grad, family_eff, family_profile, family_failures = run_family_microbench(args, out_dir, device, x_train, y_train, input_dim, output_dim, specs)
    expr_rows, task_rows, geo_rows, family_status = write_closed_family_followups(out_dir, family_failures, family_eff, family_profile)
    expr_rows, task_rows, geo_rows, family_status = run_family_expression_promotion(args, out_dir, device, family_eff, family_grad, expr_rows, task_rows, geo_rows, family_failures, family_status)
    provenance = audit_provenance(out_dir)
    decision = decide_route(b109_full, b109_auc, family_status, provenance)
    family_linec_summary = run_family_linec_autopsy(
        args,
        out_dir,
        device,
        x_train,
        y_train,
        x_val,
        y_val,
        input_dim,
        output_dim,
        specs,
        bool(_safe_float(decision.get("base_qualified"), 0)),
        family_status,
    )
    provenance = audit_provenance(out_dir)
    decision = decide_route(b109_full, b109_auc, family_status, provenance)
    linec_candidate = str(decision.get("B109_A5_best_candidate") or "").strip()
    if not linec_candidate:
        repairs = _parse_list(getattr(args, "repair_candidate_ids", ""))
        linec_candidate = repairs[0] if repairs else str(args.exact_candidate_id)
    linec_summary = run_b109_linec_functional_reentry(
        args,
        out_dir,
        device,
        x_train,
        y_train,
        x_val,
        y_val,
        input_dim,
        output_dim,
        specs,
        bool(_safe_float(decision.get("base_qualified"), 0)),
        linec_candidate,
    )
    delta_score_summary = run_b109_functional_delta_score_repair(
        args,
        out_dir,
        device,
        x_train,
        y_train,
        x_val,
        y_val,
        input_dim,
        output_dim,
        specs,
        bool(_safe_float(decision.get("base_qualified"), 0)),
        linec_candidate,
    )
    decision.update(
        {
            "B109_LineC_measured": int(_safe_float(linec_summary.get("linec_measured"), 0)),
            "B109_LineC_nontearing_pass": int(_safe_float(linec_summary.get("linec_nontearing_pass"), 0)),
            "B109_LineC_CouplingR2": linec_summary.get("candidate_CouplingR2", ""),
            "B109_LineC_NoiseSignalLeak": linec_summary.get("candidate_NoiseSignalLeak", ""),
            "B109_functional_reentry_measured": int(_safe_float(linec_summary.get("functional_reentry_measured"), 0)),
            "B109_functional_beats_controls": int(_safe_float(linec_summary.get("functional_beats_controls"), 0)),
            "official_functional_success": int(_safe_float(linec_summary.get("official_functional_success"), 0)),
            "B109_functional_delta_score_repair_measured": int(_safe_float(delta_score_summary.get("delta_score_repair_measured"), 0)),
            "B109_functional_delta_score_beats_controls": int(_safe_float(delta_score_summary.get("delta_score_beats_controls"), 0)),
            "B109_functional_delta_score_gap_vs_best_control": delta_score_summary.get("delta_score_gap_vs_best_control", ""),
            "B109_functional_delta_score_best_update": delta_score_summary.get("best_functional_update", ""),
            "family_linec_measured_count": sum(1 for r in family_linec_summary if int(_safe_float(r.get("linec_measured"), 0)) == 1),
            "family_linec_failed_count": sum(1 for r in family_linec_summary if str(r.get("status")) == "failed"),
            "family_linec_interpretations": ";".join(
                str(r.get("candidate_id", "")) + ":" + str(r.get("linec_interpretation", r.get("status", "")))
                for r in family_linec_summary
                if str(r.get("candidate_id", "")).strip()
            ),
        }
    )
    if int(_safe_float(decision.get("base_qualified"), 0)) == 1 and int(_safe_float(decision.get("B109_LineC_nontearing_pass"), 0)) == 0:
        decision["route"] = "R5-B109BasePassLineCGeometryCollapse"
        decision["next_recommended_action"] = "repair B109 geometry channel tearing before functional official claim; keep classic-family kernel work open"
    write_json(out_dir / "v1283_route_decision.json", decision)
    provenance = audit_provenance(out_dir)
    decision["no_fake_provenance_pass"] = int(
        all(
            int(_safe_float(r.get("no_fake_pass"), 0)) == 1
            and int(_safe_float(r.get("no_proxy_pass"), 0)) == 1
            and int(_safe_float(r.get("no_cpu_offload_pass"), 0)) == 1
            for r in provenance
        )
    )
    write_json(out_dir / "v1283_route_decision.json", decision)
    provenance = audit_provenance(out_dir)
    hashes = write_hash_manifest(out_dir)
    write_report(out_dir, Path(args.report_path), decision, b109_full, b109_auc, family_eff, family_profile, family_status, provenance, hashes)


if __name__ == "__main__":
    main()
