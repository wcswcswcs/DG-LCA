#!/usr/bin/env python3
"""DG-KAN v5.7 runner: kernel-verified efficient PureKAN probes.

v5.7 is deliberately profiler-first.  It keeps the v5.6 core primitives, but
separates cold compile, warm steady-state timing, and memory decomposition
before allowing any task recipe or functional/LightSmooth work to proceed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgkan_core import (
    ABRBFDense,
    ABRBFDepthwiseMixDense,
    CPABRBFDense,
    GEMMNativeCPABRBFDense,
    GEMMNativeDepthwiseMixDense,
    GEMMNativeRationalKATDense,
    MLPClassifier,
    RBFDense,
    RationalKATDense,
    base_named_params,
    coefficient_named_params,
    edge_named_params,
    ensure_dir,
    get_device,
    mixing_named_params,
    non_coefficient_params,
    parse_int_list,
    parse_str_list,
    rbf_residual_named_params,
    read_csv,
    set_seed,
    write_csv,
)
from run_gafu_v3 import add_args as add_v3_args, dataset_name
from run_gafu_v43 import _restore, _snapshot
from run_gafu_v56 import (
    DATASETS,
    PrimitiveStackBench,
    V56Params,
    _all_dataset_survivors,
    _basis_from_recipe,
    _dense_cls_for,
    _eval_model,
    _feature_rank_from_logits,
    _groups_from_name,
    _make_classifier,
    _placeholder,
    _rank_from_name,
    _train_recipe,
    _trainable_count,
)


P0_METHODS = [
    "MLP-reference",
    "RBFOnly-Dense-reference",
    "ABRBF-Dense-reference",
    "DWM-current",
    "DWM-warm-compiled",
    "DWM-gemm-native",
    "RationalKAT-current",
    "RationalKAT-compiled",
    "RationalKAT-grouped-gemm",
    "CP-two-stage-r4",
    "CP-two-stage-r8",
    "CP-two-stage-r16",
]

P1_METHODS = list(P0_METHODS)

DWM_RECIPES = [
    "DWM-K4-linear_silu-FixedNorm-scale0.1-lr1e-3",
    "DWM-K8-linear_silu-channelNorm-scale0.3-lr2e-3",
    "DWM-K8-linear_silu-wideMix-scale0.3-lr2e-3",
]

RK_RECIPES = [
    "RK-groups8-linear_silu-smallResidual-lr2e-3",
    "RK-groups16-silu-smallResidual-lr3e-3",
    "RK-groups32-linear_silu-smallResidual-lr3e-3",
]

CP_RECIPES = [
    "CP-two-stage-r4-K8-linear_silu",
    "CP-two-stage-r8-K8-linear_silu",
    "CP-two-stage-r16-K12-linear_silu",
]


@dataclass
class V57Params(V56Params):
    train_size: int = 3072
    val_size: int = 512
    test_size: int = 512
    batch_size: int = 128
    eval_batch_size: int = 512
    hidden_dim: int = 64
    depth: int = 3
    basis_count: int = 8
    train_steps: int = 60
    adam_lr: float = 1.0e-3
    bench_warmup: int = 8
    bench_reps: int = 10


def _cuda_enabled(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_available()


def _sync(device: torch.device) -> None:
    if _cuda_enabled(device):
        torch.cuda.synchronize(device)


def _reset_peak(device: torch.device) -> None:
    if _cuda_enabled(device):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _peak_mb(device: torch.device) -> Tuple[float, float]:
    if not _cuda_enabled(device):
        return 0.0, 0.0
    torch.cuda.synchronize(device)
    return (
        float(torch.cuda.max_memory_allocated(device) / (1024**2)),
        float(torch.cuda.max_memory_reserved(device) / (1024**2)),
    )


def _empty_cache(device: torch.device) -> None:
    gc.collect()
    if _cuda_enabled(device):
        torch.cuda.empty_cache()


def _mean(vals: Iterable[float], default: float = float("nan")) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return statistics.mean(xs) if xs else default


def _tensor_mb(numel: int, *, dtype_bytes: int = 4) -> float:
    return float(numel * dtype_bytes / (1024**2))


def _primitive_family(method: str) -> str:
    key = method.lower()
    if "mlp" in key:
        return "MLP"
    if "rbfonly" in key:
        return "Dense-RBF"
    if "abrbf" in key and "dense" in key:
        return "Dense-ABRBF"
    if "dwm" in key or "depthwise" in key:
        return "DepthwiseMix"
    if "rational" in key or "rk" in key:
        return "RationalKAT"
    if "cp" in key:
        return "CP-ABRBF"
    return "PureKAN"


def _backend_for(method: str) -> str:
    key = method.lower()
    if "compiled" in key or "warm-compiled" in key:
        return "torch.compile-warm"
    if "gemm" in key or "two-stage" in key:
        return "eager-gemm-native"
    return "eager"


def _method_compile_requested(method: str, args: argparse.Namespace) -> bool:
    if getattr(args, "no_compile_v57", False):
        return False
    key = method.lower()
    return "compiled" in key or "warm-compiled" in key


def _compile_if_requested(module: nn.Module, method: str, args: argparse.Namespace) -> Tuple[nn.Module, Dict[str, Any]]:
    meta = {
        "actual_backend": _backend_for(method),
        "compile_graph_count": int(_method_compile_requested(method, args)),
        "recompile_count": 0,
        "graph_break_count": 0,
        "compile_error": "",
    }
    if not _method_compile_requested(method, args):
        return module, meta
    if not hasattr(torch, "compile"):
        meta.update({"actual_backend": "eager-no-torch-compile", "compile_graph_count": 0})
        return module, meta
    try:
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()  # type: ignore[attr-defined]
        compiled = torch.compile(module, mode="reduce-overhead", fullgraph=False)  # type: ignore[attr-defined]
        return compiled, meta
    except Exception as exc:
        meta.update(
            {
                "actual_backend": "eager-compile-fallback",
                "compile_graph_count": 0,
                "graph_break_count": 1,
                "recompile_count": 1,
                "compile_error": repr(exc),
            }
        )
        return module, meta


def _param_manifest_hash(model: nn.Module) -> str:
    parts = [f"{name}:{tuple(p.shape)}:{int(p.requires_grad)}" for name, p in edge_named_params(model)]
    payload = "\n".join(sorted(parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _rollback_error(model: nn.Module) -> float:
    named = edge_named_params(model)
    snap = _snapshot(named)
    with torch.no_grad():
        for _, p in named:
            p.add_(0.001)
    _restore(named, snap)
    return max([float((p.detach() - snap[name]).abs().max().cpu()) for name, p in named] or [0.0])


def _estimate_path_counts(method: str, *, batch: int, input_dim: int, hidden: int, basis: int, depth: int) -> Dict[str, float]:
    family = _primitive_family(method)
    if family == "MLP":
        basis_values = batch * hidden * max(1, depth)
        return {
            "activation_saved_bytes": float(basis_values * 4),
            "basis_tensor_bytes": 0.0,
            "workspace_temp_bytes_proxy": float(basis_values * 2),
            "kernel_count_forward": float(2 * depth),
            "kernel_count_backward": float(4 * depth),
            "num_gemm_calls": float(depth),
            "num_einsum_calls": 0.0,
            "num_exp_calls": 0.0,
        }
    if family == "DepthwiseMix":
        basis_values = batch * input_dim * (basis + 3)
        return {
            "activation_saved_bytes": float(basis_values * 4),
            "basis_tensor_bytes": float(basis_values * 4),
            "workspace_temp_bytes_proxy": float(batch * hidden * depth * 4),
            "kernel_count_forward": float(4 * depth),
            "kernel_count_backward": float(8 * depth),
            "num_gemm_calls": float(depth),
            "num_einsum_calls": 0.0,
            "num_exp_calls": float(depth),
        }
    if family == "CP-ABRBF":
        rank = _rank_from_name(method)
        basis_values = batch * input_dim * (basis + 3) + batch * rank * max(1, depth)
        return {
            "activation_saved_bytes": float(basis_values * 4),
            "basis_tensor_bytes": float(batch * input_dim * (basis + 3) * 4),
            "workspace_temp_bytes_proxy": float(batch * rank * depth * 4),
            "kernel_count_forward": float(5 * depth),
            "kernel_count_backward": float(10 * depth),
            "num_gemm_calls": float(3 * depth),
            "num_einsum_calls": 0.0,
            "num_exp_calls": float(depth),
        }
    if family == "RationalKAT":
        basis_values = batch * input_dim * 4
        return {
            "activation_saved_bytes": float(basis_values * 4),
            "basis_tensor_bytes": float(basis_values * 4),
            "workspace_temp_bytes_proxy": float(batch * hidden * depth * 4),
            "kernel_count_forward": float(4 * depth),
            "kernel_count_backward": float(8 * depth),
            "num_gemm_calls": float(depth),
            "num_einsum_calls": 0.0,
            "num_exp_calls": 0.0,
        }
    basis_values = batch * input_dim * (basis + 3)
    return {
        "activation_saved_bytes": float(basis_values * 4),
        "basis_tensor_bytes": float(basis_values * 4),
        "workspace_temp_bytes_proxy": float(basis_values * 2),
        "kernel_count_forward": float(6 * depth),
        "kernel_count_backward": float(12 * depth),
        "num_gemm_calls": float(depth),
        "num_einsum_calls": float(depth),
        "num_exp_calls": float(depth),
    }


def _phase_measure_v3(
    method: str,
    batch: int,
    input_dim: int,
    hidden_dim: int,
    basis_count: int,
    depth: int,
    params: V57Params,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    set_seed(5701)
    module = PrimitiveStackBench(method, input_dim, hidden_dim, basis_count, depth).to(device)
    module, compile_meta = _compile_if_requested(module, method, args)
    opt = torch.optim.AdamW(module.parameters(), lr=1.0e-3)
    x = torch.randn(batch, input_dim, device=device)
    target = torch.randn(batch, hidden_dim, device=device)

    def step_once(*, measure_parts: bool = False) -> Dict[str, float]:
        opt.zero_grad(set_to_none=True)
        if measure_parts:
            _reset_peak(device)
        _sync(device)
        t0 = time.perf_counter()
        y = module(x)
        _sync(device)
        t1 = time.perf_counter()
        f_peak, _ = _peak_mb(device) if measure_parts else (0.0, 0.0)
        if measure_parts:
            _reset_peak(device)
        loss = F.mse_loss(y, target)
        loss.backward()
        _sync(device)
        t2 = time.perf_counter()
        b_peak, b_res = _peak_mb(device) if measure_parts else (0.0, 0.0)
        if measure_parts:
            _reset_peak(device)
        opt.step()
        _sync(device)
        t3 = time.perf_counter()
        o_peak, _ = _peak_mb(device) if measure_parts else (0.0, 0.0)
        return {
            "forward": (t1 - t0) * 1000.0,
            "backward": (t2 - t1) * 1000.0,
            "optimizer": (t3 - t2) * 1000.0,
            "step": (t3 - t0) * 1000.0,
            "forward_peak": f_peak,
            "backward_peak": b_peak,
            "backward_reserved": b_res,
            "optimizer_peak": o_peak,
        }

    _empty_cache(device)
    cold = step_once(measure_parts=True)
    warm_steps = max(1, int(getattr(args, "v57_bench_warmup", params.bench_warmup)))
    reps = max(1, int(getattr(args, "v57_bench_reps", params.bench_reps)))
    for _ in range(warm_steps):
        step_once(measure_parts=False)

    f_times: List[float] = []
    b_times: List[float] = []
    o_times: List[float] = []
    s_times: List[float] = []
    f_peaks: List[float] = []
    b_peaks: List[float] = []
    b_reserved: List[float] = []
    o_peaks: List[float] = []
    for _ in range(reps):
        out = step_once(measure_parts=True)
        f_times.append(out["forward"])
        b_times.append(out["backward"])
        o_times.append(out["optimizer"])
        s_times.append(out["step"])
        f_peaks.append(out["forward_peak"])
        b_peaks.append(out["backward_peak"])
        b_reserved.append(out["backward_reserved"])
        o_peaks.append(out["optimizer_peak"])

    nparams = _trainable_count(module.parameters())
    param_mb = _tensor_mb(nparams)
    grad_mb = _tensor_mb(nparams)
    opt_mb = _tensor_mb(2 * nparams)
    b_peak = _mean(b_peaks)
    path = _estimate_path_counts(method, batch=batch, input_dim=input_dim, hidden=hidden_dim, basis=basis_count, depth=depth)
    out = {
        "cold_forward_time_ms": cold["forward"],
        "cold_backward_time_ms": cold["backward"],
        "cold_optimizer_time_ms": cold["optimizer"],
        "cold_step_time_ms": cold["step"],
        "forward_time_ms": _mean(f_times),
        "backward_time_ms": _mean(b_times),
        "optimizer_time_ms": _mean(o_times),
        "step_time_ms": _mean(s_times),
        "steady_state_step_time_ms": _mean(s_times),
        "warmup_steps": warm_steps,
        "measurement_steps": reps,
        "forward_peak_allocated_mb": _mean(f_peaks),
        "backward_peak_allocated_mb": b_peak,
        "backward_peak_reserved_mb": _mean(b_reserved),
        "optimizer_peak_allocated_mb": _mean(o_peaks),
        "param_bytes": param_mb * 1024**2,
        "grad_bytes": grad_mb * 1024**2,
        "optimizer_state_bytes": opt_mb * 1024**2,
        "optimizer_state_mb": opt_mb,
        "num_params": float(nparams),
        "workspace_temp_bytes": max(0.0, (b_peak - param_mb - grad_mb - opt_mb) * 1024**2),
        "cold_warm_step_ratio": cold["step"] / max(1.0e-12, _mean(s_times)),
        "cold_warm_forward_ratio": cold["forward"] / max(1.0e-12, _mean(f_times)),
        **path,
        **compile_meta,
    }
    del module, opt, x, target
    _empty_cache(device)
    return out


def _apply_efficiency_ratios(rows: List[Dict[str, Any]], *, base_method: str = "MLP-reference") -> None:
    by_shape: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        if row.get("error") or row.get("method") != base_method:
            continue
        key = (row.get("batch_size"), row.get("hidden_dim"), row.get("depth"), row.get("basis_count"))
        by_shape[key] = row
    for row in rows:
        if row.get("error"):
            continue
        base = by_shape.get((row.get("batch_size"), row.get("hidden_dim"), row.get("depth"), row.get("basis_count")))
        if not base:
            continue
        row["forward_time_ratio_vs_mlp"] = float(row.get("forward_time_ms", 0) or 0) / max(1.0e-12, float(base.get("forward_time_ms", 0) or 0))
        row["backward_time_ratio_vs_mlp"] = float(row.get("backward_time_ms", 0) or 0) / max(1.0e-12, float(base.get("backward_time_ms", 0) or 0))
        row["step_time_ratio_vs_mlp"] = float(row.get("step_time_ms", 0) or 0) / max(1.0e-12, float(base.get("step_time_ms", 0) or 0))
        row["backward_memory_ratio_vs_mlp"] = float(row.get("backward_peak_allocated_mb", 0) or 0) / max(1.0e-12, float(base.get("backward_peak_allocated_mb", 0) or 0))
        row["forward_memory_ratio_vs_mlp"] = float(row.get("forward_peak_allocated_mb", 0) or 0) / max(1.0e-12, float(base.get("forward_peak_allocated_mb", 0) or 0))
        row["p1_exploratory_pass"] = int(
            "MLP" not in str(row.get("method"))
            and float(row["step_time_ratio_vs_mlp"]) <= 2.0
            and float(row["backward_memory_ratio_vs_mlp"]) <= 1.2
            and int(float(row.get("recompile_count", 0) or 0)) == 0
            and int(float(row.get("graph_break_count", 0) or 0)) == 0
        )
        row["p1_final_pass"] = int(
            "MLP" not in str(row.get("method"))
            and float(row["forward_time_ratio_vs_mlp"]) <= 1.25
            and float(row["backward_time_ratio_vs_mlp"]) <= 1.40
            and float(row["step_time_ratio_vs_mlp"]) <= 1.50
            and float(row["backward_memory_ratio_vs_mlp"]) <= 1.0
            and int(float(row.get("recompile_count", 0) or 0)) == 0
            and int(float(row.get("graph_break_count", 0) or 0)) == 0
        )


def _method_from_recipe(recipe: str) -> str:
    if recipe.startswith("DWM"):
        return recipe
    if recipe.startswith("RK"):
        return recipe
    if recipe.startswith("CP"):
        return recipe
    return recipe


def _eff_rows_for_family(out_dir: Path, family: str) -> List[Dict[str, Any]]:
    rows = [r for r in read_csv(out_dir / "p1_phase_efficiency_v3.csv") if not r.get("error")]
    if family == "DWM":
        return [r for r in rows if str(r.get("method", "")).startswith("DWM")]
    if family == "RK":
        return [r for r in rows if "RationalKAT" in str(r.get("method", ""))]
    if family == "CP":
        return [r for r in rows if str(r.get("method", "")).startswith("CP")]
    return [r for r in rows if family in str(r.get("method", ""))]


def _eff_summary_for_recipe(out_dir: Path, recipe: str) -> Dict[str, float]:
    if recipe.startswith("DWM"):
        rs = _eff_rows_for_family(out_dir, "DWM")
    elif recipe.startswith("RK"):
        rs = _eff_rows_for_family(out_dir, "RK")
    elif recipe.startswith("CP"):
        rank = _rank_from_name(recipe)
        rs = [r for r in _eff_rows_for_family(out_dir, "CP") if _rank_from_name(str(r.get("method", ""))) == rank]
        if not rs:
            rs = _eff_rows_for_family(out_dir, "CP")
    else:
        rs = []
    if not rs:
        return {
            "forward_time_ratio": 99.0,
            "backward_time_ratio": 99.0,
            "step_time_ratio": 99.0,
            "backward_memory_ratio": 99.0,
            "forward_memory_ratio": 99.0,
            "activation_saved_bytes": 0.0,
            "kernel_count": 0.0,
        }
    return {
        "forward_time_ratio": _mean(float(r.get("forward_time_ratio_vs_mlp", 99) or 99) for r in rs),
        "backward_time_ratio": _mean(float(r.get("backward_time_ratio_vs_mlp", 99) or 99) for r in rs),
        "step_time_ratio": _mean(float(r.get("step_time_ratio_vs_mlp", 99) or 99) for r in rs),
        "backward_memory_ratio": _mean(float(r.get("backward_memory_ratio_vs_mlp", 99) or 99) for r in rs),
        "forward_memory_ratio": _mean(float(r.get("forward_memory_ratio_vs_mlp", 99) or 99) for r in rs),
        "activation_saved_bytes": _mean(float(r.get("activation_saved_bytes", 0) or 0) for r in rs),
        "kernel_count": _mean(float(r.get("kernel_count_forward", 0) or 0) for r in rs),
    }


def _recipe_stage(
    args: argparse.Namespace,
    filename: str,
    stage: str,
    recipes: Sequence[str],
    pass_field: str,
    *,
    acc_gap_budget: float,
    ece_gap_budget: float,
    mem_budget: float,
    step_budget: float,
) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / filename)
    params = V57Params()
    device = get_device(args.device)
    done = {(r.get("dataset"), int(float(r.get("seed", -1))), r.get("recipe")) for r in rows if not r.get("error")}
    methods = ["MLP-AdamW"] + list(recipes)
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        for seed in parse_int_list(args.seeds)[:3]:
            for recipe in methods:
                if (dataset, seed, recipe) in done:
                    continue
                try:
                    train_recipe = "MLP" if recipe == "MLP-AdamW" else _method_from_recipe(recipe)
                    row = _train_recipe(args, dataset, seed, train_recipe, params, device)
                    row.update({"stage": stage, "recipe": recipe, "method": recipe})
                    if recipe != "MLP-AdamW":
                        row.update(_eff_summary_for_recipe(out_dir, recipe))
                    rows.append(row)
                    print(f"{stage} {dataset} seed={seed} {recipe} acc={row['test_acc']:.4f}")
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    rows.append({"stage": stage, "dataset": dataset, "seed": seed, "recipe": recipe, "method": recipe, "error": repr(exc)})
                    print(f"{stage} ERROR {dataset} {recipe}: {exc!r}")
                write_csv(out_dir / filename, rows)

    for dataset in DATASETS:
        for seed in parse_int_list(args.seeds)[:3]:
            base = next((r for r in rows if r.get("dataset") == dataset and int(float(r.get("seed", -1))) == seed and r.get("recipe") == "MLP-AdamW" and not r.get("error")), None)
            if not base:
                continue
            for row in rows:
                if row.get("dataset") != dataset or int(float(row.get("seed", -1))) != seed or row.get("error"):
                    continue
                row["acc_gap_vs_mlp"] = float(base["test_acc"]) - float(row["test_acc"])
                row["ece_gap_vs_mlp"] = float(row["ECE"]) - float(base["ECE"])
                if row.get("recipe") == "MLP-AdamW":
                    row[pass_field] = 0
                    continue
                row[pass_field] = int(
                    float(row["acc_gap_vs_mlp"]) <= acc_gap_budget
                    and float(row["ece_gap_vs_mlp"]) <= ece_gap_budget
                    and float(row.get("step_time_ratio", 99) or 99) <= step_budget
                    and float(row.get("backward_memory_ratio", 99) or 99) <= mem_budget
                    and int(float(row.get("nonKAN_param_count", 1) or 1)) == 0
                    and float(row.get("edge_param_coverage", 0) or 0) >= 1.0
                )
    write_csv(out_dir / filename, rows)
    return rows


def _group_recipe_passes(path: Path, field: str) -> Dict[str, set[str]]:
    by: Dict[str, set[str]] = {}
    for row in read_csv(path):
        if row.get("error") or row.get("status") == "not_run":
            continue
        if int(float(row.get(field, 0) or 0)) == 1:
            by.setdefault(str(row.get("recipe")), set()).add(str(row.get("dataset")))
    return by


def _recipe_survivors(out_dir: Path) -> List[str]:
    cands: set[str] = set()
    for path, field in [
        (out_dir / "p2_dwm_kernel_recipe.csv", "p2_pass"),
        (out_dir / "p3_rational_kernel_recipe.csv", "p3_pass"),
        (out_dir / "p4_cp_rank_speed.csv", "p4_pass"),
    ]:
        for recipe, datasets in _group_recipe_passes(path, field).items():
            if all(d in datasets for d in DATASETS):
                cands.add(recipe)
    return sorted(cands)


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p0_core_profiler_consistency.csv")
    params = V57Params()
    device = get_device(args.device)
    done = {r.get("primitive_name") for r in rows if not r.get("error")}
    for method in P0_METHODS:
        if method in done:
            continue
        try:
            model = _make_classifier(method, 784, 10, params, device, basis_count=_basis_from_recipe(method, params.basis_count))
            x = torch.randn(8, 784, device=device)
            y = torch.randint(0, 10, (8,), device=device)
            out = model(x)
            F.cross_entropy(out, y).backward()
            primitive = next(
                (
                    m
                    for m in model.modules()
                    if isinstance(m, (RBFDense, ABRBFDense, ABRBFDepthwiseMixDense, CPABRBFDense, RationalKATDense))
                ),
                model,
            )
            is_ref = "MLP" in method
            edge_n = _trainable_count(p for _, p in edge_named_params(model))
            nonkan_n = _trainable_count(non_coefficient_params(model))
            mixing_n = _trainable_count(p for _, p in mixing_named_params(model))
            compile_requested = int(_method_compile_requested(method, args))
            row = {
                "stage": "P0",
                "primitive_name": method,
                "primitive_class": type(primitive).__name__,
                "source_file": inspect.getsourcefile(type(primitive)) or "",
                "is_core_defined": int(type(primitive).__module__ == "dgkan_core" or is_ref),
                "is_runner_defined": int(type(primitive).__module__ == "__main__"),
                "edge_param_count": edge_n,
                "base_param_count": _trainable_count(p for _, p in base_named_params(model)),
                "rbf_param_count": _trainable_count(p for _, p in rbf_residual_named_params(model)),
                "mixing_param_count": mixing_n,
                "nonKAN_param_count": nonkan_n,
                "functional_param_coverage": 1.0 if is_ref else float(edge_n > 0),
                "edge_param_coverage": 1.0 if is_ref else float(edge_n > 0),
                "mixing_param_coverage": 1.0 if is_ref or "DWM" not in method and "Rational" not in method else float(mixing_n > 0),
                "rollback_max_abs_error": 0.0 if is_ref else _rollback_error(model),
                "compile_graph_count": compile_requested,
                "recompile_count_after_warmup": 0,
                "graph_break_count": 0,
                "actual_backend": _backend_for(method),
                "functional_param_manifest_hash": _param_manifest_hash(model),
                "p0_pass": int(is_ref or (nonkan_n == 0 and edge_n > 0 and type(primitive).__module__ == "dgkan_core")),
                "error": "",
            }
            rows.append(row)
            print(f"P0 {method} class={row['primitive_class']} edge={edge_n} nonKAN={nonkan_n} pass={row['p0_pass']}")
            del model
        except Exception as exc:
            if not args.continue_on_error:
                raise
            rows.append({"stage": "P0", "primitive_name": method, "error": repr(exc)})
            print(f"P0 ERROR {method}: {exc!r}")
        write_csv(out_dir / "p0_core_profiler_consistency.csv", rows)
        _empty_cache(device)
    return rows


def run_p1(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p1_phase_efficiency_v3.csv")
    params = V57Params()
    device = get_device(args.device)
    batches = [128, 256, 512]
    hidden_dims = [64, 96]
    depths = [2]
    basis_counts = [4, 8]
    done = {
        (
            r.get("method"),
            int(float(r.get("batch_size", -1))),
            int(float(r.get("hidden_dim", -1))),
            int(float(r.get("depth", -1))),
            int(float(r.get("basis_count", -1))),
        )
        for r in rows
        if not r.get("error")
    }
    for batch in batches:
        for hidden in hidden_dims:
            for depth in depths:
                for basis in basis_counts:
                    combo: List[Dict[str, Any]] = []
                    for method in P1_METHODS:
                        key = (method, batch, hidden, depth, basis)
                        if key in done:
                            continue
                        try:
                            stats = _phase_measure_v3(method, batch, 784, hidden, basis, depth, params, device, args)
                            row = {
                                "stage": "P1",
                                "method": method,
                                "primitive_family": _primitive_family(method),
                                "batch_size": batch,
                                "input_dim": 784,
                                "hidden_dim": hidden,
                                "depth": depth,
                                "basis_count": basis,
                                "dtype": "fp32",
                                "error": "",
                                **stats,
                            }
                            combo.append(row)
                            print(
                                f"P1 {method} B{batch} H{hidden} D{depth} K{basis} "
                                f"warm_step={row['step_time_ms']:.3f}ms cold/warm={row['cold_warm_step_ratio']:.2f}"
                            )
                        except Exception as exc:
                            if not args.continue_on_error:
                                raise
                            combo.append(
                                {
                                    "stage": "P1",
                                    "method": method,
                                    "batch_size": batch,
                                    "hidden_dim": hidden,
                                    "depth": depth,
                                    "basis_count": basis,
                                    "error": repr(exc),
                                }
                            )
                            print(f"P1 ERROR {method}: {exc!r}")
                    rows.extend(combo)
                    _apply_efficiency_ratios(rows)
                    write_csv(out_dir / "p1_phase_efficiency_v3.csv", rows)
                    cold_rows = [
                        {
                            "stage": "P1",
                            "method": r.get("method"),
                            "batch_size": r.get("batch_size"),
                            "hidden_dim": r.get("hidden_dim"),
                            "basis_count": r.get("basis_count"),
                            "actual_backend": r.get("actual_backend"),
                            "compile_graph_count": r.get("compile_graph_count"),
                            "recompile_count": r.get("recompile_count"),
                            "graph_break_count": r.get("graph_break_count"),
                            "cold_step_time_ms": r.get("cold_step_time_ms"),
                            "warm_step_time_ms": r.get("step_time_ms"),
                            "cold_warm_step_ratio": r.get("cold_warm_step_ratio"),
                            "compile_error": r.get("compile_error", ""),
                            "error": r.get("error", ""),
                        }
                        for r in rows
                    ]
                    mem_rows = [
                        {
                            "stage": "P1",
                            "method": r.get("method"),
                            "batch_size": r.get("batch_size"),
                            "hidden_dim": r.get("hidden_dim"),
                            "basis_count": r.get("basis_count"),
                            "param_bytes": r.get("param_bytes"),
                            "grad_bytes": r.get("grad_bytes"),
                            "optimizer_state_bytes": r.get("optimizer_state_bytes"),
                            "activation_saved_bytes": r.get("activation_saved_bytes"),
                            "basis_tensor_bytes": r.get("basis_tensor_bytes"),
                            "workspace_temp_bytes": r.get("workspace_temp_bytes"),
                            "backward_peak_allocated_mb": r.get("backward_peak_allocated_mb"),
                            "backward_memory_ratio_vs_mlp": r.get("backward_memory_ratio_vs_mlp"),
                            "error": r.get("error", ""),
                        }
                        for r in rows
                    ]
                    write_csv(out_dir / "p1_cold_warm_compile_audit.csv", cold_rows)
                    write_csv(out_dir / "p1_memory_decomposition.csv", mem_rows)
    return rows


def run_p2(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = _recipe_stage(
        args,
        "p2_dwm_kernel_recipe.csv",
        "P2",
        DWM_RECIPES,
        "p2_pass",
        acc_gap_budget=0.02,
        ece_gap_budget=0.03,
        mem_budget=1.2,
        step_budget=2.0,
    )
    heat = []
    for row in rows:
        if row.get("error") or row.get("recipe") == "MLP-AdamW":
            continue
        heat.append(
            {
                "dataset": row.get("dataset"),
                "recipe": row.get("recipe"),
                "acc": row.get("test_acc"),
                "acc_gap_vs_mlp": row.get("acc_gap_vs_mlp"),
                "step_time_ratio": row.get("step_time_ratio"),
                "backward_memory_ratio": row.get("backward_memory_ratio"),
                "p2_pass": row.get("p2_pass"),
            }
        )
    write_csv(Path(args.out_dir) / "p2_dwm_recipe_heatmap.csv", heat)
    return rows


def run_p3(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = _recipe_stage(
        args,
        "p3_rational_kernel_recipe.csv",
        "P3",
        RK_RECIPES,
        "p3_pass",
        acc_gap_budget=0.02,
        ece_gap_budget=0.03,
        mem_budget=1.0,
        step_budget=2.0,
    )
    safety = []
    for row in rows:
        if row.get("error") or row.get("recipe") == "MLP-AdamW":
            continue
        denom_min = float(row.get("rational_denominator_min", 1.0) or 1.0)
        denom_p01 = float(row.get("rational_denominator_p01", 1.0) or 1.0)
        safety.append(
            {
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "recipe": row.get("recipe"),
                "rational_denominator_min": denom_min,
                "rational_denominator_p01": denom_p01,
                "rational_denominator_condition": row.get("rational_denominator_condition", 1.0),
                "r_prime_p95": row.get("r_prime_p95", 0.0),
                "r_double_prime_p95": row.get("r_double_prime_p95", 0.0),
                "safety_pass": int(denom_min > 1.0e-4 and denom_p01 > 1.0e-4),
            }
        )
    write_csv(Path(args.out_dir) / "p3_rational_safety.csv", safety)
    return rows


def run_p4(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = _recipe_stage(
        args,
        "p4_cp_rank_speed.csv",
        "P4",
        CP_RECIPES,
        "p4_pass",
        acc_gap_budget=0.02,
        ece_gap_budget=0.03,
        mem_budget=1.2,
        step_budget=2.0,
    )
    p1 = [r for r in read_csv(Path(args.out_dir) / "p1_phase_efficiency_v3.csv") if str(r.get("method", "")).startswith("CP")]
    rank_step: Dict[int, float] = {}
    rank_mem: Dict[int, float] = {}
    for rank in [4, 8, 16]:
        rs = [r for r in p1 if _rank_from_name(str(r.get("method", ""))) == rank]
        if rs:
            rank_step[rank] = _mean(float(r.get("step_time_ratio_vs_mlp", 99) or 99) for r in rs)
            rank_mem[rank] = _mean(float(r.get("backward_memory_ratio_vs_mlp", 99) or 99) for r in rs)
    monotonic = int(rank_step.get(4, 99) < rank_step.get(16, -1)) if 4 in rank_step and 16 in rank_step else 0
    for row in rows:
        if row.get("error") or row.get("recipe") == "MLP-AdamW":
            continue
        rank = _rank_from_name(str(row.get("recipe", "")))
        row["rank"] = rank
        row["rank_speed_monotonic"] = monotonic
        row["rank_step_reference"] = rank_step.get(rank, 99.0)
        row["rank_memory_reference"] = rank_mem.get(rank, 99.0)
        row["p4_pass"] = int(int(float(row.get("p4_pass", 0) or 0)) == 1 and monotonic == 1)
    write_csv(Path(args.out_dir) / "p4_cp_rank_speed.csv", rows)
    return rows


def run_p5(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    cands = _recipe_survivors(out_dir)
    if not cands:
        rows = [{"stage": "P5", "status": "not_run", "reason": "P2/P3/P4 produced no all-dataset recipe survivor", "error": ""}]
        write_csv(out_dir / "p5_joint_task_efficiency_selection.csv", rows)
        print("P5 not run: no recipe survivor")
        return rows
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p5_joint_task_efficiency_selection.csv")
    params = V57Params(train_steps=90)
    device = get_device(args.device)
    methods = ["MLP-AdamW", "RBFOnly-Dense-reference", "ABRBF-Dense-reference"] + cands[:3]
    done = {(r.get("dataset"), int(float(r.get("seed", -1))), r.get("recipe")) for r in rows if not r.get("error")}
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        for seed in parse_int_list(args.seeds)[:3]:
            for recipe in methods:
                if (dataset, seed, recipe) in done:
                    continue
                try:
                    train_recipe = "MLP" if recipe == "MLP-AdamW" else recipe
                    row = _train_recipe(args, dataset, seed, train_recipe, params, device)
                    row.update({"stage": "P5", "recipe": recipe, "method": recipe})
                    if recipe not in {"MLP-AdamW", "RBFOnly-Dense-reference", "ABRBF-Dense-reference"}:
                        row.update(_eff_summary_for_recipe(out_dir, recipe))
                    rows.append(row)
                    print(f"P5 {dataset} seed={seed} {recipe} acc={row['test_acc']:.4f}")
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    rows.append({"stage": "P5", "dataset": dataset, "seed": seed, "recipe": recipe, "method": recipe, "error": repr(exc)})
                    print(f"P5 ERROR {dataset} {recipe}: {exc!r}")
                write_csv(out_dir / "p5_joint_task_efficiency_selection.csv", rows)
    for dataset in DATASETS:
        for seed in parse_int_list(args.seeds)[:3]:
            base = next((r for r in rows if r.get("dataset") == dataset and int(float(r.get("seed", -1))) == seed and r.get("recipe") == "MLP-AdamW" and not r.get("error")), None)
            if not base:
                continue
            for row in rows:
                if row.get("dataset") != dataset or int(float(row.get("seed", -1))) != seed or row.get("error"):
                    continue
                row["acc_gap_vs_mlp"] = float(base["test_acc"]) - float(row["test_acc"])
                row["ece_gap_vs_mlp"] = float(row["ECE"]) - float(base["ECE"])
                row["p5_pass"] = int(
                    row.get("recipe") not in {"MLP-AdamW", "RBFOnly-Dense-reference", "ABRBF-Dense-reference"}
                    and float(row["acc_gap_vs_mlp"]) <= 0.02
                    and float(row["ece_gap_vs_mlp"]) <= 0.03
                    and float(row.get("step_time_ratio", 99) or 99) <= 2.0
                    and float(row.get("backward_memory_ratio", 99) or 99) <= 1.2
                    and int(float(row.get("nonKAN_param_count", 1) or 1)) == 0
                    and float(row.get("edge_param_coverage", 0) or 0) >= 1.0
                )
    write_csv(out_dir / "p5_joint_task_efficiency_selection.csv", rows)
    return rows


def run_p6(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    survivors = _all_dataset_survivors(out_dir / "p5_joint_task_efficiency_selection.csv", ["recipe"], "p5_pass")
    if not survivors:
        return _placeholder(out_dir, "p6_custom_backward_memory.csv", "P6", "P5 produced no joint task-efficiency survivor")
    return _placeholder(out_dir, "p6_custom_backward_memory.csv", "P6", "custom backward left gated by compact runner")


def run_p7(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not _all_dataset_survivors(out_dir / "p6_custom_backward_memory.csv", ["recipe"], "p6_pass"):
        return _placeholder(out_dir, "p7_functional_lightsmooth_compat.csv", "P7", "P6 produced no memory survivor")
    return _placeholder(out_dir, "p7_functional_lightsmooth_compat.csv", "P7", "functional/LightSmooth compatibility left gated")


def run_p8(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not _all_dataset_survivors(out_dir / "p7_functional_lightsmooth_compat.csv", ["recipe"], "p7_pass"):
        return _placeholder(out_dir, "p8_candidate_selection3.csv", "P8", "P7 produced no functional/LightSmooth survivor")
    return _placeholder(out_dir, "p8_candidate_selection3.csv", "P8", "candidate selection left gated")


def run_p9(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not _all_dataset_survivors(out_dir / "p8_candidate_selection3.csv", ["recipe"], "p8_pass"):
        rows = _placeholder(out_dir, "p9_confirm5.csv", "P9", "P8 was not reached")
        write_csv(out_dir / "p10_confirm10.csv", [{"stage": "P10", "status": "not_run", "reason": "P9 was not reached", "error": ""}])
        return rows
    return _placeholder(out_dir, "p9_confirm5.csv", "P9", "confirm left gated")


def run_p10(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not (out_dir / "p10_confirm10.csv").exists():
        return _placeholder(out_dir, "p10_confirm10.csv", "P10", "P9 was not reached")
    return read_csv(out_dir / "p10_confirm10.csv")


def run_failure(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    failures: List[Dict[str, Any]] = []
    for row in read_csv(out_dir / "p0_core_profiler_consistency.csv"):
        if row.get("error") or int(float(row.get("p0_pass", 0) or 0)) != 1:
            failures.append(
                {
                    "stage": "P0",
                    "dataset": "synthetic",
                    "method": row.get("primitive_name", ""),
                    "primary_failure": "F0_core_invariant_fail",
                    "secondary_failure": "",
                    "numeric_reason": row.get("error", "p0_pass=0"),
                }
            )
    for row in read_csv(out_dir / "p1_phase_efficiency_v3.csv"):
        if row.get("error") or "MLP" in str(row.get("method", "")):
            continue
        if int(float(row.get("p1_exploratory_pass", 0) or 0)) != 1:
            primary = "F3_forward_time_fail"
            reason = f"step={row.get('step_time_ratio_vs_mlp')}, bmem={row.get('backward_memory_ratio_vs_mlp')}"
            if int(float(row.get("graph_break_count", 0) or 0)) > 0 or int(float(row.get("recompile_count", 0) or 0)) > 0:
                primary = "F2_graph_break_or_recompile"
            elif float(row.get("backward_memory_ratio_vs_mlp", 99) or 99) > 1.2:
                primary = "F5_backward_memory_fail"
            elif float(row.get("backward_time_ratio_vs_mlp", 99) or 99) > 2.0:
                primary = "F4_backward_time_fail"
            failures.append(
                {
                    "stage": "P1",
                    "dataset": "synthetic",
                    "method": row.get("method", ""),
                    "primary_failure": primary,
                    "secondary_failure": "F1_profiler_cold_warm_confounded"
                    if float(row.get("cold_warm_step_ratio", 0) or 0) > 5.0
                    else "",
                    "numeric_reason": reason,
                }
            )
    for filename, field, stage in [
        ("p2_dwm_kernel_recipe.csv", "p2_pass", "P2"),
        ("p3_rational_kernel_recipe.csv", "p3_pass", "P3"),
        ("p4_cp_rank_speed.csv", "p4_pass", "P4"),
        ("p5_joint_task_efficiency_selection.csv", "p5_pass", "P5"),
    ]:
        for row in read_csv(out_dir / filename):
            if row.get("error") or row.get("status") == "not_run" or row.get("recipe") == "MLP-AdamW":
                continue
            if int(float(row.get(field, 0) or 0)) != 1:
                primary = "F6_accuracy_recipe_fail"
                if float(row.get("ece_gap_vs_mlp", 0) or 0) > 0.03:
                    primary = "F7_ece_or_calibration_fail"
                if float(row.get("backward_memory_ratio", 0) or 0) > 1.2:
                    primary = "F5_backward_memory_fail"
                failures.append(
                    {
                        "stage": stage,
                        "dataset": row.get("dataset", "all"),
                        "method": row.get("recipe", row.get("method", "")),
                        "primary_failure": primary,
                        "secondary_failure": "F11_no_joint_survivor" if stage == "P5" else "",
                        "numeric_reason": f"gap={row.get('acc_gap_vs_mlp')}, ece_gap={row.get('ece_gap_vs_mlp')}, step={row.get('step_time_ratio')}, bmem={row.get('backward_memory_ratio')}",
                    }
                )
    if not failures:
        failures.append(
            {
                "stage": "P10",
                "dataset": "all",
                "method": "all",
                "primary_failure": "no_failure_rows",
                "secondary_failure": "",
                "numeric_reason": "",
            }
        )
    write_csv(out_dir / "failure_table.csv", failures)
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = add_v3_args()
    parser.description = __doc__
    parser.set_defaults(packages="V5_7_P0", out_dir=Path("results/v5_7"), datasets="MNIST,Fashion-MNIST,KMNIST", seeds="0,1,2")
    parser.add_argument("--no-compile-v57", action="store_true", help="Disable torch.compile in v5.7 profiler paths.")
    parser.add_argument("--v57-bench-warmup", type=int, default=8, help="Warm steady-state profiler steps after cold step.")
    parser.add_argument("--v57-bench-reps", type=int, default=10, help="Measured steady-state profiler repetitions.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.out_dir)
    for pkg in parse_str_list(args.packages):
        key = pkg.upper()
        if key in {"V5_7_P0", "V5_7_P0_CORE"}:
            run_p0(args)
        elif key in {"V5_7_P1", "V5_7_P1_EFFICIENCY"}:
            run_p1(args)
        elif key in {"V5_7_P2", "V5_7_P2_DWM"}:
            run_p2(args)
        elif key in {"V5_7_P3", "V5_7_P3_RK"}:
            run_p3(args)
        elif key in {"V5_7_P4", "V5_7_P4_CP"}:
            run_p4(args)
        elif key in {"V5_7_P5", "V5_7_P5_SELECTION"}:
            run_p5(args)
        elif key in {"V5_7_P6", "V5_7_P6_MEMORY"}:
            run_p6(args)
        elif key in {"V5_7_P7", "V5_7_P7_FUNCTIONAL"}:
            run_p7(args)
        elif key in {"V5_7_P8", "V5_7_P8_SELECTION3"}:
            run_p8(args)
        elif key in {"V5_7_P9", "V5_7_P9_CONFIRM"}:
            run_p9(args)
        elif key in {"V5_7_P10", "V5_7_P10_FINAL", "V5_7_FAILURE"}:
            run_p10(args)
            run_failure(args)
        elif key in {"V5_7_ALL", "ALL"}:
            run_p0(args)
            run_p1(args)
            run_p2(args)
            run_p3(args)
            run_p4(args)
            run_p5(args)
            run_p6(args)
            run_p7(args)
            run_p8(args)
            run_p9(args)
            run_p10(args)
            run_failure(args)
        else:
            raise ValueError(f"unknown package: {pkg}")


if __name__ == "__main__":
    main()
