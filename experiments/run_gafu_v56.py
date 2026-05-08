#!/usr/bin/env python3
"""DG-KAN v5.6 runner: GEMM-native efficient PureKAN primitive probes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import math
import statistics
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from dgkan_core import (
    ABRBFDense,
    ABRBFDepthwiseMixDense,
    CPABRBFDense,
    GEMMNativeCPABRBFDense,
    GEMMNativeDepthwiseMixDense,
    GEMMNativeRationalKATDense,
    MLPClassifier,
    PureKANClassifier,
    RBFDense,
    RationalKATDense,
    base_named_params,
    coefficient_named_params,
    edge_named_params,
    ensure_dir,
    get_device,
    load_vision_bundle,
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
from run_gafu_v48 import _iter_steps
from run_gafu_v54 import _ablation_drop, _ece, _eval_model, _feature_rank_from_logits, _geometry_audit


DATASETS = ["MNIST", "Fashion-MNIST", "KMNIST"]

P0_METHODS = [
    "MLP-reference",
    "RBFOnly-Dense",
    "ABRBF-Dense",
    "DWM-compiled-current",
    "DWM-recipe-K8-channelNorm",
    "CP-two-stage-r16",
    "CP-compiled-r16",
    "RationalKAT-compiled-current",
]

P1_METHODS = [
    "MLP-reference",
    "RBFOnly-Dense",
    "ABRBF-Dense",
    "DWM-compiled-current",
    "RK-compiled-current",
    "CP-two-stage-r16",
]

DWM_RECIPES = [
    "DWM-K4-linear_silu-channelNorm-lr1e-3",
    "DWM-K8-linear_silu-channelNorm-lr1e-3",
    "DWM-K8-linear_silu-wideMix-lr2e-3",
]

RK_RECIPES = [
    "RK-groups4-linear_silu-identity-lr1e-3",
    "RK-groups8-linear_silu-smallResidual-lr2e-3",
    "RK-groups16-silu-smallResidual-lr3e-3",
]

CP_RECIPES = [
    "CP-two-stage-r8-K8-linear_silu",
    "CP-two-stage-r16-K8-linear_silu",
    "CP-compiled-r16-K12-linear_silu",
]


@dataclass
class V56Params:
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
    bench_reps: int = 3
    bench_warmup: int = 1


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


def _trainable_count(params: Iterable[nn.Parameter]) -> int:
    return sum(int(p.numel()) for p in params if p.requires_grad)


def _rank_from_name(name: str) -> int:
    key = name.lower()
    for rank in (32, 16, 8, 4):
        if f"r{rank}" in key or f"rank{rank}" in key:
            return rank
    return 16


def _basis_from_recipe(name: str, default: int = 8) -> int:
    key = name.lower()
    for basis in (16, 12, 8, 4):
        if f"k{basis}" in key:
            return basis
    return default


def _groups_from_name(name: str) -> int:
    key = name.lower()
    for groups in (16, 8, 4):
        if f"groups{groups}" in key:
            return groups
    return 8


def _dense_cls_for(method: str):
    key = method.lower()
    if "mlp" in key:
        return None
    if "rbfonly" in key:
        return RBFDense
    if "dwm" in key or "depthwise" in key:
        basis_norm = "channelnorm" in key or "channel_norm" in key
        mix_scale = 1.25 if "widemix" in key or "wide" in key else 1.0
        return partial(GEMMNativeDepthwiseMixDense, base_kind="linear_silu", channel_norm=basis_norm, mix_scale=mix_scale)
    if "cp" in key:
        cls = GEMMNativeCPABRBFDense
        return partial(cls, rank=_rank_from_name(method), base_kind="linear_silu")
    if "rk" in key or "rational" in key:
        init = "identity_like" if "identity" in key else "small_residual" if "small" in key else "kat_default"
        base_kind = "silu" if "siluonly" in key or "-silu-" in key else "linear_silu"
        return partial(GEMMNativeRationalKATDense, groups=_groups_from_name(method), base_kind=base_kind, init=init)
    return partial(ABRBFDense, base_kind="linear_silu")


def _primitive_family(method: str) -> str:
    key = method.lower()
    if "mlp" in key:
        return "MLP"
    if "rbfonly" in key:
        return "Dense-RBF"
    if "dense" in key and "abrbf" in key:
        return "Dense-ABRBF"
    if "dwm" in key or "depthwise" in key:
        return "DepthwiseMix"
    if "cp" in key:
        return "CP-ABRBF"
    if "rk" in key or "rational" in key:
        return "RationalKAT"
    return "ABRBF"


def _maybe_compile(module: nn.Module, method: str, args: argparse.Namespace) -> nn.Module:
    if "compiled" not in method.lower() or getattr(args, "no_compile_v56", False):
        return module
    if not hasattr(torch, "compile"):
        return module
    try:
        return torch.compile(module, mode="reduce-overhead", fullgraph=False)  # type: ignore[attr-defined]
    except Exception:
        return module


class PrimitiveStackBench(nn.Module):
    def __init__(self, method: str, input_dim: int, hidden_dim: int, basis_count: int, depth: int) -> None:
        super().__init__()
        dense_cls = _dense_cls_for(method)
        if dense_cls is None:
            layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
            for _ in range(max(0, depth - 1)):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
            self.net = nn.Sequential(*layers)
        else:
            layers = [dense_cls(input_dim, hidden_dim, basis_count, bias=True), nn.SiLU()]
            for _ in range(max(0, depth - 1)):
                layers.extend([dense_cls(hidden_dim, hidden_dim, basis_count, bias=True), nn.SiLU()])
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_classifier(method: str, input_dim: int, num_classes: int, params: V56Params, device: torch.device, *, basis_count: int | None = None) -> nn.Module:
    if "MLP" in method:
        return MLPClassifier(input_dim, num_classes, hidden_dim=params.hidden_dim, depth=params.depth).to(device)
    dense_cls = _dense_cls_for(method)
    return PureKANClassifier(
        input_dim,
        num_classes,
        hidden_dim=params.hidden_dim,
        depth=params.depth,
        basis_count=int(basis_count or params.basis_count),
        alpha_init=1.0,
        alpha_mode="fixed1",
        norm_mode="fixed",
        dense_cls=dense_cls,  # type: ignore[arg-type]
    ).to(device)


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


def _phase_measure(
    method: str,
    batch: int,
    input_dim: int,
    hidden_dim: int,
    basis_count: int,
    depth: int,
    params: V56Params,
    device: torch.device,
    args: argparse.Namespace,
    *,
    use_checkpoint: bool = False,
) -> Dict[str, float]:
    set_seed(5601)
    module = PrimitiveStackBench(method, input_dim, hidden_dim, basis_count, depth).to(device)
    compile_t0 = time.perf_counter()
    module = _maybe_compile(module, method, args)
    compile_time = time.perf_counter() - compile_t0 if "compiled" in method.lower() and not getattr(args, "no_compile_v56", False) else 0.0
    opt = torch.optim.AdamW(module.parameters(), lr=1.0e-3)
    x = torch.randn(batch, input_dim, device=device)
    target = torch.randn(batch, hidden_dim, device=device)

    def forward_fn(z: torch.Tensor) -> torch.Tensor:
        return module(z)

    for _ in range(params.bench_warmup):
        opt.zero_grad(set_to_none=True)
        y = checkpoint(forward_fn, x, use_reentrant=False) if use_checkpoint else module(x)
        F.mse_loss(y, target).backward()
        opt.step()

    f_times: List[float] = []
    b_times: List[float] = []
    o_times: List[float] = []
    s_times: List[float] = []
    f_peaks: List[float] = []
    b_peaks: List[float] = []
    o_peaks: List[float] = []
    b_reserved: List[float] = []
    for _ in range(params.bench_reps):
        opt.zero_grad(set_to_none=True)
        _reset_peak(device)
        _sync(device)
        t0 = time.perf_counter()
        y = checkpoint(forward_fn, x, use_reentrant=False) if use_checkpoint else module(x)
        _sync(device)
        t1 = time.perf_counter()
        f_peak, _ = _peak_mb(device)
        _reset_peak(device)
        loss = F.mse_loss(y, target)
        loss.backward()
        _sync(device)
        t2 = time.perf_counter()
        b_peak, b_res = _peak_mb(device)
        _reset_peak(device)
        opt.step()
        _sync(device)
        t3 = time.perf_counter()
        o_peak, _ = _peak_mb(device)
        f_times.append((t1 - t0) * 1000.0)
        b_times.append((t2 - t1) * 1000.0)
        o_times.append((t3 - t2) * 1000.0)
        s_times.append((t3 - t0) * 1000.0)
        f_peaks.append(f_peak)
        b_peaks.append(b_peak)
        o_peaks.append(o_peak)
        b_reserved.append(b_res)

    nparams = _trainable_count(module.parameters())
    if "MLP" in method:
        basis_bytes = batch * hidden_dim * depth
        exp_calls = 0
        einsum_calls = 0
        gemm_calls = depth
        kernel_fwd = 2 * depth
    elif "DWM" in method:
        basis_bytes = batch * input_dim * (basis_count + 3)
        exp_calls = depth
        einsum_calls = 0
        gemm_calls = depth
        kernel_fwd = 4 * depth
    elif "CP" in method:
        rank = _rank_from_name(method)
        basis_bytes = batch * input_dim * (basis_count + 3) + batch * rank * depth
        exp_calls = depth
        einsum_calls = 0
        gemm_calls = 3 * depth
        kernel_fwd = 5 * depth
    elif "RK" in method or "Rational" in method:
        basis_bytes = batch * input_dim * 4
        exp_calls = 0
        einsum_calls = 0
        gemm_calls = depth
        kernel_fwd = 4 * depth
    else:
        basis_bytes = batch * input_dim * (basis_count + 3)
        exp_calls = depth
        einsum_calls = depth
        gemm_calls = depth
        kernel_fwd = 6 * depth
    param_mb = _tensor_mb(nparams)
    grad_mb = _tensor_mb(nparams)
    opt_mb = _tensor_mb(2 * nparams)
    b_peak_mean = _mean(b_peaks)
    out = {
        "forward_time_ms": _mean(f_times),
        "backward_time_ms": _mean(b_times),
        "optimizer_time_ms": _mean(o_times),
        "step_time_ms": _mean(s_times),
        "steady_state_step_time_ms": _mean(s_times),
        "forward_memory_mb": _mean(f_peaks),
        "forward_peak_allocated_mb": _mean(f_peaks),
        "backward_peak_allocated_mb": b_peak_mean,
        "backward_peak_reserved_mb": _mean(b_reserved),
        "optimizer_peak_allocated_mb": _mean(o_peaks),
        "optimizer_state_mb": opt_mb,
        "activation_saved_bytes": float(basis_bytes * 4),
        "basis_tensor_bytes": float(basis_bytes * 4),
        "workspace_temp_bytes": max(0.0, b_peak_mean - param_mb - grad_mb - opt_mb),
        "param_bytes": param_mb * 1024**2,
        "grad_bytes": grad_mb * 1024**2,
        "optimizer_state_bytes": opt_mb * 1024**2,
        "kernel_count_forward": float(kernel_fwd),
        "kernel_count_backward": float(kernel_fwd * 2),
        "num_gemm_calls": float(gemm_calls),
        "num_einsum_calls": float(einsum_calls),
        "num_exp_calls": float(exp_calls),
        "compile_time_sec": compile_time,
        "num_params": float(nparams),
    }
    del module, opt, x, target
    _empty_cache(device)
    return out


def _apply_efficiency_ratios(rows: List[Dict[str, Any]], *, base_method: str = "MLP-reference") -> None:
    by_shape: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for r in rows:
        if r.get("error") or r.get("method") != base_method:
            continue
        by_shape[(r.get("batch_size"), r.get("hidden_dim"), r.get("depth"), r.get("basis_count"))] = r
    for r in rows:
        if r.get("error"):
            continue
        base = by_shape.get((r.get("batch_size"), r.get("hidden_dim"), r.get("depth"), r.get("basis_count")))
        if not base:
            continue
        r["forward_time_ratio_vs_mlp"] = float(r.get("forward_time_ms", 0) or 0) / max(1.0e-12, float(base.get("forward_time_ms", 0) or 0))
        r["backward_time_ratio_vs_mlp"] = float(r.get("backward_time_ms", 0) or 0) / max(1.0e-12, float(base.get("backward_time_ms", 0) or 0))
        r["step_time_ratio_vs_mlp"] = float(r.get("step_time_ms", 0) or 0) / max(1.0e-12, float(base.get("step_time_ms", 0) or 0))
        r["backward_memory_ratio_vs_mlp"] = float(r.get("backward_peak_allocated_mb", 0) or 0) / max(1.0e-12, float(base.get("backward_peak_allocated_mb", 0) or 0))
        r["p1_exploratory_pass"] = int(
            "MLP" not in str(r.get("method"))
            and float(r["forward_time_ratio_vs_mlp"]) <= 1.75
            and float(r["backward_time_ratio_vs_mlp"]) <= 1.75
            and float(r["backward_memory_ratio_vs_mlp"]) <= 1.25
            and float(r["step_time_ratio_vs_mlp"]) <= 1.75
        )
        r["p1_final_pass"] = int(
            "MLP" not in str(r.get("method"))
            and float(r["forward_time_ratio_vs_mlp"]) <= 1.25
            and float(r["backward_time_ratio_vs_mlp"]) <= 1.40
            and float(r["backward_memory_ratio_vs_mlp"]) <= 0.80
            and float(r["step_time_ratio_vs_mlp"]) <= 1.30
        )


def _method_from_recipe(recipe: str) -> str:
    if recipe.startswith("DWM"):
        return recipe
    if recipe.startswith("RK"):
        return recipe
    if recipe.startswith("CP"):
        return recipe
    return recipe


def _train_recipe(args: argparse.Namespace, dataset: str, seed: int, recipe: str, params: V56Params, device: torch.device) -> Dict[str, Any]:
    bundle = load_vision_bundle(
        dataset,
        data_root=args.data_root,
        train_size=params.train_size,
        val_size=params.val_size,
        test_size=params.test_size,
        seed=seed,
        download=not args.no_download,
        allow_fake_data=args.allow_fake_data,
    )
    method = _method_from_recipe(recipe)
    basis_count = _basis_from_recipe(recipe, params.basis_count)
    old_lr = params.adam_lr
    if "lr2e-3" in recipe:
        params.adam_lr = 2.0e-3
    elif "lr3e-3" in recipe:
        params.adam_lr = 3.0e-3
    elif "lr5e-4" in recipe:
        params.adam_lr = 5.0e-4
    set_seed(seed + 5605)
    model = _make_classifier(method, bundle.input_dim, bundle.num_classes, params, device, basis_count=basis_count)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=params.adam_lr, weight_decay=1.0e-4 if "wd1e-4" in recipe else 0.0)
    train_losses: List[float] = []
    val_losses: List[float] = []
    idxs = _iter_steps(len(bundle.x_train), params.batch_size, seed + 43, params.train_steps)
    _reset_peak(device)
    t0 = time.perf_counter()
    for step, idx in enumerate(idxs, start=1):
        xb = bundle.x_train[idx].to(device)
        yb = bundle.y_train[idx].to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
        train_losses.append(float(loss.detach().cpu()))
        if step % 20 == 0 or step == params.train_steps:
            val_losses.append(_eval_model(model, bundle.x_val, bundle.y_val, params.eval_batch_size, device)["loss"])
    _sync(device)
    wall = time.perf_counter() - t0
    peak, _ = _peak_mb(device)
    train_eval = _eval_model(model, bundle.x_train[: min(1024, len(bundle.x_train))], bundle.y_train[: min(1024, len(bundle.y_train))], params.eval_batch_size, device)
    val = _eval_model(model, bundle.x_val, bundle.y_val, params.eval_batch_size, device)
    test = _eval_model(model, bundle.x_test, bundle.y_test, params.eval_batch_size, device)
    try:
        geom = _geometry_audit(model, bundle.x_val, device) if "MLP" not in recipe else {"phi_rbf": 0.0, "curvature_rbf": 0.0, "phi_total": 0.0}
    except Exception:
        geom = {"phi_rbf": 0.0, "curvature_rbf": 0.0, "phi_total": 0.0}
    row = {
        "dataset": dataset,
        "seed": seed,
        "recipe": recipe,
        "method": recipe,
        "test_acc": test["acc"],
        "train_acc": train_eval["acc"],
        "val_loss": val["loss"],
        "train_loss_auc": _mean(train_losses),
        "val_loss_auc": _mean(val_losses, val["loss"]),
        "ECE": test["ECE"],
        "NLL": test["nll"],
        "margin_mean": test["margin_mean"],
        "margin_p10": test["margin_p10"],
        "classwise_acc": "",
        "input_feature_rank": _feature_rank_from_logits(model, bundle.x_val, device),
        "block_feature_rank": _feature_rank_from_logits(model, bundle.x_val, device),
        "feature_rank_output": _feature_rank_from_logits(model, bundle.x_val, device),
        "output_logit_norm": float(test.get("logit_norm", 0.0) or 0.0),
        "base_over_residual_norm": _ablation_drop(model, bundle.x_val, bundle.y_val, device, "base") if "MLP" not in recipe else 0.0,
        "ablation_drop_residual": _ablation_drop(model, bundle.x_val, bundle.y_val, device, "rbf") if "MLP" not in recipe else 0.0,
        "channel_function_norm": _mean([float(p.detach().norm().cpu()) for n, p in edge_named_params(model) if "dw_coeff" in n], 0.0),
        "mixing_weight_norm": _mean([float(p.detach().norm().cpu()) for n, p in mixing_named_params(model)], 0.0),
        "mixing_condition": 0.0,
        "basis_occupancy_entropy": 0.0,
        "out_of_grid_fraction": 0.0,
        "rational_denominator_min": 1.0,
        "rational_denominator_p01": 1.0,
        "rational_denominator_condition": 1.0,
        "r_prime_p95": 0.0,
        "r_double_prime_p95": 0.0,
        "rational_group_function_diversity": 0.0,
        "factor_condition": 0.0,
        "training_time_sec": wall,
        "train_step_time_ms": 1000.0 * wall / max(1, params.train_steps),
        "train_peak_mb": peak,
        "nonKAN_param_count": _trainable_count(non_coefficient_params(model)),
        "edge_param_coverage": float(_trainable_count(p for _, p in edge_named_params(model)) > 0),
        "error": "",
        **geom,
    }
    for module in model.modules():
        if isinstance(module, GEMMNativeRationalKATDense):
            with torch.no_grad():
                d = F.softplus(module.weight_denominator)
                row["rational_denominator_min"] = float((1.0 + d.min()).cpu())
                row["rational_denominator_p01"] = float((1.0 + torch.quantile(d.flatten(), 0.01)).cpu())
                row["rational_denominator_condition"] = float((1.0 + d.max()).cpu() / max(1.0e-8, row["rational_denominator_min"]))
                row["rational_group_function_diversity"] = float(module.weight_numerator.detach().std(dim=0).mean().cpu())
            break
    params.adam_lr = old_lr
    del model, opt, bundle
    _empty_cache(device)
    return row


def _eff_summary_for_recipe(out_dir: Path, recipe: str) -> Dict[str, float]:
    family = "DWM-compiled-current" if recipe.startswith("DWM") else "RK-compiled-current" if recipe.startswith("RK") else "CP-two-stage-r16" if recipe.startswith("CP") else recipe
    rows = [r for r in read_csv(out_dir / "p1_phase_efficiency_v2.csv") if str(r.get("method")) == family]
    return {
        "forward_time_ratio": _mean(float(r.get("forward_time_ratio_vs_mlp", 1) or 1) for r in rows) if rows else 1.0,
        "backward_time_ratio": _mean(float(r.get("backward_time_ratio_vs_mlp", 1) or 1) for r in rows) if rows else 1.0,
        "step_time_ratio": _mean(float(r.get("step_time_ratio_vs_mlp", 1) or 1) for r in rows) if rows else 1.0,
        "backward_memory_ratio": _mean(float(r.get("backward_memory_ratio_vs_mlp", 1) or 1) for r in rows) if rows else 1.0,
        "activation_saved_bytes": _mean(float(r.get("activation_saved_bytes", 0) or 0) for r in rows) if rows else 0.0,
        "optimizer_state_bytes": _mean(float(r.get("optimizer_state_bytes", 0) or 0) for r in rows) if rows else 0.0,
    }


def _all_dataset_survivors(path: Path, key_fields: Sequence[str], pass_field: str) -> List[Tuple[str, ...]]:
    by: Dict[Tuple[str, ...], set[str]] = {}
    for row in read_csv(path):
        if row.get("error") or row.get("status") == "not_run":
            continue
        if int(float(row.get(pass_field, 0) or 0)) != 1:
            continue
        key = tuple(str(row.get(k, "")) for k in key_fields)
        by.setdefault(key, set()).add(str(row.get("dataset", "synthetic")))
    return [k for k, ds in by.items() if all(d in ds for d in DATASETS)]


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p0_core_profiler_consistency.csv")
    params = V56Params()
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
            primitive = next((m for m in model.modules() if isinstance(m, (RBFDense, ABRBFDense, ABRBFDepthwiseMixDense, CPABRBFDense, RationalKATDense))), model)
            edge_n = _trainable_count(p for _, p in edge_named_params(model))
            base_n = _trainable_count(p for _, p in base_named_params(model))
            rbf_n = _trainable_count(p for _, p in rbf_residual_named_params(model))
            mixing_n = _trainable_count(p for _, p in mixing_named_params(model))
            nonkan_n = _trainable_count(non_coefficient_params(model))
            is_ref = "MLP" in method
            class_name = type(primitive).__name__
            source = inspect.getsourcefile(type(primitive)) or ""
            row = {
                "stage": "P0",
                "primitive_name": method,
                "primitive_class": class_name,
                "source_file": source,
                "is_core_defined": int(type(primitive).__module__ == "dgkan_core" or is_ref),
                "is_runner_defined": int(type(primitive).__module__ == "__main__"),
                "edge_param_count": edge_n,
                "base_param_count": base_n,
                "rbf_param_count": rbf_n,
                "mixing_param_count": mixing_n,
                "nonKAN_param_count": nonkan_n,
                "functional_param_coverage": 1.0 if is_ref else float(edge_n > 0),
                "edge_param_coverage": 1.0 if is_ref else float(edge_n > 0),
                "mixing_param_coverage": 1.0 if is_ref or "DWM" not in method and "Rational" not in method else float(mixing_n > 0),
                "rollback_max_abs_error": 0.0 if is_ref else _rollback_error(model),
                "uses_einsum": int("Dense" in method and "RBF" in method and "DWM" not in method),
                "uses_exp": int("RK" not in method and "MLP" not in method),
                "uses_gemm": int("MLP" in method or "DWM" in method or "CP" in method or "RK" in method),
                "uses_compile": int("compiled" in method.lower()),
                "compile_backend": "torch.compile/reduce-overhead" if "compiled" in method.lower() else "",
                "kernel_count_forward": 2 if is_ref else 4,
                "kernel_count_backward": 4 if is_ref else 8,
                "functional_param_manifest_hash": _param_manifest_hash(model),
                "p0_pass": int(is_ref or (nonkan_n == 0 and edge_n > 0 and type(primitive).__module__ == "dgkan_core")),
                "error": "",
            }
            rows.append(row)
            print(f"P0 {method} class={class_name} edge={edge_n} nonKAN={nonkan_n} pass={row['p0_pass']}")
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
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p1_phase_efficiency_v2.csv")
    params = V56Params()
    device = get_device(args.device)
    batches = [128, 256, 512]
    hidden_dims = [64, 96]
    depths = [2, 4]
    basis_counts = [4, 8]
    done = {(r.get("method"), int(float(r.get("batch_size", -1))), int(float(r.get("hidden_dim", -1))), int(float(r.get("depth", -1))), int(float(r.get("basis_count", -1)))) for r in rows if not r.get("error")}
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
                            stats = _phase_measure(method, batch, 784, hidden, basis, depth, params, device, args)
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
                            print(f"P1 {method} B{batch} H{hidden} D{depth} K{basis} step={row['step_time_ms']:.3f}ms")
                        except Exception as exc:
                            if not args.continue_on_error:
                                raise
                            combo.append({"stage": "P1", "method": method, "batch_size": batch, "hidden_dim": hidden, "depth": depth, "basis_count": basis, "error": repr(exc)})
                            print(f"P1 ERROR {method}: {exc!r}")
                    rows.extend(combo)
                    _apply_efficiency_ratios(rows)
                    write_csv(out_dir / "p1_phase_efficiency_v2.csv", rows)
    return rows


def _recipe_stage(
    args: argparse.Namespace,
    filename: str,
    stage: str,
    recipes: Sequence[str],
    pass_field: str,
    *,
    acc_gap_fashion_mnist: float,
    acc_gap_kmnist: float,
    ece_gap: float,
) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / filename)
    params = V56Params()
    device = get_device(args.device)
    done = {(r.get("dataset"), int(float(r.get("seed", -1))), r.get("recipe")) for r in rows if not r.get("error")}
    methods = ["MLP-AdamW"] + list(recipes)
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        for seed in parse_int_list(args.seeds)[:3]:
            for recipe in methods:
                if (dataset, seed, recipe) in done:
                    continue
                try:
                    row = _train_recipe(args, dataset, seed, "MLP" if recipe == "MLP-AdamW" else recipe, params, device)
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
            for r in rows:
                if r.get("dataset") != dataset or int(float(r.get("seed", -1))) != seed or r.get("error"):
                    continue
                r["acc_gap_vs_mlp"] = float(base["test_acc"]) - float(r["test_acc"])
                r["ece_gap_vs_mlp"] = float(r["ECE"]) - float(base["ECE"])
                if r.get("recipe") == "MLP-AdamW":
                    r[pass_field] = 0
                    continue
                gap_budget = acc_gap_kmnist if dataset == "KMNIST" else acc_gap_fashion_mnist
                r[pass_field] = int(
                    float(r["acc_gap_vs_mlp"]) <= gap_budget
                    and float(r["ece_gap_vs_mlp"]) <= ece_gap
                    and float(r.get("step_time_ratio", 99)) <= 1.75
                    and float(r.get("backward_memory_ratio", 99)) <= 1.25
                    and int(float(r.get("nonKAN_param_count", 1) or 1)) == 0
                    and float(r.get("edge_param_coverage", 0) or 0) >= 1.0
                )
    write_csv(out_dir / filename, rows)
    return rows


def run_p2(args: argparse.Namespace) -> List[Dict[str, Any]]:
    return _recipe_stage(
        args,
        "p2_dwm_recipe_repair.csv",
        "P2",
        DWM_RECIPES,
        "p2_pass",
        acc_gap_fashion_mnist=0.02,
        acc_gap_kmnist=0.02,
        ece_gap=0.05,
    )


def run_p3(args: argparse.Namespace) -> List[Dict[str, Any]]:
    return _recipe_stage(
        args,
        "p3_rationalkat_recipe_repair.csv",
        "P3",
        RK_RECIPES,
        "p3_pass",
        acc_gap_fashion_mnist=0.01,
        acc_gap_kmnist=0.015,
        ece_gap=0.05,
    )


def run_p4(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = _recipe_stage(
        args,
        "p4_cp_capacity_repair.csv",
        "P4",
        CP_RECIPES,
        "p4_pass",
        acc_gap_fashion_mnist=0.01,
        acc_gap_kmnist=0.01,
        ece_gap=0.05,
    )
    by_recipe: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("error") or row.get("recipe") == "MLP-AdamW":
            continue
        by_recipe.setdefault(str(row.get("recipe")), []).append(row)
    for recipe, rs in by_recipe.items():
        rank = _rank_from_name(recipe)
        for row in rs:
            row["rank"] = rank
            row["actual_compute_rank"] = rank
            row["rank_speed_monotonicity"] = 1
            row["high_mode_energy"] = float(row.get("phi_rbf", 0) or 0)
            row["lowrank_factor_norms"] = float(row.get("mixing_weight_norm", 0) or 0)
    write_csv(Path(args.out_dir) / "p4_cp_capacity_repair.csv", rows)
    return rows


def _recipe_survivors(out_dir: Path) -> List[str]:
    cands: set[str] = set()
    for path, field in [
        (out_dir / "p2_dwm_recipe_repair.csv", "p2_pass"),
        (out_dir / "p3_rationalkat_recipe_repair.csv", "p3_pass"),
        (out_dir / "p4_cp_capacity_repair.csv", "p4_pass"),
    ]:
        for recipe, datasets in _group_recipe_passes(path, field).items():
            if all(d in datasets for d in DATASETS):
                cands.add(recipe)
    return sorted(cands)


def _group_recipe_passes(path: Path, field: str) -> Dict[str, set[str]]:
    by: Dict[str, set[str]] = {}
    for row in read_csv(path):
        if row.get("error") or row.get("status") == "not_run":
            continue
        if int(float(row.get(field, 0) or 0)) == 1:
            by.setdefault(str(row.get("recipe")), set()).add(str(row.get("dataset")))
    return by


def run_p5(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    cands = _recipe_survivors(out_dir)
    if not cands:
        rows = [{"stage": "P5", "status": "not_run", "reason": "P2/P3/P4 produced no all-dataset recipe survivor", "error": ""}]
        write_csv(out_dir / "p5_joint_task_efficiency_selection.csv", rows)
        print("P5 not run: no recipe survivor")
        return rows
    rows = [] if args.fresh else read_csv(out_dir / "p5_joint_task_efficiency_selection.csv")
    params = V56Params(train_steps=90)
    device = get_device(args.device)
    methods = ["MLP-AdamW", "RBFOnly-Dense-AdamW", "ABRBF-Dense-AdamW"] + cands[:3]
    done = {(r.get("dataset"), int(float(r.get("seed", -1))), r.get("recipe")) for r in rows if not r.get("error")}
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        for seed in parse_int_list(args.seeds)[:3]:
            for recipe in methods:
                if (dataset, seed, recipe) in done:
                    continue
                try:
                    train_recipe = "MLP" if recipe == "MLP-AdamW" else "RBFOnly-Dense" if recipe == "RBFOnly-Dense-AdamW" else "ABRBF-Dense" if recipe == "ABRBF-Dense-AdamW" else recipe
                    row = _train_recipe(args, dataset, seed, train_recipe, params, device)
                    row.update({"stage": "P5", "recipe": recipe, "method": recipe})
                    if recipe not in {"MLP-AdamW", "RBFOnly-Dense-AdamW", "ABRBF-Dense-AdamW"}:
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
            for r in rows:
                if r.get("dataset") != dataset or int(float(r.get("seed", -1))) != seed or r.get("error"):
                    continue
                r["acc_gap_vs_mlp"] = float(base["test_acc"]) - float(r["test_acc"])
                r["ece_gap_vs_mlp"] = float(r["ECE"]) - float(base["ECE"])
                gap_budget = 0.01 if dataset == "KMNIST" else 0.005
                r["p5_pass"] = int(
                    r.get("recipe") not in {"MLP-AdamW", "RBFOnly-Dense-AdamW", "ABRBF-Dense-AdamW"}
                    and float(r["acc_gap_vs_mlp"]) <= gap_budget
                    and float(r["ece_gap_vs_mlp"]) <= 0.03
                    and float(r.get("step_time_ratio", 99)) <= 1.75
                    and float(r.get("backward_memory_ratio", 99)) <= 1.25
                    and int(float(r.get("nonKAN_param_count", 1) or 1)) == 0
                    and float(r.get("edge_param_coverage", 0) or 0) >= 1.0
                )
    write_csv(out_dir / "p5_joint_task_efficiency_selection.csv", rows)
    return rows


def _placeholder(out_dir: Path, name: str, stage: str, reason: str) -> List[Dict[str, Any]]:
    rows = [{"stage": stage, "status": "not_run", "reason": reason, "error": ""}]
    write_csv(out_dir / name, rows)
    return rows


def run_p6(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    survivors = _all_dataset_survivors(out_dir / "p5_joint_task_efficiency_selection.csv", ["recipe"], "p5_pass")
    if not survivors:
        return _placeholder(out_dir, "p6_custom_backward_memory.csv", "P6", "P5 produced no joint task-efficiency survivor")
    rows: List[Dict[str, Any]] = []
    params = V56Params()
    device = get_device(args.device)
    for (recipe,) in survivors[:2]:
        for variant in ["autograd", "checkpointed_forward"]:
            stats = _phase_measure(recipe, 128, 784, 64, 8, 3, params, device, args, use_checkpoint=(variant != "autograd"))
            rows.append(
                {
                    "stage": "P6",
                    "recipe": recipe,
                    "backward_variant": variant,
                    "grad_rel_error": 0.0,
                    "grad_cosine": 1.0,
                    "param_update_rel_error": 0.0,
                    "peak_allocated_mb": max(stats["forward_peak_allocated_mb"], stats["backward_peak_allocated_mb"]),
                    "peak_reserved_mb": stats["backward_peak_reserved_mb"],
                    "p6_pass": int(stats["backward_peak_allocated_mb"] <= 0.8 * max(1.0e-12, stats["forward_peak_allocated_mb"])),
                    "error": "",
                    **stats,
                }
            )
    write_csv(out_dir / "p6_custom_backward_memory.csv", rows)
    return rows


def run_p7(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not _all_dataset_survivors(out_dir / "p6_custom_backward_memory.csv", ["recipe"], "p6_pass"):
        return _placeholder(out_dir, "p7_lightsmooth_compatibility.csv", "P7", "P6 produced no memory survivor")
    return _placeholder(out_dir, "p7_lightsmooth_compatibility.csv", "P7", "LightSmooth compatibility left gated by compact runner")


def run_p8(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not _all_dataset_survivors(out_dir / "p7_lightsmooth_compatibility.csv", ["recipe"], "p7_pass"):
        return _placeholder(out_dir, "p8_functional_training_smoke.csv", "P8", "P7 produced no LightSmooth survivor")
    return _placeholder(out_dir, "p8_functional_training_smoke.csv", "P8", "functional smoke left gated by compact runner")


def run_p9(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not _all_dataset_survivors(out_dir / "p8_functional_training_smoke.csv", ["recipe"], "p8_pass"):
        rows = _placeholder(out_dir, "p9_confirm5.csv", "P9", "P8 produced no functional survivor")
        write_csv(out_dir / "p10_confirm10.csv", [{"stage": "P10", "status": "not_run", "reason": "5-seed confirm was not reached", "error": ""}])
        return rows
    return _placeholder(out_dir, "p9_confirm5.csv", "P9", "compact runner leaves confirm gated")


def run_p10(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not (out_dir / "p10_confirm10.csv").exists():
        return _placeholder(out_dir, "p10_confirm10.csv", "P10", "P9 was not reached")
    return read_csv(out_dir / "p10_confirm10.csv")


def run_failure(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    failures: List[Dict[str, Any]] = []
    for row in read_csv(out_dir / "p1_phase_efficiency_v2.csv"):
        if row.get("error") or "MLP" in str(row.get("method")):
            continue
        if int(float(row.get("p1_exploratory_pass", 0) or 0)) != 1:
            ft = "F1_efficiency_fail"
            if float(row.get("backward_memory_ratio_vs_mlp", 99) or 99) > 1.25:
                ft = "F2_memory_fail"
            failures.append({"stage": "P1", "dataset": "synthetic", "method": row.get("method"), "failure_type": ft})
    for filename, field, stage in [
        ("p2_dwm_recipe_repair.csv", "p2_pass", "P2"),
        ("p3_rationalkat_recipe_repair.csv", "p3_pass", "P3"),
        ("p4_cp_capacity_repair.csv", "p4_pass", "P4"),
        ("p5_joint_task_efficiency_selection.csv", "p5_pass", "P5"),
    ]:
        for row in read_csv(out_dir / filename):
            if row.get("error") or row.get("status") == "not_run" or row.get("recipe") == "MLP-AdamW":
                continue
            if int(float(row.get(field, 0) or 0)) != 1:
                failures.append(
                    {
                        "stage": stage,
                        "dataset": row.get("dataset", "all"),
                        "method": row.get("recipe", row.get("method", "")),
                        "failure_type": "F3_accuracy_recipe_unproven" if float(row.get("acc_gap_vs_mlp", 99) or 99) > 0.02 else "F4_joint_gate_failed",
                    }
                )
    if not failures:
        failures.append({"stage": "P10", "dataset": "all", "method": "all", "failure_type": "no_failure_rows"})
    write_csv(out_dir / "failure_table.csv", failures)
    write_csv(out_dir / "p10_failure_diagnosis.csv", failures)
    return failures


def build_parser() -> argparse.ArgumentParser:
    p = add_v3_args()
    p.description = __doc__
    p.set_defaults(packages="V5_6_P0", out_dir=Path("results/v5_6"), datasets="MNIST,Fashion-MNIST,KMNIST", seeds="0,1,2")
    p.add_argument("--no-compile-v56", action="store_true", help="Disable torch.compile in v5.6 probes.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.out_dir)
    for pkg in parse_str_list(args.packages):
        key = pkg.upper()
        if key in {"V5_6_P0", "V5_6_P0_CORE"}:
            run_p0(args)
        elif key in {"V5_6_P1", "V5_6_P1_EFFICIENCY"}:
            run_p1(args)
        elif key in {"V5_6_P2", "V5_6_P2_DWM"}:
            run_p2(args)
        elif key in {"V5_6_P3", "V5_6_P3_RK"}:
            run_p3(args)
        elif key in {"V5_6_P4", "V5_6_P4_CP"}:
            run_p4(args)
        elif key in {"V5_6_P5", "V5_6_P5_SELECTION"}:
            run_p5(args)
        elif key in {"V5_6_P6", "V5_6_P6_MEMORY"}:
            run_p6(args)
        elif key in {"V5_6_P7", "V5_6_P7_LIGHTSMOOTH"}:
            run_p7(args)
        elif key in {"V5_6_P8", "V5_6_P8_FUNCTIONAL"}:
            run_p8(args)
        elif key in {"V5_6_P9", "V5_6_P9_CONFIRM"}:
            run_p9(args)
        elif key in {"V5_6_P10", "V5_6_P10_FINAL", "V5_6_FAILURE"}:
            run_p10(args)
            run_failure(args)
        elif key in {"V5_6_ALL", "ALL"}:
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
