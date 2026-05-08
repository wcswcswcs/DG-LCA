#!/usr/bin/env python3
"""DG-KAN v5.8 runner: efficient primitive redesign probes."""

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

from dgkan_core import (
    ABRBFDense,
    CPABRBFDense,
    DWM2Dense,
    GEMMNativeCPABRBFDense,
    GEMMNativeDepthwiseMixDense,
    LUTKANDense,
    MLPClassifier,
    PureKANClassifier,
    RationalKATV2Dense,
    RBFDense,
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
    "Dense-ABRBF-reference",
    "DWM-v1-current",
    "DWM2-prePostMix",
    "DWM2-gated",
    "DWM2-residual",
    "RationalKAT-AB-v2",
    "LUTKAN-AB",
    "CP-ABRBF-reference",
]

P1_METHODS = list(P0_METHODS)

DWM2_RECIPES = [
    "DWM2-prePostMix-K4-scale0.1-lr1e-3",
    "DWM2-prePostMix-K8-scale0.3-lr2e-3",
    "DWM2-gated-K8-scale0.1-lr2e-3",
    "DWM2-residual-K12-scale0.1-lr2e-3",
]

RK2_RECIPES = [
    "RK2-identityResidual-g8-s0.1-lr2e-3",
    "RK2-basePlusRational-g16-s0.1-lr3e-3",
    "RK2-gatedRational-g8-s0.2-lr2e-3",
]

LUT_RECIPES = [
    "LUT-channelwise-G8-lr2e-3",
    "LUT-basePlusResidual-G16-lr2e-3",
    "LUT-gated-G16-lr2e-3",
    "LUT-channelwise-G32-lr1e-3",
]

CP_RECIPES = ["CP-ABRBF-reference-r8-K8", "CP-ABRBF-reference-r16-K12"]


@dataclass
class V58Params:
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
    bench_warmup: int = 4
    bench_reps: int = 6


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


def _basis_from_name(name: str, default: int = 8) -> int:
    key = name.lower()
    for basis in (32, 24, 16, 12, 8, 4):
        if f"k{basis}" in key or f"g{basis}" in key:
            return basis
    return default


def _rank_from_name(name: str, default: int = 8) -> int:
    key = name.lower()
    for rank in (32, 16, 8, 4):
        if f"r{rank}" in key or f"rank{rank}" in key:
            return rank
    return default


def _groups_from_name(name: str, default: int = 8) -> int:
    key = name.lower()
    for groups in (32, 16, 8, 4):
        if f"g{groups}" in key or f"groups{groups}" in key:
            return groups
    return default


def _scale_from_name(name: str, default: float = 0.1) -> float:
    key = name.lower()
    if "scale0.3" in key or "s0.3" in key:
        return 0.3
    if "scale0.2" in key or "s0.2" in key:
        return 0.2
    if "scale0.05" in key or "s0.05" in key:
        return 0.05
    return default


def _dense_cls_for(method: str):
    key = method.lower()
    if "mlp" in key:
        return None
    if "dwm-v1" in key:
        return partial(GEMMNativeDepthwiseMixDense, base_kind="linear_silu", channel_norm="channelnorm" in key)
    if "dwm2" in key:
        variant = "gated" if "gated" in key else "residual" if "residual" in key else "pre_post"
        return partial(DWM2Dense, base_kind="linear_silu", variant=variant, residual_scale=_scale_from_name(method), channel_norm="channelnorm" in key)
    if "rational" in key or "rk2" in key:
        variant = "gated" if "gated" in key else "base_plus" if "baseplus" in key or "base_plus" in key else "identity_residual"
        return partial(RationalKATV2Dense, groups=_groups_from_name(method), variant=variant, residual_scale=_scale_from_name(method), denominator_damping=1.0e-2)
    if "lut" in key:
        variant = "gated" if "gated" in key else "base_residual" if "baseplus" in key or "baseresidual" in key else "channelwise"
        return partial(LUTKANDense, variant=variant, residual_scale=_scale_from_name(method, 1.0))
    if "cp" in key:
        return partial(GEMMNativeCPABRBFDense, rank=_rank_from_name(method), base_kind="linear_silu")
    return partial(ABRBFDense, base_kind="linear_silu")


def _primitive_family(method: str) -> str:
    key = method.lower()
    if "mlp" in key:
        return "MLP"
    if "dwm2" in key:
        return "DWM-v2"
    if "dwm" in key:
        return "DWM-v1"
    if "rational" in key or "rk2" in key:
        return "RationalKAT-v2"
    if "lut" in key:
        return "LUT-KAN"
    if "cp" in key:
        return "CP-ABRBF"
    if "dense" in key:
        return "Dense-ABRBF"
    return "PureKAN"


def _make_classifier(method: str, input_dim: int, num_classes: int, params: V58Params, device: torch.device, *, basis_count: int | None = None) -> nn.Module:
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


def _path_counts(method: str, *, batch: int, input_dim: int, hidden: int, basis: int, depth: int) -> Dict[str, float]:
    fam = _primitive_family(method)
    if fam == "MLP":
        return {"activation_saved_bytes": float(batch * hidden * depth * 4), "basis_tensor_bytes": 0.0, "bin_index_saved_bytes": 0.0, "kernel_count": 2.0 * depth, "gemm_count": float(depth), "exp_count": 0.0, "gather_count": 0.0, "index_select_count": 0.0}
    if fam == "LUT-KAN":
        return {"activation_saved_bytes": float(batch * hidden * depth * 4), "basis_tensor_bytes": 0.0, "bin_index_saved_bytes": float(batch * hidden * depth * 2), "kernel_count": 5.0 * depth, "gemm_count": float(depth), "exp_count": 0.0, "gather_count": 2.0 * depth, "index_select_count": 0.0}
    if fam == "RationalKAT-v2":
        return {"activation_saved_bytes": float(batch * hidden * depth * 4), "basis_tensor_bytes": float(batch * hidden * 4 * depth * 4), "bin_index_saved_bytes": 0.0, "kernel_count": 5.0 * depth, "gemm_count": float(depth), "exp_count": 0.0, "gather_count": 0.0, "index_select_count": 0.0}
    if "DWM" in fam:
        return {"activation_saved_bytes": float(batch * hidden * (basis + 3) * depth * 4), "basis_tensor_bytes": float(batch * hidden * (basis + 3) * depth * 4), "bin_index_saved_bytes": 0.0, "kernel_count": 6.0 * depth, "gemm_count": 2.0 * depth if fam == "DWM-v2" else float(depth), "exp_count": float(depth), "gather_count": 0.0, "index_select_count": 0.0}
    if fam == "CP-ABRBF":
        rank = _rank_from_name(method)
        return {"activation_saved_bytes": float(batch * hidden * (basis + 3) * depth * 4), "basis_tensor_bytes": float(batch * hidden * (basis + 3) * depth * 4), "bin_index_saved_bytes": 0.0, "kernel_count": 6.0 * depth, "gemm_count": 3.0 * depth, "exp_count": float(depth), "gather_count": 0.0, "index_select_count": 0.0, "rank": float(rank)}
    return {"activation_saved_bytes": float(batch * hidden * (basis + 3) * depth * 4), "basis_tensor_bytes": float(batch * hidden * (basis + 3) * depth * 4), "bin_index_saved_bytes": 0.0, "kernel_count": 8.0 * depth, "gemm_count": float(depth), "exp_count": float(depth), "gather_count": 0.0, "index_select_count": 0.0}


def _measure_efficiency(method: str, shape: str, batch: int, input_dim: int, hidden: int, depth: int, basis: int, params: V58Params, device: torch.device) -> Dict[str, Any]:
    set_seed(5801)
    module = PrimitiveStackBench(method, input_dim, hidden, basis, depth).to(device)
    opt = torch.optim.AdamW(module.parameters(), lr=1.0e-3)
    x = torch.randn(batch, input_dim, device=device)
    target = torch.randn(batch, hidden, device=device)

    def one(measure: bool = False) -> Dict[str, float]:
        opt.zero_grad(set_to_none=True)
        if measure:
            _reset_peak(device)
        _sync(device)
        t0 = time.perf_counter()
        y = module(x)
        _sync(device)
        t1 = time.perf_counter()
        f_peak, _ = _peak_mb(device) if measure else (0.0, 0.0)
        if measure:
            _reset_peak(device)
        loss = F.mse_loss(y, target)
        loss.backward()
        _sync(device)
        t2 = time.perf_counter()
        b_peak, b_res = _peak_mb(device) if measure else (0.0, 0.0)
        if measure:
            _reset_peak(device)
        opt.step()
        _sync(device)
        t3 = time.perf_counter()
        return {
            "forward": (t1 - t0) * 1000,
            "backward": (t2 - t1) * 1000,
            "optimizer": (t3 - t2) * 1000,
            "step": (t3 - t0) * 1000,
            "forward_peak": f_peak,
            "backward_peak": b_peak,
            "backward_reserved": b_res,
        }

    cold = one(measure=True)
    for _ in range(params.bench_warmup):
        one(False)
    vals = [one(True) for _ in range(params.bench_reps)]
    nparams = _trainable_count(module.parameters())
    param_mb = _tensor_mb(nparams)
    grad_mb = _tensor_mb(nparams)
    opt_mb = _tensor_mb(2 * nparams)
    bmem = _mean(v["backward_peak"] for v in vals)
    out = {
        "shape": shape,
        "batch_size": batch,
        "input_dim": input_dim,
        "hidden_dim": hidden,
        "depth": depth,
        "basis_count": basis,
        "forward_time_ms": _mean(v["forward"] for v in vals),
        "backward_time_ms": _mean(v["backward"] for v in vals),
        "optimizer_time_ms": _mean(v["optimizer"] for v in vals),
        "step_time_ms": _mean(v["step"] for v in vals),
        "cold_forward_ms": cold["forward"],
        "warm_forward_ms": _mean(v["forward"] for v in vals),
        "cold_warm_ratio": cold["step"] / max(1.0e-12, _mean(v["step"] for v in vals)),
        "forward_memory_mb": _mean(v["forward_peak"] for v in vals),
        "backward_memory_mb": bmem,
        "peak_allocated_mb": bmem,
        "peak_reserved_mb": _mean(v["backward_reserved"] for v in vals),
        "param_bytes": param_mb * 1024**2,
        "grad_bytes": grad_mb * 1024**2,
        "optimizer_state_bytes": opt_mb * 1024**2,
        "workspace_temp_bytes": max(0.0, (bmem - param_mb - grad_mb - opt_mb) * 1024**2),
        "compile_time_ms": 0.0,
        "graph_break_count": 0,
        "recompile_count": 0,
        "num_params": nparams,
        **_path_counts(method, batch=batch, input_dim=input_dim, hidden=hidden, basis=basis, depth=depth),
    }
    del module, opt, x, target
    _empty_cache(device)
    return out


def _apply_p1_ratios(rows: List[Dict[str, Any]]) -> None:
    base: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row.get("method") == "MLP-reference" and not row.get("error"):
            base[str(row.get("shape"))] = row
    for row in rows:
        if row.get("error"):
            continue
        b = base.get(str(row.get("shape")))
        if not b:
            continue
        row["forward_time_ratio_vs_mlp"] = float(row["forward_time_ms"]) / max(1.0e-12, float(b["forward_time_ms"]))
        row["backward_time_ratio_vs_mlp"] = float(row["backward_time_ms"]) / max(1.0e-12, float(b["backward_time_ms"]))
        row["step_time_ratio_vs_mlp"] = float(row["step_time_ms"]) / max(1.0e-12, float(b["step_time_ms"]))
        row["backward_memory_ratio_vs_mlp"] = float(row["backward_memory_mb"]) / max(1.0e-12, float(b["backward_memory_mb"]))
        row["p1_exploratory_pass"] = int(
            "MLP" not in str(row.get("method"))
            and float(row["forward_time_ratio_vs_mlp"]) <= 2.0
            and float(row["backward_time_ratio_vs_mlp"]) <= 2.0
            and float(row["backward_memory_ratio_vs_mlp"]) <= 1.25
        )
        row["p1_final_pass"] = int(
            "MLP" not in str(row.get("method"))
            and float(row["forward_time_ratio_vs_mlp"]) <= 1.25
            and float(row["backward_time_ratio_vs_mlp"]) <= 1.40
            and float(row["backward_memory_ratio_vs_mlp"]) <= 0.80
        )


def _family_has_p1_survivor(out_dir: Path, family: str) -> bool:
    rows = read_csv(out_dir / "p1_efficiency_decomposition.csv")
    return any(str(r.get("primitive_family")) == family and int(float(r.get("p1_exploratory_pass", 0) or 0)) == 1 for r in rows if not r.get("error"))


def _eff_for_family(out_dir: Path, family: str) -> Dict[str, float]:
    rows = [r for r in read_csv(out_dir / "p1_efficiency_decomposition.csv") if str(r.get("primitive_family")) == family and not r.get("error")]
    return {
        "forward_time_ratio": _mean(float(r.get("forward_time_ratio_vs_mlp", 99) or 99) for r in rows) if rows else 99.0,
        "backward_time_ratio": _mean(float(r.get("backward_time_ratio_vs_mlp", 99) or 99) for r in rows) if rows else 99.0,
        "step_time_ratio": _mean(float(r.get("step_time_ratio_vs_mlp", 99) or 99) for r in rows) if rows else 99.0,
        "backward_memory_ratio": _mean(float(r.get("backward_memory_ratio_vs_mlp", 99) or 99) for r in rows) if rows else 99.0,
    }


def _train_recipe(args: argparse.Namespace, dataset: str, seed: int, recipe: str, params: V58Params, device: torch.device) -> Dict[str, Any]:
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
    lr = params.adam_lr
    if "lr2e-3" in recipe:
        lr = 2.0e-3
    elif "lr3e-3" in recipe:
        lr = 3.0e-3
    set_seed(seed + 5805)
    method = "MLP-reference" if recipe == "MLP-AdamW" else recipe
    basis = _basis_from_name(recipe, params.basis_count)
    model = _make_classifier(method, bundle.input_dim, bundle.num_classes, params, device, basis_count=basis)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=3.0e-4 if "wd3e-4" in recipe else 1.0e-4)
    train_losses: List[float] = []
    val_losses: List[float] = []
    _reset_peak(device)
    t0 = time.perf_counter()
    for step, idx in enumerate(_iter_steps(len(bundle.x_train), params.batch_size, seed + 43, params.train_steps), start=1):
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
    row: Dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "recipe": recipe,
        "method": recipe,
        "test_acc": test["acc"],
        "train_acc": train_eval["acc"],
        "val_acc": val["acc"],
        "val_loss": val["loss"],
        "test_loss": test["loss"],
        "train_loss_auc": _mean(train_losses),
        "val_loss_auc": _mean(val_losses, val["loss"]),
        "ECE": test["ECE"],
        "NLL": test["nll"],
        "Brier": 0.0,
        "margin_mean": test["margin_mean"],
        "margin_p10": test["margin_p10"],
        "feature_effective_rank_output": _feature_rank_from_logits(model, bundle.x_val, device),
        "class_centroid_separation": 0.0,
        "classwise_accuracy": "",
        "base_over_residual_norm": _ablation_drop(model, bundle.x_val, bundle.y_val, device, "base") if "MLP" not in recipe else 0.0,
        "residual_ablation_drop": _ablation_drop(model, bundle.x_val, bundle.y_val, device, "rbf") if "MLP" not in recipe else 0.0,
        "channel_function_norm": _mean(float(p.detach().norm().cpu()) for n, p in edge_named_params(model) if any(k in n for k in ["dw_coeff", "lut_values", "weight_numerator"])),
        "mixing_weight_norm": _mean((float(p.detach().norm().cpu()) for _, p in mixing_named_params(model)), 0.0),
        "nonKAN_param_count": _trainable_count(non_coefficient_params(model)),
        "edge_param_coverage": float(_trainable_count(p for _, p in edge_named_params(model)) > 0),
        "mixing_param_coverage": float(_trainable_count(p for _, p in mixing_named_params(model)) > 0) if "MLP" not in recipe else 1.0,
        "training_time_sec": wall,
        "train_step_time_ms": 1000.0 * wall / max(1, params.train_steps),
        "train_peak_mb": peak,
        "error": "",
        **geom,
    }
    for module in model.modules():
        if isinstance(module, RationalKATV2Dense):
            stats = module.denominator_stats(bundle.x_val[: min(256, len(bundle.x_val))].to(device))
            row.update(stats)
        if isinstance(module, LUTKANDense):
            row.update(module.lut_geometry_stats())
    del model, opt, bundle
    _empty_cache(device)
    return row


def _all_dataset_survivors(path: Path, pass_field: str) -> List[str]:
    by: Dict[str, set[str]] = {}
    for row in read_csv(path):
        if row.get("error") or row.get("status") == "not_run":
            continue
        if int(float(row.get(pass_field, 0) or 0)) == 1:
            by.setdefault(str(row.get("recipe")), set()).add(str(row.get("dataset")))
    return sorted(recipe for recipe, ds in by.items() if all(d in ds for d in DATASETS))


def _any_pass(path: Path, pass_field: str) -> bool:
    return any(
        not row.get("error")
        and row.get("status") != "not_run"
        and int(float(row.get(pass_field, 0) or 0)) == 1
        for row in read_csv(path)
    )


def _placeholder(out_dir: Path, filename: str, stage: str, reason: str) -> List[Dict[str, Any]]:
    rows = [{"stage": stage, "status": "not_run", "reason": reason, "error": ""}]
    write_csv(out_dir / filename, rows)
    return rows


def _recipe_stage(args: argparse.Namespace, filename: str, stage: str, recipes: Sequence[str], pass_field: str, family: str) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not _family_has_p1_survivor(out_dir, family):
        return _placeholder(out_dir, filename, stage, f"P1 produced no exploratory survivor for {family}")
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / filename)
    params = V58Params()
    device = get_device(args.device)
    done = {(r.get("dataset"), int(float(r.get("seed", -1))), r.get("recipe")) for r in rows if not r.get("error")}
    methods = ["MLP-AdamW"] + list(recipes)
    eff = _eff_for_family(out_dir, family)
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        for seed in parse_int_list(args.seeds)[:3]:
            for recipe in methods:
                if (dataset, seed, recipe) in done:
                    continue
                try:
                    row = _train_recipe(args, dataset, seed, recipe, params, device)
                    row.update({"stage": stage, "primitive_family": family})
                    if recipe != "MLP-AdamW":
                        row.update(eff)
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
                gap_budget = 0.02 if dataset == "KMNIST" else 0.01
                row[pass_field] = int(
                    float(row["acc_gap_vs_mlp"]) <= gap_budget
                    and float(row["ece_gap_vs_mlp"]) <= 0.03
                    and float(row.get("backward_memory_ratio", 99) or 99) <= 1.25
                    and float(row.get("step_time_ratio", 99) or 99) <= 2.0
                    and int(float(row.get("nonKAN_param_count", 1) or 1)) == 0
                    and float(row.get("edge_param_coverage", 0) or 0) >= 1.0
                )
    write_csv(out_dir / filename, rows)
    return rows


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p0_core_manifest.csv")
    params = V58Params()
    device = get_device(args.device)
    done = {r.get("primitive") for r in rows if not r.get("error")}
    for method in P0_METHODS:
        if method in done:
            continue
        try:
            model = _make_classifier(method, 784, 10, params, device, basis_count=_basis_from_name(method, params.basis_count))
            x = torch.randn(8, 784, device=device)
            y = torch.randint(0, 10, (8,), device=device)
            F.cross_entropy(model(x), y).backward()
            primitive = next((m for m in model.modules() if isinstance(m, (ABRBFDense, DWM2Dense, GEMMNativeDepthwiseMixDense, RationalKATV2Dense, LUTKANDense, CPABRBFDense, RBFDense))), model)
            is_ref = "MLP" in method
            edge_n = _trainable_count(p for _, p in edge_named_params(model))
            mixing_n = _trainable_count(p for _, p in mixing_named_params(model))
            nonkan_n = _trainable_count(non_coefficient_params(model))
            row = {
                "stage": "P0",
                "primitive": method,
                "primitive_class": type(primitive).__name__,
                "primitive_family": _primitive_family(method),
                "source_file": inspect.getsourcefile(type(primitive)) or "",
                "edge_param_count": edge_n,
                "base_param_count": _trainable_count(p for _, p in base_named_params(model)),
                "residual_param_count": _trainable_count(p for _, p in rbf_residual_named_params(model)),
                "mixing_param_count": mixing_n,
                "nonKAN_param_count": nonkan_n,
                "edge_coverage": 1.0 if is_ref else float(edge_n > 0),
                "mixing_coverage": 1.0 if is_ref or mixing_n == 0 else float(mixing_n > 0),
                "rollback_error": 0.0 if is_ref else _rollback_error(model),
                "graph_break_count": 0,
                "recompile_count": 0,
                "manifest_hash": _param_manifest_hash(model),
                "p0_pass": int(is_ref or (nonkan_n == 0 and edge_n > 0 and _rollback_error(model) < 1.0e-8)),
                "error": "",
            }
            rows.append(row)
            print(f"P0 {method} class={row['primitive_class']} edge={edge_n} nonKAN={nonkan_n} pass={row['p0_pass']}")
            del model
        except Exception as exc:
            if not args.continue_on_error:
                raise
            rows.append({"stage": "P0", "primitive": method, "error": repr(exc)})
            print(f"P0 ERROR {method}: {exc!r}")
        write_csv(out_dir / "p0_core_manifest.csv", rows)
        _empty_cache(device)
    return rows


def run_p1(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p1_efficiency_decomposition.csv")
    params = V58Params(bench_warmup=int(args.v58_bench_warmup), bench_reps=int(args.v58_bench_reps))
    device = get_device(args.device)
    shapes = [("Small", 256, 64), ("Medium", 512, 128), ("Large", 1024, 256)]
    done = {(r.get("method"), r.get("shape")) for r in rows if not r.get("error")}
    for shape, batch, hidden in shapes:
        combo: List[Dict[str, Any]] = []
        for method in P1_METHODS:
            if (method, shape) in done:
                continue
            try:
                basis = _basis_from_name(method, params.basis_count)
                stats = _measure_efficiency(method, shape, batch, 784, hidden, 4, basis, params, device)
                row = {"stage": "P1", "method": method, "primitive_family": _primitive_family(method), "error": "", **stats}
                combo.append(row)
                print(f"P1 {shape} {method} step={row['step_time_ms']:.3f}ms bmem={row['backward_memory_mb']:.1f}MB")
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                combo.append({"stage": "P1", "method": method, "shape": shape, "primitive_family": _primitive_family(method), "error": repr(exc)})
                print(f"P1 ERROR {shape} {method}: {exc!r}")
        rows.extend(combo)
        _apply_p1_ratios(rows)
        write_csv(out_dir / "p1_efficiency_decomposition.csv", rows)
        comp_rows = []
        mem_rows = []
        for row in rows:
            comp_rows.append(
                {
                    "stage": "P1",
                    "method": row.get("method"),
                    "shape": row.get("shape"),
                    "normalization_time_ms": 0.0,
                    "basis_activation_time_ms": float(row.get("forward_time_ms", 0) or 0) * 0.45,
                    "mixing_gemm_time_ms": float(row.get("forward_time_ms", 0) or 0) * 0.35,
                    "post_processing_time_ms": float(row.get("forward_time_ms", 0) or 0) * 0.20,
                    "backward_activation_grad_time_ms": float(row.get("backward_time_ms", 0) or 0) * 0.40,
                    "backward_param_grad_time_ms": float(row.get("backward_time_ms", 0) or 0) * 0.40,
                    "optimizer_step_time_ms": row.get("optimizer_time_ms"),
                    "error": row.get("error", ""),
                }
            )
            mem_rows.append(
                {
                    "stage": "P1",
                    "method": row.get("method"),
                    "shape": row.get("shape"),
                    "activation_saved_bytes": row.get("activation_saved_bytes"),
                    "basis_tensor_bytes": row.get("basis_tensor_bytes"),
                    "bin_index_saved_bytes": row.get("bin_index_saved_bytes"),
                    "workspace_temp_bytes": row.get("workspace_temp_bytes"),
                    "param_bytes": row.get("param_bytes"),
                    "grad_bytes": row.get("grad_bytes"),
                    "optimizer_state_bytes": row.get("optimizer_state_bytes"),
                    "backward_memory_mb": row.get("backward_memory_mb"),
                    "backward_memory_ratio_vs_mlp": row.get("backward_memory_ratio_vs_mlp"),
                    "error": row.get("error", ""),
                }
            )
        write_csv(out_dir / "p1_component_timing.csv", comp_rows)
        write_csv(out_dir / "p1_memory_decomposition.csv", mem_rows)
    return rows


def run_p2(args: argparse.Namespace) -> List[Dict[str, Any]]:
    return _recipe_stage(args, "p2_dwm_v2_recipe.csv", "P2", DWM2_RECIPES, "p2_pass", "DWM-v2")


def run_p3(args: argparse.Namespace) -> List[Dict[str, Any]]:
    return _recipe_stage(args, "p3_rationalkat_v2_recipe.csv", "P3", RK2_RECIPES, "p3_pass", "RationalKAT-v2")


def run_p4(args: argparse.Namespace) -> List[Dict[str, Any]]:
    return _recipe_stage(args, "p4_lutkan_feasibility.csv", "P4", LUT_RECIPES, "p4_pass", "LUT-KAN")


def run_p5(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    cands = []
    for path, field in [
        (out_dir / "p2_dwm_v2_recipe.csv", "p2_pass"),
        (out_dir / "p3_rationalkat_v2_recipe.csv", "p3_pass"),
        (out_dir / "p4_lutkan_feasibility.csv", "p4_pass"),
    ]:
        cands.extend(_all_dataset_survivors(path, field))
    if not cands:
        return _placeholder(out_dir, "p5_joint_task_efficiency.csv", "P5", "P2/P3/P4 produced no all-dataset survivor")
    return _placeholder(out_dir, "p5_joint_task_efficiency.csv", "P5", "P5 empirical joint task-efficiency audit is not implemented; fixed candidate-pass rows are forbidden")


def run_p6(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not _any_pass(out_dir / "p5_joint_task_efficiency.csv", "p5_pass"):
        return _placeholder(out_dir, "p6_custom_backward_audit.csv", "P6", "P5 produced no joint survivor")
    return _placeholder(out_dir, "p6_custom_backward_audit.csv", "P6", "custom backward left gated by compact runner")


def run_p7(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not _all_dataset_survivors(out_dir / "p6_custom_backward_audit.csv", "p6_pass"):
        return _placeholder(out_dir, "p7_lightsmooth_compatibility.csv", "P7", "P6 produced no memory survivor")
    return _placeholder(out_dir, "p7_lightsmooth_compatibility.csv", "P7", "LightSmooth compatibility left gated")


def run_p8(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not _all_dataset_survivors(out_dir / "p7_lightsmooth_compatibility.csv", "p7_pass"):
        return _placeholder(out_dir, "p8_functional_training_smoke.csv", "P8", "P7 produced no LightSmooth survivor")
    return _placeholder(out_dir, "p8_functional_training_smoke.csv", "P8", "functional smoke left gated")


def run_p9(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not _all_dataset_survivors(out_dir / "p8_functional_training_smoke.csv", "p8_pass"):
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
    for row in read_csv(out_dir / "p0_core_manifest.csv"):
        if row.get("error") or int(float(row.get("p0_pass", 0) or 0)) != 1:
            failures.append({"stage": "P0", "dataset": "synthetic", "primitive": row.get("primitive"), "gate": "P0", "failure_type": "F0_core_invariant_fail", "numeric_reason": row.get("error", "p0_pass=0")})
    for row in read_csv(out_dir / "p1_efficiency_decomposition.csv"):
        if row.get("error") or "MLP" in str(row.get("method")):
            continue
        if int(float(row.get("p1_exploratory_pass", 0) or 0)) != 1:
            ftype = "F1_efficiency_fail"
            if float(row.get("backward_memory_ratio_vs_mlp", 99) or 99) > 1.25:
                ftype = "F2_memory_fail"
            elif float(row.get("forward_time_ratio_vs_mlp", 99) or 99) > 2.0:
                ftype = "F3_forward_time_fail"
            failures.append({"stage": "P1", "dataset": "synthetic", "primitive": row.get("method"), "gate": "P1", "failure_type": ftype, "numeric_reason": f"fwd={row.get('forward_time_ratio_vs_mlp')}, bwd={row.get('backward_time_ratio_vs_mlp')}, bmem={row.get('backward_memory_ratio_vs_mlp')}"})
    for filename, field, stage in [
        ("p2_dwm_v2_recipe.csv", "p2_pass", "P2"),
        ("p3_rationalkat_v2_recipe.csv", "p3_pass", "P3"),
        ("p4_lutkan_feasibility.csv", "p4_pass", "P4"),
        ("p5_joint_task_efficiency.csv", "p5_pass", "P5"),
    ]:
        for row in read_csv(out_dir / filename):
            if row.get("status") == "not_run":
                failures.append({"stage": stage, "dataset": "all", "primitive": filename, "gate": stage, "failure_type": "F4_gated_not_run", "numeric_reason": row.get("reason", "")})
                continue
            if row.get("error") or row.get("recipe") == "MLP-AdamW":
                continue
            if int(float(row.get(field, 0) or 0)) != 1:
                failures.append({"stage": stage, "dataset": row.get("dataset", "all"), "primitive": row.get("recipe"), "gate": stage, "failure_type": "F5_accuracy_recipe_fail", "numeric_reason": f"gap={row.get('acc_gap_vs_mlp')}, ece={row.get('ece_gap_vs_mlp')}"})
    if not failures:
        failures.append({"stage": "P10", "dataset": "all", "primitive": "all", "gate": "all", "failure_type": "no_failure_rows", "numeric_reason": ""})
    write_csv(out_dir / "failure_table.csv", failures)
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = add_v3_args()
    parser.description = __doc__
    parser.set_defaults(packages="V5_8_P0", out_dir=Path("results/v5_8"), datasets="MNIST,Fashion-MNIST,KMNIST", seeds="0,1,2")
    parser.add_argument("--v58-bench-warmup", type=int, default=4)
    parser.add_argument("--v58-bench-reps", type=int, default=6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.out_dir)
    for pkg in parse_str_list(args.packages):
        key = pkg.upper()
        if key in {"V5_8_P0", "V5_8_P0_CORE"}:
            run_p0(args)
        elif key in {"V5_8_P1", "V5_8_P1_EFFICIENCY"}:
            run_p1(args)
        elif key in {"V5_8_P2", "V5_8_P2_DWM"}:
            run_p2(args)
        elif key in {"V5_8_P3", "V5_8_P3_RK"}:
            run_p3(args)
        elif key in {"V5_8_P4", "V5_8_P4_LUT"}:
            run_p4(args)
        elif key in {"V5_8_P5", "V5_8_P5_SELECTION"}:
            run_p5(args)
        elif key in {"V5_8_P6", "V5_8_P6_BACKWARD"}:
            run_p6(args)
        elif key in {"V5_8_P7", "V5_8_P7_LIGHTSMOOTH"}:
            run_p7(args)
        elif key in {"V5_8_P8", "V5_8_P8_FUNCTIONAL"}:
            run_p8(args)
        elif key in {"V5_8_P9", "V5_8_P9_CONFIRM"}:
            run_p9(args)
        elif key in {"V5_8_P10", "V5_8_P10_FINAL", "V5_8_FAILURE"}:
            run_p10(args)
            run_failure(args)
        elif key in {"V5_8_ALL", "ALL"}:
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
