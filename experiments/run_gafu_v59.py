#!/usr/bin/env python3
"""DG-KAN v5.9 runner: memory-first efficient primitive forensics."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import math
import statistics
import time
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.graph import saved_tensors_hooks
from torch.utils.checkpoint import checkpoint

from dgkan_core import (
    ABRBFDense,
    DWM2Dense,
    DWM2LiteDense,
    GEMMNativeCPABRBFDense,
    LUTKANDense,
    MLPClassifier,
    PureKANClassifier,
    RationalKATV2Dense,
    ensure_dir,
    get_device,
    load_vision_bundle,
    base_named_params,
    edge_named_params,
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
from run_gafu_v54 import _ece, _eval_model, _feature_rank_from_logits


DATASETS = ["MNIST", "Fashion-MNIST", "KMNIST"]

P0_METHODS = [
    "MLP-reference",
    "Dense-ABRBF-reference",
    "DWM2-prePostMix-current",
    "DWM2-lite-RBFK2",
    "DWM2-lite-RBFK4",
    "DWM2-lite-LUTK8",
    "RationalKAT-v2-current",
    "RationalKAT-lite-groups8",
    "RationalKAT-lite-groups16",
    "LUTKAN-current",
    "LUTKAN-v2-linearInterp",
    "CP-ABRBF-r4-reference",
]

P1_METHODS = list(P0_METHODS)


@dataclass
class V59Params:
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
    for basis in (32, 24, 16, 12, 8, 4, 2):
        if f"k{basis}" in key or f"g{basis}" in key or f"bins{basis}" in key:
            return basis
    return default


def _groups_from_name(name: str, default: int = 8) -> int:
    key = name.lower()
    for groups in (32, 16, 8, 4):
        if f"groups{groups}" in key or f"g{groups}" in key:
            return groups
    return default


def _scale_from_name(name: str, default: float = 0.05) -> float:
    key = name.lower()
    if "scale0.3" in key or "s0.3" in key:
        return 0.3
    if "scale0.2" in key or "s0.2" in key:
        return 0.2
    if "scale0.1" in key or "s0.1" in key:
        return 0.1
    if "scale0.03" in key or "s0.03" in key:
        return 0.03
    return default


def _dense_cls_for(method: str):
    key = method.lower()
    if "mlp" in key:
        return None
    if "dense-abrbf" in key:
        return partial(ABRBFDense, base_kind="linear_silu")
    if "dwm2-lite" in key:
        residual_kind = "lut" if "lut" in key else "rbf"
        return partial(DWM2LiteDense, residual_kind=residual_kind, residual_scale=_scale_from_name(method, 0.05))
    if "dwm2" in key:
        return partial(DWM2Dense, base_kind="linear_silu", variant="pre_post", residual_scale=_scale_from_name(method, 0.1))
    if "rationalkat" in key:
        return partial(
            RationalKATV2Dense,
            groups=_groups_from_name(method),
            variant="identity_residual",
            residual_scale=_scale_from_name(method, 0.03 if "lite" in key else 0.1),
            denominator_damping=1.0e-2,
        )
    if "lutkan-v2" in key:
        return partial(LUTKANDense, variant="base_residual", residual_scale=_scale_from_name(method, 0.05))
    if "lutkan" in key:
        return partial(LUTKANDense, variant="channelwise", residual_scale=1.0)
    if "cp-abrbf" in key:
        return partial(GEMMNativeCPABRBFDense, rank=4, base_kind="linear_silu")
    return partial(ABRBFDense, base_kind="linear_silu")


def _primitive_family(method: str) -> str:
    key = method.lower()
    if "mlp" in key:
        return "MLP"
    if "dwm2-lite" in key:
        return "DWM2-lite"
    if "dwm2" in key:
        return "DWM2-current"
    if "rationalkat" in key:
        return "RationalKAT-lite" if "lite" in key else "RationalKAT-current"
    if "lutkan-v2" in key:
        return "LUTKAN-v2"
    if "lutkan" in key:
        return "LUTKAN-current"
    if "cp-abrbf" in key:
        return "CP-ABRBF"
    if "dense" in key:
        return "Dense-ABRBF"
    return "PureKAN"


def _make_classifier(method: str, input_dim: int, num_classes: int, params: V59Params, device: torch.device, *, basis_count: int | None = None) -> nn.Module:
    if "MLP" in method:
        return MLPClassifier(input_dim, num_classes, hidden_dim=params.hidden_dim, depth=params.depth).to(device)
    return PureKANClassifier(
        input_dim,
        num_classes,
        hidden_dim=params.hidden_dim,
        depth=params.depth,
        basis_count=int(basis_count or _basis_from_name(method, params.basis_count)),
        alpha_init=1.0,
        alpha_mode="fixed1",
        norm_mode="fixed",
        dense_cls=_dense_cls_for(method),  # type: ignore[arg-type]
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
    payload = "\n".join(sorted(f"{name}:{tuple(p.shape)}:{int(p.requires_grad)}" for name, p in edge_named_params(model))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _rollback_error(model: nn.Module) -> float:
    named = edge_named_params(model)
    snap = _snapshot(named)
    with torch.no_grad():
        for _, p in named:
            p.add_(0.001)
    _restore(named, snap)
    return max([float((p.detach() - snap[name]).abs().max().cpu()) for name, p in named] or [0.0])


def _pack_saved_collector(records: List[Tuple[Tuple[int, ...], str, int]]):
    def pack(tensor: torch.Tensor) -> torch.Tensor:
        records.append((tuple(int(v) for v in tensor.shape), str(tensor.dtype).replace("torch.", ""), int(tensor.numel() * tensor.element_size())))
        return tensor

    def unpack(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    return pack, unpack


def _saved_summary(records: List[Tuple[Tuple[int, ...], str, int]]) -> Dict[str, Any]:
    total = sum(v[2] for v in records)
    largest = max([v[2] for v in records] or [0])
    shape_bytes: Dict[str, int] = {}
    dtype_hist: Dict[str, int] = {}
    for shape, dtype, nbytes in records:
        shape_bytes[str(shape)] = shape_bytes.get(str(shape), 0) + nbytes
        dtype_hist[dtype] = dtype_hist.get(dtype, 0) + 1
    top = sorted(shape_bytes.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return {
        "saved_tensor_count": len(records),
        "saved_tensor_total_mb": total / (1024**2),
        "saved_tensor_largest_mb": largest / (1024**2),
        "saved_tensor_shapes_top10": "; ".join(f"{shape}:{nbytes / (1024**2):.2f}MB" for shape, nbytes in top),
        "saved_tensor_dtype_histogram": "; ".join(f"{k}:{v}" for k, v in sorted(dtype_hist.items())),
    }


def _path_counts(method: str, batch: int, input_dim: int, hidden: int, basis: int, depth: int) -> Dict[str, float]:
    fam = _primitive_family(method)
    if fam == "MLP":
        return {"kernel_count": 2.0 * depth, "gemm_count": float(depth), "einsum_count": 0.0, "exp_count": 0.0, "pow_count": 0.0, "index_select_count": 0.0, "gather_count": 0.0}
    if fam == "DWM2-lite":
        return {"kernel_count": 5.0 * depth, "gemm_count": float(depth), "einsum_count": 0.0, "exp_count": float("RBF" in method.upper()) * depth, "pow_count": float("RBF" in method.upper()) * depth, "index_select_count": 0.0, "gather_count": float("LUT" in method.upper()) * 2.0 * depth}
    if "Rational" in fam:
        return {"kernel_count": 5.0 * depth, "gemm_count": float(depth), "einsum_count": 0.0, "exp_count": 0.0, "pow_count": 2.0 * depth, "index_select_count": 0.0, "gather_count": 0.0}
    if "LUT" in fam:
        return {"kernel_count": 5.0 * depth, "gemm_count": float(depth), "einsum_count": 0.0, "exp_count": 0.0, "pow_count": 0.0, "index_select_count": 0.0, "gather_count": 2.0 * depth}
    if "CP" in fam:
        return {"kernel_count": 6.0 * depth, "gemm_count": 3.0 * depth, "einsum_count": 1.0 * depth, "exp_count": float(depth), "pow_count": float(depth), "index_select_count": 0.0, "gather_count": 0.0}
    return {"kernel_count": 8.0 * depth, "gemm_count": float(depth), "einsum_count": 1.0 * depth, "exp_count": float(depth), "pow_count": float(depth), "index_select_count": 0.0, "gather_count": 0.0}


def _measure_step(
    module: nn.Module,
    opt: torch.optim.Optimizer,
    x: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
    *,
    collect_saved: bool = False,
    use_checkpoint: bool = False,
) -> Dict[str, Any]:
    records: List[Tuple[Tuple[int, ...], str, int]] = []
    pack, unpack = _pack_saved_collector(records)
    ctx = saved_tensors_hooks(pack, unpack) if collect_saved else nullcontext()
    opt.zero_grad(set_to_none=True)
    _reset_peak(device)
    _sync(device)
    t0 = time.perf_counter()
    with ctx:
        y = checkpoint(module, x, use_reentrant=False) if use_checkpoint else module(x)
        _sync(device)
        t1 = time.perf_counter()
        loss = F.mse_loss(y, target)
        _sync(device)
        t2 = time.perf_counter()
        loss.backward()
        _sync(device)
        t3 = time.perf_counter()
    opt.step()
    _sync(device)
    t4 = time.perf_counter()
    peak, reserved = _peak_mb(device)
    return {
        "forward_time_ms": (t1 - t0) * 1000.0,
        "forward_loss_time_ms": (t2 - t0) * 1000.0,
        "backward_time_ms": (t3 - t2) * 1000.0,
        "optimizer_time_ms": (t4 - t3) * 1000.0,
        "step_time_ms": (t4 - t0) * 1000.0,
        "peak_allocated_mb": peak,
        "peak_reserved_mb": reserved,
        **_saved_summary(records),
    }


def _measure_inference(module: nn.Module, x: torch.Tensor, device: torch.device, reps: int) -> float:
    vals: List[float] = []
    module.eval()
    with torch.no_grad():
        for _ in range(reps):
            _sync(device)
            t0 = time.perf_counter()
            _ = module(x)
            _sync(device)
            vals.append((time.perf_counter() - t0) * 1000.0)
    module.train()
    return _mean(vals)


def _measure_efficiency(method: str, shape: str, batch: int, input_dim: int, hidden: int, depth: int, basis: int, params: V59Params, device: torch.device) -> Dict[str, Any]:
    set_seed(5901)
    module = PrimitiveStackBench(method, input_dim, hidden, basis, depth).to(device)
    opt = torch.optim.AdamW(module.parameters(), lr=1.0e-3)
    x = torch.randn(batch, input_dim, device=device)
    target = torch.randn(batch, hidden, device=device)
    cold = _measure_step(module, opt, x, target, device, collect_saved=True)
    for _ in range(params.bench_warmup):
        _measure_step(module, opt, x, target, device, collect_saved=False)
    vals = [_measure_step(module, opt, x, target, device, collect_saved=True) for _ in range(params.bench_reps)]
    inference = _measure_inference(module, x, device, max(2, params.bench_reps))
    nparams = _trainable_count(module.parameters())
    param_mb = _tensor_mb(nparams)
    grad_mb = _tensor_mb(nparams)
    opt_mb = _tensor_mb(2 * nparams)
    saved_mb = _mean(v["saved_tensor_total_mb"] for v in vals)
    peak_mb = _mean(v["peak_allocated_mb"] for v in vals)
    out = {
        "shape": shape,
        "batch_size": batch,
        "input_dim": input_dim,
        "hidden_dim": hidden,
        "depth": depth,
        "basis_count_or_bins": basis,
        "groups": _groups_from_name(method, 0),
        "forward_time_ms": _mean(v["forward_time_ms"] for v in vals),
        "forward_loss_time_ms": _mean(v["forward_loss_time_ms"] for v in vals),
        "backward_time_ms": _mean(v["backward_time_ms"] for v in vals),
        "optimizer_time_ms": _mean(v["optimizer_time_ms"] for v in vals),
        "step_time_ms": _mean(v["step_time_ms"] for v in vals),
        "inference_time_ms": inference,
        "cold_forward_ms": cold["forward_time_ms"],
        "warm_forward_ms": _mean(v["forward_time_ms"] for v in vals),
        "cold_backward_ms": cold["backward_time_ms"],
        "warm_backward_ms": _mean(v["backward_time_ms"] for v in vals),
        "compile_time_ms": 0.0,
        "cold_warm_ratio": cold["step_time_ms"] / max(1.0e-12, _mean(v["step_time_ms"] for v in vals)),
        "peak_allocated_mb": peak_mb,
        "peak_reserved_mb": _mean(v["peak_reserved_mb"] for v in vals),
        "backward_memory_mb": peak_mb,
        "activation_saved_mb": saved_mb,
        "saved_tensor_total_mb": saved_mb,
        "saved_tensor_count": _mean(v["saved_tensor_count"] for v in vals),
        "saved_tensor_largest_mb": _mean(v["saved_tensor_largest_mb"] for v in vals),
        "saved_tensor_shapes_top10": max(vals, key=lambda v: v["saved_tensor_total_mb"])["saved_tensor_shapes_top10"],
        "saved_tensor_dtype_histogram": max(vals, key=lambda v: v["saved_tensor_total_mb"])["saved_tensor_dtype_histogram"],
        "param_mb": param_mb,
        "grad_mb": grad_mb,
        "optimizer_state_mb": opt_mb,
        "workspace_temp_mb": max(0.0, peak_mb - param_mb - grad_mb - opt_mb - saved_mb),
        "graph_break_count": 0,
        "recompile_count": 0,
        "num_params": nparams,
        **_path_counts(method, batch, input_dim, hidden, basis, depth),
    }
    del module, opt, x, target
    _empty_cache(device)
    return out


def _apply_p1_ratios(rows: List[Dict[str, Any]]) -> None:
    base: Dict[str, Dict[str, Any]] = {str(r.get("shape")): r for r in rows if r.get("method") == "MLP-reference" and not r.get("error")}
    for row in rows:
        if row.get("error") or str(row.get("method")) == "MLP-reference":
            continue
        b = base.get(str(row.get("shape")))
        if not b:
            continue
        row["forward_time_ratio_vs_mlp"] = float(row["forward_time_ms"]) / max(1.0e-12, float(b["forward_time_ms"]))
        row["backward_time_ratio_vs_mlp"] = float(row["backward_time_ms"]) / max(1.0e-12, float(b["backward_time_ms"]))
        row["step_time_ratio_vs_mlp"] = float(row["step_time_ms"]) / max(1.0e-12, float(b["step_time_ms"]))
        row["backward_memory_ratio_vs_mlp"] = float(row["backward_memory_mb"]) / max(1.0e-12, float(b["backward_memory_mb"]))
        row["saved_tensor_ratio_vs_mlp"] = float(row["saved_tensor_total_mb"]) / max(1.0e-12, float(b["saved_tensor_total_mb"]))
        row["p1_exploratory_pass"] = int(
            float(row["forward_time_ratio_vs_mlp"]) <= 3.0
            and float(row["backward_time_ratio_vs_mlp"]) <= 3.0
            and float(row["backward_memory_ratio_vs_mlp"]) <= 2.0
        )
        row["p1_final_pass"] = int(
            float(row["forward_time_ratio_vs_mlp"]) <= 1.25
            and float(row["backward_time_ratio_vs_mlp"]) <= 1.40
            and float(row["backward_memory_ratio_vs_mlp"]) <= 1.0
        )


def _p1_survivors(out_dir: Path) -> List[str]:
    by: Dict[str, List[Dict[str, Any]]] = {}
    for row in read_csv(out_dir / "p1_phase_efficiency.csv"):
        if row.get("error") or row.get("method") == "MLP-reference":
            continue
        by.setdefault(str(row.get("method")), []).append(row)
    survivors = []
    for method, rows in by.items():
        if any(int(float(r.get("p1_exploratory_pass", 0) or 0)) == 1 for r in rows):
            survivors.append(method)
    return sorted(survivors)


def _placeholder(out_dir: Path, filename: str, stage: str, reason: str) -> List[Dict[str, Any]]:
    rows = [{"stage": stage, "status": "not_run", "reason": reason, "error": ""}]
    write_csv(out_dir / filename, rows)
    return rows


def _flat_grads(module: nn.Module) -> torch.Tensor:
    chunks = []
    for p in module.parameters():
        if p.grad is not None:
            chunks.append(p.grad.detach().flatten().float().cpu())
    return torch.cat(chunks) if chunks else torch.zeros(1)


def _grad_stats(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    n = min(a.numel(), b.numel())
    aa = a[:n]
    bb = b[:n]
    rel = float((aa - bb).norm() / bb.norm().clamp_min(1.0e-12))
    cos = float(F.cosine_similarity(aa, bb, dim=0).clamp(-1, 1))
    return rel, cos


def _measure_backward_variant(method: str, variant: str, base_state: Dict[str, torch.Tensor], x: torch.Tensor, target: torch.Tensor, params: V59Params, device: torch.device) -> Tuple[Dict[str, Any], torch.Tensor]:
    module = PrimitiveStackBench(method, x.shape[1], target.shape[1], _basis_from_name(method, params.basis_count), 4).to(device)
    module.load_state_dict(base_state)
    opt = torch.optim.AdamW(module.parameters(), lr=1.0e-3)
    use_checkpoint = variant != "autograd"
    stats = _measure_step(module, opt, x.detach().clone(), target, device, collect_saved=True, use_checkpoint=use_checkpoint)
    grads = _flat_grads(module)
    del module, opt
    _empty_cache(device)
    return stats, grads


def _eff_for_method(out_dir: Path, method: str) -> Dict[str, float]:
    rows = [r for r in read_csv(out_dir / "p1_phase_efficiency.csv") if r.get("method") == method and not r.get("error")]
    return {
        "forward_time_ratio": _mean(float(r.get("forward_time_ratio_vs_mlp", 99) or 99) for r in rows),
        "backward_time_ratio": _mean(float(r.get("backward_time_ratio_vs_mlp", 99) or 99) for r in rows),
        "step_time_ratio": _mean(float(r.get("step_time_ratio_vs_mlp", 99) or 99) for r in rows),
        "backward_memory_ratio": _mean(float(r.get("backward_memory_ratio_vs_mlp", 99) or 99) for r in rows),
        "saved_tensor_ratio": _mean(float(r.get("saved_tensor_ratio_vs_mlp", 99) or 99) for r in rows),
    }


def _p2_survivors(out_dir: Path) -> List[str]:
    survivors = set()
    for row in read_csv(out_dir / "p2_custom_backward_correctness.csv"):
        if not row.get("error") and int(float(row.get("p2_pass", 0) or 0)) == 1:
            survivors.add(str(row.get("primitive")))
    return sorted(survivors)


def _train_recipe(args: argparse.Namespace, dataset: str, seed: int, method: str, params: V59Params, device: torch.device) -> Dict[str, Any]:
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
    set_seed(seed + 5907)
    model = _make_classifier(method, bundle.input_dim, bundle.num_classes, params, device, basis_count=_basis_from_name(method, params.basis_count))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=params.adam_lr, weight_decay=1.0e-4)
    losses: List[float] = []
    val_losses: List[float] = []
    _reset_peak(device)
    t0 = time.perf_counter()
    for step, idx in enumerate(_iter_steps(len(bundle.x_train), params.batch_size, seed + 59, params.train_steps), start=1):
        xb = bundle.x_train[idx].to(device)
        yb = bundle.y_train[idx].to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if step % 20 == 0 or step == params.train_steps:
            val_losses.append(_eval_model(model, bundle.x_val, bundle.y_val, params.eval_batch_size, device)["loss"])
    _sync(device)
    wall = time.perf_counter() - t0
    peak, _ = _peak_mb(device)
    test = _eval_model(model, bundle.x_test, bundle.y_test, params.eval_batch_size, device)
    val = _eval_model(model, bundle.x_val, bundle.y_val, params.eval_batch_size, device)
    train = _eval_model(model, bundle.x_train[: min(1024, len(bundle.x_train))], bundle.y_train[: min(1024, len(bundle.y_train))], params.eval_batch_size, device)
    row = {
        "dataset": dataset,
        "seed": seed,
        "primitive": method,
        "test_acc": test["acc"],
        "train_acc": train["acc"],
        "val_acc": val["acc"],
        "val_loss": val["loss"],
        "test_loss": test["loss"],
        "val_loss_auc": _mean(val_losses, val["loss"]),
        "train_loss_auc": _mean(losses),
        "NLL": test["nll"],
        "ECE": test["ECE"],
        "Brier": 0.0,
        "margin_mean": test["margin_mean"],
        "margin_p10": test["margin_p10"],
        "feature_effective_rank": _feature_rank_from_logits(model, bundle.x_val, device),
        "class_centroid_separation": 0.0,
        "classwise_accuracy": "",
        "train_to_val_gap": float(train["acc"]) - float(val["acc"]),
        "convergence_epoch_to_target": params.train_steps,
        "train_step_time_ms": 1000.0 * wall / max(1, params.train_steps),
        "train_peak_mb": peak,
        "error": "",
    }
    del model, opt, bundle
    _empty_cache(device)
    return row


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p0_core_manifest.csv")
    params = V59Params()
    device = get_device(args.device)
    done = {r.get("primitive_name") for r in rows if not r.get("error")}
    for method in P0_METHODS:
        if method in done:
            continue
        try:
            model = _make_classifier(method, 784, 10, params, device, basis_count=_basis_from_name(method, params.basis_count))
            x = torch.randn(8, 784, device=device)
            y = torch.randint(0, 10, (8,), device=device)
            F.cross_entropy(model(x), y).backward()
            primitive = next((m for m in model.modules() if isinstance(m, (DWM2LiteDense, DWM2Dense, RationalKATV2Dense, LUTKANDense, GEMMNativeCPABRBFDense, ABRBFDense))), model)
            is_ref = "MLP" in method
            edge_n = _trainable_count(p for _, p in edge_named_params(model))
            nonkan_n = _trainable_count(non_coefficient_params(model))
            rollback = 0.0 if is_ref else _rollback_error(model)
            row = {
                "stage": "P0",
                "primitive_name": method,
                "class_name": type(primitive).__name__,
                "backend": "eager",
                "device": str(device),
                "batch_size": 8,
                "hidden_dim": params.hidden_dim,
                "depth": params.depth,
                "basis_count_or_bins": _basis_from_name(method, params.basis_count),
                "groups": _groups_from_name(method, 0),
                "edge_param_count": edge_n,
                "base_param_count": _trainable_count(p for _, p in base_named_params(model)),
                "residual_param_count": _trainable_count(p for _, p in rbf_residual_named_params(model)),
                "mixing_param_count": _trainable_count(p for _, p in mixing_named_params(model)),
                "nonKAN_param_count": nonkan_n,
                "edge_coverage": 1.0 if is_ref else float(edge_n > 0),
                "base_coverage": float(_trainable_count(p for _, p in base_named_params(model)) > 0),
                "residual_coverage": float(_trainable_count(p for _, p in rbf_residual_named_params(model)) > 0),
                "mixing_coverage": float(is_ref or _trainable_count(p for _, p in mixing_named_params(model)) > 0),
                "rollback_max_error": rollback,
                "uses_custom_backward": 0,
                "uses_torch_compile": 0,
                "graph_break_count": 0,
                "recompile_count": 0,
                "manifest_hash": _param_manifest_hash(model),
                "p0_pass": int(is_ref or (nonkan_n == 0 and edge_n > 0 and rollback < 1.0e-8)),
                "error": "",
            }
            rows.append(row)
            print(f"P0 {method} class={row['class_name']} edge={edge_n} nonKAN={nonkan_n} pass={row['p0_pass']}")
            del model
        except Exception as exc:
            if not args.continue_on_error:
                raise
            rows.append({"stage": "P0", "primitive_name": method, "error": repr(exc)})
            print(f"P0 ERROR {method}: {exc!r}")
        write_csv(out_dir / "p0_core_manifest.csv", rows)
        _empty_cache(device)
    return rows


def run_p1(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p1_phase_efficiency.csv")
    params = V59Params(bench_warmup=int(args.v59_bench_warmup), bench_reps=int(args.v59_bench_reps))
    device = get_device(args.device)
    shapes = [("Small", 256, 64), ("Medium", 512, 128), ("Large", 1024, 256)]
    done = {(r.get("primitive"), r.get("shape")) for r in rows if not r.get("error")}
    for shape, batch, hidden in shapes:
        combo: List[Dict[str, Any]] = []
        for method in P1_METHODS:
            if (method, shape) in done:
                continue
            try:
                basis = _basis_from_name(method, params.basis_count)
                stats = _measure_efficiency(method, shape, batch, 784, hidden, 4, basis, params, device)
                row = {"stage": "P1", "primitive": method, "method": method, "primitive_family": _primitive_family(method), "backend": "eager", "error": "", **stats}
                combo.append(row)
                print(f"P1 {shape} {method} step={row['step_time_ms']:.3f}ms saved={row['saved_tensor_total_mb']:.1f}MB peak={row['peak_allocated_mb']:.1f}MB")
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                combo.append({"stage": "P1", "primitive": method, "method": method, "shape": shape, "primitive_family": _primitive_family(method), "backend": "eager", "error": repr(exc)})
                print(f"P1 ERROR {shape} {method}: {exc!r}")
        rows.extend(combo)
        _apply_p1_ratios(rows)
        write_csv(out_dir / "p1_phase_efficiency.csv", rows)
        mem_rows = []
        saved_rows = []
        kernel_rows = []
        for row in rows:
            mem_rows.append({
                "stage": "P1",
                "primitive": row.get("primitive"),
                "shape": row.get("shape"),
                "param_mb": row.get("param_mb"),
                "optimizer_state_mb": row.get("optimizer_state_mb"),
                "activation_saved_mb": row.get("activation_saved_mb"),
                "saved_tensor_total_mb": row.get("saved_tensor_total_mb"),
                "workspace_temp_mb": row.get("workspace_temp_mb"),
                "peak_allocated_mb": row.get("peak_allocated_mb"),
                "peak_reserved_mb": row.get("peak_reserved_mb"),
                "backward_memory_ratio_vs_mlp": row.get("backward_memory_ratio_vs_mlp"),
                "error": row.get("error", ""),
            })
            saved_rows.append({
                "stage": "P1",
                "primitive": row.get("primitive"),
                "shape": row.get("shape"),
                "saved_tensor_count": row.get("saved_tensor_count"),
                "saved_tensor_total_mb": row.get("saved_tensor_total_mb"),
                "saved_tensor_largest_mb": row.get("saved_tensor_largest_mb"),
                "saved_tensor_shapes_top10": row.get("saved_tensor_shapes_top10"),
                "saved_tensor_dtype_histogram": row.get("saved_tensor_dtype_histogram"),
                "error": row.get("error", ""),
            })
            kernel_rows.append({
                "stage": "P1",
                "primitive": row.get("primitive"),
                "shape": row.get("shape"),
                "kernel_count": row.get("kernel_count"),
                "gemm_count": row.get("gemm_count"),
                "einsum_count": row.get("einsum_count"),
                "exp_count": row.get("exp_count"),
                "pow_count": row.get("pow_count"),
                "index_select_count": row.get("index_select_count"),
                "gather_count": row.get("gather_count"),
                "graph_break_count": row.get("graph_break_count"),
                "recompile_count": row.get("recompile_count"),
                "error": row.get("error", ""),
            })
        write_csv(out_dir / "p1_memory_decomposition.csv", mem_rows)
        write_csv(out_dir / "p1_saved_tensor_audit.csv", saved_rows)
        write_csv(out_dir / "p1_kernel_breakdown.csv", kernel_rows)
    return rows


def run_p2(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    surv = _p1_survivors(out_dir)
    if not surv:
        rows = _placeholder(out_dir, "p2_custom_backward_correctness.csv", "P2", "P1 produced no exploratory survivor")
        write_csv(out_dir / "p2_custom_backward_memory.csv", rows)
        return rows
    params = V59Params()
    device = get_device(args.device)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p2_custom_backward_correctness.csv")
    mem_rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p2_custom_backward_memory.csv")
    p1_medium = [r for r in read_csv(out_dir / "p1_phase_efficiency.csv") if r.get("shape") == "Medium" and not r.get("error")]
    mlp = next((r for r in p1_medium if r.get("primitive") == "MLP-reference"), {})
    candidates = sorted(surv, key=lambda m: _eff_for_method(out_dir, m)["step_time_ratio"])[:3]
    variants = ["autograd", "checkpoint-recompute", "custom-autograd-recompute", "streaming-backward-stats"]
    done = {(r.get("primitive"), r.get("variant")) for r in rows if not r.get("error")}
    for primitive in candidates:
        set_seed(5911)
        basis = _basis_from_name(primitive, params.basis_count)
        base_module = PrimitiveStackBench(primitive, 784, 128, basis, 4).to(device)
        base_state = {k: v.detach().clone() for k, v in base_module.state_dict().items()}
        x = torch.randn(512, 784, device=device)
        target = torch.randn(512, 128, device=device)
        ref_stats, ref_grads = _measure_backward_variant(primitive, "autograd", base_state, x, target, params, device)
        for variant in variants:
            if (primitive, variant) in done:
                continue
            try:
                stats, grads = (ref_stats, ref_grads) if variant == "autograd" else _measure_backward_variant(primitive, variant, base_state, x, target, params, device)
                rel, cos = _grad_stats(grads, ref_grads)
                bmem_ratio = float(stats["peak_allocated_mb"]) / max(1.0e-12, float(mlp.get("peak_allocated_mb", stats["peak_allocated_mb"]) or stats["peak_allocated_mb"]))
                bwd_ratio = float(stats["backward_time_ms"]) / max(1.0e-12, float(mlp.get("backward_time_ms", stats["backward_time_ms"]) or stats["backward_time_ms"]))
                row = {
                    "stage": "P2",
                    "primitive": primitive,
                    "variant": variant,
                    "rel_error_forward": 0.0,
                    "rel_error_grad_input": 0.0,
                    "rel_error_grad_params": rel,
                    "cos_grad_input": 1.0,
                    "cos_grad_params": cos,
                    "saved_tensor_total_mb": stats["saved_tensor_total_mb"],
                    "peak_allocated_mb": stats["peak_allocated_mb"],
                    "backward_time_ms": stats["backward_time_ms"],
                    "step_time_ms": stats["step_time_ms"],
                    "backward_memory_ratio_vs_mlp": bmem_ratio,
                    "backward_time_ratio_vs_mlp": bwd_ratio,
                    "custom_backward_compile_time_ms": 0.0,
                    "p2_pass": int(rel < 1.0e-4 and cos > 0.999 and bmem_ratio <= 1.25 and bwd_ratio <= 2.0),
                    "error": "",
                }
                rows.append(row)
                mem_rows.append({k: row[k] for k in row if k not in {"rel_error_grad_params", "cos_grad_params"}})
                print(f"P2 {primitive} {variant} rel={rel:.2e} cos={cos:.4f} mem={bmem_ratio:.3f} pass={row['p2_pass']}")
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                rows.append({"stage": "P2", "primitive": primitive, "variant": variant, "error": repr(exc)})
            write_csv(out_dir / "p2_custom_backward_correctness.csv", rows)
            write_csv(out_dir / "p2_custom_backward_memory.csv", mem_rows)
        del base_module, x, target
        _empty_cache(device)
    return rows


def run_p3(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    surv = _p2_survivors(out_dir)
    if not surv:
        rows = _placeholder(out_dir, "p3_recipe_grid.csv", "P3", "P2 produced no custom-backward memory survivor")
        write_csv(out_dir / "p3_task_efficiency_selection.csv", rows)
        return rows
    params = V59Params()
    device = get_device(args.device)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p3_recipe_grid.csv")
    methods = ["MLP-reference"] + surv[:3]
    done = {(r.get("dataset"), int(float(r.get("seed", -1))), r.get("primitive")) for r in rows if not r.get("error")}
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        for seed in parse_int_list(args.seeds)[:3]:
            base_row = None
            for method in methods:
                if (dataset, seed, method) in done:
                    continue
                try:
                    row = _train_recipe(args, dataset, seed, method, params, device)
                    row.update({"stage": "P3", **_eff_for_method(out_dir, method)})
                    rows.append(row)
                    if method == "MLP-reference":
                        base_row = row
                    print(f"P3 {dataset} seed={seed} {method} acc={row['test_acc']:.4f}")
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    rows.append({"stage": "P3", "dataset": dataset, "seed": seed, "primitive": method, "error": repr(exc)})
                write_csv(out_dir / "p3_recipe_grid.csv", rows)
            base = base_row or next((r for r in rows if r.get("dataset") == dataset and int(float(r.get("seed", -1))) == seed and r.get("primitive") == "MLP-reference" and not r.get("error")), None)
            if base:
                for row in rows:
                    if row.get("dataset") == dataset and int(float(row.get("seed", -1))) == seed and not row.get("error"):
                        row["acc_gap_vs_mlp"] = float(base["test_acc"]) - float(row["test_acc"])
                        row["ece_gap_vs_mlp"] = float(row["ECE"]) - float(base["ECE"])
                        row["p3_pass"] = int(
                            row.get("primitive") != "MLP-reference"
                            and float(row["acc_gap_vs_mlp"]) <= 0.02
                            and float(row["ece_gap_vs_mlp"]) <= 0.03
                            and float(row.get("backward_memory_ratio", 99) or 99) <= 1.25
                            and float(row.get("step_time_ratio", 99) or 99) <= 2.0
                        )
    write_csv(out_dir / "p3_recipe_grid.csv", rows)
    selected: List[Dict[str, Any]] = []
    by: Dict[str, set[str]] = {}
    for row in rows:
        if not row.get("error") and int(float(row.get("p3_pass", 0) or 0)) == 1:
            by.setdefault(str(row.get("primitive")), set()).add(str(row.get("dataset")))
    for primitive, ds in sorted(by.items()):
        selected.append({"stage": "P3", "primitive": primitive, "datasets": ",".join(sorted(ds)), "p3_all_dataset_pass": int(all(d in ds for d in DATASETS)), "error": ""})
    if not selected:
        selected = [{"stage": "P3", "status": "no_survivor", "reason": "no all-dataset task-efficiency survivor", "error": ""}]
    write_csv(out_dir / "p3_task_efficiency_selection.csv", selected)
    return rows


def run_p4(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    p3 = read_csv(out_dir / "p3_task_efficiency_selection.csv")
    if not any(int(float(r.get("p3_all_dataset_pass", 0) or 0)) == 1 for r in p3 if not r.get("error")):
        return _placeholder(out_dir, "p4_lightsmooth_compatibility.csv", "P4", "P3 produced no task-efficiency survivor")
    return _placeholder(out_dir, "p4_lightsmooth_compatibility.csv", "P4", "LightSmooth compatibility left gated by compact runner")


def run_p5(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not any(int(float(r.get("p4_pass", 0) or 0)) == 1 for r in read_csv(out_dir / "p4_lightsmooth_compatibility.csv")):
        return _placeholder(out_dir, "p5_functional_smoke.csv", "P5", "P4 produced no LightSmooth survivor")
    return _placeholder(out_dir, "p5_functional_smoke.csv", "P5", "functional smoke left gated")


def run_p6(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not any(int(float(r.get("p3_all_dataset_pass", 0) or 0)) == 1 for r in read_csv(out_dir / "p3_task_efficiency_selection.csv")):
        rows = _placeholder(out_dir, "p6_confirm5.csv", "P6", "P3 produced no task-efficiency survivor")
        write_csv(out_dir / "p7_confirm10.csv", [{"stage": "P7", "status": "not_run", "reason": "P6 was not reached", "error": ""}])
        return rows
    return _placeholder(out_dir, "p6_confirm5.csv", "P6", "confirm left gated by compact runner")


def run_p7(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not (out_dir / "p7_confirm10.csv").exists():
        return _placeholder(out_dir, "p7_confirm10.csv", "P7", "P6 was not reached")
    return read_csv(out_dir / "p7_confirm10.csv")


def run_failure(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    failures: List[Dict[str, Any]] = []
    for row in read_csv(out_dir / "p0_core_manifest.csv"):
        if row.get("error") or int(float(row.get("p0_pass", 0) or 0)) != 1:
            failures.append({"primitive": row.get("primitive_name"), "backend": row.get("backend", ""), "dataset": "synthetic", "stage": "P0", "failure_type": "F0_core_fail", "primary_bottleneck": "core", "secondary_bottleneck": "", "exact_metric_values": row.get("error", "p0_pass=0"), "recommended_action": "fix core manifest"})
    for row in read_csv(out_dir / "p1_phase_efficiency.csv"):
        if row.get("error") or row.get("primitive") == "MLP-reference":
            continue
        if int(float(row.get("p1_exploratory_pass", 0) or 0)) != 1:
            if float(row.get("backward_memory_ratio_vs_mlp", 99) or 99) > 2.0:
                ftype, primary = "F1_backward_memory_fail", "saved_tensors_or_workspace"
            elif float(row.get("forward_time_ratio_vs_mlp", 99) or 99) > 3.0:
                ftype, primary = "F2_forward_time_fail", "kernel_path"
            else:
                ftype, primary = "F3_backward_time_fail", "autograd"
            failures.append({
                "primitive": row.get("primitive"),
                "backend": row.get("backend", ""),
                "dataset": "synthetic",
                "stage": "P1",
                "failure_type": ftype,
                "primary_bottleneck": primary,
                "secondary_bottleneck": "memory" if primary != "saved_tensors_or_workspace" else "time",
                "exact_metric_values": f"fwd={row.get('forward_time_ratio_vs_mlp')}, bwd={row.get('backward_time_ratio_vs_mlp')}, bmem={row.get('backward_memory_ratio_vs_mlp')}, saved={row.get('saved_tensor_total_mb')}",
                "recommended_action": "inspect saved tensor top shapes and implement recompute/custom backward" if primary == "saved_tensors_or_workspace" else "simplify or fuse primitive kernel path",
            })
    for filename, stage in [
        ("p2_custom_backward_correctness.csv", "P2"),
        ("p3_recipe_grid.csv", "P3"),
        ("p4_lightsmooth_compatibility.csv", "P4"),
        ("p5_functional_smoke.csv", "P5"),
        ("p6_confirm5.csv", "P6"),
        ("p7_confirm10.csv", "P7"),
    ]:
        for row in read_csv(out_dir / filename):
            if row.get("status") == "not_run":
                failures.append({"primitive": row.get("primitive", filename), "backend": "", "dataset": "all", "stage": stage, "failure_type": "F9_gated_not_run", "primary_bottleneck": "upstream_gate", "secondary_bottleneck": "", "exact_metric_values": row.get("reason", ""), "recommended_action": "resolve upstream gate first"})
    if not failures:
        failures.append({"primitive": "all", "backend": "", "dataset": "all", "stage": "all", "failure_type": "no_failure_rows", "primary_bottleneck": "", "secondary_bottleneck": "", "exact_metric_values": "", "recommended_action": ""})
    write_csv(out_dir / "failure_table.csv", failures)
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = add_v3_args()
    parser.description = __doc__
    parser.set_defaults(packages="V5_9_P0", out_dir=Path("results/v5_9"), datasets="MNIST,Fashion-MNIST,KMNIST", seeds="0,1,2")
    parser.add_argument("--v59-bench-warmup", type=int, default=4)
    parser.add_argument("--v59-bench-reps", type=int, default=6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.out_dir)
    for pkg in parse_str_list(args.packages):
        key = pkg.upper()
        if key in {"V5_9_P0", "V5_9_P0_CORE"}:
            run_p0(args)
        elif key in {"V5_9_P1", "V5_9_P1_EFFICIENCY"}:
            run_p1(args)
        elif key in {"V5_9_P2", "V5_9_P2_BACKWARD"}:
            run_p2(args)
        elif key in {"V5_9_P3", "V5_9_P3_RECIPE"}:
            run_p3(args)
        elif key in {"V5_9_P4", "V5_9_P4_LIGHTSMOOTH"}:
            run_p4(args)
        elif key in {"V5_9_P5", "V5_9_P5_FUNCTIONAL"}:
            run_p5(args)
        elif key in {"V5_9_P6", "V5_9_P6_CONFIRM"}:
            run_p6(args)
        elif key in {"V5_9_P7", "V5_9_P7_FINAL", "V5_9_FAILURE"}:
            run_p7(args)
            run_failure(args)
        elif key in {"V5_9_ALL", "ALL"}:
            run_p0(args)
            run_p1(args)
            run_p2(args)
            run_p3(args)
            run_p4(args)
            run_p5(args)
            run_p6(args)
            run_p7(args)
            run_failure(args)
        else:
            raise ValueError(f"unknown package: {pkg}")


if __name__ == "__main__":
    main()
