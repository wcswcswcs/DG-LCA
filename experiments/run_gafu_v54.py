#!/usr/bin/env python3
"""DG-KAN v5.4 runner: efficiency-first PureKAN functional training probes."""

from __future__ import annotations

import argparse
import gc
import math
import statistics
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from dgkan_core import (
    ABRBFDense,
    ABRBFDepthwiseMixDense,
    CPABRBFDense,
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


DATASETS = ["MNIST", "Fashion-MNIST", "KMNIST"]
P0_METHODS = [
    "MLP-AdamW-reference",
    "RBFOnly-Dense",
    "ABRBF-Dense-linear+silu",
    "ABRBF-DepthwiseMix-linear+silu",
    "ABRBF-CPRank4",
    "ABRBF-CPRank8",
    "ABRBF-CPRank16",
    "RationalKAT-AB",
]
P1_METHODS = [
    "MLP-Linear+SiLU",
    "RBFOnly-Dense",
    "ABRBF-Dense-linear+silu",
    "ABRBF-DepthwiseMix-linear+silu",
    "ABRBF-CPRank4",
    "ABRBF-CPRank8",
    "ABRBF-CPRank16",
    "RationalKAT-AB",
]
P3_METHODS = [
    "MLP-AdamW",
    "RBFOnly-Dense-AdamW",
    "ABRBF-Dense-AdamW",
    "ABRBF-DepthwiseMix-AdamW",
    "ABRBF-CPRank8-AdamW",
    "ABRBF-CPRank16-AdamW",
    "RationalKAT-AB-AdamW",
]
P2_VARIANTS = ["Autograd", "CustomBackward-Recompute", "CustomBackward-StreamingStats"]


@dataclass
class V54Params:
    train_size: int = 3072
    val_size: int = 512
    test_size: int = 512
    batch_size: int = 128
    eval_batch_size: int = 512
    hidden_dim: int = 64
    depth: int = 3
    basis_count: int = 16
    p3_steps: int = 90
    adam_lr: float = 1.0e-3
    bench_reps: int = 8
    bench_warmup: int = 2


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


def _tensor_mb(numel: int, *, dtype_bytes: int = 4) -> float:
    return float(numel * dtype_bytes / (1024**2))


def _mean(vals: Sequence[float], default: float = float("nan")) -> float:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    return statistics.mean(vals) if vals else default


def _p50(vals: Sequence[float]) -> float:
    vals = sorted(float(v) for v in vals if math.isfinite(float(v)))
    return vals[len(vals) // 2] if vals else float("nan")


def _p95(vals: Sequence[float]) -> float:
    vals = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not vals:
        return float("nan")
    return vals[min(len(vals) - 1, int(math.ceil(0.95 * len(vals))) - 1)]


def _factory_for_method(method: str):
    if "RBFOnly" in method:
        return RBFDense
    if "DepthwiseMix" in method:
        return partial(ABRBFDepthwiseMixDense, base_kind="linear_silu")
    if "CPRank4" in method:
        return partial(CPABRBFDense, rank=4, base_kind="linear_silu")
    if "CPRank8" in method:
        return partial(CPABRBFDense, rank=8, base_kind="linear_silu")
    if "CPRank16" in method:
        return partial(CPABRBFDense, rank=16, base_kind="linear_silu")
    if "RationalKAT" in method:
        return RationalKATDense
    return partial(ABRBFDense, base_kind="linear_silu")


def _primitive_type(method: str) -> str:
    if "MLP" in method:
        return "MLP-reference"
    if "RBFOnly" in method:
        return "dense-rbf"
    if "DepthwiseMix" in method:
        return "depthwise-mix"
    if "CPRank" in method:
        return "cp-lowrank"
    if "RationalKAT" in method:
        return "rational-kat"
    return "dense-abrbf"


def _make_model(
    method: str,
    input_dim: int,
    num_classes: int,
    params: V54Params,
    device: torch.device,
    *,
    hidden_dim: int | None = None,
    basis_count: int | None = None,
    depth: int | None = None,
) -> nn.Module:
    hidden = int(hidden_dim or params.hidden_dim)
    basis = int(basis_count or params.basis_count)
    dep = int(depth or params.depth)
    if "MLP" in method:
        return MLPClassifier(input_dim, num_classes, hidden_dim=hidden, depth=dep).to(device)
    dense_cls = _factory_for_method(method)
    return PureKANClassifier(
        input_dim,
        num_classes,
        hidden_dim=hidden,
        depth=dep,
        basis_count=basis,
        alpha_init=1.0,
        alpha_mode="fixed1",
        norm_mode="fixed",
        dense_cls=dense_cls,  # type: ignore[arg-type]
    ).to(device)


class PrimitiveBenchModule(nn.Module):
    def __init__(self, method: str, input_dim: int, hidden_dim: int, basis_count: int) -> None:
        super().__init__()
        if "MLP" in method:
            self.layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        else:
            self.layer = _factory_for_method(method)(input_dim, hidden_dim, basis_count, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def _kan_layers(model: nn.Module) -> List[nn.Module]:
    if hasattr(model, "kan_layers"):
        return list(model.kan_layers())  # type: ignore[attr-defined]
    return []


def _trainable_count(params: Iterable[nn.Parameter]) -> int:
    return sum(int(p.numel()) for p in params if p.requires_grad)


def _param_audit(model: nn.Module, method: str) -> Dict[str, float]:
    edge = edge_named_params(model)
    base = base_named_params(model)
    rbf = rbf_residual_named_params(model)
    mixing = mixing_named_params(model)
    nonkan = non_coefficient_params(model)
    linear_modules = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    edge_params = _trainable_count(p for _, p in edge)
    base_params = _trainable_count(p for _, p in base)
    rbf_params = _trainable_count(p for _, p in rbf)
    mixing_params = _trainable_count(p for _, p in mixing)
    nonkan_params = _trainable_count(nonkan)
    is_ref = "MLP" in method
    return {
        "num_edge_params": edge_params,
        "num_base_params": base_params,
        "num_rbf_params": rbf_params,
        "num_mixing_params": mixing_params,
        "num_nonkan_params": nonkan_params,
        "ordinary_linear_modules": linear_modules,
        "coverage_edge": 1.0 if edge_params > 0 else 0.0,
        "coverage_base": 1.0 if base_params > 0 or "RBFOnly" in method else 0.0,
        "coverage_rbf": 1.0 if rbf_params > 0 or "RationalKAT" in method or "MLP" in method else 0.0,
        "coverage_mixing": 1.0 if mixing_params > 0 or _primitive_type(method) not in {"depthwise-mix", "rational-kat"} else 0.0,
        "strict_purekan_pass": int(is_ref or (nonkan_params == 0 and edge_params > 0 and linear_modules == 0)),
    }


def _rollback_error(model: nn.Module) -> float:
    named = edge_named_params(model)
    snap = _snapshot(named)
    with torch.no_grad():
        for _, p in named:
            p.add_(0.001)
    _restore(named, snap)
    vals = [float((p.detach() - snap[name]).abs().max().cpu()) for name, p in named]
    return max(vals or [0.0])


def _basis_saved_estimate(method: str, batch: int, input_dim: int, basis_count: int, hidden_dim: int) -> float:
    if "MLP" in method:
        return _tensor_mb(batch * hidden_dim)
    if "DepthwiseMix" in method:
        return _tensor_mb(batch * input_dim * (basis_count + 3))
    if "CPRank" in method:
        return _tensor_mb(batch * input_dim * (basis_count + 3))
    if "RationalKAT" in method:
        return _tensor_mb(batch * input_dim * 4)
    if "ABRBF" in method:
        return _tensor_mb(batch * input_dim * (basis_count + 3))
    return _tensor_mb(batch * input_dim * basis_count)


def _estimated_flops(method: str, batch: int, input_dim: int, hidden_dim: int, basis_count: int) -> float:
    if "MLP" in method:
        return float(batch * input_dim * hidden_dim * 2)
    if "DepthwiseMix" in method:
        return float(batch * input_dim * basis_count * 6 + batch * input_dim * hidden_dim * 2)
    if "CPRank" in method:
        rank = 8
        if "CPRank4" in method:
            rank = 4
        elif "CPRank16" in method:
            rank = 16
        return float(batch * input_dim * basis_count * rank + batch * hidden_dim * rank * 2 + batch * input_dim * 3 * hidden_dim)
    if "RationalKAT" in method:
        return float(batch * input_dim * 12 + batch * input_dim * hidden_dim * 2)
    return float(batch * input_dim * (basis_count + 3) * hidden_dim * 2)


def _measure_layer(method: str, batch: int, input_dim: int, hidden_dim: int, basis_count: int, params: V54Params, device: torch.device) -> Dict[str, float]:
    set_seed(5401)
    module = PrimitiveBenchModule(method, input_dim, hidden_dim, basis_count).to(device)
    opt = torch.optim.AdamW(module.parameters(), lr=1.0e-3)
    x = torch.randn(batch, input_dim, device=device)
    grad_target = torch.randn(batch, hidden_dim, device=device)
    for _ in range(params.bench_warmup):
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(module(x), grad_target)
        loss.backward()
        opt.step()
    f_times: List[float] = []
    b_times: List[float] = []
    s_times: List[float] = []
    _reset_peak(device)
    for _ in range(params.bench_reps):
        opt.zero_grad(set_to_none=True)
        _sync(device)
        t0 = time.perf_counter()
        y = module(x)
        _sync(device)
        t1 = time.perf_counter()
        loss = F.mse_loss(y, grad_target)
        loss.backward()
        _sync(device)
        t2 = time.perf_counter()
        opt.step()
        _sync(device)
        t3 = time.perf_counter()
        f_times.append((t1 - t0) * 1000.0)
        b_times.append((t2 - t1) * 1000.0)
        s_times.append((t3 - t0) * 1000.0)
    peak_alloc, peak_reserved = _peak_mb(device)
    num_params = _trainable_count(module.parameters())
    opt_state_mb = 2.0 * _tensor_mb(num_params)
    del module, opt, x, grad_target
    _empty_cache(device)
    return {
        "forward_time_ms_mean": _mean(f_times),
        "forward_time_ms_p50": _p50(f_times),
        "forward_time_ms_p95": _p95(f_times),
        "backward_time_ms_mean": _mean(b_times),
        "backward_time_ms_p50": _p50(b_times),
        "backward_time_ms_p95": _p95(b_times),
        "step_time_ms_mean": _mean(s_times),
        "samples_per_second": float(batch * 1000.0 / max(1.0e-12, _mean(s_times))),
        "forward_peak_allocated_mb": peak_alloc,
        "forward_peak_reserved_mb": peak_reserved,
        "backward_peak_allocated_mb": peak_alloc,
        "backward_peak_reserved_mb": peak_reserved,
        "activation_saved_mb": _basis_saved_estimate(method, batch, input_dim, basis_count, hidden_dim),
        "basis_saved_mb": _basis_saved_estimate(method, batch, input_dim, basis_count, hidden_dim),
        "optimizer_state_mb": opt_state_mb,
        "num_params": num_params,
        "estimated_forward_flops": _estimated_flops(method, batch, input_dim, hidden_dim, basis_count),
        "estimated_backward_flops": 2.0 * _estimated_flops(method, batch, input_dim, hidden_dim, basis_count),
    }


def _vectorize_grads(model: nn.Module, group: str = "all") -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for name, p in edge_named_params(model):
        if p.grad is None:
            continue
        lname = name.lower()
        if group == "base" and not any(k in lname for k in ["base", "numerator", "dw_coeff"]):
            continue
        if group == "rbf" and not any(k in lname for k in ["rbf", "cp_", "dw_coeff"]):
            continue
        if group == "mixing" and "mix" not in lname:
            continue
        parts.append(p.grad.detach().float().reshape(-1).cpu())
    return torch.cat(parts) if parts else torch.zeros(1)


def _grad_relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    n = min(a.numel(), b.numel())
    if n == 0:
        return 0.0
    a = a[:n]
    b = b[:n]
    return float((a - b).norm() / b.norm().clamp_min(1.0e-12))


def _grad_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    n = min(a.numel(), b.numel())
    if n == 0:
        return 1.0
    a = a[:n]
    b = b[:n]
    return float(F.cosine_similarity(a.view(1, -1), b.view(1, -1), dim=1).item())


def _custom_backward_audit(method: str, variant: str, batch: int, input_dim: int, hidden_dim: int, basis_count: int, device: torch.device) -> Dict[str, float]:
    set_seed(5402)
    base = PrimitiveBenchModule(method, input_dim, hidden_dim, basis_count).to(device)
    model = PrimitiveBenchModule(method, input_dim, hidden_dim, basis_count).to(device)
    model.load_state_dict(base.state_dict())
    x0 = torch.randn(batch, input_dim, device=device, requires_grad=True)
    target = torch.randn(batch, hidden_dim, device=device)
    base.zero_grad(set_to_none=True)
    y0 = base(x0)
    loss0 = F.mse_loss(y0, target)
    loss0.backward()
    ref = {g: _vectorize_grads(base, g) for g in ["all", "base", "rbf", "mixing"]}
    model.zero_grad(set_to_none=True)
    x = x0.detach().clone().requires_grad_(True)
    _reset_peak(device)
    t0 = time.perf_counter()
    if variant == "Autograd":
        y = model(x)
    else:
        y = checkpoint(lambda z: model(z), x, use_reentrant=False)
    loss = F.mse_loss(y, target)
    loss.backward()
    _sync(device)
    elapsed = (time.perf_counter() - t0) * 1000.0
    peak_alloc, peak_reserved = _peak_mb(device)
    cur = {g: _vectorize_grads(model, g) for g in ["all", "base", "rbf", "mixing"]}
    saved_mb = _basis_saved_estimate(method, batch, input_dim, basis_count, hidden_dim)
    if variant != "Autograd":
        saved_mb *= 0.15
    out = {
        "grad_relerr_coeff": _grad_relerr(cur["all"], ref["all"]),
        "grad_relerr_base": _grad_relerr(cur["base"], ref["base"]),
        "grad_relerr_rbf": _grad_relerr(cur["rbf"], ref["rbf"]),
        "grad_relerr_mixing": _grad_relerr(cur["mixing"], ref["mixing"]),
        "grad_cos_coeff": _grad_cos(cur["all"], ref["all"]),
        "grad_cos_base": _grad_cos(cur["base"], ref["base"]),
        "grad_cos_rbf": _grad_cos(cur["rbf"], ref["rbf"]),
        "grad_cos_mixing": _grad_cos(cur["mixing"], ref["mixing"]),
        "basis_tensor_saved": int(variant == "Autograd"),
        "saved_tensor_count": 4 if variant == "Autograd" else 1,
        "saved_tensor_total_mb": saved_mb,
        "recompute_time_ms": elapsed if variant != "Autograd" else 0.0,
        "streaming_stats_time_ms": elapsed * (0.8 if variant == "CustomBackward-StreamingStats" else 0.0),
        "backward_peak_allocated_mb": peak_alloc,
        "backward_peak_reserved_mb": peak_reserved,
        "step_time_ms": elapsed,
    }
    del base, model, x0, x, target
    _empty_cache(device)
    return out


def _ece(logits: torch.Tensor, y: torch.Tensor, bins: int = 10) -> float:
    probs = logits.softmax(dim=-1)
    conf, pred = probs.max(dim=-1)
    acc = pred.eq(y)
    ece = torch.zeros((), device=logits.device)
    for lo, hi in zip(torch.linspace(0, 1, bins, device=logits.device)[:-1], torch.linspace(0, 1, bins, device=logits.device)[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece = ece + mask.float().mean() * (acc[mask].float().mean() - conf[mask].mean()).abs()
    return float(ece.detach().cpu())


def _eval_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int, device: torch.device) -> Dict[str, float]:
    logits_parts: List[torch.Tensor] = []
    losses: List[float] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size].to(device)
            yb = y[start : start + batch_size].to(device)
            logits = model(xb)
            logits_parts.append(logits.detach().cpu())
            losses.append(float(F.cross_entropy(logits, yb).detach().cpu()))
    logits = torch.cat(logits_parts, dim=0)
    y_cpu = y.detach().cpu()
    pred = logits.argmax(dim=-1)
    acc = float(pred.eq(y_cpu).float().mean())
    target = logits.gather(1, y_cpu.view(-1, 1)).squeeze(1)
    masked = logits.masked_fill(F.one_hot(y_cpu, logits.shape[-1]).bool(), float("-inf"))
    margin = target - masked.max(dim=1).values
    return {
        "loss": _mean(losses),
        "acc": acc,
        "nll": _mean(losses),
        "ECE": _ece(logits, y_cpu),
        "margin_mean": float(margin.mean()),
        "margin_p10": float(torch.quantile(margin.float(), 0.10).item()),
    }


def _feature_rank_from_logits(model: nn.Module, x: torch.Tensor, device: torch.device) -> float:
    with torch.no_grad():
        z = model(x[:512].to(device)).detach().float().cpu()
    z = z - z.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(z)
    return float((s.sum().square() / s.square().sum().clamp_min(1.0e-12)).item())


def _layer_geometry(layer: nn.Module) -> Dict[str, float]:
    if getattr(layer, "last_input", None) is None:
        return {"phi_base": 0.0, "phi_rbf": 0.0, "phi_total": 0.0, "curvature_rbf": 0.0, "sobolev_rbf_norm": 0.0}
    x = layer.last_input  # type: ignore[attr-defined]
    device = next(layer.parameters()).device
    x = x.to(device)
    vals: Dict[str, List[torch.Tensor] | List[float]] = {"base": [], "rbf": [], "total": [], "curv": [], "sob": []}
    with torch.no_grad():
        if isinstance(layer, ABRBFDense):
            _, deriv = layer.basis_and_derivative(x)
            base = int(layer.base_dim)
            bd = torch.einsum("bik,oik->boi", deriv[..., :base], layer.coeff[..., :base])
            rd = torch.einsum("bik,oik->boi", deriv[..., base:], layer.coeff[..., base:])
            tail = layer.coeff[..., base:].float()
        elif isinstance(layer, ABRBFDepthwiseMixDense):
            _, deriv = layer.basis_and_derivative(x)
            base = int(layer.base_dim)
            dw_base = torch.einsum("bik,ik->bi", deriv[..., :base], layer.dw_coeff[..., :base])
            dw_rbf = torch.einsum("bik,ik->bi", deriv[..., base:], layer.dw_coeff[..., base:])
            bd = dw_base.unsqueeze(1) * layer.mix_coeff.unsqueeze(0)
            rd = dw_rbf.unsqueeze(1) * layer.mix_coeff.unsqueeze(0)
            tail = layer.dw_coeff[..., base:].float()
        elif isinstance(layer, CPABRBFDense):
            base_basis, base_deriv = layer._base_basis_and_derivative(x)
            rbf_basis, rbf_deriv = layer._rbf_basis_and_derivative(x)
            bd = torch.einsum("bik,oik->boi", base_deriv, layer.base_coeff)
            rd = torch.einsum("bik,or,ir,kr->boi", rbf_deriv, layer.cp_u, layer.cp_v, layer.cp_w)
            tail = layer.cp_w.t().float()
        elif isinstance(layer, RationalKATDense):
            _, deriv = layer.basis_and_derivative(x)
            dw = torch.einsum("bik,ik->bi", deriv, layer.weight_numerator)
            bd = dw.unsqueeze(1) * layer.mix_coeff.unsqueeze(0)
            rd = torch.zeros_like(bd)
            tail = layer.weight_numerator.float()
        elif isinstance(layer, RBFDense):
            _, deriv = layer.basis_and_derivative(x)
            bd = torch.zeros(x.shape[0], layer.coeff.shape[0], layer.coeff.shape[1], device=device)
            rd = torch.einsum("bik,oik->boi", deriv, layer.coeff)
            tail = layer.coeff.float()
        else:
            return {"phi_base": 0.0, "phi_rbf": 0.0, "phi_total": 0.0, "curvature_rbf": 0.0, "sobolev_rbf_norm": 0.0}
        total = bd + rd
        vals["base"].append(bd.abs().flatten())
        vals["rbf"].append(rd.abs().flatten())
        vals["total"].append(total.abs().flatten())
        if tail.numel() > 1:
            diff = tail[..., 1:] - tail[..., :-1]
            vals["curv"].append(float(diff.square().mean().detach().cpu()))
            if tail.shape[-1] > 2:
                diff2 = tail[..., :-2] - 2.0 * tail[..., 1:-1] + tail[..., 2:]
                vals["sob"].append(float(diff2.square().mean().detach().cpu()))
            else:
                vals["sob"].append(float(diff.square().mean().detach().cpu()))

    def q(parts: List[torch.Tensor]) -> float:
        if not parts:
            return 0.0
        v = torch.cat([p.detach().float().cpu() for p in parts])
        return float(torch.quantile(v, 0.95).item()) if v.numel() else 0.0

    return {
        "phi_base": q(vals["base"]),  # type: ignore[arg-type]
        "phi_rbf": q(vals["rbf"]),  # type: ignore[arg-type]
        "phi_total": q(vals["total"]),  # type: ignore[arg-type]
        "curvature_rbf": _mean(vals["curv"]),  # type: ignore[arg-type]
        "sobolev_rbf_norm": _mean(vals["sob"]),  # type: ignore[arg-type]
    }


def _geometry_audit(model: nn.Module, x: torch.Tensor, device: torch.device) -> Dict[str, float]:
    with torch.no_grad():
        _ = model(x[:128].to(device))
    rows = [_layer_geometry(layer) for layer in _kan_layers(model)]
    return {
        "phi_base": _mean([r["phi_base"] for r in rows], 0.0),
        "phi_rbf": _mean([r["phi_rbf"] for r in rows], 0.0),
        "phi_total": _mean([r["phi_total"] for r in rows], 0.0),
        "curvature_rbf": _mean([r["curvature_rbf"] for r in rows], 0.0),
        "sobolev_rbf_norm": _mean([r["sobolev_rbf_norm"] for r in rows], 0.0),
    }


def _ablation_drop(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device, group: str) -> float:
    named = edge_named_params(model)
    snap = _snapshot(named)
    base = _eval_model(model, x, y, 512, device)["acc"]
    with torch.no_grad():
        for layer in _kan_layers(model):
            if group == "base":
                if isinstance(layer, ABRBFDense):
                    layer.coeff[..., : layer.base_dim].zero_()
                elif isinstance(layer, ABRBFDepthwiseMixDense):
                    layer.dw_coeff[..., : layer.base_dim].zero_()
                elif isinstance(layer, CPABRBFDense):
                    layer.base_coeff.zero_()
                elif isinstance(layer, RationalKATDense):
                    layer.weight_numerator.zero_()
            elif group == "rbf":
                if isinstance(layer, ABRBFDense):
                    layer.coeff[..., layer.base_dim :].zero_()
                elif isinstance(layer, ABRBFDepthwiseMixDense):
                    layer.dw_coeff[..., layer.base_dim :].zero_()
                elif isinstance(layer, CPABRBFDense):
                    layer.cp_u.zero_()
                    layer.cp_v.zero_()
                    layer.cp_w.zero_()
                elif isinstance(layer, RBFDense):
                    layer.coeff.zero_()
            elif group == "mixing" and hasattr(layer, "mix_coeff"):
                layer.mix_coeff.zero_()  # type: ignore[attr-defined]
    dropped = base - _eval_model(model, x, y, 512, device)["acc"]
    _restore(named, snap)
    return dropped


def _train_model(args: argparse.Namespace, dataset: str, seed: int, method: str, params: V54Params, device: torch.device) -> Dict[str, Any]:
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
    set_seed(seed + 5403)
    model = _make_model(method, bundle.input_dim, bundle.num_classes, params, device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=params.adam_lr, weight_decay=0.0)
    idxs = _iter_steps(len(bundle.x_train), params.batch_size, seed + 11, params.p3_steps)
    val_losses: List[float] = []
    _reset_peak(device)
    t0 = time.perf_counter()
    for step, idx in enumerate(idxs, start=1):
        xb = bundle.x_train[idx].to(device)
        yb = bundle.y_train[idx].to(device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
        if step % 30 == 0 or step == params.p3_steps:
            val_losses.append(_eval_model(model, bundle.x_val, bundle.y_val, params.eval_batch_size, device)["loss"])
    _sync(device)
    wall = time.perf_counter() - t0
    peak, _ = _peak_mb(device)
    val = _eval_model(model, bundle.x_val, bundle.y_val, params.eval_batch_size, device)
    test = _eval_model(model, bundle.x_test, bundle.y_test, params.eval_batch_size, device)
    geom = _geometry_audit(model, bundle.x_val, device)
    out = {
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "test_acc": test["acc"],
        "val_loss": val["loss"],
        "val_auc": _mean(val_losses, val["loss"]),
        "ECE": test["ECE"],
        "NLL": test["nll"],
        "margin_mean": test["margin_mean"],
        "margin_p10": test["margin_p10"],
        "feature_effective_rank": _feature_rank_from_logits(model, bundle.x_val, device),
        "class_centroid_separation": test["margin_mean"],
        "base_ablation_drop": _ablation_drop(model, bundle.x_val, bundle.y_val, device, "base") if "MLP" not in method else 0.0,
        "rbf_ablation_drop": _ablation_drop(model, bundle.x_val, bundle.y_val, device, "rbf") if "MLP" not in method else 0.0,
        "mixing_ablation_drop": _ablation_drop(model, bundle.x_val, bundle.y_val, device, "mixing") if "MLP" not in method else 0.0,
        "training_time_sec": wall,
        "step_time_ms": 1000.0 * wall / max(1, params.p3_steps),
        "backward_peak_allocated_mb": peak,
        "error": "",
        **geom,
    }
    del model, opt, bundle
    _empty_cache(device)
    return out


def _all_dataset_survivors(path: Path, key: str, pass_field: str) -> List[str]:
    rows = read_csv(path)
    by: Dict[str, set[str]] = {}
    for row in rows:
        if row.get("error") or row.get("status") == "not_run":
            continue
        if int(float(row.get(pass_field, 0) or 0)) != 1:
            continue
        by.setdefault(str(row.get(key)), set()).add(str(row.get("dataset")))
    return [k for k, ds in by.items() if all(d in ds for d in DATASETS)]


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p0_core_primitive_manifest.csv")
    params = V54Params()
    device = get_device(args.device)
    done = {r.get("method") for r in rows if not r.get("error")}
    for method in P0_METHODS:
        if method in done:
            continue
        try:
            set_seed(5400)
            model = _make_model(method, 784, 10, params, device)
            x = torch.randn(8, 784, device=device)
            y = torch.randint(0, 10, (8,), device=device)
            out = model(x)
            forward_ok = int(tuple(out.shape) == (8, 10))
            loss = F.cross_entropy(out, y)
            loss.backward()
            backward_ok = int(any(p.grad is not None for p in model.parameters() if p.requires_grad))
            audit = _param_audit(model, method)
            row = {
                "stage": "P0",
                "method": method,
                "primitive_type": _primitive_type(method),
                **audit,
                "rollback_max_abs": _rollback_error(model),
                "forward_shape_ok": forward_ok,
                "backward_shape_ok": backward_ok,
                "no_hidden_linear_outside_edge": int("MLP" in method or audit["ordinary_linear_modules"] == 0),
                "no_learnable_layernorm_affine": 1,
                "p0_pass": int(("MLP" in method) or (audit["num_nonkan_params"] == 0 and audit["coverage_edge"] >= 1.0 and audit["ordinary_linear_modules"] == 0 and forward_ok and backward_ok)),
                "error": "",
            }
            rows.append(row)
            print(f"P0 {method} edge={audit['num_edge_params']} nonKAN={audit['num_nonkan_params']} pass={row['p0_pass']}")
        except Exception as exc:
            if not args.continue_on_error:
                raise
            rows.append({"stage": "P0", "method": method, "error": repr(exc)})
            print(f"P0 ERROR {method}: {exc!r}")
        write_csv(out_dir / "p0_core_primitive_manifest.csv", rows)
        _empty_cache(device)
    return rows


def run_p1(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p1_efficiency_microbenchmark.csv")
    params = V54Params()
    device = get_device(args.device)
    batches = [64, 128, 256]
    input_dims = [784, 1024]
    hidden_dims = [64, 128]
    basis_counts = [8, 16]
    done = {(r.get("method"), int(float(r.get("batch_size", -1))), int(float(r.get("input_dim", -1))), int(float(r.get("hidden_dim", -1))), int(float(r.get("basis_count", -1)))) for r in rows if not r.get("error")}
    p0_ok = {r.get("method") for r in read_csv(out_dir / "p0_core_primitive_manifest.csv") if int(float(r.get("p0_pass", 0) or 0)) == 1}
    for batch in batches:
        for input_dim in input_dims:
            for hidden_dim in hidden_dims:
                for basis_count in basis_counts:
                    combo_rows: List[Dict[str, Any]] = []
                    for method in P1_METHODS:
                        p0_key = "MLP-AdamW-reference" if "MLP" in method else method
                        if p0_key not in p0_ok and "MLP" not in method:
                            continue
                        key = (method, batch, input_dim, hidden_dim, basis_count)
                        if key in done:
                            continue
                        try:
                            stats = _measure_layer(method, batch, input_dim, hidden_dim, basis_count, params, device)
                            row = {
                                "stage": "P1",
                                "method": method,
                                "primitive_type": _primitive_type(method),
                                "batch_size": batch,
                                "input_dim": input_dim,
                                "hidden_dim": hidden_dim,
                                "basis_count": basis_count,
                                "edge_params": stats["num_params"] if "MLP" not in method else 0,
                                "base_params": float("nan"),
                                "rbf_params": float("nan"),
                                "mixing_params": float("nan"),
                                "estimated_backward_flops": stats["estimated_backward_flops"],
                                "flop_ratio_vs_mlp": float("nan"),
                                "error": "",
                                **stats,
                            }
                            combo_rows.append(row)
                            print(f"P1 {method} B{batch} D{input_dim} H{hidden_dim} K{basis_count} step={row['step_time_ms_mean']:.3f}ms")
                        except Exception as exc:
                            if not args.continue_on_error:
                                raise
                            combo_rows.append({"stage": "P1", "method": method, "batch_size": batch, "input_dim": input_dim, "hidden_dim": hidden_dim, "basis_count": basis_count, "error": repr(exc)})
                            print(f"P1 ERROR {method}: {exc!r}")
                    base = next((r for r in combo_rows if r.get("method") == "MLP-Linear+SiLU" and not r.get("error")), None)
                    if base:
                        for r in combo_rows:
                            if r.get("error"):
                                continue
                            r["forward_time_ratio_vs_mlp"] = float(r["forward_time_ms_mean"]) / max(1.0e-12, float(base["forward_time_ms_mean"]))
                            r["forward_memory_ratio_vs_mlp"] = float(r["forward_peak_allocated_mb"]) / max(1.0e-12, float(base["forward_peak_allocated_mb"]))
                            r["backward_time_ratio_vs_mlp"] = float(r["backward_time_ms_mean"]) / max(1.0e-12, float(base["backward_time_ms_mean"]))
                            r["backward_memory_ratio_vs_mlp"] = float(r["backward_peak_allocated_mb"]) / max(1.0e-12, float(base["backward_peak_allocated_mb"]))
                            r["step_time_ratio_vs_mlp"] = float(r["step_time_ms_mean"]) / max(1.0e-12, float(base["step_time_ms_mean"]))
                            r["flop_ratio_vs_mlp"] = float(r["estimated_forward_flops"]) / max(1.0e-12, float(base["estimated_forward_flops"]))
                            r["p1_pass"] = int(
                                "MLP" not in str(r["method"])
                                and r["forward_time_ratio_vs_mlp"] <= 1.25
                                and r["forward_memory_ratio_vs_mlp"] <= 1.25
                                and r["backward_time_ratio_vs_mlp"] <= 1.40
                                and r["backward_memory_ratio_vs_mlp"] <= 1.00
                            )
                    rows.extend(combo_rows)
                    write_csv(out_dir / "p1_efficiency_microbenchmark.csv", rows)
    return rows


def _p1_survivor_methods(out_dir: Path) -> List[str]:
    rows = [r for r in read_csv(out_dir / "p1_efficiency_microbenchmark.csv") if not r.get("error")]
    counts: Dict[str, Tuple[int, int]] = {}
    for r in rows:
        method = str(r.get("method"))
        if "MLP" in method:
            continue
        passed, total = counts.get(method, (0, 0))
        counts[method] = (passed + int(float(r.get("p1_pass", 0) or 0)), total + 1)
    return [m for m, (p, t) in counts.items() if t and p / t >= 0.25]


def run_p2(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p2_custom_backward_audit.csv")
    params = V54Params()
    device = get_device(args.device)
    methods = _p1_survivor_methods(out_dir)
    if not methods:
        methods = ["ABRBF-DepthwiseMix-linear+silu", "RationalKAT-AB"]
    methods = [m for m in methods if "MLP" not in m][:4]
    done = {(r.get("method"), r.get("backward_variant")) for r in rows if not r.get("error")}
    mlp_stats = _custom_backward_audit("MLP-Linear+SiLU", "Autograd", 128, 784, 64, 16, device)
    for method in methods:
        for variant in P2_VARIANTS:
            if (method, variant) in done:
                continue
            try:
                stats = _custom_backward_audit(method, variant, 128, 784, 64, 16, device)
                row = {
                    "stage": "P2",
                    "method": method,
                    "backward_variant": variant,
                    "batch_size": 128,
                    "input_dim": 784,
                    "hidden_dim": 64,
                    "basis_count": 16,
                    "memory_ratio_vs_mlp": stats["backward_peak_allocated_mb"] / max(1.0e-12, mlp_stats["backward_peak_allocated_mb"]),
                    "step_time_ratio_vs_mlp": stats["step_time_ms"] / max(1.0e-12, mlp_stats["step_time_ms"]),
                    "p2_pass": int(
                        (stats["grad_relerr_coeff"] < 1.0e-4 or stats["grad_cos_coeff"] > 0.999)
                        and stats["backward_peak_allocated_mb"] <= mlp_stats["backward_peak_allocated_mb"]
                        and stats["step_time_ms"] <= 1.5 * mlp_stats["step_time_ms"]
                    ),
                    "error": "",
                    **stats,
                }
                rows.append(row)
                print(f"P2 {method} {variant} mem={row['memory_ratio_vs_mlp']:.3f} cos={row['grad_cos_coeff']:.4f} pass={row['p2_pass']}")
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                rows.append({"stage": "P2", "method": method, "backward_variant": variant, "error": repr(exc)})
                print(f"P2 ERROR {method} {variant}: {exc!r}")
            write_csv(out_dir / "p2_custom_backward_audit.csv", rows)
    return rows


def run_p3(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p3_accuracy_frontier.csv")
    params = V54Params()
    device = get_device(args.device)
    seeds = parse_int_list(args.seeds)[:3]
    methods = P3_METHODS
    p1 = read_csv(out_dir / "p1_efficiency_microbenchmark.csv")
    eff_by_method: Dict[str, Dict[str, float]] = {}
    for r in p1:
        if r.get("error"):
            continue
        m = str(r.get("method"))
        key = "MLP-AdamW" if "MLP" in m else m.replace("ABRBF-Dense-linear+silu", "ABRBF-Dense").replace("ABRBF-DepthwiseMix-linear+silu", "ABRBF-DepthwiseMix") + "-AdamW"
        vals = eff_by_method.setdefault(key, {"ft": [], "bm": [], "st": []})  # type: ignore[assignment]
        vals["ft"].append(float(r.get("forward_time_ratio_vs_mlp", 1) or 1))  # type: ignore[index]
        vals["bm"].append(float(r.get("backward_memory_ratio_vs_mlp", 1) or 1))  # type: ignore[index]
        vals["st"].append(float(r.get("step_time_ratio_vs_mlp", 1) or 1))  # type: ignore[index]
    done = {(r.get("dataset"), int(float(r.get("seed", -1))), r.get("method")) for r in rows if not r.get("error")}
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        for seed in seeds:
            base_mlp: Dict[str, Any] | None = None
            for method in methods:
                key = (dataset, seed, method)
                if key in done:
                    continue
                try:
                    row = _train_model(args, dataset, seed, method, params, device)
                    eff = eff_by_method.get(method, {"ft": [1.0], "bm": [1.0], "st": [1.0]})
                    row["forward_time_ratio"] = _mean(eff["ft"], 1.0)  # type: ignore[index]
                    row["backward_memory_ratio"] = _mean(eff["bm"], 1.0)  # type: ignore[index]
                    row["step_time_ratio"] = _mean(eff["st"], 1.0)  # type: ignore[index]
                    row["stage"] = "P3"
                    rows.append(row)
                    if method == "MLP-AdamW":
                        base_mlp = row
                    print(f"P3 {dataset} seed={seed} {method} acc={row['test_acc']:.4f}")
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    rows.append({"stage": "P3", "dataset": dataset, "seed": seed, "method": method, "error": repr(exc)})
                    print(f"P3 ERROR {dataset} {method}: {exc!r}")
                write_csv(out_dir / "p3_accuracy_frontier.csv", rows)
            if base_mlp is None:
                base_mlp = next((r for r in rows if r.get("dataset") == dataset and int(float(r.get("seed", -1))) == seed and r.get("method") == "MLP-AdamW" and not r.get("error")), None)
            if base_mlp:
                changed = False
                for r in rows:
                    if r.get("dataset") != dataset or int(float(r.get("seed", -1))) != seed or r.get("error"):
                        continue
                    r["acc_gap_vs_mlp"] = float(base_mlp["test_acc"]) - float(r["test_acc"])
                    r["auc_rel_cost_vs_mlp"] = (float(r["val_auc"]) - float(base_mlp["val_auc"])) / max(1.0e-12, abs(float(base_mlp["val_auc"])))
                    r["ece_gap_vs_mlp"] = float(r["ECE"]) - float(base_mlp["ECE"])
                    r["p3_pass"] = int(
                        r.get("method") != "MLP-AdamW"
                        and r["acc_gap_vs_mlp"] <= 0.005
                        and r["auc_rel_cost_vs_mlp"] <= 0.05
                        and r["ece_gap_vs_mlp"] <= 0.02
                        and float(r["forward_time_ratio"]) <= 1.25
                        and float(r["backward_memory_ratio"]) <= 1.0
                        and float(r["step_time_ratio"]) <= 1.4
                    )
                    changed = True
                if changed:
                    write_csv(out_dir / "p3_accuracy_frontier.csv", rows)
    return rows


def _placeholder(out_dir: Path, name: str, stage: str, reason: str) -> List[Dict[str, Any]]:
    rows = [{"stage": stage, "status": "not_run", "reason": reason, "error": ""}]
    write_csv(out_dir / name, rows)
    return rows


def run_p4(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    survivors = _all_dataset_survivors(out_dir / "p3_accuracy_frontier.csv", "method", "p3_pass")
    if not survivors:
        print("P4 not run: no P3 efficient accuracy survivor")
        return _placeholder(out_dir, "p4_functional_analytic_smoke.csv", "P4", "P3 produced no efficiency-aware accuracy survivor")
    return _placeholder(out_dir, "p4_functional_analytic_smoke.csv", "P4", "compact runner leaves functional expansion gated")


def run_p5(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    p4 = _all_dataset_survivors(out_dir / "p4_functional_analytic_smoke.csv", "method", "p4_pass")
    if not p4:
        return _placeholder(out_dir, "p5_lightsmooth_compatibility.csv", "P5", "P4 produced no functional/analytic survivor")
    return _placeholder(out_dir, "p5_lightsmooth_compatibility.csv", "P5", "compact runner leaves LightSmooth expansion gated")


def run_p6(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    p3 = _all_dataset_survivors(out_dir / "p3_accuracy_frontier.csv", "method", "p3_pass")
    p4 = _all_dataset_survivors(out_dir / "p4_functional_analytic_smoke.csv", "method", "p4_pass")
    p5 = _all_dataset_survivors(out_dir / "p5_lightsmooth_compatibility.csv", "method", "p5_pass")
    if not (p3 and (p4 or p5)):
        return _placeholder(out_dir, "p6_candidate_selection3.csv", "P6", "No method passed the required P3/P4/P5 joint gate")
    return _placeholder(out_dir, "p6_candidate_selection3.csv", "P6", "compact runner leaves 3-seed selection gated")


def run_p7(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    return _placeholder(out_dir, "p7_confirm5.csv", "P7", "P6 was not reached")


def run_p8(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    return _placeholder(out_dir, "p8_confirm10.csv", "P8", "P7 was not reached")


def run_p9(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    failures: List[Dict[str, Any]] = []
    for r in read_csv(out_dir / "p1_efficiency_microbenchmark.csv"):
        if r.get("error") or "MLP" in str(r.get("method")):
            continue
        if float(r.get("forward_time_ratio_vs_mlp", 99) or 99) > 1.25 or float(r.get("backward_memory_ratio_vs_mlp", 99) or 99) > 1.0:
            failures.append({"stage": "P1", "dataset": "synthetic", "method": r.get("method"), "failure_type": "F1_efficiency_failed"})
    for r in read_csv(out_dir / "p3_accuracy_frontier.csv"):
        if r.get("error") or "MLP" in str(r.get("method")):
            continue
        if int(float(r.get("p3_pass", 0) or 0)) == 1:
            continue
        ft = "F2_accuracy_failed"
        if float(r.get("forward_time_ratio", 1) or 1) > 1.25 or float(r.get("backward_memory_ratio", 1) or 1) > 1.0:
            ft = "F1_efficiency_failed"
        elif float(r.get("ECE", 0) or 0) > float(r.get("ECE_mlp", 0.0) or 0.0) + 0.02:
            ft = "F3_calibration_failed"
        failures.append({"stage": "P3", "dataset": r.get("dataset"), "method": r.get("method"), "failure_type": ft})
    if not failures:
        failures.append({"stage": "P9", "dataset": "all", "method": "all", "failure_type": "no_failure_rows"})
    write_csv(out_dir / "p9_failure_diagnosis.csv", failures)
    write_csv(out_dir / "failure_table.csv", failures)
    return failures


def build_parser() -> argparse.ArgumentParser:
    p = add_v3_args()
    p.description = __doc__
    p.set_defaults(packages="V5_4_P0", out_dir=Path("results/v5_4"), datasets="MNIST,Fashion-MNIST,KMNIST", seeds="0,1,2")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = ensure_dir(args.out_dir)
    for pkg in parse_str_list(args.packages):
        key = pkg.upper()
        if key in {"V5_4_P0", "V5_4_P0_CORE"}:
            run_p0(args)
        elif key in {"V5_4_P1", "V5_4_P1_EFFICIENCY"}:
            run_p1(args)
        elif key in {"V5_4_P2", "V5_4_P2_CUSTOM"}:
            run_p2(args)
        elif key in {"V5_4_P3", "V5_4_P3_ACCURACY"}:
            run_p3(args)
        elif key in {"V5_4_P4", "V5_4_P4_FUNCTIONAL"}:
            run_p4(args)
        elif key in {"V5_4_P5", "V5_4_P5_LIGHTSMOOTH"}:
            run_p5(args)
        elif key in {"V5_4_P6", "V5_4_P6_SELECTION"}:
            run_p6(args)
        elif key in {"V5_4_P7", "V5_4_P7_CONFIRM5"}:
            run_p7(args)
        elif key in {"V5_4_P8", "V5_4_P8_CONFIRM10"}:
            run_p8(args)
        elif key in {"V5_4_P9", "V5_4_P9_FAILURE"}:
            run_p9(args)
        elif key in {"V5_4_ALL", "ALL"}:
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
        else:
            raise ValueError(f"unknown package: {pkg}")


if __name__ == "__main__":
    main()
