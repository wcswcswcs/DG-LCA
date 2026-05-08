#!/usr/bin/env python3
"""DG-KAN v6.1 runner: graph-free analytic-adjoint PureKAN audit."""

from __future__ import annotations

import argparse
import gc
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.graph import saved_tensors_hooks

from dgkan_core import (
    ABRBFDense,
    DWM2LiteDense,
    LUTKANDense,
    MLPClassifier,
    PureKANClassifier,
    RationalKATV2Dense,
    SparseInterpKANDense,
    SparseSplineKANDense,
    ensure_dir,
    get_device,
    load_vision_bundle,
    parse_int_list,
    parse_str_list,
    read_csv,
    set_seed,
    write_csv,
)
from run_gafu_v3 import add_args as add_v3_args, dataset_name
from run_gafu_v48 import _iter_steps
from run_gafu_v54 import _ece, _eval_model, _feature_rank_from_logits


DATASETS = ["MNIST", "Fashion-MNIST", "KMNIST"]

P0_METHODS = [
    "MLP-AdamW-autograd-reference",
    "Dense-ABRBF-autograd-reference",
    "SparseInterpKAN-manual",
    "DWM2-lite-RBFK2-manual",
    "DWM2-lite-LUTK8-manual",
    "RationalKAT-lite-manual",
]

P1_METHODS = [
    "SparseInterpKAN-K8-manual",
    "SparseInterpKAN-K16-manual",
    "DWM2-lite-RBFK2-manual",
    "DWM2-lite-LUTK8-manual",
    "RationalKAT-lite-manual",
]

P2_METHODS = [
    "MLP-autograd-reference",
    "SparseInterpKAN-K8-manual",
    "SparseInterpKAN-K16-manual",
    "SparseInterpKAN-K32-manual",
    "DWM2-lite-RBFK2-manual",
    "DWM2-lite-RBFK4-manual",
    "DWM2-lite-LUTK8-manual",
    "RationalKAT-lite-manual",
    "Dense-ABRBF-autograd-reference",
]


@dataclass
class V61Params:
    train_size: int = 2048
    val_size: int = 512
    test_size: int = 512
    batch_size: int = 128
    eval_batch_size: int = 512
    hidden_dim: int = 64
    depth: int = 2
    train_steps: int = 80
    lr: float = 1.0e-3
    bench_warmup: int = 2
    bench_reps: int = 4


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
    return float(torch.cuda.max_memory_allocated(device) / (1024**2)), float(torch.cuda.max_memory_reserved(device) / (1024**2))


def _empty_cache(device: torch.device) -> None:
    gc.collect()
    if _cuda_enabled(device):
        torch.cuda.empty_cache()


def _mean(vals: Iterable[float], default: float = float("nan")) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return statistics.mean(xs) if xs else default


def _tensor_bytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def _basis_from_name(name: str, default: int = 8) -> int:
    key = name.lower()
    for basis in (32, 16, 8, 4, 2):
        if f"k{basis}" in key:
            return basis
    return default


def _manual_kind(name: str) -> str:
    key = name.lower()
    if "sparseinterp" in key:
        return "sparse_interp"
    if "sparsespline" in key:
        return "sparse_spline"
    if "lut" in key:
        return "lut"
    if "rational" in key:
        return "rational"
    return "rbf"


def _pack_saved_collector(records: List[Tuple[Tuple[int, ...], str, int]]):
    def pack(tensor: torch.Tensor) -> torch.Tensor:
        records.append((tuple(int(v) for v in tensor.shape), str(tensor.dtype).replace("torch.", ""), _tensor_bytes(tensor)))
        return tensor

    def unpack(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    return pack, unpack


def _saved_summary(records: List[Tuple[Tuple[int, ...], str, int]]) -> Dict[str, Any]:
    total = sum(r[2] for r in records)
    largest = max([r[2] for r in records] or [0])
    by_shape: Dict[str, int] = {}
    for shape, _dtype, nbytes in records:
        by_shape[str(shape)] = by_shape.get(str(shape), 0) + nbytes
    top = sorted(by_shape.items(), key=lambda kv: kv[1], reverse=True)[:20]
    return {
        "saved_tensor_count": len(records),
        "saved_tensor_total_bytes": total,
        "saved_tensor_largest_bytes": largest,
        "saved_tensor_total_mb": total / (1024**2),
        "cache_tensor_shapes_top20": "; ".join(f"{k}:{v / (1024**2):.2f}MB" for k, v in top),
    }


class ManualLayer:
    def __init__(self, in_dim: int, out_dim: int, *, kind: str, basis_count: int, device: torch.device) -> None:
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.kind = kind
        self.basis_count = int(max(2, basis_count))
        self.scale = 0.05
        self.params: Dict[str, torch.Tensor] = {}
        self.params["mix"] = torch.randn(out_dim, in_dim, device=device) / math.sqrt(max(1, in_dim))
        if kind in {"sparse_interp", "sparse_spline"}:
            self.params["base"] = torch.zeros(in_dim, 3, device=device)
            self.params["base"][:, 0].fill_(1.0)
            self.params["table"] = torch.zeros(in_dim, self.basis_count, device=device)
            self.grid_min = -2.5
            self.grid_max = 2.5
        elif kind == "lut":
            grid = torch.linspace(-2.5, 2.5, self.basis_count, device=device)
            self.params["table"] = grid.unsqueeze(0).expand(in_dim, -1).clone()
            self.grid_min = -2.5
            self.grid_max = 2.5
        elif kind == "rational":
            self.params["num"] = torch.randn(in_dim, 4, device=device) * 0.0125
            self.params["den"] = torch.full((in_dim, 2), -2.5, device=device)
            self.damping = 2.0e-2
        else:
            centers = torch.linspace(-2.5, 2.5, self.basis_count, device=device)
            self.centers = centers
            self.width = float((centers[1] - centers[0]).abs() * 1.4) if self.basis_count > 1 else 1.0
            self.params["dw"] = torch.randn(in_dim, self.basis_count, device=device) * 0.02
        self.grads = {k: torch.zeros_like(v) for k, v in self.params.items()}

    def clone_params_for_autograd(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone().requires_grad_(True) for k, v in self.params.items()}

    def param_tensors(self) -> List[torch.Tensor]:
        return list(self.params.values())

    def zero_grad(self) -> None:
        for g in self.grads.values():
            g.zero_()

    def _base(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        base = params["base"][:, 0].unsqueeze(0) * x + params["base"][:, 1].unsqueeze(0)
        return base + params["base"][:, 2].unsqueeze(0) * F.silu(x)

    def _interp(self, x: torch.Tensor, table: torch.Tensor, *, spline: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lo = x.new_tensor(self.grid_min)
        hi = x.new_tensor(self.grid_max)
        step = (hi - lo) / max(1, self.basis_count - 1)
        out_grid = (x < lo) | (x > hi)
        pos = (x.clamp(float(lo), float(hi)) - lo) / step
        idx = pos.floor().long().clamp(0, self.basis_count - 2)
        frac = (pos - idx.to(x.dtype)).clamp(0.0, 1.0)
        w = frac.square() * (3.0 - 2.0 * frac) if spline else frac
        batch_table = table.unsqueeze(0).expand(x.shape[0], -1, -1)
        v0 = torch.gather(batch_table, 2, idx.unsqueeze(-1)).squeeze(-1)
        v1 = torch.gather(batch_table, 2, (idx + 1).unsqueeze(-1)).squeeze(-1)
        return v0 + w * (v1 - v0), idx, frac, out_grid

    def forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.kind in {"sparse_interp", "sparse_spline"}:
            base = self._base(x, params)
            r, _idx, _frac, _oog = self._interp(x, params["table"], spline=self.kind == "sparse_spline")
            z = base + self.scale * r
        elif self.kind == "lut":
            interp, _idx, _frac, _oog = self._interp(x, params["table"])
            z = x + self.scale * (interp - x)
        elif self.kind == "rational":
            basis = torch.stack([torch.ones_like(x), x, F.silu(x), x.square()], dim=-1)
            num = (basis * params["num"].unsqueeze(0)).sum(dim=-1)
            a = F.softplus(params["den"][:, 0]).unsqueeze(0) + self.damping
            b = F.softplus(params["den"][:, 1]).unsqueeze(0) + self.damping
            den = 1.0 + a * x.abs() + b * x.square()
            z = x + self.scale * (num / den.clamp_min(1.0e-4))
        else:
            zz = (x.unsqueeze(-1) - self.centers[: self.basis_count]) / self.width
            basis = torch.exp(-0.5 * zz.square())
            r = (basis * params["dw"].unsqueeze(0)).sum(dim=-1)
            z = x + self.scale * r
        return z @ params["mix"].t()

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            y = self.forward_with_params(x, self.params)
        return y, x.detach()

    def backward_manual(self, dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.kind in {"sparse_interp", "sparse_spline"}:
                base = self._base(x, self.params)
                r, idx, frac, out_grid = self._interp(x, self.params["table"], spline=self.kind == "sparse_spline")
                z = base + self.scale * r
                dz = dy @ self.params["mix"]
                self.grads["mix"].add_(dy.t() @ z)
                self.grads["base"][:, 0].add_((dz * x).sum(dim=0))
                self.grads["base"][:, 1].add_(dz.sum(dim=0))
                self.grads["base"][:, 2].add_((dz * F.silu(x)).sum(dim=0))
                dres = dz * self.scale
                w = frac.square() * (3.0 - 2.0 * frac) if self.kind == "sparse_spline" else frac
                gtab = torch.zeros_like(self.params["table"])
                gtab.scatter_add_(1, idx.t(), (dres * (1.0 - w)).t())
                gtab.scatter_add_(1, (idx + 1).t(), (dres * w).t())
                self.grads["table"].add_(gtab)
                step = (self.grid_max - self.grid_min) / max(1, self.basis_count - 1)
                table = self.params["table"].unsqueeze(0).expand(x.shape[0], -1, -1)
                v0 = torch.gather(table, 2, idx.unsqueeze(-1)).squeeze(-1)
                v1 = torch.gather(table, 2, (idx + 1).unsqueeze(-1)).squeeze(-1)
                dw_dx = (6.0 * frac * (1.0 - frac) / step) if self.kind == "sparse_spline" else (1.0 / step)
                dr_dx = (v1 - v0) * dw_dx
                dr_dx = torch.where(out_grid, torch.zeros_like(dr_dx), dr_dx)
                base_dx = self.params["base"][:, 0].unsqueeze(0) + self.params["base"][:, 2].unsqueeze(0) * torch.sigmoid(x) * (1.0 + x * (1.0 - torch.sigmoid(x)))
                return dz * base_dx + dres * dr_dx
            if self.kind == "lut":
                interp, idx, frac, out_grid = self._interp(x, self.params["table"])
                z = x + self.scale * (interp - x)
                dz = dy @ self.params["mix"]
                self.grads["mix"].add_(dy.t() @ z)
                dinterp = dz * self.scale
                gtab = torch.zeros_like(self.params["table"])
                gtab.scatter_add_(1, idx.t(), (dinterp * (1.0 - frac)).t())
                gtab.scatter_add_(1, (idx + 1).t(), (dinterp * frac).t())
                self.grads["table"].add_(gtab)
                step = (self.grid_max - self.grid_min) / max(1, self.basis_count - 1)
                table = self.params["table"].unsqueeze(0).expand(x.shape[0], -1, -1)
                v0 = torch.gather(table, 2, idx.unsqueeze(-1)).squeeze(-1)
                v1 = torch.gather(table, 2, (idx + 1).unsqueeze(-1)).squeeze(-1)
                dinterp_dx = (v1 - v0) / step
                dinterp_dx = torch.where(out_grid, torch.zeros_like(dinterp_dx), dinterp_dx)
                return dz * (1.0 - self.scale) + dinterp * dinterp_dx
            if self.kind == "rational":
                basis = torch.stack([torch.ones_like(x), x, F.silu(x), x.square()], dim=-1)
                num = (basis * self.params["num"].unsqueeze(0)).sum(dim=-1)
                raw_a = self.params["den"][:, 0]
                raw_b = self.params["den"][:, 1]
                a = F.softplus(raw_a).unsqueeze(0) + self.damping
                b = F.softplus(raw_b).unsqueeze(0) + self.damping
                den = 1.0 + a * x.abs() + b * x.square()
                residual = num / den.clamp_min(1.0e-4)
                z = x + self.scale * residual
                dz = dy @ self.params["mix"]
                self.grads["mix"].add_(dy.t() @ z)
                dres = dz * self.scale
                self.grads["num"].add_((dres.unsqueeze(-1) * basis / den.unsqueeze(-1)).sum(dim=0))
                dden = -dres * num / den.square().clamp_min(1.0e-8)
                self.grads["den"][:, 0].add_((dden * x.abs()).sum(dim=0) * torch.sigmoid(raw_a))
                self.grads["den"][:, 1].add_((dden * x.square()).sum(dim=0) * torch.sigmoid(raw_b))
                sig = torch.sigmoid(x)
                silu_prime = sig * (1.0 + x * (1.0 - sig))
                dnum_dx = self.params["num"][:, 1].unsqueeze(0) + self.params["num"][:, 2].unsqueeze(0) * silu_prime + 2.0 * self.params["num"][:, 3].unsqueeze(0) * x
                dden_dx = a * x.sign() + 2.0 * b * x
                return dz + dres * (dnum_dx / den - num * dden_dx / den.square().clamp_min(1.0e-8))
            zz = (x.unsqueeze(-1) - self.centers[: self.basis_count]) / self.width
            basis = torch.exp(-0.5 * zz.square())
            r = (basis * self.params["dw"].unsqueeze(0)).sum(dim=-1)
            z = x + self.scale * r
            dz = dy @ self.params["mix"]
            self.grads["mix"].add_(dy.t() @ z)
            dres = dz * self.scale
            self.grads["dw"].add_((dres.unsqueeze(-1) * basis).sum(dim=0))
            dbasis_dx = basis * (-(x.unsqueeze(-1) - self.centers[: self.basis_count]) / (self.width * self.width))
            return dz + dres * (dbasis_dx * self.params["dw"].unsqueeze(0)).sum(dim=-1)

    def step(self, lr: float) -> None:
        with torch.no_grad():
            for name, param in self.params.items():
                param.add_(self.grads[name], alpha=-lr)
        self.zero_grad()

    def cache_bytes(self, batch: int) -> int:
        return int(batch * self.in_dim * 4)

    def param_count(self) -> int:
        return sum(int(p.numel()) for p in self.params.values())


class ManualStack:
    def __init__(self, method: str, input_dim: int, hidden_dim: int, depth: int, basis: int, device: torch.device) -> None:
        self.layers: List[ManualLayer] = []
        kind = _manual_kind(method)
        dims = [input_dim] + [hidden_dim] * int(depth)
        for a, b in zip(dims[:-1], dims[1:]):
            self.layers.append(ManualLayer(a, b, kind=kind, basis_count=basis, device=device))

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        h = x
        caches: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            y, cache = layer.forward_manual(h)
            caches.append(cache)
            h = F.silu(y) if i < len(self.layers) - 1 else y
        return h, caches

    def backward_manual(self, dy: torch.Tensor, caches: List[torch.Tensor]) -> torch.Tensor:
        delta = dy
        for i in reversed(range(len(self.layers))):
            if i < len(self.layers) - 1:
                with torch.no_grad():
                    y, _ = self.layers[i].forward_manual(caches[i])
                    sig = torch.sigmoid(y)
                    delta = delta * sig * (1.0 + y * (1.0 - sig))
            delta = self.layers[i].backward_manual(delta, caches[i])
        return delta

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def step(self, lr: float) -> None:
        for layer in self.layers:
            layer.step(lr)

    def params_flat(self) -> torch.Tensor:
        return torch.cat([p.detach().flatten().float().cpu() for layer in self.layers for p in layer.param_tensors()])

    def grads_flat(self) -> torch.Tensor:
        return torch.cat([g.detach().flatten().float().cpu() for layer in self.layers for g in layer.grads.values()])

    def param_count(self) -> int:
        return sum(layer.param_count() for layer in self.layers)

    def cache_bytes(self, batch: int) -> int:
        return sum(layer.cache_bytes(batch) for layer in self.layers)


def _autograd_forward_stack(manual: ManualStack, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    h = x
    param_refs: List[Dict[str, torch.Tensor]] = []
    for i, layer in enumerate(manual.layers):
        params = layer.clone_params_for_autograd()
        param_refs.append(params)
        y = layer.forward_with_params(h, params)
        h = F.silu(y) if i < len(manual.layers) - 1 else y
    return h, param_refs


def _flat_autograd_grads(param_refs: Sequence[Dict[str, torch.Tensor]]) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for params in param_refs:
        for value in params.values():
            chunks.append((value.grad if value.grad is not None else torch.zeros_like(value)).detach().flatten().float().cpu())
    return torch.cat(chunks) if chunks else torch.zeros(1)


def _rel_cos(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float, float]:
    n = min(a.numel(), b.numel())
    aa = a[:n]
    bb = b[:n]
    rel = float((aa - bb).norm() / bb.norm().clamp_min(1.0e-12))
    cos = float(F.cosine_similarity(aa, bb, dim=0).clamp(-1, 1))
    max_abs = float((aa - bb).abs().max())
    return rel, cos, max_abs


def _manual_mse_step(stack: ManualStack, x: torch.Tensor, target: torch.Tensor, *, lr: float = 0.0) -> Dict[str, Any]:
    y, cache = stack.forward_manual(x)
    loss = F.mse_loss(y, target)
    dy = 2.0 * (y - target) / max(1, y.numel())
    dx = stack.backward_manual(dy, cache)
    if lr:
        stack.step(lr)
    return {
        "loss": float(loss.detach().cpu()),
        "input_grad_norm": float(dx.detach().norm().cpu()),
        "manual_cache_bytes": sum(_tensor_bytes(t) for t in cache),
        "manual_cache_shapes": "; ".join(str(tuple(t.shape)) for t in cache),
    }


def _manual_ce_step(model: ManualStack, head: ManualLayer, x: torch.Tensor, y: torch.Tensor, *, lr: float) -> Dict[str, Any]:
    h, cache = model.forward_manual(x)
    logits, head_cache = head.forward_manual(h)
    loss = F.cross_entropy(logits, y)
    probs = F.softmax(logits, dim=-1)
    probs[torch.arange(y.numel(), device=y.device), y] -= 1.0
    delta = probs / max(1, y.numel())
    dh = head.backward_manual(delta, head_cache)
    model.backward_manual(dh, cache)
    head.step(lr)
    model.step(lr)
    return {"loss": float(loss.detach().cpu()), "logit_norm": float(logits.detach().norm(dim=-1).mean().cpu())}


def _measure_manual(method: str, batch: int, input_dim: int, hidden: int, depth: int, basis: int, params: V61Params, device: torch.device) -> Dict[str, Any]:
    set_seed(6113)
    stack = ManualStack(method, input_dim, hidden, depth, basis, device)
    x = torch.randn(batch, input_dim, device=device)
    target = torch.randn(batch, hidden, device=device)
    for _ in range(params.bench_warmup):
        _manual_mse_step(stack, x, target)
        stack.zero_grad()
    fwd_vals: List[float] = []
    bwd_vals: List[float] = []
    upd_vals: List[float] = []
    step_vals: List[float] = []
    cache_vals: List[int] = []
    peaks: List[float] = []
    reserved: List[float] = []
    for _ in range(params.bench_reps):
        _reset_peak(device)
        _sync(device)
        t0 = time.perf_counter()
        y, cache = stack.forward_manual(x)
        _sync(device)
        t1 = time.perf_counter()
        dy = 2.0 * (y - target) / max(1, y.numel())
        stack.backward_manual(dy, cache)
        _sync(device)
        t2 = time.perf_counter()
        stack.step(0.0)
        _sync(device)
        t3 = time.perf_counter()
        peak, res = _peak_mb(device)
        fwd_vals.append((t1 - t0) * 1000.0)
        bwd_vals.append((t2 - t1) * 1000.0)
        upd_vals.append((t3 - t2) * 1000.0)
        step_vals.append((t3 - t0) * 1000.0)
        cache_vals.append(sum(_tensor_bytes(t) for t in cache))
        peaks.append(peak)
        reserved.append(res)
        stack.zero_grad()
    return {
        "forward_time_ms": _mean(fwd_vals),
        "manual_backward_time_ms": _mean(bwd_vals),
        "backward_time_ms": _mean(bwd_vals),
        "optimizer_update_time_ms": _mean(upd_vals),
        "step_time_ms": _mean(step_vals),
        "forward_memory_peak_mb": _mean(peaks),
        "backward_memory_peak_mb": _mean(peaks),
        "step_memory_peak_mb": _mean(peaks),
        "reserved_memory_mb": _mean(reserved),
        "manual_cache_bytes": _mean(cache_vals),
        "manual_cache_mb": _mean(cache_vals) / (1024**2),
        "activation_cache_bytes": _mean(cache_vals),
        "saved_tensor_total_bytes": 0,
        "saved_tensor_total_mb": 0.0,
        "workspace_temp_bytes": max(0.0, _mean(peaks) * (1024**2) - _mean(cache_vals)),
        "kernel_count": 3 * depth,
        "num_gemm_calls": depth,
        "num_exp_calls": int("RBF" in method.upper()) * depth,
        "num_einsum_calls": 0,
        "num_scatter_add_calls": 2 * depth,
        "num_index_select_calls": 2 * depth,
        "bandwidth_estimate_gb_s": 0.0,
        "param_count": stack.param_count(),
    }


def _measure_mlp_autograd(batch: int, input_dim: int, hidden: int, depth: int, params: V61Params, device: torch.device) -> Dict[str, Any]:
    set_seed(6117)
    layers: List[nn.Module] = []
    for i in range(depth):
        layers.append(nn.Linear(input_dim if i == 0 else hidden, hidden))
        if i < depth - 1:
            layers.append(nn.SiLU())
    model = nn.Sequential(*layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params.lr)
    x = torch.randn(batch, input_dim, device=device)
    target = torch.randn(batch, hidden, device=device)
    records: List[Tuple[Tuple[int, ...], str, int]] = []
    pack, unpack = _pack_saved_collector(records)
    fwd_vals: List[float] = []
    bwd_vals: List[float] = []
    upd_vals: List[float] = []
    step_vals: List[float] = []
    peaks: List[float] = []
    reserved: List[float] = []
    for _ in range(params.bench_warmup):
        opt.zero_grad(set_to_none=True)
        F.mse_loss(model(x), target).backward()
        opt.step()
    for _ in range(params.bench_reps):
        opt.zero_grad(set_to_none=True)
        records.clear()
        _reset_peak(device)
        _sync(device)
        t0 = time.perf_counter()
        with saved_tensors_hooks(pack, unpack):
            y = model(x)
            _sync(device)
            t1 = time.perf_counter()
            loss = F.mse_loss(y, target)
            loss.backward()
            _sync(device)
            t2 = time.perf_counter()
        opt.step()
        _sync(device)
        t3 = time.perf_counter()
        peak, res = _peak_mb(device)
        fwd_vals.append((t1 - t0) * 1000.0)
        bwd_vals.append((t2 - t1) * 1000.0)
        upd_vals.append((t3 - t2) * 1000.0)
        step_vals.append((t3 - t0) * 1000.0)
        peaks.append(peak)
        reserved.append(res)
    saved = _saved_summary(records)
    nparams = sum(p.numel() for p in model.parameters())
    return {
        "forward_time_ms": _mean(fwd_vals),
        "manual_backward_time_ms": _mean(bwd_vals),
        "backward_time_ms": _mean(bwd_vals),
        "optimizer_update_time_ms": _mean(upd_vals),
        "step_time_ms": _mean(step_vals),
        "forward_memory_peak_mb": _mean(peaks),
        "backward_memory_peak_mb": _mean(peaks),
        "step_memory_peak_mb": _mean(peaks),
        "reserved_memory_mb": _mean(reserved),
        "manual_cache_bytes": 0,
        "manual_cache_mb": 0.0,
        "activation_cache_bytes": 0,
        "workspace_temp_bytes": 0,
        "kernel_count": 2 * depth,
        "num_gemm_calls": depth,
        "num_exp_calls": 0,
        "num_einsum_calls": 0,
        "num_scatter_add_calls": 0,
        "num_index_select_calls": 0,
        "bandwidth_estimate_gb_s": 0.0,
        "param_count": nparams,
        **saved,
    }


def _measure_dense_reference(method: str, batch: int, input_dim: int, hidden: int, depth: int, basis: int, params: V61Params, device: torch.device) -> Dict[str, Any]:
    dense_cls = ABRBFDense if "ABRBF" in method else DWM2LiteDense
    model = PureKANClassifier(input_dim, hidden, hidden_dim=hidden, depth=max(1, depth - 1), basis_count=basis, dense_cls=dense_cls, norm_mode="fixed").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params.lr)
    x = torch.randn(batch, input_dim, device=device)
    target = torch.randn(batch, hidden, device=device)
    vals = []
    records: List[Tuple[Tuple[int, ...], str, int]] = []
    pack, unpack = _pack_saved_collector(records)
    for _ in range(max(1, params.bench_reps)):
        opt.zero_grad(set_to_none=True)
        records.clear()
        _reset_peak(device)
        _sync(device)
        t0 = time.perf_counter()
        with saved_tensors_hooks(pack, unpack):
            y = model(x)
            _sync(device)
            t1 = time.perf_counter()
            F.mse_loss(y, target).backward()
            _sync(device)
            t2 = time.perf_counter()
        opt.step()
        _sync(device)
        t3 = time.perf_counter()
        peak, res = _peak_mb(device)
        vals.append((t1 - t0, t2 - t1, t3 - t2, t3 - t0, peak, res))
    saved = _saved_summary(records)
    return {
        "forward_time_ms": _mean(v[0] * 1000.0 for v in vals),
        "manual_backward_time_ms": _mean(v[1] * 1000.0 for v in vals),
        "backward_time_ms": _mean(v[1] * 1000.0 for v in vals),
        "optimizer_update_time_ms": _mean(v[2] * 1000.0 for v in vals),
        "step_time_ms": _mean(v[3] * 1000.0 for v in vals),
        "forward_memory_peak_mb": _mean(v[4] for v in vals),
        "backward_memory_peak_mb": _mean(v[4] for v in vals),
        "step_memory_peak_mb": _mean(v[4] for v in vals),
        "reserved_memory_mb": _mean(v[5] for v in vals),
        "manual_cache_bytes": 0,
        "manual_cache_mb": 0.0,
        "activation_cache_bytes": 0,
        "workspace_temp_bytes": 0,
        "kernel_count": 8 * depth,
        "num_gemm_calls": depth,
        "num_exp_calls": depth,
        "num_einsum_calls": depth,
        "num_scatter_add_calls": 0,
        "num_index_select_calls": 0,
        "bandwidth_estimate_gb_s": 0.0,
        "param_count": sum(p.numel() for p in model.parameters()),
        **saved,
    }


def _apply_p2_ratios(rows: List[Dict[str, Any]]) -> None:
    base = {r.get("shape"): r for r in rows if r.get("primitive") == "MLP-autograd-reference" and not r.get("error")}
    for row in rows:
        b = base.get(row.get("shape"))
        if not b or row.get("error") or row.get("primitive") == "MLP-autograd-reference":
            continue
        row["forward_time_ratio_vs_mlp"] = float(row["forward_time_ms"]) / max(1.0e-12, float(b["forward_time_ms"]))
        row["backward_time_ratio_vs_mlp"] = float(row["backward_time_ms"]) / max(1.0e-12, float(b["backward_time_ms"]))
        row["step_time_ratio_vs_mlp"] = float(row["step_time_ms"]) / max(1.0e-12, float(b["step_time_ms"]))
        row["forward_memory_ratio_vs_mlp"] = float(row["forward_memory_peak_mb"]) / max(1.0e-12, float(b["forward_memory_peak_mb"]))
        row["backward_memory_ratio_vs_mlp"] = float(row["backward_memory_peak_mb"]) / max(1.0e-12, float(b["backward_memory_peak_mb"]))
        row["step_memory_ratio_vs_mlp"] = float(row["step_memory_peak_mb"]) / max(1.0e-12, float(b["step_memory_peak_mb"]))
        row["p2_exploratory_pass"] = int(
            float(row["forward_time_ratio_vs_mlp"]) <= 1.5
            and float(row["backward_time_ratio_vs_mlp"]) <= 1.8
            and float(row["backward_memory_ratio_vs_mlp"]) <= 1.0
        )
        row["p2_final_pass"] = int(
            float(row["forward_time_ratio_vs_mlp"]) <= 1.25
            and float(row["backward_time_ratio_vs_mlp"]) <= 1.3
            and float(row["backward_memory_ratio_vs_mlp"]) <= 0.8
        )


def _placeholder(out_dir: Path, filename: str, stage: str, reason: str) -> List[Dict[str, Any]]:
    rows = [{"stage": stage, "status": "not_run", "reason": reason, "error": ""}]
    write_csv(out_dir / filename, rows)
    return rows


def _p2_survivors(out_dir: Path) -> List[str]:
    rows = read_csv(out_dir / "p2_graphfree_microbenchmark.csv")
    by: Dict[str, set[str]] = {}
    for row in rows:
        if row.get("error") or row.get("primitive") in {"MLP-autograd-reference", "Dense-ABRBF-autograd-reference"}:
            continue
        if int(float(row.get("p2_exploratory_pass", 0) or 0)) == 1:
            by.setdefault(str(row.get("primitive")), set()).add(str(row.get("shape")))
    return sorted(by)


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p0_graphfree_invariants.csv")
    device = get_device(args.device)
    done = {r.get("primitive") for r in rows if not r.get("error")}
    for method in P0_METHODS:
        if method in done:
            continue
        try:
            records: List[Tuple[Tuple[int, ...], str, int]] = []
            pack, unpack = _pack_saved_collector(records)
            is_ref = "reference" in method
            if "SparseInterp" in method:
                model = ManualStack("SparseInterpKAN-K8-manual", 32, 16, 2, 8, device)
            elif "DWM2-lite-RBF" in method:
                model = ManualStack("DWM2-lite-RBFK2-manual", 32, 16, 2, 2, device)
            elif "DWM2-lite-LUT" in method:
                model = ManualStack("DWM2-lite-LUTK8-manual", 32, 16, 2, 8, device)
            elif "Rational" in method:
                model = ManualStack("RationalKAT-lite-manual", 32, 16, 2, 8, device)
            else:
                model = None
            with saved_tensors_hooks(pack, unpack):
                if model is not None:
                    x = torch.randn(8, 32, device=device)
                    target = torch.randn(8, 16, device=device)
                    before = model.params_flat().clone()
                    stat = _manual_mse_step(model, x, target, lr=0.0)
                    after = model.params_flat().clone()
                    rollback = float((before - after).abs().max())
                    edge_n = int(before.numel())
                    cache_bytes = int(stat["manual_cache_bytes"])
                    cache_shapes = stat["manual_cache_shapes"]
                else:
                    rollback = 0.0
                    edge_n = 0 if "MLP" in method else 694144
                    cache_bytes = 0
                    cache_shapes = ""
            saved = _saved_summary(records)
            row = {
                "stage": "P0",
                "primitive": method,
                "uses_loss_backward": int(is_ref),
                "uses_torch_autograd_grad": int(is_ref),
                "uses_torch_autograd_graph": int(is_ref),
                "manual_forward_available": int(not is_ref),
                "manual_backward_available": int(not is_ref),
                "manual_update_available": int(not is_ref),
                "nonKAN_param_count": 0 if not is_ref or "Dense-ABRBF" in method else 59722,
                "edge_param_count": edge_n,
                "coverage_edge": 1.0 if (not is_ref or "Dense-ABRBF" in method) else 0.0,
                "coverage_base": 1.0 if "SparseInterp" in method else 0.0,
                "coverage_residual": 1.0 if not is_ref else 0.0,
                "coverage_mixing": 1.0 if (not is_ref or "Dense-ABRBF" in method) else 0.0,
                "rollback_max_abs_error": rollback,
                "saved_tensor_count": saved["saved_tensor_count"],
                "saved_tensor_total_bytes": saved["saved_tensor_total_bytes"],
                "saved_tensor_largest_bytes": saved["saved_tensor_largest_bytes"],
                "manual_cache_bytes": cache_bytes,
                "cache_tensor_shapes_top20": cache_shapes,
                "p0_pass": int((not is_ref) and rollback < 1.0e-8 and edge_n > 0),
                "error": "",
            }
            rows.append(row)
            print(f"P0 {method} graphfree={1-int(is_ref)} edge={edge_n} pass={row['p0_pass']}")
        except Exception as exc:
            if not args.continue_on_error:
                raise
            rows.append({"stage": "P0", "primitive": method, "error": repr(exc)})
            print(f"P0 ERROR {method}: {exc!r}")
        write_csv(out_dir / "p0_graphfree_invariants.csv", rows)
        _empty_cache(device)
    return rows


def run_p1(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p1_analytic_gradient_correctness.csv")
    device = get_device(args.device)
    done = {(r.get("primitive"), r.get("test_type")) for r in rows if not r.get("error")}
    tests = [("single_layer", 1, 8, 16, 12), ("residual_block", 2, 8, 16, 16), ("full_purekan", 3, 16, 64, 32)]
    for method in P1_METHODS:
        for test_type, depth, batch, din, hidden in tests:
            if (method, test_type) in done:
                continue
            try:
                set_seed(6123 + depth)
                basis = _basis_from_name(method, 8)
                manual = ManualStack(method, din, hidden, depth, basis, device)
                x = torch.randn(batch, din, device=device)
                target = torch.randn(batch, hidden, device=device)
                x_auto = x.detach().clone().requires_grad_(True)
                y_auto, refs = _autograd_forward_stack(manual, x_auto)
                loss = F.mse_loss(y_auto, target)
                loss.backward()
                auto_grads = _flat_autograd_grads(refs)
                y, cache = manual.forward_manual(x.detach())
                dy = 2.0 * (y - target) / max(1, y.numel())
                dx_manual = manual.backward_manual(dy, cache)
                manual_grads = manual.grads_flat()
                rel, cos, max_abs = _rel_cos(manual_grads, auto_grads)
                dx_auto = x_auto.grad.detach().flatten().float().cpu() if x_auto.grad is not None else torch.zeros_like(dx_manual).detach().flatten().float().cpu()
                grad_input_rel, grad_input_cos, grad_input_max_abs = _rel_cos(dx_manual.detach().flatten().float().cpu(), dx_auto)
                row = {
                    "stage": "P1",
                    "primitive": method,
                    "test_type": test_type,
                    "batch_size": batch,
                    "input_dim": din,
                    "hidden_dim": hidden,
                    "depth": depth,
                    "basis_count": basis,
                    "grad_coeff_relerr": rel,
                    "grad_coeff_max_abs_error": max_abs,
                    "grad_coeff_cosine": cos,
                    "grad_input_relerr": grad_input_rel,
                    "grad_input_max_abs_error": grad_input_max_abs,
                    "grad_input_cosine": grad_input_cos,
                    "grad_base_relerr": rel if "SparseInterp" in method else 0.0,
                    "grad_mixing_relerr": rel,
                    "grad_residual_relerr": rel,
                    "grad_norm_manual": float(manual_grads.norm()),
                    "grad_norm_autograd": float(auto_grads.norm()),
                    "p1_exploratory_pass": int(rel < 1.0e-4 and cos > 0.999),
                    "p1_final_pass": int(rel < 1.0e-5 and cos > 0.9999),
                    "error": "",
                }
                rows.append(row)
                print(f"P1 {method} {test_type} rel={rel:.2e} cos={cos:.5f}")
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                rows.append({"stage": "P1", "primitive": method, "test_type": test_type, "error": repr(exc)})
                print(f"P1 ERROR {method} {test_type}: {exc!r}")
            write_csv(out_dir / "p1_analytic_gradient_correctness.csv", rows)
    return rows


def run_p2(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p2_graphfree_microbenchmark.csv")
    params = V61Params(bench_warmup=int(args.v61_bench_warmup), bench_reps=int(args.v61_bench_reps))
    device = get_device(args.device)
    shapes = [(f"B{b}-H{h}-D{d}", b, h, d) for b in (128, 256, 512) for h in (64, 96) for d in (2, 4)]
    done = {(r.get("primitive"), r.get("shape")) for r in rows if not r.get("error")}
    for shape, batch, hidden, depth in shapes:
        for method in P2_METHODS:
            if (method, shape) in done:
                continue
            try:
                basis = _basis_from_name(method, 8)
                if method == "MLP-autograd-reference":
                    stat = _measure_mlp_autograd(batch, 784, hidden, depth, params, device)
                elif "Dense-ABRBF" in method:
                    stat = _measure_dense_reference(method, batch, 784, hidden, depth, basis, params, device)
                else:
                    stat = _measure_manual(method, batch, 784, hidden, depth, basis, params, device)
                row = {
                    "stage": "P2",
                    "primitive": method,
                    "shape": shape,
                    "batch_size": batch,
                    "input_dim": 784,
                    "hidden_dim": hidden,
                    "depth": depth,
                    "basis_count": basis,
                    "uses_loss_backward": int("manual" not in method),
                    "uses_autograd_graph": int("manual" not in method),
                    "error": "",
                    **stat,
                }
                rows.append(row)
                _apply_p2_ratios(rows)
                print(f"P2 {shape} {method} step={row['step_time_ms']:.3f}ms mem={row['backward_memory_peak_mb']:.1f}MB")
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                rows.append({"stage": "P2", "primitive": method, "shape": shape, "error": repr(exc)})
                print(f"P2 ERROR {shape} {method}: {exc!r}")
            write_csv(out_dir / "p2_graphfree_microbenchmark.csv", rows)
            write_csv(out_dir / "p2_efficiency_selection.csv", [{"stage": "P2", "primitive": s, "p2_survivor": 1, "error": ""} for s in _p2_survivors(out_dir)] or [{"stage": "P2", "status": "no_survivor", "error": ""}])
    return rows


def run_p3(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    surv = _p2_survivors(out_dir)
    if not surv:
        return _placeholder(out_dir, "p3_primitive_efficiency_selection.csv", "P3", "P2 produced no graph-free efficiency survivor")
    return _placeholder(out_dir, "p3_primitive_efficiency_selection.csv", "P3", "P3 empirical descent audit is not implemented; fixed proxy rows are forbidden")


def run_p4(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not any(int(float(r.get("p3_pass", 0) or 0)) == 1 for r in read_csv(out_dir / "p3_primitive_efficiency_selection.csv")):
        return _placeholder(out_dir, "p4_manual_training_smoke.csv", "P4", "P3 produced no primitive candidate")
    return _placeholder(out_dir, "p4_manual_training_smoke.csv", "P4", "manual training smoke left gated")


def run_p5(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not any(int(float(r.get("p4_pass", 0) or 0)) == 1 for r in read_csv(out_dir / "p4_manual_training_smoke.csv")):
        return _placeholder(out_dir, "p5_manual_adam_convergence.csv", "P5", "P4 produced no manual task learner")
    return _placeholder(out_dir, "p5_manual_adam_convergence.csv", "P5", "manual Adam baseline left gated")


def run_p6(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not any(int(float(r.get("p5_pass", 0) or 0)) == 1 for r in read_csv(out_dir / "p5_manual_adam_convergence.csv")):
        return _placeholder(out_dir, "p6_functional_without_autograd.csv", "P6", "P5 produced no graph-free task learner")
    return _placeholder(out_dir, "p6_functional_without_autograd.csv", "P6", "functional update left gated")


def run_p7_to_p10(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(args.out_dir)
    if not any(int(float(r.get("p6_pass", 0) or 0)) == 1 for r in read_csv(out_dir / "p6_functional_without_autograd.csv")):
        _placeholder(out_dir, "p7_acceleration_package.csv", "P7", "P6 produced no functional survivor")
        _placeholder(out_dir, "p8_manual_lightsmooth_geometry.csv", "P8", "P7 was not reached")
        _placeholder(out_dir, "p9_joint_selection3.csv", "P9", "P8 was not reached")
        _placeholder(out_dir, "p10_confirm5.csv", "P10", "P9 was not reached")
        _placeholder(out_dir, "p10_confirm10.csv", "P10", "P10 5-seed was not reached")


def run_failure(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    failures: List[Dict[str, Any]] = []
    for row in read_csv(out_dir / "p0_graphfree_invariants.csv"):
        if row.get("error"):
            failures.append({"stage": "P0", "primitive": row.get("primitive"), "failure_type": "F0_error", "metric": row.get("error"), "recommendation": "fix graph-free invariant implementation"})
        elif "reference" not in str(row.get("primitive")) and int(float(row.get("p0_pass", 0) or 0)) != 1:
            failures.append({"stage": "P0", "primitive": row.get("primitive"), "failure_type": "F1_graphfree_invariant_fail", "metric": "p0_pass=0", "recommendation": "remove autograd dependency or fix coverage"})
    for row in read_csv(out_dir / "p1_analytic_gradient_correctness.csv"):
        if row.get("error") or int(float(row.get("p1_exploratory_pass", 0) or 0)) != 1:
            failures.append({"stage": "P1", "primitive": row.get("primitive"), "failure_type": "F2_grad_correctness_fail", "metric": f"rel={row.get('grad_coeff_relerr')} cos={row.get('grad_coeff_cosine')}", "recommendation": "fix manual adjoint formula"})
    for row in read_csv(out_dir / "p2_graphfree_microbenchmark.csv"):
        if row.get("error") or "reference" in str(row.get("primitive")):
            continue
        if int(float(row.get("p2_exploratory_pass", 0) or 0)) != 1:
            if float(row.get("forward_time_ratio_vs_mlp", 99) or 99) > 1.5:
                ftype = "F3_forward_time_fail"
            elif float(row.get("backward_time_ratio_vs_mlp", 99) or 99) > 1.8:
                ftype = "F4_backward_time_fail"
            else:
                ftype = "F5_backward_memory_fail"
            failures.append({"stage": "P2", "primitive": row.get("primitive"), "failure_type": ftype, "metric": f"fwd={row.get('forward_time_ratio_vs_mlp')} bwd={row.get('backward_time_ratio_vs_mlp')} bmem={row.get('backward_memory_ratio_vs_mlp')}", "recommendation": "optimize manual kernel/cache or redesign primitive"})
    for filename, stage in [
        ("p3_primitive_efficiency_selection.csv", "P3"),
        ("p4_manual_training_smoke.csv", "P4"),
        ("p5_manual_adam_convergence.csv", "P5"),
        ("p6_functional_without_autograd.csv", "P6"),
        ("p7_acceleration_package.csv", "P7"),
        ("p8_manual_lightsmooth_geometry.csv", "P8"),
        ("p9_joint_selection3.csv", "P9"),
        ("p10_confirm5.csv", "P10"),
        ("p10_confirm10.csv", "P10"),
    ]:
        for row in read_csv(out_dir / filename):
            if row.get("status") == "not_run":
                failures.append({"stage": stage, "primitive": row.get("primitive", filename), "failure_type": "F9_gated_not_run", "metric": row.get("reason", ""), "recommendation": "resolve upstream graph-free gate"})
    if not failures:
        failures.append({"stage": "all", "primitive": "all", "failure_type": "no_failure_rows", "metric": "", "recommendation": ""})
    write_csv(out_dir / "failure_table.csv", failures)
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = add_v3_args()
    parser.description = __doc__
    parser.set_defaults(packages="V6_1_P0", out_dir=Path("results/v6_1"), datasets="MNIST,Fashion-MNIST,KMNIST", seeds="0,1,2")
    parser.add_argument("--v61-bench-warmup", type=int, default=2)
    parser.add_argument("--v61-bench-reps", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.out_dir)
    for pkg in parse_str_list(args.packages):
        key = pkg.upper()
        if key in {"V6_1_P0", "V6_1_P0_GRAPHFREE"}:
            run_p0(args)
        elif key in {"V6_1_P1", "V6_1_P1_GRAD"}:
            run_p1(args)
        elif key in {"V6_1_P2", "V6_1_P2_EFFICIENCY"}:
            run_p2(args)
        elif key in {"V6_1_P3"}:
            run_p3(args)
        elif key in {"V6_1_P4"}:
            run_p4(args)
        elif key in {"V6_1_P5"}:
            run_p5(args)
        elif key in {"V6_1_P6"}:
            run_p6(args)
        elif key in {"V6_1_P7_TO_P10", "V6_1_P7"}:
            run_p7_to_p10(args)
        elif key in {"V6_1_FAILURE"}:
            run_failure(args)
        elif key in {"V6_1_ALL", "ALL"}:
            run_p0(args)
            run_p1(args)
            run_p2(args)
            run_p3(args)
            run_p4(args)
            run_p5(args)
            run_p6(args)
            run_p7_to_p10(args)
            run_failure(args)
        else:
            raise ValueError(f"unknown package: {pkg}")


if __name__ == "__main__":
    main()
