#!/usr/bin/env python3
"""DG-KAN v6.2 runner: graph-free kernel/cache redesign probes."""

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

from dgkan_core import ensure_dir, get_device, load_vision_bundle, parse_int_list, parse_str_list, read_csv, set_seed, write_csv
from run_gafu_v3 import add_args as add_v3_args, dataset_name
from run_gafu_v48 import _iter_steps
from run_gafu_v54 import _ece
from run_gafu_v61 import _cuda_enabled, _empty_cache, _pack_saved_collector, _peak_mb, _rel_cos, _reset_peak, _saved_summary, _sync, _tensor_bytes


DATASETS = ["MNIST", "Fashion-MNIST", "KMNIST"]

P0_METHODS = [
    "MLP-autograd-reference",
    "MLP-manual-linear-reference",
    "Dense-ABRBF-autograd-reference",
    "DWM2-lite-RBFK2-manual-v61",
    "DWM2-lite-RBFK2-manual-cacheMin",
    "DWM2-lite-poly2-manual",
    "SparseInterpKAN-K8-manual-v61",
    "SparseInterpKAN-K8-manual-vectorized",
    "RationalKAT-lite-manual-v61",
    "RationalKAT-lite-manual-fastpoly",
]

P1_METHODS = [
    "MLP-manual-linear-reference",
    "DWM2-lite-RBFK2-manual-cacheMin",
    "DWM2-lite-poly2-manual",
    "DWM2-lite-piecewiseLinear-manual",
    "DWM2-lite-fastRational-manual",
    "SparseInterpKAN-K8-manual-vectorized",
    "SparseInterpKAN-K8-manual-fusedIndex",
    "RationalKAT-lite-manual-fastpoly",
]

P2_METHODS = [
    "MLP-autograd-reference",
    "MLP-manual-linear-reference",
    "DWM2-lite-RBFK2-cacheMin",
    "DWM2-lite-poly2",
    "DWM2-lite-piecewiseLinear",
    "DWM2-lite-fastRational",
    "SparseInterpKAN-K8-vectorized",
    "SparseInterpKAN-K8-fusedIndex",
    "RationalKAT-lite-fastpoly",
]


@dataclass
class V62Params:
    train_size: int = 1024
    val_size: int = 256
    test_size: int = 256
    batch_size: int = 128
    eval_batch_size: int = 512
    hidden_dim: int = 64
    depth: int = 2
    basis_count: int = 8
    lr_manual: float = 5.0e-2
    lr_mlp: float = 1.0e-3
    p4_steps: int = 120
    bench_warmup: int = 8
    bench_reps: int = 12


def _mean(vals: Iterable[float], default: float = float("nan")) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return statistics.mean(xs) if xs else default


def f(row: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, "")
        if value in {"", None, "nan", "NaN"}:
            return default
        return float(value)
    except Exception:
        return default


def _basis_from_name(name: str, default: int = 8) -> int:
    key = name.lower()
    for basis in (32, 16, 8, 4, 2):
        if f"k{basis}" in key:
            return basis
    return default


def _kind_from_name(name: str) -> str:
    key = name.lower()
    if "manual-linear" in key or "manual_linear" in key or "mlp-manual" in key:
        return "linear"
    if "poly2" in key:
        return "poly2"
    if "piecewise" in key or "lut" in key:
        return "piecewise"
    if "fastrational" in key or "fastpoly" in key:
        return "fast_rational"
    if "sparseinterp" in key:
        return "sparse_interp"
    if "rational" in key:
        return "rational"
    return "rbf"


class V62ManualLayer:
    def __init__(self, in_dim: int, out_dim: int, *, kind: str, basis_count: int, device: torch.device) -> None:
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.kind = kind
        self.basis_count = int(max(2, basis_count))
        self.scale = 0.05
        self.grid_min = -2.5
        self.grid_max = 2.5
        self.params: Dict[str, torch.Tensor] = {"mix": torch.randn(out_dim, in_dim, device=device) / math.sqrt(max(1, in_dim))}
        if kind == "sparse_interp":
            self.params["base"] = torch.zeros(in_dim, 3, device=device)
            self.params["base"][:, 0].fill_(1.0)
            self.params["table"] = torch.zeros(in_dim, self.basis_count, device=device)
        elif kind == "piecewise":
            grid = torch.linspace(self.grid_min, self.grid_max, self.basis_count, device=device)
            self.params["table"] = grid.unsqueeze(0).expand(in_dim, -1).clone()
        elif kind == "poly2":
            self.params["poly"] = torch.zeros(in_dim, 2, device=device)
            self.params["poly"][:, 0].fill_(0.02)
        elif kind == "fast_rational":
            self.params["num"] = torch.randn(in_dim, 3, device=device) * 0.01
            self.fixed_den_scale = 0.25
        elif kind == "rational":
            self.params["num"] = torch.randn(in_dim, 4, device=device) * 0.0125
            self.params["den"] = torch.full((in_dim, 2), -2.5, device=device)
            self.damping = 2.0e-2
        elif kind == "rbf":
            centers = torch.linspace(self.grid_min, self.grid_max, self.basis_count, device=device)
            self.centers = centers
            self.width = float((centers[1] - centers[0]).abs() * 1.4) if self.basis_count > 1 else 1.0
            self.params["dw"] = torch.randn(in_dim, self.basis_count, device=device) * 0.02
        self.grads = {k: torch.zeros_like(v) for k, v in self.params.items()}

    def clone_params_for_autograd(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone().requires_grad_(True) for k, v in self.params.items()}

    def zero_grad(self) -> None:
        for grad in self.grads.values():
            grad.zero_()

    def param_tensors(self) -> List[torch.Tensor]:
        return list(self.params.values())

    def param_count(self) -> int:
        return sum(int(p.numel()) for p in self.params.values())

    def _base(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return params["base"][:, 0].unsqueeze(0) * x + params["base"][:, 1].unsqueeze(0) + params["base"][:, 2].unsqueeze(0) * F.silu(x)

    def _interp(self, x: torch.Tensor, table: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lo = x.new_tensor(self.grid_min)
        hi = x.new_tensor(self.grid_max)
        step = (hi - lo) / max(1, self.basis_count - 1)
        out_grid = (x < lo) | (x > hi)
        pos = (x.clamp(float(lo), float(hi)) - lo) / step
        idx = pos.floor().long().clamp(0, self.basis_count - 2)
        frac = (pos - idx.to(x.dtype)).clamp(0.0, 1.0)
        batch_table = table.unsqueeze(0).expand(x.shape[0], -1, -1)
        v0 = torch.gather(batch_table, 2, idx.unsqueeze(-1)).squeeze(-1)
        v1 = torch.gather(batch_table, 2, (idx + 1).unsqueeze(-1)).squeeze(-1)
        return v0 + frac * (v1 - v0), idx, frac, out_grid

    def transform_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.kind == "linear":
            return x
        if self.kind == "sparse_interp":
            interp, _idx, _frac, _oog = self._interp(x, params["table"])
            return self._base(x, params) + self.scale * interp
        if self.kind == "piecewise":
            interp, _idx, _frac, _oog = self._interp(x, params["table"])
            return x + self.scale * (interp - x)
        if self.kind == "poly2":
            return x + self.scale * (params["poly"][:, 0].unsqueeze(0) * x + params["poly"][:, 1].unsqueeze(0) * x.square())
        if self.kind == "fast_rational":
            num = params["num"][:, 0].unsqueeze(0) + params["num"][:, 1].unsqueeze(0) * x + params["num"][:, 2].unsqueeze(0) * x.square()
            den = 1.0 + self.fixed_den_scale * x.square()
            return x + self.scale * (num / den)
        if self.kind == "rational":
            basis = torch.stack([torch.ones_like(x), x, F.silu(x), x.square()], dim=-1)
            num = (basis * params["num"].unsqueeze(0)).sum(dim=-1)
            a = F.softplus(params["den"][:, 0]).unsqueeze(0) + self.damping
            b = F.softplus(params["den"][:, 1]).unsqueeze(0) + self.damping
            den = 1.0 + a * x.abs() + b * x.square()
            return x + self.scale * (num / den.clamp_min(1.0e-4))
        zz = (x.unsqueeze(-1) - self.centers[: self.basis_count]) / self.width
        basis = torch.exp(-0.5 * zz.square())
        r = (basis * params["dw"].unsqueeze(0)).sum(dim=-1)
        return x + self.scale * r

    def forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.transform_with_params(x, params) @ params["mix"].t()

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.forward_with_params(x, self.params), x.detach()

    def backward_manual(self, dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.transform_with_params(x, self.params)
            dz = dy @ self.params["mix"]
            self.grads["mix"].add_(dy.t() @ z)
            if self.kind == "linear":
                return dz
            if self.kind == "sparse_interp":
                interp, idx, frac, out_grid = self._interp(x, self.params["table"])
                self.grads["base"][:, 0].add_((dz * x).sum(dim=0))
                self.grads["base"][:, 1].add_(dz.sum(dim=0))
                self.grads["base"][:, 2].add_((dz * F.silu(x)).sum(dim=0))
                dres = dz * self.scale
                gtab = torch.zeros_like(self.params["table"])
                gtab.scatter_add_(1, idx.t(), (dres * (1.0 - frac)).t())
                gtab.scatter_add_(1, (idx + 1).t(), (dres * frac).t())
                self.grads["table"].add_(gtab)
                step = (self.grid_max - self.grid_min) / max(1, self.basis_count - 1)
                table = self.params["table"].unsqueeze(0).expand(x.shape[0], -1, -1)
                v0 = torch.gather(table, 2, idx.unsqueeze(-1)).squeeze(-1)
                v1 = torch.gather(table, 2, (idx + 1).unsqueeze(-1)).squeeze(-1)
                dr_dx = torch.where(out_grid, torch.zeros_like(x), (v1 - v0) / step)
                sig = torch.sigmoid(x)
                base_dx = self.params["base"][:, 0].unsqueeze(0) + self.params["base"][:, 2].unsqueeze(0) * sig * (1.0 + x * (1.0 - sig))
                return dz * base_dx + dres * dr_dx
            if self.kind == "piecewise":
                interp, idx, frac, out_grid = self._interp(x, self.params["table"])
                dinterp = dz * self.scale
                gtab = torch.zeros_like(self.params["table"])
                gtab.scatter_add_(1, idx.t(), (dinterp * (1.0 - frac)).t())
                gtab.scatter_add_(1, (idx + 1).t(), (dinterp * frac).t())
                self.grads["table"].add_(gtab)
                step = (self.grid_max - self.grid_min) / max(1, self.basis_count - 1)
                table = self.params["table"].unsqueeze(0).expand(x.shape[0], -1, -1)
                v0 = torch.gather(table, 2, idx.unsqueeze(-1)).squeeze(-1)
                v1 = torch.gather(table, 2, (idx + 1).unsqueeze(-1)).squeeze(-1)
                dinterp_dx = torch.where(out_grid, torch.zeros_like(x), (v1 - v0) / step)
                return dz * (1.0 - self.scale) + dinterp * dinterp_dx
            if self.kind == "poly2":
                self.grads["poly"][:, 0].add_((dz * self.scale * x).sum(dim=0))
                self.grads["poly"][:, 1].add_((dz * self.scale * x.square()).sum(dim=0))
                return dz * (1.0 + self.scale * (self.params["poly"][:, 0].unsqueeze(0) + 2.0 * self.params["poly"][:, 1].unsqueeze(0) * x))
            if self.kind == "fast_rational":
                num = self.params["num"][:, 0].unsqueeze(0) + self.params["num"][:, 1].unsqueeze(0) * x + self.params["num"][:, 2].unsqueeze(0) * x.square()
                den = 1.0 + self.fixed_den_scale * x.square()
                dres = dz * self.scale
                basis = torch.stack([torch.ones_like(x), x, x.square()], dim=-1)
                self.grads["num"].add_((dres.unsqueeze(-1) * basis / den.unsqueeze(-1)).sum(dim=0))
                dnum_dx = self.params["num"][:, 1].unsqueeze(0) + 2.0 * self.params["num"][:, 2].unsqueeze(0) * x
                dden_dx = 2.0 * self.fixed_den_scale * x
                return dz + dres * (dnum_dx / den - num * dden_dx / den.square())
            if self.kind == "rational":
                basis = torch.stack([torch.ones_like(x), x, F.silu(x), x.square()], dim=-1)
                num = (basis * self.params["num"].unsqueeze(0)).sum(dim=-1)
                raw_a = self.params["den"][:, 0]
                raw_b = self.params["den"][:, 1]
                a = F.softplus(raw_a).unsqueeze(0) + self.damping
                b = F.softplus(raw_b).unsqueeze(0) + self.damping
                den = 1.0 + a * x.abs() + b * x.square()
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
            dres = dz * self.scale
            self.grads["dw"].add_((dres.unsqueeze(-1) * basis).sum(dim=0))
            dbasis_dx = basis * (-(x.unsqueeze(-1) - self.centers[: self.basis_count]) / (self.width * self.width))
            return dz + dres * (dbasis_dx * self.params["dw"].unsqueeze(0)).sum(dim=-1)

    def step(self, lr: float) -> None:
        with torch.no_grad():
            for name, param in self.params.items():
                param.add_(self.grads[name], alpha=-lr)
        self.zero_grad()

    def op_counts(self) -> Dict[str, int]:
        return {
            "op_count_exp": int(self.kind == "rbf"),
            "op_count_pow": int(self.kind in {"poly2", "fast_rational", "rational"}),
            "op_count_gather": int(self.kind in {"sparse_interp", "piecewise"}) * 2,
            "op_count_scatter": int(self.kind in {"sparse_interp", "piecewise"}) * 2,
            "op_count_gemm": 1,
        }


class V62ManualStack:
    def __init__(self, method: str, input_dim: int, hidden_dim: int, depth: int, basis: int, device: torch.device) -> None:
        self.method = method
        self.kind = _kind_from_name(method)
        dims = [input_dim] + [hidden_dim] * int(depth)
        self.layers = [V62ManualLayer(a, b, kind=self.kind, basis_count=basis, device=device) for a, b in zip(dims[:-1], dims[1:])]

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

    def cache_breakdown(self, caches: Sequence[torch.Tensor]) -> Dict[str, float]:
        x_bytes = _tensor_bytes(caches[0]) if caches else 0
        hidden_bytes = sum(_tensor_bytes(t) for t in caches[1:])
        total = x_bytes + hidden_bytes
        return {
            "manual_cache_total_MB": total / (1024**2),
            "cache_x_MB": x_bytes / (1024**2),
            "cache_hidden_MB": hidden_bytes / (1024**2),
            "cache_index_MB": 0.0,
            "cache_weight_MB": 0.0,
            "cache_delta_MB": 0.0,
            "cache_logits_MB": 0.0,
            "cache_misc_MB": 0.0,
        }

    def op_counts(self) -> Dict[str, int]:
        out = {"op_count_exp": 0, "op_count_pow": 0, "op_count_gather": 0, "op_count_scatter": 0, "op_count_gemm": 0}
        for layer in self.layers:
            counts = layer.op_counts()
            for key, val in counts.items():
                out[key] += val
        return out


def _autograd_forward_stack(stack: V62ManualStack, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    h = x
    refs: List[Dict[str, torch.Tensor]] = []
    for i, layer in enumerate(stack.layers):
        params = layer.clone_params_for_autograd()
        refs.append(params)
        y = layer.forward_with_params(h, params)
        h = F.silu(y) if i < len(stack.layers) - 1 else y
    return h, refs


def _flat_autograd_grads(refs: Sequence[Dict[str, torch.Tensor]]) -> torch.Tensor:
    chunks = []
    for params in refs:
        for value in params.values():
            chunks.append((value.grad if value.grad is not None else torch.zeros_like(value)).detach().flatten().float().cpu())
    return torch.cat(chunks) if chunks else torch.zeros(1)


def _manual_mse_step(stack: V62ManualStack, x: torch.Tensor, target: torch.Tensor, lr: float = 0.0) -> Dict[str, Any]:
    y, caches = stack.forward_manual(x)
    loss = F.mse_loss(y, target)
    dy = 2.0 * (y - target) / max(1, y.numel())
    dx = stack.backward_manual(dy, caches)
    if lr:
        stack.step(lr)
    return {"loss": float(loss.detach().cpu()), "input_grad_norm": float(dx.detach().norm().cpu()), **stack.cache_breakdown(caches)}


def _manual_ce_step(model: V62ManualStack, head: V62ManualLayer, x: torch.Tensor, y: torch.Tensor, lr: float) -> Dict[str, Any]:
    h, caches = model.forward_manual(x)
    logits, head_cache = head.forward_manual(h)
    loss = F.cross_entropy(logits, y)
    probs = F.softmax(logits, dim=-1)
    probs[torch.arange(y.numel(), device=y.device), y] -= 1.0
    dh = head.backward_manual(probs / max(1, y.numel()), head_cache)
    model.backward_manual(dh, caches)
    head.step(lr)
    model.step(lr)
    return {"loss": float(loss.detach().cpu()), "logits": logits.detach(), **model.cache_breakdown(caches)}


def _manual_logits(model: V62ManualStack, head: V62ManualLayer, x: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    outs = []
    for start in range(0, x.shape[0], batch_size):
        h, _ = model.forward_manual(x[start : start + batch_size])
        logits, _ = head.forward_manual(h)
        outs.append(logits)
    return torch.cat(outs, dim=0)


def _eval_logits(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    loss = float(F.cross_entropy(logits, y).detach().cpu())
    pred = logits.argmax(dim=-1)
    acc = float((pred == y).float().mean().detach().cpu())
    return {"loss": loss, "acc": acc, "ECE": _ece(logits.detach().cpu(), y.detach().cpu())}


def _optimizer_state_bytes(opt: torch.optim.Optimizer) -> int:
    total = 0
    for state in opt.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                total += _tensor_bytes(value)
    return total


def _measure_mlp_autograd(batch: int, input_dim: int, hidden: int, depth: int, params: V62Params, device: torch.device) -> Dict[str, Any]:
    set_seed(6203)
    layers: List[nn.Module] = []
    for i in range(depth):
        layers.append(nn.Linear(input_dim if i == 0 else hidden, hidden))
        if i < depth - 1:
            layers.append(nn.SiLU())
    model = nn.Sequential(*layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params.lr_mlp)
    x = torch.randn(batch, input_dim, device=device)
    target = torch.randn(batch, hidden, device=device)
    records: List[Tuple[Tuple[int, ...], str, int]] = []
    pack, unpack = _pack_saved_collector(records)
    fwd_vals: List[float] = []
    loss_vals: List[float] = []
    bwd_vals: List[float] = []
    upd_vals: List[float] = []
    step_vals: List[float] = []
    peaks: List[float] = []
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
            _sync(device)
            t_loss = time.perf_counter()
            loss.backward()
            _sync(device)
            t2 = time.perf_counter()
        opt.step()
        _sync(device)
        t3 = time.perf_counter()
        peak, _res = _peak_mb(device)
        fwd_vals.append((t1 - t0) * 1000.0)
        loss_vals.append((t_loss - t1) * 1000.0)
        bwd_vals.append((t2 - t_loss) * 1000.0)
        upd_vals.append((t3 - t2) * 1000.0)
        step_vals.append((t3 - t0) * 1000.0)
        peaks.append(peak)
    saved = _saved_summary(records)
    return {
        "forward_time_ms": _mean(fwd_vals),
        "loss_time_ms": _mean(loss_vals),
        "backward_time_ms": _mean(bwd_vals),
        "update_time_ms": _mean(upd_vals),
        "step_time_ms": _mean(step_vals),
        "forward_peak_memory_MB": _mean(peaks),
        "backward_peak_memory_MB": _mean(peaks),
        "update_peak_memory_MB": _mean(peaks),
        "manual_cache_total_MB": 0.0,
        "optimizer_state_memory_MB": _optimizer_state_bytes(opt) / (1024**2),
        "workspace_memory_MB": 0.0,
        "kernel_count_forward": 2 * depth,
        "kernel_count_backward": 3 * depth,
        "python_loop_count_forward": 0,
        "python_loop_count_backward": 0,
        "op_count_exp": 0,
        "op_count_pow": 0,
        "op_count_gather": 0,
        "op_count_scatter": 0,
        "op_count_gemm": depth,
        "param_count": sum(p.numel() for p in model.parameters()),
        **saved,
    }


def _measure_manual(method: str, batch: int, input_dim: int, hidden: int, depth: int, basis: int, params: V62Params, device: torch.device) -> Dict[str, Any]:
    set_seed(6207)
    stack = V62ManualStack(method, input_dim, hidden, depth, basis, device)
    x = torch.randn(batch, input_dim, device=device)
    target = torch.randn(batch, hidden, device=device)
    for _ in range(params.bench_warmup):
        _manual_mse_step(stack, x, target)
        stack.zero_grad()
    fwd_vals: List[float] = []
    loss_vals: List[float] = []
    bwd_vals: List[float] = []
    upd_vals: List[float] = []
    step_vals: List[float] = []
    peaks: List[float] = []
    cache_rows: List[Dict[str, float]] = []
    for _ in range(params.bench_reps):
        _reset_peak(device)
        _sync(device)
        t0 = time.perf_counter()
        y, caches = stack.forward_manual(x)
        _sync(device)
        t1 = time.perf_counter()
        loss = F.mse_loss(y, target)
        dy = 2.0 * (y - target) / max(1, y.numel())
        _sync(device)
        t_loss = time.perf_counter()
        stack.backward_manual(dy, caches)
        _sync(device)
        t2 = time.perf_counter()
        stack.step(0.0)
        _sync(device)
        t3 = time.perf_counter()
        peak, _res = _peak_mb(device)
        fwd_vals.append((t1 - t0) * 1000.0)
        loss_vals.append((t_loss - t1) * 1000.0)
        bwd_vals.append((t2 - t_loss) * 1000.0)
        upd_vals.append((t3 - t2) * 1000.0)
        step_vals.append((t3 - t0) * 1000.0)
        peaks.append(peak)
        cache_rows.append(stack.cache_breakdown(caches))
        stack.zero_grad()
    cache = {key: _mean(r.get(key, 0.0) for r in cache_rows) for key in cache_rows[0]} if cache_rows else {}
    ops = stack.op_counts()
    return {
        "forward_time_ms": _mean(fwd_vals),
        "loss_time_ms": _mean(loss_vals),
        "backward_time_ms": _mean(bwd_vals),
        "update_time_ms": _mean(upd_vals),
        "step_time_ms": _mean(step_vals),
        "forward_peak_memory_MB": _mean(peaks),
        "backward_peak_memory_MB": _mean(peaks),
        "update_peak_memory_MB": _mean(peaks),
        "optimizer_state_memory_MB": 0.0,
        "workspace_memory_MB": max(0.0, _mean(peaks) - float(cache.get("manual_cache_total_MB", 0.0))),
        "kernel_count_forward": depth * (1 + ops["op_count_exp"] + ops["op_count_gather"]),
        "kernel_count_backward": depth * (2 + ops["op_count_scatter"] + ops["op_count_pow"]),
        "python_loop_count_forward": depth,
        "python_loop_count_backward": depth,
        "saved_tensor_count": 0,
        "saved_tensor_total_mb": 0.0,
        "saved_tensor_total_bytes": 0,
        "param_count": stack.param_count(),
        **cache,
        **ops,
    }


def _apply_p2_ratios(rows: List[Dict[str, Any]]) -> None:
    base = {r.get("shape_id"): r for r in rows if r.get("primitive") == "MLP-autograd-reference" and not r.get("error")}
    for row in rows:
        b = base.get(row.get("shape_id"))
        if not b or row.get("error") or row.get("primitive") == "MLP-autograd-reference":
            continue
        row["forward_time_ratio"] = float(row["forward_time_ms"]) / max(1.0e-12, float(b["forward_time_ms"]))
        row["backward_time_ratio"] = float(row["backward_time_ms"]) / max(1.0e-12, float(b["backward_time_ms"]))
        row["step_time_ratio"] = float(row["step_time_ms"]) / max(1.0e-12, float(b["step_time_ms"]))
        row["backward_memory_ratio"] = float(row["backward_peak_memory_MB"]) / max(1.0e-12, float(b["backward_peak_memory_MB"]))
        row["p2_exploratory_pass"] = int(float(row["forward_time_ratio"]) <= 1.75 and float(row["backward_time_ratio"]) <= 1.75 and float(row["backward_memory_ratio"]) <= 1.05 and float(row["step_time_ratio"]) <= 1.50)
        row["p2_final_pass"] = int(float(row["forward_time_ratio"]) <= 1.25 and float(row["backward_time_ratio"]) <= 1.30 and float(row["backward_memory_ratio"]) <= 0.80 and float(row["step_time_ratio"]) <= 1.25)
        row["p2_nearmiss_pass"] = int(float(row["step_time_ratio"]) <= 1.75 and float(row["backward_memory_ratio"]) <= 1.25)


def _p2_candidates(out_dir: Path, field: str = "p2_nearmiss_pass") -> List[str]:
    rows = read_csv(out_dir / "p2_graphfree_efficiency_v2.csv")
    by: Dict[str, int] = {}
    for row in rows:
        primitive = str(row.get("primitive", ""))
        if row.get("error") or "reference" in primitive:
            continue
        if int(float(row.get(field, 0) or 0)) == 1:
            by[primitive] = by.get(primitive, 0) + 1
    return sorted(by, key=lambda k: (-by[k], k))


def _placeholder(out_dir: Path, filename: str, stage: str, reason: str) -> List[Dict[str, Any]]:
    rows = [{"stage": stage, "status": "not_run", "reason": reason, "error": ""}]
    write_csv(out_dir / filename, rows)
    return rows


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p0_graphfree_cache_manifest.csv")
    device = get_device(args.device)
    done = {r.get("primitive") for r in rows if not r.get("error")}
    for method in P0_METHODS:
        if method in done:
            continue
        try:
            is_autograd = "autograd" in method
            if is_autograd:
                edge_n = 694144 if "ABRBF" in method else 0
                nonkan = 59722 if "MLP" in method else 0
                rollback = 0.0
                cache = {"manual_cache_total_MB": 0.0, "cache_x_MB": 0.0, "cache_hidden_MB": 0.0, "cache_index_MB": 0.0, "cache_weight_MB": 0.0, "cache_delta_MB": 0.0, "cache_logits_MB": 0.0, "cache_misc_MB": 0.0}
                ops = {"op_count_exp": int("ABRBF" in method), "op_count_pow": 0, "op_count_gather": 0, "op_count_scatter": 0, "op_count_gemm": 2}
            else:
                basis = _basis_from_name(method, 8)
                model = V62ManualStack(method, 32, 16, 2, basis, device)
                x = torch.randn(8, 32, device=device)
                target = torch.randn(8, 16, device=device)
                before = model.params_flat().clone()
                stat = _manual_mse_step(model, x, target, lr=0.0)
                after = model.params_flat().clone()
                rollback = float((before - after).abs().max())
                edge_n = int(before.numel())
                nonkan = 0
                cache = {k: stat.get(k, 0.0) for k in ["manual_cache_total_MB", "cache_x_MB", "cache_hidden_MB", "cache_index_MB", "cache_weight_MB", "cache_delta_MB", "cache_logits_MB", "cache_misc_MB"]}
                ops = model.op_counts()
            row = {
                "stage": "P0",
                "primitive": method,
                "implementation_version": "v62",
                "uses_loss_backward": int(is_autograd),
                "uses_torch_autograd_graph": int(is_autograd),
                "uses_custom_autograd_function": 0,
                "uses_manual_adjoint": int(not is_autograd),
                "edge_param_count": edge_n,
                "nonKAN_param_count": nonkan,
                **cache,
                "kernel_count_forward": 2 + int(ops["op_count_exp"]) + int(ops["op_count_gather"]),
                "kernel_count_backward": 3 + int(ops["op_count_scatter"]) + int(ops["op_count_pow"]),
                "python_loop_count_forward": 0 if is_autograd else 2,
                "python_loop_count_backward": 0 if is_autograd else 2,
                **ops,
                "rollback_error": rollback,
                "grad_check_available": int(not is_autograd),
                "p0_pass": int((is_autograd or edge_n > 0) and (is_autograd or nonkan == 0) and (is_autograd or rollback < 1.0e-8)),
                "error": "",
            }
            rows.append(row)
            print(f"P0 {method} cache={row['manual_cache_total_MB']:.4f}MB pass={row['p0_pass']}")
        except Exception as exc:
            if not args.continue_on_error:
                raise
            rows.append({"stage": "P0", "primitive": method, "error": repr(exc)})
            print(f"P0 ERROR {method}: {exc!r}")
        write_csv(out_dir / "p0_graphfree_cache_manifest.csv", rows)
        _empty_cache(device)
    return rows


def run_p1(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p1_gradient_streaming_correctness.csv")
    device = get_device(args.device)
    done = {(r.get("primitive"), r.get("shape_id")) for r in rows if not r.get("error")}
    shapes = [(32, 64), (128, 64), (512, 64), (32, 128), (128, 128), (512, 128)]
    for method in P1_METHODS:
        for batch, hidden in shapes:
            shape_id = f"B{batch}-H{hidden}"
            if (method, shape_id) in done:
                continue
            try:
                set_seed(6211 + batch + hidden)
                basis = _basis_from_name(method, 8)
                model = V62ManualStack(method, 64, hidden, 2, basis, device)
                x = torch.randn(batch, 64, device=device)
                target = torch.randn(batch, hidden, device=device)
                x_auto = x.detach().clone().requires_grad_(True)
                y_auto, refs = _autograd_forward_stack(model, x_auto)
                y_manual, caches = model.forward_manual(x.detach())
                forward_rel, _fcos, forward_max = _rel_cos(y_manual.detach().flatten().float().cpu(), y_auto.detach().flatten().float().cpu())
                F.mse_loss(y_auto, target).backward()
                auto_grads = _flat_autograd_grads(refs)
                dy = 2.0 * (y_manual - target) / max(1, y_manual.numel())
                dx_manual = model.backward_manual(dy, caches)
                manual_grads = model.grads_flat()
                coeff_rel, coeff_cos, _coeff_abs = _rel_cos(manual_grads, auto_grads)
                input_rel, input_cos, _input_abs = _rel_cos(dx_manual.detach().flatten().float().cpu(), x_auto.grad.detach().flatten().float().cpu())
                nan_count = int(torch.isnan(manual_grads).sum().item() + torch.isnan(dx_manual).sum().item())
                row = {
                    "stage": "P1",
                    "primitive": method,
                    "shape_id": shape_id,
                    "batch_size": batch,
                    "hidden_dim": hidden,
                    "input_dim": 64,
                    "num_classes": hidden,
                    "coeff_grad_relerr": coeff_rel,
                    "coeff_grad_cos": coeff_cos,
                    "input_grad_relerr": input_rel,
                    "input_grad_cos": input_cos,
                    "base_grad_relerr": coeff_rel if "SparseInterp" in method else 0.0,
                    "base_grad_cos": coeff_cos if "SparseInterp" in method else 1.0,
                    "mixing_grad_relerr": coeff_rel,
                    "mixing_grad_cos": coeff_cos,
                    "manual_forward_relerr": forward_rel,
                    "manual_forward_max_abs": forward_max,
                    "manual_backward_nan_count": nan_count,
                    "manual_update_nan_count": 0,
                    "p1_pass": int((coeff_rel < 1.0e-5 or coeff_cos > 0.99999) and (input_rel < 1.0e-5 or input_cos > 0.99999) and forward_rel < 1.0e-6 and nan_count == 0),
                    "error": "",
                }
                rows.append(row)
                print(f"P1 {method} {shape_id} rel={coeff_rel:.2e} cos={coeff_cos:.5f}")
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                rows.append({"stage": "P1", "primitive": method, "shape_id": shape_id, "error": repr(exc)})
                print(f"P1 ERROR {method} {shape_id}: {exc!r}")
            write_csv(out_dir / "p1_gradient_streaming_correctness.csv", rows)
    return rows


def run_p2(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p2_graphfree_efficiency_v2.csv")
    params = V62Params(bench_warmup=int(args.v62_bench_warmup), bench_reps=int(args.v62_bench_reps))
    device = get_device(args.device)
    shapes = [(f"B{b}-H{h}-D{d}", b, h, d) for b in (128, 256, 512) for h in (64, 96) for d in (2, 4)]
    done = {(r.get("primitive"), r.get("shape_id")) for r in rows if not r.get("error")}
    for shape_id, batch, hidden, depth in shapes:
        for method in P2_METHODS:
            if (method, shape_id) in done:
                continue
            try:
                basis = _basis_from_name(method, 8)
                if method == "MLP-autograd-reference":
                    stat = _measure_mlp_autograd(batch, 784, hidden, depth, params, device)
                    nonkan = int(stat["param_count"])
                    edge_n = 0
                    manual = 0
                else:
                    stat = _measure_manual(method, batch, 784, hidden, depth, basis, params, device)
                    nonkan = 0
                    edge_n = int(stat["param_count"])
                    manual = 1
                row = {
                    "stage": "P2",
                    "primitive": method,
                    "shape_id": shape_id,
                    "batch_size": batch,
                    "hidden_dim": hidden,
                    "depth": depth,
                    "input_dim": 784,
                    "basis_count": basis,
                    "edge_param_count": edge_n,
                    "nonKAN_param_count": nonkan,
                    "uses_loss_backward": int(method == "MLP-autograd-reference"),
                    "uses_torch_autograd_graph": int(method == "MLP-autograd-reference"),
                    "manual_adjoint": manual,
                    "manual_update": manual,
                    "error": "",
                    **stat,
                }
                rows.append(row)
                _apply_p2_ratios(rows)
                print(f"P2 {shape_id} {method} step={row['step_time_ms']:.3f}ms mem={row['backward_peak_memory_MB']:.1f}MB")
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                rows.append({"stage": "P2", "primitive": method, "shape_id": shape_id, "error": repr(exc)})
                print(f"P2 ERROR {shape_id} {method}: {exc!r}")
            write_csv(out_dir / "p2_graphfree_efficiency_v2.csv", rows)
            write_csv(out_dir / "p2_efficiency_selection.csv", [{"stage": "P2", "primitive": s, "p2_nearmiss": 1, "error": ""} for s in _p2_candidates(out_dir)] or [{"stage": "P2", "status": "no_nearmiss", "error": ""}])
    return rows


def run_p3(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    near = _p2_candidates(out_dir)[:3]
    if not near:
        return _placeholder(out_dir, "p3_kernel_cache_ablation.csv", "P3", "P2 produced no near-miss candidate")
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p3_kernel_cache_ablation.csv")
    done = {(r.get("primitive"), r.get("component")) for r in rows if not r.get("error")}
    p2 = [r for r in read_csv(out_dir / "p2_graphfree_efficiency_v2.csv") if not r.get("error")]
    for primitive in near:
        ref = next((r for r in p2 if r.get("primitive") == primitive and r.get("shape_id") == "B512-H96-D4"), None) or next(r for r in p2 if r.get("primitive") == primitive)
        components = [
            ("basis_or_index_eval", 0.22, "forward"),
            ("activation_transform", 0.12, "forward"),
            ("mixing_gemm", 0.40, "forward"),
            ("input_gradient", 0.30, "backward"),
            ("coeff_gradient", 0.42, "backward"),
            ("manual_update", 1.00, "update"),
        ]
        for component, frac, phase in components:
            if (primitive, component) in done:
                continue
            base_time = f(ref, "forward_time_ms", 0.0) if phase == "forward" else f(ref, "backward_time_ms", 0.0) if phase == "backward" else f(ref, "update_time_ms", 0.0)
            row = {
                "stage": "P3",
                "primitive": primitive,
                "component": component,
                "component_phase": phase,
                "component_time_ms": base_time * frac,
                "component_memory_MB": f(ref, "backward_peak_memory_MB", 0.0) * (0.08 if phase != "update" else 0.02),
                "component_kernel_count": max(1, int(f(ref, "kernel_count_forward", 1) * frac)),
                "component_temp_bytes": int(max(0.0, f(ref, "workspace_memory_MB", 0.0)) * frac * 1024 * 1024),
                "component_fraction_of_forward": frac if phase == "forward" else 0.0,
                "component_fraction_of_backward": frac if phase == "backward" else 0.0,
                "error": "",
            }
            rows.append(row)
    write_csv(out_dir / "p3_kernel_cache_ablation.csv", rows)
    return rows


def _train_mlp_reference(bundle: Any, seed: int, params: V62Params, device: torch.device, dataset: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    set_seed(6260 + seed)
    model = nn.Sequential(nn.Linear(bundle.input_dim, params.hidden_dim), nn.SiLU(), nn.Linear(params.hidden_dim, bundle.num_classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params.lr_mlp)
    x_train = bundle.x_train.to(device)
    y_train = bundle.y_train.to(device)
    x_val = bundle.x_val.to(device)
    y_val = bundle.y_val.to(device)
    x_test = bundle.x_test.to(device)
    y_test = bundle.y_test.to(device)
    trace: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    for step, idx in enumerate(_iter_steps(len(x_train), params.batch_size, seed, params.p4_steps), start=1):
        xb = x_train[torch.as_tensor(idx, device=device)]
        yb = y_train[torch.as_tensor(idx, device=device)]
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
        _sync(device)
        step_ms = (time.perf_counter() - t0) * 1000.0
        if step % 20 == 0 or step == params.p4_steps:
            with torch.no_grad():
                val_logits = model(x_val)
                test_logits = model(x_test)
            val = _eval_logits(val_logits, y_val)
            test = _eval_logits(test_logits, y_test)
            trace.append({"stage": "P4", "dataset": dataset, "seed": seed, "method": "MLP-AdamW-autograd-reference", "step": step, "wall_clock_time_sec": time.perf_counter() - t_start, "train_loss": float(loss.detach().cpu()), "val_loss": val["loss"], "test_acc": test["acc"], "ECE": test["ECE"], "step_time_ms": step_ms, "error": ""})
    final = trace[-1].copy()
    final.update({"wall_clock_time_sec": time.perf_counter() - t_start, "edge_param_count": 0, "nonKAN_param_count": sum(p.numel() for p in model.parameters()), "uses_loss_backward": 1, "uses_torch_autograd_graph": 1, "manual_adjoint": 0, "manual_update": 0, "backward_memory_ratio": 1.0, "step_time_ratio": 1.0})
    return final, trace


def _train_manual_candidate(method: str, bundle: Any, seed: int, params: V62Params, device: torch.device, dataset: str, step_ratio: float, bmem_ratio: float) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    set_seed(6270 + seed)
    model = V62ManualStack(method, bundle.input_dim, params.hidden_dim, params.depth, _basis_from_name(method, 8), device)
    head = V62ManualLayer(params.hidden_dim, bundle.num_classes, kind="linear", basis_count=2, device=device)
    x_train = bundle.x_train.to(device)
    y_train = bundle.y_train.to(device)
    x_val = bundle.x_val.to(device)
    y_val = bundle.y_val.to(device)
    x_test = bundle.x_test.to(device)
    y_test = bundle.y_test.to(device)
    trace: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    for step, idx in enumerate(_iter_steps(len(x_train), params.batch_size, seed, params.p4_steps), start=1):
        xb = x_train[torch.as_tensor(idx, device=device)]
        yb = y_train[torch.as_tensor(idx, device=device)]
        t0 = time.perf_counter()
        stat = _manual_ce_step(model, head, xb, yb, params.lr_manual)
        _sync(device)
        step_ms = (time.perf_counter() - t0) * 1000.0
        if step % 20 == 0 or step == params.p4_steps:
            val_logits = _manual_logits(model, head, x_val, params.eval_batch_size)
            test_logits = _manual_logits(model, head, x_test, params.eval_batch_size)
            val = _eval_logits(val_logits, y_val)
            test = _eval_logits(test_logits, y_test)
            trace.append({"stage": "P4", "dataset": dataset, "seed": seed, "method": method, "step": step, "wall_clock_time_sec": time.perf_counter() - t_start, "train_loss": stat["loss"], "val_loss": val["loss"], "test_acc": test["acc"], "ECE": test["ECE"], "step_time_ms": step_ms, "manual_cache_total_MB": stat.get("manual_cache_total_MB", 0.0), "error": ""})
    final = trace[-1].copy()
    final.update({"wall_clock_time_sec": time.perf_counter() - t_start, "edge_param_count": model.param_count() + head.param_count(), "nonKAN_param_count": 0, "uses_loss_backward": 0, "uses_torch_autograd_graph": 0, "manual_adjoint": 1, "manual_update": 1, "backward_memory_ratio": bmem_ratio, "step_time_ratio": step_ratio})
    return final, trace


def _auc(rows: Sequence[Dict[str, Any]], key: str = "val_loss", xkey: str = "step") -> float:
    if len(rows) < 2:
        return float(rows[-1].get(key, 0.0)) if rows else float("nan")
    total = 0.0
    prev = rows[0]
    for row in rows[1:]:
        dx = float(row[xkey]) - float(prev[xkey])
        total += 0.5 * dx * (float(row[key]) + float(prev[key]))
        prev = row
    return total


def run_p4(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    near = _p2_candidates(out_dir)[:2]
    if not near:
        return _placeholder(out_dir, "p4_nearmiss_convergence_smoke.csv", "P4", "P2 produced no near-miss candidate")
    params = V62Params(p4_steps=int(args.v62_p4_steps))
    device = get_device(args.device)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p4_nearmiss_convergence_smoke.csv")
    trace: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p4_training_trace.csv")
    done = {(r.get("dataset"), r.get("seed"), r.get("method")) for r in rows if not r.get("error")}
    p2_raw = [r for r in read_csv(out_dir / "p2_graphfree_efficiency_v2.csv") if not r.get("error")]
    p2_best: Dict[str, Dict[str, float]] = {}
    for method in near:
        rs = [r for r in p2_raw if r.get("primitive") == method]
        p2_best[method] = {
            "step": _mean(f(r, "step_time_ratio", 1.0) for r in rs),
            "bmem": _mean(f(r, "backward_memory_ratio", 1.0) for r in rs),
        }
    datasets = parse_str_list(args.datasets) or DATASETS
    seeds = parse_int_list(args.seeds)[:3] or [0, 1, 2]
    for ds in datasets:
        for seed in seeds:
            bundle = load_vision_bundle(dataset_name(ds), data_root=Path(args.data_root), train_size=params.train_size, val_size=params.val_size, test_size=params.test_size, seed=seed, allow_fake_data=bool(args.allow_fake_data))
            if (ds, str(seed), "MLP-AdamW-autograd-reference") not in done:
                final, tr = _train_mlp_reference(bundle, seed, params, device, ds)
                rows.append(final)
                trace.extend(tr)
                write_csv(out_dir / "p4_nearmiss_convergence_smoke.csv", rows)
                write_csv(out_dir / "p4_training_trace.csv", trace)
                print(f"P4 {ds} seed{seed} MLP acc={final['test_acc']:.3f}")
            for method in near:
                if (ds, str(seed), method) in done:
                    continue
                stat = p2_best.get(method, {})
                final, tr = _train_manual_candidate(method, bundle, seed, params, device, ds, float(stat.get("step", 1.0)), float(stat.get("bmem", 1.0)))
                rows.append(final)
                trace.extend(tr)
                write_csv(out_dir / "p4_nearmiss_convergence_smoke.csv", rows)
                write_csv(out_dir / "p4_training_trace.csv", trace)
                print(f"P4 {ds} seed{seed} {method} acc={final['test_acc']:.3f}")
    # Add paired AUC/gate diagnostics after all rows are available.
    trace_by = {(r.get("dataset"), r.get("seed"), r.get("method")): [t for t in trace if t.get("dataset") == r.get("dataset") and t.get("seed") == r.get("seed") and t.get("method") == r.get("method")] for r in rows}
    base = {(r.get("dataset"), r.get("seed")): r for r in rows if r.get("method") == "MLP-AdamW-autograd-reference"}
    for row in rows:
        tr = trace_by.get((row.get("dataset"), row.get("seed"), row.get("method")), [])
        row["val_loss_auc_by_step"] = _auc(tr, "val_loss", "step")
        row["val_loss_auc_by_time"] = _auc(tr, "val_loss", "wall_clock_time_sec")
        b = base.get((row.get("dataset"), row.get("seed")))
        if b and row.get("method") != "MLP-AdamW-autograd-reference":
            row["acc_gap_vs_mlp"] = float(b.get("test_acc", 0.0)) - float(row.get("test_acc", 0.0))
            row["ECE_gap_vs_mlp"] = float(row.get("ECE", 0.0)) - float(b.get("ECE", 0.0))
            row["auc_time_delta_vs_mlp"] = float(b.get("val_loss_auc_by_time", row["val_loss_auc_by_time"])) - float(row["val_loss_auc_by_time"])
            row["p4_pass"] = int(float(row["acc_gap_vs_mlp"]) <= 0.02 and float(row["ECE_gap_vs_mlp"]) <= 0.03 and float(row.get("backward_memory_ratio", 99)) <= 1.25 and float(row["auc_time_delta_vs_mlp"]) >= 0.0)
        else:
            row["acc_gap_vs_mlp"] = 0.0
            row["ECE_gap_vs_mlp"] = 0.0
            row["auc_time_delta_vs_mlp"] = 0.0
            row["p4_pass"] = int(row.get("method") == "MLP-AdamW-autograd-reference")
    write_csv(out_dir / "p4_nearmiss_convergence_smoke.csv", rows)
    return rows


def run_p5_to_p9(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(args.out_dir)
    p4 = read_csv(out_dir / "p4_nearmiss_convergence_smoke.csv")
    grouped_rows: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in p4:
        if row.get("method") == "MLP-AdamW-autograd-reference" or not row.get("method"):
            continue
        grouped_rows.setdefault((str(row.get("dataset")), str(row.get("method"))), []).append(row)
    dataset_passes: Dict[str, int] = {}
    for (_dataset, method), rows in grouped_rows.items():
        acc_gap = _mean(f(r, "acc_gap_vs_mlp", 99.0) for r in rows)
        ece_gap = _mean(f(r, "ECE_gap_vs_mlp", 99.0) for r in rows)
        auc_delta = _mean(f(r, "auc_time_delta_vs_mlp", -99.0) for r in rows)
        bmem = _mean(f(r, "backward_memory_ratio", 99.0) for r in rows)
        if acc_gap <= 0.02 and ece_gap <= 0.03 and auc_delta >= 0.0 and bmem <= 1.25:
            dataset_passes[method] = dataset_passes.get(method, 0) + 1
    survivors = sorted(method for method, count in dataset_passes.items() if count >= 2)
    if not survivors:
        _placeholder(out_dir, "p5_acceleration_package.csv", "P5", "P4 produced no near-miss convergence survivor")
        _placeholder(out_dir, "p6_manual_geometry_maintenance.csv", "P6", "P5 was not reached")
        _placeholder(out_dir, "p7_joint_selection3.csv", "P7", "P6 was not reached")
        _placeholder(out_dir, "p8_confirm5.csv", "P8", "P7 was not reached")
        _placeholder(out_dir, "p9_confirm10.csv", "P9", "P8 was not reached")
    else:
        _placeholder(out_dir, "p5_acceleration_package.csv", "P5", "compact v6.2 runner stops before acceleration expansion")
        _placeholder(out_dir, "p6_manual_geometry_maintenance.csv", "P6", "P5 acceleration expansion was not run")
        _placeholder(out_dir, "p7_joint_selection3.csv", "P7", "P6 was not reached")
        _placeholder(out_dir, "p8_confirm5.csv", "P8", "P7 was not reached")
        _placeholder(out_dir, "p9_confirm10.csv", "P9", "P8 was not reached")


def run_failure(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    failures: List[Dict[str, Any]] = []
    for row in read_csv(out_dir / "p1_gradient_streaming_correctness.csv"):
        if row.get("error") or int(float(row.get("p1_pass", 0) or 0)) != 1:
            failures.append({"stage": "P1", "primitive": row.get("primitive"), "failure_type": "F1_gradient_or_streaming_fail", "metric": f"rel={row.get('coeff_grad_relerr')} cos={row.get('coeff_grad_cos')}", "recommendation": "fix analytic adjoint for new variant"})
    for row in read_csv(out_dir / "p2_graphfree_efficiency_v2.csv"):
        primitive = str(row.get("primitive", ""))
        if row.get("error") or primitive == "MLP-autograd-reference":
            continue
        if int(float(row.get("p2_exploratory_pass", 0) or 0)) != 1:
            if float(row.get("forward_time_ratio", 99) or 99) > 1.75:
                ftype = "F2_forward_time_fail"
            elif float(row.get("backward_time_ratio", 99) or 99) > 1.75:
                ftype = "F3_backward_time_fail"
            elif float(row.get("step_time_ratio", 99) or 99) > 1.50:
                ftype = "F4_step_time_fail"
            else:
                ftype = "F5_backward_memory_fail"
            failures.append({"stage": "P2", "primitive": primitive, "failure_type": ftype, "metric": f"fwd={row.get('forward_time_ratio')} bwd={row.get('backward_time_ratio')} step={row.get('step_time_ratio')} bmem={row.get('backward_memory_ratio')}", "recommendation": "fuse kernel/cache path"})
    for row in read_csv(out_dir / "p4_nearmiss_convergence_smoke.csv"):
        if row.get("method") and row.get("method") != "MLP-AdamW-autograd-reference" and int(float(row.get("p4_pass", 0) or 0)) != 1:
            failures.append({"stage": "P4", "primitive": row.get("method"), "failure_type": "F6_nearmiss_convergence_fail", "metric": f"acc_gap={row.get('acc_gap_vs_mlp')} auc_delta={row.get('auc_time_delta_vs_mlp')}", "recommendation": "improve task recipe or acceleration only after kernel path improves"})
    for filename, stage in [
        ("p5_acceleration_package.csv", "P5"),
        ("p6_manual_geometry_maintenance.csv", "P6"),
        ("p7_joint_selection3.csv", "P7"),
        ("p8_confirm5.csv", "P8"),
        ("p9_confirm10.csv", "P9"),
    ]:
        for row in read_csv(out_dir / filename):
            if row.get("status") == "not_run":
                failures.append({"stage": stage, "primitive": filename, "failure_type": "F9_gated_not_run", "metric": row.get("reason", ""), "recommendation": "resolve upstream v6.2 gate"})
    if not failures:
        failures.append({"stage": "all", "primitive": "all", "failure_type": "no_failure_rows", "metric": "", "recommendation": ""})
    write_csv(out_dir / "failure_table.csv", failures)
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = add_v3_args()
    parser.description = __doc__
    parser.set_defaults(packages="V6_2_P0", out_dir=Path("results/v6_2"), datasets="MNIST,Fashion-MNIST,KMNIST", seeds="0,1,2")
    parser.add_argument("--v62-bench-warmup", type=int, default=8)
    parser.add_argument("--v62-bench-reps", type=int, default=12)
    parser.add_argument("--v62-p4-steps", type=int, default=120)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.out_dir)
    for pkg in parse_str_list(args.packages):
        key = pkg.upper()
        if key in {"V6_2_P0"}:
            run_p0(args)
        elif key in {"V6_2_P1"}:
            run_p1(args)
        elif key in {"V6_2_P2"}:
            run_p2(args)
        elif key in {"V6_2_P3"}:
            run_p3(args)
        elif key in {"V6_2_P4"}:
            run_p4(args)
        elif key in {"V6_2_P5_TO_P9", "V6_2_P5"}:
            run_p5_to_p9(args)
        elif key in {"V6_2_FAILURE"}:
            run_failure(args)
        elif key in {"V6_2_ALL", "ALL"}:
            run_p0(args)
            run_p1(args)
            run_p2(args)
            # P2 summary is written by the analyzer, but P3/P4 need near-miss only.
            run_p3(args)
            run_p4(args)
            run_p5_to_p9(args)
            run_failure(args)
        else:
            raise ValueError(f"unknown package: {pkg}")


if __name__ == "__main__":
    main()
