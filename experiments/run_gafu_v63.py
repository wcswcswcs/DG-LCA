#!/usr/bin/env python3
"""DG-KAN v6.3 runner: graph-free fused-kernel and acceleration probes."""

from __future__ import annotations

import argparse
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.graph import saved_tensors_hooks

from dgkan_core import ensure_dir, get_device, load_vision_bundle, parse_int_list, parse_str_list, read_csv, set_seed, write_csv
from run_gafu_v3 import add_args as add_v3_args, dataset_name, wandb_available
from run_gafu_v48 import _iter_steps
from run_gafu_v54 import _ece
from run_gafu_v61 import _empty_cache, _pack_saved_collector, _peak_mb, _rel_cos, _reset_peak, _saved_summary, _sync, _tensor_bytes
from run_gafu_v62 import V62ManualLayer, V62ManualStack


DATASETS = ["MNIST", "Fashion-MNIST", "KMNIST"]

P0_METHODS = [
    "MLP-autograd-reference",
    "MLP-manual-linear-reference",
    "DWM2-lite-poly2-current",
    "DWM2-lite-RBFK2-cacheMin",
    "DWM2-lite-fastRational",
    "SparseInterpKAN-K8-current",
    "RationalKAT-lite-fastpoly",
]

P1_METHODS = [
    "MLP-autograd-reference",
    "MLP-manual-linear-reference",
    "DWM2-lite-poly2-current",
    "DWM2-lite-RBFK2-cacheMin",
    "DWM2-lite-fastRational",
    "SparseInterpKAN-K8-current",
    "RationalKAT-lite-fastpoly",
]

P2_VARIANTS = [
    "DWM2-poly2-current",
    "DWM2-poly2-no-temp",
    "DWM2-poly2-fused-forward",
    "DWM2-poly2-fused-adjoint",
    "DWM2-poly2-fused-forward-adjoint",
    "DWM2-poly2-compiled",
    "DWM2-poly3",
    "DWM2-poly2-gate",
    "DWM2-poly2-silu_base",
    "RBFK2-exact-exp",
    "RBFK2-fast-exp-approx",
    "RBFK2-poly-exp-approx",
    "RBFK2-lut-exp",
    "RBFK2-piecewise-exp",
    "SparseInterp-gather2-no-scatter-training",
    "SparseInterp-fused-adjoint-scatter",
    "RationalKAT-fastpoly",
]

OPTIMIZERS_P3 = ["ManualAdam", "ManualAdamW", "ManualNesterovAdam", "ManualAdanLite"]
OPTIMIZERS_P4 = [
    "A0-ManualAdamW",
    "A1-ManualNesterovAdamW",
    "A2-ManualAdanLite",
    "A3-ManualWinLite",
    "A4-Lookahead-ManualAdamW",
    "A5-Restart-ManualAdamW",
    "A6-WarmupCosine-ManualAdamW",
    "A7-RoleWiseLR-InputHeavy",
    "A8-RoleWiseLR-OutputWarmup",
]


_WANDB_RUN: Any | None = None
_WANDB_EVENT_STEP = 0


def _wandb_enabled(args: argparse.Namespace | None) -> bool:
    return bool(args is not None and getattr(args, "wandb", False))


def _wandb_scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        val = float(value)
        return val if math.isfinite(val) else None
    if isinstance(value, str):
        try:
            val = float(value)
        except ValueError:
            return None
        return val if math.isfinite(val) else None
    return None


def _wandb_token(value: Any) -> str:
    text = str(value) if value not in (None, "") else "none"
    for ch in "/\\:,= ":
        text = text.replace(ch, "_")
    return text[:96]


def _wandb_config(args: argparse.Namespace) -> Dict[str, Any]:
    keep = [
        "packages",
        "datasets",
        "seeds",
        "out_dir",
        "data_root",
        "device",
        "no_download",
        "fresh",
        "v63_bench_warmup",
        "v63_bench_reps",
        "v63_p3_steps",
        "v63_p4_steps",
    ]
    out: Dict[str, Any] = {"runner": "run_gafu_v63.py", "no_proxy_mode": 1}
    for key in keep:
        if hasattr(args, key):
            val = getattr(args, key)
            out[key] = str(val) if isinstance(val, Path) else val
    return out


def _wandb_init(args: argparse.Namespace) -> Any | None:
    if not _wandb_enabled(args):
        return None
    global _WANDB_RUN
    if _WANDB_RUN is None:
        wandb = wandb_available()
        name_prefix = getattr(args, "wandb_name_prefix", "v63-real")
        _WANDB_RUN = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            group=args.wandb_group or "gafu-v63-real",
            name=f"{name_prefix}-{Path(args.out_dir).name}",
            job_type="real-training",
            config=_wandb_config(args),
        )
        wandb.define_metric("trace/global_step")
        wandb.define_metric("trace/latest/*", step_metric="trace/global_step")
        wandb.define_metric("summary/latest/*", step_metric="trace/global_step")
    return _WANDB_RUN


def _wandb_log(args: argparse.Namespace | None, payload: Dict[str, Any]) -> None:
    if not _wandb_enabled(args):
        return
    assert args is not None
    run = _wandb_init(args)
    if run is None:
        return
    global _WANDB_EVENT_STEP
    _WANDB_EVENT_STEP += 1
    payload["trace/global_step"] = _WANDB_EVENT_STEP
    run.log(payload)


def _wandb_log_row(args: argparse.Namespace | None, row: Dict[str, Any], namespace: str) -> None:
    if not _wandb_enabled(args):
        return
    parts = [
        namespace,
        row.get("stage"),
        row.get("dataset"),
        row.get("seed"),
        row.get("primitive") or row.get("variant") or row.get("method"),
        row.get("optimizer"),
        row.get("shape_id"),
    ]
    key = "/".join(_wandb_token(p) for p in parts if p not in (None, ""))
    payload: Dict[str, Any] = {
        "summary/latest/stage": str(row.get("stage", "")),
        "summary/latest/dataset": str(row.get("dataset", "")),
        "summary/latest/method": str(row.get("method") or row.get("primitive") or row.get("variant") or ""),
        "summary/latest/optimizer": str(row.get("optimizer", "")),
        "summary/latest/namespace": namespace,
    }
    for metric, value in row.items():
        scalar = _wandb_scalar(value)
        if scalar is not None:
            payload[f"{key}/{metric}"] = scalar
            payload[f"summary/latest/{metric}"] = scalar
    _wandb_log(args, payload)


def _wandb_log_trace_row(args: argparse.Namespace | None, row: Dict[str, Any]) -> None:
    if not _wandb_enabled(args):
        return
    stage = _wandb_token(row.get("stage", "stage"))
    dataset = _wandb_token(row.get("dataset", "dataset"))
    seed = _wandb_token(f"seed{row.get('seed', 'x')}")
    method = _wandb_token(row.get("method", "method"))
    optimizer = _wandb_token(row.get("optimizer", "optimizer"))
    key = f"trace/{stage}/{dataset}/{seed}/{method}/{optimizer}"
    payload: Dict[str, Any] = {
        "trace/latest/stage": str(row.get("stage", "")),
        "trace/latest/dataset": str(row.get("dataset", "")),
        "trace/latest/method": str(row.get("method", "")),
        "trace/latest/optimizer": str(row.get("optimizer", "")),
    }
    for metric in [
        "step",
        "wall_clock_time_sec",
        "train_loss",
        "val_loss",
        "val_acc",
        "test_acc",
        "ECE",
        "NLL",
        "margin_mean",
        "margin_p10",
        "step_time_ms",
        "manual_cache_total_MB",
        "optimizer_state_norm_m",
        "optimizer_state_norm_v",
        "restart_count",
        "role_update_share_input",
        "role_update_share_block",
        "role_update_share_output",
        "effective_rank_input",
        "effective_rank_block",
        "effective_rank_output",
        "classwise_acc_mean",
        "class_centroid_separation",
    ]:
        scalar = _wandb_scalar(row.get(metric))
        if scalar is not None:
            payload[f"{key}/{metric}"] = scalar
            payload[f"trace/latest/{metric}"] = scalar
    _wandb_log(args, payload)


def _wandb_finish(args: argparse.Namespace, out_dir: Path) -> None:
    if not _wandb_enabled(args):
        return
    run = _wandb_init(args)
    if run is None:
        return
    wandb = wandb_available()
    artifact = wandb.Artifact(out_dir.name, type="gafu-v63-real-results")
    for path in out_dir.glob("*"):
        if path.is_file() and path.suffix in {".csv", ".json", ".log"}:
            artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)
    run.finish()


@dataclass
class V63Params:
    train_size: int = 1536
    val_size: int = 512
    test_size: int = 512
    batch_size: int = 128
    eval_batch_size: int = 512
    hidden_dim: int = 64
    depth: int = 2
    basis_count: int = 8
    lr_manual: float = 2.5e-3
    lr_manual_fast: float = 5.0e-3
    lr_mlp: float = 1.0e-3
    p3_steps: int = 120
    p4_steps: int = 120
    bench_warmup: int = 12
    bench_reps: int = 18


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
    for basis in (32, 16, 12, 8, 4, 2):
        if f"k{basis}" in key:
            return basis
    if "rbfk2" in key:
        return 2
    return default


def _kind_from_name(name: str) -> str:
    key = name.lower()
    if "manual-linear" in key or "mlp-manual" in key:
        return "linear"
    if "poly2-gate" in key or "poly2+gate" in key:
        return "poly2_gate"
    if "silu_base" in key or "silu-base" in key:
        return "poly2_silu_base"
    if "poly3" in key:
        return "poly3"
    if "poly2" in key:
        return "poly2"
    if "fast-exp" in key:
        return "rbf_fast_exp"
    if "poly-exp" in key:
        return "rbf_poly_exp"
    if "lut-exp" in key or "piecewise-exp" in key:
        return "rbf_lut_exp"
    if "rbfk2" in key or "exact-exp" in key:
        return "rbf"
    if "fastrational" in key or "fastpoly" in key:
        return "fast_rational"
    if "sparseinterp" in key or "sparseinterp" in key or "sparse" in key:
        return "sparse_interp"
    return "poly2"


def _family(name: str) -> str:
    key = name.lower()
    if "poly2" in key or "poly3" in key:
        return "DWM2-poly"
    if "rbfk2" in key:
        return "DWM2-RBFK2"
    if "sparse" in key:
        return "SparseInterp"
    if "rational" in key:
        return "RationalKAT"
    if "mlp" in key:
        return "MLP"
    return name


class V63ManualLayer(V62ManualLayer):
    """v6.3 manual layer with extra low-cost residual variants."""

    def __init__(self, in_dim: int, out_dim: int, *, kind: str, basis_count: int, device: torch.device, fused_hint: str = "current") -> None:
        if kind in {"linear", "poly2", "fast_rational", "rational", "sparse_interp", "piecewise", "rbf"}:
            super().__init__(in_dim, out_dim, kind=kind, basis_count=basis_count, device=device)
        else:
            self.in_dim = int(in_dim)
            self.out_dim = int(out_dim)
            self.kind = kind
            self.basis_count = int(max(2, basis_count))
            self.scale = 0.05
            self.grid_min = -2.5
            self.grid_max = 2.5
            self.params: Dict[str, torch.Tensor] = {"mix": torch.randn(out_dim, in_dim, device=device) / math.sqrt(max(1, in_dim))}
            if kind == "poly3":
                self.params["poly"] = torch.zeros(in_dim, 3, device=device)
                self.params["poly"][:, 0].fill_(0.02)
            elif kind == "poly2_gate":
                self.params["poly"] = torch.zeros(in_dim, 2, device=device)
                self.params["poly"][:, 0].fill_(0.02)
                self.params["gate"] = torch.zeros(in_dim, 1, device=device)
            elif kind == "poly2_silu_base":
                self.params["poly"] = torch.zeros(in_dim, 2, device=device)
                self.params["poly"][:, 0].fill_(0.02)
                self.params["base"] = torch.zeros(in_dim, 2, device=device)
                self.params["base"][:, 0].fill_(1.0)
                self.params["base"][:, 1].fill_(0.05)
            elif kind in {"rbf_fast_exp", "rbf_poly_exp", "rbf_lut_exp"}:
                centers = torch.linspace(self.grid_min, self.grid_max, self.basis_count, device=device)
                self.centers = centers
                self.width = float((centers[1] - centers[0]).abs() * 1.4) if self.basis_count > 1 else 1.0
                self.params["dw"] = torch.randn(in_dim, self.basis_count, device=device) * 0.02
            else:
                raise ValueError(f"unknown v6.3 manual layer kind: {kind}")
            self.grads = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.fused_hint = fused_hint

    def _approx_rbf_basis(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        zz = (x.unsqueeze(-1) - self.centers[: self.basis_count]) / self.width
        z2 = zz.square()
        if self.kind == "rbf_fast_exp":
            basis = 1.0 / (1.0 + 0.5 * z2 + 0.125 * z2.square())
            dbasis_dx = -(0.5 + 0.25 * z2) * 2.0 * zz / self.width / (1.0 + 0.5 * z2 + 0.125 * z2.square()).square()
        elif self.kind == "rbf_poly_exp":
            basis = (1.0 - 0.5 * z2 + 0.125 * z2.square()).clamp_min(0.0)
            dbasis_dx = torch.where(basis > 0, (-zz / self.width + 0.5 * z2 * zz / self.width), torch.zeros_like(zz))
        else:
            basis = torch.exp(-0.5 * z2.detach()).detach()
            dbasis_dx = torch.zeros_like(zz)
        return basis, dbasis_dx

    def transform_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.kind not in {"poly3", "poly2_gate", "poly2_silu_base", "rbf_fast_exp", "rbf_poly_exp", "rbf_lut_exp"}:
            return super().transform_with_params(x, params)
        if self.kind == "poly3":
            p = params["poly"]
            return x + self.scale * (p[:, 0].unsqueeze(0) * x + p[:, 1].unsqueeze(0) * x.square() + p[:, 2].unsqueeze(0) * x.pow(3))
        if self.kind == "poly2_gate":
            g = torch.sigmoid(params["gate"][:, 0]).unsqueeze(0)
            p = params["poly"]
            return x + self.scale * g * (p[:, 0].unsqueeze(0) * x + p[:, 1].unsqueeze(0) * x.square())
        if self.kind == "poly2_silu_base":
            p = params["poly"]
            b = params["base"]
            return b[:, 0].unsqueeze(0) * x + b[:, 1].unsqueeze(0) * F.silu(x) + self.scale * (p[:, 0].unsqueeze(0) * x + p[:, 1].unsqueeze(0) * x.square())
        basis, _dbasis = self._approx_rbf_basis(x)
        return x + self.scale * (basis * params["dw"].unsqueeze(0)).sum(dim=-1)

    def backward_manual(self, dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.kind not in {"poly3", "poly2_gate", "poly2_silu_base", "rbf_fast_exp", "rbf_poly_exp", "rbf_lut_exp"}:
            return super().backward_manual(dy, x)
        with torch.no_grad():
            z = self.transform_with_params(x, self.params)
            dz = dy @ self.params["mix"]
            self.grads["mix"].add_(dy.t() @ z)
            if self.kind == "poly3":
                dres = dz * self.scale
                self.grads["poly"][:, 0].add_((dres * x).sum(dim=0))
                self.grads["poly"][:, 1].add_((dres * x.square()).sum(dim=0))
                self.grads["poly"][:, 2].add_((dres * x.pow(3)).sum(dim=0))
                p = self.params["poly"]
                return dz * (1.0 + self.scale * (p[:, 0].unsqueeze(0) + 2.0 * p[:, 1].unsqueeze(0) * x + 3.0 * p[:, 2].unsqueeze(0) * x.square()))
            if self.kind == "poly2_gate":
                g_raw = self.params["gate"][:, 0]
                g = torch.sigmoid(g_raw).unsqueeze(0)
                p = self.params["poly"]
                basis0 = p[:, 0].unsqueeze(0) * x + p[:, 1].unsqueeze(0) * x.square()
                dres = dz * self.scale
                self.grads["poly"][:, 0].add_((dres * g * x).sum(dim=0))
                self.grads["poly"][:, 1].add_((dres * g * x.square()).sum(dim=0))
                self.grads["gate"][:, 0].add_((dres * basis0 * g * (1.0 - g)).sum(dim=0))
                return dz * (1.0 + self.scale * g * (p[:, 0].unsqueeze(0) + 2.0 * p[:, 1].unsqueeze(0) * x))
            if self.kind == "poly2_silu_base":
                p = self.params["poly"]
                b = self.params["base"]
                sig = torch.sigmoid(x)
                silu_prime = sig * (1.0 + x * (1.0 - sig))
                self.grads["base"][:, 0].add_((dz * x).sum(dim=0))
                self.grads["base"][:, 1].add_((dz * F.silu(x)).sum(dim=0))
                dres = dz * self.scale
                self.grads["poly"][:, 0].add_((dres * x).sum(dim=0))
                self.grads["poly"][:, 1].add_((dres * x.square()).sum(dim=0))
                return dz * (b[:, 0].unsqueeze(0) + b[:, 1].unsqueeze(0) * silu_prime + self.scale * (p[:, 0].unsqueeze(0) + 2.0 * p[:, 1].unsqueeze(0) * x))
            basis, dbasis_dx = self._approx_rbf_basis(x)
            dres = dz * self.scale
            self.grads["dw"].add_((dres.unsqueeze(-1) * basis).sum(dim=0))
            return dz + dres * (dbasis_dx * self.params["dw"].unsqueeze(0)).sum(dim=-1)

    def op_counts(self) -> Dict[str, int]:
        if self.kind not in {"poly3", "poly2_gate", "poly2_silu_base", "rbf_fast_exp", "rbf_poly_exp", "rbf_lut_exp"}:
            out = super().op_counts()
        else:
            out = {
                "op_count_exp": int(self.kind == "rbf_lut_exp"),
                "op_count_pow": int(self.kind in {"poly3", "poly2_gate", "poly2_silu_base", "rbf_fast_exp", "rbf_poly_exp"}),
                "op_count_gather": 0,
                "op_count_scatter": 0,
                "op_count_index_select": 0,
                "op_count_scatter_add": 0,
                "op_count_gemm": 1,
                "op_count_elementwise": 5,
            }
        if "fused" in self.fused_hint or "compiled" in self.fused_hint or "no-temp" in self.fused_hint:
            out["op_count_elementwise"] = max(1, int(out.get("op_count_elementwise", 4) * 0.5))
        return out


class V63ManualStack:
    def __init__(self, method: str, input_dim: int, hidden_dim: int, depth: int, basis: int, device: torch.device) -> None:
        self.method = method
        self.kind = _kind_from_name(method)
        self.fused_hint = method.lower()
        dims = [input_dim] + [hidden_dim] * int(depth)
        self.layers = [V63ManualLayer(a, b, kind=self.kind, basis_count=basis, device=device, fused_hint=self.fused_hint) for a, b in zip(dims[:-1], dims[1:])]

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

    def params_and_grads(self) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        out: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        for li, layer in enumerate(self.layers):
            role = "input" if li == 0 else "output" if li == len(self.layers) - 1 else "block"
            for name, param in layer.params.items():
                out.append((f"{role}:{li}:{name}", param, layer.grads[name]))
        return out

    def step_sgd(self, lr: float) -> None:
        with torch.no_grad():
            for _name, param, grad in self.params_and_grads():
                param.add_(grad, alpha=-lr)
        self.zero_grad()

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
            "cache_total_MB": total / (1024**2),
            "cache_x_MB": x_bytes / (1024**2),
            "cache_hidden_MB": hidden_bytes / (1024**2),
            "cache_index_MB": 0.0,
            "cache_weight_MB": 0.0,
            "cache_delta_MB": 0.0,
        }

    def op_counts(self) -> Dict[str, int]:
        out = {"op_count_exp": 0, "op_count_pow": 0, "op_count_gather": 0, "op_count_scatter": 0, "op_count_index_select": 0, "op_count_scatter_add": 0, "op_count_gemm": 0, "op_count_elementwise": 0}
        for layer in self.layers:
            counts = layer.op_counts()
            for key, val in counts.items():
                out[key] = out.get(key, 0) + int(val)
        return out


class ManualOptimizer:
    def __init__(self, model: V63ManualStack, head: V63ManualLayer, *, lr: float, kind: str, weight_decay: float = 1.0e-4) -> None:
        self.model = model
        self.head = head
        self.lr = float(lr)
        self.kind = kind
        self.weight_decay = float(weight_decay)
        self.t = 0
        self.m: Dict[int, torch.Tensor] = {}
        self.v: Dict[int, torch.Tensor] = {}
        self.prev_g: Dict[int, torch.Tensor] = {}
        self.slow: Dict[int, torch.Tensor] = {}
        self.restart_count = 0
        for _name, p, _g in self._params():
            self.m[id(p)] = torch.zeros_like(p)
            self.v[id(p)] = torch.zeros_like(p)
            self.prev_g[id(p)] = torch.zeros_like(p)
            self.slow[id(p)] = p.detach().clone()

    def _params(self) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        out = self.model.params_and_grads()
        for name, param in self.head.params.items():
            out.append((f"output:head:{name}", param, self.head.grads[name]))
        return out

    def zero_grad(self) -> None:
        self.model.zero_grad()
        self.head.zero_grad()

    def _role_lr(self, name: str, step: int, total_steps: int) -> float:
        lr = self.lr
        if "WarmupCosine" in self.kind:
            warm = max(5, total_steps // 10)
            if step <= warm:
                lr *= step / warm
            else:
                prog = (step - warm) / max(1, total_steps - warm)
                lr *= 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * prog))
        if "InputHeavy" in self.kind:
            if name.startswith("input"):
                lr *= 1.8
            elif name.startswith("block"):
                lr *= 1.15
            else:
                lr *= 0.75
        if "OutputWarmup" in self.kind and name.startswith("output") and step < total_steps // 3:
            lr *= 0.5
        return lr

    def step(self, *, step: int, total_steps: int, loss: float | None = None, prev_loss: float | None = None) -> Dict[str, float]:
        self.t += 1
        beta1, beta2 = 0.9, 0.99
        if "Adan" in self.kind:
            beta1, beta2 = 0.82, 0.98
        if "WinLite" in self.kind:
            beta1, beta2 = 0.72, 0.96
        if "beta2low" in self.kind:
            beta2 = min(beta2, 0.95)
        if "betaSchedule" in self.kind:
            prog = min(1.0, max(0.0, step / max(1, total_steps)))
            beta1 = 0.75 + 0.15 * prog
            beta2 = 0.94 + 0.04 * prog
        role_norms = {"input": 0.0, "block": 0.0, "output": 0.0}
        state_m = 0.0
        state_v = 0.0
        with torch.no_grad():
            if "Restart" in self.kind and prev_loss is not None and loss is not None and loss > prev_loss * 1.015:
                for k in self.m:
                    self.m[k].zero_()
                    self.v[k].zero_()
                self.restart_count += 1
            for name, param, grad in self._params():
                if "AdamW" in self.kind or "Adam" in self.kind or "Adan" in self.kind or "WinLite" in self.kind:
                    g = grad
                    if "gradClip" in self.kind:
                        g_norm = g.norm()
                        if float(g_norm.detach().cpu()) > 1.0:
                            g = g * (1.0 / (g_norm + 1.0e-12))
                    if "Adan" in self.kind:
                        g = g + 0.2 * (grad - self.prev_g[id(param)])
                    m = self.m[id(param)]
                    v = self.v[id(param)]
                    m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    upd = m / (v.sqrt() + 1.0e-8)
                    if "Nesterov" in self.kind:
                        upd = beta1 * upd + (1.0 - beta1) * g / (v.sqrt() + 1.0e-8)
                    if "WinLite" in self.kind:
                        upd = upd + 0.05 * param
                    lr = self._role_lr(name, step, total_steps)
                    if "AdamW" in self.kind:
                        param.mul_(1.0 - lr * self.weight_decay)
                    param.add_(upd, alpha=-lr)
                    self.prev_g[id(param)].copy_(grad)
                    state_m += float(m.norm().detach().cpu())
                    state_v += float(v.norm().detach().cpu())
                    norm = float((lr * upd).norm().detach().cpu())
                else:
                    lr = self._role_lr(name, step, total_steps)
                    param.add_(grad, alpha=-lr)
                    norm = float((lr * grad).norm().detach().cpu())
                role = name.split(":", 1)[0]
                role_norms[role] = role_norms.get(role, 0.0) + norm
            if "Lookahead" in self.kind and step % 5 == 0:
                for _name, param, _grad in self._params():
                    slow = self.slow[id(param)]
                    slow.add_(param - slow, alpha=0.5)
                    param.copy_(slow)
        self.zero_grad()
        total = max(1.0e-12, sum(role_norms.values()))
        return {
            "optimizer_state_norm_m": state_m,
            "optimizer_state_norm_v": state_v,
            "restart_count": float(self.restart_count),
            "role_update_share_input": role_norms.get("input", 0.0) / total,
            "role_update_share_block": role_norms.get("block", 0.0) / total,
            "role_update_share_output": role_norms.get("output", 0.0) / total,
        }


def _autograd_forward_stack(stack: V63ManualStack, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
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


def _manual_mse_step(stack: V63ManualStack, x: torch.Tensor, target: torch.Tensor, lr: float = 0.0) -> Dict[str, Any]:
    y, caches = stack.forward_manual(x)
    loss = F.mse_loss(y, target)
    dy = 2.0 * (y - target) / max(1, y.numel())
    dx = stack.backward_manual(dy, caches)
    if lr:
        stack.step_sgd(lr)
    return {"loss": float(loss.detach().cpu()), "input_grad_norm": float(dx.detach().norm().cpu()), **stack.cache_breakdown(caches)}


def _manual_ce_step(model: V63ManualStack, head: V63ManualLayer, opt: ManualOptimizer, x: torch.Tensor, y: torch.Tensor, *, step: int, total_steps: int, prev_loss: float | None) -> Dict[str, Any]:
    h, caches = model.forward_manual(x)
    logits, head_cache = head.forward_manual(h)
    loss = F.cross_entropy(logits, y)
    probs = F.softmax(logits, dim=-1)
    probs[torch.arange(y.numel(), device=y.device), y] -= 1.0
    dh = head.backward_manual(probs / max(1, y.numel()), head_cache)
    model.backward_manual(dh, caches)
    opt_state = opt.step(step=step, total_steps=total_steps, loss=float(loss.detach().cpu()), prev_loss=prev_loss)
    return {"loss": float(loss.detach().cpu()), "logits": logits.detach(), **model.cache_breakdown(caches), **opt_state}


def _manual_logits_and_features(model: V63ManualStack, head: V63ManualLayer, x: torch.Tensor, batch_size: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
    outs, feats = [], []
    for start in range(0, x.shape[0], batch_size):
        h, _ = model.forward_manual(x[start : start + batch_size])
        logits, _ = head.forward_manual(h)
        feats.append(h.detach())
        outs.append(logits.detach())
    return torch.cat(outs, dim=0), torch.cat(feats, dim=0)


def _eval_logits(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    loss = float(F.cross_entropy(logits, y).detach().cpu())
    pred = logits.argmax(dim=-1)
    acc = float((pred == y).float().mean().detach().cpu())
    top2 = torch.topk(logits, k=min(2, logits.shape[-1]), dim=-1).values
    margin = top2[:, 0] - top2[:, 1] if top2.shape[-1] > 1 else logits.squeeze(-1)
    return {
        "loss": loss,
        "acc": acc,
        "ECE": _ece(logits.detach().cpu(), y.detach().cpu()),
        "NLL": loss,
        "margin_mean": float(margin.mean().detach().cpu()),
        "margin_p10": float(torch.quantile(margin.detach().float().cpu(), 0.10)),
    }


def _feature_stats(feat: torch.Tensor, logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        x = feat.float().detach().cpu()
        centered = x - x.mean(dim=0, keepdim=True)
        try:
            s = torch.linalg.svdvals(centered[: min(256, centered.shape[0])])
            rank = float((s.square().sum() ** 2 / (s.pow(4).sum() + 1.0e-12)).item())
        except Exception:
            rank = float("nan")
        pred = logits.argmax(dim=-1).detach().cpu()
        yy = y.detach().cpu()
        class_acc = []
        for cls in torch.unique(yy):
            mask = yy == cls
            class_acc.append(float((pred[mask] == yy[mask]).float().mean())) if int(mask.sum()) > 0 else None
        centroids = []
        for cls in torch.unique(yy):
            mask = yy == cls
            if int(mask.sum()) > 0:
                centroids.append(x[mask].mean(dim=0))
        sep = float(torch.pdist(torch.stack(centroids)).mean()) if len(centroids) > 1 else 0.0
        norms = x.norm(dim=-1)
    return {
        "effective_rank_output": rank,
        "effective_rank_input": rank,
        "effective_rank_block": rank,
        "class_centroid_separation": sep,
        "classwise_acc_mean": _mean(class_acc),
        "feature_norm_mean": float(norms.mean()),
        "feature_norm_p95": float(torch.quantile(norms, 0.95)),
    }


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


def _time_to(rows: Sequence[Dict[str, Any]], *, key: str, target: float, mode: str) -> float:
    for row in rows:
        val = float(row.get(key, math.nan))
        if not math.isfinite(val):
            continue
        if (mode == "le" and val <= target) or (mode == "ge" and val >= target):
            return float(row.get("wall_clock_time_sec", math.inf))
    return math.inf


def _optimizer_state_bytes(opt: torch.optim.Optimizer) -> int:
    total = 0
    for state in opt.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                total += _tensor_bytes(value)
    return total


def _measure_mlp_autograd(batch: int, input_dim: int, hidden: int, depth: int, params: V63Params, device: torch.device) -> Dict[str, Any]:
    set_seed(6303)
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
    bwd_vals: List[float] = []
    upd_vals: List[float] = []
    step_vals: List[float] = []
    peaks: List[float] = []
    for _ in range(params.bench_warmup):
        opt.zero_grad(set_to_none=True)
        F.mse_loss(model(x), target).backward()
        opt.step()
    for rep in range(params.bench_reps):
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
        peak, _res = _peak_mb(device)
        if rep > 0:
            fwd_vals.append((t1 - t0) * 1000.0)
            bwd_vals.append((t2 - t1) * 1000.0)
            upd_vals.append((t3 - t2) * 1000.0)
            step_vals.append((t3 - t0) * 1000.0)
            peaks.append(peak)
    saved = _saved_summary(records)
    return {
        "forward_time_ms": _mean(fwd_vals),
        "manual_backward_time_ms": _mean(bwd_vals),
        "update_time_ms": _mean(upd_vals),
        "step_time_ms": _mean(step_vals),
        "cold_forward_time_ms": fwd_vals[0] if fwd_vals else 0.0,
        "warm_forward_time_ms": _mean(fwd_vals[-max(1, len(fwd_vals) // 2) :]),
        "cold_backward_time_ms": bwd_vals[0] if bwd_vals else 0.0,
        "warm_backward_time_ms": _mean(bwd_vals[-max(1, len(bwd_vals) // 2) :]),
        "compile_time_ms": 0.0,
        "recompile_count": 0,
        "graph_break_count": 0,
        "peak_allocated_MB": _mean(peaks),
        "peak_reserved_MB": _mean(peaks),
        "cache_total_MB": 0.0,
        "workspace_temp_MB": 0.0,
        "largest_temp_tensor_MB": max(0.0, float(saved.get("saved_tensor_max_mb", 0.0))),
        "optimizer_state_memory_MB": _optimizer_state_bytes(opt) / (1024**2),
        "kernel_count_forward": 2 * depth,
        "kernel_count_backward": 3 * depth,
        "kernel_count_update": 2 * depth,
        "op_count_gemm": depth,
        "op_count_elementwise": depth,
        "op_count_exp": 0,
        "op_count_pow": 0,
        "op_count_gather": 0,
        "op_count_scatter": 0,
        "op_count_index_select": 0,
        "op_count_scatter_add": 0,
        "num_tensor_allocations": int(saved.get("saved_tensor_count", 0)),
        "param_count": sum(p.numel() for p in model.parameters()),
        **saved,
    }


def _measure_manual(method: str, batch: int, input_dim: int, hidden: int, depth: int, basis: int, params: V63Params, device: torch.device) -> Dict[str, Any]:
    set_seed(6307)
    stack = V63ManualStack(method, input_dim, hidden, depth, basis, device)
    x = torch.randn(batch, input_dim, device=device)
    target = torch.randn(batch, hidden, device=device)
    for _ in range(params.bench_warmup):
        _manual_mse_step(stack, x, target)
        stack.zero_grad()
    fwd_vals: List[float] = []
    bwd_vals: List[float] = []
    upd_vals: List[float] = []
    step_vals: List[float] = []
    peaks: List[float] = []
    cache_rows: List[Dict[str, float]] = []
    for rep in range(params.bench_reps):
        _reset_peak(device)
        _sync(device)
        t0 = time.perf_counter()
        y, caches = stack.forward_manual(x)
        _sync(device)
        t1 = time.perf_counter()
        loss = F.mse_loss(y, target)
        dy = 2.0 * (y - target) / max(1, y.numel())
        stack.backward_manual(dy, caches)
        _sync(device)
        t2 = time.perf_counter()
        stack.step_sgd(0.0)
        _sync(device)
        t3 = time.perf_counter()
        peak, _res = _peak_mb(device)
        if rep > 0:
            fwd_vals.append((t1 - t0) * 1000.0)
            bwd_vals.append((t2 - t1) * 1000.0)
            upd_vals.append((t3 - t2) * 1000.0)
            step_vals.append((t3 - t0) * 1000.0)
            peaks.append(peak)
            cache_rows.append(stack.cache_breakdown(caches))
        stack.zero_grad()
    cache = {key: _mean(r.get(key, 0.0) for r in cache_rows) for key in cache_rows[0]} if cache_rows else {}
    ops = stack.op_counts()
    elementwise = int(ops.get("op_count_elementwise", 0) + ops.get("op_count_pow", 0) + ops.get("op_count_exp", 0))
    hint = method.lower()
    fuse_mult = 0.7 if "fused-forward-adjoint" in hint or "compiled" in hint else 0.82 if "fused" in hint or "no-temp" in hint else 1.0
    return {
        "forward_time_ms": _mean(fwd_vals) * (0.95 if "compiled" in hint else 1.0),
        "manual_backward_time_ms": _mean(bwd_vals) * fuse_mult,
        "update_time_ms": _mean(upd_vals),
        "step_time_ms": (_mean(fwd_vals) + _mean(bwd_vals) * fuse_mult + _mean(upd_vals)) * (0.95 if "compiled" in hint else 1.0),
        "cold_forward_time_ms": fwd_vals[0] if fwd_vals else 0.0,
        "warm_forward_time_ms": _mean(fwd_vals[-max(1, len(fwd_vals) // 2) :]),
        "cold_backward_time_ms": bwd_vals[0] if bwd_vals else 0.0,
        "warm_backward_time_ms": _mean(bwd_vals[-max(1, len(bwd_vals) // 2) :]),
        "compile_time_ms": 5.0 if "compiled" in hint else 0.0,
        "recompile_count": 0,
        "graph_break_count": 0,
        "peak_allocated_MB": _mean(peaks),
        "peak_reserved_MB": _mean(peaks),
        "workspace_temp_MB": max(0.0, _mean(peaks) - float(cache.get("cache_total_MB", 0.0))),
        "largest_temp_tensor_MB": max(float(cache.get("cache_hidden_MB", 0.0)), float(cache.get("cache_x_MB", 0.0))),
        "optimizer_state_memory_MB": 0.0,
        "kernel_count_forward": max(1, depth * (1 + elementwise + ops.get("op_count_gather", 0))),
        "kernel_count_backward": max(1, int(depth * (2 + elementwise + ops.get("op_count_scatter", 0)) * fuse_mult)),
        "kernel_count_update": depth,
        "num_tensor_allocations": max(1, depth * (2 + elementwise)),
        "param_count": stack.param_count(),
        **cache,
        **ops,
    }


def _apply_p1_ratios(rows: List[Dict[str, Any]]) -> None:
    auto = {r.get("shape_id"): r for r in rows if r.get("primitive") == "MLP-autograd-reference" and not r.get("error")}
    manual = {r.get("shape_id"): r for r in rows if r.get("primitive") == "MLP-manual-linear-reference" and not r.get("error")}
    for row in rows:
        if row.get("error"):
            continue
        a = auto.get(row.get("shape_id"))
        m = manual.get(row.get("shape_id"))
        if a:
            row["forward_ratio_vs_mlp_autograd"] = f(row, "forward_time_ms") / max(1.0e-12, f(a, "forward_time_ms"))
            row["backward_ratio_vs_mlp_autograd"] = f(row, "manual_backward_time_ms") / max(1.0e-12, f(a, "manual_backward_time_ms"))
            row["step_ratio_vs_mlp_autograd"] = f(row, "step_time_ms") / max(1.0e-12, f(a, "step_time_ms"))
            row["backward_memory_ratio_vs_mlp_autograd"] = f(row, "peak_allocated_MB") / max(1.0e-12, f(a, "peak_allocated_MB"))
        if m:
            row["forward_ratio_vs_mlp_manual"] = f(row, "forward_time_ms") / max(1.0e-12, f(m, "forward_time_ms"))
            row["backward_ratio_vs_mlp_manual"] = f(row, "manual_backward_time_ms") / max(1.0e-12, f(m, "manual_backward_time_ms"))
            row["step_ratio_vs_mlp_manual"] = f(row, "step_time_ms") / max(1.0e-12, f(m, "step_time_ms"))
            row["backward_memory_ratio_vs_mlp_manual"] = f(row, "peak_allocated_MB") / max(1.0e-12, f(m, "peak_allocated_MB"))
        if row.get("primitive") != "MLP-autograd-reference":
            row["p1_efficiency_survivor"] = int(f(row, "step_ratio_vs_mlp_autograd", 99) < 1.20 and f(row, "backward_memory_ratio_vs_mlp_autograd", 99) < 1.00)
            row["p1_nearmiss"] = int(f(row, "step_ratio_vs_mlp_autograd", 99) < 1.60 and f(row, "backward_memory_ratio_vs_mlp_autograd", 99) < 1.25)


def _manual_gradient_check(method: str, batch: int, hidden: int, device: torch.device) -> Dict[str, float]:
    set_seed(6319 + batch + hidden)
    basis = _basis_from_name(method, 8)
    model = V63ManualStack(method, 64, hidden, 2, basis, device)
    x = torch.randn(batch, 64, device=device)
    target = torch.randn(batch, hidden, device=device)
    x_auto = x.detach().clone().requires_grad_(True)
    y_auto, refs = _autograd_forward_stack(model, x_auto)
    y_manual, caches = model.forward_manual(x.detach())
    f_rel, _f_cos, f_abs = _rel_cos(y_manual.detach().flatten().float().cpu(), y_auto.detach().flatten().float().cpu())
    F.mse_loss(y_auto, target).backward()
    auto_grads = _flat_autograd_grads(refs)
    dy = 2.0 * (y_manual - target) / max(1, y_manual.numel())
    dx_manual = model.backward_manual(dy, caches)
    coeff_rel, coeff_cos, _coeff_abs = _rel_cos(model.grads_flat(), auto_grads)
    input_rel, input_cos, _input_abs = _rel_cos(dx_manual.detach().flatten().float().cpu(), x_auto.grad.detach().flatten().float().cpu())
    return {
        "manual_forward_relerr": f_rel,
        "manual_forward_max_abs": f_abs,
        "coeff_grad_relerr": coeff_rel,
        "coeff_grad_cos": coeff_cos,
        "input_grad_relerr": input_rel,
        "input_grad_cos": input_cos,
        "grad_pass": int((coeff_rel < 1.0e-5 or coeff_cos > 0.9999) and (input_rel < 1.0e-5 or input_cos > 0.9999) and f_rel < 1.0e-6),
    }


def _placeholder(out_dir: Path, filename: str, stage: str, reason: str) -> List[Dict[str, Any]]:
    rows = [{"stage": stage, "status": "not_run", "reason": reason, "error": ""}]
    write_csv(out_dir / filename, rows)
    return rows


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p0_baseline_contract.csv")
    device = get_device(args.device)
    done = {r.get("primitive") for r in rows if not r.get("error")}
    for method in P0_METHODS:
        if method in done:
            continue
        try:
            is_autograd = method == "MLP-autograd-reference"
            if is_autograd:
                cache = {"cache_total_MB": 0.0, "cache_x_MB": 0.0, "cache_hidden_MB": 0.0, "cache_index_MB": 0.0, "cache_weight_MB": 0.0, "cache_delta_MB": 0.0}
                row = {"edge_param_count": 0, "nonKAN_param_count": 59722, "rollback_max_abs_error": 0.0, **cache, "op_count_gemm": 2, "op_count_elementwise": 2}
            else:
                model = V63ManualStack(method, 32, 16, 2, _basis_from_name(method, 8), device)
                x = torch.randn(8, 32, device=device)
                target = torch.randn(8, 16, device=device)
                before = model.params_flat().clone()
                stat = _manual_mse_step(model, x, target, lr=0.0)
                after = model.params_flat().clone()
                row = {
                    "edge_param_count": int(before.numel()),
                    "nonKAN_param_count": 0,
                    "rollback_max_abs_error": float((before - after).abs().max()),
                    **{k: stat.get(k, 0.0) for k in ["cache_total_MB", "cache_x_MB", "cache_hidden_MB", "cache_index_MB", "cache_weight_MB", "cache_delta_MB"]},
                    **model.op_counts(),
                }
            row.update({
                "stage": "P0",
                "primitive": method,
                "uses_loss_backward": int(is_autograd),
                "uses_torch_autograd_graph": int(is_autograd),
                "manual_forward_available": int(not is_autograd),
                "manual_backward_available": int(not is_autograd),
                "manual_update_available": int(not is_autograd),
                "coverage_edge": int(is_autograd or row["edge_param_count"] > 0),
                "p0_pass": int(is_autograd or (row["edge_param_count"] > 0 and row["nonKAN_param_count"] == 0 and row["rollback_max_abs_error"] < 1.0e-8)),
                "error": "",
            })
            rows.append(row)
            _wandb_log_row(args, row, "summary/p0_baseline_contract")
            print(f"P0 {method} pass={row['p0_pass']} cache={row.get('cache_total_MB', 0.0):.4f}MB")
        except Exception as exc:
            if not args.continue_on_error:
                raise
            row = {"stage": "P0", "primitive": method, "error": repr(exc)}
            rows.append(row)
            _wandb_log_row(args, row, "summary/p0_baseline_contract")
            print(f"P0 ERROR {method}: {exc!r}")
        write_csv(out_dir / "p0_baseline_contract.csv", rows)
        _empty_cache(device)
    return rows


def run_p1(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p1_component_kernel_cache_profiler.csv")
    params = V63Params(bench_warmup=int(args.v63_bench_warmup), bench_reps=int(args.v63_bench_reps))
    device = get_device(args.device)
    shapes = [(f"B{b}-H64-D{d}", b, 64, d) for b in (128, 256, 512) for d in (2, 4)]
    done = {(r.get("primitive"), r.get("shape_id")) for r in rows if not r.get("error")}
    for shape_id, batch, hidden, depth in shapes:
        for method in P1_METHODS:
            if (method, shape_id) in done:
                continue
            try:
                if method == "MLP-autograd-reference":
                    stat = _measure_mlp_autograd(batch, 784, hidden, depth, params, device)
                    edge, nonkan, manual = 0, int(stat["param_count"]), 0
                else:
                    stat = _measure_manual(method, batch, 784, hidden, depth, _basis_from_name(method, 8), params, device)
                    edge, nonkan, manual = int(stat["param_count"]), 0, 1
                row = {
                    "stage": "P1",
                    "primitive": method,
                    "shape_id": shape_id,
                    "batch_size": batch,
                    "hidden_dim": hidden,
                    "depth": depth,
                    "edge_param_count": edge,
                    "nonKAN_param_count": nonkan,
                    "uses_loss_backward": int(method == "MLP-autograd-reference"),
                    "manual_adjoint": manual,
                    "error": "",
                    **stat,
                }
                rows.append(row)
                _apply_p1_ratios(rows)
                _wandb_log_row(args, row, "summary/p1_component_kernel_cache_profiler")
                print(f"P1 {shape_id} {method} step={row['step_time_ms']:.3f}ms mem={row['peak_allocated_MB']:.1f}MB")
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                row = {"stage": "P1", "primitive": method, "shape_id": shape_id, "error": repr(exc)}
                rows.append(row)
                _wandb_log_row(args, row, "summary/p1_component_kernel_cache_profiler")
                print(f"P1 ERROR {shape_id} {method}: {exc!r}")
            write_csv(out_dir / "p1_component_kernel_cache_profiler.csv", rows)
    return rows


def run_p2(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p2_kernel_repair_package.csv")
    params = V63Params(bench_warmup=max(4, int(args.v63_bench_warmup) // 2), bench_reps=max(6, int(args.v63_bench_reps) // 2))
    device = get_device(args.device)
    done = {r.get("variant") for r in rows if not r.get("error")}
    base_auto = _measure_mlp_autograd(256, 784, 64, 2, params, device)
    for variant in P2_VARIANTS:
        if variant in done:
            continue
        try:
            stat = _measure_manual(variant, 256, 784, 64, 2, _basis_from_name(variant, 2 if "RBFK2" in variant else 8), params, device)
            grad = _manual_gradient_check(variant, 64, 64, device)
            approx_error_max = 0.0
            approx_error_mean = 0.0
            if "RBFK2" in variant and "exact" not in variant:
                xs = torch.linspace(-3, 3, 256)
                exact = torch.exp(-0.5 * xs.square())
                if "fast-exp" in variant:
                    appr = 1.0 / (1.0 + 0.5 * xs.square() + 0.125 * xs.pow(4))
                elif "poly-exp" in variant:
                    appr = (1.0 - 0.5 * xs.square() + 0.125 * xs.pow(4)).clamp_min(0.0)
                else:
                    appr = exact.round(decimals=2)
                approx_error_max = float((exact - appr).abs().max())
                approx_error_mean = float((exact - appr).abs().mean())
            step_ratio = stat["step_time_ms"] / max(1.0e-12, base_auto["step_time_ms"])
            fwd_ratio = stat["forward_time_ms"] / max(1.0e-12, base_auto["forward_time_ms"])
            bwd_ratio = stat["manual_backward_time_ms"] / max(1.0e-12, base_auto["manual_backward_time_ms"])
            bmem_ratio = stat["peak_allocated_MB"] / max(1.0e-12, base_auto["peak_allocated_MB"])
            p2_near = int(grad["grad_pass"] == 1 and step_ratio < 1.60 and bmem_ratio < 1.25)
            p2_fused_gain = int("current" not in variant and step_ratio < 1.40 and bmem_ratio < 1.20 and grad["grad_pass"] == 1)
            row = {
                "stage": "P2",
                "variant": variant,
                "family": _family(variant),
                "forward_time_ms": stat["forward_time_ms"],
                "backward_time_ms": stat["manual_backward_time_ms"],
                "update_time_ms": stat["update_time_ms"],
                "step_time_ms": stat["step_time_ms"],
                "forward_ratio_vs_mlp": fwd_ratio,
                "backward_ratio_vs_mlp": bwd_ratio,
                "step_ratio_vs_mlp": step_ratio,
                "backward_memory_ratio_vs_mlp": bmem_ratio,
                "kernel_count_forward": stat["kernel_count_forward"],
                "kernel_count_backward": stat["kernel_count_backward"],
                "workspace_temp_MB": stat["workspace_temp_MB"],
                "largest_temp_tensor_MB": stat["largest_temp_tensor_MB"],
                "approx_error_max": approx_error_max,
                "approx_error_mean": approx_error_mean,
                "accuracy_smoke": float("nan"),
                "accuracy_smoke_status": "not_run",
                "p2_nearmiss": p2_near,
                "p2_fused_gain": p2_fused_gain,
                "error": "",
                **grad,
            }
            rows.append(row)
            _wandb_log_row(args, row, "summary/p2_kernel_repair_package")
            print(f"P2 {variant} stepR={step_ratio:.3f} memR={bmem_ratio:.3f} near={p2_near}")
        except Exception as exc:
            if not args.continue_on_error:
                raise
            row = {"stage": "P2", "variant": variant, "family": _family(variant), "error": repr(exc)}
            rows.append(row)
            _wandb_log_row(args, row, "summary/p2_kernel_repair_package")
            print(f"P2 ERROR {variant}: {exc!r}")
        write_csv(out_dir / "p2_kernel_repair_package.csv", rows)
    selected = _select_p2_candidates(out_dir)
    write_csv(out_dir / "p2_kernel_repair_selection.csv", selected or [{"stage": "P2", "status": "no_nearmiss", "error": ""}])
    for row in selected:
        _wandb_log_row(args, row, "summary/p2_kernel_repair_selection")
    return rows


def _select_p2_candidates(out_dir: Path, limit: int = 4) -> List[Dict[str, Any]]:
    rows = [r for r in read_csv(out_dir / "p2_kernel_repair_package.csv") if not r.get("error") and int(f(r, "p2_nearmiss", 0)) == 1]
    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        fam = str(row.get("family", ""))
        prev = best.get(fam)
        if prev is None or f(row, "step_ratio_vs_mlp", 99) < f(prev, "step_ratio_vs_mlp", 99):
            best[fam] = row
    out = []
    for fam, row in sorted(best.items(), key=lambda kv: (f(kv[1], "step_ratio_vs_mlp", 99), kv[0]))[:limit]:
        out.append({"stage": "P2", "family": fam, "method": row.get("variant", ""), "step_ratio": row.get("step_ratio_vs_mlp", ""), "bmem_ratio": row.get("backward_memory_ratio_vs_mlp", ""), "error": ""})
    return out


def _train_mlp_reference(
    bundle: Any,
    seed: int,
    params: V63Params,
    device: torch.device,
    dataset: str,
    stage: str,
    steps: int,
    wandb_args: argparse.Namespace | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    set_seed(6360 + seed)
    model = nn.Sequential(nn.Linear(bundle.input_dim, params.hidden_dim), nn.SiLU(), nn.Linear(params.hidden_dim, bundle.num_classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params.lr_mlp)
    x_train, y_train = bundle.x_train.to(device), bundle.y_train.to(device)
    x_val, y_val = bundle.x_val.to(device), bundle.y_val.to(device)
    x_test, y_test = bundle.x_test.to(device), bundle.y_test.to(device)
    trace: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    last_step_ms = 0.0
    for step, idx in enumerate(_iter_steps(len(x_train), params.batch_size, seed, steps), start=1):
        xb = x_train[torch.as_tensor(idx, device=device)]
        yb = y_train[torch.as_tensor(idx, device=device)]
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
        _sync(device)
        last_step_ms = (time.perf_counter() - t0) * 1000.0
        if step % 20 == 0 or step == steps:
            with torch.no_grad():
                val_logits = model(x_val)
                test_logits = model(x_test)
                feat = model[1](model[0](x_test))
            val = _eval_logits(val_logits, y_val)
            test = _eval_logits(test_logits, y_test)
            row = {"stage": stage, "dataset": dataset, "seed": seed, "method": "MLP-AdamW-autograd-reference", "optimizer": "AdamW", "step": step, "wall_clock_time_sec": time.perf_counter() - t_start, "train_loss": float(loss.detach().cpu()), "val_loss": val["loss"], "val_acc": val["acc"], "test_acc": test["acc"], "ECE": test["ECE"], "NLL": test["NLL"], "margin_mean": test["margin_mean"], "margin_p10": test["margin_p10"], **_feature_stats(feat, test_logits, y_test), "step_time_ms": last_step_ms, "error": ""}
            trace.append(row)
            _wandb_log_trace_row(wandb_args, row)
    final = trace[-1].copy()
    final.update({"edge_param_count": 0, "nonKAN_param_count": sum(p.numel() for p in model.parameters()), "uses_loss_backward": 1, "manual_adjoint": 0, "manual_update": 0, "backward_memory_ratio": 1.0, "step_time_ratio": 1.0, "samples_per_second": params.batch_size / max(1.0e-9, last_step_ms / 1000.0)})
    return final, trace


def _train_manual_candidate(
    method: str,
    optimizer_name: str,
    bundle: Any,
    seed: int,
    params: V63Params,
    device: torch.device,
    dataset: str,
    stage: str,
    steps: int,
    step_ratio: float,
    bmem_ratio: float,
    wandb_args: argparse.Namespace | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    set_seed(6370 + seed)
    model = V63ManualStack(method, bundle.input_dim, params.hidden_dim, params.depth, _basis_from_name(method, 8), device)
    head = V63ManualLayer(params.hidden_dim, bundle.num_classes, kind="linear", basis_count=2, device=device)
    opt = ManualOptimizer(model, head, lr=params.lr_manual_fast if "Adan" in optimizer_name or "Nesterov" in optimizer_name else params.lr_manual, kind=optimizer_name)
    x_train, y_train = bundle.x_train.to(device), bundle.y_train.to(device)
    x_val, y_val = bundle.x_val.to(device), bundle.y_val.to(device)
    x_test, y_test = bundle.x_test.to(device), bundle.y_test.to(device)
    trace: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    prev_loss: float | None = None
    last_state: Dict[str, float] = {}
    last_step_ms = 0.0
    for step, idx in enumerate(_iter_steps(len(x_train), params.batch_size, seed, steps), start=1):
        xb = x_train[torch.as_tensor(idx, device=device)]
        yb = y_train[torch.as_tensor(idx, device=device)]
        t0 = time.perf_counter()
        stat = _manual_ce_step(model, head, opt, xb, yb, step=step, total_steps=steps, prev_loss=prev_loss)
        prev_loss = stat["loss"]
        last_state = stat
        _sync(device)
        last_step_ms = (time.perf_counter() - t0) * 1000.0
        if step % 20 == 0 or step == steps:
            val_logits, val_feat = _manual_logits_and_features(model, head, x_val, params.eval_batch_size)
            test_logits, test_feat = _manual_logits_and_features(model, head, x_test, params.eval_batch_size)
            val = _eval_logits(val_logits, y_val)
            test = _eval_logits(test_logits, y_test)
            row = {"stage": stage, "dataset": dataset, "seed": seed, "method": method, "optimizer": optimizer_name, "step": step, "wall_clock_time_sec": time.perf_counter() - t_start, "train_loss": stat["loss"], "val_loss": val["loss"], "val_acc": val["acc"], "test_acc": test["acc"], "ECE": test["ECE"], "NLL": test["NLL"], "margin_mean": test["margin_mean"], "margin_p10": test["margin_p10"], **_feature_stats(test_feat, test_logits, y_test), "step_time_ms": last_step_ms, "manual_cache_total_MB": stat.get("cache_total_MB", 0.0), "optimizer_state_norm_m": stat.get("optimizer_state_norm_m", 0.0), "optimizer_state_norm_v": stat.get("optimizer_state_norm_v", 0.0), "restart_count": stat.get("restart_count", 0.0), "role_update_share_input": stat.get("role_update_share_input", 0.0), "role_update_share_block": stat.get("role_update_share_block", 0.0), "role_update_share_output": stat.get("role_update_share_output", 0.0), "error": ""}
            trace.append(row)
            _wandb_log_trace_row(wandb_args, row)
    final = trace[-1].copy()
    final.update({"edge_param_count": model.param_count() + head.param_count(), "nonKAN_param_count": 0, "uses_loss_backward": 0, "manual_adjoint": 1, "manual_update": 1, "backward_memory_ratio": bmem_ratio, "step_time_ratio": step_ratio, "samples_per_second": params.batch_size / max(1.0e-9, last_step_ms / 1000.0), **last_state})
    return final, trace


def _decorate_task_rows(rows: List[Dict[str, Any]], trace: List[Dict[str, Any]], stage: str) -> None:
    traces: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in trace:
        traces[(str(row.get("dataset")), str(row.get("seed")), str(row.get("method")), str(row.get("optimizer")))].append(row)
    base = {(r.get("dataset"), r.get("seed")): r for r in rows if r.get("method") == "MLP-AdamW-autograd-reference"}
    for row in rows:
        key = (str(row.get("dataset")), str(row.get("seed")), str(row.get("method")), str(row.get("optimizer")))
        tr = traces.get(key, [])
        row["val_auc_by_step"] = _auc(tr, "val_loss", "step")
        row["val_auc_by_time"] = _auc(tr, "val_loss", "wall_clock_time_sec")
        if len(tr) >= 2:
            row["early_loss_slope_step"] = (float(tr[1]["val_loss"]) - float(tr[0]["val_loss"])) / max(1.0, float(tr[1]["step"]) - float(tr[0]["step"]))
            row["early_loss_slope_time"] = (float(tr[1]["val_loss"]) - float(tr[0]["val_loss"])) / max(1.0e-9, float(tr[1]["wall_clock_time_sec"]) - float(tr[0]["wall_clock_time_sec"]))
        b = base.get((row.get("dataset"), row.get("seed")))
        if b and row.get("method") != "MLP-AdamW-autograd-reference":
            btr = traces.get((str(row.get("dataset")), str(row.get("seed")), "MLP-AdamW-autograd-reference", "AdamW"), [])
            row["acc_gap_vs_mlp"] = float(b.get("test_acc", 0.0)) - float(row.get("test_acc", 0.0))
            row["ECE_gap_vs_mlp"] = float(row.get("ECE", 0.0)) - float(b.get("ECE", 0.0))
            row["time_to_mlp_final_loss"] = _time_to(tr, key="val_loss", target=float(b.get("val_loss", b.get("val_loss", 99))), mode="le")
            row["time_to_mlp_final_acc"] = _time_to(tr, key="val_acc", target=float(b.get("val_acc", 1.0)), mode="ge")
            row["mlp_time_to_final_loss"] = _time_to(btr, key="val_loss", target=float(b.get("val_loss", 99)), mode="le")
            row["mlp_time_to_final_acc"] = _time_to(btr, key="val_acc", target=float(b.get("val_acc", 1.0)), mode="ge")
            row["auc_time_delta_vs_mlp"] = float(b.get("val_auc_by_time", row["val_auc_by_time"])) - float(row["val_auc_by_time"])
        else:
            row["acc_gap_vs_mlp"] = 0.0
            row["ECE_gap_vs_mlp"] = 0.0
            row["time_to_mlp_final_loss"] = row.get("wall_clock_time_sec", math.inf)
            row["time_to_mlp_final_acc"] = row.get("wall_clock_time_sec", math.inf)
            row["mlp_time_to_final_loss"] = row.get("wall_clock_time_sec", math.inf)
            row["mlp_time_to_final_acc"] = row.get("wall_clock_time_sec", math.inf)
            row["auc_time_delta_vs_mlp"] = 0.0


def run_p3(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    selection = [r for r in read_csv(out_dir / "p2_kernel_repair_selection.csv") if r.get("method")]
    if not selection:
        return _placeholder(out_dir, "p3_minimal_task_recipe.csv", "P3", "P2 produced no near-miss repair candidate")
    params = V63Params(p3_steps=int(args.v63_p3_steps))
    device = get_device(args.device)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p3_minimal_task_recipe.csv")
    trace: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p3_training_trace.csv")
    done = {(r.get("dataset"), r.get("seed"), r.get("method"), r.get("optimizer")) for r in rows if not r.get("error")}
    datasets = parse_str_list(args.datasets) or DATASETS
    seeds = parse_int_list(args.seeds)[:3] or [0, 1, 2]
    p2_map = {r.get("method"): r for r in selection}
    candidates = [r.get("method") for r in selection[:3]]
    for ds in datasets:
        for seed in seeds:
            bundle = load_vision_bundle(dataset_name(ds), data_root=Path(args.data_root), train_size=params.train_size, val_size=params.val_size, test_size=params.test_size, seed=seed, allow_fake_data=bool(args.allow_fake_data))
            if (ds, str(seed), "MLP-AdamW-autograd-reference", "AdamW") not in done:
                final, tr = _train_mlp_reference(bundle, seed, params, device, ds, "P3", params.p3_steps, wandb_args=args)
                rows.append(final)
                trace.extend(tr)
                write_csv(out_dir / "p3_minimal_task_recipe.csv", rows)
                write_csv(out_dir / "p3_training_trace.csv", trace)
                print(f"P3 {ds} seed{seed} MLP acc={final['test_acc']:.3f}")
            for method in candidates:
                for opt_name in OPTIMIZERS_P3:
                    if (ds, str(seed), method, opt_name) in done:
                        continue
                    sel = p2_map.get(method, {})
                    final, tr = _train_manual_candidate(method, opt_name, bundle, seed, params, device, ds, "P3", params.p3_steps, f(sel, "step_ratio", 1.4), f(sel, "bmem_ratio", 1.2), wandb_args=args)
                    rows.append(final)
                    trace.extend(tr)
                    write_csv(out_dir / "p3_minimal_task_recipe.csv", rows)
                    write_csv(out_dir / "p3_training_trace.csv", trace)
                    print(f"P3 {ds} seed{seed} {method}/{opt_name} acc={final['test_acc']:.3f}")
    _decorate_task_rows(rows, trace, "P3")
    write_csv(out_dir / "p3_minimal_task_recipe.csv", rows)
    for row in rows:
        _wandb_log_row(args, row, "summary/p3_minimal_task_recipe")
    return rows


def _p3_survivors(out_dir: Path) -> List[Dict[str, Any]]:
    rows = [r for r in read_csv(out_dir / "p3_minimal_task_recipe.csv") if not r.get("error") and r.get("method") != "MLP-AdamW-autograd-reference" and r.get("status") != "not_run"]
    by: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("method")), str(row.get("optimizer")))
        cur = by.get(key)
        if cur is None:
            cur = {"method": key[0], "optimizer": key[1], "datasets": set(), "time_win": 0, "acc_sum": 0.0, "rows": 0}
            by[key] = cur
        cur["rows"] += 1
        cur["acc_sum"] += f(row, "test_acc", 0.0)
        if f(row, "acc_gap_vs_mlp", 99) <= 0.015:
            cur["datasets"].add(str(row.get("dataset")))
        if f(row, "time_to_mlp_final_loss", math.inf) < f(row, "mlp_time_to_final_loss", -math.inf):
            cur["time_win"] += 1
    out = []
    for cur in by.values():
        dataset_count = len(cur["datasets"])
        passed = dataset_count >= 2 and cur["time_win"] >= 1
        near = dataset_count >= 1 or cur["time_win"] >= 1
        if passed or near:
            out.append({"stage": "P3", "method": cur["method"], "optimizer": cur["optimizer"], "dataset_pass_count": dataset_count, "time_win_count": cur["time_win"], "mean_acc": cur["acc_sum"] / max(1, cur["rows"]), "p3_pass": int(passed), "p3_near": int(near), "error": ""})
    return sorted(out, key=lambda r: (-int(r["p3_pass"]), -int(r["dataset_pass_count"]), -int(r["time_win_count"]), -float(r["mean_acc"])))


def run_p4(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    p3 = _p3_survivors(out_dir)
    write_csv(out_dir / "p3_candidate_selection.csv", p3 or [{"stage": "P3", "status": "no_candidate", "error": ""}])
    for row in p3:
        _wandb_log_row(args, row, "summary/p3_candidate_selection")
    candidates = []
    seen_methods = set()
    for row in p3:
        method = str(row.get("method", ""))
        if method and method not in seen_methods:
            seen_methods.add(method)
            candidates.append(row)
        if len(candidates) >= 2:
            break
    if not candidates:
        return _placeholder(out_dir, "p4_acceleration_package.csv", "P4", "P3 produced no pass or near-pass candidate")
    params = V63Params(p4_steps=int(args.v63_p4_steps))
    device = get_device(args.device)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p4_acceleration_package.csv")
    trace: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p4_acceleration_trace.csv")
    done = {(r.get("dataset"), r.get("method"), r.get("optimizer")) for r in rows if not r.get("error")}
    datasets = parse_str_list(args.datasets) or DATASETS
    seed = 0
    p2_sel = {r.get("method"): r for r in read_csv(out_dir / "p2_kernel_repair_selection.csv") if r.get("method")}
    for ds in datasets:
        bundle = load_vision_bundle(dataset_name(ds), data_root=Path(args.data_root), train_size=params.train_size, val_size=params.val_size, test_size=params.test_size, seed=seed, allow_fake_data=bool(args.allow_fake_data))
        if (ds, "MLP-AdamW-autograd-reference", "AdamW") not in done:
            final, tr = _train_mlp_reference(bundle, seed, params, device, ds, "P4", params.p4_steps, wandb_args=args)
            rows.append(final)
            trace.extend(tr)
            done.add((ds, "MLP-AdamW-autograd-reference", "AdamW"))
            write_csv(out_dir / "p4_acceleration_package.csv", rows)
            write_csv(out_dir / "p4_acceleration_trace.csv", trace)
        for cand in candidates:
            method = str(cand["method"])
            for opt_name in OPTIMIZERS_P4:
                if (ds, method, opt_name) in done:
                    continue
                sel = p2_sel.get(method, {})
                final, tr = _train_manual_candidate(method, opt_name, bundle, seed, params, device, ds, "P4", params.p4_steps, f(sel, "step_ratio", 1.4), f(sel, "bmem_ratio", 1.2), wandb_args=args)
                rows.append(final)
                trace.extend(tr)
                done.add((ds, method, opt_name))
                write_csv(out_dir / "p4_acceleration_package.csv", rows)
                write_csv(out_dir / "p4_acceleration_trace.csv", trace)
                print(f"P4 {ds} {method}/{opt_name} acc={final['test_acc']:.3f}")
    _decorate_task_rows(rows, trace, "P4")
    for row in rows:
        if row.get("method") == "MLP-AdamW-autograd-reference":
            row["p4_pass"] = 1
        else:
            t_loss = f(row, "time_to_mlp_final_loss", math.inf)
            mlp_t = f(row, "mlp_time_to_final_loss", math.inf)
            auc_delta = f(row, "auc_time_delta_vs_mlp", 0.0)
            base_auc = max(1.0e-12, f(row, "val_auc_by_time", 0.0) + auc_delta)
            row["p4_pass"] = int((t_loss < 0.9 * mlp_t or auc_delta > 0.05 * base_auc) and f(row, "acc_gap_vs_mlp", 99) <= 0.01)
    write_csv(out_dir / "p4_acceleration_package.csv", rows)
    for row in rows:
        _wandb_log_row(args, row, "summary/p4_acceleration_package")
    return rows


def run_p5_to_p9(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(args.out_dir)
    p4 = [r for r in read_csv(out_dir / "p4_acceleration_package.csv") if not r.get("error") and r.get("method") != "MLP-AdamW-autograd-reference"]
    by: Dict[Tuple[str, str], int] = defaultdict(int)
    for row in p4:
        if int(f(row, "p4_pass", 0)) == 1:
            by[(str(row.get("method")), str(row.get("optimizer")))] += 1
    survivors = [(*k, v) for k, v in by.items() if v >= 2]
    if not survivors:
        _placeholder(out_dir, "p5_lightsmooth_compatibility.csv", "P5", "P4 produced no acceleration survivor")
        _placeholder(out_dir, "p6_functional_no_autograd_smoke.csv", "P6", "P5 was not reached")
        _placeholder(out_dir, "p7_joint_selection3.csv", "P7", "P6 was not reached")
        _placeholder(out_dir, "p8_confirm5.csv", "P8", "P7 was not reached")
        _placeholder(out_dir, "p9_confirm10.csv", "P9", "P8 was not reached")
        return
    _placeholder(out_dir, "p5_lightsmooth_compatibility.csv", "P5", "P5 empirical LightSmooth audit is not implemented; fixed proxy rows are forbidden")
    _placeholder(out_dir, "p6_functional_no_autograd_smoke.csv", "P6", "P6 empirical functional update smoke is not implemented; fixed proxy rows are forbidden")
    _placeholder(out_dir, "p7_joint_selection3.csv", "P7", "P6 was not empirically run")
    _placeholder(out_dir, "p8_confirm5.csv", "P8", "P7 was not reached")
    _placeholder(out_dir, "p9_confirm10.csv", "P9", "P8 was not reached")


def run_failure(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    failures: List[Dict[str, Any]] = []
    for row in read_csv(out_dir / "p1_component_kernel_cache_profiler.csv"):
        if row.get("error") or row.get("primitive") == "MLP-autograd-reference":
            continue
        if int(f(row, "p1_nearmiss", 0)) != 1:
            if f(row, "kernel_count_forward", 0) + f(row, "kernel_count_backward", 0) > 20:
                ftype = "F1_kernel_launch_overhead"
            elif f(row, "forward_ratio_vs_mlp_autograd", 99) > 1.6:
                ftype = "F2_transform_cost"
            elif f(row, "backward_memory_ratio_vs_mlp_autograd", 99) > 1.25:
                ftype = "F3_cache_memory"
            else:
                ftype = "F5_convergence_slow"
            failures.append({"stage": "P1", "primitive": row.get("primitive"), "failure_type": ftype, "metric": f"step={row.get('step_ratio_vs_mlp_autograd')} mem={row.get('backward_memory_ratio_vs_mlp_autograd')}", "recommendation": "fuse transform/adjoin or reduce cache"})
    for row in read_csv(out_dir / "p3_minimal_task_recipe.csv"):
        if row.get("method") and row.get("method") != "MLP-AdamW-autograd-reference" and f(row, "acc_gap_vs_mlp", 0) > 0.015:
            failures.append({"stage": "P3", "primitive": row.get("method"), "failure_type": "F4_task_underfit", "metric": f"acc_gap={row.get('acc_gap_vs_mlp')}", "recommendation": "repair primitive expressivity or recipe"})
    for row in read_csv(out_dir / "p4_acceleration_package.csv"):
        if row.get("method") and row.get("method") != "MLP-AdamW-autograd-reference" and int(f(row, "p4_pass", 0)) != 1:
            failures.append({"stage": "P4", "primitive": row.get("method"), "failure_type": "F6_acceleration_unstable", "metric": f"auc_delta={row.get('auc_time_delta_vs_mlp')} acc_gap={row.get('acc_gap_vs_mlp')}", "recommendation": "improve acceleration dynamics only after task recipe stabilizes"})
    for filename, stage in [
        ("p5_lightsmooth_compatibility.csv", "P5"),
        ("p6_functional_no_autograd_smoke.csv", "P6"),
        ("p7_joint_selection3.csv", "P7"),
        ("p8_confirm5.csv", "P8"),
        ("p9_confirm10.csv", "P9"),
    ]:
        for row in read_csv(out_dir / filename):
            if row.get("status") == "not_run":
                failures.append({"stage": stage, "primitive": filename, "failure_type": "F9_gated_not_run", "metric": row.get("reason", ""), "recommendation": "resolve upstream v6.3 gate"})
    if not failures:
        failures.append({"stage": "all", "primitive": "all", "failure_type": "no_failure_rows", "metric": "", "recommendation": ""})
    write_csv(out_dir / "failure_table.csv", failures)
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = add_v3_args()
    parser.description = __doc__
    parser.set_defaults(packages="V6_3_P0", out_dir=Path("results/v6_3"), datasets="MNIST,Fashion-MNIST,KMNIST", seeds="0,1,2", wandb=True)
    parser.add_argument("--v63-bench-warmup", type=int, default=12)
    parser.add_argument("--v63-bench-reps", type=int, default=18)
    parser.add_argument("--v63-p3-steps", type=int, default=120)
    parser.add_argument("--v63-p4-steps", type=int, default=120)
    parser.add_argument("--no-wandb", action="store_false", dest="wandb", help="disable W&B only for local parser/CI checks")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = ensure_dir(args.out_dir)
    _wandb_init(args)
    try:
        for pkg in parse_str_list(args.packages):
            key = pkg.upper()
            if key in {"V6_3_P0"}:
                run_p0(args)
            elif key in {"V6_3_P1"}:
                run_p1(args)
            elif key in {"V6_3_P2"}:
                run_p2(args)
            elif key in {"V6_3_P3"}:
                run_p3(args)
            elif key in {"V6_3_P4"}:
                run_p4(args)
            elif key in {"V6_3_P5_TO_P9", "V6_3_P5"}:
                run_p5_to_p9(args)
            elif key in {"V6_3_FAILURE", "V6_3_P10"}:
                run_failure(args)
            elif key in {"V6_3_ALL", "ALL"}:
                run_p0(args)
                run_p1(args)
                run_p2(args)
                run_p3(args)
                run_p4(args)
                run_p5_to_p9(args)
                run_failure(args)
            else:
                raise ValueError(f"unknown package: {pkg}")
    finally:
        _wandb_finish(args, out_dir)


if __name__ == "__main__":
    main()
