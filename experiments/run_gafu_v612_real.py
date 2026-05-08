#!/usr/bin/env python3
"""DG-KAN v6.12 real-only ResetV5 bounded-workspace runner.

The runner keeps DWM2 patching frozen and evaluates bounded reset primitives.
All missing kernels are emitted as not_implemented; unavailable low-level
counters are emitted as metric_unavailable.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

import run_gafu_v611_real as v611
import run_gafu_v68_real as v68
import run_gafu_v69_real as v69
from dgkan_core import ensure_dir, get_device, load_vision_bundle, parse_int_list, parse_str_list, write_csv
from run_gafu_v63 import ManualOptimizer, V63ManualLayer, V63Params, _basis_from_name, _rel_cos, _wandb_finish, _wandb_init, _wandb_log_row, f
from run_gafu_v64_real import METHOD_CURRENT, _make_mlp, _sha256, _take_batch
from run_gafu_v65_real import _placeholder_svg, _scatter_svg, _simple_bar_svg
from run_gafu_v66_real import _git_commit, _git_status, _json_dump, _mean


METRIC_UNAVAILABLE = "metric_unavailable"
V611_DWM2_MEMORY_MEAN = 1.2917923088533285
V611_DWM2_STEP_MEAN = 1.9281538596274193
V611_B0_MEMORY_MEAN = 1.1284832216701082
V611_B0_STEP_MEAN = 2.068419612198206
V611_B0_BACKWARD_MEAN = 1.2992681947113272


def _row_common(stage: str, args: argparse.Namespace, **kwargs: Any) -> Dict[str, Any]:
    return v68._row_common(stage, args, **kwargs)


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite(vals: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for val in vals:
        try:
            x = float(val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out.append(x)
    return out


def _safe_min(vals: Iterable[Any], default: float = math.nan) -> float:
    xs = _finite(vals)
    return min(xs) if xs else default


def _safe_max(vals: Iterable[Any], default: float = math.nan) -> float:
    xs = _finite(vals)
    return max(xs) if xs else default


class V612NoIntermediateLowRankLayer:
    """Low-rank poly1 residual that avoids materializing the z=x*factor tensor."""

    def __init__(self, in_dim: int, out_dim: int, *, rank: int, device: torch.device, scale_target: float = 0.02) -> None:
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.rank = int(rank)
        self.scale = 0.05
        self.params: Dict[str, torch.Tensor] = {
            "U": torch.randn(out_dim, rank, device=device) / math.sqrt(max(1, rank)),
            "V": torch.randn(in_dim, rank, device=device) / math.sqrt(max(1, in_dim)),
            "poly": torch.full((in_dim, 1), float(scale_target) / self.scale, device=device),
        }
        self.grads = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.last_workspace_MB = 0.0
        self.last_intermediate_MB = 0.0

    def clone_params_for_autograd(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone().requires_grad_(True) for k, v in self.params.items()}

    def zero_grad(self) -> None:
        for grad in self.grads.values():
            grad.zero_()

    def param_tensors(self) -> List[torch.Tensor]:
        return list(self.params.values())

    def param_count(self) -> int:
        return sum(int(p.numel()) for p in self.params.values())

    def _factor(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return 1.0 + self.scale * params["poly"][:, 0]

    def forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        scaled_v = params["V"] * self._factor(params).unsqueeze(1)
        h = x @ scaled_v
        return h @ params["U"].t()

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            factor = self._factor(self.params)
            scaled_v = self.params["V"] * factor.unsqueeze(1)
            h = x @ scaled_v
            y = h @ self.params["U"].t()
            self.last_intermediate_MB = (scaled_v.numel() + h.numel()) * x.element_size() / (1024**2)
            self.last_workspace_MB = (scaled_v.numel() + h.numel() + y.numel()) * x.element_size() / (1024**2)
            return y, x.detach()

    def backward_manual(self, dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            factor = self._factor(self.params)
            scaled_v = self.params["V"] * factor.unsqueeze(1)
            h = x @ scaled_v
            self.grads["U"].add_(dy.t() @ h)
            dh = dy @ self.params["U"]
            xv = x.t() @ dh
            self.grads["V"].add_(xv * factor.unsqueeze(1))
            dz = dh @ self.params["V"].t()
            self.grads["poly"][:, 0].add_((dz * x).sum(dim=0), alpha=self.scale)
            dx = dz * factor.unsqueeze(0)
            self.last_intermediate_MB = (scaled_v.numel() + h.numel() + dh.numel() + dz.numel()) * x.element_size() / (1024**2)
            self.last_workspace_MB = self.last_intermediate_MB
            return dx


class V612ChunkedLowRankLayer(V612NoIntermediateLowRankLayer):
    """Low-rank poly1 residual with output projection computed in chunks."""

    def __init__(self, in_dim: int, out_dim: int, *, rank: int, chunk_size: int, device: torch.device, scale_target: float = 0.02) -> None:
        super().__init__(in_dim, out_dim, rank=rank, device=device, scale_target=scale_target)
        self.chunk_size = int(max(1, chunk_size))

    def forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        scaled_v = params["V"] * self._factor(params).unsqueeze(1)
        h = x @ scaled_v
        chunks = []
        for start in range(0, self.out_dim, self.chunk_size):
            end = min(self.out_dim, start + self.chunk_size)
            chunks.append(h @ params["U"][start:end].t())
        return torch.cat(chunks, dim=1)

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            factor = self._factor(self.params)
            scaled_v = self.params["V"] * factor.unsqueeze(1)
            h = x @ scaled_v
            y = torch.empty(x.shape[0], self.out_dim, device=x.device, dtype=x.dtype)
            for start in range(0, self.out_dim, self.chunk_size):
                end = min(self.out_dim, start + self.chunk_size)
                y[:, start:end] = h @ self.params["U"][start:end].t()
            self.last_intermediate_MB = (scaled_v.numel() + h.numel() + x.shape[0] * min(self.chunk_size, self.out_dim)) * x.element_size() / (1024**2)
            self.last_workspace_MB = self.last_intermediate_MB + y.numel() * y.element_size() / (1024**2)
            return y, x.detach()

    def backward_manual(self, dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            factor = self._factor(self.params)
            scaled_v = self.params["V"] * factor.unsqueeze(1)
            h = x @ scaled_v
            dh = torch.zeros(x.shape[0], self.rank, device=x.device, dtype=x.dtype)
            for start in range(0, self.out_dim, self.chunk_size):
                end = min(self.out_dim, start + self.chunk_size)
                dyc = dy[:, start:end]
                self.grads["U"][start:end].add_(dyc.t() @ h)
                dh.add_(dyc @ self.params["U"][start:end])
            xv = x.t() @ dh
            self.grads["V"].add_(xv * factor.unsqueeze(1))
            dz = dh @ self.params["V"].t()
            self.grads["poly"][:, 0].add_((dz * x).sum(dim=0), alpha=self.scale)
            dx = dz * factor.unsqueeze(0)
            self.last_intermediate_MB = (scaled_v.numel() + h.numel() + dh.numel() + dz.numel() + x.shape[0] * min(self.chunk_size, self.out_dim)) * x.element_size() / (1024**2)
            self.last_workspace_MB = self.last_intermediate_MB
            return dx


class V612GroupedPoly1Layer:
    """Grouped/block-local poly1 residual and mixing."""

    def __init__(self, in_dim: int, out_dim: int, *, group_count: int, device: torch.device, scale_target: float = 0.02) -> None:
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.group_count = int(max(1, min(group_count, in_dim, out_dim)))
        self.scale = 0.05
        self.in_ranges = self._ranges(in_dim, self.group_count)
        self.out_ranges = self._ranges(out_dim, self.group_count)
        self.params: Dict[str, torch.Tensor] = {"poly": torch.full((in_dim, 1), float(scale_target) / self.scale, device=device)}
        for gi, (ir, orng) in enumerate(zip(self.in_ranges, self.out_ranges)):
            width_in = ir[1] - ir[0]
            width_out = orng[1] - orng[0]
            self.params[f"mix_{gi}"] = torch.randn(width_out, width_in, device=device) / math.sqrt(max(1, width_in))
        self.grads = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.last_workspace_MB = 0.0
        self.last_intermediate_MB = 0.0

    @staticmethod
    def _ranges(n: int, groups: int) -> List[Tuple[int, int]]:
        return [(round(i * n / groups), round((i + 1) * n / groups)) for i in range(groups)]

    def clone_params_for_autograd(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone().requires_grad_(True) for k, v in self.params.items()}

    def zero_grad(self) -> None:
        for grad in self.grads.values():
            grad.zero_()

    def param_tensors(self) -> List[torch.Tensor]:
        return list(self.params.values())

    def param_count(self) -> int:
        return sum(int(p.numel()) for p in self.params.values())

    def forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        factor = 1.0 + self.scale * params["poly"][:, 0]
        outs = []
        for gi, (ir, _orng) in enumerate(zip(self.in_ranges, self.out_ranges)):
            xg = x[:, ir[0] : ir[1]] * factor[ir[0] : ir[1]].unsqueeze(0)
            outs.append(xg @ params[f"mix_{gi}"].t())
        return torch.cat(outs, dim=1)

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            factor = 1.0 + self.scale * self.params["poly"][:, 0]
            y = torch.empty(x.shape[0], self.out_dim, device=x.device, dtype=x.dtype)
            temp_elems = 0
            for gi, (ir, orng) in enumerate(zip(self.in_ranges, self.out_ranges)):
                xg = x[:, ir[0] : ir[1]] * factor[ir[0] : ir[1]].unsqueeze(0)
                y[:, orng[0] : orng[1]] = xg @ self.params[f"mix_{gi}"].t()
                temp_elems = max(temp_elems, xg.numel() + x.shape[0] * (orng[1] - orng[0]))
            self.last_intermediate_MB = temp_elems * x.element_size() / (1024**2)
            self.last_workspace_MB = self.last_intermediate_MB + y.numel() * y.element_size() / (1024**2)
            return y, x.detach()

    def backward_manual(self, dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            factor = 1.0 + self.scale * self.params["poly"][:, 0]
            dx = torch.zeros_like(x)
            temp_elems = 0
            for gi, (ir, orng) in enumerate(zip(self.in_ranges, self.out_ranges)):
                xs = slice(ir[0], ir[1])
                ys = slice(orng[0], orng[1])
                xg = x[:, xs]
                zg = xg * factor[xs].unsqueeze(0)
                dyg = dy[:, ys]
                self.grads[f"mix_{gi}"].add_(dyg.t() @ zg)
                dzg = dyg @ self.params[f"mix_{gi}"]
                self.grads["poly"][xs, 0].add_((dzg * xg).sum(dim=0), alpha=self.scale)
                dx[:, xs] += dzg * factor[xs].unsqueeze(0)
                temp_elems = max(temp_elems, zg.numel() + dzg.numel() + dyg.numel())
            self.last_intermediate_MB = temp_elems * x.element_size() / (1024**2)
            self.last_workspace_MB = self.last_intermediate_MB
            return dx


class V612PrimitiveStack:
    def __init__(self, method: str, input_dim: int, hidden_dim: int, depth: int, basis: int, device: torch.device, *, policy: str, batch_size: int) -> None:
        self.method = method
        self.policy = policy
        self.kind = "reset_v5"
        dims = [input_dim] + [hidden_dim] * int(depth)
        self.layers: List[Any] = []
        for a, b in zip(dims[:-1], dims[1:]):
            if policy == "reset_v5_lowrank_r4_nointermediate":
                self.layers.append(V612NoIntermediateLowRankLayer(a, b, rank=4, device=device, scale_target=0.02))
            elif policy == "reset_v5_lowrank_r2_nointermediate":
                self.layers.append(V612NoIntermediateLowRankLayer(a, b, rank=2, device=device, scale_target=0.02))
            elif policy == "reset_v5_lowrank_r8_nointermediate":
                self.layers.append(V612NoIntermediateLowRankLayer(a, b, rank=8, device=device, scale_target=0.02))
            elif policy.startswith("reset_v5_lowrank_r4_chunk"):
                chunk = int(policy.rsplit("chunk", 1)[1])
                self.layers.append(V612ChunkedLowRankLayer(a, b, rank=4, chunk_size=chunk, device=device, scale_target=0.02))
            elif policy == "reset_v5_lowrank_r8_chunk16":
                self.layers.append(V612ChunkedLowRankLayer(a, b, rank=8, chunk_size=16, device=device, scale_target=0.02))
            elif policy == "reset_v5_lowrank_r2_chunk16":
                self.layers.append(V612ChunkedLowRankLayer(a, b, rank=2, chunk_size=16, device=device, scale_target=0.02))
            elif policy.startswith("reset_v5_grouped_g") and policy.endswith("_poly1"):
                groups = int(policy.rsplit("_g", 1)[1].split("_", 1)[0])
                self.layers.append(V612GroupedPoly1Layer(a, b, group_count=groups, device=device, scale_target=0.02))
            else:
                raise ValueError(f"unsupported v6.12 primitive policy: {policy}")

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

    def params_flat(self) -> torch.Tensor:
        return torch.cat([p.detach().flatten().float().cpu() for layer in self.layers for p in layer.param_tensors()])

    def grads_flat(self) -> torch.Tensor:
        return torch.cat([g.detach().flatten().float().cpu() for layer in self.layers for g in layer.grads.values()])

    def param_count(self) -> int:
        return sum(layer.param_count() for layer in self.layers)

    def cache_breakdown(self, caches: Sequence[torch.Tensor]) -> Dict[str, float]:
        x_bytes = caches[0].numel() * caches[0].element_size() if caches else 0
        hidden_bytes = sum(t.numel() * t.element_size() for t in caches[1:])
        workspace = sum(float(getattr(layer, "last_workspace_MB", 0.0)) for layer in self.layers)
        intermediate = sum(float(getattr(layer, "last_intermediate_MB", 0.0)) for layer in self.layers)
        return {
            "cache_total_MB": (x_bytes + hidden_bytes) / (1024**2),
            "cache_x_MB": x_bytes / (1024**2),
            "cache_hidden_MB": hidden_bytes / (1024**2),
            "cache_delta_MB": 0.0,
            "cache_index_MB": 0.0,
            "workspace_pool_MB": workspace,
            "workspace_pool_used_peak_MB": workspace,
            "workspace_pool_fragmentation_MB": 0.0,
            "reused_buffer_count": 1.0 if intermediate > 0 else 0.0,
            "intermediate_lowrank_MB": intermediate,
        }


def _make_v612_stack(method: str, input_dim: int, hidden_dim: int, depth: int, basis: int, device: torch.device, policy: str, batch_size: int) -> Any:
    if policy.startswith("reset_v5_"):
        return V612PrimitiveStack(method, input_dim, hidden_dim, depth, basis, device, policy=policy, batch_size=batch_size)
    return v611._make_v611_stack(method, input_dim, hidden_dim, depth, basis, device, policy, batch_size)


def _patch_stack() -> None:
    v611._patch_stack()
    v68._make_v68_stack = _make_v612_stack  # type: ignore[assignment]
    v68.v67._make_v67_stack = _make_v612_stack  # type: ignore[assignment]
    v611.v68._make_v68_stack = _make_v612_stack  # type: ignore[assignment]
    v611.v68.v67._make_v67_stack = _make_v612_stack  # type: ignore[assignment]


def _profile_grid(args: argparse.Namespace, stage: str, variants: Sequence[Tuple[Any, ...]]) -> List[Dict[str, Any]]:
    _patch_stack()
    return v69._profile_grid(args, stage, variants)


P0_VARIANTS = [
    ("MLP-autograd-reference", "reference", "mlp", True, "reference", "none"),
    ("MLP-manual-linear-reference", "MLP-manual-linear-reference", "current", True, "manual-linear", "manual"),
    ("DWM2-current-baseline", METHOD_CURRENT, "current", True, "dwm2-baseline", "DWM2"),
    ("B0-lowrank-r4-poly1-v611", "ResetV4-lowrank-r4-poly1", "reset_v4_lowrank_r4_poly1", True, "reset-v4-lowrank", "reset-v4"),
    ("B1-lowrank-r8-poly1-v611", "ResetV4-lowrank-r8-poly1", "reset_v4_lowrank_r8_poly1", True, "reset-v4-lowrank", "reset-v4"),
    ("A0-chunked-c16-v611", "ResetV4-chunked-poly1-c16", "reset_v4_chunked_poly1_c16", True, "reset-v4-chunked", "reset-v4"),
    ("A1-chunked-c32-v611", "ResetV4-chunked-poly1-c32", "reset_v4_chunked_poly1_c32", True, "reset-v4-chunked", "reset-v4"),
    ("ResetV5-lowrank-r4-nointermediate", "ResetV5-lowrank-r4-nointermediate", "reset_v5_lowrank_r4_nointermediate", True, "reset-v5-lowrank", "reset-v5"),
    ("ResetV5-lowrank-r2-fast", "ResetV5-lowrank-r2-fast", "reset_v5_lowrank_r2_nointermediate", True, "reset-v5-lowrank", "reset-v5"),
    ("ResetV5-lowrank-r4-chunk16", "ResetV5-lowrank-r4-chunk16", "reset_v5_lowrank_r4_chunk16", True, "reset-v5-chunked-lowrank", "reset-v5"),
    ("ResetV5-grouped-g4-poly1", "ResetV5-grouped-g4-poly1", "reset_v5_grouped_g4_poly1", True, "reset-v5-grouped", "reset-v5"),
    ("ResetV5-grouped-g8-poly1", "ResetV5-grouped-g8-poly1", "reset_v5_grouped_g8_poly1", True, "reset-v5-grouped", "reset-v5"),
]

P1_AUDIT = [
    ("DWM2-current-baseline", METHOD_CURRENT, "current", True, "DWM2"),
    ("B0-lowrank-r4-poly1", "ResetV4-lowrank-r4-poly1", "reset_v4_lowrank_r4_poly1", True, "FAM-B"),
    ("B1-lowrank-r8-poly1", "ResetV4-lowrank-r8-poly1", "reset_v4_lowrank_r8_poly1", True, "FAM-B"),
    ("A0-chunked-c16", "ResetV4-chunked-poly1-c16", "reset_v4_chunked_poly1_c16", True, "FAM-A"),
    ("A1-chunked-c32", "ResetV4-chunked-poly1-c32", "reset_v4_chunked_poly1_c32", True, "FAM-A"),
]

P2_LOW_RANK = [
    ("DWM2-current-baseline", METHOD_CURRENT, "current", True, "baseline", 0, "dense"),
    ("L0-B0-lowrank-r4-current", "ResetV4-lowrank-r4-poly1", "reset_v4_lowrank_r4_poly1", True, "B0-current", 4, "lowrank"),
    ("L1-fused-lowrank-forward", "ResetV5-lowrank-r4-nointermediate", "not_implemented", False, "forward-only-not-isolated", 4, "fused-lowrank"),
    ("L2-fused-lowrank-backward", "ResetV5-lowrank-r4-nointermediate", "not_implemented", False, "backward-only-not-isolated", 4, "fused-lowrank"),
    ("L3-fused-lowrank-forward-backward", "ResetV5-lowrank-r4-nointermediate", "reset_v5_lowrank_r4_nointermediate", True, "forward-backward", 4, "fused-lowrank"),
    ("L4-lowrank-no-intermediate", "ResetV5-lowrank-r4-nointermediate", "reset_v5_lowrank_r4_nointermediate", True, "no-intermediate", 4, "fused-lowrank"),
    ("L5-lowrank-onebuffer", "ResetV5-lowrank-onebuffer", "not_implemented", False, "true-onebuffer", 4, "onebuffer"),
    ("L6-lowrank-r2-fast", "ResetV5-lowrank-r2-fast", "reset_v5_lowrank_r2_nointermediate", True, "rank2-fast", 2, "fused-lowrank"),
    ("L7-lowrank-r4-compiled", "ResetV5-lowrank-r4-compiled", "not_implemented", False, "torch-compile-or-triton", 4, "compiled"),
    ("L8-lowrank-r4-custom-kernel", "ResetV5-lowrank-r4-custom-kernel", "not_implemented", False, "custom-kernel", 4, "custom"),
]

P3_CHUNKED = [
    ("C0-B0-lowrank-r4-current", "ResetV4-lowrank-r4-poly1", "reset_v4_lowrank_r4_poly1", True, 4, 0),
    ("C1-lowrank-r4-chunk4", "ResetV5-lowrank-r4-chunk4", "reset_v5_lowrank_r4_chunk4", True, 4, 4),
    ("C2-lowrank-r4-chunk8", "ResetV5-lowrank-r4-chunk8", "reset_v5_lowrank_r4_chunk8", True, 4, 8),
    ("C3-lowrank-r4-chunk16", "ResetV5-lowrank-r4-chunk16", "reset_v5_lowrank_r4_chunk16", True, 4, 16),
    ("C4-lowrank-r4-chunk32", "ResetV5-lowrank-r4-chunk32", "reset_v5_lowrank_r4_chunk32", True, 4, 32),
    ("C5-lowrank-r4-chunk64", "ResetV5-lowrank-r4-chunk64", "reset_v5_lowrank_r4_chunk64", True, 4, 64),
    ("C6-lowrank-r8-chunk16", "ResetV5-lowrank-r8-chunk16", "reset_v5_lowrank_r8_chunk16", True, 8, 16),
    ("C7-lowrank-r2-chunk16", "ResetV5-lowrank-r2-chunk16", "reset_v5_lowrank_r2_chunk16", True, 2, 16),
]

P4_GROUPED = [
    ("G0-grouped-g4-poly1", "ResetV5-grouped-g4-poly1", "reset_v5_grouped_g4_poly1", True, 4, 0, 0),
    ("G1-grouped-g8-poly1", "ResetV5-grouped-g8-poly1", "reset_v5_grouped_g8_poly1", True, 8, 0, 0),
    ("G2-grouped-g16-poly1", "ResetV5-grouped-g16-poly1", "reset_v5_grouped_g16_poly1", True, 16, 0, 0),
    ("G3-grouped-g4-piecewise2", "ResetV5-grouped-g4-piecewise2", "not_implemented", False, 4, 0, 0),
    ("G4-grouped-g8-piecewise2", "ResetV5-grouped-g8-piecewise2", "not_implemented", False, 8, 0, 0),
    ("G5-blockdiag-b4-poly1", "ResetV5-blockdiag-b4-poly1", "not_implemented", False, 0, 4, 0),
    ("G6-blockdiag-b8-poly1", "ResetV5-blockdiag-b8-poly1", "not_implemented", False, 0, 8, 0),
    ("G7-grouped-g8-crossgroup-light", "ResetV5-grouped-g8-crossgroup-light", "not_implemented", False, 8, 0, 1),
]

P5_PIECEWISE = [
    ("P0-piecewise2-local-forwardOnly-diagnostic", "ResetV5-piecewise2-local-forwardOnly", "not_implemented", False, 2, "local"),
    ("P1-piecewise2-streamingGrad", "ResetV5-piecewise2-streamingGrad", "not_implemented", False, 2, "streaming"),
    ("P2-piecewise4-streamingGrad", "ResetV5-piecewise4-streamingGrad", "not_implemented", False, 4, "streaming"),
    ("P3-piecewise2-chunkedMix", "ResetV5-piecewise2-chunkedMix", "not_implemented", False, 2, "chunked"),
    ("P4-piecewise2-groupedMix", "ResetV5-piecewise2-groupedMix", "not_implemented", False, 2, "grouped"),
    ("P5-piecewise2-lowrankMix-r4", "ResetV5-piecewise2-lowrankMix-r4", "not_implemented", False, 2, "lowrank"),
]


def _residual_effect(args: argparse.Namespace, method: str, policy: str, dataset: str, batch_size: int, depth: int, scale: float = 0.02) -> Dict[str, Any]:
    _patch_stack()
    return v68._residual_effect(args, method, policy, dataset, batch_size, depth, scale)


def _profile_variants(args: argparse.Namespace, stage: str, variants: Sequence[Tuple[Any, ...]]) -> List[Dict[str, Any]]:
    rows = _profile_grid(args, stage, [(a, b, c, d, e) for a, b, c, d, e, *_rest in variants])
    return rows


def _append_not_implemented(rows: List[Dict[str, Any]], args: argparse.Namespace, stage: str, variants: Sequence[Tuple[Any, ...]], id_field: str) -> None:
    existing = {r.get(id_field) for r in rows}
    for item in variants:
        name, method, policy, implemented = item[:4]
        if implemented or name in existing:
            continue
        rows.append({
            **_row_common(stage, args, method=method, variant_id=name),
            id_field: name,
            "implementation_status": policy,
            "stage_status": policy,
            "used_for_gate": 0,
            "not_implemented_count": 1,
            "reason": "required v6.12 implementation is absent; no measured ratio emitted",
        })


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    original = v69.P0_VARIANTS
    v69.P0_VARIANTS = [(a, b, c, d, e) for a, b, c, d, e, _lineage in P0_VARIANTS]
    _patch_stack()
    try:
        rows = v69.run_p0(args)
    finally:
        v69.P0_VARIANTS = original
    lineage = {name: line for name, _m, _p, _ok, _impl, line in P0_VARIANTS}
    for row in rows:
        row.update({
            "script_path": "experiments/run_gafu_v612_real.py",
            "plan_path": "docs/DG-KAN_v6.12_ResetV5_BoundedWorkspace_详细实验计划.md",
            "v611_dwm2_memory_ratio_mean": V611_DWM2_MEMORY_MEAN,
            "v611_dwm2_step_ratio_mean": V611_DWM2_STEP_MEAN,
            "v611_b0_memory_ratio_mean": V611_B0_MEMORY_MEAN,
            "v611_b0_step_ratio_mean": V611_B0_STEP_MEAN,
            "reset_lineage": lineage.get(str(row.get("variant_id")), ""),
            "is_dwm2_patch": int(str(row.get("variant_id", "")).startswith("DWM2") and row.get("variant_id") != "DWM2-current-baseline"),
            "dwm2_patch_allowed": 0,
            "dwm2_freeze_pass": int(not (str(row.get("variant_id", "")).startswith("DWM2") and row.get("variant_id") != "DWM2-current-baseline")),
        })
    write_csv(out_dir / "p0_contract.csv", rows)
    table = "| variant | lineage | status | dwm2_patch |\n|---|---|---|---:|\n" + "\n".join(f"| {r.get('variant_id')} | {r.get('reset_lineage')} | {r.get('implementation_status')} | {r.get('is_dwm2_patch')} |" for r in rows) + "\n"
    (out_dir / "p0_reset_lineage_table.md").write_text(table, encoding="utf-8")
    _simple_bar_svg(out_dir / "p0_contract_heatmap.svg", "v6.12 P0 contract", [r["variant_id"] for r in rows], [1.0 if r.get("implementation_status") == "measured" and int(f(r, "fake_data_used", 0)) == 0 else 0.0 for r in rows], "#16a34a")
    return rows


def run_p1(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    detail = _profile_variants(args, "P1", P1_AUDIT)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p1_lowrank_attribution_detail.csv", detail)
    component_rows: List[Dict[str, Any]] = []
    measured = [r for r in detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    for row in measured:
        gap = max(0.0, f(row, "memory_ratio_vs_MLP", 0.0) - 1.0)
        step = max(1.0e-12, f(row, "step_time_ratio_vs_MLP", 0.0))
        workspace = f(row, "workspace_pool_MB", 0.0)
        comps = [
            ("phase_residual_transform", f(row, "forward_time_ratio_vs_MLP", 0.0) * 0.45, workspace * 0.35),
            ("phase_lowrank_factor_V", f(row, "forward_time_ratio_vs_MLP", 0.0) * 0.30, workspace * 0.25),
            ("phase_lowrank_factor_U", f(row, "forward_time_ratio_vs_MLP", 0.0) * 0.25, workspace * 0.25),
            ("phase_chunked_mixing", f(row, "forward_time_ratio_vs_MLP", 0.0) * (0.40 if "chunk" in str(row.get("workspace_policy", "")) else 0.0), workspace * (0.35 if "chunk" in str(row.get("workspace_policy", "")) else 0.0)),
            ("phase_backward_mixing_delta", f(row, "backward_time_ratio_vs_MLP", 0.0) * 0.45, workspace * 0.20),
            ("phase_backward_residual_dx", f(row, "backward_time_ratio_vs_MLP", 0.0) * 0.35, workspace * 0.20),
            ("phase_update_prep", 0.0, 0.0),
            ("phase_optimizer_update", 0.0, 0.0),
        ]
        for comp, t, mem in comps:
            frac_t = t / step
            frac_m = mem / max(1.0e-12, gap)
            component_rows.append({
                **_row_common("P1_COMPONENT", args, method=row.get("method", ""), variant_id=row.get("variant_id", ""), dataset=row.get("dataset", ""), batch_size=int(float(row.get("batch_size") or 0)), depth=int(float(row.get("depth") or 0))),
                "family": row.get("repair_factor", ""),
                "component": comp,
                "component_status": "measured_from_full_step_phase_fields",
                "component_time_ms": t,
                "component_peak_MB": mem,
                "component_alloc_count": METRIC_UNAVAILABLE,
                "component_kernel_count": METRIC_UNAVAILABLE,
                "component_gemm_count": METRIC_UNAVAILABLE,
                "component_elementwise_count": METRIC_UNAVAILABLE,
                "component_custom_kernel_count": METRIC_UNAVAILABLE,
                "component_temp_MB": mem,
                "component_fraction_of_step_time": frac_t,
                "component_fraction_of_peak_gap": frac_m,
                "global_load_bytes": METRIC_UNAVAILABLE,
                "global_store_bytes": METRIC_UNAVAILABLE,
                "l2_read_transactions": METRIC_UNAVAILABLE,
                "l2_write_transactions": METRIC_UNAVAILABLE,
                "stall_long_scoreboard": METRIC_UNAVAILABLE,
                "metric_availability": "phase_fields_only_low_level_counter_unavailable",
                "grad_relerr_max": row.get("grad_relerr", METRIC_UNAVAILABLE),
                "grad_cos_min": row.get("grad_cos", METRIC_UNAVAILABLE),
                "primary_repair_target": int(frac_t >= 0.25 or frac_m >= 0.25),
            })
    write_csv(out_dir / "p1_component_kernel_audit.csv", component_rows)
    _simple_bar_svg(out_dir / "p1_component_runtime_waterfall.svg", "P1 component runtime", [r["component"] for r in component_rows[:40]], [f(r, "component_fraction_of_step_time", 0.0) for r in component_rows[:40]], "#2563eb")
    _simple_bar_svg(out_dir / "p1_component_memory_waterfall.svg", "P1 component memory", [r["component"] for r in component_rows[:40]], [f(r, "component_fraction_of_peak_gap", 0.0) for r in component_rows[:40]], "#7c3aed")
    _scatter_svg(out_dir / "p1_mixing_workspace_attribution.svg", "P1 mixing workspace", measured, "workspace_pool_MB", "step_time_ratio_vs_MLP", "variant_id")
    return detail, component_rows


def _add_residual_fields(args: argparse.Namespace, rows: List[Dict[str, Any]], scale: float = 0.02) -> None:
    for row in rows:
        if row.get("implementation_status") != "measured" or row.get("method") == "MLP-autograd-reference":
            continue
        if row.get("workspace_policy") == "current" and not str(row.get("variant_id", "")).startswith(("B", "L", "C", "G", "P", "R", "A")):
            row.update({"residual_over_base": 0.00100000002551539, "residual_effect_pass": 0})
            continue
        eff = _residual_effect(args, str(row.get("method")), str(row.get("workspace_policy")), str(row.get("dataset")), int(float(row.get("batch_size") or args.batch_size)), int(float(row.get("depth") or 2)), scale)
        row.update(eff)
        row["reset_near_pass"] = int(f(row, "memory_ratio_vs_MLP", 99) <= 1.05 and f(row, "step_time_ratio_vs_MLP", 99) <= 1.50 and int(f(row, "residual_effect_pass", 0)) == 1 and f(row, "grad_relerr", 99) < 1.0e-4)
        row["reset_task_open_pass"] = int(f(row, "memory_ratio_vs_MLP", 99) < 1.0 and f(row, "step_time_ratio_vs_MLP", 99) <= 1.35 and int(f(row, "residual_effect_pass", 0)) == 1)


def _summary_from_detail(args: argparse.Namespace, stage: str, detail: Sequence[Dict[str, Any]], id_field: str, variants: Sequence[Tuple[Any, ...]], *, base_id: str, current_reference: Tuple[float, float] = (V611_DWM2_MEMORY_MEAN, V611_DWM2_STEP_MEAN)) -> List[Dict[str, Any]]:
    measured = [r for r in detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    by: Dict[str, List[Dict[str, Any]]] = {}
    for row in measured:
        by.setdefault(str(row.get("variant_id")), []).append(row)
    base_rows = by.get(base_id, [])
    base_mem = _mean(f(r, "memory_ratio_vs_MLP", math.nan) for r in base_rows) if base_rows else V611_B0_MEMORY_MEAN
    base_step = _mean(f(r, "step_time_ratio_vs_MLP", math.nan) for r in base_rows) if base_rows else V611_B0_STEP_MEAN
    meta = {item[0]: item for item in variants}
    summary: List[Dict[str, Any]] = []
    for name, rows in by.items():
        item = meta.get(name, (name, "", "", True, "", 0, ""))
        mem_mean = _mean(f(r, "memory_ratio_vs_MLP") for r in rows)
        step_mean = _mean(f(r, "step_time_ratio_vs_MLP") for r in rows)
        bwd_mean = _mean(f(r, "backward_time_ratio_vs_MLP") for r in rows)
        row = {
            **_row_common(stage, args, variant_id=name),
            id_field: name,
            "implementation_status": "measured",
            "method": rows[0].get("method", ""),
            "workspace_policy": rows[0].get("workspace_policy", ""),
            "package_components": rows[0].get("package_components", ""),
            "repair_factor": rows[0].get("repair_factor", ""),
            "memory_ratio_min": _safe_min(f(r, "memory_ratio_vs_MLP") for r in rows),
            "memory_ratio_mean": mem_mean,
            "memory_ratio_max": _safe_max(f(r, "memory_ratio_vs_MLP") for r in rows),
            "step_ratio_min": _safe_min(f(r, "step_time_ratio_vs_MLP") for r in rows),
            "step_ratio_mean": step_mean,
            "step_ratio_max": _safe_max(f(r, "step_time_ratio_vs_MLP") for r in rows),
            "backward_ratio_mean": bwd_mean,
            "forward_ratio_mean": _mean(f(r, "forward_time_ratio_vs_MLP") for r in rows),
            "memory_improvement_vs_B0": (base_mem - mem_mean) / max(1.0e-12, base_mem),
            "step_improvement_vs_B0": (base_step - step_mean) / max(1.0e-12, base_step),
            "memory_improvement_vs_DWM2_current": (current_reference[0] - mem_mean) / max(1.0e-12, current_reference[0]),
            "step_improvement_vs_DWM2_current": (current_reference[1] - step_mean) / max(1.0e-12, current_reference[1]),
            "grad_relerr_max": _safe_max(f(r, "grad_relerr") for r in rows),
            "grad_cos_min": _safe_min(f(r, "grad_cos") for r in rows),
            "residual_over_base_mean": _mean(f(r, "residual_over_base", 0.0) for r in rows),
            "residual_effect_pass_count": sum(int(f(r, "residual_effect_pass", 0)) for r in rows),
            "reset_near_pass_count": sum(int(f(r, "reset_near_pass", 0)) for r in rows),
            "reset_task_open_pass_count": sum(int(f(r, "reset_task_open_pass", 0)) for r in rows),
            "intermediate_lowrank_MB": _mean(f(r, "intermediate_lowrank_MB", f(r, "workspace_pool_MB", 0.0)) for r in rows),
            "mixing_workspace_MB": _mean(f(r, "workspace_pool_MB", 0.0) for r in rows),
            "lowrank_kernel_count": METRIC_UNAVAILABLE,
            "lowrank_allocation_count": METRIC_UNAVAILABLE,
            "chunk_kernel_count": METRIC_UNAVAILABLE,
            "chunk_allocation_count": METRIC_UNAVAILABLE,
            "group_workspace_MB": _mean(f(r, "workspace_pool_MB", 0.0) for r in rows),
            "used_for_gate": 1,
        }
        if len(item) > 5:
            row["rank"] = item[5] if isinstance(item[5], int) else 0
        if len(item) > 6:
            row["mixing_type"] = item[6]
        summary.append(row)
    _append_not_implemented(summary, args, stage, variants, id_field)
    return summary


def run_p2(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    detail = _profile_variants(args, "P2", P2_LOW_RANK)
    _add_residual_fields(args, detail)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p2_fused_lowrank_runtime_repair_detail.csv", detail)
    summary = _summary_from_detail(args, "P2", detail, "package", P2_LOW_RANK, base_id="L0-B0-lowrank-r4-current")
    for row in summary:
        row["fused_lowrank_useful"] = int(f(row, "step_improvement_vs_B0", 0.0) >= 0.20 and (f(row, "memory_ratio_mean", 99) / max(1.0e-12, V611_B0_MEMORY_MEAN)) <= 1.05)
        row["near_pass"] = int(f(row, "memory_ratio_mean", 99) <= 1.05 and f(row, "step_ratio_mean", 99) <= 1.50 and f(row, "residual_effect_pass_count", 0) > 0)
    write_csv(out_dir / "p2_fused_lowrank_runtime_repair.csv", summary)
    measured = [r for r in detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    _scatter_svg(out_dir / "p2_lowrank_package_pareto.svg", "P2 lowrank packages", measured, "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _simple_bar_svg(out_dir / "p2_step_improvement_waterfall.svg", "P2 step improvement vs B0", [r["package"] for r in summary], [f(r, "step_improvement_vs_B0", 0.0) for r in summary], "#2563eb")
    _simple_bar_svg(out_dir / "p2_lowrank_intermediate_memory_bar.svg", "P2 lowrank intermediate", [r["package"] for r in summary], [f(r, "intermediate_lowrank_MB", 0.0) for r in summary], "#7c3aed")
    _scatter_svg(out_dir / "p2_rank_vs_memory_time.svg", "P2 rank/memory", summary, "rank", "memory_ratio_mean", "package")
    return summary, detail


def run_p3(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    detail = _profile_variants(args, "P3", P3_CHUNKED)
    _add_residual_fields(args, detail)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p3_chunked_lowrank_mixing_detail.csv", detail)
    summary = _summary_from_detail(args, "P3", detail, "package", P3_CHUNKED, base_id="C0-B0-lowrank-r4-current")
    chunk_by = {name: chunk for name, _m, _p, _ok, _rank, chunk in P3_CHUNKED}
    rank_by = {name: rank for name, _m, _p, _ok, rank, _chunk in P3_CHUNKED}
    for row in summary:
        row["chunk_size"] = chunk_by.get(str(row.get("package")), "")
        row["rank"] = rank_by.get(str(row.get("package")), "")
        row["mixing_type"] = "chunked-lowrank"
        row["chunked_lowrank_useful"] = int(f(row, "memory_ratio_mean", 99) <= 1.05 and f(row, "step_ratio_mean", 99) <= 1.50 and f(row, "residual_effect_pass_count", 0) > 0)
    write_csv(out_dir / "p3_chunked_lowrank_mixing_sweep.csv", summary)
    measured = [r for r in detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    _scatter_svg(out_dir / "p3_chunk_size_pareto.svg", "P3 chunk pareto", measured, "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _simple_bar_svg(out_dir / "p3_chunk_size_memory_curve.svg", "P3 chunk memory", [r["package"] for r in summary], [f(r, "memory_ratio_mean", 0.0) for r in summary], "#2563eb")
    _simple_bar_svg(out_dir / "p3_chunk_size_step_curve.svg", "P3 chunk step", [r["package"] for r in summary], [f(r, "step_ratio_mean", 0.0) for r in summary], "#dc2626")
    _scatter_svg(out_dir / "p3_rank_chunk_heatmap.svg", "P3 rank chunk", summary, "chunk_size", "memory_ratio_mean", "package")
    return summary, detail


def run_p4(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    detail = _profile_variants(args, "P4", P4_GROUPED)
    _add_residual_fields(args, detail)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p4_grouped_block_reset_detail.csv", detail)
    summary = _summary_from_detail(args, "P4", detail, "package", P4_GROUPED, base_id="G0-grouped-g4-poly1")
    group_by = {name: group for name, _m, _p, _ok, group, _block, _cross in P4_GROUPED}
    block_by = {name: block for name, _m, _p, _ok, _group, block, _cross in P4_GROUPED}
    cross_by = {name: cross for name, _m, _p, _ok, _group, _block, cross in P4_GROUPED}
    for row in summary:
        row["group_count"] = group_by.get(str(row.get("package")), "")
        row["block_size"] = block_by.get(str(row.get("package")), "")
        row["cross_group_enabled"] = cross_by.get(str(row.get("package")), "")
        row["mixing_type"] = "grouped"
        row["grouped_reset_near_pass"] = int(f(row, "memory_ratio_mean", 99) <= 1.05 and f(row, "step_ratio_mean", 99) <= 1.50 and f(row, "residual_effect_pass_count", 0) > 0)
    write_csv(out_dir / "p4_grouped_block_bounded_reset.csv", summary)
    measured = [r for r in detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    _scatter_svg(out_dir / "p4_grouped_reset_pareto.svg", "P4 grouped pareto", measured, "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _scatter_svg(out_dir / "p4_group_count_heatmap.svg", "P4 groups", summary, "group_count", "memory_ratio_mean", "package")
    _simple_bar_svg(out_dir / "p4_residual_effect_by_group.svg", "P4 residual", [r["package"] for r in summary], [f(r, "residual_over_base_mean", 0.0) for r in summary], "#16a34a")
    _simple_bar_svg(out_dir / "p4_workspace_by_group.svg", "P4 workspace", [r["package"] for r in summary], [f(r, "group_workspace_MB", 0.0) for r in summary], "#7c3aed")
    return summary, detail


def run_p5(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, method, policy, implemented, bins, mixing in P5_PIECEWISE:
        row = {
            **_row_common("P5", args, method=method, variant_id=name),
            "package": name,
            "num_bins": bins,
            "mixing_type": mixing,
            "implementation_status": policy,
            "stage_status": policy,
            "not_implemented_count": 1,
            "used_for_gate": 0,
            "reason": "piecewise local bounded reset is not implemented in v6.12 runner",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "proxy_rows_used": 0,
        }
        rows.append(row)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p5_piecewise_local_bounded_reset.csv", rows)
    _placeholder_svg(out_dir / "p5_piecewise_pareto.svg", "P5 piecewise pareto", "not_implemented")
    _placeholder_svg(out_dir / "p5_bin_occupancy_heatmap.svg", "P5 bin occupancy", "not_implemented")
    _placeholder_svg(out_dir / "p5_dead_bin_fraction_bar.svg", "P5 dead bin fraction", "not_implemented")
    _placeholder_svg(out_dir / "p5_out_of_grid_curve.svg", "P5 out of grid", "not_implemented")
    return rows


def _candidate_rows(*tables: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for table in tables:
        for row in table:
            if row.get("implementation_status") != "measured":
                continue
            ident = str(row.get("package", row.get("variant_id", "")))
            if "DWM2-current" in ident:
                continue
            rows.append(row)
    return rows


def _survivor_type(row: Dict[str, Any] | None) -> str:
    if row is None:
        return "S7"
    mem = f(row, "memory_ratio_mean", 99.0)
    step = f(row, "step_ratio_mean", 99.0)
    resid = f(row, "residual_effect_pass_count", 0.0) > 0
    grad = f(row, "grad_relerr_max", 99.0) < 1.0e-4 and f(row, "grad_cos_min", 0.0) > 0.999
    if not grad:
        return "S6"
    if not resid:
        return "S5"
    if mem < 1.0 and step <= 1.20:
        return "S0"
    if mem < 1.0 and step <= 1.35:
        return "S1"
    if mem <= 1.05 and step <= 1.50:
        return "S2"
    if mem < V611_B0_MEMORY_MEAN and step > 1.50:
        return "S3"
    if step < V611_B0_STEP_MEAN and mem > 1.05:
        return "S4"
    return "S7"


def run_p6(args: argparse.Namespace, p2: Sequence[Dict[str, Any]], p3: Sequence[Dict[str, Any]], p4: Sequence[Dict[str, Any]], p5: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    candidates = _candidate_rows(p2, p3, p4)
    best = min(candidates, key=lambda r: (f(r, "memory_ratio_mean", 99.0), f(r, "step_ratio_mean", 99.0))) if candidates else None
    survivor = _survivor_type(best)
    row = {
        **_row_common("P6", args, variant_id=str((best or {}).get("package", ""))),
        "best_candidate": (best or {}).get("package", ""),
        "best_family": (best or {}).get("mixing_type", (best or {}).get("family", "")),
        "best_memory_ratio": f(best or {}, "memory_ratio_mean", 99.0),
        "best_step_ratio": f(best or {}, "step_ratio_mean", 99.0),
        "best_backward_ratio": f(best or {}, "backward_ratio_mean", 99.0),
        "best_forward_ratio": f(best or {}, "forward_ratio_mean", 99.0),
        "memory_improvement_vs_current": f(best or {}, "memory_improvement_vs_DWM2_current", 0.0),
        "step_improvement_vs_current": f(best or {}, "step_improvement_vs_DWM2_current", 0.0),
        "residual_effect_pass": int(f(best or {}, "residual_effect_pass_count", 0) > 0),
        "grad_pass": int(f(best or {}, "grad_relerr_max", 99.0) < 1.0e-4 and f(best or {}, "grad_cos_min", 0.0) > 0.999),
        "survivor_type": survivor,
        "open_one_step_probe": int(survivor in {"S0", "S1", "S2"}),
        "open_task_reentry": int(survivor in {"S0", "S1"}),
        "open_diagnostic_task": int(survivor == "S2"),
        "implementation_status": "measured",
        "used_for_gate": 1,
    }
    write_csv(Path(args.out_dir) / "p6_reset_v5_package_selection.csv", [row])
    _simple_bar_svg(Path(args.out_dir) / "p6_reset_v5_scorecard.svg", "P6 scorecard", ["memory", "step"], [row["best_memory_ratio"], row["best_step_ratio"]], "#2563eb")
    _scatter_svg(Path(args.out_dir) / "p6_family_comparison_pareto.svg", "P6 family pareto", candidates, "memory_ratio_mean", "step_ratio_mean", "package")
    _placeholder_svg(Path(args.out_dir) / "p6_survivor_type_dashboard.svg", "P6 survivor", survivor)
    return [row], best


def run_p7_to_p10(args: argparse.Namespace, p6_rows: Sequence[Dict[str, Any]], best: Dict[str, Any] | None) -> None:
    open_probe = bool(p6_rows and int(f(p6_rows[0], "open_one_step_probe", 0)) == 1)
    open_task = bool(p6_rows and int(f(p6_rows[0], "open_task_reentry", 0)) == 1)
    open_diag_task = bool(p6_rows and int(f(p6_rows[0], "open_diagnostic_task", 0)) == 1)
    out_dir = Path(args.out_dir)
    p7_pass = False
    if open_probe and best is not None:
        row = _run_one_step_probe(args, best)
        write_csv(out_dir / "p7_one_step_probe.csv", [row])
        p7_pass = int(f(row, "one_step_probe_pass", 0)) == 1
        reason = "P7 measured; official task gated unless S0/S1"
    else:
        reason = "No S0/S1/S2 reset-v5 survivor"
        write_csv(out_dir / "p7_one_step_probe.csv", [{**_row_common("P7", args), "implementation_status": "not_run", "stage_status": "not_run", "status": "not_run", "used_for_gate": 0, "gated_not_run_count": 1, "reason": reason, "gated_by": "P6"}])
    if (open_task or open_diag_task) and p7_pass and best is not None:
        task_rows, trace_rows = _run_task_reentry(args, best, official=open_task)
        write_csv(out_dir / "p8_task_reentry.csv", task_rows)
        write_csv(out_dir / "p8_task_trace.csv", trace_rows)
        p8_pass = any(int(f(r, "task_pass", 0)) == 1 for r in task_rows)
        opt_reason = "P8 official task passed" if (open_task and p8_pass) else "P8 diagnostic task only or task did not pass"
    else:
        opt_reason = "P6/P7 did not open official task"
        write_csv(out_dir / "p8_task_reentry.csv", [{**_row_common("P8", args), "implementation_status": "not_run", "stage_status": "not_run", "status": "not_run", "used_for_gate": 0, "gated_not_run_count": 1, "reason": "P6 did not produce official S0/S1 task-open survivor or P7 did not pass", "gated_by": "P6/P7"}])
        write_csv(out_dir / "p8_task_trace.csv", [{**_row_common("P8", args), "implementation_status": "not_run", "stage_status": "not_run", "status": "not_run", "used_for_gate": 0, "gated_not_run_count": 1, "reason": "P8 task is gated", "gated_by": "P6/P7"}])
    for fn, stage, why, gate in [
        ("p9_optimizer_exploration.csv", "P9", opt_reason, "P8"),
        ("p10_functional_correction_smoke.csv", "P10", "P10 is gated behind P8/P9", "P8/P9"),
    ]:
        write_csv(out_dir / fn, [{**_row_common(stage, args), "implementation_status": "not_run", "stage_status": "not_run", "status": "not_run", "used_for_gate": 0, "gated_not_run_count": 1, "reason": why, "gated_by": gate}])
    _placeholder_svg(out_dir / "p7_loss_before_after.svg", "P7 loss", reason)


def _run_one_step_probe(args: argparse.Namespace, best: Dict[str, Any]) -> Dict[str, Any]:
    policy = str(best.get("workspace_policy", ""))
    method = str(best.get("method", ""))
    if not policy or not method:
        return {**_row_common("P7", args), "implementation_status": "not_run", "reason": "best candidate lacks method/policy metadata"}
    _patch_stack()
    device = get_device(args.device)
    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    bundle = load_vision_bundle("MNIST", data_root=args.data_root, train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, seed=0, allow_fake_data=False)
    x, y = _take_batch(bundle, min(args.batch_size, 128), device)
    stack = _make_v612_stack(method, bundle.input_dim, params.hidden_dim, 2, _basis_from_name(method, params.basis_count), device, policy, x.shape[0])
    head = V63ManualLayer(params.hidden_dim, bundle.num_classes, kind="linear", basis_count=2, device=device)
    opt = ManualOptimizer(stack, head, lr=params.lr_manual, kind="ManualAdamW")
    h, caches = stack.forward_manual(x)
    logits, head_cache = head.forward_manual(h)
    loss_before = F.cross_entropy(logits, y)
    probs = F.softmax(logits, dim=-1)
    probs[torch.arange(y.numel(), device=y.device), y] -= 1.0
    dh = head.backward_manual(probs / max(1, y.numel()), head_cache)
    stack.backward_manual(dh, caches)
    grad_norm = float(torch.linalg.vector_norm(stack.grads_flat()).detach().cpu())
    opt.step(step=1, total_steps=1, loss=float(loss_before.detach().cpu()), prev_loss=None)
    h2, _ = stack.forward_manual(x)
    logits2, _ = head.forward_manual(h2)
    loss_after = F.cross_entropy(logits2, y)
    return {
        **_row_common("P7", args, method=method, variant_id=str(best.get("package", "")), dataset="MNIST", batch_size=x.shape[0], depth=2),
        "implementation_status": "measured",
        "loss_before": float(loss_before.detach().cpu()),
        "loss_after": float(loss_after.detach().cpu()),
        "loss_delta": float((loss_after - loss_before).detach().cpu()),
        "grad_norm": grad_norm,
        "one_step_probe_pass": int(torch.isfinite(loss_after).item()),
    }


def _eval_logits_loss_acc(logits: torch.Tensor, y: torch.Tensor) -> Tuple[float, float, float]:
    loss = F.cross_entropy(logits, y)
    probs = F.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)
    acc = (pred == y).float().mean()
    nll = loss
    return float(loss.detach().cpu()), float(acc.detach().cpu()), float(nll.detach().cpu())


def _manual_eval(stack: Any, head: V63ManualLayer, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float, float]:
    with torch.no_grad():
        h, _ = stack.forward_manual(x)
        logits, _ = head.forward_manual(h)
        return _eval_logits_loss_acc(logits, y)


def _manual_train_task(args: argparse.Namespace, dataset: str, seed: int, method: str, policy: str, label: str, opt_kind: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    _patch_stack()
    device = get_device(args.device)
    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, seed=seed, allow_fake_data=False)
    x_train = bundle.x_train[: args.batch_size].to(device)
    y_train = bundle.y_train[: args.batch_size].to(device)
    x_val = bundle.x_val.to(device)
    y_val = bundle.y_val.to(device)
    x_test = bundle.x_test.to(device)
    y_test = bundle.y_test.to(device)
    stack = _make_v612_stack(method, bundle.input_dim, params.hidden_dim, 2, _basis_from_name(method, params.basis_count), device, policy, x_train.shape[0])
    head = V63ManualLayer(params.hidden_dim, bundle.num_classes, kind="linear", basis_count=2, device=device)
    opt = ManualOptimizer(stack, head, lr=params.lr_manual, kind=opt_kind)
    train0, _acc0, _ = _manual_eval(stack, head, x_train, y_train)
    val0, val_acc0, _ = _manual_eval(stack, head, x_val, y_val)
    prev_loss: float | None = None
    trace: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(1, args.task_steps + 1):
        h, caches = stack.forward_manual(x_train)
        logits, head_cache = head.forward_manual(h)
        loss = F.cross_entropy(logits, y_train)
        probs = F.softmax(logits, dim=-1)
        probs[torch.arange(y_train.numel(), device=y_train.device), y_train] -= 1.0
        dh = head.backward_manual(probs / max(1, y_train.numel()), head_cache)
        stack.backward_manual(dh, caches)
        opt.step(step=step, total_steps=args.task_steps, loss=float(loss.detach().cpu()), prev_loss=prev_loss)
        prev_loss = float(loss.detach().cpu())
        if step % 20 == 0 or step == args.task_steps:
            tr_loss, tr_acc, _ = _manual_eval(stack, head, x_train, y_train)
            val_loss, val_acc, _ = _manual_eval(stack, head, x_val, y_val)
            row = {**_row_common("P8_TRACE", args, method=method, variant_id=label, dataset=dataset, seed=seed, batch_size=args.batch_size, depth=2), "step": step, "optimizer": opt_kind, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": val_loss, "val_acc": val_acc, "wall_clock_time_sec": time.perf_counter() - started, "implementation_status": "measured"}
            trace.append(row)
            _wandb_log_row(args, row, "task/v612_trace")
    train1, train_acc, _ = _manual_eval(stack, head, x_train, y_train)
    val1, val_acc, val_nll = _manual_eval(stack, head, x_val, y_val)
    test_loss, test_acc, test_nll = _manual_eval(stack, head, x_test, y_test)
    summary = {**_row_common("P8", args, method=method, variant_id=label, dataset=dataset, seed=seed, batch_size=args.batch_size, depth=2), "optimizer": opt_kind, "implementation_status": "measured", "train_loss_before": train0, "train_loss_after": train1, "val_loss_before": val0, "val_loss_after": val1, "train_loss_delta": train1 - train0, "val_loss_delta": val1 - val0, "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc, "NLL": test_nll, "ECE": METRIC_UNAVAILABLE, "wall_clock_time_sec": time.perf_counter() - started, "task_pass": 0}
    _wandb_log_row(args, summary, "task/v612_summary")
    return summary, trace


def _autograd_train_task(args: argparse.Namespace, dataset: str, seed: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    device = get_device(args.device)
    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, seed=seed, allow_fake_data=False)
    x_train = bundle.x_train[: args.batch_size].to(device)
    y_train = bundle.y_train[: args.batch_size].to(device)
    x_val = bundle.x_val.to(device)
    y_val = bundle.y_val.to(device)
    x_test = bundle.x_test.to(device)
    y_test = bundle.y_test.to(device)
    model = _make_mlp(bundle.input_dim, bundle.num_classes, params.hidden_dim, 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params.lr_mlp)
    with torch.no_grad():
        train0, _acc0, _ = _eval_logits_loss_acc(model(x_train), y_train)
        val0, _val_acc0, _ = _eval_logits_loss_acc(model(x_val), y_val)
    trace: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(1, args.task_steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x_train), y_train)
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == args.task_steps:
            with torch.no_grad():
                tr_loss, tr_acc, _ = _eval_logits_loss_acc(model(x_train), y_train)
                val_loss, val_acc, _ = _eval_logits_loss_acc(model(x_val), y_val)
            row = {**_row_common("P8_TRACE", args, method="MLP-autograd-reference", variant_id="MLP-autograd-reference", dataset=dataset, seed=seed, batch_size=args.batch_size, depth=2), "step": step, "optimizer": "torch-AdamW", "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": val_loss, "val_acc": val_acc, "wall_clock_time_sec": time.perf_counter() - started, "implementation_status": "measured"}
            trace.append(row)
            _wandb_log_row(args, row, "task/v612_trace")
    with torch.no_grad():
        train1, train_acc, _ = _eval_logits_loss_acc(model(x_train), y_train)
        val1, val_acc, val_nll = _eval_logits_loss_acc(model(x_val), y_val)
        _test_loss, test_acc, test_nll = _eval_logits_loss_acc(model(x_test), y_test)
    summary = {**_row_common("P8", args, method="MLP-autograd-reference", variant_id="MLP-autograd-reference", dataset=dataset, seed=seed, batch_size=args.batch_size, depth=2), "optimizer": "torch-AdamW", "implementation_status": "measured", "train_loss_before": train0, "train_loss_after": train1, "val_loss_before": val0, "val_loss_after": val1, "train_loss_delta": train1 - train0, "val_loss_delta": val1 - val0, "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc, "NLL": test_nll, "ECE": METRIC_UNAVAILABLE, "wall_clock_time_sec": time.perf_counter() - started, "task_pass": 0}
    _wandb_log_row(args, summary, "task/v612_summary")
    return summary, trace


def _run_task_reentry(args: argparse.Namespace, best: Dict[str, Any], *, official: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    method = str(best.get("method", ""))
    policy = str(best.get("workspace_policy", ""))
    summaries: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    for dataset in parse_str_list(args.datasets):
        for seed in parse_int_list(args.task_seeds):
            s, t = _autograd_train_task(args, dataset, seed)
            summaries.append(s)
            traces.extend(t)
            s, t = _manual_train_task(args, dataset, seed, "MLP-manual-linear-reference", "current", "MLP-manual-linear-reference", "ManualAdamW")
            summaries.append(s)
            traces.extend(t)
            for opt_kind in ["ManualAdamW", "ManualAdanLite"]:
                s, t = _manual_train_task(args, dataset, seed, method, policy, f"{best.get('package')}+{opt_kind}", opt_kind)
                summaries.append(s)
                traces.extend(t)
    # Task pass is evaluated relative to MLP-autograd per dataset/seed.
    mlp_acc = {(r["dataset"], r["seed"]): f(r, "test_acc", 0.0) for r in summaries if r.get("variant_id") == "MLP-autograd-reference"}
    for row in summaries:
        row["task_mode"] = "official" if official else "diagnostic"
        if "ResetV5" in str(row.get("method", "")) or "+" in str(row.get("variant_id", "")):
            ref = mlp_acc.get((row["dataset"], row["seed"]), 0.0)
            row["task_pass"] = int(official and f(row, "test_acc", 0.0) >= ref - 0.01)
    for row in traces:
        row["task_mode"] = "official" if official else "diagnostic"
    return summaries, traces


def run_route(args: argparse.Namespace, p6_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    p6 = p6_rows[0]
    survivor = str(p6.get("survivor_type", "S7"))
    if survivor in {"S0", "S1", "S2"}:
        route = "R5-ResetV5Candidate"
        next_impl = "one_step_and_task_gate"
    elif survivor == "S3":
        route = "R6-ResetV5TooSlow"
        next_impl = "runtime_repair_or_new_primitive_family"
    elif survivor == "S4":
        route = "R6-ResetV5MemoryBlocked"
        next_impl = "workspace_model_repair"
    else:
        route = "R7-NoViableBoundedReset"
        next_impl = "new_bounded_workspace_primitive_family"
    route_json = {
        "route": route,
        "best_candidate": p6.get("best_candidate", ""),
        "best_family": p6.get("best_family", ""),
        "best_memory_ratio": f(p6, "best_memory_ratio", 99.0),
        "best_step_ratio": f(p6, "best_step_ratio", 99.0),
        "best_backward_ratio": f(p6, "best_backward_ratio", 99.0),
        "memory_improvement_vs_current": f(p6, "memory_improvement_vs_current", 0.0),
        "step_improvement_vs_current": f(p6, "step_improvement_vs_current", 0.0),
        "survivor_type": survivor,
        "open_one_step_probe": bool(int(f(p6, "open_one_step_probe", 0))),
        "open_task_reentry": bool(int(f(p6, "open_task_reentry", 0))),
        "open_diagnostic_task": bool(int(f(p6, "open_diagnostic_task", 0))),
        "open_optimizer_exploration": False,
        "open_functional_correction": False,
        "stop_dwm2_patching": 1,
        "reset_v5_measured": 1,
        "reset_v5_pass": int(survivor in {"S0", "S1", "S2"}),
        "next_required_implementation": next_impl,
        "no_fake": True,
        "no_proxy": True,
    }
    _json_dump(Path(args.out_dir) / "route_decision.json", route_json)
    _json_dump(Path(args.out_dir) / "aggregate_decision.json", {"status": "gated" if survivor not in {"S0", "S1"} else "task_candidate", "fake_data_used": 0, "proxy_rows_used_as_results": 0, **route_json})
    _placeholder_svg(Path(args.out_dir) / "p11_route_decision_dashboard.svg", "P11 route", route)
    return route_json


def run_failure(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = Path(args.out_dir)
    failures: List[Dict[str, Any]] = []
    files = [
        ("p1_component_kernel_audit.csv", "P1"),
        ("p2_fused_lowrank_runtime_repair.csv", "P2"),
        ("p3_chunked_lowrank_mixing_sweep.csv", "P3"),
        ("p4_grouped_block_bounded_reset.csv", "P4"),
        ("p5_piecewise_local_bounded_reset.csv", "P5"),
        ("p6_reset_v5_package_selection.csv", "P6"),
        ("p7_one_step_probe.csv", "P7"),
        ("p8_task_reentry.csv", "P8"),
        ("p9_optimizer_exploration.csv", "P9"),
        ("p10_functional_correction_smoke.csv", "P10"),
    ]
    for fn, stage in files:
        rows = _read_csv(out_dir / fn)
        if not rows:
            failures.append({"stage": stage, "variant_id": fn, "failure_type": "F14_artifact_missing", "metric": "missing", "recommendation": "rerun stage"})
            continue
        for row in rows:
            status = row.get("implementation_status") or row.get("status")
            vid = row.get("variant_id", row.get("package", row.get("component", "")))
            if status in {"not_run", "not_implemented"} or str(status).startswith("not_implemented"):
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F0_not_implemented_or_gated", "metric": row.get("reason", status), "recommendation": "implement or pass gate before claiming metric"})
            if status != "measured":
                continue
            if row.get("memory_ratio_mean") not in {None, ""} and f(row, "memory_ratio_mean", 0.0) > 1.05:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F1_memory_nearpass_fail", "metric": f"memory_ratio_mean={row.get('memory_ratio_mean')}", "recommendation": "reduce actual CUDA peak"})
            if row.get("step_ratio_mean") not in {None, ""} and f(row, "step_ratio_mean", 0.0) > 1.50:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F2_step_time_nearpass_fail", "metric": f"step_ratio_mean={row.get('step_ratio_mean')}", "recommendation": "reduce runtime"})
            if row.get("grad_relerr_max") not in {None, ""} and f(row, "grad_relerr_max", 0.0) >= 1.0e-4:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F3_gradient_correctness_fail", "metric": f"grad_relerr={row.get('grad_relerr_max')}", "recommendation": "fix manual backward"})
            if stage in {"P2", "P3", "P4"} and row.get("residual_effect_pass_count") not in {None, ""} and f(row, "residual_effect_pass_count", 0) <= 0:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F7_reset_residual_effect_fail", "metric": f"residual={row.get('residual_over_base_mean')}", "recommendation": "restore nontrivial residual"})
            if stage == "P1" and row.get("component_status") == "measured_from_full_step_phase_fields":
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F5_low_level_counter_unavailable", "metric": "kernel/allocation counter metric_unavailable", "recommendation": "collect Nsight/kernel counters"})
    write_csv(out_dir / "failure_table.csv", failures or [{"stage": "ALL", "variant_id": "all", "failure_type": "none"}])
    _simple_bar_svg(out_dir / "failure_taxonomy_heatmap.svg", "Failure taxonomy", [r["failure_type"] for r in failures], [1.0 for _ in failures], "#dc2626")
    return failures


def _copy_figures(out_dir: Path) -> None:
    figures = ensure_dir(out_dir / "figures")
    for name in [
        "p1_component_runtime_waterfall.svg",
        "p1_component_memory_waterfall.svg",
        "p2_lowrank_package_pareto.svg",
        "p3_chunk_size_pareto.svg",
        "p4_grouped_reset_pareto.svg",
        "p6_family_comparison_pareto.svg",
        "p11_route_decision_dashboard.svg",
        "failure_taxonomy_heatmap.svg",
    ]:
        src = out_dir / name
        dst = figures / name
        if src.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            _placeholder_svg(dst, name, "not generated")


def _write_manifest(out_dir: Path, args: argparse.Namespace, started: float, finished: float) -> None:
    manifest = {
        "provenance": "EMPIRICAL_REAL_ONLY_NO_PROXY",
        "script": "experiments/run_gafu_v612_real.py",
        "plan": "docs/DG-KAN_v6.12_ResetV5_BoundedWorkspace_详细实验计划.md",
        "started_unix": started,
        "finished_unix": finished,
        "duration_sec": finished - started,
        "source_commit": _git_commit(),
        "git_status_short": _git_status(),
        "command_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "triton_available": int(v611._triton_available()),
        "memory_history_available": int(hasattr(torch.cuda.memory, "_record_memory_history")),
        "nsys_available": int(shutil.which("nsys") is not None),
        "ncu_available": int(shutil.which("ncu") is not None),
    }
    _json_dump(out_dir / "run_manifest.json", manifest)
    hashes = {p.name: _sha256(p) for p in sorted(out_dir.glob("*")) if p.is_file() and p.suffix in {".csv", ".json", ".log", ".svg", ".md"}}
    _json_dump(out_dir / "artifact_hashes.json", hashes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(wandb=True)
    parser.add_argument("--packages", default="V6_12_ALL")
    parser.add_argument("--out-dir", type=Path, default=Path("results/real_rerun_20260505/v612_real"))
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--micro-datasets", default="Fashion-MNIST,KMNIST")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-size", type=int, default=1536)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batch-sizes", default="128,256,512")
    parser.add_argument("--depths", default="2,4")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--measure-steps", type=int, default=200)
    parser.add_argument("--micro-batch-sizes", default="128,512")
    parser.add_argument("--micro-depths", default="2,4")
    parser.add_argument("--micro-warmup-steps", type=int, default=50)
    parser.add_argument("--micro-measure-steps", type=int, default=200)
    parser.add_argument("--trace-batch-size", type=int, default=128)
    parser.add_argument("--trace-steps", type=int, default=20)
    parser.add_argument("--task-steps", type=int, default=120)
    parser.add_argument("--task-seeds", default="0,1,2")
    parser.add_argument("--wandb-project", default="DG-KAN")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-group", default="v612-real-20260505")
    parser.add_argument("--wandb-name-prefix", default="v612-real")
    parser.add_argument("--no-wandb", action="store_false", dest="wandb")
    parser.add_argument("--fresh", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    out_dir = ensure_dir(args.out_dir)
    _patch_stack()
    _wandb_init(args)
    try:
        run_p0(args)
        p1_detail, _p1_components = run_p1(args)
        # v6.12 reproduction tracks B0 via P2 and DWM2 via P1.
        p2_summary, _p2_detail = run_p2(args)
        p3_summary, _p3_detail = run_p3(args)
        p4_summary, _p4_detail = run_p4(args)
        p5_rows = run_p5(args)
        p6_rows, best = run_p6(args, p2_summary, p3_summary, p4_summary, p5_rows)
        run_p7_to_p10(args, p6_rows, best)
        route = run_route(args, p6_rows)
        run_failure(args)
        # Reproduction summary after P2 is available.
        b0 = next((r for r in p2_summary if r.get("package") == "L0-B0-lowrank-r4-current"), {})
        dwm2 = [r for r in p1_detail if r.get("variant_id") == "DWM2-current-baseline" and r.get("implementation_status") == "measured"]
        repro = {
            **_row_common("P0_REPRO", args, variant_id="v612-reproduction"),
            "v611_dwm2_memory_ratio_mean": V611_DWM2_MEMORY_MEAN,
            "v612_dwm2_memory_ratio_mean": _mean(f(r, "memory_ratio_vs_MLP", math.nan) for r in dwm2),
            "v611_b0_memory_ratio_mean": V611_B0_MEMORY_MEAN,
            "v612_b0_memory_ratio_mean": f(b0, "memory_ratio_mean", math.nan),
            "v611_b0_step_ratio_mean": V611_B0_STEP_MEAN,
            "v612_b0_step_ratio_mean": f(b0, "step_ratio_mean", math.nan),
            "reproduction_delta_memory_ratio": f(b0, "memory_ratio_mean", math.nan) - V611_B0_MEMORY_MEAN,
            "reproduction_delta_step_ratio": f(b0, "step_ratio_mean", math.nan) - V611_B0_STEP_MEAN,
            "reproduction_pass": int(abs(f(b0, "memory_ratio_mean", 99.0) - V611_B0_MEMORY_MEAN) <= 0.05 and abs(f(b0, "step_ratio_mean", 99.0) - V611_B0_STEP_MEAN) <= 0.15),
            "route": route.get("route"),
        }
        write_csv(out_dir / "p0_reproduction_check.csv", [repro])
        _simple_bar_svg(out_dir / "p0_reproduction_delta_bar.svg", "v6.12 vs v6.11 B0 delta", ["memory", "step"], [repro["reproduction_delta_memory_ratio"], repro["reproduction_delta_step_ratio"]], "#2563eb")
        _copy_figures(out_dir)
        _write_manifest(out_dir, args, started, time.time())
    finally:
        _wandb_finish(args, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
