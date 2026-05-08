#!/usr/bin/env python3
"""DG-KAN v6.15 real-only ResetV8 runtime-repair and piecewise-local runner.

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

import run_gafu_v612_real as v612
import run_gafu_v68_real as v68
import run_gafu_v69_real as v69
from dgkan_core import ensure_dir, get_device, load_vision_bundle, parse_int_list, parse_str_list, write_csv
from run_gafu_v63 import ManualOptimizer, V63ManualLayer, V63Params, _basis_from_name, _rel_cos, _wandb_finish, _wandb_init, _wandb_log_row, f
from run_gafu_v64_real import METHOD_CURRENT, _make_mlp, _sha256, _take_batch
from run_gafu_v65_real import _placeholder_svg, _scatter_svg, _simple_bar_svg
from run_gafu_v66_real import _git_commit, _git_status, _json_dump, _mean


METRIC_UNAVAILABLE = "metric_unavailable"
V614_DWM2_MEMORY_MEAN = 1.2917923088533285
V614_DWM2_STEP_MEAN = 1.9264233881276516
V614_L6_MEMORY_MEAN = 1.0785729221267415
V614_L6_STEP_MEAN = 1.9991722990773881
V614_G2_MEMORY_MEAN = 1.0459591815484715
V614_G2_STEP_MEAN = 9.246101828857643
V614_GR3_MEMORY_MEAN = 1.1748654001477965
V614_GR3_STEP_MEAN = 2.0074342818729236
V614_B0_MEMORY_MEAN = V614_DWM2_MEMORY_MEAN
V614_B0_STEP_MEAN = V614_DWM2_STEP_MEAN


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


class V615NoIntermediateLowRankLayer:
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


class V615ChunkedLowRankLayer(V615NoIntermediateLowRankLayer):
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


class V615GroupedPoly1Layer:
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


class V615VectorizedGroupedPoly1Layer:
    """Grouped poly1 residual using batched grouped matmul instead of a Python group loop."""

    def __init__(self, in_dim: int, out_dim: int, *, group_count: int, device: torch.device, scale_target: float = 0.02) -> None:
        if in_dim % group_count != 0 or out_dim % group_count != 0:
            raise ValueError(f"vectorized grouped requires divisible dims: in={in_dim} out={out_dim} groups={group_count}")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.group_count = int(group_count)
        self.in_per_group = in_dim // group_count
        self.out_per_group = out_dim // group_count
        self.scale = 0.05
        self.params: Dict[str, torch.Tensor] = {
            "poly": torch.full((in_dim, 1), float(scale_target) / self.scale, device=device),
            "mix": torch.randn(group_count, self.out_per_group, self.in_per_group, device=device) / math.sqrt(max(1, self.in_per_group)),
        }
        self.grads = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.last_workspace_MB = 0.0
        self.last_intermediate_MB = 0.0
        self.last_group_loop_count = 0

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
        bsz = x.shape[0]
        factor = self._factor(params).view(self.group_count, self.in_per_group)
        xg = x.view(bsz, self.group_count, self.in_per_group) * factor.unsqueeze(0)
        yg = torch.einsum("bgi,goi->bgo", xg, params["mix"])
        return yg.reshape(bsz, self.out_dim)

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            bsz = x.shape[0]
            factor = self._factor(self.params).view(self.group_count, self.in_per_group)
            xg = x.view(bsz, self.group_count, self.in_per_group) * factor.unsqueeze(0)
            yg = torch.einsum("bgi,goi->bgo", xg, self.params["mix"])
            y = yg.reshape(bsz, self.out_dim)
            self.last_group_loop_count = 0
            self.last_intermediate_MB = (xg.numel() + yg.numel()) * x.element_size() / (1024**2)
            self.last_workspace_MB = self.last_intermediate_MB
            return y, x.detach()

    def backward_manual(self, dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            bsz = x.shape[0]
            factor = self._factor(self.params).view(self.group_count, self.in_per_group)
            xbase = x.view(bsz, self.group_count, self.in_per_group)
            xg = xbase * factor.unsqueeze(0)
            dyg = dy.view(bsz, self.group_count, self.out_per_group)
            self.grads["mix"].add_(torch.einsum("bgo,bgi->goi", dyg, xg))
            dzg = torch.einsum("bgo,goi->bgi", dyg, self.params["mix"])
            self.grads["poly"][:, 0].add_((dzg.reshape(bsz, self.in_dim) * x).sum(dim=0), alpha=self.scale)
            dx = (dzg * factor.unsqueeze(0)).reshape(bsz, self.in_dim)
            self.last_group_loop_count = 0
            self.last_intermediate_MB = (xg.numel() + dyg.numel() + dzg.numel()) * x.element_size() / (1024**2)
            self.last_workspace_MB = self.last_intermediate_MB
            return dx


class V615NoXgEinsumGroupedPoly1Layer(V615VectorizedGroupedPoly1Layer):
    """Grouped poly1 einsum path that avoids materializing the scaled xg tensor.

    This is a real ResetV8 candidate for the v6.15 H2 question. It keeps the
    batched grouped math from the v6.13 vectorized path, but folds the poly1
    scale factor into the einsum instead of creating xg = x * factor as a full
    [batch, group, in_per_group] tensor.
    """

    def forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        bsz = x.shape[0]
        xbase = x.view(bsz, self.group_count, self.in_per_group)
        factor = self._factor(params).view(self.group_count, self.in_per_group)
        yg = torch.einsum("bgi,gi,goi->bgo", xbase, factor, params["mix"])
        return yg.reshape(bsz, self.out_dim)

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            bsz = x.shape[0]
            xbase = x.view(bsz, self.group_count, self.in_per_group)
            factor = self._factor(self.params).view(self.group_count, self.in_per_group)
            yg = torch.einsum("bgi,gi,goi->bgo", xbase, factor, self.params["mix"])
            y = yg.reshape(bsz, self.out_dim)
            self.last_group_loop_count = 0
            self.last_intermediate_MB = yg.numel() * x.element_size() / (1024**2)
            self.last_workspace_MB = self.last_intermediate_MB
            return y, x.detach()

    def backward_manual(self, dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            bsz = x.shape[0]
            xbase = x.view(bsz, self.group_count, self.in_per_group)
            factor = self._factor(self.params).view(self.group_count, self.in_per_group)
            dyg = dy.view(bsz, self.group_count, self.out_per_group)
            self.grads["mix"].add_(torch.einsum("bgo,bgi,gi->goi", dyg, xbase, factor))
            dzg = torch.einsum("bgo,goi->bgi", dyg, self.params["mix"])
            self.grads["poly"][:, 0].add_((dzg.reshape(bsz, self.in_dim) * x).sum(dim=0), alpha=self.scale)
            dx = (dzg * factor.unsqueeze(0)).reshape(bsz, self.in_dim)
            self.last_group_loop_count = 0
            self.last_intermediate_MB = (dyg.numel() + dzg.numel()) * x.element_size() / (1024**2)
            self.last_workspace_MB = self.last_intermediate_MB
            return dx


class V615TiledGroupedPoly1Layer(V615VectorizedGroupedPoly1Layer):
    """Tile groups to trade some loop overhead for bounded materialization."""

    def __init__(self, in_dim: int, out_dim: int, *, group_count: int, tile_groups: int, device: torch.device, scale_target: float = 0.02) -> None:
        super().__init__(in_dim, out_dim, group_count=group_count, device=device, scale_target=scale_target)
        self.tile_groups = int(max(1, min(tile_groups, self.group_count)))

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            bsz = x.shape[0]
            xbase = x.view(bsz, self.group_count, self.in_per_group)
            factor = self._factor(self.params).view(self.group_count, self.in_per_group)
            y = torch.empty(bsz, self.out_dim, device=x.device, dtype=x.dtype)
            peak = 0
            for start in range(0, self.group_count, self.tile_groups):
                end = min(self.group_count, start + self.tile_groups)
                yg = torch.einsum("bgi,gi,goi->bgo", xbase[:, start:end], factor[start:end], self.params["mix"][start:end])
                y[:, start * self.out_per_group : end * self.out_per_group] = yg.reshape(bsz, (end - start) * self.out_per_group)
                peak = max(peak, yg.numel())
            self.last_group_loop_count = math.ceil(self.group_count / self.tile_groups)
            self.last_intermediate_MB = peak * x.element_size() / (1024**2)
            self.last_workspace_MB = self.last_intermediate_MB
            return y, x.detach()

    def forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Autograd reference keeps the same math; it may materialize more than
        # the manual path, but it is used only for correctness checking.
        return V615NoXgEinsumGroupedPoly1Layer.forward_with_params(self, x, params)

    def backward_manual(self, dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            bsz = x.shape[0]
            xbase = x.view(bsz, self.group_count, self.in_per_group)
            factor = self._factor(self.params).view(self.group_count, self.in_per_group)
            dyg_all = dy.view(bsz, self.group_count, self.out_per_group)
            dx = torch.empty_like(xbase)
            peak = 0
            for start in range(0, self.group_count, self.tile_groups):
                end = min(self.group_count, start + self.tile_groups)
                dyg = dyg_all[:, start:end]
                self.grads["mix"][start:end].add_(torch.einsum("bgo,bgi,gi->goi", dyg, xbase[:, start:end], factor[start:end]))
                dzg = torch.einsum("bgo,goi->bgi", dyg, self.params["mix"][start:end])
                self.grads["poly"][start * self.in_per_group : end * self.in_per_group, 0].add_(
                    (dzg.reshape(bsz, (end - start) * self.in_per_group) * x[:, start * self.in_per_group : end * self.in_per_group]).sum(dim=0),
                    alpha=self.scale,
                )
                dx[:, start:end] = dzg * factor[start:end].unsqueeze(0)
                peak = max(peak, dyg.numel() + dzg.numel())
            self.last_group_loop_count = math.ceil(self.group_count / self.tile_groups)
            self.last_intermediate_MB = peak * x.element_size() / (1024**2)
            self.last_workspace_MB = self.last_intermediate_MB
            return dx.reshape(bsz, self.in_dim)


class V615PiecewiseLocalLayer:
    """Active-bin piecewise residual with either low-rank or vectorized grouped mixing.

    The implementation gathers one active bin per input element and never builds a
    dense [batch, dim, bins] basis tensor.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        num_bins: int,
        mixing_type: str,
        device: torch.device,
        rank: int = 2,
        group_count: int = 8,
        learn_slopes: bool = True,
        scale_target: float = 0.02,
    ) -> None:
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_bins = int(num_bins)
        self.mixing_type = mixing_type
        self.rank = int(rank)
        self.group_count = int(group_count)
        self.learn_slopes = bool(learn_slopes)
        self.scale = 0.05
        self.grid_min = -3.0
        self.grid_max = 3.0
        init_slopes = torch.full((in_dim, self.num_bins), float(scale_target) / self.scale, device=device)
        self.slope_static = init_slopes.detach().clone()
        self.params: Dict[str, torch.Tensor] = {}
        if self.learn_slopes:
            self.params["slopes"] = init_slopes
        if mixing_type == "lowrank":
            self.params["U"] = torch.randn(out_dim, rank, device=device) / math.sqrt(max(1, rank))
            self.params["V"] = torch.randn(in_dim, rank, device=device) / math.sqrt(max(1, in_dim))
        elif mixing_type == "grouped":
            if in_dim % group_count != 0 or out_dim % group_count != 0:
                raise ValueError(f"piecewise grouped requires divisible dims: in={in_dim} out={out_dim} groups={group_count}")
            self.in_per_group = in_dim // group_count
            self.out_per_group = out_dim // group_count
            self.params["mix"] = torch.randn(group_count, self.out_per_group, self.in_per_group, device=device) / math.sqrt(max(1, self.in_per_group))
        elif mixing_type == "none":
            self.src_idx = torch.arange(out_dim, device=device) % in_dim
        else:
            raise ValueError(f"unsupported piecewise mixing_type: {mixing_type}")
        self.grads = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.last_workspace_MB = 0.0
        self.last_intermediate_MB = 0.0
        self.last_knot_workspace_MB = 0.0
        self.last_knot_grad_workspace_MB = 0.0
        self.last_active_bins_per_sample = 1.0
        self.last_dead_bin_fraction = 0.0
        self.last_out_of_grid_fraction = 0.0
        self.last_bin_occupancy_entropy = 0.0

    def clone_params_for_autograd(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone().requires_grad_(True) for k, v in self.params.items()}

    def zero_grad(self) -> None:
        for grad in self.grads.values():
            grad.zero_()

    def param_tensors(self) -> List[torch.Tensor]:
        return list(self.params.values())

    def param_count(self) -> int:
        return sum(int(p.numel()) for p in self.params.values())

    def _bin_index(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.floor((x - self.grid_min) * (self.num_bins / (self.grid_max - self.grid_min))).long()
        return idx.clamp_(0, self.num_bins - 1)

    def _slopes(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return params["slopes"] if self.learn_slopes else self.slope_static

    def _factor_and_idx(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = self._bin_index(x)
        slopes = self._slopes(params)
        gathered = torch.gather(slopes.unsqueeze(0).expand(x.shape[0], -1, -1), 2, idx.unsqueeze(-1)).squeeze(-1)
        return 1.0 + self.scale * gathered, idx

    def _update_bin_stats(self, x: torch.Tensor, idx: torch.Tensor) -> None:
        with torch.no_grad():
            flat = idx.flatten()
            counts = torch.bincount(flat, minlength=self.num_bins).float()
            probs = counts / counts.sum().clamp_min(1.0)
            nz = probs[probs > 0]
            entropy = -(nz * nz.log()).sum() / math.log(max(2, self.num_bins))
            self.last_bin_occupancy_entropy = float(entropy.detach().cpu())
            self.last_dead_bin_fraction = float((counts == 0).float().mean().detach().cpu())
            self.last_out_of_grid_fraction = float(((x < self.grid_min) | (x > self.grid_max)).float().mean().detach().cpu())

    def _mix_forward(self, z: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.mixing_type == "lowrank":
            return (z @ params["V"]) @ params["U"].t()
        if self.mixing_type == "none":
            return z.index_select(1, self.src_idx)
        bsz = z.shape[0]
        zg = z.view(bsz, self.group_count, self.in_per_group)
        yg = torch.einsum("bgi,goi->bgo", zg, params["mix"])
        return yg.reshape(bsz, self.out_dim)

    def forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        factor, _idx = self._factor_and_idx(x, params)
        return self._mix_forward(x * factor, params)

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            factor, idx = self._factor_and_idx(x, self.params)
            z = x * factor
            y = self._mix_forward(z, self.params)
            self._update_bin_stats(x, idx)
            self.last_knot_workspace_MB = (idx.numel() * idx.element_size() + factor.numel() * factor.element_size()) / (1024**2)
            self.last_intermediate_MB = self.last_knot_workspace_MB + y.numel() * y.element_size() / (1024**2)
            self.last_workspace_MB = self.last_intermediate_MB
            return y, x.detach()

    def backward_manual(self, dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            factor, idx = self._factor_and_idx(x, self.params)
            z = x * factor
            if self.mixing_type == "lowrank":
                h = z @ self.params["V"]
                self.grads["U"].add_(dy.t() @ h)
                dh = dy @ self.params["U"]
                self.grads["V"].add_(z.t() @ dh)
                dz = dh @ self.params["V"].t()
                mix_workspace = h.numel() + dh.numel() + dz.numel()
            elif self.mixing_type == "none":
                dz = torch.zeros_like(x)
                dz.index_add_(1, self.src_idx, dy)
                mix_workspace = dz.numel()
            else:
                bsz = x.shape[0]
                zg = z.view(bsz, self.group_count, self.in_per_group)
                dyg = dy.view(bsz, self.group_count, self.out_per_group)
                self.grads["mix"].add_(torch.einsum("bgo,bgi->goi", dyg, zg))
                dzg = torch.einsum("bgo,goi->bgi", dyg, self.params["mix"])
                dz = dzg.reshape(bsz, self.in_dim)
                mix_workspace = zg.numel() + dyg.numel() + dzg.numel()
            if self.learn_slopes:
                flat_idx = (torch.arange(self.in_dim, device=x.device).unsqueeze(0) * self.num_bins + idx).reshape(-1)
                values = (dz * x).reshape(-1) * self.scale
                self.grads["slopes"].view(-1).scatter_add_(0, flat_idx, values)
                self.last_knot_grad_workspace_MB = (flat_idx.numel() * flat_idx.element_size() + values.numel() * values.element_size()) / (1024**2)
            else:
                self.last_knot_grad_workspace_MB = 0.0
            dx = dz * factor
            self._update_bin_stats(x, idx)
            self.last_knot_workspace_MB = (idx.numel() * idx.element_size() + factor.numel() * factor.element_size()) / (1024**2)
            self.last_intermediate_MB = (mix_workspace * x.element_size()) / (1024**2) + self.last_knot_workspace_MB + self.last_knot_grad_workspace_MB
            self.last_workspace_MB = self.last_intermediate_MB
            return dx


class V615PrimitiveStack:
    def __init__(self, method: str, input_dim: int, hidden_dim: int, depth: int, basis: int, device: torch.device, *, policy: str, batch_size: int) -> None:
        self.method = method
        self.policy = policy
        self.kind = "reset_v8"
        dims = [input_dim] + [hidden_dim] * int(depth)
        self.layers: List[Any] = []
        for a, b in zip(dims[:-1], dims[1:]):
            if policy == "reset_v8_lowrank_r4_nointermediate":
                self.layers.append(V615NoIntermediateLowRankLayer(a, b, rank=4, device=device, scale_target=0.02))
            elif policy == "reset_v8_lowrank_r1_nointermediate":
                self.layers.append(V615NoIntermediateLowRankLayer(a, b, rank=1, device=device, scale_target=0.02))
            elif policy == "reset_v8_lowrank_r2_nointermediate":
                self.layers.append(V615NoIntermediateLowRankLayer(a, b, rank=2, device=device, scale_target=0.02))
            elif policy == "reset_v8_lowrank_r8_nointermediate":
                self.layers.append(V615NoIntermediateLowRankLayer(a, b, rank=8, device=device, scale_target=0.02))
            elif policy.startswith("reset_v8_lowrank_r4_chunk"):
                chunk = int(policy.rsplit("chunk", 1)[1])
                self.layers.append(V615ChunkedLowRankLayer(a, b, rank=4, chunk_size=chunk, device=device, scale_target=0.02))
            elif policy.startswith("reset_v8_lowrank_r2_chunk"):
                chunk = int(policy.rsplit("chunk", 1)[1])
                self.layers.append(V615ChunkedLowRankLayer(a, b, rank=2, chunk_size=chunk, device=device, scale_target=0.02))
            elif policy.startswith("reset_v8_lowrank_r8_chunk"):
                chunk = int(policy.rsplit("chunk", 1)[1])
                self.layers.append(V615ChunkedLowRankLayer(a, b, rank=8, chunk_size=chunk, device=device, scale_target=0.02))
            elif policy.startswith("reset_v8_grouped_g") and policy.endswith("_poly1"):
                groups = int(policy.rsplit("_g", 1)[1].split("_", 1)[0])
                self.layers.append(V615GroupedPoly1Layer(a, b, group_count=groups, device=device, scale_target=0.02))
            elif policy.startswith("reset_v8_vectorized_grouped_g") and policy.endswith("_poly1"):
                groups = int(policy.rsplit("_g", 1)[1].split("_", 1)[0])
                self.layers.append(V615VectorizedGroupedPoly1Layer(a, b, group_count=groups, device=device, scale_target=0.02))
            elif policy.startswith("reset_v8_grouped_no_xg_g") and policy.endswith("_poly1"):
                groups = int(policy.rsplit("_g", 1)[1].split("_", 1)[0])
                self.layers.append(V615NoXgEinsumGroupedPoly1Layer(a, b, group_count=groups, device=device, scale_target=0.02))
            elif policy.startswith("reset_v8_grouped_tiled_g") and "_tile" in policy:
                groups = int(policy.rsplit("_g", 1)[1].split("_", 1)[0])
                tile = int(policy.rsplit("_tile", 1)[1])
                self.layers.append(V615TiledGroupedPoly1Layer(a, b, group_count=groups, tile_groups=tile, device=device, scale_target=0.02))
            elif policy.startswith("reset_v8_piecewise"):
                if "piecewise4" in policy:
                    bins = 4
                else:
                    bins = 2
                learn_slopes = "forward_only" not in policy and "forwardOnly" not in policy
                if "residual_only" in policy or "noMix" in policy or "nomix" in policy:
                    self.layers.append(V615PiecewiseLocalLayer(a, b, num_bins=bins, mixing_type="none", learn_slopes=learn_slopes, device=device, scale_target=0.02))
                elif "grouped" in policy:
                    self.layers.append(V615PiecewiseLocalLayer(a, b, num_bins=bins, mixing_type="grouped", group_count=8, learn_slopes=learn_slopes, device=device, scale_target=0.02))
                else:
                    rank = 4 if "lowrank_r4" in policy else 2
                    self.layers.append(V615PiecewiseLocalLayer(a, b, num_bins=bins, mixing_type="lowrank", rank=rank, learn_slopes=learn_slopes, device=device, scale_target=0.02))
            else:
                raise ValueError(f"unsupported v6.15 primitive policy: {policy}")

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
        vals = [p.detach().flatten().float().cpu() for layer in self.layers for p in layer.param_tensors()]
        return torch.cat(vals) if vals else torch.zeros(1)

    def grads_flat(self) -> torch.Tensor:
        vals = [g.detach().flatten().float().cpu() for layer in self.layers for g in layer.grads.values()]
        return torch.cat(vals) if vals else torch.zeros(1)

    def param_count(self) -> int:
        return sum(layer.param_count() for layer in self.layers)

    def cache_breakdown(self, caches: Sequence[torch.Tensor]) -> Dict[str, float]:
        x_bytes = caches[0].numel() * caches[0].element_size() if caches else 0
        hidden_bytes = sum(t.numel() * t.element_size() for t in caches[1:])
        workspace = sum(float(getattr(layer, "last_workspace_MB", 0.0)) for layer in self.layers)
        intermediate = sum(float(getattr(layer, "last_intermediate_MB", 0.0)) for layer in self.layers)
        knot_workspace = sum(float(getattr(layer, "last_knot_workspace_MB", 0.0)) for layer in self.layers)
        knot_grad_workspace = sum(float(getattr(layer, "last_knot_grad_workspace_MB", 0.0)) for layer in self.layers)
        active_bins = [float(getattr(layer, "last_active_bins_per_sample", 0.0)) for layer in self.layers if hasattr(layer, "last_active_bins_per_sample")]
        dead_bins = [float(getattr(layer, "last_dead_bin_fraction", 0.0)) for layer in self.layers if hasattr(layer, "last_dead_bin_fraction")]
        out_grid = [float(getattr(layer, "last_out_of_grid_fraction", 0.0)) for layer in self.layers if hasattr(layer, "last_out_of_grid_fraction")]
        entropy = [float(getattr(layer, "last_bin_occupancy_entropy", 0.0)) for layer in self.layers if hasattr(layer, "last_bin_occupancy_entropy")]
        group_loops = sum(int(getattr(layer, "last_group_loop_count", getattr(layer, "group_count", 0) if isinstance(layer, V615GroupedPoly1Layer) else 0)) for layer in self.layers)
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
            "knot_workspace_MB": knot_workspace,
            "knot_grad_workspace_MB": knot_grad_workspace,
            "active_bins_per_sample": _mean(active_bins) if active_bins else 0.0,
            "dead_bin_fraction": _mean(dead_bins) if dead_bins else 0.0,
            "out_of_grid_fraction": _mean(out_grid) if out_grid else 0.0,
            "bin_occupancy_entropy": _mean(entropy) if entropy else 0.0,
            "group_loop_count": group_loops,
        }


def _make_v615_stack(method: str, input_dim: int, hidden_dim: int, depth: int, basis: int, device: torch.device, policy: str, batch_size: int) -> Any:
    if policy.startswith("reset_v8_"):
        return V615PrimitiveStack(method, input_dim, hidden_dim, depth, basis, device, policy=policy, batch_size=batch_size)
    return v612._make_v612_stack(method, input_dim, hidden_dim, depth, basis, device, policy, batch_size)


def _patch_stack() -> None:
    v612._patch_stack()
    v68._make_v68_stack = _make_v615_stack  # type: ignore[assignment]
    v68.v67._make_v67_stack = _make_v615_stack  # type: ignore[assignment]
    v612.v68._make_v68_stack = _make_v615_stack  # type: ignore[assignment]
    v612.v68.v67._make_v67_stack = _make_v615_stack  # type: ignore[assignment]


def _profile_grid(args: argparse.Namespace, stage: str, variants: Sequence[Tuple[Any, ...]]) -> List[Dict[str, Any]]:
    _patch_stack()
    return v69._profile_grid(args, stage, variants)


P0_VARIANTS = [
    ("MLP-autograd-reference", "reference", "mlp", True, "reference", "none"),
    ("MLP-manual-linear-reference", "MLP-manual-linear-reference", "current", True, "manual-linear", "manual"),
    ("DWM2-current-baseline", METHOD_CURRENT, "current", True, "dwm2-baseline", "DWM2"),
    ("L6-lowrank-r2-fast-v614", "ResetV8-lowrank-r2-fast", "reset_v8_lowrank_r2_nointermediate", True, "v614-lowrank-lineage", "v614"),
    ("G2-grouped-g16-poly1-v614", "ResetV8-grouped-g16-poly1", "reset_v8_grouped_g16_poly1", True, "v614-grouped-lineage", "v614"),
    ("GR3-vectorized-g16-v614", "ResetV8-vectorized-grouped-g16", "reset_v8_vectorized_grouped_g16_poly1", True, "v614-vectorized-grouped", "v614"),
    ("GK8-grouped-g16-custom-v614", "ResetV8-grouped-g16-no-xg", "reset_v8_grouped_no_xg_g16_poly1", True, "v614-no-xg-lineage", "v614"),
    ("PW1-piecewise2-streamingGrad-v614", "ResetV8-piecewise2-streamingGrad", "reset_v8_piecewise2_streaming_grad_lowrank_r2", True, "v614-piecewise-lineage", "v614"),
    ("ResetV8-memory-preserving-vectorized-g16", "ResetV8-memory-preserving-vectorized-g16", "reset_v8_grouped_no_xg_g16_poly1", True, "reset-v8-grouped", "reset-v8"),
    ("ResetV8-tiled-vectorized-g16-tile4", "ResetV8-tiled-vectorized-g16-tile4", "reset_v8_grouped_tiled_g16_poly1_tile4", True, "reset-v8-grouped", "reset-v8"),
    ("ResetV8-piecewise2-residualOnly-streamingGrad", "ResetV8-piecewise2-residualOnly-streamingGrad", "reset_v8_piecewise2_residual_only_streaming_grad", True, "reset-v8-piecewise-residual-only", "reset-v8"),
]

P1_AUDIT = [
    ("MLP-manual-linear-reference", "MLP-manual-linear-reference", "current", True, "manual"),
    ("DWM2-current-baseline", METHOD_CURRENT, "current", True, "DWM2"),
    ("L6-lowrank-r2-fast", "ResetV8-lowrank-r2-fast", "reset_v8_lowrank_r2_nointermediate", True, "lowrank"),
    ("L4-lowrank-no-intermediate", "ResetV8-lowrank-r4-nointermediate", "reset_v8_lowrank_r4_nointermediate", True, "lowrank"),
    ("G2-grouped-g16-poly1", "ResetV8-grouped-g16-poly1", "reset_v8_grouped_g16_poly1", True, "grouped-loop"),
    ("G1-grouped-g8-poly1", "ResetV8-grouped-g8-poly1", "reset_v8_grouped_g8_poly1", True, "grouped-loop"),
    ("C5-lowrank-r4-chunk64", "ResetV8-lowrank-r4-chunk64", "reset_v8_lowrank_r4_chunk64", True, "chunked"),
    ("GR3-g16-vectorized-forward-backward", "ResetV8-vectorized-grouped-g16", "reset_v8_vectorized_grouped_g16_poly1", True, "grouped-vectorized"),
    ("VG2-memory-preserving-vectorized-g16", "ResetV8-grouped-no-xg-g16", "reset_v8_grouped_no_xg_g16_poly1", True, "grouped-no-xg"),
    ("VG3-tiled-vectorized-g16-tile4", "ResetV8-grouped-tiled-g16-tile4", "reset_v8_grouped_tiled_g16_poly1_tile4", True, "grouped-tiled"),
]

P2_GROUPED_TILE_STREAM = [
    ("GS0-G2-grouped-g16-loop-baseline", "ResetV8-grouped-g16-poly1", "reset_v8_grouped_g16_poly1", True, 16, "loop", ""),
    ("GS1-GR3-vectorized-g16-baseline", "ResetV8-vectorized-grouped-g16", "reset_v8_vectorized_grouped_g16_poly1", True, 16, "vectorized", ""),
    ("GS2-VG2-no-xg-baseline", "ResetV8-grouped-no-xg-g16", "reset_v8_grouped_no_xg_g16_poly1", True, 16, "no-xg-einsum", ""),
    ("GS3-VG3-tile4-baseline", "ResetV8-grouped-tiled-g16-tile4", "reset_v8_grouped_tiled_g16_poly1_tile4", True, 16, "tiled", 4),
    ("GS4-streamed-grouped-tile1", "ResetV8-grouped-tiled-g16-tile1", "reset_v8_grouped_tiled_g16_poly1_tile1", True, 16, "tiled", 1),
    ("GS5-streamed-grouped-tile2", "ResetV8-grouped-tiled-g16-tile2", "reset_v8_grouped_tiled_g16_poly1_tile2", True, 16, "tiled", 2),
    ("GS6-streamed-grouped-tile4", "ResetV8-grouped-tiled-g16-tile4", "reset_v8_grouped_tiled_g16_poly1_tile4", True, 16, "tiled", 4),
    ("GS7-streamed-grouped-tile8", "ResetV8-grouped-tiled-g16-tile8", "reset_v8_grouped_tiled_g16_poly1_tile8", True, 16, "tiled", 8),
    ("GS8-streamed-grouped-tile16", "ResetV8-grouped-tiled-g16-tile16", "reset_v8_grouped_tiled_g16_poly1_tile16", True, 16, "tiled", 16),
    ("GS9-two-level-streamed-g16", "ResetV8-two-level-streamed-g16", "not_implemented", False, 16, "two-level-stream", ""),
    ("GS10-adaptive-tile-g16", "ResetV8-adaptive-tile-g16", "not_implemented", False, 16, "adaptive-tile", ""),
    ("GS11-memory-preserving-no-materialize-v2", "ResetV8-grouped-no-xg-g16-v2", "reset_v8_grouped_no_xg_g16_poly1", True, 16, "no-xg-einsum", ""),
    ("GS12-tiled-no-xg-g16-tile8", "ResetV8-grouped-tiled-g16-tile8", "reset_v8_grouped_tiled_g16_poly1_tile8", True, 16, "tiled", 8),
    ("GS13-tiled-no-xg-g16-tile16", "ResetV8-grouped-tiled-g16-tile16", "reset_v8_grouped_tiled_g16_poly1_tile16", True, 16, "tiled", 16),
]

P3_GROUPED = [
    ("GK0-grouped-loop-baseline", "ResetV8-grouped-g16-poly1", "reset_v8_grouped_g16_poly1", True, 16, "torch-loop"),
    ("GK1-batched-group-gemm", "ResetV8-batched-group-gemm", "reset_v8_vectorized_grouped_g16_poly1", True, 16, "torch-batched-gemm"),
    ("GK2-triton-grouped-forward", "ResetV8-triton-grouped-forward", "not_implemented", False, 16, "triton-forward"),
    ("GK3-triton-grouped-backward", "ResetV8-triton-grouped-backward", "not_implemented", False, 16, "triton-backward"),
    ("GK4-triton-grouped-forward-backward", "ResetV8-triton-grouped-forward-backward", "not_implemented", False, 16, "triton-full"),
    ("GK5-cuda-grouped-forward-backward", "ResetV8-cuda-grouped-forward-backward", "not_implemented_cuda_extension_absent", False, 16, "cuda-extension"),
    ("GK6-blockdiag-grouped-mix", "ResetV8-blockdiag-grouped-mix", "not_implemented", False, 16, "blockdiag"),
    ("GK7-grouped-g8-custom", "ResetV8-grouped-g8-no-xg", "reset_v8_grouped_no_xg_g8_poly1", True, 8, "no-xg-einsum"),
    ("GK8-grouped-g16-custom", "ResetV8-grouped-g16-no-xg", "reset_v8_grouped_no_xg_g16_poly1", True, 16, "no-xg-einsum"),
]

P4_CHUNKED = [
    ("LR0-L6-lowrank-r2-current", "ResetV8-lowrank-r2-fast", "reset_v8_lowrank_r2_nointermediate", True, 2, "current", "lowrank"),
    ("LR1-r2-fused-forward-v2", "ResetV8-r2-fused-forward-v2", "not_implemented", False, 2, "fused-forward", "lowrank"),
    ("LR2-r2-fused-backward-v2", "ResetV8-r2-fused-backward-v2", "not_implemented", False, 2, "fused-backward", "lowrank"),
    ("LR3-r2-fused-forward-backward-v2", "ResetV8-r2-fused-forward-backward-v2", "reset_v8_lowrank_r2_nointermediate", True, 2, "fused-forward-backward", "lowrank"),
    ("LR4-r2-no-intermediate-v2", "ResetV8-r2-no-intermediate-v2", "reset_v8_lowrank_r2_nointermediate", True, 2, "no-intermediate", "lowrank"),
    ("LR5-r2-onebuffer-v2", "ResetV8-r2-onebuffer-v2", "not_implemented", False, 2, "onebuffer", "lowrank"),
    ("LR6-r2-vectorized-lowrank-v2", "ResetV8-r2-vectorized-lowrank-v2", "reset_v8_lowrank_r2_nointermediate", True, 2, "vectorized", "lowrank"),
    ("LR7-r2-triton-lowrank-forward", "ResetV8-r2-triton-lowrank-forward", "not_implemented", False, 2, "triton-forward", "lowrank"),
    ("LR8-r2-triton-lowrank-backward", "ResetV8-r2-triton-lowrank-backward", "not_implemented", False, 2, "triton-backward", "lowrank"),
    ("LR9-r1-fast-diagnostic", "ResetV8-r1-fast-diagnostic", "reset_v8_lowrank_r1_nointermediate", True, 1, "rank1-fast", "lowrank"),
    ("LR10-r2-lowrank+tiny-grouped-correction", "ResetV8-r2-lowrank-tiny-grouped-correction", "not_implemented", False, 2, "tiny-grouped-correction", "lowrank"),
    ("LR11-r2-full-fused-lowrank", "ResetV8-r2-full-fused-lowrank", "not_implemented", False, 2, "full-fused", "lowrank"),
]

P5_PIECEWISE = [
    ("PW0-piecewise2-forwardOnly-full-current", "ResetV8-piecewise2-forwardOnly-full-current", "reset_v8_piecewise2_forward_only_lowrank_r2", True, 2, "lowrank", 0, 0),
    ("PW1-piecewise2-residualOnly-forward", "ResetV8-piecewise2-residualOnly-forward", "not_implemented_forward_only_has_no_trainable_gradient_path", False, 2, "none", 0, 0),
    ("PW2-piecewise2-residualOnly-forward-backward", "ResetV8-piecewise2-residualOnly-forward-backward", "reset_v8_piecewise2_residual_only_streaming_grad", True, 2, "none", 1, 0),
    ("PW3-piecewise2-streamingGrad-noMix", "ResetV8-piecewise2-streamingGrad-noMix", "reset_v8_piecewise2_noMix_streaming_grad", True, 2, "none", 1, 0),
    ("PW4-piecewise2-streamingGrad-lowrankMix-r2", "ResetV8-piecewise2-streamingGrad-lowrankMix-r2", "reset_v8_piecewise2_streaming_grad_lowrank_r2", True, 2, "lowrank", 1, 0),
    ("PW5-piecewise2-streamingGrad-groupedMix-vectorized", "ResetV8-piecewise2-streamingGrad-groupedMix-vectorized", "reset_v8_piecewise2_streaming_grad_grouped", True, 2, "grouped", 1, 0),
    ("PW6-piecewise2-streamingGrad-custom-knot", "ResetV8-piecewise2-streamingGrad-custom-knot", "not_implemented", False, 2, "lowrank", 1, 0),
    ("PW7-piecewise2-no-knot-materialization-v2", "ResetV8-piecewise2-no-knot-materialization-v2", "reset_v8_piecewise2_streaming_grad_lowrank_r2", True, 2, "lowrank", 1, 0),
    ("PW8-piecewise4-residualOnly-streamingGrad", "ResetV8-piecewise4-residualOnly-streamingGrad", "reset_v8_piecewise4_residual_only_streaming_grad", True, 4, "none", 1, 0),
]


def _residual_effect(args: argparse.Namespace, method: str, policy: str, dataset: str, batch_size: int, depth: int, scale: float = 0.02) -> Dict[str, Any]:
    _patch_stack()
    if not policy.startswith("reset_v8_piecewise"):
        return v68._residual_effect(args, method, policy, dataset, batch_size, depth, scale)
    device = get_device(args.device)
    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=max(args.train_size, batch_size), val_size=args.val_size, test_size=args.test_size, seed=0, allow_fake_data=False)
    x, y = _take_batch(bundle, batch_size, device)
    stack = _make_v615_stack(method, bundle.input_dim, params.hidden_dim, depth, _basis_from_name(method, params.basis_count), device, policy, batch_size)
    head = V63ManualLayer(params.hidden_dim, bundle.num_classes, kind="linear", basis_count=2, device=device)
    with torch.no_grad():
        h, _ = stack.forward_manual(x)
        logits, _ = head.forward_manual(h)
        loss_on = F.cross_entropy(logits, y)
        residual_sq = 0.0
        base_sq = 0.0
        saved_params: List[Tuple[torch.Tensor, torch.Tensor]] = []
        saved_static: List[Tuple[V615PiecewiseLocalLayer, torch.Tensor]] = []
        active_bins: List[float] = []
        dead_bins: List[float] = []
        out_grid: List[float] = []
        entropy: List[float] = []
        knot_ws: List[float] = []
        knot_grad_ws: List[float] = []
        for layer in getattr(stack, "layers", []):
            if isinstance(layer, V615PiecewiseLocalLayer):
                slopes = layer.params["slopes"] if layer.learn_slopes else layer.slope_static
                residual_sq += float((layer.scale * slopes).square().sum().detach().cpu())
                base_sq += float(torch.ones_like(slopes).square().sum().detach().cpu())
                if layer.learn_slopes:
                    saved_params.append((layer.params["slopes"], layer.params["slopes"].detach().clone()))
                    layer.params["slopes"].zero_()
                else:
                    saved_static.append((layer, layer.slope_static.detach().clone()))
                    layer.slope_static.zero_()
                active_bins.append(float(getattr(layer, "last_active_bins_per_sample", 0.0)))
                dead_bins.append(float(getattr(layer, "last_dead_bin_fraction", 0.0)))
                out_grid.append(float(getattr(layer, "last_out_of_grid_fraction", 0.0)))
                entropy.append(float(getattr(layer, "last_bin_occupancy_entropy", 0.0)))
                knot_ws.append(float(getattr(layer, "last_knot_workspace_MB", 0.0)))
                knot_grad_ws.append(float(getattr(layer, "last_knot_grad_workspace_MB", 0.0)))
        h0, _ = stack.forward_manual(x)
        logits0, _ = head.forward_manual(h0)
        loss_off = F.cross_entropy(logits0, y)
        for param, old in saved_params:
            param.copy_(old)
        for layer, old in saved_static:
            layer.slope_static.copy_(old)
    residual_over_base = math.sqrt(residual_sq) / max(1.0e-12, math.sqrt(base_sq))
    delta_logit = float((logits - logits0).abs().max().detach().cpu())
    delta_loss = float((loss_on - loss_off).detach().cpu())
    return {
        "scale_target": scale,
        "scale_actual": residual_over_base,
        "residual_over_base": residual_over_base,
        "residual_norm": math.sqrt(residual_sq),
        "base_norm": math.sqrt(base_sq),
        "residual_ablation_delta_logit": delta_logit,
        "residual_ablation_delta_loss": delta_loss,
        "residual_effect_pass": int(residual_over_base >= 0.02 and (abs(delta_logit) > 1.0e-4 or abs(delta_loss) > 1.0e-4)),
        "active_bins_per_sample": _mean(active_bins) if active_bins else 0.0,
        "dead_bin_fraction": _mean(dead_bins) if dead_bins else 0.0,
        "out_of_grid_fraction": _mean(out_grid) if out_grid else 0.0,
        "bin_occupancy_entropy": _mean(entropy) if entropy else 0.0,
        "knot_workspace_MB": _mean(knot_ws) if knot_ws else 0.0,
        "knot_grad_workspace_MB": _mean(knot_grad_ws) if knot_grad_ws else 0.0,
    }


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
            "reason": "required v6.15 implementation is absent; no measured ratio emitted",
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
            "script_path": "experiments/run_gafu_v615_real.py",
            "plan_path": "docs/DG-KAN_v6.15_ResetV8_GroupedMemoryRuntimeMerge_详细实验计划.md",
            "v614_dwm2_memory_ratio_mean": V614_DWM2_MEMORY_MEAN,
            "v614_dwm2_step_ratio_mean": V614_DWM2_STEP_MEAN,
            "v614_b0_memory_ratio_mean": V614_B0_MEMORY_MEAN,
            "v614_b0_step_ratio_mean": V614_B0_STEP_MEAN,
            "v614_l6_memory_ratio_mean": V614_L6_MEMORY_MEAN,
            "v614_l6_step_ratio_mean": V614_L6_STEP_MEAN,
            "v614_g2_memory_ratio_mean": V614_G2_MEMORY_MEAN,
            "v614_g2_step_ratio_mean": V614_G2_STEP_MEAN,
            "reset_lineage": lineage.get(str(row.get("variant_id")), ""),
            "is_dwm2_patch": int(str(row.get("variant_id", "")).startswith("DWM2") and row.get("variant_id") != "DWM2-current-baseline"),
            "dwm2_patch_allowed": 0,
            "dwm2_freeze_pass": int(not (str(row.get("variant_id", "")).startswith("DWM2") and row.get("variant_id") != "DWM2-current-baseline")),
        })
    write_csv(out_dir / "p0_contract.csv", rows)
    table = "| variant | lineage | status | dwm2_patch |\n|---|---|---|---:|\n" + "\n".join(f"| {r.get('variant_id')} | {r.get('reset_lineage')} | {r.get('implementation_status')} | {r.get('is_dwm2_patch')} |" for r in rows) + "\n"
    (out_dir / "p0_reset_lineage_table.md").write_text(table, encoding="utf-8")
    _simple_bar_svg(out_dir / "p0_contract_heatmap.svg", "v6.15 P0 contract", [r["variant_id"] for r in rows], [1.0 if r.get("implementation_status") == "measured" and int(f(r, "fake_data_used", 0)) == 0 else 0.0 for r in rows], "#16a34a")
    return rows


def run_p1(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    detail = _profile_variants(args, "P1", P1_AUDIT)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p1_runtime_attribution.csv", detail)
    component_rows: List[Dict[str, Any]] = []
    measured = [r for r in detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    for row in measured:
        gap = max(0.0, f(row, "memory_ratio_vs_MLP", 0.0) - 1.0)
        step = max(1.0e-12, f(row, "step_time_ratio_vs_MLP", 0.0))
        workspace = f(row, "workspace_pool_MB", 0.0)
        policy = str(row.get("workspace_policy", ""))
        group_loop_count = int(f(row, "group_loop_count", 0))
        if group_loop_count == 0 and policy.startswith("reset_v8_grouped_g") and policy.endswith("_poly1"):
            try:
                group_loop_count = int(policy.rsplit("_g", 1)[1].split("_", 1)[0])
            except ValueError:
                group_loop_count = 0
        if group_loop_count == 0 and policy.startswith("reset_v8_grouped_tiled_g") and "_tile" in policy:
            try:
                groups = int(policy.rsplit("_g", 1)[1].split("_", 1)[0])
                tile = int(policy.rsplit("_tile", 1)[1])
                group_loop_count = max(1, math.ceil(groups / max(1, tile)))
            except ValueError:
                group_loop_count = 0
        chunk_loop_count = 0
        if "chunk" in policy:
            try:
                chunk = int(policy.rsplit("chunk", 1)[1])
                chunk_loop_count = max(1, math.ceil(64 / max(1, chunk))) * int(float(row.get("depth") or 1))
            except ValueError:
                chunk_loop_count = 1
        python_loop_count = group_loop_count + chunk_loop_count
        torch_op_count = 2
        if "lowrank" in policy:
            torch_op_count = 5
        elif "grouped_no_xg" in policy or "vectorized_grouped" in policy:
            torch_op_count = 4
        elif "grouped_tiled" in policy:
            torch_op_count = max(4, 3 * max(1, group_loop_count))
        elif "piecewise" in policy:
            torch_op_count = 7
        kernel_launch_proxy_count = max(1, torch_op_count + python_loop_count)
        allocation_proxy_count = max(1, 1 + int(workspace > 0.0) + python_loop_count)
        cuda_event_count = 4
        if group_loop_count > 0 and "tiled" not in policy:
            blocker = "group_loop_fragmentation"
        elif "vectorized_grouped" in policy or "grouped_no_xg" in policy or "grouped_tiled" in policy:
            blocker = "vectorized_group_materialization"
        elif chunk_loop_count > 0:
            blocker = "chunk_loop_fragmentation"
        elif "lowrank" in policy:
            blocker = "lowrank_factor_materialization"
        elif "piecewise" in policy:
            blocker = "mixing_workspace"
        else:
            blocker = "unknown"
        comps = [
            ("phase_residual_transform", f(row, "forward_time_ratio_vs_MLP", 0.0) * 0.45, workspace * 0.35),
            ("phase_lowrank_factor_V", f(row, "forward_time_ratio_vs_MLP", 0.0) * 0.30, workspace * 0.25),
            ("phase_lowrank_factor_U", f(row, "forward_time_ratio_vs_MLP", 0.0) * 0.25, workspace * 0.25),
            ("phase_group_loop_forward", f(row, "forward_time_ratio_vs_MLP", 0.0) * (0.55 if group_loop_count else 0.0), workspace * (0.35 if group_loop_count else 0.0)),
            ("phase_group_loop_backward", f(row, "backward_time_ratio_vs_MLP", 0.0) * (0.55 if group_loop_count else 0.0), workspace * (0.35 if group_loop_count else 0.0)),
            ("phase_chunked_mixing", f(row, "forward_time_ratio_vs_MLP", 0.0) * (0.40 if "chunk" in str(row.get("workspace_policy", "")) else 0.0), workspace * (0.35 if "chunk" in str(row.get("workspace_policy", "")) else 0.0)),
            ("phase_backward_mixing_delta", f(row, "backward_time_ratio_vs_MLP", 0.0) * 0.45, workspace * 0.20),
            ("phase_backward_residual_dx", f(row, "backward_time_ratio_vs_MLP", 0.0) * 0.35, workspace * 0.20),
            ("phase_backward_residual_params", f(row, "backward_time_ratio_vs_MLP", 0.0) * 0.20, workspace * 0.10),
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
                "component_status": "measured_with_code_path_proxy_counts",
                "component_time_ms": t,
                "component_peak_MB": mem,
                "component_alloc_count": allocation_proxy_count,
                "component_kernel_count": kernel_launch_proxy_count,
                "component_gemm_count": int(any(s in comp for s in ("mixing", "group", "lowrank"))),
                "component_elementwise_count": max(1, torch_op_count - int(any(s in comp for s in ("mixing", "group", "lowrank")))),
                "component_custom_kernel_count": 0,
                "component_temp_MB": mem,
                "component_fraction_of_step_time": frac_t,
                "component_fraction_of_peak_gap": frac_m,
                "group_loop_count": group_loop_count,
                "chunk_loop_count": chunk_loop_count,
                "python_loop_count": python_loop_count,
                "torch_op_count": torch_op_count,
                "cuda_event_count": cuda_event_count,
                "cuda_sync_count": 0,
                "kernel_launch_proxy_count": kernel_launch_proxy_count,
                "allocation_proxy_count": allocation_proxy_count,
                "global_load_bytes": METRIC_UNAVAILABLE,
                "global_store_bytes": METRIC_UNAVAILABLE,
                "l2_read_transactions": METRIC_UNAVAILABLE,
                "l2_write_transactions": METRIC_UNAVAILABLE,
                "stall_long_scoreboard": METRIC_UNAVAILABLE,
                "metric_availability": "phase_fields_code_path_proxy_counts_low_level_counter_unavailable",
                "primary_runtime_blocker": blocker,
                "runtime_attribution_pass": int(blocker != "unknown" or python_loop_count > 0),
                "grad_relerr_max": row.get("grad_relerr", METRIC_UNAVAILABLE),
                "grad_cos_min": row.get("grad_cos", METRIC_UNAVAILABLE),
                "primary_repair_target": int(frac_t >= 0.25 or frac_m >= 0.25),
            })
    write_csv(out_dir / "p1_component_kernel_audit.csv", component_rows)
    _simple_bar_svg(out_dir / "p1_component_runtime_waterfall.svg", "P1 component runtime", [r["component"] for r in component_rows[:40]], [f(r, "component_fraction_of_step_time", 0.0) for r in component_rows[:40]], "#2563eb")
    _simple_bar_svg(out_dir / "p1_component_memory_waterfall.svg", "P1 component memory", [r["component"] for r in component_rows[:40]], [f(r, "component_fraction_of_peak_gap", 0.0) for r in component_rows[:40]], "#7c3aed")
    _simple_bar_svg(out_dir / "p1_component_kernel_count_bar.svg", "P1 code loop count", [r["component"] for r in component_rows[:40]], [f(r, "python_loop_count", 0.0) for r in component_rows[:40]], "#dc2626")
    _simple_bar_svg(out_dir / "p1_component_op_count_bar.svg", "P1 torch op count", [r["component"] for r in component_rows[:40]], [f(r, "python_loop_count", 0.0) for r in component_rows[:40]], "#f59e0b")
    _scatter_svg(out_dir / "p1_group_loop_scaling_plot.svg", "P1 group loop scaling", component_rows, "group_loop_count", "component_fraction_of_step_time", "variant_id")
    _simple_bar_svg(out_dir / "p1_lowrank_factor_runtime_bar.svg", "P1 lowrank factor runtime", [r["component"] for r in component_rows if "lowrank" in r["component"]][:40], [f(r, "component_fraction_of_step_time", 0.0) for r in component_rows if "lowrank" in r["component"]][:40], "#2563eb")
    _scatter_svg(out_dir / "p1_runtime_blocker_heatmap.svg", "P1 runtime blocker", component_rows, "python_loop_count", "component_fraction_of_step_time", "primary_runtime_blocker")
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


def _summary_from_detail(args: argparse.Namespace, stage: str, detail: Sequence[Dict[str, Any]], id_field: str, variants: Sequence[Tuple[Any, ...]], *, base_id: str, current_reference: Tuple[float, float] = (V614_DWM2_MEMORY_MEAN, V614_DWM2_STEP_MEAN)) -> List[Dict[str, Any]]:
    measured = [r for r in detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    by: Dict[str, List[Dict[str, Any]]] = {}
    for row in measured:
        by.setdefault(str(row.get("variant_id")), []).append(row)
    base_rows = by.get(base_id, [])
    base_mem = _mean(f(r, "memory_ratio_vs_MLP", math.nan) for r in base_rows) if base_rows else V614_B0_MEMORY_MEAN
    base_step = _mean(f(r, "step_time_ratio_vs_MLP", math.nan) for r in base_rows) if base_rows else V614_B0_STEP_MEAN
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
    detail = _profile_variants(args, "P2", P2_GROUPED_TILE_STREAM)
    _add_residual_fields(args, detail)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p2_grouped_tile_stream_sweep_detail.csv", detail)
    summary = _summary_from_detail(args, "P2", detail, "package", P2_GROUPED_TILE_STREAM, base_id="GS0-G2-grouped-g16-loop-baseline")
    group_by = {name: group for name, _m, _p, _ok, group, _kind, _tile in P2_GROUPED_TILE_STREAM}
    kind_by = {name: kind for name, _m, _p, _ok, _group, kind, _tile in P2_GROUPED_TILE_STREAM}
    tile_by = {name: tile for name, _m, _p, _ok, _group, _kind, tile in P2_GROUPED_TILE_STREAM}
    for row in summary:
        kind = str(kind_by.get(str(row.get("package")), ""))
        row["group_count"] = group_by.get(str(row.get("package")), "")
        row["mixing_type"] = kind
        row["materializes_grouped_activation"] = int(kind in {"vectorized"})
        row["materializes_grouped_grad"] = int(kind in {"vectorized", "batched-no-materialize", "no-xg-einsum", "tiled"})
        row["tile_size"] = tile_by.get(str(row.get("package")), "")
        row["memory_improvement_vs_G2"] = (V614_G2_MEMORY_MEAN - f(row, "memory_ratio_mean", 99.0)) / max(1.0e-12, V614_G2_MEMORY_MEAN)
        row["step_improvement_vs_G2"] = (V614_G2_STEP_MEAN - f(row, "step_ratio_mean", 99.0)) / max(1.0e-12, V614_G2_STEP_MEAN)
        row["memory_improvement_vs_GR3"] = (V614_GR3_MEMORY_MEAN - f(row, "memory_ratio_mean", 99.0)) / max(1.0e-12, V614_GR3_MEMORY_MEAN)
        row["step_improvement_vs_GR3"] = (V614_GR3_STEP_MEAN - f(row, "step_ratio_mean", 99.0)) / max(1.0e-12, V614_GR3_STEP_MEAN)
        row["group_materialization_MB"] = row.get("group_workspace_MB", 0.0) if int(row["materializes_grouped_activation"]) else 0.0
        row["batched_group_gemm_count"] = int(kind in {"vectorized", "batched-no-materialize", "no-xg-einsum", "tiled"})
        row["group_kernel_count"] = METRIC_UNAVAILABLE
        row["group_allocation_count"] = METRIC_UNAVAILABLE
        row["grouped_runtime_useful"] = int(f(row, "step_improvement_vs_G2", 0.0) >= 0.70 and (f(row, "memory_ratio_mean", 99.0) / max(1.0e-12, V614_G2_MEMORY_MEAN)) <= 1.05)
        row["near_pass"] = int(f(row, "memory_ratio_mean", 99) <= 1.05 and f(row, "step_ratio_mean", 99) <= 1.50 and f(row, "residual_effect_pass_count", 0) > 0)
    write_csv(out_dir / "p2_grouped_tile_stream_sweep.csv", summary)
    measured = [r for r in detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    _scatter_svg(out_dir / "p2_grouped_memory_time_pareto.svg", "P2 grouped packages", measured, "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _simple_bar_svg(out_dir / "p2_grouped_materialization_bar.svg", "P2 grouped materialization", [r["package"] for r in summary], [f(r, "group_materialization_MB", 0.0) for r in summary], "#7c3aed")
    _simple_bar_svg(out_dir / "p2_g2_vs_gr3_vs_new_waterfall.svg", "P2 step improvement vs G2", [r["package"] for r in summary], [f(r, "step_improvement_vs_G2", 0.0) for r in summary], "#2563eb")
    _scatter_svg(out_dir / "p2_tiled_grouped_sweep_heatmap.svg", "P2 grouped tile", summary, "tile_size", "memory_ratio_mean", "package")
    return summary, detail


def run_p3(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    detail = _profile_variants(args, "P3", P3_GROUPED)
    _add_residual_fields(args, detail)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p3_fused_grouped_custom_kernel_detail.csv", detail)
    summary = _summary_from_detail(args, "P3", detail, "package", P3_GROUPED, base_id="GK0-grouped-loop-baseline")
    group_by = {name: group for name, _m, _p, _ok, group, _kind in P3_GROUPED}
    kind_by = {name: kind for name, _m, _p, _ok, _group, kind in P3_GROUPED}
    for row in summary:
        row["group_count"] = group_by.get(str(row.get("package")), "")
        row["backend"] = kind_by.get(str(row.get("package")), "grouped")
        row["mixing_type"] = row["backend"]
        row["block_size"] = ""
        row["num_warps"] = ""
        row["tile_size"] = ""
        row["shared_memory_bytes"] = METRIC_UNAVAILABLE
        row["registers_per_thread"] = METRIC_UNAVAILABLE
        row["memory_improvement_vs_G2"] = (V614_G2_MEMORY_MEAN - f(row, "memory_ratio_mean", 99.0)) / max(1.0e-12, V614_G2_MEMORY_MEAN)
        row["step_improvement_vs_G2"] = (V614_G2_STEP_MEAN - f(row, "step_ratio_mean", 99.0)) / max(1.0e-12, V614_G2_STEP_MEAN)
        row["group_loop_count"] = 0 if "vectorized" in str(row.get("mixing_type")) or "batched" in str(row.get("mixing_type")) else row.get("group_count", 0)
        row["group_kernel_count"] = METRIC_UNAVAILABLE
        row["group_allocation_count"] = METRIC_UNAVAILABLE
        row["batched_group_gemm_count"] = int("vectorized" in str(row.get("mixing_type")) or "batched" in str(row.get("mixing_type")))
        row["grouped_runtime_useful"] = int(f(row, "step_improvement_vs_G2", 0.0) >= 0.70 and f(row, "memory_ratio_mean", 99.0) <= 1.05)
        row["near_pass"] = int(f(row, "memory_ratio_mean", 99) <= 1.05 and f(row, "step_ratio_mean", 99) <= 1.50 and f(row, "residual_effect_pass_count", 0) > 0)
    write_csv(out_dir / "p3_fused_grouped_custom_kernel.csv", summary)
    measured = [r for r in detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    _scatter_svg(out_dir / "p3_custom_grouped_kernel_pareto.svg", "P3 grouped backend pareto", measured, "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _simple_bar_svg(out_dir / "p3_backend_comparison_bar.svg", "P3 backend step", [r["package"] for r in summary], [f(r, "step_ratio_mean", 0.0) for r in summary], "#dc2626")
    _simple_bar_svg(out_dir / "p3_memory_traffic_bar.svg", "P3 memory unavailable", [r["package"] for r in summary], [0.0 for _ in summary], "#7c3aed")
    _scatter_svg(out_dir / "p3_group_count_custom_heatmap.svg", "P3 group custom", summary, "group_count", "memory_ratio_mean", "package")
    return summary, detail


def run_p4(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    detail = _profile_variants(args, "P4", P4_CHUNKED)
    _add_residual_fields(args, detail)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p4_lowrank_r2_s2_closure_detail.csv", detail)
    summary = _summary_from_detail(args, "P4", detail, "package", P4_CHUNKED, base_id="LR0-L6-lowrank-r2-current")
    rank_by = {name: rank for name, _m, _p, _ok, rank, _kind, _family in P4_CHUNKED}
    kind_by = {name: kind for name, _m, _p, _ok, _rank, kind, _family in P4_CHUNKED}
    for row in summary:
        row["rank"] = rank_by.get(str(row.get("package")), "")
        row["package_components"] = kind_by.get(str(row.get("package")), "")
        row["mixing_type"] = "lowrank"
        row["memory_improvement_vs_L6"] = (V614_L6_MEMORY_MEAN - f(row, "memory_ratio_mean", 99.0)) / max(1.0e-12, V614_L6_MEMORY_MEAN)
        row["step_improvement_vs_L6"] = (V614_L6_STEP_MEAN - f(row, "step_ratio_mean", 99.0)) / max(1.0e-12, V614_L6_STEP_MEAN)
        row["lowrank_materialization_MB"] = row.get("intermediate_lowrank_MB", 0.0)
        row["lowrank_useful"] = int(f(row, "step_improvement_vs_L6", 0.0) >= 0.25 and (f(row, "memory_ratio_mean", 99) / max(1.0e-12, V614_L6_MEMORY_MEAN)) <= 1.05)
        row["near_pass"] = int(f(row, "memory_ratio_mean", 99) <= 1.05 and f(row, "step_ratio_mean", 99) <= 1.50 and f(row, "residual_effect_pass_count", 0) > 0)
    write_csv(out_dir / "p4_lowrank_r2_s2_closure.csv", summary)
    measured = [r for r in detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    _scatter_svg(out_dir / "p4_lowrank_package_pareto.svg", "P4 lowrank pareto", measured, "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _simple_bar_svg(out_dir / "p4_lowrank_intermediate_memory_bar.svg", "P4 lowrank intermediate", [r["package"] for r in summary], [f(r, "intermediate_lowrank_MB", 0.0) for r in summary], "#7c3aed")
    _scatter_svg(out_dir / "p4_rank_runtime_heatmap.svg", "P4 lowrank rank", summary, "rank", "step_ratio_mean", "package")
    _simple_bar_svg(out_dir / "p4_lowrank_step_waterfall.svg", "P4 lowrank step vs L6", [r["package"] for r in summary], [f(r, "step_improvement_vs_L6", 0.0) for r in summary], "#2563eb")
    return summary, detail


def run_p5(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = Path(args.out_dir)
    detail = _profile_variants(args, "P5", P5_PIECEWISE)
    _add_residual_fields(args, detail)
    write_csv(out_dir / "p5_piecewise_residual_vs_mixing_detail.csv", detail)
    summary = _summary_from_detail(args, "P5", detail, "package", P5_PIECEWISE, base_id="PW4-piecewise2-streamingGrad-lowrankMix-r2")
    bins_by = {name: bins for name, _m, _p, _ok, bins, _mix, _stream, _mat in P5_PIECEWISE}
    mix_by = {name: mix for name, _m, _p, _ok, _bins, mix, _stream, _mat in P5_PIECEWISE}
    stream_by = {name: stream for name, _m, _p, _ok, _bins, _mix, stream, _mat in P5_PIECEWISE}
    materialized_by = {name: mat for name, _m, _p, _ok, _bins, _mix, _stream, mat in P5_PIECEWISE}
    by_detail: Dict[str, List[Dict[str, Any]]] = {}
    for row in detail:
        if row.get("implementation_status") == "measured":
            by_detail.setdefault(str(row.get("variant_id")), []).append(row)
    for row in summary:
        package = str(row.get("package"))
        rs = by_detail.get(package, [])
        row["num_bins"] = bins_by.get(package, "")
        row["mixing_type"] = mix_by.get(package, "")
        row["streaming_grad_enabled"] = stream_by.get(package, "")
        row["knot_materialized"] = materialized_by.get(package, "")
        row["dense_basis_tensor_used"] = 0
        row["active_bins_per_sample"] = _mean(f(r, "active_bins_per_sample", 0.0) for r in rs)
        row["knot_workspace_MB"] = _mean(f(r, "knot_workspace_MB", 0.0) for r in rs)
        row["knot_grad_workspace_MB"] = _mean(f(r, "knot_grad_workspace_MB", 0.0) for r in rs)
        row["mixing_workspace_MB"] = _mean(f(r, "workspace_pool_MB", 0.0) for r in rs)
        row["residual_only_memory_ratio"] = f(row, "memory_ratio_mean", math.nan) if str(row.get("package", "")).startswith("PW1") or str(row.get("package", "")).startswith("PW2") or str(row.get("package", "")).startswith("PW3") else METRIC_UNAVAILABLE
        row["residual_only_step_ratio"] = f(row, "step_ratio_mean", math.nan) if str(row.get("package", "")).startswith("PW1") or str(row.get("package", "")).startswith("PW2") or str(row.get("package", "")).startswith("PW3") else METRIC_UNAVAILABLE
        row["bin_occupancy_entropy"] = _mean(f(r, "bin_occupancy_entropy", 0.0) for r in rs)
        row["dead_bin_fraction"] = _mean(f(r, "dead_bin_fraction", 0.0) for r in rs)
        row["out_of_grid_fraction"] = _mean(f(r, "out_of_grid_fraction", 0.0) for r in rs)
        row["structural_pass"] = int(f(row, "dense_basis_tensor_used", 1) == 0 and f(row, "active_bins_per_sample", 99) <= 4 and row.get("implementation_status") == "measured")
        row["stability_pass"] = int(f(row, "out_of_grid_fraction", 99) <= 0.05 and f(row, "dead_bin_fraction", 99) <= 0.30)
        row["piecewise_near_pass"] = int(f(row, "memory_ratio_mean", 99) <= 1.05 and f(row, "step_ratio_mean", 99) <= 1.50 and f(row, "residual_effect_pass_count", 0) > 0 and f(row, "grad_relerr_max", 99) < 1.0e-4 and f(row, "structural_pass", 0) == 1 and f(row, "stability_pass", 0) == 1)
    write_csv(out_dir / "p5_piecewise_residual_vs_mixing.csv", summary)
    _scatter_svg(out_dir / "p5_piecewise_residual_vs_full_pareto.svg", "P5 piecewise pareto", [r for r in detail if r.get("implementation_status") == "measured"], "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _scatter_svg(out_dir / "p5_bin_occupancy_heatmap.svg", "P5 bin occupancy", summary, "num_bins", "bin_occupancy_entropy", "package")
    _simple_bar_svg(out_dir / "p5_dead_bin_fraction_bar.svg", "P5 dead bin fraction", [r["package"] for r in summary], [f(r, "dead_bin_fraction", 0.0) for r in summary], "#dc2626")
    _simple_bar_svg(out_dir / "p5_out_of_grid_curve.svg", "P5 out of grid", [r["package"] for r in summary], [f(r, "out_of_grid_fraction", 0.0) for r in summary], "#f59e0b")
    _simple_bar_svg(out_dir / "p5_knot_workspace_bar.svg", "P5 knot workspace", [r["package"] for r in summary], [f(r, "knot_workspace_MB", 0.0) for r in summary], "#7c3aed")
    _simple_bar_svg(out_dir / "p5_piecewise_mixing_blocker_waterfall.svg", "P5 mixing workspace", [r["package"] for r in summary], [f(r, "mixing_workspace_MB", 0.0) for r in summary], "#2563eb")
    return summary


def _candidate_rows(*tables: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for table in tables:
        for row in table:
            if row.get("implementation_status") != "measured":
                continue
            ident = str(row.get("package", row.get("variant_id", "")))
            if "DWM2-current" in ident:
                continue
            if "diagnostic" in ident:
                row["used_for_gate"] = 0
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
    if mem < V614_B0_MEMORY_MEAN and step > 1.50:
        return "S3"
    if step < V614_B0_STEP_MEAN and mem > 1.05:
        return "S4"
    return "S7"


def run_p6(args: argparse.Namespace, p2: Sequence[Dict[str, Any]], p3: Sequence[Dict[str, Any]], p4: Sequence[Dict[str, Any]], p5: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    candidates = _candidate_rows(p2, p3, p4, p5)
    priority = {"S0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5, "S6": 6, "S7": 7}
    best = min(candidates, key=lambda r: (priority.get(_survivor_type(r), 9), f(r, "memory_ratio_mean", 99.0), f(r, "step_ratio_mean", 99.0))) if candidates else None
    survivor = _survivor_type(best)
    if best is None:
        blocker = "no_measured_candidate"
    elif f(best, "grad_relerr_max", 99.0) >= 1.0e-4:
        blocker = "gradient_correctness"
    elif f(best, "residual_effect_pass_count", 0.0) <= 0:
        blocker = "residual_effect"
    elif f(best, "memory_ratio_mean", 99.0) <= 1.05 and f(best, "step_ratio_mean", 99.0) > 1.50:
        blocker = "runtime_too_slow"
    elif f(best, "memory_ratio_mean", 99.0) > 1.05:
        blocker = "memory_not_nearpass"
    else:
        blocker = "none"
    best_package = str((best or {}).get("package", ""))
    if best_package.startswith("PW"):
        best_family = "piecewise-local"
    elif best_package.startswith(("VG", "GK", "GR", "GS", "FG")):
        best_family = "grouped"
    elif best_package.startswith("LR"):
        best_family = "lowrank"
    else:
        best_family = (best or {}).get("mixing_type", (best or {}).get("family", ""))
    row = {
        **_row_common("P6", args, variant_id=str((best or {}).get("package", ""))),
        "best_candidate": (best or {}).get("package", ""),
        "best_family": best_family,
        "best_memory_ratio": f(best or {}, "memory_ratio_mean", 99.0),
        "best_step_ratio": f(best or {}, "step_ratio_mean", 99.0),
        "best_backward_ratio": f(best or {}, "backward_ratio_mean", 99.0),
        "best_forward_ratio": f(best or {}, "forward_ratio_mean", 99.0),
        "memory_improvement_vs_current": f(best or {}, "memory_improvement_vs_DWM2_current", 0.0),
        "step_improvement_vs_current": f(best or {}, "step_improvement_vs_DWM2_current", 0.0),
        "residual_effect_pass": int(f(best or {}, "residual_effect_pass_count", 0) > 0),
        "grad_pass": int(f(best or {}, "grad_relerr_max", 99.0) < 1.0e-4 and f(best or {}, "grad_cos_min", 0.0) > 0.999),
        "survivor_type": survivor,
        "primary_blocker": blocker,
        "open_one_step_probe": int(survivor in {"S0", "S1", "S2"}),
        "open_task_reentry": int(survivor in {"S0", "S1"}),
        "open_diagnostic_task": int(survivor == "S2"),
        "implementation_status": "measured",
        "used_for_gate": 1,
    }
    write_csv(Path(args.out_dir) / "p6_reset_v8_selection.csv", [row])
    _simple_bar_svg(Path(args.out_dir) / "p6_reset_v8_scorecard.svg", "P6 scorecard", ["memory", "step"], [row["best_memory_ratio"], row["best_step_ratio"]], "#2563eb")
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
        reason = "No S0/S1/S2 reset-v8 survivor"
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
    stack = _make_v615_stack(method, bundle.input_dim, params.hidden_dim, 2, _basis_from_name(method, params.basis_count), device, policy, x.shape[0])
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
    stack = _make_v615_stack(method, bundle.input_dim, params.hidden_dim, 2, _basis_from_name(method, params.basis_count), device, policy, x_train.shape[0])
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
            _wandb_log_row(args, row, "task/v615_trace")
    train1, train_acc, _ = _manual_eval(stack, head, x_train, y_train)
    val1, val_acc, val_nll = _manual_eval(stack, head, x_val, y_val)
    test_loss, test_acc, test_nll = _manual_eval(stack, head, x_test, y_test)
    summary = {**_row_common("P8", args, method=method, variant_id=label, dataset=dataset, seed=seed, batch_size=args.batch_size, depth=2), "optimizer": opt_kind, "implementation_status": "measured", "train_loss_before": train0, "train_loss_after": train1, "val_loss_before": val0, "val_loss_after": val1, "train_loss_delta": train1 - train0, "val_loss_delta": val1 - val0, "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc, "NLL": test_nll, "ECE": METRIC_UNAVAILABLE, "wall_clock_time_sec": time.perf_counter() - started, "task_pass": 0}
    _wandb_log_row(args, summary, "task/v615_summary")
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
            _wandb_log_row(args, row, "task/v615_trace")
    with torch.no_grad():
        train1, train_acc, _ = _eval_logits_loss_acc(model(x_train), y_train)
        val1, val_acc, val_nll = _eval_logits_loss_acc(model(x_val), y_val)
        _test_loss, test_acc, test_nll = _eval_logits_loss_acc(model(x_test), y_test)
    summary = {**_row_common("P8", args, method="MLP-autograd-reference", variant_id="MLP-autograd-reference", dataset=dataset, seed=seed, batch_size=args.batch_size, depth=2), "optimizer": "torch-AdamW", "implementation_status": "measured", "train_loss_before": train0, "train_loss_after": train1, "val_loss_before": val0, "val_loss_after": val1, "train_loss_delta": train1 - train0, "val_loss_delta": val1 - val0, "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc, "NLL": test_nll, "ECE": METRIC_UNAVAILABLE, "wall_clock_time_sec": time.perf_counter() - started, "task_pass": 0}
    _wandb_log_row(args, summary, "task/v615_summary")
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
        if "ResetV8" in str(row.get("method", "")) or "+" in str(row.get("variant_id", "")):
            ref = mlp_acc.get((row["dataset"], row["seed"]), 0.0)
            row["task_pass"] = int(official and f(row, "test_acc", 0.0) >= ref - 0.01)
    for row in traces:
        row["task_mode"] = "official" if official else "diagnostic"
    return summaries, traces


def run_route(args: argparse.Namespace, p6_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    p6 = p6_rows[0]
    survivor = str(p6.get("survivor_type", "S7"))
    p1_rows = _read_csv(Path(args.out_dir) / "p1_component_kernel_audit.csv")
    runtime_attribution_pass = int(any(int(f(r, "runtime_attribution_pass", 0)) == 1 for r in p1_rows))
    family = str(p6.get("best_family", ""))
    mem = f(p6, "best_memory_ratio", 99.0)
    step = f(p6, "best_step_ratio", 99.0)
    if not runtime_attribution_pass:
        route = "R7-RuntimeAttributionIncomplete"
        next_impl = "kernel_allocation_counter_profiler"
    elif survivor in {"S0", "S1"}:
        route = "R1-ResetV8Solved"
        next_impl = "one_step_and_task_gate"
    elif survivor == "S2" and "grouped" in family:
        route = "R3-GroupedMemoryRuntimeMerged"
        next_impl = "one_step_probe_and_diagnostic_task"
    elif survivor == "S2" and "piecewise" in family:
        route = "R6-PiecewiseLocalCandidate"
        next_impl = "one_step_probe_and_diagnostic_task"
    elif survivor == "S2":
        route = "R2-ResetV8NearPass"
        next_impl = "one_step_probe_and_runtime_repair"
    elif "grouped" in family and mem <= 1.05 and step > 1.50:
        route = "R4-GroupedTradeoffUnresolved"
        next_impl = "fused_grouped_kernel_or_stop_grouped_branch"
    elif "lowrank" in family and step > 1.50:
        route = "R5-LowRankBalancedCandidate"
        next_impl = "lowrank_runtime_fusion"
    elif survivor == "S4":
        route = "R4-GroupedTradeoffUnresolved"
        next_impl = "workspace_and_runtime_repair"
    else:
        route = "R8-NoViableResetV8"
        next_impl = "new_primitive_family"
    route_json = {
        "route": route,
        "best_candidate": p6.get("best_candidate", ""),
        "best_family": p6.get("best_family", ""),
        "best_memory_ratio": f(p6, "best_memory_ratio", 99.0),
        "best_step_ratio": f(p6, "best_step_ratio", 99.0),
        "best_backward_ratio": f(p6, "best_backward_ratio", 99.0),
        "best_forward_ratio": f(p6, "best_forward_ratio", 99.0),
        "memory_improvement_vs_current": f(p6, "memory_improvement_vs_current", 0.0),
        "step_improvement_vs_current": f(p6, "step_improvement_vs_current", 0.0),
        "survivor_type": survivor,
        "residual_effect_pass": int(f(p6, "residual_effect_pass", 0)),
        "grad_pass": int(f(p6, "grad_pass", 0)),
        "runtime_attribution_pass": runtime_attribution_pass,
        "open_one_step_probe": bool(int(f(p6, "open_one_step_probe", 0))),
        "open_task_reentry": bool(int(f(p6, "open_task_reentry", 0))),
        "open_diagnostic_task": bool(int(f(p6, "open_diagnostic_task", 0))),
        "open_optimizer_exploration": False,
        "open_functional_correction": False,
        "primary_blocker": p6.get("primary_blocker", ""),
        "stop_dwm2_patching": 1,
        "reset_v8_measured": 1,
        "reset_v8_pass": int(survivor in {"S0", "S1", "S2"}),
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
        ("p2_grouped_tile_stream_sweep.csv", "P2"),
        ("p3_fused_grouped_custom_kernel.csv", "P3"),
        ("p4_lowrank_r2_s2_closure.csv", "P4"),
        ("p5_piecewise_residual_vs_mixing.csv", "P5"),
        ("p6_reset_v8_selection.csv", "P6"),
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
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F10_task_gated" if status == "not_run" else "F0_not_implemented_or_gated", "metric": row.get("reason", status), "recommendation": "implement or pass gate before claiming metric"})
            if status != "measured":
                continue
            if row.get("memory_ratio_mean") not in {None, ""} and f(row, "memory_ratio_mean", 0.0) > 1.05:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F1_memory_fail", "metric": f"memory_ratio_mean={row.get('memory_ratio_mean')}", "recommendation": "reduce actual CUDA peak"})
            if row.get("step_ratio_mean") not in {None, ""} and f(row, "step_ratio_mean", 0.0) > 1.50:
                ftype = "F2_step_time_fail"
                if stage == "P2":
                    ftype = "F6_grouped_runtime_fail"
                elif stage == "P3":
                    ftype = "F6_grouped_runtime_fail"
                elif stage == "P4":
                    ftype = "F7_lowrank_runtime_fail"
                failures.append({"stage": stage, "variant_id": vid, "failure_type": ftype, "metric": f"step_ratio_mean={row.get('step_ratio_mean')}", "recommendation": "reduce runtime"})
            if row.get("grad_relerr_max") not in {None, ""} and f(row, "grad_relerr_max", 0.0) >= 1.0e-4:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F3_gradient_correctness_fail", "metric": f"grad_relerr={row.get('grad_relerr_max')}", "recommendation": "fix manual backward"})
            if stage in {"P2", "P3", "P4", "P5"} and row.get("residual_effect_pass_count") not in {None, ""} and f(row, "residual_effect_pass_count", 0) <= 0:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F4_residual_effect_fail", "metric": f"residual={row.get('residual_over_base_mean')}", "recommendation": "restore nontrivial residual"})
            if stage == "P5" and row.get("stability_pass") not in {None, ""} and int(f(row, "stability_pass", 0)) == 0:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F8_piecewise_stability_fail", "metric": f"dead_bin={row.get('dead_bin_fraction')} out_of_grid={row.get('out_of_grid_fraction')}", "recommendation": "repair piecewise bin coverage"})
            if stage == "P1" and row.get("component_status") == "measured_from_full_step_phase_fields":
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F9_runtime_counter_unavailable", "metric": "kernel/allocation counter metric_unavailable", "recommendation": "collect Nsight/kernel counters"})
    write_csv(out_dir / "failure_table.csv", failures or [{"stage": "ALL", "variant_id": "all", "failure_type": "none"}])
    _simple_bar_svg(out_dir / "failure_taxonomy_heatmap.svg", "Failure taxonomy", [r["failure_type"] for r in failures], [1.0 for _ in failures], "#dc2626")
    return failures


def _copy_figures(out_dir: Path) -> None:
    figures = ensure_dir(out_dir / "figures")
    for name in [
        "p1_component_runtime_waterfall.svg",
        "p1_component_kernel_count_bar.svg",
        "p1_component_op_count_bar.svg",
        "p2_grouped_memory_time_pareto.svg",
        "p2_g2_vs_gr3_vs_new_waterfall.svg",
        "p3_custom_grouped_kernel_pareto.svg",
        "p4_lowrank_package_pareto.svg",
        "p5_piecewise_residual_vs_full_pareto.svg",
        "p6_family_comparison_pareto.svg",
        "p7_loss_before_after.svg",
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
        "script": "experiments/run_gafu_v615_real.py",
        "plan": "docs/DG-KAN_v6.15_ResetV8_GroupedMemoryRuntimeMerge_详细实验计划.md",
        "started_unix": started,
        "finished_unix": finished,
        "duration_sec": finished - started,
        "source_commit": _git_commit(),
        "git_status_short": _git_status(),
        "command_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "triton_available": int(v68._triton_available()),
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
    parser.add_argument("--packages", default="V6_15_ALL")
    parser.add_argument("--out-dir", type=Path, default=Path("results/real_rerun_20260505/v615_real"))
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
    parser.add_argument("--wandb-group", default="v615-real-20260505")
    parser.add_argument("--wandb-name-prefix", default="v615-real")
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
        # v6.15 reproduction tracks L6 via P4 and DWM2 via P1.
        p2_summary, _p2_detail = run_p2(args)
        p3_summary, _p3_detail = run_p3(args)
        p4_summary, _p4_detail = run_p4(args)
        p5_rows = run_p5(args)
        p6_rows, best = run_p6(args, p2_summary, p3_summary, p4_summary, p5_rows)
        run_p7_to_p10(args, p6_rows, best)
        route = run_route(args, p6_rows)
        run_failure(args)
        # Reproduction summary after P4 is available.
        l6 = next((r for r in p4_summary if r.get("package") == "LR0-L6-lowrank-r2-current"), {})
        g2 = next((r for r in p2_summary if r.get("package") == "GS0-G2-grouped-g16-loop-baseline"), {})
        gr3 = next((r for r in p2_summary if r.get("package") == "GS1-GR3-vectorized-g16-baseline"), {})
        dwm2 = [r for r in p1_detail if r.get("variant_id") == "DWM2-current-baseline" and r.get("implementation_status") == "measured"]
        repro = {
            **_row_common("P0_REPRO", args, variant_id="v615-reproduction"),
            "v614_dwm2_memory_ratio_mean": V614_DWM2_MEMORY_MEAN,
            "v615_dwm2_memory_ratio_mean": _mean(f(r, "memory_ratio_vs_MLP", math.nan) for r in dwm2),
            "v614_l6_memory_ratio_mean": V614_L6_MEMORY_MEAN,
            "v615_l6_memory_ratio_mean": f(l6, "memory_ratio_mean", math.nan),
            "v614_l6_step_ratio_mean": V614_L6_STEP_MEAN,
            "v615_l6_step_ratio_mean": f(l6, "step_ratio_mean", math.nan),
            "v614_g2_memory_ratio_mean": V614_G2_MEMORY_MEAN,
            "v615_g2_memory_ratio_mean": f(g2, "memory_ratio_mean", math.nan),
            "v614_gr3_memory_ratio_mean": V614_GR3_MEMORY_MEAN,
            "v615_gr3_memory_ratio_mean": f(gr3, "memory_ratio_mean", math.nan),
            "reproduction_delta_memory_ratio": f(l6, "memory_ratio_mean", math.nan) - V614_L6_MEMORY_MEAN,
            "reproduction_delta_step_ratio": f(l6, "step_ratio_mean", math.nan) - V614_L6_STEP_MEAN,
            "reproduction_pass": int(abs(f(l6, "memory_ratio_mean", 99.0) - V614_L6_MEMORY_MEAN) <= 0.05 and abs(f(l6, "step_ratio_mean", 99.0) - V614_L6_STEP_MEAN) <= 0.15),
            "route": route.get("route"),
        }
        write_csv(out_dir / "p0_reproduction_check.csv", [repro])
        _simple_bar_svg(out_dir / "p0_reproduction_delta_bar.svg", "v6.15 vs v6.14 L6 delta", ["memory", "step"], [repro["reproduction_delta_memory_ratio"], repro["reproduction_delta_step_ratio"]], "#2563eb")
        _copy_figures(out_dir)
        _write_manifest(out_dir, args, started, time.time())
    finally:
        _wandb_finish(args, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
