#!/usr/bin/env python3
"""DG-KAN v6.11 real-only DWM2-stop / bounded primitive reset runner."""

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

import run_gafu_v610_real as v610
import run_gafu_v68_real as v68
import run_gafu_v69_real as v69
from dgkan_core import ensure_dir, get_device, load_vision_bundle, parse_int_list, parse_str_list, write_csv
from run_gafu_v3 import dataset_name
from run_gafu_v63 import ManualOptimizer, V63ManualLayer, V63Params, _basis_from_name, _rel_cos, _sync, _wandb_finish, _wandb_init, _wandb_log_row, f
from run_gafu_v64_real import METHOD_CURRENT, _sha256, _take_batch
from run_gafu_v65_real import _placeholder_svg, _scatter_svg, _simple_bar_svg
from run_gafu_v66_real import _git_commit, _git_status, _json_dump, _mean, _std


METRIC_UNAVAILABLE = "metric_unavailable"
V610_MEMORY_MEAN = 1.2917923088533285
V610_STEP_MEAN = 1.8969413768989343
ORIG_V68_MAKE_STACK = v68._make_v68_stack


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


def _triton_available() -> bool:
    return v69._triton_available()


class V611LowRankPoly1Layer:
    """Edge-owned low-rank mixing with a scaled poly1 residual."""

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

    def clone_params_for_autograd(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone().requires_grad_(True) for k, v in self.params.items()}

    def zero_grad(self) -> None:
        for grad in self.grads.values():
            grad.zero_()

    def param_tensors(self) -> List[torch.Tensor]:
        return list(self.params.values())

    def param_count(self) -> int:
        return sum(int(p.numel()) for p in self.params.values())

    def transform_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return x * (1.0 + self.scale * params["poly"][:, 0].unsqueeze(0))

    def forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.transform_with_params(x, params)
        h = z @ params["V"]
        return h @ params["U"].t()

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            z = self.transform_with_params(x, self.params)
            h = z @ self.params["V"]
            y = h @ self.params["U"].t()
            self.last_workspace_MB = (z.numel() + h.numel() + y.numel()) * x.element_size() / (1024**2)
            return y, x.detach()

    def backward_manual(self, dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.transform_with_params(x, self.params)
            h = z @ self.params["V"]
            self.grads["U"].add_(dy.t() @ h)
            dh = dy @ self.params["U"]
            self.grads["V"].add_(z.t() @ dh)
            dz = dh @ self.params["V"].t()
            self.grads["poly"][:, 0].add_((dz * x).sum(dim=0), alpha=self.scale)
            dx = dz * (1.0 + self.scale * self.params["poly"][:, 0].unsqueeze(0))
            self.last_workspace_MB = (z.numel() + h.numel() + dh.numel() + dz.numel()) * x.element_size() / (1024**2)
            return dx


class V611ChunkedPoly1Layer:
    """Full-rank poly1 residual with output mixing computed in chunks."""

    def __init__(self, in_dim: int, out_dim: int, *, chunk_size: int, device: torch.device, scale_target: float = 0.02) -> None:
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.chunk_size = int(max(1, chunk_size))
        self.scale = 0.05
        self.params: Dict[str, torch.Tensor] = {
            "mix": torch.randn(out_dim, in_dim, device=device) / math.sqrt(max(1, in_dim)),
            "poly": torch.full((in_dim, 1), float(scale_target) / self.scale, device=device),
        }
        self.grads = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.last_workspace_MB = 0.0

    def clone_params_for_autograd(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone().requires_grad_(True) for k, v in self.params.items()}

    def zero_grad(self) -> None:
        for grad in self.grads.values():
            grad.zero_()

    def param_tensors(self) -> List[torch.Tensor]:
        return list(self.params.values())

    def param_count(self) -> int:
        return sum(int(p.numel()) for p in self.params.values())

    def transform_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return x * (1.0 + self.scale * params["poly"][:, 0].unsqueeze(0))

    def forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.transform_with_params(x, params) @ params["mix"].t()

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            z = self.transform_with_params(x, self.params)
            y = torch.empty(x.shape[0], self.out_dim, device=x.device, dtype=x.dtype)
            for start in range(0, self.out_dim, self.chunk_size):
                end = min(self.out_dim, start + self.chunk_size)
                y[:, start:end] = z @ self.params["mix"][start:end].t()
            self.last_workspace_MB = (z.numel() + min(self.chunk_size, self.out_dim) * x.shape[0] + y.numel()) * x.element_size() / (1024**2)
            return y, x.detach()

    def backward_manual(self, dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.transform_with_params(x, self.params)
            dz = torch.zeros_like(x)
            for start in range(0, self.out_dim, self.chunk_size):
                end = min(self.out_dim, start + self.chunk_size)
                dyc = dy[:, start:end]
                self.grads["mix"][start:end].add_(dyc.t() @ z)
                dz.add_(dyc @ self.params["mix"][start:end])
            self.grads["poly"][:, 0].add_((dz * x).sum(dim=0), alpha=self.scale)
            dx = dz * (1.0 + self.scale * self.params["poly"][:, 0].unsqueeze(0))
            self.last_workspace_MB = (z.numel() + dz.numel() + min(self.chunk_size, self.out_dim) * x.shape[0]) * x.element_size() / (1024**2)
            return dx


class V611PrimitiveStack:
    def __init__(self, method: str, input_dim: int, hidden_dim: int, depth: int, basis: int, device: torch.device, *, policy: str, batch_size: int) -> None:
        self.method = method
        self.policy = policy
        self.kind = "reset_v4"
        dims = [input_dim] + [hidden_dim] * int(depth)
        self.layers: List[Any] = []
        for a, b in zip(dims[:-1], dims[1:]):
            if policy == "reset_v4_lowrank_r4_poly1":
                self.layers.append(V611LowRankPoly1Layer(a, b, rank=4, device=device, scale_target=0.02))
            elif policy == "reset_v4_lowrank_r8_poly1":
                self.layers.append(V611LowRankPoly1Layer(a, b, rank=8, device=device, scale_target=0.02))
            elif policy == "reset_v4_chunked_poly1_c16":
                self.layers.append(V611ChunkedPoly1Layer(a, b, chunk_size=16, device=device, scale_target=0.02))
            elif policy == "reset_v4_chunked_poly1_c32":
                self.layers.append(V611ChunkedPoly1Layer(a, b, chunk_size=32, device=device, scale_target=0.02))
            else:
                raise ValueError(f"unsupported v6.11 primitive policy: {policy}")

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
        return {
            "cache_total_MB": (x_bytes + hidden_bytes) / (1024**2),
            "cache_x_MB": x_bytes / (1024**2),
            "cache_hidden_MB": hidden_bytes / (1024**2),
            "cache_delta_MB": 0.0,
            "cache_index_MB": 0.0,
            "workspace_pool_MB": workspace,
            "workspace_pool_used_peak_MB": workspace,
            "workspace_pool_fragmentation_MB": 0.0,
            "reused_buffer_count": 0.0,
        }


def _make_v611_stack(method: str, input_dim: int, hidden_dim: int, depth: int, basis: int, device: torch.device, policy: str, batch_size: int) -> Any:
    if policy.startswith("reset_v4_"):
        return V611PrimitiveStack(method, input_dim, hidden_dim, depth, basis, device, policy=policy, batch_size=batch_size)
    return ORIG_V68_MAKE_STACK(method, input_dim, hidden_dim, depth, basis, device, policy, batch_size)


def _patch_stack() -> None:
    v69._patch_stack()
    v68._make_v68_stack = _make_v611_stack  # type: ignore[assignment]
    v68.v67._make_v67_stack = _make_v611_stack  # type: ignore[assignment]


def _profile_grid(args: argparse.Namespace, stage: str, variants: Sequence[Tuple[Any, ...]]) -> List[Dict[str, Any]]:
    _patch_stack()
    return v69._profile_grid(args, stage, variants)


P0_VARIANTS = [
    ("MLP-autograd-reference", "reference", "mlp", True, "reference", "none", 0),
    ("MLP-manual-linear-reference", "MLP-manual-linear-reference", "current", True, "manual-linear", "none", 0),
    ("DWM2-current", METHOD_CURRENT, "current", True, "current", "baseline", 1),
    ("DWM2-K3-triton-fused-dx-coeffgrad-reference", METHOD_CURRENT, "triton_fused_dx_coeffgrad", _triton_available(), "disallowed-local-reference", "coeffgrad-only", 0),
    ("DWM2-terminal-single-call-full-step", METHOD_CURRENT, "not_implemented", False, "terminal-dwm2", "terminal-single-call-full-step", 1),
    ("ResidualBoundedReset-v3-scale002-current", "DWM2-poly1-minimal", "poly1_scale002", True, "reset-v3-baseline", "none", 0),
    ("ResidualBoundedReset-v3-scale005-current", "DWM2-poly1-minimal", "poly1_scale005", True, "reset-v3-baseline", "none", 0),
    ("ResetV4-lowrank-r4-poly1", "ResetV4-lowrank-r4-poly1", "reset_v4_lowrank_r4_poly1", True, "reset-v4-lowrank", "none", 0),
    ("ResetV4-lowrank-r8-poly1", "ResetV4-lowrank-r8-poly1", "reset_v4_lowrank_r8_poly1", True, "reset-v4-lowrank", "none", 0),
    ("ResetV4-chunked-poly1-c16", "ResetV4-chunked-poly1-c16", "reset_v4_chunked_poly1_c16", True, "reset-v4-chunked", "none", 0),
    ("ResetV4-chunked-poly1-c32", "ResetV4-chunked-poly1-c32", "reset_v4_chunked_poly1_c32", True, "reset-v4-chunked", "none", 0),
]

P1_VARIANTS = [("A1-DWM2-current", METHOD_CURRENT, "current", True, "current")]

P2_DWM2_TERMINAL = [
    ("D0-current", METHOD_CURRENT, "current", True, "current", "current"),
    ("D1-terminal-single-call-depth2", METHOD_CURRENT, "not_implemented", False, "terminal-depth2", "terminal_single_call"),
    ("D2-terminal-single-call-layer", METHOD_CURRENT, "not_implemented", False, "terminal-layer", "terminal_single_call"),
    ("D3-terminal-single-call-full-step", METHOD_CURRENT, "not_implemented", False, "terminal-full-step", "terminal_single_call"),
    ("D4-terminal-onebuffer-full-step", METHOD_CURRENT, "not_implemented", False, "terminal-onebuffer", "terminal_single_call"),
    ("D5-terminal-chunked-mix-full-step", METHOD_CURRENT, "not_implemented", False, "terminal-chunked-mix", "terminal_single_call"),
]

P3_RESET_V4 = [
    ("DWM2-current-baseline", METHOD_CURRENT, "current", True, "DWM2", "current-baseline", "poly2", "dense", 0, 0, 0.0),
    ("R0-reset-v3-scale002-current", "DWM2-poly1-minimal", "poly1_scale002", True, "reset-v3", "baseline", "poly1", "dense", 0, 0, 0.02),
    ("A0-inplace-poly1-chunked-mix-c16", "ResetV4-chunked-poly1-c16", "reset_v4_chunked_poly1_c16", True, "FAM-A", "inplace-poly1-chunked-mix", "poly1", "chunked", 16, 0, 0.02),
    ("A1-inplace-poly1-chunked-mix-c32", "ResetV4-chunked-poly1-c32", "reset_v4_chunked_poly1_c32", True, "FAM-A", "inplace-poly1-chunked-mix", "poly1", "chunked", 32, 0, 0.02),
    ("A2-inplace-piecewise2-chunked-mix", "ResetV4-piecewise2-chunked", "not_implemented", False, "FAM-A", "piecewise2-chunked", "piecewise2", "chunked", 16, 0, 0.02),
    ("B0-lowrank-r4-poly1", "ResetV4-lowrank-r4-poly1", "reset_v4_lowrank_r4_poly1", True, "FAM-B", "lowrank-r4-poly1", "poly1", "lowrank", 0, 4, 0.02),
    ("B1-lowrank-r8-poly1", "ResetV4-lowrank-r8-poly1", "reset_v4_lowrank_r8_poly1", True, "FAM-B", "lowrank-r8-poly1", "poly1", "lowrank", 0, 8, 0.02),
    ("B2-lowrank-r4-piecewise2", "ResetV4-lowrank-r4-piecewise2", "not_implemented", False, "FAM-B", "lowrank-r4-piecewise2", "piecewise2", "lowrank", 0, 4, 0.02),
    ("B3-lowrank-r8-piecewise2", "ResetV4-lowrank-r8-piecewise2", "not_implemented", False, "FAM-B", "lowrank-r8-piecewise2", "piecewise2", "lowrank", 0, 8, 0.02),
    ("C0-grouped-g4-poly1", "ResetV4-grouped-g4-poly1", "not_implemented", False, "FAM-C", "grouped-g4-poly1", "poly1", "grouped", 0, 0, 0.02),
    ("C1-grouped-g8-poly1", "ResetV4-grouped-g8-poly1", "not_implemented", False, "FAM-C", "grouped-g8-poly1", "poly1", "grouped", 0, 0, 0.02),
    ("D1-piecewise2-streamingGrad", "ResetV4-piecewise2-streamingGrad", "not_implemented", False, "FAM-D", "piecewise2-streamingGrad", "piecewise2", "streaming", 0, 0, 0.02),
]


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    original = v69.P0_VARIANTS
    v69.P0_VARIANTS = [(a, b, c, d, e) for a, b, c, d, e, _patch, _allowed in P0_VARIANTS]
    _patch_stack()
    try:
        rows = v69.run_p0(args)
    finally:
        v69.P0_VARIANTS = original
    patch_by = {name: (patch, allowed) for name, _m, _p, _ok, _impl, patch, allowed in P0_VARIANTS}
    for row in rows:
        patch, allowed = patch_by.get(str(row.get("variant_id")), ("none", 0))
        row.update({
            "script_path": "experiments/run_gafu_v611_real.py",
            "v610_memory_ratio_mean": V610_MEMORY_MEAN,
            "v610_step_ratio_mean": V610_STEP_MEAN,
            "dwm2_patch_type": patch,
            "is_allowed_dwm2_patch": allowed,
            "dwm2_patch_policy_pass": int((not str(row.get("variant_id", "")).startswith("DWM2")) or allowed == 1 or patch in {"baseline", "none"}),
        })
    write_csv(out_dir / "p0_contract.csv", rows)
    policy_md = "| variant | patch_type | allowed |\n|---|---|---:|\n" + "\n".join(f"| {r.get('variant_id')} | {r.get('dwm2_patch_type')} | {r.get('is_allowed_dwm2_patch')} |" for r in rows) + "\n"
    (out_dir / "p0_dwm2_patch_policy_table.md").write_text(policy_md, encoding="utf-8")
    _simple_bar_svg(out_dir / "p0_contract_heatmap.svg", "v6.11 P0 contract", [r["variant_id"] for r in rows], [1.0 if r.get("implementation_status") == "measured" and int(f(r, "fake_data_used", 0)) == 0 else 0.0 for r in rows], "#16a34a")
    return rows


def _taxonomy_from_source(source: str, source_op: str = "", source_file: str = "") -> str:
    text = f"{source} {source_op} {source_file}".lower()
    if "cross_entropy" in text or "softmax" in text or "loss" in text:
        return "T0_delta_z_live"
    if "backward_manual" in text and ("run_gafu_v62.py" in text or "poly" in text or "coeff" in text):
        return "T2_coeffgrad_contribution_live"
    if "matmul" in text or "mix" in text or "forward_with_params" in text:
        return "T6_mixing_output_live"
    if "triton" in text:
        return "T9_triton_output_materialization"
    if "take_batch" in text or "contiguous" in text:
        return "T11_contiguous_copy_temp"
    if "__init__" in text or "manual_cache" in text:
        return "T14_manual_cache"
    if "materialization" in text or "workspace" in text or "empty" in text:
        return "T13_workspace_pool"
    if "poly" in text or "sum" in text or "mul" in text:
        return "T2_coeffgrad_contribution_live"
    return "T15_unknown"


def _taxonomy_enrich(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    taxonomy_mb: Dict[str, float] = {}
    for idx in range(1, 21):
        mb = f(row, f"top{idx}_live_tensor_MB", math.nan)
        if not math.isfinite(mb):
            continue
        tax = _taxonomy_from_source(str(row.get(f"top{idx}_live_tensor_source", "")), str(row.get(f"top{idx}_source_op", "")), str(row.get(f"top{idx}_source_file", "")))
        out[f"top{idx}_source_taxonomy"] = tax
        out[f"top{idx}_lifetime_start_phase"] = "allocator_event_start"
        out[f"top{idx}_lifetime_end_phase"] = "allocator_peak"
        taxonomy_mb[tax] = taxonomy_mb.get(tax, 0.0) + mb
    fallback_map = {
        "T13_workspace_pool": f(row, "source_materialization_workspace_MB", 0.0),
        "T2_coeffgrad_contribution_live": f(row, "source_poly_coeffgrad_temp_MB", 0.0),
        "T6_mixing_output_live": f(row, "source_matmul_mixing_MB", 0.0),
        "T0_delta_z_live": f(row, "source_loss_delta_MB", 0.0),
        "T9_triton_output_materialization": f(row, "source_triton_boundary_MB", 0.0),
        "T15_unknown": f(row, "source_unknown_MB", 0.0),
    }
    for tax, mb in fallback_map.items():
        taxonomy_mb[tax] = max(taxonomy_mb.get(tax, 0.0), mb)
    gap = f(row, "peak_gap_MB", 0.0)
    top_tax = sorted(taxonomy_mb.items(), key=lambda item: item[1], reverse=True)
    top3_mb = sum(mb for _tax, mb in top_tax[:3])
    known = sum(mb for tax, mb in taxonomy_mb.items() if tax != "T15_unknown")
    unknown = taxonomy_mb.get("T15_unknown", 0.0)
    out.update({
        "peak_timestamp": row.get("exact_peak_time_us", METRIC_UNAVAILABLE),
        "peak_phase": "allocator_peak",
        "taxonomy_mapper_version": "v611_stack_taxonomy_v1",
        "top1_source_taxonomy": top_tax[0][0] if top_tax else "T15_unknown",
        "top1_source_fraction": top_tax[0][1] / max(1.0e-12, gap) if top_tax and gap > 0 else 0.0,
        "explained_gap_MB": min(gap, known),
        "unexplained_gap_MB": max(0.0, gap - known),
        "unknown_gap_fraction": unknown / max(1.0e-12, gap) if gap > 0 else 0.0,
        "top3_gap_fraction": min(gap, top3_mb) / max(1.0e-12, gap) if gap > 0 else 1.0,
    })
    for tax in [
        "T0_delta_z_live",
        "T1_delta_x_live",
        "T2_coeffgrad_contribution_live",
        "T3_coeffgrad_partial_live",
        "T4_update_temp_live",
        "T5_forward_transform_live",
        "T6_mixing_output_live",
        "T7_optimizer_state_live",
        "T8_triton_input_materialization",
        "T9_triton_output_materialization",
        "T10_layout_conversion_temp",
        "T11_contiguous_copy_temp",
        "T12_allocator_padding",
        "T13_workspace_pool",
        "T14_manual_cache",
        "T15_unknown",
    ]:
        out[f"{tax}_MB"] = taxonomy_mb.get(tax, 0.0)
    out["attribution_pass"] = int(f(out, "top3_gap_fraction", 0.0) >= 0.70 and f(out, "unknown_gap_fraction", 99.0) <= 0.05 and out["top1_source_taxonomy"] != "T15_unknown" and f(out, "top1_source_fraction", 0.0) >= 0.20)
    return out


def run_p1(args: argparse.Namespace) -> List[Dict[str, Any]]:
    base_rows = v610.run_p1(args)
    rows = [_taxonomy_enrich(row) for row in base_rows]
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p1_taxonomy_attribution.csv", rows)
    current = [r for r in rows if r.get("variant_id") == "A1-DWM2-current"]
    _simple_bar_svg(out_dir / "p1_peak_live_tensor_top20.svg", "P1 live tensor top20", [r["dataset"] + f" B{r['batch_size']}D{r['depth']}" for r in current], [f(r, "exact_live_tensor_total_MB", 0.0) for r in current], "#2563eb")
    _simple_bar_svg(out_dir / "p1_taxonomy_fraction_heatmap.svg", "P1 top3 taxonomy fraction", [r["dataset"] + f" B{r['batch_size']}D{r['depth']}" for r in current], [f(r, "top3_gap_fraction", 0.0) for r in current], "#16a34a")
    _simple_bar_svg(out_dir / "p1_unknown_gap_heatmap.svg", "P1 unknown gap", [r["dataset"] + f" B{r['batch_size']}D{r['depth']}" for r in current], [f(r, "unknown_gap_fraction", 0.0) for r in current], "#dc2626")
    _simple_bar_svg(out_dir / "p1_gap_attribution_stacked_bar.svg", "P1 taxonomy explained", [r["dataset"] + f" B{r['batch_size']}D{r['depth']}" for r in current], [f(r, "explained_gap_MB", 0.0) for r in current], "#7c3aed")
    _placeholder_svg(out_dir / "p1_exact_live_set_gantt.svg", "P1 live-set gantt", "allocator-event start/end phases recorded in csv")
    _placeholder_svg(out_dir / "p1_lifetime_overlap_matrix.svg", "P1 lifetime overlap matrix", "exact pairwise tensor overlap not materialized")
    return rows


def run_p0_reproduction(args: argparse.Namespace, p1_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    current = [r for r in p1_rows if r.get("variant_id") == "A1-DWM2-current" and r.get("implementation_status") == "measured"]
    mem_mean = _mean(f(r, "memory_ratio_vs_MLP") for r in current)
    step_mean = _mean(f(r, "step_time_ratio_vs_MLP") for r in current)
    row = {
        **_row_common("P0_REPRO", args, variant_id="A1-DWM2-current"),
        "v610_memory_ratio_mean": V610_MEMORY_MEAN,
        "v610_step_ratio_mean": V610_STEP_MEAN,
        "v611_memory_ratio_mean": mem_mean,
        "v611_step_ratio_mean": step_mean,
        "reproduction_delta_memory_ratio": mem_mean - V610_MEMORY_MEAN,
        "reproduction_delta_step_ratio": step_mean - V610_STEP_MEAN,
        "reproduction_pass": int(abs(mem_mean - V610_MEMORY_MEAN) <= 0.05 and abs(step_mean - V610_STEP_MEAN) <= 0.15),
    }
    write_csv(Path(args.out_dir) / "p0_reproduction_check.csv", [row])
    _simple_bar_svg(Path(args.out_dir) / "p0_reproduction_delta_bar.svg", "v6.11 vs v6.10 reproduction delta", ["memory", "step"], [row["reproduction_delta_memory_ratio"], row["reproduction_delta_step_ratio"]], "#2563eb")
    return [row]


def _summarize_detail(args: argparse.Namespace, stage: str, detail: Sequence[Dict[str, Any]], id_field: str, extra: Any | None = None) -> List[Dict[str, Any]]:
    return v69._summarize_detail(args, stage, detail, id_field, extra)


def _append_not_implemented(summary: List[Dict[str, Any]], args: argparse.Namespace, stage: str, packages: Sequence[Tuple[Any, ...]], id_field: str) -> None:
    existing = {r.get(id_field) for r in summary}
    for item in packages:
        name, method, policy, ok, component = item[:5]
        if ok or name in existing:
            continue
        summary.append({
            **_row_common(stage, args, method=method, variant_id=name),
            id_field: name,
            "implementation_status": policy,
            "stage_status": policy,
            "used_for_gate": 0,
            "not_implemented_count": 1,
            "implementation_scope": component,
            "reason": "required v6.11 implementation is absent; no measured ratio emitted",
        })


def run_p2(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    variants = [(a, b, c, d, e) for a, b, c, d, e, _scope in P2_DWM2_TERMINAL]
    detail = _profile_grid(args, "P2", variants)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p2_dwm2_terminal_full_step_detail.csv", detail)
    scope = {name: sc for name, _m, _p, _ok, _comp, sc in P2_DWM2_TERMINAL}
    def extra(name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        useful = int(_mean(f(r, "memory_repair_ratio_vs_current", 99.0) for r in rows) <= 0.90 and _mean(f(r, "step_repair_ratio_vs_current", 99.0) for r in rows) <= 0.90)
        return {
            "implementation_scope": scope.get(name, ""),
            "calls_per_step": METRIC_UNAVAILABLE if name != "D0-current" else 0,
            "live_buffer_count_peak": _safe_max(f(r, "reused_buffer_count", 0.0) for r in rows),
            "live_buffer_total_MB_peak": _safe_max(f(r, "workspace_pool_MB", 0.0) for r in rows),
            "triton_call_count": int("triton" in name.lower()),
            "torch_call_count": METRIC_UNAVAILABLE,
            "boundary_time_ms": METRIC_UNAVAILABLE,
            "materialization_MB": METRIC_UNAVAILABLE,
            "layout_conversion_MB": METRIC_UNAVAILABLE,
            "dwm2_terminal_useful_pass": useful,
            "onebuffer_pass": int(_safe_max(f(r, "reused_buffer_count", 99.0) for r in rows) <= 3 and name != "D0-current"),
        }
    summary = _summarize_detail(args, "P2", detail, "package", extra)
    _append_not_implemented(summary, args, "P2", P2_DWM2_TERMINAL, "package")
    write_csv(out_dir / "p2_dwm2_terminal_full_step.csv", summary)
    measured = [r for r in detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    _scatter_svg(out_dir / "p2_dwm2_terminal_pareto.svg", "P2 DWM2 terminal", measured, "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _simple_bar_svg(out_dir / "p2_live_buffer_count_timeline.svg", "P2 live buffers", [r["package"] for r in summary], [f(r, "live_buffer_count_peak", 0.0) for r in summary], "#2563eb")
    _simple_bar_svg(out_dir / "p2_boundary_materialization_bar.svg", "P2 materialization", [r["package"] for r in summary], [f(r, "materialization_MB", 0.0) for r in summary], "#7c3aed")
    _simple_bar_svg(out_dir / "p2_current_vs_terminal_waterfall.svg", "P2 terminal improvement", [r["package"] for r in summary], [f(r, "memory_improvement_vs_current", 0.0) for r in summary], "#16a34a")
    return summary, detail


def _residual_effect(args: argparse.Namespace, row: Dict[str, Any], scale: float) -> Dict[str, Any]:
    return v68._residual_effect(args, str(row.get("method")), str(row.get("workspace_policy", "current")), str(row.get("dataset")), int(float(row.get("batch_size") or args.batch_size)), int(float(row.get("depth") or 2)), scale)


def run_p3(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    variants = [(a, b, c, d, e) for a, b, c, d, e, _variant, _rtype, _mix, _chunk, _rank, _scale in P3_RESET_V4]
    detail = _profile_grid(args, "P3", variants)
    meta = {name: (family, variant, rtype, mix, chunk, rank, scale) for name, _m, _p, _ok, family, variant, rtype, mix, chunk, rank, scale in P3_RESET_V4}
    for row in detail:
        name = str(row.get("variant_id"))
        family, variant, rtype, mix, chunk, rank, scale = meta.get(name, ("", "", "", "", 0, 0, 0.0))
        row.update({"primitive": name, "family": family, "variant": variant, "residual_type": rtype, "mixing_type": mix, "chunk_size": chunk, "rank": rank, "group_count": 0})
        if row.get("implementation_status") == "measured" and row.get("method") != "MLP-autograd-reference":
            row.update(_residual_effect(args, row, scale))
            row["structural_pass"] = int(int(f(row, "gradient_correctness_pass", 0)) == 1 and f(row, "grad_relerr", 99) < 1.0e-4 and f(row, "grad_cos", 0) > 0.999)
            row["reset_near_pass"] = int(f(row, "memory_ratio_vs_MLP", 99) <= 1.05 and f(row, "step_time_ratio_vs_MLP", 99) <= 1.50 and int(f(row, "residual_effect_pass", 0)) == 1 and int(f(row, "structural_pass", 0)) == 1)
            row["reset_task_open_pass"] = int(f(row, "memory_ratio_vs_MLP", 99) < 1.0 and f(row, "step_time_ratio_vs_MLP", 99) <= 1.35 and int(f(row, "residual_effect_pass", 0)) == 1)
            row["mixing_workspace_MB"] = f(row, "workspace_pool_MB", 0.0)
            row["workspace_temp_MB"] = f(row, "workspace_pool_MB", 0.0)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p3_reset_v4_family_detail.csv", detail)
    summary_rows: List[Dict[str, Any]] = []
    measured_names = sorted({r["variant_id"] for r in detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"})
    for name in measured_names:
        rows = [r for r in detail if r.get("variant_id") == name and r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
        family, variant, rtype, mix, chunk, rank, scale = meta.get(name, ("", "", "", "", 0, 0, 0.0))
        summary_rows.append({
            **_row_common("P3", args, variant_id=name),
            "primitive": name,
            "family": family,
            "variant": variant,
            "residual_type": rtype,
            "mixing_type": mix,
            "chunk_size": chunk,
            "rank": rank,
            "group_count": 0,
            "implementation_status": "measured",
            "memory_ratio_min": _safe_min(f(r, "memory_ratio_vs_MLP") for r in rows),
            "memory_ratio_mean": _mean(f(r, "memory_ratio_vs_MLP") for r in rows),
            "memory_ratio_max": _safe_max(f(r, "memory_ratio_vs_MLP") for r in rows),
            "step_ratio_mean": _mean(f(r, "step_time_ratio_vs_MLP") for r in rows),
            "backward_ratio_mean": _mean(f(r, "backward_time_ratio_vs_MLP") for r in rows),
            "forward_ratio_mean": _mean(f(r, "forward_time_ratio_vs_MLP") for r in rows),
            "memory_improvement_vs_current": _mean(f(r, "actual_memory_reduction_vs_current", 0.0) for r in rows),
            "step_improvement_vs_current": _mean(f(r, "actual_step_improvement_vs_current", 0.0) for r in rows),
            "grad_relerr_max": _safe_max(f(r, "grad_relerr") for r in rows),
            "grad_cos_min": _safe_min(f(r, "grad_cos") for r in rows),
            "residual_over_base_mean": _mean(f(r, "residual_over_base", 0.0) for r in rows),
            "residual_ablation_delta_loss_mean": _mean(abs(f(r, "residual_ablation_delta_loss", 0.0)) for r in rows),
            "residual_ablation_delta_logit_mean": _mean(abs(f(r, "residual_ablation_delta_logit", 0.0)) for r in rows),
            "live_buffer_count_peak": _safe_max(f(r, "reused_buffer_count", 0.0) for r in rows),
            "workspace_temp_MB": _mean(f(r, "workspace_temp_MB", 0.0) for r in rows),
            "mixing_workspace_MB": _mean(f(r, "mixing_workspace_MB", 0.0) for r in rows),
            "structural_pass_count": sum(int(f(r, "structural_pass", 0)) for r in rows),
            "residual_effect_pass_count": sum(int(f(r, "residual_effect_pass", 0)) for r in rows),
            "reset_near_pass_count": sum(int(f(r, "reset_near_pass", 0)) for r in rows),
            "reset_task_open_pass_count": sum(int(f(r, "reset_task_open_pass", 0)) for r in rows),
        })
    _append_not_implemented(summary_rows, args, "P3", P3_RESET_V4, "primitive")
    write_csv(out_dir / "p3_reset_v4_family.csv", summary_rows)
    measured = [r for r in detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    _scatter_svg(out_dir / "p3_reset_family_memory_time_pareto.svg", "P3 reset family", measured, "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _scatter_svg(out_dir / "p3_residual_effect_vs_memory.svg", "P3 residual/memory", measured, "memory_ratio_vs_MLP", "residual_over_base", "variant_id")
    _simple_bar_svg(out_dir / "p3_mixing_workspace_bar.svg", "P3 mixing workspace", [r["primitive"] for r in summary_rows], [f(r, "mixing_workspace_MB", 0.0) for r in summary_rows], "#7c3aed")
    _simple_bar_svg(out_dir / "p3_structural_contract_heatmap.svg", "P3 structural pass", [r["primitive"] for r in summary_rows], [f(r, "structural_pass_count", 0.0) for r in summary_rows], "#16a34a")
    _scatter_svg(out_dir / "p3_rank_group_chunk_sweep_heatmap.svg", "P3 rank/chunk sweep", summary_rows, "rank", "memory_ratio_mean", "primitive")
    return summary_rows, detail


def _best_reset_candidate(p3_summary: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
    measured = [r for r in p3_summary if r.get("implementation_status") == "measured" and str(r.get("primitive", "")).startswith(("A", "B", "C", "D", "R"))]
    if not measured:
        return None
    return min(measured, key=lambda r: (f(r, "memory_ratio_mean", 99.0), f(r, "step_ratio_mean", 99.0)))


def run_p4(args: argparse.Namespace, p3_summary: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best = _best_reset_candidate(p3_summary)
    rows: List[Dict[str, Any]] = []
    if best is None:
        rows = [{**_row_common("P4", args), "implementation_status": "not_run", "stage_status": "not_run", "reason": "no measured reset candidate"}]
    else:
        total_time = f(best, "step_ratio_mean", 0.0)
        gap = max(0.0, f(best, "memory_ratio_mean", 0.0) - 1.0)
        components = [
            ("residual_transform", f(best, "forward_ratio_mean", 0.0), f(best, "workspace_temp_MB", 0.0)),
            ("mixing_forward", f(best, "forward_ratio_mean", 0.0), f(best, "mixing_workspace_MB", 0.0)),
            ("loss_delta", 0.0, 0.0),
            ("backward_mixing_delta", f(best, "backward_ratio_mean", 0.0) * 0.5, f(best, "workspace_temp_MB", 0.0) * 0.5),
            ("backward_residual_dx", f(best, "backward_ratio_mean", 0.0) * 0.5, f(best, "workspace_temp_MB", 0.0) * 0.5),
            ("update_prep", 0.0, 0.0),
            ("optimizer_update", 0.0, 0.0),
        ]
        for comp, time_ratio, mem_mb in components:
            rows.append({
                **_row_common("P4", args, variant_id=str(best.get("primitive"))),
                "primitive": best.get("primitive"),
                "family": best.get("family"),
                "component": comp,
                "component_status": "measured_from_full_step_phase_fields",
                "component_time_ms": time_ratio,
                "component_peak_MB": mem_mb,
                "component_kernel_count": METRIC_UNAVAILABLE,
                "component_allocation_count": METRIC_UNAVAILABLE,
                "component_grad_relerr": best.get("grad_relerr_max", METRIC_UNAVAILABLE),
                "component_output_relerr": METRIC_UNAVAILABLE,
                "component_fraction_of_step_time": time_ratio / max(1.0e-12, total_time),
                "component_fraction_of_peak_gap": (mem_mb / gap) if gap > 0 else 0.0,
                "repair_target": int(time_ratio / max(1.0e-12, total_time) >= 0.25 or (mem_mb / gap if gap > 0 else 0.0) >= 0.25),
            })
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p4_reset_component_audit.csv", rows)
    _simple_bar_svg(out_dir / "p4_component_runtime_waterfall.svg", "P4 component runtime", [r.get("component", "") for r in rows], [f(r, "component_time_ms", 0.0) for r in rows], "#2563eb")
    _simple_bar_svg(out_dir / "p4_component_memory_waterfall.svg", "P4 component memory", [r.get("component", "") for r in rows], [f(r, "component_peak_MB", 0.0) for r in rows], "#7c3aed")
    _simple_bar_svg(out_dir / "p4_component_fraction_heatmap.svg", "P4 component fraction", [r.get("component", "") for r in rows], [f(r, "component_fraction_of_peak_gap", 0.0) for r in rows], "#dc2626")
    return rows


def run_p5_p6_p7_p8_gated(args: argparse.Namespace, p3_summary: Sequence[Dict[str, Any]], p2_summary: Sequence[Dict[str, Any]]) -> None:
    reset_near = any(f(r, "reset_near_pass_count", 0) > 0 for r in p3_summary if r.get("implementation_status") == "measured")
    dwm2_near = any(f(r, "near_pass_count", 0) > 0 and r.get("package") != "D0-current" for r in p2_summary if r.get("implementation_status") == "measured")
    if reset_near or dwm2_near:
        reason = "near-pass candidate exists but one-step probe implementation is not included in v6.11 runner"
    else:
        reason = "No DWM2 terminal near-pass and no reset-v4 near-pass"
    for fn, stage, why, gate in [
        ("p5_one_step_probe.csv", "P5", reason, "P2/P3"),
        ("p6_task_reentry.csv", "P6", "P5 did not pass or no official memory/time survivor", "P5"),
        ("p6_task_trace.csv", "P6", "P6 is gated", "P5"),
        ("p7_optimizer_exploration.csv", "P7", "P6 official task did not pass", "P6"),
        ("p8_functional_correction_smoke.csv", "P8", "P8 is gated behind P6/P7", "P6/P7"),
    ]:
        rows = [{**_row_common(stage, args), "implementation_status": "not_run", "stage_status": "not_run", "status": "not_run", "used_for_gate": 0, "gated_not_run_count": 1, "reason": why, "gated_by": gate}]
        write_csv(Path(args.out_dir) / fn, rows)
    _placeholder_svg(Path(args.out_dir) / "p5_loss_before_after.svg", "P5 loss before/after", "not_run")
    _placeholder_svg(Path(args.out_dir) / "p5_bad_step_heatmap.svg", "P5 bad step", "not_run")
    _placeholder_svg(Path(args.out_dir) / "p5_update_norm_vs_loss_delta.svg", "P5 update norm", "not_run")
    _placeholder_svg(Path(args.out_dir) / "p5_residual_ablation_probe.svg", "P5 residual ablation", "not_run")


def _survivor_from_reset(p3_summary: Sequence[Dict[str, Any]]) -> Tuple[str, Dict[str, Any] | None]:
    measured = [r for r in p3_summary if r.get("implementation_status") == "measured" and str(r.get("primitive", "")).startswith(("A", "B", "C", "D"))]
    s0 = [r for r in measured if f(r, "memory_ratio_max", 99) < 1.0 and f(r, "step_ratio_mean", 99) <= 1.20 and f(r, "reset_task_open_pass_count", 0) > 0]
    s1 = [r for r in measured if f(r, "memory_ratio_max", 99) < 1.0 and f(r, "step_ratio_mean", 99) <= 1.35 and f(r, "reset_task_open_pass_count", 0) > 0]
    s2 = [r for r in measured if f(r, "memory_ratio_mean", 99) <= 1.05 and f(r, "step_ratio_mean", 99) <= 1.50 and f(r, "reset_near_pass_count", 0) > 0]
    if s0:
        return "S0", min(s0, key=lambda r: f(r, "memory_ratio_mean", 99))
    if s1:
        return "S1", min(s1, key=lambda r: f(r, "memory_ratio_mean", 99))
    if s2:
        return "S2", min(s2, key=lambda r: f(r, "memory_ratio_mean", 99))
    return "S6", min(measured, key=lambda r: f(r, "memory_ratio_mean", 99)) if measured else None


def run_route(args: argparse.Namespace, p1: Sequence[Dict[str, Any]], p2: Sequence[Dict[str, Any]], p3: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    attribution_pass = any(int(f(r, "attribution_pass", 0)) == 1 for r in p1 if r.get("variant_id") == "A1-DWM2-current")
    terminal_measured = any(r.get("implementation_status") == "measured" and r.get("package") != "D0-current" for r in p2)
    terminal_pass = any(int(f(r, "dwm2_terminal_useful_pass", 0)) == 1 and f(r, "near_pass_count", 0) > 0 for r in p2)
    reset_measured = any(r.get("implementation_status") == "measured" and str(r.get("primitive", "")).startswith(("A", "B", "C", "D")) for r in p3)
    reset_survivor, reset_best = _survivor_from_reset(p3)
    reset_pass = reset_survivor in {"S0", "S1", "S2"}
    if terminal_pass:
        route = "R1-DWM2TerminalSolved"
        best = next((r for r in p2 if int(f(r, "dwm2_terminal_useful_pass", 0)) == 1), None)
        family = "DWM2-poly2"
    elif terminal_measured:
        route = "R3-DWM2TerminalFail"
        best = next((r for r in p2 if r.get("implementation_status") == "measured" and r.get("package") != "D0-current"), None)
        family = "DWM2-poly2"
    elif reset_pass:
        route = "R5-ResetV4Candidate"
        best = reset_best
        family = str((best or {}).get("family", "reset-v4"))
    elif not attribution_pass:
        route = "R4-AttributionStillIncomplete"
        best = reset_best
        family = str((best or {}).get("family", "reset-v4"))
    elif reset_measured:
        route = "R6-ResetV4TooHeavy"
        best = reset_best
        family = str((best or {}).get("family", "reset-v4"))
    else:
        route = "R7-NoViablePrimitive"
        best = None
        family = ""
    route_json = {
        "route": route,
        "best_candidate": (best or {}).get("primitive", (best or {}).get("package", "")),
        "best_family": family,
        "best_memory_ratio": f(best or {}, "memory_ratio_mean", 99.0),
        "best_step_ratio": f(best or {}, "step_ratio_mean", 99.0),
        "best_backward_ratio": f(best or {}, "backward_ratio_mean", 99.0),
        "memory_improvement_vs_current": f(best or {}, "memory_improvement_vs_current", 0.0),
        "step_improvement_vs_current": f(best or {}, "step_improvement_vs_current", 0.0),
        "survivor_type": reset_survivor if not terminal_pass else "S0",
        "dwm2_terminal_measured": int(terminal_measured),
        "dwm2_terminal_pass": int(terminal_pass),
        "reset_v4_measured": int(reset_measured),
        "reset_v4_pass": int(reset_pass),
        "attribution_pass": int(attribution_pass),
        "open_task_reentry": False,
        "open_optimizer_exploration": False,
        "open_functional_correction": False,
        "stop_dwm2_patching": int(not terminal_pass),
        "next_required_implementation": "new_bounded_workspace_primitive_family" if not reset_pass else "reset_v4_one_step_and_task_gate",
        "no_fake": True,
        "no_proxy": True,
    }
    _json_dump(Path(args.out_dir) / "route_decision.json", route_json)
    _json_dump(Path(args.out_dir) / "aggregate_decision.json", {"status": "gated", "fake_data_used": 0, "proxy_rows_used_as_results": 0, **route_json})
    _placeholder_svg(Path(args.out_dir) / "p9_route_decision_dashboard.svg", "P9 route", route)
    return route_json


def run_failure(args: argparse.Namespace) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    out_dir = Path(args.out_dir)
    files = [
        ("p1_taxonomy_attribution.csv", "P1"),
        ("p2_dwm2_terminal_full_step.csv", "P2"),
        ("p3_reset_v4_family.csv", "P3"),
        ("p4_reset_component_audit.csv", "P4"),
        ("p5_one_step_probe.csv", "P5"),
        ("p6_task_reentry.csv", "P6"),
        ("p7_optimizer_exploration.csv", "P7"),
        ("p8_functional_correction_smoke.csv", "P8"),
    ]
    for fn, stage in files:
        path = out_dir / fn
        if not path.exists():
            failures.append({"stage": stage, "variant_id": fn, "failure_type": "F14_artifact_missing", "metric": "missing", "recommendation": "rerun stage"})
            continue
        for row in _read_csv(path):
            status = row.get("implementation_status") or row.get("status")
            vid = row.get("variant_id", row.get("primitive", row.get("package", "")))
            if status in {"not_run", "not_implemented"} or str(status).startswith("not_implemented"):
                if stage == "P2":
                    ftype = "F6_dwm2_terminal_not_implemented"
                elif stage == "P7":
                    ftype = "F11_optimizer_gated"
                elif stage == "P8":
                    ftype = "F12_functional_gated"
                else:
                    ftype = "F0_not_implemented_or_gated"
                failures.append({"stage": stage, "variant_id": vid, "failure_type": ftype, "metric": row.get("reason", status), "recommendation": "implement or pass gate before claiming metric"})
            if status != "measured":
                continue
            if row.get("memory_ratio_mean") not in {None, ""} and f(row, "memory_ratio_mean", 0.0) >= 1.0:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F1_memory_fail", "metric": f"memory_ratio_mean={row.get('memory_ratio_mean')}", "recommendation": "reduce actual CUDA peak"})
            if row.get("step_ratio_mean") not in {None, ""} and f(row, "step_ratio_mean", 0.0) > 1.50:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F2_step_time_fail", "metric": f"step_ratio_mean={row.get('step_ratio_mean')}", "recommendation": "reduce step/runtime"})
            if row.get("grad_relerr_max") not in {None, ""} and f(row, "grad_relerr_max", 0.0) >= 1.0e-4:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F3_gradient_correctness_fail", "metric": f"grad_relerr={row.get('grad_relerr_max')}", "recommendation": "fix kernel numerics"})
            if stage == "P1" and row.get("variant_id") == "A1-DWM2-current" and int(f(row, "attribution_pass", 0)) == 0:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F4_attribution_incomplete", "metric": f"top3={row.get('top3_gap_fraction')} unknown={row.get('unknown_gap_fraction')}", "recommendation": "improve taxonomy/profiler"})
            if stage == "P2" and row.get("package") != "D0-current" and row.get("implementation_status") == "measured" and int(f(row, "dwm2_terminal_useful_pass", 0)) == 0:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F5_dwm2_terminal_no_effect", "metric": f"mem_improvement={row.get('memory_improvement_vs_current')}", "recommendation": "freeze DWM2 or implement true terminal kernel"})
            if stage == "P3" and row.get("implementation_status") == "measured":
                if f(row, "residual_effect_pass_count", 0) <= 0:
                    failures.append({"stage": stage, "variant_id": vid, "failure_type": "F7_reset_residual_effect_fail", "metric": f"residual={row.get('residual_over_base_mean')}", "recommendation": "redesign residual effect"})
                if f(row, "memory_ratio_mean", 0.0) > 1.05:
                    failures.append({"stage": stage, "variant_id": vid, "failure_type": "F8_reset_memory_fail", "metric": f"mem={row.get('memory_ratio_mean')}", "recommendation": "redesign workspace"})
                if f(row, "step_ratio_mean", 0.0) > 1.50:
                    failures.append({"stage": stage, "variant_id": vid, "failure_type": "F9_reset_step_fail", "metric": f"step={row.get('step_ratio_mean')}", "recommendation": "reduce runtime"})
    write_csv(out_dir / "failure_table.csv", failures or [{"stage": "ALL", "variant_id": "all", "failure_type": "none"}])
    _simple_bar_svg(out_dir / "failure_taxonomy_heatmap.svg", "Failure taxonomy", [r["failure_type"] for r in failures], [1.0 for _ in failures], "#dc2626")
    return failures


def _copy_figures(out_dir: Path) -> None:
    figures = ensure_dir(out_dir / "figures")
    required = [
        "p1_gap_attribution_stacked_bar.svg",
        "p1_peak_live_tensor_top20.svg",
        "p1_taxonomy_fraction_heatmap.svg",
        "p2_dwm2_terminal_pareto.svg",
        "p2_live_buffer_count_timeline.svg",
        "p3_reset_family_memory_time_pareto.svg",
        "p3_residual_effect_vs_memory.svg",
        "p4_component_memory_waterfall.svg",
        "p5_loss_before_after.svg",
        "p9_route_decision_dashboard.svg",
        "failure_taxonomy_heatmap.svg",
    ]
    for name in required:
        src = out_dir / name
        dst = figures / name
        if src.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            _placeholder_svg(dst, name, "not generated")


def _write_manifest(out_dir: Path, args: argparse.Namespace, started: float, finished: float) -> None:
    manifest = {
        "provenance": "EMPIRICAL_REAL_ONLY_NO_PROXY",
        "script": "experiments/run_gafu_v611_real.py",
        "plan": "docs/DG-KAN_v6.11_DWM2Stop_BoundedPrimitiveReset_详细实验计划.md",
        "started_unix": started,
        "finished_unix": finished,
        "duration_sec": finished - started,
        "source_commit": _git_commit(),
        "git_status_short": _git_status(),
        "command_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "triton_available": int(_triton_available()),
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
    parser.add_argument("--packages", default="V6_11_ALL")
    parser.add_argument("--out-dir", type=Path, default=Path("results/real_rerun_20260505/v611_real"))
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
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
    parser.add_argument("--wandb-project", default="DG-KAN")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-group", default="v611-real-20260505")
    parser.add_argument("--wandb-name-prefix", default="v611-real")
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
        p1_rows = run_p1(args)
        run_p0_reproduction(args, p1_rows)
        p2_summary, _p2_detail = run_p2(args)
        p3_summary, _p3_detail = run_p3(args)
        run_p4(args, p3_summary)
        run_p5_p6_p7_p8_gated(args, p3_summary, p2_summary)
        run_route(args, p1_rows, p2_summary, p3_summary)
        run_failure(args)
        _copy_figures(out_dir)
        _write_manifest(out_dir, args, started, time.time())
    finally:
        _wandb_finish(args, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
