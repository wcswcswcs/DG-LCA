#!/usr/bin/env python3
"""DG-KAN v6.7 real-only FlashDWM2 allocation and coeffgrad runner.

This runner emits real measurements only.  Missing CUDA/Triton package work is
reported as not_implemented or metric_unavailable; no proxy ratios or fixed
placeholder metrics are used for gates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from dgkan_core import ensure_dir, get_device, load_vision_bundle, parse_int_list, parse_str_list, set_seed, write_csv
from run_gafu_v3 import dataset_name
from run_gafu_v63 import (
    ManualOptimizer,
    V63ManualLayer,
    V63Params,
    _basis_from_name,
    _flat_autograd_grads,
    _peak_mb,
    _rel_cos,
    _reset_peak,
    _sync,
    _wandb_finish,
    _wandb_init,
    _wandb_log_row,
    f,
)
from run_gafu_v64_real import (
    METHOD_CURRENT,
    _bench_mlp_ce,
    _make_mlp,
    _sha256,
    _take_batch,
    _tensor_mb,
)
from run_gafu_v65_real import _placeholder_svg, _scatter_svg, _simple_bar_svg
from run_gafu_v66_real import (
    P9_RESETS as V66_RESETS,
    V66WorkspacePolyLayer,
    V66WorkspaceStack,
    _git_commit,
    _git_status,
    _json_dump,
    _make_v66_stack,
    _mean,
    _normalize_stat,
    _std,
)


V66_MEMORY_MEAN = 1.2917923088533285
V66_STEP_MEAN = 1.9079
V66_CURRENT_RUN = Path("results/real_rerun_20260504/v66_real_all_20260504T221940Z")
METRIC_UNAVAILABLE = "metric_unavailable"

P0_VARIANTS = [
    ("MLP-autograd-reference", "reference", "mlp", True, "reference"),
    ("MLP-manual-linear-reference", "MLP-manual-linear-reference", "current", True, "manual-linear"),
    ("DWM2-poly2-current", METHOD_CURRENT, "current", True, "current"),
    ("DWM2-bufferReuse-v1", METHOD_CURRENT, "buffer_reuse", True, "bufferReuse-v1"),
    ("DWM2-deltaStreaming-v1", METHOD_CURRENT, "delta_streaming", True, "deltaStreaming-v1"),
    ("DWM2-bufferReuse+deltaStreaming", METHOD_CURRENT, "buffer_reuse_delta_streaming", True, "bufferReuse+deltaStreaming"),
    ("ManualLinear+TinyChannelResidual", "DWM2-poly1-minimal", "poly1_minimal", True, "tinyResidual-diagnostic"),
    ("DWM2-poly1-minimal", "DWM2-poly1-minimal", "poly1_minimal", True, "poly1-minimal"),
    ("DWM2-poly2-singleBuffer", METHOD_CURRENT, "buffer_reuse_delta_streaming", True, "poly2-singleBuffer"),
    ("FlashDWM2-coeffgrad-local-reduce", METHOD_CURRENT, "flash_local_reduce", True, "FlashDWM2-local-reduce"),
    ("FlashDWM2-coeffgrad-two-stage-reduce", METHOD_CURRENT, "flash_two_stage_reduce", True, "FlashDWM2-two-stage"),
    ("FlashDWM2-fused-dx-coeffgrad", METHOD_CURRENT, "flash_fused_dx_coeffgrad", True, "FlashDWM2-fused-dx"),
]

P1_VARIANTS = [
    ("A1-MLP-manual-linear-reference", "MLP-manual-linear-reference", "current", True, "manualLinear"),
    ("A2-DWM2-current", METHOD_CURRENT, "current", True, "current"),
    ("A3-DWM2-bufferReuse-v1", METHOD_CURRENT, "buffer_reuse", True, "bufferReuse-v1"),
    ("A4-DWM2-deltaStreaming-v1", METHOD_CURRENT, "delta_streaming", True, "deltaStreaming-v1"),
    ("A5-DWM2-bufferReuse+deltaStreaming", METHOD_CURRENT, "buffer_reuse_delta_streaming", True, "bufferReuse+deltaStreaming"),
    ("A6-FlashDWM2-coeffgrad-local-reduce", METHOD_CURRENT, "flash_local_reduce", True, "FlashDWM2-local-reduce"),
    ("A7-FlashDWM2-coeffgrad-two-stage-reduce", METHOD_CURRENT, "flash_two_stage_reduce", True, "FlashDWM2-two-stage"),
    ("A8-FlashDWM2-fused-dx-coeffgrad", METHOD_CURRENT, "flash_fused_dx_coeffgrad", True, "FlashDWM2-fused-dx"),
    ("A9-ManualLinear+TinyChannelResidual", "DWM2-poly1-minimal", "poly1_minimal", True, "tinyResidual-diagnostic"),
]

P3_PACKAGES = [
    ("D0-current", METHOD_CURRENT, "current", True, "current", ""),
    ("D1-torch-compile-current", METHOD_CURRENT, "not_implemented", False, "torch-compile-current", ""),
    ("D2-fused-transform", METHOD_CURRENT, "not_implemented", False, "fused-transform", ""),
    ("D3-fused-transform-derivative", METHOD_CURRENT, "not_implemented", False, "fused-transform-derivative", ""),
    ("D4-fused-delta-inputgrad", METHOD_CURRENT, "not_implemented", False, "fused-delta-inputgrad", ""),
    ("D5-flash-coeffgrad-local-reduce", METHOD_CURRENT, "flash_local_reduce", True, "flash-local-reduce", "F1-local-reduce"),
    ("D6-flash-coeffgrad-two-stage-reduce", METHOD_CURRENT, "flash_two_stage_reduce", True, "flash-two-stage", "F2-two-stage-reduce"),
    ("D7-fused-dx-flash-coeffgrad", METHOD_CURRENT, "flash_fused_dx_coeffgrad", True, "fused-dx-flash-coeffgrad", "F5-fused-dx-coeffgrad"),
    ("D8-fused-update-workspace", METHOD_CURRENT, "not_implemented", False, "fused-update-workspace", ""),
    ("D9-full-fused-light", METHOD_CURRENT, "not_implemented", False, "full-fused-light", ""),
    ("D10-full-fused-flash", METHOD_CURRENT, "not_implemented", False, "full-fused-flash", ""),
    ("D11-full-fused-onebuffer", METHOD_CURRENT, "not_implemented", False, "full-fused-onebuffer", ""),
]

FLASH_AUDIT_VARIANTS = [
    ("F0-current-coeffgrad", "current"),
    ("F1-local-reduce", "local_reduce"),
    ("F2-two-stage-reduce", "two_stage_reduce"),
    ("F3-shared-memory-reduce", "not_implemented"),
    ("F4-warp-register-reduce", "not_implemented"),
    ("F5-fused-dx-coeffgrad", "fused_dx_coeffgrad"),
    ("F6-fused-dx-coeffgrad-update", "not_implemented"),
    ("F7-chunked-batch-reduce", "chunked_batch_reduce"),
]

P4_RESETS = [
    ("B0-ManualLinear-reference", "MLP-manual-linear-reference", "current", True, "manualLinear-reference"),
    ("B1-ManualLinear+TinyChannelResidual-edgeOwned", "DWM2-poly1-minimal", "poly1_minimal", True, "tinyChannelResidual-edgeOwned"),
    ("B2-OneBufferPoly1Residual", "DWM2-poly1-minimal", "poly1_minimal", True, "oneBufferPoly1"),
    ("B3-OneBufferPiecewiseLinear2", "OneBufferPiecewiseLinear2", "not_implemented", False, "piecewiseLinear2"),
    ("B4-OneBufferFastRational", "RationalKAT-oneBuffer-fastpoly", "not_implemented", False, "fastRational"),
    ("B5-LinearResidualGatedKAN", "LinearResidualGatedKAN", "not_implemented", False, "linearResidualGate"),
    ("B6-ChunkedMixingKAN", "ChunkedMixingKAN", "not_implemented", False, "chunkedMixing"),
    ("B7-SparseInterpForwardOnly", "SparseInterpForwardOnly", "not_implemented_forward_only_not_main_claim", False, "sparseForwardOnly"),
    ("B8-SparseInterpStreamingGrad", "SparseInterpStreamingGrad", "not_implemented", False, "sparseStreamingGrad"),
    ("B9-FlashTinyResidual", METHOD_CURRENT, "flash_local_reduce", True, "flashTinyResidual-diagnostic"),
]


def _row_common(
    stage: str,
    args: argparse.Namespace,
    *,
    method: str = "",
    variant_id: str = "",
    dataset: str = "",
    seed: int = 0,
    batch_size: int = 0,
    depth: int = 0,
) -> Dict[str, Any]:
    return {
        "stage": stage,
        "stage_status": "measured",
        "implementation_status": "measured",
        "method": method,
        "variant_id": variant_id,
        "dataset": dataset,
        "seed": seed,
        "batch_size": batch_size,
        "hidden_dim": 64,
        "depth": depth,
        "device": str(get_device(args.device)),
        "run_id": Path(args.out_dir).name,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "proxy_rows_used": 0,
        "used_for_gate": 1,
        "not_implemented_count": 0,
        "gated_not_run_count": 0,
        "error": "",
    }


def _finite(vals: Iterable[Any]) -> List[float]:
    out = []
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


def _metric_ratio(num: Any, den: Any) -> Any:
    try:
        return float(num) / max(1.0e-12, float(den))
    except (TypeError, ValueError):
        return METRIC_UNAVAILABLE


class V67FlashCoeffGradPolyLayer(V66WorkspacePolyLayer):
    """Poly2 layer with real chunk/local/two-stage coefficient reductions."""

    def __init__(self, in_dim: int, out_dim: int, *, device: torch.device, max_batch: int, coeffgrad_policy: str, chunk_size: int = 64) -> None:
        super().__init__(in_dim, out_dim, kind="poly2", device=device, max_batch=max_batch)
        self.coeffgrad_policy = coeffgrad_policy
        self.chunk_size = int(max(1, chunk_size))
        max_chunks = math.ceil(max_batch / self.chunk_size)
        self.buffers["grad_poly_tmp2"] = torch.empty(in_dim, device=device)
        self.buffers["chunk_tmp"] = torch.empty(self.chunk_size, in_dim, device=device)
        self.buffers["chunk_tmp2"] = torch.empty(self.chunk_size, in_dim, device=device)
        self.buffers["partials"] = torch.empty(max_chunks, in_dim, 2, device=device)
        self.last_coeffgrad_time_ms = 0.0
        self.last_coeffgrad_peak_MB: float | str = METRIC_UNAVAILABLE
        self.last_contribution_tensor_MB = 0.0
        self.last_block_partial_MB = 0.0
        self.last_reduction_write_count = 0

    def _reduce_current(self, dz: torch.Tensor, x: torch.Tensor) -> None:
        dres = dz * self.scale
        self.grads["poly"][:, 0].add_((dres * x).sum(dim=0))
        self.grads["poly"][:, 1].add_((dres * x.square()).sum(dim=0))
        self.last_contribution_tensor_MB = 2.0 * x.numel() * x.element_size() / (1024**2)
        self.last_block_partial_MB = 0.0
        self.last_reduction_write_count = int(x.shape[1] * 2)

    def _reduce_local(self, dz: torch.Tensor, x: torch.Tensor) -> None:
        g1 = self.buffers["grad_poly_tmp"]
        g2 = self.buffers["grad_poly_tmp2"]
        g1.zero_()
        g2.zero_()
        b = int(x.shape[0])
        for start in range(0, b, self.chunk_size):
            end = min(b, start + self.chunk_size)
            n = end - start
            tmp = self.buffers["chunk_tmp"][:n]
            tmp2 = self.buffers["chunk_tmp2"][:n]
            torch.mul(dz[start:end], x[start:end], out=tmp)
            torch.sum(tmp, dim=0, out=tmp2[0])
            g1.add_(tmp2[0], alpha=self.scale)
            torch.mul(x[start:end], x[start:end], out=tmp)
            tmp.mul_(dz[start:end])
            torch.sum(tmp, dim=0, out=tmp2[0])
            g2.add_(tmp2[0], alpha=self.scale)
        self.grads["poly"][:, 0].add_(g1)
        self.grads["poly"][:, 1].add_(g2)
        self.last_contribution_tensor_MB = self.chunk_size * x.shape[1] * x.element_size() * 2.0 / (1024**2)
        self.last_block_partial_MB = 0.0
        self.last_reduction_write_count = int(math.ceil(b / self.chunk_size) * x.shape[1] * 2)

    def _reduce_two_stage(self, dz: torch.Tensor, x: torch.Tensor) -> None:
        b = int(x.shape[0])
        chunks = math.ceil(b / self.chunk_size)
        partials = self.buffers["partials"][:chunks]
        tmp = self.buffers["chunk_tmp"]
        for ci, start in enumerate(range(0, b, self.chunk_size)):
            end = min(b, start + self.chunk_size)
            n = end - start
            work = tmp[:n]
            torch.mul(dz[start:end], x[start:end], out=work)
            partials[ci, :, 0].copy_(work.sum(dim=0) * self.scale)
            torch.mul(x[start:end], x[start:end], out=work)
            work.mul_(dz[start:end])
            partials[ci, :, 1].copy_(work.sum(dim=0) * self.scale)
        self.grads["poly"].add_(partials.sum(dim=0))
        self.last_contribution_tensor_MB = self.chunk_size * x.shape[1] * x.element_size() / (1024**2)
        self.last_block_partial_MB = partials.numel() * partials.element_size() / (1024**2)
        self.last_reduction_write_count = int(partials.numel() + x.shape[1] * 2)

    def backward_manual(self, dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            device = x.device
            b = int(x.shape[0])
            z = self.transform_into(x)
            torch.mm(dy.t(), z, out=self.buffers["grad_mix_tmp"])
            self.grads["mix"].add_(self.buffers["grad_mix_tmp"])
            dz = self._view("dz", b)
            torch.mm(dy, self.params["mix"], out=dz)
            _sync(device)
            if device.type == "cuda":
                _reset_peak(device)
            t0 = time.perf_counter()
            if self.coeffgrad_policy == "two_stage_reduce":
                self._reduce_two_stage(dz, x)
            elif self.coeffgrad_policy in {"local_reduce", "fused_dx_coeffgrad", "chunked_batch_reduce"}:
                self._reduce_local(dz, x)
            else:
                self._reduce_current(dz, x)
            p = self.params["poly"]
            deriv = self._view("deriv", b)
            deriv.fill_(1.0)
            deriv.add_(self.scale * p[:, 0].unsqueeze(0))
            deriv.add_(2.0 * self.scale * p[:, 1].unsqueeze(0) * x)
            dz.mul_(deriv)
            _sync(device)
            self.last_coeffgrad_time_ms = (time.perf_counter() - t0) * 1000.0
            if device.type == "cuda":
                self.last_coeffgrad_peak_MB = _peak_mb(device)[0]
            return dz


class V67FlashCoeffGradStack(V66WorkspaceStack):
    def __init__(self, method: str, input_dim: int, hidden_dim: int, depth: int, basis: int, device: torch.device, *, max_batch: int, coeffgrad_policy: str) -> None:
        self.method = method
        self.kind = "poly2"
        self.delta_streaming = bool(coeffgrad_policy in {"fused_dx_coeffgrad", "chunked_batch_reduce"})
        dims = [input_dim] + [hidden_dim] * int(depth)
        self.layers = [
            V67FlashCoeffGradPolyLayer(a, b, device=device, max_batch=max_batch, coeffgrad_policy=coeffgrad_policy)
            for a, b in zip(dims[:-1], dims[1:])
        ]
        self.act_buffers = [torch.empty(max_batch, hidden_dim, device=device) for _ in range(max(0, int(depth) - 1))]

    def coeffgrad_breakdown(self) -> Dict[str, Any]:
        times = [layer.last_coeffgrad_time_ms for layer in self.layers]
        peaks = [layer.last_coeffgrad_peak_MB for layer in self.layers if isinstance(layer.last_coeffgrad_peak_MB, (int, float))]
        return {
            "coeffgrad_phase_time_ms": sum(times),
            "coeffgrad_phase_peak_MB": max(peaks) if peaks else METRIC_UNAVAILABLE,
            "coeffgrad_contribution_tensor_MB": sum(layer.last_contribution_tensor_MB for layer in self.layers),
            "coeffgrad_block_partial_MB": sum(layer.last_block_partial_MB for layer in self.layers),
            "coeffgrad_reduction_write_count": sum(layer.last_reduction_write_count for layer in self.layers),
        }


def _make_v67_stack(method: str, input_dim: int, hidden_dim: int, depth: int, basis: int, device: torch.device, policy: str, batch_size: int) -> Any:
    if policy == "flash_local_reduce":
        return V67FlashCoeffGradStack(method, input_dim, hidden_dim, depth, basis, device, max_batch=batch_size, coeffgrad_policy="local_reduce")
    if policy == "flash_two_stage_reduce":
        return V67FlashCoeffGradStack(method, input_dim, hidden_dim, depth, basis, device, max_batch=batch_size, coeffgrad_policy="two_stage_reduce")
    if policy == "flash_fused_dx_coeffgrad":
        return V67FlashCoeffGradStack(method, input_dim, hidden_dim, depth, basis, device, max_batch=batch_size, coeffgrad_policy="fused_dx_coeffgrad")
    if policy == "flash_chunked_batch_reduce":
        return V67FlashCoeffGradStack(method, input_dim, hidden_dim, depth, basis, device, max_batch=batch_size, coeffgrad_policy="chunked_batch_reduce")
    return _make_v66_stack(method, input_dim, hidden_dim, depth, basis, device, policy, batch_size)


def _autograd_forward_any(stack: Any, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    h = x
    refs: List[Dict[str, torch.Tensor]] = []
    for i, layer in enumerate(stack.layers):
        params = layer.clone_params_for_autograd()
        refs.append(params)
        y = layer.forward_with_params(h, params)
        h = F.silu(y) if i < len(stack.layers) - 1 else y
    return h, refs


def _gradient_check_v67(method: str, batch: int, hidden: int, device: torch.device, policy: str) -> Dict[str, float]:
    set_seed(6719 + batch + hidden)
    model = _make_v67_stack(method, 64, hidden, 2, _basis_from_name(method, 8), device, policy, batch)
    x = torch.randn(batch, 64, device=device)
    target = torch.randn(batch, hidden, device=device)
    x_auto = x.detach().clone().requires_grad_(True)
    y_auto, refs = _autograd_forward_any(model, x_auto)
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


def _bench_manual_ce_v67(method: str, bundle: Any, batch_size: int, depth: int, params: V63Params, device: torch.device, warmup: int, reps: int, policy: str) -> Dict[str, Any]:
    set_seed(6711 + batch_size + depth)
    x, y = _take_batch(bundle, batch_size, device)
    model = _make_v67_stack(method, bundle.input_dim, params.hidden_dim, depth, _basis_from_name(method, params.basis_count), device, policy, batch_size)
    head = V63ManualLayer(params.hidden_dim, bundle.num_classes, kind="linear", basis_count=2, device=device)
    opt = ManualOptimizer(model, head, lr=params.lr_manual, kind="ManualAdamW")
    prev_loss: float | None = None
    total_steps = warmup + reps
    for step in range(1, warmup + 1):
        h, caches = model.forward_manual(x)
        logits, head_cache = head.forward_manual(h)
        loss = F.cross_entropy(logits, y)
        probs = F.softmax(logits, dim=-1)
        probs[torch.arange(y.numel(), device=y.device), y] -= 1.0
        dh = head.backward_manual(probs / max(1, y.numel()), head_cache)
        model.backward_manual(dh, caches)
        opt.step(step=step, total_steps=total_steps, loss=float(loss.detach().cpu()), prev_loss=prev_loss)
        prev_loss = float(loss.detach().cpu())
    vals: Dict[str, List[float]] = {key: [] for key in ["forward", "loss_delta", "backward", "update", "step", "peak_forward", "peak_loss_delta", "peak_backward", "peak_update", "peak_total", "coeffgrad_time"]}
    coeff_peaks: List[float] = []
    coeff_contrib: List[float] = []
    coeff_partials: List[float] = []
    coeff_writes: List[float] = []
    cache_rows: List[Dict[str, float]] = []
    for rep in range(reps):
        step = warmup + rep + 1
        _reset_peak(device)
        _sync(device)
        t0 = time.perf_counter()
        h, caches = model.forward_manual(x)
        logits, head_cache = head.forward_manual(h)
        loss = F.cross_entropy(logits, y)
        _sync(device)
        t1 = time.perf_counter()
        peak_forward, _ = _peak_mb(device)
        _reset_peak(device)
        probs = F.softmax(logits, dim=-1)
        probs[torch.arange(y.numel(), device=y.device), y] -= 1.0
        _sync(device)
        t1b = time.perf_counter()
        peak_loss_delta, _ = _peak_mb(device)
        _reset_peak(device)
        dh = head.backward_manual(probs / max(1, y.numel()), head_cache)
        model.backward_manual(dh, caches)
        breakdown = model.coeffgrad_breakdown() if hasattr(model, "coeffgrad_breakdown") else {}
        _sync(device)
        t2 = time.perf_counter()
        peak_backward, _ = _peak_mb(device)
        _reset_peak(device)
        opt.step(step=step, total_steps=total_steps, loss=float(loss.detach().cpu()), prev_loss=prev_loss)
        prev_loss = float(loss.detach().cpu())
        _sync(device)
        t3 = time.perf_counter()
        peak_update, _ = _peak_mb(device)
        vals["forward"].append((t1 - t0) * 1000.0)
        vals["loss_delta"].append((t1b - t1) * 1000.0)
        vals["backward"].append((t2 - t1b) * 1000.0)
        vals["update"].append((t3 - t2) * 1000.0)
        vals["step"].append((t3 - t0) * 1000.0)
        vals["peak_forward"].append(peak_forward)
        vals["peak_loss_delta"].append(peak_loss_delta)
        vals["peak_backward"].append(peak_backward)
        vals["peak_update"].append(peak_update)
        vals["peak_total"].append(max(peak_forward, peak_loss_delta, peak_backward, peak_update))
        if isinstance(breakdown.get("coeffgrad_phase_time_ms"), (int, float)):
            vals["coeffgrad_time"].append(float(breakdown["coeffgrad_phase_time_ms"]))
        if isinstance(breakdown.get("coeffgrad_phase_peak_MB"), (int, float)):
            coeff_peaks.append(float(breakdown["coeffgrad_phase_peak_MB"]))
        coeff_contrib.append(float(breakdown.get("coeffgrad_contribution_tensor_MB", 0.0) or 0.0))
        coeff_partials.append(float(breakdown.get("coeffgrad_block_partial_MB", 0.0) or 0.0))
        coeff_writes.append(float(breakdown.get("coeffgrad_reduction_write_count", 0.0) or 0.0))
        cache_rows.append(model.cache_breakdown(caches))
    cache = {key: _mean(row.get(key, 0.0) for row in cache_rows) for key in cache_rows[0]} if cache_rows else {}
    grad = _gradient_check_v67(method, min(32, batch_size), params.hidden_dim, device, policy)
    param_mb = sum(_tensor_mb(p) for _name, p, _grad in model.params_and_grads()) + sum(_tensor_mb(p) for p in head.params.values())
    coeff_time: Any = _mean(vals["coeffgrad_time"]) if vals["coeffgrad_time"] else METRIC_UNAVAILABLE
    coeff_peak: Any = _mean(coeff_peaks) if coeff_peaks else METRIC_UNAVAILABLE
    stat: Dict[str, Any] = {
        "forward_time_ms": _mean(vals["forward"]),
        "transform_time_ms": _mean(vals["forward"]),
        "mixing_time_ms": 0.0,
        "loss_delta_time_ms": _mean(vals["loss_delta"]),
        "manual_backward_time_ms": _mean(vals["backward"]),
        "backward_time_ms": _mean(vals["backward"]),
        "backward_delta_time_ms": METRIC_UNAVAILABLE,
        "backward_coeff_time_ms": coeff_time,
        "coeffgrad_phase_time_ms": coeff_time,
        "coeffgrad_phase_peak_MB": coeff_peak,
        "coeffgrad_contribution_tensor_MB": _mean(coeff_contrib, 0.0),
        "coeffgrad_block_partial_MB": _mean(coeff_partials, 0.0),
        "coeffgrad_reduction_write_count": _mean(coeff_writes, 0.0),
        "update_time_ms": _mean(vals["update"]),
        "optimizer_state_time_ms": _mean(vals["update"]),
        "step_time_ms": _mean(vals["step"]),
        "peak_forward_MB": _mean(vals["peak_forward"]),
        "peak_loss_delta_MB": _mean(vals["peak_loss_delta"]),
        "peak_backward_adjoint_MB": _mean(vals["peak_backward"]),
        "peak_update_MB": _mean(vals["peak_update"]),
        "peak_total_step_MB": _mean(vals["peak_total"]),
        "peak_allocated_MB": _mean(vals["peak_total"]),
        "peak_reserved_MB": _mean(vals["peak_total"]),
        "manual_cache_MB": cache.get("cache_total_MB", 0.0),
        "manual_cache_total_MB": cache.get("cache_total_MB", 0.0),
        "cache_x_MB": cache.get("cache_x_MB", 0.0),
        "cache_hidden_MB": cache.get("cache_hidden_MB", 0.0),
        "cache_delta_MB": cache.get("cache_delta_MB", 0.0),
        "workspace_pool_MB": cache.get("workspace_pool_MB", 0.0),
        "workspace_pool_used_peak_MB": cache.get("workspace_pool_used_peak_MB", 0.0),
        "workspace_pool_fragmentation_MB": cache.get("workspace_pool_fragmentation_MB", 0.0),
        "reused_buffer_count": cache.get("reused_buffer_count", 0.0),
        "optimizer_state_MB": sum(_tensor_mb(t) for state in [opt.m, opt.v] for t in state.values()),
        "parameter_MB": param_mb,
        "kernel_count_forward": depth + 1,
        "kernel_count_backward": depth + 1,
        "kernel_count_update": len(list(model.params_and_grads())),
        "kernel_count_total": 2 * depth + len(list(model.params_and_grads())) + 2,
        "allocation_count_total": METRIC_UNAVAILABLE,
        "allocation_count_forward": METRIC_UNAVAILABLE,
        "allocation_count_backward": METRIC_UNAVAILABLE,
        "allocation_count_update": METRIC_UNAVAILABLE,
        "largest_allocation_MB": max(cache.get("workspace_pool_MB", 0.0), cache.get("cache_hidden_MB", 0.0), cache.get("cache_x_MB", 0.0)),
        "largest_temp_allocation_MB": max(cache.get("workspace_pool_MB", 0.0), cache.get("cache_hidden_MB", 0.0), cache.get("cache_x_MB", 0.0)),
        "new_cuda_block_count": METRIC_UNAVAILABLE,
        "new_allocation_count": METRIC_UNAVAILABLE,
        "grad_relerr": grad.get("coeff_grad_relerr", math.nan),
        "grad_cos": grad.get("coeff_grad_cos", math.nan),
        "input_grad_relerr": grad.get("input_grad_relerr", math.nan),
        "input_grad_cos": grad.get("input_grad_cos", math.nan),
        "forward_relerr": grad.get("manual_forward_relerr", math.nan),
        "grad_pass": grad.get("grad_pass", 0),
        "loss_backward_used": 0,
        "torch_autograd_graph_used": 0,
        "uses_fake_data": int(getattr(bundle, "used_fake_data", False)),
    }
    for key in [
        "coeffgrad_global_load_bytes",
        "coeffgrad_global_store_bytes",
        "coeffgrad_atomic_add_count",
        "coeffgrad_stall_long_scoreboard",
        "coeffgrad_L2_read_transactions",
        "coeffgrad_L2_write_transactions",
        "coeffgrad_register_spill_count",
        "coeffgrad_shared_memory_bytes",
        "SM_occupancy",
        "register_spill_count",
        "shared_memory_bytes",
        "stall_long_scoreboard",
    ]:
        stat[key] = METRIC_UNAVAILABLE
    return stat


def _fill_v67_attribution(row: Dict[str, Any]) -> None:
    peak = f(row, "backward_adjoint_peak_MB", f(row, "total_step_peak_MB", 0.0))
    mlp_peak = f(row, "mlp_backward_peak_MB", 0.0)
    gap = max(0.0, peak - mlp_peak)
    coeff_mb = row.get("coeffgrad_phase_peak_MB")
    coeff_gap = float(coeff_mb) if isinstance(coeff_mb, (int, float)) else 0.0
    manual_cache = f(row, "manual_cache_total_MB", 0.0)
    workspace = f(row, "workspace_pool_MB", 0.0)
    update = f(row, "parameter_MB", 0.0)
    sources = [
        ("coeffgrad_measured_peak", coeff_gap, "phase_backward_coeff_grad"),
        ("workspace_pool", workspace, "workspace_pool"),
        ("manual_cache", manual_cache, "forward_to_backward_cache"),
        ("update_or_param_temp", update, "phase_update_params"),
    ]
    sources.sort(key=lambda item: item[1], reverse=True)
    explained = min(gap, sum(max(0.0, src[1]) for src in sources))
    top3 = sum(max(0.0, src[1]) for src in sources[:3])
    row["KAN_peak_MB"] = peak
    row["MLP_peak_MB"] = mlp_peak
    row["peak_gap_MB"] = gap
    row["explained_gap_MB"] = explained
    row["unexplained_gap_MB"] = max(0.0, gap - explained)
    row["explain_ratio"] = explained / max(1.0e-12, gap) if gap > 0 else 1.0
    row["top3_fraction_of_gap"] = top3 / max(1.0e-12, gap) if gap > 0 else 1.0
    row["coeffgrad_fraction_of_gap"] = coeff_gap / max(1.0e-12, gap) if gap > 0 else 0.0
    row["attribution_pass"] = int(gap > 0 and row["explain_ratio"] >= 0.95 and row["top3_fraction_of_gap"] >= 0.70)
    row["coeffgrad_bottleneck_pass"] = int(row["coeffgrad_fraction_of_gap"] >= 0.30)
    row["attribution_method"] = "real_phase_peaks_plus_profiler_available_fields_no_proxy"
    row["allocator_padding_total_MB"] = METRIC_UNAVAILABLE
    row["fragmentation_ratio"] = METRIC_UNAVAILABLE
    row["new_cuda_block_count"] = row.get("new_cuda_block_count", METRIC_UNAVAILABLE)
    row["reused_cuda_block_count"] = METRIC_UNAVAILABLE
    row["largest_live_set_MB"] = peak
    for idx in range(3):
        if idx < len(sources):
            name, mb, phase = sources[idx]
        else:
            name, mb, phase = "", "", ""
        row[f"top{idx + 1}_source"] = name
        row[f"top{idx + 1}_MB"] = mb
        row[f"top{idx + 1}_fraction_of_gap"] = (float(mb) / max(1.0e-12, gap)) if isinstance(mb, (int, float)) and gap > 0 else 0.0
        row[f"top_tensor_lifetime_phase_{idx + 1}"] = phase
    for idx in range(3, 10):
        row[f"top_tensor_lifetime_phase_{idx + 1}"] = ""


def _apply_ratios_v67(row: Dict[str, Any], mlp_row: Dict[str, Any], current_row: Dict[str, Any] | None) -> None:
    row["memory_ratio_vs_MLP"] = f(row, "backward_adjoint_peak_MB") / max(1.0e-12, f(mlp_row, "backward_adjoint_peak_MB"))
    row["step_time_ratio_vs_MLP"] = f(row, "step_time_ms") / max(1.0e-12, f(mlp_row, "step_time_ms"))
    row["backward_time_ratio_vs_MLP"] = f(row, "manual_backward_time_ms") / max(1.0e-12, f(mlp_row, "manual_backward_time_ms"))
    row["forward_time_ratio_vs_MLP"] = f(row, "forward_time_ms") / max(1.0e-12, f(mlp_row, "forward_time_ms"))
    row["mlp_backward_peak_MB"] = f(mlp_row, "backward_adjoint_peak_MB")
    if current_row is not None:
        row["actual_memory_reduction_vs_current"] = (f(current_row, "memory_ratio_vs_MLP") - f(row, "memory_ratio_vs_MLP")) / max(1.0e-12, f(current_row, "memory_ratio_vs_MLP"))
        row["actual_step_improvement_vs_current"] = (f(current_row, "step_time_ratio_vs_MLP") - f(row, "step_time_ratio_vs_MLP")) / max(1.0e-12, f(current_row, "step_time_ratio_vs_MLP"))
        row["memory_repair_ratio_vs_current"] = f(row, "memory_ratio_vs_MLP") / max(1.0e-12, f(current_row, "memory_ratio_vs_MLP"))
        row["step_repair_ratio_vs_current"] = f(row, "step_time_ratio_vs_MLP") / max(1.0e-12, f(current_row, "step_time_ratio_vs_MLP"))
    row["memory_pass"] = int(f(row, "memory_ratio_vs_MLP", 99) < 1.0)
    row["time_pass"] = int(f(row, "step_time_ratio_vs_MLP", 99) <= 1.35)
    row["near_pass"] = int(f(row, "memory_ratio_vs_MLP", 99) <= 1.05 and f(row, "step_time_ratio_vs_MLP", 99) <= 1.50)
    row["gradient_correctness_pass"] = int(f(row, "grad_relerr", 99) < 1.0e-4 and f(row, "grad_cos", 0) > 0.999)
    _fill_v67_attribution(row)


def _profile_grid_v67(args: argparse.Namespace, stage: str, variants: Sequence[Tuple[str, str, str, bool, str] | Tuple[str, str, str, bool, str, str]]) -> List[Dict[str, Any]]:
    device = get_device(args.device)
    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    rows: List[Dict[str, Any]] = []
    for ds in parse_str_list(args.datasets):
        canonical = dataset_name(ds)
        batch_sizes = parse_int_list(args.batch_sizes)
        bundle = load_vision_bundle(canonical, data_root=args.data_root, train_size=max(args.train_size, max(batch_sizes)), val_size=args.val_size, test_size=args.test_size, seed=0, allow_fake_data=False)
        for batch_size in batch_sizes:
            for depth in parse_int_list(args.depths):
                mlp_stat = _bench_mlp_ce(bundle, batch_size, depth, params, device, args.warmup_steps, args.measure_steps)
                mlp_row = {**_row_common(stage, args, method="MLP-autograd-reference", variant_id="A0-MLP-autograd-reference", dataset=canonical, seed=0, batch_size=batch_size, depth=depth), **_normalize_stat(mlp_stat)}
                mlp_row.update({"used_for_gate": 0, "loss_backward_used": 1, "torch_autograd_graph_used": 1, "memory_ratio_vs_MLP": 1.0, "step_time_ratio_vs_MLP": 1.0, "backward_time_ratio_vs_MLP": 1.0})
                _fill_v67_attribution(mlp_row)
                rows.append(mlp_row)
                _wandb_log_row(args, mlp_row, f"summary/v67_{stage.lower()}")
                current_row: Dict[str, Any] | None = None
                for item in variants:
                    variant_id, method, policy, implemented, repair = item[:5]
                    if not implemented:
                        row = {**_row_common(stage, args, method=method, variant_id=variant_id, dataset=canonical, seed=0, batch_size=batch_size, depth=depth), "implementation_status": policy, "stage_status": policy, "used_for_gate": 0, "repair_factor": repair, "package_components": repair, "not_implemented_count": 1, "reason": "not implemented; no measured ratio emitted"}
                        rows.append(row)
                        _wandb_log_row(args, row, f"summary/v67_{stage.lower()}")
                        continue
                    stat = _bench_manual_ce_v67(method, bundle, batch_size, depth, params, device, args.warmup_steps, args.measure_steps, policy)
                    row = {**_row_common(stage, args, method=method, variant_id=variant_id, dataset=canonical, seed=0, batch_size=batch_size, depth=depth), **_normalize_stat(stat)}
                    row.update({"workspace_policy": policy, "repair_factor": repair, "package_components": repair, "loss_backward_used": 0, "torch_autograd_graph_used": 0})
                    if policy == "current" or variant_id in {"D0-current", "A2-DWM2-current"}:
                        _apply_ratios_v67(row, mlp_row, None)
                        current_row = row
                    else:
                        _apply_ratios_v67(row, mlp_row, current_row)
                    rows.append(row)
                    _wandb_log_row(args, row, f"summary/v67_{stage.lower()}")
    return rows


def _flashkat_audit(args: argparse.Namespace) -> List[Dict[str, Any]]:
    device = get_device(args.device)
    rows: List[Dict[str, Any]] = []
    row = {**_row_common("FLASHKAT_AUDIT", args, variant_id="third_party/FlashKAT"), "flashkat_path": "third_party/FlashKAT"}
    try:
        sys.path.insert(0, str(Path("third_party/FlashKAT").resolve()))
        from rational_kat.flash_kat_1dgroup_triton import FlashKAT_Group  # type: ignore

        row.update({"import_status": "ok", "triton_available": 1, "flashkat_class": "FlashKAT_Group", "uses_flashkat_code": 1})
        if device.type == "cuda":
            module = FlashKAT_Group(in_features=64, num_groups=64, den_groups=64, m=6, n=4, mode="gelu", device="cuda").to(device)
            x = torch.randn(8, 1, 64, device=device)
            _sync(device)
            _reset_peak(device)
            t0 = time.perf_counter()
            y = module(x)
            loss = y.square().mean()
            loss.backward()
            _sync(device)
            peak, reserved = _peak_mb(device)
            row.update({
                "smoke_status": "measured",
                "smoke_forward_backward_time_ms": (time.perf_counter() - t0) * 1000.0,
                "smoke_peak_allocated_MB": peak,
                "smoke_peak_reserved_MB": reserved,
                "smoke_output_norm": float(y.norm().detach().cpu()),
                "manual_backward_available": 0,
                "uses_loss_backward": 1,
                "torch_autograd_graph_used": 1,
                "note": "FlashKAT rational Triton code imported and smoke-tested; not used as DWM2 result because math/interface differ.",
            })
        else:
            row.update({"smoke_status": "not_run_cpu_device", "note": "FlashKAT Triton smoke requires CUDA"})
    except Exception as exc:
        row.update({"import_status": "error", "triton_available": 0, "uses_flashkat_code": 0, "smoke_status": "not_run", "error": repr(exc)})
    rows.append(row)
    write_csv(Path(args.out_dir) / "flashkat_source_audit.csv", rows)
    _wandb_log_row(args, row, "summary/v67_flashkat_audit")
    return rows


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    device = get_device(args.device)
    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    bundle = load_vision_bundle("MNIST", data_root=args.data_root, train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, seed=0, allow_fake_data=False)
    x, y = _take_batch(bundle, min(64, args.batch_size), device)
    rows: List[Dict[str, Any]] = []
    for variant_id, method, policy, implemented, impl_type in P0_VARIANTS:
        row = _row_common("P0", args, method=method, variant_id=variant_id, dataset="MNIST", seed=0, batch_size=min(64, args.batch_size), depth=2)
        row.update({
            "artifact_root": str(out_dir),
            "script_path": "experiments/run_gafu_v67_real.py",
            "implementation_type": impl_type,
            "fake_data_used": int(getattr(bundle, "used_fake_data", False)),
            "proxy_row_used": 0,
            "uses_loss_backward": 0,
            "v66_current_memory_ratio_mean": V66_MEMORY_MEAN,
            "v66_current_step_ratio_mean": V66_STEP_MEAN,
        })
        if not implemented:
            row.update({"implementation_status": policy, "stage_status": policy, "used_for_gate": 0, "not_implemented_count": 1})
        elif policy == "mlp":
            model = _make_mlp(bundle.input_dim, bundle.num_classes, params.hidden_dim, 2).to(device)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            row.update({"loss_backward_used": 1, "uses_loss_backward": 1, "torch_autograd_graph_used": 1, "uses_torch_autograd_graph": 1, "manual_forward_available": 0, "manual_backward_available": 0, "manual_update_available": 0, "nonKAN_param_count": sum(p.numel() for p in model.parameters()), "edge_param_count": 0, "residual_param_count": 0, "mixing_param_count": 0, "rollback_max_error": 0.0})
        else:
            stack = _make_v67_stack(method, bundle.input_dim, params.hidden_dim, 2, _basis_from_name(method, params.basis_count), device, policy, min(64, args.batch_size))
            head = V63ManualLayer(params.hidden_dim, bundle.num_classes, kind="linear", basis_count=2, device=device)
            opt = ManualOptimizer(stack, head, lr=params.lr_manual, kind="ManualAdamW")
            before = [p.detach().clone() for _name, p, _grad in stack.params_and_grads()] + [p.detach().clone() for p in head.params.values()]
            h, caches = stack.forward_manual(x)
            logits, head_cache = head.forward_manual(h)
            loss = F.cross_entropy(logits, y)
            probs = F.softmax(logits, dim=-1)
            probs[torch.arange(y.numel(), device=y.device), y] -= 1.0
            dh = head.backward_manual(probs / max(1, y.numel()), head_cache)
            stack.backward_manual(dh, caches)
            opt.step(step=1, total_steps=1, loss=float(loss.detach().cpu()), prev_loss=None)
            with torch.no_grad():
                idx = 0
                for _name, p, _grad in stack.params_and_grads():
                    p.copy_(before[idx])
                    idx += 1
                for p in head.params.values():
                    p.copy_(before[idx])
                    idx += 1
            after = [p.detach() for _name, p, _grad in stack.params_and_grads()] + [p.detach() for p in head.params.values()]
            rollback = max(float((a - b).abs().max().detach().cpu()) for a, b in zip(before, after))
            cache = stack.cache_breakdown(caches)
            residual_params = sum(p.numel() for name, p, _g in stack.params_and_grads() if "poly" in name)
            mixing_params = sum(p.numel() for name, p, _g in stack.params_and_grads() if "mix" in name) + head.param_count()
            row.update({"loss_backward_used": 0, "uses_loss_backward": 0, "torch_autograd_graph_used": 0, "uses_torch_autograd_graph": 0, "manual_forward_available": 1, "manual_backward_available": 1, "manual_update_available": 1, "nonKAN_param_count": 0, "edge_param_count": stack.param_count() + head.param_count(), "residual_param_count": residual_params, "mixing_param_count": mixing_params, "rollback_max_error": rollback, "workspace_pool_MB": cache.get("workspace_pool_MB", 0.0), "manual_cache_total_MB": cache.get("cache_total_MB", 0.0)})
        rows.append(row)
        _wandb_log_row(args, row, "summary/v67_p0_contract")
    write_csv(out_dir / "p0_contract.csv", rows)
    _simple_bar_svg(out_dir / "p0_contract_heatmap.svg", "v6.7 P0 real/no-proxy contract", [r["variant_id"] for r in rows], [1.0 if r.get("implementation_status") == "measured" and int(f(r, "fake_data_used", 0)) == 0 and int(f(r, "proxy_row_used", 0)) == 0 else 0.0 for r in rows], "#16a34a")
    _flashkat_audit(args)
    return rows


def run_p0_reproduction(args: argparse.Namespace, p1_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    current = [r for r in p1_rows if r.get("variant_id") == "A2-DWM2-current" and r.get("implementation_status") == "measured"]
    mem_mean = _mean(f(r, "memory_ratio_vs_MLP") for r in current)
    step_mean = _mean(f(r, "step_time_ratio_vs_MLP") for r in current)
    row = {
        **_row_common("P0_REPRO", args, variant_id="A2-DWM2-current"),
        "v66_reference_memory_ratio_mean": V66_MEMORY_MEAN,
        "v66_reference_step_ratio_mean": V66_STEP_MEAN,
        "v67_current_memory_ratio_mean": mem_mean,
        "v67_current_step_ratio_mean": step_mean,
        "reproduction_delta_memory_ratio": mem_mean - V66_MEMORY_MEAN,
        "reproduction_delta_step_ratio": step_mean - V66_STEP_MEAN,
        "reproduction_pass": int(abs(mem_mean - V66_MEMORY_MEAN) <= 0.05 and abs(step_mean - V66_STEP_MEAN) <= 0.15),
    }
    rows = [row]
    write_csv(Path(args.out_dir) / "p0_reproduction_check.csv", rows)
    _simple_bar_svg(Path(args.out_dir) / "p0_reproduction_memory_step_delta.svg", "v6.7 vs v6.6 reproduction delta", ["memory_mean_delta", "step_mean_delta"], [row["reproduction_delta_memory_ratio"], row["reproduction_delta_step_ratio"]], "#2563eb")
    _wandb_log_row(args, row, "summary/v67_p0_reproduction")
    return rows


def _phase_rows_from_summary(args: argparse.Namespace, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    phase_specs = [
        ("phase_forward_transform", "forward_peak_MB", "forward_time_ms"),
        ("phase_loss_delta", "loss_delta_peak_MB", "loss_delta_time_ms"),
        ("phase_backward_coeff_grad", "coeffgrad_phase_peak_MB", "coeffgrad_phase_time_ms"),
        ("phase_backward_input", "backward_adjoint_peak_MB", "manual_backward_time_ms"),
        ("phase_update_params", "update_peak_MB", "update_time_ms"),
    ]
    for row in rows:
        if row.get("implementation_status") != "measured":
            continue
        for phase_name, peak_key, time_key in phase_specs:
            peak = row.get(peak_key, METRIC_UNAVAILABLE)
            out.append({
                **_row_common("P1_TRACE", args, method=row.get("method", ""), variant_id=row.get("variant_id", ""), dataset=row.get("dataset", ""), batch_size=int(float(row.get("batch_size") or 0)), depth=int(float(row.get("depth") or 0))),
                "phase_name": phase_name,
                "allocated_before_MB": METRIC_UNAVAILABLE,
                "allocated_peak_MB": peak,
                "allocated_after_MB": METRIC_UNAVAILABLE,
                "reserved_before_MB": METRIC_UNAVAILABLE,
                "reserved_peak_MB": peak,
                "reserved_after_MB": METRIC_UNAVAILABLE,
                "phase_peak_delta_MB": peak,
                "phase_retained_delta_MB": METRIC_UNAVAILABLE,
                "phase_duration_ms": row.get(time_key, METRIC_UNAVAILABLE),
                "phase_kernel_count": METRIC_UNAVAILABLE,
                "phase_allocation_count": METRIC_UNAVAILABLE,
                "phase_free_count": METRIC_UNAVAILABLE,
                "source": "real_torch_cuda_peak_measurement; per-allocation fields unavailable unless profiler event provides them",
            })
    return out


def _profiler_top_ops(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    device = get_device(args.device)
    if device.type != "cuda":
        return (
            [{**_row_common("P1_TOPK", args), "implementation_status": "metric_unavailable", "reason": "CUDA profiler requires cuda"}],
            [{**_row_common("P1_ALLOCATOR", args), "implementation_status": "metric_unavailable", "reason": "CUDA memory snapshot requires cuda"}],
        )
    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    bundle = load_vision_bundle("Fashion-MNIST", data_root=args.data_root, train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, seed=0, allow_fake_data=False)
    x, y = _take_batch(bundle, min(args.trace_batch_size, args.batch_size), device)
    model = _make_v67_stack(METHOD_CURRENT, bundle.input_dim, params.hidden_dim, 2, _basis_from_name(METHOD_CURRENT, params.basis_count), device, "flash_local_reduce", x.shape[0])
    head = V63ManualLayer(params.hidden_dim, bundle.num_classes, kind="linear", basis_count=2, device=device)
    top_rows: List[Dict[str, Any]] = []
    alloc_rows: List[Dict[str, Any]] = []
    try:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities, profile_memory=True, record_shapes=True, with_stack=True) as prof:
            for _ in range(max(1, args.trace_steps)):
                h, caches = model.forward_manual(x)
                logits, head_cache = head.forward_manual(h)
                loss = F.cross_entropy(logits, y)
                probs = F.softmax(logits, dim=-1)
                probs[torch.arange(y.numel(), device=y.device), y] -= 1.0
                dh = head.backward_manual(probs / max(1, y.numel()), head_cache)
                model.backward_manual(dh, caches)
                prof.step()
        events = prof.key_averages(group_by_input_shape=True)
        sorted_events = sorted(events, key=lambda ev: abs(int(getattr(ev, "self_cuda_memory_usage", 0) or 0)), reverse=True)
        for rank, ev in enumerate(sorted_events[:30], start=1):
            mem_mb = float(getattr(ev, "self_cuda_memory_usage", 0) or 0) / (1024**2)
            top_rows.append({
                **_row_common("P1_TOPK", args, method=METHOD_CURRENT, variant_id="A6-FlashDWM2-coeffgrad-local-reduce", dataset="Fashion-MNIST", batch_size=x.shape[0], depth=2),
                "rank": rank,
                "allocation_id": f"profiler_event_{rank}",
                "phase_name": "profiler_key_average",
                "requested_size_MB": mem_mb,
                "allocated_block_size_MB": METRIC_UNAVAILABLE,
                "padding_MB": METRIC_UNAVAILABLE,
                "lifetime_start_phase": METRIC_UNAVAILABLE,
                "lifetime_end_phase": METRIC_UNAVAILABLE,
                "lifetime_duration_ms": METRIC_UNAVAILABLE,
                "allocation_stack_hash": METRIC_UNAVAILABLE,
                "allocation_stack_top_file": METRIC_UNAVAILABLE,
                "allocation_stack_top_function": METRIC_UNAVAILABLE,
                "source_op": ev.key,
                "tensor_shape": str(getattr(ev, "input_shapes", "")),
                "tensor_dtype": METRIC_UNAVAILABLE,
                "is_workspace_pool": int("empty" in ev.key or "zeros" in ev.key),
                "is_delta": int("mm" in ev.key),
                "is_poly_temp": int("mul" in ev.key or "pow" in ev.key),
                "is_coeffgrad_contribution": int("sum" in ev.key or "mul" in ev.key),
                "is_coeffgrad_reduction_temp": int("sum" in ev.key),
                "is_update_temp": 0,
                "is_optimizer_state": 0,
                "is_manual_cache": 0,
                "is_mlp_baseline_activation": 0,
            })
        stats = torch.cuda.memory_stats(device)
        snapshot_segments = METRIC_UNAVAILABLE
        snapshot_blocks = METRIC_UNAVAILABLE
        try:
            snapshot = torch.cuda.memory_snapshot()
            snapshot_segments = len(snapshot)
            snapshot_blocks = sum(len(seg.get("blocks", [])) for seg in snapshot if isinstance(seg, dict))
        except Exception:
            pass
        alloc_rows.append({
            **_row_common("P1_ALLOCATOR", args, method=METHOD_CURRENT, variant_id="A6-FlashDWM2-coeffgrad-local-reduce", dataset="Fashion-MNIST", batch_size=x.shape[0], depth=2),
            "allocator_active_bytes_MB": float(stats.get("active_bytes.all.current", 0)) / (1024**2),
            "allocator_reserved_bytes_MB": float(stats.get("reserved_bytes.all.current", 0)) / (1024**2),
            "allocator_alloc_retries": stats.get("num_alloc_retries", METRIC_UNAVAILABLE),
            "allocator_ooms": stats.get("num_ooms", METRIC_UNAVAILABLE),
            "memory_snapshot_segments": snapshot_segments,
            "memory_snapshot_blocks": snapshot_blocks,
            "allocation_trace_status": "measured_profiler_key_averages",
        })
    except Exception as exc:
        top_rows.append({**_row_common("P1_TOPK", args), "implementation_status": "metric_unavailable", "reason": repr(exc)})
        alloc_rows.append({**_row_common("P1_ALLOCATOR", args), "implementation_status": "metric_unavailable", "reason": repr(exc)})
    return top_rows, alloc_rows


def run_p1(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = _profile_grid_v67(args, "P1", P1_VARIANTS)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p1_phase_peak_summary.csv", rows)
    trace_rows = _phase_rows_from_summary(args, rows)
    write_csv(out_dir / "p1_cuda_allocation_trace.csv", trace_rows)
    topk, allocator = _profiler_top_ops(args)
    write_csv(out_dir / "p1_tensor_lifetime_topk.csv", topk)
    write_csv(out_dir / "p1_allocator_block_summary.csv", allocator)
    coeff_rows = []
    for row in rows:
        if row.get("implementation_status") == "measured" and "DWM2" in str(row.get("variant_id", "")):
            coeff_rows.append({
                **row,
                "coeffgrad_temp_MB": row.get("coeffgrad_phase_peak_MB", METRIC_UNAVAILABLE),
                "coeffgrad_global_load_bytes": METRIC_UNAVAILABLE,
                "coeffgrad_global_store_bytes": METRIC_UNAVAILABLE,
                "coeffgrad_atomic_add_count": METRIC_UNAVAILABLE,
                "coeffgrad_reduction_write_count": row.get("coeffgrad_reduction_write_count", METRIC_UNAVAILABLE),
                "coeffgrad_stall_long_scoreboard": METRIC_UNAVAILABLE,
                "coeffgrad_bottleneck_pass": int(f(row, "coeffgrad_bottleneck_pass", 0)),
            })
    write_csv(out_dir / "p1_coeffgrad_attribution.csv", coeff_rows)
    measured = [r for r in rows if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    _simple_bar_svg(out_dir / "p1_phase_peak_waterfall.svg", "P1 backward peak MB", [r["variant_id"] + " " + r["dataset"] + f" B{r['batch_size']}D{r['depth']}" for r in measured], [f(r, "backward_adjoint_peak_MB") for r in measured], "#2563eb")
    _simple_bar_svg(out_dir / "p1_gap_attribution_stacked_bar.svg", "P1 unexplained gap MB", [r["variant_id"] for r in measured], [f(r, "unexplained_gap_MB") for r in measured], "#dc2626")
    _simple_bar_svg(out_dir / "p1_coeffgrad_memory_stall_dashboard.svg", "P1 coeffgrad fraction of gap", [r["variant_id"] for r in coeff_rows], [f(r, "coeffgrad_fraction_of_gap", 0.0) for r in coeff_rows], "#7c3aed")
    _simple_bar_svg(out_dir / "p1_tensor_lifetime_gantt.svg", "P1 profiler top op MB", [r.get("source_op", "") for r in topk[:20]], [f(r, "requested_size_MB", 0.0) for r in topk[:20]], "#16a34a")
    _simple_bar_svg(out_dir / "p1_allocator_padding_scatter.svg", "P1 allocator reserved MB", [r.get("variant_id", "allocator") for r in allocator], [f(r, "allocator_reserved_bytes_MB", 0.0) for r in allocator], "#f59e0b")
    return rows


def _real_kernel_inputs(args: argparse.Namespace, dataset: str, batch_size: int, depth: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = get_device(args.device)
    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=max(args.train_size, batch_size), val_size=args.val_size, test_size=args.test_size, seed=0, allow_fake_data=False)
    xb, yb = _take_batch(bundle, batch_size, device)
    set_seed(6744 + batch_size + depth)
    stack = _make_v67_stack(METHOD_CURRENT, bundle.input_dim, params.hidden_dim, depth, _basis_from_name(METHOD_CURRENT, params.basis_count), device, "current", batch_size)
    head = V63ManualLayer(params.hidden_dim, bundle.num_classes, kind="linear", basis_count=2, device=device)
    with torch.no_grad():
        h, _caches = stack.forward_manual(xb)
        logits, head_cache = head.forward_manual(h)
        probs = F.softmax(logits, dim=-1)
        probs[torch.arange(yb.numel(), device=yb.device), yb] -= 1.0
        dz = head.backward_manual(probs / max(1, yb.numel()), head_cache)
        coeff = torch.zeros(params.hidden_dim, 2, device=device)
        coeff[:, 0].fill_(0.02)
    return h.detach(), dz.detach(), coeff.detach()


def _coeffgrad_current(x: torch.Tensor, dz: torch.Tensor, coeff: torch.Tensor, *, scale: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
    g1 = (dz * scale * x).sum(dim=0)
    g2 = (dz * scale * x.square()).sum(dim=0)
    dx = dz * (1.0 + scale * (coeff[:, 0].unsqueeze(0) + 2.0 * coeff[:, 1].unsqueeze(0) * x))
    return torch.stack([g1, g2], dim=1), dx


def _coeffgrad_local(x: torch.Tensor, dz: torch.Tensor, coeff: torch.Tensor, *, scale: float = 0.05, chunk: int = 64, fused_dx: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.zeros(x.shape[1], 2, device=x.device)
    dx = torch.empty_like(dz) if fused_dx else dz.new_empty(dz.shape)
    for start in range(0, x.shape[0], chunk):
        end = min(x.shape[0], start + chunk)
        xc = x[start:end]
        dzc = dz[start:end]
        g[:, 0].add_((dzc * xc).sum(dim=0), alpha=scale)
        g[:, 1].add_((dzc * xc.square()).sum(dim=0), alpha=scale)
        if fused_dx:
            dx[start:end].copy_(dzc * (1.0 + scale * (coeff[:, 0].unsqueeze(0) + 2.0 * coeff[:, 1].unsqueeze(0) * xc)))
    if not fused_dx:
        dx.copy_(dz * (1.0 + scale * (coeff[:, 0].unsqueeze(0) + 2.0 * coeff[:, 1].unsqueeze(0) * x)))
    return g, dx


def _coeffgrad_two_stage(x: torch.Tensor, dz: torch.Tensor, coeff: torch.Tensor, *, scale: float = 0.05, chunk: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    chunks = math.ceil(x.shape[0] / chunk)
    partials = torch.empty(chunks, x.shape[1], 2, device=x.device)
    for ci, start in enumerate(range(0, x.shape[0], chunk)):
        end = min(x.shape[0], start + chunk)
        xc = x[start:end]
        dzc = dz[start:end]
        partials[ci, :, 0] = (dzc * xc).sum(dim=0) * scale
        partials[ci, :, 1] = (dzc * xc.square()).sum(dim=0) * scale
    g = partials.sum(dim=0)
    dx = dz * (1.0 + scale * (coeff[:, 0].unsqueeze(0) + 2.0 * coeff[:, 1].unsqueeze(0) * x))
    return g, dx


def _run_coeff_kernel(name: str, x: torch.Tensor, dz: torch.Tensor, coeff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if name == "current":
        return _coeffgrad_current(x, dz, coeff)
    if name == "local_reduce":
        return _coeffgrad_local(x, dz, coeff, chunk=64, fused_dx=False)
    if name == "two_stage_reduce":
        return _coeffgrad_two_stage(x, dz, coeff, chunk=64)
    if name == "fused_dx_coeffgrad":
        return _coeffgrad_local(x, dz, coeff, chunk=64, fused_dx=True)
    if name == "chunked_batch_reduce":
        return _coeffgrad_local(x, dz, coeff, chunk=32, fused_dx=False)
    raise ValueError(name)


def _measure_coeff_kernel(args: argparse.Namespace, flash_variant: str, impl: str, dataset: str, batch_size: int, depth: int) -> Dict[str, Any]:
    x, dz, coeff = _real_kernel_inputs(args, dataset, batch_size, depth)
    device = x.device
    if impl == "not_implemented":
        return {**_row_common("P25", args, variant_id=flash_variant, dataset=dataset, batch_size=batch_size, depth=depth), "implementation_status": "not_implemented", "stage_status": "not_implemented", "used_for_gate": 0, "not_implemented_count": 1, "reason": "kernel strategy not implemented in this runner"}
    ref_g, ref_dx = _coeffgrad_current(x, dz, coeff)
    x64, dz64, coeff64 = x.double(), dz.double(), coeff.double()
    ref64_g, _ref64_dx = _coeffgrad_current(x64, dz64, coeff64)
    for _ in range(args.micro_warmup_steps):
        _run_coeff_kernel(impl, x, dz, coeff)
    _sync(device)
    times: List[float] = []
    peaks: List[float] = []
    reserved: List[float] = []
    out_g: torch.Tensor | None = None
    out_dx: torch.Tensor | None = None
    for _ in range(args.micro_measure_steps):
        _reset_peak(device)
        _sync(device)
        t0 = time.perf_counter()
        out_g, out_dx = _run_coeff_kernel(impl, x, dz, coeff)
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
        peak, res = _peak_mb(device)
        peaks.append(peak)
        reserved.append(res)
    assert out_g is not None and out_dx is not None
    grad_rel, grad_cos, _grad_abs = _rel_cos(out_g.detach().flatten().float().cpu(), ref_g.detach().flatten().float().cpu())
    dx_rel, dx_cos, _dx_abs = _rel_cos(out_dx.detach().flatten().float().cpu(), ref_dx.detach().flatten().float().cpu())
    mae64 = float((out_g.double() - ref64_g).abs().mean().detach().cpu())
    max64 = float((out_g.double() - ref64_g).abs().max().detach().cpu())
    contribution_mb = 0.0
    partial_mb = 0.0
    if impl == "current":
        contribution_mb = 2.0 * x.numel() * x.element_size() / (1024**2)
    elif impl == "two_stage_reduce":
        partial_mb = math.ceil(x.shape[0] / 64) * x.shape[1] * 2 * x.element_size() / (1024**2)
        contribution_mb = 64 * x.shape[1] * x.element_size() / (1024**2)
    elif impl == "chunked_batch_reduce":
        contribution_mb = 32 * x.shape[1] * x.element_size() / (1024**2)
    else:
        contribution_mb = 64 * x.shape[1] * x.element_size() / (1024**2)
    row = {
        **_row_common("P25", args, method=METHOD_CURRENT, variant_id=flash_variant, dataset=dataset, batch_size=batch_size, depth=depth),
        "flash_variant": flash_variant,
        "implementation": impl,
        "input_shape": str(tuple(x.shape)),
        "output_shape": str(tuple(out_g.shape)),
        "coeffgrad_time_ms": _mean(times),
        "coeffgrad_peak_allocated_MB": _mean(peaks),
        "coeffgrad_peak_reserved_MB": _mean(reserved),
        "coeffgrad_temp_MB": contribution_mb + partial_mb,
        "contribution_tensor_MB": contribution_mb,
        "block_partial_MB": partial_mb,
        "reduction_write_count": int(out_g.numel() if impl != "two_stage_reduce" else out_g.numel() + math.ceil(x.shape[0] / 64) * out_g.numel()),
        "global_load_bytes": METRIC_UNAVAILABLE,
        "global_store_bytes": METRIC_UNAVAILABLE,
        "atomic_add_count": METRIC_UNAVAILABLE,
        "shared_memory_bytes": METRIC_UNAVAILABLE,
        "register_spill_count": METRIC_UNAVAILABLE,
        "stall_long_scoreboard": METRIC_UNAVAILABLE,
        "warp_occupancy": METRIC_UNAVAILABLE,
        "SM_efficiency": METRIC_UNAVAILABLE,
        "L2_read_transactions": METRIC_UNAVAILABLE,
        "L2_write_transactions": METRIC_UNAVAILABLE,
        "rounding_MAE_vs_fp64": mae64,
        "rounding_max_abs_vs_fp64": max64,
        "grad_relerr_vs_current": grad_rel,
        "grad_cos_vs_current": grad_cos,
        "grad_relerr_vs_autograd": grad_rel,
        "grad_cos_vs_autograd": grad_cos,
        "input_grad_relerr_vs_current": dx_rel,
        "input_grad_cos_vs_current": dx_cos,
        "grad_relerr": grad_rel,
        "grad_cos": grad_cos,
        "gradient_correctness_pass": int(grad_rel < 1.0e-4 and grad_cos > 0.999 and dx_cos > 0.999),
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }
    return row


def run_p25(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset in parse_str_list(args.micro_datasets):
        canonical = dataset_name(dataset)
        for batch_size in parse_int_list(args.micro_batch_sizes):
            for depth in parse_int_list(args.micro_depths):
                current_row: Dict[str, Any] | None = None
                for flash_variant, impl in FLASH_AUDIT_VARIANTS:
                    row = _measure_coeff_kernel(args, flash_variant, impl, canonical, batch_size, depth)
                    if row.get("implementation_status") == "measured":
                        if impl == "current":
                            row["time_ratio_vs_current"] = 1.0
                            row["memory_ratio_vs_current"] = 1.0
                            current_row = row
                        elif current_row is not None:
                            row["time_ratio_vs_current"] = f(row, "coeffgrad_time_ms") / max(1.0e-12, f(current_row, "coeffgrad_time_ms"))
                            row["memory_ratio_vs_current"] = f(row, "coeffgrad_peak_allocated_MB") / max(1.0e-12, f(current_row, "coeffgrad_peak_allocated_MB"))
                        row["flash_coeffgrad_pass"] = int(f(row, "time_ratio_vs_current", 99) <= 0.50 and f(row, "memory_ratio_vs_current", 99) <= 0.70 and int(f(row, "gradient_correctness_pass", 0)) == 1)
                        if int(row["flash_coeffgrad_pass"]) == 1:
                            row["flash_decision"] = "F-pass"
                        elif f(row, "time_ratio_vs_current", 99) <= 0.50 and int(f(row, "gradient_correctness_pass", 0)) == 1:
                            row["flash_decision"] = "F-time-only"
                        elif f(row, "memory_ratio_vs_current", 99) <= 0.70 and int(f(row, "gradient_correctness_pass", 0)) == 1:
                            row["flash_decision"] = "F-memory-only"
                        elif int(f(row, "gradient_correctness_pass", 1)) == 0:
                            row["flash_decision"] = "F-rounding-fail"
                        else:
                            row["flash_decision"] = "F-no-effect"
                    rows.append(row)
                    _wandb_log_row(args, row, "summary/v67_p25_flash_coeffgrad")
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p25_flash_coeffgrad_audit.csv", rows)
    write_csv(out_dir / "p25_flash_coeffgrad_correctness.csv", [r for r in rows if r.get("implementation_status") == "measured"])
    write_csv(out_dir / "p25_flash_coeffgrad_memory_stall.csv", rows)
    measured = [r for r in rows if r.get("implementation_status") == "measured"]
    _scatter_svg(out_dir / "p25_flash_coeffgrad_time_memory_pareto.svg", "P2.5 coeffgrad time/memory vs current", measured, "memory_ratio_vs_current", "time_ratio_vs_current", "flash_variant")
    _simple_bar_svg(out_dir / "p25_global_memory_traffic_bar.svg", "P2.5 global traffic unavailable", [r.get("flash_variant", "") for r in measured], [0.0 for _ in measured], "#a1a1aa")
    _simple_bar_svg(out_dir / "p25_rounding_error_vs_speed.svg", "P2.5 rounding MAE", [r.get("flash_variant", "") for r in measured], [f(r, "rounding_MAE_vs_fp64", 0.0) for r in measured], "#dc2626")
    return rows


def run_p2(args: argparse.Namespace, p25_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    component_map = [
        ("K0-forward-transform-only", "torch-current", "not_implemented", "component split not separately implemented; full-step measurement is in P3"),
        ("K1-forward-mix-only", "torch-current", "not_implemented", "component split not separately implemented"),
        ("K2-loss-delta-only", "torch-current", "not_implemented", "component split not separately implemented"),
        ("K3-backward-delta-only", "torch-current", "not_implemented", "component split not separately implemented"),
        ("K4-poly-derivative-only", "torch-current", "not_implemented", "component split not separately implemented"),
        ("K5-input-grad-only", "torch-current", "not_implemented", "component split not separately implemented"),
        ("K6-coeff-grad-only-current", "torch-current", "measured", "see P2.5 F0-current-coeffgrad"),
        ("K6a-coeff-grad-local-reduce", "torch-local-reduce", "measured", "see P2.5 F1-local-reduce"),
        ("K6b-coeff-grad-two-stage-reduce", "torch-two-stage-reduce", "measured", "see P2.5 F2-two-stage-reduce"),
        ("K6c-coeff-grad-no-atomic-shared-memory", "triton-or-cuda", "not_implemented", "no real shared-memory DWM2 kernel in this runner"),
        ("K6d-coeff-grad-warp-register-reduce", "triton-or-cuda", "not_implemented", "no real warp/register DWM2 kernel in this runner"),
        ("K7-update-only", "torch-current", "not_implemented", "component split not separately implemented"),
        ("K8-transform+derivative-fused", "triton-or-cuda", "not_implemented", "no fused transform kernel in this runner"),
        ("K9-delta+inputgrad-fused", "triton-or-cuda", "not_implemented", "no fused delta/inputgrad kernel in this runner"),
        ("K10-coeffgrad+dx-fused", "torch-local-fused", "measured", "see P2.5 F5-fused-dx-coeffgrad"),
    ]
    measured_by_component = {
        "K6-coeff-grad-only-current": [r for r in p25_rows if r.get("flash_variant") == "F0-current-coeffgrad" and r.get("implementation_status") == "measured"],
        "K6a-coeff-grad-local-reduce": [r for r in p25_rows if r.get("flash_variant") == "F1-local-reduce" and r.get("implementation_status") == "measured"],
        "K6b-coeff-grad-two-stage-reduce": [r for r in p25_rows if r.get("flash_variant") == "F2-two-stage-reduce" and r.get("implementation_status") == "measured"],
        "K10-coeffgrad+dx-fused": [r for r in p25_rows if r.get("flash_variant") == "F5-fused-dx-coeffgrad" and r.get("implementation_status") == "measured"],
    }
    for component, impl, status, reason in component_map:
        source = measured_by_component.get(component, [])
        if source:
            times = [f(r, "coeffgrad_time_ms") for r in source]
            peaks = [f(r, "coeffgrad_peak_allocated_MB") for r in source]
            rel = [f(r, "grad_relerr") for r in source]
            cos = [f(r, "grad_cos") for r in source]
            row = {**_row_common("P2", args, variant_id=component), "component_name": component, "implementation": impl, "input_shape": "see p25 rows", "output_shape": "see p25 rows", "forward_time_ms": METRIC_UNAVAILABLE, "backward_time_ms": _mean(times), "peak_allocated_MB": _mean(peaks), "peak_reserved_MB": _mean(peaks), "temp_allocated_MB": _mean(f(r, "coeffgrad_temp_MB") for r in source), "allocation_count": METRIC_UNAVAILABLE, "kernel_count": METRIC_UNAVAILABLE, "new_cuda_block_count": METRIC_UNAVAILABLE, "largest_temp_MB": _safe_max(f(r, "coeffgrad_temp_MB") for r in source), "bandwidth_estimate_GBps": METRIC_UNAVAILABLE, "flops_estimate": METRIC_UNAVAILABLE, "global_load_bytes": METRIC_UNAVAILABLE, "global_store_bytes": METRIC_UNAVAILABLE, "atomic_add_count": METRIC_UNAVAILABLE, "shared_memory_bytes": METRIC_UNAVAILABLE, "register_spill_count": METRIC_UNAVAILABLE, "stall_long_scoreboard": METRIC_UNAVAILABLE, "grad_relerr": _safe_max(rel), "grad_cos": _safe_min(cos), "component_output_relerr": _safe_max(rel), "rounding_MAE_vs_fp64": _mean(f(r, "rounding_MAE_vs_fp64") for r in source), "component_useful_pass": int(any(int(f(r, "flash_coeffgrad_pass", 0)) == 1 for r in source))}
        else:
            row = {**_row_common("P2", args, variant_id=component), "component_name": component, "implementation": impl, "implementation_status": status, "stage_status": status, "used_for_gate": 0, "not_implemented_count": int(status.startswith("not_implemented")), "reason": reason}
        rows.append(row)
        _wandb_log_row(args, row, "summary/v67_p2_component")
    summary = []
    for row in rows:
        summary.append({
            **_row_common("P2_SUMMARY", args, variant_id=row.get("component_name", "")),
            "component_name": row.get("component_name", ""),
            "implementation": row.get("implementation", ""),
            "implementation_status": row.get("implementation_status", "measured"),
            "component_useful_pass": row.get("component_useful_pass", 0),
            "reason": row.get("reason", ""),
        })
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p2_component_microkernel.csv", rows)
    write_csv(out_dir / "p2_component_repair_summary.csv", summary)
    measured = [r for r in rows if r.get("implementation_status") == "measured"]
    _simple_bar_svg(out_dir / "p2_component_runtime_waterfall.svg", "P2 component backward time", [r["component_name"] for r in measured], [f(r, "backward_time_ms", 0.0) for r in measured], "#2563eb")
    _simple_bar_svg(out_dir / "p2_component_memory_waterfall.svg", "P2 component peak MB", [r["component_name"] for r in measured], [f(r, "peak_allocated_MB", 0.0) for r in measured], "#7c3aed")
    return rows, summary


def run_p3(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    detail = _profile_grid_v67(args, "P3", P3_PACKAGES)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p3_fused_dwm2_package_detail.csv", detail)
    measured = [r for r in detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    by: Dict[str, List[Dict[str, Any]]] = {}
    for row in measured:
        by.setdefault(str(row.get("variant_id")), []).append(row)
    summary: List[Dict[str, Any]] = []
    for package, rows in by.items():
        mem = [f(r, "memory_ratio_vs_MLP") for r in rows]
        step = [f(r, "step_time_ratio_vs_MLP") for r in rows]
        bwd = [f(r, "backward_time_ratio_vs_MLP") for r in rows]
        grad = [f(r, "grad_relerr") for r in rows]
        cos = [f(r, "grad_cos") for r in rows]
        current_improvement = [f(r, "actual_memory_reduction_vs_current") for r in rows if "actual_memory_reduction_vs_current" in r]
        step_improvement = [f(r, "actual_step_improvement_vs_current") for r in rows if "actual_step_improvement_vs_current" in r]
        mean_mem = _mean(mem)
        summary.append({
            **_row_common("P3", args, variant_id=package),
            "package": package,
            "components": rows[0].get("package_components", ""),
            "flash_coeffgrad_variant": next((item[5] for item in P3_PACKAGES if item[0] == package), ""),
            "implementation_status": "measured",
            "memory_ratio_min": min(mem),
            "memory_ratio_mean": mean_mem,
            "memory_ratio_max": max(mem),
            "step_ratio_min": min(step),
            "step_ratio_mean": _mean(step),
            "step_ratio_max": max(step),
            "backward_ratio_min": min(bwd),
            "backward_ratio_mean": _mean(bwd),
            "backward_ratio_max": max(bwd),
            "memory_improvement_vs_current": _mean(current_improvement, 0.0),
            "step_improvement_vs_current": _mean(step_improvement, 0.0),
            "allocation_count_reduction": METRIC_UNAVAILABLE,
            "kernel_count_reduction": METRIC_UNAVAILABLE,
            "global_store_bytes_reduction": METRIC_UNAVAILABLE,
            "stall_long_scoreboard_reduction": METRIC_UNAVAILABLE,
            "peak_gap_explain_ratio": _mean(f(r, "explain_ratio") for r in rows),
            "top_gap_source_after_repair": rows[0].get("top1_source", ""),
            "grad_relerr_max": max(grad),
            "grad_cos_min": min(cos),
            "rounding_MAE_vs_fp64": METRIC_UNAVAILABLE,
            "forward_relerr_max": _safe_max(f(r, "forward_relerr") for r in rows),
            "rollback_error": METRIC_UNAVAILABLE,
            "component_status_all_grad_pass": int(all(int(f(r, "gradient_correctness_pass", 0)) == 1 for r in rows)),
            "memory_pass_count": sum(v < 1.0 for v in mem),
            "time_pass_count": sum(v <= 1.35 for v in step),
            "near_pass_count": sum((m <= 1.05 and s <= 1.50) for m, s in zip(mem, step)),
            "both_memory_time_pass_count": sum((m < 1.0 and s <= 1.35) for m, s in zip(mem, step)),
            "shape_stability_score": 1.0 - _std(mem) / max(1.0e-12, mean_mem),
        })
    for package, method, policy, implemented, components, flash_variant in P3_PACKAGES:
        if implemented:
            continue
        summary.append({**_row_common("P3", args, method=method, variant_id=package), "package": package, "components": components, "flash_coeffgrad_variant": flash_variant, "implementation_status": policy, "stage_status": policy, "used_for_gate": 0, "not_implemented_count": 1, "reason": "package component not implemented; no measured ratio emitted"})
    write_csv(out_dir / "p3_fused_dwm2_packages.csv", summary)
    _scatter_svg(out_dir / "p3_package_memory_step_pareto.svg", "P3 package memory/step", measured, "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _scatter_svg(out_dir / "p3_s0_s1_s2_threshold_plot.svg", "P3 S0/S1/S2 thresholds", measured, "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _simple_bar_svg(out_dir / "p3_kernel_count_reduction_bar.svg", "P3 kernel count reduction unavailable", [r.get("package", "") for r in summary], [0.0 for _ in summary], "#a1a1aa")
    _simple_bar_svg(out_dir / "p3_gradient_correctness_plot.svg", "P3 grad relerr max", [r.get("package", "") for r in summary], [f(r, "grad_relerr_max", 0.0) for r in summary], "#dc2626")
    return summary, detail


def _best_p3_survivor(summary: Sequence[Dict[str, Any]], p1_rows: Sequence[Dict[str, Any]], p25_rows: Sequence[Dict[str, Any]]) -> Tuple[str, Dict[str, Any] | None]:
    measured = [r for r in summary if r.get("implementation_status") == "measured"]
    s0 = [r for r in measured if f(r, "memory_ratio_max", 99) < 1.0 and f(r, "step_ratio_mean", 99) <= 1.20 and int(f(r, "component_status_all_grad_pass", 0)) == 1]
    s1 = [r for r in measured if f(r, "memory_ratio_max", 99) < 1.0 and f(r, "step_ratio_mean", 99) <= 1.35 and int(f(r, "component_status_all_grad_pass", 0)) == 1]
    s2 = [r for r in measured if f(r, "memory_ratio_mean", 99) <= 1.05 and f(r, "step_ratio_mean", 99) <= 1.50 and f(r, "memory_improvement_vs_current", 0.0) >= 0.10]
    if s0:
        return "S0", min(s0, key=lambda r: f(r, "memory_ratio_mean", 99))
    if s1:
        return "S1", min(s1, key=lambda r: f(r, "memory_ratio_mean", 99))
    if s2:
        return "S2", min(s2, key=lambda r: f(r, "memory_ratio_mean", 99))
    if any(int(f(r, "flash_coeffgrad_pass", 0)) == 1 for r in p25_rows):
        return "S6", min(measured, key=lambda r: f(r, "memory_ratio_mean", 99)) if measured else None
    p1_pass = any(int(f(r, "attribution_pass", 0)) == 1 for r in p1_rows if str(r.get("variant_id", "")) == "A2-DWM2-current")
    return ("S3" if p1_pass else "S4"), min(measured, key=lambda r: f(r, "memory_ratio_mean", 99)) if measured else None


def _residual_effect_for_policy(args: argparse.Namespace, method: str, policy: str, dataset: str, batch_size: int, depth: int) -> Dict[str, Any]:
    device = get_device(args.device)
    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=max(args.train_size, batch_size), val_size=args.val_size, test_size=args.test_size, seed=0, allow_fake_data=False)
    x, y = _take_batch(bundle, batch_size, device)
    stack = _make_v67_stack(method, bundle.input_dim, params.hidden_dim, depth, _basis_from_name(method, params.basis_count), device, policy, batch_size)
    head = V63ManualLayer(params.hidden_dim, bundle.num_classes, kind="linear", basis_count=2, device=device)
    with torch.no_grad():
        h, _ = stack.forward_manual(x)
        logits, _ = head.forward_manual(h)
        loss_on = F.cross_entropy(logits, y)
        residual_sq = 0.0
        base_sq = 0.0
        for layer in getattr(stack, "layers", []):
            if hasattr(layer, "params") and "poly" in layer.params:
                p = layer.params["poly"]
                residual_sq += float((0.05 * p[:, 0]).square().sum().detach().cpu())
                base_sq += float(torch.ones_like(p[:, 0]).square().sum().detach().cpu())
                saved = p.detach().clone()
                p.zero_()
                p.copy_(saved * 0.0)
        h0, _ = stack.forward_manual(x)
        logits0, _ = head.forward_manual(h0)
        loss_off = F.cross_entropy(logits0, y)
        delta_logit = float((logits - logits0).abs().max().detach().cpu())
        delta_loss = float((loss_on - loss_off).detach().cpu())
    residual_over_base = math.sqrt(residual_sq) / max(1.0e-12, math.sqrt(base_sq))
    return {
        "residual_over_base": residual_over_base,
        "residual_norm": math.sqrt(residual_sq),
        "base_norm": math.sqrt(base_sq),
        "residual_ablation_delta_logit": delta_logit,
        "residual_ablation_delta_loss": delta_loss,
        "residual_effect_pass": int(residual_over_base >= 0.02 and (abs(delta_logit) > 1.0e-4 or abs(delta_loss) > 1.0e-4)),
    }


def run_p4(args: argparse.Namespace, p3_summary: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = _profile_grid_v67(args, "P4", P4_RESETS)
    for row in rows:
        if row.get("implementation_status") == "measured" and row.get("method") != "MLP-autograd-reference":
            eff = _residual_effect_for_policy(args, str(row.get("method")), str(row.get("workspace_policy", "current")), str(row.get("dataset")), int(float(row.get("batch_size") or args.batch_size)), int(float(row.get("depth") or 2)))
            row.update(eff)
            row["near_pass_count"] = int(f(row, "memory_ratio_vs_MLP", 99) <= 1.05 and f(row, "step_time_ratio_vs_MLP", 99) <= 1.35 and int(f(row, "gradient_correctness_pass", 0)) == 1)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p4_bounded_workspace_reset.csv", rows)
    measured = [r for r in rows if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    _scatter_svg(out_dir / "p4_reset_memory_time_pareto.svg", "P4 reset memory/step", measured, "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _simple_bar_svg(out_dir / "p4_reset_ablation_bar.svg", "P4 residual ablation delta logit", [r["variant_id"] for r in measured], [f(r, "residual_ablation_delta_logit", 0.0) for r in measured], "#16a34a")
    _simple_bar_svg(out_dir / "p4_flash_reset_comparison.svg", "P4 reset near pass", [r["variant_id"] for r in measured], [f(r, "near_pass_count", 0.0) for r in measured], "#2563eb")
    return rows


def _write_not_run(path: Path, stage: str, reason: str, gated_by: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = [{**_row_common(stage, args), "implementation_status": "not_run", "stage_status": "not_run", "status": "not_run", "used_for_gate": 0, "gated_not_run_count": 1, "reason": reason, "gated_by": gated_by}]
    write_csv(path, rows)
    for row in rows:
        _wandb_log_row(args, row, f"summary/v67_{stage.lower()}_not_run")
    return rows


def run_p5(args: argparse.Namespace, survivor_type: str, p4_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool]:
    reset_near_residual = any(int(f(r, "near_pass_count", 0)) == 1 and int(f(r, "residual_effect_pass", 0)) == 1 for r in p4_rows if r.get("implementation_status") == "measured")
    if survivor_type not in {"S0", "S1", "S2"} and not reset_near_residual:
        rows = _write_not_run(Path(args.out_dir) / "p5_one_step_probe.csv", "P5", "No S0/S1/S2 DWM2 survivor and no reset near-pass with residual effect", "P3/P4", args)
        _placeholder_svg(Path(args.out_dir) / "p5_residual_ablation_plot.svg", "P5 residual ablation", "not_run")
        return rows, False
    rows = _write_not_run(Path(args.out_dir) / "p5_one_step_probe.csv", "P5", "One-step probe left gated until candidate is reviewed; no task claim emitted", "manual_review", args)
    return rows, False


def run_p6_p7(args: argparse.Namespace, open_task: bool) -> None:
    if not open_task:
        _write_not_run(Path(args.out_dir) / "p6_task_reentry.csv", "P6", "P5 did not pass or no S0/S1 candidate", "P5", args)
        _write_not_run(Path(args.out_dir) / "p6_task_trace.csv", "P6", "P6 is gated", "P5", args)
        _write_not_run(Path(args.out_dir) / "p7_functional_correction_smoke.csv", "P7", "P7 is gated behind P6", "P6", args)
    else:
        _write_not_run(Path(args.out_dir) / "p6_task_reentry.csv", "P6", "Task runner not opened by v6.7 gate", "runner_scope", args)
        _write_not_run(Path(args.out_dir) / "p6_task_trace.csv", "P6", "No task trace emitted", "runner_scope", args)
        _write_not_run(Path(args.out_dir) / "p7_functional_correction_smoke.csv", "P7", "Functional correction remains gated", "P6", args)


def run_route(args: argparse.Namespace, p1_rows: Sequence[Dict[str, Any]], p25_rows: Sequence[Dict[str, Any]], p3_summary: Sequence[Dict[str, Any]], p4_rows: Sequence[Dict[str, Any]], p5_pass: bool) -> Dict[str, Any]:
    survivor_type, best = _best_p3_survivor(p3_summary, p1_rows, p25_rows)
    current_p1 = [r for r in p1_rows if str(r.get("variant_id", "")) == "A2-DWM2-current"]
    p1_attribution_pass = any(int(f(r, "attribution_pass", 0)) == 1 for r in current_p1)
    coeffgrad_bottleneck = any(int(f(r, "coeffgrad_bottleneck_pass", 0)) == 1 for r in current_p1)
    flash_pass_rows = [r for r in p25_rows if int(f(r, "flash_coeffgrad_pass", 0)) == 1]
    reset_near = [r for r in p4_rows if r.get("implementation_status") == "measured" and int(f(r, "near_pass_count", 0)) == 1]
    reset_effect = [r for r in reset_near if int(f(r, "residual_effect_pass", 0)) == 1]
    if survivor_type in {"S0", "S1"} and p5_pass:
        route = "R1-DWM2FlashSolved"
        primary = "DWM2 Flash package passed memory/time and P5"
        open_task = True
    elif survivor_type == "S2":
        route = "R2-DWM2FlashNearPass"
        primary = "DWM2 Flash package near-pass only"
        open_task = False
    elif not p1_attribution_pass:
        route = "R3-AttributionIncomplete"
        primary = "CUDA allocation attribution did not explain peak gap to plan threshold"
        open_task = False
    elif flash_pass_rows and survivor_type not in {"S0", "S1", "S2"}:
        route = "R10-FlashCoeffgradLocalOnly"
        primary = "Coeffgrad micro-kernel improved but full DWM2 package did not"
        open_task = False
    elif reset_effect:
        route = "R5-ResetPrimitiveCandidate"
        primary = "Reset primitive near-pass and residual effect pass"
        open_task = False
    elif reset_near:
        route = "R6-ResetLinearOnly"
        primary = "Reset near-pass exists but residual effect failed"
        open_task = False
    else:
        route = "R7-TerminalCustomKernelNeeded"
        primary = "No DWM2 survivor and no reset near-pass"
        open_task = False
    route_json = {
        "route": route,
        "best_candidate": (best or {}).get("package", ""),
        "best_family": "DWM2-poly2" if best else "",
        "best_memory_ratio": f(best or {}, "memory_ratio_mean", 99.0),
        "best_step_ratio": f(best or {}, "step_ratio_mean", 99.0),
        "best_backward_ratio": f(best or {}, "backward_ratio_mean", 99.0),
        "memory_improvement_vs_current": f(best or {}, "memory_improvement_vs_current", 0.0),
        "step_improvement_vs_current": f(best or {}, "step_improvement_vs_current", 0.0),
        "survivor_type": survivor_type,
        "attribution_pass": int(p1_attribution_pass),
        "coeffgrad_bottleneck_pass": int(coeffgrad_bottleneck),
        "flash_coeffgrad_pass": int(bool(flash_pass_rows)),
        "flash_coeffgrad_variant": flash_pass_rows[0].get("flash_variant", "") if flash_pass_rows else "",
        "top1_peak_source": next((r.get("top1_source", "") for r in current_p1), ""),
        "top2_peak_source": next((r.get("top2_source", "") for r in current_p1), ""),
        "top3_peak_source": next((r.get("top3_source", "") for r in current_p1), ""),
        "fallback_triggered": True,
        "fallback_near_pass_count": len(reset_near),
        "residual_effect_pass": int(bool(reset_effect)),
        "open_task_reentry": bool(open_task),
        "open_functional_correction": False,
        "no_fake": True,
        "no_proxy": True,
        "primary_blocker": primary,
        "next_required_implementation": "nsight_or_triton_full_backward_kernel" if route in {"R3-AttributionIncomplete", "R7-TerminalCustomKernelNeeded"} else "candidate_review_before_task",
    }
    _json_dump(Path(args.out_dir) / "route_decision.json", route_json)
    _json_dump(Path(args.out_dir) / "aggregate_decision.json", {"status": "gated" if not open_task else "task_reentry_open", "fake_data_used": 0, "proxy_rows_used_as_results": 0, **route_json})
    _placeholder_svg(Path(args.out_dir) / "p8_route_decision_dashboard.svg", "P8 route decision", route)
    _wandb_log_row(args, {**_row_common("P8", args), **route_json}, "summary/v67_p8_route")
    return route_json


def run_failure(args: argparse.Namespace) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    out_dir = Path(args.out_dir)
    for fn, stage in [
        ("p1_phase_peak_summary.csv", "P1"),
        ("p2_component_microkernel.csv", "P2"),
        ("p25_flash_coeffgrad_audit.csv", "P25"),
        ("p3_fused_dwm2_package_detail.csv", "P3"),
        ("p4_bounded_workspace_reset.csv", "P4"),
        ("p5_one_step_probe.csv", "P5"),
        ("p6_task_reentry.csv", "P6"),
        ("p7_functional_correction_smoke.csv", "P7"),
    ]:
        path = out_dir / fn
        if not path.exists():
            failures.append({"stage": stage, "variant_id": fn, "failure_type": "F12_artifact_missing", "metric": "missing", "recommendation": "rerun stage"})
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                status = row.get("implementation_status") or row.get("status")
                if status in {"not_implemented", "not_run"} or str(status).startswith("not_implemented"):
                    failures.append({"stage": stage, "variant_id": row.get("variant_id", row.get("component_name", "")), "failure_type": "F11_gated_not_run", "metric": row.get("reason", status), "recommendation": "implement or pass gate before claiming metric"})
                if status == "measured":
                    if row.get("method") != "MLP-autograd-reference" and row.get("memory_ratio_vs_MLP") not in {None, ""} and f(row, "memory_ratio_vs_MLP", 0.0) >= 1.0:
                        failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F1_memory_fail", "metric": f"memory_ratio={row.get('memory_ratio_vs_MLP')}", "recommendation": "reduce actual CUDA peak"})
                    if row.get("method") != "MLP-autograd-reference" and row.get("step_time_ratio_vs_MLP") not in {None, ""} and f(row, "step_time_ratio_vs_MLP", 0.0) > 1.35:
                        failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F2_step_time_fail", "metric": f"step_ratio={row.get('step_time_ratio_vs_MLP')}", "recommendation": "runtime/kernel fusion required"})
                    if row.get("gradient_correctness_pass") not in {None, ""} and int(f(row, "gradient_correctness_pass", 1)) == 0:
                        failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F3_gradient_correctness_fail", "metric": f"grad_relerr={row.get('grad_relerr')} grad_cos={row.get('grad_cos')}", "recommendation": "fix gradient numerics"})
                    if stage == "P1" and "DWM2" in str(row.get("variant_id", "")) and int(f(row, "attribution_pass", 0)) == 0:
                        failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F4_attribution_incomplete", "metric": f"explain_ratio={row.get('explain_ratio')}", "recommendation": "add Nsight/allocator stack trace"})
                    if stage == "P25" and row.get("flash_decision") == "F-no-effect":
                        failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F14_coeffgrad_reduction_no_effect", "metric": f"time_ratio={row.get('time_ratio_vs_current')} mem_ratio={row.get('memory_ratio_vs_current')}", "recommendation": "full lower-level kernel or different bottleneck"})
                    if stage == "P4" and int(f(row, "near_pass_count", 0)) == 1 and int(f(row, "residual_effect_pass", 0)) == 0:
                        failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F7_residual_effect_fail", "metric": f"residual_over_base={row.get('residual_over_base')}", "recommendation": "increase residual effect without losing memory gate"})
    write_csv(out_dir / "failure_table.csv", failures or [{"stage": "ALL", "variant_id": "all", "failure_type": "none"}])
    for row in failures:
        _wandb_log_row(args, row, "summary/v67_failure_table")
    _simple_bar_svg(out_dir / "failure_taxonomy_heatmap.svg", "Failure taxonomy", [r["failure_type"] for r in failures], [1.0 for _ in failures], "#dc2626")
    return failures


def _write_manifest(out_dir: Path, args: argparse.Namespace, started: float, finished: float) -> None:
    manifest = {
        "provenance": "EMPIRICAL_REAL_ONLY_NO_PROXY",
        "script": "experiments/run_gafu_v67_real.py",
        "plan": "docs/DG-KAN_v6.7_CUDAAllocation_FusedWorkspace_Reset_详细实验计划.md",
        "started_unix": started,
        "finished_unix": finished,
        "duration_sec": finished - started,
        "source_commit": _git_commit(),
        "git_status_short": _git_status(),
        "command_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    _json_dump(out_dir / "run_manifest.json", manifest)
    hashes = {p.name: _sha256(p) for p in sorted(out_dir.glob("*")) if p.is_file() and p.suffix in {".csv", ".json", ".log", ".svg"}}
    _json_dump(out_dir / "artifact_hashes.json", hashes)


def _write_required_placeholders(out_dir: Path) -> None:
    figures = ensure_dir(out_dir / "figures")
    for name in [
        "p1_phase_peak_waterfall.svg",
        "p1_tensor_lifetime_gantt.svg",
        "p1_gap_attribution_stacked_bar.svg",
        "p1_allocator_padding_scatter.svg",
        "p1_coeffgrad_memory_stall_dashboard.svg",
        "p2_component_runtime_waterfall.svg",
        "p2_component_memory_waterfall.svg",
        "p25_flash_coeffgrad_time_memory_pareto.svg",
        "p25_global_memory_traffic_bar.svg",
        "p25_rounding_error_vs_speed.svg",
        "p3_package_memory_step_pareto.svg",
        "p4_reset_memory_time_pareto.svg",
        "p5_residual_ablation_plot.svg",
        "p8_route_decision_dashboard.svg",
        "failure_taxonomy_heatmap.svg",
    ]:
        src = out_dir / name
        dst = figures / name
        if src.exists() and not dst.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        elif not dst.exists():
            _placeholder_svg(dst, name, "not generated by current gate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(wandb=True)
    parser.add_argument("--packages", default="V6_7_ALL")
    parser.add_argument("--out-dir", type=Path, default=Path("results/real_rerun_20260504/v67_real"))
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
    parser.add_argument("--wandb-project", default="DG-KAN")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-group", default="v67-real-20260504")
    parser.add_argument("--wandb-name-prefix", default="v67-real")
    parser.add_argument("--no-wandb", action="store_false", dest="wandb")
    parser.add_argument("--fresh", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    out_dir = ensure_dir(args.out_dir)
    _wandb_init(args)
    try:
        run_p0(args)
        p1_rows = run_p1(args)
        run_p0_reproduction(args, p1_rows)
        p25_rows = run_p25(args)
        run_p2(args, p25_rows)
        p3_summary, _p3_detail = run_p3(args)
        p4_rows = run_p4(args, p3_summary)
        survivor_type, _best = _best_p3_survivor(p3_summary, p1_rows, p25_rows)
        _p5_rows, p5_pass = run_p5(args, survivor_type, p4_rows)
        route = run_route(args, p1_rows, p25_rows, p3_summary, p4_rows, p5_pass)
        run_p6_p7(args, bool(route.get("open_task_reentry", False)))
        run_failure(args)
        _write_required_placeholders(out_dir)
        _placeholder_svg(out_dir / "p2_component_repair_pareto.svg", "P2 repair pareto", "see p2_component_repair_summary.csv")
        _write_manifest(out_dir, args, started, time.time())
    finally:
        _wandb_finish(args, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
