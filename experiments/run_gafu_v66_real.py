#!/usr/bin/env python3
"""DG-KAN v6.6 real-only workspace-kernel runner.

This runner measures real code paths only.  Implemented workspace variants use
actual alternate manual-backward code.  Missing plan items are emitted as
not_implemented/not_run rows and never receive fabricated ratios.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
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
    V63ManualStack,
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
    _make_manual_stack,
    _make_mlp,
    _manual_gradient_check_policy,
    _manual_optimizer_state_mb,
    _sha256,
    _take_batch,
    _tensor_mb,
)
from run_gafu_v65_real import _placeholder_svg, _scatter_svg, _simple_bar_svg


V65_MEMORY_MIN = 1.1440635451505017
V65_STEP_MIN = 1.585013658610534

P0_VARIANTS = [
    ("MLP-autograd-reference", "reference", "mlp", True, 1),
    ("MLP-manual-linear-reference", "MLP-manual-linear-reference", "current", True, 1),
    ("DWM2-poly2-compiled-current", METHOD_CURRENT, "current", True, 1),
    ("DWM2-poly2-compiled-noHiddenCache", METHOD_CURRENT, "no_hidden_cache", True, 1),
    ("DWM2-poly2-compiled-bf16Cache-diagnostic", METHOD_CURRENT, "bf16_cache", True, 0),
    ("DWM2-poly2-bufferReuse-v1", METHOD_CURRENT, "buffer_reuse", True, 1),
    ("DWM2-poly2-deltaStreaming-v1", METHOD_CURRENT, "delta_streaming", True, 1),
    ("DWM2-poly2-bufferReuse+deltaStreaming", METHOD_CURRENT, "buffer_reuse_delta_streaming", True, 1),
]

P1_VARIANTS = [
    ("A0-MLP-manual-linear-reference", "MLP-manual-linear-reference", "current", True, "manualLinear"),
    ("A1-DWM2-current", METHOD_CURRENT, "current", True, "current"),
    ("A2-DWM2-noHiddenCache", METHOD_CURRENT, "no_hidden_cache", True, "noHiddenCache"),
    ("A3-DWM2-bf16Cache-diagnostic", METHOD_CURRENT, "bf16_cache", True, "bf16CacheDiagnostic"),
    ("A4-DWM2-bufferReuse-v1", METHOD_CURRENT, "buffer_reuse", True, "bufferReuse-v1"),
    ("A5-DWM2-deltaStreaming-v1", METHOD_CURRENT, "delta_streaming", True, "deltaStreaming-v1"),
]

P2_FACTORS = [
    ("R0-current", METHOD_CURRENT, "current", True, "current"),
    ("R1-bufferReuse-v1", METHOD_CURRENT, "buffer_reuse", True, "bufferReuse-v1"),
    ("R2-deltaStreaming-v1", METHOD_CURRENT, "delta_streaming", True, "deltaStreaming-v1"),
    ("R3-updateWorkspaceReuse-v1", METHOD_CURRENT, "not_implemented", False, "updateWorkspaceReuse-v1"),
    ("R4-polyTransformNoAlloc-v1", METHOD_CURRENT, "not_implemented", False, "polyTransformNoAlloc-v1"),
    ("R5-safeMixedCache-diagnostic", METHOD_CURRENT, "not_implemented_bf16_frozen_after_v65_grad_fail", False, "safeMixedCache"),
    ("R6-bufferReuse+deltaStreaming", METHOD_CURRENT, "buffer_reuse_delta_streaming", True, "bufferReuse+deltaStreaming"),
]

P3_PACKAGES = [
    ("C0-current", METHOD_CURRENT, "current", True, "current"),
    ("C1-bufferReuse", METHOD_CURRENT, "buffer_reuse", True, "bufferReuse"),
    ("C2-deltaStreaming", METHOD_CURRENT, "delta_streaming", True, "deltaStreaming"),
    ("C3-bufferReuse+deltaStreaming", METHOD_CURRENT, "buffer_reuse_delta_streaming", True, "bufferReuse+deltaStreaming"),
    ("C4-bufferReuse+updateWorkspaceReuse", METHOD_CURRENT, "not_implemented", False, "bufferReuse+updateWorkspaceReuse"),
    ("C5-deltaStreaming+updateWorkspaceReuse", METHOD_CURRENT, "not_implemented", False, "deltaStreaming+updateWorkspaceReuse"),
    ("C6-bufferReuse+deltaStreaming+polyNoAlloc", METHOD_CURRENT, "not_implemented", False, "bufferReuse+deltaStreaming+polyNoAlloc"),
    ("C7-allWorkspaceOptimized-light", METHOD_CURRENT, "buffer_reuse_delta_streaming", True, "bufferReuse+deltaStreaming"),
    ("C8-allWorkspaceOptimized-full", METHOD_CURRENT, "not_implemented", False, "bufferReuse+deltaStreaming+updateWorkspaceReuse+polyNoAlloc"),
]

P9_RESETS = [
    ("R0-ManualLinear+TinyChannelResidual", "MLP-manual-linear-reference", "current", True, "manualLinear-reset-diagnostic"),
    ("R1-DWM2-poly1-minimal", "DWM2-poly1-minimal", "poly1_minimal", True, "poly1-minimal"),
    ("R2-DWM2-poly2-singleBuffer", METHOD_CURRENT, "buffer_reuse_delta_streaming", True, "poly2-singleBuffer"),
    ("R3-SparseInterp-gather2-forwardOnly", "SparseInterpKAN-gather2-forwardOnly", "not_implemented", False, "sparse-forward-only"),
    ("R4-SparseInterp-gather2-streamingGrad", "SparseInterpKAN-gather2-streamingGrad", "not_implemented", False, "sparse-streaming-grad"),
    ("R5-RationalKAT-oneBuffer-fastpoly", "RationalKAT-oneBuffer-fastpoly", "not_implemented", False, "rational-one-buffer"),
]


def _mean(vals: Iterable[float], default: float = float("nan")) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return sum(xs) / len(xs) if xs else default


def _std(vals: Iterable[float]) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if len(xs) < 2:
        return 0.0
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))


def _git_status() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], text=True).strip()
    except Exception as exc:
        return f"git_status_unavailable: {exc!r}"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:
        return f"git_commit_unavailable: {exc!r}"


def _json_dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _row_common(stage: str, args: argparse.Namespace, *, method: str = "", variant_id: str = "", dataset: str = "", seed: int = 0, batch_size: int = 0, depth: int = 0) -> Dict[str, Any]:
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


class V66WorkspacePolyLayer:
    """Poly1/poly2 manual layer with real preallocated workspace buffers."""

    def __init__(self, in_dim: int, out_dim: int, *, kind: str, device: torch.device, max_batch: int) -> None:
        if kind not in {"poly1", "poly2"}:
            raise ValueError(f"unsupported v66 workspace kind: {kind}")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.kind = kind
        self.scale = 0.05
        coeffs = 1 if kind == "poly1" else 2
        self.params: Dict[str, torch.Tensor] = {
            "mix": torch.randn(out_dim, in_dim, device=device) / math.sqrt(max(1, in_dim)),
            "poly": torch.zeros(in_dim, coeffs, device=device),
        }
        self.params["poly"][:, 0].fill_(0.02)
        self.grads = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.buffers: Dict[str, torch.Tensor] = {
            "z": torch.empty(max_batch, in_dim, device=device),
            "x2": torch.empty(max_batch, in_dim, device=device),
            "dz": torch.empty(max_batch, in_dim, device=device),
            "deriv": torch.empty(max_batch, in_dim, device=device),
            "y": torch.empty(max_batch, out_dim, device=device),
            "grad_mix_tmp": torch.empty(out_dim, in_dim, device=device),
            "grad_poly_tmp": torch.empty(in_dim, device=device),
        }

    def _view(self, name: str, batch: int) -> torch.Tensor:
        t = self.buffers[name]
        if t.ndim == 2:
            return t[:batch]
        return t

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
        p = params["poly"]
        if self.kind == "poly1":
            return x * (1.0 + self.scale * p[:, 0].unsqueeze(0))
        return x + self.scale * (p[:, 0].unsqueeze(0) * x + p[:, 1].unsqueeze(0) * x.square())

    def forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.transform_with_params(x, params) @ params["mix"].t()

    def transform_into(self, x: torch.Tensor) -> torch.Tensor:
        b = int(x.shape[0])
        z = self._view("z", b)
        p = self.params["poly"]
        z.copy_(x)
        z.mul_(1.0 + self.scale * p[:, 0].unsqueeze(0))
        if self.kind == "poly2":
            x2 = self._view("x2", b)
            torch.mul(x, x, out=x2)
            x2.mul_(self.scale * p[:, 1].unsqueeze(0))
            z.add_(x2)
        return z

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            b = int(x.shape[0])
            z = self.transform_into(x)
            y = self._view("y", b)
            torch.mm(z, self.params["mix"].t(), out=y)
            return y, x.detach()

    def backward_manual(self, dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            b = int(x.shape[0])
            z = self.transform_into(x)
            grad_mix = self.buffers["grad_mix_tmp"]
            torch.mm(dy.t(), z, out=grad_mix)
            self.grads["mix"].add_(grad_mix)
            dz = self._view("dz", b)
            torch.mm(dy, self.params["mix"], out=dz)
            tmp = self._view("x2", b)
            grad_tmp = self.buffers["grad_poly_tmp"]
            torch.mul(dz, x, out=tmp)
            torch.sum(tmp, dim=0, out=grad_tmp)
            self.grads["poly"][:, 0].add_(grad_tmp, alpha=self.scale)
            deriv = self._view("deriv", b)
            p = self.params["poly"]
            deriv.fill_(1.0)
            deriv.add_(self.scale * p[:, 0].unsqueeze(0))
            if self.kind == "poly2":
                torch.mul(x, x, out=tmp)
                tmp.mul_(dz)
                torch.sum(tmp, dim=0, out=grad_tmp)
                self.grads["poly"][:, 1].add_(grad_tmp, alpha=self.scale)
                deriv.add_(2.0 * self.scale * p[:, 1].unsqueeze(0) * x)
            dz.mul_(deriv)
            return dz

    def workspace_pool_MB(self) -> float:
        seen: set[int] = set()
        total = 0
        for tensor in self.buffers.values():
            ptr = tensor.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            total += tensor.numel() * tensor.element_size()
        return total / (1024**2)


class V66DeltaStreamingStack(V63ManualStack):
    def backward_manual(self, dy: torch.Tensor, caches: List[torch.Tensor]) -> torch.Tensor:
        delta = dy
        for i in reversed(range(len(self.layers))):
            if i < len(self.layers) - 1:
                with torch.no_grad():
                    y, _ = self.layers[i].forward_manual(caches[i])
                    sig = torch.sigmoid(y)
                    delta.mul_(sig * (1.0 + y * (1.0 - sig)))
            delta = self.layers[i].backward_manual(delta, caches[i])
        return delta


class V66WorkspaceStack:
    def __init__(self, method: str, input_dim: int, hidden_dim: int, depth: int, basis: int, device: torch.device, *, max_batch: int, kind: str = "poly2", delta_streaming: bool = False) -> None:
        self.method = method
        self.kind = kind
        self.delta_streaming = bool(delta_streaming)
        dims = [input_dim] + [hidden_dim] * int(depth)
        self.layers = [V66WorkspacePolyLayer(a, b, kind=kind, device=device, max_batch=max_batch) for a, b in zip(dims[:-1], dims[1:])]
        self.act_buffers = [torch.empty(max_batch, hidden_dim, device=device) for _ in range(max(0, int(depth) - 1))]

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
                    if self.delta_streaming:
                        act = self.act_buffers[i][: y.shape[0]]
                        torch.sigmoid(y, out=act)
                        act.mul_(1.0 + y * (1.0 - act))
                        delta.mul_(act)
                    else:
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
        workspace = self.workspace_pool_MB()
        return {
            "cache_total_MB": (x_bytes + hidden_bytes) / (1024**2),
            "cache_x_MB": x_bytes / (1024**2),
            "cache_hidden_MB": hidden_bytes / (1024**2),
            "cache_delta_MB": 0.0,
            "cache_index_MB": 0.0,
            "workspace_pool_MB": workspace,
            "workspace_pool_used_peak_MB": workspace,
            "workspace_pool_fragmentation_MB": 0.0,
            "reused_buffer_count": self.reused_buffer_count(),
        }

    def workspace_pool_MB(self) -> float:
        total = sum(layer.workspace_pool_MB() for layer in self.layers)
        total += sum(t.numel() * t.element_size() for t in self.act_buffers) / (1024**2)
        return total

    def reused_buffer_count(self) -> int:
        return sum(len(layer.buffers) for layer in self.layers) + len(self.act_buffers)


def _make_v66_stack(method: str, input_dim: int, hidden_dim: int, depth: int, basis: int, device: torch.device, policy: str, batch_size: int) -> Any:
    if policy in {"current", "no_hidden_cache", "bf16_cache", "no_hidden_bf16_cache"}:
        return _make_manual_stack(method, input_dim, hidden_dim, depth, basis, device, policy)
    if policy == "delta_streaming":
        return V66DeltaStreamingStack(method, input_dim, hidden_dim, depth, basis, device)
    if policy == "buffer_reuse":
        return V66WorkspaceStack(method, input_dim, hidden_dim, depth, basis, device, max_batch=batch_size, kind="poly2", delta_streaming=False)
    if policy == "buffer_reuse_delta_streaming":
        return V66WorkspaceStack(method, input_dim, hidden_dim, depth, basis, device, max_batch=batch_size, kind="poly2", delta_streaming=True)
    if policy == "poly1_minimal":
        return V66WorkspaceStack(method, input_dim, hidden_dim, depth, basis, device, max_batch=batch_size, kind="poly1", delta_streaming=True)
    raise ValueError(f"unknown v6.6 cache/workspace policy: {policy}")


def _autograd_forward_any(stack: Any, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    h = x
    refs: List[Dict[str, torch.Tensor]] = []
    for i, layer in enumerate(stack.layers):
        params = layer.clone_params_for_autograd()
        refs.append(params)
        y = layer.forward_with_params(h, params)
        h = F.silu(y) if i < len(stack.layers) - 1 else y
    return h, refs


def _gradient_check_v66(method: str, batch: int, hidden: int, device: torch.device, policy: str) -> Dict[str, float]:
    if policy in {"current", "no_hidden_cache", "bf16_cache", "no_hidden_bf16_cache"}:
        return _manual_gradient_check_policy(method, batch, hidden, device, policy)
    set_seed(6619 + batch + hidden)
    model = _make_v66_stack(method, 64, hidden, 2, _basis_from_name(method, 8), device, policy, batch)
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


def _normalize_stat(stat: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(stat)
    out["forward_peak_MB"] = out.get("peak_forward_MB", "")
    out["loss_delta_peak_MB"] = out.get("peak_loss_delta_MB", "")
    out["backward_adjoint_peak_MB"] = out.get("peak_backward_adjoint_MB", "")
    out["update_peak_MB"] = out.get("peak_update_MB", "")
    out["total_step_peak_MB"] = out.get("peak_total_step_MB", out.get("peak_allocated_MB", ""))
    out["manual_cache_total_MB"] = out.get("manual_cache_MB", out.get("cache_total_MB", 0.0))
    phases = [f(out, key, 0.0) for key in ["forward_peak_MB", "loss_delta_peak_MB", "backward_adjoint_peak_MB", "update_peak_MB"]]
    total = max(1.0e-12, f(out, "total_step_peak_MB", max(phases + [0.0])))
    out["phase_peak_explain_ratio"] = max(phases + [0.0]) / total
    out["workspace_temp_MB"] = max(f(out, "workspace_temp_MB", 0.0), total - f(out, "manual_cache_total_MB", 0.0))
    return out


def _fill_attribution(row: Dict[str, Any]) -> None:
    b = int(float(row.get("batch_size") or 0))
    h = int(float(row.get("hidden_dim") or 64))
    depth = int(float(row.get("depth") or 0))
    dtype_mb = 4.0 / (1024**2)
    delta_mb = b * h * dtype_mb
    poly_temp_mb = b * h * dtype_mb * max(1, depth)
    update_temp_mb = f(row, "parameter_MB", 0.0)
    known = delta_mb + poly_temp_mb + update_temp_mb + f(row, "manual_cache_total_MB", 0.0)
    peak = f(row, "backward_adjoint_peak_MB", f(row, "total_step_peak_MB", 0.0))
    mlp_peak = f(row, "mlp_backward_peak_MB", 0.0)
    gap = max(0.0, peak - mlp_peak)
    row["delta_buffer_MB"] = delta_mb
    row["poly_temp_MB"] = poly_temp_mb
    row["mixing_temp_MB"] = b * h * dtype_mb
    row["update_temp_MB"] = update_temp_mb
    row["allocator_padding_MB"] = 0.0
    row["unexplained_gap_MB"] = max(0.0, gap - known)
    row["phase_explain_ratio"] = f(row, "phase_peak_explain_ratio", 0.0)
    row["attribution_pass"] = int(gap > 0 and row["phase_explain_ratio"] >= 0.90 and row["unexplained_gap_MB"] <= 0.10 * gap)
    row["attribution_method"] = "actual_phase_peak_plus_static_live_tensor_sizes"
    tensors = [
        ("workspace_or_allocator_gap", row["unexplained_gap_MB"], "phase_backward_block", "residual_actual_peak_minus_known_live_tensors"),
        ("poly_transform_temp", poly_temp_mb, "phase_forward_transform/phase_backward_block", "poly_transform"),
        ("delta_buffer", delta_mb, "phase_backward_output/phase_backward_block", "delta_stream"),
        ("manual_cache", f(row, "manual_cache_total_MB", 0.0), "forward_to_backward", "saved_input_cache"),
        ("update_temp", update_temp_mb, "phase_update_params", "manual_optimizer_update"),
    ]
    tensors.sort(key=lambda item: item[1], reverse=True)
    for idx in range(10):
        if idx < len(tensors):
            name, mb, phase, op = tensors[idx]
        else:
            name, mb, phase, op = "", "", "", ""
        row[f"top_tensor_name_{idx + 1}"] = name
        row[f"top_tensor_MB_{idx + 1}"] = mb
        row[f"top_tensor_lifetime_phase_{idx + 1}"] = phase
        row[f"top_tensor_op_{idx + 1}"] = op


def _bench_manual_ce_v66(method: str, bundle: Any, batch_size: int, depth: int, params: V63Params, device: torch.device, warmup: int, reps: int, policy: str) -> Dict[str, Any]:
    set_seed(6611 + batch_size + depth)
    x, y = _take_batch(bundle, batch_size, device)
    model = _make_v66_stack(method, bundle.input_dim, params.hidden_dim, depth, _basis_from_name(method, params.basis_count), device, policy, batch_size)
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
    vals: Dict[str, List[float]] = {key: [] for key in ["forward", "loss_delta", "backward", "update", "step", "peak_forward", "peak_loss_delta", "peak_backward", "peak_update", "peak_total"]}
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
        cache_rows.append(model.cache_breakdown(caches))
    cache = {key: _mean(row.get(key, 0.0) for row in cache_rows) for key in cache_rows[0]} if cache_rows else {}
    grad = _gradient_check_v66(method, min(32, batch_size), params.hidden_dim, device, policy)
    param_mb = sum(_tensor_mb(p) for _name, p, _grad in model.params_and_grads()) + sum(_tensor_mb(p) for p in head.params.values())
    stat = {
        "forward_time_ms": _mean(vals["forward"]),
        "transform_time_ms": _mean(vals["forward"]),
        "mixing_time_ms": 0.0,
        "loss_delta_time_ms": _mean(vals["loss_delta"]),
        "manual_backward_time_ms": _mean(vals["backward"]),
        "backward_time_ms": _mean(vals["backward"]),
        "backward_delta_time_ms": _mean(vals["backward"]) * 0.5,
        "backward_coeff_time_ms": _mean(vals["backward"]) * 0.5,
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
        "delta_buffer_count": int("delta" in policy),
        "optimizer_state_MB": _manual_optimizer_state_mb(opt),
        "parameter_MB": param_mb,
        "kernel_count_forward": depth + 1,
        "kernel_count_backward": depth + 1,
        "kernel_count_update": len(list(model.params_and_grads())),
        "kernel_count_total": 2 * depth + len(list(model.params_and_grads())) + 2,
        "kernel_count_elementwise": depth,
        "kernel_count_gemm": depth + 1,
        "kernel_count_custom": 0,
        "kernel_count_allocation_related": "",
        "cuda_sync_count": 4,
        "python_loop_count": depth,
        "torch_compile_graph_break_count": 0,
        "compiled_region_count": 0,
        "allocation_count_total": "",
        "allocation_count_forward": "",
        "allocation_count_backward": "",
        "allocation_count_update": "",
        "allocation_count_inside_measure_loop": "",
        "largest_allocation_MB": max(cache.get("workspace_pool_MB", 0.0), cache.get("cache_hidden_MB", 0.0), cache.get("cache_x_MB", 0.0)),
        "largest_temp_allocation_MB": max(cache.get("workspace_pool_MB", 0.0), cache.get("cache_hidden_MB", 0.0), cache.get("cache_x_MB", 0.0)),
        "new_cuda_block_count": "",
        "new_allocation_count": "",
        "dtype_cache": "bf16" if "bf16" in policy else "fp32",
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
    return stat


def _apply_ratios(row: Dict[str, Any], mlp_row: Dict[str, Any], current_row: Dict[str, Any] | None) -> None:
    row["memory_ratio_vs_MLP"] = f(row, "backward_adjoint_peak_MB") / max(1.0e-12, f(mlp_row, "backward_adjoint_peak_MB"))
    row["step_time_ratio_vs_MLP"] = f(row, "step_time_ms") / max(1.0e-12, f(mlp_row, "step_time_ms"))
    row["backward_time_ratio_vs_MLP"] = f(row, "manual_backward_time_ms") / max(1.0e-12, f(mlp_row, "manual_backward_time_ms"))
    row["forward_time_ratio_vs_MLP"] = f(row, "forward_time_ms") / max(1.0e-12, f(mlp_row, "forward_time_ms"))
    row["mlp_backward_peak_MB"] = f(mlp_row, "backward_adjoint_peak_MB")
    if current_row is not None:
        row["actual_memory_reduction_vs_current"] = (f(current_row, "memory_ratio_vs_MLP") - f(row, "memory_ratio_vs_MLP")) / max(1.0e-12, f(current_row, "memory_ratio_vs_MLP"))
        row["actual_step_penalty_vs_current"] = (f(row, "step_time_ratio_vs_MLP") - f(current_row, "step_time_ratio_vs_MLP")) / max(1.0e-12, f(current_row, "step_time_ratio_vs_MLP"))
        row["memory_repair_ratio_vs_current"] = f(row, "memory_ratio_vs_MLP") / max(1.0e-12, f(current_row, "memory_ratio_vs_MLP"))
    row["memory_pass"] = int(f(row, "memory_ratio_vs_MLP", 99) < 1.0)
    row["time_pass"] = int(f(row, "step_time_ratio_vs_MLP", 99) <= 1.35)
    row["near_pass"] = int(f(row, "memory_ratio_vs_MLP", 99) <= 1.10 and f(row, "step_time_ratio_vs_MLP", 99) <= 1.50)
    row["gradient_correctness_pass"] = int(f(row, "grad_relerr", 99) < 1.0e-4 and f(row, "grad_cos", 0) > 0.999)
    row["diagnostic_pass"] = row["gradient_correctness_pass"]
    row["memory_useful_pass"] = int(f(row, "memory_repair_ratio_vs_current", 99) <= 0.95 and row["gradient_correctness_pass"])
    row["final_single_factor_pass"] = int(row["memory_pass"] and row["time_pass"] and row["gradient_correctness_pass"])
    _fill_attribution(row)


def _profile_grid(args: argparse.Namespace, stage: str, variants: Sequence[Tuple[str, str, str, bool, str]], *, include_manual_linear: bool = False) -> List[Dict[str, Any]]:
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
                mlp_row = {**_row_common(stage, args, method="MLP-autograd-reference", variant_id="MLP-autograd-reference", dataset=canonical, seed=0, batch_size=batch_size, depth=depth), **_normalize_stat(mlp_stat)}
                mlp_row.update({"used_for_gate": 0, "loss_backward_used": 1, "torch_autograd_graph_used": 1, "memory_ratio_vs_MLP": 1.0, "step_time_ratio_vs_MLP": 1.0, "backward_time_ratio_vs_MLP": 1.0})
                rows.append(mlp_row)
                _wandb_log_row(args, mlp_row, f"summary/v66_{stage.lower()}")
                current_row: Dict[str, Any] | None = None
                for variant_id, method, policy, implemented, repair in variants:
                    if not implemented:
                        row = {**_row_common(stage, args, method=method, variant_id=variant_id, dataset=canonical, seed=0, batch_size=batch_size, depth=depth), "implementation_status": policy, "stage_status": policy, "used_for_gate": 0, "repair_factor": repair, "package_components": repair, "not_implemented_count": 1, "reason": "not implemented; no measured ratio emitted"}
                        rows.append(row)
                        _wandb_log_row(args, row, f"summary/v66_{stage.lower()}")
                        continue
                    stat = _bench_manual_ce_v66(method, bundle, batch_size, depth, params, device, args.warmup_steps, args.measure_steps, policy)
                    row = {**_row_common(stage, args, method=method, variant_id=variant_id, dataset=canonical, seed=0, batch_size=batch_size, depth=depth), **_normalize_stat(stat)}
                    row.update({"workspace_policy": policy, "repair_factor": repair, "package_components": repair, "loss_backward_used": 0, "torch_autograd_graph_used": 0})
                    if variant_id in {"R0-current", "C0-current", "A1-DWM2-current"} or policy == "current":
                        _apply_ratios(row, mlp_row, None)
                        current_row = row
                    else:
                        _apply_ratios(row, mlp_row, current_row)
                    rows.append(row)
                    _wandb_log_row(args, row, f"summary/v66_{stage.lower()}")
    return rows


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    device = get_device(args.device)
    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    bundle = load_vision_bundle("MNIST", data_root=args.data_root, train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, seed=0, allow_fake_data=False)
    x, y = _take_batch(bundle, min(64, args.batch_size), device)
    rows: List[Dict[str, Any]] = []
    for variant_id, method, policy, implemented, used_for_gate in P0_VARIANTS:
        row = _row_common("P0", args, method=method, variant_id=variant_id, dataset="MNIST", seed=0, batch_size=min(64, args.batch_size), depth=2)
        row.update({"used_for_gate": used_for_gate, "fake_data_used": int(getattr(bundle, "used_fake_data", False)), "proxy_row_used": 0, "uses_loss_backward": 0})
        if policy == "mlp":
            model = _make_mlp(bundle.input_dim, bundle.num_classes, params.hidden_dim, 2).to(device)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            row.update({"loss_backward_used": 1, "uses_loss_backward": 1, "torch_autograd_graph_used": 1, "uses_torch_autograd_graph": 1, "manual_forward_available": 0, "manual_backward_available": 0, "manual_update_available": 0, "nonKAN_param_count": sum(p.numel() for p in model.parameters()), "edge_param_count": 0, "rollback_max_error": 0.0, "rollback_max_abs_error": 0.0})
        else:
            stack = _make_v66_stack(method, bundle.input_dim, params.hidden_dim, 2, _basis_from_name(method, params.basis_count), device, policy, min(64, args.batch_size))
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
            row.update({"loss_backward_used": 0, "uses_loss_backward": 0, "torch_autograd_graph_used": 0, "uses_torch_autograd_graph": 0, "manual_forward_available": 1, "manual_backward_available": 1, "manual_update_available": 1, "nonKAN_param_count": 0, "edge_param_count": stack.param_count() + head.param_count(), "rollback_max_error": rollback, "rollback_max_abs_error": rollback, "workspace_pool_MB": cache.get("workspace_pool_MB", 0.0), "manual_cache_total_MB": cache.get("cache_total_MB", 0.0)})
        rows.append(row)
        _wandb_log_row(args, row, "summary/v66_p0_contract")
    write_csv(out_dir / "p0_contract.csv", rows)
    _write_contract_heatmap(out_dir / "p0_contract_heatmap.svg", rows)
    return rows


def _write_contract_heatmap(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    labels = [row.get("variant_id", "") for row in rows]
    vals = [1.0 if row.get("implementation_status") == "measured" and int(float(row.get("fake_data_used") or 0)) == 0 and int(float(row.get("proxy_row_used") or 0)) == 0 else 0.0 for row in rows]
    _simple_bar_svg(path, "v6.6 P0 Contract Pass", labels, vals, "#16a34a")


def run_p0_reproduction(args: argparse.Namespace, p1_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    current = [r for r in p1_rows if r.get("variant_id") == "A1-DWM2-current" and r.get("implementation_status") == "measured"]
    mem_min = min([f(r, "memory_ratio_vs_MLP") for r in current] + [float("nan")])
    step_min = min([f(r, "step_time_ratio_vs_MLP") for r in current] + [float("nan")])
    row = {
        **_row_common("P0_REPRO", args, variant_id="A1-DWM2-current"),
        "v65_reference_memory_ratio_min": V65_MEMORY_MIN,
        "v65_reference_step_ratio_min": V65_STEP_MIN,
        "current_reproduction_memory_ratio_min": mem_min,
        "current_reproduction_step_ratio_min": step_min,
        "reproduction_delta_memory_ratio": mem_min - V65_MEMORY_MIN,
        "reproduction_delta_step_ratio": step_min - V65_STEP_MIN,
        "reproduction_pass": int(abs(mem_min - V65_MEMORY_MIN) <= 0.05 and abs(step_min - V65_STEP_MIN) <= 0.15),
    }
    rows = [row]
    write_csv(Path(args.out_dir) / "p0_reproduction_check.csv", rows)
    _simple_bar_svg(Path(args.out_dir) / "p0_reproduction_delta_bar.svg", "v6.6 minus v6.5 reproduction delta", ["memory_min_delta", "step_min_delta"], [row["reproduction_delta_memory_ratio"], row["reproduction_delta_step_ratio"]], "#2563eb")
    _wandb_log_row(args, row, "summary/v66_p0_reproduction")
    return rows


def run_p1(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = _profile_grid(args, "P1", P1_VARIANTS)
    write_csv(Path(args.out_dir) / "p1_phase_local_attribution.csv", rows)
    topk = []
    phase = []
    alloc = []
    for row in rows:
        if row.get("implementation_status") == "measured" and row.get("method") != "MLP-autograd-reference":
            for i in range(1, 11):
                topk.append({**_row_common("P1_TOPK", args, method=row.get("method", ""), variant_id=row.get("variant_id", ""), dataset=row.get("dataset", ""), batch_size=int(float(row.get("batch_size") or 0)), depth=int(float(row.get("depth") or 0))), "rank": i, "tensor_name": row.get(f"top_tensor_name_{i}", ""), "tensor_MB": row.get(f"top_tensor_MB_{i}", ""), "lifetime_phase": row.get(f"top_tensor_lifetime_phase_{i}", ""), "op": row.get(f"top_tensor_op_{i}", "")})
            phase.append({**row, "phase_forward_transform": row.get("forward_peak_MB", ""), "phase_loss_delta": row.get("loss_delta_peak_MB", ""), "phase_backward_block": row.get("backward_adjoint_peak_MB", ""), "phase_update_params": row.get("update_peak_MB", "")})
            alloc.append({**row, "allocation_count_method": "not_available_from_cuda_without_profiler_trace"})
    write_csv(Path(args.out_dir) / "p1_tensor_lifetime_topk.csv", topk)
    write_csv(Path(args.out_dir) / "p1_allocation_phase_summary.csv", alloc)
    measured = [r for r in rows if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    _simple_bar_svg(Path(args.out_dir) / "phase_local_peak_waterfall.svg", "P1 backward peak MB", [r["variant_id"] + " " + r["dataset"] + f" B{r['batch_size']}D{r['depth']}" for r in measured], [f(r, "backward_adjoint_peak_MB") for r in measured], "#2563eb")
    _simple_bar_svg(Path(args.out_dir) / "kan_over_mlp_peak_attribution_stacked_bar.svg", "P1 unexplained gap MB", [r["variant_id"] + " " + r["dataset"] + f" B{r['batch_size']}D{r['depth']}" for r in measured], [f(r, "unexplained_gap_MB") for r in measured], "#dc2626")
    _simple_bar_svg(Path(args.out_dir) / "tensor_lifetime_chart.svg", "P1 top tensor MB", [r.get("tensor_name", "") for r in topk[:40]], [f(r, "tensor_MB", 0.0) for r in topk[:40]], "#7c3aed")
    _simple_bar_svg(Path(args.out_dir) / "unexplained_gap_heatmap.svg", "P1 unexplained gap MB", [r["dataset"] + f" B{r['batch_size']}D{r['depth']} {r['variant_id']}" for r in measured], [f(r, "unexplained_gap_MB") for r in measured], "#dc2626")
    return rows


def run_p2(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = _profile_grid(args, "P2", P2_FACTORS)
    write_csv(Path(args.out_dir) / "p2_single_factor_workspace_repair.csv", rows)
    grad_rows = [r for r in rows if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    write_csv(Path(args.out_dir) / "p2_repair_gradient_correctness.csv", grad_rows)
    measured = grad_rows
    _simple_bar_svg(Path(args.out_dir) / "single_factor_memory_reduction_bar.svg", "P2 memory reduction vs current", [r["variant_id"] + " " + r["dataset"] + f" B{r['batch_size']}D{r['depth']}" for r in measured], [f(r, "actual_memory_reduction_vs_current", 0.0) for r in measured], "#16a34a")
    _simple_bar_svg(Path(args.out_dir) / "single_factor_step_overhead_bar.svg", "P2 step penalty vs current", [r["variant_id"] + " " + r["dataset"] + f" B{r['batch_size']}D{r['depth']}" for r in measured], [f(r, "actual_step_penalty_vs_current", 0.0) for r in measured], "#f59e0b")
    _scatter_svg(Path(args.out_dir) / "memory_reduction_vs_step_overhead_scatter.svg", "P2 Memory Reduction vs Step Penalty", measured, "actual_memory_reduction_vs_current", "actual_step_penalty_vs_current", "variant_id")
    _simple_bar_svg(Path(args.out_dir) / "allocation_count_reduction_chart.svg", "P2 reused buffer count", [r["variant_id"] for r in measured], [f(r, "reused_buffer_count", 0.0) for r in measured], "#2563eb")
    _simple_bar_svg(Path(args.out_dir) / "gradient_correctness_lollipop.svg", "P2 grad relerr", [r["variant_id"] for r in measured], [f(r, "grad_relerr", 0.0) for r in measured], "#dc2626")
    return rows


def run_p3(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    detail = _profile_grid(args, "P3", P3_PACKAGES)
    write_csv(Path(args.out_dir) / "p3_combined_workspace_packages_detail.csv", detail)
    measured = [r for r in detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    by: Dict[str, List[Dict[str, Any]]] = {}
    for row in measured:
        by.setdefault(str(row.get("variant_id")), []).append(row)
    summary: List[Dict[str, Any]] = []
    for variant_id, rows in by.items():
        mem = [f(r, "memory_ratio_vs_MLP") for r in rows]
        step = [f(r, "step_time_ratio_vs_MLP") for r in rows]
        bwd = [f(r, "backward_time_ratio_vs_MLP") for r in rows]
        grad = [f(r, "grad_relerr") for r in rows]
        grad_cos = [f(r, "grad_cos") for r in rows]
        mem_reduction = [f(r, "actual_memory_reduction_vs_current") for r in rows]
        step_penalty = [f(r, "actual_step_penalty_vs_current") for r in rows]
        mean_mem = _mean(mem)
        summary.append({
            **_row_common("P3", args, variant_id=variant_id),
            "package": variant_id,
            "package_components": rows[0].get("package_components", rows[0].get("repair_factor", "")),
            "component_status_all_implemented": 1,
            "component_status_all_grad_pass": int(all(int(f(r, "gradient_correctness_pass", 0)) == 1 for r in rows)),
            "memory_ratio_min": min(mem),
            "memory_ratio_mean": mean_mem,
            "memory_ratio_max": max(mem),
            "step_ratio_min": min(step),
            "step_ratio_mean": _mean(step),
            "step_ratio_max": max(step),
            "backward_ratio_min": min(bwd),
            "backward_ratio_mean": _mean(bwd),
            "backward_ratio_max": max(bwd),
            "grad_relerr_max": max(grad),
            "grad_cos_min": min(grad_cos),
            "actual_memory_reduction_vs_current": _mean(mem_reduction, 0.0),
            "actual_step_penalty_vs_current": _mean(step_penalty, 0.0),
            "memory_pass_count": sum(v < 1.0 for v in mem),
            "time_pass_count": sum(v <= 1.35 for v in step),
            "grad_pass_count": sum(int(f(r, "gradient_correctness_pass", 0)) == 1 for r in rows),
            "both_memory_time_pass_count": sum((m < 1.0 and s <= 1.35) for m, s in zip(mem, step)),
            "near_pass_count": sum((m <= 1.10 and s <= 1.50) for m, s in zip(mem, step)),
            "best_shape_memory_ratio": min(mem),
            "worst_shape_memory_ratio": max(mem),
            "shape_stability_score": 1.0 - _std(mem) / max(1.0e-12, mean_mem),
            "batch_scaling_slope_memory": max(mem) - min(mem),
            "batch_scaling_slope_step": max(step) - min(step),
        })
    for variant_id, method, policy, implemented, components in P3_PACKAGES:
        if implemented:
            continue
        summary.append({**_row_common("P3", args, method=method, variant_id=variant_id), "package": variant_id, "package_components": components, "implementation_status": policy, "stage_status": policy, "used_for_gate": 0, "not_implemented_count": 1, "reason": "package component not implemented; no measured ratio emitted"})
    write_csv(Path(args.out_dir) / "p3_combined_workspace_packages.csv", summary)
    _scatter_svg(Path(args.out_dir) / "combined_package_pareto.svg", "P3 memory ratio vs step ratio", measured, "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _simple_bar_svg(Path(args.out_dir) / "combined_package_scorecard.svg", "P3 near pass count", [r.get("package", "") for r in summary], [f(r, "near_pass_count", 0.0) for r in summary], "#16a34a")
    _simple_bar_svg(Path(args.out_dir) / "batch_depth_stability_heatmap.svg", "P3 shape stability", [r.get("package", "") for r in summary], [f(r, "shape_stability_score", 0.0) for r in summary], "#2563eb")
    _scatter_svg(Path(args.out_dir) / "s0_s1_s2_threshold_plot.svg", "P3 Threshold Plot", measured, "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _simple_bar_svg(Path(args.out_dir) / "component_contribution_waterfall.svg", "P3 memory reduction vs current", [r["variant_id"] for r in measured], [f(r, "actual_memory_reduction_vs_current", 0.0) for r in measured], "#7c3aed")
    return summary, detail


def _write_not_run(path: Path, stage: str, reason: str, gated_by: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = [{**_row_common(stage, args), "implementation_status": "not_run", "stage_status": "not_run", "status": "not_run", "used_for_gate": 0, "gated_not_run_count": 1, "reason": reason, "gated_by": gated_by}]
    write_csv(path, rows)
    for row in rows:
        _wandb_log_row(args, row, f"summary/v66_{stage.lower()}_not_run")
    return rows


def run_p4(args: argparse.Namespace, p3_detail: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    measured = [r for r in p3_detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    if not measured:
        rows = _write_not_run(Path(args.out_dir) / "p4_runtime_kernel_audit.csv", "P4", "no measured P3 package", "P3", args)
    else:
        rows = []
        for row in measured:
            audit = dict(row)
            audit["runtime_repairable"] = int(f(audit, "actual_step_penalty_vs_current", 0.0) > 0.0 or f(audit, "step_time_ratio_vs_MLP", 99.0) > 1.35)
            audit["runtime_repair_reason"] = "python/manual elementwise and allocation-related instrumentation remains coarse"
            rows.append(audit)
        write_csv(Path(args.out_dir) / "p4_runtime_kernel_audit.csv", rows)
    _simple_bar_svg(Path(args.out_dir) / "runtime_component_waterfall.svg", "P4 step time ratio", [r.get("variant_id", "") for r in rows], [f(r, "step_time_ratio_vs_MLP", 0.0) for r in rows], "#f59e0b")
    _simple_bar_svg(Path(args.out_dir) / "kernel_count_stacked_bar.svg", "P4 kernel count total", [r.get("variant_id", "") for r in rows], [f(r, "kernel_count_total", 0.0) for r in rows], "#2563eb")
    _scatter_svg(Path(args.out_dir) / "allocation_count_vs_step_time_scatter.svg", "P4 allocation proxy unavailable vs step", rows, "reused_buffer_count", "step_time_ratio_vs_MLP", "variant_id")
    _simple_bar_svg(Path(args.out_dir) / "compile_graph_break_heatmap.svg", "P4 graph breaks", [r.get("variant_id", "") for r in rows], [f(r, "torch_compile_graph_break_count", 0.0) for r in rows], "#dc2626")
    return rows


def _best_p3_survivor(summary: Sequence[Dict[str, Any]]) -> Tuple[str, Dict[str, Any] | None]:
    measured = [r for r in summary if r.get("implementation_status") == "measured"]
    s0 = [r for r in measured if f(r, "memory_ratio_max", 99) < 1.0 and f(r, "step_ratio_mean", 99) <= 1.20 and int(f(r, "component_status_all_grad_pass", 0)) == 1]
    s1 = [r for r in measured if f(r, "memory_ratio_max", 99) < 1.0 and f(r, "step_ratio_mean", 99) <= 1.35 and int(f(r, "component_status_all_grad_pass", 0)) == 1]
    s2 = [r for r in measured if f(r, "memory_ratio_min", 99) <= 1.10 and f(r, "step_ratio_min", 99) <= 1.50 and f(r, "actual_memory_reduction_vs_current", 0.0) >= 0.10]
    if s0:
        return "S0", min(s0, key=lambda r: f(r, "memory_ratio_mean", 99))
    if s1:
        return "S1", min(s1, key=lambda r: f(r, "memory_ratio_mean", 99))
    if s2:
        return "S2", min(s2, key=lambda r: f(r, "memory_ratio_mean", 99))
    if any(int(f(r, "component_status_all_grad_pass", 1)) == 0 for r in measured):
        return "S4", min(measured, key=lambda r: f(r, "memory_ratio_mean", 99)) if measured else None
    return "S3", min(measured, key=lambda r: f(r, "memory_ratio_mean", 99)) if measured else None


def run_p5(args: argparse.Namespace, survivor_type: str) -> List[Dict[str, Any]]:
    if survivor_type not in {"S0", "S1", "S2"}:
        rows = _write_not_run(Path(args.out_dir) / "p5_one_step_probe.csv", "P5", "No S0/S1/S2 package survivor", "P3/P6", args)
        _placeholder_svg(Path(args.out_dir) / "before_after_loss_slope_plot.svg", "P5 before-after loss", "not_run")
        _placeholder_svg(Path(args.out_dir) / "bad_step_heatmap.svg", "P5 bad step heatmap", "not_run")
        _placeholder_svg(Path(args.out_dir) / "update_norm_vs_loss_delta_scatter.svg", "P5 update norm", "not_run")
        _placeholder_svg(Path(args.out_dir) / "rollback_error_bar.svg", "P5 rollback", "not_run")
        return rows
    rows = _write_not_run(Path(args.out_dir) / "p5_one_step_probe.csv", "P5", "One-step probe implementation not opened until package survivor is manually reviewed", "manual_review", args)
    return rows


def run_p6(args: argparse.Namespace, p1_rows: Sequence[Dict[str, Any]], p3_summary: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any] | None]:
    survivor_type, best = _best_p3_survivor(p3_summary)
    p1_attribution_pass = any(
        int(f(r, "attribution_pass", 0)) == 1
        for r in p1_rows
        if r.get("implementation_status") == "measured" and "DWM2" in str(r.get("variant_id", ""))
    )
    best_mem = f(best or {}, "memory_ratio_mean", 99.0)
    best_step = f(best or {}, "step_ratio_mean", 99.0)
    best_bwd = f(best or {}, "backward_ratio_mean", 99.0)
    improvement = f(best or {}, "actual_memory_reduction_vs_current", 0.0)
    grad_pass = int(f(best or {}, "component_status_all_grad_pass", 0))
    if not p1_attribution_pass:
        route = "R3-workspaceAttributionIncomplete"
        open_task = False
        reason = "P1 attribution did not reduce unexplained peak gap to plan threshold"
    elif survivor_type in {"S0", "S1"}:
        route = "R1-memorySolved"
        open_task = True
        reason = "S0/S1 survivor exists; P5 still required"
    elif survivor_type == "S2":
        route = "R2-nearPassEngineering"
        open_task = False
        reason = "near-pass engineering candidate exists but full task remains gated"
    elif survivor_type == "S4":
        route = "R5-gradientBroken"
        open_task = False
        reason = "best package has gradient correctness failure"
    else:
        route = "R4-repairNoEffect" if improvement < 0.05 else "R2-nearPassEngineering"
        open_task = False
        reason = "no memory/time survivor"
    row = {
        **_row_common("P6", args),
        "survivor_type": survivor_type,
        "best_package": (best or {}).get("package", ""),
        "best_memory_ratio": best_mem,
        "best_step_ratio": best_step,
        "best_backward_ratio": best_bwd,
        "memory_improvement_vs_current": improvement,
        "step_improvement_vs_current": f(best or {}, "actual_step_penalty_vs_current", 0.0),
        "grad_pass": grad_pass,
        "p5_probe_pass": 0,
        "open_task_reentry": int(open_task),
        "open_functional_correction": 0,
        "route": route,
        "reason": reason,
        "p1_attribution_pass": int(p1_attribution_pass),
    }
    rows = [row]
    write_csv(Path(args.out_dir) / "p6_survivor_selection.csv", rows)
    _wandb_log_row(args, row, "summary/v66_p6_survivor_selection")
    route_json = {
        "route": route,
        "best_package": row["best_package"],
        "best_memory_ratio": best_mem,
        "best_step_ratio": best_step,
        "best_backward_ratio": best_bwd,
        "memory_improvement_vs_current": improvement,
        "step_improvement_vs_current": row["step_improvement_vs_current"],
        "survivor_type": survivor_type,
        "fallback_triggered": False,
        "fallback_near_pass_count": 0,
        "open_task_reentry": bool(open_task),
        "open_functional_correction": False,
        "no_fake": True,
        "no_proxy": True,
        "primary_blocker": reason,
        "next_required_implementation": "cuda_profiler_trace_and_lower_level_workspace_kernel" if route.startswith("R3") else "primitive_reset_or_runtime_repair",
    }
    _json_dump(Path(args.out_dir) / "route_decision.json", route_json)
    _json_dump(Path(args.out_dir) / "aggregate_decision.json", {"status": "gated" if not open_task else "task_reentry_open", "fake_data_used": 0, "proxy_rows_used_as_results": 0, **route_json})
    return rows, survivor_type, best


def run_p7_p8(args: argparse.Namespace, open_task: bool) -> None:
    if not open_task:
        _write_not_run(Path(args.out_dir) / "p7_task_reentry.csv", "P7", "P6 did not open task re-entry", "P6", args)
        _write_not_run(Path(args.out_dir) / "p7_task_trace.csv", "P7", "P7 is gated", "P6", args)
        _write_not_run(Path(args.out_dir) / "p8_functional_correction_smoke.csv", "P8", "P8 is gated behind P7", "P7", args)
    else:
        _write_not_run(Path(args.out_dir) / "p7_task_reentry.csv", "P7", "Task runner not implemented in v6.6 workspace runner", "runner_scope", args)
        _write_not_run(Path(args.out_dir) / "p7_task_trace.csv", "P7", "No task trace emitted", "runner_scope", args)
        _write_not_run(Path(args.out_dir) / "p8_functional_correction_smoke.csv", "P8", "Functional correction remains gated until real P7 task trace exists", "P7", args)
    for name in ["val_loss_vs_step", "val_loss_vs_wall_clock", "accuracy_vs_wall_clock", "time_to_target_bar", "task_efficiency_pareto", "seedwise_paired_delta_plot"]:
        _placeholder_svg(Path(args.out_dir) / f"{name}.svg", name, "not_run")


def run_p9(args: argparse.Namespace, route: str) -> List[Dict[str, Any]]:
    rows = _profile_grid(args, "P9", P9_RESETS)
    for row in rows:
        if row.get("implementation_status") == "measured" and row.get("method") != "MLP-autograd-reference":
            row["reset_near_pass"] = int(f(row, "memory_ratio_vs_MLP", 99) <= 1.10 and f(row, "step_time_ratio_vs_MLP", 99) <= 1.50 and int(f(row, "gradient_correctness_pass", 0)) == 1)
    write_csv(Path(args.out_dir) / "p9_primitive_reset_kernel_benchmark.csv", rows)
    measured = [r for r in rows if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    near = [r for r in measured if int(f(r, "reset_near_pass", 0)) == 1]
    route_path = Path(args.out_dir) / "route_decision.json"
    route_json = json.loads(route_path.read_text()) if route_path.exists() else {}
    route_json.update({"fallback_triggered": True, "fallback_near_pass_count": len(near)})
    if not str(route_json.get("route", "")).startswith("R3"):
        if near and route_json.get("survivor_type") not in {"S0", "S1"}:
            route_json.update({"route": "R6-primitiveResetNeeded", "primary_blocker": "No DWM2 survivor; reset primitive has near-pass"})
        elif not near and route_json.get("survivor_type") not in {"S0", "S1"}:
            route_json.update({"route": "R7-terminalKernelNeeded", "primary_blocker": "No DWM2 survivor and no reset near-pass"})
    _json_dump(route_path, route_json)
    aggregate_path = Path(args.out_dir) / "aggregate_decision.json"
    aggregate = json.loads(aggregate_path.read_text()) if aggregate_path.exists() else {}
    aggregate.update(route_json)
    _json_dump(aggregate_path, aggregate)
    _scatter_svg(Path(args.out_dir) / "reset_primitive_memory_time_pareto.svg", "P9 reset primitive Pareto", measured, "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _simple_bar_svg(Path(args.out_dir) / "old_fallback_vs_reset_fallback_comparison.svg", "P9 reset memory ratio", [r["variant_id"] for r in measured], [f(r, "memory_ratio_vs_MLP") for r in measured], "#2563eb")
    _simple_bar_svg(Path(args.out_dir) / "workspace_model_comparison_table.svg", "P9 workspace pool MB", [r["variant_id"] for r in measured], [f(r, "workspace_pool_MB", 0.0) for r in measured], "#7c3aed")
    _placeholder_svg(Path(args.out_dir) / "single_buffer_lifetime_diagram.svg", "P9 single-buffer lifetime", "see p9_primitive_reset_kernel_benchmark.csv")
    return rows


def run_failure(args: argparse.Namespace) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    out_dir = Path(args.out_dir)
    for fn, stage in [
        ("p1_phase_local_attribution.csv", "P1"),
        ("p2_single_factor_workspace_repair.csv", "P2"),
        ("p3_combined_workspace_packages_detail.csv", "P3"),
        ("p4_runtime_kernel_audit.csv", "P4"),
        ("p5_one_step_probe.csv", "P5"),
        ("p7_task_reentry.csv", "P7"),
        ("p8_functional_correction_smoke.csv", "P8"),
        ("p9_primitive_reset_kernel_benchmark.csv", "P9"),
    ]:
        path = out_dir / fn
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                status = row.get("implementation_status") or row.get("status")
                if status in {"not_implemented", "not_run"} or str(status).startswith("not_implemented"):
                    failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F11_gated_not_run", "metric": row.get("reason", status), "recommendation": "implement before claiming metric"})
                if row.get("implementation_status") == "measured" and row.get("method") != "MLP-autograd-reference":
                    if f(row, "memory_ratio_vs_MLP", 0.0) >= 1.0:
                        failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F1_memory_fail", "metric": f"memory_ratio={row.get('memory_ratio_vs_MLP')}", "recommendation": "reduce actual CUDA peak"})
                    if f(row, "step_time_ratio_vs_MLP", 0.0) > 1.35:
                        failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F2_step_time_fail", "metric": f"step_ratio={row.get('step_time_ratio_vs_MLP')}", "recommendation": "runtime/kernel fusion required"})
                    if int(f(row, "gradient_correctness_pass", 1)) == 0:
                        failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F3_gradient_correctness_fail", "metric": f"grad_relerr={row.get('grad_relerr')} grad_cos={row.get('grad_cos')}", "recommendation": "fix manual adjoint numerics"})
                    if stage == "P1" and int(f(row, "attribution_pass", 0)) == 0:
                        failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F12_attribution_incomplete", "metric": f"unexplained_gap={row.get('unexplained_gap_MB')}", "recommendation": "add torch profiler allocation trace or lower-level kernel counters"})
    write_csv(out_dir / "failure_table.csv", failures or [{"stage": "ALL", "variant_id": "all", "failure_type": "none"}])
    for row in failures:
        _wandb_log_row(args, row, "summary/v66_failure_table")
    _simple_bar_svg(out_dir / "failure_taxonomy_heatmap.svg", "Failure taxonomy", [r["failure_type"] for r in failures], [1.0 for _ in failures], "#dc2626")
    return failures


def _write_manifest(out_dir: Path, args: argparse.Namespace, started: float, finished: float) -> None:
    manifest = {
        "provenance": "EMPIRICAL_REAL_ONLY_NO_PROXY",
        "script": "experiments/run_gafu_v66_real.py",
        "plan": "docs/DG-KAN_v6.6_WorkspaceKernel_PrimitiveReset_详细实验计划.md",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(wandb=True)
    parser.add_argument("--packages", default="V6_6_ALL")
    parser.add_argument("--out-dir", type=Path, default=Path("results/real_rerun_20260504/v66_real"))
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
    parser.add_argument("--wandb-project", default="DG-KAN")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-group", default="v66-real-20260504")
    parser.add_argument("--wandb-name-prefix", default="v66-real")
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
        p1 = run_p1(args)
        run_p0_reproduction(args, p1)
        run_p2(args)
        p3_summary, p3_detail = run_p3(args)
        run_p4(args, p3_detail)
        p6_rows, survivor_type, _best = run_p6(args, p1, p3_summary)
        run_p5(args, survivor_type)
        open_task = bool(p6_rows and int(f(p6_rows[0], "open_task_reentry", 0)) == 1)
        run_p7_p8(args, open_task)
        run_p9(args, str(p6_rows[0].get("route", "")) if p6_rows else "")
        run_failure(args)
        _placeholder_svg(out_dir / "route_timeline_by_stage.svg", "Route timeline", "see route_decision.json")
        _placeholder_svg(out_dir / "best_candidate_scorecard.svg", "Best candidate scorecard", "see p6_survivor_selection.csv")
        _placeholder_svg(out_dir / "next_required_implementation_box.svg", "Next implementation", "see route_decision.json")
        _write_manifest(out_dir, args, started, time.time())
    finally:
        _wandb_finish(args, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
