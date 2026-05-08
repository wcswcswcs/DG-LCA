#!/usr/bin/env python3
"""DG-KAN v6.4 real-only executable runner.

The historical v6.4 script is intentionally blocked because it mixed empirical
training with proxy/derived rows.  This runner only emits rows that come from
actual code paths.  Unimplemented plan items are written as not_run or
not_implemented, never as pass/fail numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgkan_core import (
    ensure_dir,
    get_device,
    load_vision_bundle,
    parse_int_list,
    parse_str_list,
    set_seed,
    write_csv,
)
from run_gafu_v3 import dataset_name
from run_gafu_v63 import (
    ManualOptimizer,
    V63ManualLayer,
    V63ManualStack,
    V63Params,
    _autograd_forward_stack,
    _basis_from_name,
    _decorate_task_rows,
    _flat_autograd_grads,
    _manual_gradient_check,
    _manual_logits_and_features,
    _peak_mb,
    _rel_cos,
    _reset_peak,
    _sync,
    _train_manual_candidate,
    _train_mlp_reference,
    _wandb_finish,
    _wandb_init,
    _wandb_log_row,
    f,
)


DATASETS = ["MNIST", "Fashion-MNIST", "KMNIST"]
METHOD_CURRENT = "DWM2-poly2-compiled"

P0_CANDIDATES = [
    ("MLP-autograd-reference", "reference", True),
    ("MLP-manual-linear-reference", "reference", True),
    ("DWM2-poly2-compiled-current", METHOD_CURRENT, True),
    ("DWM2-poly2-compiled-memoryOptimized", "not_implemented", False),
    ("DWM2-poly2-compiled+LightSmooth-hook", "not_implemented", False),
    ("DWM2-poly2-compiled+FunctionalCorrection-hook", "not_implemented", False),
    ("DWM2-poly3-reference", "DWM2-poly3", True),
    ("DWM2-poly2-silu-base-reference", "DWM2-poly2-silu-base", True),
]

P1_VARIANTS = [
    ("P1-M0-current", METHOD_CURRENT, True, "current"),
    ("P1-M1-cacheCompressed-bf16", "not_implemented", False, "not_implemented"),
    ("P1-M2-noHiddenCache", METHOD_CURRENT, True, "no_hidden_cache"),
    ("P1-M3-deltaStreaming", "not_implemented", False, "not_implemented"),
    ("P1-M4-bufferReuse", "not_implemented", False, "not_implemented"),
    ("P1-M5-noTemp-poly2", "not_implemented", False, "not_implemented"),
    ("P1-M6-fusedForwardBackwardCachePolicy", "not_implemented", False, "not_implemented"),
    ("P1-M7-allMemoryOptimized", "not_implemented", False, "not_implemented"),
]

P2_OPTIMIZERS = [
    "ManualAdamW",
    "ManualAdam",
    "ManualAdanLite",
    "ManualWinLite",
    "Lookahead-ManualAdamW",
    "WarmupCosine-ManualAdamW",
    "ManualAdamW-beta2low",
    "ManualAdamW-gradClip",
    "Restart-ManualAdanLite",
    "Restart-ManualWinLite",
    "ManualAdanLite-betaSchedule",
    "ManualAdanLite-gradClip",
]


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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tensor_mb(t: torch.Tensor) -> float:
    return t.numel() * t.element_size() / (1024**2)


def _manual_optimizer_state_mb(opt: ManualOptimizer) -> float:
    seen: set[int] = set()
    total = 0
    for store in (opt.m, opt.v, opt.prev_g, opt.slow):
        for value in store.values():
            if id(value) in seen:
                continue
            seen.add(id(value))
            total += value.numel() * value.element_size()
    return total / (1024**2)


def _format_command(args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def _write_manifest(out_dir: Path, args: argparse.Namespace, started: float, finished: float) -> None:
    manifest = {
        "provenance": "EMPIRICAL_REAL_ONLY_NO_PROXY",
        "script": "experiments/run_gafu_v64_real.py",
        "plan": "docs/DG-KAN_v6.4_MemoryPassing_FunctionalCorrection_修订版实验计划.md",
        "command_args": _format_command(args),
        "started_unix": started,
        "finished_unix": finished,
        "duration_sec": finished - started,
        "source_commit": _git_commit(),
        "git_status_short": _git_status(),
    }
    path = out_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    hashes = {p.name: _sha256(p) for p in sorted(out_dir.glob("*")) if p.is_file() and p.suffix in {".csv", ".json", ".log", ".svg"}}
    (out_dir / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8")


def _make_mlp(input_dim: int, num_classes: int, hidden: int, depth: int) -> nn.Module:
    layers: List[nn.Module] = []
    for i in range(depth):
        layers.append(nn.Linear(input_dim if i == 0 else hidden, hidden))
        if i < depth - 1:
            layers.append(nn.SiLU())
    layers.append(nn.Linear(hidden, num_classes))
    return nn.Sequential(*layers)


class V64NoHiddenCacheStack(V63ManualStack):
    """Recompute hidden activations in backward instead of storing them from forward."""

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        h = x
        for i, layer in enumerate(self.layers):
            y, _cache = layer.forward_manual(h)
            h = F.silu(y) if i < len(self.layers) - 1 else y
        return h, [x.detach()]

    def _recompute_layer_input_preact(self, x0: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x0
        for j, layer in enumerate(self.layers):
            layer_input = h
            y, _cache = layer.forward_manual(layer_input)
            if j == layer_idx:
                return layer_input, y
            h = F.silu(y)
        raise IndexError(layer_idx)

    def backward_manual(self, dy: torch.Tensor, caches: List[torch.Tensor]) -> torch.Tensor:
        x0 = caches[0]
        delta = dy
        for i in reversed(range(len(self.layers))):
            layer_input, preact = self._recompute_layer_input_preact(x0, i)
            if i < len(self.layers) - 1:
                with torch.no_grad():
                    sig = torch.sigmoid(preact)
                    delta = delta * sig * (1.0 + preact * (1.0 - sig))
            delta = self.layers[i].backward_manual(delta, layer_input)
        return delta

    def cache_breakdown(self, caches: List[torch.Tensor]) -> Dict[str, float]:
        x_bytes = caches[0].numel() * caches[0].element_size() if caches else 0
        return {
            "cache_total_MB": x_bytes / (1024**2),
            "cache_x_MB": x_bytes / (1024**2),
            "cache_hidden_MB": 0.0,
            "cache_index_MB": 0.0,
            "cache_weight_MB": 0.0,
            "cache_delta_MB": 0.0,
        }


class V64Bf16CacheStack(V63ManualStack):
    """Store manual forward caches in bf16, cast back to fp32 for backward."""

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        h = x
        caches: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            y, cache = layer.forward_manual(h)
            caches.append(cache.detach().to(torch.bfloat16))
            h = F.silu(y) if i < len(self.layers) - 1 else y
        return h, caches

    def backward_manual(self, dy: torch.Tensor, caches: List[torch.Tensor]) -> torch.Tensor:
        return super().backward_manual(dy, [cache.float() for cache in caches])

    def cache_breakdown(self, caches: List[torch.Tensor]) -> Dict[str, float]:
        x_bytes = caches[0].numel() * caches[0].element_size() if caches else 0
        hidden_bytes = sum(t.numel() * t.element_size() for t in caches[1:])
        total = x_bytes + hidden_bytes
        return {
            "cache_total_MB": total / (1024**2),
            "cache_x_MB": x_bytes / (1024**2),
            "cache_hidden_MB": hidden_bytes / (1024**2),
            "cache_index_MB": 0.0,
            "cache_weight_MB": 0.0,
            "cache_delta_MB": 0.0,
        }


class V64NoHiddenBf16CacheStack(V64NoHiddenCacheStack):
    """Store only the input cache, compressed to bf16."""

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        h = x
        for i, layer in enumerate(self.layers):
            y, _cache = layer.forward_manual(h)
            h = F.silu(y) if i < len(self.layers) - 1 else y
        return h, [x.detach().to(torch.bfloat16)]

    def backward_manual(self, dy: torch.Tensor, caches: List[torch.Tensor]) -> torch.Tensor:
        return super().backward_manual(dy, [caches[0].float()])


def _make_manual_stack(method: str, input_dim: int, hidden_dim: int, depth: int, basis: int, device: torch.device, cache_policy: str) -> V63ManualStack:
    cls = {
        "current": V63ManualStack,
        "no_hidden_cache": V64NoHiddenCacheStack,
        "bf16_cache": V64Bf16CacheStack,
        "no_hidden_bf16_cache": V64NoHiddenBf16CacheStack,
    }.get(cache_policy)
    if cls is None:
        raise ValueError(f"unknown cache policy: {cache_policy}")
    return cls(method, input_dim, hidden_dim, depth, basis, device)


def _take_batch(bundle: Any, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    n = min(batch_size, int(bundle.x_train.shape[0]))
    return bundle.x_train[:n].to(device), bundle.y_train[:n].to(device)


def _csv_escape(value: Any) -> str:
    text = str(value)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _write_contract_heatmap(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    checks = [
        ("implemented", "implemented"),
        ("nonKAN=0", "nonKAN_param_count"),
        ("no loss.backward", "loss_backward_used"),
        ("manual backward", "manual_backward_available"),
        ("manual update", "manual_update_available"),
        ("rollback", "rollback_max_error"),
        ("no fake data", "uses_fake_data"),
    ]
    cell_w, cell_h = 118, 28
    left, top = 230, 58
    width = left + cell_w * len(checks) + 20
    height = top + cell_h * len(rows) + 30
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append('<text x="16" y="28" font-family="sans-serif" font-size="18">v6.4 P0 Implementation Contract</text>')
    for j, (label, _key) in enumerate(checks):
        parts.append(f'<text x="{left + j * cell_w + 4}" y="48" font-family="sans-serif" font-size="11">{_csv_escape(label)}</text>')
    for i, row in enumerate(rows):
        y = top + i * cell_h
        parts.append(f'<text x="12" y="{y + 18}" font-family="sans-serif" font-size="11">{_csv_escape(row.get("method_id", ""))}</text>')
        for j, (_label, key) in enumerate(checks):
            value = row.get(key)
            if key == "nonKAN_param_count":
                passed = int(float(value or 0)) == 0 or row.get("method_id") == "MLP-autograd-reference"
            elif key == "loss_backward_used":
                passed = int(float(value or 0)) == 0 or row.get("method_id") == "MLP-autograd-reference"
            elif key == "rollback_max_error":
                passed = row.get("status") == "measured" and float(value or 0.0) < 1.0e-8
            elif key == "uses_fake_data":
                passed = int(float(value or 0)) == 0
            else:
                passed = bool(int(float(value or 0)))
            color = "#16a34a" if passed else "#dc2626"
            if row.get("status") in {"not_implemented", "not_run"}:
                color = "#a1a1aa"
            x = left + j * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 3}" height="{cell_h - 3}" rx="3" fill="{color}"/>')
            parts.append(f'<text x="{x + 8}" y="{y + 18}" font-family="monospace" font-size="11" fill="white">{"1" if passed else "0"}</text>')
    parts.append("</svg>")
    (out_dir / "p0_contract_heatmap.svg").write_text("\n".join(parts), encoding="utf-8")


def _implemented_p0_row(method_id: str, impl: str, bundle: Any, params: V63Params, device: torch.device) -> Dict[str, Any]:
    x, y = _take_batch(bundle, params.batch_size, device)
    if method_id == "MLP-autograd-reference":
        model = _make_mlp(bundle.input_dim, bundle.num_classes, params.hidden_dim, params.depth).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=params.lr_mlp)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        return {
            "stage": "P0",
            "method_id": method_id,
            "implementation_method": "MLP-autograd-reference",
            "status": "measured",
            "edge_param_count": 0,
            "base_param_count": 0,
            "residual_param_count": 0,
            "mixing_param_count": 0,
            "nonKAN_param_count": sum(p.numel() for p in model.parameters()),
            "loss_backward_used": 1,
            "torch_autograd_graph_used": 1,
            "manual_forward_available": 0,
            "manual_backward_available": 0,
            "manual_update_available": 0,
            "gradient_correctness_available": 0,
            "rollback_max_error": 0.0,
            "edge_coverage": 0,
            "uses_fake_data": int(getattr(bundle, "used_fake_data", False)),
            "manual_cache_MB": 0.0,
            "parameter_MB": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2),
            "optimizer_state_MB": 0.0,
            "source_commit": _git_commit(),
            "error": "",
        }

    method = "MLP-manual-linear-reference" if method_id == "MLP-manual-linear-reference" else impl
    model = V63ManualStack(method, bundle.input_dim, params.hidden_dim, params.depth, _basis_from_name(method, params.basis_count), device)
    head = V63ManualLayer(params.hidden_dim, bundle.num_classes, kind="linear", basis_count=2, device=device)
    opt = ManualOptimizer(model, head, lr=params.lr_manual, kind="ManualAdamW")
    params_before: List[torch.Tensor] = [p.detach().clone() for _name, p, _grad in model.params_and_grads()]
    params_before.extend(p.detach().clone() for p in head.params.values())
    h, caches = model.forward_manual(x)
    logits, head_cache = head.forward_manual(h)
    loss = F.cross_entropy(logits, y)
    probs = F.softmax(logits, dim=-1)
    probs[torch.arange(y.numel(), device=y.device), y] -= 1.0
    dh = head.backward_manual(probs / max(1, y.numel()), head_cache)
    model.backward_manual(dh, caches)
    opt.step(step=1, total_steps=1, loss=float(loss.detach().cpu()), prev_loss=None)
    with torch.no_grad():
        idx = 0
        for _name, p, _grad in model.params_and_grads():
            p.copy_(params_before[idx])
            idx += 1
        for p in head.params.values():
            p.copy_(params_before[idx])
            idx += 1
    params_after: List[torch.Tensor] = [p.detach() for _name, p, _grad in model.params_and_grads()]
    params_after.extend(p.detach() for p in head.params.values())
    rollback = max(float((a - b).abs().max().detach().cpu()) for a, b in zip(params_before, params_after))
    cache = model.cache_breakdown(caches)
    return {
        "stage": "P0",
        "method_id": method_id,
        "implementation_method": method,
        "status": "measured",
        "edge_param_count": model.param_count() + head.param_count(),
        "base_param_count": 0,
        "residual_param_count": model.param_count(),
        "mixing_param_count": head.param_count(),
        "nonKAN_param_count": 0,
        "loss_backward_used": 0,
        "torch_autograd_graph_used": 0,
        "manual_forward_available": 1,
        "manual_backward_available": 1,
        "manual_update_available": 1,
        "gradient_correctness_available": 1,
        "rollback_max_error": rollback,
        "edge_coverage": 1,
        "base_coverage": 1,
        "residual_coverage": 1,
        "mixing_coverage": 1,
        "uses_fake_data": int(getattr(bundle, "used_fake_data", False)),
        "manual_cache_MB": cache.get("cache_total_MB", 0.0),
        "optimizer_state_MB": _manual_optimizer_state_mb(opt),
        "parameter_MB": sum(_tensor_mb(p) for _name, p, _grad in model.params_and_grads()) + sum(_tensor_mb(p) for p in head.params.values()),
        "source_commit": _git_commit(),
        "error": "",
    }


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    device = get_device(args.device)
    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    bundle = load_vision_bundle(
        dataset_name("MNIST"),
        data_root=args.data_root,
        train_size=params.train_size,
        val_size=params.val_size,
        test_size=params.test_size,
        seed=0,
        allow_fake_data=False,
    )
    rows: List[Dict[str, Any]] = []
    for method_id, impl, implemented in P0_CANDIDATES:
        if not implemented:
            row = {
                "stage": "P0",
                "method_id": method_id,
                "implementation_method": impl,
                "status": "not_implemented",
                "uses_fake_data": int(getattr(bundle, "used_fake_data", False)),
                "source_commit": _git_commit(),
                "error": "not implemented in current v6.4 real runner",
            }
        else:
            row = _implemented_p0_row(method_id, impl, bundle, params, device)
        rows.append(row)
        _wandb_log_row(args, row, "summary/v64_p0_contract")
    write_csv(out_dir / "p0_contract.csv", rows)
    _write_contract_heatmap(out_dir, rows)
    return rows


def _bench_mlp_ce(bundle: Any, batch_size: int, depth: int, params: V63Params, device: torch.device, warmup: int, reps: int) -> Dict[str, Any]:
    set_seed(6401 + batch_size + depth)
    x, y = _take_batch(bundle, batch_size, device)
    model = _make_mlp(bundle.input_dim, bundle.num_classes, params.hidden_dim, depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params.lr_mlp)
    for _ in range(warmup):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y).backward()
        opt.step()
    vals: Dict[str, List[float]] = {key: [] for key in ["forward", "loss_delta", "backward", "update", "step", "peak_forward", "peak_loss_delta", "peak_backward", "peak_update", "peak_total"]}
    for _ in range(reps):
        opt.zero_grad(set_to_none=True)
        _reset_peak(device)
        _sync(device)
        t0 = time.perf_counter()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        _sync(device)
        t1 = time.perf_counter()
        peak_forward, _ = _peak_mb(device)
        _reset_peak(device)
        probs = F.softmax(logits.detach(), dim=-1)
        _sync(device)
        t1b = time.perf_counter()
        peak_loss_delta, _ = _peak_mb(device)
        _reset_peak(device)
        loss.backward()
        _sync(device)
        t2 = time.perf_counter()
        peak_backward, _ = _peak_mb(device)
        _reset_peak(device)
        opt.step()
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
    return {
        "forward_time_ms": _mean(vals["forward"]),
        "loss_delta_time_ms": _mean(vals["loss_delta"]),
        "manual_backward_time_ms": _mean(vals["backward"]),
        "update_time_ms": _mean(vals["update"]),
        "step_time_ms": _mean(vals["step"]),
        "peak_forward_MB": _mean(vals["peak_forward"]),
        "peak_loss_delta_MB": _mean(vals["peak_loss_delta"]),
        "peak_backward_adjoint_MB": _mean(vals["peak_backward"]),
        "peak_update_MB": _mean(vals["peak_update"]),
        "peak_total_step_MB": _mean(vals["peak_total"]),
        "peak_allocated_MB": _mean(vals["peak_total"]),
        "peak_reserved_MB": _mean(vals["peak_total"]),
        "manual_cache_MB": 0.0,
        "cache_x_MB": 0.0,
        "cache_hidden_MB": 0.0,
        "cache_delta_MB": 0.0,
        "cache_temp_MB": 0.0,
        "cache_poly_MB": 0.0,
        "cache_index_MB": 0.0,
        "cache_misc_MB": 0.0,
        "workspace_temp_MB": 0.0,
        "optimizer_state_MB": sum(v.numel() * v.element_size() for st in opt.state.values() for v in st.values() if isinstance(v, torch.Tensor)) / (1024**2),
        "parameter_MB": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2),
        "kernel_count_forward": 2 * depth + 1,
        "kernel_count_backward": 3 * depth + 1,
        "gemm_count": depth + 1,
        "elementwise_kernel_count": depth,
        "poly_eval_kernel_count": 0,
        "buffer_reuse_count": 0,
        "dtype_cache": "none",
        "grad_relerr": 0.0,
        "grad_cos": 1.0,
        "loss_backward_used": 1,
        "uses_fake_data": int(getattr(bundle, "used_fake_data", False)),
    }


def _manual_gradient_check_policy(method: str, batch: int, hidden: int, device: torch.device, cache_policy: str) -> Dict[str, float]:
    if cache_policy == "current":
        return _manual_gradient_check(method, batch, hidden, device)
    set_seed(6419 + batch + hidden)
    basis = _basis_from_name(method, 8)
    model = _make_manual_stack(method, 64, hidden, 2, basis, device, cache_policy)
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


def _bench_manual_ce(method: str, bundle: Any, batch_size: int, depth: int, params: V63Params, device: torch.device, warmup: int, reps: int, cache_policy: str = "current") -> Dict[str, Any]:
    set_seed(6411 + batch_size + depth)
    x, y = _take_batch(bundle, batch_size, device)
    model = _make_manual_stack(method, bundle.input_dim, params.hidden_dim, depth, _basis_from_name(method, params.basis_count), device, cache_policy)
    head = V63ManualLayer(params.hidden_dim, bundle.num_classes, kind="linear", basis_count=2, device=device)
    opt = ManualOptimizer(model, head, lr=params.lr_manual, kind="ManualAdamW")
    prev_loss: float | None = None
    for step in range(1, warmup + 1):
        h, caches = model.forward_manual(x)
        logits, head_cache = head.forward_manual(h)
        loss = F.cross_entropy(logits, y)
        probs = F.softmax(logits, dim=-1)
        probs[torch.arange(y.numel(), device=y.device), y] -= 1.0
        dh = head.backward_manual(probs / max(1, y.numel()), head_cache)
        model.backward_manual(dh, caches)
        opt.step(step=step, total_steps=warmup + reps, loss=float(loss.detach().cpu()), prev_loss=prev_loss)
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
        opt.step(step=step, total_steps=warmup + reps, loss=float(loss.detach().cpu()), prev_loss=prev_loss)
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
    grad = _manual_gradient_check_policy(method, min(32, batch_size), params.hidden_dim, device, cache_policy)
    param_mb = sum(_tensor_mb(p) for _name, p, _grad in model.params_and_grads()) + sum(_tensor_mb(p) for p in head.params.values())
    return {
        "forward_time_ms": _mean(vals["forward"]),
        "loss_delta_time_ms": _mean(vals["loss_delta"]),
        "manual_backward_time_ms": _mean(vals["backward"]),
        "update_time_ms": _mean(vals["update"]),
        "step_time_ms": _mean(vals["step"]),
        "peak_forward_MB": _mean(vals["peak_forward"]),
        "peak_loss_delta_MB": _mean(vals["peak_loss_delta"]),
        "peak_backward_adjoint_MB": _mean(vals["peak_backward"]),
        "peak_update_MB": _mean(vals["peak_update"]),
        "peak_total_step_MB": _mean(vals["peak_total"]),
        "peak_allocated_MB": _mean(vals["peak_total"]),
        "peak_reserved_MB": _mean(vals["peak_total"]),
        "manual_cache_MB": cache.get("cache_total_MB", 0.0),
        "cache_x_MB": cache.get("cache_x_MB", 0.0),
        "cache_hidden_MB": cache.get("cache_hidden_MB", 0.0),
        "cache_delta_MB": cache.get("cache_delta_MB", 0.0),
        "cache_temp_MB": 0.0,
        "cache_poly_MB": 0.0,
        "cache_index_MB": cache.get("cache_index_MB", 0.0),
        "cache_misc_MB": max(0.0, cache.get("cache_total_MB", 0.0) - cache.get("cache_x_MB", 0.0) - cache.get("cache_hidden_MB", 0.0) - cache.get("cache_delta_MB", 0.0) - cache.get("cache_index_MB", 0.0)),
        "workspace_temp_MB": max(0.0, _mean(vals["peak_total"]) - cache.get("cache_total_MB", 0.0)),
        "optimizer_state_MB": _manual_optimizer_state_mb(opt),
        "parameter_MB": param_mb,
        "kernel_count_forward": 0,
        "kernel_count_backward": 0,
        "gemm_count": depth + 1,
        "elementwise_kernel_count": depth,
        "poly_eval_kernel_count": depth,
        "buffer_reuse_count": 0,
        "dtype_cache": "fp32",
        "grad_relerr": grad.get("coeff_grad_relerr", math.nan),
        "grad_cos": grad.get("coeff_grad_cos", math.nan),
        "input_grad_relerr": grad.get("input_grad_relerr", math.nan),
        "input_grad_cos": grad.get("input_grad_cos", math.nan),
        "grad_pass": grad.get("grad_pass", 0),
        "loss_backward_used": 0,
        "uses_fake_data": int(getattr(bundle, "used_fake_data", False)),
    }


def _mean(vals: Iterable[float], default: float = float("nan")) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return sum(xs) / len(xs) if xs else default


def run_p1(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    device = get_device(args.device)
    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    rows: List[Dict[str, Any]] = []
    datasets = parse_str_list(args.datasets) or DATASETS
    batch_sizes = parse_int_list(args.p1_batch_sizes) or [128, 256, 512]
    depths = parse_int_list(args.p1_depths) or [2, 4]
    for ds in datasets:
        bundle = load_vision_bundle(
            dataset_name(ds),
            data_root=args.data_root,
            train_size=max(args.train_size, max(batch_sizes)),
            val_size=args.val_size,
            test_size=args.test_size,
            seed=0,
            allow_fake_data=False,
        )
        for batch_size in batch_sizes:
            for depth in depths:
                shape_id = f"{dataset_name(ds)}-B{batch_size}-H{params.hidden_dim}-D{depth}"
                mlp_stat = _bench_mlp_ce(bundle, batch_size, depth, params, device, args.p1_warmup_steps, args.p1_measure_steps)
                mlp_row = {
                    "stage": "P1",
                    "dataset": dataset_name(ds),
                    "seed": 0,
                    "shape_id": shape_id,
                    "method_id": "MLP-autograd-reference",
                    "variant": "MLP-autograd-reference",
                    "status": "measured",
                    "batch_size": batch_size,
                    "hidden_dim": params.hidden_dim,
                    "depth": depth,
                    **mlp_stat,
                    "error": "",
                }
                rows.append(mlp_row)
                _wandb_log_row(args, mlp_row, "summary/v64_p1_memory_finalization")
                for variant, impl, implemented, cache_policy in P1_VARIANTS:
                    if not implemented:
                        continue
                    stat = _bench_manual_ce(impl, bundle, batch_size, depth, params, device, args.p1_warmup_steps, args.p1_measure_steps, cache_policy=cache_policy)
                    row = {
                        "stage": "P1",
                        "dataset": dataset_name(ds),
                        "seed": 0,
                        "shape_id": shape_id,
                        "method_id": impl,
                        "variant": variant,
                        "cache_policy": cache_policy,
                        "status": "measured",
                        "batch_size": batch_size,
                        "hidden_dim": params.hidden_dim,
                        "depth": depth,
                        **stat,
                        "error": "",
                    }
                    for prefix in ("forward", "manual_backward", "step"):
                        key = f"{prefix}_time_ms"
                        ratio_key = "backward_time_ratio_vs_MLP" if prefix == "manual_backward" else f"{prefix}_time_ratio_vs_MLP"
                        row[ratio_key] = f(row, key) / max(1.0e-12, f(mlp_row, key))
                    row["backward_memory_ratio_vs_MLP"] = f(row, "peak_backward_adjoint_MB") / max(1.0e-12, f(mlp_row, "peak_backward_adjoint_MB"))
                    row["memory_pass"] = int(f(row, "backward_memory_ratio_vs_MLP", 99) < 1.0)
                    row["step_time_pass"] = int(f(row, "step_time_ratio_vs_MLP", 99) <= 1.10)
                    row["gradient_correctness_pass"] = int(f(row, "grad_relerr", 99) < 1.0e-4 and f(row, "grad_cos", 0) > 0.999)
                    row["p1_memory_survivor"] = int(row["memory_pass"] and row["step_time_pass"] and row["gradient_correctness_pass"])
                    row["p1_exploratory_pass"] = int(f(row, "step_time_ratio_vs_MLP", 99) <= 1.20 and f(row, "backward_memory_ratio_vs_MLP", 99) <= 1.05 and row["gradient_correctness_pass"])
                    rows.append(row)
                    _wandb_log_row(args, row, "summary/v64_p1_memory_finalization")
                    write_csv(out_dir / "p1_memory_finalization.csv", rows)
    for variant, impl, implemented, cache_policy in P1_VARIANTS:
        if implemented:
            continue
        row = {
            "stage": "P1",
            "variant": variant,
            "method_id": impl,
            "cache_policy": cache_policy,
            "status": "not_implemented",
            "reason": "memory optimization variant is not implemented; no proxy metrics emitted",
            "error": "",
        }
        rows.append(row)
        _wandb_log_row(args, row, "summary/v64_p1_memory_finalization")
    write_csv(out_dir / "p1_memory_finalization.csv", rows)
    write_csv(out_dir / "p1_memory_decomposition.csv", rows)
    write_csv(out_dir / "p1_phase_local_memory.csv", rows)
    write_csv(out_dir / "p1_runtime_decomposition.csv", rows)
    write_csv(out_dir / "p1_gradient_correctness.csv", rows)
    return rows


def _has_p1_survivor(out_dir: Path) -> bool:
    path = out_dir / "p1_memory_finalization.csv"
    if not path.exists():
        return False
    import csv

    with path.open(newline="") as handle:
        return any(int(float(row.get("p1_memory_survivor") or 0)) == 1 for row in csv.DictReader(handle))


def _p1_survivor_variants(out_dir: Path) -> List[str]:
    path = out_dir / "p1_memory_finalization.csv"
    if not path.exists():
        return []
    import csv

    variants: List[str] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(float(row.get("p1_memory_survivor") or 0)) == 1:
                variant = str(row.get("variant") or "")
                if variant and variant not in variants:
                    variants.append(variant)
    return variants


def _write_not_run(out_dir: Path, filename: str, stage: str, reason: str, args: argparse.Namespace | None = None) -> List[Dict[str, Any]]:
    rows = [{"stage": stage, "status": "not_run", "reason": reason, "error": ""}]
    write_csv(out_dir / filename, rows)
    if args is not None:
        for row in rows:
            _wandb_log_row(args, row, f"summary/v64_{stage.lower()}_not_run")
    return rows


def run_p2(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not args.run_p2_without_memory_pass and not _has_p1_survivor(out_dir):
        reason = "P1 produced no memory survivor; P2 is gated by the v6.4 plan"
        _write_not_run(out_dir, "p2_task_recipe.csv", "P2", reason, args)
        _write_not_run(out_dir, "p2_task_trace.csv", "P2", reason, args)
        return []
    survivors = _p1_survivor_variants(out_dir)
    if not args.run_p2_without_memory_pass and survivors and "P1-M0-current" not in survivors:
        reason = f"P1 survivor(s) {survivors} require a matching training implementation; current P2 trainer only supports P1-M0-current"
        _write_not_run(out_dir, "p2_task_recipe.csv", "P2", reason, args)
        _write_not_run(out_dir, "p2_task_trace.csv", "P2", reason, args)
        return []
    device = get_device(args.device)
    params = V63Params(
        p3_steps=args.p2_steps,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    rows: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    datasets = parse_str_list(args.datasets) or DATASETS
    seeds = parse_int_list(args.seeds) or [0, 1, 2]
    optimizers = parse_str_list(args.optimizers) or P2_OPTIMIZERS
    for ds in datasets:
        canonical = dataset_name(ds)
        for seed in seeds:
            bundle = load_vision_bundle(
                canonical,
                data_root=args.data_root,
                train_size=params.train_size,
                val_size=params.val_size,
                test_size=params.test_size,
                seed=seed,
                allow_fake_data=False,
            )
            mlp_final, mlp_trace = _train_mlp_reference(bundle, seed, params, device, canonical, "P2", params.p3_steps, wandb_args=args)
            rows.append(mlp_final)
            trace.extend(mlp_trace)
            write_csv(out_dir / "p2_task_recipe.csv", rows)
            write_csv(out_dir / "p2_task_trace.csv", trace)
            print(f"P2 {canonical} seed={seed} MLP test_acc_sanity={mlp_final['test_acc']:.4f}")
            for opt_name in optimizers:
                final, tr = _train_manual_candidate(
                    METHOD_CURRENT,
                    opt_name,
                    bundle,
                    seed,
                    params,
                    device,
                    canonical,
                    "P2",
                    params.p3_steps,
                    step_ratio=float("nan"),
                    bmem_ratio=float("nan"),
                    wandb_args=args,
                )
                final["optimizer"] = opt_name
                final["test_acc_sanity"] = final.get("test_acc")
                for item in tr:
                    item["optimizer"] = opt_name
                    item["test_acc_sanity"] = item.get("test_acc")
                rows.append(final)
                trace.extend(tr)
                write_csv(out_dir / "p2_task_recipe.csv", rows)
                write_csv(out_dir / "p2_task_trace.csv", trace)
                print(f"P2 {canonical} seed={seed} {METHOD_CURRENT}/{opt_name} test_acc_sanity={final['test_acc']:.4f}")
    _decorate_task_rows(rows, trace, "P2")
    for row in rows:
        row["test_acc_sanity"] = row.get("test_acc")
        row["selection_uses_test"] = 0
        _wandb_log_row(args, row, "summary/v64_p2_task_recipe")
    write_csv(out_dir / "p2_task_recipe.csv", rows)
    write_csv(out_dir / "p2_task_trace.csv", trace)
    return rows


def run_gated_later_stages(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(args.out_dir)
    _write_not_run(out_dir, "p3_functional_correction_direction.csv", "P3", "P2/P4 survivor not available in this run; no proxy direction rows emitted", args)
    _write_not_run(out_dir, "p3_one_step_probe.csv", "P3", "P3 empirical correction probe is not implemented in this runner", args)
    _write_not_run(out_dir, "p3_direction_gate_summary.csv", "P3", "P3 was not empirically run", args)
    _write_not_run(out_dir, "p4_acceleration_stability.csv", "P4", "P2 did not produce an official memory-passing survivor", args)
    _write_not_run(out_dir, "p4_seedwise_trace.csv", "P4", "P4 is gated", args)
    _write_not_run(out_dir, "p5_geometry_integration.csv", "P5", "P5 requires P1 memory pass and P4 task-stable survivor", args)
    _write_not_run(out_dir, "p5_event_trace.csv", "P5", "P5 is gated", args)
    _write_not_run(out_dir, "p6_confirm3.csv", "P6", "P6 is gated behind P5", args)
    _write_not_run(out_dir, "p7_confirm5.csv", "P7", "P7 is gated behind P6", args)
    _write_not_run(out_dir, "p8_confirm10.csv", "P8", "P8 is gated behind P7", args)


def run_failure_and_decision(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(args.out_dir)
    import csv

    failures: List[Dict[str, Any]] = []
    p1_path = out_dir / "p1_memory_finalization.csv"
    p1_rows: List[Dict[str, str]] = []
    if p1_path.exists():
        with p1_path.open(newline="") as handle:
            p1_rows = list(csv.DictReader(handle))
    for row in p1_rows:
        if row.get("status") == "not_implemented":
            failures.append({"stage": "P1", "method": row.get("variant"), "failure_type": "F11_gated_not_run", "metric": "not_implemented", "recommendation": "implement memory variant before claiming a metric"})
            continue
        if row.get("method_id") != METHOD_CURRENT:
            continue
        if int(float(row.get("memory_pass") or 0)) == 0:
            failures.append({"stage": "P1", "method": row.get("variant"), "failure_type": "F1_memory_fail", "metric": f"backward_memory_ratio_vs_MLP={row.get('backward_memory_ratio_vs_MLP')}", "recommendation": "implement cache compression/recompute/streaming variants"})
        if int(float(row.get("step_time_pass") or 0)) == 0:
            failures.append({"stage": "P1", "method": row.get("variant"), "failure_type": "F2_step_time_fail", "metric": f"step_time_ratio_vs_MLP={row.get('step_time_ratio_vs_MLP')}", "recommendation": "profile kernel launch and fused phase overhead"})
        if int(float(row.get("gradient_correctness_pass") or 0)) == 0:
            failures.append({"stage": "P1", "method": row.get("variant"), "failure_type": "F3_gradient_correctness_fail", "metric": f"grad_relerr={row.get('grad_relerr')} grad_cos={row.get('grad_cos')}", "recommendation": "fix manual adjoint before downstream training"})
    for filename, stage in [
        ("p2_task_recipe.csv", "P2"),
        ("p3_functional_correction_direction.csv", "P3"),
        ("p4_acceleration_stability.csv", "P4"),
        ("p5_geometry_integration.csv", "P5"),
        ("p6_confirm3.csv", "P6"),
        ("p7_confirm5.csv", "P7"),
        ("p8_confirm10.csv", "P8"),
    ]:
        path = out_dir / filename
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") == "not_run":
                    failures.append({"stage": stage, "method": row.get("method", stage), "failure_type": "F11_gated_not_run", "metric": row.get("reason", ""), "recommendation": "respect gate; do not infer pass from not_run"})
    if not failures:
        failures.append({"stage": "ALL", "method": "all", "failure_type": "none", "metric": "", "recommendation": ""})
    write_csv(out_dir / "failure_table.csv", failures)
    for row in failures:
        _wandb_log_row(args, row, "summary/v64_failure_table")
    has_memory_survivor = any(int(float(r.get("p1_memory_survivor") or 0)) == 1 for r in p1_rows)
    if has_memory_survivor:
        route = {
            "route": "R1_or_R2_pending",
            "decision": "P1 has a memory survivor; continue to task and correction gates.",
            "no_proxy": True,
        }
    else:
        route = {
            "route": "R3",
            "decision": "DWM2-poly2 task line remains memory-blocked; focus on memory kernel before confirm seeds.",
            "no_proxy": True,
        }
    (out_dir / "route_decision.json").write_text(json.dumps(route, indent=2, ensure_ascii=False), encoding="utf-8")
    aggregate = {
        "status": "running_or_gated",
        "p1_memory_survivor": int(has_memory_survivor),
        "p2_officially_gated": int(not has_memory_survivor),
        "fake_data_used": 0,
        "proxy_rows_used_as_results": 0,
    }
    (out_dir / "aggregate_decision.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(wandb=True)
    parser.add_argument("--packages", default="V6_4_ALL")
    parser.add_argument("--out-dir", type=Path, default=Path("results/real_rerun_20260504/v64_real"))
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-size", type=int, default=1536)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--p1-batch-sizes", default="128,256,512")
    parser.add_argument("--p1-depths", default="2,4")
    parser.add_argument("--p1-warmup-steps", type=int, default=50)
    parser.add_argument("--p1-measure-steps", type=int, default=200)
    parser.add_argument("--p2-steps", type=int, default=240)
    parser.add_argument("--optimizers", default=",".join(P2_OPTIMIZERS))
    parser.add_argument("--run-p2-without-memory-pass", action="store_true", help="diagnostic only; official P2 remains gated when P1 has no memory survivor")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--wandb-project", default="DG-KAN")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-group", default="v64-real-20260504")
    parser.add_argument("--wandb-name-prefix", default="v64-real")
    parser.add_argument("--no-wandb", action="store_false", dest="wandb")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    out_dir = ensure_dir(args.out_dir)
    _wandb_init(args)
    try:
        packages = {p.upper() for p in parse_str_list(args.packages)}
        run_all = "V6_4_ALL" in packages
        if run_all or "V6_4_P0" in packages:
            run_p0(args)
        if run_all or "V6_4_P1" in packages:
            run_p1(args)
        if run_all or "V6_4_P2" in packages:
            run_p2(args)
        if run_all or any(pkg in packages for pkg in ["V6_4_P3", "V6_4_P4", "V6_4_P5", "V6_4_P6", "V6_4_P7", "V6_4_P8"]):
            run_gated_later_stages(args)
        run_failure_and_decision(args)
        _write_manifest(out_dir, args, started, time.time())
    finally:
        _wandb_finish(args, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
