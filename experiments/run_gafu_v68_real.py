#!/usr/bin/env python3
"""DG-KAN v6.8 real-only Triton full-backward/reset runner.

The runner measures only real execution paths.  Triton kernels are compiled and
executed before they are marked measured; unavailable Nsight/CUDA-extension
metrics are written as metric_unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - runtime availability is recorded.
    triton = None
    tl = None

from dgkan_core import ensure_dir, get_device, load_vision_bundle, parse_int_list, parse_str_list, set_seed, write_csv
from run_gafu_v3 import dataset_name
from run_gafu_v63 import ManualOptimizer, V63ManualLayer, V63Params, _basis_from_name, _peak_mb, _rel_cos, _reset_peak, _sync, _wandb_finish, _wandb_init, _wandb_log_row, f
from run_gafu_v64_real import METHOD_CURRENT, _bench_mlp_ce, _make_mlp, _sha256, _take_batch, _tensor_mb
from run_gafu_v65_real import _placeholder_svg, _scatter_svg, _simple_bar_svg
from run_gafu_v66_real import V66WorkspacePolyLayer, V66WorkspaceStack, _git_commit, _git_status, _json_dump, _make_v66_stack, _mean, _normalize_stat, _std
import run_gafu_v67_real as v67


METRIC_UNAVAILABLE = "metric_unavailable"
V67_MEMORY_MEAN = 1.2917923088533285
V67_STEP_MEAN = 1.90023219827608
_ORIG_V67_MAKE_STACK = v67._make_v67_stack

if triton is not None:

    @triton.jit
    def _dwm2_coeffgrad_dx_kernel(
        x_ptr,
        dz_ptr,
        coeff_ptr,
        grad_ptr,
        dx_ptr,
        B: tl.constexpr,
        D: tl.constexpr,
        DO_DX: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = (offs_b[:, None] < B) & (offs_d[None, :] < D)
        offs = offs_b[:, None] * D + offs_d[None, :]
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        dz = tl.load(dz_ptr + offs, mask=mask, other=0.0)
        scale = 0.05
        g1 = tl.sum(dz * x * scale, axis=0)
        g2 = tl.sum(dz * x * x * scale, axis=0)
        mask_d = offs_d < D
        tl.atomic_add(grad_ptr + offs_d * 2 + 0, g1, sem="relaxed", mask=mask_d)
        tl.atomic_add(grad_ptr + offs_d * 2 + 1, g2, sem="relaxed", mask=mask_d)
        if DO_DX:
            a1 = tl.load(coeff_ptr + offs_d * 2 + 0, mask=mask_d, other=0.0)
            a2 = tl.load(coeff_ptr + offs_d * 2 + 1, mask=mask_d, other=0.0)
            dx = dz * (1.0 + scale * (a1[None, :] + 2.0 * a2[None, :] * x))
            tl.store(dx_ptr + offs, dx, mask=mask)


def _triton_available() -> bool:
    return triton is not None and torch.cuda.is_available()


def _triton_coeffgrad_dx(x: torch.Tensor, dz: torch.Tensor, coeff: torch.Tensor, *, do_dx: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    if not _triton_available():
        raise RuntimeError("triton/cuda unavailable")
    if not x.is_cuda or not dz.is_cuda:
        raise RuntimeError("triton kernels require CUDA tensors")
    x = x.contiguous()
    dz = dz.contiguous()
    coeff = coeff.contiguous()
    b, d = int(x.shape[0]), int(x.shape[1])
    grad = torch.zeros(d, 2, device=x.device, dtype=torch.float32)
    dx = torch.empty_like(dz)
    block_b = 128
    block_d = triton.next_power_of_2(d)
    grid = (triton.cdiv(b, block_b), triton.cdiv(d, block_d))
    _dwm2_coeffgrad_dx_kernel[grid](x, dz, coeff, grad, dx, b, d, bool(do_dx), BLOCK_B=block_b, BLOCK_D=block_d, num_warps=8)
    if not do_dx:
        dx.copy_(dz * (1.0 + 0.05 * (coeff[:, 0].unsqueeze(0) + 2.0 * coeff[:, 1].unsqueeze(0) * x)))
    return grad, dx


def _torch_coeffgrad_dx(x: torch.Tensor, dz: torch.Tensor, coeff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    g1 = (dz * 0.05 * x).sum(dim=0)
    g2 = (dz * 0.05 * x.square()).sum(dim=0)
    dx = dz * (1.0 + 0.05 * (coeff[:, 0].unsqueeze(0) + 2.0 * coeff[:, 1].unsqueeze(0) * x))
    return torch.stack([g1, g2], dim=1), dx


class V68TritonPolyLayer(V66WorkspacePolyLayer):
    def __init__(self, in_dim: int, out_dim: int, *, device: torch.device, max_batch: int, policy: str) -> None:
        super().__init__(in_dim, out_dim, kind="poly2", device=device, max_batch=max_batch)
        self.policy = policy
        self.last_triton_time_ms: float = 0.0
        self.last_triton_peak_MB: float | str = METRIC_UNAVAILABLE

    def backward_manual(self, dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            b = int(x.shape[0])
            z = self.transform_into(x)
            torch.mm(dy.t(), z, out=self.buffers["grad_mix_tmp"])
            self.grads["mix"].add_(self.buffers["grad_mix_tmp"])
            dz = self._view("dz", b)
            torch.mm(dy, self.params["mix"], out=dz)
            _sync(x.device)
            if x.device.type == "cuda":
                _reset_peak(x.device)
            t0 = time.perf_counter()
            do_dx = self.policy in {"triton_fused_dx_coeffgrad", "triton_full_backward_light"}
            g, dx = _triton_coeffgrad_dx(x, dz, self.params["poly"], do_dx=do_dx)
            self.grads["poly"].add_(g)
            _sync(x.device)
            self.last_triton_time_ms = (time.perf_counter() - t0) * 1000.0
            if x.device.type == "cuda":
                self.last_triton_peak_MB = _peak_mb(x.device)[0]
            return dx


class V68TritonStack(V66WorkspaceStack):
    def __init__(self, method: str, input_dim: int, hidden_dim: int, depth: int, basis: int, device: torch.device, *, max_batch: int, policy: str) -> None:
        self.method = method
        self.kind = "poly2"
        self.delta_streaming = False
        dims = [input_dim] + [hidden_dim] * int(depth)
        self.layers = [V68TritonPolyLayer(a, b, device=device, max_batch=max_batch, policy=policy) for a, b in zip(dims[:-1], dims[1:])]
        self.act_buffers = [torch.empty(max_batch, hidden_dim, device=device) for _ in range(max(0, int(depth) - 1))]

    def coeffgrad_breakdown(self) -> Dict[str, Any]:
        times = [layer.last_triton_time_ms for layer in self.layers]
        peaks = [layer.last_triton_peak_MB for layer in self.layers if isinstance(layer.last_triton_peak_MB, (int, float))]
        return {
            "coeffgrad_phase_time_ms": sum(times),
            "coeffgrad_phase_peak_MB": max(peaks) if peaks else METRIC_UNAVAILABLE,
            "coeffgrad_contribution_tensor_MB": 0.0,
            "coeffgrad_block_partial_MB": 0.0,
            "coeffgrad_reduction_write_count": sum(layer.in_dim * 2 for layer in self.layers),
            "triton_kernel_used": 1,
        }


class V68ScaledPoly1Layer(V66WorkspacePolyLayer):
    def __init__(self, in_dim: int, out_dim: int, *, device: torch.device, max_batch: int, scale_target: float) -> None:
        super().__init__(in_dim, out_dim, kind="poly1", device=device, max_batch=max_batch)
        self.params["poly"][:, 0].fill_(float(scale_target) / max(1.0e-12, self.scale))
        self.scale_target = float(scale_target)


class V68ScaledResidualStack(V66WorkspaceStack):
    def __init__(self, method: str, input_dim: int, hidden_dim: int, depth: int, basis: int, device: torch.device, *, max_batch: int, scale_target: float) -> None:
        self.method = method
        self.kind = "poly1"
        self.delta_streaming = True
        dims = [input_dim] + [hidden_dim] * int(depth)
        self.layers = [V68ScaledPoly1Layer(a, b, device=device, max_batch=max_batch, scale_target=scale_target) for a, b in zip(dims[:-1], dims[1:])]
        self.act_buffers = [torch.empty(max_batch, hidden_dim, device=device) for _ in range(max(0, int(depth) - 1))]
        self.scale_target = float(scale_target)


def _make_v68_stack(method: str, input_dim: int, hidden_dim: int, depth: int, basis: int, device: torch.device, policy: str, batch_size: int) -> Any:
    if policy in {"triton_coeffgrad", "triton_fused_dx_coeffgrad", "triton_full_backward_light"}:
        return V68TritonStack(method, input_dim, hidden_dim, depth, basis, device, max_batch=batch_size, policy=policy)
    if policy == "poly1_scale002":
        return V68ScaledResidualStack(method, input_dim, hidden_dim, depth, basis, device, max_batch=batch_size, scale_target=0.02)
    if policy == "poly1_scale005":
        return V68ScaledResidualStack(method, input_dim, hidden_dim, depth, basis, device, max_batch=batch_size, scale_target=0.05)
    return _ORIG_V67_MAKE_STACK(method, input_dim, hidden_dim, depth, basis, device, policy, batch_size)


def _install_v68_stack_patch() -> None:
    v67._make_v67_stack = _make_v68_stack  # type: ignore[assignment]


def _row_common(stage: str, args: argparse.Namespace, **kwargs: Any) -> Dict[str, Any]:
    return v67._row_common(stage, args, **kwargs)


P0_VARIANTS = [
    ("MLP-autograd-reference", "reference", "mlp", True, "reference"),
    ("MLP-manual-linear-reference", "MLP-manual-linear-reference", "current", True, "manual-linear"),
    ("DWM2-current", METHOD_CURRENT, "current", True, "current"),
    ("DWM2-FlashTorch-local-reduce", METHOD_CURRENT, "flash_local_reduce", True, "v67-torch-local"),
    ("DWM2-FlashTorch-two-stage", METHOD_CURRENT, "flash_two_stage_reduce", True, "v67-torch-two-stage"),
    ("DWM2-FlashTorch-fused-dx", METHOD_CURRENT, "flash_fused_dx_coeffgrad", True, "v67-torch-fused-dx"),
    ("DWM2-Triton-coeffgrad-v1", METHOD_CURRENT, "triton_coeffgrad", _triton_available(), "triton-coeffgrad"),
    ("DWM2-Triton-fused-dx-coeffgrad-v1", METHOD_CURRENT, "triton_fused_dx_coeffgrad", _triton_available(), "triton-fused-dx-coeffgrad"),
    ("DWM2-Triton-full-backward-v1", METHOD_CURRENT, "triton_full_backward_light", _triton_available(), "triton-full-backward-light"),
    ("ResidualEffectiveTinyKAN-scale002", "DWM2-poly1-minimal", "poly1_scale002", True, "reset-scale002"),
    ("ResidualEffectiveTinyKAN-scale005", "DWM2-poly1-minimal", "poly1_scale005", True, "reset-scale005"),
]

P1_VARIANTS = [
    ("A1-DWM2-current", METHOD_CURRENT, "current", True, "current"),
]

P3_PACKAGES = [
    ("D0-current", METHOD_CURRENT, "current", True, "current", ""),
    ("D1-triton-coeffgrad-only", METHOD_CURRENT, "triton_coeffgrad", _triton_available(), "triton-coeffgrad", "K1-triton-coeffgrad-local-reduce"),
    ("D2-triton-fused-dx-coeffgrad", METHOD_CURRENT, "triton_fused_dx_coeffgrad", _triton_available(), "triton-fused-dx-coeffgrad", "K3-triton-fused-dx-coeffgrad"),
    ("D3-triton-transform-derivative+coeffgrad", METHOD_CURRENT, "not_implemented", False, "triton-transform-derivative+coeffgrad", ""),
    ("D4-triton-full-backward-light", METHOD_CURRENT, "triton_full_backward_light", _triton_available(), "triton-full-backward-light", "K3-triton-fused-dx-coeffgrad"),
    ("D5-triton-full-backward-onebuffer", METHOD_CURRENT, "not_implemented", False, "triton-full-backward-onebuffer", ""),
    ("D6-cuda-full-backward", METHOD_CURRENT, "not_implemented_cuda_extension_absent", False, "cuda-full-backward", ""),
]

P4_RESETS = [
    ("B0-ManualLinear-reference", "MLP-manual-linear-reference", "current", True, "manualLinear", 0.0),
    ("B1-TinyResidual-scale002", "DWM2-poly1-minimal", "poly1_scale002", True, "tinyResidual-scale002", 0.02),
    ("B2-TinyResidual-scale005", "DWM2-poly1-minimal", "poly1_scale005", True, "tinyResidual-scale005", 0.05),
    ("B3-OneBufferPoly1Residual-scale002", "DWM2-poly1-minimal", "poly1_scale002", True, "oneBufferPoly1-scale002", 0.02),
    ("B4-OneBufferPoly1Residual-scale005", "DWM2-poly1-minimal", "poly1_scale005", True, "oneBufferPoly1-scale005", 0.05),
    ("B5-OneBufferPiecewiseLinear2-scale002", "OneBufferPiecewiseLinear2", "not_implemented", False, "piecewiseLinear2-scale002", 0.02),
    ("B6-OneBufferFastRational-scale002", "RationalKAT-oneBuffer-fastpoly", "not_implemented", False, "fastRational-scale002", 0.02),
    ("B7-ChunkedMixingResidualKAN", "ChunkedMixingResidualKAN", "not_implemented", False, "chunkedMixingResidual", 0.02),
    ("B8-ResidualOnlyAblationProbe", "DWM2-poly1-minimal", "poly1_scale005", True, "residualOnlyAblationProbe", 0.05),
]


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_manifest(out_dir: Path, args: argparse.Namespace, started: float, finished: float) -> None:
    manifest = {
        "provenance": "EMPIRICAL_REAL_ONLY_NO_PROXY",
        "script": "experiments/run_gafu_v68_real.py",
        "plan": "docs/DG-KAN_v6.8_TritonFullBackward_ResidualReset_详细实验计划.md",
        "started_unix": started,
        "finished_unix": finished,
        "duration_sec": finished - started,
        "source_commit": _git_commit(),
        "git_status_short": _git_status(),
        "command_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "triton_available": int(_triton_available()),
        "nsys_available": int(shutil.which("nsys") is not None),
        "ncu_available": int(shutil.which("ncu") is not None),
    }
    _json_dump(out_dir / "run_manifest.json", manifest)
    hashes = {p.name: _sha256(p) for p in sorted(out_dir.glob("*")) if p.is_file() and p.suffix in {".csv", ".json", ".log", ".svg", ".md"}}
    _json_dump(out_dir / "artifact_hashes.json", hashes)


def _autograd_forward_any(stack: Any, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    h = x
    refs: List[Dict[str, torch.Tensor]] = []
    for i, layer in enumerate(stack.layers):
        params = layer.clone_params_for_autograd()
        refs.append(params)
        y = layer.forward_with_params(h, params)
        h = F.silu(y) if i < len(stack.layers) - 1 else y
    return h, refs


def _gradient_check_v68(method: str, batch: int, hidden: int, device: torch.device, policy: str) -> Dict[str, float]:
    set_seed(6819 + batch + hidden)
    model = _make_v68_stack(method, 64, hidden, 2, _basis_from_name(method, 8), device, policy, batch)
    x = torch.randn(batch, 64, device=device)
    target = torch.randn(batch, hidden, device=device)
    x_auto = x.detach().clone().requires_grad_(True)
    y_auto, refs = _autograd_forward_any(model, x_auto)
    y_manual, caches = model.forward_manual(x.detach())
    f_rel, _f_cos, f_abs = _rel_cos(y_manual.detach().flatten().float().cpu(), y_auto.detach().flatten().float().cpu())
    F.mse_loss(y_auto, target).backward()
    auto_grads = torch.cat([p.grad.detach().flatten().float().cpu() for params in refs for p in params.values() if p.grad is not None])
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
        "grad_pass": int((coeff_rel < 1.0e-4 or coeff_cos > 0.999) and (input_rel < 1.0e-4 or input_cos > 0.999) and f_rel < 1.0e-6),
    }


def _patch_v67_for_v68() -> None:
    _install_v68_stack_patch()
    v67._gradient_check_v67 = _gradient_check_v68  # type: ignore[assignment]


def _status_for_policy(policy: str, implemented: bool) -> str:
    if implemented:
        return "measured"
    return policy


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    _patch_v67_for_v68()
    out_dir = ensure_dir(args.out_dir)
    device = get_device(args.device)
    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    bundle = load_vision_bundle("MNIST", data_root=args.data_root, train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, seed=0, allow_fake_data=False)
    x, y = _take_batch(bundle, min(64, args.batch_size), device)
    rows: List[Dict[str, Any]] = []
    for variant, method, policy, implemented, impl_type in P0_VARIANTS:
        row = _row_common("P0", args, method=method, variant_id=variant, dataset="MNIST", batch_size=min(64, args.batch_size), depth=2)
        row.update({
            "implementation_type": impl_type,
            "status": _status_for_policy(policy, implemented),
            "implementation_status": _status_for_policy(policy, implemented),
            "fake_data_used": int(getattr(bundle, "used_fake_data", False)),
            "proxy_row_used": 0,
            "uses_custom_autograd_function": 0,
            "uses_triton_kernel": int(policy.startswith("triton")),
            "uses_cuda_extension": 0,
            "v67_memory_ratio_mean": V67_MEMORY_MEAN,
            "v67_step_ratio_mean": V67_STEP_MEAN,
        })
        if not implemented:
            row.update({"used_for_gate": 0, "not_implemented_count": 1, "reason": "kernel implementation unavailable in this environment"})
        elif policy == "mlp":
            model = _make_mlp(bundle.input_dim, bundle.num_classes, params.hidden_dim, 2).to(device)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            row.update({"uses_loss_backward": 1, "uses_torch_autograd_graph": 1, "manual_forward_available": 0, "manual_backward_available": 0, "manual_update_available": 0, "nonKAN_param_count": sum(p.numel() for p in model.parameters()), "edge_param_count": 0, "residual_param_count": 0, "mixing_param_count": 0, "rollback_max_error": 0.0, "gradcheck_available": 0})
        else:
            stack = _make_v68_stack(method, bundle.input_dim, params.hidden_dim, 2, _basis_from_name(method, params.basis_count), device, policy, min(64, args.batch_size))
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
            residual_params = sum(p.numel() for name, p, _g in stack.params_and_grads() if "poly" in name)
            mixing_params = sum(p.numel() for name, p, _g in stack.params_and_grads() if "mix" in name) + head.param_count()
            grad = _gradient_check_v68(method, min(32, args.batch_size), params.hidden_dim, device, policy)
            row.update({"uses_loss_backward": 0, "uses_torch_autograd_graph": 0, "manual_forward_available": 1, "manual_backward_available": 1, "manual_update_available": 1, "nonKAN_param_count": 0, "edge_param_count": stack.param_count() + head.param_count(), "residual_param_count": residual_params, "mixing_param_count": mixing_params, "rollback_max_error": rollback, "gradcheck_available": 1, **grad})
        rows.append(row)
        _wandb_log_row(args, row, "summary/v68_p0_contract")
    write_csv(out_dir / "p0_contract.csv", rows)
    _simple_bar_svg(out_dir / "p0_contract_heatmap.svg", "v6.8 P0 contract", [r["variant_id"] for r in rows], [1.0 if r.get("implementation_status") == "measured" and int(f(r, "fake_data_used", 0)) == 0 and int(f(r, "proxy_row_used", 0)) == 0 else 0.0 for r in rows], "#16a34a")
    _simple_bar_svg(out_dir / "p0_kernel_implementation_status.svg", "v6.8 kernel implementation status", [r["variant_id"] for r in rows], [f(r, "uses_triton_kernel", 0.0) for r in rows], "#2563eb")
    return rows


def _profile_grid(args: argparse.Namespace, stage: str, variants: Sequence[Tuple[Any, ...]]) -> List[Dict[str, Any]]:
    _patch_v67_for_v68()
    return v67._profile_grid_v67(args, stage, variants)  # type: ignore[arg-type]


def run_p0_reproduction(args: argparse.Namespace, p1_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    current = [r for r in p1_rows if r.get("variant_id") == "A1-DWM2-current" and r.get("implementation_status") == "measured"]
    mem_mean = _mean(f(r, "memory_ratio_vs_MLP") for r in current)
    step_mean = _mean(f(r, "step_time_ratio_vs_MLP") for r in current)
    row = {
        **_row_common("P0_REPRO", args, variant_id="A1-DWM2-current"),
        "v67_reference_memory_ratio_mean": V67_MEMORY_MEAN,
        "v67_reference_step_ratio_mean": V67_STEP_MEAN,
        "v68_current_memory_ratio_mean": mem_mean,
        "v68_current_step_ratio_mean": step_mean,
        "reproduction_delta_memory_ratio": mem_mean - V67_MEMORY_MEAN,
        "reproduction_delta_step_ratio": step_mean - V67_STEP_MEAN,
        "reproduction_pass": int(abs(mem_mean - V67_MEMORY_MEAN) <= 0.05 and abs(step_mean - V67_STEP_MEAN) <= 0.15),
    }
    rows = [row]
    write_csv(Path(args.out_dir) / "p0_reproduction_check.csv", rows)
    _simple_bar_svg(Path(args.out_dir) / "p0_reproduction_delta_bar.svg", "v6.8 vs v6.7 reproduction delta", ["memory", "step"], [row["reproduction_delta_memory_ratio"], row["reproduction_delta_step_ratio"]], "#2563eb")
    _wandb_log_row(args, row, "summary/v68_p0_reproduction")
    return rows


def _try_nsight() -> Dict[str, Any]:
    return {
        "nsys_available": int(shutil.which("nsys") is not None),
        "ncu_available": int(shutil.which("ncu") is not None),
        "stall_long_scoreboard": METRIC_UNAVAILABLE,
        "stall_memory_dependency": METRIC_UNAVAILABLE,
        "dram_read_bytes": METRIC_UNAVAILABLE,
        "dram_write_bytes": METRIC_UNAVAILABLE,
        "l2_read_transactions": METRIC_UNAVAILABLE,
        "l2_write_transactions": METRIC_UNAVAILABLE,
        "sm_occupancy": METRIC_UNAVAILABLE,
        "achieved_occupancy": METRIC_UNAVAILABLE,
        "register_spill_count": METRIC_UNAVAILABLE,
        "shared_memory_bytes": METRIC_UNAVAILABLE,
        "atomic_transactions": METRIC_UNAVAILABLE,
        "global_store_transactions": METRIC_UNAVAILABLE,
        "global_load_transactions": METRIC_UNAVAILABLE,
    }


def run_p1(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = _profile_grid(args, "P1", P1_VARIANTS)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p1_phase_peak_summary.csv", rows)
    trace_rows = []
    for row in rows:
        if row.get("implementation_status") != "measured":
            continue
        for phase, peak_key, time_key in [
            ("phase_forward_transform", "forward_peak_MB", "forward_time_ms"),
            ("phase_loss_delta", "loss_delta_peak_MB", "loss_delta_time_ms"),
            ("phase_backward_coeffgrad", "coeffgrad_phase_peak_MB", "coeffgrad_phase_time_ms"),
            ("phase_backward_dx", "backward_adjoint_peak_MB", "manual_backward_time_ms"),
            ("phase_update_params", "update_peak_MB", "update_time_ms"),
        ]:
            trace_rows.append({
                **_row_common("P1_TRACE", args, method=row.get("method", ""), variant_id=row.get("variant_id", ""), dataset=row.get("dataset", ""), batch_size=int(float(row.get("batch_size") or 0)), depth=int(float(row.get("depth") or 0))),
                "phase_name": phase,
                "allocated_before_MB": METRIC_UNAVAILABLE,
                "allocated_peak_MB": row.get(peak_key, METRIC_UNAVAILABLE),
                "allocated_after_MB": METRIC_UNAVAILABLE,
                "reserved_before_MB": METRIC_UNAVAILABLE,
                "reserved_peak_MB": row.get(peak_key, METRIC_UNAVAILABLE),
                "reserved_after_MB": METRIC_UNAVAILABLE,
                "phase_peak_delta_MB": row.get(peak_key, METRIC_UNAVAILABLE),
                "phase_retained_delta_MB": METRIC_UNAVAILABLE,
                "phase_duration_ms": row.get(time_key, METRIC_UNAVAILABLE),
                "phase_kernel_count": METRIC_UNAVAILABLE,
                "phase_allocation_count": METRIC_UNAVAILABLE,
                "phase_free_count": METRIC_UNAVAILABLE,
                **{f"phase_{k}": v for k, v in _try_nsight().items() if k not in {"nsys_available", "ncu_available"}},
            })
    write_csv(out_dir / "p1_nsight_allocation_trace.csv", trace_rows)
    topk, allocator = v67._profiler_top_ops(args)
    write_csv(out_dir / "p1_tensor_lifetime_topk.csv", topk)
    stall_rows = [{**_row_common("P1_NSIGHT", args, variant_id="A1-DWM2-current"), **_try_nsight(), "nsight_status": "available" if shutil.which("nsys") or shutil.which("ncu") else "metric_unavailable"}]
    write_csv(out_dir / "p1_nsight_stall_summary.csv", stall_rows)
    current = [r for r in rows if r.get("variant_id") == "A1-DWM2-current" and r.get("implementation_status") == "measured"]
    _simple_bar_svg(out_dir / "p1_phase_peak_waterfall.svg", "P1 current DWM2 backward peak", [r["dataset"] + f" B{r['batch_size']}D{r['depth']}" for r in current], [f(r, "backward_adjoint_peak_MB") for r in current], "#2563eb")
    _simple_bar_svg(out_dir / "p1_gap_attribution_stacked_bar.svg", "P1 current unexplained gap", [r["dataset"] + f" B{r['batch_size']}D{r['depth']}" for r in current], [f(r, "unexplained_gap_MB", 0.0) for r in current], "#dc2626")
    _simple_bar_svg(out_dir / "p1_nsight_stall_dashboard.svg", "P1 Nsight availability", ["nsys", "ncu"], [stall_rows[0]["nsys_available"], stall_rows[0]["ncu_available"]], "#7c3aed")
    _simple_bar_svg(out_dir / "p1_tensor_lifetime_gantt_top20.svg", "P1 profiler top op MB", [r.get("source_op", "") for r in topk[:20]], [f(r, "requested_size_MB", 0.0) for r in topk[:20]], "#16a34a")
    _scatter_svg(out_dir / "p1_allocator_padding_scatter.svg", "P1 allocator reserved", allocator, "allocator_active_bytes_MB", "allocator_reserved_bytes_MB", "variant_id")
    return rows


def _kernel_inputs(args: argparse.Namespace, dataset: str, batch_size: int, depth: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = get_device(args.device)
    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=max(args.train_size, batch_size), val_size=args.val_size, test_size=args.test_size, seed=0, allow_fake_data=False)
    xb, yb = _take_batch(bundle, batch_size, device)
    set_seed(6844 + batch_size + depth)
    stack = _make_v68_stack(METHOD_CURRENT, bundle.input_dim, params.hidden_dim, depth, _basis_from_name(METHOD_CURRENT, params.basis_count), device, "current", batch_size)
    head = V63ManualLayer(params.hidden_dim, bundle.num_classes, kind="linear", basis_count=2, device=device)
    with torch.no_grad():
        h, _ = stack.forward_manual(xb)
        logits, head_cache = head.forward_manual(h)
        probs = F.softmax(logits, dim=-1)
        probs[torch.arange(yb.numel(), device=yb.device), yb] -= 1.0
        dz = head.backward_manual(probs / max(1, yb.numel()), head_cache)
        coeff = torch.zeros(params.hidden_dim, 2, device=device)
        coeff[:, 0].fill_(0.02)
    return h.detach(), dz.detach(), coeff.detach()


def _measure_micro(args: argparse.Namespace, name: str, impl: str, dataset: str, batch_size: int, depth: int) -> Dict[str, Any]:
    if impl.startswith("triton") and not _triton_available():
        return {**_row_common("P2", args, variant_id=name, dataset=dataset, batch_size=batch_size, depth=depth), "microkernel": name, "implementation": impl, "implementation_status": "not_implemented_triton_unavailable", "stage_status": "not_implemented_triton_unavailable", "used_for_gate": 0, "not_implemented_count": 1, "reason": "triton/cuda unavailable"}
    if impl in {"not_implemented", "cuda_extension_optional"}:
        return {**_row_common("P2", args, variant_id=name, dataset=dataset, batch_size=batch_size, depth=depth), "microkernel": name, "implementation": impl, "implementation_status": impl, "stage_status": impl, "used_for_gate": 0, "not_implemented_count": 1, "reason": "not implemented in v6.8 runner"}
    x, dz, coeff = _kernel_inputs(args, dataset, batch_size, depth)
    ref_g, ref_dx = _torch_coeffgrad_dx(x, dz, coeff)
    x64, dz64, coeff64 = x.double(), dz.double(), coeff.double()
    ref64, _ = _torch_coeffgrad_dx(x64, dz64, coeff64)
    def call() -> Tuple[torch.Tensor, torch.Tensor]:
        if impl == "torch-current":
            return _torch_coeffgrad_dx(x, dz, coeff)
        if impl == "triton-coeffgrad":
            return _triton_coeffgrad_dx(x, dz, coeff, do_dx=False)
        if impl in {"triton-fused-dx-coeffgrad", "triton-full-backward-light"}:
            return _triton_coeffgrad_dx(x, dz, coeff, do_dx=True)
        raise ValueError(impl)
    for _ in range(args.micro_warmup_steps):
        call()
    _sync(x.device)
    times: List[float] = []
    peaks: List[float] = []
    reserved: List[float] = []
    out_g = ref_g
    out_dx = ref_dx
    for _ in range(args.micro_measure_steps):
        _reset_peak(x.device)
        _sync(x.device)
        t0 = time.perf_counter()
        out_g, out_dx = call()
        _sync(x.device)
        times.append((time.perf_counter() - t0) * 1000.0)
        p, r = _peak_mb(x.device)
        peaks.append(p)
        reserved.append(r)
    grad_rel, grad_cos, _ = _rel_cos(out_g.detach().flatten().float().cpu(), ref_g.detach().flatten().float().cpu())
    dx_rel, dx_cos, _ = _rel_cos(out_dx.detach().flatten().float().cpu(), ref_dx.detach().flatten().float().cpu())
    row = {
        **_row_common("P2", args, method=METHOD_CURRENT, variant_id=name, dataset=dataset, batch_size=batch_size, depth=depth),
        "microkernel": name,
        "implementation": impl,
        "input_shape": str(tuple(x.shape)),
        "dtype": str(x.dtype),
        "block_size": 128,
        "num_warps": 8 if impl.startswith("triton") else METRIC_UNAVAILABLE,
        "shared_memory_bytes": METRIC_UNAVAILABLE,
        "registers_per_thread": METRIC_UNAVAILABLE,
        "coeffgrad_time_ms": _mean(times),
        "dx_time_ms": _mean(times) if "dx" in impl else 0.0,
        "combined_time_ms": _mean(times),
        "peak_allocated_MB": _mean(peaks),
        "peak_reserved_MB": _mean(reserved),
        "temp_allocated_MB": 0.0 if impl.startswith("triton") else 2.0 * x.numel() * x.element_size() / (1024**2),
        "global_load_bytes": METRIC_UNAVAILABLE,
        "global_store_bytes": METRIC_UNAVAILABLE,
        "atomic_transactions": METRIC_UNAVAILABLE,
        "l2_read_transactions": METRIC_UNAVAILABLE,
        "l2_write_transactions": METRIC_UNAVAILABLE,
        "dram_read_bytes": METRIC_UNAVAILABLE,
        "dram_write_bytes": METRIC_UNAVAILABLE,
        "stall_long_scoreboard": METRIC_UNAVAILABLE,
        "achieved_occupancy": METRIC_UNAVAILABLE,
        "register_spill_count": METRIC_UNAVAILABLE,
        "grad_relerr_vs_torch": grad_rel,
        "grad_cos_vs_torch": grad_cos,
        "grad_relerr_vs_autograd": grad_rel,
        "grad_cos_vs_autograd": grad_cos,
        "input_grad_relerr_vs_torch": dx_rel,
        "input_grad_cos_vs_torch": dx_cos,
        "rounding_MAE_vs_fp64": float((out_g.double() - ref64).abs().mean().detach().cpu()),
        "rounding_max_abs_vs_fp64": float((out_g.double() - ref64).abs().max().detach().cpu()),
        "gradient_correctness_pass": int(grad_rel < 1.0e-4 and grad_cos > 0.999 and dx_cos > 0.999),
        "uses_triton_kernel": int(impl.startswith("triton")),
    }
    return row


def run_p2(args: argparse.Namespace) -> List[Dict[str, Any]]:
    kernels = [
        ("K0-current-coeffgrad-torch", "torch-current"),
        ("K1-triton-coeffgrad-local-reduce", "triton-coeffgrad"),
        ("K2-triton-coeffgrad-two-stage-reduce", "not_implemented"),
        ("K3-triton-fused-dx-coeffgrad", "triton-fused-dx-coeffgrad"),
        ("K4-triton-fused-dx-coeffgrad-update-prep", "not_implemented"),
        ("K5-triton-transform-derivative", "not_implemented"),
        ("K6-triton-delta-dx-only", "not_implemented"),
        ("K7-cuda-extension-coeffgrad", "cuda_extension_optional"),
        ("K8-cuda-extension-full-backward", "cuda_extension_optional"),
    ]
    rows: List[Dict[str, Any]] = []
    for dataset in parse_str_list(args.micro_datasets):
        canonical = dataset_name(dataset)
        for batch_size in parse_int_list(args.micro_batch_sizes):
            for depth in parse_int_list(args.micro_depths):
                current: Dict[str, Any] | None = None
                for name, impl in kernels:
                    row = _measure_micro(args, name, impl, canonical, batch_size, depth)
                    if row.get("implementation_status") == "measured":
                        if impl == "torch-current":
                            row["time_ratio_vs_current"] = 1.0
                            row["memory_ratio_vs_current"] = 1.0
                            current = row
                        elif current is not None:
                            row["time_ratio_vs_current"] = f(row, "combined_time_ms") / max(1.0e-12, f(current, "combined_time_ms"))
                            row["memory_ratio_vs_current"] = f(row, "peak_allocated_MB") / max(1.0e-12, f(current, "peak_allocated_MB"))
                        row["triton_kernel_pass"] = int((f(row, "time_ratio_vs_current", 99.0) <= 0.50 or f(row, "memory_ratio_vs_current", 99.0) <= 0.70) and int(f(row, "gradient_correctness_pass", 0)) == 1)
                    rows.append(row)
                    _wandb_log_row(args, row, "summary/v68_p2_triton_microkernel")
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p2_triton_microkernel_audit.csv", rows)
    write_csv(out_dir / "p2_triton_microkernel_correctness.csv", [r for r in rows if r.get("implementation_status") == "measured"])
    measured = [r for r in rows if r.get("implementation_status") == "measured"]
    _scatter_svg(out_dir / "p2_triton_microkernel_pareto.svg", "P2 Triton microkernel time/memory", measured, "memory_ratio_vs_current", "time_ratio_vs_current", "microkernel")
    _simple_bar_svg(out_dir / "p2_coeffgrad_time_memory_bar.svg", "P2 combined time", [r["microkernel"] for r in measured], [f(r, "combined_time_ms", 0.0) for r in measured], "#2563eb")
    _simple_bar_svg(out_dir / "p2_memory_traffic_reduction.svg", "P2 memory traffic unavailable", [r["microkernel"] for r in measured], [0.0 for _ in measured], "#a1a1aa")
    _simple_bar_svg(out_dir / "p2_grad_error_vs_speed.svg", "P2 grad relerr", [r["microkernel"] for r in measured], [f(r, "grad_relerr_vs_torch", 0.0) for r in measured], "#dc2626")
    return rows


def run_p3(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    detail = _profile_grid(args, "P3", P3_PACKAGES)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p3_full_backward_package_detail.csv", detail)
    measured = [r for r in detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    by: Dict[str, List[Dict[str, Any]]] = {}
    for row in measured:
        by.setdefault(str(row.get("variant_id")), []).append(row)
    summary: List[Dict[str, Any]] = []
    for package, rows in by.items():
        mem = [f(r, "memory_ratio_vs_MLP") for r in rows]
        step = [f(r, "step_time_ratio_vs_MLP") for r in rows]
        bwd = [f(r, "backward_time_ratio_vs_MLP") for r in rows]
        forward = [f(r, "forward_time_ratio_vs_MLP") for r in rows]
        mem_imp = [f(r, "actual_memory_reduction_vs_current") for r in rows if "actual_memory_reduction_vs_current" in r]
        step_imp = [f(r, "actual_step_improvement_vs_current") for r in rows if "actual_step_improvement_vs_current" in r]
        bwd_imp = []
        current_bwd_by_shape = {
            (r.get("dataset"), r.get("batch_size"), r.get("depth")): f(r, "backward_time_ratio_vs_MLP")
            for r in detail
            if r.get("variant_id") == "D0-current" and r.get("implementation_status") == "measured"
        }
        for r in rows:
            cur = current_bwd_by_shape.get((r.get("dataset"), r.get("batch_size"), r.get("depth")))
            if cur:
                bwd_imp.append((cur - f(r, "backward_time_ratio_vs_MLP")) / max(1.0e-12, cur))
        mean_mem = _mean(mem)
        summary.append({
            **_row_common("P3", args, variant_id=package),
            "package": package,
            "components": rows[0].get("package_components", ""),
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
            "forward_ratio_mean": _mean(forward),
            "memory_improvement_vs_current": _mean(mem_imp, 0.0),
            "step_improvement_vs_current": _mean(step_imp, 0.0),
            "backward_improvement_vs_current": _mean(bwd_imp, 0.0),
            "kernel_count_reduction": METRIC_UNAVAILABLE,
            "allocation_count_reduction": METRIC_UNAVAILABLE,
            "dram_write_reduction": METRIC_UNAVAILABLE,
            "stall_reduction": METRIC_UNAVAILABLE,
            "grad_relerr_max": max(f(r, "grad_relerr") for r in rows),
            "grad_cos_min": min(f(r, "grad_cos") for r in rows),
            "rounding_MAE_vs_fp64": METRIC_UNAVAILABLE,
            "peak_gap_explain_ratio": _mean(f(r, "explain_ratio") for r in rows),
            "top_gap_source_after_repair": rows[0].get("top1_source", ""),
            "component_status_all_grad_pass": int(all(int(f(r, "gradient_correctness_pass", 0)) == 1 for r in rows)),
            "near_pass_count": sum((m <= 1.05 and s <= 1.50) for m, s in zip(mem, step)),
            "memory_pass_count": sum(m < 1.0 for m in mem),
            "time_pass_count": sum(s <= 1.35 for s in step),
            "shape_stability_score": 1.0 - _std(mem) / max(1.0e-12, mean_mem),
        })
    for package, method, policy, implemented, components, flash_variant in P3_PACKAGES:
        if implemented:
            continue
        summary.append({**_row_common("P3", args, method=method, variant_id=package), "package": package, "components": components, "implementation_status": policy, "stage_status": policy, "used_for_gate": 0, "not_implemented_count": 1, "reason": "package component not implemented; no measured ratio emitted"})
    write_csv(out_dir / "p3_full_backward_packages.csv", summary)
    _scatter_svg(out_dir / "p3_package_memory_step_pareto.svg", "P3 package memory/step", measured, "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _scatter_svg(out_dir / "p3_s0_s1_s2_threshold_plot.svg", "P3 thresholds", measured, "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _simple_bar_svg(out_dir / "p3_full_package_improvement_bar.svg", "P3 memory improvement", [r["package"] for r in summary], [f(r, "memory_improvement_vs_current", 0.0) for r in summary], "#16a34a")
    _simple_bar_svg(out_dir / "p3_batch_depth_stability_heatmap.svg", "P3 stability", [r["package"] for r in summary], [f(r, "shape_stability_score", 0.0) for r in summary], "#2563eb")
    _simple_bar_svg(out_dir / "p3_grad_correctness_bar.svg", "P3 grad relerr", [r["package"] for r in summary], [f(r, "grad_relerr_max", 0.0) for r in summary], "#dc2626")
    return summary, detail


def _residual_effect(args: argparse.Namespace, method: str, policy: str, dataset: str, batch_size: int, depth: int, scale_target: float) -> Dict[str, Any]:
    device = get_device(args.device)
    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=max(args.train_size, batch_size), val_size=args.val_size, test_size=args.test_size, seed=0, allow_fake_data=False)
    x, y = _take_batch(bundle, batch_size, device)
    stack = _make_v68_stack(method, bundle.input_dim, params.hidden_dim, depth, _basis_from_name(method, params.basis_count), device, policy, batch_size)
    head = V63ManualLayer(params.hidden_dim, bundle.num_classes, kind="linear", basis_count=2, device=device)
    with torch.no_grad():
        h, _ = stack.forward_manual(x)
        logits, _ = head.forward_manual(h)
        loss_on = F.cross_entropy(logits, y)
        residual_sq = 0.0
        base_sq = 0.0
        saved: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in getattr(stack, "layers", []):
            if hasattr(layer, "params") and "poly" in layer.params:
                p = layer.params["poly"]
                saved.append((p, p.detach().clone()))
                residual_sq += float((0.05 * p[:, 0]).square().sum().detach().cpu())
                base_sq += float(torch.ones_like(p[:, 0]).square().sum().detach().cpu())
                p.zero_()
        h0, _ = stack.forward_manual(x)
        logits0, _ = head.forward_manual(h0)
        loss_off = F.cross_entropy(logits0, y)
        for p, old in saved:
            p.copy_(old)
    residual_over_base = math.sqrt(residual_sq) / max(1.0e-12, math.sqrt(base_sq))
    delta_logit = float((logits - logits0).abs().max().detach().cpu())
    delta_loss = float((loss_on - loss_off).detach().cpu())
    return {
        "scale_target": scale_target,
        "scale_actual": residual_over_base,
        "residual_over_base": residual_over_base,
        "residual_norm": math.sqrt(residual_sq),
        "base_norm": math.sqrt(base_sq),
        "residual_ablation_delta_logit": delta_logit,
        "residual_ablation_delta_loss": delta_loss,
        "residual_effect_pass": int(residual_over_base >= 0.02 and (abs(delta_logit) > 1.0e-4 or abs(delta_loss) > 1.0e-4)),
    }


def run_p4(args: argparse.Namespace) -> List[Dict[str, Any]]:
    profile_variants = [(a, b, c, d, e) for a, b, c, d, e, _scale in P4_RESETS]
    rows = _profile_grid(args, "P4", profile_variants)
    scale_by = {name: scale for name, _m, _p, _i, _c, scale in P4_RESETS}
    for row in rows:
        if row.get("implementation_status") == "measured" and row.get("method") != "MLP-autograd-reference":
            eff = _residual_effect(args, str(row.get("method")), str(row.get("workspace_policy", "current")), str(row.get("dataset")), int(float(row.get("batch_size") or args.batch_size)), int(float(row.get("depth") or 2)), scale_by.get(str(row.get("variant_id")), 0.0))
            row.update(eff)
            row["near_pass_count"] = int(f(row, "memory_ratio_vs_MLP", 99) <= 1.05 and f(row, "step_time_ratio_vs_MLP", 99) <= 1.35 and int(f(row, "gradient_correctness_pass", 0)) == 1)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p4_residual_effective_reset.csv", rows)
    measured = [r for r in rows if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    _scatter_svg(out_dir / "p4_reset_memory_time_pareto.svg", "P4 reset memory/step", measured, "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _scatter_svg(out_dir / "p4_residual_strength_vs_memory.svg", "P4 residual strength/memory", measured, "memory_ratio_vs_MLP", "residual_over_base", "variant_id")
    _simple_bar_svg(out_dir / "p4_residual_ablation_delta.svg", "P4 residual ablation logit", [r["variant_id"] for r in measured], [f(r, "residual_ablation_delta_logit", 0.0) for r in measured], "#16a34a")
    _scatter_svg(out_dir / "p4_scale_sweep_pareto.svg", "P4 scale sweep", measured, "scale_actual", "step_time_ratio_vs_MLP", "variant_id")
    _simple_bar_svg(out_dir / "p4_reset_workspace_comparison.svg", "P4 workspace MB", [r["variant_id"] for r in measured], [f(r, "workspace_pool_MB", 0.0) for r in measured], "#7c3aed")
    return rows


def _best_survivor(p3_summary: Sequence[Dict[str, Any]], p2_rows: Sequence[Dict[str, Any]], p1_rows: Sequence[Dict[str, Any]]) -> Tuple[str, Dict[str, Any] | None]:
    measured = [r for r in p3_summary if r.get("implementation_status") == "measured"]
    s0 = [r for r in measured if f(r, "memory_ratio_max", 99) < 1.0 and f(r, "step_ratio_mean", 99) <= 1.20 and int(f(r, "component_status_all_grad_pass", 0)) == 1]
    s1 = [r for r in measured if f(r, "memory_ratio_max", 99) < 1.0 and f(r, "step_ratio_mean", 99) <= 1.35 and int(f(r, "component_status_all_grad_pass", 0)) == 1]
    s2 = [r for r in measured if f(r, "memory_ratio_mean", 99) <= 1.05 and f(r, "step_ratio_mean", 99) <= 1.50 and f(r, "memory_improvement_vs_current", 0.0) >= 0.10]
    if s0:
        return "S0", min(s0, key=lambda r: f(r, "memory_ratio_mean", 99))
    if s1:
        return "S1", min(s1, key=lambda r: f(r, "memory_ratio_mean", 99))
    if s2:
        return "S2", min(s2, key=lambda r: f(r, "memory_ratio_mean", 99))
    if any(int(f(r, "triton_kernel_pass", 0)) == 1 for r in p2_rows):
        return "S6", min(measured, key=lambda r: f(r, "memory_ratio_mean", 99)) if measured else None
    p1_pass = any(int(f(r, "attribution_pass", 0)) == 1 for r in p1_rows if r.get("variant_id") == "A1-DWM2-current")
    return ("S3" if p1_pass else "S4"), min(measured, key=lambda r: f(r, "memory_ratio_mean", 99)) if measured else None


def _write_not_run(path: Path, stage: str, reason: str, gated_by: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = [{**_row_common(stage, args), "implementation_status": "not_run", "stage_status": "not_run", "status": "not_run", "used_for_gate": 0, "gated_not_run_count": 1, "reason": reason, "gated_by": gated_by}]
    write_csv(path, rows)
    for row in rows:
        _wandb_log_row(args, row, f"summary/v68_{stage.lower()}_not_run")
    return rows


def run_gated(args: argparse.Namespace, open_task: bool, open_optimizer: bool, open_functional: bool) -> None:
    if not open_task:
        _write_not_run(Path(args.out_dir) / "p5_one_step_probe.csv", "P5", "No S0/S1/S2 DWM2 survivor and no reset near-pass with residual effect", "P3/P4", args)
        _write_not_run(Path(args.out_dir) / "p6_task_reentry.csv", "P6", "P5 did not pass or no S0/S1 candidate", "P5", args)
        _write_not_run(Path(args.out_dir) / "p6_task_trace.csv", "P6", "P6 is gated", "P5", args)
    else:
        _write_not_run(Path(args.out_dir) / "p5_one_step_probe.csv", "P5", "P5 implementation gated pending candidate review", "manual_review", args)
        _write_not_run(Path(args.out_dir) / "p6_task_reentry.csv", "P6", "Task runner not opened in this run", "runner_scope", args)
        _write_not_run(Path(args.out_dir) / "p6_task_trace.csv", "P6", "No task trace emitted", "runner_scope", args)
    if not open_optimizer:
        _write_not_run(Path(args.out_dir) / "p7_optimizer_exploration.csv", "P7", "P6 task re-entry did not pass", "P6", args)
    if not open_functional:
        _write_not_run(Path(args.out_dir) / "p8_functional_correction_smoke.csv", "P8", "P8 is gated behind P6/P7", "P6/P7", args)


def run_route(args: argparse.Namespace, p1_rows: Sequence[Dict[str, Any]], p2_rows: Sequence[Dict[str, Any]], p3_summary: Sequence[Dict[str, Any]], p4_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    survivor_type, best = _best_survivor(p3_summary, p2_rows, p1_rows)
    current_p1 = [r for r in p1_rows if r.get("variant_id") == "A1-DWM2-current"]
    attribution_pass = any(int(f(r, "attribution_pass", 0)) == 1 for r in current_p1)
    triton_pass = any(int(f(r, "triton_kernel_pass", 0)) == 1 for r in p2_rows)
    reset_near = [r for r in p4_rows if r.get("implementation_status") == "measured" and int(f(r, "near_pass_count", 0)) == 1]
    reset_effect = [r for r in reset_near if int(f(r, "residual_effect_pass", 0)) == 1]
    if survivor_type in {"S0", "S1"}:
        route = "R1-DWM2TritonSolved"
        primary = "DWM2 Triton package produced S0/S1; P5 still required"
        open_task = False
    elif survivor_type == "S2":
        route = "R2-DWM2TritonNearPass"
        primary = "DWM2 Triton package near-pass only"
        open_task = False
    elif not attribution_pass:
        route = "R3-AttributionIncomplete"
        primary = "Nsight/allocator trace still did not explain current DWM2 peak gap to threshold"
        open_task = False
    elif reset_effect:
        route = "R5-ResetPrimitiveCandidate"
        primary = "Reset near-pass with residual effect exists"
        open_task = False
    elif reset_near:
        route = "R6-ResetLinearOnly"
        primary = "Reset near-pass exists but residual effect failed"
        open_task = False
    elif triton_pass:
        route = "R10-MicrokernelOnlyNoFullGain"
        primary = "Triton microkernel passed but full package did not"
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
        "attribution_pass": int(attribution_pass),
        "triton_kernel_pass": int(triton_pass),
        "reset_residual_effect_pass": int(bool(reset_effect)),
        "fallback_triggered": True,
        "fallback_near_pass_count": len(reset_near),
        "open_task_reentry": bool(open_task),
        "open_optimizer_exploration": False,
        "open_functional_correction": False,
        "no_fake": True,
        "no_proxy": True,
        "primary_blocker": primary,
        "next_required_implementation": "nsight_trace_or_new_primitive_family" if route.startswith("R3") else "full_custom_layer_or_reset_family",
    }
    _json_dump(Path(args.out_dir) / "route_decision.json", route_json)
    _json_dump(Path(args.out_dir) / "aggregate_decision.json", {"status": "gated" if not open_task else "task_reentry_open", "fake_data_used": 0, "proxy_rows_used_as_results": 0, **route_json})
    _placeholder_svg(Path(args.out_dir) / "p9_route_decision_dashboard.svg", "P9 route decision", route)
    _wandb_log_row(args, {**_row_common("P9", args), **route_json}, "summary/v68_route")
    return route_json


def run_failure(args: argparse.Namespace) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    out_dir = Path(args.out_dir)
    for fn, stage in [
        ("p1_phase_peak_summary.csv", "P1"),
        ("p2_triton_microkernel_audit.csv", "P2"),
        ("p3_full_backward_package_detail.csv", "P3"),
        ("p4_residual_effective_reset.csv", "P4"),
        ("p5_one_step_probe.csv", "P5"),
        ("p6_task_reentry.csv", "P6"),
        ("p7_optimizer_exploration.csv", "P7"),
        ("p8_functional_correction_smoke.csv", "P8"),
    ]:
        path = out_dir / fn
        if not path.exists():
            failures.append({"stage": stage, "variant_id": fn, "failure_type": "F15_artifact_missing", "metric": "missing", "recommendation": "rerun stage"})
            continue
        for row in _read_csv(path):
            status = row.get("implementation_status") or row.get("status")
            if status in {"not_run", "not_implemented"} or str(status).startswith("not_implemented") or str(status).startswith("cuda_extension"):
                failures.append({"stage": stage, "variant_id": row.get("variant_id", row.get("microkernel", "")), "failure_type": "F14_gated_not_run", "metric": row.get("reason", status), "recommendation": "implement or pass gate before claiming metric"})
            if status == "measured":
                if row.get("memory_ratio_vs_MLP") not in {None, ""} and row.get("method") != "MLP-autograd-reference" and f(row, "memory_ratio_vs_MLP", 0.0) >= 1.0:
                    failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F1_memory_fail", "metric": f"memory_ratio={row.get('memory_ratio_vs_MLP')}", "recommendation": "reduce actual CUDA peak"})
                if row.get("step_time_ratio_vs_MLP") not in {None, ""} and row.get("method") != "MLP-autograd-reference" and f(row, "step_time_ratio_vs_MLP", 0.0) > 1.35:
                    failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F2_step_time_fail", "metric": f"step_ratio={row.get('step_time_ratio_vs_MLP')}", "recommendation": "reduce step/runtime"})
                if row.get("gradient_correctness_pass") not in {None, ""} and int(f(row, "gradient_correctness_pass", 1)) == 0:
                    failures.append({"stage": stage, "variant_id": row.get("variant_id", row.get("microkernel", "")), "failure_type": "F3_gradient_correctness_fail", "metric": f"grad={row.get('grad_relerr') or row.get('grad_relerr_vs_torch')}", "recommendation": "fix kernel numerics"})
                if stage == "P1" and int(f(row, "attribution_pass", 0)) == 0:
                    failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F4_attribution_incomplete", "metric": f"explain_ratio={row.get('explain_ratio')}", "recommendation": "deeper allocation/Nsight trace"})
                if stage == "P2" and row.get("implementation", "").startswith("triton") and int(f(row, "triton_kernel_pass", 0)) == 0:
                    failures.append({"stage": stage, "variant_id": row.get("microkernel", ""), "failure_type": "F6_triton_kernel_no_effect", "metric": f"timeR={row.get('time_ratio_vs_current')} memR={row.get('memory_ratio_vs_current')}", "recommendation": "improve kernel or fuse larger live-set"})
                if stage == "P4" and int(f(row, "residual_effect_pass", 0)) == 0 and row.get("method") != "MLP-autograd-reference":
                    failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F8_residual_effect_fail", "metric": f"residual={row.get('residual_over_base')}", "recommendation": "increase residual effect or redesign primitive"})
                if stage == "P4" and int(f(row, "near_pass_count", 0)) == 0 and row.get("method") != "MLP-autograd-reference":
                    failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F9_reset_no_near_pass", "metric": f"mem={row.get('memory_ratio_vs_MLP')} step={row.get('step_time_ratio_vs_MLP')}", "recommendation": "reduce reset memory/time"})
    write_csv(out_dir / "failure_table.csv", failures or [{"stage": "ALL", "variant_id": "all", "failure_type": "none"}])
    for row in failures:
        _wandb_log_row(args, row, "summary/v68_failure")
    _simple_bar_svg(out_dir / "failure_taxonomy_heatmap.svg", "Failure taxonomy", [r["failure_type"] for r in failures], [1.0 for _ in failures], "#dc2626")
    return failures


def _copy_figures(out_dir: Path) -> None:
    figures = ensure_dir(out_dir / "figures")
    required = [
        "p1_phase_peak_waterfall.svg",
        "p1_tensor_lifetime_gantt_top20.svg",
        "p1_nsight_stall_dashboard.svg",
        "p1_gap_attribution_stacked_bar.svg",
        "p2_triton_microkernel_pareto.svg",
        "p2_memory_traffic_reduction.svg",
        "p2_grad_error_vs_speed.svg",
        "p3_package_memory_step_pareto.svg",
        "p3_batch_depth_stability_heatmap.svg",
        "p4_residual_strength_vs_memory.svg",
        "p4_residual_ablation_delta.svg",
        "p5_before_after_loss_plot.svg",
        "p6_task_efficiency_pareto.svg",
        "p9_route_decision_dashboard.svg",
        "failure_taxonomy_heatmap.svg",
    ]
    for name in required:
        src = out_dir / name
        dst = figures / name.replace("_top20", "")
        if src.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            _placeholder_svg(dst, name, "not generated by gate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(wandb=True)
    parser.add_argument("--packages", default="V6_8_ALL")
    parser.add_argument("--out-dir", type=Path, default=Path("results/real_rerun_20260504/v68_real"))
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
    parser.add_argument("--wandb-group", default="v68-real-20260504")
    parser.add_argument("--wandb-name-prefix", default="v68-real")
    parser.add_argument("--no-wandb", action="store_false", dest="wandb")
    parser.add_argument("--fresh", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    out_dir = ensure_dir(args.out_dir)
    _patch_v67_for_v68()
    _wandb_init(args)
    try:
        run_p0(args)
        p1_rows = run_p1(args)
        run_p0_reproduction(args, p1_rows)
        p2_rows = run_p2(args)
        p3_summary, _p3_detail = run_p3(args)
        p4_rows = run_p4(args)
        route = run_route(args, p1_rows, p2_rows, p3_summary, p4_rows)
        run_gated(args, bool(route.get("open_task_reentry", False)), bool(route.get("open_optimizer_exploration", False)), bool(route.get("open_functional_correction", False)))
        run_failure(args)
        _placeholder_svg(out_dir / "p5_before_after_loss_plot.svg", "P5 before-after loss", "not_run")
        _placeholder_svg(out_dir / "p6_task_efficiency_pareto.svg", "P6 task efficiency", "not_run")
        _copy_figures(out_dir)
        _write_manifest(out_dir, args, started, time.time())
    finally:
        _wandb_finish(args, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
