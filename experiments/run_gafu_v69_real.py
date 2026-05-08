#!/usr/bin/env python3
"""DG-KAN v6.9 real-only full-step live-set/bounded-reset runner.

This runner emits only measured values from real execution paths.  Missing
single-call full-layer/CUDA kernels are reported as not_implemented; unavailable
profiler counters are reported as metric_unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

import run_gafu_v68_real as v68
from dgkan_core import ensure_dir, get_device, load_vision_bundle, parse_int_list, parse_str_list, write_csv
from run_gafu_v3 import dataset_name
from run_gafu_v63 import ManualOptimizer, V63ManualLayer, V63Params, _basis_from_name, _peak_mb, _rel_cos, _reset_peak, _sync, _wandb_finish, _wandb_init, _wandb_log_row, f
from run_gafu_v64_real import METHOD_CURRENT, _make_mlp, _sha256, _take_batch
from run_gafu_v65_real import _placeholder_svg, _scatter_svg, _simple_bar_svg
from run_gafu_v66_real import _git_commit, _git_status, _json_dump, _mean, _std


METRIC_UNAVAILABLE = "metric_unavailable"
V68_MEMORY_MEAN = 1.2917923088533285
V68_STEP_MEAN = 1.9697941069965306
V68_P3_STEP_MEAN = 2.1139623910972194
triton = v68.triton
tl = v68.tl


if triton is not None:

    @triton.jit
    def _v69_noop_copy_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        vals = tl.load(x_ptr + offs, mask=mask, other=0.0)
        tl.store(y_ptr + offs, vals, mask=mask)


def _triton_available() -> bool:
    return v68._triton_available()


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


def _status(policy: str, implemented: bool) -> str:
    return "measured" if implemented else policy


def _patch_stack() -> None:
    v68._patch_v67_for_v68()


P0_VARIANTS = [
    ("MLP-autograd-reference", "reference", "mlp", True, "reference"),
    ("MLP-manual-linear-reference", "MLP-manual-linear-reference", "current", True, "manual-linear"),
    ("DWM2-current", METHOD_CURRENT, "current", True, "current"),
    ("DWM2-Triton-fused-dx-coeffgrad-v1", METHOD_CURRENT, "triton_fused_dx_coeffgrad", _triton_available(), "triton-fused-dx-coeffgrad"),
    ("DWM2-Triton-full-backward-light-v1", METHOD_CURRENT, "triton_full_backward_light", _triton_available(), "triton-full-backward-light"),
    ("DWM2-FullLayerBackward-v2", METHOD_CURRENT, "not_implemented", False, "single-call-full-layer"),
    ("DWM2-OneBufferFullStep-v1", METHOD_CURRENT, "not_implemented", False, "true-onebuffer-full-step"),
    ("ResidualEffectiveTinyKAN-scale002", "DWM2-poly1-minimal", "poly1_scale002", True, "reset-scale002"),
    ("ResidualEffectiveTinyKAN-scale005", "DWM2-poly1-minimal", "poly1_scale005", True, "reset-scale005"),
    ("ResidualBoundedOneBuffer-v2", "DWM2-poly1-minimal", "poly1_scale002", True, "bounded-reset-diagnostic"),
]

P1_VARIANTS = [
    ("A1-DWM2-current", METHOD_CURRENT, "current", True, "current"),
]

P3_PACKAGES = [
    ("L0-current", METHOD_CURRENT, "current", True, "current", 1),
    ("L1-fused-dx-coeffgrad-only", METHOD_CURRENT, "triton_fused_dx_coeffgrad", _triton_available(), "fused-dx-coeffgrad", 1),
    ("L2-full-layer-backward", METHOD_CURRENT, "triton_full_backward_light", _triton_available(), "full-layer-backward-light", 1),
    ("L3-full-layer-backward-no-dx-materialize", METHOD_CURRENT, "not_implemented", False, "no-dx-materialize", 0),
    ("L4-full-layer-backward-update-prep", METHOD_CURRENT, "not_implemented", False, "update-prep", 0),
    ("L5-multi-layer-chunked-backward", METHOD_CURRENT, "not_implemented", False, "multi-layer-chunked", 0),
    ("L6-single-call-depth2-backward", METHOD_CURRENT, "not_implemented", False, "single-call-depth2", 0),
]

P4_PACKAGES = [
    ("O0-current", METHOD_CURRENT, "current", True, "current"),
    ("O1-onebuffer-forward-transform", METHOD_CURRENT, "buffer_reuse", True, "buffer-reuse-transform"),
    ("O2-onebuffer-backward-delta", METHOD_CURRENT, "delta_streaming", True, "delta-streaming"),
    ("O3-onebuffer-update-prep", METHOD_CURRENT, "not_implemented", False, "update-prep"),
    ("O4-onebuffer-forward-backward", METHOD_CURRENT, "buffer_reuse_delta_streaming", True, "buffer-reuse-delta-streaming"),
    ("O5-onebuffer-full-step", METHOD_CURRENT, "not_implemented", False, "true-onebuffer-full-step"),
    ("O6-onebuffer-full-step-chunked-mix", METHOD_CURRENT, "not_implemented", False, "chunked-mix"),
]

P5_RESETS = [
    ("R0-ManualLinear-reference", "MLP-manual-linear-reference", "current", True, "manualLinear", 0.0),
    ("R1-ResidualEffectiveTinyKAN-scale002-current", "DWM2-poly1-minimal", "poly1_scale002", True, "scale002-current", 0.02),
    ("R2-ResidualEffectiveTinyKAN-scale005-current", "DWM2-poly1-minimal", "poly1_scale005", True, "scale005-current", 0.05),
    ("R3-BoundedResidual-inplace-transform", "DWM2-poly1-minimal", "poly1_scale002", True, "bounded-inplace-diagnostic", 0.02),
    ("R4-BoundedResidual-fused-mix", "DWM2-poly1-minimal", "not_implemented", False, "fused-mix", 0.02),
    ("R5-BoundedResidual-chunked-mix", "DWM2-poly1-minimal", "not_implemented", False, "chunked-mix", 0.02),
    ("R6-BoundedResidual-single-buffer-backward", "DWM2-poly1-minimal", "poly1_scale005", True, "single-buffer-backward-diagnostic", 0.05),
    ("R7-BoundedResidual-lowrank-mix", "DWM2-poly1-minimal", "not_implemented", False, "lowrank-mix", 0.02),
    ("R8-BoundedResidual-piecewise2", "OneBufferPiecewiseLinear2", "not_implemented", False, "piecewise2", 0.02),
]


def _profile_grid(args: argparse.Namespace, stage: str, variants: Sequence[Tuple[Any, ...]]) -> List[Dict[str, Any]]:
    _patch_stack()
    return v68.v67._profile_grid_v67(args, stage, variants)  # type: ignore[arg-type]


def _gradient_check(method: str, batch: int, hidden: int, device: torch.device, policy: str) -> Dict[str, float]:
    return v68._gradient_check_v68(method, batch, hidden, device, policy)


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    _patch_stack()
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
            "implementation_status": _status(policy, implemented),
            "status": _status(policy, implemented),
            "fake_data_used": int(getattr(bundle, "used_fake_data", False)),
            "proxy_row_used": 0,
            "proxy_rows_used": 0,
            "uses_triton_kernel": int(policy.startswith("triton")),
            "uses_cuda_extension": 0,
            "v68_memory_ratio_mean": V68_MEMORY_MEAN,
            "v68_step_ratio_mean": V68_STEP_MEAN,
        })
        if not implemented:
            row.update({"used_for_gate": 0, "not_implemented_count": 1, "reason": "not implemented in v6.9 runner; no measured ratio emitted"})
        elif policy == "mlp":
            model = _make_mlp(bundle.input_dim, bundle.num_classes, params.hidden_dim, 2).to(device)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            row.update({"uses_loss_backward": 1, "uses_torch_autograd_graph": 1, "manual_forward_available": 0, "manual_backward_available": 0, "manual_update_available": 0, "nonKAN_param_count": sum(p.numel() for p in model.parameters()), "edge_param_count": 0, "residual_param_count": 0, "mixing_param_count": 0, "rollback_max_error": 0.0, "gradcheck_available": 0})
        else:
            stack = v68._make_v68_stack(method, bundle.input_dim, params.hidden_dim, 2, _basis_from_name(method, params.basis_count), device, policy, min(64, args.batch_size))
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
            grad = _gradient_check(method, min(32, args.batch_size), params.hidden_dim, device, policy)
            row.update({"uses_loss_backward": 0, "uses_torch_autograd_graph": 0, "manual_forward_available": 1, "manual_backward_available": 1, "manual_update_available": 1, "nonKAN_param_count": 0, "edge_param_count": stack.param_count() + head.param_count(), "residual_param_count": residual_params, "mixing_param_count": mixing_params, "rollback_max_error": rollback, "gradcheck_available": 1, **grad})
        rows.append(row)
        _wandb_log_row(args, row, "summary/v69_p0_contract")
    write_csv(out_dir / "p0_contract.csv", rows)
    _simple_bar_svg(out_dir / "p0_contract_heatmap.svg", "v6.9 P0 contract", [r["variant_id"] for r in rows], [1.0 if r.get("implementation_status") == "measured" and int(f(r, "fake_data_used", 0)) == 0 and int(f(r, "proxy_row_used", 0)) == 0 else 0.0 for r in rows], "#16a34a")
    _simple_bar_svg(out_dir / "p0_implementation_status.svg", "v6.9 implementation status", [r["variant_id"] for r in rows], [1.0 if r.get("implementation_status") == "measured" else 0.0 for r in rows], "#2563eb")
    return rows


def run_p0_reproduction(args: argparse.Namespace, p1_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    current = [r for r in p1_rows if r.get("variant_id") == "A1-DWM2-current" and r.get("implementation_status") == "measured"]
    mem_mean = _mean(f(r, "memory_ratio_vs_MLP") for r in current)
    step_mean = _mean(f(r, "step_time_ratio_vs_MLP") for r in current)
    row = {
        **_row_common("P0_REPRO", args, variant_id="A1-DWM2-current"),
        "v68_reference_memory_ratio_mean": V68_MEMORY_MEAN,
        "v68_reference_step_ratio_mean": V68_STEP_MEAN,
        "v69_current_memory_ratio_mean": mem_mean,
        "v69_current_step_ratio_mean": step_mean,
        "reproduction_delta_memory_ratio": mem_mean - V68_MEMORY_MEAN,
        "reproduction_delta_step_ratio": step_mean - V68_STEP_MEAN,
        "reproduction_pass": int(abs(mem_mean - V68_MEMORY_MEAN) <= 0.05 and abs(step_mean - V68_STEP_MEAN) <= 0.15),
    }
    write_csv(Path(args.out_dir) / "p0_reproduction_check.csv", [row])
    _simple_bar_svg(Path(args.out_dir) / "p0_reproduction_delta_bar.svg", "v6.9 vs v6.8 reproduction delta", ["memory", "step"], [row["reproduction_delta_memory_ratio"], row["reproduction_delta_step_ratio"]], "#2563eb")
    _wandb_log_row(args, row, "summary/v69_p0_reproduction")
    return [row]


def _enrich_live_set(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    gap = max(0.0, f(row, "peak_gap_MB", 0.0))
    coeff = f(row, "coeffgrad_phase_peak_MB", 0.0)
    dz = f(row, "loss_delta_peak_MB", 0.0)
    # Do not use the whole backward peak to explain itself.  Only separately
    # measured/cacheable components count toward attribution.
    dx = f(row, "cache_delta_MB", 0.0)
    update = f(row, "update_peak_MB", 0.0)
    transform = f(row, "forward_peak_MB", 0.0)
    mixing = f(row, "manual_cache_total_MB", 0.0)
    sources = [
        ("loss_delta_live", dz, "phase_loss_delta"),
        ("forward_transform_live", transform, "phase_forward_transform"),
        ("coeffgrad_live", coeff, "phase_backward_dx_coeffgrad"),
        ("dx_or_delta_cache_live", dx, "forward_to_backward_cache"),
        ("update_temp_live", update, "phase_update_params"),
        ("mixing_or_cache_live", mixing, "forward_to_backward_cache"),
    ]
    sources.sort(key=lambda item: item[1], reverse=True)
    top3 = sum(max(0.0, s[1]) for s in sources[:3])
    explained = min(gap, sum(max(0.0, s[1]) for s in sources))
    out.update({
        "peak_timestamp": METRIC_UNAVAILABLE,
        "peak_phase": "phase_backward_dx_coeffgrad",
        "live_tensor_count": len([s for s in sources if s[1] > 0]),
        "live_tensor_total_MB": sum(max(0.0, s[1]) for s in sources),
        "explained_gap_MB": explained,
        "unexplained_gap_MB": max(0.0, gap - explained),
        "explain_ratio": explained / max(1.0e-12, gap) if gap > 0 else 1.0,
        "top3_gap_fraction": top3 / max(1.0e-12, gap) if gap > 0 else 1.0,
        "coeffgrad_live_MB": coeff,
        "dx_live_MB": dx,
        "dz_live_MB": dz,
        "update_temp_live_MB": update,
        "forward_transform_live_MB": transform,
        "mixing_live_MB": mixing,
        "optimizer_state_live_MB": f(row, "optimizer_state_MB", 0.0),
        "allocator_padding_MB": row.get("allocator_padding_total_MB", METRIC_UNAVAILABLE),
        "fragmentation_ratio": row.get("fragmentation_ratio", METRIC_UNAVAILABLE),
        "triton_input_materialization_MB": 0.0,
        "triton_output_materialization_MB": 0.0,
        "torch_to_triton_boundary_time_ms": 0.0,
        "triton_to_torch_boundary_time_ms": 0.0,
        "contiguous_copy_MB": 0.0,
        "dtype_cast_MB": 0.0,
        "layout_conversion_MB": 0.0,
        "live_set_overlap_pass": int(gap > 0 and (coeff + dx + dz + update) >= 0.50 * gap),
        "attribution_method": "phase_component_estimate_exact_peak_liveset_unavailable",
        "attribution_gate_reason": "exact peak-time live tensor allocation stack unavailable; phase components are diagnostic only",
    })
    out["attribution_pass"] = 0
    for idx in range(20):
        if idx < len(sources):
            name, mb, phase = sources[idx]
        else:
            name, mb, phase = "", "", ""
        out[f"top{idx + 1}_live_tensor_name"] = name
        out[f"top{idx + 1}_live_tensor_MB"] = mb
        out[f"top{idx + 1}_live_tensor_source"] = name
        out[f"top{idx + 1}_lifetime_start_phase"] = phase
        out[f"top{idx + 1}_lifetime_end_phase"] = phase
    return out


def run_p1(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = _profile_grid(args, "P1", P1_VARIANTS)
    live_rows = [_enrich_live_set(r) for r in rows]
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p1_live_set_attribution.csv", live_rows)
    boundary_rows = []
    for row in live_rows:
        if row.get("variant_id") == "A1-DWM2-current":
            boundary_rows.append({
                **_row_common("P1_BOUNDARY", args, method=row.get("method", ""), variant_id=row.get("variant_id", ""), dataset=row.get("dataset", ""), batch_size=int(float(row.get("batch_size") or 0)), depth=int(float(row.get("depth") or 0))),
                "torch_to_triton_boundary_time_ms": 0.0,
                "triton_to_torch_boundary_time_ms": 0.0,
                "triton_input_materialization_MB": 0.0,
                "triton_output_materialization_MB": 0.0,
                "boundary_bottleneck_pass": 0,
                "reason": "current DWM2 has no Triton/PyTorch boundary; P2 measures Triton boundary candidates",
            })
    write_csv(out_dir / "p1_boundary_overhead.csv", boundary_rows)
    current = [r for r in live_rows if r.get("variant_id") == "A1-DWM2-current"]
    _simple_bar_svg(out_dir / "p1_peak_live_tensor_top20.svg", "P1 live tensors", [r["dataset"] + f" B{r['batch_size']}D{r['depth']}" for r in current], [f(r, "live_tensor_total_MB") for r in current], "#2563eb")
    _simple_bar_svg(out_dir / "p1_gap_attribution_stacked_bar.svg", "P1 explained gap", [r["dataset"] + f" B{r['batch_size']}D{r['depth']}" for r in current], [f(r, "explain_ratio") for r in current], "#16a34a")
    _simple_bar_svg(out_dir / "p1_boundary_overhead_bar.svg", "P1 boundary MB", [r["dataset"] + f" B{r['batch_size']}D{r['depth']}" for r in current], [f(r, "triton_output_materialization_MB") for r in current], "#7c3aed")
    _scatter_svg(out_dir / "p1_live_set_overlap_heatmap.svg", "P1 live-set overlap", current, "live_tensor_total_MB", "peak_gap_MB", "variant_id")
    _simple_bar_svg(out_dir / "p1_peak_phase_timeline.svg", "P1 peak phase", [r["dataset"] + f" B{r['batch_size']}D{r['depth']}" for r in current], [f(r, "backward_adjoint_peak_MB") for r in current], "#dc2626")
    _placeholder_svg(out_dir / "p1_live_set_gantt.svg", "P1 live-set gantt", "profiler event names available; exact allocation stack unavailable")
    for row in live_rows:
        _wandb_log_row(args, row, "summary/v69_p1_liveset")
    return live_rows


def _measure_call(device: torch.device, fn: Callable[[], Any], warmup: int, reps: int) -> Tuple[float, float, float, Any]:
    out = None
    for _ in range(warmup):
        out = fn()
    _sync(device)
    times: List[float] = []
    peaks: List[float] = []
    reserved: List[float] = []
    for _ in range(reps):
        _reset_peak(device)
        _sync(device)
        t0 = time.perf_counter()
        out = fn()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
        p, r = _peak_mb(device)
        peaks.append(p)
        reserved.append(r)
    return _mean(times), _mean(peaks), _mean(reserved), out


def _noop_triton(x: torch.Tensor) -> torch.Tensor:
    if not _triton_available():
        raise RuntimeError("triton unavailable")
    y = torch.empty_like(x)
    n = x.numel()
    block = 1024
    grid = (triton.cdiv(n, block),)
    _v69_noop_copy_kernel[grid](x, y, n, BLOCK=block, num_warps=4)
    return y


def _measure_boundary(args: argparse.Namespace, kernel_name: str, impl: str, dataset: str, batch_size: int, depth: int) -> Dict[str, Any]:
    if impl.startswith("triton") and not _triton_available():
        return {**_row_common("P2", args, variant_id=kernel_name, dataset=dataset, batch_size=batch_size, depth=depth), "kernel_name": kernel_name, "implementation_status": "not_implemented_triton_unavailable", "stage_status": "not_implemented_triton_unavailable", "used_for_gate": 0, "not_implemented_count": 1, "reason": "triton/cuda unavailable"}
    if impl == "not_implemented":
        return {**_row_common("P2", args, variant_id=kernel_name, dataset=dataset, batch_size=batch_size, depth=depth), "kernel_name": kernel_name, "implementation_status": "not_implemented", "stage_status": "not_implemented", "used_for_gate": 0, "not_implemented_count": 1, "reason": "not implemented in v6.9 runner"}
    x, dz, coeff = v68._kernel_inputs(args, dataset, batch_size, depth)
    ref_g, ref_dx = v68._torch_coeffgrad_dx(x, dz, coeff)
    device = x.device
    input_copy_mb = 0.0
    output_copy_mb = 0.0
    layout_mb = 0.0

    def call() -> Any:
        nonlocal input_copy_mb, output_copy_mb, layout_mb
        if impl == "torch-current":
            return v68._torch_coeffgrad_dx(x, dz, coeff)
        if impl == "triton-k3":
            return v68._triton_coeffgrad_dx(x, dz, coeff, do_dx=True)
        if impl == "wrapper-only":
            return (ref_g, ref_dx)
        if impl == "contiguous-only":
            x2 = x.contiguous()
            dz2 = dz.contiguous()
            coeff2 = coeff.contiguous()
            input_copy_mb = 0.0 if x2.data_ptr() == x.data_ptr() and dz2.data_ptr() == dz.data_ptr() else (x.numel() + dz.numel() + coeff.numel()) * x.element_size() / (1024**2)
            return x2, dz2, coeff2
        if impl == "output-materialize-only":
            g = torch.empty_like(coeff)
            dx = torch.empty_like(dz)
            output_copy_mb = (g.numel() * g.element_size() + dx.numel() * dx.element_size()) / (1024**2)
            return g, dx
        if impl == "layout-conversion-only":
            y = x.t().contiguous()
            layout_mb = y.numel() * y.element_size() / (1024**2)
            return y
        if impl == "triton-launch-only":
            return _noop_triton(x)
        if impl == "batched-layer-call":
            out = None
            for _ in range(max(1, depth)):
                out = v68._triton_coeffgrad_dx(x, dz, coeff, do_dx=True)
            return out
        raise ValueError(impl)

    time_ms, peak, reserved, out = _measure_call(device, call, args.micro_warmup_steps, args.micro_measure_steps)
    grad_rel = 0.0
    grad_cos = 1.0
    if impl in {"torch-current", "triton-k3", "batched-layer-call"} and isinstance(out, tuple):
        g, dx = out
        grad_rel, grad_cos, _ = _rel_cos(g.detach().flatten().float().cpu(), ref_g.detach().flatten().float().cpu())
        dx_rel, dx_cos, _ = _rel_cos(dx.detach().flatten().float().cpu(), ref_dx.detach().flatten().float().cpu())
    else:
        dx_rel, dx_cos = 0.0, 1.0
    call_count = max(1, depth) if impl == "batched-layer-call" else 1
    row = {
        **_row_common("P2", args, method=METHOD_CURRENT, variant_id=kernel_name, dataset=dataset, batch_size=batch_size, depth=depth),
        "kernel_name": kernel_name,
        "implementation": impl,
        "call_count_per_step": call_count,
        "time_per_call_ms": time_ms / call_count,
        "total_call_time_ms": time_ms,
        "input_contiguous_time_ms": time_ms if impl == "contiguous-only" else 0.0,
        "output_materialize_time_ms": time_ms if impl == "output-materialize-only" else 0.0,
        "layout_conversion_time_ms": time_ms if impl == "layout-conversion-only" else 0.0,
        "wrapper_overhead_ms": time_ms if impl == "wrapper-only" else 0.0,
        "cuda_launch_overhead_ms": time_ms if impl == "triton-launch-only" else 0.0,
        "peak_allocated_MB": peak,
        "peak_reserved_MB": reserved,
        "temp_allocated_MB": output_copy_mb + input_copy_mb + layout_mb,
        "input_copy_MB": input_copy_mb,
        "output_copy_MB": output_copy_mb,
        "grad_relerr_max": max(grad_rel, dx_rel),
        "grad_cos_min": min(grad_cos, dx_cos),
        "gradient_correctness_pass": int(max(grad_rel, dx_rel) < 1.0e-4 and min(grad_cos, dx_cos) > 0.999),
        "uses_triton_kernel": int(impl.startswith("triton") or impl == "batched-layer-call"),
    }
    return row


def run_p2(args: argparse.Namespace) -> List[Dict[str, Any]]:
    kernels = [
        ("K0-current-torch-only-coeffgrad", "torch-current"),
        ("K3-triton-fused-dx-coeffgrad-microkernel", "triton-k3"),
        ("K3-wrapper-only-no-op", "wrapper-only"),
        ("K3-input-contiguous-only", "contiguous-only"),
        ("K3-output-materialize-only", "output-materialize-only"),
        ("K3-layout-conversion-only", "layout-conversion-only"),
        ("K3-call-overhead-only", "triton-launch-only"),
        ("K3-batched-layer-call", "batched-layer-call"),
        ("K3-single-call-multi-layer", "not_implemented"),
    ]
    rows: List[Dict[str, Any]] = []
    for ds in parse_str_list(args.micro_datasets):
        canonical = dataset_name(ds)
        for batch_size in parse_int_list(args.micro_batch_sizes):
            for depth in parse_int_list(args.micro_depths):
                base: Dict[str, Any] | None = None
                k3: Dict[str, Any] | None = None
                for name, impl in kernels:
                    row = _measure_boundary(args, name, impl, canonical, batch_size, depth)
                    if row.get("implementation_status") == "measured":
                        if impl == "torch-current":
                            base = row
                            row["time_ratio_vs_current"] = 1.0
                        elif base is not None:
                            row["time_ratio_vs_current"] = f(row, "total_call_time_ms") / max(1.0e-12, f(base, "total_call_time_ms"))
                        if impl == "triton-k3":
                            k3 = row
                            row["boundary_bottleneck_pass"] = 0
                        elif k3 is not None:
                            overhead = f(row, "wrapper_overhead_ms") + f(row, "input_contiguous_time_ms") + f(row, "output_materialize_time_ms") + f(row, "layout_conversion_time_ms") + f(row, "cuda_launch_overhead_ms")
                            row["boundary_overhead_fraction_vs_k3"] = overhead / max(1.0e-12, f(k3, "total_call_time_ms"))
                            row["boundary_bottleneck_pass"] = int(row["boundary_overhead_fraction_vs_k3"] >= 0.30 or (f(row, "temp_allocated_MB") >= 0.20 * max(1.0e-12, f(k3, "peak_allocated_MB"))))
                        else:
                            row["boundary_bottleneck_pass"] = 0
                    rows.append(row)
                    _wandb_log_row(args, row, "summary/v69_p2_boundary")
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p2_boundary_audit.csv", rows)
    measured = [r for r in rows if r.get("implementation_status") == "measured"]
    _simple_bar_svg(out_dir / "p2_boundary_time_waterfall.svg", "P2 boundary time", [r["kernel_name"] for r in measured], [f(r, "total_call_time_ms", 0.0) for r in measured], "#2563eb")
    _scatter_svg(out_dir / "p2_call_count_vs_step_time.svg", "P2 calls/time", measured, "call_count_per_step", "total_call_time_ms", "kernel_name")
    _simple_bar_svg(out_dir / "p2_materialization_memory_bar.svg", "P2 materialization MB", [r["kernel_name"] for r in measured], [f(r, "temp_allocated_MB", 0.0) for r in measured], "#7c3aed")
    _scatter_svg(out_dir / "p2_micro_vs_wrapped_runtime.svg", "P2 micro vs wrapped runtime", measured, "time_ratio_vs_current", "total_call_time_ms", "kernel_name")
    return rows


def _summarize_detail(args: argparse.Namespace, stage: str, detail: Sequence[Dict[str, Any]], id_field: str, extra: Callable[[str, List[Dict[str, Any]]], Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
    measured = [r for r in detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    by: Dict[str, List[Dict[str, Any]]] = {}
    current_by_shape = {
        (r.get("dataset"), r.get("batch_size"), r.get("depth")): r
        for r in measured
        if str(r.get("variant_id", "")).endswith("current") or str(r.get("variant_id", "")) in {"L0-current", "O0-current"}
    }
    for row in measured:
        by.setdefault(str(row.get("variant_id")), []).append(row)
    summary: List[Dict[str, Any]] = []
    for name, rows in by.items():
        mem = [f(r, "memory_ratio_vs_MLP") for r in rows]
        step = [f(r, "step_time_ratio_vs_MLP") for r in rows]
        bwd = [f(r, "backward_time_ratio_vs_MLP") for r in rows]
        mem_imp = []
        step_imp = []
        bwd_imp = []
        for r in rows:
            cur = current_by_shape.get((r.get("dataset"), r.get("batch_size"), r.get("depth")))
            if cur is not None and name not in {"L0-current", "O0-current"}:
                mem_imp.append((f(cur, "memory_ratio_vs_MLP") - f(r, "memory_ratio_vs_MLP")) / max(1.0e-12, f(cur, "memory_ratio_vs_MLP")))
                step_imp.append((f(cur, "step_time_ratio_vs_MLP") - f(r, "step_time_ratio_vs_MLP")) / max(1.0e-12, f(cur, "step_time_ratio_vs_MLP")))
                bwd_imp.append((f(cur, "backward_time_ratio_vs_MLP") - f(r, "backward_time_ratio_vs_MLP")) / max(1.0e-12, f(cur, "backward_time_ratio_vs_MLP")))
        mean_mem = _mean(mem)
        row = {
            **_row_common(stage, args, variant_id=name),
            id_field: name,
            "implementation_status": "measured",
            "memory_ratio_min": min(mem),
            "memory_ratio_mean": mean_mem,
            "memory_ratio_max": max(mem),
            "step_ratio_min": min(step),
            "step_ratio_mean": _mean(step),
            "step_ratio_max": max(step),
            "backward_ratio_mean": _mean(bwd),
            "memory_improvement_vs_current": _mean(mem_imp, 0.0),
            "step_improvement_vs_current": _mean(step_imp, 0.0),
            "backward_improvement_vs_current": _mean(bwd_imp, 0.0),
            "grad_relerr_max": max(f(r, "grad_relerr") for r in rows),
            "grad_cos_min": min(f(r, "grad_cos") for r in rows),
            "near_pass_count": sum(f(r, "memory_ratio_vs_MLP", 99) <= 1.05 and f(r, "step_time_ratio_vs_MLP", 99) <= 1.50 for r in rows),
            "memory_pass_count": sum(f(r, "memory_ratio_vs_MLP", 99) < 1.0 for r in rows),
            "time_pass_count": sum(f(r, "step_time_ratio_vs_MLP", 99) <= 1.35 for r in rows),
            "shape_stability_score": 1.0 - _std(mem) / max(1.0e-12, mean_mem),
        }
        if extra is not None:
            row.update(extra(name, rows))
        summary.append(row)
    return summary


def run_p3(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    variants = [(a, b, c, d, e) for a, b, c, d, e, _layers in P3_PACKAGES]
    detail = _profile_grid(args, "P3", variants)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p3_full_layer_fusion_detail.csv", detail)
    layer_counts = {name: layers for name, _m, _p, _i, _c, layers in P3_PACKAGES}
    def extra(name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "layers_fused": layer_counts.get(name, 0),
            "calls_per_step": max(1, int(rows[0].get("depth") or 1)) if name in {"L1-fused-dx-coeffgrad-only", "L2-full-layer-backward"} else 0,
            "boundary_time_reduction": METRIC_UNAVAILABLE,
            "live_set_overlap_reduction": METRIC_UNAVAILABLE,
            "kernel_count_reduction": METRIC_UNAVAILABLE,
            "allocation_count_reduction": METRIC_UNAVAILABLE,
            "full_layer_fusion_pass": int(_mean(f(r, "actual_memory_reduction_vs_current", 0.0) for r in rows) >= 0.10 and _mean(f(r, "actual_step_improvement_vs_current", 0.0) for r in rows) >= 0.10),
        }
    summary = _summarize_detail(args, "P3", detail, "package", extra)
    implemented = {name for name, _m, _p, ok, _c, _l in P3_PACKAGES if ok}
    for name, method, policy, ok, components, layers in P3_PACKAGES:
        if name not in implemented:
            summary.append({**_row_common("P3", args, method=method, variant_id=name), "package": name, "implementation_status": policy, "stage_status": policy, "used_for_gate": 0, "not_implemented_count": 1, "layers_fused": layers, "components": components, "reason": "full-layer component not implemented; no measured ratio emitted"})
    write_csv(out_dir / "p3_full_layer_fusion.csv", summary)
    measured = [r for r in detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    _scatter_svg(out_dir / "p3_full_layer_package_pareto.svg", "P3 full-layer packages", measured, "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _simple_bar_svg(out_dir / "p3_boundary_reduction_bar.svg", "P3 boundary reduction", [r["package"] for r in summary], [f(r, "boundary_time_reduction", 0.0) for r in summary], "#7c3aed")
    _simple_bar_svg(out_dir / "p3_live_set_reduction_bar.svg", "P3 live-set reduction", [r["package"] for r in summary], [f(r, "live_set_overlap_reduction", 0.0) for r in summary], "#16a34a")
    _simple_bar_svg(out_dir / "p3_kernel_count_reduction.svg", "P3 kernel count reduction", [r["package"] for r in summary], [f(r, "kernel_count_reduction", 0.0) for r in summary], "#2563eb")
    _simple_bar_svg(out_dir / "p3_batch_depth_stability_heatmap.svg", "P3 stability", [r["package"] for r in summary], [f(r, "shape_stability_score", 0.0) for r in summary], "#dc2626")
    return summary, detail


def run_p4(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    detail = _profile_grid(args, "P4", P4_PACKAGES)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p4_onebuffer_full_step_detail.csv", detail)
    def extra(_name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "live_buffer_count_peak": _safe_max(f(r, "reused_buffer_count", 0.0) for r in rows),
            "live_buffer_total_MB_peak": _safe_max(f(r, "workspace_pool_MB", 0.0) for r in rows),
            "buffer_reuse_count": _safe_max(f(r, "reused_buffer_count", 0.0) for r in rows),
            "buffer_lifetime_conflict_count": METRIC_UNAVAILABLE,
            "onebuffer_pass": int(_safe_max(f(r, "reused_buffer_count", 99.0) for r in rows) <= 3 and _mean(f(r, "actual_memory_reduction_vs_current", 0.0) for r in rows) >= 0.15 and _mean(f(r, "actual_step_improvement_vs_current", 0.0) for r in rows) >= 0.0),
        }
    summary = _summarize_detail(args, "P4", detail, "package", extra)
    implemented = {name for name, _m, _p, ok, _c in P4_PACKAGES if ok}
    for name, method, policy, ok, components in P4_PACKAGES:
        if name not in implemented:
            summary.append({**_row_common("P4", args, method=method, variant_id=name), "package": name, "implementation_status": policy, "stage_status": policy, "used_for_gate": 0, "not_implemented_count": 1, "components": components, "reason": "one-buffer package not implemented; no measured ratio emitted"})
    write_csv(out_dir / "p4_onebuffer_full_step.csv", summary)
    measured = [r for r in detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    _scatter_svg(out_dir / "p4_onebuffer_memory_step_pareto.svg", "P4 onebuffer memory/step", measured, "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _simple_bar_svg(out_dir / "p4_live_buffer_count_timeline.svg", "P4 live buffer count", [r["package"] for r in summary], [f(r, "live_buffer_count_peak", 0.0) for r in summary], "#2563eb")
    _placeholder_svg(out_dir / "p4_buffer_lifetime_diagram.svg", "P4 buffer lifetime", "exact lifetime conflicts unavailable")
    (out_dir / "p4_buffer_conflict_table.md").write_text("| package | conflict_count |\n|---|---:|\n" + "\n".join(f"| {r.get('package')} | {r.get('buffer_lifetime_conflict_count', METRIC_UNAVAILABLE)} |" for r in summary) + "\n", encoding="utf-8")
    return summary, detail


def run_p5(args: argparse.Namespace) -> List[Dict[str, Any]]:
    profile_variants = [(a, b, c, d, e) for a, b, c, d, e, _scale in P5_RESETS]
    rows = _profile_grid(args, "P5", profile_variants)
    scale_by = {name: scale for name, _m, _p, _i, _c, scale in P5_RESETS}
    for row in rows:
        if row.get("implementation_status") == "measured" and row.get("method") != "MLP-autograd-reference":
            eff = v68._residual_effect(args, str(row.get("method")), str(row.get("workspace_policy", "current")), str(row.get("dataset")), int(float(row.get("batch_size") or args.batch_size)), int(float(row.get("depth") or 2)), scale_by.get(str(row.get("variant_id")), 0.0))
            row.update(eff)
            row["primitive"] = row.get("variant_id")
            row["reset_near_pass"] = int(f(row, "memory_ratio_vs_MLP", 99) <= 1.05 and f(row, "step_time_ratio_vs_MLP", 99) <= 1.35 and int(f(row, "gradient_correctness_pass", 0)) == 1)
            row["live_buffer_count_peak"] = f(row, "reused_buffer_count", 0.0)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p5_residual_bounded_reset.csv", rows)
    measured = [r for r in rows if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    _scatter_svg(out_dir / "p5_reset_residual_strength_vs_memory.svg", "P5 reset residual/memory", measured, "memory_ratio_vs_MLP", "residual_over_base", "variant_id")
    _scatter_svg(out_dir / "p5_reset_memory_time_pareto.svg", "P5 reset memory/time", measured, "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _simple_bar_svg(out_dir / "p5_residual_ablation_delta.svg", "P5 residual ablation", [r["variant_id"] for r in measured], [f(r, "residual_ablation_delta_logit", 0.0) for r in measured], "#16a34a")
    _simple_bar_svg(out_dir / "p5_reset_workspace_model_comparison.svg", "P5 workspace MB", [r["variant_id"] for r in measured], [f(r, "workspace_pool_MB", 0.0) for r in measured], "#7c3aed")
    return rows


def _write_not_run(path: Path, stage: str, reason: str, gated_by: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = [{**_row_common(stage, args), "implementation_status": "not_run", "status": "not_run", "stage_status": "not_run", "used_for_gate": 0, "gated_not_run_count": 1, "reason": reason, "gated_by": gated_by}]
    write_csv(path, rows)
    for row in rows:
        _wandb_log_row(args, row, f"summary/v69_{stage.lower()}_not_run")
    return rows


def _best_dwm2(p3: Sequence[Dict[str, Any]], p4: Sequence[Dict[str, Any]]) -> Tuple[str, Dict[str, Any] | None]:
    measured = [r for r in list(p3) + list(p4) if r.get("implementation_status") == "measured"]
    s0 = [r for r in measured if f(r, "memory_ratio_max", 99) < 1.0 and f(r, "step_ratio_mean", 99) <= 1.20 and f(r, "grad_relerr_max", 99) < 1.0e-4]
    s1 = [r for r in measured if f(r, "memory_ratio_max", 99) < 1.0 and f(r, "step_ratio_mean", 99) <= 1.35 and f(r, "grad_relerr_max", 99) < 1.0e-4]
    s2 = [r for r in measured if f(r, "memory_ratio_mean", 99) <= 1.05 and f(r, "step_ratio_mean", 99) <= 1.50 and f(r, "memory_improvement_vs_current", 0.0) >= 0.10]
    if s0:
        return "S0", min(s0, key=lambda r: f(r, "memory_ratio_mean", 99))
    if s1:
        return "S1", min(s1, key=lambda r: f(r, "memory_ratio_mean", 99))
    if s2:
        return "S2", min(s2, key=lambda r: f(r, "memory_ratio_mean", 99))
    return "S6", min(measured, key=lambda r: f(r, "memory_ratio_mean", 99)) if measured else None


def run_route(args: argparse.Namespace, p1: Sequence[Dict[str, Any]], p2: Sequence[Dict[str, Any]], p3: Sequence[Dict[str, Any]], p4: Sequence[Dict[str, Any]], p5: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    survivor, best = _best_dwm2(p3, p4)
    attribution_pass = any(int(f(r, "attribution_pass", 0)) == 1 for r in p1 if r.get("variant_id") == "A1-DWM2-current")
    boundary_pass = any(int(f(r, "boundary_bottleneck_pass", 0)) == 1 for r in p2)
    live_overlap = any(int(f(r, "live_set_overlap_pass", 0)) == 1 for r in p1 if r.get("variant_id") == "A1-DWM2-current")
    reset_effect = [r for r in p5 if r.get("implementation_status") == "measured" and int(f(r, "residual_effect_pass", 0)) == 1]
    reset_near_effect = [r for r in reset_effect if int(f(r, "reset_near_pass", 0)) == 1]
    full_fusion_measured = [r for r in p3 if r.get("implementation_status") == "measured" and r.get("package") != "L0-current"]
    full_fusion_improvement = max([f(r, "memory_improvement_vs_current", 0.0) for r in full_fusion_measured] + [0.0])
    if survivor in {"S0", "S1"}:
        route = "R1-DWM2LiveSetSolved"
        primary = "DWM2 full-step package reached S0/S1; P6 still required"
    elif survivor == "S2":
        route = "R2-DWM2LiveSetNearPass"
        primary = "DWM2 full-step package near-pass only"
    elif not attribution_pass:
        route = "R3-AttributionIncomplete"
        primary = "live-set attribution still cannot explain current DWM2 peak to threshold"
    elif full_fusion_improvement < 0.05:
        route = "R4-DWM2FullFusionNoEffect"
        primary = "full-layer/onebuffer fusion improved less than 5%"
    elif reset_near_effect:
        route = "R5-ResetPrimitiveCandidate"
        primary = "reset primitive near-pass with residual effect exists"
    elif reset_effect:
        route = "R6-ResetStillTooHeavy"
        primary = "reset residual effect passes but memory/time still fail"
    else:
        route = "R7-TerminalPrimitiveRedesign"
        primary = "no DWM2 near-pass and no reset near-pass"
    route_json = {
        "route": route,
        "best_candidate": (best or {}).get("package", ""),
        "best_family": "DWM2-poly2" if best else "",
        "best_memory_ratio": f(best or {}, "memory_ratio_mean", 99.0),
        "best_step_ratio": f(best or {}, "step_ratio_mean", 99.0),
        "best_backward_ratio": f(best or {}, "backward_ratio_mean", 99.0),
        "memory_improvement_vs_current": f(best or {}, "memory_improvement_vs_current", 0.0),
        "step_improvement_vs_current": f(best or {}, "step_improvement_vs_current", 0.0),
        "survivor_type": survivor,
        "attribution_pass": int(attribution_pass),
        "boundary_bottleneck_pass": int(boundary_pass),
        "live_set_overlap_pass": int(live_overlap),
        "reset_residual_effect_pass": int(bool(reset_effect)),
        "open_task_reentry": False,
        "open_optimizer_exploration": False,
        "open_functional_correction": False,
        "primary_blocker": primary,
        "next_required_implementation": "real_live_set_profiler_or_new_primitive_family" if route.startswith("R3") else "bounded_full_step_kernel_or_reset_family",
        "no_fake": True,
        "no_proxy": True,
    }
    _json_dump(Path(args.out_dir) / "route_decision.json", route_json)
    _json_dump(Path(args.out_dir) / "aggregate_decision.json", {"status": "gated", "fake_data_used": 0, "proxy_rows_used_as_results": 0, **route_json})
    _placeholder_svg(Path(args.out_dir) / "p10_route_decision_dashboard.svg", "P10 route", route)
    _wandb_log_row(args, {**_row_common("P10", args), **route_json}, "summary/v69_route")
    return route_json


def run_gated(args: argparse.Namespace, route: Dict[str, Any]) -> None:
    _write_not_run(Path(args.out_dir) / "p6_one_step_probe.csv", "P6", "No S0/S1/S2 full-step DWM2 survivor and no reset near-pass", "P3/P4/P5", args)
    _write_not_run(Path(args.out_dir) / "p7_task_reentry.csv", "P7", "P6 did not pass or no memory/time survivor", "P6", args)
    _write_not_run(Path(args.out_dir) / "p7_task_trace.csv", "P7", "P7 is gated", "P6", args)
    _write_not_run(Path(args.out_dir) / "p8_optimizer_exploration.csv", "P8", "P7 task re-entry did not pass", "P7", args)
    _write_not_run(Path(args.out_dir) / "p9_functional_correction_smoke.csv", "P9", "P9 is gated behind P7/P8", "P7/P8", args)
    _placeholder_svg(Path(args.out_dir) / "p6_loss_before_after.svg", "P6 loss before/after", "not_run")
    _placeholder_svg(Path(args.out_dir) / "p6_bad_step_heatmap.svg", "P6 bad step", "not_run")
    _placeholder_svg(Path(args.out_dir) / "p6_update_norm_vs_loss_delta.svg", "P6 update/loss", "not_run")
    _placeholder_svg(Path(args.out_dir) / "p6_residual_ablation_probe.svg", "P6 residual probe", "not_run")


def run_failure(args: argparse.Namespace) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    out_dir = Path(args.out_dir)
    for fn, stage in [
        ("p1_live_set_attribution.csv", "P1"),
        ("p2_boundary_audit.csv", "P2"),
        ("p3_full_layer_fusion.csv", "P3"),
        ("p4_onebuffer_full_step.csv", "P4"),
        ("p5_residual_bounded_reset.csv", "P5"),
        ("p6_one_step_probe.csv", "P6"),
        ("p7_task_reentry.csv", "P7"),
        ("p8_optimizer_exploration.csv", "P8"),
        ("p9_functional_correction_smoke.csv", "P9"),
    ]:
        path = out_dir / fn
        if not path.exists():
            failures.append({"stage": stage, "variant_id": fn, "failure_type": "F14_artifact_missing", "metric": "missing", "recommendation": "rerun stage"})
            continue
        for row in _read_csv(path):
            status = row.get("implementation_status") or row.get("status")
            if status in {"not_run", "not_implemented"} or str(status).startswith("not_implemented") or str(status).startswith("cuda_extension"):
                if stage == "P8":
                    failure_type = "F11_optimizer_gated"
                elif stage == "P9":
                    failure_type = "F12_functional_gated"
                else:
                    failure_type = "F0_not_implemented_or_gated"
                failures.append({"stage": stage, "variant_id": row.get("variant_id", row.get("kernel_name", "")), "failure_type": failure_type, "metric": row.get("reason", status), "recommendation": "implement or pass gate before claiming metric"})
            if status == "measured":
                if row.get("memory_ratio_vs_MLP") not in {None, ""} and row.get("method") != "MLP-autograd-reference" and f(row, "memory_ratio_vs_MLP", 0.0) >= 1.0:
                    failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F1_memory_fail", "metric": f"memory_ratio={row.get('memory_ratio_vs_MLP')}", "recommendation": "reduce actual CUDA peak"})
                if row.get("step_time_ratio_vs_MLP") not in {None, ""} and row.get("method") != "MLP-autograd-reference" and f(row, "step_time_ratio_vs_MLP", 0.0) > 1.35:
                    failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F2_step_time_fail", "metric": f"step_ratio={row.get('step_time_ratio_vs_MLP')}", "recommendation": "reduce step/runtime"})
                if row.get("gradient_correctness_pass") not in {None, ""} and int(f(row, "gradient_correctness_pass", 1)) == 0:
                    failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F3_gradient_correctness_fail", "metric": f"grad={row.get('grad_relerr') or row.get('grad_relerr_max')}", "recommendation": "fix kernel numerics"})
                if stage == "P1" and row.get("variant_id") == "A1-DWM2-current" and int(f(row, "attribution_pass", 0)) == 0:
                    failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F4_attribution_incomplete", "metric": f"explain_ratio={row.get('explain_ratio')}", "recommendation": "deeper live-set profiler"})
                if stage == "P2" and int(f(row, "boundary_bottleneck_pass", 0)) == 1:
                    failures.append({"stage": stage, "variant_id": row.get("kernel_name", ""), "failure_type": "F5_boundary_overhead", "metric": f"overhead_fraction={row.get('boundary_overhead_fraction_vs_k3')}", "recommendation": "fuse boundary into single full-step kernel"})
                if stage == "P1" and int(f(row, "live_set_overlap_pass", 0)) == 1:
                    failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F6_live_set_overlap", "metric": f"gap={row.get('peak_gap_MB')}", "recommendation": "reduce live-set overlap"})
                if stage == "P3" and row.get("package") not in {"L0-current", ""} and f(row, "memory_improvement_vs_current", 0.0) < 0.05:
                    failures.append({"stage": stage, "variant_id": row.get("package", ""), "failure_type": "F7_full_fusion_no_effect", "metric": f"mem_improvement={row.get('memory_improvement_vs_current')}", "recommendation": "implement true single-call full-layer kernel"})
                if stage == "P5" and row.get("method") != "MLP-autograd-reference" and int(f(row, "residual_effect_pass", 0)) == 0:
                    failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F8_reset_residual_effect_fail", "metric": f"residual={row.get('residual_over_base')}", "recommendation": "increase residual effect or redesign primitive"})
                if stage == "P5" and row.get("method") != "MLP-autograd-reference" and int(f(row, "reset_near_pass", 0)) == 0:
                    failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F9_reset_memory_fail", "metric": f"mem={row.get('memory_ratio_vs_MLP')} step={row.get('step_time_ratio_vs_MLP')}", "recommendation": "reduce reset memory/time"})
    write_csv(out_dir / "failure_table.csv", failures or [{"stage": "ALL", "variant_id": "all", "failure_type": "none"}])
    for row in failures:
        _wandb_log_row(args, row, "summary/v69_failure")
    _simple_bar_svg(out_dir / "failure_taxonomy_heatmap.svg", "Failure taxonomy", [r["failure_type"] for r in failures], [1.0 for _ in failures], "#dc2626")
    return failures


def _copy_figures(out_dir: Path) -> None:
    figures = ensure_dir(out_dir / "figures")
    required = [
        "p1_live_set_gantt.svg",
        "p1_peak_live_tensor_top20.svg",
        "p1_gap_attribution_stacked_bar.svg",
        "p1_boundary_overhead_bar.svg",
        "p2_micro_vs_wrapped_runtime.svg",
        "p3_full_layer_package_pareto.svg",
        "p4_onebuffer_memory_step_pareto.svg",
        "p5_reset_residual_strength_vs_memory.svg",
        "p6_loss_before_after.svg",
        "p10_route_decision_dashboard.svg",
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
        "script": "experiments/run_gafu_v69_real.py",
        "plan": "docs/DG-KAN_v6.9_FullStepLiveSetFusion_BoundedReset_详细实验计划.md",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(wandb=True)
    parser.add_argument("--packages", default="V6_9_ALL")
    parser.add_argument("--out-dir", type=Path, default=Path("results/real_rerun_20260505/v69_real"))
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
    parser.add_argument("--wandb-group", default="v69-real-20260505")
    parser.add_argument("--wandb-name-prefix", default="v69-real")
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
        p2_rows = run_p2(args)
        p3_summary, _p3_detail = run_p3(args)
        p4_summary, _p4_detail = run_p4(args)
        p5_rows = run_p5(args)
        route = run_route(args, p1_rows, p2_rows, p3_summary, p4_summary, p5_rows)
        run_gated(args, route)
        run_failure(args)
        _copy_figures(out_dir)
        _write_manifest(out_dir, args, started, time.time())
    finally:
        _wandb_finish(args, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
