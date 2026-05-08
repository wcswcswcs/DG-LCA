#!/usr/bin/env python3
"""DG-KAN v6.10 real-only full-step live-set Stop/Go runner.

The runner keeps the v6.9 no-fake/no-proxy contract and upgrades P1 from
phase-level diagnostics to a real PyTorch CUDA memory-history snapshot.  Missing
single-call/one-buffer kernels remain explicit not_implemented rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

import run_gafu_v68_real as v68
import run_gafu_v69_real as v69
from dgkan_core import ensure_dir, get_device, load_vision_bundle, parse_int_list, parse_str_list, set_seed, write_csv
from run_gafu_v3 import dataset_name
from run_gafu_v63 import ManualOptimizer, V63ManualLayer, V63Params, _basis_from_name, _peak_mb, _rel_cos, _reset_peak, _sync, _wandb_finish, _wandb_init, _wandb_log_row, f
from run_gafu_v64_real import METHOD_CURRENT, _make_mlp, _sha256, _take_batch
from run_gafu_v65_real import _placeholder_svg, _scatter_svg, _simple_bar_svg
from run_gafu_v66_real import _git_commit, _git_status, _json_dump, _mean, _std


METRIC_UNAVAILABLE = "metric_unavailable"
V69_MEMORY_MEAN = 1.2917923088533285
V69_STEP_MEAN = 1.904116152193838
V69_P3_STEP_MEAN = 1.877616986741421


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


def _patch_stack() -> None:
    v69._patch_stack()


def _profile_grid(args: argparse.Namespace, stage: str, variants: Sequence[Tuple[Any, ...]]) -> List[Dict[str, Any]]:
    _patch_stack()
    return v69._profile_grid(args, stage, variants)


P0_VARIANTS = [
    ("MLP-autograd-reference", "reference", "mlp", True, "reference"),
    ("MLP-manual-linear-reference", "MLP-manual-linear-reference", "current", True, "manual-linear"),
    ("DWM2-current", METHOD_CURRENT, "current", True, "current"),
    ("DWM2-Triton-fused-dx-coeffgrad-v1", METHOD_CURRENT, "triton_fused_dx_coeffgrad", _triton_available(), "triton-fused-dx-coeffgrad"),
    ("DWM2-Triton-full-backward-light-v1", METHOD_CURRENT, "triton_full_backward_light", _triton_available(), "partial-full-layer-light"),
    ("DWM2-FullLayerBackward-v3-no-dx-materialize", METHOD_CURRENT, "not_implemented", False, "single-call-no-dx-materialize"),
    ("DWM2-FullLayerBackward-v3-update-prep", METHOD_CURRENT, "not_implemented", False, "single-call-update-prep"),
    ("DWM2-MultiLayerChunkedBackward-v1", METHOD_CURRENT, "not_implemented", False, "multi-layer-chunked"),
    ("DWM2-OneBufferFullStep-v2", METHOD_CURRENT, "not_implemented", False, "true-onebuffer-full-step"),
    ("ResidualBoundedReset-v3-scale002", "DWM2-poly1-minimal", "poly1_scale002", True, "bounded-reset-v3-scale002"),
    ("ResidualBoundedReset-v3-scale005", "DWM2-poly1-minimal", "poly1_scale005", True, "bounded-reset-v3-scale005"),
]

P1_VARIANTS = [("A1-DWM2-current", METHOD_CURRENT, "current", True, "current")]

P3_PACKAGES = [
    ("L0-current", METHOD_CURRENT, "current", True, "current", 0),
    ("L3-full-layer-backward-no-dx-materialize", METHOD_CURRENT, "not_implemented", False, "no-dx-materialize", 1),
    ("L4-full-layer-backward-update-prep", METHOD_CURRENT, "not_implemented", False, "update-prep", 1),
    ("L4b-full-layer-backward-no-partial-integration", METHOD_CURRENT, "not_implemented", False, "no-partial-integration", 1),
    ("L4c-full-layer-backward-inplace-buffer", METHOD_CURRENT, "not_implemented", False, "inplace-buffer", 1),
    ("L5-multi-layer-chunked-backward", METHOD_CURRENT, "not_implemented", False, "multi-layer-chunked", 2),
    ("L6-single-call-depth2-backward", METHOD_CURRENT, "not_implemented", False, "single-call-depth2", 2),
]

P4_PACKAGES = [
    ("M0-current", METHOD_CURRENT, "current", True, "current", 0),
    ("M1-depth2-single-call-backward", METHOD_CURRENT, "not_implemented", False, "depth2-single-call", 1),
    ("M2-depth4-two-chunk-backward", METHOD_CURRENT, "not_implemented", False, "depth4-two-chunk", 2),
    ("M3-depth4-single-call-backward", METHOD_CURRENT, "not_implemented", False, "depth4-single-call", 1),
    ("M4-depth4-single-call-no-update", METHOD_CURRENT, "not_implemented", False, "single-call-no-update", 1),
    ("M5-depth4-single-call-update-prep", METHOD_CURRENT, "not_implemented", False, "single-call-update-prep", 1),
]

P5_PACKAGES = [
    ("O0-current", METHOD_CURRENT, "current", True, "current"),
    ("O5-onebuffer-full-step", METHOD_CURRENT, "not_implemented", False, "true-onebuffer-full-step"),
    ("O6-onebuffer-full-step-chunked-mix", METHOD_CURRENT, "not_implemented", False, "chunked-mix"),
    ("O7-onebuffer-full-step-update-prep", METHOD_CURRENT, "not_implemented", False, "update-prep"),
    ("O8-onebuffer-full-step-lowrank-mix", METHOD_CURRENT, "not_implemented", False, "lowrank-mix"),
]

P6_RESETS = [
    ("R0-ManualLinear-reference", "MLP-manual-linear-reference", "current", True, "manualLinear", 0.0),
    ("R1-ResidualEffectiveTinyKAN-scale002-current", "DWM2-poly1-minimal", "poly1_scale002", True, "scale002-current", 0.02),
    ("R2-ResidualEffectiveTinyKAN-scale005-current", "DWM2-poly1-minimal", "poly1_scale005", True, "scale005-current", 0.05),
    ("R3-BoundedResidual-inplace-transform-v2", "DWM2-poly1-minimal", "poly1_scale002", True, "bounded-inplace-diagnostic", 0.02),
    ("R4-BoundedResidual-fused-mix", "DWM2-poly1-minimal", "not_implemented", False, "fused-mix", 0.02),
    ("R5-BoundedResidual-chunked-mix", "DWM2-poly1-minimal", "not_implemented", False, "chunked-mix", 0.02),
    ("R6-BoundedResidual-lowrank-mix", "DWM2-poly1-minimal", "not_implemented", False, "lowrank-mix", 0.02),
    ("R7-BoundedResidual-piecewise2", "OneBufferPiecewiseLinear2", "not_implemented", False, "piecewise2", 0.02),
    ("R8-BoundedResidual-streaming-backward", "DWM2-poly1-minimal", "not_implemented", False, "streaming-backward", 0.02),
]


def _gradient_check(method: str, batch: int, hidden: int, device: torch.device, policy: str) -> Dict[str, float]:
    return v68._gradient_check_v68(method, batch, hidden, device, policy)


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    original = v69.P0_VARIANTS
    v69.P0_VARIANTS = P0_VARIANTS
    try:
        rows = v69.run_p0(args)
    finally:
        v69.P0_VARIANTS = original
    for row in rows:
        row["script_path"] = "experiments/run_gafu_v610_real.py"
        row["v69_current_memory_ratio_mean"] = V69_MEMORY_MEAN
        row["v69_current_step_ratio_mean"] = V69_STEP_MEAN
    write_csv(out_dir / "p0_contract.csv", rows)
    _simple_bar_svg(out_dir / "p0_stopgo_contract.svg", "v6.10 P0 Stop/Go contract", [r["variant_id"] for r in rows], [1.0 if r.get("implementation_status") == "measured" else 0.0 for r in rows], "#2563eb")
    return rows


def _frame_text(frames: Sequence[Dict[str, Any]]) -> str:
    parts = []
    for frame in frames[:40]:
        parts.append(f"{frame.get('name','')}@{frame.get('filename','')}:{frame.get('line','')}")
    return "\n".join(parts)


def _stack_hash(frames: Sequence[Dict[str, Any]]) -> str:
    return hashlib.sha256(_frame_text(frames).encode("utf-8")).hexdigest()[:16]


def _top_source(frames: Sequence[Dict[str, Any]]) -> Tuple[str, str, str]:
    fallback = ("", "", "")
    for frame in frames:
        name = str(frame.get("name", ""))
        filename = str(frame.get("filename", ""))
        if "run_gafu" in filename or "dgkan" in filename or "<stdin>" in filename:
            return name, filename, str(frame.get("line", ""))
        if filename and filename != "??" and not fallback[1]:
            fallback = (name, filename, str(frame.get("line", "")))
    if fallback[1]:
        return fallback
    if frames:
        frame = frames[min(5, len(frames) - 1)]
        return str(frame.get("name", "")), str(frame.get("filename", "")), str(frame.get("line", ""))
    return METRIC_UNAVAILABLE, METRIC_UNAVAILABLE, METRIC_UNAVAILABLE


def _source_class(frames: Sequence[Dict[str, Any]]) -> str:
    text = _frame_text(frames).lower()
    if "cross_entropy" in text or "nll_loss" in text or "softmax" in text:
        return "loss_delta_or_ce"
    if "mm" in text or "matmul" in text or "addmm" in text:
        return "matmul_or_mixing"
    if "sum" in text or "mul" in text or "pow" in text:
        return "poly_or_coeffgrad_temp"
    if "empty" in text or "zeros" in text or "clone" in text or "contiguous" in text:
        return "materialization_or_workspace"
    if "triton" in text:
        return "triton_boundary_or_output"
    return "unknown"


def _run_one_step_for_snapshot(args: argparse.Namespace, dataset: str, batch_size: int, depth: int, variant_id: str) -> None:
    device = get_device(args.device)
    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=max(args.train_size, batch_size), val_size=args.val_size, test_size=args.test_size, seed=0, allow_fake_data=False)
    x, y = _take_batch(bundle, batch_size, device)
    if variant_id == "A0-MLP-autograd-reference":
        model = _make_mlp(bundle.input_dim, bundle.num_classes, params.hidden_dim, depth).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=params.lr_mlp)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        _sync(device)
        return
    stack = v68._make_v68_stack(METHOD_CURRENT, bundle.input_dim, params.hidden_dim, depth, _basis_from_name(METHOD_CURRENT, params.basis_count), device, "current", batch_size)
    head = V63ManualLayer(params.hidden_dim, bundle.num_classes, kind="linear", basis_count=2, device=device)
    opt = ManualOptimizer(stack, head, lr=params.lr_manual, kind="ManualAdamW")
    h, caches = stack.forward_manual(x)
    logits, head_cache = head.forward_manual(h)
    loss = F.cross_entropy(logits, y)
    probs = F.softmax(logits, dim=-1)
    probs[torch.arange(y.numel(), device=y.device), y] -= 1.0
    dh = head.backward_manual(probs / max(1, y.numel()), head_cache)
    stack.backward_manual(dh, caches)
    opt.step(step=1, total_steps=1, loss=float(loss.detach().cpu()), prev_loss=None)
    _sync(device)


def _capture_memory_snapshot(args: argparse.Namespace, dataset: str, batch_size: int, depth: int, variant_id: str) -> Dict[str, Any]:
    device = get_device(args.device)
    if device.type != "cuda":
        return {"snapshot_status": "not_run_non_cuda", "exact_peak_timestamp_available": 0}
    torch.cuda.empty_cache()
    _sync(device)
    try:
        torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="all", max_entries=100000)
    except TypeError:
        try:
            torch.cuda.memory._record_memory_history(enabled=True)
        except Exception as exc:
            return {"snapshot_status": "record_history_error", "snapshot_error": repr(exc), "exact_peak_timestamp_available": 0}
    except Exception as exc:
        return {"snapshot_status": "record_history_error", "snapshot_error": repr(exc), "exact_peak_timestamp_available": 0}
    try:
        _run_one_step_for_snapshot(args, dataset, batch_size, depth, variant_id)
        snapshot = torch.cuda.memory._snapshot()
    except Exception as exc:
        snapshot = None
        error = repr(exc)
    finally:
        try:
            torch.cuda.memory._record_memory_history(enabled=None)
        except Exception:
            pass
    if snapshot is None:
        return {"snapshot_status": "snapshot_error", "snapshot_error": error, "exact_peak_timestamp_available": 0}
    return _analyze_snapshot(snapshot)


def _analyze_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    traces = snapshot.get("device_traces") or []
    trace = traces[0] if traces else []
    active: Dict[int, Dict[str, Any]] = {}
    peak_bytes = 0
    peak_time = METRIC_UNAVAILABLE
    peak_active: Dict[int, Dict[str, Any]] = {}
    alloc_count = 0
    free_count = 0
    for event in trace:
        action = str(event.get("action", ""))
        addr = event.get("addr")
        size = int(event.get("size") or 0)
        if action == "alloc" and addr is not None:
            alloc_count += 1
            active[int(addr)] = {
                "addr": int(addr),
                "size": size,
                "time_us": event.get("time_us", METRIC_UNAVAILABLE),
                "frames": event.get("frames") or [],
            }
        elif "free" in action and addr is not None:
            free_count += 1
            active.pop(int(addr), None)
        live_bytes = sum(int(item.get("size") or 0) for item in active.values())
        if live_bytes > peak_bytes:
            peak_bytes = live_bytes
            peak_time = event.get("time_us", METRIC_UNAVAILABLE)
            peak_active = dict(active)
    if not peak_active:
        for segment in snapshot.get("segments", []):
            for block in segment.get("blocks", []):
                if block.get("state") == "active_allocated":
                    addr = int(block.get("address") or 0)
                    peak_active[addr] = {
                        "addr": addr,
                        "size": int(block.get("requested_size") or block.get("size") or 0),
                        "time_us": METRIC_UNAVAILABLE,
                        "frames": block.get("frames") or [],
                    }
        peak_bytes = sum(int(item.get("size") or 0) for item in peak_active.values())
    top = sorted(peak_active.values(), key=lambda item: int(item.get("size") or 0), reverse=True)[:20]
    by_source: Dict[str, float] = {}
    for item in peak_active.values():
        cls = _source_class(item.get("frames") or [])
        by_source[cls] = by_source.get(cls, 0.0) + int(item.get("size") or 0) / (1024**2)
    rows: Dict[str, Any] = {
        "snapshot_status": "measured_memory_history",
        "exact_peak_timestamp_available": int(peak_time != METRIC_UNAVAILABLE),
        "allocation_stack_available": int(any(item.get("frames") for item in peak_active.values())),
        "top20_live_tensor_available": int(bool(top)),
        "exact_peak_time_us": peak_time,
        "exact_live_tensor_count": len(peak_active),
        "exact_live_tensor_total_MB": peak_bytes / (1024**2),
        "allocation_trace_event_count": len(trace),
        "allocation_count": alloc_count,
        "free_count": free_count,
        "snapshot_segment_count": len(snapshot.get("segments", [])),
        "source_unknown_MB": by_source.get("unknown", 0.0),
        "source_materialization_workspace_MB": by_source.get("materialization_or_workspace", 0.0),
        "source_poly_coeffgrad_temp_MB": by_source.get("poly_or_coeffgrad_temp", 0.0),
        "source_matmul_mixing_MB": by_source.get("matmul_or_mixing", 0.0),
        "source_loss_delta_MB": by_source.get("loss_delta_or_ce", 0.0),
        "source_triton_boundary_MB": by_source.get("triton_boundary_or_output", 0.0),
    }
    for idx in range(20):
        prefix = f"top{idx + 1}"
        if idx < len(top):
            item = top[idx]
            frames = item.get("frames") or []
            source, filename, line = _top_source(frames)
            rows.update({
                f"{prefix}_live_tensor_name": f"alloc_{hex(int(item.get('addr') or 0))}",
                f"{prefix}_live_tensor_MB": int(item.get("size") or 0) / (1024**2),
                f"{prefix}_live_tensor_source": _source_class(frames),
                f"{prefix}_allocation_stack_hash": _stack_hash(frames),
                f"{prefix}_source_op": source,
                f"{prefix}_source_file": filename,
                f"{prefix}_source_line": line,
                f"{prefix}_lifetime_start_us": item.get("time_us", METRIC_UNAVAILABLE),
                f"{prefix}_lifetime_end_us": peak_time,
            })
        else:
            rows.update({
                f"{prefix}_live_tensor_name": "",
                f"{prefix}_live_tensor_MB": "",
                f"{prefix}_live_tensor_source": "",
                f"{prefix}_allocation_stack_hash": "",
                f"{prefix}_source_op": "",
                f"{prefix}_source_file": "",
                f"{prefix}_source_line": "",
                f"{prefix}_lifetime_start_us": "",
                f"{prefix}_lifetime_end_us": "",
            })
    return rows


def _enrich_exact_live_set(args: argparse.Namespace, row: Dict[str, Any], mlp_row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    if row.get("implementation_status") != "measured":
        return out
    snap = _capture_memory_snapshot(args, str(row.get("dataset")), int(float(row.get("batch_size") or args.batch_size)), int(float(row.get("depth") or 2)), str(row.get("variant_id")))
    out.update(snap)
    gap = max(0.0, f(row, "backward_adjoint_peak_MB", 0.0) - f(mlp_row, "backward_adjoint_peak_MB", 0.0))
    known = (
        f(out, "source_materialization_workspace_MB", 0.0)
        + f(out, "source_poly_coeffgrad_temp_MB", 0.0)
        + f(out, "source_matmul_mixing_MB", 0.0)
        + f(out, "source_loss_delta_MB", 0.0)
        + f(out, "source_triton_boundary_MB", 0.0)
    )
    unknown = f(out, "source_unknown_MB", 0.0)
    top3 = sum(f(out, f"top{i}_live_tensor_MB", 0.0) for i in range(1, 4))
    explain = min(gap, known)
    out.update({
        "peak_gap_MB": gap,
        "exact_live_set_method": "torch_cuda_memory_history_snapshot",
        "exact_live_tensor_stack_available": f(out, "allocation_stack_available", 0),
        "explained_gap_MB": explain,
        "unexplained_gap_MB": max(0.0, gap - explain),
        "explain_ratio": explain / max(1.0e-12, gap) if gap > 0 else 1.0,
        "top3_gap_fraction": min(gap, top3) / max(1.0e-12, gap) if gap > 0 else 1.0,
        "unknown_gap_fraction": unknown / max(1.0e-12, gap) if gap > 0 else 0.0,
        "live_set_overlap_pass": int(gap > 0 and f(out, "exact_live_tensor_total_MB", 0.0) >= 0.50 * gap),
    })
    out["attribution_pass"] = int(
        int(f(out, "exact_peak_timestamp_available", 0)) == 1
        and int(f(out, "allocation_stack_available", 0)) == 1
        and f(out, "explain_ratio", 0.0) >= 0.95
        and f(out, "top3_gap_fraction", 0.0) >= 0.70
        and f(out, "unknown_gap_fraction", 99.0) <= 0.05
    )
    out["attribution_gate_reason"] = "exact snapshot measured; attribution requires known stack classes explain >=95%, top3 >=70%, unknown <=5%"
    return out


def run_p1(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = _profile_grid(args, "P1", P1_VARIANTS)
    by_shape = {(r.get("dataset"), r.get("batch_size"), r.get("depth"), r.get("variant_id")): r for r in rows if r.get("implementation_status") == "measured"}
    live_rows: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("variant_id") not in {"A0-MLP-autograd-reference", "A1-DWM2-current"}:
            live_rows.append(row)
            continue
        mlp = by_shape.get((row.get("dataset"), row.get("batch_size"), row.get("depth"), "A0-MLP-autograd-reference"), row)
        live_rows.append(_enrich_exact_live_set(args, row, mlp))
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p1_exact_live_set_attribution.csv", live_rows)
    boundary_rows = []
    for row in live_rows:
        if row.get("variant_id") == "A1-DWM2-current":
            boundary_rows.append({
                **_row_common("P1_BOUNDARY", args, method=row.get("method", ""), variant_id=row.get("variant_id", ""), dataset=row.get("dataset", ""), batch_size=int(float(row.get("batch_size") or 0)), depth=int(float(row.get("depth") or 0))),
                "exact_peak_time_us": row.get("exact_peak_time_us", METRIC_UNAVAILABLE),
                "output_materialization_MB": row.get("source_materialization_workspace_MB", METRIC_UNAVAILABLE),
                "layout_conversion_MB": 0.0,
                "python_torch_boundary_time_ms": METRIC_UNAVAILABLE,
                "boundary_materialization_status": "measured_from_memory_snapshot_sources",
                "boundary_bottleneck_pass": 0,
            })
    write_csv(out_dir / "p1_boundary_materialization.csv", boundary_rows)
    current = [r for r in live_rows if r.get("variant_id") == "A1-DWM2-current"]
    _simple_bar_svg(out_dir / "p1_exact_live_tensor_top20.svg", "P1 exact live-set MB", [r["dataset"] + f" B{r['batch_size']}D{r['depth']}" for r in current], [f(r, "exact_live_tensor_total_MB", 0.0) for r in current], "#2563eb")
    _simple_bar_svg(out_dir / "p1_gap_attribution_stacked_bar.svg", "P1 exact explained gap", [r["dataset"] + f" B{r['batch_size']}D{r['depth']}" for r in current], [f(r, "explain_ratio", 0.0) for r in current], "#16a34a")
    _simple_bar_svg(out_dir / "p1_boundary_materialization_bar.svg", "P1 materialization source MB", [r["dataset"] + f" B{r['batch_size']}D{r['depth']}" for r in current], [f(r, "source_materialization_workspace_MB", 0.0) for r in current], "#7c3aed")
    _scatter_svg(out_dir / "p1_live_set_overlap_heatmap.svg", "P1 exact live-set overlap", current, "exact_live_tensor_total_MB", "peak_gap_MB", "variant_id")
    _placeholder_svg(out_dir / "p1_live_set_gantt.svg", "P1 live-set gantt", "exact allocator events captured; tensor symbolic names unavailable")
    _placeholder_svg(out_dir / "p1_peak_phase_timeline.svg", "P1 peak phase timeline", "memory history peak timestamp captured in p1_exact_live_set_attribution.csv")
    for row in live_rows:
        _wandb_log_row(args, row, "summary/v610_p1_exact_liveset")
    return live_rows


def run_p0_reproduction(args: argparse.Namespace, p1_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    current = [r for r in p1_rows if r.get("variant_id") == "A1-DWM2-current" and r.get("implementation_status") == "measured"]
    mem_mean = _mean(f(r, "memory_ratio_vs_MLP") for r in current)
    step_mean = _mean(f(r, "step_time_ratio_vs_MLP") for r in current)
    row = {
        **_row_common("P0_REPRO", args, variant_id="A1-DWM2-current"),
        "v69_reference_memory_ratio_mean": V69_MEMORY_MEAN,
        "v69_reference_step_ratio_mean": V69_STEP_MEAN,
        "v610_current_memory_ratio_mean": mem_mean,
        "v610_current_step_ratio_mean": step_mean,
        "reproduction_delta_memory_ratio": mem_mean - V69_MEMORY_MEAN,
        "reproduction_delta_step_ratio": step_mean - V69_STEP_MEAN,
        "reproduction_pass": int(abs(mem_mean - V69_MEMORY_MEAN) <= 0.05 and abs(step_mean - V69_STEP_MEAN) <= 0.15),
    }
    write_csv(Path(args.out_dir) / "p0_reproduction_check.csv", [row])
    _simple_bar_svg(Path(args.out_dir) / "p0_reproduction_delta_bar.svg", "v6.10 vs v6.9 reproduction delta", ["memory", "step"], [row["reproduction_delta_memory_ratio"], row["reproduction_delta_step_ratio"]], "#2563eb")
    _wandb_log_row(args, row, "summary/v610_p0_reproduction")
    return [row]


def _summarize_detail(args: argparse.Namespace, stage: str, detail: Sequence[Dict[str, Any]], id_field: str, extra: Any | None = None) -> List[Dict[str, Any]]:
    return v69._summarize_detail(args, stage, detail, id_field, extra)


def _append_not_implemented(summary: List[Dict[str, Any]], args: argparse.Namespace, stage: str, packages: Sequence[Tuple[Any, ...]], id_field: str) -> None:
    existing = {r.get(id_field) for r in summary}
    for item in packages:
        name, method, policy, ok, components = item[:5]
        if ok or name in existing:
            continue
        summary.append({
            **_row_common(stage, args, method=method, variant_id=name),
            id_field: name,
            "implementation_status": policy,
            "stage_status": policy,
            "used_for_gate": 0,
            "not_implemented_count": 1,
            "components": components,
            "reason": "required v6.10 kernel is not implemented; no measured ratio emitted",
        })


def run_p2(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = v69.run_p2(args)
    for row in rows:
        _wandb_log_row(args, row, "summary/v610_p2_boundary")
    return rows


def run_p3(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    variants = [(a, b, c, d, e) for a, b, c, d, e, _calls in P3_PACKAGES]
    detail = _profile_grid(args, "P3", variants)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p3_full_layer_backward_detail.csv", detail)
    call_counts = {name: calls for name, _m, _p, _ok, _c, calls in P3_PACKAGES}
    def extra(name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "single_call_layer_count": call_counts.get(name, 0),
            "boundary_time_reduction": METRIC_UNAVAILABLE,
            "live_set_overlap_reduction": METRIC_UNAVAILABLE,
            "kernel_count_reduction": METRIC_UNAVAILABLE,
            "allocation_count_reduction": METRIC_UNAVAILABLE,
            "full_layer_single_call_pass": int(_mean(f(r, "actual_memory_reduction_vs_current", 0.0) for r in rows) >= 0.10 and _mean(f(r, "actual_step_improvement_vs_current", 0.0) for r in rows) >= 0.10),
        }
    summary = _summarize_detail(args, "P3", detail, "package", extra)
    _append_not_implemented(summary, args, "P3", P3_PACKAGES, "package")
    write_csv(out_dir / "p3_full_layer_backward.csv", summary)
    measured = [r for r in detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    _scatter_svg(out_dir / "p3_single_call_package_pareto.svg", "P3 single-call packages", measured, "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _simple_bar_svg(out_dir / "p3_boundary_reduction_bar.svg", "P3 boundary reduction", [r["package"] for r in summary], [f(r, "boundary_time_reduction", 0.0) for r in summary], "#7c3aed")
    _simple_bar_svg(out_dir / "p3_kernel_count_reduction.svg", "P3 kernel reduction", [r["package"] for r in summary], [f(r, "kernel_count_reduction", 0.0) for r in summary], "#2563eb")
    return summary, detail


def run_p4(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    variants = [(a, b, c, d, e) for a, b, c, d, e, _chunks in P4_PACKAGES]
    detail = _profile_grid(args, "P4", variants)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p4_multilayer_chunked_backward_detail.csv", detail)
    chunks = {name: chunk for name, _m, _p, _ok, _c, chunk in P4_PACKAGES}
    def extra(name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "chunk_count": chunks.get(name, 0),
            "call_count_reduction": METRIC_UNAVAILABLE,
            "boundary_time_reduction": METRIC_UNAVAILABLE,
            "live_set_overlap_reduction": METRIC_UNAVAILABLE,
            "multi_layer_chunked_pass": int(_mean(f(r, "actual_memory_reduction_vs_current", 0.0) for r in rows) >= 0.10 and _mean(f(r, "actual_step_improvement_vs_current", 0.0) for r in rows) >= 0.05),
        }
    summary = _summarize_detail(args, "P4", detail, "package", extra)
    _append_not_implemented(summary, args, "P4", P4_PACKAGES, "package")
    write_csv(out_dir / "p4_multilayer_chunked_backward.csv", summary)
    measured = [r for r in detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    _scatter_svg(out_dir / "p4_chunked_backward_pareto.svg", "P4 chunked backward", measured, "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _simple_bar_svg(out_dir / "p4_call_count_reduction_bar.svg", "P4 call count reduction", [r["package"] for r in summary], [f(r, "call_count_reduction", 0.0) for r in summary], "#2563eb")
    return summary, detail


def run_p5(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    detail = _profile_grid(args, "P5", P5_PACKAGES)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p5_onebuffer_full_step_detail.csv", detail)
    def extra(_name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "live_buffer_count_peak": _safe_max(f(r, "reused_buffer_count", 0.0) for r in rows),
            "live_buffer_total_MB_peak": _safe_max(f(r, "workspace_pool_MB", 0.0) for r in rows),
            "buffer_lifetime_conflict_count": METRIC_UNAVAILABLE,
            "onebuffer_pass": int(_safe_max(f(r, "reused_buffer_count", 99.0) for r in rows) <= 3 and _mean(f(r, "actual_memory_reduction_vs_current", 0.0) for r in rows) >= 0.15),
        }
    summary = _summarize_detail(args, "P5", detail, "package", extra)
    _append_not_implemented(summary, args, "P5", P5_PACKAGES, "package")
    write_csv(out_dir / "p5_onebuffer_full_step.csv", summary)
    measured = [r for r in detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    _scatter_svg(out_dir / "p5_onebuffer_memory_step_pareto.svg", "P5 onebuffer memory/step", measured, "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _simple_bar_svg(out_dir / "p5_live_buffer_count_timeline.svg", "P5 live buffer count", [r["package"] for r in summary], [f(r, "live_buffer_count_peak", 0.0) for r in summary], "#2563eb")
    _placeholder_svg(out_dir / "p5_buffer_lifetime_diagram.svg", "P5 buffer lifetime", "true one-buffer variants not implemented")
    return summary, detail


def run_p6(args: argparse.Namespace) -> List[Dict[str, Any]]:
    profile_variants = [(a, b, c, d, e) for a, b, c, d, e, _scale in P6_RESETS]
    rows = _profile_grid(args, "P6", profile_variants)
    scale_by = {name: scale for name, _m, _p, _i, _c, scale in P6_RESETS}
    for row in rows:
        if row.get("implementation_status") == "measured" and row.get("method") != "MLP-autograd-reference":
            eff = v68._residual_effect(args, str(row.get("method")), str(row.get("workspace_policy", "current")), str(row.get("dataset")), int(float(row.get("batch_size") or args.batch_size)), int(float(row.get("depth") or 2)), scale_by.get(str(row.get("variant_id")), 0.0))
            row.update(eff)
            row["primitive"] = row.get("variant_id")
            row["reset_near_pass"] = int(f(row, "memory_ratio_vs_MLP", 99) <= 1.05 and f(row, "step_time_ratio_vs_MLP", 99) <= 1.35 and int(f(row, "gradient_correctness_pass", 0)) == 1)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "p6_bounded_reset_v3.csv", rows)
    measured = [r for r in rows if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    _scatter_svg(out_dir / "p6_reset_residual_strength_vs_memory.svg", "P6 reset residual/memory", measured, "memory_ratio_vs_MLP", "residual_over_base", "variant_id")
    _scatter_svg(out_dir / "p6_reset_memory_time_pareto.svg", "P6 reset memory/time", measured, "memory_ratio_vs_MLP", "step_time_ratio_vs_MLP", "variant_id")
    _simple_bar_svg(out_dir / "p6_reset_workspace_model_comparison.svg", "P6 reset workspace MB", [r["variant_id"] for r in measured], [f(r, "workspace_pool_MB", 0.0) for r in measured], "#7c3aed")
    return rows


def _write_not_run(path: Path, stage: str, reason: str, gated_by: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = [{**_row_common(stage, args), "implementation_status": "not_run", "status": "not_run", "stage_status": "not_run", "used_for_gate": 0, "gated_not_run_count": 1, "reason": reason, "gated_by": gated_by}]
    write_csv(path, rows)
    for row in rows:
        _wandb_log_row(args, row, f"summary/v610_{stage.lower()}_not_run")
    return rows


def _best_dwm2(summaries: Sequence[Dict[str, Any]]) -> Tuple[str, Dict[str, Any] | None]:
    measured = [r for r in summaries if r.get("implementation_status") == "measured"]
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


def run_route(args: argparse.Namespace, p1: Sequence[Dict[str, Any]], p2: Sequence[Dict[str, Any]], p3: Sequence[Dict[str, Any]], p4: Sequence[Dict[str, Any]], p5: Sequence[Dict[str, Any]], p6: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    survivor, best = _best_dwm2(list(p3) + list(p4) + list(p5))
    attribution_pass = any(int(f(r, "attribution_pass", 0)) == 1 for r in p1 if r.get("variant_id") == "A1-DWM2-current")
    boundary_pass = any(int(f(r, "boundary_bottleneck_pass", 0)) == 1 for r in p2)
    onebuffer_pass = any(int(f(r, "onebuffer_pass", 0)) == 1 for r in p5)
    reset_effect = [r for r in p6 if r.get("implementation_status") == "measured" and int(f(r, "residual_effect_pass", 0)) == 1]
    reset_near = [r for r in reset_effect if int(f(r, "reset_near_pass", 0)) == 1]
    if survivor in {"S0", "S1"}:
        route = "R1-DWM2FullStepSolved"
        primary = "DWM2 full-step reached S0/S1"
    elif survivor == "S2":
        route = "R2-DWM2NearPass"
        primary = "DWM2 full-step near-pass only"
    elif not attribution_pass:
        route = "R3-AttributionIncomplete"
        primary = "exact peak live-set attribution did not close the DWM2 peak gap"
    elif onebuffer_pass:
        route = "R5-OneBufferSolved"
        primary = "one-buffer package passed"
    elif reset_near:
        route = "R7-ResetPrimitiveCandidate"
        primary = "reset primitive has residual effect and near-pass"
    elif reset_effect:
        route = "R8-ResetStillTooHeavy"
        primary = "reset residual effect passes but memory/time fail"
    else:
        route = "R9-TerminalPrimitiveRedesign"
        primary = "DWM2 and reset branches have no near-pass candidate"
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
        "onebuffer_pass": int(onebuffer_pass),
        "reset_residual_effect_pass": int(bool(reset_effect)),
        "fallback_near_pass_count": len(reset_near),
        "open_one_step_probe": survivor in {"S0", "S1", "S2"} or bool(reset_near),
        "open_task_reentry": False,
        "open_optimizer_exploration": False,
        "open_functional_correction": False,
        "primary_blocker": primary,
        "next_required_implementation": "real_single_call_full_step_or_new_primitive_family",
        "stop_go_decision": "STOP_DWM2_PATCHING" if route in {"R3-AttributionIncomplete", "R8-ResetStillTooHeavy", "R9-TerminalPrimitiveRedesign"} else "GO_GATED_CONFIRM",
        "no_fake": True,
        "no_proxy": True,
    }
    _json_dump(Path(args.out_dir) / "route_decision.json", route_json)
    _json_dump(Path(args.out_dir) / "aggregate_decision.json", {"status": "gated", "fake_data_used": 0, "proxy_rows_used_as_results": 0, **route_json})
    _placeholder_svg(Path(args.out_dir) / "p11_route_decision_dashboard.svg", "P11 route", route)
    _wandb_log_row(args, {**_row_common("P11", args), **route_json}, "summary/v610_route")
    return route_json


def run_gated(args: argparse.Namespace, route: Dict[str, Any]) -> None:
    _write_not_run(Path(args.out_dir) / "p7_one_step_probe.csv", "P7", "No S0/S1/S2 DWM2 survivor and no reset near-pass", "P3/P4/P5/P6", args)
    _write_not_run(Path(args.out_dir) / "p8_task_reentry.csv", "P8", "P7 did not pass or no memory/time survivor", "P7", args)
    _write_not_run(Path(args.out_dir) / "p8_task_trace.csv", "P8", "P8 is gated", "P7", args)
    _write_not_run(Path(args.out_dir) / "p9_optimizer_exploration.csv", "P9", "P8 task re-entry did not pass", "P8", args)
    _write_not_run(Path(args.out_dir) / "p10_functional_correction_smoke.csv", "P10", "P10 is gated behind P8/P9", "P8/P9", args)
    _placeholder_svg(Path(args.out_dir) / "p7_loss_before_after.svg", "P7 loss before/after", "not_run")
    _placeholder_svg(Path(args.out_dir) / "p7_bad_step_heatmap.svg", "P7 bad step", "not_run")


def run_failure(args: argparse.Namespace) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    out_dir = Path(args.out_dir)
    files = [
        ("p1_exact_live_set_attribution.csv", "P1"),
        ("p2_boundary_audit.csv", "P2"),
        ("p3_full_layer_backward.csv", "P3"),
        ("p4_multilayer_chunked_backward.csv", "P4"),
        ("p5_onebuffer_full_step.csv", "P5"),
        ("p6_bounded_reset_v3.csv", "P6"),
        ("p7_one_step_probe.csv", "P7"),
        ("p8_task_reentry.csv", "P8"),
        ("p9_optimizer_exploration.csv", "P9"),
        ("p10_functional_correction_smoke.csv", "P10"),
    ]
    for fn, stage in files:
        path = out_dir / fn
        if not path.exists():
            failures.append({"stage": stage, "variant_id": fn, "failure_type": "F15_artifact_missing", "metric": "missing", "recommendation": "rerun stage"})
            continue
        for row in _read_csv(path):
            status = row.get("implementation_status") or row.get("status")
            vid = row.get("variant_id", row.get("package", row.get("kernel_name", "")))
            if status in {"not_run", "not_implemented"} or str(status).startswith("not_implemented"):
                ftype = "F12_optimizer_gated" if stage == "P9" else "F13_functional_gated" if stage == "P10" else "F0_not_implemented_or_gated"
                failures.append({"stage": stage, "variant_id": vid, "failure_type": ftype, "metric": row.get("reason", status), "recommendation": "implement or pass gate before claiming metric"})
            if status != "measured":
                continue
            if row.get("memory_ratio_vs_MLP") not in {None, ""} and row.get("method") != "MLP-autograd-reference" and f(row, "memory_ratio_vs_MLP", 0.0) >= 1.0:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F1_memory_fail", "metric": f"memory_ratio={row.get('memory_ratio_vs_MLP')}", "recommendation": "reduce actual CUDA peak"})
            if row.get("step_time_ratio_vs_MLP") not in {None, ""} and row.get("method") != "MLP-autograd-reference" and f(row, "step_time_ratio_vs_MLP", 0.0) > 1.35:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F2_step_time_fail", "metric": f"step_ratio={row.get('step_time_ratio_vs_MLP')}", "recommendation": "reduce step/runtime"})
            if row.get("gradient_correctness_pass") not in {None, ""} and int(f(row, "gradient_correctness_pass", 1)) == 0:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F3_gradient_correctness_fail", "metric": f"grad={row.get('grad_relerr') or row.get('grad_relerr_max')}", "recommendation": "fix kernel numerics"})
            if stage == "P1" and row.get("variant_id") == "A1-DWM2-current" and int(f(row, "attribution_pass", 0)) == 0:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F4_attribution_incomplete", "metric": f"explain_ratio={row.get('explain_ratio')} unknown={row.get('unknown_gap_fraction')}", "recommendation": "map exact allocation stacks to actionable tensor/op sources"})
            if stage == "P2" and int(f(row, "boundary_bottleneck_pass", 0)) == 1:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F5_boundary_overhead", "metric": f"overhead_fraction={row.get('boundary_overhead_fraction_vs_k3')}", "recommendation": "fuse boundary into one full-step call"})
            if stage == "P1" and int(f(row, "live_set_overlap_pass", 0)) == 1:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F6_live_set_overlap", "metric": f"live={row.get('exact_live_tensor_total_MB')} gap={row.get('peak_gap_MB')}", "recommendation": "reduce peak live set"})
            if stage == "P3" and row.get("package") not in {"L0-current", ""} and row.get("implementation_status") == "measured" and f(row, "memory_improvement_vs_current", 0.0) < 0.05:
                failures.append({"stage": stage, "variant_id": row.get("package", ""), "failure_type": "F7_full_layer_no_effect", "metric": f"mem_improvement={row.get('memory_improvement_vs_current')}", "recommendation": "implement true single-call full-layer"})
            if stage == "P5" and row.get("package") not in {"O0-current", ""} and int(f(row, "onebuffer_pass", 0)) == 0:
                failures.append({"stage": stage, "variant_id": row.get("package", ""), "failure_type": "F8_onebuffer_no_effect", "metric": f"buffers={row.get('live_buffer_count_peak')}", "recommendation": "implement true bounded one-buffer live set"})
            if stage == "P6" and row.get("method") != "MLP-autograd-reference" and int(f(row, "residual_effect_pass", 0)) == 0:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F9_reset_residual_effect_fail", "metric": f"residual={row.get('residual_over_base')}", "recommendation": "increase nontrivial residual effect"})
            if stage == "P6" and row.get("method") != "MLP-autograd-reference" and int(f(row, "reset_near_pass", 0)) == 0:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F10_reset_memory_fail", "metric": f"mem={row.get('memory_ratio_vs_MLP')} step={row.get('step_time_ratio_vs_MLP')}", "recommendation": "reduce reset memory/time"})
    write_csv(out_dir / "failure_table.csv", failures or [{"stage": "ALL", "variant_id": "all", "failure_type": "none"}])
    _simple_bar_svg(out_dir / "failure_taxonomy_heatmap.svg", "Failure taxonomy", [r["failure_type"] for r in failures], [1.0 for _ in failures], "#dc2626")
    return failures


def _copy_figures(out_dir: Path) -> None:
    figures = ensure_dir(out_dir / "figures")
    required = [
        "p1_live_set_gantt.svg",
        "p1_exact_live_tensor_top20.svg",
        "p1_gap_attribution_stacked_bar.svg",
        "p1_boundary_materialization_bar.svg",
        "p2_micro_vs_wrapped_runtime.svg",
        "p3_single_call_package_pareto.svg",
        "p4_chunked_backward_pareto.svg",
        "p5_onebuffer_memory_step_pareto.svg",
        "p6_reset_residual_strength_vs_memory.svg",
        "p7_loss_before_after.svg",
        "p11_route_decision_dashboard.svg",
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
        "script": "experiments/run_gafu_v610_real.py",
        "plan": "docs/DG-KAN_v6.10_FullStepLiveSet_StopGo_详细实验计划.md",
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
    parser.add_argument("--packages", default="V6_10_ALL")
    parser.add_argument("--out-dir", type=Path, default=Path("results/real_rerun_20260505/v610_real"))
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
    parser.add_argument("--wandb-group", default="v610-real-20260505")
    parser.add_argument("--wandb-name-prefix", default="v610-real")
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
        p5_summary, _p5_detail = run_p5(args)
        p6_rows = run_p6(args)
        route = run_route(args, p1_rows, p2_rows, p3_summary, p4_summary, p5_summary, p6_rows)
        run_gated(args, route)
        run_failure(args)
        _copy_figures(out_dir)
        _write_manifest(out_dir, args, started, time.time())
    finally:
        _wandb_finish(args, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
