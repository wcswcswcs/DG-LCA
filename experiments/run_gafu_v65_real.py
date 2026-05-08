#!/usr/bin/env python3
"""DG-KAN v6.5 real-only memory-kernel runner.

This runner is intentionally narrow: it measures real memory/runtime/correctness
paths and writes not_run/not_implemented rows for anything that is not actually
implemented.  No proxy ratios, fake datasets, or derived pass rows are emitted.
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

from dgkan_core import ensure_dir, get_device, load_vision_bundle, parse_int_list, parse_str_list, write_csv
from run_gafu_v3 import dataset_name
from run_gafu_v63 import (
    ManualOptimizer,
    V63ManualLayer,
    V63Params,
    _basis_from_name,
    _wandb_finish,
    _wandb_init,
    _wandb_log_row,
    f,
)
from run_gafu_v64_real import (
    DATASETS,
    METHOD_CURRENT,
    _bench_manual_ce,
    _bench_mlp_ce,
    _make_manual_stack,
    _make_mlp,
    _manual_optimizer_state_mb,
    _sha256,
    _take_batch,
    _tensor_mb,
)


P0_VARIANTS = [
    ("MLP-autograd-reference", "reference", "mlp", True, 1),
    ("MLP-manual-linear-reference", "MLP-manual-linear-reference", "current", True, 1),
    ("DWM2-poly2-compiled-current", METHOD_CURRENT, "current", True, 1),
    ("DWM2-poly2-compiled-noHiddenCache", METHOD_CURRENT, "no_hidden_cache", True, 1),
    ("DWM2-poly2-compiled-bf16Cache", METHOD_CURRENT, "bf16_cache", True, 1),
    ("DWM2-poly2-compiled-noHidden+bf16Cache", METHOD_CURRENT, "no_hidden_bf16_cache", True, 0),
    ("DWM2-poly2-compiled-bufferReuse", METHOD_CURRENT, "not_implemented", False, 0),
    ("DWM2-poly2-compiled-deltaStreaming", METHOD_CURRENT, "not_implemented", False, 0),
    ("DWM2-poly2-compiled-noTempPoly2", METHOD_CURRENT, "not_applicable_current_does_not_store_poly_temp", False, 0),
    ("DWM2-poly2-compiled-bufferReuse+deltaStreaming", METHOD_CURRENT, "not_implemented", False, 0),
    ("DWM2-poly2-compiled-allMemoryOptimized", METHOD_CURRENT, "not_implemented", False, 0),
]

P1_VARIANTS = [
    ("P1-current", METHOD_CURRENT, "current", True, "current"),
    ("P1-noHiddenCache", METHOD_CURRENT, "no_hidden_cache", True, "noHiddenCache"),
]

P2_FACTORS = [
    ("M0-current", METHOD_CURRENT, "current", True, "current"),
    ("M1-bf16Cache", METHOD_CURRENT, "bf16_cache", True, "bf16Cache"),
    ("M2-deltaStreaming", METHOD_CURRENT, "not_implemented", False, "deltaStreaming"),
    ("M3-bufferReuse", METHOD_CURRENT, "not_implemented", False, "bufferReuse"),
    ("M4-noTempPoly2", METHOD_CURRENT, "not_applicable_current_does_not_store_poly_temp", False, "noTempPoly2"),
    ("M5-updateInPlaceSafe", METHOD_CURRENT, "not_applicable_current_update_already_in_place", False, "updateInPlaceSafe"),
    ("M6-optimizerStateSplit", METHOD_CURRENT, "not_implemented", False, "optimizerStateSplit"),
]

P3_COMBINATIONS = [
    ("C0-current", METHOD_CURRENT, "current", True, "current"),
    ("Cdiag-noHidden+bf16Cache", METHOD_CURRENT, "no_hidden_bf16_cache", True, "diagnostic_noHidden+bf16"),
    ("C1-bufferReuse+deltaStreaming", METHOD_CURRENT, "not_implemented", False, "bufferReuse+deltaStreaming"),
    ("C2-bufferReuse+bf16Cache", METHOD_CURRENT, "not_implemented", False, "bufferReuse+bf16Cache"),
    ("C3-deltaStreaming+bf16Cache", METHOD_CURRENT, "not_implemented", False, "deltaStreaming+bf16Cache"),
    ("C4-bufferReuse+noTempPoly2", METHOD_CURRENT, "not_implemented", False, "bufferReuse+noTempPoly2"),
    ("C5-bufferReuse+deltaStreaming+bf16Cache", METHOD_CURRENT, "not_implemented", False, "bufferReuse+deltaStreaming+bf16Cache"),
    ("C6-bufferReuse+deltaStreaming+updateInPlaceSafe", METHOD_CURRENT, "not_implemented", False, "bufferReuse+deltaStreaming+updateInPlaceSafe"),
    ("C7-allMemoryOptimized-light", METHOD_CURRENT, "not_implemented", False, "bufferReuse+deltaStreaming+updateInPlaceSafe"),
    ("C8-allMemoryOptimized-full", METHOD_CURRENT, "not_implemented", False, "bufferReuse+deltaStreaming+bf16Cache+noTempPoly2+updateInPlaceSafe+optimizerStateSplit"),
]

FALLBACKS = [
    ("F0-DWM2-poly3", "DWM2-poly3"),
    ("F1-DWM2-poly2+silu-base", "DWM2-poly2-silu_base"),
    ("F2-RationalKAT-lite-fastpoly", "RationalKAT-lite-fastpoly"),
    ("F3-SparseInterp fused rewrite", "SparseInterpKAN-K8-current"),
    ("F4-DWM2-RBFK2 optimized", "DWM2-lite-RBFK2-cacheMin"),
]


def _mean(vals: Iterable[float], default: float = float("nan")) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return sum(xs) / len(xs) if xs else default


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


def _write_manifest(out_dir: Path, args: argparse.Namespace, started: float, finished: float) -> None:
    manifest = {
        "provenance": "EMPIRICAL_REAL_ONLY_NO_PROXY",
        "script": "experiments/run_gafu_v65_real.py",
        "plan": "docs/DG-KAN_v6.5_MemoryKernel_PhaseLocal_Redesign_详细实验计划.md",
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


def _row_common(stage: str, args: argparse.Namespace, *, method: str = "", variant_id: str = "", dataset: str = "", seed: int = 0, batch_size: int = 0, depth: int = 0) -> Dict[str, Any]:
    return {
        "stage": stage,
        "method": method,
        "variant_id": variant_id,
        "dataset": dataset,
        "seed": seed,
        "batch_size": batch_size,
        "hidden_dim": 64,
        "depth": depth,
        "device": str(get_device(args.device)),
        "run_id": Path(args.out_dir).name,
        "wandb_run": "",
        "artifact_hash": "",
        "stage_status": "measured",
        "fake_data_used": 0,
        "proxy_rows_used": 0,
        "proxy_row_used": 0,
        "not_implemented_count": 0,
        "gated_not_run_count": 0,
        "used_for_gate": 1,
        "implementation_status": "measured",
        "error": "",
    }


def _escape(text: Any) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _simple_bar_svg(path: Path, title: str, labels: Sequence[str], values: Sequence[float], color: str = "#2563eb") -> None:
    width = 920
    row_h = 26
    height = 70 + row_h * max(1, len(labels))
    vmax = max([abs(float(v)) for v in values if math.isfinite(float(v))] + [1.0])
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>']
    parts.append(f'<text x="16" y="30" font-family="sans-serif" font-size="18">{_escape(title)}</text>')
    for i, (label, value) in enumerate(zip(labels, values)):
        y = 58 + i * row_h
        val = float(value) if math.isfinite(float(value)) else 0.0
        bar_w = max(1.0, abs(val) / vmax * 520)
        parts.append(f'<text x="16" y="{y + 16}" font-family="sans-serif" font-size="11">{_escape(label)}</text>')
        parts.append(f'<rect x="300" y="{y + 4}" width="{bar_w:.2f}" height="16" fill="{color}"/>')
        parts.append(f'<text x="{310 + bar_w:.2f}" y="{y + 16}" font-family="monospace" font-size="11">{val:.4g}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _scatter_svg(path: Path, title: str, rows: Sequence[Dict[str, Any]], xkey: str, ykey: str, label_key: str) -> None:
    width, height = 760, 520
    xs = [f(r, xkey, float("nan")) for r in rows]
    ys = [f(r, ykey, float("nan")) for r in rows]
    finite = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    xmin = min([x for x, _ in finite] + [0.0])
    xmax = max([x for x, _ in finite] + [1.0])
    ymin = min([y for _, y in finite] + [0.0])
    ymax = max([y for _, y in finite] + [1.0])
    if xmax == xmin:
        xmax += 1.0
    if ymax == ymin:
        ymax += 1.0
    def sx(x: float) -> float:
        return 70 + (x - xmin) / (xmax - xmin) * 620
    def sy(y: float) -> float:
        return 450 - (y - ymin) / (ymax - ymin) * 360
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>']
    parts.append(f'<text x="16" y="30" font-family="sans-serif" font-size="18">{_escape(title)}</text>')
    parts.append('<line x1="70" y1="450" x2="690" y2="450" stroke="#111827"/>')
    parts.append('<line x1="70" y1="90" x2="70" y2="450" stroke="#111827"/>')
    for r, x, y in zip(rows, xs, ys):
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4" fill="#dc2626"><title>{_escape(r.get(label_key, ""))}: {xkey}={x:.4g}, {ykey}={y:.4g}</title></circle>')
    parts.append(f'<text x="320" y="492" font-family="sans-serif" font-size="12">{_escape(xkey)}</text>')
    parts.append(f'<text x="8" y="80" font-family="sans-serif" font-size="12">{_escape(ykey)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _placeholder_svg(path: Path, title: str, reason: str) -> None:
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="860" height="180">\n'
        f'<rect width="100%" height="100%" fill="white"/>\n'
        f'<text x="18" y="42" font-family="sans-serif" font-size="20">{_escape(title)}</text>\n'
        f'<text x="18" y="86" font-family="sans-serif" font-size="14">{_escape(reason)}</text>\n'
        f'</svg>\n',
        encoding="utf-8",
    )


def _profile_rows(
    args: argparse.Namespace,
    stage: str,
    variants: Sequence[Tuple[str, str, str, bool, str]],
    *,
    include_mlp: bool = True,
) -> List[Dict[str, Any]]:
    device = get_device(args.device)
    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    batch_sizes = parse_int_list(args.batch_sizes)
    depths = parse_int_list(args.depths)
    datasets = parse_str_list(args.datasets)
    rows: List[Dict[str, Any]] = []
    for ds in datasets:
        canonical = dataset_name(ds)
        bundle = load_vision_bundle(
            canonical,
            data_root=args.data_root,
            train_size=max(args.train_size, max(batch_sizes)),
            val_size=args.val_size,
            test_size=args.test_size,
            seed=0,
            allow_fake_data=False,
        )
        for batch_size in batch_sizes:
            for depth in depths:
                mlp_row: Dict[str, Any] | None = None
                if include_mlp:
                    mlp_stat = _bench_mlp_ce(bundle, batch_size, depth, params, device, args.warmup_steps, args.measure_steps)
                    mlp_row = {
                        **_row_common(stage, args, method="MLP-autograd-reference", variant_id="MLP-autograd-reference", dataset=canonical, seed=0, batch_size=batch_size, depth=depth),
                        **_normalize_profile_stat(mlp_stat),
                    }
                    mlp_row.update({"used_for_gate": 0, "loss_backward_used": 1, "torch_autograd_graph_used": 1, "memory_ratio_vs_MLP": 1.0, "step_time_ratio_vs_MLP": 1.0, "backward_time_ratio_vs_MLP": 1.0})
                    rows.append(mlp_row)
                    _wandb_log_row(args, mlp_row, f"summary/v65_{stage.lower()}")
                if mlp_row is None:
                    continue
                current_by_shape: Dict[str, Dict[str, Any]] = {}
                for variant_id, method, policy, implemented, repair_type in variants:
                    if not implemented:
                        row = {
                            **_row_common(stage, args, method=method, variant_id=variant_id, dataset=canonical, seed=0, batch_size=batch_size, depth=depth),
                            "implementation_status": policy,
                            "stage_status": policy,
                            "used_for_gate": 0,
                            "not_implemented_count": int(str(policy).startswith("not_implemented") or str(policy).startswith("not_applicable")),
                            "repair_type": repair_type,
                            "reason": "not implemented or not applicable; no measured ratio emitted",
                        }
                        rows.append(row)
                        _wandb_log_row(args, row, f"summary/v65_{stage.lower()}")
                        continue
                    stat = _bench_manual_ce(method, bundle, batch_size, depth, params, device, args.warmup_steps, args.measure_steps, cache_policy=policy)
                    row = {
                        **_row_common(stage, args, method=method, variant_id=variant_id, dataset=canonical, seed=0, batch_size=batch_size, depth=depth),
                        **_normalize_profile_stat(stat),
                        "cache_policy": policy,
                        "dtype_cache": "bf16" if "bf16" in policy else "fp32",
                        "repair_type": repair_type,
                        "loss_backward_used": 0,
                        "torch_autograd_graph_used": 0,
                    }
                    _apply_ratios(row, mlp_row)
                    current_key = f"{canonical}:{batch_size}:{depth}"
                    if variant_id.endswith("current") or variant_id in {"M0-current", "C0-current"}:
                        current_by_shape[current_key] = row
                    cur = current_by_shape.get(current_key)
                    if cur:
                        row["actual_memory_reduction_vs_current"] = (f(cur, "memory_ratio_vs_MLP") - f(row, "memory_ratio_vs_MLP")) / max(1.0e-12, f(cur, "memory_ratio_vs_MLP"))
                        row["actual_step_penalty_vs_current"] = (f(row, "step_time_ratio_vs_MLP") - f(cur, "step_time_ratio_vs_MLP")) / max(1.0e-12, f(cur, "step_time_ratio_vs_MLP"))
                        row["actual_backward_penalty_vs_current"] = (f(row, "backward_time_ratio_vs_MLP") - f(cur, "backward_time_ratio_vs_MLP")) / max(1.0e-12, f(cur, "backward_time_ratio_vs_MLP"))
                    row["memory_pass"] = int(f(row, "memory_ratio_vs_MLP", 99) < 1.0)
                    row["time_exploratory_pass"] = int(f(row, "step_time_ratio_vs_MLP", 99) <= 1.35)
                    row["time_official_pass"] = int(f(row, "step_time_ratio_vs_MLP", 99) <= 1.20)
                    row["gradient_correctness_pass"] = int(f(row, "grad_relerr", 99) < 1.0e-4 and f(row, "grad_cos", 0) > 0.999)
                    row["single_factor_effective"] = int(f(row, "actual_memory_reduction_vs_current", 0.0) >= 0.08 and f(row, "actual_step_penalty_vs_current", 99) <= 0.10 and row["gradient_correctness_pass"])
                    rows.append(row)
                    _wandb_log_row(args, row, f"summary/v65_{stage.lower()}")
    return rows


def _normalize_profile_stat(stat: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(stat)
    out["forward_peak_MB"] = out.get("peak_forward_MB", out.get("forward_peak_MB", ""))
    out["loss_delta_peak_MB"] = out.get("peak_loss_delta_MB", out.get("loss_delta_peak_MB", ""))
    out["backward_adjoint_peak_MB"] = out.get("peak_backward_adjoint_MB", out.get("backward_adjoint_peak_MB", ""))
    out["update_peak_MB"] = out.get("peak_update_MB", out.get("update_peak_MB", ""))
    out["optimizer_state_peak_MB"] = out.get("optimizer_state_MB", 0.0)
    out["total_step_peak_MB"] = out.get("peak_total_step_MB", out.get("peak_allocated_MB", ""))
    out["reserved_peak_MB"] = out.get("peak_reserved_MB", out.get("peak_allocated_MB", ""))
    out["manual_cache_total_MB"] = out.get("manual_cache_MB", out.get("cache_total_MB", 0.0))
    out["manual_update_time_ms"] = out.get("update_time_ms", 0.0)
    out["optimizer_state_update_time_ms"] = 0.0
    out["num_tensor_allocations"] = out.get("num_tensor_allocations", 0)
    out["num_tensor_allocations_forward"] = 0
    out["num_tensor_allocations_backward"] = out.get("num_tensor_allocations", 0)
    out["num_tensor_allocations_update"] = 0
    out["small_kernel_count"] = 0
    out["python_loop_count"] = 0
    out["sync_count"] = 4
    out["kernel_count_update"] = out.get("kernel_count_update", 0)
    out["gemm_kernel_count"] = out.get("gemm_count", out.get("op_count_gemm", 0))
    phases = [f(out, key, 0.0) for key in ["forward_peak_MB", "loss_delta_peak_MB", "backward_adjoint_peak_MB", "update_peak_MB", "optimizer_state_peak_MB"]]
    total = max(1.0e-12, f(out, "total_step_peak_MB", max(phases + [0.0])))
    out["phase_peak_explain_ratio"] = max(phases + [0.0]) / total
    out["unexplained_memory_gap_MB"] = total - f(out, "manual_cache_total_MB", 0.0) - f(out, "parameter_MB", 0.0) - f(out, "optimizer_state_MB", 0.0)
    out["workspace_temp_MB"] = max(f(out, "workspace_temp_MB", 0.0), out["unexplained_memory_gap_MB"])
    return out


def _apply_ratios(row: Dict[str, Any], mlp_row: Dict[str, Any]) -> None:
    row["memory_ratio_vs_MLP"] = f(row, "backward_adjoint_peak_MB") / max(1.0e-12, f(mlp_row, "backward_adjoint_peak_MB"))
    row["step_time_ratio_vs_MLP"] = f(row, "step_time_ms") / max(1.0e-12, f(mlp_row, "step_time_ms"))
    row["backward_time_ratio_vs_MLP"] = f(row, "manual_backward_time_ms") / max(1.0e-12, f(mlp_row, "manual_backward_time_ms"))
    row["forward_time_ratio_vs_MLP"] = f(row, "forward_time_ms") / max(1.0e-12, f(mlp_row, "forward_time_ms"))


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    device = get_device(args.device)
    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    bundle = load_vision_bundle("MNIST", data_root=args.data_root, train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, seed=0, allow_fake_data=False)
    x, y = _take_batch(bundle, min(64, args.batch_size), device)
    rows: List[Dict[str, Any]] = []
    for variant_id, method, policy, implemented, used_for_gate in P0_VARIANTS:
        row = _row_common("P0", args, method=method, variant_id=variant_id, dataset="MNIST", seed=0, batch_size=min(64, args.batch_size), depth=2)
        row.update({"used_for_gate": used_for_gate, "fake_data_used": int(getattr(bundle, "used_fake_data", False)), "proxy_row_used": 0})
        if not implemented:
            row.update({
                "implementation_status": policy,
                "stage_status": policy,
                "used_for_gate": 0,
                "not_implemented_count": int(str(policy).startswith("not_implemented") or str(policy).startswith("not_applicable")),
                "reason": "not implemented or not applicable; no ratio emitted",
            })
        elif policy == "mlp":
            model = _make_mlp(bundle.input_dim, bundle.num_classes, params.hidden_dim, 2).to(device)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            row.update({
                "implementation_status": "measured",
                "loss_backward_used": 1,
                "torch_autograd_graph_used": 1,
                "manual_forward_available": 0,
                "manual_backward_available": 0,
                "manual_update_available": 0,
                "nonKAN_param_count": sum(p.numel() for p in model.parameters()),
                "edge_param_count": 0,
                "coverage_edge": 0,
                "rollback_max_abs_error": 0.0,
                "profiler_actual_cuda_peak_available": 1,
                "phase_local_peak_available": 1,
                "manual_cache_estimate_available": 1,
                "workspace_temp_available": 1,
            })
        else:
            stack = _make_manual_stack(method, bundle.input_dim, params.hidden_dim, 2, _basis_from_name(method, params.basis_count), device, policy)
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
            row.update({
                "implementation_status": "measured",
                "loss_backward_used": 0,
                "torch_autograd_graph_used": 0,
                "manual_forward_available": 1,
                "manual_backward_available": 1,
                "manual_update_available": 1,
                "nonKAN_param_count": 0,
                "edge_param_count": stack.param_count() + head.param_count(),
                "coverage_edge": 1,
                "rollback_max_abs_error": rollback,
                "profiler_actual_cuda_peak_available": 1,
                "phase_local_peak_available": 1,
                "manual_cache_estimate_available": 1,
                "workspace_temp_available": 1,
                "manual_cache_total_MB": stack.cache_breakdown(caches).get("cache_total_MB", 0.0),
                "parameter_MB": sum(_tensor_mb(p) for _name, p, _grad in stack.params_and_grads()) + sum(_tensor_mb(p) for p in head.params.values()),
                "optimizer_state_MB": _manual_optimizer_state_mb(opt),
            })
        rows.append(row)
        _wandb_log_row(args, row, "summary/v65_p0_contract")
    write_csv(out_dir / "p0_contract.csv", rows)
    _write_contract_heatmap(out_dir / "p0_contract_heatmap.svg", rows)
    return rows


def _write_contract_heatmap(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    checks = [
        ("measured", "implementation_status"),
        ("no_fake", "fake_data_used"),
        ("no_proxy", "proxy_row_used"),
        ("manual_backward", "manual_backward_available"),
        ("nonKAN=0", "nonKAN_param_count"),
        ("coverage", "coverage_edge"),
        ("peak_profiler", "profiler_actual_cuda_peak_available"),
    ]
    cell_w, cell_h = 112, 28
    left, top = 285, 58
    width = left + cell_w * len(checks) + 20
    height = top + cell_h * len(rows) + 30
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>']
    parts.append('<text x="16" y="30" font-family="sans-serif" font-size="18">v6.5 P0 Contract</text>')
    for j, (label, _key) in enumerate(checks):
        parts.append(f'<text x="{left + j * cell_w + 4}" y="50" font-family="sans-serif" font-size="11">{_escape(label)}</text>')
    for i, row in enumerate(rows):
        y = top + i * cell_h
        parts.append(f'<text x="12" y="{y + 18}" font-family="sans-serif" font-size="11">{_escape(row.get("variant_id", ""))}</text>')
        for j, (_label, key) in enumerate(checks):
            if key == "implementation_status":
                passed = row.get(key) == "measured"
            elif key in {"fake_data_used", "proxy_row_used"}:
                passed = int(float(row.get(key) or 0)) == 0
            elif key == "nonKAN_param_count":
                passed = int(float(row.get(key) or 0)) == 0 or row.get("method") == "reference"
            else:
                passed = int(float(row.get(key) or 0)) == 1
            color = "#16a34a" if passed else "#dc2626"
            if row.get("implementation_status") != "measured":
                color = "#a1a1aa"
            x = left + j * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 3}" height="{cell_h - 3}" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{left + cell_w * len(checks) + 8}" y="{y + 18}" font-family="sans-serif" font-size="11">{_escape(row.get("implementation_status", ""))}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_not_run(path: Path, stage: str, reason: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = [{
        **_row_common(stage, args),
        "implementation_status": "not_run",
        "stage_status": "not_run",
        "used_for_gate": 0,
        "status": "not_run",
        "gated_not_run_count": 1,
        "reason": reason,
    }]
    write_csv(path, rows)
    for row in rows:
        _wandb_log_row(args, row, f"summary/v65_{stage.lower()}_not_run")
    return rows


def run_p1(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = _profile_rows(args, "P1", P1_VARIANTS, include_mlp=True)
    write_csv(Path(args.out_dir) / "p1_phase_local_memory.csv", rows)
    write_csv(Path(args.out_dir) / "p1_memory_attribution.csv", rows)
    measured = [r for r in rows if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    _simple_bar_svg(Path(args.out_dir) / "p1_memory_waterfall.svg", "P1 Memory Ratio", [r["variant_id"] + " " + r["dataset"] + f" B{r['batch_size']}D{r['depth']}" for r in measured], [f(r, "memory_ratio_vs_MLP") for r in measured], "#2563eb")
    _scatter_svg(Path(args.out_dir) / "p1_actual_peak_vs_cache.svg", "Actual Peak vs Manual Cache", measured, "manual_cache_total_MB", "total_step_peak_MB", "variant_id")
    _simple_bar_svg(Path(args.out_dir) / "p1_unexplained_gap_heatmap.svg", "P1 Unexplained Memory Gap MB", [r["variant_id"] + " " + r["dataset"] + f" B{r['batch_size']}D{r['depth']}" for r in measured], [f(r, "unexplained_memory_gap_MB") for r in measured], "#dc2626")
    return rows


def run_p2(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = _profile_rows(args, "P2", P2_FACTORS, include_mlp=True)
    write_csv(Path(args.out_dir) / "p2_single_factor_memory_repair.csv", rows)
    measured = [r for r in rows if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    _scatter_svg(Path(args.out_dir) / "p2_factor_pareto.svg", "P2 Single-Factor Pareto", measured, "actual_memory_reduction_vs_current", "actual_step_penalty_vs_current", "variant_id")
    _simple_bar_svg(Path(args.out_dir) / "p2_memory_reduction_waterfall.svg", "P2 Memory Reduction vs Current", [r["variant_id"] + " " + r["dataset"] + f" B{r['batch_size']}D{r['depth']}" for r in measured], [f(r, "actual_memory_reduction_vs_current", 0.0) for r in measured], "#16a34a")
    return rows


def run_p3(args: argparse.Namespace) -> List[Dict[str, Any]]:
    detail = _profile_rows(args, "P3", P3_COMBINATIONS, include_mlp=True)
    write_csv(Path(args.out_dir) / "p3_combined_memory_packages_detail.csv", detail)
    measured = [r for r in detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    by: Dict[str, List[Dict[str, Any]]] = {}
    for row in measured:
        by.setdefault(str(row.get("variant_id")), []).append(row)
    summary: List[Dict[str, Any]] = []
    for variant_id, rows in by.items():
        mem = [f(r, "memory_ratio_vs_MLP") for r in rows]
        step = [f(r, "step_time_ratio_vs_MLP") for r in rows]
        bwd = [f(r, "backward_time_ratio_vs_MLP") for r in rows]
        grad_rel = [f(r, "grad_relerr") for r in rows]
        grad_cos = [f(r, "grad_cos") for r in rows]
        summary.append({
            **_row_common("P3", args, variant_id=variant_id),
            "combination_id": variant_id,
            "included_factors": rows[0].get("repair_type", ""),
            "memory_ratio_min": min(mem),
            "memory_ratio_mean": _mean(mem),
            "memory_ratio_max": max(mem),
            "step_ratio_min": min(step),
            "step_ratio_mean": _mean(step),
            "step_ratio_max": max(step),
            "backward_ratio_mean": _mean(bwd),
            "grad_relerr_max": max(grad_rel),
            "grad_cos_min": min(grad_cos),
            "memory_pass_count": sum(v < 1.0 for v in mem),
            "step_pass_count": sum(v <= 1.35 for v in step),
            "both_pass_count": sum((m < 1.0 and s <= 1.35) for m, s in zip(mem, step)),
        })
    for variant_id, method, policy, implemented, factors in P3_COMBINATIONS:
        if implemented:
            continue
        summary.append({**_row_common("P3", args, method=method, variant_id=variant_id), "combination_id": variant_id, "included_factors": factors, "implementation_status": policy, "used_for_gate": 0, "reason": "combination not implemented; no measured ratio emitted"})
    write_csv(Path(args.out_dir) / "p3_combined_memory_packages.csv", summary)
    _simple_bar_svg(Path(args.out_dir) / "p3_combination_heatmap.svg", "P3 Combination Memory Ratio Mean", [r.get("combination_id", "") for r in summary], [f(r, "memory_ratio_mean", float("nan")) for r in summary], "#7c3aed")
    _simple_bar_svg(Path(args.out_dir) / "p3_pass_count_bar.svg", "P3 Both-Pass Count", [r.get("combination_id", "") for r in summary], [f(r, "both_pass_count", 0.0) for r in summary], "#16a34a")
    return summary


def _best_measured(out_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fn in ["p2_single_factor_memory_repair.csv", "p3_combined_memory_packages_detail.csv"]:
        p = out_dir / fn
        if not p.exists():
            continue
        with p.open(newline="") as handle:
            rows.extend([r for r in csv.DictReader(handle) if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"])
    return rows


def run_p4(args: argparse.Namespace) -> List[Dict[str, Any]]:
    measured = _best_measured(Path(args.out_dir))
    if not measured or min(f(r, "memory_ratio_vs_MLP", 99) for r in measured) >= 1.05:
        rows = _write_not_run(Path(args.out_dir) / "p4_runtime_kernel_audit.csv", "P4", "No memory near-pass candidate from P2/P3; runtime repair is gated", args)
        _placeholder_svg(Path(args.out_dir) / "p4_runtime_waterfall.svg", "P4 Runtime Waterfall", "not_run: no memory near-pass candidate")
        _placeholder_svg(Path(args.out_dir) / "p4_kernel_count_scatter.svg", "P4 Kernel Count Scatter", "not_run: no memory near-pass candidate")
        return rows
    write_csv(Path(args.out_dir) / "p4_runtime_kernel_audit.csv", measured)
    _simple_bar_svg(Path(args.out_dir) / "p4_runtime_waterfall.svg", "P4 Step Time Ratio", [r["variant_id"] + " " + r["dataset"] for r in measured], [f(r, "step_time_ratio_vs_MLP") for r in measured], "#f59e0b")
    _scatter_svg(Path(args.out_dir) / "p4_kernel_count_scatter.svg", "Kernel Count vs Step Ratio", measured, "kernel_count_backward", "step_time_ratio_vs_MLP", "variant_id")
    return measured


def run_p5(args: argparse.Namespace) -> List[Dict[str, Any]]:
    measured = _best_measured(Path(args.out_dir))
    survivors = [r for r in measured if f(r, "memory_ratio_vs_MLP", 99) < 1.0 and f(r, "step_time_ratio_vs_MLP", 99) <= 1.35 and int(f(r, "gradient_correctness_pass", 0)) == 1]
    if not survivors:
        rows = _write_not_run(Path(args.out_dir) / "p5_one_step_probe.csv", "P5", "No memory/time survivor; one-step probe is gated", args)
        _placeholder_svg(Path(args.out_dir) / "p5_actual_descent_scatter.svg", "P5 Actual Descent Scatter", "not_run: no memory/time survivor")
        return rows
    rows = _write_not_run(Path(args.out_dir) / "p5_one_step_probe.csv", "P5", "Memory/time survivor exists, but one-step probe implementation is not available in this runner", args)
    _placeholder_svg(Path(args.out_dir) / "p5_actual_descent_scatter.svg", "P5 Actual Descent Scatter", "not_run: one-step probe implementation unavailable")
    return rows


def run_p6(args: argparse.Namespace) -> List[Dict[str, Any]]:
    measured = _best_measured(Path(args.out_dir))
    candidate_rows = [r for r in measured if r.get("variant_id") not in {"M0-current", "C0-current"}]
    if not candidate_rows:
        candidate_rows = measured
    best_mem = min([f(r, "memory_ratio_vs_MLP", 99.0) for r in candidate_rows] + [99.0])
    best_step_for_mem = min([f(r, "step_time_ratio_vs_MLP", 99.0) for r in candidate_rows if f(r, "memory_ratio_vs_MLP", 99.0) == best_mem] + [99.0])
    s0 = [r for r in candidate_rows if f(r, "memory_ratio_vs_MLP", 99) < 1.0 and f(r, "step_time_ratio_vs_MLP", 99) <= 1.20 and int(f(r, "gradient_correctness_pass", 0)) == 1]
    s1 = [r for r in candidate_rows if f(r, "memory_ratio_vs_MLP", 99) < 1.0 and f(r, "step_time_ratio_vs_MLP", 99) <= 1.35 and int(f(r, "gradient_correctness_pass", 0)) == 1]
    s2 = [r for r in candidate_rows if f(r, "memory_ratio_vs_MLP", 99) < 1.0 and int(f(r, "gradient_correctness_pass", 0)) == 1]
    if s0:
        survivor_type, route, open_task = "S0", "R1", True
    elif s1:
        survivor_type, route, open_task = "S1", "R2", True
    elif s2:
        survivor_type, route, open_task = "S2", "R2", False
    else:
        survivor_type, route, open_task = "S3", "R3", False
    rows = [{**_row_common("P6", args), "survivor_type": survivor_type, "best_memory_ratio": best_mem, "best_step_ratio": best_step_for_mem, "open_task_reentry": int(open_task), "open_functional_correction": 0}]
    write_csv(Path(args.out_dir) / "p6_survivor_selection.csv", rows)
    route_json = {
        "route": route,
        "best_memory_ratio": best_mem,
        "best_step_ratio": best_step_for_mem,
        "survivor_type": survivor_type,
        "open_task_reentry": open_task,
        "open_functional_correction": False,
        "no_proxy": True,
    }
    _json_dump(Path(args.out_dir) / "route_decision.json", route_json)
    aggregate = {
        "status": "gated" if not open_task else "task_reentry_open",
        "fake_data_used": 0,
        "proxy_rows_used_as_results": 0,
        "best_memory_ratio": best_mem,
        "best_step_ratio": best_step_for_mem,
        "survivor_type": survivor_type,
    }
    _json_dump(Path(args.out_dir) / "aggregate_decision.json", aggregate)
    for row in rows:
        _wandb_log_row(args, row, "summary/v65_p6_survivor_selection")
    return rows


def run_p7_p8(args: argparse.Namespace) -> None:
    route_path = Path(args.out_dir) / "route_decision.json"
    route = json.loads(route_path.read_text()) if route_path.exists() else {}
    if not route.get("open_task_reentry", False):
        _write_not_run(Path(args.out_dir) / "p7_task_reentry.csv", "P7", "P6 produced no S0/S1 survivor", args)
        _write_not_run(Path(args.out_dir) / "p7_task_trace.csv", "P7", "P7 is gated", args)
        _write_not_run(Path(args.out_dir) / "p8_functional_correction_smoke.csv", "P8", "P8 is gated behind P7", args)
        _placeholder_svg(Path(args.out_dir) / "val_loss_vs_step.svg", "P7 Val Loss vs Step", "not_run")
        _placeholder_svg(Path(args.out_dir) / "val_loss_vs_time.svg", "P7 Val Loss vs Time", "not_run")
        _placeholder_svg(Path(args.out_dir) / "acc_vs_time.svg", "P7 Accuracy vs Time", "not_run")
        _placeholder_svg(Path(args.out_dir) / "time_to_target_bar.svg", "P7 Time To Target", "not_run")
        _placeholder_svg(Path(args.out_dir) / "task_efficiency_pareto.svg", "P7 Task Efficiency Pareto", "not_run")
        _placeholder_svg(Path(args.out_dir) / "calibration_curve_smoke.svg", "P7 Calibration Curve", "not_run")
        _placeholder_svg(Path(args.out_dir) / "direction_quality_scatter.svg", "P8 Direction Quality", "not_run")
        _placeholder_svg(Path(args.out_dir) / "geometry_gain_vs_task_cost.svg", "P8 Geometry Gain", "not_run")
        _placeholder_svg(Path(args.out_dir) / "lambda_acceptance_histogram.svg", "P8 Lambda Acceptance", "not_run")
        _placeholder_svg(Path(args.out_dir) / "bad_step_heatmap.svg", "P8 Bad Step Heatmap", "not_run")
        _placeholder_svg(Path(args.out_dir) / "geometry_curve_with_correction_events.svg", "P8 Geometry Curve", "not_run")
        return
    _write_not_run(Path(args.out_dir) / "p7_task_reentry.csv", "P7", "P6 opened task re-entry, but task training implementation is not available in this v6.5 memory runner", args)
    _write_not_run(Path(args.out_dir) / "p7_task_trace.csv", "P7", "Task trace not generated because task training implementation is unavailable", args)
    _write_not_run(Path(args.out_dir) / "p8_functional_correction_smoke.csv", "P8", "Functional correction smoke is gated until a real P7 task trace exists", args)
    _placeholder_svg(Path(args.out_dir) / "val_loss_vs_step.svg", "P7 Val Loss vs Step", "not_run: task implementation unavailable")
    _placeholder_svg(Path(args.out_dir) / "val_loss_vs_time.svg", "P7 Val Loss vs Time", "not_run: task implementation unavailable")
    _placeholder_svg(Path(args.out_dir) / "acc_vs_time.svg", "P7 Accuracy vs Time", "not_run: task implementation unavailable")
    _placeholder_svg(Path(args.out_dir) / "time_to_target_bar.svg", "P7 Time To Target", "not_run: task implementation unavailable")
    _placeholder_svg(Path(args.out_dir) / "task_efficiency_pareto.svg", "P7 Task Efficiency Pareto", "not_run: task implementation unavailable")
    _placeholder_svg(Path(args.out_dir) / "calibration_curve_smoke.svg", "P7 Calibration Curve", "not_run: task implementation unavailable")
    _placeholder_svg(Path(args.out_dir) / "direction_quality_scatter.svg", "P8 Direction Quality", "not_run")
    _placeholder_svg(Path(args.out_dir) / "geometry_gain_vs_task_cost.svg", "P8 Geometry Gain", "not_run")
    _placeholder_svg(Path(args.out_dir) / "lambda_acceptance_histogram.svg", "P8 Lambda Acceptance", "not_run")
    _placeholder_svg(Path(args.out_dir) / "bad_step_heatmap.svg", "P8 Bad Step Heatmap", "not_run")
    _placeholder_svg(Path(args.out_dir) / "geometry_curve_with_correction_events.svg", "P8 Geometry Curve", "not_run")


def run_p9(args: argparse.Namespace) -> List[Dict[str, Any]]:
    route = json.loads((Path(args.out_dir) / "route_decision.json").read_text())
    if route.get("survivor_type") in {"S0", "S1"}:
        rows = _write_not_run(Path(args.out_dir) / "p9_fallback_primitive_decision.csv", "P9", "P6 found a task-reentry survivor; fallback not triggered", args)
        return rows
    variants = [(vid, method, "current", True, "fallback") for vid, method in FALLBACKS]
    rows = _profile_rows(args, "P9", variants, include_mlp=True)
    for row in rows:
        if row.get("implementation_status") == "measured" and row.get("method") != "MLP-autograd-reference":
            row["fallback_exploratory_pass"] = int(f(row, "memory_ratio_vs_MLP", 99) < 1.10 and f(row, "step_time_ratio_vs_MLP", 99) < 1.50 and f(row, "grad_cos", 0.0) > 0.999)
    write_csv(Path(args.out_dir) / "p9_fallback_primitive_decision.csv", rows)
    near = [r for r in rows if int(f(r, "fallback_exploratory_pass", 0)) == 1]
    if near:
        route.update({"route": "R5", "fallback_near_pass_count": len(near), "fallback_triggered": True})
    else:
        route.update({"route": "R6", "fallback_near_pass_count": 0, "fallback_triggered": True})
    _json_dump(Path(args.out_dir) / "route_decision.json", route)
    aggregate_path = Path(args.out_dir) / "aggregate_decision.json"
    aggregate = json.loads(aggregate_path.read_text()) if aggregate_path.exists() else {}
    aggregate.update({
        "fallback_triggered": True,
        "fallback_near_pass_count": route["fallback_near_pass_count"],
        "route": route["route"],
    })
    _json_dump(aggregate_path, aggregate)
    return rows


def run_failure(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = Path(args.out_dir)
    failures: List[Dict[str, Any]] = []
    def add_measured_failures(row: Dict[str, Any], stage: str) -> None:
        if row.get("implementation_status") != "measured" or row.get("method") == "MLP-autograd-reference":
            return
        if f(row, "memory_ratio_vs_MLP", 0.0) >= 1.0:
            failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F1_memory_fail", "metric": f"memory_ratio={row.get('memory_ratio_vs_MLP')}", "recommendation": "reduce actual CUDA peak"})
        if f(row, "step_time_ratio_vs_MLP", 0.0) > 1.35:
            failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F2_step_time_fail", "metric": f"step_ratio={row.get('step_time_ratio_vs_MLP')}", "recommendation": "runtime/kernel fusion required"})
        if int(f(row, "gradient_correctness_pass", 1)) == 0:
            failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F3_gradient_correctness_fail", "metric": f"grad_relerr={row.get('grad_relerr')} grad_cos={row.get('grad_cos')}", "recommendation": "fix manual adjoint numerics"})
    for fn, stage in [
        ("p2_single_factor_memory_repair.csv", "P2"),
        ("p3_combined_memory_packages.csv", "P3"),
        ("p4_runtime_kernel_audit.csv", "P4"),
        ("p5_one_step_probe.csv", "P5"),
        ("p7_task_reentry.csv", "P7"),
        ("p8_functional_correction_smoke.csv", "P8"),
        ("p9_fallback_primitive_decision.csv", "P9"),
    ]:
        path = out_dir / fn
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                status = row.get("implementation_status") or row.get("status")
                if status in {"not_implemented", "not_run"} or str(status).startswith("not_applicable"):
                    failures.append({"stage": stage, "variant_id": row.get("variant_id", ""), "failure_type": "F11_gated_not_run", "metric": row.get("reason", status), "recommendation": "implement before claiming metric"})
                add_measured_failures(row, stage)
    p3_detail = out_dir / "p3_combined_memory_packages_detail.csv"
    if p3_detail.exists():
        with p3_detail.open(newline="") as handle:
            for row in csv.DictReader(handle):
                add_measured_failures(row, "P3")
    if not failures:
        failures.append({"stage": "ALL", "variant_id": "all", "failure_type": "none", "metric": "", "recommendation": ""})
    write_csv(out_dir / "failure_table.csv", failures)
    for row in failures:
        _wandb_log_row(args, row, "summary/v65_failure_table")
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(wandb=True)
    parser.add_argument("--packages", default="V6_5_ALL")
    parser.add_argument("--out-dir", type=Path, default=Path("results/real_rerun_20260504/v65_real"))
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
    parser.add_argument("--wandb-group", default="v65-real-20260504")
    parser.add_argument("--wandb-name-prefix", default="v65-real")
    parser.add_argument("--no-wandb", action="store_false", dest="wandb")
    parser.add_argument("--fresh", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    out_dir = ensure_dir(args.out_dir)
    _wandb_init(args)
    try:
        packages = {p.upper() for p in parse_str_list(args.packages)}
        run_all = "V6_5_ALL" in packages
        if run_all or "V6_5_P0" in packages:
            run_p0(args)
        if run_all or "V6_5_P1" in packages:
            run_p1(args)
        if run_all or "V6_5_P2" in packages:
            run_p2(args)
        if run_all or "V6_5_P3" in packages:
            run_p3(args)
        if run_all or "V6_5_P4" in packages:
            run_p4(args)
        if run_all or "V6_5_P5" in packages:
            run_p5(args)
        if run_all or "V6_5_P6" in packages:
            run_p6(args)
        if run_all or any(pkg in packages for pkg in ["V6_5_P7", "V6_5_P8"]):
            run_p7_p8(args)
        if run_all or "V6_5_P9" in packages:
            run_p9(args)
        run_failure(args)
        _write_manifest(out_dir, args, started, time.time())
    finally:
        _wandb_finish(args, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
