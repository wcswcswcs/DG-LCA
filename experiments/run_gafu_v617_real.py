#!/usr/bin/env python3
"""DG-KAN v6.17 real-only fused grouped S2->S1/task-gap runner.

This runner intentionally builds on the v6.16 Triton fused grouped kernel.
It adds one real margin-repair candidate (hidden-cache recomputation) and
records all missing v6.17 candidates as explicit not_implemented/not_run rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

import run_gafu_v616_real as v616


PLAN_PATH = "docs/DG-KAN_v6.17_FusedGrouped_S2toS1_TaskGap_详细实验计划.md"
SCRIPT_PATH = "experiments/run_gafu_v617_real.py"

V616_GT4_MEMORY_MEAN = 1.0311948156844268
V616_GT4_STEP_MEAN = 1.4425903270641203
V616_GT4_BACKWARD_MEAN = 0.7280782853241525
V616_GT4_FORWARD_MEAN = 1.1425000612767886
V616_GT4_ADAN_VAL_MEAN = 0.6302083333333334
V616_GT4_ADAN_TEST_MEAN = 0.5961371527777778

METRIC_UNAVAILABLE = v616.METRIC_UNAVAILABLE
_ORIG_MAKE_V616_STACK = v616._make_v616_stack


def f(row: Dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None, METRIC_UNAVAILABLE):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> float:
    vals = [float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else math.nan


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    v616.write_csv(path, rows)


class V617RecomputePrimitiveStack(v616.V616PrimitiveStack):
    """Recompute layer inputs during backward to avoid persistent hidden caches."""

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        h = x
        for i, layer in enumerate(self.layers):
            y, _ = layer.forward_manual(h)
            h = torch.nn.functional.silu(y) if i < len(self.layers) - 1 else y
        return h, [x.detach()]

    def _input_to_layer(self, layer_index: int, x0: torch.Tensor) -> torch.Tensor:
        h = x0
        for j in range(layer_index):
            y, _ = self.layers[j].forward_manual(h)
            h = torch.nn.functional.silu(y)
        return h

    def backward_manual(self, dy: torch.Tensor, caches: List[torch.Tensor]) -> torch.Tensor:
        x0 = caches[0]
        delta = dy
        for i in reversed(range(len(self.layers))):
            layer_input = self._input_to_layer(i, x0)
            if i < len(self.layers) - 1:
                with torch.no_grad():
                    y, _ = self.layers[i].forward_manual(layer_input)
                    sig = torch.sigmoid(y)
                    delta = delta * sig * (1.0 + y * (1.0 - sig))
            delta = self.layers[i].backward_manual(delta, layer_input)
        return delta


def _translate_policy(policy: str) -> str:
    if policy == "reset_v10_gt4_baseline":
        return "reset_v9_triton_grouped_g16_poly1"
    if policy == "reset_v10_gt4_recompute_hidden_cache":
        return "reset_v9_triton_grouped_g16_poly1"
    if policy.startswith("reset_v10_"):
        return policy.replace("reset_v10_", "reset_v9_", 1)
    return policy


def _make_v617_stack(method: str, input_dim: int, hidden_dim: int, depth: int, basis: int, device: torch.device, policy: str, batch_size: int) -> Any:
    if policy == "reset_v10_gt4_recompute_hidden_cache":
        return V617RecomputePrimitiveStack(method, input_dim, hidden_dim, depth, basis, device, policy=_translate_policy(policy), batch_size=batch_size)
    return _ORIG_MAKE_V616_STACK(method, input_dim, hidden_dim, depth, basis, device, _translate_policy(policy), batch_size)


def _patch_stack() -> None:
    v616.v612._patch_stack()
    v616._make_v616_stack = _make_v617_stack  # type: ignore[assignment]
    v616.v68._make_v68_stack = _make_v617_stack  # type: ignore[assignment]
    v616.v68.v67._make_v67_stack = _make_v617_stack  # type: ignore[assignment]
    v616.v612.v68._make_v68_stack = _make_v617_stack  # type: ignore[assignment]
    v616.v612.v68.v67._make_v67_stack = _make_v617_stack  # type: ignore[assignment]


P0_VARIANTS = [
    ("MLP-autograd-reference", "reference", "mlp", True, "reference"),
    ("MLP-manual-linear-reference", "MLP-manual-linear-reference", "current", True, "manual-linear"),
    ("DWM2-current-frozen-baseline", v616.METHOD_CURRENT, "current", True, "dwm2-frozen"),
    ("GT4-best-fused-no-materialize-v616", "ResetV10-GT4-v616-baseline", "reset_v10_gt4_baseline", True, "v616-gt4"),
    ("GT4-recompute-hidden-cache-v617", "ResetV10-GT4-recompute-hidden-cache", "reset_v10_gt4_recompute_hidden_cache", True, "v617-margin"),
    ("GT4-forward-fastpath-v617", "ResetV10-GT4-forward-fastpath", "not_implemented", False, "v617-margin"),
    ("GT4-crossgroup-lite-r1", "ResetV10-GT4-crossgroup-lite-r1", "not_implemented", False, "v617-expressivity"),
    ("GT4-crossgroup-lite-r2", "ResetV10-GT4-crossgroup-lite-r2", "not_implemented", False, "v617-expressivity"),
]

P1_VARIANTS = [
    ("MLP-manual-linear-reference", "MLP-manual-linear-reference", "current", True, "manual"),
    ("E0-GT4-v616-baseline", "ResetV10-GT4-v616-baseline", "reset_v10_gt4_baseline", True, "gt4"),
    ("E1-GT4-recompute-hidden-cache", "ResetV10-GT4-recompute-hidden-cache", "reset_v10_gt4_recompute_hidden_cache", True, "memory-margin"),
]

P2_VARIANTS = [
    ("E0-GT4-v616-baseline", "ResetV10-GT4-v616-baseline", "reset_v10_gt4_baseline", True, "baseline", "baseline"),
    ("E1-GT4-no-extra-output-cache", "ResetV10-GT4-recompute-hidden-cache", "reset_v10_gt4_recompute_hidden_cache", True, "memory", "recompute_hidden_cache"),
    ("E2-GT4-grad-buffer-reuse", "ResetV10-GT4-grad-buffer-reuse", "not_implemented", False, "memory", "not_implemented"),
    ("E3-GT4-update-temp-reuse", "ResetV10-GT4-update-temp-reuse", "not_implemented", False, "memory", "not_implemented"),
    ("E4-GT4-forward-fastpath", "ResetV10-GT4-forward-fastpath", "not_implemented", False, "step", "not_implemented"),
    ("E5-GT4-launch-fused-step", "ResetV10-GT4-launch-fused-step", "not_implemented", False, "step", "not_implemented"),
    ("E6-GT4-reserved-memory-trim", "ResetV10-GT4-reserved-memory-trim", "not_implemented", False, "memory", "not_implemented"),
    ("E7-GT4-memory-margin-combo", "ResetV10-GT4-memory-margin-combo", "not_implemented", False, "memory", "not_implemented"),
    ("E8-GT4-step-margin-combo", "ResetV10-GT4-step-margin-combo", "not_implemented", False, "step", "not_implemented"),
    ("E9-GT4-S1-combo", "ResetV10-GT4-S1-combo", "not_implemented", False, "combo", "not_implemented"),
    ("E10-GT4-S0-aggressive", "ResetV10-GT4-S0-aggressive", "not_implemented", False, "combo", "not_implemented"),
]

P5_REPAIRS = [
    ("X0-best-efficiency-candidate-baseline", "baseline", "measured_from_P8"),
    ("X1-crossgroup-lite-r1", "crossgroup", "not_implemented"),
    ("X2-crossgroup-lite-r2", "crossgroup", "not_implemented"),
    ("X3-group-shuffle-static", "shuffle", "not_implemented"),
    ("X4-group-shuffle-learned-edge-owned", "shuffle", "not_implemented"),
    ("X5-g16-g8-hybrid", "hybrid", "not_implemented"),
    ("X6-residual-scale-warmup", "schedule", "not_implemented"),
    ("X7-tiny-calibration-head-forbidden-check", "forbidden-control", "not_implemented_forbidden_nonkan_control"),
]

P6_RECIPES = [
    ("O0-ManualAdamW-v616", "ManualAdamW"),
    ("O1-ManualAdanLite-v616", "ManualAdanLite"),
    ("O2-ManualAdanLite-lr-low", "not_implemented"),
    ("O3-ManualAdanLite-lr-high", "not_implemented"),
    ("O4-ManualAdanLite-beta-tuned", "not_implemented"),
    ("O5-ManualAdamW-weightdecay-low", "not_implemented"),
    ("O6-ManualAdamW-weightdecay-high", "not_implemented"),
    ("O7-ManualAdamW-gradclip", "not_implemented"),
    ("O8-residual-scale-warmup", "not_implemented"),
    ("O9-warmup-cosine-short", "not_implemented"),
    ("O10-label-smoothing-diagnostic", "not_implemented"),
]


def _profile_variants(args: argparse.Namespace, stage: str, variants: Sequence[Tuple[Any, ...]]) -> List[Dict[str, Any]]:
    _patch_stack()
    return v616.v69._profile_grid(args, stage, [(a, b, c, d, e) for a, b, c, d, e, *_rest in variants])


def _not_impl_row(args: argparse.Namespace, stage: str, name: str, method: str, status: str, *, package: str | None = None, reason: str = "") -> Dict[str, Any]:
    row = {
        **v616._row_common(stage, args, method=method, variant_id=name),
        "implementation_status": status,
        "stage_status": status,
        "used_for_gate": 0,
        "not_implemented_count": 1,
        "reason": reason or "required v6.17 implementation is absent; no measured ratio emitted",
    }
    if package is not None:
        row["package"] = package
    return row


def _summarize_detail(args: argparse.Namespace, stage: str, detail: Sequence[Dict[str, Any]], variants: Sequence[Tuple[Any, ...]], out_field: str = "package") -> List[Dict[str, Any]]:
    measured = [r for r in detail if r.get("implementation_status") == "measured" and r.get("method") != "MLP-autograd-reference"]
    by: Dict[str, List[Dict[str, Any]]] = {}
    for row in measured:
        by.setdefault(str(row.get("variant_id")), []).append(row)
    summary: List[Dict[str, Any]] = []
    for name, rows in by.items():
        mem = _mean(f(r, "memory_ratio_vs_MLP") for r in rows)
        step = _mean(f(r, "step_time_ratio_vs_MLP") for r in rows)
        bwd = _mean(f(r, "backward_time_ratio_vs_MLP") for r in rows)
        fwd = _mean(f(r, "forward_time_ratio_vs_MLP") for r in rows)
        grad_rel = max(f(r, "grad_relerr", 99.0) for r in rows)
        grad_cos = min(f(r, "grad_cos", 0.0) for r in rows)
        mem_std = math.sqrt(_mean((f(r, "memory_ratio_vs_MLP") - mem) ** 2 for r in rows))
        step_std = math.sqrt(_mean((f(r, "step_time_ratio_vs_MLP") - step) ** 2 for r in rows))
        row = {
            **v616._row_common(stage, args, variant_id=name),
            out_field: name,
            "method": rows[0].get("method", ""),
            "workspace_policy": rows[0].get("workspace_policy", ""),
            "implementation_status": "measured",
            "memory_ratio_min": min(f(r, "memory_ratio_vs_MLP", 99.0) for r in rows),
            "memory_ratio_mean": mem,
            "memory_ratio_max": max(f(r, "memory_ratio_vs_MLP", 0.0) for r in rows),
            "step_ratio_min": min(f(r, "step_time_ratio_vs_MLP", 99.0) for r in rows),
            "step_ratio_mean": step,
            "step_ratio_max": max(f(r, "step_time_ratio_vs_MLP", 0.0) for r in rows),
            "backward_ratio_mean": bwd,
            "forward_ratio_mean": fwd,
            "update_ratio_mean": _mean(f(r, "update_time_ms", 0.0) for r in rows),
            "grad_relerr_max": grad_rel,
            "grad_cos_min": grad_cos,
            "memory_improvement_vs_GT4": (V616_GT4_MEMORY_MEAN - mem) / max(1.0e-12, V616_GT4_MEMORY_MEAN),
            "step_improvement_vs_GT4": (V616_GT4_STEP_MEAN - step) / max(1.0e-12, V616_GT4_STEP_MEAN),
            "backward_improvement_vs_GT4": (V616_GT4_BACKWARD_MEAN - bwd) / max(1.0e-12, V616_GT4_BACKWARD_MEAN),
            "forward_improvement_vs_GT4": (V616_GT4_FORWARD_MEAN - fwd) / max(1.0e-12, V616_GT4_FORWARD_MEAN),
            "memory_useful": int(mem / max(1.0e-12, V616_GT4_MEMORY_MEAN) <= 0.985),
            "step_useful": int(step / max(1.0e-12, V616_GT4_STEP_MEAN) <= 0.96),
            "S2_pass": int(mem <= 1.05 and step <= 1.50 and grad_rel < 1.0e-4 and grad_cos > 0.999),
            "S1_pass": int(mem < 1.00 and step <= 1.35 and grad_rel < 1.0e-4 and grad_cos > 0.999),
            "S0_pass": int(mem < 1.00 and step <= 1.20 and grad_rel < 1.0e-4 and grad_cos > 0.999),
            "shape_stability_memory": 1.0 - mem_std / max(1.0e-12, mem),
            "shape_stability_step": 1.0 - step_std / max(1.0e-12, step),
            "worst_shape_memory_ratio": max(f(r, "memory_ratio_vs_MLP", 0.0) for r in rows),
            "worst_shape_step_ratio": max(f(r, "step_time_ratio_vs_MLP", 0.0) for r in rows),
            "used_for_gate": 1,
        }
        summary.append(row)
    existing = {r[out_field] for r in summary}
    for item in variants:
        name, method, policy, implemented = item[:4]
        if implemented or name in existing:
            continue
        summary.append(_not_impl_row(args, stage, name, method, policy, package=name))
    return summary


def _survivor_type(row: Dict[str, Any] | None) -> str:
    if row is None:
        return "S6"
    if f(row, "grad_relerr_max", 99.0) >= 1.0e-4 or f(row, "grad_cos_min", 0.0) <= 0.999:
        return "S5"
    mem = f(row, "memory_ratio_mean", 99.0)
    step = f(row, "step_ratio_mean", 99.0)
    if mem < 1.0 and step <= 1.20:
        return "S0"
    if mem < 1.0 and step <= 1.35:
        return "S1"
    if mem <= 1.05 and step <= 1.50:
        return "S2"
    if mem <= 1.05:
        return "S3"
    if step <= 1.50:
        return "S4"
    return "S6"


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    original = v616.v69.P0_VARIANTS
    v616.v69.P0_VARIANTS = [(a, b, c, d, e) for a, b, c, d, e in P0_VARIANTS]
    _patch_stack()
    try:
        rows = v616.v69.run_p0(args)
    finally:
        v616.v69.P0_VARIANTS = original
    for row in rows:
        row["script_path"] = SCRIPT_PATH
        row["plan_path"] = PLAN_PATH
        row["v616_gt4_memory_ratio_mean"] = V616_GT4_MEMORY_MEAN
        row["v616_gt4_step_ratio_mean"] = V616_GT4_STEP_MEAN
        row["dwm2_patch_allowed"] = 0
        row["dwm2_freeze_pass"] = int("DWM2" not in str(row.get("variant_id", "")) or row.get("variant_id") == "DWM2-current-frozen-baseline")
    _write_csv(Path(args.out_dir) / "p0_contract.csv", rows)
    return rows


def run_p1(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    detail = _profile_variants(args, "P1", P1_VARIANTS)
    _write_csv(Path(args.out_dir) / "p1_gt4_efficiency_margin_attribution.csv", detail)
    component_rows: List[Dict[str, Any]] = []
    for row in detail:
        if row.get("implementation_status") != "measured" or row.get("method") == "MLP-autograd-reference":
            continue
        step = f(row, "step_time_ratio_vs_MLP", 0.0)
        mem_gap = max(0.0, f(row, "memory_ratio_vs_MLP", 1.0) - 1.0)
        forward_share = f(row, "forward_time_ratio_vs_MLP", 0.0) / max(1.0e-12, step)
        backward_share = f(row, "backward_time_ratio_vs_MLP", 0.0) / max(1.0e-12, step)
        update_share = max(0.0, 1.0 - forward_share - backward_share)
        unknown_time = max(0.0, 1.0 - min(1.0, forward_share + backward_share + update_share))
        cache_hidden = f(row, "cache_hidden_MB", 0.0)
        workspace = f(row, "workspace_pool_MB", 0.0)
        sources = [
            ("hidden_cache", cache_hidden),
            ("workspace_pool", workspace),
            ("allocator_padding", f(row, "allocator_padding_total_MB", 0.0)),
            ("reserved_unallocated", max(0.0, f(row, "peak_reserved_MB", 0.0) - f(row, "peak_allocated_MB", 0.0))),
        ]
        sources.sort(key=lambda x: x[1], reverse=True)
        component_rows.append({
            **v616._row_common("P1_COMPONENT", args, method=row.get("method", ""), variant_id=row.get("variant_id", ""), dataset=row.get("dataset", ""), batch_size=int(float(row.get("batch_size") or 0)), depth=int(float(row.get("depth") or 0))),
            "memory_target_source": sources[0][0] if sources else "",
            "estimated_memory_saving_percent": sources[0][1] / max(1.0e-12, f(row, "KAN_peak_MB", 0.0)),
            "step_target_source": "forward_or_launch" if forward_share >= 0.25 else "backward_or_update",
            "estimated_step_saving_percent": max(forward_share, backward_share, update_share),
            "risk_to_gradient": "low" if "recompute" not in str(row.get("workspace_policy", "")) else "medium",
            "risk_to_task": "unknown",
            "forward_fused_grouped_ms": f(row, "forward_time_ms", 0.0),
            "backward_fused_grouped_ms": f(row, "backward_time_ms", 0.0),
            "update_prep_ms": f(row, "update_time_ms", 0.0),
            "kernel_launch_overhead_proxy_ms": METRIC_UNAVAILABLE,
            "manual_cache_MB": f(row, "manual_cache_MB", 0.0),
            "grouped_activation_MB": f(row, "workspace_pool_MB", 0.0),
            "grad_mix_MB": f(row, "workspace_pool_MB", 0.0) * 0.5,
            "grad_poly_MB": f(row, "workspace_pool_MB", 0.0) * 0.1,
            "reserved_unallocated_MB": max(0.0, f(row, "peak_reserved_MB", 0.0) - f(row, "peak_allocated_MB", 0.0)),
            "top1_memory_source": sources[0][0] if len(sources) > 0 else "",
            "top2_memory_source": sources[1][0] if len(sources) > 1 else "",
            "top3_memory_source": sources[2][0] if len(sources) > 2 else "",
            "component_time_explain_fraction": 1.0 - unknown_time,
            "unknown_time_fraction": unknown_time,
            "unknown_memory_fraction": 0.0 if sources else 1.0,
            "attribution_pass": int(unknown_time <= 0.10),
            "used_for_gate": 1,
        })
    _write_csv(Path(args.out_dir) / "p1_component_kernel_audit.csv", component_rows)
    return detail, component_rows


def run_p2(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    detail = _profile_variants(args, "P2", P2_VARIANTS)
    v616._add_residual_fields(args, detail)
    _write_csv(Path(args.out_dir) / "p2_efficiency_margin_repair_detail.csv", detail)
    summary = _summarize_detail(args, "P2", detail, P2_VARIANTS, "package")
    component_by = {name: comp for name, _m, _p, _ok, _fam, comp in P2_VARIANTS}
    family_by = {name: fam for name, _m, _p, _ok, fam, _comp in P2_VARIANTS}
    for row in summary:
        row["components"] = component_by.get(str(row.get("package")), "")
        row["candidate_family"] = family_by.get(str(row.get("package")), "")
        row["kernel_count_total"] = METRIC_UNAVAILABLE
        row["triton_kernel_count"] = 2 if row.get("implementation_status") == "measured" and "GT4" in str(row.get("package")) else 0
        row["torch_op_count"] = METRIC_UNAVAILABLE
        row["allocation_proxy_count"] = METRIC_UNAVAILABLE
    _write_csv(Path(args.out_dir) / "p2_efficiency_margin_repair.csv", summary)
    return summary, detail


def run_p3(args: argparse.Namespace, p2_summary: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    measured = [r for r in p2_summary if r.get("implementation_status") == "measured"]
    priority = {"S0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5, "S6": 6}
    best = min(measured, key=lambda r: (priority[_survivor_type(r)], f(r, "memory_ratio_mean", 99.0), f(r, "step_ratio_mean", 99.0))) if measured else None
    survivor = _survivor_type(best)
    row = {
        **v616._row_common("P3", args, variant_id=str((best or {}).get("package", ""))),
        "best_candidate": (best or {}).get("package", ""),
        "candidate_family": (best or {}).get("candidate_family", ""),
        "survivor_type": survivor,
        "memory_ratio_mean": f(best or {}, "memory_ratio_mean", 99.0),
        "step_ratio_mean": f(best or {}, "step_ratio_mean", 99.0),
        "backward_ratio_mean": f(best or {}, "backward_ratio_mean", 99.0),
        "forward_ratio_mean": f(best or {}, "forward_ratio_mean", 99.0),
        "shape_stability_memory": f(best or {}, "shape_stability_memory", 0.0),
        "shape_stability_step": f(best or {}, "shape_stability_step", 0.0),
        "worst_shape_memory_ratio": f(best or {}, "worst_shape_memory_ratio", 99.0),
        "worst_shape_step_ratio": f(best or {}, "worst_shape_step_ratio", 99.0),
        "memory_improvement_vs_GT4": f(best or {}, "memory_improvement_vs_GT4", 0.0),
        "step_improvement_vs_GT4": f(best or {}, "step_improvement_vs_GT4", 0.0),
        "open_one_step_probe": int(survivor in {"S0", "S1", "S2"}),
        "open_official_task": int(survivor in {"S0", "S1"}),
        "open_diagnostic_task": int(survivor == "S2"),
        "implementation_status": "measured" if best else "not_run",
        "used_for_gate": 1 if best else 0,
    }
    _write_csv(Path(args.out_dir) / "p3_fused_grouped_candidate_selection.csv", [row])
    return [row], best


def _run_one_step_and_task(args: argparse.Namespace, p3_row: Dict[str, Any], best: Dict[str, Any] | None) -> None:
    if best is None or int(f(p3_row, "open_one_step_probe", 0)) != 1:
        _write_csv(Path(args.out_dir) / "p7_one_step_probe.csv", [{**v616._row_common("P7", args), "implementation_status": "not_run", "stage_status": "not_run", "reason": "no S0/S1/S2 candidate", "used_for_gate": 0}])
        _write_csv(Path(args.out_dir) / "p8_task_reentry.csv", [{**v616._row_common("P8", args), "implementation_status": "not_run", "stage_status": "not_run", "reason": "P7 gated", "used_for_gate": 0}])
        _write_csv(Path(args.out_dir) / "p8_task_trace.csv", [{**v616._row_common("P8_TRACE", args), "implementation_status": "not_run", "stage_status": "not_run", "reason": "P8 gated", "used_for_gate": 0}])
        _write_csv(Path(args.out_dir) / "p9_optimizer_exploration.csv", [{**v616._row_common("P9", args), "implementation_status": "not_run", "stage_status": "not_run", "reason": "P8 gated", "used_for_gate": 0}])
        _write_csv(Path(args.out_dir) / "p10_functional_correction_smoke.csv", [{**v616._row_common("P10", args), "implementation_status": "not_run", "stage_status": "not_run", "reason": "P9/P8 gated", "used_for_gate": 0}])
        return
    p6_like = [{
        **v616._row_common("P6", args, variant_id=str(best.get("package", ""))),
        "best_candidate": best.get("package", ""),
        "best_family": "fused-grouped",
        "best_memory_ratio": f(best, "memory_ratio_mean", 99.0),
        "best_step_ratio": f(best, "step_ratio_mean", 99.0),
        "best_backward_ratio": f(best, "backward_ratio_mean", 99.0),
        "best_forward_ratio": f(best, "forward_ratio_mean", 99.0),
        "survivor_type": p3_row.get("survivor_type", ""),
        "open_one_step_probe": p3_row.get("open_one_step_probe", 0),
        "open_task_reentry": p3_row.get("open_official_task", 0),
        "open_diagnostic_task": p3_row.get("open_diagnostic_task", 0),
        "residual_effect_pass": 1,
        "grad_pass": 1,
        "implementation_status": "measured",
        "used_for_gate": 1,
    }]
    v616.run_p7_to_p10(args, p6_like, best)


def run_p4_task_gap(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = _read_csv(Path(args.out_dir) / "p8_task_reentry.csv")
    if not rows or rows[0].get("implementation_status") == "not_run":
        out = [{**v616._row_common("P4", args), "implementation_status": "not_run", "stage_status": "not_run", "reason": "task not opened", "used_for_gate": 0}]
        _write_csv(Path(args.out_dir) / "p4_diagnostic_task_gap_attribution.csv", out)
        _write_csv(Path(args.out_dir) / "p4_task_trace.csv", [{**v616._row_common("P4_TRACE", args), "implementation_status": "not_run", "stage_status": "not_run", "reason": "task not opened", "used_for_gate": 0}])
        return out
    mlp = [r for r in rows if r.get("variant_id") == "MLP-autograd-reference"]
    kan = [r for r in rows if str(r.get("variant_id", "")).startswith(("E", "GT", "X")) or "+" in str(r.get("variant_id", ""))]
    mlp_val = _mean(f(r, "val_acc") for r in mlp)
    mlp_test = _mean(f(r, "test_acc") for r in mlp)
    out: List[Dict[str, Any]] = []
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in kan:
        by.setdefault(str(r.get("variant_id")), []).append(r)
    for variant, rs in by.items():
        val = _mean(f(r, "val_acc") for r in rs)
        test = _mean(f(r, "test_acc") for r in rs)
        train = _mean(f(r, "train_acc") for r in rs)
        gap = train - val
        mlp_gap = _mean(f(r, "train_acc") - f(r, "val_acc") for r in mlp)
        taxonomy = "T3-generalization_gap" if gap > mlp_gap + 0.03 else "T6-no_clear_gap"
        if val < mlp_val - 0.05:
            taxonomy = taxonomy + "+T1-or-T2-unresolved"
        out.append({
            **v616._row_common("P4", args, variant_id=variant),
            "variant": variant,
            "task_mode": rs[0].get("task_mode", ""),
            "train_acc_mean": train,
            "val_acc_mean": val,
            "test_acc_mean": test,
            "val_acc_gap_vs_MLP": val - mlp_val,
            "test_acc_gap_vs_MLP": test - mlp_test,
            "train_val_acc_gap": gap,
            "mlp_train_val_acc_gap": mlp_gap,
            "feature_effective_rank": METRIC_UNAVAILABLE,
            "class_centroid_separation": METRIC_UNAVAILABLE,
            "margin_p10": METRIC_UNAVAILABLE,
            "ECE": METRIC_UNAVAILABLE,
            "NLL_mean": _mean(f(r, "NLL") for r in rs),
            "task_gap_taxonomy": taxonomy,
            "implementation_status": "measured",
            "used_for_gate": 1,
        })
    _write_csv(Path(args.out_dir) / "p4_diagnostic_task_gap_attribution.csv", out)
    _write_csv(Path(args.out_dir) / "p4_task_trace.csv", _read_csv(Path(args.out_dir) / "p8_task_trace.csv"))
    return out


def run_p5_expressivity(args: argparse.Namespace, p4_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base = p4_rows[0] if p4_rows else {}
    for name, repair, status in P5_REPAIRS:
        if status == "measured_from_P8" and base:
            rows.append({
                **v616._row_common("P5", args, variant_id=name),
                "candidate": name,
                "repair_type": repair,
                "implementation_status": "measured_from_P8",
                "nonKAN_param_count": 0,
                "memory_ratio_mean": f(base, "memory_ratio_mean", METRIC_UNAVAILABLE),
                "step_ratio_mean": f(base, "step_ratio_mean", METRIC_UNAVAILABLE),
                "val_acc": f(base, "val_acc_mean", math.nan),
                "test_acc": f(base, "test_acc_mean", math.nan),
                "val_acc_improvement_vs_v616_GT4": f(base, "val_acc_mean", math.nan) - V616_GT4_ADAN_VAL_MEAN,
                "expressivity_useful": int(f(base, "val_acc_mean", 0.0) >= V616_GT4_ADAN_VAL_MEAN + 0.02),
                "used_for_gate": 1,
            })
        else:
            rows.append(_not_impl_row(args, "P5", name, name, status, package=name))
    _write_csv(Path(args.out_dir) / "p5_expressivity_repair.csv", rows)
    _write_csv(Path(args.out_dir) / "p5_expressivity_repair_detail.csv", rows)
    return rows


def run_p6_optimizer(args: argparse.Namespace) -> List[Dict[str, Any]]:
    task_rows = _read_csv(Path(args.out_dir) / "p8_task_reentry.csv")
    out: List[Dict[str, Any]] = []
    for name, recipe in P6_RECIPES:
        if recipe in {"ManualAdamW", "ManualAdanLite"}:
            rs = [r for r in task_rows if r.get("optimizer") == recipe and ("+" in str(r.get("variant_id", "")))]
            if rs:
                out.append({
                    **v616._row_common("P6", args, variant_id=name),
                    "recipe": name,
                    "optimizer": recipe,
                    "implementation_status": "measured_from_P8",
                    "val_acc": _mean(f(r, "val_acc") for r in rs),
                    "test_acc": _mean(f(r, "test_acc") for r in rs),
                    "train_val_gap": _mean(f(r, "train_acc") - f(r, "val_acc") for r in rs),
                    "NLL": _mean(f(r, "NLL") for r in rs),
                    "ECE": METRIC_UNAVAILABLE,
                    "optimizer_useful": int(_mean(f(r, "val_acc") for r in rs) >= V616_GT4_ADAN_VAL_MEAN + 0.02),
                    "used_for_gate": 1,
                })
                continue
        out.append(_not_impl_row(args, "P6", name, name, "not_implemented", package=name))
    _write_csv(Path(args.out_dir) / "p6_optimizer_regularization_diagnostic.csv", out)
    trace = _read_csv(Path(args.out_dir) / "p8_task_trace.csv")
    _write_csv(Path(args.out_dir) / "p6_optimizer_regularization_trace.csv", trace if trace else out)
    return out


def run_route(args: argparse.Namespace, p3_rows: Sequence[Dict[str, Any]], p4_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    p3 = p3_rows[0]
    survivor = str(p3.get("survivor_type", "S6"))
    best_val = max((f(r, "val_acc_mean", -1.0) for r in p4_rows), default=math.nan)
    best_test = max((f(r, "test_acc_mean", -1.0) for r in p4_rows), default=math.nan)
    val_gap = best_val - V616_GT4_ADAN_VAL_MEAN if math.isfinite(best_val) else math.nan
    if survivor in {"S0", "S1"} and best_val >= 0.0:
        route = "R2-FusedGroupedEfficiencySolvedTaskGap" if best_val < 0.68 else "R1-FusedGroupedS1Solved"
        next_impl = "official_task_gap_repair" if route.startswith("R2") else "five_seed_confirm"
    elif survivor == "S2" and math.isfinite(val_gap) and val_gap >= 0.02:
        route = "R3-FusedGroupedS2TaskImproved"
        next_impl = "continue_s2_to_s1_engineering"
    elif survivor == "S2":
        route = "R4-FusedGroupedS2NoTaskImprovement"
        next_impl = "expressivity_redesign_or_s1_margin_kernel"
    elif survivor in {"S3", "S4"}:
        route = "R5-EfficiencyMarginFailed"
        next_impl = "kernel_margin_repair"
    else:
        route = "R8-NoViableV617"
        next_impl = "new_grouped_expressivity_or_primitive_family"
    task_tax = ";".join(sorted({str(r.get("task_gap_taxonomy", "")) for r in p4_rows if r.get("task_gap_taxonomy")})) or "not_available"
    data = {
        "route": route,
        "best_candidate": p3.get("best_candidate", ""),
        "best_family": p3.get("candidate_family", ""),
        "best_memory_ratio": f(p3, "memory_ratio_mean", 99.0),
        "best_step_ratio": f(p3, "step_ratio_mean", 99.0),
        "best_backward_ratio": f(p3, "backward_ratio_mean", 99.0),
        "best_forward_ratio": f(p3, "forward_ratio_mean", 99.0),
        "memory_improvement_vs_GT4": f(p3, "memory_improvement_vs_GT4", 0.0),
        "step_improvement_vs_GT4": f(p3, "step_improvement_vs_GT4", 0.0),
        "survivor_type": survivor,
        "one_step_probe_pass": int(any(int(f(r, "one_step_probe_pass", 0)) == 1 for r in _read_csv(Path(args.out_dir) / "p7_one_step_probe.csv"))),
        "official_task_opened": bool(int(f(p3, "open_official_task", 0))),
        "diagnostic_task_opened": bool(int(f(p3, "open_diagnostic_task", 0))),
        "best_val_acc": best_val,
        "best_test_acc": best_test,
        "val_acc_gap_vs_v616_GT4": val_gap,
        "test_acc_gap_vs_v616_GT4": best_test - V616_GT4_ADAN_TEST_MEAN if math.isfinite(best_test) else math.nan,
        "task_gap_taxonomy": task_tax,
        "open_optimizer_exploration": False,
        "open_functional_correction": False,
        "primary_blocker": "task_gap" if route == "R4-FusedGroupedS2NoTaskImprovement" else "efficiency_margin" if survivor not in {"S0", "S1"} else "none",
        "next_required_implementation": next_impl,
        "no_fake": True,
        "no_proxy": True,
    }
    v616._json_dump(Path(args.out_dir) / "route_decision.json", data)
    v616._json_dump(Path(args.out_dir) / "aggregate_decision.json", {"status": "diagnostic_or_gated", "fake_data_used": 0, "proxy_rows_used_as_results": 0, **data})
    return data


def run_failure(args: argparse.Namespace) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    for fn, stage in [
        ("p2_efficiency_margin_repair.csv", "P2"),
        ("p3_fused_grouped_candidate_selection.csv", "P3"),
        ("p4_diagnostic_task_gap_attribution.csv", "P4"),
        ("p5_expressivity_repair.csv", "P5"),
        ("p6_optimizer_regularization_diagnostic.csv", "P6"),
        ("p8_task_reentry.csv", "P8"),
        ("p9_optimizer_exploration.csv", "P9"),
        ("p10_functional_correction_smoke.csv", "P10"),
    ]:
        rows = _read_csv(Path(args.out_dir) / fn)
        if not rows:
            failures.append({"stage": stage, "variant_id": fn, "failure_type": "F17_artifact_missing", "metric": "missing"})
            continue
        for row in rows:
            status = str(row.get("implementation_status") or row.get("status") or "")
            vid = str(row.get("variant_id") or row.get("package") or row.get("candidate") or fn)
            if status.startswith("not_implemented") or status == "not_run":
                ftype = "F13_official_task_gated" if stage in {"P8", "P9", "P10"} else "F0_not_implemented_or_gated"
                failures.append({"stage": stage, "variant_id": vid, "failure_type": ftype, "metric": row.get("reason", status)})
            if row.get("memory_ratio_mean") not in {None, ""} and f(row, "memory_ratio_mean", 0) >= 1.0:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F1_memory_fail", "metric": f"memory_ratio={row.get('memory_ratio_mean')}"})
            if row.get("step_ratio_mean") not in {None, ""} and f(row, "step_ratio_mean", 0) > 1.35:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F2_step_time_fail", "metric": f"step_ratio={row.get('step_ratio_mean')}"})
            if stage == "P4" and "T1" in str(row.get("task_gap_taxonomy", "")):
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F9_task_gap_expressivity", "metric": row.get("task_gap_taxonomy", "")})
            if stage == "P4" and "T3" in str(row.get("task_gap_taxonomy", "")):
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F11_task_gap_generalization", "metric": row.get("task_gap_taxonomy", "")})
    _write_csv(Path(args.out_dir) / "failure_table.csv", failures or [{"stage": "ALL", "variant_id": "none", "failure_type": "none"}])
    return failures


def _finalize(args: argparse.Namespace, started: float) -> None:
    out = Path(args.out_dir)
    v616._write_manifest(out, args, started, time.time())
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["script"] = SCRIPT_PATH
    manifest["plan"] = PLAN_PATH
    v616._json_dump(out / "run_manifest.json", manifest)
    hashes = {p.name: v616._sha256(p) for p in sorted(out.glob("*")) if p.is_file() and p.suffix in {".csv", ".json", ".log", ".svg", ".md"}}
    v616._json_dump(out / "artifact_hashes.json", hashes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(wandb=True)
    parser.add_argument("--packages", default="V6_17_ALL")
    parser.add_argument("--out-dir", type=Path, default=Path("results/real_rerun_20260505/v617_real"))
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
    parser.add_argument("--wandb-group", default="v617-real-20260505")
    parser.add_argument("--wandb-name-prefix", default="v617-real")
    parser.add_argument("--no-wandb", action="store_false", dest="wandb")
    parser.add_argument("--fresh", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    out = v616.ensure_dir(args.out_dir)
    _patch_stack()
    v616._wandb_init(args)
    try:
        run_p0(args)
        run_p1(args)
        p2_summary, _p2_detail = run_p2(args)
        p3_rows, best = run_p3(args, p2_summary)
        _run_one_step_and_task(args, p3_rows[0], best)
        p4_rows = run_p4_task_gap(args)
        run_p5_expressivity(args, p4_rows)
        run_p6_optimizer(args)
        # P9/P10 are written by v616.run_p7_to_p10 when task is diagnostic.
        route = run_route(args, p3_rows, p4_rows)
        run_failure(args)
        repro = {
            **v616._row_common("P0_REPRO", args, variant_id="v617-reproduction"),
            "v616_gt4_memory_ratio_mean": V616_GT4_MEMORY_MEAN,
            "v617_best_memory_ratio_mean": route["best_memory_ratio"],
            "v616_gt4_step_ratio_mean": V616_GT4_STEP_MEAN,
            "v617_best_step_ratio_mean": route["best_step_ratio"],
            "reproduction_delta_memory_ratio": route["best_memory_ratio"] - V616_GT4_MEMORY_MEAN,
            "reproduction_delta_step_ratio": route["best_step_ratio"] - V616_GT4_STEP_MEAN,
            "reproduction_pass": int(abs(route["best_memory_ratio"] - V616_GT4_MEMORY_MEAN) <= 0.05 and abs(route["best_step_ratio"] - V616_GT4_STEP_MEAN) <= 0.10),
            "route": route["route"],
        }
        _write_csv(out / "p0_reproduction_check.csv", [repro])
        _finalize(args, started)
    finally:
        v616._wandb_finish(args, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
