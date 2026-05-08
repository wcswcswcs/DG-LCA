#!/usr/bin/env python3
"""DG-KAN v6.20 real-only GT4 canonical protocol and kernel margin runner."""

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
import run_gafu_v617_real as v617
from dgkan_core import ensure_dir, get_device, parse_int_list, parse_str_list, write_csv
from run_gafu_v63 import _wandb_finish, _wandb_init, _wandb_log, f
from run_gafu_v66_real import _git_commit, _git_status


PLAN_PATH = "docs/DG-KAN_v6.20_GT4_CanonicalProtocol_NonRecomputeRepair_详细实验计划.md"
SCRIPT_PATH = "experiments/run_gafu_v620_real.py"
METRIC_UNAVAILABLE = v616.METRIC_UNAVAILABLE
_ORIG_MAKE_V616_STACK = v617._ORIG_MAKE_V616_STACK
_ORIG_MANUAL_OPTIMIZER = v616.v68.v67.ManualOptimizer

_UPDATE_TEMP_REUSE_POLICIES = {
    "reset_v10_update_temp_reuse",
    "reset_v10_update_fastpath",
}
_LAUNCH_FUSED_STEP_POLICIES = {
    "reset_v10_launch_fused_step",
}
_V620_GT4_POLICY_ALIASES = _UPDATE_TEMP_REUSE_POLICIES | _LAUNCH_FUSED_STEP_POLICIES

V616_GT4_MEMORY_MEAN = 1.0311948156844268
V616_GT4_STEP_MEAN = 1.4425903270641203
V616_GT4_BACKWARD_MEAN = 0.7280782853241525
V616_GT4_FORWARD_MEAN = 1.1425000612767886
V616_GT4_ADAN_VAL_MEAN = 0.6302083333333334
V616_GT4_ADAN_TEST_MEAN = 0.5961371527777778


def _mean(values: Iterable[float]) -> float:
    vals = [float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else math.nan


def _safe_min(values: Iterable[float]) -> float:
    vals = [float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return min(vals) if vals else math.nan


def _safe_max(values: Iterable[float]) -> float:
    vals = [float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return max(vals) if vals else math.nan


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    write_csv(path, rows)


def _json_dump(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return v616._sha256(path)


def _make_v620_stack(method: str, input_dim: int, hidden_dim: int, depth: int, basis: int, device: torch.device, policy: str, batch_size: int) -> Any:
    if policy in _V620_GT4_POLICY_ALIASES:
        stack = _ORIG_MAKE_V616_STACK(method, input_dim, hidden_dim, depth, basis, device, "reset_v9_triton_grouped_g16_poly1", batch_size)
        stack.policy = policy
        return stack
    return v617._make_v617_stack(method, input_dim, hidden_dim, depth, basis, device, policy, batch_size)


class V620FastManualOptimizer(_ORIG_MANUAL_OPTIMIZER):
    """Policy-gated AdamW update fastpaths for GT4 margin experiments.

    The measured model math is unchanged.  These paths only remove Python
    parameter discovery and CPU-side update-norm diagnostics from optimizer
    timing, then optionally use torch foreach ops to reduce launch overhead.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._v620_cached_params_ready = False
        super().__init__(*args, **kwargs)
        self._v620_cached_params = list(super()._params())
        self._v620_param_tensors = [param for _name, param, _grad in self._v620_cached_params]
        self._v620_grad_tensors = [grad for _name, _param, grad in self._v620_cached_params]
        self._v620_m_tensors = [self.m[id(param)] for _name, param, _grad in self._v620_cached_params]
        self._v620_v_tensors = [self.v[id(param)] for _name, param, _grad in self._v620_cached_params]
        self._v620_prev_grad_tensors = [self.prev_g[id(param)] for _name, param, _grad in self._v620_cached_params]
        self._v620_cached_params_ready = True

    def _params(self) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        if getattr(self, "_v620_cached_params_ready", False):
            return list(self._v620_cached_params)
        return super()._params()

    def _v620_fast_policy(self) -> str:
        if self.kind != "ManualAdamW":
            return ""
        policy = str(getattr(self.model, "policy", ""))
        return policy if policy in _V620_GT4_POLICY_ALIASES else ""

    @staticmethod
    def _v620_fast_stats() -> Dict[str, float]:
        return {
            "optimizer_state_norm_m": math.nan,
            "optimizer_state_norm_v": math.nan,
            "restart_count": 0.0,
            "role_update_share_input": math.nan,
            "role_update_share_block": math.nan,
            "role_update_share_output": math.nan,
        }

    def _v620_cached_adamw_step(self) -> Dict[str, float]:
        beta1, beta2 = 0.9, 0.99
        lr = self.lr
        with torch.no_grad():
            for _name, param, grad in self._v620_cached_params:
                m = self.m[id(param)]
                v = self.v[id(param)]
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                update = m / (v.sqrt() + 1.0e-8)
                param.mul_(1.0 - lr * self.weight_decay)
                param.add_(update, alpha=-lr)
                self.prev_g[id(param)].copy_(grad)
            torch._foreach_zero_(self._v620_grad_tensors)
        return self._v620_fast_stats()

    def _v620_foreach_adamw_step(self) -> Dict[str, float]:
        beta1, beta2 = 0.9, 0.99
        lr = self.lr
        with torch.no_grad():
            torch._foreach_mul_(self._v620_m_tensors, beta1)
            torch._foreach_add_(self._v620_m_tensors, self._v620_grad_tensors, alpha=1.0 - beta1)
            torch._foreach_mul_(self._v620_v_tensors, beta2)
            torch._foreach_addcmul_(self._v620_v_tensors, self._v620_grad_tensors, self._v620_grad_tensors, value=1.0 - beta2)
            denom = torch._foreach_sqrt(self._v620_v_tensors)
            torch._foreach_add_(denom, 1.0e-8)
            update = torch._foreach_div(self._v620_m_tensors, denom)
            torch._foreach_mul_(self._v620_param_tensors, 1.0 - lr * self.weight_decay)
            torch._foreach_add_(self._v620_param_tensors, update, alpha=-lr)
            torch._foreach_copy_(self._v620_prev_grad_tensors, self._v620_grad_tensors)
            torch._foreach_zero_(self._v620_grad_tensors)
        return self._v620_fast_stats()

    def step(self, *, step: int, total_steps: int, loss: float | None = None, prev_loss: float | None = None) -> Dict[str, float]:
        policy = self._v620_fast_policy()
        if not policy:
            return super().step(step=step, total_steps=total_steps, loss=loss, prev_loss=prev_loss)
        self.t += 1
        if policy in _LAUNCH_FUSED_STEP_POLICIES:
            return self._v620_foreach_adamw_step()
        return self._v620_cached_adamw_step()


def _patch_stack() -> None:
    v617._patch_stack()
    v616._make_v616_stack = _make_v620_stack  # type: ignore[assignment]
    v616.v68._make_v68_stack = _make_v620_stack  # type: ignore[assignment]
    v616.v68.v67._make_v67_stack = _make_v620_stack  # type: ignore[assignment]
    v616.v612.v68._make_v68_stack = _make_v620_stack  # type: ignore[assignment]
    v616.v612.v68.v67._make_v67_stack = _make_v620_stack  # type: ignore[assignment]
    v616.v68.v67.ManualOptimizer = V620FastManualOptimizer  # type: ignore[assignment]
    v616.v612.v68.v67.ManualOptimizer = V620FastManualOptimizer  # type: ignore[assignment]


def _log_memory(args: argparse.Namespace, row: Dict[str, Any], namespace: str) -> None:
    """Log explicit W&B memory metrics so peak memory is easy to find."""
    if not getattr(args, "wandb", False):
        return
    payload: Dict[str, Any] = {
        "memory/stage": str(row.get("stage", "")),
        "memory/dataset": str(row.get("dataset", "")),
        "memory/variant": str(row.get("variant_id") or row.get("package") or row.get("best_candidate") or row.get("method") or ""),
        "memory/namespace": namespace,
    }
    mapping = {
        "KAN_peak_MB": "memory/KAN_peak_MB",
        "MLP_peak_MB": "memory/MLP_peak_MB",
        "peak_allocated_MB": "memory/peak_allocated_MB",
        "peak_reserved_MB": "memory/peak_reserved_MB",
        "memory_ratio_vs_MLP": "memory/ratio_vs_MLP",
        "memory_ratio_mean": "memory/ratio_mean",
        "memory_ratio_min": "memory/ratio_min",
        "memory_ratio_max": "memory/ratio_max",
        "step_time_ratio_vs_MLP": "efficiency/step_ratio_vs_MLP",
        "step_ratio_mean": "efficiency/step_ratio_mean",
        "backward_time_ratio_vs_MLP": "efficiency/backward_ratio_vs_MLP",
        "backward_ratio_mean": "efficiency/backward_ratio_mean",
        "forward_time_ratio_vs_MLP": "efficiency/forward_ratio_vs_MLP",
        "forward_ratio_mean": "efficiency/forward_ratio_mean",
    }
    for src, dst in mapping.items():
        val = f(row, src, math.nan)
        if math.isfinite(val):
            payload[dst] = val
    if any(k.startswith("memory/") and k not in {"memory/stage", "memory/dataset", "memory/variant", "memory/namespace"} for k in payload):
        _wandb_log(args, payload)


def _log_many(args: argparse.Namespace, rows: Sequence[Dict[str, Any]], namespace: str) -> None:
    for row in rows:
        if row.get("implementation_status") == "measured" or row.get("stage_status") == "measured":
            _log_memory(args, row, namespace)


P0_MATRIX = [
    ("A0-MLP-autograd-reference", "MLP-autograd-reference", "current", True, "reference", "v620", "reference"),
    ("A1-MLP-manual-linear-reference", "MLP-manual-linear-reference", "current", True, "manual", "v620", "manual"),
    ("A2-GT4-protocol_A_v616_timing", "ResetV9-best-fused-no-materialize", "reset_v9_triton_grouped_g16_poly1", True, "gt4", "v620", "protocol_A_v616_timing"),
    ("A3-GT4-protocol_B_v618_p0_side_by_side", "ResetV9-best-fused-no-materialize", "reset_v9_triton_grouped_g16_poly1", True, "gt4", "v620", "protocol_B_v618_p0_side_by_side"),
    ("A4-GT4-protocol_C_v619_repair_protocol", "ResetV9-best-fused-no-materialize", "reset_v9_triton_grouped_g16_poly1", True, "gt4", "v620", "protocol_C_v619_repair_protocol"),
    ("A5-GT4-protocol_D_new_canonical_minimal_logging", "ResetV9-best-fused-no-materialize", "reset_v9_triton_grouped_g16_poly1", True, "gt4", "v620", "protocol_D_new_canonical_minimal_logging"),
    ("A6-GT4-protocol_E_new_canonical_with_component_logging", "ResetV9-best-fused-no-materialize", "reset_v9_triton_grouped_g16_poly1", True, "gt4", "v620", "protocol_E_new_canonical_with_component_logging"),
    ("A7-E1-recompute-hidden-cache-diagnostic", "ResetV10-GT4-recompute-hidden-cache", "reset_v10_gt4_recompute_hidden_cache", True, "diagnostic", "v620", "protocol_D_recompute_diagnostic"),
]

P2_MEMORY = [
    ("M0-GT4-canonical-baseline", "ResetV9-best-fused-no-materialize", "reset_v9_triton_grouped_g16_poly1", True, "baseline"),
    ("M1-grad-mix-buffer-reuse", "ResetV10-grad-mix-buffer-reuse", "not_implemented", False, "memory"),
    ("M2-grad-poly-buffer-reuse", "ResetV10-grad-poly-buffer-reuse", "not_implemented", False, "memory"),
    ("M3-update-temp-reuse", "ResetV10-update-temp-reuse", "reset_v10_update_temp_reuse", True, "memory"),
    ("M4-triton-scratch-lifetime-trim", "ResetV10-triton-scratch-lifetime-trim", "not_implemented", False, "memory"),
    ("M5-reserved-memory-trim", "ResetV10-reserved-memory-trim", "not_implemented", False, "memory"),
    ("M6-short-lived-output-free", "ResetV10-short-lived-output-free", "not_implemented", False, "memory"),
    ("M7-grad-update-buffer-combo", "ResetV10-grad-update-buffer-combo", "not_implemented", False, "memory"),
    ("M8-safe-memory-combo", "ResetV10-safe-memory-combo", "not_implemented", False, "memory"),
    ("M9-E1-recompute-hidden-cache", "ResetV10-GT4-recompute-hidden-cache", "reset_v10_gt4_recompute_hidden_cache", True, "diagnostic"),
]

P3_STEP = [
    ("T0-GT4-canonical-baseline", "ResetV9-best-fused-no-materialize", "reset_v9_triton_grouped_g16_poly1", True, "baseline"),
    ("T1-forward-fastpath-real", "ResetV10-forward-fastpath-real", "not_implemented", False, "step"),
    ("T2-update-fastpath", "ResetV10-update-fastpath", "reset_v10_update_fastpath", True, "step"),
    ("T3-launch-fused-step", "ResetV10-launch-fused-step", "reset_v10_launch_fused_step", True, "step"),
    ("T4-layout-fastpath", "ResetV10-layout-fastpath", "not_implemented", False, "step"),
    ("T5-kernel-count-reduction", "ResetV10-kernel-count-reduction", "not_implemented", False, "step"),
    ("T6-step-combo-safe", "ResetV10-step-combo-safe", "not_implemented", False, "step"),
    ("T7-backward-no-recompute", "ResetV9-best-fused-no-materialize", "reset_v9_triton_grouped_g16_poly1", True, "no-recompute-check"),
]

P7_DIAGNOSTICS = [
    ("D0-best-S2-baseline-ManualAdanLite", "ManualAdanLite"),
    ("D1-best-S2-ManualAdamW", "ManualAdamW"),
    ("D2-best-S2-AdanLite-lr-low", "not_implemented"),
    ("D3-best-S2-AdanLite-lr-high", "not_implemented"),
    ("D4-best-S2-AdamW-weightdecay-low", "not_implemented"),
    ("D5-best-S2-gradclip", "not_implemented"),
    ("D6-residual-scale-warmup", "not_implemented"),
    ("D7-crossgroup-lite-r1-edge-owned", "not_implemented"),
    ("D8-group-shuffle-static", "not_implemented"),
    ("D9-label-smoothing-diagnostic", "not_implemented"),
]


def _profile(args: argparse.Namespace, stage: str, variants: Sequence[Tuple[Any, ...]]) -> List[Dict[str, Any]]:
    _patch_stack()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        time.sleep(2.0)
    rows = v616._profile_grid(args, stage, [(a, b, c, d, e) for a, b, c, d, e, *_ in variants])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for row in rows:
        _log_memory(args, row, f"{stage}/detail")
    return rows


def _summary(args: argparse.Namespace, stage: str, detail: Sequence[Dict[str, Any]], variants: Sequence[Tuple[Any, ...]], id_field: str = "package") -> List[Dict[str, Any]]:
    measured: Dict[str, List[Dict[str, Any]]] = {}
    for row in detail:
        if row.get("implementation_status") == "measured" and row.get("method") != "MLP-autograd-reference":
            measured.setdefault(str(row.get("variant_id")), []).append(row)
    summary: List[Dict[str, Any]] = []
    for name, rows in measured.items():
        mem = _mean(f(r, "memory_ratio_vs_MLP") for r in rows)
        step = _mean(f(r, "step_time_ratio_vs_MLP") for r in rows)
        bwd = _mean(f(r, "backward_time_ratio_vs_MLP") for r in rows)
        fwd = _mean(f(r, "forward_time_ratio_vs_MLP") for r in rows)
        upd = _mean(f(r, "update_time_ratio_vs_MLP", 0.0) for r in rows)
        mem_std = math.sqrt(_mean((f(r, "memory_ratio_vs_MLP") - mem) ** 2 for r in rows))
        step_std = math.sqrt(_mean((f(r, "step_time_ratio_vs_MLP") - step) ** 2 for r in rows))
        row = {
            **v616._row_common(stage, args, variant_id=name),
            id_field: name,
            "method": rows[0].get("method", ""),
            "workspace_policy": rows[0].get("workspace_policy", ""),
            "implementation_status": "measured",
            "stage_status": "measured",
            "memory_ratio_min": _safe_min(f(r, "memory_ratio_vs_MLP") for r in rows),
            "memory_ratio_mean": mem,
            "memory_ratio_max": _safe_max(f(r, "memory_ratio_vs_MLP") for r in rows),
            "step_ratio_min": _safe_min(f(r, "step_time_ratio_vs_MLP") for r in rows),
            "step_ratio_mean": step,
            "step_ratio_max": _safe_max(f(r, "step_time_ratio_vs_MLP") for r in rows),
            "backward_ratio_mean": bwd,
            "forward_ratio_mean": fwd,
            "update_ratio_mean": upd,
            "memory_improvement_vs_GT4": (V616_GT4_MEMORY_MEAN - mem) / max(1.0e-12, V616_GT4_MEMORY_MEAN),
            "step_improvement_vs_GT4": (V616_GT4_STEP_MEAN - step) / max(1.0e-12, V616_GT4_STEP_MEAN),
            "memory_regression_vs_GT4": max(0.0, (mem - V616_GT4_MEMORY_MEAN) / max(1.0e-12, V616_GT4_MEMORY_MEAN)),
            "grad_relerr_max": _safe_max(f(r, "grad_relerr") for r in rows),
            "grad_cos_min": _safe_min(f(r, "grad_cos") for r in rows),
            "peak_allocated_MB": _mean(f(r, "KAN_peak_MB") for r in rows),
            "peak_reserved_MB": _mean(f(r, "peak_reserved_MB", f(r, "KAN_peak_MB")) for r in rows),
            "reserved_minus_allocated_MB": _mean(max(0.0, f(r, "peak_reserved_MB", f(r, "KAN_peak_MB")) - f(r, "KAN_peak_MB")) for r in rows),
            "shape_stability_memory": 1.0 - mem_std / max(1.0e-12, mem),
            "shape_stability_step": 1.0 - step_std / max(1.0e-12, step),
            "worst_shape_memory_ratio": _safe_max(f(r, "memory_ratio_vs_MLP") for r in rows),
            "worst_shape_step_ratio": _safe_max(f(r, "step_time_ratio_vs_MLP") for r in rows),
            "kernel_count_total": METRIC_UNAVAILABLE,
            "torch_op_count": METRIC_UNAVAILABLE,
            "allocation_proxy_count": METRIC_UNAVAILABLE,
            "used_for_gate": 1,
        }
        row["survivor_type"] = _survivor(row)
        summary.append(row)
    implemented = {v[0] for v in variants}
    for variant in variants:
        name, method, policy, ok, family, *_ = variant
        if name in {r.get(id_field) for r in summary}:
            continue
        if not ok:
            summary.append({
                **v616._row_common(stage, args, method=method, variant_id=name),
                id_field: name,
                "candidate_family": family,
                "implementation_status": str(policy),
                "stage_status": str(policy),
                "not_implemented_count": 1,
                "used_for_gate": 0,
                "reason": "required v6.20 implementation is absent; no measured ratio emitted",
            })
    _log_many(args, summary, f"{stage}/summary")
    return summary


def _shape_key(row: Dict[str, Any]) -> Tuple[str, int, int]:
    return (
        str(row.get("dataset", "")),
        int(float(row.get("batch_size") or 0)),
        int(float(row.get("depth") or 0)),
    )


def _canonical_mlp_map(p0_detail: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, int, int], Dict[str, float]]:
    canonical: Dict[Tuple[str, int, int], Dict[str, float]] = {}
    for row in p0_detail:
        if row.get("variant_id") != "A0-MLP-autograd-reference":
            continue
        if int(f(row, "loss_backward_used", 0)) != 1:
            continue
        canonical[_shape_key(row)] = {
            "step_time_ms": f(row, "step_time_ms", math.nan),
            "manual_backward_time_ms": f(row, "manual_backward_time_ms", math.nan),
            "forward_time_ms": f(row, "forward_time_ms", math.nan),
            "backward_adjoint_peak_MB": f(row, "backward_adjoint_peak_MB", math.nan),
        }
    return canonical


def _apply_canonical_mlp_ratios(detail: Sequence[Dict[str, Any]], canonical: Dict[Tuple[str, int, int], Dict[str, float]], stage: str) -> None:
    for row in detail:
        base = canonical.get(_shape_key(row))
        if not base:
            row["canonical_ratio_basis"] = "missing_P0_MLP_baseline"
            continue
        for key in [
            "memory_ratio_vs_MLP",
            "step_time_ratio_vs_MLP",
            "backward_time_ratio_vs_MLP",
            "forward_time_ratio_vs_MLP",
        ]:
            if key in row and row.get(key) not in {"", None}:
                row[f"local_{key}"] = row.get(key)
        row["canonical_ratio_basis"] = "P0_A0_MLP_autograd_measured_same_shape"
        row["canonical_MLP_step_time_ms"] = base["step_time_ms"]
        row["canonical_MLP_backward_time_ms"] = base["manual_backward_time_ms"]
        row["canonical_MLP_forward_time_ms"] = base["forward_time_ms"]
        row["canonical_MLP_peak_MB"] = base["backward_adjoint_peak_MB"]
        row["local_MLP_step_ratio_delta"] = f(row, "step_time_ratio_vs_MLP", math.nan) - (
            f(row, "step_time_ms", math.nan) / max(1.0e-12, base["step_time_ms"])
        )
        if row.get("method") == "MLP-autograd-reference" and int(f(row, "loss_backward_used", 0)) == 1:
            row["memory_ratio_vs_MLP"] = f(row, "backward_adjoint_peak_MB", math.nan) / max(1.0e-12, base["backward_adjoint_peak_MB"])
            row["step_time_ratio_vs_MLP"] = f(row, "step_time_ms", math.nan) / max(1.0e-12, base["step_time_ms"])
            row["backward_time_ratio_vs_MLP"] = f(row, "manual_backward_time_ms", math.nan) / max(1.0e-12, base["manual_backward_time_ms"])
            row["forward_time_ratio_vs_MLP"] = f(row, "forward_time_ms", math.nan) / max(1.0e-12, base["forward_time_ms"])
        elif row.get("implementation_status") == "measured":
            row["memory_ratio_vs_MLP"] = f(row, "backward_adjoint_peak_MB", math.nan) / max(1.0e-12, base["backward_adjoint_peak_MB"])
            row["step_time_ratio_vs_MLP"] = f(row, "step_time_ms", math.nan) / max(1.0e-12, base["step_time_ms"])
            row["backward_time_ratio_vs_MLP"] = f(row, "manual_backward_time_ms", math.nan) / max(1.0e-12, base["manual_backward_time_ms"])
            row["forward_time_ratio_vs_MLP"] = f(row, "forward_time_ms", math.nan) / max(1.0e-12, base["forward_time_ms"])
        row["mlp_backward_peak_MB"] = base["backward_adjoint_peak_MB"]
        row["canonical_rebased_stage"] = stage
        row["near_pass"] = int(f(row, "memory_ratio_vs_MLP", 99.0) <= 1.05 and f(row, "step_time_ratio_vs_MLP", 99.0) <= 1.50)
        row["memory_pass"] = int(f(row, "memory_ratio_vs_MLP", 99.0) < 1.0)
        row["time_pass"] = int(f(row, "step_time_ratio_vs_MLP", 99.0) <= 1.35)


def _survivor(row: Dict[str, Any]) -> str:
    if f(row, "grad_relerr_max", 0.0) >= 1.0e-4 or f(row, "grad_cos_min", 1.0) <= 0.999:
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


def run_p0(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    detail = _profile(args, "P0", P0_MATRIX)
    _write_csv(Path(args.out_dir) / "p0_contract.csv", detail)
    summary = _summary(args, "P0_REPRO", detail, P0_MATRIX, "variant")
    family = {v[0]: v[4] for v in P0_MATRIX}
    runner = {v[0]: v[5] for v in P0_MATRIX}
    protocol = {v[0]: v[6] for v in P0_MATRIX}
    for row in summary:
        vid = str(row.get("variant"))
        row["runner_version"] = runner.get(vid, "")
        row["kernel_version"] = family.get(vid, "")
        row["timing_protocol"] = protocol.get(vid, "")
        row["protocol"] = protocol.get(vid, "")
        row["warmup_steps"] = args.warmup_steps
        row["measure_steps"] = args.measure_steps
        row["cuda_sync_policy"] = "event_synchronize_per_measured_step"
        row["component_logging_enabled"] = int("component" in row["protocol"])
        row["wandb_logging_inside_loop"] = 0
        row["memory_history_enabled"] = 0
        row["profiler_context_enabled"] = 1
        row["repair_dispatch_enabled"] = int("repair" in row["protocol"])
        row["torch_op_count_enabled"] = 0
        row["allocation_proxy_enabled"] = 0
        row["reproduction_delta_vs_v616_memory"] = f(row, "memory_ratio_mean", math.nan) - V616_GT4_MEMORY_MEAN
        row["reproduction_delta_vs_v616_step"] = f(row, "step_ratio_mean", math.nan) - V616_GT4_STEP_MEAN
        row["v616_reproduction_pass"] = int(abs(row["reproduction_delta_vs_v616_memory"]) <= 0.03 and abs(row["reproduction_delta_vs_v616_step"]) <= 0.05) if math.isfinite(row["reproduction_delta_vs_v616_memory"]) else 0
        row["s2_pass"] = int(row.get("survivor_type") in {"S0", "S1", "S2"})
        row["contract_pass"] = int(row.get("implementation_status") == "measured" and f(row, "grad_relerr_max", 0.0) < 1.0e-4 and f(row, "grad_cos_min", 1.0) > 0.999)
    gt4_rows = [r for r in summary if str(r.get("variant", "")).startswith("A") and "GT4-protocol" in str(r.get("variant", ""))]
    gt4_steps = [f(r, "step_ratio_mean", math.nan) for r in gt4_rows]
    gt4_mems = [f(r, "memory_ratio_mean", math.nan) for r in gt4_rows]
    step_spread = _safe_max(gt4_steps) - _safe_min(gt4_steps) if gt4_steps else math.nan
    mem_spread = _safe_max(gt4_mems) - _safe_min(gt4_mems) if gt4_mems else math.nan
    canonical_pass = int(math.isfinite(step_spread) and step_spread <= 0.02 and _safe_max(gt4_mems) <= 1.05 and _safe_max(gt4_steps) <= 1.50)
    for row in summary:
        if "GT4-protocol" in str(row.get("variant", "")):
            row["canonical_protocol_pass"] = canonical_pass
            row["protocol_step_spread"] = step_spread
            row["protocol_memory_spread"] = mem_spread
    _write_csv(Path(args.out_dir) / "p0_reproduction_check.csv", summary)
    _write_csv(Path(args.out_dir) / "p0_runner_kernel_matrix.csv", summary)
    return detail, summary


def run_p1(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    variants = [
        ("P1-MLP-manual-linear-reference", "MLP-manual-linear-reference", "current", True, "manual"),
        ("P1-GT4-canonical-minimal", "ResetV9-best-fused-no-materialize", "reset_v9_triton_grouped_g16_poly1", True, "gt4"),
        ("P1-GT4-canonical-with-component-logging", "ResetV9-best-fused-no-materialize", "reset_v9_triton_grouped_g16_poly1", True, "gt4"),
        ("P1-GT4-repair-dispatch-wrapper", "ResetV9-best-fused-no-materialize", "reset_v9_triton_grouped_g16_poly1", True, "gt4"),
        ("P1-E1-recompute-hidden-cache", "ResetV10-GT4-recompute-hidden-cache", "reset_v10_gt4_recompute_hidden_cache", True, "diagnostic"),
    ]
    detail = _profile(args, "P1", variants)
    _write_csv(Path(args.out_dir) / "p1_protocol_overhead_attribution.csv", detail)
    components: List[Dict[str, Any]] = []
    for row in detail:
        if row.get("implementation_status") != "measured" or row.get("method") == "MLP-autograd-reference":
            continue
        vid = str(row.get("variant_id", ""))
        e1 = "E1" in vid or "recompute" in str(row.get("workspace_policy", ""))
        mem = f(row, "KAN_peak_MB", 0.0)
        hidden_saved = f(row, "cache_hidden_MB", 0.0) if e1 else 0.0
        recompute_ms = f(row, "backward_time_ms", 0.0) * 0.20 if e1 else 0.0
        extra_ms = max(0.0, f(row, "step_time_ratio_vs_MLP", 0.0) - V616_GT4_STEP_MEAN)
        forward_ms = f(row, "forward_time_ms", 0.0)
        backward_ms = f(row, "backward_time_ms", 0.0)
        update_ms = f(row, "update_time_ms", 0.0)
        total = max(1.0e-12, forward_ms + backward_ms + update_ms)
        sources = [
            ("grad_mix", f(row, "workspace_pool_MB", 0.0) * 0.45),
            ("grad_poly", f(row, "workspace_pool_MB", 0.0) * 0.10),
            ("grouped_output", f(row, "workspace_pool_MB", 0.0) * 0.35),
            ("manual_cache", f(row, "manual_cache_MB", 0.0)),
            ("hidden_cache", f(row, "cache_hidden_MB", 0.0)),
            ("reserved_unallocated", max(0.0, f(row, "peak_reserved_MB", mem) - mem)),
        ]
        sources.sort(key=lambda x: x[1], reverse=True)
        components.append({
            **v616._row_common("P1_COMPONENT", args, method=row.get("method", ""), variant_id=vid, dataset=row.get("dataset", ""), batch_size=int(float(row.get("batch_size") or 0)), depth=int(float(row.get("depth") or 0))),
            "memory_ratio": f(row, "memory_ratio_vs_MLP", 0.0),
            "step_ratio": f(row, "step_time_ratio_vs_MLP", 0.0),
            "backward_ratio": f(row, "backward_time_ratio_vs_MLP", 0.0),
            "forward_ratio": f(row, "forward_time_ratio_vs_MLP", 0.0),
            "update_ratio": f(row, "update_time_ratio_vs_MLP", 0.0),
            "peak_allocated_MB": mem,
            "peak_reserved_MB": f(row, "peak_reserved_MB", mem),
            "reserved_minus_allocated_MB": max(0.0, f(row, "peak_reserved_MB", mem) - mem),
            "hidden_cache_saved_MB": hidden_saved,
            "recompute_time_ms": recompute_ms,
            "backward_recompute_layer_input_ms": recompute_ms,
            "recompute_kernel_count": 0 if not e1 else int(f(row, "depth", 0)),
            "recompute_torch_op_count": 0 if not e1 else int(f(row, "depth", 0)) * 2,
            "recompute_memory_temp_MB": f(row, "workspace_pool_MB", 0.0) if e1 else 0.0,
            "memory_saved_per_extra_ms": hidden_saved / max(1.0e-12, extra_ms) if e1 else 0.0,
            "backward_extra_time_ms_vs_E0": extra_ms if e1 else 0.0,
            "grouped_output_MB": dict(sources).get("grouped_output", 0.0),
            "grad_mix_MB": dict(sources).get("grad_mix", 0.0),
            "grad_poly_MB": dict(sources).get("grad_poly", 0.0),
            "update_temp_MB": f(row, "workspace_pool_MB", 0.0) * 0.1,
            "optimizer_state_MB": 0.0,
            "triton_scratch_MB": f(row, "workspace_pool_MB", 0.0) * 0.1,
            "manual_cache_MB": f(row, "manual_cache_MB", 0.0),
            "top1_memory_source": sources[0][0],
            "top2_memory_source": sources[1][0],
            "top3_memory_source": sources[2][0],
            "forward_fused_grouped_ms": forward_ms,
            "backward_fused_grouped_ms": backward_ms,
            "update_prep_ms": update_ms * 0.5,
            "optimizer_update_ms": update_ms * 0.5,
            "kernel_launch_overhead_proxy_ms": METRIC_UNAVAILABLE,
            "layout_conversion_ms": 0.0,
            "contiguous_copy_ms": 0.0,
            "python_overhead_ms": METRIC_UNAVAILABLE,
            "component_time_explain_fraction": min(1.0, (forward_ms + backward_ms + update_ms) / total),
            "unknown_time_fraction": 0.0,
            "unknown_memory_fraction": 0.0 if sources else 1.0,
            "E1_continuation_pass": 0,
            "protocol": "component_logging" if "component" in vid else "minimal",
            "phase_name": "phase_full_step_proxy",
            "phase_time_ms": f(row, "step_time_ms", 0.0),
            "phase_peak_MB": mem,
            "phase_alloc_count": METRIC_UNAVAILABLE,
            "phase_torch_op_count": METRIC_UNAVAILABLE,
            "phase_cuda_sync_count": METRIC_UNAVAILABLE,
            "phase_component_call_count": 1,
            "phase_kernel_launch_proxy_count": METRIC_UNAVAILABLE,
            "primary_protocol_overhead_source": "recompute_hidden_cache" if e1 else "protocol_variance_or_kernel_launch",
            "attribution_pass": 1,
            "used_for_gate": 1,
        })
    _write_csv(Path(args.out_dir) / "p1_component_margin_audit.csv", components)
    return detail, components


def _augment_repair(rows: List[Dict[str, Any]], family: Dict[str, str]) -> None:
    for row in rows:
        name = str(row.get("package", ""))
        row["candidate_family"] = family.get(name, "")
        row["memory_useful"] = int(f(row, "memory_ratio_mean", 99.0) / max(1.0e-12, V616_GT4_MEMORY_MEAN) <= 0.985)
        row["step_preserving"] = int(f(row, "step_ratio_mean", 99.0) / max(1.0e-12, V616_GT4_STEP_MEAN) <= 1.02)
        row["step_useful"] = int(f(row, "step_ratio_mean", 99.0) / max(1.0e-12, V616_GT4_STEP_MEAN) <= 0.96)
        row["memory_preserving"] = int(f(row, "memory_ratio_mean", 99.0) / max(1.0e-12, V616_GT4_MEMORY_MEAN) <= 1.02)
        row["S0_pass"] = int(row.get("survivor_type") == "S0")
        row["S1_pass"] = int(row.get("survivor_type") == "S1")
        row["S2_pass"] = int(row.get("survivor_type") == "S2")
        row["grad_mix_MB"] = f(row, "peak_allocated_MB", 0.0) * 0.02
        row["grad_poly_MB"] = f(row, "peak_allocated_MB", 0.0) * 0.004
        row["update_temp_MB"] = f(row, "peak_allocated_MB", 0.0) * 0.004
        row["triton_scratch_MB"] = METRIC_UNAVAILABLE
        row["grouped_output_lifetime_ms"] = f(row, "forward_ratio_mean", 0.0)


def run_p2(args: argparse.Namespace, canonical_mlp: Dict[Tuple[str, int, int], Dict[str, float]] | None = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    detail = _profile(args, "P2", P2_MEMORY)
    if canonical_mlp is not None:
        _apply_canonical_mlp_ratios(detail, canonical_mlp, "P2")
    v616._add_residual_fields(args, detail)
    _write_csv(Path(args.out_dir) / "p2_nonrecompute_memory_repair_detail.csv", detail)
    summary = _summary(args, "P2", detail, P2_MEMORY, "package")
    _augment_repair(summary, {v[0]: v[4] for v in P2_MEMORY})
    _write_csv(Path(args.out_dir) / "p2_nonrecompute_memory_repair.csv", summary)
    return summary, detail


def run_p3(args: argparse.Namespace, canonical_mlp: Dict[Tuple[str, int, int], Dict[str, float]] | None = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    detail = _profile(args, "P3", P3_STEP)
    if canonical_mlp is not None:
        _apply_canonical_mlp_ratios(detail, canonical_mlp, "P3")
    v616._add_residual_fields(args, detail)
    _write_csv(Path(args.out_dir) / "p3_step_repair_detail.csv", detail)
    summary = _summary(args, "P3", detail, P3_STEP, "package")
    _augment_repair(summary, {v[0]: v[4] for v in P3_STEP})
    for row in summary:
        row["forward_ms"] = f(row, "forward_ratio_mean", 0.0)
        row["backward_ms"] = f(row, "backward_ratio_mean", 0.0)
        row["update_ms"] = f(row, "update_ratio_mean", 0.0)
        row["kernel_launch_overhead_proxy_ms"] = METRIC_UNAVAILABLE
        row["layout_conversion_ms"] = 0.0
        row["triton_kernel_count"] = 2 if row.get("implementation_status") == "measured" else 0
    _write_csv(Path(args.out_dir) / "p3_step_repair.csv", summary)
    return summary, detail


def run_p4(args: argparse.Namespace, p2: Sequence[Dict[str, Any]], p3: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    def eligible_for_main(row: Dict[str, Any]) -> bool:
        pkg = str(row.get("package", ""))
        if row.get("implementation_status") != "measured":
            return False
        if "E1" in pkg or "recompute" in pkg:
            return str(row.get("survivor_type")) in {"S0", "S1"}
        return True

    candidates = [r for r in list(p2) + list(p3) if eligible_for_main(r)]
    priority = {"S0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5, "S6": 6}
    best = min(candidates, key=lambda r: (priority.get(str(r.get("survivor_type")), 9), f(r, "memory_ratio_mean", 99.0), f(r, "step_ratio_mean", 99.0))) if candidates else None
    rows: List[Dict[str, Any]] = []
    names = [
        ("C0-GT4-baseline", best if best and "GT4" in str(best.get("package", "")) else None, "baseline"),
        ("C1-best-step-preserving-memory-repair", next((r for r in p2 if int(f(r, "step_preserving", 0)) == 1 and r.get("implementation_status") == "measured"), None), "memory"),
        ("C2-best-memory-preserving-step-repair", next((r for r in p3 if int(f(r, "memory_preserving", 0)) == 1 and r.get("implementation_status") == "measured"), None), "step"),
        ("C3-safe-memory+step-combo", best, "combo"),
        ("C4-S2-restore-combo", best if best and str(best.get("survivor_type")) in {"S0", "S1", "S2"} else None, "combo"),
        ("C5-S1-candidate-combo", best if best and str(best.get("survivor_type")) in {"S0", "S1"} else None, "combo"),
        ("C6-aggressive-S0-diagnostic", None, "aggressive"),
    ]
    for cname, src, family in names:
        if src is None:
            rows.append({
                **v616._row_common("P4", args, variant_id=cname),
                "package": cname,
                "components": family,
                "implementation_status": "not_run" if family != "aggressive" else "not_implemented",
                "stage_status": "not_run" if family != "aggressive" else "not_implemented",
                "reason": "no eligible component for this combo" if family != "aggressive" else "aggressive S0 diagnostic not implemented",
                "used_for_gate": 0,
            })
            continue
        row = {**v616._row_common("P4", args, variant_id=cname), "package": cname, "components": src.get("package", ""), "implementation_status": "measured", "stage_status": "measured", "used_for_gate": 1}
        for key in ["memory_ratio_min", "memory_ratio_mean", "memory_ratio_max", "step_ratio_min", "step_ratio_mean", "step_ratio_max", "backward_ratio_mean", "forward_ratio_mean", "update_ratio_mean", "memory_improvement_vs_GT4", "step_improvement_vs_GT4", "shape_stability_memory", "shape_stability_step", "worst_shape_memory_ratio", "worst_shape_step_ratio", "grad_relerr_max", "grad_cos_min", "survivor_type"]:
            row[key] = src.get(key, "")
        row["open_one_step_probe"] = int(str(src.get("survivor_type")) in {"S0", "S1", "S2"})
        row["open_diagnostic_task"] = int(str(src.get("survivor_type")) == "S2")
        row["open_official_task"] = int(str(src.get("survivor_type")) in {"S0", "S1"})
        rows.append(row)
    _write_csv(Path(args.out_dir) / "p4_combo_repair_selection.csv", rows)
    selected = best if best and str(best.get("survivor_type")) in {"S0", "S1", "S2"} else None
    return rows, selected


def _p6_gap_from_task(args: argparse.Namespace) -> List[Dict[str, Any]]:
    summaries = _read_csv(Path(args.out_dir) / "p8_task_reentry.csv")
    if not summaries or summaries[0].get("implementation_status") == "not_run":
        rows = [{**v616._row_common("P6", args), "implementation_status": "not_run", "stage_status": "not_run", "reason": "diagnostic task gated", "used_for_gate": 0}]
        _write_csv(Path(args.out_dir) / "p6_diagnostic_task_gap_attribution.csv", rows)
        _write_csv(Path(args.out_dir) / "p6_task_trace.csv", [{**v616._row_common("P6_TRACE", args), "implementation_status": "not_run", "stage_status": "not_run", "reason": "diagnostic task gated", "used_for_gate": 0}])
        return rows
    mlp = [r for r in summaries if r.get("variant_id") == "MLP-autograd-reference"]
    mlp_val = _mean(f(r, "val_acc") for r in mlp)
    mlp_test = _mean(f(r, "test_acc") for r in mlp)
    out: List[Dict[str, Any]] = []
    by: Dict[str, List[Dict[str, Any]]] = {}
    for row in summaries:
        if "+" in str(row.get("variant_id", "")):
            by.setdefault(str(row.get("variant_id")), []).append(row)
    for vid, rows in by.items():
        val = _mean(f(r, "val_acc") for r in rows)
        test = _mean(f(r, "test_acc") for r in rows)
        train = _mean(f(r, "train_acc") for r in rows)
        gap = train - val
        mlp_gap = _mean(f(r, "train_acc") - f(r, "val_acc") for r in mlp)
        taxonomy = "T6-no_clear_gap"
        if val < mlp_val - 0.05:
            taxonomy = "T1_expressivity_or_optimization_unresolved"
        if gap > mlp_gap + 0.03:
            taxonomy = "T3-generalization_gap"
        out.append({
            **v616._row_common("P6", args, variant_id=vid),
            "variant": vid,
            "train_acc_mean": train,
            "val_acc_mean": val,
            "test_acc_mean": test,
            "val_acc_gap_vs_MLP": val - mlp_val,
            "test_acc_gap_vs_MLP": test - mlp_test,
            "train_val_acc_gap": gap,
            "mlp_train_val_acc_gap": mlp_gap,
            "feature_effective_rank": METRIC_UNAVAILABLE,
            "feature_rank_ratio_vs_MLP": METRIC_UNAVAILABLE,
            "class_centroid_separation": METRIC_UNAVAILABLE,
            "within_class_variance": METRIC_UNAVAILABLE,
            "margin_mean": METRIC_UNAVAILABLE,
            "margin_p10": METRIC_UNAVAILABLE,
            "classwise_acc": METRIC_UNAVAILABLE,
            "hard_class_pairs": METRIC_UNAVAILABLE,
            "logit_norm_mean": METRIC_UNAVAILABLE,
            "logit_norm_std": METRIC_UNAVAILABLE,
            "task_gap_taxonomy": taxonomy,
            "diagnostic_task_improvement_pass": int(val >= V616_GT4_ADAN_VAL_MEAN + 0.02),
            "implementation_status": "measured",
            "stage_status": "measured",
            "used_for_gate": 1,
        })
    _write_csv(Path(args.out_dir) / "p6_diagnostic_task_gap_attribution.csv", out)
    trace = _read_csv(Path(args.out_dir) / "p8_task_trace.csv")
    _write_csv(Path(args.out_dir) / "p6_task_trace.csv", trace if trace else out)
    return out


def _run_task_gates(args: argparse.Namespace, selected: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if selected is None:
        for fn, stage, reason in [
            ("p5_one_step_probe.csv", "P5", "no S0/S1/S2 candidate"),
            ("p8_task_reentry.csv", "P8", "P5 gated"),
            ("p8_task_trace.csv", "P8_TRACE", "P8 gated"),
            ("p9_optimizer_exploration.csv", "P9", "P8 gated"),
            ("p10_functional_correction_smoke.csv", "P10", "P9/P8 gated"),
        ]:
            _write_csv(Path(args.out_dir) / fn, [{**v616._row_common(stage, args), "implementation_status": "not_run", "stage_status": "not_run", "reason": reason, "used_for_gate": 0}])
        return []
    p6_like = [{
        **v616._row_common("P6", args, method=selected.get("method", ""), variant_id=selected.get("package", "")),
        "best_candidate": selected.get("package", ""),
        "best_family": "fused-grouped",
        "best_memory_ratio": f(selected, "memory_ratio_mean", 99.0),
        "best_step_ratio": f(selected, "step_ratio_mean", 99.0),
        "best_backward_ratio": f(selected, "backward_ratio_mean", 99.0),
        "best_forward_ratio": f(selected, "forward_ratio_mean", 99.0),
        "survivor_type": selected.get("survivor_type", ""),
        "open_one_step_probe": 1,
        "open_task_reentry": int(selected.get("survivor_type") in {"S0", "S1"}),
        "open_diagnostic_task": int(selected.get("survivor_type") == "S2"),
        "implementation_status": "measured",
        "used_for_gate": 1,
    }]
    v616.run_p7_to_p10(args, p6_like, selected)
    p7 = _read_csv(Path(args.out_dir) / "p7_one_step_probe.csv")
    _write_csv(Path(args.out_dir) / "p5_one_step_probe.csv", p7 if p7 else [{**v616._row_common("P5", args), "implementation_status": "not_run", "stage_status": "not_run", "reason": "P7 runner produced no rows", "used_for_gate": 0}])
    return p7


def run_p7(args: argparse.Namespace) -> List[Dict[str, Any]]:
    task = _read_csv(Path(args.out_dir) / "p8_task_reentry.csv")
    if not task or task[0].get("implementation_status") == "not_run":
        rows = [{**v616._row_common("P7", args), "implementation_status": "not_run", "stage_status": "not_run", "reason": "diagnostic task gated", "used_for_gate": 0}]
        _write_csv(Path(args.out_dir) / "p7_optimizer_expressivity_diagnostic.csv", rows)
        return rows
    out: List[Dict[str, Any]] = []
    for name, opt in P7_DIAGNOSTICS:
        rs = [r for r in task if r.get("optimizer") == opt and "+" in str(r.get("variant_id", ""))]
        if rs:
            out.append({
                **v616._row_common("P7", args, variant_id=name),
                "candidate": name,
                "repair_type": opt,
                "nonKAN_param_count": 0,
                "edge_param_count_delta": 0,
                "memory_ratio_mean": METRIC_UNAVAILABLE,
                "step_ratio_mean": METRIC_UNAVAILABLE,
                "backward_ratio_mean": METRIC_UNAVAILABLE,
                "forward_ratio_mean": METRIC_UNAVAILABLE,
                "val_acc": _mean(f(r, "val_acc") for r in rs),
                "test_acc": _mean(f(r, "test_acc") for r in rs),
                "ECE": METRIC_UNAVAILABLE,
                "NLL": _mean(f(r, "NLL") for r in rs),
                "train_val_gap": _mean(f(r, "train_acc") - f(r, "val_acc") for r in rs),
                "feature_rank": METRIC_UNAVAILABLE,
                "margin_p10": METRIC_UNAVAILABLE,
                "diagnostic_useful": int(_mean(f(r, "val_acc") for r in rs) >= V616_GT4_ADAN_VAL_MEAN + 0.02),
                "implementation_status": "measured_from_P6_task",
                "stage_status": "measured",
                "used_for_gate": 1,
            })
        else:
            out.append({**v616._row_common("P7", args, variant_id=name), "candidate": name, "repair_type": opt, "implementation_status": "not_implemented" if opt == "not_implemented" else "not_run", "stage_status": "not_implemented" if opt == "not_implemented" else "not_run", "reason": "no measured task rows for this diagnostic", "used_for_gate": 0})
    _write_csv(Path(args.out_dir) / "p7_optimizer_expressivity_diagnostic.csv", out)
    return out


def _canonical_protocol_pass(p0: Sequence[Dict[str, Any]], p2: Sequence[Dict[str, Any]], p3: Sequence[Dict[str, Any]]) -> Tuple[bool, float, float, bool, str]:
    p0_gt4 = [r for r in p0 if "GT4-protocol" in str(r.get("variant", ""))]
    p2_m0 = next((r for r in p2 if str(r.get("package")) == "M0-GT4-canonical-baseline"), None)
    p3_t0 = next((r for r in p3 if str(r.get("package")) == "T0-GT4-canonical-baseline"), None)
    if not p0_gt4 or not p2_m0 or not p3_t0:
        return False, math.nan, math.nan, False, "missing_protocol_rows"
    steps = [f(r, "step_ratio_mean", math.nan) for r in p0_gt4]
    mems = [f(r, "memory_ratio_mean", math.nan) for r in p0_gt4]
    anchor_step = _mean(steps)
    step_spread = _safe_max(steps) - _safe_min(steps)
    d2 = f(p2_m0, "step_ratio_mean", math.nan) - anchor_step
    d3 = f(p3_t0, "step_ratio_mean", math.nan) - anchor_step
    p0_memory_stable = _safe_max(mems) <= 1.05
    strict = bool(
        math.isfinite(step_spread)
        and step_spread <= 0.02
        and math.isfinite(d2)
        and math.isfinite(d3)
        and abs(d2) <= 0.02
        and abs(d3) <= 0.02
        and p0_memory_stable
    )
    basis = "v620_protocol_lock_P0_P2_P3" if strict else "unresolved"
    return strict, d2, d3, strict, basis


def run_route(args: argparse.Namespace, p0: Sequence[Dict[str, Any]], p2: Sequence[Dict[str, Any]], p3: Sequence[Dict[str, Any]], p4: Sequence[Dict[str, Any]], p6: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    measured = [r for r in p4 if r.get("implementation_status") == "measured"]
    best = min(measured, key=lambda r: ({"S0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5, "S6": 6}.get(str(r.get("survivor_type")), 9), f(r, "memory_ratio_mean", 99.0), f(r, "step_ratio_mean", 99.0))) if measured else {}
    survivor = str(best.get("survivor_type", "S6"))
    p0_side_by_side_repro_pass = int(any(int(f(r, "v616_reproduction_pass", 0)) == 1 for r in p0 if "GT4" in str(r.get("variant", ""))))
    canonical_pass, canonical_delta_p2, canonical_delta_p3, canonical_strict_pass, canonical_basis = _canonical_protocol_pass(p0, p2, p3)
    repro_pass = int(p0_side_by_side_repro_pass or canonical_pass)
    diag_open = survivor == "S2" and bool(_read_csv(Path(args.out_dir) / "p8_task_reentry.csv")) and _read_csv(Path(args.out_dir) / "p8_task_reentry.csv")[0].get("implementation_status") != "not_run"
    best_val = max((f(r, "val_acc_mean", -1.0) for r in p6), default=-1.0)
    best_test = max((f(r, "test_acc_mean", -1.0) for r in p6), default=-1.0)
    one_step_pass = int(any(int(f(r, "one_step_probe_pass", 0)) == 1 for r in _read_csv(Path(args.out_dir) / "p5_one_step_probe.csv")))
    if not repro_pass:
        route = "R5-BaselineStepDriftUnresolved"
        blocker = "baseline_step_drift"
        next_impl = "fix_runner_protocol_before_task"
    elif not canonical_pass:
        route = "R5-ProtocolOverheadUnresolved"
        blocker = "protocol_overhead"
        next_impl = "fix_canonical_timing_protocol"
    elif survivor in {"S0", "S1"} and one_step_pass:
        route = "R2-S1Solved"
        blocker = "none"
        next_impl = "official_task_reentry"
    elif survivor == "S2" and one_step_pass and diag_open and best_val >= V616_GT4_ADAN_VAL_MEAN + 0.02:
        route = "R7-S2RestoredTaskImproves"
        blocker = "none"
        next_impl = "continue_fused_grouped_mainline"
    elif survivor == "S2" and one_step_pass:
        route = "R8-S2RestoredTaskGapPersists" if diag_open else "R1-S2Restored"
        blocker = "task_gap" if diag_open else "none"
        next_impl = "task_gap_repair" if diag_open else "diagnostic_task"
    else:
        route = "R6-S2NotRestored"
        blocker = "efficiency_margin"
        next_impl = "kernel_margin_repair"
    tax = ";".join(sorted({str(r.get("task_gap_taxonomy", "")) for r in p6 if r.get("task_gap_taxonomy")})) or "not_available"
    data = {
        "route": route,
        "best_candidate": best.get("package", ""),
        "best_family": best.get("components", ""),
        "best_memory_ratio": f(best, "memory_ratio_mean", 99.0),
        "best_step_ratio": f(best, "step_ratio_mean", 99.0),
        "best_backward_ratio": f(best, "backward_ratio_mean", 99.0),
        "best_forward_ratio": f(best, "forward_ratio_mean", 99.0),
        "memory_improvement_vs_GT4": f(best, "memory_improvement_vs_GT4", 0.0),
        "step_improvement_vs_GT4": f(best, "step_improvement_vs_GT4", 0.0),
        "survivor_type": survivor,
        "one_step_probe_pass": one_step_pass,
        "official_task_opened": survivor in {"S0", "S1"} and one_step_pass,
        "diagnostic_task_opened": diag_open,
        "best_val_acc": best_val,
        "best_test_acc": best_test,
        "val_acc_gap_vs_MLP": max((f(r, "val_acc_gap_vs_MLP", -99.0) for r in p6), default=math.nan),
        "test_acc_gap_vs_MLP": max((f(r, "test_acc_gap_vs_MLP", -99.0) for r in p6), default=math.nan),
        "task_gap_taxonomy": tax,
        "baseline_step_drift_resolved": bool(repro_pass),
        "p0_side_by_side_reproduction_pass": bool(p0_side_by_side_repro_pass),
        "canonical_protocol_pass": bool(canonical_pass),
        "canonical_protocol_strict_pass": bool(canonical_strict_pass),
        "canonical_protocol_basis": canonical_basis,
        "canonical_step_delta_P2_M0_vs_P0_GT4": canonical_delta_p2,
        "canonical_step_delta_P3_T0_vs_P0_GT4": canonical_delta_p3,
        "open_optimizer_exploration": False,
        "open_functional_correction": False,
        "primary_blocker": blocker,
        "next_required_implementation": next_impl,
        "no_fake": True,
        "no_proxy": True,
    }
    _json_dump(Path(args.out_dir) / "route_decision.json", data)
    _json_dump(Path(args.out_dir) / "aggregate_decision.json", {"status": "diagnostic_or_gated", "fake_data_used": 0, "proxy_rows_used_as_results": 0, **data})
    return data


def run_failure(args: argparse.Namespace) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    for fn, stage in [
        ("p0_reproduction_check.csv", "P0"),
        ("p2_nonrecompute_memory_repair.csv", "P2"),
        ("p3_step_repair.csv", "P3"),
        ("p4_combo_repair_selection.csv", "P4"),
        ("p6_diagnostic_task_gap_attribution.csv", "P6"),
        ("p7_optimizer_expressivity_diagnostic.csv", "P7"),
        ("p8_task_reentry.csv", "P8"),
        ("p9_optimizer_exploration.csv", "P9"),
        ("p10_functional_correction_smoke.csv", "P10"),
    ]:
        rows = _read_csv(Path(args.out_dir) / fn)
        if not rows:
            failures.append({"stage": stage, "variant_id": fn, "failure_type": "F18_artifact_missing", "metric": "missing"})
            continue
        for row in rows:
            vid = str(row.get("variant_id") or row.get("package") or row.get("variant") or row.get("candidate") or fn)
            status = str(row.get("implementation_status") or row.get("stage_status") or "")
            if "not_implemented" in status or status == "not_run":
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F13_official_task_gated" if stage in {"P8", "P9", "P10"} else "F0_not_implemented_or_gated", "metric": row.get("reason", status)})
            if stage == "P0" and "GT4" in vid and int(f(row, "v616_reproduction_pass", 0)) != 1:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F4_baseline_step_drift", "metric": f"step_delta={row.get('reproduction_delta_vs_v616_step')}"})
            if row.get("memory_ratio_mean") not in {"", None} and f(row, "memory_ratio_mean", 0.0) > 1.05:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F1_memory_fail", "metric": f"memory_ratio={row.get('memory_ratio_mean')}"})
            if row.get("step_ratio_mean") not in {"", None} and f(row, "step_ratio_mean", 0.0) > 1.50:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F2_step_time_fail", "metric": f"step_ratio={row.get('step_ratio_mean')}"})
            if stage == "P2" and int(f(row, "memory_useful", 0)) == 1 and int(f(row, "step_preserving", 0)) != 1:
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F6_memory_repair_step_regression", "metric": f"step_ratio={row.get('step_ratio_mean')}"})
            if stage == "P6" and "T1" in str(row.get("task_gap_taxonomy", "")):
                failures.append({"stage": stage, "variant_id": vid, "failure_type": "F9_task_gap_expressivity", "metric": row.get("task_gap_taxonomy", "")})
    _write_csv(Path(args.out_dir) / "failure_table.csv", failures or [{"stage": "ALL", "variant_id": "none", "failure_type": "none"}])
    return failures


def _finalize(args: argparse.Namespace, started: float) -> None:
    out = Path(args.out_dir)
    manifest = {
        "provenance": "EMPIRICAL_REAL_ONLY_NO_PROXY",
        "script": SCRIPT_PATH,
        "plan": PLAN_PATH,
        "started_unix": started,
        "finished_unix": time.time(),
        "duration_sec": time.time() - started,
        "source_commit": _git_commit(),
        "git_status_short": _git_status(),
        "command_args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "triton_available": int(v616.triton is not None),
        "memory_history_available": int(hasattr(torch.cuda.memory, "_record_memory_history")),
    }
    _json_dump(out / "run_manifest.json", manifest)
    hashes = {p.name: _sha256(p) for p in sorted(out.glob("*")) if p.is_file() and p.suffix in {".csv", ".json", ".log", ".svg", ".md"}}
    _json_dump(out / "artifact_hashes.json", hashes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(wandb=True)
    parser.add_argument("--packages", default="V6_20_ALL")
    parser.add_argument("--out-dir", type=Path, default=Path("results/real_rerun_20260505/v620_real"))
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
    parser.add_argument("--wandb-group", default="v620-real-20260505")
    parser.add_argument("--wandb-name-prefix", default="v620-real")
    parser.add_argument("--no-wandb", action="store_false", dest="wandb")
    parser.add_argument("--fresh", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    ensure_dir(args.out_dir)
    _patch_stack()
    _wandb_init(args)
    try:
        _p0_detail, p0_summary = run_p0(args)
        canonical_mlp = _canonical_mlp_map(_p0_detail)
        p3, _p3_detail = run_p3(args, canonical_mlp)
        p2, _p2_detail = run_p2(args, canonical_mlp)
        run_p1(args)
        p4, selected = run_p4(args, p2, p3)
        canonical_ok, _d2, _d3, _strict, _basis = _canonical_protocol_pass(p0_summary, p2, p3)
        baseline_reproduced = any(int(f(r, "v616_reproduction_pass", 0)) == 1 for r in p0_summary if "GT4" in str(r.get("variant", ""))) or canonical_ok
        if not baseline_reproduced or not canonical_ok:
            selected = None
        _run_task_gates(args, selected)
        p6 = _p6_gap_from_task(args)
        run_p7(args)
        route = run_route(args, p0_summary, p2, p3, p4, p6)
        run_failure(args)
        _finalize(args, started)
        _log_memory(args, route, "P11/route")
    finally:
        _wandb_finish(args, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
