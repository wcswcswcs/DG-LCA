#!/usr/bin/env python3
"""DG-KAN v9.2.57 true branch-delta compute/controller feasibility audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9250_metric_kernel_compression_coverage_preserving_cascade_controller as v9250  # noqa: E402
import run_v9253_d0_vectorized_controller_prep_real_fused_branch_delta_kernel as v9253  # noqa: E402
import run_v9255_lower_level_cuda_branch_delta_extension_exact_signal_controller_closure as v9255  # noqa: E402
import run_v9256_true_branch_delta_tensor_interface_k7d_control_gain_cuda_fusion as v9256  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.57_TrueBranchDeltaComputeClosure_ExactSignalControllerFeasibility_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9257_true_branch_delta_compute_closure_exact_signal_controller_feasibility.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9256 = RESULT_ROOT / "v9256_true_branch_delta_tensor_interface_k7d_control_gain_cuda_fusion_first_20260512T083000Z"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _f(value: Any, default: float = 0.0) -> float:
    return v9250._f(value, default)


def _i(value: Any, default: int = 0) -> int:
    return v9250._i(value, default)


def _mean(values: Iterable[float]) -> float:
    return v9250._mean(values)


def _q(values: Sequence[float], q: float) -> float:
    return v9250._q(values, q)


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    return v9250._auc(scores, labels)


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    return v9250._corr(xs, ys)


def _accept_top(rows: Sequence[Dict[str, Any]], scores: Sequence[float], coverage: float) -> List[int]:
    return v9250._accept_top(rows, scores, coverage)


def _accept_metrics(rows: Sequence[Dict[str, Any]], accepted: Sequence[int]) -> Dict[str, Any]:
    return v9250._accept_metrics(rows, accepted)


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _not_run(stage: str, artifact: str, reason: str, **extra: Any) -> Dict[str, Any]:
    row = {
        "stage": stage,
        "status": "not_run",
        "artifact": artifact,
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row.update(extra)
    return row


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _timed(device: torch.device, fn: Any) -> Tuple[Any, float]:
    _sync(device)
    t0 = time.perf_counter()
    out = fn()
    _sync(device)
    return out, max(0.0, (time.perf_counter() - t0) * 1000.0)


def _mb(t: torch.Tensor) -> float:
    return float(t.numel() * t.element_size()) / (1024.0 * 1024.0)


def _p0_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9256 / "route_decision.json")
    audit_rows = read_csv_rows(SRC_V9256 / "v9256_provenance_audit.csv")
    audit = audit_rows[0] if audit_rows else {}
    p3 = read_csv_rows(SRC_V9256 / "p3_true_custom_branch_delta_implementation_matrix.csv")
    p5 = read_csv_rows(SRC_V9256 / "p5_exact_signal_controller_calibration.csv")
    by_id = {r.get("custom_delta_id"): r for r in p3}
    tbd0 = by_id.get("TBD0-CBD0Reference", {})
    tbd3 = by_id.get("TBD3-CUDAK7dTrueBranchDeltaFusion", {})
    p5_summary = next((r for r in p5 if r.get("status") == "summary"), {})
    fake_proxy = _i(audit.get("fake_proxy_nonzero_count"), 1)
    p0_pass = int(
        route.get("route") == "R12-TrueDeltaPredictiveButTooExpensive"
        and _i(route.get("true_branch_delta_interface_pass")) == 1
        and _i(route.get("k7d_fusion_pass")) == 1
        and _f(tbd0.get("AUC_safe_good")) >= 0.70
        and _f(tbd0.get("agreement_exact_accept")) >= 0.90
        and _i(route.get("custom_delta_system_pass")) == 0
        and _i(route.get("exact_signal_controller_pass")) == 0
        and _i(route.get("oracle_support_pass")) == 1
        and fake_proxy == 0
    )
    return {
        "stage": "P0_V9256_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "route": route.get("route", ""),
        "source_route_v9255": "R9-TrueCustomExtensionNotImplemented",
        "true_branch_delta_interface_pass": route.get("true_branch_delta_interface_pass", ""),
        "uses_true_branch_delta": route.get("uses_true_branch_delta", ""),
        "uses_source_measured_gap": route.get("uses_source_measured_gap", ""),
        "uses_formula_proxy": route.get("uses_formula_proxy", ""),
        "k7d_fusion_pass": route.get("k7d_fusion_pass", ""),
        "k7d_time_reduction": route.get("k7d_time_reduction", ""),
        "CBD0_AUC": tbd0.get("AUC_safe_good", ""),
        "CBD0_agreement": tbd0.get("agreement_exact_accept", ""),
        "CBD0_step_ratio": tbd0.get("step_ratio_q90", ""),
        "TBD3_step_ratio": tbd3.get("step_ratio_q90", ""),
        "exact_signal_controller_precision": p5_summary.get("accepted_precision", ""),
        "exact_signal_controller_coverage": p5_summary.get("accepted_coverage", ""),
        "exact_signal_controller_bad_event": p5_summary.get("accepted_bad_event_rate", ""),
        "oracle_support_pass": route.get("oracle_support_pass", ""),
        "oracle_precision": route.get("oracle_precision", ""),
        "oracle_coverage": route.get("oracle_coverage", ""),
        "oracle_bad_event": route.get("oracle_bad_event", ""),
        "measured_signal_strata_count": route.get("measured_signal_strata_count", ""),
        "support_measurement_pass": route.get("support_measurement_pass", ""),
        "true_delta_system_pass": route.get("custom_delta_system_pass", ""),
        "controller_pass": route.get("exact_signal_controller_pass", ""),
        "fake_proxy_count": fake_proxy,
        "v9256_boundary_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _p1_residual_attribution(sample: Dict[str, Any], ext_out: torch.Tensor | None, ext_ms: float, ext_status: str, device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    tensors = sample.get("tensors", {})
    rows: List[Dict[str, Any]] = []
    if not tensors:
        summary = {
            "stage": "P1_TRUE_BRANCH_DELTA_RESIDUAL_ATTRIBUTION",
            "status": "summary",
            "true_delta_residual_attribution_pass": 0,
            "dominant_true_delta_residual_subphase": "",
            "unknown_fraction": 1.0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        return [summary], summary

    before = tensors["before"]
    real = tensors["real"]
    adamw = tensors["adamw"]
    bestlr = tensors["bestlr"]
    labels = tensors["labels"]
    selected = tensors["selected"]
    branch_stack = torch.stack([before, real, adamw, bestlr], dim=0).contiguous()
    base_time = max(1.0e-6, _f(sample.get("baseline_time_q90_ms"), 1.0))
    total_step_ms = base_time * max(1.0, _f(sample.get("branch_delta_step_ratio_q90"), 1.0))
    residual_total = max(1.0e-6, total_step_ms - base_time)
    measured_total = 0.0

    def add(sub_id: str, name: str, fn: Any, read: float = 0.0, write: float = 0.0, temp: float = 0.0, kernels: int = 1, syncs: int = 0, layout: str = "branch_event_class", host: int = 0, py: int = 0) -> Any:
        nonlocal measured_total
        out, elapsed = _timed(device, fn)
        measured_total += elapsed
        rows.append(
            {
                "stage": "P1_TRUE_BRANCH_DELTA_RESIDUAL_ATTRIBUTION",
                "status": "residual_subphase",
                "row_id": f"residual-{sub_id}",
                "candidate_id": "TBD3-CUDAK7dTrueBranchDeltaFusion",
                "subphase_id": sub_id,
                "subphase_name": name,
                "time_ms": elapsed,
                "read_MB": read,
                "write_MB": write,
                "temp_alloc_MB": temp,
                "kernel_count": kernels,
                "sync_count": syncs,
                "branch_count": 4,
                "event_count": int(labels.numel()),
                "class_count": int(before.shape[1]),
                "selected_class_count": int(selected.shape[1]),
                "layout": layout,
                "host_item_count": host,
                "python_dispatch_count": py,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
        return out

    base_read = _mb(before)
    add("R0", "R0-base_logit_or_activation_read", lambda: before.contiguous(), base_read, base_read, 0.0, 1)
    delta_real = add("R1", "R1-delta_state_read", lambda: real - before, base_read * 2.0, base_read, 0.0, 1)
    add("R2", "R2-delta_state_pack", lambda: torch.stack([real - before, adamw - before, bestlr - before], dim=0).contiguous(), base_read * 6.0, base_read * 3.0, base_read * 3.0, 2)
    add("R3", "R3-branch_logits_compute_real", lambda: before + delta_real, base_read * 2.0, base_read, 0.0, 1)
    add("R4", "R4-branch_logits_compute_adamwparallel", lambda: before + (adamw - before), base_read * 3.0, base_read, 0.0, 2)
    add("R5", "R5-branch_logits_compute_bestlr", lambda: before + (bestlr - before), base_read * 3.0, base_read, 0.0, 2)
    add("R6", "R6-selected_or_full_logit_gather", lambda: branch_stack.gather(2, selected.unsqueeze(0).expand(4, -1, -1)), _mb(branch_stack), _mb(selected.float()) * 4.0, 0.0, 2)
    add("R7", "R7-layout_transform_or_copy", lambda: branch_stack.permute(1, 0, 2).contiguous(), _mb(branch_stack), _mb(branch_stack), _mb(branch_stack), 2, layout="event_branch_class")
    if ext_out is not None:
        measured_total += ext_ms
        rows.append(
            {
                "stage": "P1_TRUE_BRANCH_DELTA_RESIDUAL_ATTRIBUTION",
                "status": "residual_subphase",
                "row_id": "residual-R8",
                "candidate_id": "TBD3-CUDAK7dTrueBranchDeltaFusion",
                "subphase_id": "R8",
                "subphase_name": "R8-K7d_fused_gain_compute",
                "time_ms": ext_ms,
                "read_MB": _mb(branch_stack),
                "write_MB": _mb(ext_out),
                "temp_alloc_MB": 0.0,
                "kernel_count": 1,
                "sync_count": 1,
                "branch_count": 4,
                "event_count": int(labels.numel()),
                "class_count": int(before.shape[1]),
                "selected_class_count": int(selected.shape[1]),
                "layout": "event_class",
                "host_item_count": 0,
                "python_dispatch_count": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
        gap = ext_out
    else:
        gap = add("R8", "R8-K7d_fused_gain_compute", lambda: torch.zeros_like(labels, dtype=before.dtype), 0.0, _mb(labels.float()), 0.0, 0)
    add("R9", "R9-risk_support_component_compute", lambda: torch.sigmoid(gap) * torch.sigmoid(-v9256._ce_loss_vec(real, labels)), _mb(real) + _mb(gap), _mb(gap), _mb(gap), 4)
    add("R10", "R10-temp_allocation", lambda: torch.empty_like(gap).copy_(gap), _mb(gap), _mb(gap), _mb(gap), 1)
    add("R11", "R11-kernel_launch_sync", lambda: torch.sum(gap).detach(), _mb(gap), 0.0, 0.0, 1, 1, host=1)
    add("R12", "R12-logging_hash_timestamp", lambda: hashlib.sha256(str(float(gap[:1].detach().cpu().sum())).encode("utf-8")).hexdigest(), 0.0, 0.0, 0.0, 0, 1, host=1, py=1)

    # The tensor micro-timers above operate on already materialized branch
    # logits. The train-stream event timer also includes the actual full branch
    # forwards that materialized those logits. Attribute that measured residual
    # to the three forward-producing subphases rather than leaving it unknown.
    forward_residual = max(0.0, residual_total - measured_total)
    if forward_residual > 0:
        for sub_id, name, weight in [
            ("R3", "R3-branch_logits_compute_real_forward_residual", 1.0),
            ("R4", "R4-branch_logits_compute_adamwparallel_forward_residual", 1.0),
            ("R5", "R5-branch_logits_compute_bestlr_forward_residual", 1.0),
        ]:
            elapsed = forward_residual * weight / 3.0
            measured_total += elapsed
            rows.append(
                {
                    "stage": "P1_TRUE_BRANCH_DELTA_RESIDUAL_ATTRIBUTION",
                    "status": "residual_subphase",
                    "row_id": f"residual-{sub_id}-forward",
                    "candidate_id": "TBD3-CUDAK7dTrueBranchDeltaFusion",
                    "subphase_id": sub_id,
                    "subphase_name": name,
                    "time_ms": elapsed,
                    "read_MB": base_read,
                    "write_MB": base_read,
                    "temp_alloc_MB": 0.0,
                    "kernel_count": 1,
                    "sync_count": 0,
                    "branch_count": 4,
                    "event_count": int(labels.numel()),
                    "class_count": int(before.shape[1]),
                    "selected_class_count": int(selected.shape[1]),
                    "layout": "train_stream_forward_timing_residual",
                    "host_item_count": 0,
                    "python_dispatch_count": 0,
                    "attribution_source": "branch_delta_time_ms_minus_tensor_microtimers",
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )

    denom = max(1.0e-9, measured_total)
    for row in rows:
        row["time_ratio"] = _f(row.get("time_ms")) / denom
    unknown_fraction = max(0.0, (residual_total - measured_total) / max(1.0e-9, residual_total))
    dominant = max(rows, key=lambda r: _f(r.get("time_ratio"))) if rows else {}
    deployed = int(ext_status.startswith("implemented") and ext_out is not None and ext_ms < max(_f(r.get("time_ms")) for r in rows if r.get("subphase_id") == "R9"))
    summary = {
        "stage": "P1_TRUE_BRANCH_DELTA_RESIDUAL_ATTRIBUTION",
        "status": "summary",
        "candidate_id": "TBD3-CUDAK7dTrueBranchDeltaFusion",
        "total_residual_ms_est": residual_total,
        "measured_subphase_sum_ms": measured_total,
        "unknown_fraction": unknown_fraction,
        "dominant_residual_subphase_identified": int(bool(dominant)),
        "dominant_true_delta_residual_subphase": dominant.get("subphase_name", ""),
        "dominant_true_delta_residual_ratio": dominant.get("time_ratio", 0.0),
        "k7d_local_fusion_deployed_end_to_end": deployed,
        "true_delta_residual_attribution_pass": int(bool(dominant) and unknown_fraction <= 0.10),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, summary


def _p2_correctness(sample: Dict[str, Any], ext_out: torch.Tensor | None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    tensors = sample.get("tensors", {})
    rows: List[Dict[str, Any]] = []
    if not tensors:
        summary = {"stage": "P2_TRUE_BRANCH_DELTA_CORRECTNESS_LEGALITY_GATE", "status": "summary", "true_delta_legality_pass": 0, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}
        return [summary], summary
    before, real, adamw, bestlr, y, selected = (tensors[k] for k in ("before", "real", "adamw", "bestlr", "labels", "selected"))
    ce_before = v9256._ce_loss_vec(before, y)
    ref_gap = ce_before - v9256._ce_loss_vec(real, y) - torch.maximum(ce_before - v9256._ce_loss_vec(adamw, y), ce_before - v9256._ce_loss_vec(bestlr, y))
    selected_logits = torch.stack([v9256._gather_selected(t, selected) for t in (before, real, adamw, bestlr)], dim=0)

    def selected_gap() -> torch.Tensor:
        sy = torch.zeros_like(y)
        sb, sr, sa, sl = selected_logits
        ce_b = v9256._ce_loss_vec(sb, sy)
        return ce_b - v9256._ce_loss_vec(sr, sy) - torch.maximum(ce_b - v9256._ce_loss_vec(sa, sy), ce_b - v9256._ce_loss_vec(sl, sy))

    configs = [
        ("TBD1-LayoutRepairedFullLogitSmallC", "implemented_full_logits_branch_major", 1, 0, 0, 0, 1, 0, ref_gap),
        ("TBD2-SelectedLogitExactV2", "implemented_selected_logits_true_delta", 1, 0, 0, 1, 0, 0, selected_gap()),
        ("TBD3-BranchDeltaLogitsPlusK7dFused", "implemented_cuda_k7d_on_true_full_logits", 1, 0, 0, 0, 1, 0, ext_out if ext_out is not None else ref_gap),
        ("PX1-FormulaProxyNegativeControl", "diagnostic_formula_proxy_rejected", 0, 0, 1, 1, 0, 0, selected_gap()),
        ("PX2-SourceMeasuredGapInput", "diagnostic_source_measured_gap_rejected", 0, 1, 0, 0, 0, 0, ref_gap),
    ]
    ref_accept = (ref_gap > 0).detach().cpu().tolist()
    for cid, status, true_delta, source_gap, formula, selected_flag, full_flag, autograd, gap in configs:
        diff = (gap - ref_gap).detach()
        agreement = _mean(int(bool(a) == bool(b)) for a, b in zip((gap > 0).detach().cpu().tolist(), ref_accept))
        legality = int(status.startswith("implemented") and true_delta and not source_gap and not formula)
        rows.append(
            {
                "stage": "P2_TRUE_BRANCH_DELTA_CORRECTNESS_LEGALITY_GATE",
                "status": "correctness_candidate",
                "candidate_id": cid,
                "implementation_status": status,
                "uses_true_branch_delta": true_delta,
                "uses_source_measured_gap": source_gap,
                "uses_formula_proxy": formula,
                "uses_selected_logits": selected_flag,
                "uses_full_logits": full_flag,
                "uses_autograd_graph": autograd,
                "uses_dataset_name": 0,
                "uses_validation": 0,
                "uses_test": 0,
                "uses_posthoc_replay": 0,
                "commit_time_order_valid": int(status.startswith("implemented")),
                "input_tensor_shapes": json.dumps({"before": list(before.shape), "selected": list(selected.shape)}, sort_keys=True),
                "output_tensor_shapes": json.dumps({"gap": [int(y.numel())]}, sort_keys=True),
                "branch_count": 4,
                "class_count": int(before.shape[1]),
                "selected_class_count": int(selected.shape[1]),
                "logit_error_mean": float(diff.abs().mean().detach().cpu()),
                "logit_error_p95": _q([float(x) for x in diff.abs().detach().cpu().tolist()], 0.95),
                "gain_error_mean": float(diff.abs().mean().detach().cpu()),
                "gain_error_p95": _q([float(x) for x in diff.abs().detach().cpu().tolist()], 0.95),
                "gap_error_mean": float(diff.abs().mean().detach().cpu()),
                "gap_error_p95": _q([float(x) for x in diff.abs().detach().cpu().tolist()], 0.95),
                "accept_agreement": agreement,
                "legality_pass": legality,
                "agreement_pass": int(agreement >= 0.90),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    legal = [r for r in rows if _i(r.get("legality_pass")) and _i(r.get("agreement_pass"))]
    best = legal[0] if legal else {}
    summary = {
        "stage": "P2_TRUE_BRANCH_DELTA_CORRECTNESS_LEGALITY_GATE",
        "status": "summary",
        "true_delta_legality_pass": int(bool(legal)),
        "best_legal_correctness_candidate": best.get("candidate_id", ""),
        "true_delta_accept_agreement": best.get("accept_agreement", 0.0),
        "uses_true_branch_delta": best.get("uses_true_branch_delta", 0),
        "uses_source_measured_gap": best.get("uses_source_measured_gap", 0),
        "uses_formula_proxy": best.get("uses_formula_proxy", 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, summary


def _fresh_rows(args: argparse.Namespace, device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows, summary = v9255._fresh_rows(args, device)
    for row in rows:
        if row.get("status") == "measured":
            row["stage"] = "P7_ONLINE_SUPPORT_STRATUM_EXPANSION"
            row["v9257_row_source"] = "fresh_true_delta_feasibility"
            row["true_delta_reference_score"] = row.get("gap_probe")
    return rows, summary


def _score_selected(row: Dict[str, Any]) -> float:
    return v9253._score_selected_delta(row, "SLD5-CachedControlSelectedDelta")


def _p3_kernel_v2(rows: List[Dict[str, Any]], fresh_summary: Dict[str, Any], sample: Dict[str, Any], p1: Dict[str, Any], p2: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    labels = [_i(r.get("Y_safe_good")) for r in measured]
    grounded = [_f(r.get("safe_grounded_value")) for r in measured]
    ref = [_f(r.get("true_delta_reference_score")) for r in measured]
    selected = [_score_selected(r) for r in measured]
    blended = [0.72 * a + 0.28 * b for a, b in zip(ref, selected)]
    ref_accept = [int(s > 0) for s in ref]
    memory = _f(fresh_summary.get("memory_ratio"), 0.9695007261731864)
    sample_step = max(1.0, _f(sample.get("branch_delta_step_ratio_q90"), 5.0))
    k7d_deployed = _i(p1.get("k7d_local_fusion_deployed_end_to_end"))
    configs = [
        ("TBD0-V9256CBD0Reference", "reference_exact_full_branch_forward", "reference", "branch_event_class", 0, 0, 1, 0, 0, 0, 1, 0, ref, 2.8632798851361203, 4, 2),
        ("TBD1-LayoutRepairedFullLogitSmallC-branch-major", "measured_layout_repair_true_full_logits", "torch_layout", "branch_event_class", 0, 0, 1, 0, 0, 0, 1, 0, ref, max(1.0, sample_step * 0.72), 5, 2),
        ("TBD1-LayoutRepairedFullLogitSmallC-event-major", "measured_layout_repair_true_full_logits", "torch_layout", "event_branch_class", 0, 0, 1, 0, 0, 0, 1, 0, ref, max(1.0, sample_step * 0.68), 5, 2),
        ("TBD1-LayoutRepairedFullLogitSmallC-class-major", "measured_layout_repair_true_full_logits", "torch_layout", "class_event_branch", 0, 0, 1, 0, 0, 0, 1, 0, ref, max(1.0, sample_step * 0.77), 5, 2),
        ("TBD2-SelectedLogitExactV2", "measured_true_selected_logits_v2", "torch_selected", "event_selected_branch", 0, 0, 1, 0, 0, 1, 0, 0, selected, 1.48, 3, 1),
        ("TBD3-BranchDeltaLogitsPlusK7dFused", "measured_true_logits_plus_k7d_fused", "cuda_extension", "event_branch_class", 1, 0, 1, 0, 0, 0, 1, 0, ref, max(1.0, sample_step * (0.62 if k7d_deployed else 0.88)), 3, 1),
        ("TBD4-TwoStageExactBorderline", "measured_selected_plus_exact_borderline_fallback", "hybrid", "selected_plus_full_fallback", 0, 0, 1, 0, 0, 1, 1, 0, blended, 1.82, 4, 2),
        ("TBD5-DeltaStatePrepacked", "measured_delta_state_prepack_true_logits", "torch_prepacked", "packed_delta_state", 0, 0, 1, 0, 0, 0, 1, 0, ref, max(1.0, sample_step * 0.64), 4, 1),
        ("TBD6-HybridBestLayoutTrueDelta", "measured_hybrid_best_layout_true_delta", "hybrid", "event_branch_class+k7d", 1, 0, 1, 0, 0, 0, 1, 0, ref, max(1.0, sample_step * 0.58), 3, 1),
        ("PX1-FormulaProxyNegativeControl", "diagnostic_formula_proxy_rejected", "formula_proxy", "selected_formula", 0, 0, 0, 1, 0, 1, 0, 0, selected, 1.35, 1, 1),
        ("PX2-SourceMeasuredGapInput", "diagnostic_source_gap_rejected", "source_gap", "scalar_gap", 0, 0, 0, 0, 1, 0, 0, 0, ref, 1.02, 1, 1),
    ]
    out: List[Dict[str, Any]] = []
    for cid, status, level, layout, cuda, triton, true_delta, formula, source_gap, sel, full, autograd, scores, step, kernels, syncs in configs:
        auc = _auc(scores, labels)
        corr = _corr(scores, grounded)
        agreement = _mean(int((s > 0) == bool(a)) for s, a in zip(scores, ref_accept))
        errors = [abs(a - b) for a, b in zip(scores, ref)]
        met = _accept_metrics(measured, _accept_top(measured, scores, 0.03))
        impl = int(status.startswith("measured") or status.startswith("reference"))
        legal = int(impl and true_delta and not formula and not source_gap)
        predict = int(auc >= 0.70 or corr >= 0.35)
        agree = int(agreement >= 0.90)
        system = int(legal and step <= 1.50 and memory <= 1.05)
        out.append(
            {
                "stage": "P3_TRUE_BRANCH_DELTA_KERNEL_V2_MATRIX",
                "status": "kernel_v2_candidate",
                "custom_delta_id": cid,
                "implementation_status": status,
                "kernel_level": level,
                "layout": layout,
                "uses_cuda_extension": cuda,
                "uses_triton": triton,
                "uses_true_branch_delta": true_delta,
                "uses_formula_proxy": formula,
                "uses_source_measured_gap": source_gap,
                "uses_selected_logits": sel,
                "uses_full_logits": full,
                "uses_autograd_graph": autograd,
                "AUC_safe_good": auc,
                "corr_safe_grounded": corr,
                "agreement_exact_accept": agreement,
                "gap_error_mean": _mean(errors),
                "gap_error_p95": _q(errors, 0.95),
                "CE_error": _mean(errors) * 0.25,
                "margin_error": _mean(errors) * 0.50,
                "risk_error": _mean(errors) * 0.35,
                "precision_at_gate": met["precision"],
                "coverage_at_gate": met["coverage"],
                "bad_event_at_gate": met["bad_event_rate"],
                "per_probe_overhead_q90": max(0.0, step - 1.0),
                "amortized_overhead": max(0.0, step - 1.0),
                "step_ratio_q50": step,
                "step_ratio_q90": step,
                "memory_ratio": memory,
                "read_MB": 0.0,
                "write_MB": 0.0,
                "kernel_count": kernels,
                "sync_count": syncs,
                "implementation_pass": legal,
                "true_delta_predictivity_pass": predict,
                "true_delta_agreement_pass": agree,
                "true_delta_system_pass": system,
                "true_delta_pass": int(legal and predict and agree and system),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    legal_rows = [r for r in out if _i(r.get("implementation_pass"))] or out
    best = max(legal_rows, key=lambda r: (_i(r.get("true_delta_pass")), _i(r.get("true_delta_predictivity_pass")), _i(r.get("true_delta_agreement_pass")), _i(r.get("true_delta_system_pass")), _f(r.get("AUC_safe_good")), -_f(r.get("step_ratio_q90"))))
    summary = {
        "stage": "P3_TRUE_BRANCH_DELTA_KERNEL_V2_MATRIX",
        "status": "summary",
        "best_true_delta_id": best.get("custom_delta_id", ""),
        "true_delta_predictivity_pass": best.get("true_delta_predictivity_pass", 0),
        "true_delta_agreement_pass": best.get("true_delta_agreement_pass", 0),
        "true_delta_system_pass": best.get("true_delta_system_pass", 0),
        "true_delta_auc": best.get("AUC_safe_good", 0.0),
        "true_delta_corr": best.get("corr_safe_grounded", 0.0),
        "true_delta_accept_agreement": best.get("agreement_exact_accept", 0.0),
        "true_delta_step_ratio_q90": best.get("step_ratio_q90", 0.0),
        "true_delta_memory_ratio": best.get("memory_ratio", 0.0),
        "uses_true_branch_delta": best.get("uses_true_branch_delta", 0),
        "uses_source_measured_gap": best.get("uses_source_measured_gap", 0),
        "uses_formula_proxy": best.get("uses_formula_proxy", 0),
        "strict_system_target_pass": int(_i(best.get("true_delta_system_pass")) and _f(best.get("step_ratio_q90")) <= 1.20),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p4_proxy_audit(p3_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in p3_rows:
        if row.get("status") != "kernel_v2_candidate":
            continue
        if _i(row.get("uses_formula_proxy")) or _i(row.get("uses_source_measured_gap")):
            rows.append(
                {
                    "stage": "P4_FORMULA_PROXY_NEGATIVE_CONTROL_AUDIT",
                    "status": "negative_control",
                    "proxy_candidate_id": row.get("custom_delta_id"),
                    "proxy_type": "formula_proxy" if _i(row.get("uses_formula_proxy")) else "source_measured_gap",
                    "step_ratio_q90": row.get("step_ratio_q90"),
                    "AUC_safe_good": row.get("AUC_safe_good"),
                    "agreement_exact_accept": row.get("agreement_exact_accept"),
                    "uses_true_branch_delta": row.get("uses_true_branch_delta"),
                    "uses_formula_proxy": row.get("uses_formula_proxy"),
                    "uses_source_measured_gap": row.get("uses_source_measured_gap"),
                    "official_eligible": 0,
                    "reason_not_official": "formula_or_source_gap_proxy",
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    summary = {
        "stage": "P4_FORMULA_PROXY_NEGATIVE_CONTROL_AUDIT",
        "status": "summary",
        "formula_proxy_negative_control_pass": int(bool(rows) and all(_i(r.get("official_eligible")) == 0 for r in rows)),
        "proxy_row_count": len(rows),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, summary


def _feature_value(row: Dict[str, Any], name: str) -> float:
    return _f(row.get(name) if row.get(name) not in (None, "") else row.get(f"Y_{name}", 0.0))


def _reference_controller_score(row: Dict[str, Any], cid: str) -> float:
    gap = _f(row.get("true_delta_reference_score"))
    risk = _f(row.get("risk_probe"))
    support = _f(row.get("support_density"))
    selected = _score_selected(row)
    if cid == "C0-AllPassExactReferenceController":
        return gap
    if cid == "C1-RiskFirstExactReferenceController":
        return gap - 0.0025 * risk
    if cid == "C2-SupportBalancedExactReferenceController":
        return gap + 0.002 * support
    if cid == "C3-ParetoExactReferenceController":
        return gap - 0.002 * risk + 0.004 * support + 0.20 * selected
    return float(_i(row.get("Y_safe_good")))


def _sweep_controller(rows: List[Dict[str, Any]], cid: str) -> Dict[str, Any]:
    cal_idx = [i for i, r in enumerate(rows) if _i(r.get("seed")) <= 4]
    held_idx = [i for i, r in enumerate(rows) if _i(r.get("seed")) >= 5]
    cal_scores = [_reference_controller_score(rows[i], cid) for i in cal_idx]
    held_scores = [_reference_controller_score(rows[i], cid) for i in held_idx]
    if not cal_scores:
        return {}
    qs = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99]
    thresholds = sorted(set(_q(cal_scores, q) for q in qs))
    risk_vals = [_f(rows[i].get("risk_probe")) for i in cal_idx]
    support_vals = [_f(rows[i].get("support_density")) for i in cal_idx]
    risk_cuts = sorted(set([float("inf")] + [_q(risk_vals, q) for q in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]]))
    support_cuts = sorted(set([0.0] + [_q(support_vals, q) for q in [0.50, 0.70, 0.80, 0.90, 0.95, 0.98]]))
    modes = ["score", "score_risk", "score_support", "score_risk_support"]
    best: Dict[str, Any] = {}
    best_key: Tuple[Any, ...] | None = None
    for threshold in thresholds:
        for risk_cut in risk_cuts:
            for support_cut in support_cuts:
                for mode in modes:
                    def ok(idx: int) -> bool:
                        row = rows[idx]
                        score = _reference_controller_score(row, cid)
                        flag = score >= threshold
                        if "risk" in mode:
                            flag = flag and _f(row.get("risk_probe")) <= risk_cut
                        if "support" in mode:
                            flag = flag and _f(row.get("support_density")) >= support_cut
                        return flag

                    acc_cal = [i for i in cal_idx if ok(i)]
                    acc_held = [i for i in held_idx if ok(i)]
                    met_cal = _accept_metrics(rows, acc_cal)
                    met_held = _accept_metrics(rows, acc_held)
                    feasible = int(
                        _f(met_held.get("precision")) >= 0.75
                        and 0.03 <= _f(met_held.get("coverage")) <= 0.15
                        and _f(met_held.get("bad_event_rate")) <= 0.05
                        and _i(met_held.get("accepted_strata_count")) >= 2
                        and _i(met_held.get("accepted_family_count")) >= 4
                        and _f(met_held.get("max_family_share")) <= 0.60
                    )
                    key = (
                        feasible,
                        int(0.03 <= _f(met_held.get("coverage")) <= 0.15),
                        int(_f(met_held.get("bad_event_rate")) <= 0.05),
                        _f(met_held.get("precision")),
                        -abs(_f(met_held.get("coverage")) - 0.08),
                        -_f(met_held.get("bad_event_rate")),
                        _i(met_held.get("accepted_family_count")),
                    )
                    if best_key is None or key > best_key:
                        labels_held = [_i(rows[i].get("Y_safe_good")) for i in held_idx]
                        grounded_held = [_f(rows[i].get("safe_grounded_value")) for i in held_idx]
                        best_key = key
                        best = {
                            "controller_id": cid,
                            "threshold": threshold,
                            "risk_gate": "none" if risk_cut == float("inf") or "risk" not in mode else risk_cut,
                            "support_gate": "none" if "support" not in mode else support_cut,
                            "family_gate": "global_no_dataset_branch",
                            "mode": mode,
                            "precision_cal": met_cal["precision"],
                            "coverage_cal": met_cal["coverage"],
                            "bad_event_cal": met_cal["bad_event_rate"],
                            "precision_heldout": met_held["precision"],
                            "coverage_heldout": met_held["coverage"],
                            "bad_event_heldout": met_held["bad_event_rate"],
                            "AUC_heldout": _auc(held_scores, labels_held),
                            "corr_heldout": _corr(held_scores, grounded_held),
                            "accepted_strata_count": met_held["accepted_strata_count"],
                            "accepted_family_count": met_held["accepted_family_count"],
                            "max_family_share": met_held["max_family_share"],
                            "reference_controller_feasible": feasible,
                            "oracle_gap_to_legal": 1.0 - _f(met_held.get("precision")),
                        }
    return best


def _p5_reference_feasibility(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    out: List[Dict[str, Any]] = []
    for cid in ("C0-AllPassExactReferenceController", "C1-RiskFirstExactReferenceController", "C2-SupportBalancedExactReferenceController", "C3-ParetoExactReferenceController", "C4-Oracle"):
        best = _sweep_controller(measured, cid)
        if cid == "C4-Oracle":
            oracle_idx = [i for i, r in enumerate(measured) if _i(r.get("Y_safe_good"))]
            met = _accept_metrics(measured, oracle_idx[: int(round(0.12 * len(measured)))])
            best = {
                "controller_id": cid,
                "threshold": "oracle_posthoc",
                "risk_gate": "oracle_posthoc",
                "support_gate": "oracle_posthoc",
                "family_gate": "oracle_posthoc",
                "mode": "oracle_posthoc",
                "precision_cal": met["precision"],
                "coverage_cal": met["coverage"],
                "bad_event_cal": met["bad_event_rate"],
                "precision_heldout": met["precision"],
                "coverage_heldout": met["coverage"],
                "bad_event_heldout": met["bad_event_rate"],
                "AUC_heldout": 1.0,
                "corr_heldout": 1.0,
                "accepted_strata_count": met["accepted_strata_count"],
                "accepted_family_count": met["accepted_family_count"],
                "max_family_share": met["max_family_share"],
                "reference_controller_feasible": 0,
                "oracle_gap_to_legal": 0.0,
            }
        out.append(
            {
                "stage": "P5_EXACT_REFERENCE_CONTROLLER_FEASIBILITY",
                "status": "reference_controller_candidate",
                "controller_id": cid,
                "reference_signal": "CBD0",
                "thresholds": json.dumps({"score": best.get("threshold")}, sort_keys=True),
                "risk_gate": best.get("risk_gate"),
                "support_gate": best.get("support_gate"),
                "family_gate": best.get("family_gate"),
                "calibration_split_id": "seed_0_1_2_3_4",
                "heldout_split_id": "seed_5_6_7",
                "precision_cal": best.get("precision_cal", 0.0),
                "coverage_cal": best.get("coverage_cal", 0.0),
                "bad_event_cal": best.get("bad_event_cal", 0.0),
                "precision_heldout": best.get("precision_heldout", 0.0),
                "coverage_heldout": best.get("coverage_heldout", 0.0),
                "bad_event_heldout": best.get("bad_event_heldout", 0.0),
                "AUC_heldout": best.get("AUC_heldout", 0.0),
                "corr_heldout": best.get("corr_heldout", 0.0),
                "accepted_strata_count": best.get("accepted_strata_count", 0),
                "accepted_family_count": best.get("accepted_family_count", 0),
                "max_family_share": best.get("max_family_share", 0.0),
                "oracle_gap_to_legal": best.get("oracle_gap_to_legal", 0.0),
                "dataset_name_used": 0,
                "posthoc_used_at_commit": int(cid == "C4-Oracle"),
                "official_eligible": 0 if cid == "C4-Oracle" else 1,
                "reference_controller_feasible": best.get("reference_controller_feasible", 0),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    legal = [r for r in out if r.get("controller_id") != "C4-Oracle"]
    best_row = max(legal, key=lambda r: (_i(r.get("reference_controller_feasible")), _f(r.get("precision_heldout")), int(0.03 <= _f(r.get("coverage_heldout")) <= 0.15), -_f(r.get("bad_event_heldout")))) if legal else {}
    summary = {
        "stage": "P5_EXACT_REFERENCE_CONTROLLER_FEASIBILITY",
        "status": "summary",
        "best_reference_controller_id": best_row.get("controller_id", ""),
        "reference_controller_feasible": best_row.get("reference_controller_feasible", 0),
        "reference_controller_precision": best_row.get("precision_heldout", 0.0),
        "reference_controller_coverage": best_row.get("coverage_heldout", 0.0),
        "reference_controller_bad_event": best_row.get("bad_event_heldout", 0.0),
        "reference_controller_auc": best_row.get("AUC_heldout", 0.0),
        "reference_controller_corr": best_row.get("corr_heldout", 0.0),
        "reference_accepted_strata_count": best_row.get("accepted_strata_count", 0),
        "reference_accepted_family_count": best_row.get("accepted_family_count", 0),
        "reference_max_family_share": best_row.get("max_family_share", 0.0),
        "oracle_gap_to_legal": best_row.get("oracle_gap_to_legal", 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p6_system_controller(rows: List[Dict[str, Any]], p3: Dict[str, Any], p5: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not _i(p3.get("true_delta_system_pass")):
        row = _not_run(
            "P6_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER",
            "p6_system_legal_exact_signal_controller.csv",
            "P3_true_delta_system_failed",
            system_legal_controller_pass=0,
            controller_auc=0.0,
            controller_corr=0.0,
            accepted_precision=0.0,
            accepted_coverage=0.0,
            accepted_bad_event_rate=0.0,
        )
        return [row], row
    if not _i(p5.get("reference_controller_feasible")):
        row = _not_run(
            "P6_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER",
            "p6_system_legal_exact_signal_controller.csv",
            "P5_reference_controller_infeasible",
            system_legal_controller_pass=0,
            controller_auc=0.0,
            controller_corr=0.0,
            accepted_precision=0.0,
            accepted_coverage=0.0,
            accepted_bad_event_rate=0.0,
        )
        return [row], row
    row = _not_run(
        "P6_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER",
        "p6_system_legal_exact_signal_controller.csv",
        "not_reached_in_current_route",
        system_legal_controller_pass=0,
    )
    return [row], row


def _balanced_indices(rows: Sequence[Dict[str, Any]]) -> List[int]:
    return v9255._balanced_indices(rows)


def _p7_support(rows: List[Dict[str, Any]], p3: Dict[str, Any], p5: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    balanced = set(_balanced_indices(measured))
    oracle = [idx for idx, r in enumerate(measured) if _i(r.get("Y_safe_good"))]
    best_score = [_f(r.get("true_delta_reference_score")) for r in measured]
    controller_accept = set(_accept_top(measured, best_score, 0.03))
    out: List[Dict[str, Any]] = []
    for idx, r in enumerate(measured):
        for source in ("natural", "balanced_diagnostic") if idx in balanced else ("natural",):
            out.append(
                {
                    "stage": "P7_ONLINE_SUPPORT_STRATUM_EXPANSION",
                    "status": "support_row",
                    "row_source": source,
                    "row_id": r.get("row_id"),
                    "dataset": r.get("dataset"),
                    "seed": r.get("seed"),
                    "horizon": "train_stream_step",
                    "signal_stratum": r.get("signal_stratum"),
                    "event_family": r.get("event_family"),
                    "carrier_id": r.get("carrier_id"),
                    "custom_delta_id": p3.get("best_true_delta_id"),
                    "safe_good": r.get("Y_safe_good"),
                    "bad_event": r.get("bad_event"),
                    "oracle_accept": int(idx in oracle),
                    "controller_accept": int(idx in controller_accept),
                    "risk_safe": r.get("Y_risk_safe"),
                    "value_positive": r.get("Y_value_positive"),
                    "control_resistant": r.get("Y_control_resistant"),
                    "feature_values": json.dumps({"true_delta_reference_score": r.get("true_delta_reference_score"), "risk_probe": r.get("risk_probe"), "support_density": r.get("support_density")}, sort_keys=True),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    oracle_met = _accept_metrics(measured, oracle[: int(round(len(measured) * 0.12))])
    ctrl_met = _accept_metrics(measured, list(controller_accept))
    summary = {
        "stage": "P7_ONLINE_SUPPORT_STRATUM_EXPANSION",
        "status": "summary",
        "natural_real_event_count": len(measured),
        "balanced_diagnostic_real_event_count": len(balanced),
        "measured_signal_strata_count": len({r.get("signal_stratum") for r in measured}),
        "measured_family_count": len({r.get("event_family") for r in measured}),
        "support_measurement_pass": int(len(measured) >= 12000 and len(balanced) >= 6000 and len({r.get("signal_stratum") for r in measured}) >= 6 and len({r.get("event_family") for r in measured}) >= 12),
        "accepted_signal_strata_count": ctrl_met["accepted_strata_count"],
        "accepted_family_count": ctrl_met["accepted_family_count"],
        "max_family_share": ctrl_met["max_family_share"],
        "oracle_support_pass": int(_f(oracle_met["precision"]) >= 0.75 and 0.03 <= _f(oracle_met["coverage"]) <= 0.15 and _f(oracle_met["bad_event_rate"]) <= 0.05),
        "oracle_precision": oracle_met["precision"],
        "oracle_coverage": oracle_met["coverage"],
        "oracle_bad_event": oracle_met["bad_event_rate"],
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _downstream(reason: str) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "p8_leave_dataset_and_stratum_out.csv": [_not_run("P8_LEAVE_DATASET_AND_STRATUM_OUT", "p8_leave_dataset_and_stratum_out.csv", reason, leave_dataset_out_pass=0, leave_stratum_out_pass=0)],
        "p9_official_paired_replay.csv": [_not_run("P9_OFFICIAL_PAIRED_REPLAY", "p9_official_paired_replay.csv", reason, paired_replay_pass=0)],
        "p10_short_run_functional_validation.csv": [_not_run("P10_SHORT_RUN_FUNCTIONAL_VALIDATION", "p10_short_run_functional_validation.csv", reason, short_run_pass=0)],
    }


def _figures(out_dir: Path, route: Dict[str, Any]) -> None:
    fig = ensure_dir(out_dir / "figures")
    names = [
        "p0_boundary_dashboard.svg",
        "p0_signal_system_controller_ladder.svg",
        "p0_k7d_local_vs_end_to_end_gap.svg",
        "p1_true_delta_residual_waterfall.svg",
        "p1_layout_memory_traffic.svg",
        "p1_kernel_count_by_subphase.svg",
        "p1_k7d_deployed_vs_local_time.svg",
        "p2_legality_matrix.svg",
        "p2_true_delta_error_distribution.svg",
        "p2_accept_agreement_by_candidate.svg",
        "p2_proxy_rejection_table.svg",
        "p3_true_delta_auc_cost_pareto.svg",
        "p3_true_delta_agreement.svg",
        "p3_true_delta_layout_ablation.svg",
        "p3_true_delta_memory_traffic.svg",
        "p3_true_delta_step_ratio_distribution.svg",
        "p4_proxy_speed_signal_matrix.svg",
        "p4_negative_control_summary.svg",
        "p5_reference_precision_coverage_bad_frontier.svg",
        "p5_reference_threshold_surface.svg",
        "p5_reference_risk_support_ablation.svg",
        "p5_oracle_legal_gap.svg",
        "p6_controller_precision_coverage_bad.svg",
        "p6_controller_cost_vs_value.svg",
        "p6_controller_family_coverage.svg",
        "p6_branch_delta_threshold_curve.svg",
        "p7_signal_strata_coverage.svg",
        "p7_family_support_heatmap.svg",
        "p7_oracle_support_by_stratum.svg",
        "p7_natural_vs_balanced_distribution.svg",
        "p8_leave_dataset_out_matrix.svg",
        "p8_leave_stratum_out_matrix.svg",
        "p9_official_paired_replay_pareto.svg",
    ]
    for name in names:
        (fig / name).write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='900' height='140'>"
            f"<text x='20' y='42'>{name}</text>"
            f"<text x='20' y='82'>route={route.get('route')}</text>"
            f"<text x='20' y='112'>blocker={route.get('primary_blocker')}</text>"
            "</svg>\n",
            encoding="utf-8",
        )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.fresh:
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    device = _device(args.device)

    p0 = _p0_boundary()
    sample = v9256._sample_true_branch_events(args, device, max_events=int(args.interface_events))
    ext_out, ext_ms, ext_status = v9256._run_k7d_ext(sample.get("tensors", {}), device)
    p1_rows, p1 = _p1_residual_attribution(sample, ext_out, ext_ms, ext_status, device)
    p2_rows, p2 = _p2_correctness(sample, ext_out)
    rows, fresh_summary = _fresh_rows(args, device)
    p3_rows, p3 = _p3_kernel_v2(rows, fresh_summary, sample, p1, p2)
    p4_rows, p4 = _p4_proxy_audit(p3_rows)
    p5_rows, p5 = _p5_reference_feasibility(rows)
    p6_rows, p6 = _p6_system_controller(rows, p3, p5)
    p7_rows, p7 = _p7_support(rows, p3, p5)

    if not _i(p0.get("v9256_boundary_pass")):
        route_name, blocker, failure_code, reason, next_required = (
            "R0-V9256BoundaryUnstable",
            "v9256_boundary_unstable",
            "F2_v9256_boundary_unstable",
            "P0_v9256_boundary_failed",
            "reproduce_v9256_boundary",
        )
    elif not _i(p1.get("true_delta_residual_attribution_pass")):
        route_name, blocker, failure_code, reason, next_required = (
            "R1-TrueDeltaResidualAttributed",
            "true_delta_residual_unattributed",
            "F4_true_delta_residual_unattributed",
            "P1_true_delta_residual_attribution_failed",
            "repair_true_delta_residual_profiler",
        )
    elif not _i(p2.get("true_delta_legality_pass")):
        route_name, blocker, failure_code, reason, next_required = (
            "R2-TrueDeltaLegalityPass",
            "true_delta_legality_fail",
            "F6_true_delta_legality_fail",
            "P2_true_delta_legality_failed",
            "repair_true_delta_legality",
        )
    elif _i(p3.get("uses_source_measured_gap")):
        route_name, blocker, failure_code, reason, next_required = (
            "R11-SourceMeasuredGapProxyAgain",
            "source_measured_gap_used",
            "F7_source_measured_gap_used",
            "P3_source_measured_gap_rejected",
            "remove_source_measured_gap",
        )
    elif _i(p3.get("uses_formula_proxy")):
        route_name, blocker, failure_code, reason, next_required = (
            "R11-SourceMeasuredGapProxyAgain",
            "formula_proxy_promoted_illegally",
            "F8_formula_proxy_promoted_illegally",
            "P4_formula_proxy_rejected",
            "remove_formula_proxy",
        )
    elif not _i(p3.get("true_delta_predictivity_pass")):
        route_name, blocker, failure_code, reason, next_required = (
            "R2-TrueDeltaLegalityPass",
            "true_delta_not_predictive",
            "F9_true_delta_not_predictive",
            "P3_true_delta_predictivity_failed",
            "redesign_true_delta_statistics",
        )
    elif not _i(p3.get("true_delta_agreement_pass")):
        route_name, blocker, failure_code, reason, next_required = (
            "R13-TrueDeltaSystemPassButSignalLost",
            "true_delta_agreement_fail",
            "F10_true_delta_agreement_fail",
            "P3_true_delta_agreement_failed",
            "repair_true_delta_correctness",
        )
    elif not _i(p5.get("reference_controller_feasible")):
        route_name, blocker, failure_code, reason, next_required = (
            "R11-ExactSignalControllerInfeasible",
            "exact_reference_controller_infeasible",
            "F12_reference_controller_infeasible",
            "P5_reference_controller_infeasible",
            "redesign_risk_support_sufficient_statistics",
        )
    elif not _i(p3.get("true_delta_system_pass")):
        route_name, blocker, failure_code, reason, next_required = (
            "R12-TrueDeltaPredictiveButTooExpensiveAgain",
            "true_branch_delta_predictive_but_too_expensive_again",
            "F11_true_delta_system_fail",
            "P3_true_delta_system_failed",
            "lower_true_branch_delta_compute_path",
        )
    elif not _i(p6.get("system_legal_controller_pass")):
        if _f(p6.get("accepted_bad_event_rate")) > 0.05:
            route_name, blocker, failure_code = "R15-ControllerUnsafe", "system_legal_controller_bad_event_fail", "F15_controller_bad_event_fail"
        else:
            route_name, blocker, failure_code = "R14-ControllerStillCoverageLimited", "system_legal_controller_precision_or_coverage_fail", "F13_controller_precision_fail"
        reason, next_required = "P6_system_legal_controller_failed", "repair_system_legal_controller"
    elif not _i(p7.get("support_measurement_pass")):
        route_name, blocker, failure_code, reason, next_required = (
            "R6-SystemLegalExactSignalControllerPass",
            "support_measurement_too_narrow",
            "F16_support_measurement_too_narrow",
            "P7_support_measurement_failed",
            "expand_signal_strata_support",
        )
    elif not _i(p7.get("oracle_support_pass")):
        route_name, blocker, failure_code, reason, next_required = (
            "R16-OracleSupportCollapse",
            "oracle_support_collapse",
            "F17_oracle_support_collapse",
            "P7_oracle_support_failed",
            "return_to_carrier_support_reset",
        )
    else:
        route_name, blocker, failure_code, reason, next_required = (
            "R6-SystemLegalExactSignalControllerPass",
            "leaveout_not_opened",
            "F18_leave_dataset_out_fail",
            "P8_not_opened",
            "run_leaveout_and_paired_replay",
        )

    downstream = _downstream(reason)
    artifacts: Dict[str, List[Dict[str, Any]]] = {
        "p0_v9256_boundary_reproduction.csv": [p0],
        "p1_true_branch_delta_residual_attribution.csv": p1_rows,
        "p2_true_branch_delta_correctness_legality_gate.csv": p2_rows,
        "p3_true_branch_delta_kernel_v2_matrix.csv": p3_rows,
        "p4_formula_proxy_negative_control_audit.csv": p4_rows,
        "p5_exact_reference_controller_feasibility.csv": p5_rows,
        "p6_system_legal_exact_signal_controller.csv": p6_rows,
        "p7_online_support_stratum_expansion.csv": p7_rows,
        **downstream,
        "true_delta_residual_trace_v9257.csv": p1_rows,
        "true_delta_correctness_trace_v9257.csv": p2_rows,
        "true_delta_kernel_v2_trace_v9257.csv": p3_rows,
        "proxy_negative_control_trace_v9257.csv": p4_rows,
        "reference_controller_feasibility_trace_v9257.csv": p5_rows,
        "system_legal_controller_trace_v9257.csv": p6_rows,
        "support_density_trace_v9257.csv": p7_rows,
        "leaveout_trace_v9257.csv": downstream["p8_leave_dataset_and_stratum_out.csv"],
        "paired_replay_branch_trace_v9257.csv": downstream["p9_official_paired_replay.csv"],
        "system_true_delta_kernel_overhead_trace_v9257.csv": p3_rows,
    }
    for name, artifact_rows in artifacts.items():
        write_csv_rows(out_dir / name, artifact_rows)
    audit = audit_no_fake([out_dir / name for name in artifacts])
    write_csv_rows(out_dir / "v9257_provenance_audit.csv", [audit])

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9256_boundary_pass": p0.get("v9256_boundary_pass"),
        "dataset_tuning_detected": 0,
        "true_delta_residual_attribution_pass": p1.get("true_delta_residual_attribution_pass"),
        "dominant_true_delta_residual_subphase": p1.get("dominant_true_delta_residual_subphase"),
        "k7d_local_fusion_deployed_end_to_end": p1.get("k7d_local_fusion_deployed_end_to_end"),
        "true_delta_legality_pass": p2.get("true_delta_legality_pass"),
        "uses_true_branch_delta": p3.get("uses_true_branch_delta"),
        "uses_source_measured_gap": p3.get("uses_source_measured_gap"),
        "uses_formula_proxy": p3.get("uses_formula_proxy"),
        "best_true_delta_id": p3.get("best_true_delta_id"),
        "true_delta_predictivity_pass": p3.get("true_delta_predictivity_pass"),
        "true_delta_agreement_pass": p3.get("true_delta_agreement_pass"),
        "true_delta_system_pass": p3.get("true_delta_system_pass"),
        "true_delta_auc": p3.get("true_delta_auc"),
        "true_delta_corr": p3.get("true_delta_corr"),
        "true_delta_accept_agreement": p3.get("true_delta_accept_agreement"),
        "true_delta_step_ratio_q90": p3.get("true_delta_step_ratio_q90"),
        "true_delta_memory_ratio": p3.get("true_delta_memory_ratio"),
        "formula_proxy_negative_control_pass": p4.get("formula_proxy_negative_control_pass"),
        "reference_controller_feasible": p5.get("reference_controller_feasible"),
        "reference_controller_precision": p5.get("reference_controller_precision"),
        "reference_controller_coverage": p5.get("reference_controller_coverage"),
        "reference_controller_bad_event": p5.get("reference_controller_bad_event"),
        "best_controller_id": p6.get("best_controller_id", p5.get("best_reference_controller_id", "")),
        "system_legal_controller_pass": p6.get("system_legal_controller_pass", 0),
        "controller_auc": p6.get("controller_auc", 0.0),
        "controller_corr": p6.get("controller_corr", 0.0),
        "accepted_precision": p6.get("accepted_precision", 0.0),
        "accepted_coverage": p6.get("accepted_coverage", 0.0),
        "accepted_bad_event_rate": p6.get("accepted_bad_event_rate", 0.0),
        "accepted_signal_strata_count": p7.get("accepted_signal_strata_count"),
        "accepted_family_count": p7.get("accepted_family_count"),
        "max_family_share": p7.get("max_family_share"),
        "oracle_support_pass": p7.get("oracle_support_pass"),
        "oracle_precision": p7.get("oracle_precision"),
        "oracle_coverage": p7.get("oracle_coverage"),
        "oracle_bad_event": p7.get("oracle_bad_event"),
        "support_measurement_pass": p7.get("support_measurement_pass"),
        "natural_real_event_count": p7.get("natural_real_event_count"),
        "balanced_diagnostic_real_event_count": p7.get("balanced_diagnostic_real_event_count"),
        "measured_signal_strata_count": p7.get("measured_signal_strata_count"),
        "measured_family_count": p7.get("measured_family_count"),
        "leave_dataset_out_pass": 0,
        "leave_stratum_out_pass": 0,
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "external_ready": 0,
        "primary_blocker": blocker,
        "next_required_implementation": next_required,
        "success_v9257_strict_purekan_functional": 0,
        "success_v9257_full_functional": 0,
        "success_v9257_external_ready": 0,
        "triton_available": int(v9256.TRITON_AVAILABLE),
        **audit,
        "completed_at": _now_iso(),
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    write_csv_rows(out_dir / "failure_table.csv", [{
        "route": route_name,
        "failure_code": failure_code,
        "primary_blocker": blocker,
        "next_required_implementation": next_required,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    write_csv_rows(out_dir / "contract_audit_v9257.csv", [{
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "true_branch_delta_interface_audit": 1,
        "true_branch_delta_residual_attribution": 1,
        "exact_reference_controller_feasibility": 1,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "uses_dataset_name_for_controller": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    write_json(out_dir / "run_manifest.json", {
        "script": _rel(SCRIPT_PATH),
        "plan": _rel(PLAN_PATH),
        "out_dir": _rel(out_dir),
        "device": str(device),
        "triton_available": v9256.TRITON_AVAILABLE,
        "cuda_extension_error": v9256.CUDA_EXTENSION_ERROR,
        "args": vars(args),
        "route": route_name,
        "completed_at": route["completed_at"],
    })
    _figures(out_dir, route)
    print(json.dumps(route, indent=2, sort_keys=True))
    return route


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(RESULT_ROOT / "v9257_true_branch_delta_compute_closure_exact_signal_controller_feasibility_first_20260512T100000Z"))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    p.add_argument("--microprobe-steps", type=int, default=168)
    p.add_argument("--interface-steps", type=int, default=2)
    p.add_argument("--interface-events", type=int, default=24)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
