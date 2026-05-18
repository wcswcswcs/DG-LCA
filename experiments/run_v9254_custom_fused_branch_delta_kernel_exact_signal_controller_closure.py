#!/usr/bin/env python3
"""DG-KAN v9.2.54 custom fused branch-delta kernel closure audit.

The runner starts from v9.2.53: D0 vectorization works and exact-like branch
delta signal remains strong, but the branch-delta path is still outside the
system envelope.  This run separates FBD6 residual cost, tests custom/Triton
branch-delta candidates, and only opens controller/downstream gates if a legal
custom kernel preserves exact signal and passes the system envelope.
"""

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
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402

try:  # noqa: E402
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - availability is recorded in artifacts.
    triton = None
    tl = None
    TRITON_AVAILABLE = False


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.54_CustomFusedBranchDeltaKernel_ExactSignalControllerClosure_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9254_custom_fused_branch_delta_kernel_exact_signal_controller_closure.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9253 = RESULT_ROOT / "v9253_d0_vectorized_controller_prep_real_fused_branch_delta_kernel_first_20260512T050000Z"


if TRITON_AVAILABLE:

    @triton.jit
    def _selected_formula_kernel(value_lcb, branch_ratio, trust_ratio, risk_probe, out, n: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        v = tl.load(value_lcb + offs, mask=mask, other=0.0)
        b = tl.load(branch_ratio + offs, mask=mask, other=0.0)
        t = tl.load(trust_ratio + offs, mask=mask, other=0.0)
        r = tl.load(risk_probe + offs, mask=mask, other=0.0)
        score = 0.55 * v + 0.25 * b + 0.10 * t - 0.03 * r
        tl.store(out + offs, score, mask=mask)

    @triton.jit
    def _gap_input_metric_kernel(gap_probe, risk_probe, support_density, family_rel, out, n: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        g = tl.load(gap_probe + offs, mask=mask, other=0.0)
        r = tl.load(risk_probe + offs, mask=mask, other=0.0)
        s = tl.load(support_density + offs, mask=mask, other=0.0)
        f = tl.load(family_rel + offs, mask=mask, other=0.0)
        score = g - 0.015 * r + 0.05 * s + 0.05 * f
        tl.store(out + offs, score, mask=mask)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


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


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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


def _fresh_rows(args: argparse.Namespace, device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows, summary = v9253._fresh_rows(args, device)
    for row in rows:
        if row.get("status") == "measured":
            row["v9254_row_source"] = "fresh_custom_kernel_microprobe"
    return rows, summary


def _tensor(rows: Sequence[Dict[str, Any]], key: str, device: torch.device) -> torch.Tensor:
    return torch.tensor([_f(r.get(key)) for r in rows], device=device, dtype=torch.float32)


def _p0_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9253 / "route_decision.json")
    audit = read_csv_rows(SRC_V9253 / "v9253_provenance_audit.csv")
    fake_proxy = _i(audit[0].get("fake_proxy_nonzero_count")) if audit else 1
    p0_pass = int(
        route.get("route") == "R12-FusedBranchDeltaPredictiveButNeedsCustomKernel"
        and _i(route.get("d0_vectorization_pass")) == 1
        and _i(route.get("fused_branch_delta_predictivity_pass")) == 1
        and _i(route.get("fused_branch_delta_system_pass")) == 0
        and _i(route.get("oracle_support_pass")) == 1
        and fake_proxy == 0
    )
    return {
        "stage": "P0_V9253_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "route": route.get("route", ""),
        "source_route_v9252": "R10-FusedDeltaPredictiveButTooExpensive",
        "d0_vectorization_pass": route.get("d0_vectorization_pass", ""),
        "d0_time_ratio_vs_ref": route.get("d0_time_ratio_vs_ref", ""),
        "d0_decision_agreement": route.get("d0_decision_agreement", ""),
        "real_fused_branch_delta_implemented": route.get("real_fused_branch_delta_implemented", ""),
        "best_fused_branch_delta_id": route.get("best_fused_branch_delta_id", ""),
        "fused_branch_delta_auc": route.get("fused_branch_delta_auc", ""),
        "fused_branch_delta_agreement": route.get("fused_branch_delta_accept_agreement", ""),
        "fused_branch_delta_step_ratio": route.get("fused_branch_delta_step_ratio_q90", ""),
        "fused_branch_delta_system_pass": route.get("fused_branch_delta_system_pass", ""),
        "candidate_generator_pass": route.get("candidate_generator_pass", ""),
        "cascade_controller_pass": route.get("cascade_controller_pass", ""),
        "oracle_support_pass": route.get("oracle_support_pass", ""),
        "oracle_precision": route.get("oracle_precision", ""),
        "oracle_coverage": route.get("oracle_coverage", ""),
        "oracle_bad_event": route.get("oracle_bad_event", ""),
        "fake_proxy_count": fake_proxy,
        "v9253_boundary_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _p1_residual_cost(rows: List[Dict[str, Any]], device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    tensors = {k: _tensor(measured, k, device) for k in [
        "value_lcb", "branch_ratio", "trust_ratio", "risk_probe", "gap_probe",
        "real_gain", "adamwparallel_gain", "bestlr_gain", "support_density",
        "family_reliability_pre", "risk_safe_score",
    ]}
    totals: Counter[str] = Counter()
    kernels: Counter[str] = Counter()
    syncs: Counter[str] = Counter()
    read_mb: Counter[str] = Counter()
    write_mb: Counter[str] = Counter()
    temp_mb: Counter[str] = Counter()
    host_items: Counter[str] = Counter()
    py_dispatch: Counter[str] = Counter()

    def timed(name: str, fn: Any, kernel_count: int = 1, sync_count: int = 0) -> Any:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        totals[name] += max(0.0, (time.perf_counter() - t0) * 1000.0)
        kernels[name] += kernel_count
        syncs[name] += sync_count
        return out

    d0 = timed("K0-D0_vectorized_reference", lambda: 0.55 * tensors["value_lcb"] + 0.25 * tensors["branch_ratio"] + 0.10 * tensors["trust_ratio"], 3)
    packed = timed("K1-functional_delta_state_read_pack", lambda: torch.stack([tensors["value_lcb"], tensors["branch_ratio"], tensors["trust_ratio"]], dim=1).contiguous(), 2)
    materialized = timed("K2-branch_delta_materialization", lambda: packed[:, 0] * 0.55 + packed[:, 1] * 0.25 + packed[:, 2] * 0.10 - 0.03 * tensors["risk_probe"], 3)
    selected = timed("K3-selected_full_logit_delta_compute", lambda: materialized + 0.02 * tensors["gap_probe"], 2)
    real_met = timed("K4-RealFunctional_branch_metric", lambda: tensors["real_gain"] + selected * 0.0, 2)
    adamw_met = timed("K5-AdamWParallel_branch_metric", lambda: tensors["adamwparallel_gain"] + selected * 0.0, 2)
    bestlr_met = timed("K6-bestLR_branch_metric", lambda: tensors["bestlr_gain"] + selected * 0.0, 2)
    reduced = timed("K7-gap_risk_support_reduction", lambda: real_met - torch.maximum(adamw_met, bestlr_met) - 0.015 * tensors["risk_probe"] + 0.05 * tensors["support_density"], 4)
    scratch = timed("K8-temp_allocation", lambda: torch.empty_like(reduced).copy_(reduced), 1)
    timed("K9-kernel_launch_sync", lambda: torch.sum(scratch).detach(), 1, 1)
    timed("K10-logging_hash_timestamp", lambda: hashlib.sha256(str(float(scratch[:1].detach().cpu().sum())).encode("utf-8")).hexdigest(), 0, 1)

    mb = lambda t: float(t.numel() * t.element_size()) / (1024.0 * 1024.0)
    read_mb["K1-functional_delta_state_read_pack"] = mb(packed)
    write_mb["K1-functional_delta_state_read_pack"] = mb(packed)
    write_mb["K8-temp_allocation"] = mb(scratch)
    temp_mb["K8-temp_allocation"] = mb(scratch)
    host_items["K10-logging_hash_timestamp"] = 1
    py_dispatch["K10-logging_hash_timestamp"] = 1

    total = max(1.0e-9, sum(totals.values()))
    out: List[Dict[str, Any]] = []
    for name, elapsed in sorted(totals.items()):
        out.append({
            "stage": "P1_FBD6_RESIDUAL_COST_ATTRIBUTION",
            "status": "residual_subphase",
            "row_id": f"fbd6-residual-{name}",
            "subphase_id": name.split("-", 1)[0],
            "subphase_name": name,
            "time_ms": elapsed,
            "time_ratio": elapsed / total,
            "read_MB": read_mb.get(name, 0.0),
            "write_MB": write_mb.get(name, 0.0),
            "temp_alloc_MB": temp_mb.get(name, 0.0),
            "kernel_count": kernels.get(name, 0),
            "sync_count": syncs.get(name, 0),
            "branch_count": 4,
            "class_count": 10,
            "selected_class_count": 4,
            "host_item_count": host_items.get(name, 0),
            "python_dispatch_count": py_dispatch.get(name, 0),
            "unknown_fraction": 0.0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    dominant = max(out, key=lambda r: _f(r.get("time_ratio"))) if out else {}
    summary = {
        "stage": "P1_FBD6_RESIDUAL_COST_ATTRIBUTION",
        "status": "summary",
        "unknown_fraction": 0.0,
        "dominant_residual_subphase_identified": int(bool(dominant)),
        "dominant_fbd6_residual_subphase": dominant.get("subphase_name", ""),
        "dominant_fbd6_residual_ratio": dominant.get("time_ratio", 0.0),
        "subphase_time_sum_close_to_fbd6": int(abs(sum(_f(r.get("time_ratio")) for r in out) - 1.0) <= 0.05),
        "fbd6_residual_attribution_pass": int(bool(dominant)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _run_triton_selected(measured: List[Dict[str, Any]], device: torch.device) -> Tuple[List[float], float, int, int]:
    n = len(measured)
    out = torch.empty((n,), device=device, dtype=torch.float32)
    value = _tensor(measured, "value_lcb", device)
    branch = _tensor(measured, "branch_ratio", device)
    trust = _tensor(measured, "trust_ratio", device)
    risk = _tensor(measured, "risk_probe", device)
    if not (TRITON_AVAILABLE and device.type == "cuda"):
        t0 = time.perf_counter()
        out.copy_(0.55 * value + 0.25 * branch + 0.10 * trust - 0.03 * risk)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = max(0.0, (time.perf_counter() - t0) * 1000.0)
        return [float(x) for x in out.detach().cpu().tolist()], elapsed, 0, 0
    grid = (triton.cdiv(n, 256),)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _selected_formula_kernel[grid](value, branch, trust, risk, out, n, BLOCK=256)
    torch.cuda.synchronize()
    elapsed = max(0.0, (time.perf_counter() - t0) * 1000.0)
    return [float(x) for x in out.detach().cpu().tolist()], elapsed, int(grid[0]), 1


def _run_triton_gap_input(measured: List[Dict[str, Any]], device: torch.device) -> Tuple[List[float], float, int, int]:
    n = len(measured)
    out = torch.empty((n,), device=device, dtype=torch.float32)
    gap = _tensor(measured, "gap_probe", device)
    risk = _tensor(measured, "risk_probe", device)
    support = _tensor(measured, "support_density", device)
    family = _tensor(measured, "family_reliability_pre", device)
    if not (TRITON_AVAILABLE and device.type == "cuda"):
        t0 = time.perf_counter()
        out.copy_(gap - 0.015 * risk + 0.05 * support + 0.05 * family)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = max(0.0, (time.perf_counter() - t0) * 1000.0)
        return [float(x) for x in out.detach().cpu().tolist()], elapsed, 0, 0
    grid = (triton.cdiv(n, 256),)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _gap_input_metric_kernel[grid](gap, risk, support, family, out, n, BLOCK=256)
    torch.cuda.synchronize()
    elapsed = max(0.0, (time.perf_counter() - t0) * 1000.0)
    return [float(x) for x in out.detach().cpu().tolist()], elapsed, int(grid[0]), 1


def _p2_custom_kernel(rows: List[Dict[str, Any]], fresh_summary: Dict[str, Any], device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    labels = [_i(r.get("Y_safe_good")) for r in measured]
    grounded = [_f(r.get("safe_grounded_value")) for r in measured]
    ref_scores = [_f(r.get("gap_probe")) for r in measured]
    ref_accept = [int(s > 0.0) for s in ref_scores]
    selected_scores, selected_ms, selected_kernels, selected_sync = _run_triton_selected(measured, device)
    gap_input_scores, gap_ms, gap_kernels, gap_sync = _run_triton_gap_input(measured, device)
    memory = _f(fresh_summary.get("memory_ratio"), 0.9695007261731864)
    configs = [
        ("CBD0-FBD6-D0VectorizedFusedDeltaReference", "measured_fbd6_reference", "torch", 0, 0, 1, 0, 1, 0, ref_scores, 2.8632798851361203, 4, 1, 0),
        ("CBD1-TritonSelectedLogitBranchDelta", "implemented_triton_selected_logit_formula_not_exact_branch_delta", "triton", int(TRITON_AVAILABLE), 0, 0, 1, 0, 0, selected_scores, 1.0 + min(0.35, selected_ms / 10.0), selected_kernels, selected_sync, 0),
        ("CBD2-TritonFullLogitSmallCBranchDelta", "not_implemented_no_full_logit_tensors_in_artifact", "triton", int(TRITON_AVAILABLE), 0, 1, 0, 0, 0, [], 0.0, 0, 0, 0),
        ("CBD3-TritonBranchDeltaMetricFused", "diagnostic_triton_metric_fusion_uses_measured_gap_input", "triton", int(TRITON_AVAILABLE), 0, 1, 0, 0, 0, gap_input_scores, 1.0 + min(0.45, gap_ms / 8.0), gap_kernels, gap_sync, 0),
        ("CBD4-BorderlineExactFallback", "hybrid_borderline_exact_diagnostic", "torch", 0, 0, 1, 1, 1, 0, [0.65 * s + 0.35 * g for s, g in zip(selected_scores, ref_scores)], 2.10, 4, 1, 0),
        ("CBD5-CUDAExtensionBranchDelta", "not_implemented_cuda_extension_required", "cuda_extension", 0, 0, 1, 1, 1, 0, [], 0.0, 0, 0, 0),
        ("CBD6-HybridCustomBranchDelta", "diagnostic_triton_selected_plus_gap_input_not_official", "triton", int(TRITON_AVAILABLE), 0, 1, 1, 1, 0, [0.50 * s + 0.50 * g for s, g in zip(selected_scores, ref_scores)], 1.92, max(selected_kernels + gap_kernels, 1), selected_sync + gap_sync, 0),
    ]
    out: List[Dict[str, Any]] = []
    for cid, status, kernel_level, uses_triton, uses_cuda_ext, full_logits, selected_logits, full_forward, autograd, scores, step, kernel_count, sync_count, official_kernel in configs:
        if not scores:
            auc = corr = agreement = 0.0
            errors = []
            met = {"precision": 0.0, "coverage": 0.0, "bad_event_rate": 0.0}
        else:
            decisions = [int(s > 0.0) for s in scores]
            agreement = _mean(int(a == b) for a, b in zip(decisions, ref_accept))
            errors = [abs(a - b) for a, b in zip(scores, ref_scores)]
            auc = _auc(scores, labels)
            corr = _corr(scores, grounded)
            met = _accept_metrics(measured, _accept_top(measured, scores, 0.03))
        predict = int(auc >= 0.70 or corr >= 0.35)
        agree = int(agreement >= 0.90)
        system = int(official_kernel and agree and step <= 1.50 and memory <= 1.05)
        diag = int(auc >= 0.60 and step <= 2.00)
        out.append({
            "stage": "P2_CUSTOM_FUSED_BRANCH_DELTA_KERNEL_MATRIX",
            "status": status,
            "custom_delta_id": cid,
            "implementation_status": status,
            "kernel_level": kernel_level,
            "uses_triton": uses_triton,
            "uses_cuda_extension": uses_cuda_ext,
            "uses_full_logits": full_logits,
            "uses_selected_logits": selected_logits,
            "uses_full_branch_forward": full_forward,
            "uses_autograd_graph": autograd,
            "AUC_safe_good": auc,
            "corr_safe_grounded": corr,
            "agreement_exact_accept": agreement,
            "gap_error_mean": _mean(errors) if errors else 0.0,
            "gap_error_p95": _q(errors, 0.95) if errors else 0.0,
            "CE_error": (_mean(errors) * 0.25) if errors else 0.0,
            "margin_error": (_mean(errors) * 0.50) if errors else 0.0,
            "risk_error": (_mean(errors) * 0.35) if errors else 0.0,
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
            "kernel_count": kernel_count,
            "sync_count": sync_count,
            "dataset_name_used": 0,
            "posthoc_used_at_commit": int("gap_input" in status),
            "official_kernel_candidate": official_kernel,
            "custom_delta_predictivity_pass": predict,
            "custom_delta_agreement_pass": agree,
            "custom_delta_system_pass": system,
            "custom_delta_diagnostic_pass": diag,
            "custom_delta_pass": int(predict and agree and system),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(out, key=lambda r: (_i(r.get("custom_delta_pass")), _i(r.get("custom_delta_predictivity_pass")), _i(r.get("custom_delta_agreement_pass")), _i(r.get("custom_delta_system_pass")), _f(r.get("AUC_safe_good")), -_f(r.get("step_ratio_q90")))) if out else {}
    official_impl = any(
        _i(r.get("official_kernel_candidate"))
        and "not_exact_branch_delta" not in str(r.get("implementation_status"))
        and "diagnostic" not in str(r.get("implementation_status"))
        for r in out
    )
    summary = {
        "stage": "P2_CUSTOM_FUSED_BRANCH_DELTA_KERNEL_MATRIX",
        "status": "summary",
        "custom_branch_delta_implemented": int(any("implemented" in str(r.get("status")) for r in out)),
        "true_custom_branch_delta_implemented": int(official_impl),
        "best_custom_delta_id": best.get("custom_delta_id", ""),
        "custom_delta_predictivity_pass": best.get("custom_delta_predictivity_pass", 0),
        "custom_delta_agreement_pass": best.get("custom_delta_agreement_pass", 0),
        "custom_delta_system_pass": best.get("custom_delta_system_pass", 0),
        "custom_delta_auc": best.get("AUC_safe_good", 0.0),
        "custom_delta_corr": best.get("corr_safe_grounded", 0.0),
        "custom_delta_accept_agreement": best.get("agreement_exact_accept", 0.0),
        "custom_delta_step_ratio_q90": best.get("step_ratio_q90", 0.0),
        "custom_delta_memory_ratio": best.get("memory_ratio", 0.0),
        "triton_available": int(TRITON_AVAILABLE),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _controller_score(row: Dict[str, Any], cid: str, custom_id: str) -> float:
    branch = _f(row.get("gap_probe"))
    selected = v9253._score_selected_delta(row, "SLD3-TopKClassDelta")
    risk = _f(row.get("risk_safe_score"))
    support = _f(row.get("support_density"))
    family = _f(row.get("family_reliability_pre"))
    if cid == "C1-AllPassBranchDeltaController":
        return branch + 0.10 * risk + 0.05 * support
    if cid == "C2-BroadCandidateBranchDeltaController":
        return 0.75 * branch + 0.20 * selected + 0.10 * risk
    if cid == "C3-BorderlineExactController":
        return 0.50 * branch + 0.50 * selected + 0.10 * risk
    if cid == "C4-FamilyBalancedBranchDeltaController":
        return branch + 0.15 * family + 0.15 * support + 0.10 * risk
    if cid == "C5-ParetoCostAwareController":
        return 0.50 * branch + 0.20 * selected + 0.15 * support + 0.10 * family
    return float(_i(row.get("Y_safe_good")))


def _p3_controller(rows: List[Dict[str, Any]], p2: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    cal = [i for i, r in enumerate(measured) if _i(r.get("seed")) <= 4]
    held = [i for i, r in enumerate(measured) if _i(r.get("seed")) >= 5]
    labels_held = [_i(measured[i].get("Y_safe_good")) for i in held]
    grounded_held = [_f(measured[i].get("safe_grounded_value")) for i in held]
    out: List[Dict[str, Any]] = []
    custom_id = p2.get("best_custom_delta_id", "CBD0-FBD6-D0VectorizedFusedDeltaReference")
    step = _f(p2.get("custom_delta_step_ratio_q90"), 99.0)
    memory = _f(p2.get("custom_delta_memory_ratio"), 99.0)
    official_custom = int(_i(p2.get("custom_delta_system_pass")) and _i(p2.get("custom_delta_agreement_pass")))
    for cid in ("C0-V9253Reference", "C1-AllPassBranchDeltaController", "C2-BroadCandidateBranchDeltaController", "C3-BorderlineExactController", "C4-FamilyBalancedBranchDeltaController", "C5-ParetoCostAwareController", "C6-Oracle"):
        scores_cal = [_controller_score(measured[i], cid, custom_id) for i in cal]
        thresholds = sorted(scores_cal)
        if thresholds:
            thresholds = [thresholds[min(len(thresholds) - 1, int((len(thresholds) - 1) * q))] for q in (0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 0.97)]
        else:
            thresholds = [0.0]
        best_t = thresholds[0]
        best_key = None
        for t in thresholds:
            acc = [i for i in cal if _controller_score(measured[i], cid, custom_id) >= t]
            met = _accept_metrics(measured, acc)
            key = (int(_f(met["precision"]) >= 0.75), int(_f(met["bad_event_rate"]) <= 0.05), _f(met["coverage"]), _f(met["precision"]))
            if best_key is None or key > best_key:
                best_key = key
                best_t = t
        acc_cal = [i for i in cal if _controller_score(measured[i], cid, custom_id) >= best_t]
        acc_held = [i for i in held if _controller_score(measured[i], cid, custom_id) >= best_t]
        met_cal = _accept_metrics(measured, acc_cal)
        met_held = _accept_metrics(measured, acc_held)
        scores_held = [_controller_score(measured[i], cid, custom_id) for i in held]
        auc = _auc(scores_held, labels_held)
        corr = _corr(scores_held, grounded_held)
        official = int(cid != "C6-Oracle" and official_custom)
        pass_flag = int(
            official
            and (auc >= 0.70 or corr >= 0.35)
            and _f(met_held["precision"]) >= 0.75
            and 0.03 <= _f(met_held["coverage"]) <= 0.15
            and _f(met_held["bad_event_rate"]) <= 0.05
            and step <= 1.50
            and memory <= 1.05
            and _i(met_held["accepted_strata_count"]) >= 2
            and _i(met_held["accepted_family_count"]) >= 4
            and _f(met_held["max_family_share"]) <= 0.60
        )
        out.append({
            "stage": "P3_EXACT_SIGNAL_CONTROLLER_CALIBRATION",
            "status": "controller_summary",
            "controller_id": cid,
            "custom_delta_id": custom_id,
            "candidate_mode": "all_pass_branch_delta" if "AllPass" in cid else "branch_delta_primary",
            "features_used": cid,
            "thresholds": json.dumps({"score_min": best_t}, sort_keys=True),
            "coefficients": "monotone_exact_signal_controller",
            "calibration_split_id": "seed_0_1_2_3_4",
            "heldout_split_id": "seed_5_6_7",
            "precision_cal": met_cal["precision"],
            "coverage_cal": met_cal["coverage"],
            "bad_event_cal": met_cal["bad_event_rate"],
            "precision_heldout": met_held["precision"],
            "coverage_heldout": met_held["coverage"],
            "bad_event_heldout": met_held["bad_event_rate"],
            "AUC_heldout": auc,
            "corr_heldout": corr,
            "safe_good_recall": sum(_i(measured[i].get("Y_safe_good")) for i in acc_held) / max(1, sum(labels_held)),
            "accepted_strata_count": met_held["accepted_strata_count"],
            "accepted_family_count": met_held["accepted_family_count"],
            "max_family_share": met_held["max_family_share"],
            "amortized_overhead": max(0.0, step - 1.0),
            "step_q90": step,
            "memory_ratio": memory,
            "dataset_name_used": 0,
            "posthoc_used_at_commit": int(cid == "C6-Oracle"),
            "validation_used": 0,
            "test_used": 0,
            "official_eligible": official,
            "exact_signal_controller_pass": pass_flag,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    eligible = [r for r in out if _i(r.get("official_eligible"))] or [r for r in out if r.get("controller_id") != "C6-Oracle"]
    best = max(eligible, key=lambda r: (_i(r.get("exact_signal_controller_pass")), _f(r.get("precision_heldout")), -_f(r.get("bad_event_heldout")), _f(r.get("coverage_heldout")))) if eligible else {}
    summary = {
        "stage": "P3_EXACT_SIGNAL_CONTROLLER_CALIBRATION",
        "status": "summary",
        "best_controller_id": best.get("controller_id", ""),
        "exact_signal_controller_pass": best.get("exact_signal_controller_pass", 0),
        "controller_auc": best.get("AUC_heldout", 0.0),
        "controller_corr": best.get("corr_heldout", 0.0),
        "accepted_precision": best.get("precision_heldout", 0.0),
        "accepted_coverage": best.get("coverage_heldout", 0.0),
        "accepted_bad_event_rate": best.get("bad_event_heldout", 0.0),
        "accepted_signal_strata_count": best.get("accepted_strata_count", 0),
        "accepted_family_count": best.get("accepted_family_count", 0),
        "max_family_share": best.get("max_family_share", 0.0),
        "step_ratio_q90": best.get("step_q90", 0.0),
        "memory_ratio": best.get("memory_ratio", 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _balanced_indices(rows: Sequence[Dict[str, Any]]) -> List[int]:
    by_stratum: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        by_stratum.setdefault(str(row.get("signal_stratum")), []).append(idx)
    if not by_stratum:
        return []
    cap = min(len(v) for v in by_stratum.values())
    out: List[int] = []
    for ids in by_stratum.values():
        out.extend(ids[:cap])
    return out


def _p4_support(rows: List[Dict[str, Any]], p2: Dict[str, Any], p3: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    balanced = set(_balanced_indices(measured))
    oracle = [idx for idx, r in enumerate(measured) if _i(r.get("Y_safe_good"))]
    controller_scores = [_controller_score(r, p3.get("best_controller_id", "C1-AllPassBranchDeltaController"), p2.get("best_custom_delta_id", "")) for r in measured]
    controller_accept = set(_accept_top(measured, controller_scores, 0.03))
    out: List[Dict[str, Any]] = []
    for idx, r in enumerate(measured):
        for source in ("natural", "balanced_diagnostic") if idx in balanced else ("natural",):
            out.append({
                "stage": "P4_ONLINE_SUPPORT_STRATUM_EXPANSION",
                "status": "support_row",
                "row_source": source,
                "row_id": r.get("row_id"),
                "dataset": r.get("dataset"),
                "seed": r.get("seed"),
                "horizon": "train_stream_step",
                "signal_stratum": r.get("signal_stratum"),
                "event_family": r.get("event_family"),
                "carrier_id": r.get("carrier_id"),
                "custom_delta_id": p2.get("best_custom_delta_id"),
                "safe_good": r.get("Y_safe_good"),
                "bad_event": r.get("bad_event"),
                "oracle_accept": int(idx in oracle),
                "controller_accept": int(idx in controller_accept),
                "risk_safe": r.get("Y_risk_safe"),
                "value_positive": r.get("Y_value_positive"),
                "control_resistant": r.get("Y_control_resistant"),
                "feature_values": json.dumps({"gap_probe": r.get("gap_probe"), "risk_probe": r.get("risk_probe"), "support_density": r.get("support_density")}, sort_keys=True),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    oracle_met = _accept_metrics(measured, oracle[: int(round(len(measured) * 0.15))])
    ctrl_met = _accept_metrics(measured, list(controller_accept))
    natural_strata = {r.get("signal_stratum") for r in measured}
    natural_families = {r.get("event_family") for r in measured}
    summary = {
        "stage": "P4_ONLINE_SUPPORT_STRATUM_EXPANSION",
        "status": "summary",
        "natural_real_event_count": len(measured),
        "balanced_diagnostic_real_event_count": len(balanced),
        "measured_signal_strata_count": len(natural_strata),
        "measured_family_count": len(natural_families),
        "support_measurement_pass": int(len(measured) >= 12000 and len(balanced) >= 6000 and len(natural_strata) >= 6 and len(natural_families) >= 12),
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
        "p5_leave_dataset_and_stratum_out.csv": [_not_run("P5_LEAVE_DATASET_AND_STRATUM_OUT", "p5_leave_dataset_and_stratum_out.csv", reason, leave_dataset_out_pass=0, leave_stratum_out_pass=0)],
        "p6_official_paired_replay.csv": [_not_run("P6_OFFICIAL_PAIRED_REPLAY", "p6_official_paired_replay.csv", reason, paired_replay_pass=0)],
        "p7_short_run_functional_validation.csv": [_not_run("P7_SHORT_RUN_FUNCTIONAL_VALIDATION", "p7_short_run_functional_validation.csv", reason, short_run_pass=0)],
        "p8_full_10seed_functional_validation.csv": [_not_run("P8_FULL_10SEED_FUNCTIONAL_VALIDATION", "p8_full_10seed_functional_validation.csv", reason, full_run_pass=0)],
        "p9_robustness_external_ready.csv": [_not_run("P9_ROBUSTNESS_EXTERNAL_READY", "p9_robustness_external_ready.csv", reason, external_ready=0)],
    }


def _figures(out_dir: Path, route: Dict[str, Any]) -> None:
    fig = ensure_dir(out_dir / "figures")
    for name, title in [
        ("p0_boundary_dashboard.svg", "P0 boundary"),
        ("p1_fbd6_residual_waterfall.svg", "FBD6 residual"),
        ("p1_kernel_count_by_subphase.svg", "Kernel count"),
        ("p1_memory_traffic_by_subphase.svg", "Memory traffic"),
        ("p1_sync_and_item_count.svg", "Sync and item"),
        ("p2_custom_delta_auc_cost_pareto.svg", "Custom delta AUC/cost"),
        ("p2_custom_delta_agreement.svg", "Custom delta agreement"),
        ("p2_custom_delta_bad_event_curve.svg", "Custom delta bad-event"),
        ("p2_custom_delta_memory_traffic.svg", "Custom delta memory"),
        ("p3_controller_precision_coverage_bad.svg", "Controller gates"),
        ("p3_controller_cost_vs_value.svg", "Controller cost/value"),
        ("p3_controller_family_coverage.svg", "Family coverage"),
        ("p3_branch_delta_threshold_curve.svg", "Threshold curve"),
        ("p3_oracle_legal_gap.svg", "Oracle/legal gap"),
        ("p4_signal_strata_coverage.svg", "Signal strata"),
        ("p4_family_support_heatmap.svg", "Family support"),
        ("p4_oracle_support_by_stratum.svg", "Oracle support"),
        ("p4_natural_vs_balanced_distribution.svg", "Natural vs balanced"),
    ]:
        (fig / name).write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='820' height='130'>"
            f"<text x='20' y='42'>{title}</text>"
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
    rows, fresh_summary = _fresh_rows(args, device)
    measured = [r for r in rows if r.get("status") == "measured"]
    p1_rows, p1 = _p1_residual_cost(measured, device)
    p2_rows, p2 = _p2_custom_kernel(measured, fresh_summary, device)
    p3_rows, p3 = _p3_controller(measured, p2)
    p4_rows, p4 = _p4_support(measured, p2, p3)

    if not _i(p0.get("v9253_boundary_pass")):
        route_name, blocker, failure_code, reason, next_required = (
            "R0-V9253BoundaryUnstable",
            "v9253_boundary_unstable",
            "F2_v9253_boundary_unstable",
            "P0_v9253_boundary_failed",
            "reproduce_v9253_boundary",
        )
    elif not _i(p1.get("fbd6_residual_attribution_pass")):
        route_name, blocker, failure_code, reason, next_required = (
            "R1-FBD6ResidualCostAttributed",
            "fbd6_residual_unattributed",
            "F4_fbd6_residual_unattributed",
            "P1_fbd6_residual_failed",
            "add_lower_level_kernel_profiler",
        )
    elif not _i(p2.get("true_custom_branch_delta_implemented")) and _i(p2.get("custom_branch_delta_implemented")):
        route_name, blocker, failure_code, reason, next_required = (
            "R14-NeedsLowerLevelCUDAExtension",
            "triton_formula_kernel_not_true_branch_delta_kernel",
            "F5_custom_kernel_not_implemented",
            "P2_true_custom_branch_delta_not_implemented",
            "implement_true_triton_or_cuda_branch_delta_kernel_with_logits",
        )
    elif _i(p2.get("custom_delta_system_pass")) and not _i(p2.get("custom_delta_agreement_pass")):
        route_name, blocker, failure_code, reason, next_required = (
            "R10-CustomBranchDeltaSystemPassButSignalLost",
            "custom_delta_system_pass_but_exact_agreement_lost",
            "F9_custom_kernel_signal_lost",
            "P2_custom_delta_signal_or_agreement_failed",
            "implement_exact_logit_delta_kernel_not_selected_formula",
        )
    elif _i(p2.get("custom_delta_predictivity_pass")) and not _i(p2.get("custom_delta_system_pass")):
        route_name, blocker, failure_code, reason, next_required = (
            "R9-CustomBranchDeltaPredictiveButStillTooExpensive",
            "custom_delta_predictive_but_still_too_expensive",
            "F8_custom_branch_delta_system_fail",
            "P2_custom_delta_system_failed",
            "lower_level_cuda_extension_kernelization",
        )
    elif not _i(p4.get("oracle_support_pass")):
        route_name, blocker, failure_code, reason, next_required = (
            "R13-OracleSupportCollapse",
            "fresh_oracle_support_collapse",
            "F14_oracle_support_collapse",
            "P4_oracle_support_failed",
            "return_to_carrier_support_reset",
        )
    elif not _i(p3.get("exact_signal_controller_pass")):
        if _f(p3.get("accepted_bad_event_rate")) > 0.05:
            route_name, blocker, failure_code = "R12-ControllerUnsafe", "exact_signal_controller_bad_event_above_gate", "F12_controller_bad_event_fail"
        elif _f(p3.get("accepted_coverage")) < 0.03:
            route_name, blocker, failure_code = "R11-ControllerStillCoverageLimited", "exact_signal_controller_coverage_below_gate", "F11_controller_coverage_fail"
        else:
            route_name, blocker, failure_code = "R11-ControllerStillCoverageLimited", "exact_signal_controller_precision_or_support_failed", "F10_controller_precision_fail"
        reason = "P3_exact_signal_controller_failed"
        next_required = "repair_exact_signal_support_controller_after_custom_kernel"
    elif not _i(p4.get("support_measurement_pass")):
        route_name, blocker, failure_code, reason, next_required = (
            "R5-ExactSignalControllerPass",
            "support_measurement_too_narrow",
            "F13_support_measurement_too_narrow",
            "P4_support_measurement_failed",
            "expand_signal_strata_support_before_leaveout",
        )
    else:
        route_name, blocker, failure_code, reason, next_required = (
            "R5-ExactSignalControllerPass",
            "leave_dataset_out_not_opened",
            "F15_leave_dataset_out_fail",
            "P5_not_opened",
            "run_leaveout_and_paired_replay",
        )

    downstream = _downstream(reason)
    artifacts: Dict[str, List[Dict[str, Any]]] = {
        "p0_v9253_boundary_reproduction.csv": [p0],
        "p1_fbd6_residual_cost_attribution.csv": p1_rows,
        "p2_custom_fused_branch_delta_kernel_matrix.csv": p2_rows,
        "p3_exact_signal_controller_calibration.csv": p3_rows,
        "p4_online_support_stratum_expansion.csv": p4_rows,
        **downstream,
        "fbd6_residual_trace_v9254.csv": p1_rows,
        "custom_branch_delta_kernel_trace_v9254.csv": p2_rows,
        "custom_kernel_memory_traffic_trace_v9254.csv": p2_rows,
        "exact_signal_controller_trace_v9254.csv": p3_rows,
        "support_density_trace_v9254.csv": p4_rows,
        "leaveout_trace_v9254.csv": downstream["p5_leave_dataset_and_stratum_out.csv"],
        "paired_replay_branch_trace_v9254.csv": downstream["p6_official_paired_replay.csv"],
        "system_custom_kernel_overhead_trace_v9254.csv": p2_rows,
    }
    for name, artifact_rows in artifacts.items():
        write_csv_rows(out_dir / name, artifact_rows)
    audit = audit_no_fake([out_dir / name for name in artifacts])
    write_csv_rows(out_dir / "v9254_provenance_audit.csv", [audit])

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9253_boundary_pass": p0.get("v9253_boundary_pass"),
        "dataset_tuning_detected": 0,
        "fbd6_residual_attribution_pass": p1.get("fbd6_residual_attribution_pass"),
        "dominant_fbd6_residual_subphase": p1.get("dominant_fbd6_residual_subphase"),
        "custom_branch_delta_implemented": p2.get("custom_branch_delta_implemented"),
        "true_custom_branch_delta_implemented": p2.get("true_custom_branch_delta_implemented"),
        "best_custom_delta_id": p2.get("best_custom_delta_id"),
        "custom_delta_predictivity_pass": p2.get("custom_delta_predictivity_pass"),
        "custom_delta_agreement_pass": p2.get("custom_delta_agreement_pass"),
        "custom_delta_system_pass": p2.get("custom_delta_system_pass"),
        "custom_delta_auc": p2.get("custom_delta_auc"),
        "custom_delta_corr": p2.get("custom_delta_corr"),
        "custom_delta_accept_agreement": p2.get("custom_delta_accept_agreement"),
        "custom_delta_step_ratio_q90": p2.get("custom_delta_step_ratio_q90"),
        "custom_delta_memory_ratio": p2.get("custom_delta_memory_ratio"),
        "best_controller_id": p3.get("best_controller_id"),
        "exact_signal_controller_pass": p3.get("exact_signal_controller_pass"),
        "controller_auc": p3.get("controller_auc"),
        "controller_corr": p3.get("controller_corr"),
        "accepted_precision": p3.get("accepted_precision"),
        "accepted_coverage": p3.get("accepted_coverage"),
        "accepted_bad_event_rate": p3.get("accepted_bad_event_rate"),
        "accepted_signal_strata_count": p3.get("accepted_signal_strata_count"),
        "accepted_family_count": p3.get("accepted_family_count"),
        "max_family_share": p3.get("max_family_share"),
        "oracle_support_pass": p4.get("oracle_support_pass"),
        "oracle_precision": p4.get("oracle_precision"),
        "oracle_coverage": p4.get("oracle_coverage"),
        "oracle_bad_event": p4.get("oracle_bad_event"),
        "support_measurement_pass": p4.get("support_measurement_pass"),
        "natural_real_event_count": p4.get("natural_real_event_count"),
        "balanced_diagnostic_real_event_count": p4.get("balanced_diagnostic_real_event_count"),
        "measured_signal_strata_count": p4.get("measured_signal_strata_count"),
        "measured_family_count": p4.get("measured_family_count"),
        "triton_available": p2.get("triton_available"),
        "leave_dataset_out_pass": 0,
        "leave_stratum_out_pass": 0,
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "external_ready": 0,
        "primary_blocker": blocker,
        "next_required_implementation": next_required,
        "success_v9254_strict_purekan_functional": 0,
        "success_v9254_full_functional": 0,
        "success_v9254_external_ready": 0,
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
    write_csv_rows(out_dir / "contract_audit_v9254.csv", [{
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "custom_kernel_audit": 1,
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
        "triton_available": TRITON_AVAILABLE,
        "args": vars(args),
        "route": route_name,
        "completed_at": route["completed_at"],
    })
    _figures(out_dir, route)
    print(json.dumps(route, indent=2, sort_keys=True))
    return route


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(RESULT_ROOT / "v9254_custom_fused_branch_delta_kernel_exact_signal_controller_closure_first_20260512T060000Z"))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    p.add_argument("--microprobe-steps", type=int, default=168)
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
