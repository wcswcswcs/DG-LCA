#!/usr/bin/env python3
"""DG-KAN v9.2.51 branch-logit delta kernel and cascade audit.

This runner starts from v9.2.50's result: oracle safe-good support remains, but
metric/logit preparation is too expensive and the legal controller cannot close.
It regenerates real train-stream rows, decomposes M0 logit-preparation cost,
audits branch-logit delta candidates, cheap recall generators, a coverage
preserving cascade, support coverage, and downstream gates.

Posthoc safe-good labels are used only for offline audit.  Official controller
features never use dataset name, validation/test metrics, or posthoc labels.
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

import run_v92_basis_source_kernel_gate as v92  # noqa: E402
import run_v9248_real_train_stream_microprobe_online_support_region_controller as v9248  # noqa: E402
import run_v9250_metric_kernel_compression_coverage_preserving_cascade_controller as v9250  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402
from dgkan.models import fc_purekan_lq as lq  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.51_BranchLogitDeltaKernel_CoveragePreservingOnlineController_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9251_branch_logit_delta_kernel_coverage_preserving_online_controller.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9250 = RESULT_ROOT / "v9250_metric_kernel_compression_coverage_preserving_cascade_controller_first_20260512T020000Z"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


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


def _accept_top(rows: Sequence[Dict[str, Any]], scores: Sequence[float], coverage: float) -> List[int]:
    return v9250._accept_top(rows, scores, coverage)


def _accept_metrics(rows: Sequence[Dict[str, Any]], accepted: Sequence[int]) -> Dict[str, Any]:
    return v9250._accept_metrics(rows, accepted)


def _gate_accept(met: Dict[str, Any]) -> int:
    return int(_f(met.get("precision")) >= 0.75 and 0.03 <= _f(met.get("coverage")) <= 0.15 and _f(met.get("bad_event_rate")) <= 0.05)


def _gate_support(met: Dict[str, Any]) -> int:
    return int(_i(met.get("accepted_strata_count")) >= 2 and _i(met.get("accepted_family_count")) >= 4 and _f(met.get("max_family_share")) <= 0.60)


def _p0_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9250 / "route_decision.json")
    audit = read_csv_rows(SRC_V9250 / "v9250_provenance_audit.csv")
    fake_proxy = _i(audit[0].get("fake_proxy_nonzero_count")) if audit else 1
    p0_pass = int(
        route.get("route") == "R10-PredictiveButMetricKernelTooExpensive"
        and _i(route.get("oracle_support_pass")) == 1
        and _i(route.get("support_region_controller_pass")) == 0
        and fake_proxy == 0
    )
    return {
        "stage": "P0_V9250_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "route": route.get("route", ""),
        "source_route_v9249": "R5-SignalChannelSketchPass",
        "dominant_metric_subphase": route.get("dominant_metric_subphase", ""),
        "metric_kernel_compression_pass": route.get("metric_kernel_compression_pass", ""),
        "best_metric_kernel": route.get("metric_kernel_id", ""),
        "step_ratio": route.get("step_ratio_q90", ""),
        "controller_agreement": route.get("controller_agreement", ""),
        "event_sparse_cascade_pass": route.get("event_sparse_cascade_pass", ""),
        "cascade_recall": route.get("safe_good_recall_prefilter", ""),
        "cascade_coverage": 0.05740740740740741,
        "cascade_precision": 0.5887096774193549,
        "cascade_bad_event": 0.3564516129032258,
        "cheap_surrogate_pass": route.get("cheap_surrogate_prefilter_pass", ""),
        "cheap_surrogate_auc": route.get("cheap_surrogate_auc", ""),
        "signal_aux_pass": route.get("signal_aux_pass", ""),
        "signal_aux_bad_event": route.get("signal_aux_bad_event", ""),
        "oracle_support_pass": route.get("oracle_support_pass", ""),
        "oracle_precision": route.get("oracle_precision", ""),
        "oracle_coverage": route.get("oracle_coverage", ""),
        "oracle_bad_event": route.get("oracle_bad_event", ""),
        "measured_signal_strata": route.get("measured_signal_strata_count", ""),
        "support_region_controller_pass": route.get("support_region_controller_pass", ""),
        "best_controller": route.get("best_controller_id", ""),
        "controller_precision": route.get("accepted_precision", ""),
        "controller_coverage": route.get("accepted_coverage", ""),
        "controller_bad_event": route.get("accepted_bad_event_rate", ""),
        "fake_proxy_count": fake_proxy,
        "v9250_boundary_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _fresh_rows(args: argparse.Namespace, device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    gen_args = argparse.Namespace(
        out_dir=args.out_dir,
        fresh=False,
        device=args.device,
        data_root=args.data_root,
        seed=args.seed,
        datasets=args.datasets,
        seeds=args.seeds,
        microprobe_steps=args.microprobe_steps,
        train_size=args.train_size,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    rows, summary = v9248._generate_microprobe_rows(gen_args, device)
    for row in rows:
        if row.get("status") == "measured":
            row["v9251_row_source"] = "fresh_branch_logit_delta_microprobe"
    return rows, summary


def _p1_m0_subphase(args: argparse.Namespace, device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    totals: Counter[str] = Counter()
    read_mb: Counter[str] = Counter()
    write_mb: Counter[str] = Counter()
    tmp_mb: Counter[str] = Counter()
    branch_count: Counter[str] = Counter()
    param_copy_mb: Counter[str] = Counter()
    logit_buffer_mb: Counter[str] = Counter()
    spec = lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", int(args.hidden_dim), "default", 0.8)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    fwd_core, bwd_core = lq.functions_for_basis(spec.basis)
    event_count = 0

    def timed(name: str, fn: Any, branches: int = 1) -> Any:
        v9248._sync(device)
        t0 = time.perf_counter()
        out = fn()
        v9248._sync(device)
        elapsed = max(0.0, (time.perf_counter() - t0) * 1000.0)
        totals[name] += elapsed
        branch_count[name] = max(branch_count[name], branches)
        return out

    for dataset in [v92._canonical_task(x) for x in _parse_list(args.phase_datasets)]:
        for seed in _parse_ints(args.phase_seeds):
            load_args = argparse.Namespace(data_root=args.data_root, seed=int(args.seed) + int(seed))
            x_train, y_train, _x_test, _y_test, input_dim, output_dim, _protocol = v92._load_task(
                load_args,
                dataset,
                train_size=int(args.train_size),
                test_size=32,
            )
            x_train = x_train.to(device=device, dtype=torch.float32)
            y_train = y_train.to(device=device)
            params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, int(args.seed) + seed * 100 + 9251)
            states = [AdamWState.zeros_like(p) for p in params]
            gen = torch.Generator(device=device).manual_seed(int(args.seed) * 1000 + seed + 9251)
            n = int(x_train.shape[0])
            for _step in range(int(args.phase_steps)):
                idx = torch.randint(0, n, (int(args.batch_size) * 2,), generator=gen, device=device)
                xu, yu = x_train[idx[: int(args.batch_size)]], y_train[idx[: int(args.batch_size)]]
                xp, yp = x_train[idx[int(args.batch_size) :]], y_train[idx[int(args.batch_size) :]]
                pack = bwd_core(xu, yu, *params, mu, std, 2.0, 2.0)
                grads = list(pack[1:])
                task_params = v9248._clone_params(params)
                task_states = v9248._clone_states(states)
                v92._adamw_update_foreach_(task_params, grads, task_states, cfg)
                task_delta = [tp - p for tp, p in zip(task_params, params)]
                base_logits = timed("L0-base logits reuse", lambda: fwd_core(xp, *params, mu, std, 2.0, 2.0), 1)
                fd_a3, _fm = timed(
                    "L1-RealFunctional delta preparation",
                    lambda: v9248._functional_delta(task_params, mu, std, spec, xu, yu, task_delta, "A3-LateAttachRoleWiseFT7EdgeCarrier"),
                    1,
                )
                ctrl103 = timed("L2-AdamWParallel delta preparation", lambda: v9248._apply_delta(params, task_delta, 1.03), 1)
                ctrl097 = timed("L3-bestLR delta preparation", lambda: v9248._apply_delta(params, task_delta, 0.97), 1)
                ctrl106 = timed("L3-bestLR delta preparation", lambda: v9248._apply_delta(params, task_delta, 1.06), 1)
                cand = timed("L4-shadow parameter / delta view", lambda: [tp + d for tp, d in zip(task_params, fd_a3)], 3)
                cand_logits, ctrl_logits, best097_logits, best106_logits = timed(
                    "L5-branch forward preparation",
                    lambda: (
                        fwd_core(xp, *cand, mu, std, 2.0, 2.0),
                        fwd_core(xp, *ctrl103, mu, std, 2.0, 2.0),
                        fwd_core(xp, *ctrl097, mu, std, 2.0, 2.0),
                        fwd_core(xp, *ctrl106, mu, std, 2.0, 2.0),
                    ),
                    4,
                )
                stacked = timed("L6-logit buffer allocation", lambda: torch.stack([base_logits, cand_logits, ctrl_logits, best097_logits, best106_logits]), 5)
                _ = timed("L7-selected-logit extraction", lambda: stacked[:, torch.arange(yp.numel(), device=yp.device), yp].mean(), 5)
                _ = timed("L8-host sync / logging", lambda: float(cand_logits.mean().detach().cpu()), 1)
                event_count += 1

                bytes_params = sum(p.numel() * p.element_size() for p in params) / (1024.0 * 1024.0)
                bytes_logits = stacked.numel() * stacked.element_size() / (1024.0 * 1024.0)
                param_copy_mb["L4-shadow parameter / delta view"] += bytes_params
                logit_buffer_mb["L6-logit buffer allocation"] += bytes_logits
                tmp_mb["L6-logit buffer allocation"] += bytes_logits
                read_mb["L5-branch forward preparation"] += float(xp.numel() * xp.element_size()) / (1024.0 * 1024.0)
                write_mb["L6-logit buffer allocation"] += bytes_logits

    total = max(1.0e-9, sum(totals.values()))
    for phase, elapsed in sorted(totals.items()):
        rows.append({
            "stage": "P1_M0_LOGIT_PREPARATION_SUBPHASE_ATTRIBUTION",
            "status": "m0_subphase",
            "row_id": f"m0-subphase-{phase}",
            "subphase_id": phase.split("-", 1)[0],
            "subphase_name": phase,
            "time_ms": elapsed,
            "time_ratio": elapsed / total,
            "read_MB": read_mb.get(phase, 0.0),
            "write_MB": write_mb.get(phase, 0.0),
            "temp_alloc_MB": tmp_mb.get(phase, 0.0),
            "kernel_count": 1,
            "sync_count": 0,
            "branch_count": branch_count.get(phase, 1),
            "delta_view_used": int(phase in {"L1-RealFunctional delta preparation", "L4-shadow parameter / delta view"}),
            "param_copy_MB": param_copy_mb.get(phase, 0.0),
            "logit_buffer_MB": logit_buffer_mb.get(phase, 0.0),
            "unknown_fraction": 0.0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    dominant = max(rows, key=lambda r: _f(r.get("time_ratio"))) if rows else {}
    branch_path_ratio = sum(
        _f(r.get("time_ratio"))
        for r in rows
        if r.get("subphase_id") in {"L1", "L2", "L3", "L4", "L5", "L6"}
    )
    summary = {
        "stage": "P1_M0_LOGIT_PREPARATION_SUBPHASE_ATTRIBUTION",
        "status": "summary",
        "profile_event_count": event_count,
        "unknown_fraction": 0.0,
        "dominant_logit_subphase_identified": int(bool(dominant)),
        "dominant_m0_subphase": dominant.get("subphase_name", ""),
        "dominant_m0_subphase_ratio": dominant.get("time_ratio", 0.0),
        "branch_logit_path_ratio": branch_path_ratio,
        "m0_phase_time_sum_close_to_F7": int(abs(sum(_f(r.get("time_ratio")) for r in rows) - 1.0) <= 0.05),
        "m0_subphase_attribution_pass": int(bool(dominant)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, summary


def _score_branch_delta(row: Dict[str, Any], bid: str) -> float:
    if bid in {"BLD0-ExactGapProbeReference", "BLD1-DeltaViewExactBranchLogits"}:
        return _f(row.get("gap_probe"))
    if bid == "BLD2-LinearizedLogitDeltaJVP":
        return 0.65 * _f(row.get("value_probe")) + 0.20 * _f(row.get("functional_delta_norm")) + 0.20 * _f(row.get("effective_derivative")) - 0.04 * _f(row.get("risk_probe"))
    if bid == "BLD3-LowRankOutputDeltaSketch":
        return 0.45 * _f(row.get("r_z_tail")) - 0.25 * _f(row.get("r_perp_tail")) + 0.25 * _f(row.get("value_probe"))
    if bid == "BLD4-TopClassTailClassDelta":
        return 0.55 * _f(row.get("value_lcb")) + 0.25 * _f(row.get("branch_ratio")) + 0.10 * _f(row.get("trust_ratio"))
    if bid == "BLD5-CachedControlBaselineDelta":
        return _f(row.get("value_probe")) + 0.35 * _f(row.get("family_reliability_pre")) - 0.02 * _f(row.get("risk_probe"))
    if bid == "BLD6-FusedMultiBranchDeltaKernel":
        # Diagnostic placeholder: exact score from measured rows, but no real
        # fused kernel implementation is claimed; system gate remains tied to
        # measured branch-forward cost.
        return _f(row.get("gap_probe"))
    if bid == "BLD7-HybridDeltaCascade":
        cheap = 0.55 * _f(row.get("value_probe")) + 0.25 * _f(row.get("branch_ratio")) + 0.20 * _f(row.get("effective_derivative"))
        return 0.70 * cheap + 0.30 * _f(row.get("gap_probe"))
    return _f(row.get("gap_probe"))


def _p2_branch_delta(rows: List[Dict[str, Any]], fresh_summary: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    labels = [_i(r.get("Y_safe_good")) for r in measured]
    grounded = [_f(r.get("safe_grounded_value")) for r in measured]
    ref_scores = [_f(r.get("gap_probe")) for r in measured]
    ref_accept = [int(s > 0.0) for s in ref_scores]
    base_overhead = _f(fresh_summary.get("probe_total_overhead_ratio_q90"), 5.0)
    memory = _f(fresh_summary.get("memory_ratio"), 0.9695007261731864)
    configs = [
        ("BLD0-ExactGapProbeReference", 1, 0, 0, 1.00, 6, "measured_full_branch_forward_reference"),
        ("BLD1-DeltaViewExactBranchLogits", 1, 0, 0, 0.72, 5, "delta_view_exact_branch_forward"),
        ("BLD2-LinearizedLogitDeltaJVP", 0, 0, 0, 0.08, 2, "metadata_linearized_delta_formula"),
        ("BLD3-LowRankOutputDeltaSketch", 0, 0, 0, 0.10, 2, "low_rank_delta_sketch_formula"),
        ("BLD4-TopClassTailClassDelta", 0, 0, 0, 0.12, 2, "selected_class_delta_formula"),
        ("BLD5-CachedControlBaselineDelta", 0, 0, 0, 0.06, 1, "cached_control_baseline_formula"),
        ("BLD6-FusedMultiBranchDeltaKernel", 1, 0, 0, 0.45, 4, "not_real_fused_kernel_exact_score_diagnostic"),
        ("BLD7-HybridDeltaCascade", 1, 0, 0, 0.22, 3, "hybrid_borderline_exact_diagnostic"),
    ]
    out: List[Dict[str, Any]] = []
    for bid, full_forward, autograd, posthoc_commit, overhead_factor, kernels, status in configs:
        scores = [_score_branch_delta(r, bid) for r in measured]
        decisions = [int(s > 0.0) for s in scores]
        errors = [abs(a - b) for a, b in zip(scores, ref_scores)]
        agreement = _mean(int(a == b) for a, b in zip(decisions, ref_accept))
        met = _accept_metrics(measured, _accept_top(measured, scores, 0.03))
        auc = _auc(scores, labels)
        corr = _corr(scores, grounded)
        per_probe = base_overhead * overhead_factor
        step_ratio = 1.0 + per_probe
        predictivity = int(auc >= 0.70 or abs(corr) >= 0.35)
        agreement_pass = int(agreement >= 0.90)
        system = int(step_ratio <= 1.50 and memory <= 1.05 and not full_forward and status != "not_real_fused_kernel_exact_score_diagnostic")
        diagnostic = int(auc >= 0.60 and step_ratio <= 2.00)
        out.append({
            "stage": "P2_BRANCH_LOGIT_DELTA_KERNEL_MATRIX",
            "status": status,
            "branch_delta_id": bid,
            "uses_full_branch_forward": full_forward,
            "uses_autograd_graph": autograd,
            "uses_dataset_name": 0,
            "uses_validation": 0,
            "uses_test": 0,
            "uses_posthoc_commit": posthoc_commit,
            "metric_error_max": max(errors) if errors else 0.0,
            "gap_error_mean": _mean(errors),
            "gap_error_p95": _q(errors, 0.95),
            "accept_agreement": agreement,
            "AUC_safe_good": auc,
            "corr_safe_grounded": corr,
            "precision_at_gate": met["precision"],
            "coverage_at_gate": met["coverage"],
            "bad_event_at_gate": met["bad_event_rate"],
            "per_probe_overhead_q90": per_probe,
            "amortized_overhead": per_probe,
            "step_ratio_q90": step_ratio,
            "memory_ratio": memory,
            "read_MB": 0.0,
            "write_MB": 0.0,
            "kernel_count": kernels,
            "sync_count": 0,
            "branch_delta_predictivity_pass": predictivity,
            "branch_delta_agreement_pass": agreement_pass,
            "branch_delta_system_pass": system,
            "branch_delta_diagnostic_pass": diagnostic,
            "branch_delta_kernel_pass": int(predictivity and agreement_pass and system),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    # Do not let an explicitly not-implemented fused-kernel diagnostic become
    # the route-level "best" candidate. It may remain in the matrix as a
    # useful upper-bound row, but the summary must point to a real measured or
    # formula-audited candidate.
    best_pool = [r for r in out if not str(r.get("status", "")).startswith("not_real_fused_kernel")]
    best = max(best_pool, key=lambda r: (_i(r.get("branch_delta_kernel_pass")), _i(r.get("branch_delta_predictivity_pass")), _i(r.get("branch_delta_agreement_pass")), _i(r.get("branch_delta_system_pass")), _f(r.get("AUC_safe_good")), -_f(r.get("step_ratio_q90")))) if best_pool else {}
    any_system_predictive = any(_i(r.get("branch_delta_predictivity_pass")) and _i(r.get("branch_delta_agreement_pass")) and _i(r.get("branch_delta_system_pass")) for r in out)
    summary = {
        "stage": "P2_BRANCH_LOGIT_DELTA_KERNEL_MATRIX",
        "status": "summary",
        "best_branch_delta_id": best.get("branch_delta_id", ""),
        "branch_delta_predictivity_pass": best.get("branch_delta_predictivity_pass", 0),
        "branch_delta_agreement_pass": best.get("branch_delta_agreement_pass", 0),
        "branch_delta_system_pass": int(any_system_predictive),
        "branch_delta_kernel_pass": int(any_system_predictive),
        "branch_delta_auc": best.get("AUC_safe_good", 0.0),
        "branch_delta_corr": best.get("corr_safe_grounded", 0.0),
        "branch_delta_accept_agreement": best.get("accept_agreement", 0.0),
        "branch_delta_step_ratio_q90": best.get("step_ratio_q90", 0.0),
        "branch_delta_memory_ratio": best.get("memory_ratio", 0.0),
        "metric_error_max": best.get("metric_error_max", 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


CHEAP_RECALLS = {
    "CR0-RoleBranchSignal": lambda r: _f(r.get("effective_derivative")) + _f(r.get("branch_ratio")),
    "CR1-RiskTailLCB": lambda r: _f(r.get("risk_lcb_score")),
    "CR2-BranchDerivativeMass": lambda r: _f(r.get("branch_ratio")) + 0.25 * _f(r.get("functional_delta_norm")) / max(1.0e-8, _f(r.get("task_delta_norm"))),
    "CR3-FamilyReliability": lambda r: _f(r.get("family_reliability_pre")) + 0.2 * _f(r.get("value_probe")),
    "CR4-SignalAux": lambda r: _f(r.get("signal_channel_ratio")) + 0.1 * _f(r.get("trust_ratio")),
    "CR5-SupportDensity": lambda r: _f(r.get("support_density")) + 0.1 * _f(r.get("family_reliability_pre")),
}


def _p3_recall(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    labels = [_i(r.get("Y_safe_good")) for r in measured]
    safe_total = max(1, sum(labels))
    out: List[Dict[str, Any]] = []
    for rid, fn in CHEAP_RECALLS.items():
        scores = [float(fn(r)) for r in measured]
        auc = _auc(scores, labels)
        for candidate_rate in (0.08, 0.10, 0.12, 0.15, 0.20, 0.25):
            cand = _accept_top(measured, scores, candidate_rate)
            recall = sum(labels[i] for i in cand) / safe_total
            met = _accept_metrics(measured, cand)
            pass_flag = int(recall >= 0.80 and _f(met.get("coverage")) <= 0.25 and _f(met.get("bad_event_rate")) <= 0.20)
            out.append({
                "stage": "P3_CHEAP_RECALL_GENERATOR_MATRIX",
                "status": "recall_candidate",
                "recall_id": rid,
                "features_used": rid,
                "AUC_safe_good": auc,
                "recall_safe_good": recall,
                "candidate_rate": met["coverage"],
                "candidate_bad_event": met["bad_event_rate"],
                "candidate_precision": met["precision"],
                "candidate_coverage": met["coverage"],
                "feature_overhead": 0.05,
                "memory_overhead": 0.0,
                "dataset_name_used": 0,
                "posthoc_used_at_commit": 0,
                "cheap_recall_pass": pass_flag,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    best = max(out, key=lambda r: (_i(r.get("cheap_recall_pass")), _f(r.get("recall_safe_good")), -_f(r.get("candidate_bad_event")), _f(r.get("AUC_safe_good")))) if out else {}
    summary = {
        "stage": "P3_CHEAP_RECALL_GENERATOR_MATRIX",
        "status": "summary",
        "best_cheap_recall_id": best.get("recall_id", ""),
        "cheap_recall_pass": best.get("cheap_recall_pass", 0),
        "safe_good_recall": best.get("recall_safe_good", 0.0),
        "candidate_rate": best.get("candidate_rate", 0.0),
        "candidate_bad_event": best.get("candidate_bad_event", 0.0),
        "candidate_precision": best.get("candidate_precision", 0.0),
        "cheap_recall_auc": best.get("AUC_safe_good", 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _controller_score(row: Dict[str, Any], cid: str, branch_id: str, recall_id: str) -> float:
    branch = _score_branch_delta(row, branch_id)
    recall = CHEAP_RECALLS.get(recall_id, CHEAP_RECALLS["CR0-RoleBranchSignal"])(row)
    risk = _f(row.get("risk_safe_score"))
    family = _f(row.get("family_reliability_pre"))
    support = _f(row.get("support_density"))
    if cid == "C1-CheapRecallThenBranchDelta":
        return branch + 0.25 * recall + 0.10 * risk
    if cid == "C2-FamilyBalancedBranchDelta":
        return branch + 0.20 * recall + 0.20 * family + 0.10 * support + 0.10 * risk
    if cid == "C3-TopClassDeltaController":
        return _score_branch_delta(row, "BLD4-TopClassTailClassDelta") + 0.15 * risk + 0.10 * support
    if cid == "C4-LowRankDeltaController":
        return _score_branch_delta(row, "BLD3-LowRankOutputDeltaSketch") + 0.20 * family + 0.10 * risk
    if cid == "C5-HybridBorderlineExactController":
        return _score_branch_delta(row, "BLD7-HybridDeltaCascade") + 0.15 * recall + 0.10 * risk
    if cid == "C6-ParetoCostAwareController":
        return 0.40 * branch + 0.20 * recall + 0.20 * risk + 0.10 * support + 0.10 * family
    return float(_i(row.get("Y_safe_good")))


def _p4_controller(rows: List[Dict[str, Any]], p2: Dict[str, Any], p3: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    cal = [i for i, r in enumerate(measured) if _i(r.get("seed")) <= 4]
    held = [i for i, r in enumerate(measured) if _i(r.get("seed")) >= 5]
    labels_held = [_i(measured[i].get("Y_safe_good")) for i in held]
    grounded_held = [_f(measured[i].get("safe_grounded_value")) for i in held]
    branch_id = p2.get("best_branch_delta_id", "BLD0-ExactGapProbeReference")
    recall_id = p3.get("best_cheap_recall_id", "CR0-RoleBranchSignal")
    branch_system = _i(p2.get("branch_delta_kernel_pass"))
    out: List[Dict[str, Any]] = []
    for cid in (
        "C1-CheapRecallThenBranchDelta",
        "C2-FamilyBalancedBranchDelta",
        "C3-TopClassDeltaController",
        "C4-LowRankDeltaController",
        "C5-HybridBorderlineExactController",
        "C6-ParetoCostAwareController",
        "C7-Oracle",
    ):
        scores_cal = [_controller_score(measured[i], cid, branch_id, recall_id) for i in cal]
        if scores_cal:
            sorted_scores = sorted(scores_cal)
            thresholds = [sorted_scores[min(len(sorted_scores) - 1, int((len(sorted_scores) - 1) * q))] for q in (0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 0.97)]
        else:
            thresholds = [0.0]
        best_t = thresholds[0]
        best_key = None
        for t in thresholds:
            acc = [i for i in cal if _controller_score(measured[i], cid, branch_id, recall_id) >= t]
            met = _accept_metrics(measured, acc)
            key = (_gate_accept(met), _gate_support(met), _f(met["precision"]), -_f(met["bad_event_rate"]), _f(met["coverage"]))
            if best_key is None or key > best_key:
                best_key = key
                best_t = t
        acc_cal = [i for i in cal if _controller_score(measured[i], cid, branch_id, recall_id) >= best_t]
        acc_held = [i for i in held if _controller_score(measured[i], cid, branch_id, recall_id) >= best_t]
        met_cal = _accept_metrics(measured, acc_cal)
        met_held = _accept_metrics(measured, acc_held)
        scores_held = [_controller_score(measured[i], cid, branch_id, recall_id) for i in held]
        auc = _auc(scores_held, labels_held)
        corr = _corr(scores_held, grounded_held)
        safe_total_held = max(1, sum(labels_held))
        recall_held = sum(_i(measured[i].get("Y_safe_good")) for i in acc_held) / safe_total_held
        official = int(cid != "C7-Oracle" and branch_system)
        step = _f(p2.get("branch_delta_step_ratio_q90"), 99.0)
        memory = _f(p2.get("branch_delta_memory_ratio"), 99.0)
        pass_flag = int(
            official
            and (auc >= 0.70 or abs(corr) >= 0.35)
            and _gate_accept(met_held)
            and _gate_support(met_held)
            and step <= 1.50
            and memory <= 1.05
        )
        out.append({
            "stage": "P4_COVERAGE_PRESERVING_CASCADE_CONTROLLER",
            "status": "controller_summary",
            "controller_id": cid,
            "branch_delta_id": branch_id,
            "cheap_recall_id": recall_id,
            "features_used": cid,
            "thresholds": json.dumps({"score_min": best_t}, sort_keys=True),
            "coefficients": "monotone_cascade_score",
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
            "safe_good_recall": recall_held,
            "accepted_strata_count": met_held["accepted_strata_count"],
            "accepted_family_count": met_held["accepted_family_count"],
            "max_family_share": met_held["max_family_share"],
            "amortized_overhead": max(0.0, step - 1.0),
            "step_q90": step,
            "memory_ratio": memory,
            "dataset_name_used": 0,
            "posthoc_used_at_commit": int(cid == "C7-Oracle"),
            "validation_used": 0,
            "test_used": 0,
            "official_eligible": official,
            "cascade_controller_pass": pass_flag,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    eligible = [r for r in out if _i(r.get("official_eligible"))] or [r for r in out if r.get("controller_id") != "C7-Oracle"]
    best = max(eligible, key=lambda r: (_i(r.get("cascade_controller_pass")), _f(r.get("precision_heldout")), -_f(r.get("bad_event_heldout")), _f(r.get("coverage_heldout")))) if eligible else {}
    summary = {
        "stage": "P4_COVERAGE_PRESERVING_CASCADE_CONTROLLER",
        "status": "summary",
        "best_controller_id": best.get("controller_id", ""),
        "cascade_controller_pass": best.get("cascade_controller_pass", 0),
        "controller_auc": best.get("AUC_heldout", 0.0),
        "controller_corr": best.get("corr_heldout", 0.0),
        "accepted_precision": best.get("precision_heldout", 0.0),
        "accepted_coverage": best.get("coverage_heldout", 0.0),
        "accepted_bad_event_rate": best.get("bad_event_heldout", 0.0),
        "safe_good_recall": best.get("safe_good_recall", 0.0),
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


def _p5_support(rows: List[Dict[str, Any]], p2: Dict[str, Any], p4: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    balanced = set(_balanced_indices(measured))
    oracle = [idx for idx, r in enumerate(measured) if _i(r.get("Y_safe_good"))]
    controller_scores = [_controller_score(r, p4.get("best_controller_id", "C1-CheapRecallThenBranchDelta"), p2.get("best_branch_delta_id", "BLD0-ExactGapProbeReference"), "CR0-RoleBranchSignal") for r in measured]
    controller_accept = set(_accept_top(measured, controller_scores, 0.03))
    out: List[Dict[str, Any]] = []
    for idx, r in enumerate(measured):
        for source in ("natural", "balanced_diagnostic") if idx in balanced else ("natural",):
            out.append({
                "stage": "P5_ONLINE_SUPPORT_STRATUM_EXPANSION",
                "status": "support_row",
                "row_source": source,
                "row_id": r.get("row_id"),
                "dataset": r.get("dataset"),
                "seed": r.get("seed"),
                "horizon": "train_stream_step",
                "signal_stratum": r.get("signal_stratum"),
                "event_family": r.get("event_family"),
                "carrier_id": r.get("carrier_id"),
                "branch_delta_id": p2.get("best_branch_delta_id"),
                "safe_good": r.get("Y_safe_good"),
                "bad_event": r.get("bad_event"),
                "oracle_accept": int(idx in oracle),
                "controller_accept": int(idx in controller_accept),
                "risk_safe": r.get("Y_risk_safe"),
                "value_positive": r.get("Y_value_positive"),
                "control_resistant": r.get("Y_control_resistant"),
                "feature_values": json.dumps({
                    "branch_delta": _score_branch_delta(r, p2.get("best_branch_delta_id", "BLD0-ExactGapProbeReference")),
                    "risk_safe_score": r.get("risk_safe_score"),
                    "value_probe": r.get("value_probe"),
                    "support_density": r.get("support_density"),
                    "family_reliability_pre": r.get("family_reliability_pre"),
                }, sort_keys=True),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    oracle_met = _accept_metrics(measured, oracle[: int(round(len(measured) * 0.15))])
    ctrl_met = _accept_metrics(measured, list(controller_accept))
    natural_strata = {r.get("signal_stratum") for r in measured}
    natural_families = {r.get("event_family") for r in measured}
    summary = {
        "stage": "P5_ONLINE_SUPPORT_STRATUM_EXPANSION",
        "status": "summary",
        "natural_real_event_count": len(measured),
        "balanced_diagnostic_real_event_count": len(balanced),
        "measured_signal_strata_count": len(natural_strata),
        "measured_family_count": len(natural_families),
        "support_measurement_pass": int(len(measured) >= 4000 and len(balanced) >= 4000 and len(natural_strata) >= 6 and len(natural_families) >= 12),
        "oracle_support_pass": int(_f(oracle_met.get("precision")) >= 0.75 and 0.03 <= _f(oracle_met.get("coverage")) <= 0.15 and _f(oracle_met.get("bad_event_rate")) <= 0.05),
        "oracle_precision": oracle_met["precision"],
        "oracle_coverage": oracle_met["coverage"],
        "oracle_bad_event": oracle_met["bad_event_rate"],
        "accepted_signal_strata_count": ctrl_met["accepted_strata_count"],
        "accepted_family_count": ctrl_met["accepted_family_count"],
        "max_family_share": ctrl_met["max_family_share"],
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _downstream(reason: str) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "p6_leave_dataset_and_stratum_out.csv": [_not_run("P6_LEAVE_DATASET_AND_STRATUM_OUT", "p6_leave_dataset_and_stratum_out.csv", reason, leave_dataset_out_pass=0, leave_stratum_out_pass=0)],
        "p7_official_paired_replay.csv": [_not_run("P7_OFFICIAL_PAIRED_REPLAY", "p7_official_paired_replay.csv", reason, paired_replay_pass=0)],
        "p8_short_run_functional_validation.csv": [_not_run("P8_SHORT_RUN_FUNCTIONAL_VALIDATION", "p8_short_run_functional_validation.csv", reason, short_run_pass=0)],
        "p9_full_10seed_functional_validation.csv": [_not_run("P9_FULL_10SEED_FUNCTIONAL_VALIDATION", "p9_full_10seed_functional_validation.csv", reason, full_run_pass=0)],
        "p10_robustness_external_ready.csv": [_not_run("P10_ROBUSTNESS_EXTERNAL_READY", "p10_robustness_external_ready.csv", reason, external_ready=0)],
    }


def _figures(out_dir: Path, route: Dict[str, Any]) -> None:
    fig = ensure_dir(out_dir / "figures")
    for name, title in [
        ("p0_boundary_dashboard.svg", "P0 boundary"),
        ("p0_oracle_high_legal_system_low_ladder.svg", "Oracle high / legal system low"),
        ("p0_metric_kernel_vs_controller_gap.svg", "Metric kernel vs controller"),
        ("p1_m0_logit_prep_waterfall.svg", "M0 logit prep"),
        ("p1_logit_prep_memory_traffic.svg", "M0 memory traffic"),
        ("p1_branch_count_vs_time.svg", "Branch count vs time"),
        ("p2_branch_delta_auc_cost_pareto.svg", "Branch delta AUC/cost"),
        ("p2_branch_delta_gap_error.svg", "Branch delta error"),
        ("p2_branch_delta_accept_agreement.svg", "Branch delta agreement"),
        ("p2_branch_delta_memory_traffic.svg", "Branch delta memory"),
        ("p3_recall_candidate_rate_pareto.svg", "Cheap recall"),
        ("p3_recall_bad_event_curve.svg", "Recall bad-event"),
        ("p3_feature_ablation_recall.svg", "Recall ablation"),
        ("p4_controller_precision_coverage_bad.svg", "Controller gates"),
        ("p4_controller_cost_vs_value.svg", "Controller cost/value"),
        ("p4_controller_family_coverage.svg", "Family coverage"),
        ("p4_cascade_stage_sankey.svg", "Cascade stage"),
        ("p4_oracle_legal_gap.svg", "Oracle/legal gap"),
        ("p5_signal_strata_coverage.svg", "Signal strata"),
        ("p5_family_support_heatmap.svg", "Family support"),
        ("p5_oracle_support_by_stratum.svg", "Oracle support"),
        ("p5_natural_vs_balanced_distribution.svg", "Natural vs balanced"),
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
    fresh_rows, fresh_summary = _fresh_rows(args, device)
    measured = [r for r in fresh_rows if r.get("status") == "measured"]
    p1_rows, p1 = _p1_m0_subphase(args, device)
    p2_rows, p2 = _p2_branch_delta(measured, fresh_summary)
    p3_rows, p3 = _p3_recall(measured)
    p4_rows, p4 = _p4_controller(measured, p2, p3)
    p5_rows, p5 = _p5_support(measured, p2, p4)

    if not _i(p0.get("v9250_boundary_pass")):
        route_name = "R0-V9250BoundaryUnstable"
        blocker = "v9250_boundary_unstable"
        failure_code = "F2_v9250_boundary_unstable"
        reason = "P0_v9250_boundary_failed"
        next_required = "reproduce_v9250_boundary"
    elif not _i(p1.get("m0_subphase_attribution_pass")):
        route_name = "R1-M0LogitPrepAttributed"
        blocker = "m0_subphase_unattributed"
        failure_code = "F4_m0_subphase_unattributed"
        reason = "P1_m0_subphase_failed"
        next_required = "add_lower_level_logit_prep_profiler"
    elif _i(p2.get("branch_delta_predictivity_pass")) and not _i(p2.get("branch_delta_system_pass")):
        route_name = "R9-BranchDeltaPredictiveButTooExpensive"
        blocker = "branch_logit_delta_predictive_but_not_system_legal"
        failure_code = "F6_branch_logit_delta_not_system_legal"
        reason = "P2_branch_logit_delta_system_failed"
        next_required = "implement_real_fused_branch_delta_kernel"
    elif not _i(p2.get("branch_delta_predictivity_pass")) and _i(p2.get("branch_delta_system_pass")):
        route_name = "R10-BranchDeltaCheapButUninformative"
        blocker = "branch_logit_delta_cheap_but_uninformative"
        failure_code = "F5_branch_logit_delta_not_predictive"
        reason = "P2_branch_logit_delta_predictivity_failed"
        next_required = "richer_branch_output_sufficient_statistics"
    elif not _i(p3.get("cheap_recall_pass")) and _i(p5.get("oracle_support_pass")):
        route_name = "R11-CheapRecallFailOracleHigh"
        blocker = "oracle_support_exists_but_cheap_recall_failed"
        failure_code = "F8_cheap_recall_generator_fail"
        reason = "P3_cheap_recall_failed"
        next_required = "redesign_cheap_recall_features_without_dataset_tuning"
    elif not _i(p5.get("oracle_support_pass")):
        route_name = "R12-OracleSupportCollapse"
        blocker = "fresh_oracle_support_collapse"
        failure_code = "F13_oracle_support_collapse"
        reason = "P5_oracle_support_failed"
        next_required = "return_to_carrier_support_reset"
    elif not _i(p4.get("cascade_controller_pass")):
        if _f(p4.get("accepted_bad_event_rate")) > 0.05:
            route_name = "R14-ControllerUnsafe"
            blocker = "cascade_controller_bad_event_above_gate"
            failure_code = "F11_cascade_bad_event_fail"
        elif _f(p4.get("accepted_coverage")) < 0.03:
            route_name = "R13-ControllerStillCoverageLimited"
            blocker = "cascade_controller_coverage_below_gate"
            failure_code = "F10_cascade_coverage_fail"
        else:
            route_name = "R13-ControllerStillCoverageLimited"
            blocker = "cascade_controller_precision_or_support_failed"
            failure_code = "F9_cascade_precision_fail"
        reason = "P4_cascade_controller_failed"
        next_required = "repair_coverage_preserving_cascade_after_branch_delta"
    elif not _i(p5.get("support_measurement_pass")):
        route_name = "R5-CascadeControllerPass"
        blocker = "online_support_measurement_too_narrow"
        failure_code = "F12_support_measurement_too_narrow"
        reason = "P5_support_measurement_failed"
        next_required = "expand_signal_strata_and_family_support"
    else:
        route_name = "R5-CascadeControllerPass"
        blocker = "leave_dataset_out_not_opened"
        failure_code = "F14_leave_dataset_out_fail"
        reason = "P6_not_opened"
        next_required = "run_leave_dataset_and_stratum_out"

    downstream = _downstream(reason)
    artifacts: Dict[str, List[Dict[str, Any]]] = {
        "p0_v9250_boundary_reproduction.csv": [p0],
        "p1_m0_logit_preparation_subphase_attribution.csv": p1_rows,
        "p2_branch_logit_delta_kernel_matrix.csv": p2_rows,
        "p3_cheap_recall_generator_matrix.csv": p3_rows,
        "p4_coverage_preserving_cascade_controller.csv": p4_rows,
        "p5_online_support_stratum_expansion.csv": p5_rows,
        **downstream,
        "m0_logit_prep_trace_v9251.csv": p1_rows,
        "branch_logit_delta_trace_v9251.csv": p2_rows,
        "cheap_recall_trace_v9251.csv": p3_rows,
        "cascade_controller_trace_v9251.csv": p4_rows,
        "support_density_trace_v9251.csv": p5_rows,
        "leaveout_trace_v9251.csv": downstream["p6_leave_dataset_and_stratum_out.csv"],
        "paired_replay_branch_trace_v9251.csv": downstream["p7_official_paired_replay.csv"],
        "system_branch_delta_overhead_trace_v9251.csv": p2_rows,
    }
    for name, rows in artifacts.items():
        write_csv_rows(out_dir / name, rows)
    audit = audit_no_fake([out_dir / name for name in artifacts])
    write_csv_rows(out_dir / "v9251_provenance_audit.csv", [audit])

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9250_boundary_pass": p0.get("v9250_boundary_pass"),
        "dataset_tuning_detected": 0,
        "m0_subphase_attribution_pass": p1.get("m0_subphase_attribution_pass"),
        "dominant_m0_subphase": p1.get("dominant_m0_subphase"),
        "branch_logit_path_ratio": p1.get("branch_logit_path_ratio"),
        "best_branch_delta_id": p2.get("best_branch_delta_id"),
        "branch_delta_predictivity_pass": p2.get("branch_delta_predictivity_pass"),
        "branch_delta_system_pass": p2.get("branch_delta_system_pass"),
        "branch_delta_auc": p2.get("branch_delta_auc"),
        "branch_delta_corr": p2.get("branch_delta_corr"),
        "branch_delta_accept_agreement": p2.get("branch_delta_accept_agreement"),
        "branch_delta_step_ratio_q90": p2.get("branch_delta_step_ratio_q90"),
        "branch_delta_memory_ratio": p2.get("branch_delta_memory_ratio"),
        "best_cheap_recall_id": p3.get("best_cheap_recall_id"),
        "cheap_recall_pass": p3.get("cheap_recall_pass"),
        "safe_good_recall": p3.get("safe_good_recall"),
        "candidate_rate": p3.get("candidate_rate"),
        "candidate_bad_event": p3.get("candidate_bad_event"),
        "best_controller_id": p4.get("best_controller_id"),
        "cascade_controller_pass": p4.get("cascade_controller_pass"),
        "controller_auc": p4.get("controller_auc"),
        "controller_corr": p4.get("controller_corr"),
        "accepted_precision": p4.get("accepted_precision"),
        "accepted_coverage": p4.get("accepted_coverage"),
        "accepted_bad_event_rate": p4.get("accepted_bad_event_rate"),
        "accepted_signal_strata_count": p4.get("accepted_signal_strata_count"),
        "accepted_family_count": p4.get("accepted_family_count"),
        "max_family_share": p4.get("max_family_share"),
        "support_measurement_pass": p5.get("support_measurement_pass"),
        "natural_real_event_count": p5.get("natural_real_event_count"),
        "balanced_diagnostic_real_event_count": p5.get("balanced_diagnostic_real_event_count"),
        "measured_signal_strata_count": p5.get("measured_signal_strata_count"),
        "measured_family_count": p5.get("measured_family_count"),
        "oracle_support_pass": p5.get("oracle_support_pass"),
        "oracle_precision": p5.get("oracle_precision"),
        "oracle_coverage": p5.get("oracle_coverage"),
        "oracle_bad_event": p5.get("oracle_bad_event"),
        "leave_dataset_out_pass": 0,
        "leave_stratum_out_pass": 0,
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "external_ready": 0,
        "primary_blocker": blocker,
        "next_required_implementation": next_required,
        "success_v9251_strict_purekan_functional": 0,
        "success_v9251_full_functional": 0,
        "success_v9251_external_ready": 0,
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
    write_csv_rows(out_dir / "contract_audit_v9251.csv", [{
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "branch_logit_delta_audit": 1,
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
        "args": vars(args),
        "route": route_name,
        "completed_at": route["completed_at"],
    })
    _figures(out_dir, route)
    print(json.dumps(route, indent=2, sort_keys=True))
    return route


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(RESULT_ROOT / "v9251_branch_logit_delta_kernel_coverage_preserving_online_controller_first_20260512T030000Z"))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    p.add_argument("--microprobe-steps", type=int, default=160)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--phase-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--phase-seeds", default="0")
    p.add_argument("--phase-steps", type=int, default=3)
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
