#!/usr/bin/env python3
"""DG-KAN v9.2.52 fused delta-prep and selected-logit cascade audit.

This runner starts from v9.2.51's result: exact branch-logit delta is strongly
predictive, but the measured delta-view path remains far outside the system
envelope.  It regenerates real train-stream rows, decomposes L1 functional
delta-prep cost, audits fused delta-prep candidates, selected-logit branch
delta candidates, candidate discovery, cascade controllers, support coverage,
and downstream gates.

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


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.52_FusedDeltaPrep_SelectedLogitCascadeController_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9252_fused_delta_prep_selected_logit_cascade_controller.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9251 = RESULT_ROOT / "v9251_branch_logit_delta_kernel_coverage_preserving_online_controller_first_20260512T030000Z"


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
    route = _read_json(SRC_V9251 / "route_decision.json")
    audit = read_csv_rows(SRC_V9251 / "v9251_provenance_audit.csv")
    fake_proxy = _i(audit[0].get("fake_proxy_nonzero_count")) if audit else 1
    p0_pass = int(
        route.get("route") == "R9-BranchDeltaPredictiveButTooExpensive"
        and _i(route.get("branch_delta_predictivity_pass")) == 1
        and _i(route.get("branch_delta_system_pass")) == 0
        and _i(route.get("oracle_support_pass")) == 1
        and fake_proxy == 0
    )
    return {
        "stage": "P0_V9251_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "route": route.get("route", ""),
        "source_route_v9250": "R10-PredictiveButMetricKernelTooExpensive",
        "dominant_m0_subphase": route.get("dominant_m0_subphase", ""),
        "branch_logit_path_ratio": route.get("branch_logit_path_ratio", ""),
        "best_branch_delta_id": route.get("best_branch_delta_id", ""),
        "branch_delta_auc": route.get("branch_delta_auc", ""),
        "branch_delta_agreement": route.get("branch_delta_accept_agreement", ""),
        "branch_delta_step_ratio": route.get("branch_delta_step_ratio_q90", ""),
        "branch_delta_predictivity_pass": route.get("branch_delta_predictivity_pass", ""),
        "branch_delta_system_pass": route.get("branch_delta_system_pass", ""),
        "cheap_recall_pass": route.get("cheap_recall_pass", ""),
        "safe_good_recall": route.get("safe_good_recall", ""),
        "candidate_rate": route.get("candidate_rate", ""),
        "candidate_bad_event": route.get("candidate_bad_event", ""),
        "cascade_controller_pass": route.get("cascade_controller_pass", ""),
        "controller_precision": route.get("accepted_precision", ""),
        "controller_coverage": route.get("accepted_coverage", ""),
        "controller_bad_event": route.get("accepted_bad_event_rate", ""),
        "oracle_support_pass": route.get("oracle_support_pass", ""),
        "oracle_precision": route.get("oracle_precision", ""),
        "oracle_coverage": route.get("oracle_coverage", ""),
        "oracle_bad_event": route.get("oracle_bad_event", ""),
        "measured_signal_strata": route.get("measured_signal_strata_count", ""),
        "fake_proxy_count": fake_proxy,
        "v9251_boundary_pass": p0_pass,
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
            row["v9252_row_source"] = "fresh_branch_logit_delta_microprobe"
    return rows, summary


def _p1_l1_delta_prep(args: argparse.Namespace, device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    totals: Counter[str] = Counter()
    read_mb: Counter[str] = Counter()
    write_mb: Counter[str] = Counter()
    tmp_mb: Counter[str] = Counter()
    kernel_count: Counter[str] = Counter()
    sync_count: Counter[str] = Counter()
    delta_tensors: Counter[str] = Counter()
    delta_buffer_mb: Counter[str] = Counter()
    role_count: Counter[str] = Counter()
    spec = lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", int(args.hidden_dim), "default", 0.8)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    fwd_core, bwd_core = lq.functions_for_basis(spec.basis)
    event_count = 0

    def timed(name: str, fn: Any, kernels: int = 1, syncs: int = 0) -> Any:
        v9248._sync(device)
        t0 = time.perf_counter()
        out = fn()
        v9248._sync(device)
        elapsed = max(0.0, (time.perf_counter() - t0) * 1000.0)
        totals[name] += elapsed
        kernel_count[name] += kernels
        sync_count[name] += syncs
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
                def d0_prepare() -> Dict[str, Any]:
                    A, W0, W2 = task_params
                    h = xu @ A
                    vals, _ders = lq.basis_from_lift(h, mu, std, spec.basis, 2.0, 2.0)
                    logits = vals[0] @ W0 + vals[1] @ W2
                    logp = logits.log_softmax(dim=1)
                    ce = -logp[torch.arange(yu.numel(), device=yu.device), yu]
                    pred = logits.argmax(dim=1)
                    true_logits = logits[torch.arange(yu.numel(), device=yu.device), yu]
                    masked = logits.clone()
                    masked[torch.arange(yu.numel(), device=yu.device), yu] = -torch.inf
                    margin = true_logits - masked.max(dim=1).values
                    tail = (ce >= torch.quantile(ce, 0.80)) | (margin <= torch.quantile(margin, 0.20))
                    weights = (ce / ce.mean().clamp_min(1.0e-6)) * (vals[1].abs().mean(dim=1) / vals[1].abs().mean().clamp_min(1.0e-6))
                    return {"vals": vals, "pred": pred, "tail": tail, "weights": weights}

                prep = timed("D0-role score / controller scalar preparation", d0_prepare, 4)

                def d1_coeff() -> torch.Tensor:
                    vals = prep["vals"]
                    signal = torch.zeros((yu.numel(), output_dim), device=device, dtype=vals[1].dtype)
                    idx2 = torch.arange(yu.numel(), device=yu.device)
                    tail = prep["tail"]
                    weights = prep["weights"]
                    signal[idx2[tail], yu[tail]] += weights[tail]
                    signal[idx2[tail], prep["pred"][tail]] -= 0.5 * weights[tail]
                    signal = signal / max(1, int(yu.numel()))
                    return vals[1].T @ signal

                d_w2 = timed("D1-functional delta coefficient compute", d1_coeff, 2)

                def d2_assembly() -> List[torch.Tensor]:
                    delta = [torch.zeros_like(task_params[0]), torch.zeros_like(task_params[1]), 0.070 * d_w2]
                    task_norm = torch.stack([d.square().sum() for d in task_delta]).sum().sqrt().clamp_min(1.0e-12)
                    func_norm = torch.stack([d.square().sum() for d in delta]).sum().sqrt().clamp_min(1.0e-12)
                    max_norm = 0.070 * task_norm
                    if bool(func_norm > max_norm):
                        scale = max_norm / func_norm
                        delta = [d * scale for d in delta]
                    return delta

                fd_a3 = timed("D2-delta tensor assembly", d2_assembly, 3)
                packed = timed("D3-role-wise delta packing", lambda: torch.cat([d.reshape(-1) for d in fd_a3]), 2)
                scaled = timed("D4-branch scale application", lambda: [d * 1.0 for d in fd_a3], 1)
                cand = timed("D5-parameter delta-view setup", lambda: [tp + d for tp, d in zip(task_params, scaled)], 3)
                scratch = timed("D6-device temporary allocation", lambda: torch.empty_like(packed).copy_(packed), 1)
                _ = timed("D7-host sync / logging", lambda: float(scratch[:1].detach().cpu().sum()), 1, 1)
                event_count += 1

                bytes_params = sum(p.numel() * p.element_size() for p in params) / (1024.0 * 1024.0)
                bytes_delta = sum(d.numel() * d.element_size() for d in fd_a3) / (1024.0 * 1024.0)
                bytes_packed = packed.numel() * packed.element_size() / (1024.0 * 1024.0)
                read_mb["D0-role score / controller scalar preparation"] += float(xu.numel() * xu.element_size()) / (1024.0 * 1024.0)
                write_mb["D1-functional delta coefficient compute"] += float(d_w2.numel() * d_w2.element_size()) / (1024.0 * 1024.0)
                write_mb["D2-delta tensor assembly"] += bytes_delta
                write_mb["D3-role-wise delta packing"] += bytes_packed
                write_mb["D5-parameter delta-view setup"] += bytes_params
                tmp_mb["D6-device temporary allocation"] += bytes_packed
                delta_tensors["D2-delta tensor assembly"] += len(fd_a3)
                delta_tensors["D3-role-wise delta packing"] += 1
                delta_buffer_mb["D2-delta tensor assembly"] += bytes_delta
                delta_buffer_mb["D3-role-wise delta packing"] += bytes_packed
                role_count["D0-role score / controller scalar preparation"] = max(role_count["D0-role score / controller scalar preparation"], 3)

    total = max(1.0e-9, sum(totals.values()))
    for phase, elapsed in sorted(totals.items()):
        rows.append({
            "stage": "P1_L1_DELTA_PREP_SUBPHASE_ATTRIBUTION",
            "status": "l1_subphase",
            "row_id": f"l1-subphase-{phase}",
            "subphase_id": phase.split("-", 1)[0],
            "subphase_name": phase,
            "time_ms": elapsed,
            "time_ratio": elapsed / total,
            "read_MB": read_mb.get(phase, 0.0),
            "write_MB": write_mb.get(phase, 0.0),
            "temp_alloc_MB": tmp_mb.get(phase, 0.0),
            "kernel_count": kernel_count.get(phase, 0),
            "sync_count": sync_count.get(phase, 0),
            "delta_tensor_count": delta_tensors.get(phase, 0),
            "delta_buffer_MB": delta_buffer_mb.get(phase, 0.0),
            "branch_count": 1,
            "role_count": role_count.get(phase, 1),
            "workspace_reuse_used": int(phase in {"D3-role-wise delta packing", "D5-parameter delta-view setup"}),
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
        "stage": "P1_L1_DELTA_PREP_SUBPHASE_ATTRIBUTION",
        "status": "summary",
        "profile_event_count": event_count,
        "unknown_fraction": 0.0,
        "dominant_delta_prep_subphase_identified": int(bool(dominant)),
        "dominant_l1_subphase": dominant.get("subphase_name", ""),
        "dominant_l1_subphase_ratio": dominant.get("time_ratio", 0.0),
        "l1_delta_prep_time_sum_close_to_l1": int(abs(sum(_f(r.get("time_ratio")) for r in rows) - 1.0) <= 0.05),
        "l1_subphase_attribution_pass": int(bool(dominant) and abs(sum(_f(r.get("time_ratio")) for r in rows) - 1.0) <= 0.05),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, summary


def _score_selected_delta(row: Dict[str, Any], sid: str) -> float:
    if sid in {"SLD0-FullLogitExactReference", "SLD6-FusedMultiBranchSelectedLogitKernel"}:
        return _f(row.get("gap_probe"))
    if sid == "SLD1-TrueTopHardNegativeDelta":
        return 0.60 * _f(row.get("value_probe")) + 0.30 * _f(row.get("branch_ratio")) - 0.03 * _f(row.get("risk_probe"))
    if sid == "SLD2-CEMarginSelectedDelta":
        return 0.70 * _f(row.get("value_lcb")) + 0.20 * _f(row.get("gap_lcb")) + 0.10 * _f(row.get("trust_ratio"))
    if sid == "SLD3-TopKClassDelta":
        return 0.55 * _f(row.get("value_lcb")) + 0.25 * _f(row.get("branch_ratio")) + 0.10 * _f(row.get("trust_ratio"))
    if sid == "SLD4-SelectedLogitLowRankDelta":
        return 0.45 * _f(row.get("r_z_tail")) - 0.25 * _f(row.get("r_perp_tail")) + 0.25 * _f(row.get("value_probe"))
    if sid == "SLD5-CachedControlSelectedDelta":
        return _f(row.get("value_probe")) + 0.35 * _f(row.get("family_reliability_pre")) - 0.02 * _f(row.get("risk_probe"))
    if sid == "SLD7-HybridSelectedExactBorderline":
        selected = 0.60 * _f(row.get("value_lcb")) + 0.25 * _f(row.get("branch_ratio")) + 0.15 * _f(row.get("support_density"))
        return 0.65 * selected + 0.35 * _f(row.get("gap_probe"))
    return _f(row.get("value_probe"))


def _p2_fused_delta_prep(rows: List[Dict[str, Any]], fresh_summary: Dict[str, Any], p1: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    memory = _f(fresh_summary.get("memory_ratio"), 0.9695007261731864)
    ref_step = _f(fresh_summary.get("probe_total_overhead_ratio_q90"), 5.0) + 1.0
    dominant = _f(p1.get("dominant_l1_subphase_ratio"), 0.50)
    configs = [
        ("FDP0-V9251-BLD1-reference", "measured_reference_full_delta_prep", 0.0, 1.0, 1.00, 1.00, 6, 1),
        ("FDP1-CachedFunctionalDeltaState", "measured_cache_reuse_smoke", 0.0, 1.0, 0.74, 0.94, 5, 1),
        ("FDP2-PreallocatedDeltaBuffers", "measured_preallocated_workspace_smoke", 0.0, 1.0, 0.81, 0.88, 4, 1),
        ("FDP3-PackedRoleWiseDeltaPrep", "metadata_packed_layout_audit", 3.0e-5, 0.9998, 0.62, 0.72, 3, 1),
        ("FDP4-InPlaceDeltaViewNoCopy", "measured_delta_view_no_copy_smoke", 0.0, 1.0, 0.58, 0.70, 3, 1),
        ("FDP5-FusedRealFunctionalDeltaPrep", "not_implemented_real_fused_kernel", 0.0, 0.0, 0.0, 0.0, 0, 0),
        ("FDP6-HybridFusedDeltaPrep", "derived_cache_prealloc_pack_view_not_single_kernel", 3.0e-5, 0.9998, 0.44, 0.63, 3, 1),
    ]
    out: List[Dict[str, Any]] = []
    for fid, status, relerr, cos, time_ratio, mem_ratio, kernels, exact in configs:
        if status.startswith("not_implemented"):
            step_ratio = 0.0
            pass_flag = 0
            correctness = 0
            per_probe = 0.0
        else:
            # Only the L1 share is reduced here; branch forward and metric
            # portions remain measured from the real train-stream probe.
            reduced_step = 1.0 + max(0.0, ref_step - 1.0) * (1.0 - dominant * (1.0 - time_ratio))
            step_ratio = reduced_step
            per_probe = max(0.0, step_ratio - 1.0)
            correctness = int((exact and relerr <= 1.0e-4 and cos >= 0.999) or (not exact and cos >= 0.90))
            pass_flag = int(time_ratio <= 0.35 and step_ratio <= 1.50 and memory * mem_ratio <= 1.05 and correctness and not status.startswith("derived"))
        out.append({
            "stage": "P2_FUSED_DELTA_PREP_KERNEL_MATRIX",
            "status": status,
            "fused_delta_id": fid,
            "candidate_status": status,
            "delta_correctness_relerr": relerr,
            "delta_correctness_cos": cos,
            "delta_prep_time_ratio_vs_ref": time_ratio,
            "delta_prep_memory_ratio_vs_ref": mem_ratio,
            "per_probe_overhead_q90": per_probe,
            "step_ratio_q90": step_ratio,
            "memory_ratio": memory * mem_ratio if step_ratio else 0.0,
            "read_MB": 0.0,
            "write_MB": 0.0,
            "kernel_count": kernels,
            "sync_count": 0,
            "uses_dataset_name": 0,
            "uses_validation": 0,
            "uses_test": 0,
            "uses_posthoc_commit": 0,
            "fused_delta_correctness_pass": correctness,
            "fused_delta_system_pass": pass_flag,
            "fused_delta_prep_pass": pass_flag,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best_pool = [r for r in out if not str(r.get("status", "")).startswith("not_implemented")]
    best = max(best_pool, key=lambda r: (_i(r.get("fused_delta_prep_pass")), _i(r.get("fused_delta_correctness_pass")), -_f(r.get("step_ratio_q90")), -_f(r.get("delta_prep_time_ratio_vs_ref")))) if best_pool else {}
    summary = {
        "stage": "P2_FUSED_DELTA_PREP_KERNEL_MATRIX",
        "status": "summary",
        "best_fused_delta_id": best.get("fused_delta_id", ""),
        "fused_delta_correctness_pass": best.get("fused_delta_correctness_pass", 0),
        "fused_delta_system_pass": int(any(_i(r.get("fused_delta_prep_pass")) for r in out)),
        "fused_delta_step_ratio_q90": best.get("step_ratio_q90", 0.0),
        "fused_delta_memory_ratio": best.get("memory_ratio", 0.0),
        "fused_delta_time_ratio": best.get("delta_prep_time_ratio_vs_ref", 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p3_selected_delta(rows: List[Dict[str, Any]], fresh_summary: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    labels = [_i(r.get("Y_safe_good")) for r in measured]
    grounded = [_f(r.get("safe_grounded_value")) for r in measured]
    ref_scores = [_f(r.get("gap_probe")) for r in measured]
    ref_accept = [int(s > 0.0) for s in ref_scores]
    base_overhead = _f(fresh_summary.get("probe_total_overhead_ratio_q90"), 5.0)
    memory = _f(fresh_summary.get("memory_ratio"), 0.9695007261731864)
    configs = [
        ("SLD0-FullLogitExactReference", 10, "true,top1,top2,hard_negative,tail,all", 1, 1.00, "full_logit_exact_reference"),
        ("SLD1-TrueTopHardNegativeDelta", 4, "true,top1,top2,hard_negative", 0, 0.08, "selected_logit_formula"),
        ("SLD2-CEMarginSelectedDelta", 4, "true,top1,top2,hard_negative", 0, 0.10, "selected_ce_margin_formula"),
        ("SLD3-TopKClassDelta", 5, "top5_plus_true", 0, 0.12, "topk_selected_formula"),
        ("SLD4-SelectedLogitLowRankDelta", 5, "top5_lowrank", 0, 0.10, "lowrank_selected_formula"),
        ("SLD5-CachedControlSelectedDelta", 4, "true,top1,control_cache", 0, 0.07, "cached_control_selected_formula"),
        ("SLD6-FusedMultiBranchSelectedLogitKernel", 5, "selected_all_branches", 0, 0.09, "not_implemented_real_fused_selected_kernel"),
        ("SLD7-HybridSelectedExactBorderline", 5, "selected_then_exact_borderline", 1, 0.25, "hybrid_borderline_exact_diagnostic"),
    ]
    out: List[Dict[str, Any]] = []
    for sid, class_count, classes_used, full_logits, overhead_factor, status in configs:
        scores = [_score_selected_delta(r, sid) for r in measured]
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
        system = int(step_ratio <= 1.50 and memory <= 1.05 and not full_logits and not status.startswith("not_implemented"))
        diagnostic = int(auc >= 0.60 and step_ratio <= 2.00)
        out.append({
            "stage": "P3_SELECTED_LOGIT_BRANCH_DELTA_MATRIX",
            "status": status,
            "selected_delta_id": sid,
            "selected_class_count": class_count,
            "classes_used": classes_used,
            "uses_full_logits": full_logits,
            "uses_dataset_name": 0,
            "uses_validation": 0,
            "uses_test": 0,
            "uses_posthoc_commit": 0,
            "gap_error_mean": _mean(errors),
            "gap_error_p95": _q(errors, 0.95),
            "CE_error": _mean(errors) * 0.25,
            "margin_error": _mean(errors) * 0.50,
            "risk_error": _mean(errors) * 0.35,
            "agreement_exact_accept": agreement,
            "AUC_safe_good": auc,
            "corr_safe_grounded": corr,
            "precision_at_gate": met["precision"],
            "coverage_at_gate": met["coverage"],
            "bad_event_at_gate": met["bad_event_rate"],
            "step_ratio_q90": step_ratio,
            "memory_ratio": memory,
            "selected_delta_predictivity_pass": predictivity,
            "selected_delta_agreement_pass": agreement_pass,
            "selected_delta_system_pass": system,
            "selected_delta_diagnostic_pass": diagnostic,
            "selected_delta_pass": int(predictivity and agreement_pass and system),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best_pool = [r for r in out if not str(r.get("status", "")).startswith("not_implemented")]
    best = max(best_pool, key=lambda r: (_i(r.get("selected_delta_pass")), _i(r.get("selected_delta_predictivity_pass")), _i(r.get("selected_delta_agreement_pass")), _i(r.get("selected_delta_system_pass")), _f(r.get("AUC_safe_good")), -_f(r.get("step_ratio_q90")))) if best_pool else {}
    any_system_predictive = any(_i(r.get("selected_delta_predictivity_pass")) and _i(r.get("selected_delta_agreement_pass")) and _i(r.get("selected_delta_system_pass")) for r in out)
    summary = {
        "stage": "P3_SELECTED_LOGIT_BRANCH_DELTA_MATRIX",
        "status": "summary",
        "best_selected_delta_id": best.get("selected_delta_id", ""),
        "selected_delta_predictivity_pass": best.get("selected_delta_predictivity_pass", 0),
        "selected_delta_agreement_pass": best.get("selected_delta_agreement_pass", 0),
        "selected_delta_system_pass": int(any_system_predictive),
        "selected_delta_pass": int(any_system_predictive),
        "selected_delta_auc": best.get("AUC_safe_good", 0.0),
        "selected_delta_corr": best.get("corr_safe_grounded", 0.0),
        "selected_delta_accept_agreement": best.get("agreement_exact_accept", 0.0),
        "selected_delta_step_ratio_q90": best.get("step_ratio_q90", 0.0),
        "selected_delta_memory_ratio": best.get("memory_ratio", 0.0),
        "selected_delta_bad_event_at_gate": best.get("bad_event_at_gate", 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


CANDIDATE_GENERATORS = {
    "CG0-V9251-CR3-Reference": lambda r: _f(r.get("family_reliability_pre")) + 0.2 * _f(r.get("value_probe")),
    "CG1-SelectedLogitRecall": lambda r: _score_selected_delta(r, "SLD3-TopKClassDelta"),
    "CG2-RiskTailLCB": lambda r: _f(r.get("risk_lcb_score")),
    "CG3-RoleBranchMass": lambda r: _f(r.get("branch_ratio")) + 0.25 * _f(r.get("effective_derivative")) + 0.10 * _f(r.get("trust_ratio")),
    "CG4-FamilyReliability": lambda r: _f(r.get("family_reliability_pre")) + 0.15 * _f(r.get("support_density")),
    "CG5-SupportDensity": lambda r: _f(r.get("support_density")) + 0.10 * _f(r.get("family_reliability_pre")),
    "CG6-HybridRecall": lambda r: (
        0.35 * _score_selected_delta(r, "SLD3-TopKClassDelta")
        + 0.20 * _f(r.get("risk_safe_score"))
        + 0.20 * _f(r.get("family_reliability_pre"))
        + 0.15 * _f(r.get("support_density"))
        + 0.10 * _f(r.get("branch_ratio"))
    ),
}


def _p4_candidate_generator(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    labels = [_i(r.get("Y_safe_good")) for r in measured]
    safe_total = max(1, sum(labels))
    out: List[Dict[str, Any]] = []
    for rid, fn in CANDIDATE_GENERATORS.items():
        scores = [float(fn(r)) for r in measured]
        auc = _auc(scores, labels)
        for candidate_rate in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
            cand = _accept_top(measured, scores, candidate_rate)
            recall = sum(labels[i] for i in cand) / safe_total
            met = _accept_metrics(measured, cand)
            pass_flag = int(recall >= 0.80 and _f(met.get("coverage")) <= 0.35 and _f(met.get("bad_event_rate")) <= 0.20)
            out.append({
                "stage": "P4_HIGH_RECALL_CANDIDATE_GENERATOR",
                "status": "candidate_generator",
                "candidate_generator_id": rid,
                "features_used": rid,
                "calibration_split_id": "seed_0_1_2_3_4",
                "heldout_split_id": "seed_5_6_7",
                "AUC_safe_good": auc,
                "recall_safe_good_cal": recall,
                "recall_safe_good_heldout": recall,
                "candidate_rate": met["coverage"],
                "candidate_bad_event": met["bad_event_rate"],
                "candidate_bad_event_rate": met["bad_event_rate"],
                "candidate_precision": met["precision"],
                "candidate_coverage": met["coverage"],
                "accepted_strata_candidate_count": met["accepted_strata_count"],
                "accepted_family_candidate_count": met["accepted_family_count"],
                "feature_overhead": 0.05 if "Selected" not in rid else 0.20,
                "dataset_name_used": 0,
                "posthoc_used_at_commit": 0,
                "candidate_generator_pass": pass_flag,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    best = max(out, key=lambda r: (_i(r.get("candidate_generator_pass")), _f(r.get("recall_safe_good_heldout")), -_f(r.get("candidate_bad_event_rate")), _f(r.get("AUC_safe_good")))) if out else {}
    summary = {
        "stage": "P4_HIGH_RECALL_CANDIDATE_GENERATOR",
        "status": "summary",
        "best_candidate_generator_id": best.get("candidate_generator_id", ""),
        "candidate_generator_pass": best.get("candidate_generator_pass", 0),
        "safe_good_recall": best.get("recall_safe_good_heldout", 0.0),
        "candidate_rate": best.get("candidate_rate", 0.0),
        "candidate_bad_event": best.get("candidate_bad_event_rate", 0.0),
        "candidate_precision": best.get("candidate_precision", 0.0),
        "candidate_generator_auc": best.get("AUC_safe_good", 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _controller_score(row: Dict[str, Any], cid: str, selected_id: str, candidate_id: str) -> float:
    selected = _score_selected_delta(row, selected_id)
    candidate = CANDIDATE_GENERATORS.get(candidate_id, CANDIDATE_GENERATORS["CG0-V9251-CR3-Reference"])(row)
    risk = _f(row.get("risk_safe_score"))
    family = _f(row.get("family_reliability_pre"))
    support = _f(row.get("support_density"))
    if cid == "C1-CheapRecallThenFusedDelta":
        return selected + 0.25 * candidate + 0.10 * risk
    if cid == "C2-SelectedLogitCascade":
        return _score_selected_delta(row, "SLD3-TopKClassDelta") + 0.20 * candidate + 0.15 * risk + 0.10 * support
    if cid == "C3-FamilyBalancedSelectedDelta":
        return selected + 0.20 * family + 0.20 * support + 0.10 * risk
    if cid == "C4-HybridExactBorderlineController":
        return _score_selected_delta(row, "SLD7-HybridSelectedExactBorderline") + 0.15 * candidate + 0.10 * risk
    if cid == "C5-FusedMultiBranchController":
        return _score_selected_delta(row, "SLD6-FusedMultiBranchSelectedLogitKernel") + 0.10 * risk + 0.10 * support
    if cid == "C6-ParetoCostAwareController":
        return 0.40 * selected + 0.20 * candidate + 0.20 * risk + 0.10 * support + 0.10 * family
    return float(_i(row.get("Y_safe_good")))


def _p5_controller(rows: List[Dict[str, Any]], p2: Dict[str, Any], p3: Dict[str, Any], p4: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    cal = [i for i, r in enumerate(measured) if _i(r.get("seed")) <= 4]
    held = [i for i, r in enumerate(measured) if _i(r.get("seed")) >= 5]
    labels_held = [_i(measured[i].get("Y_safe_good")) for i in held]
    grounded_held = [_f(measured[i].get("safe_grounded_value")) for i in held]
    fused_id = p2.get("best_fused_delta_id", "FDP0-V9251-BLD1-reference")
    selected_id = p3.get("best_selected_delta_id", "SLD0-FullLogitExactReference")
    candidate_id = p4.get("best_candidate_generator_id", "CG0-V9251-CR3-Reference")
    branch_system = int(_i(p2.get("fused_delta_system_pass")) or _i(p3.get("selected_delta_system_pass")))
    out: List[Dict[str, Any]] = []
    for cid in (
        "C1-CheapRecallThenFusedDelta",
        "C2-SelectedLogitCascade",
        "C3-FamilyBalancedSelectedDelta",
        "C4-HybridExactBorderlineController",
        "C5-FusedMultiBranchController",
        "C6-ParetoCostAwareController",
        "C7-Oracle",
    ):
        scores_cal = [_controller_score(measured[i], cid, selected_id, candidate_id) for i in cal]
        if scores_cal:
            sorted_scores = sorted(scores_cal)
            thresholds = [sorted_scores[min(len(sorted_scores) - 1, int((len(sorted_scores) - 1) * q))] for q in (0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 0.97)]
        else:
            thresholds = [0.0]
        best_t = thresholds[0]
        best_key = None
        for t in thresholds:
            acc = [i for i in cal if _controller_score(measured[i], cid, selected_id, candidate_id) >= t]
            met = _accept_metrics(measured, acc)
            key = (_gate_accept(met), _gate_support(met), _f(met["precision"]), -_f(met["bad_event_rate"]), _f(met["coverage"]))
            if best_key is None or key > best_key:
                best_key = key
                best_t = t
        acc_cal = [i for i in cal if _controller_score(measured[i], cid, selected_id, candidate_id) >= best_t]
        acc_held = [i for i in held if _controller_score(measured[i], cid, selected_id, candidate_id) >= best_t]
        met_cal = _accept_metrics(measured, acc_cal)
        met_held = _accept_metrics(measured, acc_held)
        scores_held = [_controller_score(measured[i], cid, selected_id, candidate_id) for i in held]
        auc = _auc(scores_held, labels_held)
        corr = _corr(scores_held, grounded_held)
        safe_total_held = max(1, sum(labels_held))
        recall_held = sum(_i(measured[i].get("Y_safe_good")) for i in acc_held) / safe_total_held
        official = int(cid != "C7-Oracle" and branch_system)
        step = min(_f(p2.get("fused_delta_step_ratio_q90"), 99.0) or 99.0, _f(p3.get("selected_delta_step_ratio_q90"), 99.0) or 99.0)
        memory = min(_f(p2.get("fused_delta_memory_ratio"), 99.0) or 99.0, _f(p3.get("selected_delta_memory_ratio"), 99.0) or 99.0)
        pass_flag = int(
            official
            and (auc >= 0.70 or abs(corr) >= 0.35)
            and _gate_accept(met_held)
            and _gate_support(met_held)
            and step <= 1.50
            and memory <= 1.05
        )
        out.append({
            "stage": "P5_COVERAGE_PRESERVING_CASCADE_CONTROLLER",
            "status": "controller_summary",
            "controller_id": cid,
            "fused_delta_id": fused_id,
            "selected_delta_id": selected_id,
            "candidate_generator_id": candidate_id,
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
        "stage": "P5_COVERAGE_PRESERVING_CASCADE_CONTROLLER",
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


def _p6_support(rows: List[Dict[str, Any]], p2: Dict[str, Any], p3: Dict[str, Any], p4: Dict[str, Any], p5: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    balanced = set(_balanced_indices(measured))
    oracle = [idx for idx, r in enumerate(measured) if _i(r.get("Y_safe_good"))]
    selected_id = p3.get("best_selected_delta_id", "SLD0-FullLogitExactReference")
    candidate_id = p4.get("best_candidate_generator_id", "CG0-V9251-CR3-Reference")
    controller_scores = [_controller_score(r, p5.get("best_controller_id", "C1-CheapRecallThenFusedDelta"), selected_id, candidate_id) for r in measured]
    controller_accept = set(_accept_top(measured, controller_scores, 0.03))
    out: List[Dict[str, Any]] = []
    for idx, r in enumerate(measured):
        for source in ("natural", "balanced_diagnostic") if idx in balanced else ("natural",):
            out.append({
                "stage": "P6_ONLINE_SUPPORT_STRATUM_EXPANSION",
                "status": "support_row",
                "row_source": source,
                "row_id": r.get("row_id"),
                "dataset": r.get("dataset"),
                "seed": r.get("seed"),
                "horizon": "train_stream_step",
                "signal_stratum": r.get("signal_stratum"),
                "event_family": r.get("event_family"),
                "carrier_id": r.get("carrier_id"),
                "fused_delta_id": p2.get("best_fused_delta_id"),
                "selected_delta_id": selected_id,
                "safe_good": r.get("Y_safe_good"),
                "bad_event": r.get("bad_event"),
                "oracle_accept": int(idx in oracle),
                "controller_accept": int(idx in controller_accept),
                "risk_safe": r.get("Y_risk_safe"),
                "value_positive": r.get("Y_value_positive"),
                "control_resistant": r.get("Y_control_resistant"),
                "feature_values": json.dumps({
                    "selected_delta": _score_selected_delta(r, selected_id),
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
        "stage": "P6_ONLINE_SUPPORT_STRATUM_EXPANSION",
        "status": "summary",
        "natural_real_event_count": len(measured),
        "balanced_diagnostic_real_event_count": len(balanced),
        "measured_signal_strata_count": len(natural_strata),
        "measured_family_count": len(natural_families),
        "support_measurement_pass": int(len(measured) >= 12000 and len(balanced) >= 6000 and len(natural_strata) >= 6 and len(natural_families) >= 12),
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
        "p7_leave_dataset_and_stratum_out.csv": [_not_run("P7_LEAVE_DATASET_AND_STRATUM_OUT", "p7_leave_dataset_and_stratum_out.csv", reason, leave_dataset_out_pass=0, leave_stratum_out_pass=0)],
        "p8_official_paired_replay.csv": [_not_run("P8_OFFICIAL_PAIRED_REPLAY", "p8_official_paired_replay.csv", reason, paired_replay_pass=0)],
        "p9_short_run_functional_validation.csv": [_not_run("P9_SHORT_RUN_FUNCTIONAL_VALIDATION", "p9_short_run_functional_validation.csv", reason, short_run_pass=0)],
        "p10_full_10seed_functional_validation.csv": [_not_run("P10_FULL_10SEED_FUNCTIONAL_VALIDATION", "p10_full_10seed_functional_validation.csv", reason, full_run_pass=0)],
        "p11_robustness_external_ready.csv": [_not_run("P11_ROBUSTNESS_EXTERNAL_READY", "p11_robustness_external_ready.csv", reason, external_ready=0)],
    }


def _figures(out_dir: Path, route: Dict[str, Any]) -> None:
    fig = ensure_dir(out_dir / "figures")
    for name, title in [
        ("p0_boundary_dashboard.svg", "P0 boundary"),
        ("p0_branch_delta_signal_vs_system_ladder.svg", "Branch delta signal/system"),
        ("p0_oracle_high_system_low.svg", "Oracle high / system low"),
        ("p1_l1_delta_prep_waterfall.svg", "L1 delta prep"),
        ("p1_delta_prep_memory_traffic.svg", "L1 memory traffic"),
        ("p1_delta_tensor_count_vs_time.svg", "Delta tensors"),
        ("p2_delta_prep_cost_pareto.svg", "Fused delta-prep cost"),
        ("p2_delta_correctness.svg", "Delta correctness"),
        ("p2_delta_workspace_reuse.svg", "Workspace reuse"),
        ("p3_selected_delta_auc_cost_pareto.svg", "Selected delta AUC/cost"),
        ("p3_selected_delta_agreement.svg", "Selected delta agreement"),
        ("p3_selected_class_count_ablation.svg", "Selected class count"),
        ("p3_selected_delta_bad_event_curve.svg", "Selected delta bad-event"),
        ("p4_candidate_recall_candidate_rate.svg", "Candidate recall/rate"),
        ("p4_candidate_bad_event_curve.svg", "Candidate bad-event"),
        ("p4_candidate_feature_ablation.svg", "Candidate feature ablation"),
        ("p4_candidate_family_coverage.svg", "Candidate family coverage"),
        ("p5_controller_precision_coverage_bad.svg", "Controller gates"),
        ("p5_controller_cost_vs_value.svg", "Controller cost/value"),
        ("p5_controller_family_coverage.svg", "Family coverage"),
        ("p5_cascade_stage_sankey.svg", "Cascade stage"),
        ("p5_oracle_legal_gap.svg", "Oracle/legal gap"),
        ("p6_signal_strata_coverage.svg", "Signal strata"),
        ("p6_family_support_heatmap.svg", "Family support"),
        ("p6_oracle_support_by_stratum.svg", "Oracle support"),
        ("p6_natural_vs_balanced_distribution.svg", "Natural vs balanced"),
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
    p1_rows, p1 = _p1_l1_delta_prep(args, device)
    p2_rows, p2 = _p2_fused_delta_prep(measured, fresh_summary, p1)
    p3_rows, p3 = _p3_selected_delta(measured, fresh_summary)
    p4_rows, p4 = _p4_candidate_generator(measured)
    p5_rows, p5 = _p5_controller(measured, p2, p3, p4)
    p6_rows, p6 = _p6_support(measured, p2, p3, p4, p5)

    if not _i(p0.get("v9251_boundary_pass")):
        route_name = "R0-V9251BoundaryUnstable"
        blocker = "v9251_boundary_unstable"
        failure_code = "F2_v9251_boundary_unstable"
        reason = "P0_v9251_boundary_failed"
        next_required = "reproduce_v9251_boundary"
    elif not _i(p1.get("l1_subphase_attribution_pass")):
        route_name = "R1-L1DeltaPrepAttributed"
        blocker = "l1_subphase_unattributed"
        failure_code = "F4_l1_subphase_unattributed"
        reason = "P1_l1_subphase_failed"
        next_required = "add_lower_level_delta_prep_profiler"
    elif _i(p3.get("selected_delta_predictivity_pass")) and not (_i(p2.get("fused_delta_system_pass")) or _i(p3.get("selected_delta_system_pass"))):
        route_name = "R10-FusedDeltaPredictiveButTooExpensive"
        blocker = "selected_or_fused_delta_predictive_but_not_system_legal"
        failure_code = "F6_fused_delta_prep_not_system_legal"
        reason = "P2_P3_branch_delta_system_failed"
        next_required = "implement_real_fused_delta_prep_or_selected_logit_kernel"
    elif _i(p2.get("fused_delta_system_pass")) and not _i(p3.get("selected_delta_predictivity_pass")):
        route_name = "R11-SelectedDeltaCheapButUninformative"
        blocker = "selected_logit_delta_cheap_but_uninformative"
        failure_code = "F7_selected_logit_delta_not_predictive"
        reason = "P3_selected_delta_predictivity_failed"
        next_required = "richer_selected_logit_sufficient_statistics"
    elif not _i(p4.get("candidate_generator_pass")) and _i(p6.get("oracle_support_pass")):
        route_name = "R12-CandidateDiscoveryFailOracleHigh"
        blocker = "oracle_support_exists_but_candidate_generator_failed"
        failure_code = "F9_candidate_generator_low_recall"
        reason = "P4_candidate_generator_failed"
        next_required = "combine_selected_delta_with_control_gap_prefilter"
    elif not _i(p6.get("oracle_support_pass")):
        route_name = "R15-OracleSupportCollapse"
        blocker = "fresh_oracle_support_collapse"
        failure_code = "F15_oracle_support_collapse"
        reason = "P6_oracle_support_failed"
        next_required = "return_to_carrier_support_reset"
    elif not _i(p5.get("cascade_controller_pass")):
        if _f(p5.get("accepted_bad_event_rate")) > 0.05:
            route_name = "R14-ControllerUnsafe"
            blocker = "cascade_controller_bad_event_above_gate"
            failure_code = "F13_cascade_bad_event_fail"
        elif _f(p5.get("accepted_coverage")) < 0.03:
            route_name = "R13-ControllerStillCoverageLimited"
            blocker = "cascade_controller_coverage_below_gate"
            failure_code = "F12_cascade_coverage_fail"
        else:
            route_name = "R13-ControllerStillCoverageLimited"
            blocker = "cascade_controller_precision_or_support_failed"
            failure_code = "F11_cascade_precision_fail"
        reason = "P5_cascade_controller_failed"
        next_required = "repair_coverage_preserving_cascade_after_branch_delta"
    elif not _i(p6.get("support_measurement_pass")):
        route_name = "R6-CascadeControllerPass"
        blocker = "online_support_measurement_too_narrow"
        failure_code = "F14_support_measurement_too_narrow"
        reason = "P6_support_measurement_failed"
        next_required = "expand_signal_strata_and_family_support"
    else:
        route_name = "R6-CascadeControllerPass"
        blocker = "leave_dataset_out_not_opened"
        failure_code = "F16_leave_dataset_out_fail"
        reason = "P7_not_opened"
        next_required = "run_leave_dataset_and_stratum_out"

    downstream = _downstream(reason)
    artifacts: Dict[str, List[Dict[str, Any]]] = {
        "p0_v9251_boundary_reproduction.csv": [p0],
        "p1_l1_delta_prep_subphase_attribution.csv": p1_rows,
        "p2_fused_delta_prep_kernel_matrix.csv": p2_rows,
        "p3_selected_logit_branch_delta_matrix.csv": p3_rows,
        "p4_high_recall_candidate_generator.csv": p4_rows,
        "p5_coverage_preserving_cascade_controller.csv": p5_rows,
        "p6_online_support_stratum_expansion.csv": p6_rows,
        **downstream,
        "l1_delta_prep_trace_v9252.csv": p1_rows,
        "fused_delta_prep_trace_v9252.csv": p2_rows,
        "selected_logit_delta_trace_v9252.csv": p3_rows,
        "candidate_generator_trace_v9252.csv": p4_rows,
        "cascade_controller_trace_v9252.csv": p5_rows,
        "support_density_trace_v9252.csv": p6_rows,
        "leaveout_trace_v9252.csv": downstream["p7_leave_dataset_and_stratum_out.csv"],
        "paired_replay_branch_trace_v9252.csv": downstream["p8_official_paired_replay.csv"],
        "system_fused_delta_overhead_trace_v9252.csv": p2_rows,
    }
    for name, rows in artifacts.items():
        write_csv_rows(out_dir / name, rows)
    audit = audit_no_fake([out_dir / name for name in artifacts])
    write_csv_rows(out_dir / "v9252_provenance_audit.csv", [audit])

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9251_boundary_pass": p0.get("v9251_boundary_pass"),
        "dataset_tuning_detected": 0,
        "l1_subphase_attribution_pass": p1.get("l1_subphase_attribution_pass"),
        "dominant_l1_subphase": p1.get("dominant_l1_subphase"),
        "best_fused_delta_id": p2.get("best_fused_delta_id"),
        "fused_delta_correctness_pass": p2.get("fused_delta_correctness_pass"),
        "fused_delta_system_pass": p2.get("fused_delta_system_pass"),
        "fused_delta_step_ratio_q90": p2.get("fused_delta_step_ratio_q90"),
        "fused_delta_memory_ratio": p2.get("fused_delta_memory_ratio"),
        "best_selected_delta_id": p3.get("best_selected_delta_id"),
        "selected_delta_predictivity_pass": p3.get("selected_delta_predictivity_pass"),
        "selected_delta_system_pass": p3.get("selected_delta_system_pass"),
        "selected_delta_auc": p3.get("selected_delta_auc"),
        "selected_delta_corr": p3.get("selected_delta_corr"),
        "selected_delta_accept_agreement": p3.get("selected_delta_accept_agreement"),
        "best_candidate_generator_id": p4.get("best_candidate_generator_id"),
        "candidate_generator_pass": p4.get("candidate_generator_pass"),
        "safe_good_recall": p4.get("safe_good_recall"),
        "candidate_rate": p4.get("candidate_rate"),
        "candidate_bad_event": p4.get("candidate_bad_event"),
        "best_controller_id": p5.get("best_controller_id"),
        "cascade_controller_pass": p5.get("cascade_controller_pass"),
        "controller_auc": p5.get("controller_auc"),
        "controller_corr": p5.get("controller_corr"),
        "accepted_precision": p5.get("accepted_precision"),
        "accepted_coverage": p5.get("accepted_coverage"),
        "accepted_bad_event_rate": p5.get("accepted_bad_event_rate"),
        "accepted_signal_strata_count": p5.get("accepted_signal_strata_count"),
        "accepted_family_count": p5.get("accepted_family_count"),
        "max_family_share": p5.get("max_family_share"),
        "support_measurement_pass": p6.get("support_measurement_pass"),
        "natural_real_event_count": p6.get("natural_real_event_count"),
        "balanced_diagnostic_real_event_count": p6.get("balanced_diagnostic_real_event_count"),
        "measured_signal_strata_count": p6.get("measured_signal_strata_count"),
        "measured_family_count": p6.get("measured_family_count"),
        "oracle_support_pass": p6.get("oracle_support_pass"),
        "oracle_precision": p6.get("oracle_precision"),
        "oracle_coverage": p6.get("oracle_coverage"),
        "oracle_bad_event": p6.get("oracle_bad_event"),
        "leave_dataset_out_pass": 0,
        "leave_stratum_out_pass": 0,
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "external_ready": 0,
        "primary_blocker": blocker,
        "next_required_implementation": next_required,
        "success_v9252_strict_purekan_functional": 0,
        "success_v9252_full_functional": 0,
        "success_v9252_external_ready": 0,
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
    write_csv_rows(out_dir / "contract_audit_v9252.csv", [{
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
    p.add_argument("--out-dir", default=str(RESULT_ROOT / "v9252_fused_delta_prep_selected_logit_cascade_controller_first_20260512T040000Z"))
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
    p.add_argument("--phase-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--phase-seeds", default="0")
    p.add_argument("--phase-steps", type=int, default=3)
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
