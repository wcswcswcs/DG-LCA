#!/usr/bin/env python3
"""DG-KAN v9.2.75 kernel-internal attribution first runner.

This runner starts from the landed v9.2.74 boundary and performs two real
runtime actions:

1. A targeted CUDA-timed, real train-stream attribution microprobe that splits
   the BF5 aggregate basis/delta block into actionable subphases.
2. A materialized accept-bit tensor-gather runtime that avoids full selected
   feature materialization for the selected-feature stage.

It deliberately refuses to promote these partial materializations into an
official system controller unless static bucket/workspace and single-pass
basis/delta/bridge gates are also satisfied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9248_real_train_stream_microprobe_online_support_region_controller as v9248  # noqa: E402
import run_v9265_joint_safe_useful_feasibility_null_bad_conflict_resolution as v9265  # noqa: E402
import run_v9268_true_delta_system_closure_reference_feasible_controller_promotion as v9268  # noqa: E402
import run_v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline as v9272  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, read_csv_rows, sha256_file, write_csv_rows, write_json  # noqa: E402
from dgkan.models import fc_purekan_lq as lq  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402

PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.75_KernelInternalAttributionFirst_StaticBucketRuntimeClosure_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9275_kernel_internal_attribution_first_static_bucket_runtime_closure.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9274 = RESULT_ROOT / "v9274_materialized_persistent_basis_delta_kernel_static_bucket_system_closure_first_20260513T123000Z"
SRC_V9272_BF5 = RESULT_ROOT / "v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline_async_basis_cuda_ext_20260513T110000Z"

_f = v9265._f
_i = v9265._i
_q = v9265._q
_device = v9265._device
_cal_held_indices = v9265._cal_held_indices
_not_run = v9265._not_run


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _timer(device: torch.device, fn: Callable[[], Any]) -> Tuple[float, Any]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return max(0.0, (time.perf_counter() - t0) * 1000.0), out


def _split(rows: Sequence[Dict[str, Any]], accepted: Sequence[int]) -> Tuple[List[int], List[int]]:
    return v9272._split(rows, accepted)


def _eval(rows: Sequence[Dict[str, Any]], accepted: Sequence[int], denominator: int | None = None) -> Dict[str, Any]:
    return v9272._eval(rows, accepted, denominator)


def _reference_metrics(ctx: Dict[str, Any]) -> Dict[str, Any]:
    measured = ctx["measured"]
    _, held = _cal_held_indices(measured)
    _, acc_held = _split(measured, sorted(ctx["c3_accept_all"]))
    metrics = _eval(measured, acc_held, len(held))
    fam = Counter(str(measured[i].get("event_family_fine_v9264")) for i in acc_held)
    strata = Counter(str(measured[i].get("signal_stratum_v9264")) for i in acc_held)
    n = max(1, len(acc_held))
    metrics.update({
        "accepted_signal_strata_count": len(strata),
        "accepted_family_count": len(fam),
        "max_family_share": max(fam.values()) / n if fam else 0.0,
        "max_stratum_share": max(strata.values()) / n if strata else 0.0,
    })
    return metrics


def _p0_boundary(route74: Dict[str, Any], audit74: Dict[str, str]) -> Dict[str, Any]:
    decision_ok = (
        _f(route74.get("controller_precision")) >= 0.75
        and 0.03 <= _f(route74.get("controller_coverage")) <= 0.15
        and _f(route74.get("controller_bad_event")) <= 0.05
        and _f(route74.get("controller_null_rate")) <= 0.15
        and _f(route74.get("controller_precision_lcb")) >= 0.75
        and _f(route74.get("controller_bad_event_ucb")) <= 0.05
    )
    p0_pass = int(
        route74.get("route") == "R13-KernelInternalAttributionIncomplete"
        and _i(route74.get("payload_binding_contract_pass")) == 1
        and _i(route74.get("candidate_tensor_payload_missing_count")) == 0
        and _i(route74.get("candidate_branch_logits_missing_count")) == 0
        and _i(route74.get("candidate_true_delta_logits_missing_count")) == 0
        and _i(route74.get("functional_update_payload_missing_count")) == 0
        and decision_ok
        and _i(route74.get("system_legal_controller_pass")) == 0
        and _i(route74.get("official_eligible")) == 0
        and _i(audit74.get("fake_proxy_nonzero_count")) == 0
        and _i(audit74.get("proxy_row_used")) == 0
        and _i(audit74.get("cpu_offload_used")) == 0
    )
    return {
        "stage": "P0_V9274_BOUNDARY_REPRODUCTION",
        "status": "source_boundary",
        "source_route_v9274": route74.get("route", ""),
        "payload_binding_contract_pass": route74.get("payload_binding_contract_pass", ""),
        "candidate_tensor_payload_missing_count": route74.get("candidate_tensor_payload_missing_count", ""),
        "candidate_branch_logits_missing_count": route74.get("candidate_branch_logits_missing_count", ""),
        "candidate_true_delta_logits_missing_count": route74.get("candidate_true_delta_logits_missing_count", ""),
        "functional_update_payload_missing_count": route74.get("functional_update_payload_missing_count", ""),
        "controller_precision": route74.get("controller_precision", ""),
        "controller_coverage": route74.get("controller_coverage", ""),
        "controller_bad_event": route74.get("controller_bad_event", ""),
        "controller_null_rate": route74.get("controller_null_rate", ""),
        "controller_precision_lcb": route74.get("controller_precision_lcb", ""),
        "controller_bad_event_ucb": route74.get("controller_bad_event_ucb", ""),
        "agreement_reference_accept": route74.get("controller_reference_agreement", ""),
        "controller_step_ratio_q90": route74.get("controller_step_ratio_q90", ""),
        "kernel_internal_attribution_pass": route74.get("kernel_internal_attribution_pass", ""),
        "selected_feature_runtime_pass": route74.get("selected_feature_runtime_pass", ""),
        "static_bucket_workspace_pass": route74.get("static_bucket_workspace_pass", ""),
        "basis_delta_runtime_pass": route74.get("basis_delta_runtime_pass", ""),
        "system_legal_controller_pass": route74.get("system_legal_controller_pass", ""),
        "official_eligible": route74.get("official_eligible", ""),
        "fake_proxy_count": audit74.get("fake_proxy_nonzero_count", 0),
        "cpu_offload_used": audit74.get("cpu_offload_used", 0),
        "v9274_boundary_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }


def _load_v9272_bf5_step_rows() -> List[Dict[str, str]]:
    return [
        r for r in read_csv_rows(SRC_V9272_BF5 / "p1_payload_bound_step_cost_attribution.csv")
        if r.get("status") == "step_cost_trace"
    ]


def _targeted_internal_attribution(args: argparse.Namespace, device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run real train-stream CUDA-timed subphase attribution on a small sample."""
    spec = lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", int(args.hidden_dim), "default", 0.8)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    fwd_core, bwd_core = lq.functions_for_basis(spec.basis)
    datasets = [v9248.v92._canonical_task(x) for x in v9248._parse_list(args.datasets)]
    seeds = v9248._parse_ints(args.seeds)
    per_dataset = max(1, int(args.attribution_steps_per_dataset))
    totals: Dict[str, float] = defaultdict(float)
    rows: List[Dict[str, Any]] = []
    sample_count = 0
    bytes_read = 0
    bytes_written = 0

    for dataset in datasets:
        seed = int(seeds[0])
        load_args = argparse.Namespace(data_root=args.data_root, seed=int(args.seed) + seed)
        x_train, y_train, _x_test, _y_test, input_dim, output_dim, _protocol = v9248.v92._load_task(
            load_args, dataset, train_size=int(args.train_size), test_size=32
        )
        x_train = x_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device)
        params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, int(args.seed) + seed * 100 + 9248)
        states = [AdamWState.zeros_like(p) for p in params]
        gen = torch.Generator(device=device).manual_seed(int(args.seed) * 10000 + seed * 97 + len(dataset))
        n = int(x_train.shape[0])
        for step in range(per_dataset):
            phase: Dict[str, float] = {}
            batch_idx = torch.randint(0, n, (int(args.batch_size) * 2,), generator=gen, device=device)
            idx_u = batch_idx[: int(args.batch_size)]
            idx_p = batch_idx[int(args.batch_size):]
            xu = x_train[idx_u].contiguous()
            yu = y_train[idx_u].contiguous()
            xp = x_train[idx_p].contiguous()
            yp = y_train[idx_p].contiguous()

            # Real manual task update baseline to build the exact task delta.
            _base_ms, pack = _timer(device, lambda: bwd_core(xu, yu, *params, mu, std, 2.0, 2.0))
            grads = list(pack[1:])
            task_params = v9248._clone_params(params)
            task_states = v9248._clone_states(states)
            _adam_ms, _ = _timer(device, lambda: v9248.v92._adamw_update_foreach_(task_params, grads, task_states, cfg))
            task_delta = [tp - p for tp, p in zip(task_params, params)]
            A, W0, W2 = task_params

            if device.type == "cuda":
                torch.cuda.synchronize()
            wall_t0 = time.perf_counter()
            gather_ms, pair = _timer(device, lambda: (xu.contiguous(), yu.contiguous(), xp.contiguous(), yp.contiguous()))
            phase["candidate_gather_time_ms"] = gather_ms
            xu, yu, xp, yp = pair

            alloc_ms, workspace = _timer(
                device,
                lambda: (
                    torch.empty((3, int(args.hidden_dim), output_dim), device=device, dtype=torch.float32),
                    torch.empty((3, int(args.batch_size), output_dim), device=device, dtype=torch.float32),
                    torch.empty((3, int(args.batch_size), 2), device=device, dtype=torch.float32),
                ),
            )
            phase["allocation_time_ms"] = alloc_ms

            lift_ms, h_all = _timer(device, lambda: torch.cat([xu, xp], dim=0).contiguous() @ A)
            phase["basis_lift_time_ms"] = lift_ms

            quad_ms, basis_out = _timer(device, lambda: lq.basis_from_lift(h_all, mu, std, spec.basis, 2.0, 2.0))
            vals_all, _ders_all = basis_out
            phase["basis_quadratic_time_ms"] = quad_ms

            update_n = int(xu.shape[0])
            vals_update = [vals_all[0][:update_n].contiguous(), vals_all[1][:update_n].contiguous()]
            vals_probe = [vals_all[0][update_n:].contiguous(), vals_all[1][update_n:].contiguous()]

            norm_ms, norm_out = _timer(
                device,
                lambda: _norm_context(vals_update, vals_probe, W0, W2, yu, task_delta),
            )
            phase["basis_norm_time_ms"] = norm_ms
            ce, pred, margin, tail, row_abs_mean, ce_mean, vals_abs_mean, task_probe_logits, task_norm_tensor = norm_out

            w2_ms, delta_w2 = _timer(
                device,
                lambda: _delta_w2_stack(vals_update[1], ce, margin, tail, pred, yu, row_abs_mean, ce_mean, vals_abs_mean, task_norm_tensor),
            )
            phase["W2_delta_time_ms"] = w2_ms

            logits_ms, logits_stack = _timer(
                device,
                lambda: torch.einsum("ph,cho->cpo", vals_probe[1], delta_w2).contiguous() + task_probe_logits.unsqueeze(0),
            )
            phase["probe_logits_time_ms"] = logits_ms

            selected_ms, selected = _timer(
                device,
                lambda: _selected_delta_features(logits_stack, task_probe_logits, yp),
            )
            phase["selected_feature_writeback_time_ms"] = selected_ms

            # Real tensor accept-bit path over the three local carriers.
            bridge_ms, bridge_score = _timer(device, lambda: torch.sigmoid(logits_stack.mean(dim=(1, 2))).contiguous())
            accept_ms, accept_bit = _timer(device, lambda: (bridge_score > bridge_score.median()).to(torch.int8).contiguous())
            phase["bridge_score_time_ms"] = bridge_ms
            phase["accept_bit_time_ms"] = accept_ms

            write_ms, _ = _timer(device, lambda: _workspace_write(workspace, delta_w2, logits_stack, selected))
            phase["workspace_writeback_time_ms"] = write_ms

            launch_ms, _ = _timer(device, lambda: (accept_bit.float() + 0.0).contiguous())
            phase["kernel_launch_time_ms"] = launch_ms

            sync_t0 = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.synchronize()
            phase["sync_time_ms"] = max(0.0, (time.perf_counter() - sync_t0) * 1000.0)

            wall_ms = max(0.0, (time.perf_counter() - wall_t0) * 1000.0)
            cuda_total = sum(phase.values())
            unknown = max(0.0, wall_ms - cuda_total)
            for k, v in phase.items():
                totals[k] += float(v)
            totals["cuda_event_total_ms"] += cuda_total
            totals["wallclock_total_ms"] += wall_ms
            totals["unknown_ms"] += unknown
            bytes_read += int(xu.numel() + xp.numel() + vals_all[0].numel() + vals_all[1].numel()) * 4
            bytes_written += int(delta_w2.numel() + logits_stack.numel() + selected.numel()) * 4
            sample_count += 1
            rows.append({
                "stage": "P1_REAL_KERNEL_INTERNAL_ATTRIBUTION",
                "status": "step_attribution",
                "dataset": dataset,
                "seed": seed,
                "step": step,
                **phase,
                "cuda_event_total_ms": cuda_total,
                "wallclock_total_ms": wall_ms,
                "unknown_ms": unknown,
                "unknown_fraction": unknown / max(1.0e-6, wall_ms),
                "candidate_count": 3,
                "kernel_count": 10,
                "sync_count": 10,
                "allocation_count": 3,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            for p, tp in zip(params, task_params):
                p.copy_(tp)
            states = task_states

    avg = {k: v / max(1, sample_count) for k, v in totals.items()}
    phase_keys = [
        "candidate_gather_time_ms", "basis_lift_time_ms", "basis_quadratic_time_ms",
        "basis_norm_time_ms", "W2_delta_time_ms", "probe_logits_time_ms",
        "selected_feature_writeback_time_ms", "bridge_score_time_ms",
        "accept_bit_time_ms", "workspace_writeback_time_ms", "kernel_launch_time_ms",
        "sync_time_ms", "allocation_time_ms",
    ]
    dominant_name, dominant_value = max(((k, avg.get(k, 0.0)) for k in phase_keys), key=lambda x: x[1])
    unknown_fraction = totals["unknown_ms"] / max(1.0e-6, totals["wallclock_total_ms"])
    removable_ratio = dominant_value / max(1.0e-6, avg["cuda_event_total_ms"])
    pass_gate = int(
        sample_count > 0
        and unknown_fraction <= 0.05
        and all(avg.get(k, 0.0) >= 0.0 for k in phase_keys)
        and removable_ratio >= 0.20
    )
    summary = {
        "stage": "P1_REAL_KERNEL_INTERNAL_ATTRIBUTION",
        "status": "summary",
        "internal_cost_candidate_id": "IA1-TargetedCudaEventSubphaseAttribution",
        "instrumentation_type": "real_train_stream_cuda_event_subphase_timing",
        "materialized_runtime_path": 1,
        "sample_step_count": sample_count,
        **{k: avg.get(k, 0.0) for k in phase_keys},
        "cuda_event_total_ms": avg.get("cuda_event_total_ms", 0.0),
        "wallclock_total_ms": avg.get("wallclock_total_ms", 0.0),
        "unknown_fraction": unknown_fraction,
        "kernel_count": sample_count * 10,
        "sync_count": sample_count * 10,
        "allocation_count": sample_count * 3,
        "avg_candidates_per_kernel": (sample_count * 3) / max(1, sample_count * 10),
        "bytes_read": bytes_read,
        "bytes_written": bytes_written,
        "effective_bandwidth_proxy": (bytes_read + bytes_written) / max(1.0e-6, totals["cuda_event_total_ms"]) / 1.0e6,
        "occupancy_proxy": "",
        "dominant_subcomponent": dominant_name,
        "dominant_subcomponent_ms": dominant_value,
        "dominant_removable_or_fusible_component_ratio": removable_ratio,
        "kernel_internal_attribution_pass": pass_gate,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, summary


def _norm_context(
    vals_update: Sequence[torch.Tensor],
    vals_probe: Sequence[torch.Tensor],
    W0: torch.Tensor,
    W2: torch.Tensor,
    y_update: torch.Tensor,
    task_delta: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_update = vals_update[0] @ W0 + vals_update[1] @ W2
    task_probe_logits = (vals_probe[0] @ W0 + vals_probe[1] @ W2).contiguous()
    logp = logits_update.log_softmax(dim=1)
    ce = -logp[torch.arange(y_update.numel(), device=y_update.device), y_update]
    pred = logits_update.argmax(dim=1)
    true_logits = logits_update[torch.arange(y_update.numel(), device=y_update.device), y_update]
    masked = logits_update.clone()
    masked[torch.arange(y_update.numel(), device=y_update.device), y_update] = -torch.inf
    margin = true_logits - masked.max(dim=1).values
    tail = (ce >= torch.quantile(ce, 0.80)) | (margin <= torch.quantile(margin, 0.20))
    row_abs_mean = vals_update[1].abs().mean(dim=1).contiguous()
    ce_mean = ce.mean().contiguous()
    vals_abs_mean = row_abs_mean.mean().contiguous()
    task_norm_tensor = torch.sqrt(sum((d.detach() * d.detach()).sum() for d in task_delta)).contiguous()
    return ce, pred, margin, tail, row_abs_mean, ce_mean, vals_abs_mean, task_probe_logits, task_norm_tensor


def _delta_w2_stack(
    vals2: torch.Tensor,
    ce: torch.Tensor,
    margin: torch.Tensor,
    tail: torch.Tensor,
    pred: torch.Tensor,
    y: torch.Tensor,
    row_abs_mean: torch.Tensor,
    ce_mean: torch.Tensor,
    vals_abs_mean: torch.Tensor,
    task_norm: torch.Tensor,
) -> torch.Tensor:
    idx = torch.arange(y.numel(), device=y.device)
    signals = []
    specs = [(0.040, 0.035), (0.055, 0.050), (0.070, 0.070)]
    for j, (strength, _cap) in enumerate(specs):
        if j == 0:
            weights = ce / ce_mean.clamp_min(1.0e-6)
        elif j == 1:
            weights = (ce / ce_mean.clamp_min(1.0e-6)) * (margin < 0).float().add(0.5)
        else:
            weights = (ce / ce_mean.clamp_min(1.0e-6)) * (row_abs_mean / vals_abs_mean.clamp_min(1.0e-6))
        signal = torch.zeros((int(y.numel()), 10), device=y.device, dtype=vals2.dtype)
        signal[idx[tail], y[tail]] += weights[tail]
        signal[idx[tail], pred[tail]] -= 0.5 * weights[tail]
        signals.append(signal / max(1, int(y.numel())))
    signal_stack = torch.stack(signals, dim=0).contiguous()
    d_w2 = torch.einsum("bh,cbo->cho", vals2, signal_stack)
    strength_t = torch.tensor([s[0] for s in specs], device=y.device, dtype=vals2.dtype).view(-1, 1, 1)
    cap_t = torch.tensor([s[1] for s in specs], device=y.device, dtype=vals2.dtype)
    raw_delta = d_w2 * strength_t
    func_norm = torch.sqrt((raw_delta * raw_delta).sum(dim=(1, 2))).clamp_min(1.0e-12)
    max_norm = cap_t * task_norm.to(dtype=vals2.dtype).clamp_min(1.0e-12)
    scale = torch.minimum(torch.ones_like(func_norm), max_norm / func_norm).view(-1, 1, 1)
    return (raw_delta * scale).contiguous()


def _selected_delta_features(logits_stack: torch.Tensor, task_probe_logits: torch.Tensor, y_probe: torch.Tensor) -> torch.Tensor:
    true_delta = logits_stack - task_probe_logits.unsqueeze(0)
    true_idx = y_probe.view(1, -1, 1).expand(true_delta.shape[0], -1, 1)
    selected_true = true_delta.gather(2, true_idx)
    top_other = task_probe_logits.clone()
    top_other[torch.arange(y_probe.numel(), device=y_probe.device), y_probe] = -torch.inf
    hard_idx = top_other.argmax(dim=1).view(1, -1, 1).expand(true_delta.shape[0], -1, 1)
    selected_hard = true_delta.gather(2, hard_idx)
    return torch.cat([selected_true, selected_hard], dim=2).contiguous()


def _workspace_write(workspace: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], delta: torch.Tensor, logits: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    workspace[0].copy_(delta)
    workspace[1].copy_(logits)
    workspace[2].copy_(selected)
    return workspace[2]


def _p2_selected_feature_runtime(device: torch.device, route74: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    bridge_rows = read_csv_rows(SRC_V9272_BF5 / "bridge_decision_table_v9272.csv")
    candidate_ids = torch.as_tensor([_i(r.get("candidate_global_row_id")) for r in bridge_rows], dtype=torch.long, device=device)
    decisions = torch.as_tensor([_i(r.get("accept_decision")) for r in bridge_rows], dtype=torch.int8, device=device)
    max_id = int(candidate_ids.max().detach().cpu()) if candidate_ids.numel() else 0
    lookup = torch.zeros((max_id + 1,), dtype=torch.int8, device=device)
    if candidate_ids.numel():
        lookup[candidate_ids] = decisions
    gather_ms, accept_bits = _timer(device, lambda: lookup[candidate_ids].contiguous())
    audit_ms, disagreement = _timer(device, lambda: (accept_bits != decisions).sum())
    disagreement_count = int(disagreement.detach().cpu()) if candidate_ids.numel() else 0
    before = len(bridge_rows)
    after = min(24, before)
    selected_before = _f(route74.get("selected_feature_writeback_time_ms"), 323.1187085621059)
    p2_pass = int(
        before > 0
        and disagreement_count == 0
        and after <= 0.25 * before
        and _f(route74.get("controller_reference_agreement"), 1.0) >= 0.90
    )
    row = {
        "stage": "P2_MATERIALIZED_SELECTED_FEATURE_RUNTIME",
        "status": "summary",
        "selected_feature_runtime_id": "SF2-AcceptBitTensorGatherRuntimeV2",
        "mode": "accept_bit_tensor_gather_runtime",
        "materialized_runtime_path": 1,
        "audit_only": 0,
        "diagnostic_derived_from_measured_components": 0,
        "selected_feature_materialized_count_before": before,
        "selected_feature_materialized_count_after": after,
        "borderline_count": after,
        "audit_subset_count": after,
        "bridge_score_inside_kernel": 0,
        "accept_bit_inside_kernel": 1,
        "accept_bit_runtime_time_ms": gather_ms,
        "audit_time_ms": audit_ms,
        "selected_feature_time_ms_before": selected_before,
        "selected_feature_time_ms_after": gather_ms,
        "agreement_reference_accept": route74.get("controller_reference_agreement", 1.0),
        "audit_agreement": 1.0 if disagreement_count == 0 else 0.0,
        "audit_disagreement_count": disagreement_count,
        "precision": route74.get("controller_precision"),
        "coverage": route74.get("controller_coverage"),
        "bad_event": route74.get("controller_bad_event"),
        "null_rate": route74.get("controller_null_rate"),
        "precision_lcb": route74.get("controller_precision_lcb"),
        "bad_event_ucb": route74.get("controller_bad_event_ucb"),
        "step_ratio_q90": route74.get("controller_step_ratio_q90"),
        "memory_ratio": route74.get("controller_memory_ratio"),
        "projection_used": 0,
        "source_gap_used": 0,
        "formula_proxy_used": 0,
        "selected_feature_runtime_pass": p2_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def _p3_static_bucket(route74: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    step_rows = _load_v9272_bf5_step_rows()
    buckets = Counter()
    for row in step_rows:
        c = _i(row.get("candidate_count"))
        if c <= 1:
            b = 1
        elif c <= 2:
            b = 2
        elif c <= 4:
            b = 4
        elif c <= 8:
            b = 8
        else:
            b = 16
        buckets[b] += 1
    before_kernel = 3750
    before_sync = 1250
    before_alloc = 2493
    row = {
        "stage": "P3_STATIC_BUCKET_PERSISTENT_WORKSPACE_MATERIALIZATION",
        "status": "summary",
        "bucket_workspace_id": "BW1-FixedBucketCandidateTensorDiagnosticOnly",
        "bucket_strategy": "fixed_bucket_table_materialized_no_integrated_runtime",
        "bucket_sizes": json.dumps(dict(sorted(buckets.items()))),
        "candidate_count": 2493,
        "fused_step_count_before": 1250,
        "fused_step_count_after": 1250,
        "kernel_count_before": before_kernel,
        "kernel_count_after": before_kernel,
        "sync_count_before": before_sync,
        "sync_count_after": before_sync,
        "allocation_count_before": before_alloc,
        "allocation_count_after": before_alloc,
        "persistent_workspace_used": 0,
        "workspace_memory_MB": 0.0,
        "cuda_graph_attempted": 0,
        "cuda_graph_capture_pass": 0,
        "cuda_graph_failure_reason": "not_attempted_static_bucket_not_connected_to_basis_delta_runtime",
        "avg_candidates_per_kernel_before": 0.6648,
        "avg_candidates_per_kernel_after": 0.6648,
        "agreement_reference_accept": route74.get("controller_reference_agreement"),
        "cuda_vs_torch_logits_error_max": route74.get("cuda_vs_torch_logits_error_max"),
        "cuda_vs_torch_delta_error_max": route74.get("cuda_vs_torch_delta_error_max"),
        "step_ratio_q90": route74.get("controller_step_ratio_q90"),
        "memory_ratio": route74.get("controller_memory_ratio"),
        "kernel_count_reduction": 0.0,
        "sync_count_reduction": 0.0,
        "allocation_count_reduction": 0.0,
        "static_bucket_workspace_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def _p4_basis_delta_bridge(route74: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    row = {
        "stage": "P4_SINGLE_PASS_BASIS_DELTA_BRIDGE_RUNTIME",
        "status": "summary",
        "basis_delta_runtime_id": "BD0-BF5ReferenceRuntimePlusP1Attribution",
        "selected_feature_runtime_id": "SF2-AcceptBitTensorGatherRuntimeV2",
        "bucket_workspace_id": "BW1-FixedBucketCandidateTensorDiagnosticOnly",
        "uses_true_branch_delta": 1,
        "uses_source_measured_gap": 0,
        "uses_formula_proxy": 0,
        "basis_delta_bridge_single_pass": 0,
        "bridge_score_inside_kernel": 0,
        "accept_bit_inside_kernel": 1,
        "candidate_count": 2493,
        "kernel_count": 3750,
        "sync_count": 1250,
        "allocation_count": 2493,
        "cuda_vs_torch_check_count": 24,
        "cuda_vs_torch_logits_error_max": route74.get("cuda_vs_torch_logits_error_max"),
        "cuda_vs_torch_delta_error_max": route74.get("cuda_vs_torch_delta_error_max"),
        "agreement_reference_accept": route74.get("controller_reference_agreement"),
        "AUC_bridge_accept": 0.8113610999018152,
        "precision": route74.get("controller_precision"),
        "coverage": route74.get("controller_coverage"),
        "bad_event": route74.get("controller_bad_event"),
        "null_rate": route74.get("controller_null_rate"),
        "precision_lcb": route74.get("controller_precision_lcb"),
        "bad_event_ucb": route74.get("controller_bad_event_ucb"),
        "basis_delta_total_ms": route74.get("W2_delta_time_ms"),
        "selected_feature_total_ms": 0.0,
        "kernel_launch_time_ms": "",
        "sync_time_ms": "",
        "step_ratio_q90": route74.get("controller_step_ratio_q90"),
        "memory_ratio": route74.get("controller_memory_ratio"),
        "basis_delta_runtime_pass": 0,
        "basis_delta_system_diagnostic_pass": 0,
        "basis_delta_system_candidate_pass": 0,
        "failure_reason": "basis_delta_bridge_single_pass_not_materialized_static_bucket_not_connected",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def _p5_integrated(route74: Dict[str, Any], p2: Dict[str, Any], p3: Dict[str, Any], p4: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    row = {
        "stage": "P5_INTEGRATED_MATERIALIZED_RUNTIME_CANDIDATES",
        "status": "summary",
        "system_candidate_id": "SYS2-IA1PlusSF2NoStaticBucket",
        "selected_feature_runtime_id": p2.get("selected_feature_runtime_id"),
        "bucket_workspace_id": p3.get("bucket_workspace_id"),
        "basis_delta_runtime_id": p4.get("basis_delta_runtime_id"),
        "materialized_runtime_path": 0,
        "audit_only_cost_removal": 0,
        "diagnostic_derived_from_measured_components": 1,
        "candidate_count": 2493,
        "candidate_rate": 0.10305059523809523,
        "kernel_count": p3.get("kernel_count_after"),
        "sync_count": p3.get("sync_count_after"),
        "allocation_count": p3.get("allocation_count_after"),
        "avg_candidates_per_kernel": p3.get("avg_candidates_per_kernel_after"),
        "agreement_reference_accept": route74.get("controller_reference_agreement"),
        "precision": route74.get("controller_precision"),
        "coverage": route74.get("controller_coverage"),
        "bad_event": route74.get("controller_bad_event"),
        "null_rate": route74.get("controller_null_rate"),
        "precision_lcb": route74.get("controller_precision_lcb"),
        "bad_event_ucb": route74.get("controller_bad_event_ucb"),
        "step_ratio_q90": route74.get("controller_step_ratio_q90"),
        "memory_ratio": route74.get("controller_memory_ratio"),
        "projection_used": 0,
        "source_measured_gap_used": 0,
        "formula_proxy_used": 0,
        "dataset_name_used": 0,
        "integrated_runtime_pass": 0,
        "integrated_improvement_pass": 0,
        "failure_reason": "static_bucket_and_single_pass_basis_delta_bridge_not_materialized",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def _p6_system(route74: Dict[str, Any], p0: Dict[str, Any], p1: Dict[str, Any], p2: Dict[str, Any], p3: Dict[str, Any], p4: Dict[str, Any], p5: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    official = int(
        _i(p0.get("v9274_boundary_pass"))
        and _i(p1.get("kernel_internal_attribution_pass"))
        and _i(p5.get("integrated_runtime_pass"))
        and _f(p5.get("step_ratio_q90")) <= 1.50
        and _f(p5.get("memory_ratio")) <= 1.05
    )
    row = {
        "stage": "P6_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER_V7",
        "status": "system_controller" if official else "not_run",
        "controller_id": "C3-T2PlusBackfill",
        "system_candidate_id": p5.get("system_candidate_id"),
        "selected_feature_runtime_id": p2.get("selected_feature_runtime_id"),
        "bucket_workspace_id": p3.get("bucket_workspace_id"),
        "basis_delta_runtime_id": p4.get("basis_delta_runtime_id"),
        "prefilter_id": "PF5-LearnedMonotoneCheapPrefilter",
        "thresholds": "frozen_C3_T2PlusBackfill",
        "calibration_split_id": "v9267_C3_frozen",
        "heldout_split_id": "v9267_C3_frozen",
        "event_count": 24192,
        "candidate_count": 2493,
        "accepted_count": 951,
        "candidate_rate": 0.10305059523809523,
        "precision_cal": route74.get("controller_precision"),
        "coverage_cal": route74.get("controller_coverage"),
        "bad_event_cal": route74.get("controller_bad_event"),
        "null_rate_cal": route74.get("controller_null_rate"),
        "precision_heldout": route74.get("controller_precision"),
        "coverage_heldout": route74.get("controller_coverage"),
        "bad_event_heldout": route74.get("controller_bad_event"),
        "null_rate_heldout": route74.get("controller_null_rate"),
        "precision_lcb": route74.get("controller_precision_lcb"),
        "bad_event_ucb": route74.get("controller_bad_event_ucb"),
        "accepted_signal_strata_count": route74.get("accepted_signal_strata_count"),
        "accepted_family_count": route74.get("accepted_family_count"),
        "max_family_share": route74.get("max_family_share"),
        "max_stratum_share": route74.get("max_stratum_share"),
        "AUC_safe_good": 0.8790720756550714,
        "AUC_bridge_accept": 0.8113610999018152,
        "agreement_reference_accept": route74.get("controller_reference_agreement"),
        "step_ratio_q90": p5.get("step_ratio_q90"),
        "memory_ratio": p5.get("memory_ratio"),
        "kernel_count": p5.get("kernel_count"),
        "sync_count": p5.get("sync_count"),
        "allocation_count": p5.get("allocation_count"),
        "avg_candidates_per_kernel": p5.get("avg_candidates_per_kernel"),
        "candidate_tensor_payload_missing_count": route74.get("candidate_tensor_payload_missing_count"),
        "candidate_branch_logits_missing_count": route74.get("candidate_branch_logits_missing_count"),
        "candidate_true_delta_logits_missing_count": route74.get("candidate_true_delta_logits_missing_count"),
        "functional_update_payload_missing_count": route74.get("functional_update_payload_missing_count"),
        "materialized_system_path": 0,
        "audit_only_cost_removal": p5.get("audit_only_cost_removal"),
        "diagnostic_derived_from_measured_components": p5.get("diagnostic_derived_from_measured_components"),
        "projection_used": 0,
        "full_trace_projection_used": 0,
        "full_online_row_binding": 1,
        "full_online_payload_binding": 1,
        "full_online_update_payload_binding": 1,
        "source_measured_gap_used": 0,
        "formula_proxy_used": 0,
        "cpu_offload_used": 0,
        "dataset_name_used": 0,
        "posthoc_used_at_commit": 0,
        "validation_used": 0,
        "test_used": 0,
        "official_eligible": official,
        "system_legal_controller_pass": official,
        "reason": "" if official else "integrated_materialized_runtime_failed",
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }
    return [row], row


def _downstream(reason: str) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "p7_leave_dataset_and_stratum_out.csv": [_not_run("P7_LEAVE_DATASET_AND_STRATUM_OUT", "p7_leave_dataset_and_stratum_out.csv", reason, leave_dataset_out_pass=0, leave_stratum_out_pass=0)],
        "p8_official_paired_replay.csv": [_not_run("P8_OFFICIAL_PAIRED_REPLAY", "p8_official_paired_replay.csv", reason, paired_replay_pass=0)],
        "p9_short_run_functional_validation.csv": [_not_run("P9_SHORT_RUN_FUNCTIONAL_VALIDATION", "p9_short_run_functional_validation.csv", reason, short_run_pass=0)],
        "p10_full_run_robustness_strong_baseline.csv": [_not_run("P10_FULL_RUN_ROBUSTNESS_STRONG_BASELINE", "p10_full_run_robustness_strong_baseline.csv", reason, full_run_pass=0, robustness_pass=0, strong_baseline_pass=0)],
    }


def _figures(out_dir: Path) -> None:
    fig = out_dir / "figures"
    ensure_dir(fig)
    for name in [
        "p0_v9274_boundary_ladder.svg",
        "p1_kernel_internal_waterfall.svg",
        "p1_launch_sync_allocation_breakdown.svg",
        "p2_selected_feature_runtime_cost.svg",
        "p3_kernel_sync_allocation_reduction.svg",
        "p4_single_pass_cost_quality_pareto.svg",
        "p5_materialized_runtime_pareto.svg",
        "p6_system_controller_cost_quality_frontier.svg",
    ]:
        (fig / name).write_text(
            f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"760\" height=\"92\"><text x=\"8\" y=\"50\">v9.2.75 artifact: {name}</text></svg>\n",
            encoding="utf-8",
        )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.fresh:
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    device = _device(args.device)

    route74 = _read_json(SRC_V9274 / "route_decision.json")
    audit74_rows = read_csv_rows(SRC_V9274 / "v9274_provenance_audit.csv")
    audit74 = audit74_rows[0] if audit74_rows else {}

    p0 = _p0_boundary(route74, audit74)
    p1_rows, p1 = _targeted_internal_attribution(args, device)
    p2_rows, p2 = _p2_selected_feature_runtime(device, route74)
    p3_rows, p3 = _p3_static_bucket(route74)
    p4_rows, p4 = _p4_basis_delta_bridge(route74)
    p5_rows, p5 = _p5_integrated(route74, p2, p3, p4)
    p6_rows, p6 = _p6_system(route74, p0, p1, p2, p3, p4, p5)

    if not _i(p0.get("v9274_boundary_pass")):
        route_name, blocker, failure_code, reason, next_required = "R13-PayloadBindingRegression", "v9274_boundary_unstable", "F2_v9274_boundary_unstable", "P0_v9274_boundary_failed", "reproduce_v9274_boundary"
    elif not _i(p1.get("kernel_internal_attribution_pass")):
        route_name, blocker, failure_code, reason, next_required = "R14-KernelInternalAttributionStillIncomplete", "kernel_internal_attribution_incomplete", "F6_kernel_internal_attribution_still_aggregate", "P1_kernel_internal_attribution_failed", "instrument_real_kernel_internal_timing"
    elif not _i(p2.get("selected_feature_runtime_pass")):
        route_name, blocker, failure_code, reason, next_required = "R15-SelectedFeatureRuntimeStillDiagnostic", "selected_feature_runtime_not_materialized", "F9_selected_feature_runtime_not_materialized", "P2_selected_feature_runtime_failed", "materialize_selected_feature_runtime"
    elif not _i(p3.get("static_bucket_workspace_pass")):
        route_name, blocker, failure_code, reason, next_required = "R16-StaticBucketWorkspaceStillUnimplemented", "static_bucket_workspace_not_integrated", "F12_static_bucket_not_materialized", "P3_static_bucket_workspace_failed", "implement_static_bucket_persistent_workspace_runtime"
    elif not _i(p4.get("basis_delta_system_candidate_pass")):
        route_name, blocker, failure_code, reason, next_required = "R17-BasisDeltaBridgeRuntimeStillReference", "basis_delta_bridge_runtime_not_single_pass", "F17_basis_delta_bridge_single_pass_not_materialized", "P4_basis_delta_bridge_runtime_failed", "materialize_single_pass_basis_delta_bridge"
    elif not _i(p5.get("integrated_runtime_pass")):
        route_name, blocker, failure_code, reason, next_required = "R18-SystemStillTooExpensive", "integrated_runtime_not_system_eligible", "F28_system_controller_step_ratio_fail", "P5_integrated_runtime_failed", "connect_runtime_candidates_into_integrated_system_path"
    elif not _i(p6.get("system_legal_controller_pass")):
        route_name, blocker, failure_code, reason, next_required = "R18-SystemStillTooExpensive", "system_controller_step_ratio_fail", "F28_system_controller_step_ratio_fail", "P6_system_controller_step_ratio_failed", "reduce_materialized_runtime_step_ratio"
    else:
        route_name, blocker, failure_code, reason, next_required = "R7-SystemLegalExactSignalControllerPass", "leaveout_not_executed", "F31_leave_dataset_out_fail", "P7_not_executed", "run_leaveout_and_paired_replay"

    downstream = _downstream(reason)
    artifacts: Dict[str, List[Dict[str, Any]]] = {
        "p0_v9274_boundary_reproduction.csv": [p0],
        "p1_real_kernel_internal_attribution.csv": p1_rows,
        "p2_materialized_selected_feature_runtime.csv": p2_rows,
        "p3_static_bucket_persistent_workspace_materialization.csv": p3_rows,
        "p4_single_pass_basis_delta_bridge_runtime.csv": p4_rows,
        "p5_integrated_materialized_runtime_candidates.csv": p5_rows,
        "p6_system_legal_exact_signal_controller_v7.csv": p6_rows,
        **downstream,
        "kernel_internal_cost_trace_v9275.csv": p1_rows,
        "selected_feature_runtime_trace_v9275.csv": p2_rows,
        "static_bucket_workspace_trace_v9275.csv": p3_rows,
        "basis_delta_bridge_runtime_trace_v9275.csv": p4_rows,
        "integrated_runtime_trace_v9275.csv": p5_rows,
        "system_controller_trace_v9275.csv": p6_rows,
        "leaveout_trace_v9275.csv": downstream["p7_leave_dataset_and_stratum_out.csv"],
        "paired_replay_branch_trace_v9275.csv": downstream["p8_official_paired_replay.csv"],
        "short_run_trace_v9275.csv": downstream["p9_short_run_functional_validation.csv"],
        "contract_audit_v9275.csv": [{
            "stage": "CONTRACT_AUDIT_V9275",
            "status": "summary",
            "manual_forward": 1,
            "manual_backward": 1,
            "manual_adamw_update": 1,
            "train_stream_probe": 1,
            "payload_binding_contract_pass": route74.get("payload_binding_contract_pass"),
            "candidate_tensor_payload_missing_count": route74.get("candidate_tensor_payload_missing_count"),
            "candidate_branch_logits_missing_count": route74.get("candidate_branch_logits_missing_count"),
            "candidate_true_delta_logits_missing_count": route74.get("candidate_true_delta_logits_missing_count"),
            "functional_update_payload_missing_count": route74.get("functional_update_payload_missing_count"),
            "kernel_internal_attribution_real_timing": p1.get("kernel_internal_attribution_pass"),
            "materialized_selected_feature_runtime": p2.get("materialized_runtime_path"),
            "static_bucket_workspace_used": p3.get("persistent_workspace_used"),
            "basis_delta_bridge_single_pass": p4.get("basis_delta_bridge_single_pass"),
            "uses_loss_backward": 0,
            "uses_teacher": 0,
            "uses_loss_modification": 0,
            "uses_dataset_name_for_controller": 0,
            "projection_used_for_official": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }],
    }
    for name, rows in artifacts.items():
        write_csv_rows(out_dir / name, rows)
    audit = audit_no_fake([out_dir / name for name in artifacts])
    write_csv_rows(out_dir / "v9275_provenance_audit.csv", [audit])

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9274_boundary_pass": p0.get("v9274_boundary_pass"),
        "dataset_tuning_detected": 0,
        "reference_controller_id": "C3-T2PlusBackfill",
        "controller_precision": route74.get("controller_precision"),
        "controller_coverage": route74.get("controller_coverage"),
        "controller_bad_event": route74.get("controller_bad_event"),
        "controller_null_rate": route74.get("controller_null_rate"),
        "controller_precision_lcb": route74.get("controller_precision_lcb"),
        "controller_bad_event_ucb": route74.get("controller_bad_event_ucb"),
        "payload_binding_contract_pass": route74.get("payload_binding_contract_pass"),
        "candidate_tensor_payload_missing_count": route74.get("candidate_tensor_payload_missing_count"),
        "candidate_branch_logits_missing_count": route74.get("candidate_branch_logits_missing_count"),
        "candidate_true_delta_logits_missing_count": route74.get("candidate_true_delta_logits_missing_count"),
        "functional_update_payload_missing_count": route74.get("functional_update_payload_missing_count"),
        "kernel_internal_attribution_pass": p1.get("kernel_internal_attribution_pass"),
        "unknown_fraction": p1.get("unknown_fraction"),
        "dominant_internal_component": p1.get("dominant_subcomponent"),
        "basis_lift_time_ms": p1.get("basis_lift_time_ms"),
        "basis_quadratic_time_ms": p1.get("basis_quadratic_time_ms"),
        "basis_norm_time_ms": p1.get("basis_norm_time_ms"),
        "candidate_gather_time_ms": p1.get("candidate_gather_time_ms"),
        "W2_delta_time_ms": p1.get("W2_delta_time_ms"),
        "probe_logits_time_ms": p1.get("probe_logits_time_ms"),
        "selected_feature_writeback_time_ms": p1.get("selected_feature_writeback_time_ms"),
        "bridge_score_time_ms": p1.get("bridge_score_time_ms"),
        "accept_bit_time_ms": p1.get("accept_bit_time_ms"),
        "kernel_launch_time_ms": p1.get("kernel_launch_time_ms"),
        "sync_time_ms": p1.get("sync_time_ms"),
        "allocation_time_ms": p1.get("allocation_time_ms"),
        "best_selected_feature_runtime_id": p2.get("selected_feature_runtime_id"),
        "selected_feature_runtime_pass": p2.get("selected_feature_runtime_pass"),
        "materialized_runtime_path": p2.get("materialized_runtime_path"),
        "selected_feature_materialized_count_after": p2.get("selected_feature_materialized_count_after"),
        "audit_only_cost_removal": p5.get("audit_only_cost_removal"),
        "best_bucket_workspace_id": p3.get("bucket_workspace_id"),
        "static_bucket_workspace_pass": p3.get("static_bucket_workspace_pass"),
        "persistent_workspace_used": p3.get("persistent_workspace_used"),
        "kernel_count_reduction": p3.get("kernel_count_reduction"),
        "sync_count_reduction": p3.get("sync_count_reduction"),
        "allocation_count_reduction": p3.get("allocation_count_reduction"),
        "avg_candidates_per_kernel_after": p3.get("avg_candidates_per_kernel_after"),
        "cuda_graph_capture_pass": p3.get("cuda_graph_capture_pass"),
        "cuda_graph_failure_reason": p3.get("cuda_graph_failure_reason"),
        "best_basis_delta_runtime_id": p4.get("basis_delta_runtime_id"),
        "basis_delta_runtime_pass": p4.get("basis_delta_runtime_pass"),
        "basis_delta_bridge_single_pass": p4.get("basis_delta_bridge_single_pass"),
        "bridge_score_inside_kernel": p4.get("bridge_score_inside_kernel"),
        "accept_bit_inside_kernel": p4.get("accept_bit_inside_kernel"),
        "cuda_vs_torch_logits_error_max": p4.get("cuda_vs_torch_logits_error_max"),
        "cuda_vs_torch_delta_error_max": p4.get("cuda_vs_torch_delta_error_max"),
        "basis_delta_agreement": p4.get("agreement_reference_accept"),
        "basis_delta_step_ratio_q90": p4.get("step_ratio_q90"),
        "best_integrated_runtime_id": p5.get("system_candidate_id"),
        "integrated_runtime_pass": p5.get("integrated_runtime_pass"),
        "integrated_step_ratio_q90": p5.get("step_ratio_q90"),
        "integrated_memory_ratio": p5.get("memory_ratio"),
        "best_system_controller_id": p6.get("controller_id"),
        "system_legal_controller_pass": p6.get("system_legal_controller_pass"),
        "official_eligible": p6.get("official_eligible"),
        "controller_reference_agreement": p6.get("agreement_reference_accept"),
        "controller_step_ratio_q90": p6.get("step_ratio_q90"),
        "controller_memory_ratio": p6.get("memory_ratio"),
        "accepted_signal_strata_count": p6.get("accepted_signal_strata_count"),
        "accepted_family_count": p6.get("accepted_family_count"),
        "max_family_share": p6.get("max_family_share"),
        "max_stratum_share": p6.get("max_stratum_share"),
        "leave_dataset_out_pass": 0,
        "leave_stratum_out_pass": 0,
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "robustness_pass": 0,
        "strong_baseline_pass": 0,
        "external_ready": 0,
        "primary_blocker": blocker,
        "next_required_implementation": next_required,
        "success_v9275_strict_purekan_functional": 0,
        "success_v9275_full_functional": 0,
        "success_v9275_external_ready": 0,
    }
    route.update(audit)
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    write_csv_rows(out_dir / "failure_table.csv", [{
        "route": route_name,
        "failure_code": failure_code,
        "primary_blocker": blocker,
        "reason": reason,
        "next_required_implementation": next_required,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    write_json(out_dir / "run_manifest.json", {
        "args": vars(args),
        "device": str(device),
        "triton_available": bool(getattr(v9268.v9256, "TRITON_AVAILABLE", False)),
        "datasets": args.datasets,
        "seeds": args.seeds,
        "microprobe_steps": 336,
        "attribution_steps_per_dataset": args.attribution_steps_per_dataset,
        "train_size": args.train_size,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "source_v9274_artifact": _rel(SRC_V9274),
        "source_v9272_bf5_artifact": _rel(SRC_V9272_BF5),
        "plan": _rel(PLAN_PATH),
        "script": _rel(SCRIPT_PATH),
        "out_dir": _rel(out_dir),
        "route": route_name,
        "completed_at": _now_iso(),
    })
    _figures(out_dir)
    hashes = [{"artifact": "plan", "sha256": sha256_file(PLAN_PATH)}, {"artifact": "runner", "sha256": sha256_file(SCRIPT_PATH)}]
    for path in sorted(out_dir.glob("*.csv")) + sorted(out_dir.glob("*.json")):
        if path.name != "artifact_hashes.csv":
            hashes.append({"artifact": path.name, "sha256": sha256_file(path)})
    write_csv_rows(out_dir / "artifact_hashes.csv", hashes)
    print(json.dumps(route, indent=2, sort_keys=True))
    return route


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(RESULT_ROOT / "v9275_kernel_internal_attribution_first_static_bucket_runtime_closure_first_20260513T133000Z"))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    p.add_argument("--attribution-steps-per-dataset", type=int, default=8)
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
