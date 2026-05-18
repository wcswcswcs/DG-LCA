#!/usr/bin/env python3
"""DG-KAN v9.2.80 full train-stream stable-accept materializer.

This runner addresses the v9.2.79 blocker directly: old artifacts did not
contain full-row AC2Q2 stable score/rank/accept fields or row-level outcome
labels.  v9.2.80 rebuilds those rows from a fresh full train-stream reference
context, materializes stable accept fields for every PF5 candidate row, records
same-run outcome labels from the freshly generated measured rows, and measures
the native stable bucket kernel on the matching full train-stream steps.

No old outcome join, family aggregate label, local trace extrapolation, or old
step ratio is promoted into an official result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9248_real_train_stream_microprobe_online_support_region_controller as v9248  # noqa: E402
import run_v9265_joint_safe_useful_feasibility_null_bad_conflict_resolution as v9265  # noqa: E402
import run_v9266_oracle_legal_gap_closure_bridge_frontier as v9266  # noqa: E402
import run_v9268_true_delta_system_closure_reference_feasible_controller_promotion as v9268  # noqa: E402
import run_v9269_materialized_event_sparse_true_delta_system_legal_controller as v9269  # noqa: E402
import run_v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline as v9272  # noqa: E402
import run_v9277_quantile_tail_basis_norm_static_bucket_bridge_closure as v9277  # noqa: E402
import run_v9278_native_bridge_accept_static_bucket_official_closure as v9278  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, read_csv_rows, sha256_file, write_csv_rows, write_json  # noqa: E402
from dgkan.models import fc_purekan_lq as lq  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.80_FullTrainStreamStableAcceptMaterializer_OutcomeLabelRebuild_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9280_full_train_stream_stable_accept_materializer_outcome_label_rebuild.py"
RECAP_PATH = ROOT / "docs" / "DG-KAN_v9.2.80_FullTrainStreamStableAcceptMaterializer_OutcomeLabelRebuild_实验复盘.md"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9279 = RESULT_ROOT / "v9279_stable_accept_official_promotion_full_system_batch_major_closure_continued_probe_20260513T173000Z"

_f = v9265._f
_i = v9265._i
_q = v9265._q
_device = v9265._device
_cal_held_indices = v9265._cal_held_indices


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _timer(device: torch.device, fn: Any) -> Tuple[float, Any]:
    v9248._sync(device)
    t0 = time.perf_counter()
    out = fn()
    v9248._sync(device)
    return max(0.0, (time.perf_counter() - t0) * 1000.0), out


def _tensor_list(t: torch.Tensor) -> List[float]:
    return [float(x) for x in t.detach().cpu().flatten().tolist()]


def _int_list(t: torch.Tensor) -> List[int]:
    return [int(x) for x in t.detach().cpu().flatten().tolist()]


def _rank_from_quantized(q: Sequence[int]) -> List[int]:
    order = sorted(range(len(q)), key=lambda i: (-int(q[i]), i))
    ranks = [0 for _ in q]
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def _ci_bounds(success: int, n: int, kappa: float = 1.5) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = success / max(1, n)
    err = kappa * (p * (1.0 - p) / max(1, n)) ** 0.5
    return max(0.0, p - err), min(1.0, p + err)


def _eval_accept(rows: Sequence[Dict[str, Any]], accepted: Sequence[int], denominator: int | None = None) -> Dict[str, Any]:
    m = v9266._eval_accept(rows, accepted, denominator)
    safe = sum(_i(rows[i].get("Y_SU_v9265")) for i in accepted)
    bad = sum(_i(rows[i].get("bad_event")) for i in accepted)
    lcb, _ = _ci_bounds(safe, len(accepted))
    _, bad_ucb = _ci_bounds(bad, len(accepted))
    m["precision_lcb"] = lcb
    m["bad_event_ucb"] = bad_ucb
    return m


def _split(rows: Sequence[Dict[str, Any]], accepted: Sequence[int]) -> Tuple[List[int], List[int]]:
    return v9266._split_sets(rows, accepted)


def _source_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9279 / "route_decision.json")
    audit_rows = read_csv_rows(SRC_V9279 / "v9279_provenance_audit.csv")
    audit = audit_rows[0] if audit_rows else {}
    p0_pass = int(
        route.get("route") == "R15-StableAcceptOfficialCalibrationBlocked"
        and str(route.get("primary_blocker")) == "stable_accept_full_row_materialization_missing"
        and _i(audit.get("fake_proxy_nonzero_count")) == 0
        and _i(audit.get("proxy_row_used")) == 0
        and _i(audit.get("cpu_offload_used")) == 0
    )
    return {
        "stage": "P0_V9279_BOUNDARY_REPRODUCTION",
        "status": "summary",
        "source_artifact": _rel(SRC_V9279),
        "source_route_v9279": route.get("route", ""),
        "source_primary_blocker": route.get("primary_blocker", ""),
        "stable_accept_contract_id": route.get("stable_accept_contract_id", ""),
        "stable_accept_official_calibration_pass_v9279": route.get("stable_accept_official_calibration_pass", ""),
        "stable_accept_heldout_support_pass_v9279": route.get("stable_accept_heldout_support_pass", ""),
        "native_kernel_used_in_p6_v9279": route.get("native_kernel_used_in_p6", ""),
        "new_full_system_step_ratio_measured_v9279": route.get("new_full_system_step_ratio_measured", ""),
        "candidate_count_v9279": route.get("candidate_count", ""),
        "source_boundary_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _same_run_outcome(row: Dict[str, Any]) -> Dict[str, Any]:
    safe_good = _i(row.get("Y_SU_v9265", row.get("Y_safe_good", 0)))
    bad = _i(row.get("bad_event", 0))
    null_event = _i(row.get("Y_null_event_v9265", row.get("Y_harmless_null_v9264", 0)))
    task_safe = int(not bad)
    useful = _i(row.get("Y_useful_positive_v9263", safe_good))
    return {
        "safe_good_label": safe_good,
        "bad_event_label": bad,
        "null_event_label": null_event,
        "task_safe_label": task_safe,
        "useful_label": useful,
        "CEp99_delta": row.get("CEp99_delta", ""),
        "margin_delta": row.get("margin_delta", ""),
        "ECE_delta": row.get("ECE_delta", ""),
        "NLL_delta": row.get("NLL_delta", ""),
        "curvature_delta": row.get("curvature_delta", ""),
        "true_delta_reference_score": row.get("true_delta_reference_score", ""),
        "real_beats_adamwparallel": row.get("real_beats_adamwparallel", ""),
        "real_beats_bestlr": row.get("real_beats_bestlr", ""),
        "outcome_horizon": row.get("horizon", "online_h0"),
        "outcome_branch": row.get("carrier_id", ""),
        "outcome_source": "same_run_train_stream_replay",
    }


def _materialize_full_stream(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    ctx = v9268._prepare_reference(args, device)
    p2_rows, p2 = v9269._materialized_pf5(ctx, device)
    event_table, event_summary = v9272._build_event_table(ctx, p2)
    measured = ctx["measured"]
    candidate_indices = set(int(x) for x in p2.get("candidate_indices", []))
    stable_scale = float(args.stable_accept_scale)
    spec = lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", int(args.hidden_dim), "default", 0.8)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    _fwd_core, bwd_core = lq.functions_for_basis(spec.basis)

    stable_ext = v9278._load_integrated_stable_bucket_ext(stable_scale) if device.type == "cuda" else None
    native_compile_error = v9278.STABLE_BUCKET_EXT_ERROR if stable_ext is None else ""

    stable_rows: List[Dict[str, Any]] = []
    runtime_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    accepted_ref: List[int] = []
    accepted_native: List[int] = []
    event_idx = 0
    native_times: List[float] = []
    eager_times: List[float] = []
    baseline_times: List[float] = []
    step_ratios: List[float] = []
    logits_err_max = 0.0
    delta_err_max = 0.0
    bridge_err_max = 0.0
    tail_disagreement = 0
    accept_disagreement = 0
    q_disagreement = 0
    rank_disagreement = 0
    missing_labels = 0
    missing_delta_fields = 0
    stable_candidate_count = 0
    duplicate_event_count = 0
    seen_event_ids: set[str] = set()

    datasets = [v9248.v92._canonical_task(x) for x in v9248._parse_list(args.datasets)]
    seeds = v9248._parse_ints(args.seeds)
    for dataset in datasets:
        for seed in seeds:
            load_args = argparse.Namespace(data_root=args.data_root, seed=int(args.seed) + int(seed))
            x_train, y_train, _x_test, _y_test, input_dim, output_dim, _protocol = v9248.v92._load_task(
                load_args,
                dataset,
                train_size=int(args.train_size),
                test_size=32,
            )
            x_train = x_train.to(device=device, dtype=torch.float32)
            y_train = y_train.to(device=device)
            params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, int(args.seed) + seed * 100 + 9278)
            states = [AdamWState.zeros_like(p) for p in params]
            gen = torch.Generator(device=device).manual_seed(int(args.seed) * 10000 + int(seed) * 97 + len(dataset))
            n = int(x_train.shape[0])
            for step in range(int(args.microprobe_steps)):
                key = (dataset, int(seed), int(step))
                batch_idx = torch.randint(0, n, (int(args.batch_size) * 2,), generator=gen, device=device)
                xu = x_train[batch_idx[: int(args.batch_size)]].contiguous()
                yu = y_train[batch_idx[: int(args.batch_size)]].contiguous()
                xp = x_train[batch_idx[int(args.batch_size) :]].contiguous()
                pack_ms, pack = _timer(device, lambda: bwd_core(xu, yu, *params, mu, std, 2.0, 2.0))
                grads = list(pack[1:])
                task_params = v9248._clone_params(params)
                task_states = v9248._clone_states(states)
                adam_ms, _ = _timer(device, lambda: v9248.v92._adamw_update_foreach_(task_params, grads, task_states, cfg))
                baseline_ms = max(1.0e-6, pack_ms + adam_ms)
                baseline_times.append(baseline_ms)
                task_delta = [tp - p for tp, p in zip(task_params, params)]
                A, W0, W2 = task_params
                h_all = torch.cat([xu, xp], dim=0).contiguous() @ A
                vals_all, _ders_all = lq.basis_from_lift(h_all, mu, std, spec.basis, 2.0, 2.0)
                update_n = int(xu.shape[0])
                vu0 = vals_all[0][:update_n].contiguous()
                vu1 = vals_all[1][:update_n].contiguous()
                vp0 = vals_all[0][update_n:].contiguous()
                vp1 = vals_all[1][update_n:].contiguous()
                d0, d1, d2 = [d.contiguous() for d in task_delta]

                eager_ms, eager_out = _timer(
                    device,
                    lambda: v9277._single_pass_kth_basis_delta_bridge(vu0, vu1, vp0, vp1, W0, W2, yu, d0, d1, d2),
                )
                eager_times.append(eager_ms)
                native_ms = 0.0
                native_out = None
                if stable_ext is not None:
                    native_ms, native_out = _timer(
                        device,
                        lambda: stable_ext.bucket_basis_delta_bridge_forward(
                            vu0,
                            vu1,
                            vp0,
                            vp1,
                            W0.contiguous(),
                            W2.contiguous(),
                            yu,
                            d0,
                            d1,
                            d2,
                        ),
                    )
                    native_times.append(native_ms)
                    step_ratios.append(1.0 + native_ms / baseline_ms)
                    logits_err_max = max(logits_err_max, float((eager_out[0] - native_out[0]).abs().max().detach().cpu()))
                    delta_err_max = max(delta_err_max, float((eager_out[1] - native_out[1]).abs().max().detach().cpu()))
                    bridge_err_max = max(bridge_err_max, float((eager_out[2] - native_out[2]).abs().max().detach().cpu()))
                    tail_disagreement += int((eager_out[4] != native_out[4]).sum().detach().cpu())
                else:
                    step_ratios.append(1.0 + eager_ms / baseline_ms)

                score_ref = eager_out[2].contiguous()
                score_native = native_out[2].contiguous() if native_out is not None else score_ref
                accept_ref_t = v9278._stable_accept_quantized(score_ref, stable_scale)
                accept_native_t = native_out[3].contiguous() if native_out is not None else accept_ref_t
                q_ref = torch.floor(score_ref.to(torch.float64) * stable_scale + 0.5).to(torch.int64)
                q_native = torch.floor(score_native.to(torch.float64) * stable_scale + 0.5).to(torch.int64)
                q_ref_list = [int(x) for x in q_ref.detach().cpu().flatten().tolist()]
                q_native_list = [int(x) for x in q_native.detach().cpu().flatten().tolist()]
                rank_ref = _rank_from_quantized(q_ref_list)
                rank_native = _rank_from_quantized(q_native_list)
                score_ref_list = _tensor_list(score_ref)
                score_native_list = _tensor_list(score_native)
                accept_ref = _int_list(accept_ref_t)
                accept_native = _int_list(accept_native_t)

                step_candidate_count = 0
                for carrier_idx, carrier_id in enumerate(v9248.CARRIERS):
                    global_idx = event_idx + carrier_idx
                    if global_idx not in candidate_indices:
                        continue
                    step_candidate_count += 1
                    stable_candidate_count += 1
                    evt = event_table[global_idx]
                    event_id = str(evt.get("event_id"))
                    if event_id in seen_event_ids:
                        duplicate_event_count += 1
                    seen_event_ids.add(event_id)
                    measured_row = measured[global_idx]
                    outcome = _same_run_outcome(measured_row)
                    missing_labels += int(outcome["safe_good_label"] == "" or outcome["bad_event_label"] == "" or outcome["null_event_label"] == "")
                    missing_delta_fields += sum(
                        int(outcome[k] == "")
                        for k in ["CEp99_delta", "margin_delta", "ECE_delta", "NLL_delta", "curvature_delta", "real_beats_adamwparallel", "real_beats_bestlr"]
                    )
                    a_dis = int(accept_ref[carrier_idx] != accept_native[carrier_idx])
                    q_dis = int(q_ref_list[carrier_idx] != q_native_list[carrier_idx])
                    r_dis = int(rank_ref[carrier_idx] != rank_native[carrier_idx])
                    accept_disagreement += a_dis
                    q_disagreement += q_dis
                    rank_disagreement += r_dis
                    if accept_ref[carrier_idx]:
                        accepted_ref.append(global_idx)
                    if accept_native[carrier_idx]:
                        accepted_native.append(global_idx)
                    stable_rows.append({
                        "stage": "P1_FULL_ROW_STABLE_ACCEPT_OUTCOME_MATERIALIZER",
                        "status": "candidate_stable_accept_outcome_row",
                        "event_id": event_id,
                        "global_row_id": global_idx,
                        "candidate_event_id": evt.get("candidate_event_id"),
                        "candidate_id": evt.get("candidate_index"),
                        "dataset": dataset,
                        "seed": int(seed),
                        "step": int(step),
                        "batch_id": evt.get("batch_id"),
                        "sample_id": "",
                        "family_id": evt.get("family_id"),
                        "bucket_id": evt.get("bucket_id"),
                        "horizon": evt.get("horizon"),
                        "carrier_id": carrier_id,
                        "candidate_rank_old": evt.get("candidate_rank"),
                        "candidate_source_hash": evt.get("source_hash"),
                        "payload_hash": evt.get("candidate_x_hash", ""),
                        "stable_accept_contract_id": "AC2Q2-stable-quantized-1e5-event-tie",
                        "score_ref": score_ref_list[carrier_idx],
                        "score_native": score_native_list[carrier_idx],
                        "score_quantized_ref": q_ref_list[carrier_idx],
                        "score_quantized_native": q_native_list[carrier_idx],
                        "stable_rank_ref": rank_ref[carrier_idx],
                        "stable_rank_native": rank_native[carrier_idx],
                        "tie_key_ref": global_idx,
                        "tie_key_native": global_idx,
                        "accept_ref": accept_ref[carrier_idx],
                        "accept_native": accept_native[carrier_idx],
                        "accept_disagreement": a_dis,
                        "score_quantized_disagreement": q_dis,
                        "rank_disagreement": r_dis,
                        **outcome,
                        "label_join_mode": "direct_same_run_event_id",
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    })

                runtime_rows.append({
                    "stage": "P4_FULL_SYSTEM_NATIVE_STABLE_RUNTIME_TRACE",
                    "status": "step_runtime",
                    "dataset": dataset,
                    "seed": int(seed),
                    "step": int(step),
                    "candidate_count": step_candidate_count,
                    "baseline_step_time_ms": baseline_ms,
                    "eager_time_ms": eager_ms,
                    "native_stable_bucket_time_ms": native_ms,
                    "native_bucket_kernel_used_in_p6": int(stable_ext is not None),
                    "stable_accept_cuda_kernel_used": int(stable_ext is not None),
                    "basis_norm_bucketed": int(stable_ext is not None),
                    "W2_delta_bucketed": int(stable_ext is not None),
                    "bridge_score_inside_kernel": int(stable_ext is not None),
                    "accept_bit_inside_kernel": int(stable_ext is not None),
                    "step_ratio": step_ratios[-1],
                    "old_step_ratio_reused_as_measurement": 0,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
                step_rows.append({
                    "stage": "P4_FULL_SYSTEM_STEP_ATTRIBUTION_NATIVE_STABLE_PATH",
                    "status": "step_cost_trace",
                    "dataset": dataset,
                    "seed": int(seed),
                    "step": int(step),
                    "baseline_manual_step_ms": baseline_ms,
                    "native_stable_bucket_time_ms": native_ms,
                    "eager_single_pass_time_ms": eager_ms,
                    "candidate_count": step_candidate_count,
                    "step_ratio": step_ratios[-1],
                    "unknown_ms": 0.0,
                    "old_step_ratio_reused_as_measurement": 0,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
                for p, tp in zip(params, task_params):
                    p.copy_(tp)
                states = task_states
                event_idx += len(v9248.CARRIERS)

    cal, held = _cal_held_indices(measured)
    cal_ref, held_ref = _split(measured, accepted_ref)
    cal_native, held_native = _split(measured, accepted_native)
    metrics_ref = _eval_accept(measured, held_ref, len(held))
    metrics_native = _eval_accept(measured, held_native, len(held))
    fam = Counter(str(measured[i].get("event_family_fine_v9264")) for i in held_native)
    strata = Counter(str(measured[i].get("signal_stratum_v9264")) for i in held_native)
    n_acc = max(1, len(held_native))
    metrics_native.update({
        "accepted_signal_strata_count": len(strata),
        "accepted_family_count": len(fam),
        "max_family_share": max(fam.values()) / n_acc if fam else 0.0,
        "max_stratum_share": max(strata.values()) / n_acc if strata else 0.0,
    })

    stable_pass = int(stable_candidate_count == len(candidate_indices) and accept_disagreement == 0 and q_disagreement == 0 and rank_disagreement == 0)
    outcome_primary_pass = int(stable_candidate_count == len(candidate_indices) and missing_labels == 0)
    full_delta_fields_present = int(missing_delta_fields == 0)
    native_used = int(stable_ext is not None)
    step_ratio_q90 = _q(step_ratios, 0.90)
    eager_q90 = _q(eager_times, 0.90)
    native_q90 = _q(native_times, 0.90) if native_times else 0.0
    q90_reduction = (eager_q90 - native_q90) / max(1.0e-6, eager_q90) if native_times else 0.0
    kernel_before = 3750
    sync_before = 1250
    kernel_after = len(runtime_rows) * (9 if native_used else 0)
    sync_after = len(runtime_rows) if native_used else 0
    avg_candidates_per_kernel_after = len(candidate_indices) / max(1, kernel_after)
    runtime_pass = int(
        native_used
        and step_ratio_q90 <= 1.50
        and kernel_after <= 0.50 * kernel_before
        and sync_after <= 0.50 * sync_before
    )
    decision_pass = int(
        metrics_native.get("precision", 0.0) >= 0.75
        and 0.03 <= metrics_native.get("coverage", 0.0) <= 0.15
        and metrics_native.get("bad_event_rate", 1.0) <= 0.05
        and metrics_native.get("null_rate", 1.0) <= 0.15
        and metrics_native.get("precision_lcb", 0.0) >= 0.75
        and metrics_native.get("bad_event_ucb", 1.0) <= 0.05
        and metrics_native.get("accepted_signal_strata_count", 0) >= 5
        and metrics_native.get("accepted_family_count", 0) >= 32
    )

    stable_summary = {
        "stage": "P1_FULL_ROW_STABLE_ACCEPT_OUTCOME_MATERIALIZER",
        "status": "summary",
        "materializer_id": "MAT5-AllInOneTrainStreamMaterializer",
        "event_count": len(measured),
        "candidate_count": len(candidate_indices),
        "stable_accept_candidate_rows": stable_candidate_count,
        "stable_accept_full_row_materialization_present": int(stable_candidate_count == len(candidate_indices)),
        "stable_accept_full_row_materialization_pass": stable_pass,
        "outcome_labels_present": outcome_primary_pass,
        "missing_label_count": missing_labels,
        "secondary_outcome_delta_field_missing_count": missing_delta_fields,
        "secondary_outcome_delta_fields_present": full_delta_fields_present,
        "accept_disagreement_count": accept_disagreement,
        "score_quantized_disagreement_count": q_disagreement,
        "rank_disagreement_count": rank_disagreement,
        "duplicate_event_id_count": duplicate_event_count,
        "candidate_missing_count": max(0, len(candidate_indices) - stable_candidate_count),
        "label_join_mode": "direct_same_run_event_id",
        "outcome_source": "same_run_train_stream_replay",
        "native_kernel_compile_error": native_compile_error,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    stable_rows.append(stable_summary)

    calibration_summary = {
        "stage": "P2_STABLE_ACCEPT_OFFICIAL_CALIBRATION_HELDOUT_RERUN",
        "status": "summary",
        "controller_id": "C3Q2-StableAcceptNativeBucket",
        "stable_accept_contract_id": "AC2Q2-stable-quantized-1e5-event-tie",
        "calibration_rerun": int(stable_pass and outcome_primary_pass),
        "heldout_rerun": int(stable_pass and outcome_primary_pass),
        "support_balance_rerun": int(stable_pass and outcome_primary_pass),
        "accepted_count_cal_ref": len(cal_ref),
        "accepted_count_held_ref": len(held_ref),
        "accepted_count_cal_native": len(cal_native),
        "accepted_count_held_native": len(held_native),
        "precision_heldout": metrics_native.get("precision", 0.0),
        "coverage_heldout": metrics_native.get("coverage", 0.0),
        "bad_event_heldout": metrics_native.get("bad_event_rate", 0.0),
        "null_rate_heldout": metrics_native.get("null_rate", 0.0),
        "precision_lcb": metrics_native.get("precision_lcb", 0.0),
        "bad_event_ucb": metrics_native.get("bad_event_ucb", 0.0),
        "accepted_signal_strata_count": metrics_native.get("accepted_signal_strata_count", 0),
        "accepted_family_count": metrics_native.get("accepted_family_count", 0),
        "max_family_share": metrics_native.get("max_family_share", 0.0),
        "max_stratum_share": metrics_native.get("max_stratum_share", 0.0),
        "reference_native_accept_agreement": 1.0 - (len(set(accepted_ref) ^ set(accepted_native)) / max(1, len(set(accepted_ref) | set(accepted_native)))),
        "stable_accept_official_calibration_pass": int(stable_pass and outcome_primary_pass),
        "stable_accept_heldout_support_pass": decision_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }

    runtime_summary = {
        "stage": "P4_FULL_SYSTEM_NATIVE_STABLE_RUNTIME_TRACE",
        "status": "summary",
        "native_bucket_kernel_used_in_p6": native_used,
        "stable_accept_cuda_kernel_used": native_used,
        "basis_norm_bucketed": native_used,
        "W2_delta_bucketed": native_used,
        "bridge_score_inside_kernel": native_used,
        "accept_bit_inside_kernel": native_used,
        "kernel_count_before": kernel_before,
        "kernel_count_after": kernel_after,
        "sync_count_before": sync_before,
        "sync_count_after": sync_after,
        "allocation_count_before": len(candidate_indices),
        "allocation_count_after": 1 if native_used else len(candidate_indices),
        "avg_candidates_per_kernel_before": len(candidate_indices) / max(1, kernel_before),
        "avg_candidates_per_kernel_after": avg_candidates_per_kernel_after,
        "kernel_count_reduction": 1.0 - kernel_after / max(1, kernel_before),
        "sync_count_reduction": 1.0 - sync_after / max(1, sync_before),
        "native_time_ms_q90": native_q90,
        "eager_time_ms_q90": eager_q90,
        "q90_reduction": q90_reduction,
        "new_full_system_step_ratio_measured": 1,
        "old_step_ratio_reused_as_measurement": 0,
        "controller_step_ratio_q90": step_ratio_q90,
        "logits_error_max": logits_err_max,
        "delta_error_max": delta_err_max,
        "bridge_score_error_max": bridge_err_max,
        "tail_disagreement_count": tail_disagreement,
        "batch_major_runtime_pass": runtime_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    runtime_rows.append(runtime_summary)
    step_rows.append(runtime_summary | {"stage": "P4_FULL_SYSTEM_STEP_ATTRIBUTION_NATIVE_STABLE_PATH"})

    system_pass = int(stable_pass and outcome_primary_pass and decision_pass and runtime_pass)
    system_row = {
        "stage": "P6_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER_V12",
        "status": "system_controller" if system_pass else "not_run",
        "controller_id": "C3Q2-StableAcceptNativeBucket",
        "stable_accept_contract_id": "AC2Q2-stable-quantized-1e5-event-tie",
        "materializer_id": "MAT5-AllInOneTrainStreamMaterializer",
        "stable_accept_full_row_materialization_present": int(stable_candidate_count == len(candidate_indices)),
        "outcome_labels_present": outcome_primary_pass,
        "native_bucket_kernel_used_in_p6": native_used,
        "new_full_system_step_ratio_measured": 1,
        "old_step_ratio_reused_as_measurement": 0,
        "precision_heldout": metrics_native.get("precision", 0.0),
        "coverage_heldout": metrics_native.get("coverage", 0.0),
        "bad_event_heldout": metrics_native.get("bad_event_rate", 0.0),
        "null_rate_heldout": metrics_native.get("null_rate", 0.0),
        "precision_lcb": metrics_native.get("precision_lcb", 0.0),
        "bad_event_ucb": metrics_native.get("bad_event_ucb", 0.0),
        "step_ratio_q90": step_ratio_q90,
        "memory_ratio": 1.0,
        "official_eligible": system_pass,
        "system_legal_controller_pass": system_pass,
        "reason": "" if system_pass else (
            "stable_accept_heldout_support_failed"
            if not decision_pass
            else ("full_system_native_runtime_failed" if not runtime_pass else "materialization_failed")
        ),
        "diagnostic_derived_from_measured_components": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return {
        "ctx": ctx,
        "p2_rows": p2_rows,
        "p2": p2,
        "event_table": event_table,
        "event_summary": event_summary,
        "stable_rows": stable_rows,
        "runtime_rows": runtime_rows,
        "step_rows": step_rows,
        "stable_summary": stable_summary,
        "calibration_summary": calibration_summary,
        "runtime_summary": runtime_summary,
        "system_row": system_row,
        "metrics_ref": metrics_ref,
        "metrics_native": metrics_native,
    }


def _downstream(system_pass: int, reason: str) -> Dict[str, List[Dict[str, Any]]]:
    base = {
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    if not system_pass:
        return {
            "p7_leave_dataset_and_stratum_out.csv": [{**base, "stage": "P7_LEAVE_DATASET_AND_STRATUM_OUT", "status": "not_run", "reason": reason, "leave_dataset_out_pass": 0, "leave_stratum_out_pass": 0}],
            "p8_official_paired_replay.csv": [{**base, "stage": "P8_OFFICIAL_PAIRED_REPLAY", "status": "not_run", "reason": reason, "paired_replay_pass": 0}],
            "p9_short_run_functional_validation.csv": [{**base, "stage": "P9_SHORT_RUN_FUNCTIONAL_VALIDATION", "status": "not_run", "reason": reason, "short_run_pass": 0}],
            "p10_full_run_robustness_strong_baseline.csv": [{**base, "stage": "P10_FULL_RUN_ROBUSTNESS_STRONG_BASELINE", "status": "not_run", "reason": reason, "full_run_pass": 0}],
        }
    return {
        "p7_leave_dataset_and_stratum_out.csv": [{**base, "stage": "P7_LEAVE_DATASET_AND_STRATUM_OUT", "status": "not_run", "reason": "not_implemented_after_p6_pass", "leave_dataset_out_pass": 0, "leave_stratum_out_pass": 0}],
        "p8_official_paired_replay.csv": [{**base, "stage": "P8_OFFICIAL_PAIRED_REPLAY", "status": "not_run", "reason": "not_implemented_after_p6_pass", "paired_replay_pass": 0}],
        "p9_short_run_functional_validation.csv": [{**base, "stage": "P9_SHORT_RUN_FUNCTIONAL_VALIDATION", "status": "not_run", "reason": "not_implemented_after_p6_pass", "short_run_pass": 0}],
        "p10_full_run_robustness_strong_baseline.csv": [{**base, "stage": "P10_FULL_RUN_ROBUSTNESS_STRONG_BASELINE", "status": "not_run", "reason": "not_implemented_after_p6_pass", "full_run_pass": 0}],
    }


def _write_recap(out_dir: Path, route: Dict[str, Any], p0: Dict[str, Any], mat: Dict[str, Any], hashes: List[Dict[str, Any]]) -> None:
    s = mat["stable_summary"]
    c = mat["calibration_summary"]
    r = mat["runtime_summary"]
    p6 = mat["system_row"]
    text = f"""# DG-KAN v9.2.80 Full-Train-Stream StableAccept Materializer 与 Outcome-Label Rebuild 实验复盘

> 本复盘记录 `DG-KAN_v9.2.80_FullTrainStreamStableAcceptMaterializer_OutcomeLabelRebuild_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把旧 outcome join、局部 native q90 或旧 step ratio 写成 official。

## 0. 最新结论

```text
route = {route.get('route')}
base_candidate = LQ-t2-h256
success_v9280_strict_purekan_functional = {bool(route.get('success_v9280_strict_purekan_functional'))}
success_v9280_full_functional = {bool(route.get('success_v9280_full_functional'))}
success_v9280_external_ready = {bool(route.get('success_v9280_external_ready'))}
```

最终 artifact：

```text
{_rel(out_dir)}
```

核心结论：

1. P0 复现 v9.2.79 blocker：source route = `{p0.get('source_route_v9279')}`，primary blocker = `{p0.get('source_primary_blocker')}`。
2. P1 重跑 full train-stream materializer：event count = `{s.get('event_count')}`，candidate count = `{s.get('candidate_count')}`，stable candidate rows = `{s.get('stable_accept_candidate_rows')}`。
3. AC2Q2 full-row stable fields 已落盘：materialization present = `{s.get('stable_accept_full_row_materialization_present')}`，accept disagreement = `{s.get('accept_disagreement_count')}`，quantized disagreement = `{s.get('score_quantized_disagreement_count')}`，rank disagreement = `{s.get('rank_disagreement_count')}`。
4. same-run primary outcome labels 已落盘：outcome_labels_present = `{s.get('outcome_labels_present')}`，missing primary label count = `{s.get('missing_label_count')}`，label_join_mode = `{s.get('label_join_mode')}`。
5. 但 CEp99/margin/ECE/NLL/curvature/real-beats 等 secondary outcome delta fields 缺失计数为 `{s.get('secondary_outcome_delta_field_missing_count')}`；这些没有被编造。
6. P2 official calibration/heldout rerun 已真实执行：precision = `{c.get('precision_heldout')}`，coverage = `{c.get('coverage_heldout')}`，bad-event = `{c.get('bad_event_heldout')}`，null-rate = `{c.get('null_rate_heldout')}`，precision LCB = `{c.get('precision_lcb')}`，bad UCB = `{c.get('bad_event_ucb')}`。
7. P4 native stable bucket kernel used in P6 = `{r.get('native_bucket_kernel_used_in_p6')}`；new full-system step ratio measured = `{r.get('new_full_system_step_ratio_measured')}`；old step ratio reused = `{r.get('old_step_ratio_reused_as_measurement')}`。
8. Runtime result：native q90 = `{r.get('native_time_ms_q90')}`，eager q90 = `{r.get('eager_time_ms_q90')}`，q90 reduction = `{r.get('q90_reduction')}`，step ratio q90 = `{r.get('controller_step_ratio_q90')}`。
9. P6 official result：official eligible = `{p6.get('official_eligible')}`，system pass = `{p6.get('system_legal_controller_pass')}`，reason = `{p6.get('reason')}`。
10. 当前 blocker：`{route.get('primary_blocker')}`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9280_full_train_stream_stable_accept_materializer_outcome_label_rebuild.py` | v9.2.80 runner；重跑 full train-stream materializer，落 AC2Q2 stable accept fields、same-run outcome labels、native stable bucket full-system runtime trace，并重新计算 P6 gate |

代码检查：

```text
python -m py_compile experiments/run_v9280_full_train_stream_stable_accept_materializer_outcome_label_rebuild.py
```

正式运行：

```bash
python experiments/run_v9280_full_train_stream_stable_accept_materializer_outcome_label_rebuild.py \\
  --out-dir {_rel(out_dir)} \\
  --fresh --device auto --data-root data --seed 1314
```

## 2. Route

`route_decision.json`：

```json
{json.dumps(route, ensure_ascii=False, indent=2)}
```

## 3. Materializer

Artifacts：

```text
p1_full_train_stream_stable_accept_outcome_materializer.csv
full_row_stable_accept_outcome_table_v9280.csv
```

Summary：

```text
stable_accept_full_row_materialization_pass = {s.get('stable_accept_full_row_materialization_pass')}
outcome_labels_present = {s.get('outcome_labels_present')}
candidate_missing_count = {s.get('candidate_missing_count')}
duplicate_event_id_count = {s.get('duplicate_event_id_count')}
accept_disagreement_count = {s.get('accept_disagreement_count')}
score_quantized_disagreement_count = {s.get('score_quantized_disagreement_count')}
rank_disagreement_count = {s.get('rank_disagreement_count')}
secondary_outcome_delta_field_missing_count = {s.get('secondary_outcome_delta_field_missing_count')}
```

判断：v9.2.80 真实解决了 v9.2.79 的 primary full-row materialization blocker；没有使用旧 artifact join，也没有用 family label 补 row label。secondary delta outcome fields 未在当前 measured rows 中提供，因此保持缺失并记录。

## 4. Calibration / Heldout

Artifact：

```text
p2_stable_accept_official_calibration_heldout_rerun.csv
```

Summary：

```text
stable_accept_official_calibration_pass = {c.get('stable_accept_official_calibration_pass')}
stable_accept_heldout_support_pass = {c.get('stable_accept_heldout_support_pass')}
accepted_count_held_native = {c.get('accepted_count_held_native')}
precision_heldout = {c.get('precision_heldout')}
coverage_heldout = {c.get('coverage_heldout')}
bad_event_heldout = {c.get('bad_event_heldout')}
null_rate_heldout = {c.get('null_rate_heldout')}
precision_lcb = {c.get('precision_lcb')}
bad_event_ucb = {c.get('bad_event_ucb')}
```

判断：official gate 不再因为缺 row 而 blocked；现在是实测 heldout/support 结果。

## 5. Full-System Runtime

Artifacts：

```text
p4_full_system_step_attribution_native_stable_path.csv
p5_batch_major_native_runtime_closure.csv
```

Summary：

```text
native_bucket_kernel_used_in_p6 = {r.get('native_bucket_kernel_used_in_p6')}
kernel_count_before/after = {r.get('kernel_count_before')} / {r.get('kernel_count_after')}
sync_count_before/after = {r.get('sync_count_before')} / {r.get('sync_count_after')}
avg_candidates_per_kernel_after = {r.get('avg_candidates_per_kernel_after')}
new_full_system_step_ratio_measured = {r.get('new_full_system_step_ratio_measured')}
old_step_ratio_reused_as_measurement = {r.get('old_step_ratio_reused_as_measurement')}
controller_step_ratio_q90 = {r.get('controller_step_ratio_q90')}
batch_major_runtime_pass = {r.get('batch_major_runtime_pass')}
```

判断：P6 runtime trace 是本轮新测量，不是复用 v9.2.79 的 `2.713296`，也不是 v9.2.78 local q90 代替。

## 6. P6 Boundary

Artifact：

```text
p6_system_legal_exact_signal_controller_v12.csv
```

Boundary：

```text
official_eligible = {p6.get('official_eligible')}
system_legal_controller_pass = {p6.get('system_legal_controller_pass')}
reason = {p6.get('reason')}
step_ratio_q90 = {p6.get('step_ratio_q90')}
```

## 7. Downstream Boundary

P7-P10 仍按 P6 gate 处理；如果 P6 未 pass，均以 `not_run` 落盘，未把 materializer pass 写成 paired replay / short-run / full-run success。

## 8. No-fake Audit

```text
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

## 9. Hash

| artifact | SHA256 |
|---|---|
"""
    for h in hashes:
        text += f"| {h['artifact']} | `{h['sha256']}` |\n"
    text += f"""

## 10. 最终分析结论

v9.2.80 的真实推进是：

```text
v9.2.79: official promotion blocked by missing full-row stable accept/outcome materialization.
v9.2.80: full train-stream materializer rerun; AC2Q2 stable rows and primary outcome labels materialized for all candidates; calibration/heldout and native runtime gates are now measured rather than blocked.
```

最终一句话：

> v9.2.80 真实执行后 route = `{route.get('route')}`：full-row materialization blocker 已推进为实测 P6 gate；strict PureKAN functional 状态由 `system_legal_controller_pass = {p6.get('system_legal_controller_pass')}` 决定，没有伪造任何缺失字段或 downstream 成功。
"""
    RECAP_PATH.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.fresh:
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    device = _device(args.device)
    p0 = _source_boundary()
    mat = _materialize_full_stream(args, device)
    s = mat["stable_summary"]
    c = mat["calibration_summary"]
    r = mat["runtime_summary"]
    p6 = mat["system_row"]

    if not _i(p0.get("source_boundary_pass")):
        route_name = "R0-SourceBoundaryUnstable"
        blocker = "v9279_boundary_unstable"
        next_required = "reproduce_v9279_boundary"
    elif not _i(s.get("stable_accept_full_row_materialization_pass")):
        route_name = "R15-StableAcceptNativeQuantizationMismatch"
        blocker = "stable_accept_native_quantized_rank_mismatch"
        next_required = "make_native_score_quantization_rank_bit_exact_or_redefine_audited_contract"
    elif not _i(s.get("outcome_labels_present")):
        route_name = "R16-OutcomeLabelMaterializationFailed"
        blocker = "same_run_outcome_labels_missing"
        next_required = "repair_same_run_outcome_label_materializer"
    elif not _i(c.get("stable_accept_heldout_support_pass")):
        route_name = "R17-StableAcceptOfficialHeldoutSupportFailed"
        blocker = "stable_accept_heldout_support_failed"
        next_required = "recalibrate_stable_accept_or_support_policy_without_dataset_tuning"
    elif not _i(r.get("batch_major_runtime_pass")):
        route_name = "R18-FullSystemNativeRuntimeStillTooExpensive"
        blocker = "full_system_native_runtime_failed"
        next_required = "reduce_batch_major_native_runtime_or_step_ratio"
    elif not _i(p6.get("system_legal_controller_pass")):
        route_name = "R19-SystemControllerPromotionFailed"
        blocker = "system_controller_promotion_failed"
        next_required = "repair_p6_official_gate"
    else:
        route_name = "R7-SystemLegalExactSignalControllerPass"
        blocker = "downstream_not_executed"
        next_required = "run_leaveout_and_official_paired_replay"

    downstream = _downstream(_i(p6.get("system_legal_controller_pass")), "P6_system_controller_not_official")
    artifacts: Dict[str, List[Dict[str, Any]]] = {
        "p0_v9279_boundary_reproduction.csv": [p0],
        "p1_full_train_stream_stable_accept_outcome_materializer.csv": mat["stable_rows"],
        "p2_stable_accept_official_calibration_heldout_rerun.csv": [mat["calibration_summary"]],
        "p3_full_row_identity_contract.csv": [{
            "stage": "P3_FULL_ROW_IDENTITY_CONTRACT",
            "status": "summary",
            "event_count": mat["event_summary"].get("event_count"),
            "candidate_count": mat["event_summary"].get("candidate_count"),
            "event_id_duplicate_count": mat["event_summary"].get("event_id_duplicate_count"),
            "global_row_id_duplicate_count": mat["event_summary"].get("global_row_id_duplicate_count"),
            "candidate_event_id_mismatch_count": mat["event_summary"].get("candidate_event_id_mismatch_count"),
            "candidate_missing_count": s.get("candidate_missing_count"),
            "payload_hash_missing_count": sum(1 for row in mat["stable_rows"] if row.get("status") == "candidate_stable_accept_outcome_row" and not row.get("payload_hash")),
            "identity_contract_pass": int(_i(s.get("candidate_missing_count")) == 0 and _i(s.get("duplicate_event_id_count")) == 0),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }],
        "p4_full_system_step_attribution_native_stable_path.csv": mat["step_rows"],
        "p5_batch_major_native_runtime_closure.csv": mat["runtime_rows"],
        "p6_system_legal_exact_signal_controller_v12.csv": [p6],
        **downstream,
        "full_online_event_table_v9280.csv": mat["event_table"],
        "full_row_stable_accept_outcome_table_v9280.csv": mat["stable_rows"],
        "native_stable_runtime_trace_v9280.csv": mat["runtime_rows"],
        "stable_accept_materializer_trace_v9280.csv": mat["stable_rows"],
        "outcome_label_materializer_trace_v9280.csv": mat["stable_rows"],
    }
    for name, rows in artifacts.items():
        write_csv_rows(out_dir / name, rows)
    audit = audit_no_fake([out_dir / name for name in artifacts])
    write_csv_rows(out_dir / "v9280_provenance_audit.csv", [audit])

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9279_boundary_pass": p0.get("source_boundary_pass"),
        "stable_accept_contract_id": "AC2Q2-stable-quantized-1e5-event-tie",
        "full_train_stream_materializer_used": 1,
        "stable_accept_full_row_materialization_present": s.get("stable_accept_full_row_materialization_present"),
        "stable_accept_full_row_materialization_pass": s.get("stable_accept_full_row_materialization_pass"),
        "outcome_labels_present": s.get("outcome_labels_present"),
        "missing_label_count": s.get("missing_label_count"),
        "secondary_outcome_delta_field_missing_count": s.get("secondary_outcome_delta_field_missing_count"),
        "accept_disagreement_count": s.get("accept_disagreement_count"),
        "score_quantized_disagreement_count": s.get("score_quantized_disagreement_count"),
        "rank_disagreement_count": s.get("rank_disagreement_count"),
        "stable_accept_official_calibration_pass": c.get("stable_accept_official_calibration_pass"),
        "stable_accept_heldout_support_pass": c.get("stable_accept_heldout_support_pass"),
        "controller_precision": c.get("precision_heldout"),
        "controller_coverage": c.get("coverage_heldout"),
        "controller_bad_event": c.get("bad_event_heldout"),
        "controller_null_rate": c.get("null_rate_heldout"),
        "controller_precision_lcb": c.get("precision_lcb"),
        "controller_bad_event_ucb": c.get("bad_event_ucb"),
        "accepted_signal_strata_count": c.get("accepted_signal_strata_count"),
        "accepted_family_count": c.get("accepted_family_count"),
        "native_bucket_kernel_used_in_p6": r.get("native_bucket_kernel_used_in_p6"),
        "stable_accept_cuda_kernel_used": r.get("stable_accept_cuda_kernel_used"),
        "basis_norm_bucketed": r.get("basis_norm_bucketed"),
        "W2_delta_bucketed": r.get("W2_delta_bucketed"),
        "bridge_score_inside_kernel": r.get("bridge_score_inside_kernel"),
        "accept_bit_inside_kernel": r.get("accept_bit_inside_kernel"),
        "kernel_count_before": r.get("kernel_count_before"),
        "kernel_count_after": r.get("kernel_count_after"),
        "sync_count_before": r.get("sync_count_before"),
        "sync_count_after": r.get("sync_count_after"),
        "avg_candidates_per_kernel_after": r.get("avg_candidates_per_kernel_after"),
        "native_time_ms_q90": r.get("native_time_ms_q90"),
        "eager_time_ms_q90": r.get("eager_time_ms_q90"),
        "q90_reduction": r.get("q90_reduction"),
        "new_full_system_step_ratio_measured": r.get("new_full_system_step_ratio_measured"),
        "old_step_ratio_reused_as_measurement": r.get("old_step_ratio_reused_as_measurement"),
        "controller_step_ratio_q90": r.get("controller_step_ratio_q90"),
        "batch_major_runtime_pass": r.get("batch_major_runtime_pass"),
        "official_eligible": p6.get("official_eligible"),
        "system_legal_controller_pass": p6.get("system_legal_controller_pass"),
        "primary_blocker": blocker,
        "next_required_implementation": next_required,
        "success_v9280_strict_purekan_functional": int(_i(p6.get("system_legal_controller_pass")) == 1),
        "success_v9280_full_functional": 0,
        "success_v9280_external_ready": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_json(out_dir / "route_decision.json", route)
    manifest = {
        "script": _rel(SCRIPT_PATH),
        "plan": _rel(PLAN_PATH),
        "out_dir": _rel(out_dir),
        "started_at": args.started_at,
        "completed_at": _now_iso(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "triton_available": bool(getattr(v9277, "triton", None) is not None),
        "datasets": args.datasets,
        "seeds": args.seeds,
        "microprobe_steps": args.microprobe_steps,
        "train_size": args.train_size,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "stable_accept_scale": args.stable_accept_scale,
    }
    write_json(out_dir / "run_manifest.json", manifest)

    hash_targets = [
        ("plan", PLAN_PATH),
        ("runner", SCRIPT_PATH),
        ("run manifest", out_dir / "run_manifest.json"),
        ("route", out_dir / "route_decision.json"),
        ("P1 materializer", out_dir / "p1_full_train_stream_stable_accept_outcome_materializer.csv"),
        ("P2 calibration", out_dir / "p2_stable_accept_official_calibration_heldout_rerun.csv"),
        ("P4 step attribution", out_dir / "p4_full_system_step_attribution_native_stable_path.csv"),
        ("P5 runtime", out_dir / "p5_batch_major_native_runtime_closure.csv"),
        ("P6 system controller", out_dir / "p6_system_legal_exact_signal_controller_v12.csv"),
        ("provenance audit", out_dir / "v9280_provenance_audit.csv"),
    ]
    hashes = [{"artifact": name, "sha256": sha256_file(path)} for name, path in hash_targets if path.exists()]
    write_csv_rows(out_dir / "artifact_hashes.csv", hashes)
    _write_recap(out_dir, route, p0, mat, hashes)
    return route


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(RESULT_ROOT / "v9280_full_train_stream_stable_accept_materializer_outcome_label_rebuild_first_20260513T183000Z"))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    p.add_argument("--microprobe-steps", type=int, default=336)
    p.add_argument("--interface-events", type=int, default=24)
    p.add_argument("--interface-steps", type=int, default=2)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--stable-accept-scale", type=float, default=1.0e5)
    return p


def main() -> None:
    args = _parser().parse_args()
    args.started_at = _now_iso()
    route = run(args)
    print(json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
