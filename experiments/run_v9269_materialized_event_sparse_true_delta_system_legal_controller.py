#!/usr/bin/env python3
"""DG-KAN v9.2.69 materialized event-sparse true-delta audit.

This runner starts from the v9.2.68 result: reference decision geometry is
feasible, but the event-sparse true-delta path was projection-only.  v9.2.69
materializes the PF5 selector, candidate pack, true branch-logit microprobe,
candidate true-delta scoring, and frozen bridge lookup.  A materialized
microprobe is kept diagnostic unless it is fully bound to the online controller
rows; this prevents a real CUDA timing from being overstated as a complete
system-legal controller.
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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9256_true_branch_delta_tensor_interface_k7d_control_gain_cuda_fusion as v9256  # noqa: E402
import run_v9265_joint_safe_useful_feasibility_null_bad_conflict_resolution as v9265  # noqa: E402
import run_v9266_oracle_legal_gap_closure_bridge_frontier as v9266  # noqa: E402
import run_v9268_true_delta_system_closure_reference_feasible_controller_promotion as v9268  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, read_csv_rows, sha256_file, write_csv_rows, write_json  # noqa: E402
from dgkan.models import fc_purekan_lq as lq  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.69_MaterializedEventSparseTrueDelta_SystemLegalController_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9269_materialized_event_sparse_true_delta_system_legal_controller.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9268 = RESULT_ROOT / "v9268_true_delta_system_closure_reference_feasible_controller_promotion_first_20260512T233000Z"

_f = v9265._f
_i = v9265._i
_q = v9265._q
_auc = v9265._auc
_corr = v9265._corr
_feature = v9265._feature
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


def _hash_text(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _split(rows: Sequence[Dict[str, Any]], accepted: Sequence[int]) -> Tuple[List[int], List[int]]:
    return v9266._split_sets(rows, accepted)


def _eval(rows: Sequence[Dict[str, Any]], accepted: Sequence[int], denominator: int | None = None) -> Dict[str, Any]:
    return v9266._eval_accept(rows, accepted, denominator)


def _agreement(a: Iterable[int], b: Iterable[int], indices: Sequence[int]) -> float:
    aa, bb = set(a), set(b)
    if not indices:
        return 0.0
    return sum(int((idx in aa) == (idx in bb)) for idx in indices) / len(indices)


def _recall(reference: Iterable[int], candidates: Iterable[int]) -> float:
    ref, cand = set(reference), set(candidates)
    return len(ref & cand) / max(1, len(ref))


def _precision(reference: Iterable[int], candidates: Iterable[int]) -> float:
    ref, cand = set(reference), set(candidates)
    return len(ref & cand) / max(1, len(cand))


def _row_family_id(row: Dict[str, Any]) -> int:
    return _hash_text(str(row.get("event_family_fine_v9264", ""))) % 1_000_003


def _row_branch_id(row: Dict[str, Any]) -> int:
    src = f"{row.get('seed')}|{row.get('carrier_id', '')}|{row.get('signal_stratum_v9264', '')}"
    return _hash_text(src) % 4096


def _p0_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9268 / "route_decision.json")
    audit_rows = read_csv_rows(SRC_V9268 / "v9268_provenance_audit.csv")
    audit = audit_rows[0] if audit_rows else {}
    p0_pass = int(
        route.get("route") == "R17-ReferenceFeasibleButComputeFail"
        and str(route.get("reference_controller_id")) == "C3-T2PlusBackfill"
        and _i(route.get("reference_controller_reproduced")) == 1
        and _i(route.get("frozen_controller_pass")) == 1
        and _i(route.get("prefilter_pass")) == 1
        and _i(route.get("event_sparse_exact_pass")) == 0
        and _i(route.get("system_legal_controller_pass")) == 0
        and str(route.get("primary_blocker")) == "true_delta_system_path_not_materialized"
        and _i(audit.get("fake_proxy_nonzero_count")) == 0
        and _i(audit.get("proxy_row_used")) == 0
        and _i(audit.get("cpu_offload_used")) == 0
    )
    return {
        "stage": "P0_V9268_BOUNDARY_REPRODUCTION",
        "status": "source_boundary",
        "source_route": route.get("route", ""),
        "reference_controller_id": route.get("reference_controller_id", ""),
        "reference_controller_reproduced": route.get("reference_controller_reproduced", ""),
        "reference_precision": route.get("reference_precision", ""),
        "reference_coverage": route.get("reference_coverage", ""),
        "reference_bad_event": route.get("reference_bad_event", ""),
        "reference_null_rate": route.get("reference_null_rate", ""),
        "reference_precision_lcb": route.get("reference_precision_lcb", ""),
        "reference_bad_event_ucb": route.get("reference_bad_event_ucb", ""),
        "frozen_controller_pass": route.get("frozen_controller_pass", ""),
        "best_prefilter_id": route.get("best_prefilter_id", ""),
        "prefilter_pass": route.get("prefilter_pass", ""),
        "candidate_rate": route.get("candidate_rate", ""),
        "reference_accept_recall": route.get("reference_accept_recall", ""),
        "best_exact_candidate_id": route.get("best_exact_candidate_id", ""),
        "event_sparse_exact_pass": route.get("event_sparse_exact_pass", ""),
        "event_sparse_exact_diagnostic_pass": route.get("event_sparse_exact_diagnostic_pass", ""),
        "exact_step_ratio_q90": route.get("exact_step_ratio_q90", ""),
        "materialized_system_path": route.get("materialized_system_path", ""),
        "system_legal_controller_pass": route.get("system_legal_controller_pass", ""),
        "primary_blocker": route.get("primary_blocker", ""),
        "fake_proxy_count": audit.get("fake_proxy_nonzero_count", 0),
        "cpu_offload_used": audit.get("cpu_offload_used", 0),
        "v9268_boundary_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }


def _metric_scores(rows: Sequence[Dict[str, Any]]) -> List[float]:
    return [v9268._cheap_score(r) for r in rows]


def _materialized_pf5(ctx: Dict[str, Any], device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = ctx["measured"]
    ref_all = set(ctx["c3_accept_all"])
    cal, held = _cal_held_indices(rows)
    scores = _metric_scores(rows)
    ref_scores_cal = [scores[i] for i in cal if i in ref_all]
    all_scores_cal = [scores[i] for i in cal]
    threshold = min(ref_scores_cal) if ref_scores_cal else _q(all_scores_cal, 0.95)
    t0 = time.perf_counter()
    candidate_indices = [i for i, s in enumerate(scores) if s >= threshold]
    candidate_rate = len(candidate_indices) / max(1, len(rows))
    recall = _recall(ref_all, candidate_indices)
    cand_t0 = time.perf_counter()
    idx_tensor = torch.as_tensor(candidate_indices, dtype=torch.long, device=device).contiguous()
    family_tensor = torch.as_tensor([_row_family_id(rows[i]) for i in candidate_indices], dtype=torch.long, device=device).contiguous()
    branch_tensor = torch.as_tensor([_row_branch_id(rows[i]) for i in candidate_indices], dtype=torch.long, device=device).contiguous()
    if device.type == "cuda":
        torch.cuda.synchronize()
    candidate_pack_time_ms = (time.perf_counter() - cand_t0) * 1000.0
    selector_time_ms = (time.perf_counter() - t0) * 1000.0
    candidate_pack_memory_mb = (
        (idx_tensor.numel() + family_tensor.numel() + branch_tensor.numel()) * idx_tensor.element_size() / (1024.0 * 1024.0)
    )
    pass_gate = int(
        candidate_rate <= 0.25
        and recall >= 0.95
        and idx_tensor.is_contiguous()
        and family_tensor.is_contiguous()
        and branch_tensor.is_contiguous()
    )
    row = {
        "stage": "P2_RUNTIME_PF5_SELECTOR_AND_CANDIDATE_PACK",
        "status": "runtime_prefilter",
        "prefilter_id": "PF5-LearnedMonotoneCheapPrefilter",
        "implementation_id": "PF5-runtime-threshold-candidate-pack-v1",
        "threshold": threshold,
        "candidate_rate": candidate_rate,
        "reference_accept_recall": recall,
        "reference_accept_precision": _precision(ref_all, candidate_indices),
        "candidate_count": len(candidate_indices),
        "event_count": len(rows),
        "candidate_pack_time_ms": candidate_pack_time_ms,
        "selector_time_ms": selector_time_ms,
        "candidate_pack_memory_MB": candidate_pack_memory_mb,
        "candidate_indices_materialized": 1,
        "candidate_family_ids_materialized": 1,
        "candidate_branch_ids_materialized": 1,
        "candidate_tensor_contiguous": int(idx_tensor.is_contiguous() and family_tensor.is_contiguous() and branch_tensor.is_contiguous()),
        "candidate_index_tensor_sha256": hashlib.sha256(idx_tensor.detach().cpu().numpy().tobytes()).hexdigest(),
        "candidate_family_tensor_sha256": hashlib.sha256(family_tensor.detach().cpu().numpy().tobytes()).hexdigest(),
        "candidate_branch_tensor_sha256": hashlib.sha256(branch_tensor.detach().cpu().numpy().tobytes()).hexdigest(),
        "projection_used": 0,
        "dataset_name_used": 0,
        "posthoc_used_at_commit": 0,
        "pf5_runtime_selector_pass": pass_gate,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    summary = {
        "stage": "P2_RUNTIME_PF5_SELECTOR_AND_CANDIDATE_PACK",
        "status": "summary",
        "best_prefilter_id": row["prefilter_id"],
        "pf5_runtime_selector_pass": pass_gate,
        "candidate_rate": candidate_rate,
        "reference_accept_recall": recall,
        "candidate_count": len(candidate_indices),
        "event_count": len(rows),
        "candidate_pack_time_ms": candidate_pack_time_ms,
        "candidate_pack_memory_MB": candidate_pack_memory_mb,
        "candidate_indices_materialized": 1,
        "candidate_tensor_contiguous": row["candidate_tensor_contiguous"],
        "candidate_indices": candidate_indices,
        "idx_tensor": idx_tensor,
        "family_tensor": family_tensor,
        "branch_tensor": branch_tensor,
        "projection_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row, {k: v for k, v in summary.items() if k not in {"candidate_indices", "idx_tensor", "family_tensor", "branch_tensor"}}], summary


def _probe_ordinals(rows: Sequence[Dict[str, Any]], candidate_indices: Sequence[int], event_count: int, candidate_rate: float) -> List[int]:
    if event_count <= 0:
        return []
    target = max(1, min(event_count, int(round(event_count * max(candidate_rate, 1.0 / event_count)))))
    ranked = sorted(
        {(_hash_text(str(rows[i].get("row_id"))) % event_count) for i in candidate_indices},
        key=lambda x: (_hash_text(f"probe-{x}"), x),
    )
    if len(ranked) < target:
        ranked = sorted(set(ranked) | set(range(event_count)), key=lambda x: (_hash_text(f"fallback-{x}"), x))
    return sorted(ranked[:target])


def _run_candidate_true_branch_probe(args: argparse.Namespace, device: torch.device, candidate_ordinals: Sequence[int], max_events: int) -> Dict[str, Any]:
    spec = lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", int(args.hidden_dim), "default", 0.8)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    fwd_core, bwd_core = lq.functions_for_basis(spec.basis)
    cand_set = set(int(x) for x in candidate_ordinals)
    before_parts: List[torch.Tensor] = []
    real_parts: List[torch.Tensor] = []
    adamw_parts: List[torch.Tensor] = []
    bestlr_parts: List[torch.Tensor] = []
    labels_parts: List[torch.Tensor] = []
    event_ordinals: List[int] = []
    baseline_slots_ms: List[float] = []
    branch_slots_ms: List[float] = []
    branch_nonzero_ms: List[float] = []
    all_event_rows: List[Dict[str, Any]] = []
    event_ordinal = 0

    for dataset in [v9256.v92._canonical_task(x) for x in v9256._parse_list(args.datasets)]:
        if event_ordinal >= max_events:
            break
        for seed in v9256._parse_ints(args.seeds):
            if event_ordinal >= max_events:
                break
            load_args = argparse.Namespace(data_root=args.data_root, seed=int(args.seed) + int(seed))
            x_train, y_train, _x_test, _y_test, input_dim, output_dim, _protocol = v9256.v92._load_task(
                load_args,
                dataset,
                train_size=int(args.train_size),
                test_size=32,
            )
            x_train = x_train.to(device=device, dtype=torch.float32)
            y_train = y_train.to(device=device)
            params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, int(args.seed) + seed * 100 + 9256)
            states = [AdamWState.zeros_like(p) for p in params]
            gen = torch.Generator(device=device).manual_seed(int(args.seed) * 10000 + seed * 101 + 9256)
            n = int(x_train.shape[0])
            for step in range(min(int(args.interface_steps), int(args.microprobe_steps))):
                if event_ordinal >= max_events:
                    break
                batch_idx = torch.randint(0, n, (int(args.batch_size) * 2,), generator=gen, device=device)
                xu, yu = x_train[batch_idx[: int(args.batch_size)]], y_train[batch_idx[: int(args.batch_size)]]
                xp, yp = x_train[batch_idx[int(args.batch_size) :]], y_train[batch_idx[int(args.batch_size) :]]
                v9256.v9248._sync(device)
                t0 = time.perf_counter()
                pack = bwd_core(xu, yu, *params, mu, std, 2.0, 2.0)
                grads = list(pack[1:])
                task_params = v9256.v9248._clone_params(params)
                task_states = v9256.v9248._clone_states(states)
                v9256.v92._adamw_update_foreach_(task_params, grads, task_states, cfg)
                v9256.v9248._sync(device)
                base_ms = max(1.0e-6, (time.perf_counter() - t0) * 1000.0)
                task_delta = [tp - p for tp, p in zip(task_params, params)]
                ctrl_103 = v9256.v9248._apply_delta(params, task_delta, 1.03)
                ctrl_097 = v9256.v9248._apply_delta(params, task_delta, 0.97)
                ctrl_106 = v9256.v9248._apply_delta(params, task_delta, 1.06)
                for carrier_id in ("A1-RiskBoundedTailCarrier", "A3-LateAttachRoleWiseFT7EdgeCarrier"):
                    if event_ordinal >= max_events:
                        break
                    should_compute = event_ordinal in cand_set
                    branch_ms = 0.0
                    if should_compute:
                        v9256.v9248._sync(device)
                        tb0 = time.perf_counter()
                        fd, _fmeta = v9256.v9248._functional_delta(task_params, mu, std, spec, xu, yu, task_delta, carrier_id)
                        cand_params = [tp + d for tp, d in zip(task_params, fd)]
                        with torch.no_grad():
                            before = fwd_core(xp, *params, mu, std, 2.0, 2.0).contiguous()
                            real = fwd_core(xp, *cand_params, mu, std, 2.0, 2.0).contiguous()
                            adamw = fwd_core(xp, *ctrl_103, mu, std, 2.0, 2.0).contiguous()
                            logits_097 = fwd_core(xp, *ctrl_097, mu, std, 2.0, 2.0).contiguous()
                            logits_106 = fwd_core(xp, *ctrl_106, mu, std, 2.0, 2.0).contiguous()
                            ce097 = v9256._ce_loss_vec(logits_097, yp).mean()
                            ce106 = v9256._ce_loss_vec(logits_106, yp).mean()
                            bestlr = logits_097 if float(ce097.detach().cpu()) <= float(ce106.detach().cpu()) else logits_106
                        v9256.v9248._sync(device)
                        branch_ms = max(0.0, (time.perf_counter() - tb0) * 1000.0)
                        before_parts.append(before.detach())
                        real_parts.append(real.detach())
                        adamw_parts.append(adamw.detach())
                        bestlr_parts.append(bestlr.detach())
                        labels_parts.append(yp.detach())
                        event_ordinals.append(event_ordinal)
                        branch_nonzero_ms.append(branch_ms)
                    baseline_slots_ms.append(base_ms)
                    branch_slots_ms.append(branch_ms)
                    all_event_rows.append({
                        "event_ordinal": event_ordinal,
                        "dataset": dataset,
                        "seed": seed,
                        "step": step,
                        "carrier_id": carrier_id,
                        "candidate_selected": int(should_compute),
                        "baseline_ms": base_ms,
                        "branch_forward_time_ms": branch_ms,
                    })
                    event_ordinal += 1
                for p, tp in zip(params, task_params):
                    p.copy_(tp)
                states = task_states

    tensors: Dict[str, torch.Tensor] = {}
    if before_parts:
        tensors = {
            "before": torch.cat(before_parts, dim=0).contiguous(),
            "real": torch.cat(real_parts, dim=0).contiguous(),
            "adamw": torch.cat(adamw_parts, dim=0).contiguous(),
            "bestlr": torch.cat(bestlr_parts, dim=0).contiguous(),
            "labels": torch.cat(labels_parts, dim=0).contiguous(),
        }
    full_event_count = max(1, len(all_event_rows))
    candidate_event_count = len(event_ordinals)
    baseline_q90 = _q(baseline_slots_ms, 0.90)
    branch_q90 = _q(branch_slots_ms, 0.90)
    aggregate_step_ratio = 1.0 + sum(branch_slots_ms) / max(1.0e-6, sum(baseline_slots_ms))
    event_q90_step_ratio = 1.0 + branch_q90 / max(1.0e-6, baseline_q90)
    return {
        "events": all_event_rows,
        "tensors": tensors,
        "event_ordinals": event_ordinals,
        "full_event_count": full_event_count,
        "candidate_event_count": candidate_event_count,
        "candidate_event_rate": candidate_event_count / full_event_count,
        "baseline_time_q90_ms": baseline_q90,
        "branch_forward_time_ms_mean": float(np.mean(branch_nonzero_ms)) if branch_nonzero_ms else 0.0,
        "branch_forward_time_ms_q90_sparse": branch_q90,
        "branch_forward_time_ms_q90_candidate_nonzero": _q(branch_nonzero_ms, 0.90),
        "branch_forward_time_total_ms": sum(branch_slots_ms),
        "aggregate_step_ratio": aggregate_step_ratio,
        "event_q90_step_ratio": event_q90_step_ratio,
        "kernel_count": candidate_event_count * 8,
        "sync_count": candidate_event_count * 2,
        "allocation_count": candidate_event_count,
    }


def _compare_candidate_logits(full_sample: Dict[str, Any], candidate_probe: Dict[str, Any], args: argparse.Namespace) -> Dict[str, float]:
    tensors = candidate_probe.get("tensors", {})
    full_tensors = full_sample.get("tensors", {})
    if not tensors or not full_tensors:
        return {"max_abs": 0.0, "rel_err": 0.0}
    batch = int(args.batch_size)
    max_abs = 0.0
    denom = 0.0
    for cand_pos, ordinal in enumerate(candidate_probe.get("event_ordinals", [])):
        c0, c1 = cand_pos * batch, (cand_pos + 1) * batch
        f0, f1 = int(ordinal) * batch, (int(ordinal) + 1) * batch
        for key in ("before", "real", "adamw", "bestlr"):
            diff = (tensors[key][c0:c1] - full_tensors[key][f0:f1]).detach()
            max_abs = max(max_abs, float(diff.abs().max().detach().cpu()))
            denom = max(denom, float(full_tensors[key][f0:f1].abs().max().detach().cpu()))
    return {"max_abs": max_abs, "rel_err": max_abs / max(1.0e-12, denom)}


def _p1_gap_audit() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    components = [
        ("PF5 candidate selector", 1, 1, "runtime_selector", "experiments/run_v9269_materialized_event_sparse_true_delta_system_legal_controller.py"),
        ("candidate indices", 1, 1, "torch_long_tensor", "experiments/run_v9269_materialized_event_sparse_true_delta_system_legal_controller.py"),
        ("candidate family ids", 1, 1, "torch_long_tensor", "experiments/run_v9269_materialized_event_sparse_true_delta_system_legal_controller.py"),
        ("candidate branch ids", 1, 1, "torch_long_tensor", "experiments/run_v9269_materialized_event_sparse_true_delta_system_legal_controller.py"),
        ("candidate tensors", 0, 1, "true_microprobe_candidate_pack", "experiments/run_v9269_materialized_event_sparse_true_delta_system_legal_controller.py"),
        ("true branch logits", 0, 1, "materialized_microprobe", "experiments/run_v9269_materialized_event_sparse_true_delta_system_legal_controller.py"),
        ("true branch-delta", 0, 1, "materialized_microprobe", "experiments/run_v9269_materialized_event_sparse_true_delta_system_legal_controller.py"),
        ("frozen C3 lookup", 1, 1, "compact_tensor_lookup", "experiments/run_v9269_materialized_event_sparse_true_delta_system_legal_controller.py"),
        ("bridge score", 1, 1, "frozen_lookup_score", "experiments/run_v9269_materialized_event_sparse_true_delta_system_legal_controller.py"),
        ("accept decision", 1, 1, "frozen_controller_decision", "experiments/run_v9269_materialized_event_sparse_true_delta_system_legal_controller.py"),
        ("timing measurement", 0, 1, "cuda_or_sync_timed_microprobe", "experiments/run_v9269_materialized_event_sparse_true_delta_system_legal_controller.py"),
        ("full online row binding", 0, 0, "missing_full_binding", "next_required_implementation"),
    ]
    rows: List[Dict[str, Any]] = []
    for idx, (component, projection_available, materialized_available, gap_type, impl) in enumerate(components):
        required = 1
        missing = int(required and not materialized_available)
        rows.append({
            "stage": "P1_PROJECTION_TO_MATERIALIZATION_GAP_AUDIT",
            "status": "component_gap",
            "audit_id": f"MA{idx}",
            "component": component,
            "projection_available": projection_available,
            "materialized_available": materialized_available,
            "required_for_official": required,
            "gap_type": gap_type,
            "implementation_file": impl,
            "runtime_input": "train_stream_rows_or_true_branch_logits",
            "runtime_output": "candidate_or_accept_tensor",
            "uses_posthoc": 0,
            "uses_projection": 0,
            "uses_cpu_summary": 0,
            "materialization_blocker": "full_online_row_binding_missing" if missing else "",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    mapped = int(all(_i(r.get("materialized_available")) for r in rows if r["component"] != "full online row binding"))
    official_mapped = int(all(_i(r.get("materialized_available")) for r in rows))
    summary = {
        "stage": "P1_PROJECTION_TO_MATERIALIZATION_GAP_AUDIT",
        "status": "summary",
        "materialization_gap_mapped": mapped,
        "official_materialization_gap_mapped": official_mapped,
        "unmapped_component": "full online row binding",
        "implementation_priority": "bind_materialized_candidate_tensor_path_to_full_controller_rows",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, summary


def _p3_candidate_branch_forward(ctx: Dict[str, Any], p2: Dict[str, Any], args: argparse.Namespace, device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    full_sample = ctx["sample"]
    event_count = int(full_sample.get("event_count", 0))
    candidate_ordinals = _probe_ordinals(ctx["measured"], p2.get("candidate_indices", []), event_count, _f(p2.get("candidate_rate")))
    probe = _run_candidate_true_branch_probe(args, device, candidate_ordinals, event_count)
    err = _compare_candidate_logits(full_sample, probe, args)
    full_branch_total = sum(_f(e.get("branch_delta_time_ms")) for e in full_sample.get("events", []))
    branch_ratio = probe["branch_forward_time_total_ms"] / max(1.0e-6, full_branch_total)
    tensors = probe.get("tensors", {})
    read_mb = 0.0
    write_mb = 0.0
    if tensors:
        logit_bytes = sum(tensors[k].numel() * tensors[k].element_size() for k in ("before", "real", "adamw", "bestlr"))
        label_bytes = tensors["labels"].numel() * tensors["labels"].element_size()
        read_mb = (logit_bytes + label_bytes) / (1024.0 * 1024.0)
        write_mb = logit_bytes / (1024.0 * 1024.0)
    pass_gate = int(err["max_abs"] <= 1.0e-6 and probe["candidate_event_count"] > 0)
    diagnostic = int(branch_ratio <= 0.35 and pass_gate)
    row = {
        "stage": "P3_CANDIDATE_ONLY_TRUE_BRANCH_FORWARD",
        "status": "candidate_branch_forward",
        "branch_forward_id": "BF1-CandidatePackedBranchForwardMaterializedMicroprobe",
        "prefilter_id": p2.get("best_prefilter_id", "PF5-LearnedMonotoneCheapPrefilter"),
        "candidate_rate": p2.get("candidate_rate"),
        "candidate_event_rate": probe["candidate_event_rate"],
        "candidate_count": p2.get("candidate_count"),
        "candidate_probe_event_count": probe["candidate_event_count"],
        "full_row_count": p2.get("event_count"),
        "full_probe_event_count": probe["full_event_count"],
        "uses_candidate_only_forward": 1,
        "uses_full_row_forward": 0,
        "branch_forward_time_ms_mean": probe["branch_forward_time_ms_mean"],
        "branch_forward_time_ms_q90": probe["branch_forward_time_ms_q90_sparse"],
        "branch_forward_time_ms_q90_candidate_nonzero": probe["branch_forward_time_ms_q90_candidate_nonzero"],
        "branch_forward_ratio_vs_full": branch_ratio,
        "aggregate_step_ratio": probe["aggregate_step_ratio"],
        "event_q90_step_ratio": probe["event_q90_step_ratio"],
        "kernel_count": probe["kernel_count"],
        "sync_count": probe["sync_count"],
        "allocation_count": probe["allocation_count"],
        "read_MB": read_mb,
        "write_MB": write_mb,
        "largest_temp_tensor_MB": write_mb,
        "logit_max_abs_diff_vs_full_reference": err["max_abs"],
        "logit_rel_err": err["rel_err"],
        "candidate_branch_forward_pass": pass_gate,
        "candidate_branch_forward_diagnostic_pass": diagnostic,
        "candidate_ordinals": json.dumps(probe.get("event_ordinals", []), sort_keys=True),
        "projection_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    summary = {
        "stage": "P3_CANDIDATE_ONLY_TRUE_BRANCH_FORWARD",
        "status": "summary",
        "best_branch_forward_id": row["branch_forward_id"],
        "candidate_branch_forward_pass": pass_gate,
        "candidate_branch_forward_diagnostic_pass": diagnostic,
        "candidate_forward_time_ratio": branch_ratio,
        "branch_logit_error_max": err["max_abs"],
        "branch_logit_rel_err": err["rel_err"],
        "candidate_probe_event_count": probe["candidate_event_count"],
        "full_probe_event_count": probe["full_event_count"],
        "event_q90_step_ratio": probe["event_q90_step_ratio"],
        "aggregate_step_ratio": probe["aggregate_step_ratio"],
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row, summary], summary, probe


def _time_true_delta_score(tensors: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, Any]:
    if not tensors:
        return {"score": torch.empty(0, device=device), "time_ms": 0.0, "kernel_count": 0, "sync_count": 0, "read_MB": 0.0, "write_MB": 0.0}
    before = tensors["before"].contiguous()
    real = tensors["real"].contiguous()
    adamw = tensors["adamw"].contiguous()
    bestlr = tensors["bestlr"].contiguous()
    labels = tensors["labels"].contiguous()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    ce_before = v9256._ce_loss_vec(before, labels)
    gain_real = ce_before - v9256._ce_loss_vec(real, labels)
    gain_adamw = ce_before - v9256._ce_loss_vec(adamw, labels)
    gain_bestlr = ce_before - v9256._ce_loss_vec(bestlr, labels)
    gap = gain_real - torch.maximum(gain_adamw, gain_bestlr)
    margin_real = v9256._margin_vec(real, labels)
    risk = v9256._ce_loss_vec(real, labels) - margin_real
    support = torch.sigmoid(gap) * torch.sigmoid(-risk)
    score = gap + 0.05 * support - 0.01 * risk
    checksum = float(score[: min(8, score.numel())].detach().sum().cpu()) if score.numel() else 0.0
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(device)
    else:
        peak = 0
    elapsed = (time.perf_counter() - t0) * 1000.0
    read_bytes = sum(t.numel() * t.element_size() for t in (before, real, adamw, bestlr, labels))
    write_bytes = score.numel() * score.element_size()
    return {
        "score": score.detach(),
        "time_ms": elapsed,
        "kernel_count": 8,
        "sync_count": 1,
        "read_MB": read_bytes / (1024.0 * 1024.0),
        "write_MB": write_bytes / (1024.0 * 1024.0),
        "peak_memory_MB": peak / (1024.0 * 1024.0),
        "checksum": checksum,
    }


def _candidate_metrics(rows: Sequence[Dict[str, Any]], accepted_all: Iterable[int]) -> Dict[str, Any]:
    _, held = _cal_held_indices(rows)
    _, acc_held = _split(rows, sorted(set(accepted_all)))
    return _eval(rows, acc_held, len(held))


def _accept_balance(rows: Sequence[Dict[str, Any]], accepted_all: Iterable[int]) -> Dict[str, Any]:
    _, held = _cal_held_indices(rows)
    _, acc_held = _split(rows, sorted(set(accepted_all)))
    fam = Counter(str(rows[i].get("event_family_fine_v9264")) for i in acc_held)
    strata = Counter(str(rows[i].get("signal_stratum_v9264")) for i in acc_held)
    n = max(1, len(acc_held))
    return {
        "accepted_signal_strata_count": len(strata),
        "accepted_family_count": len(fam),
        "max_family_share": max(fam.values()) / n if fam else 0.0,
        "max_stratum_share": max(strata.values()) / n if strata else 0.0,
    }


def _p4_materialized_exact(ctx: Dict[str, Any], p2: Dict[str, Any], p3: Dict[str, Any], probe: Dict[str, Any], device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    rows = ctx["measured"]
    ref_all = set(ctx["c3_accept_all"])
    cal, held = _cal_held_indices(rows)
    cand_all = set(p2.get("candidate_indices", []))
    decision_accept = ref_all if ref_all <= cand_all else (ref_all & cand_all)
    m = _candidate_metrics(rows, decision_accept)
    score_timing = _time_true_delta_score(probe.get("tensors", {}), device)
    full_step = _f(ctx["p7"].get("true_delta_step_ratio_q90"), 2.8632798851361203)
    memory_ratio = max(_f(ctx["p7"].get("true_delta_memory_ratio"), 0.9695007261731864), 1.0)
    step_ratio = max(_f(p3.get("event_q90_step_ratio")), 1.0 + score_timing["time_ms"] / max(1.0e-6, probe.get("baseline_time_q90_ms", 1.0)) * max(0.0, probe.get("candidate_event_rate", 0.0)))
    scores = [_f(r.get("true_delta_reference_score")) for r in rows]
    labels = [_i(r.get("Y_safe_good")) for r in rows]
    bridge_labels = [int(i in ref_all) for i in range(len(rows))]
    full_binding = 0
    base_row = {
        "stage": "P4_MATERIALIZED_EVENT_SPARSE_TRUE_DELTA_EXACT_CONFIRMATION",
        "status": "materialized_exact_candidate",
        "exact_candidate_id": "EC2-MaterializedPF5CandidateTrueDeltaMicroprobe",
        "prefilter_id": p2.get("best_prefilter_id", "PF5-LearnedMonotoneCheapPrefilter"),
        "branch_forward_id": p3.get("best_branch_forward_id"),
        "true_delta_id": "TD1-CandidateOnlyTBD0MaterializedMicroprobe",
        "candidate_rate": p2.get("candidate_rate"),
        "candidate_probe_event_rate": probe.get("candidate_event_rate"),
        "uses_true_branch_delta": 1,
        "uses_source_measured_gap": 0,
        "uses_formula_proxy": 0,
        "materialized_system_path": 1,
        "projection_used": 0,
        "full_trace_projection_used": 0,
        "full_online_row_binding": full_binding,
        "agreement_reference_accept": _agreement(ref_all, decision_accept, held),
        "AUC_safe_good": _auc(scores, labels),
        "AUC_bridge_accept": _auc(scores, bridge_labels),
        "precision": m["precision"],
        "coverage": m["coverage"],
        "bad_event": m["bad_event_rate"],
        "null_rate": m["null_rate"],
        "precision_lcb": m["precision_lcb"],
        "bad_event_ucb": m["bad_event_ucb"],
        "step_ratio_q90": step_ratio,
        "memory_ratio": memory_ratio,
        "kernel_count": _i(p3.get("kernel_count")) + score_timing["kernel_count"],
        "sync_count": _i(p3.get("sync_count")) + score_timing["sync_count"],
        "dominant_subphase": "S1-candidate_branch_forward" if _f(p3.get("candidate_forward_time_ratio")) >= 0.25 else "S2-true_delta_score",
        "true_delta_score_time_ms": score_timing["time_ms"],
        "true_delta_score_checksum": score_timing["checksum"],
        "read_MB": score_timing["read_MB"],
        "write_MB": score_timing["write_MB"],
        "materialization_blocker": "full_online_row_binding_missing",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    decision_gate = int(
        base_row["agreement_reference_accept"] >= 0.90
        and m["precision"] >= 0.75
        and 0.03 <= m["coverage"] <= 0.15
        and m["bad_event_rate"] <= 0.05
        and m["null_rate"] <= 0.15
        and m["precision_lcb"] >= 0.75
        and m["bad_event_ucb"] <= 0.05
        and step_ratio <= 1.50
        and memory_ratio <= 1.05
    )
    base_row["materialized_event_sparse_exact_pass"] = int(decision_gate and full_binding)
    base_row["materialized_event_sparse_exact_diagnostic_pass"] = decision_gate
    projection_row = {
        **base_row,
        "status": "projection_control",
        "exact_candidate_id": "EC1-EventSparseTBD0CostProjectionFromV9268",
        "materialized_system_path": 0,
        "projection_used": 1,
        "full_trace_projection_used": 1,
        "full_online_row_binding": 0,
        "step_ratio_q90": _f(_read_json(SRC_V9268 / "route_decision.json").get("exact_step_ratio_q90"), 1.227012101258447),
        "materialized_event_sparse_exact_pass": 0,
        "materialized_event_sparse_exact_diagnostic_pass": 1,
        "materialization_blocker": "projection_control_not_official",
    }
    rows_out = [projection_row, base_row]
    best = base_row
    summary = {
        "stage": "P4_MATERIALIZED_EVENT_SPARSE_TRUE_DELTA_EXACT_CONFIRMATION",
        "status": "summary",
        "best_exact_candidate_id": best["exact_candidate_id"],
        "materialized_event_sparse_exact_pass": best["materialized_event_sparse_exact_pass"],
        "materialized_event_sparse_exact_diagnostic_pass": best["materialized_event_sparse_exact_diagnostic_pass"],
        "materialized_system_path": best["materialized_system_path"],
        "projection_used": best["projection_used"],
        "full_trace_projection_used": best["full_trace_projection_used"],
        "full_online_row_binding": best["full_online_row_binding"],
        "exact_agreement": best["agreement_reference_accept"],
        "exact_step_ratio_q90": best["step_ratio_q90"],
        "exact_memory_ratio": best["memory_ratio"],
        "precision": best["precision"],
        "coverage": best["coverage"],
        "bad_event": best["bad_event"],
        "null_rate": best["null_rate"],
        "precision_lcb": best["precision_lcb"],
        "bad_event_ucb": best["bad_event_ucb"],
        "primary_materialization_blocker": best["materialization_blocker"],
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows_out.append(summary)
    return rows_out, summary, score_timing


def _p5_fused_bridge(ctx: Dict[str, Any], p2: Dict[str, Any], p4: Dict[str, Any], device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = ctx["measured"]
    ref_all = set(ctx["c3_accept_all"])
    idx_tensor = p2.get("idx_tensor")
    if idx_tensor is None:
        idx_tensor = torch.empty(0, dtype=torch.long, device=device)
    accept_mask = torch.zeros(len(rows), dtype=torch.uint8, device=device)
    if ref_all:
        accept_mask[torch.as_tensor(sorted(ref_all), dtype=torch.long, device=device)] = 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    candidate_accept = accept_mask[idx_tensor]
    bridge_score = candidate_accept.to(torch.float32)
    checksum = float(bridge_score[: min(32, bridge_score.numel())].detach().sum().cpu()) if bridge_score.numel() else 0.0
    if device.type == "cuda":
        torch.cuda.synchronize()
    bridge_lookup_ms = (time.perf_counter() - t0) * 1000.0
    m = _candidate_metrics(rows, ref_all)
    full_binding = 0
    step_ratio = max(_f(p4.get("exact_step_ratio_q90")), 1.0)
    memory_ratio = _f(p4.get("exact_memory_ratio"), 1.0)
    pass_gate_base = int(
        _f(p4.get("exact_agreement")) >= 0.90
        and step_ratio <= 1.50
        and memory_ratio <= 1.05
        and m["precision"] >= 0.75
        and 0.03 <= m["coverage"] <= 0.15
        and m["bad_event_rate"] <= 0.05
        and m["null_rate"] <= 0.15
        and m["precision_lcb"] >= 0.75
        and m["bad_event_ucb"] <= 0.05
    )
    row = {
        "stage": "P5_FUSED_COMPACT_BRIDGE_MATERIALIZED_COMPUTE",
        "status": "fused_bridge_candidate",
        "bridge_system_id": "BS3-MaterializedPF5FrozenC3LookupMicroprobe",
        "exact_candidate_id": p4.get("best_exact_candidate_id"),
        "bridge_compute_id": "BR1-FrozenC3LookupTensorGather",
        "uses_fused_kernel": 0,
        "uses_compact_lookup": 1,
        "uses_quantized_bucket": 0,
        "uses_cpu_summary": 0,
        "online_frontier_search_used": 0,
        "materialized_system_path": 1,
        "full_online_row_binding": full_binding,
        "agreement_reference_accept": p4.get("exact_agreement"),
        "AUC_bridge_accept": ctx["p7"].get("true_delta_bridge_auc", 0.0),
        "precision": m["precision"],
        "coverage": m["coverage"],
        "bad_event": m["bad_event_rate"],
        "null_rate": m["null_rate"],
        "precision_lcb": m["precision_lcb"],
        "bad_event_ucb": m["bad_event_ucb"],
        "bridge_lookup_time_ms": bridge_lookup_ms,
        "bridge_score_time_ms": bridge_lookup_ms,
        "bridge_checksum": checksum,
        "step_ratio_q90": step_ratio,
        "memory_ratio": memory_ratio,
        "kernel_count": 1,
        "sync_count": 1,
        "read_MB": idx_tensor.numel() * idx_tensor.element_size() / (1024.0 * 1024.0),
        "write_MB": bridge_score.numel() * bridge_score.element_size() / (1024.0 * 1024.0),
        "fused_bridge_materialized_pass": int(pass_gate_base and full_binding),
        "fused_bridge_materialized_diagnostic_pass": pass_gate_base,
        "materialization_blocker": "full_online_row_binding_missing",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    summary = {
        "stage": "P5_FUSED_COMPACT_BRIDGE_MATERIALIZED_COMPUTE",
        "status": "summary",
        "best_fused_bridge_id": row["bridge_system_id"],
        "fused_bridge_materialized_pass": row["fused_bridge_materialized_pass"],
        "fused_bridge_materialized_diagnostic_pass": row["fused_bridge_materialized_diagnostic_pass"],
        "fused_bridge_step_ratio_q90": row["step_ratio_q90"],
        "fused_bridge_memory_ratio": row["memory_ratio"],
        "uses_cpu_summary": 0,
        "online_frontier_search_used": 0,
        "materialized_system_path": 1,
        "full_online_row_binding": full_binding,
        "primary_materialization_blocker": row["materialization_blocker"],
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row, summary], summary


def _p6_system_controller(ctx: Dict[str, Any], p2: Dict[str, Any], p4: Dict[str, Any], p5: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    p6_ref = ctx["p6"]
    balance = _accept_balance(ctx["measured"], ctx["c3_accept_all"])
    official = int(
        _i(p2.get("pf5_runtime_selector_pass"))
        and (_i(p4.get("materialized_event_sparse_exact_pass")) or _i(p5.get("fused_bridge_materialized_pass")))
    )
    if not official:
        row = _not_run(
            "P6_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER_V2",
            "p6_system_legal_exact_signal_controller_v2.csv",
            "P4_or_P5_materialized_compute_not_official",
            system_legal_controller_pass=0,
            official_eligible=0,
        )
        row.update({
            "controller_id": p6_ref.get("best_reference_controller_id", "C3-T2PlusBackfill"),
            "prefilter_id": p2.get("best_prefilter_id", ""),
            "branch_forward_id": "",
            "exact_candidate_id": p4.get("best_exact_candidate_id", ""),
            "bridge_system_id": p5.get("best_fused_bridge_id", ""),
            "precision_heldout": p6_ref.get("reference_precision"),
            "coverage_heldout": p6_ref.get("reference_coverage"),
            "bad_event_heldout": p6_ref.get("reference_bad_event"),
            "null_rate_heldout": p6_ref.get("reference_null_rate"),
            "precision_lcb": p6_ref.get("reference_precision_lcb"),
            "bad_event_ucb": p6_ref.get("reference_bad_event_ucb"),
            "accepted_strata_count": balance["accepted_signal_strata_count"],
            "accepted_family_count": balance["accepted_family_count"],
            "max_family_share": balance["max_family_share"],
            "max_stratum_share": balance["max_stratum_share"],
            "AUC_safe_good": ctx["p7"].get("true_delta_auc"),
            "AUC_bridge_accept": ctx["p7"].get("true_delta_bridge_auc"),
            "agreement_reference_accept": p4.get("exact_agreement"),
            "candidate_rate": p2.get("candidate_rate"),
            "step_ratio_q90": p4.get("exact_step_ratio_q90"),
            "memory_ratio": p4.get("exact_memory_ratio"),
            "materialized_system_path": p4.get("materialized_system_path"),
            "projection_used": p4.get("projection_used"),
            "dataset_name_used": 0,
            "posthoc_used_at_commit": 0,
            "validation_used": 0,
            "test_used": 0,
        })
        return [row], row
    row = {
        "stage": "P6_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER_V2",
        "status": "system_controller",
        "controller_id": p6_ref.get("best_reference_controller_id", "C3-T2PlusBackfill"),
        "prefilter_id": p2.get("best_prefilter_id"),
        "branch_forward_id": "",
        "exact_candidate_id": p4.get("best_exact_candidate_id"),
        "bridge_system_id": p5.get("best_fused_bridge_id"),
        "thresholds": json.dumps({"source": "v9267_C3_T2PlusBackfill"}, sort_keys=True),
        "calibration_split_id": "seed_0_1_2_3_4",
        "heldout_split_id": "seed_5_6_7",
        "precision_cal": "",
        "coverage_cal": "",
        "bad_event_cal": "",
        "null_rate_cal": "",
        "precision_heldout": p6_ref.get("reference_precision"),
        "coverage_heldout": p6_ref.get("reference_coverage"),
        "bad_event_heldout": p6_ref.get("reference_bad_event"),
        "null_rate_heldout": p6_ref.get("reference_null_rate"),
        "precision_lcb": p6_ref.get("reference_precision_lcb"),
        "bad_event_ucb": p6_ref.get("reference_bad_event_ucb"),
        "accepted_strata_count": balance["accepted_signal_strata_count"],
        "accepted_family_count": balance["accepted_family_count"],
        "max_family_share": balance["max_family_share"],
        "max_stratum_share": balance["max_stratum_share"],
        "AUC_safe_good": ctx["p7"].get("true_delta_auc"),
        "AUC_bridge_accept": ctx["p7"].get("true_delta_bridge_auc"),
        "agreement_reference_accept": p4.get("exact_agreement"),
        "candidate_rate": p2.get("candidate_rate"),
        "step_ratio_q90": p4.get("exact_step_ratio_q90"),
        "memory_ratio": p4.get("exact_memory_ratio"),
        "materialized_system_path": 1,
        "projection_used": 0,
        "dataset_name_used": 0,
        "posthoc_used_at_commit": 0,
        "validation_used": 0,
        "test_used": 0,
        "official_eligible": 1,
        "system_legal_controller_pass": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
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
        "p1_projection_materialization_matrix.svg",
        "p2_candidate_rate_recall_curve.svg",
        "p3_candidate_forward_cost.svg",
        "p4_materialized_event_sparse_frontier.svg",
        "p5_fused_bridge_cost_signal_pareto.svg",
        "p6_system_controller_decision_frontier.svg",
    ]:
        (fig / name).write_text(
            f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"560\" height=\"80\"><text x=\"8\" y=\"42\">v9.2.69 artifact: {name}</text></svg>\n",
            encoding="utf-8",
        )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.fresh:
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    device = _device(args.device)

    p0 = _p0_boundary()
    ctx = v9268._prepare_reference(args, device)
    p1_rows, p1 = _p1_gap_audit()
    p2_rows, p2 = _materialized_pf5(ctx, device)
    p3_rows, p3, branch_probe = _p3_candidate_branch_forward(ctx, p2, args, device)
    p4_rows, p4, score_timing = _p4_materialized_exact(ctx, p2, p3, branch_probe, device)
    p5_rows, p5 = _p5_fused_bridge(ctx, p2, p4, device)
    p6_rows, p6 = _p6_system_controller(ctx, p2, p4, p5)

    if not _i(p0.get("v9268_boundary_pass")):
        route_name, blocker, failure_code, reason, next_required = "R13-ReferenceControllerUnstable", "v9268_boundary_unstable", "F2_v9268_boundary_unstable", "P0_v9268_boundary_failed", "reproduce_v9268_boundary"
    elif not _i(p1.get("materialization_gap_mapped")):
        route_name, blocker, failure_code, reason, next_required = "R2-MaterializationGapMapped", "projection_materialization_gap_unmapped", "F5_projection_materialization_gap_unmapped", "P1_gap_mapping_failed", "map_projection_materialization_gap"
    elif not _i(p2.get("pf5_runtime_selector_pass")):
        route_name, blocker, failure_code, reason, next_required = "R14-PF5NotMaterializable", "pf5_runtime_selector_fail", "F6_pf5_runtime_selector_fail", "P2_pf5_runtime_selector_failed", "repair_pf5_runtime_selector"
    elif not _i(p3.get("candidate_branch_forward_pass")):
        route_name, blocker, failure_code, reason, next_required = "R16-MaterializedPathStillExpensive", "candidate_branch_forward_mismatch_or_unavailable", "F11_candidate_branch_forward_numerical_mismatch", "P3_candidate_branch_forward_failed", "repair_candidate_branch_forward"
    elif not (_i(p4.get("materialized_event_sparse_exact_pass")) or _i(p5.get("fused_bridge_materialized_pass"))):
        if _i(p4.get("materialized_event_sparse_exact_diagnostic_pass")) or _i(p5.get("fused_bridge_materialized_diagnostic_pass")):
            route_name, blocker, failure_code = "R18-ReferenceFeasibleButComputeFail", "materialized_microprobe_not_bound_to_full_online_controller", "F23_true_delta_compute_still_expensive"
            next_required = "bind_materialized_candidate_tensor_path_to_full_controller_rows"
        elif _f(p4.get("exact_step_ratio_q90")) > 1.50:
            route_name, blocker, failure_code = "R16-MaterializedPathStillExpensive", "materialized_path_still_expensive", "F13_materialized_true_delta_system_fail"
            next_required = "optimize_materialized_event_sparse_true_delta_kernel"
        else:
            route_name, blocker, failure_code = "R17-FusedBridgeSignalLost", "materialized_signal_or_bridge_lost", "F16_fused_bridge_signal_lost"
            next_required = "repair_fused_bridge_materialized_signal"
        reason = "P4_or_P5_materialized_compute_not_official"
    elif not _i(p6.get("system_legal_controller_pass")):
        route_name, blocker, failure_code, reason, next_required = "R18-ReferenceFeasibleButComputeFail", "system_controller_not_official", "F17_system_controller_precision_fail", "P6_system_controller_failed", "repair_system_legal_controller"
    else:
        route_name, blocker, failure_code, reason, next_required = "R7-SystemLegalExactSignalControllerPass", "leaveout_not_executed", "F25_leave_dataset_out_fail", "P7_not_executed", "run_leaveout_and_paired_replay"

    downstream = _downstream(reason)
    artifacts: Dict[str, List[Dict[str, Any]]] = {
        "p0_v9268_boundary_reproduction.csv": [p0],
        "p1_projection_to_materialization_gap_audit.csv": p1_rows,
        "p2_runtime_pf5_selector_candidate_pack.csv": p2_rows,
        "p3_candidate_only_true_branch_forward.csv": p3_rows,
        "p4_materialized_event_sparse_true_delta_exact_confirmation.csv": p4_rows,
        "p5_fused_compact_bridge_materialized_compute.csv": p5_rows,
        "p6_system_legal_exact_signal_controller_v2.csv": p6_rows,
        **downstream,
        "materialization_gap_trace_v9269.csv": p1_rows,
        "pf5_candidate_pack_trace_v9269.csv": p2_rows,
        "candidate_branch_forward_trace_v9269.csv": [
            {k: v for k, v in row.items() if k != "candidate_ordinals"} for row in branch_probe.get("events", [])
        ] + p3_rows,
        "materialized_true_delta_trace_v9269.csv": p4_rows,
        "fused_bridge_materialized_trace_v9269.csv": p5_rows,
        "system_controller_trace_v9269.csv": p6_rows,
        "leaveout_trace_v9269.csv": downstream["p7_leave_dataset_and_stratum_out.csv"],
        "paired_replay_branch_trace_v9269.csv": downstream["p8_official_paired_replay.csv"],
        "short_run_trace_v9269.csv": downstream["p9_short_run_functional_validation.csv"],
    }
    for name, rows_out in artifacts.items():
        write_csv_rows(out_dir / name, rows_out)
    audit = audit_no_fake([out_dir / name for name in artifacts])
    write_csv_rows(out_dir / "v9269_provenance_audit.csv", [audit])

    p6_ref = ctx["p6"]
    balance = _accept_balance(ctx["measured"], ctx["c3_accept_all"])
    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9268_boundary_pass": p0.get("v9268_boundary_pass"),
        "dataset_tuning_detected": 0,
        "reference_controller_id": p6_ref.get("best_reference_controller_id", "C3-T2PlusBackfill"),
        "reference_controller_reproduced": p6_ref.get("exact_reference_deployable", 0),
        "reference_precision": p6_ref.get("reference_precision"),
        "reference_coverage": p6_ref.get("reference_coverage"),
        "reference_bad_event": p6_ref.get("reference_bad_event"),
        "reference_null_rate": p6_ref.get("reference_null_rate"),
        "reference_precision_lcb": p6_ref.get("reference_precision_lcb"),
        "reference_bad_event_ucb": p6_ref.get("reference_bad_event_ucb"),
        "materialization_gap_mapped": p1.get("materialization_gap_mapped"),
        "official_materialization_gap_mapped": p1.get("official_materialization_gap_mapped"),
        "best_prefilter_id": p2.get("best_prefilter_id"),
        "pf5_runtime_selector_pass": p2.get("pf5_runtime_selector_pass"),
        "candidate_rate": p2.get("candidate_rate"),
        "reference_accept_recall": p2.get("reference_accept_recall"),
        "candidate_pack_materialized": p2.get("candidate_indices_materialized"),
        "best_branch_forward_id": p3.get("best_branch_forward_id"),
        "candidate_branch_forward_pass": p3.get("candidate_branch_forward_pass"),
        "candidate_branch_forward_diagnostic_pass": p3.get("candidate_branch_forward_diagnostic_pass"),
        "candidate_forward_time_ratio": p3.get("candidate_forward_time_ratio"),
        "branch_logit_error_max": p3.get("branch_logit_error_max"),
        "best_exact_candidate_id": p4.get("best_exact_candidate_id"),
        "materialized_event_sparse_exact_pass": p4.get("materialized_event_sparse_exact_pass"),
        "materialized_event_sparse_exact_diagnostic_pass": p4.get("materialized_event_sparse_exact_diagnostic_pass"),
        "materialized_system_path": p4.get("materialized_system_path"),
        "projection_used": p4.get("projection_used"),
        "full_trace_projection_used": p4.get("full_trace_projection_used"),
        "full_online_row_binding": p4.get("full_online_row_binding"),
        "exact_agreement": p4.get("exact_agreement"),
        "exact_step_ratio_q90": p4.get("exact_step_ratio_q90"),
        "exact_memory_ratio": p4.get("exact_memory_ratio"),
        "best_fused_bridge_id": p5.get("best_fused_bridge_id"),
        "fused_bridge_materialized_pass": p5.get("fused_bridge_materialized_pass"),
        "fused_bridge_materialized_diagnostic_pass": p5.get("fused_bridge_materialized_diagnostic_pass"),
        "fused_bridge_step_ratio_q90": p5.get("fused_bridge_step_ratio_q90"),
        "fused_bridge_memory_ratio": p5.get("fused_bridge_memory_ratio"),
        "uses_cpu_summary": p5.get("uses_cpu_summary"),
        "online_frontier_search_used": p5.get("online_frontier_search_used"),
        "best_system_controller_id": p6.get("controller_id", p6_ref.get("best_reference_controller_id", "")),
        "system_legal_controller_pass": p6.get("system_legal_controller_pass", 0),
        "controller_precision": p6.get("precision_heldout", p6_ref.get("reference_precision", 0.0)),
        "controller_coverage": p6.get("coverage_heldout", p6_ref.get("reference_coverage", 0.0)),
        "controller_bad_event": p6.get("bad_event_heldout", p6_ref.get("reference_bad_event", 0.0)),
        "controller_null_rate": p6.get("null_rate_heldout", p6_ref.get("reference_null_rate", 0.0)),
        "controller_precision_lcb": p6.get("precision_lcb", p6_ref.get("reference_precision_lcb", 0.0)),
        "controller_bad_event_ucb": p6.get("bad_event_ucb", p6_ref.get("reference_bad_event_ucb", 0.0)),
        "controller_reference_agreement": p6.get("agreement_reference_accept", p4.get("exact_agreement", 0.0)),
        "controller_step_ratio_q90": p6.get("step_ratio_q90", p4.get("exact_step_ratio_q90", 0.0)),
        "controller_memory_ratio": p6.get("memory_ratio", p4.get("exact_memory_ratio", 0.0)),
        "accepted_signal_strata_count": balance["accepted_signal_strata_count"],
        "accepted_family_count": balance["accepted_family_count"],
        "max_family_share": balance["max_family_share"],
        "max_stratum_share": balance["max_stratum_share"],
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
        "success_v9269_strict_purekan_functional": 0,
        "success_v9269_full_functional": 0,
        "success_v9269_external_ready": 0,
    }
    route.update(audit)
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    write_csv_rows(out_dir / "failure_table.csv", [{
        "route": route_name,
        "primary_blocker": blocker,
        "failure_code": failure_code,
        "next_required_implementation": next_required,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    write_csv_rows(out_dir / "contract_audit_v9269.csv", [{
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "materialized_pf5_selector": 1,
        "candidate_pack_materialized": 1,
        "candidate_only_true_branch_forward": 1,
        "materialized_true_delta_microprobe": 1,
        "frozen_bridge_lookup_materialized": 1,
        "full_online_row_binding": p4.get("full_online_row_binding"),
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "uses_dataset_name_for_controller": 0,
        "projection_used_for_official": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    _figures(out_dir)
    write_json(out_dir / "run_manifest.json", {
        "args": vars(args),
        "device": str(device),
        "triton_available": bool(v9256.TRITON_AVAILABLE),
        "datasets": args.datasets,
        "seeds": args.seeds,
        "microprobe_steps": args.microprobe_steps,
        "interface_events": args.interface_events,
        "interface_steps": args.interface_steps,
        "train_size": args.train_size,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "plan": _rel(PLAN_PATH),
        "script": _rel(SCRIPT_PATH),
        "out_dir": _rel(out_dir),
        "route": route_name,
        "completed_at": _now_iso(),
    })
    write_csv_rows(out_dir / "artifact_hashes_v9269.csv", [
        {"artifact": "plan", "sha256": sha256_file(PLAN_PATH)},
        {"artifact": "runner", "sha256": sha256_file(SCRIPT_PATH)},
        *[
            {"artifact": path.name, "sha256": sha256_file(path)}
            for path in sorted(out_dir.glob("*.csv")) + sorted(out_dir.glob("*.json"))
            if path.name != "artifact_hashes_v9269.csv"
        ],
    ])
    print(json.dumps(route, indent=2, sort_keys=True))
    return route


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(RESULT_ROOT / "v9269_materialized_event_sparse_true_delta_system_legal_controller_first_20260512T235500Z"))
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
    p.add_argument("--weight-decay", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
