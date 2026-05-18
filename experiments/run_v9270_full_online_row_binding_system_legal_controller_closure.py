#!/usr/bin/env python3
"""DG-KAN v9.2.70 full-online row binding audit.

v9.2.69 materialized the PF5/event-sparse true-delta path only as a
microprobe.  This runner promotes only the parts that can be bound to full
online controller rows.  Missing raw candidate tensors, branch logits, or
functional update payloads are recorded as blocking gaps instead of being
reconstructed from projected or source-measured summaries.
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

import run_v9265_joint_safe_useful_feasibility_null_bad_conflict_resolution as v9265  # noqa: E402
import run_v9266_oracle_legal_gap_closure_bridge_frontier as v9266  # noqa: E402
import run_v9268_true_delta_system_closure_reference_feasible_controller_promotion as v9268  # noqa: E402
import run_v9269_materialized_event_sparse_true_delta_system_legal_controller as v9269  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, read_csv_rows, sha256_file, write_csv_rows, write_json  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.70_FullOnlineRowBinding_SystemLegalControllerClosure_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9270_full_online_row_binding_system_legal_controller_closure.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9269 = RESULT_ROOT / "v9269_materialized_event_sparse_true_delta_system_legal_controller_first_20260512T235500Z"

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


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_id(text: str, mod: int = 1_000_003) -> int:
    return int(_hash_text(text)[:16], 16) % mod


def _split(rows: Sequence[Dict[str, Any]], accepted: Sequence[int]) -> Tuple[List[int], List[int]]:
    return v9266._split_sets(rows, accepted)


def _eval(rows: Sequence[Dict[str, Any]], accepted: Sequence[int], denominator: int | None = None) -> Dict[str, Any]:
    return v9266._eval_accept(rows, accepted, denominator)


def _agreement(a: Iterable[int], b: Iterable[int], indices: Sequence[int]) -> float:
    aa, bb = set(a), set(b)
    return sum(int((idx in aa) == (idx in bb)) for idx in indices) / max(1, len(indices))


def _recall(reference: Iterable[int], candidates: Iterable[int]) -> float:
    ref, cand = set(reference), set(candidates)
    return len(ref & cand) / max(1, len(ref))


def _row_str(row: Dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        val = row.get(key)
        if val not in (None, ""):
            return str(val)
    return default


def _row_int(row: Dict[str, Any], *keys: str, default: int = 0) -> int:
    for key in keys:
        val = row.get(key)
        if val not in (None, ""):
            return _i(val, default)
    return default


def _p0_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9269 / "route_decision.json")
    audit_rows = read_csv_rows(SRC_V9269 / "v9269_provenance_audit.csv")
    audit = audit_rows[0] if audit_rows else {}
    p0_pass = int(
        route.get("route") == "R18-ReferenceFeasibleButComputeFail"
        and route.get("reference_controller_id") == "C3-T2PlusBackfill"
        and _i(route.get("reference_controller_reproduced")) == 1
        and _i(route.get("pf5_runtime_selector_pass")) == 1
        and _i(route.get("materialized_event_sparse_exact_diagnostic_pass")) == 1
        and _i(route.get("materialized_event_sparse_exact_pass")) == 0
        and _i(route.get("full_online_row_binding")) == 0
        and _i(route.get("system_legal_controller_pass")) == 0
        and _i(audit.get("fake_proxy_nonzero_count")) == 0
        and _i(audit.get("proxy_row_used")) == 0
        and _i(audit.get("cpu_offload_used")) == 0
    )
    return {
        "stage": "P0_V9269_BOUNDARY_REPRODUCTION",
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
        "pf5_runtime_selector_pass": route.get("pf5_runtime_selector_pass", ""),
        "candidate_rate": route.get("candidate_rate", ""),
        "reference_accept_recall": route.get("reference_accept_recall", ""),
        "materialized_event_sparse_exact_diagnostic_pass": route.get("materialized_event_sparse_exact_diagnostic_pass", ""),
        "materialized_event_sparse_exact_pass": route.get("materialized_event_sparse_exact_pass", ""),
        "full_online_row_binding": route.get("full_online_row_binding", ""),
        "exact_step_ratio_q90": route.get("exact_step_ratio_q90", ""),
        "system_legal_controller_pass": route.get("system_legal_controller_pass", ""),
        "primary_blocker": route.get("primary_blocker", ""),
        "fake_proxy_count": audit.get("fake_proxy_nonzero_count", 0),
        "cpu_offload_used": audit.get("cpu_offload_used", 0),
        "v9269_boundary_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }


def _build_full_event_table(ctx: Dict[str, Any], p2: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = ctx["measured"]
    ref_all = set(ctx["c3_accept_all"])
    cand_indices = list(p2.get("candidate_indices", []))
    cand_set = set(cand_indices)
    cand_rank = {idx: rank for rank, idx in enumerate(cand_indices)}
    rows: List[Dict[str, Any]] = []
    scores = v9269._metric_scores(measured)
    for idx, row in enumerate(measured):
        dataset = _row_str(row, "dataset", "task", default="unknown_dataset")
        seed = _row_int(row, "seed", default=0)
        step = _row_int(row, "step", default=0)
        carrier = _row_str(row, "carrier_id", "role_id", default="unknown_role")
        signal_stratum = _row_str(row, "signal_stratum_v9264", "signal_stratum", default="unknown_stratum")
        family = _row_str(row, "event_family_fine_v9264", "event_family", default="unknown_family")
        global_row_id = idx
        event_id = f"v9270-event-{idx:06d}"
        family_id = v9269._row_family_id(row)
        branch_id = v9269._row_branch_id(row)
        bucket_id = _hash_id(f"{dataset}|{seed}|{signal_stratum}|{family}")
        role_id = _hash_id(carrier, mod=4096)
        is_cand = int(idx in cand_set)
        source_hash = _hash_text(f"{idx}|{row.get('row_id', '')}|{dataset}|{seed}|{step}|{signal_stratum}|{family}")[:32]
        rows.append({
            "global_row_id": global_row_id,
            "event_id": event_id,
            "dataset": dataset,
            "seed": seed,
            "step": step,
            "batch_id": _hash_id(f"{dataset}|{seed}|{step}", mod=100_000),
            "microbatch_id": _hash_id(f"{dataset}|{seed}|{step}|{carrier}", mod=100_000),
            "horizon": _row_str(row, "horizon", "horizon_id", default="online_h0"),
            "signal_stratum": signal_stratum,
            "event_family": family,
            "family_id": family_id,
            "bucket_id": bucket_id,
            "role_id": role_id,
            "branch_id": branch_id,
            "candidate_source": "PF5-LearnedMonotoneCheapPrefilter" if is_cand else "none",
            "source_hash": source_hash,
            "candidate_flag": is_cand,
            "candidate_index": cand_rank[idx] if is_cand else "",
            "candidate_rank": cand_rank[idx] if is_cand else "",
            "candidate_score": scores[idx],
            "candidate_family_id": family_id if is_cand else "",
            "candidate_bucket_id": bucket_id if is_cand else "",
            "candidate_branch_id": branch_id if is_cand else "",
            "candidate_event_id": event_id if is_cand else "",
            "candidate_global_row_id": global_row_id if is_cand else "",
            "accept_decision": int(idx in ref_all),
            "bridge_score": 1.0 if idx in ref_all else 0.0,
            "trim_flag": int(idx not in ref_all and is_cand),
            "backfill_flag": int(idx in ref_all and is_cand),
            "candidate_metadata_bound": is_cand,
            "candidate_tensor_payload_bound": 0,
            "candidate_branch_logits_bound": 0,
            "candidate_true_delta_logits_bound": 0,
            "functional_update_payload_bound": 0,
            "projection_used": 0,
            "full_trace_projection_used": 0,
            "source_measured_gap_used": 0,
            "formula_proxy_used": 0,
            "uses_cpu_summary": 0,
            "online_frontier_search_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    event_ids = [r["event_id"] for r in rows]
    global_ids = [r["global_row_id"] for r in rows]
    candidate_ids = [r["candidate_index"] for r in rows if r["candidate_flag"]]
    candidate_pack_missing = sum(1 for r in rows if r["candidate_flag"] and not r["candidate_tensor_payload_bound"])
    branch_logits_missing = sum(1 for r in rows if r["candidate_flag"] and not r["candidate_branch_logits_bound"])
    true_delta_missing = sum(1 for r in rows if r["candidate_flag"] and not r["candidate_true_delta_logits_bound"])
    update_payload_missing = sum(1 for r in rows if r["accept_decision"] and not r["functional_update_payload_bound"])
    summary = {
        "event_count": len(rows),
        "candidate_count": len(cand_indices),
        "accepted_count": len(ref_all),
        "event_id_duplicate_count": len(event_ids) - len(set(event_ids)),
        "global_row_id_duplicate_count": len(global_ids) - len(set(global_ids)),
        "candidate_index_duplicate_count": len(candidate_ids) - len(set(candidate_ids)),
        "event_table_missing_id_count": sum(1 for r in rows if not r["event_id"] or r["global_row_id"] == ""),
        "candidate_missing_id_count": sum(1 for r in rows if r["candidate_flag"] and (r["candidate_event_id"] == "" or r["candidate_global_row_id"] == "")),
        "candidate_event_id_mismatch_count": sum(1 for r in rows if r["candidate_flag"] and r["candidate_event_id"] != r["event_id"]),
        "candidate_metadata_bound_count": sum(1 for r in rows if r["candidate_metadata_bound"]),
        "candidate_tensor_payload_missing_count": candidate_pack_missing,
        "candidate_branch_logits_missing_count": branch_logits_missing,
        "candidate_true_delta_logits_missing_count": true_delta_missing,
        "functional_update_payload_missing_count": update_payload_missing,
        "candidate_pack_payload_bound": int(candidate_pack_missing == 0 and len(candidate_indices) > 0),
        "branch_logits_full_online_bound": int(branch_logits_missing == 0 and len(candidate_indices) > 0),
        "true_delta_full_online_bound": int(true_delta_missing == 0 and len(candidate_indices) > 0),
        "functional_update_payload_bound": int(update_payload_missing == 0 and len(ref_all) > 0),
        "source_hash_missing_count": sum(1 for r in rows if not r["source_hash"]),
    }
    return rows, summary


def _p1_binding_audit(event_summary: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidate_count = _i(event_summary.get("candidate_count"))
    accepted_count = _i(event_summary.get("accepted_count"))
    components = [
        ("online event table", 1, 1, 0, "global_row_id/event_id/source_hash"),
        ("PF5 candidate selector", 1, 1, 0, "candidate_flag/index/rank/score"),
        ("candidate id tensors", 1, 1, 0, "candidate_global_row_id/event_id/family/bucket/branch"),
        ("candidate tensor payload", 0, 0, event_summary["candidate_tensor_payload_missing_count"], "candidate_x/y/logits_base not stored for full candidate rows"),
        ("candidate branch logits", 0, 0, event_summary["candidate_branch_logits_missing_count"], "candidate-only branch logits not materialized for all PF5 rows"),
        ("candidate true-delta logits", 0, 0, event_summary["candidate_true_delta_logits_missing_count"], "true delta logits not bound beyond microprobe"),
        ("frozen C3 bridge lookup", 1, 1, 0, "compact accept lookup over full event ids"),
        ("bridge score and accept decision", 1, 1, 0, "tensor gather over full online row ids"),
        ("functional update payload", 0, 0, event_summary["functional_update_payload_missing_count"], "accepted-row update payload not materialized"),
    ]
    rows: List[Dict[str, Any]] = []
    for idx, (component, materialized, bound, missing, note) in enumerate(components):
        rows.append({
            "stage": "P1_FULL_ONLINE_ROW_BINDING_CONTRACT_AUDIT",
            "status": "component_contract",
            "audit_id": f"BIND{idx}",
            "component": component,
            "required_for_official": 1,
            "materialized_path": materialized,
            "full_online_row_binding": bound,
            "missing_count": missing,
            "duplicate_count": 0,
            "mismatch_count": 0,
            "projection_used": 0,
            "full_trace_projection_used": 0,
            "source_measured_gap_used": 0,
            "formula_proxy_used": 0,
            "uses_cpu_summary": 0,
            "online_frontier_search_used": 0,
            "note": note,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    hard_missing = sum(_i(r["missing_count"]) for r in rows)
    full_binding = int(all(_i(r["full_online_row_binding"]) for r in rows) and hard_missing == 0)
    event_identity_pass = int(
        event_summary["event_id_duplicate_count"] == 0
        and event_summary["global_row_id_duplicate_count"] == 0
        and event_summary["event_table_missing_id_count"] == 0
        and event_summary["candidate_missing_id_count"] == 0
        and event_summary["candidate_event_id_mismatch_count"] == 0
    )
    summary = {
        "stage": "P1_FULL_ONLINE_ROW_BINDING_CONTRACT_AUDIT",
        "status": "summary",
        "full_online_row_binding_contract_pass": full_binding,
        "full_online_row_binding": full_binding,
        "event_identity_binding_pass": event_identity_pass,
        "candidate_count": candidate_count,
        "accepted_count": accepted_count,
        "candidate_tensor_payload_missing_count": event_summary["candidate_tensor_payload_missing_count"],
        "candidate_branch_logits_missing_count": event_summary["candidate_branch_logits_missing_count"],
        "candidate_true_delta_logits_missing_count": event_summary["candidate_true_delta_logits_missing_count"],
        "functional_update_payload_missing_count": event_summary["functional_update_payload_missing_count"],
        "primary_binding_blocker": "candidate_tensor_and_true_delta_payload_missing_for_full_online_rows",
        "materialized_system_path": 0,
        "projection_used": 0,
        "full_trace_projection_used": 0,
        "source_measured_gap_used": 0,
        "formula_proxy_used": 0,
        "uses_cpu_summary": 0,
        "online_frontier_search_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, summary


def _p2_selector_binding(ctx: Dict[str, Any], p2: Dict[str, Any], event_summary: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = ctx["measured"]
    ref_all = set(ctx["c3_accept_all"])
    cand = set(p2.get("candidate_indices", []))
    pass_gate = int(
        _i(event_summary["event_id_duplicate_count"]) == 0
        and _i(event_summary["candidate_index_duplicate_count"]) == 0
        and _i(event_summary["candidate_missing_id_count"]) == 0
        and _i(event_summary["candidate_event_id_mismatch_count"]) == 0
        and _f(p2.get("candidate_rate")) <= 0.25
        and _f(p2.get("reference_accept_recall")) >= 0.95
    )
    row = {
        "stage": "P2_FULL_ONLINE_EVENT_TABLE_PF5_SELECTOR_BINDING",
        "status": "full_online_selector_binding",
        "prefilter_id": p2.get("best_prefilter_id", "PF5-LearnedMonotoneCheapPrefilter"),
        "event_count": len(rows),
        "candidate_count": p2.get("candidate_count"),
        "candidate_rate": p2.get("candidate_rate"),
        "reference_accept_recall": p2.get("reference_accept_recall"),
        "reference_accept_precision": len(ref_all & cand) / max(1, len(cand)),
        "selector_time_ms": p2.get("selector_time_ms", ""),
        "candidate_pack_time_ms": p2.get("candidate_pack_time_ms"),
        "candidate_pack_memory_MB": p2.get("candidate_pack_memory_MB"),
        "event_id_duplicate_count": event_summary["event_id_duplicate_count"],
        "global_row_id_duplicate_count": event_summary["global_row_id_duplicate_count"],
        "candidate_index_duplicate_count": event_summary["candidate_index_duplicate_count"],
        "candidate_missing_id_count": event_summary["candidate_missing_id_count"],
        "candidate_event_id_mismatch_count": event_summary["candidate_event_id_mismatch_count"],
        "candidate_metadata_bound": 1,
        "candidate_payload_bound": event_summary["candidate_pack_payload_bound"],
        "candidate_payload_missing_count": event_summary["candidate_tensor_payload_missing_count"],
        "event_table_materialized": 1,
        "candidate_indices_materialized": 1,
        "candidate_pack_binding_pass": event_summary["candidate_pack_payload_bound"],
        "pf5_full_online_selector_pass": pass_gate,
        "projection_used": 0,
        "dataset_name_used": 0,
        "posthoc_used_at_commit": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    summary = {
        "stage": "P2_FULL_ONLINE_EVENT_TABLE_PF5_SELECTOR_BINDING",
        "status": "summary",
        "pf5_full_online_selector_pass": pass_gate,
        "candidate_pack_binding_pass": event_summary["candidate_pack_payload_bound"],
        "candidate_rate": row["candidate_rate"],
        "reference_accept_recall": row["reference_accept_recall"],
        "candidate_count": row["candidate_count"],
        "event_count": row["event_count"],
        "candidate_payload_missing_count": row["candidate_payload_missing_count"],
        "projection_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row, summary], summary


def _source_v9269_microprobe() -> Dict[str, Any]:
    route = _read_json(SRC_V9269 / "route_decision.json")
    return {
        "candidate_forward_time_ratio": _f(route.get("candidate_forward_time_ratio")),
        "branch_logit_error_max": _f(route.get("branch_logit_error_max")),
        "exact_step_ratio_q90": _f(route.get("exact_step_ratio_q90")),
        "exact_memory_ratio": _f(route.get("exact_memory_ratio"), 1.0),
        "exact_agreement": _f(route.get("exact_agreement")),
        "reference_precision": _f(route.get("reference_precision")),
        "reference_coverage": _f(route.get("reference_coverage")),
        "reference_bad_event": _f(route.get("reference_bad_event")),
        "reference_null_rate": _f(route.get("reference_null_rate")),
        "reference_precision_lcb": _f(route.get("reference_precision_lcb")),
        "reference_bad_event_ucb": _f(route.get("reference_bad_event_ucb")),
    }


def _p3_branch_forward(event_summary: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    src = _source_v9269_microprobe()
    rows = [
        {
            "stage": "P3_FULL_ONLINE_CANDIDATE_ONLY_BRANCH_FORWARD",
            "status": "diagnostic_microprobe_source_v9269",
            "branch_forward_id": "BF0-V9269MaterializedMicroprobeDiagnostic",
            "candidate_only_forward_materialized": 1,
            "full_online_row_binding": 0,
            "candidate_probe_only": 1,
            "candidate_count": event_summary["candidate_count"],
            "payload_missing_count": event_summary["candidate_tensor_payload_missing_count"],
            "branch_logits_missing_count": event_summary["candidate_branch_logits_missing_count"],
            "branch_forward_time_ratio": src["candidate_forward_time_ratio"],
            "logit_max_abs_diff_vs_full_reference": src["branch_logit_error_max"],
            "candidate_branch_forward_diagnostic_pass": int(src["branch_logit_error_max"] <= 1.0e-6),
            "candidate_branch_forward_full_online_pass": 0,
            "projection_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "stage": "P3_FULL_ONLINE_CANDIDATE_ONLY_BRANCH_FORWARD",
            "status": "not_run_missing_full_candidate_payload",
            "branch_forward_id": "BF1-FullOnlineCandidatePackedBranchForward",
            "candidate_only_forward_materialized": 0,
            "full_online_row_binding": 0,
            "candidate_probe_only": 0,
            "candidate_count": event_summary["candidate_count"],
            "payload_missing_count": event_summary["candidate_tensor_payload_missing_count"],
            "branch_logits_missing_count": event_summary["candidate_branch_logits_missing_count"],
            "candidate_branch_forward_diagnostic_pass": 0,
            "candidate_branch_forward_full_online_pass": 0,
            "reason": "candidate_x_y_logits_payload_missing_for_full_online_candidate_rows",
            "projection_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
    ]
    summary = {
        "stage": "P3_FULL_ONLINE_CANDIDATE_ONLY_BRANCH_FORWARD",
        "status": "summary",
        "best_branch_forward_id": "BF0-V9269MaterializedMicroprobeDiagnostic",
        "candidate_branch_forward_diagnostic_pass": rows[0]["candidate_branch_forward_diagnostic_pass"],
        "candidate_branch_forward_full_online_pass": 0,
        "candidate_forward_time_ratio": src["candidate_forward_time_ratio"],
        "branch_logit_error_max": src["branch_logit_error_max"],
        "full_online_row_binding": 0,
        "primary_binding_blocker": "candidate_x_y_logits_payload_missing_for_full_online_candidate_rows",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, summary


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


def _p4_true_delta(ctx: Dict[str, Any], event_summary: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    src = _source_v9269_microprobe()
    m = _reference_metrics(ctx)
    decision_gate = int(
        src["exact_agreement"] >= 0.90
        and m["precision"] >= 0.75
        and 0.03 <= m["coverage"] <= 0.15
        and m["bad_event_rate"] <= 0.05
        and m["null_rate"] <= 0.15
        and m["precision_lcb"] >= 0.75
        and m["bad_event_ucb"] <= 0.05
        and src["exact_step_ratio_q90"] <= 1.50
        and src["exact_memory_ratio"] <= 1.05
    )
    row = {
        "stage": "P4_FULL_ONLINE_MATERIALIZED_TRUE_DELTA_EXACT_CONFIRMATION",
        "status": "full_online_exact_candidate_blocked",
        "exact_candidate_id": "TD1-FullOnlinePF5CandidateTrueDelta",
        "diagnostic_source_id": "EC2-MaterializedPF5CandidateTrueDeltaMicroprobe",
        "uses_true_branch_delta": 1,
        "uses_source_measured_gap": 0,
        "uses_formula_proxy": 0,
        "materialized_system_path": 0,
        "projection_used": 0,
        "full_trace_projection_used": 0,
        "full_online_row_binding": 0,
        "candidate_count": event_summary["candidate_count"],
        "candidate_true_delta_logits_missing_count": event_summary["candidate_true_delta_logits_missing_count"],
        "functional_update_payload_missing_count": event_summary["functional_update_payload_missing_count"],
        "agreement_reference_accept": src["exact_agreement"],
        "precision": m["precision"],
        "coverage": m["coverage"],
        "bad_event": m["bad_event_rate"],
        "null_rate": m["null_rate"],
        "precision_lcb": m["precision_lcb"],
        "bad_event_ucb": m["bad_event_ucb"],
        "step_ratio_q90": src["exact_step_ratio_q90"],
        "memory_ratio": src["exact_memory_ratio"],
        "decision_cost_diagnostic_pass": decision_gate,
        "full_online_materialized_true_delta_pass": 0,
        "materialization_blocker": "candidate_true_delta_logits_not_bound_to_full_online_rows",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    summary = {
        "stage": "P4_FULL_ONLINE_MATERIALIZED_TRUE_DELTA_EXACT_CONFIRMATION",
        "status": "summary",
        "best_exact_candidate_id": row["exact_candidate_id"],
        "materialized_system_path": 0,
        "projection_used": 0,
        "full_trace_projection_used": 0,
        "full_online_row_binding": 0,
        "exact_agreement": src["exact_agreement"],
        "exact_step_ratio_q90": src["exact_step_ratio_q90"],
        "exact_memory_ratio": src["exact_memory_ratio"],
        "decision_cost_diagnostic_pass": decision_gate,
        "full_online_materialized_true_delta_pass": 0,
        "primary_binding_blocker": row["materialization_blocker"],
        "precision": m["precision"],
        "coverage": m["coverage"],
        "bad_event": m["bad_event_rate"],
        "null_rate": m["null_rate"],
        "precision_lcb": m["precision_lcb"],
        "bad_event_ucb": m["bad_event_ucb"],
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row, summary], summary


def _p5_bridge_lookup(ctx: Dict[str, Any], p2: Dict[str, Any], p4: Dict[str, Any], device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = ctx["measured"]
    ref_all = set(ctx["c3_accept_all"])
    idx_tensor = p2.get("idx_tensor")
    if idx_tensor is None:
        idx_tensor = torch.empty(0, dtype=torch.long, device=device)
    accept_mask = torch.zeros(len(measured), dtype=torch.uint8, device=device)
    if ref_all:
        accept_mask[torch.as_tensor(sorted(ref_all), dtype=torch.long, device=device)] = 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    gathered = accept_mask[idx_tensor]
    bridge_score = gathered.to(torch.float32)
    checksum = float(bridge_score[: min(32, bridge_score.numel())].detach().sum().cpu()) if bridge_score.numel() else 0.0
    if device.type == "cuda":
        torch.cuda.synchronize()
    bridge_ms = (time.perf_counter() - t0) * 1000.0
    m = _reference_metrics(ctx)
    pass_gate = int(
        _f(p4.get("exact_agreement")) >= 0.90
        and _f(p4.get("exact_step_ratio_q90")) <= 1.50
        and _f(p4.get("exact_memory_ratio")) <= 1.05
        and m["precision"] >= 0.75
        and 0.03 <= m["coverage"] <= 0.15
        and m["bad_event_rate"] <= 0.05
        and m["null_rate"] <= 0.15
        and m["precision_lcb"] >= 0.75
        and m["bad_event_ucb"] <= 0.05
    )
    row = {
        "stage": "P5_FULL_ONLINE_FUSED_COMPACT_BRIDGE_COMPUTE",
        "status": "full_online_bridge_lookup",
        "bridge_system_id": "BR1-FullOnlineFrozenC3LookupTensorGather",
        "exact_candidate_id": p4.get("best_exact_candidate_id"),
        "uses_compact_lookup": 1,
        "uses_cpu_summary": 0,
        "online_frontier_search_used": 0,
        "materialized_system_path": 1,
        "full_online_row_binding": 1,
        "candidate_count": p2.get("candidate_count"),
        "bridge_lookup_time_ms": bridge_ms,
        "bridge_checksum": checksum,
        "precision": m["precision"],
        "coverage": m["coverage"],
        "bad_event": m["bad_event_rate"],
        "null_rate": m["null_rate"],
        "precision_lcb": m["precision_lcb"],
        "bad_event_ucb": m["bad_event_ucb"],
        "agreement_reference_accept": p4.get("exact_agreement"),
        "step_ratio_q90": p4.get("exact_step_ratio_q90"),
        "memory_ratio": p4.get("exact_memory_ratio"),
        "fused_bridge_full_online_pass": pass_gate,
        "projection_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    summary = {
        "stage": "P5_FULL_ONLINE_FUSED_COMPACT_BRIDGE_COMPUTE",
        "status": "summary",
        "best_fused_bridge_id": row["bridge_system_id"],
        "fused_bridge_full_online_pass": pass_gate,
        "bridge_lookup_time_ms": bridge_ms,
        "uses_cpu_summary": 0,
        "online_frontier_search_used": 0,
        "materialized_system_path": 1,
        "full_online_row_binding": 1,
        "fused_bridge_step_ratio_q90": p4.get("exact_step_ratio_q90"),
        "fused_bridge_memory_ratio": p4.get("exact_memory_ratio"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row, summary], summary


def _p6_system_controller(ctx: Dict[str, Any], p1: Dict[str, Any], p2: Dict[str, Any], p4: Dict[str, Any], p5: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    m = _reference_metrics(ctx)
    official = int(
        _i(p1.get("full_online_row_binding_contract_pass"))
        and _i(p2.get("pf5_full_online_selector_pass"))
        and _i(p2.get("candidate_pack_binding_pass"))
        and _i(p4.get("full_online_materialized_true_delta_pass"))
        and _i(p5.get("fused_bridge_full_online_pass"))
    )
    row = {
        "stage": "P6_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER_V3",
        "status": "system_controller" if official else "not_run",
        "controller_id": "C3-T2PlusBackfill",
        "official_eligible": official,
        "system_legal_controller_pass": official,
        "reason": "" if official else "P1_or_P4_full_online_binding_failed",
        "prefilter_id": p2.get("best_prefilter_id", "PF5-LearnedMonotoneCheapPrefilter"),
        "exact_candidate_id": p4.get("best_exact_candidate_id"),
        "bridge_system_id": p5.get("best_fused_bridge_id"),
        "precision_heldout": m["precision"],
        "coverage_heldout": m["coverage"],
        "bad_event_heldout": m["bad_event_rate"],
        "null_rate_heldout": m["null_rate"],
        "precision_lcb": m["precision_lcb"],
        "bad_event_ucb": m["bad_event_ucb"],
        "accepted_signal_strata_count": m["accepted_signal_strata_count"],
        "accepted_family_count": m["accepted_family_count"],
        "max_family_share": m["max_family_share"],
        "max_stratum_share": m["max_stratum_share"],
        "agreement_reference_accept": p4.get("exact_agreement"),
        "candidate_rate": p2.get("candidate_rate"),
        "step_ratio_q90": p4.get("exact_step_ratio_q90"),
        "memory_ratio": p4.get("exact_memory_ratio"),
        "materialized_system_path": int(_i(p4.get("full_online_materialized_true_delta_pass")) and _i(p5.get("fused_bridge_full_online_pass"))),
        "full_online_row_binding": p1.get("full_online_row_binding"),
        "projection_used": 0,
        "dataset_name_used": 0,
        "posthoc_used_at_commit": 0,
        "validation_used": 0,
        "test_used": 0,
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
        "p1_full_online_binding_matrix.svg",
        "p2_pf5_full_online_selector.svg",
        "p3_branch_forward_binding_gap.svg",
        "p4_true_delta_binding_gap.svg",
        "p5_bridge_lookup_binding.svg",
        "p6_system_controller_boundary.svg",
    ]:
        (fig / name).write_text(
            f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"640\" height=\"90\"><text x=\"8\" y=\"48\">v9.2.70 artifact: {name}</text></svg>\n",
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
    p2_v9269_rows, p2_v9269 = v9269._materialized_pf5(ctx, device)
    full_event_table, event_summary = _build_full_event_table(ctx, p2_v9269)
    p1_rows, p1 = _p1_binding_audit(event_summary)
    p2_rows, p2 = _p2_selector_binding(ctx, p2_v9269, event_summary)
    p3_rows, p3 = _p3_branch_forward(event_summary)
    p4_rows, p4 = _p4_true_delta(ctx, event_summary)
    p5_rows, p5 = _p5_bridge_lookup(ctx, p2_v9269, p4, device)
    p6_rows, p6 = _p6_system_controller(ctx, p1, p2, p4, p5)

    if not _i(p0.get("v9269_boundary_pass")):
        route_name, blocker, failure_code, reason, next_required = "R13-ReferenceControllerUnstable", "v9269_boundary_unstable", "F2_v9269_boundary_unstable", "P0_v9269_boundary_failed", "reproduce_v9269_boundary"
    elif not _i(p1.get("full_online_row_binding_contract_pass")):
        route_name, blocker, failure_code, reason, next_required = "R14-FullOnlineBindingFail", "candidate_tensor_payload_missing_for_full_online_rows", "F27_full_online_row_binding_fail", "P1_full_online_binding_contract_failed", "persist_candidate_x_y_logits_and_update_payload_for_full_online_rows"
    elif not _i(p2.get("candidate_pack_binding_pass")):
        route_name, blocker, failure_code, reason, next_required = "R16-CandidatePackBindingFail", "candidate_pack_payload_unbound", "F10_candidate_pack_binding_fail", "P2_candidate_pack_binding_failed", "bind_candidate_payload_tensors"
    elif not _i(p3.get("candidate_branch_forward_full_online_pass")):
        route_name, blocker, failure_code, reason, next_required = "R17-CandidateForwardFullOnlineTooExpensive", "candidate_branch_forward_full_online_fail", "F12_candidate_forward_not_candidate_only", "P3_candidate_forward_failed", "materialize_full_candidate_branch_forward"
    elif not _i(p4.get("full_online_materialized_true_delta_pass")):
        route_name, blocker, failure_code, reason, next_required = "R18-MaterializedTrueDeltaFullOnlineTooExpensive", "true_delta_full_online_binding_fail", "F13_materialized_true_delta_system_fail", "P4_full_online_true_delta_failed", "bind_true_delta_to_full_online_rows"
    elif not _i(p5.get("fused_bridge_full_online_pass")):
        route_name, blocker, failure_code, reason, next_required = "R19-FrozenBridgeBindingFail", "frozen_bridge_binding_fail", "F15_frozen_bridge_lookup_not_materialized", "P5_bridge_binding_failed", "repair_frozen_bridge_binding"
    elif not _i(p6.get("system_legal_controller_pass")):
        route_name, blocker, failure_code, reason, next_required = "R20-SystemControllerStillBlocked", "system_controller_not_official", "F17_system_controller_precision_fail", "P6_system_controller_failed", "repair_system_controller"
    else:
        route_name, blocker, failure_code, reason, next_required = "R7-SystemLegalExactSignalControllerPass", "leaveout_not_executed", "F25_leave_dataset_out_fail", "P7_not_executed", "run_leaveout_and_paired_replay"

    downstream = _downstream(reason)
    artifacts: Dict[str, List[Dict[str, Any]]] = {
        "p0_v9269_boundary_reproduction.csv": [p0],
        "p1_full_online_row_binding_contract_audit.csv": p1_rows,
        "p2_full_online_event_table_pf5_selector_binding.csv": p2_rows,
        "p3_full_online_candidate_only_branch_forward.csv": p3_rows,
        "p4_full_online_materialized_true_delta_exact_confirmation.csv": p4_rows,
        "p5_full_online_fused_compact_bridge_compute.csv": p5_rows,
        "p6_system_legal_exact_signal_controller_v3.csv": p6_rows,
        **downstream,
        "full_online_event_table_v9270.csv": full_event_table,
        "full_online_binding_trace_v9270.csv": p1_rows,
        "pf5_full_online_selector_trace_v9270.csv": p2_rows,
        "candidate_pack_full_online_trace_v9270.csv": p2_v9269_rows + p2_rows,
        "candidate_branch_forward_full_online_trace_v9270.csv": p3_rows,
        "full_online_true_delta_trace_v9270.csv": p4_rows,
        "full_online_bridge_lookup_trace_v9270.csv": p5_rows,
        "system_controller_trace_v9270.csv": p6_rows,
        "leaveout_trace_v9270.csv": downstream["p7_leave_dataset_and_stratum_out.csv"],
        "paired_replay_branch_trace_v9270.csv": downstream["p8_official_paired_replay.csv"],
        "short_run_trace_v9270.csv": downstream["p9_short_run_functional_validation.csv"],
    }
    for name, rows_out in artifacts.items():
        write_csv_rows(out_dir / name, rows_out)
    audit = audit_no_fake([out_dir / name for name in artifacts])
    write_csv_rows(out_dir / "v9270_provenance_audit.csv", [audit])

    m = _reference_metrics(ctx)
    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9269_boundary_pass": p0.get("v9269_boundary_pass"),
        "reference_controller_id": "C3-T2PlusBackfill",
        "reference_controller_reproduced": 1,
        "reference_precision": m["precision"],
        "reference_coverage": m["coverage"],
        "reference_bad_event": m["bad_event_rate"],
        "reference_null_rate": m["null_rate"],
        "reference_precision_lcb": m["precision_lcb"],
        "reference_bad_event_ucb": m["bad_event_ucb"],
        "full_online_row_binding_contract_pass": p1.get("full_online_row_binding_contract_pass"),
        "full_online_row_binding": p1.get("full_online_row_binding"),
        "event_identity_binding_pass": p1.get("event_identity_binding_pass"),
        "candidate_tensor_payload_missing_count": p1.get("candidate_tensor_payload_missing_count"),
        "candidate_branch_logits_missing_count": p1.get("candidate_branch_logits_missing_count"),
        "candidate_true_delta_logits_missing_count": p1.get("candidate_true_delta_logits_missing_count"),
        "functional_update_payload_missing_count": p1.get("functional_update_payload_missing_count"),
        "best_prefilter_id": p2.get("prefilter_id", "PF5-LearnedMonotoneCheapPrefilter"),
        "pf5_full_online_selector_pass": p2.get("pf5_full_online_selector_pass"),
        "candidate_pack_binding_pass": p2.get("candidate_pack_binding_pass"),
        "candidate_rate": p2.get("candidate_rate"),
        "reference_accept_recall": p2.get("reference_accept_recall"),
        "candidate_count": p2.get("candidate_count"),
        "event_count": p2.get("event_count"),
        "best_branch_forward_id": p3.get("best_branch_forward_id"),
        "candidate_branch_forward_diagnostic_pass": p3.get("candidate_branch_forward_diagnostic_pass"),
        "candidate_branch_forward_full_online_pass": p3.get("candidate_branch_forward_full_online_pass"),
        "candidate_forward_time_ratio": p3.get("candidate_forward_time_ratio"),
        "branch_logit_error_max": p3.get("branch_logit_error_max"),
        "best_exact_candidate_id": p4.get("best_exact_candidate_id"),
        "full_online_materialized_true_delta_pass": p4.get("full_online_materialized_true_delta_pass"),
        "decision_cost_diagnostic_pass": p4.get("decision_cost_diagnostic_pass"),
        "materialized_system_path": p4.get("materialized_system_path"),
        "projection_used": p4.get("projection_used"),
        "full_trace_projection_used": p4.get("full_trace_projection_used"),
        "source_measured_gap_used": 0,
        "formula_proxy_used": 0,
        "exact_agreement": p4.get("exact_agreement"),
        "exact_step_ratio_q90": p4.get("exact_step_ratio_q90"),
        "exact_memory_ratio": p4.get("exact_memory_ratio"),
        "best_fused_bridge_id": p5.get("best_fused_bridge_id"),
        "fused_bridge_full_online_pass": p5.get("fused_bridge_full_online_pass"),
        "fused_bridge_step_ratio_q90": p5.get("fused_bridge_step_ratio_q90"),
        "fused_bridge_memory_ratio": p5.get("fused_bridge_memory_ratio"),
        "uses_cpu_summary": p5.get("uses_cpu_summary"),
        "online_frontier_search_used": p5.get("online_frontier_search_used"),
        "system_legal_controller_pass": p6.get("system_legal_controller_pass", 0),
        "official_eligible": p6.get("official_eligible", 0),
        "controller_precision": p6.get("precision_heldout", m["precision"]),
        "controller_coverage": p6.get("coverage_heldout", m["coverage"]),
        "controller_bad_event": p6.get("bad_event_heldout", m["bad_event_rate"]),
        "controller_null_rate": p6.get("null_rate_heldout", m["null_rate"]),
        "controller_precision_lcb": p6.get("precision_lcb", m["precision_lcb"]),
        "controller_bad_event_ucb": p6.get("bad_event_ucb", m["bad_event_ucb"]),
        "accepted_signal_strata_count": m["accepted_signal_strata_count"],
        "accepted_family_count": m["accepted_family_count"],
        "max_family_share": m["max_family_share"],
        "max_stratum_share": m["max_stratum_share"],
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
        "success_v9270_strict_purekan_functional": 0,
        "success_v9270_full_functional": 0,
        "success_v9270_external_ready": 0,
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
    write_csv_rows(out_dir / "contract_audit_v9270.csv", [{
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "full_online_event_table": 1,
        "materialized_pf5_selector": 1,
        "candidate_pack_metadata_materialized": 1,
        "candidate_tensor_payload_bound": event_summary["candidate_pack_payload_bound"],
        "candidate_only_branch_forward_full_online": p3.get("candidate_branch_forward_full_online_pass"),
        "materialized_true_delta_full_online": p4.get("full_online_materialized_true_delta_pass"),
        "frozen_bridge_lookup_materialized": p5.get("fused_bridge_full_online_pass"),
        "full_online_row_binding": p1.get("full_online_row_binding"),
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
        "triton_available": bool(getattr(v9268.v9256, "TRITON_AVAILABLE", False)),
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
    write_csv_rows(out_dir / "artifact_hashes_v9270.csv", [
        {"artifact": "plan", "sha256": sha256_file(PLAN_PATH)},
        {"artifact": "runner", "sha256": sha256_file(SCRIPT_PATH)},
        *[
            {"artifact": path.name, "sha256": sha256_file(path)}
            for path in sorted(out_dir.glob("*.csv")) + sorted(out_dir.glob("*.json"))
            if path.name != "artifact_hashes_v9270.csv"
        ],
    ])
    print(json.dumps(route, indent=2, sort_keys=True))
    return route


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(RESULT_ROOT / "v9270_full_online_row_binding_system_legal_controller_closure_first_20260513T010000Z"))
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
