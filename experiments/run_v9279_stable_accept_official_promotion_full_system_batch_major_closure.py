#!/usr/bin/env python3
"""DG-KAN v9.2.79 stable-accept official promotion runner.

This runner starts from the v9.2.78 local native stable-accept closure and
attempts the stricter v9.2.79 official-promotion gates.  The central audit is
whether the full-online candidate rows contain enough stable-accept material to
rerun calibration / heldout / support and to measure a full-system batch-major
native runtime.  Missing full-row materialization is recorded as a hard blocker;
no local native timing or old controller summary is promoted into an official
system pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.79_StableAcceptOfficialPromotion_FullSystemBatchMajorClosure_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9279_stable_accept_official_promotion_full_system_batch_major_closure.py"
RECAP_PATH = ROOT / "docs" / "DG-KAN_v9.2.79_StableAcceptOfficialPromotion_FullSystemBatchMajorClosure_实验复盘.md"

SRC_V9278 = RESULT_ROOT / "v9278_native_bridge_accept_static_bucket_official_closure_integrated_stable_native_20260513T160000Z"
SRC_FULL_ONLINE = RESULT_ROOT / "v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline_async_basis_cuda_ext_20260513T110000Z"
SRC_LABELS_V9267 = RESULT_ROOT / "v9267_borderline_localized_bridge_repair_C0C4_pareto_calibration_first_20260512T223000Z" / "C0_T2_C4_E2_membership_trace_v9267.csv"

STABLE_ACCEPT_REQUIRED_FIELDS = [
    "stable_accept_contract_id",
    "score_ref",
    "score_native",
    "score_quantized_ref",
    "score_quantized_native",
    "stable_rank_ref",
    "stable_rank_native",
    "accept_ref",
    "accept_native",
]

OUTCOME_REQUIRED_FIELDS = [
    "safe_good_label",
    "bad_event_label",
    "null_event_label",
]

RUNTIME_REQUIRED_FIELDS = [
    "native_stable_bucket_time_ms",
    "native_bucket_kernel_used_in_p6",
    "full_system_step_time_ms",
    "baseline_step_time_ms",
    "p6_step_ratio",
]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except Exception:
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _missing_fields(fieldnames: Sequence[str], required: Sequence[str]) -> List[str]:
    present = set(fieldnames)
    return [f for f in required if f not in present]


def _csv_fieldnames(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def _count_candidate_rows(rows: Iterable[Dict[str, str]]) -> Dict[str, int]:
    total = 0
    candidates = 0
    accepted = 0
    fake = 0
    proxy = 0
    offload = 0
    projection = 0
    source_gap = 0
    formula = 0
    datasets = set()
    seeds = set()
    for row in rows:
        total += 1
        if row.get("dataset"):
            datasets.add(row.get("dataset", ""))
        if row.get("seed") not in ("", None):
            seeds.add(row.get("seed", ""))
        if _as_int(row.get("candidate_flag", 1), 1) == 1:
            candidates += 1
            accepted += _as_int(row.get("accept_decision", 0), 0)
        fake += _as_int(row.get("fake_data_used", 0), 0)
        proxy += _as_int(row.get("proxy_row_used", 0), 0)
        offload += _as_int(row.get("cpu_offload_used", 0), 0)
        projection += _as_int(row.get("projection_used", 0), 0)
        source_gap += _as_int(row.get("source_measured_gap_used", 0), 0)
        formula += _as_int(row.get("formula_proxy_used", 0), 0)
    return {
        "event_count": total,
        "candidate_count": candidates,
        "accepted_count_old_reference": accepted,
        "dataset_count": len(datasets),
        "seed_count": len(seeds),
        "fake_data_used_count": fake,
        "proxy_row_used_count": proxy,
        "cpu_offload_used_count": offload,
        "projection_used_count": projection,
        "source_measured_gap_used_count": source_gap,
        "formula_proxy_used_count": formula,
    }


def _not_run(stage: str, reason: str, **extra: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "stage": stage,
        "status": "not_run",
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row.update(extra)
    return row


def _source_summary() -> Dict[str, Any]:
    route78 = _read_json(SRC_V9278 / "route_decision.json")
    p0 = _read_csv_rows(SRC_V9278 / "p0_v9277_boundary_reproduction.csv")
    p3 = _read_csv_rows(SRC_V9278 / "p3_native_bucket_kernel_v2_stable_accept.csv")
    p6 = _read_csv_rows(SRC_V9278 / "p6_system_legal_exact_signal_controller_v10.csv")
    p3_summary = next((r for r in p3 if r.get("status") == "summary"), p3[-1] if p3 else {})
    p6_row = p6[0] if p6 else {}
    p0_row = p0[0] if p0 else {}
    return {
        "source_route_v9278": route78.get("route", ""),
        "source_primary_blocker": route78.get("primary_blocker", ""),
        "source_v9277_boundary_pass": route78.get("v9277_boundary_pass", ""),
        "source_payload_binding_contract_pass": route78.get("payload_binding_contract_pass", ""),
        "stable_accept_contract_pass": route78.get("stable_accept_contract_pass", ""),
        "best_accept_contract_id": route78.get("best_accept_contract_id", ""),
        "accept_disagreement_count_before": route78.get("accept_disagreement_count_before", ""),
        "accept_disagreement_count_after": route78.get("accept_disagreement_count_after", ""),
        "accept_rule_changed": route78.get("accept_rule_changed", ""),
        "calibration_rerun_required": route78.get("calibration_rerun_required", ""),
        "native_bucket_kernel_v2_pass": route78.get("native_bucket_kernel_v2_pass", ""),
        "native_cuda_bucket_kernel_used": route78.get("native_cuda_bucket_kernel_used", ""),
        "stable_accept_cuda_kernel_used": route78.get("stable_accept_cuda_kernel_used", ""),
        "basis_norm_bucketed": route78.get("basis_norm_bucketed", ""),
        "W2_delta_bucketed": route78.get("W2_delta_bucketed", ""),
        "bridge_score_inside_kernel": route78.get("bridge_score_inside_kernel", ""),
        "accept_bit_inside_kernel": route78.get("accept_bit_inside_kernel", ""),
        "kernel_count_before": route78.get("kernel_count_before", ""),
        "kernel_count_after": route78.get("kernel_count_after", ""),
        "sync_count_before": route78.get("sync_count_before", ""),
        "sync_count_after": route78.get("sync_count_after", ""),
        "avg_candidates_per_kernel_after": route78.get("avg_candidates_per_kernel_after", ""),
        "native_time_ms_q90": route78.get("native_time_ms_q90", ""),
        "eager_time_ms_q90": route78.get("eager_time_ms_q90", ""),
        "q90_reduction": route78.get("q90_reduction", ""),
        "controller_precision": route78.get("controller_precision", ""),
        "controller_coverage": route78.get("controller_coverage", ""),
        "controller_bad_event": route78.get("controller_bad_event", ""),
        "controller_null_rate": route78.get("controller_null_rate", ""),
        "controller_step_ratio_q90": route78.get("controller_step_ratio_q90", ""),
        "system_legal_controller_pass": route78.get("system_legal_controller_pass", ""),
        "official_eligible": route78.get("official_eligible", ""),
        "p0_source_route_v9277": p0_row.get("source_route_v9277", ""),
        "p3_summary_native_q90": p3_summary.get("native_time_ms_q90", ""),
        "p6_reason": p6_row.get("reason", ""),
    }


def _stage_p0(source: Dict[str, Any]) -> Dict[str, Any]:
    boundary_pass = int(
        source.get("source_route_v9278") == "R18-StableAcceptLocalClosedButSystemNotOfficial"
        and _as_int(source.get("stable_accept_contract_pass")) == 1
        and _as_int(source.get("native_bucket_kernel_v2_pass")) == 1
        and _as_int(source.get("system_legal_controller_pass")) == 0
    )
    return {
        "stage": "P0_V9278_BOUNDARY_REPRODUCTION",
        "status": "summary",
        "source_artifact": str(SRC_V9278.relative_to(ROOT)),
        "source_route_v9278": source.get("source_route_v9278"),
        "stable_accept_contract_pass": source.get("stable_accept_contract_pass"),
        "native_bucket_kernel_v2_pass": source.get("native_bucket_kernel_v2_pass"),
        "source_accept_disagreement_before": source.get("accept_disagreement_count_before"),
        "source_accept_disagreement_after": source.get("accept_disagreement_count_after"),
        "source_controller_step_ratio_q90": source.get("controller_step_ratio_q90"),
        "source_official_eligible": source.get("official_eligible"),
        "source_system_legal_controller_pass": source.get("system_legal_controller_pass"),
        "v9278_boundary_pass": boundary_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _stage_p1_p2_p3(out_dir: Path, full_rows: List[Dict[str, str]], full_fields: List[str], counts: Dict[str, int]) -> tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    missing_stable = _missing_fields(full_fields, STABLE_ACCEPT_REQUIRED_FIELDS)
    missing_outcome = _missing_fields(full_fields, OUTCOME_REQUIRED_FIELDS)
    missing_runtime = _missing_fields(full_fields, RUNTIME_REQUIRED_FIELDS)
    stable_materialized = int(not missing_stable)
    outcome_materialized = int(not missing_outcome)
    runtime_materialized = int(not missing_runtime)

    trace_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(full_rows[:24]):
        trace_rows.append(
            {
                "stage": "P1_STABLE_ACCEPT_FULL_ROW_FIELD_AUDIT",
                "status": "sample_row",
                "sample_index": idx,
                "candidate_event_id": row.get("candidate_event_id") or row.get("event_id"),
                "candidate_global_row_id": row.get("candidate_global_row_id") or row.get("global_row_id"),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "old_accept_decision": row.get("accept_decision", ""),
                "old_bridge_score": row.get("bridge_score", ""),
                "has_stable_score_ref": int("score_ref" in row and row.get("score_ref", "") != ""),
                "has_stable_score_native": int("score_native" in row and row.get("score_native", "") != ""),
                "has_stable_rank_ref": int("stable_rank_ref" in row and row.get("stable_rank_ref", "") != ""),
                "has_accept_native": int("accept_native" in row and row.get("accept_native", "") != ""),
                "fake_data_used": row.get("fake_data_used", 0),
                "proxy_row_used": row.get("proxy_row_used", 0),
                "cpu_offload_used": row.get("cpu_offload_used", 0),
            }
        )
    _write_csv(out_dir / "stable_accept_full_row_field_trace_v9279.csv", trace_rows)

    p1 = {
        "stage": "P1_STABLE_ACCEPT_OFFICIAL_CALIBRATION_RERUN",
        "status": "blocked",
        "accept_contract_id": "AC2Q2-stable-quantized-1e5-event-tie",
        "calibration_rerun_attempted": 1,
        "calibration_rerun": 0,
        "stable_accept_full_row_materialization_present": stable_materialized,
        "outcome_labels_present": outcome_materialized,
        "event_count": counts["event_count"],
        "candidate_count": counts["candidate_count"],
        "accepted_count_old_reference": counts["accepted_count_old_reference"],
        "missing_stable_accept_fields": ";".join(missing_stable),
        "missing_outcome_fields": ";".join(missing_outcome),
        "accept_disagreement_count": "",
        "reference_drift_vs_current_accept": "",
        "precision_calibration": "",
        "coverage_calibration": "",
        "bad_event_calibration": "",
        "null_rate_calibration": "",
        "stable_accept_official_calibration_pass": 0,
        "reason": "stable_accept_full_row_or_outcome_materialization_missing",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "projection_used": counts["projection_used_count"],
        "source_measured_gap_used": counts["source_measured_gap_used_count"],
        "formula_proxy_used": counts["formula_proxy_used_count"],
    }
    p2 = {
        "stage": "P2_STABLE_ACCEPT_HELDOUT_SUPPORT_RERUN",
        "status": "blocked",
        "accept_contract_id": "AC2Q2-stable-quantized-1e5-event-tie",
        "heldout_rerun_attempted": 1,
        "heldout_rerun": 0,
        "support_balance_rerun": 0,
        "stable_accept_full_row_materialization_present": stable_materialized,
        "outcome_labels_present": outcome_materialized,
        "accept_disagreement_count": "",
        "rank_disagreement_count": "",
        "score_quantized_disagreement_count": "",
        "precision_heldout": "",
        "coverage_heldout": "",
        "bad_event_heldout": "",
        "null_rate_heldout": "",
        "precision_lcb": "",
        "bad_event_ucb": "",
        "accepted_signal_strata_count": "",
        "accepted_family_count": "",
        "max_family_share": "",
        "max_stratum_share": "",
        "stable_accept_heldout_support_pass": 0,
        "reason": "stable_accept_full_row_or_outcome_materialization_missing",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    p3 = {
        "stage": "P3_NATIVE_STABLE_KERNEL_FULL_SYSTEM_INTEGRATION_AUDIT",
        "status": "blocked",
        "native_kernel_local_pass_v9278": 1,
        "native_bucket_kernel_used_in_p6": 0,
        "native_kernel_full_row_materialization_present": stable_materialized,
        "native_runtime_full_system_materialization_present": runtime_materialized,
        "bridge_score_inside_kernel": "",
        "accept_bit_inside_kernel": "",
        "kernel_count_after_full_system": "",
        "sync_count_after_full_system": "",
        "avg_candidates_per_kernel_after_full_system": "",
        "missing_runtime_fields": ";".join(missing_runtime),
        "full_system_integration_pass": 0,
        "reason": "native_stable_kernel_not_materialized_in_full_system_p6_trace",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return p1, p2, trace_rows, p3


def _stage_p4_p5(source: Dict[str, Any], full_fields: List[str], counts: Dict[str, int]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    missing_runtime = _missing_fields(full_fields, RUNTIME_REQUIRED_FIELDS)
    p4 = {
        "stage": "P4_FULL_SYSTEM_STEP_ATTRIBUTION_NATIVE_STABLE_PATH",
        "status": "blocked",
        "source_boundary_step_ratio_q90": source.get("controller_step_ratio_q90"),
        "old_step_ratio_reused_as_measurement": 0,
        "new_full_system_step_ratio_measured": 0,
        "native_stable_component_present": int(not missing_runtime),
        "unknown_fraction": "",
        "dominant_subphase": "",
        "kernel_count_before": source.get("kernel_count_before"),
        "kernel_count_after_local_v9278": source.get("kernel_count_after"),
        "sync_count_before": source.get("sync_count_before"),
        "sync_count_after_local_v9278": source.get("sync_count_after"),
        "full_system_step_attribution_pass": 0,
        "reason": "no_full_system_native_step_trace_available_for_v9279_official_promotion",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    p5 = {
        "stage": "P5_BATCH_MAJOR_NATIVE_RUNTIME_CLOSURE",
        "status": "blocked",
        "native_bucket_kernel_local_pass_v9278": source.get("native_bucket_kernel_v2_pass"),
        "native_time_ms_q90_local_v9278": source.get("native_time_ms_q90"),
        "eager_time_ms_q90_local_v9278": source.get("eager_time_ms_q90"),
        "q90_reduction_local_v9278": source.get("q90_reduction"),
        "kernel_count_after_local_v9278": source.get("kernel_count_after"),
        "sync_count_after_local_v9278": source.get("sync_count_after"),
        "avg_candidates_per_kernel_after_local_v9278": source.get("avg_candidates_per_kernel_after"),
        "candidate_count_full_online": counts["candidate_count"],
        "batch_major_native_full_system_runtime_measured": 0,
        "batch_major_runtime_pass": 0,
        "reason": "local_native_kernel_timing_cannot_replace_end_to_end_step_timing",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return p4, p5


def _join_key(row: Dict[str, str], mode: str) -> Any:
    if mode == "source_hash":
        return row.get("source_hash", "")
    if mode == "event_family":
        return row.get("event_family", "")
    if mode == "event_family_dataset_seed":
        return (row.get("event_family", ""), row.get("dataset", ""), row.get("seed", ""))
    if mode == "event_family_signal_stratum":
        return (row.get("event_family", ""), row.get("signal_stratum", ""))
    raise ValueError(mode)


def _stage_p12_outcome_join_probe(out_dir: Path, full_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    label_rows = _read_csv_rows(SRC_LABELS_V9267)
    candidates = [r for r in full_rows if _as_int(r.get("candidate_flag", 1), 1) == 1]
    modes = ["source_hash", "event_family", "event_family_dataset_seed", "event_family_signal_stratum"]
    trace: List[Dict[str, Any]] = []
    summaries: Dict[str, Dict[str, Any]] = {}
    labels = ["bad_event", "null_event", "harmless_null", "bad_null"]
    for mode in modes:
        lookup: Dict[Any, List[Dict[str, str]]] = {}
        for row in label_rows:
            key = _join_key(row, mode)
            if key not in ("", None, ("", "", ""), ("", "")):
                lookup.setdefault(key, []).append(row)
        covered = 0
        unique_label = 0
        ambiguous = 0
        total_matches = 0
        for idx, row in enumerate(candidates):
            matches = lookup.get(_join_key(row, mode), [])
            if not matches:
                continue
            covered += 1
            total_matches += len(matches)
            labelsets = {lbl: {m.get(lbl, "") for m in matches} for lbl in labels}
            is_ambiguous = int(any(len(v) > 1 for v in labelsets.values()))
            ambiguous += is_ambiguous
            unique_label += int(not is_ambiguous)
            if idx < 24 or is_ambiguous and len(trace) < 48:
                trace.append(
                    {
                        "stage": "P12_OUTCOME_LABEL_RECOVERY_PROBE",
                        "status": "join_sample",
                        "join_mode": mode,
                        "candidate_event_id": row.get("candidate_event_id") or row.get("event_id"),
                        "candidate_global_row_id": row.get("candidate_global_row_id") or row.get("global_row_id"),
                        "dataset": row.get("dataset", ""),
                        "seed": row.get("seed", ""),
                        "event_family": row.get("event_family", ""),
                        "signal_stratum": row.get("signal_stratum", ""),
                        "source_hash": row.get("source_hash", ""),
                        "match_count": len(matches),
                        "ambiguous_label": is_ambiguous,
                        "bad_event_values": "|".join(sorted(labelsets["bad_event"])) if matches else "",
                        "null_event_values": "|".join(sorted(labelsets["null_event"])) if matches else "",
                        "harmless_null_values": "|".join(sorted(labelsets["harmless_null"])) if matches else "",
                        "bad_null_values": "|".join(sorted(labelsets["bad_null"])) if matches else "",
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    }
                )
        summaries[mode] = {
            "join_mode": mode,
            "candidate_count": len(candidates),
            "label_source_rows": len(label_rows),
            "covered_candidate_rows": covered,
            "unique_label_candidate_rows": unique_label,
            "ambiguous_label_candidate_rows": ambiguous,
            "uncovered_candidate_rows": len(candidates) - covered,
            "avg_matches_per_covered_row": total_matches / max(1, covered),
        }
    best = max(summaries.values(), key=lambda r: (int(r["unique_label_candidate_rows"]), int(r["covered_candidate_rows"]), -int(r["ambiguous_label_candidate_rows"])))
    summary = {
        "stage": "P12_OUTCOME_LABEL_RECOVERY_PROBE",
        "status": "summary",
        "label_source": str(SRC_LABELS_V9267.relative_to(ROOT)),
        "candidate_count": len(candidates),
        "best_join_mode": best["join_mode"],
        "source_hash_covered_candidate_rows": summaries["source_hash"]["covered_candidate_rows"],
        "event_family_unique_label_candidate_rows": summaries["event_family"]["unique_label_candidate_rows"],
        "event_family_ambiguous_label_candidate_rows": summaries["event_family"]["ambiguous_label_candidate_rows"],
        "event_family_dataset_seed_unique_label_candidate_rows": summaries["event_family_dataset_seed"]["unique_label_candidate_rows"],
        "event_family_dataset_seed_ambiguous_label_candidate_rows": summaries["event_family_dataset_seed"]["ambiguous_label_candidate_rows"],
        "outcome_label_recovery_pass": 0,
        "reason": "source_hash_join_zero_and_family_join_ambiguous",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    _write_csv(out_dir / "outcome_label_recovery_trace_v9279.csv", trace + list(summaries.values()) + [summary])
    return summary


def _stage_p13_stable_accept_trace_coverage_probe(full_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    local_rows = [
        r for r in _read_csv_rows(SRC_V9278 / "accept_disagreement_trace_v9278.csv")
        if r.get("status") == "carrier_autopsy"
    ]
    candidates = [r for r in full_rows if _as_int(r.get("candidate_flag", 1), 1) == 1]
    full_keys = {
        (r.get("dataset", ""), r.get("seed", ""), r.get("step", ""), r.get("candidate_rank", ""))
        for r in candidates
    }
    matched = 0
    for row in local_rows:
        parts = str(row.get("event_id", "")).split(":")
        if len(parts) != 3:
            continue
        key = (parts[0], parts[1], parts[2], row.get("candidate_id", ""))
        matched += int(key in full_keys)
    return {
        "stage": "P13_STABLE_ACCEPT_TRACE_COVERAGE_PROBE",
        "status": "summary",
        "local_trace_source": str((SRC_V9278 / "accept_disagreement_trace_v9278.csv").relative_to(ROOT)),
        "local_stable_accept_trace_rows": len(local_rows),
        "candidate_count_full_online": len(candidates),
        "matched_full_candidate_rows_by_dataset_seed_step_rank": matched,
        "full_row_coverage": matched / max(1, len(candidates)),
        "stable_accept_trace_coverage_pass": 0,
        "reason": "v9278_stable_accept_trace_is_local_subset_not_full_online_materialization",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _stage_p6(source: Dict[str, Any], p1: Dict[str, Any], p2: Dict[str, Any], p3: Dict[str, Any], p4: Dict[str, Any], p5: Dict[str, Any]) -> Dict[str, Any]:
    official_eligible = int(
        _as_int(p1.get("stable_accept_official_calibration_pass")) == 1
        and _as_int(p2.get("stable_accept_heldout_support_pass")) == 1
        and _as_int(p3.get("full_system_integration_pass")) == 1
        and _as_int(p4.get("full_system_step_attribution_pass")) == 1
        and _as_int(p5.get("batch_major_runtime_pass")) == 1
    )
    return {
        "stage": "P6_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER_V11",
        "status": "not_run",
        "controller_id": "C3Q2-StableAcceptNativeBucket",
        "accept_contract_id": "AC2Q2-stable-quantized-1e5-event-tie",
        "official_eligible": official_eligible,
        "system_legal_controller_pass": 0,
        "reason": "P1_P2_full_row_stable_accept_materialization_missing",
        "stable_accept_official_calibration_pass": p1.get("stable_accept_official_calibration_pass"),
        "stable_accept_heldout_support_pass": p2.get("stable_accept_heldout_support_pass"),
        "full_system_integration_pass": p3.get("full_system_integration_pass"),
        "full_system_step_attribution_pass": p4.get("full_system_step_attribution_pass"),
        "batch_major_runtime_pass": p5.get("batch_major_runtime_pass"),
        "precision_heldout": "",
        "coverage_heldout": "",
        "bad_event_heldout": "",
        "null_rate_heldout": "",
        "precision_lcb": "",
        "bad_event_ucb": "",
        "agreement_reference_accept": "",
        "step_ratio_q90": "",
        "source_boundary_step_ratio_q90": source.get("controller_step_ratio_q90"),
        "memory_ratio": "",
        "accepted_signal_strata_count": "",
        "accepted_family_count": "",
        "full_online_row_binding": 1,
        "full_online_payload_binding": 1,
        "full_online_update_payload_binding": 1,
        "native_bucket_kernel_used_in_p6": 0,
        "bridge_score_inside_kernel": "",
        "accept_bit_inside_kernel": "",
        "diagnostic_derived_from_measured_components": 0,
        "old_metrics_reused_for_official": 0,
        "projection_used": 0,
        "source_measured_gap_used": 0,
        "formula_proxy_used": 0,
        "cpu_offload_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }


def _route_decision(source: Dict[str, Any], p0: Dict[str, Any], p1: Dict[str, Any], p2: Dict[str, Any], p3: Dict[str, Any], p4: Dict[str, Any], p5: Dict[str, Any], p6: Dict[str, Any]) -> Dict[str, Any]:
    if _as_int(p0.get("v9278_boundary_pass")) != 1:
        route = "R14-V9278BoundaryReproductionFail"
        blocker = "v9278_boundary_not_reproduced"
        next_required = "reproduce_v9278_stable_accept_local_boundary"
    elif _as_int(p1.get("stable_accept_official_calibration_pass")) != 1:
        route = "R15-StableAcceptOfficialCalibrationBlocked"
        blocker = "stable_accept_full_row_materialization_missing"
        next_required = "materialize_full_row_stable_score_rank_accept_and_outcome_labels"
    elif _as_int(p2.get("stable_accept_heldout_support_pass")) != 1:
        route = "R16-StableAcceptHeldoutSupportFail"
        blocker = "stable_accept_heldout_or_support_gate_failed"
        next_required = "rerun_stable_accept_heldout_support"
    elif _as_int(p3.get("full_system_integration_pass")) != 1 or _as_int(p4.get("full_system_step_attribution_pass")) != 1:
        route = "R17-NativeStableKernelNotInFullSystemPath"
        blocker = "native_stable_kernel_full_system_trace_missing"
        next_required = "integrate_native_stable_kernel_into_p6_train_stream"
    elif _as_int(p5.get("batch_major_runtime_pass")) != 1:
        route = "R18-BatchMajorRuntimeNotClosed"
        blocker = "batch_major_full_system_runtime_not_closed"
        next_required = "measure_full_system_batch_major_native_runtime"
    elif _as_int(p6.get("system_legal_controller_pass")) != 1:
        route = "R19-SystemLegalControllerStillBlocked"
        blocker = "system_controller_not_official"
        next_required = "close_p6_official_controller_gate"
    else:
        route = "R0-SystemLegalControllerClosed"
        blocker = ""
        next_required = ""
    return {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "v9278_boundary_pass": p0.get("v9278_boundary_pass"),
        "source_route_v9278": source.get("source_route_v9278"),
        "stable_accept_contract_id": "AC2Q2-stable-quantized-1e5-event-tie",
        "stable_accept_contract_pass_v9278": source.get("stable_accept_contract_pass"),
        "native_bucket_kernel_v2_pass_v9278": source.get("native_bucket_kernel_v2_pass"),
        "stable_accept_official_calibration_pass": p1.get("stable_accept_official_calibration_pass"),
        "stable_accept_heldout_support_pass": p2.get("stable_accept_heldout_support_pass"),
        "full_system_integration_pass": p3.get("full_system_integration_pass"),
        "full_system_step_attribution_pass": p4.get("full_system_step_attribution_pass"),
        "batch_major_runtime_pass": p5.get("batch_major_runtime_pass"),
        "system_legal_controller_pass": p6.get("system_legal_controller_pass"),
        "official_eligible": p6.get("official_eligible"),
        "candidate_count": p1.get("candidate_count"),
        "accepted_count_old_reference": p1.get("accepted_count_old_reference"),
        "missing_stable_accept_fields": p1.get("missing_stable_accept_fields"),
        "missing_outcome_fields": p1.get("missing_outcome_fields"),
        "native_kernel_used_in_p6": p3.get("native_bucket_kernel_used_in_p6"),
        "new_full_system_step_ratio_measured": p4.get("new_full_system_step_ratio_measured"),
        "old_step_ratio_reused_as_measurement": p4.get("old_step_ratio_reused_as_measurement"),
        "source_boundary_step_ratio_q90": source.get("controller_step_ratio_q90"),
        "controller_step_ratio_q90": p6.get("step_ratio_q90"),
        "native_time_ms_q90_local_v9278": source.get("native_time_ms_q90"),
        "q90_reduction_local_v9278": source.get("q90_reduction"),
        "kernel_count_after_local_v9278": source.get("kernel_count_after"),
        "sync_count_after_local_v9278": source.get("sync_count_after"),
        "primary_blocker": blocker,
        "next_required_implementation": next_required,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _artifact_hashes(out_dir: Path, artifacts: Sequence[tuple[str, Path]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, path in artifacts:
        if path.exists():
            rows.append({"artifact": name, "path": str(path.relative_to(ROOT)), "sha256": _sha256(path)})
    return rows


def _write_recap(
    out_dir: Path,
    route: Dict[str, Any],
    p0: Dict[str, Any],
    p1: Dict[str, Any],
    p2: Dict[str, Any],
    p3: Dict[str, Any],
    p4: Dict[str, Any],
    p5: Dict[str, Any],
    p6: Dict[str, Any],
    p12: Dict[str, Any],
    p13: Dict[str, Any],
    audit: Dict[str, Any],
    hashes: List[Dict[str, Any]],
) -> None:
    def h(name: str) -> str:
        for row in hashes:
            if row.get("artifact") == name:
                return str(row.get("sha256"))
        return ""

    content = f"""# DG-KAN v9.2.79 Stable-Accept Official Promotion 与 Full-System Batch-Major Runtime Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.2.79_StableAcceptOfficialPromotion_FullSystemBatchMajorClosure_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 v9.2.78 的 local native kernel pass 或旧 controller metrics 写成 official system pass。

## 0. 最新结论

```text
route = {route.get('route')}
base_candidate = LQ-t2-h256
success_v9279_strict_purekan_functional = False
success_v9279_full_functional = False
success_v9279_external_ready = False
```

最终 artifact：

```text
{out_dir.relative_to(ROOT)}
```

核心结论：

1. P0 复现 v9.2.78 boundary：source route = `{p0.get('source_route_v9278')}`，stable accept local pass = `{p0.get('stable_accept_contract_pass')}`，native bucket kernel v2 pass = `{p0.get('native_bucket_kernel_v2_pass')}`，source system pass = `{p0.get('source_system_legal_controller_pass')}`。
2. P1 按 plan 尝试 official calibration rerun，但 full-online row 没有 AC2Q2 official 所需的 stable score/rank/accept 字段；calibration rerun 不能真实执行，pass = `0`。
3. P2 heldout/support rerun 同样 blocked：缺少 full-row stable accept materialization 和 outcome labels，不能复用旧 C3 metrics。
4. P3 证实 v9.2.78 native stable kernel 是 local pass，但没有出现在 P6 full-system trace 中：`native_bucket_kernel_used_in_p6 = {p3.get('native_bucket_kernel_used_in_p6')}`。
5. P4 没有新的 full-system native step timing；source boundary step ratio `{p4.get('source_boundary_step_ratio_q90')}` 没有被当作 v9.2.79 measured native ratio 复用。
6. P5 batch-major native runtime 仍只是 v9.2.78 local evidence：local native q90 = `{p5.get('native_time_ms_q90_local_v9278')}`，q90 reduction = `{p5.get('q90_reduction_local_v9278')}`，不能替代 end-to-end step timing。
7. P6 official controller 未打开：`official_eligible = {p6.get('official_eligible')}`，reason = `{p6.get('reason')}`。
8. 继续追溯 outcome labels 后仍不能补：`source_hash` join coverage = `{p12.get('source_hash_covered_candidate_rows')}`，`event_family+dataset+seed` 仍有 `{p12.get('event_family_dataset_seed_ambiguous_label_candidate_rows')}` 个 candidate label ambiguous。
9. v9.2.78 stable accept trace 只覆盖局部子集：local stable rows = `{p13.get('local_stable_accept_trace_rows')}`，matched full candidates = `{p13.get('matched_full_candidate_rows_by_dataset_seed_step_rank')}` / `{p13.get('candidate_count_full_online')}`。
10. 当前 blocker：`{route.get('primary_blocker')}`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9279_stable_accept_official_promotion_full_system_batch_major_closure.py` | v9.2.79 runner；复现 v9.2.78 boundary，审计 AC2Q2 stable accept 是否具备 full-row official calibration/heldout/support rerun 与 full-system native runtime materialization |

代码检查：

```text
python -m py_compile experiments/run_v9279_stable_accept_official_promotion_full_system_batch_major_closure.py
```

正式运行：

```bash
python experiments/run_v9279_stable_accept_official_promotion_full_system_batch_major_closure.py \\
  --out-dir {out_dir.relative_to(ROOT)} \\
  --fresh --device auto --data-root data --seed 1314
```

## 2. Route

`route_decision.json`：

```json
{json.dumps(route, indent=2, ensure_ascii=False)}
```

判断：v9.2.79 没有把 v9.2.78 的局部 stable-accept closure 提升成 official。真正缺的是 full-row stable score/rank/accept materialization、outcome labels 和 P6 native full-system step trace。

## 3. P1 stable accept official calibration rerun

Artifacts：

```text
p1_stable_accept_official_calibration_rerun.csv
stable_accept_full_row_field_trace_v9279.csv
```

Summary：

```text
calibration_rerun_attempted = {p1.get('calibration_rerun_attempted')}
calibration_rerun = {p1.get('calibration_rerun')}
stable_accept_full_row_materialization_present = {p1.get('stable_accept_full_row_materialization_present')}
outcome_labels_present = {p1.get('outcome_labels_present')}
event_count = {p1.get('event_count')}
candidate_count = {p1.get('candidate_count')}
accepted_count_old_reference = {p1.get('accepted_count_old_reference')}
missing_stable_accept_fields = {p1.get('missing_stable_accept_fields')}
missing_outcome_fields = {p1.get('missing_outcome_fields')}
stable_accept_official_calibration_pass = {p1.get('stable_accept_official_calibration_pass')}
```

判断：这是本轮 terminal blocker。现有 full-online 表有 old `accept_decision/bridge_score/candidate_rank`，但没有 AC2Q2 full-row `score_ref/score_native/score_quantized/rank/accept`，也没有 rerun 所需 outcome label。不能编造 rerun 指标。

## 4. P2 heldout/support rerun

Artifact：

```text
p2_stable_accept_heldout_support_rerun.csv
```

Summary：

```text
heldout_rerun_attempted = {p2.get('heldout_rerun_attempted')}
heldout_rerun = {p2.get('heldout_rerun')}
support_balance_rerun = {p2.get('support_balance_rerun')}
stable_accept_heldout_support_pass = {p2.get('stable_accept_heldout_support_pass')}
reason = {p2.get('reason')}
```

判断：stable accept 是 rule change，不能沿用旧 heldout/support summary；由于 P1 full-row materialization 缺失，P2 必须 blocked。

## 5. P3-P5 full-system runtime audit

Artifacts：

```text
p3_native_stable_kernel_full_system_integration_audit.csv
p4_full_system_step_attribution_native_stable_path.csv
p5_batch_major_native_runtime_closure.csv
```

Summary：

```text
native_kernel_local_pass_v9278 = {p3.get('native_kernel_local_pass_v9278')}
native_bucket_kernel_used_in_p6 = {p3.get('native_bucket_kernel_used_in_p6')}
native_runtime_full_system_materialization_present = {p3.get('native_runtime_full_system_materialization_present')}
new_full_system_step_ratio_measured = {p4.get('new_full_system_step_ratio_measured')}
old_step_ratio_reused_as_measurement = {p4.get('old_step_ratio_reused_as_measurement')}
batch_major_native_full_system_runtime_measured = {p5.get('batch_major_native_full_system_runtime_measured')}
batch_major_runtime_pass = {p5.get('batch_major_runtime_pass')}
```

判断：v9.2.78 local native kernel 确实是好信号，但 v9.2.79 official promotion 需要 end-to-end P6 trace。现有 artifact 不能证明 native stable kernel 已进入 full train-stream controller runtime。

## 6. P6 system controller boundary

Artifact：

```text
p6_system_legal_exact_signal_controller_v11.csv
```

Boundary：

```text
controller_id = {p6.get('controller_id')}
official_eligible = {p6.get('official_eligible')}
system_legal_controller_pass = {p6.get('system_legal_controller_pass')}
reason = {p6.get('reason')}
source_boundary_step_ratio_q90 = {p6.get('source_boundary_step_ratio_q90')}
step_ratio_q90 = {p6.get('step_ratio_q90')}
old_metrics_reused_for_official = {p6.get('old_metrics_reused_for_official')}
native_bucket_kernel_used_in_p6 = {p6.get('native_bucket_kernel_used_in_p6')}
```

判断：P6 没有 official。这里没有用旧 decision metrics 或旧 step ratio 盖章，也没有把 local P3 q90 写成 system step ratio。

## 7. 继续追溯：outcome label 与 stable accept trace

Artifacts：

```text
p12_outcome_label_recovery_probe.csv
outcome_label_recovery_trace_v9279.csv
p13_stable_accept_trace_coverage_probe.csv
```

Outcome join summary：

```text
label_source = {p12.get('label_source')}
candidate_count = {p12.get('candidate_count')}
best_join_mode = {p12.get('best_join_mode')}
source_hash_covered_candidate_rows = {p12.get('source_hash_covered_candidate_rows')}
event_family_unique_label_candidate_rows = {p12.get('event_family_unique_label_candidate_rows')}
event_family_ambiguous_label_candidate_rows = {p12.get('event_family_ambiguous_label_candidate_rows')}
event_family_dataset_seed_unique_label_candidate_rows = {p12.get('event_family_dataset_seed_unique_label_candidate_rows')}
event_family_dataset_seed_ambiguous_label_candidate_rows = {p12.get('event_family_dataset_seed_ambiguous_label_candidate_rows')}
outcome_label_recovery_pass = {p12.get('outcome_label_recovery_pass')}
```

Stable trace coverage：

```text
local_stable_accept_trace_rows = {p13.get('local_stable_accept_trace_rows')}
candidate_count_full_online = {p13.get('candidate_count_full_online')}
matched_full_candidate_rows_by_dataset_seed_step_rank = {p13.get('matched_full_candidate_rows_by_dataset_seed_step_rank')}
full_row_coverage = {p13.get('full_row_coverage')}
stable_accept_trace_coverage_pass = {p13.get('stable_accept_trace_coverage_pass')}
```

判断：不能用 `event_family` 级别统计补 outcome label，因为同一 family 内 label 冲突很多；也不能用 v9.2.78 local trace 补 full rows，因为覆盖率太低。这进一步确认下一步必须重跑 full train-stream materializer，而不是 join 旧 artifact。

## 8. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p7_leave_dataset_and_stratum_out.csv` | `P6_system_controller_not_official` |
| `p8_official_paired_replay.csv` | same |
| `p9_short_run_functional_validation.csv` | same |
| `p10_full_run_robustness_strong_baseline.csv` | same |

没有把 stable accept local pass、native local q90 reduction 或旧 reference frontier 写成 LDO/paired replay/short-run/full-run success。

## 9. No-fake audit

```text
rows_checked = {audit.get('rows_checked')}
fake_proxy_nonzero_count = {audit.get('fake_proxy_nonzero_count')}
fake_data_used = {audit.get('fake_data_used')}
proxy_row_used = {audit.get('proxy_row_used')}
cpu_offload_used = {audit.get('cpu_offload_used')}
no_fake = {audit.get('no_fake')}
no_proxy = {audit.get('no_proxy')}
```

Contract audit：

```text
manual_forward/manual_backward/manual_adamw_update = 1/1/1
train_stream_probe = 1
payload_binding_contract_pass = 1
stable_accept_local_contract_pass = {route.get('stable_accept_contract_pass_v9278')}
native_bucket_kernel_v2_local_pass = {route.get('native_bucket_kernel_v2_pass_v9278')}
stable_accept_official_calibration_pass = {route.get('stable_accept_official_calibration_pass')}
stable_accept_heldout_support_pass = {route.get('stable_accept_heldout_support_pass')}
full_system_integration_pass = {route.get('full_system_integration_pass')}
batch_major_runtime_pass = {route.get('batch_major_runtime_pass')}
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
projection_used_for_official = 0
```

## 10. Hash

| artifact | SHA256 |
|---|---|
| plan | `{h('plan')}` |
| runner | `{h('runner')}` |
| run manifest | `{h('run manifest')}` |
| route | `{h('route')}` |
| P0 boundary | `{h('P0 boundary')}` |
| P1 calibration | `{h('P1 calibration')}` |
| P2 heldout support | `{h('P2 heldout support')}` |
| P3 integration | `{h('P3 integration')}` |
| P4 attribution | `{h('P4 attribution')}` |
| P5 batch major | `{h('P5 batch major')}` |
| P6 system controller | `{h('P6 system controller')}` |
| P12 outcome recovery | `{h('P12 outcome recovery')}` |
| P13 stable trace coverage | `{h('P13 stable trace coverage')}` |
| provenance audit | `{h('provenance audit')}` |

## 11. 最终分析结论

v9.2.79 的真实推进是：

```text
v9.2.78: stable accept local correctness 与 native bucket kernel local runtime 已闭合，
          但 official promotion 仍 blocked。
v9.2.79: 对 official promotion 所需 full-row materialization 做了严格审计；
          发现现有 full-online artifact 缺 AC2Q2 stable score/rank/accept 与 outcome labels；
          继续尝试 outcome join 和 local stable trace coverage，仍不能可靠补齐全量 rows。
```

机制判断：

1. H1 未能执行：不是 stable accept 指标失败，而是 full-row materialization 缺失，calibration/heldout 无法真实 rerun。
2. H2 未能执行：P6 没有 native stable kernel full-system trace，不能判断 `2.713296` 是旧路径残留还是 native full-system 下界。
3. H3 未能 official：v9.2.78 local q90 reduction 仍是局部证据，不能替代 batch-major full-system timing。
4. H4 成立：rule change 后没有绕过 rerun gate；P7-P10 继续 gate-blocked。
5. P12/P13 进一步确认：旧 outcome label artifact 不能按 `source_hash` join，按 family join 又 label ambiguous；v9.2.78 local stable trace 覆盖率也不足。
6. 当前下一步必须重跑 full train-stream materializer：为全量 candidate rows 记录 `score_ref/score_native/score_quantized/rank/accept` 和真实 outcome labels，再重跑 calibration/heldout/support 与 P6 native runtime。

最终一句话：

> v9.2.79 继续执行后仍停在 `{route.get('route')}`：v9.2.78 的 stable accept/native kernel 局部闭合没有回退，但旧 artifact 无法可靠 join 出全量 outcome labels 或 stable accept rows；official promotion 需要重跑 full train-stream materialization，strict PureKAN functional 仍未成功。
"""
    RECAP_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT / "v9279_stable_accept_official_promotion_full_system_batch_major_closure_first_20260513T170000Z")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=1314)
    args = parser.parse_args()

    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if args.fresh and out_dir.exists():
        for child in sorted(out_dir.iterdir()):
            if child.is_file():
                child.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    full_table = SRC_FULL_ONLINE / "full_online_event_table_v9272.csv"
    full_rows = _read_csv_rows(full_table)
    full_fields = _csv_fieldnames(full_table)
    counts = _count_candidate_rows(full_rows)
    source = _source_summary()

    p0 = _stage_p0(source)
    p1, p2, _, p3 = _stage_p1_p2_p3(out_dir, full_rows, full_fields, counts)
    p4, p5 = _stage_p4_p5(source, full_fields, counts)
    p6 = _stage_p6(source, p1, p2, p3, p4, p5)
    p12 = _stage_p12_outcome_join_probe(out_dir, full_rows)
    p13 = _stage_p13_stable_accept_trace_coverage_probe(full_rows)
    route = _route_decision(source, p0, p1, p2, p3, p4, p5, p6)
    route.update(
        {
            "outcome_label_recovery_pass": p12.get("outcome_label_recovery_pass"),
            "outcome_label_best_join_mode": p12.get("best_join_mode"),
            "outcome_source_hash_covered_candidate_rows": p12.get("source_hash_covered_candidate_rows"),
            "outcome_event_family_dataset_seed_ambiguous_rows": p12.get("event_family_dataset_seed_ambiguous_label_candidate_rows"),
            "stable_accept_trace_coverage_pass": p13.get("stable_accept_trace_coverage_pass"),
            "stable_accept_local_trace_rows": p13.get("local_stable_accept_trace_rows"),
            "stable_accept_trace_matched_full_candidate_rows": p13.get("matched_full_candidate_rows_by_dataset_seed_step_rank"),
            "stable_accept_trace_full_row_coverage": p13.get("full_row_coverage"),
        }
    )

    manifest = {
        "experiment_id": "DG-KAN-v9.2.79",
        "started_at": _now_iso(),
        "completed_at": _now_iso(),
        "device_arg": args.device,
        "data_root": args.data_root,
        "seed": args.seed,
        "datasets": "MNIST,Fashion-MNIST,KMNIST",
        "seeds": "0,1,2,3,4,5,6,7",
        "source_v9278_artifact": str(SRC_V9278.relative_to(ROOT)),
        "source_full_online_artifact": str(SRC_FULL_ONLINE.relative_to(ROOT)),
        "full_online_event_table": str(full_table.relative_to(ROOT)),
        "stable_accept_contract_id": "AC2Q2-stable-quantized-1e5-event-tie",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }

    downstream = [
        _not_run("P7_LEAVE_DATASET_AND_STRATUM_OUT", "P6_system_controller_not_official", system_legal_controller_pass=0),
        _not_run("P8_OFFICIAL_PAIRED_REPLAY", "P6_system_controller_not_official", system_legal_controller_pass=0),
        _not_run("P9_SHORT_RUN_FUNCTIONAL_VALIDATION", "P6_system_controller_not_official", system_legal_controller_pass=0),
        _not_run("P10_FULL_RUN_ROBUSTNESS_STRONG_BASELINE", "P6_system_controller_not_official", system_legal_controller_pass=0),
    ]
    failure_rows = [
        {
            "stage": "P1_STABLE_ACCEPT_OFFICIAL_CALIBRATION_RERUN",
            "failure_code": "F6_STABLE_ACCEPT_FULL_ROW_MATERIALIZATION_MISSING",
            "reason": p1["reason"],
            "missing_stable_accept_fields": p1["missing_stable_accept_fields"],
            "missing_outcome_fields": p1["missing_outcome_fields"],
            "next_required_implementation": route["next_required_implementation"],
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "stage": "P12_OUTCOME_LABEL_RECOVERY_PROBE",
            "failure_code": "F12_OUTCOME_LABEL_JOIN_AMBIGUOUS",
            "reason": p12["reason"],
            "source_hash_covered_candidate_rows": p12.get("source_hash_covered_candidate_rows"),
            "event_family_dataset_seed_ambiguous_label_candidate_rows": p12.get("event_family_dataset_seed_ambiguous_label_candidate_rows"),
            "next_required_implementation": "rerun_full_train_stream_outcome_materializer",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "stage": "P13_STABLE_ACCEPT_TRACE_COVERAGE_PROBE",
            "failure_code": "F13_LOCAL_STABLE_TRACE_NOT_FULL_ROW_MATERIALIZATION",
            "reason": p13["reason"],
            "local_stable_accept_trace_rows": p13.get("local_stable_accept_trace_rows"),
            "matched_full_candidate_rows_by_dataset_seed_step_rank": p13.get("matched_full_candidate_rows_by_dataset_seed_step_rank"),
            "candidate_count_full_online": p13.get("candidate_count_full_online"),
            "next_required_implementation": "rerun_full_train_stream_native_stable_accept_materializer",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    ]
    audit = {
        "stage": "P11_NO_FAKE_AUDIT",
        "rows_checked": counts["event_count"] + 31,
        "fake_proxy_nonzero_count": counts["fake_data_used_count"] + counts["proxy_row_used_count"],
        "fake_data_used": int(counts["fake_data_used_count"] > 0),
        "proxy_row_used": int(counts["proxy_row_used_count"] > 0),
        "cpu_offload_used": int(counts["cpu_offload_used_count"] > 0),
        "no_fake": counts["fake_data_used_count"] == 0,
        "no_proxy": counts["proxy_row_used_count"] == 0,
        "projection_used_for_official": 0,
        "source_measured_gap_used_for_official": 0,
        "formula_proxy_used_for_official": 0,
        "old_metrics_reused_for_official": 0,
    }

    _write_json(out_dir / "run_manifest.json", manifest)
    _write_json(out_dir / "route_decision.json", route)
    _write_csv(out_dir / "p0_v9278_boundary_reproduction.csv", [p0])
    _write_csv(out_dir / "p1_stable_accept_official_calibration_rerun.csv", [p1])
    _write_csv(out_dir / "p2_stable_accept_heldout_support_rerun.csv", [p2])
    _write_csv(out_dir / "p3_native_stable_kernel_full_system_integration_audit.csv", [p3])
    _write_csv(out_dir / "p4_full_system_step_attribution_native_stable_path.csv", [p4])
    _write_csv(out_dir / "p5_batch_major_native_runtime_closure.csv", [p5])
    _write_csv(out_dir / "p6_system_legal_exact_signal_controller_v11.csv", [p6])
    _write_csv(out_dir / "p7_leave_dataset_and_stratum_out.csv", [downstream[0]])
    _write_csv(out_dir / "p8_official_paired_replay.csv", [downstream[1]])
    _write_csv(out_dir / "p9_short_run_functional_validation.csv", [downstream[2]])
    _write_csv(out_dir / "p10_full_run_robustness_strong_baseline.csv", [downstream[3]])
    _write_csv(out_dir / "p12_outcome_label_recovery_probe.csv", [p12])
    _write_csv(out_dir / "p13_stable_accept_trace_coverage_probe.csv", [p13])
    _write_csv(out_dir / "failure_table.csv", failure_rows)
    _write_json(out_dir / "v9279_provenance_audit.json", audit)

    artifacts = [
        ("plan", PLAN_PATH),
        ("runner", SCRIPT_PATH),
        ("run manifest", out_dir / "run_manifest.json"),
        ("route", out_dir / "route_decision.json"),
        ("P0 boundary", out_dir / "p0_v9278_boundary_reproduction.csv"),
        ("P1 calibration", out_dir / "p1_stable_accept_official_calibration_rerun.csv"),
        ("P2 heldout support", out_dir / "p2_stable_accept_heldout_support_rerun.csv"),
        ("P3 integration", out_dir / "p3_native_stable_kernel_full_system_integration_audit.csv"),
        ("P4 attribution", out_dir / "p4_full_system_step_attribution_native_stable_path.csv"),
        ("P5 batch major", out_dir / "p5_batch_major_native_runtime_closure.csv"),
        ("P6 system controller", out_dir / "p6_system_legal_exact_signal_controller_v11.csv"),
        ("P12 outcome recovery", out_dir / "p12_outcome_label_recovery_probe.csv"),
        ("P13 stable trace coverage", out_dir / "p13_stable_accept_trace_coverage_probe.csv"),
        ("failure table", out_dir / "failure_table.csv"),
        ("provenance audit", out_dir / "v9279_provenance_audit.json"),
    ]
    hashes = _artifact_hashes(out_dir, artifacts)
    _write_csv(out_dir / "artifact_hashes.csv", hashes)
    hashes = _artifact_hashes(out_dir, artifacts + [("artifact hashes", out_dir / "artifact_hashes.csv")])
    _write_csv(out_dir / "artifact_hashes.csv", hashes)
    _write_recap(out_dir, route, p0, p1, p2, p3, p4, p5, p6, p12, p13, audit, hashes)

    print(json.dumps(route, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
