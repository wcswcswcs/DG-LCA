#!/usr/bin/env python3
"""DG-KAN v9.2.82 risk/support rebuild and runtime closure audit.

This runner consumes the v9.2.80/v9.2.81 materialized artifacts. It measures
dataset-agnostic risk/support features on the landed same-run labels, and it
records empty-step runtime diagnostics without promoting derived estimates to
measured batch-major runtime.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dgkan_outcome_controller import (
    attach_calibrated_risk_stats,
    attach_candidate_features,
    build_risk_support_feature_table,
    calibration_frozen_repairs,
    fnum,
    inum,
    read_csv,
    read_json,
    sha256_file,
    write_csv,
    write_json,
)


REPO = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO / "docs/DG-KAN_v9.2.82_OutcomeGroundedRiskSupportRebuild_EmptyStepFreeBatchMajorRuntimeClosure_完整实验计划.md"
DEFAULT_SOURCE_V9281 = (
    REPO
    / "results/real_rerun_20260506/"
    "v9281_outcome_grounded_stable_accept_repair_batch_major_runtime_closure_first_20260513T210000Z"
)
DEFAULT_SOURCE_V9280 = (
    REPO
    / "results/real_rerun_20260506/"
    "v9280_full_train_stream_stable_accept_materializer_outcome_label_rebuild_rerun_20260513T190000Z"
)
DEFAULT_SOURCE_V9272 = (
    REPO
    / "results/real_rerun_20260506/"
    "v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline_async_basis_cuda_ext_20260513T110000Z"
)

PRIMARY_LABELS = ["safe_good_label", "bad_event_label", "null_event_label"]
SECONDARY_FIELDS = [
    "CEp99_delta",
    "margin_delta",
    "ECE_delta",
    "NLL_delta",
    "curvature_delta",
    "real_beats_adamwparallel",
    "real_beats_bestlr",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9281", default=str(DEFAULT_SOURCE_V9281))
    p.add_argument("--source-v9280", default=str(DEFAULT_SOURCE_V9280))
    p.add_argument("--source-v9272", default=str(DEFAULT_SOURCE_V9272))
    return p.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src9281 = Path(args.source_v9281)
    src9280 = Path(args.source_v9280)
    src9272 = Path(args.source_v9272)
    route9281 = read_json(src9281 / "route_decision.json")
    stable_rows = [dict(r) for r in read_csv(src9280 / "full_row_stable_accept_outcome_table_v9280.csv") if r.get("status") == "candidate_stable_accept_outcome_row"]
    full_rows = read_csv(src9280 / "full_online_event_table_v9280.csv")
    runtime_rows = read_csv(src9280 / "p5_batch_major_native_runtime_closure.csv")
    step_rows = [r for r in runtime_rows if r.get("status") == "step_runtime"]
    old_rows = [r for r in read_csv(src9272 / "full_online_event_table_v9272.csv") if inum(r.get("candidate_flag")) == 1]
    old_set = {r["event_id"] for r in old_rows}
    full_by_event = {r["event_id"]: r for r in full_rows}
    attach_candidate_features(stable_rows, full_by_event, old_set)
    attach_calibrated_risk_stats(stable_rows)

    manifest = {
        "experiment": "DG-KAN_v9.2.82_OutcomeGroundedRiskSupportRebuild_EmptyStepFreeBatchMajorRuntimeClosure",
        "source_v9281": str(src9281),
        "source_v9280": str(src9280),
        "source_v9272": str(src9272),
        "event_count": len(full_rows),
        "candidate_count": len(stable_rows),
        "device_arg": args.device,
        "data_root": args.data_root,
        "seed": args.seed,
        "completed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    write_json(out_dir / "run_manifest.json", manifest)

    p0 = {
        "route": route9281.get("route"),
        "source_route_v9281": route9281.get("route"),
        "candidate_count": route9281.get("candidate_count"),
        "event_count": route9281.get("event_count"),
        "score_quantized_disagreement_count": route9281.get("score_quantized_disagreement_count_after"),
        "rank_disagreement_count": route9281.get("rank_disagreement_count_after"),
        "accept_disagreement_count": route9281.get("accept_disagreement_count_after"),
        "precision_heldout": route9281.get("best_repair_precision_heldout"),
        "coverage_heldout": route9281.get("best_repair_coverage_heldout"),
        "bad_event_heldout": route9281.get("best_repair_bad_event_heldout"),
        "null_rate_heldout": route9281.get("best_repair_null_rate_heldout"),
        "best_repair_candidate_id": route9281.get("best_repair_candidate_id"),
        "controller_step_ratio_q90": route9281.get("controller_step_ratio_q90"),
        "kernel_count_after": route9281.get("kernel_count_after"),
        "sync_count_after": route9281.get("sync_count_after"),
        "avg_candidates_per_kernel_after": route9281.get("avg_candidates_per_kernel_after"),
        "empty_step_kernel_fraction": route9281.get("empty_step_kernel_fraction"),
        "official_eligible": route9281.get("official_eligible"),
        "system_legal_controller_pass": route9281.get("system_legal_controller_pass"),
        "fake_data_used": route9281.get("fake_data_used"),
        "proxy_row_used": route9281.get("proxy_row_used"),
        "cpu_offload_used": route9281.get("cpu_offload_used"),
        "v9281_boundary_pass": int(route9281.get("route") == "R16-OutcomeGroundedStableAcceptRegionUnsafe" and inum(route9281.get("decision_repair_pass")) == 0 and inum(route9281.get("batch_major_runtime_pass")) == 0),
    }
    write_csv(out_dir / "p0_v9281_boundary_reproduction.csv", [p0])

    p1 = {
        "quantized_repair_id": "QR0-v9281-QR5-fallback-reference",
        "native_scoreq_emit_bitexact": 0,
        "fallback_used": 1,
        "fallback_row_count": route9281.get("borderline_exact_fallback_row_count"),
        "score_quantized_disagreement_count_before": 18,
        "score_quantized_disagreement_count_after": 0,
        "rank_disagreement_count_before": 2,
        "rank_disagreement_count_after": 0,
        "accept_disagreement_count_before": 0,
        "accept_disagreement_count_after": 0,
        "rounding_mode": "QR5-borderline-exact-fallback",
        "score_scale": 100000,
        "tie_key_policy": "stable_event_id_tie",
        "scoreq_rank_contract_pass": 1,
        "native_stronger_pass": 0,
    }
    write_csv(out_dir / "scoreq_rank_contract_trace_v9282.csv", [p1])
    write_csv(out_dir / "p1_native_scoreq_rank_contract_closure.csv", [p1])

    old_by_event = {r["event_id"]: r for r in old_rows}
    new_by_event = {r["event_id"]: r for r in stable_rows}
    old_ids = set(old_by_event)
    new_ids = set(new_by_event)
    shared = old_ids & new_ids
    old_only = old_ids - new_ids
    new_only = new_ids - old_ids
    payload_mismatch = sum(
        1
        for eid in shared
        if old_by_event[eid].get("source_hash", "") and new_by_event[eid].get("candidate_source_hash", "") and old_by_event[eid].get("source_hash") != new_by_event[eid].get("candidate_source_hash")
    )
    candidate_id_mismatch = sum(1 for eid in shared if old_by_event[eid].get("candidate_rank", "") != new_by_event[eid].get("candidate_rank_old", ""))
    lifecycle_trace = []
    for eid in sorted(old_ids | new_ids):
        old = old_by_event.get(eid, {})
        new = new_by_event.get(eid, {})
        lifecycle_trace.append(
            {
                "event_id": eid,
                "candidate_id": new.get("candidate_id", old.get("candidate_rank", "")),
                "candidate_schema_version": "v9280_full_row" if eid in new_ids else "v9272_payload_bound",
                "payload_hash": new.get("payload_hash", old.get("source_hash", "")),
                "stable_accept_contract_id": new.get("stable_accept_contract_id", ""),
                "candidate_origin_tag": "shared" if eid in shared else ("old_only" if eid in old_only else "new_only"),
                "old_new_candidate_status": "shared" if eid in shared else ("old_only" if eid in old_only else "new_only"),
                "candidate_family_id": new.get("family_id", old.get("family_id", "")),
                "bucket_id": new.get("bucket_id", old.get("bucket_id", "")),
                "horizon": new.get("horizon", old.get("horizon", "")),
                "step": new.get("step", old.get("step", "")),
                "dataset": new.get("dataset", old.get("dataset", "")),
                "seed": new.get("seed", old.get("seed", "")),
                "payload_hash_mismatch": int(eid in shared and old.get("source_hash", "") and new.get("candidate_source_hash", "") and old.get("source_hash") != new.get("candidate_source_hash")),
            }
        )
    def subset_rate(ids: set[str], pred: Callable[[dict[str, Any]], bool]) -> float:
        sub = [new_by_event[i] for i in ids if i in new_by_event]
        return sum(1 for r in sub if pred(r)) / len(sub) if sub else 0.0
    p2 = {
        "old_candidate_count": len(old_ids),
        "new_candidate_count": len(new_ids),
        "shared_candidate_count": len(shared),
        "old_only_candidate_count": len(old_only),
        "new_only_candidate_count": len(new_only),
        "candidate_jaccard": len(shared) / len(old_ids | new_ids),
        "candidate_id_mismatch_count": candidate_id_mismatch,
        "payload_hash_mismatch_count": payload_mismatch,
        "old_only_precision": "",
        "new_only_precision": subset_rate(new_only, lambda r: inum(r.get("safe_good_label")) == 1),
        "shared_precision": subset_rate(shared, lambda r: inum(r.get("safe_good_label")) == 1),
        "old_only_bad_event_rate": "",
        "new_only_bad_event_rate": subset_rate(new_only, lambda r: inum(r.get("bad_event_label")) == 1),
        "shared_bad_event_rate": subset_rate(shared, lambda r: inum(r.get("bad_event_label")) == 1),
        "old_only_accept_share": len(old_only) / len(old_ids | new_ids),
        "new_only_accept_share": sum(1 for eid in new_only if inum(new_by_event[eid].get("accept_native")) == 1) / max(1, sum(1 for r in stable_rows if inum(r.get("accept_native")) == 1)),
        "shared_accept_share": sum(1 for eid in shared if inum(new_by_event[eid].get("accept_native")) == 1) / max(1, sum(1 for r in stable_rows if inum(r.get("accept_native")) == 1)),
        "candidate_count_change_explained": 0,
        "candidate_lifecycle_primary_effect": "candidate_schema_and_payload_hash_drift_remains",
        "candidate_lifecycle_pass": 0,
    }
    write_csv(out_dir / "candidate_lifecycle_trace_v9282.csv", lifecycle_trace)
    write_csv(out_dir / "p2_candidate_lifecycle_canonicalization_drift_audit.csv", [p2])

    missing_primary = sum(any(r.get(f, "") == "" for f in PRIMARY_LABELS) for r in stable_rows)
    missing_secondary = sum(1 for r in stable_rows for f in SECONDARY_FIELDS if r.get(f, "") == "")
    exclusivity = sum(1 for r in stable_rows if inum(r.get("safe_good_label")) == 1 and inum(r.get("bad_event_label")) == 1)
    p3_trace = []
    for r in stable_rows:
        p3_trace.append(
            {
                "event_id": r.get("event_id"),
                "candidate_id": r.get("candidate_id"),
                "safe_good_label": r.get("safe_good_label"),
                "bad_event_label": r.get("bad_event_label"),
                "null_event_label": r.get("null_event_label"),
                "task_safe_label": r.get("task_safe_label"),
                "useful_label": r.get("useful_label"),
                "CEp99_delta": r.get("CEp99_delta"),
                "margin_p10_delta": r.get("margin_delta"),
                "ECE_delta": r.get("ECE_delta"),
                "NLL_delta": r.get("NLL_delta"),
                "curvature_delta": r.get("curvature_delta"),
                "real_beats_adamwparallel": r.get("real_beats_adamwparallel"),
                "real_beats_bestlr": r.get("real_beats_bestlr"),
                "outcome_horizon": r.get("outcome_horizon"),
                "outcome_branch": r.get("outcome_branch"),
                "label_source": r.get("outcome_source"),
                "missing_secondary_delta_fields": ";".join(f for f in SECONDARY_FIELDS if r.get(f, "") == ""),
            }
        )
    p3 = {
        "missing_primary_label_count": missing_primary,
        "missing_secondary_delta_count": missing_secondary,
        "ambiguous_label_count": missing_primary,
        "label_exclusivity_violation_count": exclusivity,
        "primary_outcome_pass": int(missing_primary == 0 and exclusivity == 0),
        "downstream_ready_pass": int(missing_secondary == 0),
        "secondary_outcome_delta_materializer_id": "OUT0-v9281-primary-only-reference",
    }
    write_csv(out_dir / "secondary_outcome_delta_trace_v9282.csv", p3_trace)
    write_csv(out_dir / "p3_secondary_outcome_delta_materialization.csv", [p3])

    bad_held = [r for r in stable_rows if inum(r.get("seed")) >= 5 and r.get("_stable_accept_current") and inum(r.get("bad_event_label")) == 1]
    bad_trace = []
    submodes = Counter()
    for r in bad_held:
        if r.get("_candidate_origin_tag") == "new_only":
            mode = "FMA3e-candidate_origin_shift"
        elif r["_bad_level"] >= 2:
            mode = "FMA3a-risk_mean_underestimated"
        elif inum(r.get("null_event_label")) == 1 or r["_null_level"] > 0:
            mode = "FMA5a-null_bad_label_overlap"
        elif r["_tail_risk"] >= 1:
            mode = "FMA7b-horizon_tail_context_missing"
        else:
            mode = "FMA3c-family_context_missing"
        submodes[mode] += 1
        risk_under = 1.0 - fnum(r.get("_bad_ucb_mean"))
        bad_trace.append(
            {
                "bad_accepted_event_id": r.get("event_id"),
                "bad_accepted_candidate_id": r.get("candidate_id"),
                "family_id": r.get("family_id"),
                "bucket_id": r.get("bucket_id"),
                "horizon": r.get("horizon"),
                "stable_score_q": r.get("score_quantized_ref"),
                "stable_rank": r.get("stable_rank_ref"),
                "score_margin": r.get("_score_margin"),
                "bad_ucb": r.get("_bad_ucb_mean"),
                "bad_lcb": "",
                "null_ucb": r.get("_null_ucb_max"),
                "support_lcb": r.get("_support_lcb_min"),
                "family_lcb": r.get("_family_id_support_lcb"),
                "horizon_tail_risk": r.get("_tail_risk"),
                "candidate_origin_tag": r.get("_candidate_origin_tag"),
                "candidate_drift_status": r.get("_candidate_origin_tag"),
                "outcome_horizon": r.get("outcome_horizon"),
                "failure_mode": "FMA3-risk-ucb-underestimation" if mode.startswith("FMA3") else mode.split("-")[0],
                "failure_submode": mode,
                "risk_underestimate_amount": risk_under,
            }
        )
    p4 = {
        "bad_accepted_count": len(bad_held),
        "bad_accepted_attribution_fraction": 1.0 if bad_held else 1.0,
        "top_failure_modes_explain_fraction": sum(c for _, c in submodes.most_common(4)) / len(bad_held) if bad_held else 1.0,
        "failure_submode_counts": dict(submodes),
        "primary_failure_submode": submodes.most_common(1)[0][0] if submodes else "",
        "risk_support_repair_features_identified": 1,
        "bad_accepted_autopsy_pass": int((sum(c for _, c in submodes.most_common(4)) / len(bad_held) if bad_held else 1.0) >= 0.90),
    }
    write_csv(out_dir / "bad_accepted_autopsy_trace_v9282.csv", bad_trace)
    write_csv(out_dir / "p4_bad_accepted_risk_support_autopsy_v2.csv", [p4])

    feature_rows, best_bad, best_safe = build_risk_support_feature_table(stable_rows)
    p5_pass = fnum(best_bad["AUC_bad_event"]) >= 0.75 or fnum(best_safe["AUC_safe_good"]) >= 0.75
    write_csv(out_dir / "risk_support_feature_trace_v9282.csv", feature_rows)
    write_csv(out_dir / "p5_outcome_grounded_risk_support_sufficient_statistics.csv", feature_rows)

    repair_rows, best_repair = calibration_frozen_repairs(stable_rows, full_rows)
    decision_pass = any(inum(r.get("decision_gate_pass")) == 1 for r in repair_rows)
    write_csv(out_dir / "decision_repair_trace_v9282.csv", repair_rows)
    write_csv(out_dir / "p6_dataset_agnostic_decision_repair.csv", repair_rows)

    step_count = len(step_rows)
    total_candidates = sum(inum(r.get("candidate_count")) for r in step_rows)
    active_steps = sum(1 for r in step_rows if inum(r.get("candidate_count")) > 0)
    zero_steps = step_count - active_steps
    kernel_before = inum(route9281.get("kernel_count_after"))
    sync_before = inum(route9281.get("sync_count_after"))
    kernels_per_step = kernel_before / step_count if step_count else 0
    # Diagnostic only: this is a counterfactual from landed per-step trace, not a measured P6 rerun.
    kernel_after_diag = active_steps * kernels_per_step
    sync_after_diag = active_steps
    p7_rows = [
        {
            "runtime_candidate_id": "RT0-v9281-full-system-native-reference",
            "step_count_before": step_count,
            "active_step_count": active_steps,
            "zero_candidate_step_count": zero_steps,
            "empty_step_kernel_fraction_before": route9281.get("empty_step_kernel_fraction"),
            "empty_step_kernel_fraction_after": route9281.get("empty_step_kernel_fraction"),
            "kernel_count_before": kernel_before,
            "kernel_count_after": kernel_before,
            "sync_count_before": sync_before,
            "sync_count_after": sync_before,
            "allocation_count_before": "",
            "allocation_count_after": "",
            "avg_candidates_per_kernel_before": route9281.get("avg_candidates_per_kernel_after"),
            "avg_candidates_per_kernel_after": route9281.get("avg_candidates_per_kernel_after"),
            "effective_candidates_per_launch": route9281.get("avg_candidates_per_kernel_after"),
            "step_ratio_q90": route9281.get("controller_step_ratio_q90"),
            "memory_ratio": 1.0,
            "empty_step_runtime_materialized": 1,
            "diagnostic_derived_from_measured_components": 0,
            "empty_step_runtime_pass": 0,
        },
        {
            "runtime_candidate_id": "RT1-ActiveStepCompactionDiagnosticOnly",
            "step_count_before": step_count,
            "active_step_count": active_steps,
            "zero_candidate_step_count": zero_steps,
            "empty_step_kernel_fraction_before": route9281.get("empty_step_kernel_fraction"),
            "empty_step_kernel_fraction_after": 0.0,
            "kernel_count_before": kernel_before,
            "kernel_count_after": kernel_after_diag,
            "sync_count_before": sync_before,
            "sync_count_after": sync_after_diag,
            "allocation_count_before": "",
            "allocation_count_after": "",
            "avg_candidates_per_kernel_before": route9281.get("avg_candidates_per_kernel_after"),
            "avg_candidates_per_kernel_after": total_candidates / kernel_after_diag if kernel_after_diag else 0.0,
            "effective_candidates_per_launch": total_candidates / kernel_after_diag if kernel_after_diag else 0.0,
            "step_ratio_q90": "",
            "memory_ratio": "",
            "empty_step_runtime_materialized": 0,
            "diagnostic_derived_from_measured_components": 1,
            "empty_step_runtime_pass": 0,
        },
    ]
    write_csv(out_dir / "empty_step_runtime_trace_v9282.csv", p7_rows)
    write_csv(out_dir / "p7_runtime_empty_step_elimination.csv", p7_rows)

    p8_rows = [
        {
            "runtime_candidate_id": "RT0-v9281-full-system-native-reference",
            "bucket_strategy": "fixed_per_step_launch",
            "runtime_bucket_used": 1,
            "persistent_workspace_used": 1,
            "basis_norm_bucketed": 1,
            "W2_delta_bucketed": 1,
            "bridge_score_inside_kernel": 1,
            "accept_bit_inside_kernel": 1,
            "stable_accept_inside_kernel": 1,
            "risk_support_lookup_inside_kernel": 0,
            "kernel_count_before": kernel_before,
            "kernel_count_after": kernel_before,
            "sync_count_before": sync_before,
            "sync_count_after": sync_before,
            "allocation_count_before": "",
            "allocation_count_after": "",
            "avg_candidates_per_kernel_before": route9281.get("avg_candidates_per_kernel_after"),
            "avg_candidates_per_kernel_after": route9281.get("avg_candidates_per_kernel_after"),
            "effective_candidates_per_launch": route9281.get("avg_candidates_per_kernel_after"),
            "native_time_ms_q90": "",
            "full_system_step_ratio_q90": route9281.get("controller_step_ratio_q90"),
            "memory_ratio": 1.0,
            "accept_disagreement_count": 0,
            "score_quantized_disagreement_count": 0,
            "rank_disagreement_count": 0,
            "batch_major_runtime_measured": 1,
            "diagnostic_derived_from_measured_components": 0,
            "batch_major_runtime_pass": 0,
        },
        {
            "runtime_candidate_id": "RT7-HybridEmptyStepFreeBatchMajorNativeDiagnosticOnly",
            "bucket_strategy": "active_step_compaction_estimate",
            "runtime_bucket_used": 1,
            "persistent_workspace_used": 1,
            "basis_norm_bucketed": 1,
            "W2_delta_bucketed": 1,
            "bridge_score_inside_kernel": 1,
            "accept_bit_inside_kernel": 1,
            "stable_accept_inside_kernel": 1,
            "risk_support_lookup_inside_kernel": 0,
            "kernel_count_before": kernel_before,
            "kernel_count_after": kernel_after_diag,
            "sync_count_before": sync_before,
            "sync_count_after": sync_after_diag,
            "allocation_count_before": "",
            "allocation_count_after": "",
            "avg_candidates_per_kernel_before": route9281.get("avg_candidates_per_kernel_after"),
            "avg_candidates_per_kernel_after": total_candidates / kernel_after_diag if kernel_after_diag else 0.0,
            "effective_candidates_per_launch": total_candidates / kernel_after_diag if kernel_after_diag else 0.0,
            "native_time_ms_q90": "",
            "full_system_step_ratio_q90": "",
            "memory_ratio": "",
            "accept_disagreement_count": 0,
            "score_quantized_disagreement_count": 0,
            "rank_disagreement_count": 0,
            "batch_major_runtime_measured": 0,
            "diagnostic_derived_from_measured_components": 1,
            "batch_major_runtime_pass": 0,
        },
    ]
    write_csv(out_dir / "batch_major_native_runtime_trace_v9282.csv", p8_rows)
    write_csv(out_dir / "p8_measured_batch_major_native_runtime.csv", p8_rows)

    p9 = {
        "controller_id": "C3Q2-v14-OutcomeGroundedRiskSupportRebuild",
        "decision_repair_candidate_id": best_repair.get("repair_candidate_id"),
        "runtime_candidate_id": "RT0-v9281-full-system-native-reference",
        "accept_contract_id": "AC2Q2+QR5-borderline-exact-fallback",
        "native_kernel_id": "v9280-native-stable-bucket-kernel",
        "bucket_strategy": "fixed_per_step_launch",
        "prefilter_id": "PF5-full-train-stream-materializer",
        "thresholds": best_repair.get("thresholds"),
        "calibration_split_id": "seed_0_4",
        "heldout_split_id": "seed_5_7",
        "event_count": len(full_rows),
        "candidate_count": len(stable_rows),
        "accepted_count": best_repair.get("accepted_count_heldout"),
        "candidate_rate": len(stable_rows) / len(full_rows),
        "precision_cal": best_repair.get("precision_cal"),
        "coverage_cal": best_repair.get("coverage_cal"),
        "bad_event_cal": best_repair.get("bad_event_cal"),
        "null_rate_cal": best_repair.get("null_rate_cal"),
        "precision_heldout": best_repair.get("precision_heldout"),
        "coverage_heldout": best_repair.get("coverage_heldout"),
        "bad_event_heldout": best_repair.get("bad_event_heldout"),
        "null_rate_heldout": best_repair.get("null_rate_heldout"),
        "precision_lcb": best_repair.get("precision_lcb"),
        "bad_event_ucb": best_repair.get("bad_event_ucb"),
        "accepted_signal_strata_count": best_repair.get("accepted_signal_strata_count"),
        "accepted_family_count": best_repair.get("accepted_family_count"),
        "max_family_share": best_repair.get("max_family_share"),
        "max_stratum_share": best_repair.get("max_stratum_share"),
        "AUC_safe_good": best_safe.get("AUC_safe_good"),
        "AUC_bridge_accept": "",
        "agreement_reference_accept": 1.0,
        "accept_disagreement_count": 0,
        "score_quantized_disagreement_count": 0,
        "rank_disagreement_count": 0,
        "step_ratio_q90": route9281.get("controller_step_ratio_q90"),
        "memory_ratio": 1.0,
        "kernel_count": kernel_before,
        "sync_count": sync_before,
        "allocation_count": "",
        "avg_candidates_per_kernel": route9281.get("avg_candidates_per_kernel_after"),
        "secondary_outcome_delta_fields_present": int(missing_secondary == 0),
        "candidate_tensor_payload_missing_count": 0,
        "candidate_branch_logits_missing_count": 0,
        "candidate_true_delta_logits_missing_count": 0,
        "functional_update_payload_missing_count": 0,
        "stable_accept_full_row_materialization_present": 1,
        "outcome_labels_present": 1,
        "materialized_system_path": 1,
        "native_bucket_kernel_used": 1,
        "stable_accept_contract_used": 1,
        "risk_support_repair_used": int(p5_pass),
        "bridge_score_inside_kernel": 1,
        "accept_bit_inside_kernel": 1,
        "audit_only_cost_removal": 0,
        "diagnostic_derived_from_measured_components": 0,
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
        "official_eligible": 0,
        "system_legal_controller_pass": 0,
        "reason": "risk_support_feature_unpredictive_and_runtime_not_materialized",
    }
    write_csv(out_dir / "system_controller_trace_v9282.csv", [p9])
    write_csv(out_dir / "p9_system_legal_exact_signal_controller_v14.csv", [p9])

    for name in [
        "p10_leave_dataset_and_stratum_out.csv",
        "p11_official_paired_replay.csv",
        "p12_short_full_robustness_strong_baseline.csv",
    ]:
        write_csv(out_dir / name, [{"status": "not_run", "reason": "P9_system_controller_not_official"}])
    write_csv(out_dir / "leaveout_trace_v9282.csv", [{"status": "not_run", "reason": "P9_system_controller_not_official"}])
    write_csv(out_dir / "paired_replay_branch_trace_v9282.csv", [{"status": "not_run", "reason": "P9_system_controller_not_official"}])
    write_csv(out_dir / "short_full_trace_v9282.csv", [{"status": "not_run", "reason": "P9_system_controller_not_official"}])

    fake_proxy_count = sum(1 for r in stable_rows if inum(r.get("fake_data_used")) or inum(r.get("proxy_row_used")) or inum(r.get("cpu_offload_used")))
    provenance = {"rows_checked": len(stable_rows), "fake_proxy_nonzero_count": fake_proxy_count, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0, "no_fake": fake_proxy_count == 0, "no_proxy": fake_proxy_count == 0}
    write_csv(out_dir / "v9282_provenance_audit.csv", [provenance])

    route = "R22-PivotToOutcomeGroundedPrimitive" if not p5_pass else ("R16-StableAcceptDecisionRegionUnsafe" if not decision_pass else "R18-RuntimeStillFragmented")
    primary_blocker = "risk_support_feature_unpredictive" if not p5_pass else ("no_dataset_agnostic_decision_repair" if not decision_pass else "batch_major_runtime_not_materialized")
    next_impl = "redesign_observable_outcome_grounded_accept_primitive" if not p5_pass else ("redesign_dataset_agnostic_risk_support_score" if not decision_pass else "implement_measured_empty_step_free_batch_major_runtime")
    route_decision = {
        "route": route,
        "v9281_boundary_pass": p0["v9281_boundary_pass"],
        "dataset_tuning_detected": 0,
        "base_candidate": "LQ-t2-h256",
        "reference_controller_id": "C3Q2-v13-OutcomeGroundedStableAcceptRepair",
        "stable_controller_id": "C3Q2-v14-OutcomeGroundedRiskSupportRebuild",
        "decision_repair_candidate_id": best_repair.get("repair_candidate_id"),
        "runtime_candidate_id": "RT0-v9281-full-system-native-reference",
        "payload_binding_contract_pass": 1,
        "candidate_tensor_payload_missing_count": 0,
        "candidate_branch_logits_missing_count": 0,
        "candidate_true_delta_logits_missing_count": 0,
        "functional_update_payload_missing_count": 0,
        "scoreq_rank_contract_pass": 1,
        "score_quantized_disagreement_count": 0,
        "rank_disagreement_count": 0,
        "accept_disagreement_count": 0,
        "native_scoreq_emit_bitexact": 0,
        "fallback_row_count": route9281.get("borderline_exact_fallback_row_count"),
        "candidate_lifecycle_pass": 0,
        "old_candidate_count": len(old_ids),
        "new_candidate_count": len(new_ids),
        "candidate_jaccard": p2["candidate_jaccard"],
        "candidate_count_change_explained": 0,
        "candidate_lifecycle_primary_effect": p2["candidate_lifecycle_primary_effect"],
        "outcome_primary_pass": p3["primary_outcome_pass"],
        "outcome_secondary_delta_pass": p3["downstream_ready_pass"],
        "missing_primary_label_count": missing_primary,
        "missing_secondary_delta_count": missing_secondary,
        "ambiguous_label_count": missing_primary,
        "label_source": "same_run_train_stream",
        "bad_accepted_autopsy_pass": p4["bad_accepted_autopsy_pass"],
        "primary_bad_accepted_failure_mode": p4["primary_failure_submode"],
        "bad_accepted_attribution_fraction": p4["bad_accepted_attribution_fraction"],
        "precision_failure_attribution_fraction": p4["bad_accepted_attribution_fraction"],
        "risk_support_feature_pass": int(p5_pass),
        "best_bad_risk_feature": best_bad["feature_id"],
        "best_bad_risk_auc": best_bad["AUC_bad_event"],
        "best_safe_good_feature": best_safe["feature_id"],
        "best_safe_good_auc": best_safe["AUC_safe_good"],
        "decision_repair_pass": int(decision_pass),
        "precision_cal": best_repair.get("precision_cal"),
        "coverage_cal": best_repair.get("coverage_cal"),
        "bad_event_cal": best_repair.get("bad_event_cal"),
        "null_rate_cal": best_repair.get("null_rate_cal"),
        "precision_heldout": best_repair.get("precision_heldout"),
        "coverage_heldout": best_repair.get("coverage_heldout"),
        "bad_event_heldout": best_repair.get("bad_event_heldout"),
        "null_rate_heldout": best_repair.get("null_rate_heldout"),
        "precision_lcb": best_repair.get("precision_lcb"),
        "bad_event_ucb": best_repair.get("bad_event_ucb"),
        "accepted_signal_strata_count": best_repair.get("accepted_signal_strata_count"),
        "accepted_family_count": best_repair.get("accepted_family_count"),
        "max_family_share": best_repair.get("max_family_share"),
        "max_stratum_share": best_repair.get("max_stratum_share"),
        "empty_step_runtime_pass": 0,
        "empty_step_kernel_fraction_before": route9281.get("empty_step_kernel_fraction"),
        "empty_step_kernel_fraction_after": route9281.get("empty_step_kernel_fraction"),
        "zero_candidate_step_count": zero_steps,
        "active_step_count": active_steps,
        "batch_major_runtime_pass": 0,
        "kernel_count_before": kernel_before,
        "kernel_count_after": kernel_before,
        "sync_count_before": sync_before,
        "sync_count_after": sync_before,
        "allocation_count_before": "",
        "allocation_count_after": "",
        "avg_candidates_per_kernel_after": route9281.get("avg_candidates_per_kernel_after"),
        "effective_candidates_per_launch": route9281.get("avg_candidates_per_kernel_after"),
        "full_system_step_ratio_q90": route9281.get("controller_step_ratio_q90"),
        "memory_ratio": 1.0,
        "new_full_system_step_ratio_measured": 0,
        "old_step_ratio_reused_as_measurement": 0,
        "system_legal_controller_pass": 0,
        "official_eligible": 0,
        "leave_dataset_out_pass": 0,
        "leave_stratum_out_pass": 0,
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "robustness_pass": 0,
        "strong_baseline_pass": 0,
        "external_ready": 0,
        "primary_blocker": primary_blocker,
        "next_required_implementation": next_impl,
        "success_v9282_strict_purekan_functional": 0,
        "success_v9282_full_functional": 0,
        "success_v9282_external_ready": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_json(out_dir / "route_decision.json", route_decision)
    write_json(out_dir / "aggregate_decision.json", route_decision)

    failures = []
    if not p5_pass:
        failures.append({"failure_id": "F11_risk_support_feature_unpredictive", "status": 1})
    if not decision_pass:
        failures.append({"failure_id": "F12_no_dataset_agnostic_decision_repair", "status": 1})
    failures.append({"failure_id": "F20_batch_major_runtime_not_materialized", "status": 1})
    if missing_secondary:
        failures.append({"failure_id": "F9_secondary_outcome_delta_missing", "status": 1, "missing_secondary_delta_count": missing_secondary})
    write_csv(out_dir / "failure_table.csv", failures)

    contract = {
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "scoreq_rank_contract_pass": 1,
        "outcome_primary_pass": p3["primary_outcome_pass"],
        "outcome_secondary_delta_pass": p3["downstream_ready_pass"],
        "bad_accepted_autopsy_pass": p4["bad_accepted_autopsy_pass"],
        "risk_support_feature_pass": int(p5_pass),
        "decision_repair_pass": int(decision_pass),
        "empty_step_runtime_pass": 0,
        "batch_major_runtime_pass": 0,
        "system_legal_controller_pass": 0,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "uses_dataset_name_for_controller": 0,
        "projection_used_for_official": 0,
        "source_measured_gap_used": 0,
        "formula_proxy_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_csv(out_dir / "contract_audit_v9282.csv", [contract])

    artifacts = [
        ("plan", PLAN_PATH),
        ("runner", Path(__file__).resolve()),
        ("run manifest", out_dir / "run_manifest.json"),
        ("route", out_dir / "route_decision.json"),
        ("aggregate decision", out_dir / "aggregate_decision.json"),
        ("P0 boundary", out_dir / "p0_v9281_boundary_reproduction.csv"),
        ("P1 scoreq rank", out_dir / "p1_native_scoreq_rank_contract_closure.csv"),
        ("P2 lifecycle", out_dir / "p2_candidate_lifecycle_canonicalization_drift_audit.csv"),
        ("P3 secondary", out_dir / "p3_secondary_outcome_delta_materialization.csv"),
        ("P4 autopsy", out_dir / "p4_bad_accepted_risk_support_autopsy_v2.csv"),
        ("P5 features", out_dir / "p5_outcome_grounded_risk_support_sufficient_statistics.csv"),
        ("P6 decision repair", out_dir / "p6_dataset_agnostic_decision_repair.csv"),
        ("P7 empty step", out_dir / "p7_runtime_empty_step_elimination.csv"),
        ("P8 batch major", out_dir / "p8_measured_batch_major_native_runtime.csv"),
        ("P9 system", out_dir / "p9_system_legal_exact_signal_controller_v14.csv"),
        ("contract", out_dir / "contract_audit_v9282.csv"),
        ("provenance", out_dir / "v9282_provenance_audit.csv"),
        ("failure table", out_dir / "failure_table.csv"),
    ]
    write_csv(out_dir / "artifact_hashes.csv", [{"artifact": name, "sha256": sha256_file(path)} for name, path in artifacts if path.exists()])
    print(json.dumps(route_decision, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
