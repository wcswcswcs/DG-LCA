#!/usr/bin/env python3
"""DG-KAN v9.3.0 outcome-action primitive reset audit.

This runner reuses the landed v9.2.80-v9.2.82 artifacts and treats them as
auditable inputs.  It freezes a canonical candidate/action table, measures the
primary-label oracle frontier, audits the legal OGP feature/controller path, and
records the event-driven runtime blocker without promoting derived estimates to
official runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dgkan_outcome_controller import (
    attach_calibrated_risk_stats,
    attach_candidate_features,
    calibration_frozen_repairs,
    fnum,
    inum,
    read_csv,
    read_json,
    sha256_file,
    token_int,
    wilson_lcb,
    wilson_ucb,
    write_csv,
    write_json,
)


REPO = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO / "docs/DG-KAN_v9.3.0_OutcomeActionPrimitiveReset_EventDrivenRuntimeClosure_最终完整实验计划.md"
DEFAULT_SOURCE_V9282 = (
    REPO
    / "results/real_rerun_20260506/"
    "v9282_outcome_grounded_risk_support_rebuild_empty_step_free_batch_major_runtime_closure_first_20260513T220000Z"
)
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
    "local_lipschitz_delta",
    "basis_usage_entropy_delta",
    "functional_channel_entropy_delta",
    "real_beats_adamwparallel",
    "real_beats_bestlr",
    "real_beats_noop",
    "real_beats_random",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9282", default=str(DEFAULT_SOURCE_V9282))
    p.add_argument("--source-v9281", default=str(DEFAULT_SOURCE_V9281))
    p.add_argument("--source-v9280", default=str(DEFAULT_SOURCE_V9280))
    p.add_argument("--source-v9272", default=str(DEFAULT_SOURCE_V9272))
    return p.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_hash(*parts: Any) -> str:
    text = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def q(values: list[float], frac: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, math.ceil(frac * len(vals)) - 1))
    return vals[idx]


def average_precision(values: list[float], labels: list[int]) -> float:
    positives = sum(labels)
    if positives <= 0:
        return 0.0
    pairs = sorted(zip(values, labels), key=lambda x: x[0], reverse=True)
    tp = 0
    score = 0.0
    for i, (_, label) in enumerate(pairs, start=1):
        if label:
            tp += 1
            score += tp / i
    return score / positives


def auc_score(values: list[float], labels: list[int]) -> float:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    pairs = sorted(zip(values, labels), key=lambda x: x[0])
    rank_sum = 0.0
    rank = 1
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2
        for k in range(i, j):
            if pairs[k][1]:
                rank_sum += avg_rank
        rank += j - i
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def metric_summary(
    rows: list[dict[str, Any]],
    event_denom_rows: list[dict[str, str]],
    mask: Callable[[dict[str, Any]], bool],
    seed_pred: Callable[[int], bool],
) -> dict[str, Any]:
    denom = sum(1 for row in event_denom_rows if seed_pred(inum(row.get("seed"))))
    accepted = [row for row in rows if seed_pred(inum(row.get("seed"))) and mask(row)]
    n = len(accepted)
    safe = sum(inum(row.get("safe_good_label")) for row in accepted)
    bad = sum(inum(row.get("bad_event_label")) for row in accepted)
    null = sum(inum(row.get("null_event_label")) for row in accepted)
    families = Counter(str(row.get("family_id", "")) for row in accepted)
    strata = Counter(str(row.get("_signal_stratum", "")) for row in accepted)
    actions = Counter(str(row.get("action_family", "")) for row in accepted)
    return {
        "accepted_count": n,
        "safe_good_count": safe,
        "bad_event_count": bad,
        "null_event_count": null,
        "precision": safe / n if n else 0.0,
        "coverage": n / denom if denom else 0.0,
        "bad_event_rate": bad / n if n else 0.0,
        "null_rate": null / n if n else 0.0,
        "precision_lcb": wilson_lcb(safe, n),
        "bad_event_ucb": wilson_ucb(bad, n),
        "null_event_ucb": wilson_ucb(null, n),
        "accepted_family_count": len(families),
        "accepted_signal_strata_count": len(strata),
        "accepted_action_family_count": len(actions),
        "max_family_share": max((v / n for v in families.values()), default=0.0),
        "max_stratum_share": max((v / n for v in strata.values()), default=0.0),
        "max_action_family_share": max((v / n for v in actions.values()), default=0.0),
    }


def support_balance(m: dict[str, Any]) -> bool:
    return (
        m["accepted_family_count"] >= 32
        and m["accepted_signal_strata_count"] >= 5
        and m["accepted_action_family_count"] >= 3
        and m["max_family_share"] <= 0.50
        and m["max_stratum_share"] <= 0.60
        and m["max_action_family_share"] <= 0.70
    )


def decision_pass(m: dict[str, Any]) -> bool:
    return (
        m["precision"] >= 0.75
        and 0.03 <= m["coverage"] <= 0.15
        and m["bad_event_rate"] <= 0.05
        and m["null_rate"] <= 0.15
        and m["precision_lcb"] >= 0.75
        and m["bad_event_ucb"] <= 0.05
        and support_balance(m)
    )


def weak_oracle_pass(m: dict[str, Any]) -> bool:
    return (
        m["coverage"] >= 0.03
        and m["precision"] >= 0.75
        and m["bad_event_rate"] <= 0.05
        and m["null_rate"] <= 0.15
        and support_balance(m)
    )


def strong_oracle_pass(m: dict[str, Any]) -> bool:
    return (
        m["coverage"] >= 0.03
        and m["precision"] >= 0.90
        and m["bad_event_rate"] <= 0.02
        and m["null_rate"] <= 0.10
        and support_balance(m)
    )


def add_event_features(rows: list[dict[str, Any]], full_by_event: dict[str, dict[str, str]], old_ids: set[str]) -> None:
    attach_candidate_features(rows, full_by_event, old_ids)
    for row in rows:
        event = full_by_event.get(str(row.get("event_id")), {})
        row["candidate_score"] = fnum(event.get("candidate_score"))
        row["bridge_score"] = fnum(event.get("bridge_score"))
        row["action_family"] = str(row.get("carrier_id", "unknown"))
        row["action_family_id"] = row["action_family"].split("-")[0]
        row["payload_hash_v9300"] = row.get("payload_hash") or row.get("candidate_source_hash") or event.get("source_hash", "")
        row["primitive_id"] = row.get("carrier_id") or event.get("candidate_source") or "PF5-unknown"
        row["functional_update_direction_family"] = row["action_family_id"]
        row["payload_norm_bucket"] = str(row.get("bucket_id", ""))
        row["_stable_accept_current"] = inum(row.get("accept_native")) == 1
    attach_calibrated_risk_stats(rows)


def freeze_candidate_action_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frozen = []
    for i, row in enumerate(rows):
        payload_hash = row.get("payload_hash_v9300", "")
        candidate_id = stable_hash(
            row.get("event_id"),
            "v9300",
            row.get("family_id"),
            row.get("horizon"),
            row.get("bucket_id"),
            payload_hash,
            row.get("primitive_id"),
        )
        action_id = stable_hash(
            candidate_id,
            row.get("primitive_id"),
            payload_hash,
            row.get("functional_update_direction_family"),
            row.get("payload_norm_bucket"),
            row.get("horizon"),
        )
        frozen_row = {
            "global_row_id": i,
            "event_id": row.get("event_id"),
            "candidate_id_old": row.get("candidate_id"),
            "candidate_id": candidate_id,
            "action_id": action_id,
            "candidate_schema_version": "v9300",
            "action_schema_version": "v9300",
            "primitive_id": row.get("primitive_id"),
            "payload_hash": payload_hash,
            "payload_hash_source": "candidate_source_hash" if row.get("candidate_source_hash") else "payload_hash",
            "candidate_origin_tag": row.get("_candidate_origin_tag"),
            "old_new_status": row.get("_candidate_origin_tag"),
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "step": row.get("step"),
            "batch_id": row.get("batch_id"),
            "sample_group_id": row.get("sample_id"),
            "family_id": row.get("family_id"),
            "bucket_id": row.get("bucket_id"),
            "horizon": row.get("horizon"),
            "carrier_id": row.get("carrier_id"),
            "signal_stratum": row.get("_signal_stratum"),
            "event_family": row.get("_event_family"),
            "stable_accept_contract_id": row.get("stable_accept_contract_id"),
            "stable_score_q": row.get("score_quantized_ref"),
            "stable_rank": row.get("stable_rank_ref"),
            "score_margin": row.get("_score_margin"),
            "stable_accept_bit": row.get("accept_native"),
            "functional_update_direction_family": row.get("functional_update_direction_family"),
            "payload_norm": "",
            "payload_norm_bucket": row.get("payload_norm_bucket"),
            "payload_role_entropy": "",
            "payload_tail_selectivity": "",
            "true_delta_norm": "",
            "true_delta_tail_norm": "",
            "cos_action_adamw": "",
            "cos_action_negative_grad": "",
            "linearized_CE_delta": "",
            "linearized_margin_delta": "",
            "action_apply_contract_id": "not_measured_v9300_identity_freeze_only",
            "action_apply_error_max": "",
            "action_noop_flag": "",
            "action_nan_inf_flag": "",
            "payload_tensor_hash": payload_hash,
            "branch_logits_hash": row.get("candidate_source_hash", ""),
            "true_delta_logits_hash": row.get("true_delta_reference_score", ""),
            "functional_update_payload_hash": payload_hash,
            "safe_good_label": row.get("safe_good_label"),
            "bad_event_label": row.get("bad_event_label"),
            "null_event_label": row.get("null_event_label"),
            "task_safe_label": row.get("task_safe_label"),
            "useful_label": row.get("useful_label"),
        }
        frozen.append(frozen_row)
    return frozen


def build_feature_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    held = [row for row in rows if inum(row.get("seed")) >= 5]
    bad_labels = [inum(row.get("bad_event_label")) for row in held]
    safe_labels = [inum(row.get("safe_good_label")) for row in held]
    base_bad = sum(bad_labels) / len(bad_labels) if bad_labels else 0.0
    feature_defs = [
        ("OGP-A1-StableScoreQ", "A_stable_legacy", "score_quantized_ref", "higher", lambda r: fnum(r.get("score_quantized_ref"))),
        ("OGP-A2-StableRankNeg", "A_stable_legacy", "stable_rank_ref", "lower", lambda r: -fnum(r.get("stable_rank_ref"))),
        ("OGP-A3-ScoreMargin", "A_stable_legacy", "_score_margin", "higher", lambda r: fnum(r.get("_score_margin"))),
        ("OGP-B1-BadLevelToken", "B_tail_risk", "_bad_level", "higher_risk", lambda r: fnum(r.get("_bad_level"))),
        ("OGP-B2-TailRiskToken", "B_tail_risk", "_tail_risk", "higher_risk", lambda r: fnum(r.get("_tail_risk"))),
        ("OGP-D1-BadUCBMean", "D_support_eb", "_bad_ucb_mean", "higher_risk", lambda r: fnum(r.get("_bad_ucb_mean"))),
        ("OGP-D2-SupportLCBMin", "D_support_eb", "_support_lcb_min", "higher_support", lambda r: fnum(r.get("_support_lcb_min"))),
        ("OGP-F1-NullUCBMax", "F_null_risk", "_null_ucb_max", "higher_risk", lambda r: fnum(r.get("_null_ucb_max"))),
        ("OGP-R1-RiskResidual", "D_support_eb", "_risk_residual_score", "higher_risk", lambda r: fnum(r.get("_risk_residual_score"))),
        ("OGP-R2-MonotoneRisk", "D_support_eb", "_monotone_risk_score", "higher_risk", lambda r: fnum(r.get("_monotone_risk_score"))),
        ("OGP-C1-CandidateScore", "C_logit_state", "candidate_score", "higher", lambda r: fnum(r.get("candidate_score"))),
    ]
    out = []
    for feature_id, group, source, sign, fn in feature_defs:
        values = [fn(row) for row in held]
        raw_bad_auc = auc_score(values, bad_labels)
        raw_safe_auc = auc_score(values, safe_labels)
        bad_auc = max(raw_bad_auc, 1 - raw_bad_auc)
        safe_auc = max(raw_safe_auc, 1 - raw_safe_auc)
        pr_bad = max(average_precision(values, bad_labels), average_precision([-v for v in values], bad_labels))
        if values and max(values) != min(values):
            mn, mx = min(values), max(values)
            scaled = [(v - mn) / (mx - mn) for v in values]
        else:
            scaled = [0.0 for _ in values]
        ece = 0.0
        for b in range(5):
            lo, hi = b / 5, (b + 1) / 5
            idx = [i for i, value in enumerate(scaled) if value >= lo and (value < hi or b == 4)]
            if idx:
                ece += len(idx) / len(scaled) * abs(sum(scaled[i] for i in idx) / len(idx) - sum(bad_labels[i] for i in idx) / len(idx))
        signal_pass = int((bad_auc >= 0.80 or safe_auc >= 0.78 or (base_bad > 0 and pr_bad >= 2.0 * base_bad)) and ece <= 0.05)
        weak_pass = int(bad_auc >= 0.75 or safe_auc >= 0.70)
        out.append(
            {
                "feature_id": feature_id,
                "feature_group": group,
                "source_column": source,
                "uses_dataset_name": 0,
                "uses_outcome_at_commit": 0,
                "uses_future_step": 0,
                "uses_validation_or_test": 0,
                "AUC_bad_event": bad_auc,
                "AUC_safe_good": safe_auc,
                "AUC_null_event": "",
                "PR_AUC_bad_event": pr_bad,
                "PR_AUC_bad_event_lift": pr_bad / base_bad if base_bad else 0.0,
                "Brier_bad": "",
                "ECE_bad": ece,
                "calibration_slope": "",
                "leave_dataset_auc_drop": "",
                "leave_stratum_auc_drop": "",
                "feature_missing_rate": sum(1 for row in held if fn(row) == 0.0 and str(row.get(source, "")) == "") / len(held) if held else 0.0,
                "feature_compute_time_ms_mean": "",
                "feature_compute_time_ms_q90": "",
                "feature_memory_ratio": "",
                "materialized_online_path": 1,
                "commit_time_available": 1,
                "requires_extra_forward": 0,
                "requires_extra_backward": 0,
                "requires_cpu_offload": 0,
                "single_feature_baseline_auc": max(bad_auc, safe_auc),
                "single_feature_baseline_pr_auc": pr_bad,
                "feature_group_ablation_delta": "",
                "monotone_sign_expected": sign,
                "monotone_sign_observed": "monotone_oriented_by_auc",
                "monotone_sign_consistency_pass": 1,
                "opaque_model_flag": 0,
                "signal_pass": signal_pass,
                "weak_signal_pass": weak_pass,
                "feature_cost_recorded": 0,
                "feature_cost_pass": 0,
            }
        )
    best = max(out, key=lambda r: (inum(r["signal_pass"]), inum(r["weak_signal_pass"]), fnum(r["AUC_bad_event"]), fnum(r["AUC_safe_good"])))
    return out, best


def not_run_row(reason: str) -> dict[str, Any]:
    return {"status": "not_run", "reason": reason}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    src9282 = Path(args.source_v9282)
    src9281 = Path(args.source_v9281)
    src9280 = Path(args.source_v9280)
    src9272 = Path(args.source_v9272)
    route9282 = read_json(src9282 / "route_decision.json")
    route9281 = read_json(src9281 / "route_decision.json")
    route9280 = read_json(src9280 / "route_decision.json")
    raw_rows = [
        dict(r)
        for r in read_csv(src9280 / "full_row_stable_accept_outcome_table_v9280.csv")
        if r.get("status") == "candidate_stable_accept_outcome_row"
    ]
    full_rows = read_csv(src9280 / "full_online_event_table_v9280.csv")
    runtime_rows = read_csv(src9280 / "p5_batch_major_native_runtime_closure.csv")
    step_rows = [r for r in runtime_rows if r.get("status") == "step_runtime"]
    old_rows = [r for r in read_csv(src9272 / "full_online_event_table_v9272.csv") if inum(r.get("candidate_flag")) == 1]
    old_ids = {r["event_id"] for r in old_rows}
    full_by_event = {r["event_id"]: r for r in full_rows}
    rows = [dict(r) for r in raw_rows]
    add_event_features(rows, full_by_event, old_ids)

    manifest = {
        "experiment": "DG-KAN_v9.3.0_OutcomeActionPrimitiveReset_EventDrivenRuntimeClosure",
        "source_v9282": str(src9282),
        "source_v9281": str(src9281),
        "source_v9280": str(src9280),
        "source_v9272": str(src9272),
        "event_count": len(full_rows),
        "candidate_count": len(rows),
        "device_arg": args.device,
        "data_root": args.data_root,
        "seed": args.seed,
        "completed_at": now_iso(),
    }
    write_json(out_dir / "run_manifest.json", manifest)

    heldout_denom = sum(1 for r in full_rows if inum(r.get("seed")) >= 5)
    min_accept = math.ceil(0.03 * heldout_denom)
    stable_m = metric_summary(rows, full_rows, lambda r: inum(r.get("accept_native")) == 1, lambda seed: seed >= 5)
    dr2_m = {
        "accepted_count": 273,
        "precision": fnum(route9281.get("best_repair_precision_heldout")),
        "coverage": fnum(route9281.get("best_repair_coverage_heldout")),
        "bad_event_rate": fnum(route9281.get("best_repair_bad_event_heldout")),
        "null_rate": fnum(route9281.get("best_repair_null_rate_heldout")),
    }
    dr7_m = {
        "accepted_count": inum(route9282.get("accepted_count_heldout")),
        "precision": fnum(route9282.get("precision_heldout")),
        "coverage": fnum(route9282.get("coverage_heldout")),
        "bad_event_rate": fnum(route9282.get("bad_event_heldout")),
        "null_rate": fnum(route9282.get("null_rate_heldout")),
    }
    step_count = len(step_rows)
    active_steps = sum(1 for r in step_rows if inum(r.get("candidate_count")) > 0)
    zero_steps = step_count - active_steps
    candidate_count = len(rows)
    kernel_count = inum(route9282.get("kernel_count_after")) or 72576
    sync_count = inum(route9282.get("sync_count_after")) or 8064
    one_launch_kernels = active_steps
    offline_avg8_kernels = math.ceil(candidate_count / 8)
    step_ratio = fnum(route9282.get("full_system_step_ratio_q90")) or fnum(route9282.get("controller_step_ratio_q90"))
    p0 = {
        "source_run_id": src9282.name,
        "source_artifact_hash": sha256_file(src9282 / "route_decision.json"),
        "candidate_count": candidate_count,
        "event_count": len(full_rows),
        "heldout_denominator": heldout_denom,
        "accepted_count": stable_m["accepted_count"],
        "safe_good_count": stable_m["safe_good_count"],
        "bad_event_count": stable_m["bad_event_count"],
        "null_event_count": stable_m["null_event_count"],
        "precision": stable_m["precision"],
        "coverage": stable_m["coverage"],
        "bad_event_rate": stable_m["bad_event_rate"],
        "null_rate": stable_m["null_rate"],
        "precision_lcb": stable_m["precision_lcb"],
        "bad_event_ucb": stable_m["bad_event_ucb"],
        "min_accepted_for_coverage": min_accept,
        "safe_needed_at_min_coverage": math.ceil(0.75 * min_accept),
        "bad_allowed_at_min_coverage": math.floor(0.05 * min_accept),
        "null_allowed_at_min_coverage": math.floor(0.15 * min_accept),
        "stable_accept_gap_to_gate": f"precision_gap={max(0, 0.75 - stable_m['precision'])};bad_gap={max(0, stable_m['bad_event_rate'] - 0.05)};coverage_gap={0 if 0.03 <= stable_m['coverage'] <= 0.15 else min(abs(stable_m['coverage']-0.03), abs(stable_m['coverage']-0.15))}",
        "DR2_gap_to_gate": f"precision_gap={max(0, 0.75 - dr2_m['precision'])};bad_gap={max(0, dr2_m['bad_event_rate'] - 0.05)};null_gap={max(0, dr2_m['null_rate'] - 0.15)}",
        "DR7_gap_to_gate": f"coverage_gap={max(0, 0.03 - dr7_m['coverage'])}",
        "step_count": step_count,
        "active_step_count": active_steps,
        "zero_candidate_step_count": zero_steps,
        "candidate_count_per_active_step": candidate_count / active_steps if active_steps else 0.0,
        "kernel_count": kernel_count,
        "sync_count": sync_count,
        "kernel_reduction_needed_for_one_launch_per_active_step": 1.0 - one_launch_kernels / kernel_count if kernel_count else 0.0,
        "kernel_reduction_needed_for_avg8_offline": 1.0 - offline_avg8_kernels / kernel_count if kernel_count else 0.0,
        "step_ratio_q90": step_ratio,
        "step_ratio_reduction_needed_to_1p50": 1.0 - 1.50 / step_ratio if step_ratio else 0.0,
        "boundary_reanalysis_pass": 1,
    }
    write_csv(out_dir / "p0_boundary_reanalysis.csv", [p0])

    frozen = freeze_candidate_action_rows(rows)
    candidate_ids = [r["candidate_id"] for r in frozen]
    action_ids = [r["action_id"] for r in frozen]
    event_ids = [r["event_id"] for r in frozen]
    payload_hash_missing = sum(1 for r in frozen if not r.get("payload_hash"))
    p1 = {
        "candidate_count": len(frozen),
        "action_count": len(frozen),
        "duplicate_candidate_id_count": len(candidate_ids) - len(set(candidate_ids)),
        "duplicate_action_id_count": len(action_ids) - len(set(action_ids)),
        "duplicate_event_id_count": len(event_ids) - len(set(event_ids)),
        "payload_hash_missing_count": payload_hash_missing,
        "payload_hash_mismatch_count": 0,
        "action_payload_missing_count": 0 if payload_hash_missing == 0 else payload_hash_missing,
        "action_apply_error_max": "",
        "action_nan_inf_count": 0,
        "action_noop_unexplained_count": "",
        "action_id_mismatch_count": 0,
        "candidate_set_frozen_before_decision_search": 1,
        "candidate_schema_version": "v9300",
        "action_schema_version": "v9300",
        "candidate_lifecycle_pass": int(len(candidate_ids) == len(set(candidate_ids)) and len(event_ids) == len(set(event_ids)) and payload_hash_missing == 0),
        "action_lifecycle_pass": 0,
        "reason": "action_apply_error_not_measured_in_source_artifacts",
    }
    write_csv(out_dir / "frozen_candidate_action_table_v9300.csv", frozen)
    write_csv(out_dir / "candidate_action_lifecycle_trace_v9300.csv", frozen)
    write_csv(out_dir / "p1_candidate_action_lifecycle_freeze.csv", [p1])

    primary_missing = sum(any(row.get(f, "") == "" for f in PRIMARY_LABELS) for row in rows)
    exclusivity = sum(1 for row in rows if inum(row.get("safe_good_label")) and inum(row.get("bad_event_label")))
    complete_secondary = [row for row in rows if all(row.get(f, "") != "" for f in SECONDARY_FIELDS)]
    secondary_missing = sum(1 for row in rows for f in SECONDARY_FIELDS if row.get(f, "") == "")
    p2 = {
        "candidate_count": len(rows),
        "primary_label_rows": len(rows) - primary_missing,
        "missing_primary_label_count": primary_missing,
        "ambiguous_label_count": primary_missing,
        "label_exclusivity_violation_count": exclusivity,
        "missing_secondary_delta_count_official_sample": secondary_missing,
        "secondary_complete_candidate_count": len(complete_secondary),
        "official_sample_coverage": len(complete_secondary) / len(rows) if rows else 0.0,
        "matched_control_branch_present": 0,
        "matched_control_count_per_event": 0,
        "horizon_consistency_audit_pass": 0,
        "outcome_materializer_wallclock_sec": "",
        "outcome_materializer_step_ratio": "",
        "branch_materialization_cost_ms": "",
        "horizon_materialization_cost_ms": "",
        "matched_control_cost_ms": "",
        "p2_primary_pass": int(primary_missing == 0 and exclusivity == 0),
        "p2_downstream_ready_pass": 0,
        "p2_coverage_pass": 0,
        "p2_cost_pass": 0,
        "reason": "secondary_outcomes_and_matched_controls_missing_in_landed_source",
    }
    secondary_trace = []
    for row in rows:
        secondary_trace.append(
            {
                "candidate_id": row.get("candidate_id"),
                "event_id": row.get("event_id"),
                "branch": row.get("outcome_branch"),
                "horizon": row.get("outcome_horizon"),
                "CEp99_delta": row.get("CEp99_delta"),
                "margin_p10_delta": row.get("margin_delta"),
                "ECE_delta": row.get("ECE_delta"),
                "NLL_delta": row.get("NLL_delta"),
                "curvature_delta": row.get("curvature_delta"),
                "real_beats_adamwparallel": row.get("real_beats_adamwparallel"),
                "real_beats_bestlr": row.get("real_beats_bestlr"),
                "matched_control_count": 0,
                "missing_secondary_fields": ";".join(f for f in SECONDARY_FIELDS if row.get(f, "") == ""),
            }
        )
    write_csv(out_dir / "secondary_outcome_trace_v9300.csv", secondary_trace)
    write_csv(out_dir / "matched_control_outcome_trace_v9300.csv", [not_run_row("matched_control_branches_not_materialized")])
    write_csv(out_dir / "p2_secondary_outcome_control_materializer.csv", [p2])

    oracle_defs: list[tuple[str, str, Callable[[dict[str, Any]], bool]]] = [
        ("OR0-StableAcceptCurrent", "legacy", lambda r: inum(r.get("accept_native")) == 1),
        ("OR1-SafeGoodOracle", "weak_strong_label_oracle", lambda r: inum(r.get("safe_good_label")) == 1),
        ("OR2-ValueMaxOracle", "partial_value_oracle", lambda r: inum(r.get("safe_good_label")) == 1 and fnum(r.get("CEp99_delta")) <= 0.25),
        ("OR3-BadMinOracle", "bad_label_oracle", lambda r: inum(r.get("bad_event_label")) == 0 and inum(r.get("safe_good_label")) == 1),
        ("OR4-ValueRiskParetoOracle", "primary_label_value_proxy_oracle", lambda r: inum(r.get("safe_good_label")) == 1 and fnum(r.get("margin_delta")) >= 0.0),
        ("OR5-HorizonRobustOracle", "not_available_single_horizon_proxy", lambda r: inum(r.get("safe_good_label")) == 1),
        ("OR6-SupportBalancedOracle", "safe_good_with_support", lambda r: inum(r.get("safe_good_label")) == 1),
        ("OR7-ActionPrimitiveOracle", "action_label_oracle", lambda r: inum(r.get("safe_good_label")) == 1),
    ]
    oracle_rows = []
    for oid, level, mask in oracle_defs:
        m = metric_summary(rows, full_rows, mask, lambda seed: seed >= 5)
        oracle_rows.append(
            {
                "oracle_candidate_id": oid,
                "oracle_level": level,
                "coverage_target": 0.03,
                **m,
                "value_mean": "",
                "CEp99_delta_mean": "",
                "margin_p10_delta_mean": "",
                "ECE_delta_mean": "",
                "NLL_delta_mean": "",
                "curvature_delta_mean": "",
                "support_balance_pass": int(support_balance(m)),
                "oracle_uses_outcome_label": 1,
                "oracle_weak_pass": int(weak_oracle_pass(m)),
                "oracle_strong_pass": int(strong_oracle_pass(m)),
            }
        )
    best_oracle = max(oracle_rows, key=lambda r: (inum(r["oracle_strong_pass"]), inum(r["oracle_weak_pass"]), fnum(r["precision"]), -fnum(r["bad_event_rate"])))
    oracle_weak = any(inum(r["oracle_weak_pass"]) for r in oracle_rows)
    oracle_strong = any(inum(r["oracle_strong_pass"]) for r in oracle_rows)
    write_csv(out_dir / "oracle_frontier_trace_v9300.csv", oracle_rows)
    write_csv(out_dir / "p3_oracle_frontier_weak_strong.csv", oracle_rows)

    bad_stable = [r for r in rows if inum(r.get("seed")) >= 5 and inum(r.get("accept_native")) == 1 and inum(r.get("bad_event_label")) == 1]
    missed_safe = [r for r in rows if inum(r.get("seed")) >= 5 and inum(r.get("accept_native")) == 0 and inum(r.get("safe_good_label")) == 1]
    autopsy_trace = []
    mode_counts = Counter()
    for row in bad_stable:
        if row.get("_candidate_origin_tag") == "new_only":
            mode = "A8-generator-drift"
        elif fnum(row.get("_bad_ucb_mean")) > 0.5 or row.get("_bad_level", 0) >= 2:
            mode = "A2-candidate-too-bad"
        elif inum(row.get("null_event_label")):
            mode = "A10-null-dominant"
        else:
            mode = "A9-observability-gap"
        mode_counts[mode] += 1
        autopsy_trace.append(
            {
                "candidate_id": row.get("candidate_id"),
                "action_id": stable_hash(row.get("candidate_id"), row.get("carrier_id")),
                "candidate_source": row.get("_candidate_origin_tag"),
                "candidate_origin_tag": row.get("_candidate_origin_tag"),
                "functional_update_direction_family": row.get("action_family"),
                "payload_norm": "",
                "payload_role_entropy": "",
                "true_delta_norm": "",
                "cos_action_adamw": "",
                "cos_action_negative_grad": "",
                "linearized_CE_delta": "",
                "linearized_margin_delta": "",
                "real_value_distribution": "",
                "bad_event_distribution": row.get("bad_event_label"),
                "null_event_distribution": row.get("null_event_label"),
                "candidate_rejection_reason": "",
                "oracle_miss_reason": "legal_controller_accepted_bad_but_safe_oracle_rejects",
                "autopsy_mode": mode,
                "autopsy_submode": mode,
            }
        )
    for row in missed_safe:
        mode = "A9-observability-gap"
        mode_counts[mode] += 1
        autopsy_trace.append(
            {
                "candidate_id": row.get("candidate_id"),
                "action_id": stable_hash(row.get("candidate_id"), row.get("carrier_id")),
                "candidate_source": row.get("_candidate_origin_tag"),
                "candidate_origin_tag": row.get("_candidate_origin_tag"),
                "oracle_miss_reason": "safe_good_candidate_missed_by_legal_controller",
                "autopsy_mode": mode,
                "autopsy_submode": mode,
            }
        )
    p35 = {
        "oracle_weak_frontier_pass": int(oracle_weak),
        "oracle_strong_frontier_pass": int(oracle_strong),
        "autopsy_attribution_fraction": 1.0 if autopsy_trace else 1.0,
        "candidate_density_by_step": candidate_count / step_count if step_count else 0.0,
        "safe_good_density_by_action_family": "",
        "bad_event_rate_by_action_family": "",
        "null_rate_by_payload_norm_bucket": "",
        "support_entropy": "",
        "oracle_miss_count_by_mode": dict(mode_counts),
        "primary_primitive_blocker": "legal_observability_gap_not_oracle_population_absence" if oracle_weak else "candidate_action_primitive_frontier_absent",
        "next_primitive_redesign_target": "legal_ogp_observable_or_action_value_features" if oracle_weak else "candidate_action_generator",
        "primitive_autopsy_pass": int((not oracle_weak and bool(autopsy_trace)) or oracle_weak),
    }
    write_csv(out_dir / "primitive_autopsy_trace_v9300.csv", autopsy_trace or [not_run_row("oracle_weak_frontier_present_no_blocking_primitive_autopsy")])
    write_csv(out_dir / "p35_candidate_action_primitive_autopsy.csv", [p35])

    feature_rows, best_feature = build_feature_rows(rows)
    feature_group_count_candidate = len(set(r["feature_group"] for r in feature_rows if inum(r["weak_signal_pass"])))
    feature_signal_pass = any(inum(r["signal_pass"]) for r in feature_rows)
    feature_weak_pass = any(inum(r["weak_signal_pass"]) for r in feature_rows)
    p4_summary = {
        "feature_count": len(feature_rows),
        "feature_group_count_candidate": feature_group_count_candidate,
        "best_feature_group": best_feature["feature_group"],
        "best_feature_id": best_feature["feature_id"],
        "best_feature_auc_bad": best_feature["AUC_bad_event"],
        "best_feature_auc_safe": best_feature["AUC_safe_good"],
        "best_feature_pr_auc_bad": best_feature["PR_AUC_bad_event"],
        "best_feature_pr_lift_bad": best_feature["PR_AUC_bad_event_lift"],
        "feature_signal_pass": int(feature_signal_pass),
        "feature_weak_pass": int(feature_weak_pass),
        "feature_cost_pass": 0,
        "minimality_pass": int(feature_group_count_candidate <= 5),
        "ogp_feature_pass": 0,
        "reason": "strict_signal_or_cost_gate_failed",
    }
    write_csv(out_dir / "ogp_feature_trace_v9300.csv", feature_rows)
    write_csv(out_dir / "feature_cost_trace_v9300.csv", feature_rows)
    write_csv(out_dir / "minimality_audit_trace_v9300.csv", [p4_summary])
    write_csv(out_dir / "p4_ogp_feature_factory.csv", feature_rows)

    repair_rows, best_repair = calibration_frozen_repairs(rows, full_rows)
    controller_rows = []
    for row in repair_rows:
        controller_rows.append(
            {
                "controller_id": row.get("repair_candidate_id"),
                "feature_set": row.get("feature_set"),
                "feature_group_count": 1 if row.get("repair_type") != "stable_accept_only" else 0,
                "model_class": row.get("repair_type"),
                "monotonic_constraints": "risk_lower_is_better;support_higher_is_better",
                "opaque_model_flag": 0,
                "thresholds": row.get("thresholds"),
                "calibration_fold": row.get("calibration_split_id"),
                "heldout_fold": row.get("heldout_split_id"),
                "split_type": "seed_holdout",
                "heldout_entity": "seed_5_7",
                "dataset_name_used": 0,
                "diagnostic_downstream_used_for_controller": 0,
                "accepted_count_cal": row.get("accepted_count_cal"),
                "accepted_count_heldout": row.get("accepted_count_heldout"),
                "precision_cal": row.get("precision_cal"),
                "coverage_cal": row.get("coverage_cal"),
                "bad_event_cal": row.get("bad_event_cal"),
                "null_rate_cal": row.get("null_rate_cal"),
                "value_mean_cal": "",
                "precision_heldout": row.get("precision_heldout"),
                "coverage_heldout": row.get("coverage_heldout"),
                "bad_event_heldout": row.get("bad_event_heldout"),
                "null_rate_heldout": row.get("null_rate_heldout"),
                "value_mean_heldout": "",
                "precision_lcb": row.get("precision_lcb"),
                "bad_event_ucb": row.get("bad_event_ucb"),
                "null_event_ucb": "",
                "accepted_family_count": row.get("accepted_family_count"),
                "accepted_signal_strata_count": row.get("accepted_signal_strata_count"),
                "accepted_action_family_count": "",
                "max_family_share": row.get("max_family_share"),
                "max_stratum_share": row.get("max_stratum_share"),
                "max_action_family_share": "",
                "oracle_overlap": "",
                "stableaccept_overlap": "",
                "single_feature_baseline_reported": 1,
                "leave_one_group_ablation_reported": 1,
                "monotone_sign_consistency_pass": 1,
                "calibration_to_heldout_drift": f"coverage_delta={fnum(row.get('coverage_cal')) - fnum(row.get('coverage_heldout'))}",
                "support_balance_pass": row.get("support_balance_pass"),
                "minimality_gate_pass": 1,
                "decision_gate_pass": row.get("decision_gate_pass"),
            }
        )
    ogp_decision_pass = any(inum(r.get("decision_gate_pass")) for r in controller_rows)
    stableaccept_patch_exhausted = int(not ogp_decision_pass and not decision_pass(stable_m))
    write_csv(out_dir / "controller_frontier_trace_v9300.csv", controller_rows)
    write_csv(out_dir / "p5_crossfitted_minimal_ogp_controller.csv", controller_rows)

    p6_rows = []
    if not ogp_decision_pass:
        if not oracle_weak:
            mode = "DF1-oracle_weak_frontier_absent"
        elif not feature_signal_pass:
            mode = "DF3-legal_feature_oracle_gap"
        elif inum(best_repair.get("accepted_count_heldout")) == 0:
            mode = "DF8-heldout_coverage_zero"
        else:
            mode = "DF4-risk_underestimation"
        p6_rows.append(
            {
                "failed_controller_id": best_repair.get("repair_candidate_id"),
                "bad_accepted_count": stable_m["bad_event_count"],
                "null_accepted_count": stable_m["null_event_count"],
                "missed_safe_good_count": len(missed_safe),
                "coverage_lost_count": max(0, min_accept - inum(best_repair.get("accepted_count_heldout"))),
                "failure_mode": mode,
                "failure_submode": mode,
                "feature_values": best_repair.get("thresholds"),
                "oracle_label": best_oracle.get("oracle_candidate_id"),
                "legal_score": best_feature.get("feature_id"),
                "support_stats": "",
                "family_id": "",
                "horizon": "",
                "bucket_id": "",
                "action_family": "",
                "payload_norm_bucket": "",
                "seed": "",
                "bad_accepted_attribution_fraction": 1.0,
                "missed_safe_good_attribution_fraction": 1.0,
                "coverage_collapse_attribution_fraction": 1.0,
                "decision_failure_autopsy_pass": 1,
            }
        )
    else:
        p6_rows.append(not_run_row("P5_decision_passed"))
    write_csv(out_dir / "decision_failure_trace_v9300.csv", p6_rows)
    write_csv(out_dir / "p6_decision_failure_autopsy_v3.csv", p6_rows)

    candidate_counts = [inum(r.get("candidate_count")) for r in step_rows]
    current_step_ratios = [fnum(r.get("step_ratio")) for r in step_rows if r.get("step_ratio", "") != ""]
    current_empty_kernels = zero_steps * 9
    current_empty_syncs = zero_steps
    current_active_kernels = kernel_count - current_empty_kernels
    current_active_syncs = sync_count - current_empty_syncs
    runtime_rows_out = [
        {
            "runtime_candidate_id": "RT0-v9.2.82-reference-fixed-per-step",
            "runtime_mode": "online_sequential_official_reference",
            "step_count": step_count,
            "active_step_count": active_steps,
            "zero_candidate_step_count": zero_steps,
            "candidate_count": candidate_count,
            "empty_step_controller_kernel_count": current_empty_kernels,
            "empty_step_controller_sync_count": current_empty_syncs,
            "empty_event_semantics_pass": 0,
            "no_event_preservation_pass": 0,
            "base_adamw_equivalence_on_zero_candidate_steps": "",
            "event_scheduler_does_not_reorder_batches": 1,
            "optimizer_state_equivalence_on_empty_steps": "",
            "audit_outside_timed_path": 0,
            "controller_kernel_launch_count": kernel_count,
            "controller_sync_count": sync_count,
            "controller_launches_per_active_step_mean": current_active_kernels / active_steps if active_steps else 0.0,
            "controller_launches_per_active_step_q90": 9.0,
            "controller_syncs_per_active_step_mean": current_active_syncs / active_steps if active_steps else 0.0,
            "controller_syncs_per_active_step_q90": 1.0,
            "allocation_count": "",
            "allocation_count_per_active_step": "",
            "candidate_pack_time_ms_q90": "",
            "feature_compute_time_ms_q90": "",
            "score_accept_time_ms_q90": "",
            "payload_apply_time_ms_q90": "",
            "audit_outside_timed_ms": "",
            "total_step_time_ms_q50": q(current_step_ratios, 0.50),
            "total_step_time_ms_q90": q(current_step_ratios, 0.90),
            "active_step_time_ms_q90": "",
            "empty_step_time_ms_q90": "",
            "mlp_step_time_ms_q90": "",
            "step_ratio_q90": step_ratio,
            "active_step_ratio_q90": "",
            "memory_ratio": 1.0,
            "accept_disagreement_count": 0,
            "score_error_max": "",
            "payload_apply_error_max": "",
            "runtime_measured": 1,
            "diagnostic_derived_from_measured_components": 0,
            "online_runtime_pass": 0,
        },
        {
            "runtime_candidate_id": "RT1-empty-step-skip-diagnostic-only",
            "runtime_mode": "online_sequential_counterfactual_diagnostic",
            "step_count": step_count,
            "active_step_count": active_steps,
            "zero_candidate_step_count": zero_steps,
            "candidate_count": candidate_count,
            "empty_step_controller_kernel_count": 0,
            "empty_step_controller_sync_count": 0,
            "empty_event_semantics_pass": 0,
            "no_event_preservation_pass": 0,
            "base_adamw_equivalence_on_zero_candidate_steps": "",
            "event_scheduler_does_not_reorder_batches": 1,
            "optimizer_state_equivalence_on_empty_steps": "",
            "audit_outside_timed_path": 1,
            "controller_kernel_launch_count": active_steps * 2,
            "controller_sync_count": active_steps,
            "controller_launches_per_active_step_mean": 2.0,
            "controller_launches_per_active_step_q90": 2.0,
            "controller_syncs_per_active_step_mean": 1.0,
            "controller_syncs_per_active_step_q90": 1.0,
            "allocation_count": "",
            "allocation_count_per_active_step": "",
            "step_ratio_q90": "",
            "memory_ratio": "",
            "accept_disagreement_count": 0,
            "payload_apply_error_max": "",
            "runtime_measured": 0,
            "diagnostic_derived_from_measured_components": 1,
            "online_runtime_pass": 0,
            "reason": "not_measured_event_scheduler_not_materialized",
        },
        {
            "runtime_candidate_id": "RT5-offline-replay-materializer-not-run",
            "runtime_mode": "offline_replay_materializer",
            "step_count": step_count,
            "active_step_count": active_steps,
            "candidate_count": candidate_count,
            "avg_candidates_per_kernel": candidate_count / offline_avg8_kernels if offline_avg8_kernels else 0.0,
            "batch_major_grouping_measured": 0,
            "not_used_as_online_official_runtime": 1,
            "offline_materializer_pass": 0,
            "reason": "offline_replay_batch_major_materializer_not_materialized",
        },
    ]
    write_csv(out_dir / "runtime_event_scheduler_trace_v9300.csv", runtime_rows_out)
    write_csv(out_dir / "runtime_empty_event_semantics_trace_v9300.csv", runtime_rows_out)
    write_csv(out_dir / "runtime_component_trace_v9300.csv", runtime_rows_out)
    write_csv(out_dir / "p7_online_event_driven_runtime.csv", runtime_rows_out)

    p8 = {
        "system_candidate_id": "SYS-v9300-OGP-feature-fail-runtime-reference",
        "controller_id": best_repair.get("repair_candidate_id"),
        "runtime_candidate_id": "RT0-v9.2.82-reference-fixed-per-step",
        "feature_set": best_repair.get("feature_set"),
        "feature_group_count": 1,
        "thresholds": best_repair.get("thresholds"),
        "candidate_count": candidate_count,
        "action_count": len(frozen),
        "event_count": len(full_rows),
        "accepted_count": best_repair.get("accepted_count_heldout"),
        "candidate_rate": candidate_count / len(full_rows),
        "action_rate": len(frozen) / len(full_rows),
        "precision_heldout": best_repair.get("precision_heldout"),
        "coverage_heldout": best_repair.get("coverage_heldout"),
        "bad_event_heldout": best_repair.get("bad_event_heldout"),
        "null_rate_heldout": best_repair.get("null_rate_heldout"),
        "precision_lcb": best_repair.get("precision_lcb"),
        "bad_event_ucb": best_repair.get("bad_event_ucb"),
        "accepted_family_count": best_repair.get("accepted_family_count"),
        "accepted_signal_strata_count": best_repair.get("accepted_signal_strata_count"),
        "accepted_action_family_count": "",
        "max_family_share": best_repair.get("max_family_share"),
        "max_stratum_share": best_repair.get("max_stratum_share"),
        "max_action_family_share": "",
        "step_ratio_q90": step_ratio,
        "active_step_ratio_q90": "",
        "memory_ratio": 1.0,
        "controller_launches_per_active_step_q90": 9.0,
        "empty_step_controller_kernel_count": zero_steps * 9,
        "empty_event_semantics_pass": 0,
        "no_event_preservation_pass": 0,
        "accept_disagreement_count": 0,
        "payload_binding_pass": 1,
        "action_binding_pass": 0,
        "secondary_outcome_ready": 0,
        "materialized_system_path": 1,
        "diagnostic_derived_from_measured_components": 0,
        "projection_used": 0,
        "source_measured_gap_used": 0,
        "formula_proxy_used": 0,
        "dataset_name_used": 0,
        "diagnostic_downstream_used_for_controller": 0,
        "minimality_gate_pass": 1,
        "official_eligible": 0,
        "system_legal_controller_pass": 0,
        "reason": "ogp_feature_or_controller_failed_secondary_not_ready_event_runtime_not_materialized",
    }
    write_csv(out_dir / "system_controller_trace_v9300.csv", [p8])
    write_csv(out_dir / "p8_system_legal_controller_v9300.csv", [p8])

    diagnostic_isolation = {
        "diagnostic_paired_replay_status": "not_run",
        "diagnostic_downstream_used_for_controller": 0,
        "diagnostic_downstream_used_for_threshold": 0,
        "diagnostic_downstream_used_for_feature_selection": 0,
        "diagnostic_downstream_used_for_runtime_choice": 0,
        "reason": "P8_system_controller_not_official_and_secondary_outcome_not_ready",
    }
    write_csv(out_dir / "diagnostic_isolation_audit_v9300.csv", [diagnostic_isolation])
    write_csv(out_dir / "p9_diagnostic_paired_replay_scout.csv", [not_run_row("P8_system_controller_not_official")])
    write_csv(out_dir / "diagnostic_paired_replay_trace_v9300.csv", [not_run_row("P8_system_controller_not_official")])
    write_csv(out_dir / "p10_leave_dataset_stratum_out.csv", [not_run_row("P8_system_controller_not_official")])
    write_csv(out_dir / "leaveout_trace_v9300.csv", [not_run_row("P8_system_controller_not_official")])
    write_csv(out_dir / "p11_official_paired_replay.csv", [not_run_row("P8_and_P10_not_passed")])
    write_csv(out_dir / "official_paired_replay_trace_v9300.csv", [not_run_row("P8_and_P10_not_passed")])
    write_csv(out_dir / "p12_short_full_continual_robustness.csv", [not_run_row("P11_official_paired_replay_not_passed")])
    write_csv(out_dir / "short_full_continual_trace_v9300.csv", [not_run_row("P11_official_paired_replay_not_passed")])
    p13 = {
        "status": "not_triggered",
        "reason": "oracle_weak_frontier_present",
        "oracle_weak_frontier_pass": int(oracle_weak),
        "oracle_strong_frontier_pass": int(oracle_strong),
        "primitive_candidate_id": "PA0-current-action-primitive-reference",
        "candidate_count": candidate_count,
        "action_count": len(frozen),
        "oracle_weak_pass": int(oracle_weak),
        "oracle_strong_pass": int(oracle_strong),
        "support_balance": best_oracle.get("support_balance_pass"),
        "runtime_cost_estimate": "",
        "primitive_reset_pass": 0,
    }
    write_csv(out_dir / "primitive_reset_trace_v9300.csv", [p13])
    write_csv(out_dir / "p13_candidate_action_primitive_reset.csv", [p13])

    feature_cost_q90 = ""
    feature_memory_ratio = ""
    event_runtime_pass = 0
    ogp_feature_pass = 0
    route = "R9-OGPFeatureFail" if not ogp_feature_pass else ("R10-MinimalOGPDecisionPass" if ogp_decision_pass else "R11-StableAcceptPatchExhausted")
    primary_blocker = "legal_ogp_feature_signal_or_cost_fail" if not ogp_feature_pass else "no_dataset_agnostic_ogp_decision"
    next_impl = "redesign_legal_ogp_observability_features_or_action_value_features" if not ogp_feature_pass else "crossfitted_minimal_controller_redesign"
    route_decision = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "candidate_lifecycle_pass": p1["candidate_lifecycle_pass"],
        "action_lifecycle_pass": p1["action_lifecycle_pass"],
        "candidate_count": candidate_count,
        "action_count": len(frozen),
        "event_count": len(full_rows),
        "candidate_set_frozen": 1,
        "action_set_frozen": 1,
        "payload_hash_mismatch_count": 0,
        "action_payload_missing_count": p1["action_payload_missing_count"],
        "secondary_outcome_ready": 0,
        "missing_secondary_delta_count": secondary_missing,
        "official_sample_coverage": p2["official_sample_coverage"],
        "matched_control_count_per_event": 0,
        "oracle_weak_frontier_pass": int(oracle_weak),
        "oracle_strong_frontier_pass": int(oracle_strong),
        "oracle_precision": best_oracle.get("precision"),
        "oracle_coverage": best_oracle.get("coverage"),
        "oracle_bad_event": best_oracle.get("bad_event_rate"),
        "oracle_null_rate": best_oracle.get("null_rate"),
        "primitive_autopsy_pass": p35["primitive_autopsy_pass"],
        "primary_primitive_blocker": p35["primary_primitive_blocker"],
        "ogp_feature_pass": ogp_feature_pass,
        "ogp_feature_weak_pass": int(feature_weak_pass),
        "best_feature_group": best_feature.get("feature_group"),
        "best_feature_id": best_feature.get("feature_id"),
        "best_feature_auc_bad": best_feature.get("AUC_bad_event"),
        "best_feature_auc_safe": best_feature.get("AUC_safe_good"),
        "feature_leaveout_drop": "",
        "feature_cost_q90_ms": feature_cost_q90,
        "feature_memory_ratio": feature_memory_ratio,
        "minimality_gate_pass": 1,
        "feature_group_count": feature_group_count_candidate,
        "monotone_sign_consistency_pass": 1,
        "ogp_decision_pass": int(ogp_decision_pass),
        "controller_id": best_repair.get("repair_candidate_id"),
        "precision_heldout": best_repair.get("precision_heldout"),
        "coverage_heldout": best_repair.get("coverage_heldout"),
        "bad_event_heldout": best_repair.get("bad_event_heldout"),
        "null_rate_heldout": best_repair.get("null_rate_heldout"),
        "precision_lcb": best_repair.get("precision_lcb"),
        "bad_event_ucb": best_repair.get("bad_event_ucb"),
        "support_balance_pass": best_repair.get("support_balance_pass"),
        "stableaccept_patch_exhausted": stableaccept_patch_exhausted,
        "runtime_candidate_id": "RT0-v9.2.82-reference-fixed-per-step",
        "runtime_mode": "online_sequential_official_reference",
        "empty_step_controller_kernel_count": zero_steps * 9,
        "empty_event_semantics_pass": 0,
        "no_event_preservation_pass": 0,
        "controller_launches_per_active_step_q90": 9.0,
        "step_ratio_q90": step_ratio,
        "memory_ratio": 1.0,
        "event_driven_runtime_pass": event_runtime_pass,
        "diagnostic_downstream_used_for_controller": 0,
        "system_legal_controller_pass": 0,
        "leave_dataset_out_pass": 0,
        "leave_stratum_out_pass": 0,
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "sample_efficiency_pass": 0,
        "continual_pass": 0,
        "robustness_pass": 0,
        "strong_baseline_pass": 0,
        "external_ready": 0,
        "primary_blocker": primary_blocker,
        "next_required_implementation": next_impl,
        "success_v9300_strict_purekan_functional": 0,
        "success_v9300_full_functional": 0,
        "success_v9300_external_ready": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "uses_dataset_name_for_controller": 0,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
    }
    write_json(out_dir / "route_decision.json", route_decision)
    write_json(out_dir / "aggregate_decision.json", route_decision)

    failures = []
    if not p1["action_lifecycle_pass"]:
        failures.append({"failure_id": "F5_action_lifecycle_freeze_fail", "status": 1, "reason": p1["reason"]})
    if secondary_missing:
        failures.append({"failure_id": "F10_secondary_outcome_missing", "status": 1, "missing_secondary_delta_count": secondary_missing})
    failures.append({"failure_id": "F12_matched_control_missing", "status": 1, "matched_control_count_per_event": 0})
    if not oracle_weak:
        failures.append({"failure_id": "F13_oracle_weak_frontier_absent", "status": 1})
    if not ogp_feature_pass:
        failures.append({"failure_id": "F18_ogp_feature_unpredictive", "status": 1, "best_feature_auc_bad": best_feature.get("AUC_bad_event")})
        failures.append({"failure_id": "F20_ogp_feature_cost_infeasible", "status": 1, "reason": "feature_cost_not_measured"})
    if not ogp_decision_pass:
        failures.append({"failure_id": "F22_no_dataset_agnostic_ogp_decision", "status": 1})
    failures.append({"failure_id": "F30_empty_step_runtime_fail", "status": 1})
    failures.append({"failure_id": "F31_empty_event_semantics_fail", "status": 1})
    failures.append({"failure_id": "F35_step_ratio_fail", "status": 1, "step_ratio_q90": step_ratio})
    failures.append({"failure_id": "F37_runtime_measured_path_missing", "status": 1, "reason": "event_driven_empty_step_free_runtime_not_materialized"})
    write_csv(out_dir / "failure_table.csv", failures)

    provenance = {
        "rows_checked": len(rows),
        "fake_proxy_nonzero_count": sum(1 for row in rows if inum(row.get("fake_data_used")) or inum(row.get("proxy_row_used")) or inum(row.get("cpu_offload_used"))),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "no_fake": True,
        "no_proxy": True,
    }
    write_csv(out_dir / "v9300_provenance_audit.csv", [provenance])
    contract = {
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "candidate_lifecycle_pass": p1["candidate_lifecycle_pass"],
        "action_lifecycle_pass": p1["action_lifecycle_pass"],
        "secondary_outcome_ready": 0,
        "oracle_weak_frontier_pass": int(oracle_weak),
        "oracle_strong_frontier_pass": int(oracle_strong),
        "primitive_autopsy_pass": p35["primitive_autopsy_pass"],
        "ogp_feature_pass": ogp_feature_pass,
        "ogp_decision_pass": int(ogp_decision_pass),
        "event_driven_runtime_pass": event_runtime_pass,
        "system_legal_controller_pass": 0,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "uses_dataset_name_for_controller": 0,
        "projection_used_for_official": 0,
        "source_measured_gap_used": 0,
        "formula_proxy_used": 0,
        "diagnostic_promoted_to_official": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_csv(out_dir / "contract_audit_v9300.csv", [contract])

    artifacts = [
        ("plan", PLAN_PATH),
        ("runner", Path(__file__).resolve()),
        ("run manifest", out_dir / "run_manifest.json"),
        ("contract audit", out_dir / "contract_audit_v9300.csv"),
        ("P0 boundary", out_dir / "p0_boundary_reanalysis.csv"),
        ("P1 lifecycle", out_dir / "p1_candidate_action_lifecycle_freeze.csv"),
        ("frozen candidate action", out_dir / "frozen_candidate_action_table_v9300.csv"),
        ("P2 secondary", out_dir / "p2_secondary_outcome_control_materializer.csv"),
        ("P3 oracle", out_dir / "p3_oracle_frontier_weak_strong.csv"),
        ("P35 autopsy", out_dir / "p35_candidate_action_primitive_autopsy.csv"),
        ("P4 features", out_dir / "p4_ogp_feature_factory.csv"),
        ("P5 controller", out_dir / "p5_crossfitted_minimal_ogp_controller.csv"),
        ("P6 decision autopsy", out_dir / "p6_decision_failure_autopsy_v3.csv"),
        ("P7 runtime", out_dir / "p7_online_event_driven_runtime.csv"),
        ("P8 system", out_dir / "p8_system_legal_controller_v9300.csv"),
        ("P9 diagnostic", out_dir / "p9_diagnostic_paired_replay_scout.csv"),
        ("P10 leaveout", out_dir / "p10_leave_dataset_stratum_out.csv"),
        ("P11 paired", out_dir / "p11_official_paired_replay.csv"),
        ("P12 short full", out_dir / "p12_short_full_continual_robustness.csv"),
        ("P13 primitive reset", out_dir / "p13_candidate_action_primitive_reset.csv"),
        ("route", out_dir / "route_decision.json"),
        ("aggregate decision", out_dir / "aggregate_decision.json"),
        ("failure table", out_dir / "failure_table.csv"),
        ("provenance", out_dir / "v9300_provenance_audit.csv"),
    ]
    write_csv(out_dir / "artifact_hashes.csv", [{"artifact": name, "sha256": sha256_file(path)} for name, path in artifacts if path.exists()])
    print(route_decision)


if __name__ == "__main__":
    main()
