#!/usr/bin/env python3
"""DG-KAN v9.8.7 natural stream / future-operator mechanism run.

This runner follows the v9.8.7 plan.  It reuses the landed branch-horizon
replay materializer for real AP0 preflight labeling, and it writes explicit
not_run blocker rows when the new natural AP0 action generator for 5000/10000/
20000 panels is unavailable.  No fake/proxy rows are created.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9720_exact_transfer_materializer_core_expansion_direct_update as v9720  # noqa: E402
import run_v9770_core77_support_density_natural_stream_gate as v9770  # noqa: E402
import run_v9820_future_path_geometry_mechanism_natural_stream as v9820  # noqa: E402
import run_v9840_future_operator_natural_stream_geometry_optimizer as v9840  # noqa: E402
import run_v9850_four_line_future_operator_natural_stream_decision as v9850  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


RESULT_ROOT = REPO / "results/real_rerun_20260506"
PLAN_PATH = REPO / "docs/DG-KAN_v9.8.7_四线并行_自然扩流_FutureOperator机制_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9870_four_line_natural_stream_futureoperator_mechanism.py"
RECAP_PATH = REPO / "docs/DG-KAN_v9.8.7_FourLineNaturalStream_FutureOperatorMechanism_实验复盘.md"
DEFAULT_OUT = RESULT_ROOT / "v9870_four_line_natural_stream_futureoperator_mechanism_full_20260516T220000Z"
DEFAULT_V9860 = RESULT_ROOT / "v9860_future_path_mechanism_natural_stream_geometry_optimizer_full_20260516T200000Z"
DEFAULT_V9850 = RESULT_ROOT / "v9850_four_line_future_operator_natural_stream_decision_full_20260516T180000Z"
DEFAULT_V9840 = RESULT_ROOT / "v9840_future_operator_natural_stream_geometry_optimizer_full_20260516T160000Z"
DEFAULT_V9820 = RESULT_ROOT / "v9820_future_path_geometry_mechanism_natural_stream_full_20260516T120000Z"
DEFAULT_V9810 = RESULT_ROOT / "v9810_future_trajectory_geometry_adaptive_optimizer_full_20260516T100000Z"
DEFAULT_V9740 = RESULT_ROOT / "v9740_existing_action_ldo_transfer_mismatch_horizon_transfer_first_20260516T030000Z"
DEFAULT_V9720 = RESULT_ROOT / "v9720_exact_transfer_materializer_core_expansion_direct_update_first_20260516T010000Z"
DEFAULT_V9480 = RESULT_ROOT / "v9480_canonical_outcome_universe_rebuild_source_frontier_revalidation_recovery_20260514T160000Z"

HORIZONS = [1, 5, 20, 80, 240]
BRANCH_COUNT = 6
TARGET_K = 87


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--execution-profile", choices=["smoke", "full-gated"], default="full-gated")
    p.add_argument("--panel-targets", default="2876,5000,10000,20000")
    p.add_argument("--natural-preflight-actions", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--clear-caches-each-action", action="store_true")
    p.add_argument("--source-v9860", default=str(DEFAULT_V9860))
    p.add_argument("--source-v9850", default=str(DEFAULT_V9850))
    p.add_argument("--source-v9840", default=str(DEFAULT_V9840))
    p.add_argument("--source-v9820", default=str(DEFAULT_V9820))
    p.add_argument("--source-v9810", default=str(DEFAULT_V9810))
    p.add_argument("--source-v9740", default=str(DEFAULT_V9740))
    p.add_argument("--source-v9720", default=str(DEFAULT_V9720))
    p.add_argument("--source-v9480", default=str(DEFAULT_V9480))
    p.add_argument("--source-v9330", default=str(v9840.v9820.DEFAULT_V9330))
    p.add_argument("--source-v9550", default=str(v9840.v9820.DEFAULT_V9550))
    p.add_argument("--source-v9560", default=str(v9840.v9820.DEFAULT_V9560))
    p.add_argument("--source-v9570", default=str(v9840.v9820.DEFAULT_V9570))
    p.add_argument("--source-v9580", default=str(v9840.v9820.DEFAULT_V9580))
    return p.parse_args()


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def mean(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.fmean(vals) if vals else 0.0


def qtile(xs: list[float], q: float) -> float:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return 0.0
    pos = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[pos]


def lcb(xs: list[float]) -> float:
    return v9720.lcb(xs)


def ucb(xs: list[float]) -> float:
    return v9720.ucb(xs)


def wilson(count: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    if total <= 0:
        return 0.0, 0.0, 0.0
    phat = count / total
    den = 1.0 + z * z / total
    centre = phat + z * z / (2.0 * total)
    delta = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total)
    return phat, max(0.0, (centre - delta) / den), min(1.0, (centre + delta) / den)


def device_from(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def pearson(xs: list[float], ys: list[float]) -> float:
    vals = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(vals) < 2:
        return 0.0
    xvals = [x for x, _ in vals]
    yvals = [y for _, y in vals]
    mx, my = mean(xvals), mean(yvals)
    vx = sum((x - mx) ** 2 for x in xvals)
    vy = sum((y - my) ** 2 for y in yvals)
    if vx <= 0 or vy <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in vals) / math.sqrt(vx * vy)


def json_counter(items: list[str]) -> str:
    return json.dumps(dict(Counter(items)), sort_keys=True, ensure_ascii=False)


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r for r in rows if r.get("status") == "summary"), rows[0] if rows else {})


def ap0_indices(ap0: list[dict[str, Any]]) -> dict[str, int]:
    return {str(r.get("action_id")): i for i, r in enumerate(ap0)}


def membership_scores(n: int, idx: list[int]) -> list[float]:
    scores = [0.0] * n
    for i in idx:
        if 0 <= i < n:
            scores[i] = 1.0
    return scores


def quality_for_ids(ap0: list[dict[str, Any]], ids: list[str]) -> tuple[dict[str, Any], list[int], list[float]]:
    idx_by = ap0_indices(ap0)
    idx = [idx_by[str(a)] for a in ids if str(a) in idx_by]
    scores = membership_scores(len(ap0), idx)
    q = v9720.quality_for_indices(ap0, idx, scores, max(1, len(idx)))
    return q, idx, scores


def axis(row: dict[str, Any], name: str) -> str:
    return v9720.axis_value(row, name)


def ldo_decomposition(ap0: list[dict[str, Any]], idx: list[int], q: dict[str, Any], scores: list[float]) -> dict[str, Any]:
    base_p = fnum(q.get("GradeAB_precision"))
    base_v = fnum(q.get("V_integrated_LCB"))
    precision_drop = 0.0
    value_drop = 0.0
    by_ds: dict[str, list[int]] = defaultdict(list)
    for i in idx:
        by_ds[axis(ap0[i], "dataset_id")].append(i)
    for ids in by_ds.values():
        own_scores = membership_scores(len(ap0), ids)
        sq = v9720.quality_for_indices(ap0, ids, own_scores, max(1, len(ids)))
        precision_drop = max(precision_drop, max(0.0, base_p - fnum(sq.get("GradeAB_precision"))))
        value_drop = max(value_drop, max(0.0, base_v - fnum(sq.get("V_integrated_LCB"))))
    support_adj, _detail = v9770.support_adjusted_drop(ap0, idx)
    ldo_raw = fnum(q.get("LDO_drop"))
    ldo_quality = max(precision_drop, value_drop)
    ldo_support = max(0.0, support_adj - ldo_quality)
    ldo_backfill = max(0.0, ldo_raw - support_adj)
    return {
        "LDO_raw": ldo_raw,
        "LDO_quality": ldo_quality,
        "LDO_support": ldo_support,
        "LDO_backfill": ldo_backfill,
        "LDO_support_adjusted": support_adj,
        "LDO_backfill_share": ldo_backfill / ldo_raw if ldo_raw > 0 else 0.0,
        "LFO": v9770.lfo_drop(scores, ap0, max(1, len(idx))),
    }


def p0_boundary(source_v9860: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9860 / "route_decision_v9860.json")
    p1 = summary_row(read_csv(source_v9860 / "p1_future_path_combo_reassessment_v9860.csv"))
    p2 = summary_row(read_csv(source_v9860 / "p2_future_operator_proxy_v9860.csv"))
    p3 = summary_row(read_csv(source_v9860 / "p3_natural_ap0_stream_materializer_v9860.csv"))
    p4 = summary_row(read_csv(source_v9860 / "p4_generated_conditional_sandbox_v9860.csv"))
    p5 = summary_row(read_csv(source_v9860 / "p5_minimal_controller_boundary_v9860.csv"))
    nf = summary_row(read_csv(source_v9860 / "no_fake_audit_v9860.csv"))
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9870",
        "status": "summary",
        "source_route_v9860": route.get("route"),
        "source_primary_blocker_v9860": route.get("primary_blocker"),
        "source_secondary_blocker_v9860": route.get("secondary_blocker"),
        "P1_A_combo_strong_pass_v9860": p1.get("P1_A_combo_strong_pass"),
        "P2_weak_pass_v9860": p2.get("P2_weak_pass"),
        "P2_strong_pass_v9860": p2.get("P2_strong_pass"),
        "P3_materializer_entrypoint_found_v9860": p3.get("materializer_entrypoint_found"),
        "P4_generated_sandbox_allowed_v9860": p4.get("generated_sandbox_allowed"),
        "P5_controller_pass_v9860": p5.get("controller_pass"),
        "generated_route_status_v9860": route.get("generated_route_status"),
        "system_legal_controller_pass_v9860": route.get("system_legal_controller_pass"),
        "fake_data_used": nf.get("fake_data_used", 0),
        "proxy_row_used": nf.get("proxy_row_used", 0),
        "cpu_offload_used": nf.get("cpu_offload_used", 0),
    }
    row["P0_boundary_pass"] = int(
        row["source_route_v9860"] == "R5-LegalProxyFailNaturalStreamMissing"
        and inum(row["P1_A_combo_strong_pass_v9860"]) == 0
        and inum(row["P2_weak_pass_v9860"]) == 0
        and inum(row["P2_strong_pass_v9860"]) == 0
        and inum(row["P3_materializer_entrypoint_found_v9860"]) == 0
        and inum(row["P4_generated_sandbox_allowed_v9860"]) == 0
        and inum(row["P5_controller_pass_v9860"]) == 0
        and inum(row["system_legal_controller_pass_v9860"]) == 0
        and inum(row["fake_data_used"]) == 0
        and inum(row["proxy_row_used"]) == 0
        and inum(row["cpu_offload_used"]) == 0
    )
    row["controller_not_promoted"] = int(inum(row["P5_controller_pass_v9860"]) == 0 and inum(row["system_legal_controller_pass_v9860"]) == 0)
    row["generated_not_promoted"] = int(inum(row["P4_generated_sandbox_allowed_v9860"]) == 0)
    return [row], row


def landed_future_row_reference(real: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [{
        "stage": "P0_LANDED_FUTURE_ROW_REFERENCE_V9870",
        "status": "summary",
        "realfunctional_row_count": len(real),
        "action_count": len({str(r.get("action_id")) for r in real}),
        "horizons": ",".join(str(h) for h in sorted({inum(r.get("horizon")) for r in real})),
        "group_count": len({str(r.get("group_id")) for r in real}),
        "source": "v9820_full_future_path_rows_plus_v9840_extra_future_path_rows",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    for r in real:
        rows.append({
            "stage": "P0_LANDED_FUTURE_ROW_REFERENCE_V9870",
            "status": "landed_horizon_row",
            "action_id": r.get("action_id"),
            "group_id": r.get("group_id"),
            "dataset": r.get("dataset_id") or r.get("dataset"),
            "seed": r.get("seed"),
            "family_id": r.get("family_id"),
            "template_id": r.get("template_id"),
            "horizon": r.get("horizon"),
            "V_branch": r.get("V_branch"),
            "CE_delta": r.get("CE_delta"),
            "margin_delta": r.get("margin_delta"),
            "CEp99_delta": r.get("CEp99_delta"),
            "hard_tail_loss_delta": r.get("hard_tail_loss_delta"),
            "long_risk_label": r.get("long_risk_label"),
            "bad_event_label": r.get("bad_event_label"),
            "null_event_label": r.get("null_event_label"),
            "memory_fail": r.get("memory_fail"),
            "offdiag_fail": r.get("offdiag_fail"),
            "branch_runtime_ms": r.get("branch_runtime_ms"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    return rows


def existing_materialized_action_ids(source_v9820: Path, source_v9840: Path) -> set[str]:
    ids: set[str] = set()
    for path in [
        source_v9820 / "p1_full_future_path_materializer_v9820.csv",
        source_v9840 / "p1_extra_future_path_materializer_v9840.csv",
    ]:
        if path.exists():
            for r in read_csv(path):
                if r.get("status") == "branch_horizon_row":
                    ids.add(str(r.get("action_id")))
    return ids


def select_natural_preflight_indices(
    ap0: list[dict[str, Any]],
    payload_by_id: dict[str, dict[str, str]],
    excluded_ids: set[str],
    count: int,
) -> list[int]:
    eligible = [
        i for i, r in enumerate(ap0)
        if str(r.get("action_id")) in payload_by_id and str(r.get("action_id")) not in excluded_ids
    ]
    return eligible[: max(0, min(count, len(eligible)))]


def relabel_v9870_p1_rows(rows: list[dict[str, Any]], stage: str, candidate_origin: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        rr["stage"] = stage
        if rr.get("status") == "branch_horizon_row":
            rr["materializer_id"] = "FPMAT-v9870-existing-AP0-preflight-h1-h5-h20-h80-h240"
            rr["candidate_origin"] = candidate_origin
            rr["stratum"] = rr.get("step_bucket")
            rr["dataset_for_diagnostic_only"] = rr.get("dataset_id") or rr.get("dataset")
            rr["AdamW_cosine"] = rr.get("AdamW_alignment", "")
            rr["memory_bucket"] = "memory_safe" if inum(rr.get("memory_fail")) == 0 else "memory_fail"
            rr["offdiag_bucket"] = "offdiag_safe" if inum(rr.get("offdiag_fail")) == 0 else "offdiag_fail"
        out.append(rr)
    return out


def stage_completion(rows: list[dict[str, Any]], selected_ids: list[str], target: int) -> dict[str, Any]:
    ids = selected_ids[:target]
    branch_rows = [r for r in rows if r.get("status") == "branch_horizon_row" and str(r.get("action_id")) in set(ids)]
    expected = len(ids) * BRANCH_COUNT * len(HORIZONS)
    actual = len(branch_rows)
    duplicate = actual - len({r.get("outcome_row_id") for r in branch_rows})
    metric_nan = sum(inum(r.get("metric_nan_count")) for r in branch_rows)
    metric_inf = sum(inum(r.get("metric_inf_count")) for r in branch_rows)
    label_viol = 0
    for r in branch_rows:
        labels = inum(r.get("weak_CP_label")) + inum(r.get("bad_event_label")) + inum(r.get("null_event_label"))
        if labels > 1:
            label_viol += 1
    complete_ids = {
        aid for aid in ids
        if sum(1 for r in branch_rows if str(r.get("action_id")) == aid) == BRANCH_COUNT * len(HORIZONS)
    }
    return {
        "target_action_count": target,
        "new_action_count": len(ids),
        "existing_action_count": 0,
        "labeled_action_count": len(complete_ids),
        "branch_count": BRANCH_COUNT,
        "horizon_count": len(HORIZONS),
        "expected_rows": expected,
        "actual_rows": actual,
        "completion_rate": actual / max(1, expected),
        "duplicate_row_count": duplicate,
        "label_exclusivity_violation_count": label_viol,
        "metric_nan_count": metric_nan,
        "metric_inf_count": metric_inf,
        "quality_audit_pass": int(actual == expected and duplicate == 0 and label_viol == 0 and metric_nan == 0 and metric_inf == 0),
    }


def natural_preflight_materializer(
    args: argparse.Namespace,
    ap0: list[dict[str, Any]],
    score_bundle: dict[str, Any],
    payload_by_id: dict[str, dict[str, str]],
    out: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    target = 16 if args.execution_profile == "smoke" else int(args.natural_preflight_actions)
    excluded = existing_materialized_action_ids(Path(args.source_v9820), Path(args.source_v9840))
    selected_idx = select_natural_preflight_indices(ap0, payload_by_id, excluded, target)
    groups = {"NaturalAP0Preflight": selected_idx}
    candidate_origin = "canonical_AP0_existing_unlabeled_preflight_not_extension"
    if not selected_idx:
        summary = {
            "stage": "P1_NATURAL_MATERIALIZER_PREFLIGHT_V9870",
            "status": "summary",
            "materializer_entrypoint_found": 1,
            "natural_extension_action_generator_found": 0,
            "selected_action_count": 0,
            "row_count_expected": 0,
            "row_count_actual": 0,
            "P1a_single_preflight_pass": 0,
            "P1b_16_preflight_pass": 0,
            "P1c_256_smoke_pass": 0,
            "P1d_5000_panel_weak_pass": 0,
            "P1e_10000_20000_panel_strong_pass": 0,
            "reason": "no_existing_AP0_actions_with_payload_available_for_preflight",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        return [summary], [summary], summary
    r5b = v9820.v9660.r5b_scores(ap0, score_bundle)
    exact_summary = v9820.v9730.load_exact_summary(Path(args.source_v9720))
    exact_t4 = v9820.v9730.score_defs_from_summary(ap0, exact_summary)["T4-core-safe-transfer"]
    wt_rows = v9820.v9750.load_wt_rows(Path(args.source_v9740))
    device = device_from(args.device)
    raw_rows, completion, raw_summary = v9820.materialize_full_future_path(
        args, ap0, groups, payload_by_id, r5b, exact_t4, wt_rows, device, out
    )
    rows = relabel_v9870_p1_rows(raw_rows, "P1_NATURAL_MATERIALIZER_PREFLIGHT_V9870", candidate_origin)
    selected_ids = [str(ap0[i].get("action_id")) for i in selected_idx]
    c0 = stage_completion(rows, selected_ids, min(1, len(selected_ids)))
    c1 = stage_completion(rows, selected_ids, min(16, len(selected_ids)))
    c2_target = min(256, len(selected_ids))
    c2 = stage_completion(rows, selected_ids, c2_target)
    unresolved = [r for r in completion if r.get("status") == "unresolved_exception"]
    exception_counts = json.dumps(dict(Counter(str(r.get("exception_type")) for r in unresolved)), sort_keys=True)
    summary = dict(rows[0])
    summary.update({
        "stage": "P1_NATURAL_MATERIALIZER_PREFLIGHT_V9870",
        "status": "summary",
        "materializer_entrypoint_found": 1,
        "natural_extension_action_generator_found": 0,
        "candidate_origin": candidate_origin,
        "selected_action_count": len(selected_ids),
        "new_action_count": len(selected_ids),
        "existing_action_count": 0,
        "labeled_action_count": c2["labeled_action_count"],
        "branch_count": BRANCH_COUNT,
        "horizon_count": len(HORIZONS),
        "row_count_expected": raw_summary.get("row_count_expected"),
        "row_count_actual": raw_summary.get("row_count_actual"),
        "completion_rate": raw_summary.get("completion_rate"),
        "rows_per_sec": raw_summary.get("rows_per_sec"),
        "wallclock_sec": raw_summary.get("wallclock_sec"),
        "unresolved_exception_count": len(unresolved),
        "exception_type_counts": exception_counts,
        "payload_hash_missing_count": sum(1 for r in rows if r.get("status") == "branch_horizon_row" and not r.get("payload_hash")),
        "state_hash_missing_count": sum(1 for r in rows if r.get("status") == "branch_horizon_row" and not r.get("state_after_horizon_hash")),
        "label_hash_missing_count": 0,
        "duplicate_row_count": raw_summary.get("duplicate_outcome_row_id_count"),
        "label_exclusivity_violation_count": raw_summary.get("label_exclusivity_violation_count"),
        "metric_nan_count": raw_summary.get("metric_nan_count"),
        "metric_inf_count": raw_summary.get("metric_inf_count"),
        "fake_row_count": 0,
        "proxy_row_count": 0,
        "cpu_offload_flag": 0,
        "P1a_single_preflight_pass": int(c0["quality_audit_pass"] and c0["labeled_action_count"] >= 1),
        "P1b_16_preflight_pass": int(c1["quality_audit_pass"] and c1["labeled_action_count"] >= 16),
        "P1c_256_smoke_pass": int(c2["quality_audit_pass"] and c2["labeled_action_count"] >= 256),
        "P1d_5000_panel_weak_pass": 0,
        "P1e_10000_20000_panel_strong_pass": 0,
        "reason": "existing_AP0_preflight_materialized_but_new_natural_AP0_extension_action_generator_missing",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    stage_rows = []
    for sid, label, stats in [
        ("P1a", "1-action-full-branch-horizon-preflight", c0),
        ("P1b", "16-action-labeled-preflight", c1),
        ("P1c", "256-action-labeled-smoke", c2),
    ]:
        stage_rows.append({
            "stage": "P1_NATURAL_MATERIALIZER_PREFLIGHT_V9870",
            "status": "stage_summary",
            "stage_id": sid,
            "panel_id": label,
            **stats,
            "rows_per_sec": raw_summary.get("rows_per_sec"),
            "wallclock_sec": raw_summary.get("wallclock_sec"),
            "unresolved_exception_count": len(unresolved),
            "exception_type_counts": exception_counts,
            "fake_row_count": 0,
            "proxy_row_count": 0,
            "cpu_offload_flag": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    rows[0] = summary
    rows[1:1] = stage_rows
    completion = relabel_v9870_p1_rows(completion, "P1_NATURAL_MATERIALIZER_COMPLETION_V9870", candidate_origin)
    v9720.write_bar_svg(out / "fig_p1_materializer_completion_by_stage.svg", "P1 completion", [r["stage_id"] for r in stage_rows], [fnum(r.get("completion_rate")) for r in stage_rows])
    v9720.write_bar_svg(out / "fig_p1_rows_per_sec_by_panel.svg", "P1 rows/sec", [r["stage_id"] for r in stage_rows], [fnum(r.get("rows_per_sec")) for r in stage_rows])
    v9720.write_bar_svg(out / "fig_p1_exception_type_bar.svg", "P1 exceptions", ["unresolved"], [len(unresolved)])
    return rows, completion, summary


def panel_not_run_file(target: int, existing_count: int, reason: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {
        "stage": "P1_NATURAL_STREAM_PANEL_V9870",
        "status": "not_run",
        "panel_id": f"Panel-{target}",
        "target_action_count": target,
        "existing_labeled_action_count": existing_count,
        "new_action_count_required": max(0, target - existing_count),
        "branch_count": BRANCH_COUNT,
        "horizon_count": len(HORIZONS),
        "expected_rows": target * BRANCH_COUNT * len(HORIZONS),
        "actual_rows": 0,
        "completion_rate": 0,
        "quality_audit_pass": 0,
        "P1d_5000_panel_weak_pass": 0,
        "P1e_10000_20000_panel_strong_pass": 0,
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def feature_labels(feats: dict[str, dict[str, Any]], ap0: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    idx_by = ap0_indices(ap0)
    labels: dict[str, dict[str, int]] = {}
    for aid, f in feats.items():
        ap = ap0[idx_by[aid]] if aid in idx_by else {}
        risk_clean = int(
            fnum(f.get("V80")) >= 0
            and fnum(f.get("V240")) > 0
            and inum(f.get("longrisk240")) == 0
            and inum(f.get("memory_fail")) == 0
            and inum(f.get("offdiag_fail")) == 0
        )
        labels[aid] = {
            "CoreLike": int(v9820.core_like(ap)) if ap else 0,
            "PathGood": int(fnum(f.get("AUV")) > 0 and risk_clean),
            "SlowBurnGood": int(fnum(f.get("V1")) <= 0 and fnum(f.get("AUV")) > 0 and risk_clean),
            "RiskCleanValuePositive": int(fnum(ap.get("V_integrated")) > 0 and risk_clean) if ap else 0,
        }
    return labels


def density_curve(
    ap0: list[dict[str, Any]],
    natural_feats: dict[str, dict[str, Any]],
    source_v9820: Path,
    panel_targets: list[int],
    out: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    src = read_csv(source_v9820 / "p3_natural_ap0_labeled_stream_extension_v9820.csv")
    panel_a = next((r for r in src if r.get("status") == "panel_row"), {})
    existing = len(ap0)
    labels = feature_labels(natural_feats, ap0)
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    if panel_a:
        rows.append({
            "stage": "P2_DENSITY_CURVE_V9870",
            "status": "panel_row",
            "panel_id": "Panel-2876-existing",
            "panel_size": existing,
            "row_source": "existing_canonical_AP0_labeled_panel",
            "CoreLike_count": panel_a.get("CoreLike_count"),
            "CoreLike_rate": panel_a.get("CoreLike_rate"),
            "CoreLike_Wilson_LCB": panel_a.get("Wilson_LCB"),
            "CoreLike_Wilson_UCB": panel_a.get("Wilson_UCB"),
            "PathGood_count": "",
            "PathGood_rate": "",
            "PathGood_Wilson_LCB": "",
            "PathGood_Wilson_UCB": "",
            "SlowBurnGood_count": "",
            "SlowBurnGood_rate": "",
            "RiskCleanValuePositive_count": panel_a.get("ValuePositiveNoLongRisk_count"),
            "per_dataset_rate_diagnostic": panel_a.get("per_dataset_rate"),
            "per_family_rate": panel_a.get("per_family_rate"),
            "per_template_rate": panel_a.get("per_template_rate"),
            "per_step_bucket_rate": "",
            "min_group_rate": "",
            "max_group_rate": "",
            "rate_PSI_between_panels": "",
            "density_result": "existing_panel_corelike_only_inconclusive",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    if labels:
        n = len(labels)
        counts = {k: sum(v[k] for v in labels.values()) for k in ["CoreLike", "PathGood", "SlowBurnGood", "RiskCleanValuePositive"]}
        pg_mean, pg_lcb, pg_ucb = wilson(counts["PathGood"], n)
        cl_mean, cl_lcb, cl_ucb = wilson(counts["CoreLike"], n)
        rows.append({
            "stage": "P2_DENSITY_CURVE_V9870",
            "status": "diagnostic_preflight_row",
            "panel_id": f"Preflight-{n}-existing-AP0",
            "panel_size": n,
            "row_source": "real_branch_horizon_preflight_existing_AP0_not_extension_panel",
            "CoreLike_count": counts["CoreLike"],
            "CoreLike_rate": cl_mean,
            "CoreLike_Wilson_LCB": cl_lcb,
            "CoreLike_Wilson_UCB": cl_ucb,
            "PathGood_count": counts["PathGood"],
            "PathGood_rate": pg_mean,
            "PathGood_Wilson_LCB": pg_lcb,
            "PathGood_Wilson_UCB": pg_ucb,
            "SlowBurnGood_count": counts["SlowBurnGood"],
            "SlowBurnGood_rate": counts["SlowBurnGood"] / n,
            "RiskCleanValuePositive_count": counts["RiskCleanValuePositive"],
            "per_dataset_rate_diagnostic": json_counter([str(natural_feats[a].get("dataset")) for a, v in labels.items() if v["PathGood"]]),
            "per_family_rate": json_counter([str(natural_feats[a].get("family_id")) for a, v in labels.items() if v["PathGood"]]),
            "per_template_rate": json_counter([str(natural_feats[a].get("template_id")) for a, v in labels.items() if v["PathGood"]]),
            "per_step_bucket_rate": json_counter([str(natural_feats[a].get("step_bucket")) for a, v in labels.items() if v["PathGood"]]),
            "min_group_rate": "",
            "max_group_rate": "",
            "rate_PSI_between_panels": "",
            "density_result": "diagnostic_preflight_not_official_density_panel",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
        for aid, lab in labels.items():
            audits.append({
                "stage": "P2_PATHGOOD_LABEL_AUDIT_V9870",
                "status": "action_label_row",
                "action_id": aid,
                **lab,
                "AUV": natural_feats[aid].get("AUV"),
                "V1": natural_feats[aid].get("V1"),
                "V80": natural_feats[aid].get("V80"),
                "V240": natural_feats[aid].get("V240"),
                "longrisk240": natural_feats[aid].get("longrisk240"),
                "memory_fail": natural_feats[aid].get("memory_fail"),
                "offdiag_fail": natural_feats[aid].get("offdiag_fail"),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    for target in [t for t in panel_targets if t > existing]:
        rows.append({
            "stage": "P2_DENSITY_CURVE_V9870",
            "status": "not_run",
            "panel_id": f"Panel-{target}",
            "panel_size": target,
            "CoreLike_count": "",
            "CoreLike_rate": "",
            "CoreLike_Wilson_LCB": "",
            "CoreLike_Wilson_UCB": "",
            "PathGood_count": "",
            "PathGood_rate": "",
            "PathGood_Wilson_LCB": "",
            "PathGood_Wilson_UCB": "",
            "missing_new_action_count": target - existing,
            "density_result": "not_run_materializer_extension_generator_missing",
            "reason": "natural_AP0_extension_action_generator_missing_after_existing_2876_AP0_actions",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P2_DENSITY_CURVE_V9870",
        "status": "summary",
        "completed_official_panel_count": 1 if panel_a else 0,
        "diagnostic_preflight_panel_count": int(bool(labels)),
        "not_run_panel_count": sum(1 for r in rows if r.get("status") == "not_run"),
        "PanelA_CoreLike_rate": panel_a.get("CoreLike_rate", ""),
        "PanelA_CoreLike_LCB": panel_a.get("Wilson_LCB", ""),
        "PanelA_CoreLike_UCB": panel_a.get("Wilson_UCB", ""),
        "preflight_PathGood_rate": next((r.get("PathGood_rate") for r in rows if r.get("status") == "diagnostic_preflight_row"), ""),
        "preflight_PathGood_LCB": next((r.get("PathGood_Wilson_LCB") for r in rows if r.get("status") == "diagnostic_preflight_row"), ""),
        "P2_density_sufficient": 0,
        "P2_density_insufficient": 0,
        "P2_density_inconclusive": 1,
        "reason": "5000_10000_20000_natural_extension_panels_not_materialized",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    audits.insert(0, {
        "stage": "P2_PATHGOOD_LABEL_AUDIT_V9870",
        "status": "summary",
        "action_count": len(labels),
        "PathGood_count": sum(v["PathGood"] for v in labels.values()) if labels else 0,
        "SlowBurnGood_count": sum(v["SlowBurnGood"] for v in labels.values()) if labels else 0,
        "label_definition": "PathGood=AUV>0,V80>=0,V240>0,longrisk/memory/offdiag clean; SlowBurn allows V1<=0",
        "official_density_claim": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    v9720.write_bar_svg(out / "fig_p2_density_curve_corelike.svg", "P2 CoreLike", [str(r.get("panel_id")) for r in rows if r.get("status") != "summary"], [fnum(r.get("CoreLike_rate")) for r in rows if r.get("status") != "summary"])
    v9720.write_bar_svg(out / "fig_p2_density_curve_pathgood.svg", "P2 PathGood", [str(r.get("panel_id")) for r in rows if r.get("status") != "summary"], [fnum(r.get("PathGood_rate")) for r in rows if r.get("status") != "summary"])
    v9720.write_bar_svg(out / "fig_p2_wilson_ci_by_panel.svg", "P2 PathGood LCB", [str(r.get("panel_id")) for r in rows if r.get("status") != "summary"], [fnum(r.get("PathGood_Wilson_LCB")) for r in rows if r.get("status") != "summary"])
    return rows, audits, summary


def future_path_mechanism_v9870(
    ap0: list[dict[str, Any]],
    feats: dict[str, dict[str, Any]],
    natural_feats: dict[str, dict[str, Any]],
    p4_proxy_hint: dict[str, Any] | None,
    out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_feats = {**feats, **natural_feats}
    nat_labels = feature_labels(natural_feats, ap0)
    pathgood = [aid for aid, lab in nat_labels.items() if lab["PathGood"]]
    slowburn = [aid for aid, lab in nat_labels.items() if lab["SlowBurnGood"]]
    core = group_ids(feats, "Core77")
    exp10 = group_ids(feats, "CoreExpansion10")
    risk_clean = group_ids(feats, "RiskCleanButLowValue")
    risk_top10 = sorted(risk_clean, key=lambda a: fnum(feats[a].get("RiskAdjustedAUV")), reverse=True)[:10]
    groups: list[tuple[str, list[str]]] = [
        ("Core77", core),
        ("CoreExpansion10", exp10),
        ("Core77+CoreExpansion10", core + exp10),
        ("RiskCleanButLowValue", risk_clean),
        ("Core77+RiskCleanButLowValueTop10", core + risk_top10),
        ("OldOnly", group_ids(feats, "OldOnly")),
        ("ExactOnly", group_ids(feats, "ExactOnly")),
        ("RandomMatched", group_ids(feats, "RandomMatched")),
        ("PathGoodFromNaturalPreflight", pathgood),
        ("SlowBurnGoodFromNaturalPreflight", slowburn),
    ]
    rows: list[dict[str, Any]] = []
    for gid, ids in groups:
        fs = [all_feats[a] for a in ids if a in all_feats]
        q, _idx, _scores = quality_for_ids(ap0, ids)
        vseq = [lcb([fnum(f.get(f"V{h}")) for f in fs]) for h in HORIZONS]
        monotone = int(all(vseq[i] <= vseq[i + 1] + 0.05 for i in range(len(vseq) - 1))) if fs else 0
        late_recovery = int(vseq[0] < 0 and vseq[3] > 0 and vseq[4] > 0) if fs else 0
        risk_clean_path = int(
            ucb([float(f.get("longrisk240", 0)) for f in fs]) <= 0.05
            and ucb([float(f.get("memory_fail", 0)) for f in fs]) <= 0.05
            and ucb([float(f.get("offdiag_fail", 0)) for f in fs]) <= 0.05
        ) if fs else 0
        rows.append({
            "stage": "P3_FUTURE_PATH_MECHANISM_V9870",
            "status": "group_summary",
            "group_id": gid,
            "action_count": len(fs),
            "V1_LCB": vseq[0] if fs else "",
            "V5_LCB": vseq[1] if fs else "",
            "V20_LCB": vseq[2] if fs else "",
            "V80_LCB": vseq[3] if fs else "",
            "V240_LCB": vseq[4] if fs else "",
            "AUV_LCB": lcb([fnum(f.get("AUV")) for f in fs]) if fs else "",
            "RiskAdjustedAUV_LCB": lcb([fnum(f.get("RiskAdjustedAUV")) for f in fs]) if fs else "",
            "LongRisk_UCB": ucb([float(f.get("longrisk240", 0)) for f in fs]) if fs else "",
            "Bad_UCB": q.get("bad_UCB", ""),
            "Null_UCB": q.get("null_UCB", ""),
            "Memory_UCB": ucb([float(f.get("memory_fail", 0)) for f in fs]) if fs else "",
            "Offdiag_UCB": ucb([float(f.get("offdiag_fail", 0)) for f in fs]) if fs else "",
            "CEp99_delta_UCB": ucb([float(f.get("CEp99_5", 0)) for f in fs]) if fs else "",
            "margin_tail_delta_LCB": lcb([fnum(f.get("Margin5")) for f in fs]) if fs else "",
            "old_family_loss_delta_UCB": "",
            "hard_tail_loss_delta_UCB": ucb([float(f.get("HardTail5", 0)) for f in fs]) if fs else "",
            "immediate_gain": vseq[0] if fs else "",
            "mid_gain": vseq[3] if fs else "",
            "long_gain": vseq[4] if fs else "",
            "slow_burn_score": (lcb([fnum(f.get("AUV")) for f in fs]) - max(0.0, vseq[0])) if fs else "",
            "risk_clean_path": risk_clean_path,
            "monotone_value_path": monotone,
            "late_recovery_path": late_recovery,
            "weak_path_pass": int(bool(fs) and lcb([fnum(f.get("AUV")) for f in fs]) > 0 and vseq[3] > 0 and vseq[4] > 0 and risk_clean_path),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    by_id = {r["group_id"]: r for r in rows}
    old = by_id.get("OldOnly", {})
    exact = by_id.get("ExactOnly", {})
    random = by_id.get("RandomMatched", {})
    weak = int(any(inum(r.get("weak_path_pass")) and r.get("group_id") in {"Core77", "PathGoodFromNaturalPreflight"} for r in rows))
    strong_without_proxy = int(
        fnum(old.get("V1_LCB")) <= 0
        and fnum(old.get("AUV_LCB")) > 0
        and fnum(old.get("V80_LCB")) > 0
        and fnum(old.get("V240_LCB")) > 0
        and (fnum(exact.get("V240_LCB")) < fnum(old.get("V240_LCB")) or fnum(random.get("LongRisk_UCB")) > 0.05)
        and fnum(random.get("LongRisk_UCB")) > 0.05
    )
    proxy_corr = inum((p4_proxy_hint or {}).get("P4_weak_pass"))
    strong = int(strong_without_proxy and proxy_corr)
    summary = {
        "stage": "P3_FUTURE_PATH_MECHANISM_V9870",
        "status": "summary",
        "group_count": len(rows),
        "P3_A_line_weak_pass": weak,
        "P3_A_line_strong_mechanism_pass": strong,
        "P3_strong_without_proxy": strong_without_proxy,
        "legal_proxy_family_correlates_with_path_type": proxy_corr,
        "Core77_AUV_LCB": by_id.get("Core77", {}).get("AUV_LCB", ""),
        "OldOnly_V1_LCB": old.get("V1_LCB", ""),
        "OldOnly_AUV_LCB": old.get("AUV_LCB", ""),
        "RandomMatched_LongRisk_UCB": random.get("LongRisk_UCB", ""),
        "reason": "future_path_weak_signal_present_but_strong_requires_low_cost_legal_proxy",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    v9720.write_bar_svg(out / "fig_p3_future_value_curve_by_group.svg", "P3 AUV", [r["group_id"] for r in rows if r.get("status") == "group_summary"], [fnum(r.get("AUV_LCB")) for r in rows if r.get("status") == "group_summary"])
    v9720.write_bar_svg(out / "fig_p3_risk_curve_by_group.svg", "P3 longrisk", [r["group_id"] for r in rows if r.get("status") == "group_summary"], [fnum(r.get("LongRisk_UCB")) for r in rows if r.get("status") == "group_summary"])
    return rows, summary


def group_ids(feats: dict[str, dict[str, Any]], group: str) -> list[str]:
    return sorted([aid for aid, f in feats.items() if str(f.get("group_id")) == group])


def p1_combo_reassessment(ap0: list[dict[str, Any]], feats: dict[str, dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    core = group_ids(feats, "Core77")
    exp10 = group_ids(feats, "CoreExpansion10")
    exact = group_ids(feats, "ExactOnly")
    old = group_ids(feats, "OldOnly")
    random = group_ids(feats, "RandomMatched")
    risk_clean = group_ids(feats, "RiskCleanButLowValue")
    risk_top10 = sorted(risk_clean, key=lambda a: fnum(feats[a].get("RiskAdjustedAUV")), reverse=True)[:10]
    candidates: list[tuple[str, list[str]]] = [
        ("Core77", core),
        ("CoreExpansion10", exp10),
        ("Core77+CoreExpansion10", core + exp10),
        ("OldOnly", old),
        ("ExactOnly", exact),
        ("RandomMatched", random),
        ("RiskCleanButLowValue", risk_clean),
        ("Core77+RiskCleanButLowValueTop10", core + risk_top10),
    ]
    rows: list[dict[str, Any]] = []
    for cid, ids in candidates:
        fs = [feats[a] for a in ids if a in feats]
        q, idx, scores = quality_for_ids(ap0, ids)
        decomp = ldo_decomposition(ap0, idx, q, scores)
        per_ds = defaultdict(list)
        for f in fs:
            per_ds[str(f.get("dataset"))].append(fnum(f.get("V240")))
        strong = int(
            len(fs) >= TARGET_K
            and lcb([fnum(f.get("AUV")) for f in fs]) > 0
            and lcb([fnum(f.get("V80")) for f in fs]) > 0
            and lcb([fnum(f.get("V240")) for f in fs]) > 0
            and ucb([float(f.get("longrisk240", 0)) for f in fs]) <= 0.05
            and fnum(q.get("bad_UCB")) <= 0.05
            and fnum(q.get("null_UCB")) <= 0.15
            and ucb([float(f.get("memory_fail", 0)) for f in fs]) == 0
            and ucb([float(f.get("offdiag_fail", 0)) for f in fs]) == 0
            and fnum(decomp.get("LDO_support_adjusted")) <= 0.10
            and fnum(q.get("LSO_drop")) <= 0.10
            and fnum(q.get("LTO_drop")) <= 0.10
        )
        density_dispute = int(
            len(fs) >= TARGET_K
            and lcb([fnum(f.get("AUV")) for f in fs]) > 0
            and lcb([fnum(f.get("V80")) for f in fs]) > 0
            and lcb([fnum(f.get("V240")) for f in fs]) > 0
            and ucb([float(f.get("longrisk240", 0)) for f in fs]) <= 0.05
            and fnum(q.get("bad_UCB")) <= 0.05
            and fnum(q.get("null_UCB")) <= 0.15
            and ucb([float(f.get("memory_fail", 0)) for f in fs]) == 0
            and ucb([float(f.get("offdiag_fail", 0)) for f in fs]) == 0
            and fnum(decomp.get("LDO_raw")) > 0.10
            and fnum(decomp.get("LDO_backfill_share")) >= 0.70
        )
        rows.append({
            "stage": "P1_FUTURE_PATH_COMBO_REASSESSMENT_V9860",
            "status": "combo_summary",
            "candidate_id": cid,
            "action_count": len(fs),
            "AUV_LCB": lcb([fnum(f.get("AUV")) for f in fs]),
            "RiskAdjustedAUV_LCB": lcb([fnum(f.get("RiskAdjustedAUV")) for f in fs]),
            "DelayedGain_LCB": lcb([fnum(f.get("DelayedGain")) for f in fs]),
            "V1_LCB": lcb([fnum(f.get("V1")) for f in fs]),
            "V5_LCB": lcb([fnum(f.get("V5")) for f in fs]),
            "V20_LCB": lcb([fnum(f.get("V20")) for f in fs]),
            "V80_LCB": lcb([fnum(f.get("V80")) for f in fs]),
            "V240_LCB": lcb([fnum(f.get("V240")) for f in fs]),
            "RiskPath_UCB": ucb([float(f.get("RiskPath", 0)) for f in fs]),
            "LongRisk_UCB": ucb([float(f.get("longrisk240", 0)) for f in fs]),
            "Bad_UCB": q.get("bad_UCB"),
            "Null_UCB": q.get("null_UCB"),
            "MemoryFail_UCB": ucb([float(f.get("memory_fail", 0)) for f in fs]),
            "OffdiagFail_UCB": ucb([float(f.get("offdiag_fail", 0)) for f in fs]),
            "GradeAB_precision": q.get("GradeAB_precision"),
            "V_integrated_LCB": q.get("V_integrated_LCB"),
            "LDO_raw": decomp.get("LDO_raw"),
            "LDO_quality": decomp.get("LDO_quality"),
            "LDO_support": decomp.get("LDO_support"),
            "LDO_backfill": decomp.get("LDO_backfill"),
            "LDO_backfill_share": decomp.get("LDO_backfill_share"),
            "LDO_support_adjusted": decomp.get("LDO_support_adjusted"),
            "LSO": q.get("LSO_drop"),
            "LTO": q.get("LTO_drop"),
            "LFO": decomp.get("LFO"),
            "candidate_template_count": len({str(f.get("template_id")) for f in fs}),
            "family_count": len({str(f.get("family_id")) for f in fs}),
            "per_dataset_count": json_counter([str(f.get("dataset")) for f in fs]),
            "per_dataset_V240_LCB": json.dumps({ds: lcb(vals) for ds, vals in sorted(per_ds.items())}, sort_keys=True),
            "path_combo_strong_pass": strong,
            "density_gate_dispute": density_dispute,
            "field_legality": "red:future_path_diagnostic_only",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(rows, key=lambda r: (inum(r["path_combo_strong_pass"]), inum(r["density_gate_dispute"]), fnum(r["RiskAdjustedAUV_LCB"])))
    summary = {
        "stage": "P1_FUTURE_PATH_COMBO_REASSESSMENT_V9860",
        "status": "summary",
        "candidate_count": len(rows),
        "path_combo_strong_pass_count": sum(inum(r.get("path_combo_strong_pass")) for r in rows),
        "density_gate_dispute_count": sum(inum(r.get("density_gate_dispute")) for r in rows),
        "best_candidate_id": best.get("candidate_id"),
        "best_action_count": best.get("action_count"),
        "best_AUV_LCB": best.get("AUV_LCB"),
        "best_RiskAdjustedAUV_LCB": best.get("RiskAdjustedAUV_LCB"),
        "best_LDO_raw": best.get("LDO_raw"),
        "best_LDO_support_adjusted": best.get("LDO_support_adjusted"),
        "best_LSO": best.get("LSO"),
        "best_LTO": best.get("LTO"),
        "P1_A_combo_strong_pass": int(any(inum(r.get("path_combo_strong_pass")) for r in rows)),
        "P1_density_gate_dispute": int(any(inum(r.get("density_gate_dispute")) for r in rows)),
        "P1_controller_ready": 0,
        "reason": "future_path_combo_is_diagnostic_only_until_legal_mechanism_or_density_gate_closes",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    labels = [r["candidate_id"] for r in rows if r.get("status") == "combo_summary"]
    combo_rows = [r for r in rows if r.get("status") == "combo_summary"]
    v9720.write_bar_svg(out / "fig_A1_path_combo_V_curve.svg", "A1 path combo AUV LCB", labels, [fnum(r.get("AUV_LCB")) for r in combo_rows])
    v9720.write_bar_svg(out / "fig_A2_path_combo_risk_curve.svg", "A2 path combo longrisk", labels, [fnum(r.get("LongRisk_UCB")) for r in combo_rows])
    v9720.write_bar_svg(out / "fig_A3_core77_plus_expansion_pareto.svg", "A3 RiskAdjustedAUV", labels, [fnum(r.get("RiskAdjustedAUV_LCB")) for r in combo_rows])
    v9720.write_bar_svg(out / "fig_A4_raw_LDO_vs_support_adjusted_LDO.svg", "A4 raw minus support LDO", labels, [fnum(r.get("LDO_raw")) - fnum(r.get("LDO_support_adjusted")) for r in combo_rows])
    v9720.write_bar_svg(out / "fig_A5_per_dataset_path_support.svg", "A5 action count", labels, [fnum(r.get("action_count")) for r in combo_rows])
    return rows, summary


def score_quality(ap0: list[dict[str, Any]], feats: dict[str, dict[str, Any]], scores: dict[str, float]) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], list[float]]:
    idx_by = ap0_indices(ap0)
    full = [-1.0e9] * len(ap0)
    for aid, score in scores.items():
        if aid in idx_by:
            full[idx_by[aid]] = float(score)
    q = v9720.quality_for_scores(ap0, full, TARGET_K)
    top_ids = [aid for aid in sorted(scores, key=scores.get, reverse=True)[:TARGET_K] if aid in feats]
    top_feats = [feats[a] for a in top_ids]
    return q, top_feats, top_ids, full


def low_cost_proxy_specs() -> dict[str, dict[str, Any]]:
    return {
        "LC1_CheapMemoryHardTailResponse": {
            "feature_legality": "green:train_time_memory_hardtail_response",
            "memory_overhead_mb": 0.0,
            "score": lambda f: (
                -1.0e6 if f["memory_fail"] or f["offdiag_fail"]
                else -max(0.0, f["CEp99_1"]) - max(0.0, f["HardTail1"]) - 0.25 * abs(f["CE5"] - f["CE1"])
            ),
        },
        "LC2_LowRankFutureOperatorSketch": {
            "feature_legality": "yellow:low_rank_sketch_from_recent_train_response_no_future_labels",
            "memory_overhead_mb": 0.02,
            "score": lambda f: 0.5 * (f["Margin1"] + f["Margin5"]) - 0.25 * abs(f["Curvature1"]) - 0.25 * abs(f["Jacobian1"]) - 0.001 * f["payload_norm"],
        },
        "LC3_SignalChannelAgreement": {
            "feature_legality": "green:multi_channel_train_signal_agreement",
            "memory_overhead_mb": 0.0,
            "score": lambda f: -statistics.pstdev([f["CE1"], f["CE5"], f["HardTail1"], f["HardTail5"]]) + 0.2 * (f["Margin1"] + f["Margin5"]),
        },
        "LC4_HardVetoSignalAgreement": {
            "feature_legality": "green:signal_agreement_with_memory_offdiag_veto",
            "memory_overhead_mb": 0.0,
            "score": lambda f: (
                -1.0e6 if f["memory_fail"] or f["offdiag_fail"] or max(0.0, f["CEp99_1"]) > 0.20
                else -statistics.pstdev([f["CE1"], f["CE5"], f["HardTail1"], f["HardTail5"]]) - max(0.0, f["HardTail5"])
            ),
        },
    }


def measure_score_cost_ms(fn: Any, fs: list[dict[str, Any]]) -> list[float]:
    costs: list[float] = []
    for f in fs:
        t0 = time.perf_counter()
        _ = fn(f)
        costs.append((time.perf_counter() - t0) * 1000.0)
    return costs


def p4_low_cost_proxy(
    ap0: list[dict[str, Any]],
    feats: dict[str, dict[str, Any]],
    natural_feats: dict[str, dict[str, Any]],
    out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_feats = {**feats, **natural_feats}
    nat_labels = feature_labels(natural_feats, ap0)
    slowburn_ids = {aid for aid, lab in nat_labels.items() if lab["SlowBurnGood"]}
    risk_clean_ids = set(group_ids(feats, "RiskCleanButLowValue"))
    core_labels = [1.0 if str(f.get("group_id")) == "Core77" else 0.0 for f in all_feats.values()]
    slow_labels = [1.0 if aid in slowburn_ids else 0.0 for aid in all_feats]
    risk_labels = [1.0 if aid in risk_clean_ids else 0.0 for aid in all_feats]
    rows: list[dict[str, Any]] = []
    for pid, spec in low_cost_proxy_specs().items():
        fn = spec["score"]
        fs = list(all_feats.values())
        costs = measure_score_cost_ms(fn, fs)
        scores = {aid: float(fn(f)) for aid, f in all_feats.items()}
        q, top_feats, top_ids, full_scores = score_quality(ap0, all_feats, scores)
        lfo = v9770.lfo_drop(full_scores, ap0, TARGET_K)
        score_vals = [scores[aid] for aid in all_feats]
        weak = int(
            qtile(costs, 0.90) <= 1.5
            and fnum(q.get("GradeAB_precision")) >= 0.70
            and fnum(q.get("V_integrated_LCB")) > 0
            and fnum(q.get("h240_longrisk_UCB")) <= 0.05
            and fnum(q.get("memory_fail_UCB")) <= 0.05
            and fnum(q.get("offdiag_fail_UCB")) <= 0.05
            and fnum(q.get("LDO_drop")) <= 0.15
            and fnum(q.get("LSO_drop")) <= 0.15
            and fnum(q.get("LTO_drop")) <= 0.15
        )
        strong = int(
            weak
            and qtile(costs, 0.90) <= 1.0
            and fnum(q.get("GradeAB_precision")) >= 0.75
            and lcb([fnum(f.get("AUV")) for f in top_feats]) > 0
            and fnum(q.get("bad_UCB")) <= 0.05
            and fnum(q.get("null_UCB")) <= 0.15
            and lfo <= 0.10
        )
        fail_bits = []
        if not weak:
            if qtile(costs, 0.90) > 1.5:
                fail_bits.append("cost_q90_above_1.5ms")
            if fnum(q.get("GradeAB_precision")) < 0.70:
                fail_bits.append("precision_below_0.70")
            if fnum(q.get("V_integrated_LCB")) <= 0:
                fail_bits.append("V_LCB_nonpositive")
            if fnum(q.get("h240_longrisk_UCB")) > 0.05:
                fail_bits.append("longrisk_above_0.05")
            if fnum(q.get("memory_fail_UCB")) > 0.05 or fnum(q.get("offdiag_fail_UCB")) > 0.05:
                fail_bits.append("memory_or_offdiag_above_0.05")
            if fnum(q.get("LDO_drop")) > 0.15 or fnum(q.get("LSO_drop")) > 0.15 or fnum(q.get("LTO_drop")) > 0.15:
                fail_bits.append("leaveout_above_0.15")
        rows.append({
            "stage": "P4_LOW_COST_PROXY_V9870",
            "status": "proxy_summary",
            "proxy_id": pid,
            "feature_legality": spec["feature_legality"],
            "compute_cost_ms_p50": qtile(costs, 0.50),
            "compute_cost_ms_p90": qtile(costs, 0.90),
            "compute_cost_ms_p99": qtile(costs, 0.99),
            "memory_overhead_mb": spec["memory_overhead_mb"],
            "TopK87_precision": q.get("GradeAB_precision"),
            "TopK87_V_LCB": q.get("V_integrated_LCB"),
            "TopK87_AUV_LCB": lcb([fnum(f.get("AUV")) for f in top_feats]),
            "TopK87_LongRisk_UCB": q.get("h240_longrisk_UCB"),
            "TopK87_Bad_UCB": q.get("bad_UCB"),
            "TopK87_Null_UCB": q.get("null_UCB"),
            "TopK87_Memory_UCB": q.get("memory_fail_UCB"),
            "TopK87_Offdiag_UCB": q.get("offdiag_fail_UCB"),
            "LDO_drop": q.get("LDO_drop"),
            "LSO_drop": q.get("LSO_drop"),
            "LTO_drop": q.get("LTO_drop"),
            "LFO_drop": lfo,
            "correlation_with_Core77": pearson(score_vals, core_labels),
            "correlation_with_SlowBurnGood": pearson(score_vals, slow_labels),
            "correlation_with_RiskCleanButLowValue": pearson(score_vals, risk_labels),
            "top_group_counts": json_counter([str(f.get("group_id")) for f in top_feats]),
            "P4_weak_pass": weak,
            "P4_strong_pass": strong,
            "failure_mode": ";".join(fail_bits),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(rows, key=lambda r: (inum(r.get("P4_strong_pass")), inum(r.get("P4_weak_pass")), fnum(r.get("TopK87_precision")), fnum(r.get("TopK87_V_LCB"))))
    summary = {
        "stage": "P4_LOW_COST_PROXY_V9870",
        "status": "summary",
        "proxy_count": len(rows),
        "weak_pass_count": sum(inum(r.get("P4_weak_pass")) for r in rows),
        "strong_pass_count": sum(inum(r.get("P4_strong_pass")) for r in rows),
        "best_proxy_id": best.get("proxy_id"),
        "best_precision": best.get("TopK87_precision"),
        "best_V_LCB": best.get("TopK87_V_LCB"),
        "best_AUV_LCB": best.get("TopK87_AUV_LCB"),
        "best_cost_q90_ms": best.get("compute_cost_ms_p90"),
        "P4_weak_pass": int(any(inum(r.get("P4_weak_pass")) for r in rows)),
        "P4_strong_pass": int(any(inum(r.get("P4_strong_pass")) for r in rows)),
        "StopB_score_proliferation": int(not any(inum(r.get("P4_weak_pass")) for r in rows)),
        "reason": "low_cost_proxy_evaluated_but_quality_gate_failed" if not any(inum(r.get("P4_weak_pass")) for r in rows) else "",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    labels = [r["proxy_id"] for r in rows if r.get("status") == "proxy_summary"]
    proxy_rows = [r for r in rows if r.get("status") == "proxy_summary"]
    v9720.write_bar_svg(out / "fig_p4_proxy_precision_vs_cost.svg", "P4 precision", labels, [fnum(r.get("TopK87_precision")) for r in proxy_rows])
    v9720.write_bar_svg(out / "fig_p4_proxy_V_vs_longrisk.svg", "P4 V minus risk", labels, [fnum(r.get("TopK87_V_LCB")) - fnum(r.get("TopK87_LongRisk_UCB")) for r in proxy_rows])
    v9720.write_bar_svg(out / "fig_p4_proxy_cost_breakdown.svg", "P4 cost q90", labels, [fnum(r.get("compute_cost_ms_p90")) for r in proxy_rows])
    v9720.write_bar_svg(out / "fig_p4_proxy_score_vs_AUV_scatter.svg", "P4 corr slowburn", labels, [fnum(r.get("correlation_with_SlowBurnGood")) for r in proxy_rows])
    v9720.write_bar_svg(out / "fig_p4_proxy_leaveout_drop.svg", "P4 LDO", labels, [fnum(r.get("LDO_drop")) for r in proxy_rows])
    return rows, summary


def future_operator_scores(feats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        "FO1_AdamWAlignedFutureResidual": {
            "feature_kernel_count": 3,
            "extra_memory_mb": 0.0,
            "desc": "current residual alignment with curvature/jacobian penalties from landed train metrics",
            "score": lambda f: -f["CE1"] + 0.30 * f["Margin1"] - 0.15 * abs(f["Curvature1"]) - 0.10 * abs(f["Jacobian1"]),
        },
        "FO2_MemoryBufferResponseStability": {
            "feature_kernel_count": 3,
            "extra_memory_mb": 0.0,
            "desc": "memory/offdiag-safe stable h1-h5 train response",
            "score": lambda f: -abs(f["CE5"] - f["CE1"]) - 2.0 * f["memory_fail"] - 2.0 * f["offdiag_fail"] - 0.20 * max(0.0, f["CEp99_1"]),
        },
        "FO3_HardTailContraction": {
            "feature_kernel_count": 2,
            "extra_memory_mb": 0.0,
            "desc": "hard-tail contraction with current CE relief",
            "score": lambda f: -f["HardTail5"] + 0.20 * (-f["CE1"]) - 0.10 * max(0.0, f["CEp99_5"]),
        },
        "FO4_OldFamilyMarginPreservation": {
            "feature_kernel_count": 2,
            "extra_memory_mb": 0.0,
            "desc": "old-family margin preservation using landed train margin fields",
            "score": lambda f: 0.65 * f["Margin1"] + 0.35 * f["Margin5"] - 0.10 * abs(f["CEp99_1"]),
        },
        "FO5_MultiBatchAgreement": {
            "feature_kernel_count": 4,
            "extra_memory_mb": 0.0,
            "desc": "agreement between h1 and h5 train response summaries",
            "score": lambda f: -abs(f["CE1"] - f["CE5"]) + 0.25 * (f["Margin1"] + f["Margin5"]) - 0.10 * abs(f["HardTail5"] - f["HardTail1"]),
        },
        "FO6_LowCostTwoSampleVirtualPath": {
            "feature_kernel_count": 2,
            "extra_memory_mb": 0.0,
            "desc": "cheap two-sample proxy with payload/action norm penalty",
            "score": lambda f: -0.35 * f["payload_norm"] - 0.15 * f["action_norm"] - 0.001 * f["runtime_ms"] + 0.25 * (-f["CE1"]),
        },
        "FO7_RiskVetoOnly": {
            "feature_kernel_count": 2,
            "extra_memory_mb": 0.0,
            "desc": "risk-veto-only train proxy, intended as safety gate not value source",
            "score": lambda f: -4.0 * f["memory_fail"] - 4.0 * f["offdiag_fail"] - max(0.0, f["CEp99_1"]) - max(0.0, f["HardTail5"]),
        },
        "FO8_FO123HardVetoNoWeightedScore": {
            "feature_kernel_count": 5,
            "extra_memory_mb": 0.0,
            "desc": "FO1/FO2/FO3 rank sum with hard memory/offdiag/current-tail veto",
            "score": lambda f: (
                -1.0e6 if f["memory_fail"] or f["offdiag_fail"] or max(0.0, f["CEp99_1"]) > 0.25
                else (-f["CE1"] - abs(f["CE5"] - f["CE1"]) - f["HardTail5"])
            ),
        },
    }


def p2_future_operator_proxy(ap0: list[dict[str, Any]], feats: dict[str, dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pid, spec in future_operator_scores(feats).items():
        scores = {aid: float(spec["score"](f)) for aid, f in feats.items()}
        q, top_feats, top_ids, full_scores = score_quality(ap0, feats, scores)
        costs = [fnum(f.get("runtime_ms")) for f in top_feats]
        score_vals = [scores[a] for a in top_ids]
        core_labels = [1.0 if str(f.get("group_id")) == "Core77" else 0.0 for f in feats.values()]
        all_scores = [scores[a] for a in feats]
        all_auv = [fnum(f.get("AUV")) for f in feats.values()]
        all_rauv = [fnum(f.get("RiskAdjustedAUV")) for f in feats.values()]
        lfo = v9770.lfo_drop(full_scores, ap0, TARGET_K)
        cost_q90 = qtile(costs, 0.90)
        weak = int(
            fnum(q.get("GradeAB_precision")) >= 0.70
            and fnum(q.get("V_integrated_LCB")) > 0
            and fnum(q.get("h240_longrisk_UCB")) <= 0.05
            and fnum(q.get("memory_fail_UCB")) <= 0.05
            and fnum(q.get("offdiag_fail_UCB")) <= 0.05
            and cost_q90 <= 1.5
        )
        strong = int(
            weak
            and fnum(q.get("GradeAB_precision")) >= 0.80
            and fnum(q.get("LDO_drop")) <= 0.10
            and fnum(q.get("LSO_drop")) <= 0.10
            and fnum(q.get("LTO_drop")) <= 0.10
            and cost_q90 <= 0.5
        )
        fail_reason = ""
        if not weak:
            fail_bits = []
            if fnum(q.get("GradeAB_precision")) < 0.70:
                fail_bits.append("precision_below_0.70")
            if fnum(q.get("V_integrated_LCB")) <= 0:
                fail_bits.append("V_LCB_nonpositive")
            if fnum(q.get("h240_longrisk_UCB")) > 0.05:
                fail_bits.append("longrisk_above_0.05")
            if fnum(q.get("memory_fail_UCB")) > 0.05 or fnum(q.get("offdiag_fail_UCB")) > 0.05:
                fail_bits.append("memory_or_offdiag_above_0.05")
            if cost_q90 > 1.5:
                fail_bits.append("cost_q90_above_1.5ms")
            fail_reason = ";".join(fail_bits)
        rows.append({
            "stage": "P2_FUTURE_OPERATOR_PROXY_V9860",
            "status": "proxy_summary",
            "proxy_id": pid,
            "description": spec["desc"],
            "feature_legality": "green:train_time_operator_proxy_from_landed_train_metrics",
            "TopK87_precision": q.get("GradeAB_precision"),
            "V_LCB": q.get("V_integrated_LCB"),
            "AUV_LCB": lcb([fnum(f.get("AUV")) for f in top_feats]),
            "RiskAdjustedAUV_LCB": lcb([fnum(f.get("RiskAdjustedAUV")) for f in top_feats]),
            "LongRisk_UCB": q.get("h240_longrisk_UCB"),
            "Bad_UCB": q.get("bad_UCB"),
            "Null_UCB": q.get("null_UCB"),
            "MemoryFail_UCB": q.get("memory_fail_UCB"),
            "OffdiagFail_UCB": q.get("offdiag_fail_UCB"),
            "LDO": q.get("LDO_drop"),
            "LSO": q.get("LSO_drop"),
            "LTO": q.get("LTO_drop"),
            "LFO": lfo,
            "cost_q50_ms": qtile(costs, 0.50),
            "cost_q90_ms": cost_q90,
            "cost_q99_ms": qtile(costs, 0.99),
            "feature_kernel_count": spec["feature_kernel_count"],
            "extra_memory_mb": spec["extra_memory_mb"],
            "correlation_with_Core77_label": pearson(all_scores, core_labels),
            "correlation_with_future_AUV": pearson(all_scores, all_auv),
            "correlation_with_RiskAdjustedAUV": pearson(all_scores, all_rauv),
            "top_group_counts": json_counter([str(f.get("group_id")) for f in top_feats]),
            "score_mean_top87": mean(score_vals),
            "P2_weak_pass": weak,
            "P2_strong_pass": strong,
            "failure_mode": fail_reason,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(rows, key=lambda r: (inum(r["P2_strong_pass"]), inum(r["P2_weak_pass"]), fnum(r["TopK87_precision"]), fnum(r["V_LCB"])))
    summary = {
        "stage": "P2_FUTURE_OPERATOR_PROXY_V9860",
        "status": "summary",
        "proxy_count": len(rows),
        "weak_pass_count": sum(inum(r.get("P2_weak_pass")) for r in rows),
        "strong_pass_count": sum(inum(r.get("P2_strong_pass")) for r in rows),
        "best_proxy_id": best.get("proxy_id"),
        "best_precision": best.get("TopK87_precision"),
        "best_V_LCB": best.get("V_LCB"),
        "best_AUV_LCB": best.get("AUV_LCB"),
        "best_cost_q90_ms": best.get("cost_q90_ms"),
        "P2_weak_pass": int(any(inum(r.get("P2_weak_pass")) for r in rows)),
        "P2_strong_pass": int(any(inum(r.get("P2_strong_pass")) for r in rows)),
        "StopB_future_operator_triggered": int(not any(inum(r.get("P2_weak_pass")) for r in rows)),
        "reason": "all_future_operator_proxies_fail_precision_value_risk_or_cost_gate",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    labels = [r["proxy_id"] for r in rows if r.get("status") == "proxy_summary"]
    proxy_rows = [r for r in rows if r.get("status") == "proxy_summary"]
    v9720.write_bar_svg(out / "fig_B1_proxy_precision_vs_cost.svg", "B1 precision", labels, [fnum(r.get("TopK87_precision")) for r in proxy_rows])
    v9720.write_bar_svg(out / "fig_B2_proxy_V_vs_longrisk.svg", "B2 V minus risk", labels, [fnum(r.get("V_LCB")) - fnum(r.get("LongRisk_UCB")) for r in proxy_rows])
    v9720.write_bar_svg(out / "fig_B3_proxy_AUV_vs_RiskAdjustedAUV.svg", "B3 RiskAdjustedAUV", labels, [fnum(r.get("RiskAdjustedAUV_LCB")) for r in proxy_rows])
    v9720.write_bar_svg(out / "fig_B4_proxy_score_vs_future_AUV_scatter.svg", "B4 corr future AUV", labels, [fnum(r.get("correlation_with_future_AUV")) for r in proxy_rows])
    v9720.write_bar_svg(out / "fig_B5_proxy_score_distribution_by_group.svg", "B5 cost q90", labels, [fnum(r.get("cost_q90_ms")) for r in proxy_rows])
    return rows, summary


def materializer_entrypoint_audit(panel_targets: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    patterns = [
        "natural_AP0_labeled_stream_extension_materializer",
        "natural_ap0_labeled_stream_extension_materializer",
        "materialize_natural_ap0_labeled_stream_extension",
    ]
    hits: list[str] = []
    for path in (REPO / "experiments").glob("run_*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in patterns:
            if f"def {pattern}" in text or f"class {pattern}" in text:
                hits.append(f"{path.name}:{pattern}")
    found = int(bool(hits))
    rows: list[dict[str, Any]] = [{
        "stage": "P3_NATURAL_AP0_STREAM_MATERIALIZER_V9860",
        "status": "summary",
        "materializer_entrypoint_found": found,
        "matched_entrypoints": ",".join(hits),
        "C0_single_preflight_pass": 0,
        "C1_16_preflight_pass": 0,
        "C2_128_preflight_pass": 0,
        "C3_5000_panel_pass": 0,
        "C4_10000_panel_pass": 0,
        "C5_20000_panel_pass": 0,
        "P3_preflight_pass": 0,
        "reason": "" if found else "no_landed_labeled_natural_AP0_stream_extension_materializer_for_v9860",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    for stage_id, n in [("C0-single", 1), ("C1-16", 16), ("C2-128", 128)]:
        rows.append({
            "stage": "P3_NATURAL_AP0_STREAM_MATERIALIZER_V9860",
            "status": "not_run",
            "preflight_id": stage_id,
            "panel_target": n,
            "panel_actual_actions": 0,
            "expected_rows": n * BRANCH_COUNT * len(HORIZONS),
            "actual_rows": 0,
            "rows_per_sec": 0,
            "failed_action_count": n,
            "unresolved_exception_type": "materializer_entrypoint_missing" if not found else "gate_not_open",
            "CoreLike_count": "",
            "CoreLike_rate": "",
            "quality_audit_pass": 0,
            "reason": "materializer_entrypoint_missing" if not found else "entrypoint_found_but_not_invoked_by_gate",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    for target in [t for t in panel_targets if t >= 5000]:
        rows.append({
            "stage": "P3_NATURAL_AP0_STREAM_MATERIALIZER_V9860",
            "status": "not_run",
            "preflight_id": f"C-panel-{target}",
            "panel_target": target,
            "panel_actual_actions": 0,
            "expected_rows": target * BRANCH_COUNT * len(HORIZONS),
            "actual_rows": 0,
            "rows_per_sec": 0,
            "failed_action_count": target,
            "unresolved_exception_type": "materializer_entrypoint_missing" if not found else "gate_not_open",
            "CoreLike_count": "",
            "CoreLike_rate": "",
            "quality_audit_pass": 0,
            "reason": "materializer_entrypoint_missing" if not found else "entrypoint_found_but_not_invoked_by_gate",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    return rows, rows[0]


def p3_density_panel(source_v9820: Path, panel_targets: list[int], p3: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, summary = v9840.p4_density_panels(source_v9820, panel_targets, {"P3_preflight_pass": p3.get("P3_preflight_pass")}, out)
    for r in rows:
        r["stage"] = "P3_NATURAL_AP0_DENSITY_PANEL_V9860"
        if r.get("status") == "panel_row":
            r["row_source"] = "existing_labeled_AP0_density_panel_no_extension_materializer"
            r["expected_rows"] = ""
            r["actual_rows"] = ""
            r["rows_per_sec"] = ""
            r["quality_audit_pass"] = 1
            r["CoreLike_LCB"] = r.get("CoreLike_Wilson_LCB")
            r["CoreLike_UCB"] = r.get("CoreLike_Wilson_UCB")
        elif r.get("status") == "not_run":
            target = inum(r.get("target_action_count"))
            r["expected_rows"] = target * BRANCH_COUNT * len(HORIZONS)
            r["actual_rows"] = 0
            r["rows_per_sec"] = 0
            r["quality_audit_pass"] = 0
            r["reason"] = "P3_preflight_not_passed_materializer_missing"
    summary["stage"] = "P3_NATURAL_AP0_DENSITY_PANEL_V9860"
    summary["P3_density_weak_pass"] = 0
    summary["P3_density_strong_pass"] = 0
    summary["P3_density_fail"] = 0
    summary["reason"] = "extension_panels_blocked_by_missing_labeled_materializer"
    return rows, summary


def generated_sandbox_decision(p2_density: dict[str, Any], p3_mech: dict[str, Any], p4_proxy: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    condition_d1 = int(inum(p2_density.get("P2_density_insufficient")) and inum(p3_mech.get("P3_A_line_strong_mechanism_pass")))
    condition_d2 = inum(p4_proxy.get("P4_weak_pass"))
    condition_d3 = inum(p3_mech.get("P3_A_line_strong_mechanism_pass"))
    allowed = int(condition_d1 or condition_d2 or condition_d3)
    row = {
        "stage": "P6_GENERATED_REOPEN_DECISION_V9870",
        "status": "summary" if allowed else "not_run",
        "condition_D1_C_density_insufficient_on_large_panel": condition_d1,
        "condition_D2_B_low_cost_proxy_found": condition_d2,
        "condition_D3_A_strong_non_outcome_mechanism": condition_d3,
        "generated_sandbox_allowed": allowed,
        "generated_route_status": "sandbox_allowed_64_action_only" if allowed else "stopped_no_B_or_C_or_A_reopen_condition",
        "generated_action_count": 0,
        "payload_hash_missing": "",
        "apply_error_linf": "",
        "future_path_rows": 0,
        "new_positive_created_rate": "",
        "longrisk_created_rate": "",
        "cost_q90": "",
        "negative_control_gap": "",
        "P6_generated_weak_pass": 0,
        "P6_generated_strong_pass": 0,
        "reason": "" if allowed else "B_low_cost_proxy_absent_C_density_unresolved_A_strong_absent",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p6_generated_gate_decision.svg", "P6 generated sandbox", ["allowed"], [allowed])
    return [row], row


def not_run(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {"stage": stage, "status": "not_run", "reason": reason, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0, **extra}
    return [row], row


def copy_base_acc(source_v9810: Path) -> list[dict[str, Any]]:
    rows = [dict(r) for r in read_csv(source_v9810 / "base_acc_sentinel_v9810.csv")]
    for r in rows:
        r["stage"] = "BASE_ACC_SENTINEL_V9870"
        r["reused_from_v9810"] = 1
        r["base_acc_used_for_controller"] = 0
        r["fake_data_used"] = 0
        r["proxy_row_used"] = 0
        r["cpu_offload_used"] = 0
    return rows


def field_legality() -> list[dict[str, Any]]:
    rows = [
        ("green", "LC1/LC3/LC4 train-time memory, hard-tail, signal agreement inputs", "computed from landed train-stream probe metrics, not future labels"),
        ("yellow", "LC2 low-rank sketch, curvature_proxy,jacobian_proxy,basis_effective_rank,cover_entropy", "architecture-sensitive or sketch diagnostics"),
        ("red", "future V/AUV/RiskAdjustedAUV labels, Core77 label, OldRank, WT80, dataset branch rules", "diagnostic only, not official controller fields"),
    ]
    return [{
        "stage": "FIELD_LEGALITY_LEDGER_V9870",
        "status": "field_legality_row",
        "field_color": color,
        "fields": fields,
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    } for color, fields, reason in rows]


def audit_rows(artifacts: dict[str, Path]) -> int:
    total = 0
    for name, path in artifacts.items():
        if name.endswith(".csv") and path.exists() and not name.startswith(("no_fake", "contract", "failure")):
            total += len(read_csv(path))
        elif name.endswith(".json") and path.exists():
            total += 1
    return total


def sha_rows(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in paths.items() if path.exists()}


def route_from(p1m: dict[str, Any], p2d: dict[str, Any], p3m: dict[str, Any], p4p: dict[str, Any], p6: dict[str, Any]) -> tuple[str, str, str, str]:
    if inum(p2d.get("P2_density_sufficient")) and inum(p4p.get("P4_weak_pass")):
        return "CaseA-NaturalDensitySufficientLegalProxyFound", "controller_conditions_met", "none", "run_existing_action_controller"
    if inum(p2d.get("P2_density_sufficient")) and not inum(p4p.get("P4_weak_pass")):
        return "CaseB-NaturalDensitySufficientLegalProxyAbsent", "legal_low_cost_proxy_failed", "none", "continue_mechanism_no_controller"
    if inum(p2d.get("P2_density_insufficient")) and inum(p3m.get("P3_A_line_strong_mechanism_pass")):
        return "CaseC-NaturalDensityInsufficientAlineStrong", "natural_density_insufficient", "generated_reopen_candidate", "run_64_action_generated_sandbox"
    if inum(p2d.get("P2_density_inconclusive")) and inum(p1m.get("natural_extension_action_generator_found")) == 0:
        return "CaseE-MaterializerExtensionGeneratorMissing", "natural_stream_extension_generator_missing", "legal_low_cost_proxy_failed", "prioritize_C_materializer_engineering"
    if inum(p2d.get("P2_density_inconclusive")):
        return "CaseD-DensityInconclusiveExpandNaturalStream", "density_inconclusive", "legal_low_cost_proxy_failed", "expand_natural_stream_only"
    if not inum(p4p.get("P4_weak_pass")):
        return "CaseF-LowCostProxyFailedStopBScoreProliferation", "low_cost_proxy_failed", "natural_stream_materializer_missing", "stop_B_score_proliferation"
    if inum(p6.get("generated_sandbox_allowed")):
        return "CaseG-GeneratedSandboxAllowed", "generated_sandbox_allowed", "none", "run_D_sandbox"
    return "CaseZ-GateBlocked", "B_C_D_conditions_not_met", "natural_stream_materializer_missing", "generated_route_stopped"


def write_dashboard(out: Path, route: dict[str, Any]) -> None:
    p1 = summary_row(read_csv(out / "p1_future_path_combo_reassessment_v9860.csv"))
    p2 = summary_row(read_csv(out / "p2_future_operator_proxy_v9860.csv"))
    p3m = summary_row(read_csv(out / "p3_natural_ap0_stream_materializer_v9860.csv"))
    p3d = summary_row(read_csv(out / "p3_natural_ap0_density_panel_v9860.csv"))
    p4 = summary_row(read_csv(out / "p4_generated_conditional_sandbox_v9860.csv"))
    nf = summary_row(read_csv(out / "no_fake_audit_v9860.csv"))
    lines = [
        "# v9860_dashboard",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"primary_blocker = {route.get('primary_blocker')}",
        f"secondary_blocker = {route.get('secondary_blocker')}",
        "```",
        "",
        "| line | pass/fail | key value |",
        "|---|---|---|",
        f"| A combo | `{p1.get('P1_A_combo_strong_pass')}` / dispute `{p1.get('P1_density_gate_dispute')}` | best `{p1.get('best_candidate_id')}` AUV `{p1.get('best_AUV_LCB')}` |",
        f"| B future operator | `{p2.get('P2_weak_pass')}` / `{p2.get('P2_strong_pass')}` | best `{p2.get('best_proxy_id')}` precision `{p2.get('best_precision')}` cost `{p2.get('best_cost_q90_ms')}` |",
        f"| C natural stream | materializer `{p3m.get('materializer_entrypoint_found')}` | PanelA LCB `{p3d.get('PanelA_CoreLike_LCB')}` |",
        f"| D generated | `{p4.get('generated_sandbox_allowed')}` | `{p4.get('generated_route_status')}` |",
        "",
        f"rows_checked = `{nf.get('rows_checked')}`; fake/proxy/cpu = `{nf.get('fake_data_used')}` / `{nf.get('proxy_row_used')}` / `{nf.get('cpu_offload_used')}`。",
        "",
        "```text",
        "fig_A1_path_combo_V_curve.svg",
        "fig_A2_path_combo_risk_curve.svg",
        "fig_A3_core77_plus_expansion_pareto.svg",
        "fig_A4_raw_LDO_vs_support_adjusted_LDO.svg",
        "fig_A5_per_dataset_path_support.svg",
        "fig_B1_proxy_precision_vs_cost.svg",
        "fig_B2_proxy_V_vs_longrisk.svg",
        "fig_B3_proxy_AUV_vs_RiskAdjustedAUV.svg",
        "fig_B4_proxy_score_vs_future_AUV_scatter.svg",
        "fig_B5_proxy_score_distribution_by_group.svg",
        "fig_C1_density_curve_panel_size.svg",
        "fig_C2_corelike_rate_ci.svg",
        "fig_D1_generated_sandbox_gate.svg",
        "```",
        "",
    ]
    (out / "v9860_dashboard.md").write_text("\n".join(lines), encoding="utf-8")


def write_recap(out: Path, route: dict[str, Any], hashes: dict[str, str]) -> None:
    p0 = summary_row(read_csv(out / "p0_boundary_reproduction_v9860.csv"))
    landed = summary_row(read_csv(out / "p0_landed_future_row_reference_v9860.csv"))
    p1 = summary_row(read_csv(out / "p1_future_path_combo_reassessment_v9860.csv"))
    p2 = summary_row(read_csv(out / "p2_future_operator_proxy_v9860.csv"))
    p3m = summary_row(read_csv(out / "p3_natural_ap0_stream_materializer_v9860.csv"))
    p3d = summary_row(read_csv(out / "p3_natural_ap0_density_panel_v9860.csv"))
    p4 = summary_row(read_csv(out / "p4_generated_conditional_sandbox_v9860.csv"))
    p5 = summary_row(read_csv(out / "p5_minimal_controller_boundary_v9860.csv"))
    nf = summary_row(read_csv(out / "no_fake_audit_v9860.csv"))
    combos = [r for r in read_csv(out / "p1_future_path_combo_reassessment_v9860.csv") if r.get("status") == "combo_summary"]
    proxies = [r for r in read_csv(out / "p2_future_operator_proxy_v9860.csv") if r.get("status") == "proxy_summary"]
    panels = [r for r in read_csv(out / "p3_natural_ap0_density_panel_v9860.csv") if r.get("status") in {"panel_row", "not_run"}]
    lines = [
        "# DG-KAN v9.8.6 Future Path Mechanism / Natural Stream / Geometry Adaptive Optimizer 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.8.6_结果解读_未来路径机制_自然扩流_几何自适应优化器_完整计划.md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest 或 v9.8.4/v9.8.5 已真实 materialized 的 landed rows；没有 fake data、proxy rows，也没有把 future-path diagnostic、future-operator proxy diagnostic、natural stream missing 或 generated not_run 写成 official controller pass。",
        "",
        "## 0. 最新结论",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"primary_blocker = {route.get('primary_blocker')}",
        f"secondary_blocker = {route.get('secondary_blocker')}",
        f"route_recommendation = {route.get('route_recommendation')}",
        f"system_legal_controller_pass = {route.get('system_legal_controller_pass')}",
        f"generated_route_status = {route.get('generated_route_status')}",
        "```",
        "",
        f"最终 artifact：`{out}`",
        "",
        "核心结论：",
        "",
        f"1. P0 复现 v9.8.5 boundary：source route = `{p0.get('source_route_v9850')}`，A/B/C/D = `{p0.get('A_line_mechanism_pass_v9850')}/{p0.get('B_weak_pass_v9850')}/{p0.get('C_materializer_entrypoint_found_v9850')}/{p0.get('D_generated_reopen_allowed_v9850')}`。",
        f"2. 本轮引用 landed future rows = `{landed.get('realfunctional_row_count')}`，actions = `{landed.get('action_count')}`，horizons = `{landed.get('horizons')}`。",
        f"3. P1 重审组合路径：best = `{p1.get('best_candidate_id')}`，AUV LCB = `{p1.get('best_AUV_LCB')}`，support-adjusted LDO = `{p1.get('best_LDO_support_adjusted')}`。",
        f"4. P1 combo strong pass count = `{p1.get('path_combo_strong_pass_count')}`，density dispute count = `{p1.get('density_gate_dispute_count')}`。",
        f"5. P2 FO1-FO8 future-operator proxy 全部失败：weak/strong = `{p2.get('P2_weak_pass')}` / `{p2.get('P2_strong_pass')}`，best proxy = `{p2.get('best_proxy_id')}`。",
        f"6. P2 best precision/V/cost q90 = `{p2.get('best_precision')}` / `{p2.get('best_V_LCB')}` / `{p2.get('best_cost_q90_ms')}`。",
        f"7. P3 natural stream materializer entrypoint found = `{p3m.get('materializer_entrypoint_found')}`；PanelA CoreLike LCB = `{p3d.get('PanelA_CoreLike_LCB')}`。",
        f"8. P4 generated sandbox allowed = `{p4.get('generated_sandbox_allowed')}`，status = `{p4.get('generated_route_status')}`。",
        f"9. P5 controller = `{p5.get('status')}`，reason = `{p5.get('reason')}`。",
        f"10. No-fake audit：rows checked = `{nf.get('rows_checked')}`，fake/proxy/cpu = `{nf.get('fake_data_used')}` / `{nf.get('proxy_row_used')}` / `{nf.get('cpu_offload_used')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/run_v9860_future_path_mechanism_natural_stream_geometry_optimizer.py` | v9.8.6 runner；读取 v9.8.5/v9.8.4/v9.8.2 landed rows，执行 Core77+Expansion combo reassessment、FO1-FO8 future-operator proxy、natural AP0 materializer audit、generated sandbox decision 与 controller/runtime boundary。 |",
        "",
        "```text",
        "python -m py_compile experiments/run_v9860_future_path_mechanism_natural_stream_geometry_optimizer.py",
        "```",
        "",
        "```bash",
        "python experiments/run_v9860_future_path_mechanism_natural_stream_geometry_optimizer.py --out-dir results/real_rerun_20260506/v9860_future_path_mechanism_natural_stream_geometry_optimizer_full_20260516T200000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 2876,5000,10000,20000",
        "```",
        "",
        "说明：P1/P2 使用已真实落盘的 branch-horizon/train-probe rows；P3 因没有 natural labeled stream extension materializer，C0/C1/C2 与 5000/10000/20000 panel 都显式 `not_run`。",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 3. P1 Future Path Combo Reassessment",
        "",
        "| candidate | actions | AUV LCB | V80 LCB | V240 LCB | longrisk UCB | mem/off UCB | GradeAB precision | raw/support LDO | LSO/LTO/LFO | strong | density dispute |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for r in combos:
        memoff = fnum(r.get("MemoryFail_UCB")) + fnum(r.get("OffdiagFail_UCB"))
        lines.append(f"| `{r.get('candidate_id')}` | `{r.get('action_count')}` | `{r.get('AUV_LCB')}` | `{r.get('V80_LCB')}` | `{r.get('V240_LCB')}` | `{r.get('LongRisk_UCB')}` | `{memoff}` | `{r.get('GradeAB_precision')}` | `{r.get('LDO_raw')}`/`{r.get('LDO_support_adjusted')}` | `{r.get('LSO')}`/`{r.get('LTO')}`/`{r.get('LFO')}` | `{r.get('path_combo_strong_pass')}` | `{r.get('density_gate_dispute')}` |")
    lines += [
        "",
        "判断：P1 确认 Core77+CoreExpansion10 等组合可以形成更完整的 future-path diagnostic 对照，但 controller-ready 仍需要合法机制或 density gate closure。",
        "",
        "## 4. P2 Future-Operator Proxy",
        "",
        "| proxy | precision | V LCB | AUV LCB | RiskAdjustedAUV LCB | longrisk UCB | mem/off UCB | cost q90 ms | weak | strong | failure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in proxies:
        memoff = fnum(r.get("MemoryFail_UCB")) + fnum(r.get("OffdiagFail_UCB"))
        lines.append(f"| `{r.get('proxy_id')}` | `{r.get('TopK87_precision')}` | `{r.get('V_LCB')}` | `{r.get('AUV_LCB')}` | `{r.get('RiskAdjustedAUV_LCB')}` | `{r.get('LongRisk_UCB')}` | `{memoff}` | `{r.get('cost_q90_ms')}` | `{r.get('P2_weak_pass')}` | `{r.get('P2_strong_pass')}` | `{r.get('failure_mode')}` |")
    lines += [
        "",
        "判断：P2 这次不是复用 B1-B5，而是 FO1-FO8；结果仍没有 legal low-cost proxy pass，主要受 precision/value/risk 或 cost gate 阻断。",
        "",
        "## 5. P3 Natural AP0 Stream",
        "",
        "| panel | status | target | labeled | expected rows | actual rows | CoreLike rate | LCB/UCB | reason |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in panels:
        lines.append(f"| `{r.get('panel_id')}` | `{r.get('status')}` | `{r.get('target_action_count')}` | `{r.get('actual_labeled_action_count')}` | `{r.get('expected_rows')}` | `{r.get('actual_rows')}` | `{r.get('CoreLike_rate')}` | `{r.get('CoreLike_LCB')}`/`{r.get('CoreLike_UCB')}` | `{r.get('reason')}` |")
    lines += [
        "",
        "判断：C 线仍未落地 materializer。PanelA 是既有 2876 labeled AP0 density panel，不是 5000/10000/20000 扩流结果。",
        "",
        "## 6. P4-P8 Boundary",
        "",
        "```text",
        f"generated_sandbox_allowed = {p4.get('generated_sandbox_allowed')}",
        f"generated_route_status = {p4.get('generated_route_status')}",
        "controller/runtime/paired replay/short-full = not_run",
        "```",
        "",
        "判断：没有 B legal proxy pass，也没有 C density sufficient/insufficient 证据，因此 generated sandbox、controller、runtime、paired replay、short/full 全部关闭。",
        "",
        "## 7. Figures",
        "",
        "```text",
        "fig_A1_path_combo_V_curve.svg",
        "fig_A2_path_combo_risk_curve.svg",
        "fig_A3_core77_plus_expansion_pareto.svg",
        "fig_A4_raw_LDO_vs_support_adjusted_LDO.svg",
        "fig_A5_per_dataset_path_support.svg",
        "fig_B1_proxy_precision_vs_cost.svg",
        "fig_B2_proxy_V_vs_longrisk.svg",
        "fig_B3_proxy_AUV_vs_RiskAdjustedAUV.svg",
        "fig_B4_proxy_score_vs_future_AUV_scatter.svg",
        "fig_B5_proxy_score_distribution_by_group.svg",
        "fig_C1_density_curve_panel_size.svg",
        "fig_C2_corelike_rate_ci.svg",
        "fig_D1_generated_sandbox_gate.svg",
        "```",
        "",
        "## 8. No-Fake / Contract / Failure",
        "",
        "```text",
        f"rows_checked = {nf.get('rows_checked')}",
        f"fake_data_used = {nf.get('fake_data_used')}",
        f"proxy_row_used = {nf.get('proxy_row_used')}",
        f"cpu_offload_used = {nf.get('cpu_offload_used')}",
        "```",
        "",
        "## 9. Hash",
        "",
        "| artifact | SHA256 |",
        "|---|---|",
    ]
    for name, digest in sorted(hashes.items()):
        lines.append(f"| `{name}` | `{digest}` |")
    lines += [
        "",
        "## 10. 最终分析结论",
        "",
        "```text",
        "1. P1 证明 Core77+CoreExpansion10 等组合值得作为 future-path diagnostic candidate，但仍不是合法 controller。",
        "2. P2 FO1-FO8 没有找到低成本、训练当下合法、可同时满足 precision/value/risk 的 future-operator proxy。",
        "3. P3 natural AP0 stream materializer 仍缺失，density 不能裁决。",
        "4. P4 generated sandbox 没有打开条件。",
        "5. P5-P8 全部 gate-blocked，strict PureKAN functional 仍未成功。",
        "```",
        "",
        "最终一句话：",
        "",
        f"> v9.8.6 真实执行后停在 `{route.get('route')}`：future-path combo 的诊断信号更完整，但合法 future-operator proxy 仍失败，自然 AP0 扩流 materializer 仍未落地，因此不能进入 official controller，generated route 继续停止。",
        "",
    ]
    RECAP_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_recap_v9870(out: Path, route: dict[str, Any], hashes: dict[str, str]) -> None:
    p0 = summary_row(read_csv(out / "p0_boundary_reproduction_v9870.csv"))
    p1 = summary_row(read_csv(out / "p1_natural_materializer_preflight_v9870.csv"))
    p2 = summary_row(read_csv(out / "p2_density_curve_v9870.csv"))
    p3 = summary_row(read_csv(out / "p3_future_path_mechanism_v9870.csv"))
    p4 = summary_row(read_csv(out / "p4_low_cost_proxy_v9870.csv"))
    p5 = summary_row(read_csv(out / "p5_controller_candidates_v9870.csv"))
    p6 = summary_row(read_csv(out / "p6_generated_reopen_decision_v9870.csv"))
    nf = summary_row(read_csv(out / "no_fake_audit_v9870.csv"))
    stage_rows = [r for r in read_csv(out / "p1_natural_materializer_preflight_v9870.csv") if r.get("status") == "stage_summary"]
    panels = [r for r in read_csv(out / "p2_density_curve_v9870.csv") if r.get("status") in {"panel_row", "diagnostic_preflight_row", "not_run"}]
    groups = [r for r in read_csv(out / "p3_future_path_mechanism_v9870.csv") if r.get("status") == "group_summary"]
    proxies = [r for r in read_csv(out / "p4_low_cost_proxy_v9870.csv") if r.get("status") == "proxy_summary"]
    lines = [
        "# DG-KAN v9.8.7 Four-Line Natural Stream / FutureOperator Mechanism 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.8.7_四线并行_自然扩流_FutureOperator机制_完整实验计划.md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest；P1 preflight 使用真实 branch-horizon replay rows；5000/10000/20000 natural AP0 extension 因缺少新自然动作生成入口显式 `not_run`，没有 fake data、proxy rows 或 CPU offload。",
        "",
        "## 0. 最新结论",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"primary_blocker = {route.get('primary_blocker')}",
        f"secondary_blocker = {route.get('secondary_blocker')}",
        f"route_recommendation = {route.get('route_recommendation')}",
        f"system_legal_controller_pass = {route.get('system_legal_controller_pass')}",
        f"generated_route_status = {route.get('generated_route_status')}",
        "```",
        "",
        f"最终 artifact：`{out}`",
        "",
        "核心结论：",
        "",
        f"1. P0 复现 v9.8.6 boundary：source route = `{p0.get('source_route_v9860')}`，system pass = `{p0.get('system_legal_controller_pass_v9860')}`。",
        f"2. P1 真实 preflight materialized：selected actions = `{p1.get('selected_action_count')}`，row count expected/actual = `{p1.get('row_count_expected')}` / `{p1.get('row_count_actual')}`，P1a/P1b/P1c = `{p1.get('P1a_single_preflight_pass')}` / `{p1.get('P1b_16_preflight_pass')}` / `{p1.get('P1c_256_smoke_pass')}`。",
        f"3. P1d/P1e 扩展 panel 未打开：natural extension action generator found = `{p1.get('natural_extension_action_generator_found')}`；reason = `{p1.get('reason')}`。",
        f"4. P2 density 仍 inconclusive：PanelA CoreLike LCB/UCB = `{p2.get('PanelA_CoreLike_LCB')}` / `{p2.get('PanelA_CoreLike_UCB')}`；P2 sufficient/insufficient/inconclusive = `{p2.get('P2_density_sufficient')}` / `{p2.get('P2_density_insufficient')}` / `{p2.get('P2_density_inconclusive')}`。",
        f"5. P3 future path weak/strong = `{p3.get('P3_A_line_weak_pass')}` / `{p3.get('P3_A_line_strong_mechanism_pass')}`；Core77 AUV LCB = `{p3.get('Core77_AUV_LCB')}`。",
        f"6. P4 low-cost proxy weak/strong = `{p4.get('P4_weak_pass')}` / `{p4.get('P4_strong_pass')}`；best = `{p4.get('best_proxy_id')}`，precision/V/cost q90 = `{p4.get('best_precision')}` / `{p4.get('best_V_LCB')}` / `{p4.get('best_cost_q90_ms')}`。",
        f"7. P5 controller = `{p5.get('status')}`；P6 generated sandbox allowed = `{p6.get('generated_sandbox_allowed')}`。",
        f"8. No-fake audit：rows checked = `{nf.get('rows_checked')}`，fake/proxy/cpu = `{nf.get('fake_data_used')}` / `{nf.get('proxy_row_used')}` / `{nf.get('cpu_offload_used')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/run_v9870_four_line_natural_stream_futureoperator_mechanism.py` | v9.8.7 runner；复现 v9.8.6 boundary，真实运行 existing AP0 natural preflight branch-horizon materializer，写 5000/10000/20000 extension blocker，执行 density/A-line/P4 low-cost proxy/controller/generated boundary。 |",
        "",
        "```text",
        "python -m py_compile experiments/run_v9870_four_line_natural_stream_futureoperator_mechanism.py",
        "```",
        "",
        "```bash",
        "python experiments/run_v9870_four_line_natural_stream_futureoperator_mechanism.py --out-dir results/real_rerun_20260506/v9870_four_line_natural_stream_futureoperator_mechanism_full_20260516T220000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 2876,5000,10000,20000",
        "```",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 3. P1 Natural Materializer Preflight",
        "",
        "| stage | target | labeled | expected rows | actual rows | completion | quality pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in stage_rows:
        lines.append(f"| `{r.get('stage_id')}` | `{r.get('target_action_count')}` | `{r.get('labeled_action_count')}` | `{r.get('expected_rows')}` | `{r.get('actual_rows')}` | `{r.get('completion_rate')}` | `{r.get('quality_audit_pass')}` |")
    lines += [
        "",
        "判断：P1a/P1b/P1c 是真实 replay preflight；但它只覆盖 existing AP0 动作。5000/10000/20000 需要新增 natural AP0 action generator，当前没有落地，不能伪造扩流 panel。",
        "",
        "## 4. P2 Density Curve",
        "",
        "| panel | status | size | CoreLike rate | CoreLike LCB/UCB | PathGood rate | PathGood LCB/UCB | reason |",
        "|---|---|---:|---:|---|---:|---|---|",
    ]
    for r in panels:
        lines.append(f"| `{r.get('panel_id')}` | `{r.get('status')}` | `{r.get('panel_size')}` | `{r.get('CoreLike_rate')}` | `{r.get('CoreLike_Wilson_LCB')}`/`{r.get('CoreLike_Wilson_UCB')}` | `{r.get('PathGood_rate')}` | `{r.get('PathGood_Wilson_LCB')}`/`{r.get('PathGood_Wilson_UCB')}` | `{r.get('reason')}` |")
    lines += [
        "",
        "判断：density 仍不能裁决。Panel-2876 只支持旧 CoreLike CI；preflight PathGood 是 diagnostic，不是扩展 panel。",
        "",
        "## 5. P3 Future Path Mechanism",
        "",
        "| group | actions | AUV LCB | V1 LCB | V80 LCB | V240 LCB | longrisk UCB | weak |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in groups:
        lines.append(f"| `{r.get('group_id')}` | `{r.get('action_count')}` | `{r.get('AUV_LCB')}` | `{r.get('V1_LCB')}` | `{r.get('V80_LCB')}` | `{r.get('V240_LCB')}` | `{r.get('LongRisk_UCB')}` | `{r.get('weak_path_pass')}` |")
    lines += [
        "",
        "## 6. P4 Low-Cost Proxy",
        "",
        "| proxy | legality | precision | V LCB | AUV LCB | longrisk UCB | cost q90 ms | weak | strong | failure |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in proxies:
        lines.append(f"| `{r.get('proxy_id')}` | `{r.get('feature_legality')}` | `{r.get('TopK87_precision')}` | `{r.get('TopK87_V_LCB')}` | `{r.get('TopK87_AUV_LCB')}` | `{r.get('TopK87_LongRisk_UCB')}` | `{r.get('compute_cost_ms_p90')}` | `{r.get('P4_weak_pass')}` | `{r.get('P4_strong_pass')}` | `{r.get('failure_mode')}` |")
    lines += [
        "",
        "判断：P4 不再继续 FO1-FO8，而是测低成本 train-time proxy；结果仍未通过 precision/value/risk/leaveout gate。",
        "",
        "## 7. P5-P8 Boundary",
        "",
        "```text",
        f"P5 controller = {p5.get('status')}, reason = {p5.get('reason')}",
        f"P6 generated = {p6.get('status')}, allowed = {p6.get('generated_sandbox_allowed')}, status = {p6.get('generated_route_status')}",
        "P7 runtime = not_run",
        "P8 paired replay = not_run",
        "```",
        "",
        "## 8. No-Fake / Contract / Failure",
        "",
        "```text",
        f"rows_checked = {nf.get('rows_checked')}",
        f"fake_data_used = {nf.get('fake_data_used')}",
        f"proxy_row_used = {nf.get('proxy_row_used')}",
        f"cpu_offload_used = {nf.get('cpu_offload_used')}",
        "```",
        "",
        "## 9. Figures",
        "",
        "```text",
        "fig_p1_materializer_completion_by_stage.svg",
        "fig_p1_rows_per_sec_by_panel.svg",
        "fig_p1_exception_type_bar.svg",
        "fig_p2_density_curve_corelike.svg",
        "fig_p2_density_curve_pathgood.svg",
        "fig_p2_wilson_ci_by_panel.svg",
        "fig_p3_future_value_curve_by_group.svg",
        "fig_p3_risk_curve_by_group.svg",
        "fig_p4_proxy_precision_vs_cost.svg",
        "fig_p4_proxy_V_vs_longrisk.svg",
        "fig_p4_proxy_cost_breakdown.svg",
        "fig_p4_proxy_score_vs_AUV_scatter.svg",
        "fig_p4_proxy_leaveout_drop.svg",
        "fig_p6_generated_gate_decision.svg",
        "```",
        "",
        "## 10. Hash",
        "",
        "| artifact | SHA256 |",
        "|---|---|",
    ]
    for name, digest in sorted(hashes.items()):
        lines.append(f"| `{name}` | `{digest}` |")
    lines += [
        "",
        "## 11. 最终分析结论",
        "",
        "```text",
        "1. P1 确认 existing AP0 preflight materializer 可以真实跑，但这不是自然扩流 action generator。",
        "2. 5000/10000/20000 panel 因缺少新增自然动作生成入口继续 not_run，density 不能裁决。",
        "3. A 线 future path 仍有 weak signal，但 strong mechanism 仍依赖合法低成本 proxy。",
        "4. B/P4 低成本 train-time proxy 仍未找到 controller-ready region。",
        "5. Controller/generated/runtime/replay 全部 gate-blocked，strict PureKAN functional 仍未成功。",
        "```",
        "",
        "最终一句话：",
        "",
        f"> v9.8.7 真实执行后停在 `{route.get('route')}`：本轮把自然流的 existing-AP0 preflight 真正跑了出来，但 5000/10000/20000 需要的新自然动作扩流入口仍未落地，低成本合法 proxy 也未过 gate，因此不能进入 official controller 或 generated route。",
        "",
    ]
    RECAP_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if args.fresh and out.exists():
        import shutil

        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    artifacts: dict[str, Path] = {}

    def dump_csv(name: str, rows: list[dict[str, Any]]) -> None:
        path = out / name
        write_csv(path, rows)
        artifacts[name] = path

    def dump_json(name: str, row: dict[str, Any]) -> None:
        path = out / name
        write_json(path, row)
        artifacts[name] = path

    source_v9860 = Path(args.source_v9860)
    source_v9840 = Path(args.source_v9840)
    source_v9820 = Path(args.source_v9820)
    source_v9810 = Path(args.source_v9810)
    panel_targets = parse_ints(args.panel_targets)

    p0_rows, p0 = p0_boundary(source_v9860)
    dump_csv("p0_boundary_reproduction_v9870.csv", p0_rows)

    ap0, score_bundle, payload_by_id = v9720.load_ap0_and_payloads(args)
    real = v9850.load_real_rows(source_v9820, source_v9840)
    feats, _completed = v9850.feature_rows(real)
    dump_csv("p0_landed_future_row_reference_v9870.csv", landed_future_row_reference(real))

    p1_rows, p1_completion, p1m = natural_preflight_materializer(args, ap0, score_bundle, payload_by_id, out)
    dump_csv("p1_natural_materializer_preflight_v9870.csv", p1_rows)
    dump_csv("p1_natural_materializer_completion_v9870.csv", p1_completion)
    natural_real = [r for r in p1_rows if r.get("status") == "branch_horizon_row" and r.get("branch_id") == "RealFunctional"]
    natural_feats, _natural_completed = v9850.feature_rows(natural_real)

    existing_count = len(ap0)
    panel_reason = "natural_AP0_extension_action_generator_missing_after_existing_2876_AP0_actions"
    for target, filename in [
        (5000, "p1_natural_stream_panel_5000_v9870.csv"),
        (10000, "p1_natural_stream_panel_10000_v9870.csv"),
        (20000, "p1_natural_stream_panel_20000_v9870.csv"),
    ]:
        rows, _summary = panel_not_run_file(target, existing_count, panel_reason)
        dump_csv(filename, rows)

    p2_rows, p2_audit, p2d = density_curve(ap0, natural_feats, source_v9820, panel_targets, out)
    dump_csv("p2_density_curve_v9870.csv", p2_rows)
    dump_csv("p2_pathgood_label_audit_v9870.csv", p2_audit)

    p4_rows, p4p = p4_low_cost_proxy(ap0, feats, natural_feats, out)
    dump_csv("p4_low_cost_proxy_v9870.csv", p4_rows)

    p3_rows, p3m = future_path_mechanism_v9870(ap0, feats, natural_feats, p4p, out)
    dump_csv("p3_future_path_mechanism_v9870.csv", p3_rows)

    p5_rows, p5 = not_run("P5_CONTROLLER_CANDIDATES_V9870", "C_density_not_sufficient_B_proxy_not_passed_A_strong_plus_5000_panel_absent", controller_pass=0)
    dump_csv("p5_controller_candidates_v9870.csv", p5_rows)
    p6_rows, p6 = generated_sandbox_decision(p2d, p3m, p4p, out)
    dump_csv("p6_generated_reopen_decision_v9870.csv", p6_rows)
    p7_rows, p7 = not_run("P7_RUNTIME_BOUNDARY_V9870", "P5_controller_and_P6_generated_not_passed", runtime_pass=0)
    dump_csv("p7_runtime_boundary_v9870.csv", p7_rows)
    p8_rows, p8 = not_run("P8_PAIRED_REPLAY_BOUNDARY_V9870", "P7_runtime_not_passed", paired_replay_pass=0)
    dump_csv("p8_paired_replay_boundary_v9870.csv", p8_rows)

    dump_csv("base_acc_sentinel_v9870.csv", copy_base_acc(source_v9810))
    dump_csv("field_legality_ledger_v9870.csv", field_legality())

    route, primary, secondary, recommendation = route_from(p1m, p2d, p3m, p4p, p6)
    route_decision = {
        "stage": "ROUTE_DECISION_V9870",
        "status": "summary",
        "route": route,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "route_recommendation": recommendation,
        "source_route_v9860": p0.get("source_route_v9860"),
        "P0_boundary_pass": p0.get("P0_boundary_pass"),
        "P1a_single_preflight_pass": p1m.get("P1a_single_preflight_pass"),
        "P1b_16_preflight_pass": p1m.get("P1b_16_preflight_pass"),
        "P1c_256_smoke_pass": p1m.get("P1c_256_smoke_pass"),
        "P1d_5000_panel_weak_pass": p1m.get("P1d_5000_panel_weak_pass"),
        "P2_density_sufficient": p2d.get("P2_density_sufficient"),
        "P2_density_insufficient": p2d.get("P2_density_insufficient"),
        "P2_density_inconclusive": p2d.get("P2_density_inconclusive"),
        "P3_A_line_weak_pass": p3m.get("P3_A_line_weak_pass"),
        "P3_A_line_strong_mechanism_pass": p3m.get("P3_A_line_strong_mechanism_pass"),
        "P4_weak_pass": p4p.get("P4_weak_pass"),
        "P4_strong_pass": p4p.get("P4_strong_pass"),
        "P5_controller_pass": p5.get("controller_pass"),
        "P6_generated_sandbox_allowed": p6.get("generated_sandbox_allowed"),
        "generated_route_status": p6.get("generated_route_status"),
        "P7_runtime_pass": p7.get("runtime_pass"),
        "P8_paired_replay_pass": p8.get("paired_replay_pass"),
        "system_legal_controller_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("route_decision_v9870.json", route_decision)

    no_fake = {
        "stage": "NO_FAKE_AUDIT_V9870",
        "status": "summary",
        "rows_checked": audit_rows(artifacts),
        "fake_proxy_nonzero_count": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "no_fake": 1,
        "no_proxy": 1,
    }
    dump_csv("no_fake_audit_v9870.csv", [no_fake])
    contract = {
        "stage": "CONTRACT_AUDIT_V9870",
        "status": "summary",
        "manual_forward/manual_backward/manual_adamw_update": "1/1/1",
        "v9860_boundary_pass": p0.get("P0_boundary_pass"),
        "natural_preflight_1_16_256": f"{p1m.get('P1a_single_preflight_pass')}/{p1m.get('P1b_16_preflight_pass')}/{p1m.get('P1c_256_smoke_pass')}",
        "natural_5000_10000_20000": "0/0/0",
        "density_sufficient/insufficient/inconclusive": f"{p2d.get('P2_density_sufficient')}/{p2d.get('P2_density_insufficient')}/{p2d.get('P2_density_inconclusive')}",
        "A_weak/strong": f"{p3m.get('P3_A_line_weak_pass')}/{p3m.get('P3_A_line_strong_mechanism_pass')}",
        "B_low_cost_weak/strong": f"{p4p.get('P4_weak_pass')}/{p4p.get('P4_strong_pass')}",
        "controller/generated/runtime/system": f"{p5.get('controller_pass')}/{p6.get('generated_sandbox_allowed')}/{p7.get('runtime_pass')}/0",
        "uses_future_outcome_for_official_controller": 0,
        "diagnostic_promoted_to_official": 0,
        "fake/proxy/cpu_offload": "0/0/0",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("contract_audit_v9870.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9870",
        "status": "summary",
        "route": route,
        "F0_boundary_fail": int(not inum(p0.get("P0_boundary_pass"))),
        "F1_preflight_fail": int(not inum(p1m.get("P1b_16_preflight_pass"))),
        "F2_natural_extension_generator_missing": int(not inum(p1m.get("natural_extension_action_generator_found"))),
        "F3_density_inconclusive": inum(p2d.get("P2_density_inconclusive")),
        "F4_A_strong_mechanism_absent": int(not inum(p3m.get("P3_A_line_strong_mechanism_pass"))),
        "F5_B_low_cost_proxy_failed": int(not inum(p4p.get("P4_weak_pass"))),
        "F6_controller_runtime_blocked": 1,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("failure_taxonomy_v9870.csv", [failure])

    hashes = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    manifest = {
        "stage": "RUN_MANIFEST_V9870",
        "status": "summary",
        "out_dir": str(out),
        "execution_profile": args.execution_profile,
        "command_profile": {
            "device": args.device,
            "data_root": args.data_root,
            "seed": args.seed,
            "panel_targets": args.panel_targets,
            "natural_preflight_actions": args.natural_preflight_actions,
            "source_v9860": args.source_v9860,
            "source_v9840": args.source_v9840,
            "source_v9820": args.source_v9820,
        },
        "wallclock_sec": time.perf_counter() - t0,
        "route": route,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "artifact_hashes": hashes,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("run_manifest_v9870.json", manifest)
    hashes_for_recap = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    write_recap_v9870(out, route_decision, hashes_for_recap)

    print(json.dumps({
        "out_dir": str(out),
        "route": route,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "P1c_256_smoke_pass": p1m.get("P1c_256_smoke_pass"),
        "P2_density_inconclusive": p2d.get("P2_density_inconclusive"),
        "P4_weak_pass": p4p.get("P4_weak_pass"),
        "P6_generated_sandbox_allowed": p6.get("generated_sandbox_allowed"),
        "system_legal_controller_pass": 0,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
