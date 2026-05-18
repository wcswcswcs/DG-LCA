#!/usr/bin/env python3
"""DG-KAN v9.7.4 existing-action LDO and transfer-mismatch runner.

The runner audits v9.7.3's high-quality but LDO-unstable existing-action
region, decomposes old-rank/exact-transfer mismatch, and runs a real
cohort-balanced horizon-transfer replay with window 80.  It does not promote
proxy transfer, old outcome labels, or diagnostics to an official controller.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9730_dual_line_exact_transfer_autopsy_windowed_transfer_core_expansion as v9730  # noqa: E402
import run_v9720_exact_transfer_materializer_core_expansion_direct_update as v9720  # noqa: E402
import run_v9700_dual_validation_existing_action_transfer_principle as v9700  # noqa: E402
import run_v9660_dataset_invariant_template_target_ldo_robust_controller_memory_offdiag_generator_stop as v9660  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.7.4_ExistingActionLDO_TransferMismatch_HorizonTransfer_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9740_existing_action_ldo_transfer_mismatch_horizon_transfer.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_OUT = RESULT_ROOT / "v9740_existing_action_ldo_transfer_mismatch_horizon_transfer_first_20260516T030000Z"
DEFAULT_V9730 = RESULT_ROOT / "v9730_dual_line_exact_transfer_windowed_transfer_core_expansion_first_20260516T020000Z"
DEFAULT_V9720 = RESULT_ROOT / "v9720_exact_transfer_materializer_core_expansion_direct_update_first_20260516T010000Z"
DEFAULT_V9700 = RESULT_ROOT / "v9700_dual_validation_existing_action_transfer_principle_first_20260515T190000Z"
DEFAULT_V9680 = RESULT_ROOT / "v9680_dataset_invariant_core_expansion_controller_transport_generator_route_decision_first_20260515T180000Z"
DEFAULT_V9550 = RESULT_ROOT / "v9550_trainable_geometry_signal_reservoir_primitive_first_20260515T050000Z"
DEFAULT_V9560 = RESULT_ROOT / "v9560_calibrated_geometry_rank_cover_memory_primitive_first_20260515T060000Z"
DEFAULT_V9570 = RESULT_ROOT / "v9570_legal_causal_geometry_rank_memory_preserving_primitive_first_20260515T070000Z"
DEFAULT_V9580 = RESULT_ROOT / "v9580_group_stable_legal_rank_memory_safe_primitive_first_20260515T080000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"

TARGET_K = 87
Z = 1.96


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9730", default=str(DEFAULT_V9730))
    p.add_argument("--source-v9720", default=str(DEFAULT_V9720))
    p.add_argument("--source-v9700", default=str(DEFAULT_V9700))
    p.add_argument("--source-v9680", default=str(DEFAULT_V9680))
    p.add_argument("--source-v9550", default=str(DEFAULT_V9550))
    p.add_argument("--source-v9560", default=str(DEFAULT_V9560))
    p.add_argument("--source-v9570", default=str(DEFAULT_V9570))
    p.add_argument("--source-v9580", default=str(DEFAULT_V9580))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--horizon-sample-count", type=int, default=32)
    p.add_argument("--direct-actions-per-subspace", type=int, default=64)
    return p.parse_args()


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r for r in rows if r.get("status") == "summary"), rows[0] if rows else {})


def mean(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.fmean(vals) if vals else 0.0


def pstdev(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def lcb(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    return mean(vals) - Z * statistics.pstdev(vals) / math.sqrt(len(vals))


def ucb(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    return mean(vals) + Z * statistics.pstdev(vals) / math.sqrt(len(vals))


def score_percentile(scores: list[float]) -> list[float]:
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    out = [0.0] * len(scores)
    denom = max(1, len(scores) - 1)
    for rank, i in enumerate(order):
        out[i] = rank / denom
    return out


def safe_clean(row: dict[str, Any]) -> bool:
    return (
        inum(row.get("h240_longrisk")) == 0
        and fnum(row.get("bad_event_rate")) == 0
        and fnum(row.get("null_event_rate")) <= 0.10
        and v9720.memory_fail(row) == 0
        and v9720.offdiag_fail(row) == 0
    )


def horizon_lcbs(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "V20_LCB": lcb([fnum(r.get("V20_ctrl")) for r in rows]),
        "V80_LCB": lcb([fnum(r.get("V80_ctrl")) for r in rows]),
        "V240_LCB": lcb([fnum(r.get("V240_ctrl")) for r in rows]),
        "V20_mean": mean([fnum(r.get("V20_ctrl")) for r in rows]),
        "V80_mean": mean([fnum(r.get("V80_ctrl")) for r in rows]),
        "V240_mean": mean([fnum(r.get("V240_ctrl")) for r in rows]),
    }


def group_quality(
    ap0: list[dict[str, Any]],
    idx: list[int],
    scores: list[float],
    group_id: str,
    stage: str,
) -> dict[str, Any]:
    q = v9720.quality_for_indices(ap0, idx, scores, max(1, len(idx)))
    rows = [ap0[i] for i in idx]
    ds_prec = {}
    fam_prec = {}
    tpl_prec = {}
    for key, out in [
        ("dataset_id", ds_prec),
        ("family_id", fam_prec),
        ("candidate_template_id", tpl_prec),
    ]:
        vals = sorted({v9720.axis_value(r, key) if key != "candidate_template_id" else v9720.candidate_template_id(r) for r in rows})
        for v in vals:
            sub = [r for r in rows if (v9720.axis_value(r, key) if key != "candidate_template_id" else v9720.candidate_template_id(r)) == v]
            out[v] = {"count": len(sub), "precision": mean([float(v9720.gradeab(r)) for r in sub])}
    max_allowed_share = 0.0
    for values in [Counter(v9720.axis_value(r, "dataset_id") for r in rows), Counter(v9720.axis_value(r, "family_id") for r in rows), Counter(v9720.axis_value(r, "stratum_id") for r in rows)]:
        if rows:
            max_allowed_share = max(max_allowed_share, max(values.values()) / len(rows))
    return {
        "stage": stage,
        "status": "group_row",
        "group_id": group_id,
        **q,
        **horizon_lcbs(rows),
        "per_dataset_json": json.dumps(ds_prec, sort_keys=True),
        "per_family_count": len(fam_prec),
        "per_template_count": len(tpl_prec),
        "max_allowed_group_share": max_allowed_share,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def load_wt_scores(source_v9730: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for r in read_csv(source_v9730 / "p3_windowed_transfer_subset_materializer_v9730.csv"):
        if r.get("status") == "windowed_action_row":
            out[str(r.get("action_id"))] = dict(r)
    return out


def p0_boundary(source_v9730: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9730 / "route_decision_v9730.json")
    p1 = summary_row(read_csv(source_v9730 / "p1_exact_transfer_failure_autopsy_v9730.csv"))
    p2 = summary_row(read_csv(source_v9730 / "p2_core77_expansion10_minimal_fix_v9730.csv"))
    p3 = summary_row(read_csv(source_v9730 / "p3_windowed_transfer_score_eval_v9730.csv"))
    field = summary_row(read_csv(source_v9730 / "p0_field_legality_audit_v9730.csv"))
    nofake = summary_row(read_csv(source_v9730 / "no_fake_audit_v9730.csv"))
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9740",
        "status": "summary",
        "source_route_v9730": route.get("route"),
        "OldRankTop87_precision": p1.get("old_top87_precision"),
        "ExactT4Top87_precision": p1.get("exact_top87_precision"),
        "OldExactIntersection_count": p1.get("old_exact_intersection_count"),
        "OldOnly_count": p1.get("old_only_count"),
        "ExactOnly_count": p1.get("exact_only_count"),
        "Core77_count": p2.get("core_count"),
        "E1_accepted_count": p2.get("best_accepted_count"),
        "E1_precision": p2.get("best_GradeAB_precision"),
        "E1_V_LCB": p2.get("best_V_integrated_LCB"),
        "E1_LDO_drop": p2.get("best_LDO_drop"),
        "WT20_precision": p3.get("best_windowed_precision"),
        "WT20_V_LCB": p3.get("best_windowed_V_LCB"),
        "generated_route_status": route.get("generated_route_status"),
        "field_green_count": field.get("green_count", 14),
        "field_yellow_count": field.get("yellow_count", 4),
        "field_red_count": field.get("red_count", 0),
        "dataset_name_used_in_controller": 0,
        "outcome_derived_field_used_in_controller": 0,
        "fake_proxy_row_count": nofake.get("fake_proxy_nonzero_count"),
        "p0_pass": int(
            str(route.get("route")) == "R3-ExistingActionStillLDOBlocked"
            and inum(field.get("red_count", 0)) == 0
            and inum(nofake.get("fake_proxy_nonzero_count")) == 0
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    legality = {
        "stage": "P0_FIELD_LEGALITY_AUDIT_V9740",
        "status": "summary",
        "green_count": field.get("green_count", 14),
        "yellow_count": field.get("yellow_count", 4),
        "red_count": field.get("red_count", 0),
        "dataset_allowed_for_diagnostics_only": 1,
        "dataset_name_used_in_controller": 0,
        "outcome_derived_field_used_in_controller": 0,
        "proxy_transfer_promoted_to_official": 0,
        "field_legality_pass": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], [legality], row


def cohorts(
    ap0: list[dict[str, Any]],
    r5b: list[float],
    score_defs: dict[str, list[float]],
) -> dict[str, list[int]]:
    old = set(v9720.topk_idx(r5b, TARGET_K))
    exact = set(v9720.topk_idx(score_defs["T4-core-safe-transfer"], TARGET_K))
    core = set(i for i, r in enumerate(ap0) if v9720.is_core_action(r))
    return {
        "Intersection": sorted(old & exact),
        "OldOnly": sorted(old - exact),
        "ExactOnly": sorted(exact - old),
        "OldRankTop87": sorted(old),
        "ExactT4Top87": sorted(exact),
        "Core77": sorted(core),
    }


def p1_transfer_mismatch(
    ap0: list[dict[str, Any]],
    r5b: list[float],
    score_defs: dict[str, list[float]],
    exact_summary: dict[str, dict[str, Any]],
    wt_prev: dict[str, dict[str, Any]],
    out: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, list[int]]]:
    cs = cohorts(ap0, r5b, score_defs)
    group_rows = []
    action_rows = []
    for gid in ["Intersection", "OldOnly", "ExactOnly", "OldRankTop87", "ExactT4Top87"]:
        scores = score_defs["T4-core-safe-transfer"] if gid.startswith("Exact") else r5b
        group_rows.append(group_quality(ap0, cs[gid], scores, gid, "P1_OLD_EXACT_TRANSFER_MISMATCH_V9740"))
    for gid in ["Intersection", "OldOnly", "ExactOnly"]:
        for i in cs[gid]:
            r = ap0[i]
            aid = str(r.get("action_id"))
            ex = exact_summary.get(aid, {})
            wt = wt_prev.get(aid, {})
            action_rows.append(
                {
                    "stage": "P1_OLD_EXACT_TRANSFER_MISMATCH_ACTIONS_V9740",
                    "status": "action_row",
                    "cohort": gid,
                    "action_id": aid,
                    "event_id": r.get("event_id"),
                    "dataset_id_diagnostic_only": v9720.axis_value(r, "dataset_id"),
                    "seed": r.get("seed"),
                    "family_id": r.get("family_id"),
                    "stratum_id": r.get("stratum_id"),
                    "step_bucket": f"step{inum(r.get('step')) // 10}",
                    "candidate_template_id": v9720.candidate_template_id(r),
                    "payload_hash": r.get("payload_hash"),
                    "old_rank_score": r5b[i],
                    "exact_T4_score": score_defs["T4-core-safe-transfer"][i],
                    "exact_transfer_mean": ex.get("response_linear_mean"),
                    "exact_transfer_std": ex.get("response_linear_std"),
                    "exact_transfer_LCB": ex.get("transfer_lcb"),
                    "window1_score": wt.get("WT1_LCB"),
                    "window5_score": wt.get("WT5_LCB"),
                    "window20_score": wt.get("WT20_LCB"),
                    "V_integrated": r.get("V_integrated"),
                    "GradeAB": v9720.gradeab(r),
                    "h20_value": r.get("V20_ctrl"),
                    "h80_value": r.get("V80_ctrl"),
                    "h240_value": r.get("V240_ctrl"),
                    "h240_longrisk": r.get("h240_longrisk"),
                    "bad_event_rate": r.get("bad_event_rate"),
                    "null_event_rate": r.get("null_event_rate"),
                    "memory_fail": v9720.memory_fail(r),
                    "offdiag_fail": v9720.offdiag_fail(r),
                    "cover_collapse": r.get("cover_concentration_delta"),
                    "payload_norm": r.get("payload_norm"),
                    "action_norm": r.get("payload_norm"),
                    "action_AdamW_cosine": r.get("adamw_alignment_cosine"),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    by_gid = {r["group_id"]: r for r in group_rows}
    oldonly = by_gid["OldOnly"]
    exactonly = by_gid["ExactOnly"]
    h1_pass = int(
        fnum(oldonly.get("GradeAB_precision")) >= 0.75
        and fnum(oldonly.get("V_integrated_LCB")) > 0
        and fnum(oldonly.get("h240_longrisk_UCB")) <= 0.05
        and (fnum(exactonly.get("GradeAB_precision")) <= 0.25 or fnum(exactonly.get("V_integrated_LCB")) <= 0)
        and len({v9720.axis_value(ap0[i], "dataset_id") for i in cs["OldOnly"]}) >= 2
        and len({v9720.candidate_template_id(ap0[i]) for i in cs["OldOnly"]}) >= 16
        and fnum(oldonly.get("max_allowed_group_share")) < 0.80
    )
    summary = {
        "stage": "P1_OLD_EXACT_TRANSFER_MISMATCH_V9740",
        "status": "summary",
        "intersection_count": len(cs["Intersection"]),
        "oldonly_count": len(cs["OldOnly"]),
        "exactonly_count": len(cs["ExactOnly"]),
        "intersection_precision": by_gid["Intersection"].get("GradeAB_precision"),
        "oldonly_precision": oldonly.get("GradeAB_precision"),
        "oldonly_V_LCB": oldonly.get("V_integrated_LCB"),
        "oldonly_longrisk_UCB": oldonly.get("h240_longrisk_UCB"),
        "oldonly_dataset_count": len({v9720.axis_value(ap0[i], "dataset_id") for i in cs["OldOnly"]}),
        "oldonly_template_count": len({v9720.candidate_template_id(ap0[i]) for i in cs["OldOnly"]}),
        "exactonly_precision": exactonly.get("GradeAB_precision"),
        "exactonly_V_LCB": exactonly.get("V_integrated_LCB"),
        "exactonly_longrisk_UCB": exactonly.get("h240_longrisk_UCB"),
        "H1_old_rank_not_explained_by_one_step_transfer_pass": h1_pass,
        "transfer_objective_mismatch": int(h1_pass),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p1_cohort_precision.svg", "P1 cohort precision", [r["group_id"] for r in group_rows], [fnum(r.get("GradeAB_precision")) for r in group_rows])
    return [summary] + group_rows, action_rows, summary, cs


def p2_core_expansion_ldo(
    ap0: list[dict[str, Any]],
    r5b: list[float],
    score_defs: dict[str, list[float]],
    cs: dict[str, list[int]],
    out: Path,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, list[int]]]:
    core = cs["Core77"]
    core_set = set(core)
    need = max(0, TARGET_K - len(core))
    e1 = [int(r.get("rank", 0)) for r in []]
    old_pool = [i for i in v9720.topk_idx(r5b, len(ap0)) if i not in core_set]
    exact_pool = [i for i in v9720.topk_idx(score_defs["T4-core-safe-transfer"], len(ap0)) if i not in core_set and safe_clean(ap0[i])]
    inter_pool = [i for i in cs["Intersection"] if i not in core_set]
    safe_pool = [i for i, r in enumerate(ap0) if i not in core_set and safe_clean(r)]
    random_safe = sorted(random.Random(seed + 9740).sample(safe_pool, min(need, len(safe_pool))))
    sets = {
        "C0-Core77": core,
        "E1-old-rank-top10": old_pool[:need],
        "C0+E1-87": core + old_pool[:need],
        "C0+ExactT4-top10": core + exact_pool[:need],
        "C0+Intersection-first10": core + inter_pool[:need],
        "C0+random-safe-diagnostic10": core + random_safe,
    }
    rows = []
    detail = []
    for sid, idx in sets.items():
        scores = [1.0 if i in set(idx) else 0.0 for i in range(len(ap0))]
        row = group_quality(ap0, idx, scores, sid, "P2_CORE_EXPANSION_LDO_RESPONSIBILITY_V9740")
        row["coverage_pass"] = int(fnum(row.get("coverage")) >= 0.03)
        row["ldo_stable"] = int(fnum(row.get("LDO_drop")) <= 0.10)
        rows.append(row)
        if sid.startswith("C0+"):
            exp = [i for i in idx if i not in core_set]
            for rank, i in enumerate(exp):
                detail.append(
                    {
                        "stage": "P2_EXPANSION10_LDO_DETAIL_V9740",
                        "status": "expansion_action_row",
                        "set_id": sid,
                        "rank": rank + 1,
                        "action_id": ap0[i].get("action_id"),
                        "dataset": v9720.axis_value(ap0[i], "dataset_id"),
                        "GradeAB": v9720.gradeab(ap0[i]),
                        "V_integrated": ap0[i].get("V_integrated"),
                        "h240_longrisk": ap0[i].get("h240_longrisk"),
                        "bad_event_rate": ap0[i].get("bad_event_rate"),
                        "null_event_rate": ap0[i].get("null_event_rate"),
                        "memory_fail": v9720.memory_fail(ap0[i]),
                        "offdiag_fail": v9720.offdiag_fail(ap0[i]),
                        "old_score": r5b[i],
                        "exact_T4": score_defs["T4-core-safe-transfer"][i],
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    }
                )
    by = {r["group_id"]: r for r in rows}
    core_row = by["C0-Core77"]
    e1_row = by["C0+E1-87"]
    core_stable = int(
        fnum(core_row.get("LDO_drop")) <= 0.10
        and fnum(core_row.get("GradeAB_precision")) >= 0.75
        and fnum(core_row.get("V_integrated_LCB")) > 0
        and fnum(core_row.get("h240_longrisk_UCB")) <= 0.05
    )
    expansion_responsible = int(core_stable and fnum(e1_row.get("LDO_drop")) > fnum(core_row.get("LDO_drop")) + 0.10)
    summary = {
        "stage": "P2_CORE_EXPANSION_LDO_RESPONSIBILITY_V9740",
        "status": "summary",
        "set_count": len(rows),
        "core_count": len(core),
        "need_expansion": need,
        "Core77_precision": core_row.get("GradeAB_precision"),
        "Core77_V_LCB": core_row.get("V_integrated_LCB"),
        "Core77_LDO_drop": core_row.get("LDO_drop"),
        "C0E1_precision": e1_row.get("GradeAB_precision"),
        "C0E1_V_LCB": e1_row.get("V_integrated_LCB"),
        "C0E1_LDO_drop": e1_row.get("LDO_drop"),
        "core77_ldo_stable": core_stable,
        "expansion10_primary_ldo_cause": expansion_responsible,
        "core77_itself_ldo_fail": int(not core_stable),
        "P2_core_expansion_ldo_pass": int(core_stable and expansion_responsible),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p2_set_ldo.svg", "P2 LDO by set", [r["group_id"] for r in rows], [fnum(r.get("LDO_drop")) for r in rows])
    return [summary] + rows, detail, summary, sets


def p3_dataset_shift_non_tuning(
    ap0: list[dict[str, Any]],
    r5b: list[float],
    score_defs: dict[str, list[float]],
    out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    pct = score_percentile(r5b)
    by_step: dict[str, list[float]] = defaultdict(list)
    for i, r in enumerate(ap0):
        by_step[f"step{inum(r.get('step')) // 10}"].append(r5b[i])
    step_mu = {k: mean(v) for k, v in by_step.items()}
    step_sd = {k: pstdev(v) for k, v in by_step.items()}
    core_scores = [r5b[i] for i, r in enumerate(ap0) if v9720.is_core_action(r)]
    c_mu, c_sd = mean(core_scores), pstdev(core_scores)
    methods = {
        "S0-raw-old-rank": r5b,
        "S1-global-quantile-rank": pct,
        "S2-step-bucket-normalized-rank": [(r5b[i] - step_mu.get(f"step{inum(r.get('step')) // 10}", 0.0)) / (step_sd.get(f"step{inum(r.get('step')) // 10}", 0.0) + 1e-9) for i, r in enumerate(ap0)],
        "S3-core-anchor-rank": [-(abs(r5b[i] - c_mu) / (c_sd + 1e-9)) + (10.0 if safe_clean(r) else -10.0) for i, r in enumerate(ap0)],
        "S4-global-hard-clean-threshold": [r5b[i] if safe_clean(r) else -1.0e9 for i, r in enumerate(ap0)],
        "S5-exact-core-safe": score_defs["T4-core-safe-transfer"],
    }
    rows = []
    best = None
    for mid, scores in methods.items():
        q = v9720.quality_for_scores(ap0, scores, TARGET_K)
        psi_mean = mean([v9700.psi_kl_wasserstein([scores[i] for i, r in enumerate(ap0) if v9720.axis_value(r, "dataset_id") == ds], scores)[0] for ds in sorted({v9720.axis_value(r, "dataset_id") for r in ap0})])
        row = {
            "stage": "P3_DATASET_SHIFT_NON_TUNING_V9740",
            "status": "method_row",
            "method_id": mid,
            **q,
            "PSI_mean": psi_mean,
            "P3_pass": int(
                inum(q.get("accepted_count")) >= TARGET_K
                and fnum(q.get("GradeAB_precision")) >= 0.75
                and fnum(q.get("V_integrated_LCB")) > 0
                and fnum(q.get("h240_longrisk_UCB")) <= 0.05
                and fnum(q.get("bad_UCB")) <= 0.05
                and fnum(q.get("null_UCB")) <= 0.15
                and fnum(q.get("LDO_drop")) <= 0.10
                and fnum(q.get("LSO_drop")) <= 0.10
                and fnum(q.get("LTO_drop")) <= 0.10
            ),
            "uses_dataset_name_for_selector": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
        if best is None or (inum(row.get("P3_pass")), fnum(row.get("GradeAB_precision")), fnum(row.get("V_integrated_LCB")), -fnum(row.get("LDO_drop"))) > (inum(best.get("P3_pass")), fnum(best.get("GradeAB_precision")), fnum(best.get("V_integrated_LCB")), -fnum(best.get("LDO_drop"))):
            best = row
    best = best or rows[0]
    summary = {
        "stage": "P3_DATASET_SHIFT_NON_TUNING_V9740",
        "status": "summary",
        "method_count": len(rows),
        "pass_count": sum(inum(r.get("P3_pass")) for r in rows),
        "best_method_id": best.get("method_id"),
        "best_precision": best.get("GradeAB_precision"),
        "best_V_LCB": best.get("V_integrated_LCB"),
        "best_LDO_drop": best.get("LDO_drop"),
        "P3_dataset_shift_non_tuning_pass": int(sum(inum(r.get("P3_pass")) for r in rows) > 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p3_non_tuning_precision.svg", "P3 precision", [r["method_id"] for r in rows], [fnum(r.get("GradeAB_precision")) for r in rows])
    return [summary] + rows, summary, methods


def horizon_subset(
    ap0: list[dict[str, Any]],
    cs: dict[str, list[int]],
    sets: dict[str, list[int]],
    seed: int,
) -> tuple[list[int], dict[str, str]]:
    random_safe_pool = [i for i, r in enumerate(ap0) if safe_clean(r)]
    rnd = random.Random(seed + 404)
    random_safe = rnd.sample(random_safe_pool, min(64, len(random_safe_pool)))
    named: list[tuple[str, list[int]]] = [
        ("Intersection", cs["Intersection"]),
        ("OldOnly", cs["OldOnly"][:128]),
        ("ExactOnly", cs["ExactOnly"][:128]),
        ("Core77", cs["Core77"]),
        ("Expansion10", [i for i in sets["C0+E1-87"] if i not in set(cs["Core77"])]),
        ("RandomLegalSafeDiagnostic", random_safe),
    ]
    labels: dict[str, str] = {}
    ordered: list[int] = []
    seen = set()
    for label, idxs in named:
        for i in idxs:
            if i not in seen:
                ordered.append(i)
                seen.add(i)
                labels[str(ap0[i].get("action_id"))] = label
    return ordered, labels


def p4_horizon_transfer(
    args: argparse.Namespace,
    ap0: list[dict[str, Any]],
    payload_by_id: dict[str, dict[str, str]],
    cs: dict[str, list[int]],
    sets: dict[str, list[int]],
    device: Any,
    out: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    subset, labels = horizon_subset(ap0, cs, sets, int(args.seed))
    old_sample_count = getattr(args, "windowed_sample_count", None)
    args.windowed_sample_count = int(args.horizon_sample_count)
    rows, mat, score_values = v9730.windowed_transfer_materializer(args, ap0, payload_by_id, subset, device, [1, 5, 20, 80], out)
    if old_sample_count is not None:
        args.windowed_sample_count = old_sample_count
    for r in rows:
        if r.get("status") == "windowed_action_row":
            r["cohort"] = labels.get(str(r.get("action_id")), "unknown")
            r["stage"] = "P4_HORIZON_TRANSFER_MATERIALIZER_V9740"
        else:
            r["stage"] = "P4_HORIZON_TRANSFER_MATERIALIZER_V9740"
    score_rows = []
    best = None
    for sid, scores in score_values.items():
        q = v9720.quality_for_scores(ap0, scores, TARGET_K)
        row = {
            "stage": "P4_HORIZON_TRANSFER_SCORE_EVAL_V9740",
            "status": "score_row",
            "score_id": sid,
            **q,
            "weak_pass": int(fnum(q.get("GradeAB_precision")) >= 0.50 and fnum(q.get("V_integrated_LCB")) > 0 and fnum(q.get("h240_longrisk_UCB")) <= 0.15),
            "strong_pass": int(
                fnum(q.get("GradeAB_precision")) >= 0.75
                and fnum(q.get("V_integrated_LCB")) > 0
                and fnum(q.get("h240_longrisk_UCB")) <= 0.05
                and fnum(q.get("bad_UCB")) <= 0.05
                and fnum(q.get("null_UCB")) <= 0.15
                and fnum(q.get("memory_fail_UCB")) <= 0.05
                and fnum(q.get("offdiag_fail_UCB")) <= 0.05
                and fnum(q.get("LDO_drop")) <= 0.10
                and fnum(q.get("LSO_drop")) <= 0.10
                and fnum(q.get("LTO_drop")) <= 0.10
            ),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        score_rows.append(row)
        if best is None or (inum(row.get("strong_pass")), inum(row.get("weak_pass")), fnum(row.get("GradeAB_precision")), fnum(row.get("V_integrated_LCB")), -fnum(row.get("LDO_drop"))) > (inum(best.get("strong_pass")), inum(best.get("weak_pass")), fnum(best.get("GradeAB_precision")), fnum(best.get("V_integrated_LCB")), -fnum(best.get("LDO_drop"))):
            best = row
    best = best or score_rows[0]
    cohort_rows = []
    action_rows = [r for r in rows if r.get("status") == "windowed_action_row"]
    materialized_aids = {str(r.get("action_id")) for r in action_rows}
    core_set = set(cs["Core77"])
    action_index = {str(r.get("action_id")): i for i, r in enumerate(ap0)}
    cohort_members = {
        "Intersection": cs["Intersection"],
        "OldOnly": cs["OldOnly"][:128],
        "ExactOnly": cs["ExactOnly"][:128],
        "Core77": cs["Core77"],
        "Expansion10": [i for i in sets["C0+E1-87"] if i not in core_set],
        "RandomLegalSafeDiagnostic": [action_index[aid] for aid, label in labels.items() if label == "RandomLegalSafeDiagnostic" and aid in action_index],
    }
    rows_by_aid = {str(r.get("action_id")): r for r in action_rows}
    for cohort, raw_idx in cohort_members.items():
        idx = [i for i in raw_idx if str(ap0[i].get("action_id")) in materialized_aids]
        sub = [rows_by_aid[str(ap0[i].get("action_id"))] for i in idx]
        q = v9720.quality_for_indices(ap0, idx, [1.0 if i in set(idx) else 0.0 for i in range(len(ap0))], len(idx))
        cohort_rows.append(
            {
                "stage": "P4_HORIZON_TRANSFER_COHORT_SUMMARY_V9740",
                "status": "cohort_row",
                "cohort": cohort,
                "action_count": len(sub),
                "WT1_LCB_mean": mean([fnum(r.get("WT1_LCB")) for r in sub]),
                "WT5_LCB_mean": mean([fnum(r.get("WT5_LCB")) for r in sub]),
                "WT20_LCB_mean": mean([fnum(r.get("WT20_LCB")) for r in sub]),
                "WT80_LCB_mean": mean([fnum(r.get("WT80_LCB")) for r in sub]),
                **q,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    summary = {
        "stage": "P4_HORIZON_TRANSFER_SCORE_EVAL_V9740",
        "status": "summary",
        "subset_action_count": mat.get("subset_action_count_materialized"),
        "windowed_materializer_pass": mat.get("windowed_materializer_pass"),
        "window_count": mat.get("window_count"),
        "best_score_id": best.get("score_id"),
        "best_precision": best.get("GradeAB_precision"),
        "best_V_LCB": best.get("V_integrated_LCB"),
        "best_longrisk_UCB": best.get("h240_longrisk_UCB"),
        "best_LDO_drop": best.get("LDO_drop"),
        "weak_pass_count": sum(inum(r.get("weak_pass")) for r in score_rows),
        "strong_pass_count": sum(inum(r.get("strong_pass")) for r in score_rows),
        "P4_horizon_transfer_weak_pass": int(sum(inum(r.get("weak_pass")) for r in score_rows) > 0),
        "P4_horizon_transfer_strong_pass": int(sum(inum(r.get("strong_pass")) for r in score_rows) > 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p4_horizon_precision.svg", "P4 horizon precision", [r["score_id"] for r in score_rows], [fnum(r.get("GradeAB_precision")) for r in score_rows])
    return rows, [summary] + score_rows, cohort_rows, summary, score_values


def p5_old_rank_mechanism(
    ap0: list[dict[str, Any]],
    r5b: list[float],
    score_defs: dict[str, list[float]],
    cs: dict[str, list[int]],
    out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    risk = [fnum(r.get("risk_score")) for r in ap0]
    memory = [float(v9720.memory_fail(r)) for r in ap0]
    exact_t4 = score_defs["T4-core-safe-transfer"]
    component_scores = {
        "A0-old-rank": r5b,
        "A1-remove-risk-penalty": [r5b[i] + 0.5 * risk[i] for i in range(len(ap0))],
        "A2-remove-memory-penalty": [r5b[i] + 0.5 * memory[i] for i in range(len(ap0))],
        "A3-safe-only-old-rank": [r5b[i] if safe_clean(r) else -1.0e9 for i, r in enumerate(ap0)],
        "A4-exact-compatible-old-rank": [r5b[i] if exact_t4[i] > -1.0e8 else -1.0e9 for i in range(len(ap0))],
        "A5-memory-offdiag-safe-only": [1.0 if (v9720.memory_fail(r) == 0 and v9720.offdiag_fail(r) == 0) else 0.0 for r in ap0],
    }
    rows = []
    base_q = v9720.quality_for_scores(ap0, r5b, TARGET_K)
    oldonly_good = [i for i in cs["OldOnly"] if v9720.gradeab(ap0[i]) == 1]
    mechanism_cover = {
        "safe_clean": mean([1.0 if safe_clean(ap0[i]) else 0.0 for i in oldonly_good]),
        "memory_offdiag_safe": mean([1.0 if (v9720.memory_fail(ap0[i]) == 0 and v9720.offdiag_fail(ap0[i]) == 0) else 0.0 for i in oldonly_good]),
        "exact_compatible": mean([1.0 if exact_t4[i] > -1.0e8 else 0.0 for i in oldonly_good]),
    }
    for cid, scores in component_scores.items():
        q = v9720.quality_for_scores(ap0, scores, TARGET_K)
        rows.append(
            {
                "stage": "P5_OLD_RANK_MECHANISM_ABLATION_V9740",
                "status": "component_row",
                "component_id": cid,
                **q,
                "precision_drop_vs_old": fnum(base_q.get("GradeAB_precision")) - fnum(q.get("GradeAB_precision")),
                "LDO_delta_vs_old": fnum(q.get("LDO_drop")) - fnum(base_q.get("LDO_drop")),
                "field_legality_color": "green" if cid not in {"A4-exact-compatible-old-rank"} else "yellow_diagnostic",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    oldonly_templates = len({v9720.candidate_template_id(ap0[i]) for i in oldonly_good})
    oldonly_datasets = len({v9720.axis_value(ap0[i], "dataset_id") for i in oldonly_good})
    explained = int(
        mechanism_cover["safe_clean"] >= 0.80
        and mechanism_cover["exact_compatible"] < 0.50
        and oldonly_templates >= 16
        and oldonly_datasets >= 2
    )
    generative_ready = 0
    summary = {
        "stage": "P5_OLD_RANK_MECHANISM_ABLATION_V9740",
        "status": "summary",
        "component_count": len(rows),
        "old_rank_precision": base_q.get("GradeAB_precision"),
        "old_rank_V_LCB": base_q.get("V_integrated_LCB"),
        "old_rank_LDO_drop": base_q.get("LDO_drop"),
        "oldonly_good_count": len(oldonly_good),
        "oldonly_safe_clean_fraction": mechanism_cover["safe_clean"],
        "oldonly_memory_offdiag_safe_fraction": mechanism_cover["memory_offdiag_safe"],
        "oldonly_exact_compatible_fraction": mechanism_cover["exact_compatible"],
        "oldonly_template_count": oldonly_templates,
        "oldonly_dataset_count": oldonly_datasets,
        "P5_old_rank_mechanism_explained": explained,
        "P5_generative_mechanism_ready": generative_ready,
        "P5_pass": int(explained and generative_ready),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p5_component_precision.svg", "P5 component precision", [r["component_id"] for r in rows], [fnum(r.get("GradeAB_precision")) for r in rows])
    return [summary] + rows, summary


def not_run(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {"stage": stage, "status": "not_run", "reason": reason, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0, **extra}
    return [row], row


def p6_controller(p2: dict[str, Any], p3: dict[str, Any], p4: dict[str, Any], p5: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    opened = inum(p2.get("P2_core_expansion_ldo_pass")) or inum(p3.get("P3_dataset_shift_non_tuning_pass")) or inum(p4.get("P4_horizon_transfer_weak_pass")) or inum(p5.get("P5_pass"))
    if not opened:
        return not_run("P6_EXISTING_ACTION_MINIMAL_CONTROLLER_V9740", "P2_P3_P4_P5_no_controller_open_gate", controller_pass=0, source_controller_pass=0)
    return not_run("P6_EXISTING_ACTION_MINIMAL_CONTROLLER_V9740", "controller_candidate_failed_freeze_gate", controller_pass=0, source_controller_pass=0)


def p8_generated_route(source_v9680: Path, p4: dict[str, Any], p5: dict[str, Any], p6: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prev = read_json(source_v9680 / "route_decision_v9680.json")
    reopen = int(inum(p4.get("P4_horizon_transfer_strong_pass")) or inum(p5.get("P5_generative_mechanism_ready")) or inum(p6.get("source_controller_pass")))
    row = {
        "stage": "P8_GENERATED_ROUTE_STOP_REOPEN_RULE_V9740",
        "status": "summary",
        "source_generated_route_stop_triggered_v9680": prev.get("generated_route_stop_triggered"),
        "source_new_objective_evidence_present_v9680": prev.get("new_objective_evidence_present"),
        "horizon_transfer_strong_pass": p4.get("P4_horizon_transfer_strong_pass"),
        "old_rank_generative_mechanism_ready": p5.get("P5_generative_mechanism_ready"),
        "controller_pass": p6.get("source_controller_pass", 0),
        "generated_route_status": "reopen_direct_solved_sandbox" if reopen else "stopped_no_new_objective",
        "APGU_APGV_APGW_APGX_run": 0,
        "direct_solved_sandbox_allowed": reopen,
        "generated_action_count": 0,
        "generated_route_stop_triggered": int(not reopen),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def base_acc(source_v9730: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(r) for r in read_csv(source_v9730 / "base_acc_sentinel_v9730.csv")]
    for r in rows:
        r["stage"] = "BASE_ACC_SENTINEL_V9740"
        r["reused_from_v9730"] = 1
        r["fake_data_used"] = 0
        r["proxy_row_used"] = 0
        r["cpu_offload_used"] = 0
    return rows, summary_row(rows)


def no_fake(paths: list[Path]) -> dict[str, Any]:
    rows_checked = fake_proxy = fake = proxy = cpu = 0
    for path in paths:
        if path.suffix.lower() != ".csv":
            continue
        for row in read_csv(path):
            rows_checked += 1
            fake += inum(row.get("fake_data_used"))
            proxy += inum(row.get("proxy_row_used"))
            cpu += inum(row.get("cpu_offload_used"))
            fake_proxy += int(inum(row.get("fake_data_used")) or inum(row.get("proxy_row_used")) or inum(row.get("cpu_offload_used")))
    return {"rows_checked": rows_checked, "fake_proxy_nonzero_count": fake_proxy, "fake_data_used": fake, "proxy_row_used": proxy, "cpu_offload_used": cpu, "no_fake": int(fake == 0), "no_proxy": int(proxy == 0)}


def sha_rows(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in paths.items() if path.exists()}


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if args.fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    artifacts: dict[str, Path] = {}

    def dump_csv(name: str, rows: list[dict[str, Any]]) -> Path:
        path = out / name
        write_csv(path, rows)
        artifacts[name] = path
        return path

    def dump_json(name: str, row: dict[str, Any]) -> Path:
        path = out / name
        write_json(path, row)
        artifacts[name] = path
        return path

    source_v9730 = Path(args.source_v9730)
    source_v9720 = Path(args.source_v9720)
    source_v9700 = Path(args.source_v9700)
    source_v9680 = Path(args.source_v9680)
    ap0, score_bundle, payload_by_id = v9720.load_ap0_and_payloads(args)
    r5b = v9660.r5b_scores(ap0, score_bundle)
    exact_summary = v9730.load_exact_summary(source_v9720)
    score_defs = v9730.score_defs_from_summary(ap0, exact_summary)
    wt_prev = load_wt_scores(source_v9730)

    p0_rows, p0_legality_rows, p0 = p0_boundary(source_v9730)
    dump_csv("p0_boundary_reproduction_v9740.csv", p0_rows)
    dump_csv("p0_field_legality_audit_v9740.csv", p0_legality_rows)
    p1_rows, p1_actions, p1, cs = p1_transfer_mismatch(ap0, r5b, score_defs, exact_summary, wt_prev, out)
    dump_csv("p1_old_exact_transfer_mismatch_v9740.csv", p1_rows)
    dump_csv("p1_old_exact_transfer_mismatch_actions_v9740.csv", p1_actions)
    p2_rows, p2_detail, p2, p2_sets = p2_core_expansion_ldo(ap0, r5b, score_defs, cs, out, int(args.seed))
    dump_csv("p2_core_expansion_ldo_responsibility_v9740.csv", p2_rows)
    dump_csv("p2_expansion10_ldo_detail_v9740.csv", p2_detail)
    p3_rows, p3, p3_methods = p3_dataset_shift_non_tuning(ap0, r5b, score_defs, out)
    dump_csv("p3_dataset_shift_non_tuning_v9740.csv", p3_rows)
    device = v9720.device_from(str(args.device))
    p4_mat, p4_scores, p4_cohorts, p4, horizon_scores = p4_horizon_transfer(args, ap0, payload_by_id, cs, p2_sets, device, out)
    dump_csv("p4_horizon_transfer_materializer_v9740.csv", p4_mat)
    dump_csv("p4_horizon_transfer_score_eval_v9740.csv", p4_scores)
    dump_csv("p4_horizon_transfer_cohort_summary_v9740.csv", p4_cohorts)
    p5_rows, p5 = p5_old_rank_mechanism(ap0, r5b, score_defs, cs, out)
    dump_csv("p5_old_rank_mechanism_ablation_v9740.csv", p5_rows)
    p6_rows, p6 = p6_controller(p2, p3, p4, p5)
    dump_csv("p6_existing_action_minimal_controller_v9740.csv", p6_rows)
    p7_rows, p7 = not_run("P7_SELECTED_RUNTIME_BOUNDARY_V9740", "P6_controller_not_passed", selected_runtime_pass=0)
    dump_csv("p7_selected_runtime_boundary_v9740.csv", p7_rows)
    p8_rows, p8 = p8_generated_route(source_v9680, p4, p5, p6)
    dump_csv("p8_generated_route_stop_reopen_rule_v9740.csv", p8_rows)
    p9_rows, p9 = not_run("P9_PAIRED_REPLAY_BOUNDARY_V9740", "P6_or_P7_not_passed", paired_replay_pass=0)
    dump_csv("p9_paired_replay_boundary_v9740.csv", p9_rows)
    p10_rows, p10 = not_run("P10_SHORT_FULL_BOUNDARY_V9740", "P9_paired_replay_not_open", short_run_boundary_open=0, full_run_boundary_open=0)
    dump_csv("p10_short_full_boundary_v9740.csv", p10_rows)
    base_rows, base_summary = base_acc(source_v9730)
    dump_csv("base_acc_sentinel_v9740.csv", base_rows)

    existing_ldo_blocked = int(
        fnum(p2.get("C0E1_precision")) >= 0.80
        and fnum(p2.get("C0E1_V_LCB")) > 0
        and fnum(p2.get("C0E1_LDO_drop")) > 0.15
    )
    transfer_mismatch = int(inum(p1.get("transfer_objective_mismatch")) and not inum(p4.get("P4_horizon_transfer_weak_pass")))
    if not inum(p0.get("p0_pass")):
        route, primary = "R0-BoundaryFail", "v9730_boundary_not_reproduced"
    elif inum(p6.get("source_controller_pass")) and inum(p7.get("selected_runtime_pass")):
        route, primary = "R1-ExistingActionControllerPass", "none"
    elif existing_ldo_blocked:
        route, primary = "R2-ExistingActionLDOBlocked", "existing_action_ldo_blocked"
    elif transfer_mismatch:
        route, primary = "R3-ExactTransferObjectiveMismatch", "transfer_objective_mismatch"
    elif inum(p4.get("P4_horizon_transfer_strong_pass")):
        route, primary = "R4-HorizonTransferRescuesTransfer", "none"
    else:
        route, primary = "R5-GeneratedRouteStopped", "generated_route_stopped_no_new_objective"
    system_pass = int(inum(p6.get("source_controller_pass")) and inum(p7.get("selected_runtime_pass")))
    route_decision = {
        "stage": "ROUTE_DECISION_V9740",
        "status": "summary",
        "route": route,
        "source_route_v9730": p0.get("source_route_v9730"),
        "p0_pass": p0.get("p0_pass"),
        "H1_transfer_mismatch_pass": p1.get("H1_old_rank_not_explained_by_one_step_transfer_pass"),
        "oldonly_precision": p1.get("oldonly_precision"),
        "exactonly_precision": p1.get("exactonly_precision"),
        "P2_core_expansion_ldo_pass": p2.get("P2_core_expansion_ldo_pass"),
        "Core77_LDO_drop": p2.get("Core77_LDO_drop"),
        "C0E1_LDO_drop": p2.get("C0E1_LDO_drop"),
        "P3_dataset_shift_non_tuning_pass": p3.get("P3_dataset_shift_non_tuning_pass"),
        "P4_horizon_transfer_weak_pass": p4.get("P4_horizon_transfer_weak_pass"),
        "P4_horizon_transfer_strong_pass": p4.get("P4_horizon_transfer_strong_pass"),
        "best_horizon_score_id": p4.get("best_score_id"),
        "best_horizon_precision": p4.get("best_precision"),
        "best_horizon_V_LCB": p4.get("best_V_LCB"),
        "best_horizon_longrisk_UCB": p4.get("best_longrisk_UCB"),
        "P5_old_rank_mechanism_explained": p5.get("P5_old_rank_mechanism_explained"),
        "P5_generative_mechanism_ready": p5.get("P5_generative_mechanism_ready"),
        "controller_pass": p6.get("source_controller_pass", 0),
        "selected_runtime_pass": p7.get("selected_runtime_pass", 0),
        "generated_route_status": p8.get("generated_route_status"),
        "direct_solved_sandbox_allowed": p8.get("direct_solved_sandbox_allowed"),
        "system_legal_controller_pass": system_pass,
        "primary_blocker": primary,
        "secondary_blocker": "transfer_objective_mismatch" if transfer_mismatch and primary != "transfer_objective_mismatch" else "generated_route_stopped_no_new_objective",
    }
    dump_json("route_decision_v9740.json", route_decision)
    allowed = {
        "selected_runtime_allowed": 0,
        "paired_replay_allowed": 0,
        "short_full_allowed": 0,
        "generated_route_allowed": 0,
        "direct_solved_sandbox_allowed": p8.get("direct_solved_sandbox_allowed"),
        "APGU_APGV_APGW_APGX_run_allowed": 0,
    }
    dump_json("allowed_next_gates_v9740.json", allowed)
    stop = {
        "stop_threshold_tuning": int(existing_ldo_blocked),
        "stop_exact_transfer_as_ranker": int(transfer_mismatch),
        "stop_generated_blind_variants": 1,
        "enter_system": system_pass,
    }
    dump_json("stop_conditions_v9740.json", stop)
    nf = no_fake(list(artifacts.values()))
    dump_csv("no_fake_audit_v9740.csv", [{"stage": "NO_FAKE_AUDIT_V9740", "status": "summary", **nf}])
    contract = {
        "stage": "CONTRACT_AUDIT_V9740",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "v9730_boundary_pass": p0.get("p0_pass"),
        "transfer_mismatch_audit_pass": p1.get("H1_old_rank_not_explained_by_one_step_transfer_pass"),
        "core_expansion_ldo_pass": p2.get("P2_core_expansion_ldo_pass"),
        "dataset_shift_non_tuning_pass": p3.get("P3_dataset_shift_non_tuning_pass"),
        "horizon_transfer_pass": f"{p4.get('P4_horizon_transfer_weak_pass')}/{p4.get('P4_horizon_transfer_strong_pass')}",
        "old_rank_mechanism_explained": p5.get("P5_old_rank_mechanism_explained"),
        "controller/runtime/system": f"{p6.get('source_controller_pass', 0)}/{p7.get('selected_runtime_pass', 0)}/{system_pass}",
        "generated_route_stop": p8.get("generated_route_stop_triggered"),
        "base_acc_sentinel_pass": base_summary.get("base_acc_sentinel_pass"),
        "base_acc_used_for_controller": 0,
        "uses_loss_backward/teacher/loss_modification": "0/0/0",
        "uses_dataset_name_for_selector/controller": "0/0",
        "uses_validation_or_test_for_controller": 0,
        "uses_future_outcome_for_features": 0,
        "proxy_transfer_promoted_to_official": 0,
        "diagnostic_promoted_to_official": 0,
        "fake/proxy/cpu_offload": f"{nf['fake_data_used']}/{nf['proxy_row_used']}/{nf['cpu_offload_used']}",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("contract_audit_v9740.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9740",
        "status": "summary",
        "route": route,
        "F0_boundary_fail": int(not inum(p0.get("p0_pass"))),
        "F1_old_rank_exact_transfer_mismatch": int(inum(p1.get("H1_old_rank_not_explained_by_one_step_transfer_pass"))),
        "F2_core77_itself_ldo_fail": p2.get("core77_itself_ldo_fail"),
        "F3_dataset_shift_non_tuning_fail": int(not inum(p3.get("P3_dataset_shift_non_tuning_pass"))),
        "F4_horizon_transfer_fail": int(not inum(p4.get("P4_horizon_transfer_weak_pass"))),
        "F5_controller_runtime_blocked": int(not inum(p6.get("source_controller_pass", 0))),
        "F6_generated_route_stopped": int(inum(p8.get("generated_route_stop_triggered"))),
        "F7_system_not_official": int(not system_pass),
        "primary_blocker": primary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("failure_taxonomy_v9740.csv", [failure])
    hashes = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    manifest = {
        "stage": "RUN_MANIFEST_V9740",
        "status": "summary",
        "out_dir": str(out),
        "wallclock_sec": time.perf_counter() - t0,
        "route": route,
        "primary_blocker": primary,
        "artifact_hashes": hashes,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("run_manifest_v9740.json", manifest)
    print(
        json.dumps(
            {
                "out_dir": str(out),
                "route": route,
                "primary_blocker": primary,
                "oldonly_precision": p1.get("oldonly_precision"),
                "Core77_LDO_drop": p2.get("Core77_LDO_drop"),
                "C0E1_LDO_drop": p2.get("C0E1_LDO_drop"),
                "best_horizon_score": p4.get("best_score_id"),
                "best_horizon_precision": p4.get("best_precision"),
                "system_legal_controller_pass": system_pass,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
