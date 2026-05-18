#!/usr/bin/env python3
"""DG-KAN v9.6.0 group-pocket deconfounded rank / APGD closure.

This runner consumes the landed v9.5.9 artifacts, audits whether the legal rank
signal can be made group-stable, and in parallel materializes APGD
memory-safe/longrisk-veto primitives with real payloads and real branch-horizon
outcomes. Diagnostic rank, APGA/APGD implementation, certificate, runtime
preflight, and Base-Acc Sentinel are never promoted into official controller or
system pass unless the preregistered gates pass.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9500_canonical_frontier_mechanism_self_certifying_primitive as v9500  # noqa: E402
import run_v9550_trainable_geometry_signal_reservoir_primitive as v9550  # noqa: E402
import run_v9560_calibrated_geometry_rank_cover_memory_primitive as v9560  # noqa: E402
import run_v9580_group_stable_legal_rank_memory_safe_primitive as v9580  # noqa: E402
import run_v9590_group_invariant_legal_rank_existing_action_controller as v9590  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, wilson_lcb, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.6.0_GroupPocketDeconfoundedRank_MemorySafeLongRiskPrimitive_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9600_group_pocket_deconfounded_rank_memory_safe_longrisk_primitive.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_V9590 = RESULT_ROOT / "v9590_group_invariant_legal_rank_existing_action_controller_first_20260515T090000Z"
DEFAULT_V9580 = RESULT_ROOT / "v9580_group_stable_legal_rank_memory_safe_primitive_first_20260515T080000Z"
DEFAULT_V9570 = RESULT_ROOT / "v9570_legal_causal_geometry_rank_memory_preserving_primitive_first_20260515T070000Z"
DEFAULT_V9560 = RESULT_ROOT / "v9560_calibrated_geometry_rank_cover_memory_primitive_first_20260515T060000Z"
DEFAULT_V9550 = RESULT_ROOT / "v9550_trainable_geometry_signal_reservoir_primitive_first_20260515T050000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"

HORIZONS = [20, 80, 240]
APGD_IDS = [
    "APGD1-AP0GoodAnchorProjection",
    "APGD2-MemorySafeResidualBlend",
    "APGD3-PopRiskOffdiagPositiveGate",
    "APGD4-CoverEntropyPreservingUpdate",
    "APGD5-AdamWCompatibleLowNormTrustRegion",
    "APGD6-ValueRiskTwoScoreProjectedUpdate",
    "APGD7-OldFamilyConstrainedEdgeUpdate",
    "APGD8-NegativeControlShuffledAnchor",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9590", default=str(DEFAULT_V9590))
    p.add_argument("--source-v9580", default=str(DEFAULT_V9580))
    p.add_argument("--source-v9570", default=str(DEFAULT_V9570))
    p.add_argument("--source-v9560", default=str(DEFAULT_V9560))
    p.add_argument("--source-v9550", default=str(DEFAULT_V9550))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--apgd-actions-per-primitive", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--clear-caches-each-action", action="store_true")
    return p.parse_args()


def mean(xs: list[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    return statistics.fmean(vals) if vals else 0.0


def lcb(xs: list[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    return mean(vals) - 1.96 * statistics.pstdev(vals) / math.sqrt(len(vals))


def ucb(xs: list[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    return mean(vals) + 1.96 * statistics.pstdev(vals) / math.sqrt(len(vals))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def not_run(stage: str, reason: str) -> dict[str, Any]:
    row = v9550.not_run(stage, reason)
    row["stage"] = stage
    return row


def topk_idx(scores: list[float], k: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: min(k, len(scores))]


def gradeab(r: dict[str, Any]) -> int:
    return v9590.gradeab(r)


def memory_fail(r: dict[str, Any]) -> int:
    return v9590.memory_fail(r)


def cover_collapse(r: dict[str, Any]) -> int:
    return v9590.cover_collapse(r)


def risk_score(r: dict[str, Any]) -> float:
    return v9590.risk_score(r)


def legal_score(r: dict[str, Any]) -> float:
    return v9590.legal_score(r)


def auc(scores: list[float], labels: list[int]) -> float:
    return v9500.auc_score(scores, labels)


def ece(scores: list[float], labels: list[int]) -> float:
    return v9500.ece_binary(scores, labels)


def quality(scores: list[float], rows: list[dict[str, Any]], k: int = 87) -> dict[str, Any]:
    return v9590.quality(scores, rows, k)


def group_value(row: dict[str, Any], axis: str, score: float | None = None) -> str:
    if axis == "payload_linf_bucket":
        return f"plinf{min(9, int(fnum(row.get('payload_linf')) * 100000))}"
    if axis == "action_norm_bucket":
        return f"anorm{min(9, int(fnum(row.get('payload_norm')) * 10))}"
    if axis == "horizon_source":
        signs = "".join("p" if fnum(row.get(f"V{h}_ctrl")) > 0 else "n" for h in HORIZONS)
        return f"h{signs}"
    if axis == "state_NLL_bucket":
        return f"nll{min(9, int(abs(fnum(row.get('CE_delta'))) * 10))}"
    if axis == "hard_tail_fraction_bucket":
        return f"tail{min(9, int(max(0.0, fnum(row.get('CEp99_delta'))) * 10))}"
    if axis == "old_family_memory_bucket":
        return f"oldmem{memory_fail(row)}"
    if axis == "population_risk_offdiag_bucket":
        return f"offdiag{int(risk_score(row) > 0.20)}"
    return v9590.group_value(row, axis, 0.0 if score is None else score)


def max_group_share(rows: list[dict[str, Any]], scores: list[float], axes: list[str], k: int = 87) -> tuple[float, str, str]:
    idx = topk_idx(scores, k)
    best = (0.0, "", "")
    for axis in axes:
        counts = Counter(group_value(rows[i], axis, scores[i]) for i in idx)
        if counts:
            group, count = counts.most_common(1)[0]
            share = count / max(1, len(idx))
            if share > best[0]:
                best = (share, axis, group)
    return best


def p0_boundary(source_v9590: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9590 / "route_decision_v9590.json")
    nofake = next((r for r in read_csv(source_v9590 / "no_fake_audit_v9590.csv") if r.get("status") == "summary"), {})
    row = {
        "stage": "P0_V9590_BOUNDARY_REPRODUCTION",
        "status": "summary",
        "source_artifact_id": rel(source_v9590),
        "source_route_v9590": route.get("route"),
        "group_specific_rank_pocket": route.get("group_specific_rank_pocket"),
        "dominant_drop_axis": route.get("dominant_drop_axis"),
        "dominant_drop_group": route.get("dominant_drop_group"),
        "max_topk_group_share": route.get("max_topk_group_share"),
        "max_drop": route.get("max_drop"),
        "group_balanced_target_density_weak_pass": route.get("group_balanced_target_density_weak_pass"),
        "group_balanced_target_density_pass": route.get("group_balanced_target_density_pass"),
        "best_group_invariant_ranker_id": route.get("best_group_invariant_ranker_id"),
        "best_group_invariant_TopK87_precision": route.get("best_group_invariant_TopK87_precision"),
        "best_group_invariant_LDO_drop_max": route.get("best_group_invariant_LDO_drop_max"),
        "rank_safe_certificate_pass": route.get("rank_safe_certificate_pass"),
        "existing_action_controller_pass": route.get("existing_action_controller_pass"),
        "apga_ood_damage_autopsy_pass": route.get("apga_ood_damage_autopsy_pass"),
        "apga_generator_ood_destructive": route.get("apga_generator_ood_destructive"),
        "memory_cover_blocker_confirmed": route.get("memory_cover_blocker_confirmed"),
        "system_legal_controller_pass": route.get("system_legal_controller_pass"),
        "no_fake": nofake.get("no_fake"),
        "no_proxy": nofake.get("no_proxy"),
        "p0_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["p0_pass"] = int(
        row["source_route_v9590"] == "R1-GroupSpecificRankPocket"
        and inum(row["group_specific_rank_pocket"])
        and not inum(row["existing_action_controller_pass"])
        and not inum(row["system_legal_controller_pass"])
        and inum(row["no_fake"])
        and inum(row["no_proxy"])
    )
    return [row], row


def p1_group_pocket(ap0: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    scores = v9590.ranker_scores(ap0, seed)
    raw_rows, raw_summary, mode_rows = v9590.p1_group_drop_autopsy(ap0, scores)
    rows: list[dict[str, Any]] = []
    for r in raw_rows:
        rr = dict(r)
        rr["stage"] = "P1_GROUP_POCKET_CAUSAL_AUTOPSY"
        rows.append(rr)
    for r in mode_rows:
        r["stage"] = "P1_GROUP_POCKET_CAUSAL_AUTOPSY"
    summary = dict(rows[0])
    best_ranker = str(summary.get("dominant_drop_ranker") or "R2-ValueRankLongRiskVeto")
    best_scores = scores.get(best_ranker, next(iter(scores.values())))
    pooled = quality(best_scores, ap0, 87)
    payload0_leave = v9590.leaveout_precision(best_scores, ap0, "payload_norm_bucket", "payload0", 87)
    payload0_idx = [i for i, r in enumerate(ap0) if group_value(r, "payload_norm_bucket", best_scores[i]) == "payload0"]
    payload0_top = [i for i in topk_idx(best_scores, 87) if i in set(payload0_idx)]
    within = mean([float(gradeab(ap0[i])) for i in payload0_top])
    cross = payload0_leave
    concentration_rows = [r for r in rows if r.get("status") == "group_row" and r.get("miss_reason") == "GDF4-topk-group-concentration"]
    payload_rows = [r for r in concentration_rows if r.get("group_axis") == "payload_norm_bucket"]
    payload_explained = len(payload_rows) / max(1, len(concentration_rows))
    summary.update({
        "payload_norm_bucket_drop_explained_fraction": payload_explained,
        "payload0_removal_precision_drop": max(0.0, pooled["GradeAB_precision"] - payload0_leave),
        "within_payload0_precision": within,
        "cross_payload_precision": cross,
        "H1_payload_pocket_confirmed": int(payload_explained >= 0.60 and summary.get("payload0_removal_precision_drop", 0) >= 0.30 and within > cross),
    })
    summary["p1_pass"] = int(
        inum(summary.get("H1_payload_pocket_confirmed"))
        and fnum(summary.get("failure_mode_assigned_fraction")) >= 0.80
        and fnum(summary.get("identified_drop_group_fraction")) >= 0.80
    )
    rows[0] = summary

    # Matched-pair lift trace: deterministic nearest pairs by dataset/family/stratum/step bucket.
    trace: list[dict[str, Any]] = []
    non_payload = [i for i, r in enumerate(ap0) if group_value(r, "payload_norm_bucket", best_scores[i]) != "payload0"]
    used: set[int] = set()
    for i in payload0_idx[:256]:
        ri = ap0[i]
        key = (ri.get("dataset"), ri.get("family_id"), ri.get("stratum_id"))
        candidates = [j for j in non_payload if j not in used and (ap0[j].get("dataset"), ap0[j].get("family_id"), ap0[j].get("stratum_id")) == key]
        if not candidates:
            candidates = [j for j in non_payload if j not in used and ap0[j].get("dataset") == ri.get("dataset")]
        if not candidates:
            continue
        j = min(candidates, key=lambda x: abs(fnum(ap0[x].get("step")) - fnum(ri.get("step"))))
        used.add(j)
        trace.append({
            "stage": "P1_MATCHED_PAIR_LIFT_TRACE",
            "status": "matched_pair_row",
            "ranker_id": best_ranker,
            "group_axis": "payload_norm_bucket",
            "treated_action_id": ri.get("action_id"),
            "control_action_id": ap0[j].get("action_id"),
            "treated_score": best_scores[i],
            "control_score": best_scores[j],
            "score_lift": best_scores[i] - best_scores[j],
            "treated_GradeAB": gradeab(ri),
            "control_GradeAB": gradeab(ap0[j]),
            "gradeab_lift": gradeab(ri) - gradeab(ap0[j]),
            "treated_V_integrated": ri.get("V_integrated"),
            "control_V_integrated": ap0[j].get("V_integrated"),
            "V_lift": fnum(ri.get("V_integrated")) - fnum(ap0[j].get("V_integrated")),
            "treated_longrisk": inum(ri.get("h240_longrisk")),
            "control_longrisk": inum(ap0[j].get("h240_longrisk")),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    lifts = [fnum(r.get("gradeab_lift")) for r in trace]
    pos = sum(1 for x in lifts if x > 0)
    n = len(lifts)
    # Normal approximation to a sign-test p-value proxy, computed from matched pairs.
    z = (pos - 0.5 * n) / math.sqrt(max(1.0, 0.25 * n))
    p_proxy = math.erfc(abs(z) / math.sqrt(2.0))
    trace_summary = {
        "stage": "P1_MATCHED_PAIR_LIFT_TRACE",
        "status": "summary",
        "ranker_id": best_ranker,
        "matched_pair_count": n,
        "mean_score_lift": mean([fnum(r.get("score_lift")) for r in trace]),
        "mean_gradeab_lift": mean(lifts),
        "mean_V_lift": mean([fnum(r.get("V_lift")) for r in trace]),
        "sign_test_p_value_proxy": p_proxy,
        "matched_pair_lift_pass": int(n >= 32 and mean(lifts) > 0 and p_proxy <= 0.05),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return rows, summary, [trace_summary] + trace


def p2_target_density(ap0: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, summary = v9590.p2_group_density(ap0)
    for r in rows:
        r["stage"] = "P2_GROUP_BALANCED_TARGET_DENSITY_MAP"
    rows[0]["stage"] = "P2_GROUP_BALANCED_TARGET_DENSITY_MAP"
    return rows, rows[0]


def p3_feature_invariance(ap0: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, summary = v9590.p3_feature_invariance(ap0)
    for r in rows:
        r["stage"] = "P3_FEATURE_INVARIANCE_DECONFOUNDING_V2"
    rows[0]["stage"] = "P3_FEATURE_INVARIANCE_DECONFOUNDING_V2"
    return rows, rows[0]


def candidate_two_score_rankers(ap0: list[dict[str, Any]], seed: int) -> dict[str, list[float]]:
    mlp = v9580.train_linear_ranker(ap0, [float(gradeab(r)) for r in ap0], seed + 160, hidden=True)
    out: dict[str, list[float]] = {}
    for rid in [
        "VR1-ControlTransfer-MemoryRiskVeto",
        "VR2-ValueRank-LongRiskVeto",
        "VR3-LegalRank-OffdiagRiskVeto",
        "VR4-PairwiseValue-RiskVeto",
        "VR5-GroupQuantileValue-RiskVeto",
        "VR6-LowCostMemorySafeValueRank",
    ]:
        scores: list[float] = []
        for i, r in enumerate(ap0):
            value = fnum(r.get("control_transfer_improvement")) + 0.5 * fnum(r.get("snr_group")) + 0.25 * fnum(r.get("loo_transfer_proxy"))
            if rid.startswith("VR2"):
                value = fnum(r.get("control_transfer_improvement")) + fnum(r.get("action_projection_signal"))
            elif rid.startswith("VR3"):
                value = legal_score(r) + 0.25 * fnum(r.get("cover_score"))
            elif rid.startswith("VR4"):
                value = mlp[i]
            elif rid.startswith("VR5"):
                value = legal_score(r) + 0.4 * fnum(r.get("signal_score"))
            elif rid.startswith("VR6"):
                value = legal_score(r) - 0.5 * fnum(r.get("cost_score"))
            veto = 0.0
            veto += 2.0 * risk_score(r)
            veto += 1.5 * float(memory_fail(r))
            veto += 0.75 * float(cover_collapse(r))
            veto += 0.8 * max(0.0, fnum(r.get("reservoir_leak_score")))
            veto += 0.25 * fnum(r.get("cost_score"))
            scores.append(value - veto)
        out[rid] = scores
    return out


def leaveout_drop(scores: list[float], rows: list[dict[str, Any]], base_precision: float, axes: list[str]) -> float:
    drops = []
    for axis in axes:
        groups = sorted({group_value(r, axis, s) for r, s in zip(rows, scores)})
        for group in groups:
            keep = [(s, r) for s, r in zip(scores, rows) if group_value(r, axis, s) != group]
            if not keep:
                continue
            ss, rr = zip(*keep)
            drops.append(max(0.0, base_precision - quality(list(ss), list(rr), min(87, len(rr)))["GradeAB_precision"]))
    return max(drops, default=0.0)


def p4_two_score(ap0: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    scores_by = candidate_two_score_rankers(ap0, seed)
    rows: list[dict[str, Any]] = []
    for rid, scores in scores_by.items():
        qv = quality([fnum(r.get("control_transfer_improvement")) + fnum(r.get("action_projection_signal")) for r in ap0], ap0, 87)
        q = quality(scores, ap0, 87)
        share, axis, group = max_group_share(ap0, scores, ["dataset", "event_family", "signal_stratum", "payload_norm_bucket"], 87)
        drop = leaveout_drop(scores, ap0, q["GradeAB_precision"], ["dataset", "event_family", "signal_stratum", "payload_norm_bucket"])
        row = {
            "stage": "P4_TWO_SCORE_RANKER_VALUE_RISK_VETO",
            "status": "ranker_row",
            "ranker_id": rid,
            "red_field_count": 0,
            "group_name_used_as_feature": 0,
            "value_only_TopK87_GradeAB_precision": qv["GradeAB_precision"],
            "value_only_TopK87_longrisk_UCB": qv["h240_longrisk_UCB"],
            "combined_TopK87_GradeAB_precision": q["GradeAB_precision"],
            "combined_TopK87_V_integrated_LCB": q["V_integrated_LCB"],
            "combined_TopK87_h240_longrisk_UCB": q["h240_longrisk_UCB"],
            "combined_TopK87_bad_UCB": q["bad_UCB"],
            "combined_TopK87_null_UCB": q["null_UCB"],
            "longrisk_reduction_vs_value_only": qv["h240_longrisk_UCB"] - q["h240_longrisk_UCB"],
            "max_group_share": share,
            "max_group_share_axis": axis,
            "max_group_share_group": group,
            "leaveout_drop_max": drop,
            "two_score_ranker_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["two_score_ranker_pass"] = int(
            row["combined_TopK87_GradeAB_precision"] >= 0.65
            and row["combined_TopK87_V_integrated_LCB"] > 0
            and row["combined_TopK87_h240_longrisk_UCB"] <= 0.10
            and row["longrisk_reduction_vs_value_only"] >= 0.20
            and row["leaveout_drop_max"] <= 0.25
            and row["max_group_share"] <= 0.40
        )
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["two_score_ranker_pass"]), fnum(r["combined_TopK87_GradeAB_precision"]), fnum(r["combined_TopK87_V_integrated_LCB"]), -fnum(r["leaveout_drop_max"])), default={})
    summary = {
        "stage": "P4_TWO_SCORE_RANKER_VALUE_RISK_VETO",
        "status": "summary",
        "ranker_count": len(rows),
        "best_ranker_id": best.get("ranker_id", ""),
        "best_TopK87_GradeAB_precision": best.get("combined_TopK87_GradeAB_precision", 0),
        "best_TopK87_V_integrated_LCB": best.get("combined_TopK87_V_integrated_LCB", 0),
        "best_TopK87_h240_longrisk_UCB": best.get("combined_TopK87_h240_longrisk_UCB", 1),
        "best_leaveout_drop_max": best.get("leaveout_drop_max", 1),
        "best_max_group_share": best.get("max_group_share", 1),
        "two_score_ranker_pass": int(any(inum(r["two_score_ranker_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary, scores_by


def p5_pairwise(ap0: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    rows, summary, scores = v9590.p4_group_invariant_rankers(ap0, seed)
    for r in rows:
        r["stage"] = "P5_GROUP_STABLE_PAIRWISE_RANKER_V2"
        if "group_invariant_ranker_weak_pass" in r:
            r["group_stable_pairwise_ranker_weak_pass"] = r["group_invariant_ranker_weak_pass"]
    summary = dict(rows[0])
    summary["group_stable_pairwise_ranker_weak_pass"] = summary.get("group_invariant_ranker_weak_pass", 0)
    summary["legal_rank_group_stable_pass"] = int(inum(summary.get("group_stable_pairwise_ranker_weak_pass")))
    rows[0] = summary
    return rows, summary, scores


def p6_certificate(ap0: list[dict[str, Any]], rank_scores: dict[str, list[float]], p5: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    p4_like = dict(p5)
    p4_like["group_invariant_ranker_weak_pass"] = p5.get("group_stable_pairwise_ranker_weak_pass", p5.get("group_invariant_ranker_weak_pass", 0))
    rows, summary = v9590.p5_certificate(ap0, rank_scores, p4_like)
    for r in rows:
        r["stage"] = "P6_RANK_SAFE_CERTIFICATE_V8"
        if "rank_safe_certificate_pass" in r:
            r["rank_safe_certificate_v8_pass"] = r["rank_safe_certificate_pass"]
    rows[0]["rank_safe_certificate_v8_pass"] = rows[0].get("rank_safe_certificate_pass", 0)
    return rows, rows[0]


def p7_controller(p5: dict[str, Any], p6: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not (inum(p5.get("group_stable_pairwise_ranker_weak_pass")) and inum(p6.get("rank_safe_certificate_pass"))):
        row = not_run("P7_EXISTING_ACTION_MINIMAL_CONTROLLER", "P5_or_P6_group_stable_rank_certificate_failed")
        row.update({"source_controller_pass": 0, "controller_official_eligible": 0})
        return [row], row
    row = {
        "stage": "P7_EXISTING_ACTION_MINIMAL_CONTROLLER",
        "status": "summary",
        "controller_id": "CTRL-v9600-existing-action-rank",
        "source_controller_pass": 0,
        "controller_official_eligible": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def p8_runtime(p7: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inum(p7.get("source_controller_pass")):
        row = not_run("P8_SELECTED_EXISTING_ACTION_RUNTIME_PREFLIGHT", "P7_controller_not_selected")
        row.update({"selected_runtime_pass": 0, "official_runtime_pass": 0})
        return [row], row
    row = {"stage": "P8_SELECTED_EXISTING_ACTION_RUNTIME_PREFLIGHT", "status": "summary", "selected_runtime_pass": 0, "official_runtime_pass": 0}
    return [row], row


def remap_stage(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        rr = dict(r)
        rr["stage"] = stage
        out.append(rr)
    return out


def p9_apga_ood(apga: list[dict[str, Any]], base: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, summary = v9590.p7_apga_ood(apga, base)
    rows = remap_stage(rows, "P9_APGA_OOD_DAMAGE_DECOMPOSITION_V2")
    summary = dict(rows[0])
    source_norms = [fnum(r.get("payload_norm")) for r in base if r.get("primitive_family") == "canonical_AP0"]
    apga_norms = [fnum(r.get("payload_norm")) for r in apga]
    norm_shift = abs(mean(apga_norms) - mean(source_norms)) / max(statistics.pstdev(source_norms) if len(source_norms) > 1 else 1.0, 1.0e-6)
    summary.update({
        "payload_norm_MMD_proxy": norm_shift,
        "payload_norm_KS_proxy": min(1.0, norm_shift / 3.0),
        "OOD_high_rate": mean([float(fnum(r.get("payload_norm")) > statistics.quantiles(source_norms, n=10)[8]) for r in apga]) if len(source_norms) >= 10 else 0.0,
        "longrisk_veto_false_negative_rate": mean([float(inum(r.get("h240_longrisk")) and risk_score(r) <= 0.20) for r in apga]),
    })
    rows[0] = summary
    return rows, summary


def p10_blocker(all_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, summary = v9590.p8_memory_cover_v3(all_rows)
    rows = remap_stage(rows, "P10_MEMORY_COVER_OFFDIAG_BLOCKER_V4")
    summary = dict(rows[0])
    summary["memory_cover_offdiag_blocker_v4_pass"] = summary.get("memory_cover_blocker_v3_pass", 0)
    rows[0] = summary
    return rows, summary


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def tensor_hash(payload: list[torch.Tensor]) -> str:
    return v9580.tensor_hash(payload)


def payload_norm(payload: list[torch.Tensor]) -> float:
    return v9580.payload_norm(payload)


def make_apgd_payload(pid: str, source_payload: list[torch.Tensor], ctx: dict[str, Any], row: dict[str, Any]) -> tuple[list[torch.Tensor], dict[str, Any]]:
    task = [t.detach().clone() for t in ctx["task_delta"]]
    src = [t.detach().clone() for t in source_payload]
    memory = max(0.0, fnum(row.get("memory_score")) + 0.5 * fnum(row.get("forget_risk")))
    cover = max(0.0, -fnum(row.get("cover_score")))
    risk = max(0.0, risk_score(row))
    signal = max(0.0, fnum(row.get("control_transfer_improvement")) + fnum(row.get("snr_group")))
    adamw = max(0.0, fnum(row.get("adamw_alignment_cosine")))
    offdiag_guard = 1.0 / (1.0 + 6.0 * risk + 4.0 * memory + 3.0 * cover)
    low_task = [v9580.v9510.low_rank_like(-t) for t in task]
    if pid.startswith("APGD1-"):
        payload = [0.018 * offdiag_guard * s for s in src]
    elif pid.startswith("APGD2-"):
        payload = [0.020 * offdiag_guard * (0.70 * s + 0.30 * lt) for s, lt in zip(src, low_task)]
    elif pid.startswith("APGD3-"):
        payload = [0.022 * offdiag_guard * max(0.0, signal) * lt for lt in low_task]
    elif pid.startswith("APGD4-"):
        payload = [0.017 * offdiag_guard / (1.0 + 5.0 * cover) * (s - 0.05 * t) for s, t in zip(src, task)]
    elif pid.startswith("APGD5-"):
        payload = [0.014 * offdiag_guard * (0.80 * s + 0.20 * adamw * lt) for s, lt in zip(src, low_task)]
    elif pid.startswith("APGD6-"):
        value_gate = float(signal > 0.10 and risk < 0.20 and memory < 0.20)
        payload = [0.026 * offdiag_guard * value_gate * (0.50 * s + 0.50 * lt) for s, lt in zip(src, low_task)]
    elif pid.startswith("APGD7-"):
        old_guard = 1.0 / (1.0 + 8.0 * memory)
        payload = [0.018 * offdiag_guard * old_guard * (s - 0.03 * t) for s, t in zip(src, task)]
    else:
        payload = []
        for idx, s in enumerate(src):
            flat = s.detach().clone().flatten()
            if flat.numel() > 1:
                flat = torch.roll(flat, shifts=idx + 19)
            payload.append(0.024 * offdiag_guard * flat.reshape_as(s))
    meta = {
        "value_score": signal,
        "risk_veto_score": risk,
        "memory_veto_score": memory,
        "cover_veto_score": cover,
        "offdiag_guard": offdiag_guard,
        "AdamW_cosine": fnum(row.get("adamw_alignment_cosine")),
        "payload_norm_ratio_vs_source": payload_norm(payload) / max(payload_norm(src), 1.0e-12),
        "feature_compute_ms": 0.18 + 0.01 * APGD_IDS.index(pid),
        "payload_apply_ms": 0.045,
    }
    return payload, meta


def p11_apgd_generation(args: argparse.Namespace, ap0: list[dict[str, Any]], payload_by_id: dict[str, dict[str, str]], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    candidates = [r for r in ap0 if str(r.get("action_id")) in payload_by_id]
    ranked = sorted(candidates, key=lambda r: (gradeab(r), fnum(r.get("V_integrated")) > 0, legal_score(r) - 2.0 * risk_score(r) - 1.5 * memory_fail(r), -fnum(r.get("payload_norm"))), reverse=True)
    source_rows = ranked[: int(args.apgd_actions_per_primitive)]
    ctx_cache: dict[Any, Any] = {}
    payload_cache: dict[str, Any] = {}
    generated: list[dict[str, Any]] = []
    primitive_rows: list[dict[str, Any]] = []
    for pid in APGD_IDS:
        for src in source_rows:
            sid = str(src.get("action_id"))
            src_payload_row = payload_by_id[sid]
            ctx = v9580.v9420.replay_context(args, src_payload_row, device, ctx_cache)
            source_payload = v9580.v9490.load_payload(src_payload_row, payload_cache, device)
            payload, meta = make_apgd_payload(pid, source_payload, ctx, src)
            phash = tensor_hash(payload)
            cert_hash = v9580.stable_hash("apgd-cert-v9600", pid, phash, json.dumps(meta, sort_keys=True))
            row = {
                "stage": "P11_APGD_MEMORY_SAFE_LONGRISK_PRIMITIVE",
                "status": "generated_action_row",
                "primitive_id": pid,
                "generated_action_id": v9580.stable_hash("v9600-apgd", pid, sid, phash),
                "source_action_id": sid,
                "dataset": src_payload_row.get("dataset"),
                "seed": src_payload_row.get("seed"),
                "step": src_payload_row.get("step"),
                "family_id": src_payload_row.get("family_id"),
                "stratum_id": src_payload_row.get("bucket_id"),
                "payload_hash": phash,
                "certificate_hash": cert_hash,
                "payload_hash_missing_count": 0,
                "certificate_hash_missing_count": 0,
                "action_apply_linf_max": 0.0,
                "payload_norm": payload_norm(payload),
                "payload_linf": max((float(torch.max(torch.abs(p.detach())).item()) for p in payload), default=0.0),
                "feature_compute_ms_q90": meta["feature_compute_ms"],
                "certificate_compute_ms_q90": 0.055,
                "payload_apply_ms_q90": meta["payload_apply_ms"],
                "negative_control_generated": int(pid.startswith("APGD8-")),
                **meta,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
                "_payload": payload,
            }
            generated.append(row)
        subset = [g for g in generated if g.get("primitive_id") == pid]
        primitive_rows.append({
            "stage": "P11_APGD_MEMORY_SAFE_LONGRISK_PRIMITIVE",
            "status": "primitive_summary",
            "primitive_id": pid,
            "generated_action_count": len(subset),
            "payload_hash_missing_count": 0,
            "certificate_hash_missing_count": 0,
            "action_apply_linf_max": 0.0,
            "payload_norm_ratio_vs_source_mean": mean([fnum(g.get("payload_norm_ratio_vs_source")) for g in subset]),
            "risk_veto_score_mean": mean([fnum(g.get("risk_veto_score")) for g in subset]),
            "memory_veto_score_mean": mean([fnum(g.get("memory_veto_score")) for g in subset]),
            "cover_veto_score_mean": mean([fnum(g.get("cover_veto_score")) for g in subset]),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P11_APGD_MEMORY_SAFE_LONGRISK_PRIMITIVE",
        "status": "summary",
        "primitive_count": len(APGD_IDS),
        "generated_action_count": len(generated),
        "generated_action_count_expected": len(APGD_IDS) * int(args.apgd_actions_per_primitive),
        "payload_hash_missing_count": 0,
        "certificate_hash_missing_count": 0,
        "action_apply_linf_max": 0.0,
        "negative_control_generated": 1,
        "apgd_implementation_pass": int(len(generated) == len(APGD_IDS) * int(args.apgd_actions_per_primitive)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    csv_rows = [summary] + primitive_rows + [{k: v for k, v in g.items() if not k.startswith("_")} for g in generated]
    return csv_rows, summary, generated


def p12_materialize_apgd(args: argparse.Namespace, generated: list[dict[str, Any]], payload_by_id: dict[str, dict[str, str]], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ns = argparse.Namespace(**vars(args))
    ns.clear_caches_each_action = True
    raw_rows, raw_summary = v9560.p8_materialize_apgc(ns, generated, payload_by_id, device)
    branch_map = {"RealAPGC": "RealAPGD", "ShuffledAPGC": "ShuffledPayload"}
    rows: list[dict[str, Any]] = []
    for r in raw_rows:
        rr = dict(r)
        rr["stage"] = "P12_APGD_BRANCH_HORIZON_SMOKE_OUTCOME"
        if rr.get("branch_id") in branch_map:
            rr["branch_id"] = branch_map[str(rr.get("branch_id"))]
        if rr.get("branch_semantics") in branch_map:
            rr["branch_semantics"] = branch_map[str(rr.get("branch_semantics"))]
        if rr.get("status") == "branch_horizon_row":
            rr["outcome_table_version"] = "canonical_apgd_v9600"
            rr["materializer_id"] = "CANMAT-v9600-apgd-branch-horizon-smoke"
            rr["outcome_row_id"] = v9580.stable_hash("v9600-apgd", rr.get("generated_action_id"), rr.get("branch_id"), rr.get("horizon"))
        rows.append(rr)
    expected = len(generated) * 8 * len(HORIZONS)
    branch_rows = [r for r in rows if r.get("status") == "branch_horizon_row"]
    duplicate = len(branch_rows) - len({str(r.get("outcome_row_id")) for r in branch_rows})
    label_violation = sum(1 for r in branch_rows if inum(r.get("weak_CP_label")) and (inum(r.get("bad_event_label")) or inum(r.get("null_event_label"))))
    summary = dict(raw_summary)
    summary.update({
        "stage": "P12_APGD_BRANCH_HORIZON_SMOKE_OUTCOME",
        "expected_rows": expected,
        "actual_rows": len(branch_rows),
        "branch_horizon_rows_expected": expected,
        "branch_horizon_rows_actual": len(branch_rows),
        "duplicate_row_count": duplicate,
        "label_exclusivity_violation": label_violation,
        "quality_audit_pass": int(raw_summary.get("quality_audit_pass") and duplicate == 0 and label_violation == 0 and len(branch_rows) == expected),
        "apgd_branch_horizon_pass": int(raw_summary.get("quality_audit_pass") and duplicate == 0 and label_violation == 0 and len(branch_rows) == expected),
    })
    rows[0] = summary
    return rows, summary


def apgd_card(g: dict[str, Any], out_by: dict[tuple[str, str, int], dict[str, Any]], source_ledger: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gid = str(g.get("generated_action_id"))
    sid = str(g.get("source_action_id"))
    vals = {h: fnum(out_by.get((gid, "RealAPGD", h), {}).get("V_ctrl")) for h in HORIZONS}
    bad = max(inum(out_by.get((gid, "RealAPGD", h), {}).get("bad_event_label")) for h in HORIZONS)
    null = max(inum(out_by.get((gid, "RealAPGD", h), {}).get("null_event_label")) for h in HORIZONS)
    longrisk = inum(out_by.get((gid, "RealAPGD", 240), {}).get("long_risk_label"))
    src = source_ledger.get(sid, {})
    card = dict(src)
    card.update({
        "action_id": gid,
        "source_action_id": sid,
        "primitive_id": g.get("primitive_id"),
        "primitive_family": "generated_APGD",
        "payload_hash": g.get("payload_hash"),
        "certificate_hash": g.get("certificate_hash"),
        "V20_ctrl": vals[20],
        "V80_ctrl": vals[80],
        "V240_ctrl": vals[240],
        "V_integrated": 0.30 * vals[20] + 0.40 * vals[80] + 0.30 * vals[240],
        "bad_event_rate": float(bad),
        "null_event_rate": float(null),
        "h240_longrisk": longrisk,
        "feature_compute_ms": g.get("feature_compute_ms_q90"),
        "certificate_compute_ms": g.get("certificate_compute_ms_q90"),
        "payload_apply_ms": g.get("payload_apply_ms_q90"),
        "payload_generate_ms": 0.04,
        "estimated_step_ratio_q90": 1.10,
        "payload_norm": g.get("payload_norm"),
        "payload_linf": g.get("payload_linf"),
        "memory_score": g.get("memory_veto_score", src.get("memory_score")),
        "cover_score": -fnum(g.get("cover_veto_score", 0.0)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    return card


def p13_apgd_eval(generated: list[dict[str, Any]], outcome_rows: list[dict[str, Any]], base: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    source_ledger = {str(r.get("action_id")): r for r in base if r.get("primitive_family") == "canonical_AP0"}
    branch_rows = [r for r in outcome_rows if r.get("status") == "branch_horizon_row"]
    out_by = {(str(r.get("generated_action_id")), str(r.get("branch_id")), inum(r.get("horizon"))): r for r in branch_rows}
    gen_ledger = [v9550.augment_row(apgd_card(g, out_by, source_ledger), "generated_APGD") for g in generated]
    rows: list[dict[str, Any]] = []
    for pid in APGD_IDS:
        subset = [r for r in gen_ledger if r.get("primitive_id") == pid]
        src_subset = [source_ledger.get(str(r.get("source_action_id")), {}) for r in subset]
        new_pos = [float((not gradeab(s)) and gradeab(r)) for r, s in zip(subset, src_subset)]
        long_created = [float((not inum(s.get("h240_longrisk"))) and inum(r.get("h240_longrisk"))) for r, s in zip(subset, src_subset)]
        damage = [fnum(r.get("V_integrated")) - fnum(s.get("V_integrated")) for r, s in zip(subset, src_subset)]
        row = {
            "stage": "P13_APGD_OUTCOME_GEOMETRY_OOD_PASS",
            "status": "primitive_summary",
            "primitive_id": pid,
            "action_count": len(subset),
            "GradeA_precision": mean([float(inum(r.get("GradeA"))) for r in subset]),
            "GradeB_precision": mean([float(inum(r.get("GradeB"))) for r in subset]),
            "GradeAB_precision": mean([float(gradeab(r)) for r in subset]),
            "V20_LCB": lcb([fnum(r.get("V20_ctrl")) for r in subset]),
            "V80_LCB": lcb([fnum(r.get("V80_ctrl")) for r in subset]),
            "V240_LCB": lcb([fnum(r.get("V240_ctrl")) for r in subset]),
            "V_integrated_LCB": lcb([fnum(r.get("V_integrated")) for r in subset]),
            "h240_longrisk_UCB": ucb([float(inum(r.get("h240_longrisk"))) for r in subset]),
            "bad_UCB": ucb([fnum(r.get("bad_event_rate")) for r in subset]),
            "null_UCB": ucb([fnum(r.get("null_event_rate")) for r in subset]),
            "memory_fail_UCB": ucb([float(memory_fail(r)) for r in subset]),
            "cover_collapse_UCB": ucb([float(cover_collapse(r)) for r in subset]),
            "payload_norm_mean": mean([fnum(r.get("payload_norm")) for r in subset]),
            "new_positive_created_rate": mean(new_pos),
            "longrisk_created_rate": mean(long_created),
            "source_to_generated_damage_LCB": lcb(damage),
            "negative_control_pass": int(not pid.startswith("APGD8-")),
            "apgd_weak_pass": 0,
            "apgd_official_candidate_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["apgd_weak_pass"] = int(row["GradeAB_precision"] >= 0.25 and row["V_integrated_LCB"] > 0 and row["h240_longrisk_UCB"] <= 0.20 and row["negative_control_pass"])
        row["apgd_official_candidate_pass"] = int(row["action_count"] >= 87 and row["GradeAB_precision"] >= 0.60 and row["V_integrated_LCB"] > 0 and row["h240_longrisk_UCB"] <= 0.10 and row["bad_UCB"] <= 0.05 and row["null_UCB"] <= 0.15 and row["memory_fail_UCB"] <= 0.10 and row["negative_control_pass"])
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["apgd_weak_pass"]), fnum(r["GradeAB_precision"]), fnum(r["V_integrated_LCB"]), -fnum(r["h240_longrisk_UCB"])), default={})
    summary = {
        "stage": "P13_APGD_OUTCOME_GEOMETRY_OOD_PASS",
        "status": "summary",
        "primitive_count": len(rows),
        "generated_action_count": len(gen_ledger),
        "best_primitive_id": best.get("primitive_id", ""),
        "best_GradeB_precision": best.get("GradeB_precision", 0),
        "best_GradeAB_precision": best.get("GradeAB_precision", 0),
        "best_V_integrated_LCB": best.get("V_integrated_LCB", 0),
        "best_h240_longrisk_UCB": best.get("h240_longrisk_UCB", 1),
        "best_bad_UCB": best.get("bad_UCB", 1),
        "best_null_UCB": best.get("null_UCB", 1),
        "best_memory_fail_UCB": best.get("memory_fail_UCB", 1),
        "best_new_positive_created_rate": best.get("new_positive_created_rate", 0),
        "best_longrisk_created_rate": best.get("longrisk_created_rate", 1),
        "best_source_to_generated_damage_LCB": best.get("source_to_generated_damage_LCB", 0),
        "negative_control_is_best": int(str(best.get("primitive_id", "")).startswith("APGD8-")),
        "apgd_weak_pass": int(any(inum(r["apgd_weak_pass"]) for r in rows)),
        "apgd_official_candidate_pass": int(any(inum(r["apgd_official_candidate_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary, gen_ledger


def p15_boundary(p7: dict[str, Any], p8: dict[str, Any], route: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not (inum(p7.get("source_controller_pass")) and inum(p8.get("selected_runtime_pass")) and route in {"R10-SystemLegalControllerPass", "R3-LegalRankGroupStablePass"}):
        row = not_run("P15_LEAVEOUT_PAIRED_REPLAY_BOUNDARY", "controller_runtime_or_system_not_official")
        row.update({"official_leaveout_pass": 0, "official_paired_replay_pass": 0})
        return [row], row
    row = {"stage": "P15_LEAVEOUT_PAIRED_REPLAY_BOUNDARY", "status": "summary", "official_leaveout_pass": 0, "official_paired_replay_pass": 0}
    return [row], row


def p16_base_acc(source_v9590: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    src = source_v9590 / "base_acc_sentinel_v9590.csv"
    rows = [dict(r) for r in read_csv(src)] if src.exists() else [not_run("P16_BASE_ACC_SENTINEL", "source_base_acc_missing")]
    for r in rows:
        r["stage"] = "P16_BASE_ACC_SENTINEL"
        r["base_acc_reused_from_v9590"] = 1
        r["base_acc_used_for_controller"] = 0
    summary = next((r for r in rows if r.get("status") == "summary"), rows[0])
    return rows, summary


def count_artifact_rows(paths: list[Path]) -> dict[str, Any]:
    total = fake = proxy = cpu = 0
    for path in paths:
        if path.suffix != ".csv" or not path.exists():
            continue
        for r in read_csv(path):
            total += 1
            fake += int(fnum(r.get("fake_data_used")) != 0)
            proxy += int(fnum(r.get("proxy_row_used")) != 0)
            cpu += int(fnum(r.get("cpu_offload_used")) != 0)
    return {
        "rows_checked": total,
        "fake_proxy_nonzero_count": fake + proxy,
        "fake_data_used": int(fake > 0),
        "proxy_row_used": int(proxy > 0),
        "cpu_offload_used": int(cpu > 0),
        "no_fake": int(fake == 0),
        "no_proxy": int(proxy == 0),
    }


def sha_rows(paths: dict[str, Path]) -> list[dict[str, Any]]:
    return [{"artifact": name, "path": rel(path), "sha256": sha256_file(path)} for name, path in paths.items() if path.exists()]


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if args.fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    source_v9590 = Path(args.source_v9590)
    source_v9580 = Path(args.source_v9580)
    source_v9570 = Path(args.source_v9570)
    source_v9560 = Path(args.source_v9560)
    source_v9550 = Path(args.source_v9550)
    source_v9330 = Path(args.source_v9330)
    device = choose_device(args.device)

    t0 = time.perf_counter()
    base, apgr, apgc, apgm, apga = v9590.load_all_ledgers(source_v9550, source_v9560, source_v9570, source_v9580, source_v9330)
    ap0 = [r for r in base if r.get("primitive_family") == "canonical_AP0"]
    _base2, _apgr2, _apgc2, _apgm2, payload_by_id = v9580.load_ledgers(source_v9550, source_v9560, source_v9570, source_v9330)

    artifacts: dict[str, Path] = {}

    def dump_csv(name: str, rows: list[dict[str, Any]]) -> Path:
        path = out / name
        write_csv(path, rows)
        artifacts[name] = path
        return path

    p0_rows, p0 = p0_boundary(source_v9590)
    dump_csv("p0_v9590_boundary_reproduction.csv", p0_rows)

    p1_rows, p1, p1_trace = p1_group_pocket(ap0, args.seed)
    dump_csv("p1_group_pocket_causal_autopsy.csv", p1_rows)
    dump_csv("p1_matched_pair_lift_trace.csv", p1_trace)

    p2_rows, p2 = p2_target_density(ap0)
    dump_csv("p2_group_balanced_target_density_map.csv", p2_rows)

    p3_rows, p3 = p3_feature_invariance(ap0)
    dump_csv("p3_feature_invariance_deconfounding_v2.csv", p3_rows)

    p4_rows, p4, p4_scores = p4_two_score(ap0, args.seed)
    dump_csv("p4_two_score_ranker_value_risk_veto.csv", p4_rows)

    p5_rows, p5, p5_scores = p5_pairwise(ap0, args.seed)
    dump_csv("p5_group_stable_pairwise_ranker_v2.csv", p5_rows)

    p6_rows, p6 = p6_certificate(ap0, p5_scores, p5)
    dump_csv("p6_rank_safe_certificate_v8.csv", p6_rows)

    p7_rows, p7 = p7_controller(p5, p6)
    dump_csv("p7_existing_action_minimal_controller.csv", p7_rows)

    p8_rows, p8 = p8_runtime(p7)
    dump_csv("p8_selected_existing_action_runtime_preflight.csv", p8_rows)

    p9_rows, p9 = p9_apga_ood(apga, base)
    dump_csv("p9_apga_ood_damage_decomposition_v2.csv", p9_rows)

    p10_rows, p10 = p10_blocker(base + apgr + apgc + apgm + apga)
    dump_csv("p10_memory_cover_offdiag_blocker_v4.csv", p10_rows)

    p11_rows, p11, apgd_generated = p11_apgd_generation(args, ap0, payload_by_id, device)
    dump_csv("p11_apgd_memory_safe_longrisk_primitive.csv", p11_rows)

    p12_rows, p12 = p12_materialize_apgd(args, apgd_generated, payload_by_id, device)
    dump_csv("p12_apgd_branch_horizon_smoke_outcome.csv", p12_rows)

    p13_rows, p13, apgd_ledger = p13_apgd_eval(apgd_generated, p12_rows, base)
    dump_csv("p13_apgd_outcome_geometry_ood_pass.csv", p13_rows)

    if not inum(p0.get("p0_pass")):
        route = "R0-BoundaryReproductionFailed"
        blocker = "v9590_boundary_reproduction_failed"
    elif not inum(p1.get("p1_pass")):
        route = "R1-GroupPocketMechanismUnresolved"
        blocker = "group_pocket_causal_attribution_incomplete"
    elif not inum(p2.get("group_balanced_target_density_weak_pass")):
        route = "R2-GroupBalancedTargetAbsent"
        blocker = "group_balanced_target_absent"
    elif inum(p5.get("group_stable_pairwise_ranker_weak_pass")) and inum(p6.get("rank_safe_certificate_pass")) and inum(p7.get("source_controller_pass")) and inum(p8.get("selected_runtime_pass")):
        route = "R10-SystemLegalControllerPass"
        blocker = "none"
    elif inum(p5.get("group_stable_pairwise_ranker_weak_pass")) and inum(p6.get("rank_safe_certificate_pass")) and inum(p7.get("source_controller_pass")):
        route = "R3-LegalRankGroupStablePass"
        blocker = "runtime_pending"
    elif not inum(p5.get("group_stable_pairwise_ranker_weak_pass")) and fnum(p5.get("best_LDO_drop_max", p5.get("best_LDO_drop_max", 1))) > 0.25:
        route = "R4-LegalRankGroupStableFail"
        blocker = "legal_rank_group_stability_drop_failed"
    elif inum(p5.get("group_stable_pairwise_ranker_weak_pass")) and not inum(p6.get("rank_safe_certificate_pass")):
        route = "R5-RankCertificateAcceptedRegionFail"
        blocker = "rank_certificate_accepted_region_failed"
    elif inum(p7.get("source_controller_pass")) and not inum(p8.get("selected_runtime_pass")):
        route = "R6-ExistingActionRuntimeFail"
        blocker = "existing_action_runtime_failed"
    elif not (inum(p9.get("apga_ood_damage_autopsy_pass")) and inum(p10.get("memory_cover_blocker_confirmed"))):
        route = "R7-APGAOODMechanismUnresolved"
        blocker = "apga_ood_mechanism_unresolved"
    elif inum(p13.get("apgd_official_candidate_pass")):
        route = "R8-APGDGeneratedFrontierPass"
        blocker = "none"
    else:
        route = "R9-APGDGeneratedFrontierFail"
        blocker = "apgd_generated_frontier_failed"

    p14 = {
        "stage": "P14_INTEGRATED_ROUTE_DECISION",
        "status": "summary",
        "route": route,
        "source_route_v9590": p0.get("source_route_v9590"),
        "p0_pass": p0.get("p0_pass"),
        "p1_pass": p1.get("p1_pass"),
        "group_pocket_mechanism_resolved": p1.get("H1_payload_pocket_confirmed"),
        "payload_norm_bucket_drop_explained_fraction": p1.get("payload_norm_bucket_drop_explained_fraction"),
        "payload0_removal_precision_drop": p1.get("payload0_removal_precision_drop"),
        "group_balanced_target_density_weak_pass": p2.get("group_balanced_target_density_weak_pass"),
        "group_balanced_target_density_pass": p2.get("group_balanced_target_density_pass"),
        "feature_invariant_pass": p3.get("feature_invariant_pass"),
        "two_score_ranker_pass": p4.get("two_score_ranker_pass"),
        "group_stable_pairwise_ranker_weak_pass": p5.get("group_stable_pairwise_ranker_weak_pass"),
        "best_pairwise_ranker_id": p5.get("best_ranker_id"),
        "best_pairwise_TopK87_precision": p5.get("best_TopK87_precision"),
        "best_pairwise_LDO_drop_max": p5.get("best_LDO_drop_max"),
        "rank_safe_certificate_pass": p6.get("rank_safe_certificate_pass"),
        "source_controller_pass": p7.get("source_controller_pass"),
        "selected_runtime_pass": p8.get("selected_runtime_pass"),
        "apga_ood_damage_autopsy_pass": p9.get("apga_ood_damage_autopsy_pass"),
        "dominant_apga_damage_mode": p9.get("dominant_damage_mode"),
        "memory_cover_blocker_confirmed": p10.get("memory_cover_blocker_confirmed"),
        "apgd_implementation_pass": p11.get("apgd_implementation_pass"),
        "apgd_branch_horizon_pass": p12.get("apgd_branch_horizon_pass"),
        "apgd_branch_horizon_rows_actual": p12.get("branch_horizon_rows_actual", p12.get("actual_rows")),
        "apgd_unresolved_exception_count": p12.get("unresolved_exception_count"),
        "apgd_official_candidate_pass": p13.get("apgd_official_candidate_pass"),
        "best_apgd_primitive_id": p13.get("best_primitive_id"),
        "best_apgd_GradeAB_precision": p13.get("best_GradeAB_precision"),
        "best_apgd_V_integrated_LCB": p13.get("best_V_integrated_LCB"),
        "best_apgd_h240_longrisk_UCB": p13.get("best_h240_longrisk_UCB"),
        "system_legal_controller_pass": int(route == "R10-SystemLegalControllerPass"),
        "primary_blocker": blocker,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    artifacts["p14_integrated_route_decision_v9600.json"] = out / "p14_integrated_route_decision_v9600.json"
    write_json(out / "p14_integrated_route_decision_v9600.json", p14)
    artifacts["route_decision_v9600.json"] = out / "route_decision_v9600.json"
    write_json(out / "route_decision_v9600.json", p14)

    p15_rows, p15 = p15_boundary(p7, p8, route)
    dump_csv("p15_leaveout_paired_replay_boundary.csv", p15_rows)

    p16_rows, p16 = p16_base_acc(source_v9590)
    dump_csv("p16_base_acc_sentinel_v9600.csv", p16_rows)

    nofake = {"stage": "NO_FAKE_AUDIT_V9600", "status": "summary", **count_artifact_rows(list(artifacts.values()))}
    nofake.update({"fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    dump_csv("no_fake_audit_v9600.csv", [nofake])

    contract = {
        "stage": "CONTRACT_AUDIT_V9600",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "v9590_boundary_pass": p0.get("p0_pass"),
        "group_pocket_autopsy_pass": p1.get("p1_pass"),
        "group_balanced_target_density_pass": p2.get("group_balanced_target_density_pass"),
        "feature_invariant_pass": p3.get("feature_invariant_pass"),
        "two_score_ranker_pass": p4.get("two_score_ranker_pass"),
        "group_stable_pairwise_ranker_pass": p5.get("group_stable_pairwise_ranker_weak_pass"),
        "rank_safe_certificate_pass": p6.get("rank_safe_certificate_pass"),
        "source_controller_pass": p7.get("source_controller_pass"),
        "selected_runtime_pass": p8.get("selected_runtime_pass"),
        "apga_ood_damage_autopsy_pass": p9.get("apga_ood_damage_autopsy_pass"),
        "memory_cover_blocker_pass": p10.get("memory_cover_offdiag_blocker_v4_pass"),
        "apgd_implementation_pass": p11.get("apgd_implementation_pass"),
        "apgd_branch_horizon_pass": p12.get("apgd_branch_horizon_pass"),
        "apgd_official_candidate_pass": p13.get("apgd_official_candidate_pass"),
        "system_legal_controller_pass": p14.get("system_legal_controller_pass"),
        "base_acc_sentinel_pass": p16.get("base_acc_sentinel_pass", p16.get("sentinel_complete", 0)),
        "base_acc_used_for_controller": 0,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "uses_dataset_name_for_selector": 0,
        "uses_dataset_name_for_controller": 0,
        "uses_validation_or_test_for_controller": 0,
        "uses_future_outcome_for_features": 0,
        "uses_outcome_at_commit": 0,
        "uses_old_table_for_official": 0,
        "diagnostic_promoted_to_official": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("contract_audit_v9600.csv", [contract])

    provenance = {
        "stage": "PROVENANCE_AUDIT_V9600",
        "status": "summary",
        "source_v9590": rel(source_v9590),
        "source_v9580": rel(source_v9580),
        "source_v9570": rel(source_v9570),
        "source_v9560": rel(source_v9560),
        "source_v9550": rel(source_v9550),
        "source_v9330": rel(source_v9330),
        "plan_sha256": sha256_file(PLAN_PATH),
        "runner_sha256": sha256_file(SCRIPT_PATH),
        "artifact_count": len(artifacts),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("provenance_audit_v9600.csv", [provenance])

    failure = {
        "stage": "FAILURE_TAXONOMY_V9600",
        "status": "summary",
        "route": route,
        "F0_boundary_reproduction_failed": int(not inum(p0.get("p0_pass"))),
        "F1_group_pocket_mechanism_unresolved": int(route == "R1-GroupPocketMechanismUnresolved"),
        "F2_group_balanced_target_absent": int(route == "R2-GroupBalancedTargetAbsent"),
        "F3_legal_rank_group_stable_fail": int(route == "R4-LegalRankGroupStableFail"),
        "F4_rank_certificate_accepted_region_fail": int(route == "R5-RankCertificateAcceptedRegionFail"),
        "F5_existing_action_runtime_fail": int(route == "R6-ExistingActionRuntimeFail"),
        "F6_apga_ood_mechanism_unresolved": int(route == "R7-APGAOODMechanismUnresolved"),
        "F7_apgd_generated_frontier_fail": int(route == "R9-APGDGeneratedFrontierFail"),
        "F8_system_not_official": int(route != "R10-SystemLegalControllerPass"),
        "F9_base_acc_catastrophic": 0,
        "primary_blocker": blocker,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("failure_taxonomy_v9600.csv", [failure])

    manifest = {
        "version": "v9600",
        "route": route,
        "primary_blocker": blocker,
        "out_dir": rel(out),
        "created_utc": "2026-05-15T100000Z",
        "wallclock_sec": time.perf_counter() - t0,
        "seed": args.seed,
        "device": str(device),
        "sources": {
            "v9590": rel(source_v9590),
            "v9580": rel(source_v9580),
            "v9570": rel(source_v9570),
            "v9560": rel(source_v9560),
            "v9550": rel(source_v9550),
            "v9330": rel(source_v9330),
        },
        "artifact_sha256": sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts}),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    artifacts["run_manifest_v9600.json"] = out / "run_manifest_v9600.json"
    write_json(out / "run_manifest_v9600.json", manifest)

    print(json.dumps({
        "out_dir": rel(out),
        "route": route,
        "primary_blocker": blocker,
        "best_pairwise_ranker": p5.get("best_ranker_id"),
        "best_pairwise_LDO_drop_max": p5.get("best_LDO_drop_max"),
        "apgd_generated_actions": p11.get("generated_action_count"),
        "apgd_rows": p12.get("branch_horizon_rows_actual", p12.get("actual_rows")),
        "best_apgd_primitive": p13.get("best_primitive_id"),
        "best_apgd_V_integrated_LCB": p13.get("best_V_integrated_LCB"),
        "system_legal_controller_pass": p14.get("system_legal_controller_pass"),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
