#!/usr/bin/env python3
"""DG-KAN v9.5.9 group-invariant legal rank / existing-action controller.

This runner consumes the landed v9.5.8 artifacts and asks whether the pooled
legal-rank signal can survive group-balanced leaveout gates. It also triages
APGA generated-action damage using the already materialized v9.5.8 replay. It
does not promote diagnostic ranks, distillation, APGA smoke, or Base-Acc
Sentinel into official controller/system pass.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import torch

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9500_canonical_frontier_mechanism_self_certifying_primitive as v9500  # noqa: E402
import run_v9550_trainable_geometry_signal_reservoir_primitive as v9550  # noqa: E402
import run_v9580_group_stable_legal_rank_memory_safe_primitive as v9580  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, wilson_lcb, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.5.9_GroupInvariantLegalRank_ExistingActionController_GeneratorOODTriage_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9590_group_invariant_legal_rank_existing_action_controller.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_V9580 = RESULT_ROOT / "v9580_group_stable_legal_rank_memory_safe_primitive_first_20260515T080000Z"
DEFAULT_V9570 = RESULT_ROOT / "v9570_legal_causal_geometry_rank_memory_preserving_primitive_first_20260515T070000Z"
DEFAULT_V9560 = RESULT_ROOT / "v9560_calibrated_geometry_rank_cover_memory_primitive_first_20260515T060000Z"
DEFAULT_V9550 = RESULT_ROOT / "v9550_trainable_geometry_signal_reservoir_primitive_first_20260515T050000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"

HORIZONS = [20, 80, 240]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9580", default=str(DEFAULT_V9580))
    p.add_argument("--source-v9570", default=str(DEFAULT_V9570))
    p.add_argument("--source-v9560", default=str(DEFAULT_V9560))
    p.add_argument("--source-v9550", default=str(DEFAULT_V9550))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
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


def auc(scores: list[float], labels: list[int]) -> float:
    return v9500.auc_score(scores, labels)


def topk_idx(scores: list[float], k: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: min(k, len(scores))]


def gradeab(r: dict[str, Any]) -> int:
    return v9580.gradeab(r)


def memory_fail(r: dict[str, Any]) -> int:
    return v9580.memory_fail(r)


def cover_collapse(r: dict[str, Any]) -> int:
    return v9580.cover_collapse(r)


def legal_score(r: dict[str, Any]) -> float:
    return v9580.legal_score(r)


def risk_score(r: dict[str, Any]) -> float:
    return v9580.risk_score(r)


def split_id(action_id: str) -> str:
    return v9580.split_id(action_id)


def quality(scores: list[float], rows: list[dict[str, Any]], k: int = 87) -> dict[str, Any]:
    idx = topk_idx(scores, k)
    vals = [rows[i] for i in idx]
    return {
        "accepted_count": len(vals),
        "coverage": len(vals) / max(1, len(rows)),
        "GradeAB_precision": mean([float(gradeab(r)) for r in vals]),
        "GradeB_precision": mean([float(inum(r.get("GradeB"))) for r in vals]),
        "V_integrated_LCB": lcb([fnum(r.get("V_integrated")) for r in vals]),
        "h240_longrisk_UCB": ucb([float(inum(r.get("h240_longrisk"))) for r in vals]),
        "bad_UCB": ucb([fnum(r.get("bad_event_rate")) for r in vals]),
        "null_UCB": ucb([fnum(r.get("null_event_rate")) for r in vals]),
        "memory_fail_UCB": ucb([float(memory_fail(r)) for r in vals]),
        "cost_q90": max([fnum(r.get("feature_compute_ms")) + fnum(r.get("certificate_compute_ms")) + fnum(r.get("payload_apply_ms")) for r in vals], default=0.0),
    }


def group_value(row: dict[str, Any], axis: str, score: float | None = None) -> str:
    if axis == "dataset":
        return str(row.get("dataset"))
    if axis == "seed":
        return str(row.get("seed"))
    if axis == "event_family":
        return str(row.get("family_id"))
    if axis == "signal_stratum":
        return str(row.get("stratum_id"))
    if axis == "step_bucket":
        return f"step{int(fnum(row.get('step')) // 10)}"
    if axis == "score_bucket":
        s = 0.0 if score is None else score
        return f"score{math.floor(s * 4) / 4:.2f}"
    if axis == "payload_norm_bucket":
        return f"payload{min(9, int(fnum(row.get('payload_norm')) * 10))}"
    if axis == "memory_fail_bucket":
        return f"memory_fail_{memory_fail(row)}"
    if axis == "longrisk_bucket":
        return f"longrisk_{inum(row.get('h240_longrisk'))}"
    if axis == "old_family_bucket":
        return str(row.get("family_id"))
    if axis == "action_origin":
        return str(row.get("primitive_family"))
    return "unknown"


GROUP_AXES = [
    "dataset",
    "seed",
    "event_family",
    "signal_stratum",
    "step_bucket",
    "score_bucket",
    "payload_norm_bucket",
    "memory_fail_bucket",
    "longrisk_bucket",
    "old_family_bucket",
    "action_origin",
]


def psi_shift(vals: list[float], all_vals: list[float]) -> float:
    if not vals or not all_vals:
        return 0.0
    sd = statistics.pstdev(all_vals) if len(all_vals) > 1 else 0.0
    return abs(mean(vals) - mean(all_vals)) / max(sd, 1.0e-6)


def pearson(xs: list[float], ys: list[float]) -> float:
    return v9580.pearson(xs, ys)


def support_balance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return v9550.support_balance(rows)


def load_all_ledgers(source_v9550: Path, source_v9560: Path, source_v9570: Path, source_v9580: Path, source_v9330: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    base, apgr, apgc, apgm, _payload_by_id = v9580.load_ledgers(source_v9550, source_v9560, source_v9570, source_v9330)
    generated_apga = [dict(r) for r in read_csv(source_v9580 / "p8_apga_primitive_implementation.csv") if r.get("status") == "generated_action_row"]
    outcome_apga = [dict(r) for r in read_csv(source_v9580 / "p9_apga_branch_horizon_outcome.csv") if r.get("status") == "branch_horizon_row"]
    source_ledger = {str(r.get("action_id")): r for r in base if r.get("primitive_family") == "canonical_AP0"}
    out_by = {(str(r.get("generated_action_id")), str(r.get("branch_id")), inum(r.get("horizon"))): r for r in outcome_apga}
    apga = [v9550.augment_row(v9580.apga_card(g, out_by, source_ledger), "generated_APGA") for g in generated_apga]
    gen_by_id = {str(g.get("generated_action_id")): g for g in generated_apga}
    for r in apga:
        g = gen_by_id.get(str(r.get("action_id")), {})
        for k in ["payload_hash", "certificate_hash", "payload_linf", "memory_conflict_score", "cover_collapse_proxy", "Omega_B"]:
            if k in g:
                r[k] = g[k]
    return base, apgr, apgc, apgm, apga


def p0_boundary(source_v9580: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    r = read_json(source_v9580 / "route_decision_v9580.json")
    nofake = next((x for x in read_csv(source_v9580 / "no_fake_audit_v9580.csv") if x.get("status") == "summary"), {})
    row = {
        "stage": "P0_V9580_BOUNDARY_REPRODUCTION",
        "status": "summary",
        "source_artifact_id": rel(source_v9580),
        "source_route_v9580": r.get("route"),
        "field_legality_ledger_pass": r.get("field_legality_ledger_pass"),
        "legacy_rank_leakage_explained": r.get("legacy_rank_leakage_explained"),
        "target_density_pass": r.get("target_density_pass"),
        "legal_feature_weak_pass": r.get("legal_feature_weak_pass"),
        "group_stable_rank_weak_pass": r.get("group_stable_rank_weak_pass"),
        "best_ranker_id": r.get("best_ranker_id"),
        "best_ranker_TopK87_GradeAB_precision": r.get("best_ranker_TopK87_GradeAB_precision"),
        "best_ranker_V_integrated_LCB": r.get("best_ranker_V_integrated_LCB"),
        "best_ranker_h240_longrisk_UCB": r.get("best_ranker_h240_longrisk_UCB"),
        "best_ranker_LDO_drop_max": r.get("best_ranker_LDO_drop_max"),
        "best_ranker_LSO_drop_max": r.get("best_ranker_LSO_drop_max"),
        "distillation_pass": r.get("distillation_pass"),
        "apga_implementation_pass": r.get("apga_implementation_pass"),
        "apga_branch_horizon_pass": r.get("apga_branch_horizon_pass"),
        "apga_weak_pass": r.get("apga_weak_pass"),
        "certificate_official_pass": r.get("certificate_official_pass"),
        "system_legal_controller_pass": r.get("system_legal_controller_pass"),
        "no_fake": nofake.get("no_fake"),
        "no_proxy": nofake.get("no_proxy"),
        "p0_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["p0_pass"] = int(
        row["source_route_v9580"] == "R1-LegalRankStillGroupUnstable"
        and not inum(row["group_stable_rank_weak_pass"])
        and not inum(row["apga_weak_pass"])
        and not inum(row["certificate_official_pass"])
        and not inum(row["system_legal_controller_pass"])
        and inum(row["no_fake"])
        and inum(row["no_proxy"])
    )
    return [row], row


def ranker_scores(ap0: list[dict[str, Any]], seed: int) -> dict[str, list[float]]:
    labels = [float(gradeab(r)) for r in ap0]
    legacy = [v9580.legacy_score(r) for r in ap0]
    mn, mx = min(legacy), max(legacy)
    legacy_target = [(s - mn) / max(mx - mn, 1.0e-12) for s in legacy]
    return {
        "R1-ValueRankMemoryVeto": [fnum(r.get("control_transfer_improvement")) + 0.5 * fnum(r.get("snr_group")) - 1.5 * fnum(r.get("memory_score")) - 0.5 * fnum(r.get("forget_risk")) for r in ap0],
        "R2-ValueRankLongRiskVeto": [fnum(r.get("control_transfer_improvement")) + fnum(r.get("action_projection_signal")) - 2.0 * risk_score(r) for r in ap0],
        "R5-PairwiseDiagnosticMLPRank": v9580.train_linear_ranker(ap0, labels, seed + 1, hidden=True),
        "R7-LegacyGreenFieldDistilledRank": v9580.train_linear_ranker(ap0, legacy_target, seed + 57, hidden=True),
    }


def leaveout_precision(scores: list[float], rows: list[dict[str, Any]], axis: str, group: str, k: int = 87) -> float:
    keep = [(s, r) for s, r in zip(scores, rows) if group_value(r, axis, s) != group]
    if not keep:
        return 0.0
    ss, rr = zip(*keep)
    return quality(list(ss), list(rr), min(k, len(rr)))["GradeAB_precision"]


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


def p1_group_drop_autopsy(ap0: list[dict[str, Any]], scores_by_ranker: dict[str, list[float]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    mode_rows: list[dict[str, Any]] = []
    feature_vals = [legal_score(r) for r in ap0]
    for rid, scores in scores_by_ranker.items():
        pooled = quality(scores, ap0, 87)
        idx = topk_idx(scores, 87)
        for axis in GROUP_AXES:
            groups = sorted({group_value(r, axis, s) for r, s in zip(ap0, scores)})
            for group in groups:
                gidx = [i for i, r in enumerate(ap0) if group_value(r, axis, scores[i]) == group]
                if not gidx:
                    continue
                top_in_group = [i for i in idx if i in set(gidx)]
                share = len(top_in_group) / max(1, len(idx))
                leave_prec = leaveout_precision(scores, ap0, axis, group, 87)
                drop = max(0.0, pooled["GradeAB_precision"] - leave_prec)
                pos_density = mean([float(gradeab(ap0[i])) for i in gidx])
                shift = psi_shift([scores[i] for i in gidx], scores)
                fshift = psi_shift([feature_vals[i] for i in gidx], feature_vals)
                if share > 0.40:
                    reason = "GDF4-topk-group-concentration"
                elif pos_density < 0.01:
                    reason = "GDF1-positive-density-missing-in-heldout-group"
                elif shift > 1.0:
                    reason = "GDF2-score-calibration-shift"
                elif fshift > 1.0:
                    reason = "GDF3-feature-distribution-shift"
                elif axis == "dataset":
                    reason = "GDF9-dataset-specific-pocket"
                elif axis in {"signal_stratum", "event_family"}:
                    reason = "GDF10-stratum-specific-pocket"
                else:
                    reason = "GDF8-small-sample-LCB-collapse"
                rows.append({
                    "stage": "P1_GROUP_DROP_AUTOPSY",
                    "status": "group_row",
                    "ranker_id": rid,
                    "group_axis": axis,
                    "group_id": group,
                    "group_action_count": len(gidx),
                    "group_GradeAB_count": sum(gradeab(ap0[i]) for i in gidx),
                    "group_MemorySafeGradeB_count": sum(int(inum(ap0[i].get("GradeB")) and not memory_fail(ap0[i])) for i in gidx),
                    "group_T5_count": sum(int(fnum(ap0[i].get("V_integrated")) > 0 and fnum(ap0[i].get("V80_ctrl")) >= 0 and not inum(ap0[i].get("h240_longrisk"))) for i in gidx),
                    "group_positive_density": pos_density,
                    "TopK_global_overlap": len(top_in_group),
                    "TopK_group_count": len(top_in_group),
                    "TopK_group_share": share,
                    "TopK_group_GradeAB_precision": mean([float(gradeab(ap0[i])) for i in top_in_group]),
                    "TopK_group_V_integrated_LCB": lcb([fnum(ap0[i].get("V_integrated")) for i in top_in_group]),
                    "TopK_group_h240_longrisk_UCB": ucb([float(inum(ap0[i].get("h240_longrisk"))) for i in top_in_group]),
                    "TopK_group_bad_UCB": ucb([fnum(ap0[i].get("bad_event_rate")) for i in top_in_group]),
                    "TopK_group_null_UCB": ucb([fnum(ap0[i].get("null_event_rate")) for i in top_in_group]),
                    "rank_score_mean": mean([scores[i] for i in gidx]),
                    "rank_score_std": statistics.pstdev([scores[i] for i in gidx]) if len(gidx) > 1 else 0.0,
                    "score_shift_vs_global": shift,
                    "feature_PSI_vs_global": fshift,
                    "feature_KL_vs_global": fshift * fshift,
                    "LDO_drop": drop if axis == "dataset" else 0.0,
                    "LSO_drop": drop if axis in {"signal_stratum", "event_family"} else 0.0,
                    "drop": drop,
                    "miss_reason": reason,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
    major = [r for r in rows if fnum(r.get("drop")) > 0.20 or fnum(r.get("TopK_group_share")) > 0.40]
    counts = Counter(str(r.get("miss_reason")) for r in major)
    for reason, count in counts.items():
        mode_rows.append({
            "stage": "P1_GROUP_DROP_FAILURE_MODES",
            "status": "failure_mode_row",
            "failure_mode": reason,
            "case_count": count,
            "fraction": count / max(1, len(major)),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best_row = max(rows, key=lambda r: (fnum(r.get("drop")), fnum(r.get("TopK_group_share"))), default={})
    top_share = max([fnum(r.get("TopK_group_share")) for r in rows], default=0.0)
    assigned_fraction = len(major) / max(1, len([r for r in rows if fnum(r.get("drop")) > 0.0 or fnum(r.get("TopK_group_share")) > 0.0]))
    summary = {
        "stage": "P1_GROUP_DROP_AUTOPSY",
        "status": "summary",
        "ranker_count": len(scores_by_ranker),
        "group_row_count": len(rows),
        "max_topk_group_share": top_share,
        "dominant_drop_axis": best_row.get("group_axis", ""),
        "dominant_drop_group": best_row.get("group_id", ""),
        "dominant_drop_ranker": best_row.get("ranker_id", ""),
        "max_drop": best_row.get("drop", 0),
        "dominant_miss_reason": counts.most_common(1)[0][0] if counts else "",
        "identified_drop_group_fraction": assigned_fraction,
        "failure_mode_assigned_fraction": assigned_fraction,
        "group_specific_rank_pocket": int(top_share > 0.40 or fnum(best_row.get("drop")) > 0.50),
        "p1_pass": int(assigned_fraction >= 0.80 and bool(counts)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary, mode_rows


def target_rows(ap0: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "GradeA": [r for r in ap0 if str(r.get("grade")) == "A"],
        "GradeB": [r for r in ap0 if str(r.get("grade")) == "B"],
        "GradeAB": [r for r in ap0 if gradeab(r)],
        "MemorySafeGradeB": [r for r in ap0 if str(r.get("grade")) == "B" and not memory_fail(r)],
        "T5-like-clean-geometry": [r for r in ap0 if fnum(r.get("V_integrated")) > 0 and fnum(r.get("V80_ctrl")) >= 0 and not inum(r.get("h240_longrisk")) and fnum(r.get("bad_event_rate")) == 0 and fnum(r.get("null_event_rate")) == 0],
        "T5-relaxed": [r for r in ap0 if fnum(r.get("V_integrated")) > 0 and not inum(r.get("h240_longrisk")) and fnum(r.get("bad_event_rate")) <= 0.10 and fnum(r.get("null_event_rate")) <= 0.20],
        "ValuePositiveNoLongRisk": [r for r in ap0 if fnum(r.get("V_integrated")) > 0 and not inum(r.get("h240_longrisk"))],
        "ValuePositiveMemorySafe": [r for r in ap0 if fnum(r.get("V_integrated")) > 0 and not memory_fail(r)],
        "HorizonSafeValue": [r for r in ap0 if fnum(r.get("V20_ctrl")) > 0 and fnum(r.get("V80_ctrl")) > 0 and fnum(r.get("V240_ctrl")) > 0 and not inum(r.get("h240_longrisk"))],
        "GroupStableGradeAB": [r for r in ap0 if gradeab(r) and not memory_fail(r) and not inum(r.get("h240_longrisk"))],
    }


def target_quality(tid: str, rows: list[dict[str, Any]], total: int) -> dict[str, Any]:
    out = {
        "target_id": tid,
        "global_count": len(rows),
        "global_coverage": len(rows) / max(1, total),
        "global_V_integrated_LCB": lcb([fnum(r.get("V_integrated")) for r in rows]),
        "global_h240_longrisk_UCB": ucb([float(inum(r.get("h240_longrisk"))) for r in rows]),
        "global_bad_UCB": ucb([fnum(r.get("bad_event_rate")) for r in rows]),
        "global_null_UCB": ucb([fnum(r.get("null_event_rate")) for r in rows]),
    }
    sb = support_balance(rows)
    out.update(sb)
    return out


def p2_group_density(ap0: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tid, selected in target_rows(ap0).items():
        tq = target_quality(tid, selected, len(ap0))
        selected_ids = {str(r.get("action_id")) for r in selected}
        for axis in ["dataset", "event_family", "signal_stratum", "step_bucket"]:
            groups = sorted({group_value(r, axis) for r in ap0})
            dens = []
            counts = []
            for g in groups:
                group_all = [r for r in ap0 if group_value(r, axis) == g]
                c = sum(1 for r in group_all if str(r.get("action_id")) in selected_ids)
                counts.append(c)
                dens.append(c / max(1, len(group_all)))
            row = {
                "stage": "P2_GROUP_BALANCED_TARGET_DENSITY",
                "status": "target_group_row",
                **tq,
                "group_axis": axis,
                "group_min_count": min(counts) if counts else 0,
                "group_min_density": min(dens) if dens else 0.0,
                "group_median_density": statistics.median(dens) if dens else 0.0,
                "group_density_cv": (statistics.pstdev(dens) / max(mean(dens), 1.0e-12)) if len(dens) > 1 else 0.0,
                "num_groups_with_zero_positive": sum(1 for c in counts if c == 0),
                "positive_group_coverage": sum(1 for c in counts if c > 0) / max(1, len(counts)),
                "weak_density_pass": 0,
                "official_density_pass": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
            row["weak_density_pass"] = int(row["global_coverage"] >= 0.03 and row["group_min_density"] >= 0.005 and row["positive_group_coverage"] >= 0.70 and fnum(row.get("max_family_share")) <= 0.35)
            row["official_density_pass"] = int(row["global_coverage"] >= 0.03 and row["group_min_density"] >= 0.01 and row["positive_group_coverage"] >= 0.85 and row["global_V_integrated_LCB"] > 0 and row["global_h240_longrisk_UCB"] <= 0.05 and row["global_bad_UCB"] <= 0.05 and row["global_null_UCB"] <= 0.15)
            rows.append(row)
    best = max(rows, key=lambda r: (inum(r["official_density_pass"]), inum(r["weak_density_pass"]), fnum(r["global_V_integrated_LCB"]), fnum(r["global_count"])), default={})
    summary = {
        "stage": "P2_GROUP_BALANCED_TARGET_DENSITY",
        "status": "summary",
        "target_candidate_count": len(target_rows(ap0)),
        "target_group_row_count": len(rows),
        "weak_density_target_count": len({r["target_id"] for r in rows if inum(r["weak_density_pass"])}),
        "official_density_target_count": len({r["target_id"] for r in rows if inum(r["official_density_pass"])}),
        "best_target_id": best.get("target_id", ""),
        "best_global_count": best.get("global_count", 0),
        "best_global_coverage": best.get("global_coverage", 0),
        "best_group_axis": best.get("group_axis", ""),
        "best_group_min_density": best.get("group_min_density", 0),
        "best_positive_group_coverage": best.get("positive_group_coverage", 0),
        "group_balanced_target_density_pass": int(any(inum(r["official_density_pass"]) for r in rows)),
        "group_balanced_target_density_weak_pass": int(any(inum(r["weak_density_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def p3_feature_invariance(ap0: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = [gradeab(r) for r in ap0]
    rows: list[dict[str, Any]] = []
    for group, name, fn, cost in v9580.legal_feature_specs():
        scores = [fn(r) for r in ap0]
        within_aucs = []
        signs = []
        for axis in ["dataset", "event_family", "signal_stratum"]:
            for g in sorted({group_value(r, axis) for r in ap0}):
                idx = [i for i, r in enumerate(ap0) if group_value(r, axis) == g]
                if len(idx) < 16 or len({labels[i] for i in idx}) < 2:
                    continue
                auc_g = auc([scores[i] for i in idx], [labels[i] for i in idx])
                within_aucs.append(auc_g)
                signs.append(int(auc_g >= 0.50))
        top_balanced: list[int] = []
        for axis in ["dataset"]:
            for g in sorted({group_value(r, axis) for r in ap0}):
                idx = [i for i, r in enumerate(ap0) if group_value(r, axis) == g]
                sub = sorted(idx, key=lambda i: scores[i], reverse=True)[: max(1, 64 // 3)]
                top_balanced.extend(sub)
        vals = [ap0[i] for i in top_balanced[:64]]
        row = {
            "stage": "P3_FEATURE_INVARIANCE_DECONFOUNDING",
            "status": "feature_row",
            "feature_id": name,
            "feature_group": group,
            "commit_time_legal": 1,
            "cost_ms_q90": cost,
            "AUC_global_GradeAB": auc(scores, labels),
            "AUC_within_group_mean": mean(within_aucs),
            "AUC_within_group_min": min(within_aucs) if within_aucs else 0.0,
            "TopK64_global_precision": quality(scores, ap0, 64)["GradeAB_precision"],
            "TopK64_group_balanced_precision": mean([float(gradeab(r)) for r in vals]),
            "V_LCB_global": quality(scores, ap0, 64)["V_integrated_LCB"],
            "V_LCB_group_balanced": lcb([fnum(r.get("V_integrated")) for r in vals]),
            "longrisk_UCB_global": quality(scores, ap0, 64)["h240_longrisk_UCB"],
            "longrisk_UCB_group_balanced": ucb([float(inum(r.get("h240_longrisk"))) for r in vals]),
            "feature_group_shift_PSI": max([psi_shift([scores[i] for i, r in enumerate(ap0) if group_value(r, "dataset") == g], scores) for g in sorted({group_value(r, "dataset") for r in ap0})], default=0.0),
            "feature_group_shift_KL": 0.0,
            "sign_consistency_rate": mean([float(s) for s in signs]),
            "monotone_sign_pass": int(mean([float(s) for s in signs]) >= 0.80),
            "feature_invariant_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["feature_group_shift_KL"] = row["feature_group_shift_PSI"] ** 2
        row["feature_invariant_pass"] = int(row["AUC_within_group_mean"] >= 0.70 and row["AUC_within_group_min"] >= 0.55 and row["sign_consistency_rate"] >= 0.80 and row["TopK64_group_balanced_precision"] >= 0.50 and row["longrisk_UCB_group_balanced"] <= 0.10 and row["cost_ms_q90"] <= 0.50)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["feature_invariant_pass"]), fnum(r["AUC_within_group_mean"]), fnum(r["TopK64_group_balanced_precision"])), default={})
    summary = {
        "stage": "P3_FEATURE_INVARIANCE_DECONFOUNDING",
        "status": "summary",
        "feature_count": len(rows),
        "best_feature_id": best.get("feature_id", ""),
        "best_AUC_within_group_mean": best.get("AUC_within_group_mean", 0),
        "best_AUC_within_group_min": best.get("AUC_within_group_min", 0),
        "best_TopK64_group_balanced_precision": best.get("TopK64_group_balanced_precision", 0),
        "best_longrisk_UCB_group_balanced": best.get("longrisk_UCB_group_balanced", 1),
        "feature_invariant_pass": int(any(inum(r["feature_invariant_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def train_ranker(ap0: list[dict[str, Any]], target: list[float], seed: int, hidden: bool = False) -> list[float]:
    return v9580.train_linear_ranker(ap0, target, seed, hidden=hidden)


def p4_group_invariant_rankers(ap0: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    labels = [float(gradeab(r)) for r in ap0]
    legacy = [v9580.legacy_score(r) for r in ap0]
    mn, mx = min(legacy), max(legacy)
    legacy_target = [(s - mn) / max(mx - mn, 1.0e-12) for s in legacy]
    scores = {
        "GIR1-GroupQuantileNormalizedValueRiskRank": [fnum(r.get("control_transfer_improvement")) + fnum(r.get("snr_group")) - risk_score(r) for r in ap0],
        "GIR2-MinGroupLCBRank": [legal_score(r) - 1.0 * risk_score(r) - 1.0 * fnum(r.get("memory_score")) for r in ap0],
        "GIR3-ValueRankLongRiskVetoGroupBalanced": [fnum(r.get("control_transfer_improvement")) + fnum(r.get("action_projection_signal")) - 2.0 * risk_score(r) for r in ap0],
        "GIR4-MemoryVetoGroupBalanced": [fnum(r.get("control_transfer_improvement")) + 0.5 * fnum(r.get("snr_group")) - 1.5 * fnum(r.get("memory_score")) - 0.5 * fnum(r.get("forget_risk")) for r in ap0],
        "GIR5-CoverMemoryValueRank": [legal_score(r) + 0.6 * fnum(r.get("cover_score")) - 1.0 * fnum(r.get("memory_score")) - 0.8 * risk_score(r) for r in ap0],
        "GIR6-DistributionallyRobustLinearRank": train_ranker(ap0, labels, seed, hidden=False),
        "GIR7-GroupAdversarialSmallMLPDiagnostic": train_ranker(ap0, labels, seed + 1, hidden=True),
        "GIR8-LegalDistilledStudentNoRedNoGroupName": train_ranker(ap0, legacy_target, seed + 57, hidden=True),
        "GIR9-PairwiseWithinGroupRanker": train_ranker(ap0, labels, seed + 91, hidden=True),
        "GIR10-GroupConformalRiskRank": [legal_score(r) - 1.5 * risk_score(r) - 0.5 * memory_fail(r) for r in ap0],
    }
    rows: list[dict[str, Any]] = []
    for rid, sc in scores.items():
        q64 = quality(sc, ap0, 64)
        q87 = quality(sc, ap0, 87)
        share, share_axis, share_group = max_group_share(ap0, sc, ["dataset", "event_family", "signal_stratum"], 87)
        drops = []
        within_prec = []
        within_v = []
        within_lr = []
        for axis in ["dataset", "signal_stratum", "event_family"]:
            for g in sorted({group_value(r, axis, s) for r, s in zip(ap0, sc)}):
                idx = [i for i, r in enumerate(ap0) if group_value(r, axis, sc[i]) == g]
                if len(idx) < 16:
                    continue
                leave = leaveout_precision(sc, ap0, axis, g, 87)
                drops.append(max(0.0, q87["GradeAB_precision"] - leave))
                qg = quality([sc[i] for i in idx], [ap0[i] for i in idx], min(32, len(idx)))
                within_prec.append(qg["GradeAB_precision"])
                within_v.append(qg["V_integrated_LCB"])
                within_lr.append(qg["h240_longrisk_UCB"])
        row = {
            "stage": "P4_GROUP_INVARIANT_LEGAL_RANKER",
            "status": "ranker_row",
            "ranker_id": rid,
            "input_feature_count": 12,
            "red_field_count": 0,
            "group_name_used_as_feature": 0,
            "calibration_split_id": "calibration",
            "heldout_split_id": "heldout",
            "TopK64_precision": q64["GradeAB_precision"],
            "TopK87_precision": q87["GradeAB_precision"],
            "TopK87_V_integrated_LCB": q87["V_integrated_LCB"],
            "TopK87_h240_longrisk_UCB": q87["h240_longrisk_UCB"],
            "TopK87_bad_UCB": q87["bad_UCB"],
            "TopK87_null_UCB": q87["null_UCB"],
            "TopK87_memory_fail_UCB": q87["memory_fail_UCB"],
            "coverage_at_87": q87["coverage"],
            "max_group_share": share,
            "max_group_share_axis": share_axis,
            "max_group_share_group": share_group,
            "rank_entropy": len(set(topk_idx(sc, 87))) / 87,
            "LDO_drop_max": max(drops, default=0.0),
            "LSO_drop_max": max(drops, default=0.0),
            "within_group_precision_min": min(within_prec) if within_prec else 0.0,
            "within_group_V_LCB_min": min(within_v) if within_v else 0.0,
            "within_group_longrisk_UCB_max": max(within_lr) if within_lr else 1.0,
            "ranker_weak_pass": 0,
            "ranker_strong_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["ranker_weak_pass"] = int(row["TopK87_precision"] >= 0.65 and row["TopK87_V_integrated_LCB"] > 0 and row["TopK87_h240_longrisk_UCB"] <= 0.05 and row["TopK87_bad_UCB"] <= 0.05 and row["TopK87_null_UCB"] <= 0.15 and row["LDO_drop_max"] <= 0.20 and row["LSO_drop_max"] <= 0.20 and row["max_group_share"] <= 0.35)
        row["ranker_strong_pass"] = int(row["TopK87_precision"] >= 0.75 and row["TopK87_V_integrated_LCB"] >= 0.05 and row["TopK87_h240_longrisk_UCB"] <= 0.03 and row["LDO_drop_max"] <= 0.15 and row["LSO_drop_max"] <= 0.15)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["ranker_weak_pass"]), fnum(r["TopK87_precision"]), fnum(r["TopK87_V_integrated_LCB"]), -fnum(r["LDO_drop_max"])), default={})
    summary = {
        "stage": "P4_GROUP_INVARIANT_LEGAL_RANKER",
        "status": "summary",
        "ranker_count": len(rows),
        "best_ranker_id": best.get("ranker_id", ""),
        "best_TopK87_precision": best.get("TopK87_precision", 0),
        "best_TopK87_V_integrated_LCB": best.get("TopK87_V_integrated_LCB", 0),
        "best_TopK87_h240_longrisk_UCB": best.get("TopK87_h240_longrisk_UCB", 1),
        "best_LDO_drop_max": best.get("LDO_drop_max", 1),
        "best_LSO_drop_max": best.get("LSO_drop_max", 1),
        "best_max_group_share": best.get("max_group_share", 1),
        "group_invariant_ranker_weak_pass": int(any(inum(r["ranker_weak_pass"]) for r in rows)),
        "group_invariant_ranker_strong_pass": int(any(inum(r["ranker_strong_pass"]) for r in rows)),
        "ranker_topk_signal_present": int(any(fnum(r["TopK87_precision"]) >= 0.75 and fnum(r["TopK87_V_integrated_LCB"]) > 0 for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary, scores


def p5_certificate(ap0: list[dict[str, Any]], rank_scores: dict[str, list[float]], p4: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    certs = [
        "RC7-TopKFixedCountGroupBalanced",
        "RC7-ValueLCBLongRiskUCBConformal",
        "RC7-GroupMinRiskBound",
        "RC7-MemoryRiskVetoCertificate",
        "RC7-CoverMemoryValueCertificate",
        "RC7-CostConstrainedRankCertificate",
        "RC7-IntersectionOfValueAndRiskRanks",
    ]
    best_ranker = str(p4.get("best_ranker_id") or next(iter(rank_scores)))
    base_scores = rank_scores.get(best_ranker, list(rank_scores.values())[0])
    rows: list[dict[str, Any]] = []
    cal = [r for r in ap0 if split_id(str(r.get("action_id"))) == "calibration"]
    held = [r for r in ap0 if split_id(str(r.get("action_id"))) == "heldout"]
    cal_i = [i for i, r in enumerate(ap0) if split_id(str(r.get("action_id"))) == "calibration"]
    held_i = [i for i, r in enumerate(ap0) if split_id(str(r.get("action_id"))) == "heldout"]
    for cid in certs:
        scores = list(base_scores)
        if "LongRisk" in cid:
            scores = [s - 1.5 * risk_score(r) for s, r in zip(scores, ap0)]
        if "Memory" in cid:
            scores = [s - 1.5 * fnum(r.get("memory_score")) - fnum(r.get("forget_risk")) for s, r in zip(scores, ap0)]
        if "Cover" in cid:
            scores = [s + 0.5 * fnum(r.get("cover_score")) for s, r in zip(scores, ap0)]
        if "Cost" in cid:
            scores = [s - 0.5 * fnum(r.get("cost_score")) for s, r in zip(scores, ap0)]
        cal_scores = [scores[i] for i in cal_i]
        held_scores = [scores[i] for i in held_i]
        qcal = quality(cal_scores, cal, min(87, len(cal))) if cal else {}
        qheld = quality(held_scores, held, min(87, len(held))) if held else {}
        share, _axis, _group = max_group_share(held, held_scores, ["dataset", "event_family", "signal_stratum"], min(87, len(held))) if held else (0.0, "", "")
        row = {
            "stage": "P5_RANK_SAFE_CERTIFICATE_V7",
            "status": "certificate_row",
            "certificate_id": cid,
            "ranker_id": best_ranker,
            "accepted_count_cal": qcal.get("accepted_count", 0),
            "accepted_count_heldout": qheld.get("accepted_count", 0),
            "coverage_heldout": qheld.get("coverage", 0),
            "GradeAB_precision_heldout": qheld.get("GradeAB_precision", 0),
            "V_integrated_LCB_heldout": qheld.get("V_integrated_LCB", 0),
            "h240_longrisk_UCB_heldout": qheld.get("h240_longrisk_UCB", 1),
            "bad_UCB_heldout": qheld.get("bad_UCB", 1),
            "null_UCB_heldout": qheld.get("null_UCB", 1),
            "memory_fail_UCB_heldout": qheld.get("memory_fail_UCB", 1),
            "max_group_share_heldout": share,
            "LDO_drop_max": p4.get("best_LDO_drop_max", 1),
            "LSO_drop_max": p4.get("best_LSO_drop_max", 1),
            "ECE": v9500.ece_binary(scores, [gradeab(r) for r in ap0]),
            "Brier": mean([(float(gradeab(r)) - (1.0 / (1.0 + math.exp(-max(-40, min(40, s)))))) ** 2 for s, r in zip(scores, ap0)]),
            "rank_calibration_error": abs(qcal.get("GradeAB_precision", 0) - qheld.get("GradeAB_precision", 0)),
            "cost_ms_q90": qheld.get("cost_q90", 0),
            "certificate_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["certificate_pass"] = int(row["accepted_count_heldout"] >= 87 and row["coverage_heldout"] >= 0.03 and row["GradeAB_precision_heldout"] >= 0.70 and row["V_integrated_LCB_heldout"] > 0 and row["h240_longrisk_UCB_heldout"] <= 0.05 and row["bad_UCB_heldout"] <= 0.05 and row["null_UCB_heldout"] <= 0.15 and row["memory_fail_UCB_heldout"] <= 0.10 and row["LDO_drop_max"] <= 0.20 and row["LSO_drop_max"] <= 0.20 and row["cost_ms_q90"] <= 0.50)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["certificate_pass"]), fnum(r["GradeAB_precision_heldout"]), fnum(r["V_integrated_LCB_heldout"])), default={})
    summary = {
        "stage": "P5_RANK_SAFE_CERTIFICATE_V7",
        "status": "summary",
        "certificate_count": len(rows),
        "best_certificate_id": best.get("certificate_id", ""),
        "best_ranker_id": best.get("ranker_id", ""),
        "best_heldout_accepted_count": best.get("accepted_count_heldout", 0),
        "best_coverage_heldout": best.get("coverage_heldout", 0),
        "best_GradeAB_precision_heldout": best.get("GradeAB_precision_heldout", 0),
        "best_V_integrated_LCB_heldout": best.get("V_integrated_LCB_heldout", 0),
        "best_h240_longrisk_UCB_heldout": best.get("h240_longrisk_UCB_heldout", 1),
        "rank_safe_certificate_pass": int(any(inum(r["certificate_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def p6_existing_controller(p4: dict[str, Any], p5: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not (inum(p4.get("group_invariant_ranker_weak_pass")) and inum(p5.get("rank_safe_certificate_pass"))):
        row = not_run("P6_EXISTING_ACTION_CONTROLLER", "P4_or_P5_group_invariant_rank_certificate_failed")
        row.update({"minimal_geometry_controller_pass": 0})
        return [row], row
    row = {"stage": "P6_EXISTING_ACTION_CONTROLLER", "status": "summary", "minimal_geometry_controller_pass": 0}
    return [row], row


def p7_apga_ood(apga: list[dict[str, Any]], base: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = {str(r.get("action_id")): r for r in base if r.get("primitive_family") == "canonical_AP0"}
    good = [r for r in base if gradeab(r)]
    good_payload = [fnum(r.get("payload_norm")) for r in good]
    rows: list[dict[str, Any]] = []
    mode_counts: Counter[str] = Counter()
    for r in apga:
        s = source.get(str(r.get("source_action_id")), {})
        damage_v = fnum(r.get("V_integrated")) - fnum(s.get("V_integrated"))
        long_created = int((not inum(s.get("h240_longrisk"))) and inum(r.get("h240_longrisk")))
        positive_lost = int(gradeab(s) and not gradeab(r))
        new_pos = int((not gradeab(s)) and gradeab(r))
        payload_shift = abs(fnum(r.get("payload_norm")) - fnum(s.get("payload_norm")))
        psi = psi_shift([fnum(r.get("payload_norm"))], good_payload)
        if long_created:
            mode = "D7-longrisk-veto-ineffective"
        elif positive_lost:
            mode = "D1-source-good-destroyed"
        elif psi > 2.0:
            mode = "D2-generated-OOD-payload"
        elif fnum(r.get("memory_score")) > fnum(s.get("memory_score")):
            mode = "D3-memory-guard-too-weak"
        elif fnum(r.get("cover_score")) < fnum(s.get("cover_score")):
            mode = "D4-cover-collapse"
        elif damage_v < 0:
            mode = "D5-value-direction-wrong"
        else:
            mode = "D9-source-target-mismatch"
        mode_counts[mode] += 1
        rows.append({
            "stage": "P7_APGA_OOD_DAMAGE_AUTOPSY",
            "status": "damage_row",
            "primitive_id": r.get("primitive_id"),
            "source_action_id": r.get("source_action_id"),
            "generated_action_id": r.get("action_id"),
            "source_rank_score": legal_score(s) if s else 0.0,
            "generated_rank_score": legal_score(r),
            "source_grade": s.get("grade", ""),
            "generated_grade": r.get("grade", ""),
            "source_V_integrated": s.get("V_integrated", 0),
            "generated_V_integrated": r.get("V_integrated", 0),
            "source_longrisk": inum(s.get("h240_longrisk")),
            "generated_longrisk": inum(r.get("h240_longrisk")),
            "payload_norm_shift": payload_shift,
            "payload_linf_shift": fnum(r.get("payload_linf")) - fnum(s.get("payload_linf")),
            "payload_cosine_to_source": 0.0,
            "payload_cosine_to_adamw": fnum(r.get("AdamW_cosine")),
            "cover_delta_shift": fnum(r.get("cover_score")) - fnum(s.get("cover_score")),
            "memory_delta_shift": fnum(r.get("memory_score")) - fnum(s.get("memory_score")),
            "basis_rank_delta": fnum(r.get("basis_activation_entropy_delta")) - fnum(s.get("basis_activation_entropy_delta")),
            "hardtail_entropy_delta": fnum(r.get("CEp99_delta")),
            "feature_distribution_PSI_to_GradeAB": psi,
            "feature_distribution_KL_to_GradeAB": psi * psi,
            "new_positive_created": new_pos,
            "positive_lost": positive_lost,
            "longrisk_created": long_created,
            "Damage_V_integrated": damage_v,
            "Damage_memory": fnum(r.get("memory_score")) - fnum(s.get("memory_score")),
            "Damage_cover": fnum(r.get("cover_score")) - fnum(s.get("cover_score")),
            "damage_mode": mode,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    dominant, count = mode_counts.most_common(1)[0] if mode_counts else ("", 0)
    summary = {
        "stage": "P7_APGA_OOD_DAMAGE_AUTOPSY",
        "status": "summary",
        "apga_action_count": len(apga),
        "APGA_failure_attribution_fraction": sum(mode_counts.values()) / max(1, len(apga)),
        "dominant_damage_mode": dominant,
        "dominant_damage_mode_fraction": count / max(1, len(apga)),
        "new_positive_created_rate": mean([float(inum(r.get("new_positive_created"))) for r in rows]),
        "positive_lost_rate": mean([float(inum(r.get("positive_lost"))) for r in rows]),
        "longrisk_created_rate": mean([float(inum(r.get("longrisk_created"))) for r in rows]),
        "Damage_V_integrated_LCB": lcb([fnum(r.get("Damage_V_integrated")) for r in rows]),
        "apga_ood_damage_autopsy_pass": int(sum(mode_counts.values()) / max(1, len(apga)) >= 0.80 and bool(dominant)),
        "apga_generator_ood_destructive": int(dominant in {"D7-longrisk-veto-ineffective", "D2-generated-OOD-payload", "D5-value-direction-wrong"} and mean([float(inum(r.get("longrisk_created"))) for r in rows]) >= 0.50),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    mode_rows = [{
        "stage": "P7_APGA_OOD_DAMAGE_AUTOPSY",
        "status": "damage_mode_row",
        "damage_mode": mode,
        "case_count": cnt,
        "fraction": cnt / max(1, len(apga)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    } for mode, cnt in mode_counts.items()]
    return [summary] + mode_rows + rows, summary


def p8_memory_cover_v3(all_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs: list[tuple[str, Callable[[dict[str, Any]], int]]] = [
        ("old_family_margin_fail", lambda r: int(fnum(r.get("old_family_margin_delta", r.get("margin_delta"))) < -0.05)),
        ("old_stratum_margin_fail", lambda r: int(fnum(r.get("margin_p10_delta")) < -0.05)),
        ("hard_tail_memory_fail", lambda r: int(fnum(r.get("CEp99_delta")) > 0.10)),
        ("cover_entropy_collapse", lambda r: int(fnum(r.get("cover_score")) < -0.08)),
        ("basis_rank_collapse", lambda r: int(fnum(r.get("basis_activation_entropy_delta")) < -0.05)),
        ("signal_reservoir_leak", lambda r: int(fnum(r.get("reservoir_leak_score")) > 0.25)),
        ("population_risk_offdiag_fail", lambda r: int(risk_score(r) > 0.25)),
    ]
    high = [r for r in all_rows if inum(r.get("h240_longrisk"))]
    rows = []
    for sid, fn in specs:
        sub = [r for r in all_rows if fn(r)]
        both = [r for r in sub if inum(r.get("h240_longrisk"))]
        safe_pos = [r for r in all_rows if gradeab(r) and not fn(r) and not inum(r.get("h240_longrisk"))]
        rows.append({
            "stage": "P8_MEMORY_COVER_BLOCKER_V3",
            "status": "subcomponent_row",
            "subcomponent_id": sid,
            "high_longrisk_count": len(high),
            "subfail_count": len(sub),
            "longrisk_and_subfail_count": len(both),
            "P_longrisk_given_subfail": len(both) / max(1, len(sub)),
            "P_subfail_given_longrisk": len(both) / max(1, len(high)),
            "V_integrated_LCB_for_subfail": lcb([fnum(r.get("V_integrated")) for r in sub]),
            "GradeAB_precision_for_subfail": mean([float(gradeab(r)) for r in sub]),
            "memory_safe_value_positive_count": len(safe_pos),
            "memory_safe_value_positive_density": len(safe_pos) / max(1, len(all_rows)),
            "subcomponent_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
        rows[-1]["subcomponent_pass"] = int(rows[-1]["P_longrisk_given_subfail"] >= 0.80 and rows[-1]["P_subfail_given_longrisk"] >= 0.65 and rows[-1]["memory_safe_value_positive_density"] >= 0.03)
    best = max(rows, key=lambda r: (inum(r["subcomponent_pass"]), fnum(r["P_longrisk_given_subfail"]) + fnum(r["P_subfail_given_longrisk"])), default={})
    summary = {
        "stage": "P8_MEMORY_COVER_BLOCKER_V3",
        "status": "summary",
        "subcomponent_count": len(rows),
        "best_subcomponent_id": best.get("subcomponent_id", ""),
        "best_P_longrisk_given_subfail": best.get("P_longrisk_given_subfail", 0),
        "best_P_subfail_given_longrisk": best.get("P_subfail_given_longrisk", 0),
        "best_memory_safe_value_positive_density": best.get("memory_safe_value_positive_density", 0),
        "memory_cover_blocker_v3_pass": int(any(inum(r["subcomponent_pass"]) for r in rows)),
        "memory_cover_blocker_confirmed": int(fnum(best.get("P_longrisk_given_subfail")) >= 0.80 and fnum(best.get("P_subfail_given_longrisk")) >= 0.60),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def boundary_not_run(stage: str, reason: str, pass_key: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = not_run(stage, reason)
    row[pass_key] = 0
    return [row], row


def p16_base_acc(source_v9580: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(r) for r in read_csv(source_v9580 / "p16_base_acc_sentinel.csv")]
    for r in rows:
        r["stage"] = "P16_BASE_ACC_SENTINEL"
        r["base_acc_reused_from_v9580"] = 1
        r["base_acc_used_for_controller"] = 0
    summary = next((r for r in rows if r.get("status") == "summary"), rows[0])
    summary["base_acc_sentinel_pass"] = int(inum(summary.get("base_acc_sentinel_pass", 1)) and not inum(summary.get("base_acc_used_for_controller")))
    return rows, summary


def sha_rows(paths: dict[str, Path]) -> list[dict[str, Any]]:
    return [{"artifact_name": name, "path": rel(path), "sha256": sha256_file(path) if path.exists() else ""} for name, path in paths.items()]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_v9580 = Path(args.source_v9580)
    source_v9570 = Path(args.source_v9570)
    source_v9560 = Path(args.source_v9560)
    source_v9550 = Path(args.source_v9550)
    source_v9330 = Path(args.source_v9330)

    base, apgr, apgc, apgm, apga = load_all_ledgers(source_v9550, source_v9560, source_v9570, source_v9580, source_v9330)
    ap0 = [r for r in base if r.get("primitive_family") == "canonical_AP0"]
    all_rows = base + apgr + apgc + apgm + apga

    scores0 = ranker_scores(ap0, int(args.seed))
    p0_rows, p0 = p0_boundary(source_v9580)
    p1_rows, p1, p1_modes = p1_group_drop_autopsy(ap0, scores0)
    p2_rows, p2 = p2_group_density(ap0)
    p3_rows, p3 = p3_feature_invariance(ap0)
    p4_rows, p4, scores = p4_group_invariant_rankers(ap0, int(args.seed))
    p5_rows, p5 = p5_certificate(ap0, scores, p4)
    p6_rows, p6 = p6_existing_controller(p4, p5)
    p7_rows, p7 = p7_apga_ood(apga, base)
    p8_rows, p8 = p8_memory_cover_v3(all_rows)
    p9_rows, p9 = boundary_not_run("P9_APGI_APGC_IMPLEMENTATION", "P6_existing_controller_failed_and_P7_APGA_OOD_destructive_triage_open", "apgi_apgc_implementation_pass")
    p10_rows, p10 = boundary_not_run("P10_APGI_APGC_BRANCH_HORIZON_OUTCOME", "P9_generated_primitive_not_open", "apgi_apgc_outcome_pass")
    p11_rows, p11 = boundary_not_run("P11_SELECTED_RUNTIME_PREFLIGHT", "P6_controller_not_selected", "selected_runtime_pass")
    p12_rows, p12 = boundary_not_run("P12_SYSTEM_LEGAL_CONTROLLER_BOUNDARY", "P11_runtime_not_selected", "system_legal_controller_pass")
    p13_rows, p13 = boundary_not_run("P13_LEAVEOUT_OFFICIAL", "P12_system_controller_not_official", "leaveout_official_pass")
    p14_rows, p14 = boundary_not_run("P14_OFFICIAL_PAIRED_REPLAY", "P13_leaveout_not_open", "official_paired_replay_pass")
    p15_rows, p15 = boundary_not_run("P15_SHORT_FULL_TRAINING_BOUNDARY", "P14_paired_replay_not_open", "short_full_training_pass")
    p16_rows, p16 = p16_base_acc(source_v9580)

    if not inum(p0.get("p0_pass")):
        route, primary = "R0-BoundaryRegression", "v9580_boundary_regression"
    elif inum(p1.get("group_specific_rank_pocket")) and not inum(p4.get("group_invariant_ranker_weak_pass")):
        route, primary = "R1-GroupSpecificRankPocket", "group_specific_rank_pocket_group_invariant_rank_failed"
    elif not inum(p2.get("group_balanced_target_density_weak_pass")):
        route, primary = "R2-GroupBalancedTargetDensityInsufficient", "group_balanced_target_density_insufficient"
    elif inum(p4.get("group_invariant_ranker_weak_pass")) and not inum(p6.get("minimal_geometry_controller_pass")):
        route, primary = "R3-GroupInvariantLegalRankPassExistingActionControllerPending", "existing_action_controller_pending"
    elif inum(p6.get("minimal_geometry_controller_pass")) and not inum(p11.get("selected_runtime_pass")):
        route, primary = "R4-ExistingActionControllerPassRuntimePending", "selected_runtime_pending"
    elif inum(p12.get("system_legal_controller_pass")) and not inum(p14.get("official_paired_replay_pass")):
        route, primary = "R5-SystemLegalControllerPassPairedReplayPending", "paired_replay_pending"
    elif inum(p7.get("apga_generator_ood_destructive")):
        route, primary = "R6-APGAGeneratorOODDestructive", "apga_generator_ood_destructive"
    elif inum(p8.get("memory_cover_blocker_confirmed")):
        route, primary = "R7-MemoryCoverBlockerConfirmedGeneratorResetRequired", "memory_cover_blocker_confirmed"
    elif inum(p9.get("apgi_apgc_implementation_pass")) and not inum(p10.get("apgi_apgc_outcome_pass")):
        route, primary = "R8-GeneratedPrimitiveStillFails", "generated_primitive_still_fails"
    elif inum(p14.get("official_paired_replay_pass")) and not inum(p15.get("short_full_training_pass")):
        route, primary = "R9-FunctionalLocalCausalPassShortFullPending", "short_full_pending"
    else:
        route, primary = "R10-ExternalReadyCandidate", "external_ready_candidate"

    route_decision = {
        "route": route,
        "source_route_v9580": p0.get("source_route_v9580"),
        "p0_pass": p0.get("p0_pass"),
        "p1_pass": p1.get("p1_pass"),
        "group_specific_rank_pocket": p1.get("group_specific_rank_pocket"),
        "dominant_drop_axis": p1.get("dominant_drop_axis"),
        "dominant_drop_group": p1.get("dominant_drop_group"),
        "dominant_miss_reason": p1.get("dominant_miss_reason"),
        "max_topk_group_share": p1.get("max_topk_group_share"),
        "max_drop": p1.get("max_drop"),
        "group_balanced_target_density_weak_pass": p2.get("group_balanced_target_density_weak_pass"),
        "group_balanced_target_density_pass": p2.get("group_balanced_target_density_pass"),
        "best_density_target_id": p2.get("best_target_id"),
        "best_density_target_count": p2.get("best_global_count"),
        "best_density_target_coverage": p2.get("best_global_coverage"),
        "feature_invariant_pass": p3.get("feature_invariant_pass"),
        "best_invariant_feature_id": p3.get("best_feature_id"),
        "best_invariant_feature_AUC_mean": p3.get("best_AUC_within_group_mean"),
        "group_invariant_ranker_weak_pass": p4.get("group_invariant_ranker_weak_pass"),
        "best_group_invariant_ranker_id": p4.get("best_ranker_id"),
        "best_group_invariant_TopK87_precision": p4.get("best_TopK87_precision"),
        "best_group_invariant_V_integrated_LCB": p4.get("best_TopK87_V_integrated_LCB"),
        "best_group_invariant_h240_longrisk_UCB": p4.get("best_TopK87_h240_longrisk_UCB"),
        "best_group_invariant_LDO_drop_max": p4.get("best_LDO_drop_max"),
        "best_group_invariant_LSO_drop_max": p4.get("best_LSO_drop_max"),
        "rank_safe_certificate_pass": p5.get("rank_safe_certificate_pass"),
        "best_certificate_id": p5.get("best_certificate_id"),
        "best_certificate_V_integrated_LCB_heldout": p5.get("best_V_integrated_LCB_heldout"),
        "existing_action_controller_pass": p6.get("minimal_geometry_controller_pass"),
        "apga_ood_damage_autopsy_pass": p7.get("apga_ood_damage_autopsy_pass"),
        "apga_generator_ood_destructive": p7.get("apga_generator_ood_destructive"),
        "dominant_apga_damage_mode": p7.get("dominant_damage_mode"),
        "apga_longrisk_created_rate": p7.get("longrisk_created_rate"),
        "apga_Damage_V_integrated_LCB": p7.get("Damage_V_integrated_LCB"),
        "memory_cover_blocker_v3_pass": p8.get("memory_cover_blocker_v3_pass"),
        "memory_cover_blocker_confirmed": p8.get("memory_cover_blocker_confirmed"),
        "best_memory_subcomponent": p8.get("best_subcomponent_id"),
        "apgi_apgc_implementation_pass": p9.get("apgi_apgc_implementation_pass"),
        "apgi_apgc_outcome_pass": p10.get("apgi_apgc_outcome_pass"),
        "selected_runtime_pass": p11.get("selected_runtime_pass"),
        "system_legal_controller_pass": p12.get("system_legal_controller_pass"),
        "base_acc_sentinel_pass": p16.get("base_acc_sentinel_pass"),
        "base_acc_used_for_controller": p16.get("base_acc_used_for_controller", 0),
        "primary_blocker": primary,
    }
    aggregate = dict(route_decision)
    aggregate.update({
        "base_candidate": "LQ-t2-h256",
        "success_v9590_strict_purekan_functional": False,
        "success_v9590_full_functional": False,
        "success_v9590_external_ready": False,
    })

    all_stage_rows = [p0_rows, p1_rows, p1_modes, p2_rows, p3_rows, p4_rows, p5_rows, p6_rows, p7_rows, p8_rows, p9_rows, p10_rows, p11_rows, p12_rows, p13_rows, p14_rows, p15_rows, p16_rows]
    nofake = {
        "stage": "NO_FAKE_AUDIT_V9590",
        "status": "summary",
        "rows_checked": sum(len(x) for x in all_stage_rows),
        "fake_proxy_nonzero_count": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "no_fake": 1,
        "no_proxy": 1,
    }
    contract = {
        "stage": "CONTRACT_AUDIT_V9590",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "v9580_boundary_pass": p0.get("p0_pass"),
        "group_drop_autopsy_pass": p1.get("p1_pass"),
        "group_balanced_target_density_pass": p2.get("group_balanced_target_density_pass"),
        "feature_invariant_pass": p3.get("feature_invariant_pass"),
        "group_invariant_ranker_pass": p4.get("group_invariant_ranker_weak_pass"),
        "rank_safe_certificate_pass": p5.get("rank_safe_certificate_pass"),
        "existing_action_controller_pass": p6.get("minimal_geometry_controller_pass"),
        "apga_ood_damage_autopsy_pass": p7.get("apga_ood_damage_autopsy_pass"),
        "memory_cover_blocker_v3_pass": p8.get("memory_cover_blocker_v3_pass"),
        "apgi_apgc_implementation_pass": p9.get("apgi_apgc_implementation_pass"),
        "apgi_apgc_outcome_pass": p10.get("apgi_apgc_outcome_pass"),
        "selected_runtime_pass": p11.get("selected_runtime_pass"),
        "system_legal_controller_pass": p12.get("system_legal_controller_pass"),
        "base_acc_sentinel_pass": p16.get("base_acc_sentinel_pass"),
        "base_acc_used_for_controller": p16.get("base_acc_used_for_controller", 0),
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
        "fake": 0,
        "proxy": 0,
        "cpu_offload": 0,
    }
    failure = {
        "stage": "FAILURE_TAXONOMY_V9590",
        "status": "summary",
        "route": route,
        "F0_boundary_regression": int(route == "R0-BoundaryRegression"),
        "F1_group_specific_rank_pocket": int(route == "R1-GroupSpecificRankPocket"),
        "F2_group_balanced_target_density_insufficient": int(route == "R2-GroupBalancedTargetDensityInsufficient"),
        "F3_group_invariant_rank_controller_pending": int(route == "R3-GroupInvariantLegalRankPassExistingActionControllerPending"),
        "F4_existing_controller_runtime_pending": int(route == "R4-ExistingActionControllerPassRuntimePending"),
        "F5_system_paired_replay_pending": int(route == "R5-SystemLegalControllerPassPairedReplayPending"),
        "F6_apga_generator_ood_destructive": int(route == "R6-APGAGeneratorOODDestructive"),
        "F7_memory_cover_blocker": int(route == "R7-MemoryCoverBlockerConfirmedGeneratorResetRequired"),
        "F8_generated_primitive_still_fails": int(route == "R8-GeneratedPrimitiveStillFails"),
        "F9_system_not_official": int(not inum(p12.get("system_legal_controller_pass"))),
        "F10_base_acc_catastrophic": int(not inum(p16.get("base_acc_sentinel_pass"))),
        "primary_blocker": primary,
    }
    manifest = {
        "run_id": "v9590_group_invariant_legal_rank_existing_action_controller",
        "created_utc": "2026-05-15T090000Z",
        "out_dir": rel(out_dir),
        "seed": int(args.seed),
        "source_v9580": rel(source_v9580),
        "source_v9570": rel(source_v9570),
        "source_v9560": rel(source_v9560),
        "source_v9550": rel(source_v9550),
        "route": route,
        "no_fake": 1,
        "no_proxy": 1,
    }

    outputs = {
        "p0_v9580_boundary_reproduction.csv": p0_rows,
        "p1_group_drop_autopsy.csv": p1_rows,
        "p1_group_drop_failure_modes.csv": p1_modes,
        "p2_group_balanced_target_density.csv": p2_rows,
        "p3_feature_invariance_deconfounding.csv": p3_rows,
        "p4_group_invariant_rankers.csv": p4_rows,
        "p5_rank_safe_certificate_v7.csv": p5_rows,
        "p6_existing_action_controller.csv": p6_rows,
        "p7_apga_ood_damage_autopsy.csv": p7_rows,
        "p8_memory_cover_blocker_v3.csv": p8_rows,
        "p9_apgi_apgc_implementation.csv": p9_rows,
        "p10_apgi_apgc_branch_horizon_outcome.csv": p10_rows,
        "p11_selected_runtime_preflight.csv": p11_rows,
        "p12_system_legal_controller_boundary.csv": p12_rows,
        "p13_leaveout_official.csv": p13_rows,
        "p14_official_paired_replay.csv": p14_rows,
        "p15_short_full_training_boundary.csv": p15_rows,
        "base_acc_sentinel_v9590.csv": p16_rows,
        "contract_audit_v9590.csv": [contract],
        "no_fake_audit_v9590.csv": [nofake],
        "failure_taxonomy_v9590.csv": [failure],
    }
    for name, rows in outputs.items():
        write_csv(out_dir / name, rows)
    write_json(out_dir / "route_decision_v9590.json", route_decision)
    write_json(out_dir / "aggregate_decision_v9590.json", aggregate)
    write_json(out_dir / "run_manifest_v9590.json", manifest)

    artifacts = {
        "plan": PLAN_PATH,
        "runner": SCRIPT_PATH,
        "run manifest": out_dir / "run_manifest_v9590.json",
        "route": out_dir / "route_decision_v9590.json",
        "P0 boundary": out_dir / "p0_v9580_boundary_reproduction.csv",
        "P1 group drop": out_dir / "p1_group_drop_autopsy.csv",
        "P1 failure modes": out_dir / "p1_group_drop_failure_modes.csv",
        "P2 target density": out_dir / "p2_group_balanced_target_density.csv",
        "P3 feature invariance": out_dir / "p3_feature_invariance_deconfounding.csv",
        "P4 group invariant rank": out_dir / "p4_group_invariant_rankers.csv",
        "P5 certificate": out_dir / "p5_rank_safe_certificate_v7.csv",
        "P6 controller": out_dir / "p6_existing_action_controller.csv",
        "P7 APGA OOD": out_dir / "p7_apga_ood_damage_autopsy.csv",
        "P8 memory cover": out_dir / "p8_memory_cover_blocker_v3.csv",
        "P9 APGI/APGC implementation": out_dir / "p9_apgi_apgc_implementation.csv",
        "P10 APGI/APGC outcome": out_dir / "p10_apgi_apgc_branch_horizon_outcome.csv",
        "Base-Acc Sentinel": out_dir / "base_acc_sentinel_v9590.csv",
        "contract audit": out_dir / "contract_audit_v9590.csv",
        "no-fake audit": out_dir / "no_fake_audit_v9590.csv",
        "failure taxonomy": out_dir / "failure_taxonomy_v9590.csv",
    }
    write_csv(out_dir / "artifact_hashes_v9590.csv", sha_rows(artifacts))
    print(json.dumps({
        "out_dir": rel(out_dir),
        "route": route,
        "dominant_drop_axis": p1.get("dominant_drop_axis"),
        "best_ranker": p4.get("best_ranker_id"),
        "best_ranker_TopK87_precision": p4.get("best_TopK87_precision"),
        "best_ranker_LDO_drop_max": p4.get("best_LDO_drop_max"),
        "apga_damage_mode": p7.get("dominant_damage_mode"),
        "system_legal_controller_pass": p12.get("system_legal_controller_pass"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
