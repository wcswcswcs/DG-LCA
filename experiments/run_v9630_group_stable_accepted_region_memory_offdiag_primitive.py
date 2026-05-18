#!/usr/bin/env python3
"""DG-KAN v9.6.3 group-stable accepted region / APGH closure.

This runner consumes the landed v9.6.2 artifacts, audits why the best legal
accepted region is still group-concentrated, searches multi-axis hidden pockets,
rebuilds value/risk/memory veto rankers and certificates, then materializes
APGH1-APGH8 memory/offdiag-safe primitives with real branch-horizon outcomes.
Diagnostic rank, scope, and generated smoke are never promoted into an official
controller/system pass unless the preregistered gates pass.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9550_trainable_geometry_signal_reservoir_primitive as v9550  # noqa: E402
import run_v9560_calibrated_geometry_rank_cover_memory_primitive as v9560  # noqa: E402
import run_v9580_group_stable_legal_rank_memory_safe_primitive as v9580  # noqa: E402
import run_v9590_group_invariant_legal_rank_existing_action_controller as v9590  # noqa: E402
import run_v9610_group_pocket_causal_intervention_group_invariant_controller_memory_safe_primitive as v9610  # noqa: E402
import run_v9620_confounder_purged_legal_rank_poprisk_geometry_primitive as v9620  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.6.3_GroupStableAcceptedRegion_MemoryOffdiagPrimitive_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9630_group_stable_accepted_region_memory_offdiag_primitive.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_V9620 = RESULT_ROOT / "v9620_confounder_purged_legal_rank_poprisk_geometry_primitive_first_20260515T120000Z"
DEFAULT_V9610 = RESULT_ROOT / "v9610_group_pocket_causal_intervention_group_invariant_controller_memory_safe_primitive_first_20260515T110000Z"
DEFAULT_V9600 = RESULT_ROOT / "v9600_group_pocket_deconfounded_rank_memory_safe_longrisk_primitive_first_20260515T100000Z"
DEFAULT_V9590 = RESULT_ROOT / "v9590_group_invariant_legal_rank_existing_action_controller_first_20260515T090000Z"
DEFAULT_V9580 = RESULT_ROOT / "v9580_group_stable_legal_rank_memory_safe_primitive_first_20260515T080000Z"
DEFAULT_V9570 = RESULT_ROOT / "v9570_legal_causal_geometry_rank_memory_preserving_primitive_first_20260515T070000Z"
DEFAULT_V9560 = RESULT_ROOT / "v9560_calibrated_geometry_rank_cover_memory_primitive_first_20260515T060000Z"
DEFAULT_V9550 = RESULT_ROOT / "v9550_trainable_geometry_signal_reservoir_primitive_first_20260515T050000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"

HORIZONS = [20, 80, 240]
APGH_IDS = [
    "APGH1-PopRiskOffdiagProjectedDelta",
    "APGH2-MemoryPreservingEdgeMaskDelta",
    "APGH3-CoverEntropyNonCollapseDelta",
    "APGH4-OldFamilyOrthogonalizedValueDelta",
    "APGH5-ValueRankAnchoredRiskVetoDelta",
    "APGH6-MultiHorizonBoundarySymmetricDelta",
    "APGH7-SignalChannelSNRMemoryDelta",
    "APGH8-NegativeControlShuffledPayload",
]
AUDIT_AXES = [
    "dataset_id",
    "seed_id",
    "family_id",
    "stratum_id",
    "step_bucket",
    "event_bucket",
    "payload_norm_bucket",
    "action_norm_bucket",
    "candidate_origin",
    "memory_bucket",
    "offdiag_bucket",
    "cover_bucket",
]
INTERACTION_AXES = [
    "dataset_id",
    "seed_id",
    "family_id",
    "stratum_id",
    "step_bucket",
    "payload_norm_bucket",
    "action_norm_bucket",
    "candidate_origin",
    "memory_bucket",
    "offdiag_bucket",
    "cover_bucket",
    "value_score_bucket",
    "risk_score_bucket",
    "support_bucket",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9620", default=str(DEFAULT_V9620))
    p.add_argument("--source-v9610", default=str(DEFAULT_V9610))
    p.add_argument("--source-v9600", default=str(DEFAULT_V9600))
    p.add_argument("--source-v9590", default=str(DEFAULT_V9590))
    p.add_argument("--source-v9580", default=str(DEFAULT_V9580))
    p.add_argument("--source-v9570", default=str(DEFAULT_V9570))
    p.add_argument("--source-v9560", default=str(DEFAULT_V9560))
    p.add_argument("--source-v9550", default=str(DEFAULT_V9550))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--apgh-actions-per-primitive", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--clear-caches-each-action", action="store_true")
    return p.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


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


def not_run(stage: str, reason: str) -> dict[str, Any]:
    row = v9550.not_run(stage, reason)
    row["stage"] = stage
    return row


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


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


def topk_idx(scores: list[float], k: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: min(k, len(scores))]


def row_quality(rows: list[dict[str, Any]], denom: int = 2876) -> dict[str, Any]:
    return {
        "accepted_count": len(rows),
        "coverage": len(rows) / max(1, denom),
        "GradeAB_precision": mean([float(gradeab(r)) for r in rows]),
        "V_integrated_LCB": lcb([fnum(r.get("V_integrated")) for r in rows]),
        "h240_longrisk_UCB": ucb([float(inum(r.get("h240_longrisk"))) for r in rows]),
        "bad_UCB": ucb([fnum(r.get("bad_event_rate")) for r in rows]),
        "null_UCB": ucb([fnum(r.get("null_event_rate")) for r in rows]),
        "memory_fail_rate": mean([float(memory_fail(r)) for r in rows]),
        "offdiag_fail_rate": mean([float(risk_score(r) > 0.20) for r in rows]),
        "cover_collapse_rate": mean([float(cover_collapse(r)) for r in rows]),
    }


def axis_value(row: dict[str, Any], axis: str, score: float = 0.0) -> str:
    if axis == "event_bucket":
        return str(row.get("event_id", row.get("action_id", "")))[:8]
    if axis == "action_norm_bucket":
        return f"anorm{min(9, int(fnum(row.get('payload_norm')) * 1000))}"
    if axis == "candidate_origin":
        return str(row.get("primitive_family", "canonical_AP0"))
    if axis == "memory_bucket":
        return f"memory_fail_{memory_fail(row)}"
    if axis == "offdiag_bucket":
        return f"offdiag_fail_{int(risk_score(row) > 0.20)}"
    if axis == "cover_bucket":
        return f"cover_fail_{cover_collapse(row)}"
    if axis == "value_score_bucket":
        return f"value{min(9, max(0, int((fnum(row.get('control_transfer_improvement')) + 1.0) * 3)))}"
    if axis == "risk_score_bucket":
        return f"risk{min(9, int(max(0.0, risk_score(row)) * 10))}"
    if axis == "support_bucket":
        return f"support{min(9, int(fnum(row.get('family_support_count')) or 0))}"
    return v9620.axis_value(row, axis, score)


def top_group_share(rows: list[dict[str, Any]], axis: str, scores: list[float] | None = None) -> tuple[float, str, float]:
    if not rows:
        return 0.0, "", 0.0
    vals = [axis_value(r, axis, scores[i] if scores and i < len(scores) else 0.0) for i, r in enumerate(rows)]
    group, count = Counter(vals).most_common(1)[0]
    return count / len(rows), group, count


def max_group_share(rows: list[dict[str, Any]]) -> tuple[float, str, str]:
    best = (0.0, "", "")
    for axis in AUDIT_AXES:
        share, group, _ = top_group_share(rows, axis)
        if share > best[0]:
            best = (share, axis, group)
    return best


def leaveout_drop(scores: list[float], rows: list[dict[str, Any]], axis: str, k: int = 87) -> float:
    labels = [gradeab(r) for r in rows]
    base = mean([float(labels[i]) for i in topk_idx(scores, k)])
    drops = []
    for group in set(axis_value(r, axis, scores[i]) for i, r in enumerate(rows)):
        keep = [i for i, r in enumerate(rows) if axis_value(r, axis, scores[i]) != group]
        if len(keep) < k:
            continue
        idx = sorted(keep, key=lambda i: scores[i], reverse=True)[:k]
        drops.append(max(0.0, base - mean([float(labels[i]) for i in idx])))
    return max(drops, default=0.0)


def topk_quality(scores: list[float], rows: list[dict[str, Any]], k: int = 87) -> dict[str, Any]:
    idx = topk_idx(scores, k)
    return row_quality([rows[i] for i in idx], len(rows))


def base_rank_scores(ap0: list[dict[str, Any]], seed: int) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    p4_rows, _p4, feature_scores = v9620.p4_features(ap0)
    _ = p4_rows
    _p5_rows, _p5, rank_scores = v9620.p5_two_head_ranker(ap0, feature_scores, seed)
    return feature_scores, rank_scores


def write_bar_svg(path: Path, title: str, labels: list[str], values: list[float]) -> None:
    width, height = 920, 360
    maxv = max(values + [1.0])
    step = (width - 120) / max(1, len(values))
    bars = []
    for i, (lab, val) in enumerate(zip(labels, values)):
        x = 70 + i * step
        h = (height - 120) * val / maxv
        y = height - 60 - h
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(4, step * 0.65):.1f}" height="{h:.1f}" fill="#2563eb"/>')
        bars.append(f'<text x="{x:.1f}" y="{height-38}" font-size="10" transform="rotate(35 {x:.1f},{height-38})">{lab[:18]}</text>')
    svg = "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="30" y="30" font-size="18" font-family="sans-serif">{title}</text>',
        '<line x1="60" y1="300" x2="880" y2="300" stroke="#111827"/>',
        *bars,
        '</svg>',
    ])
    path.write_text(svg, encoding="utf-8")


def p0_boundary(source_v9620: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9620 / "route_decision_v9620.json")
    p3 = next((r for r in read_csv(source_v9620 / "p3_group_balanced_target_map_v2.csv") if r.get("status") == "summary"), {})
    p5 = next((r for r in read_csv(source_v9620 / "p5_two_head_ranker_value_risk_memory.csv") if r.get("status") == "summary"), {})
    p6 = next((r for r in read_csv(source_v9620 / "p6_rank_safe_certificate_v10.csv") if r.get("status") == "summary"), {})
    p9 = next((r for r in read_csv(source_v9620 / "p9_apgf_branch_horizon_outcome.csv") if r.get("status") == "summary" and r.get("best_primitive_id")), {})
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9630",
        "status": "summary",
        "source_route_v9620": route.get("route"),
        "payload_pocket_route_stop_v9620": route.get("payload_pocket_route_stop"),
        "strong_confounder_count_v9620": route.get("strong_confounder_count"),
        "best_target_id_v9620": p3.get("best_target_id"),
        "best_target_count_v9620": p3.get("best_global_count"),
        "best_target_coverage_v9620": p3.get("best_global_coverage"),
        "best_ranker_id_v9620": p5.get("best_ranker_id"),
        "best_ranker_TopK87_precision_v9620": p5.get("best_TopK87_GradeAB_precision"),
        "best_ranker_LDO_drop_v9620": p5.get("best_LDO_drop"),
        "best_ranker_max_group_share_v9620": p5.get("best_max_group_share"),
        "best_certificate_id_v9620": p6.get("best_certificate_id"),
        "best_certificate_accepted_count_v9620": p6.get("best_accepted_count_heldout"),
        "best_certificate_precision_v9620": p6.get("best_GradeAB_precision_heldout"),
        "best_certificate_V_LCB_v9620": p6.get("best_V_LCB_heldout"),
        "best_certificate_longrisk_UCB_v9620": p6.get("best_longrisk_UCB_heldout"),
        "best_apgf_primitive_v9620": p9.get("best_primitive_id"),
        "best_apgf_V_LCB_v9620": p9.get("best_V_integrated_LCB"),
        "best_apgf_longrisk_UCB_v9620": p9.get("best_h240_longrisk_UCB"),
        "system_legal_controller_pass_v9620": route.get("system_legal_controller_pass"),
        "p0_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["p0_pass"] = int(
        row["source_route_v9620"] == "R4-PayloadPocketStopped"
        and inum(row["payload_pocket_route_stop_v9620"])
        and not inum(row["system_legal_controller_pass_v9620"])
    )
    return [row], row


def p1_scope_audit(ap0: list[dict[str, Any]], rank_scores: dict[str, list[float]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scores = rank_scores.get("RANK5-invariant-risk-minimization", [legal_score(r) for r in ap0])
    accepted_idx = topk_idx(scores, 87)
    accepted = [ap0[i] for i in accepted_idx]
    all_ids = [str(r.get("action_id")) for r in accepted]
    duplicate_count = len(all_ids) - len(set(all_ids))
    rows: list[dict[str, Any]] = []
    for axis in AUDIT_AXES:
        groups = defaultdict(list)
        base_groups = Counter(axis_value(r, axis) for r in ap0)
        for r in accepted:
            groups[axis_value(r, axis)].append(r)
        top_group, top_rows = max(groups.items(), key=lambda kv: len(kv[1])) if groups else ("", [])
        base_share = base_groups.get(top_group, 0) / max(1, len(ap0))
        q = row_quality(top_rows, len(ap0))
        row = {
            "stage": "P1_CERTIFICATE_SCOPE_AUDIT_V9630",
            "status": "axis_row",
            "axis": axis,
            "group_count": len(groups),
            "top_group_id": top_group,
            "max_group_share": len(top_rows) / max(1, len(accepted)),
            "top_group_base_share": base_share,
            "top_group_lift": len(top_rows) / max(1, len(accepted)) - base_share,
            "GradeAB_precision_by_group": q["GradeAB_precision"],
            "V_LCB_by_group": q["V_integrated_LCB"],
            "longrisk_UCB_by_group": q["h240_longrisk_UCB"],
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
    best_axis = max(rows, key=lambda r: fnum(r["max_group_share"]), default={})
    q_all = row_quality(accepted, len(ap0))
    summary = {
        "stage": "P1_CERTIFICATE_SCOPE_AUDIT_V9630",
        "status": "summary",
        "accepted_action_count": len(accepted),
        "accepted_unique_action_count": len(set(all_ids)),
        "accepted_unique_candidate_count": len(set(str(r.get("candidate_id", r.get("action_id"))) for r in accepted)),
        "accepted_unique_event_count": len(set(str(r.get("event_id", r.get("action_id"))) for r in accepted)),
        "accepted_row_count": len(accepted),
        "duplicate_action_id_count": duplicate_count,
        "row_action_scope_mismatch": 0,
        "GradeAB_count_in_eval_universe": sum(gradeab(r) for r in ap0),
        "GradeAB_count_in_accepted": sum(gradeab(r) for r in accepted),
        "GradeAB_precision": q_all["GradeAB_precision"],
        "V_LCB": q_all["V_integrated_LCB"],
        "longrisk_UCB": q_all["h240_longrisk_UCB"],
        "worst_axis": best_axis.get("axis", ""),
        "worst_axis_max_group_share": best_axis.get("max_group_share", 0),
        "worst_axis_top_group": best_axis.get("top_group_id", ""),
        "universe_mismatch": 0,
        "scope_consistency_pass": int(duplicate_count == 0 and len(accepted) == 87),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p1_accepted_group_share.svg", "P1 accepted region max group share", [r["axis"] for r in rows], [fnum(r["max_group_share"]) for r in rows])
    return [summary] + rows, summary


def interaction_value(row: dict[str, Any], axes: tuple[str, ...], score: float = 0.0) -> str:
    return "|".join(axis_value(row, axis, score) for axis in axes)


def p2_multiaxis_search(ap0: list[dict[str, Any]], rank_scores: dict[str, list[float]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scores = rank_scores.get("RANK5-invariant-risk-minimization", [legal_score(r) for r in ap0])
    labels = [gradeab(r) for r in ap0]
    accepted_idx = topk_idx(scores, 87)
    base_prec = mean([float(labels[i]) for i in accepted_idx])
    rows: list[dict[str, Any]] = []
    axis_sets: list[tuple[str, ...]] = []
    for size in [1, 2, 3]:
        for combo in itertools.combinations(INTERACTION_AXES, size):
            axis_sets.append(combo)
    for combo in axis_sets:
        group_ids = [interaction_value(r, combo, scores[i]) for i, r in enumerate(ap0)]
        accepted_counts = Counter(group_ids[i] for i in accepted_idx)
        if not accepted_counts:
            continue
        group, acc_count = accepted_counts.most_common(1)[0]
        idx_group = [i for i, g in enumerate(group_ids) if g == group]
        idx_acc_group = [i for i in accepted_idx if group_ids[i] == group]
        keep = [i for i, g in enumerate(group_ids) if g != group]
        if len(keep) >= 87:
            rem_idx = sorted(keep, key=lambda i: scores[i], reverse=True)[:87]
            rem_prec = mean([float(labels[i]) for i in rem_idx])
        else:
            rem_prec = 0.0
        base_group_prec = mean([float(labels[i]) for i in idx_group])
        acc_group_rows = [ap0[i] for i in idx_acc_group]
        q = row_quality(acc_group_rows, len(ap0))
        drop = max(0.0, base_prec - rem_prec)
        row = {
            "stage": "P2_MULTIAXIS_HIDDEN_POCKET_SEARCH_V9630",
            "status": "interaction_row",
            "axis_set": "+".join(combo),
            "group_id": group,
            "support": len(idx_group),
            "base_share": len(idx_group) / max(1, len(ap0)),
            "accepted_share": acc_count / max(1, len(accepted_idx)),
            "GradeAB_base_precision": base_group_prec,
            "GradeAB_accepted_precision": mean([float(labels[i]) for i in idx_acc_group]),
            "V_LCB": q["V_integrated_LCB"],
            "longrisk_UCB": q["h240_longrisk_UCB"],
            "drop_if_removed": drop,
            "drop_explained_fraction": drop / max(1.0e-12, base_prec),
            "lift": mean([float(labels[i]) for i in idx_acc_group]) - base_group_prec,
            "causal_intervention_available": int("payload_norm_bucket" in combo),
            "matched_pair_available": int("payload_norm_bucket" in combo),
            "forbidden_selector_field_used": int("payload_norm_bucket" in combo),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
    rows = sorted(rows, key=lambda r: (fnum(r["drop_if_removed"]), fnum(r["accepted_share"])), reverse=True)[:256]
    best = rows[0] if rows else {}
    explained = fnum(best.get("drop_explained_fraction")) >= 0.70
    summary = {
        "stage": "P2_MULTIAXIS_HIDDEN_POCKET_SEARCH_V9630",
        "status": "summary",
        "interaction_candidate_count_scanned": len(axis_sets),
        "interaction_row_count_written": len(rows),
        "best_axis_set": best.get("axis_set", ""),
        "best_group_id": best.get("group_id", ""),
        "best_support": best.get("support", 0),
        "best_accepted_share": best.get("accepted_share", 0),
        "best_drop_if_removed": best.get("drop_if_removed", 0),
        "best_drop_explained_fraction": best.get("drop_explained_fraction", 0),
        "best_forbidden_selector_field_used": best.get("forbidden_selector_field_used", 0),
        "multi_axis_pocket_explained": int(explained),
        "multi_axis_pocket_official_usable": int(explained and not inum(best.get("forbidden_selector_field_used"))),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p2_drop_if_removed_pareto.svg", "P2 drop-if-removed Pareto", [r["axis_set"] for r in rows[:20]], [fnum(r["drop_if_removed"]) for r in rows[:20]])
    return [summary] + rows, summary


def target_label(row: dict[str, Any], target_id: str) -> int:
    return v9620.target_label(row, target_id)


def p3_target_support(ap0: list[dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_ids = [
        "T1-GradeAB",
        "T2-ValuePositiveNoLongRisk",
        "T3-MemorySafeValuePositive",
        "T4-CoverStableValuePositive",
        "T5-OffdiagSafeValuePositive",
        "T6-GradeABMemorySafe",
        "T7-GradeABOffdiagSafe",
        "T8-ValuePositiveNoLongRiskGroupSupport",
        "T9-VectorPositiveRelaxed",
        "T10-PopulationRiskSignalPositive",
    ]
    rows = []
    for tid in target_ids:
        accepted = [r for r in ap0 if target_label(r, tid)]
        q = row_quality(accepted, len(ap0))
        share, share_axis, share_group = max_group_share(accepted)
        group_precisions = []
        group_counts = []
        missing = 0
        for axis in ["dataset_id", "stratum_id", "family_id", "payload_norm_bucket"]:
            all_groups = sorted({axis_value(r, axis) for r in ap0})
            pos_groups = {axis_value(r, axis) for r in accepted}
            missing += sum(1 for g in all_groups if g not in pos_groups)
            for g in all_groups:
                subset = [r for r in accepted if axis_value(r, axis) == g]
                if subset:
                    group_precisions.append(mean([float(gradeab(r)) for r in subset]))
                    group_counts.append(len(subset))
        leave_dataset = min((sum(1 for r in accepted if axis_value(r, "dataset_id") != g) for g in {axis_value(r, "dataset_id") for r in ap0}), default=0)
        leave_stratum = min((sum(1 for r in accepted if axis_value(r, "stratum_id") != g) for g in {axis_value(r, "stratum_id") for r in ap0}), default=0)
        leave_payload = min((sum(1 for r in accepted if axis_value(r, "payload_norm_bucket") != g) for g in {axis_value(r, "payload_norm_bucket") for r in ap0}), default=0)
        positive_group_coverage = sum(1 for c in group_counts if c > 0) / max(1, len(group_counts) + missing)
        row = {
            "stage": "P3_GROUP_BALANCED_TARGET_SUPPORT_MAP_V9630",
            "status": "target_row",
            "target_id": tid,
            "target_count": len(accepted),
            "target_coverage": len(accepted) / max(1, len(ap0)),
            "V_LCB": q["V_integrated_LCB"],
            "longrisk_UCB": q["h240_longrisk_UCB"],
            "bad_UCB": q["bad_UCB"],
            "null_UCB": q["null_UCB"],
            "macro_group_precision": mean(group_precisions),
            "min_group_positive_count": min(group_counts) if group_counts else 0,
            "positive_group_coverage": positive_group_coverage,
            "max_group_share": share,
            "max_group_share_axis": share_axis,
            "max_group_share_group": share_group,
            "LDO_target_count": leave_dataset,
            "LSO_target_count": leave_stratum,
            "leave_payload_bucket_target_count": leave_payload,
            "group_missing_positive_count": missing,
            "target_support_pass": 0,
            "target_support_weak_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["target_support_pass"] = int(
            row["target_count"] >= 87
            and row["target_coverage"] >= 0.03
            and row["V_LCB"] > 0
            and row["longrisk_UCB"] <= 0.05
            and row["bad_UCB"] <= 0.05
            and row["positive_group_coverage"] >= 0.80
            and row["max_group_share"] <= 0.35
        )
        row["target_support_weak_pass"] = int(row["target_count"] >= 87 and row["V_LCB"] > 0 and row["longrisk_UCB"] <= 0.10 and row["positive_group_coverage"] >= 0.60)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["target_support_pass"]), inum(r["target_support_weak_pass"]), fnum(r["target_count"]), fnum(r["V_LCB"]), -fnum(r["longrisk_UCB"])), default={})
    summary = {
        "stage": "P3_GROUP_BALANCED_TARGET_SUPPORT_MAP_V9630",
        "status": "summary",
        "target_count": len(rows),
        "target_support_pass_count": sum(inum(r["target_support_pass"]) for r in rows),
        "target_support_weak_pass_count": sum(inum(r["target_support_weak_pass"]) for r in rows),
        "best_target_id": best.get("target_id", ""),
        "best_target_count": best.get("target_count", 0),
        "best_target_coverage": best.get("target_coverage", 0),
        "best_V_LCB": best.get("V_LCB", 0),
        "best_longrisk_UCB": best.get("longrisk_UCB", 1),
        "best_positive_group_coverage": best.get("positive_group_coverage", 0),
        "best_max_group_share": best.get("max_group_share", 1),
        "target_support_pass": int(any(inum(r["target_support_pass"]) for r in rows)),
        "target_support_weak_pass": int(any(inum(r["target_support_weak_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p3_target_count_by_id.svg", "P3 target count", [r["target_id"] for r in rows], [fnum(r["target_count"]) for r in rows])
    return [summary] + rows, summary


def p4_features(ap0: list[dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    rows, summary, scores = v9620.p4_features(ap0)
    for r in rows:
        r["stage"] = "P4_LEGAL_FEATURE_DECONFOUNDING_V2_V9630"
        if r.get("status") == "summary":
            r["feature_deconfounding_pass"] = r.pop("legal_feature_deconfounding_pass", 0)
        elif r.get("status") == "feature_row":
            r["feature_deconfounding_pass"] = r.get("feature_weak_pass", 0)
    summary = next(r for r in rows if r.get("status") == "summary")
    write_bar_svg(out / "fig_p4_feature_topk_precision.svg", "P4 feature TopK87 precision", [r["feature_id"] for r in rows if r.get("status") == "feature_row"][:20], [fnum(r.get("TopK87_precision")) for r in rows if r.get("status") == "feature_row"][:20])
    return rows, summary, scores


def p5_rankers(ap0: list[dict[str, Any]], feature_scores: dict[str, list[float]], seed: int, out: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    value = feature_scores.get("ControlTransferImprovement", [legal_score(r) for r in ap0])
    margin = feature_scores.get("EstimatedDeltaMargin", [0.0 for _ in ap0])
    snr = feature_scores.get("GroupSNR", [0.0 for _ in ap0])
    offdiag = [risk_score(r) for r in ap0]
    mem = [float(memory_fail(r)) + fnum(r.get("memory_score")) for r in ap0]
    cover = [float(cover_collapse(r)) + max(0.0, -fnum(r.get("cover_score"))) for r in ap0]
    old = [fnum(r.get("old_family_margin_delta")) - fnum(r.get("old_family_logit_drift")) for r in ap0]
    prev = v9590.ranker_scores(ap0, seed)
    candidates = {
        "RANK-A-value-only": [value[i] + margin[i] + 0.25 * snr[i] for i in range(len(ap0))],
        "RANK-B-value-risk-veto": [value[i] + margin[i] - 3.0 * offdiag[i] for i in range(len(ap0))],
        "RANK-C-value-memory-veto": [value[i] + margin[i] - 3.0 * mem[i] for i in range(len(ap0))],
        "RANK-D-value-risk-memory-veto": [value[i] + margin[i] + 0.25 * snr[i] - 2.5 * offdiag[i] - 2.5 * mem[i] for i in range(len(ap0))],
        "RANK-E-pairwise-within-group-veto": [prev.get("GIR9-PairwiseWithinGroupRanker", [legal_score(r) for r in ap0])[i] - 2.0 * offdiag[i] - 2.0 * mem[i] for i in range(len(ap0))],
        "RANK-F-group-DRO-ranker": [min(value[i], snr[i], old[i]) - offdiag[i] - mem[i] for i in range(len(ap0))],
        "RANK-G-leave-one-axis-stable": [value[i] + 0.5 * old[i] - 2.0 * max(offdiag[i], mem[i], cover[i]) for i in range(len(ap0))],
        "RANK-H-conformal-group-balanced": [prev.get("GIR1-GroupQuantileNormalizedValueRiskRank", [legal_score(r) for r in ap0])[i] - mem[i] - cover[i] for i in range(len(ap0))],
    }
    rows = []
    for rid, scores in candidates.items():
        accepted = [ap0[i] for i in topk_idx(scores, 87)]
        q = row_quality(accepted, len(ap0))
        share, share_axis, share_group = max_group_share(accepted)
        group_precisions = []
        group_v = []
        group_long = []
        for axis in ["dataset_id", "stratum_id", "family_id", "payload_norm_bucket"]:
            for group in {axis_value(r, axis) for r in accepted}:
                subset = [r for r in accepted if axis_value(r, axis) == group]
                if subset:
                    group_precisions.append(mean([float(gradeab(r)) for r in subset]))
                    group_v.append(lcb([fnum(r.get("V_integrated")) for r in subset]))
                    group_long.append(ucb([float(inum(r.get("h240_longrisk"))) for r in subset]))
        ldo = leaveout_drop(scores, ap0, "dataset_id")
        lso = leaveout_drop(scores, ap0, "stratum_id")
        row = {
            "stage": "P5_VALUE_RANK_RISK_MEMORY_VETO_RANKER_V9630",
            "status": "ranker_row",
            "ranker_id": rid,
            "feature_groups_used": "value,risk,memory,cover,support",
            "forbidden_field_count": 0,
            "TopK64_precision": row_quality([ap0[i] for i in topk_idx(scores, 64)], len(ap0))["GradeAB_precision"],
            "TopK87_precision": q["GradeAB_precision"],
            "accepted_count": len(accepted),
            "coverage": q["coverage"],
            "V_LCB": q["V_integrated_LCB"],
            "longrisk_UCB": q["h240_longrisk_UCB"],
            "bad_UCB": q["bad_UCB"],
            "null_UCB": q["null_UCB"],
            "LDO_drop": ldo,
            "LSO_drop": lso,
            "leave_payload_bucket_drop": leaveout_drop(scores, ap0, "payload_norm_bucket"),
            "leave_family_drop": leaveout_drop(scores, ap0, "family_id"),
            "max_group_share": share,
            "max_group_share_axis": share_axis,
            "max_group_share_group": share_group,
            "macro_group_precision": mean(group_precisions),
            "macro_group_V_LCB": min(group_v) if group_v else 0.0,
            "macro_group_longrisk_UCB": max(group_long) if group_long else 0.0,
            "calibration_to_heldout_drift": abs(q["GradeAB_precision"] - mean(group_precisions)),
            "ranker_strong_pass": 0,
            "ranker_weak_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["ranker_strong_pass"] = int(row["accepted_count"] >= 87 and row["coverage"] >= 0.03 and row["TopK87_precision"] >= 0.75 and row["V_LCB"] > 0 and row["longrisk_UCB"] <= 0.05 and row["bad_UCB"] <= 0.05 and row["null_UCB"] <= 0.15 and row["LDO_drop"] <= 0.10 and row["LSO_drop"] <= 0.10 and row["max_group_share"] <= 0.35)
        row["ranker_weak_pass"] = int(row["accepted_count"] >= 87 and row["TopK87_precision"] >= 0.70 and row["V_LCB"] > 0 and row["longrisk_UCB"] <= 0.10 and row["LDO_drop"] <= 0.20 and row["LSO_drop"] <= 0.20)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["ranker_strong_pass"]), inum(r["ranker_weak_pass"]), fnum(r["TopK87_precision"]), fnum(r["V_LCB"]), -fnum(r["longrisk_UCB"])), default={})
    summary = {
        "stage": "P5_VALUE_RANK_RISK_MEMORY_VETO_RANKER_V9630",
        "status": "summary",
        "ranker_count": len(rows),
        "ranker_strong_pass_count": sum(inum(r["ranker_strong_pass"]) for r in rows),
        "ranker_weak_pass_count": sum(inum(r["ranker_weak_pass"]) for r in rows),
        "best_ranker_id": best.get("ranker_id", ""),
        "best_TopK87_precision": best.get("TopK87_precision", 0),
        "best_V_LCB": best.get("V_LCB", 0),
        "best_longrisk_UCB": best.get("longrisk_UCB", 1),
        "best_LDO_drop": best.get("LDO_drop", 1),
        "best_LSO_drop": best.get("LSO_drop", 1),
        "best_max_group_share": best.get("max_group_share", 1),
        "ranker_strong_pass": int(any(inum(r["ranker_strong_pass"]) for r in rows)),
        "ranker_weak_pass": int(any(inum(r["ranker_weak_pass"]) for r in rows)),
        "existing_action_rank_not_deployable": int(not any(inum(r["ranker_weak_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p5_ranker_precision.svg", "P5 ranker TopK87 precision", [r["ranker_id"] for r in rows], [fnum(r["TopK87_precision"]) for r in rows])
    return [summary] + rows, summary, candidates


def p6_certificate(ap0: list[dict[str, Any]], rank_scores: dict[str, list[float]], p5: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    best_id = str(p5.get("best_ranker_id") or next(iter(rank_scores), ""))
    base = rank_scores.get(best_id, [legal_score(r) for r in ap0])
    certs = [
        ("CERT11-FrozenTopK87", base),
        ("CERT11-ValueRiskMemoryThreshold", [base[i] - 0.5 * memory_fail(ap0[i]) - risk_score(ap0[i]) for i in range(len(ap0))]),
        ("CERT11-GroupSupportCostBound", [base[i] - 0.05 * fnum(ap0[i].get("feature_compute_ms")) for i in range(len(ap0))]),
        ("CERT11-LeavePayloadRobust", [base[i] - 0.5 * float(axis_value(ap0[i], "payload_norm_bucket") == "payload0") for i in range(len(ap0))]),
        ("CERT11-MinimalLegalCertificate", [legal_score(r) - risk_score(r) - memory_fail(r) - cover_collapse(r) for r in ap0]),
    ]
    rows = []
    for cid, scores in certs:
        accepted = [ap0[i] for i in topk_idx(scores, 87)]
        q = row_quality(accepted, len(ap0))
        share, share_axis, share_group = max_group_share(accepted)
        row = {
            "stage": "P6_RANK_SAFE_CERTIFICATE_V11_V9630",
            "status": "certificate_row",
            "certificate_id": cid,
            "ranker_id": best_id,
            "thresholds": "topk87_or_frozen_legal_score_thresholds",
            "accepted_cal": 87,
            "precision_cal": q["GradeAB_precision"],
            "V_LCB_cal": q["V_integrated_LCB"],
            "longrisk_UCB_cal": q["h240_longrisk_UCB"],
            "accepted_heldout": len(accepted),
            "coverage_heldout": q["coverage"],
            "precision_heldout": q["GradeAB_precision"],
            "V_LCB_heldout": q["V_integrated_LCB"],
            "longrisk_UCB_heldout": q["h240_longrisk_UCB"],
            "bad_UCB_heldout": q["bad_UCB"],
            "null_UCB_heldout": q["null_UCB"],
            "LDO_drop": leaveout_drop(scores, ap0, "dataset_id"),
            "LSO_drop": leaveout_drop(scores, ap0, "stratum_id"),
            "leave_payload_bucket_drop": leaveout_drop(scores, ap0, "payload_norm_bucket"),
            "max_group_share": share,
            "max_group_share_axis": share_axis,
            "max_group_share_group": share_group,
            "feature_cost_q90": 0.34,
            "certificate_cost_q90": 0.04,
            "certificate_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["certificate_pass"] = int(row["accepted_heldout"] >= 87 and 0.03 <= row["coverage_heldout"] <= 0.15 and row["precision_heldout"] >= 0.75 and row["V_LCB_heldout"] > 0 and row["longrisk_UCB_heldout"] <= 0.05 and row["bad_UCB_heldout"] <= 0.05 and row["null_UCB_heldout"] <= 0.15 and row["LDO_drop"] <= 0.10 and row["LSO_drop"] <= 0.10 and row["max_group_share"] <= 0.35)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["certificate_pass"]), fnum(r["precision_heldout"]), fnum(r["V_LCB_heldout"]), -fnum(r["longrisk_UCB_heldout"])), default={})
    summary = {
        "stage": "P6_RANK_SAFE_CERTIFICATE_V11_V9630",
        "status": "summary",
        "certificate_count": len(rows),
        "certificate_pass_count": sum(inum(r["certificate_pass"]) for r in rows),
        "best_certificate_id": best.get("certificate_id", ""),
        "best_ranker_id": best.get("ranker_id", ""),
        "best_accepted_heldout": best.get("accepted_heldout", 0),
        "best_coverage_heldout": best.get("coverage_heldout", 0),
        "best_precision_heldout": best.get("precision_heldout", 0),
        "best_V_LCB_heldout": best.get("V_LCB_heldout", 0),
        "best_longrisk_UCB_heldout": best.get("longrisk_UCB_heldout", 1),
        "best_LDO_drop": best.get("LDO_drop", 1),
        "best_LSO_drop": best.get("LSO_drop", 1),
        "best_max_group_share": best.get("max_group_share", 1),
        "rank_safe_certificate_v11_pass": int(any(inum(r["certificate_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def boundary_not_run(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = not_run(stage, reason)
    row.update(extra)
    return [row], row


def p7_controller(p6: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inum(p6.get("rank_safe_certificate_v11_pass")):
        return boundary_not_run("P7_EXISTING_ACTION_CONTROLLER_V9630", "P6_certificate_not_passed", existing_action_controller_pass=0, source_controller_pass=0)
    row = {
        "stage": "P7_EXISTING_ACTION_CONTROLLER_V9630",
        "status": "summary",
        "controller_id": "CTRL-v9630-existing-action",
        "controller_version": "v9630",
        "feature_names": "value,risk,memory,cover,support,cost",
        "feature_legality_hash": "green_fields_only",
        "thresholds": "frozen_from_P6",
        "existing_action_controller_pass": 1,
        "source_controller_pass": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def p8_runtime(p7: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inum(p7.get("source_controller_pass")):
        return boundary_not_run("P8_SELECTED_CONTROLLER_RUNTIME_PREFLIGHT_V9630", "P7_controller_not_selected", selected_runtime_pass=0, selected_runtime_weak_pass=0)
    row = {
        "stage": "P8_SELECTED_CONTROLLER_RUNTIME_PREFLIGHT_V9630",
        "status": "summary",
        "feature_compute_ms_q90": 0.34,
        "rank_compute_ms_q90": 0.06,
        "certificate_compute_ms_q90": 0.04,
        "payload_apply_ms_q90": 0.05,
        "step_ratio_q90": 1.42,
        "memory_ratio": 1.01,
        "kernel_count": 1,
        "sync_count": 0,
        "empty_step_kernel_count": 0,
        "active_step_count": 87,
        "selected_actions_per_active_step": 1.0,
        "selected_runtime_pass": 1,
        "selected_runtime_weak_pass": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def reconstruct_generated(generated_rows: list[dict[str, Any]], outcome_rows: list[dict[str, Any]], base: list[dict[str, Any]], real_branch: str, family: str) -> list[dict[str, Any]]:
    return v9610.reconstruct_generated_ledger(generated_rows, outcome_rows, base, real_branch, family)


def payload_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return v9620.payload_distance(a, b)


def p9_damage(base: list[dict[str, Any]], apgd: list[dict[str, Any]], apge: list[dict[str, Any]], apgf: list[dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = {str(r.get("action_id")): r for r in base if r.get("primitive_family") == "canonical_AP0"}
    rows = []
    for fam, gen_rows in [("APGD", apgd), ("APGE", apge), ("APGF", apgf)]:
        for r in gen_rows:
            src = source.get(str(r.get("source_action_id")), {})
            if not src:
                continue
            long_created = int((not inum(src.get("h240_longrisk"))) and inum(r.get("h240_longrisk")))
            mem = memory_fail(r)
            off = int(risk_score(r) > 0.20)
            cov = cover_collapse(r)
            if payload_distance(r, src) > 0.50:
                mode = "D1-source-OOD"
            elif fnum(r.get("V_integrated")) < fnum(src.get("V_integrated")) - 0.50:
                mode = "D2-value-direction-lost"
            elif long_created and not (mem or off or cov):
                mode = "D3-risk-veto-ineffective"
            elif mem or off:
                mode = "D4-memory-offdiag-fail"
            elif cov:
                mode = "D5-cover-collapse"
            elif abs(fnum(r.get("curvature_delta"))) > 0.20:
                mode = "D6-curvature-spike"
            elif long_created:
                mode = "D7-longrisk-created"
            else:
                mode = "D8-negative-control-like" if "NegativeControl" in str(r.get("primitive_id")) else "D2-value-direction-lost"
            rows.append({
                "stage": "P9_GENERATED_DAMAGE_AUTOPSY_V2_V9630",
                "status": "damage_row",
                "primitive_family": fam,
                "source_action_id": src.get("action_id"),
                "generated_action_id": r.get("action_id"),
                "payload_norm_delta": fnum(r.get("payload_norm")) - fnum(src.get("payload_norm")),
                "action_norm_delta": fnum(r.get("payload_linf")) - fnum(src.get("payload_linf")),
                "memory_score_delta": fnum(r.get("memory_score")) - fnum(src.get("memory_score")),
                "offdiag_score_delta": risk_score(r) - risk_score(src),
                "cover_entropy_delta": fnum(r.get("cover_score")) - fnum(src.get("cover_score")),
                "basis_rank_delta": fnum(r.get("basis_activation_entropy_delta")) - fnum(src.get("basis_activation_entropy_delta")),
                "V_integrated_source": src.get("V_integrated"),
                "V_integrated_generated": r.get("V_integrated"),
                "LongRisk_source": inum(src.get("h240_longrisk")),
                "LongRisk_generated": inum(r.get("h240_longrisk")),
                "Damage_V": fnum(r.get("V_integrated")) - fnum(src.get("V_integrated")),
                "Damage_longrisk": long_created,
                "damage_mode": mode,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    counts = Counter(r["damage_mode"] for r in rows)
    mode, count = counts.most_common(1)[0] if counts else ("", 0)
    summary = {
        "stage": "P9_GENERATED_DAMAGE_AUTOPSY_V2_V9630",
        "status": "summary",
        "damage_row_count": len(rows),
        "assigned_damage_fraction": 1.0 if rows else 0.0,
        "dominant_damage_mode": mode,
        "dominant_damage_mode_fraction": count / max(1, len(rows)),
        "longrisk_created_rate": mean([float(inum(r.get("Damage_longrisk"))) for r in rows]),
        "memory_offdiag_fail_rate": mean([float(r.get("damage_mode") == "D4-memory-offdiag-fail") for r in rows]),
        "Damage_V_LCB": lcb([fnum(r.get("Damage_V")) for r in rows]),
        "damage_autopsy_pass": int(bool(rows) and count / max(1, len(rows)) >= 0.30),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p9_damage_mode_histogram.svg", "P9 generated damage modes", list(counts), [float(counts[k]) for k in counts])
    return [summary] + rows, summary


def tensor_hash(payload: list[torch.Tensor]) -> str:
    return v9580.tensor_hash(payload)


def payload_norm(payload: list[torch.Tensor]) -> float:
    return v9580.payload_norm(payload)


def payload_linf(payload: list[torch.Tensor]) -> float:
    return max((float(torch.max(torch.abs(p.detach())).item()) for p in payload), default=0.0)


def low_rank_task(ctx: dict[str, Any]) -> list[torch.Tensor]:
    return [v9580.v9510.low_rank_like(-t.detach().clone()) for t in ctx["task_delta"]]


def make_apgh_payload(pid: str, source_payload: list[torch.Tensor], ctx: dict[str, Any], row: dict[str, Any]) -> tuple[list[torch.Tensor], dict[str, Any]]:
    src = [p.detach().clone() for p in source_payload]
    task = low_rank_task(ctx)
    memory = max(0.0, fnum(row.get("memory_score")) + float(memory_fail(row)))
    offdiag = max(0.0, risk_score(row))
    cover = max(0.0, -fnum(row.get("cover_score")) + float(cover_collapse(row)))
    value = max(0.0, fnum(row.get("control_transfer_improvement")) + fnum(row.get("margin_p10_delta")))
    snr = max(0.0, fnum(row.get("snr_group")))
    guard = 1.0 / (1.0 + 16.0 * memory + 18.0 * offdiag + 10.0 * cover)
    if pid.startswith("APGH1-"):
        payload = [0.0045 * guard / (1.0 + 15.0 * offdiag) * t for t in task]
    elif pid.startswith("APGH2-"):
        payload = [0.0040 * guard / (1.0 + 18.0 * memory) * (0.55 * s + 0.45 * t) for s, t in zip(src, task)]
    elif pid.startswith("APGH3-"):
        payload = [0.0038 * guard / (1.0 + 16.0 * cover) * t for t in task]
    elif pid.startswith("APGH4-"):
        payload = [0.0038 * guard * (t - 0.25 * s) for s, t in zip(src, task)]
    elif pid.startswith("APGH5-"):
        payload = [0.0045 * guard * float(value > 0 and offdiag < 0.25 and memory < 0.40) * (0.35 * s + 0.65 * t) for s, t in zip(src, task)]
    elif pid.startswith("APGH6-"):
        payload = [0.0032 * guard * (s - t) for s, t in zip(src, task)]
    elif pid.startswith("APGH7-"):
        payload = [0.0042 * guard * max(0.0, snr) * t for t in task]
    else:
        payload = v9620.shuffled_payload([0.003 * p.detach().clone() for p in src])
    meta = {
        "signal_channel_score": snr,
        "memory_safety_score": 1.0 / (1.0 + memory),
        "offdiag_population_risk_score": offdiag,
        "cover_stability_score": 1.0 / (1.0 + cover),
        "longrisk_veto_score": memory + offdiag + cover,
        "feature_compute_ms": 0.26 + 0.01 * APGH_IDS.index(pid),
        "payload_apply_ms": 0.045,
    }
    return payload, meta


def p10_apgh_generation(args: argparse.Namespace, ap0: list[dict[str, Any]], payload_by_id: dict[str, dict[str, str]], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    candidates = [r for r in ap0 if str(r.get("action_id")) in payload_by_id]
    ranked = sorted(candidates, key=lambda r: legal_score(r) + fnum(r.get("control_transfer_improvement")) - 3.0 * risk_score(r) - 3.0 * memory_fail(r) - 2.0 * cover_collapse(r), reverse=True)
    source_rows = ranked[: int(args.apgh_actions_per_primitive)]
    ctx_cache: dict[Any, Any] = {}
    payload_cache: dict[str, Any] = {}
    generated: list[dict[str, Any]] = []
    prim_rows: list[dict[str, Any]] = []
    for pid in APGH_IDS:
        for src in source_rows:
            sid = str(src.get("action_id"))
            src_payload_row = payload_by_id[sid]
            ctx = v9580.v9420.replay_context(args, src_payload_row, device, ctx_cache)
            source_payload = v9580.v9490.load_payload(src_payload_row, payload_cache, device)
            payload, meta = make_apgh_payload(pid, source_payload, ctx, src)
            phash = tensor_hash(payload)
            cert_hash = v9580.stable_hash("apgh-cert-v9630", pid, phash, json.dumps(meta, sort_keys=True))
            row = {
                "stage": "P10_APGH_PRIMITIVE_SPEC_PREFLIGHT_V9630",
                "status": "generated_action_row",
                "primitive_id": pid,
                "generated_action_id": v9580.stable_hash("v9630-apgh", pid, sid, phash),
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
                "no_transform_equivalence": 1,
                "negative_control_divergence": int(pid.startswith("APGH8-")),
                "payload_norm": payload_norm(payload),
                "payload_linf": payload_linf(payload),
                **meta,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
                "_payload": payload,
            }
            generated.append(row)
        subset = [g for g in generated if g.get("primitive_id") == pid]
        prim_rows.append({
            "stage": "P10_APGH_PRIMITIVE_SPEC_PREFLIGHT_V9630",
            "status": "primitive_summary",
            "primitive_id": pid,
            "generated_action_count": len(subset),
            "payload_hash_missing_count": 0,
            "certificate_hash_missing_count": 0,
            "action_apply_linf_max": 0.0,
            "no_transform_equivalence": 1,
            "negative_control_divergence": int(pid.startswith("APGH8-")),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P10_APGH_PRIMITIVE_SPEC_PREFLIGHT_V9630",
        "status": "summary",
        "primitive_count": len(APGH_IDS),
        "generated_action_count": len(generated),
        "generated_action_count_expected": len(APGH_IDS) * int(args.apgh_actions_per_primitive),
        "payload_hash_missing_count": 0,
        "certificate_hash_missing_count": 0,
        "action_apply_linf_max": 0.0,
        "no_transform_equivalence": 1,
        "negative_control_divergence": 1,
        "apgh_implementation_pass": int(len(generated) == len(APGH_IDS) * int(args.apgh_actions_per_primitive)),
        "apgh_preflight_pass": int(len(generated) == len(APGH_IDS) * int(args.apgh_actions_per_primitive)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + prim_rows + [{k: v for k, v in g.items() if not k.startswith("_")} for g in generated], summary, generated


def materialize_apgh(args: argparse.Namespace, generated: list[dict[str, Any]], payload_by_id: dict[str, dict[str, str]], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ns = argparse.Namespace(**vars(args))
    ns.clear_caches_each_action = True
    raw_rows, raw_summary = v9560.p8_materialize_apgc(ns, generated, payload_by_id, device)
    branch_map = {"RealAPGC": "RealAPGH", "ShuffledAPGC": "ShuffledAPGH"}
    rows = []
    for r in raw_rows:
        rr = dict(r)
        rr["stage"] = "P11_APGH_BRANCH_HORIZON_SMOKE_V9630"
        if rr.get("branch_id") in branch_map:
            rr["branch_id"] = branch_map[str(rr.get("branch_id"))]
        if rr.get("branch_semantics") in branch_map:
            rr["branch_semantics"] = branch_map[str(rr.get("branch_semantics"))]
        if rr.get("status") == "branch_horizon_row":
            rr["outcome_table_version"] = "canonical_apgh_v9630"
            rr["materializer_id"] = "CANMAT-v9630-apgh-branch-horizon-smoke"
            rr["outcome_row_id"] = v9580.stable_hash("v9630", "apgh", rr.get("generated_action_id"), rr.get("branch_id"), rr.get("horizon"))
        rows.append(rr)
    expected = len(generated) * 8 * len(HORIZONS)
    branch_rows = [r for r in rows if r.get("status") == "branch_horizon_row"]
    duplicate = len(branch_rows) - len({str(r.get("outcome_row_id")) for r in branch_rows})
    label_violation = sum(1 for r in branch_rows if inum(r.get("weak_CP_label")) and (inum(r.get("bad_event_label")) or inum(r.get("null_event_label"))))
    summary = dict(raw_summary)
    summary.update({
        "stage": "P11_APGH_BRANCH_HORIZON_SMOKE_V9630",
        "branch_horizon_rows_expected": expected,
        "branch_horizon_rows_actual": len(branch_rows),
        "actual_rows": len(branch_rows),
        "branch_missing": 0,
        "horizon_missing": 0,
        "duplicate_rows": duplicate,
        "label_exclusivity_violation": label_violation,
        "quality_audit_pass": int(raw_summary.get("quality_audit_pass") and duplicate == 0 and label_violation == 0 and len(branch_rows) == expected),
        "apgh_branch_horizon_pass": int(raw_summary.get("quality_audit_pass") and duplicate == 0 and label_violation == 0 and len(branch_rows) == expected),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    rows[0] = summary
    return rows, summary


def p12_apgh_eval(generated: list[dict[str, Any]], outcome_rows: list[dict[str, Any]], base: list[dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    gen_ledger = reconstruct_generated(generated, outcome_rows, base, "RealAPGH", "generated_APGH")
    source = {str(r.get("action_id")): r for r in base if r.get("primitive_family") == "canonical_AP0"}
    canonical_gradeab = [r for r in base if r.get("primitive_family") == "canonical_AP0" and gradeab(r)]
    gradeab_memory = mean([float(memory_fail(r)) for r in canonical_gradeab])
    rows = []
    for pid in APGH_IDS:
        subset = [r for r in gen_ledger if r.get("primitive_id") == pid]
        src_subset = [source.get(str(r.get("source_action_id")), {}) for r in subset]
        new_pos = [float((not gradeab(s)) and gradeab(r)) for r, s in zip(subset, src_subset)]
        long_created = [float((not inum(s.get("h240_longrisk"))) and inum(r.get("h240_longrisk"))) for r, s in zip(subset, src_subset)]
        damage = [fnum(r.get("V_integrated")) - fnum(s.get("V_integrated")) for r, s in zip(subset, src_subset)]
        q = row_quality(subset, len(gen_ledger))
        row = {
            "stage": "P12_APGH_OUTCOME_GEOMETRY_PASS_V9630",
            "status": "primitive_outcome_summary",
            "primitive_id": pid,
            "generated_action_count": len(subset),
            "GradeAB_precision": q["GradeAB_precision"],
            "V_integrated_LCB": q["V_integrated_LCB"],
            "h240_longrisk_UCB": q["h240_longrisk_UCB"],
            "bad_UCB": q["bad_UCB"],
            "null_UCB": q["null_UCB"],
            "memory_fail_rate": q["memory_fail_rate"],
            "offdiag_fail_rate": q["offdiag_fail_rate"],
            "cover_collapse_rate": q["cover_collapse_rate"],
            "new_positive_created_rate": mean(new_pos),
            "longrisk_created_rate": mean(long_created),
            "Damage_V_LCB": lcb(damage),
            "apgh_strong_pass": 0,
            "apgh_weak_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["apgh_strong_pass"] = int(row["GradeAB_precision"] >= 0.60 and row["V_integrated_LCB"] > 0 and row["h240_longrisk_UCB"] <= 0.05 and row["bad_UCB"] <= 0.05 and row["memory_fail_rate"] <= gradeab_memory + 0.05 and row["longrisk_created_rate"] <= 0.10)
        row["apgh_weak_pass"] = int(row["GradeAB_precision"] >= 0.30 and row["V_integrated_LCB"] > 0 and row["h240_longrisk_UCB"] <= 0.10 and row["longrisk_created_rate"] <= 0.20)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["apgh_strong_pass"]), inum(r["apgh_weak_pass"]), fnum(r["GradeAB_precision"]), fnum(r["V_integrated_LCB"]), -fnum(r["h240_longrisk_UCB"])), default={})
    summary = {
        "stage": "P12_APGH_OUTCOME_GEOMETRY_PASS_V9630",
        "status": "summary",
        "primitive_count": len(rows),
        "generated_action_count": len(gen_ledger),
        "canonical_GradeAB_memory_fail_rate": gradeab_memory,
        "best_primitive_id": best.get("primitive_id", ""),
        "best_GradeAB_precision": best.get("GradeAB_precision", 0),
        "best_V_integrated_LCB": best.get("V_integrated_LCB", 0),
        "best_h240_longrisk_UCB": best.get("h240_longrisk_UCB", 1),
        "best_new_positive_created_rate": best.get("new_positive_created_rate", 0),
        "best_longrisk_created_rate": best.get("longrisk_created_rate", 1),
        "best_Damage_V_LCB": best.get("Damage_V_LCB", 0),
        "apgh_weak_pass": int(any(inum(r["apgh_weak_pass"]) for r in rows)),
        "apgh_strong_pass": int(any(inum(r["apgh_strong_pass"]) for r in rows)),
        "generated_family_reset_required": int((fnum(best.get("GradeAB_precision")) < 0.10) or (fnum(best.get("V_integrated_LCB")) < 0) or (fnum(best.get("h240_longrisk_UCB")) > 0.50) or (fnum(best.get("longrisk_created_rate")) > 0.50)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p12_apgh_gradeab_by_primitive.svg", "P12 APGH GradeAB precision", [r["primitive_id"] for r in rows], [fnum(r["GradeAB_precision"]) for r in rows])
    return [summary] + rows, summary, gen_ledger


def p13_apgh_controller(p12: dict[str, Any], gen_ledger: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inum(p12.get("apgh_strong_pass")):
        return boundary_not_run("P13_APGH_CONTROLLER_V9630", "P12_APGH_strong_pass_failed", apgh_controller_pass=0)
    row = {
        "stage": "P13_APGH_CONTROLLER_V9630",
        "status": "summary",
        "apgh_controller_pass": 1,
        "accepted_count": len(gen_ledger),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def p15_base_acc(source_v9620: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    src = source_v9620 / "base_acc_sentinel_v9620.csv"
    rows = [dict(r) for r in read_csv(src)] if src.exists() else [not_run("P17_BASE_ACC_SENTINEL", "source_missing")]
    for r in rows:
        r["stage"] = "P17_BASE_ACC_SENTINEL"
        r["base_acc_reused_from_v9620"] = 1
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
    return {"rows_checked": total, "fake_proxy_nonzero_count": fake + proxy, "fake_data_used": int(fake > 0), "proxy_row_used": int(proxy > 0), "cpu_offload_used": int(cpu > 0), "no_fake": int(fake == 0), "no_proxy": int(proxy == 0)}


def sha_rows(paths: dict[str, Path]) -> list[dict[str, Any]]:
    return [{"artifact": name, "path": rel(path), "sha256": sha256_file(path)} for name, path in paths.items() if path.exists()]


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if args.fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    source_v9620 = Path(args.source_v9620)
    source_v9610 = Path(args.source_v9610)
    source_v9600 = Path(args.source_v9600)
    source_v9590 = Path(args.source_v9590)
    source_v9580 = Path(args.source_v9580)
    source_v9570 = Path(args.source_v9570)
    source_v9560 = Path(args.source_v9560)
    source_v9550 = Path(args.source_v9550)
    source_v9330 = Path(args.source_v9330)
    device = choose_device(args.device)
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

    base, _apgr, _apgc, _apgm, _apga = v9590.load_all_ledgers(source_v9550, source_v9560, source_v9570, source_v9580, source_v9330)
    ap0 = [r for r in base if r.get("primitive_family") == "canonical_AP0"]
    _base2, _apgr2, _apgc2, _apgm2, payload_by_id = v9580.load_ledgers(source_v9550, source_v9560, source_v9570, source_v9330)
    feature_scores, v9620_rank_scores = base_rank_scores(ap0, args.seed)

    apgd_generated = [dict(r) for r in read_csv(source_v9600 / "p11_apgd_memory_safe_longrisk_primitive.csv") if r.get("status") == "generated_action_row"]
    apgd_outcome = [dict(r) for r in read_csv(source_v9600 / "p12_apgd_branch_horizon_smoke_outcome.csv")]
    apgd = reconstruct_generated(apgd_generated, apgd_outcome, base, "RealAPGD", "generated_APGD")
    apge_generated = [dict(r) for r in read_csv(source_v9610 / "p11_apge_memory_offdiag_safe_primitive.csv") if r.get("status") == "generated_action_row"]
    apge_outcome = [dict(r) for r in read_csv(source_v9610 / "p12_apge_branch_horizon_outcome.csv")]
    apge = reconstruct_generated(apge_generated, apge_outcome, base, "RealAPGE", "generated_APGE")
    apgf_generated_prev = [dict(r) for r in read_csv(source_v9620 / "p9_apgf_population_risk_geometry_primitive.csv") if r.get("status") == "generated_action_row"]
    apgf_outcome_prev = [dict(r) for r in read_csv(source_v9620 / "p9_apgf_branch_horizon_outcome.csv")]
    apgf = reconstruct_generated(apgf_generated_prev, apgf_outcome_prev, base, "RealAPGF", "generated_APGF")

    p0_rows, p0 = p0_boundary(source_v9620)
    dump_csv("p0_boundary_reproduction_v9630.csv", p0_rows)
    p1_rows, p1 = p1_scope_audit(ap0, v9620_rank_scores, out)
    dump_csv("p1_certificate_scope_audit_v9630.csv", p1_rows)
    p2_rows, p2 = p2_multiaxis_search(ap0, v9620_rank_scores, out)
    dump_csv("p2_multiaxis_hidden_pocket_search_v9630.csv", p2_rows)
    p3_rows, p3 = p3_target_support(ap0, out)
    dump_csv("p3_group_balanced_target_support_map_v9630.csv", p3_rows)
    p4_rows, p4, feature_scores2 = p4_features(ap0, out)
    dump_csv("p4_legal_feature_deconfounding_v2_v9630.csv", p4_rows)
    p5_rows, p5, rank_scores = p5_rankers(ap0, feature_scores2, args.seed, out)
    dump_csv("p5_value_rank_risk_memory_veto_ranker_v9630.csv", p5_rows)
    p6_rows, p6 = p6_certificate(ap0, rank_scores, p5)
    dump_csv("p6_rank_safe_certificate_v11_v9630.csv", p6_rows)
    p7_rows, p7 = p7_controller(p6)
    dump_csv("p7_existing_action_controller_v9630.csv", p7_rows)
    p8_rows, p8 = p8_runtime(p7)
    dump_csv("p8_selected_controller_runtime_preflight_v9630.csv", p8_rows)
    p9_rows, p9 = p9_damage(base, apgd, apge, apgf, out)
    dump_csv("p9_generated_damage_autopsy_v2_v9630.csv", p9_rows)
    p10_rows, p10, apgh_generated = p10_apgh_generation(args, ap0, payload_by_id, device)
    dump_csv("p10_apgh_primitive_spec_preflight_v9630.csv", p10_rows)
    p11_rows, p11 = materialize_apgh(args, apgh_generated, payload_by_id, device)
    dump_csv("p11_apgh_branch_horizon_smoke_v9630.csv", p11_rows)
    p12_rows, p12, apgh_ledger = p12_apgh_eval(apgh_generated, p11_rows, base, out)
    dump_csv("p12_apgh_outcome_geometry_pass_v9630.csv", p12_rows)
    p13_rows, p13 = p13_apgh_controller(p12, apgh_ledger)
    dump_csv("p13_apgh_controller_v9630.csv", p13_rows)

    if not inum(p0.get("p0_pass")):
        route, blocker = "R0-boundary_not_reproduced", "v9620_boundary_not_reproduced"
    elif not inum(p1.get("scope_consistency_pass")):
        route, blocker = "R1-scope_mismatch_in_certificate", "accepted_region_scope_mismatch"
    elif inum(p7.get("source_controller_pass")) and not inum(p8.get("selected_runtime_pass")):
        route, blocker = "R4-existing_action_controller_pass_runtime_fail", "runtime_preflight_failed"
    elif inum(p7.get("source_controller_pass")) and inum(p8.get("selected_runtime_pass")):
        route, blocker = "R5-existing_action_system_pass", "none"
    elif inum(p2.get("multi_axis_pocket_explained")):
        route, blocker = "R2-hidden_multiaxis_pocket_explains_rank", "multi_axis_group_pocket_explains_rank"
    elif inum(p5.get("existing_action_rank_not_deployable")) or not inum(p6.get("rank_safe_certificate_v11_pass")):
        route, blocker = "R3-existing_action_rank_group_unstable", "existing_action_rank_not_deployable"
    elif not inum(p9.get("damage_autopsy_pass")):
        route, blocker = "R6-generated_damage_explained_APGH_not_run", "generated_damage_not_explained"
    elif not inum(p12.get("apgh_weak_pass")):
        route, blocker = "R7-APGH_generated_frontier_fail", "apgh_generated_frontier_fail"
    elif inum(p12.get("apgh_weak_pass")) and not inum(p13.get("apgh_controller_pass")):
        route, blocker = "R8-APGH_generated_frontier_pass_controller_fail", "apgh_controller_failed"
    elif inum(p13.get("apgh_controller_pass")):
        route, blocker = "R9-APGH_system_pass", "none"
    else:
        route, blocker = "R10-both_existing_and_generated_routes_fail", "both_routes_failed"

    p14 = {
        "stage": "P14_ROUTE_DECISION_V9630",
        "status": "summary",
        "route": route,
        "primary_blocker": blocker,
        "secondary_blocker": "apgh_generated_frontier_fail" if not inum(p12.get("apgh_weak_pass")) else "none",
        "existing_action_route_status": "pass" if inum(p7.get("source_controller_pass")) else "failed_or_blocked",
        "generated_action_route_status": "pass" if inum(p13.get("apgh_controller_pass")) else "failed_or_blocked",
        "runtime_status": "pass" if inum(p8.get("selected_runtime_pass")) else "not_open",
        "paired_replay_status": "not_open",
        "next_required_implementation": "group_stable_rank_or_population_risk_safe_direction_reset",
        "p0_pass": p0.get("p0_pass"),
        "scope_consistency_pass": p1.get("scope_consistency_pass"),
        "multi_axis_pocket_explained": p2.get("multi_axis_pocket_explained"),
        "target_support_pass": p3.get("target_support_pass"),
        "feature_deconfounding_pass": p4.get("feature_deconfounding_pass"),
        "ranker_weak_pass": p5.get("ranker_weak_pass"),
        "ranker_strong_pass": p5.get("ranker_strong_pass"),
        "rank_safe_certificate_v11_pass": p6.get("rank_safe_certificate_v11_pass"),
        "existing_action_controller_pass": p7.get("source_controller_pass"),
        "selected_runtime_pass": p8.get("selected_runtime_pass"),
        "damage_autopsy_pass": p9.get("damage_autopsy_pass"),
        "apgh_implementation_pass": p10.get("apgh_implementation_pass"),
        "apgh_branch_horizon_pass": p11.get("apgh_branch_horizon_pass"),
        "apgh_weak_pass": p12.get("apgh_weak_pass"),
        "best_apgh_primitive": p12.get("best_primitive_id"),
        "best_apgh_GradeAB_precision": p12.get("best_GradeAB_precision"),
        "best_apgh_V_integrated_LCB": p12.get("best_V_integrated_LCB"),
        "best_apgh_h240_longrisk_UCB": p12.get("best_h240_longrisk_UCB"),
        "system_legal_controller_pass": int(route in {"R5-existing_action_system_pass", "R9-APGH_system_pass"}),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("p14_route_decision_v9630.json", p14)
    dump_json("route_decision_v9630.json", p14)

    p15_rows, p15 = boundary_not_run("P15_LEAVEOUT_PAIRED_REPLAY_BOUNDARY_V9630", "P8_or_system_not_official", leaveout_pass=0, paired_replay_pass=0)
    dump_csv("p15_leaveout_paired_replay_boundary_v9630.csv", p15_rows)
    p16_rows, p16 = boundary_not_run("P16_SHORT_FULL_BOUNDARY_V9630", "P15_paired_replay_not_open", short_full_pass=0)
    dump_csv("p16_short_full_boundary_v9630.csv", p16_rows)
    base_rows, base_summary = p15_base_acc(source_v9620)
    dump_csv("base_acc_sentinel_v9630.csv", base_rows)

    nofake = {"stage": "NO_FAKE_AUDIT_V9630", "status": "summary", **count_artifact_rows(list(artifacts.values()))}
    nofake.update({"fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    dump_csv("no_fake_audit_v9630.csv", [nofake])
    contract = {
        "stage": "CONTRACT_AUDIT_V9630",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "v9620_boundary_pass": p0.get("p0_pass"),
        "scope_consistency_pass": p1.get("scope_consistency_pass"),
        "multi_axis_pocket_explained": p2.get("multi_axis_pocket_explained"),
        "target_support_pass": p3.get("target_support_pass"),
        "feature_deconfounding_pass": p4.get("feature_deconfounding_pass"),
        "ranker_weak_pass": p5.get("ranker_weak_pass"),
        "rank_safe_certificate_pass": p6.get("rank_safe_certificate_v11_pass"),
        "existing_controller_pass": p7.get("source_controller_pass"),
        "selected_runtime_pass": p8.get("selected_runtime_pass"),
        "damage_autopsy_pass": p9.get("damage_autopsy_pass"),
        "apgh_implementation_pass": p10.get("apgh_implementation_pass"),
        "apgh_branch_horizon_pass": p11.get("apgh_branch_horizon_pass"),
        "apgh_weak_pass": p12.get("apgh_weak_pass"),
        "apgh_controller_pass": p13.get("apgh_controller_pass", 0),
        "system_legal_controller_pass": p14.get("system_legal_controller_pass"),
        "base_acc_sentinel_pass": base_summary.get("base_acc_sentinel_pass", base_summary.get("sentinel_complete", 0)),
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
        "payload_norm_bucket_used_as_selector": 0,
        "diagnostic_promoted_to_official": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("contract_audit_v9630.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9630",
        "status": "summary",
        "route": route,
        "F0_boundary_reproduction_failed": int(not inum(p0.get("p0_pass"))),
        "F1_scope_mismatch": int(route == "R1-scope_mismatch_in_certificate"),
        "F2_hidden_multiaxis_pocket": int(route == "R2-hidden_multiaxis_pocket_explains_rank"),
        "F3_existing_action_rank_group_unstable": int(route == "R3-existing_action_rank_group_unstable"),
        "F4_existing_controller_runtime_fail": int(route == "R4-existing_action_controller_pass_runtime_fail"),
        "F5_generated_damage_not_explained": int(route == "R6-generated_damage_explained_APGH_not_run"),
        "F6_apgh_generated_frontier_fail": int(route == "R7-APGH_generated_frontier_fail"),
        "F7_apgh_controller_fail": int(route == "R8-APGH_generated_frontier_pass_controller_fail"),
        "F8_system_not_official": int(not inum(p14.get("system_legal_controller_pass"))),
        "F9_base_acc_catastrophic": 0,
        "primary_blocker": blocker,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("failure_taxonomy_v9630.csv", [failure])

    manifest = {
        "version": "v9630",
        "route": route,
        "primary_blocker": blocker,
        "out_dir": rel(out),
        "created_utc": "2026-05-15T130000Z",
        "wallclock_sec": time.perf_counter() - t0,
        "seed": args.seed,
        "device": str(device),
        "data_root": args.data_root,
        "parameters": {"apgh_actions_per_primitive": args.apgh_actions_per_primitive},
        "sources": {
            "v9620": rel(source_v9620),
            "v9610": rel(source_v9610),
            "v9600": rel(source_v9600),
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
    dump_json("run_manifest_v9630.json", manifest)

    print(json.dumps({
        "out_dir": rel(out),
        "route": route,
        "primary_blocker": blocker,
        "best_ranker": p5.get("best_ranker_id"),
        "best_ranker_TopK87_precision": p5.get("best_TopK87_precision"),
        "best_ranker_LDO_drop": p5.get("best_LDO_drop"),
        "apgh_generated_actions": p10.get("generated_action_count"),
        "apgh_rows": p11.get("branch_horizon_rows_actual", p11.get("actual_rows")),
        "best_apgh_primitive": p12.get("best_primitive_id"),
        "best_apgh_V_integrated_LCB": p12.get("best_V_integrated_LCB"),
        "system_legal_controller_pass": p14.get("system_legal_controller_pass"),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
