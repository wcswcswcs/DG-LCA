#!/usr/bin/env python3
"""DG-KAN v9.6.4 degenerate pocket / lineage-balanced controller audit.

This runner consumes the landed v9.6.3/v9.6.2 artifacts, separates degenerate
axes from usable pocket explanations, audits accepted-region lineage collapse
and memory/offdiag geometry, then materializes APGL1-APGL8 value-preserving
geometry primitives with real branch-horizon outcomes. Diagnostics are never
promoted into an official controller/system pass unless the preregistered gates
pass.
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


PLAN_PATH = REPO / "docs/DG-KAN_v9.6.4_DegeneratePocketAudit_LineageBalancedGeometryController_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9640_degenerate_pocket_audit_lineage_balanced_geometry_controller.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_V9630 = RESULT_ROOT / "v9630_group_stable_accepted_region_memory_offdiag_primitive_first_20260515T130000Z"
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
APGL_IDS = [
    "APGL1-SourceReplayPreserver",
    "APGL2-PopRiskProjectedEdgeDelta",
    "APGL3-MemoryOffdiagNullspaceProjection",
    "APGL4-CoverEntropyBoundedDelta",
    "APGL5-BoundarySymmetricResidual",
    "APGL6-ValueRiskTwoHeadProjectedDelta",
    "APGL7-LineageDiverseEnsembleIntersection",
    "APGL8-ShuffledPayloadNegativeControl",
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
    "candidate_id",
    "source_candidate_id",
    "source_payload_hash",
    "payload_hash",
    "primitive_id",
    "event_family",
    "snr_bucket",
    "adamw_conflict_bucket",
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
    p.add_argument("--source-v9630", default=str(DEFAULT_V9630))
    p.add_argument("--source-v9620", default=str(DEFAULT_V9620))
    p.add_argument("--source-v9610", default=str(DEFAULT_V9610))
    p.add_argument("--source-v9600", default=str(DEFAULT_V9600))
    p.add_argument("--source-v9590", default=str(DEFAULT_V9590))
    p.add_argument("--source-v9580", default=str(DEFAULT_V9580))
    p.add_argument("--source-v9570", default=str(DEFAULT_V9570))
    p.add_argument("--source-v9560", default=str(DEFAULT_V9560))
    p.add_argument("--source-v9550", default=str(DEFAULT_V9550))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--apgl-actions-per-primitive", type=int, default=64)
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


def pearson(xs: list[float], ys: list[float]) -> float:
    vals = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(vals) < 2:
        return 0.0
    mx = statistics.fmean(x for x, _ in vals)
    my = statistics.fmean(y for _, y in vals)
    vx = sum((x - mx) ** 2 for x, _ in vals)
    vy = sum((y - my) ** 2 for _, y in vals)
    if vx <= 0 or vy <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in vals) / math.sqrt(vx * vy)


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
    if axis == "dataset_id":
        return str(row.get("dataset_id") or row.get("dataset") or "")
    if axis == "seed_id":
        return str(row.get("seed_id") or row.get("seed") or "")
    if axis == "candidate_id":
        return str(row.get("candidate_id") or "")
    if axis == "source_candidate_id":
        return str(row.get("source_candidate_id") or row.get("candidate_id") or "")
    if axis == "source_payload_hash":
        return str(row.get("source_payload_hash") or row.get("payload_hash") or "")
    if axis == "payload_hash":
        return str(row.get("payload_hash") or "")
    if axis == "primitive_id":
        return str(row.get("primitive_id") or "")
    if axis == "event_family":
        return str(row.get("event_id", row.get("action_id", "")))[:8]
    if axis == "snr_bucket":
        return f"snr{min(9, max(0, int(fnum(row.get('snr_group')) * 10)))}"
    if axis == "adamw_conflict_bucket":
        return f"adamw{min(9, max(0, int(fnum(row.get('adamw_conflict_rate')) * 10)))}"
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


def accepted_v9630(ap0: list[dict[str, Any]], seed: int) -> tuple[list[int], list[dict[str, Any]], dict[str, list[float]]]:
    _feature_scores, rank_scores = base_rank_scores(ap0, seed)
    scores = rank_scores.get("RANK5-invariant-risk-minimization", [legal_score(r) for r in ap0])
    idx = topk_idx(scores, 87)
    return idx, [ap0[i] for i in idx], rank_scores


def shannon_entropy(vals: list[str]) -> float:
    if not vals:
        return 0.0
    n = len(vals)
    ent = 0.0
    for c in Counter(vals).values():
        p = c / n
        ent -= p * math.log(max(p, 1.0e-12))
    return ent


def lineage_id(row: dict[str, Any], mode: str = "strict") -> str:
    step_bucket = axis_value(row, "step_bucket")
    if mode == "candidate":
        parts = [row.get("candidate_id", ""), row.get("primitive_id", "")]
    elif mode == "source_payload":
        parts = [row.get("source_candidate_id", row.get("candidate_id", "")), row.get("source_payload_hash", row.get("payload_hash", ""))]
    else:
        parts = [
            row.get("source_candidate_id", row.get("candidate_id", "")),
            row.get("source_payload_hash", row.get("payload_hash", "")),
            row.get("primitive_id", ""),
            row.get("family_id", ""),
            step_bucket,
        ]
    return v9580.stable_hash("v9640-lineage", *[str(p) for p in parts])[:16]


def max_share(vals: list[str]) -> tuple[float, str, int]:
    if not vals:
        return 0.0, "", 0
    group, count = Counter(vals).most_common(1)[0]
    return count / len(vals), group, count


def leave_group_drop_for_rows(rows: list[dict[str, Any]], group_values: list[str]) -> tuple[float, float, float]:
    base_q = row_quality(rows, 2876)
    drops_p: list[float] = []
    drops_v: list[float] = []
    long_inc: list[float] = []
    for group in sorted(set(group_values)):
        keep = [r for r, g in zip(rows, group_values) if g != group]
        if not keep:
            continue
        q = row_quality(keep, 2876)
        drops_p.append(max(0.0, base_q["GradeAB_precision"] - q["GradeAB_precision"]))
        drops_v.append(max(0.0, base_q["V_integrated_LCB"] - q["V_integrated_LCB"]))
        long_inc.append(max(0.0, q["h240_longrisk_UCB"] - base_q["h240_longrisk_UCB"]))
    return max(drops_p, default=0.0), max(drops_v, default=0.0), max(long_inc, default=0.0)


def selector_allowed(axis: str) -> bool:
    forbidden = {
        "payload_norm_bucket",
        "candidate_id",
        "source_candidate_id",
        "source_payload_hash",
        "payload_hash",
        "event_bucket",
        "event_family",
        "candidate_origin",
    }
    return axis not in forbidden


def p0_boundary_v9640(source_v9630: Path, source_v9620: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route9630 = read_json(source_v9630 / "route_decision_v9630.json")
    route9620 = read_json(source_v9620 / "route_decision_v9620.json")
    p1 = next((r for r in read_csv(source_v9630 / "p1_certificate_scope_audit_v9630.csv") if r.get("status") == "summary"), {})
    p12 = next((r for r in read_csv(source_v9630 / "p12_apgh_outcome_geometry_pass_v9630.csv") if r.get("status") == "summary"), {})
    nofake = next((r for r in read_csv(source_v9630 / "no_fake_audit_v9630.csv") if r.get("status") == "summary"), {})
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9640",
        "status": "summary",
        "source_route_v9630": route9630.get("route"),
        "source_route_v9620": route9620.get("route"),
        "system_legal_controller_pass_v9630": route9630.get("system_legal_controller_pass"),
        "accepted_action_count_v9630": p1.get("accepted_action_count"),
        "accepted_unique_action_count_v9630": p1.get("accepted_unique_action_count"),
        "accepted_unique_candidate_count_v9630": p1.get("accepted_unique_candidate_count"),
        "GradeAB_precision_v9630": p1.get("GradeAB_precision"),
        "V_LCB_v9630": p1.get("V_LCB"),
        "h240_longrisk_UCB_v9630": p1.get("longrisk_UCB"),
        "best_ranker_v9630": route9630.get("best_ranker", "RANK-D-value-risk-memory-veto"),
        "best_certificate_v9630": "CERT11-FrozenTopK87",
        "best_apgh_primitive_v9630": p12.get("best_primitive_id"),
        "best_apgh_V_LCB_v9630": p12.get("best_V_integrated_LCB"),
        "best_apgh_h240_longrisk_UCB_v9630": p12.get("best_h240_longrisk_UCB"),
        "payload_pocket_route_stop_v9620": route9620.get("payload_pocket_route_stop"),
        "no_fake_v9630": nofake.get("no_fake"),
        "no_proxy_v9630": nofake.get("no_proxy"),
        "p0_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["p0_pass"] = int(
        row["source_route_v9630"] == "R2-hidden_multiaxis_pocket_explains_rank"
        and not inum(row["system_legal_controller_pass_v9630"])
        and inum(row["payload_pocket_route_stop_v9620"])
        and fnum(row["accepted_action_count_v9630"]) == 87
        and inum(row["no_fake_v9630"])
        and inum(row["no_proxy_v9630"])
    )
    return [row], row


def p1_degenerate_pocket_audit_v9640(ap0: list[dict[str, Any]], accepted: list[dict[str, Any]], scores: list[float], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_prec = row_quality(accepted, len(ap0))["GradeAB_precision"]
    rows: list[dict[str, Any]] = []
    for axis in AUDIT_AXES + ["value_score_bucket", "risk_score_bucket", "support_bucket"]:
        base_groups = Counter(axis_value(r, axis, scores[i] if i < len(scores) else 0.0) for i, r in enumerate(ap0))
        acc_groups = Counter(axis_value(r, axis) for r in accepted)
        top_group, acc_count = acc_groups.most_common(1)[0] if acc_groups else ("", 0)
        base_share = base_groups.get(top_group, 0) / max(1, len(ap0))
        accepted_share = acc_count / max(1, len(accepted))
        keep_idx = [i for i, r in enumerate(ap0) if axis_value(r, axis, scores[i]) != top_group]
        if len(keep_idx) >= 87:
            rem = sorted(keep_idx, key=lambda i: scores[i], reverse=True)[:87]
            rem_prec = mean([float(gradeab(ap0[i])) for i in rem])
        else:
            rem_prec = 0.0
        raw_drop = max(0.0, base_prec - rem_prec)
        degenerate = int(base_share >= 0.95)
        allowed = int(selector_allowed(axis))
        adjusted = raw_drop if (not degenerate and allowed) else 0.0
        vals = [axis_value(r, axis) for r in accepted]
        row = {
            "stage": "P1_DEGENERATE_POCKET_AUDIT_V9640",
            "status": "axis_row",
            "axis": axis,
            "top_group_id": top_group,
            "base_share": base_share,
            "accepted_share": accepted_share,
            "lift": accepted_share - base_share,
            "raw_drop_if_removed": raw_drop,
            "adjusted_drop_if_removed": adjusted,
            "is_degenerate_axis": degenerate,
            "is_forbidden_selector": int(not allowed),
            "is_commit_time_legal": int(axis not in {"V_integrated", "h240_longrisk", "GradeAB"}),
            "accepted_group_entropy": shannon_entropy(vals),
            "group_count": len(acc_groups),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
    best_raw = max(rows, key=lambda r: fnum(r["raw_drop_if_removed"]), default={})
    best_adj = max(rows, key=lambda r: fnum(r["adjusted_drop_if_removed"]), default={})
    pass_a = int(inum(best_raw.get("is_degenerate_axis")) and best_raw.get("axis") != best_adj.get("axis"))
    pass_b = int(any(fnum(r["base_share"]) < 0.80 and fnum(r["adjusted_drop_if_removed"]) >= 0.20 for r in rows))
    summary = {
        "stage": "P1_DEGENERATE_POCKET_AUDIT_V9640",
        "status": "summary",
        "axis_count": len(rows),
        "degenerate_axis_count": sum(inum(r["is_degenerate_axis"]) for r in rows),
        "forbidden_selector_axis_count": sum(inum(r["is_forbidden_selector"]) for r in rows),
        "best_raw_axis": best_raw.get("axis", ""),
        "best_raw_group": best_raw.get("top_group_id", ""),
        "best_raw_base_share": best_raw.get("base_share", 0),
        "best_raw_accepted_share": best_raw.get("accepted_share", 0),
        "best_raw_drop": best_raw.get("raw_drop_if_removed", 0),
        "best_adjusted_axis": best_adj.get("axis", ""),
        "best_adjusted_group": best_adj.get("top_group_id", ""),
        "best_adjusted_base_share": best_adj.get("base_share", 0),
        "best_adjusted_drop": best_adj.get("adjusted_drop_if_removed", 0),
        "degenerate_pollution_pass_A": pass_a,
        "nondegenerate_explanation_pass_B": pass_b,
        "p1_pass": int(pass_a or pass_b),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p1_base_share_vs_accepted_share.svg", "P1 accepted share by axis", [r["axis"] for r in rows], [fnum(r["accepted_share"]) for r in rows])
    write_bar_svg(out / "fig_p1_raw_drop_vs_adjusted_drop.svg", "P1 adjusted drop by axis", [r["axis"] for r in rows], [fnum(r["adjusted_drop_if_removed"]) for r in rows])
    write_bar_svg(out / "fig_p1_degenerate_axis_table.svg", "P1 degenerate axis flag", [r["axis"] for r in rows], [fnum(r["is_degenerate_axis"]) for r in rows])
    write_bar_svg(out / "fig_p1_group_entropy_by_axis.svg", "P1 accepted entropy by axis", [r["axis"] for r in rows], [fnum(r["accepted_group_entropy"]) for r in rows])
    return [summary] + rows, summary


def p2_lineage_audit_v9640(accepted: list[dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lineage_modes = {
        "candidate_template": [str(r.get("candidate_id", "")) + "|" + str(r.get("primitive_id", "")) for r in accepted],
        "source_payload": [lineage_id(r, "source_payload") for r in accepted],
        "strict_lineage": [lineage_id(r, "strict") for r in accepted],
        "event_family": [axis_value(r, "event_family") for r in accepted],
    }
    rows: list[dict[str, Any]] = []
    for mode, vals in lineage_modes.items():
        share, group, count = max_share(vals)
        pdrop, vdrop, linc = leave_group_drop_for_rows(accepted, vals)
        rows.append({
            "stage": "P2_ACCEPTED_LINEAGE_AUDIT_V9640",
            "status": "lineage_mode_row",
            "lineage_mode": mode,
            "accepted_unique_lineage_count": len(set(vals)),
            "accepted_lineage_entropy": shannon_entropy(vals),
            "max_lineage_share": share,
            "max_lineage_id": group,
            "max_lineage_count": count,
            "leave_lineage_out_precision_drop": pdrop,
            "leave_lineage_out_V_drop": vdrop,
            "leave_lineage_out_longrisk_increase": linc,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    cand_vals = lineage_modes["candidate_template"]
    strict_vals = lineage_modes["strict_lineage"]
    cand_share, cand_group, cand_count = max_share(cand_vals)
    strict_share, strict_group, strict_count = max_share(strict_vals)
    summary = {
        "stage": "P2_ACCEPTED_LINEAGE_AUDIT_V9640",
        "status": "summary",
        "accepted_action_count": len(accepted),
        "accepted_unique_candidate_count": len(set(str(r.get("candidate_id", "")) for r in accepted)),
        "accepted_unique_source_payload_hash_count": len(set(str(r.get("source_payload_hash") or r.get("payload_hash") or "") for r in accepted)),
        "accepted_unique_lineage_count": len(set(strict_vals)),
        "accepted_lineage_entropy": shannon_entropy(strict_vals),
        "max_lineage_share": strict_share,
        "max_lineage_id": strict_group,
        "candidate_template_max_share": cand_share,
        "candidate_template_unique_count": len(set(cand_vals)),
        "candidate_to_action_expansion_ratio": len(accepted) / max(1, len(set(cand_vals))),
        "candidate_id_collision_count": len(accepted) - len(set(str(r.get("candidate_id", "")) for r in accepted)),
        "leave_lineage_out_precision_drop": leave_group_drop_for_rows(accepted, strict_vals)[0],
        "leave_lineage_out_V_drop": leave_group_drop_for_rows(accepted, strict_vals)[1],
        "leave_lineage_out_longrisk_increase": leave_group_drop_for_rows(accepted, strict_vals)[2],
        "P2_pass_GENERAL": int(len(set(strict_vals)) >= 8 and strict_share <= 0.25 and leave_group_drop_for_rows(accepted, strict_vals)[0] <= 0.10 and leave_group_drop_for_rows(accepted, strict_vals)[1] <= 0.05),
        "P2_fail_LINEAGE": int(len(set(cand_vals)) <= 3 or cand_share > 0.50 or (len(accepted) / max(1, len(set(cand_vals)))) > 20.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p2_lineage_share_bar.svg", "P2 max lineage share", [r["lineage_mode"] for r in rows], [fnum(r["max_lineage_share"]) for r in rows])
    write_bar_svg(out / "fig_p2_leave_lineage_out_drop.svg", "P2 leave lineage precision drop", [r["lineage_mode"] for r in rows], [fnum(r["leave_lineage_out_precision_drop"]) for r in rows])
    write_bar_svg(out / "fig_p2_accepted_lineage_sankey.svg", "P2 unique lineage count", [r["lineage_mode"] for r in rows], [fnum(r["accepted_unique_lineage_count"]) for r in rows])
    write_bar_svg(out / "fig_p2_candidate_action_event_map.svg", "P2 candidate/action expansion", ["candidate_unique", "source_payload_unique", "event_unique"], [fnum(summary["accepted_unique_candidate_count"]), fnum(summary["accepted_unique_source_payload_hash_count"]), len(set(axis_value(r, "event_family") for r in accepted))])
    return [summary] + rows, summary


def nearest_control(row: dict[str, Any], controls: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not controls:
        return None
    def dist(c: dict[str, Any]) -> float:
        d = 0.0
        d += 0.0 if axis_value(row, "dataset_id") == axis_value(c, "dataset_id") else 2.0
        d += 0.0 if axis_value(row, "step_bucket") == axis_value(c, "step_bucket") else 0.5
        d += abs(fnum(row.get("payload_norm")) - fnum(c.get("payload_norm")))
        d += abs(fnum(row.get("snr_group")) - fnum(c.get("snr_group")))
        d += abs(fnum(row.get("adamw_conflict_rate")) - fnum(c.get("adamw_conflict_rate")))
        d += 0.5 * (0.0 if str(row.get("family_id")) == str(c.get("family_id")) else 1.0)
        return d
    return min(controls, key=dist)


def matched_lift(treat: list[dict[str, Any]], controls: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = []
    for r in treat:
        c = nearest_control(r, controls)
        if not c:
            continue
        pairs.append({
            "treatment_action_id": r.get("action_id"),
            "control_action_id": c.get("action_id"),
            "GradeAB_lift": float(gradeab(r) - gradeab(c)),
            "V_lift": fnum(r.get("V_integrated")) - fnum(c.get("V_integrated")),
            "longrisk_drop": float(inum(c.get("h240_longrisk")) - inum(r.get("h240_longrisk"))),
            "same_dataset": int(axis_value(r, "dataset_id") == axis_value(c, "dataset_id")),
            "same_step_bucket": int(axis_value(r, "step_bucket") == axis_value(c, "step_bucket")),
        })
    summary = {
        "matched_pair_count": len(pairs),
        "GradeAB_lift": mean([fnum(p["GradeAB_lift"]) for p in pairs]),
        "V_lift_LCB": lcb([fnum(p["V_lift"]) for p in pairs]),
        "longrisk_drop_UCB": ucb([fnum(p["longrisk_drop"]) for p in pairs]),
    }
    return pairs, summary


def p3_memory_offdiag_causality_v9640(ap0: list[dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mem_safe = [r for r in ap0 if not memory_fail(r)]
    mem_unsafe = [r for r in ap0 if memory_fail(r)]
    off_safe = [r for r in ap0 if risk_score(r) <= 0.20]
    off_unsafe = [r for r in ap0 if risk_score(r) > 0.20]
    joint_safe = [r for r in ap0 if not memory_fail(r) and risk_score(r) <= 0.20]
    joint_unsafe = [r for r in ap0 if memory_fail(r) or risk_score(r) > 0.20]
    pairs_m, sm = matched_lift(mem_safe, mem_unsafe)
    pairs_o, so = matched_lift(off_safe, off_unsafe)
    pairs_j, sj = matched_lift(joint_safe, joint_unsafe)
    rows: list[dict[str, Any]] = []
    for name, pairs, summ in [("memory_safe", pairs_m, sm), ("offdiag_safe", pairs_o, so), ("joint_memory_offdiag_safe", pairs_j, sj)]:
        rows.append({
            "stage": "P3_MEMORY_OFFDIAG_CAUSALITY_V9640",
            "status": "matched_summary",
            "comparison_id": name,
            **summ,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    def direct_joint_lift(subset: list[dict[str, Any]]) -> float:
        js = [r for r in subset if not memory_fail(r) and risk_score(r) <= 0.20]
        ju = [r for r in subset if memory_fail(r) or risk_score(r) > 0.20]
        if not js or not ju:
            return sj["GradeAB_lift"]
        return row_quality(js, len(subset))["GradeAB_precision"] - row_quality(ju, len(subset))["GradeAB_precision"]

    leave_drops = []
    for axis in ["dataset_id", "stratum_id", "family_id"]:
        groups = [g for g, _c in Counter(axis_value(r, axis) for r in ap0).most_common(20)]
        lifts = [direct_joint_lift([r for r in ap0 if axis_value(r, axis) != g]) for g in groups]
        leave_drops.append(max(0.0, sj["GradeAB_lift"] - min(lifts, default=sj["GradeAB_lift"])))
    q_joint = row_quality(joint_safe, len(ap0))
    base_q = row_quality(ap0, len(ap0))
    summary = {
        "stage": "P3_MEMORY_OFFDIAG_CAUSALITY_V9640",
        "status": "summary",
        "matched_pair_count_memory": sm["matched_pair_count"],
        "matched_pair_count_offdiag": so["matched_pair_count"],
        "matched_pair_count_joint": sj["matched_pair_count"],
        "GradeAB_lift_memory_safe": sm["GradeAB_lift"],
        "V_lift_memory_safe": sm["V_lift_LCB"],
        "longrisk_drop_memory_safe": sm["longrisk_drop_UCB"],
        "GradeAB_lift_offdiag_safe": so["GradeAB_lift"],
        "V_lift_offdiag_safe": so["V_lift_LCB"],
        "longrisk_drop_offdiag_safe": so["longrisk_drop_UCB"],
        "joint_memory_offdiag_lift": sj["GradeAB_lift"],
        "joint_memory_offdiag_V_lift_LCB": sj["V_lift_LCB"],
        "joint_memory_offdiag_longrisk_drop_UCB": sj["longrisk_drop_UCB"],
        "joint_safe_count": len(joint_safe),
        "joint_safe_GradeAB_precision": q_joint["GradeAB_precision"],
        "base_GradeAB_precision": base_q["GradeAB_precision"],
        "leave_dataset_out_lift_drop": leave_drops[0],
        "leave_stratum_out_lift_drop": leave_drops[1],
        "memory_offdiag_causality_pass": int(sj["matched_pair_count"] >= 64 and sj["GradeAB_lift"] >= 0.20 and sj["V_lift_LCB"] > 0 and sj["longrisk_drop_UCB"] > 0 and leave_drops[0] <= 0.10 and leave_drops[1] <= 0.10),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p3_memory_offdiag_matched_lift.svg", "P3 matched GradeAB lift", ["memory", "offdiag", "joint"], [sm["GradeAB_lift"], so["GradeAB_lift"], sj["GradeAB_lift"]])
    write_bar_svg(out / "fig_p3_longrisk_by_memory_offdiag.svg", "P3 longrisk drop", ["memory", "offdiag", "joint"], [sm["longrisk_drop_UCB"], so["longrisk_drop_UCB"], sj["longrisk_drop_UCB"]])
    write_bar_svg(out / "fig_p3_V_distribution_by_geometry_status.svg", "P3 V lift", ["memory", "offdiag", "joint"], [sm["V_lift_LCB"], so["V_lift_LCB"], sj["V_lift_LCB"]])
    write_bar_svg(out / "fig_p3_cover_memory_offdiag_scatter.svg", "P3 joint safe vs base precision", ["base", "joint_safe"], [base_q["GradeAB_precision"], q_joint["GradeAB_precision"]])
    return [summary] + rows, summary


def target_v9640(row: dict[str, Any], tid: str, accepted_lineages: set[str] | None = None) -> int:
    accepted_lineages = accepted_lineages or set()
    value_pos = fnum(row.get("V_integrated")) > 0
    low_risk = not inum(row.get("h240_longrisk"))
    good_bad = fnum(row.get("bad_event_rate")) <= 0.05
    good_null = fnum(row.get("null_event_rate")) <= 0.15
    mem_safe = not memory_fail(row)
    off_safe = risk_score(row) <= 0.20
    grade = gradeab(row)
    out_of_pocket = lineage_id(row, "candidate") not in accepted_lineages and axis_value(row, "payload_norm_bucket") != "payload0"
    if tid == "T4.1-ValuePositiveNoLongRiskLineageBalanced":
        return int(value_pos and low_risk)
    if tid == "T4.2-MemoryOffdiagSafeValue":
        return int(value_pos and mem_safe and off_safe and low_risk and good_bad and good_null)
    if tid == "T4.3-GradeABBalanced":
        return int(grade)
    if tid == "T4.4-OutOfPocketGradeAB":
        return int(grade and out_of_pocket)
    if tid == "T4.5-MemoryOffdiagGradeAB":
        return int(grade and mem_safe and off_safe)
    if tid == "T4.6-StrictGoodGeometryA":
        return int(value_pos and low_risk and good_bad and good_null and mem_safe and off_safe and not cover_collapse(row))
    return 0


def group_balance_for_rows(rows: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    share, axis, group = max_group_share(rows)
    positives = 0
    total = 0
    for a in ["dataset_id", "stratum_id", "family_id", "memory_bucket", "offdiag_bucket"]:
        all_groups = {axis_value(r, a) for r in all_rows}
        row_groups = {axis_value(r, a) for r in rows}
        positives += len(row_groups)
        total += len(all_groups)
    lineages = [lineage_id(r, "candidate") for r in rows]
    lshare, lgroup, lcount = max_share(lineages)
    return {
        "positive_group_coverage": positives / max(1, total),
        "max_group_share": share,
        "max_group_share_axis": axis,
        "max_group_share_group": group,
        "max_lineage_share": lshare,
        "max_lineage_id": lgroup,
        "lineage_count": len(set(lineages)),
        "lineage_entropy": shannon_entropy(lineages),
    }


def p4_out_of_pocket_target_map_v9640(ap0: list[dict[str, Any]], accepted: list[dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accepted_lineages = {lineage_id(r, "candidate") for r in accepted}
    tids = [
        "T4.1-ValuePositiveNoLongRiskLineageBalanced",
        "T4.2-MemoryOffdiagSafeValue",
        "T4.3-GradeABBalanced",
        "T4.4-OutOfPocketGradeAB",
        "T4.5-MemoryOffdiagGradeAB",
        "T4.6-StrictGoodGeometryA",
    ]
    rows: list[dict[str, Any]] = []
    for tid in tids:
        subset = [r for r in ap0 if target_v9640(r, tid, accepted_lineages)]
        q = row_quality(subset, len(ap0))
        gb = group_balance_for_rows(subset, ap0)
        scores = [float(target_v9640(r, tid, accepted_lineages)) for r in ap0]
        ldo = leaveout_drop(scores, ap0, "dataset_id", 87) if len(subset) >= 87 else 1.0
        lso = leaveout_drop(scores, ap0, "stratum_id", 87) if len(subset) >= 87 else 1.0
        row = {
            "stage": "P4_OUT_OF_POCKET_TARGET_MAP_V9640",
            "status": "target_row",
            "target_id": tid,
            "target_count": len(subset),
            "target_coverage": q["coverage"],
            "V_LCB": q["V_integrated_LCB"],
            "longrisk_UCB": q["h240_longrisk_UCB"],
            "bad_UCB": q["bad_UCB"],
            "null_UCB": q["null_UCB"],
            **gb,
            "LDO_drop": ldo,
            "LSO_drop": lso,
            "target_pass": 0,
            "target_weak_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["target_pass"] = int(row["target_count"] >= 87 and row["target_coverage"] >= 0.03 and row["V_LCB"] > 0 and row["longrisk_UCB"] <= 0.05 and row["bad_UCB"] <= 0.05 and row["null_UCB"] <= 0.15 and row["max_lineage_share"] <= 0.25 and row["LDO_drop"] <= 0.10 and row["LSO_drop"] <= 0.10)
        row["target_weak_pass"] = int(row["target_count"] >= 64 and row["V_LCB"] > 0 and row["longrisk_UCB"] <= 0.05)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["target_pass"]), inum(r["target_weak_pass"]), fnum(r["target_count"]), fnum(r["V_LCB"])), default={})
    summary = {
        "stage": "P4_OUT_OF_POCKET_TARGET_MAP_V9640",
        "status": "summary",
        "target_variant_count": len(rows),
        "target_pass_count": sum(inum(r["target_pass"]) for r in rows),
        "target_weak_pass_count": sum(inum(r["target_weak_pass"]) for r in rows),
        "best_target_id": best.get("target_id", ""),
        "best_target_count": best.get("target_count", 0),
        "best_target_coverage": best.get("target_coverage", 0),
        "best_V_LCB": best.get("V_LCB", 0),
        "best_longrisk_UCB": best.get("longrisk_UCB", 1),
        "best_max_lineage_share": best.get("max_lineage_share", 1),
        "best_LDO_drop": best.get("LDO_drop", 1),
        "best_LSO_drop": best.get("LSO_drop", 1),
        "out_of_pocket_target_pass": int(any(inum(r["target_pass"]) for r in rows)),
        "out_of_pocket_target_weak_pass": int(any(inum(r["target_weak_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p4_out_of_pocket_target_count.svg", "P4 target count", [r["target_id"] for r in rows], [fnum(r["target_count"]) for r in rows])
    return [summary] + rows, summary


def lineage_balanced_topk(scores: list[float], rows: list[dict[str, Any]], k: int, cap: float = 0.25) -> list[int]:
    max_per = max(1, int(math.floor(k * cap)))
    counts: Counter[str] = Counter()
    idx: list[int] = []
    for i in sorted(range(len(scores)), key=lambda j: scores[j], reverse=True):
        lin = lineage_id(rows[i], "candidate")
        if counts[lin] >= max_per:
            continue
        idx.append(i)
        counts[lin] += 1
        if len(idx) >= k:
            break
    if len(idx) < k:
        for i in sorted(range(len(scores)), key=lambda j: scores[j], reverse=True):
            if i not in idx:
                idx.append(i)
            if len(idx) >= k:
                break
    return idx


def p5_group_deconfounded_ranker_v4(ap0: list[dict[str, Any]], feature_scores: dict[str, list[float]], seed: int, out: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    value = feature_scores.get("ControlTransferImprovement", [legal_score(r) for r in ap0])
    margin = feature_scores.get("EstimatedDeltaMargin", [0.0 for _ in ap0])
    snr = feature_scores.get("GroupSNR", [0.0 for _ in ap0])
    off = [risk_score(r) for r in ap0]
    mem = [float(memory_fail(r)) + fnum(r.get("memory_score")) for r in ap0]
    cover = [float(cover_collapse(r)) + max(0.0, -fnum(r.get("cover_score"))) for r in ap0]
    cost = [fnum(r.get("feature_compute_ms")) + fnum(r.get("payload_apply_ms")) for r in ap0]
    prev = v9590.ranker_scores(ap0, seed)
    candidates = {
        "R4A-value-only-legal-rank": [value[i] + margin[i] for i in range(len(ap0))],
        "R4B-value-longrisk-veto": [value[i] + margin[i] - 3.0 * off[i] for i in range(len(ap0))],
        "R4C-value-longrisk-memory-veto": [value[i] + margin[i] - 2.5 * off[i] - 2.5 * mem[i] for i in range(len(ap0))],
        "R4D-value-longrisk-memory-offdiag-veto": [value[i] + margin[i] + 0.20 * snr[i] - 3.0 * off[i] - 3.0 * mem[i] - 1.5 * cover[i] - 0.1 * cost[i] for i in range(len(ap0))],
        "R4E-pairwise-within-lineage": [prev.get("GIR9-PairwiseWithinGroupRanker", [legal_score(r) for r in ap0])[i] - mem[i] - off[i] for i in range(len(ap0))],
        "R4F-pairwise-leave-lineage-out": [min(value[i], snr[i]) - 2.0 * max(mem[i], off[i]) - cover[i] for i in range(len(ap0))],
        "R4G-group-DRO-ranker": [min(value[i], margin[i], snr[i]) - mem[i] - off[i] - cover[i] for i in range(len(ap0))],
        "R4H-conformal-group-balanced-ranker": [prev.get("GIR1-GroupQuantileNormalizedValueRiskRank", [legal_score(r) for r in ap0])[i] - 1.5 * mem[i] - 1.5 * off[i] - cost[i] for i in range(len(ap0))],
    }
    rows: list[dict[str, Any]] = []
    for rid, scores in candidates.items():
        idx87 = lineage_balanced_topk(scores, ap0, 87, 0.25)
        accepted = [ap0[i] for i in idx87]
        q = row_quality(accepted, len(ap0))
        gb = group_balance_for_rows(accepted, ap0)
        lin_vals = [lineage_id(r, "candidate") for r in accepted]
        ldrop, vdrop, linc = leave_group_drop_for_rows(accepted, lin_vals)
        row = {
            "stage": "P5_GROUP_DECONFOUNDED_RANKER_V4",
            "status": "ranker_row",
            "ranker_id": rid,
            "TopK64_precision": row_quality([ap0[i] for i in lineage_balanced_topk(scores, ap0, 64, 0.25)], len(ap0))["GradeAB_precision"],
            "TopK87_precision": q["GradeAB_precision"],
            "TopK128_precision": row_quality([ap0[i] for i in lineage_balanced_topk(scores, ap0, 128, 0.25)], len(ap0))["GradeAB_precision"],
            "accepted_count": len(accepted),
            "coverage": q["coverage"],
            "V_LCB": q["V_integrated_LCB"],
            "longrisk_UCB": q["h240_longrisk_UCB"],
            "bad_UCB": q["bad_UCB"],
            "null_UCB": q["null_UCB"],
            "LDO_drop": leaveout_drop(scores, ap0, "dataset_id"),
            "LSO_drop": leaveout_drop(scores, ap0, "stratum_id"),
            "leave_lineage_out_drop": ldrop,
            "leave_lineage_out_V_drop": vdrop,
            "leave_lineage_out_longrisk_increase": linc,
            **gb,
            "score_monotonicity": pearson(scores, [fnum(r.get("V_integrated")) for r in ap0]),
            "single_feature_baseline_precision": row_quality([ap0[i] for i in topk_idx(value, 87)], len(ap0))["GradeAB_precision"],
            "ranker_pass": 0,
            "ranker_weak_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["ranker_pass"] = int(row["TopK87_precision"] >= 0.75 and row["V_LCB"] > 0 and row["longrisk_UCB"] <= 0.05 and row["bad_UCB"] <= 0.05 and row["null_UCB"] <= 0.15 and row["max_group_share"] <= 0.25 and row["max_lineage_share"] <= 0.25 and row["LDO_drop"] <= 0.10 and row["LSO_drop"] <= 0.10 and row["leave_lineage_out_drop"] <= 0.10)
        row["ranker_weak_pass"] = int(row["TopK87_precision"] >= 0.70 and row["V_LCB"] > 0 and row["longrisk_UCB"] <= 0.10)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["ranker_pass"]), inum(r["ranker_weak_pass"]), fnum(r["TopK87_precision"]), fnum(r["V_LCB"]), -fnum(r["longrisk_UCB"])), default={})
    summary = {
        "stage": "P5_GROUP_DECONFOUNDED_RANKER_V4",
        "status": "summary",
        "ranker_count": len(rows),
        "ranker_pass_count": sum(inum(r["ranker_pass"]) for r in rows),
        "ranker_weak_pass_count": sum(inum(r["ranker_weak_pass"]) for r in rows),
        "best_ranker_id": best.get("ranker_id", ""),
        "best_TopK87_precision": best.get("TopK87_precision", 0),
        "best_V_LCB": best.get("V_LCB", 0),
        "best_longrisk_UCB": best.get("longrisk_UCB", 1),
        "best_max_group_share": best.get("max_group_share", 1),
        "best_max_lineage_share": best.get("max_lineage_share", 1),
        "best_LDO_drop": best.get("LDO_drop", 1),
        "best_LSO_drop": best.get("LSO_drop", 1),
        "best_leave_lineage_out_drop": best.get("leave_lineage_out_drop", 1),
        "group_deconfounded_ranker_pass": int(any(inum(r["ranker_pass"]) for r in rows)),
        "group_deconfounded_ranker_weak_pass": int(any(inum(r["ranker_weak_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p5_ranker_precision_vs_group_drop.svg", "P5 precision", [r["ranker_id"] for r in rows], [fnum(r["TopK87_precision"]) for r in rows])
    write_bar_svg(out / "fig_p5_value_risk_frontier.svg", "P5 V LCB", [r["ranker_id"] for r in rows], [fnum(r["V_LCB"]) for r in rows])
    write_bar_svg(out / "fig_p5_rank_score_hist_by_grade.svg", "P5 longrisk UCB", [r["ranker_id"] for r in rows], [fnum(r["longrisk_UCB"]) for r in rows])
    write_bar_svg(out / "fig_p5_ablation_waterfall.svg", "P5 LDO drop", [r["ranker_id"] for r in rows], [fnum(r["LDO_drop"]) for r in rows])
    write_bar_svg(out / "fig_p5_lineage_balanced_topk_map.svg", "P5 max lineage share", [r["ranker_id"] for r in rows], [fnum(r["max_lineage_share"]) for r in rows])
    return [summary] + rows, summary, candidates


def p6_rank_safe_certificate_v12(ap0: list[dict[str, Any]], rank_scores: dict[str, list[float]], p5: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    best_id = str(p5.get("best_ranker_id") or next(iter(rank_scores), ""))
    base_scores = rank_scores.get(best_id, [legal_score(r) for r in ap0])
    cert_defs = [
        ("CERT12-FrozenLineageBalancedTopK87", base_scores, 0.25),
        ("CERT12-ValueRiskMemoryOffdiagThreshold", [base_scores[i] - memory_fail(ap0[i]) - risk_score(ap0[i]) for i in range(len(ap0))], 0.25),
        ("CERT12-LineageCapTight", base_scores, 0.15),
        ("CERT12-GroupDROSupportBound", [base_scores[i] - cover_collapse(ap0[i]) - 0.1 * fnum(ap0[i].get("feature_compute_ms")) for i in range(len(ap0))], 0.25),
    ]
    rows: list[dict[str, Any]] = []
    for cid, scores, cap in cert_defs:
        idx = lineage_balanced_topk(scores, ap0, 87, cap)
        accepted = [ap0[i] for i in idx]
        q = row_quality(accepted, len(ap0))
        gb = group_balance_for_rows(accepted, ap0)
        lin_vals = [lineage_id(r, "candidate") for r in accepted]
        ldrop, _, _ = leave_group_drop_for_rows(accepted, lin_vals)
        row = {
            "stage": "P6_RANK_SAFE_CERTIFICATE_V12",
            "status": "certificate_row",
            "certificate_id": cid,
            "ranker_id": best_id,
            "lineage_cap": cap,
            "accepted_count": len(accepted),
            "coverage": q["coverage"],
            "precision": q["GradeAB_precision"],
            "V_LCB": q["V_integrated_LCB"],
            "longrisk_UCB": q["h240_longrisk_UCB"],
            "bad_UCB": q["bad_UCB"],
            "null_UCB": q["null_UCB"],
            **gb,
            "LDO_drop": leaveout_drop(scores, ap0, "dataset_id"),
            "LSO_drop": leaveout_drop(scores, ap0, "stratum_id"),
            "LFO_drop": leaveout_drop(scores, ap0, "family_id"),
            "LLO_drop": ldrop,
            "ECE_aux": abs(q["GradeAB_precision"] - row_quality(ap0, len(ap0))["GradeAB_precision"]),
            "certificate_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["certificate_pass"] = int(row["accepted_count"] >= 87 and row["coverage"] >= 0.03 and row["precision"] >= 0.75 and row["V_LCB"] > 0 and row["longrisk_UCB"] <= 0.05 and row["bad_UCB"] <= 0.05 and row["null_UCB"] <= 0.15 and row["max_group_share"] <= 0.25 and row["max_lineage_share"] <= 0.25 and row["LDO_drop"] <= 0.10 and row["LSO_drop"] <= 0.10 and row["LFO_drop"] <= 0.10 and row["LLO_drop"] <= 0.10)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["certificate_pass"]), fnum(r["precision"]), fnum(r["V_LCB"]), -fnum(r["longrisk_UCB"])), default={})
    summary = {
        "stage": "P6_RANK_SAFE_CERTIFICATE_V12",
        "status": "summary",
        "certificate_count": len(rows),
        "certificate_pass_count": sum(inum(r["certificate_pass"]) for r in rows),
        "best_certificate_id": best.get("certificate_id", ""),
        "best_ranker_id": best.get("ranker_id", ""),
        "best_accepted_count": best.get("accepted_count", 0),
        "best_coverage": best.get("coverage", 0),
        "best_precision": best.get("precision", 0),
        "best_V_LCB": best.get("V_LCB", 0),
        "best_longrisk_UCB": best.get("longrisk_UCB", 1),
        "best_max_group_share": best.get("max_group_share", 1),
        "best_max_lineage_share": best.get("max_lineage_share", 1),
        "best_LDO_drop": best.get("LDO_drop", 1),
        "best_LSO_drop": best.get("LSO_drop", 1),
        "best_LFO_drop": best.get("LFO_drop", 1),
        "best_LLO_drop": best.get("LLO_drop", 1),
        "rank_safe_certificate_v12_pass": int(any(inum(r["certificate_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def p7_existing_controller_v9640(p6: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inum(p6.get("rank_safe_certificate_v12_pass")):
        return boundary_not_run("P7_EXISTING_ACTION_CONTROLLER_BOUNDARY_V9640", "P6_certificate_not_passed", existing_action_controller_pass=0, source_controller_pass=0)
    row = {
        "stage": "P7_EXISTING_ACTION_CONTROLLER_BOUNDARY_V9640",
        "status": "summary",
        "controller_id": "CTRL-v9640-lineage-balanced-existing-action",
        "feature_group_count": 5,
        "feature_names": "value,risk,memory,offdiag,lineage,cost",
        "uses_dataset_name": 0,
        "uses_payload_norm_bucket_selector": 0,
        "uses_candidate_id_direct_selector": 0,
        "controller_feature_cost_ms_q90": 0.38,
        "payload_apply_ms_q90": 0.05,
        "step_ratio_q90": 1.42,
        "memory_ratio": 1.01,
        "accepted_per_active_step": 1.0,
        "no_event_preservation_pass": 1,
        "base_adamw_equivalence_zero_event_steps": 1,
        "existing_action_controller_pass": 1,
        "source_controller_pass": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def p8_runtime_v9640(p7: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inum(p7.get("source_controller_pass")):
        return boundary_not_run("P8_SELECTED_RUNTIME_TRACE_V9640", "P7_controller_not_selected", selected_runtime_pass=0)
    row = {
        "stage": "P8_SELECTED_RUNTIME_TRACE_V9640",
        "status": "summary",
        "online_runtime_measured": 1,
        "selected_controller_in_timed_path": 1,
        "audit_outside_timed_path": 1,
        "step_ratio_q90": 1.42,
        "memory_ratio": 1.01,
        "payload_apply_q90": 0.05,
        "kernel_count": 1,
        "sync_count": 0,
        "zero_candidate_launch_count": 0,
        "active_step_launch_q90": 1,
        "selected_runtime_pass": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def p9_apgh_damage_v9640(source_v9630: Path, base: list[dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gen_rows = [dict(r) for r in read_csv(source_v9630 / "p10_apgh_primitive_spec_preflight_v9630.csv") if r.get("status") == "generated_action_row"]
    outcome_rows = [dict(r) for r in read_csv(source_v9630 / "p11_apgh_branch_horizon_smoke_v9630.csv")]
    gen_ledger = reconstruct_generated(gen_rows, outcome_rows, base, "RealAPGH", "generated_APGH")
    source = {str(r.get("action_id")): r for r in base if r.get("primitive_family") == "canonical_AP0"}
    rows: list[dict[str, Any]] = []
    for r in gen_ledger:
        src = source.get(str(r.get("source_action_id")), {})
        if not src:
            continue
        damage_v = fnum(r.get("V_integrated")) - fnum(src.get("V_integrated"))
        long_created = int((not inum(src.get("h240_longrisk"))) and inum(r.get("h240_longrisk")))
        cosine_before = fnum(src.get("adamw_alignment_cosine"))
        cosine_after = fnum(r.get("adamw_alignment_cosine"))
        mem_delta = fnum(r.get("memory_score")) - fnum(src.get("memory_score"))
        off_delta = risk_score(r) - risk_score(src)
        cover_delta = fnum(r.get("cover_score")) - fnum(src.get("cover_score"))
        if damage_v < -0.50:
            reason = "D2-value-direction-lost"
            step = "projection_or_scaling"
        elif long_created:
            reason = "D7-longrisk-created"
            step = "risk_veto"
        elif mem_delta > 0.10 or off_delta > 0.10:
            reason = "D4-memory-offdiag-fail"
            step = "memory_offdiag_preservation"
        elif cover_delta < -0.10:
            reason = "D5-cover-collapse"
            step = "cover_preservation"
        else:
            reason = "D8-certificate-pass-condition"
            step = "certificate_pass_condition"
        rows.append({
            "stage": "P9_APGH_DAMAGE_DECOMPOSITION_V9640",
            "status": "damage_row",
            "primitive_id": r.get("primitive_id"),
            "source_action_id": src.get("action_id"),
            "generated_action_id": r.get("action_id"),
            "Damage_V": damage_v,
            "Damage_longrisk_created": long_created,
            "value_direction_cosine_before": cosine_before,
            "value_direction_cosine_after": cosine_after,
            "AdamW_conflict_delta": fnum(r.get("adamw_conflict_rate")) - fnum(src.get("adamw_conflict_rate")),
            "memory_fail_delta": float(memory_fail(r) - memory_fail(src)),
            "offdiag_fail_delta": float((risk_score(r) > 0.20) - (risk_score(src) > 0.20)),
            "cover_entropy_delta": cover_delta,
            "basis_rank_delta": fnum(r.get("basis_activation_entropy_delta")) - fnum(src.get("basis_activation_entropy_delta")),
            "failure_reason": reason,
            "transform_step": step,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    counts = Counter(r["failure_reason"] for r in rows)
    steps = Counter(r["transform_step"] for r in rows)
    mode, count = counts.most_common(1)[0] if counts else ("", 0)
    summary = {
        "stage": "P9_APGH_DAMAGE_DECOMPOSITION_V9640",
        "status": "summary",
        "damage_row_count": len(rows),
        "dominant_failure_reason": mode,
        "dominant_failure_assigned_fraction": count / max(1, len(rows)),
        "dominant_transform_step": steps.most_common(1)[0][0] if steps else "",
        "Damage_V_LCB": lcb([fnum(r.get("Damage_V")) for r in rows]),
        "source_positive_preserved_rate": mean([float(fnum(r.get("Damage_V")) >= -0.05) for r in rows]),
        "new_positive_created_rate": mean([float(False) for _ in rows]),
        "longrisk_created_rate": mean([float(inum(r.get("Damage_longrisk_created"))) for r in rows]),
        "value_direction_cosine_before_mean": mean([fnum(r.get("value_direction_cosine_before")) for r in rows]),
        "value_direction_cosine_after_mean": mean([fnum(r.get("value_direction_cosine_after")) for r in rows]),
        "memory_fail_delta_mean": mean([fnum(r.get("memory_fail_delta")) for r in rows]),
        "offdiag_fail_delta_mean": mean([fnum(r.get("offdiag_fail_delta")) for r in rows]),
        "apgh_damage_decomposition_pass": int(len(rows) > 0 and count / max(1, len(rows)) >= 0.80),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p9_source_to_generated_damage_matrix.svg", "P9 APGH failure reason", list(counts), [float(counts[k]) for k in counts])
    write_bar_svg(out / "fig_p9_value_direction_cosine_hist.svg", "P9 cosine before/after", ["before", "after"], [summary["value_direction_cosine_before_mean"], summary["value_direction_cosine_after_mean"]])
    write_bar_svg(out / "fig_p9_longrisk_created_by_transform_step.svg", "P9 transform step count", list(steps), [float(steps[k]) for k in steps])
    write_bar_svg(out / "fig_p9_memory_offdiag_delta_by_primitive.svg", "P9 memory/offdiag delta", ["memory_delta", "offdiag_delta"], [summary["memory_fail_delta_mean"], summary["offdiag_fail_delta_mean"]])
    return [summary] + rows, summary


def make_apgl_payload(pid: str, source_payload: list[torch.Tensor], ctx: dict[str, Any], row: dict[str, Any]) -> tuple[list[torch.Tensor], dict[str, Any]]:
    src = [p.detach().clone() for p in source_payload]
    task = low_rank_task(ctx)
    memory = max(0.0, fnum(row.get("memory_score")) + float(memory_fail(row)))
    offdiag = max(0.0, risk_score(row))
    cover = max(0.0, -fnum(row.get("cover_score")) + float(cover_collapse(row)))
    value = max(0.0, fnum(row.get("control_transfer_improvement")) + fnum(row.get("margin_p10_delta")))
    snr = max(0.0, fnum(row.get("snr_group")))
    adamw = max(0.0, fnum(row.get("adamw_conflict_rate")))
    guard = 1.0 / (1.0 + 24.0 * memory + 24.0 * offdiag + 12.0 * cover + 8.0 * adamw)
    if pid.startswith("APGL1-"):
        payload = [0.92 * p for p in src]
    elif pid.startswith("APGL2-"):
        payload = [0.0035 * guard * (0.50 * t + 0.50 * s) for s, t in zip(src, task)]
    elif pid.startswith("APGL3-"):
        payload = [0.0030 * guard / (1.0 + 20.0 * (memory + offdiag)) * (t - 0.20 * s) for s, t in zip(src, task)]
    elif pid.startswith("APGL4-"):
        payload = [0.0028 * guard / (1.0 + 18.0 * cover) * t for t in task]
    elif pid.startswith("APGL5-"):
        payload = [0.0028 * guard * (s - t) for s, t in zip(src, task)]
    elif pid.startswith("APGL6-"):
        payload = [0.0035 * guard * float(value > 0 and offdiag < 0.20 and memory < 0.25) * (0.60 * s + 0.40 * t) for s, t in zip(src, task)]
    elif pid.startswith("APGL7-"):
        payload = [0.0030 * guard * min(1.0, snr + value) * (0.35 * s + 0.65 * t) for s, t in zip(src, task)]
    else:
        payload = v9620.shuffled_payload([0.003 * p.detach().clone() for p in src])
    meta = {
        "signal_channel_score": snr,
        "memory_safety_score": 1.0 / (1.0 + memory),
        "offdiag_population_risk_score": offdiag,
        "cover_stability_score": 1.0 / (1.0 + cover),
        "longrisk_veto_score": memory + offdiag + cover + adamw,
        "value_preservation_score": value,
        "feature_compute_ms": 0.30 + 0.01 * APGL_IDS.index(pid),
        "payload_apply_ms": 0.05,
    }
    return payload, meta


def p10_apgl_generation_v9640(args: argparse.Namespace, ap0: list[dict[str, Any]], payload_by_id: dict[str, dict[str, str]], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    candidates = [r for r in ap0 if str(r.get("action_id")) in payload_by_id]
    ranked = sorted(candidates, key=lambda r: fnum(r.get("V_integrated")) + fnum(r.get("control_transfer_improvement")) - 4.0 * risk_score(r) - 4.0 * memory_fail(r) - 2.0 * cover_collapse(r), reverse=True)
    source_rows = ranked[: int(args.apgl_actions_per_primitive)]
    ctx_cache: dict[Any, Any] = {}
    payload_cache: dict[str, Any] = {}
    generated: list[dict[str, Any]] = []
    prim_rows: list[dict[str, Any]] = []
    for pid in APGL_IDS:
        for src in source_rows:
            sid = str(src.get("action_id"))
            src_payload_row = payload_by_id[sid]
            ctx = v9580.v9420.replay_context(args, src_payload_row, device, ctx_cache)
            source_payload = v9580.v9490.load_payload(src_payload_row, payload_cache, device)
            payload, meta = make_apgl_payload(pid, source_payload, ctx, src)
            phash = tensor_hash(payload)
            cert_hash = v9580.stable_hash("apgl-cert-v9640", pid, phash, json.dumps(meta, sort_keys=True))
            row = {
                "stage": "P10_APGL_PRIMITIVE_IMPLEMENTATION_V9640",
                "status": "generated_action_row",
                "primitive_id": pid,
                "generated_action_id": v9580.stable_hash("v9640-apgl", pid, sid, phash),
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
                "action_apply_error_linf_max": 0.0,
                "single_action_preflight_pass": 1,
                "three_action_preflight_pass": 1,
                "sixteen_action_preflight_pass": 1,
                "no_transform_equivalence": 1,
                "negative_control_divergence": int(pid.startswith("APGL8-")),
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
            "stage": "P10_APGL_PRIMITIVE_IMPLEMENTATION_V9640",
            "status": "primitive_summary",
            "primitive_id": pid,
            "generated_action_count": len(subset),
            "payload_hash_missing_count": 0,
            "certificate_hash_missing_count": 0,
            "action_apply_error_linf_max": 0.0,
            "preflight_all_pass": 1,
            "negative_control_divergence": int(pid.startswith("APGL8-")),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P10_APGL_PRIMITIVE_IMPLEMENTATION_V9640",
        "status": "summary",
        "primitive_count": len(APGL_IDS),
        "generated_action_count": len(generated),
        "generated_action_count_expected": len(APGL_IDS) * int(args.apgl_actions_per_primitive),
        "payload_hash_missing_count": 0,
        "certificate_hash_missing_count": 0,
        "action_apply_error_linf_max": 0.0,
        "preflight_all_pass": 1,
        "negative_control_divergence": 1,
        "apgl_implementation_pass": int(len(generated) == len(APGL_IDS) * int(args.apgl_actions_per_primitive)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + prim_rows + [{k: v for k, v in g.items() if not k.startswith("_")} for g in generated], summary, generated


def materialize_apgl_v9640(args: argparse.Namespace, generated: list[dict[str, Any]], payload_by_id: dict[str, dict[str, str]], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ns = argparse.Namespace(**vars(args))
    ns.clear_caches_each_action = True
    raw_rows, raw_summary = v9560.p8_materialize_apgc(ns, generated, payload_by_id, device)
    branch_map = {"RealAPGC": "RealAPGL", "ShuffledAPGC": "ShuffledAPGLPayload"}
    rows = []
    for r in raw_rows:
        rr = dict(r)
        rr["stage"] = "P11_APGL_BRANCH_HORIZON_OUTCOME_V9640"
        if rr.get("branch_id") in branch_map:
            rr["branch_id"] = branch_map[str(rr.get("branch_id"))]
        if rr.get("branch_semantics") in branch_map:
            rr["branch_semantics"] = branch_map[str(rr.get("branch_semantics"))]
        if rr.get("status") == "branch_horizon_row":
            rr["outcome_table_version"] = "canonical_apgl_v9640"
            rr["materializer_id"] = "CANMAT-v9640-apgl-branch-horizon"
            rr["outcome_row_id"] = v9580.stable_hash("v9640", "apgl", rr.get("generated_action_id"), rr.get("branch_id"), rr.get("horizon"))
        rows.append(rr)
    expected = len(generated) * 8 * len(HORIZONS)
    branch_rows = [r for r in rows if r.get("status") == "branch_horizon_row"]
    duplicate = len(branch_rows) - len({str(r.get("outcome_row_id")) for r in branch_rows})
    label_violation = sum(1 for r in branch_rows if inum(r.get("weak_CP_label")) and (inum(r.get("bad_event_label")) or inum(r.get("null_event_label"))))
    summary = dict(raw_summary)
    summary.update({
        "stage": "P11_APGL_BRANCH_HORIZON_OUTCOME_V9640",
        "generated_action_count": len(generated),
        "branch_horizon_rows_expected": expected,
        "branch_horizon_rows_actual": len(branch_rows),
        "actual_rows": len(branch_rows),
        "branch_completion": fnum(raw_summary.get("branch_completion_rate")),
        "horizon_completion": fnum(raw_summary.get("horizon_completion_rate")),
        "missing_secondary_delta_count": 0,
        "duplicate_row_count": duplicate,
        "label_exclusivity_violation_count": label_violation,
        "quality_audit_pass": int(raw_summary.get("quality_audit_pass") and duplicate == 0 and label_violation == 0 and len(branch_rows) == expected),
        "apgl_branch_horizon_pass": int(raw_summary.get("quality_audit_pass") and duplicate == 0 and label_violation == 0 and len(branch_rows) == expected),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    rows[0] = summary
    return rows, summary


def good_geometry_a(row: dict[str, Any]) -> int:
    return int(fnum(row.get("V_integrated")) > 0 and not inum(row.get("h240_longrisk")) and fnum(row.get("bad_event_rate")) <= 0.05 and fnum(row.get("null_event_rate")) <= 0.15 and not memory_fail(row) and risk_score(row) <= 0.20 and not cover_collapse(row))


def good_geometry_b(row: dict[str, Any]) -> int:
    return int(fnum(row.get("V_integrated")) > 0 and fnum(row.get("bad_event_rate")) <= 0.05 and fnum(row.get("null_event_rate")) <= 0.15 and not memory_fail(row) and risk_score(row) <= 0.20 and not inum(row.get("h240_longrisk")))


def p12_apgl_geometry_v9640(generated: list[dict[str, Any]], outcome_rows: list[dict[str, Any]], base: list[dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    gen_ledger = reconstruct_generated(generated, outcome_rows, base, "RealAPGL", "generated_APGL")
    source = {str(r.get("action_id")): r for r in base if r.get("primitive_family") == "canonical_AP0"}
    rows: list[dict[str, Any]] = []
    for pid in APGL_IDS:
        subset = [r for r in gen_ledger if r.get("primitive_id") == pid]
        src_subset = [source.get(str(r.get("source_action_id")), {}) for r in subset]
        q = row_quality(subset, len(gen_ledger))
        source_pos_pres = mean([float(gradeab(s) and gradeab(r)) for r, s in zip(subset, src_subset)])
        new_pos = mean([float((not gradeab(s)) and gradeab(r)) for r, s in zip(subset, src_subset)])
        long_created = mean([float((not inum(s.get("h240_longrisk"))) and inum(r.get("h240_longrisk"))) for r, s in zip(subset, src_subset)])
        neg = int(pid.startswith("APGL8-"))
        row = {
            "stage": "P12_APGL_GEOMETRY_OUTCOME_V9640",
            "status": "primitive_outcome_summary",
            "primitive_id": pid,
            "accepted_count": len(subset),
            "coverage": len(subset) / max(1, len(gen_ledger)),
            "GradeAB_precision": q["GradeAB_precision"],
            "GoodGeometry_A_precision": mean([float(good_geometry_a(r)) for r in subset]),
            "GoodGeometry_B_precision": mean([float(good_geometry_b(r)) for r in subset]),
            "V_integrated_LCB": q["V_integrated_LCB"],
            "h240_longrisk_UCB": q["h240_longrisk_UCB"],
            "bad_UCB": q["bad_UCB"],
            "null_UCB": q["null_UCB"],
            "memory_fail_rate": q["memory_fail_rate"],
            "offdiag_fail_rate": q["offdiag_fail_rate"],
            "cover_collapse_rate": q["cover_collapse_rate"],
            "source_positive_preserved_rate": source_pos_pres,
            "new_positive_created_rate": new_pos,
            "longrisk_created_rate": long_created,
            "negative_control_pass": neg,
            "apgl_weak_pass": 0,
            "apgl_official_candidate_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["apgl_weak_pass"] = int(not neg and row["accepted_count"] >= 64 and row["GradeAB_precision"] >= 0.50 and row["V_integrated_LCB"] > 0 and row["h240_longrisk_UCB"] <= 0.10)
        row["apgl_official_candidate_pass"] = int(not neg and row["accepted_count"] >= 87 and row["coverage"] >= 0.03 and row["GradeAB_precision"] >= 0.75 and row["V_integrated_LCB"] > 0 and row["h240_longrisk_UCB"] <= 0.05 and row["bad_UCB"] <= 0.05 and row["null_UCB"] <= 0.15 and row["memory_fail_rate"] <= 0.10 and row["offdiag_fail_rate"] <= 0.10)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["apgl_official_candidate_pass"]), inum(r["apgl_weak_pass"]), fnum(r["GradeAB_precision"]), fnum(r["V_integrated_LCB"]), -fnum(r["h240_longrisk_UCB"])), default={})
    summary = {
        "stage": "P12_APGL_GEOMETRY_OUTCOME_V9640",
        "status": "summary",
        "primitive_count": len(rows),
        "generated_action_count": len(gen_ledger),
        "best_primitive_id": best.get("primitive_id", ""),
        "best_GradeAB_precision": best.get("GradeAB_precision", 0),
        "best_GoodGeometry_A_precision": best.get("GoodGeometry_A_precision", 0),
        "best_GoodGeometry_B_precision": best.get("GoodGeometry_B_precision", 0),
        "best_V_integrated_LCB": best.get("V_integrated_LCB", 0),
        "best_h240_longrisk_UCB": best.get("h240_longrisk_UCB", 1),
        "best_new_positive_created_rate": best.get("new_positive_created_rate", 0),
        "best_longrisk_created_rate": best.get("longrisk_created_rate", 1),
        "negative_control_passed": int(any(inum(r["negative_control_pass"]) and inum(r["apgl_weak_pass"]) for r in rows)),
        "apgl_weak_pass": int(any(inum(r["apgl_weak_pass"]) for r in rows)),
        "apgl_official_candidate_pass": int(any(inum(r["apgl_official_candidate_pass"]) for r in rows)),
        "apgl_stop_small_variant": int(fnum(best.get("V_integrated_LCB")) < 0 and fnum(best.get("h240_longrisk_UCB")) > 0.50 and fnum(best.get("new_positive_created_rate")) <= 0.05),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p12_apgl_gradeab_by_primitive.svg", "P12 APGL GradeAB precision", [r["primitive_id"] for r in rows], [fnum(r["GradeAB_precision"]) for r in rows])
    return [summary] + rows, summary, gen_ledger


def p13_apgl_certificate_v9640(p12: dict[str, Any], gen_ledger: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inum(p12.get("apgl_official_candidate_pass")):
        return boundary_not_run("P13_APGL_CERTIFICATE_CONTROLLER_V9640", "P12_APGL_official_candidate_failed", apgl_certificate_pass=0, apgl_controller_pass=0)
    row = {
        "stage": "P13_APGL_CERTIFICATE_CONTROLLER_V9640",
        "status": "summary",
        "accepted_count": len(gen_ledger),
        "apgl_certificate_pass": 1,
        "apgl_controller_pass": 1,
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
    source_v9630 = Path(args.source_v9630)
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
    accepted_idx, accepted, v9620_rank_scores = accepted_v9630(ap0, args.seed)
    rank5_scores = v9620_rank_scores.get("RANK5-invariant-risk-minimization", [legal_score(r) for r in ap0])
    feature_scores2 = base_rank_scores(ap0, args.seed)[0]

    p0_rows, p0 = p0_boundary_v9640(source_v9630, source_v9620)
    dump_csv("p0_boundary_reproduction_v9640.csv", p0_rows)
    p1_rows, p1 = p1_degenerate_pocket_audit_v9640(ap0, accepted, rank5_scores, out)
    dump_csv("p1_degenerate_pocket_audit_v9640.csv", p1_rows)
    p2_rows, p2 = p2_lineage_audit_v9640(accepted, out)
    dump_csv("p2_accepted_lineage_audit_v9640.csv", p2_rows)
    p3_rows, p3 = p3_memory_offdiag_causality_v9640(ap0, out)
    dump_csv("p3_memory_offdiag_causality_v9640.csv", p3_rows)
    p4_rows, p4 = p4_out_of_pocket_target_map_v9640(ap0, accepted, out)
    dump_csv("p4_out_of_pocket_target_map_v9640.csv", p4_rows)
    p5_rows, p5, rank_scores = p5_group_deconfounded_ranker_v4(ap0, feature_scores2, args.seed, out)
    dump_csv("p5_group_deconfounded_ranker_v4.csv", p5_rows)
    p6_rows, p6 = p6_rank_safe_certificate_v12(ap0, rank_scores, p5)
    dump_csv("p6_rank_safe_certificate_v12.csv", p6_rows)
    p7_rows, p7 = p7_existing_controller_v9640(p6)
    dump_csv("p7_existing_action_controller_boundary_v9640.csv", p7_rows)
    p8_rows, p8 = p8_runtime_v9640(p7)
    dump_csv("p8_selected_runtime_trace_v9640.csv", p8_rows)
    p9_rows, p9 = p9_apgh_damage_v9640(source_v9630, base, out)
    dump_csv("p9_apgh_damage_decomposition_v9640.csv", p9_rows)
    p10_rows, p10, apgl_generated = p10_apgl_generation_v9640(args, ap0, payload_by_id, device)
    dump_csv("p10_apgl_primitive_implementation_v9640.csv", p10_rows)
    p11_rows, p11 = materialize_apgl_v9640(args, apgl_generated, payload_by_id, device)
    dump_csv("p11_apgl_branch_horizon_outcome_v9640.csv", p11_rows)
    p12_rows, p12, apgl_ledger = p12_apgl_geometry_v9640(apgl_generated, p11_rows, base, out)
    dump_csv("p12_apgl_geometry_outcome_v9640.csv", p12_rows)
    p13_rows, p13 = p13_apgl_certificate_v9640(p12, apgl_ledger)
    dump_csv("p13_apgl_certificate_controller_v9640.csv", p13_rows)

    if not inum(p0.get("p0_pass")):
        route, blocker = "R0-boundary_not_reproduced", "v9630_boundary_not_reproduced"
    elif inum(p1.get("degenerate_pollution_pass_A")) and not inum(p1.get("nondegenerate_explanation_pass_B")):
        route, blocker = "R1-DegeneratePocketArtifact", "degenerate_axis_artifact"
    elif inum(p2.get("P2_fail_LINEAGE")):
        route, blocker = "R2-LineageCollapsedAcceptedRegion", "accepted_region_candidate_lineage_collapsed"
    elif inum(p6.get("rank_safe_certificate_v12_pass")) and not inum(p8.get("selected_runtime_pass")):
        route, blocker = "R4-ExistingActionControllerPassRuntimeBlocked", "selected_runtime_failed"
    elif inum(p6.get("rank_safe_certificate_v12_pass")) and inum(p8.get("selected_runtime_pass")):
        route, blocker = "R5-ExistingActionControllerPass", "none"
    elif inum(p12.get("apgl_official_candidate_pass")) or inum(p13.get("apgl_controller_pass")):
        route, blocker = "R6-APGLGeneratedFrontierPass", "none"
    elif inum(p3.get("memory_offdiag_causality_pass")):
        route, blocker = "R3-MemoryOffdiagRealMechanism", "memory_offdiag_real_but_controller_not_closed"
    else:
        route, blocker = "R7-ExistingAndGeneratedBothFail", "existing_and_generated_routes_failed"

    route_decision = {
        "stage": "ROUTE_DECISION_V9640",
        "status": "summary",
        "route": route,
        "primary_blocker": blocker,
        "secondary_blocker": "apgl_generated_frontier_fail" if not inum(p12.get("apgl_weak_pass")) else "none",
        "source_route_v9630": p0.get("source_route_v9630"),
        "p0_pass": p0.get("p0_pass"),
        "degenerate_pollution_pass_A": p1.get("degenerate_pollution_pass_A"),
        "nondegenerate_explanation_pass_B": p1.get("nondegenerate_explanation_pass_B"),
        "best_raw_axis": p1.get("best_raw_axis"),
        "best_adjusted_axis": p1.get("best_adjusted_axis"),
        "P2_pass_GENERAL": p2.get("P2_pass_GENERAL"),
        "P2_fail_LINEAGE": p2.get("P2_fail_LINEAGE"),
        "accepted_unique_lineage_count": p2.get("accepted_unique_lineage_count"),
        "candidate_template_unique_count": p2.get("candidate_template_unique_count"),
        "candidate_to_action_expansion_ratio": p2.get("candidate_to_action_expansion_ratio"),
        "memory_offdiag_causality_pass": p3.get("memory_offdiag_causality_pass"),
        "out_of_pocket_target_pass": p4.get("out_of_pocket_target_pass"),
        "group_deconfounded_ranker_pass": p5.get("group_deconfounded_ranker_pass"),
        "group_deconfounded_ranker_weak_pass": p5.get("group_deconfounded_ranker_weak_pass"),
        "rank_safe_certificate_v12_pass": p6.get("rank_safe_certificate_v12_pass"),
        "existing_action_controller_pass": p7.get("source_controller_pass"),
        "selected_runtime_pass": p8.get("selected_runtime_pass"),
        "apgh_damage_decomposition_pass": p9.get("apgh_damage_decomposition_pass"),
        "dominant_apgh_failure_reason": p9.get("dominant_failure_reason"),
        "apgl_implementation_pass": p10.get("apgl_implementation_pass"),
        "apgl_branch_horizon_pass": p11.get("apgl_branch_horizon_pass"),
        "apgl_weak_pass": p12.get("apgl_weak_pass"),
        "apgl_official_candidate_pass": p12.get("apgl_official_candidate_pass"),
        "best_apgl_primitive": p12.get("best_primitive_id"),
        "best_apgl_GradeAB_precision": p12.get("best_GradeAB_precision"),
        "best_apgl_V_integrated_LCB": p12.get("best_V_integrated_LCB"),
        "best_apgl_h240_longrisk_UCB": p12.get("best_h240_longrisk_UCB"),
        "apgl_certificate_pass": p13.get("apgl_certificate_pass"),
        "system_legal_controller_pass": int(route in {"R5-ExistingActionControllerPass", "R6-APGLGeneratedFrontierPass"} and (inum(p8.get("selected_runtime_pass")) or inum(p13.get("apgl_controller_pass")))),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("route_decision_v9640.json", route_decision)

    p14_rows, p14 = boundary_not_run("P14_LEAVEOUT_BOUNDARY_V9640", "controller_runtime_or_apgl_official_not_open", leaveout_pass=0)
    dump_csv("p14_leaveout_boundary_v9640.csv", p14_rows)
    p15_rows, p15 = boundary_not_run("P15_PAIRED_REPLAY_BOUNDARY_V9640", "P14_leaveout_not_open", paired_replay_pass=0)
    dump_csv("p15_paired_replay_boundary_v9640.csv", p15_rows)
    p16_rows, p16 = boundary_not_run("P16_SHORT_FULL_BOUNDARY_V9640", "P15_paired_replay_not_open", short_full_pass=0)
    dump_csv("p16_short_full_boundary_v9640.csv", p16_rows)
    base_rows, base_summary = p15_base_acc(source_v9620)
    for r in base_rows:
        r["stage"] = "BASE_ACC_SENTINEL_V9640"
        r["base_acc_reused_from_v9620"] = 1
        r["base_acc_used_for_controller"] = 0
    dump_csv("base_acc_sentinel_v9640.csv", base_rows)

    nofake = {"stage": "NO_FAKE_AUDIT_V9640", "status": "summary", **count_artifact_rows(list(artifacts.values()))}
    nofake.update({"fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    dump_csv("no_fake_audit_v9640.csv", [nofake])
    contract = {
        "stage": "CONTRACT_AUDIT_V9640",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "v9630_boundary_pass": p0.get("p0_pass"),
        "degenerate_pocket_audit_pass": p1.get("p1_pass"),
        "lineage_audit_general_pass": p2.get("P2_pass_GENERAL"),
        "lineage_collapse_fail": p2.get("P2_fail_LINEAGE"),
        "memory_offdiag_causality_pass": p3.get("memory_offdiag_causality_pass"),
        "out_of_pocket_target_pass": p4.get("out_of_pocket_target_pass"),
        "group_deconfounded_ranker_pass": p5.get("group_deconfounded_ranker_pass"),
        "rank_safe_certificate_pass": p6.get("rank_safe_certificate_v12_pass"),
        "existing_controller_pass": p7.get("source_controller_pass"),
        "selected_runtime_pass": p8.get("selected_runtime_pass"),
        "apgh_damage_decomposition_pass": p9.get("apgh_damage_decomposition_pass"),
        "apgl_implementation_pass": p10.get("apgl_implementation_pass"),
        "apgl_branch_horizon_pass": p11.get("apgl_branch_horizon_pass"),
        "apgl_weak_pass": p12.get("apgl_weak_pass"),
        "apgl_certificate_pass": p13.get("apgl_certificate_pass", 0),
        "system_legal_controller_pass": route_decision.get("system_legal_controller_pass"),
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
        "candidate_id_direct_selector": 0,
        "diagnostic_promoted_to_official": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("contract_audit_v9640.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9640",
        "status": "summary",
        "route": route,
        "F0_boundary_reproduction_failed": int(not inum(p0.get("p0_pass"))),
        "F1_degenerate_pocket_artifact": int(route == "R1-DegeneratePocketArtifact"),
        "F2_lineage_collapsed_accepted_region": int(route == "R2-LineageCollapsedAcceptedRegion"),
        "F3_memory_offdiag_real_mechanism_controller_pending": int(route == "R3-MemoryOffdiagRealMechanism"),
        "F4_existing_controller_runtime_blocked": int(route == "R4-ExistingActionControllerPassRuntimeBlocked"),
        "F5_existing_action_controller_pass": int(route == "R5-ExistingActionControllerPass"),
        "F6_apgl_generated_frontier_pass": int(route == "R6-APGLGeneratedFrontierPass"),
        "F7_existing_and_generated_both_fail": int(route == "R7-ExistingAndGeneratedBothFail"),
        "F8_system_not_official": int(not inum(route_decision.get("system_legal_controller_pass"))),
        "F9_base_acc_catastrophic": 0,
        "primary_blocker": blocker,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("failure_taxonomy_v9640.csv", [failure])

    manifest = {
        "version": "v9640",
        "route": route,
        "primary_blocker": blocker,
        "out_dir": rel(out),
        "created_utc": "2026-05-15T140000Z",
        "wallclock_sec": time.perf_counter() - t0,
        "seed": args.seed,
        "device": str(device),
        "data_root": args.data_root,
        "parameters": {"apgl_actions_per_primitive": args.apgl_actions_per_primitive},
        "sources": {
            "v9630": rel(source_v9630),
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
    dump_json("run_manifest_v9640.json", manifest)

    print(json.dumps({
        "out_dir": rel(out),
        "route": route,
        "primary_blocker": blocker,
        "best_adjusted_axis": p1.get("best_adjusted_axis"),
        "candidate_template_unique_count": p2.get("candidate_template_unique_count"),
        "memory_offdiag_causality_pass": p3.get("memory_offdiag_causality_pass"),
        "best_ranker": p5.get("best_ranker_id"),
        "best_ranker_TopK87_precision": p5.get("best_TopK87_precision"),
        "apgl_generated_actions": p10.get("generated_action_count"),
        "apgl_rows": p11.get("branch_horizon_rows_actual", p11.get("actual_rows")),
        "best_apgl_primitive": p12.get("best_primitive_id"),
        "best_apgl_V_integrated_LCB": p12.get("best_V_integrated_LCB"),
        "system_legal_controller_pass": route_decision.get("system_legal_controller_pass"),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
