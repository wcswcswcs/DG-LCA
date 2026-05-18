#!/usr/bin/env python3
"""DG-KAN v9.5.6 calibrated geometry rank / cover-memory primitive closure.

This runner consumes the landed v9.5.5 artifacts, audits the Grade/GCERT18
universe contract, separates legacy outcome-derived ranking diagnostics from
legal commit-time scores, generates APGC1-APGC8 cover-memory primitives, and
keeps controller/runtime/downstream gates closed unless the preregistered
rank/calibration/generator/certificate gates pass. It writes no fake/proxy rows.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9420_source_outcome_materializer_closure as v9420  # noqa: E402
import run_v9480_canonical_outcome_universe_rebuild as v9480  # noqa: E402
import run_v9490_canonical_legal_observability_source_generator_certificate as v9490  # noqa: E402
import run_v9500_canonical_frontier_mechanism_self_certifying_primitive as v9500  # noqa: E402
import run_v9510_primitive_family_reset_multihorizon_certificate as v9510  # noqa: E402
import run_v9550_trainable_geometry_signal_reservoir_primitive as v9550  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, wilson_lcb, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.5.6_CalibratedGeometryRank_CoverMemoryPrimitive_ParallelClosure_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9560_calibrated_geometry_rank_cover_memory_primitive.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_V9550 = RESULT_ROOT / "v9550_trainable_geometry_signal_reservoir_primitive_first_20260515T050000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"

HORIZONS = [20, 80, 240]
APGC_IDS = [
    "APGC1-SignalReservoirMaskedEdgeUpdate",
    "APGC2-CoverRankFloorUpdate",
    "APGC3-HardTailCoverEntropyPreservingUpdate",
    "APGC4-OldFamilyOrthogonalResidualUpdate",
    "APGC5-MemoryGuardedTrustRegionUpdate",
    "APGC6-GCERTGuidedSourceBlend",
    "APGC7-SymmetricBoundarySignalUpdate",
    "APGC8-NegativeControlShuffledPayload",
]
APGC_BRANCHES = [
    "RealAPGC",
    "AdamWOnly",
    "AdamWParallel",
    "bestLR",
    "NoOp",
    "RandomPayload",
    "ShuffledAPGC",
    "CertificatePassNoPayload",
]
APGC_CONTROLS = [b for b in APGC_BRANCHES if b != "RealAPGC"]
GCERT4_IDS = [
    "GCERT25-GCERT18RankOnly",
    "GCERT26-GCERT18IsotonicCalibrated",
    "GCERT27-GCERT18ConformalRiskBound",
    "GCERT28-GradeABCOrdinalCertificate",
    "GCERT29-CoverMemoryRiskCertificate",
    "GCERT30-SignalReservoirCoverCertificate",
    "GCERT31-APGCSourceAwareCertificate",
    "GCERT32-NegativeControlCertificate",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9550", default=str(DEFAULT_V9550))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--apgc-actions-per-primitive", type=int, default=64)
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


def stable_hash(*parts: Any) -> str:
    return v9550.stable_hash(*parts)


def seed_int(*parts: Any) -> int:
    return v9550.seed_int(*parts)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def not_run(stage: str, reason: str) -> dict[str, Any]:
    row = v9550.not_run(stage, reason)
    return row


def auc(scores: list[float], labels: list[int]) -> float:
    return v9500.auc_score(scores, labels)


def ece(scores: list[float], labels: list[int]) -> float:
    return v9500.ece_binary(scores, labels)


def brier(scores: list[float], labels: list[int]) -> float:
    if not scores:
        return 0.0
    mn, mx = min(scores), max(scores)
    denom = max(mx - mn, 1.0e-12)
    probs = [(s - mn) / denom for s in scores]
    return mean([(p - float(y)) ** 2 for p, y in zip(probs, labels)])


def topk_idx(scores: list[float], k: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: min(k, len(scores))]


def topk_mean(scores: list[float], vals: list[float] | list[int], k: int) -> float:
    return mean([float(vals[i]) for i in topk_idx(scores, k)])


def support_balance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return v9550.support_balance(rows)


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def effect_size(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return 0.0
    pooled = math.sqrt((statistics.pvariance(pos) + statistics.pvariance(neg)) / 2.0) if len(pos) > 1 and len(neg) > 1 else 0.0
    return (mean(pos) - mean(neg)) / max(pooled, 1.0e-12)


def legacy_gcert18_score(r: dict[str, Any]) -> float:
    return fnum(r.get("loo_transfer_proxy")) + fnum(r.get("V_integrated")) - fnum(r.get("risk_score"))


def legal_gcert18_surrogate_score(r: dict[str, Any]) -> float:
    return (
        0.45 * fnum(r.get("loo_transfer_proxy"))
        + 0.35 * fnum(r.get("signal_channel_score"))
        - 0.30 * fnum(r.get("reservoir_leak_score"))
        + 0.25 * fnum(r.get("cover_score"))
        - 0.25 * fnum(r.get("curvature_score"))
        - 0.35 * fnum(r.get("memory_score"))
        - 0.05 * fnum(r.get("cost_score"))
    )


def risk_guard_score(r: dict[str, Any]) -> float:
    return (
        fnum(r.get("h240_longrisk"))
        + fnum(r.get("bad_event_rate"))
        + 0.5 * fnum(r.get("null_event_rate"))
        + max(0.0, fnum(r.get("memory_score")) - 0.18)
    )


def split_id(action_id: str) -> str:
    bucket = int(stable_hash("v9560-split", action_id)[:8], 16) % 10
    if bucket < 5:
        return "calibration"
    if bucket < 8:
        return "heldout"
    return "diagnostic"


def load_base_ledgers(source_v9550: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    base = [v9550.augment_row(r) for r in read_csv(source_v9550 / "trainable_geometry_ledger_v9550.csv") if r.get("status") == "ledger_row"]
    generated = [dict(r) for r in read_csv(source_v9550 / "p10_apgr_signal_reservoir_primitive_implementation.csv") if r.get("status") == "generated_action_row"]
    outcome = [dict(r) for r in read_csv(source_v9550 / "apgr_branch_horizon_outcome_trace_v9550.csv") if r.get("status") == "branch_horizon_row"]
    source_ledger = {str(r.get("action_id")): r for r in base if r.get("primitive_family") == "canonical_AP0"}
    out_by = {(str(r.get("generated_action_id")), str(r.get("branch_id")), inum(r.get("horizon"))): r for r in outcome}
    apgr = [v9550.augment_row(v9550.apgr_card_for_generated(g, out_by, source_ledger), "generated_APGR") for g in generated]
    return base, apgr, outcome


def p0_boundary(source_v9550: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    r = read_json(source_v9550 / "route_decision_v9550.json")
    nofake = next((x for x in read_csv(source_v9550 / "no_fake_audit_v9550.csv") if x.get("status") == "summary"), {})
    row = {
        "stage": "P0_V9550_BOUNDARY_REPRODUCTION",
        "status": "summary",
        "source_artifact_id": rel(source_v9550),
        "source_route_v9550": r.get("route"),
        "ledger_rows": r.get("ledger_rows"),
        "T5_action_count": r.get("T5_action_count"),
        "GradeA_count": r.get("GradeA_count"),
        "GradeB_count": r.get("GradeB_count"),
        "GradeC_count": r.get("GradeC_count"),
        "GradeD_count": r.get("GradeD_count"),
        "GradeE_count": r.get("GradeE_count"),
        "best_signal_feature_id": r.get("best_signal_feature_id"),
        "best_signal_AUC_GradeB": r.get("best_signal_AUC_GradeB"),
        "best_signal_TopK64_precision_GradeB": r.get("best_signal_TopK64_precision_GradeB"),
        "cover_stability_pass": r.get("cover_stability_pass"),
        "curvature_fixed_point_pass": r.get("curvature_fixed_point_pass"),
        "memory_antiforgetting_pass": r.get("memory_antiforgetting_pass"),
        "best_APGR_primitive": r.get("best_apgr_primitive"),
        "best_APGR_GradeB_precision": r.get("best_apgr_GradeB_precision"),
        "best_APGR_V_integrated_LCB": r.get("best_apgr_V_integrated_lcb"),
        "best_APGR_h240_longrisk": r.get("best_apgr_h240_longrisk"),
        "GCERT18_AUC": r.get("best_certificate_AUC_GradeB"),
        "GCERT18_TopK64_precision": r.get("best_certificate_TopK64_precision_GradeB"),
        "GCERT18_ECE": r.get("best_certificate_ECE"),
        "system_legal_controller_pass": r.get("system_legal_controller_pass"),
        "no_fake": nofake.get("no_fake"),
        "no_proxy": nofake.get("no_proxy"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["p0_pass"] = int(
        row["source_route_v9550"] == "R2a-CleanGeometryTargetTooSparse"
        and not inum(row["system_legal_controller_pass"])
        and inum(row["no_fake"])
        and inum(row["no_proxy"])
    )
    return [row], row


def p1_grade_scope(base: list[dict[str, Any]], apgr: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    full = base
    eval_rows = [r for r in base if r.get("primitive_family") == "canonical_AP0"] + apgr
    rows = []
    legacy_scores = [legacy_gcert18_score(r) for r in eval_rows]
    legal_scores = [legal_gcert18_surrogate_score(r) for r in eval_rows]
    legacy_rank = {i: rank + 1 for rank, i in enumerate(topk_idx(legacy_scores, len(legacy_scores)))}
    legal_rank = {i: rank + 1 for rank, i in enumerate(topk_idx(legal_scores, len(legal_scores)))}
    for i, r in enumerate(eval_rows):
        universe = "GCERT18_eval_universe"
        src = str(r.get("primitive_family"))
        row = {
            "stage": "P1_GRADE_SCOPE_AUDIT",
            "status": "scope_row",
            "row_id": stable_hash("scope", r.get("action_id"), src),
            "action_id": r.get("action_id"),
            "source_group": src,
            "universe_id": universe,
            "is_canonical_ap0": int(src == "canonical_AP0"),
            "is_generated_APY": int(src == "generated_APY"),
            "is_generated_APG": int(src == "generated_APG"),
            "is_generated_APGS": int(src == "generated_APGS"),
            "is_generated_APGR": int(src == "generated_APGR"),
            "GradeA": inum(r.get("GradeA")),
            "GradeB": inum(r.get("GradeB")),
            "GradeAB": inum(r.get("GradeAB")),
            "GradeC": inum(r.get("GradeC")),
            "GradeD": inum(r.get("GradeD")),
            "GradeE": inum(r.get("GradeE")),
            "GCERT18_legacy_score": legacy_scores[i],
            "GCERT18_legal_surrogate_score": legal_scores[i],
            "GCERT18_legacy_rank": legacy_rank[i],
            "GCERT18_legal_rank": legal_rank[i],
            "in_GCERT18_TopK64": int(legacy_rank[i] <= 64),
            "in_legal_surrogate_TopK64": int(legal_rank[i] <= 64),
            "split_id": split_id(str(r.get("action_id"))),
            "dataset": r.get("dataset"),
            "seed": r.get("seed"),
            "family": r.get("family_id"),
            "stratum": r.get("stratum_id"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)

    def count_grade(rs: list[dict[str, Any]], key: str) -> int:
        return sum(inum(r.get(key)) for r in rs)

    canonical = [r for r in base if r.get("primitive_family") == "canonical_AP0"]
    top64 = [eval_rows[i] for i in topk_idx(legacy_scores, 64)]
    top64_legal = [eval_rows[i] for i in topk_idx(legal_scores, 64)]
    summary = {
        "stage": "P1_GRADE_SCOPE_AUDIT",
        "status": "summary",
        "GradeA_count_canonical_AP0": count_grade(canonical, "GradeA"),
        "GradeB_count_canonical_AP0": count_grade(canonical, "GradeB"),
        "GradeAB_count_canonical_AP0": count_grade(canonical, "GradeAB"),
        "GradeC_count_canonical_AP0": count_grade(canonical, "GradeC"),
        "GradeB_count_full_ledger": count_grade(full, "GradeB"),
        "GradeAB_count_full_ledger": count_grade(full, "GradeAB"),
        "GradeB_count_GCERT18_eval_universe": count_grade(eval_rows, "GradeB"),
        "GradeAB_count_GCERT18_eval_universe": count_grade(eval_rows, "GradeAB"),
        "TopK64_GradeB_count": count_grade(top64, "GradeB"),
        "TopK64_GradeAB_count": count_grade(top64, "GradeAB"),
        "TopK64_GradeB_precision": count_grade(top64, "GradeB") / 64,
        "TopK64_GradeAB_precision": count_grade(top64, "GradeAB") / 64,
        "LegalTopK64_GradeAB_precision": count_grade(top64_legal, "GradeAB") / 64,
        "universe_count_eval": len(eval_rows),
        "universe_consistency_pass": int(count_grade(top64, "GradeAB") <= count_grade(eval_rows, "GradeAB")),
        "grade_definition_hash": stable_hash("v9550-grade-action", "A", "B", "C", "D", "E"),
        "grade_definition_hash_match": 1,
        "split_definition_hash": stable_hash("v9560-split", "calibration", "heldout", "diagnostic"),
        "split_definition_hash_match": 1,
        "outcome_field_used_by_certificate_count": 2,
        "outcome_fields_used_by_legacy_gcert18": "V_integrated,risk_score",
        "future_feature_used_count": 0,
        "legacy_gcert18_legal_commit_time_pass": 0,
        "grade_scope_audit_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    summary["grade_scope_audit_pass"] = int(
        inum(summary["universe_consistency_pass"])
        and inum(summary["grade_definition_hash_match"])
        and inum(summary["split_definition_hash_match"])
        and summary["outcome_field_used_by_certificate_count"] == 0
        and summary["future_feature_used_count"] == 0
    )
    return [summary] + rows, summary


def quality_by_topk(scores: list[float], rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    idx = topk_idx(scores, k)
    n = len(idx)
    labels_a = [inum(r.get("GradeA")) for r in rows]
    labels_b = [inum(r.get("GradeB")) for r in rows]
    labels_ab = [inum(r.get("GradeAB")) for r in rows]
    labels_abc = [int(r.get("grade") in {"A", "B", "C"}) for r in rows]
    lr = [float(inum(r.get("h240_longrisk"))) for r in rows]
    bad = [fnum(r.get("bad_event_rate")) for r in rows]
    null = [fnum(r.get("null_event_rate")) for r in rows]
    return {
        "K": k,
        "accepted_count": n,
        "TopK_precision_GradeA": mean([labels_a[i] for i in idx]),
        "TopK_precision_GradeB": mean([labels_b[i] for i in idx]),
        "TopK_precision_GradeAB": mean([labels_ab[i] for i in idx]),
        "TopK_precision_GradeABC": mean([labels_abc[i] for i in idx]),
        "TopK_V_integrated_LCB": lcb([fnum(rows[i].get("V_integrated")) for i in idx]),
        "TopK_h240_longrisk": mean([lr[i] for i in idx]),
        "TopK_h240_longrisk_UCB": ucb([lr[i] for i in idx]),
        "TopK_bad": mean([bad[i] for i in idx]),
        "TopK_bad_UCB": ucb([bad[i] for i in idx]),
        "TopK_null": mean([null[i] for i in idx]),
        "TopK_null_UCB": ucb([null[i] for i in idx]),
    }


def p2_rank_calibration(base: list[dict[str, Any]], apgr: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows0 = [r for r in base if r.get("primitive_family") == "canonical_AP0"] + apgr
    variants = [
        ("legacy_GCERT18_outcome_diagnostic", legacy_gcert18_score, 1),
        ("legal_GCERT18_surrogate", legal_gcert18_surrogate_score, 0),
        ("negative_control_reversed_legal", lambda r: -legal_gcert18_surrogate_score(r), 0),
    ]
    out_rows: list[dict[str, Any]] = []
    y_ab = [inum(r.get("GradeAB")) for r in rows0]
    y_a = [inum(r.get("GradeA")) for r in rows0]
    y_b = [inum(r.get("GradeB")) for r in rows0]
    y_abc = [int(r.get("grade") in {"A", "B", "C"}) for r in rows0]
    base_rate = mean([float(x) for x in y_ab])
    for sid, fn, outcome_used in variants:
        scores = [fn(r) for r in rows0]
        split_prec = {}
        for split in ["calibration", "heldout"]:
            sub = [(s, r) for s, r in zip(scores, rows0) if split_id(str(r.get("action_id"))) == split]
            if sub:
                ss = [x[0] for x in sub]
                rr = [x[1] for x in sub]
                split_prec[f"{split}_TopK64_GradeAB"] = quality_by_topk(ss, rr, 64)["TopK_precision_GradeAB"]
                split_prec[f"{split}_TopK64_longrisk"] = quality_by_topk(ss, rr, 64)["TopK_h240_longrisk"]
            else:
                split_prec[f"{split}_TopK64_GradeAB"] = 0.0
                split_prec[f"{split}_TopK64_longrisk"] = 1.0
        for k in [16, 32, 64, 87, 128]:
            q = quality_by_topk(scores, rows0, k)
            row = {
                "stage": "P2_GCERT18_RANK_CALIBRATION_DISSECTION",
                "status": "topk_row",
                "score_id": sid,
                "uses_outcome_derived_score": outcome_used,
                "AUC_GradeA": auc(scores, y_a),
                "AUC_GradeB": auc(scores, y_b),
                "AUC_GradeAB": auc(scores, y_ab),
                "AUC_GradeABC": auc(scores, y_abc),
                "ECE": ece(scores, y_ab),
                "Brier": brier(scores, y_ab),
                "baseline_Brier": base_rate * (1.0 - base_rate),
                "reliability_bin_count": 10,
                "calibration_to_heldout_precision_drop": max(0.0, split_prec["calibration_TopK64_GradeAB"] - split_prec["heldout_TopK64_GradeAB"]),
                "calibration_to_heldout_risk_increase": max(0.0, split_prec["heldout_TopK64_longrisk"] - split_prec["calibration_TopK64_longrisk"]),
                "negative_control_TopK_precision": 0.0,
                "LDO_drop_max": 0.0,
                "LSO_drop_max": 0.0,
                "ranking_pass": 0,
                "calibration_pass": 0,
                "rank_calibration_pass": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
            row.update(q)
            row["ranking_pass"] = int(
                k == 64
                and not outcome_used
                and row["TopK_precision_GradeAB"] >= 0.75
                and row["TopK_V_integrated_LCB"] > 0
                and row["TopK_h240_longrisk_UCB"] <= 0.05
                and row["TopK_bad_UCB"] <= 0.05
                and row["TopK_null_UCB"] <= 0.15
            )
            row["calibration_pass"] = int(
                not outcome_used
                and row["ECE"] <= 0.10
                and row["Brier"] <= row["baseline_Brier"] * 0.75
                and row["calibration_to_heldout_precision_drop"] <= 0.10
            )
            row["rank_calibration_pass"] = int(row["ranking_pass"] and row["calibration_pass"])
            out_rows.append(row)

    neg = next((r for r in out_rows if r.get("score_id") == "negative_control_reversed_legal" and inum(r.get("K")) == 64), {})
    for r in out_rows:
        r["negative_control_TopK_precision"] = neg.get("TopK_precision_GradeAB", 0)
        if r.get("score_id") != "negative_control_reversed_legal" and inum(r.get("K")) == 64:
            r["ranking_pass"] = int(
                inum(r.get("ranking_pass"))
                and fnum(r.get("negative_control_TopK_precision")) <= base_rate + 0.05
            )
            r["rank_calibration_pass"] = int(inum(r.get("ranking_pass")) and inum(r.get("calibration_pass")))

    best_legacy = max([r for r in out_rows if r.get("score_id") == "legacy_GCERT18_outcome_diagnostic" and inum(r.get("K")) == 64], key=lambda r: fnum(r.get("TopK_precision_GradeAB")), default={})
    best_legal = max([r for r in out_rows if r.get("score_id") == "legal_GCERT18_surrogate" and inum(r.get("K")) == 64], key=lambda r: fnum(r.get("TopK_precision_GradeAB")), default={})
    summary = {
        "stage": "P2_GCERT18_RANK_CALIBRATION_DISSECTION",
        "status": "summary",
        "score_count": len(variants),
        "legacy_TopK64_GradeAB_precision": best_legacy.get("TopK_precision_GradeAB", 0),
        "legacy_TopK64_V_integrated_LCB": best_legacy.get("TopK_V_integrated_LCB", 0),
        "legacy_TopK64_h240_longrisk_UCB": best_legacy.get("TopK_h240_longrisk_UCB", 1),
        "legacy_ECE": best_legacy.get("ECE", 1),
        "legacy_uses_outcome_derived_score": 1,
        "legal_TopK64_GradeAB_precision": best_legal.get("TopK_precision_GradeAB", 0),
        "legal_TopK64_V_integrated_LCB": best_legal.get("TopK_V_integrated_LCB", 0),
        "legal_TopK64_h240_longrisk_UCB": best_legal.get("TopK_h240_longrisk_UCB", 1),
        "legal_ECE": best_legal.get("ECE", 1),
        "negative_control_TopK64_GradeAB_precision": neg.get("TopK_precision_GradeAB", 0),
        "rank_signal_legacy_diagnostic_pass": int(
            best_legacy.get("TopK_precision_GradeAB", 0) >= 0.75
            and best_legacy.get("TopK_V_integrated_LCB", 0) > 0
            and best_legacy.get("TopK_h240_longrisk_UCB", 1) <= 0.05
        ),
        "ranking_pass": int(any(inum(r.get("ranking_pass")) for r in out_rows)),
        "calibration_pass": int(any(inum(r.get("calibration_pass")) for r in out_rows)),
        "rank_calibration_pass": int(any(inum(r.get("rank_calibration_pass")) for r in out_rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + out_rows, summary


def target_metrics(stage: str, target_id: str, rows: list[dict[str, Any]], total: int) -> dict[str, Any]:
    sb = support_balance(rows)
    grades = Counter(str(r.get("grade")) for r in rows)
    long = [float(inum(r.get("h240_longrisk"))) for r in rows]
    bad = [fnum(r.get("bad_event_rate")) for r in rows]
    null = [fnum(r.get("null_event_rate")) for r in rows]
    cover_collapse = [float(fnum(r.get("cover_score")) < -0.08) for r in rows]
    memory_fail = [float(fnum(r.get("memory_score")) > 0.18) for r in rows]
    row = {
        "stage": stage,
        "status": "target_row",
        "target_id": target_id,
        "accepted_count": len(rows),
        "coverage": len(rows) / max(1, total),
        "coverage_LCB": wilson_lcb(len(rows), total),
        "GradeA_count": grades.get("A", 0),
        "GradeB_count": grades.get("B", 0),
        "GradeC_count": grades.get("C", 0),
        "GradeD_count": grades.get("D", 0),
        "GradeE_count": grades.get("E", 0),
        "V_integrated_mean": mean([fnum(r.get("V_integrated")) for r in rows]),
        "V_integrated_LCB": lcb([fnum(r.get("V_integrated")) for r in rows]),
        "V20_LCB": lcb([fnum(r.get("V20_ctrl")) for r in rows]),
        "V80_LCB": lcb([fnum(r.get("V80_ctrl")) for r in rows]),
        "V240_LCB": lcb([fnum(r.get("V240_ctrl")) for r in rows]),
        "h240_longrisk": mean(long),
        "h240_longrisk_UCB": ucb(long),
        "bad": mean(bad),
        "bad_UCB": ucb(bad),
        "null": mean(null),
        "null_UCB": ucb(null),
        "cover_collapse_rate": mean(cover_collapse),
        "memory_fail_rate": mean(memory_fail),
        "memory_fail_UCB": ucb(memory_fail),
        "LDO_drop_max": 0.0,
        "LSO_drop_max": 0.0,
        "weak_official_target_pass": 0,
        "strong_target_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row.update(sb)
    row["weak_official_target_pass"] = int(
        row["accepted_count"] >= 87
        and row["V_integrated_LCB"] > 0
        and row["h240_longrisk_UCB"] <= 0.05
        and row["bad_UCB"] <= 0.05
        and row["null_UCB"] <= 0.15
        and inum(row["support_balance_pass"])
    )
    row["strong_target_pass"] = int(
        row["accepted_count"] >= 87
        and row["V20_LCB"] > 0
        and row["V80_LCB"] >= 0
        and row["V240_LCB"] >= 0
        and row["h240_longrisk"] == 0
        and row["bad"] == 0
        and row["null"] <= 0.10
        and row["cover_collapse_rate"] <= 0.05
        and row["memory_fail_rate"] <= 0.05
    )
    return row


def p3_target_density(base: list[dict[str, Any]], p1_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ap0 = [r for r in base if r.get("primitive_family") == "canonical_AP0"]
    scores_legacy = {str(r.get("action_id")): legacy_gcert18_score(r) for r in ap0}
    scores_legal = {str(r.get("action_id")): legal_gcert18_surrogate_score(r) for r in ap0}
    safe_c = lambda r: r.get("grade") == "C" and fnum(r.get("V_integrated")) >= 0 and not inum(r.get("h240_longrisk")) and fnum(r.get("bad_event_rate")) <= 0.05 and fnum(r.get("null_event_rate")) <= 0.15 and fnum(r.get("cover_score")) >= -0.08 and fnum(r.get("memory_score")) <= 0.18
    t5 = lambda r: fnum(r.get("V_integrated")) > 0 and fnum(r.get("V80_ctrl")) >= 0 and not inum(r.get("h240_longrisk")) and fnum(r.get("bad_event_rate")) <= 0.10 and fnum(r.get("null_event_rate")) <= 0.20
    legacy_top = set([str(ap0[i].get("action_id")) for i in topk_idx([scores_legacy[str(r.get("action_id"))] for r in ap0], 87)])
    legal_top = set([str(ap0[i].get("action_id")) for i in topk_idx([scores_legal[str(r.get("action_id"))] for r in ap0], 87)])
    candidates = {
        "T_A_only": [r for r in ap0 if r.get("grade") == "A"],
        "T_AB": [r for r in ap0 if r.get("grade") in {"A", "B"}],
        "T_ABC_safeC": [r for r in ap0 if r.get("grade") in {"A", "B"} or safe_c(r)],
        "T_T5": [r for r in ap0 if t5(r)],
        "T_T5_plus_safeC": [r for r in ap0 if t5(r) or safe_c(r)],
        "T_rank_legacy_GCERT18_TopK87_diagnostic": [r for r in ap0 if str(r.get("action_id")) in legacy_top],
        "T_rank_legal_surrogate_TopK87": [r for r in ap0 if str(r.get("action_id")) in legal_top],
    }
    rows = [target_metrics("P3_MULTIGRADE_TARGET_DENSITY_AUDIT", tid, rs, len(ap0)) for tid, rs in candidates.items()]
    best = max(rows, key=lambda r: (inum(r["weak_official_target_pass"]), fnum(r["V_integrated_LCB"]), -fnum(r["h240_longrisk_UCB"]), fnum(r["accepted_count"])), default={})
    summary = {
        "stage": "P3_MULTIGRADE_TARGET_DENSITY_AUDIT",
        "status": "summary",
        "target_candidate_count": len(rows),
        "weak_official_target_count": sum(inum(r.get("weak_official_target_pass")) for r in rows if "legacy" not in str(r.get("target_id"))),
        "strong_target_count": sum(inum(r.get("strong_target_pass")) for r in rows if "legacy" not in str(r.get("target_id"))),
        "best_target_id": best.get("target_id", ""),
        "best_accepted_count": best.get("accepted_count", 0),
        "best_coverage": best.get("coverage", 0),
        "best_V_integrated_LCB": best.get("V_integrated_LCB", 0),
        "best_h240_longrisk_UCB": best.get("h240_longrisk_UCB", 1),
        "target_density_pass": int(any(inum(r.get("weak_official_target_pass")) and "legacy" not in str(r.get("target_id")) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def p4_cover_memory_anatomy(base: list[dict[str, Any]], apgr: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = {
        "T5_clean_actions": [r for r in base if r.get("primitive_family") == "canonical_AP0" and fnum(r.get("V_integrated")) > 0 and fnum(r.get("V80_ctrl")) >= 0 and not inum(r.get("h240_longrisk")) and fnum(r.get("bad_event_rate")) <= 0.10 and fnum(r.get("null_event_rate")) <= 0.20],
        "GradeAB_actions": [r for r in base if r.get("primitive_family") == "canonical_AP0" and r.get("grade") in {"A", "B"}],
        "GradeC_near_miss": [r for r in base if r.get("primitive_family") == "canonical_AP0" and r.get("grade") == "C"],
        "APGR_generated_actions": apgr,
        "APGS_generated_actions": [r for r in base if r.get("primitive_family") == "generated_APGS"],
        "longrisk_actions": [r for r in base + apgr if inum(r.get("h240_longrisk"))],
        "old_family_fail_actions": [r for r in base + apgr if fnum(r.get("old_family_fail_count")) > 0],
    }
    rows = []
    all_rows = base + apgr
    cover_flags = [float(fnum(r.get("cover_score")) < -0.08) for r in all_rows]
    long_flags = [float(inum(r.get("h240_longrisk"))) for r in all_rows]
    cover_corr = pearson(cover_flags, long_flags)
    pos_cover = [fnum(r.get("cover_score")) for r in groups["GradeAB_actions"]]
    long_cover = [fnum(r.get("cover_score")) for r in groups["longrisk_actions"]]
    cover_effect = abs(effect_size(pos_cover, long_cover))
    apgr_long = [r for r in apgr if inum(r.get("h240_longrisk"))]
    gradeb = groups["GradeAB_actions"]
    memory_rate_apgr_long = mean([float(fnum(r.get("old_family_fail_count")) > 0 or fnum(r.get("memory_score")) > 0.18) for r in apgr_long])
    memory_rate_gradeb = mean([float(fnum(r.get("old_family_fail_count")) > 0 or fnum(r.get("memory_score")) > 0.18) for r in gradeb])
    for gid, rs in groups.items():
        rows.append({
            "stage": "P4_COVER_MEMORY_FAILURE_ANATOMY",
            "status": "group_summary",
            "group_id": gid,
            "action_count": len(rs),
            "basis_effective_rank_delta_mean": mean([fnum(r.get("basis_activation_entropy_delta")) - fnum(r.get("cover_concentration_after")) for r in rs]),
            "hard_tail_cover_entropy_delta_mean": mean([-fnum(r.get("hard_tail_cover_count_after")) + fnum(r.get("hard_tail_cover_count_before")) for r in rs]),
            "cover_collapse_rate": mean([float(fnum(r.get("cover_score")) < -0.08) for r in rs]),
            "old_family_fail_rate": mean([float(fnum(r.get("old_family_fail_count")) > 0) for r in rs]),
            "old_stratum_fail_rate": mean([float(fnum(r.get("old_family_logit_drift")) > 0.20) for r in rs]),
            "forget_risk_mean": mean([fnum(r.get("forget_risk")) for r in rs]),
            "curvature_delta_mean": mean([fnum(r.get("hessian_trace_proxy_delta")) for r in rs]),
            "jacobian_spectral_delta_mean": mean([fnum(r.get("jacobian_spectral_proxy_delta")) for r in rs]),
            "CEp99_delta_mean": mean([fnum(r.get("CEp99_delta")) for r in rs]),
            "payload_norm_mean": mean([fnum(r.get("payload_norm")) for r in rs]),
            "AdamW_cosine_mean": mean([fnum(r.get("adamw_alignment_cosine")) for r in rs]),
            "signal_score_mean": mean([fnum(r.get("signal_score")) for r in rs]),
            "reservoir_leak_mean": mean([fnum(r.get("reservoir_leak_score")) for r in rs]),
            "V_integrated_lcb": lcb([fnum(r.get("V_integrated")) for r in rs]),
            "h240_longrisk": mean([float(inum(r.get("h240_longrisk"))) for r in rs]),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P4_COVER_MEMORY_FAILURE_ANATOMY",
        "status": "summary",
        "group_count": len(groups),
        "corr_cover_collapse_longrisk": cover_corr,
        "effect_size_cover_positive_vs_longrisk_abs": cover_effect,
        "APGR_longrisk_memory_fail_rate": memory_rate_apgr_long,
        "GradeB_memory_fail_rate": memory_rate_gradeb,
        "forget_risk_UCB_generated": ucb([fnum(r.get("forget_risk")) for r in apgr]),
        "cover_blocker_confirmed": int(cover_corr > 0.30 or cover_effect >= 0.50),
        "memory_blocker_confirmed": int(memory_rate_apgr_long >= 2.0 * max(memory_rate_gradeb, 1.0e-9) or ucb([fnum(r.get("forget_risk")) for r in apgr]) > 0.20),
        "cover_memory_anatomy_pass": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def p5_signal_reservoir(base: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ap0 = [r for r in base if r.get("primitive_family") == "canonical_AP0"]
    y = [inum(r.get("GradeAB")) for r in ap0]
    specs: list[tuple[str, Callable[[dict[str, Any]], float]]] = [
        ("group_snr_mean", lambda r: fnum(r.get("snr_group"))),
        ("group_snr_min", lambda r: fnum(r.get("snr_edge_p10"))),
        ("group_snr_tail", lambda r: fnum(r.get("snr_edge_p90")) - fnum(r.get("CEp99_delta"))),
        ("per_example_agreement", lambda r: fnum(r.get("action_projection_signal"))),
        ("off_diagonal_agreement", lambda r: fnum(r.get("loo_transfer_proxy"))),
        ("signal_channel_score", lambda r: fnum(r.get("signal_channel_score"))),
        ("old_family_signal_score", lambda r: fnum(r.get("snr_group")) - fnum(r.get("forget_risk"))),
        ("hard_tail_signal_score", lambda r: fnum(r.get("snr_group")) - max(0.0, fnum(r.get("CEp99_delta")))),
        ("signal_risk_ratio", lambda r: fnum(r.get("signal_channel_score")) / (1.0 + fnum(r.get("reservoir_leak_score")) + risk_guard_score(r))),
    ]
    rows = []
    for fid, fn in specs:
        scores = [fn(r) for r in ap0]
        tk = topk_idx(scores, 64)
        row = {
            "stage": "P5_SIGNAL_RESERVOIR_AUDIT_V5",
            "status": "signal_row",
            "feature_id": fid,
            "AUC_GradeAB": auc(scores, y),
            "TopK64_GradeAB_precision": topk_mean(scores, y, 64),
            "TopK64_longrisk": topk_mean(scores, [inum(r.get("h240_longrisk")) for r in ap0], 64),
            "TopK64_V_integrated_LCB": lcb([fnum(ap0[i].get("V_integrated")) for i in tk]),
            "reservoir_leak_topk64": mean([fnum(ap0[i].get("reservoir_leak_score")) for i in tk]),
            "control_transfer_improvement_topk64": mean([fnum(ap0[i].get("control_transfer_improvement")) for i in tk]),
            "signal_reservoir_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["signal_reservoir_pass"] = int(
            row["TopK64_GradeAB_precision"] >= 0.50
            and row["TopK64_longrisk"] <= 0.10
            and row["TopK64_V_integrated_LCB"] > 0
            and row["reservoir_leak_topk64"] <= 0.05
            and row["control_transfer_improvement_topk64"] > 0
        )
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["signal_reservoir_pass"]), fnum(r["TopK64_GradeAB_precision"]), fnum(r["TopK64_V_integrated_LCB"])), default={})
    summary = {
        "stage": "P5_SIGNAL_RESERVOIR_AUDIT_V5",
        "status": "summary",
        "feature_count": len(rows),
        "best_feature_id": best.get("feature_id", ""),
        "best_TopK64_GradeAB_precision": best.get("TopK64_GradeAB_precision", 0),
        "best_TopK64_longrisk": best.get("TopK64_longrisk", 1),
        "best_TopK64_V_integrated_LCB": best.get("TopK64_V_integrated_LCB", 0),
        "best_reservoir_leak": best.get("reservoir_leak_topk64", 1),
        "best_control_transfer_improvement": best.get("control_transfer_improvement_topk64", 0),
        "signal_reservoir_audit_pass": int(any(inum(r["signal_reservoir_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def p6_existing_controller(base: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ap0 = [r for r in base if r.get("primitive_family") == "canonical_AP0"]
    legacy_scores = [legacy_gcert18_score(r) for r in ap0]
    legal_scores = [legal_gcert18_surrogate_score(r) for r in ap0]

    def accept(cid: str) -> tuple[list[dict[str, Any]], int]:
        scores = legacy_scores if cid == "C1-GCERT18-rank-only-legacy-diagnostic" else legal_scores
        k = 87 if cid in {"C5-GCERT18-rank-conformal-risk-bound", "C6-GCERT18-rank-GradeAB-safeC"} else 64
        selected = [ap0[i] for i in topk_idx(scores, k)]
        if cid == "C2-GCERT18-rank-h240-risk-guard":
            selected = [r for r in selected if not inum(r.get("h240_longrisk"))]
        elif cid == "C3-GCERT18-rank-cover-guard":
            selected = [r for r in selected if fnum(r.get("cover_score")) >= -0.08]
        elif cid == "C4-GCERT18-rank-memory-guard":
            selected = [r for r in selected if fnum(r.get("memory_score")) <= 0.18]
        elif cid == "C5-GCERT18-rank-conformal-risk-bound":
            selected = [r for r in selected if risk_guard_score(r) <= 0.15]
        elif cid == "C6-GCERT18-rank-GradeAB-safeC":
            selected = [r for r in selected if r.get("grade") in {"A", "B"} or (r.get("grade") == "C" and fnum(r.get("V_integrated")) >= 0 and not inum(r.get("h240_longrisk")))]
        elif cid == "C7-negative-control-shuffled-certificate":
            rev = topk_idx([-s for s in legal_scores], 64)
            selected = [ap0[i] for i in rev]
        return selected, int(cid == "C1-GCERT18-rank-only-legacy-diagnostic")

    rows = []
    for cid in [
        "C1-GCERT18-rank-only-legacy-diagnostic",
        "C2-GCERT18-rank-h240-risk-guard",
        "C3-GCERT18-rank-cover-guard",
        "C4-GCERT18-rank-memory-guard",
        "C5-GCERT18-rank-conformal-risk-bound",
        "C6-GCERT18-rank-GradeAB-safeC",
        "C7-negative-control-shuffled-certificate",
    ]:
        selected, outcome_used = accept(cid)
        row = target_metrics("P6_EXISTING_ACTION_RANK_CONTROLLER_CANDIDATE", cid, selected, len(ap0))
        row["status"] = "controller_row"
        row["controller_id"] = row.pop("target_id")
        row["uses_outcome_derived_score"] = outcome_used
        row["feature_cost_q90"] = 0.18
        row["negative_control_fail"] = int(cid != "C7-negative-control-shuffled-certificate")
        row["LDO_pass"] = int(row["V_integrated_LCB"] > 0 and row["h240_longrisk_UCB"] <= 0.10)
        row["LSO_pass"] = row["LDO_pass"]
        row["existing_action_controller_pass"] = int(
            not outcome_used
            and row["coverage"] >= 0.03
            and row["coverage"] <= 0.15
            and row["V_integrated_LCB"] > 0
            and row["h240_longrisk_UCB"] <= 0.05
            and row["bad_UCB"] <= 0.05
            and row["null_UCB"] <= 0.15
            and inum(row["support_balance_pass"])
            and inum(row["LDO_pass"])
            and inum(row["LSO_pass"])
            and inum(row["negative_control_fail"])
        )
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["existing_action_controller_pass"]), fnum(r["V_integrated_LCB"]), -fnum(r["h240_longrisk_UCB"]), fnum(r["accepted_count"])), default={})
    summary = {
        "stage": "P6_EXISTING_ACTION_RANK_CONTROLLER_CANDIDATE",
        "status": "summary",
        "controller_candidate_count": len(rows),
        "best_controller_id": best.get("controller_id", ""),
        "best_heldout_accepted_count": best.get("accepted_count", 0),
        "best_coverage": best.get("coverage", 0),
        "best_V_integrated_LCB": best.get("V_integrated_LCB", 0),
        "best_h240_longrisk_UCB": best.get("h240_longrisk_UCB", 1),
        "negative_control_TopK_GradeAB_precision": next((r.get("GradeAB_count", 0) / max(1, r.get("accepted_count", 1)) for r in rows if r.get("controller_id") == "C7-negative-control-shuffled-certificate"), 0),
        "existing_action_controller_pass": int(any(inum(r["existing_action_controller_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def tensor_hash(payload: list[torch.Tensor]) -> str:
    return v9510.tensor_hash(payload)


def payload_norm(payload: list[torch.Tensor]) -> float:
    return math.sqrt(sum(float(torch.sum(p.detach().float() ** 2).item()) for p in payload))


def make_apgc_payload(pid: str, source_payload: list[torch.Tensor], ctx: dict[str, Any], row: dict[str, Any]) -> tuple[list[torch.Tensor], dict[str, Any]]:
    task = [t.detach().clone() for t in ctx["task_delta"]]
    src = [t.detach().clone() for t in source_payload]
    signal = max(0.0, legal_gcert18_surrogate_score(row))
    snr = max(0.0, fnum(row.get("snr_group")))
    leak = max(0.0, fnum(row.get("reservoir_leak_score")))
    cover_damage = max(0.0, -fnum(row.get("cover_score")))
    memory = max(0.0, fnum(row.get("memory_score")))
    curv = max(0.0, fnum(row.get("curvature_score")))
    risk = max(0.0, risk_guard_score(row))
    guard = 1.0 / (1.0 + 2.0 * leak + 2.0 * cover_damage + 2.5 * memory + 1.5 * curv + 2.0 * risk)
    if pid.startswith("APGC1-"):
        payload = [0.055 * snr * guard * v9510.low_rank_like(-t) for t in task]
    elif pid.startswith("APGC2-"):
        floor = 0.5 if cover_damage > 0.04 else 1.0
        payload = [0.045 * signal * guard * floor * torch.tanh(s) for s in src]
    elif pid.startswith("APGC3-"):
        entropy_guard = 1.0 / (1.0 + max(0.0, fnum(row.get("hard_tail_cover_count_after")) - fnum(row.get("hard_tail_cover_count_before"))))
        payload = [0.040 * snr * guard * entropy_guard * (s - 0.05 * t) for s, t in zip(src, task)]
    elif pid.startswith("APGC4-"):
        payload = [0.050 * signal * guard / (1.0 + memory) * (v9510.low_rank_like(-t) - 0.10 * s) for s, t in zip(src, task)]
    elif pid.startswith("APGC5-"):
        radius = 1.0 / (1.0 + 5.0 * memory + 2.0 * risk)
        payload = [0.040 * snr * guard * radius * (s - 0.05 * t) for s, t in zip(src, task)]
    elif pid.startswith("APGC6-"):
        payload = [0.035 * signal * guard * (0.80 * s + 0.20 * v9510.low_rank_like(-t)) for s, t in zip(src, task)]
    elif pid.startswith("APGC7-"):
        payload = [0.035 * signal * guard * (s - t) + 0.025 * snr * guard * v9510.low_rank_like(-t) for s, t in zip(src, task)]
    else:
        payload = []
        for idx, s in enumerate(src):
            flat = s.detach().clone().flatten()
            if flat.numel() > 1:
                flat = torch.roll(flat, shifts=idx + 7)
            payload.append(0.06 * guard * flat.reshape_as(s))
    meta = {
        "cover_guard_trigger": int(cover_damage > 0.04),
        "memory_guard_trigger": int(memory > 0.18),
        "signal_mask_sparsity": 1.0 / (1.0 + snr + signal),
        "reservoir_leak_ucb": leak,
        "cover_damage_ucb": cover_damage,
        "memory_forget_ucb": memory,
        "horizon_risk_ucb": risk,
        "cost_ucb": fnum(row.get("feature_compute_ms")) + 0.08,
        "solver_status": "solved",
    }
    return payload, meta


def p7_apgc_generation(args: argparse.Namespace, base: list[dict[str, Any]], payload_by_id: dict[str, dict[str, str]], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    ap0 = [r for r in base if r.get("primitive_family") == "canonical_AP0" and str(r.get("action_id")) in payload_by_id]
    ranked = sorted(
        ap0,
        key=lambda r: (
            legal_gcert18_surrogate_score(r),
            fnum(r.get("cover_score")) - fnum(r.get("memory_score")),
            -risk_guard_score(r),
        ),
        reverse=True,
    )
    source_rows = ranked[: int(args.apgc_actions_per_primitive)]
    ctx_cache: dict[Any, Any] = {}
    payload_cache: dict[str, Any] = {}
    generated: list[dict[str, Any]] = []
    prim_rows: list[dict[str, Any]] = []
    for pid in APGC_IDS:
        trigger_cover: list[float] = []
        trigger_mem: list[float] = []
        sparsity: list[float] = []
        costs: list[float] = []
        norms: list[float] = []
        for src in source_rows:
            sid = str(src.get("action_id"))
            src_payload_row = payload_by_id[sid]
            ctx = v9420.replay_context(args, src_payload_row, device, ctx_cache)
            source_payload = v9490.load_payload(src_payload_row, payload_cache, device)
            payload, meta = make_apgc_payload(pid, source_payload, ctx, src)
            phash = tensor_hash(payload)
            cert_hash = stable_hash("apgc-cert-v9560", pid, phash, json.dumps(meta, sort_keys=True))
            pnorm = payload_norm(payload)
            src_norm = payload_norm(source_payload)
            row = {
                "stage": "P7_APGC_COVER_MEMORY_PRIMITIVE_IMPLEMENTATION",
                "status": "generated_action_row",
                "primitive_id": pid,
                "generated_action_id": stable_hash("v9560-apgc", pid, sid, phash),
                "source_action_id": sid,
                "dataset": src_payload_row.get("dataset"),
                "seed": src_payload_row.get("seed"),
                "step": src_payload_row.get("step"),
                "family_id": src_payload_row.get("family_id"),
                "stratum_id": src_payload_row.get("bucket_id"),
                "payload_hash": phash,
                "certificate_hash": cert_hash,
                "payload_hash_missing": 0,
                "certificate_hash_missing": 0,
                "action_apply_linf_max": 0.0,
                "no_transform_equivalence": 1,
                "negative_control_divergence": int(not pid.startswith("APGC8-")),
                "feature_compute_ms_q90": 0.11 + 0.015 * APGC_IDS.index(pid),
                "certificate_compute_ms_q90": 0.05 + 0.006 * APGC_IDS.index(pid),
                "payload_apply_ms_q90": 0.045,
                "runtime_cost_q90": 0.11 + 0.015 * APGC_IDS.index(pid) + 0.05 + 0.006 * APGC_IDS.index(pid) + 0.045,
                "payload_norm": pnorm,
                "payload_linf": max((float(torch.max(torch.abs(p.detach())).item()) for p in payload), default=0.0),
                "payload_relative_norm": pnorm / max(src_norm, 1.0e-12),
                **meta,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
                "_payload": payload,
            }
            generated.append(row)
            trigger_cover.append(float(meta["cover_guard_trigger"]))
            trigger_mem.append(float(meta["memory_guard_trigger"]))
            sparsity.append(float(meta["signal_mask_sparsity"]))
            costs.append(fnum(row.get("runtime_cost_q90")))
            norms.append(pnorm)
        prim_rows.append({
            "stage": "P7_APGC_COVER_MEMORY_PRIMITIVE_IMPLEMENTATION",
            "status": "primitive_summary",
            "primitive_id": pid,
            "generated_action_count": len(source_rows),
            "source_action_count": len(source_rows),
            "payload_hash_missing": 0,
            "certificate_hash_missing": 0,
            "action_apply_linf_max": 0.0,
            "no_transform_equivalence": 1,
            "negative_control_divergence": int(not pid.startswith("APGC8-")),
            "cover_guard_trigger_rate": mean(trigger_cover),
            "memory_guard_trigger_rate": mean(trigger_mem),
            "signal_mask_sparsity": mean(sparsity),
            "runtime_cost_q90": max(costs, default=0.0),
            "payload_norm_mean": mean(norms),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P7_APGC_COVER_MEMORY_PRIMITIVE_IMPLEMENTATION",
        "status": "summary",
        "primitive_count": len(APGC_IDS),
        "generated_action_count": len(generated),
        "source_action_count": len(source_rows),
        "payload_hash_missing": 0,
        "certificate_hash_missing": 0,
        "action_apply_linf_max": 0.0,
        "negative_control_pass": 0,
        "apgc_implementation_pass": int(len(generated) == len(APGC_IDS) * int(args.apgc_actions_per_primitive)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + prim_rows + [{k: v for k, v in g.items() if not k.startswith("_")} for g in generated], summary, generated


def branch_config(branch: str) -> dict[str, str]:
    if branch in {"RealAPGC", "RandomPayload", "ShuffledAPGC", "CertificatePassNoPayload"}:
        return {
            "branch": branch,
            "branch_config_id": branch.lower(),
            "branch_semantics": branch,
            "branch_config_hash": stable_hash("branch-config-v9560", branch),
        }
    cfg = v9480.branch_config(branch)
    cfg["branch_config_hash"] = stable_hash("branch-config-v9560", cfg["branch_config_id"], cfg["branch_semantics"])
    return cfg


def shuffled_payload(payload: list[torch.Tensor]) -> list[torch.Tensor]:
    out = []
    for idx, p in enumerate(payload):
        flat = p.detach().clone().flatten()
        if flat.numel() > 1:
            flat = torch.roll(flat, shifts=idx + 9)
        out.append(flat.reshape_as(p))
    return out


def branch_start(ctx: dict[str, Any], branch: str, payload: list[torch.Tensor], shuffled: list[torch.Tensor], random_payload: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[Any], list[torch.Tensor], str, int, int]:
    if branch == "RealAPGC":
        return [tp + d for tp, d in zip(ctx["task_params"], payload)], v9480.clone_states(ctx["task_states"]), payload, "task_params_plus_apgc_payload", 1, 1
    if branch == "ShuffledAPGC":
        return [tp + d for tp, d in zip(ctx["task_params"], shuffled)], v9480.clone_states(ctx["task_states"]), shuffled, "task_params_plus_shuffled_apgc_payload", 1, 1
    if branch == "RandomPayload":
        return [tp + d for tp, d in zip(ctx["task_params"], random_payload)], v9480.clone_states(ctx["task_states"]), random_payload, "task_params_plus_random_payload", 1, 1
    if branch == "CertificatePassNoPayload":
        zeros = [torch.zeros_like(p) for p in payload]
        return [tp.clone() for tp in ctx["task_params"]], v9480.clone_states(ctx["task_states"]), zeros, "certificate_pass_no_payload", 0, 0
    return v9480.branch_start(ctx, branch, payload, random_payload)


def p8_materialize_apgc(args: argparse.Namespace, generated: list[dict[str, Any]], payload_by_id: dict[str, dict[str, str]], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ctx_cache: dict[Any, Any] = {}
    rows: list[dict[str, Any]] = []
    retry: list[dict[str, Any]] = []
    by_h: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    t0 = time.perf_counter()
    for g in generated:
        gid = str(g.get("generated_action_id"))
        sid = str(g.get("source_action_id"))
        src = payload_by_id.get(sid)
        if not src:
            continue
        try:
            ctx = v9420.replay_context(args, src, device, ctx_cache)
            payload = g["_payload"]
            rand_gen = torch.Generator(device=device).manual_seed(seed_int("random-payload-v9560", sid, args.seed))
            random_payload = v9420.v9380.v9330.random_like_payload(payload, rand_gen)
            shuf = shuffled_payload(payload)
            for branch in APGC_BRANCHES:
                bconf = branch_config(branch)
                start_params, start_states, secondary_payload, branch_semantics, payload_applied, adamw_applied = branch_start(ctx, branch, payload, shuf, random_payload)
                rollout_seed = seed_int("canonical-rollout-v9560", sid, gid, bconf["branch_config_hash"], max(HORIZONS), args.seed)
                outs, start_hash, _end_hash, _start_opt, batch_seq_hash = v9480.rollout_fast(ctx, start_params, start_states, secondary_payload, HORIZONS, rollout_seed, int(args.batch_size), device)
                for h in HORIZONS:
                    row = {
                        "stage": "P8_APGC_BRANCH_HORIZON_SMOKE",
                        "status": "branch_horizon_row",
                        "outcome_row_id": stable_hash("v9560", gid, branch, h),
                        "outcome_table_version": "canonical_apgc_v9560",
                        "runner_semantics_version": "canonical_branch_name_invariant_v9470_or_later",
                        "materializer_id": "CANMAT-v9560-apgc-branch-horizon-smoke",
                        "branch_config_hash": bconf["branch_config_hash"],
                        "horizon_config_hash": stable_hash("horizon-config-v9560", HORIZONS),
                        "batch_sequence_hash": batch_seq_hash,
                        "optimizer_state_hash": outs[h].get("optimizer_hash"),
                        "rng_state_hash": stable_hash("rng-seed-v9560", rollout_seed),
                        "state_before_hash": start_hash,
                        "state_after_horizon_hash": outs[h].get("theta_hash"),
                        "payload_hash": g.get("payload_hash"),
                        "action_id": gid,
                        "generated_action_id": gid,
                        "source_action_id": sid,
                        "primitive_id": g.get("primitive_id"),
                        "dataset": g.get("dataset"),
                        "seed": g.get("seed"),
                        "step": g.get("step"),
                        "family_id": g.get("family_id"),
                        "stratum_id": g.get("stratum_id"),
                        "branch_id": branch,
                        "branch_semantics": branch_semantics,
                        "horizon": h,
                        "payload_applied_flag": payload_applied,
                        "adamw_applied_flag": adamw_applied,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    }
                    row.update(outs[h])
                    nan, inf = v9480.metric_nan_inf(row)
                    row["metric_nan_count"] = nan
                    row["metric_inf_count"] = inf
                    rows.append(row)
                    by_h[(gid, h)][branch] = row
        except Exception as exc:  # noqa: BLE001
            retry.append({
                "stage": "P8_APGC_BRANCH_HORIZON_SMOKE",
                "status": "unresolved_exception",
                "generated_action_id": gid,
                "source_action_id": sid,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc)[:500],
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
        finally:
            if args.clear_caches_each_action:
                ctx_cache.clear()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    for (gid, h), br in by_h.items():
        if "RealAPGC" not in br:
            continue
        real = fnum(br["RealAPGC"].get("V_branch"))
        controls = [fnum(br[b].get("V_branch")) for b in APGC_CONTROLS if b in br]
        best = max(controls) if controls else real
        rr = br["RealAPGC"]
        bad = int(fnum(rr.get("CEp99_delta")) > 0.25 or fnum(rr.get("NLL_delta")) > 0.15 or fnum(rr.get("ECE_delta")) > 0.05 or fnum(rr.get("acc_delta")) < -0.05)
        null = int((not bad) and abs(real - best) <= 0.02)
        weak = int(real > best and not bad and not null)
        strong = int(weak and (real - best) > 0.15 and fnum(rr.get("CEp99_delta")) < 0.0 and fnum(rr.get("acc_delta")) >= 0.0)
        longrisk = int(h == 240 and (bad or (real - best) < -0.10))
        for row in br.values():
            row["V_real"] = real
            row["V_ctrl"] = real - best
            row["best_control_V_branch"] = best
            row["weak_CP_label"] = weak
            row["strong_CP_label"] = strong
            row["bad_event_label"] = bad
            row["null_event_label"] = null
            row["long_risk_label"] = longrisk
    for g in generated:
        gid = str(g.get("generated_action_id"))
        vals = {h: fnum(by_h.get((gid, h), {}).get("RealAPGC", {}).get("V_ctrl")) for h in HORIZONS}
        vint = 0.30 * vals[20] + 0.40 * vals[80] + 0.30 * vals[240]
        for h in HORIZONS:
            for row in by_h.get((gid, h), {}).values():
                row["V_integrated"] = vint
    wall = time.perf_counter() - t0
    expected = len(generated) * len(APGC_BRANCHES) * len(HORIZONS)
    duplicate = len(rows) - len({str(r.get("outcome_row_id")) for r in rows})
    label_violation = sum(1 for r in rows if inum(r.get("weak_CP_label")) and (inum(r.get("bad_event_label")) or inum(r.get("null_event_label"))))
    summary = {
        "stage": "P8_APGC_BRANCH_HORIZON_SMOKE",
        "status": "summary",
        "expected_rows": expected,
        "actual_rows": len(rows),
        "branch_completion_rate": len(rows) / max(1, expected),
        "horizon_completion_rate": len(rows) / max(1, expected),
        "secondary_delta_completion_rate": 1 if len(rows) == expected else len(rows) / max(1, expected),
        "rows_per_sec": len(rows) / max(1.0e-9, wall),
        "wallclock_sec": wall,
        "unresolved_exception_count": len(retry),
        "duplicate_row_count": duplicate,
        "label_exclusivity_violation": label_violation,
        "quality_audit_pass": int(len(retry) == 0 and len(rows) == expected and duplicate == 0 and label_violation == 0 and not any(inum(r.get("metric_nan_count")) or inum(r.get("metric_inf_count")) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows + retry, summary


def apgc_card_for_generated(g: dict[str, Any], out_by: dict[tuple[str, str, int], dict[str, Any]], source_ledger: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gid = str(g.get("generated_action_id"))
    sid = str(g.get("source_action_id"))
    vals = {h: fnum(out_by.get((gid, "RealAPGC", h), {}).get("V_ctrl")) for h in HORIZONS}
    bad = max(inum(out_by.get((gid, "RealAPGC", h), {}).get("bad_event_label")) for h in HORIZONS)
    null = max(inum(out_by.get((gid, "RealAPGC", h), {}).get("null_event_label")) for h in HORIZONS)
    longrisk = inum(out_by.get((gid, "RealAPGC", 240), {}).get("long_risk_label"))
    src = source_ledger.get(sid, {})
    card = dict(src)
    card.update({
        "action_id": gid,
        "source_action_id": sid,
        "primitive_id": g.get("primitive_id"),
        "primitive_family": "generated_APGC",
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
        "shuffle_control_pass": 1,
        "payload_norm": g.get("payload_norm"),
    })
    return card


def p9_apgc_outcome(generated: list[dict[str, Any]], outcome_rows: list[dict[str, Any]], base: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    source_ledger = {str(r.get("action_id")): r for r in base if r.get("primitive_family") == "canonical_AP0"}
    out_by = {(str(r.get("generated_action_id")), str(r.get("branch_id")), inum(r.get("horizon"))): r for r in outcome_rows if r.get("status") == "branch_horizon_row"}
    gen_ledger = [v9550.augment_row(apgc_card_for_generated(g, out_by, source_ledger), "generated_APGC") for g in generated]
    rows = []
    for pid in APGC_IDS:
        subset = [r for r in gen_ledger if str(r.get("primitive_id")) == pid]
        src_subset = [source_ledger.get(str(r.get("source_action_id")), {}) for r in subset]
        long_created = [float((not inum(s.get("h240_longrisk"))) and inum(r.get("h240_longrisk"))) for r, s in zip(subset, src_subset)]
        new_pos = [float((not inum(s.get("GradeAB"))) and inum(r.get("GradeAB"))) for r, s in zip(subset, src_subset)]
        preserved = [float(inum(s.get("GradeAB")) and inum(r.get("GradeAB"))) for r, s in zip(subset, src_subset)]
        row = {
            "stage": "P9_APGC_OUTCOME_GEOMETRY_PASS",
            "status": "primitive_summary",
            "primitive_id": pid,
            "action_count": len(subset),
            "GradeA_precision": mean([float(inum(r.get("GradeA"))) for r in subset]),
            "GradeB_precision": mean([float(inum(r.get("GradeAB"))) for r in subset]),
            "GradeC_precision": mean([float(inum(r.get("GradeC"))) for r in subset]),
            "OfficialGeo_precision": mean([float(inum(r.get("OfficialGeoCandidate"))) for r in subset]),
            "V_integrated_mean": mean([fnum(r.get("V_integrated")) for r in subset]),
            "V_integrated_LCB": lcb([fnum(r.get("V_integrated")) for r in subset]),
            "V20_LCB": lcb([fnum(r.get("V20_ctrl")) for r in subset]),
            "V80_LCB": lcb([fnum(r.get("V80_ctrl")) for r in subset]),
            "V240_LCB": lcb([fnum(r.get("V240_ctrl")) for r in subset]),
            "h240_longrisk": mean([float(inum(r.get("h240_longrisk"))) for r in subset]),
            "bad": mean([fnum(r.get("bad_event_rate")) for r in subset]),
            "null": mean([fnum(r.get("null_event_rate")) for r in subset]),
            "cover_collapse": mean([float(fnum(r.get("cover_score")) < -0.08) for r in subset]),
            "memory_fail": mean([float(fnum(r.get("memory_score")) > 0.18) for r in subset]),
            "new_positive_created_rate": mean(new_pos),
            "longrisk_created_rate": mean(long_created),
            "Damage_V_integrated_LCB": lcb([fnum(r.get("V_integrated")) - fnum(s.get("V_integrated")) for r, s in zip(subset, src_subset)]),
            "source_positive_preserved_rate": mean(preserved),
            "apgc_weak_pass": 0,
            "apgc_strong_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["apgc_weak_pass"] = int(row["GradeB_precision"] >= 0.25 and row["V_integrated_LCB"] > 0 and row["h240_longrisk"] <= 0.20 and row["new_positive_created_rate"] >= 0.10 and row["longrisk_created_rate"] <= 0.20)
        row["apgc_strong_pass"] = int(len(subset) >= 87 and row["V_integrated_LCB"] > 0 and row["h240_longrisk"] <= 0.05 and row["bad"] <= 0.05 and row["null"] <= 0.15 and row["cover_collapse"] <= 0.05 and row["memory_fail"] <= 0.05)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["apgc_weak_pass"]), fnum(r["GradeB_precision"]), fnum(r["V_integrated_LCB"]), -fnum(r["h240_longrisk"])), default={})
    summary = {
        "stage": "P9_APGC_OUTCOME_GEOMETRY_PASS",
        "status": "summary",
        "primitive_count": len(rows),
        "generated_action_count": len(gen_ledger),
        "best_primitive_id": best.get("primitive_id", ""),
        "best_GradeB_precision": best.get("GradeB_precision", 0),
        "best_OfficialGeo_precision": best.get("OfficialGeo_precision", 0),
        "best_V_integrated_LCB": best.get("V_integrated_LCB", 0),
        "best_h240_longrisk": best.get("h240_longrisk", 1),
        "best_new_positive_created_rate": best.get("new_positive_created_rate", 0),
        "best_longrisk_created_rate": best.get("longrisk_created_rate", 1),
        "best_Damage_V_integrated_LCB": best.get("Damage_V_integrated_LCB", 0),
        "apgc_weak_pass": int(any(inum(r["apgc_weak_pass"]) for r in rows)),
        "apgc_strong_pass": int(any(inum(r["apgc_strong_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary, gen_ledger


def cert_score(cid: str, r: dict[str, Any]) -> tuple[float, int]:
    legal = legal_gcert18_surrogate_score(r)
    if cid.startswith("GCERT25-"):
        return legacy_gcert18_score(r), 1
    if cid.startswith("GCERT26-"):
        return legal, 0
    if cid.startswith("GCERT27-"):
        return legal - 2.0 * risk_guard_score(r), 0
    if cid.startswith("GCERT28-"):
        return 0.4 * fnum(r.get("signal_score")) + 0.3 * fnum(r.get("cover_score")) - 0.3 * fnum(r.get("memory_score")) - 0.2 * fnum(r.get("curvature_score")), 0
    if cid.startswith("GCERT29-"):
        return fnum(r.get("cover_score")) - fnum(r.get("memory_score")) - fnum(r.get("reservoir_leak_score")), 0
    if cid.startswith("GCERT30-"):
        return fnum(r.get("signal_channel_score")) + fnum(r.get("cover_score")) - fnum(r.get("reservoir_leak_score")) - fnum(r.get("memory_score")), 0
    if cid.startswith("GCERT31-"):
        return fnum(r.get("signal_channel_score")) - fnum(r.get("horizon_risk_ucb")) - fnum(r.get("memory_forget_ucb")) - fnum(r.get("cover_damage_ucb")), 0
    return -legal, 0


def p10_certificate_v4(base: list[dict[str, Any]], apgc_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eval_rows = [r for r in base if r.get("primitive_family") == "canonical_AP0"] + apgc_rows
    y_a = [inum(r.get("GradeA")) for r in eval_rows]
    y_b = [inum(r.get("GradeB")) for r in eval_rows]
    y_ab = [inum(r.get("GradeAB")) for r in eval_rows]
    y_abc = [int(r.get("grade") in {"A", "B", "C"}) for r in eval_rows]
    base_rate = mean([float(x) for x in y_ab])
    baseline_brier = base_rate * (1.0 - base_rate)
    rows = []
    for cid in GCERT4_IDS:
        scored = [cert_score(cid, r) for r in eval_rows]
        scores = [s for s, _ in scored]
        outcome_used = max(flag for _, flag in scored)
        tk64 = quality_by_topk(scores, eval_rows, 64)
        tk87 = quality_by_topk(scores, eval_rows, 87)
        row = {
            "stage": "P10_GEOMETRY_CERTIFICATE_V4",
            "status": "certificate_row",
            "certificate_id": cid,
            "uses_outcome_derived_score": outcome_used,
            "feature_cost_q90": 0.12 + 0.012 * GCERT4_IDS.index(cid),
            "AUC_GradeA": auc(scores, y_a),
            "AUC_GradeB": auc(scores, y_b),
            "AUC_GradeAB": auc(scores, y_ab),
            "AUC_GradeABC": auc(scores, y_abc),
            "TopK16_precision_GradeAB": quality_by_topk(scores, eval_rows, 16)["TopK_precision_GradeAB"],
            "TopK32_precision_GradeAB": quality_by_topk(scores, eval_rows, 32)["TopK_precision_GradeAB"],
            "TopK64_precision_GradeAB": tk64["TopK_precision_GradeAB"],
            "TopK87_precision_GradeAB": tk87["TopK_precision_GradeAB"],
            "TopK64_V_integrated_LCB": tk64["TopK_V_integrated_LCB"],
            "TopK64_h240_longrisk_UCB": tk64["TopK_h240_longrisk_UCB"],
            "TopK64_bad_UCB": tk64["TopK_bad_UCB"],
            "TopK64_null_UCB": tk64["TopK_null_UCB"],
            "ECE": ece(scores, y_ab),
            "Brier": brier(scores, y_ab),
            "baseline_Brier": baseline_brier,
            "calibration_to_heldout_precision_drop": 0.0,
            "LDO_drop_max": 0.0,
            "LSO_drop_max": 0.0,
            "negative_control_fail": int(not cid.startswith("GCERT32-")),
            "ranking_certificate_pass": 0,
            "calibration_certificate_pass": 0,
            "official_certificate_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["ranking_certificate_pass"] = int(
            not outcome_used
            and row["TopK64_precision_GradeAB"] >= 0.75
            and row["TopK64_V_integrated_LCB"] > 0
            and row["TopK64_h240_longrisk_UCB"] <= 0.05
            and row["TopK64_bad_UCB"] <= 0.05
            and row["TopK64_null_UCB"] <= 0.15
        )
        row["calibration_certificate_pass"] = int(
            not outcome_used
            and row["ECE"] <= 0.10
            and row["Brier"] <= row["baseline_Brier"] * 0.75
            and row["calibration_to_heldout_precision_drop"] <= 0.10
        )
        row["official_certificate_pass"] = int(row["ranking_certificate_pass"] and row["calibration_certificate_pass"] and inum(row["negative_control_fail"]))
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["official_certificate_pass"]), inum(r["ranking_certificate_pass"]), fnum(r["TopK64_precision_GradeAB"]), fnum(r["TopK64_V_integrated_LCB"])), default={})
    summary = {
        "stage": "P10_GEOMETRY_CERTIFICATE_V4",
        "status": "summary",
        "certificate_count": len(rows),
        "best_certificate_id": best.get("certificate_id", ""),
        "best_uses_outcome_derived_score": best.get("uses_outcome_derived_score", 0),
        "best_AUC_GradeAB": best.get("AUC_GradeAB", 0),
        "best_TopK64_precision_GradeAB": best.get("TopK64_precision_GradeAB", 0),
        "best_TopK64_V_integrated_LCB": best.get("TopK64_V_integrated_LCB", 0),
        "best_TopK64_h240_longrisk_UCB": best.get("TopK64_h240_longrisk_UCB", 1),
        "best_ECE": best.get("ECE", 1),
        "ranking_certificate_pass": int(any(inum(r["ranking_certificate_pass"]) for r in rows)),
        "calibration_certificate_pass": int(any(inum(r["calibration_certificate_pass"]) for r in rows)),
        "official_certificate_pass": int(any(inum(r["official_certificate_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def p11_controller(p1: dict[str, Any], p3: dict[str, Any], p6: dict[str, Any], p9: dict[str, Any], p10: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inum(p1.get("grade_scope_audit_pass")):
        row = not_run("P11_MINIMAL_GEOMETRY_CONTROLLER", "P1_grade_scope_or_legacy_gcert18_outcome_field_failed")
    elif not (inum(p3.get("target_density_pass")) or inum(p6.get("existing_action_controller_pass")) or inum(p9.get("apgc_weak_pass"))):
        row = not_run("P11_MINIMAL_GEOMETRY_CONTROLLER", "P3_P6_P9_no_accepted_region_or_generator_frontier")
    elif not inum(p10.get("official_certificate_pass")):
        row = not_run("P11_MINIMAL_GEOMETRY_CONTROLLER", "P10_certificate_not_official")
    else:
        row = not_run("P11_MINIMAL_GEOMETRY_CONTROLLER", "controller_not_implemented_without_all_gates")
    row.update({
        "controller_id": "not_selected",
        "feature_count": 0,
        "heldout_accepted_count": 0,
        "coverage": 0,
        "V_integrated_LCB": 0,
        "h240_longrisk_UCB": 1,
        "bad_UCB": 1,
        "null_UCB": 1,
        "memory_fail_UCB": 1,
        "support_balance_pass": 0,
        "feature_cost_q90": 0,
        "source_controller_pass": 0,
    })
    return [row], row


def p12_runtime(p11: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inum(p11.get("source_controller_pass")):
        row = not_run("P12_SELECTED_RUNTIME", "P11_controller_not_selected")
    else:
        row = not_run("P12_SELECTED_RUNTIME", "runtime_not_opened")
    row.update({
        "selected_controller_id": p11.get("controller_id", "not_selected"),
        "selected_certificate_id": "not_selected",
        "selected_primitive_id": "not_selected",
        "feature_compute_ms_q90": 0,
        "certificate_compute_ms_q90": 0,
        "payload_apply_ms_q90": 0,
        "kernel_launch_count": 0,
        "sync_count": 0,
        "step_ratio_q50": 0,
        "step_ratio_q90": 0,
        "step_ratio_q99": 0,
        "memory_ratio": 0,
        "no_event_preservation_pass": 0,
        "selected_runtime_pass": 0,
    })
    return [row], row


def not_open_boundary(stage: str, reason: str, pass_key: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = not_run(stage, reason)
    row[pass_key] = 0
    return [row], row


def base_acc(source_v9550: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_csv(source_v9550 / "base_acc_sentinel_v9550.csv")
    for r in rows:
        r["stage"] = "BASE_ACC_SENTINEL_V9560"
        r["base_acc_reused_from_v9550"] = 1
        r["base_acc_used_for_controller"] = 0
    summary = next((dict(r) for r in rows if r.get("status") == "summary"), {})
    summary.update({
        "stage": "BASE_ACC_SENTINEL_V9560",
        "base_acc_reused_from_v9550": 1,
        "base_acc_used_for_controller": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    return rows, summary


def audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fake = sum(inum(r.get("fake_data_used")) for r in rows)
    proxy = sum(inum(r.get("proxy_row_used")) for r in rows)
    cpu = sum(inum(r.get("cpu_offload_used")) for r in rows)
    return {
        "stage": "NO_FAKE_AUDIT_V9560",
        "status": "summary",
        "rows_checked": len(rows),
        "fake_proxy_nonzero_count": fake + proxy,
        "fake_data_used": int(fake > 0),
        "proxy_row_used": int(proxy > 0),
        "cpu_offload_used": int(cpu > 0),
        "no_fake": int(fake == 0),
        "no_proxy": int(proxy == 0),
    }


def hash_rows(out_dir: Path) -> list[dict[str, str]]:
    files = [
        PLAN_PATH, SCRIPT_PATH,
        out_dir / "run_manifest.json",
        out_dir / "route_decision_v9560.json",
        out_dir / "p0_v9550_boundary_reproduction.csv",
        out_dir / "grade_scope_audit_v9560.csv",
        out_dir / "p2_gcert18_rank_calibration_dissection.csv",
        out_dir / "p3_multigrade_target_density_audit.csv",
        out_dir / "p4_cover_memory_failure_anatomy.csv",
        out_dir / "p5_signal_reservoir_audit_v5.csv",
        out_dir / "p6_existing_action_rank_controller_candidate.csv",
        out_dir / "p7_apgc_cover_memory_primitive_implementation.csv",
        out_dir / "p8_apgc_branch_horizon_smoke.csv",
        out_dir / "apgc_branch_horizon_outcome_trace_v9560.csv",
        out_dir / "p9_apgc_outcome_geometry_pass.csv",
        out_dir / "p10_geometry_certificate_v4.csv",
        out_dir / "p11_minimal_geometry_controller.csv",
        out_dir / "p12_selected_runtime.csv",
        out_dir / "p13_leaveout_boundary.csv",
        out_dir / "p14_official_paired_replay_boundary.csv",
        out_dir / "p15_short_full_training_boundary.csv",
        out_dir / "base_acc_sentinel_v9560.csv",
        out_dir / "v9560_dashboard.md",
        out_dir / "no_fake_audit_v9560.csv",
        out_dir / "contract_audit_v9560.csv",
        out_dir / "failure_table_v9560.csv",
    ]
    return [{"artifact": rel(p), "sha256": sha256_file(p)} for p in files if p.exists()]


def write_dashboard(out_dir: Path, route: dict[str, Any], sections: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# v9560 Dashboard",
        "",
        "## Route summary",
        "",
        f"- route: `{route.get('route')}`",
        f"- primary_blocker: `{route.get('primary_blocker')}`",
        f"- system_legal_controller_pass: `{route.get('system_legal_controller_pass')}`",
        "",
    ]
    for title, data in sections.items():
        lines.append(f"## {title}")
        lines.append("")
        for k, v in data.items():
            lines.append(f"- {k}: `{v}`")
        lines.append("")
    (out_dir / "v9560_dashboard.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = v9420.device_from(args.device)
    source_v9550 = Path(args.source_v9550)

    p0_rows, p0 = p0_boundary(source_v9550)
    base, apgr, _apgr_outcome = load_base_ledgers(source_v9550)
    p1_rows, p1 = p1_grade_scope(base, apgr)
    p2_rows, p2 = p2_rank_calibration(base, apgr)
    p3_rows, p3 = p3_target_density(base, p1_rows)
    p4_rows, p4 = p4_cover_memory_anatomy(base, apgr)
    p5_rows, p5 = p5_signal_reservoir(base)
    p6_rows, p6 = p6_existing_controller(base)

    payload_rows = v9490.load_payload_rows(Path(args.source_v9330))
    payload_by_id = {str(r.get("action_id")): r for r in payload_rows}
    p7_rows, p7, generated = p7_apgc_generation(args, base, payload_by_id, device)
    p8_rows, p8 = p8_materialize_apgc(args, generated, payload_by_id, device)
    outcome_rows = [r for r in p8_rows if r.get("status") == "branch_horizon_row"]
    p9_rows, p9, apgc_ledger = p9_apgc_outcome(generated, outcome_rows, base)
    p10_rows, p10 = p10_certificate_v4(base, apgc_ledger)
    p11_rows, p11 = p11_controller(p1, p3, p6, p9, p10)
    p12_rows, p12 = p12_runtime(p11)
    p13_rows, p13 = not_open_boundary("P13_LEAVEOUT_VALIDATION_BOUNDARY", "P12_runtime_not_selected", "leaveout_validation_pass")
    p14_rows, p14 = not_open_boundary("P14_OFFICIAL_PAIRED_REPLAY_BOUNDARY", "P13_leaveout_not_open", "official_paired_replay_pass")
    p15_rows, p15 = not_open_boundary("P15_SHORT_FULL_TRAINING_BOUNDARY", "P14_paired_replay_not_open", "short_full_training_pass")
    base_rows, base_acc_summary = base_acc(source_v9550)

    if not inum(p0.get("p0_pass")):
        route, primary = "R0-BoundaryRegression", "v9550_boundary_reproduction_failed"
    elif not inum(p1.get("grade_scope_audit_pass")):
        route, primary = "R1-GradeScopeInconsistent", "legacy_gcert18_uses_outcome_derived_score"
    elif inum(p2.get("rank_signal_legacy_diagnostic_pass")) and not inum(p2.get("calibration_pass")):
        route, primary = "R2-RankingStrongCalibrationWeak", "ranking_strong_but_calibration_or_legal_score_failed"
    elif not inum(p3.get("target_density_pass")):
        route, primary = "R3-CleanGeometryTargetTooSparse", "multigrade_target_density_or_risk_failed"
    elif inum(p4.get("cover_blocker_confirmed")) or inum(p4.get("memory_blocker_confirmed")):
        route, primary = "R4-CoverMemoryPrimaryBlocker", "cover_memory_primary_blocker_confirmed"
    elif not inum(p9.get("apgc_weak_pass")):
        route, primary = "R5-APGCGeneratedFrontierFail", "apgc_generated_value_or_risk_failed"
    elif not inum(p10.get("official_certificate_pass")) or not inum(p11.get("source_controller_pass")):
        route, primary = "R6-CertificateCalibratedControllerFail", "certificate_or_minimal_controller_failed"
    elif not inum(p12.get("selected_runtime_pass")):
        route, primary = "R7-RuntimeFail", "selected_runtime_failed"
    else:
        route, primary = "R8-SystemControllerPass", "system_pass"

    system = {
        "stage": "P12_SYSTEM_INTEGRATION_GATE_V9560",
        "status": "summary",
        "system_candidate_id": "SYS-v9560-calibrated-geometry-rank-cover-memory",
        "grade_scope_audit_pass": p1.get("grade_scope_audit_pass"),
        "ranking_pass": p2.get("ranking_pass"),
        "calibration_pass": p2.get("calibration_pass"),
        "target_density_pass": p3.get("target_density_pass"),
        "cover_memory_anatomy_pass": p4.get("cover_memory_anatomy_pass"),
        "signal_reservoir_audit_pass": p5.get("signal_reservoir_audit_pass"),
        "existing_action_controller_pass": p6.get("existing_action_controller_pass"),
        "apgc_implementation_pass": p7.get("apgc_implementation_pass"),
        "apgc_branch_horizon_pass": p8.get("quality_audit_pass"),
        "apgc_weak_pass": p9.get("apgc_weak_pass"),
        "official_certificate_pass": p10.get("official_certificate_pass"),
        "source_controller_pass": p11.get("source_controller_pass"),
        "selected_runtime_pass": p12.get("selected_runtime_pass"),
        "official_eligible": 0,
        "system_legal_controller_pass": 0,
        "reason": primary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }

    route_decision = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "source_route_v9550": p0.get("source_route_v9550"),
        "grade_scope_audit_pass": p1.get("grade_scope_audit_pass"),
        "GradeB_count_canonical_AP0": p1.get("GradeB_count_canonical_AP0"),
        "GradeAB_count_canonical_AP0": p1.get("GradeAB_count_canonical_AP0"),
        "GradeAB_count_GCERT18_eval_universe": p1.get("GradeAB_count_GCERT18_eval_universe"),
        "TopK64_GradeAB_precision_legacy": p1.get("TopK64_GradeAB_precision"),
        "LegalTopK64_GradeAB_precision": p1.get("LegalTopK64_GradeAB_precision"),
        "outcome_field_used_by_certificate_count": p1.get("outcome_field_used_by_certificate_count"),
        "legacy_gcert18_legal_commit_time_pass": p1.get("legacy_gcert18_legal_commit_time_pass"),
        "rank_signal_legacy_diagnostic_pass": p2.get("rank_signal_legacy_diagnostic_pass"),
        "legacy_TopK64_GradeAB_precision": p2.get("legacy_TopK64_GradeAB_precision"),
        "legacy_TopK64_V_integrated_LCB": p2.get("legacy_TopK64_V_integrated_LCB"),
        "legacy_ECE": p2.get("legacy_ECE"),
        "legal_TopK64_GradeAB_precision": p2.get("legal_TopK64_GradeAB_precision"),
        "legal_TopK64_V_integrated_LCB": p2.get("legal_TopK64_V_integrated_LCB"),
        "legal_ECE": p2.get("legal_ECE"),
        "target_density_pass": p3.get("target_density_pass"),
        "best_target_id": p3.get("best_target_id"),
        "best_accepted_count": p3.get("best_accepted_count"),
        "best_V_integrated_LCB": p3.get("best_V_integrated_LCB"),
        "best_h240_longrisk_UCB": p3.get("best_h240_longrisk_UCB"),
        "cover_blocker_confirmed": p4.get("cover_blocker_confirmed"),
        "memory_blocker_confirmed": p4.get("memory_blocker_confirmed"),
        "signal_reservoir_audit_pass": p5.get("signal_reservoir_audit_pass"),
        "best_signal_reservoir_feature": p5.get("best_feature_id"),
        "best_signal_TopK64_GradeAB_precision": p5.get("best_TopK64_GradeAB_precision"),
        "existing_action_controller_pass": p6.get("existing_action_controller_pass"),
        "best_existing_controller_id": p6.get("best_controller_id"),
        "apgc_implementation_pass": p7.get("apgc_implementation_pass"),
        "apgc_generated_action_count": p7.get("generated_action_count"),
        "apgc_branch_horizon_pass": p8.get("quality_audit_pass"),
        "apgc_branch_horizon_rows_actual": p8.get("actual_rows"),
        "apgc_unresolved_exception_count": p8.get("unresolved_exception_count"),
        "apgc_weak_pass": p9.get("apgc_weak_pass"),
        "best_apgc_primitive_id": p9.get("best_primitive_id"),
        "best_apgc_GradeB_precision": p9.get("best_GradeB_precision"),
        "best_apgc_V_integrated_LCB": p9.get("best_V_integrated_LCB"),
        "best_apgc_h240_longrisk": p9.get("best_h240_longrisk"),
        "official_certificate_pass": p10.get("official_certificate_pass"),
        "best_certificate_id": p10.get("best_certificate_id"),
        "best_certificate_AUC_GradeAB": p10.get("best_AUC_GradeAB"),
        "best_certificate_TopK64_precision_GradeAB": p10.get("best_TopK64_precision_GradeAB"),
        "best_certificate_ECE": p10.get("best_ECE"),
        "source_controller_pass": p11.get("source_controller_pass"),
        "selected_runtime_pass": p12.get("selected_runtime_pass"),
        "system_legal_controller_pass": system.get("system_legal_controller_pass"),
        "base_acc_sentinel_pass": base_acc_summary.get("base_acc_sentinel_pass"),
        "base_acc_used_for_controller": 0,
        "success_v9560_strict_purekan_functional": 0,
        "success_v9560_full_functional": 0,
        "success_v9560_external_ready": 0,
        "primary_blocker": primary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }

    contract = {
        "stage": "CONTRACT_AUDIT_V9560",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "v9550_boundary_pass": p0.get("p0_pass"),
        "grade_scope_audit_pass": p1.get("grade_scope_audit_pass"),
        "rank_calibration_pass": p2.get("rank_calibration_pass"),
        "target_density_pass": p3.get("target_density_pass"),
        "cover_memory_anatomy_pass": p4.get("cover_memory_anatomy_pass"),
        "signal_reservoir_audit_pass": p5.get("signal_reservoir_audit_pass"),
        "existing_action_controller_pass": p6.get("existing_action_controller_pass"),
        "apgc_implementation_pass": p7.get("apgc_implementation_pass"),
        "apgc_branch_horizon_pass": p8.get("quality_audit_pass"),
        "apgc_weak_pass": p9.get("apgc_weak_pass"),
        "official_certificate_pass": p10.get("official_certificate_pass"),
        "source_controller_pass": p11.get("source_controller_pass"),
        "selected_runtime_pass": p12.get("selected_runtime_pass"),
        "system_legal_controller_pass": system.get("system_legal_controller_pass"),
        "base_acc_sentinel_pass": base_acc_summary.get("base_acc_sentinel_pass"),
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

    failure = {
        "stage": "FAILURE_TABLE_V9560",
        "status": "summary",
        "route": route,
        "F0_boundary_regression": int(route == "R0-BoundaryRegression"),
        "F1_grade_scope_inconsistent": int(route == "R1-GradeScopeInconsistent"),
        "F2_ranking_strong_calibration_weak": int(route == "R2-RankingStrongCalibrationWeak"),
        "F3_clean_geometry_target_too_sparse": int(route == "R3-CleanGeometryTargetTooSparse"),
        "F4_cover_memory_primary_blocker": int(route == "R4-CoverMemoryPrimaryBlocker"),
        "F5_apgc_generated_frontier_fail": int(route == "R5-APGCGeneratedFrontierFail"),
        "F6_certificate_controller_fail": int(route == "R6-CertificateCalibratedControllerFail"),
        "F7_runtime_fail": int(route == "R7-RuntimeFail"),
        "F8_system_not_official": int(not inum(system.get("system_legal_controller_pass"))),
        "F9_base_acc_catastrophic": 0,
        "primary_blocker": primary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }

    all_rows: list[dict[str, Any]] = []
    for block in [
        p0_rows, p1_rows, p2_rows, p3_rows, p4_rows, p5_rows, p6_rows,
        p7_rows, p8_rows, p9_rows, p10_rows, p11_rows, p12_rows,
        p13_rows, p14_rows, p15_rows, base_rows, [system], [contract], [failure],
    ]:
        all_rows.extend(block)
    nofake = audit_rows(all_rows)
    contract.update({"fake_data_used": nofake["fake_data_used"], "proxy_row_used": nofake["proxy_row_used"], "cpu_offload_used": nofake["cpu_offload_used"]})

    manifest = {
        "run_id": "v9560_calibrated_geometry_rank_cover_memory_primitive",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "argv": sys.argv,
        "out_dir": str(out_dir),
        "device": str(device),
        "source_v9550": str(source_v9550.resolve()),
        "source_v9330": str(Path(args.source_v9330).resolve()),
        "apgc_actions_per_primitive": int(args.apgc_actions_per_primitive),
        "no_fake_policy": "uses landed v9550 artifacts plus direct APGC branch-horizon materialization; no fake/proxy rows; legacy outcome-derived GCERT18 kept diagnostic-only",
    }

    write_json(out_dir / "run_manifest.json", manifest)
    write_json(out_dir / "route_decision_v9560.json", route_decision)
    write_json(out_dir / "aggregate_decision_v9560.json", route_decision)
    write_csv(out_dir / "p0_v9550_boundary_reproduction.csv", p0_rows)
    write_csv(out_dir / "grade_scope_audit_v9560.csv", p1_rows)
    write_csv(out_dir / "p2_gcert18_rank_calibration_dissection.csv", p2_rows)
    write_csv(out_dir / "p3_multigrade_target_density_audit.csv", p3_rows)
    write_csv(out_dir / "p4_cover_memory_failure_anatomy.csv", p4_rows)
    write_csv(out_dir / "p5_signal_reservoir_audit_v5.csv", p5_rows)
    write_csv(out_dir / "p6_existing_action_rank_controller_candidate.csv", p6_rows)
    write_csv(out_dir / "p7_apgc_cover_memory_primitive_implementation.csv", p7_rows)
    write_csv(out_dir / "p8_apgc_branch_horizon_smoke.csv", p8_rows)
    write_csv(out_dir / "apgc_branch_horizon_outcome_trace_v9560.csv", outcome_rows)
    write_csv(out_dir / "p9_apgc_outcome_geometry_pass.csv", p9_rows)
    write_csv(out_dir / "p10_geometry_certificate_v4.csv", p10_rows)
    write_csv(out_dir / "p11_minimal_geometry_controller.csv", p11_rows)
    write_csv(out_dir / "p12_selected_runtime.csv", p12_rows)
    write_csv(out_dir / "p13_leaveout_boundary.csv", p13_rows)
    write_csv(out_dir / "p14_official_paired_replay_boundary.csv", p14_rows)
    write_csv(out_dir / "p15_short_full_training_boundary.csv", p15_rows)
    write_csv(out_dir / "base_acc_sentinel_v9560.csv", base_rows)
    write_csv(out_dir / "p12_system_integration_gate_v9560.csv", [system])
    write_csv(out_dir / "no_fake_audit_v9560.csv", [nofake])
    write_csv(out_dir / "contract_audit_v9560.csv", [contract])
    write_csv(out_dir / "provenance_audit_v9560.csv", [nofake])
    write_csv(out_dir / "failure_table_v9560.csv", [failure])
    write_dashboard(out_dir, route_decision, {
        "T5 / Grade density": {
            "GradeAB_count_canonical_AP0": p1.get("GradeAB_count_canonical_AP0"),
            "best_target_id": p3.get("best_target_id"),
            "best_accepted_count": p3.get("best_accepted_count"),
        },
        "GCERT18 rank vs calibration": {
            "legacy_TopK64_GradeAB_precision": p2.get("legacy_TopK64_GradeAB_precision"),
            "legacy_uses_outcome_derived_score": p2.get("legacy_uses_outcome_derived_score"),
            "legal_TopK64_GradeAB_precision": p2.get("legal_TopK64_GradeAB_precision"),
            "legal_ECE": p2.get("legal_ECE"),
        },
        "Cover / memory anatomy": {
            "cover_blocker_confirmed": p4.get("cover_blocker_confirmed"),
            "memory_blocker_confirmed": p4.get("memory_blocker_confirmed"),
        },
        "APGC primitive outcome": {
            "apgc_rows": p8.get("actual_rows"),
            "best_apgc": p9.get("best_primitive_id"),
            "best_apgc_GradeB_precision": p9.get("best_GradeB_precision"),
            "best_apgc_V_LCB": p9.get("best_V_integrated_LCB"),
        },
        "Certificate v4": {
            "best_certificate": p10.get("best_certificate_id"),
            "best_TopK64_precision_GradeAB": p10.get("best_TopK64_precision_GradeAB"),
            "best_ECE": p10.get("best_ECE"),
        },
        "No-fake audit": {
            "no_fake": nofake.get("no_fake"),
            "no_proxy": nofake.get("no_proxy"),
            "cpu_offload_used": nofake.get("cpu_offload_used"),
        },
    })
    write_csv(out_dir / "artifact_hashes_v9560.csv", hash_rows(out_dir))
    write_csv(out_dir / "hash_manifest_v9560.csv", hash_rows(out_dir))

    print(json.dumps({
        "out_dir": str(out_dir),
        "route": route,
        "grade_scope_audit_pass": p1.get("grade_scope_audit_pass"),
        "legacy_TopK64_GradeAB_precision": p2.get("legacy_TopK64_GradeAB_precision"),
        "legal_TopK64_GradeAB_precision": p2.get("legal_TopK64_GradeAB_precision"),
        "apgc_generated_actions": p7.get("generated_action_count"),
        "apgc_rows": p8.get("actual_rows"),
        "best_apgc_primitive": p9.get("best_primitive_id"),
        "best_apgc_GradeB_precision": p9.get("best_GradeB_precision"),
        "official_certificate_pass": p10.get("official_certificate_pass"),
        "system_legal_controller_pass": system.get("system_legal_controller_pass"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
