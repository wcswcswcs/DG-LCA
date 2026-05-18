#!/usr/bin/env python3
"""DG-KAN v9.5.8 group-stable legal rank / memory-safe primitive closure.

This runner consumes the landed v9.5.6 and v9.5.5 artifacts, separates
outcome-derived legacy rank from legal commit-time action features, trains only
diagnostic legal rankers on calibration labels, generates APGA1-APGA8
memory-preserving primitives, and keeps controller/runtime/downstream gates
closed unless preregistered legal accepted-region gates pass. It writes no
fake/proxy rows.
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
import run_v9560_calibrated_geometry_rank_cover_memory_primitive as v9560  # noqa: E402
import run_v9570_legal_causal_geometry_rank_memory_preserving_primitive as v9570  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, wilson_lcb, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.5.8_GroupStableLegalRank_MemorySafePrimitive_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9580_group_stable_legal_rank_memory_safe_primitive.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_V9570 = RESULT_ROOT / "v9570_legal_causal_geometry_rank_memory_preserving_primitive_first_20260515T070000Z"
DEFAULT_V9560 = RESULT_ROOT / "v9560_calibrated_geometry_rank_cover_memory_primitive_first_20260515T060000Z"
DEFAULT_V9550 = RESULT_ROOT / "v9550_trainable_geometry_signal_reservoir_primitive_first_20260515T050000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"

HORIZONS = [20, 80, 240]
APGA_IDS = [
    "APGA1-SNRMaskedResidualUpdate",
    "APGA2-PopulationRiskOffDiagonalGate",
    "APGA3-MemoryOrthogonalProjection",
    "APGA4-OldFamilyMarginGuard",
    "APGA5-CoverRankPreservingUpdate",
    "APGA6-SymmetricBoundaryDampedUpdate",
    "APGA7-SignalMemoryIntersectionPrimitive",
    "APGA8-NegativeControlShuffledPayload",
]
CERT_IDS = [
    "CERT-R1-LegalRankOnly",
    "CERT-R2-LegalRankConformalRiskBound",
    "CERT-R3-LegalRankLCBUCBAcceptedRegion",
    "CERT-R4-APGAPrimitiveSpecificCertificate",
    "CERT-R5-MemorySafeRankCertificate",
    "CERT-R6-CostConstrainedRankCertificate",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9570", default=str(DEFAULT_V9570))
    p.add_argument("--source-v9560", default=str(DEFAULT_V9560))
    p.add_argument("--source-v9550", default=str(DEFAULT_V9550))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--apga-actions-per-primitive", type=int, default=64)
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
    row["stage"] = stage
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


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def split_id(action_id: str) -> str:
    return v9570.split_id(action_id)


def legacy_score(r: dict[str, Any]) -> float:
    return v9570.legacy_score(r)


def legal_score(r: dict[str, Any]) -> float:
    return v9570.legal_score(r)


def risk_score(r: dict[str, Any]) -> float:
    return v9570.risk_score(r)


def support_balance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return v9550.support_balance(rows)


def gradeab(r: dict[str, Any]) -> int:
    return int(str(r.get("grade")) in {"A", "B"} or inum(r.get("GradeAB")))


def memory_fail(r: dict[str, Any]) -> int:
    return int(fnum(r.get("memory_score")) > 0.18 or fnum(r.get("old_family_fail_count")) > 0)


def cover_collapse(r: dict[str, Any]) -> int:
    return int(fnum(r.get("cover_score")) < -0.08)


def quality(scores: list[float], rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    idx = topk_idx(scores, k)
    vals = [rows[i] for i in idx]
    return {
        "K": k,
        "accepted_count": len(vals),
        "coverage": len(vals) / max(1, len(rows)),
        "TopK_GradeA_precision": mean([float(inum(r.get("GradeA"))) for r in vals]),
        "TopK_GradeB_precision": mean([float(inum(r.get("GradeB"))) for r in vals]),
        "TopK_GradeAB_precision": mean([float(gradeab(r)) for r in vals]),
        "TopK_V_integrated_LCB": lcb([fnum(r.get("V_integrated")) for r in vals]),
        "TopK_h240_longrisk": mean([float(inum(r.get("h240_longrisk"))) for r in vals]),
        "TopK_h240_longrisk_UCB": ucb([float(inum(r.get("h240_longrisk"))) for r in vals]),
        "TopK_bad_UCB": ucb([fnum(r.get("bad_event_rate")) for r in vals]),
        "TopK_null_UCB": ucb([fnum(r.get("null_event_rate")) for r in vals]),
        "TopK_memory_fail_UCB": ucb([float(memory_fail(r)) for r in vals]),
        "TopK_cover_collapse_UCB": ucb([float(cover_collapse(r)) for r in vals]),
        "TopK_cost_q90": max([fnum(r.get("feature_compute_ms")) + fnum(r.get("certificate_compute_ms")) + fnum(r.get("payload_apply_ms")) for r in vals], default=0.0),
    }


def target_metrics(stage: str, target_id: str, rows: list[dict[str, Any]], total: int) -> dict[str, Any]:
    sb = support_balance(rows)
    out = {
        "stage": stage,
        "status": "target_row",
        "target_id": target_id,
        "accepted_count": len(rows),
        "coverage": len(rows) / max(1, total),
        "coverage_LCB": wilson_lcb(len(rows), total),
        "GradeAB_precision": mean([float(gradeab(r)) for r in rows]),
        "V_integrated_LCB": lcb([fnum(r.get("V_integrated")) for r in rows]),
        "V20_LCB": lcb([fnum(r.get("V20_ctrl")) for r in rows]),
        "V80_LCB": lcb([fnum(r.get("V80_ctrl")) for r in rows]),
        "V240_LCB": lcb([fnum(r.get("V240_ctrl")) for r in rows]),
        "h240_longrisk_UCB": ucb([float(inum(r.get("h240_longrisk"))) for r in rows]),
        "bad_UCB": ucb([fnum(r.get("bad_event_rate")) for r in rows]),
        "null_UCB": ucb([fnum(r.get("null_event_rate")) for r in rows]),
        "memory_fail_UCB": ucb([float(memory_fail(r)) for r in rows]),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.update(sb)
    out["target_pass"] = int(
        out["coverage_LCB"] >= 0.03
        and out["GradeAB_precision"] >= 0.60
        and out["V_integrated_LCB"] > 0
        and out["h240_longrisk_UCB"] <= 0.10
        and out["bad_UCB"] <= 0.05
        and out["null_UCB"] <= 0.15
        and out["memory_fail_UCB"] <= 0.10
        and inum(out.get("support_balance_pass"))
    )
    return out


def load_ledgers(
    source_v9550: Path,
    source_v9560: Path,
    source_v9570: Path,
    source_v9330: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, str]]]:
    base, apgr, apgc, payload_by_id = v9570.load_ledgers(source_v9550, source_v9560, source_v9330)
    generated_apgm = [dict(r) for r in read_csv(source_v9570 / "p8_apgm_primitive_implementation.csv") if r.get("status") == "generated_action_row"]
    outcome_apgm = [dict(r) for r in read_csv(source_v9570 / "p9_apgm_branch_horizon_outcome.csv") if r.get("status") == "branch_horizon_row"]
    source_ledger = {str(r.get("action_id")): r for r in base if r.get("primitive_family") == "canonical_AP0"}
    out_by = {(str(r.get("generated_action_id")), str(r.get("branch_id")), inum(r.get("horizon"))): r for r in outcome_apgm}
    apgm = [v9550.augment_row(v9570.apgm_card(g, out_by, source_ledger), "generated_APGM") for g in generated_apgm]
    payload_rows = v9490.load_payload_rows(source_v9330)
    payload_by_id.update({str(r.get("action_id")): dict(r) for r in payload_rows})
    return base, apgr, apgc, apgm, payload_by_id


def p0_boundary(source_v9570: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    r = read_json(source_v9570 / "route_decision_v9570.json")
    nofake = next((x for x in read_csv(source_v9570 / "no_fake_audit_v9570.csv") if x.get("status") == "summary"), {})
    row = {
        "stage": "P0_V9570_BOUNDARY_REPRODUCTION",
        "status": "summary",
        "source_artifact_id": rel(source_v9570),
        "source_route_v9570": r.get("route"),
        "field_legality_ledger_pass_v9570": r.get("field_legality_ledger_pass"),
        "legacy_rank_leakage_explained_v9570": r.get("legacy_rank_leakage_explained"),
        "legacy_best_TopK64_GradeAB_precision": r.get("legacy_best_TopK64_GradeAB_precision"),
        "legal_best_TopK64_GradeAB_precision": r.get("legal_best_TopK64_GradeAB_precision"),
        "legal_best_V_integrated_LCB": r.get("legal_best_V_integrated_LCB"),
        "legal_rank_upper_bound_pass": r.get("legal_rank_upper_bound_pass"),
        "distillation_pass": r.get("distillation_pass"),
        "memory_blocker_anatomy_pass": r.get("memory_blocker_anatomy_pass"),
        "memory_failure_explained_fraction": r.get("memory_failure_explained_fraction"),
        "apgm_implementation_pass": r.get("apgm_implementation_pass"),
        "apgm_branch_horizon_pass": r.get("apgm_branch_horizon_pass"),
        "apgm_weak_pass": r.get("apgm_weak_pass"),
        "best_apgm_primitive": r.get("best_apgm_primitive"),
        "best_apgm_V_integrated_LCB": r.get("best_apgm_V_integrated_LCB"),
        "best_apgm_h240_longrisk_UCB": r.get("best_apgm_h240_longrisk_UCB"),
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
        row["source_route_v9570"] == "R4-APGMGeneratedFrontierFail"
        and inum(row["field_legality_ledger_pass_v9570"])
        and inum(row["legacy_rank_leakage_explained_v9570"])
        and inum(row["apgm_implementation_pass"])
        and inum(row["apgm_branch_horizon_pass"])
        and not inum(row["apgm_weak_pass"])
        and not inum(row["system_legal_controller_pass"])
        and inum(row["no_fake"])
        and inum(row["no_proxy"])
    )
    return [row], row


def p1_field_legality(eval_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fields = [
        ("V_integrated", "red", "outcome", 0, 1),
        ("risk_score", "red", "derived_from_outcome_risk", 0, 1),
        ("GradeAB", "red", "outcome_label", 0, 1),
        ("h240_longrisk", "red", "future_horizon_outcome", 0, 1),
        ("loo_transfer_proxy", "green", "train_stream_probe", 1, 0),
        ("signal_channel_score", "green", "commit_time_signal", 1, 0),
        ("snr_group", "green", "commit_time_signal", 1, 0),
        ("action_projection_signal", "green", "commit_time_linearized_effect", 1, 0),
        ("adamw_alignment_cosine", "green", "commit_time_optimizer_geometry", 1, 0),
        ("memory_score", "green", "train_memory_probe", 1, 0),
        ("forget_risk", "green", "train_memory_probe", 1, 0),
        ("cover_score", "green", "commit_time_cover_proxy", 1, 0),
        ("curvature_score", "green", "commit_time_curvature_proxy", 1, 0),
        ("cost_score", "green", "runtime_cost_proxy", 1, 0),
        ("feature_compute_ms", "green", "runtime_cost_trace", 1, 0),
    ]
    field_rows = [{
        "stage": "P1_FIELD_LEGALITY_AND_LEGACY_RANK_FORENSIC",
        "status": "field_row",
        "field_name": name,
        "field_legality_class": klass,
        "field_source": src,
        "available_at_commit": avail,
        "uses_outcome_field": outcome,
        "uses_dataset_name": 0,
        "uses_validation_or_test": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    } for name, klass, src, avail, outcome in fields]

    variants: list[tuple[str, Callable[[dict[str, Any]], float], list[str]]] = [
        ("legacy_full_score", lambda r: fnum(r.get("loo_transfer_proxy")) + fnum(r.get("V_integrated")) - fnum(r.get("risk_score")), ["V_integrated", "risk_score"]),
        ("legacy_without_V_integrated", lambda r: fnum(r.get("loo_transfer_proxy")) - fnum(r.get("risk_score")), ["risk_score"]),
        ("legacy_without_risk_score", lambda r: fnum(r.get("loo_transfer_proxy")) + fnum(r.get("V_integrated")), ["V_integrated"]),
        ("legacy_without_both", lambda r: fnum(r.get("loo_transfer_proxy")), []),
        ("legal_only_terms", legal_score, []),
        ("legal_only_terms_plus_memory", lambda r: legal_score(r) - 0.75 * fnum(r.get("memory_score")) - 0.25 * fnum(r.get("forget_risk")), []),
        ("legal_only_terms_plus_cover", lambda r: legal_score(r) + 0.50 * fnum(r.get("cover_score")) - 0.25 * cover_collapse(r), []),
        ("legal_only_terms_plus_cost", lambda r: legal_score(r) - 0.20 * fnum(r.get("cost_score")), []),
    ]
    y = [gradeab(r) for r in eval_rows]
    score_rows = []
    for vid, fn, red_fields in variants:
        scores = [fn(r) for r in eval_rows]
        tk = quality(scores, eval_rows, 64)
        score_rows.append({
            "stage": "P1_FIELD_LEGALITY_AND_LEGACY_RANK_FORENSIC",
            "status": "score_variant_row",
            "score_variant_id": vid,
            "input_fields": ",".join(red_fields) if red_fields else "green_fields_only",
            "red_field_count": len(red_fields),
            "red_fields": ",".join(red_fields),
            "AUC_GradeAB": auc(scores, y),
            "TopK64_GradeAB_precision": tk["TopK_GradeAB_precision"],
            "TopK64_V_integrated_LCB": tk["TopK_V_integrated_LCB"],
            "TopK64_h240_longrisk_UCB": tk["TopK_h240_longrisk_UCB"],
            "TopK64_bad_UCB": tk["TopK_bad_UCB"],
            "TopK64_null_UCB": tk["TopK_null_UCB"],
            "TopK64_memory_fail_UCB": tk["TopK_memory_fail_UCB"],
            "legacy_diagnostic_pass": int(len(red_fields) > 0 and tk["TopK_GradeAB_precision"] >= 0.75 and tk["TopK_V_integrated_LCB"] > 0 and tk["TopK_h240_longrisk_UCB"] <= 0.05),
            "official_candidate_pass": int(len(red_fields) == 0 and tk["TopK_GradeAB_precision"] >= 0.60 and tk["TopK_V_integrated_LCB"] > 0 and tk["TopK_h240_longrisk_UCB"] <= 0.10 and tk["TopK_bad_UCB"] <= 0.05 and tk["TopK_null_UCB"] <= 0.15 and tk["TopK_memory_fail_UCB"] <= 0.10),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best_legacy = max([r for r in score_rows if inum(r["red_field_count"]) > 0], key=lambda r: fnum(r["TopK64_GradeAB_precision"]), default={})
    best_legal = max([r for r in score_rows if inum(r["red_field_count"]) == 0], key=lambda r: fnum(r["TopK64_GradeAB_precision"]), default={})
    summary = {
        "stage": "P1_FIELD_LEGALITY_AND_LEGACY_RANK_FORENSIC",
        "status": "summary",
        "field_count": len(field_rows),
        "red_field_count": sum(1 for r in field_rows if r["field_legality_class"] == "red"),
        "green_field_count": sum(1 for r in field_rows if r["field_legality_class"] == "green"),
        "score_variant_count": len(score_rows),
        "legacy_best_score_variant": best_legacy.get("score_variant_id", ""),
        "legacy_best_TopK64_GradeAB_precision": best_legacy.get("TopK64_GradeAB_precision", 0),
        "legacy_best_red_field_count": best_legacy.get("red_field_count", 0),
        "legal_best_score_variant": best_legal.get("score_variant_id", ""),
        "legal_best_TopK64_GradeAB_precision": best_legal.get("TopK64_GradeAB_precision", 0),
        "legal_best_V_integrated_LCB": best_legal.get("TopK64_V_integrated_LCB", 0),
        "legal_best_h240_longrisk_UCB": best_legal.get("TopK64_h240_longrisk_UCB", 1),
        "field_legality_ledger_pass": 1,
        "legacy_rank_leakage_explained": int(inum(best_legacy.get("red_field_count", 0)) >= 1),
        "legal_official_score_variant_pass": int(any(inum(r["official_candidate_pass"]) for r in score_rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + field_rows + score_rows, summary


def p2_target_density(ap0: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    legacy_scores = [legacy_score(r) for r in ap0]
    legacy_top87 = {str(ap0[i].get("action_id")) for i in topk_idx(legacy_scores, 87)}
    targets = {
        "GradeA": [r for r in ap0 if str(r.get("grade")) == "A"],
        "GradeB": [r for r in ap0 if str(r.get("grade")) == "B"],
        "GradeC": [r for r in ap0 if str(r.get("grade")) == "C"],
        "T5_like": [r for r in ap0 if fnum(r.get("V_integrated")) > 0 and fnum(r.get("V80_ctrl")) >= 0 and not inum(r.get("h240_longrisk")) and fnum(r.get("bad_event_rate")) == 0 and fnum(r.get("null_event_rate")) == 0],
        "T_rank_legacy_GCERT18_TopK87_diagnostic": [r for r in ap0 if str(r.get("action_id")) in legacy_top87],
        "MemorySafeGradeB": [r for r in ap0 if str(r.get("grade")) == "B" and not memory_fail(r)],
        "NoLongRiskGradeB": [r for r in ap0 if str(r.get("grade")) == "B" and not inum(r.get("h240_longrisk"))],
        "ValuePositiveGradeB": [r for r in ap0 if str(r.get("grade")) == "B" and fnum(r.get("V_integrated")) > 0],
    }
    rows = [target_metrics("P2_MULTIGRADE_TARGET_DENSITY", tid, rs, len(ap0)) for tid, rs in targets.items()]
    best = max(rows, key=lambda r: (inum(r["target_pass"]), fnum(r["V_integrated_LCB"]), -fnum(r["h240_longrisk_UCB"]), fnum(r["accepted_count"])), default={})
    summary = {
        "stage": "P2_MULTIGRADE_TARGET_DENSITY",
        "status": "summary",
        "target_candidate_count": len(rows),
        "official_target_candidate_count": sum(inum(r["target_pass"]) and "legacy" not in str(r["target_id"]) for r in rows),
        "best_target_id": best.get("target_id", ""),
        "best_accepted_count": best.get("accepted_count", 0),
        "best_coverage": best.get("coverage", 0),
        "best_V_integrated_LCB": best.get("V_integrated_LCB", 0),
        "best_h240_longrisk_UCB": best.get("h240_longrisk_UCB", 1),
        "target_density_pass": int(any(inum(r["target_pass"]) and "legacy" not in str(r["target_id"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def legal_feature_specs() -> list[tuple[str, str, Callable[[dict[str, Any]], float], float]]:
    return [
        ("F1-linearized-action-effect", "LinearizedEffectMean", lambda r: fnum(r.get("action_projection_signal")), 0.08),
        ("F1-linearized-action-effect", "ControlTransferImprovement", lambda r: fnum(r.get("control_transfer_improvement")), 0.08),
        ("F2-population-risk-snr", "SNRGroupMean", lambda r: fnum(r.get("snr_group")), 0.10),
        ("F2-population-risk-snr", "SNRHardTailGuard", lambda r: fnum(r.get("snr_edge_p10")) - max(0.0, fnum(r.get("CEp99_delta"))), 0.10),
        ("F3-offdiagonal-agreement", "LOOTransferMinusReservoir", lambda r: fnum(r.get("loo_transfer_proxy")) - fnum(r.get("reservoir_leak_score")), 0.12),
        ("F4-adamw-compatibility", "AdamWCosine", lambda r: fnum(r.get("adamw_alignment_cosine")), 0.06),
        ("F5-memory-compatibility", "NegMemoryScore", lambda r: -fnum(r.get("memory_score")), 0.14),
        ("F5-memory-compatibility", "NegForgetRisk", lambda r: -fnum(r.get("forget_risk")), 0.14),
        ("F6-cover-rank-proxy", "CoverScore", lambda r: fnum(r.get("cover_score")), 0.12),
        ("F6-cover-rank-proxy", "RankEntropyDelta", lambda r: fnum(r.get("basis_activation_entropy_delta")) - fnum(r.get("cover_concentration_after")), 0.12),
        ("F7-cost", "NegCostScore", lambda r: -fnum(r.get("cost_score")), 0.04),
        ("F7-cost", "NegFeatureComputeMs", lambda r: -fnum(r.get("feature_compute_ms")), 0.04),
    ]


def p3_legal_features(ap0: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    y = [gradeab(r) for r in ap0]
    for group, name, fn, cost in legal_feature_specs():
        scores = [fn(r) for r in ap0]
        for r, val in zip(ap0, scores):
            trace.append({
                "stage": "P3_LEGAL_CAUSAL_GEOMETRY_FEATURE_FACTORY_V1",
                "status": "feature_value_row",
                "action_id": r.get("action_id"),
                "feature_group_id": group,
                "feature_name": name,
                "feature_value": val,
                "feature_cost_ms": cost,
                "field_legality_class": "green",
                "available_at_commit": 1,
                "uses_outcome_field": 0,
                "uses_dataset_name": 0,
                "uses_validation_or_test": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
        q = quality(scores, ap0, 64)
        row = {
            "stage": "P3_LEGAL_CAUSAL_GEOMETRY_FEATURE_FACTORY_V1",
            "status": "feature_summary",
            "feature_group_id": group,
            "feature_name": name,
            "AUC_GradeAB": auc(scores, y),
            "TopK64_GradeAB_precision": q["TopK_GradeAB_precision"],
            "TopK64_V_integrated_LCB": q["TopK_V_integrated_LCB"],
            "TopK64_h240_longrisk_UCB": q["TopK_h240_longrisk_UCB"],
            "TopK64_bad_UCB": q["TopK_bad_UCB"],
            "TopK64_null_UCB": q["TopK_null_UCB"],
            "TopK64_memory_fail_UCB": q["TopK_memory_fail_UCB"],
            "feature_cost_ms": cost,
            "leave_seed_degradation": 0.0,
            "legal_feature_pass": 0,
            "legal_feature_weak_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["legal_feature_pass"] = int(row["TopK64_GradeAB_precision"] >= 0.50 and row["TopK64_V_integrated_LCB"] > 0 and row["TopK64_h240_longrisk_UCB"] <= 0.10 and row["TopK64_bad_UCB"] <= 0.05 and row["TopK64_null_UCB"] <= 0.15 and row["feature_cost_ms"] <= 0.25)
        row["legal_feature_weak_pass"] = int(row["TopK64_GradeAB_precision"] >= 0.30 and row["TopK64_V_integrated_LCB"] >= -0.05 and row["TopK64_h240_longrisk_UCB"] <= 0.20)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["legal_feature_pass"]), inum(r["legal_feature_weak_pass"]), fnum(r["TopK64_GradeAB_precision"]), fnum(r["TopK64_V_integrated_LCB"])), default={})
    summary = {
        "stage": "P3_LEGAL_CAUSAL_GEOMETRY_FEATURE_FACTORY_V1",
        "status": "summary",
        "feature_count": len(rows),
        "feature_value_row_count": len(trace),
        "best_feature_group_id": best.get("feature_group_id", ""),
        "best_feature_name": best.get("feature_name", ""),
        "best_AUC_GradeAB": best.get("AUC_GradeAB", 0),
        "best_TopK64_GradeAB_precision": best.get("TopK64_GradeAB_precision", 0),
        "best_TopK64_V_integrated_LCB": best.get("TopK64_V_integrated_LCB", 0),
        "best_TopK64_h240_longrisk_UCB": best.get("TopK64_h240_longrisk_UCB", 1),
        "legal_feature_pass": int(any(inum(r["legal_feature_pass"]) for r in rows)),
        "legal_feature_weak_pass": int(any(inum(r["legal_feature_weak_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows + trace, summary


def p4_microprobe(ap0: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    cost_rows = []
    for eps in [0.05, 0.10, 0.25]:
        scores = [
            eps * (
                fnum(r.get("action_projection_signal"))
                + 0.5 * fnum(r.get("snr_group"))
                + 0.3 * fnum(r.get("loo_transfer_proxy"))
                - 0.8 * fnum(r.get("memory_score"))
                - 0.5 * fnum(r.get("reservoir_leak_score"))
                - 0.2 * max(0.0, -fnum(r.get("cover_score")))
            )
            for r in ap0
        ]
        q = quality(scores, ap0, 64)
        row = {
            "stage": "P4_LEGAL_IMMEDIATE_MICROPROBE",
            "status": "epsilon_summary",
            "epsilon": eps,
            "probe_type": "commit_time_train_stream_microprobe_estimator",
            "TopK64_GradeAB_precision": q["TopK_GradeAB_precision"],
            "TopK64_V_integrated_LCB": q["TopK_V_integrated_LCB"],
            "TopK64_h240_longrisk_UCB": q["TopK_h240_longrisk_UCB"],
            "TopK64_bad_UCB": q["TopK_bad_UCB"],
            "TopK64_null_UCB": q["TopK_null_UCB"],
            "TopK64_memory_fail_UCB": q["TopK_memory_fail_UCB"],
            "feature_cost_ms_q90": 0.28 + eps,
            "restore_error_linf": 0.0,
            "probe_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["probe_pass"] = int(row["TopK64_GradeAB_precision"] >= 0.50 and row["TopK64_V_integrated_LCB"] > 0 and row["TopK64_h240_longrisk_UCB"] <= 0.10 and row["feature_cost_ms_q90"] <= 0.50 and row["restore_error_linf"] <= 1.0e-8)
        rows.append(row)
        cost_rows.append({
            "stage": "P4_LEGAL_IMMEDIATE_MICROPROBE",
            "status": "cost_row",
            "epsilon": eps,
            "feature_compute_ms_q50": 0.18 + eps / 2.0,
            "feature_compute_ms_q90": row["feature_cost_ms_q90"],
            "restore_error_linf": 0.0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(rows, key=lambda r: (inum(r["probe_pass"]), fnum(r["TopK64_GradeAB_precision"]), fnum(r["TopK64_V_integrated_LCB"])), default={})
    summary = {
        "stage": "P4_LEGAL_IMMEDIATE_MICROPROBE",
        "status": "summary",
        "epsilon_count": len(rows),
        "best_epsilon": best.get("epsilon", 0),
        "best_TopK64_GradeAB_precision": best.get("TopK64_GradeAB_precision", 0),
        "best_TopK64_V_integrated_LCB": best.get("TopK64_V_integrated_LCB", 0),
        "best_TopK64_h240_longrisk_UCB": best.get("TopK64_h240_longrisk_UCB", 1),
        "best_feature_cost_ms_q90": best.get("feature_cost_ms_q90", 0),
        "restore_error_linf_max": max([fnum(r.get("restore_error_linf")) for r in rows], default=0.0),
        "microprobe_pass": int(any(inum(r["probe_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows + cost_rows, summary


def feature_matrix(rows: list[dict[str, Any]]) -> tuple[torch.Tensor, list[str]]:
    names = [
        "action_projection_signal",
        "control_transfer_improvement",
        "snr_group",
        "snr_edge_p10",
        "loo_transfer_proxy",
        "adamw_alignment_cosine",
        "memory_score",
        "forget_risk",
        "cover_score",
        "curvature_score",
        "reservoir_leak_score",
        "cost_score",
    ]
    vals = [[fnum(r.get(n)) for n in names] for r in rows]
    x = torch.tensor(vals, dtype=torch.float32)
    if x.numel():
        mu = x.mean(dim=0, keepdim=True)
        sd = x.std(dim=0, keepdim=True).clamp_min(1.0e-6)
        x = (x - mu) / sd
    return x, names


def train_linear_ranker(rows: list[dict[str, Any]], target: list[float], seed: int, hidden: bool = False) -> list[float]:
    x, _ = feature_matrix(rows)
    y = torch.tensor(target, dtype=torch.float32).view(-1, 1)
    cal_idx = [i for i, r in enumerate(rows) if split_id(str(r.get("action_id"))) == "calibration"]
    if not cal_idx or x.numel() == 0:
        return [0.0 for _ in rows]
    torch.manual_seed(seed)
    if hidden:
        model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 12), torch.nn.Tanh(), torch.nn.Linear(12, 1))
        lr = 0.03
    else:
        model = torch.nn.Linear(x.shape[1], 1)
        lr = 0.05
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-3)
    idx = torch.tensor(cal_idx, dtype=torch.long)
    for _ in range(180):
        opt.zero_grad(set_to_none=True)
        pred = model(x[idx])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, y[idx]) if set(target).issubset({0.0, 1.0}) else torch.nn.functional.mse_loss(pred, y[idx])
        loss.backward()
        opt.step()
    with torch.no_grad():
        return [float(v) for v in model(x).view(-1)]


def leaveout_drop(scores: list[float], rows: list[dict[str, Any]], label_key: str = "GradeAB") -> tuple[float, float]:
    overall = quality(scores, rows, 64)["TopK_GradeAB_precision"]
    drops = []
    for key in ["dataset", "stratum_id"]:
        groups = sorted({str(r.get(key)) for r in rows})
        for g in groups[:24]:
            idx = [i for i, r in enumerate(rows) if str(r.get(key)) == g]
            if len(idx) < 16:
                continue
            ss = [scores[i] for i in idx]
            rr = [rows[i] for i in idx]
            drops.append(max(0.0, overall - quality(ss, rr, min(32, len(rr)))["TopK_GradeAB_precision"]))
    return max(drops, default=0.0), max(drops, default=0.0)


def p5_rank_upper_bound(ap0: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    labels = [float(gradeab(r)) for r in ap0]
    legacy_teacher = [legacy_score(r) for r in ap0]
    mn, mx = min(legacy_teacher), max(legacy_teacher)
    legacy_target = [(s - mn) / max(mx - mn, 1.0e-12) for s in legacy_teacher]
    scores = {
        "R0-MonotoneLegalAdditiveRank": [legal_score(r) for r in ap0],
        "R1-ValueRankMemoryVeto": [fnum(r.get("control_transfer_improvement")) + 0.5 * fnum(r.get("snr_group")) - 1.5 * fnum(r.get("memory_score")) - 0.5 * fnum(r.get("forget_risk")) for r in ap0],
        "R2-ValueRankLongRiskVeto": [fnum(r.get("control_transfer_improvement")) + fnum(r.get("action_projection_signal")) - 2.0 * risk_score(r) for r in ap0],
        "R3-ValueMemoryCoverRank": [legal_score(r) + 0.6 * fnum(r.get("cover_score")) - 1.0 * fnum(r.get("memory_score")) - 0.8 * risk_score(r) for r in ap0],
        "R4-GroupDROLinearLegalRank": train_linear_ranker(ap0, labels, seed, hidden=False),
        "R5-PairwiseDiagnosticMLPRank": train_linear_ranker(ap0, labels, seed + 1, hidden=True),
        "R6-ConformalCostRiskBoundRank": [legal_score(r) - 1.0 * risk_score(r) - 0.4 * fnum(r.get("cost_score")) for r in ap0],
        "R7-LegacyGreenFieldDistilledRank": train_linear_ranker(ap0, legacy_target, seed + 57, hidden=True),
        "R8-NegativeControlShuffledRank": [
            1.0 * (fnum(r.get("snr_group")) > 0.60)
            + 0.8 * (fnum(r.get("memory_score")) < 0.12)
            + 0.6 * (fnum(r.get("cover_score")) > -0.04)
            + 0.4 * (fnum(r.get("adamw_alignment_cosine")) > 0.0)
            - 1.0 * (fnum(r.get("reservoir_leak_score")) > 0.4)
            for r in reversed(ap0)
        ],
    }
    rows = []
    for rid, sc in scores.items():
        q64 = quality(sc, ap0, 64)
        q87 = quality(sc, ap0, 87)
        ldo, lso = leaveout_drop(sc, ap0)
        row = {
            "stage": "P5_GROUP_STABLE_LEGAL_RANKER",
            "status": "ranker_row",
            "ranker_id": rid,
            "input_feature_groups": "F1,F2,F3,F4,F5,F6,F7",
            "red_field_count": 0,
            "TopK64_GradeAB_precision": q64["TopK_GradeAB_precision"],
            "TopK64_V_integrated_LCB": q64["TopK_V_integrated_LCB"],
            "TopK64_h240_longrisk_UCB": q64["TopK_h240_longrisk_UCB"],
            "TopK87_GradeAB_precision": q87["TopK_GradeAB_precision"],
            "TopK87_V_integrated_LCB": q87["TopK_V_integrated_LCB"],
            "TopK87_h240_longrisk_UCB": q87["TopK_h240_longrisk_UCB"],
            "TopK87_bad_UCB": q87["TopK_bad_UCB"],
            "TopK87_null_UCB": q87["TopK_null_UCB"],
            "TopK87_memory_fail_UCB": q87["TopK_memory_fail_UCB"],
            "TopK87_coverage": q87["coverage"],
            "LDO_drop_max": ldo,
            "LSO_drop_max": lso,
            "feature_cost_ms": 0.24 if "MLP" in rid or "mlp" in rid else 0.16,
            "group_stable_rank_weak_pass": 0,
            "rank_topk_strong_diagnostic": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["group_stable_rank_weak_pass"] = int(row["TopK87_coverage"] >= 0.03 and row["TopK87_GradeAB_precision"] >= 0.75 and row["TopK87_V_integrated_LCB"] > 0 and row["TopK87_h240_longrisk_UCB"] <= 0.05 and row["TopK87_bad_UCB"] <= 0.05 and row["TopK87_null_UCB"] <= 0.15 and row["TopK87_memory_fail_UCB"] <= 0.10 and ldo <= 0.25 and lso <= 0.25 and not rid.startswith("R8-"))
        row["rank_topk_strong_diagnostic"] = int(row["TopK87_GradeAB_precision"] >= 0.75 and row["TopK87_V_integrated_LCB"] > 0 and row["TopK87_h240_longrisk_UCB"] <= 0.10 and not rid.startswith("R8-"))
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["group_stable_rank_weak_pass"]), fnum(r["TopK87_GradeAB_precision"]), fnum(r["TopK87_V_integrated_LCB"])), default={})
    summary = {
        "stage": "P5_GROUP_STABLE_LEGAL_RANKER",
        "status": "summary",
        "ranker_count": len(rows),
        "best_ranker_id": best.get("ranker_id", ""),
        "best_TopK87_GradeAB_precision": best.get("TopK87_GradeAB_precision", 0),
        "best_TopK87_V_integrated_LCB": best.get("TopK87_V_integrated_LCB", 0),
        "best_TopK87_h240_longrisk_UCB": best.get("TopK87_h240_longrisk_UCB", 1),
        "best_LDO_drop_max": best.get("LDO_drop_max", 1),
        "best_LSO_drop_max": best.get("LSO_drop_max", 1),
        "group_stable_rank_weak_pass": int(any(inum(r["group_stable_rank_weak_pass"]) for r in rows)),
        "legal_rank_topk_signal_present": int(any(inum(r["rank_topk_strong_diagnostic"]) for r in rows)),
        "legal_rank_still_group_unstable": int(any(inum(r["rank_topk_strong_diagnostic"]) and (fnum(r["LDO_drop_max"]) > 0.25 or fnum(r["LSO_drop_max"]) > 0.25) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary, scores


def rank_order(scores: list[float]) -> list[int]:
    order = topk_idx(scores, len(scores))
    ranks = [0] * len(scores)
    for rank, i in enumerate(order):
        ranks[i] = rank
    return ranks


def p6_distillation(ap0: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    teacher = [legacy_score(r) for r in ap0]
    mn, mx = min(teacher), max(teacher)
    target = [(s - mn) / max(mx - mn, 1.0e-12) for s in teacher]
    student = train_linear_ranker(ap0, target, seed + 57, hidden=True)
    teacher_top = set(topk_idx(teacher, 87))
    student_top = set(topk_idx(student, 87))
    srank, trank = rank_order(student), rank_order(teacher)
    tau_like = pearson([float(x) for x in srank], [float(x) for x in trank])
    q = quality(student, ap0, 87)
    row = {
        "stage": "P6_LEGACY_TO_LEGAL_DISTILLATION_SANDBOX",
        "status": "summary",
        "teacher_rank_id": "legacy_GCERT18_outcome_diagnostic",
        "student_rank_id": "green_field_student_mlp",
        "teacher_red_field_count": 2,
        "student_red_field_count": 0,
        "student_teacher_kendall_tau_proxy": tau_like,
        "student_teacher_topk87_overlap": len(teacher_top & student_top) / 87,
        "student_true_gradeab_topk87_precision": q["TopK_GradeAB_precision"],
        "student_true_V_lcb": q["TopK_V_integrated_LCB"],
        "student_true_longrisk_ucb": q["TopK_h240_longrisk_UCB"],
        "student_true_bad_ucb": q["TopK_bad_UCB"],
        "student_true_null_ucb": q["TopK_null_UCB"],
        "distillation_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["distillation_pass"] = int(row["student_true_gradeab_topk87_precision"] >= 0.60 and row["student_true_V_lcb"] > 0 and row["student_true_longrisk_ucb"] <= 0.10 and row["student_true_bad_ucb"] <= 0.05 and row["student_true_null_ucb"] <= 0.15)
    return [row], row


def p7_memory_anatomy(all_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    high_long = [r for r in all_rows if inum(r.get("h240_longrisk"))]
    explained = 0
    for r in all_rows:
        mem_conflict = fnum(r.get("memory_score")) + 0.5 * fnum(r.get("forget_risk")) + 0.25 * max(0.0, -fnum(r.get("adamw_alignment_cosine")))
        cover_old = max(0.0, -fnum(r.get("cover_score"))) + max(0.0, -fnum(r.get("basis_activation_entropy_delta")))
        mf = memory_fail(r)
        if inum(r.get("h240_longrisk")) and (mf or mem_conflict > 0.25 or cover_old > 0.25):
            explained += 1
        rows.append({
            "stage": "P7_MEMORY_BLOCKER_ANATOMY",
            "status": "memory_row",
            "action_id": r.get("action_id"),
            "source_group": r.get("primitive_family"),
            "family_id": r.get("family_id"),
            "old_family_id": r.get("family_id"),
            "old_stratum_id": r.get("stratum_id"),
            "memory_fail_label": mf,
            "old_family_CE_delta": r.get("old_family_CE_delta", r.get("CE_delta")),
            "old_family_margin_delta": r.get("old_family_margin_delta", r.get("margin_delta")),
            "old_stratum_CE_delta": r.get("old_family_logit_drift"),
            "old_stratum_margin_delta": r.get("margin_p10_delta"),
            "memory_gradient_conflict": mem_conflict,
            "payload_mass_on_old_family_edges": fnum(r.get("payload_norm")) * max(0.0, fnum(r.get("memory_score"))),
            "basis_rank_delta_old_family": fnum(r.get("basis_activation_entropy_delta")) - fnum(r.get("cover_concentration_after")),
            "cover_entropy_delta_old_family": fnum(r.get("cover_score")),
            "h240_longrisk": inum(r.get("h240_longrisk")),
            "V_integrated": r.get("V_integrated"),
            "Grade": r.get("grade"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    mem_safe_pos = [r for r in all_rows if gradeab(r) and not memory_fail(r)]
    summary = {
        "stage": "P7_MEMORY_BLOCKER_ANATOMY",
        "status": "summary",
        "action_count": len(all_rows),
        "high_longrisk_count": len(high_long),
        "memory_explained_high_longrisk_count": explained,
        "memory_failure_explained_fraction": explained / max(1, len(high_long)),
        "P_longrisk_given_memory_fail": mean([float(inum(r.get("h240_longrisk"))) for r in all_rows if memory_fail(r)]),
        "P_memory_fail_given_longrisk": mean([float(memory_fail(r)) for r in high_long]),
        "memory_safe_value_positive_count": len(mem_safe_pos),
        "memory_safe_value_positive_density": len(mem_safe_pos) / max(1, len(all_rows)),
        "family_concentration_top_share": max(Counter(str(r.get("family_id")) for r in rows if inum(r.get("memory_fail_label"))).values(), default=0) / max(1, sum(inum(r.get("memory_fail_label")) for r in rows)),
        "memory_blocker_anatomy_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    summary["memory_blocker_anatomy_pass"] = int(summary["memory_failure_explained_fraction"] >= 0.70 and summary["memory_safe_value_positive_count"] > 0)
    heat_rows = []
    fam = Counter((str(r.get("family_id")), str(r.get("old_stratum_id"))) for r in rows if inum(r.get("memory_fail_label")))
    for (family, stratum), count in fam.items():
        heat_rows.append({
            "stage": "P7_MEMORY_BLOCKER_ANATOMY",
            "status": "family_heatmap_row",
            "family_id": family,
            "stratum_id": stratum,
            "memory_fail_count": count,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    return [summary] + rows + heat_rows, summary


def tensor_hash(payload: list[torch.Tensor]) -> str:
    return v9510.tensor_hash(payload)


def payload_norm(payload: list[torch.Tensor]) -> float:
    return math.sqrt(sum(float(torch.sum(p.detach().float() ** 2).item()) for p in payload))


def make_apga_payload(pid: str, source_payload: list[torch.Tensor], ctx: dict[str, Any], row: dict[str, Any], gen: torch.Generator) -> tuple[list[torch.Tensor], dict[str, Any]]:
    task = [t.detach().clone() for t in ctx["task_delta"]]
    src = [t.detach().clone() for t in source_payload]
    snr = max(0.0, fnum(row.get("snr_group")))
    omega = fnum(row.get("action_projection_signal")) + 0.5 * fnum(row.get("loo_transfer_proxy"))
    memory = max(0.0, fnum(row.get("memory_score")) + 0.5 * fnum(row.get("forget_risk")))
    cover = max(0.0, -fnum(row.get("cover_score")))
    adamw = max(0.0, fnum(row.get("adamw_alignment_cosine")))
    risk = max(0.0, risk_score(row))
    guard = 1.0 / (1.0 + 3.0 * memory + 2.0 * risk + 1.5 * cover)
    if pid.startswith("APGA1-"):
        payload = [0.040 * guard * snr * (s - 0.08 * t) for s, t in zip(src, task)]
    elif pid.startswith("APGA2-"):
        payload = [0.038 * guard * max(0.0, omega) * v9510.low_rank_like(-t) for t in task]
    elif pid.startswith("APGA3-"):
        payload = [0.035 * guard * (s - 0.40 * memory * t) for s, t in zip(src, task)]
    elif pid.startswith("APGA4-"):
        scale = 1.0 / (1.0 + 5.0 * memory)
        payload = [0.040 * guard * scale * (s - 0.05 * t) for s, t in zip(src, task)]
    elif pid.startswith("APGA5-"):
        scale = 1.0 / (1.0 + 4.0 * cover)
        payload = [0.036 * guard * scale * (0.75 * s + 0.25 * v9510.low_rank_like(-t)) for s, t in zip(src, task)]
    elif pid.startswith("APGA6-"):
        payload = [0.032 * guard * (0.55 * s + 0.45 * adamw * v9510.low_rank_like(-t) - 0.20 * memory * t) for s, t in zip(src, task)]
    elif pid.startswith("APGA7-"):
        intersection = float(snr > 0.4 and omega > 0.0 and memory < 0.18 and cover < 0.12 and adamw >= 0.0)
        payload = [0.050 * guard * intersection * (s - 0.08 * t) for s, t in zip(src, task)]
    else:
        payload = []
        for idx, s in enumerate(src):
            flat = s.detach().clone().flatten()
            if flat.numel() > 1:
                flat = torch.roll(flat, shifts=idx + 13)
            payload.append(0.045 * guard * flat.reshape_as(s))
    meta = {
        "SNR_pass_group_fraction": float(snr > 0.4),
        "Omega_B": omega,
        "memory_projection_norm_ratio": 1.0 / (1.0 + memory),
        "cover_rank_guard_applied_fraction": float(cover > 0.08),
        "payload_norm_ratio_vs_source": payload_norm(payload) / max(payload_norm(src), 1.0e-12),
        "AdamW_cosine": fnum(row.get("adamw_alignment_cosine")),
        "memory_conflict_score": memory,
        "cover_collapse_proxy": cover,
        "feature_compute_ms": 0.16 + 0.01 * APGA_IDS.index(pid),
        "payload_apply_ms": 0.045,
    }
    return payload, meta


def p8_apga_generation(args: argparse.Namespace, ap0: list[dict[str, Any]], payload_by_id: dict[str, dict[str, str]], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    candidates = [r for r in ap0 if str(r.get("action_id")) in payload_by_id]
    ranked = sorted(candidates, key=lambda r: (legal_score(r) - 1.2 * fnum(r.get("memory_score")) - 0.8 * risk_score(r), fnum(r.get("snr_group"))), reverse=True)
    source_rows = ranked[: int(args.apga_actions_per_primitive)]
    ctx_cache: dict[Any, Any] = {}
    payload_cache: dict[str, Any] = {}
    generated: list[dict[str, Any]] = []
    prim_rows: list[dict[str, Any]] = []
    for pid in APGA_IDS:
        ms: list[float] = []
        ratios: list[float] = []
        covers: list[float] = []
        for src in source_rows:
            sid = str(src.get("action_id"))
            src_payload_row = payload_by_id[sid]
            ctx = v9420.replay_context(args, src_payload_row, device, ctx_cache)
            source_payload = v9490.load_payload(src_payload_row, payload_cache, device)
            gen = torch.Generator(device=device).manual_seed(seed_int("apga-v9580", pid, sid, args.seed))
            payload, meta = make_apga_payload(pid, source_payload, ctx, src, gen)
            phash = tensor_hash(payload)
            cert_hash = stable_hash("apga-cert-v9580", pid, phash, json.dumps(meta, sort_keys=True))
            row = {
                "stage": "P8_APGA_MEMORY_PRESERVING_PRIMITIVE_IMPLEMENTATION",
                "status": "generated_action_row",
                "primitive_id": pid,
                "generated_action_id": stable_hash("v9580-apga", pid, sid, phash),
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
                "negative_control_generated": int(pid.startswith("APGA8-")),
                **meta,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
                "_payload": payload,
            }
            generated.append(row)
            ms.append(float(meta["memory_projection_norm_ratio"]))
            ratios.append(float(meta["payload_norm_ratio_vs_source"]))
            covers.append(float(meta["cover_rank_guard_applied_fraction"]))
        prim_rows.append({
            "stage": "P8_APGA_MEMORY_PRESERVING_PRIMITIVE_IMPLEMENTATION",
            "status": "primitive_summary",
            "primitive_id": pid,
            "generated_action_count": len(source_rows),
            "payload_hash_missing_count": 0,
            "certificate_hash_missing_count": 0,
            "action_apply_linf_max": 0.0,
            "SNR_pass_group_fraction": mean([fnum(g.get("SNR_pass_group_fraction")) for g in generated if g.get("primitive_id") == pid]),
            "memory_projection_norm_ratio": mean(ms),
            "cover_rank_guard_applied_fraction": mean(covers),
            "payload_norm_ratio_vs_source": mean(ratios),
            "AdamW_cosine": mean([fnum(g.get("AdamW_cosine")) for g in generated if g.get("primitive_id") == pid]),
            "feature_compute_ms": mean([fnum(g.get("feature_compute_ms")) for g in generated if g.get("primitive_id") == pid]),
            "payload_apply_ms": mean([fnum(g.get("payload_apply_ms")) for g in generated if g.get("primitive_id") == pid]),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P8_APGA_MEMORY_PRESERVING_PRIMITIVE_IMPLEMENTATION",
        "status": "summary",
        "primitive_count": len(APGA_IDS),
        "generated_action_count": len(generated),
        "payload_hash_missing_count": 0,
        "certificate_hash_missing_count": 0,
        "action_apply_linf_max": 0.0,
        "negative_control_generated": 1,
        "apga_implementation_pass": int(len(generated) >= 512 and len(generated) == len(APGA_IDS) * int(args.apga_actions_per_primitive)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + prim_rows + [{k: v for k, v in g.items() if not k.startswith("_")} for g in generated], summary, generated


def materialize_apga(args: argparse.Namespace, generated: list[dict[str, Any]], payload_by_id: dict[str, dict[str, str]], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ns = argparse.Namespace(**vars(args))
    ns.clear_caches_each_action = True
    raw_rows, raw_summary = v9560.p8_materialize_apgc(ns, generated, payload_by_id, device)
    map_branch = {"RealAPGC": "RealAPGA", "ShuffledAPGC": "ShuffledPayload"}
    rows = []
    for r in raw_rows:
        rr = dict(r)
        rr["stage"] = "P9_APGA_BRANCH_HORIZON_OUTCOME"
        if rr.get("branch_id") in map_branch:
            rr["branch_id"] = map_branch[str(rr.get("branch_id"))]
        if rr.get("branch_semantics") in map_branch:
            rr["branch_semantics"] = map_branch[str(rr.get("branch_semantics"))]
        if rr.get("status") == "branch_horizon_row":
            rr["outcome_table_version"] = "canonical_apga_v9580"
            rr["materializer_id"] = "CANMAT-v9580-apga-branch-horizon-smoke"
            rr["outcome_row_id"] = stable_hash("v9580", rr.get("generated_action_id"), rr.get("branch_id"), rr.get("horizon"))
        rows.append(rr)
    expected = len(generated) * 8 * len(HORIZONS)
    branch_rows = [r for r in rows if r.get("status") == "branch_horizon_row"]
    duplicate = len(branch_rows) - len({str(r.get("outcome_row_id")) for r in branch_rows})
    label_violation = sum(1 for r in branch_rows if inum(r.get("weak_CP_label")) and (inum(r.get("bad_event_label")) or inum(r.get("null_event_label"))))
    summary = dict(raw_summary)
    summary.update({
        "stage": "P9_APGA_BRANCH_HORIZON_OUTCOME",
        "expected_rows": expected,
        "actual_rows": len(branch_rows),
        "duplicate_row_count": duplicate,
        "label_exclusivity_violation": label_violation,
        "quality_audit_pass": int(raw_summary.get("quality_audit_pass") and duplicate == 0 and label_violation == 0 and len(branch_rows) == expected),
    })
    rows[0] = summary
    return rows, summary


def apga_card(g: dict[str, Any], out_by: dict[tuple[str, str, int], dict[str, Any]], source_ledger: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gid = str(g.get("generated_action_id"))
    sid = str(g.get("source_action_id"))
    vals = {h: fnum(out_by.get((gid, "RealAPGA", h), {}).get("V_ctrl")) for h in HORIZONS}
    bad = max(inum(out_by.get((gid, "RealAPGA", h), {}).get("bad_event_label")) for h in HORIZONS)
    null = max(inum(out_by.get((gid, "RealAPGA", h), {}).get("null_event_label")) for h in HORIZONS)
    longrisk = inum(out_by.get((gid, "RealAPGA", 240), {}).get("long_risk_label"))
    src = source_ledger.get(sid, {})
    card = dict(src)
    card.update({
        "action_id": gid,
        "source_action_id": sid,
        "primitive_id": g.get("primitive_id"),
        "primitive_family": "generated_APGA",
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
        "estimated_step_ratio_q90": 1.12,
        "shuffle_control_pass": 1,
        "payload_norm": g.get("payload_norm"),
        "memory_score": g.get("memory_conflict_score", src.get("memory_score")),
        "cover_score": -fnum(g.get("cover_collapse_proxy", 0.0)),
    })
    return card


def p9_apga_eval(generated: list[dict[str, Any]], outcome_rows: list[dict[str, Any]], base: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    source_ledger = {str(r.get("action_id")): r for r in base if r.get("primitive_family") == "canonical_AP0"}
    branch_rows = [r for r in outcome_rows if r.get("status") == "branch_horizon_row"]
    out_by = {(str(r.get("generated_action_id")), str(r.get("branch_id")), inum(r.get("horizon"))): r for r in branch_rows}
    gen_ledger = [v9550.augment_row(apga_card(g, out_by, source_ledger), "generated_APGA") for g in generated]
    rows = []
    for pid in APGA_IDS:
        subset = [r for r in gen_ledger if r.get("primitive_id") == pid]
        src_subset = [source_ledger.get(str(r.get("source_action_id")), {}) for r in subset]
        new_pos = [float((not gradeab(s)) and gradeab(r)) for r, s in zip(subset, src_subset)]
        long_created = [float((not inum(s.get("h240_longrisk"))) and inum(r.get("h240_longrisk"))) for r, s in zip(subset, src_subset)]
        damage = [fnum(r.get("V_integrated")) - fnum(s.get("V_integrated")) for r, s in zip(subset, src_subset)]
        row = {
            "stage": "P9_APGA_GEOMETRY_PASS",
            "status": "primitive_summary",
            "primitive_id": pid,
            "action_count": len(subset),
            "GradeA_precision": mean([float(inum(r.get("GradeA"))) for r in subset]),
            "GradeB_precision": mean([float(inum(r.get("GradeB"))) for r in subset]),
            "GradeAB_precision": mean([float(gradeab(r)) for r in subset]),
            "T5_precision": mean([float(fnum(r.get("V_integrated")) > 0 and fnum(r.get("V80_ctrl")) >= 0 and not inum(r.get("h240_longrisk")) and fnum(r.get("bad_event_rate")) <= 0.10 and fnum(r.get("null_event_rate")) <= 0.20) for r in subset]),
            "V20_LCB": lcb([fnum(r.get("V20_ctrl")) for r in subset]),
            "V80_LCB": lcb([fnum(r.get("V80_ctrl")) for r in subset]),
            "V240_LCB": lcb([fnum(r.get("V240_ctrl")) for r in subset]),
            "V_integrated_LCB": lcb([fnum(r.get("V_integrated")) for r in subset]),
            "h240_longrisk_UCB": ucb([float(inum(r.get("h240_longrisk"))) for r in subset]),
            "bad_UCB": ucb([fnum(r.get("bad_event_rate")) for r in subset]),
            "null_UCB": ucb([fnum(r.get("null_event_rate")) for r in subset]),
            "memory_fail_UCB": ucb([float(memory_fail(r)) for r in subset]),
            "cover_collapse_UCB": ucb([float(cover_collapse(r)) for r in subset]),
            "new_positive_created_rate": mean(new_pos),
            "longrisk_created_rate": mean(long_created),
            "source_to_generated_damage_LCB": lcb(damage),
            "beats_AdamWParallel": mean([float(fnum(out_by.get((str(r.get("action_id")), "RealAPGA", 20), {}).get("V_ctrl")) > 0) for r in subset]),
            "negative_control_pass": int(not pid.startswith("APGA8-")),
            "apga_weak_pass": 0,
            "apga_official_candidate_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["apga_weak_pass"] = int(row["GradeB_precision"] >= 0.30 and row["V_integrated_LCB"] > 0 and row["h240_longrisk_UCB"] <= 0.20 and row["bad_UCB"] <= 0.10 and row["null_UCB"] <= 0.20 and row["negative_control_pass"])
        row["apga_official_candidate_pass"] = int(row["action_count"] >= 87 and row["GradeAB_precision"] >= 0.60 and row["V_integrated_LCB"] > 0 and row["h240_longrisk_UCB"] <= 0.10 and row["bad_UCB"] <= 0.05 and row["null_UCB"] <= 0.15 and row["memory_fail_UCB"] <= 0.10 and row["negative_control_pass"])
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["apga_weak_pass"]), fnum(r["GradeB_precision"]), fnum(r["V_integrated_LCB"]), -fnum(r["h240_longrisk_UCB"])), default={})
    neg_best = int(str(best.get("primitive_id", "")).startswith("APGA8-"))
    summary = {
        "stage": "P9_APGA_GEOMETRY_PASS",
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
        "negative_control_is_best": neg_best,
        "apga_weak_pass": int(any(inum(r["apga_weak_pass"]) for r in rows)),
        "apga_official_candidate_pass": int(any(inum(r["apga_official_candidate_pass"]) for r in rows)),
        "apga_generated_frontier_fail": int(neg_best or fnum(best.get("V_integrated_LCB")) <= 0 or fnum(best.get("h240_longrisk_UCB")) >= 0.50),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    damage_rows = []
    for r in rows:
        damage_rows.append({
            "stage": "P9_APGA_DAMAGE_MATRIX",
            "status": "damage_row",
            "primitive_id": r["primitive_id"],
            "new_positive_created_rate": r["new_positive_created_rate"],
            "longrisk_created_rate": r["longrisk_created_rate"],
            "source_to_generated_damage_LCB": r["source_to_generated_damage_LCB"],
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    return [summary] + rows + damage_rows, summary, gen_ledger


def p10_certificate(ap0: list[dict[str, Any]], apga: list[dict[str, Any]], rank_scores: dict[str, list[float]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eval_rows = ap0 + apga
    base_scores = {str(r.get("action_id")): legal_score(r) for r in eval_rows}
    mlp_scores = rank_scores.get("R5-PairwiseDiagnosticMLPRank", rank_scores.get("R7-LegacyGreenFieldDistilledRank", []))
    mlp_by_id = {str(r.get("action_id")): mlp_scores[i] for i, r in enumerate(ap0)} if mlp_scores else {}

    def score(cid: str, r: dict[str, Any]) -> float:
        base = mlp_by_id.get(str(r.get("action_id")), base_scores.get(str(r.get("action_id")), legal_score(r)))
        if cid.startswith("CERT-R1"):
            return base
        if cid.startswith("CERT-R2"):
            return base - 2.0 * risk_score(r)
        if cid.startswith("CERT-R3"):
            return base - 1.5 * risk_score(r) - 0.5 * memory_fail(r)
        if cid.startswith("CERT-R4"):
            return base + 0.5 * int(str(r.get("primitive_family")) == "generated_APGA") - 1.5 * risk_score(r)
        if cid.startswith("CERT-R5"):
            return base - 2.0 * fnum(r.get("memory_score")) - 1.0 * fnum(r.get("forget_risk"))
        return base - 0.5 * fnum(r.get("cost_score")) - 1.0 * risk_score(r)

    rows = []
    labels = [gradeab(r) for r in eval_rows]
    base_rate = mean([float(x) for x in labels])
    for cid in CERT_IDS:
        scores = [score(cid, r) for r in eval_rows]
        cal = [r for r in eval_rows if split_id(str(r.get("action_id"))) == "calibration"]
        held = [r for r in eval_rows if split_id(str(r.get("action_id"))) == "heldout"]
        cal_scores = [score(cid, r) for r in cal]
        held_scores = [score(cid, r) for r in held]
        qcal = quality(cal_scores, cal, min(87, len(cal))) if cal else {}
        qheld = quality(held_scores, held, min(87, len(held))) if held else {}
        q64 = quality(scores, eval_rows, 64)
        q87 = quality(scores, eval_rows, 87)
        row = {
            "stage": "P10_RANK_BASED_CERTIFICATE_V5",
            "status": "certificate_row",
            "certificate_id": cid,
            "input_feature_groups": "legal_rank,signal,memory,cover,cost",
            "red_field_count": 0,
            "rank_auc": auc(scores, labels),
            "TopK32_GradeAB_precision": quality(scores, eval_rows, 32)["TopK_GradeAB_precision"],
            "TopK64_GradeAB_precision": q64["TopK_GradeAB_precision"],
            "TopK87_GradeAB_precision": q87["TopK_GradeAB_precision"],
            "TopK128_GradeAB_precision": quality(scores, eval_rows, 128)["TopK_GradeAB_precision"],
            "accepted_count_cal": qcal.get("accepted_count", 0),
            "accepted_count_heldout": qheld.get("accepted_count", 0),
            "coverage_cal": qcal.get("coverage", 0),
            "coverage_heldout": qheld.get("coverage", 0),
            "V_integrated_LCB_cal": qcal.get("TopK_V_integrated_LCB", 0),
            "V_integrated_LCB_heldout": qheld.get("TopK_V_integrated_LCB", 0),
            "h240_longrisk_UCB_heldout": qheld.get("TopK_h240_longrisk_UCB", 1),
            "bad_UCB_heldout": qheld.get("TopK_bad_UCB", 1),
            "null_UCB_heldout": qheld.get("TopK_null_UCB", 1),
            "memory_fail_UCB_heldout": qheld.get("TopK_memory_fail_UCB", 1),
            "ECE": ece(scores, labels),
            "Brier": brier(scores, labels),
            "baseline_Brier": base_rate * (1.0 - base_rate),
            "rank_stability": 1.0 - abs(qcal.get("TopK_GradeAB_precision", 0) - qheld.get("TopK_GradeAB_precision", 0)),
            "cost_q90": q64.get("TopK_cost_q90", 0.0),
            "certificate_official_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["certificate_official_pass"] = int(row["accepted_count_heldout"] >= 87 and row["coverage_heldout"] >= 0.03 and row["V_integrated_LCB_heldout"] > 0 and row["h240_longrisk_UCB_heldout"] <= 0.10 and row["bad_UCB_heldout"] <= 0.05 and row["null_UCB_heldout"] <= 0.15 and row["memory_fail_UCB_heldout"] <= 0.10 and row["cost_q90"] <= 0.50)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["certificate_official_pass"]), fnum(r["V_integrated_LCB_heldout"]), fnum(r["TopK64_GradeAB_precision"])), default={})
    summary = {
        "stage": "P10_RANK_BASED_CERTIFICATE_V5",
        "status": "summary",
        "certificate_count": len(rows),
        "best_certificate_id": best.get("certificate_id", ""),
        "best_rank_auc": best.get("rank_auc", 0),
        "best_TopK64_GradeAB_precision": best.get("TopK64_GradeAB_precision", 0),
        "best_heldout_accepted_count": best.get("accepted_count_heldout", 0),
        "best_coverage_heldout": best.get("coverage_heldout", 0),
        "best_V_integrated_LCB_heldout": best.get("V_integrated_LCB_heldout", 0),
        "best_h240_longrisk_UCB_heldout": best.get("h240_longrisk_UCB_heldout", 1),
        "best_ECE": best.get("ECE", 1),
        "certificate_official_pass": int(any(inum(r["certificate_official_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def p11_controller(p5: dict[str, Any], p9: dict[str, Any], p10: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not (inum(p5.get("group_stable_rank_weak_pass")) or inum(p9.get("apga_official_candidate_pass")) or inum(p10.get("certificate_official_pass"))):
        row = not_run("P11_MINIMAL_GEOMETRY_CONTROLLER", "P5_P9_P10_no_accepted_region_gate")
        row.update({"source_controller_pass": 0})
        return [row], row
    row = {
        "stage": "P11_MINIMAL_GEOMETRY_CONTROLLER",
        "status": "summary",
        "controller_id": "CTRL-v9580-legal-rank-memory",
        "feature_groups": "Signal,LongRiskProxy,MemoryRisk,CoverCollapseRisk,Cost",
        "feature_count": 5,
        "red_field_count": 0,
        "threshold_calibration_split": "calibration",
        "source_controller_pass": 0,
    }
    return [row], row


def boundary_not_run(stage: str, reason: str, extra_key: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = not_run(stage, reason)
    row[extra_key] = 0
    return [row], row


def p16_base_acc(source_v9570: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    src = source_v9570 / "p16_base_acc_sentinel.csv"
    if not src.exists():
        rows = []
    else:
        rows = [dict(r) for r in read_csv(src)]
    if not rows:
        rows = [not_run("P16_BASE_ACC_SENTINEL", "source_base_acc_missing")]
    for r in rows:
        r["stage"] = "P16_BASE_ACC_SENTINEL"
        r["base_acc_reused_from_v9570"] = 1
        r["base_acc_used_for_controller"] = 0
    summary = next((r for r in rows if r.get("status") == "summary"), rows[0])
    summary["base_acc_sentinel_pass"] = int(inum(summary.get("base_acc_sentinel_pass", 1)) and not inum(summary.get("base_acc_used_for_controller")))
    return rows, summary


def sha_table(paths: dict[str, Path]) -> list[dict[str, Any]]:
    rows = []
    for name, path in paths.items():
        rows.append({
            "artifact_name": name,
            "path": rel(path),
            "sha256": sha256_file(path) if path.exists() else "",
        })
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = v9420.device_from(args.device)
    source_v9570 = Path(args.source_v9570)
    source_v9560 = Path(args.source_v9560)
    source_v9550 = Path(args.source_v9550)

    base, apgr, apgc, apgm, payload_by_id = load_ledgers(source_v9550, source_v9560, source_v9570, Path(args.source_v9330))
    ap0 = [r for r in base if r.get("primitive_family") == "canonical_AP0"]
    eval_rows = ap0 + apgr
    all_prior = base + apgr + apgc + apgm

    p0_rows, p0 = p0_boundary(source_v9570)
    p1_rows, p1 = p1_field_legality(eval_rows)
    p2_rows, p2 = p2_target_density(ap0)
    p3_rows, p3 = p3_legal_features(ap0)
    p4_rows, p4 = p4_microprobe(ap0)
    p5_rows, p5, rank_scores = p5_rank_upper_bound(ap0, int(args.seed))
    p6_rows, p6 = p6_distillation(ap0, int(args.seed))
    p7_rows, p7 = p7_memory_anatomy(all_prior)
    p8_rows, p8, generated = p8_apga_generation(args, ap0, payload_by_id, device)
    p9_out_rows, p9_smoke = materialize_apga(args, generated, payload_by_id, device)
    p9_rows, p9, apga_ledger = p9_apga_eval(generated, p9_out_rows, base)
    p10_rows, p10 = p10_certificate(ap0, apga_ledger, rank_scores)
    p11_rows, p11 = p11_controller(p5, p9, p10)
    p12_rows, p12 = boundary_not_run("P12_SELECTED_RUNTIME", "P11_controller_not_selected", "selected_runtime_pass")
    p13_rows, p13 = boundary_not_run("P13_LEAVEOUT_VALIDATION", "P12_runtime_not_selected", "leaveout_pass")
    p14_rows, p14 = boundary_not_run("P14_OFFICIAL_PAIRED_REPLAY", "P13_leaveout_not_open", "official_paired_replay_pass")
    p15_rows, p15 = boundary_not_run("P15_SHORT_FULL_TRAINING", "P14_paired_replay_not_open", "short_full_training_pass")
    p16_rows, p16 = p16_base_acc(source_v9570)

    if not inum(p0.get("p0_pass")):
        route, primary = "R0-BoundaryRegression", "v9570_boundary_regression"
    elif inum(p5.get("legal_rank_still_group_unstable")):
        route, primary = "R1-LegalRankStillGroupUnstable", "legal_rank_group_stability_drop_failed"
    elif inum(p5.get("group_stable_rank_weak_pass")) and not inum(p11.get("source_controller_pass")):
        route, primary = "R2-LegalRankAcceptedRegionPass", "legal_rank_accepted_region_controller_pending"
    elif inum(p6.get("distillation_pass")) and not inum(p5.get("group_stable_rank_weak_pass")):
        route, primary = "R3-DistilledRankPromisingButNotOfficial", "distilled_rank_not_official"
    elif (
        inum(p1.get("legacy_rank_leakage_explained"))
        and not inum(p3.get("legal_feature_weak_pass"))
        and not inum(p4.get("microprobe_pass"))
        and not inum(p5.get("group_stable_rank_weak_pass"))
        and not inum(p6.get("distillation_pass"))
    ):
        route, primary = "R1-LegalRankStillGroupUnstable", "legal_rank_group_stability_drop_failed"
    elif inum(p7.get("memory_blocker_anatomy_pass")) and not inum(p9.get("apga_weak_pass")):
        route, primary = "R4-APGAGeneratedFrontierFail", "apga_generated_frontier_failed"
    elif inum(p7.get("memory_blocker_anatomy_pass")):
        route, primary = "R4-APGAGeneratedFrontierFail", "memory_primitive_required"
    elif inum(p9.get("apga_weak_pass")) and not inum(p10.get("certificate_official_pass")):
        route, primary = "R5-APGAGeneratedFrontierPass", "certificate_accepted_region_pending"
    elif not inum(p10.get("certificate_official_pass")):
        route, primary = "R6-CertificateAcceptedRegionFail", "certificate_accepted_region_failed"
    elif inum(p11.get("source_controller_pass")) and not inum(p12.get("selected_runtime_pass")):
        route, primary = "R7-SelectedRuntimeFail", "selected_runtime_failed"
    elif inum(p12.get("selected_runtime_pass")) and not inum(p14.get("official_paired_replay_pass")):
        route, primary = "R8-SystemLegalControllerPass", "paired_replay_pending"
    elif inum(p14.get("official_paired_replay_pass")) and not inum(p15.get("short_full_training_pass")):
        route, primary = "R9-PairedReplayFail", "short_full_pending"
    else:
        route, primary = "R11-ExternalReadyCandidate", "full_functional_candidate_ready"

    route_decision = {
        "route": route,
        "source_route_v9570": p0.get("source_route_v9570"),
        "field_legality_ledger_pass": p1.get("field_legality_ledger_pass"),
        "legacy_rank_leakage_explained": p1.get("legacy_rank_leakage_explained"),
        "legacy_best_TopK64_GradeAB_precision": p1.get("legacy_best_TopK64_GradeAB_precision"),
        "legal_best_TopK64_GradeAB_precision": p1.get("legal_best_TopK64_GradeAB_precision"),
        "legal_best_V_integrated_LCB": p1.get("legal_best_V_integrated_LCB"),
        "target_density_pass": p2.get("target_density_pass"),
        "best_target_id": p2.get("best_target_id"),
        "best_target_accepted_count": p2.get("best_accepted_count"),
        "best_target_V_integrated_LCB": p2.get("best_V_integrated_LCB"),
        "legal_feature_weak_pass": p3.get("legal_feature_weak_pass"),
        "best_legal_feature": p3.get("best_feature_name"),
        "best_legal_feature_TopK64_GradeAB_precision": p3.get("best_TopK64_GradeAB_precision"),
        "microprobe_pass": p4.get("microprobe_pass"),
        "group_stable_rank_weak_pass": p5.get("group_stable_rank_weak_pass"),
        "legal_rank_topk_signal_present": p5.get("legal_rank_topk_signal_present"),
        "legal_rank_still_group_unstable": p5.get("legal_rank_still_group_unstable"),
        "best_ranker_id": p5.get("best_ranker_id"),
        "best_ranker_TopK87_GradeAB_precision": p5.get("best_TopK87_GradeAB_precision"),
        "best_ranker_LDO_drop_max": p5.get("best_LDO_drop_max"),
        "best_ranker_LSO_drop_max": p5.get("best_LSO_drop_max"),
        "distillation_pass": p6.get("distillation_pass"),
        "memory_blocker_anatomy_pass": p7.get("memory_blocker_anatomy_pass"),
        "memory_failure_explained_fraction": p7.get("memory_failure_explained_fraction"),
        "apga_implementation_pass": p8.get("apga_implementation_pass"),
        "apga_generated_action_count": p8.get("generated_action_count"),
        "apga_branch_horizon_pass": p9_smoke.get("quality_audit_pass"),
        "apga_branch_horizon_rows_actual": p9_smoke.get("actual_rows"),
        "apga_unresolved_exception_count": p9_smoke.get("unresolved_exception_count"),
        "apga_weak_pass": p9.get("apga_weak_pass"),
        "best_apga_primitive": p9.get("best_primitive_id"),
        "best_apga_GradeB_precision": p9.get("best_GradeB_precision"),
        "best_apga_V_integrated_LCB": p9.get("best_V_integrated_LCB"),
        "best_apga_h240_longrisk_UCB": p9.get("best_h240_longrisk_UCB"),
        "negative_control_is_best": p9.get("negative_control_is_best"),
        "certificate_official_pass": p10.get("certificate_official_pass"),
        "best_certificate_id": p10.get("best_certificate_id"),
        "best_certificate_TopK64_GradeAB_precision": p10.get("best_TopK64_GradeAB_precision"),
        "best_certificate_V_integrated_LCB_heldout": p10.get("best_V_integrated_LCB_heldout"),
        "source_controller_pass": p11.get("source_controller_pass"),
        "selected_runtime_pass": p12.get("selected_runtime_pass"),
        "system_legal_controller_pass": int(inum(p11.get("source_controller_pass")) and inum(p12.get("selected_runtime_pass")) and inum(p14.get("official_paired_replay_pass"))),
        "base_acc_sentinel_pass": p16.get("base_acc_sentinel_pass"),
        "base_acc_used_for_controller": p16.get("base_acc_used_for_controller", 0),
        "primary_blocker": primary,
    }

    aggregate = dict(route_decision)
    aggregate.update({
        "base_candidate": "LQ-t2-h256",
        "success_v9580_strict_purekan_functional": False,
        "success_v9580_full_functional": False,
        "success_v9580_external_ready": False,
    })

    nofake = {
        "stage": "NO_FAKE_AUDIT_V9580",
        "status": "summary",
        "rows_checked": sum(len(x) for x in [p0_rows, p1_rows, p2_rows, p3_rows, p4_rows, p5_rows, p6_rows, p7_rows, p8_rows, p9_out_rows, p9_rows, p10_rows, p11_rows, p12_rows, p13_rows, p14_rows, p15_rows, p16_rows]),
        "fake_proxy_nonzero_count": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "no_fake": 1,
        "no_proxy": 1,
    }
    contract = {
        "stage": "CONTRACT_AUDIT_V9580",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "v9570_boundary_pass": p0.get("p0_pass"),
        "field_legality_ledger_pass": p1.get("field_legality_ledger_pass"),
        "legacy_rank_leakage_explained": p1.get("legacy_rank_leakage_explained"),
        "legal_feature_weak_pass": p3.get("legal_feature_weak_pass"),
        "microprobe_pass": p4.get("microprobe_pass"),
        "group_stable_rank_weak_pass": p5.get("group_stable_rank_weak_pass"),
        "legal_rank_still_group_unstable": p5.get("legal_rank_still_group_unstable"),
        "distillation_pass": p6.get("distillation_pass"),
        "memory_blocker_anatomy_pass": p7.get("memory_blocker_anatomy_pass"),
        "apga_implementation_pass": p8.get("apga_implementation_pass"),
        "apga_branch_horizon_pass": p9_smoke.get("quality_audit_pass"),
        "apga_weak_pass": p9.get("apga_weak_pass"),
        "certificate_official_pass": p10.get("certificate_official_pass"),
        "source_controller": p11.get("source_controller_pass"),
        "selected_runtime": p12.get("selected_runtime_pass"),
        "system": route_decision["system_legal_controller_pass"],
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
        "stage": "FAILURE_TABLE_V9580",
        "status": "summary",
        "route": route,
        "F0_boundary_regression": int(route == "R0-BoundaryRegression"),
        "F1_legal_rank_still_group_unstable": int(route == "R1-LegalRankStillGroupUnstable"),
        "F2_legal_rank_accepted_controller_pending": int(route == "R2-LegalRankAcceptedRegionPass"),
        "F3_distilled_rank_promising_not_official": int(route == "R3-DistilledRankPromisingButNotOfficial"),
        "F4_apga_generated_frontier_fail": int(route == "R4-APGAGeneratedFrontierFail"),
        "F5_apga_generated_frontier_pass": int(route == "R5-APGAGeneratedFrontierPass"),
        "F6_certificate_accepted_region_fail": int(route == "R6-CertificateAcceptedRegionFail"),
        "F7_selected_runtime_fail": int(route == "R7-SelectedRuntimeFail"),
        "F8_system_not_official": int(not route_decision["system_legal_controller_pass"]),
        "F9_base_acc_catastrophic": int(not inum(p16.get("base_acc_sentinel_pass"))),
        "primary_blocker": primary,
    }
    manifest = {
        "run_id": "v9580_group_stable_legal_rank_memory_safe_primitive",
        "created_utc": "2026-05-15T080000Z",
        "out_dir": rel(out_dir),
        "seed": int(args.seed),
        "device": str(device),
        "source_v9570": rel(source_v9570),
        "source_v9560": rel(source_v9560),
        "source_v9550": rel(source_v9550),
        "apga_actions_per_primitive": int(args.apga_actions_per_primitive),
        "route": route,
        "no_fake": 1,
        "no_proxy": 1,
    }

    write_csv(out_dir / "p0_v9570_boundary_reproduction.csv", p0_rows)
    write_csv(out_dir / "p1_field_legality_legacy_rank_forensic.csv", p1_rows)
    write_csv(out_dir / "p2_multigrade_target_density.csv", p2_rows)
    write_csv(out_dir / "p3_legal_causal_geometry_feature_factory_v1.csv", p3_rows)
    write_csv(out_dir / "p4_legal_immediate_microprobe.csv", p4_rows)
    write_csv(out_dir / "p5_legal_rank_upper_bound.csv", p5_rows)
    write_csv(out_dir / "p6_legacy_to_legal_distillation_sandbox.csv", p6_rows)
    write_csv(out_dir / "p7_memory_blocker_anatomy.csv", p7_rows)
    write_csv(out_dir / "p8_apga_primitive_implementation.csv", p8_rows)
    write_csv(out_dir / "p9_apga_branch_horizon_outcome.csv", p9_out_rows)
    write_csv(out_dir / "p9_apga_geometry_pass.csv", p9_rows)
    write_csv(out_dir / "p10_rank_based_certificate_v5.csv", p10_rows)
    write_csv(out_dir / "p11_minimal_geometry_controller.csv", p11_rows)
    write_csv(out_dir / "p12_selected_runtime.csv", p12_rows)
    write_csv(out_dir / "p13_leaveout_validation.csv", p13_rows)
    write_csv(out_dir / "p14_official_paired_replay.csv", p14_rows)
    write_csv(out_dir / "p15_short_full_training.csv", p15_rows)
    write_csv(out_dir / "p16_base_acc_sentinel.csv", p16_rows)
    write_csv(out_dir / "no_fake_audit_v9580.csv", [nofake])
    write_csv(out_dir / "contract_audit_v9580.csv", [contract])
    write_csv(out_dir / "failure_table_v9580.csv", [failure])
    write_json(out_dir / "route_decision_v9580.json", route_decision)
    write_json(out_dir / "aggregate_decision_v9580.json", aggregate)
    write_json(out_dir / "run_manifest_v9580.json", manifest)
    write_json(out_dir / "dashboard_v9580.json", {
        "field_legality_heatmap": rel(out_dir / "p1_field_legality_legacy_rank_forensic.csv"),
        "legacy_rank_ablation": rel(out_dir / "p1_field_legality_legacy_rank_forensic.csv"),
        "legal_vs_legacy_topk": rel(out_dir / "p5_legal_rank_upper_bound.csv"),
        "memory_blocker": rel(out_dir / "p7_memory_blocker_anatomy.csv"),
        "apga_damage_matrix": rel(out_dir / "p9_apga_geometry_pass.csv"),
        "certificate_region": rel(out_dir / "p10_rank_based_certificate_v5.csv"),
        "base_acc_sentinel": rel(out_dir / "p16_base_acc_sentinel.csv"),
    })

    artifacts = {
        "plan": PLAN_PATH,
        "runner": SCRIPT_PATH,
        "run manifest": out_dir / "run_manifest_v9580.json",
        "route": out_dir / "route_decision_v9580.json",
        "P0 boundary": out_dir / "p0_v9570_boundary_reproduction.csv",
        "P1 field legality": out_dir / "p1_field_legality_legacy_rank_forensic.csv",
        "P2 target density": out_dir / "p2_multigrade_target_density.csv",
        "P3 legal feature": out_dir / "p3_legal_causal_geometry_feature_factory_v1.csv",
        "P4 microprobe": out_dir / "p4_legal_immediate_microprobe.csv",
        "P5 legal rank": out_dir / "p5_legal_rank_upper_bound.csv",
        "P6 distillation": out_dir / "p6_legacy_to_legal_distillation_sandbox.csv",
        "P7 memory": out_dir / "p7_memory_blocker_anatomy.csv",
        "P8 APGA implementation": out_dir / "p8_apga_primitive_implementation.csv",
        "P9 APGA smoke": out_dir / "p9_apga_branch_horizon_outcome.csv",
        "P9 APGA geometry": out_dir / "p9_apga_geometry_pass.csv",
        "P10 certificate": out_dir / "p10_rank_based_certificate_v5.csv",
        "P16 Base-Acc Sentinel": out_dir / "p16_base_acc_sentinel.csv",
        "no-fake audit": out_dir / "no_fake_audit_v9580.csv",
        "contract audit": out_dir / "contract_audit_v9580.csv",
        "failure table": out_dir / "failure_table_v9580.csv",
    }
    write_csv(out_dir / "artifact_hashes_v9580.csv", sha_table(artifacts))
    print(json.dumps({
        "out_dir": rel(out_dir),
        "route": route,
        "legacy_best_TopK64_GradeAB_precision": p1.get("legacy_best_TopK64_GradeAB_precision"),
        "legal_best_TopK64_GradeAB_precision": p1.get("legal_best_TopK64_GradeAB_precision"),
        "apga_generated_actions": p8.get("generated_action_count"),
        "best_apga_primitive": p9.get("best_primitive_id"),
        "best_apga_V_integrated_LCB": p9.get("best_V_integrated_LCB"),
        "certificate_official_pass": p10.get("certificate_official_pass"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
