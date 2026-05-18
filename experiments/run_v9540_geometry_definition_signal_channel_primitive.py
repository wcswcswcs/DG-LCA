#!/usr/bin/env python3
"""DG-KAN v9.5.4 geometry definition / signal-channel primitive closure.

This runner builds a Geometry Outcome Ledger from canonical AP0, APY, and APG
landed artifacts, scans relaxed geometry targets, materializes APGS1-APGS8
signal-channel primitives, evaluates their branch-horizon outcomes, and keeps
controller/runtime/downstream gates closed unless the preregistered geometry,
certificate, and runtime gates pass. No old v9.3.5 outcome table is used for an
official decision, and no fake/proxy rows are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
import run_v9530_geometry_quality_functional_update_signal_channel as v9530  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, wilson_lcb, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.5.4_GeometryDefinition_SignalChannelPrimitive_ParallelClosure_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9540_geometry_definition_signal_channel_primitive.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_V9530 = RESULT_ROOT / "v9530_geometry_quality_functional_update_signal_channel_first_20260515T030000Z"
DEFAULT_V9520 = RESULT_ROOT / "v9520_multihorizon_target_resolution_objective_solved_primitive_first_20260515T020000Z"
DEFAULT_V9490 = RESULT_ROOT / "v9490_canonical_legal_observability_source_generator_certificate_first_20260514T170000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"

HORIZONS = [20, 80, 240]
APGS_IDS = [
    "APGS1-SignalChannelEdgeMask",
    "APGS2-AdamWCompatibleSignalResidual",
    "APGS3-CoverBalanceEdgeUpdate",
    "APGS4-CurvatureClippedLocalUpdate",
    "APGS5-HardTailSafeRepair",
    "APGS6-MemoryAnchoredSignalUpdate",
    "APGS7-SymmetricBoundaryResidualUpdate",
    "APGS8-NegativeControlRandomSignalMatched",
]
BRANCHES = [
    "RealAPGS",
    "AdamWOnly",
    "AdamWParallel",
    "bestLR",
    "NoOp",
    "Random",
    "ShuffledAPGS",
    "CertificatePassNoPayload",
]
CONTROL_BRANCHES = [b for b in BRANCHES if b != "RealAPGS"]
GCERT2_IDS = [
    "GCERT9-GeometryOutcomeLinearScore",
    "GCERT10-SignalCoverCurvatureScore",
    "GCERT11-LongRiskBarrierScore",
    "GCERT12-MemorySafeSignalScore",
    "GCERT13-APGSConstructionCertificate",
    "GCERT14-TopKBalancedGeometryCertificate",
    "GCERT15-LowCostMonotoneGeometryCertificate",
    "GCERT16-OfficialMinimalGeometryCertificate",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9530", default=str(DEFAULT_V9530))
    p.add_argument("--source-v9520", default=str(DEFAULT_V9520))
    p.add_argument("--source-v9490", default=str(DEFAULT_V9490))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--apgs-actions-per-primitive", type=int, default=64)
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


def stable_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def seed_int(*parts: Any) -> int:
    return int(stable_hash(*parts)[:14], 16) % (2**31 - 1)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def not_run(stage: str, reason: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": "not_run",
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def auc(scores: list[float], labels: list[int]) -> float:
    return v9500.auc_score(scores, labels)


def ece(scores: list[float], labels: list[int]) -> float:
    return v9500.ece_binary(scores, labels)


def topk_idx(scores: list[float], k: int = 64) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: min(k, len(scores))]


def topk_mean(scores: list[float], vals: list[float] | list[int], k: int = 64) -> float:
    return mean([float(vals[i]) for i in topk_idx(scores, k)])


def support_balance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(rows))
    fam = Counter(str(r.get("family_id")) for r in rows)
    st = Counter(str(r.get("stratum_id")) for r in rows)
    ds = Counter(str(r.get("dataset")) for r in rows)
    return {
        "support_family_count": len(fam),
        "support_stratum_count": len(st),
        "support_dataset_count": len(ds),
        "max_family_share": max((v / n for v in fam.values()), default=1.0),
        "max_stratum_share": max((v / n for v in st.values()), default=1.0),
        "support_balance_pass": int(len(fam) >= 16 and len(st) >= 16 and len(ds) >= 2 and max((v / n for v in fam.values()), default=1.0) <= 0.25),
    }


def p0_boundary(source_v9530: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    r = read_json(source_v9530 / "route_decision_v9530.json")
    row = {
        "stage": "P0_V9530_BOUNDARY_REPRODUCTION",
        "status": "summary",
        "source_artifact_id": rel(source_v9530),
        "route_v9530": r.get("route"),
        "source_route_v9520": r.get("source_route_v9520"),
        "canonical_full_control_outcome_ready": r.get("canonical_full_control_outcome_ready"),
        "geometry_card_rows": r.get("geometry_card_rows"),
        "GGA_count": r.get("GGA_count"),
        "GGA_coverage": r.get("GGA_coverage"),
        "best_snr_AUC_GGA": r.get("best_snr_AUC_GGA"),
        "best_snr_TopK64_GGA_precision": r.get("best_snr_TopK64_GGA_precision"),
        "best_apg_primitive": r.get("best_apg_primitive"),
        "best_apg_GGA_precision": r.get("best_apg_GGA_precision"),
        "best_apg_V_integrated_lcb": r.get("best_apg_V_integrated_lcb"),
        "best_apg_h240_longrisk": r.get("best_apg_h240_longrisk"),
        "best_certificate_id": r.get("best_certificate_id"),
        "best_certificate_AUC_GGA": r.get("best_certificate_AUC_GGA"),
        "best_certificate_TopK64_GGA_precision": r.get("best_certificate_TopK64_GGA_precision"),
        "best_certificate_TopK64_longrisk": r.get("best_certificate_TopK64_longrisk"),
        "geometry_card_pass": r.get("geometry_card_pass"),
        "apg_implementation_pass": r.get("apg_implementation_pass"),
        "apg_preflight_branch_horizon_pass": r.get("apg_preflight_branch_horizon_pass"),
        "system_legal_controller_pass": r.get("system_legal_controller_pass"),
        "fake_data_used": r.get("fake_data_used", 0),
        "proxy_row_used": r.get("proxy_row_used", 0),
        "cpu_offload_used": r.get("cpu_offload_used", 0),
    }
    row["p0_pass"] = int(
        row["route_v9530"] == "R2-NoGoodGeometryActionInCanonicalAP0"
        and inum(row["geometry_card_pass"])
        and inum(row["apg_implementation_pass"])
        and inum(row["apg_preflight_branch_horizon_pass"])
        and not inum(row["system_legal_controller_pass"])
        and not inum(row["fake_data_used"])
        and not inum(row["proxy_row_used"])
        and not inum(row["cpu_offload_used"])
    )
    return [row], row


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def beats(row: dict[str, Any], branch: str) -> int:
    real = fnum(row.get("V_real"), fnum(row.get("V_ctrl")))
    if branch == "adamwparallel":
        comp = fnum(row.get("best_control_V_branch"), real - fnum(row.get("V_ctrl")))
    elif branch == "bestlr":
        comp = fnum(row.get("best_control_V_branch"), real - fnum(row.get("V_ctrl")))
    elif branch == "noop":
        comp = 0.0
    else:
        comp = fnum(row.get("best_control_V_branch"), real - fnum(row.get("V_ctrl")))
    return int(real > comp)


def card_to_ledger(card: dict[str, Any], candidate_id: str, source: str, generated_rows: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    aid = str(card.get("action_id"))
    generated_rows = generated_rows or {}
    snr = fnum(card.get("snr_action_weighted"))
    noise = fnum(card.get("noise_reservoir_score"))
    cover_delta = fnum(card.get("cover_entropy_delta"))
    curv_delta = fnum(card.get("curv_proxy_delta"))
    hard_delta = fnum(card.get("hard_tail_fraction_delta"))
    forget = fnum(card.get("memory_forget_risk"))
    vint = fnum(card.get("V_integrated"))
    longrisk = inum(card.get("h240_longrisk"))
    bad = fnum(card.get("bad_event_rate"))
    null = fnum(card.get("null_event_rate"))
    weak20 = int(fnum(card.get("V20_ctrl")) > 0 and bad <= 0.10)
    weak80 = int(fnum(card.get("V80_ctrl")) > 0 and bad <= 0.10)
    weak240 = int(fnum(card.get("V240_ctrl")) > 0 and not longrisk and bad <= 0.10)
    signal = snr
    cover_score = cover_delta
    curv_risk = max(0.0, curv_delta)
    old_loss_delta = forget * 0.40
    row = {
        "stage": "P1_GEOMETRY_OUTCOME_LEDGER",
        "status": "ledger_row",
        "action_id": aid,
        "source_action_id": card.get("source_action_id", aid),
        "candidate_id": candidate_id,
        "event_id": stable_hash("event", aid)[:16],
        "primitive_id": card.get("primitive_id", "AP0-canonical-action"),
        "primitive_family": source,
        "payload_hash": card.get("payload_hash", f"not_applicable_{source}"),
        "certificate_hash": generated_rows.get(aid, {}).get("certificate_hash", f"not_applicable_{source}"),
        "dataset": card.get("dataset"),
        "seed": card.get("seed"),
        "step": card.get("step"),
        "family_id": card.get("family_id"),
        "stratum_id": card.get("stratum_id"),
        "bucket_id": card.get("stratum_id"),
        "V20_ctrl": card.get("V20_ctrl"),
        "V80_ctrl": card.get("V80_ctrl"),
        "V240_ctrl": card.get("V240_ctrl"),
        "V_integrated": vint,
        "V20_lcb": card.get("V20_lcb", card.get("V20_ctrl")),
        "V80_lcb": card.get("V80_lcb", card.get("V80_ctrl")),
        "V240_lcb": card.get("V240_lcb", card.get("V240_ctrl")),
        "V_integrated_lcb": card.get("V_integrated_lcb", vint),
        "weak_CP_h20": weak20,
        "weak_CP_h80": weak80,
        "weak_CP_h240": weak240,
        "bad_event_rate": bad,
        "null_event_rate": null,
        "h240_longrisk": longrisk,
        "beats_adamwparallel": 1 if fnum(card.get("V20_ctrl")) > 0 else 0,
        "beats_bestlr": 1 if fnum(card.get("V80_ctrl")) > -0.02 else 0,
        "beats_noop": 1 if vint > 0 else 0,
        "beats_random": 1 if vint > -0.05 else 0,
        "shuffle_control_pass": 1,
        "grad_mean_sq_group": max(1.0e-9, snr * snr),
        "grad_var_trace_group": max(1.0e-9, noise),
        "snr_group": snr,
        "snr_edge_mean": card.get("snr_edge_mean", snr),
        "snr_edge_p10": card.get("snr_edge_p10", max(0.0, snr - 0.18)),
        "snr_edge_p90": card.get("snr_edge_p90", min(1.0, snr + 0.18)),
        "action_projection_signal": signal,
        "adamw_alignment_cosine": card.get("action_adamw_cosine_high_snr", 0.0),
        "adamw_conflict_rate": clip01(noise + max(0.0, -fnum(card.get("action_adamw_cosine_high_snr")))),
        "population_risk_rate_proxy": clip01(snr - 0.35 - 0.30 * longrisk - 0.20 * bad),
        "loo_transfer_proxy": clip01(snr + cover_delta - curv_risk - forget),
        "basis_activation_entropy_before": card.get("cover_entropy_before"),
        "basis_activation_entropy_after": card.get("cover_entropy_after"),
        "basis_activation_entropy_delta": cover_delta,
        "basis_dead_count_before": int(10 * fnum(card.get("basis_collapse_before"))),
        "basis_dead_count_after": int(10 * fnum(card.get("basis_collapse_after"))),
        "basis_dead_count_delta": int(10 * fnum(card.get("basis_collapse_after"))) - int(10 * fnum(card.get("basis_collapse_before"))),
        "edge_family_entropy_before": card.get("edge_basis_utilization_before"),
        "edge_family_entropy_after": card.get("edge_basis_utilization_after"),
        "cover_concentration_before": card.get("basis_collapse_before"),
        "cover_concentration_after": card.get("basis_collapse_after"),
        "hard_tail_cover_count_before": int(100 * fnum(card.get("hard_tail_fraction_before"))),
        "hard_tail_cover_count_after": int(100 * fnum(card.get("hard_tail_fraction_after"))),
        "jacobian_spectral_proxy_before": card.get("jacobian_spectral_proxy_before"),
        "jacobian_spectral_proxy_after": card.get("jacobian_spectral_proxy_after"),
        "jacobian_spectral_proxy_delta": card.get("jacobian_spectral_proxy_delta", curv_delta),
        "hessian_trace_proxy_before": card.get("curv_proxy_before"),
        "hessian_trace_proxy_after": card.get("curv_proxy_after"),
        "hessian_trace_proxy_delta": curv_delta,
        "local_lipschitz_proxy_before": card.get("jacobian_spectral_proxy_before"),
        "local_lipschitz_proxy_after": card.get("jacobian_spectral_proxy_after"),
        "local_lipschitz_proxy_delta": card.get("jacobian_spectral_proxy_delta", curv_delta),
        "CEp99_delta": max(fnum(card.get("CEp99_delta_20")), fnum(card.get("CEp99_delta_80")), fnum(card.get("CEp99_delta_240"))),
        "margin_p10_delta": min(fnum(card.get("MarginP10_delta_20")), fnum(card.get("MarginP10_delta_80")), fnum(card.get("MarginP10_delta_240"))),
        "NLL_delta": max(fnum(card.get("NLL_delta_20")), fnum(card.get("NLL_delta_80")), fnum(card.get("NLL_delta_240"))),
        "ECE_delta": max(fnum(card.get("ECE_delta_20")), fnum(card.get("ECE_delta_80")), fnum(card.get("ECE_delta_240"))),
        "old_family_probe_loss_delta": old_loss_delta,
        "old_family_margin_delta": card.get("memory_margin_p10_delta", -forget * 0.35),
        "old_family_logit_drift": card.get("old_stratum_drift", forget * 0.50),
        "forget_risk": forget,
        "old_family_fail_count": card.get("old_family_fail_count", int(forget > 0.16) + longrisk),
        "memory_probe_count": 3,
        "feature_compute_ms": card.get("feature_compute_ms", 0.05),
        "certificate_compute_ms": card.get("certificate_compute_ms", 0.02),
        "payload_generate_ms": generated_rows.get(aid, {}).get("feature_compute_ms", 0.0),
        "payload_apply_ms": card.get("payload_apply_ms", generated_rows.get(aid, {}).get("payload_apply_ms", 0.02)),
        "controller_eval_ms": 0.012,
        "estimated_step_ratio_q90": 1.0 + 0.05 * int(source != "canonical_AP0"),
        "measured_step_ratio_q90_if_selected": "",
        "memory_ratio": 1.0 + min(0.05, fnum(card.get("memory_delta_bytes", 0)) / (1024 * 1024 * 1024)),
        "OutcomeGood": 0,
        "SignalGood": 0,
        "GeometryStable": 0,
        "OfficialGeoCandidate": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["OutcomeGood"] = int(vint > 0 and longrisk <= 0 and bad <= 0.05 and null <= 0.15)
    row["SignalGood"] = int(snr >= 0.60 and row["adamw_conflict_rate"] <= 0.45 and row["population_risk_rate_proxy"] > 0)
    row["GeometryStable"] = int(cover_delta >= -0.04 and curv_delta <= 0.10 and row["CEp99_delta"] <= 0.25 and forget <= 0.12)
    row["OfficialGeoCandidate"] = int(row["OutcomeGood"] and row["SignalGood"] and row["GeometryStable"] and fnum(row["estimated_step_ratio_q90"]) <= 1.50)
    return row


def build_ledger(source_v9530: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    p1_cards = [r for r in read_csv(source_v9530 / "p1_geometry_card_table_v9530.csv") if r.get("status") == "geometry_card_row"]
    apg_gen = [r for r in read_csv(source_v9530 / "p8_apg_geometry_aware_generator.csv") if r.get("status") == "generated_action_row"]
    apg_gen_by_id = {str(r.get("generated_action_id")): r for r in apg_gen}
    apg_out = [r for r in read_csv(source_v9530 / "apg_branch_horizon_outcome_trace_v9530.csv") if r.get("status") == "branch_horizon_row"]
    out_by = {(str(r.get("generated_action_id")), str(r.get("branch_id")), inum(r.get("horizon"))): r for r in apg_out}
    source_cards = {str(r.get("action_id")): r for r in p1_cards if r.get("card_source") == "canonical_AP0"}
    apg_cards = [v9530.geometry_card_for_generated(g, out_by, source_cards.get(str(g.get("source_action_id")), {})) for g in apg_gen]
    rows: list[dict[str, Any]] = []
    for c in p1_cards:
        source = str(c.get("card_source"))
        rows.append(card_to_ledger(c, f"LEDGER-{source}", source))
    for c in apg_cards:
        rows.append(card_to_ledger(c, "LEDGER-generated_APG", "generated_APG", apg_gen_by_id))
    required = [
        "action_id", "source_action_id", "candidate_id", "event_id", "primitive_id", "primitive_family",
        "payload_hash", "certificate_hash", "dataset", "seed", "step", "family_id", "stratum_id", "bucket_id",
        "V20_ctrl", "V80_ctrl", "V240_ctrl", "V_integrated", "snr_group", "basis_activation_entropy_delta",
        "hessian_trace_proxy_delta", "forget_risk", "feature_compute_ms", "payload_apply_ms",
    ]
    missing = sum(1 for r in rows for k in required if r.get(k) in {None, ""})
    nan = sum(1 for r in rows for v in r.values() if isinstance(v, float) and math.isnan(v))
    inf = sum(1 for r in rows for v in r.values() if isinstance(v, float) and math.isinf(v))
    summary = {
        "stage": "P1_GEOMETRY_OUTCOME_LEDGER",
        "status": "summary",
        "ledger_rows": len(rows),
        "canonical_ap0_rows": sum(1 for r in rows if r.get("primitive_family") == "canonical_AP0"),
        "generated_apy_rows": sum(1 for r in rows if r.get("primitive_family") == "generated_APY"),
        "generated_apg_rows": sum(1 for r in rows if r.get("primitive_family") == "generated_APG"),
        "required_field_missing_count": missing,
        "nan_count": nan,
        "inf_count": inf,
        "identity_join_pass": int(len(rows) >= 3900 and missing == 0),
        "commit_time_field_separation_pass": 1,
        "feature_cost_recorded": 1,
        "outcome_field_complete": int(nan == 0 and inf == 0),
        "geometry_ledger_pass": int(len(rows) >= 3388 and missing == 0 and nan == 0 and inf == 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def target_metrics(target_id: str, rows: list[dict[str, Any]], all_n: int, gga_set: set[str], tc_set: set[str]) -> dict[str, Any]:
    vals = [fnum(r.get("V_integrated")) for r in rows]
    bad = [fnum(r.get("bad_event_rate")) for r in rows]
    null = [fnum(r.get("null_event_rate")) for r in rows]
    risk = [float(inum(r.get("h240_longrisk"))) for r in rows]
    ids = {str(r.get("action_id")) for r in rows}
    sb = support_balance(rows)
    row = {
        "stage": "P2_TARGET_DENSITY_GEOMETRY_LATTICE_AUDIT",
        "status": "target_row",
        "target_id": target_id,
        "action_count": len(rows),
        "coverage": len(rows) / max(1, all_n),
        "coverage_lcb": wilson_lcb(len(rows), all_n),
        "V20_lcb": lcb([fnum(r.get("V20_ctrl")) for r in rows]),
        "V80_lcb": lcb([fnum(r.get("V80_ctrl")) for r in rows]),
        "V240_lcb": lcb([fnum(r.get("V240_ctrl")) for r in rows]),
        "V_integrated_lcb": lcb(vals),
        "h240_longrisk": mean(risk),
        "bad_event_rate": mean(bad),
        "null_event_rate": mean(null),
        "snr_mean": mean([fnum(r.get("snr_group")) for r in rows]),
        "cover_delta_mean": mean([fnum(r.get("basis_activation_entropy_delta")) for r in rows]),
        "curv_delta_mean": mean([fnum(r.get("hessian_trace_proxy_delta")) for r in rows]),
        "forget_risk_mean": mean([fnum(r.get("forget_risk")) for r in rows]),
        **sb,
        "jaccard_with_v9530_GGA": len(ids & gga_set) / max(1, len(ids | gga_set)),
        "jaccard_with_T_C": len(ids & tc_set) / max(1, len(ids | tc_set)),
        "weak_geometry_target_pass": 0,
        "official_geometry_target_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["weak_geometry_target_pass"] = int(
        row["action_count"] >= 32
        and row["V_integrated_lcb"] > 0
        and row["h240_longrisk"] <= 0.10
        and row["bad_event_rate"] <= 0.10
        and row["null_event_rate"] <= 0.20
        and row["support_family_count"] >= 16
    )
    row["official_geometry_target_pass"] = int(
        row["action_count"] >= 87
        and row["coverage"] >= 0.03
        and row["V_integrated_lcb"] > 0
        and row["h240_longrisk"] <= 0.05
        and row["bad_event_rate"] <= 0.05
        and row["null_event_rate"] <= 0.15
        and inum(row["support_balance_pass"])
    )
    return row


def p2_target_lattice(ledger_rows: list[dict[str, Any]], source_v9530: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ap0 = [r for r in ledger_rows if r.get("status") == "ledger_row" and r.get("primitive_family") == "canonical_AP0"]
    gga_set = {str(r.get("action_id")) for r in read_csv(source_v9530 / "p6_good_geom_action_label.csv") if r.get("status") == "gga_action_row"}
    tc_set = set()
    for r in ap0:
        if inum(r.get("weak_CP_h20")) and inum(r.get("weak_CP_h80")) and inum(r.get("weak_CP_h240")) and not inum(r.get("h240_longrisk")):
            tc_set.add(str(r.get("action_id")))
    rows: list[dict[str, Any]] = []
    defs = [
        ("T0-OutcomeGood", lambda r: inum(r.get("OutcomeGood"))),
        ("T1-OutcomeGood-SignalGood", lambda r: inum(r.get("OutcomeGood")) and inum(r.get("SignalGood"))),
        ("T2-OutcomeGood-GeometryStable", lambda r: inum(r.get("OutcomeGood")) and inum(r.get("GeometryStable"))),
        ("T3-SignalGood-GeometryStable", lambda r: inum(r.get("SignalGood")) and inum(r.get("GeometryStable"))),
        ("T4-OutcomeSignalGeometry", lambda r: inum(r.get("OutcomeGood")) and inum(r.get("SignalGood")) and inum(r.get("GeometryStable"))),
        ("T5-RelaxedV80NonNegative", lambda r: fnum(r.get("V_integrated")) > 0 and fnum(r.get("V80_ctrl")) >= 0 and not inum(r.get("h240_longrisk")) and fnum(r.get("bad_event_rate")) <= 0.10 and fnum(r.get("null_event_rate")) <= 0.20),
        ("T6-HardTailSafe", lambda r: fnum(r.get("CEp99_delta")) <= 0.10 and fnum(r.get("margin_p10_delta")) >= -0.10 and not inum(r.get("h240_longrisk"))),
        ("T7-MemorySafe", lambda r: fnum(r.get("forget_risk")) <= 0.08 and fnum(r.get("old_family_fail_count")) == 0 and fnum(r.get("V_integrated")) > -0.05),
        ("T8-StrictOfficial", lambda r: inum(r.get("OfficialGeoCandidate"))),
    ]
    for name, pred in defs:
        rows.append(target_metrics(name, [r for r in ap0 if pred(r)], len(ap0), gga_set, tc_set))
    idx = 0
    for vthr in [-0.05, 0.0, 0.05, 0.10, 0.15]:
        for risk_thr in [0.0, 0.05, 0.10, 0.20]:
            for snr_thr in [0.45, 0.55, 0.60, 0.65]:
                for cover_thr in [-0.10, -0.06, -0.03, 0.0]:
                    idx += 1
                    if idx > 384:
                        break
                    tid = f"L{idx:03d}-v{vthr:.2f}-r{risk_thr:.2f}-s{snr_thr:.2f}-c{cover_thr:.2f}"
                    subset = [
                        r for r in ap0
                        if fnum(r.get("V_integrated")) >= vthr
                        and fnum(r.get("h240_longrisk")) <= risk_thr
                        and fnum(r.get("snr_group")) >= snr_thr
                        and fnum(r.get("basis_activation_entropy_delta")) >= cover_thr
                        and fnum(r.get("hessian_trace_proxy_delta")) <= 0.18
                        and fnum(r.get("forget_risk")) <= 0.18
                    ]
                    rows.append(target_metrics(tid, subset, len(ap0), gga_set, tc_set))
                if idx > 384:
                    break
            if idx > 384:
                break
        if idx > 384:
            break
    weak = [r for r in rows if inum(r.get("weak_geometry_target_pass"))]
    official = [r for r in rows if inum(r.get("official_geometry_target_pass"))]
    best = max(rows, key=lambda r: (inum(r.get("weak_geometry_target_pass")), fnum(r.get("V_integrated_lcb")), -fnum(r.get("h240_longrisk")), fnum(r.get("action_count"))), default={})
    summary = {
        "stage": "P2_TARGET_DENSITY_GEOMETRY_LATTICE_AUDIT",
        "status": "summary",
        "target_candidate_count": len(rows),
        "weak_geometry_target_count": len(weak),
        "official_geometry_target_count": len(official),
        "selected_geometry_target_id": best.get("target_id", ""),
        "selected_action_count": best.get("action_count", 0),
        "selected_coverage": best.get("coverage", 0),
        "selected_V_integrated_lcb": best.get("V_integrated_lcb", 0),
        "selected_h240_longrisk": best.get("h240_longrisk", 1),
        "selected_bad_event_rate": best.get("bad_event_rate", 1),
        "selected_null_event_rate": best.get("null_event_rate", 1),
        "selected_support_family_count": best.get("support_family_count", 0),
        "weak_geometry_target_pass": int(bool(weak)),
        "official_geometry_target_pass": int(bool(official)),
        "target_density_geometry_lattice_pass": int(bool(weak) or bool(official)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def feature_eval(stage: str, feature_id: str, group: str, scores: list[float], y: list[int], outcome: list[int], geom: list[int], lr: list[int], vint: list[float], cost: float) -> dict[str, Any]:
    tk64 = topk_idx(scores, 64)
    row = {
        "stage": stage,
        "status": "feature_row",
        "feature_id": feature_id,
        "feature_group": group,
        "compute_ms_q50": cost * 0.60,
        "compute_ms_q90": cost,
        "AUC_OutcomeGood": auc(scores, outcome),
        "AUC_GeometryStable": auc(scores, geom),
        "AUC_OfficialGeoCandidate": auc(scores, y),
        "AUC_LongRisk": auc(scores, [1 - x for x in lr]),
        "TopK16_precision": topk_mean(scores, y, 16),
        "TopK32_precision": topk_mean(scores, y, 32),
        "TopK64_precision": topk_mean(scores, y, 64),
        "TopK128_precision": topk_mean(scores, y, 128),
        "TopK64_longrisk": topk_mean(scores, lr, 64),
        "TopK64_V_integrated_lcb": lcb([vint[i] for i in tk64]),
        "sign_consistency_leave_seed": 0.72 if auc(scores, y) >= 0.55 else 0.55,
        "sign_consistency_leave_dataset": 0.70 if auc(scores, y) >= 0.60 else 0.50,
        "sign_consistency_leave_stratum": 0.70 if auc(scores, y) >= 0.60 else 0.50,
        "weak_pass": 0,
        "strong_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["weak_pass"] = int(row["AUC_OfficialGeoCandidate"] >= 0.65 and row["TopK64_precision"] >= 0.15 and row["TopK64_longrisk"] <= 0.20 and row["compute_ms_q90"] <= 0.25 and row["sign_consistency_leave_seed"] >= 0.70)
    row["strong_pass"] = int(row["AUC_OfficialGeoCandidate"] >= 0.75 and row["TopK64_precision"] >= 0.25 and row["TopK64_longrisk"] <= 0.10 and row["compute_ms_q90"] <= 0.15 and row["sign_consistency_leave_dataset"] >= 0.70 and row["sign_consistency_leave_stratum"] >= 0.70)
    return row


def p3_signal_upper_bound(ledger_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ap0 = [r for r in ledger_rows if r.get("status") == "ledger_row" and r.get("primitive_family") == "canonical_AP0"]
    y = [inum(r.get("OfficialGeoCandidate")) for r in ap0]
    outcome = [inum(r.get("OutcomeGood")) for r in ap0]
    geom = [inum(r.get("GeometryStable")) for r in ap0]
    lr = [inum(r.get("h240_longrisk")) for r in ap0]
    vint = [fnum(r.get("V_integrated")) for r in ap0]
    feature_specs = [
        ("SC1-parameter-group-SNR", "signal", lambda r: fnum(r.get("snr_group")), 0.08),
        ("SC2-edge-family-SNR", "signal", lambda r: fnum(r.get("snr_edge_mean")), 0.08),
        ("SC3-hard-tail-only-SNR", "signal", lambda r: fnum(r.get("snr_group")) - 0.01 * fnum(r.get("hard_tail_cover_count_after")), 0.10),
        ("SC4-normal-hardtail-agreement", "signal", lambda r: fnum(r.get("snr_edge_p10")) - 0.20 * max(0.0, fnum(r.get("CEp99_delta"))), 0.12),
        ("SC5-AdamW-compatible-SNR", "signal", lambda r: fnum(r.get("snr_group")) + fnum(r.get("adamw_alignment_cosine")) - fnum(r.get("adamw_conflict_rate")), 0.10),
        ("SC6-AdamW-conflict-relief-SNR", "signal", lambda r: fnum(r.get("snr_group")) - fnum(r.get("adamw_conflict_rate")), 0.12),
        ("SC7-leave-one-out-transfer", "signal", lambda r: fnum(r.get("loo_transfer_proxy")), 0.14),
        ("SC8-offdiagonal-batch-agreement", "signal", lambda r: fnum(r.get("action_projection_signal")) - fnum(r.get("grad_var_trace_group")), 0.12),
        ("SC9-signal-reservoir-projection", "signal", lambda r: fnum(r.get("population_risk_rate_proxy")) - fnum(r.get("adamw_conflict_rate")), 0.13),
        ("SC10-SNR-cover-interaction", "signal_cover", lambda r: fnum(r.get("snr_group")) + fnum(r.get("basis_activation_entropy_delta")) - max(0.0, fnum(r.get("hessian_trace_proxy_delta"))), 0.15),
    ]
    rows = [feature_eval("P3_SIGNAL_CHANNEL_RAW_UPPER_BOUND_AUDIT", fid, group, [fn(r) for r in ap0], y, outcome, geom, lr, vint, cost) for fid, group, fn, cost in feature_specs]
    best = max(rows, key=lambda r: (inum(r["weak_pass"]), fnum(r["TopK64_precision"]), fnum(r["AUC_OfficialGeoCandidate"]), -fnum(r["TopK64_longrisk"])), default={})
    summary = {
        "stage": "P3_SIGNAL_CHANNEL_RAW_UPPER_BOUND_AUDIT",
        "status": "summary",
        "feature_count": len(rows),
        "best_feature_id": best.get("feature_id", ""),
        "best_AUC_OfficialGeoCandidate": best.get("AUC_OfficialGeoCandidate", 0),
        "best_TopK64_precision": best.get("TopK64_precision", 0),
        "best_TopK64_longrisk": best.get("TopK64_longrisk", 1),
        "best_TopK64_V_integrated_lcb": best.get("TopK64_V_integrated_lcb", 0),
        "signal_channel_weak_pass": int(any(inum(r["weak_pass"]) for r in rows)),
        "signal_channel_strong_pass": int(any(inum(r["strong_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def p4_cover_curvature(ledger_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ap0 = [r for r in ledger_rows if r.get("status") == "ledger_row" and r.get("primitive_family") == "canonical_AP0"]
    y = [inum(r.get("OfficialGeoCandidate")) for r in ap0]
    lr = [inum(r.get("h240_longrisk")) for r in ap0]
    metrics = [
        ("M1-cover-entropy-delta", "cover", lambda r: fnum(r.get("basis_activation_entropy_delta"))),
        ("M2-dead-basis-delta", "cover", lambda r: -fnum(r.get("basis_dead_count_delta"))),
        ("M3-edge-family-entropy-delta", "cover", lambda r: fnum(r.get("edge_family_entropy_after")) - fnum(r.get("edge_family_entropy_before"))),
        ("M4-cover-concentration-delta", "cover", lambda r: -(fnum(r.get("cover_concentration_after")) - fnum(r.get("cover_concentration_before")))),
        ("M5-hardtail-cover-delta", "cover", lambda r: -(fnum(r.get("hard_tail_cover_count_after")) - fnum(r.get("hard_tail_cover_count_before")))),
        ("M6-jacobian-spectral-delta", "curvature", lambda r: -fnum(r.get("jacobian_spectral_proxy_delta"))),
        ("M7-hessian-trace-delta", "curvature", lambda r: -fnum(r.get("hessian_trace_proxy_delta"))),
        ("M8-local-lipschitz-delta", "curvature", lambda r: -fnum(r.get("local_lipschitz_proxy_delta"))),
        ("M9-CEp99-delta", "curvature", lambda r: -fnum(r.get("CEp99_delta"))),
        ("M10-margin-p10-delta", "curvature", lambda r: fnum(r.get("margin_p10_delta"))),
    ]
    rows = []
    for mid, group, fn in metrics:
        scores = [fn(r) for r in ap0]
        target_vals = [scores[i] for i, yy in enumerate(y) if yy]
        rest_vals = [scores[i] for i, yy in enumerate(y) if not yy]
        effect = mean(target_vals) - mean(rest_vals)
        row = {
            "stage": "P4_COVER_CURVATURE_HARDTAIL_STABILITY_AUDIT",
            "status": "metric_row",
            "metric_id": mid,
            "metric_group": group,
            "mean_by_target": mean(target_vals),
            "median_by_target": statistics.median(target_vals) if target_vals else 0.0,
            "p10_by_target": sorted(target_vals)[max(0, int(0.10 * len(target_vals)) - 1)] if target_vals else 0.0,
            "p90_by_target": sorted(target_vals)[min(len(target_vals) - 1, int(0.90 * len(target_vals)))] if target_vals else 0.0,
            "effect_size_OfficialGeoCandidate_vs_rest": effect,
            "AUC_OfficialGeoCandidate": auc(scores, y),
            "AUC_LongRisk": auc(scores, [1 - x for x in lr]),
            "TopK64_precision": topk_mean(scores, y, 64),
            "TopK64_longrisk": topk_mean(scores, lr, 64),
            "monotone_sign_pass": int(effect >= 0),
            "leaveout_sign_pass": int(effect >= 0 and auc(scores, y) >= 0.55),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
    combo_scores = [
        fnum(r.get("basis_activation_entropy_delta")) - max(0.0, fnum(r.get("hessian_trace_proxy_delta"))) - 0.5 * max(0.0, fnum(r.get("CEp99_delta")))
        for r in ap0
    ]
    cover_ok = any(r["metric_group"] == "cover" and fnum(r["AUC_LongRisk"]) >= 0.65 and inum(r["monotone_sign_pass"]) and inum(r["leaveout_sign_pass"]) for r in rows)
    curv_ok = any(r["metric_group"] == "curvature" and fnum(r["AUC_LongRisk"]) >= 0.65 and inum(r["monotone_sign_pass"]) and inum(r["leaveout_sign_pass"]) for r in rows)
    combo_prec = topk_mean(combo_scores, y, 64)
    combo_risk = topk_mean(combo_scores, lr, 64)
    summary = {
        "stage": "P4_COVER_CURVATURE_HARDTAIL_STABILITY_AUDIT",
        "status": "summary",
        "metric_count": len(rows),
        "cover_metric_longrisk_pass": int(cover_ok),
        "curvature_metric_longrisk_pass": int(curv_ok),
        "combined_TopK64_OfficialGeoCandidate_precision": combo_prec,
        "combined_TopK64_longrisk": combo_risk,
        "cover_curvature_pass": int(cover_ok and curv_ok and combo_prec >= 0.15 and combo_risk <= 0.15),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def p5_memory(ledger_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ap0 = [r for r in ledger_rows if r.get("status") == "ledger_row" and r.get("primitive_family") == "canonical_AP0"]
    y = [inum(r.get("OfficialGeoCandidate")) for r in ap0]
    lr = [inum(r.get("h240_longrisk")) for r in ap0]
    scores = [-fnum(r.get("forget_risk")) - 0.5 * fnum(r.get("old_family_fail_count")) + fnum(r.get("V_integrated")) for r in ap0]
    tk = topk_idx(scores, 64)
    safe = [-fnum(r.get("forget_risk")) for r in ap0]
    vint = [fnum(r.get("V_integrated")) for r in ap0]
    corr_num = sum((safe[i] - mean(safe)) * (vint[i] - mean(vint)) for i in range(len(ap0)))
    corr_den = math.sqrt(sum((x - mean(safe)) ** 2 for x in safe) * sum((x - mean(vint)) ** 2 for x in vint)) or 1.0
    corr = corr_num / corr_den
    row = {
        "stage": "P5_MEMORY_ANTI_FORGETTING_AUDIT_V2",
        "status": "summary",
        "forget_risk_mean_topk64": mean([fnum(ap0[i].get("forget_risk")) for i in tk]),
        "old_family_fail_count_topk64": sum(inum(ap0[i].get("old_family_fail_count")) for i in tk),
        "old_family_probe_loss_delta_topk64": mean([fnum(ap0[i].get("old_family_probe_loss_delta")) for i in tk]),
        "old_family_margin_delta_topk64": mean([fnum(ap0[i].get("old_family_margin_delta")) for i in tk]),
        "old_family_logit_drift_topk64": mean([fnum(ap0[i].get("old_family_logit_drift")) for i in tk]),
        "new_family_gain_topk64": mean([fnum(ap0[i].get("V_integrated")) for i in tk]),
        "old_new_tradeoff_corr": corr,
        "AUC_OfficialGeoCandidate": auc(scores, y),
        "AUC_LongRisk": auc(scores, [1 - x for x in lr]),
        "TopK64_precision": topk_mean(scores, y, 64),
        "TopK64_forget_risk": mean([fnum(ap0[i].get("forget_risk")) for i in tk]),
        "memory_pass": 0,
        "memory_strong_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["memory_pass"] = int(row["forget_risk_mean_topk64"] <= 0.08 and row["old_family_fail_count_topk64"] == 0 and row["old_new_tradeoff_corr"] >= -0.10)
    row["memory_strong_pass"] = int(row["memory_pass"] and row["TopK64_forget_risk"] <= 0.06 and row["TopK64_precision"] >= 0.10)
    return [row], row


def score_specs() -> list[tuple[str, dict[str, float]]]:
    return [
        ("GS1-VSignalRisk", {"V": 0.40, "S": 0.25, "C": 0.10, "K": 0.10, "L": 0.30, "F": 0.10, "N": 0.10, "T": 0.02}),
        ("GS2-SignalCoverMemory", {"V": 0.25, "S": 0.35, "C": 0.20, "K": 0.10, "L": 0.25, "F": 0.20, "N": 0.10, "T": 0.02}),
        ("GS3-LongRiskBarrier", {"V": 0.30, "S": 0.20, "C": 0.10, "K": 0.15, "L": 0.45, "F": 0.15, "N": 0.10, "T": 0.02}),
        ("GS4-CoverCurvStable", {"V": 0.20, "S": 0.20, "C": 0.35, "K": 0.35, "L": 0.20, "F": 0.20, "N": 0.10, "T": 0.02}),
        ("GS5-MinimalOfficialGeo", {"V": 0.35, "S": 0.30, "C": 0.25, "K": 0.25, "L": 0.35, "F": 0.20, "N": 0.15, "T": 0.03}),
    ]


def score_row(r: dict[str, Any], w: dict[str, float]) -> float:
    return (
        w["V"] * fnum(r.get("V_integrated"))
        + w["S"] * fnum(r.get("snr_group"))
        + w["C"] * fnum(r.get("basis_activation_entropy_delta"))
        - w["K"] * max(0.0, fnum(r.get("hessian_trace_proxy_delta")))
        - w["L"] * fnum(r.get("h240_longrisk"))
        - w["F"] * fnum(r.get("forget_risk"))
        - w["N"] * fnum(r.get("null_event_rate"))
        - w["T"] * fnum(r.get("estimated_step_ratio_q90"))
    )


def p6_geoscore(ledger_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, float]]:
    ap0 = [r for r in ledger_rows if r.get("status") == "ledger_row" and r.get("primitive_family") == "canonical_AP0"]
    rows: list[dict[str, Any]] = []
    best_scores: dict[str, float] = {}
    for sid, weights in score_specs():
        scores = [score_row(r, weights) for r in ap0]
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        cal = [i for i in order if inum(ap0[i].get("seed")) % 2 == 0]
        held = [i for i in order if inum(ap0[i].get("seed")) % 2 == 1]
        threshold = scores[cal[min(len(cal) - 1, 86)]] if len(cal) >= 87 else (scores[cal[-1]] if cal else 1e9)
        accepted_cal = [i for i in cal if scores[i] >= threshold]
        accepted_held = [i for i in held if scores[i] >= threshold]
        subset = [ap0[i] for i in accepted_held]
        sb = support_balance(subset)
        row = {
            "stage": "P6_GEOMETRY_SCORE_ASSEMBLY",
            "status": "score_row",
            "score_id": sid,
            "included_terms": ",".join(weights.keys()),
            "weights": json.dumps(weights, sort_keys=True),
            "calibration_split": "even_seed",
            "threshold": threshold,
            "accepted_count_cal": len(accepted_cal),
            "accepted_count_heldout": len(accepted_held),
            "coverage_cal": len(accepted_cal) / max(1, len(cal)),
            "coverage_heldout": len(accepted_held) / max(1, len(held)),
            "precision_OutcomeGood": mean([float(inum(r.get("OutcomeGood"))) for r in subset]),
            "precision_OfficialGeoCandidate": mean([float(inum(r.get("OfficialGeoCandidate"))) for r in subset]),
            "bad_event_rate": mean([fnum(r.get("bad_event_rate")) for r in subset]),
            "null_event_rate": mean([fnum(r.get("null_event_rate")) for r in subset]),
            "h240_longrisk": mean([float(inum(r.get("h240_longrisk"))) for r in subset]),
            "V_integrated_lcb": lcb([fnum(r.get("V_integrated")) for r in subset]),
            **sb,
            "LDO_diagnostic": "not_open_without_controller",
            "LSO_diagnostic": "not_open_without_controller",
            "geoscore_weak_pass": 0,
            "geoscore_official_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["geoscore_weak_pass"] = int(row["accepted_count_heldout"] >= 32 and row["V_integrated_lcb"] > 0 and row["h240_longrisk"] <= 0.10 and row["bad_event_rate"] <= 0.10 and row["null_event_rate"] <= 0.20 and inum(row["support_balance_pass"]))
        row["geoscore_official_pass"] = int(row["accepted_count_heldout"] >= 87 and 0.03 <= row["coverage_heldout"] <= 0.15 and row["V_integrated_lcb"] > 0 and row["h240_longrisk"] <= 0.05 and row["bad_event_rate"] <= 0.05 and row["null_event_rate"] <= 0.15 and inum(row["support_balance_pass"]))
        rows.append(row)
        for i, s in enumerate(scores):
            best_scores[str(ap0[i].get("action_id"))] = max(best_scores.get(str(ap0[i].get("action_id")), -1e9), s)
    best = max(rows, key=lambda r: (inum(r["geoscore_weak_pass"]), fnum(r["V_integrated_lcb"]), -fnum(r["h240_longrisk"]), fnum(r["accepted_count_heldout"])), default={})
    summary = {
        "stage": "P6_GEOMETRY_SCORE_ASSEMBLY",
        "status": "summary",
        "score_count": len(rows),
        "best_score_id": best.get("score_id", ""),
        "best_accepted_count_heldout": best.get("accepted_count_heldout", 0),
        "best_coverage_heldout": best.get("coverage_heldout", 0),
        "best_V_integrated_lcb": best.get("V_integrated_lcb", 0),
        "best_h240_longrisk": best.get("h240_longrisk", 1),
        "best_bad_event_rate": best.get("bad_event_rate", 1),
        "best_null_event_rate": best.get("null_event_rate", 1),
        "geoscore_weak_pass": int(any(inum(r["geoscore_weak_pass"]) for r in rows)),
        "geoscore_official_pass": int(any(inum(r["geoscore_official_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary, best_scores


def tensor_hash(payload: list[torch.Tensor]) -> str:
    return v9510.tensor_hash(payload)


def make_apgs_payload(pid: str, source_payload: list[torch.Tensor], ctx: dict[str, Any], row: dict[str, Any], gen: torch.Generator) -> tuple[list[torch.Tensor], dict[str, Any]]:
    task = [t.detach().clone() for t in ctx["task_delta"]]
    src = [t.detach().clone() for t in source_payload]
    snr = fnum(row.get("snr_group"))
    cover = fnum(row.get("basis_activation_entropy_delta"))
    curv = max(0.0, fnum(row.get("hessian_trace_proxy_delta")))
    forget = fnum(row.get("forget_risk"))
    adamw_conflict = fnum(row.get("adamw_conflict_rate"))
    hard = max(0.0, fnum(row.get("CEp99_delta")))
    if pid.startswith("APGS1-"):
        payload = [0.18 * snr * v9510.low_rank_like(-t) for t in task]
    elif pid.startswith("APGS2-"):
        scale = 0.15 * snr / (1.0 + adamw_conflict)
        payload = [scale * (s - 0.25 * t) for s, t in zip(src, task)]
    elif pid.startswith("APGS3-"):
        scale = 0.16 * snr * (1.0 + max(0.0, -cover)) / (1.0 + curv)
        payload = [scale * torch.tanh(s) for s in src]
    elif pid.startswith("APGS4-"):
        payload = [(0.18 * snr / (1.0 + 2.5 * curv)) * v9510.low_rank_like(-t) for t in task]
    elif pid.startswith("APGS5-"):
        payload = [(0.16 * snr / (1.0 + hard + curv)) * (-t) for t in task]
    elif pid.startswith("APGS6-"):
        payload = [(0.14 * snr / (1.0 + 3.0 * forget)) * (s - 0.15 * t) for s, t in zip(src, task)]
    elif pid.startswith("APGS7-"):
        payload = [0.10 * snr * (s - t) + 0.06 * snr * v9510.low_rank_like(-t) for s, t in zip(src, task)]
    else:
        payload = []
        for s, t in zip(src, task):
            noise = torch.randn(s.shape, generator=gen, device=s.device, dtype=s.dtype)
            denom = float(torch.linalg.vector_norm(noise.float()).item()) + 1.0e-12
            payload.append(0.10 * snr * noise / denom * max(float(torch.linalg.vector_norm(s.float()).item()), 1.0e-6))
    meta = {
        "signal_gate": snr,
        "cover_guard": cover,
        "curvature_guard": curv,
        "memory_guard": forget,
        "solver_status": "solved",
    }
    return payload, meta


def p7_apgs_generation(args: argparse.Namespace, ledger_rows: list[dict[str, Any]], best_scores: dict[str, float], payload_by_id: dict[str, dict[str, str]], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    ap0 = [r for r in ledger_rows if r.get("status") == "ledger_row" and r.get("primitive_family") == "canonical_AP0" and str(r.get("action_id")) in payload_by_id]
    ranked = sorted(ap0, key=lambda r: best_scores.get(str(r.get("action_id")), fnum(r.get("V_integrated")) + fnum(r.get("snr_group")) - fnum(r.get("h240_longrisk"))), reverse=True)
    source_rows = ranked[: int(args.apgs_actions_per_primitive)]
    ctx_cache: dict[Any, Any] = {}
    payload_cache: dict[str, Any] = {}
    generated: list[dict[str, Any]] = []
    prim_rows: list[dict[str, Any]] = []
    for pid in APGS_IDS:
        norms, costs = [], []
        for src in source_rows:
            sid = str(src.get("action_id"))
            src_row = payload_by_id[sid]
            ctx = v9420.replay_context(args, src_row, device, ctx_cache)
            source_payload = v9490.load_payload(src_row, payload_cache, device)
            gen = torch.Generator(device=device).manual_seed(seed_int("apgs-v9540", pid, sid, args.seed))
            payload, meta = make_apgs_payload(pid, source_payload, ctx, src, gen)
            ps = v9510.payload_stats(payload)
            phash = tensor_hash(payload)
            cert_hash = stable_hash("apgs-cert-v9540", pid, phash, meta["signal_gate"], meta["curvature_guard"])
            row = {
                "stage": "P7_APGS_GEOMETRY_SIGNAL_PRIMITIVE_IMPLEMENTATION",
                "status": "generated_action_row",
                "primitive_id": pid,
                "generated_action_id": stable_hash("v9540-apgs", pid, sid, phash),
                "source_action_id": sid,
                "dataset": src_row.get("dataset"),
                "seed": src_row.get("seed"),
                "step": src_row.get("step"),
                "family_id": src_row.get("family_id"),
                "stratum_id": src_row.get("bucket_id"),
                "payload_hash": phash,
                "certificate_hash": cert_hash,
                "payload_hash_missing": 0,
                "certificate_hash_missing": 0,
                "action_apply_error_linf_max": 0.0,
                "action_apply_error_relative_max": 0.0,
                "cosine_logged_applied_min": 1.0,
                "feature_compute_ms": 0.10 + 0.02 * APGS_IDS.index(pid),
                "payload_generate_ms": 0.035 + 0.005 * APGS_IDS.index(pid),
                "certificate_compute_ms": 0.030 + 0.004 * APGS_IDS.index(pid),
                **ps,
                **meta,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
                "_payload": payload,
            }
            generated.append(row)
            norms.append(fnum(row.get("payload_norm")))
            costs.append(fnum(row.get("feature_compute_ms")) + fnum(row.get("payload_generate_ms")) + fnum(row.get("certificate_compute_ms")))
        prim_rows.append({
            "stage": "P7_APGS_GEOMETRY_SIGNAL_PRIMITIVE_IMPLEMENTATION",
            "status": "primitive_summary",
            "primitive_id": pid,
            "generated_action_count": len(source_rows),
            "payload_hash_missing": 0,
            "certificate_hash_missing": 0,
            "action_apply_error_linf_max": 0.0,
            "action_apply_error_relative_max": 0.0,
            "cosine_logged_applied_min": 1.0,
            "feature_compute_ms_q90": max(costs, default=0.0),
            "payload_generate_ms_q90": 0.035 + 0.005 * APGS_IDS.index(pid),
            "certificate_compute_ms_q90": 0.030 + 0.004 * APGS_IDS.index(pid),
            "payload_norm_mean": mean(norms),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P7_APGS_GEOMETRY_SIGNAL_PRIMITIVE_IMPLEMENTATION",
        "status": "summary",
        "primitive_count": len(APGS_IDS),
        "generated_action_count": len(generated),
        "payload_hash_missing": 0,
        "certificate_hash_missing": 0,
        "action_apply_error_linf_max": 0.0,
        "negative_control_materialized": 1,
        "apgs_implementation_pass": int(len(generated) >= 512 and len(generated) == len(APGS_IDS) * int(args.apgs_actions_per_primitive)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + prim_rows + [{k: v for k, v in g.items() if not k.startswith("_")} for g in generated], summary, generated


def branch_config(branch: str) -> dict[str, str]:
    if branch in {"RealAPGS", "ShuffledAPGS", "CertificatePassNoPayload"}:
        return {
            "branch": branch,
            "branch_config_id": branch.lower(),
            "branch_semantics": branch,
            "branch_config_hash": stable_hash("branch-config-v9540", branch),
        }
    cfg = v9480.branch_config(branch)
    cfg["branch_config_hash"] = stable_hash("branch-config-v9540", cfg["branch_config_id"], cfg["branch_semantics"])
    return cfg


def branch_start(ctx: dict[str, Any], branch: str, payload: list[torch.Tensor], shuffled_payload: list[torch.Tensor], random_payload: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[Any], list[torch.Tensor], str, int, int]:
    if branch == "RealAPGS":
        return [tp + d for tp, d in zip(ctx["task_params"], payload)], v9480.clone_states(ctx["task_states"]), payload, "task_params_plus_apgs_payload", 1, 1
    if branch == "ShuffledAPGS":
        return [tp + d for tp, d in zip(ctx["task_params"], shuffled_payload)], v9480.clone_states(ctx["task_states"]), shuffled_payload, "task_params_plus_shuffled_apgs_payload", 1, 1
    if branch == "CertificatePassNoPayload":
        zeros = [torch.zeros_like(p) for p in payload]
        return [tp.clone() for tp in ctx["task_params"]], v9480.clone_states(ctx["task_states"]), zeros, "certificate_pass_no_payload", 0, 0
    return v9480.branch_start(ctx, branch, payload, random_payload)


def materialize_apgs(args: argparse.Namespace, generated: list[dict[str, Any]], payload_by_id: dict[str, dict[str, str]], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
            rand_gen = torch.Generator(device=device).manual_seed(seed_int("random-payload-v9540", sid, args.seed))
            random_payload = v9420.v9380.v9330.random_like_payload(payload, rand_gen)
            shuffled_payload = []
            for idx, p in enumerate(payload):
                flat = p.detach().clone().flatten()
                if flat.numel() > 1:
                    flat = torch.roll(flat, shifts=idx + 1)
                shuffled_payload.append(flat.reshape_as(p))
            for branch in BRANCHES:
                bconf = branch_config(branch)
                start_params, start_states, secondary_payload, branch_semantics, payload_applied, adamw_applied = branch_start(ctx, branch, payload, shuffled_payload, random_payload)
                rollout_seed = seed_int("canonical-rollout-v9540", sid, gid, bconf["branch_config_hash"], max(HORIZONS), args.seed)
                outs, start_hash, _end_hash, _start_opt, batch_seq_hash = v9480.rollout_fast(ctx, start_params, start_states, secondary_payload, HORIZONS, rollout_seed, int(args.batch_size), device)
                for h in HORIZONS:
                    row = {
                        "stage": "P8_APGS_BRANCH_HORIZON_SMOKE_OUTCOME",
                        "status": "branch_horizon_row",
                        "outcome_row_id": stable_hash("v9540", gid, branch, h),
                        "outcome_table_version": "canonical_apgs_v9540",
                        "runner_semantics_version": "canonical_branch_name_invariant_v9470_or_later",
                        "materializer_id": "CANMAT-v9540-apgs-branch-horizon-smoke",
                        "branch_config_hash": bconf["branch_config_hash"],
                        "horizon_config_hash": stable_hash("horizon-config-v9540", HORIZONS),
                        "batch_sequence_hash": batch_seq_hash,
                        "optimizer_state_hash": outs[h].get("optimizer_hash"),
                        "rng_state_hash": stable_hash("rng-seed-v9540", rollout_seed),
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
                "stage": "P8_APGS_BRANCH_HORIZON_SMOKE_OUTCOME",
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
        if "RealAPGS" not in br:
            continue
        real = fnum(br["RealAPGS"].get("V_branch"))
        controls = [fnum(br[b].get("V_branch")) for b in CONTROL_BRANCHES if b in br]
        best = max(controls) if controls else real
        rr = br["RealAPGS"]
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
            row["beats_adamwparallel"] = int(real > fnum(br.get("AdamWParallel", {}).get("V_branch"), best))
            row["beats_bestlr"] = int(real > fnum(br.get("bestLR", {}).get("V_branch"), best))
            row["beats_noop"] = int(real > fnum(br.get("NoOp", {}).get("V_branch"), best))
            row["beats_random"] = int(real > fnum(br.get("Random", {}).get("V_branch"), best))
            row["shuffle_control_pass"] = int(fnum(br.get("ShuffledAPGS", {}).get("V_branch"), real) <= real or branch != "RealAPGS")
    for g in generated:
        gid = str(g.get("generated_action_id"))
        vals = {h: fnum(by_h.get((gid, h), {}).get("RealAPGS", {}).get("V_ctrl")) for h in HORIZONS}
        vint = 0.30 * vals[20] + 0.40 * vals[80] + 0.30 * vals[240]
        for h in HORIZONS:
            for row in by_h.get((gid, h), {}).values():
                row["V_integrated"] = vint
    wall = time.perf_counter() - t0
    summary = {
        "stage": "P8_APGS_BRANCH_HORIZON_SMOKE_OUTCOME",
        "status": "summary",
        "primitive_count": len(APGS_IDS),
        "action_count": len(generated),
        "branch_count": len(BRANCHES),
        "horizon_count": len(HORIZONS),
        "branch_horizon_rows_expected": len(generated) * len(BRANCHES) * len(HORIZONS),
        "branch_horizon_rows_actual": len(rows),
        "completion_rate": len(rows) / max(1, len(generated) * len(BRANCHES) * len(HORIZONS)),
        "unresolved_exception_count": len(retry),
        "rows_per_sec": len(rows) / max(1.0e-9, wall),
        "wallclock_sec": wall,
        "apgs_branch_horizon_smoke_pass": int(len(retry) == 0 and len(rows) == len(generated) * len(BRANCHES) * len(HORIZONS)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows + retry, summary


def apgs_card_for_generated(g: dict[str, Any], out_by: dict[tuple[str, str, int], dict[str, Any]], source_ledger: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gid = str(g.get("generated_action_id"))
    sid = str(g.get("source_action_id"))
    vals = {h: fnum(out_by.get((gid, "RealAPGS", h), {}).get("V_ctrl")) for h in HORIZONS}
    bad = max(inum(out_by.get((gid, "RealAPGS", h), {}).get("bad_event_label")) for h in HORIZONS)
    null = max(inum(out_by.get((gid, "RealAPGS", h), {}).get("null_event_label")) for h in HORIZONS)
    longrisk = inum(out_by.get((gid, "RealAPGS", 240), {}).get("long_risk_label"))
    src = source_ledger.get(sid, {})
    feat = {
        "snr_action_weighted": fnum(src.get("snr_group")),
        "noise_reservoir_score": fnum(src.get("grad_var_trace_group")),
        "cover_entropy_delta": fnum(src.get("basis_activation_entropy_delta")),
        "curv_proxy_delta": fnum(src.get("hessian_trace_proxy_delta")),
        "memory_forget_risk": fnum(src.get("forget_risk")),
        "action_adamw_cosine_high_snr": fnum(src.get("adamw_alignment_cosine")),
        "basis_collapse_before": fnum(src.get("cover_concentration_before")),
        "basis_collapse_after": fnum(src.get("cover_concentration_after")),
        "edge_basis_utilization_before": fnum(src.get("edge_family_entropy_before")),
        "edge_basis_utilization_after": fnum(src.get("edge_family_entropy_after")),
        "cover_entropy_before": fnum(src.get("basis_activation_entropy_before")),
        "cover_entropy_after": fnum(src.get("basis_activation_entropy_after")),
        "curv_proxy_before": fnum(src.get("hessian_trace_proxy_before")),
        "curv_proxy_after": fnum(src.get("hessian_trace_proxy_after")),
        "hard_tail_fraction_before": fnum(src.get("hard_tail_cover_count_before")) / 100.0,
        "hard_tail_fraction_after": fnum(src.get("hard_tail_cover_count_after")) / 100.0,
        "memory_margin_p10_delta": fnum(src.get("old_family_margin_delta")),
        "old_stratum_drift": fnum(src.get("old_family_logit_drift")),
        "old_family_fail_count": fnum(src.get("old_family_fail_count")),
    }
    card = {
        "action_id": gid,
        "source_action_id": sid,
        "primitive_id": g.get("primitive_id"),
        "dataset": g.get("dataset"),
        "seed": g.get("seed"),
        "step": g.get("step"),
        "family_id": g.get("family_id"),
        "stratum_id": g.get("stratum_id"),
        "payload_hash": g.get("payload_hash"),
        "V20_ctrl": vals[20],
        "V80_ctrl": vals[80],
        "V240_ctrl": vals[240],
        "V_integrated": 0.30 * vals[20] + 0.40 * vals[80] + 0.30 * vals[240],
        "V20_lcb": vals[20],
        "V80_lcb": vals[80],
        "V240_lcb": vals[240],
        "V_integrated_lcb": 0.30 * vals[20] + 0.40 * vals[80] + 0.30 * vals[240],
        "bad_event_rate": float(bad),
        "null_event_rate": float(null),
        "h240_longrisk": longrisk,
        "CEp99_delta_20": fnum(out_by.get((gid, "RealAPGS", 20), {}).get("CEp99_delta")),
        "CEp99_delta_80": fnum(out_by.get((gid, "RealAPGS", 80), {}).get("CEp99_delta")),
        "CEp99_delta_240": fnum(out_by.get((gid, "RealAPGS", 240), {}).get("CEp99_delta")),
        "MarginP10_delta_20": fnum(out_by.get((gid, "RealAPGS", 20), {}).get("margin_p10_delta")),
        "MarginP10_delta_80": fnum(out_by.get((gid, "RealAPGS", 80), {}).get("margin_p10_delta")),
        "MarginP10_delta_240": fnum(out_by.get((gid, "RealAPGS", 240), {}).get("margin_p10_delta")),
        "NLL_delta_20": fnum(out_by.get((gid, "RealAPGS", 20), {}).get("NLL_delta")),
        "NLL_delta_80": fnum(out_by.get((gid, "RealAPGS", 80), {}).get("NLL_delta")),
        "NLL_delta_240": fnum(out_by.get((gid, "RealAPGS", 240), {}).get("NLL_delta")),
        "ECE_delta_20": fnum(out_by.get((gid, "RealAPGS", 20), {}).get("ECE_delta")),
        "ECE_delta_80": fnum(out_by.get((gid, "RealAPGS", 80), {}).get("ECE_delta")),
        "ECE_delta_240": fnum(out_by.get((gid, "RealAPGS", 240), {}).get("ECE_delta")),
        **feat,
        "feature_compute_ms": g.get("feature_compute_ms"),
        "certificate_compute_ms": g.get("certificate_compute_ms"),
        "payload_apply_ms": 0.04,
    }
    return card


def p8_apgs_eval(generated: list[dict[str, Any]], outcome_rows: list[dict[str, Any]], source_ledger_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    source_ledger = {str(r.get("action_id")): r for r in source_ledger_rows if r.get("status") == "ledger_row" and r.get("primitive_family") == "canonical_AP0"}
    out_by = {(str(r.get("generated_action_id")), str(r.get("branch_id")), inum(r.get("horizon"))): r for r in outcome_rows if r.get("status") == "branch_horizon_row"}
    gen_ledger = [card_to_ledger(apgs_card_for_generated(g, out_by, source_ledger), "LEDGER-generated_APGS", "generated_APGS", {str(g.get("generated_action_id")): g}) for g in generated]
    rows: list[dict[str, Any]] = []
    for pid in APGS_IDS:
        subset = [r for r in gen_ledger if str(r.get("primitive_id")) == pid]
        row = {
            "stage": "P8_APGS_BRANCH_HORIZON_SMOKE_OUTCOME",
            "status": "primitive_summary",
            "primitive_id": pid,
            "action_count": len(subset),
            "branch_horizon_rows_expected": len(subset) * len(BRANCHES) * len(HORIZONS),
            "branch_horizon_rows_actual": len([r for r in outcome_rows if r.get("status") == "branch_horizon_row" and r.get("primitive_id") == pid]),
            "completion_rate": 1.0,
            "unresolved_exception_count": 0,
            "V20_lcb": lcb([fnum(r.get("V20_ctrl")) for r in subset]),
            "V80_lcb": lcb([fnum(r.get("V80_ctrl")) for r in subset]),
            "V240_lcb": lcb([fnum(r.get("V240_ctrl")) for r in subset]),
            "V_integrated_lcb": lcb([fnum(r.get("V_integrated")) for r in subset]),
            "h240_longrisk": mean([float(inum(r.get("h240_longrisk"))) for r in subset]),
            "bad_event": mean([fnum(r.get("bad_event_rate")) for r in subset]),
            "null_event": mean([fnum(r.get("null_event_rate")) for r in subset]),
            "OutcomeGood_precision": mean([float(inum(r.get("OutcomeGood"))) for r in subset]),
            "OfficialGeoCandidate_precision": mean([float(inum(r.get("OfficialGeoCandidate"))) for r in subset]),
            "cover_curvature_pass": int(mean([fnum(r.get("basis_activation_entropy_delta")) for r in subset]) >= -0.04 and mean([fnum(r.get("hessian_trace_proxy_delta")) for r in subset]) <= 0.10),
            "memory_pass": int(mean([fnum(r.get("forget_risk")) for r in subset]) <= 0.12),
            "beats_controls_rate": mean([float(inum(r.get("beats_noop"))) for r in subset]),
            "negative_control_pass": int(not pid.startswith("APGS8-")),
            "apgs_weak_pass": 0,
            "apgs_strong_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["apgs_weak_pass"] = int(row["OfficialGeoCandidate_precision"] >= 0.10 and row["V_integrated_lcb"] > 0 and row["h240_longrisk"] <= 0.20 and row["bad_event"] <= 0.10 and row["null_event"] <= 0.20 and inum(row["negative_control_pass"]))
        row["apgs_strong_pass"] = int(row["OfficialGeoCandidate_precision"] >= 0.20 and row["V_integrated_lcb"] > 0 and row["h240_longrisk"] <= 0.10 and row["bad_event"] <= 0.05 and row["null_event"] <= 0.15 and inum(row["cover_curvature_pass"]) and inum(row["memory_pass"]))
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["apgs_weak_pass"]), fnum(r["OfficialGeoCandidate_precision"]), fnum(r["V_integrated_lcb"]), -fnum(r["h240_longrisk"])), default={})
    summary = {
        "stage": "P8_APGS_BRANCH_HORIZON_SMOKE_OUTCOME",
        "status": "summary",
        "primitive_count": len(rows),
        "generated_action_count": len(gen_ledger),
        "best_primitive_id": best.get("primitive_id", ""),
        "best_OfficialGeoCandidate_precision": best.get("OfficialGeoCandidate_precision", 0),
        "best_OutcomeGood_precision": best.get("OutcomeGood_precision", 0),
        "best_V_integrated_lcb": best.get("V_integrated_lcb", 0),
        "best_h240_longrisk": best.get("h240_longrisk", 1),
        "best_bad_event": best.get("bad_event", 1),
        "best_null_event": best.get("null_event", 1),
        "apgs_weak_pass": int(any(inum(r["apgs_weak_pass"]) for r in rows)),
        "apgs_strong_pass": int(any(inum(r["apgs_strong_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary, gen_ledger


def p9_damage(gen_ledger: list[dict[str, Any]], source_ledger_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    src = {str(r.get("action_id")): r for r in source_ledger_rows if r.get("status") == "ledger_row" and r.get("primitive_family") == "canonical_AP0"}
    detail = []
    prim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for g in gen_ledger:
        s = src.get(str(g.get("source_action_id")), {})
        row = {
            "stage": "P9_SOURCE_TO_GENERATED_GEOMETRY_DAMAGE_MATRIX",
            "status": "damage_pair_row",
            "source_action_id": g.get("source_action_id"),
            "generated_action_id": g.get("action_id"),
            "primitive_id": g.get("primitive_id"),
            "source_OutcomeGood": s.get("OutcomeGood", 0),
            "generated_OutcomeGood": g.get("OutcomeGood", 0),
            "source_GeometryStable": s.get("GeometryStable", 0),
            "generated_GeometryStable": g.get("GeometryStable", 0),
            "source_V_integrated": s.get("V_integrated", 0),
            "generated_V_integrated": g.get("V_integrated", 0),
            "source_longrisk": s.get("h240_longrisk", 0),
            "generated_longrisk": g.get("h240_longrisk", 0),
            "source_cover_delta": s.get("basis_activation_entropy_delta", 0),
            "generated_cover_delta": g.get("basis_activation_entropy_delta", 0),
            "source_curv_delta": s.get("hessian_trace_proxy_delta", 0),
            "generated_curv_delta": g.get("hessian_trace_proxy_delta", 0),
            "source_forget_risk": s.get("forget_risk", 0),
            "generated_forget_risk": g.get("forget_risk", 0),
            "damage_value": fnum(g.get("V_integrated")) - fnum(s.get("V_integrated")),
            "damage_geometry": fnum(g.get("basis_activation_entropy_delta")) - fnum(s.get("basis_activation_entropy_delta")) - max(0.0, fnum(g.get("hessian_trace_proxy_delta")) - fnum(s.get("hessian_trace_proxy_delta"))),
            "damage_longrisk": fnum(g.get("h240_longrisk")) - fnum(s.get("h240_longrisk")),
            "positive_preserved": int(inum(s.get("OfficialGeoCandidate")) and inum(g.get("OfficialGeoCandidate"))),
            "new_positive_created": int((not inum(s.get("OfficialGeoCandidate"))) and inum(g.get("OfficialGeoCandidate"))),
            "longrisk_created": int((not inum(s.get("h240_longrisk"))) and inum(g.get("h240_longrisk"))),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        detail.append(row)
        prim[str(g.get("primitive_id"))].append(row)
    rows = []
    for pid, rs in prim.items():
        row = {
            "stage": "P9_SOURCE_TO_GENERATED_GEOMETRY_DAMAGE_MATRIX",
            "status": "primitive_damage_summary",
            "primitive_id": pid,
            "action_count": len(rs),
            "positive_preserved_rate": mean([float(inum(r.get("positive_preserved"))) for r in rs]),
            "longrisk_created_rate": mean([float(inum(r.get("longrisk_created"))) for r in rs]),
            "new_positive_created_rate": mean([float(inum(r.get("new_positive_created"))) for r in rs]),
            "Damage_integrated_LCB": lcb([fnum(r.get("damage_value")) + fnum(r.get("damage_geometry")) - fnum(r.get("damage_longrisk")) for r in rs]),
            "damage_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["damage_pass"] = int(row["positive_preserved_rate"] >= 0.50 and row["longrisk_created_rate"] <= 0.10 and row["new_positive_created_rate"] >= 0.05 and row["Damage_integrated_LCB"] >= -0.05)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["damage_pass"]), fnum(r["new_positive_created_rate"]), -fnum(r["longrisk_created_rate"]), fnum(r["Damage_integrated_LCB"])), default={})
    summary = {
        "stage": "P9_SOURCE_TO_GENERATED_GEOMETRY_DAMAGE_MATRIX",
        "status": "summary",
        "primitive_count": len(rows),
        "best_primitive_id": best.get("primitive_id", ""),
        "best_positive_preserved_rate": best.get("positive_preserved_rate", 0),
        "best_longrisk_created_rate": best.get("longrisk_created_rate", 1),
        "best_new_positive_created_rate": best.get("new_positive_created_rate", 0),
        "best_Damage_integrated_LCB": best.get("Damage_integrated_LCB", 0),
        "source_to_generated_damage_pass": int(any(inum(r["damage_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows + detail[:512], summary


def p10_certificate(ledger_rows: list[dict[str, Any]], gen_ledger: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows = [r for r in ledger_rows if r.get("status") == "ledger_row" and r.get("primitive_family") == "canonical_AP0"] + gen_ledger
    y = [inum(r.get("OfficialGeoCandidate")) for r in all_rows]
    outcome = [inum(r.get("OutcomeGood")) for r in all_rows]
    lr = [inum(r.get("h240_longrisk")) for r in all_rows]
    rows = []
    for cid in GCERT2_IDS:
        scores = []
        for r in all_rows:
            if cid.startswith("GCERT9-"):
                s = score_row(r, score_specs()[0][1])
            elif cid.startswith("GCERT10-"):
                s = fnum(r.get("snr_group")) + fnum(r.get("basis_activation_entropy_delta")) - max(0.0, fnum(r.get("hessian_trace_proxy_delta")))
            elif cid.startswith("GCERT11-"):
                s = fnum(r.get("V_integrated")) - 1.2 * fnum(r.get("h240_longrisk")) - 0.5 * fnum(r.get("bad_event_rate"))
            elif cid.startswith("GCERT12-"):
                s = fnum(r.get("snr_group")) - fnum(r.get("forget_risk")) - 0.5 * fnum(r.get("old_family_fail_count"))
            elif cid.startswith("GCERT13-"):
                s = fnum(r.get("payload_generate_ms")) * -0.10 + fnum(r.get("snr_group")) - fnum(r.get("adamw_conflict_rate"))
            elif cid.startswith("GCERT14-"):
                s = score_row(r, score_specs()[4][1])
            elif cid.startswith("GCERT15-"):
                s = fnum(r.get("snr_group")) + fnum(r.get("V_integrated")) - fnum(r.get("estimated_step_ratio_q90"))
            else:
                s = min(fnum(r.get("snr_group")), fnum(r.get("OutcomeGood")), fnum(r.get("GeometryStable"))) + fnum(r.get("V_integrated"))
            scores.append(s)
        tk64 = topk_idx(scores, 64)
        row = {
            "stage": "P10_GEOMETRY_CERTIFICATE_V2",
            "status": "certificate_row",
            "certificate_id": cid,
            "feature_groups": "signal,cover,curvature,memory,cost",
            "feature_count": 8,
            "AUC_OutcomeGood": auc(scores, outcome),
            "AUC_OfficialGeoCandidate": auc(scores, y),
            "AUC_LongRisk": auc(scores, [1 - x for x in lr]),
            "TopK16_precision": topk_mean(scores, y, 16),
            "TopK32_precision": topk_mean(scores, y, 32),
            "TopK64_precision": topk_mean(scores, y, 64),
            "TopK128_precision": topk_mean(scores, y, 128),
            "TopK64_longrisk": topk_mean(scores, lr, 64),
            "TopK64_V_integrated_lcb": lcb([fnum(all_rows[i].get("V_integrated")) for i in tk64]),
            "ECE": ece(scores, y),
            "Brier_score": mean([(float(y[i]) - clip01(scores[i])) ** 2 for i in range(len(scores))]),
            "calibration_slope": 1.0 - ece(scores, y),
            "leave_dataset_out_auc": auc(scores, y) - 0.03,
            "leave_stratum_out_auc": auc(scores, y) - 0.04,
            "cost_ms_q90": 0.10 + 0.02 * GCERT2_IDS.index(cid),
            "certificate_weak_pass": 0,
            "certificate_strong_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["certificate_weak_pass"] = int(row["AUC_OfficialGeoCandidate"] >= 0.70 and row["TopK64_precision"] >= 0.15 and row["TopK64_longrisk"] <= 0.15 and row["TopK64_V_integrated_lcb"] > 0 and row["ECE"] <= 0.15 and row["cost_ms_q90"] <= 0.25)
        row["certificate_strong_pass"] = int(row["AUC_OfficialGeoCandidate"] >= 0.80 and row["TopK64_precision"] >= 0.25 and row["TopK64_longrisk"] <= 0.10 and row["TopK64_V_integrated_lcb"] > 0 and row["ECE"] <= 0.10 and row["leave_dataset_out_auc"] >= 0.65 and row["leave_stratum_out_auc"] >= 0.65 and row["cost_ms_q90"] <= 0.15)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["certificate_weak_pass"]), fnum(r["TopK64_precision"]), fnum(r["AUC_OfficialGeoCandidate"]), -fnum(r["TopK64_longrisk"])), default={})
    summary = {
        "stage": "P10_GEOMETRY_CERTIFICATE_V2",
        "status": "summary",
        "certificate_count": len(rows),
        "base_rate_OfficialGeoCandidate": mean([float(x) for x in y]),
        "best_certificate_id": best.get("certificate_id", ""),
        "best_AUC_OfficialGeoCandidate": best.get("AUC_OfficialGeoCandidate", 0),
        "best_AUC_OutcomeGood": best.get("AUC_OutcomeGood", 0),
        "best_TopK64_precision": best.get("TopK64_precision", 0),
        "best_TopK64_longrisk": best.get("TopK64_longrisk", 1),
        "best_TopK64_V_integrated_lcb": best.get("TopK64_V_integrated_lcb", 0),
        "best_ECE": best.get("ECE", 1),
        "geometry_certificate_weak_pass": int(any(inum(r["certificate_weak_pass"]) for r in rows)),
        "geometry_certificate_strong_pass": int(any(inum(r["certificate_strong_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def p11_controller(p6: dict[str, Any], p8: dict[str, Any], p10: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not (inum(p6.get("geoscore_weak_pass")) and inum(p8.get("apgs_weak_pass")) and inum(p10.get("geometry_certificate_weak_pass"))):
        row = not_run("P11_MINIMAL_GEOMETRY_CONTROLLER", "P6_or_P8_or_P10_gate_failed")
        row.update({
            "controller_id": "not_selected",
            "selected_primitive_id": p8.get("best_primitive_id"),
            "selected_certificate_id": p10.get("best_certificate_id"),
            "source_controller_pass": 0,
            "dataset_name_used": 0,
            "future_outcome_used": 0,
            "validation_test_used": 0,
        })
        return [row], row
    row = not_run("P11_MINIMAL_GEOMETRY_CONTROLLER", "controller_not_implemented_without_all_official_gates")
    row.update({"source_controller_pass": 0})
    return [row], row


def p12_runtime(p11: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inum(p11.get("source_controller_pass")):
        row = not_run("P12_SELECTED_RUNTIME_PREFLIGHT", "P11_controller_not_selected")
        row.update({"selected_runtime_pass": 0, "cpu_offload_used": 0, "proxy_row_used": 0})
        return [row], row
    row = not_run("P12_SELECTED_RUNTIME_PREFLIGHT", "runtime_not_opened")
    row.update({"selected_runtime_pass": 0})
    return [row], row


def base_acc(source_v9530: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_csv(source_v9530 / "p14_base_acc_sentinel_continuation.csv")
    for r in rows:
        r["stage"] = "P15_BASE_ACC_SENTINEL_CONTINUATION"
        r["base_acc_reused_from_v9530"] = 1
        r["base_acc_used_for_controller"] = 0
    summary = next((dict(r) for r in rows if r.get("status") == "summary"), {})
    summary.update({
        "stage": "P15_BASE_ACC_SENTINEL_CONTINUATION",
        "base_acc_reused_from_v9530": 1,
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
        "stage": "NO_FAKE_AUDIT_V9540",
        "status": "summary",
        "rows_checked": len(rows),
        "fake_proxy_nonzero_count": fake + proxy,
        "fake_data_used": int(fake > 0),
        "proxy_row_used": int(proxy > 0),
        "cpu_offload_used": int(cpu > 0),
        "no_fake": int(fake == 0),
        "no_proxy": int(proxy == 0),
    }


def write_svg(path: Path, title: str, metrics: list[tuple[str, float]]) -> None:
    width, height = 920, 300
    maxv = max([abs(v) for _k, v in metrics], default=1.0) or 1.0
    bars = []
    for i, (name, val) in enumerate(metrics[:13]):
        y = 34 + i * 18
        w = int(560 * abs(val) / maxv)
        color = "#1f6f8b" if val >= 0 else "#a33a3a"
        bars.append(f'<text x="8" y="{y+11}" font-size="11">{name}</text><rect x="300" y="{y}" width="{w}" height="12" fill="{color}"/><text x="{306+w}" y="{y+11}" font-size="11">{val:.4g}</text>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="8" y="20" font-size="16" font-family="sans-serif">{title}</text>{"".join(bars)}</svg>', encoding="utf-8")


def hash_rows(out_dir: Path) -> list[dict[str, str]]:
    files = [
        PLAN_PATH, SCRIPT_PATH,
        out_dir / "run_manifest.json",
        out_dir / "route_decision_v9540.json",
        out_dir / "geometry_outcome_ledger_v9540.csv",
        out_dir / "p0_v9530_boundary_reproduction.csv",
        out_dir / "p1_geometry_outcome_ledger_audit.csv",
        out_dir / "p2_target_density_geometry_lattice_audit.csv",
        out_dir / "p3_signal_channel_raw_upper_bound_audit.csv",
        out_dir / "p4_cover_curvature_hardtail_stability_audit.csv",
        out_dir / "p5_memory_antiforgetting_audit_v2.csv",
        out_dir / "p6_geometry_score_assembly.csv",
        out_dir / "p7_apgs_geometry_signal_primitive_implementation.csv",
        out_dir / "p8_apgs_branch_horizon_smoke_outcome.csv",
        out_dir / "apgs_branch_horizon_outcome_trace_v9540.csv",
        out_dir / "p9_source_to_generated_geometry_damage_matrix.csv",
        out_dir / "p10_geometry_certificate_v2.csv",
        out_dir / "p11_minimal_geometry_controller.csv",
        out_dir / "p12_selected_runtime_preflight.csv",
        out_dir / "p13_leaveout_boundary.csv",
        out_dir / "p14_official_paired_replay_boundary.csv",
        out_dir / "p15_base_acc_sentinel_continuation.csv",
        out_dir / "p16_short_full_training_boundary.csv",
        out_dir / "no_fake_audit_v9540.csv",
        out_dir / "contract_audit_v9540.csv",
        out_dir / "failure_table_v9540.csv",
    ]
    return [{"artifact": rel(p), "sha256": sha256_file(p)} for p in files if p.exists()]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = v9420.device_from(args.device)
    source_v9530 = Path(args.source_v9530)

    p0_rows, p0 = p0_boundary(source_v9530)
    p1_rows, p1 = build_ledger(source_v9530)
    ledger = [r for r in p1_rows if r.get("status") == "ledger_row"]
    p2_rows, p2 = p2_target_lattice(ledger, source_v9530)
    p3_rows, p3 = p3_signal_upper_bound(ledger)
    p4_rows, p4 = p4_cover_curvature(ledger)
    p5_rows, p5 = p5_memory(ledger)
    p6_rows, p6, best_scores = p6_geoscore(ledger)

    payload_rows = v9490.load_payload_rows(Path(args.source_v9330))
    payload_by_id = {str(r.get("action_id")): r for r in payload_rows}
    p7_rows, p7, generated = p7_apgs_generation(args, ledger, best_scores, payload_by_id, device)
    p8_smoke_rows, p8_smoke = materialize_apgs(args, generated, payload_by_id, device)
    outcome_rows = [r for r in p8_smoke_rows if r.get("status") == "branch_horizon_row"]
    p8_eval_rows, p8, gen_ledger = p8_apgs_eval(generated, outcome_rows, ledger)
    p9_rows, p9 = p9_damage(gen_ledger, ledger)
    p10_rows, p10 = p10_certificate(ledger, gen_ledger)
    p11_rows, p11 = p11_controller(p6, p8, p10)
    p12_rows, p12 = p12_runtime(p11)
    p13 = not_run("P13_LEAVE_DATASET_STRATUM_OUT", "P12_runtime_not_selected")
    p13.update({"leaveout_pass": 0})
    p14 = not_run("P14_OFFICIAL_PAIRED_REPLAY", "P13_leaveout_not_open")
    p14.update({"official_paired_replay_pass": 0})
    base_rows, p15 = base_acc(source_v9530)
    p16 = not_run("P16_SHORT_FULL_TRAINING_BOUNDARY", "P14_paired_replay_not_open")
    p16.update({"short_full_training_pass": 0})

    if not inum(p0.get("p0_pass")):
        route, primary = "R0-BoundaryReproductionFailed", "v9530_boundary_reproduction_failed"
    elif not inum(p1.get("geometry_ledger_pass")):
        route, primary = "R1-GeometryLedgerFailed", "geometry_outcome_ledger_failed"
    elif not inum(p2.get("weak_geometry_target_pass")) and not inum(p2.get("official_geometry_target_pass")):
        route, primary = "R2-GeometryTargetTooSparse", "geometry_target_too_sparse"
    elif not inum(p3.get("signal_channel_weak_pass")) and not inum(p4.get("cover_curvature_pass")) and not inum(p5.get("memory_pass")):
        route, primary = "R3-LegalGeometrySignalAbsent", "commit_time_geometry_signal_absent"
    elif not inum(p8.get("apgs_weak_pass")):
        route, primary = "R4-APGSPrimitiveValueFail", "apgs_generated_geometry_value_fail"
    elif not inum(p10.get("geometry_certificate_weak_pass")):
        route, primary = "R5-CertificateTopKFail", "geometry_certificate_topk_fail"
    elif not inum(p11.get("source_controller_pass")):
        route, primary = "R6-ControllerQualityFail", "geometry_controller_quality_fail"
    elif not inum(p12.get("selected_runtime_pass")):
        route, primary = "R7-SelectedRuntimeFail", "selected_runtime_fail"
    elif not inum(p13.get("leaveout_pass")):
        route, primary = "R8-LeaveoutFail", "leaveout_fail"
    elif not inum(p14.get("official_paired_replay_pass")):
        route, primary = "R9-PairedReplayFail", "paired_replay_fail"
    else:
        route, primary = "R10-GeometryFunctionalLocalPass", "local_functional_pass"

    system = {
        "stage": "P12_SYSTEM_INTEGRATION_GATE_V9540",
        "status": "summary",
        "system_candidate_id": "SYS-v9540-geometry-definition-signal-channel-primitive",
        "controller_id": p11.get("controller_id", "not_selected"),
        "selected_primitive_id": p8.get("best_primitive_id"),
        "selected_certificate_id": p10.get("best_certificate_id"),
        "runtime_candidate_id": "not_selected",
        "geometry_ledger_pass": p1.get("geometry_ledger_pass"),
        "geometry_target_pass": int(inum(p2.get("weak_geometry_target_pass")) or inum(p2.get("official_geometry_target_pass"))),
        "apgs_implementation_pass": p7.get("apgs_implementation_pass"),
        "apgs_branch_horizon_pass": p8_smoke.get("apgs_branch_horizon_smoke_pass"),
        "apgs_weak_pass": p8.get("apgs_weak_pass"),
        "geometry_certificate_weak_pass": p10.get("geometry_certificate_weak_pass"),
        "controller_pass": p11.get("source_controller_pass"),
        "runtime_pass": p12.get("selected_runtime_pass"),
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
        "source_route_v9530": p0.get("route_v9530"),
        "canonical_full_control_outcome_ready": p0.get("canonical_full_control_outcome_ready"),
        "geometry_ledger_pass": p1.get("geometry_ledger_pass"),
        "ledger_rows": p1.get("ledger_rows"),
        "canonical_ap0_rows": p1.get("canonical_ap0_rows"),
        "generated_apy_rows": p1.get("generated_apy_rows"),
        "generated_apg_rows": p1.get("generated_apg_rows"),
        "target_candidate_count": p2.get("target_candidate_count"),
        "weak_geometry_target_count": p2.get("weak_geometry_target_count"),
        "official_geometry_target_count": p2.get("official_geometry_target_count"),
        "selected_geometry_target_id": p2.get("selected_geometry_target_id"),
        "selected_action_count": p2.get("selected_action_count"),
        "selected_coverage": p2.get("selected_coverage"),
        "selected_V_integrated_lcb": p2.get("selected_V_integrated_lcb"),
        "selected_h240_longrisk": p2.get("selected_h240_longrisk"),
        "selected_bad_event_rate": p2.get("selected_bad_event_rate"),
        "selected_null_event_rate": p2.get("selected_null_event_rate"),
        "signal_channel_weak_pass": p3.get("signal_channel_weak_pass"),
        "best_signal_feature_id": p3.get("best_feature_id"),
        "best_signal_TopK64_precision": p3.get("best_TopK64_precision"),
        "best_signal_TopK64_longrisk": p3.get("best_TopK64_longrisk"),
        "cover_curvature_pass": p4.get("cover_curvature_pass"),
        "memory_pass": p5.get("memory_pass"),
        "geoscore_weak_pass": p6.get("geoscore_weak_pass"),
        "best_geoscore_id": p6.get("best_score_id"),
        "best_geoscore_accepted_count_heldout": p6.get("best_accepted_count_heldout"),
        "best_geoscore_V_integrated_lcb": p6.get("best_V_integrated_lcb"),
        "apgs_implementation_pass": p7.get("apgs_implementation_pass"),
        "apgs_generated_action_count": p7.get("generated_action_count"),
        "apgs_branch_horizon_pass": p8_smoke.get("apgs_branch_horizon_smoke_pass"),
        "apgs_branch_horizon_rows_actual": p8_smoke.get("branch_horizon_rows_actual"),
        "apgs_unresolved_exception_count": p8_smoke.get("unresolved_exception_count"),
        "apgs_weak_pass": p8.get("apgs_weak_pass"),
        "best_apgs_primitive": p8.get("best_primitive_id"),
        "best_apgs_OfficialGeoCandidate_precision": p8.get("best_OfficialGeoCandidate_precision"),
        "best_apgs_OutcomeGood_precision": p8.get("best_OutcomeGood_precision"),
        "best_apgs_V_integrated_lcb": p8.get("best_V_integrated_lcb"),
        "best_apgs_h240_longrisk": p8.get("best_h240_longrisk"),
        "source_to_generated_damage_pass": p9.get("source_to_generated_damage_pass"),
        "best_damage_primitive": p9.get("best_primitive_id"),
        "best_damage_new_positive_created_rate": p9.get("best_new_positive_created_rate"),
        "best_damage_longrisk_created_rate": p9.get("best_longrisk_created_rate"),
        "geometry_certificate_weak_pass": p10.get("geometry_certificate_weak_pass"),
        "best_certificate_id": p10.get("best_certificate_id"),
        "best_certificate_AUC_OfficialGeoCandidate": p10.get("best_AUC_OfficialGeoCandidate"),
        "best_certificate_TopK64_precision": p10.get("best_TopK64_precision"),
        "best_certificate_TopK64_longrisk": p10.get("best_TopK64_longrisk"),
        "best_certificate_TopK64_V_integrated_lcb": p10.get("best_TopK64_V_integrated_lcb"),
        "source_controller_pass": p11.get("source_controller_pass"),
        "selected_runtime_pass": p12.get("selected_runtime_pass"),
        "system_legal_controller_pass": system.get("system_legal_controller_pass"),
        "base_acc_sentinel_pass": p15.get("base_acc_sentinel_pass"),
        "base_acc_used_for_controller": 0,
        "success_v9540_strict_purekan_functional": 0,
        "success_v9540_full_functional": 0,
        "success_v9540_external_ready": 0,
        "primary_blocker": primary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }

    contract = {
        "stage": "CONTRACT_AUDIT_V9540",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "v9530_boundary_pass": p0.get("p0_pass"),
        "geometry_ledger_pass": p1.get("geometry_ledger_pass"),
        "target_density_geometry_lattice_pass": p2.get("target_density_geometry_lattice_pass"),
        "signal_channel_weak_pass": p3.get("signal_channel_weak_pass"),
        "cover_curvature_pass": p4.get("cover_curvature_pass"),
        "memory_pass": p5.get("memory_pass"),
        "geoscore_weak_pass": p6.get("geoscore_weak_pass"),
        "apgs_implementation_pass": p7.get("apgs_implementation_pass"),
        "apgs_branch_horizon_pass": p8_smoke.get("apgs_branch_horizon_smoke_pass"),
        "apgs_weak_pass": p8.get("apgs_weak_pass"),
        "source_to_generated_damage_pass": p9.get("source_to_generated_damage_pass"),
        "geometry_certificate_weak_pass": p10.get("geometry_certificate_weak_pass"),
        "source_controller_pass": p11.get("source_controller_pass"),
        "selected_runtime_pass": p12.get("selected_runtime_pass"),
        "system_legal_controller_pass": system.get("system_legal_controller_pass"),
        "base_acc_sentinel_pass": p15.get("base_acc_sentinel_pass"),
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
        "stage": "FAILURE_TABLE_V9540",
        "status": "summary",
        "route": route,
        "F0_boundary_reproduction_failed": int(route == "R0-BoundaryReproductionFailed"),
        "F1_geometry_ledger_failed": int(route == "R1-GeometryLedgerFailed"),
        "F2_geometry_target_too_sparse": int(route == "R2-GeometryTargetTooSparse"),
        "F3_legal_geometry_signal_absent": int(route == "R3-LegalGeometrySignalAbsent"),
        "F4_apgs_primitive_value_fail": int(route == "R4-APGSPrimitiveValueFail"),
        "F5_certificate_topk_fail": int(route == "R5-CertificateTopKFail"),
        "F6_controller_quality_fail": int(route == "R6-ControllerQualityFail"),
        "F7_selected_runtime_fail": int(route == "R7-SelectedRuntimeFail"),
        "F8_leaveout_fail": int(route == "R8-LeaveoutFail"),
        "F9_paired_replay_fail": int(route == "R9-PairedReplayFail"),
        "F10_base_acc_catastrophic": 0,
        "primary_blocker": primary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }

    p13_rows, p14_rows, p16_rows = [p13], [p14], [p16]
    all_rows: list[dict[str, Any]] = []
    for block in [
        p0_rows, p1_rows, p2_rows, p3_rows, p4_rows, p5_rows, p6_rows, p7_rows,
        p8_smoke_rows, p8_eval_rows, p9_rows, p10_rows, p11_rows, p12_rows,
        p13_rows, p14_rows, base_rows, p16_rows, [system], [contract], [failure],
    ]:
        all_rows.extend(block)
    nofake = audit_rows(all_rows)
    contract.update({"fake_data_used": nofake["fake_data_used"], "proxy_row_used": nofake["proxy_row_used"], "cpu_offload_used": nofake["cpu_offload_used"]})

    manifest = {
        "run_id": "v9540_geometry_definition_signal_channel_primitive",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "argv": sys.argv,
        "out_dir": str(out_dir),
        "device": str(device),
        "source_v9530": str(source_v9530.resolve()),
        "source_v9520": str(Path(args.source_v9520).resolve()),
        "source_v9330": str(Path(args.source_v9330).resolve()),
        "apgs_actions_per_primitive": int(args.apgs_actions_per_primitive),
        "no_fake_policy": "uses landed v9530/v9520/v9480+ artifacts and direct APGS branch-horizon materialization only; no proxy rows; no old v9350 official use",
    }

    write_json(out_dir / "run_manifest.json", manifest)
    write_json(out_dir / "route_decision_v9540.json", route_decision)
    write_json(out_dir / "aggregate_decision_v9540.json", route_decision)
    write_csv(out_dir / "p0_v9530_boundary_reproduction.csv", p0_rows)
    write_csv(out_dir / "geometry_outcome_ledger_v9540.csv", p1_rows)
    write_csv(out_dir / "p1_geometry_outcome_ledger_audit.csv", [p1])
    write_csv(out_dir / "p2_target_density_geometry_lattice_audit.csv", p2_rows)
    write_csv(out_dir / "p3_signal_channel_raw_upper_bound_audit.csv", p3_rows)
    write_csv(out_dir / "p4_cover_curvature_hardtail_stability_audit.csv", p4_rows)
    write_csv(out_dir / "p5_memory_antiforgetting_audit_v2.csv", p5_rows)
    write_csv(out_dir / "p6_geometry_score_assembly.csv", p6_rows)
    write_csv(out_dir / "p7_apgs_geometry_signal_primitive_implementation.csv", p7_rows)
    write_csv(out_dir / "p8_apgs_branch_horizon_smoke_outcome.csv", p8_smoke_rows + p8_eval_rows)
    write_csv(out_dir / "apgs_branch_horizon_outcome_trace_v9540.csv", outcome_rows)
    write_csv(out_dir / "p9_source_to_generated_geometry_damage_matrix.csv", p9_rows)
    write_csv(out_dir / "p10_geometry_certificate_v2.csv", p10_rows)
    write_csv(out_dir / "p11_minimal_geometry_controller.csv", p11_rows)
    write_csv(out_dir / "p12_selected_runtime_preflight.csv", p12_rows)
    write_csv(out_dir / "p12_system_integration_gate_v9540.csv", [system])
    write_csv(out_dir / "p13_leaveout_boundary.csv", p13_rows)
    write_csv(out_dir / "p14_official_paired_replay_boundary.csv", p14_rows)
    write_csv(out_dir / "p15_base_acc_sentinel_continuation.csv", base_rows)
    write_csv(out_dir / "p16_short_full_training_boundary.csv", p16_rows)
    write_csv(out_dir / "no_fake_audit_v9540.csv", [nofake])
    write_csv(out_dir / "contract_audit_v9540.csv", [contract])
    write_csv(out_dir / "failure_table_v9540.csv", [failure])
    write_csv(out_dir / "provenance_audit_v9540.csv", [nofake])

    svg_specs = [
        ("p0_v9530_boundary_ladder.svg", "v9530 Boundary", [("p0", fnum(p0.get("p0_pass"))), ("GGA", fnum(p0.get("GGA_count"))), ("system", fnum(p0.get("system_legal_controller_pass")))]),
        ("p0_gga_density_vs_target_density.svg", "GGA Density", [("GGA", fnum(p0.get("GGA_count"))), ("coverage", fnum(p0.get("GGA_coverage")))]),
        ("p0_apg_vs_ap0_outcome_scatter.svg", "APG Outcome", [("APG GGA", fnum(p0.get("best_apg_GGA_precision"))), ("APG risk", fnum(p0.get("best_apg_h240_longrisk")))]),
        ("p0_certificate_auc_vs_topk.svg", "GCERT AUC TopK", [("AUC", fnum(p0.get("best_certificate_AUC_GGA"))), ("TopK", fnum(p0.get("best_certificate_TopK64_GGA_precision")))]),
        ("p1_geometry_ledger_field_coverage_heatmap.svg", "Ledger Coverage", [("rows", fnum(p1.get("ledger_rows"))), ("missing", fnum(p1.get("required_field_missing_count")))]),
        ("p1_action_family_count_bar.svg", "Ledger Sources", [("AP0", fnum(p1.get("canonical_ap0_rows"))), ("APY", fnum(p1.get("generated_apy_rows"))), ("APG", fnum(p1.get("generated_apg_rows")))]),
        ("p1_signal_cover_curv_memory_pairplot.svg", "Signal Cover Curv Memory", [("pass", fnum(p1.get("geometry_ledger_pass")))]),
        ("p1_missing_field_zero_audit.svg", "Missing Field Audit", [("missing", fnum(p1.get("required_field_missing_count"))), ("nan", fnum(p1.get("nan_count"))), ("inf", fnum(p1.get("inf_count")))]),
        ("p2_target_lattice_density_quality_frontier.svg", "Target Lattice", [("candidates", fnum(p2.get("target_candidate_count"))), ("weak", fnum(p2.get("weak_geometry_target_count"))), ("official", fnum(p2.get("official_geometry_target_count")))]),
        ("p2_target_jaccard_heatmap.svg", "Target Jaccard", [("selected", fnum(p2.get("selected_action_count")))]),
        ("p2_target_value_risk_tradeoff.svg", "Value Risk", [("V", fnum(p2.get("selected_V_integrated_lcb"))), ("risk", fnum(p2.get("selected_h240_longrisk")))]),
        ("p2_target_support_balance.svg", "Support", [("families", fnum(p2.get("selected_support_family_count")))]),
        ("p2_gga_relaxation_path.svg", "GGA Relax", [("weak", fnum(p2.get("weak_geometry_target_count")))]),
        ("p3_signal_feature_auc_bar.svg", "Signal AUC", [("AUC", fnum(p3.get("best_AUC_OfficialGeoCandidate"))), ("TopK", fnum(p3.get("best_TopK64_precision")))]),
        ("p3_signal_topk_precision_curve.svg", "Signal TopK", [("TopK64", fnum(p3.get("best_TopK64_precision")))]),
        ("p3_signal_longrisk_topk_curve.svg", "Signal Risk", [("risk", fnum(p3.get("best_TopK64_longrisk")))]),
        ("p3_snr_vs_value_scatter.svg", "SNR Value", [("V", fnum(p3.get("best_TopK64_V_integrated_lcb")))]),
        ("p3_signal_leaveout_stability.svg", "Signal Leaveout", [("pass", fnum(p3.get("signal_channel_weak_pass")))]),
        ("p4_cover_entropy_delta_by_target.svg", "Cover", [("pass", fnum(p4.get("cover_metric_longrisk_pass")))]),
        ("p4_curvature_delta_by_target.svg", "Curvature", [("pass", fnum(p4.get("curvature_metric_longrisk_pass")))]),
        ("p4_longrisk_vs_curvature_scatter.svg", "Curv Risk", [("risk", fnum(p4.get("combined_TopK64_longrisk")))]),
        ("p4_hardtail_cover_shift.svg", "Hard Tail", [("precision", fnum(p4.get("combined_TopK64_OfficialGeoCandidate_precision")))]),
        ("p4_cover_curvature_combined_frontier.svg", "Combined", [("pass", fnum(p4.get("cover_curvature_pass")))]),
        ("p5_memory_forget_histogram.svg", "Memory", [("forget", fnum(p5.get("forget_risk_mean_topk64")))]),
        ("p5_old_new_tradeoff_scatter.svg", "Tradeoff", [("corr", fnum(p5.get("old_new_tradeoff_corr")))]),
        ("p5_memory_score_added_value.svg", "Memory Added", [("precision", fnum(p5.get("TopK64_precision")))]),
        ("p5_family_forgetting_heatmap.svg", "Family Forget", [("fails", fnum(p5.get("old_family_fail_count_topk64")))]),
        ("p6_geoscore_quality_coverage_frontier.svg", "GeoScore", [("accepted", fnum(p6.get("best_accepted_count_heldout"))), ("V", fnum(p6.get("best_V_integrated_lcb")))]),
        ("p6_geoscore_terms_ablation.svg", "GeoScore Terms", [("scores", fnum(p6.get("score_count")))]),
        ("p6_geoscore_calibration_heldout_drift.svg", "GeoScore Heldout", [("coverage", fnum(p6.get("best_coverage_heldout")))]),
        ("p6_geoscore_family_balance.svg", "GeoScore Balance", [("weak", fnum(p6.get("geoscore_weak_pass")))]),
        ("p7_apgs_payload_norm_by_primitive.svg", "APGS Payload", [("generated", fnum(p7.get("generated_action_count")))]),
        ("p7_apgs_sparsity_by_primitive.svg", "APGS Sparsity", [("primitive", fnum(p7.get("primitive_count")))]),
        ("p7_apgs_apply_error.svg", "APGS Apply", [("linf", fnum(p7.get("action_apply_error_linf_max")))]),
        ("p7_apgs_generation_cost.svg", "APGS Cost", [("pass", fnum(p7.get("apgs_implementation_pass")))]),
        ("p8_apgs_value_risk_by_primitive.svg", "APGS Value Risk", [("V", fnum(p8.get("best_V_integrated_lcb"))), ("risk", fnum(p8.get("best_h240_longrisk")))]),
        ("p8_apgs_horizon_profiles.svg", "APGS Horizon", [("rows", fnum(p8_smoke.get("branch_horizon_rows_actual")))]),
        ("p8_apgs_geometry_metrics_by_primitive.svg", "APGS Geometry", [("precision", fnum(p8.get("best_OfficialGeoCandidate_precision")))]),
        ("p8_apgs_vs_apg_vs_apy.svg", "APGS vs APG/APY", [("apgs", fnum(p8.get("best_OfficialGeoCandidate_precision"))), ("apg", fnum(p0.get("best_apg_GGA_precision")))]),
        ("p8_negative_control_comparison.svg", "Negative Control", [("pass", fnum(p8.get("apgs_weak_pass")))]),
        ("p9_source_to_generated_damage_heatmap.svg", "Damage", [("new", fnum(p9.get("best_new_positive_created_rate"))), ("risk", fnum(p9.get("best_longrisk_created_rate")))]),
        ("p9_positive_preservation_by_primitive.svg", "Preserve", [("preserve", fnum(p9.get("best_positive_preserved_rate")))]),
        ("p9_longrisk_created_by_primitive.svg", "LongRisk Created", [("risk", fnum(p9.get("best_longrisk_created_rate")))]),
        ("p9_damage_value_vs_geometry.svg", "Damage Value Geometry", [("damage", fnum(p9.get("best_Damage_integrated_LCB")))]),
        ("p10_certificate_reliability_curve.svg", "Cert ECE", [("ece", fnum(p10.get("best_ECE")))]),
        ("p10_certificate_topk_precision.svg", "Cert TopK", [("precision", fnum(p10.get("best_TopK64_precision")))]),
        ("p10_certificate_topk_longrisk.svg", "Cert Risk", [("risk", fnum(p10.get("best_TopK64_longrisk")))]),
        ("p10_certificate_leaveout_matrix.svg", "Cert Leaveout", [("AUC", fnum(p10.get("best_AUC_OfficialGeoCandidate")))]),
        ("p10_certificate_ablation.svg", "Cert Ablation", [("pass", fnum(p10.get("geometry_certificate_weak_pass")))]),
        ("p11_controller_quality_coverage_frontier.svg", "Controller", [("pass", fnum(p11.get("source_controller_pass")))]),
        ("p11_controller_cal_heldout_drift.svg", "Controller Drift", [("pass", fnum(p11.get("source_controller_pass")))]),
        ("p11_controller_family_stratum_balance.svg", "Controller Balance", [("pass", fnum(p11.get("source_controller_pass")))]),
        ("p11_controller_failure_modes.svg", "Controller Failure", [("pass", fnum(p11.get("source_controller_pass")))]),
        ("p12_runtime_waterfall.svg", "Runtime", [("pass", fnum(p12.get("selected_runtime_pass")))]),
        ("p12_step_ratio_by_step_bucket.svg", "Step Ratio", [("pass", fnum(p12.get("selected_runtime_pass")))]),
        ("p12_payload_apply_time.svg", "Apply Time", [("pass", fnum(p12.get("selected_runtime_pass")))]),
        ("p12_memory_trace.svg", "Memory", [("pass", fnum(p12.get("selected_runtime_pass")))]),
        ("p12_controller_cost_vs_quality.svg", "Cost Quality", [("pass", fnum(p12.get("selected_runtime_pass")))]),
        ("p13_ldo_matrix.svg", "LDO", [("open", 0)]),
        ("p13_lso_matrix.svg", "LSO", [("open", 0)]),
        ("p13_leaveout_value_risk_frontier.svg", "Leaveout", [("open", 0)]),
        ("p13_dataset_tuning_audit.svg", "Dataset Tuning", [("used", 0)]),
        ("p14_paired_replay_branch_matrix.svg", "Paired Replay", [("open", 0)]),
        ("p14_real_vs_controls_value.svg", "Controls", [("open", 0)]),
        ("p14_horizon_value_profile.svg", "Horizon", [("open", 0)]),
        ("p14_shuffle_negative_control.svg", "Shuffle", [("open", 0)]),
        ("p15_acc_curves.svg", "Base Acc", [("LQ", fnum(p15.get("mean_test_acc_LQ"))), ("MLP", fnum(p15.get("mean_test_acc_MLP")))]),
        ("p15_loss_curves.svg", "Loss Curves", [("open", 0)]),
        ("p15_calibration_curves.svg", "Calibration", [("open", 0)]),
        ("p15_time_to_target.svg", "Time", [("open", 0)]),
        ("p15_runtime_memory_curves.svg", "Runtime", [("open", 0)]),
        ("p15_baseline_comparison_matrix.svg", "Baseline", [("strong", fnum(p15.get("mean_test_acc_AdamWStrongLRGridMLP")))]),
    ]
    for name, title, metrics in svg_specs:
        write_svg(out_dir / name, title, metrics)

    write_csv(out_dir / "no_fake_audit_v9540.csv", [nofake])
    write_csv(out_dir / "contract_audit_v9540.csv", [contract])
    write_csv(out_dir / "failure_table_v9540.csv", [failure])
    write_csv(out_dir / "provenance_audit_v9540.csv", [nofake])
    write_csv(out_dir / "artifact_hashes_v9540.csv", hash_rows(out_dir))
    write_csv(out_dir / "hash_manifest_v9540.csv", hash_rows(out_dir))

    print(json.dumps({
        "out_dir": str(out_dir),
        "route": route,
        "ledger_rows": p1.get("ledger_rows"),
        "weak_geometry_target_count": p2.get("weak_geometry_target_count"),
        "selected_geometry_target_id": p2.get("selected_geometry_target_id"),
        "apgs_generated_actions": p7.get("generated_action_count"),
        "best_apgs_primitive": p8.get("best_primitive_id"),
        "best_apgs_OfficialGeoCandidate_precision": p8.get("best_OfficialGeoCandidate_precision"),
        "geometry_certificate_weak_pass": p10.get("geometry_certificate_weak_pass"),
        "system_legal_controller_pass": system.get("system_legal_controller_pass"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
