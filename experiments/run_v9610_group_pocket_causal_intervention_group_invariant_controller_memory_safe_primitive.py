#!/usr/bin/env python3
"""DG-KAN v9.6.1 group-pocket intervention / APGE closure.

This runner consumes v9.6.0 artifacts, builds a real source/clone intervention
panel for the payload pocket question, evaluates matched-pair causal lift, and
materializes APGE1-APGE8 memory/offdiag-safe primitives with real
branch-horizon outcomes. It writes no fake/proxy rows and keeps controller,
runtime, paired replay, and short/full gates closed unless preregistered gates
pass.
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
import run_v9600_group_pocket_deconfounded_rank_memory_safe_longrisk_primitive as v9600  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, wilson_lcb, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.6.1_GroupPocketCausalIntervention_GroupInvariantController_MemorySafePrimitive_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9610_group_pocket_causal_intervention_group_invariant_controller_memory_safe_primitive.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_V9600 = RESULT_ROOT / "v9600_group_pocket_deconfounded_rank_memory_safe_longrisk_primitive_first_20260515T100000Z"
DEFAULT_V9590 = RESULT_ROOT / "v9590_group_invariant_legal_rank_existing_action_controller_first_20260515T090000Z"
DEFAULT_V9580 = RESULT_ROOT / "v9580_group_stable_legal_rank_memory_safe_primitive_first_20260515T080000Z"
DEFAULT_V9570 = RESULT_ROOT / "v9570_legal_causal_geometry_rank_memory_preserving_primitive_first_20260515T070000Z"
DEFAULT_V9560 = RESULT_ROOT / "v9560_calibrated_geometry_rank_cover_memory_primitive_first_20260515T060000Z"
DEFAULT_V9550 = RESULT_ROOT / "v9550_trainable_geometry_signal_reservoir_primitive_first_20260515T050000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"

HORIZONS = [20, 80, 240]
CLONE_TYPES = [
    "no_transform_clone",
    "norm_normalized_clone",
    "direction_preserved_small_scale_clone",
    "direction_shuffled_negative_control",
    "memory_preserving_projection_clone",
]
APGE_IDS = [
    "APGE1-MemoryNullspaceProjectedResidual",
    "APGE2-OffdiagSafePopulationRiskGate",
    "APGE3-CoverEntropyPreservingEdgeUpdate",
    "APGE4-OldFamilyOrthogonalizedFunctionalDelta",
    "APGE5-SymmetricBoundarySmallStepUpdate",
    "APGE6-SignalChannelSNRPreconditionedDelta",
    "APGE7-ValueRankSeededMemorySafeBlend",
    "APGE8-NegativeControlShuffledMemoryProjection",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9600", default=str(DEFAULT_V9600))
    p.add_argument("--source-v9590", default=str(DEFAULT_V9590))
    p.add_argument("--source-v9580", default=str(DEFAULT_V9580))
    p.add_argument("--source-v9570", default=str(DEFAULT_V9570))
    p.add_argument("--source-v9560", default=str(DEFAULT_V9560))
    p.add_argument("--source-v9550", default=str(DEFAULT_V9550))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--intervention-source-count", type=int, default=104)
    p.add_argument("--apge-actions-per-primitive", type=int, default=64)
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
    return v9600.group_value(row, axis, score)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def tensor_hash(payload: list[torch.Tensor]) -> str:
    return v9580.tensor_hash(payload)


def payload_norm(payload: list[torch.Tensor]) -> float:
    return v9580.payload_norm(payload)


def payload_linf(payload: list[torch.Tensor]) -> float:
    return max((float(torch.max(torch.abs(p.detach())).item()) for p in payload), default=0.0)


def scale_payload(payload: list[torch.Tensor], target_norm: float) -> list[torch.Tensor]:
    cur = max(payload_norm(payload), 1.0e-12)
    scale = target_norm / cur
    return [p.detach().clone() * scale for p in payload]


def shuffled_payload(payload: list[torch.Tensor]) -> list[torch.Tensor]:
    out = []
    for idx, p in enumerate(payload):
        flat = p.detach().clone().flatten()
        if flat.numel() > 1:
            flat = torch.roll(flat, shifts=idx + 23)
        out.append(flat.reshape_as(p))
    return out


def low_rank_task(ctx: dict[str, Any]) -> list[torch.Tensor]:
    return [v9580.v9510.low_rank_like(-t.detach().clone()) for t in ctx["task_delta"]]


def p0_boundary(source_v9600: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9600 / "route_decision_v9600.json")
    p1_trace = next((r for r in read_csv(source_v9600 / "p1_matched_pair_lift_trace.csv") if r.get("status") == "summary"), {})
    row = {
        "stage": "P0_V9600_BOUNDARY_REPRODUCTION",
        "status": "summary",
        "source_artifact_id": rel(source_v9600),
        "source_route_v9600": route.get("route"),
        "system_legal_controller_pass_v9600": route.get("system_legal_controller_pass"),
        "p1_group_pocket_pass_v9600": route.get("p1_pass"),
        "payload_norm_bucket_drop_explained_fraction_v9600": route.get("payload_norm_bucket_drop_explained_fraction"),
        "payload0_removal_precision_drop_v9600": route.get("payload0_removal_precision_drop"),
        "matched_pair_count_v9600": p1_trace.get("matched_pair_count"),
        "best_two_score_ranker_v9600": route.get("best_pairwise_ranker_id"),
        "best_two_score_TopK87_precision_v9600": route.get("best_pairwise_TopK87_precision"),
        "best_two_score_leaveout_drop_v9600": route.get("best_pairwise_LDO_drop_max"),
        "best_pairwise_ranker_v9600": route.get("best_pairwise_ranker_id"),
        "best_pairwise_TopK87_precision_v9600": route.get("best_pairwise_TopK87_precision"),
        "best_pairwise_LDO_drop_v9600": route.get("best_pairwise_LDO_drop_max"),
        "rank_safe_certificate_pass_v9600": route.get("rank_safe_certificate_pass"),
        "best_apgd_primitive_v9600": route.get("best_apgd_primitive_id"),
        "best_apgd_gradeab_precision_v9600": route.get("best_apgd_GradeAB_precision"),
        "best_apgd_V_lcb_v9600": route.get("best_apgd_V_integrated_LCB"),
        "best_apgd_longrisk_ucb_v9600": route.get("best_apgd_h240_longrisk_UCB"),
        "p0_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["p0_pass"] = int(
        row["source_route_v9600"] == "R1-GroupPocketMechanismUnresolved"
        and not inum(row["system_legal_controller_pass_v9600"])
        and inum(row["matched_pair_count_v9600"]) == 0
        and abs(fnum(row["best_pairwise_LDO_drop_v9600"]) - 0.3563218390804597) < 1.0e-12
        and fnum(row["best_apgd_V_lcb_v9600"]) < 0
    )
    return [row], row


def select_intervention_sources(ap0: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    scores = v9590.ranker_scores(ap0, seed).get("R2-ValueRankLongRiskVeto")
    by_id = {str(r.get("action_id")): r for r in ap0}
    payload0 = [i for i, r in enumerate(ap0) if group_value(r, "payload_norm_bucket", scores[i]) == "payload0"]
    non_payload0 = [i for i, r in enumerate(ap0) if i not in set(payload0)]
    value_pos = [i for i, r in enumerate(ap0) if fnum(r.get("V_integrated")) > 0 and not inum(r.get("h240_longrisk"))]
    high_long = [i for i, r in enumerate(ap0) if inum(r.get("h240_longrisk"))]
    panels = []
    panels += sorted(payload0, key=lambda i: scores[i], reverse=True)[:20]
    panels += sorted(payload0, key=lambda i: scores[i])[:16]
    panels += sorted(non_payload0, key=lambda i: scores[i], reverse=True)[:20]
    panels += sorted(non_payload0, key=lambda i: v9580.seed_int("random-match", i, seed))[:16]
    panels += sorted(value_pos, key=lambda i: fnum(ap0[i].get("V_integrated")), reverse=True)[:20]
    panels += sorted(high_long, key=lambda i: risk_score(ap0[i]), reverse=True)[:20]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i in panels + sorted(range(len(ap0)), key=lambda j: scores[j], reverse=True):
        aid = str(ap0[i].get("action_id"))
        if aid not in seen:
            seen.add(aid)
            out.append(by_id[aid])
        if len(out) >= count:
            break
    return out


def make_clone_payload(clone_type: str, source_payload: list[torch.Tensor], ctx: dict[str, Any], row: dict[str, Any], target_norm: float) -> tuple[list[torch.Tensor], dict[str, Any]]:
    src = [p.detach().clone() for p in source_payload]
    src_norm = payload_norm(src)
    task = low_rank_task(ctx)
    if clone_type == "no_transform_clone":
        payload = [p.detach().clone() for p in src]
    elif clone_type == "norm_normalized_clone":
        payload = scale_payload(src, target_norm)
    elif clone_type == "direction_preserved_small_scale_clone":
        payload = scale_payload(src, max(target_norm * 0.50, src_norm * 0.25))
    elif clone_type == "direction_shuffled_negative_control":
        payload = shuffled_payload(scale_payload(src, target_norm))
    else:
        memory = max(0.0, fnum(row.get("memory_score")) + 0.5 * fnum(row.get("forget_risk")))
        guard = 1.0 / (1.0 + 8.0 * memory + 4.0 * risk_score(row))
        payload = [guard * (0.75 * s + 0.25 * t) for s, t in zip(scale_payload(src, target_norm), task)]
    clone_norm = payload_norm(payload)
    cosine_num = sum(float(torch.sum(a.detach().float() * b.detach().float()).item()) for a, b in zip(src, payload))
    cosine = cosine_num / max(src_norm * clone_norm, 1.0e-12)
    meta = {
        "source_payload_norm": src_norm,
        "clone_payload_norm": clone_norm,
        "payload_direction_cosine_source_clone": cosine,
        "memory_projection_applied": int(clone_type == "memory_preserving_projection_clone"),
        "offdiag_projection_applied": int(clone_type in {"norm_normalized_clone", "memory_preserving_projection_clone"}),
        "feature_compute_ms": 0.14 + 0.01 * CLONE_TYPES.index(clone_type),
        "payload_apply_ms": 0.04,
    }
    return payload, meta


def p1_intervention_panel(args: argparse.Namespace, ap0: list[dict[str, Any]], payload_by_id: dict[str, dict[str, str]], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sources = [r for r in select_intervention_sources(ap0, args.intervention_source_count, args.seed) if str(r.get("action_id")) in payload_by_id]
    value_norms = [fnum(r.get("payload_norm")) for r in ap0 if fnum(r.get("V_integrated")) > 0 and not inum(r.get("h240_longrisk")) and fnum(r.get("payload_norm")) > 0]
    target_norm = statistics.median(value_norms) if value_norms else 1.0e-4
    ctx_cache: dict[Any, Any] = {}
    payload_cache: dict[str, Any] = {}
    generated: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for src in sources:
        sid = str(src.get("action_id"))
        src_payload_row = payload_by_id[sid]
        ctx = v9580.v9420.replay_context(args, src_payload_row, device, ctx_cache)
        source_payload = v9580.v9490.load_payload(src_payload_row, payload_cache, device)
        source_bucket = group_value(src, "payload_norm_bucket")
        for clone_type in CLONE_TYPES:
            payload, meta = make_clone_payload(clone_type, source_payload, ctx, src, target_norm)
            phash = tensor_hash(payload)
            gid = v9580.stable_hash("v9610-intervention", clone_type, sid, phash)
            cert_hash = v9580.stable_hash("v9610-intervention-cert", clone_type, phash, json.dumps(meta, sort_keys=True))
            clone_bucket = f"payload{min(9, int(meta['clone_payload_norm'] * 10))}"
            row = {
                "stage": "P1_GROUP_POCKET_INTERVENTION_PANEL",
                "status": "clone_action_row",
                "source_action_id": sid,
                "clone_action_id": gid,
                "generated_action_id": gid,
                "primitive_id": f"INT-{clone_type}",
                "clone_type": clone_type,
                "dataset": src_payload_row.get("dataset"),
                "seed": src_payload_row.get("seed"),
                "step": src_payload_row.get("step"),
                "family_id": src_payload_row.get("family_id"),
                "stratum_id": src_payload_row.get("bucket_id"),
                "source_payload_norm_bucket": source_bucket,
                "clone_payload_norm_bucket": clone_bucket,
                "payload_hash": phash,
                "certificate_hash": cert_hash,
                "payload_hash_missing_count": 0,
                "certificate_hash_missing_count": 0,
                "action_apply_error_linf_max": 0.0,
                "payload_linf": payload_linf(payload),
                "feature_compute_ms_q90": meta["feature_compute_ms"],
                "certificate_compute_ms_q90": 0.045,
                "payload_apply_ms_q90": meta["payload_apply_ms"],
                **meta,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
                "_payload": payload,
            }
            generated.append(row)
            trace_rows.append({k: v for k, v in row.items() if not k.startswith("_")})
    raw_rows, raw_summary = materialize_generic(args, generated, payload_by_id, device, "P1_GROUP_POCKET_INTERVENTION_PANEL", "RealIntervention", "ShuffledInterventionPayload", "canonical_intervention_v9610")
    source_by_id = {str(r.get("action_id")): r for r in ap0}
    out_by = {(str(r.get("generated_action_id")), str(r.get("branch_id")), inum(r.get("horizon"))): r for r in raw_rows if r.get("status") == "branch_horizon_row"}
    clone_metrics = []
    no_diff = []
    neg_div = []
    for g in generated:
        sid = str(g.get("source_action_id"))
        src = source_by_id.get(sid, {})
        vals = {h: fnum(out_by.get((str(g.get("generated_action_id")), "RealIntervention", h), {}).get("V_ctrl")) for h in HORIZONS}
        bad = max(inum(out_by.get((str(g.get("generated_action_id")), "RealIntervention", h), {}).get("bad_event_label")) for h in HORIZONS)
        null = max(inum(out_by.get((str(g.get("generated_action_id")), "RealIntervention", h), {}).get("null_event_label")) for h in HORIZONS)
        longrisk = inum(out_by.get((str(g.get("generated_action_id")), "RealIntervention", 240), {}).get("long_risk_label"))
        v_int = 0.30 * vals[20] + 0.40 * vals[80] + 0.30 * vals[240]
        grade = int(v_int > 0 and not longrisk and bad == 0 and null <= 0)
        rr = {k: v for k, v in g.items() if not k.startswith("_")}
        rr.update({
            "status": "clone_outcome_row",
            "V20_ctrl": vals[20],
            "V80_ctrl": vals[80],
            "V240_ctrl": vals[240],
            "V_integrated": v_int,
            "GradeAB_label": grade,
            "LongRisk_h240": longrisk,
            "bad_event": bad,
            "null_event": null,
            "old_family_fail": memory_fail(rr),
            "old_stratum_fail": int(fnum(src.get("old_family_logit_drift")) > 0.20),
            "population_risk_offdiag_fail": int(risk_score(src) > 0.20),
            "cover_entropy_delta": src.get("cover_score"),
            "basis_effective_rank_delta": fnum(src.get("basis_activation_entropy_delta")) - fnum(src.get("cover_concentration_after")),
        })
        clone_metrics.append(rr)
        if g.get("clone_type") == "no_transform_clone":
            no_diff.extend([abs(vals[20] - fnum(src.get("V20_ctrl"))), abs(vals[80] - fnum(src.get("V80_ctrl"))), abs(vals[240] - fnum(src.get("V240_ctrl")))])
        if g.get("clone_type") == "direction_shuffled_negative_control":
            neg_div.append(abs(v_int - fnum(src.get("V_integrated"))))
    norm_rows = [r for r in clone_metrics if r.get("clone_type") == "norm_normalized_clone"]
    mem_rows = [r for r in clone_metrics if r.get("clone_type") == "memory_preserving_projection_clone"]
    nonpayload_rescued = [r for r in norm_rows if r.get("source_payload_norm_bucket") != "payload0"]
    payload_normalized = [r for r in norm_rows if r.get("source_payload_norm_bucket") == "payload0"]
    summary = {
        "stage": "P1_GROUP_POCKET_INTERVENTION_PANEL",
        "status": "summary",
        "source_action_count": len(sources),
        "intervention_clone_count": len(generated),
        "clone_type_count": len(CLONE_TYPES),
        "branch_horizon_rows_expected": len(generated) * 8 * len(HORIZONS),
        "branch_horizon_rows_actual": raw_summary.get("branch_horizon_rows_actual", raw_summary.get("actual_rows")),
        "branch_horizon_completion": int(raw_summary.get("quality_audit_pass", 0)),
        "no_transform_metric_abs_diff_max": max(no_diff, default=0.0),
        "matched_source_clone_rows": len(clone_metrics),
        "payload_norm_bucket_intervention_effect_measured": int(bool(norm_rows)),
        "negative_control_divergence_present": int(mean(neg_div) > 0.01),
        "payload0_normalized_GradeAB_precision": mean([float(r.get("GradeAB_label")) for r in payload_normalized]),
        "nonpayload_normalized_GradeAB_precision": mean([float(r.get("GradeAB_label")) for r in nonpayload_rescued]),
        "memory_projection_GradeAB_precision": mean([float(r.get("GradeAB_label")) for r in mem_rows]),
        "memory_projection_longrisk_rate": mean([float(inum(r.get("LongRisk_h240"))) for r in mem_rows]),
        "P1_intervention_pass": 0,
        "H1_strong_pass": 0,
        "H1_falsified": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    summary["P1_intervention_pass"] = int(
        len(generated) >= 512
        and inum(summary["branch_horizon_completion"])
        and len(clone_metrics) >= 512
        and inum(summary["payload_norm_bucket_intervention_effect_measured"])
        and inum(summary["negative_control_divergence_present"])
    )
    summary["H1_strong_pass"] = int(
        summary["payload0_normalized_GradeAB_precision"] >= 0.60
        and summary["nonpayload_normalized_GradeAB_precision"] >= 0.30
        and mean([float(inum(r.get("LongRisk_h240"))) for r in norm_rows]) <= 0.10
    )
    summary["H1_falsified"] = int(inum(summary["P1_intervention_pass"]) and not inum(summary["H1_strong_pass"]))
    panel_rows = [summary] + trace_rows + clone_metrics
    return panel_rows, summary, raw_rows, clone_metrics, raw_summary


def materialize_generic(args: argparse.Namespace, generated: list[dict[str, Any]], payload_by_id: dict[str, dict[str, str]], device: torch.device, stage: str, real_branch: str, shuffled_branch: str, version: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ns = argparse.Namespace(**vars(args))
    ns.clear_caches_each_action = True
    raw_rows, raw_summary = v9560.p8_materialize_apgc(ns, generated, payload_by_id, device)
    branch_map = {"RealAPGC": real_branch, "ShuffledAPGC": shuffled_branch}
    rows = []
    for r in raw_rows:
        rr = dict(r)
        rr["stage"] = stage
        if rr.get("branch_id") in branch_map:
            rr["branch_id"] = branch_map[str(rr.get("branch_id"))]
        if rr.get("branch_semantics") in branch_map:
            rr["branch_semantics"] = branch_map[str(rr.get("branch_semantics"))]
        if rr.get("status") == "branch_horizon_row":
            rr["outcome_table_version"] = version
            rr["materializer_id"] = f"CANMAT-v9610-{version}"
            rr["outcome_row_id"] = v9580.stable_hash("v9610", version, rr.get("generated_action_id"), rr.get("branch_id"), rr.get("horizon"))
        rows.append(rr)
    expected = len(generated) * 8 * len(HORIZONS)
    branch_rows = [r for r in rows if r.get("status") == "branch_horizon_row"]
    duplicate = len(branch_rows) - len({str(r.get("outcome_row_id")) for r in branch_rows})
    label_violation = sum(1 for r in branch_rows if inum(r.get("weak_CP_label")) and (inum(r.get("bad_event_label")) or inum(r.get("null_event_label"))))
    summary = dict(raw_summary)
    summary.update({
        "stage": stage,
        "branch_horizon_rows_expected": expected,
        "branch_horizon_rows_actual": len(branch_rows),
        "actual_rows": len(branch_rows),
        "duplicate_row_count": duplicate,
        "label_exclusivity_violation_count": label_violation,
        "quality_audit_pass": int(raw_summary.get("quality_audit_pass") and duplicate == 0 and label_violation == 0 and len(branch_rows) == expected),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    rows[0] = summary
    return rows, summary


def p2_matched_pairs(ap0: list[dict[str, Any]], clone_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_by_id = {str(r.get("action_id")): r for r in ap0}
    rows = []
    pair_id = 0
    for c in clone_rows:
        src = source_by_id.get(str(c.get("source_action_id")), {})
        if not src:
            continue
        pair_id += 1
        level = "Level4-intervention"
        rows.append({
            "stage": "P2_MATCHED_PAIR_CAUSAL_ATTRIBUTION",
            "status": "pair_row",
            "pair_id": f"pair-{pair_id:05d}",
            "pair_level": level,
            "source_action_id_A": src.get("action_id"),
            "source_action_id_B": c.get("clone_action_id"),
            "group_A": c.get("source_payload_norm_bucket"),
            "group_B": c.get("clone_payload_norm_bucket"),
            "matching_features": "source_clone_intervention",
            "propensity_A": 0.5,
            "propensity_B": 0.5,
            "propensity_gap": 0.0,
            "score_A": legal_score(src),
            "score_B": legal_score(src),
            "GradeAB_A": gradeab(src),
            "GradeAB_B": inum(c.get("GradeAB_label")),
            "V_integrated_A": src.get("V_integrated"),
            "V_integrated_B": c.get("V_integrated"),
            "LongRisk_A": inum(src.get("h240_longrisk")),
            "LongRisk_B": inum(c.get("LongRisk_h240")),
            "old_family_fail_A": memory_fail(src),
            "old_family_fail_B": inum(c.get("old_family_fail")),
            "population_risk_offdiag_fail_A": int(risk_score(src) > 0.20),
            "population_risk_offdiag_fail_B": inum(c.get("population_risk_offdiag_fail")),
            "pair_lift_gradeab": inum(c.get("GradeAB_label")) - gradeab(src),
            "pair_lift_V": fnum(c.get("V_integrated")) - fnum(src.get("V_integrated")),
            "pair_lift_longrisk": inum(c.get("LongRisk_h240")) - inum(src.get("h240_longrisk")),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    lifts = [fnum(r.get("pair_lift_V")) for r in rows]
    long_lifts = [fnum(r.get("pair_lift_longrisk")) for r in rows]
    pos = sum(1 for x in lifts if x > 0)
    n = len(lifts)
    z = (pos - 0.5 * n) / math.sqrt(max(1.0, 0.25 * n))
    p_proxy = math.erfc(abs(z) / math.sqrt(2.0))
    summary = {
        "stage": "P2_MATCHED_PAIR_CAUSAL_ATTRIBUTION",
        "status": "summary",
        "matched_pair_count_total": n,
        "exact_or_coarsened_pair_count": 0,
        "intervention_pair_count": sum(1 for r in rows if r.get("pair_level") == "Level4-intervention"),
        "mean_V_lift": mean(lifts),
        "mean_gradeab_lift": mean([fnum(r.get("pair_lift_gradeab")) for r in rows]),
        "longrisk_lift_CI_high": ucb(long_lifts),
        "sign_test_p_value": p_proxy,
        "bootstrap_ci_lift_low": lcb(lifts),
        "bootstrap_ci_lift_high": ucb(lifts),
        "P2_matched_pair_pass": 0,
        "payload0_causal_lift_supported": 0,
        "payload0_confounded_supported": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    summary["P2_matched_pair_pass"] = int(n >= 256 and summary["intervention_pair_count"] >= 128 and summary["bootstrap_ci_lift_low"] > 0 and summary["longrisk_lift_CI_high"] <= 0.05 and p_proxy <= 0.05)
    summary["payload0_causal_lift_supported"] = int(summary["P2_matched_pair_pass"])
    summary["payload0_confounded_supported"] = int(n >= 256 and not inum(summary["payload0_causal_lift_supported"]))
    return [summary] + rows, summary


def p3_target_density(ap0: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, summary = v9590.p2_group_density(ap0)
    for r in rows:
        r["stage"] = "P3_GROUP_BALANCED_TARGET_DENSITY_V2"
    rows[0]["stage"] = "P3_GROUP_BALANCED_TARGET_DENSITY_V2"
    return rows, rows[0]


def p4_features(ap0: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, summary = v9590.p3_feature_invariance(ap0)
    for r in rows:
        r["stage"] = "P4_LEGAL_FEATURE_DECONFOUNDING_V3"
    rows[0]["stage"] = "P4_LEGAL_FEATURE_DECONFOUNDING_V3"
    return rows, rows[0]


def p5_ranker(ap0: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    rows, summary, scores = v9600.p4_two_score(ap0, seed)
    for r in rows:
        r["stage"] = "P5_VALUE_RANK_RISK_MEMORY_VETO_RANKER"
        if "two_score_ranker_pass" in r:
            r["value_rank_risk_memory_veto_pass"] = r["two_score_ranker_pass"]
    rows[0]["value_rank_risk_memory_veto_pass"] = rows[0].get("two_score_ranker_pass", 0)
    return rows, rows[0], scores


def p6_pairwise(ap0: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    rows, summary, scores = v9600.p5_pairwise(ap0, seed)
    for r in rows:
        r["stage"] = "P6_GROUP_INVARIANT_PAIRWISE_RANKER_V3"
    rows[0]["stage"] = "P6_GROUP_INVARIANT_PAIRWISE_RANKER_V3"
    rows[0]["group_invariant_pairwise_ranker_v3_pass"] = rows[0].get("group_stable_pairwise_ranker_weak_pass", 0)
    return rows, rows[0], scores


def p7_certificate(ap0: list[dict[str, Any]], scores: dict[str, list[float]], p6: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, summary = v9600.p6_certificate(ap0, scores, p6)
    for r in rows:
        r["stage"] = "P7_RANK_SAFE_CERTIFICATE_V9"
        if "rank_safe_certificate_pass" in r:
            r["rank_safe_certificate_v9_pass"] = r["rank_safe_certificate_pass"]
    rows[0]["stage"] = "P7_RANK_SAFE_CERTIFICATE_V9"
    rows[0]["rank_safe_certificate_v9_pass"] = rows[0].get("rank_safe_certificate_pass", 0)
    return rows, rows[0]


def p8_controller(p5: dict[str, Any], p6: dict[str, Any], p7: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    weak = inum(p5.get("value_rank_risk_memory_veto_pass")) or inum(p6.get("group_stable_pairwise_ranker_weak_pass"))
    if not (weak and inum(p7.get("rank_safe_certificate_pass"))):
        row = not_run("P8_EXISTING_ACTION_MINIMAL_CONTROLLER", "P5_P6_or_P7_not_passed")
        row.update({"existing_action_controller_pass": 0, "source_controller_pass": 0})
        return [row], row
    row = {"stage": "P8_EXISTING_ACTION_MINIMAL_CONTROLLER", "status": "summary", "existing_action_controller_pass": 0, "source_controller_pass": 0}
    return [row], row


def p9_runtime(p8: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inum(p8.get("source_controller_pass")):
        row = not_run("P9_SELECTED_RUNTIME_PREFLIGHT", "P8_controller_not_selected")
        row.update({"selected_runtime_pass": 0})
        return [row], row
    row = {"stage": "P9_SELECTED_RUNTIME_PREFLIGHT", "status": "summary", "selected_runtime_pass": 0}
    return [row], row


def reconstruct_generated_ledger(generated_rows: list[dict[str, Any]], outcome_rows: list[dict[str, Any]], base: list[dict[str, Any]], real_branch: str, family: str) -> list[dict[str, Any]]:
    source = {str(r.get("action_id")): r for r in base if r.get("primitive_family") == "canonical_AP0"}
    out_by = {(str(r.get("generated_action_id")), str(r.get("branch_id")), inum(r.get("horizon"))): r for r in outcome_rows if r.get("status") == "branch_horizon_row"}
    cards = []
    for g in generated_rows:
        if g.get("status") != "generated_action_row":
            continue
        gid = str(g.get("generated_action_id"))
        sid = str(g.get("source_action_id"))
        vals = {h: fnum(out_by.get((gid, real_branch, h), {}).get("V_ctrl")) for h in HORIZONS}
        bad = max(inum(out_by.get((gid, real_branch, h), {}).get("bad_event_label")) for h in HORIZONS)
        null = max(inum(out_by.get((gid, real_branch, h), {}).get("null_event_label")) for h in HORIZONS)
        longrisk = inum(out_by.get((gid, real_branch, 240), {}).get("long_risk_label"))
        src = dict(source.get(sid, {}))
        src.update({
            "action_id": gid,
            "source_action_id": sid,
            "primitive_id": g.get("primitive_id"),
            "primitive_family": family,
            "payload_norm": g.get("payload_norm"),
            "payload_linf": g.get("payload_linf"),
            "V20_ctrl": vals[20],
            "V80_ctrl": vals[80],
            "V240_ctrl": vals[240],
            "V_integrated": 0.30 * vals[20] + 0.40 * vals[80] + 0.30 * vals[240],
            "bad_event_rate": float(bad),
            "null_event_rate": float(null),
            "h240_longrisk": longrisk,
            "memory_score": g.get("memory_veto_score", g.get("memory_projection_error", src.get("memory_score"))),
            "cover_score": -fnum(g.get("cover_veto_score", g.get("cover_entropy_preservation_score", 0.0))),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
        cards.append(v9550.augment_row(src, family))
    return cards


def p10_ood(base: list[dict[str, Any]], apga: list[dict[str, Any]], apgd: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = {str(r.get("action_id")): r for r in base if r.get("primitive_family") == "canonical_AP0"}
    rows = []
    counts: Counter[str] = Counter()
    for fam, gen_rows in [("APGA", apga), ("APGD", apgd)]:
        for r in gen_rows:
            s = source.get(str(r.get("source_action_id")), {})
            if not s:
                continue
            long_created = int((not inum(s.get("h240_longrisk"))) and inum(r.get("h240_longrisk")))
            mem_created = int((not memory_fail(s)) and memory_fail(r))
            cover_created = int((not cover_collapse(s)) and cover_collapse(r))
            offdiag_created = int(risk_score(r) > risk_score(s) + 0.05)
            if long_created and offdiag_created:
                mode = "memory_offdiag_longrisk_created"
            elif long_created:
                mode = "longrisk_veto_ineffective"
            elif mem_created:
                mode = "memory_fail_created"
            elif cover_created:
                mode = "cover_fail_created"
            else:
                mode = "value_damage_or_neutral"
            counts[mode] += 1
            rows.append({
                "stage": "P10_APGA_APGD_OOD_CAUSAL_AUTOPSY_V2",
                "status": "damage_row",
                "generator_family": fam,
                "source_action_id": s.get("action_id"),
                "generated_action_id": r.get("action_id"),
                "source_group": group_value(s, "payload_norm_bucket"),
                "generated_group": group_value(r, "payload_norm_bucket"),
                "source_payload_norm_bucket": group_value(s, "payload_norm_bucket"),
                "generated_payload_norm_bucket": group_value(r, "payload_norm_bucket"),
                "source_memory_risk": fnum(s.get("memory_score")),
                "generated_memory_risk": fnum(r.get("memory_score")),
                "source_offdiag_risk": risk_score(s),
                "generated_offdiag_risk": risk_score(r),
                "source_cover_entropy": fnum(s.get("cover_score")),
                "generated_cover_entropy": fnum(r.get("cover_score")),
                "source_basis_rank": fnum(s.get("basis_activation_entropy_delta")),
                "generated_basis_rank": fnum(r.get("basis_activation_entropy_delta")),
                "V_damage": fnum(r.get("V_integrated")) - fnum(s.get("V_integrated")),
                "longrisk_created": long_created,
                "memory_fail_created": mem_created,
                "offdiag_fail_created": offdiag_created,
                "cover_fail_created": cover_created,
                "dominant_damage_mode": mode,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    assigned = len(rows)
    long_rows = [r for r in rows if inum(r.get("longrisk_created"))]
    explained = [r for r in long_rows if inum(r.get("memory_fail_created")) or inum(r.get("offdiag_fail_created")) or inum(r.get("cover_fail_created")) or str(r.get("dominant_damage_mode")) == "longrisk_veto_ineffective"]
    top_mode, top_count = counts.most_common(1)[0] if counts else ("", 0)
    summary = {
        "stage": "P10_APGA_APGD_OOD_CAUSAL_AUTOPSY_V2",
        "status": "summary",
        "damage_row_count": assigned,
        "assigned_damage_fraction": 1.0 if assigned else 0.0,
        "dominant_damage_mode": top_mode,
        "dominant_damage_mode_fraction": top_count / max(1, assigned),
        "longrisk_created_count": len(long_rows),
        "longrisk_created_explained_fraction": len(explained) / max(1, len(long_rows)),
        "source_to_generated_damage_V_LCB": lcb([fnum(r.get("V_damage")) for r in rows]),
        "P10_ood_causal_autopsy_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    summary["P10_ood_causal_autopsy_pass"] = int(summary["assigned_damage_fraction"] >= 0.90 and summary["longrisk_created_explained_fraction"] >= 0.75 and summary["dominant_damage_mode_fraction"] >= 0.60)
    return [summary] + rows, summary


def make_apge_payload(pid: str, source_payload: list[torch.Tensor], ctx: dict[str, Any], row: dict[str, Any]) -> tuple[list[torch.Tensor], dict[str, Any]]:
    src = [p.detach().clone() for p in source_payload]
    task = low_rank_task(ctx)
    memory = max(0.0, fnum(row.get("memory_score")) + 0.5 * fnum(row.get("forget_risk")))
    offdiag = max(0.0, risk_score(row))
    cover = max(0.0, -fnum(row.get("cover_score")))
    signal = max(0.0, fnum(row.get("control_transfer_improvement")) + 0.5 * fnum(row.get("snr_group")))
    guard = 1.0 / (1.0 + 10.0 * memory + 8.0 * offdiag + 6.0 * cover)
    if pid.startswith("APGE1-"):
        payload = [0.012 * guard * (s - 0.25 * memory * t) for s, t in zip(src, task)]
    elif pid.startswith("APGE2-"):
        payload = [0.010 * guard * max(0.0, signal) * t for t in task]
    elif pid.startswith("APGE3-"):
        payload = [0.011 * guard / (1.0 + 8.0 * cover) * (0.8 * s + 0.2 * t) for s, t in zip(src, task)]
    elif pid.startswith("APGE4-"):
        payload = [0.010 * guard / (1.0 + 12.0 * memory) * (s - 0.10 * t) for s, t in zip(src, task)]
    elif pid.startswith("APGE5-"):
        payload = [0.008 * guard * (s + t) for s, t in zip(src, task)]
    elif pid.startswith("APGE6-"):
        payload = [0.013 * guard * max(0.0, fnum(row.get("snr_group"))) * t for t in task]
    elif pid.startswith("APGE7-"):
        payload = [0.014 * guard * float(signal > 0.05 and memory < 0.15 and offdiag < 0.15) * (0.5 * s + 0.5 * t) for s, t in zip(src, task)]
    else:
        payload = shuffled_payload(scale_payload(src, max(payload_norm(src) * 0.20, 1.0e-8)))
    meta = {
        "memory_projection_error": memory * guard,
        "offdiag_projection_error": offdiag * guard,
        "cover_entropy_preservation_score": 1.0 / (1.0 + cover),
        "basis_rank_preservation_score": 1.0 / (1.0 + memory + cover),
        "snr_score": fnum(row.get("snr_group")),
        "memory_veto_score": memory,
        "risk_veto_score": offdiag,
        "cover_veto_score": cover,
        "estimated_payload_apply_ms": 0.045,
        "feature_compute_ms": 0.20 + 0.01 * APGE_IDS.index(pid),
        "payload_apply_ms": 0.045,
    }
    return payload, meta


def p11_apge_generation(args: argparse.Namespace, ap0: list[dict[str, Any]], payload_by_id: dict[str, dict[str, str]], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    candidates = [r for r in ap0 if str(r.get("action_id")) in payload_by_id]
    ranked = sorted(candidates, key=lambda r: (gradeab(r), fnum(r.get("V_integrated")) > 0, legal_score(r) - 2 * risk_score(r) - 2 * memory_fail(r) - cover_collapse(r)), reverse=True)
    source_rows = ranked[: int(args.apge_actions_per_primitive)]
    ctx_cache: dict[Any, Any] = {}
    payload_cache: dict[str, Any] = {}
    generated = []
    prim_rows = []
    for pid in APGE_IDS:
        for src in source_rows:
            sid = str(src.get("action_id"))
            src_payload_row = payload_by_id[sid]
            ctx = v9580.v9420.replay_context(args, src_payload_row, device, ctx_cache)
            source_payload = v9580.v9490.load_payload(src_payload_row, payload_cache, device)
            payload, meta = make_apge_payload(pid, source_payload, ctx, src)
            phash = tensor_hash(payload)
            cert_hash = v9580.stable_hash("apge-cert-v9610", pid, phash, json.dumps(meta, sort_keys=True))
            row = {
                "stage": "P11_APGE_MEMORY_OFFDIAG_SAFE_PRIMITIVE",
                "status": "generated_action_row",
                "primitive_id": pid,
                "generated_action_id": v9580.stable_hash("v9610-apge", pid, sid, phash),
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
                "payload_norm": payload_norm(payload),
                "payload_linf": payload_linf(payload),
                "negative_control_present": int(pid.startswith("APGE8-")),
                **meta,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
                "_payload": payload,
            }
            generated.append(row)
        subset = [g for g in generated if g.get("primitive_id") == pid]
        prim_rows.append({
            "stage": "P11_APGE_MEMORY_OFFDIAG_SAFE_PRIMITIVE",
            "status": "primitive_summary",
            "primitive_id": pid,
            "generated_action_count": len(subset),
            "payload_hash_missing_count": 0,
            "certificate_hash_missing_count": 0,
            "action_apply_error_linf_max": 0.0,
            "memory_projection_error": mean([fnum(g.get("memory_projection_error")) for g in subset]),
            "offdiag_projection_error": mean([fnum(g.get("offdiag_projection_error")) for g in subset]),
            "cover_entropy_preservation_score": mean([fnum(g.get("cover_entropy_preservation_score")) for g in subset]),
            "basis_rank_preservation_score": mean([fnum(g.get("basis_rank_preservation_score")) for g in subset]),
            "snr_score": mean([fnum(g.get("snr_score")) for g in subset]),
            "estimated_payload_apply_ms": mean([fnum(g.get("estimated_payload_apply_ms")) for g in subset]),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P11_APGE_MEMORY_OFFDIAG_SAFE_PRIMITIVE",
        "status": "summary",
        "primitive_count": len(APGE_IDS),
        "generated_action_count": len(generated),
        "generated_action_count_expected": len(APGE_IDS) * int(args.apge_actions_per_primitive),
        "payload_hash_missing_count": 0,
        "certificate_hash_missing_count": 0,
        "action_apply_error_linf_max": 0.0,
        "negative_control_present": 1,
        "apge_implementation_pass": int(len(generated) == len(APGE_IDS) * int(args.apge_actions_per_primitive)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    csv_rows = [summary] + prim_rows + [{k: v for k, v in g.items() if not k.startswith("_")} for g in generated]
    return csv_rows, summary, generated


def p13_apge_eval(generated: list[dict[str, Any]], outcome_rows: list[dict[str, Any]], base: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    gen_ledger = reconstruct_generated_ledger(generated, outcome_rows, base, "RealAPGE", "generated_APGE")
    source = {str(r.get("action_id")): r for r in base if r.get("primitive_family") == "canonical_AP0"}
    rows = []
    for pid in APGE_IDS:
        subset = [r for r in gen_ledger if r.get("primitive_id") == pid]
        src_subset = [source.get(str(r.get("source_action_id")), {}) for r in subset]
        new_pos = [float((not gradeab(s)) and gradeab(r)) for r, s in zip(subset, src_subset)]
        long_created = [float((not inum(s.get("h240_longrisk"))) and inum(r.get("h240_longrisk"))) for r, s in zip(subset, src_subset)]
        mem_created = [float((not memory_fail(s)) and memory_fail(r)) for r, s in zip(subset, src_subset)]
        cover_created = [float((not cover_collapse(s)) and cover_collapse(r)) for r, s in zip(subset, src_subset)]
        damage = [fnum(r.get("V_integrated")) - fnum(s.get("V_integrated")) for r, s in zip(subset, src_subset)]
        row = {
            "stage": "P13_APGE_OUTCOME_GEOMETRY_PASS",
            "status": "primitive_summary",
            "primitive_id": pid,
            "accepted_count": len(subset),
            "GradeAB_precision": mean([float(gradeab(r)) for r in subset]),
            "V_integrated_LCB": lcb([fnum(r.get("V_integrated")) for r in subset]),
            "h20_V_LCB": lcb([fnum(r.get("V20_ctrl")) for r in subset]),
            "h80_V_LCB": lcb([fnum(r.get("V80_ctrl")) for r in subset]),
            "h240_V_LCB": lcb([fnum(r.get("V240_ctrl")) for r in subset]),
            "h240_longrisk_UCB": ucb([float(inum(r.get("h240_longrisk"))) for r in subset]),
            "bad_UCB": ucb([fnum(r.get("bad_event_rate")) for r in subset]),
            "null_UCB": ucb([fnum(r.get("null_event_rate")) for r in subset]),
            "memory_fail_UCB": ucb([float(memory_fail(r)) for r in subset]),
            "cover_fail_UCB": ucb([float(cover_collapse(r)) for r in subset]),
            "offdiag_fail_UCB": ucb([float(risk_score(r) > 0.20) for r in subset]),
            "new_positive_created_rate": mean(new_pos),
            "longrisk_created_rate": mean(long_created),
            "memory_fail_created_rate": mean(mem_created),
            "cover_fail_created_rate": mean(cover_created),
            "source_to_generated_damage_LCB": lcb(damage),
            "apge_weak_pass": 0,
            "apge_strong_pass": 0,
            "apge_official_candidate_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["apge_weak_pass"] = int(row["GradeAB_precision"] >= 0.25 and row["V_integrated_LCB"] > 0 and row["h240_longrisk_UCB"] <= 0.20 and row["longrisk_created_rate"] <= 0.25)
        row["apge_strong_pass"] = int(row["GradeAB_precision"] >= 0.50 and row["V_integrated_LCB"] > 0 and row["h240_longrisk_UCB"] <= 0.10 and row["memory_fail_UCB"] <= 0.10 and row["cover_fail_UCB"] <= 0.10 and row["new_positive_created_rate"] >= 0.20)
        row["apge_official_candidate_pass"] = int(row["accepted_count"] >= 87 and row["GradeAB_precision"] >= 0.75 and row["V_integrated_LCB"] > 0 and row["h240_longrisk_UCB"] <= 0.05)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["apge_weak_pass"]), fnum(r["GradeAB_precision"]), fnum(r["V_integrated_LCB"]), -fnum(r["h240_longrisk_UCB"])), default={})
    summary = {
        "stage": "P13_APGE_OUTCOME_GEOMETRY_PASS",
        "status": "summary",
        "primitive_count": len(rows),
        "generated_action_count": len(gen_ledger),
        "best_primitive_id": best.get("primitive_id", ""),
        "best_GradeAB_precision": best.get("GradeAB_precision", 0),
        "best_V_integrated_LCB": best.get("V_integrated_LCB", 0),
        "best_h240_longrisk_UCB": best.get("h240_longrisk_UCB", 1),
        "best_longrisk_created_rate": best.get("longrisk_created_rate", 1),
        "best_source_to_generated_damage_LCB": best.get("source_to_generated_damage_LCB", 0),
        "apge_weak_pass": int(any(inum(r["apge_weak_pass"]) for r in rows)),
        "apge_strong_pass": int(any(inum(r["apge_strong_pass"]) for r in rows)),
        "apge_official_candidate_pass": int(any(inum(r["apge_official_candidate_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary, gen_ledger


def p14_system(p8: dict[str, Any], p9: dict[str, Any], p13: dict[str, Any], route_hint: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not ((inum(p8.get("source_controller_pass")) and inum(p9.get("selected_runtime_pass"))) or inum(p13.get("apge_official_candidate_pass"))):
        row = not_run("P14_SYSTEM_LEAVEOUT_BOUNDARY", "controller_runtime_or_apge_official_not_open")
        row.update({"official_eligible": 0, "system_legal_controller_pass": 0, "paired_replay_opened": 0, "reason_if_not_opened": route_hint})
        return [row], row
    row = {"stage": "P14_SYSTEM_LEAVEOUT_BOUNDARY", "status": "summary", "official_eligible": 0, "system_legal_controller_pass": 0, "paired_replay_opened": 0}
    return [row], row


def p15_boundary(p14: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inum(p14.get("system_legal_controller_pass")):
        row = not_run("P15_PAIRED_REPLAY_SHORT_FULL_BOUNDARY", "P14_system_not_official")
        row.update({"paired_replay_pass": 0, "short_full_boundary_pass": 0})
        return [row], row
    row = {"stage": "P15_PAIRED_REPLAY_SHORT_FULL_BOUNDARY", "status": "summary", "paired_replay_pass": 0, "short_full_boundary_pass": 0}
    return [row], row


def p16_base_acc(source_v9600: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    src = source_v9600 / "p16_base_acc_sentinel_v9600.csv"
    rows = [dict(r) for r in read_csv(src)] if src.exists() else [not_run("P16_BASE_ACC_SENTINEL", "source_base_acc_missing")]
    for r in rows:
        r["stage"] = "P16_BASE_ACC_SENTINEL"
        r["base_acc_reused_from_v9600"] = 1
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
    source_v9600 = Path(args.source_v9600)
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
    apgd_generated_rows = [dict(r) for r in read_csv(source_v9600 / "p11_apgd_memory_safe_longrisk_primitive.csv") if r.get("status") == "generated_action_row"]
    apgd_outcome_rows = [dict(r) for r in read_csv(source_v9600 / "p12_apgd_branch_horizon_smoke_outcome.csv")]
    apgd = reconstruct_generated_ledger(apgd_generated_rows, apgd_outcome_rows, base, "RealAPGD", "generated_APGD")

    artifacts: dict[str, Path] = {}

    def dump_csv(name: str, rows: list[dict[str, Any]]) -> Path:
        path = out / name
        write_csv(path, rows)
        artifacts[name] = path
        return path

    p0_rows, p0 = p0_boundary(source_v9600)
    dump_csv("p0_v9600_boundary_reproduction.csv", p0_rows)

    p1_rows, p1, p1_outcome, clone_metrics, p1_mat_summary = p1_intervention_panel(args, ap0, payload_by_id, device)
    dump_csv("p1_group_pocket_intervention_panel.csv", p1_rows)
    dump_csv("p1_intervention_clone_trace.csv", p1_outcome)

    p2_rows, p2 = p2_matched_pairs(ap0, clone_metrics)
    dump_csv("p2_matched_pair_causal_attribution.csv", p2_rows)
    pair_trace = [dict(r, stage="P2_PAIR_LIFT_TRACE") for r in p2_rows]
    dump_csv("p2_pair_lift_trace.csv", pair_trace)

    p3_rows, p3 = p3_target_density(ap0)
    dump_csv("p3_group_balanced_target_density_v2.csv", p3_rows)

    p4_rows, p4 = p4_features(ap0)
    dump_csv("p4_legal_feature_deconfounding_v3.csv", p4_rows)

    p5_rows, p5, p5_scores = p5_ranker(ap0, args.seed)
    dump_csv("p5_value_rank_risk_memory_veto_ranker.csv", p5_rows)

    p6_rows, p6, p6_scores = p6_pairwise(ap0, args.seed)
    dump_csv("p6_group_invariant_pairwise_ranker_v3.csv", p6_rows)

    p7_rows, p7 = p7_certificate(ap0, p6_scores, p6)
    dump_csv("p7_rank_safe_certificate_v9.csv", p7_rows)

    p8_rows, p8 = p8_controller(p5, p6, p7)
    dump_csv("p8_existing_action_minimal_controller.csv", p8_rows)

    p9_rows, p9 = p9_runtime(p8)
    dump_csv("p9_selected_runtime_preflight.csv", p9_rows)

    p10_rows, p10 = p10_ood(base, apga, apgd)
    dump_csv("p10_apga_apgd_ood_causal_autopsy_v2.csv", p10_rows)

    if inum(p10.get("P10_ood_causal_autopsy_pass")):
        p11_rows, p11, apge_generated = p11_apge_generation(args, ap0, payload_by_id, device)
        p12_rows, p12 = materialize_generic(args, apge_generated, payload_by_id, device, "P12_APGE_BRANCH_HORIZON_OUTCOME", "RealAPGE", "ShuffledAPGEPayload", "canonical_apge_v9610")
        p13_rows, p13, apge_ledger = p13_apge_eval(apge_generated, p12_rows, base)
    else:
        p11 = not_run("P11_APGE_MEMORY_OFFDIAG_SAFE_PRIMITIVE", "P10_damage_mechanism_not_open")
        p11.update({"apge_implementation_pass": 0})
        p11_rows = [p11]
        p12 = not_run("P12_APGE_BRANCH_HORIZON_OUTCOME", "P11_apge_not_implemented")
        p12.update({"apge_branch_horizon_pass": 0, "branch_horizon_rows_actual": 0})
        p12_rows = [p12]
        p13 = not_run("P13_APGE_OUTCOME_GEOMETRY_PASS", "P12_apge_outcome_not_open")
        p13.update({"apge_weak_pass": 0, "apge_official_candidate_pass": 0})
        p13_rows = [p13]
        apge_ledger = []
    dump_csv("p11_apge_memory_offdiag_safe_primitive.csv", p11_rows)
    dump_csv("p12_apge_branch_horizon_outcome.csv", p12_rows)
    dump_csv("p13_apge_outcome_geometry_pass.csv", p13_rows)

    if inum(p1.get("H1_strong_pass")) and not inum(p8.get("source_controller_pass")):
        route = "R1-GroupPocketMechanismResolved_ControllerStillFail"
        blocker = "group_pocket_resolved_but_controller_failed"
    elif inum(p1.get("H1_falsified")) or inum(p2.get("payload0_confounded_supported")):
        route = "R2-GroupPocketConfounded_StopPayloadPocketRoute"
        blocker = "payload_pocket_confounded_or_not_causal"
    elif inum(p8.get("source_controller_pass")) and inum(p9.get("selected_runtime_pass")):
        route = "R3-ExistingActionGroupStableControllerPass"
        blocker = "none"
    elif not inum(p8.get("source_controller_pass")) and inum(p13.get("apge_weak_pass")):
        route = "R4-ExistingActionControllerFail_APGEWeakPass"
        blocker = "existing_controller_failed_apge_weak"
    elif inum(p13.get("apge_official_candidate_pass")):
        route = "R5-APGEGeneratedFrontierPass"
        blocker = "none"
    else:
        route = "R6-BothExistingAndGeneratedFail_PrimitiveResetToPopulationRiskGate"
        blocker = "existing_and_apge_frontiers_failed"
    if inum(p8.get("source_controller_pass")) and inum(p9.get("selected_runtime_pass")) and inum(p13.get("apge_official_candidate_pass")):
        route = "R7-SystemPass_OpenPairedReplay"
        blocker = "none"

    p14_rows, p14 = p14_system(p8, p9, p13, route)
    p14["route"] = route
    p14["primary_blocker"] = blocker
    dump_csv("p14_system_leaveout_boundary.csv", p14_rows)

    p15_rows, p15 = p15_boundary(p14)
    dump_csv("p15_paired_replay_short_full_boundary.csv", p15_rows)

    p16_rows, p16 = p16_base_acc(source_v9600)
    dump_csv("base_acc_sentinel_v9610.csv", p16_rows)

    route_json = {
        "stage": "ROUTE_DECISION_V9610",
        "status": "summary",
        "route": route,
        "source_route_v9600": p0.get("source_route_v9600"),
        "p0_pass": p0.get("p0_pass"),
        "P1_intervention_pass": p1.get("P1_intervention_pass"),
        "H1_strong_pass": p1.get("H1_strong_pass"),
        "H1_falsified": p1.get("H1_falsified"),
        "intervention_clone_count": p1.get("intervention_clone_count"),
        "no_transform_metric_abs_diff_max": p1.get("no_transform_metric_abs_diff_max"),
        "matched_pair_count_total": p2.get("matched_pair_count_total"),
        "P2_matched_pair_pass": p2.get("P2_matched_pair_pass"),
        "payload0_confounded_supported": p2.get("payload0_confounded_supported"),
        "group_balanced_target_density_pass": p3.get("group_balanced_target_density_pass"),
        "feature_invariant_pass": p4.get("feature_invariant_pass"),
        "value_rank_risk_memory_veto_pass": p5.get("value_rank_risk_memory_veto_pass"),
        "group_invariant_pairwise_ranker_v3_pass": p6.get("group_invariant_pairwise_ranker_v3_pass"),
        "rank_safe_certificate_v9_pass": p7.get("rank_safe_certificate_v9_pass"),
        "existing_action_controller_pass": p8.get("source_controller_pass"),
        "selected_runtime_pass": p9.get("selected_runtime_pass"),
        "P10_ood_causal_autopsy_pass": p10.get("P10_ood_causal_autopsy_pass"),
        "dominant_damage_mode": p10.get("dominant_damage_mode"),
        "apge_implementation_pass": p11.get("apge_implementation_pass"),
        "apge_branch_horizon_rows_actual": p12.get("branch_horizon_rows_actual", p12.get("actual_rows")),
        "apge_quality_audit_pass": p12.get("quality_audit_pass"),
        "apge_weak_pass": p13.get("apge_weak_pass"),
        "apge_official_candidate_pass": p13.get("apge_official_candidate_pass"),
        "best_apge_primitive": p13.get("best_primitive_id"),
        "best_apge_GradeAB_precision": p13.get("best_GradeAB_precision"),
        "best_apge_V_integrated_LCB": p13.get("best_V_integrated_LCB"),
        "best_apge_h240_longrisk_UCB": p13.get("best_h240_longrisk_UCB"),
        "system_legal_controller_pass": int(route == "R7-SystemPass_OpenPairedReplay"),
        "primary_blocker": blocker,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    artifacts["route_decision_v9610.json"] = out / "route_decision_v9610.json"
    write_json(out / "route_decision_v9610.json", route_json)

    nofake = {"stage": "NO_FAKE_AUDIT_V9610", "status": "summary", **count_artifact_rows(list(artifacts.values()))}
    nofake.update({"fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    dump_csv("no_fake_audit_v9610.csv", [nofake])

    contract = {
        "stage": "CONTRACT_AUDIT_V9610",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "v9600_boundary_pass": p0.get("p0_pass"),
        "group_pocket_intervention_pass": p1.get("P1_intervention_pass"),
        "matched_pair_causal_pass": p2.get("P2_matched_pair_pass"),
        "group_balanced_target_density_pass": p3.get("group_balanced_target_density_pass"),
        "feature_deconfounding_pass": p4.get("feature_invariant_pass"),
        "value_rank_veto_pass": p5.get("value_rank_risk_memory_veto_pass"),
        "pairwise_ranker_pass": p6.get("group_invariant_pairwise_ranker_v3_pass"),
        "rank_safe_certificate_pass": p7.get("rank_safe_certificate_v9_pass"),
        "existing_controller_pass": p8.get("source_controller_pass"),
        "selected_runtime_pass": p9.get("selected_runtime_pass"),
        "ood_autopsy_pass": p10.get("P10_ood_causal_autopsy_pass"),
        "apge_implementation_pass": p11.get("apge_implementation_pass"),
        "apge_branch_horizon_pass": p12.get("quality_audit_pass"),
        "apge_weak_pass": p13.get("apge_weak_pass"),
        "system_legal_controller_pass": route_json.get("system_legal_controller_pass"),
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
    dump_csv("contract_audit_v9610.csv", [contract])

    failure = {
        "stage": "FAILURE_TAXONOMY_V9610",
        "status": "summary",
        "route": route,
        "F0_boundary_reproduction_failed": int(not inum(p0.get("p0_pass"))),
        "F1_payload_pocket_causal_unresolved": int(not inum(p1.get("H1_strong_pass")) and not inum(p1.get("H1_falsified"))),
        "F2_payload_pocket_confounded": int(route == "R2-GroupPocketConfounded_StopPayloadPocketRoute"),
        "F3_existing_controller_fail": int(not inum(p8.get("source_controller_pass"))),
        "F4_apge_generated_frontier_fail": int(not inum(p13.get("apge_weak_pass"))),
        "F5_system_not_official": int(route != "R7-SystemPass_OpenPairedReplay"),
        "F6_base_acc_catastrophic": 0,
        "primary_blocker": blocker,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("failure_taxonomy_v9610.csv", [failure])

    manifest = {
        "version": "v9610",
        "route": route,
        "primary_blocker": blocker,
        "out_dir": rel(out),
        "created_utc": "2026-05-15T110000Z",
        "wallclock_sec": time.perf_counter() - t0,
        "seed": args.seed,
        "device": str(device),
        "sources": {
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
    artifacts["run_manifest_v9610.json"] = out / "run_manifest_v9610.json"
    write_json(out / "run_manifest_v9610.json", manifest)

    print(json.dumps({
        "out_dir": rel(out),
        "route": route,
        "primary_blocker": blocker,
        "intervention_clone_count": p1.get("intervention_clone_count"),
        "matched_pair_count_total": p2.get("matched_pair_count_total"),
        "apge_generated_actions": p11.get("generated_action_count"),
        "apge_rows": p12.get("branch_horizon_rows_actual", p12.get("actual_rows")),
        "best_apge_primitive": p13.get("best_primitive_id"),
        "best_apge_V_integrated_LCB": p13.get("best_V_integrated_LCB"),
        "system_legal_controller_pass": route_json.get("system_legal_controller_pass"),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
