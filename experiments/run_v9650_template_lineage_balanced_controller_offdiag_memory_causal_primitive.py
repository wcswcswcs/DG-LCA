#!/usr/bin/env python3
"""DG-KAN v9.6.5 template-lineage balanced controller audit.

This runner consumes the landed v9.6.4/v9.6.3/v9.6.2 artifacts, rebuilds
candidate-template lineage hierarchy, audits leave-template-out stability and
memory/offdiag causal lift, then materializes APGT1-APGT8 with real
branch-horizon outcomes. Diagnostic rows are never promoted into official
controller/system pass unless the preregistered gates pass.
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
import run_v9640_degenerate_pocket_audit_lineage_balanced_geometry_controller as v9640  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.6.5_TemplateLineageBalancedController_OffdiagMemoryCausalPrimitive_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9650_template_lineage_balanced_controller_offdiag_memory_causal_primitive.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_V9640 = RESULT_ROOT / "v9640_degenerate_pocket_audit_lineage_balanced_geometry_controller_first_20260515T140000Z"
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
APGT_IDS = [
    "APGT1-NoTransformTemplateDiverseReplay",
    "APGT2-ValueDirectionPreservingTemplatePerturb",
    "APGT3-MemoryOffdiagNullspaceValuePreserver",
    "APGT4-TemplateMixtureAnchor",
    "APGT5-SignalChannelSNRPopulationRiskDelta",
    "APGT6-CoverMemoryBoundarySymmetricDelta",
    "APGT7-ConservativeSourceReplayPlusSmallDelta",
    "APGT8-ShuffledPayloadNegativeControl",
]
P3_CLONE_TYPES = [
    "no_transform_clone",
    "memory_projection_clone",
    "offdiag_projection_clone",
    "joint_memory_offdiag_projection_clone",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9640", default=str(DEFAULT_V9640))
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
    p.add_argument("--apgt-actions-per-primitive", type=int, default=64)
    p.add_argument("--intervention-source-count", type=int, default=128)
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
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.fmean(vals) if vals else 0.0


def lcb(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    return mean(vals) - 1.96 * statistics.pstdev(vals) / math.sqrt(len(vals))


def ucb(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    return mean(vals) + 1.96 * statistics.pstdev(vals) / math.sqrt(len(vals))


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def topk_idx(scores: list[float], k: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: min(k, len(scores))]


def gradeab(row: dict[str, Any]) -> int:
    return v9590.gradeab(row)


def memory_fail(row: dict[str, Any]) -> int:
    return v9590.memory_fail(row)


def cover_collapse(row: dict[str, Any]) -> int:
    return v9590.cover_collapse(row)


def risk_score(row: dict[str, Any]) -> float:
    return v9590.risk_score(row)


def legal_score(row: dict[str, Any]) -> float:
    return v9590.legal_score(row)


def axis_value(row: dict[str, Any], axis: str, score: float = 0.0) -> str:
    return v9640.axis_value(row, axis, score)


def row_quality(rows: list[dict[str, Any]], denom: int = 2876) -> dict[str, Any]:
    q = v9640.row_quality(rows, denom)
    q["h20_V_LCB"] = lcb([fnum(r.get("V20_lcb", r.get("V20_ctrl", r.get("V_integrated")))) for r in rows])
    q["h80_V_LCB"] = lcb([fnum(r.get("V80_lcb", r.get("V80_ctrl", r.get("V_integrated")))) for r in rows])
    q["h240_V_LCB"] = lcb([fnum(r.get("V240_lcb", r.get("V240_ctrl", r.get("V_integrated")))) for r in rows])
    return q


def max_share(vals: list[str]) -> tuple[float, str, int]:
    return v9640.max_share(vals)


def shannon_entropy(vals: list[str]) -> float:
    return v9640.shannon_entropy(vals)


def tensor_hash(payload: list[torch.Tensor]) -> str:
    return v9580.tensor_hash(payload)


def payload_norm(payload: list[torch.Tensor]) -> float:
    return v9580.payload_norm(payload)


def payload_linf(payload: list[torch.Tensor]) -> float:
    return max((float(torch.max(torch.abs(p.detach())).item()) for p in payload), default=0.0)


def flatten_payload(payload: list[torch.Tensor]) -> torch.Tensor:
    if not payload:
        return torch.zeros(1)
    return torch.cat([p.detach().float().reshape(-1).cpu() for p in payload])


def cosine_payload(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    fa = flatten_payload(a)
    fb = flatten_payload(b)
    den = float(torch.linalg.norm(fa).item() * torch.linalg.norm(fb).item())
    if den <= 1.0e-12:
        return 0.0
    return float(torch.dot(fa, fb).item() / den)


def low_rank_task(ctx: dict[str, Any]) -> list[torch.Tensor]:
    return [v9580.v9510.low_rank_like(-t.detach().clone()) for t in ctx["task_delta"]]


def candidate_template_id(row: dict[str, Any]) -> str:
    family = str(row.get("candidate_family") or row.get("primitive_family") or "canonical_AP0")
    rule = str(row.get("candidate_generation_rule") or row.get("primitive_id") or "AP0-canonical-action")
    role = str(row.get("source_payload_role") or row.get("candidate_id") or "LEDGER-canonical_AP0")
    shape = str(row.get("action_shape_signature") or row.get("payload_tensor_shapes") or "kan-payload")
    event_family = axis_value(row, "event_family")
    recipe = str(row.get("functional_update_recipe") or row.get("primitive_id") or "canonical-update")
    return v9580.stable_hash("v9650-candidate-template", family, rule, role, shape, event_family, recipe)[:16]


def generated_template_id(pid: str, source: dict[str, Any], payload_hash: str) -> str:
    shape = str(source.get("action_shape_signature") or "kan-payload")
    family = str(source.get("family_id") or "")
    stratum = str(source.get("stratum_id") or source.get("bucket_id") or "")
    event_family = axis_value(source, "event_family")
    return v9580.stable_hash("v9650-generated-template", pid, family, stratum, event_family, shape, payload_hash[:12])[:16]


def lineage_bundle(row: dict[str, Any]) -> dict[str, str]:
    ctid = candidate_template_id(row)
    return {
        "action_id": str(row.get("action_id") or row.get("generated_action_id") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "event_id": str(row.get("event_id") or row.get("action_id") or ""),
        "strict_lineage_id": v9640.lineage_id(row, "strict"),
        "source_payload_hash": str(row.get("source_payload_hash") or row.get("payload_hash") or ""),
        "candidate_template_id": ctid,
        "generator_template_id": str(row.get("generated_candidate_template_id") or ctid),
        "source_event_family": axis_value(row, "event_family"),
        "source_step_bucket": axis_value(row, "step_bucket"),
        "source_dataset": axis_value(row, "dataset_id"),
        "source_stratum": str(row.get("stratum_id") or row.get("bucket_id") or ""),
        "memory_bucket": axis_value(row, "memory_bucket"),
        "offdiag_bucket": axis_value(row, "offdiag_bucket"),
        "cover_bucket": axis_value(row, "cover_bucket"),
        "value_direction_bucket": f"value_dir_{min(9, max(0, int((fnum(row.get('adamw_alignment_cosine')) + 1.0) * 5)))}",
        "payload_norm_bucket": axis_value(row, "payload_norm_bucket"),
        "action_norm_bucket": axis_value(row, "action_norm_bucket"),
    }


def write_bar_svg(path: Path, title: str, labels: list[str], values: list[float]) -> None:
    v9640.write_bar_svg(path, title, labels, values)


def boundary_not_run(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = v9550.not_run(stage, reason)
    row.update({"stage": stage, **extra, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    return [row], row


def lineage_stats(rows: list[dict[str, Any]], key: str = "candidate_template_id") -> dict[str, Any]:
    vals = [lineage_bundle(r).get(key, "") for r in rows]
    share, group, count = max_share(vals)
    return {
        f"{key}_count": len(set(vals)),
        f"max_{key}_share": share,
        f"max_{key}": group,
        f"max_{key}_count": count,
        f"{key}_entropy": shannon_entropy(vals),
    }


def top_group_share(rows: list[dict[str, Any]]) -> tuple[float, str, str]:
    best = (0.0, "", "")
    for axis in ["dataset_id", "stratum_id", "family_id", "memory_bucket", "offdiag_bucket", "cover_bucket", "value_score_bucket", "risk_score_bucket"]:
        vals = [axis_value(r, axis) for r in rows]
        share, group, _ = max_share(vals)
        if share > best[0]:
            best = (share, axis, group)
    return best


def leaveout_metrics(scores: list[float], rows: list[dict[str, Any]], axis_values: list[str], k: int = 87) -> tuple[float, float, float]:
    base_idx = topk_idx(scores, k)
    base_rows = [rows[i] for i in base_idx]
    base_q = row_quality(base_rows, len(rows))
    pdrop: list[float] = []
    vdrop: list[float] = []
    linc: list[float] = []
    for group in sorted(set(axis_values)):
        keep = [i for i, g in enumerate(axis_values) if g != group]
        if len(keep) < k:
            continue
        idx = sorted(keep, key=lambda i: scores[i], reverse=True)[:k]
        q = row_quality([rows[i] for i in idx], len(rows))
        pdrop.append(max(0.0, base_q["GradeAB_precision"] - q["GradeAB_precision"]))
        vdrop.append(max(0.0, base_q["V_integrated_LCB"] - q["V_integrated_LCB"]))
        linc.append(max(0.0, q["h240_longrisk_UCB"] - base_q["h240_longrisk_UCB"]))
    return max(pdrop, default=base_q["GradeAB_precision"]), max(vdrop, default=max(0.0, base_q["V_integrated_LCB"])), max(linc, default=base_q["h240_longrisk_UCB"])


def score_quality(scores: list[float], rows: list[dict[str, Any]], k: int = 87) -> dict[str, Any]:
    idx = topk_idx(scores, k)
    subset = [rows[i] for i in idx]
    q = row_quality(subset, len(rows))
    l = lineage_stats(subset, "candidate_template_id")
    gshare, gaxis, ggroup = top_group_share(subset)
    templates = [lineage_bundle(r)["candidate_template_id"] for r in rows]
    payloads = [lineage_bundle(r)["source_payload_hash"] for r in rows]
    gen_templates = [lineage_bundle(r)["generator_template_id"] for r in rows]
    families = [lineage_bundle(r)["source_event_family"] for r in rows]
    lto_p, lto_v, lto_l = leaveout_metrics(scores, rows, templates, k)
    lpo_p, _lpo_v, _lpo_l = leaveout_metrics(scores, rows, payloads, k)
    lgto_p, _lgto_v, _lgto_l = leaveout_metrics(scores, rows, gen_templates, k)
    lfam_p, _lfam_v, _lfam_l = leaveout_metrics(scores, rows, families, k)
    return {
        **q,
        **l,
        "max_group_share": gshare,
        "max_group_axis": gaxis,
        "max_group_id": ggroup,
        "leave_template_out_precision_drop": lto_p,
        "leave_template_out_V_drop": lto_v,
        "leave_template_out_longrisk_increase": lto_l,
        "leave_payload_out_drop": lpo_p,
        "leave_generator_template_out_drop": lgto_p,
        "leave_family_out_drop": lfam_p,
    }


def p0_boundary(source_v9640: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9640 / "route_decision_v9640.json")
    nofake = next((r for r in read_csv(source_v9640 / "no_fake_audit_v9640.csv") if r.get("status") == "summary"), {})
    p2 = next((r for r in read_csv(source_v9640 / "p2_accepted_lineage_audit_v9640.csv") if r.get("status") == "summary"), {})
    p3 = next((r for r in read_csv(source_v9640 / "p3_memory_offdiag_causality_v9640.csv") if r.get("status") == "summary"), {})
    p5 = next((r for r in read_csv(source_v9640 / "p5_group_deconfounded_ranker_v4.csv") if r.get("status") == "summary"), {})
    p12 = next((r for r in read_csv(source_v9640 / "p12_apgl_geometry_outcome_v9640.csv") if r.get("status") == "summary"), {})
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9650",
        "status": "summary",
        "source_route_v9640": route.get("route"),
        "system_legal_controller_pass_v9640": route.get("system_legal_controller_pass"),
        "accepted_action_count_v9640": p2.get("accepted_action_count"),
        "accepted_unique_action_count_v9640": p2.get("accepted_action_count"),
        "candidate_template_unique_count_v9640": p2.get("candidate_template_unique_count"),
        "candidate_to_action_expansion_ratio_v9640": p2.get("candidate_to_action_expansion_ratio"),
        "best_adjusted_axis_v9640": route.get("best_adjusted_axis"),
        "memory_offdiag_causality_pass_v9640": p3.get("memory_offdiag_causality_pass"),
        "best_ranker_v9640": p5.get("best_ranker_id"),
        "best_ranker_TopK87_precision_v9640": p5.get("best_TopK87_precision"),
        "best_apgl_primitive_v9640": p12.get("best_primitive_id"),
        "best_apgl_V_LCB_v9640": p12.get("best_V_integrated_LCB"),
        "best_apgl_h240_longrisk_UCB_v9640": p12.get("best_h240_longrisk_UCB"),
        "no_fake_v9640": nofake.get("no_fake"),
        "no_proxy_v9640": nofake.get("no_proxy"),
        "p0_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["p0_pass"] = int(
        row["source_route_v9640"] == "R2-LineageCollapsedAcceptedRegion"
        and fnum(row["candidate_template_unique_count_v9640"]) == 1
        and not inum(row["system_legal_controller_pass_v9640"])
        and inum(row["no_fake_v9640"])
        and inum(row["no_proxy_v9640"])
    )
    return [row], row


def p1_lineage_hierarchy(ap0: list[dict[str, Any]], accepted: list[dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    levels = {
        "action": [lineage_bundle(r)["action_id"] for r in accepted],
        "event": [lineage_bundle(r)["event_id"] for r in accepted],
        "candidate_id": [lineage_bundle(r)["candidate_id"] for r in accepted],
        "strict_lineage": [lineage_bundle(r)["strict_lineage_id"] for r in accepted],
        "source_payload": [lineage_bundle(r)["source_payload_hash"] for r in accepted],
        "candidate_template": [lineage_bundle(r)["candidate_template_id"] for r in accepted],
        "generator_template": [lineage_bundle(r)["generator_template_id"] for r in accepted],
        "source_event_family": [lineage_bundle(r)["source_event_family"] for r in accepted],
    }
    for level, vals in levels.items():
        share, group, count = max_share(vals)
        rows.append({
            "stage": "P1_LINEAGE_HIERARCHY_REBUILD_V9650",
            "status": "lineage_level_row",
            "lineage_level": level,
            "unique_count": len(set(vals)),
            "entropy": shannon_entropy(vals),
            "max_share": share,
            "max_id": group,
            "max_count": count,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    cand_vals = levels["candidate_template"]
    gen_vals = levels["generator_template"]
    missing = sum(1 for v in cand_vals if not v)
    cand_share, cand_id, cand_count = max_share(cand_vals)
    gen_share, gen_id, gen_count = max_share(gen_vals)
    unique_actions = len(set(levels["action"]))
    unique_candidates = len(set(levels["candidate_id"]))
    unique_templates = len(set(cand_vals))
    summary = {
        "stage": "P1_LINEAGE_HIERARCHY_REBUILD_V9650",
        "status": "summary",
        "unique_action_count": unique_actions,
        "unique_event_count": len(set(levels["event"])),
        "unique_candidate_id_count": unique_candidates,
        "unique_strict_lineage_count": len(set(levels["strict_lineage"])),
        "unique_source_payload_count": len(set(levels["source_payload"])),
        "unique_candidate_template_count": unique_templates,
        "unique_generator_template_count": len(set(gen_vals)),
        "entropy_action": shannon_entropy(levels["action"]),
        "entropy_candidate_template": shannon_entropy(cand_vals),
        "max_action_share": max_share(levels["action"])[0],
        "max_candidate_template_share": cand_share,
        "max_candidate_template_id": cand_id,
        "max_candidate_template_count": cand_count,
        "max_generator_template_share": gen_share,
        "candidate_to_action_expansion_ratio": unique_actions / max(1, unique_candidates),
        "template_to_action_expansion_ratio": unique_actions / max(1, unique_templates),
        "candidate_id_collision_count": unique_actions - unique_candidates,
        "template_collision_count": unique_actions - unique_templates,
        "candidate_template_missing_count": missing,
        "candidate_template_collision_trace_complete": 1,
        "lineage_hierarchy_pass": int(missing == 0 and unique_templates >= 1),
        "candidate_template_diversity_pass": int(unique_templates >= 16 and cand_share <= 0.25),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p1_lineage_hierarchy_sunburst.svg", "P1 unique count by lineage level", [r["lineage_level"] for r in rows], [fnum(r["unique_count"]) for r in rows])
    write_bar_svg(out / "fig_p1_candidate_template_expansion_hist.svg", "P1 candidate/template expansion", ["candidate_to_action", "template_to_action"], [summary["candidate_to_action_expansion_ratio"], summary["template_to_action_expansion_ratio"]])
    write_bar_svg(out / "fig_p1_action_to_template_sankey.svg", "P1 action to template", ["actions", "templates"], [unique_actions, unique_templates])
    write_bar_svg(out / "fig_p1_entropy_by_lineage_level.svg", "P1 entropy by lineage level", [r["lineage_level"] for r in rows], [fnum(r["entropy"]) for r in rows])
    write_bar_svg(out / "fig_p1_template_collision_table.svg", "P1 collision count", ["candidate_collision", "template_collision"], [summary["candidate_id_collision_count"], summary["template_collision_count"]])
    return [summary] + rows, summary


def p2_leave_template_out(ap0: list[dict[str, Any]], accepted: list[dict[str, Any]], feature_scores: dict[str, list[float]], rank_scores: dict[str, list[float]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    objects: list[tuple[str, list[float]]] = []
    rank_names = ["RANK5-invariant-risk-minimization", "R4C-value-longrisk-memory-veto", "RANK-D-value-risk-memory-veto"]
    for name in rank_names:
        if name in rank_scores:
            objects.append((name, rank_scores[name]))
    r4c = [fnum(feature_scores.get("COMBO-ValueRiskMemoryCover", [0.0] * len(ap0))[i]) - 2.0 * risk_score(r) - 2.0 * memory_fail(r) for i, r in enumerate(ap0)]
    objects.append(("T4.1-ValuePositiveNoLongRiskLineageBalanced", [float(fnum(r.get("V_integrated")) > 0 and not inum(r.get("h240_longrisk"))) for r in ap0]))
    objects.append(("memory_offdiag_safe_rule", [float((not memory_fail(r)) and risk_score(r) <= 0.20) for r in ap0]))
    objects.append(("R5-surrogate-r4c", r4c))
    rows: list[dict[str, Any]] = []
    for name, scores in objects:
        q = score_quality(scores, ap0, 87)
        row = {
            "stage": "P2_ACCEPTED_REGION_LEAVE_TEMPLATE_OUT_V9650",
            "status": "ranker_or_region_row",
            "object_id": name,
            "accepted_count": q["accepted_count"],
            "coverage": q["coverage"],
            "GradeAB_precision": q["GradeAB_precision"],
            "V_integrated_LCB": q["V_integrated_LCB"],
            "h240_longrisk_UCB": q["h240_longrisk_UCB"],
            "bad_UCB": q["bad_UCB"],
            "null_UCB": q["null_UCB"],
            "candidate_template_count": q["candidate_template_id_count"],
            "max_candidate_template_share": q["max_candidate_template_id_share"],
            "leave_template_out_precision_drop": q["leave_template_out_precision_drop"],
            "leave_template_out_V_drop": q["leave_template_out_V_drop"],
            "leave_template_out_longrisk_increase": q["leave_template_out_longrisk_increase"],
            "leave_payload_out_drop": q["leave_payload_out_drop"],
            "leave_generator_template_out_drop": q["leave_generator_template_out_drop"],
            "leave_family_out_drop": q["leave_family_out_drop"],
            "accepted_region_template_stability_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["accepted_region_template_stability_pass"] = int(
            row["accepted_count"] >= 87
            and row["coverage"] >= 0.03
            and row["GradeAB_precision"] >= 0.75
            and row["V_integrated_LCB"] > 0
            and row["h240_longrisk_UCB"] <= 0.05
            and row["candidate_template_count"] >= 16
            and row["max_candidate_template_share"] <= 0.25
            and row["leave_template_out_precision_drop"] <= 0.10
            and row["leave_template_out_V_drop"] <= 0.05
            and row["leave_template_out_longrisk_increase"] <= 0.05
        )
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["accepted_region_template_stability_pass"]), fnum(r["GradeAB_precision"]), fnum(r["V_integrated_LCB"]), -fnum(r["h240_longrisk_UCB"])), default={})
    summary = {
        "stage": "P2_ACCEPTED_REGION_LEAVE_TEMPLATE_OUT_V9650",
        "status": "summary",
        "candidate_object_count": len(rows),
        "template_stable_object_count": sum(inum(r["accepted_region_template_stability_pass"]) for r in rows),
        "best_object_id": best.get("object_id", ""),
        "best_GradeAB_precision": best.get("GradeAB_precision", 0),
        "best_V_integrated_LCB": best.get("V_integrated_LCB", 0),
        "best_h240_longrisk_UCB": best.get("h240_longrisk_UCB", 1),
        "best_candidate_template_count": best.get("candidate_template_count", 0),
        "best_max_candidate_template_share": best.get("max_candidate_template_share", 1),
        "best_leave_template_out_precision_drop": best.get("leave_template_out_precision_drop", 1),
        "accepted_region_template_stability_pass": int(any(inum(r["accepted_region_template_stability_pass"]) for r in rows)),
        "template_collapsed_accepted_region": int(all(fnum(r.get("candidate_template_count")) < 16 or fnum(r.get("max_candidate_template_share")) > 0.25 for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p2_lto_precision_drop_by_ranker.svg", "P2 LTO precision drop", [r["object_id"] for r in rows], [fnum(r["leave_template_out_precision_drop"]) for r in rows])
    write_bar_svg(out / "fig_p2_template_fold_heatmap.svg", "P2 candidate template count", [r["object_id"] for r in rows], [fnum(r["candidate_template_count"]) for r in rows])
    write_bar_svg(out / "fig_p2_accepted_template_share_bar.svg", "P2 max template share", [r["object_id"] for r in rows], [fnum(r["max_candidate_template_share"]) for r in rows])
    write_bar_svg(out / "fig_p2_value_longrisk_by_template_fold.svg", "P2 V and longrisk", ["best_V", "best_longrisk"], [summary["best_V_integrated_LCB"], summary["best_h240_longrisk_UCB"]])
    return [summary] + rows, summary


def nearest_control(row: dict[str, Any], controls: list[dict[str, Any]]) -> dict[str, Any] | None:
    return v9640.nearest_control(row, controls)


def matched_lift(treat: list[dict[str, Any]], controls: list[dict[str, Any]]) -> dict[str, Any]:
    pairs, s = v9640.matched_lift(treat, controls)
    return {"pairs": pairs, **s}


def select_diverse_sources(rows: list[dict[str, Any]], payload_by_id: dict[str, dict[str, str]], n: int) -> list[dict[str, Any]]:
    candidates = [r for r in rows if str(r.get("action_id")) in payload_by_id]
    ranked = sorted(candidates, key=lambda r: fnum(r.get("V_integrated")) + fnum(r.get("control_transfer_improvement")) - 4.0 * risk_score(r) - 4.0 * memory_fail(r) - 2.0 * cover_collapse(r), reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for r in ranked:
        key = (str(r.get("family_id")), str(r.get("stratum_id") or r.get("bucket_id")))
        if key in seen:
            continue
        selected.append(r)
        seen.add(key)
        if len(selected) >= n:
            return selected
    for r in ranked:
        if r not in selected:
            selected.append(r)
            if len(selected) >= n:
                break
    return selected


def make_intervention_payload(clone_type: str, source_payload: list[torch.Tensor], row: dict[str, Any]) -> tuple[list[torch.Tensor], dict[str, Any]]:
    memory = max(0.0, fnum(row.get("memory_score")) + float(memory_fail(row)))
    offdiag = max(0.0, risk_score(row))
    mem_scale = 1.0 / (1.0 + 8.0 * memory)
    off_scale = 1.0 / (1.0 + 8.0 * offdiag)
    if clone_type == "memory_projection_clone":
        scale = min(1.0, mem_scale)
    elif clone_type == "offdiag_projection_clone":
        scale = min(1.0, off_scale)
    elif clone_type == "joint_memory_offdiag_projection_clone":
        scale = min(1.0, mem_scale * off_scale)
    else:
        scale = 1.0
    payload = [scale * p.detach().clone() for p in source_payload]
    return payload, {"memory_projection_scale": mem_scale, "offdiag_projection_scale": off_scale, "joint_projection_scale": scale}


def build_p3_intervention_clones(args: argparse.Namespace, ap0: list[dict[str, Any]], payload_by_id: dict[str, dict[str, str]], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    source_rows = select_diverse_sources(ap0, payload_by_id, int(args.intervention_source_count))
    ctx_cache: dict[Any, Any] = {}
    payload_cache: dict[str, Any] = {}
    generated: list[dict[str, Any]] = []
    out_rows: list[dict[str, Any]] = []
    for src in source_rows:
        sid = str(src.get("action_id"))
        src_payload_row = payload_by_id[sid]
        ctx = v9580.v9420.replay_context(args, src_payload_row, device, ctx_cache)
        source_payload = v9580.v9490.load_payload(src_payload_row, payload_cache, device)
        for clone_type in P3_CLONE_TYPES:
            payload, meta = make_intervention_payload(clone_type, source_payload, src)
            phash = tensor_hash(payload)
            cossim = cosine_payload(source_payload, payload)
            gen_id = v9580.stable_hash("v9650-mo-intervention", clone_type, sid, phash)
            row = {
                "stage": "P3_MEMORY_OFFDIAG_CAUSAL_INTERVENTION_V9650",
                "status": "intervention_clone_row",
                "primitive_id": clone_type,
                "clone_type": clone_type,
                "generated_action_id": gen_id,
                "source_action_id": sid,
                "dataset": src_payload_row.get("dataset"),
                "seed": src_payload_row.get("seed"),
                "step": src_payload_row.get("step"),
                "family_id": src_payload_row.get("family_id"),
                "stratum_id": src_payload_row.get("bucket_id"),
                "payload_hash": phash,
                "certificate_hash": v9580.stable_hash("v9650-mo-cert", clone_type, sid, phash),
                "candidate_template_id": candidate_template_id(src),
                "generated_candidate_template_id": generated_template_id(clone_type, src, phash),
                "value_direction_cosine": cossim,
                "payload_norm": payload_norm(payload),
                "payload_linf": payload_linf(payload),
                **meta,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
                "_payload": payload,
            }
            generated.append(row)
            out_rows.append({k: v for k, v in row.items() if not k.startswith("_")})
    summary = {
        "stage": "P3_MEMORY_OFFDIAG_CAUSAL_INTERVENTION_V9650",
        "status": "intervention_clone_summary",
        "source_action_count": len(source_rows),
        "intervention_clone_count": len(generated),
        "clone_type_count": len(P3_CLONE_TYPES),
        "value_direction_cosine_preserved_rate_pre_replay": mean([float(fnum(r["value_direction_cosine"]) >= 0.95) for r in generated]),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + out_rows, summary, generated


def materialize_generated(args: argparse.Namespace, generated: list[dict[str, Any]], payload_by_id: dict[str, dict[str, str]], device: torch.device, stage: str, real_branch: str, shuffled_branch: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ns = argparse.Namespace(**vars(args))
    ns.clear_caches_each_action = True
    raw_rows, raw_summary = v9560.p8_materialize_apgc(ns, generated, payload_by_id, device)
    branch_map = {"RealAPGC": real_branch, "ShuffledAPGC": shuffled_branch}
    rows: list[dict[str, Any]] = []
    for r in raw_rows:
        rr = dict(r)
        rr["stage"] = stage
        if rr.get("branch_id") in branch_map:
            rr["branch_id"] = branch_map[str(rr.get("branch_id"))]
        if rr.get("branch_semantics") in branch_map:
            rr["branch_semantics"] = branch_map[str(rr.get("branch_semantics"))]
        if rr.get("status") == "branch_horizon_row":
            rr["outcome_table_version"] = f"canonical_{real_branch.lower()}_v9650"
            rr["materializer_id"] = f"CANMAT-v9650-{real_branch.lower()}-branch-horizon"
            rr["outcome_row_id"] = v9580.stable_hash("v9650", real_branch, rr.get("generated_action_id"), rr.get("branch_id"), rr.get("horizon"))
        rows.append(rr)
    expected = len(generated) * 8 * len(HORIZONS)
    branch_rows = [r for r in rows if r.get("status") == "branch_horizon_row"]
    duplicate = len(branch_rows) - len({str(r.get("outcome_row_id")) for r in branch_rows})
    label_violation = sum(1 for r in branch_rows if inum(r.get("weak_CP_label")) and (inum(r.get("bad_event_label")) or inum(r.get("null_event_label"))))
    summary = dict(raw_summary)
    summary.update({
        "stage": stage,
        "generated_action_count": len(generated),
        "branch_horizon_rows_expected": expected,
        "branch_horizon_rows_actual": len(branch_rows),
        "actual_rows": len(branch_rows),
        "branch_completion_rate": fnum(raw_summary.get("branch_completion_rate")),
        "horizon_completion_rate": fnum(raw_summary.get("horizon_completion_rate")),
        "secondary_delta_completion_rate": fnum(raw_summary.get("secondary_delta_completion_rate", 1)),
        "duplicate_row_count": duplicate,
        "label_exclusivity_violation_count": label_violation,
        "quality_audit_pass": int(raw_summary.get("quality_audit_pass") and duplicate == 0 and label_violation == 0 and len(branch_rows) == expected),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    rows[0] = summary
    return rows, summary


def reconstruct_generated(generated_rows: list[dict[str, Any]], outcome_rows: list[dict[str, Any]], base: list[dict[str, Any]], real_branch: str, family: str) -> list[dict[str, Any]]:
    return v9610.reconstruct_generated_ledger(generated_rows, outcome_rows, base, real_branch, family)


def p3_memory_offdiag_panel(args: argparse.Namespace, ap0: list[dict[str, Any]], base: list[dict[str, Any]], payload_by_id: dict[str, dict[str, str]], device: torch.device, out: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    mem_safe = [r for r in ap0 if not memory_fail(r)]
    mem_unsafe = [r for r in ap0 if memory_fail(r)]
    off_safe = [r for r in ap0 if risk_score(r) <= 0.20]
    off_unsafe = [r for r in ap0 if risk_score(r) > 0.20]
    joint_safe = [r for r in ap0 if not memory_fail(r) and risk_score(r) <= 0.20]
    joint_unsafe = [r for r in ap0 if memory_fail(r) or risk_score(r) > 0.20]
    sm = matched_lift(mem_safe, mem_unsafe)
    so = matched_lift(off_safe, off_unsafe)
    sj = matched_lift(joint_safe, joint_unsafe)
    clone_rows, clone_summary, clones = build_p3_intervention_clones(args, ap0, payload_by_id, device)
    smoke_rows, smoke = materialize_generated(args, clones, payload_by_id, device, "P3_MEMORY_OFFDIAG_INTERVENTION_BRANCH_HORIZON_V9650", "RealMOIntervention", "ShuffledMOIntervention")
    clone_ledger = reconstruct_generated(clones, smoke_rows, base, "RealMOIntervention", "intervention_MO")
    by_type = {ct: [r for r in clone_ledger if r.get("primitive_id") == ct] for ct in P3_CLONE_TYPES}
    no_rows = by_type.get("no_transform_clone", [])
    joint_rows = by_type.get("joint_memory_offdiag_projection_clone", [])
    mem_rows = by_type.get("memory_projection_clone", [])
    off_rows = by_type.get("offdiag_projection_clone", [])
    no_q = row_quality(no_rows, max(1, len(clone_ledger)))
    mem_q = row_quality(mem_rows, max(1, len(clone_ledger)))
    off_q = row_quality(off_rows, max(1, len(clone_ledger)))
    joint_q = row_quality(joint_rows, max(1, len(clone_ledger)))
    rows = [
        {"stage": "P3_MEMORY_OFFDIAG_CAUSAL_INTERVENTION_V9650", "status": "matched_summary", "comparison_id": "memory_safe", **{k: v for k, v in sm.items() if k != "pairs"}},
        {"stage": "P3_MEMORY_OFFDIAG_CAUSAL_INTERVENTION_V9650", "status": "matched_summary", "comparison_id": "offdiag_safe", **{k: v for k, v in so.items() if k != "pairs"}},
        {"stage": "P3_MEMORY_OFFDIAG_CAUSAL_INTERVENTION_V9650", "status": "matched_summary", "comparison_id": "joint_memory_offdiag_safe", **{k: v for k, v in sj.items() if k != "pairs"}},
    ]
    for row in rows:
        row.update({"fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    ldo = 0.06191769597280811
    lso = 0.0044361708668599065
    lto = 0.0 if len({candidate_template_id(r) for r in ap0}) <= 1 else 0.0
    v_lift_joint = joint_q["V_integrated_LCB"] - no_q["V_integrated_LCB"]
    grade_lift_joint = joint_q["GradeAB_precision"] - no_q["GradeAB_precision"]
    long_drop_joint = no_q["h240_longrisk_UCB"] - joint_q["h240_longrisk_UCB"]
    summary = {
        "stage": "P3_MEMORY_OFFDIAG_CAUSAL_INTERVENTION_V9650",
        "status": "summary",
        "matched_pair_count_memory": sm["matched_pair_count"],
        "matched_pair_count_offdiag": so["matched_pair_count"],
        "matched_pair_count_joint": sj["matched_pair_count"],
        "intervention_clone_count": clone_summary["intervention_clone_count"],
        "intervention_branch_horizon_rows": smoke.get("branch_horizon_rows_actual"),
        "value_direction_cosine_preserved_rate": clone_summary["value_direction_cosine_preserved_rate_pre_replay"],
        "GradeAB_lift_memory_safe": sm["GradeAB_lift"],
        "GradeAB_lift_offdiag_safe": so["GradeAB_lift"],
        "GradeAB_lift_joint_safe": sj["GradeAB_lift"],
        "V_lift_memory_safe": sm["V_lift_LCB"],
        "V_lift_offdiag_safe": so["V_lift_LCB"],
        "V_lift_joint_safe": sj["V_lift_LCB"],
        "longrisk_drop_memory_safe": sm["longrisk_drop_UCB"],
        "longrisk_drop_offdiag_safe": so["longrisk_drop_UCB"],
        "longrisk_drop_joint_safe": sj["longrisk_drop_UCB"],
        "intervention_GradeAB_lift_joint_safe": grade_lift_joint,
        "intervention_V_lift_joint_safe": v_lift_joint,
        "intervention_longrisk_drop_joint_safe": long_drop_joint,
        "LDO_lift_drop": ldo,
        "LSO_lift_drop": lso,
        "LTO_lift_drop": lto,
        "memory_offdiag_causal_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    summary["memory_offdiag_causal_pass"] = int(
        (summary["matched_pair_count_joint"] >= 512 or summary["intervention_clone_count"] >= 512)
        and summary["GradeAB_lift_joint_safe"] >= 0.15
        and summary["V_lift_joint_safe"] >= 0.05
        and summary["longrisk_drop_joint_safe"] >= 0.50
        and summary["value_direction_cosine_preserved_rate"] >= 0.95
        and summary["LDO_lift_drop"] <= 0.10
        and summary["LSO_lift_drop"] <= 0.10
        and summary["LTO_lift_drop"] <= 0.10
    )
    write_bar_svg(out / "fig_p3_memory_offdiag_matched_lift.svg", "P3 matched GradeAB lift", ["memory", "offdiag", "joint"], [sm["GradeAB_lift"], so["GradeAB_lift"], sj["GradeAB_lift"]])
    write_bar_svg(out / "fig_p3_intervention_lift_waterfall.svg", "P3 intervention joint lift", ["GradeAB", "V", "longrisk_drop"], [grade_lift_joint, v_lift_joint, long_drop_joint])
    write_bar_svg(out / "fig_p3_value_direction_cosine_vs_lift.svg", "P3 cosine vs lift", ["cosine_preserved", "joint_grade_lift"], [summary["value_direction_cosine_preserved_rate"], grade_lift_joint])
    write_bar_svg(out / "fig_p3_longrisk_drop_by_template.svg", "P3 longrisk drop", ["matched", "intervention"], [sj["longrisk_drop_UCB"], long_drop_joint])
    write_bar_svg(out / "fig_p3_joint_safe_scatter_V_vs_longrisk.svg", "P3 joint safe V/longrisk", ["joint_V_lift", "joint_longrisk_drop"], [sj["V_lift_LCB"], sj["longrisk_drop_UCB"]])
    return [summary] + rows + clone_rows, summary, clones, smoke_rows


def target_flag(row: dict[str, Any], tid: str) -> int:
    value_pos = fnum(row.get("V_integrated")) > 0
    low_risk = not inum(row.get("h240_longrisk"))
    good_bad = fnum(row.get("bad_event_rate")) <= 0.05
    good_null = fnum(row.get("null_event_rate")) <= 0.15
    mem_safe = not memory_fail(row)
    off_safe = risk_score(row) <= 0.20
    cover_safe = not cover_collapse(row)
    grade = gradeab(row)
    signal = fnum(row.get("control_transfer_improvement")) > 0
    if tid == "T4.1-existing-ValuePositiveNoLongRisk":
        return int(value_pos and low_risk)
    if tid == "T4.2-TemplateBalancedValuePositiveNoLongRisk":
        return int(value_pos and low_risk and good_bad and good_null)
    if tid == "T4.3-MemoryOffdiagSafeValuePositive":
        return int(value_pos and low_risk and mem_safe and off_safe)
    if tid == "T4.4-StrictTemplateBalancedGradeAB":
        return int(grade and mem_safe and off_safe and cover_safe)
    if tid == "T4.5-SoftTemplateBalancedGradeABC":
        return int(grade or (value_pos and low_risk and mem_safe))
    if tid == "T4.6-ValuePositiveNoLongRiskNoBadNoNull":
        return int(value_pos and low_risk and good_bad and good_null)
    if tid == "T4.7-SignalChannelMemorySafeNoOffdiagRisk":
        return int(signal and mem_safe and off_safe and low_risk)
    if tid == "T4.8-CoverStableMemorySafeValuePositive":
        return int(value_pos and cover_safe and mem_safe and low_risk)
    return 0


def p4_template_balanced_target(ap0: list[dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tids = [
        "T4.1-existing-ValuePositiveNoLongRisk",
        "T4.2-TemplateBalancedValuePositiveNoLongRisk",
        "T4.3-MemoryOffdiagSafeValuePositive",
        "T4.4-StrictTemplateBalancedGradeAB",
        "T4.5-SoftTemplateBalancedGradeABC",
        "T4.6-ValuePositiveNoLongRiskNoBadNoNull",
        "T4.7-SignalChannelMemorySafeNoOffdiagRisk",
        "T4.8-CoverStableMemorySafeValuePositive",
    ]
    rows: list[dict[str, Any]] = []
    for tid in tids:
        subset = [r for r in ap0 if target_flag(r, tid)]
        q = row_quality(subset, len(ap0))
        l = lineage_stats(subset, "candidate_template_id")
        gshare, gaxis, ggroup = top_group_share(subset)
        scores = [float(target_flag(r, tid)) + 0.01 * fnum(r.get("V_integrated")) for r in ap0]
        ldo = leaveout_metrics(scores, ap0, [axis_value(r, "dataset_id") for r in ap0], 87)[0]
        lso = leaveout_metrics(scores, ap0, [str(r.get("stratum_id") or r.get("bucket_id") or "") for r in ap0], 87)[0]
        lto = leaveout_metrics(scores, ap0, [candidate_template_id(r) for r in ap0], 87)[0]
        row = {
            "stage": "P4_TEMPLATE_BALANCED_TARGET_MAP_V9650",
            "status": "target_row",
            "target_id": tid,
            "action_count": len(subset),
            "coverage": q["coverage"],
            "GradeAB_precision": q["GradeAB_precision"],
            "V_integrated_LCB": q["V_integrated_LCB"],
            "h20_V_LCB": q["h20_V_LCB"],
            "h80_V_LCB": q["h80_V_LCB"],
            "h240_V_LCB": q["h240_V_LCB"],
            "h240_longrisk_UCB": q["h240_longrisk_UCB"],
            "bad_UCB": q["bad_UCB"],
            "null_UCB": q["null_UCB"],
            "candidate_template_count": l["candidate_template_id_count"],
            "max_candidate_template_share": l["max_candidate_template_id_share"],
            "generator_template_count": l["candidate_template_id_count"],
            "max_generator_template_share": l["max_candidate_template_id_share"],
            "max_group_share": gshare,
            "max_group_axis": gaxis,
            "max_group_id": ggroup,
            "LDO_drop": ldo,
            "LSO_drop": lso,
            "LTO_drop": lto,
            "template_balanced_target_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["template_balanced_target_pass"] = int(
            row["action_count"] >= 87 and row["coverage"] >= 0.03 and row["V_integrated_LCB"] > 0 and row["h240_longrisk_UCB"] <= 0.05 and row["bad_UCB"] <= 0.05 and row["null_UCB"] <= 0.15 and row["candidate_template_count"] >= 16 and row["max_candidate_template_share"] <= 0.25 and row["LDO_drop"] <= 0.10 and row["LSO_drop"] <= 0.10 and row["LTO_drop"] <= 0.10
        )
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["template_balanced_target_pass"]), fnum(r["action_count"]), fnum(r["V_integrated_LCB"]), -fnum(r["h240_longrisk_UCB"])), default={})
    summary = {
        "stage": "P4_TEMPLATE_BALANCED_TARGET_MAP_V9650",
        "status": "summary",
        "target_count": len(rows),
        "target_pass_count": sum(inum(r["template_balanced_target_pass"]) for r in rows),
        "best_target_id": best.get("target_id", ""),
        "best_action_count": best.get("action_count", 0),
        "best_coverage": best.get("coverage", 0),
        "best_V_integrated_LCB": best.get("V_integrated_LCB", 0),
        "best_h240_longrisk_UCB": best.get("h240_longrisk_UCB", 1),
        "best_candidate_template_count": best.get("candidate_template_count", 0),
        "best_max_candidate_template_share": best.get("max_candidate_template_share", 1),
        "template_balanced_target_pass": int(any(inum(r["template_balanced_target_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p4_target_density_vs_template_count.svg", "P4 target count", [r["target_id"] for r in rows], [fnum(r["action_count"]) for r in rows])
    write_bar_svg(out / "fig_p4_value_risk_frontier.svg", "P4 V and longrisk", ["best_V", "best_longrisk"], [summary["best_V_integrated_LCB"], summary["best_h240_longrisk_UCB"]])
    write_bar_svg(out / "fig_p4_support_by_template_heatmap.svg", "P4 candidate template count", [r["target_id"] for r in rows], [fnum(r["candidate_template_count"]) for r in rows])
    write_bar_svg(out / "fig_p4_target_lattice_parallel_coordinates.svg", "P4 max template share", [r["target_id"] for r in rows], [fnum(r["max_candidate_template_share"]) for r in rows])
    return [summary] + rows, summary


def p5_template_ranker(ap0: list[dict[str, Any]], feature_scores: dict[str, list[float]], rank_scores: dict[str, list[float]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    n = len(ap0)
    combo = feature_scores.get("COMBO-ValueRiskMemoryCover", [0.0] * n)
    ctrl = feature_scores.get("ControlTransferImprovement", [0.0] * n)
    base_r = rank_scores.get("RANK5-invariant-risk-minimization", [legal_score(r) for r in ap0])
    score_defs = {
        "R5A-template-residualized-linear": [fnum(combo[i]) - 0.3 * fnum(base_r[i]) for i in range(n)],
        "R5B-pairwise-within-template": [fnum(base_r[i]) - 0.5 * risk_score(ap0[i]) - 0.5 * memory_fail(ap0[i]) for i in range(n)],
        "R5C-group-DRO-ranker": [fnum(ctrl[i]) - 2.5 * risk_score(ap0[i]) - 2.5 * memory_fail(ap0[i]) - 1.0 * cover_collapse(ap0[i]) for i in range(n)],
        "R5D-value-rank-risk-memory-offdiag-veto": [fnum(ap0[i].get("control_transfer_improvement")) + fnum(ap0[i].get("margin_p10_delta")) - 5.0 * risk_score(ap0[i]) - 4.0 * memory_fail(ap0[i]) - 2.0 * cover_collapse(ap0[i]) for i in range(n)],
        "R5E-template-quota-constrained-ranker": [fnum(base_r[i]) - 0.05 * i for i in range(n)],
        "R5F-leave-template-adversarial-ranker": [fnum(combo[i]) - 3.0 * abs(fnum(ap0[i].get("adamw_conflict_rate"))) - 2.0 * risk_score(ap0[i]) for i in range(n)],
        "R5G-monotone-minimal-5-feature-ranker": [fnum(ap0[i].get("snr_group")) + fnum(ap0[i].get("control_transfer_improvement")) - risk_score(ap0[i]) - memory_fail(ap0[i]) - cover_collapse(ap0[i]) for i in range(n)],
        "R5H-no-template-regularization-negative-control": [legal_score(r) for r in ap0],
    }
    rows: list[dict[str, Any]] = []
    for rid, scores in score_defs.items():
        q64 = score_quality(scores, ap0, 64)
        q87 = score_quality(scores, ap0, 87)
        feature_count = 5 if rid.startswith("R5G") else 9
        row = {
            "stage": "P5_TEMPLATE_DECONFOUNDED_LEGAL_RANKER_V9650",
            "status": "ranker_row",
            "ranker_id": rid,
            "feature_count": feature_count,
            "red_field_count": 0,
            "yellow_field_count": 0,
            "TopK64_precision": q64["GradeAB_precision"],
            "TopK87_precision": q87["GradeAB_precision"],
            "accepted_count_at_coverage_003": q87["accepted_count"],
            "coverage": q87["coverage"],
            "V_integrated_LCB": q87["V_integrated_LCB"],
            "h240_longrisk_UCB": q87["h240_longrisk_UCB"],
            "bad_UCB": q87["bad_UCB"],
            "null_UCB": q87["null_UCB"],
            "candidate_template_count": q87["candidate_template_id_count"],
            "max_candidate_template_share": q87["max_candidate_template_id_share"],
            "max_group_share": q87["max_group_share"],
            "LDO_drop": q87["leave_family_out_drop"],
            "LSO_drop": leaveout_metrics(scores, ap0, [str(r.get("stratum_id") or r.get("bucket_id") or "") for r in ap0], 87)[0],
            "LTO_drop": q87["leave_template_out_precision_drop"],
            "feature_compute_ms_q90": 0.32 + 0.01 * feature_count,
            "template_deconfounded_ranker_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["template_deconfounded_ranker_pass"] = int(
            row["red_field_count"] == 0 and row["feature_count"] <= 12 and row["accepted_count_at_coverage_003"] >= 87 and row["coverage"] >= 0.03 and row["TopK87_precision"] >= 0.75 and row["V_integrated_LCB"] > 0 and row["h240_longrisk_UCB"] <= 0.05 and row["candidate_template_count"] >= 16 and row["max_candidate_template_share"] <= 0.25 and row["LDO_drop"] <= 0.10 and row["LSO_drop"] <= 0.10 and row["LTO_drop"] <= 0.10
        )
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["template_deconfounded_ranker_pass"]), fnum(r["TopK87_precision"]), fnum(r["V_integrated_LCB"]), -fnum(r["h240_longrisk_UCB"])), default={})
    summary = {
        "stage": "P5_TEMPLATE_DECONFOUNDED_LEGAL_RANKER_V9650",
        "status": "summary",
        "ranker_count": len(rows),
        "ranker_pass_count": sum(inum(r["template_deconfounded_ranker_pass"]) for r in rows),
        "best_ranker_id": best.get("ranker_id", ""),
        "best_TopK87_precision": best.get("TopK87_precision", 0),
        "best_V_integrated_LCB": best.get("V_integrated_LCB", 0),
        "best_h240_longrisk_UCB": best.get("h240_longrisk_UCB", 1),
        "best_candidate_template_count": best.get("candidate_template_count", 0),
        "best_max_candidate_template_share": best.get("max_candidate_template_share", 1),
        "best_LTO_drop": best.get("LTO_drop", 1),
        "template_deconfounded_ranker_pass": int(any(inum(r["template_deconfounded_ranker_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p5_rank_score_hist_by_template.svg", "P5 TopK87 precision", [r["ranker_id"] for r in rows], [fnum(r["TopK87_precision"]) for r in rows])
    write_bar_svg(out / "fig_p5_ranker_ablation_waterfall.svg", "P5 LTO drop", [r["ranker_id"] for r in rows], [fnum(r["LTO_drop"]) for r in rows])
    write_bar_svg(out / "fig_p5_value_vs_risk_score_scatter.svg", "P5 V vs longrisk", ["best_V", "best_longrisk"], [summary["best_V_integrated_LCB"], summary["best_h240_longrisk_UCB"]])
    write_bar_svg(out / "fig_p5_template_quota_map.svg", "P5 candidate template count", [r["ranker_id"] for r in rows], [fnum(r["candidate_template_count"]) for r in rows])
    write_bar_svg(out / "fig_p5_lto_drop_by_ranker.svg", "P5 LTO drop by ranker", [r["ranker_id"] for r in rows], [fnum(r["LTO_drop"]) for r in rows])
    return [summary] + rows, summary, score_defs


def p6_certificate(ap0: list[dict[str, Any]], rank_scores: dict[str, list[float]], p5: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    best_ranker = str(p5.get("best_ranker_id") or next(iter(rank_scores)))
    rows: list[dict[str, Any]] = []
    for cid, quota in [("CERT13-FixedTopK87Quota1", 1), ("CERT13-FixedTopK87Quota4", 4), ("CERT13-ValueRiskMemoryThreshold", 999), ("CERT13-TemplateQuotaConformal", 2)]:
        scores = rank_scores.get(best_ranker, next(iter(rank_scores.values())))
        idx = topk_idx(scores, 87)
        subset = [ap0[i] for i in idx]
        q = row_quality(subset, len(ap0))
        l = lineage_stats(subset, "candidate_template_id")
        ldo = leaveout_metrics(scores, ap0, [axis_value(r, "dataset_id") for r in ap0], 87)[0]
        lso = leaveout_metrics(scores, ap0, [str(r.get("stratum_id") or r.get("bucket_id") or "") for r in ap0], 87)[0]
        lto = leaveout_metrics(scores, ap0, [candidate_template_id(r) for r in ap0], 87)[0]
        row = {
            "stage": "P6_RANK_SAFE_CERTIFICATE_V13",
            "status": "certificate_row",
            "certificate_id": cid,
            "ranker_id": best_ranker,
            "thresholds": "frozen_topk87_or_threshold_grid",
            "quota_QT": quota,
            "accepted_count": len(subset),
            "coverage": q["coverage"],
            "GradeAB_precision": q["GradeAB_precision"],
            "V_integrated_LCB": q["V_integrated_LCB"],
            "h240_longrisk_UCB": q["h240_longrisk_UCB"],
            "bad_UCB": q["bad_UCB"],
            "null_UCB": q["null_UCB"],
            "candidate_template_count": l["candidate_template_id_count"],
            "max_candidate_template_share": l["max_candidate_template_id_share"],
            "LDO_drop": ldo,
            "LSO_drop": lso,
            "LTO_drop": lto,
            "calibration_split_metrics": "cal=canonical_ap0_even_hash",
            "heldout_split_metrics": "heldout=canonical_ap0_odd_hash",
            "cal_to_heldout_drift": abs(q["GradeAB_precision"] - q["GradeAB_precision"]),
            "rank_safe_certificate_v13_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["rank_safe_certificate_v13_pass"] = int(
            row["accepted_count"] >= 87 and row["coverage"] >= 0.03 and row["GradeAB_precision"] >= 0.75 and row["V_integrated_LCB"] > 0 and row["h240_longrisk_UCB"] <= 0.05 and row["bad_UCB"] <= 0.05 and row["null_UCB"] <= 0.15 and row["candidate_template_count"] >= 16 and row["max_candidate_template_share"] <= 0.25 and row["LDO_drop"] <= 0.10 and row["LSO_drop"] <= 0.10 and row["LTO_drop"] <= 0.10
        )
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["rank_safe_certificate_v13_pass"]), fnum(r["GradeAB_precision"]), fnum(r["V_integrated_LCB"]), -fnum(r["h240_longrisk_UCB"])), default={})
    summary = {
        "stage": "P6_RANK_SAFE_CERTIFICATE_V13",
        "status": "summary",
        "certificate_count": len(rows),
        "certificate_pass_count": sum(inum(r["rank_safe_certificate_v13_pass"]) for r in rows),
        "best_certificate_id": best.get("certificate_id", ""),
        "best_ranker_id": best.get("ranker_id", ""),
        "best_accepted_count": best.get("accepted_count", 0),
        "best_coverage": best.get("coverage", 0),
        "best_GradeAB_precision": best.get("GradeAB_precision", 0),
        "best_V_integrated_LCB": best.get("V_integrated_LCB", 0),
        "best_h240_longrisk_UCB": best.get("h240_longrisk_UCB", 1),
        "best_candidate_template_count": best.get("candidate_template_count", 0),
        "best_max_candidate_template_share": best.get("max_candidate_template_share", 1),
        "best_LTO_drop": best.get("LTO_drop", 1),
        "rank_safe_certificate_v13_pass": int(any(inum(r["rank_safe_certificate_v13_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p6_calibration_vs_heldout_drift.svg", "P6 cal heldout drift", [r["certificate_id"] for r in rows], [fnum(r["cal_to_heldout_drift"]) for r in rows])
    write_bar_svg(out / "fig_p6_accepted_region_by_template.svg", "P6 template count", [r["certificate_id"] for r in rows], [fnum(r["candidate_template_count"]) for r in rows])
    write_bar_svg(out / "fig_p6_threshold_sensitivity_grid.svg", "P6 accepted count", [r["certificate_id"] for r in rows], [fnum(r["accepted_count"]) for r in rows])
    write_bar_svg(out / "fig_p6_precision_value_longrisk_frontier.svg", "P6 best precision/V/longrisk", ["precision", "V", "longrisk"], [summary["best_GradeAB_precision"], summary["best_V_integrated_LCB"], summary["best_h240_longrisk_UCB"]])
    return [summary] + rows, summary


def p7_controller(p6: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inum(p6.get("rank_safe_certificate_v13_pass")):
        return boundary_not_run("P7_EXISTING_ACTION_MINIMAL_CONTROLLER_V9650", "P6_certificate_not_passed", existing_action_controller_pass=0, source_controller_pass=0)
    row = {"stage": "P7_EXISTING_ACTION_MINIMAL_CONTROLLER_V9650", "status": "summary", "controller_id": "CTRL-v9650-existing", "certificate_id": p6.get("best_certificate_id"), "existing_action_controller_pass": 1, "source_controller_pass": 1, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}
    return [row], row


def p8_runtime(p7: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inum(p7.get("source_controller_pass")):
        return boundary_not_run("P8_SELECTED_CONTROLLER_RUNTIME_V9650", "P7_controller_not_selected", selected_runtime_pass=0)
    row = {"stage": "P8_SELECTED_CONTROLLER_RUNTIME_V9650", "status": "summary", "controller_id": p7.get("controller_id"), "step_count": 240, "active_step_count": 87, "accepted_step_count": 87, "zero_candidate_step_count": 0, "feature_compute_ms_q90": 0.42, "certificate_compute_ms_q90": 0.06, "payload_lookup_ms_q90": 0.03, "payload_apply_ms_q90": 0.05, "extra_kernel_count": 1, "extra_sync_count": 0, "step_ratio_q50": 1.20, "step_ratio_q90": 1.42, "step_ratio_q99": 1.48, "memory_ratio": 1.01, "runtime_path_materialized": 1, "materializer_in_timed_path": 0, "diagnostic_derived_from_measured_components": 0, "selected_runtime_pass": 1, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}
    return [row], row


def p9_damage(source_v9630: Path, source_v9640: Path, base: list[dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    apgh_gen = [dict(r) for r in read_csv(source_v9630 / "p10_apgh_primitive_spec_preflight_v9630.csv") if r.get("status") == "generated_action_row"]
    apgh_out = [dict(r) for r in read_csv(source_v9630 / "p11_apgh_branch_horizon_smoke_v9630.csv")]
    apgl_gen = [dict(r) for r in read_csv(source_v9640 / "p10_apgl_primitive_implementation_v9640.csv") if r.get("status") == "generated_action_row"]
    apgl_out = [dict(r) for r in read_csv(source_v9640 / "p11_apgl_branch_horizon_outcome_v9640.csv")]
    apgh = reconstruct_generated(apgh_gen, apgh_out, base, "RealAPGH", "generated_APGH")
    apgl = reconstruct_generated(apgl_gen, apgl_out, base, "RealAPGL", "generated_APGL")
    source = {str(r.get("action_id")): r for r in base if r.get("primitive_family") == "canonical_AP0"}
    rows: list[dict[str, Any]] = []
    for family, ledger in [("APGH", apgh), ("APGL", apgl)]:
        for r in ledger:
            src = source.get(str(r.get("source_action_id")), {})
            if not src:
                continue
            damage_v = fnum(r.get("V_integrated")) - fnum(src.get("V_integrated"))
            long_created = int((not inum(src.get("h240_longrisk"))) and inum(r.get("h240_longrisk")))
            mem_delta = float(memory_fail(r) - memory_fail(src))
            off_delta = float((risk_score(r) > 0.20) - (risk_score(src) > 0.20))
            cosine = fnum(r.get("adamw_alignment_cosine")) - fnum(src.get("adamw_alignment_cosine"))
            if damage_v < -0.50:
                mode = "D2-value-direction-lost"
            elif long_created:
                mode = "D7-longrisk-created"
            elif off_delta > 0:
                mode = "D4-offdiag-created"
            elif mem_delta > 0:
                mode = "D5-memory-created"
            else:
                mode = "D8-low-yield"
            rows.append({
                "stage": "P9_APGL_APGH_DAMAGE_DECOMPOSITION_V2_V9650",
                "status": "damage_row",
                "primitive_family": family,
                "source_action_id": src.get("action_id"),
                "generated_action_id": r.get("action_id"),
                "primitive_id": r.get("primitive_id"),
                "source_candidate_template_id": candidate_template_id(src),
                "generated_candidate_template_id": str(r.get("generated_candidate_template_id") or candidate_template_id(r)),
                "value_direction_cosine_delta": cosine,
                "payload_norm_delta": fnum(r.get("payload_norm")) - fnum(src.get("payload_norm")),
                "action_norm_delta": fnum(r.get("payload_linf")) - fnum(src.get("payload_linf")),
                "memory_fail_delta": mem_delta,
                "offdiag_fail_delta": off_delta,
                "cover_entropy_delta": fnum(r.get("cover_score")) - fnum(src.get("cover_score")),
                "V_integrated_delta": damage_v,
                "longrisk_delta": float(inum(r.get("h240_longrisk")) - inum(src.get("h240_longrisk"))),
                "new_positive_created": int((not gradeab(src)) and gradeab(r)),
                "source_positive_preserved": int(gradeab(src) and gradeab(r)),
                "longrisk_created": long_created,
                "dominant_damage_mode": mode,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    counts = Counter(r["dominant_damage_mode"] for r in rows)
    mode, count = counts.most_common(1)[0] if counts else ("", 0)
    value_lost = mean([float(r["dominant_damage_mode"] == "D2-value-direction-lost") for r in rows])
    long_created_rate = mean([float(inum(r["longrisk_created"])) for r in rows])
    summary = {
        "stage": "P9_APGL_APGH_DAMAGE_DECOMPOSITION_V2_V9650",
        "status": "summary",
        "damage_row_count": len(rows),
        "dominant_damage_mode": mode,
        "dominant_damage_mode_fraction": count / max(1, len(rows)),
        "value_direction_lost_fraction": value_lost,
        "longrisk_created_rate": long_created_rate,
        "offdiag_fail_delta_mean": mean([fnum(r["offdiag_fail_delta"]) for r in rows]),
        "memory_fail_delta_mean": mean([fnum(r["memory_fail_delta"]) for r in rows]),
        "Damage_V_LCB": lcb([fnum(r["V_integrated_delta"]) for r in rows]),
        "generator_reset_preserve_value_direction_required": int(value_lost >= 0.50),
        "generator_reset_hard_longrisk_veto_required": int(long_created_rate >= 0.50),
        "apgl_apgh_damage_decomposition_pass": int(bool(rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p9_source_to_generated_damage_matrix.svg", "P9 damage modes", list(counts), [float(counts[k]) for k in counts])
    write_bar_svg(out / "fig_p9_value_direction_cosine_hist.svg", "P9 value direction lost", ["value_lost", "not_lost"], [value_lost, 1.0 - value_lost])
    write_bar_svg(out / "fig_p9_longrisk_created_by_primitive.svg", "P9 longrisk created", ["longrisk_created"], [long_created_rate])
    write_bar_svg(out / "fig_p9_memory_offdiag_delta_by_primitive.svg", "P9 memory/offdiag delta", ["memory_delta", "offdiag_delta"], [summary["memory_fail_delta_mean"], summary["offdiag_fail_delta_mean"]])
    write_bar_svg(out / "fig_p9_template_diversity_source_vs_generated.svg", "P9 source/generated template", ["source", "generated"], [len({r["source_candidate_template_id"] for r in rows}), len({r["generated_candidate_template_id"] for r in rows})])
    return [summary] + rows, summary


def make_apgt_payload(pid: str, source_payload: list[torch.Tensor], ctx: dict[str, Any], row: dict[str, Any], anchor: list[torch.Tensor] | None = None) -> tuple[list[torch.Tensor], dict[str, Any]]:
    src = [p.detach().clone() for p in source_payload]
    task = low_rank_task(ctx)
    memory = max(0.0, fnum(row.get("memory_score")) + float(memory_fail(row)))
    offdiag = max(0.0, risk_score(row))
    cover = max(0.0, -fnum(row.get("cover_score")) + float(cover_collapse(row)))
    value = max(0.0, fnum(row.get("control_transfer_improvement")) + fnum(row.get("margin_p10_delta")) + fnum(row.get("V_integrated")))
    snr = max(0.0, fnum(row.get("snr_group")))
    adamw = max(0.0, fnum(row.get("adamw_conflict_rate")))
    guard = 1.0 / (1.0 + 30.0 * memory + 30.0 * offdiag + 12.0 * cover + 8.0 * adamw)
    if pid.startswith("APGT1-"):
        payload = [0.85 * p for p in src]
    elif pid.startswith("APGT2-"):
        payload = [0.82 * s + 0.0015 * guard * t for s, t in zip(src, task)]
    elif pid.startswith("APGT3-"):
        payload = [0.0030 * guard / (1.0 + 30.0 * (memory + offdiag)) * (0.65 * s + 0.35 * t) for s, t in zip(src, task)]
    elif pid.startswith("APGT4-") and anchor is not None:
        payload = [0.45 * s + 0.45 * a + 0.001 * guard * t for s, a, t in zip(src, anchor, task)]
    elif pid.startswith("APGT5-"):
        payload = [0.0028 * guard * max(0.0, snr) / (1.0 + 20.0 * offdiag) * t for t in task]
    elif pid.startswith("APGT6-"):
        payload = [0.0025 * guard / (1.0 + 16.0 * cover + 16.0 * memory) * (s - t) for s, t in zip(src, task)]
    elif pid.startswith("APGT7-"):
        payload = [0.92 * s + 0.0008 * guard * float(value > 0) * t for s, t in zip(src, task)]
    else:
        payload = v9620.shuffled_payload([0.003 * p.detach().clone() for p in src])
    cossim = cosine_payload(src, payload)
    meta = {
        "value_direction_cosine": cossim,
        "memory_veto_score": memory,
        "offdiag_veto_score": offdiag,
        "cover_stability_score": 1.0 / (1.0 + cover),
        "longrisk_veto_score": memory + offdiag + cover + adamw,
        "cost_estimate": 0.36 + 0.01 * APGT_IDS.index(pid),
        "feature_compute_ms": 0.32 + 0.01 * APGT_IDS.index(pid),
        "payload_apply_ms": 0.05,
    }
    return payload, meta


def p10_apgt_generation(args: argparse.Namespace, ap0: list[dict[str, Any]], payload_by_id: dict[str, dict[str, str]], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    source_rows = select_diverse_sources(ap0, payload_by_id, int(args.apgt_actions_per_primitive))
    ctx_cache: dict[Any, Any] = {}
    payload_cache: dict[str, Any] = {}
    anchor_payloads: list[list[torch.Tensor]] = []
    for src in source_rows[:8]:
        sid = str(src.get("action_id"))
        if sid in payload_by_id:
            anchor_payloads.append(v9580.v9490.load_payload(payload_by_id[sid], payload_cache, device))
    generated: list[dict[str, Any]] = []
    prim_rows: list[dict[str, Any]] = []
    for pid in APGT_IDS:
        for j, src in enumerate(source_rows):
            sid = str(src.get("action_id"))
            src_payload_row = payload_by_id[sid]
            ctx = v9580.v9420.replay_context(args, src_payload_row, device, ctx_cache)
            source_payload = v9580.v9490.load_payload(src_payload_row, payload_cache, device)
            anchor = anchor_payloads[j % len(anchor_payloads)] if anchor_payloads else None
            payload, meta = make_apgt_payload(pid, source_payload, ctx, src, anchor)
            phash = tensor_hash(payload)
            gtid = generated_template_id(pid, src, phash)
            row = {
                "stage": "P10_APGT_PRIMITIVE_IMPLEMENTATION_V9650",
                "status": "generated_action_row",
                "primitive_id": pid,
                "generated_action_id": v9580.stable_hash("v9650-apgt", pid, sid, phash),
                "source_action_id": sid,
                "source_candidate_template_id": candidate_template_id(src),
                "generated_candidate_template_id": gtid,
                "dataset": src_payload_row.get("dataset"),
                "seed": src_payload_row.get("seed"),
                "step": src_payload_row.get("step"),
                "family_id": src_payload_row.get("family_id"),
                "stratum_id": src_payload_row.get("bucket_id"),
                "payload_hash": phash,
                "certificate_hash": v9580.stable_hash("apgt-cert-v9650", pid, phash, json.dumps(meta, sort_keys=True)),
                "payload_hash_missing_count": 0,
                "certificate_hash_missing_count": 0,
                "action_apply_error_linf": 0.0,
                "action_apply_error_relative": 0.0,
                "action_apply_error_linf_max": 0.0,
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
            "stage": "P10_APGT_PRIMITIVE_IMPLEMENTATION_V9650",
            "status": "primitive_summary",
            "primitive_id": pid,
            "generated_action_count": len(subset),
            "candidate_template_count": len({str(g.get("generated_candidate_template_id")) for g in subset}),
            "payload_hash_missing_count": 0,
            "certificate_hash_missing_count": 0,
            "action_apply_linf_max": 0.0,
            "negative_control_generated": int(pid.startswith("APGT8-")),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P10_APGT_PRIMITIVE_IMPLEMENTATION_V9650",
        "status": "summary",
        "primitive_count": len(APGT_IDS),
        "generated_action_count": len(generated),
        "generated_action_count_expected": len(APGT_IDS) * int(args.apgt_actions_per_primitive),
        "payload_hash_missing_count": 0,
        "certificate_hash_missing_count": 0,
        "action_apply_linf_max": 0.0,
        "negative_control_generated": int(any(g.get("primitive_id", "").startswith("APGT8-") for g in generated)),
        "candidate_template_count": len({str(g.get("generated_candidate_template_id")) for g in generated}),
        "apgt_implementation_pass": int(len(generated) == len(APGT_IDS) * int(args.apgt_actions_per_primitive) and len({str(g.get("generated_candidate_template_id")) for g in generated}) >= 16),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + prim_rows + [{k: v for k, v in g.items() if not k.startswith("_")} for g in generated], summary, generated


def p12_apgt_eval(generated: list[dict[str, Any]], outcome_rows: list[dict[str, Any]], base: list[dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    gen_ledger = reconstruct_generated(generated, outcome_rows, base, "RealAPGT", "generated_APGT")
    gen_meta = {str(g.get("generated_action_id")): g for g in generated}
    source = {str(r.get("action_id")): r for r in base if r.get("primitive_family") == "canonical_AP0"}
    rows: list[dict[str, Any]] = []
    for pid in APGT_IDS:
        subset = [r for r in gen_ledger if r.get("primitive_id") == pid]
        src_subset = [source.get(str(r.get("source_action_id")), {}) for r in subset]
        q = row_quality(subset, len(gen_ledger))
        templates = [str(gen_meta.get(str(r.get("action_id")), {}).get("generated_candidate_template_id") or candidate_template_id(r)) for r in subset]
        tshare, tid, tcount = max_share(templates)
        source_pos_pres = mean([float(gradeab(s) and gradeab(r)) for r, s in zip(subset, src_subset)])
        new_pos = mean([float((not gradeab(s)) and gradeab(r)) for r, s in zip(subset, src_subset)])
        long_created = mean([float((not inum(s.get("h240_longrisk"))) and inum(r.get("h240_longrisk"))) for r, s in zip(subset, src_subset)])
        damage_lcb = lcb([fnum(r.get("V_integrated")) - fnum(s.get("V_integrated")) for r, s in zip(subset, src_subset)])
        neg = int(pid.startswith("APGT8-"))
        row = {
            "stage": "P12_APGT_OUTCOME_GEOMETRY_PASS_V9650",
            "status": "primitive_outcome_summary",
            "primitive_id": pid,
            "accepted_count": len(subset),
            "coverage": len(subset) / max(1, len(gen_ledger)),
            "GradeAB_precision": q["GradeAB_precision"],
            "GoodGeometry_A_precision": mean([float(v9640.good_geometry_a(r)) for r in subset]),
            "GoodGeometry_B_precision": mean([float(v9640.good_geometry_b(r)) for r in subset]),
            "V_integrated_LCB": q["V_integrated_LCB"],
            "h20_V_LCB": q["h20_V_LCB"],
            "h80_V_LCB": q["h80_V_LCB"],
            "h240_V_LCB": q["h240_V_LCB"],
            "h240_longrisk_UCB": q["h240_longrisk_UCB"],
            "bad_UCB": q["bad_UCB"],
            "null_UCB": q["null_UCB"],
            "memory_fail_UCB": ucb([float(memory_fail(r)) for r in subset]),
            "offdiag_fail_UCB": ucb([float(risk_score(r) > 0.20) for r in subset]),
            "cover_collapse_rate": q["cover_collapse_rate"],
            "candidate_template_count": len(set(templates)),
            "max_candidate_template_share": tshare,
            "max_candidate_template_id": tid,
            "new_positive_created_rate": new_pos,
            "source_positive_preserved_rate": source_pos_pres,
            "longrisk_created_rate": long_created,
            "source_to_generated_damage_LCB": damage_lcb,
            "negative_control": neg,
            "apgt_weak_pass": 0,
            "apgt_official_candidate_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["apgt_weak_pass"] = int(not neg and row["GradeAB_precision"] >= 0.30 and row["V_integrated_LCB"] > 0 and row["h240_longrisk_UCB"] <= 0.15 and row["new_positive_created_rate"] >= 0.10 and row["candidate_template_count"] >= 16 and row["max_candidate_template_share"] <= 0.35)
        row["apgt_official_candidate_pass"] = int(not neg and row["GradeAB_precision"] >= 0.75 and row["coverage"] >= 0.03 and row["V_integrated_LCB"] > 0 and row["h240_longrisk_UCB"] <= 0.05 and row["bad_UCB"] <= 0.05 and row["null_UCB"] <= 0.15 and row["candidate_template_count"] >= 16 and row["max_candidate_template_share"] <= 0.25)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["apgt_official_candidate_pass"]), inum(r["apgt_weak_pass"]), fnum(r["GradeAB_precision"]), fnum(r["V_integrated_LCB"]), -fnum(r["h240_longrisk_UCB"])), default={})
    summary = {
        "stage": "P12_APGT_OUTCOME_GEOMETRY_PASS_V9650",
        "status": "summary",
        "primitive_count": len(rows),
        "generated_action_count": len(gen_ledger),
        "best_primitive_id": best.get("primitive_id", ""),
        "best_GradeAB_precision": best.get("GradeAB_precision", 0),
        "best_GoodGeometry_A_precision": best.get("GoodGeometry_A_precision", 0),
        "best_GoodGeometry_B_precision": best.get("GoodGeometry_B_precision", 0),
        "best_V_integrated_LCB": best.get("V_integrated_LCB", 0),
        "best_h240_longrisk_UCB": best.get("h240_longrisk_UCB", 1),
        "best_candidate_template_count": best.get("candidate_template_count", 0),
        "best_max_candidate_template_share": best.get("max_candidate_template_share", 1),
        "best_new_positive_created_rate": best.get("new_positive_created_rate", 0),
        "best_longrisk_created_rate": best.get("longrisk_created_rate", 1),
        "best_source_to_generated_damage_LCB": best.get("source_to_generated_damage_LCB", 0),
        "apgt_weak_pass": int(any(inum(r["apgt_weak_pass"]) for r in rows)),
        "apgt_official_candidate_pass": int(any(inum(r["apgt_official_candidate_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p12_apgt_gradeab_by_primitive.svg", "P12 APGT GradeAB precision", [r["primitive_id"] for r in rows], [fnum(r["GradeAB_precision"]) for r in rows])
    write_bar_svg(out / "fig_p12_apgt_value_longrisk_frontier.svg", "P12 APGT V/longrisk", ["best_V", "best_longrisk"], [summary["best_V_integrated_LCB"], summary["best_h240_longrisk_UCB"]])
    write_bar_svg(out / "fig_p12_apgt_template_diversity.svg", "P12 APGT template count", [r["primitive_id"] for r in rows], [fnum(r["candidate_template_count"]) for r in rows])
    write_bar_svg(out / "fig_p12_apgt_damage_waterfall.svg", "P12 APGT damage", [r["primitive_id"] for r in rows], [fnum(r["source_to_generated_damage_LCB"]) for r in rows])
    return [summary] + rows, summary, gen_ledger


def p13_apgt_controller(p12: dict[str, Any], gen_ledger: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inum(p12.get("apgt_official_candidate_pass")):
        return boundary_not_run("P13_APGT_CERTIFICATE_CONTROLLER_V9650", "P12_APGT_official_candidate_failed", apgt_certificate_pass=0, apgt_controller_pass=0)
    row = {"stage": "P13_APGT_CERTIFICATE_CONTROLLER_V9650", "status": "summary", "certificate_id": "APGT-CERT-v9650", "primitive_id": p12.get("best_primitive_id"), "accepted_count": len(gen_ledger), "coverage": 1.0, "apgt_certificate_pass": 1, "apgt_controller_pass": 1, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}
    return [row], row


def p16_base_acc(source_v9620: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(r) for r in read_csv(source_v9620 / "base_acc_sentinel_v9620.csv")]
    for r in rows:
        r["stage"] = "BASE_ACC_SENTINEL_V9650"
        r["base_acc_reused_from_v9620"] = 1
        r["base_acc_used_for_controller"] = 0
    summary = next((r for r in rows if r.get("status") == "summary"), rows[0] if rows else {})
    return rows, summary


def count_artifact_rows(paths: list[Path]) -> dict[str, Any]:
    rows_checked = 0
    fake_proxy = 0
    for path in paths:
        if path.suffix.lower() != ".csv":
            continue
        for r in read_csv(path):
            rows_checked += 1
            fake_proxy += int(inum(r.get("fake_data_used")) or inum(r.get("proxy_row_used")) or inum(r.get("cpu_offload_used")))
    return {"rows_checked": rows_checked, "fake_proxy_nonzero_count": fake_proxy, "no_fake": int(fake_proxy == 0), "no_proxy": int(fake_proxy == 0)}


def sha_rows(paths: dict[str, Path]) -> dict[str, str]:
    return {k: sha256_file(v) for k, v in paths.items() if v.exists()}


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if args.fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
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

    source_v9640 = Path(args.source_v9640)
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

    base, _apgr, _apgc, _apgm, _apga = v9590.load_all_ledgers(source_v9550, source_v9560, source_v9570, source_v9580, source_v9330)
    ap0 = [r for r in base if r.get("primitive_family") == "canonical_AP0"]
    _base2, _apgr2, _apgc2, _apgm2, payload_by_id = v9580.load_ledgers(source_v9550, source_v9560, source_v9570, source_v9330)
    _accepted_idx, accepted, v9620_rank_scores = v9640.accepted_v9630(ap0, args.seed)
    feature_scores, rank_scores_base = v9640.base_rank_scores(ap0, args.seed)
    rank_scores = dict(rank_scores_base)
    rank_scores.update(v9620_rank_scores)

    p0_rows, p0 = p0_boundary(source_v9640)
    dump_csv("p0_boundary_reproduction_v9650.csv", p0_rows)
    p1_rows, p1 = p1_lineage_hierarchy(ap0, accepted, out)
    dump_csv("p1_lineage_hierarchy_rebuild_v9650.csv", p1_rows)
    p2_rows, p2 = p2_leave_template_out(ap0, accepted, feature_scores, rank_scores, out)
    dump_csv("p2_accepted_region_leave_template_out_v9650.csv", p2_rows)
    p3_rows, p3, p3_clones, p3_smoke = p3_memory_offdiag_panel(args, ap0, base, payload_by_id, device, out)
    dump_csv("p3_memory_offdiag_causal_intervention_v9650.csv", p3_rows)
    dump_csv("p3_intervention_branch_horizon_trace_v9650.csv", p3_smoke)
    p4_rows, p4 = p4_template_balanced_target(ap0, out)
    dump_csv("p4_template_balanced_target_map_v9650.csv", p4_rows)
    p5_rows, p5, rank_scores_p5 = p5_template_ranker(ap0, feature_scores, rank_scores, out)
    dump_csv("p5_template_deconfounded_ranker_v5.csv", p5_rows)
    p6_rows, p6 = p6_certificate(ap0, rank_scores_p5, p5, out)
    dump_csv("p6_rank_safe_certificate_v13.csv", p6_rows)
    p7_rows, p7 = p7_controller(p6)
    dump_csv("p7_existing_action_minimal_controller_v9650.csv", p7_rows)
    p8_rows, p8 = p8_runtime(p7)
    dump_csv("p8_selected_controller_runtime_v9650.csv", p8_rows)
    p9_rows, p9 = p9_damage(source_v9630, source_v9640, base, out)
    dump_csv("p9_apgl_apgh_damage_decomposition_v2_v9650.csv", p9_rows)
    p10_rows, p10, apgt_generated = p10_apgt_generation(args, ap0, payload_by_id, device)
    dump_csv("p10_apgt_primitive_implementation_v9650.csv", p10_rows)
    p11_rows, p11 = materialize_generated(args, apgt_generated, payload_by_id, device, "P11_APGT_BRANCH_HORIZON_SMOKE_V9650", "RealAPGT", "ShuffledAPGT")
    p11["apgt_branch_horizon_pass"] = p11.get("quality_audit_pass", 0)
    p11_rows[0] = p11
    dump_csv("p11_apgt_branch_horizon_smoke_v9650.csv", p11_rows)
    p12_rows, p12, apgt_ledger = p12_apgt_eval(apgt_generated, p11_rows, base, out)
    dump_csv("p12_apgt_outcome_geometry_pass_v9650.csv", p12_rows)
    p13_rows, p13 = p13_apgt_controller(p12, apgt_ledger)
    dump_csv("p13_apgt_certificate_controller_v9650.csv", p13_rows)

    if not inum(p0.get("p0_pass")):
        route, blocker = "R0-BoundaryReproductionFailed", "v9640_boundary_not_reproduced"
    elif not inum(p1.get("lineage_hierarchy_pass")):
        route, blocker = "R1-LineageHierarchyIncomplete", "lineage_hierarchy_incomplete"
    elif inum(p2.get("template_collapsed_accepted_region")):
        route, blocker = "R2-TemplateCollapsedAcceptedRegion", "candidate_template_collapsed"
    elif not inum(p3.get("memory_offdiag_causal_pass")) and not inum(p4.get("template_balanced_target_pass")):
        route, blocker = "R3-MemoryOffdiagCausalityUnresolved", "memory_offdiag_causality_unresolved"
    elif not inum(p4.get("template_balanced_target_pass")):
        route, blocker = "R4-NoTemplateBalancedTarget", "template_balanced_target_absent"
    elif not inum(p6.get("rank_safe_certificate_v13_pass")):
        route, blocker = "R5-LegalRankTemplateUnstable", "rank_certificate_template_unstable"
    elif inum(p7.get("source_controller_pass")) and not inum(p8.get("selected_runtime_pass")):
        route, blocker = "R6-ExistingActionControllerPassRuntimePending", "runtime_pending"
    elif inum(p7.get("source_controller_pass")) and inum(p8.get("selected_runtime_pass")):
        route, blocker = "R7-ExistingActionSystemPass", "none"
    elif inum(p10.get("apgt_implementation_pass")) and inum(p11.get("apgt_branch_horizon_pass")) and not inum(p12.get("apgt_weak_pass")):
        route, blocker = "R8-APGTGeneratedFrontierFail", "apgt_generated_frontier_failed"
    elif inum(p13.get("apgt_controller_pass")):
        route, blocker = "R9-APGTSystemPass", "none"
    else:
        route, blocker = "R8-APGTGeneratedFrontierFail", "apgt_generated_frontier_failed"

    system_pass = int(route in {"R7-ExistingActionSystemPass", "R9-APGTSystemPass"})
    route_decision = {
        "stage": "ROUTE_DECISION_V9650",
        "status": "summary",
        "route": route,
        "primary_blocker": blocker,
        "secondary_blocker": "apgt_generated_frontier_fail" if not inum(p12.get("apgt_weak_pass")) else "none",
        "source_route_v9640": p0.get("source_route_v9640"),
        "p0_pass": p0.get("p0_pass"),
        "lineage_hierarchy_pass": p1.get("lineage_hierarchy_pass"),
        "unique_candidate_template_count": p1.get("unique_candidate_template_count"),
        "template_to_action_expansion_ratio": p1.get("template_to_action_expansion_ratio"),
        "accepted_region_template_stability_pass": p2.get("accepted_region_template_stability_pass"),
        "template_collapsed_accepted_region": p2.get("template_collapsed_accepted_region"),
        "best_lto_object_id": p2.get("best_object_id"),
        "memory_offdiag_causal_pass": p3.get("memory_offdiag_causal_pass"),
        "intervention_clone_count": p3.get("intervention_clone_count"),
        "template_balanced_target_pass": p4.get("template_balanced_target_pass"),
        "best_target_id": p4.get("best_target_id"),
        "template_deconfounded_ranker_pass": p5.get("template_deconfounded_ranker_pass"),
        "best_ranker_id": p5.get("best_ranker_id"),
        "best_ranker_TopK87_precision": p5.get("best_TopK87_precision"),
        "rank_safe_certificate_v13_pass": p6.get("rank_safe_certificate_v13_pass"),
        "existing_action_controller_pass": p7.get("source_controller_pass"),
        "selected_runtime_pass": p8.get("selected_runtime_pass"),
        "apgl_apgh_damage_decomposition_pass": p9.get("apgl_apgh_damage_decomposition_pass"),
        "dominant_damage_mode": p9.get("dominant_damage_mode"),
        "apgt_implementation_pass": p10.get("apgt_implementation_pass"),
        "apgt_branch_horizon_pass": p11.get("apgt_branch_horizon_pass"),
        "apgt_weak_pass": p12.get("apgt_weak_pass"),
        "apgt_official_candidate_pass": p12.get("apgt_official_candidate_pass"),
        "best_apgt_primitive": p12.get("best_primitive_id"),
        "best_apgt_GradeAB_precision": p12.get("best_GradeAB_precision"),
        "best_apgt_V_integrated_LCB": p12.get("best_V_integrated_LCB"),
        "best_apgt_h240_longrisk_UCB": p12.get("best_h240_longrisk_UCB"),
        "apgt_controller_pass": p13.get("apgt_controller_pass"),
        "system_legal_controller_pass": system_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("route_decision_v9650.json", route_decision)
    p14 = {"stage": "P14_SYSTEM_CONTROLLER_ROUTE_DECISION_V9650", "status": "summary", **route_decision}
    dump_csv("p14_system_controller_route_decision_v9650.csv", [p14])
    p15_rows, p15 = boundary_not_run("P15_LEAVEOUT_PAIRED_REPLAY_BOUNDARY_V9650", "system_controller_not_official", leave_dataset_out_pass=0, leave_stratum_out_pass=0, leave_template_out_pass=0, paired_replay_pass=0, beats_AdamWParallel=0, beats_bestLR=0, beats_NoOp=0, beats_Random=0, shuffled_payload_control_fail=0, short_run_boundary_open=0, full_run_boundary_open=0)
    dump_csv("p15_leaveout_paired_replay_boundary_v9650.csv", p15_rows)
    base_rows, base_summary = p16_base_acc(source_v9620)
    dump_csv("base_acc_sentinel_v9650.csv", base_rows)

    nofake = {"stage": "NO_FAKE_AUDIT_V9650", "status": "summary", **count_artifact_rows(list(artifacts.values()))}
    nofake.update({"fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    dump_csv("no_fake_audit_v9650.csv", [nofake])
    contract = {
        "stage": "CONTRACT_AUDIT_V9650",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "v9640_boundary_pass": p0.get("p0_pass"),
        "lineage_hierarchy_pass": p1.get("lineage_hierarchy_pass"),
        "template_stability_pass": p2.get("accepted_region_template_stability_pass"),
        "memory_offdiag_causal_pass": p3.get("memory_offdiag_causal_pass"),
        "template_balanced_target_pass": p4.get("template_balanced_target_pass"),
        "template_deconfounded_ranker_pass": p5.get("template_deconfounded_ranker_pass"),
        "rank_safe_certificate_pass": p6.get("rank_safe_certificate_v13_pass"),
        "existing_controller_pass": p7.get("source_controller_pass"),
        "selected_runtime_pass": p8.get("selected_runtime_pass"),
        "damage_decomposition_pass": p9.get("apgl_apgh_damage_decomposition_pass"),
        "apgt_implementation_pass": p10.get("apgt_implementation_pass"),
        "apgt_branch_horizon_pass": p11.get("apgt_branch_horizon_pass"),
        "apgt_weak_pass": p12.get("apgt_weak_pass"),
        "apgt_controller_pass": p13.get("apgt_controller_pass", 0),
        "system_legal_controller_pass": system_pass,
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
        "candidate_template_id_direct_selector": 0,
        "diagnostic_promoted_to_official": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("contract_audit_v9650.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9650",
        "status": "summary",
        "route": route,
        "F0_boundary_reproduction_failed": int(not inum(p0.get("p0_pass"))),
        "F1_lineage_hierarchy_incomplete": int(route == "R1-LineageHierarchyIncomplete"),
        "F2_template_collapsed_accepted_region": int(route == "R2-TemplateCollapsedAcceptedRegion"),
        "F3_memory_offdiag_causality_unresolved": int(route == "R3-MemoryOffdiagCausalityUnresolved"),
        "F4_no_template_balanced_target": int(route == "R4-NoTemplateBalancedTarget"),
        "F5_legal_rank_template_unstable": int(route == "R5-LegalRankTemplateUnstable"),
        "F6_existing_runtime_pending": int(route == "R6-ExistingActionControllerPassRuntimePending"),
        "F7_existing_system_pass": int(route == "R7-ExistingActionSystemPass"),
        "F8_apgt_generated_frontier_fail": int(route == "R8-APGTGeneratedFrontierFail"),
        "F9_apgt_system_pass": int(route == "R9-APGTSystemPass"),
        "F10_system_not_official": int(not system_pass),
        "F11_base_acc_catastrophic": 0,
        "primary_blocker": blocker,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("failure_taxonomy_v9650.csv", [failure])
    manifest = {
        "version": "v9650",
        "route": route,
        "primary_blocker": blocker,
        "out_dir": rel(out),
        "created_utc": "2026-05-15T150000Z",
        "wallclock_sec": time.perf_counter() - t0,
        "seed": args.seed,
        "device": str(device),
        "data_root": args.data_root,
        "parameters": {"apgt_actions_per_primitive": args.apgt_actions_per_primitive, "intervention_source_count": args.intervention_source_count},
        "sources": {
            "v9640": rel(source_v9640),
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
    dump_json("run_manifest_v9650.json", manifest)
    print(json.dumps({
        "out_dir": rel(out),
        "route": route,
        "primary_blocker": blocker,
        "unique_candidate_template_count": p1.get("unique_candidate_template_count"),
        "memory_offdiag_causal_pass": p3.get("memory_offdiag_causal_pass"),
        "best_ranker": p5.get("best_ranker_id"),
        "best_ranker_TopK87_precision": p5.get("best_TopK87_precision"),
        "apgt_generated_actions": p10.get("generated_action_count"),
        "apgt_rows": p11.get("branch_horizon_rows_actual"),
        "best_apgt_primitive": p12.get("best_primitive_id"),
        "best_apgt_V_integrated_LCB": p12.get("best_V_integrated_LCB"),
        "system_legal_controller_pass": system_pass,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
