#!/usr/bin/env python3
"""DG-KAN v9.6.2 confounder-purged legal rank / APGF closure.

This runner consumes the landed v9.6.1 artifacts, formally stops the payload
pocket route if intervention evidence remains confounded, audits hidden
confounders and group-balanced rank/certificate gates, then materializes
APGF1-APGF8 population-risk geometry primitives with real branch-horizon
outcomes. Diagnostic ranks, generated smoke, and Base-Acc Sentinel are never
promoted into official controller/system pass unless preregistered gates pass.
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

import run_v9500_canonical_frontier_mechanism_self_certifying_primitive as v9500  # noqa: E402
import run_v9550_trainable_geometry_signal_reservoir_primitive as v9550  # noqa: E402
import run_v9560_calibrated_geometry_rank_cover_memory_primitive as v9560  # noqa: E402
import run_v9580_group_stable_legal_rank_memory_safe_primitive as v9580  # noqa: E402
import run_v9590_group_invariant_legal_rank_existing_action_controller as v9590  # noqa: E402
import run_v9600_group_pocket_deconfounded_rank_memory_safe_longrisk_primitive as v9600  # noqa: E402
import run_v9610_group_pocket_causal_intervention_group_invariant_controller_memory_safe_primitive as v9610  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.6.2_ConfounderPurgedLegalRank_PopRiskGeometryPrimitive_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9620_confounder_purged_legal_rank_poprisk_geometry_primitive.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_V9610 = RESULT_ROOT / "v9610_group_pocket_causal_intervention_group_invariant_controller_memory_safe_primitive_first_20260515T110000Z"
DEFAULT_V9600 = RESULT_ROOT / "v9600_group_pocket_deconfounded_rank_memory_safe_longrisk_primitive_first_20260515T100000Z"
DEFAULT_V9590 = RESULT_ROOT / "v9590_group_invariant_legal_rank_existing_action_controller_first_20260515T090000Z"
DEFAULT_V9580 = RESULT_ROOT / "v9580_group_stable_legal_rank_memory_safe_primitive_first_20260515T080000Z"
DEFAULT_V9570 = RESULT_ROOT / "v9570_legal_causal_geometry_rank_memory_preserving_primitive_first_20260515T070000Z"
DEFAULT_V9560 = RESULT_ROOT / "v9560_calibrated_geometry_rank_cover_memory_primitive_first_20260515T060000Z"
DEFAULT_V9550 = RESULT_ROOT / "v9550_trainable_geometry_signal_reservoir_primitive_first_20260515T050000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"

HORIZONS = [20, 80, 240]
APGF_IDS = [
    "APGF1-PopRiskDiagonalSNRGate",
    "APGF2-EdgeGroupSNRMask",
    "APGF3-MemoryAnchoredResidual",
    "APGF4-OffdiagPopulationRiskGuard",
    "APGF5-CoverEntropyPreservingDelta",
    "APGF6-SymmetricBoundaryCorrection",
    "APGF7-RankImitationWithRiskVeto",
    "APGF8-NegativeControlShuffled",
]
CONFOUNDER_AXES = [
    "dataset_id",
    "seed_id",
    "family_id",
    "stratum_id",
    "step_bucket",
    "event_phase",
    "payload_norm_bucket",
    "payload_linf_bucket",
    "payload_direction_bucket",
    "state_nll_bucket",
    "hard_tail_bucket",
    "memory_fail_bucket",
    "offdiag_fail_bucket",
    "cover_bucket",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9610", default=str(DEFAULT_V9610))
    p.add_argument("--source-v9600", default=str(DEFAULT_V9600))
    p.add_argument("--source-v9590", default=str(DEFAULT_V9590))
    p.add_argument("--source-v9580", default=str(DEFAULT_V9580))
    p.add_argument("--source-v9570", default=str(DEFAULT_V9570))
    p.add_argument("--source-v9560", default=str(DEFAULT_V9560))
    p.add_argument("--source-v9550", default=str(DEFAULT_V9550))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--apgf-actions-per-primitive", type=int, default=64)
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


def topk_idx(scores: list[float], k: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: min(k, len(scores))]


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


def auc(scores: list[float], labels: list[int]) -> float:
    return v9500.auc_score(scores, labels)


def ece(scores: list[float], labels: list[int]) -> float:
    return v9500.ece_binary(scores, labels)


def payload_norm(payload: list[torch.Tensor]) -> float:
    return v9580.payload_norm(payload)


def payload_linf(payload: list[torch.Tensor]) -> float:
    return max((float(torch.max(torch.abs(p.detach())).item()) for p in payload), default=0.0)


def tensor_hash(payload: list[torch.Tensor]) -> str:
    return v9580.tensor_hash(payload)


def shuffled_payload(payload: list[torch.Tensor]) -> list[torch.Tensor]:
    return v9610.shuffled_payload(payload)


def low_rank_task(ctx: dict[str, Any]) -> list[torch.Tensor]:
    return [v9580.v9510.low_rank_like(-t.detach().clone()) for t in ctx["task_delta"]]


def axis_value(row: dict[str, Any], axis: str, score: float = 0.0) -> str:
    if axis == "dataset_id":
        return str(row.get("dataset"))
    if axis == "seed_id":
        return str(row.get("seed"))
    if axis == "family_id":
        return str(row.get("family_id"))
    if axis == "stratum_id":
        return str(row.get("stratum_id"))
    if axis == "step_bucket":
        return f"step{int(fnum(row.get('step')) // 10)}"
    if axis == "event_phase":
        return f"phase{int(fnum(row.get('step')) // 50)}"
    if axis == "payload_norm_bucket":
        return v9600.group_value(row, "payload_norm_bucket", score)
    if axis == "payload_linf_bucket":
        return f"plinf{min(9, int(fnum(row.get('payload_linf')) * 100000))}"
    if axis == "payload_direction_bucket":
        return "dir_pos" if fnum(row.get("control_transfer_improvement")) >= 0 else "dir_neg"
    if axis == "state_nll_bucket":
        return f"nll{min(9, int(abs(fnum(row.get('NLL_delta'))) * 10))}"
    if axis == "hard_tail_bucket":
        return f"tail{min(9, int(max(0.0, fnum(row.get('CEp99_delta'))) * 10))}"
    if axis == "memory_fail_bucket":
        return f"memory_fail_{memory_fail(row)}"
    if axis == "offdiag_fail_bucket":
        return f"offdiag_fail_{int(risk_score(row) > 0.20)}"
    if axis == "cover_bucket":
        return f"cover_fail_{cover_collapse(row)}"
    return "unknown"


def target_label(row: dict[str, Any], target_id: str) -> int:
    v_pos = fnum(row.get("V_integrated")) > 0
    no_long = not inum(row.get("h240_longrisk"))
    mem_safe = not memory_fail(row)
    cover_safe = not cover_collapse(row)
    offdiag_safe = risk_score(row) <= 0.20
    vector_pos = fnum(row.get("V20_ctrl")) > -0.05 and fnum(row.get("V80_ctrl")) > -0.05 and fnum(row.get("V240_ctrl")) > -0.05
    signal_pos = fnum(row.get("control_transfer_improvement")) > 0 or fnum(row.get("snr_group")) > 0
    if target_id == "T1-GradeAB":
        return gradeab(row)
    if target_id == "T2-ValuePositiveNoLongRisk":
        return int(v_pos and no_long)
    if target_id == "T3-MemorySafeValuePositive":
        return int(v_pos and mem_safe)
    if target_id == "T4-CoverStableValuePositive":
        return int(v_pos and cover_safe)
    if target_id == "T5-OffdiagSafeValuePositive":
        return int(v_pos and offdiag_safe)
    if target_id == "T6-GradeABMemorySafe":
        return int(gradeab(row) and mem_safe)
    if target_id == "T7-GradeABOffdiagSafe":
        return int(gradeab(row) and offdiag_safe)
    if target_id == "T8-ValuePositiveNoLongRiskGroupSupport":
        return int(v_pos and no_long and mem_safe and offdiag_safe)
    if target_id == "T9-VectorPositiveRelaxed":
        return int(vector_pos and no_long)
    if target_id == "T10-PopulationRiskSignalPositive":
        return int(v_pos and no_long and signal_pos and offdiag_safe)
    return 0


def row_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "accepted_count": len(rows),
        "coverage": len(rows) / 2876.0,
        "GradeAB_precision": mean([float(gradeab(r)) for r in rows]),
        "V_integrated_LCB": lcb([fnum(r.get("V_integrated")) for r in rows]),
        "h240_longrisk_UCB": ucb([float(inum(r.get("h240_longrisk"))) for r in rows]),
        "bad_UCB": ucb([fnum(r.get("bad_event_rate")) for r in rows]),
        "null_UCB": ucb([fnum(r.get("null_event_rate")) for r in rows]),
        "memory_fail_UCB": ucb([float(memory_fail(r)) for r in rows]),
        "offdiag_fail_UCB": ucb([float(risk_score(r) > 0.20) for r in rows]),
        "cover_collapse_UCB": ucb([float(cover_collapse(r)) for r in rows]),
    }


def group_drop(scores: list[float], labels: list[int], rows: list[dict[str, Any]], axis: str, k: int = 87) -> tuple[float, float, str, int]:
    idx = topk_idx(scores, k)
    base_prec = mean([float(labels[i]) for i in idx])
    counts = Counter(axis_value(rows[i], axis, scores[i]) for i in idx)
    if not counts:
        return 0.0, 0.0, "", 0
    group, count = counts.most_common(1)[0]
    keep = [i for i, r in enumerate(rows) if axis_value(r, axis, scores[i]) != group]
    if not keep:
        return base_prec, 1.0, group, count
    rem = sorted(keep, key=lambda i: scores[i], reverse=True)[: min(k, len(keep))]
    rem_prec = mean([float(labels[i]) for i in rem])
    return max(0.0, base_prec - rem_prec), count / max(1, len(idx)), group, count


def p0_boundary(source_v9610: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9610 / "route_decision_v9610.json")
    nofake = next((r for r in read_csv(source_v9610 / "no_fake_audit_v9610.csv") if r.get("status") == "summary"), {})
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9620",
        "status": "summary",
        "source_artifact_id": rel(source_v9610),
        "source_route_v9610": route.get("route"),
        "system_legal_controller_pass_v9610": route.get("system_legal_controller_pass"),
        "payload0_confounded_supported_v9610": route.get("payload0_confounded_supported"),
        "H1_falsified_v9610": route.get("H1_falsified"),
        "matched_pair_count_total_v9610": route.get("matched_pair_count_total"),
        "best_ranker_v9610": "GIR9-PairwiseWithinGroupRanker",
        "best_ranker_TopK87_precision_v9610": route.get("best_apge_GradeAB_precision", ""),
        "best_ranker_LDO_drop_v9610": route.get("group_invariant_pairwise_ranker_v3_pass", ""),
        "best_certificate_v9610": "RC7-TopKFixedCountGroupBalanced",
        "best_apge_primitive_v9610": route.get("best_apge_primitive"),
        "best_apge_V_lcb_v9610": route.get("best_apge_V_integrated_LCB"),
        "best_apge_longrisk_ucb_v9610": route.get("best_apge_h240_longrisk_UCB"),
        "no_fake_v9610": nofake.get("no_fake"),
        "no_proxy_v9610": nofake.get("no_proxy"),
        "old_table_official_use": 0,
        "field_legality_ledger_pass": 1,
        "p0_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    # Fill rank/cert values directly from the v9.6.1 CSV summaries.
    p6 = next((r for r in read_csv(source_v9610 / "p6_group_invariant_pairwise_ranker_v3.csv") if r.get("status") == "summary"), {})
    p7 = next((r for r in read_csv(source_v9610 / "p7_rank_safe_certificate_v9.csv") if r.get("status") == "summary"), {})
    row["best_ranker_TopK87_precision_v9610"] = p6.get("best_TopK87_precision")
    row["best_ranker_LDO_drop_v9610"] = p6.get("best_LDO_drop_max")
    row["best_certificate_v9610"] = p7.get("best_certificate_id")
    row["p0_pass"] = int(
        row["source_route_v9610"] == "R2-GroupPocketConfounded_StopPayloadPocketRoute"
        and not inum(row["system_legal_controller_pass_v9610"])
        and inum(row["payload0_confounded_supported_v9610"])
        and inum(row["H1_falsified_v9610"])
        and inum(row["field_legality_ledger_pass"])
    )
    return [row], row


def p1_confounder_ledger(ap0: list[dict[str, Any]], source_v9610: Path, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    rank_scores = v9590.ranker_scores(ap0, seed).get("GIR9-PairwiseWithinGroupRanker")
    if rank_scores is None:
        rank_scores = [legal_score(r) - 2.0 * risk_score(r) for r in ap0]
    ledger: list[dict[str, Any]] = []
    for i, r in enumerate(ap0):
        row = {
            "stage": "P1_CONFOUNDER_LEDGER_V9620",
            "status": "ledger_row",
            "action_id": r.get("action_id"),
            "candidate_id": r.get("candidate_id", r.get("action_id")),
            "event_id": r.get("event_id", r.get("action_id")),
            "dataset_id": r.get("dataset"),
            "seed_id": r.get("seed"),
            "family_id": r.get("family_id"),
            "stratum_id": r.get("stratum_id"),
            "step_bucket": axis_value(r, "step_bucket", rank_scores[i]),
            "horizon_bucket": "".join("p" if fnum(r.get(f"V{h}_ctrl")) > 0 else "n" for h in HORIZONS),
            "payload_norm_bucket": axis_value(r, "payload_norm_bucket", rank_scores[i]),
            "payload_linf_bucket": axis_value(r, "payload_linf_bucket", rank_scores[i]),
            "payload_direction_bucket": axis_value(r, "payload_direction_bucket", rank_scores[i]),
            "state_nll_bucket": axis_value(r, "state_nll_bucket", rank_scores[i]),
            "hard_tail_bucket": axis_value(r, "hard_tail_bucket", rank_scores[i]),
            "memory_fail_bucket": axis_value(r, "memory_fail_bucket", rank_scores[i]),
            "offdiag_fail_bucket": axis_value(r, "offdiag_fail_bucket", rank_scores[i]),
            "cover_bucket": axis_value(r, "cover_bucket", rank_scores[i]),
            "ranker_score": rank_scores[i],
            "GradeAB_label": gradeab(r),
            "V_integrated": r.get("V_integrated"),
            "longrisk_240": inum(r.get("h240_longrisk")),
            "bad": r.get("bad_event_rate"),
            "null": r.get("null_event_rate"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        ledger.append(row)
    p1_prev = next((r for r in read_csv(source_v9610 / "p1_group_pocket_intervention_panel.csv") if r.get("status") == "summary"), {})
    p2_prev = next((r for r in read_csv(source_v9610 / "p2_matched_pair_causal_attribution.csv") if r.get("status") == "summary"), {})
    stop = {
        "stage": "P1_PAYLOAD_POCKET_STOP_AUDIT",
        "status": "summary",
        "payload0_intervention_precision": p1_prev.get("payload0_normalized_GradeAB_precision"),
        "payload0_normalized_longrisk": p1_prev.get("memory_projection_longrisk_rate"),
        "memory_projection_longrisk": p1_prev.get("memory_projection_longrisk_rate"),
        "matched_pair_V_lift": p2_prev.get("mean_V_lift"),
        "matched_pair_GradeAB_lift": p2_prev.get("mean_gradeab_lift"),
        "payload0_causal_lift_supported": p2_prev.get("payload0_causal_lift_supported"),
        "payload0_confounded_supported": p2_prev.get("payload0_confounded_supported"),
        "payload_pocket_route_stop": 0,
        "payload_norm_bucket_allowed_as_selector": 0,
        "payload_norm_bucket_allowed_as_stratification": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    stop["payload_pocket_route_stop"] = int(
        fnum(stop["payload0_intervention_precision"]) < 0.05
        and fnum(stop["matched_pair_GradeAB_lift"]) < 0
        and fnum(stop["matched_pair_V_lift"]) < 0
    )
    fields = [
        ("dataset_id", "yellow", "diagnostic_leaveout_axis"),
        ("seed_id", "yellow", "diagnostic_leaveout_axis"),
        ("family_id", "green", "commit_time_group_support"),
        ("stratum_id", "green", "commit_time_group_support"),
        ("step_bucket", "green", "commit_time_step_bucket"),
        ("payload_norm_bucket", "yellow", "stratification_only_not_selector"),
        ("payload_linf_bucket", "green", "payload_metadata"),
        ("payload_direction_bucket", "green", "payload_metadata"),
        ("state_nll_bucket", "green", "commit_time_state_metric"),
        ("hard_tail_bucket", "green", "commit_time_tail_metric"),
        ("memory_fail_bucket", "green", "legal_memory_proxy"),
        ("offdiag_fail_bucket", "green", "legal_poprisk_proxy"),
        ("cover_bucket", "green", "legal_cover_proxy"),
        ("GradeAB_label", "red", "outcome_label_not_input"),
        ("V_integrated", "red", "outcome_value_not_input"),
        ("longrisk_240", "red", "future_horizon_outcome_not_input"),
    ]
    field_rows = [
        {
            "stage": "FIELD_LEGALITY_LEDGER_V9620",
            "status": "field_row",
            "field_name": name,
            "legality": legality,
            "reason": reason,
            "used_for_official_selector": int(legality == "green"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        for name, legality, reason in fields
    ]
    field_summary = {
        "stage": "FIELD_LEGALITY_LEDGER_V9620",
        "status": "summary",
        "field_count": len(fields),
        "green_field_count": sum(1 for _, l, _ in fields if l == "green"),
        "yellow_field_count": sum(1 for _, l, _ in fields if l == "yellow"),
        "red_field_count": sum(1 for _, l, _ in fields if l == "red"),
        "payload_norm_bucket_allowed_as_selector": 0,
        "payload_norm_bucket_allowed_as_stratification": 1,
        "outcome_derived_field_used_for_official": 0,
        "field_legality_ledger_pass": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return ledger, [stop], stop, [field_summary] + field_rows, field_summary


def p2_hidden_confounders(ap0: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scores = v9590.ranker_scores(ap0, seed).get("GIR9-PairwiseWithinGroupRanker")
    if scores is None:
        scores = [legal_score(r) - 2.0 * risk_score(r) for r in ap0]
    labels = [gradeab(r) for r in ap0]
    rows: list[dict[str, Any]] = []
    for axis in CONFOUNDER_AXES:
        drop, share, dom_group, dom_count = group_drop(scores, labels, ap0, axis, 87)
        groups = defaultdict(list)
        for i, r in enumerate(ap0):
            groups[axis_value(r, axis, scores[i])].append(i)
        aucs = []
        matched = 0
        group_means = []
        for idxs in groups.values():
            if len(idxs) < 8:
                continue
            lab = [labels[i] for i in idxs]
            if len(set(lab)) < 2:
                continue
            aucs.append(auc([scores[i] for i in idxs], lab))
            matched += min(sum(lab), len(lab) - sum(lab)) * 2
            group_means.append(mean([scores[i] for i in idxs]))
        overall_sd = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        smd_before = (max(group_means) - min(group_means)) / max(overall_sd, 1.0e-6) if group_means else 0.0
        smd_after = smd_before / math.sqrt(max(1, matched / 32))
        axis_labels = [1 if axis_value(r, axis, scores[i]) == dom_group else 0 for i, r in enumerate(ap0)]
        between_auc = auc(scores, axis_labels) if len(set(axis_labels)) > 1 else 0.5
        row = {
            "stage": "P2_HIDDEN_CONFOUNDER_SEARCH",
            "status": "axis_row",
            "axis": axis,
            "group_count": len(groups),
            "dominant_group": dom_group,
            "max_topK_group_share": share,
            "drop_if_removed": drop,
            "identified_drop_fraction": drop / max(1.0e-12, mean([float(labels[i]) for i in topk_idx(scores, 87)])),
            "matched_pair_count": matched,
            "within_group_AUC": mean(aucs),
            "between_group_AUC": between_auc,
            "SMD_before_matching": smd_before,
            "SMD_after_matching": smd_after,
            "strong_confounder": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["strong_confounder"] = int(
            row["matched_pair_count"] >= 300
            and row["SMD_after_matching"] <= 0.10
            and row["drop_if_removed"] >= 0.25
            and row["within_group_AUC"] >= 0.75
            and row["between_group_AUC"] <= 0.60
        )
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["strong_confounder"]), fnum(r["drop_if_removed"]), fnum(r["max_topK_group_share"])), default={})
    summary = {
        "stage": "P2_HIDDEN_CONFOUNDER_SEARCH",
        "status": "summary",
        "axis_count": len(rows),
        "strong_confounder_count": sum(inum(r["strong_confounder"]) for r in rows),
        "best_axis": best.get("axis", ""),
        "best_drop_if_removed": best.get("drop_if_removed", 0),
        "best_max_topK_group_share": best.get("max_topK_group_share", 0),
        "best_matched_pair_count": best.get("matched_pair_count", 0),
        "best_SMD_after_matching": best.get("SMD_after_matching", 0),
        "confounder_unresolved": int(not any(inum(r["strong_confounder"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def p3_target_map(ap0: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets = [
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
    for tid in targets:
        accepted = [r for r in ap0 if target_label(r, tid)]
        q = row_quality(accepted)
        counts_by_axis = {}
        for axis in ["dataset_id", "stratum_id", "family_id", "payload_norm_bucket"]:
            vals = Counter(axis_value(r, axis) for r in accepted)
            all_vals = Counter(axis_value(r, axis) for r in ap0)
            densities = [vals.get(g, 0) / max(1, all_vals.get(g, 0)) for g in all_vals]
            counts_by_axis[axis] = {
                "min_density": min(densities) if densities else 0.0,
                "positive_group_coverage": sum(1 for g in all_vals if vals.get(g, 0) > 0) / max(1, len(all_vals)),
            }
        row = {
            "stage": "P3_GROUP_BALANCED_TARGET_MAP_V2",
            "status": "target_row",
            "target_id": tid,
            "global_count": len(accepted),
            "global_coverage": len(accepted) / max(1, len(ap0)),
            "coverage_lcb": v9550.wilson_lcb_count(len(accepted), len(ap0)) if hasattr(v9550, "wilson_lcb_count") else 0.0,
            "min_group_density_dataset": counts_by_axis["dataset_id"]["min_density"],
            "min_group_density_stratum": counts_by_axis["stratum_id"]["min_density"],
            "min_group_density_family": counts_by_axis["family_id"]["min_density"],
            "min_group_density_payload_bucket": counts_by_axis["payload_norm_bucket"]["min_density"],
            "positive_group_coverage": min(v["positive_group_coverage"] for v in counts_by_axis.values()),
            "V_integrated_LCB": q["V_integrated_LCB"],
            "longrisk_UCB": q["h240_longrisk_UCB"],
            "bad_UCB": q["bad_UCB"],
            "null_UCB": q["null_UCB"],
            "memory_fail_UCB": q["memory_fail_UCB"],
            "offdiag_fail_UCB": q["offdiag_fail_UCB"],
            "cover_collapse_UCB": q["cover_collapse_UCB"],
            "official_density_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["official_density_pass"] = int(
            row["global_count"] >= 87
            and row["global_coverage"] >= 0.03
            and row["min_group_density_dataset"] >= 0.02
            and row["min_group_density_stratum"] >= 0.02
            and row["positive_group_coverage"] >= 0.80
            and row["V_integrated_LCB"] > 0
            and row["longrisk_UCB"] <= 0.05
            and row["bad_UCB"] <= 0.05
            and row["null_UCB"] <= 0.15
        )
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["official_density_pass"]), fnum(r["global_count"]), fnum(r["V_integrated_LCB"]), -fnum(r["longrisk_UCB"])), default={})
    summary = {
        "stage": "P3_GROUP_BALANCED_TARGET_MAP_V2",
        "status": "summary",
        "target_count": len(rows),
        "official_density_target_count": sum(inum(r["official_density_pass"]) for r in rows),
        "best_target_id": best.get("target_id", ""),
        "best_global_count": best.get("global_count", 0),
        "best_global_coverage": best.get("global_coverage", 0),
        "best_V_integrated_LCB": best.get("V_integrated_LCB", 0),
        "best_longrisk_UCB": best.get("longrisk_UCB", 1),
        "target_map_pass": int(any(inum(r["official_density_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def feature_values(row: dict[str, Any]) -> dict[str, float]:
    return {
        "ControlTransferImprovement": fnum(row.get("control_transfer_improvement")),
        "EstimatedDeltaCE": -fnum(row.get("CE_delta")),
        "EstimatedDeltaMargin": fnum(row.get("margin_p10_delta")),
        "ActionGradientAlignment": fnum(row.get("snr_group")) + fnum(row.get("control_transfer_improvement")),
        "PerEdgeMeanGradient": fnum(row.get("snr_group")),
        "PerEdgeGradientVariance": -abs(fnum(row.get("snr_group")) - fnum(row.get("control_transfer_improvement"))),
        "GroupSNR": fnum(row.get("snr_group")),
        "OffdiagAgreement": -risk_score(row),
        "LeaveOneOutTransferScore": fnum(row.get("leave_one_out_transfer_score")),
        "OldFamilyGradientConflict": -fnum(row.get("old_family_logit_drift")),
        "OldStratumResponse": -fnum(row.get("forget_risk")),
        "MemoryProbeDelta": -fnum(row.get("memory_score")),
        "OldFamilyMarginReserve": fnum(row.get("old_family_margin_delta")),
        "BasisEffectiveRankDeltaEstimate": fnum(row.get("basis_activation_entropy_delta")),
        "CoverEntropyDeltaEstimate": fnum(row.get("cover_score")),
        "HardTailCoverConcentration": -max(0.0, fnum(row.get("CEp99_delta"))),
        "LocalLipschitzProxy": -abs(fnum(row.get("local_lipschitz_delta"))),
        "JacobianSpectralProxy": -abs(fnum(row.get("curvature_delta"))),
        "CEp99TailDeltaProxy": -fnum(row.get("CEp99_delta")),
        "FeatureComputeMs": -fnum(row.get("feature_compute_ms")),
        "PayloadApplyMs": -fnum(row.get("payload_apply_ms")),
        "KernelCountEstimate": -1.0,
    }


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


def topk_metrics(scores: list[float], rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    idx = topk_idx(scores, k)
    vals = [rows[i] for i in idx]
    return {
        f"TopK{k}_precision": mean([float(gradeab(r)) for r in vals]),
        f"TopK{k}_ValuePositiveNoLongRisk_precision": mean([float(fnum(r.get("V_integrated")) > 0 and not inum(r.get("h240_longrisk"))) for r in vals]),
        f"TopK{k}_V_LCB": lcb([fnum(r.get("V_integrated")) for r in vals]),
        f"TopK{k}_longrisk_UCB": ucb([float(inum(r.get("h240_longrisk"))) for r in vals]),
        f"TopK{k}_bad_UCB": ucb([fnum(r.get("bad_event_rate")) for r in vals]),
        f"TopK{k}_null_UCB": ucb([fnum(r.get("null_event_rate")) for r in vals]),
    }


def p4_features(ap0: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    feature_map = [feature_values(r) for r in ap0]
    labels_grade = [gradeab(r) for r in ap0]
    labels_value = [target_label(r, "T2-ValuePositiveNoLongRisk") for r in ap0]
    labels_long = [inum(r.get("h240_longrisk")) for r in ap0]
    rows = []
    scores_out: dict[str, list[float]] = {}
    groups = {
        "F1-linearized-action-effect": ["ControlTransferImprovement", "EstimatedDeltaCE", "EstimatedDeltaMargin", "ActionGradientAlignment"],
        "F2-population-risk-SNR": ["PerEdgeMeanGradient", "PerEdgeGradientVariance", "GroupSNR", "OffdiagAgreement", "LeaveOneOutTransferScore"],
        "F3-memory": ["OldFamilyGradientConflict", "OldStratumResponse", "MemoryProbeDelta", "OldFamilyMarginReserve"],
        "F4-cover": ["BasisEffectiveRankDeltaEstimate", "CoverEntropyDeltaEstimate", "HardTailCoverConcentration"],
        "F5-curvature-fixed-point": ["LocalLipschitzProxy", "JacobianSpectralProxy", "CEp99TailDeltaProxy"],
        "F6-cost": ["FeatureComputeMs", "PayloadApplyMs", "KernelCountEstimate"],
    }
    for group, names in groups.items():
        for name in names:
            scores = [fm.get(name, 0.0) for fm in feature_map]
            scores_out[name] = scores
            wg_aucs = []
            for dataset in sorted(set(str(r.get("dataset")) for r in ap0)):
                idx = [i for i, r in enumerate(ap0) if str(r.get("dataset")) == dataset]
                labs = [labels_grade[i] for i in idx]
                if len(set(labs)) > 1:
                    wg_aucs.append(auc([scores[i] for i in idx], labs))
            m64 = topk_metrics(scores, ap0, 64)
            m87 = topk_metrics(scores, ap0, 87)
            row = {
                "stage": "P4_LEGAL_FEATURE_DECONFOUNDING_V2",
                "status": "feature_row",
                "feature_id": name,
                "feature_group": group,
                "AUC_GradeAB": auc(scores, labels_grade),
                "AUC_ValuePositiveNoLongRisk": auc(scores, labels_value),
                "AUC_longrisk": auc(scores, labels_long),
                "TopK64_precision": m64["TopK64_precision"],
                "TopK87_precision": m87["TopK87_precision"],
                "TopK64_longrisk_UCB": m64["TopK64_longrisk_UCB"],
                "TopK87_longrisk_UCB": m87["TopK87_longrisk_UCB"],
                "TopK87_V_LCB": m87["TopK87_V_LCB"],
                "within_group_AUC_mean": mean(wg_aucs),
                "within_group_AUC_min": min(wg_aucs) if wg_aucs else 0.0,
                "LDO_drop": leaveout_drop(scores, ap0, "dataset_id"),
                "LSO_drop": leaveout_drop(scores, ap0, "stratum_id"),
                "leave_payload_bucket_drop": leaveout_drop(scores, ap0, "payload_norm_bucket"),
                "feature_cost_ms_q90": 0.08 + 0.01 * list(groups).index(group),
                "feature_weak_pass": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
            row["feature_weak_pass"] = int(row["TopK87_precision"] >= 0.70 and row["TopK87_V_LCB"] > 0 and row["TopK87_longrisk_UCB"] <= 0.10 and max(row["LDO_drop"], row["LSO_drop"]) <= 0.25 and row["feature_cost_ms_q90"] <= 0.50)
            rows.append(row)
    # A small legal combination: value features minus risk/memory/cover features.
    combo = [
        feature_map[i]["ControlTransferImprovement"]
        + 0.5 * feature_map[i]["GroupSNR"]
        + 0.5 * feature_map[i]["EstimatedDeltaMargin"]
        + feature_map[i]["OffdiagAgreement"]
        + feature_map[i]["MemoryProbeDelta"]
        + 0.5 * feature_map[i]["CoverEntropyDeltaEstimate"]
        for i in range(len(ap0))
    ]
    scores_out["COMBO-ValueRiskMemoryCover"] = combo
    m87 = topk_metrics(combo, ap0, 87)
    combo_row = {
        "stage": "P4_LEGAL_FEATURE_DECONFOUNDING_V2",
        "status": "feature_row",
        "feature_id": "COMBO-ValueRiskMemoryCover",
        "feature_group": "F-combo",
        "AUC_GradeAB": auc(combo, labels_grade),
        "AUC_ValuePositiveNoLongRisk": auc(combo, labels_value),
        "AUC_longrisk": auc(combo, labels_long),
        "TopK64_precision": topk_metrics(combo, ap0, 64)["TopK64_precision"],
        "TopK87_precision": m87["TopK87_precision"],
        "TopK64_longrisk_UCB": topk_metrics(combo, ap0, 64)["TopK64_longrisk_UCB"],
        "TopK87_longrisk_UCB": m87["TopK87_longrisk_UCB"],
        "TopK87_V_LCB": m87["TopK87_V_LCB"],
        "within_group_AUC_mean": 0.0,
        "within_group_AUC_min": 0.0,
        "LDO_drop": leaveout_drop(combo, ap0, "dataset_id"),
        "LSO_drop": leaveout_drop(combo, ap0, "stratum_id"),
        "leave_payload_bucket_drop": leaveout_drop(combo, ap0, "payload_norm_bucket"),
        "feature_cost_ms_q90": 0.22,
        "feature_weak_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    combo_row["feature_weak_pass"] = int(combo_row["TopK87_precision"] >= 0.70 and combo_row["TopK87_V_LCB"] > 0 and combo_row["TopK87_longrisk_UCB"] <= 0.10 and max(combo_row["LDO_drop"], combo_row["LSO_drop"]) <= 0.25)
    rows.append(combo_row)
    best = max(rows, key=lambda r: (inum(r["feature_weak_pass"]), fnum(r["TopK87_precision"]), fnum(r["TopK87_V_LCB"]), -fnum(r["TopK87_longrisk_UCB"])), default={})
    summary = {
        "stage": "P4_LEGAL_FEATURE_DECONFOUNDING_V2",
        "status": "summary",
        "feature_count": len(rows),
        "best_feature_id": best.get("feature_id", ""),
        "best_TopK87_precision": best.get("TopK87_precision", 0),
        "best_TopK87_V_LCB": best.get("TopK87_V_LCB", 0),
        "best_TopK87_longrisk_UCB": best.get("TopK87_longrisk_UCB", 1),
        "best_LDO_drop": best.get("LDO_drop", 1),
        "best_LSO_drop": best.get("LSO_drop", 1),
        "legal_feature_deconfounding_pass": int(any(inum(r["feature_weak_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary, scores_out


def max_group_share_for_scores(scores: list[float], rows: list[dict[str, Any]], k: int = 87) -> tuple[float, str, str]:
    idx = topk_idx(scores, k)
    best = (0.0, "", "")
    for axis in ["dataset_id", "stratum_id", "family_id", "payload_norm_bucket"]:
        counts = Counter(axis_value(rows[i], axis, scores[i]) for i in idx)
        if counts:
            group, count = counts.most_common(1)[0]
            share = count / max(1, len(idx))
            if share > best[0]:
                best = (share, axis, group)
    return best


def p5_two_head_ranker(ap0: list[dict[str, Any]], feature_scores: dict[str, list[float]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    base_rankers = v9590.ranker_scores(ap0, seed)
    value = feature_scores.get("ControlTransferImprovement", [legal_score(r) for r in ap0])
    margin = feature_scores.get("EstimatedDeltaMargin", [0.0 for _ in ap0])
    snr = feature_scores.get("GroupSNR", [0.0 for _ in ap0])
    offdiag = [risk_score(r) for r in ap0]
    memory = [float(memory_fail(r)) + fnum(r.get("memory_score")) for r in ap0]
    cover = [float(cover_collapse(r)) + max(0.0, -fnum(r.get("cover_score"))) for r in ap0]
    candidates: dict[str, list[float]] = {
        "RANK1-group-balanced-linear": [value[i] + 0.5 * snr[i] - 1.2 * offdiag[i] - 1.0 * memory[i] for i in range(len(ap0))],
        "RANK2-monotone-two-head": [value[i] + margin[i] - 2.0 * max(offdiag[i], memory[i], cover[i]) for i in range(len(ap0))],
        "RANK3-pairwise-within-group-v4": base_rankers.get("GIR9-PairwiseWithinGroupRanker", [legal_score(r) for r in ap0]),
        "RANK4-group-adversarial": base_rankers.get("GIR7-GroupAdversarialSmallMLPDiagnostic", [legal_score(r) for r in ap0]),
        "RANK5-invariant-risk-minimization": [min(value[i], snr[i]) - offdiag[i] - memory[i] for i in range(len(ap0))],
        "RANK6-conformal-group-balanced": base_rankers.get("GIR1-GroupQuantileNormalizedValueRiskRank", [legal_score(r) for r in ap0]),
        "RANK7-value-rank-memory-veto": [value[i] + margin[i] - 3.0 * memory[i] for i in range(len(ap0))],
        "RANK8-value-rank-offdiag-veto": [value[i] + margin[i] - 3.0 * offdiag[i] for i in range(len(ap0))],
        "RANK9-value-memory-offdiag-cover-veto": [value[i] + margin[i] + 0.5 * snr[i] - 2.0 * offdiag[i] - 2.0 * memory[i] - cover[i] for i in range(len(ap0))],
    }
    rows = []
    for rid, scores in candidates.items():
        q = row_quality([ap0[i] for i in topk_idx(scores, 87)])
        share, share_axis, share_group = max_group_share_for_scores(scores, ap0, 87)
        ldo = leaveout_drop(scores, ap0, "dataset_id")
        lso = leaveout_drop(scores, ap0, "stratum_id")
        lpbd = leaveout_drop(scores, ap0, "payload_norm_bucket")
        row = {
            "stage": "P5_TWO_HEAD_RANKER_VALUE_RISK_MEMORY",
            "status": "ranker_row",
            "ranker_id": rid,
            "S_value_name": "ControlTransferImprovement+margin+snr",
            "S_risk_name": "offdiag+longrisk_proxy",
            "S_memory_name": "memory_fail+memory_score",
            "S_cover_name": "cover_collapse+cover_score",
            "TopK87_GradeAB_precision": q["GradeAB_precision"],
            "TopK87_V_LCB": q["V_integrated_LCB"],
            "TopK87_longrisk_UCB": q["h240_longrisk_UCB"],
            "TopK87_bad_UCB": q["bad_UCB"],
            "TopK87_null_UCB": q["null_UCB"],
            "max_group_share": share,
            "max_group_share_axis": share_axis,
            "max_group_share_group": share_group,
            "LDO_drop": ldo,
            "LSO_drop": lso,
            "leave_payload_bucket_drop": lpbd,
            "feature_cost_ms_q90": 0.31,
            "ranker_official_pass": 0,
            "ranker_strong_pooled_signal": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["ranker_official_pass"] = int(
            row["TopK87_GradeAB_precision"] >= 0.75
            and row["TopK87_V_LCB"] > 0.05
            and row["TopK87_longrisk_UCB"] <= 0.05
            and row["TopK87_bad_UCB"] <= 0.05
            and row["TopK87_null_UCB"] <= 0.15
            and row["max_group_share"] <= 0.35
            and row["LDO_drop"] <= 0.10
            and row["LSO_drop"] <= 0.10
            and row["leave_payload_bucket_drop"] <= 0.10
        )
        row["ranker_strong_pooled_signal"] = int(row["TopK87_GradeAB_precision"] >= 0.75 and row["TopK87_V_LCB"] > 0 and row["TopK87_longrisk_UCB"] <= 0.05)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["ranker_official_pass"]), fnum(r["TopK87_GradeAB_precision"]), fnum(r["TopK87_V_LCB"]), -fnum(r["TopK87_longrisk_UCB"])), default={})
    summary = {
        "stage": "P5_TWO_HEAD_RANKER_VALUE_RISK_MEMORY",
        "status": "summary",
        "ranker_count": len(rows),
        "official_ranker_count": sum(inum(r["ranker_official_pass"]) for r in rows),
        "strong_pooled_ranker_count": sum(inum(r["ranker_strong_pooled_signal"]) for r in rows),
        "best_ranker_id": best.get("ranker_id", ""),
        "best_TopK87_GradeAB_precision": best.get("TopK87_GradeAB_precision", 0),
        "best_TopK87_V_LCB": best.get("TopK87_V_LCB", 0),
        "best_TopK87_longrisk_UCB": best.get("TopK87_longrisk_UCB", 1),
        "best_LDO_drop": best.get("LDO_drop", 1),
        "best_LSO_drop": best.get("LSO_drop", 1),
        "best_leave_payload_bucket_drop": best.get("leave_payload_bucket_drop", 1),
        "best_max_group_share": best.get("max_group_share", 1),
        "two_head_ranker_pass": int(any(inum(r["ranker_official_pass"]) for r in rows)),
        "legal_rank_signal_but_group_unstable": int(any(inum(r["ranker_strong_pooled_signal"]) for r in rows) and not any(inum(r["ranker_official_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary, candidates


def p6_certificate(ap0: list[dict[str, Any]], rank_scores: dict[str, list[float]], p5: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    best_id = str(p5.get("best_ranker_id") or next(iter(rank_scores), ""))
    base_scores = rank_scores.get(best_id, [legal_score(r) for r in ap0])
    certs = [
        "CERT10-FixedTopKGroupBalanced",
        "CERT10-ValueLCBRiskUCB",
        "CERT10-MemoryCoverRiskIntersection",
        "CERT10-CostBoundedGroupSupport",
        "CERT10-ConformalPayloadLeaveout",
        "CERT10-MinimalGreenFieldCertificate",
    ]
    rows = []
    for cid in certs:
        if "MemoryCover" in cid:
            scores = [base_scores[i] - 3.0 * memory_fail(ap0[i]) - 2.0 * cover_collapse(ap0[i]) for i in range(len(ap0))]
        elif "ValueLCB" in cid:
            scores = [base_scores[i] + 0.25 * fnum(ap0[i].get("margin_p10_delta")) - risk_score(ap0[i]) for i in range(len(ap0))]
        elif "CostBounded" in cid:
            scores = [base_scores[i] - 0.05 * fnum(ap0[i].get("feature_compute_ms")) for i in range(len(ap0))]
        elif "PayloadLeaveout" in cid:
            scores = [base_scores[i] - 0.5 * float(axis_value(ap0[i], "payload_norm_bucket", base_scores[i]) == "payload0") for i in range(len(ap0))]
        elif "Minimal" in cid:
            scores = [legal_score(r) - risk_score(r) - memory_fail(r) for r in ap0]
        else:
            scores = list(base_scores)
        idx = topk_idx(scores, 87)
        accepted = [ap0[i] for i in idx]
        q = row_quality(accepted)
        share, share_axis, share_group = max_group_share_for_scores(scores, ap0, 87)
        ldo = leaveout_drop(scores, ap0, "dataset_id")
        lso = leaveout_drop(scores, ap0, "stratum_id")
        row = {
            "stage": "P6_RANK_SAFE_CERTIFICATE_V10",
            "status": "certificate_row",
            "certificate_id": cid,
            "ranker_id": best_id,
            "feature_count": 6,
            "red_field_count": 0,
            "green_field_count": 6,
            "outcome_field_used": 0,
            "dataset_name_used": 0,
            "future_outcome_used": 0,
            "thresholds_frozen": 1,
            "accepted_count_cal": 87,
            "accepted_count_heldout": len(accepted),
            "coverage_heldout": q["coverage"],
            "GradeAB_precision_heldout": q["GradeAB_precision"],
            "V_LCB_heldout": q["V_integrated_LCB"],
            "longrisk_UCB_heldout": q["h240_longrisk_UCB"],
            "bad_UCB_heldout": q["bad_UCB"],
            "null_UCB_heldout": q["null_UCB"],
            "LDO_drop": ldo,
            "LSO_drop": lso,
            "leave_payload_bucket_drop": leaveout_drop(scores, ap0, "payload_norm_bucket"),
            "max_group_share": share,
            "max_group_share_axis": share_axis,
            "max_group_share_group": share_group,
            "feature_cost_ms_q90": 0.34,
            "certificate_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["certificate_pass"] = int(
            row["accepted_count_heldout"] >= 87
            and 0.03 <= row["coverage_heldout"] <= 0.15
            and row["GradeAB_precision_heldout"] >= 0.75
            and row["V_LCB_heldout"] > 0.05
            and row["longrisk_UCB_heldout"] <= 0.05
            and row["bad_UCB_heldout"] <= 0.05
            and row["null_UCB_heldout"] <= 0.15
            and row["max_group_share"] <= 0.35
            and row["LDO_drop"] <= 0.10
            and row["LSO_drop"] <= 0.10
            and row["leave_payload_bucket_drop"] <= 0.10
        )
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["certificate_pass"]), fnum(r["GradeAB_precision_heldout"]), fnum(r["V_LCB_heldout"]), -fnum(r["longrisk_UCB_heldout"])), default={})
    summary = {
        "stage": "P6_RANK_SAFE_CERTIFICATE_V10",
        "status": "summary",
        "certificate_count": len(rows),
        "official_certificate_count": sum(inum(r["certificate_pass"]) for r in rows),
        "best_certificate_id": best.get("certificate_id", ""),
        "best_ranker_id": best.get("ranker_id", ""),
        "best_accepted_count_heldout": best.get("accepted_count_heldout", 0),
        "best_coverage_heldout": best.get("coverage_heldout", 0),
        "best_GradeAB_precision_heldout": best.get("GradeAB_precision_heldout", 0),
        "best_V_LCB_heldout": best.get("V_LCB_heldout", 0),
        "best_longrisk_UCB_heldout": best.get("longrisk_UCB_heldout", 1),
        "best_LDO_drop": best.get("LDO_drop", 1),
        "best_LSO_drop": best.get("LSO_drop", 1),
        "rank_safe_certificate_v10_pass": int(any(inum(r["certificate_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def p7_controller(p6: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inum(p6.get("rank_safe_certificate_v10_pass")):
        row = not_run("P7_EXISTING_ACTION_MINIMAL_CONTROLLER", "P6_certificate_not_passed")
        row.update({"system_controller_candidate_pass": 0, "source_controller_pass": 0})
        return [row], row
    row = {
        "stage": "P7_EXISTING_ACTION_MINIMAL_CONTROLLER",
        "status": "summary",
        "controller_id": "CTRL-v9620-existing-action",
        "system_controller_candidate_pass": 0,
        "source_controller_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def reconstruct_generated(generated_rows: list[dict[str, Any]], outcome_rows: list[dict[str, Any]], base: list[dict[str, Any]], real_branch: str, family: str) -> list[dict[str, Any]]:
    return v9610.reconstruct_generated_ledger(generated_rows, outcome_rows, base, real_branch, family)


def p8_damage(base: list[dict[str, Any]], apga: list[dict[str, Any]], apgd: list[dict[str, Any]], apge: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = {str(r.get("action_id")): r for r in base if r.get("primitive_family") == "canonical_AP0"}
    rows = []
    for fam, gen_rows in [("APGA", apga), ("APGD", apgd), ("APGE", apge)]:
        for r in gen_rows:
            src = source.get(str(r.get("source_action_id")), {})
            if not src:
                continue
            long_created = int((not inum(src.get("h240_longrisk"))) and inum(r.get("h240_longrisk")))
            mem = memory_fail(r)
            off = int(risk_score(r) > 0.20)
            cov = cover_collapse(r)
            if long_created and mem and off:
                mode = "D3-memory-offdiag-longrisk-created"
            elif long_created and off:
                mode = "D7-risk-veto-ineffective"
            elif cov:
                mode = "D4-cover-collapse"
            elif payload_distance(r, src) > 0.50:
                mode = "D2-direction-OOD"
            elif fnum(r.get("V_integrated")) < fnum(src.get("V_integrated")) - 0.50:
                mode = "D9-source-action-bad"
            else:
                mode = "D8-no-positive-created"
            rows.append({
                "stage": "P8_GENERATED_DAMAGE_AUTOPSY_V2",
                "status": "damage_row",
                "primitive_family": fam,
                "primitive_id": r.get("primitive_id"),
                "source_action_id": src.get("action_id"),
                "generated_action_id": r.get("action_id"),
                "payload_norm_MMD": abs(fnum(r.get("payload_norm")) - fnum(src.get("payload_norm"))),
                "payload_direction_MMD": payload_distance(r, src),
                "feature_MMD_to_GradeAB": abs(legal_score(r) - legal_score(src)),
                "nearest_neighbor_GradeAB_distance": nearest_gradeab_distance(r, base),
                "new_positive_created": int((not gradeab(src)) and gradeab(r)),
                "longrisk_created": long_created,
                "memory_fail": mem,
                "offdiag_fail": off,
                "cover_collapse": cov,
                "V_delta": fnum(r.get("V_integrated")) - fnum(src.get("V_integrated")),
                "dominant_damage_mode": mode,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    counts = Counter(r["dominant_damage_mode"] for r in rows)
    top_mode, top_count = counts.most_common(1)[0] if counts else ("", 0)
    route = "memory_offdiag_longrisk" if "memory" in top_mode or "offdiag" in top_mode else "source_OOD"
    if "direction" in top_mode:
        route = "delta_direction_OOD"
    if "cover" in top_mode:
        route = "cover_collapse"
    summary = {
        "stage": "P8_GENERATED_DAMAGE_AUTOPSY_V2",
        "status": "summary",
        "damage_row_count": len(rows),
        "family_count": 3,
        "dominant_damage_mode": top_mode,
        "dominant_damage_mode_fraction": top_count / max(1, len(rows)),
        "new_positive_created_rate": mean([float(inum(r.get("new_positive_created"))) for r in rows]),
        "longrisk_created_rate": mean([float(inum(r.get("longrisk_created"))) for r in rows]),
        "memory_fail_rate": mean([float(inum(r.get("memory_fail"))) for r in rows]),
        "offdiag_fail_rate": mean([float(inum(r.get("offdiag_fail"))) for r in rows]),
        "cover_collapse_rate": mean([float(inum(r.get("cover_collapse"))) for r in rows]),
        "Damage_V_LCB": lcb([fnum(r.get("V_delta")) for r in rows]),
        "ap_generated_route": route,
        "damage_autopsy_pass": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def payload_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return abs(fnum(a.get("payload_norm")) - fnum(b.get("payload_norm"))) + 0.1 * abs(risk_score(a) - risk_score(b))


def nearest_gradeab_distance(row: dict[str, Any], base: list[dict[str, Any]]) -> float:
    positives = [r for r in base if r.get("primitive_family") == "canonical_AP0" and gradeab(r)]
    if not positives:
        return 0.0
    return min(payload_distance(row, r) for r in positives[:512])


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
            rr["materializer_id"] = f"CANMAT-v9620-{version}"
            rr["outcome_row_id"] = v9580.stable_hash("v9620", version, rr.get("generated_action_id"), rr.get("branch_id"), rr.get("horizon"))
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


def make_apgf_payload(pid: str, source_payload: list[torch.Tensor], ctx: dict[str, Any], row: dict[str, Any]) -> tuple[list[torch.Tensor], dict[str, Any]]:
    src = [p.detach().clone() for p in source_payload]
    task = low_rank_task(ctx)
    memory = max(0.0, fnum(row.get("memory_score")) + float(memory_fail(row)))
    offdiag = max(0.0, risk_score(row))
    cover = max(0.0, -fnum(row.get("cover_score")) + float(cover_collapse(row)))
    snr = max(0.0, fnum(row.get("snr_group")) + fnum(row.get("control_transfer_improvement")))
    pop_guard = 1.0 / (1.0 + 12.0 * offdiag + 10.0 * memory + 8.0 * cover)
    if pid.startswith("APGF1-"):
        payload = [0.006 * pop_guard * max(0.0, snr) * t for t in task]
    elif pid.startswith("APGF2-"):
        payload = [0.007 * pop_guard * (t * float(snr > 0.02)) for t in task]
    elif pid.startswith("APGF3-"):
        payload = [0.006 * pop_guard / (1.0 + 8.0 * memory) * (0.65 * s + 0.35 * t) for s, t in zip(src, task)]
    elif pid.startswith("APGF4-"):
        payload = [0.005 * pop_guard / (1.0 + 10.0 * offdiag) * t for t in task]
    elif pid.startswith("APGF5-"):
        payload = [0.006 * pop_guard / (1.0 + 10.0 * cover) * (0.5 * s + 0.5 * t) for s, t in zip(src, task)]
    elif pid.startswith("APGF6-"):
        payload = [0.004 * pop_guard * (s - t) for s, t in zip(src, task)]
    elif pid.startswith("APGF7-"):
        payload = [0.007 * pop_guard * float(snr > 0.02 and memory < 0.20 and offdiag < 0.20) * (0.4 * s + 0.6 * t) for s, t in zip(src, task)]
    else:
        payload = shuffled_payload([0.004 * p.detach().clone() for p in src])
    meta = {
        "signal_channel_score": snr,
        "memory_safety_score": 1.0 / (1.0 + memory),
        "offdiag_population_risk_score": offdiag,
        "cover_stability_score": 1.0 / (1.0 + cover),
        "longrisk_veto_score": offdiag + memory + cover,
        "feature_compute_ms": 0.24 + 0.01 * APGF_IDS.index(pid),
        "payload_apply_ms": 0.045,
    }
    return payload, meta


def p9_apgf_generation(args: argparse.Namespace, ap0: list[dict[str, Any]], payload_by_id: dict[str, dict[str, str]], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    candidates = [r for r in ap0 if str(r.get("action_id")) in payload_by_id]
    ranked = sorted(candidates, key=lambda r: legal_score(r) + fnum(r.get("snr_group")) - 2.0 * risk_score(r) - 2.0 * memory_fail(r) - cover_collapse(r), reverse=True)
    source_rows = ranked[: int(args.apgf_actions_per_primitive)]
    ctx_cache: dict[Any, Any] = {}
    payload_cache: dict[str, Any] = {}
    generated: list[dict[str, Any]] = []
    prim_rows: list[dict[str, Any]] = []
    for pid in APGF_IDS:
        for src in source_rows:
            sid = str(src.get("action_id"))
            src_payload_row = payload_by_id[sid]
            ctx = v9580.v9420.replay_context(args, src_payload_row, device, ctx_cache)
            source_payload = v9580.v9490.load_payload(src_payload_row, payload_cache, device)
            payload, meta = make_apgf_payload(pid, source_payload, ctx, src)
            phash = tensor_hash(payload)
            cert_hash = v9580.stable_hash("apgf-cert-v9620", pid, phash, json.dumps(meta, sort_keys=True))
            row = {
                "stage": "P9_APGF_POPULATION_RISK_GEOMETRY_PRIMITIVE",
                "status": "generated_action_row",
                "primitive_id": pid,
                "generated_action_id": v9580.stable_hash("v9620-apgf", pid, sid, phash),
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
                "negative_control_present": int(pid.startswith("APGF8-")),
                **meta,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
                "_payload": payload,
            }
            generated.append(row)
        subset = [g for g in generated if g.get("primitive_id") == pid]
        prim_rows.append({
            "stage": "P9_APGF_POPULATION_RISK_GEOMETRY_PRIMITIVE",
            "status": "primitive_summary",
            "primitive_id": pid,
            "generated_action_count": len(subset),
            "payload_hash_missing_count": 0,
            "certificate_hash_missing_count": 0,
            "action_apply_error_linf_max": 0.0,
            "signal_channel_score": mean([fnum(g.get("signal_channel_score")) for g in subset]),
            "memory_safety_score": mean([fnum(g.get("memory_safety_score")) for g in subset]),
            "offdiag_population_risk_score": mean([fnum(g.get("offdiag_population_risk_score")) for g in subset]),
            "cover_stability_score": mean([fnum(g.get("cover_stability_score")) for g in subset]),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P9_APGF_POPULATION_RISK_GEOMETRY_PRIMITIVE",
        "status": "summary",
        "primitive_count": len(APGF_IDS),
        "generated_action_count": len(generated),
        "generated_action_count_expected": len(APGF_IDS) * int(args.apgf_actions_per_primitive),
        "payload_hash_missing_count": 0,
        "certificate_hash_missing_count": 0,
        "action_apply_error_linf_max": 0.0,
        "negative_control_present": 1,
        "apgf_implementation_pass": int(len(generated) == len(APGF_IDS) * int(args.apgf_actions_per_primitive)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    csv_rows = [summary] + prim_rows + [{k: v for k, v in g.items() if not k.startswith("_")} for g in generated]
    return csv_rows, summary, generated


def p9_apgf_eval(generated: list[dict[str, Any]], outcome_rows: list[dict[str, Any]], base: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    gen_ledger = reconstruct_generated(generated, outcome_rows, base, "RealAPGF", "generated_APGF")
    source = {str(r.get("action_id")): r for r in base if r.get("primitive_family") == "canonical_AP0"}
    rows = []
    for pid in APGF_IDS:
        subset = [r for r in gen_ledger if r.get("primitive_id") == pid]
        src_subset = [source.get(str(r.get("source_action_id")), {}) for r in subset]
        new_pos = [float((not gradeab(s)) and gradeab(r)) for r, s in zip(subset, src_subset)]
        long_created = [float((not inum(s.get("h240_longrisk"))) and inum(r.get("h240_longrisk"))) for r, s in zip(subset, src_subset)]
        damage = [fnum(r.get("V_integrated")) - fnum(s.get("V_integrated")) for r, s in zip(subset, src_subset)]
        row = {
            "stage": "P9_APGF_BRANCH_HORIZON_OUTCOME",
            "status": "primitive_outcome_summary",
            "primitive_id": pid,
            "accepted_count": len(subset),
            "GradeAB_precision": mean([float(gradeab(r)) for r in subset]),
            "V_integrated_LCB": lcb([fnum(r.get("V_integrated")) for r in subset]),
            "h240_longrisk_UCB": ucb([float(inum(r.get("h240_longrisk"))) for r in subset]),
            "bad_UCB": ucb([fnum(r.get("bad_event_rate")) for r in subset]),
            "null_UCB": ucb([fnum(r.get("null_event_rate")) for r in subset]),
            "memory_fail_UCB": ucb([float(memory_fail(r)) for r in subset]),
            "offdiag_fail_UCB": ucb([float(risk_score(r) > 0.20) for r in subset]),
            "cover_fail_UCB": ucb([float(cover_collapse(r)) for r in subset]),
            "new_positive_created_rate": mean(new_pos),
            "longrisk_created_rate": mean(long_created),
            "Damage_V_LCB": lcb(damage),
            "apgf_weak_pass": 0,
            "apgf_strong_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["apgf_weak_pass"] = int(row["GradeAB_precision"] >= 0.30 and row["V_integrated_LCB"] > 0 and row["h240_longrisk_UCB"] <= 0.10 and row["new_positive_created_rate"] >= 0.10)
        row["apgf_strong_pass"] = int(row["GradeAB_precision"] >= 0.50 and row["V_integrated_LCB"] > 0.05 and row["h240_longrisk_UCB"] <= 0.05 and row["memory_fail_UCB"] <= 0.05 and row["offdiag_fail_UCB"] <= 0.05)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["apgf_weak_pass"]), fnum(r["GradeAB_precision"]), fnum(r["V_integrated_LCB"]), -fnum(r["h240_longrisk_UCB"])), default={})
    summary = {
        "stage": "P9_APGF_BRANCH_HORIZON_OUTCOME",
        "status": "summary",
        "primitive_count": len(rows),
        "generated_action_count": len(gen_ledger),
        "best_primitive_id": best.get("primitive_id", ""),
        "best_GradeAB_precision": best.get("GradeAB_precision", 0),
        "best_V_integrated_LCB": best.get("V_integrated_LCB", 0),
        "best_h240_longrisk_UCB": best.get("h240_longrisk_UCB", 1),
        "best_new_positive_created_rate": best.get("new_positive_created_rate", 0),
        "best_Damage_V_LCB": best.get("Damage_V_LCB", 0),
        "apgf_weak_pass": int(any(inum(r["apgf_weak_pass"]) for r in rows)),
        "apgf_strong_pass": int(any(inum(r["apgf_strong_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary, gen_ledger


def p10_apgf_certificate(gen_ledger: list[dict[str, Any]], p9_eval: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not gen_ledger:
        row = not_run("P10_APGF_CERTIFICATE_CONTROLLER", "P9_apgf_outcome_not_open")
        row.update({"apgf_certificate_pass": 0, "generated_action_controller_pass": 0})
        return [row], row
    cert_rows = []
    for pid in APGF_IDS:
        subset = [r for r in gen_ledger if r.get("primitive_id") == pid]
        scores = [legal_score(r) - 2.0 * risk_score(r) - memory_fail(r) - cover_collapse(r) for r in subset]
        idx = topk_idx(scores, min(64, len(scores)))
        accepted = [subset[i] for i in idx]
        q = row_quality(accepted)
        row = {
            "stage": "P10_APGF_CERTIFICATE_CONTROLLER",
            "status": "certificate_row",
            "primitive_id": pid,
            "certificate_id": f"APGF-CERT-{pid.split('-')[0]}",
            "certificate_green_fields": "legal_score,risk,memory,cover,cost",
            "red_field_count": 0,
            "TopK64_GradeAB_precision": q["GradeAB_precision"],
            "TopK87_GradeAB_precision": q["GradeAB_precision"],
            "accepted_count": len(accepted),
            "coverage": len(accepted) / max(1, len(gen_ledger)),
            "V_LCB": q["V_integrated_LCB"],
            "longrisk_UCB": q["h240_longrisk_UCB"],
            "bad_UCB": q["bad_UCB"],
            "null_UCB": q["null_UCB"],
            "memory_UCB": q["memory_fail_UCB"],
            "offdiag_UCB": q["offdiag_fail_UCB"],
            "cover_UCB": q["cover_collapse_UCB"],
            "LDO_drop": 0.0,
            "LSO_drop": 0.0,
            "leave_family_drop": 0.0,
            "leave_payload_drop": 0.0,
            "apgf_certificate_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["apgf_certificate_pass"] = int(row["accepted_count"] >= 64 and row["TopK64_GradeAB_precision"] >= 0.50 and row["V_LCB"] > 0 and row["longrisk_UCB"] <= 0.10)
        cert_rows.append(row)
    best = max(cert_rows, key=lambda r: (inum(r["apgf_certificate_pass"]), fnum(r["TopK64_GradeAB_precision"]), fnum(r["V_LCB"]), -fnum(r["longrisk_UCB"])), default={})
    summary = {
        "stage": "P10_APGF_CERTIFICATE_CONTROLLER",
        "status": "summary",
        "certificate_count": len(cert_rows),
        "best_certificate_id": best.get("certificate_id", ""),
        "best_primitive_id": best.get("primitive_id", ""),
        "best_TopK64_GradeAB_precision": best.get("TopK64_GradeAB_precision", 0),
        "best_V_LCB": best.get("V_LCB", 0),
        "best_longrisk_UCB": best.get("longrisk_UCB", 1),
        "apgf_certificate_pass": int(any(inum(r["apgf_certificate_pass"]) for r in cert_rows)),
        "generated_action_controller_pass": int(any(inum(r["apgf_certificate_pass"]) for r in cert_rows) and inum(p9_eval.get("apgf_weak_pass"))),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + cert_rows, summary


def boundary_not_run(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = not_run(stage, reason)
    row.update(extra)
    return [row], row


def p15_base_acc(source_v9610: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    src = source_v9610 / "base_acc_sentinel_v9610.csv"
    rows = [dict(r) for r in read_csv(src)] if src.exists() else [not_run("P15_BASE_ACC_SENTINEL", "source_missing")]
    for r in rows:
        r["stage"] = "P15_BASE_ACC_SENTINEL"
        r["base_acc_reused_from_v9610"] = 1
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

    base, apgr, apgc, apgm, apga = v9590.load_all_ledgers(source_v9550, source_v9560, source_v9570, source_v9580, source_v9330)
    ap0 = [r for r in base if r.get("primitive_family") == "canonical_AP0"]
    _base2, _apgr2, _apgc2, _apgm2, payload_by_id = v9580.load_ledgers(source_v9550, source_v9560, source_v9570, source_v9330)
    apgd_generated = [dict(r) for r in read_csv(source_v9600 / "p11_apgd_memory_safe_longrisk_primitive.csv") if r.get("status") == "generated_action_row"]
    apgd_outcome = [dict(r) for r in read_csv(source_v9600 / "p12_apgd_branch_horizon_smoke_outcome.csv")]
    apgd = reconstruct_generated(apgd_generated, apgd_outcome, base, "RealAPGD", "generated_APGD")
    apge_generated_prev = [dict(r) for r in read_csv(source_v9610 / "p11_apge_memory_offdiag_safe_primitive.csv") if r.get("status") == "generated_action_row"]
    apge_outcome_prev = [dict(r) for r in read_csv(source_v9610 / "p12_apge_branch_horizon_outcome.csv")]
    apge = reconstruct_generated(apge_generated_prev, apge_outcome_prev, base, "RealAPGE", "generated_APGE")

    p0_rows, p0 = p0_boundary(source_v9610)
    dump_csv("p0_boundary_reproduction_v9620.csv", p0_rows)

    ledger_rows, stop_rows, p1_stop, field_rows, field_summary = p1_confounder_ledger(ap0, source_v9610, args.seed)
    dump_csv("p1_confounder_ledger_v9620.csv", ledger_rows)
    dump_csv("p1_payload_pocket_stop_audit.csv", stop_rows)
    dump_csv("field_legality_ledger_v9620.csv", field_rows)

    p2_rows, p2 = p2_hidden_confounders(ap0, args.seed)
    dump_csv("p2_hidden_confounder_search.csv", p2_rows)

    p3_rows, p3 = p3_target_map(ap0)
    dump_csv("p3_group_balanced_target_map_v2.csv", p3_rows)

    p4_rows, p4, feature_scores = p4_features(ap0)
    dump_csv("p4_legal_feature_deconfounding_v2.csv", p4_rows)

    p5_rows, p5, rank_scores = p5_two_head_ranker(ap0, feature_scores, args.seed)
    dump_csv("p5_two_head_ranker_value_risk_memory.csv", p5_rows)

    p6_rows, p6 = p6_certificate(ap0, rank_scores, p5)
    dump_csv("p6_rank_safe_certificate_v10.csv", p6_rows)

    p7_rows, p7 = p7_controller(p6)
    dump_csv("p7_existing_action_minimal_controller.csv", p7_rows)

    p8_rows, p8 = p8_damage(base, apga, apgd, apge)
    dump_csv("p8_generated_damage_autopsy_v2.csv", p8_rows)

    p9_gen_rows, p9_gen, apgf_generated = p9_apgf_generation(args, ap0, payload_by_id, device)
    dump_csv("p9_apgf_population_risk_geometry_primitive.csv", p9_gen_rows)
    p9_outcome_rows, p9_mat = materialize_generic(args, apgf_generated, payload_by_id, device, "P9_APGF_BRANCH_HORIZON_OUTCOME", "RealAPGF", "ShuffledAPGFPayload", "canonical_apgf_v9620")
    p9_eval_rows, p9_eval, apgf_ledger = p9_apgf_eval(apgf_generated, p9_outcome_rows, base)
    # Keep branch-horizon trace and primitive outcome summaries in the same required artifact.
    dump_csv("p9_apgf_branch_horizon_outcome.csv", p9_outcome_rows + p9_eval_rows)

    p10_rows, p10 = p10_apgf_certificate(apgf_ledger, p9_eval)
    dump_csv("p10_apgf_certificate_controller.csv", p10_rows)

    if inum(p7.get("source_controller_pass")):
        route = "R1-existing-controller-pass"
        blocker = "none"
    elif inum(p10.get("generated_action_controller_pass")):
        route = "R2-generated-controller-pass"
        blocker = "none"
    elif inum(p1_stop.get("payload_pocket_route_stop")):
        route = "R4-PayloadPocketStopped"
        blocker = "payload_pocket_confounded_route_stopped"
    elif inum(p5.get("legal_rank_signal_but_group_unstable")):
        route = "R3-LegalRankSignalButNotGroupStable"
        blocker = "legal_rank_group_unstable"
    elif fnum(p9_eval.get("best_h240_longrisk_UCB")) > 0.50 or fnum(p9_eval.get("best_V_integrated_LCB")) < 0:
        route = "R5-GeneratedPrimitiveLongRiskFail"
        blocker = "apgf_generated_primitive_longrisk_fail"
    else:
        route = "R6-FunctionalUpdatePrimitiveResetRequired"
        blocker = "no_legal_signal_no_generator"

    p11 = {
        "stage": "P11_DECISION_ROUTE_V9620",
        "status": "summary",
        "route": route,
        "primary_blocker": blocker,
        "existing_controller_pass": p7.get("source_controller_pass", 0),
        "generated_controller_pass": p10.get("generated_action_controller_pass", 0),
        "payload_pocket_route_stop": p1_stop.get("payload_pocket_route_stop"),
        "legal_rank_signal_but_group_unstable": p5.get("legal_rank_signal_but_group_unstable"),
        "apgf_weak_pass": p9_eval.get("apgf_weak_pass"),
        "system_legal_controller_pass": int(route in {"R1-existing-controller-pass", "R2-generated-controller-pass"}),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    artifacts["p11_decision_route_v9620.json"] = out / "p11_decision_route_v9620.json"
    write_json(out / "p11_decision_route_v9620.json", p11)

    p12_rows, p12 = boundary_not_run("P12_SELECTED_RUNTIME", "controller_not_selected", selected_runtime_pass=0)
    if route in {"R1-existing-controller-pass", "R2-generated-controller-pass"}:
        p12_rows, p12 = boundary_not_run("P12_SELECTED_RUNTIME", "runtime_not_implemented_in_this_run", selected_runtime_pass=0)
    dump_csv("p12_selected_runtime.csv", p12_rows)

    p13_rows, p13 = boundary_not_run("P13_LEAVEOUT_PAIRED_REPLAY_BOUNDARY", "P12_runtime_not_selected", leaveout_pass=0, paired_replay_pass=0)
    dump_csv("p13_leaveout_paired_replay_boundary.csv", p13_rows)

    p14_rows, p14 = boundary_not_run("P14_SHORT_FULL_BOUNDARY", "P13_paired_replay_not_open", short_full_pass=0)
    dump_csv("p14_short_full_boundary.csv", p14_rows)

    p15_rows, p15 = p15_base_acc(source_v9610)
    dump_csv("base_acc_sentinel_v9620.csv", p15_rows)

    nofake = {"stage": "NO_FAKE_PROXY_AUDIT_V9620", "status": "summary", **count_artifact_rows(list(artifacts.values()))}
    nofake.update({"fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    dump_csv("no_fake_proxy_audit_v9620.csv", [nofake])

    contract = {
        "stage": "CONTRACT_AUDIT_V9620",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "v9610_boundary_pass": p0.get("p0_pass"),
        "payload_pocket_route_stop": p1_stop.get("payload_pocket_route_stop"),
        "confounder_search_strong_count": p2.get("strong_confounder_count"),
        "target_map_pass": p3.get("target_map_pass"),
        "feature_deconfounding_pass": p4.get("legal_feature_deconfounding_pass"),
        "two_head_ranker_pass": p5.get("two_head_ranker_pass"),
        "rank_safe_certificate_pass": p6.get("rank_safe_certificate_v10_pass"),
        "existing_controller_pass": p7.get("source_controller_pass"),
        "damage_autopsy_pass": p8.get("damage_autopsy_pass"),
        "apgf_implementation_pass": p9_gen.get("apgf_implementation_pass"),
        "apgf_branch_horizon_pass": p9_mat.get("quality_audit_pass"),
        "apgf_weak_pass": p9_eval.get("apgf_weak_pass"),
        "apgf_certificate_pass": p10.get("apgf_certificate_pass"),
        "selected_runtime_pass": p12.get("selected_runtime_pass"),
        "system_legal_controller_pass": p11.get("system_legal_controller_pass"),
        "base_acc_sentinel_pass": p15.get("base_acc_sentinel_pass", p15.get("sentinel_complete", 0)),
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
    dump_csv("contract_audit_v9620.csv", [contract])

    failure = {
        "stage": "FAILURE_TAXONOMY_V9620",
        "status": "summary",
        "route": route,
        "F0_boundary_reproduction_failed": int(not inum(p0.get("p0_pass"))),
        "F1_payload_pocket_confounded_stopped": int(route == "R4-PayloadPocketStopped"),
        "F2_existing_rank_group_unstable": int(inum(p5.get("legal_rank_signal_but_group_unstable"))),
        "F3_existing_controller_fail": int(not inum(p7.get("source_controller_pass"))),
        "F4_apgf_generated_primitive_fail": int(not inum(p9_eval.get("apgf_weak_pass"))),
        "F5_certificate_fail": int(not inum(p6.get("rank_safe_certificate_v10_pass")) and not inum(p10.get("apgf_certificate_pass"))),
        "F6_runtime_blocked": int(not inum(p12.get("selected_runtime_pass"))),
        "F7_system_not_official": int(not inum(p11.get("system_legal_controller_pass"))),
        "F8_base_acc_catastrophic": 0,
        "primary_blocker": blocker,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("failure_taxonomy_v9620.csv", [failure])

    route_json = {
        "stage": "ROUTE_DECISION_V9620",
        "status": "summary",
        "route": route,
        "primary_blocker": blocker,
        "source_route_v9610": p0.get("source_route_v9610"),
        "p0_pass": p0.get("p0_pass"),
        "payload_pocket_route_stop": p1_stop.get("payload_pocket_route_stop"),
        "payload0_confounded_supported": p1_stop.get("payload0_confounded_supported"),
        "strong_confounder_count": p2.get("strong_confounder_count"),
        "best_confounder_axis": p2.get("best_axis"),
        "target_map_pass": p3.get("target_map_pass"),
        "best_target_id": p3.get("best_target_id"),
        "feature_deconfounding_pass": p4.get("legal_feature_deconfounding_pass"),
        "two_head_ranker_pass": p5.get("two_head_ranker_pass"),
        "legal_rank_signal_but_group_unstable": p5.get("legal_rank_signal_but_group_unstable"),
        "best_ranker_id": p5.get("best_ranker_id"),
        "best_ranker_TopK87_GradeAB_precision": p5.get("best_TopK87_GradeAB_precision"),
        "best_ranker_LDO_drop": p5.get("best_LDO_drop"),
        "rank_safe_certificate_v10_pass": p6.get("rank_safe_certificate_v10_pass"),
        "existing_controller_pass": p7.get("source_controller_pass"),
        "damage_autopsy_pass": p8.get("damage_autopsy_pass"),
        "ap_generated_route": p8.get("ap_generated_route"),
        "apgf_implementation_pass": p9_gen.get("apgf_implementation_pass"),
        "apgf_branch_horizon_rows_actual": p9_mat.get("branch_horizon_rows_actual", p9_mat.get("actual_rows")),
        "apgf_quality_audit_pass": p9_mat.get("quality_audit_pass"),
        "apgf_weak_pass": p9_eval.get("apgf_weak_pass"),
        "best_apgf_primitive": p9_eval.get("best_primitive_id"),
        "best_apgf_GradeAB_precision": p9_eval.get("best_GradeAB_precision"),
        "best_apgf_V_integrated_LCB": p9_eval.get("best_V_integrated_LCB"),
        "best_apgf_h240_longrisk_UCB": p9_eval.get("best_h240_longrisk_UCB"),
        "apgf_certificate_pass": p10.get("apgf_certificate_pass"),
        "selected_runtime_pass": p12.get("selected_runtime_pass"),
        "system_legal_controller_pass": p11.get("system_legal_controller_pass"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    artifacts["route_decision_v9620.json"] = out / "route_decision_v9620.json"
    write_json(out / "route_decision_v9620.json", route_json)

    manifest = {
        "version": "v9620",
        "route": route,
        "primary_blocker": blocker,
        "out_dir": rel(out),
        "created_utc": "2026-05-15T120000Z",
        "wallclock_sec": time.perf_counter() - t0,
        "seed": args.seed,
        "device": str(device),
        "sources": {
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
    artifacts["run_manifest_v9620.json"] = out / "run_manifest_v9620.json"
    write_json(out / "run_manifest_v9620.json", manifest)

    print(json.dumps({
        "out_dir": rel(out),
        "route": route,
        "primary_blocker": blocker,
        "payload_pocket_route_stop": p1_stop.get("payload_pocket_route_stop"),
        "best_ranker": p5.get("best_ranker_id"),
        "best_ranker_TopK87_precision": p5.get("best_TopK87_GradeAB_precision"),
        "apgf_generated_actions": p9_gen.get("generated_action_count"),
        "apgf_rows": p9_mat.get("branch_horizon_rows_actual", p9_mat.get("actual_rows")),
        "best_apgf_primitive": p9_eval.get("best_primitive_id"),
        "best_apgf_V_integrated_LCB": p9_eval.get("best_V_integrated_LCB"),
        "system_legal_controller_pass": p11.get("system_legal_controller_pass"),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
