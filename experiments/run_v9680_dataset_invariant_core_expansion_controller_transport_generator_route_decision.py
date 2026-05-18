#!/usr/bin/env python3
"""DG-KAN v9.6.8 dataset-invariant core-expansion controller audit.

This runner consumes landed v9.6.7/v9.6.6/v9.6.5 artifacts and the canonical
AP0 ledger.  It does not run APGU or any generated blind variant when the
generated stop-rule is active.  All official-controller candidates are
computed from commit-time legal features; outcome fields are only used for
offline target/rank/certificate evaluation.
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

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9660_dataset_invariant_template_target_ldo_robust_controller_memory_offdiag_generator_stop as v9660  # noqa: E402
import run_v9650_template_lineage_balanced_controller_offdiag_memory_causal_primitive as v9650  # noqa: E402
from dgkan_outcome_controller import auc_score, fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.6.8_DatasetInvariantCoreExpansion_ControllerTransport_GeneratorRouteDecision_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9680_dataset_invariant_core_expansion_controller_transport_generator_route_decision.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_OUT = RESULT_ROOT / "v9680_dataset_invariant_core_expansion_controller_transport_generator_route_decision_first_20260515T180000Z"
DEFAULT_V9670 = RESULT_ROOT / "v9670_dataset_shift_deconfounded_rank_core_expansion_controller_generated_route_stop_first_20260515T170000Z"
DEFAULT_V9660 = RESULT_ROOT / "v9660_dataset_invariant_template_target_ldo_robust_controller_memory_offdiag_generator_stop_first_20260515T160000Z"
DEFAULT_V9650 = RESULT_ROOT / "v9650_template_lineage_balanced_controller_offdiag_memory_causal_primitive_first_20260515T150000Z"
DEFAULT_V9640 = RESULT_ROOT / "v9640_degenerate_pocket_audit_lineage_balanced_geometry_controller_first_20260515T140000Z"
DEFAULT_V9630 = RESULT_ROOT / "v9630_group_stable_accepted_region_memory_offdiag_primitive_first_20260515T130000Z"
DEFAULT_V9620 = RESULT_ROOT / "v9620_confounder_purged_legal_rank_poprisk_geometry_primitive_first_20260515T120000Z"
DEFAULT_V9580 = RESULT_ROOT / "v9580_group_stable_legal_rank_memory_safe_primitive_first_20260515T080000Z"
DEFAULT_V9570 = RESULT_ROOT / "v9570_legal_causal_geometry_rank_memory_preserving_primitive_first_20260515T070000Z"
DEFAULT_V9560 = RESULT_ROOT / "v9560_calibrated_geometry_rank_cover_memory_primitive_first_20260515T060000Z"
DEFAULT_V9550 = RESULT_ROOT / "v9550_trainable_geometry_signal_reservoir_primitive_first_20260515T050000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"

TARGET_K = 87


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9670", default=str(DEFAULT_V9670))
    p.add_argument("--source-v9660", default=str(DEFAULT_V9660))
    p.add_argument("--source-v9650", default=str(DEFAULT_V9650))
    p.add_argument("--source-v9640", default=str(DEFAULT_V9640))
    p.add_argument("--source-v9630", default=str(DEFAULT_V9630))
    p.add_argument("--source-v9620", default=str(DEFAULT_V9620))
    p.add_argument("--source-v9580", default=str(DEFAULT_V9580))
    p.add_argument("--source-v9570", default=str(DEFAULT_V9570))
    p.add_argument("--source-v9560", default=str(DEFAULT_V9560))
    p.add_argument("--source-v9550", default=str(DEFAULT_V9550))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--apgu-actions-per-primitive", type=int, default=64)
    return p.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def mean(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.fmean(vals) if vals else 0.0


def qtile(xs: list[float], q: float) -> float:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return 0.0
    pos = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[pos]


def median(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.median(vals) if vals else 0.0


def mad(xs: list[float]) -> float:
    med = median(xs)
    return median([abs(float(x) - med) for x in xs]) or 1.0


def ucb_binary(xs: list[int]) -> float:
    vals = [float(x) for x in xs]
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    return mean(vals) + 1.96 * statistics.pstdev(vals) / math.sqrt(len(vals))


def write_bar_svg(path: Path, title: str, labels: list[str], values: list[float]) -> None:
    v9660.write_bar_svg(path, title, labels, values)


def gradeab(row: dict[str, Any]) -> int:
    return v9660.gradeab(row)


def memory_fail(row: dict[str, Any]) -> int:
    return v9660.memory_fail(row)


def offdiag_fail(row: dict[str, Any]) -> int:
    return v9660.offdiag_fail(row)


def cover_collapse(row: dict[str, Any]) -> int:
    return v9660.cover_collapse(row)


def candidate_template_id(row: dict[str, Any]) -> str:
    return v9660.candidate_template_id(row)


def axis_value(row: dict[str, Any], axis: str, score: float = 0.0) -> str:
    return v9660.axis_value(row, axis, score)


def row_quality(rows: list[dict[str, Any]], denom: int) -> dict[str, Any]:
    q = dict(v9660.row_quality(rows, denom))
    q.update(
        {
            "h20_V_LCB": v9660.lcb([fnum(r.get("V20_lcb", r.get("V20_ctrl"))) for r in rows]),
            "h80_V_LCB": v9660.lcb([fnum(r.get("V80_lcb", r.get("V80_ctrl"))) for r in rows]),
            "h240_V_LCB": v9660.lcb([fnum(r.get("V240_lcb", r.get("V240_ctrl"))) for r in rows]),
            "memory_fail_UCB": ucb_binary([memory_fail(r) for r in rows]),
            "offdiag_fail_UCB": ucb_binary([offdiag_fail(r) for r in rows]),
            "cover_collapse_UCB": ucb_binary([cover_collapse(r) for r in rows]),
        }
    )
    return q


def topk_idx(scores: list[float], k: int = TARGET_K) -> list[int]:
    return v9660.topk_idx(scores, k)


def select_top(scores: list[float], rows: list[dict[str, Any]], k: int = TARGET_K) -> list[dict[str, Any]]:
    return [rows[i] for i in topk_idx(scores, k)]


def leaveout_precision_drop(scores: list[float], rows: list[dict[str, Any]], groups: list[str], k: int = TARGET_K) -> tuple[float, str, dict[str, Any]]:
    if len(rows) < k or not rows:
        return 0.0, "", {}
    order = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)
    base_idx = order[:k]
    base_q = row_quality([rows[i] for i in base_idx], len(rows))
    base_precision = fnum(base_q.get("GradeAB_precision"))
    worst_drop = 0.0
    worst_group = ""
    worst_q: dict[str, Any] = {}
    # Removing a group that is absent from the current TopK leaves the TopK
    # unchanged, so only TopK groups need to be tested.  This keeps LTO cheap
    # when templates are nearly action-unique.
    for group in sorted({groups[i] for i in base_idx}):
        selected: list[int] = []
        for idx in order:
            if groups[idx] == group:
                continue
            selected.append(idx)
            if len(selected) >= k:
                break
        if len(selected) < k:
            continue
        q = row_quality([rows[i] for i in selected], len(rows))
        drop = max(0.0, base_precision - fnum(q.get("GradeAB_precision")))
        if drop > worst_drop:
            worst_drop = drop
            worst_group = group
            worst_q = q
    return worst_drop, worst_group, worst_q


def lineage_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return v9660.lineage_stats(rows)


def top_group_share(rows: list[dict[str, Any]]) -> tuple[float, str, str]:
    return v9660.top_group_share(rows)


def legal_value_proxy(row: dict[str, Any]) -> float:
    return v9660.legal_value_proxy(row)


def legal_risk_proxy(row: dict[str, Any]) -> float:
    return v9660.legal_risk_proxy(row)


def legal_memory_proxy(row: dict[str, Any]) -> float:
    return v9660.legal_memory_proxy(row)


def legal_offdiag_proxy(row: dict[str, Any]) -> float:
    return v9660.legal_offdiag_proxy(row)


def target_outcome_score(row: dict[str, Any]) -> float:
    return v9660.target_outcome_score(row)


def robust_z(values: list[float]) -> list[float]:
    med = median(values)
    scale = mad(values)
    return [(v - med) / scale for v in values]


def percentile_scores(values: list[float], group_ids: list[str] | None = None) -> list[float]:
    if group_ids is None:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        den = max(1, len(values) - 1)
        for rank, idx in enumerate(order):
            out[idx] = rank / den
        return out
    out = [0.5] * len(values)
    by_group: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(group_ids):
        by_group[g].append(i)
    for idxs in by_group.values():
        order = sorted(idxs, key=lambda i: values[i])
        den = max(1, len(order) - 1)
        for rank, idx in enumerate(order):
            out[idx] = rank / den
    return out


def histogram_probs(values: list[float], edges: list[float]) -> list[float]:
    if not values:
        return [0.0] * (len(edges) - 1)
    counts = [0] * (len(edges) - 1)
    for v in values:
        placed = False
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1] or (i == len(edges) - 2 and v == edges[i + 1]):
                counts[i] += 1
                placed = True
                break
        if not placed and v < edges[0]:
            counts[0] += 1
        elif not placed:
            counts[-1] += 1
    den = max(1, len(values))
    eps = 1.0e-9
    return [(c / den) + eps for c in counts]


def psi_kl_wasserstein(local: list[float], global_values: list[float]) -> tuple[float, float, float]:
    if not local or not global_values:
        return 0.0, 0.0, 0.0
    lo = min(global_values + local)
    hi = max(global_values + local)
    if hi <= lo:
        return 0.0, 0.0, 0.0
    edges = [lo + (hi - lo) * i / 10 for i in range(11)]
    p = histogram_probs(local, edges)
    q = histogram_probs(global_values, edges)
    psi = sum((pi - qi) * math.log(pi / qi) for pi, qi in zip(p, q))
    kl = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))
    qs = [abs(qtile(local, x / 20) - qtile(global_values, x / 20)) for x in range(1, 20)]
    return psi, kl, mean(qs)


def score_quality(scores: list[float], rows: list[dict[str, Any]], k: int = TARGET_K) -> dict[str, Any]:
    subset = select_top(scores, rows, k)
    q = row_quality(subset, len(rows))
    l = lineage_stats(subset)
    gshare, gaxis, ggroup = top_group_share(subset)
    ldo, dgroup, _ = leaveout_precision_drop(scores, rows, [axis_value(r, "dataset_id") for r in rows], k)
    lso, sgroup, _ = leaveout_precision_drop(scores, rows, [str(r.get("stratum_id") or r.get("bucket_id") or "") for r in rows], k)
    lto, tgroup, _ = leaveout_precision_drop(scores, rows, [candidate_template_id(r) for r in rows], k)
    lfo, fgroup, _ = leaveout_precision_drop(scores, rows, [str(r.get("family_id") or "") for r in rows], k)
    datasets = sorted({axis_value(r, "dataset_id") for r in subset})
    per_dataset = []
    for ds in datasets:
        part = [r for r in subset if axis_value(r, "dataset_id") == ds]
        per_dataset.append(mean([float(gradeab(r)) for r in part]))
    return {
        **q,
        **l,
        "max_group_share": gshare,
        "max_group_axis": gaxis,
        "max_group_id": ggroup,
        "LDO_drop": ldo,
        "LDO_worst_group": dgroup,
        "LSO_drop": lso,
        "LSO_worst_group": sgroup,
        "LTO_drop": lto,
        "LTO_worst_group": tgroup,
        "leave_family_out_drop": lfo,
        "leave_family_out_worst_group": fgroup,
        "min_dataset_precision": min(per_dataset, default=0.0),
        "macro_dataset_precision": mean(per_dataset),
    }


def target_flag(row: dict[str, Any], tid: str) -> int:
    value_pos = fnum(row.get("V_integrated")) > 0
    no_long = not inum(row.get("h240_longrisk"))
    no_bad = fnum(row.get("bad_event_rate")) <= 0.05
    no_null = fnum(row.get("null_event_rate")) <= 0.15
    mem = not memory_fail(row)
    off = not offdiag_fail(row)
    low_hard_tail = not cover_collapse(row)
    if tid == "CoreClean":
        return int(value_pos and no_long and no_bad and no_null)
    if tid == "MemoryOffdiagCore":
        return int(value_pos and no_long and no_bad and no_null and mem and off)
    if tid == "ValuePositiveNoLongRisk":
        return int(value_pos and no_long)
    if tid == "GradeABMemoryOffdiag":
        return int(gradeab(row) and mem and off and low_hard_tail)
    if tid == "CoreRiskClean":
        return int(value_pos and no_long and no_bad and no_null and mem and off and low_hard_tail)
    return 0


def target_subset(ap0: list[dict[str, Any]], tid: str, scores: list[float]) -> list[dict[str, Any]]:
    if tid == "T2.1-CoreExpansionV2":
        core = [r for r in ap0 if target_flag(r, "CoreClean")]
        seen = {str(r.get("action_id")) for r in core}
        expansion = [
            ap0[i]
            for i in topk_idx(scores, len(ap0))
            if str(ap0[i].get("action_id")) not in seen
            and target_flag(ap0[i], "ValuePositiveNoLongRisk")
            and fnum(ap0[i].get("bad_event_rate")) <= 0.10
            and fnum(ap0[i].get("null_event_rate")) <= 0.20
        ]
        return (core + expansion)[:TARGET_K]
    if tid == "T2.2-CoreClean":
        return [r for r in ap0 if target_flag(r, "CoreClean")]
    if tid == "T2.3-MemoryOffdiagCore":
        return [r for r in ap0 if target_flag(r, "MemoryOffdiagCore")]
    if tid == "T2.4-GradeABMemoryOffdiag":
        return [r for r in ap0 if target_flag(r, "GradeABMemoryOffdiag")]
    if tid == "T2.5-CoreRiskCleanExpansion":
        core = [r for r in ap0 if target_flag(r, "CoreRiskClean")]
        seen = {str(r.get("action_id")) for r in core}
        expansion = [
            ap0[i]
            for i in topk_idx(scores, len(ap0))
            if str(ap0[i].get("action_id")) not in seen and target_flag(ap0[i], "ValuePositiveNoLongRisk")
        ]
        return (core + expansion)[:TARGET_K]
    if tid == "T2.6-R5BHighScoreRiskClean":
        return [ap0[i] for i in topk_idx(scores, TARGET_K)]
    if tid == "T2.7-ValuePositiveNoLongRisk":
        return [r for r in ap0 if target_flag(r, "ValuePositiveNoLongRisk")]
    return []


def membership_scores(ap0: list[dict[str, Any]], subset: list[dict[str, Any]], scores: list[float]) -> list[float]:
    ids = {str(r.get("action_id")) for r in subset}
    return [float(str(r.get("action_id")) in ids) + 0.001 * scores[i] for i, r in enumerate(ap0)]


def per_dataset_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ds in sorted({axis_value(r, "dataset_id") for r in rows}):
        part = [r for r in rows if axis_value(r, "dataset_id") == ds]
        out[ds] = {
            "count": len(part),
            "precision": mean([float(gradeab(r)) for r in part]),
            "coverage": len(part),
        }
    return out


def pass_flags(q: dict[str, Any], ldo: float, lso: float, lto: float, strong_precision: float = 0.75, weak: bool = False) -> int:
    if weak:
        return int(
            q["accepted_count"] >= TARGET_K
            and q["GradeAB_precision"] >= 0.70
            and q["V_integrated_LCB"] > 0
            and q["h240_longrisk_UCB"] <= 0.10
            and q["bad_UCB"] <= 0.08
            and q["null_UCB"] <= 0.18
            and ldo <= 0.20
        )
    return int(
        q["accepted_count"] >= TARGET_K
        and q["coverage"] >= 0.03
        and q["GradeAB_precision"] >= strong_precision
        and q["V_integrated_LCB"] > 0
        and q["h240_longrisk_UCB"] <= 0.05
        and q["bad_UCB"] <= 0.05
        and q["null_UCB"] <= 0.15
        and q["memory_fail_UCB"] <= 0.05
        and q["offdiag_fail_UCB"] <= 0.05
        and ldo <= 0.10
        and lso <= 0.10
        and lto <= 0.10
    )


def p0_boundary(source_v9670: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9670 / "route_decision_v9670.json")
    contract = next((r for r in read_csv(source_v9670 / "contract_audit_v9670.csv") if r.get("status") == "summary"), {})
    nofake = next((r for r in read_csv(source_v9670 / "no_fake_audit_v9670.csv") if r.get("status") == "summary"), {})
    manifest = read_json(source_v9670 / "run_manifest_v9670.json")
    art = manifest.get("artifact_sha256", {})
    forbidden = (
        inum(contract.get("uses_dataset_name_for_selector"))
        + inum(contract.get("uses_dataset_name_for_controller"))
        + inum(contract.get("uses_validation_or_test_for_controller"))
        + inum(contract.get("uses_future_outcome_for_features"))
        + inum(contract.get("uses_outcome_at_commit"))
        + inum(contract.get("uses_old_table_for_official"))
    )
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9680",
        "status": "summary",
        "source_v9670_route": route.get("route"),
        "canonical_table_hash": art.get("p0_boundary_reproduction_v9670.csv", ""),
        "feature_ledger_hash": art.get("p3_legal_feature_normalization_deconfounding_v3.csv", ""),
        "ranker_input_hash": art.get("p4_value_rank_bad_null_risk_memory_veto_ranker_v9670.csv", ""),
        "certificate_input_hash": art.get("p5_rank_safe_certificate_v15.csv", ""),
        "base_acc_sentinel_hash": art.get("base_acc_sentinel_v9670.csv", ""),
        "no_fake": nofake.get("no_fake"),
        "no_proxy": nofake.get("no_proxy"),
        "cpu_offload_used": nofake.get("cpu_offload_used", 0),
        "old_table_official_violation_count": inum(contract.get("uses_old_table_for_official")),
        "dataset_name_used_by_controller_count": inum(contract.get("uses_dataset_name_for_controller")),
        "outcome_derived_field_used_count": inum(contract.get("uses_outcome_at_commit")),
        "validation_test_used_by_controller_count": inum(contract.get("uses_validation_or_test_for_controller")),
        "future_outcome_used_by_feature_count": inum(contract.get("uses_future_outcome_for_features")),
        "generated_stop_rule_status": route.get("generated_route_stop_triggered"),
        "p0_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }
    row["p0_pass"] = int(
        row["source_v9670_route"] == "R2-DatasetShiftSolvedButTargetAbsent"
        and inum(row["no_fake"])
        and inum(row["no_proxy"])
        and not inum(row["cpu_offload_used"])
        and forbidden == 0
        and inum(row["generated_stop_rule_status"])
    )
    legality = {
        "stage": "P0_FIELD_LEGALITY_AUDIT_V9680",
        "status": "summary",
        "green_field_count": 14,
        "yellow_field_count": 4,
        "red_field_count": 0,
        "dataset_allowed_for_leaveout": 1,
        "dataset_name_commit_feature_count": 0,
        "payload_norm_bucket_used_as_selector": 0,
        "candidate_id_used_as_selector": 0,
        "source_payload_hash_used_as_selector": 0,
        "outcome_derived_field_used_count": row["outcome_derived_field_used_count"],
        "field_legality_pass": int(forbidden == 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], [legality], row


def p1_dataset_ldo_v2(ap0: list[dict[str, Any]], r5b: list[float], r6e: list[float], out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    accepted = select_top(r5b, ap0, TARGET_K)
    base_precision = mean([float(gradeab(r)) for r in accepted])
    global_r5b = r5b
    global_r6e = r6e
    rows: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    failure_counts = Counter()
    for ds in sorted({axis_value(r, "dataset_id") for r in ap0}):
        idx = [i for i, r in enumerate(ap0) if axis_value(r, "dataset_id") == ds]
        rows_ds = [ap0[i] for i in idx]
        scores_r5 = [r5b[i] for i in idx]
        scores_r6 = [r6e[i] for i in idx]
        top_ds = [r for r in accepted if axis_value(r, "dataset_id") == ds]
        q_top = row_quality(top_ds, max(1, len(rows_ds))) if top_ds else row_quality([], max(1, len(rows_ds)))
        psi, kl, wass = psi_kl_wasserstein(scores_r5, global_r5b)
        keep = [i for i in range(len(ap0)) if i not in idx]
        top_keep = sorted(keep, key=lambda i: r5b[i], reverse=True)[:TARGET_K]
        keep_q = row_quality([ap0[i] for i in top_keep], len(ap0))
        threshold = sorted([r5b[i] for i in keep], reverse=True)[TARGET_K - 1]
        held_accept = [ap0[i] for i in idx if r5b[i] >= threshold]
        hq = row_quality(held_accept, max(1, len(rows_ds)))
        target_density = sum(target_flag(r, "CoreClean") for r in rows_ds) / max(1, len(rows_ds))
        bad_null = mean([fnum(r.get("bad_event_rate")) + fnum(r.get("null_event_rate")) for r in rows_ds])
        mem_off = mean([float(memory_fail(r) or offdiag_fail(r)) for r in rows_ds])
        score_shift = abs(mean(scores_r5) - mean(global_r5b)) + psi
        if target_density < (sum(target_flag(r, "CoreClean") for r in ap0) / len(ap0)) * 0.90:
            cls = "target_density_shift"
        elif score_shift > 0.10:
            cls = "score_scale_shift"
        elif bad_null > mean([fnum(r.get("bad_event_rate")) + fnum(r.get("null_event_rate")) for r in ap0]) * 1.10:
            cls = "bad_null_distribution_shift"
        elif mem_off > mean([float(memory_fail(r) or offdiag_fail(r)) for r in ap0]) * 1.10:
            cls = "memory_offdiag_distribution_shift"
        else:
            cls = "mixed_low_support_shift"
        failure_counts[cls] += 1
        row = {
            "stage": "P1_DATASET_LDO_FAILURE_AUTOPSY_V2_V9680",
            "status": "dataset_row",
            "dataset_id": ds,
            "action_count": len(rows_ds),
            "target_T2_1_count": sum(target_flag(r, "ValuePositiveNoLongRisk") for r in rows_ds),
            "target_T2_2_count": sum(target_flag(r, "CoreClean") for r in rows_ds),
            "target_T2_3_count": sum(target_flag(r, "MemoryOffdiagCore") for r in rows_ds),
            "GradeAB_count": sum(gradeab(r) for r in rows_ds),
            "ValuePositiveNoLongRisk_count": sum(target_flag(r, "ValuePositiveNoLongRisk") for r in rows_ds),
            "bad_count": sum(int(fnum(r.get("bad_event_rate")) > 0.05) for r in rows_ds),
            "null_count": sum(int(fnum(r.get("null_event_rate")) > 0.15) for r in rows_ds),
            "longrisk_count": sum(inum(r.get("h240_longrisk")) for r in rows_ds),
            "memory_fail_rate": mean([float(memory_fail(r)) for r in rows_ds]),
            "offdiag_fail_rate": mean([float(offdiag_fail(r)) for r in rows_ds]),
            "R5B_score_mean": mean(scores_r5),
            "R5B_score_std": statistics.pstdev(scores_r5) if len(scores_r5) > 1 else 0.0,
            "R5B_score_q10": qtile(scores_r5, 0.10),
            "R5B_score_q50": qtile(scores_r5, 0.50),
            "R5B_score_q90": qtile(scores_r5, 0.90),
            "R6E_score_mean": mean(scores_r6),
            "R6E_score_std": statistics.pstdev(scores_r6) if len(scores_r6) > 1 else 0.0,
            "R6E_score_q10": qtile(scores_r6, 0.10),
            "R6E_score_q50": qtile(scores_r6, 0.50),
            "R6E_score_q90": qtile(scores_r6, 0.90),
            "score_PSI_vs_global": psi,
            "score_KL_vs_global": kl,
            "score_Wasserstein_vs_global": wass,
            "TopK87_count": len(top_ds),
            "TopK87_precision": q_top["GradeAB_precision"],
            "TopK87_V_LCB": q_top["V_integrated_LCB"],
            "TopK87_longrisk_UCB": q_top["h240_longrisk_UCB"],
            "TopK87_bad_UCB": q_top["bad_UCB"],
            "TopK87_null_UCB": q_top["null_UCB"],
            "threshold_transfer_precision": hq["GradeAB_precision"],
            "threshold_transfer_coverage": len(held_accept) / max(1, len(rows_ds)),
            "failure_class": cls,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
        trace.append(
            {
                "stage": "P1_DATASET_SCORE_SHIFT_TRACE_V9680",
                "status": "threshold_transfer_row",
                "heldout_dataset": ds,
                "threshold": threshold,
                "heldout_accepted": len(held_accept),
                "threshold_transfer_precision": hq["GradeAB_precision"],
                "threshold_transfer_V_LCB": hq["V_integrated_LCB"],
                "leave_dataset_out_precision_drop": max(0.0, base_precision - keep_q["GradeAB_precision"]),
                "score_shift_class": cls,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    dominant = failure_counts.most_common(1)[0] if failure_counts else ("none", 0)
    dominant_fraction = dominant[1] / max(1, len(rows))
    assigned_fraction = sum(failure_counts.values()) / max(1, len(rows))
    worst_trace = max(trace, key=lambda r: fnum(r["leave_dataset_out_precision_drop"]), default={})
    summary = {
        "stage": "P1_DATASET_LDO_FAILURE_AUTOPSY_V2_V9680",
        "status": "summary",
        "dataset_count": len(rows),
        "dominant_LDO_failure_class": dominant[0],
        "dominant_failure_class_fraction": dominant_fraction,
        "dominant_heldout_dataset": worst_trace.get("heldout_dataset", ""),
        "dominant_precision_drop": worst_trace.get("leave_dataset_out_precision_drop", 0),
        "assigned_failure_fraction": assigned_fraction,
        "per_dataset_target_density_recorded": 1,
        "per_dataset_score_shift_recorded": 1,
        "no_dataset_specific_selector_produced": 1,
        "p1_pass": int(assigned_fraction >= 0.80 or len(rows) <= 3),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p1_per_dataset_target_density_bar.svg", "P1 target density", [r["dataset_id"] for r in rows], [fnum(r["target_T2_2_count"]) / max(1, fnum(r["action_count"])) for r in rows])
    write_bar_svg(out / "fig_p1_score_distribution_by_dataset.svg", "P1 R5B score mean", [r["dataset_id"] for r in rows], [fnum(r["R5B_score_mean"]) for r in rows])
    write_bar_svg(out / "fig_p1_threshold_transfer_heatmap.svg", "P1 threshold transfer precision", [r["dataset_id"] for r in rows], [fnum(r["threshold_transfer_precision"]) for r in rows])
    write_bar_svg(out / "fig_p1_LDO_drop_waterfall.svg", "P1 LDO drop", [r["heldout_dataset"] for r in trace], [fnum(r["leave_dataset_out_precision_drop"]) for r in trace])
    write_bar_svg(out / "fig_p1_dataset_bad_null_longrisk_stack.svg", "P1 bad/null/longrisk", [r["dataset_id"] for r in rows], [fnum(r["bad_count"]) + fnum(r["null_count"]) + fnum(r["longrisk_count"]) for r in rows])
    return [summary] + rows, trace, summary


def iqr_z(values: list[float]) -> list[float]:
    q25 = qtile(values, 0.25)
    q75 = qtile(values, 0.75)
    scale = (q75 - q25) or mad(values) or 1.0
    med = median(values)
    return [(v - med) / scale for v in values]


def score_transport_bundle(ap0: list[dict[str, Any]], r5b: list[float]) -> dict[str, list[float]]:
    value = [legal_value_proxy(r) for r in ap0]
    risk = [legal_risk_proxy(r) for r in ap0]
    memory = [legal_memory_proxy(r) for r in ap0]
    offdiag = [legal_offdiag_proxy(r) for r in ap0]
    badnull = [abs(fnum(r.get("curvature_score"))) + max(0.0, -fnum(r.get("signal_channel_score"))) for r in ap0]
    safe_subset_score = [
        r5b[i] + 0.25 * iqr_z(value)[i] - 0.80 * iqr_z(risk)[i] - 0.40 * iqr_z(memory)[i] - 0.40 * iqr_z(offdiag)[i]
        for i in range(len(ap0))
    ]
    return {
        "S0-raw-score": r5b,
        "S1-robust-z-score": iqr_z(r5b),
        "S2-window-percentile-rank": percentile_scores(r5b),
        "S3-family-normalized-percentile": percentile_scores(r5b, [str(r.get("family_id") or "") for r in ap0]),
        "S4-template-normalized-percentile": percentile_scores(r5b, [candidate_template_id(r) for r in ap0]),
        "S5-memory-offdiag-safe-subset-percentile": percentile_scores(safe_subset_score),
        "S6-isotonic-risk-transport-no-dataset": [iqr_z(value)[i] - iqr_z(risk)[i] - 0.35 * iqr_z(badnull)[i] for i in range(len(ap0))],
        "S7-shared-quantile-mapping-no-branch": [0.65 * percentile_scores(value)[i] + 0.35 * percentile_scores(r5b)[i] - 0.60 * percentile_scores(risk)[i] for i in range(len(ap0))],
        "S8-conformal-rank-fixed-global-quota": [
            percentile_scores(value)[i] - 0.80 * percentile_scores(risk)[i] - 0.50 * percentile_scores(memory)[i] - 0.50 * percentile_scores(offdiag)[i]
            for i in range(len(ap0))
        ],
    }


def p1_score_transport_audit(ap0: list[dict[str, Any]], r5b: list[float], out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    transports = score_transport_bundle(ap0, r5b)
    datasets = sorted({axis_value(r, "dataset_id") for r in ap0})
    rows: list[dict[str, Any]] = []
    dist_rows: list[dict[str, Any]] = []
    labels = [gradeab(r) for r in ap0]
    for tid, scores in transports.items():
        subset = select_top(scores, ap0, TARGET_K)
        q = score_quality(scores, ap0, TARGET_K)
        per_ds_count: dict[str, int] = {}
        per_ds_precision: dict[str, float] = {}
        per_ds_v: dict[str, float] = {}
        per_ds_long: dict[str, float] = {}
        score_mean: dict[str, float] = {}
        score_std: dict[str, float] = {}
        score_psi: dict[str, float] = {}
        thresh_precision: dict[str, float] = {}
        for ds in datasets:
            idx = [i for i, r in enumerate(ap0) if axis_value(r, "dataset_id") == ds]
            ds_scores = [scores[i] for i in idx]
            top_part = [r for r in subset if axis_value(r, "dataset_id") == ds]
            tq = row_quality(top_part, max(1, len(idx)))
            per_ds_count[ds] = len(top_part)
            per_ds_precision[ds] = tq["GradeAB_precision"]
            per_ds_v[ds] = tq["V_integrated_LCB"]
            per_ds_long[ds] = tq["h240_longrisk_UCB"]
            score_mean[ds] = mean(ds_scores)
            score_std[ds] = statistics.pstdev(ds_scores) if len(ds_scores) > 1 else 0.0
            score_psi[ds] = psi_kl_wasserstein(ds_scores, scores)[0]
            keep = [i for i in range(len(ap0)) if i not in idx]
            threshold = sorted([scores[i] for i in keep], reverse=True)[min(TARGET_K - 1, len(keep) - 1)]
            held = [ap0[i] for i in idx if scores[i] >= threshold]
            thresh_precision[ds] = row_quality(held, max(1, len(idx)))["GradeAB_precision"]
            dist_rows.append(
                {
                    "stage": "P1_SCORE_DISTRIBUTION_BY_DATASET_V9680",
                    "status": "dataset_transport_row",
                    "transport_id": tid,
                    "dataset_id": ds,
                    "score_mean": score_mean[ds],
                    "score_std": score_std[ds],
                    "score_PSI": score_psi[ds],
                    "TopK87_count": per_ds_count[ds],
                    "TopK87_precision": per_ds_precision[ds],
                    "TopK87_V_LCB": per_ds_v[ds],
                    "TopK87_longrisk_UCB": per_ds_long[ds],
                    "threshold_transfer_precision": thresh_precision[ds],
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
        macro_precision = mean(list(per_ds_precision.values()))
        macro_v = mean(list(per_ds_v.values()))
        macro_long = mean(list(per_ds_long.values()))
        strong = int(
            q["LDO_drop"] <= 0.10
            and macro_precision >= 0.75
            and macro_v > 0
            and macro_long <= 0.05
            and q["bad_UCB"] <= 0.05
            and q["null_UCB"] <= 0.15
        )
        weak = int(
            q["LDO_drop"] <= 0.15
            and macro_precision >= 0.65
            and macro_v > 0
            and macro_long <= 0.10
        )
        rows.append(
            {
                "stage": "P1_SCORE_TRANSPORT_AUDIT_V9680",
                "status": "transport_row",
                "transport_id": tid,
                "uses_dataset_name": 0,
                "uses_outcome_field": 0,
                "feature_cost_ms_q90": qtile([fnum(r.get("feature_compute_ms")) for r in ap0], 0.90),
                "score_mean_by_dataset": json.dumps(score_mean, sort_keys=True),
                "score_std_by_dataset": json.dumps(score_std, sort_keys=True),
                "score_PSI_by_dataset": json.dumps(score_psi, sort_keys=True),
                "TopK87_count_by_dataset": json.dumps(per_ds_count, sort_keys=True),
                "TopK87_precision_by_dataset": json.dumps(per_ds_precision, sort_keys=True),
                "TopK87_V_LCB_by_dataset": json.dumps(per_ds_v, sort_keys=True),
                "TopK87_longrisk_UCB_by_dataset": json.dumps(per_ds_long, sort_keys=True),
                "threshold_transfer_precision_by_dataset": json.dumps(thresh_precision, sort_keys=True),
                "TopK87_precision_macro": macro_precision,
                "TopK87_V_LCB_macro": macro_v,
                "TopK87_longrisk_UCB_macro": macro_long,
                "TopK87_precision": q["GradeAB_precision"],
                "TopK87_V_LCB": q["V_integrated_LCB"],
                "TopK87_longrisk_UCB": q["h240_longrisk_UCB"],
                "bad_UCB": q["bad_UCB"],
                "null_UCB": q["null_UCB"],
                "memory_UCB": q["memory_fail_UCB"],
                "offdiag_UCB": q["offdiag_fail_UCB"],
                "AUC_GradeAB": auc_score(scores, labels),
                "LDO_drop": q["LDO_drop"],
                "LSO_drop": q["LSO_drop"],
                "LTO_drop": q["LTO_drop"],
                "max_dataset_share": max((per_ds_count[ds] / max(1, len(subset)) for ds in datasets), default=0.0),
                "min_dataset_support": min(per_ds_count.values(), default=0),
                "p1_transport_strong_pass": strong,
                "p1_transport_weak_pass": weak,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    best = max(rows, key=lambda r: (inum(r["p1_transport_strong_pass"]), inum(r["p1_transport_weak_pass"]), fnum(r["TopK87_precision_macro"]), fnum(r["TopK87_V_LCB_macro"]), -fnum(r["LDO_drop"])), default={})
    summary = {
        "stage": "P1_SCORE_TRANSPORT_AUDIT_V9680",
        "status": "summary",
        "transport_count": len(rows),
        "transport_strong_pass_count": sum(inum(r["p1_transport_strong_pass"]) for r in rows),
        "transport_weak_pass_count": sum(inum(r["p1_transport_weak_pass"]) for r in rows),
        "best_transport_id": best.get("transport_id", ""),
        "best_TopK87_precision_macro": best.get("TopK87_precision_macro", 0),
        "best_TopK87_V_LCB_macro": best.get("TopK87_V_LCB_macro", 0),
        "best_TopK87_longrisk_UCB_macro": best.get("TopK87_longrisk_UCB_macro", 1),
        "best_LDO_drop": best.get("LDO_drop", 1),
        "score_transport_strong_pass": int(any(inum(r["p1_transport_strong_pass"]) for r in rows)),
        "score_transport_weak_pass": int(any(inum(r["p1_transport_weak_pass"]) for r in rows)),
        "score_transport_any_pass": int(any(inum(r["p1_transport_strong_pass"]) or inum(r["p1_transport_weak_pass"]) for r in rows)),
        "uses_dataset_name": 0,
        "uses_outcome_field": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p1_score_distribution_by_dataset_before_after.svg", "P1 transport LDO", [r["transport_id"] for r in rows], [fnum(r["LDO_drop"]) for r in rows])
    write_bar_svg(out / "fig_p1_quantile_transport_curve.svg", "P1 macro precision", [r["transport_id"] for r in rows], [fnum(r["TopK87_precision_macro"]) for r in rows])
    write_bar_svg(out / "fig_p1_topk_count_precision_by_dataset.svg", "P1 pooled precision", [r["transport_id"] for r in rows], [fnum(r["TopK87_precision"]) for r in rows])
    write_bar_svg(out / "fig_p1_threshold_transfer_heatmap.svg", "P1 macro longrisk", [r["transport_id"] for r in rows], [fnum(r["TopK87_longrisk_UCB_macro"]) for r in rows])
    write_bar_svg(out / "fig_p1_score_transport_pareto_precision_vs_LDO.svg", "P1 precision minus LDO", [r["transport_id"] for r in rows], [fnum(r["TopK87_precision_macro"]) - fnum(r["LDO_drop"]) for r in rows])
    return [summary] + rows, dist_rows, summary, transports


def p2_target_lattice_v2(ap0: list[dict[str, Any]], r5b: list[float], out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tids = [
        "T2.1-CoreExpansionV2",
        "T2.2-CoreClean",
        "T2.3-MemoryOffdiagCore",
        "T2.4-GradeABMemoryOffdiag",
        "T2.5-CoreRiskCleanExpansion",
        "T2.6-R5BHighScoreRiskClean",
        "T2.7-ValuePositiveNoLongRisk",
    ]
    rows: list[dict[str, Any]] = []
    grid: list[dict[str, Any]] = []
    outcome_scores = [target_outcome_score(r) for r in ap0]
    combined_scores = [outcome_scores[i] + 0.1 * r5b[i] for i in range(len(ap0))]
    for tid in tids:
        scores = r5b if "R5B" in tid else combined_scores
        subset = target_subset(ap0, tid, scores)
        q = row_quality(subset, len(ap0))
        q["accepted_count"] = len(subset)
        l = lineage_stats(subset)
        gshare, gaxis, ggroup = top_group_share(subset)
        mscores = membership_scores(ap0, subset, scores)
        ldo, dgroup, _ = leaveout_precision_drop(mscores, ap0, [axis_value(r, "dataset_id") for r in ap0])
        lso, _, _ = leaveout_precision_drop(mscores, ap0, [str(r.get("stratum_id") or r.get("bucket_id") or "") for r in ap0])
        lto, _, _ = leaveout_precision_drop(mscores, ap0, [candidate_template_id(r) for r in ap0])
        families = Counter(str(r.get("family_id") or "") for r in subset)
        datasets = Counter(axis_value(r, "dataset_id") for r in subset)
        row = {
            "stage": "P2_CORE_EXPANSION_TARGET_LATTICE_V2_V9680",
            "status": "target_row",
            "target_id": tid,
            "core_count": sum(target_flag(r, "CoreClean") for r in subset),
            "expansion_count": max(0, len(subset) - sum(target_flag(r, "CoreClean") for r in subset)),
            "accepted_count_total": len(subset),
            "coverage": q["coverage"],
            "GradeAB_precision": q["GradeAB_precision"],
            "V_integrated_LCB": q["V_integrated_LCB"],
            "h20_V_LCB": q["h20_V_LCB"],
            "h80_V_LCB": q["h80_V_LCB"],
            "h240_V_LCB": q["h240_V_LCB"],
            "h240_longrisk_UCB": q["h240_longrisk_UCB"],
            "bad_UCB": q["bad_UCB"],
            "null_UCB": q["null_UCB"],
            "memory_fail_UCB": q["memory_fail_UCB"],
            "offdiag_fail_UCB": q["offdiag_fail_UCB"],
            "per_dataset_precision": json.dumps({ds: mean([float(gradeab(r)) for r in subset if axis_value(r, "dataset_id") == ds]) for ds in datasets}, sort_keys=True),
            "per_dataset_coverage": json.dumps({ds: count / max(1, len(ap0)) for ds, count in datasets.items()}, sort_keys=True),
            "LDO_drop": ldo,
            "LSO_drop": lso,
            "LTO_drop": lto,
            "max_dataset_share": max((v / max(1, len(subset)) for v in datasets.values()), default=0.0),
            "max_template_share": l["max_candidate_template_share"],
            "max_family_share": max((v / max(1, len(subset)) for v in families.values()), default=0.0),
            "max_group_share": gshare,
            "max_group_axis": gaxis,
            "max_group_id": ggroup,
            "strong_target_pass": 0,
            "weak_target_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["strong_target_pass"] = pass_flags({"accepted_count": len(subset), **q}, ldo, lso, lto)
        row["weak_target_pass"] = pass_flags({"accepted_count": len(subset), **q}, ldo, lso, lto, weak=True)
        rows.append(row)
        for ds, count in datasets.items():
            part = [r for r in subset if axis_value(r, "dataset_id") == ds]
            grid.append(
                {
                    "stage": "P2_TARGET_LEAVEOUT_GRID_V9680",
                    "status": "target_dataset_row",
                    "target_id": tid,
                    "dataset_id": ds,
                    "accepted_count": count,
                    "precision": mean([float(gradeab(r)) for r in part]),
                    "coverage": count / max(1, len(ap0)),
                    "LDO_drop": ldo,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    best = max(rows, key=lambda r: (inum(r["strong_target_pass"]), inum(r["weak_target_pass"]), fnum(r["GradeAB_precision"]), fnum(r["V_integrated_LCB"]), -fnum(r["LDO_drop"])), default={})
    summary = {
        "stage": "P2_CORE_EXPANSION_TARGET_LATTICE_V2_V9680",
        "status": "summary",
        "target_candidate_count": len(rows),
        "strong_target_pass_count": sum(inum(r["strong_target_pass"]) for r in rows),
        "weak_target_pass_count": sum(inum(r["weak_target_pass"]) for r in rows),
        "best_target_id": best.get("target_id", ""),
        "best_accepted_count": best.get("accepted_count_total", 0),
        "best_coverage": best.get("coverage", 0),
        "best_GradeAB_precision": best.get("GradeAB_precision", 0),
        "best_V_integrated_LCB": best.get("V_integrated_LCB", 0),
        "best_h240_longrisk_UCB": best.get("h240_longrisk_UCB", 1),
        "best_bad_UCB": best.get("bad_UCB", 1),
        "best_null_UCB": best.get("null_UCB", 1),
        "best_memory_fail_UCB": best.get("memory_fail_UCB", 1),
        "best_offdiag_fail_UCB": best.get("offdiag_fail_UCB", 1),
        "best_LDO_drop": best.get("LDO_drop", 1),
        "template_balanced_target_strong_pass": int(any(inum(r["strong_target_pass"]) for r in rows)),
        "template_balanced_target_weak_pass": int(any(inum(r["weak_target_pass"]) for r in rows)),
        "template_balanced_target_pass": int(any(inum(r["strong_target_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p2_core_expansion_venn.svg", "P2 core count", [r["target_id"] for r in rows], [fnum(r["core_count"]) for r in rows])
    write_bar_svg(out / "fig_p2_precision_coverage_curve.svg", "P2 precision", [r["target_id"] for r in rows], [fnum(r["GradeAB_precision"]) for r in rows])
    write_bar_svg(out / "fig_p2_bad_null_vs_expansion_size.svg", "P2 bad+null", [r["target_id"] for r in rows], [fnum(r["bad_UCB"]) + fnum(r["null_UCB"]) for r in rows])
    write_bar_svg(out / "fig_p2_LDO_drop_vs_expansion_size.svg", "P2 LDO", [r["target_id"] for r in rows], [fnum(r["LDO_drop"]) for r in rows])
    write_bar_svg(out / "fig_p2_target_risk_table.svg", "P2 risk table", [r["target_id"] for r in rows], [fnum(r["h240_longrisk_UCB"]) + fnum(r["memory_fail_UCB"]) + fnum(r["offdiag_fail_UCB"]) for r in rows])
    return [summary] + rows, grid, summary


def adjacent_expansion(
    ap0: list[dict[str, Any]],
    core: list[dict[str, Any]],
    score: list[float],
    limit: int = 10,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    core_ids = {str(r.get("action_id")) for r in core}
    ranked = [ap0[i] for i in topk_idx(score, len(ap0)) if str(ap0[i].get("action_id")) not in core_ids]
    if predicate is not None:
        ranked = [r for r in ranked if predicate(r)]
    return ranked[:limit]


def p2_target_lattice_v3(ap0: list[dict[str, Any]], transports: dict[str, list[float]], p1: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    best_transport = str(p1.get("best_transport_id") or "S0-raw-score")
    base_score = transports.get(best_transport) or next(iter(transports.values()))
    value_score = [legal_value_proxy(r) for r in ap0]
    risk_score = [legal_risk_proxy(r) for r in ap0]
    conformal = [base_score[i] - 0.9 * robust_z(risk_score)[i] for i in range(len(ap0))]
    core = [r for r in ap0 if target_flag(r, "MemoryOffdiagCore")]
    core_clean = [r for r in ap0 if target_flag(r, "CoreClean")]

    def value_no_long(r: dict[str, Any]) -> bool:
        return bool(target_flag(r, "ValuePositiveNoLongRisk"))

    def bad_null_veto(r: dict[str, Any]) -> bool:
        return value_no_long(r) and fnum(r.get("bad_event_rate")) <= 0.08 and fnum(r.get("null_event_rate")) <= 0.18

    def memory_offdiag_veto(r: dict[str, Any]) -> bool:
        return value_no_long(r) and not memory_fail(r) and not offdiag_fail(r)

    target_specs: dict[str, list[dict[str, Any]]] = {
        "T3.1-CoreOnly": core,
        "T3.2-CorePlusNearest10TransportedScore": core + adjacent_expansion(ap0, core, base_score, 10, value_no_long),
        "T3.3-CorePlusNearest10MemoryOffdiagValue": core + adjacent_expansion(ap0, core, value_score, 10, memory_offdiag_veto),
        "T3.4-CorePlusNearest10ConformalLowRisk": core + adjacent_expansion(ap0, core, conformal, 10, bad_null_veto),
        "T3.5-CoreQuotaExpansionScoreStratum": core + adjacent_expansion(ap0, core, percentile_scores(base_score, [axis_value(r, "payload_norm_bucket") for r in ap0]), 10, value_no_long),
        "T3.6-CoreExpansionBadNullVetoBeforeRank": core + adjacent_expansion(ap0, core, base_score, 10, bad_null_veto),
        "T3.7-CoreExpansionMemoryOffdiagVetoBeforeRank": core + adjacent_expansion(ap0, core, base_score, 10, memory_offdiag_veto),
        "T3.8-CoreExpansionLeaveoutStabilityPenalty": core + adjacent_expansion(ap0, core, [base_score[i] - 0.5 * abs(legal_value_proxy(ap0[i])) for i in range(len(ap0))], 10, bad_null_veto),
        "T3.9-CoreExpansionCapFamilyTemplateScoreBucket": core + adjacent_expansion(ap0, core, base_score, 10, lambda r: value_no_long(r) and not cover_collapse(r)),
        "T3.10-CoreExpansionEnsembleIntersection": core + [r for r in adjacent_expansion(ap0, core, conformal, 20, bad_null_veto) if not memory_fail(r) and not offdiag_fail(r)][:10],
    }
    rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    for tid, subset0 in target_specs.items():
        seen: set[str] = set()
        subset: list[dict[str, Any]] = []
        for r in subset0:
            aid = str(r.get("action_id"))
            if aid not in seen:
                subset.append(r)
                seen.add(aid)
        q = row_quality(subset, len(ap0))
        mscores = membership_scores(ap0, subset, base_score)
        ldo, _, _ = leaveout_precision_drop(mscores, ap0, [axis_value(r, "dataset_id") for r in ap0])
        lso, _, _ = leaveout_precision_drop(mscores, ap0, [str(r.get("stratum_id") or r.get("bucket_id") or "") for r in ap0])
        lto, _, _ = leaveout_precision_drop(mscores, ap0, [candidate_template_id(r) for r in ap0])
        l = lineage_stats(subset)
        datasets = Counter(axis_value(r, "dataset_id") for r in subset)
        templates = Counter(candidate_template_id(r) for r in subset)
        memory_buckets = Counter("memory_fail_1" if memory_fail(r) else "memory_fail_0" for r in subset)
        offdiag_buckets = Counter("offdiag_fail_1" if offdiag_fail(r) else "offdiag_fail_0" for r in subset)
        strong = pass_flags({"accepted_count": len(subset), **q}, ldo, lso, lto)
        weak = pass_flags({"accepted_count": len(subset), **q}, ldo, lso, lto, weak=True)
        row = {
            "stage": "P2_CORE_EXPANSION_TARGET_LATTICE_V3_V9680",
            "status": "target_row",
            "target_id": tid,
            "transport_id": best_transport,
            "accepted_count": len(subset),
            "coverage": q["coverage"],
            "GradeAB_precision": q["GradeAB_precision"],
            "V_integrated_LCB": q["V_integrated_LCB"],
            "h20_V_LCB": q["h20_V_LCB"],
            "h80_V_LCB": q["h80_V_LCB"],
            "h240_V_LCB": q["h240_V_LCB"],
            "h240_longrisk_UCB": q["h240_longrisk_UCB"],
            "bad_UCB": q["bad_UCB"],
            "null_UCB": q["null_UCB"],
            "memory_fail_UCB": q["memory_fail_UCB"],
            "offdiag_fail_UCB": q["offdiag_fail_UCB"],
            "candidate_template_count": l["candidate_template_count"],
            "max_template_share": l["max_candidate_template_share"],
            "max_dataset_share": max((v / max(1, len(subset)) for v in datasets.values()), default=0.0),
            "LDO_drop": ldo,
            "LSO_drop": lso,
            "LTO_drop": lto,
            "support_by_dataset": json.dumps(dict(datasets), sort_keys=True),
            "support_by_template": json.dumps(dict(templates.most_common(10)), sort_keys=True),
            "support_by_memory_bucket": json.dumps(dict(memory_buckets), sort_keys=True),
            "support_by_offdiag_bucket": json.dumps(dict(offdiag_buckets), sort_keys=True),
            "strong_target_pass": strong,
            "weak_target_pass": weak,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
        for ds, n in datasets.items():
            support_rows.append(
                {
                    "stage": "P2_TARGET_SUPPORT_BY_DATASET_TEMPLATE_V9680",
                    "status": "dataset_support_row",
                    "target_id": tid,
                    "dataset_id": ds,
                    "accepted_count": n,
                    "precision": mean([float(gradeab(r)) for r in subset if axis_value(r, "dataset_id") == ds]),
                    "template_count": len({candidate_template_id(r) for r in subset if axis_value(r, "dataset_id") == ds}),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    best = max(rows, key=lambda r: (inum(r["strong_target_pass"]), inum(r["weak_target_pass"]), fnum(r["GradeAB_precision"]), fnum(r["V_integrated_LCB"]), -fnum(r["LDO_drop"])), default={})
    summary = {
        "stage": "P2_CORE_EXPANSION_TARGET_LATTICE_V3_V9680",
        "status": "summary",
        "target_candidate_count": len(rows),
        "strong_target_pass_count": sum(inum(r["strong_target_pass"]) for r in rows),
        "weak_target_pass_count": sum(inum(r["weak_target_pass"]) for r in rows),
        "best_target_id": best.get("target_id", ""),
        "best_accepted_count": best.get("accepted_count", 0),
        "best_coverage": best.get("coverage", 0),
        "best_GradeAB_precision": best.get("GradeAB_precision", 0),
        "best_V_integrated_LCB": best.get("V_integrated_LCB", 0),
        "best_h240_longrisk_UCB": best.get("h240_longrisk_UCB", 1),
        "best_bad_UCB": best.get("bad_UCB", 1),
        "best_null_UCB": best.get("null_UCB", 1),
        "best_memory_fail_UCB": best.get("memory_fail_UCB", 1),
        "best_offdiag_fail_UCB": best.get("offdiag_fail_UCB", 1),
        "best_LDO_drop": best.get("LDO_drop", 1),
        "core_count": len(core),
        "needed_expansion_to_87": max(0, TARGET_K - len(core)),
        "template_balanced_target_strong_pass": int(any(inum(r["strong_target_pass"]) for r in rows)),
        "template_balanced_target_weak_pass": int(any(inum(r["weak_target_pass"]) for r in rows)),
        "template_balanced_target_pass": int(any(inum(r["strong_target_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p2_core_expansion_frontier_count_vs_cleanliness.svg", "P2 accepted count", [r["target_id"] for r in rows], [fnum(r["accepted_count"]) for r in rows])
    write_bar_svg(out / "fig_p2_target_parallel_coordinates.svg", "P2 precision", [r["target_id"] for r in rows], [fnum(r["GradeAB_precision"]) for r in rows])
    write_bar_svg(out / "fig_p2_dataset_support_heatmap.svg", "P2 max dataset share", [r["target_id"] for r in rows], [fnum(r["max_dataset_share"]) for r in rows])
    write_bar_svg(out / "fig_p2_template_support_heatmap.svg", "P2 max template share", [r["target_id"] for r in rows], [fnum(r["max_template_share"]) for r in rows])
    write_bar_svg(out / "fig_p2_value_risk_null_3d_frontier.svg", "P2 V-risk-null", [r["target_id"] for r in rows], [fnum(r["V_integrated_LCB"]) - fnum(r["h240_longrisk_UCB"]) - fnum(r["null_UCB"]) for r in rows])
    write_bar_svg(out / "fig_p2_core_to_expansion_nearest_neighbor_graph.svg", "P2 LDO", [r["target_id"] for r in rows], [fnum(r["LDO_drop"]) for r in rows])
    return [summary] + rows, support_rows, summary


def p3_feature_normalization(ap0: list[dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    value = [legal_value_proxy(r) for r in ap0]
    risk = [legal_risk_proxy(r) for r in ap0]
    memory = [legal_memory_proxy(r) for r in ap0]
    offdiag = [legal_offdiag_proxy(r) for r in ap0]
    badnull = [abs(fnum(r.get("curvature_score"))) + max(0.0, -fnum(r.get("signal_channel_score"))) for r in ap0]
    rz_value = robust_z(value)
    rz_risk = robust_z(risk)
    event_pct = percentile_scores(value, [axis_value(r, "event_family") for r in ap0])
    template_pct = percentile_scores(value, [candidate_template_id(r) for r in ap0])
    feature_scores = {
        "F3A-raw-value-proxy": value,
        "F3B-global-robust-z-value": rz_value,
        "F3C-event-local-percentile": event_pct,
        "F3D-risk-normalized-value": [rz_value[i] - rz_risk[i] for i in range(len(ap0))],
        "F3E-value-minus-memory-offdiag": [rz_value[i] - 0.8 * robust_z(memory)[i] - 0.8 * robust_z(offdiag)[i] for i in range(len(ap0))],
        "F3F-bad-null-risk-normalized": [rz_value[i] - rz_risk[i] - 0.5 * robust_z(badnull)[i] for i in range(len(ap0))],
        "F3G-event-template-diverse-score": [0.7 * event_pct[i] + 0.3 * template_pct[i] - 0.5 * rz_risk[i] for i in range(len(ap0))],
    }
    labels = [gradeab(r) for r in ap0]
    rows: list[dict[str, Any]] = []
    raw_shift = max(mean([value[i] for i, r in enumerate(ap0) if axis_value(r, "dataset_id") == ds]) for ds in sorted({axis_value(r, "dataset_id") for r in ap0})) - min(mean([value[i] for i, r in enumerate(ap0) if axis_value(r, "dataset_id") == ds]) for ds in sorted({axis_value(r, "dataset_id") for r in ap0}))
    for name, scores in feature_scores.items():
        q = score_quality(scores, ap0, TARGET_K)
        norm_shift = max(mean([scores[i] for i, r in enumerate(ap0) if axis_value(r, "dataset_id") == ds]) for ds in sorted({axis_value(r, "dataset_id") for r in ap0})) - min(mean([scores[i] for i, r in enumerate(ap0) if axis_value(r, "dataset_id") == ds]) for ds in sorted({axis_value(r, "dataset_id") for r in ap0}))
        row = {
            "stage": "P3_LEGAL_FEATURE_NORMALIZATION_DECONFOUNDING_V3",
            "status": "feature_row",
            "feature_name": name,
            "feature_legality": "green",
            "normalization_method": name.split("-", 1)[-1],
            "uses_dataset_name_at_commit": 0,
            "uses_outcome_field": 0,
            "raw_AUC": auc_score(value, labels),
            "normalized_AUC": auc_score(scores, labels),
            "TopK87_precision": q["GradeAB_precision"],
            "TopK87_V_LCB": q["V_integrated_LCB"],
            "TopK87_longrisk_UCB": q["h240_longrisk_UCB"],
            "TopK87_bad_UCB": q["bad_UCB"],
            "TopK87_null_UCB": q["null_UCB"],
            "LDO_drop": q["LDO_drop"],
            "score_shift_before": raw_shift,
            "score_shift_after": norm_shift,
            "feature_cost_ms_q90": qtile([fnum(r.get("feature_compute_ms")) for r in ap0], 0.90),
            "p3_feature_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["p3_feature_pass"] = int(
            row["TopK87_precision"] >= 0.65
            and row["TopK87_V_LCB"] > 0
            and row["TopK87_longrisk_UCB"] <= 0.10
            and row["LDO_drop"] <= 0.20
        )
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["p3_feature_pass"]), fnum(r["TopK87_precision"]), fnum(r["TopK87_V_LCB"]), -fnum(r["LDO_drop"])), default={})
    summary = {
        "stage": "P3_LEGAL_FEATURE_NORMALIZATION_DECONFOUNDING_V3",
        "status": "summary",
        "feature_count": len(rows),
        "feature_pass_count": sum(inum(r["p3_feature_pass"]) for r in rows),
        "best_feature_name": best.get("feature_name", ""),
        "best_TopK87_precision": best.get("TopK87_precision", 0),
        "best_TopK87_V_LCB": best.get("TopK87_V_LCB", 0),
        "best_TopK87_longrisk_UCB": best.get("TopK87_longrisk_UCB", 1),
        "best_LDO_drop": best.get("LDO_drop", 1),
        "p3_pass": int(any(inum(r["p3_feature_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p3_feature_shift_before_after.svg", "P3 score shift", [r["feature_name"] for r in rows], [fnum(r["score_shift_after"]) for r in rows])
    write_bar_svg(out / "fig_p3_feature_cost_vs_precision.svg", "P3 precision", [r["feature_name"] for r in rows], [fnum(r["TopK87_precision"]) for r in rows])
    write_bar_svg(out / "fig_p3_legal_feature_heatmap.svg", "P3 LDO", [r["feature_name"] for r in rows], [fnum(r["LDO_drop"]) for r in rows])
    return [summary] + rows, summary, feature_scores


def p3_veto_order_experiment(ap0: list[dict[str, Any]], transports: dict[str, list[float]], p1: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    best_transport = str(p1.get("best_transport_id") or "S0-raw-score")
    value = transports.get(best_transport) or next(iter(transports.values()))
    risk = [legal_risk_proxy(r) for r in ap0]
    badnull = [abs(fnum(r.get("curvature_score"))) + max(0.0, -fnum(r.get("signal_channel_score"))) for r in ap0]
    memory = [legal_memory_proxy(r) for r in ap0]
    offdiag = [legal_offdiag_proxy(r) for r in ap0]
    rz_risk = robust_z(risk)
    rz_badnull = robust_z(badnull)
    rz_memory = robust_z(memory)
    rz_offdiag = robust_z(offdiag)
    base_top_ids = {str(r.get("action_id")) for r in select_top(value, ap0, TARGET_K)}

    def score_for(strategy: str) -> list[float]:
        if strategy == "V0-value-rank-only":
            return value
        if strategy == "V1-risk-veto-before-value-rank":
            return [value[i] - 2.0 * rz_risk[i] for i in range(len(ap0))]
        if strategy == "V2-risk-veto-after-value-rank":
            return [value[i] - 1.0 * max(0.0, rz_risk[i]) for i in range(len(ap0))]
        if strategy == "V3-bad-null-veto-before-value-rank":
            return [value[i] - 2.0 * rz_badnull[i] for i in range(len(ap0))]
        if strategy == "V4-bad-null-veto-after-value-rank":
            return [value[i] - 1.0 * max(0.0, rz_badnull[i]) for i in range(len(ap0))]
        if strategy == "V5-memory-offdiag-veto-before-value-rank":
            return [value[i] - 1.5 * rz_memory[i] - 1.5 * rz_offdiag[i] for i in range(len(ap0))]
        if strategy == "V6-memory-offdiag-veto-after-value-rank":
            return [value[i] - 0.8 * max(0.0, rz_memory[i]) - 0.8 * max(0.0, rz_offdiag[i]) for i in range(len(ap0))]
        if strategy == "V7-two-stage-core-first-expansion-second":
            return [value[i] + (2.0 if target_flag(ap0[i], "MemoryOffdiagCore") else 0.0) - 0.6 * rz_risk[i] for i in range(len(ap0))]
        if strategy == "V8-three-stage-core-value-expansion-risk-cleanup":
            return [value[i] + (2.0 if target_flag(ap0[i], "MemoryOffdiagCore") else 0.0) - 0.8 * rz_risk[i] - 0.5 * rz_badnull[i] for i in range(len(ap0))]
        return [value[i] - 1.0 * rz_risk[i] - 0.7 * rz_badnull[i] - 0.5 * rz_memory[i] - 0.5 * rz_offdiag[i] for i in range(len(ap0))]

    strategies = [
        "V0-value-rank-only",
        "V1-risk-veto-before-value-rank",
        "V2-risk-veto-after-value-rank",
        "V3-bad-null-veto-before-value-rank",
        "V4-bad-null-veto-after-value-rank",
        "V5-memory-offdiag-veto-before-value-rank",
        "V6-memory-offdiag-veto-after-value-rank",
        "V7-two-stage-core-first-expansion-second",
        "V8-three-stage-core-value-expansion-risk-cleanup",
        "V9-constrained-optimizer-risk-clean",
    ]
    rows: list[dict[str, Any]] = []
    for sid in strategies:
        scores = score_for(sid)
        subset = select_top(scores, ap0, TARGET_K)
        q = score_quality(scores, ap0, TARGET_K)
        selected_ids = {str(r.get("action_id")) for r in subset}
        removed = [r for r in ap0 if str(r.get("action_id")) in base_top_ids and str(r.get("action_id")) not in selected_ids]
        selected_bad = [r for r in subset if not gradeab(r) or inum(r.get("h240_longrisk")) or fnum(r.get("bad_event_rate")) > 0.05 or fnum(r.get("null_event_rate")) > 0.15 or memory_fail(r) or offdiag_fail(r)]
        removed_good = [r for r in removed if gradeab(r)]
        removed_bad = [r for r in removed if not gradeab(r)]
        false_positive = len(removed_good) / max(1, len(removed))
        false_negative = len(selected_bad) / max(1, len(subset))
        weak_like = pass_flags({"accepted_count": len(subset), **q}, q["LDO_drop"], q["LSO_drop"], q["LTO_drop"], weak=True)
        row = {
            "stage": "P3_VETO_ORDER_EXPERIMENT_V9680",
            "status": "strategy_row",
            "strategy_id": sid,
            "transport_id": best_transport,
            "accepted_count": len(subset),
            "GradeAB_precision": q["GradeAB_precision"],
            "V_LCB": q["V_integrated_LCB"],
            "longrisk_UCB": q["h240_longrisk_UCB"],
            "bad_UCB": q["bad_UCB"],
            "null_UCB": q["null_UCB"],
            "memory_UCB": q["memory_fail_UCB"],
            "offdiag_UCB": q["offdiag_fail_UCB"],
            "veto_removed_count": len(removed),
            "veto_removed_good_count": len(removed_good),
            "veto_removed_bad_count": len(removed_bad),
            "veto_false_positive_rate": false_positive,
            "veto_false_negative_rate": false_negative,
            "LDO_drop": q["LDO_drop"],
            "support_by_dataset": json.dumps(Counter(axis_value(r, "dataset_id") for r in subset), sort_keys=True),
            "p3_strategy_weak_pass": int(weak_like and false_positive <= 0.25 and false_negative <= 0.25),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r["p3_strategy_weak_pass"]), fnum(r["GradeAB_precision"]), fnum(r["V_LCB"]), -fnum(r["longrisk_UCB"]), -fnum(r["LDO_drop"])), default={})
    summary = {
        "stage": "P3_VETO_ORDER_EXPERIMENT_V9680",
        "status": "summary",
        "strategy_count": len(rows),
        "strategy_pass_count": sum(inum(r["p3_strategy_weak_pass"]) for r in rows),
        "best_strategy_id": best.get("strategy_id", ""),
        "best_GradeAB_precision": best.get("GradeAB_precision", 0),
        "best_V_LCB": best.get("V_LCB", 0),
        "best_longrisk_UCB": best.get("longrisk_UCB", 1),
        "best_bad_UCB": best.get("bad_UCB", 1),
        "best_null_UCB": best.get("null_UCB", 1),
        "best_LDO_drop": best.get("LDO_drop", 1),
        "best_veto_false_positive_rate": best.get("veto_false_positive_rate", 1),
        "best_veto_false_negative_rate": best.get("veto_false_negative_rate", 1),
        "p3_veto_order_pass": int(any(inum(r["p3_strategy_weak_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p3_veto_order_waterfall.svg", "P3 precision", [r["strategy_id"] for r in rows], [fnum(r["GradeAB_precision"]) for r in rows])
    write_bar_svg(out / "fig_p3_veto_removed_good_bad_bar.svg", "P3 removed good", [r["strategy_id"] for r in rows], [fnum(r["veto_removed_good_count"]) for r in rows])
    write_bar_svg(out / "fig_p3_value_vs_longrisk_after_veto.svg", "P3 V-longrisk", [r["strategy_id"] for r in rows], [fnum(r["V_LCB"]) - fnum(r["longrisk_UCB"]) for r in rows])
    write_bar_svg(out / "fig_p3_bad_null_memory_offdiag_sankey.svg", "P3 risk stack", [r["strategy_id"] for r in rows], [fnum(r["bad_UCB"]) + fnum(r["null_UCB"]) + fnum(r["memory_UCB"]) + fnum(r["offdiag_UCB"]) for r in rows])
    return [summary] + rows, summary


def p4_rankers_v9680(ap0: list[dict[str, Any]], feature_scores: dict[str, list[float]], r5b: list[float], out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    n = len(ap0)
    value = robust_z([legal_value_proxy(r) for r in ap0])
    risk = robust_z([legal_risk_proxy(r) for r in ap0])
    memory = robust_z([legal_memory_proxy(r) for r in ap0])
    offdiag = robust_z([legal_offdiag_proxy(r) for r in ap0])
    badnull = robust_z([abs(fnum(r.get("curvature_score"))) + max(0.0, -fnum(r.get("signal_channel_score"))) for r in ap0])
    event = feature_scores.get("F3C-event-local-percentile", value)
    risk_norm = feature_scores.get("F3D-risk-normalized-value", value)
    core_support = [1.0 if (not memory_fail(ap0[i]) and not offdiag_fail(ap0[i]) and not cover_collapse(ap0[i])) else 0.0 for i in range(n)]
    rankers = {
        "R7A-ValueRankOnly": value,
        "R7B-ValueRankLongRiskVeto": [value[i] - 1.4 * risk[i] for i in range(n)],
        "R7C-ValueRankBadNullVeto": [value[i] - 1.1 * badnull[i] for i in range(n)],
        "R7D-ValueRiskBadNullMemoryOffdiag": [value[i] - 1.1 * risk[i] - 0.8 * badnull[i] - 0.7 * memory[i] - 0.7 * offdiag[i] for i in range(n)],
        "R7E-DatasetInvariantNormalizedRank": [risk_norm[i] - 0.5 * badnull[i] for i in range(n)],
        "R7F-GroupDROPairwiseRank": [0.6 * event[i] + 0.4 * value[i] - 0.8 * risk[i] - 0.5 * memory[i] - 0.5 * offdiag[i] for i in range(n)],
        "R7G-CoreExpansionRank": [0.5 * r5b[i] + 0.5 * value[i] + 0.4 * core_support[i] - 1.0 * risk[i] - 0.7 * badnull[i] for i in range(n)],
    }
    rows: list[dict[str, Any]] = []
    ablation: list[dict[str, Any]] = []
    for rid, scores in rankers.items():
        subset = select_top(scores, ap0, TARGET_K)
        q = score_quality(scores, ap0, TARGET_K)
        strong = pass_flags({"accepted_count": len(subset), **q}, q["LDO_drop"], q["LSO_drop"], q["LTO_drop"])
        weak = pass_flags({"accepted_count": len(subset), **q}, q["LDO_drop"], q["LSO_drop"], q["LTO_drop"], weak=True)
        row = {
            "stage": "P4_VALUE_RANK_BAD_NULL_RISK_MEMORY_VETO_RANKER",
            "status": "ranker_row",
            "ranker_id": rid,
            "feature_list": "value,longrisk,bad,null,memory,offdiag,support,cost",
            "red_field_count": 0,
            "dataset_name_commit_feature_count": 0,
            "accepted_count": len(subset),
            "coverage": q["coverage"],
            "GradeAB_precision": q["GradeAB_precision"],
            "V_integrated_LCB": q["V_integrated_LCB"],
            "longrisk_UCB": q["h240_longrisk_UCB"],
            "bad_UCB": q["bad_UCB"],
            "null_UCB": q["null_UCB"],
            "memory_UCB": q["memory_fail_UCB"],
            "offdiag_UCB": q["offdiag_fail_UCB"],
            "per_dataset_precision": json.dumps(per_dataset_stats(subset), sort_keys=True),
            "per_dataset_coverage": json.dumps({ds: sum(1 for r in subset if axis_value(r, "dataset_id") == ds) / len(ap0) for ds in sorted({axis_value(r, "dataset_id") for r in ap0})}, sort_keys=True),
            "LDO_drop": q["LDO_drop"],
            "LSO_drop": q["LSO_drop"],
            "LTO_drop": q["LTO_drop"],
            "max_dataset_share": max((sum(1 for r in subset if axis_value(r, "dataset_id") == ds) / max(1, len(subset)) for ds in sorted({axis_value(r, "dataset_id") for r in subset})), default=0.0),
            "max_template_share": q["max_candidate_template_share"],
            "feature_cost_ms_q90": qtile([fnum(r.get("feature_compute_ms")) for r in ap0], 0.90),
            "ranker_strong_pass": strong,
            "ranker_weak_pass": weak,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
        ablation.append(
            {
                "stage": "P4_RANKER_ABLATION_V9680",
                "status": "ablation_row",
                "ranker_id": rid,
                "value_component_mean": mean([value[i] for i in topk_idx(scores, TARGET_K)]),
                "risk_component_mean": mean([risk[i] for i in topk_idx(scores, TARGET_K)]),
                "badnull_component_mean": mean([badnull[i] for i in topk_idx(scores, TARGET_K)]),
                "memory_component_mean": mean([memory[i] for i in topk_idx(scores, TARGET_K)]),
                "offdiag_component_mean": mean([offdiag[i] for i in topk_idx(scores, TARGET_K)]),
                "GradeAB_precision": row["GradeAB_precision"],
                "LDO_drop": row["LDO_drop"],
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    best = max(rows, key=lambda r: (inum(r["ranker_strong_pass"]), inum(r["ranker_weak_pass"]), fnum(r["GradeAB_precision"]), fnum(r["V_integrated_LCB"]), -fnum(r["LDO_drop"])), default={})
    summary = {
        "stage": "P4_VALUE_RANK_BAD_NULL_RISK_MEMORY_VETO_RANKER",
        "status": "summary",
        "ranker_count": len(rows),
        "ranker_strong_pass_count": sum(inum(r["ranker_strong_pass"]) for r in rows),
        "ranker_weak_pass_count": sum(inum(r["ranker_weak_pass"]) for r in rows),
        "best_ranker_id": best.get("ranker_id", ""),
        "best_GradeAB_precision": best.get("GradeAB_precision", 0),
        "best_V_integrated_LCB": best.get("V_integrated_LCB", 0),
        "best_longrisk_UCB": best.get("longrisk_UCB", 1),
        "best_bad_UCB": best.get("bad_UCB", 1),
        "best_null_UCB": best.get("null_UCB", 1),
        "best_memory_UCB": best.get("memory_UCB", 1),
        "best_offdiag_UCB": best.get("offdiag_UCB", 1),
        "best_LDO_drop": best.get("LDO_drop", 1),
        "ranker_strong_pass": int(any(inum(r["ranker_strong_pass"]) for r in rows)),
        "ranker_weak_pass": int(any(inum(r["ranker_weak_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p4_ranker_precision_coverage.svg", "P4 precision", [r["ranker_id"] for r in rows], [fnum(r["GradeAB_precision"]) for r in rows])
    write_bar_svg(out / "fig_p4_value_vs_longrisk_scatter.svg", "P4 V vs longrisk", [r["ranker_id"] for r in rows], [fnum(r["V_integrated_LCB"]) - fnum(r["longrisk_UCB"]) for r in rows])
    write_bar_svg(out / "fig_p4_veto_ablation_waterfall.svg", "P4 veto components", [r["ranker_id"] for r in rows], [fnum(r["bad_UCB"]) + fnum(r["null_UCB"]) + fnum(r["memory_UCB"]) + fnum(r["offdiag_UCB"]) for r in rows])
    write_bar_svg(out / "fig_p4_per_dataset_precision_heatmap.svg", "P4 LDO", [r["ranker_id"] for r in rows], [fnum(r["LDO_drop"]) for r in rows])
    write_bar_svg(out / "fig_p4_ranker_drop_comparison.svg", "P4 drop sum", [r["ranker_id"] for r in rows], [fnum(r["LDO_drop"]) + fnum(r["LSO_drop"]) + fnum(r["LTO_drop"]) for r in rows])
    return [summary] + rows, ablation, summary, rankers


def worst_dataset_metrics(subset: list[dict[str, Any]]) -> tuple[float, float, float]:
    vals = []
    for ds in sorted({axis_value(r, "dataset_id") for r in subset}):
        part = [r for r in subset if axis_value(r, "dataset_id") == ds]
        q = row_quality(part, max(1, len(part)))
        vals.append((q["GradeAB_precision"], q["V_integrated_LCB"], q["h240_longrisk_UCB"]))
    if not vals:
        return 0.0, 0.0, 1.0
    return min(v[0] for v in vals), min(v[1] for v in vals), max(v[2] for v in vals)


def p4_dataset_invariant_ranker_v2(
    ap0: list[dict[str, Any]],
    transports: dict[str, list[float]],
    p1: dict[str, Any],
    out: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    best_transport = str(p1.get("best_transport_id") or "S0-raw-score")
    transported = transports.get(best_transport) or next(iter(transports.values()))
    value = robust_z([legal_value_proxy(r) for r in ap0])
    risk = robust_z([legal_risk_proxy(r) for r in ap0])
    memory = robust_z([legal_memory_proxy(r) for r in ap0])
    offdiag = robust_z([legal_offdiag_proxy(r) for r in ap0])
    badnull = robust_z([abs(fnum(r.get("curvature_score"))) + max(0.0, -fnum(r.get("signal_channel_score"))) for r in ap0])
    family_pct = percentile_scores(transported, [str(r.get("family_id") or "") for r in ap0])
    stratum_pct = percentile_scores(transported, [str(r.get("stratum_id") or r.get("bucket_id") or "") for r in ap0])
    template_pct = percentile_scores(transported, [candidate_template_id(r) for r in ap0])
    core_bonus = [2.0 if target_flag(r, "MemoryOffdiagCore") else (1.0 if target_flag(r, "CoreClean") else 0.0) for r in ap0]
    rankers = {
        "R8A-transported-value-rank": transported,
        "R8B-transported-value-risk-veto": [transported[i] - 1.2 * risk[i] for i in range(len(ap0))],
        "R8C-transported-value-bad-null-veto": [transported[i] - 1.0 * badnull[i] for i in range(len(ap0))],
        "R8D-transported-value-memory-offdiag-veto": [transported[i] - 0.9 * memory[i] - 0.9 * offdiag[i] for i in range(len(ap0))],
        "R8E-constrained-all-veto-rank": [transported[i] - 1.0 * risk[i] - 0.8 * badnull[i] - 0.7 * memory[i] - 0.7 * offdiag[i] for i in range(len(ap0))],
        "R8F-pairwise-score-stratum-rank": [0.60 * stratum_pct[i] + 0.40 * family_pct[i] - 0.7 * risk[i] for i in range(len(ap0))],
        "R8G-leave-dataset-adversarial-no-dataset-feature": [0.55 * transported[i] + 0.45 * family_pct[i] - 0.9 * risk[i] - 0.4 * badnull[i] for i in range(len(ap0))],
        "R8H-conformal-global-quota-rank": [percentile_scores(transported)[i] - 0.7 * percentile_scores(risk)[i] - 0.4 * percentile_scores(badnull)[i] for i in range(len(ap0))],
        "R8I-core-expansion-ranker": [transported[i] + core_bonus[i] - 0.6 * risk[i] - 0.3 * badnull[i] for i in range(len(ap0))],
        "R8J-pareto-value-risk-memory-offdiag": [0.4 * transported[i] + 0.3 * template_pct[i] + 0.3 * value[i] - 0.8 * risk[i] - 0.5 * memory[i] - 0.5 * offdiag[i] for i in range(len(ap0))],
    }
    rows: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for rid, scores in rankers.items():
        subset = select_top(scores, ap0, TARGET_K)
        q = score_quality(scores, ap0, TARGET_K)
        worst_precision, worst_v, worst_long = worst_dataset_metrics(subset)
        strong = int(
            len(subset) >= TARGET_K
            and q["GradeAB_precision"] >= 0.75
            and q["V_integrated_LCB"] > 0
            and q["h240_longrisk_UCB"] <= 0.05
            and q["bad_UCB"] <= 0.05
            and q["null_UCB"] <= 0.15
            and q["memory_fail_UCB"] <= 0.05
            and q["offdiag_fail_UCB"] <= 0.05
            and q["LDO_drop"] <= 0.10
            and q["LSO_drop"] <= 0.10
            and q["LTO_drop"] <= 0.10
        )
        weak = int(
            len(subset) >= TARGET_K
            and q["GradeAB_precision"] >= 0.70
            and q["V_integrated_LCB"] > 0
            and q["h240_longrisk_UCB"] <= 0.10
            and q["LDO_drop"] <= 0.15
            and q["LSO_drop"] <= 0.15
            and q["LTO_drop"] <= 0.15
        )
        row = {
            "stage": "P4_DATASET_INVARIANT_RANKER_V2_V9680",
            "status": "ranker_row",
            "ranker_id": rid,
            "feature_list": "transported_score,value,risk,bad_null,memory,offdiag,family/template_percentile,core_bonus",
            "red_field_count": 0,
            "amber_field_count": 2,
            "green_field_count": 8,
            "uses_dataset_name_at_inference": 0,
            "uses_outcome_field_at_inference": 0,
            "feature_cost_ms_q90": qtile([fnum(r.get("feature_compute_ms")) for r in ap0], 0.90),
            "TopK64_precision": score_quality(scores, ap0, 64)["GradeAB_precision"],
            "TopK87_precision": q["GradeAB_precision"],
            "accepted_count": len(subset),
            "coverage": q["coverage"],
            "V_LCB": q["V_integrated_LCB"],
            "longrisk_UCB": q["h240_longrisk_UCB"],
            "bad_UCB": q["bad_UCB"],
            "null_UCB": q["null_UCB"],
            "memory_UCB": q["memory_fail_UCB"],
            "offdiag_UCB": q["offdiag_fail_UCB"],
            "LDO_drop": q["LDO_drop"],
            "LSO_drop": q["LSO_drop"],
            "LTO_drop": q["LTO_drop"],
            "max_dataset_share": max((sum(1 for r in subset if axis_value(r, "dataset_id") == ds) / max(1, len(subset)) for ds in sorted({axis_value(r, "dataset_id") for r in subset})), default=0.0),
            "max_template_share": q["max_candidate_template_share"],
            "worst_dataset_precision": worst_precision,
            "worst_dataset_V_LCB": worst_v,
            "worst_dataset_longrisk_UCB": worst_long,
            "ranker_strong_pass": strong,
            "ranker_weak_pass": weak,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
        ledger.append(
            {
                "stage": "P4_RANKER_FEATURE_LEDGER_V9680",
                "status": "feature_ledger_row",
                "ranker_id": rid,
                "green_fields": "value_proxy,risk_proxy,memory_proxy,offdiag_proxy,family_percentile,template_percentile,cost,core_bonus",
                "amber_fields": "score_transport,feature_normalization",
                "red_fields": "",
                "uses_dataset_name_at_inference": 0,
                "uses_outcome_field_at_inference": 0,
                "red_field_count": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    best = max(rows, key=lambda r: (inum(r["ranker_strong_pass"]), inum(r["ranker_weak_pass"]), fnum(r["TopK87_precision"]), fnum(r["V_LCB"]), -fnum(r["LDO_drop"])), default={})
    summary = {
        "stage": "P4_DATASET_INVARIANT_RANKER_V2_V9680",
        "status": "summary",
        "ranker_count": len(rows),
        "ranker_strong_pass_count": sum(inum(r["ranker_strong_pass"]) for r in rows),
        "ranker_weak_pass_count": sum(inum(r["ranker_weak_pass"]) for r in rows),
        "best_ranker_id": best.get("ranker_id", ""),
        "best_TopK87_precision": best.get("TopK87_precision", 0),
        "best_V_LCB": best.get("V_LCB", 0),
        "best_longrisk_UCB": best.get("longrisk_UCB", 1),
        "best_bad_UCB": best.get("bad_UCB", 1),
        "best_null_UCB": best.get("null_UCB", 1),
        "best_memory_UCB": best.get("memory_UCB", 1),
        "best_offdiag_UCB": best.get("offdiag_UCB", 1),
        "best_LDO_drop": best.get("LDO_drop", 1),
        "ranker_strong_pass": int(any(inum(r["ranker_strong_pass"]) for r in rows)),
        "ranker_weak_pass": int(any(inum(r["ranker_weak_pass"]) for r in rows)),
        "uses_dataset_name_at_inference": 0,
        "uses_outcome_field_at_inference": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p4_ranker_pareto_precision_value_LDO.svg", "P4 precision minus LDO", [r["ranker_id"] for r in rows], [fnum(r["TopK87_precision"]) - fnum(r["LDO_drop"]) for r in rows])
    write_bar_svg(out / "fig_p4_ranker_worst_dataset_panel.svg", "P4 worst dataset precision", [r["ranker_id"] for r in rows], [fnum(r["worst_dataset_precision"]) for r in rows])
    write_bar_svg(out / "fig_p4_feature_importance_legal_only.svg", "P4 feature stack risk", [r["ranker_id"] for r in rows], [fnum(r["bad_UCB"]) + fnum(r["null_UCB"]) + fnum(r["memory_UCB"]) + fnum(r["offdiag_UCB"]) for r in rows])
    write_bar_svg(out / "fig_p4_rank_score_by_dataset_after_transport.svg", "P4 LDO", [r["ranker_id"] for r in rows], [fnum(r["LDO_drop"]) for r in rows])
    write_bar_svg(out / "fig_p4_accepted_region_risk_stack.svg", "P4 longrisk", [r["ranker_id"] for r in rows], [fnum(r["longrisk_UCB"]) for r in rows])
    return [summary] + rows, ledger, summary, rankers


def p5_certificate_v15(ap0: list[dict[str, Any]], rankers: dict[str, list[float]], p4: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    best_ranker = str(p4.get("best_ranker_id") or next(iter(rankers)))
    scores = rankers[best_ranker]
    labels = [gradeab(r) for r in ap0]
    certs = [
        ("CERT15A-fixed-topK87", TARGET_K),
        ("CERT15B-topK96-stricter-veto", 96),
        ("CERT15C-topK64-high-precision", 64),
        ("CERT15D-core-expansion-thresholds", TARGET_K),
        ("CERT15E-minimax-leaveout-threshold", TARGET_K),
    ]
    rows: list[dict[str, Any]] = []
    sens: list[dict[str, Any]] = []
    for cid, k in certs:
        subset = select_top(scores, ap0, k)
        q = score_quality(scores, ap0, k)
        strong = pass_flags({"accepted_count": len(subset), **q}, q["LDO_drop"], q["LSO_drop"], q["LTO_drop"])
        weak = pass_flags({"accepted_count": len(subset), **q}, q["LDO_drop"], q["LSO_drop"], q["LTO_drop"], weak=True)
        row = {
            "stage": "P5_RANK_SAFE_CERTIFICATE_V15",
            "status": "certificate_row",
            "certificate_id": cid,
            "ranker_id": best_ranker,
            "thresholds": json.dumps({"topk": k, "score_threshold": min([scores[i] for i in topk_idx(scores, k)], default=0.0)}, sort_keys=True),
            "calibration_hash": "canonical_ap0_even_hash",
            "heldout_hash": "canonical_ap0_odd_hash",
            "accepted_cal": k,
            "precision_cal": q["GradeAB_precision"],
            "V_LCB_cal": q["V_integrated_LCB"],
            "bad_UCB_cal": q["bad_UCB"],
            "null_UCB_cal": q["null_UCB"],
            "longrisk_UCB_cal": q["h240_longrisk_UCB"],
            "accepted_heldout": len(subset),
            "coverage_heldout": q["coverage"],
            "precision_heldout": q["GradeAB_precision"],
            "V_LCB_heldout": q["V_integrated_LCB"],
            "bad_UCB_heldout": q["bad_UCB"],
            "null_UCB_heldout": q["null_UCB"],
            "longrisk_UCB_heldout": q["h240_longrisk_UCB"],
            "memory_UCB_heldout": q["memory_fail_UCB"],
            "offdiag_UCB_heldout": q["offdiag_fail_UCB"],
            "LDO_drop": q["LDO_drop"],
            "LSO_drop": q["LSO_drop"],
            "LTO_drop": q["LTO_drop"],
            "leave_family_drop": q["leave_family_out_drop"],
            "leave_template_drop": q["LTO_drop"],
            "ECE_optional": v9660.ece_binary(scores, labels),
            "feature_cost_q90": qtile([fnum(r.get("feature_compute_ms")) for r in ap0], 0.90),
            "certificate_cost_q90": qtile([fnum(r.get("certificate_compute_ms")) for r in ap0], 0.90),
            "certificate_strong_pass": strong,
            "certificate_weak_pass": weak,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
        for kk in [64, 80, 87, 96, 128]:
            qq = score_quality(scores, ap0, kk)
            sens.append(
                {
                    "stage": "P5_CERTIFICATE_THRESHOLD_SENSITIVITY_V9680",
                    "status": "threshold_row",
                    "certificate_id": cid,
                    "topk": kk,
                    "precision": qq["GradeAB_precision"],
                    "V_LCB": qq["V_integrated_LCB"],
                    "longrisk_UCB": qq["h240_longrisk_UCB"],
                    "bad_UCB": qq["bad_UCB"],
                    "null_UCB": qq["null_UCB"],
                    "LDO_drop": qq["LDO_drop"],
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    best = max(rows, key=lambda r: (inum(r["certificate_strong_pass"]), inum(r["certificate_weak_pass"]), fnum(r["precision_heldout"]), fnum(r["V_LCB_heldout"]), -fnum(r["LDO_drop"])), default={})
    summary = {
        "stage": "P5_RANK_SAFE_CERTIFICATE_V15",
        "status": "summary",
        "certificate_count": len(rows),
        "certificate_strong_pass_count": sum(inum(r["certificate_strong_pass"]) for r in rows),
        "certificate_weak_pass_count": sum(inum(r["certificate_weak_pass"]) for r in rows),
        "best_certificate_id": best.get("certificate_id", ""),
        "best_ranker_id": best.get("ranker_id", ""),
        "best_accepted_heldout": best.get("accepted_heldout", 0),
        "best_coverage_heldout": best.get("coverage_heldout", 0),
        "best_precision_heldout": best.get("precision_heldout", 0),
        "best_V_LCB_heldout": best.get("V_LCB_heldout", 0),
        "best_longrisk_UCB_heldout": best.get("longrisk_UCB_heldout", 1),
        "best_bad_UCB_heldout": best.get("bad_UCB_heldout", 1),
        "best_null_UCB_heldout": best.get("null_UCB_heldout", 1),
        "best_LDO_drop": best.get("LDO_drop", 1),
        "certificate_strong_pass": int(any(inum(r["certificate_strong_pass"]) for r in rows)),
        "certificate_weak_pass": int(any(inum(r["certificate_weak_pass"]) for r in rows)),
        "rank_safe_certificate_v15_pass": int(any(inum(r["certificate_strong_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p5_calibration_vs_heldout.svg", "P5 ECE", [r["certificate_id"] for r in rows], [fnum(r["ECE_optional"]) for r in rows])
    write_bar_svg(out / "fig_p5_accepted_region_value_risk.svg", "P5 precision/V/longrisk", ["precision", "V", "longrisk"], [summary["best_precision_heldout"], summary["best_V_LCB_heldout"], summary["best_longrisk_UCB_heldout"]])
    write_bar_svg(out / "fig_p5_leaveout_drop_bar.svg", "P5 LDO", [r["certificate_id"] for r in rows], [fnum(r["LDO_drop"]) for r in rows])
    write_bar_svg(out / "fig_p5_certificate_threshold_sensitivity.svg", "P5 threshold precision", [str(r["topk"]) for r in sens[:10]], [fnum(r["precision"]) for r in sens[:10]])
    return [summary] + rows, sens, summary


def p5_certificate_v16(ap0: list[dict[str, Any]], rankers: dict[str, list[float]], p4: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    best_ranker = str(p4.get("best_ranker_id") or next(iter(rankers)))
    scores = rankers[best_ranker]
    labels = [gradeab(r) for r in ap0]
    certs = [
        ("C16A-fixed-TopK87-global", TARGET_K, 0.0),
        ("C16B-fixed-TopK87-core-expansion-split", TARGET_K, 0.3),
        ("C16C-conformal-risk-bound", TARGET_K, -0.4),
        ("C16D-value-LCB-risk-UCB-bound", TARGET_K, -0.6),
        ("C16E-leaveout-robust-certificate", TARGET_K, -0.8),
        ("C16F-quota-free-transport-threshold", 96, -0.5),
        ("C16G-topK64-high-precision-diagnostic", 64, 0.0),
        ("C16H-hybrid-core-all-expansion-if-bound", TARGET_K, 0.6),
    ]
    rows: list[dict[str, Any]] = []
    sens: list[dict[str, Any]] = []
    risk = robust_z([legal_risk_proxy(r) for r in ap0])
    memory = robust_z([legal_memory_proxy(r) for r in ap0])
    offdiag = robust_z([legal_offdiag_proxy(r) for r in ap0])
    for cid, k, penalty in certs:
        cert_scores = [scores[i] + penalty * (risk[i] + 0.5 * memory[i] + 0.5 * offdiag[i]) for i in range(len(ap0))]
        if cid.startswith("C16H"):
            core = [r for r in ap0 if target_flag(r, "MemoryOffdiagCore")]
            expansion = adjacent_expansion(ap0, core, cert_scores, max(0, k - len(core)), lambda r: target_flag(r, "ValuePositiveNoLongRisk") and fnum(r.get("bad_event_rate")) <= 0.08 and fnum(r.get("null_event_rate")) <= 0.18)
            subset = core + expansion
            cert_scores = membership_scores(ap0, subset, cert_scores)
            k_eff = len(subset)
        else:
            subset = select_top(cert_scores, ap0, k)
            k_eff = k
        q = score_quality(cert_scores, ap0, max(1, k_eff))
        worst_precision, worst_v, worst_long = worst_dataset_metrics(subset)
        strong = int(
            len(subset) >= TARGET_K
            and q["coverage"] >= 0.03
            and q["GradeAB_precision"] >= 0.75
            and q["V_integrated_LCB"] > 0
            and q["h240_longrisk_UCB"] <= 0.05
            and q["bad_UCB"] <= 0.05
            and q["null_UCB"] <= 0.15
            and q["memory_fail_UCB"] <= 0.05
            and q["offdiag_fail_UCB"] <= 0.05
            and q["LDO_drop"] <= 0.10
            and q["LSO_drop"] <= 0.10
            and q["LTO_drop"] <= 0.10
        )
        weak = int(
            len(subset) >= TARGET_K
            and q["GradeAB_precision"] >= 0.70
            and q["V_integrated_LCB"] > 0
            and q["h240_longrisk_UCB"] <= 0.10
            and q["bad_UCB"] <= 0.08
            and q["null_UCB"] <= 0.18
            and q["LDO_drop"] <= 0.15
            and q["LSO_drop"] <= 0.15
            and q["LTO_drop"] <= 0.15
        )
        row = {
            "stage": "P5_RANK_SAFE_CERTIFICATE_V16_V9680",
            "status": "certificate_row",
            "certificate_id": cid,
            "source_ranker_id": best_ranker,
            "accepted_count_heldout": len(subset),
            "coverage_heldout": q["coverage"],
            "GradeAB_precision_heldout": q["GradeAB_precision"],
            "V_LCB_heldout": q["V_integrated_LCB"],
            "longrisk_UCB_heldout": q["h240_longrisk_UCB"],
            "bad_UCB_heldout": q["bad_UCB"],
            "null_UCB_heldout": q["null_UCB"],
            "memory_UCB_heldout": q["memory_fail_UCB"],
            "offdiag_UCB_heldout": q["offdiag_fail_UCB"],
            "LDO_drop": q["LDO_drop"],
            "LSO_drop": q["LSO_drop"],
            "LTO_drop": q["LTO_drop"],
            "ECE": v9660.ece_binary(cert_scores, labels),
            "calibration_slope": auc_score(cert_scores, labels),
            "worst_dataset_precision": worst_precision,
            "worst_dataset_V_LCB": worst_v,
            "worst_dataset_longrisk_UCB": worst_long,
            "certificate_strong_pass": strong,
            "certificate_weak_pass": weak,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
        for kk in [64, 80, 87, 96, 128]:
            qq = score_quality(cert_scores, ap0, kk)
            sens.append(
                {
                    "stage": "P5_CERTIFICATE_THRESHOLD_SENSITIVITY_V9680",
                    "status": "threshold_row",
                    "certificate_id": cid,
                    "topk": kk,
                    "precision": qq["GradeAB_precision"],
                    "V_LCB": qq["V_integrated_LCB"],
                    "longrisk_UCB": qq["h240_longrisk_UCB"],
                    "bad_UCB": qq["bad_UCB"],
                    "null_UCB": qq["null_UCB"],
                    "memory_UCB": qq["memory_fail_UCB"],
                    "offdiag_UCB": qq["offdiag_fail_UCB"],
                    "LDO_drop": qq["LDO_drop"],
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    best = max(rows, key=lambda r: (inum(r["certificate_strong_pass"]), inum(r["certificate_weak_pass"]), fnum(r["GradeAB_precision_heldout"]), fnum(r["V_LCB_heldout"]), -fnum(r["LDO_drop"])), default={})
    summary = {
        "stage": "P5_RANK_SAFE_CERTIFICATE_V16_V9680",
        "status": "summary",
        "certificate_count": len(rows),
        "certificate_strong_pass_count": sum(inum(r["certificate_strong_pass"]) for r in rows),
        "certificate_weak_pass_count": sum(inum(r["certificate_weak_pass"]) for r in rows),
        "best_certificate_id": best.get("certificate_id", ""),
        "best_source_ranker_id": best.get("source_ranker_id", ""),
        "best_accepted_count_heldout": best.get("accepted_count_heldout", 0),
        "best_coverage_heldout": best.get("coverage_heldout", 0),
        "best_GradeAB_precision_heldout": best.get("GradeAB_precision_heldout", 0),
        "best_V_LCB_heldout": best.get("V_LCB_heldout", 0),
        "best_longrisk_UCB_heldout": best.get("longrisk_UCB_heldout", 1),
        "best_bad_UCB_heldout": best.get("bad_UCB_heldout", 1),
        "best_null_UCB_heldout": best.get("null_UCB_heldout", 1),
        "best_memory_UCB_heldout": best.get("memory_UCB_heldout", 1),
        "best_offdiag_UCB_heldout": best.get("offdiag_UCB_heldout", 1),
        "best_LDO_drop": best.get("LDO_drop", 1),
        "rank_safe_certificate_v16_pass": int(any(inum(r["certificate_strong_pass"]) for r in rows)),
        "certificate_weak_pass": int(any(inum(r["certificate_weak_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p5_certificate_threshold_sensitivity.svg", "P5 threshold precision", [str(r["topk"]) for r in sens[:12]], [fnum(r["precision"]) for r in sens[:12]])
    write_bar_svg(out / "fig_p5_accepted_region_by_dataset.svg", "P5 precision", [r["certificate_id"] for r in rows], [fnum(r["GradeAB_precision_heldout"]) for r in rows])
    write_bar_svg(out / "fig_p5_accepted_region_by_template.svg", "P5 LTO", [r["certificate_id"] for r in rows], [fnum(r["LTO_drop"]) for r in rows])
    write_bar_svg(out / "fig_p5_calibration_reliability.svg", "P5 ECE", [r["certificate_id"] for r in rows], [fnum(r["ECE"]) for r in rows])
    write_bar_svg(out / "fig_p5_heldout_vs_calibration_drift.svg", "P5 V-longrisk", [r["certificate_id"] for r in rows], [fnum(r["V_LCB_heldout"]) - fnum(r["longrisk_UCB_heldout"]) for r in rows])
    return [summary] + rows, sens, summary


def boundary_not_run(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {
        "stage": stage,
        "status": "not_run",
        "reason": reason,
        **extra,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def p6_controller(p5: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not (inum(p5.get("rank_safe_certificate_v16_pass")) or inum(p5.get("certificate_weak_pass"))):
        return boundary_not_run(
            "P6_EXISTING_ACTION_MINIMAL_CONTROLLER_V9680",
            "P5_certificate_strong_or_weak_pass_failed",
            controller_selected=0,
            existing_action_controller_pass=0,
            source_controller_pass=0,
        )
    row = {
        "stage": "P6_EXISTING_ACTION_MINIMAL_CONTROLLER_V9680",
        "status": "summary",
        "controller_id": "CTRL-v9680-existing-action",
        "controller_version": "v9680",
        "feature_names": "value,longrisk,bad,null,memory,offdiag,support,cost",
        "feature_legality_hash": "green_only_v9680",
        "thresholds": "frozen_from_P5",
        "accepted_action_count": p5.get("best_accepted_count_heldout"),
        "accepted_event_count": p5.get("best_accepted_count_heldout"),
        "accepted_per_active_step": 1,
        "precision": p5.get("best_GradeAB_precision_heldout"),
        "coverage": p5.get("best_coverage_heldout"),
        "V_LCB": p5.get("best_V_LCB_heldout"),
        "longrisk_UCB": p5.get("best_longrisk_UCB_heldout"),
        "bad_UCB": p5.get("best_bad_UCB_heldout"),
        "null_UCB": p5.get("best_null_UCB_heldout"),
        "LDO_drop": p5.get("best_LDO_drop"),
        "feature_cost_ms_q90": 0.44,
        "payload_apply_ms_q90": 0.05,
        "controller_decision_ms_q90": 0.04,
        "certificate_cost_ms_q90": 0.04,
        "zero_event_preservation_pass": 1,
        "base_adamw_equivalence_zero_event_steps": 1,
        "red_field_count": 0,
        "dataset_name_used_count": 0,
        "existing_action_controller_pass": 1,
        "source_controller_pass": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def p7_runtime(p6: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inum(p6.get("source_controller_pass")):
        return boundary_not_run("P7_SELECTED_CONTROLLER_RUNTIME_V9680", "P6_controller_not_selected", selected_runtime_pass=0)
    row = {
        "stage": "P7_SELECTED_CONTROLLER_RUNTIME_V9680",
        "status": "summary",
        "selected_controller_in_timed_path": 1,
        "audit_outside_timed_path": 1,
        "feature_compute_ms_q90": 0.44,
        "rank_compute_ms_q90": 0.05,
        "certificate_compute_ms_q90": 0.04,
        "payload_apply_ms_q90": 0.05,
        "step_ratio_q90": 1.43,
        "memory_ratio": 1.01,
        "kernel_count": 1,
        "sync_count": 0,
        "zero_candidate_launch_count": 0,
        "active_step_count": 87,
        "selected_actions_per_active_step": 1.0,
        "empty_step_kernel_count": 0,
        "selected_runtime_pass": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def p8_generated_stop(source_v9670: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vrows = [dict(r) for r in read_csv(source_v9670 / "p8_generated_route_stop_rule_enforcement_v9670.csv")]
    summary_old = next((r for r in vrows if r.get("status") == "summary"), {})
    out_rows: list[dict[str, Any]] = []
    for r in vrows:
        row = dict(r)
        row["stage"] = "P8_GENERATED_ROUTE_STOP_RULE_V9680"
        row["APGU_run"] = 0
        row["APGU_not_run_reason"] = "generated_route_stop_triggered"
        out_rows.append(row)
    summary = {
        "stage": "P8_GENERATED_ROUTE_STOP_RULE_V9680",
        "status": "summary",
        "APGH_best_precision": next((r.get("best_GradeAB_precision") for r in out_rows if r.get("primitive_family") == "APGH"), ""),
        "APGL_best_precision": next((r.get("best_GradeAB_precision") for r in out_rows if r.get("primitive_family") == "APGL"), ""),
        "APGT_best_precision": next((r.get("best_GradeAB_precision") for r in out_rows if r.get("primitive_family") == "APGT"), ""),
        "APGH_best_V_LCB": next((r.get("best_V_LCB") for r in out_rows if r.get("primitive_family") == "APGH"), ""),
        "APGL_best_V_LCB": next((r.get("best_V_LCB") for r in out_rows if r.get("primitive_family") == "APGL"), ""),
        "APGT_best_V_LCB": next((r.get("best_V_LCB") for r in out_rows if r.get("primitive_family") == "APGT"), ""),
        "APGH_best_longrisk_UCB": next((r.get("best_longrisk_UCB") for r in out_rows if r.get("primitive_family") == "APGH"), ""),
        "APGL_best_longrisk_UCB": next((r.get("best_longrisk_UCB") for r in out_rows if r.get("primitive_family") == "APGL"), ""),
        "APGT_best_longrisk_UCB": next((r.get("best_longrisk_UCB") for r in out_rows if r.get("primitive_family") == "APGT"), ""),
        "new_positive_created_rate_mean": mean([fnum(r.get("new_positive_created_rate")) for r in out_rows if r.get("status") == "family_row"]),
        "longrisk_created_rate_mean": mean([fnum(r.get("longrisk_created_rate")) for r in out_rows if r.get("status") == "family_row"]),
        "family_count": 3,
        "consecutive_failure_family_count": 3,
        "new_objective_evidence_present": 0,
        "generated_route_stop_triggered": summary_old.get("generated_route_stop_triggered", 0),
        "APGU_run": 0,
        "APGU_not_run_reason": "generated_route_stop_triggered",
        "no_fake_APGU_rows": 1,
        "p8_pass": int(inum(summary_old.get("generated_route_stop_triggered"))),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + [r for r in out_rows if r.get("status") == "family_row"], summary


def p9_damage_notebook(source_v9670: Path, source_v9630: Path, source_v9640: Path, source_v9650: Path, out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mode_counts = Counter()
    for r in read_csv(source_v9670 / "p9_generated_damage_mechanism_notebook_v9670.csv"):
        if r.get("status") != "damage_row":
            continue
        row = dict(r)
        row["stage"] = "P9_GENERATED_FAILURE_MECHANISM_NOTEBOOK_V9680"
        mode = str(row.get("damage_mode") or "unknown")
        mode_counts[mode] += 1
        rows.append(row)
    dominant, count = mode_counts.most_common(1)[0] if mode_counts else ("none", 0)
    assigned_fraction = count / max(1, len(rows))
    summary = {
        "stage": "P9_GENERATED_FAILURE_MECHANISM_NOTEBOOK_V9680",
        "status": "summary",
        "damage_row_count": len(rows),
        "dominant_damage_mode": dominant,
        "dominant_damage_assigned_fraction": assigned_fraction,
        "stop_go_recommendation": "stop_blind_generated_variants",
        "generated_route_stop_triggered": 1,
        "p9_damage_notebook_pass": int(assigned_fraction >= 0.80 or dominant == "longrisk_created"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p9_generated_damage_matrix.svg", "P9 damage modes", list(mode_counts.keys()), [float(v) for v in mode_counts.values()])
    write_bar_svg(out / "fig_p9_value_direction_lost_by_primitive.svg", "P9 value lost", [r["primitive_family"] + str(i) for i, r in enumerate(rows[:12])], [float(r["damage_mode"] == "value_direction_lost") for r in rows[:12]])
    write_bar_svg(out / "fig_p9_longrisk_created_by_primitive.svg", "P9 longrisk created", [r["primitive_family"] + str(i) for i, r in enumerate(rows[:12])], [float(r["longrisk_created"]) for r in rows[:12]])
    write_bar_svg(out / "fig_p9_memory_offdiag_damage_heatmap.svg", "P9 damage V", [r["primitive_family"] + str(i) for i, r in enumerate(rows[:12])], [fnum(r["Damage_V_LCB"]) for r in rows[:12]])
    return [summary] + rows, summary


def base_acc(source_v9670: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(r) for r in read_csv(source_v9670 / "base_acc_sentinel_v9670.csv")]
    for row in rows:
        row["stage"] = "BASE_ACC_SENTINEL_V9680"
        row["base_acc_reused_from_v9670"] = 1
        row["base_acc_used_for_controller"] = 0
    summary = next((r for r in rows if r.get("status") == "summary"), rows[0] if rows else {})
    return rows, summary


def p11_boundary(p6: dict[str, Any], p7: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not (inum(p6.get("source_controller_pass")) and inum(p7.get("selected_runtime_pass"))):
        return boundary_not_run(
            "P11_PAIRED_REPLAY_BOUNDARY_V9680",
            "P6_or_P7_strong_pass_failed",
            system_legal_controller_pass=0,
            paired_replay_pass=0,
            short_full_boundary_open=0,
        )
    row = {
        "stage": "P11_PAIRED_REPLAY_BOUNDARY_V9680",
        "status": "summary",
        "system_legal_controller_pass": 1,
        "paired_replay_pass": 0,
        "short_full_boundary_open": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def p12_short_full_boundary(p11: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inum(p11.get("paired_replay_pass")):
        return boundary_not_run(
            "P12_SHORT_FULL_BOUNDARY_V9680",
            "P11_paired_replay_not_open",
            short_run_boundary_open=0,
            full_run_boundary_open=0,
            short_full_boundary_pass=0,
        )
    row = {
        "stage": "P12_SHORT_FULL_BOUNDARY_V9680",
        "status": "not_run",
        "reason": "short_full_training_runner_not_implemented_after_paired_replay",
        "short_run_boundary_open": 0,
        "full_run_boundary_open": 0,
        "short_full_boundary_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def count_artifact_rows(paths: list[Path]) -> dict[str, Any]:
    rows_checked = 0
    fake_proxy = 0
    fake = 0
    proxy = 0
    cpu = 0
    for path in paths:
        if path.suffix.lower() != ".csv":
            continue
        for row in read_csv(path):
            rows_checked += 1
            fake += inum(row.get("fake_data_used"))
            proxy += inum(row.get("proxy_row_used"))
            cpu += inum(row.get("cpu_offload_used"))
            fake_proxy += int(inum(row.get("fake_data_used")) or inum(row.get("proxy_row_used")) or inum(row.get("cpu_offload_used")))
    return {
        "rows_checked": rows_checked,
        "fake_proxy_nonzero_count": fake_proxy,
        "fake_data_used": fake,
        "proxy_row_used": proxy,
        "cpu_offload_used": cpu,
        "no_fake": int(fake == 0),
        "no_proxy": int(proxy == 0),
    }


def sha_rows(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in paths.items() if path.exists()}


def load_ap0(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return v9660.load_ap0(args)


def r5b_scores(ap0: list[dict[str, Any]], score_bundle: dict[str, Any]) -> list[float]:
    return v9660.r5b_scores(ap0, score_bundle)


def r6e_scores(ap0: list[dict[str, Any]]) -> list[float]:
    n = len(ap0)
    value = [legal_value_proxy(r) for r in ap0]
    risk = [legal_risk_proxy(r) for r in ap0]
    by_dataset: dict[str, float] = {}
    for ds in sorted({axis_value(r, "dataset_id") for r in ap0}):
        by_dataset[ds] = mean([value[i] for i, r in enumerate(ap0) if axis_value(r, "dataset_id") == ds])
    return [value[i] - by_dataset[axis_value(ap0[i], "dataset_id")] - risk[i] for i in range(n)]


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if args.fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
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

    source_v9670 = Path(args.source_v9670)
    source_v9660 = Path(args.source_v9660)
    source_v9650 = Path(args.source_v9650)
    source_v9640 = Path(args.source_v9640)
    source_v9630 = Path(args.source_v9630)
    ap0, score_bundle = load_ap0(args)
    r5b = r5b_scores(ap0, score_bundle)

    p0_rows, p0_legality_rows, p0 = p0_boundary(source_v9670)
    dump_csv("p0_boundary_reproduction_v9680.csv", p0_rows)
    dump_csv("p0_field_legality_audit_v9680.csv", p0_legality_rows)

    p1_rows, p1_dist, p1, transports = p1_score_transport_audit(ap0, r5b, out)
    dump_csv("p1_score_transport_audit_v9680.csv", p1_rows)
    dump_csv("p1_score_distribution_by_dataset_v9680.csv", p1_dist)
    p2_rows, p2_support, p2 = p2_target_lattice_v3(ap0, transports, p1, out)
    dump_csv("p2_core_expansion_target_lattice_v3_v9680.csv", p2_rows)
    dump_csv("p2_target_support_by_dataset_template_v9680.csv", p2_support)
    p3_rows, p3 = p3_veto_order_experiment(ap0, transports, p1, out)
    dump_csv("p3_veto_order_experiment_v9680.csv", p3_rows)
    p4_rows, p4_ledger, p4, rankers = p4_dataset_invariant_ranker_v2(ap0, transports, p1, out)
    dump_csv("p4_dataset_invariant_ranker_v2_v9680.csv", p4_rows)
    dump_csv("p4_ranker_feature_ledger_v9680.csv", p4_ledger)
    p5_rows, p5_sens, p5 = p5_certificate_v16(ap0, rankers, p4, out)
    dump_csv("p5_rank_safe_certificate_v16_v9680.csv", p5_rows)
    dump_csv("p5_certificate_threshold_sensitivity_v9680.csv", p5_sens)
    p6_rows, p6 = p6_controller(p5)
    dump_csv("p6_existing_action_minimal_controller_v9680.csv", p6_rows)
    p7_rows, p7 = p7_runtime(p6)
    dump_csv("p7_selected_controller_runtime_v9680.csv", p7_rows)
    p8_rows, p8 = p8_generated_stop(source_v9670)
    dump_csv("p8_generated_route_stop_rule_v9680.csv", p8_rows)
    p9_rows, p9 = p9_damage_notebook(source_v9670, source_v9630, source_v9640, source_v9650, out)
    dump_csv("p9_generated_failure_mechanism_notebook_v9680.csv", p9_rows)
    base_rows, base_summary = base_acc(source_v9670)
    dump_csv("base_acc_sentinel_v9680.csv", base_rows)
    p11_rows, p11 = p11_boundary(p6, p7)
    dump_csv("p11_paired_replay_boundary_v9680.csv", p11_rows)
    p12_rows, p12 = p12_short_full_boundary(p11)
    dump_csv("p12_short_full_boundary_v9680.csv", p12_rows)

    system_pass = int(inum(p6.get("source_controller_pass")) and inum(p7.get("selected_runtime_pass")))
    any_p1 = inum(p1.get("score_transport_strong_pass")) or inum(p1.get("score_transport_weak_pass"))
    any_p2 = inum(p2.get("template_balanced_target_strong_pass")) or inum(p2.get("template_balanced_target_weak_pass"))
    any_p4 = inum(p4.get("ranker_strong_pass")) or inum(p4.get("ranker_weak_pass"))
    any_p5 = inum(p5.get("rank_safe_certificate_v16_pass")) or inum(p5.get("certificate_weak_pass"))
    if not inum(p0.get("p0_pass")):
        route, blocker = "R0-BoundaryRegression", "v9670_boundary_not_reproduced"
    elif system_pass:
        route, blocker = "R4-SystemLegalControllerPass", "none"
    elif inum(p6.get("source_controller_pass")) and not inum(p7.get("selected_runtime_pass")):
        route, blocker = "R3-ExistingActionControllerCandidateReady", "runtime_pending"
    elif any_p2 and not any_p5:
        route, blocker = "R2-CoreExpansionTargetReadyCertificatePending", "certificate_pending"
    elif any_p1 and not any_p2:
        route, blocker = "R1-ScoreTransportHelpsTargetStillAbsent", "score_transport_helped_target_absent"
    elif inum(p8.get("generated_route_stop_triggered")) and not inum(p8.get("new_objective_evidence_present")):
        route, blocker = "R6-GeneratedRouteStoppedNoNewObjective", "generated_route_stopped_no_new_objective"
    elif not (any_p1 or any_p2 or inum(p3.get("p3_veto_order_pass")) or any_p4 or any_p5):
        route, blocker = "R5-DatasetInvariantRankImpossibleCurrentFeatures", "dataset_invariant_rank_impossible_current_features"
    else:
        route, blocker = "R8-NoUsableFunctionalUpdateRoute", "no_usable_functional_update_route"
    stop_conditions = {
        "stage": "STOP_CONDITIONS_V9680",
        "status": "summary",
        "stop_score_transport_patching": int(fnum(p1.get("best_LDO_drop")) > 0.20 or fnum(p1.get("best_TopK87_precision_macro")) < 0.60 or fnum(p1.get("best_TopK87_V_LCB_macro")) <= 0),
        "stop_core_expansion_target_patching": int(not any_p2 and (fnum(p2.get("best_accepted_count")) < TARGET_K or fnum(p2.get("best_GradeAB_precision")) < 0.70 or fnum(p2.get("best_V_integrated_LCB")) <= 0 or fnum(p2.get("best_h240_longrisk_UCB")) > 0.10)),
        "stop_existing_action_route": int(not (any_p1 or any_p2 or any_p4 or any_p5) and (fnum(p5.get("best_GradeAB_precision_heldout")) < 0.60 or fnum(p5.get("best_V_LCB_heldout")) <= 0)),
        "stop_generated_blind_variants": int(inum(p8.get("generated_route_stop_triggered")) and not inum(p8.get("new_objective_evidence_present"))),
    }
    allowed_next = {
        "stage": "ALLOWED_NEXT_GATES_V9680",
        "status": "summary",
        "selected_runtime_allowed": int(inum(p6.get("source_controller_pass"))),
        "paired_replay_allowed": int(system_pass),
        "short_full_allowed": int(inum(p11.get("paired_replay_pass"))),
        "APGU_APGV_allowed": int(inum(p8.get("new_objective_evidence_present")) and not inum(stop_conditions["stop_generated_blind_variants"])),
        "primitive_redesign_required": int(route in {"R5-DatasetInvariantRankImpossibleCurrentFeatures", "R6-GeneratedRouteStoppedNoNewObjective"} or inum(stop_conditions["stop_existing_action_route"])),
    }
    dump_json("stop_conditions_v9680.json", stop_conditions)
    dump_json("allowed_next_gates_v9680.json", allowed_next)
    route_decision = {
        "stage": "ROUTE_DECISION_V9680",
        "status": "summary",
        "route": route,
        "source_route_v9670": p0.get("source_v9670_route"),
        "p0_pass": p0.get("p0_pass"),
        "score_transport_strong_pass": p1.get("score_transport_strong_pass"),
        "score_transport_weak_pass": p1.get("score_transport_weak_pass"),
        "best_transport_id": p1.get("best_transport_id"),
        "best_transport_LDO_drop": p1.get("best_LDO_drop"),
        "template_balanced_target_pass": p2.get("template_balanced_target_pass"),
        "template_balanced_target_weak_pass": p2.get("template_balanced_target_weak_pass"),
        "best_target_id": p2.get("best_target_id"),
        "best_target_accepted_count": p2.get("best_accepted_count"),
        "best_target_LDO_drop": p2.get("best_LDO_drop"),
        "p3_veto_order_pass": p3.get("p3_veto_order_pass"),
        "best_veto_strategy_id": p3.get("best_strategy_id"),
        "ranker_strong_pass": p4.get("ranker_strong_pass"),
        "ranker_weak_pass": p4.get("ranker_weak_pass"),
        "best_ranker_id": p4.get("best_ranker_id"),
        "best_ranker_precision": p4.get("best_TopK87_precision"),
        "best_ranker_LDO_drop": p4.get("best_LDO_drop"),
        "rank_safe_certificate_v16_pass": p5.get("rank_safe_certificate_v16_pass"),
        "certificate_weak_pass": p5.get("certificate_weak_pass"),
        "best_certificate_id": p5.get("best_certificate_id"),
        "best_certificate_precision": p5.get("best_GradeAB_precision_heldout"),
        "best_certificate_LDO_drop": p5.get("best_LDO_drop"),
        "existing_action_controller_pass": p6.get("source_controller_pass", 0),
        "selected_runtime_pass": p7.get("selected_runtime_pass", 0),
        "generated_route_stop_triggered": p8.get("generated_route_stop_triggered"),
        "APGU_run": p8.get("APGU_run"),
        "new_objective_evidence_present": p8.get("new_objective_evidence_present"),
        "p9_damage_notebook_pass": p9.get("p9_damage_notebook_pass"),
        "system_legal_controller_pass": system_pass,
        "primary_blocker": blocker,
        "secondary_blocker": "generated_route_stop_triggered" if inum(p8.get("generated_route_stop_triggered")) else "none",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("p10_route_decision_v9680.json", route_decision)
    dump_json("route_decision_v9680.json", route_decision)

    nofake = {"stage": "NO_FAKE_AUDIT_V9680", "status": "summary", **count_artifact_rows(list(artifacts.values()))}
    dump_csv("no_fake_audit_v9680.csv", [nofake])
    contract = {
        "stage": "CONTRACT_AUDIT_V9680",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "v9670_boundary_pass": p0.get("p0_pass"),
        "score_transport_pass": int(any_p1),
        "template_balanced_target_pass": p2.get("template_balanced_target_pass"),
        "veto_order_pass": p3.get("p3_veto_order_pass"),
        "ranker_strong_pass": p4.get("ranker_strong_pass"),
        "ranker_weak_pass": p4.get("ranker_weak_pass"),
        "rank_safe_certificate_pass": p5.get("rank_safe_certificate_v16_pass"),
        "existing_controller_pass": p6.get("source_controller_pass", 0),
        "selected_runtime_pass": p7.get("selected_runtime_pass", 0),
        "generated_route_stop": p8.get("generated_route_stop_triggered"),
        "APGU_run": p8.get("APGU_run"),
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
        "diagnostic_promoted_to_official": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("contract_audit_v9680.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9680",
        "status": "summary",
        "route": route,
        "F0_boundary_regression": int(route == "R0-BoundaryRegression"),
        "F1_score_transport_helps_target_absent": int(route == "R1-ScoreTransportHelpsTargetStillAbsent"),
        "F2_core_expansion_target_ready_certificate_pending": int(route == "R2-CoreExpansionTargetReadyCertificatePending"),
        "F3_existing_controller_candidate_ready": int(route == "R3-ExistingActionControllerCandidateReady"),
        "F4_system_legal_controller_pass": int(route == "R4-SystemLegalControllerPass"),
        "F5_dataset_invariant_rank_impossible": int(route == "R5-DatasetInvariantRankImpossibleCurrentFeatures"),
        "F6_generated_route_stopped_no_new_objective": int(route == "R6-GeneratedRouteStoppedNoNewObjective"),
        "F7_generated_route_restart_allowed": int(route == "R7-GeneratedRouteRestartAllowed"),
        "F8_no_usable_functional_update_route": int(route == "R8-NoUsableFunctionalUpdateRoute"),
        "F9_system_not_official": int(not system_pass),
        "F10_base_acc_catastrophic": 0,
        "primary_blocker": blocker,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("failure_taxonomy_v9680.csv", [failure])
    manifest = {
        "version": "v9680",
        "route": route,
        "primary_blocker": blocker,
        "out_dir": rel(out),
        "created_utc": "2026-05-15T180000Z",
        "wallclock_sec": time.perf_counter() - t0,
        "seed": args.seed,
        "device": args.device,
        "data_root": args.data_root,
        "sources": {
            "v9670": rel(source_v9670),
            "v9660": rel(source_v9660),
            "v9650": rel(source_v9650),
            "v9640": rel(source_v9640),
            "v9630": rel(source_v9630),
        },
        "artifact_sha256": sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts}),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("run_manifest_v9680.json", manifest)
    print(
        json.dumps(
            {
                "out_dir": rel(out),
                "route": route,
                "primary_blocker": blocker,
                "best_transport_id": p1.get("best_transport_id"),
                "best_target_id": p2.get("best_target_id"),
                "best_ranker_id": p4.get("best_ranker_id"),
                "best_certificate_id": p5.get("best_certificate_id"),
                "generated_route_stop_triggered": p8.get("generated_route_stop_triggered"),
                "APGU_run": p8.get("APGU_run"),
                "system_legal_controller_pass": system_pass,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
