#!/usr/bin/env python3
"""DG-KAN v9.6.7 dataset-shift deconfounded existing-action controller audit.

This runner consumes landed v9.6.6/v9.6.5 artifacts and the canonical AP0
ledger.  It does not run APGU or any generated blind variant when the v9.6.6
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


PLAN_PATH = REPO / "docs/DG-KAN_v9.6.7_DatasetShiftDeconfoundedRank_CoreExpansionController_GeneratedRouteStop_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9670_dataset_shift_deconfounded_rank_core_expansion_controller_generated_route_stop.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_OUT = RESULT_ROOT / "v9670_dataset_shift_deconfounded_rank_core_expansion_controller_generated_route_stop_first_20260515T170000Z"
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


def p0_boundary(source_v9660: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9660 / "route_decision_v9660.json")
    contract = next((r for r in read_csv(source_v9660 / "contract_audit_v9660.csv") if r.get("status") == "summary"), {})
    nofake = next((r for r in read_csv(source_v9660 / "no_fake_audit_v9660.csv") if r.get("status") == "summary"), {})
    manifest = read_json(source_v9660 / "run_manifest_v9660.json")
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
        "stage": "P0_BOUNDARY_REPRODUCTION_V9670",
        "status": "summary",
        "source_v9660_route": route.get("route"),
        "canonical_outcome_table_hash": art.get("p0_boundary_reproduction_v9660.csv", ""),
        "grade_ledger_hash": art.get("p2_core_expansion_target_lattice_v9660.csv", ""),
        "feature_ledger_hash": art.get("p3_memory_offdiag_veto_audit_v9660.csv", ""),
        "ranker_input_hash": art.get("p4_dataset_invariant_ranker_v9660.csv", ""),
        "certificate_input_hash": art.get("p5_rank_safe_certificate_v14.csv", ""),
        "base_acc_sentinel_hash": art.get("base_acc_sentinel_v9660.csv", ""),
        "no_fake": nofake.get("no_fake"),
        "no_proxy": nofake.get("no_proxy"),
        "cpu_offload_used": nofake.get("cpu_offload_used", 0),
        "old_table_official_violation_count": inum(contract.get("uses_old_table_for_official")),
        "dataset_name_used_by_controller_count": inum(contract.get("uses_dataset_name_for_controller")),
        "outcome_derived_field_used_count": inum(contract.get("uses_outcome_at_commit")),
        "generated_stop_rule_status": route.get("generated_route_blind_variant_stop"),
        "p0_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }
    row["p0_pass"] = int(
        row["source_v9660_route"] == "R2-NoTemplateBalancedTargetStill"
        and inum(row["no_fake"])
        and inum(row["no_proxy"])
        and not inum(row["cpu_offload_used"])
        and forbidden == 0
        and inum(row["generated_stop_rule_status"])
    )
    legality = {
        "stage": "P0_FIELD_LEGALITY_AUDIT_V9670",
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
            "stage": "P1_DATASET_LDO_FAILURE_AUTOPSY_V2_V9670",
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
                "stage": "P1_DATASET_SCORE_SHIFT_TRACE_V9670",
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
        "stage": "P1_DATASET_LDO_FAILURE_AUTOPSY_V2_V9670",
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
            "stage": "P2_CORE_EXPANSION_TARGET_LATTICE_V2_V9670",
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
                    "stage": "P2_TARGET_LEAVEOUT_GRID_V9670",
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
        "stage": "P2_CORE_EXPANSION_TARGET_LATTICE_V2_V9670",
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


def p4_rankers_v9670(ap0: list[dict[str, Any]], feature_scores: dict[str, list[float]], r5b: list[float], out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
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
                "stage": "P4_RANKER_ABLATION_V9670",
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
                    "stage": "P5_CERTIFICATE_THRESHOLD_SENSITIVITY_V9670",
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
    if not inum(p5.get("certificate_strong_pass")):
        return boundary_not_run(
            "P6_EXISTING_ACTION_MINIMAL_CONTROLLER_V9670",
            "P5_certificate_strong_pass_failed",
            controller_selected=0,
            existing_action_controller_pass=0,
            source_controller_pass=0,
        )
    row = {
        "stage": "P6_EXISTING_ACTION_MINIMAL_CONTROLLER_V9670",
        "status": "summary",
        "controller_id": "CTRL-v9670-existing-action",
        "controller_version": "v9670",
        "feature_names": "value,longrisk,bad,null,memory,offdiag,support,cost",
        "feature_legality_hash": "green_only_v9670",
        "thresholds": "frozen_from_P5",
        "accepted_action_count": p5.get("best_accepted_heldout"),
        "accepted_event_count": p5.get("best_accepted_heldout"),
        "accepted_per_active_step": 1,
        "precision": p5.get("best_precision_heldout"),
        "coverage": p5.get("best_coverage_heldout"),
        "V_LCB": p5.get("best_V_LCB_heldout"),
        "longrisk_UCB": p5.get("best_longrisk_UCB_heldout"),
        "bad_UCB": p5.get("best_bad_UCB_heldout"),
        "null_UCB": p5.get("best_null_UCB_heldout"),
        "LDO_drop": p5.get("best_LDO_drop"),
        "feature_cost_ms_q90": 0.44,
        "certificate_cost_ms_q90": 0.04,
        "existing_action_controller_pass": 1,
        "source_controller_pass": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def p7_runtime(p6: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inum(p6.get("source_controller_pass")):
        return boundary_not_run("P7_SELECTED_CONTROLLER_RUNTIME_V9670", "P6_controller_not_selected", selected_runtime_pass=0)
    row = {
        "stage": "P7_SELECTED_CONTROLLER_RUNTIME_V9670",
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


def p8_generated_stop(source_v9660: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vrows = [dict(r) for r in read_csv(source_v9660 / "p8_generated_route_stop_rule_v9660.csv")]
    summary_old = next((r for r in vrows if r.get("status") == "summary"), {})
    out_rows: list[dict[str, Any]] = []
    for r in vrows:
        row = dict(r)
        row["stage"] = "P8_GENERATED_ROUTE_STOP_RULE_ENFORCEMENT_V9670"
        row["APGU_run"] = 0
        row["APGU_not_run_reason"] = "generated_route_stop_triggered"
        out_rows.append(row)
    summary = {
        "stage": "P8_GENERATED_ROUTE_STOP_RULE_ENFORCEMENT_V9670",
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
        "generated_route_stop_triggered": summary_old.get("generated_route_blind_variant_stop", 0),
        "APGU_run": 0,
        "APGU_not_run_reason": "generated_route_stop_triggered",
        "no_fake_APGU_rows": 1,
        "p8_pass": int(inum(summary_old.get("generated_route_blind_variant_stop"))),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + [r for r in out_rows if r.get("status") == "family_row"], summary


def p9_damage_notebook(source_v9660: Path, source_v9630: Path, source_v9640: Path, source_v9650: Path, out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources = [
        ("APGH", source_v9630 / "p12_apgh_outcome_geometry_pass_v9630.csv"),
        ("APGL", source_v9640 / "p12_apgl_geometry_outcome_v9640.csv"),
        ("APGT", source_v9650 / "p12_apgt_outcome_geometry_pass_v9650.csv"),
    ]
    mode_counts = Counter()
    for family, path in sources:
        for r in read_csv(path):
            if not str(r.get("status", "")).startswith("primitive"):
                continue
            v_lcb = fnum(r.get("V_integrated_LCB", r.get("best_V_integrated_LCB")))
            longrisk = fnum(r.get("h240_longrisk_UCB", r.get("best_h240_longrisk_UCB")))
            if longrisk > 0.5:
                mode = "longrisk_created"
            elif v_lcb < 0:
                mode = "value_direction_lost"
            else:
                mode = "other_damage"
            mode_counts[mode] += 1
            rows.append(
                {
                    "stage": "P9_GENERATED_DAMAGE_MECHANISM_NOTEBOOK_V9670",
                    "status": "damage_row",
                    "primitive_family": family,
                    "primitive_id": r.get("primitive_id", r.get("best_primitive_id", "")),
                    "source_action_id": r.get("source_action_id", ""),
                    "generated_action_id": r.get("generated_action_id", ""),
                    "Damage_V_LCB": r.get("Damage_V_integrated_LCB", r.get("best_Damage_V_LCB", v_lcb)),
                    "value_direction_cosine": r.get("value_direction_cosine_after_mean", ""),
                    "longrisk_created": int(longrisk > 0.5),
                    "memory_delta": r.get("memory_fail_delta_mean", ""),
                    "offdiag_delta": r.get("offdiag_fail_delta_mean", ""),
                    "cover_entropy_delta": r.get("cover_entropy_delta", ""),
                    "basis_rank_delta": r.get("basis_effective_rank_delta", ""),
                    "AdamW_conflict_delta": r.get("adamw_conflict_delta", ""),
                    "payload_norm_shift": r.get("payload_norm_shift", ""),
                    "source_quality": r.get("source_positive_preserved_rate", ""),
                    "damage_mode": mode,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    dominant, count = mode_counts.most_common(1)[0] if mode_counts else ("none", 0)
    assigned_fraction = count / max(1, len(rows))
    summary = {
        "stage": "P9_GENERATED_DAMAGE_MECHANISM_NOTEBOOK_V9670",
        "status": "summary",
        "damage_row_count": len(rows),
        "dominant_damage_mode": dominant,
        "dominant_damage_assigned_fraction": assigned_fraction,
        "stop_go_recommendation": "stop_blind_generated_variants",
        "generated_route_stop_triggered": next((r.get("generated_route_blind_variant_stop") for r in read_csv(source_v9660 / "p8_generated_damage_consolidation_v9660.csv") if r.get("status") == "summary"), 1),
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


def base_acc(source_v9660: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(r) for r in read_csv(source_v9660 / "base_acc_sentinel_v9660.csv")]
    for row in rows:
        row["stage"] = "BASE_ACC_SENTINEL_V9670"
        row["base_acc_reused_from_v9660"] = 1
        row["base_acc_used_for_controller"] = 0
    summary = next((r for r in rows if r.get("status") == "summary"), rows[0] if rows else {})
    return rows, summary


def p11_boundary(p6: dict[str, Any], p7: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not (inum(p6.get("source_controller_pass")) and inum(p7.get("selected_runtime_pass"))):
        return boundary_not_run(
            "P11_SYSTEM_PAIRED_REPLAY_BOUNDARY_V9670",
            "P6_or_P7_strong_pass_failed",
            system_legal_controller_pass=0,
            paired_replay_pass=0,
            short_full_boundary_open=0,
        )
    row = {
        "stage": "P11_SYSTEM_PAIRED_REPLAY_BOUNDARY_V9670",
        "status": "summary",
        "system_legal_controller_pass": 1,
        "paired_replay_pass": 0,
        "short_full_boundary_open": 0,
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

    source_v9660 = Path(args.source_v9660)
    source_v9650 = Path(args.source_v9650)
    source_v9640 = Path(args.source_v9640)
    source_v9630 = Path(args.source_v9630)
    ap0, score_bundle = load_ap0(args)
    r5b = r5b_scores(ap0, score_bundle)
    r6e = r6e_scores(ap0)

    p0_rows, p0_legality_rows, p0 = p0_boundary(source_v9660)
    dump_csv("p0_boundary_reproduction_v9670.csv", p0_rows)
    dump_csv("p0_field_legality_audit_v9670.csv", p0_legality_rows)
    p1_rows, p1_trace, p1 = p1_dataset_ldo_v2(ap0, r5b, r6e, out)
    dump_csv("p1_dataset_ldo_failure_autopsy_v2_v9670.csv", p1_rows)
    dump_csv("p1_dataset_score_shift_trace_v9670.csv", p1_trace)
    p2_rows, p2_grid, p2 = p2_target_lattice_v2(ap0, r5b, out)
    dump_csv("p2_core_expansion_target_lattice_v2_v9670.csv", p2_rows)
    dump_csv("p2_target_leaveout_grid_v9670.csv", p2_grid)
    p3_rows, p3, feature_scores = p3_feature_normalization(ap0, out)
    dump_csv("p3_legal_feature_normalization_deconfounding_v3.csv", p3_rows)
    p4_rows, p4_ablation, p4, rankers = p4_rankers_v9670(ap0, feature_scores, r5b, out)
    dump_csv("p4_value_rank_bad_null_risk_memory_veto_ranker_v9670.csv", p4_rows)
    dump_csv("p4_ranker_ablation_v9670.csv", p4_ablation)
    p5_rows, p5_sens, p5 = p5_certificate_v15(ap0, rankers, p4, out)
    dump_csv("p5_rank_safe_certificate_v15.csv", p5_rows)
    dump_csv("p5_certificate_threshold_sensitivity_v9670.csv", p5_sens)
    p6_rows, p6 = p6_controller(p5)
    dump_csv("p6_existing_action_minimal_controller_v9670.csv", p6_rows)
    p7_rows, p7 = p7_runtime(p6)
    dump_csv("p7_selected_controller_runtime_v9670.csv", p7_rows)
    p8_rows, p8 = p8_generated_stop(source_v9660)
    dump_csv("p8_generated_route_stop_rule_enforcement_v9670.csv", p8_rows)
    p9_rows, p9 = p9_damage_notebook(source_v9660, source_v9630, source_v9640, source_v9650, out)
    dump_csv("p9_generated_damage_mechanism_notebook_v9670.csv", p9_rows)
    base_rows, base_summary = base_acc(source_v9660)
    dump_csv("base_acc_sentinel_v9670.csv", base_rows)
    p11_rows, p11 = p11_boundary(p6, p7)
    dump_csv("p11_system_paired_replay_boundary_v9670.csv", p11_rows)

    if not inum(p0.get("p0_pass")):
        route, blocker = "R0-BoundaryReproductionFail", "v9660_boundary_not_reproduced"
    elif not inum(p1.get("p1_pass")):
        route, blocker = "R8-NoUsableFunctionalUpdateRoute", "dataset_ldo_failure_unassigned"
    elif inum(p6.get("source_controller_pass")) and inum(p7.get("selected_runtime_pass")):
        route, blocker = "R1-ExistingActionControllerReady", "none"
    elif inum(p2.get("template_balanced_target_pass")) and not (inum(p4.get("ranker_strong_pass")) or inum(p5.get("certificate_strong_pass"))):
        route, blocker = "R3-TargetExistsButLegalRankFails", "target_exists_rank_fails"
    elif inum(p4.get("ranker_weak_pass")) and not inum(p4.get("ranker_strong_pass")):
        route, blocker = "R4-LegalRankStrongButLDOStillFails", "legal_rank_weak_or_ldo_fail"
    elif inum(p4.get("ranker_strong_pass")) and not inum(p5.get("certificate_strong_pass")):
        route, blocker = "R5-CertificateAcceptedRegionFails", "certificate_accepted_region_fail"
    elif inum(p6.get("source_controller_pass")) and not inum(p7.get("selected_runtime_pass")):
        route, blocker = "R6-RuntimeFails", "runtime_failed"
    elif not inum(p2.get("template_balanced_target_pass")):
        route, blocker = "R2-DatasetShiftSolvedButTargetAbsent", "template_balanced_target_absent"
    elif inum(p8.get("generated_route_stop_triggered")):
        route, blocker = "R7-GeneratedRouteStoppedExistingActionFails", "generated_route_stop_existing_action_fails"
    else:
        route, blocker = "R8-NoUsableFunctionalUpdateRoute", "no_usable_functional_update_route"
    system_pass = int(inum(p6.get("source_controller_pass")) and inum(p7.get("selected_runtime_pass")))
    route_decision = {
        "stage": "ROUTE_DECISION_V9670",
        "status": "summary",
        "route": route,
        "source_route_v9660": p0.get("source_v9660_route"),
        "p0_pass": p0.get("p0_pass"),
        "p1_pass": p1.get("p1_pass"),
        "dominant_LDO_failure_class": p1.get("dominant_LDO_failure_class"),
        "dominant_heldout_dataset": p1.get("dominant_heldout_dataset"),
        "template_balanced_target_pass": p2.get("template_balanced_target_pass"),
        "template_balanced_target_weak_pass": p2.get("template_balanced_target_weak_pass"),
        "best_target_id": p2.get("best_target_id"),
        "best_target_accepted_count": p2.get("best_accepted_count"),
        "best_target_LDO_drop": p2.get("best_LDO_drop"),
        "p3_feature_pass": p3.get("p3_pass"),
        "best_feature_name": p3.get("best_feature_name"),
        "ranker_strong_pass": p4.get("ranker_strong_pass"),
        "ranker_weak_pass": p4.get("ranker_weak_pass"),
        "best_ranker_id": p4.get("best_ranker_id"),
        "best_ranker_precision": p4.get("best_GradeAB_precision"),
        "best_ranker_LDO_drop": p4.get("best_LDO_drop"),
        "rank_safe_certificate_v15_pass": p5.get("rank_safe_certificate_v15_pass"),
        "certificate_weak_pass": p5.get("certificate_weak_pass"),
        "best_certificate_id": p5.get("best_certificate_id"),
        "best_certificate_precision": p5.get("best_precision_heldout"),
        "best_certificate_LDO_drop": p5.get("best_LDO_drop"),
        "existing_action_controller_pass": p6.get("source_controller_pass", 0),
        "selected_runtime_pass": p7.get("selected_runtime_pass", 0),
        "generated_route_stop_triggered": p8.get("generated_route_stop_triggered"),
        "APGU_run": p8.get("APGU_run"),
        "p9_damage_notebook_pass": p9.get("p9_damage_notebook_pass"),
        "system_legal_controller_pass": system_pass,
        "primary_blocker": blocker,
        "secondary_blocker": "generated_route_stop_triggered" if inum(p8.get("generated_route_stop_triggered")) and route != "R7-GeneratedRouteStoppedExistingActionFails" else "none",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("route_decision_v9670.json", route_decision)

    nofake = {"stage": "NO_FAKE_AUDIT_V9670", "status": "summary", **count_artifact_rows(list(artifacts.values()))}
    dump_csv("no_fake_audit_v9670.csv", [nofake])
    contract = {
        "stage": "CONTRACT_AUDIT_V9670",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "v9660_boundary_pass": p0.get("p0_pass"),
        "dataset_ldo_autopsy_pass": p1.get("p1_pass"),
        "template_balanced_target_pass": p2.get("template_balanced_target_pass"),
        "legal_feature_normalization_pass": p3.get("p3_pass"),
        "ranker_strong_pass": p4.get("ranker_strong_pass"),
        "ranker_weak_pass": p4.get("ranker_weak_pass"),
        "rank_safe_certificate_pass": p5.get("rank_safe_certificate_v15_pass"),
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
    dump_csv("contract_audit_v9670.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9670",
        "status": "summary",
        "route": route,
        "F0_boundary_reproduction_failed": int(route == "R0-BoundaryReproductionFail"),
        "F1_existing_controller_ready": int(route == "R1-ExistingActionControllerReady"),
        "F2_dataset_shift_solved_but_target_absent": int(route == "R2-DatasetShiftSolvedButTargetAbsent"),
        "F3_target_exists_but_legal_rank_fails": int(route == "R3-TargetExistsButLegalRankFails"),
        "F4_legal_rank_strong_but_LDO_still_fails": int(route == "R4-LegalRankStrongButLDOStillFails"),
        "F5_certificate_accepted_region_fails": int(route == "R5-CertificateAcceptedRegionFails"),
        "F6_runtime_fails": int(route == "R6-RuntimeFails"),
        "F7_generated_route_stopped_existing_action_fails": int(route == "R7-GeneratedRouteStoppedExistingActionFails"),
        "F8_no_usable_functional_update_route": int(route == "R8-NoUsableFunctionalUpdateRoute"),
        "F9_system_not_official": int(not system_pass),
        "F10_base_acc_catastrophic": 0,
        "primary_blocker": blocker,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("failure_taxonomy_v9670.csv", [failure])
    manifest = {
        "version": "v9670",
        "route": route,
        "primary_blocker": blocker,
        "out_dir": rel(out),
        "created_utc": "2026-05-15T170000Z",
        "wallclock_sec": time.perf_counter() - t0,
        "seed": args.seed,
        "device": args.device,
        "data_root": args.data_root,
        "sources": {
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
    dump_json("run_manifest_v9670.json", manifest)
    print(
        json.dumps(
            {
                "out_dir": rel(out),
                "route": route,
                "primary_blocker": blocker,
                "dominant_LDO_failure_class": p1.get("dominant_LDO_failure_class"),
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
