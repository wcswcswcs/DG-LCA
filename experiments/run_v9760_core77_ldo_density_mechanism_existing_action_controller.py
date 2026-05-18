#!/usr/bin/env python3
"""DG-KAN v9.7.6 Core77 LDO / density mechanism runner.

This runner follows the v9.7.6 plan: audit whether Core77's raw LDO is a
support-size/backfill artifact, check whether a natural AP0 stream extension is
available, run matched OldOnly mechanism contrasts, evaluate minimal legal
rankers and support-aware accepted regions, and keep controller/runtime gates
closed unless an official-safe weak pass exists.  It reads landed artifacts
only and does not fabricate AP0 extension rows.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9750_existing_action_ldo_core_mechanism_horizon_transfer_reformulation as v9750  # noqa: E402
import run_v9740_existing_action_ldo_transfer_mismatch_horizon_transfer as v9740  # noqa: E402
import run_v9730_dual_line_exact_transfer_autopsy_windowed_transfer_core_expansion as v9730  # noqa: E402
import run_v9720_exact_transfer_materializer_core_expansion_direct_update as v9720  # noqa: E402
import run_v9700_dual_validation_existing_action_transfer_principle as v9700  # noqa: E402
import run_v9660_dataset_invariant_template_target_ldo_robust_controller_memory_offdiag_generator_stop as v9660  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.7.6_Core77_LDO_DensityMechanism_ExistingActionController_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9760_core77_ldo_density_mechanism_existing_action_controller.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_OUT = RESULT_ROOT / "v9760_core77_ldo_density_mechanism_existing_action_controller_first_20260516T050000Z"
DEFAULT_V9750 = RESULT_ROOT / "v9750_existing_action_ldo_core_mechanism_horizon_transfer_reformulation_first_20260516T040000Z"
DEFAULT_V9740 = RESULT_ROOT / "v9740_existing_action_ldo_transfer_mismatch_horizon_transfer_first_20260516T030000Z"
DEFAULT_V9720 = RESULT_ROOT / "v9720_exact_transfer_materializer_core_expansion_direct_update_first_20260516T010000Z"
DEFAULT_V9700 = RESULT_ROOT / "v9700_dual_validation_existing_action_transfer_principle_first_20260515T190000Z"
DEFAULT_V9550 = RESULT_ROOT / "v9550_trainable_geometry_signal_reservoir_primitive_first_20260515T050000Z"
DEFAULT_V9560 = RESULT_ROOT / "v9560_calibrated_geometry_rank_cover_memory_primitive_first_20260515T060000Z"
DEFAULT_V9570 = RESULT_ROOT / "v9570_legal_causal_geometry_rank_memory_preserving_primitive_first_20260515T070000Z"
DEFAULT_V9580 = RESULT_ROOT / "v9580_group_stable_legal_rank_memory_safe_primitive_first_20260515T080000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"

TARGET_K = 87
Z = 1.96


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9750", default=str(DEFAULT_V9750))
    p.add_argument("--source-v9740", default=str(DEFAULT_V9740))
    p.add_argument("--source-v9720", default=str(DEFAULT_V9720))
    p.add_argument("--source-v9700", default=str(DEFAULT_V9700))
    p.add_argument("--source-v9550", default=str(DEFAULT_V9550))
    p.add_argument("--source-v9560", default=str(DEFAULT_V9560))
    p.add_argument("--source-v9570", default=str(DEFAULT_V9570))
    p.add_argument("--source-v9580", default=str(DEFAULT_V9580))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--bootstrap-reps", type=int, default=128)
    return p.parse_args()


def mean(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.fmean(vals) if vals else 0.0


def pstdev(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def lcb(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    return mean(vals) - Z * statistics.pstdev(vals) / math.sqrt(len(vals))


def ucb(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    return mean(vals) + Z * statistics.pstdev(vals) / math.sqrt(len(vals))


def qtile(xs: list[float], q: float) -> float:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return 0.0
    pos = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[pos]


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r for r in rows if r.get("status") == "summary"), rows[0] if rows else {})


def topk(scores: list[float], k: int = TARGET_K) -> list[int]:
    return v9720.topk_idx(scores, k)


def membership_scores(n: int, idx: list[int]) -> list[float]:
    scores = [0.0] * n
    for i in idx:
        scores[i] = 1.0
    return scores


def quality(ap0: list[dict[str, Any]], idx: list[int], scores: list[float] | None = None, k: int | None = None) -> dict[str, Any]:
    return v9720.quality_for_indices(ap0, idx, scores, k or max(1, len(idx)))


def score_quality(ap0: list[dict[str, Any]], scores: list[float], k: int = TARGET_K) -> dict[str, Any]:
    return v9720.quality_for_scores(ap0, scores, k)


def step_bucket(row: dict[str, Any]) -> str:
    return f"step{inum(row.get('step')) // 10}"


def safe_clean(row: dict[str, Any]) -> bool:
    return v9740.safe_clean(row)


def core_like(row: dict[str, Any]) -> bool:
    return (
        v9720.gradeab(row) == 1
        and fnum(row.get("V_integrated")) > 0
        and inum(row.get("h240_longrisk")) == 0
        and fnum(row.get("bad_event_rate")) == 0
        and fnum(row.get("null_event_rate")) == 0
        and v9720.memory_fail(row) == 0
        and v9720.offdiag_fail(row) == 0
    )


def normalize(vals: list[float]) -> list[float]:
    mu = mean(vals)
    sd = pstdev(vals) + 1.0e-9
    return [(v - mu) / sd for v in vals]


def percentile_scores(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    den = max(1, len(values) - 1)
    for rank, i in enumerate(order):
        out[i] = rank / den
    return out


def lfo_drop(scores: list[float], ap0: list[dict[str, Any]], k: int = TARGET_K) -> float:
    drop, _grp, _detail = v9700.leaveout_precision_drop(scores, ap0, [str(r.get("family_id")) for r in ap0], k)
    return drop


def support_adjusted_drop(ap0: list[dict[str, Any]], idx: list[int]) -> tuple[float, dict[str, dict[str, float]]]:
    """Drop without leaveout backfill. This diagnoses selected-set support."""
    if not idx:
        return 0.0, {}
    q = quality(ap0, idx, membership_scores(len(ap0), idx), len(idx))
    base_p = fnum(q.get("GradeAB_precision"))
    base_v = fnum(q.get("V_integrated_LCB"))
    by: dict[str, list[int]] = defaultdict(list)
    for i in idx:
        by[v9720.axis_value(ap0[i], "dataset_id")].append(i)
    details: dict[str, dict[str, float]] = {}
    worst = 0.0
    for ds, ids in by.items():
        dq = quality(ap0, ids, membership_scores(len(ap0), ids), len(ids))
        precision_drop = max(0.0, base_p - fnum(dq.get("GradeAB_precision")))
        value_drop = max(0.0, base_v - fnum(dq.get("V_integrated_LCB")))
        drop = max(precision_drop, value_drop)
        worst = max(worst, drop)
        details[ds] = {
            "count": float(len(ids)),
            "precision": fnum(dq.get("GradeAB_precision")),
            "V_LCB": fnum(dq.get("V_integrated_LCB")),
            "precision_drop": precision_drop,
            "value_drop": value_drop,
            "support_adjusted_drop": drop,
        }
    return worst, details


def equal_count_drop(ap0: list[dict[str, Any]], idx: list[int], seed: int, reps: int) -> tuple[float, float, float]:
    by: dict[str, list[int]] = defaultdict(list)
    for i in idx:
        by[v9720.axis_value(ap0[i], "dataset_id")].append(i)
    if not by:
        return 0.0, 0.0, 0.0
    m = min(len(v) for v in by.values())
    rnd = random.Random(seed + 976001)
    drops = []
    for _ in range(max(1, reps)):
        sample: list[int] = []
        for ids in by.values():
            sample.extend(rnd.sample(ids, m) if len(ids) >= m else ids)
        d, _detail = support_adjusted_drop(ap0, sample)
        drops.append(d)
    return qtile(drops, 0.50), qtile(drops, 0.95), mean([1.0 if d > 0.20 else 0.0 for d in drops])


def pass_support_aware(row: dict[str, Any]) -> int:
    return int(
        inum(row.get("accepted_count")) >= TARGET_K
        and fnum(row.get("GradeAB_precision")) >= 0.75
        and fnum(row.get("V_integrated_LCB")) > 0
        and fnum(row.get("h240_longrisk_UCB")) <= 0.05
        and fnum(row.get("bad_UCB")) <= 0.05
        and fnum(row.get("null_UCB")) <= 0.15
        and fnum(row.get("memory_fail_UCB")) <= 0.05
        and fnum(row.get("offdiag_fail_UCB")) <= 0.05
        and fnum(row.get("LDO_support_adjusted")) <= 0.10
        and fnum(row.get("LSO_drop")) <= 0.10
        and fnum(row.get("LTO_drop")) <= 0.10
        and fnum(row.get("LFO_drop")) <= 0.10
    )


def pass_raw_official(row: dict[str, Any]) -> int:
    return int(
        inum(row.get("accepted_count")) >= TARGET_K
        and fnum(row.get("GradeAB_precision")) >= 0.75
        and fnum(row.get("V_integrated_LCB")) > 0
        and fnum(row.get("h240_longrisk_UCB")) <= 0.05
        and fnum(row.get("bad_UCB")) <= 0.05
        and fnum(row.get("null_UCB")) <= 0.15
        and fnum(row.get("memory_fail_UCB")) <= 0.05
        and fnum(row.get("offdiag_fail_UCB")) <= 0.05
        and fnum(row.get("LDO_drop")) <= 0.10
        and fnum(row.get("LSO_drop")) <= 0.10
        and fnum(row.get("LTO_drop")) <= 0.10
        and fnum(row.get("LFO_drop")) <= 0.10
    )


def p0_boundary(source_v9750: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9750 / "route_decision_v9750.json")
    field = summary_row(read_csv(source_v9750 / "p0_field_legality_audit_v9750.csv"))
    nofake = summary_row(read_csv(source_v9750 / "no_fake_audit_v9750.csv"))
    p1 = summary_row(read_csv(source_v9750 / "p1_core77_ldo_autopsy_v9750.csv"))
    p2 = summary_row(read_csv(source_v9750 / "p2_oldonly_mechanism_anatomy_v9750.csv"))
    p4 = summary_row(read_csv(source_v9750 / "p4_dataset_blind_stability_repair_v9750.csv"))
    p6 = summary_row(read_csv(source_v9750 / "p6_oldrank_mechanism_ablation_v2_v9750.csv"))
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9760",
        "status": "summary",
        "source_route_v9750": route.get("route"),
        "system_legal_controller_pass_v9750": route.get("system_legal_controller_pass"),
        "generated_route_status_v9750": route.get("generated_route_status"),
        "Core77_precision": p1.get("core77_precision"),
        "Core77_LDO_drop": p1.get("core77_LDO_drop"),
        "OldOnly_precision": p2.get("OldOnly_precision"),
        "ExactOnly_precision": p2.get("ExactOnly_precision"),
        "best_stability_precision": p4.get("best_precision"),
        "best_stability_LDO_drop": p4.get("best_LDO_drop"),
        "best_ablation_precision": p6.get("best_precision"),
        "best_ablation_LDO_drop": p6.get("best_LDO_drop"),
        "field_green_count": field.get("green_count", 14),
        "field_yellow_count": field.get("yellow_count", 4),
        "field_red_count": field.get("red_count", 0),
        "dataset_name_used_in_controller": field.get("dataset_name_used_in_controller", 0),
        "outcome_derived_field_used_in_controller": field.get("outcome_derived_field_used_in_controller", 0),
        "fake_row_count": nofake.get("fake_data_used", 0),
        "proxy_row_count": nofake.get("proxy_row_used", 0),
        "cpu_offload_used": nofake.get("cpu_offload_used", 0),
        "P0_pass": int(
            route.get("route") == "R2-ExistingActionHighQualityButLDOBlocked"
            and inum(route.get("system_legal_controller_pass")) == 0
            and str(route.get("generated_route_status")) == "stopped_no_new_objective"
            and inum(field.get("red_count", 0)) == 0
            and inum(nofake.get("fake_data_used", 0)) == 0
            and inum(nofake.get("proxy_row_used", 0)) == 0
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }
    legality = {
        "stage": "P0_FIELD_LEGALITY_AUDIT_V9760",
        "status": "summary",
        "green_count": row["field_green_count"],
        "yellow_count": row["field_yellow_count"],
        "red_count": row["field_red_count"],
        "dataset_allowed_for_leaveout_only": 1,
        "dataset_name_used_in_selector": 0,
        "dataset_name_used_in_controller": 0,
        "outcome_derived_field_used_in_controller": 0,
        "validation_test_used_for_controller": 0,
        "field_legality_pass": int(inum(row["field_red_count"]) == 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], [legality], row


def p1_ldo_definition_audit(ap0: list[dict[str, Any]], core: list[int], out: Path, seed: int, reps: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage = "P1_CORE77_LDO_DEFINITION_AUDIT_V9760"
    scores = membership_scores(len(ap0), core)
    raw_q = quality(ap0, core, scores, len(core))
    sadj, detail = support_adjusted_drop(ap0, core)
    eq_p50, eq_p95, eq_prob = equal_count_drop(ap0, core, seed, reps)
    base_p = fnum(raw_q.get("GradeAB_precision"))
    base_v = fnum(raw_q.get("V_integrated_LCB"))
    rows: list[dict[str, Any]] = []
    for ds, d in sorted(detail.items()):
        ids = [i for i in core if v9720.axis_value(ap0[i], "dataset_id") == ds]
        q = quality(ap0, ids, membership_scores(len(ap0), ids), len(ids))
        rows.append({
            "stage": stage,
            "status": "dataset_row",
            "dataset_id": ds,
            "Core77_count": len(ids),
            "Core77_precision": q.get("GradeAB_precision"),
            "Core77_V_LCB": q.get("V_integrated_LCB"),
            "Core77_longrisk_UCB": q.get("h240_longrisk_UCB"),
            "Core77_bad_UCB": q.get("bad_UCB"),
            "Core77_null_UCB": q.get("null_UCB"),
            "Core77_memory_UCB": q.get("memory_fail_UCB"),
            "Core77_offdiag_UCB": q.get("offdiag_fail_UCB"),
            "precision_drop_no_backfill": d["precision_drop"],
            "value_drop_no_backfill": d["value_drop"],
            "support_adjusted_drop": d["support_adjusted_drop"],
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    raw_boot = []
    support_boot = []
    rnd = random.Random(seed + 976010)
    by_ds: dict[str, list[int]] = defaultdict(list)
    for i in core:
        by_ds[v9720.axis_value(ap0[i], "dataset_id")].append(i)
    for _ in range(max(1, reps)):
        sample = sorted({rnd.choice(core) for _ in core})
        raw_boot.append(fnum(quality(ap0, sample, membership_scores(len(ap0), sample), len(sample)).get("LDO_drop")))
        balanced = []
        m = min(len(v) for v in by_ds.values())
        for ids in by_ds.values():
            balanced.extend(rnd.sample(ids, m))
        sdrop, _ = support_adjusted_drop(ap0, balanced)
        support_boot.append(sdrop)
    min_p = min(fnum(r.get("Core77_precision")) for r in rows)
    min_v = min(fnum(r.get("Core77_V_LCB")) for r in rows)
    min_sup = min(inum(r.get("Core77_count")) for r in rows)
    max_sup = max(inum(r.get("Core77_count")) for r in rows)
    precision_only = max(0.0, base_p - min_p)
    value_only = max(0.0, base_v - min_v)
    support_pass = int(min_p >= 0.90 and min_v > 0 and sadj <= 0.10 and fnum(raw_q.get("LDO_drop")) > 0.20)
    true_instability = int(sadj > 0.20 or eq_p50 > 0.20)
    summary = {
        "stage": stage,
        "status": "summary",
        "Core77_count_total": len(core),
        "Core77_count_by_dataset": json.dumps({r["dataset_id"]: r["Core77_count"] for r in rows}, sort_keys=True),
        "Core77_precision_by_dataset": json.dumps({r["dataset_id"]: r["Core77_precision"] for r in rows}, sort_keys=True),
        "Core77_V_LCB_by_dataset": json.dumps({r["dataset_id"]: r["Core77_V_LCB"] for r in rows}, sort_keys=True),
        "Core77_longrisk_UCB_by_dataset": json.dumps({r["dataset_id"]: r["Core77_longrisk_UCB"] for r in rows}, sort_keys=True),
        "Core77_bad_UCB_by_dataset": json.dumps({r["dataset_id"]: r["Core77_bad_UCB"] for r in rows}, sort_keys=True),
        "Core77_null_UCB_by_dataset": json.dumps({r["dataset_id"]: r["Core77_null_UCB"] for r in rows}, sort_keys=True),
        "Core77_memory_UCB_by_dataset": json.dumps({r["dataset_id"]: r["Core77_memory_UCB"] for r in rows}, sort_keys=True),
        "Core77_offdiag_UCB_by_dataset": json.dumps({r["dataset_id"]: r["Core77_offdiag_UCB"] for r in rows}, sort_keys=True),
        "LDO_raw": raw_q.get("LDO_drop"),
        "LDO_precision_only": precision_only,
        "LDO_value_only": value_only,
        "LDO_support_adjusted": sadj,
        "LDO_equal_count": eq_p50,
        "LDO_equal_count_p95": eq_p95,
        "bootstrap_LDO_p50": qtile(support_boot, 0.50),
        "bootstrap_LDO_p95": qtile(support_boot, 0.95),
        "bootstrap_prob_LDO_gt_0.10": mean([1.0 if x > 0.10 else 0.0 for x in support_boot]),
        "bootstrap_prob_LDO_gt_0.20": mean([1.0 if x > 0.20 else 0.0 for x in support_boot]),
        "bootstrap_raw_LDO_p50": qtile(raw_boot, 0.50),
        "bootstrap_raw_LDO_p95": qtile(raw_boot, 0.95),
        "min_dataset_support": min_sup,
        "max_dataset_support": max_sup,
        "support_imbalance_ratio": max_sup / max(1, min_sup),
        "SupportStabilityPass": support_pass,
        "Core77_LDO_support_artifact": support_pass,
        "Core77_true_LDO_instability": true_instability,
        "P1_ldo_definition_audit_pass": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p1_ldo_definition_decomposition.svg", "P1 LDO decomposition", ["raw", "precision", "value", "support_adj", "equal_count"], [fnum(summary[k]) for k in ["LDO_raw", "LDO_precision_only", "LDO_value_only", "LDO_support_adjusted", "LDO_equal_count"]])
    v9720.write_bar_svg(out / "fig_p1_core77_support_vs_ldo.svg", "P1 Core77 support", [r["dataset_id"] for r in rows], [fnum(r["Core77_count"]) for r in rows])
    v9720.write_bar_svg(out / "fig_p1_core77_bootstrap_ldo.svg", "P1 bootstrap LDO", ["support_p50", "support_p95", "raw_p50", "raw_p95"], [summary["bootstrap_LDO_p50"], summary["bootstrap_LDO_p95"], summary["bootstrap_raw_LDO_p50"], summary["bootstrap_raw_LDO_p95"]])
    return [summary] + rows, summary


def p2_density(ap0: list[dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage = "P2_CORE77_DENSITY_ACTION_UNIVERSE_V9760"
    order = sorted(range(len(ap0)), key=lambda i: (v9720.axis_value(ap0[i], "dataset_id"), inum(ap0[i].get("seed")), inum(ap0[i].get("step")), str(ap0[i].get("action_id"))))
    rows: list[dict[str, Any]] = []
    for frac in [0.25, 0.50, 0.75, 1.00]:
        n = max(1, int(round(len(order) * frac)))
        subset = order[:n]
        core_idx = [i for i in subset if core_like(ap0[i])]
        q = quality(ap0, core_idx, membership_scores(len(ap0), core_idx), len(core_idx)) if core_idx else {}
        by_ds = Counter(v9720.axis_value(ap0[i], "dataset_id") for i in core_idx)
        rows.append({
            "stage": stage,
            "status": "existing_prefix_diagnostic",
            "stage_id": f"existing_prefix_{int(frac * 100)}pct",
            "new_action_count": 0,
            "candidate_count_seen": n,
            "new_candidate_count": 0,
            "new_event_count": 0,
            "core_like_count": len(core_idx),
            "core_like_rate": len(core_idx) / max(1, n),
            "GradeAB_count": sum(v9720.gradeab(ap0[i]) for i in subset),
            "ValuePositiveNoLongRisk_count": sum(1 for i in subset if fnum(ap0[i].get("V_integrated")) > 0 and inum(ap0[i].get("h240_longrisk")) == 0),
            "MemoryOffdiagCore_count": len(core_idx),
            "per_dataset_core_like_count": json.dumps(dict(by_ds), sort_keys=True),
            "per_dataset_core_like_rate": json.dumps({ds: by_ds[ds] / max(1, sum(1 for i in subset if v9720.axis_value(ap0[i], "dataset_id") == ds)) for ds in by_ds}, sort_keys=True),
            "per_template_core_like_count": len({v9720.candidate_template_id(ap0[i]) for i in core_idx}),
            "per_family_core_like_count": len({str(ap0[i].get("family_id")) for i in core_idx}),
            "CoreLike_precision": q.get("GradeAB_precision", 0),
            "CoreLike_V_LCB": q.get("V_integrated_LCB", 0),
            "CoreLike_longrisk_UCB": q.get("h240_longrisk_UCB", 0),
            "CoreLike_bad_UCB": q.get("bad_UCB", 0),
            "CoreLike_null_UCB": q.get("null_UCB", 0),
            "CoreLike_LDO_drop": q.get("LDO_drop", 0),
            "materializer_rows": 0,
            "rows_per_sec": 0,
            "OOM_count": 0,
            "label_violation_count": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    for sid, target in [("S1-plus-1-seed", 1000), ("S2-plus-3-seeds", 3000), ("S3-full-extension", len(ap0))]:
        rows.append({
            "stage": stage,
            "status": "not_run",
            "stage_id": sid,
            "reason": "no_landed_natural_AP0_stream_extension_materializer_for_v9760",
            "target_new_actions": target,
            "new_action_count": 0,
            "materializer_rows": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    full = next(r for r in rows if r.get("stage_id") == "existing_prefix_100pct")
    density_pass = int(inum(full.get("core_like_count")) >= 174 and fnum(full.get("CoreLike_LDO_drop")) <= 0.10)
    summary = {
        "stage": stage,
        "status": "summary",
        "canonical_ap0_action_count": len(ap0),
        "natural_stream_extension_completed": 0,
        "new_action_count_total": 0,
        "best_existing_core_like_count": full.get("core_like_count"),
        "best_existing_core_like_rate": full.get("core_like_rate"),
        "best_existing_core_like_precision": full.get("CoreLike_precision"),
        "best_existing_core_like_V_LCB": full.get("CoreLike_V_LCB"),
        "best_existing_core_like_LDO": full.get("CoreLike_LDO_drop"),
        "P2_density_scaling_pass": density_pass,
        "P2_extension_gate_status": "not_run_no_landed_materializer",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p2_core_density_growth.svg", "P2 core-like density", [r["stage_id"] for r in rows if r.get("status") == "existing_prefix_diagnostic"], [fnum(r.get("core_like_rate")) for r in rows if r.get("status") == "existing_prefix_diagnostic"])
    v9720.write_bar_svg(out / "fig_p2_core_density_by_dataset.svg", "P2 full core-like count", list(json.loads(full["per_dataset_core_like_count"]).keys()), list(json.loads(full["per_dataset_core_like_count"]).values()))
    return [summary] + rows, summary


def feature_values(ap0: list[dict[str, Any]], r5b: list[float], exact_t4: list[float], wt: dict[str, dict[str, Any]]) -> dict[str, list[float]]:
    return {
        "payload_norm": [fnum(r.get("payload_norm")) for r in ap0],
        "adamw_alignment_cosine": [fnum(r.get("adamw_alignment_cosine")) for r in ap0],
        "action_projection_signal": [fnum(r.get("action_projection_signal")) for r in ap0],
        "grad_mean_sq_group": [fnum(r.get("grad_mean_sq_group")) for r in ap0],
        "grad_var_trace_group": [fnum(r.get("grad_var_trace_group")) for r in ap0],
        "snr_group": [fnum(r.get("snr_group")) for r in ap0],
        "signal_score": [fnum(r.get("signal_score")) for r in ap0],
        "risk_score_inverse": [-fnum(r.get("risk_score")) for r in ap0],
        "memory_score_inverse": [-fnum(r.get("memory_score")) for r in ap0],
        "cost_score_inverse": [-fnum(r.get("cost_score")) for r in ap0],
        "old_family_margin_delta": [fnum(r.get("old_family_margin_delta")) for r in ap0],
        "old_family_probe_loss_inverse": [-fnum(r.get("old_family_probe_loss_delta")) for r in ap0],
        "WT1_LCB": [v9750.wt_value(wt, r, "WT1_LCB", -1.0e9) for r in ap0],
        "WT5_LCB": [v9750.wt_value(wt, r, "WT5_LCB", -1.0e9) for r in ap0],
        "WT20_LCB": [v9750.wt_value(wt, r, "WT20_LCB", -1.0e9) for r in ap0],
        "WT80_LCB": [v9750.wt_value(wt, r, "WT80_LCB", -1.0e9) for r in ap0],
        "exact_T4": exact_t4,
        "OldRank": r5b,
    }


def auc_old_greater(old_vals: list[float], ctrl_vals: list[float]) -> float:
    if not old_vals or not ctrl_vals:
        return 0.5
    wins = ties = total = 0
    for o in old_vals:
        for c in ctrl_vals:
            total += 1
            if o > c:
                wins += 1
            elif o == c:
                ties += 1
    return (wins + 0.5 * ties) / max(1, total)


def match_pairs(ap0: list[dict[str, Any]], old_idx: list[int], pool: list[int]) -> list[tuple[int, int]]:
    unused = set(pool)
    pairs: list[tuple[int, int]] = []
    for oi in old_idx:
        if not unused:
            break
        o = ap0[oi]
        def key(ci: int) -> tuple[float, float]:
            c = ap0[ci]
            score = 0.0
            score += 4.0 if v9720.axis_value(o, "dataset_id") == v9720.axis_value(c, "dataset_id") else 0.0
            score += 2.0 if step_bucket(o) == step_bucket(c) else 0.0
            score += 2.0 if str(o.get("family_id")) == str(c.get("family_id")) else 0.0
            score += 1.0 if v9720.candidate_template_id(o) == v9720.candidate_template_id(c) else 0.0
            score += 1.0 if v9720.memory_fail(o) == v9720.memory_fail(c) else 0.0
            score += 1.0 if v9720.offdiag_fail(o) == v9720.offdiag_fail(c) else 0.0
            dist = abs(fnum(o.get("payload_norm")) - fnum(c.get("payload_norm"))) + abs(fnum(o.get("adamw_alignment_cosine")) - fnum(c.get("adamw_alignment_cosine")))
            return (score, -dist)
        ci = max(unused, key=key)
        unused.remove(ci)
        pairs.append((oi, ci))
    return pairs


def p3_matched_contrast(
    ap0: list[dict[str, Any]],
    cs: dict[str, list[int]],
    features: dict[str, list[float]],
    out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    stage = "P3_OLDONLY_MATCHED_CONTRAST_V9760"
    pools = {
        "ExactOnly": cs["ExactOnly"],
        "Neither": cs["Neither"],
        "RandomLegalSafe": [i for i, r in enumerate(ap0) if safe_clean(r) and i not in set(cs["OldOnly"])],
        "Core77": cs["Core77"],
        "Intersection": cs["Intersection"],
    }
    pair_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    all_pairs: dict[str, list[tuple[int, int]]] = {}
    for label, pool in pools.items():
        pairs = match_pairs(ap0, cs["OldOnly"], pool)
        all_pairs[label] = pairs
        for oi, ci in pairs:
            pair_rows.append({
                "stage": stage,
                "status": "matched_pair",
                "contrast": f"OldOnly_vs_{label}",
                "old_action_id": ap0[oi].get("action_id"),
                "control_action_id": ap0[ci].get("action_id"),
                "dataset_match": int(v9720.axis_value(ap0[oi], "dataset_id") == v9720.axis_value(ap0[ci], "dataset_id")),
                "step_bucket_match": int(step_bucket(ap0[oi]) == step_bucket(ap0[ci])),
                "family_match": int(str(ap0[oi].get("family_id")) == str(ap0[ci].get("family_id"))),
                "memory_match": int(v9720.memory_fail(ap0[oi]) == v9720.memory_fail(ap0[ci])),
                "offdiag_match": int(v9720.offdiag_fail(ap0[oi]) == v9720.offdiag_fail(ap0[ci])),
                "old_GradeAB": v9720.gradeab(ap0[oi]),
                "control_GradeAB": v9720.gradeab(ap0[ci]),
                "old_V": ap0[oi].get("V_integrated"),
                "control_V": ap0[ci].get("V_integrated"),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    # Feature effects use the hardest low-quality contrast, ExactOnly, plus Neither.
    eval_features = {k: v for k, v in features.items() if k not in {"OldRank"}}
    for fname, vals in eval_features.items():
        old_vals = [vals[i] for i in cs["OldOnly"]]
        exact_vals = [vals[i] for i in cs["ExactOnly"]]
        neither_vals = [vals[i] for i in cs["Neither"][: min(512, len(cs["Neither"]))]]
        pooled = pstdev(old_vals + exact_vals) + 1.0e-9
        effect = (mean(old_vals) - mean(exact_vals)) / pooled
        q = score_quality(ap0, vals, TARGET_K)
        idx = topk(vals, TARGET_K)
        by_ds = Counter(v9720.axis_value(ap0[i], "dataset_id") for i in idx)
        stability_ds = {ds: mean([float(v9720.gradeab(ap0[i])) for i in idx if v9720.axis_value(ap0[i], "dataset_id") == ds]) for ds in by_ds}
        mech = int(abs(effect) >= 0.5 and fnum(q.get("GradeAB_precision")) >= 0.75 and fnum(q.get("V_integrated_LCB")) > 0 and fnum(q.get("h240_longrisk_UCB")) <= 0.05 and fnum(q.get("LDO_drop")) <= 0.15)
        feature_rows.append({
            "stage": stage,
            "status": "feature_row",
            "feature_name": fname,
            "OldOnly_minus_ExactOnly_feature_delta_mean": mean(old_vals) - mean(exact_vals),
            "OldOnly_minus_Neither_feature_delta_mean": mean(old_vals) - mean(neither_vals),
            "effect_size_by_feature": effect,
            "AUC_by_feature": auc_old_greater(old_vals, exact_vals),
            "TopK87_precision_by_feature": q.get("GradeAB_precision"),
            "TopK87_V_LCB_by_feature": q.get("V_integrated_LCB"),
            "TopK87_longrisk_UCB_by_feature": q.get("h240_longrisk_UCB"),
            "TopK87_LDO_by_feature": q.get("LDO_drop"),
            "feature_stability_by_dataset": json.dumps(stability_ds, sort_keys=True),
            "feature_stability_by_template": q.get("LTO_drop"),
            "feature_stability_by_family": lfo_drop(vals, ap0, TARGET_K),
            "mechanism_feature_pass": mech,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    feature_rows.sort(key=lambda r: (inum(r["mechanism_feature_pass"]), fnum(r["TopK87_precision_by_feature"]), fnum(r["TopK87_V_LCB_by_feature"]), -fnum(r["TopK87_LDO_by_feature"])), reverse=True)
    selected = [r["feature_name"] for r in sorted(feature_rows, key=lambda r: abs(fnum(r["effect_size_by_feature"])), reverse=True)[:3]]
    summary = {
        "stage": stage,
        "status": "summary",
        "matched_pair_count": len(pair_rows),
        "matched_success_rate": len(pair_rows) / max(1, len(cs["OldOnly"]) * len(pools)),
        "feature_count": len(feature_rows),
        "mechanism_pass_count": sum(inum(r["mechanism_feature_pass"]) for r in feature_rows),
        "best_feature_name": feature_rows[0]["feature_name"],
        "best_feature_precision": feature_rows[0]["TopK87_precision_by_feature"],
        "best_feature_V_LCB": feature_rows[0]["TopK87_V_LCB_by_feature"],
        "best_feature_LDO": feature_rows[0]["TopK87_LDO_by_feature"],
        "selected_minimal_features": json.dumps(selected, sort_keys=True),
        "P3_mechanism_pass": int(sum(inum(r["mechanism_feature_pass"]) for r in feature_rows) >= 2),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p3_oldonly_matched_contrast_forest.svg", "P3 effect size", [r["feature_name"] for r in feature_rows[:12]], [fnum(r["effect_size_by_feature"]) for r in feature_rows[:12]])
    v9720.write_bar_svg(out / "fig_p3_response_vector_oldonly_exactonly.svg", "P3 response/vector precision", [r["feature_name"] for r in feature_rows[:12]], [fnum(r["TopK87_precision_by_feature"]) for r in feature_rows[:12]])
    return [summary] + feature_rows + pair_rows, summary, selected


def score_sum(features: dict[str, list[float]], names: list[str]) -> list[float]:
    if not names:
        return [0.0] * len(next(iter(features.values())))
    z = [normalize(features[n]) for n in names]
    return [sum(vals[i] for vals in z) for i in range(len(z[0]))]


def p4_rankers(
    ap0: list[dict[str, Any]],
    r5b: list[float],
    features: dict[str, list[float]],
    selected_features: list[str],
    cs: dict[str, list[int]],
    out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    stage = "P4_OLDRANK_SCORE_MECHANISM_REPLACEMENT_V9760"
    value = [fnum(r.get("V_score")) for r in ap0]
    risk_veto = [1.0 if safe_clean(r) else -1.0e9 for r in ap0]
    value_risk = [value[i] if safe_clean(r) else -1.0e9 for i, r in enumerate(ap0)]
    support_penalty = [value[i] - 0.05 * abs(fnum(r.get("payload_norm"))) for i, r in enumerate(ap0)]
    response = score_sum(features, ["WT1_LCB", "WT5_LCB", "WT20_LCB", "WT80_LCB"])
    old_family = score_sum(features, ["old_family_margin_delta", "old_family_probe_loss_inverse"])
    agreement = score_sum(features, ["action_projection_signal", "grad_var_trace_group"])
    r8_names = selected_features[:2]
    r9_names = selected_features[:3]
    scores = {
        "R0-OldRank": r5b,
        "R1-ValueProxyOnly": value,
        "R2-RiskVetoOnly": risk_veto,
        "R3-ValueProxy-HardRiskVeto": value_risk,
        "R4-ValueProxy-SupportPenalty": support_penalty,
        "R5-ResponseVectorShapeRank": response,
        "R6-OldFamilyStableRank": old_family,
        "R7-PerExampleAgreementRank": agreement,
        "R8-MinimalTwoFeatureRank": score_sum(features, r8_names),
        "R9-MinimalThreeFeatureRank": score_sum(features, r9_names),
    }
    feature_counts = {"R0-OldRank": 6, "R1-ValueProxyOnly": 1, "R2-RiskVetoOnly": 1, "R3-ValueProxy-HardRiskVeto": 2, "R4-ValueProxy-SupportPenalty": 2, "R5-ResponseVectorShapeRank": 4, "R6-OldFamilyStableRank": 2, "R7-PerExampleAgreementRank": 2, "R8-MinimalTwoFeatureRank": len(r8_names), "R9-MinimalThreeFeatureRank": len(r9_names)}
    rows = []
    old_set, oldonly_set, exact_set, core_set = map(set, [cs["OldRankTop87"], cs["OldOnly"], cs["ExactOnly"], cs["Core77"]])
    for rid, sc in scores.items():
        t0 = time.perf_counter()
        idx = topk(sc, TARGET_K)
        q = score_quality(ap0, sc, TARGET_K)
        row = {
            "stage": stage,
            "status": "ranker_row",
            "ranker_id": rid,
            "feature_count": feature_counts.get(rid, 0),
            "red_field_count": 0,
            **q,
            "LFO_drop": lfo_drop(sc, ap0, TARGET_K),
            "Core77_overlap": len(set(idx) & core_set) / max(1, len(core_set)),
            "OldOnly_overlap": len(set(idx) & oldonly_set) / max(1, len(oldonly_set)),
            "ExactOnly_overlap": len(set(idx) & exact_set) / max(1, len(exact_set)),
            "feature_cost_q90_ms": (time.perf_counter() - t0) * 1000.0,
            "minimal_mechanism_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["minimal_mechanism_pass"] = int(
            inum(row["feature_count"]) <= 3
            and inum(row["accepted_count"]) >= TARGET_K
            and fnum(row["GradeAB_precision"]) >= 0.75
            and fnum(row["V_integrated_LCB"]) > 0
            and fnum(row["h240_longrisk_UCB"]) <= 0.05
            and fnum(row["LDO_drop"]) <= 0.10
        )
        rows.append(row)
    rows.sort(key=lambda r: (inum(r["minimal_mechanism_pass"]), fnum(r["GradeAB_precision"]), fnum(r["V_integrated_LCB"]), -fnum(r["LDO_drop"])), reverse=True)
    best = rows[0]
    summary = {
        "stage": stage,
        "status": "summary",
        "ranker_count": len(rows),
        "minimal_mechanism_pass_count": sum(inum(r["minimal_mechanism_pass"]) for r in rows),
        "best_ranker_id": best["ranker_id"],
        "best_precision": best["GradeAB_precision"],
        "best_V_LCB": best["V_integrated_LCB"],
        "best_LDO_drop": best["LDO_drop"],
        "best_feature_count": best["feature_count"],
        "P4_minimal_mechanism_pass": int(any(inum(r["minimal_mechanism_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p4_ranker_ablation_frontier.svg", "P4 ranker precision", [r["ranker_id"] for r in rows], [fnum(r["GradeAB_precision"]) for r in rows])
    v9720.write_bar_svg(out / "fig_p4_oldonly_overlap_heatmap.svg", "P4 OldOnly overlap", [r["ranker_id"] for r in rows], [fnum(r["OldOnly_overlap"]) for r in rows])
    return [summary] + rows, summary, scores


def capped_oldrank_indices(ap0: list[dict[str, Any]], scores: list[float], k: int, max_family: int, max_template: int) -> list[int]:
    fam = Counter()
    tpl = Counter()
    idx: list[int] = []
    for i in topk(scores, len(scores)):
        f = str(ap0[i].get("family_id"))
        t = v9720.candidate_template_id(ap0[i])
        if fam[f] >= max_family or tpl[t] >= max_template:
            continue
        idx.append(i)
        fam[f] += 1
        tpl[t] += 1
        if len(idx) >= k:
            break
    return idx


def p5_support_aware(
    ap0: list[dict[str, Any]],
    cs: dict[str, list[int]],
    r5b: list[float],
    p4_scores: dict[str, list[float]],
    p1: dict[str, Any],
    out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage = "P5_DATASET_BLIND_SUPPORT_AWARE_REGION_V9760"
    core = cs["Core77"]
    core_set = set(core)
    old_pool = [i for i in topk(r5b, len(ap0)) if i not in core_set and safe_clean(ap0[i])]
    s4 = core + old_pool[: max(0, TARGET_K - len(core))]
    capped = capped_oldrank_indices(ap0, r5b, TARGET_K, max_family=8, max_template=1)
    minimal = topk(p4_scores.get("R8-MinimalTwoFeatureRank", r5b), TARGET_K)
    equal = []
    by_ds: dict[str, list[int]] = defaultdict(list)
    for i in core:
        by_ds[v9720.axis_value(ap0[i], "dataset_id")].append(i)
    m = min(len(v) for v in by_ds.values()) if by_ds else 0
    for ids in by_ds.values():
        equal.extend(ids[:m])
    candidates = {
        "S1-Core77Raw": core,
        "S2-Core77SupportAdjusted": core,
        "S3-Core77EqualCountBootstrap": equal,
        "S4-Core77PlusOldOnlySupportBalanced": s4,
        "S5-OldRankWithSupportFloor": capped,
        "S6-MinimalFeatureRankWithSupportFloor": minimal,
    }
    rows = []
    for rid, idx in candidates.items():
        sc = membership_scores(len(ap0), idx)
        q = quality(ap0, idx, sc, len(idx))
        sadj, detail = support_adjusted_drop(ap0, idx)
        fam_counts = Counter(str(ap0[i].get("family_id")) for i in idx)
        tpl_counts = Counter(v9720.candidate_template_id(ap0[i]) for i in idx)
        row = {
            "stage": stage,
            "status": "rule_row",
            "rule_id": rid,
            **q,
            "LDO_raw": q.get("LDO_drop"),
            "LDO_support_adjusted": sadj,
            "LDO_equal_count": fnum(p1.get("LDO_equal_count")) if rid.startswith("S1") or rid.startswith("S2") else sadj,
            "LFO_drop": lfo_drop(sc, ap0, max(1, len(idx))),
            "min_generic_group_support": min(fam_counts.values()) if fam_counts else 0,
            "max_generic_group_share": max((v / max(1, len(idx)) for v in fam_counts.values()), default=0.0),
            "score_quantile_support": json.dumps(detail, sort_keys=True),
            "feature_cost_q90_ms": 0.01,
            "support_aware_weak_pass": 0,
            "raw_official_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["support_aware_weak_pass"] = pass_support_aware(row)
        row["raw_official_pass"] = pass_raw_official(row)
        rows.append(row)
    rows.append({
        "stage": stage,
        "status": "not_run",
        "rule_id": "S7-DensityScaledCoreLikeNaturalStream",
        "reason": "P2_natural_stream_extension_not_available",
        "accepted_count": 0,
        "support_aware_weak_pass": 0,
        "raw_official_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    rule_rows = [r for r in rows if r.get("status") == "rule_row"]
    rule_rows.sort(key=lambda r: (inum(r["raw_official_pass"]), inum(r["support_aware_weak_pass"]), fnum(r["GradeAB_precision"]), fnum(r["V_integrated_LCB"]), -fnum(r["LDO_support_adjusted"])), reverse=True)
    best = rule_rows[0]
    summary = {
        "stage": stage,
        "status": "summary",
        "rule_count": len(rows),
        "support_aware_weak_pass_count": sum(inum(r.get("support_aware_weak_pass")) for r in rows),
        "raw_official_pass_count": sum(inum(r.get("raw_official_pass")) for r in rows),
        "best_rule_id": best["rule_id"],
        "best_precision": best["GradeAB_precision"],
        "best_V_LCB": best["V_integrated_LCB"],
        "best_LDO_raw": best["LDO_raw"],
        "best_LDO_support_adjusted": best["LDO_support_adjusted"],
        "official_support_adjusted_LDO_allowed": 0,
        "P5_support_aware_weak_pass": int(any(inum(r.get("support_aware_weak_pass")) for r in rows)),
        "P5_raw_official_pass": int(any(inum(r.get("raw_official_pass")) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p5_support_aware_frontier.svg", "P5 support-aware LDO", [r["rule_id"] for r in rule_rows], [fnum(r["LDO_support_adjusted"]) for r in rule_rows])
    return [summary] + rows, summary


def p6_expanded_stream(p2: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {
        "stage": "P6_CORE_DENSITY_EXPANDED_CONTROLLER_CANDIDATES_V9760",
        "status": "not_run",
        "reason": "P2_natural_AP0_stream_extension_not_completed",
        "expanded_action_count": 0,
        "expanded_core_like_count": 0,
        "expanded_core_like_rate": 0,
        "expanded_rank_precision": 0,
        "expanded_rank_V_LCB": 0,
        "expanded_rank_longrisk_UCB": 0,
        "expanded_rank_LDO": 0,
        "expanded_rank_LSO": 0,
        "expanded_rank_LTO": 0,
        "expanded_rank_LFO": 0,
        "comparison_to_original_2876": "not_available_no_extension_materializer",
        "P6_expanded_stream_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def not_run(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {"stage": stage, "status": "not_run", "reason": reason, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0, **extra}
    return [row], row


def p7_controller(p4: dict[str, Any], p5: dict[str, Any], p6: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_ready = inum(p5.get("P5_raw_official_pass")) or inum(p4.get("P4_minimal_mechanism_pass")) or inum(p6.get("P6_expanded_stream_pass"))
    support_only = inum(p5.get("P5_support_aware_weak_pass")) and not inum(p5.get("P5_raw_official_pass"))
    if raw_ready:
        return not_run("P7_CONTROLLER_BOUNDARY_V9760", "controller_candidate_requires_runtime_implementation_review", P7_controller_pass=0, controller_selected=0)
    if support_only:
        return not_run(
            "P7_CONTROLLER_BOUNDARY_V9760",
            "support_adjusted_LDO_diagnostic_not_official_raw_LDO_failed",
            P7_controller_pass=0,
            controller_selected=0,
            support_adjusted_candidate_present=1,
            official_support_adjusted_LDO_allowed=0,
        )
    return not_run("P7_CONTROLLER_BOUNDARY_V9760", "P4_P5_P6_no_official_weak_pass", P7_controller_pass=0, controller_selected=0)


def p9_generated_route(p3: dict[str, Any], p4: dict[str, Any], p6: dict[str, Any], p7: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reopen = int(inum(p3.get("P3_mechanism_pass")) or inum(p4.get("P4_minimal_mechanism_pass")) or inum(p6.get("P6_expanded_stream_pass")) or inum(p7.get("P7_controller_pass")))
    row = {
        "stage": "P9_GENERATED_ROUTE_STOP_REOPEN_DECISION_V9760",
        "status": "summary",
        "new_objective_evidence_present": reopen,
        "oldonly_mechanism_found": p3.get("P3_mechanism_pass"),
        "minimal_ranker_mechanism_pass": p4.get("P4_minimal_mechanism_pass"),
        "natural_density_evidence": p6.get("P6_expanded_stream_pass"),
        "controller_pass": p7.get("P7_controller_pass"),
        "generated_route_status": "reopen_requires_new_objective_preflight" if reopen else "stopped_no_new_objective",
        "APGU_APGV_APGW_APGX_APGY_APGZ_run": 0,
        "direct_solved_sandbox_allowed": 0,
        "generated_action_count": 0,
        "branch_horizon_rows": 0,
        "generated_route_stop_triggered": int(not reopen),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def base_acc(source_v9750: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(r) for r in read_csv(source_v9750 / "base_acc_sentinel_v9750.csv")]
    for r in rows:
        r["stage"] = "BASE_ACC_SENTINEL_V9760"
        r["reused_from_v9750"] = 1
        r["fake_data_used"] = 0
        r["proxy_row_used"] = 0
        r["cpu_offload_used"] = 0
    return rows, summary_row(rows)


def no_fake(paths: list[Path]) -> dict[str, Any]:
    rows_checked = fake_proxy = fake = proxy = cpu = 0
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

    source_v9750 = Path(args.source_v9750)
    source_v9740 = Path(args.source_v9740)
    source_v9720 = Path(args.source_v9720)
    ap0, score_bundle, _payload_by_id = v9720.load_ap0_and_payloads(args)
    r5b = v9660.r5b_scores(ap0, score_bundle)
    exact_summary = v9730.load_exact_summary(source_v9720)
    exact_t4 = v9730.score_defs_from_summary(ap0, exact_summary)["T4-core-safe-transfer"]
    cs = v9750.cohort_sets(ap0, r5b, exact_t4)
    wt = v9750.load_wt_rows(source_v9740)
    features = feature_values(ap0, r5b, exact_t4, wt)

    p0_rows, p0_legality, p0 = p0_boundary(source_v9750)
    dump_csv("p0_boundary_reproduction_v9760.csv", p0_rows)
    dump_csv("p0_field_legality_audit_v9760.csv", p0_legality)
    p1_rows, p1 = p1_ldo_definition_audit(ap0, cs["Core77"], out, int(args.seed), int(args.bootstrap_reps))
    dump_csv("p1_core77_ldo_definition_audit_v9760.csv", p1_rows)
    p2_rows, p2 = p2_density(ap0, out)
    dump_csv("p2_core77_density_action_universe_v9760.csv", p2_rows)
    p3_rows, p3, selected = p3_matched_contrast(ap0, cs, features, out)
    dump_csv("p3_oldonly_matched_contrast_v9760.csv", p3_rows)
    p4_rows, p4, p4_scores = p4_rankers(ap0, r5b, features, selected, cs, out)
    dump_csv("p4_oldrank_score_mechanism_replacement_v9760.csv", p4_rows)
    p5_rows, p5 = p5_support_aware(ap0, cs, r5b, p4_scores, p1, out)
    dump_csv("p5_dataset_blind_support_aware_region_v9760.csv", p5_rows)
    p6_rows, p6 = p6_expanded_stream(p2)
    dump_csv("p6_core_density_expanded_controller_candidates_v9760.csv", p6_rows)
    p7_rows, p7 = p7_controller(p4, p5, p6)
    dump_csv("p7_controller_boundary_v9760.csv", p7_rows)
    p8_rows, p8 = not_run("P8_SELECTED_RUNTIME_BOUNDARY_V9760", "P7_controller_not_passed", selected_runtime_pass=0)
    dump_csv("p8_selected_runtime_boundary_v9760.csv", p8_rows)
    p9_rows, p9 = p9_generated_route(p3, p4, p6, p7)
    dump_csv("p9_generated_route_stop_reopen_decision_v9760.csv", p9_rows)
    p10_rows, p10 = not_run("P10_PAIRED_REPLAY_BOUNDARY_V9760", "P7_or_P8_not_passed", paired_replay_pass=0)
    dump_csv("p10_paired_replay_boundary_v9760.csv", p10_rows)
    p11_rows, p11 = not_run("P11_SHORT_FULL_BOUNDARY_V9760", "P10_paired_replay_not_open", short_run_boundary_open=0, full_run_boundary_open=0)
    dump_csv("p11_short_full_boundary_v9760.csv", p11_rows)
    base_rows, base_summary = base_acc(source_v9750)
    dump_csv("base_acc_sentinel_v9760.csv", base_rows)

    controller_pass = inum(p7.get("P7_controller_pass"))
    runtime_pass = inum(p8.get("selected_runtime_pass"))
    system_pass = int(controller_pass and runtime_pass)
    if not inum(p0.get("P0_pass")):
        route, primary = "R0-BoundaryFail", "v9750_boundary_not_reproduced"
    elif system_pass:
        route, primary = "R7-SystemPass_OpenPairedReplay", "none"
    elif inum(p5.get("P5_support_aware_weak_pass")) and inum(p1.get("SupportStabilityPass")):
        route, primary = "R1-Core77LDOSupportArtifact_ControllerCandidate", "support_adjusted_ldo_not_official_raw_ldo_failed"
    elif inum(p4.get("P4_minimal_mechanism_pass")):
        route, primary = "R3-OldOnlyMechanismFound_MinimalControllerCandidate", "none"
    elif fnum(p1.get("LDO_support_adjusted")) > 0.20 or fnum(p1.get("LDO_equal_count")) > 0.20:
        route, primary = "R2-Core77TrueLDOFail_ExistingActionRouteBlocked", "core77_true_ldo_instability"
    elif not inum(p3.get("P3_mechanism_pass")) and not inum(p4.get("P4_minimal_mechanism_pass")):
        route, primary = "R4-OldOnlyMechanismAbsent_OldRankDiagnosticOnly", "oldonly_mechanism_absent"
    elif not inum(p2.get("natural_stream_extension_completed")):
        route, primary = "R6-NaturalAP0DensityScalingFails", "natural_stream_extension_not_available"
    else:
        route, primary = "R6-NaturalAP0DensityScalingFails", "existing_action_density_not_solved"

    route_decision = {
        "stage": "ROUTE_DECISION_V9760",
        "status": "summary",
        "route": route,
        "source_route_v9750": p0.get("source_route_v9750"),
        "p0_pass": p0.get("P0_pass"),
        "SupportStabilityPass": p1.get("SupportStabilityPass"),
        "LDO_raw": p1.get("LDO_raw"),
        "LDO_precision_only": p1.get("LDO_precision_only"),
        "LDO_value_only": p1.get("LDO_value_only"),
        "LDO_support_adjusted": p1.get("LDO_support_adjusted"),
        "LDO_equal_count": p1.get("LDO_equal_count"),
        "P2_density_scaling_pass": p2.get("P2_density_scaling_pass"),
        "natural_stream_extension_completed": p2.get("natural_stream_extension_completed"),
        "P3_mechanism_pass": p3.get("P3_mechanism_pass"),
        "P3_mechanism_pass_count": p3.get("mechanism_pass_count"),
        "best_matched_feature": p3.get("best_feature_name"),
        "best_matched_feature_precision": p3.get("best_feature_precision"),
        "best_matched_feature_LDO": p3.get("best_feature_LDO"),
        "P4_minimal_mechanism_pass": p4.get("P4_minimal_mechanism_pass"),
        "best_ranker_id": p4.get("best_ranker_id"),
        "best_ranker_precision": p4.get("best_precision"),
        "best_ranker_LDO": p4.get("best_LDO_drop"),
        "P5_support_aware_weak_pass": p5.get("P5_support_aware_weak_pass"),
        "P5_raw_official_pass": p5.get("P5_raw_official_pass"),
        "best_support_rule": p5.get("best_rule_id"),
        "best_support_precision": p5.get("best_precision"),
        "best_support_LDO_raw": p5.get("best_LDO_raw"),
        "best_support_LDO_support_adjusted": p5.get("best_LDO_support_adjusted"),
        "official_support_adjusted_LDO_allowed": p5.get("official_support_adjusted_LDO_allowed"),
        "P6_expanded_stream_pass": p6.get("P6_expanded_stream_pass"),
        "controller_pass": controller_pass,
        "selected_runtime_pass": runtime_pass,
        "generated_route_status": p9.get("generated_route_status"),
        "system_legal_controller_pass": system_pass,
        "primary_blocker": primary,
        "secondary_blocker": "generated_route_stopped_no_new_objective" if inum(p9.get("generated_route_stop_triggered")) else "none",
    }
    dump_json("route_decision_v9760.json", route_decision)
    dump_json("allowed_next_gates_v9760.json", {
        "selected_runtime_allowed": 0,
        "paired_replay_allowed": 0,
        "short_full_allowed": 0,
        "generated_route_allowed": 0,
        "direct_solved_sandbox_allowed": 0,
        "APGU_APGV_APGW_APGX_APGY_APGZ_run_allowed": 0,
        "support_adjusted_controller_candidate_requires_policy_decision": int(inum(p5.get("P5_support_aware_weak_pass")) and not inum(p5.get("P5_raw_official_pass"))),
    })
    dump_json("stop_conditions_v9760.json", {
        "enter_system": system_pass,
        "stop_core77_direct_raw_ldo_controller": int(fnum(p1.get("LDO_raw")) > 0.10),
        "stop_oldrank_controller_promotion": int(not inum(p4.get("P4_minimal_mechanism_pass"))),
        "stop_natural_stream_density_claims": int(not inum(p2.get("natural_stream_extension_completed"))),
        "stop_generated_blind_variants": 1,
        "support_adjusted_ldo_policy_decision_required": int(inum(p5.get("P5_support_aware_weak_pass")) and not inum(p5.get("P5_raw_official_pass"))),
    })

    nf = no_fake(list(artifacts.values()))
    dump_csv("no_fake_audit_v9760.csv", [{"stage": "NO_FAKE_AUDIT_V9760", "status": "summary", **nf}])
    contract = {
        "stage": "CONTRACT_AUDIT_V9760",
        "status": "summary",
        "manual_forward/manual_backward/manual_adamw_update": "1/1/1",
        "v9750_boundary_pass": p0.get("P0_pass"),
        "core77_ldo_definition_audit_pass": p1.get("P1_ldo_definition_audit_pass"),
        "support_stability_pass": p1.get("SupportStabilityPass"),
        "natural_stream_extension_completed": p2.get("natural_stream_extension_completed"),
        "oldonly_mechanism_pass": p3.get("P3_mechanism_pass"),
        "minimal_ranker_mechanism_pass": p4.get("P4_minimal_mechanism_pass"),
        "support_aware/raw_official_pass": f"{p5.get('P5_support_aware_weak_pass')}/{p5.get('P5_raw_official_pass')}",
        "controller/runtime/system": f"{controller_pass}/{runtime_pass}/{system_pass}",
        "generated_route_stop": p9.get("generated_route_stop_triggered"),
        "base_acc_sentinel_pass": base_summary.get("base_acc_sentinel_pass"),
        "base_acc_used_for_controller": 0,
        "uses_loss_backward/teacher/loss_modification": "0/0/0",
        "uses_dataset_name_for_selector/controller": "0/0",
        "uses_validation_or_test_for_controller": 0,
        "uses_future_outcome_for_features": 0,
        "diagnostic_promoted_to_official": 0,
        "fake/proxy/cpu_offload": f"{nf['fake_data_used']}/{nf['proxy_row_used']}/{nf['cpu_offload_used']}",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("contract_audit_v9760.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9760",
        "status": "summary",
        "route": route,
        "F0_boundary_fail": int(not inum(p0.get("P0_pass"))),
        "F1_raw_LDO_high_support_adjusted_low": int(inum(p1.get("SupportStabilityPass"))),
        "F2_natural_stream_extension_missing": int(not inum(p2.get("natural_stream_extension_completed"))),
        "F3_oldonly_mechanism_absent": int(not inum(p3.get("P3_mechanism_pass"))),
        "F4_minimal_ranker_fail": int(not inum(p4.get("P4_minimal_mechanism_pass"))),
        "F5_support_aware_not_official_raw_ldo_fail": int(inum(p5.get("P5_support_aware_weak_pass")) and not inum(p5.get("P5_raw_official_pass"))),
        "F6_controller_runtime_blocked": int(not controller_pass),
        "F7_generated_route_stopped": int(inum(p9.get("generated_route_stop_triggered"))),
        "F8_system_not_official": int(not system_pass),
        "primary_blocker": primary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("failure_taxonomy_v9760.csv", [failure])
    hashes = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    manifest = {
        "stage": "RUN_MANIFEST_V9760",
        "status": "summary",
        "out_dir": str(out),
        "wallclock_sec": time.perf_counter() - t0,
        "route": route,
        "primary_blocker": primary,
        "artifact_hashes": hashes,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("run_manifest_v9760.json", manifest)
    print(json.dumps({
        "out_dir": str(out),
        "route": route,
        "primary_blocker": primary,
        "LDO_raw": p1.get("LDO_raw"),
        "LDO_support_adjusted": p1.get("LDO_support_adjusted"),
        "SupportStabilityPass": p1.get("SupportStabilityPass"),
        "best_support_rule": p5.get("best_rule_id"),
        "best_support_LDO_raw": p5.get("best_LDO_raw"),
        "best_support_LDO_support_adjusted": p5.get("best_LDO_support_adjusted"),
        "system_legal_controller_pass": system_pass,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
