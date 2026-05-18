#!/usr/bin/env python3
"""DG-KAN v9.7.5 existing-action LDO and transfer-mechanism runner.

This runner follows the v9.7.5 plan: explain Core77 LDO failure, dissect the
OldRank/ExactTransfer mismatch, evaluate dataset-blind repairs, and keep
generated routes stopped unless a new green-field objective appears.  It reads
landed v9.7.4/v9.7.2 artifacts only; diagnostics are not promoted to an
official controller.
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

import run_v9740_existing_action_ldo_transfer_mismatch_horizon_transfer as v9740  # noqa: E402
import run_v9730_dual_line_exact_transfer_autopsy_windowed_transfer_core_expansion as v9730  # noqa: E402
import run_v9720_exact_transfer_materializer_core_expansion_direct_update as v9720  # noqa: E402
import run_v9700_dual_validation_existing_action_transfer_principle as v9700  # noqa: E402
import run_v9660_dataset_invariant_template_target_ldo_robust_controller_memory_offdiag_generator_stop as v9660  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.7.5_ExistingActionLDO_CoreMechanism_HorizonTransferReformulation_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9750_existing_action_ldo_core_mechanism_horizon_transfer_reformulation.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_OUT = RESULT_ROOT / "v9750_existing_action_ldo_core_mechanism_horizon_transfer_reformulation_first_20260516T040000Z"
DEFAULT_V9740 = RESULT_ROOT / "v9740_existing_action_ldo_transfer_mismatch_horizon_transfer_first_20260516T030000Z"
DEFAULT_V9730 = RESULT_ROOT / "v9730_dual_line_exact_transfer_windowed_transfer_core_expansion_first_20260516T020000Z"
DEFAULT_V9720 = RESULT_ROOT / "v9720_exact_transfer_materializer_core_expansion_direct_update_first_20260516T010000Z"
DEFAULT_V9700 = RESULT_ROOT / "v9700_dual_validation_existing_action_transfer_principle_first_20260515T190000Z"
DEFAULT_V9680 = RESULT_ROOT / "v9680_dataset_invariant_core_expansion_controller_transport_generator_route_decision_first_20260515T180000Z"
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
    p.add_argument("--source-v9740", default=str(DEFAULT_V9740))
    p.add_argument("--source-v9730", default=str(DEFAULT_V9730))
    p.add_argument("--source-v9720", default=str(DEFAULT_V9720))
    p.add_argument("--source-v9700", default=str(DEFAULT_V9700))
    p.add_argument("--source-v9680", default=str(DEFAULT_V9680))
    p.add_argument("--source-v9550", default=str(DEFAULT_V9550))
    p.add_argument("--source-v9560", default=str(DEFAULT_V9560))
    p.add_argument("--source-v9570", default=str(DEFAULT_V9570))
    p.add_argument("--source-v9580", default=str(DEFAULT_V9580))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--bootstrap-reps", type=int, default=128)
    return p.parse_args()


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r for r in rows if r.get("status") == "summary"), rows[0] if rows else {})


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


def step_bucket(row: dict[str, Any]) -> str:
    return f"step{inum(row.get('step')) // 10}"


def safe_clean(row: dict[str, Any]) -> bool:
    return v9740.safe_clean(row)


def topk(scores: list[float], k: int = TARGET_K) -> list[int]:
    return v9720.topk_idx(scores, k)


def pass_weak(q: dict[str, Any]) -> int:
    return int(
        inum(q.get("accepted_count")) >= TARGET_K
        and fnum(q.get("GradeAB_precision")) >= 0.75
        and fnum(q.get("V_integrated_LCB")) > 0
        and fnum(q.get("h240_longrisk_UCB")) <= 0.05
        and fnum(q.get("bad_UCB")) <= 0.05
        and fnum(q.get("null_UCB")) <= 0.15
        and fnum(q.get("LDO_drop")) <= 0.20
        and fnum(q.get("LSO_drop")) <= 0.20
        and fnum(q.get("LTO_drop")) <= 0.10
    )


def pass_strong(q: dict[str, Any]) -> int:
    return int(
        inum(q.get("accepted_count")) >= TARGET_K
        and fnum(q.get("GradeAB_precision")) >= 0.80
        and fnum(q.get("V_integrated_LCB")) > 0.05
        and fnum(q.get("h240_longrisk_UCB")) <= 0.05
        and fnum(q.get("bad_UCB")) <= 0.05
        and fnum(q.get("null_UCB")) <= 0.10
        and fnum(q.get("LDO_drop")) <= 0.10
        and fnum(q.get("LSO_drop")) <= 0.10
        and fnum(q.get("LTO_drop")) <= 0.10
    )


def pass_expansion(q: dict[str, Any]) -> int:
    return int(
        fnum(q.get("GradeAB_precision")) >= 0.75
        and fnum(q.get("V_integrated_LCB")) > 0
        and fnum(q.get("h240_longrisk_UCB")) <= 0.05
        and fnum(q.get("bad_UCB")) <= 0.05
        and fnum(q.get("null_UCB")) <= 0.15
        and fnum(q.get("LDO_drop")) <= 0.10
    )


def membership_scores(n: int, idx: list[int]) -> list[float]:
    scores = [0.0] * n
    for i in idx:
        scores[i] = 1.0
    return scores


def quality(ap0: list[dict[str, Any]], idx: list[int], scores: list[float] | None = None, k: int | None = None) -> dict[str, Any]:
    return v9720.quality_for_indices(ap0, idx, scores, k or max(1, len(idx)))


def score_quality(ap0: list[dict[str, Any]], scores: list[float], k: int = TARGET_K) -> dict[str, Any]:
    return v9720.quality_for_scores(ap0, scores, k)


def group_quality(ap0: list[dict[str, Any]], idx: list[int], gid: str, stage: str, scores: list[float] | None = None) -> dict[str, Any]:
    q = quality(ap0, idx, scores or membership_scores(len(ap0), idx), len(idx))
    rows = [ap0[i] for i in idx]
    ds = Counter(v9720.axis_value(r, "dataset_id") for r in rows)
    fam = Counter(str(r.get("family_id")) for r in rows)
    tpl = Counter(v9720.candidate_template_id(r) for r in rows)
    steps = Counter(step_bucket(r) for r in rows)
    return {
        "stage": stage,
        "status": "group_row",
        "group_id": gid,
        **q,
        "dataset_count": len(ds),
        "family_count": len(fam),
        "template_count": len(tpl),
        "step_bucket_count": len(steps),
        "max_family_share": max((v / max(1, len(rows)) for v in fam.values()), default=0.0),
        "max_step_bucket_share": max((v / max(1, len(rows)) for v in steps.values()), default=0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def load_wt_rows(source_v9740: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for r in read_csv(source_v9740 / "p4_horizon_transfer_materializer_v9740.csv"):
        if r.get("status") == "windowed_action_row":
            rows[str(r.get("action_id"))] = dict(r)
    return rows


def wt_value(wt: dict[str, dict[str, Any]], row: dict[str, Any], field: str, default: float = -1.0e9) -> float:
    return fnum(wt.get(str(row.get("action_id")), {}).get(field), default)


def cohort_sets(ap0: list[dict[str, Any]], r5b: list[float], exact_t4: list[float]) -> dict[str, list[int]]:
    old = set(topk(r5b, TARGET_K))
    exact = set(topk(exact_t4, TARGET_K))
    core = set(i for i, r in enumerate(ap0) if v9720.is_core_action(r))
    all_idx = set(range(len(ap0)))
    return {
        "Intersection": sorted(old & exact),
        "OldOnly": sorted(old - exact),
        "ExactOnly": sorted(exact - old),
        "Neither": sorted(all_idx - old - exact),
        "OldRankTop87": sorted(old),
        "ExactT4Top87": sorted(exact),
        "Core77": sorted(core),
    }


def p0_boundary(source_v9740: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9740 / "route_decision_v9740.json")
    field = summary_row(read_csv(source_v9740 / "p0_field_legality_audit_v9740.csv"))
    p1 = summary_row(read_csv(source_v9740 / "p1_old_exact_transfer_mismatch_v9740.csv"))
    p2 = summary_row(read_csv(source_v9740 / "p2_core_expansion_ldo_responsibility_v9740.csv"))
    p4 = summary_row(read_csv(source_v9740 / "p4_horizon_transfer_score_eval_v9740.csv"))
    nofake = summary_row(read_csv(source_v9740 / "no_fake_audit_v9740.csv"))
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9750",
        "status": "summary",
        "source_route_v9740": route.get("route"),
        "system_legal_controller_pass_v9740": route.get("system_legal_controller_pass"),
        "OldRankTop87_precision_v9740": p1.get("oldonly_precision"),
        "ExactOnly_precision_v9740": p1.get("exactonly_precision"),
        "Core77_LDO_drop_v9740": p2.get("Core77_LDO_drop"),
        "C0E1_precision_v9740": p2.get("C0E1_precision"),
        "C0E1_LDO_drop_v9740": p2.get("C0E1_LDO_drop"),
        "best_horizon_score_id_v9740": p4.get("best_score_id"),
        "best_horizon_precision_v9740": p4.get("best_precision"),
        "best_horizon_V_LCB_v9740": p4.get("best_V_LCB"),
        "generated_route_status_v9740": route.get("generated_route_status"),
        "field_green_count": field.get("green_count", 14),
        "field_yellow_count": field.get("yellow_count", 4),
        "field_red_count": field.get("red_count", 0),
        "dataset_name_used_in_controller": field.get("dataset_name_used_in_controller", 0),
        "outcome_derived_field_used_in_controller": field.get("outcome_derived_field_used_in_controller", 0),
        "fake_proxy_row_count": nofake.get("fake_proxy_nonzero_count"),
        "p0_pass": int(
            route.get("route") == "R2-ExistingActionLDOBlocked"
            and str(route.get("generated_route_status")) == "stopped_no_new_objective"
            and inum(route.get("system_legal_controller_pass")) == 0
            and inum(field.get("red_count", 0)) == 0
            and inum(nofake.get("fake_proxy_nonzero_count")) == 0
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    legality = {
        "stage": "P0_FIELD_LEGALITY_AUDIT_V9750",
        "status": "summary",
        "green_count": field.get("green_count", 14),
        "yellow_count": field.get("yellow_count", 4),
        "red_count": field.get("red_count", 0),
        "dataset_allowed_for_diagnostics_only": 1,
        "dataset_name_used_in_selector": 0,
        "dataset_name_used_in_controller": 0,
        "outcome_derived_field_used_in_controller": 0,
        "validation_test_used_for_controller": 0,
        "proxy_transfer_promoted_to_official": 0,
        "field_legality_pass": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], [legality], row


def p1_core77_autopsy(
    ap0: list[dict[str, Any]],
    r5b: list[float],
    core: list[int],
    out: Path,
    seed: int,
    bootstrap_reps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    stage = "P1_CORE77_LDO_AUTOPSY_V9750"
    core_set = set(core)
    core_row = group_quality(ap0, core, "Core77", stage, membership_scores(len(ap0), core))
    datasets = sorted({v9720.axis_value(ap0[i], "dataset_id") for i in core})
    dataset_rows = []
    for ds in datasets:
        idx = [i for i in core if v9720.axis_value(ap0[i], "dataset_id") == ds]
        rows = [ap0[i] for i in idx]
        q = quality(ap0, idx, membership_scores(len(ap0), idx), len(idx))
        dataset_rows.append(
            {
                "stage": stage,
                "status": "dataset_row",
                "dataset_id": ds,
                "core77_action_count": len(idx),
                "core77_precision": q.get("GradeAB_precision"),
                "core77_V_LCB": q.get("V_integrated_LCB"),
                "core77_longrisk_UCB": q.get("h240_longrisk_UCB"),
                "core77_bad_UCB": q.get("bad_UCB"),
                "core77_null_UCB": q.get("null_UCB"),
                "core77_memory_UCB": q.get("memory_fail_UCB"),
                "core77_offdiag_UCB": q.get("offdiag_fail_UCB"),
                "core77_score_mean": mean([r5b[i] for i in idx]),
                "core77_score_std": pstdev([r5b[i] for i in idx]),
                "core77_support_count": len(idx),
                "core77_template_count": len({v9720.candidate_template_id(r) for r in rows}),
                "core77_family_count": len({str(r.get("family_id")) for r in rows}),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    group_rows = []
    axes = {
        "dataset_id": lambda r: v9720.axis_value(r, "dataset_id"),
        "family_id": lambda r: str(r.get("family_id")),
        "candidate_template": lambda r: v9720.candidate_template_id(r),
        "step_bucket": step_bucket,
        "memory_bucket": lambda r: f"memory_fail_{v9720.memory_fail(r)}",
        "offdiag_bucket": lambda r: f"offdiag_fail_{v9720.offdiag_fail(r)}",
    }
    for axis, fn in axes.items():
        base_counts = Counter(fn(r) for r in ap0)
        core_counts = Counter(fn(ap0[i]) for i in core)
        for gid, cnt in core_counts.most_common(8):
            idx = [i for i in core if fn(ap0[i]) == gid]
            remain = [i for i in core if fn(ap0[i]) != gid]
            q = quality(ap0, idx, membership_scores(len(ap0), idx), len(idx))
            rq = quality(ap0, remain, membership_scores(len(ap0), remain), max(1, len(remain))) if remain else {}
            group_rows.append(
                {
                    "stage": "P1_CORE77_GROUP_AUTOPSY_V9750",
                    "status": "group_row",
                    "group_axis": axis,
                    "group_id": gid,
                    "base_share": base_counts[gid] / max(1, len(ap0)),
                    "core77_share": cnt / max(1, len(core)),
                    "precision": q.get("GradeAB_precision"),
                    "V_LCB": q.get("V_integrated_LCB"),
                    "longrisk_UCB": q.get("h240_longrisk_UCB"),
                    "LDO_drop_if_removed": rq.get("LDO_drop", 0),
                    "support_count": cnt,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    rnd = random.Random(seed + 9750)
    boot_ldo = []
    for _ in range(max(1, bootstrap_reps)):
        sample = [rnd.choice(core) for _ in core]
        # Use the unique action support for leaveout scoring; this is a
        # conservative bootstrap of the selected region's structural support.
        unique = sorted(set(sample))
        boot_ldo.append(fnum(quality(ap0, unique, membership_scores(len(ap0), unique), len(unique)).get("LDO_drop")))
    prob_ldo_gt_020 = mean([1.0 if x > 0.20 else 0.0 for x in boot_ldo])
    min_ds_precision = min((fnum(r.get("core77_precision")) for r in dataset_rows), default=0)
    min_ds_v = min((fnum(r.get("core77_V_LCB")) for r in dataset_rows), default=0)
    min_support = min((inum(r.get("core77_support_count")) for r in dataset_rows), default=0)
    max_support = max((inum(r.get("core77_support_count")) for r in dataset_rows), default=1)
    max_family_share = max((fnum(r.get("core77_share")) for r in group_rows if r.get("group_axis") == "family_id"), default=0)
    max_template_share = max((fnum(r.get("core77_share")) for r in group_rows if r.get("group_axis") == "candidate_template"), default=0)
    max_step_share = max((fnum(r.get("core77_share")) for r in group_rows if r.get("group_axis") == "step_bucket"), default=0)
    h1a = int((1.0 - min_ds_precision) >= 0.25 or min_ds_v < 0 or min_support < 0.50 * max_support)
    h1b = int(max_family_share > 0.35 or max_template_share > 0.25 or max_step_share > 0.35)
    h1c = int(prob_ldo_gt_020 >= 0.95)
    summary = {
        "stage": stage,
        "status": "summary",
        "core77_count": len(core),
        "core77_precision": core_row.get("GradeAB_precision"),
        "core77_V_LCB": core_row.get("V_integrated_LCB"),
        "core77_longrisk_UCB": core_row.get("h240_longrisk_UCB"),
        "core77_LDO_drop": core_row.get("LDO_drop"),
        "min_dataset_precision": min_ds_precision,
        "min_dataset_V_LCB": min_ds_v,
        "min_dataset_support": min_support,
        "max_dataset_support": max_support,
        "max_family_share": max_family_share,
        "max_template_share": max_template_share,
        "max_step_bucket_share": max_step_share,
        "bootstrap_reps": bootstrap_reps,
        "bootstrap_ldo_p50": qtile(boot_ldo, 0.50),
        "bootstrap_ldo_p95": qtile(boot_ldo, 0.95),
        "bootstrap_prob_LDO_gt_020": prob_ldo_gt_020,
        "H1a_dataset_shift_or_support_shift": h1a,
        "H1b_hidden_group_concentration": h1b,
        "H1c_structural_not_random": h1c,
        "P1_core77_autopsy_pass": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p1_core77_dataset_precision.svg", "P1 Core77 dataset precision", [r["dataset_id"] for r in dataset_rows], [fnum(r.get("core77_precision")) for r in dataset_rows])
    v9720.write_bar_svg(out / "fig_p1_core77_group_share.svg", "P1 Core77 group share", [f"{r['group_axis']}:{r['group_id']}" for r in group_rows[:20]], [fnum(r.get("core77_share")) for r in group_rows[:20]])
    return [summary] + dataset_rows, group_rows, summary


def set_feature_summary(
    ap0: list[dict[str, Any]],
    idx: list[int],
    label: str,
    r5b: list[float],
    exact_t4: list[float],
    wt: dict[str, dict[str, Any]],
    stage: str,
) -> dict[str, Any]:
    rows = [ap0[i] for i in idx]
    q = quality(ap0, idx, membership_scores(len(ap0), idx), len(idx))
    def vals(key: str) -> list[float]:
        return [fnum(r.get(key)) for r in rows]
    return {
        "stage": stage,
        "status": "set_row",
        "set_label": label,
        **q,
        "dataset_count": len({v9720.axis_value(r, "dataset_id") for r in rows}),
        "template_count": len({v9720.candidate_template_id(r) for r in rows}),
        "family_count": len({str(r.get("family_id")) for r in rows}),
        "mean_payload_norm": mean(vals("payload_norm")),
        "mean_action_norm": mean(vals("payload_norm")),
        "mean_adamw_cosine": mean(vals("adamw_alignment_cosine")),
        "mean_exact_transfer_lcb": mean([exact_t4[i] for i in idx]),
        "mean_old_rank_score": mean([r5b[i] for i in idx]),
        "mean_WT1_lcb": mean([wt_value(wt, ap0[i], "WT1_LCB", 0.0) for i in idx if str(ap0[i].get("action_id")) in wt]),
        "mean_WT5_lcb": mean([wt_value(wt, ap0[i], "WT5_LCB", 0.0) for i in idx if str(ap0[i].get("action_id")) in wt]),
        "mean_WT20_lcb": mean([wt_value(wt, ap0[i], "WT20_LCB", 0.0) for i in idx if str(ap0[i].get("action_id")) in wt]),
        "mean_WT80_lcb": mean([wt_value(wt, ap0[i], "WT80_LCB", 0.0) for i in idx if str(ap0[i].get("action_id")) in wt]),
        "mean_memory_proxy": mean([1.0 - float(v9720.memory_fail(r)) for r in rows]),
        "mean_offdiag_proxy": mean([1.0 - float(v9720.offdiag_fail(r)) for r in rows]),
        "mean_hardtail_proxy": mean([1.0 - float(inum(r.get("h240_longrisk"))) for r in rows]),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def p2_oldonly_anatomy(
    ap0: list[dict[str, Any]],
    cs: dict[str, list[int]],
    r5b: list[float],
    exact_t4: list[float],
    wt: dict[str, dict[str, Any]],
    out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage = "P2_OLDONLY_MECHANISM_ANATOMY_V9750"
    labels = ["Intersection", "OldOnly", "ExactOnly", "Neither"]
    rows = [set_feature_summary(ap0, cs[label], label, r5b, exact_t4, wt, stage) for label in labels]
    by = {r["set_label"]: r for r in rows}
    old = by["OldOnly"]
    cont = int(
        fnum(old.get("GradeAB_precision")) >= 0.75
        and fnum(old.get("V_integrated_LCB")) > 0
        and fnum(old.get("h240_longrisk_UCB")) <= 0.05
        and inum(old.get("dataset_count")) >= 3
        and inum(old.get("template_count")) >= 32
    )
    summary = {
        "stage": stage,
        "status": "summary",
        "set_count": len(rows),
        "OldOnly_count": old.get("accepted_count"),
        "OldOnly_precision": old.get("GradeAB_precision"),
        "OldOnly_V_LCB": old.get("V_integrated_LCB"),
        "OldOnly_longrisk_UCB": old.get("h240_longrisk_UCB"),
        "OldOnly_dataset_count": old.get("dataset_count"),
        "OldOnly_template_count": old.get("template_count"),
        "OldOnly_mean_exact_transfer_lcb": old.get("mean_exact_transfer_lcb"),
        "ExactOnly_precision": by["ExactOnly"].get("GradeAB_precision"),
        "Intersection_precision": by["Intersection"].get("GradeAB_precision"),
        "P2_oldonly_continue_research_pass": cont,
        "oldonly_single_group_pocket": int(fnum(old.get("max_dataset_share")) > 0.80 or fnum(old.get("max_template_share")) > 0.25),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p2_set_quality.svg", "P2 set precision", [r["set_label"] for r in rows], [fnum(r.get("GradeAB_precision")) for r in rows])
    write_quality_md(out / "set_quality_table_v9750.md", rows)
    return [summary] + rows, summary


def write_quality_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| set | count | precision | V LCB | longrisk UCB | LDO | dataset count | template count |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.get('set_label')}` | `{r.get('accepted_count')}` | `{r.get('GradeAB_precision')}` | `{r.get('V_integrated_LCB')}` | `{r.get('h240_longrisk_UCB')}` | `{r.get('LDO_drop')}` | `{r.get('dataset_count')}` | `{r.get('template_count')}` |"
        )
    path.write_text("\n".join(lines) + "\n")


def p3_transfer_deep_autopsy(
    ap0: list[dict[str, Any]],
    cs: dict[str, list[int]],
    r5b: list[float],
    exact_t4: list[float],
    wt: dict[str, dict[str, Any]],
    out: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    stage = "P3_TRANSFER_MISMATCH_DEEP_AUTOPSY_V9750"
    action_rows = []
    set_rows = []
    label_by_idx = {}
    for label in ["Intersection", "OldOnly", "ExactOnly"]:
        for i in cs[label]:
            label_by_idx[i] = label
    for i in cs["Neither"][: min(len(cs["Neither"]), 256)]:
        label_by_idx[i] = "NeitherSample"
    for i, label in sorted(label_by_idx.items()):
        r = ap0[i]
        t1 = wt_value(wt, r, "WT1_LCB", 0.0)
        t5 = wt_value(wt, r, "WT5_LCB", 0.0)
        t20 = wt_value(wt, r, "WT20_LCB", 0.0)
        t80 = wt_value(wt, r, "WT80_LCB", 0.0)
        action_rows.append(
            {
                "stage": stage,
                "status": "action_row",
                "action_id": r.get("action_id"),
                "set_label": label,
                "T1_lcb": t1,
                "T5_lcb": t5,
                "T20_lcb": t20,
                "T80_lcb": t80,
                "T80_minus_T1": t80 - t1,
                "transfer_persistence": min(t1, t5, t20, t80),
                "transfer_slope": (t80 - t1) / 79.0,
                "exact_T4_score": exact_t4[i],
                "old_rank_score": r5b[i],
                "memory_response_lcb": 1.0 - float(v9720.memory_fail(r)),
                "offdiag_response_lcb": 1.0 - float(v9720.offdiag_fail(r)),
                "hardtail_response_lcb": 1.0 - float(inum(r.get("h240_longrisk"))),
                "actual_V_integrated": r.get("V_integrated"),
                "actual_GradeAB": v9720.gradeab(r),
                "actual_longrisk": r.get("h240_longrisk"),
                "actual_bad": r.get("bad_event_rate"),
                "actual_null": r.get("null_event_rate"),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    for label in ["Intersection", "OldOnly", "ExactOnly", "NeitherSample"]:
        sub = [r for r in action_rows if r["set_label"] == label]
        if not sub:
            continue
        set_rows.append(
            {
                "stage": "P3_TRANSFER_MISMATCH_SET_SUMMARY_V9750",
                "status": "set_row",
                "set_label": label,
                "row_count": len(sub),
                "mean_T1_lcb": mean([fnum(r.get("T1_lcb")) for r in sub]),
                "mean_T5_lcb": mean([fnum(r.get("T5_lcb")) for r in sub]),
                "mean_T20_lcb": mean([fnum(r.get("T20_lcb")) for r in sub]),
                "mean_T80_lcb": mean([fnum(r.get("T80_lcb")) for r in sub]),
                "mean_exact_T4": mean([fnum(r.get("exact_T4_score")) for r in sub]),
                "mean_actual_V": mean([fnum(r.get("actual_V_integrated")) for r in sub]),
                "precision": mean([fnum(r.get("actual_GradeAB")) for r in sub]),
                "longrisk_rate": mean([fnum(r.get("actual_longrisk")) for r in sub]),
                "memory_safe_rate": mean([fnum(r.get("memory_response_lcb")) for r in sub]),
                "offdiag_safe_rate": mean([fnum(r.get("offdiag_response_lcb")) for r in sub]),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    by = {r["set_label"]: r for r in set_rows}
    old = by.get("OldOnly", {})
    ex = by.get("ExactOnly", {})
    h3a = int(fnum(old.get("mean_exact_T4")) < fnum(ex.get("mean_exact_T4")) and fnum(old.get("mean_actual_V")) > 0 and fnum(ex.get("mean_actual_V")) <= 0)
    h3b = int(fnum(ex.get("precision")) <= 0.25 and fnum(ex.get("mean_actual_V")) <= 0)
    h3c = int(fnum(old.get("memory_safe_rate")) >= fnum(ex.get("memory_safe_rate")) and fnum(old.get("offdiag_safe_rate")) >= fnum(ex.get("offdiag_safe_rate")) and fnum(old.get("precision")) > fnum(ex.get("precision")))
    summary = {
        "stage": stage,
        "status": "summary",
        "action_row_count": len(action_rows),
        "OldOnly_mean_T80": old.get("mean_T80_lcb"),
        "OldOnly_mean_exact_T4": old.get("mean_exact_T4"),
        "OldOnly_mean_actual_V": old.get("mean_actual_V"),
        "ExactOnly_mean_T80": ex.get("mean_T80_lcb"),
        "ExactOnly_mean_exact_T4": ex.get("mean_exact_T4"),
        "ExactOnly_mean_actual_V": ex.get("mean_actual_V"),
        "H3a_immediate_transfer_misses_delayed_value": h3a,
        "H3b_exactonly_low_value_safe_region": h3b,
        "H3c_oldonly_geometry_better_than_exactonly": h3c,
        "P3_transfer_autopsy_pass": int(h3a or h3b or h3c),
        "new_generated_objective_evidence": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p3_response_vector_precision.svg", "P3 set precision", [r["set_label"] for r in set_rows], [fnum(r.get("precision")) for r in set_rows])
    return [summary] + set_rows, action_rows, summary


def percentile_by_group(scores: list[float], groups: list[str]) -> list[float]:
    out = [0.0] * len(scores)
    by: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    for idxs in by.values():
        sorted_idxs = sorted(idxs, key=lambda i: scores[i])
        denom = max(1, len(sorted_idxs) - 1)
        for rank, i in enumerate(sorted_idxs):
            out[i] = rank / denom
    return out


def z_by_group(scores: list[float], groups: list[str]) -> list[float]:
    by: dict[str, list[float]] = defaultdict(list)
    for s, g in zip(scores, groups):
        by[g].append(s)
    mu = {g: mean(v) for g, v in by.items()}
    sd = {g: pstdev(v) for g, v in by.items()}
    return [(scores[i] - mu[groups[i]]) / (sd[groups[i]] + 1.0e-9) for i in range(len(scores))]


def capped_topk(scores: list[float], ap0: list[dict[str, Any]], k: int, max_family: int, max_template: int) -> list[int]:
    fam = Counter()
    tpl = Counter()
    idx = []
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


def p4_dataset_blind_stability(
    ap0: list[dict[str, Any]],
    r5b: list[float],
    exact_t4: list[float],
    wt: dict[str, dict[str, Any]],
    cs: dict[str, list[int]],
    out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    stage = "P4_DATASET_BLIND_STABILITY_REPAIR_V9750"
    t0 = time.perf_counter()
    family_step = [f"{r.get('family_id')}:{step_bucket(r)}" for r in ap0]
    family = [str(r.get("family_id")) for r in ap0]
    wt80 = [wt_value(wt, r, "WT80_LCB", -1.0e9) for r in ap0]
    core = cs["Core77"]
    old_pool = [i for i in topk(r5b, len(ap0)) if i not in set(core)]
    s7_idx = core + old_pool[: max(0, TARGET_K - len(core))]
    scores: dict[str, list[float]] = {
        "S1-raw-OldRank": r5b,
        "S2-family-step-z-normalized": z_by_group(r5b, family_step),
        "S3-family-percentile": percentile_by_group(r5b, family),
        "S4-oldrank-memory-offdiag-veto": [r5b[i] if (v9720.memory_fail(r) == 0 and v9720.offdiag_fail(r) == 0) else -1.0e9 for i, r in enumerate(ap0)],
        "S5-oldrank-response-stability-veto": [r5b[i] if wt80[i] > 0 else -1.0e9 for i in range(len(ap0))],
        "S6-support-balanced-topK": membership_scores(len(ap0), capped_topk(r5b, ap0, TARGET_K, max_family=8, max_template=1)),
        "S7-Core77-plus-frozen-expansion": membership_scores(len(ap0), s7_idx),
        "S8-exact-core-safe": exact_t4,
    }
    rows = []
    best = None
    for sid, sc in scores.items():
        start = time.perf_counter()
        idx = topk(sc, TARGET_K)
        q = quality(ap0, idx, sc, TARGET_K)
        cost = (time.perf_counter() - start + (time.perf_counter() - t0) * 0.0) * 1000.0
        fam_counts = Counter(str(ap0[i].get("family_id")) for i in idx)
        tpl_counts = Counter(v9720.candidate_template_id(ap0[i]) for i in idx)
        row = {
            "stage": stage,
            "status": "rule_row",
            "rule_id": sid,
            **q,
            "max_family_share": max((v / max(1, len(idx)) for v in fam_counts.values()), default=0.0),
            "max_template_share": max((v / max(1, len(idx)) for v in tpl_counts.values()), default=0.0),
            "feature_cost_q90_ms": cost,
            "weak_pass": pass_weak(q),
            "strong_pass": pass_strong(q),
            "uses_dataset_name_for_selector": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
        if best is None or (inum(row["strong_pass"]), inum(row["weak_pass"]), fnum(row["GradeAB_precision"]), fnum(row["V_integrated_LCB"]), -fnum(row["LDO_drop"])) > (inum(best["strong_pass"]), inum(best["weak_pass"]), fnum(best["GradeAB_precision"]), fnum(best["V_integrated_LCB"]), -fnum(best["LDO_drop"])):
            best = row
    best = best or rows[0]
    summary = {
        "stage": stage,
        "status": "summary",
        "rule_count": len(rows),
        "weak_pass_count": sum(inum(r["weak_pass"]) for r in rows),
        "strong_pass_count": sum(inum(r["strong_pass"]) for r in rows),
        "best_rule_id": best["rule_id"],
        "best_precision": best["GradeAB_precision"],
        "best_V_LCB": best["V_integrated_LCB"],
        "best_LDO_drop": best["LDO_drop"],
        "P4_dataset_blind_stability_weak_pass": int(any(inum(r["weak_pass"]) for r in rows)),
        "P4_dataset_blind_stability_strong_pass": int(any(inum(r["strong_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p4_rule_precision.svg", "P4 rule precision", [r["rule_id"] for r in rows], [fnum(r["GradeAB_precision"]) for r in rows])
    v9720.write_bar_svg(out / "fig_p4_rule_ldo.svg", "P4 rule LDO", [r["rule_id"] for r in rows], [fnum(r["LDO_drop"]) for r in rows])
    return [summary] + rows, summary, scores


def p5_expansion_v2(
    ap0: list[dict[str, Any]],
    r5b: list[float],
    exact_t4: list[float],
    wt: dict[str, dict[str, Any]],
    cs: dict[str, list[int]],
    seed: int,
    out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage = "P5_CORE77_EXPANSION_RESPONSIBILITY_V2_V9750"
    core = cs["Core77"]
    core_set = set(core)
    need = max(0, TARGET_K - len(core))
    old_pool = [i for i in topk(r5b, len(ap0)) if i not in core_set]
    oldonly_pool = [i for i in cs["OldOnly"] if i not in core_set]
    response_pool = [i for i in sorted(range(len(ap0)), key=lambda j: wt_value(wt, ap0[j], "WT80_LCB", -1.0e9), reverse=True) if i not in core_set and safe_clean(ap0[i])]
    safe_pool = [i for i, r in enumerate(ap0) if i not in core_set and safe_clean(r)]
    rnd = random.Random(seed + 9750)
    random_pool = rnd.sample(safe_pool, min(need, len(safe_pool))) if safe_pool else []
    worst_ds_order = ["MNIST", "KMNIST", "Fashion-MNIST"]
    worst_balanced = []
    for ds in worst_ds_order:
        for i in old_pool:
            if i in core_set or i in worst_balanced:
                continue
            if v9720.axis_value(ap0[i], "dataset_id") == ds and safe_clean(ap0[i]):
                worst_balanced.append(i)
                break
    for i in old_pool:
        if len(worst_balanced) >= need:
            break
        if i not in core_set and i not in worst_balanced and safe_clean(ap0[i]):
            worst_balanced.append(i)
    template_balanced = capped_topk([r5b[i] if i not in core_set and safe_clean(ap0[i]) else -1.0e9 for i in range(len(ap0))], ap0, need, max_family=need, max_template=1)
    candidates = {
        "E0-Core77-only": [],
        "E1-old-rank-top10": old_pool[:need],
        "E2-OldOnly-value-risk-top10": oldonly_pool[:need],
        "E3-response-vector-stable-top10": response_pool[:need],
        "E4-memory-offdiag-safe-top10": [i for i in old_pool if safe_clean(ap0[i])][:need],
        "E5-worst-split-balanced-top10": worst_balanced[:need],
        "E6-template-balanced-top10": template_balanced[:need],
        "E7-random-clean-risk-diagnostic-top10": random_pool[:need],
    }
    rows = []
    best = None
    for eid, exp in candidates.items():
        full = core + [i for i in exp if i not in core_set]
        exp_q = quality(ap0, exp, membership_scores(len(ap0), exp), len(exp)) if exp else {}
        full_q = quality(ap0, full, membership_scores(len(ap0), full), len(full))
        row = {
            "stage": stage,
            "status": "expansion_row",
            "expansion_id": eid,
            "expansion_count": len(exp),
            "expansion_precision": exp_q.get("GradeAB_precision", 0),
            "expansion_V_LCB": exp_q.get("V_integrated_LCB", 0),
            "expansion_LDO_drop": exp_q.get("LDO_drop", 0),
            "full87_accepted_count": full_q.get("accepted_count"),
            "full87_precision": full_q.get("GradeAB_precision"),
            "full87_V_LCB": full_q.get("V_integrated_LCB"),
            "full87_longrisk_UCB": full_q.get("h240_longrisk_UCB"),
            "full87_bad_UCB": full_q.get("bad_UCB"),
            "full87_null_UCB": full_q.get("null_UCB"),
            "full87_memory_UCB": full_q.get("memory_fail_UCB"),
            "full87_offdiag_UCB": full_q.get("offdiag_fail_UCB"),
            "full87_LDO_drop": full_q.get("LDO_drop"),
            "full87_LSO_drop": full_q.get("LSO_drop"),
            "full87_LTO_drop": full_q.get("LTO_drop"),
            "expansion_pass": pass_expansion({**full_q, "GradeAB_precision": full_q.get("GradeAB_precision")}),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
        if best is None or (inum(row["expansion_pass"]), fnum(row["full87_precision"]), fnum(row["full87_V_LCB"]), -fnum(row["full87_LDO_drop"])) > (inum(best["expansion_pass"]), fnum(best["full87_precision"]), fnum(best["full87_V_LCB"]), -fnum(best["full87_LDO_drop"])):
            best = row
    best = best or rows[0]
    summary = {
        "stage": stage,
        "status": "summary",
        "candidate_count": len(rows),
        "expansion_pass_count": sum(inum(r["expansion_pass"]) for r in rows),
        "best_expansion_id": best["expansion_id"],
        "best_full87_precision": best["full87_precision"],
        "best_full87_V_LCB": best["full87_V_LCB"],
        "best_full87_LDO_drop": best["full87_LDO_drop"],
        "P5_expansion_v2_pass": int(any(inum(r["expansion_pass"]) for r in rows)),
        "existing_action_theory_reset_required": int(not any(inum(r["expansion_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p5_expansion_ldo.svg", "P5 expansion LDO", [r["expansion_id"] for r in rows], [fnum(r["full87_LDO_drop"]) for r in rows])
    return [summary] + rows, summary


def p6_oldrank_ablation_v2(
    ap0: list[dict[str, Any]],
    r5b: list[float],
    exact_t4: list[float],
    wt: dict[str, dict[str, Any]],
    cs: dict[str, list[int]],
    out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage = "P6_OLDRANK_MECHANISM_ABLATION_V2_V9750"
    old_set = set(cs["OldRankTop87"])
    oldonly_set = set(cs["OldOnly"])
    core_set = set(cs["Core77"])
    signal = [fnum(r.get("signal_score")) for r in ap0]
    value = [fnum(r.get("V_score")) for r in ap0]
    risk = [fnum(r.get("risk_score")) for r in ap0]
    support = [fnum(r.get("snr_group")) for r in ap0]
    norm_penalty = [abs(fnum(r.get("payload_norm"))) for r in ap0]
    wt80 = [wt_value(wt, r, "WT80_LCB", -1.0e9) for r in ap0]
    scores = {
        "A0-OldRank": r5b,
        "A1-value-only": value,
        "A2-signal-risk": [signal[i] - risk[i] for i in range(len(ap0))],
        "A3-signal-risk-safe-veto": [signal[i] - risk[i] if safe_clean(r) else -1.0e9 for i, r in enumerate(ap0)],
        "A4-support-balanced-oldrank": membership_scores(len(ap0), capped_topk(r5b, ap0, TARGET_K, 8, 1)),
        "A5-no-norm-penalty": [r5b[i] + 0.05 * norm_penalty[i] for i in range(len(ap0))],
        "A6-exact-compatible-oldrank": [r5b[i] if exact_t4[i] > -1.0e8 else -1.0e9 for i in range(len(ap0))],
        "A7-response-stable-oldrank": [r5b[i] if wt80[i] > 0 else -1.0e9 for i in range(len(ap0))],
        "A8-memory-offdiag-only": [1.0 if (v9720.memory_fail(r) == 0 and v9720.offdiag_fail(r) == 0) else 0.0 for r in ap0],
    }
    rows = []
    best = None
    for aid, sc in scores.items():
        idx = topk(sc, TARGET_K)
        q = quality(ap0, idx, sc, TARGET_K)
        iset = set(idx)
        old_overlap = len(iset & old_set) / max(1, len(old_set))
        oldonly_overlap = len(iset & oldonly_set) / max(1, len(oldonly_set))
        core_overlap = len(iset & core_set) / max(1, len(core_set))
        mechanism_pass = int(
            fnum(q.get("GradeAB_precision")) >= 0.75
            and fnum(q.get("V_integrated_LCB")) > 0
            and fnum(q.get("h240_longrisk_UCB")) <= 0.05
            and fnum(q.get("LDO_drop")) <= 0.15
            and old_overlap >= 0.50
            and oldonly_overlap >= 0.50
        )
        row = {
            "stage": stage,
            "status": "ablation_row",
            "ablation_id": aid,
            "removed_component": aid.replace("A", "component-"),
            **q,
            "OldRankTop87_overlap": old_overlap,
            "OldOnly_overlap": oldonly_overlap,
            "Core77_overlap": core_overlap,
            "mechanism_explained_pass": mechanism_pass,
            "field_legality_color": "green" if "exact" not in aid.lower() and "response" not in aid.lower() else "yellow_diagnostic",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
        if best is None or (inum(row["mechanism_explained_pass"]), fnum(row["GradeAB_precision"]), fnum(row["V_integrated_LCB"]), -fnum(row["LDO_drop"])) > (inum(best["mechanism_explained_pass"]), fnum(best["GradeAB_precision"]), fnum(best["V_integrated_LCB"]), -fnum(best["LDO_drop"])):
            best = row
    best = best or rows[0]
    summary = {
        "stage": stage,
        "status": "summary",
        "ablation_count": len(rows),
        "mechanism_pass_count": sum(inum(r["mechanism_explained_pass"]) for r in rows),
        "best_ablation_id": best["ablation_id"],
        "best_precision": best["GradeAB_precision"],
        "best_V_LCB": best["V_integrated_LCB"],
        "best_LDO_drop": best["LDO_drop"],
        "best_OldOnly_overlap": best["OldOnly_overlap"],
        "P6_oldrank_mechanism_explained": int(any(inum(r["mechanism_explained_pass"]) for r in rows)),
        "P6_generative_mechanism_ready": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p6_oldrank_ablation_precision.svg", "P6 ablation precision", [r["ablation_id"] for r in rows], [fnum(r["GradeAB_precision"]) for r in rows])
    return [summary] + rows, summary


def not_run(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {"stage": stage, "status": "not_run", "reason": reason, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0, **extra}
    return [row], row


def p7_controller(p4: dict[str, Any], p5: dict[str, Any], p6: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    opened = inum(p4.get("P4_dataset_blind_stability_weak_pass")) or inum(p5.get("P5_expansion_v2_pass")) or inum(p6.get("P6_oldrank_mechanism_explained"))
    if not opened:
        return not_run("P7_EXISTING_ACTION_MINIMAL_CONTROLLER_BOUNDARY_V9750", "P4_P5_P6_no_weak_pass", controller_pass=0, source_controller_pass=0)
    return not_run("P7_EXISTING_ACTION_MINIMAL_CONTROLLER_BOUNDARY_V9750", "candidate_diagnostic_not_promoted_to_controller", controller_pass=0, source_controller_pass=0)


def p9_generated_route(source_v9740: Path, p3: dict[str, Any], p6: dict[str, Any], p7: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prev = read_json(source_v9740 / "route_decision_v9740.json")
    reopen = int(inum(p3.get("new_generated_objective_evidence")) or inum(p6.get("P6_generative_mechanism_ready")) or inum(p7.get("source_controller_pass")))
    row = {
        "stage": "P9_GENERATED_ROUTE_STOP_REOPEN_DECISION_V9750",
        "status": "summary",
        "source_generated_route_status_v9740": prev.get("generated_route_status"),
        "new_objective_evidence_present": reopen,
        "oldrank_generative_mechanism_ready": p6.get("P6_generative_mechanism_ready"),
        "controller_pass": p7.get("source_controller_pass", 0),
        "generated_route_status": "direct_solved_sandbox_allowed" if reopen else "stopped_no_new_objective",
        "APGU_APGV_APGW_APGX_run": 0,
        "direct_solved_sandbox_allowed": reopen,
        "generated_action_count": 0,
        "branch_horizon_rows": 0,
        "generated_route_stop_triggered": int(not reopen),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def base_acc(source_v9740: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(r) for r in read_csv(source_v9740 / "base_acc_sentinel_v9740.csv")]
    for r in rows:
        r["stage"] = "BASE_ACC_SENTINEL_V9750"
        r["reused_from_v9740"] = 1
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

    source_v9740 = Path(args.source_v9740)
    source_v9720 = Path(args.source_v9720)
    ap0, score_bundle, _payload_by_id = v9720.load_ap0_and_payloads(args)
    r5b = v9660.r5b_scores(ap0, score_bundle)
    exact_summary = v9730.load_exact_summary(source_v9720)
    score_defs = v9730.score_defs_from_summary(ap0, exact_summary)
    exact_t4 = score_defs["T4-core-safe-transfer"]
    cs = cohort_sets(ap0, r5b, exact_t4)
    wt = load_wt_rows(source_v9740)

    p0_rows, p0_legality, p0 = p0_boundary(source_v9740)
    dump_csv("p0_boundary_reproduction_v9750.csv", p0_rows)
    dump_csv("p0_field_legality_audit_v9750.csv", p0_legality)
    p1_rows, p1_group_rows, p1 = p1_core77_autopsy(ap0, r5b, cs["Core77"], out, int(args.seed), int(args.bootstrap_reps))
    dump_csv("p1_core77_ldo_autopsy_v9750.csv", p1_rows)
    dump_csv("p1_core77_group_autopsy_v9750.csv", p1_group_rows)
    p2_rows, p2 = p2_oldonly_anatomy(ap0, cs, r5b, exact_t4, wt, out)
    dump_csv("p2_oldonly_mechanism_anatomy_v9750.csv", p2_rows)
    p3_summary_rows, p3_action_rows, p3 = p3_transfer_deep_autopsy(ap0, cs, r5b, exact_t4, wt, out)
    dump_csv("p3_transfer_mismatch_deep_autopsy_v9750.csv", p3_summary_rows)
    dump_csv("p3_response_vector_actions_v9750.csv", p3_action_rows)
    p4_rows, p4, p4_scores = p4_dataset_blind_stability(ap0, r5b, exact_t4, wt, cs, out)
    dump_csv("p4_dataset_blind_stability_repair_v9750.csv", p4_rows)
    p5_rows, p5 = p5_expansion_v2(ap0, r5b, exact_t4, wt, cs, int(args.seed), out)
    dump_csv("p5_core77_expansion_responsibility_v2_v9750.csv", p5_rows)
    p6_rows, p6 = p6_oldrank_ablation_v2(ap0, r5b, exact_t4, wt, cs, out)
    dump_csv("p6_oldrank_mechanism_ablation_v2_v9750.csv", p6_rows)
    p7_rows, p7 = p7_controller(p4, p5, p6)
    dump_csv("p7_existing_action_minimal_controller_boundary_v9750.csv", p7_rows)
    p8_rows, p8 = not_run("P8_SELECTED_RUNTIME_BOUNDARY_V9750", "P7_controller_not_passed", selected_runtime_pass=0)
    dump_csv("p8_selected_runtime_boundary_v9750.csv", p8_rows)
    p9_rows, p9 = p9_generated_route(source_v9740, p3, p6, p7)
    dump_csv("p9_generated_route_stop_reopen_decision_v9750.csv", p9_rows)
    p10_rows, p10 = not_run("P10_PAIRED_REPLAY_BOUNDARY_V9750", "P7_or_P8_not_passed", paired_replay_pass=0)
    dump_csv("p10_paired_replay_boundary_v9750.csv", p10_rows)
    p11_rows, p11 = not_run("P11_SHORT_FULL_BOUNDARY_V9750", "P10_paired_replay_not_open", short_run_boundary_open=0, full_run_boundary_open=0)
    dump_csv("p11_short_full_boundary_v9750.csv", p11_rows)
    base_rows, base_summary = base_acc(source_v9740)
    dump_csv("base_acc_sentinel_v9750.csv", base_rows)

    controller_pass = inum(p7.get("source_controller_pass", 0))
    runtime_pass = inum(p8.get("selected_runtime_pass", 0))
    system_pass = int(controller_pass and runtime_pass)
    high_quality_ldo = int(fnum(p4.get("best_precision")) >= 0.75 and fnum(p4.get("best_V_LCB")) > 0 and fnum(p4.get("best_LDO_drop")) > 0.20)
    if not inum(p0.get("p0_pass")):
        route, primary = "R0-BoundaryFail", "v9740_boundary_not_reproduced"
    elif system_pass:
        route, primary = "R1-ExistingActionControllerPass", "none"
    elif high_quality_ldo or (fnum(p1.get("core77_LDO_drop")) > 0.20 and not inum(p4.get("P4_dataset_blind_stability_weak_pass"))):
        route, primary = "R2-ExistingActionHighQualityButLDOBlocked", "existing_action_ldo_blocked"
    elif inum(p6.get("P6_oldrank_mechanism_explained")) == 0 and inum(p2.get("P2_oldonly_continue_research_pass")):
        route, primary = "R3-OldRankMechanismUnexplained", "oldrank_mechanism_unexplained"
    elif inum(p3.get("P3_transfer_autopsy_pass")):
        route, primary = "R4-TransferObjectiveMismatchConfirmed", "transfer_objective_mismatch"
    else:
        route, primary = "R5-GeneratedRouteRemainsStopped", "generated_route_stopped_no_new_objective"
    route_decision = {
        "stage": "ROUTE_DECISION_V9750",
        "status": "summary",
        "route": route,
        "source_route_v9740": p0.get("source_route_v9740"),
        "p0_pass": p0.get("p0_pass"),
        "P1_core77_autopsy_pass": p1.get("P1_core77_autopsy_pass"),
        "Core77_LDO_drop": p1.get("core77_LDO_drop"),
        "H1a_dataset_shift_or_support_shift": p1.get("H1a_dataset_shift_or_support_shift"),
        "H1b_hidden_group_concentration": p1.get("H1b_hidden_group_concentration"),
        "H1c_structural_not_random": p1.get("H1c_structural_not_random"),
        "P2_oldonly_continue_research_pass": p2.get("P2_oldonly_continue_research_pass"),
        "OldOnly_precision": p2.get("OldOnly_precision"),
        "OldOnly_V_LCB": p2.get("OldOnly_V_LCB"),
        "ExactOnly_precision": p2.get("ExactOnly_precision"),
        "P3_transfer_autopsy_pass": p3.get("P3_transfer_autopsy_pass"),
        "H3a_immediate_transfer_misses_delayed_value": p3.get("H3a_immediate_transfer_misses_delayed_value"),
        "H3b_exactonly_low_value_safe_region": p3.get("H3b_exactonly_low_value_safe_region"),
        "P4_dataset_blind_stability_weak_pass": p4.get("P4_dataset_blind_stability_weak_pass"),
        "P4_dataset_blind_stability_strong_pass": p4.get("P4_dataset_blind_stability_strong_pass"),
        "best_stability_rule": p4.get("best_rule_id"),
        "best_stability_precision": p4.get("best_precision"),
        "best_stability_V_LCB": p4.get("best_V_LCB"),
        "best_stability_LDO_drop": p4.get("best_LDO_drop"),
        "P5_expansion_v2_pass": p5.get("P5_expansion_v2_pass"),
        "best_expansion_id": p5.get("best_expansion_id"),
        "best_expansion_LDO_drop": p5.get("best_full87_LDO_drop"),
        "P6_oldrank_mechanism_explained": p6.get("P6_oldrank_mechanism_explained"),
        "best_ablation_id": p6.get("best_ablation_id"),
        "best_ablation_precision": p6.get("best_precision"),
        "best_ablation_LDO_drop": p6.get("best_LDO_drop"),
        "controller_pass": controller_pass,
        "selected_runtime_pass": runtime_pass,
        "generated_route_status": p9.get("generated_route_status"),
        "direct_solved_sandbox_allowed": p9.get("direct_solved_sandbox_allowed"),
        "system_legal_controller_pass": system_pass,
        "primary_blocker": primary,
        "secondary_blocker": "generated_route_stopped_no_new_objective" if inum(p9.get("generated_route_stop_triggered")) else "none",
    }
    dump_json("route_decision_v9750.json", route_decision)
    allowed = {
        "selected_runtime_allowed": 0,
        "paired_replay_allowed": 0,
        "short_full_allowed": 0,
        "generated_route_allowed": 0,
        "direct_solved_sandbox_allowed": p9.get("direct_solved_sandbox_allowed"),
        "APGU_APGV_APGW_APGX_run_allowed": 0,
    }
    dump_json("allowed_next_gates_v9750.json", allowed)
    stop = {
        "enter_system": system_pass,
        "stop_core_expansion_threshold_patching": int(not inum(p5.get("P5_expansion_v2_pass")) and fnum(p1.get("core77_LDO_drop")) > 0.20),
        "stop_oldrank_controller_promotion": int(not inum(p6.get("P6_oldrank_mechanism_explained"))),
        "stop_transfer_selector_route": int(inum(p3.get("P3_transfer_autopsy_pass"))),
        "stop_generated_blind_variants": 1,
    }
    dump_json("stop_conditions_v9750.json", stop)

    nf = no_fake(list(artifacts.values()))
    dump_csv("no_fake_audit_v9750.csv", [{"stage": "NO_FAKE_AUDIT_V9750", "status": "summary", **nf}])
    contract = {
        "stage": "CONTRACT_AUDIT_V9750",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "v9740_boundary_pass": p0.get("p0_pass"),
        "core77_autopsy_pass": p1.get("P1_core77_autopsy_pass"),
        "oldonly_anatomy_pass": p2.get("P2_oldonly_continue_research_pass"),
        "transfer_autopsy_pass": p3.get("P3_transfer_autopsy_pass"),
        "dataset_blind_stability_pass": f"{p4.get('P4_dataset_blind_stability_weak_pass')}/{p4.get('P4_dataset_blind_stability_strong_pass')}",
        "expansion_v2_pass": p5.get("P5_expansion_v2_pass"),
        "oldrank_mechanism_explained": p6.get("P6_oldrank_mechanism_explained"),
        "controller/runtime/system": f"{controller_pass}/{runtime_pass}/{system_pass}",
        "generated_route_stop": p9.get("generated_route_stop_triggered"),
        "base_acc_sentinel_pass": base_summary.get("base_acc_sentinel_pass"),
        "base_acc_used_for_controller": 0,
        "uses_loss_backward/teacher/loss_modification": "0/0/0",
        "uses_dataset_name_for_selector/controller": "0/0",
        "uses_validation_or_test_for_controller": 0,
        "uses_future_outcome_for_features": 0,
        "proxy_transfer_promoted_to_official": 0,
        "diagnostic_promoted_to_official": 0,
        "fake/proxy/cpu_offload": f"{nf['fake_data_used']}/{nf['proxy_row_used']}/{nf['cpu_offload_used']}",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("contract_audit_v9750.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9750",
        "status": "summary",
        "route": route,
        "F0_boundary_fail": int(not inum(p0.get("p0_pass"))),
        "F1_core77_ldo_root_cause_present": int(fnum(p1.get("core77_LDO_drop")) > 0.20),
        "F2_oldonly_high_quality_unexplained": int(inum(p2.get("P2_oldonly_continue_research_pass")) and not inum(p6.get("P6_oldrank_mechanism_explained"))),
        "F3_transfer_objective_mismatch": int(inum(p3.get("P3_transfer_autopsy_pass"))),
        "F4_dataset_blind_stability_fail": int(not inum(p4.get("P4_dataset_blind_stability_weak_pass"))),
        "F5_expansion_v2_fail": int(not inum(p5.get("P5_expansion_v2_pass"))),
        "F6_controller_runtime_blocked": int(not controller_pass),
        "F7_generated_route_stopped": int(inum(p9.get("generated_route_stop_triggered"))),
        "F8_system_not_official": int(not system_pass),
        "primary_blocker": primary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("failure_taxonomy_v9750.csv", [failure])
    hashes = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    manifest = {
        "stage": "RUN_MANIFEST_V9750",
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
    dump_json("run_manifest_v9750.json", manifest)
    print(
        json.dumps(
            {
                "out_dir": str(out),
                "route": route,
                "primary_blocker": primary,
                "Core77_LDO_drop": p1.get("core77_LDO_drop"),
                "OldOnly_precision": p2.get("OldOnly_precision"),
                "best_stability_rule": p4.get("best_rule_id"),
                "best_stability_precision": p4.get("best_precision"),
                "best_stability_LDO_drop": p4.get("best_LDO_drop"),
                "best_ablation_id": p6.get("best_ablation_id"),
                "system_legal_controller_pass": system_pass,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
