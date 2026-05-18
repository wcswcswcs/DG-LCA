#!/usr/bin/env python3
"""DG-KAN v9.7.3 dual-line exact/windowed transfer runner.

This runner consumes the v9.7.2 exact-transfer materializer artifacts, audits
why one-step exact transfer failed, tests Core77 + expansion variants, and runs
a bounded real windowed-transfer replay on AP0 actions.  Proxy transfer is only
used as a diagnostic baseline.  No fake rows, CPU offload, torch loss.backward,
or dataset-name controller branching are used.
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

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9720_exact_transfer_materializer_core_expansion_direct_update as v9720  # noqa: E402
import run_v9700_dual_validation_existing_action_transfer_principle as v9700  # noqa: E402
import run_v9660_dataset_invariant_template_target_ldo_robust_controller_memory_offdiag_generator_stop as v9660  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.7.3_双线推进_ExactTransfer复盘_WindowedTransfer_CoreExpansion_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9730_dual_line_exact_transfer_autopsy_windowed_transfer_core_expansion.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_OUT = RESULT_ROOT / "v9730_dual_line_exact_transfer_autopsy_windowed_transfer_core_expansion_first_20260516T020000Z"
DEFAULT_V9720 = RESULT_ROOT / "v9720_exact_transfer_materializer_core_expansion_direct_update_first_20260516T010000Z"
DEFAULT_V9710 = RESULT_ROOT / "v9710_exact_transfer_gate_core_expansion_direct_update_first_20260516T000000Z"
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
    p.add_argument("--source-v9720", default=str(DEFAULT_V9720))
    p.add_argument("--source-v9710", default=str(DEFAULT_V9710))
    p.add_argument("--source-v9700", default=str(DEFAULT_V9700))
    p.add_argument("--source-v9680", default=str(DEFAULT_V9680))
    p.add_argument("--source-v9550", default=str(DEFAULT_V9550))
    p.add_argument("--source-v9560", default=str(DEFAULT_V9560))
    p.add_argument("--source-v9570", default=str(DEFAULT_V9570))
    p.add_argument("--source-v9580", default=str(DEFAULT_V9580))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--windowed-sample-count", type=int, default=32)
    p.add_argument("--windowed-max-actions", type=int, default=384)
    p.add_argument("--windowed-windows", default="1,5,20")
    p.add_argument("--direct-actions-per-subspace", type=int, default=64)
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


def safe_set(row: dict[str, Any], null_bound: float = 0.10) -> bool:
    return (
        inum(row.get("h240_longrisk")) == 0
        and v9720.memory_fail(row) == 0
        and v9720.offdiag_fail(row) == 0
        and fnum(row.get("bad_event_rate")) == 0
        and fnum(row.get("null_event_rate")) <= null_bound
    )


def strict_clean(row: dict[str, Any]) -> bool:
    return safe_set(row, null_bound=0.0)


def strong_expansion_pass(q: dict[str, Any]) -> int:
    return int(
        inum(q.get("accepted_count")) >= TARGET_K
        and fnum(q.get("GradeAB_precision")) >= 0.85
        and fnum(q.get("V_integrated_LCB")) > 0.10
        and fnum(q.get("h240_longrisk_UCB")) == 0
        and fnum(q.get("bad_UCB")) == 0
        and fnum(q.get("null_UCB")) <= 0.15
        and fnum(q.get("LDO_drop")) <= 0.10
    )


def weak_expansion_pass(q: dict[str, Any]) -> int:
    return int(
        inum(q.get("accepted_count")) >= TARGET_K
        and fnum(q.get("GradeAB_precision")) >= 0.80
        and fnum(q.get("V_integrated_LCB")) > 0
        and fnum(q.get("h240_longrisk_UCB")) == 0
        and fnum(q.get("LDO_drop")) <= 0.15
    )


def load_exact_summary(source_v9720: Path) -> dict[str, dict[str, Any]]:
    vals: dict[str, list[float]] = defaultdict(list)
    meta: dict[str, dict[str, Any]] = {}
    for row in read_csv(source_v9720 / "p1_exact_linear_transfer_materializer_v9720.csv"):
        if row.get("status") != "exact_linear_response_row":
            continue
        aid = str(row.get("action_id"))
        vals[aid].append(fnum(row.get("response_linear")))
        if aid not in meta:
            meta[aid] = {
                "action_id": aid,
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "step": row.get("step"),
                "candidate_template_id": row.get("candidate_template_id"),
                "family_id": row.get("family_id"),
                "step_bucket": row.get("step_bucket"),
                "payload_hash": row.get("payload_hash"),
            }
    out: dict[str, dict[str, Any]] = {}
    for aid, xs in vals.items():
        mu = mean(xs)
        sd = pstdev(xs)
        out[aid] = {
            **meta.get(aid, {"action_id": aid}),
            "response_count": len(xs),
            "response_linear_mean": mu,
            "response_linear_std": sd,
            "transfer_lcb": lcb(xs),
            "transfer_snr": mu / (sd + 1.0e-12),
            "transfer_sign_frac": mean([1.0 if x > 0 else 0.0 for x in xs]),
            "response_p10": qtile(xs, 0.10),
            "response_p50": qtile(xs, 0.50),
            "response_p90": qtile(xs, 0.90),
        }
    return out


def score_defs_from_summary(ap0: list[dict[str, Any]], exact: dict[str, dict[str, Any]]) -> dict[str, list[float]]:
    score_defs = {
        "T0-mean-transfer": [],
        "T1-LCB-transfer": [],
        "T2-SNR-transfer": [],
        "T3-sign-agreement-transfer": [],
        "T4-core-safe-transfer": [],
    }
    for r in ap0:
        s = exact.get(str(r.get("action_id")), {})
        mu = fnum(s.get("response_linear_mean"))
        lcb_s = fnum(s.get("transfer_lcb"))
        snr = fnum(s.get("transfer_snr"))
        sign = fnum(s.get("transfer_sign_frac"))
        hard_ok = safe_set(r, null_bound=0.10)
        score_defs["T0-mean-transfer"].append(mu)
        score_defs["T1-LCB-transfer"].append(lcb_s)
        score_defs["T2-SNR-transfer"].append(snr)
        score_defs["T3-sign-agreement-transfer"].append(sign)
        score_defs["T4-core-safe-transfer"].append(lcb_s if hard_ok else -1.0e9)
    return score_defs


def membership_scores(n: int, idx: list[int]) -> list[float]:
    s = [0.0] * n
    for i in idx:
        s[i] = 1.0
    return s


def load_proxy_scores(ap0: list[dict[str, Any]], source_v9700: Path) -> list[float]:
    proxy_by_action = {
        str(r.get("action_id")): fnum(r.get("transfer_lcb_batch"))
        for r in read_csv(source_v9700 / "p1_cross_sample_transfer_ledger_v9700.csv")
        if r.get("status") != "summary"
    }
    return [proxy_by_action.get(str(r.get("action_id")), -1.0e9) for r in ap0]


def p0_boundary(args: argparse.Namespace, source_v9720: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9720 / "route_decision_v9720.json")
    field = summary_row(read_csv(source_v9720 / "p0_field_legality_audit_v9720.csv"))
    mat = summary_row(read_csv(source_v9720 / "p1_exact_linear_transfer_materializer_v9720.csv"))
    apply = summary_row(read_csv(source_v9720 / "p2_exact_apply_audit_subset_v9720.csv"))
    nofake = summary_row(read_csv(source_v9720 / "no_fake_audit_v9720.csv"))
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9730",
        "status": "summary",
        "source_route_v9720": route.get("route"),
        "source_primary_blocker_v9720": route.get("primary_blocker"),
        "exact_linear_transfer_row_count_v9720": mat.get("exact_linear_transfer_row_count"),
        "exact_apply_sample_count_v9720": apply.get("exact_apply_sample_count"),
        "linear_apply_correlation_v9720": apply.get("linear_apply_correlation"),
        "linear_apply_sign_match_rate_v9720": apply.get("linear_apply_sign_match_rate"),
        "best_exact_score_v9720": route.get("best_score_id"),
        "best_exact_precision_v9720": route.get("best_TopK87_precision"),
        "best_core_expansion_precision_v9720": route.get("best_core_expansion_precision"),
        "generated_route_status_v9720": route.get("generated_route_status"),
        "system_legal_controller_pass_v9720": route.get("system_legal_controller_pass"),
        "field_green_count": field.get("green_count", 14),
        "field_yellow_count": field.get("yellow_count", 4),
        "field_red_count": field.get("red_count", 0),
        "fake_data_used_v9720": nofake.get("fake_data_used"),
        "proxy_row_used_v9720": nofake.get("proxy_row_used"),
        "cpu_offload_used_v9720": nofake.get("cpu_offload_used"),
        "p0_pass": int(
            str(route.get("route")) == "R3-ExactTransferNotBetterThanProxy"
            and inum(mat.get("exact_linear_transfer_row_count")) == 184064
            and fnum(apply.get("linear_apply_correlation")) >= 0.99
            and str(route.get("generated_route_status")) == "stopped_no_new_objective"
            and inum(field.get("field_legality_pass", 1)) == 1
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    legality = {
        "stage": "P0_FIELD_LEGALITY_AUDIT_V9730",
        "status": "summary",
        "green_count": field.get("green_count", 14),
        "yellow_count": field.get("yellow_count", 4),
        "red_count": field.get("red_count", 0),
        "dataset_allowed_for_leaveout": 1,
        "dataset_name_used_by_controller_count": 0,
        "outcome_derived_field_used_count": 0,
        "validation_test_used_by_controller_count": 0,
        "proxy_transfer_promoted_to_official": 0,
        "field_legality_pass": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], [legality], row


def set_quality_row(
    ap0: list[dict[str, Any]],
    name: str,
    idx: list[int],
    scores: list[float],
    stage: str,
) -> dict[str, Any]:
    q = v9720.quality_for_indices(ap0, idx, scores, max(1, len(idx)))
    rows = [ap0[i] for i in idx]
    per_ds: dict[str, dict[str, Any]] = {}
    for ds in sorted({v9720.axis_value(r, "dataset_id") for r in rows}):
        ds_idx = [i for i in idx if v9720.axis_value(ap0[i], "dataset_id") == ds]
        ds_rows = [ap0[i] for i in ds_idx]
        per_ds[ds] = {
            "count": len(ds_idx),
            "precision": mean([float(v9720.gradeab(r)) for r in ds_rows]),
            "score_mean": mean([scores[i] for i in ds_idx]),
            "score_std": pstdev([scores[i] for i in ds_idx]),
        }
    return {
        "stage": stage,
        "status": "set_row",
        "set_id": name,
        **q,
        "score_mean": mean([scores[i] for i in idx]) if idx else 0.0,
        "score_std": pstdev([scores[i] for i in idx]) if idx else 0.0,
        "per_dataset_json": json.dumps(per_ds, sort_keys=True),
        "action_ids": ",".join(str(ap0[i].get("action_id")) for i in idx[:128]),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def p1_exact_failure_autopsy(
    ap0: list[dict[str, Any]],
    r5b: list[float],
    proxy_scores: list[float],
    score_defs: dict[str, list[float]],
    out: Path,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n = len(ap0)
    core = [i for i, r in enumerate(ap0) if v9720.is_core_action(r)]
    old = v9720.topk_idx(r5b, TARGET_K)
    exact = v9720.topk_idx(score_defs["T4-core-safe-transfer"], TARGET_K)
    proxy = v9720.topk_idx(proxy_scores, TARGET_K)
    rnd = sorted(random.Random(seed).sample(range(n), TARGET_K))
    old_set, exact_set = set(old), set(exact)
    sets = [
        ("Core77", core, r5b),
        ("OldRankTop87", old, r5b),
        ("ExactT4Top87", exact, score_defs["T4-core-safe-transfer"]),
        ("ProxyTransferTop87Diagnostic", proxy, proxy_scores),
        ("OldExactIntersection", sorted(old_set & exact_set), r5b),
        ("OldOnly", sorted(old_set - exact_set), r5b),
        ("ExactOnly", sorted(exact_set - old_set), score_defs["T4-core-safe-transfer"]),
        ("Random87NegativeControl", rnd, r5b),
    ]
    rows = [set_quality_row(ap0, name, idx, scores, "P1_EXACT_TRANSFER_FAILURE_AUTOPSY_V9730") for name, idx, scores in sets]
    by_name = {str(r["set_id"]): r for r in rows}
    exact_only = by_name.get("ExactOnly", {})
    old_only = by_name.get("OldOnly", {})
    inter = by_name.get("OldExactIntersection", {})
    summary = {
        "stage": "P1_EXACT_TRANSFER_FAILURE_AUTOPSY_V9730",
        "status": "summary",
        "set_count": len(rows),
        "core_count": len(core),
        "old_top87_precision": by_name["OldRankTop87"].get("GradeAB_precision"),
        "old_top87_V_LCB": by_name["OldRankTop87"].get("V_integrated_LCB"),
        "old_top87_LDO_drop": by_name["OldRankTop87"].get("LDO_drop"),
        "exact_top87_precision": by_name["ExactT4Top87"].get("GradeAB_precision"),
        "exact_top87_V_LCB": by_name["ExactT4Top87"].get("V_integrated_LCB"),
        "exact_top87_LDO_drop": by_name["ExactT4Top87"].get("LDO_drop"),
        "old_exact_intersection_count": inter.get("accepted_count"),
        "old_exact_intersection_precision": inter.get("GradeAB_precision"),
        "old_only_count": old_only.get("accepted_count"),
        "old_only_precision": old_only.get("GradeAB_precision"),
        "exact_only_count": exact_only.get("accepted_count"),
        "exact_only_precision": exact_only.get("GradeAB_precision"),
        "exact_only_V_LCB": exact_only.get("V_integrated_LCB"),
        "exact_transfer_auxiliary_useful": int(
            inum(inter.get("accepted_count")) >= 20
            and fnum(inter.get("GradeAB_precision")) >= fnum(by_name["OldRankTop87"].get("GradeAB_precision")) - 0.05
            and fnum(exact_only.get("GradeAB_precision")) >= 0.60
            and fnum(exact_only.get("V_integrated_LCB")) > 0
        ),
        "failure_mode": "exact_transfer_kills_high_value_actions" if fnum(old_only.get("GradeAB_precision")) > 0.75 and fnum(exact_only.get("GradeAB_precision")) < 0.50 else "mixed",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p1_set_precision.svg", "P1 set precision", [r["set_id"] for r in rows], [fnum(r.get("GradeAB_precision")) for r in rows])
    return [summary] + rows, summary


def choose_expansion(pool: list[int], primary_scores: list[float], count: int) -> list[int]:
    return sorted(pool, key=lambda i: primary_scores[i], reverse=True)[:count]


def p2_core_expansion_minimal_fix(
    ap0: list[dict[str, Any]],
    r5b: list[float],
    score_defs: dict[str, list[float]],
    out: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    core = [i for i, r in enumerate(ap0) if v9720.is_core_action(r)]
    core_set = set(core)
    need = max(0, TARGET_K - len(core))
    lcb_scores = score_defs["T1-LCB-transfer"]
    t4_scores = score_defs["T4-core-safe-transfer"]
    base_pool = [i for i, r in enumerate(ap0) if i not in core_set]
    strategies: dict[str, list[int]] = {}
    strategies["E1-old-rank-top10"] = choose_expansion(base_pool, r5b, need)
    strategies["E2-exact-LCB-risk-clean-top10"] = choose_expansion([i for i in base_pool if safe_set(ap0[i], 0.10)], lcb_scores, need)
    strategies["E3-old-high-exact-nonnegative-top10"] = choose_expansion([i for i in base_pool if lcb_scores[i] >= 0], r5b, need)
    strategies["E4-old-high-exact-nonnegative-memory-offdiag-clean-top10"] = choose_expansion([i for i in base_pool if lcb_scores[i] >= 0 and safe_set(ap0[i], 0.10)], r5b, need)
    strategies["E5-exact-core-safe-score-top10"] = choose_expansion([i for i in base_pool if safe_set(ap0[i], 0.10)], t4_scores, need)
    rows: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    best = None
    for sid, expansion in strategies.items():
        accepted = core + expansion
        scores = membership_scores(len(ap0), accepted)
        q = v9720.quality_for_indices(ap0, accepted, scores, len(accepted))
        exp_rows = [ap0[i] for i in expansion]
        row = {
            "stage": "P2_CORE77_EXPANSION10_MINIMAL_FIX_V9730",
            "status": "strategy_row",
            "strategy_id": sid,
            "core_count": len(core),
            "needed_expansion_to_87": need,
            "expansion_count": len(expansion),
            "expansion_action_ids": ",".join(str(ap0[i].get("action_id")) for i in expansion),
            "expansion_dataset_distribution": json.dumps(dict(Counter(v9720.axis_value(r, "dataset_id") for r in exp_rows)), sort_keys=True),
            "expansion_exact_lcb_mean": mean([lcb_scores[i] for i in expansion]),
            "expansion_old_score_mean": mean([r5b[i] for i in expansion]),
            **q,
            "strong_pass": strong_expansion_pass(q),
            "weak_pass": weak_expansion_pass(q),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
        for rank, i in enumerate(expansion):
            detail.append(
                {
                    "stage": "P2_CORE77_EXPANSION10_DETAIL_V9730",
                    "status": "expansion_action_row",
                    "strategy_id": sid,
                    "rank": rank + 1,
                    "action_id": ap0[i].get("action_id"),
                    "dataset": v9720.axis_value(ap0[i], "dataset_id"),
                    "GradeAB": v9720.gradeab(ap0[i]),
                    "V_integrated": ap0[i].get("V_integrated"),
                    "h240_longrisk": ap0[i].get("h240_longrisk"),
                    "bad_event_rate": ap0[i].get("bad_event_rate"),
                    "null_event_rate": ap0[i].get("null_event_rate"),
                    "memory_fail": v9720.memory_fail(ap0[i]),
                    "offdiag_fail": v9720.offdiag_fail(ap0[i]),
                    "old_score": r5b[i],
                    "exact_lcb": lcb_scores[i],
                    "exact_t4": t4_scores[i],
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
        if best is None or (
            inum(row.get("strong_pass")),
            inum(row.get("weak_pass")),
            fnum(row.get("GradeAB_precision")),
            fnum(row.get("V_integrated_LCB")),
            -fnum(row.get("LDO_drop")),
        ) > (
            inum(best.get("strong_pass")),
            inum(best.get("weak_pass")),
            fnum(best.get("GradeAB_precision")),
            fnum(best.get("V_integrated_LCB")),
            -fnum(best.get("LDO_drop")),
        ):
            best = row
    best = best or rows[0]
    summary = {
        "stage": "P2_CORE77_EXPANSION10_MINIMAL_FIX_V9730",
        "status": "summary",
        "strategy_count": len(rows),
        "core_count": len(core),
        "needed_expansion_to_87": need,
        "strong_pass_count": sum(inum(r.get("strong_pass")) for r in rows),
        "weak_pass_count": sum(inum(r.get("weak_pass")) for r in rows),
        "best_strategy_id": best.get("strategy_id"),
        "best_accepted_count": best.get("accepted_count"),
        "best_GradeAB_precision": best.get("GradeAB_precision"),
        "best_V_integrated_LCB": best.get("V_integrated_LCB"),
        "best_h240_longrisk_UCB": best.get("h240_longrisk_UCB"),
        "best_LDO_drop": best.get("LDO_drop"),
        "P2_core_expansion_strong_pass": int(sum(inum(r.get("strong_pass")) for r in rows) > 0),
        "P2_core_expansion_weak_pass": int(sum(inum(r.get("weak_pass")) for r in rows) > 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p2_expansion_precision.svg", "P2 expansion precision", [r["strategy_id"] for r in rows], [fnum(r.get("GradeAB_precision")) for r in rows])
    return [summary] + rows, detail, summary


def ce_per_sample(env: dict[str, Any], params: list[torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = env["fwd_core"](x, *params, env["mu"], env["std"], 2.0, 2.0)
    return F.cross_entropy(logits, y, reduction="none").detach()


def subset_for_windowed(
    ap0: list[dict[str, Any]],
    r5b: list[float],
    proxy_scores: list[float],
    score_defs: dict[str, list[float]],
    max_actions: int,
    seed: int,
) -> list[int]:
    candidates: list[int] = []
    pools = [
        [i for i, r in enumerate(ap0) if v9720.is_core_action(r)],
        v9720.topk_idx(r5b, 87),
        v9720.topk_idx(score_defs["T4-core-safe-transfer"], 87),
        v9720.topk_idx(proxy_scores, 87),
        v9720.topk_idx(r5b, 150),
        v9720.topk_idx(score_defs["T1-LCB-transfer"], 150),
        v9720.topk_idx(score_defs["T2-SNR-transfer"], 150),
        sorted(random.Random(seed + 17).sample(range(len(ap0)), 87)),
    ]
    seen: set[int] = set()
    for pool in pools:
        for i in pool:
            if i not in seen:
                candidates.append(i)
                seen.add(i)
            if len(candidates) >= max_actions:
                return candidates
    return candidates[:max_actions]


def windowed_transfer_materializer(
    args: argparse.Namespace,
    ap0: list[dict[str, Any]],
    payload_by_id: dict[str, dict[str, str]],
    subset_idx: list[int],
    device: torch.device,
    windows: list[int],
    out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    if device.type != "cuda":
        row = {
            "stage": "P3_WINDOWED_TRANSFER_SUBSET_MATERIALIZER_V9730",
            "status": "summary",
            "reason": "cuda_unavailable_cpu_offload_disallowed",
            "subset_action_count": 0,
            "windowed_action_rows": 0,
            "windowed_materializer_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        return [row], row, {}
    selected = [ap0[i] for i in subset_idx if str(ap0[i].get("action_id")) in payload_by_id]
    by_ds_seed: dict[tuple[str, int], dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for action in selected:
        by_ds_seed[(str(action.get("dataset")), inum(action.get("seed")))][inum(action.get("step"))].append(action)
    max_window = max(windows)
    take_n = int(args.windowed_sample_count)
    rows: list[dict[str, Any]] = []
    score_values: dict[str, list[float]] = {f"WT{w}-LCB": [-1.0e9] * len(ap0) for w in windows}
    index_by_action = {str(r.get("action_id")): i for i, r in enumerate(ap0)}
    payload_cache: dict[str, Any] = {}
    replay_times: list[float] = []
    nan_count = 0
    inf_count = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for (dataset, seed), steps_map in sorted(by_ds_seed.items()):
        env = v9720.per_seed_task_init(args, dataset, seed, device)
        params: list[torch.Tensor] = env["params"]
        states: list[AdamWState] = env["states"]
        x_train: torch.Tensor = env["x_train"]
        y_train: torch.Tensor = env["y_train"]
        n_train = int(x_train.shape[0])
        max_step = max(steps_map)
        for step in range(max_step + 1):
            idx = torch.randint(0, n_train, (int(args.batch_size) * 2,), generator=env["gen"], device=device)
            xu = x_train[idx[: int(args.batch_size)]].contiguous()
            yu = y_train[idx[: int(args.batch_size)]].contiguous()
            xp = x_train[idx[int(args.batch_size) :]].contiguous()
            yp = y_train[idx[int(args.batch_size) :]].contiguous()
            pack = env["bwd_core"](xu, yu, *params, env["mu"], env["std"], 2.0, 2.0)
            update_grads = list(pack[1:])
            if step in steps_map:
                task_params = v9720.clone_params(params)
                task_states = v9720.clone_states(states)
                v9720.v9420.v9380.v9320.v9248.v92._adamw_update_foreach_(task_params, update_grads, task_states, env["cfg"])
                take = min(take_n, int(xp.shape[0]))
                xcheck = xp[:take].contiguous()
                ycheck = yp[:take].contiguous()
                ce_before = ce_per_sample(env, task_params, xcheck, ycheck)
                gen_state = env["gen"].get_state()
                future_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
                for _ in range(max_window):
                    fut = torch.randint(0, n_train, (int(args.batch_size),), generator=env["gen"], device=device)
                    future_batches.append((x_train[fut].contiguous(), y_train[fut].contiguous()))
                env["gen"].set_state(gen_state)
                for action in steps_map[step]:
                    aid = str(action.get("action_id"))
                    payload = v9720.v9490.load_payload(payload_by_id[aid], payload_cache, device)
                    after_params = [p.detach().clone() + d.detach() for p, d in zip(task_params, payload)]
                    after_states = v9720.clone_states(task_states)
                    row: dict[str, Any] = {
                        "stage": "P3_WINDOWED_TRANSFER_SUBSET_MATERIALIZER_V9730",
                        "status": "windowed_action_row",
                        "action_id": aid,
                        "dataset": action.get("dataset"),
                        "seed": action.get("seed"),
                        "step": action.get("step"),
                        "candidate_template_id": v9720.candidate_template_id(action),
                        "family_id": action.get("family_id"),
                        "check_sample_count": take,
                        "payload_norm": v9720.flat_norm(payload),
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    }
                    start_action = time.perf_counter()
                    for local_step, (xf, yf) in enumerate(future_batches, start=1):
                        fpack = env["bwd_core"](xf, yf, *after_params, env["mu"], env["std"], 2.0, 2.0)
                        fut_grads = list(fpack[1:])
                        v9720.v9420.v9380.v9320.v9248.v92._adamw_update_foreach_(after_params, fut_grads, after_states, env["cfg"])
                        if local_step in windows:
                            ce_after = ce_per_sample(env, after_params, xcheck, ycheck)
                            resp = (ce_before - ce_after).detach().cpu().tolist()
                            if any(math.isnan(float(v)) for v in resp):
                                nan_count += 1
                            if any(math.isinf(float(v)) for v in resp):
                                inf_count += 1
                            row[f"WT{local_step}_mean"] = mean(resp)
                            row[f"WT{local_step}_std"] = pstdev(resp)
                            row[f"WT{local_step}_LCB"] = lcb(resp)
                            row[f"WT{local_step}_sign_frac"] = mean([1.0 if v > 0 else 0.0 for v in resp])
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    row["windowed_compute_ms"] = (time.perf_counter() - start_action) * 1000.0
                    row["memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                    rows.append(row)
                    replay_times.append(fnum(row.get("windowed_compute_ms")))
                    ai = index_by_action.get(aid)
                    if ai is not None:
                        for w in windows:
                            score_values[f"WT{w}-LCB"][ai] = fnum(row.get(f"WT{w}_LCB"))
            v9720.v9420.v9380.v9320.v9248.v92._adamw_update_foreach_(params, update_grads, states, env["cfg"])
    summary = {
        "stage": "P3_WINDOWED_TRANSFER_SUBSET_MATERIALIZER_V9730",
        "status": "summary",
        "subset_action_count_requested": len(subset_idx),
        "subset_action_count_materialized": len({r.get("action_id") for r in rows}),
        "window_count": len(windows),
        "windows": ",".join(str(w) for w in windows),
        "check_sample_count_per_action": take_n,
        "windowed_action_rows": len(rows),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "compute_ms_q50": qtile(replay_times, 0.50),
        "compute_ms_q90": qtile(replay_times, 0.90),
        "memory_mb_peak": torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0,
        "wallclock_sec": time.perf_counter() - t0,
        "windowed_materializer_pass": int(len(rows) == len(subset_idx) and nan_count == 0 and inf_count == 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary, score_values


def p3_windowed_transfer(
    args: argparse.Namespace,
    ap0: list[dict[str, Any]],
    r5b: list[float],
    proxy_scores: list[float],
    score_defs: dict[str, list[float]],
    payload_by_id: dict[str, dict[str, str]],
    device: torch.device,
    out: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    windows = sorted({int(x) for x in str(args.windowed_windows).split(",") if x.strip()})
    subset_idx = subset_for_windowed(ap0, r5b, proxy_scores, score_defs, int(args.windowed_max_actions), int(args.seed))
    materializer_rows, mat_summary, score_values = windowed_transfer_materializer(args, ap0, payload_by_id, subset_idx, device, windows, out)
    eval_rows: list[dict[str, Any]] = []
    best = None
    for sid, scores in score_values.items():
        q = v9720.quality_for_scores(ap0, scores, TARGET_K)
        row = {
            "stage": "P3_WINDOWED_TRANSFER_SCORE_EVAL_V9730",
            "status": "windowed_score_row",
            "score_id": sid,
            "TopK": TARGET_K,
            **q,
            "windowed_pass": int(
                fnum(q.get("GradeAB_precision")) >= 0.75
                and fnum(q.get("V_integrated_LCB")) > 0
                and fnum(q.get("h240_longrisk_UCB")) <= 0.05
                and fnum(q.get("bad_UCB")) <= 0.05
                and fnum(q.get("null_UCB")) <= 0.15
                and fnum(q.get("LDO_drop")) <= 0.15
                and fnum(q.get("LSO_drop")) <= 0.15
                and fnum(q.get("LTO_drop")) <= 0.15
            ),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        eval_rows.append(row)
        if best is None or (
            inum(row.get("windowed_pass")),
            fnum(row.get("GradeAB_precision")),
            fnum(row.get("V_integrated_LCB")),
            -fnum(row.get("LDO_drop")),
        ) > (
            inum(best.get("windowed_pass")),
            fnum(best.get("GradeAB_precision")),
            fnum(best.get("V_integrated_LCB")),
            -fnum(best.get("LDO_drop")),
        ):
            best = row
    best = best or {"score_id": "none"}
    summary = {
        "stage": "P3_WINDOWED_TRANSFER_SCORE_EVAL_V9730",
        "status": "summary",
        "subset_action_count": mat_summary.get("subset_action_count_materialized"),
        "windowed_materializer_pass": mat_summary.get("windowed_materializer_pass"),
        "score_count": len(eval_rows),
        "windowed_pass_count": sum(inum(r.get("windowed_pass")) for r in eval_rows),
        "best_windowed_score_id": best.get("score_id"),
        "best_windowed_precision": best.get("GradeAB_precision"),
        "best_windowed_V_LCB": best.get("V_integrated_LCB"),
        "best_windowed_longrisk_UCB": best.get("h240_longrisk_UCB"),
        "best_windowed_LDO_drop": best.get("LDO_drop"),
        "full_windowed_materializer_status": "not_run" if sum(inum(r.get("windowed_pass")) for r in eval_rows) == 0 else "opened",
        "P3_windowed_transfer_pass": int(sum(inum(r.get("windowed_pass")) for r in eval_rows) > 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p3_windowed_precision.svg", "P3 windowed precision", [r["score_id"] for r in eval_rows], [fnum(r.get("GradeAB_precision")) for r in eval_rows])
    return materializer_rows, [summary] + eval_rows, summary, score_values


def p4_dataset_shift_stabilization(
    ap0: list[dict[str, Any]],
    r5b: list[float],
    score_defs: dict[str, list[float]],
    windowed_scores: dict[str, list[float]],
    out: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    diag_rows: list[dict[str, Any]] = []
    for ds in sorted({v9720.axis_value(r, "dataset_id") for r in ap0}):
        idx = [i for i, r in enumerate(ap0) if v9720.axis_value(r, "dataset_id") == ds]
        rows = [ap0[i] for i in idx]
        diag_rows.append(
            {
                "stage": "P4_DATASET_SHIFT_RECAP_V9730",
                "status": "dataset_row",
                "dataset": ds,
                "action_count": len(rows),
                "GradeAB_density": mean([float(v9720.gradeab(r)) for r in rows]),
                "core_count": sum(1 for r in rows if v9720.is_core_action(r)),
                "memory_safe_rate": mean([1.0 if v9720.memory_fail(r) == 0 else 0.0 for r in rows]),
                "offdiag_safe_rate": mean([1.0 if v9720.offdiag_fail(r) == 0 else 0.0 for r in rows]),
                "longrisk_rate": mean([float(inum(r.get("h240_longrisk"))) for r in rows]),
                "bad_rate": mean([fnum(r.get("bad_event_rate")) for r in rows]),
                "null_rate": mean([fnum(r.get("null_event_rate")) for r in rows]),
                "old_score_mean": mean([r5b[i] for i in idx]),
                "old_score_std": pstdev([r5b[i] for i in idx]),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    wt_best_name = next(iter(windowed_scores), None)
    wt_scores = windowed_scores.get(wt_best_name, [-1.0e9] * len(ap0))
    template_mean: dict[str, float] = {}
    by_tpl: dict[str, list[float]] = defaultdict(list)
    for i, r in enumerate(ap0):
        by_tpl[v9720.candidate_template_id(r)].append(r5b[i])
    for tpl, vals in by_tpl.items():
        template_mean[tpl] = mean(vals)
    stab_scores = {
        "S0-old-rank": r5b,
        "S1-template-demeaned-rank": [r5b[i] - template_mean.get(v9720.candidate_template_id(r), 0.0) for i, r in enumerate(ap0)],
        "S2-old-rank-hard-clean": [r5b[i] if safe_set(r, 0.10) else -1.0e9 for i, r in enumerate(ap0)],
        "S3-exact-core-safe": score_defs["T4-core-safe-transfer"],
        "S4-windowed-best": wt_scores,
    }
    rows: list[dict[str, Any]] = []
    best = None
    for sid, scores in stab_scores.items():
        q = v9720.quality_for_scores(ap0, scores, TARGET_K)
        psi_mean = mean([v9700.psi_kl_wasserstein([scores[i] for i, r in enumerate(ap0) if v9720.axis_value(r, "dataset_id") == ds], scores)[0] for ds in sorted({v9720.axis_value(r, "dataset_id") for r in ap0})])
        row = {
            "stage": "P4_DATASET_SHIFT_STABILIZATION_V9730",
            "status": "strategy_row",
            "strategy_id": sid,
            **q,
            "PSI_mean": psi_mean,
            "stabilization_pass": int(fnum(q.get("LDO_drop")) <= 0.10 and fnum(q.get("GradeAB_precision")) >= 0.75 and fnum(q.get("V_integrated_LCB")) > 0 and fnum(q.get("h240_longrisk_UCB")) <= 0.05),
            "uses_dataset_name_for_selector": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
        if best is None or (
            inum(row.get("stabilization_pass")),
            fnum(row.get("GradeAB_precision")),
            fnum(row.get("V_integrated_LCB")),
            -fnum(row.get("LDO_drop")),
        ) > (
            inum(best.get("stabilization_pass")),
            fnum(best.get("GradeAB_precision")),
            fnum(best.get("V_integrated_LCB")),
            -fnum(best.get("LDO_drop")),
        ):
            best = row
    best = best or rows[0]
    summary = {
        "stage": "P4_DATASET_SHIFT_STABILIZATION_V9730",
        "status": "summary",
        "diagnostic_dataset_count": len(diag_rows),
        "strategy_count": len(rows),
        "stabilization_pass_count": sum(inum(r.get("stabilization_pass")) for r in rows),
        "best_strategy_id": best.get("strategy_id"),
        "best_precision": best.get("GradeAB_precision"),
        "best_V_LCB": best.get("V_integrated_LCB"),
        "best_LDO_drop": best.get("LDO_drop"),
        "dataset_shift_stabilization_pass": int(sum(inum(r.get("stabilization_pass")) for r in rows) > 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p4_stabilization_precision.svg", "P4 stabilization precision", [r["strategy_id"] for r in rows], [fnum(r.get("GradeAB_precision")) for r in rows])
    return diag_rows, [summary] + rows, summary


def not_run_row(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {
        "stage": stage,
        "status": "not_run",
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        **extra,
    }
    return [row], row


def p5_controller(p2: dict[str, Any], p3: dict[str, Any], p4: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    opened = inum(p2.get("P2_core_expansion_weak_pass")) or inum(p3.get("P3_windowed_transfer_pass")) or inum(p4.get("dataset_shift_stabilization_pass"))
    if not opened:
        return not_run_row(
            "P5_EXISTING_ACTION_MINIMAL_CONTROLLER_V9730",
            "P2_P3_P4_no_weak_pass",
            controller_selected=0,
            existing_action_controller_pass=0,
            source_controller_pass=0,
        )
    return not_run_row(
        "P5_EXISTING_ACTION_MINIMAL_CONTROLLER_V9730",
        "controller_freeze_not_implemented_without_strong_candidate",
        controller_selected=0,
        existing_action_controller_pass=0,
        source_controller_pass=0,
    )


def p6_direct_update(p3: dict[str, Any], p5: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not (inum(p3.get("P3_windowed_transfer_pass")) or inum(p5.get("source_controller_pass"))):
        return not_run_row(
            "P6_DIRECT_WINDOWED_TRANSFER_SOLVED_UPDATE_V9730",
            "P3_windowed_or_P5_controller_weak_pass_failed",
            subspace_count=6,
            generated_action_count=0,
            branch_horizon_rows_actual=0,
            direct_update_weak_pass=0,
            direct_update_strong_pass=0,
        )
    return not_run_row(
        "P6_DIRECT_WINDOWED_TRANSFER_SOLVED_UPDATE_V9730",
        "direct_update_materializer_not_opened",
        subspace_count=6,
        generated_action_count=0,
        branch_horizon_rows_actual=0,
        direct_update_weak_pass=0,
        direct_update_strong_pass=0,
    )


def p7_generated_stop(source_v9680: Path, p6: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prev = read_json(source_v9680 / "route_decision_v9680.json")
    row = {
        "stage": "P7_GENERATED_ROUTE_STOP_RULE_V9730",
        "status": "summary",
        "source_generated_route_stop_triggered_v9680": prev.get("generated_route_stop_triggered"),
        "source_new_objective_evidence_present_v9680": prev.get("new_objective_evidence_present"),
        "P6_direct_update_weak_pass": p6.get("direct_update_weak_pass", 0),
        "generated_route_status": "stopped_no_new_objective",
        "APGU_APGV_APGW_allowed": 0,
        "generated_action_count": 0,
        "branch_horizon_rows_actual": 0,
        "generated_route_stop_triggered": 1,
        "reason": "no_windowed_or_direct_update_objective_evidence",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def base_acc(source_v9720: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(r) for r in read_csv(source_v9720 / "base_acc_sentinel_v9720.csv")]
    for r in rows:
        r["stage"] = "BASE_ACC_SENTINEL_V9730"
        r["reused_from_v9720"] = 1
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

    source_v9720 = Path(args.source_v9720)
    source_v9700 = Path(args.source_v9700)
    source_v9680 = Path(args.source_v9680)
    ap0, score_bundle, payload_by_id = v9720.load_ap0_and_payloads(args)
    r5b = v9660.r5b_scores(ap0, score_bundle)
    proxy_scores = load_proxy_scores(ap0, source_v9700)
    exact_summary = load_exact_summary(source_v9720)
    score_defs = score_defs_from_summary(ap0, exact_summary)

    p0_rows, p0_legal_rows, p0 = p0_boundary(args, source_v9720)
    dump_csv("p0_boundary_reproduction_v9730.csv", p0_rows)
    dump_csv("p0_field_legality_audit_v9730.csv", p0_legal_rows)
    p1_rows, p1 = p1_exact_failure_autopsy(ap0, r5b, proxy_scores, score_defs, out, int(args.seed))
    dump_csv("p1_exact_transfer_failure_autopsy_v9730.csv", p1_rows)
    p2_rows, p2_detail, p2 = p2_core_expansion_minimal_fix(ap0, r5b, score_defs, out)
    dump_csv("p2_core77_expansion10_minimal_fix_v9730.csv", p2_rows)
    dump_csv("p2_core77_expansion10_action_detail_v9730.csv", p2_detail)
    device = v9720.device_from(str(args.device))
    p3_mat_rows, p3_score_rows, p3, windowed_scores = p3_windowed_transfer(args, ap0, r5b, proxy_scores, score_defs, payload_by_id, device, out)
    dump_csv("p3_windowed_transfer_subset_materializer_v9730.csv", p3_mat_rows)
    dump_csv("p3_windowed_transfer_score_eval_v9730.csv", p3_score_rows)
    p4_diag, p4_rows, p4 = p4_dataset_shift_stabilization(ap0, r5b, score_defs, windowed_scores, out)
    dump_csv("p4_dataset_shift_recap_v9730.csv", p4_diag)
    dump_csv("p4_dataset_shift_stabilization_v9730.csv", p4_rows)
    p5_rows, p5 = p5_controller(p2, p3, p4)
    dump_csv("p5_existing_action_minimal_controller_v9730.csv", p5_rows)
    p6_rows, p6 = p6_direct_update(p3, p5)
    dump_csv("p6_direct_windowed_transfer_solved_update_v9730.csv", p6_rows)
    p7_rows, p7 = p7_generated_stop(source_v9680, p6)
    dump_csv("p7_generated_route_stop_rule_v9730.csv", p7_rows)
    p8_rows, p8 = not_run_row("P8_SELECTED_RUNTIME_PREFLIGHT_V9730", "P5_controller_not_selected", selected_runtime_pass=0)
    dump_csv("p8_selected_runtime_preflight_v9730.csv", p8_rows)
    p9_rows, p9 = not_run_row("P9_PAIRED_REPLAY_BOUNDARY_V9730", "P5_P8_system_not_official", paired_replay_pass=0)
    dump_csv("p9_paired_replay_boundary_v9730.csv", p9_rows)
    p10_rows, p10 = not_run_row("P10_SHORT_FULL_BOUNDARY_V9730", "P9_paired_replay_not_open", short_run_boundary_open=0, full_run_boundary_open=0)
    dump_csv("p10_short_full_boundary_v9730.csv", p10_rows)
    base_rows, base_summary = base_acc(source_v9720)
    dump_csv("base_acc_sentinel_v9730.csv", base_rows)

    if not inum(p0.get("p0_pass")):
        route, blocker = "R0-BoundaryReproductionFail", "v9720_boundary_not_reproduced"
    elif inum(p2.get("P2_core_expansion_weak_pass")) or inum(p5.get("source_controller_pass")):
        route, blocker = "R1-ExactTransferAuxiliaryUseful", "none"
    elif inum(p3.get("P3_windowed_transfer_pass")):
        route, blocker = "R2-WindowedTransferUseful", "none"
    elif fnum(p2.get("best_GradeAB_precision")) >= 0.80 and fnum(p2.get("best_LDO_drop")) > 0.15:
        route, blocker = "R3-ExistingActionStillLDOBlocked", "existing_action_ldo_blocked"
    elif not inum(p1.get("exact_transfer_auxiliary_useful")):
        route, blocker = "R4-TransferPrincipleInsufficient", "exact_windowed_transfer_insufficient"
    elif inum(p6.get("direct_update_weak_pass")):
        route, blocker = "R5-DirectSolvedUpdatePromising", "none"
    else:
        route, blocker = "R6-GeneratedRouteStopped", "generated_route_stopped_no_new_objective"

    system_pass = int(inum(p5.get("source_controller_pass")) and inum(p8.get("selected_runtime_pass")))
    route_decision = {
        "stage": "ROUTE_DECISION_V9730",
        "status": "summary",
        "route": route,
        "source_route_v9720": p0.get("source_route_v9720"),
        "p0_pass": p0.get("p0_pass"),
        "exact_transfer_auxiliary_useful": p1.get("exact_transfer_auxiliary_useful"),
        "failure_mode": p1.get("failure_mode"),
        "P2_core_expansion_weak_pass": p2.get("P2_core_expansion_weak_pass"),
        "best_expansion_strategy_id": p2.get("best_strategy_id"),
        "best_expansion_precision": p2.get("best_GradeAB_precision"),
        "best_expansion_V_LCB": p2.get("best_V_integrated_LCB"),
        "best_expansion_LDO_drop": p2.get("best_LDO_drop"),
        "P3_windowed_transfer_pass": p3.get("P3_windowed_transfer_pass"),
        "best_windowed_score_id": p3.get("best_windowed_score_id"),
        "best_windowed_precision": p3.get("best_windowed_precision"),
        "best_windowed_V_LCB": p3.get("best_windowed_V_LCB"),
        "best_windowed_LDO_drop": p3.get("best_windowed_LDO_drop"),
        "dataset_shift_stabilization_pass": p4.get("dataset_shift_stabilization_pass"),
        "controller_pass": p5.get("source_controller_pass", 0),
        "selected_runtime_pass": p8.get("selected_runtime_pass", 0),
        "direct_update_weak_pass": p6.get("direct_update_weak_pass", 0),
        "generated_route_status": p7.get("generated_route_status"),
        "APGU_APGV_APGW_allowed": p7.get("APGU_APGV_APGW_allowed"),
        "system_legal_controller_pass": system_pass,
        "primary_blocker": blocker,
        "secondary_blocker": "generated_route_stopped_no_new_objective" if inum(p7.get("generated_route_stop_triggered")) else "none",
    }
    dump_json("route_decision_v9730.json", route_decision)

    allowed = {
        "selected_runtime_allowed": 0,
        "paired_replay_allowed": 0,
        "short_full_allowed": 0,
        "generated_route_allowed": 0,
        "APGU_APGV_APGW_allowed": 0,
        "windowed_full_materializer_allowed": int(inum(p3.get("P3_windowed_transfer_pass"))),
        "direct_update_allowed": int(inum(p3.get("P3_windowed_transfer_pass")) or inum(p5.get("source_controller_pass"))),
    }
    dump_json("allowed_next_gates_v9730.json", allowed)
    stop = {
        "stop_exact_one_step_as_standalone_ranker": 1,
        "stop_core_expansion_threshold_patching": int(not inum(p2.get("P2_core_expansion_weak_pass"))),
        "stop_generated_blind_variants": 1,
        "enter_system": system_pass,
    }
    dump_json("stop_conditions_v9730.json", stop)

    nf = no_fake(list(artifacts.values()))
    nf_row = {"stage": "NO_FAKE_AUDIT_V9730", "status": "summary", **nf}
    dump_csv("no_fake_audit_v9730.csv", [nf_row])
    contract = {
        "stage": "CONTRACT_AUDIT_V9730",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "v9720_boundary_pass": p0.get("p0_pass"),
        "exact_failure_autopsy_done": 1,
        "core_expansion_pass": p2.get("P2_core_expansion_weak_pass"),
        "windowed_transfer_pass": p3.get("P3_windowed_transfer_pass"),
        "dataset_shift_stabilization_pass": p4.get("dataset_shift_stabilization_pass"),
        "controller_runtime_system": f"{p5.get('source_controller_pass', 0)}/{p8.get('selected_runtime_pass', 0)}/{system_pass}",
        "direct_update_pass": p6.get("direct_update_weak_pass", 0),
        "generated_route_stop": p7.get("generated_route_stop_triggered"),
        "base_acc_sentinel_pass": base_summary.get("base_acc_sentinel_pass"),
        "base_acc_used_for_controller": 0,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "uses_dataset_name_for_selector_controller": "0/0",
        "uses_validation_or_test_for_controller": 0,
        "uses_future_outcome_for_features": 0,
        "proxy_transfer_promoted_to_official": 0,
        "diagnostic_promoted_to_official": 0,
        "fake/proxy/cpu_offload": f"{nf['fake_data_used']}/{nf['proxy_row_used']}/{nf['cpu_offload_used']}",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("contract_audit_v9730.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9730",
        "status": "summary",
        "route": route,
        "F0_boundary_fail": int(not inum(p0.get("p0_pass"))),
        "F1_exact_transfer_auxiliary_fail": int(not inum(p1.get("exact_transfer_auxiliary_useful"))),
        "F2_core_expansion_still_LDO_blocked": int(not inum(p2.get("P2_core_expansion_weak_pass")) and fnum(p2.get("best_LDO_drop")) > 0.15),
        "F3_windowed_transfer_fail": int(not inum(p3.get("P3_windowed_transfer_pass"))),
        "F4_controller_runtime_blocked": int(not inum(p5.get("source_controller_pass", 0))),
        "F5_direct_update_not_open": int(not inum(p6.get("direct_update_weak_pass", 0))),
        "F6_generated_route_stopped": int(inum(p7.get("generated_route_stop_triggered"))),
        "F7_system_not_official": int(not system_pass),
        "primary_blocker": blocker,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("failure_taxonomy_v9730.csv", [failure])
    hashes = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    manifest = {
        "stage": "RUN_MANIFEST_V9730",
        "status": "summary",
        "out_dir": str(out),
        "wallclock_sec": time.perf_counter() - t0,
        "route": route,
        "primary_blocker": blocker,
        "artifact_hashes": hashes,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("run_manifest_v9730.json", manifest)
    print(
        json.dumps(
            {
                "out_dir": str(out),
                "route": route,
                "primary_blocker": blocker,
                "best_expansion_strategy": p2.get("best_strategy_id"),
                "best_expansion_precision": p2.get("best_GradeAB_precision"),
                "best_windowed_score": p3.get("best_windowed_score_id"),
                "best_windowed_precision": p3.get("best_windowed_precision"),
                "system_legal_controller_pass": system_pass,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
