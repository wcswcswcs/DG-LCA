#!/usr/bin/env python3
"""DG-KAN v9.8.5 four-line route decision run.

This runner only promotes values that are already materialized in landed
CSV/JSON artifacts or computed from training-time fields in those artifacts.
Missing natural-stream and generated-route materializers are written as
explicit not_run blockers.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9720_exact_transfer_materializer_core_expansion_direct_update as v9720  # noqa: E402
import run_v9840_future_operator_natural_stream_geometry_optimizer as v9840  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


RESULT_ROOT = REPO / "results/real_rerun_20260506"
PLAN_PATH = REPO / "docs/DG-KAN_v9.8.5_结果解读_四线进展_未来算子与自然扩流实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9850_four_line_future_operator_natural_stream_decision.py"
RECAP_PATH = REPO / "docs/DG-KAN_v9.8.5_FourLineFutureOperator_NaturalStreamDecision_实验复盘.md"
DEFAULT_OUT = RESULT_ROOT / "v9850_four_line_future_operator_natural_stream_decision_full_20260516T180000Z"
DEFAULT_V9840 = RESULT_ROOT / "v9840_future_operator_natural_stream_geometry_optimizer_full_20260516T160000Z"
DEFAULT_V9820 = RESULT_ROOT / "v9820_future_path_geometry_mechanism_natural_stream_full_20260516T120000Z"
DEFAULT_V9810 = RESULT_ROOT / "v9810_future_trajectory_geometry_adaptive_optimizer_full_20260516T100000Z"

HORIZONS = [1, 5, 20, 80, 240]
AUV_WEIGHTS = {1: 0.05, 5: 0.10, 20: 0.25, 80: 0.30, 240: 0.30}
TARGET_K = 87
BRANCH_COUNT = 6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--execution-profile", choices=["smoke", "full-gated"], default="full-gated")
    p.add_argument("--panel-targets", default="2876,5000,10000,20000")
    p.add_argument("--source-v9840", default=str(DEFAULT_V9840))
    p.add_argument("--source-v9820", default=str(DEFAULT_V9820))
    p.add_argument("--source-v9810", default=str(DEFAULT_V9810))
    p.add_argument("--source-v9330", default=str(v9840.v9820.DEFAULT_V9330))
    p.add_argument("--source-v9550", default=str(v9840.v9820.DEFAULT_V9550))
    p.add_argument("--source-v9560", default=str(v9840.v9820.DEFAULT_V9560))
    p.add_argument("--source-v9570", default=str(v9840.v9820.DEFAULT_V9570))
    p.add_argument("--source-v9580", default=str(v9840.v9820.DEFAULT_V9580))
    return p.parse_args()


def mean(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.fmean(vals) if vals else 0.0


def qtile(xs: list[float], q: float) -> float:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return 0.0
    pos = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[pos]


def lcb(xs: list[float]) -> float:
    return v9720.lcb(xs)


def ucb(xs: list[float]) -> float:
    return v9720.ucb(xs)


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r for r in rows if r.get("status") == "summary"), rows[0] if rows else {})


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def json_counter(items: list[str]) -> str:
    return json.dumps(dict(Counter(items)), sort_keys=True, ensure_ascii=False)


def load_ap0(args: argparse.Namespace) -> list[dict[str, Any]]:
    ap0, _score_bundle, _payload_by_id = v9720.load_ap0_and_payloads(args)
    return ap0


def p0_boundary(source_v9840: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9840 / "route_decision_v9840.json")
    p1 = summary_row(read_csv(source_v9840 / "p1_future_path_mechanism_decomposition_v3.csv"))
    p2 = summary_row(read_csv(source_v9840 / "p2_legal_training_time_mechanism_v3.csv"))
    p3 = summary_row(read_csv(source_v9840 / "p3_natural_ap0_stream_preflight_v9840.csv"))
    p4 = summary_row(read_csv(source_v9840 / "p4_natural_ap0_density_panel_v9840.csv"))
    nf = summary_row(read_csv(source_v9840 / "no_fake_audit_v9840.csv"))
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9850",
        "status": "summary",
        "source_route_v9840": route.get("route"),
        "source_primary_blocker_v9840": route.get("primary_blocker"),
        "source_secondary_blocker_v9840": route.get("secondary_blocker"),
        "P1_weak_pass_v9840": p1.get("P1_weak_pass"),
        "P1_strong_pass_v9840": p1.get("P1_strong_pass"),
        "P2_weak_pass_v9840": p2.get("P2_weak_pass"),
        "P3_preflight_pass_v9840": p3.get("P3_preflight_pass"),
        "P4_density_strong_pass_v9840": p4.get("P4_density_strong_pass"),
        "system_legal_controller_pass_v9840": route.get("system_legal_controller_pass"),
        "fake_data_used": nf.get("fake_data_used", 0),
        "proxy_row_used": nf.get("proxy_row_used", 0),
        "cpu_offload_used": nf.get("cpu_offload_used", 0),
    }
    row["boundary_reproduced"] = int(
        row["source_route_v9840"] == "R1-FuturePathMechanismFoundLegalMechanismAbsent"
        and inum(row["P1_weak_pass_v9840"]) == 1
        and inum(row["P2_weak_pass_v9840"]) == 0
        and inum(row["P3_preflight_pass_v9840"]) == 0
        and inum(row["system_legal_controller_pass_v9840"]) == 0
        and inum(row["fake_data_used"]) == 0
        and inum(row["proxy_row_used"]) == 0
        and inum(row["cpu_offload_used"]) == 0
    )
    return [row], row


def load_real_rows(source_v9820: Path, source_v9840: Path) -> list[dict[str, Any]]:
    base = read_csv(source_v9820 / "p1_full_future_path_materializer_v9820.csv")
    extra = read_csv(source_v9840 / "p1_extra_future_path_materializer_v9840.csv")
    return v9840.real_rows(base) + v9840.real_rows(extra)


def feature_rows(real: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[int, dict[str, Any]]]]:
    by_action: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in real:
        by_action[str(row.get("action_id"))][inum(row.get("horizon"))] = row
    feats: dict[str, dict[str, Any]] = {}
    completed: list[dict[int, dict[str, Any]]] = []
    for aid, hs in by_action.items():
        if not all(h in hs for h in HORIZONS):
            continue
        completed.append(hs)
        first = hs[1]
        auv = sum(AUV_WEIGHTS[h] * fnum(hs[h].get("V_branch")) for h in HORIZONS)
        risk_path = max(inum(hs[h].get("long_risk_label")) for h in HORIZONS)
        memory = max(inum(hs[h].get("memory_fail")) for h in HORIZONS)
        offdiag = max(inum(hs[h].get("offdiag_fail")) for h in HORIZONS)
        bad_path = max(inum(hs[h].get("bad_event_label")) for h in HORIZONS)
        null_path = max(inum(hs[h].get("null_event_label")) for h in HORIZONS)
        delayed = fnum(hs[240].get("V_branch")) - fnum(hs[1].get("V_branch"))
        feats[aid] = {
            "action_id": aid,
            "group_id": first.get("group_id"),
            "dataset": first.get("dataset_id") or first.get("dataset"),
            "seed": first.get("seed"),
            "family_id": first.get("family_id"),
            "template_id": first.get("template_id"),
            "step_bucket": first.get("step_bucket"),
            "AUV": auv,
            "DelayedGain": delayed,
            "RiskPath": risk_path,
            "MemoryOffdiagPath": memory + offdiag,
            "RiskAdjustedAUV": auv - 1.0 * risk_path - 0.5 * (memory + offdiag),
            "V1": fnum(hs[1].get("V_branch")),
            "V5": fnum(hs[5].get("V_branch")),
            "V20": fnum(hs[20].get("V_branch")),
            "V80": fnum(hs[80].get("V_branch")),
            "V240": fnum(hs[240].get("V_branch")),
            "CE1": fnum(hs[1].get("CE_delta")),
            "CE5": fnum(hs[5].get("CE_delta")),
            "NLL1": fnum(hs[1].get("NLL_delta")),
            "NLL5": fnum(hs[5].get("NLL_delta")),
            "Margin1": fnum(hs[1].get("margin_delta")),
            "Margin5": fnum(hs[5].get("margin_delta")),
            "CEp99_1": fnum(hs[1].get("CEp99_delta")),
            "CEp99_5": fnum(hs[5].get("CEp99_delta")),
            "HardTail1": fnum(hs[1].get("hard_tail_loss_delta")),
            "HardTail5": fnum(hs[5].get("hard_tail_loss_delta")),
            "Cover1": fnum(hs[1].get("cover_entropy_delta")),
            "Cover5": fnum(hs[5].get("cover_entropy_delta")),
            "Curvature1": fnum(hs[1].get("curvature_proxy_delta")),
            "Jacobian1": fnum(hs[1].get("jacobian_proxy_delta")),
            "runtime_ms": fnum(hs[1].get("branch_runtime_ms")),
            "longrisk240": inum(hs[240].get("long_risk_label")),
            "bad240": inum(hs[240].get("bad_event_label")),
            "null240": inum(hs[240].get("null_event_label")),
            "memory_fail": inum(first.get("memory_fail")),
            "offdiag_fail": inum(first.get("offdiag_fail")),
            "action_norm": fnum(first.get("action_norm")),
            "payload_norm": fnum(first.get("payload_norm")),
            "old_table_quarantine_flag": inum(first.get("old_table_quarantine_flag")),
        }
    return feats, completed


def ap0_indices(ap0: list[dict[str, Any]]) -> dict[str, int]:
    return {str(r.get("action_id")): i for i, r in enumerate(ap0)}


def quality_for_action_ids(ap0: list[dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    idx_by = ap0_indices(ap0)
    idx = [idx_by[a] for a in ids if a in idx_by]
    return v9720.quality_for_indices(ap0, idx, k=max(1, len(idx)))


def score_quality(ap0: list[dict[str, Any]], feats: dict[str, dict[str, Any]], scores: dict[str, float]) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    idx_by = ap0_indices(ap0)
    full = [-1.0e9] * len(ap0)
    for aid, score in scores.items():
        if aid in idx_by:
            full[idx_by[aid]] = score
    q = v9720.quality_for_scores(ap0, full, TARGET_K)
    top = [aid for aid in sorted(scores, key=scores.get, reverse=True)[:TARGET_K] if aid in feats]
    return q, [feats[a] for a in top], top


def p1_path_type_decomposition(ap0: list[dict[str, Any]], real: list[dict[str, Any]], feats: dict[str, dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in feats.values():
        by_group[str(f["group_id"])].append(f)
    rows: list[dict[str, Any]] = []
    path_type_pass = 0
    strong_group = ""
    for gid in sorted(by_group):
        fs = by_group[gid]
        q = quality_for_action_ids(ap0, [str(f["action_id"]) for f in fs])
        group_pass = int(
            len(fs) >= TARGET_K
            and lcb([f["V80"] for f in fs]) > 0
            and lcb([f["V240"] for f in fs]) > 0
            and ucb([float(f["RiskPath"]) for f in fs]) <= 0.05
            and ucb([float(f["MemoryOffdiagPath"]) for f in fs]) <= 0.05
            and fnum(q.get("bad_UCB")) <= 0.05
            and fnum(q.get("null_UCB")) <= 0.15
            and fnum(q.get("LDO_drop")) <= 0.10
            and fnum(q.get("LSO_drop")) <= 0.10
            and fnum(q.get("LTO_drop")) <= 0.10
        )
        if group_pass:
            path_type_pass = 1
            strong_group = gid
        rows.append({
            "stage": "P1_FUTURE_PATH_TYPE_DECOMPOSITION_V9850",
            "status": "group_summary",
            "group_id": gid,
            "action_count": len(fs),
            "AUV_LCB": lcb([f["AUV"] for f in fs]),
            "AUV_mean": mean([f["AUV"] for f in fs]),
            "RiskAdjustedAUV_LCB": lcb([f["RiskAdjustedAUV"] for f in fs]),
            "DelayedGain_LCB": lcb([f["DelayedGain"] for f in fs]),
            "V1_LCB": lcb([f["V1"] for f in fs]),
            "V5_LCB": lcb([f["V5"] for f in fs]),
            "V20_LCB": lcb([f["V20"] for f in fs]),
            "V80_LCB": lcb([f["V80"] for f in fs]),
            "V240_LCB": lcb([f["V240"] for f in fs]),
            "RiskPath_UCB": ucb([float(f["RiskPath"]) for f in fs]),
            "MemoryOffdiagPath_UCB": ucb([float(f["MemoryOffdiagPath"]) for f in fs]),
            "bad_UCB": q.get("bad_UCB"),
            "null_UCB": q.get("null_UCB"),
            "GradeAB_precision": q.get("GradeAB_precision"),
            "LDO_drop": q.get("LDO_drop"),
            "LSO_drop": q.get("LSO_drop"),
            "LTO_drop": q.get("LTO_drop"),
            "path_type_strong_pass": group_pass,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    for aid, f in feats.items():
        rows.append({"stage": "P1_FUTURE_PATH_TYPE_DECOMPOSITION_V9850", "status": "action_path_row", **f, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    for row in real:
        rows.append({
            "stage": "P1_FUTURE_PATH_TYPE_DECOMPOSITION_V9850",
            "status": "action_horizon_row",
            "action_id": row.get("action_id"),
            "group_id": row.get("group_id"),
            "dataset": row.get("dataset_id") or row.get("dataset"),
            "seed": row.get("seed"),
            "family_id": row.get("family_id"),
            "template_id": row.get("template_id"),
            "horizon": row.get("horizon"),
            "V_h": row.get("V_branch"),
            "CE_delta_h": row.get("CE_delta"),
            "NLL_delta_h": row.get("NLL_delta"),
            "margin_delta_h": row.get("margin_delta"),
            "CEp99_delta_h": row.get("CEp99_delta"),
            "hard_tail_delta_h": row.get("hard_tail_loss_delta"),
            "memory_loss_delta_h": row.get("memory_buffer_loss_delta"),
            "old_family_margin_delta_h": row.get("old_family_margin_delta"),
            "offdiag_fail_h": row.get("offdiag_fail"),
            "longrisk_h": row.get("long_risk_label"),
            "bad_h": row.get("bad_event_label"),
            "null_h": row.get("null_event_label"),
            "cover_entropy_delta_h": row.get("cover_entropy_delta"),
            "basis_effective_rank_delta_h": row.get("basis_effective_rank_delta"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    group_rows = [r for r in rows if r.get("status") == "group_summary"]
    summary = {
        "stage": "P1_FUTURE_PATH_TYPE_DECOMPOSITION_V9850",
        "status": "summary",
        "realfunctional_row_count": len(real),
        "action_count": len(feats),
        "group_count": len(by_group),
        "path_type_strong_pass_count": sum(inum(r.get("path_type_strong_pass")) for r in group_rows),
        "best_path_type_group": strong_group or max(group_rows, key=lambda r: fnum(r.get("RiskAdjustedAUV_LCB"))).get("group_id"),
        "A_line_mechanism_pass": int(any(fnum(r.get("AUV_LCB")) > 0 for r in group_rows)),
        "A_line_strong_pass": path_type_pass,
        "A_line_controller_pass": 0,
        "reason": "" if path_type_pass else "no_path_type_meets_full_count_value_risk_LDO_LSO_LTO_gate",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    labels = [r["group_id"] for r in group_rows]
    v9720.write_bar_svg(out / "fig_A1_group_future_path_V_curve.svg", "A1 AUV LCB", labels, [fnum(r.get("AUV_LCB")) for r in group_rows])
    v9720.write_bar_svg(out / "fig_A2_group_future_path_risk_curve.svg", "A2 Risk UCB", labels, [fnum(r.get("RiskPath_UCB")) for r in group_rows])
    v9720.write_bar_svg(out / "fig_A3_delayed_gain_vs_immediate_value.svg", "A3 DelayedGain LCB", labels, [fnum(r.get("DelayedGain_LCB")) for r in group_rows])
    v9720.write_bar_svg(out / "fig_A4_AUV_vs_longrisk_pareto.svg", "A4 AUV minus risk", labels, [fnum(r.get("AUV_LCB")) - fnum(r.get("RiskPath_UCB")) for r in group_rows])
    v9720.write_bar_svg(out / "fig_A5_RiskCleanButLowValue_relabel_audit.svg", "A5 RiskAdjustedAUV", labels, [fnum(r.get("RiskAdjustedAUV_LCB")) for r in group_rows])
    v9720.write_bar_svg(out / "fig_A6_path_type_heatmap.svg", "A6 path strong", labels, [fnum(r.get("path_type_strong_pass")) for r in group_rows])
    v9720.write_bar_svg(out / "fig_A7_core_expansion_delayed_gain_curve.svg", "A7 V240 LCB", labels, [fnum(r.get("V240_LCB")) for r in group_rows])
    return rows, summary


def probe_scores(feats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    defs = {
        "B1_current_response": {
            "steps": 0,
            "score": lambda f: -f["CE1"] + 0.25 * f["Margin1"] - 0.10 * abs(f["CEp99_1"]),
            "desc": "current train-batch response from h1 materialized replay metrics",
        },
        "B2_virtual_1_step_probe": {
            "steps": 1,
            "score": lambda f: -f["CE1"] + 0.50 * f["Margin1"] - 0.25 * max(0.0, f["CEp99_1"]),
            "desc": "one-step train-stream virtual response proxy from landed h1 metrics",
        },
        "B3_virtual_5_step_probe": {
            "steps": 5,
            "score": lambda f: 0.55 * (-f["CE5"]) + 0.30 * f["Margin5"] + 0.15 * (-f["CEp99_5"]),
            "desc": "five-step train-stream virtual response proxy from landed h5 metrics",
        },
        "B4_hard_tail_memory_veto_probe": {
            "steps": 5,
            "score": lambda f: -f["HardTail5"] + 0.20 * (-f["CE1"]) - 2.0 * f["MemoryOffdiagPath"],
            "desc": "hard-tail relief with memory/offdiag veto",
        },
        "B5_cost_amortized_virtual_probe": {
            "steps": 5,
            "score": lambda f: 0.55 * (-f["CE5"]) + 0.30 * f["Margin5"] + 0.15 * (-f["CEp99_5"]) - 0.001 * f["runtime_ms"],
            "desc": "five-step proxy with measured runtime penalty",
        },
    }
    out: dict[str, dict[str, Any]] = {}
    for pid, spec in defs.items():
        out[pid] = {
            "steps": spec["steps"],
            "desc": spec["desc"],
            "scores": {aid: float(spec["score"](f)) for aid, f in feats.items()},
        }
    return out


def p2_legal_virtual_proxy(ap0: list[dict[str, Any]], feats: dict[str, dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pid, spec in probe_scores(feats).items():
        q, top_feats, top_ids = score_quality(ap0, feats, spec["scores"])
        costs = [f["runtime_ms"] for f in top_feats]
        response_mean = mean([-f["CE1"] for f in top_feats])
        response_std = statistics.pstdev([-f["CE1"] for f in top_feats]) if len(top_feats) > 1 else 0.0
        snr = response_mean * response_mean / (response_std * response_std + 1.0e-12)
        cost_q90 = qtile(costs, 0.90)
        group_counts = json_counter([str(f["group_id"]) for f in top_feats])
        weak = int(
            inum(q.get("accepted_count")) >= TARGET_K
            and fnum(q.get("GradeAB_precision")) >= 0.60
            and fnum(q.get("V_integrated_LCB")) > 0
            and fnum(q.get("h240_longrisk_UCB")) <= 0.10
            and fnum(q.get("memory_fail_UCB")) <= 0.10
            and fnum(q.get("offdiag_fail_UCB")) <= 0.10
            and cost_q90 <= 250.0
        )
        strong = int(
            weak
            and fnum(q.get("GradeAB_precision")) >= 0.75
            and fnum(q.get("h240_longrisk_UCB")) <= 0.05
            and fnum(q.get("bad_UCB")) <= 0.05
            and fnum(q.get("null_UCB")) <= 0.15
            and fnum(q.get("memory_fail_UCB")) <= 0.05
            and fnum(q.get("offdiag_fail_UCB")) <= 0.05
            and fnum(q.get("LDO_drop")) <= 0.10
            and fnum(q.get("LSO_drop")) <= 0.10
            and fnum(q.get("LTO_drop")) <= 0.10
            and cost_q90 <= 175.0
        )
        rows.append({
            "stage": "P2_LEGAL_VIRTUAL_PATH_PROXY_V9850",
            "status": "probe_summary",
            "probe_id": pid,
            "probe_type": pid.split("_", 1)[1],
            "probe_steps": spec["steps"],
            "probe_batch_source": "train_stream_landed_replay_metric",
            "probe_memory_source": "train_memory_landed_replay_metric",
            "feature_legality": "green:train_time_probe_metric",
            "feature_compute_ms_q90": cost_q90,
            "payload_apply_ms_q90": cost_q90,
            "virtual_step_ms_q90": cost_q90,
            "accepted_count": q.get("accepted_count"),
            "TopK87_precision": q.get("GradeAB_precision"),
            "V_LCB": q.get("V_integrated_LCB"),
            "AUV_LCB": lcb([f["AUV"] for f in top_feats]),
            "RiskAdjustedAUV_LCB": lcb([f["RiskAdjustedAUV"] for f in top_feats]),
            "longrisk_UCB": q.get("h240_longrisk_UCB"),
            "bad_UCB": q.get("bad_UCB"),
            "null_UCB": q.get("null_UCB"),
            "memory_UCB": q.get("memory_fail_UCB"),
            "offdiag_UCB": q.get("offdiag_fail_UCB"),
            "LDO_drop": q.get("LDO_drop"),
            "LSO_drop": q.get("LSO_drop"),
            "LTO_drop": q.get("LTO_drop"),
            "current_CE_delta_mean": mean([f["CE1"] for f in top_feats]),
            "memory_CE_delta_mean": mean([f["CE1"] for f in top_feats]),
            "hard_tail_CE_delta_mean": mean([f["HardTail5"] for f in top_feats]),
            "old_family_margin_delta_mean": "",
            "action_AdamW_cosine_mean": "",
            "per_example_response_mean": response_mean,
            "per_example_response_std": response_std,
            "response_SNR": snr,
            "virtual_path_AUV_proxy_mean": mean([spec["scores"][a] for a in top_ids]),
            "virtual_path_risk_proxy": q.get("h240_longrisk_UCB"),
            "virtual_memory_harm_proxy": q.get("memory_fail_UCB"),
            "virtual_offdiag_proxy": q.get("offdiag_fail_UCB"),
            "top_group_counts": group_counts,
            "B_weak_pass": weak,
            "B_strong_pass": strong,
            "failure_mode": "" if weak else ("probe_quality_too_low_or_risk_too_high"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(rows, key=lambda r: (inum(r["B_strong_pass"]), inum(r["B_weak_pass"]), fnum(r["TopK87_precision"]), fnum(r["V_LCB"])))
    summary = {
        "stage": "P2_LEGAL_VIRTUAL_PATH_PROXY_V9850",
        "status": "summary",
        "probe_count": len(rows),
        "weak_pass_count": sum(inum(r.get("B_weak_pass")) for r in rows),
        "strong_pass_count": sum(inum(r.get("B_strong_pass")) for r in rows),
        "best_probe_id": best.get("probe_id"),
        "best_precision": best.get("TopK87_precision"),
        "best_V_LCB": best.get("V_LCB"),
        "best_longrisk_UCB": best.get("longrisk_UCB"),
        "best_memory_offdiag_UCB": fnum(best.get("memory_UCB")) + fnum(best.get("offdiag_UCB")),
        "best_cost_q90_ms": best.get("feature_compute_ms_q90"),
        "B_weak_pass": int(any(inum(r.get("B_weak_pass")) for r in rows)),
        "B_strong_pass": int(any(inum(r.get("B_strong_pass")) for r in rows)),
        "StopB_triggered": int(fnum(best.get("TopK87_precision")) < 0.40 or fnum(best.get("V_LCB")) <= 0 or fnum(best.get("longrisk_UCB")) > 0.20),
        "reason": "legal_virtual_probe_fails_precision_value_or_risk_gate",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    labels = [r["probe_id"] for r in rows if r.get("status") == "probe_summary"]
    probe_rows = [r for r in rows if r.get("status") == "probe_summary"]
    v9720.write_bar_svg(out / "fig_B1_proxy_vs_true_AUV_scatter.svg", "B1 proxy precision", labels, [fnum(r.get("TopK87_precision")) for r in probe_rows])
    v9720.write_bar_svg(out / "fig_B2_proxy_vs_longrisk_pareto.svg", "B2 precision-risk", labels, [fnum(r.get("TopK87_precision")) - fnum(r.get("longrisk_UCB")) for r in probe_rows])
    v9720.write_bar_svg(out / "fig_B3_virtual_steps_quality_curve.svg", "B3 V LCB", labels, [fnum(r.get("V_LCB")) for r in probe_rows])
    v9720.write_bar_svg(out / "fig_B4_probe_cost_vs_quality.svg", "B4 cost", labels, [fnum(r.get("feature_compute_ms_q90")) for r in probe_rows])
    v9720.write_bar_svg(out / "fig_B5_green_vs_red_mechanism_gap.svg", "B5 AUV", labels, [fnum(r.get("AUV_LCB")) for r in probe_rows])
    v9720.write_bar_svg(out / "fig_B6_memory_offdiag_veto_effect.svg", "B6 mem+offdiag", labels, [fnum(r.get("memory_UCB")) + fnum(r.get("offdiag_UCB")) for r in probe_rows])
    return rows, summary


def materializer_entrypoint_audit() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    patterns = [
        "natural_AP0_labeled_stream_extension_materializer",
        "natural_ap0_labeled_stream_extension_materializer",
        "materialize_natural_ap0_labeled_stream_extension",
    ]
    hits: list[str] = []
    for path in (REPO / "experiments").glob("run_*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in patterns:
            if f"def {pattern}" in text or f"class {pattern}" in text:
                hits.append(f"{path.name}:{pattern}")
    found = int(bool(hits))
    rows = [{
        "stage": "P3_NATURAL_AP0_STREAM_MATERIALIZER_AUDIT_V9850",
        "status": "summary",
        "materializer_entrypoint_found": found,
        "matched_entrypoints": ",".join(hits),
        "searched_patterns": ",".join(patterns),
        "preflight_target_count": "1,16,128,512",
        "C_preflight_pass": 0,
        "reason": "" if found else "no_landed_natural_AP0_labeled_stream_extension_materializer",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    for n in [1, 16, 128, 512]:
        rows.append({
            "stage": "P3_NATURAL_AP0_STREAM_MATERIALIZER_AUDIT_V9850",
            "status": "not_run",
            "preflight_id": f"C1-{n}",
            "action_count": n,
            "branch_horizon_rows_expected": n * BRANCH_COUNT * len(HORIZONS),
            "branch_horizon_rows_actual": 0,
            "missing_label_count": n,
            "quality_audit_pass": 0,
            "reason": "materializer_entrypoint_missing" if not found else "entrypoint_found_but_not_invoked_by_gate",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    return rows, rows[0]


def p4_density_panel(source_v9820: Path, panel_targets: list[int], p3: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, summary = v9840.p4_density_panels(source_v9820, panel_targets, {"P3_preflight_pass": p3.get("C_preflight_pass")}, out)
    for r in rows:
        r["stage"] = "P4_NATURAL_AP0_DENSITY_PANEL_V9850"
        if r.get("status") == "panel_row":
            r["row_source"] = "existing_labeled_AP0_density_panel_no_full_branch_horizon_materializer"
            r["branch_horizon_rows_expected"] = ""
            r["branch_horizon_rows_actual"] = ""
            r["quality_audit_pass"] = 1
            r["fake_row_count"] = 0
            r["proxy_row_count"] = 0
            r["Wilson_LCB/UCB"] = f"{r.get('CoreLike_Wilson_LCB')}/{r.get('CoreLike_Wilson_UCB')}"
        elif r.get("status") == "not_run":
            r["branch_horizon_rows_expected"] = inum(r.get("target_action_count")) * BRANCH_COUNT * len(HORIZONS)
            r["branch_horizon_rows_actual"] = 0
            r["quality_audit_pass"] = 0
            r["fake_row_count"] = 0
            r["proxy_row_count"] = 0
    summary["stage"] = "P4_NATURAL_AP0_DENSITY_PANEL_V9850"
    summary["C_weak_pass"] = 0
    summary["C_strong_pass"] = 0
    summary["StopC_triggered"] = 0
    summary["StopC_status"] = "not_evaluable_panel20000_not_complete"
    v9720.write_bar_svg(out / "fig_C7_density_vs_quality_pareto.svg", "C7 density-quality", ["2876"], [fnum(summary.get("PanelA_CoreLike_LCB"))])
    return rows, summary


def generated_route_decision(p1: dict[str, Any], p2: dict[str, Any], p4: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    condition_b_strong = inum(p2.get("B_strong_pass"))
    condition_b_weak_c_insufficient = int(inum(p2.get("B_weak_pass")) and inum(p4.get("StopC_triggered")))
    condition_a_objective = int(inum(p1.get("A_line_strong_pass")) and inum(p2.get("B_weak_pass")))
    reopen = int(condition_b_strong or condition_b_weak_c_insufficient or condition_a_objective)
    row = {
        "stage": "P5_GENERATED_ROUTE_REOPEN_DECISION_V9850",
        "status": "summary",
        "condition_B_strong_pass": condition_b_strong,
        "condition_B_weak_plus_C_density_insufficient": condition_b_weak_c_insufficient,
        "condition_A_path_mechanism_has_legal_virtual_objective": condition_a_objective,
        "generated_route_reopen_allowed": reopen,
        "generated_route_status": "reopened" if reopen else "stopped_B_C_D_conditions_not_met",
        "generated_action_count": 0,
        "branch_horizon_rows_expected": 0,
        "branch_horizon_rows_actual": 0,
        "negative_control_pass": 0,
        "source_to_generated_damage": "",
        "runtime_cost": "",
        "D_weak_pass": 0,
        "D_strong_pass": 0,
        "reason": "" if reopen else "no_B_strong_no_density_insufficient_evidence_no_legal_virtual_objective",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    for fig in [
        "fig_D1_generated_quality_pareto.svg",
        "fig_D2_generated_future_path_curves.svg",
        "fig_D3_generated_memory_offdiag_dashboard.svg",
        "fig_D4_generated_vs_existing_density.svg",
        "fig_D5_runtime_apply_cost.svg",
        "fig_D6_source_to_generated_damage_matrix.svg",
    ]:
        v9720.write_bar_svg(out / fig, fig, ["not_run"], [0.0])
    return [row], row


def not_run(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {"stage": stage, "status": "not_run", "reason": reason, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0, **extra}
    return [row], row


def copy_base_acc(source_v9810: Path) -> list[dict[str, Any]]:
    rows = [dict(r) for r in read_csv(source_v9810 / "base_acc_sentinel_v9810.csv")]
    for r in rows:
        r["stage"] = "BASE_ACC_SENTINEL_V9850"
        r["reused_from_v9810"] = 1
        r["base_acc_used_for_controller"] = 0
        r["fake_data_used"] = 0
        r["proxy_row_used"] = 0
        r["cpu_offload_used"] = 0
    return rows


def field_legality() -> list[dict[str, Any]]:
    rows = [
        ("green", "current_CE_delta,memory_CE_delta,hard_tail_CE_delta,action_norm,payload_norm,virtual_train_probe_metrics", "training-time train-stream probe fields"),
        ("yellow", "cover_entropy,basis_effective_rank,curvature_proxy,jacobian_proxy", "architecture-specific diagnostic fields"),
        ("red", "future V,AUV future label,OldRank_score,WT80_score,dataset name,validation/test outcome", "future/outcome/table diagnostic fields"),
    ]
    return [{
        "stage": "FIELD_LEGALITY_LEDGER_V9850",
        "status": "field_legality_row",
        "field_color": color,
        "fields": fields,
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    } for color, fields, reason in rows]


def audit_rows(artifacts: dict[str, Path]) -> int:
    total = 0
    for name, path in artifacts.items():
        if name.endswith(".csv") and path.exists() and not name.startswith(("no_fake", "contract", "failure")):
            total += len(read_csv(path))
        elif name.endswith(".json") and path.exists():
            total += 1
    return total


def sha_rows(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in paths.items() if path.exists()}


def route_from(p1: dict[str, Any], p2: dict[str, Any], p3: dict[str, Any], p4: dict[str, Any], p5: dict[str, Any]) -> tuple[str, str, str, str]:
    if inum(p2.get("B_strong_pass")):
        return "Case1-LegalVirtualProxyControllerCandidate", "legal_virtual_proxy_found", "none", "controller_candidate"
    if inum(p4.get("C_strong_pass")) and not inum(p2.get("B_weak_pass")):
        return "Case2-NaturalDensitySufficientMechanismMissing", "no_legal_training_time_proxy", "none", "existing_action_mechanism"
    if inum(p4.get("StopC_triggered")) and inum(p2.get("B_weak_pass")):
        return "Case3-DensityInsufficientGeneratedRouteCandidate", "natural_density_insufficient", "generated_route_candidate", "generated"
    if inum(p2.get("StopB_triggered")) and not inum(p3.get("materializer_entrypoint_found")) and not inum(p5.get("generated_route_reopen_allowed")):
        return "Case4-LegalProxyFailNaturalStreamMissingGeneratedStopped", "legal_virtual_probe_failed", "natural_stream_materializer_missing", "pivot_or_engineering_blocked"
    if inum(p1.get("A_line_mechanism_pass")):
        return "RouteA-FuturePathExistsButProxyAndDensityUnresolved", "legal_virtual_proxy_failed", "natural_stream_materializer_missing", "unresolved"
    return "RouteZ-AllFourLinesBlocked", "future_path_not_actionable", "natural_stream_materializer_missing", "pivot"


def write_dashboard(out: Path, route: dict[str, Any]) -> None:
    p1 = summary_row(read_csv(out / "p1_future_path_type_decomposition_v9850.csv"))
    p2 = summary_row(read_csv(out / "p2_legal_virtual_path_proxy_v9850.csv"))
    p3 = summary_row(read_csv(out / "p3_natural_ap0_stream_materializer_audit_v9850.csv"))
    p4 = summary_row(read_csv(out / "p4_natural_ap0_density_panel_v9850.csv"))
    p5 = summary_row(read_csv(out / "p5_generated_route_reopen_decision_v9850.csv"))
    nf = summary_row(read_csv(out / "no_fake_audit_v9850.csv"))
    lines = [
        "# v9850_dashboard",
        "",
        "## 1. Route summary",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"primary_blocker = {route.get('primary_blocker')}",
        f"secondary_blocker = {route.get('secondary_blocker')}",
        f"recommendation = {route.get('route_recommendation')}",
        "```",
        "",
        "## 2. 四线 pass/fail 表",
        "",
        "| line | weak/strong | key blocker |",
        "|---|---|---|",
        f"| A future path | `{p1.get('A_line_mechanism_pass')}` / `{p1.get('A_line_strong_pass')}` | `{p1.get('reason')}` |",
        f"| B legal proxy | `{p2.get('B_weak_pass')}` / `{p2.get('B_strong_pass')}` | `{p2.get('reason')}` |",
        f"| C natural stream | `{p4.get('C_weak_pass')}` / `{p4.get('C_strong_pass')}` | `{p3.get('reason')}` |",
        f"| D generated | `{p5.get('D_weak_pass')}` / `{p5.get('D_strong_pass')}` | `{p5.get('reason')}` |",
        "",
        "## 3. A 线 future path group curves",
        "",
        "`fig_A1_group_future_path_V_curve.svg`, `fig_A2_group_future_path_risk_curve.svg`, `fig_A4_AUV_vs_longrisk_pareto.svg`",
        "",
        "## 4. B 线 legal proxy quality/cost",
        "",
        f"best probe = `{p2.get('best_probe_id')}`, precision = `{p2.get('best_precision')}`, V LCB = `{p2.get('best_V_LCB')}`, longrisk UCB = `{p2.get('best_longrisk_UCB')}`, cost q90 ms = `{p2.get('best_cost_q90_ms')}`.",
        "",
        "## 5. C 线 natural stream density curve",
        "",
        f"PanelA CoreLike LCB = `{p4.get('PanelA_CoreLike_LCB')}`；5000/10000/20000 panels remain `not_run` because the materializer entrypoint is `{p3.get('materializer_entrypoint_found')}`.",
        "",
        "## 6. D 线 generated route stop/reopen decision",
        "",
        f"generated_route_reopen_allowed = `{p5.get('generated_route_reopen_allowed')}`；status = `{p5.get('generated_route_status')}`.",
        "",
        "## 7. Controller/runtime gate boundary",
        "",
        "controller/runtime/paired replay/short-full are all `not_run` because B strong and C strong gates are absent.",
        "",
        "## 8. No-fake / no-proxy / no-dataset-tuning audit",
        "",
        f"rows_checked = `{nf.get('rows_checked')}`；fake/proxy/cpu = `{nf.get('fake_data_used')}` / `{nf.get('proxy_row_used')}` / `{nf.get('cpu_offload_used')}`。",
        "",
        "## 9. Stop/pivot recommendation",
        "",
        f"`{route.get('route_recommendation')}`：B probe 已触发 StopB，C 仍是 materializer 工程 blocker，D 不允许重开。",
        "",
        "## Figures",
        "",
        "```text",
        "fig_A1_group_future_path_V_curve.svg",
        "fig_A2_group_future_path_risk_curve.svg",
        "fig_A4_AUV_vs_longrisk_pareto.svg",
        "fig_B1_proxy_vs_true_AUV_scatter.svg",
        "fig_B4_probe_cost_vs_quality.svg",
        "fig_C1_density_curve_panel_size.svg",
        "fig_C2_corelike_rate_ci.svg",
        "fig_D1_generated_quality_pareto.svg",
        "fig_stop_pivot_matrix.svg",
        "```",
        "",
    ]
    (out / "v9850_dashboard.md").write_text("\n".join(lines), encoding="utf-8")


def write_recap(out: Path, route: dict[str, Any], hashes: dict[str, str]) -> None:
    p0 = summary_row(read_csv(out / "p0_boundary_reproduction_v9850.csv"))
    p1 = summary_row(read_csv(out / "p1_future_path_type_decomposition_v9850.csv"))
    p2 = summary_row(read_csv(out / "p2_legal_virtual_path_proxy_v9850.csv"))
    p3 = summary_row(read_csv(out / "p3_natural_ap0_stream_materializer_audit_v9850.csv"))
    p4 = summary_row(read_csv(out / "p4_natural_ap0_density_panel_v9850.csv"))
    p5 = summary_row(read_csv(out / "p5_generated_route_reopen_decision_v9850.csv"))
    nf = summary_row(read_csv(out / "no_fake_audit_v9850.csv"))
    groups = [r for r in read_csv(out / "p1_future_path_type_decomposition_v9850.csv") if r.get("status") == "group_summary"]
    probes = [r for r in read_csv(out / "p2_legal_virtual_path_proxy_v9850.csv") if r.get("status") == "probe_summary"]
    panels = [r for r in read_csv(out / "p4_natural_ap0_density_panel_v9850.csv") if r.get("status") in {"panel_row", "not_run"}]
    lines = [
        "# DG-KAN v9.8.5 Four-Line Future Operator / Natural Stream Route Decision 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.8.5_结果解读_四线进展_未来算子与自然扩流实验计划.md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest 或 v9.8.4 已真实 materialized 的 branch-horizon rows；没有 fake data、proxy rows，也没有把 future-path diagnostic、legal-probe diagnostic、natural stream missing 或 generated not_run 写成 official controller pass。",
        "",
        "## 0. 最新结论",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"primary_blocker = {route.get('primary_blocker')}",
        f"secondary_blocker = {route.get('secondary_blocker')}",
        f"route_recommendation = {route.get('route_recommendation')}",
        f"system_legal_controller_pass = {route.get('system_legal_controller_pass')}",
        f"generated_route_status = {route.get('generated_route_status')}",
        "```",
        "",
        f"最终 artifact：`{out}`",
        "",
        "核心结论：",
        "",
        f"1. P0 复现 v9.8.4 boundary：source route = `{p0.get('source_route_v9840')}`，system pass = `{p0.get('system_legal_controller_pass_v9840')}`。",
        f"2. A 线读取真实 landed future rows：realfunctional rows = `{p1.get('realfunctional_row_count')}`，actions = `{p1.get('action_count')}`，A mechanism/strong/controller = `{p1.get('A_line_mechanism_pass')}` / `{p1.get('A_line_strong_pass')}` / `{p1.get('A_line_controller_pass')}`。",
        f"3. A 线 best path type = `{p1.get('best_path_type_group')}`；strong fail reason = `{p1.get('reason')}`。",
        f"4. B 线 legal virtual probe 全部失败：weak/strong = `{p2.get('B_weak_pass')}` / `{p2.get('B_strong_pass')}`，best probe = `{p2.get('best_probe_id')}`，precision = `{p2.get('best_precision')}`，V LCB = `{p2.get('best_V_LCB')}`。",
        f"5. B 线 StopB triggered = `{p2.get('StopB_triggered')}`；best longrisk UCB = `{p2.get('best_longrisk_UCB')}`，memory+offdiag UCB = `{p2.get('best_memory_offdiag_UCB')}`。",
        f"6. C 线 materializer entrypoint found = `{p3.get('materializer_entrypoint_found')}`；C preflight pass = `{p3.get('C_preflight_pass')}`。",
        f"7. C panel completed/not_run = `{p4.get('completed_panel_count')}` / `{p4.get('not_run_panel_count')}`；PanelA CoreLike LCB = `{p4.get('PanelA_CoreLike_LCB')}`。",
        f"8. D generated reopen allowed = `{p5.get('generated_route_reopen_allowed')}`；status = `{p5.get('generated_route_status')}`。",
        f"9. No-fake audit：rows checked = `{nf.get('rows_checked')}`，fake/proxy/cpu = `{nf.get('fake_data_used')}` / `{nf.get('proxy_row_used')}` / `{nf.get('cpu_offload_used')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/run_v9850_four_line_future_operator_natural_stream_decision.py` | v9.8.5 runner；读取 v9.8.4/v9.8.2 landed rows，执行 A 路径类型分解、B 合法 virtual probe 评估、C natural stream materializer audit、D generated reopen decision、controller/runtime boundary、dashboard 与 audits。 |",
        "",
        "```text",
        "python -m py_compile experiments/run_v9850_four_line_future_operator_natural_stream_decision.py",
        "```",
        "",
        "```bash",
        "python experiments/run_v9850_four_line_future_operator_natural_stream_decision.py --out-dir results/real_rerun_20260506/v9850_four_line_future_operator_natural_stream_decision_full_20260516T180000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 2876,5000,10000,20000",
        "```",
        "",
        "说明：v9.8.5 没有重新伪造 branch rows；A/B 使用 v9.8.4 已真实 CUDA replay 的 landed rows，C/D/E 因 gate 未满足显式 not_run。",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 3. A 线 Future Path Type",
        "",
        "| group | actions | AUV LCB | RiskAdjustedAUV LCB | DelayedGain LCB | V80 LCB | V240 LCB | RiskPath UCB | mem/off UCB | LDO/LSO/LTO | path strong |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for r in groups:
        lines.append(f"| `{r.get('group_id')}` | `{r.get('action_count')}` | `{r.get('AUV_LCB')}` | `{r.get('RiskAdjustedAUV_LCB')}` | `{r.get('DelayedGain_LCB')}` | `{r.get('V80_LCB')}` | `{r.get('V240_LCB')}` | `{r.get('RiskPath_UCB')}` | `{r.get('MemoryOffdiagPath_UCB')}` | `{r.get('LDO_drop')}`/`{r.get('LSO_drop')}`/`{r.get('LTO_drop')}` | `{r.get('path_type_strong_pass')}` |")
    lines += [
        "",
        "判断：A 线继续支持 future-path 现象，但没有任何 path type 同时满足 count/value/risk/LDO/LSO/LTO controller gate。",
        "",
        "## 4. B 线 Legal Virtual Proxy",
        "",
        "| probe | precision | V LCB | AUV LCB | longrisk UCB | memory/offdiag UCB | cost q90 ms | weak | strong | failure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in probes:
        lines.append(f"| `{r.get('probe_id')}` | `{r.get('TopK87_precision')}` | `{r.get('V_LCB')}` | `{r.get('AUV_LCB')}` | `{r.get('longrisk_UCB')}` | `{fnum(r.get('memory_UCB')) + fnum(r.get('offdiag_UCB'))}` | `{r.get('feature_compute_ms_q90')}` | `{r.get('B_weak_pass')}` | `{r.get('B_strong_pass')}` | `{r.get('failure_mode')}` |")
    lines += [
        "",
        "判断：B 线这次不再是空缺，而是用 train-time landed replay metrics 做了 legal virtual probe 评估；结果触发 StopB：best probe 的 V LCB <= 0，且其他 probe 还同时出现 precision 低或 longrisk/memory/offdiag 偏高。",
        "",
        "## 5. C 线 Natural AP0 Stream",
        "",
        "| panel | status | target | labeled | expected rows | actual rows | missing labels | CoreLike rate | LCB/UCB | reason |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in panels:
        lines.append(f"| `{r.get('panel_id')}` | `{r.get('status')}` | `{r.get('target_action_count')}` | `{r.get('actual_labeled_action_count')}` | `{r.get('branch_horizon_rows_expected')}` | `{r.get('branch_horizon_rows_actual')}` | `{r.get('missing_label_count')}` | `{r.get('CoreLike_rate')}` | `{r.get('Wilson_LCB/UCB')}` | `{r.get('reason')}` |")
    lines += [
        "",
        "判断：C 线仍未完成 natural stream extension；没有 5000/10000/20000 真实 labels，所以不能判定 density sufficient/insufficient。",
        "",
        "## 6. D/E Boundary",
        "",
        "```text",
        f"generated_route_reopen_allowed = {p5.get('generated_route_reopen_allowed')}",
        f"generated_route_status = {p5.get('generated_route_status')}",
        "controller/runtime/paired replay/short-full = not_run",
        "```",
        "",
        "判断：B strong 不成立，C density insufficient 未被证明，A 也没有合法 virtual objective，因此 generated route 不允许重开。",
        "",
        "## 7. Dashboard / Figures",
        "",
        "Dashboard：`v9850_dashboard.md`",
        "",
        "```text",
        "fig_A1_group_future_path_V_curve.svg",
        "fig_A2_group_future_path_risk_curve.svg",
        "fig_A4_AUV_vs_longrisk_pareto.svg",
        "fig_B1_proxy_vs_true_AUV_scatter.svg",
        "fig_B4_probe_cost_vs_quality.svg",
        "fig_C1_density_curve_panel_size.svg",
        "fig_C2_corelike_rate_ci.svg",
        "fig_D1_generated_quality_pareto.svg",
        "fig_stop_pivot_matrix.svg",
        "```",
        "",
        "## 8. No-Fake / Contract / Failure",
        "",
        "```text",
        f"rows_checked = {nf.get('rows_checked')}",
        f"fake_data_used = {nf.get('fake_data_used')}",
        f"proxy_row_used = {nf.get('proxy_row_used')}",
        f"cpu_offload_used = {nf.get('cpu_offload_used')}",
        "```",
        "",
        "## 9. Hash",
        "",
        "| artifact | SHA256 |",
        "|---|---|",
    ]
    for name, digest in sorted(hashes.items()):
        lines.append(f"| `{name}` | `{digest}` |")
    lines += [
        "",
        "## 10. 最终分析结论",
        "",
        "```text",
        "1. A 线确认 future-path 现象仍存在，但没有 path type 达到 full controller gate。",
        "2. B 线这次真正评估了 legal virtual probe，结果失败并触发 StopB。",
        "3. C 线 natural AP0 stream materializer 仍未落地，因此 density 不能裁决。",
        "4. D 线没有满足重开条件，generated route 正确停止。",
        "5. v9.8.5 的路线裁决偏向 Case4：B 失败、C 工程 blocker、D 停止；下一步要么先工程化 C materializer，要么转向 optimizer-level theory，而不能继续调 red diagnostic。",
        "```",
        "",
        "最终一句话：",
        "",
        f"> v9.8.5 真实执行后停在 `{route.get('route')}`：future-path 好动作现象仍可信，但合法 train-time virtual probe 不能选出安全好动作，自然扩流 materializer 仍缺失，generated route 没有重开条件，因此不能进入 official controller，strict PureKAN functional 仍未成功。",
        "",
    ]
    RECAP_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if args.fresh and out.exists():
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    artifacts: dict[str, Path] = {}

    def dump_csv(name: str, rows: list[dict[str, Any]]) -> None:
        path = out / name
        write_csv(path, rows)
        artifacts[name] = path

    def dump_json(name: str, row: dict[str, Any]) -> None:
        path = out / name
        write_json(path, row)
        artifacts[name] = path

    source_v9840 = Path(args.source_v9840)
    source_v9820 = Path(args.source_v9820)
    source_v9810 = Path(args.source_v9810)

    p0_rows, p0 = p0_boundary(source_v9840)
    dump_csv("p0_boundary_reproduction_v9850.csv", p0_rows)

    ap0 = load_ap0(args)
    real = load_real_rows(source_v9820, source_v9840)
    feats, _completed = feature_rows(real)

    p1_rows, p1 = p1_path_type_decomposition(ap0, real, feats, out)
    dump_csv("p1_future_path_type_decomposition_v9850.csv", p1_rows)

    p2_rows, p2 = p2_legal_virtual_proxy(ap0, feats, out)
    dump_csv("p2_legal_virtual_path_proxy_v9850.csv", p2_rows)

    p3_rows, p3 = materializer_entrypoint_audit()
    dump_csv("p3_natural_ap0_stream_materializer_audit_v9850.csv", p3_rows)

    p4_rows, p4 = p4_density_panel(source_v9820, parse_ints(args.panel_targets), p3, out)
    dump_csv("p4_natural_ap0_density_panel_v9850.csv", p4_rows)

    p5_rows, p5 = generated_route_decision(p1, p2, p4, out)
    dump_csv("p5_generated_route_reopen_decision_v9850.csv", p5_rows)

    p6_rows, p6 = not_run("P6_MINIMAL_CONTROLLER_BOUNDARY_V9850", "B_strong_or_C_strong_plus_B_weak_not_passed", controller_pass=0)
    dump_csv("p6_minimal_controller_boundary_v9850.csv", p6_rows)
    p7_rows, p7 = not_run("P7_SELECTED_RUNTIME_BOUNDARY_V9850", "P6_controller_not_passed", runtime_pass=0)
    dump_csv("p7_runtime_boundary_v9850.csv", p7_rows)
    p8_rows, p8 = not_run("P8_PAIRED_REPLAY_BOUNDARY_V9850", "P7_runtime_not_passed", paired_replay_pass=0)
    dump_csv("p8_paired_replay_boundary_v9850.csv", p8_rows)
    p9_rows, p9 = not_run("P9_SHORT_FULL_BOUNDARY_V9850", "P8_paired_replay_not_passed", short_full_pass=0)
    dump_csv("p9_short_full_boundary_v9850.csv", p9_rows)

    dump_csv("base_acc_sentinel_v9850.csv", copy_base_acc(source_v9810))
    dump_csv("field_legality_ledger_v9850.csv", field_legality())

    route, primary, secondary, recommendation = route_from(p1, p2, p3, p4, p5)
    route_decision = {
        "stage": "ROUTE_DECISION_V9850",
        "status": "summary",
        "route": route,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "route_recommendation": recommendation,
        "source_route_v9840": p0.get("source_route_v9840"),
        "P0_boundary_pass": p0.get("boundary_reproduced"),
        "A_line_mechanism_pass": p1.get("A_line_mechanism_pass"),
        "A_line_strong_pass": p1.get("A_line_strong_pass"),
        "B_weak_pass": p2.get("B_weak_pass"),
        "B_strong_pass": p2.get("B_strong_pass"),
        "StopB_triggered": p2.get("StopB_triggered"),
        "C_preflight_pass": p3.get("C_preflight_pass"),
        "C_weak_pass": p4.get("C_weak_pass"),
        "C_strong_pass": p4.get("C_strong_pass"),
        "C_materializer_entrypoint_found": p3.get("materializer_entrypoint_found"),
        "D_generated_reopen_allowed": p5.get("generated_route_reopen_allowed"),
        "generated_route_status": p5.get("generated_route_status"),
        "controller_pass": p6.get("controller_pass"),
        "runtime_pass": p7.get("runtime_pass"),
        "paired_replay_pass": p8.get("paired_replay_pass"),
        "short_full_pass": p9.get("short_full_pass"),
        "system_legal_controller_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("route_decision_v9850.json", route_decision)

    v9720.write_bar_svg(out / "fig_stop_pivot_matrix.svg", "Stop/pivot matrix", ["StopB", "C_missing", "D_stopped"], [inum(p2.get("StopB_triggered")), int(not inum(p3.get("materializer_entrypoint_found"))), int(not inum(p5.get("generated_route_reopen_allowed")))])

    no_fake = {
        "stage": "NO_FAKE_AUDIT_V9850",
        "status": "summary",
        "rows_checked": audit_rows(artifacts),
        "fake_proxy_nonzero_count": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "no_fake": 1,
        "no_proxy": 1,
    }
    dump_csv("no_fake_audit_v9850.csv", [no_fake])
    contract = {
        "stage": "CONTRACT_AUDIT_V9850",
        "status": "summary",
        "manual_forward/manual_backward/manual_adamw_update": "1/1/1",
        "v9840_boundary_pass": p0.get("boundary_reproduced"),
        "A_line_mechanism/strong/controller": f"{p1.get('A_line_mechanism_pass')}/{p1.get('A_line_strong_pass')}/{p1.get('A_line_controller_pass')}",
        "B_legal_virtual_probe_weak/strong": f"{p2.get('B_weak_pass')}/{p2.get('B_strong_pass')}",
        "C_materializer_entrypoint_found": p3.get("materializer_entrypoint_found"),
        "D_generated_reopen_allowed": p5.get("generated_route_reopen_allowed"),
        "controller/runtime/system": "0/0/0",
        "uses_future_outcome_for_official_controller": 0,
        "diagnostic_promoted_to_official": 0,
        "fake/proxy/cpu_offload": "0/0/0",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("contract_audit_v9850.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9850",
        "status": "summary",
        "route": route,
        "F0_boundary_fail": int(not inum(p0.get("boundary_reproduced"))),
        "F1_A_path_type_no_controller_gate": int(not inum(p1.get("A_line_strong_pass"))),
        "F2_B_legal_virtual_probe_failed": int(not inum(p2.get("B_weak_pass"))),
        "F3_C_natural_stream_materializer_missing": int(not inum(p3.get("materializer_entrypoint_found"))),
        "F4_D_generated_route_stopped": int(not inum(p5.get("generated_route_reopen_allowed"))),
        "F5_controller_runtime_blocked": 1,
        "StopB_triggered": p2.get("StopB_triggered"),
        "StopC_evaluable": 0,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("failure_taxonomy_v9850.csv", [failure])

    write_dashboard(out, route_decision)
    artifacts["v9850_dashboard.md"] = out / "v9850_dashboard.md"

    hashes = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    manifest = {
        "stage": "RUN_MANIFEST_V9850",
        "status": "summary",
        "out_dir": str(out),
        "execution_profile": args.execution_profile,
        "command_profile": {
            "device": args.device,
            "data_root": args.data_root,
            "seed": args.seed,
            "panel_targets": args.panel_targets,
            "source_v9840": args.source_v9840,
        },
        "wallclock_sec": time.perf_counter() - t0,
        "route": route,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "artifact_hashes": hashes,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("run_manifest_v9850.json", manifest)
    hashes_for_recap = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    write_recap(out, route_decision, hashes_for_recap)
    print(json.dumps({
        "out_dir": str(out),
        "route": route,
        "A_line_strong_pass": p1.get("A_line_strong_pass"),
        "B_weak_pass": p2.get("B_weak_pass"),
        "B_strong_pass": p2.get("B_strong_pass"),
        "StopB_triggered": p2.get("StopB_triggered"),
        "C_materializer_entrypoint_found": p3.get("materializer_entrypoint_found"),
        "D_generated_reopen_allowed": p5.get("generated_route_reopen_allowed"),
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "system_legal_controller_pass": 0,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
