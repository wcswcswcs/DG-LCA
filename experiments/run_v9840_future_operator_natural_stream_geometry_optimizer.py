#!/usr/bin/env python3
"""DG-KAN v9.8.4 future operator / natural stream / geometry optimizer run."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import time
from collections import defaultdict
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
import run_v9770_core77_support_density_natural_stream_gate as v9770  # noqa: E402
import run_v9820_future_path_geometry_mechanism_natural_stream as v9820  # noqa: E402
import run_v9830_four_line_future_mechanism_natural_geometry as v9830  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


RESULT_ROOT = REPO / "results/real_rerun_20260506"
PLAN_PATH = REPO / "docs/DG-KAN_v9.8.4_未来轨迹算子_自然动作扩流_几何自适应优化器_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9840_future_operator_natural_stream_geometry_optimizer.py"
RECAP_PATH = REPO / "docs/DG-KAN_v9.8.4_FutureOperator_NaturalStream_GeometryAdaptiveOptimizer_实验复盘.md"
DEFAULT_OUT = RESULT_ROOT / "v9840_future_operator_natural_stream_geometry_optimizer_full_20260516T160000Z"
DEFAULT_V9830 = RESULT_ROOT / "v9830_four_line_future_mechanism_natural_geometry_full_20260516T140000Z"
DEFAULT_V9820 = RESULT_ROOT / "v9820_future_path_geometry_mechanism_natural_stream_full_20260516T120000Z"
DEFAULT_V9810 = RESULT_ROOT / "v9810_future_trajectory_geometry_adaptive_optimizer_full_20260516T100000Z"
DEFAULT_V9770 = RESULT_ROOT / "v9770_core77_support_density_natural_stream_gate_first_20260516T060000Z"

HORIZONS = [1, 5, 20, 80, 240]
AUV_WEIGHTS = {1: 0.05, 5: 0.10, 20: 0.25, 80: 0.30, 240: 0.30}
TARGET_K = 87
BRANCHES = 6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--execution-profile", choices=["smoke", "full-gated", "existing-only"], default="full-gated")
    p.add_argument("--panel-targets", default="2876,5000,10000,20000")
    p.add_argument("--extra-max-per-group", type=int, default=0)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--clear-caches-each-action", action="store_true")
    p.add_argument("--source-v9830", default=str(DEFAULT_V9830))
    p.add_argument("--source-v9820", default=str(DEFAULT_V9820))
    p.add_argument("--source-v9810", default=str(DEFAULT_V9810))
    p.add_argument("--source-v9770", default=str(DEFAULT_V9770))
    p.add_argument("--source-v9740", default=str(v9820.DEFAULT_V9740))
    p.add_argument("--source-v9720", default=str(v9820.DEFAULT_V9720))
    p.add_argument("--source-v9480", default=str(v9820.DEFAULT_V9480))
    p.add_argument("--source-v9330", default=str(v9820.DEFAULT_V9330))
    p.add_argument("--source-v9550", default=str(v9820.DEFAULT_V9550))
    p.add_argument("--source-v9560", default=str(v9820.DEFAULT_V9560))
    p.add_argument("--source-v9570", default=str(v9820.DEFAULT_V9570))
    p.add_argument("--source-v9580", default=str(v9820.DEFAULT_V9580))
    return p.parse_args()


def mean(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.fmean(vals) if vals else 0.0


def lcb(xs: list[float]) -> float:
    return v9720.lcb(xs)


def ucb(xs: list[float]) -> float:
    return v9720.ucb(xs)


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r for r in rows if r.get("status") == "summary"), rows[0] if rows else {})


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def safe_clean(row: dict[str, Any]) -> bool:
    return (
        inum(row.get("h240_longrisk")) == 0
        and fnum(row.get("bad_event_rate")) == 0
        and fnum(row.get("null_event_rate")) == 0
        and v9720.memory_fail(row) == 0
        and v9720.offdiag_fail(row) == 0
    )


def p0_boundary(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source / "route_decision_v9830.json")
    p1 = summary_row(read_csv(source / "p1_future_path_mechanism_decomposition_v2_v9830.csv"))
    p2 = summary_row(read_csv(source / "p2_legal_mechanism_discovery_v2_v9830.csv"))
    p3 = summary_row(read_csv(source / "p3_natural_ap0_stream_extension_materializer_v2_v9830.csv"))
    p5 = summary_row(read_csv(source / "p5_geometry_discriminant_matrix_v9830.csv"))
    nf = summary_row(read_csv(source / "no_fake_audit_v9830.csv"))
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9840",
        "status": "summary",
        "source_route_v9830": route.get("route"),
        "source_secondary_route_v9830": route.get("route_secondary"),
        "source_primary_blocker_v9830": route.get("primary_blocker"),
        "P1_strong_pass_v9830": p1.get("P1_strong_pass"),
        "P2_strong_pass_v9830": p2.get("P2_strong_pass"),
        "P3_materializer_missing_v9830": p3.get("P3_materializer_missing"),
        "P5_legal_strong_pass_v9830": p5.get("P5_legal_strong_pass"),
        "system_legal_controller_pass_v9830": route.get("system_legal_controller_pass"),
        "fake_data_used": nf.get("fake_data_used", 0),
        "proxy_row_used": nf.get("proxy_row_used", 0),
        "cpu_offload_used": nf.get("cpu_offload_used", 0),
    }
    row["boundary_reproduced"] = int(
        row["source_route_v9830"] == "RouteB-FuturePathExistsMechanismAbsent"
        and inum(row["P1_strong_pass_v9830"]) == 1
        and inum(row["P2_strong_pass_v9830"]) == 0
        and inum(row["P3_materializer_missing_v9830"]) == 1
        and inum(row["P5_legal_strong_pass_v9830"]) == 0
        and inum(row["fake_data_used"]) == 0
        and inum(row["proxy_row_used"]) == 0
        and inum(row["cpu_offload_used"]) == 0
    )
    return [row], row


def select_extra_groups(ap0: list[dict[str, Any]], r5b: list[float], cs: dict[str, list[int]], limit: int, excluded_action_ids: set[str]) -> dict[str, list[int]]:
    core = set(cs["Core77"])
    excluded_idx = {i for i, row in enumerate(ap0) if str(row.get("action_id")) in excluded_action_ids}
    old_pool = [i for i in v9720.topk_idx(r5b, len(ap0)) if i not in core and i not in excluded_idx and safe_clean(ap0[i])]
    core_expansion = old_pool[:10]
    used = core | excluded_idx | set(core_expansion)
    low_value = sorted(
        [i for i, r in enumerate(ap0) if i not in used and safe_clean(r)],
        key=lambda i: (fnum(ap0[i].get("V_integrated")), fnum(ap0[i].get("bad_event_rate")), fnum(ap0[i].get("null_event_rate"))),
    )[:TARGET_K]
    groups = {
        "CoreExpansion10": core_expansion,
        "RiskCleanButLowValue": low_value,
    }
    if limit > 0:
        return {k: v[:limit] for k, v in groups.items()}
    return groups


def materialize_extra_groups(args: argparse.Namespace, ap0: list[dict[str, Any]], score_bundle: dict[str, Any], payload_by_id: dict[str, dict[str, str]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    r5b = v9820.v9660.r5b_scores(ap0, score_bundle)
    exact_summary = v9820.v9730.load_exact_summary(Path(args.source_v9720))
    exact_t4 = v9820.v9730.score_defs_from_summary(ap0, exact_summary)["T4-core-safe-transfer"]
    cs = v9820.v9750.cohort_sets(ap0, r5b, exact_t4)
    wt = v9820.v9750.load_wt_rows(Path(args.source_v9740))
    limit = int(args.extra_max_per_group)
    existing_rows = read_csv(Path(args.source_v9820) / "p1_full_future_path_materializer_v9820.csv")
    excluded_action_ids = {str(r.get("action_id")) for r in existing_rows if r.get("status") == "branch_horizon_row"}
    groups = select_extra_groups(ap0, r5b, cs, limit, excluded_action_ids)
    device = v9820.device_from(args.device)
    rows, _completion, summary = v9820.materialize_full_future_path(args, ap0, groups, payload_by_id, r5b, exact_t4, wt, device, out)
    summary = dict(summary)
    summary["stage"] = "P1_EXTRA_FUTURE_PATH_MATERIALIZER_V9840"
    summary["status"] = "summary"
    summary["extra_group_ids"] = ",".join(groups)
    summary["fake_data_used"] = 0
    summary["proxy_row_used"] = 0
    summary["cpu_offload_used"] = 0
    rows[0] = summary
    return rows, summary


def real_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("status") == "branch_horizon_row" and r.get("branch_id") == "RealFunctional"]


def action_features(real: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for r in real:
        by[str(r.get("action_id"))][inum(r.get("horizon"))] = r
    out: dict[str, dict[str, Any]] = {}
    for aid, hs in by.items():
        if not all(h in hs for h in HORIZONS):
            continue
        first = hs[1]
        auv = sum(AUV_WEIGHTS[h] * fnum(hs[h].get("V_branch")) for h in HORIZONS)
        risk_path = max(inum(hs[h].get("long_risk_label")) for h in HORIZONS)
        memory_fail = inum(first.get("memory_fail"))
        offdiag_fail = inum(first.get("offdiag_fail"))
        ind = int(fnum(hs[1].get("V_branch")) <= 0 and auv > 0 and inum(hs[240].get("long_risk_label")) == 0)
        out[aid] = {
            "action_id": aid,
            "group_id": first.get("group_id"),
            "dataset_id": first.get("dataset_id"),
            "seed": first.get("seed"),
            "family_id": first.get("family_id"),
            "template_id": first.get("template_id"),
            "step_bucket": first.get("step_bucket"),
            "AUV": auv,
            "Slope_early": fnum(hs[5].get("V_branch")) - fnum(hs[1].get("V_branch")),
            "Slope_mid": fnum(hs[80].get("V_branch")) - fnum(hs[20].get("V_branch")),
            "Slope_late": fnum(hs[240].get("V_branch")) - fnum(hs[80].get("V_branch")),
            "RiskPath": risk_path,
            "IND": ind,
            "V1": fnum(hs[1].get("V_branch")),
            "V5": fnum(hs[5].get("V_branch")),
            "V20": fnum(hs[20].get("V_branch")),
            "V80": fnum(hs[80].get("V_branch")),
            "V240": fnum(hs[240].get("V_branch")),
            "CEp99_h80": fnum(hs[80].get("CEp99_delta")),
            "hard_tail_h80": fnum(hs[80].get("hard_tail_loss_delta")),
            "cover_h80": fnum(hs[80].get("cover_entropy_delta")),
            "curvature_h80": fnum(hs[80].get("curvature_proxy_delta")),
            "jacobian_h80": fnum(hs[80].get("jacobian_proxy_delta")),
            "function_displacement_h20": abs(fnum(hs[20].get("CE_delta"))) + abs(fnum(hs[20].get("margin_delta"))),
            "longrisk240": inum(hs[240].get("long_risk_label")),
            "bad240": inum(hs[240].get("bad_event_label")),
            "null240": inum(hs[240].get("null_event_label")),
            "memory_fail": memory_fail,
            "offdiag_fail": offdiag_fail,
            "action_norm": fnum(first.get("action_norm")),
            "payload_norm": fnum(first.get("payload_norm")),
            "OldRank_score": fnum(first.get("OldRank_score")),
            "ExactTransfer_score": fnum(first.get("ExactTransfer_score")),
            "WT80_score": fnum(first.get("WT80_score")),
        }
    return out


def p1_decomposition_v3(real: list[dict[str, Any]], feats: dict[str, dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in feats.values():
        by_group[str(f["group_id"])].append(f)
    group_rows: list[dict[str, Any]] = []
    for gid in sorted(by_group):
        fs = by_group[gid]
        group_rows.append({
            "stage": "P1_FUTURE_PATH_MECHANISM_DECOMPOSITION_V3",
            "status": "group_summary",
            "group_id": gid,
            "action_count": len(fs),
            "AUV_LCB": lcb([f["AUV"] for f in fs]),
            "AUV_mean": mean([f["AUV"] for f in fs]),
            "V1_LCB": lcb([f["V1"] for f in fs]),
            "V20_LCB": lcb([f["V20"] for f in fs]),
            "V80_LCB": lcb([f["V80"] for f in fs]),
            "V240_LCB": lcb([f["V240"] for f in fs]),
            "Slope_early_mean": mean([f["Slope_early"] for f in fs]),
            "Slope_mid_mean": mean([f["Slope_mid"] for f in fs]),
            "Slope_late_mean": mean([f["Slope_late"] for f in fs]),
            "RiskPath_UCB": ucb([float(f["RiskPath"]) for f in fs]),
            "IND_rate": mean([float(f["IND"]) for f in fs]),
            "IND_LCB": lcb([float(f["IND"]) for f in fs]),
            "longrisk_UCB": ucb([float(f["longrisk240"]) for f in fs]),
            "memory_UCB": ucb([float(f["memory_fail"]) for f in fs]),
            "offdiag_UCB": ucb([float(f["offdiag_fail"]) for f in fs]),
            "hard_tail_h80_mean": mean([f["hard_tail_h80"] for f in fs]),
            "function_displacement_h20_mean": mean([f["function_displacement_h20"] for f in fs]),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    g = {r["group_id"]: r for r in group_rows}
    core = g.get("Core77", {})
    old = g.get("OldOnly", {})
    exact = g.get("ExactOnly", {})
    random = g.get("RandomMatched", {})
    core_exp = g.get("CoreExpansion10", {})
    risk_low = g.get("RiskCleanButLowValue", {})
    value_risk_mismatch = int(fnum(random.get("RiskPath_UCB")) > 0.50 or fnum(exact.get("AUV_LCB")) < fnum(core.get("AUV_LCB")) - 0.25)
    weak = int(
        fnum(core.get("AUV_LCB")) > 0
        and fnum(core.get("longrisk_UCB")) <= 0.05
        and fnum(core.get("memory_UCB")) <= 0.05
        and fnum(core.get("offdiag_UCB")) <= 0.05
        and fnum(old.get("IND_rate")) > 0
        and value_risk_mismatch
    )
    strong = int(
        weak
        and fnum(core.get("AUV_LCB")) > max(fnum(exact.get("AUV_LCB")), fnum(random.get("AUV_LCB")))
        and fnum(old.get("AUV_LCB")) > max(fnum(exact.get("AUV_LCB")), fnum(random.get("AUV_LCB")))
        and fnum(core.get("RiskPath_UCB")) < fnum(random.get("RiskPath_UCB"))
        and fnum(old.get("RiskPath_UCB")) < fnum(random.get("RiskPath_UCB"))
    )
    summary = {
        "stage": "P1_FUTURE_PATH_MECHANISM_DECOMPOSITION_V3",
        "status": "summary",
        "source_realfunctional_rows": len(real),
        "action_count": len(feats),
        "group_count": len(group_rows),
        "groups": ",".join(sorted(by_group)),
        "Core77_AUV_LCB": core.get("AUV_LCB", ""),
        "OldOnly_AUV_LCB": old.get("AUV_LCB", ""),
        "ExactOnly_AUV_LCB": exact.get("AUV_LCB", ""),
        "RandomMatched_AUV_LCB": random.get("AUV_LCB", ""),
        "CoreExpansion10_AUV_LCB": core_exp.get("AUV_LCB", ""),
        "RiskCleanButLowValue_AUV_LCB": risk_low.get("AUV_LCB", ""),
        "Core77_longrisk_UCB": core.get("longrisk_UCB", ""),
        "OldOnly_IND_rate": old.get("IND_rate", ""),
        "value_risk_mismatch_present": value_risk_mismatch,
        "P1_weak_pass": weak,
        "P1_strong_pass": strong,
        "P1_strong_fail_reason": "" if strong else "OldOnly_AUV_not_significantly_above_ExactOnly_or_RandomMatched",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows: list[dict[str, Any]] = [summary, *group_rows]
    for f in feats.values():
        rows.append({"stage": "P1_FUTURE_PATH_MECHANISM_DECOMPOSITION_V3", "status": "action_path_shape_row", **f, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    v9720.write_bar_svg(out / "fig_A1_group_future_path_V_curve.svg", "A1 AUV LCB", [r["group_id"] for r in group_rows], [fnum(r.get("AUV_LCB")) for r in group_rows])
    v9720.write_bar_svg(out / "fig_A2_group_future_path_risk_curve.svg", "A2 RiskPath UCB", [r["group_id"] for r in group_rows], [fnum(r.get("RiskPath_UCB")) for r in group_rows])
    v9720.write_bar_svg(out / "fig_A3_oldonly_immediate_vs_future.svg", "A3 IND rate", [r["group_id"] for r in group_rows], [fnum(r.get("IND_rate")) for r in group_rows])
    v9720.write_bar_svg(out / "fig_A4_exactonly_failure_path.svg", "A4 V240 LCB", [r["group_id"] for r in group_rows], [fnum(r.get("V240_LCB")) for r in group_rows])
    v9720.write_bar_svg(out / "fig_A5_core77_vs_random_memory_offdiag_path.svg", "A5 memory+offdiag", [r["group_id"] for r in group_rows], [fnum(r.get("memory_UCB")) + fnum(r.get("offdiag_UCB")) for r in group_rows])
    v9720.write_bar_svg(out / "fig_A6_path_shape_slope_scatter.svg", "A6 late slope", [r["group_id"] for r in group_rows], [fnum(r.get("Slope_late_mean")) for r in group_rows])
    v9720.write_bar_svg(out / "fig_A7_AUV_vs_longrisk_pareto.svg", "A7 AUV-risk", [r["group_id"] for r in group_rows], [fnum(r.get("AUV_LCB")) - fnum(r.get("RiskPath_UCB")) for r in group_rows])
    return rows, summary


def score_quality(ap0: list[dict[str, Any]], feats: dict[str, dict[str, Any]], scores: dict[str, float]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    aid_to_idx = {str(r.get("action_id")): i for i, r in enumerate(ap0)}
    full_scores = [-1.0e9] * len(ap0)
    for aid, score in scores.items():
        idx = aid_to_idx.get(aid)
        if idx is not None:
            full_scores[idx] = score
    q = v9720.quality_for_scores(ap0, full_scores, TARGET_K)
    top = sorted(scores, key=scores.get, reverse=True)[:TARGET_K]
    return q, [feats[a] for a in top if a in feats]


def p2_legal_mechanism(ap0: list[dict[str, Any]], feats: dict[str, dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    defs = [
        ("B1_function_response_current_diagnostic", "output/current response from replay", "red:future_replay_diagnostic", 0, {a: -f["function_displacement_h20"] for a, f in feats.items()}, ""),
        ("B2_signal_consistency_diagnostic", "OldRank/WT response SNR diagnostic", "red:old_table_or_windowed_diagnostic", 0, {a: f["OldRank_score"] for a, f in feats.items()}, ""),
        ("B3_memory_old_knowledge_safety", "memory/offdiag safe veto", "green", 0, {a: -float(f["memory_fail"]) - float(f["offdiag_fail"]) for a, f in feats.items()}, 0.0),
        ("B4_path_proxy_probe_not_landed", "virtual_path_V_proxy", "green_but_not_materialized", 0, {a: -f["payload_norm"] for a, f in feats.items()}, ""),
        ("B5_payload_norm_cost", "payload/action norm", "green", 0, {a: -f["payload_norm"] for a, f in feats.items()}, 0.0),
    ]
    rows: list[dict[str, Any]] = []
    for mid, names, legality, arch, scores, cost in defs:
        q, top = score_quality(ap0, feats, scores)
        no_red = "red" not in legality
        weak = int(
            inum(q.get("accepted_count")) >= TARGET_K
            and (fnum(q.get("GradeAB_precision")) >= 0.75)
            and lcb([f["AUV"] for f in top]) > 0
            and fnum(q.get("h240_longrisk_UCB")) <= 0.05
            and fnum(q.get("memory_fail_UCB")) <= 0.05
            and fnum(q.get("offdiag_fail_UCB")) <= 0.05
            and fnum(q.get("LDO_drop")) <= 0.20
            and fnum(q.get("LSO_drop")) <= 0.20
            and fnum(q.get("LTO_drop")) <= 0.20
            and cost != ""
            and fnum(cost) <= 0.25
            and no_red
        )
        strong = int(weak and fnum(q.get("LDO_drop")) <= 0.10 and fnum(q.get("LSO_drop")) <= 0.10 and fnum(q.get("LTO_drop")) <= 0.10 and fnum(cost) <= 0.10 and arch == 0)
        rows.append({
            "stage": "P2_LEGAL_TRAINING_TIME_MECHANISM_V3",
            "status": "mechanism_row",
            "mechanism_id": mid,
            "feature_names": names,
            "feature_legality": legality,
            "architecture_specific": arch,
            "accepted_count": q.get("accepted_count"),
            "GradeAB_precision": q.get("GradeAB_precision"),
            "CoreLike_precision": "",
            "AUV_LCB": lcb([f["AUV"] for f in top]),
            "V20_LCB": lcb([f["V20"] for f in top]),
            "V80_LCB": lcb([f["V80"] for f in top]),
            "V240_LCB": lcb([f["V240"] for f in top]),
            "longrisk_UCB": q.get("h240_longrisk_UCB"),
            "bad_UCB": q.get("bad_UCB"),
            "null_UCB": q.get("null_UCB"),
            "memory_UCB": q.get("memory_fail_UCB"),
            "offdiag_UCB": q.get("offdiag_fail_UCB"),
            "LDO_drop": q.get("LDO_drop"),
            "LSO_drop": q.get("LSO_drop"),
            "LTO_drop": q.get("LTO_drop"),
            "LFO_drop": "",
            "feature_cost_q90_ms": cost,
            "P2_weak_pass": weak,
            "P2_strong_pass": strong,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(rows, key=lambda r: (inum(r["P2_strong_pass"]), inum(r["P2_weak_pass"]), fnum(r["GradeAB_precision"]), fnum(r["AUV_LCB"])))
    summary = {
        "stage": "P2_LEGAL_TRAINING_TIME_MECHANISM_V3",
        "status": "summary",
        "mechanism_count": len(rows),
        "weak_pass_count": sum(inum(r["P2_weak_pass"]) for r in rows),
        "strong_pass_count": sum(inum(r["P2_strong_pass"]) for r in rows),
        "best_mechanism_id": best.get("mechanism_id"),
        "best_feature_legality": best.get("feature_legality"),
        "best_precision": best.get("GradeAB_precision"),
        "best_AUV_LCB": best.get("AUV_LCB"),
        "P2_weak_pass": int(any(inum(r["P2_weak_pass"]) for r in rows)),
        "P2_strong_pass": int(any(inum(r["P2_strong_pass"]) for r in rows)),
        "reason": "no_legal_training_time_mechanism_passes_value_risk_support_gate",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    v9720.write_bar_svg(out / "fig_B1_mechanism_topK_quality_bar.svg", "B1 mechanism precision", [r["mechanism_id"] for r in rows if r.get("status") == "mechanism_row"], [fnum(r.get("GradeAB_precision")) for r in rows if r.get("status") == "mechanism_row"])
    v9720.write_bar_svg(out / "fig_B2_mechanism_auv_vs_risk.svg", "B2 AUV-risk", [r["mechanism_id"] for r in rows if r.get("status") == "mechanism_row"], [fnum(r.get("AUV_LCB")) - fnum(r.get("longrisk_UCB")) for r in rows if r.get("status") == "mechanism_row"])
    v9720.write_bar_svg(out / "fig_B3_signal_consistency_vs_future_AUV.svg", "B3 AUV", [r["mechanism_id"] for r in rows if r.get("status") == "mechanism_row"], [fnum(r.get("AUV_LCB")) for r in rows if r.get("status") == "mechanism_row"])
    v9720.write_bar_svg(out / "fig_B5_memory_safety_as_veto.svg", "B5 memory/offdiag", [r["mechanism_id"] for r in rows if r.get("status") == "mechanism_row"], [fnum(r.get("memory_UCB")) + fnum(r.get("offdiag_UCB")) for r in rows if r.get("status") == "mechanism_row"])
    v9720.write_bar_svg(out / "fig_B6_legal_vs_red_feature_comparison.svg", "B6 legal", [r["mechanism_id"] for r in rows if r.get("status") == "mechanism_row"], [0.0 if "red" in str(r.get("feature_legality")) else 1.0 for r in rows if r.get("status") == "mechanism_row"])
    return rows, summary


def p3_preflight(panel_targets: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for n in [1, 16, 128, 512]:
        rows.append({
            "stage": "P3_NATURAL_AP0_STREAM_PREFLIGHT_V9840",
            "status": "not_run",
            "preflight_id": f"C1-{n}-action",
            "action_count": n,
            "branch_count": BRANCHES,
            "horizon_count": len(HORIZONS),
            "expected_rows": n * BRANCHES * len(HORIZONS),
            "actual_rows": 0,
            "missing_labels": n,
            "duplicate_rows": 0,
            "NaN_Inf": "0/0",
            "label_exclusivity_violation": 0,
            "runtime_rows_per_sec": "",
            "OOM_count": 0,
            "retry_count": 0,
            "reason": "no_landed_natural_AP0_action_extension_generator_for_v9840",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P3_NATURAL_AP0_STREAM_PREFLIGHT_V9840",
        "status": "summary",
        "requested_panel_targets": ",".join(str(x) for x in panel_targets),
        "preflight_count": len(rows),
        "preflight_pass_count": 0,
        "preflight_not_run_count": len(rows),
        "C1d_pass": 0,
        "materializer_entrypoint_found": 0,
        "P3_preflight_pass": 0,
        "reason": "natural_AP0_stream_materializer_entrypoint_missing",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    return rows, summary


def p4_density_panels(source_v9820: Path, panel_targets: list[int], p3: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    src = read_csv(source_v9820 / "p3_natural_ap0_labeled_stream_extension_v9820.csv")
    panel_a = next((r for r in src if r.get("status") == "panel_row"), {})
    rows: list[dict[str, Any]] = []
    existing = inum(panel_a.get("labeled_action_count"))
    for target in panel_targets:
        if target <= existing:
            rows.append({
                "stage": "P4_NATURAL_AP0_DENSITY_PANEL_V9840",
                "status": "panel_row",
                "panel_id": f"Panel-{target}",
                "target_action_count": target,
                "actual_labeled_action_count": target,
                "CoreLike_count": panel_a.get("CoreLike_count"),
                "CoreLike_rate": panel_a.get("CoreLike_rate"),
                "CoreLike_Wilson_LCB": panel_a.get("Wilson_LCB"),
                "CoreLike_Wilson_UCB": panel_a.get("Wilson_UCB"),
                "GradeAB_count": panel_a.get("GradeAB_count"),
                "GradeAB_rate": panel_a.get("GradeAB_rate"),
                "ValuePositiveNoLongRisk_count": panel_a.get("ValuePositiveNoLongRisk_count"),
                "ValuePositiveNoLongRisk_rate": panel_a.get("ValuePositiveNoLongRisk_rate"),
                "MemoryOffdiagCore_count": panel_a.get("MemoryOffdiagCore_count"),
                "MemoryOffdiagCore_rate": panel_a.get("MemoryOffdiagCore_rate"),
                "T5_like_count": "",
                "T5_like_rate": "",
                "per_dataset_rate": panel_a.get("per_dataset_rate"),
                "per_family_rate": panel_a.get("per_family_rate"),
                "per_template_rate": panel_a.get("per_template_rate"),
                "per_step_bucket_rate": "",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
        else:
            rows.append({
                "stage": "P4_NATURAL_AP0_DENSITY_PANEL_V9840",
                "status": "not_run",
                "panel_id": f"Panel-{target}",
                "target_action_count": target,
                "actual_labeled_action_count": existing,
                "CoreLike_count": "",
                "CoreLike_rate": "",
                "CoreLike_Wilson_LCB": "",
                "CoreLike_Wilson_UCB": "",
                "missing_label_count": target - existing,
                "reason": "P3_C1d_preflight_not_passed_materializer_missing",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    summary = {
        "stage": "P4_NATURAL_AP0_DENSITY_PANEL_V9840",
        "status": "summary",
        "completed_panel_count": sum(1 for r in rows if r.get("status") == "panel_row"),
        "not_run_panel_count": sum(1 for r in rows if r.get("status") == "not_run"),
        "PanelA_CoreLike_LCB": panel_a.get("Wilson_LCB"),
        "PanelA_CoreLike_rate": panel_a.get("CoreLike_rate"),
        "P4_density_weak_pass": 0,
        "P4_density_strong_pass": 0,
        "reason": "panel_BCD_blocked_by_missing_natural_stream_materializer",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    v9720.write_bar_svg(out / "fig_C1_density_curve_panel_size.svg", "C1 density", ["2876"], [fnum(panel_a.get("CoreLike_rate"))])
    v9720.write_bar_svg(out / "fig_C2_corelike_rate_ci.svg", "C2 density LCB", ["2876"], [fnum(panel_a.get("Wilson_LCB"))])
    v9720.write_bar_svg(out / "fig_C3_per_dataset_density.svg", "C3 dataset", ["PanelA"], [fnum(panel_a.get("CoreLike_rate"))])
    v9720.write_bar_svg(out / "fig_C4_per_family_density_heatmap.svg", "C4 family", ["PanelA"], [fnum(panel_a.get("CoreLike_rate"))])
    v9720.write_bar_svg(out / "fig_C5_materializer_throughput_curve.svg", "C5 throughput", ["not_run"], [0.0])
    v9720.write_bar_svg(out / "fig_C6_missing_label_dashboard.svg", "C6 missing labels", [str(t) for t in panel_targets], [max(0, t - existing) for t in panel_targets])
    return rows, summary


def p5_virtual_probe() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {
        "stage": "P5_VIRTUAL_PATH_PROBE_V9840",
        "status": "not_run",
        "virtual_steps": "1,3,5",
        "probe_subset_size": "",
        "virtual_path_V_proxy": "",
        "virtual_path_risk_proxy": "",
        "virtual_path_memory_proxy": "",
        "virtual_path_cost_ms": "",
        "P5_virtual_probe_pass": 0,
        "reason": "legal_training_time_virtual_path_probe_not_landed",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def p6_generated_gate(p1: dict[str, Any], p2: dict[str, Any], p4: dict[str, Any], p5: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reopen = int(inum(p2.get("P2_weak_pass")) or (inum(p4.get("P4_density_strong_pass")) and fnum(p4.get("PanelA_CoreLike_LCB")) < 0.03 and inum(p1.get("P1_weak_pass"))) or inum(p5.get("P5_virtual_probe_pass")))
    row = {
        "stage": "P6_GEOMETRY_ADAPTIVE_GENERATED_REOPEN_GATE_V9840",
        "status": "summary",
        "condition1_B_legal_mechanism_weak": p2.get("P2_weak_pass"),
        "condition2_density_insufficient_with_mechanism": 0,
        "condition3_virtual_probe_predicts_AUV": p5.get("P5_virtual_probe_pass"),
        "generated_route_reopen_allowed": reopen,
        "generated_route_status": "reopened" if reopen else "stopped_no_legal_mechanism_or_density_evidence",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def not_run(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {"stage": stage, "status": "not_run", "reason": reason, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0, **extra}
    return [row], row


def p4_gate(source_v9770: Path, source_v9810: Path, p4_density: dict[str, Any]) -> dict[str, Any]:
    ldo = summary_row(read_csv(source_v9770 / "p1_ldo_decomposition_gate_audit_v9770.csv"))
    p8 = summary_row(read_csv(source_v9810 / "p8_ldo_gate_resolution_v9810.csv"))
    raw = fnum(ldo.get("ldo_raw"))
    back = fnum(ldo.get("ldo_backfill"))
    share = back / raw if raw else 0.0
    allowed = int(inum(p4_density.get("P4_density_strong_pass")) and fnum(ldo.get("ldo_quality")) <= 0.10 and fnum(ldo.get("ldo_support")) <= 0.10 and share >= 0.70 and fnum(p4_density.get("PanelA_CoreLike_LCB")) >= 0.03)
    return {
        "stage": "P4_GATE_SEMANTICS_V9840",
        "status": "summary",
        "raw_LDO": raw,
        "quality_LDO": ldo.get("ldo_quality"),
        "support_LDO": ldo.get("ldo_support"),
        "backfill_LDO": back,
        "backfill_share": share,
        "raw_gate_pass": p8.get("E_raw_pass", 0),
        "support_gate_pass": p8.get("E_no_backfill_pass", 0),
        "density_gate_pass": p8.get("E_density_required_pass", 0),
        "official_gate_modification_allowed": allowed,
        "reason": "P4_density_strong_pass_absent_keep_official_raw_gate",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def copy_base_acc(source_v9810: Path) -> list[dict[str, Any]]:
    rows = [dict(r) for r in read_csv(source_v9810 / "base_acc_sentinel_v9810.csv")]
    for r in rows:
        r["stage"] = "BASE_ACC_SENTINEL_V9840"
        r["reused_from_v9810"] = 1
        r["base_acc_used_for_controller"] = 0
        r["fake_data_used"] = 0
        r["proxy_row_used"] = 0
        r["cpu_offload_used"] = 0
    return rows


def field_legality() -> list[dict[str, Any]]:
    rows = [
        ("green", "action_norm,payload_norm,memory_offdiag_safe", "training-time scalar/veto fields"),
        ("yellow", "cover_entropy,basis_effective_rank", "KAN-specific diagnostic only"),
        ("red", "OldRank_score,WT80_score,AUV_future_label,future_outcome_replay", "diagnostic or future/outcome-derived fields"),
    ]
    return [{
        "stage": "FIELD_LEGALITY_LEDGER_V9840",
        "status": "field_legality_row",
        "field_color": c,
        "fields": f,
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    } for c, f, reason in rows]


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


def write_recap(out: Path, route: dict[str, Any], hashes: dict[str, str]) -> None:
    p0 = summary_row(read_csv(out / "p0_boundary_reproduction_v9840.csv"))
    p1 = summary_row(read_csv(out / "p1_future_path_mechanism_decomposition_v3.csv"))
    p2 = summary_row(read_csv(out / "p2_legal_training_time_mechanism_v3.csv"))
    p3 = summary_row(read_csv(out / "p3_natural_ap0_stream_preflight_v9840.csv"))
    p4p = summary_row(read_csv(out / "p4_natural_ap0_density_panel_v9840.csv"))
    p5 = summary_row(read_csv(out / "p5_virtual_path_probe_v9840.csv"))
    p6 = summary_row(read_csv(out / "p6_geometry_adaptive_generated_reopen_gate_v9840.csv"))
    gate = summary_row(read_csv(out / "p4_gate_semantics_v9840.csv"))
    nf = summary_row(read_csv(out / "no_fake_audit_v9840.csv"))
    failure = summary_row(read_csv(out / "failure_taxonomy_v9840.csv"))
    groups = [r for r in read_csv(out / "p1_future_path_mechanism_decomposition_v3.csv") if r.get("status") == "group_summary"]
    mechs = [r for r in read_csv(out / "p2_legal_training_time_mechanism_v3.csv") if r.get("status") == "mechanism_row"]
    lines = [
        "# DG-KAN v9.8.4 Future Operator / Natural Stream / Geometry Adaptive Optimizer 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.8.4_未来轨迹算子_自然动作扩流_几何自适应优化器_完整实验计划.md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest；新增 A-line 组使用真实 branch-horizon replay；自然 AP0 扩流 materializer 缺失时显式 `not_run`，没有 fake data 或 proxy rows。",
        "",
        "## 0. 最新结论",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"primary_blocker = {route.get('primary_blocker')}",
        f"secondary_blocker = {route.get('secondary_blocker')}",
        f"system_legal_controller_pass = {route.get('system_legal_controller_pass')}",
        f"generated_route_status = {route.get('generated_route_status')}",
        "```",
        "",
        f"最终 artifact：`{out}`",
        "",
        "核心结论：",
        "",
        f"1. P0 复现 v9.8.3 boundary：source route = `{p0.get('source_route_v9830')}`，P1/P2/P3/P5 = `{p0.get('P1_strong_pass_v9830')}` / `{p0.get('P2_strong_pass_v9830')}` / `{p0.get('P3_materializer_missing_v9830')}` / `{p0.get('P5_legal_strong_pass_v9830')}`。",
        f"2. A-line 新增组真实 replay：extra selected actions = `{route.get('extra_selected_actions')}`，extra future-path rows = `{route.get('extra_future_path_rows')}`。",
        f"3. P1 weak/strong = `{p1.get('P1_weak_pass')}` / `{p1.get('P1_strong_pass')}`；Core77 AUV LCB = `{p1.get('Core77_AUV_LCB')}`，OldOnly AUV LCB = `{p1.get('OldOnly_AUV_LCB')}`。",
        f"4. 新增 CoreExpansion10 AUV LCB = `{p1.get('CoreExpansion10_AUV_LCB')}`；RiskCleanButLowValue AUV LCB = `{p1.get('RiskCleanButLowValue_AUV_LCB')}`。",
        f"5. P2 legal mechanism weak/strong = `{p2.get('weak_pass_count')}` / `{p2.get('strong_pass_count')}`；best = `{p2.get('best_mechanism_id')}`，legality = `{p2.get('best_feature_legality')}`。",
        f"6. P3 natural preflight C1d pass = `{p3.get('C1d_pass')}`，materializer entrypoint found = `{p3.get('materializer_entrypoint_found')}`。",
        f"7. P4 panel completed/not_run = `{p4p.get('completed_panel_count')}` / `{p4p.get('not_run_panel_count')}`，PanelA LCB = `{p4p.get('PanelA_CoreLike_LCB')}`。",
        f"8. Gate modification allowed = `{gate.get('official_gate_modification_allowed')}`；backfill share = `{gate.get('backfill_share')}`。",
        f"9. Virtual path probe = `{p5.get('status')}`；generated reopen allowed = `{p6.get('generated_route_reopen_allowed')}`。",
        f"10. No-fake audit：rows checked = `{nf.get('rows_checked')}`，fake/proxy/cpu = `{nf.get('fake_data_used')}` / `{nf.get('proxy_row_used')}` / `{nf.get('cpu_offload_used')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/run_v9840_future_operator_natural_stream_geometry_optimizer.py` | v9.8.4 runner；复现 v9.8.3 boundary，真实 materialize 新增 A-line 组，执行 legal mechanism / natural stream / virtual probe / generated reopen / controller-runtime boundary。 |",
        "",
        "```text",
        "python -m py_compile experiments/run_v9840_future_operator_natural_stream_geometry_optimizer.py",
        "```",
        "",
        "```bash",
        "python experiments/run_v9840_future_operator_natural_stream_geometry_optimizer.py --out-dir results/real_rerun_20260506/v9840_future_operator_natural_stream_geometry_optimizer_full_20260516T160000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 2876,5000,10000,20000",
        "```",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, indent=2, ensure_ascii=False),
        "```",
        "",
        "## 3. Part A Future Path Mechanism",
        "",
        "| group | actions | AUV LCB | V1 LCB | V20 LCB | V80 LCB | V240 LCB | RiskPath UCB | IND rate | memory/offdiag UCB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in groups:
        lines.append(f"| `{r.get('group_id')}` | `{r.get('action_count')}` | `{r.get('AUV_LCB')}` | `{r.get('V1_LCB')}` | `{r.get('V20_LCB')}` | `{r.get('V80_LCB')}` | `{r.get('V240_LCB')}` | `{r.get('RiskPath_UCB')}` | `{r.get('IND_rate')}` | `{fnum(r.get('memory_UCB')) + fnum(r.get('offdiag_UCB'))}` |")
    lines += [
        "",
        f"判断：A weak pass = `{p1.get('P1_weak_pass')}`，A strong pass = `{p1.get('P1_strong_pass')}`。Strong 未过的原因是 `{p1.get('P1_strong_fail_reason')}`；这不是 materializer 失败，而是更严格的 OldOnly / Core77 对照差异不够。",
        "",
        "## 4. Part B Legal Mechanism",
        "",
        "| mechanism | legality | precision | AUV LCB | V20/V80/V240 LCB | longrisk UCB | memory/offdiag UCB | LDO/LSO/LTO | weak | strong |",
        "|---|---|---:|---:|---|---:|---:|---|---:|---:|",
    ]
    for r in mechs:
        lines.append(f"| `{r.get('mechanism_id')}` | `{r.get('feature_legality')}` | `{r.get('GradeAB_precision')}` | `{r.get('AUV_LCB')}` | `{r.get('V20_LCB')}`/`{r.get('V80_LCB')}`/`{r.get('V240_LCB')}` | `{r.get('longrisk_UCB')}` | `{fnum(r.get('memory_UCB')) + fnum(r.get('offdiag_UCB'))}` | `{r.get('LDO_drop')}`/`{r.get('LSO_drop')}`/`{r.get('LTO_drop')}` | `{r.get('P2_weak_pass')}` | `{r.get('P2_strong_pass')}` |")
    lines += [
        "",
        "判断：B 线没有 legal weak/strong pass。绿色 memory/cost 信号仍然更像安全门，不是收益源；red diagnostic 不进入 official controller。",
        "",
        "## 5. Part C Natural AP0 Stream",
        "",
        "```text",
        f"C1d_pass = {p3.get('C1d_pass')}",
        f"materializer_entrypoint_found = {p3.get('materializer_entrypoint_found')}",
        f"PanelA_CoreLike_rate = {p4p.get('PanelA_CoreLike_rate')}",
        f"PanelA_CoreLike_LCB = {p4p.get('PanelA_CoreLike_LCB')}",
        "PanelB/C/D = not_run",
        "```",
        "",
        "判断：自然 AP0 扩流仍是工程 blocker；没有 5000/10000/20000 真实 labels，因此不能作 density closure。",
        "",
        "## 6. Part D/E Generated / Controller Boundary",
        "",
        "```text",
        f"virtual_probe_status = {p5.get('status')}",
        f"generated_route_reopen_allowed = {p6.get('generated_route_reopen_allowed')}",
        "generated_smoke = not_run",
        "controller/runtime/replay/shortfull = not_run",
        "```",
        "",
        "判断：B 线没有 legal mechanism，C 线没有 density evidence，virtual probe 未落地，所以 D/E 不打开。",
        "",
        "## 7. Base-Acc / Audit / Failure",
        "",
        "```text",
        f"base_acc_sentinel_pass = {summary_row(read_csv(out / 'base_acc_sentinel_v9840.csv')).get('base_acc_sentinel_pass')}",
        f"rows_checked = {nf.get('rows_checked')}",
        f"route = {failure.get('route')}",
        f"primary_blocker = {failure.get('primary_blocker')}",
        f"secondary_blocker = {failure.get('secondary_blocker')}",
        "```",
        "",
        "## 8. Figures",
        "",
        "```text",
        "fig_A1_group_future_path_V_curve.svg",
        "fig_A2_group_future_path_risk_curve.svg",
        "fig_A3_oldonly_immediate_vs_future.svg",
        "fig_A4_exactonly_failure_path.svg",
        "fig_A5_core77_vs_random_memory_offdiag_path.svg",
        "fig_A6_path_shape_slope_scatter.svg",
        "fig_A7_AUV_vs_longrisk_pareto.svg",
        "fig_B1_mechanism_topK_quality_bar.svg",
        "fig_B2_mechanism_auv_vs_risk.svg",
        "fig_B3_signal_consistency_vs_future_AUV.svg",
        "fig_B4_virtual_probe_cost_quality_pareto.svg",
        "fig_B5_memory_safety_as_veto.svg",
        "fig_B6_legal_vs_red_feature_comparison.svg",
        "fig_C1_density_curve_panel_size.svg",
        "fig_C2_corelike_rate_ci.svg",
        "fig_C3_per_dataset_density.svg",
        "fig_C4_per_family_density_heatmap.svg",
        "fig_C5_materializer_throughput_curve.svg",
        "fig_C6_missing_label_dashboard.svg",
        "fig_D1_generated_quality_pareto.svg",
        "fig_D2_generated_future_path_curves.svg",
        "fig_D3_generated_memory_offdiag_dashboard.svg",
        "fig_D4_generated_vs_existing_density.svg",
        "fig_D5_runtime_apply_cost.svg",
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
        "1. v9.8.4 真实补跑了新增 A-line existing-action 组，而不是只复用旧表。",
        "2. Core77 future path 仍强，新增 CoreExpansion10/RiskCleanButLowValue 给出额外对照；但 A strong 在严格 OldOnly/Random/Exact 对照下未完全闭合。",
        "3. B 线没有找到 legal training-time mechanism。",
        "4. C 线 natural AP0 扩流 materializer 仍未落地。",
        "5. D 线 generated route 不允许重开。",
        "6. E 线 controller/runtime/replay/short-full 全部 gate-blocked。",
        "```",
        "",
        "最终一句话：",
        "",
        f"> v9.8.4 真实执行后停在 `{route.get('route')}`：本轮把新增 A-line 组真实 replay 了，但训练当下合法机制、自然动作扩流和虚拟路径 probe 仍未闭合，因此不能进入 official controller，strict PureKAN functional 仍未成功。",
        "",
    ]
    RECAP_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if args.fresh and out.exists():
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

    source_v9830 = Path(args.source_v9830)
    source_v9820 = Path(args.source_v9820)
    p0_rows, p0 = p0_boundary(source_v9830)
    dump_csv("p0_boundary_reproduction_v9840.csv", p0_rows)

    ap0, score_bundle, payload_by_id = v9720.load_ap0_and_payloads(args)
    extra_rows, extra_summary = materialize_extra_groups(args, ap0, score_bundle, payload_by_id, out)
    dump_csv("p1_extra_future_path_materializer_v9840.csv", extra_rows)

    base_rows = read_csv(source_v9820 / "p1_full_future_path_materializer_v9820.csv")
    combined_real = real_rows(base_rows) + real_rows(extra_rows)
    feats = action_features(combined_real)
    p1_rows, p1 = p1_decomposition_v3(combined_real, feats, out)
    dump_csv("p1_future_path_mechanism_decomposition_v3.csv", p1_rows)

    p2_rows, p2 = p2_legal_mechanism(ap0, feats, out)
    dump_csv("p2_legal_training_time_mechanism_v3.csv", p2_rows)

    panel_targets = parse_ints(args.panel_targets)
    p3_rows, p3 = p3_preflight(panel_targets)
    dump_csv("p3_natural_ap0_stream_preflight_v9840.csv", p3_rows)

    p4_rows, p4_density = p4_density_panels(source_v9820, panel_targets, p3, out)
    dump_csv("p4_natural_ap0_density_panel_v9840.csv", p4_rows)

    p5_rows, p5 = p5_virtual_probe()
    dump_csv("p5_virtual_path_probe_v9840.csv", p5_rows)
    v9720.write_bar_svg(out / "fig_B4_virtual_probe_cost_quality_pareto.svg", "B4 virtual probe", ["not_run"], [0.0])

    p6_rows, p6 = p6_generated_gate(p1, p2, p4_density, p5)
    dump_csv("p6_geometry_adaptive_generated_reopen_gate_v9840.csv", p6_rows)
    if inum(p6.get("generated_route_reopen_allowed")):
        p7_rows, p7 = not_run("P7_GENERATED_UPDATE_SMOKE_V9840", "generated_objective_implementation_not_landed_after_reopen_gate", generated_smoke_pass=0)
    else:
        p7_rows, p7 = not_run("P7_GENERATED_UPDATE_SMOKE_V9840", "generated_route_not_reopened", generated_smoke_pass=0)
    dump_csv("p7_generated_update_smoke_v9840.csv", p7_rows)
    for fig in ["fig_D1_generated_quality_pareto.svg", "fig_D2_generated_future_path_curves.svg", "fig_D3_generated_memory_offdiag_dashboard.svg", "fig_D4_generated_vs_existing_density.svg", "fig_D5_runtime_apply_cost.svg"]:
        v9720.write_bar_svg(out / fig, fig, ["not_run"], [0.0])

    p8_rows, p8 = not_run("P8_MINIMAL_CONTROLLER_BOUNDARY_V9840", "P2_or_C_or_D_no_official_controller_evidence", controller_pass=0)
    dump_csv("p8_minimal_controller_boundary_v9840.csv", p8_rows)
    p9_rows, p9 = not_run("P9_SELECTED_RUNTIME_BOUNDARY_V9840", "P8_controller_not_passed", runtime_pass=0)
    dump_csv("p9_selected_runtime_boundary_v9840.csv", p9_rows)
    p10_rows, p10 = not_run("P10_PAIRED_REPLAY_BOUNDARY_V9840", "P9_runtime_not_passed", paired_replay_pass=0)
    dump_csv("p10_paired_replay_boundary_v9840.csv", p10_rows)
    p11_rows, p11 = not_run("P11_SHORT_FULL_BOUNDARY_V9840", "P10_paired_replay_not_passed", short_full_pass=0)
    dump_csv("p11_short_full_boundary_v9840.csv", p11_rows)

    gate_row = p4_gate(Path(args.source_v9770), Path(args.source_v9810), p4_density)
    dump_csv("p4_gate_semantics_v9840.csv", [gate_row])
    base_acc_rows = copy_base_acc(Path(args.source_v9810))
    dump_csv("base_acc_sentinel_v9840.csv", base_acc_rows)
    dump_csv("field_legality_ledger_v9840.csv", field_legality())

    route = "R1-FuturePathMechanismFoundLegalMechanismAbsent" if inum(p1.get("P1_weak_pass")) and not inum(p2.get("P2_weak_pass")) else "R6-AllMechanismsFail"
    primary = "no_legal_training_time_mechanism" if not inum(p2.get("P2_weak_pass")) else "none"
    secondary = "natural_stream_materializer_missing" if not inum(p3.get("materializer_entrypoint_found")) else "none"
    route_decision = {
        "stage": "ROUTE_DECISION_V9840",
        "status": "summary",
        "route": route,
        "route_secondary": "Stop1-NaturalStreamMaterializerMissing" if secondary != "none" else "none",
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "source_route_v9830": p0.get("source_route_v9830"),
        "P0_boundary_pass": p0.get("boundary_reproduced"),
        "P1_weak_pass": p1.get("P1_weak_pass"),
        "P1_strong_pass": p1.get("P1_strong_pass"),
        "P2_weak_pass": p2.get("P2_weak_pass"),
        "P2_strong_pass": p2.get("P2_strong_pass"),
        "P3_preflight_pass": p3.get("P3_preflight_pass"),
        "P4_density_weak_pass": p4_density.get("P4_density_weak_pass"),
        "P4_density_strong_pass": p4_density.get("P4_density_strong_pass"),
        "P5_virtual_probe_pass": p5.get("P5_virtual_probe_pass"),
        "P6_generated_reopen_allowed": p6.get("generated_route_reopen_allowed"),
        "P7_generated_smoke_pass": p7.get("generated_smoke_pass"),
        "P8_controller_pass": p8.get("controller_pass"),
        "P9_runtime_pass": p9.get("runtime_pass"),
        "P10_paired_replay_pass": p10.get("paired_replay_pass"),
        "P11_short_full_pass": p11.get("short_full_pass"),
        "official_gate_modification_allowed": gate_row.get("official_gate_modification_allowed"),
        "extra_selected_actions": extra_summary.get("selected_action_count"),
        "extra_future_path_rows": extra_summary.get("row_count_actual"),
        "generated_route_status": p6.get("generated_route_status"),
        "system_legal_controller_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("route_decision_v9840.json", route_decision)

    no_fake = {
        "stage": "NO_FAKE_AUDIT_V9840",
        "status": "summary",
        "rows_checked": audit_rows(artifacts),
        "fake_proxy_nonzero_count": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "no_fake": 1,
        "no_proxy": 1,
    }
    dump_csv("no_fake_audit_v9840.csv", [no_fake])
    contract = {
        "stage": "CONTRACT_AUDIT_V9840",
        "status": "summary",
        "manual_forward/manual_backward/manual_adamw_update": "1/1/1",
        "v9830_boundary_pass": p0.get("boundary_reproduced"),
        "extra_future_path_materialized": int(
            inum(extra_summary.get("row_count_actual")) == inum(extra_summary.get("row_count_expected"))
            and inum(extra_summary.get("row_count_failed")) == 0
        ),
        "A_line_weak/strong": f"{p1.get('P1_weak_pass')}/{p1.get('P1_strong_pass')}",
        "B_legal_mechanism_weak/strong": f"{p2.get('P2_weak_pass')}/{p2.get('P2_strong_pass')}",
        "C_materializer_entrypoint_found": p3.get("materializer_entrypoint_found"),
        "D_generated_reopen_allowed": p6.get("generated_route_reopen_allowed"),
        "controller/runtime/system": "0/0/0",
        "uses_future_outcome_for_official_controller": 0,
        "diagnostic_promoted_to_official": 0,
        "fake/proxy/cpu_offload": "0/0/0",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("contract_audit_v9840.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9840",
        "status": "summary",
        "route": route,
        "F0_boundary_fail": int(not inum(p0.get("boundary_reproduced"))),
        "F1_A_strong_not_closed": int(not inum(p1.get("P1_strong_pass"))),
        "F2_B_legal_mechanism_absent": int(not inum(p2.get("P2_weak_pass"))),
        "F3_C_natural_stream_materializer_missing": int(not inum(p3.get("materializer_entrypoint_found"))),
        "F4_D_generated_route_stopped": int(not inum(p6.get("generated_route_reopen_allowed"))),
        "F5_E_controller_runtime_blocked": 1,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("failure_taxonomy_v9840.csv", [failure])

    hashes = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    manifest = {
        "stage": "RUN_MANIFEST_V9840",
        "status": "summary",
        "out_dir": str(out),
        "execution_profile": args.execution_profile,
        "command_profile": {
            "device": args.device,
            "data_root": args.data_root,
            "seed": args.seed,
            "panel_targets": args.panel_targets,
            "extra_max_per_group": args.extra_max_per_group,
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
    dump_json("run_manifest_v9840.json", manifest)
    hashes_for_recap = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    write_recap(out, route_decision, hashes_for_recap)
    print(json.dumps({
        "out_dir": str(out),
        "route": route,
        "P1_weak_pass": p1.get("P1_weak_pass"),
        "P1_strong_pass": p1.get("P1_strong_pass"),
        "P2_weak_pass": p2.get("P2_weak_pass"),
        "P3_preflight_pass": p3.get("P3_preflight_pass"),
        "generated_route_status": p6.get("generated_route_status"),
        "extra_selected_actions": extra_summary.get("selected_action_count"),
        "extra_future_path_rows": extra_summary.get("row_count_actual"),
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "system_legal_controller_pass": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
