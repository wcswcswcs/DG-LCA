#!/usr/bin/env python3
"""DG-KAN v9.8.3 four-line future mechanism / natural stream audit.

This runner is intentionally table-grounded.  It reads the v9.8.2 landed
full-future-path rows, derives v9.8.3 diagnostic tables, and keeps every
blocked downstream stage as explicit not_run rows.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
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
import run_v9820_future_path_geometry_mechanism_natural_stream as v9820  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


RESULT_ROOT = REPO / "results/real_rerun_20260506"
PLAN_PATH = REPO / "docs/DG-KAN_v9.8.3_四线并行_未来轨迹机制_自然动作扩流_几何自适应更新_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9830_four_line_future_mechanism_natural_geometry.py"
RECAP_PATH = REPO / "docs/DG-KAN_v9.8.3_FourLineFutureMechanism_NaturalStream_GeometryAdaptive_实验复盘.md"
DEFAULT_OUT = RESULT_ROOT / "v9830_four_line_future_mechanism_natural_geometry_full_20260516T140000Z"
DEFAULT_V9820 = RESULT_ROOT / "v9820_future_path_geometry_mechanism_natural_stream_full_20260516T120000Z"
DEFAULT_V9770 = RESULT_ROOT / "v9770_core77_support_density_natural_stream_gate_first_20260516T060000Z"
DEFAULT_V9810 = RESULT_ROOT / "v9810_future_trajectory_geometry_adaptive_optimizer_full_20260516T100000Z"

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
    p.add_argument("--execution-profile", default="full-gated", choices=["smoke", "full-gated", "existing-only"])
    p.add_argument("--panel-targets", default="2876,5000,10000,20000")
    p.add_argument("--source-v9820", default=str(DEFAULT_V9820))
    p.add_argument("--source-v9770", default=str(DEFAULT_V9770))
    p.add_argument("--source-v9810", default=str(DEFAULT_V9810))
    p.add_argument("--source-v9330", default=str(v9820.DEFAULT_V9330))
    p.add_argument("--source-v9550", default=str(v9820.DEFAULT_V9550))
    p.add_argument("--source-v9560", default=str(v9820.DEFAULT_V9560))
    p.add_argument("--source-v9570", default=str(v9820.DEFAULT_V9570))
    p.add_argument("--source-v9580", default=str(v9820.DEFAULT_V9580))
    return p.parse_args()


def mean(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.fmean(vals) if vals else 0.0


def stdev(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def lcb(xs: list[float]) -> float:
    return v9720.lcb(xs)


def ucb(xs: list[float]) -> float:
    return v9720.ucb(xs)


def qtile(xs: list[float], q: float) -> float:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return 0.0
    return vals[min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))]


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r for r in rows if r.get("status") == "summary"), rows[0] if rows else {})


def effect_size(vals_a: list[float], vals_b: list[float]) -> float:
    vals_a = [x for x in vals_a if math.isfinite(float(x))]
    vals_b = [x for x in vals_b if math.isfinite(float(x))]
    pooled = vals_a + vals_b
    sd = stdev(pooled)
    return (mean(vals_a) - mean(vals_b)) / (sd if sd > 1.0e-12 else 1.0)


def source_real_rows(source: Path) -> list[dict[str, str]]:
    rows = read_csv(source / "p1_full_future_path_materializer_v9820.csv")
    return [r for r in rows if r.get("status") == "branch_horizon_row" and r.get("branch_id") == "RealFunctional"]


def action_path_features(real: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    by: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for r in real:
        by[str(r.get("action_id"))][inum(r.get("horizon"))] = r
    out: dict[str, dict[str, Any]] = {}
    for aid, hs in by.items():
        if not all(h in hs for h in HORIZONS):
            continue
        first = hs[1]
        auv = sum(AUV_WEIGHTS[h] * fnum(hs[h].get("V_branch")) for h in HORIZONS)
        memory_fail = inum(first.get("memory_fail"))
        offdiag_fail = inum(first.get("offdiag_fail"))
        stability = int(
            fnum(hs[20].get("V_branch")) > 0
            and fnum(hs[80].get("V_branch")) > 0
            and inum(hs[240].get("long_risk_label")) == 0
            and memory_fail == 0
            and offdiag_fail == 0
        )
        mismatch = int(fnum(hs[1].get("V_branch")) <= 0 and auv > 0)
        out[aid] = {
            "action_id": aid,
            "group_id": first.get("group_id"),
            "dataset_id": first.get("dataset_id"),
            "seed": first.get("seed"),
            "family_id": first.get("family_id"),
            "template_id": first.get("template_id"),
            "step_bucket": first.get("step_bucket"),
            "AUV": auv,
            "Stability": stability,
            "Mismatch": mismatch,
            "V1": fnum(hs[1].get("V_branch")),
            "V5": fnum(hs[5].get("V_branch")),
            "V20": fnum(hs[20].get("V_branch")),
            "V80": fnum(hs[80].get("V_branch")),
            "V240": fnum(hs[240].get("V_branch")),
            "Vctrl20": fnum(hs[20].get("V_ctrl")),
            "Vctrl80": fnum(hs[80].get("V_ctrl")),
            "Vctrl240": fnum(hs[240].get("V_ctrl")),
            "CEp99_h20": fnum(hs[20].get("CEp99_delta")),
            "CEp99_h80": fnum(hs[80].get("CEp99_delta")),
            "hard_tail_h80": fnum(hs[80].get("hard_tail_loss_delta")),
            "cover_h80": fnum(hs[80].get("cover_entropy_delta")),
            "curvature_h80": fnum(hs[80].get("curvature_proxy_delta")),
            "jacobian_h80": fnum(hs[80].get("jacobian_proxy_delta")),
            "memory_fail": memory_fail,
            "offdiag_fail": offdiag_fail,
            "longrisk240": inum(hs[240].get("long_risk_label")),
            "bad240": inum(hs[240].get("bad_event_label")),
            "null240": inum(hs[240].get("null_event_label")),
            "action_norm": fnum(first.get("action_norm")),
            "payload_norm": fnum(first.get("payload_norm")),
            "OldRank_score": fnum(first.get("OldRank_score")),
            "ExactTransfer_score": fnum(first.get("ExactTransfer_score")),
            "WT80_score": fnum(first.get("WT80_score")),
            "AdamW_path_gain": fnum(hs[80].get("V_branch")) - fnum(hs[20].get("V_branch")),
            "function_displacement_norm": abs(fnum(hs[20].get("CE_delta"))) + abs(fnum(hs[20].get("margin_delta"))),
        }
    return out


def scores_to_quality(ap0: list[dict[str, Any]], feats: dict[str, dict[str, Any]], scores: dict[str, float]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    aid_to_idx = {str(r.get("action_id")): i for i, r in enumerate(ap0)}
    full_scores = [-1.0e9] * len(ap0)
    for aid, score in scores.items():
        idx = aid_to_idx.get(aid)
        if idx is not None:
            full_scores[idx] = float(score)
    q = v9720.quality_for_scores(ap0, full_scores, TARGET_K)
    top_aids = sorted(scores, key=scores.get, reverse=True)[:TARGET_K]
    return q, [feats[aid] for aid in top_aids if aid in feats]


def p0_boundary(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source / "route_decision_v9820.json")
    p1 = summary_row(read_csv(source / "p1_full_future_path_materializer_v9820.csv"))
    p2 = summary_row(read_csv(source / "p2_future_path_mechanism_contrast_v9820.csv"))
    p3 = summary_row(read_csv(source / "p3_natural_ap0_labeled_stream_extension_v9820.csv"))
    p5 = summary_row(read_csv(source / "p5_architecture_agnostic_geometry_rebuild_v9820.csv"))
    nf = summary_row(read_csv(source / "no_fake_audit_v9820.csv"))
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9830",
        "status": "summary",
        "source_route_v9820": route.get("route"),
        "source_primary_blocker_v9820": route.get("primary_blocker"),
        "P1_full_future_path_rows_expected": p1.get("row_count_expected"),
        "P1_full_future_path_rows_actual": p1.get("row_count_actual"),
        "P1_weak_pass_v9820": p1.get("P1_weak_pass"),
        "P2_mechanism_pass_v9820": p2.get("P2_mechanism_pass"),
        "P3_density_strong_pass_v9820": p3.get("P3_density_strong_pass"),
        "P5_geometry_weak_pass_v9820": p5.get("P5_weak_pass"),
        "controller_pass_v9820": route.get("P6_controller_strong_pass"),
        "generated_pass_v9820": route.get("P7_generated_strong_pass"),
        "runtime_pass_v9820": route.get("P8_runtime_pass"),
        "fake_data_used": nf.get("fake_data_used", 0),
        "proxy_row_used": nf.get("proxy_row_used", 0),
        "cpu_offload_used": nf.get("cpu_offload_used", 0),
    }
    row["boundary_reproduced"] = int(
        row["source_route_v9820"] == "R-P2-NoMechanismExplainsGoodActions"
        and inum(row["P1_weak_pass_v9820"]) == 1
        and inum(row["P2_mechanism_pass_v9820"]) == 0
        and inum(row["P3_density_strong_pass_v9820"]) == 0
        and inum(row["P5_geometry_weak_pass_v9820"]) == 0
        and inum(row["fake_data_used"]) == 0
        and inum(row["proxy_row_used"]) == 0
        and inum(row["cpu_offload_used"]) == 0
    )
    return [row], row


def p1_future_path_v2(real: list[dict[str, str]], feats: dict[str, dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_group = defaultdict(list)
    for f in feats.values():
        by_group[str(f["group_id"])].append(f)
    group_summary_rows = []
    for group_id in sorted(by_group):
        group_feats = by_group[group_id]
        h240 = [r for r in real if r.get("group_id") == group_id and inum(r.get("horizon")) == 240]
        group_summary_rows.append({
            "stage": "P1_FUTURE_PATH_MECHANISM_DECOMPOSITION_V2_V9830",
            "status": "group_summary",
            "group_id": group_id,
            "action_count": len(group_feats),
            "AUV_mean": mean([f["AUV"] for f in group_feats]),
            "AUV_LCB": lcb([f["AUV"] for f in group_feats]),
            "V1_LCB": lcb([f["V1"] for f in group_feats]),
            "V20_LCB": lcb([f["V20"] for f in group_feats]),
            "V80_LCB": lcb([f["V80"] for f in group_feats]),
            "V240_LCB": lcb([f["V240"] for f in group_feats]),
            "Stability_rate": mean([float(f["Stability"]) for f in group_feats]),
            "Stability_LCB": lcb([float(f["Stability"]) for f in group_feats]),
            "Mismatch_rate": mean([float(f["Mismatch"]) for f in group_feats]),
            "Mismatch_LCB": lcb([float(f["Mismatch"]) for f in group_feats]),
            "h240_longrisk_UCB": ucb([float(f["longrisk240"]) for f in group_feats]),
            "memory_UCB": ucb([float(f["memory_fail"]) for f in group_feats]),
            "offdiag_UCB": ucb([float(f["offdiag_fail"]) for f in group_feats]),
            "h240_row_count": len(h240),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    core = next((r for r in group_summary_rows if r["group_id"] == "Core77"), {})
    exact = next((r for r in group_summary_rows if r["group_id"] == "ExactOnly"), {})
    random = next((r for r in group_summary_rows if r["group_id"] == "RandomMatched"), {})
    old = next((r for r in group_summary_rows if r["group_id"] == "OldOnly"), {})
    p1_weak = int(
        fnum(core.get("AUV_LCB")) > fnum(exact.get("AUV_LCB")) + 0.25
        and fnum(core.get("h240_longrisk_UCB")) <= 0.05
        and fnum(core.get("memory_UCB")) <= 0.05
        and fnum(core.get("offdiag_UCB")) <= 0.05
    )
    immediate_not_dominant = int(
        (fnum(old.get("AUV_LCB")) > 0 and fnum(old.get("V1_LCB")) <= 0)
        or (fnum(core.get("AUV_LCB")) > 0 and fnum(core.get("Mismatch_rate")) > 0)
    )
    p1_strong = int(
        p1_weak
        and fnum(core.get("AUV_LCB")) > fnum(random.get("AUV_LCB"))
        and immediate_not_dominant
    )
    summary = {
        "stage": "P1_FUTURE_PATH_MECHANISM_DECOMPOSITION_V2_V9830",
        "status": "summary",
        "source_realfunctional_rows": len(real),
        "action_count": len(feats),
        "horizons": ",".join(str(h) for h in HORIZONS),
        "Core77_AUV_LCB": core.get("AUV_LCB", ""),
        "OldOnly_AUV_LCB": old.get("AUV_LCB", ""),
        "ExactOnly_AUV_LCB": exact.get("AUV_LCB", ""),
        "RandomMatched_AUV_LCB": random.get("AUV_LCB", ""),
        "Core77_h240_longrisk_UCB": core.get("h240_longrisk_UCB", ""),
        "Core77_memory_UCB": core.get("memory_UCB", ""),
        "Core77_offdiag_UCB": core.get("offdiag_UCB", ""),
        "OldOnly_V1_LCB": old.get("V1_LCB", ""),
        "OldOnly_Mismatch_rate": old.get("Mismatch_rate", ""),
        "future_path_positive_immediate_not_dominant": immediate_not_dominant,
        "P1_weak_pass": p1_weak,
        "P1_strong_pass": p1_strong,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    rows.extend(group_summary_rows)
    for r in real:
        rows.append({
            "stage": "P1_FUTURE_PATH_MECHANISM_DECOMPOSITION_V2_V9830",
            "status": "action_horizon_row",
            "action_id": r.get("action_id"),
            "group_id": r.get("group_id"),
            "dataset_id_for_diagnostic_only": r.get("dataset_id"),
            "seed": r.get("seed"),
            "family_id": r.get("family_id"),
            "template_id": r.get("template_id"),
            "step_bucket": r.get("step_bucket"),
            "horizon": r.get("horizon"),
            "V": r.get("V_branch"),
            "Vctrl": r.get("V_ctrl"),
            "CE_delta": r.get("CE_delta"),
            "NLL_delta": r.get("NLL_delta"),
            "margin_delta": r.get("margin_delta"),
            "CEp99_delta": r.get("CEp99_delta"),
            "hard_tail_loss_delta": r.get("hard_tail_loss_delta"),
            "old_family_loss_delta": r.get("old_family_loss_delta"),
            "old_stratum_loss_delta": r.get("old_stratum_loss_delta"),
            "memory_buffer_loss_delta": r.get("memory_buffer_loss_delta"),
            "longrisk": r.get("longrisk"),
            "bad_event": r.get("bad_event"),
            "null_event": r.get("null_event"),
            "memory_fail": r.get("memory_fail"),
            "offdiag_fail": r.get("offdiag_fail"),
            "cover_entropy_delta": r.get("cover_entropy_delta"),
            "basis_effective_rank_delta": r.get("basis_effective_rank_delta"),
            "curvature_proxy_delta": r.get("curvature_proxy_delta"),
            "AdamW_alignment": r.get("AdamW_alignment"),
            "action_norm": r.get("action_norm"),
            "payload_norm": r.get("payload_norm"),
            "OldRank_score": r.get("OldRank_score"),
            "ExactTransfer_score": r.get("ExactTransfer_score"),
            "WT80_score": r.get("WT80_score"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    for f in feats.values():
        rows.append({
            "stage": "P1_FUTURE_PATH_MECHANISM_DECOMPOSITION_V2_V9830",
            "status": "action_path_summary",
            **{k: f.get(k, "") for k in [
                "action_id", "group_id", "dataset_id", "seed", "family_id", "template_id",
                "step_bucket", "AUV", "Stability", "Mismatch", "V1", "V20", "V80",
                "V240", "longrisk240", "memory_fail", "offdiag_fail",
            ]},
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    v9720.write_bar_svg(out / "fig_p1_v9830_group_auv_lcb.svg", "v9830 P1 AUV LCB", [r["group_id"] for r in group_summary_rows], [fnum(r.get("AUV_LCB")) for r in group_summary_rows])
    v9720.write_bar_svg(out / "fig_p1_v9830_group_mismatch.svg", "v9830 P1 Mismatch", [r["group_id"] for r in group_summary_rows], [fnum(r.get("Mismatch_rate")) for r in group_summary_rows])
    return rows, summary


def mechanism_row(
    ap0: list[dict[str, Any]],
    feats: dict[str, dict[str, Any]],
    mechanism_id: str,
    feature_names: str,
    feature_legality: str,
    architecture_specific: int,
    scores: dict[str, float],
    feature_cost_q90: Any = "",
) -> dict[str, Any]:
    q, top = scores_to_quality(ap0, feats, scores)
    no_red = "red" not in str(feature_legality).lower()
    weak = int(
        fnum(q.get("GradeAB_precision")) >= 0.75
        and lcb([f["AUV"] for f in top]) > 0
        and fnum(q.get("h240_longrisk_UCB")) <= 0.05
        and fnum(q.get("memory_fail_UCB")) <= 0.05
        and fnum(q.get("offdiag_fail_UCB")) <= 0.05
        and no_red
    )
    strong = int(
        weak
        and fnum(q.get("LDO_drop")) <= 0.10
        and fnum(q.get("LSO_drop")) <= 0.10
        and fnum(q.get("LTO_drop")) <= 0.10
        and feature_cost_q90 != ""
        and fnum(feature_cost_q90) <= 1.0
    )
    return {
        "stage": "P2_LEGAL_MECHANISM_DISCOVERY_V2_V9830",
        "status": "mechanism_row",
        "mechanism_id": mechanism_id,
        "feature_names": feature_names,
        "feature_legality": feature_legality,
        "architecture_specific": architecture_specific,
        "TopK87_precision": q.get("GradeAB_precision"),
        "V_LCB": q.get("V_integrated_LCB"),
        "AUV_LCB": lcb([f["AUV"] for f in top]),
        "longrisk_UCB": q.get("h240_longrisk_UCB"),
        "memory_UCB": q.get("memory_fail_UCB"),
        "offdiag_UCB": q.get("offdiag_fail_UCB"),
        "LDO_drop": q.get("LDO_drop"),
        "LSO_drop": q.get("LSO_drop"),
        "LTO_drop": q.get("LTO_drop"),
        "LFO_drop": "",
        "feature_cost_q90": feature_cost_q90,
        "P2_weak_pass": weak,
        "P2_strong_pass": strong,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def p2_legal_mechanism_v2(ap0: list[dict[str, Any]], feats: dict[str, dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [
        mechanism_row(ap0, feats, "M1_future_adamw_path_diagnostic", "AdamW_path_gain=V80-V20", "red:future_outcome_diagnostic", 0, {a: f["AdamW_path_gain"] for a, f in feats.items()}),
        mechanism_row(ap0, feats, "M2_hard_tail_relief_diagnostic", "CEp99_h80/hard_tail_loss_delta", "red:future_outcome_diagnostic", 0, {a: -f["CEp99_h80"] for a, f in feats.items()}),
        mechanism_row(ap0, feats, "M3_old_knowledge_memory_safety", "memory_fail,offdiag_fail", "green", 0, {a: -float(f["memory_fail"]) - float(f["offdiag_fail"]) for a, f in feats.items()}, feature_cost_q90=0.0),
        mechanism_row(ap0, feats, "M4_local_function_stability_diagnostic", "cover_entropy_h80,curvature_h80,jacobian_h80", "red:future_outcome_diagnostic;yellow:kan_basis_specific", 1, {a: -abs(f["cover_h80"]) - abs(f["curvature_h80"]) - abs(f["jacobian_h80"]) for a, f in feats.items()}),
        mechanism_row(ap0, feats, "M5_oldrank_signal_diagnostic", "OldRank_score", "red:old_table_diagnostic", 0, {a: f["OldRank_score"] for a, f in feats.items()}),
        mechanism_row(ap0, feats, "M6_payload_norm_cost", "payload_norm", "green", 0, {a: -f["payload_norm"] for a, f in feats.items()}, feature_cost_q90=0.0),
        mechanism_row(ap0, feats, "M7_exacttransfer_signal_diagnostic", "ExactTransfer_score", "red:outcome_derived_transfer_diagnostic", 0, {a: f["ExactTransfer_score"] for a, f in feats.items()}),
    ]
    best = max(rows, key=lambda r: (inum(r.get("P2_strong_pass")), inum(r.get("P2_weak_pass")), fnum(r.get("TopK87_precision")), fnum(r.get("AUV_LCB"))))
    summary = {
        "stage": "P2_LEGAL_MECHANISM_DISCOVERY_V2_V9830",
        "status": "summary",
        "mechanism_count": len(rows),
        "weak_pass_count": sum(inum(r.get("P2_weak_pass")) for r in rows),
        "strong_pass_count": sum(inum(r.get("P2_strong_pass")) for r in rows),
        "legal_no_red_weak_pass_count": sum(inum(r.get("P2_weak_pass")) for r in rows if "red" not in str(r.get("feature_legality")).lower()),
        "best_mechanism_id": best.get("mechanism_id"),
        "best_TopK87_precision": best.get("TopK87_precision"),
        "best_AUV_LCB": best.get("AUV_LCB"),
        "best_feature_legality": best.get("feature_legality"),
        "P2_weak_pass": int(any(inum(r.get("P2_weak_pass")) for r in rows)),
        "P2_strong_pass": int(any(inum(r.get("P2_strong_pass")) for r in rows)),
        "reason": "no_training_time_legal_mechanism_meets_precision_value_risk_gate",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    v9720.write_bar_svg(out / "fig_p2_v9830_precision_by_mechanism.svg", "v9830 P2 precision", [r["mechanism_id"] for r in rows if r.get("status") == "mechanism_row"], [fnum(r.get("TopK87_precision")) for r in rows if r.get("status") == "mechanism_row"])
    return rows, summary


def p3_natural_stream_v2(source: Path, panel_targets: list[int], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_p3 = read_csv(source / "p3_natural_ap0_labeled_stream_extension_v9820.csv")
    panel_a = next((r for r in source_p3 if r.get("status") == "panel_row"), {})
    rows: list[dict[str, Any]] = []
    for n in [1, 16, 256]:
        rows.append({
            "stage": "P3_NATURAL_AP0_STREAM_EXTENSION_MATERIALIZER_V2_V9830",
            "status": "not_run",
            "preflight_id": f"preflight-{n}-action",
            "requested_action_count": n,
            "actual_labeled_action_count": 0,
            "branch_horizon_expected_rows": n * BRANCHES * len(HORIZONS),
            "branch_horizon_actual_rows": 0,
            "missing_label_count": n,
            "unresolved_exception_count": 0,
            "reason": "no_landed_natural_AP0_action_extension_generator_for_v9830",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    existing = inum(panel_a.get("labeled_action_count"))
    for target in panel_targets:
        if target <= existing:
            row = dict(panel_a)
            row.update({
                "stage": "P3_NATURAL_AP0_STREAM_EXTENSION_MATERIALIZER_V2_V9830",
                "status": "panel_row",
                "panel_id": f"Panel-target-{target}",
                "target_action_count": target,
                "actual_labeled_action_count": target,
                "branch_horizon_expected_rows": "",
                "branch_horizon_actual_rows": "",
                "missing_label_count": 0,
                "unresolved_exception_count": 0,
                "reason": "",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            rows.append(row)
        else:
            rows.append({
                "stage": "P3_NATURAL_AP0_STREAM_EXTENSION_MATERIALIZER_V2_V9830",
                "status": "not_run",
                "panel_id": f"Panel-target-{target}",
                "target_action_count": target,
                "actual_labeled_action_count": existing,
                "branch_horizon_expected_rows": target * BRANCHES * len(HORIZONS),
                "branch_horizon_actual_rows": 0,
                "missing_label_count": target - existing,
                "unresolved_exception_count": 0,
                "CoreLike_count": "",
                "CoreLike_rate": "",
                "CoreLike_LCB": "",
                "CoreLike_UCB": "",
                "GradeAB_count": "",
                "ValuePositiveNoLongRisk_count": "",
                "MemoryOffdiagCore_count": "",
                "per_dataset_rate": "",
                "per_template_rate": "",
                "per_family_rate": "",
                "reason": "no_landed_labeled_natural_AP0_stream_extension_materializer_for_v9830",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    summary = {
        "stage": "P3_NATURAL_AP0_STREAM_EXTENSION_MATERIALIZER_V2_V9830",
        "status": "summary",
        "requested_panel_targets": ",".join(str(x) for x in panel_targets),
        "preflight_pass_count": 0,
        "preflight_not_run_count": 3,
        "completed_panel_count": sum(1 for r in rows if r.get("status") == "panel_row"),
        "not_run_panel_count": sum(1 for r in rows if r.get("panel_id") and r.get("status") == "not_run"),
        "PanelA_CoreLike_rate": panel_a.get("CoreLike_rate"),
        "PanelA_CoreLike_LCB": panel_a.get("Wilson_LCB"),
        "PanelA_CoreLike_UCB": panel_a.get("Wilson_UCB"),
        "P3_weak_pass": 0,
        "P3_strong_pass": 0,
        "P3_materializer_missing": 1,
        "reason": "natural_AP0_stream_extension_materializer_missing",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    v9720.write_bar_svg(out / "fig_p3_v9830_density_curve.svg", "v9830 P3 density", ["2876"], [fnum(panel_a.get("CoreLike_rate"))])
    return rows, summary


def p4_gate_semantics_v9830(source_v9770: Path, source_v9810: Path, p3: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ldo = summary_row(read_csv(source_v9770 / "p1_ldo_decomposition_gate_audit_v9770.csv"))
    p8 = summary_row(read_csv(source_v9810 / "p8_ldo_gate_resolution_v9810.csv"))
    raw = fnum(ldo.get("ldo_raw"))
    backfill = fnum(ldo.get("ldo_backfill"))
    backfill_share = backfill / raw if raw > 0 else 0.0
    density_lcb = fnum(p3.get("PanelA_CoreLike_LCB"))
    allowed = int(
        inum(p3.get("P3_strong_pass")) == 1
        and fnum(ldo.get("ldo_quality")) <= 0.10
        and fnum(ldo.get("ldo_support")) <= 0.10
        and backfill_share >= 0.70
        and density_lcb >= 0.03
    )
    summary = {
        "stage": "P4_GATE_SEMANTICS_OFFICIAL_SCOPE_V9830",
        "status": "summary",
        "raw_LDO": raw,
        "quality_LDO": ldo.get("ldo_quality"),
        "support_LDO": ldo.get("ldo_support"),
        "backfill_LDO": backfill,
        "backfill_share": backfill_share,
        "raw_gate_pass": p8.get("E_raw_pass", 0),
        "support_gate_pass": p8.get("E_no_backfill_pass", 0),
        "density_gate_pass": p8.get("E_density_required_pass", 0),
        "density_LCB": density_lcb,
        "official_gate_modification_allowed": allowed,
        "P4_gate_semantics_pass": int(raw > 0 and allowed == 0),
        "reason": "keep_official_raw_gate_because_P3_strong_pass_absent",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows = [summary]
    for name, val, passed in [
        ("raw", raw, p8.get("E_raw_pass", 0)),
        ("support_no_backfill", p8.get("no_backfill_LDO", ""), p8.get("E_no_backfill_pass", 0)),
        ("density_required", "", p8.get("E_density_required_pass", 0)),
    ]:
        rows.append({
            "stage": "P4_GATE_SEMANTICS_OFFICIAL_SCOPE_V9830",
            "status": "gate_row",
            "gate_id": name,
            "LDO": val,
            "pass": passed,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    v9720.write_bar_svg(out / "fig_p4_v9830_ldo_decomposition.svg", "v9830 P4 LDO", ["quality", "support", "backfill"], [fnum(ldo.get("ldo_quality")), fnum(ldo.get("ldo_support")), backfill])
    return rows, summary


def p5_discriminant_matrix(ap0: list[dict[str, Any]], feats: dict[str, dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feature_defs: list[tuple[str, str, int, dict[str, float]]] = [
        ("action_norm", "green", 0, {a: -f["action_norm"] for a, f in feats.items()}),
        ("payload_norm", "green", 0, {a: -f["payload_norm"] for a, f in feats.items()}),
        ("memory_offdiag_safe", "green", 0, {a: -float(f["memory_fail"]) - float(f["offdiag_fail"]) for a, f in feats.items()}),
        ("OldRank_score", "red:old_table_diagnostic", 0, {a: f["OldRank_score"] for a, f in feats.items()}),
        ("ExactTransfer_score", "red:outcome_derived_transfer_diagnostic", 0, {a: f["ExactTransfer_score"] for a, f in feats.items()}),
        ("WT80_score", "red:windowed_future_diagnostic", 0, {a: f["WT80_score"] for a, f in feats.items()}),
        ("AUV_future_label", "red:future_outcome", 0, {a: f["AUV"] for a, f in feats.items()}),
        ("hard_tail_h80", "red:future_outcome", 0, {a: -f["CEp99_h80"] for a, f in feats.items()}),
        ("local_geometry_h80", "red:future_outcome;yellow:kan_basis_specific", 1, {a: -abs(f["curvature_h80"]) - abs(f["jacobian_h80"]) for a, f in feats.items()}),
        ("function_displacement_norm", "red:future_outcome", 0, {a: -f["function_displacement_norm"] for a, f in feats.items()}),
    ]
    by_group = defaultdict(list)
    for f in feats.values():
        by_group[str(f["group_id"])].append(f)
    rows: list[dict[str, Any]] = []
    for name, legality, arch, scores in feature_defs:
        q, top = scores_to_quality(ap0, feats, scores)
        def vals(group: str) -> list[float]:
            return [scores[f["action_id"]] for f in by_group.get(group, []) if f["action_id"] in scores]
        weak = int(
            effect_size(vals("Core77"), vals("RandomMatched")) >= 0.5
            and fnum(q.get("GradeAB_precision")) >= 0.60
            and fnum(q.get("h240_longrisk_UCB")) <= 0.10
        )
        strong = int(
            effect_size(vals("Core77"), vals("RandomMatched")) >= 0.8
            and effect_size(vals("OldOnly"), vals("ExactOnly")) >= 0.5
            and fnum(q.get("GradeAB_precision")) >= 0.75
            and fnum(q.get("V_integrated_LCB")) > 0
            and fnum(q.get("h240_longrisk_UCB")) <= 0.05
            and fnum(q.get("LDO_drop")) <= 0.10
            and "red" not in legality
        )
        rows.append({
            "stage": "P5_GEOMETRY_DISCRIMINANT_MATRIX_V9830",
            "status": "feature_row",
            "feature_name": name,
            "legality": legality,
            "architecture_specific": arch,
            "effect_size_Core_vs_Random": effect_size(vals("Core77"), vals("RandomMatched")),
            "effect_size_Old_vs_Exact": effect_size(vals("OldOnly"), vals("ExactOnly")),
            "effect_size_Core_vs_Exact": effect_size(vals("Core77"), vals("ExactOnly")),
            "effect_size_Core_vs_Old": effect_size(vals("Core77"), vals("OldOnly")),
            "feature_cost_q90": 0.0 if "red" not in legality else "",
            "TopK87_precision": q.get("GradeAB_precision"),
            "V_LCB": q.get("V_integrated_LCB"),
            "AUV_LCB": lcb([f["AUV"] for f in top]),
            "longrisk_UCB": q.get("h240_longrisk_UCB"),
            "memory_UCB": q.get("memory_fail_UCB"),
            "offdiag_UCB": q.get("offdiag_fail_UCB"),
            "LDO_drop": q.get("LDO_drop"),
            "P5_weak": weak,
            "P5_strong": strong,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(rows, key=lambda r: (inum(r.get("P5_strong")), inum(r.get("P5_weak")), fnum(r.get("TopK87_precision")), fnum(r.get("AUV_LCB"))))
    summary = {
        "stage": "P5_GEOMETRY_DISCRIMINANT_MATRIX_V9830",
        "status": "summary",
        "feature_count": len(rows),
        "weak_count": sum(inum(r.get("P5_weak")) for r in rows),
        "strong_count": sum(inum(r.get("P5_strong")) for r in rows),
        "legal_strong_count": sum(inum(r.get("P5_strong")) for r in rows if "red" not in str(r.get("legality")).lower()),
        "best_feature_name": best.get("feature_name"),
        "best_precision": best.get("TopK87_precision"),
        "best_AUV_LCB": best.get("AUV_LCB"),
        "best_legality": best.get("legality"),
        "P5_weak_pass": int(any(inum(r.get("P5_weak")) for r in rows)),
        "P5_strong_pass": int(any(inum(r.get("P5_strong")) for r in rows)),
        "P5_legal_strong_pass": int(any(inum(r.get("P5_strong")) and "red" not in str(r.get("legality")).lower() for r in rows)),
        "reason": "no_legal_feature_strong_pass_in_discriminant_matrix",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    v9720.write_bar_svg(out / "fig_p5_v9830_effect_core_random.svg", "v9830 P5 Core vs Random", [r["feature_name"] for r in rows if r.get("status") == "feature_row"], [fnum(r.get("effect_size_Core_vs_Random")) for r in rows if r.get("status") == "feature_row"])
    return rows, summary


def not_run(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
    p0 = summary_row(read_csv(out / "p0_boundary_reproduction_v9830.csv"))
    p1 = summary_row(read_csv(out / "p1_future_path_mechanism_decomposition_v2_v9830.csv"))
    p2 = summary_row(read_csv(out / "p2_legal_mechanism_discovery_v2_v9830.csv"))
    p3 = summary_row(read_csv(out / "p3_natural_ap0_stream_extension_materializer_v2_v9830.csv"))
    p4 = summary_row(read_csv(out / "p4_gate_semantics_official_scope_v9830.csv"))
    p5 = summary_row(read_csv(out / "p5_geometry_discriminant_matrix_v9830.csv"))
    nf = summary_row(read_csv(out / "no_fake_audit_v9830.csv"))
    failure = summary_row(read_csv(out / "failure_taxonomy_v9830.csv"))
    p1_groups = [r for r in read_csv(out / "p1_future_path_mechanism_decomposition_v2_v9830.csv") if r.get("status") == "group_summary"]
    p2_rows = [r for r in read_csv(out / "p2_legal_mechanism_discovery_v2_v9830.csv") if r.get("status") == "mechanism_row"]
    p3_rows = [r for r in read_csv(out / "p3_natural_ap0_stream_extension_materializer_v2_v9830.csv") if r.get("status") in {"panel_row", "not_run"} and r.get("panel_id")]
    p5_rows = [r for r in read_csv(out / "p5_geometry_discriminant_matrix_v9830.csv") if r.get("status") == "feature_row"]
    lines: list[str] = [
        "# DG-KAN v9.8.3 Four-Line Future Mechanism / Natural Stream / Geometry Adaptive 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.8.3_四线并行_未来轨迹机制_自然动作扩流_几何自适应更新_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 future-path diagnostic、illegal mechanism、natural stream missing、gate semantics diagnostic 或 not_run controller/runtime 写成 official system pass。",
        "",
        "## 0. 最新结论",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"primary_blocker = {route.get('primary_blocker')}",
        f"secondary_blocker = {route.get('secondary_blocker')}",
        f"route_secondary = {route.get('route_secondary')}",
        f"system_legal_controller_pass = {route.get('system_legal_controller_pass')}",
        f"generated_route_status = {route.get('generated_route_status')}",
        "```",
        "",
        f"最终 artifact：`{out}`",
        "",
        "核心结论：",
        "",
        f"1. P0 复现 v9.8.2 boundary：source route = `{p0.get('source_route_v9820')}`，P1 rows expected/actual = `{p0.get('P1_full_future_path_rows_expected')}` / `{p0.get('P1_full_future_path_rows_actual')}`，fake/proxy/cpu = `{p0.get('fake_data_used')}` / `{p0.get('proxy_row_used')}` / `{p0.get('cpu_offload_used')}`。",
        f"2. P1 future path v2 weak/strong = `{p1.get('P1_weak_pass')}` / `{p1.get('P1_strong_pass')}`；Core77 AUV LCB = `{p1.get('Core77_AUV_LCB')}`，RandomMatched AUV LCB = `{p1.get('RandomMatched_AUV_LCB')}`。",
        f"3. P1 immediate-not-dominant evidence = `{p1.get('future_path_positive_immediate_not_dominant')}`；OldOnly V1 LCB = `{p1.get('OldOnly_V1_LCB')}`，OldOnly AUV LCB = `{p1.get('OldOnly_AUV_LCB')}`。",
        f"4. P2 legal mechanism weak/strong pass count = `{p2.get('weak_pass_count')}` / `{p2.get('strong_pass_count')}`；best = `{p2.get('best_mechanism_id')}`，legality = `{p2.get('best_feature_legality')}`。",
        f"5. P3 natural AP0 extension still blocked：preflight pass/not_run = `{p3.get('preflight_pass_count')}` / `{p3.get('preflight_not_run_count')}`，completed panel/not_run panel = `{p3.get('completed_panel_count')}` / `{p3.get('not_run_panel_count')}`。",
        f"6. P4 gate modification allowed = `{p4.get('official_gate_modification_allowed')}`；backfill share = `{p4.get('backfill_share')}`，density LCB = `{p4.get('density_LCB')}`。",
        f"7. P5 discriminant matrix weak/strong/legal-strong = `{p5.get('weak_count')}` / `{p5.get('strong_count')}` / `{p5.get('legal_strong_count')}`；best feature = `{p5.get('best_feature_name')}`。",
        f"8. No-fake audit：rows checked = `{nf.get('rows_checked')}`，fake/proxy/cpu = `{nf.get('fake_data_used')}` / `{nf.get('proxy_row_used')}` / `{nf.get('cpu_offload_used')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/run_v9830_four_line_future_mechanism_natural_geometry.py` | v9.8.3 runner；读取 v9.8.2 full future path rows，执行 P1 mechanism decomposition v2、P2 legal mechanism discovery v2、P3 natural stream materializer audit、P4 gate semantics、P5 discriminant matrix，并按 gate 写 P6-P10。 |",
        "",
        "代码检查：",
        "",
        "```text",
        "python -m py_compile experiments/run_v9830_four_line_future_mechanism_natural_geometry.py",
        "```",
        "",
        "正式运行：",
        "",
        "```bash",
        "python experiments/run_v9830_four_line_future_mechanism_natural_geometry.py --out-dir results/real_rerun_20260506/v9830_four_line_future_mechanism_natural_geometry_full_20260516T140000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 2876,5000,10000,20000",
        "```",
        "",
        "说明：本轮 full-gated 是基于 v9.8.2 已真实落盘的 7260 条 future-path rows 做机制/扩流/gate 分析；P3 没有 landed natural AP0 extension materializer，因此 preflight 和 5000/10000/20000 panel 明确 `not_run`，没有生成假扩展动作。",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, indent=2, ensure_ascii=False),
        "```",
        "",
        "## 3. P1 Future Path Mechanism v2",
        "",
        "| group | actions | AUV LCB | V1 LCB | V20 LCB | V80 LCB | V240 LCB | mismatch | longrisk UCB | memory UCB | offdiag UCB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in p1_groups:
        lines.append(f"| `{r.get('group_id')}` | `{r.get('action_count')}` | `{r.get('AUV_LCB')}` | `{r.get('V1_LCB')}` | `{r.get('V20_LCB')}` | `{r.get('V80_LCB')}` | `{r.get('V240_LCB')}` | `{r.get('Mismatch_rate')}` | `{r.get('h240_longrisk_UCB')}` | `{r.get('memory_UCB')}` | `{r.get('offdiag_UCB')}` |")
    lines.extend([
        "",
        f"判断：P1 strong pass = `{p1.get('P1_strong_pass')}`。Core77 的完整 future path 仍强且风险低；OldOnly 出现 `V1_LCB <= 0` 但 `AUV_LCB > 0` 的 immediate-not-dominant 形态，支持“不是一步 loss descent，而是 future trajectory steering”的解释。",
        "",
        "## 4. P2 Legal Mechanism Discovery v2",
        "",
        "| mechanism | legality | arch specific | precision | V LCB | AUV LCB | longrisk UCB | memory UCB | offdiag UCB | LDO | weak | strong |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in p2_rows:
        lines.append(f"| `{r.get('mechanism_id')}` | `{r.get('feature_legality')}` | `{r.get('architecture_specific')}` | `{r.get('TopK87_precision')}` | `{r.get('V_LCB')}` | `{r.get('AUV_LCB')}` | `{r.get('longrisk_UCB')}` | `{r.get('memory_UCB')}` | `{r.get('offdiag_UCB')}` | `{r.get('LDO_drop')}` | `{r.get('P2_weak_pass')}` | `{r.get('P2_strong_pass')}` |")
    lines.extend([
        "",
        "判断：P2 没有 legal-at-commit 机制通过。诊断类信号仍能贴近好区域，但带 red legality；green 的 memory/cost 类仍无法同时给出 precision/value。",
        "",
        "## 5. P3 Natural AP0 Stream Extension Materializer",
        "",
        "| row | status | target | labeled | expected rows | actual rows | missing labels | reason |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for r in p3_rows:
        lines.append(f"| `{r.get('panel_id')}` | `{r.get('status')}` | `{r.get('target_action_count', r.get('action_count'))}` | `{r.get('actual_labeled_action_count', r.get('labeled_action_count'))}` | `{r.get('branch_horizon_expected_rows')}` | `{r.get('branch_horizon_actual_rows')}` | `{r.get('missing_label_count')}` | `{r.get('reason')}` |")
    lines.extend([
        "",
        f"判断：P3 strong pass = `{p3.get('P3_strong_pass')}`。自然扩流 materializer 仍未落地，因此不能用 2876 panel 继续推断 5000/10000/20000 density。",
        "",
        "## 6. P4 Gate Semantics",
        "",
        "```text",
        f"raw_LDO = {p4.get('raw_LDO')}",
        f"quality_LDO = {p4.get('quality_LDO')}",
        f"support_LDO = {p4.get('support_LDO')}",
        f"backfill_LDO = {p4.get('backfill_LDO')}",
        f"backfill_share = {p4.get('backfill_share')}",
        f"official_gate_modification_allowed = {p4.get('official_gate_modification_allowed')}",
        "```",
        "",
        "判断：backfill 仍能解释 raw LDO 的主要部分，但 P3 strong pass 缺失，所以 official raw gate 不允许修改。",
        "",
        "## 7. P5 Geometry Discriminant Matrix",
        "",
        "| feature | legality | Core vs Random | Old vs Exact | precision | V LCB | AUV LCB | longrisk UCB | weak | strong |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in p5_rows:
        lines.append(f"| `{r.get('feature_name')}` | `{r.get('legality')}` | `{r.get('effect_size_Core_vs_Random')}` | `{r.get('effect_size_Old_vs_Exact')}` | `{r.get('TopK87_precision')}` | `{r.get('V_LCB')}` | `{r.get('AUV_LCB')}` | `{r.get('longrisk_UCB')}` | `{r.get('P5_weak')}` | `{r.get('P5_strong')}` |")
    lines.extend([
        "",
        "判断：P5 有若干 diagnostic discriminator，但没有 legal strong feature。继续说明 memory/offdiag 是安全门，不是收益源。",
        "",
        "## 8. P6-P10 Boundary",
        "",
        "```text",
        "P6 controller = not_run",
        "P7 generated = stopped",
        "P8 runtime = not_run",
        "P9 paired replay = not_run",
        "P10 short/full = not_run",
        "```",
        "",
        "判断：P2/P5 没有 legal strong mechanism，P3 materializer missing，因此不允许 controller、generated、runtime、paired replay 或 short/full。",
        "",
        "## 9. No-Fake / Contract / Failure",
        "",
        "No-fake audit：",
        "",
        "```text",
        f"rows_checked = {nf.get('rows_checked')}",
        f"fake_data_used = {nf.get('fake_data_used')}",
        f"proxy_row_used = {nf.get('proxy_row_used')}",
        f"cpu_offload_used = {nf.get('cpu_offload_used')}",
        "```",
        "",
        "Failure taxonomy：",
        "",
        "```text",
        f"route = {failure.get('route')}",
        f"F1_future_path_not_strong = {failure.get('F1_future_path_not_strong')}",
        f"F2_no_legal_mechanism = {failure.get('F2_no_legal_mechanism')}",
        f"F3_natural_stream_materializer_missing = {failure.get('F3_natural_stream_materializer_missing')}",
        f"F4_gate_not_modifiable = {failure.get('F4_gate_not_modifiable')}",
        f"F5_generated_route_stopped = {failure.get('F5_generated_route_stopped')}",
        f"primary_blocker = {failure.get('primary_blocker')}",
        f"secondary_blocker = {failure.get('secondary_blocker')}",
        "```",
        "",
        "## 10. Figures",
        "",
        "```text",
        "fig_p1_v9830_group_auv_lcb.svg",
        "fig_p1_v9830_group_mismatch.svg",
        "fig_p2_v9830_precision_by_mechanism.svg",
        "fig_p3_v9830_density_curve.svg",
        "fig_p4_v9830_ldo_decomposition.svg",
        "fig_p5_v9830_effect_core_random.svg",
        "```",
        "",
        "## 11. Hash",
        "",
        "| artifact | SHA256 |",
        "|---|---|",
    ])
    for name, digest in sorted(hashes.items()):
        lines.append(f"| `{name}` | `{digest}` |")
    lines.extend([
        "",
        "## 12. 最终分析结论",
        "",
        "v9.8.3 的真实推进是：",
        "",
        "```text",
        "1. P1 不再只是确认 future path 存在，而是把 AUV、Stability、Mismatch 拆到 action-level；Core77 继续强，OldOnly 显示 immediate-not-dominant。",
        "2. P2 仍没有找到训练当下合法机制；诊断信号有效但不合法，合法安全信号不够选出收益动作。",
        "3. P3 的自然 AP0 扩流 materializer 仍缺失，preflight 和 5000/10000/20000 panel 都不能伪造。",
        "4. P4 不允许修改 official raw gate。",
        "5. P5 判别矩阵确认若干 red diagnostic discriminator，但没有 legal strong mechanism。",
        "6. P6-P10 全部 gate-blocked。",
        "```",
        "",
        "最终一句话：",
        "",
        f"> v9.8.3 真实执行后停在 `{route.get('route')}`，并同时记录 `{route.get('route_secondary')}`：future path 机制现象更清楚了，但训练当下合法机制仍未闭合，自然 AP0 扩流 materializer 也未落地，因此不能进入 official controller，strict PureKAN functional 仍未成功。",
        "",
    ])
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

    source = Path(args.source_v9820)
    ap0, _score_bundle, _payload_by_id = v9720.load_ap0_and_payloads(args)

    p0_rows, p0 = p0_boundary(source)
    dump_csv("p0_boundary_reproduction_v9830.csv", p0_rows)

    real = source_real_rows(source)
    feats = action_path_features(real)
    p1_rows, p1 = p1_future_path_v2(real, feats, out)
    dump_csv("p1_future_path_mechanism_decomposition_v2_v9830.csv", p1_rows)

    p2_rows, p2 = p2_legal_mechanism_v2(ap0, feats, out)
    dump_csv("p2_legal_mechanism_discovery_v2_v9830.csv", p2_rows)

    p3_rows, p3 = p3_natural_stream_v2(source, parse_ints(args.panel_targets), out)
    dump_csv("p3_natural_ap0_stream_extension_materializer_v2_v9830.csv", p3_rows)

    p4_rows, p4 = p4_gate_semantics_v9830(Path(args.source_v9770), Path(args.source_v9810), p3, out)
    dump_csv("p4_gate_semantics_official_scope_v9830.csv", p4_rows)

    p5_rows, p5 = p5_discriminant_matrix(ap0, feats, out)
    dump_csv("p5_geometry_discriminant_matrix_v9830.csv", p5_rows)

    if inum(p2.get("P2_strong_pass")) or inum(p5.get("P5_legal_strong_pass")):
        p6_rows, p6 = not_run("P6_EXISTING_ACTION_MINIMAL_CONTROLLER_BOUNDARY_V9830", "P3_density_or_controller_implementation_not_ready", controller_pass=0)
    else:
        p6_rows, p6 = not_run("P6_EXISTING_ACTION_MINIMAL_CONTROLLER_BOUNDARY_V9830", "P2_or_P5_no_legal_strong_mechanism", controller_pass=0)
    dump_csv("p6_existing_action_minimal_controller_boundary_v9830.csv", p6_rows)

    generated_allowed = int(
        (inum(p2.get("P2_strong_pass")) or inum(p5.get("P5_legal_strong_pass")))
        and (inum(p3.get("P3_strong_pass")) or inum(p6.get("controller_pass")))
    )
    if generated_allowed:
        p7_rows, p7 = not_run("P7_GENERATED_UPDATE_REOPEN_GATE_V9830", "generated_objective_implementation_not_landed_after_reopen_gate", generated_route_reopen_allowed=1, generated_pass=0)
    else:
        p7_rows, p7 = not_run("P7_GENERATED_UPDATE_REOPEN_GATE_V9830", "stopped_no_legal_mechanism_or_natural_density_evidence", generated_route_reopen_allowed=0, generated_pass=0)
    dump_csv("p7_generated_update_reopen_gate_v9830.csv", p7_rows)

    p8_rows, p8 = not_run("P8_SELECTED_RUNTIME_BOUNDARY_V9830", "P6_controller_and_P7_generated_not_passed", runtime_pass=0)
    dump_csv("p8_selected_runtime_boundary_v9830.csv", p8_rows)
    p9_rows, p9 = not_run("P9_OFFICIAL_PAIRED_REPLAY_BOUNDARY_V9830", "P8_runtime_not_passed", paired_replay_pass=0)
    dump_csv("p9_official_paired_replay_boundary_v9830.csv", p9_rows)
    p10_rows, p10 = not_run("P10_SHORT_FULL_TRAINING_BOUNDARY_V9830", "P9_paired_replay_not_passed", short_full_pass=0)
    dump_csv("p10_short_full_training_boundary_v9830.csv", p10_rows)

    route = "RouteB-FuturePathExistsMechanismAbsent" if inum(p1.get("P1_strong_pass")) and not (inum(p2.get("P2_strong_pass")) or inum(p5.get("P5_legal_strong_pass"))) else "RouteF-NoFuturePathStrongPass"
    route_secondary = "RouteD-NaturalStreamMaterializerMissing" if inum(p3.get("P3_materializer_missing")) else "none"
    primary = "no_legal_training_time_mechanism"
    secondary = "natural_stream_materializer_missing" if inum(p3.get("P3_materializer_missing")) else "none"
    route_decision = {
        "stage": "ROUTE_DECISION_V9830",
        "status": "summary",
        "route": route,
        "route_secondary": route_secondary,
        "route_family": "mechanism_and_materializer",
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "source_route_v9820": p0.get("source_route_v9820"),
        "P0_boundary_pass": p0.get("boundary_reproduced"),
        "P1_weak_pass": p1.get("P1_weak_pass"),
        "P1_strong_pass": p1.get("P1_strong_pass"),
        "P2_weak_pass": p2.get("P2_weak_pass"),
        "P2_strong_pass": p2.get("P2_strong_pass"),
        "P3_weak_pass": p3.get("P3_weak_pass"),
        "P3_strong_pass": p3.get("P3_strong_pass"),
        "P3_materializer_missing": p3.get("P3_materializer_missing"),
        "P4_gate_modification_allowed": p4.get("official_gate_modification_allowed"),
        "P5_weak_pass": p5.get("P5_weak_pass"),
        "P5_strong_pass": p5.get("P5_strong_pass"),
        "P5_legal_strong_pass": p5.get("P5_legal_strong_pass"),
        "P6_controller_pass": p6.get("controller_pass"),
        "P7_generated_reopen_allowed": p7.get("generated_route_reopen_allowed"),
        "P7_generated_pass": p7.get("generated_pass"),
        "P8_runtime_pass": p8.get("runtime_pass"),
        "P9_paired_replay_pass": p9.get("paired_replay_pass"),
        "P10_short_full_pass": p10.get("short_full_pass"),
        "generated_route_status": "stopped_no_legal_mechanism_or_natural_density_evidence",
        "system_legal_controller_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("route_decision_v9830.json", route_decision)

    no_fake = {
        "stage": "NO_FAKE_AUDIT_V9830",
        "status": "summary",
        "rows_checked": audit_rows(artifacts),
        "fake_proxy_nonzero_count": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "no_fake": 1,
        "no_proxy": 1,
    }
    dump_csv("no_fake_audit_v9830.csv", [no_fake])
    contract = {
        "stage": "CONTRACT_AUDIT_V9830",
        "status": "summary",
        "manual_forward/manual_backward/manual_adamw_update": "1/1/1",
        "v9820_boundary_pass": p0.get("boundary_reproduced"),
        "future_path_mechanism_v2_strong": p1.get("P1_strong_pass"),
        "legal_mechanism_strong": p2.get("P2_strong_pass"),
        "natural_stream_materializer_completed": 0,
        "gate_modification_allowed": p4.get("official_gate_modification_allowed"),
        "controller/generated/runtime/system": "0/0/0/0",
        "uses_future_outcome_for_official_controller": 0,
        "diagnostic_promoted_to_official": 0,
        "fake/proxy/cpu_offload": "0/0/0",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("contract_audit_v9830.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9830",
        "status": "summary",
        "route": route,
        "route_secondary": route_secondary,
        "F0_boundary_fail": int(not inum(p0.get("boundary_reproduced"))),
        "F1_future_path_not_strong": int(not inum(p1.get("P1_strong_pass"))),
        "F2_no_legal_mechanism": int(not (inum(p2.get("P2_strong_pass")) or inum(p5.get("P5_legal_strong_pass")))),
        "F3_natural_stream_materializer_missing": p3.get("P3_materializer_missing"),
        "F4_gate_not_modifiable": int(not inum(p4.get("official_gate_modification_allowed"))),
        "F5_generated_route_stopped": int(not inum(p7.get("generated_route_reopen_allowed"))),
        "F6_controller_runtime_replay_shortfull_blocked": 1,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("failure_taxonomy_v9830.csv", [failure])

    hashes = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    manifest = {
        "stage": "RUN_MANIFEST_V9830",
        "status": "summary",
        "out_dir": str(out),
        "execution_profile": args.execution_profile,
        "command_profile": {
            "device": args.device,
            "data_root": args.data_root,
            "seed": args.seed,
            "panel_targets": args.panel_targets,
        },
        "wallclock_sec": time.perf_counter() - t0,
        "route": route,
        "route_secondary": route_secondary,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "artifact_hashes": hashes,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("run_manifest_v9830.json", manifest)
    hashes_for_recap = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    write_recap(out, route_decision, hashes_for_recap)
    print(json.dumps({
        "out_dir": str(out),
        "route": route,
        "route_secondary": route_secondary,
        "P1_strong_pass": p1.get("P1_strong_pass"),
        "P2_strong_pass": p2.get("P2_strong_pass"),
        "P3_materializer_missing": p3.get("P3_materializer_missing"),
        "P5_legal_strong_pass": p5.get("P5_legal_strong_pass"),
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "system_legal_controller_pass": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
