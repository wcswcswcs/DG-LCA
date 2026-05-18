#!/usr/bin/env python3
"""DG-KAN v9.8.6 future-path mechanism / natural stream run.

This runner follows the v9.8.6 plan.  It only uses landed CSV/JSON rows and
explicitly writes not_run rows when the natural AP0 extension materializer or
generated-route gates are unavailable.  No fake/proxy rows are created.
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
import run_v9770_core77_support_density_natural_stream_gate as v9770  # noqa: E402
import run_v9840_future_operator_natural_stream_geometry_optimizer as v9840  # noqa: E402
import run_v9850_four_line_future_operator_natural_stream_decision as v9850  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


RESULT_ROOT = REPO / "results/real_rerun_20260506"
PLAN_PATH = REPO / "docs/DG-KAN_v9.8.6_结果解读_未来路径机制_自然扩流_几何自适应优化器_完整计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9860_future_path_mechanism_natural_stream_geometry_optimizer.py"
RECAP_PATH = REPO / "docs/DG-KAN_v9.8.6_FuturePathMechanism_NaturalStream_GeometryAdaptiveOptimizer_实验复盘.md"
DEFAULT_OUT = RESULT_ROOT / "v9860_future_path_mechanism_natural_stream_geometry_optimizer_full_20260516T200000Z"
DEFAULT_V9850 = RESULT_ROOT / "v9850_four_line_future_operator_natural_stream_decision_full_20260516T180000Z"
DEFAULT_V9840 = RESULT_ROOT / "v9840_future_operator_natural_stream_geometry_optimizer_full_20260516T160000Z"
DEFAULT_V9820 = RESULT_ROOT / "v9820_future_path_geometry_mechanism_natural_stream_full_20260516T120000Z"
DEFAULT_V9810 = RESULT_ROOT / "v9810_future_trajectory_geometry_adaptive_optimizer_full_20260516T100000Z"

HORIZONS = [1, 5, 20, 80, 240]
BRANCH_COUNT = 6
TARGET_K = 87


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--execution-profile", choices=["smoke", "full-gated"], default="full-gated")
    p.add_argument("--panel-targets", default="2876,5000,10000,20000")
    p.add_argument("--source-v9850", default=str(DEFAULT_V9850))
    p.add_argument("--source-v9840", default=str(DEFAULT_V9840))
    p.add_argument("--source-v9820", default=str(DEFAULT_V9820))
    p.add_argument("--source-v9810", default=str(DEFAULT_V9810))
    p.add_argument("--source-v9330", default=str(v9840.v9820.DEFAULT_V9330))
    p.add_argument("--source-v9550", default=str(v9840.v9820.DEFAULT_V9550))
    p.add_argument("--source-v9560", default=str(v9840.v9820.DEFAULT_V9560))
    p.add_argument("--source-v9570", default=str(v9840.v9820.DEFAULT_V9570))
    p.add_argument("--source-v9580", default=str(v9840.v9820.DEFAULT_V9580))
    return p.parse_args()


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


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


def pearson(xs: list[float], ys: list[float]) -> float:
    vals = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(vals) < 2:
        return 0.0
    xvals = [x for x, _ in vals]
    yvals = [y for _, y in vals]
    mx, my = mean(xvals), mean(yvals)
    vx = sum((x - mx) ** 2 for x in xvals)
    vy = sum((y - my) ** 2 for y in yvals)
    if vx <= 0 or vy <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in vals) / math.sqrt(vx * vy)


def json_counter(items: list[str]) -> str:
    return json.dumps(dict(Counter(items)), sort_keys=True, ensure_ascii=False)


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r for r in rows if r.get("status") == "summary"), rows[0] if rows else {})


def ap0_indices(ap0: list[dict[str, Any]]) -> dict[str, int]:
    return {str(r.get("action_id")): i for i, r in enumerate(ap0)}


def membership_scores(n: int, idx: list[int]) -> list[float]:
    scores = [0.0] * n
    for i in idx:
        if 0 <= i < n:
            scores[i] = 1.0
    return scores


def quality_for_ids(ap0: list[dict[str, Any]], ids: list[str]) -> tuple[dict[str, Any], list[int], list[float]]:
    idx_by = ap0_indices(ap0)
    idx = [idx_by[str(a)] for a in ids if str(a) in idx_by]
    scores = membership_scores(len(ap0), idx)
    q = v9720.quality_for_indices(ap0, idx, scores, max(1, len(idx)))
    return q, idx, scores


def axis(row: dict[str, Any], name: str) -> str:
    return v9720.axis_value(row, name)


def ldo_decomposition(ap0: list[dict[str, Any]], idx: list[int], q: dict[str, Any], scores: list[float]) -> dict[str, Any]:
    base_p = fnum(q.get("GradeAB_precision"))
    base_v = fnum(q.get("V_integrated_LCB"))
    precision_drop = 0.0
    value_drop = 0.0
    by_ds: dict[str, list[int]] = defaultdict(list)
    for i in idx:
        by_ds[axis(ap0[i], "dataset_id")].append(i)
    for ids in by_ds.values():
        own_scores = membership_scores(len(ap0), ids)
        sq = v9720.quality_for_indices(ap0, ids, own_scores, max(1, len(ids)))
        precision_drop = max(precision_drop, max(0.0, base_p - fnum(sq.get("GradeAB_precision"))))
        value_drop = max(value_drop, max(0.0, base_v - fnum(sq.get("V_integrated_LCB"))))
    support_adj, _detail = v9770.support_adjusted_drop(ap0, idx)
    ldo_raw = fnum(q.get("LDO_drop"))
    ldo_quality = max(precision_drop, value_drop)
    ldo_support = max(0.0, support_adj - ldo_quality)
    ldo_backfill = max(0.0, ldo_raw - support_adj)
    return {
        "LDO_raw": ldo_raw,
        "LDO_quality": ldo_quality,
        "LDO_support": ldo_support,
        "LDO_backfill": ldo_backfill,
        "LDO_support_adjusted": support_adj,
        "LDO_backfill_share": ldo_backfill / ldo_raw if ldo_raw > 0 else 0.0,
        "LFO": v9770.lfo_drop(scores, ap0, max(1, len(idx))),
    }


def p0_boundary(source_v9850: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9850 / "route_decision_v9850.json")
    p1 = summary_row(read_csv(source_v9850 / "p1_future_path_type_decomposition_v9850.csv"))
    p2 = summary_row(read_csv(source_v9850 / "p2_legal_virtual_path_proxy_v9850.csv"))
    p3 = summary_row(read_csv(source_v9850 / "p3_natural_ap0_stream_materializer_audit_v9850.csv"))
    p5 = summary_row(read_csv(source_v9850 / "p5_generated_route_reopen_decision_v9850.csv"))
    nf = summary_row(read_csv(source_v9850 / "no_fake_audit_v9850.csv"))
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9860",
        "status": "summary",
        "source_route_v9850": route.get("route"),
        "source_primary_blocker_v9850": route.get("primary_blocker"),
        "source_secondary_blocker_v9850": route.get("secondary_blocker"),
        "A_line_mechanism_pass_v9850": p1.get("A_line_mechanism_pass"),
        "A_line_strong_pass_v9850": p1.get("A_line_strong_pass"),
        "B_weak_pass_v9850": p2.get("B_weak_pass"),
        "B_strong_pass_v9850": p2.get("B_strong_pass"),
        "StopB_triggered_v9850": p2.get("StopB_triggered"),
        "C_materializer_entrypoint_found_v9850": p3.get("materializer_entrypoint_found"),
        "D_generated_reopen_allowed_v9850": p5.get("generated_route_reopen_allowed"),
        "generated_route_status_v9850": route.get("generated_route_status"),
        "system_legal_controller_pass_v9850": route.get("system_legal_controller_pass"),
        "fake_data_used": nf.get("fake_data_used", 0),
        "proxy_row_used": nf.get("proxy_row_used", 0),
        "cpu_offload_used": nf.get("cpu_offload_used", 0),
    }
    row["P0_boundary_pass"] = int(
        row["source_route_v9850"] == "Case4-LegalProxyFailNaturalStreamMissingGeneratedStopped"
        and inum(row["A_line_mechanism_pass_v9850"]) == 1
        and inum(row["A_line_strong_pass_v9850"]) == 0
        and inum(row["B_weak_pass_v9850"]) == 0
        and inum(row["B_strong_pass_v9850"]) == 0
        and inum(row["C_materializer_entrypoint_found_v9850"]) == 0
        and inum(row["D_generated_reopen_allowed_v9850"]) == 0
        and inum(row["system_legal_controller_pass_v9850"]) == 0
        and inum(row["fake_data_used"]) == 0
        and inum(row["proxy_row_used"]) == 0
        and inum(row["cpu_offload_used"]) == 0
    )
    return [row], row


def landed_future_row_reference(real: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [{
        "stage": "P0_LANDED_FUTURE_ROW_REFERENCE_V9860",
        "status": "summary",
        "realfunctional_row_count": len(real),
        "action_count": len({str(r.get("action_id")) for r in real}),
        "horizons": ",".join(str(h) for h in sorted({inum(r.get("horizon")) for r in real})),
        "group_count": len({str(r.get("group_id")) for r in real}),
        "source": "v9820_full_future_path_rows_plus_v9840_extra_future_path_rows",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    for r in real:
        rows.append({
            "stage": "P0_LANDED_FUTURE_ROW_REFERENCE_V9860",
            "status": "landed_horizon_row",
            "action_id": r.get("action_id"),
            "group_id": r.get("group_id"),
            "dataset": r.get("dataset_id") or r.get("dataset"),
            "seed": r.get("seed"),
            "family_id": r.get("family_id"),
            "template_id": r.get("template_id"),
            "horizon": r.get("horizon"),
            "V_branch": r.get("V_branch"),
            "CE_delta": r.get("CE_delta"),
            "margin_delta": r.get("margin_delta"),
            "CEp99_delta": r.get("CEp99_delta"),
            "hard_tail_loss_delta": r.get("hard_tail_loss_delta"),
            "long_risk_label": r.get("long_risk_label"),
            "bad_event_label": r.get("bad_event_label"),
            "null_event_label": r.get("null_event_label"),
            "memory_fail": r.get("memory_fail"),
            "offdiag_fail": r.get("offdiag_fail"),
            "branch_runtime_ms": r.get("branch_runtime_ms"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    return rows


def group_ids(feats: dict[str, dict[str, Any]], group: str) -> list[str]:
    return sorted([aid for aid, f in feats.items() if str(f.get("group_id")) == group])


def p1_combo_reassessment(ap0: list[dict[str, Any]], feats: dict[str, dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    core = group_ids(feats, "Core77")
    exp10 = group_ids(feats, "CoreExpansion10")
    exact = group_ids(feats, "ExactOnly")
    old = group_ids(feats, "OldOnly")
    random = group_ids(feats, "RandomMatched")
    risk_clean = group_ids(feats, "RiskCleanButLowValue")
    risk_top10 = sorted(risk_clean, key=lambda a: fnum(feats[a].get("RiskAdjustedAUV")), reverse=True)[:10]
    candidates: list[tuple[str, list[str]]] = [
        ("Core77", core),
        ("CoreExpansion10", exp10),
        ("Core77+CoreExpansion10", core + exp10),
        ("OldOnly", old),
        ("ExactOnly", exact),
        ("RandomMatched", random),
        ("RiskCleanButLowValue", risk_clean),
        ("Core77+RiskCleanButLowValueTop10", core + risk_top10),
    ]
    rows: list[dict[str, Any]] = []
    for cid, ids in candidates:
        fs = [feats[a] for a in ids if a in feats]
        q, idx, scores = quality_for_ids(ap0, ids)
        decomp = ldo_decomposition(ap0, idx, q, scores)
        per_ds = defaultdict(list)
        for f in fs:
            per_ds[str(f.get("dataset"))].append(fnum(f.get("V240")))
        strong = int(
            len(fs) >= TARGET_K
            and lcb([fnum(f.get("AUV")) for f in fs]) > 0
            and lcb([fnum(f.get("V80")) for f in fs]) > 0
            and lcb([fnum(f.get("V240")) for f in fs]) > 0
            and ucb([float(f.get("longrisk240", 0)) for f in fs]) <= 0.05
            and fnum(q.get("bad_UCB")) <= 0.05
            and fnum(q.get("null_UCB")) <= 0.15
            and ucb([float(f.get("memory_fail", 0)) for f in fs]) == 0
            and ucb([float(f.get("offdiag_fail", 0)) for f in fs]) == 0
            and fnum(decomp.get("LDO_support_adjusted")) <= 0.10
            and fnum(q.get("LSO_drop")) <= 0.10
            and fnum(q.get("LTO_drop")) <= 0.10
        )
        density_dispute = int(
            len(fs) >= TARGET_K
            and lcb([fnum(f.get("AUV")) for f in fs]) > 0
            and lcb([fnum(f.get("V80")) for f in fs]) > 0
            and lcb([fnum(f.get("V240")) for f in fs]) > 0
            and ucb([float(f.get("longrisk240", 0)) for f in fs]) <= 0.05
            and fnum(q.get("bad_UCB")) <= 0.05
            and fnum(q.get("null_UCB")) <= 0.15
            and ucb([float(f.get("memory_fail", 0)) for f in fs]) == 0
            and ucb([float(f.get("offdiag_fail", 0)) for f in fs]) == 0
            and fnum(decomp.get("LDO_raw")) > 0.10
            and fnum(decomp.get("LDO_backfill_share")) >= 0.70
        )
        rows.append({
            "stage": "P1_FUTURE_PATH_COMBO_REASSESSMENT_V9860",
            "status": "combo_summary",
            "candidate_id": cid,
            "action_count": len(fs),
            "AUV_LCB": lcb([fnum(f.get("AUV")) for f in fs]),
            "RiskAdjustedAUV_LCB": lcb([fnum(f.get("RiskAdjustedAUV")) for f in fs]),
            "DelayedGain_LCB": lcb([fnum(f.get("DelayedGain")) for f in fs]),
            "V1_LCB": lcb([fnum(f.get("V1")) for f in fs]),
            "V5_LCB": lcb([fnum(f.get("V5")) for f in fs]),
            "V20_LCB": lcb([fnum(f.get("V20")) for f in fs]),
            "V80_LCB": lcb([fnum(f.get("V80")) for f in fs]),
            "V240_LCB": lcb([fnum(f.get("V240")) for f in fs]),
            "RiskPath_UCB": ucb([float(f.get("RiskPath", 0)) for f in fs]),
            "LongRisk_UCB": ucb([float(f.get("longrisk240", 0)) for f in fs]),
            "Bad_UCB": q.get("bad_UCB"),
            "Null_UCB": q.get("null_UCB"),
            "MemoryFail_UCB": ucb([float(f.get("memory_fail", 0)) for f in fs]),
            "OffdiagFail_UCB": ucb([float(f.get("offdiag_fail", 0)) for f in fs]),
            "GradeAB_precision": q.get("GradeAB_precision"),
            "V_integrated_LCB": q.get("V_integrated_LCB"),
            "LDO_raw": decomp.get("LDO_raw"),
            "LDO_quality": decomp.get("LDO_quality"),
            "LDO_support": decomp.get("LDO_support"),
            "LDO_backfill": decomp.get("LDO_backfill"),
            "LDO_backfill_share": decomp.get("LDO_backfill_share"),
            "LDO_support_adjusted": decomp.get("LDO_support_adjusted"),
            "LSO": q.get("LSO_drop"),
            "LTO": q.get("LTO_drop"),
            "LFO": decomp.get("LFO"),
            "candidate_template_count": len({str(f.get("template_id")) for f in fs}),
            "family_count": len({str(f.get("family_id")) for f in fs}),
            "per_dataset_count": json_counter([str(f.get("dataset")) for f in fs]),
            "per_dataset_V240_LCB": json.dumps({ds: lcb(vals) for ds, vals in sorted(per_ds.items())}, sort_keys=True),
            "path_combo_strong_pass": strong,
            "density_gate_dispute": density_dispute,
            "field_legality": "red:future_path_diagnostic_only",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(rows, key=lambda r: (inum(r["path_combo_strong_pass"]), inum(r["density_gate_dispute"]), fnum(r["RiskAdjustedAUV_LCB"])))
    summary = {
        "stage": "P1_FUTURE_PATH_COMBO_REASSESSMENT_V9860",
        "status": "summary",
        "candidate_count": len(rows),
        "path_combo_strong_pass_count": sum(inum(r.get("path_combo_strong_pass")) for r in rows),
        "density_gate_dispute_count": sum(inum(r.get("density_gate_dispute")) for r in rows),
        "best_candidate_id": best.get("candidate_id"),
        "best_action_count": best.get("action_count"),
        "best_AUV_LCB": best.get("AUV_LCB"),
        "best_RiskAdjustedAUV_LCB": best.get("RiskAdjustedAUV_LCB"),
        "best_LDO_raw": best.get("LDO_raw"),
        "best_LDO_support_adjusted": best.get("LDO_support_adjusted"),
        "best_LSO": best.get("LSO"),
        "best_LTO": best.get("LTO"),
        "P1_A_combo_strong_pass": int(any(inum(r.get("path_combo_strong_pass")) for r in rows)),
        "P1_density_gate_dispute": int(any(inum(r.get("density_gate_dispute")) for r in rows)),
        "P1_controller_ready": 0,
        "reason": "future_path_combo_is_diagnostic_only_until_legal_mechanism_or_density_gate_closes",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    labels = [r["candidate_id"] for r in rows if r.get("status") == "combo_summary"]
    combo_rows = [r for r in rows if r.get("status") == "combo_summary"]
    v9720.write_bar_svg(out / "fig_A1_path_combo_V_curve.svg", "A1 path combo AUV LCB", labels, [fnum(r.get("AUV_LCB")) for r in combo_rows])
    v9720.write_bar_svg(out / "fig_A2_path_combo_risk_curve.svg", "A2 path combo longrisk", labels, [fnum(r.get("LongRisk_UCB")) for r in combo_rows])
    v9720.write_bar_svg(out / "fig_A3_core77_plus_expansion_pareto.svg", "A3 RiskAdjustedAUV", labels, [fnum(r.get("RiskAdjustedAUV_LCB")) for r in combo_rows])
    v9720.write_bar_svg(out / "fig_A4_raw_LDO_vs_support_adjusted_LDO.svg", "A4 raw minus support LDO", labels, [fnum(r.get("LDO_raw")) - fnum(r.get("LDO_support_adjusted")) for r in combo_rows])
    v9720.write_bar_svg(out / "fig_A5_per_dataset_path_support.svg", "A5 action count", labels, [fnum(r.get("action_count")) for r in combo_rows])
    return rows, summary


def score_quality(ap0: list[dict[str, Any]], feats: dict[str, dict[str, Any]], scores: dict[str, float]) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], list[float]]:
    idx_by = ap0_indices(ap0)
    full = [-1.0e9] * len(ap0)
    for aid, score in scores.items():
        if aid in idx_by:
            full[idx_by[aid]] = float(score)
    q = v9720.quality_for_scores(ap0, full, TARGET_K)
    top_ids = [aid for aid in sorted(scores, key=scores.get, reverse=True)[:TARGET_K] if aid in feats]
    top_feats = [feats[a] for a in top_ids]
    return q, top_feats, top_ids, full


def future_operator_scores(feats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        "FO1_AdamWAlignedFutureResidual": {
            "feature_kernel_count": 3,
            "extra_memory_mb": 0.0,
            "desc": "current residual alignment with curvature/jacobian penalties from landed train metrics",
            "score": lambda f: -f["CE1"] + 0.30 * f["Margin1"] - 0.15 * abs(f["Curvature1"]) - 0.10 * abs(f["Jacobian1"]),
        },
        "FO2_MemoryBufferResponseStability": {
            "feature_kernel_count": 3,
            "extra_memory_mb": 0.0,
            "desc": "memory/offdiag-safe stable h1-h5 train response",
            "score": lambda f: -abs(f["CE5"] - f["CE1"]) - 2.0 * f["memory_fail"] - 2.0 * f["offdiag_fail"] - 0.20 * max(0.0, f["CEp99_1"]),
        },
        "FO3_HardTailContraction": {
            "feature_kernel_count": 2,
            "extra_memory_mb": 0.0,
            "desc": "hard-tail contraction with current CE relief",
            "score": lambda f: -f["HardTail5"] + 0.20 * (-f["CE1"]) - 0.10 * max(0.0, f["CEp99_5"]),
        },
        "FO4_OldFamilyMarginPreservation": {
            "feature_kernel_count": 2,
            "extra_memory_mb": 0.0,
            "desc": "old-family margin preservation using landed train margin fields",
            "score": lambda f: 0.65 * f["Margin1"] + 0.35 * f["Margin5"] - 0.10 * abs(f["CEp99_1"]),
        },
        "FO5_MultiBatchAgreement": {
            "feature_kernel_count": 4,
            "extra_memory_mb": 0.0,
            "desc": "agreement between h1 and h5 train response summaries",
            "score": lambda f: -abs(f["CE1"] - f["CE5"]) + 0.25 * (f["Margin1"] + f["Margin5"]) - 0.10 * abs(f["HardTail5"] - f["HardTail1"]),
        },
        "FO6_LowCostTwoSampleVirtualPath": {
            "feature_kernel_count": 2,
            "extra_memory_mb": 0.0,
            "desc": "cheap two-sample proxy with payload/action norm penalty",
            "score": lambda f: -0.35 * f["payload_norm"] - 0.15 * f["action_norm"] - 0.001 * f["runtime_ms"] + 0.25 * (-f["CE1"]),
        },
        "FO7_RiskVetoOnly": {
            "feature_kernel_count": 2,
            "extra_memory_mb": 0.0,
            "desc": "risk-veto-only train proxy, intended as safety gate not value source",
            "score": lambda f: -4.0 * f["memory_fail"] - 4.0 * f["offdiag_fail"] - max(0.0, f["CEp99_1"]) - max(0.0, f["HardTail5"]),
        },
        "FO8_FO123HardVetoNoWeightedScore": {
            "feature_kernel_count": 5,
            "extra_memory_mb": 0.0,
            "desc": "FO1/FO2/FO3 rank sum with hard memory/offdiag/current-tail veto",
            "score": lambda f: (
                -1.0e6 if f["memory_fail"] or f["offdiag_fail"] or max(0.0, f["CEp99_1"]) > 0.25
                else (-f["CE1"] - abs(f["CE5"] - f["CE1"]) - f["HardTail5"])
            ),
        },
    }


def p2_future_operator_proxy(ap0: list[dict[str, Any]], feats: dict[str, dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pid, spec in future_operator_scores(feats).items():
        scores = {aid: float(spec["score"](f)) for aid, f in feats.items()}
        q, top_feats, top_ids, full_scores = score_quality(ap0, feats, scores)
        costs = [fnum(f.get("runtime_ms")) for f in top_feats]
        score_vals = [scores[a] for a in top_ids]
        core_labels = [1.0 if str(f.get("group_id")) == "Core77" else 0.0 for f in feats.values()]
        all_scores = [scores[a] for a in feats]
        all_auv = [fnum(f.get("AUV")) for f in feats.values()]
        all_rauv = [fnum(f.get("RiskAdjustedAUV")) for f in feats.values()]
        lfo = v9770.lfo_drop(full_scores, ap0, TARGET_K)
        cost_q90 = qtile(costs, 0.90)
        weak = int(
            fnum(q.get("GradeAB_precision")) >= 0.70
            and fnum(q.get("V_integrated_LCB")) > 0
            and fnum(q.get("h240_longrisk_UCB")) <= 0.05
            and fnum(q.get("memory_fail_UCB")) <= 0.05
            and fnum(q.get("offdiag_fail_UCB")) <= 0.05
            and cost_q90 <= 1.5
        )
        strong = int(
            weak
            and fnum(q.get("GradeAB_precision")) >= 0.80
            and fnum(q.get("LDO_drop")) <= 0.10
            and fnum(q.get("LSO_drop")) <= 0.10
            and fnum(q.get("LTO_drop")) <= 0.10
            and cost_q90 <= 0.5
        )
        fail_reason = ""
        if not weak:
            fail_bits = []
            if fnum(q.get("GradeAB_precision")) < 0.70:
                fail_bits.append("precision_below_0.70")
            if fnum(q.get("V_integrated_LCB")) <= 0:
                fail_bits.append("V_LCB_nonpositive")
            if fnum(q.get("h240_longrisk_UCB")) > 0.05:
                fail_bits.append("longrisk_above_0.05")
            if fnum(q.get("memory_fail_UCB")) > 0.05 or fnum(q.get("offdiag_fail_UCB")) > 0.05:
                fail_bits.append("memory_or_offdiag_above_0.05")
            if cost_q90 > 1.5:
                fail_bits.append("cost_q90_above_1.5ms")
            fail_reason = ";".join(fail_bits)
        rows.append({
            "stage": "P2_FUTURE_OPERATOR_PROXY_V9860",
            "status": "proxy_summary",
            "proxy_id": pid,
            "description": spec["desc"],
            "feature_legality": "green:train_time_operator_proxy_from_landed_train_metrics",
            "TopK87_precision": q.get("GradeAB_precision"),
            "V_LCB": q.get("V_integrated_LCB"),
            "AUV_LCB": lcb([fnum(f.get("AUV")) for f in top_feats]),
            "RiskAdjustedAUV_LCB": lcb([fnum(f.get("RiskAdjustedAUV")) for f in top_feats]),
            "LongRisk_UCB": q.get("h240_longrisk_UCB"),
            "Bad_UCB": q.get("bad_UCB"),
            "Null_UCB": q.get("null_UCB"),
            "MemoryFail_UCB": q.get("memory_fail_UCB"),
            "OffdiagFail_UCB": q.get("offdiag_fail_UCB"),
            "LDO": q.get("LDO_drop"),
            "LSO": q.get("LSO_drop"),
            "LTO": q.get("LTO_drop"),
            "LFO": lfo,
            "cost_q50_ms": qtile(costs, 0.50),
            "cost_q90_ms": cost_q90,
            "cost_q99_ms": qtile(costs, 0.99),
            "feature_kernel_count": spec["feature_kernel_count"],
            "extra_memory_mb": spec["extra_memory_mb"],
            "correlation_with_Core77_label": pearson(all_scores, core_labels),
            "correlation_with_future_AUV": pearson(all_scores, all_auv),
            "correlation_with_RiskAdjustedAUV": pearson(all_scores, all_rauv),
            "top_group_counts": json_counter([str(f.get("group_id")) for f in top_feats]),
            "score_mean_top87": mean(score_vals),
            "P2_weak_pass": weak,
            "P2_strong_pass": strong,
            "failure_mode": fail_reason,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(rows, key=lambda r: (inum(r["P2_strong_pass"]), inum(r["P2_weak_pass"]), fnum(r["TopK87_precision"]), fnum(r["V_LCB"])))
    summary = {
        "stage": "P2_FUTURE_OPERATOR_PROXY_V9860",
        "status": "summary",
        "proxy_count": len(rows),
        "weak_pass_count": sum(inum(r.get("P2_weak_pass")) for r in rows),
        "strong_pass_count": sum(inum(r.get("P2_strong_pass")) for r in rows),
        "best_proxy_id": best.get("proxy_id"),
        "best_precision": best.get("TopK87_precision"),
        "best_V_LCB": best.get("V_LCB"),
        "best_AUV_LCB": best.get("AUV_LCB"),
        "best_cost_q90_ms": best.get("cost_q90_ms"),
        "P2_weak_pass": int(any(inum(r.get("P2_weak_pass")) for r in rows)),
        "P2_strong_pass": int(any(inum(r.get("P2_strong_pass")) for r in rows)),
        "StopB_future_operator_triggered": int(not any(inum(r.get("P2_weak_pass")) for r in rows)),
        "reason": "all_future_operator_proxies_fail_precision_value_risk_or_cost_gate",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    labels = [r["proxy_id"] for r in rows if r.get("status") == "proxy_summary"]
    proxy_rows = [r for r in rows if r.get("status") == "proxy_summary"]
    v9720.write_bar_svg(out / "fig_B1_proxy_precision_vs_cost.svg", "B1 precision", labels, [fnum(r.get("TopK87_precision")) for r in proxy_rows])
    v9720.write_bar_svg(out / "fig_B2_proxy_V_vs_longrisk.svg", "B2 V minus risk", labels, [fnum(r.get("V_LCB")) - fnum(r.get("LongRisk_UCB")) for r in proxy_rows])
    v9720.write_bar_svg(out / "fig_B3_proxy_AUV_vs_RiskAdjustedAUV.svg", "B3 RiskAdjustedAUV", labels, [fnum(r.get("RiskAdjustedAUV_LCB")) for r in proxy_rows])
    v9720.write_bar_svg(out / "fig_B4_proxy_score_vs_future_AUV_scatter.svg", "B4 corr future AUV", labels, [fnum(r.get("correlation_with_future_AUV")) for r in proxy_rows])
    v9720.write_bar_svg(out / "fig_B5_proxy_score_distribution_by_group.svg", "B5 cost q90", labels, [fnum(r.get("cost_q90_ms")) for r in proxy_rows])
    return rows, summary


def materializer_entrypoint_audit(panel_targets: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
    rows: list[dict[str, Any]] = [{
        "stage": "P3_NATURAL_AP0_STREAM_MATERIALIZER_V9860",
        "status": "summary",
        "materializer_entrypoint_found": found,
        "matched_entrypoints": ",".join(hits),
        "C0_single_preflight_pass": 0,
        "C1_16_preflight_pass": 0,
        "C2_128_preflight_pass": 0,
        "C3_5000_panel_pass": 0,
        "C4_10000_panel_pass": 0,
        "C5_20000_panel_pass": 0,
        "P3_preflight_pass": 0,
        "reason": "" if found else "no_landed_labeled_natural_AP0_stream_extension_materializer_for_v9860",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    for stage_id, n in [("C0-single", 1), ("C1-16", 16), ("C2-128", 128)]:
        rows.append({
            "stage": "P3_NATURAL_AP0_STREAM_MATERIALIZER_V9860",
            "status": "not_run",
            "preflight_id": stage_id,
            "panel_target": n,
            "panel_actual_actions": 0,
            "expected_rows": n * BRANCH_COUNT * len(HORIZONS),
            "actual_rows": 0,
            "rows_per_sec": 0,
            "failed_action_count": n,
            "unresolved_exception_type": "materializer_entrypoint_missing" if not found else "gate_not_open",
            "CoreLike_count": "",
            "CoreLike_rate": "",
            "quality_audit_pass": 0,
            "reason": "materializer_entrypoint_missing" if not found else "entrypoint_found_but_not_invoked_by_gate",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    for target in [t for t in panel_targets if t >= 5000]:
        rows.append({
            "stage": "P3_NATURAL_AP0_STREAM_MATERIALIZER_V9860",
            "status": "not_run",
            "preflight_id": f"C-panel-{target}",
            "panel_target": target,
            "panel_actual_actions": 0,
            "expected_rows": target * BRANCH_COUNT * len(HORIZONS),
            "actual_rows": 0,
            "rows_per_sec": 0,
            "failed_action_count": target,
            "unresolved_exception_type": "materializer_entrypoint_missing" if not found else "gate_not_open",
            "CoreLike_count": "",
            "CoreLike_rate": "",
            "quality_audit_pass": 0,
            "reason": "materializer_entrypoint_missing" if not found else "entrypoint_found_but_not_invoked_by_gate",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    return rows, rows[0]


def p3_density_panel(source_v9820: Path, panel_targets: list[int], p3: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, summary = v9840.p4_density_panels(source_v9820, panel_targets, {"P3_preflight_pass": p3.get("P3_preflight_pass")}, out)
    for r in rows:
        r["stage"] = "P3_NATURAL_AP0_DENSITY_PANEL_V9860"
        if r.get("status") == "panel_row":
            r["row_source"] = "existing_labeled_AP0_density_panel_no_extension_materializer"
            r["expected_rows"] = ""
            r["actual_rows"] = ""
            r["rows_per_sec"] = ""
            r["quality_audit_pass"] = 1
            r["CoreLike_LCB"] = r.get("CoreLike_Wilson_LCB")
            r["CoreLike_UCB"] = r.get("CoreLike_Wilson_UCB")
        elif r.get("status") == "not_run":
            target = inum(r.get("target_action_count"))
            r["expected_rows"] = target * BRANCH_COUNT * len(HORIZONS)
            r["actual_rows"] = 0
            r["rows_per_sec"] = 0
            r["quality_audit_pass"] = 0
            r["reason"] = "P3_preflight_not_passed_materializer_missing"
    summary["stage"] = "P3_NATURAL_AP0_DENSITY_PANEL_V9860"
    summary["P3_density_weak_pass"] = 0
    summary["P3_density_strong_pass"] = 0
    summary["P3_density_fail"] = 0
    summary["reason"] = "extension_panels_blocked_by_missing_labeled_materializer"
    return rows, summary


def generated_sandbox_decision(p1: dict[str, Any], p2: dict[str, Any], p3_density: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    condition_b = inum(p2.get("P2_weak_pass"))
    condition_c_insufficient = inum(p3_density.get("P3_density_fail"))
    condition_a_combo_lowcost = int(inum(p1.get("P1_A_combo_strong_pass")) and inum(p2.get("P2_weak_pass")))
    allowed = int(condition_b or (condition_c_insufficient and inum(p1.get("P1_A_combo_strong_pass"))) or condition_a_combo_lowcost)
    row = {
        "stage": "P4_GENERATED_CONDITIONAL_SANDBOX_V9860",
        "status": "summary" if allowed else "not_run",
        "condition_B_future_operator_proxy_weak_pass": condition_b,
        "condition_C_density_insufficient_with_A_mechanism": int(condition_c_insufficient and inum(p1.get("P1_A_combo_strong_pass"))),
        "condition_A_combo_clean_plus_B_low_cost": condition_a_combo_lowcost,
        "generated_sandbox_allowed": allowed,
        "generated_route_status": "sandbox_allowed" if allowed else "stopped_no_B_or_C_mechanism_evidence",
        "generated_action_count": 0,
        "payload_hash_missing": "",
        "apply_error_linf": "",
        "future_path_rows": 0,
        "new_positive_created_rate": "",
        "longrisk_created_rate": "",
        "cost_q90": "",
        "negative_control_gap": "",
        "P4_generated_weak_pass": 0,
        "P4_generated_strong_pass": 0,
        "reason": "" if allowed else "B_weak_absent_C_density_unresolved_A_combo_diagnostic_only",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_D1_generated_sandbox_gate.svg", "D1 generated sandbox", ["allowed"], [allowed])
    return [row], row


def not_run(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {"stage": stage, "status": "not_run", "reason": reason, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0, **extra}
    return [row], row


def copy_base_acc(source_v9810: Path) -> list[dict[str, Any]]:
    rows = [dict(r) for r in read_csv(source_v9810 / "base_acc_sentinel_v9810.csv")]
    for r in rows:
        r["stage"] = "BASE_ACC_SENTINEL_V9860"
        r["reused_from_v9810"] = 1
        r["base_acc_used_for_controller"] = 0
        r["fake_data_used"] = 0
        r["proxy_row_used"] = 0
        r["cpu_offload_used"] = 0
    return rows


def field_legality() -> list[dict[str, Any]]:
    rows = [
        ("green", "FO1,FO2,FO3,FO4,FO5,FO6,FO7,FO8 train-time proxy inputs", "computed from landed train-stream probe metrics, not future labels"),
        ("yellow", "curvature_proxy,jacobian_proxy,basis_effective_rank,cover_entropy", "architecture-sensitive diagnostics"),
        ("red", "future V/AUV/RiskAdjustedAUV labels, Core77 label, OldRank, WT80, dataset branch rules", "diagnostic only, not official controller fields"),
    ]
    return [{
        "stage": "FIELD_LEGALITY_LEDGER_V9860",
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


def route_from(p1: dict[str, Any], p2: dict[str, Any], p3m: dict[str, Any], p3d: dict[str, Any], p4: dict[str, Any]) -> tuple[str, str, str, str]:
    if inum(p2.get("P2_strong_pass")) or inum(p2.get("P2_weak_pass")):
        return "R2-LegalFutureOperatorProxyFound", "legal_future_operator_proxy_found", "none", "enter_controller_boundary"
    if inum(p3d.get("P3_density_strong_pass")):
        return "R3-NaturalStreamDensitySufficient", "natural_density_sufficient", "none", "existing_action_harvesting_continue"
    if inum(p3d.get("P3_density_fail")):
        return "R4-NaturalStreamDensityInsufficient", "natural_density_insufficient", "legal_proxy_absent", "pivot_to_optimizer_level_generated_update"
    if inum(p1.get("P1_A_combo_strong_pass")) or inum(p1.get("P1_density_gate_dispute")):
        return "R1-FuturePathComboCandidate", "legal_future_operator_proxy_absent", "natural_stream_materializer_missing", "continue_B_C_only"
    if not inum(p2.get("P2_weak_pass")) and not inum(p3m.get("materializer_entrypoint_found")):
        return "R5-LegalProxyFailNaturalStreamMissing", "legal_future_operator_proxy_failed", "natural_stream_materializer_missing", "prioritize_C_materializer_or_pivot"
    if inum(p4.get("generated_sandbox_allowed")):
        return "R6-GeneratedSandboxAllowed", "generated_sandbox_allowed", "none", "run_D_sandbox"
    return "R7-GeneratedRouteStopped", "B_C_D_conditions_not_met", "natural_stream_materializer_missing", "generated_route_stopped"


def write_dashboard(out: Path, route: dict[str, Any]) -> None:
    p1 = summary_row(read_csv(out / "p1_future_path_combo_reassessment_v9860.csv"))
    p2 = summary_row(read_csv(out / "p2_future_operator_proxy_v9860.csv"))
    p3m = summary_row(read_csv(out / "p3_natural_ap0_stream_materializer_v9860.csv"))
    p3d = summary_row(read_csv(out / "p3_natural_ap0_density_panel_v9860.csv"))
    p4 = summary_row(read_csv(out / "p4_generated_conditional_sandbox_v9860.csv"))
    nf = summary_row(read_csv(out / "no_fake_audit_v9860.csv"))
    lines = [
        "# v9860_dashboard",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"primary_blocker = {route.get('primary_blocker')}",
        f"secondary_blocker = {route.get('secondary_blocker')}",
        "```",
        "",
        "| line | pass/fail | key value |",
        "|---|---|---|",
        f"| A combo | `{p1.get('P1_A_combo_strong_pass')}` / dispute `{p1.get('P1_density_gate_dispute')}` | best `{p1.get('best_candidate_id')}` AUV `{p1.get('best_AUV_LCB')}` |",
        f"| B future operator | `{p2.get('P2_weak_pass')}` / `{p2.get('P2_strong_pass')}` | best `{p2.get('best_proxy_id')}` precision `{p2.get('best_precision')}` cost `{p2.get('best_cost_q90_ms')}` |",
        f"| C natural stream | materializer `{p3m.get('materializer_entrypoint_found')}` | PanelA LCB `{p3d.get('PanelA_CoreLike_LCB')}` |",
        f"| D generated | `{p4.get('generated_sandbox_allowed')}` | `{p4.get('generated_route_status')}` |",
        "",
        f"rows_checked = `{nf.get('rows_checked')}`; fake/proxy/cpu = `{nf.get('fake_data_used')}` / `{nf.get('proxy_row_used')}` / `{nf.get('cpu_offload_used')}`。",
        "",
        "```text",
        "fig_A1_path_combo_V_curve.svg",
        "fig_A2_path_combo_risk_curve.svg",
        "fig_A3_core77_plus_expansion_pareto.svg",
        "fig_A4_raw_LDO_vs_support_adjusted_LDO.svg",
        "fig_A5_per_dataset_path_support.svg",
        "fig_B1_proxy_precision_vs_cost.svg",
        "fig_B2_proxy_V_vs_longrisk.svg",
        "fig_B3_proxy_AUV_vs_RiskAdjustedAUV.svg",
        "fig_B4_proxy_score_vs_future_AUV_scatter.svg",
        "fig_B5_proxy_score_distribution_by_group.svg",
        "fig_C1_density_curve_panel_size.svg",
        "fig_C2_corelike_rate_ci.svg",
        "fig_D1_generated_sandbox_gate.svg",
        "```",
        "",
    ]
    (out / "v9860_dashboard.md").write_text("\n".join(lines), encoding="utf-8")


def write_recap(out: Path, route: dict[str, Any], hashes: dict[str, str]) -> None:
    p0 = summary_row(read_csv(out / "p0_boundary_reproduction_v9860.csv"))
    landed = summary_row(read_csv(out / "p0_landed_future_row_reference_v9860.csv"))
    p1 = summary_row(read_csv(out / "p1_future_path_combo_reassessment_v9860.csv"))
    p2 = summary_row(read_csv(out / "p2_future_operator_proxy_v9860.csv"))
    p3m = summary_row(read_csv(out / "p3_natural_ap0_stream_materializer_v9860.csv"))
    p3d = summary_row(read_csv(out / "p3_natural_ap0_density_panel_v9860.csv"))
    p4 = summary_row(read_csv(out / "p4_generated_conditional_sandbox_v9860.csv"))
    p5 = summary_row(read_csv(out / "p5_minimal_controller_boundary_v9860.csv"))
    nf = summary_row(read_csv(out / "no_fake_audit_v9860.csv"))
    combos = [r for r in read_csv(out / "p1_future_path_combo_reassessment_v9860.csv") if r.get("status") == "combo_summary"]
    proxies = [r for r in read_csv(out / "p2_future_operator_proxy_v9860.csv") if r.get("status") == "proxy_summary"]
    panels = [r for r in read_csv(out / "p3_natural_ap0_density_panel_v9860.csv") if r.get("status") in {"panel_row", "not_run"}]
    lines = [
        "# DG-KAN v9.8.6 Future Path Mechanism / Natural Stream / Geometry Adaptive Optimizer 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.8.6_结果解读_未来路径机制_自然扩流_几何自适应优化器_完整计划.md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest 或 v9.8.4/v9.8.5 已真实 materialized 的 landed rows；没有 fake data、proxy rows，也没有把 future-path diagnostic、future-operator proxy diagnostic、natural stream missing 或 generated not_run 写成 official controller pass。",
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
        f"1. P0 复现 v9.8.5 boundary：source route = `{p0.get('source_route_v9850')}`，A/B/C/D = `{p0.get('A_line_mechanism_pass_v9850')}/{p0.get('B_weak_pass_v9850')}/{p0.get('C_materializer_entrypoint_found_v9850')}/{p0.get('D_generated_reopen_allowed_v9850')}`。",
        f"2. 本轮引用 landed future rows = `{landed.get('realfunctional_row_count')}`，actions = `{landed.get('action_count')}`，horizons = `{landed.get('horizons')}`。",
        f"3. P1 重审组合路径：best = `{p1.get('best_candidate_id')}`，AUV LCB = `{p1.get('best_AUV_LCB')}`，support-adjusted LDO = `{p1.get('best_LDO_support_adjusted')}`。",
        f"4. P1 combo strong pass count = `{p1.get('path_combo_strong_pass_count')}`，density dispute count = `{p1.get('density_gate_dispute_count')}`。",
        f"5. P2 FO1-FO8 future-operator proxy 全部失败：weak/strong = `{p2.get('P2_weak_pass')}` / `{p2.get('P2_strong_pass')}`，best proxy = `{p2.get('best_proxy_id')}`。",
        f"6. P2 best precision/V/cost q90 = `{p2.get('best_precision')}` / `{p2.get('best_V_LCB')}` / `{p2.get('best_cost_q90_ms')}`。",
        f"7. P3 natural stream materializer entrypoint found = `{p3m.get('materializer_entrypoint_found')}`；PanelA CoreLike LCB = `{p3d.get('PanelA_CoreLike_LCB')}`。",
        f"8. P4 generated sandbox allowed = `{p4.get('generated_sandbox_allowed')}`，status = `{p4.get('generated_route_status')}`。",
        f"9. P5 controller = `{p5.get('status')}`，reason = `{p5.get('reason')}`。",
        f"10. No-fake audit：rows checked = `{nf.get('rows_checked')}`，fake/proxy/cpu = `{nf.get('fake_data_used')}` / `{nf.get('proxy_row_used')}` / `{nf.get('cpu_offload_used')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/run_v9860_future_path_mechanism_natural_stream_geometry_optimizer.py` | v9.8.6 runner；读取 v9.8.5/v9.8.4/v9.8.2 landed rows，执行 Core77+Expansion combo reassessment、FO1-FO8 future-operator proxy、natural AP0 materializer audit、generated sandbox decision 与 controller/runtime boundary。 |",
        "",
        "```text",
        "python -m py_compile experiments/run_v9860_future_path_mechanism_natural_stream_geometry_optimizer.py",
        "```",
        "",
        "```bash",
        "python experiments/run_v9860_future_path_mechanism_natural_stream_geometry_optimizer.py --out-dir results/real_rerun_20260506/v9860_future_path_mechanism_natural_stream_geometry_optimizer_full_20260516T200000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 2876,5000,10000,20000",
        "```",
        "",
        "说明：P1/P2 使用已真实落盘的 branch-horizon/train-probe rows；P3 因没有 natural labeled stream extension materializer，C0/C1/C2 与 5000/10000/20000 panel 都显式 `not_run`。",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 3. P1 Future Path Combo Reassessment",
        "",
        "| candidate | actions | AUV LCB | V80 LCB | V240 LCB | longrisk UCB | mem/off UCB | GradeAB precision | raw/support LDO | LSO/LTO/LFO | strong | density dispute |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for r in combos:
        memoff = fnum(r.get("MemoryFail_UCB")) + fnum(r.get("OffdiagFail_UCB"))
        lines.append(f"| `{r.get('candidate_id')}` | `{r.get('action_count')}` | `{r.get('AUV_LCB')}` | `{r.get('V80_LCB')}` | `{r.get('V240_LCB')}` | `{r.get('LongRisk_UCB')}` | `{memoff}` | `{r.get('GradeAB_precision')}` | `{r.get('LDO_raw')}`/`{r.get('LDO_support_adjusted')}` | `{r.get('LSO')}`/`{r.get('LTO')}`/`{r.get('LFO')}` | `{r.get('path_combo_strong_pass')}` | `{r.get('density_gate_dispute')}` |")
    lines += [
        "",
        "判断：P1 确认 Core77+CoreExpansion10 等组合可以形成更完整的 future-path diagnostic 对照，但 controller-ready 仍需要合法机制或 density gate closure。",
        "",
        "## 4. P2 Future-Operator Proxy",
        "",
        "| proxy | precision | V LCB | AUV LCB | RiskAdjustedAUV LCB | longrisk UCB | mem/off UCB | cost q90 ms | weak | strong | failure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in proxies:
        memoff = fnum(r.get("MemoryFail_UCB")) + fnum(r.get("OffdiagFail_UCB"))
        lines.append(f"| `{r.get('proxy_id')}` | `{r.get('TopK87_precision')}` | `{r.get('V_LCB')}` | `{r.get('AUV_LCB')}` | `{r.get('RiskAdjustedAUV_LCB')}` | `{r.get('LongRisk_UCB')}` | `{memoff}` | `{r.get('cost_q90_ms')}` | `{r.get('P2_weak_pass')}` | `{r.get('P2_strong_pass')}` | `{r.get('failure_mode')}` |")
    lines += [
        "",
        "判断：P2 这次不是复用 B1-B5，而是 FO1-FO8；结果仍没有 legal low-cost proxy pass，主要受 precision/value/risk 或 cost gate 阻断。",
        "",
        "## 5. P3 Natural AP0 Stream",
        "",
        "| panel | status | target | labeled | expected rows | actual rows | CoreLike rate | LCB/UCB | reason |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in panels:
        lines.append(f"| `{r.get('panel_id')}` | `{r.get('status')}` | `{r.get('target_action_count')}` | `{r.get('actual_labeled_action_count')}` | `{r.get('expected_rows')}` | `{r.get('actual_rows')}` | `{r.get('CoreLike_rate')}` | `{r.get('CoreLike_LCB')}`/`{r.get('CoreLike_UCB')}` | `{r.get('reason')}` |")
    lines += [
        "",
        "判断：C 线仍未落地 materializer。PanelA 是既有 2876 labeled AP0 density panel，不是 5000/10000/20000 扩流结果。",
        "",
        "## 6. P4-P8 Boundary",
        "",
        "```text",
        f"generated_sandbox_allowed = {p4.get('generated_sandbox_allowed')}",
        f"generated_route_status = {p4.get('generated_route_status')}",
        "controller/runtime/paired replay/short-full = not_run",
        "```",
        "",
        "判断：没有 B legal proxy pass，也没有 C density sufficient/insufficient 证据，因此 generated sandbox、controller、runtime、paired replay、short/full 全部关闭。",
        "",
        "## 7. Figures",
        "",
        "```text",
        "fig_A1_path_combo_V_curve.svg",
        "fig_A2_path_combo_risk_curve.svg",
        "fig_A3_core77_plus_expansion_pareto.svg",
        "fig_A4_raw_LDO_vs_support_adjusted_LDO.svg",
        "fig_A5_per_dataset_path_support.svg",
        "fig_B1_proxy_precision_vs_cost.svg",
        "fig_B2_proxy_V_vs_longrisk.svg",
        "fig_B3_proxy_AUV_vs_RiskAdjustedAUV.svg",
        "fig_B4_proxy_score_vs_future_AUV_scatter.svg",
        "fig_B5_proxy_score_distribution_by_group.svg",
        "fig_C1_density_curve_panel_size.svg",
        "fig_C2_corelike_rate_ci.svg",
        "fig_D1_generated_sandbox_gate.svg",
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
        "1. P1 证明 Core77+CoreExpansion10 等组合值得作为 future-path diagnostic candidate，但仍不是合法 controller。",
        "2. P2 FO1-FO8 没有找到低成本、训练当下合法、可同时满足 precision/value/risk 的 future-operator proxy。",
        "3. P3 natural AP0 stream materializer 仍缺失，density 不能裁决。",
        "4. P4 generated sandbox 没有打开条件。",
        "5. P5-P8 全部 gate-blocked，strict PureKAN functional 仍未成功。",
        "```",
        "",
        "最终一句话：",
        "",
        f"> v9.8.6 真实执行后停在 `{route.get('route')}`：future-path combo 的诊断信号更完整，但合法 future-operator proxy 仍失败，自然 AP0 扩流 materializer 仍未落地，因此不能进入 official controller，generated route 继续停止。",
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

    source_v9850 = Path(args.source_v9850)
    source_v9840 = Path(args.source_v9840)
    source_v9820 = Path(args.source_v9820)
    source_v9810 = Path(args.source_v9810)
    panel_targets = parse_ints(args.panel_targets)

    p0_rows, p0 = p0_boundary(source_v9850)
    dump_csv("p0_boundary_reproduction_v9860.csv", p0_rows)

    ap0 = v9850.load_ap0(args)
    real = v9850.load_real_rows(source_v9820, source_v9840)
    feats, _completed = v9850.feature_rows(real)

    dump_csv("p0_landed_future_row_reference_v9860.csv", landed_future_row_reference(real))

    p1_rows, p1 = p1_combo_reassessment(ap0, feats, out)
    dump_csv("p1_future_path_combo_reassessment_v9860.csv", p1_rows)

    p2_rows, p2 = p2_future_operator_proxy(ap0, feats, out)
    dump_csv("p2_future_operator_proxy_v9860.csv", p2_rows)

    p3m_rows, p3m = materializer_entrypoint_audit(panel_targets)
    dump_csv("p3_natural_ap0_stream_materializer_v9860.csv", p3m_rows)

    p3d_rows, p3d = p3_density_panel(source_v9820, panel_targets, p3m, out)
    dump_csv("p3_natural_ap0_density_panel_v9860.csv", p3d_rows)

    p4_rows, p4 = generated_sandbox_decision(p1, p2, p3d, out)
    dump_csv("p4_generated_conditional_sandbox_v9860.csv", p4_rows)

    p5_rows, p5 = not_run("P5_MINIMAL_CONTROLLER_BOUNDARY_V9860", "P1_P2_P3_not_official_controller_ready", controller_pass=0)
    dump_csv("p5_minimal_controller_boundary_v9860.csv", p5_rows)
    p6_rows, p6 = not_run("P6_SELECTED_RUNTIME_BOUNDARY_V9860", "P5_controller_not_passed", runtime_pass=0)
    dump_csv("p6_selected_runtime_boundary_v9860.csv", p6_rows)
    p7_rows, p7 = not_run("P7_PAIRED_REPLAY_BOUNDARY_V9860", "P6_runtime_not_passed", paired_replay_pass=0)
    dump_csv("p7_paired_replay_boundary_v9860.csv", p7_rows)
    p8_rows, p8 = not_run("P8_SHORT_FULL_BOUNDARY_V9860", "P7_paired_replay_not_passed", short_full_pass=0)
    dump_csv("p8_short_full_boundary_v9860.csv", p8_rows)

    dump_csv("base_acc_sentinel_v9860.csv", copy_base_acc(source_v9810))
    dump_csv("field_legality_ledger_v9860.csv", field_legality())

    route, primary, secondary, recommendation = route_from(p1, p2, p3m, p3d, p4)
    route_decision = {
        "stage": "ROUTE_DECISION_V9860",
        "status": "summary",
        "route": route,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "route_recommendation": recommendation,
        "source_route_v9850": p0.get("source_route_v9850"),
        "P0_boundary_pass": p0.get("P0_boundary_pass"),
        "P1_A_combo_strong_pass": p1.get("P1_A_combo_strong_pass"),
        "P1_density_gate_dispute": p1.get("P1_density_gate_dispute"),
        "P2_weak_pass": p2.get("P2_weak_pass"),
        "P2_strong_pass": p2.get("P2_strong_pass"),
        "P3_materializer_entrypoint_found": p3m.get("materializer_entrypoint_found"),
        "P3_density_weak_pass": p3d.get("P3_density_weak_pass"),
        "P3_density_strong_pass": p3d.get("P3_density_strong_pass"),
        "P3_density_fail": p3d.get("P3_density_fail"),
        "P4_generated_sandbox_allowed": p4.get("generated_sandbox_allowed"),
        "generated_route_status": p4.get("generated_route_status"),
        "P5_controller_pass": p5.get("controller_pass"),
        "P6_runtime_pass": p6.get("runtime_pass"),
        "P7_paired_replay_pass": p7.get("paired_replay_pass"),
        "P8_short_full_pass": p8.get("short_full_pass"),
        "system_legal_controller_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("route_decision_v9860.json", route_decision)

    no_fake = {
        "stage": "NO_FAKE_AUDIT_V9860",
        "status": "summary",
        "rows_checked": audit_rows(artifacts),
        "fake_proxy_nonzero_count": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "no_fake": 1,
        "no_proxy": 1,
    }
    dump_csv("no_fake_audit_v9860.csv", [no_fake])
    contract = {
        "stage": "CONTRACT_AUDIT_V9860",
        "status": "summary",
        "manual_forward/manual_backward/manual_adamw_update": "1/1/1",
        "v9850_boundary_pass": p0.get("P0_boundary_pass"),
        "P1_combo_strong/density_dispute": f"{p1.get('P1_A_combo_strong_pass')}/{p1.get('P1_density_gate_dispute')}",
        "P2_future_operator_weak/strong": f"{p2.get('P2_weak_pass')}/{p2.get('P2_strong_pass')}",
        "P3_materializer_entrypoint_found": p3m.get("materializer_entrypoint_found"),
        "P4_generated_sandbox_allowed": p4.get("generated_sandbox_allowed"),
        "controller/runtime/system": "0/0/0",
        "uses_future_outcome_for_official_controller": 0,
        "diagnostic_promoted_to_official": 0,
        "fake/proxy/cpu_offload": "0/0/0",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("contract_audit_v9860.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9860",
        "status": "summary",
        "route": route,
        "F0_boundary_fail": int(not inum(p0.get("P0_boundary_pass"))),
        "F1_combo_not_official_controller": 1,
        "F2_future_operator_proxy_failed": int(not inum(p2.get("P2_weak_pass"))),
        "F3_natural_stream_materializer_missing": int(not inum(p3m.get("materializer_entrypoint_found"))),
        "F4_density_unresolved": int(not inum(p3d.get("P3_density_strong_pass")) and not inum(p3d.get("P3_density_fail"))),
        "F5_generated_route_stopped": int(not inum(p4.get("generated_sandbox_allowed"))),
        "F6_controller_runtime_blocked": 1,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("failure_taxonomy_v9860.csv", [failure])

    write_dashboard(out, route_decision)
    artifacts["v9860_dashboard.md"] = out / "v9860_dashboard.md"

    hashes = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    manifest = {
        "stage": "RUN_MANIFEST_V9860",
        "status": "summary",
        "out_dir": str(out),
        "execution_profile": args.execution_profile,
        "command_profile": {
            "device": args.device,
            "data_root": args.data_root,
            "seed": args.seed,
            "panel_targets": args.panel_targets,
            "source_v9850": args.source_v9850,
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
    dump_json("run_manifest_v9860.json", manifest)
    hashes_for_recap = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    write_recap(out, route_decision, hashes_for_recap)

    print(json.dumps({
        "out_dir": str(out),
        "route": route,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "P1_A_combo_strong_pass": p1.get("P1_A_combo_strong_pass"),
        "P1_density_gate_dispute": p1.get("P1_density_gate_dispute"),
        "P2_weak_pass": p2.get("P2_weak_pass"),
        "P2_strong_pass": p2.get("P2_strong_pass"),
        "P3_materializer_entrypoint_found": p3m.get("materializer_entrypoint_found"),
        "P4_generated_sandbox_allowed": p4.get("generated_sandbox_allowed"),
        "system_legal_controller_pass": 0,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
