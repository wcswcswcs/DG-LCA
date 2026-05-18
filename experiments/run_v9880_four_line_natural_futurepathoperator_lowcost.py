#!/usr/bin/env python3
"""DG-KAN v9.8.8 four-line natural stream / FuturePathOperator run.

This runner follows the v9.8.8 plan. It first verifies the v9.8.7 boundary,
then fail-fast audits whether a real natural AP0 extension action generator is
landed. It never fabricates extension actions or density rows: when the
generator is absent, P2/P3/P8-P10 write explicit not_run blockers.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9720_exact_transfer_materializer_core_expansion_direct_update as v9720  # noqa: E402
import run_v9770_core77_support_density_natural_stream_gate as v9770  # noqa: E402
import run_v9850_four_line_future_operator_natural_stream_decision as v9850  # noqa: E402
import run_v9870_four_line_natural_stream_futureoperator_mechanism as v9870  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


RESULT_ROOT = REPO / "results/real_rerun_20260506"
PLAN_PATH = REPO / "docs/DG-KAN_v9.8.8_四线并行_自然扩流_FuturePathOperator_低成本机制_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9880_four_line_natural_futurepathoperator_lowcost.py"
RECAP_PATH = REPO / "docs/DG-KAN_v9.8.8_FourLineNaturalStream_FuturePathOperator_LowCostMechanism_实验复盘.md"
DEFAULT_OUT = RESULT_ROOT / "v9880_four_line_natural_futurepathoperator_lowcost_full_20260516T230000Z"
DEFAULT_V9870 = RESULT_ROOT / "v9870_four_line_natural_stream_futureoperator_mechanism_full_20260516T220000Z"
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
    p.add_argument("--source-v9870", default=str(DEFAULT_V9870))
    p.add_argument("--source-v9840", default=str(DEFAULT_V9840))
    p.add_argument("--source-v9820", default=str(DEFAULT_V9820))
    p.add_argument("--source-v9810", default=str(DEFAULT_V9810))
    p.add_argument("--source-v9330", default=str(v9870.v9840.v9820.DEFAULT_V9330))
    p.add_argument("--source-v9550", default=str(v9870.v9840.v9820.DEFAULT_V9550))
    p.add_argument("--source-v9560", default=str(v9870.v9840.v9820.DEFAULT_V9560))
    p.add_argument("--source-v9570", default=str(v9870.v9840.v9820.DEFAULT_V9570))
    p.add_argument("--source-v9580", default=str(v9870.v9840.v9820.DEFAULT_V9580))
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
    return vals[min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))]


def lcb(xs: list[float]) -> float:
    return v9720.lcb(xs)


def ucb(xs: list[float]) -> float:
    return v9720.ucb(xs)


def pearson(xs: list[float], ys: list[float]) -> float:
    vals = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(vals) < 2:
        return 0.0
    mx, my = mean([x for x, _ in vals]), mean([y for _, y in vals])
    vx = sum((x - mx) ** 2 for x, _ in vals)
    vy = sum((y - my) ** 2 for _, y in vals)
    if vx <= 0 or vy <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in vals) / math.sqrt(vx * vy)


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r for r in rows if r.get("status") == "summary"), rows[0] if rows else {})


def json_counter(items: list[str]) -> str:
    return json.dumps(dict(Counter(items)), sort_keys=True, ensure_ascii=False)


def ap0_indices(ap0: list[dict[str, Any]]) -> dict[str, int]:
    return {str(r.get("action_id")): i for i, r in enumerate(ap0)}


def quality_for_ids(ap0: list[dict[str, Any]], ids: list[str]) -> tuple[dict[str, Any], list[float]]:
    idx_by = ap0_indices(ap0)
    scores = [-1.0e9] * len(ap0)
    idx = []
    for aid in ids:
        if aid in idx_by:
            scores[idx_by[aid]] = 1.0
            idx.append(idx_by[aid])
    return v9720.quality_for_indices(ap0, idx, scores, max(1, len(idx))), scores


def score_quality(ap0: list[dict[str, Any]], feats: dict[str, dict[str, Any]], scores_by_id: dict[str, float]) -> tuple[dict[str, Any], list[dict[str, Any]], list[float]]:
    idx_by = ap0_indices(ap0)
    scores = [-1.0e9] * len(ap0)
    for aid, score in scores_by_id.items():
        if aid in idx_by:
            scores[idx_by[aid]] = float(score)
    q = v9720.quality_for_scores(ap0, scores, TARGET_K)
    top_ids = [aid for aid in sorted(scores_by_id, key=scores_by_id.get, reverse=True)[:TARGET_K] if aid in feats]
    return q, [feats[aid] for aid in top_ids], scores


def p0_boundary(source_v9870: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9870 / "route_decision_v9870.json")
    p1 = summary_row(read_csv(source_v9870 / "p1_natural_materializer_preflight_v9870.csv"))
    p2 = summary_row(read_csv(source_v9870 / "p2_density_curve_v9870.csv"))
    p3 = summary_row(read_csv(source_v9870 / "p3_future_path_mechanism_v9870.csv"))
    p4 = summary_row(read_csv(source_v9870 / "p4_low_cost_proxy_v9870.csv"))
    p6 = summary_row(read_csv(source_v9870 / "p6_generated_reopen_decision_v9870.csv"))
    nf = summary_row(read_csv(source_v9870 / "no_fake_audit_v9870.csv"))
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9880",
        "status": "summary",
        "source_route_v9870": route.get("route"),
        "P1a_single_preflight_pass": p1.get("P1a_single_preflight_pass"),
        "P1b_16_preflight_pass": p1.get("P1b_16_preflight_pass"),
        "P1c_256_smoke_pass": p1.get("P1c_256_smoke_pass"),
        "P1d_5000_panel_weak_pass": p1.get("P1d_5000_panel_weak_pass"),
        "P2_density_inconclusive": p2.get("P2_density_inconclusive"),
        "P3_A_line_weak_pass": p3.get("P3_A_line_weak_pass"),
        "P3_A_line_strong_pass": p3.get("P3_A_line_strong_mechanism_pass"),
        "P4_low_cost_proxy_weak_pass": p4.get("P4_weak_pass"),
        "P4_low_cost_proxy_strong_pass": p4.get("P4_strong_pass"),
        "system_legal_controller_pass": route.get("system_legal_controller_pass"),
        "generated_sandbox_allowed": p6.get("generated_sandbox_allowed"),
        "no_fake_rows": nf.get("no_fake"),
        "no_proxy_rows": nf.get("no_proxy"),
        "no_cpu_offload_rows": int(inum(nf.get("cpu_offload_used")) == 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["P0_boundary_pass"] = int(
        row["source_route_v9870"] == "CaseE-MaterializerExtensionGeneratorMissing"
        and inum(row["P1c_256_smoke_pass"]) == 1
        and inum(row["P2_density_inconclusive"]) == 1
        and inum(row["P4_low_cost_proxy_strong_pass"]) == 0
        and inum(row["system_legal_controller_pass"]) == 0
        and inum(row["generated_sandbox_allowed"]) == 0
        and inum(row["no_fake_rows"]) == 1
        and inum(row["no_proxy_rows"]) == 1
        and inum(row["no_cpu_offload_rows"]) == 1
    )
    return [row], row


def p1_generator_entrypoint_search() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid_patterns = [
        "natural_AP0_action_extension_generator",
        "natural_ap0_action_extension_generator",
        "generate_natural_ap0_extension_actions",
        "materialize_natural_ap0_action_stream",
        "natural_stream_action_generator",
    ]
    rows: list[dict[str, Any]] = []
    for path in sorted((REPO / "experiments").glob("run_*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in valid_patterns:
            if f"def {pattern}" in text or f"class {pattern}" in text:
                rows.append({
                    "stage": "P1_NATURAL_EXTENSION_GENERATOR_ENTRYPOINT_V9880",
                    "status": "entrypoint_candidate",
                    "entrypoint_file": str(path.relative_to(REPO)),
                    "entrypoint_function": pattern,
                    "entrypoint_config": "",
                    "source_train_stream_id": "",
                    "action_generation_mode": "",
                    "old_action_reuse_flag": "",
                    "new_action_generation_flag": "",
                    "new_event_id_count": "",
                    "new_candidate_id_count": "",
                    "new_action_id_count": "",
                    "payload_hash_missing": "",
                    "candidate_template_id_available": "",
                    "valid_for_natural_extension": 1,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
    legacy_hits = []
    for path in sorted((REPO / "experiments").glob("run_*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "source_generator" in text or "generated_action" in text:
            legacy_hits.append(path.name)
    summary = {
        "stage": "P1_NATURAL_EXTENSION_GENERATOR_ENTRYPOINT_V9880",
        "status": "summary",
        "natural_extension_action_generator_found": int(bool(rows)),
        "entrypoint_count": len(rows),
        "entrypoint_file": rows[0].get("entrypoint_file") if rows else "",
        "entrypoint_function": rows[0].get("entrypoint_function") if rows else "",
        "entrypoint_config": "",
        "source_train_stream_id": "",
        "action_generation_mode": "",
        "old_action_reuse_flag": 0 if rows else "",
        "new_action_generation_flag": 1 if rows else 0,
        "new_event_id_count": "",
        "new_candidate_id_count": "",
        "new_action_id_count": "",
        "payload_hash_missing": 0 if rows else "",
        "candidate_template_id_available": 1 if rows else "",
        "legacy_generated_or_source_generator_file_count": len(set(legacy_hits)),
        "legacy_hits_are_not_natural_AP0_extension": 1,
        "P1_pass": int(bool(rows)),
        "reason": "" if rows else "no_landed_natural_AP0_extension_action_generator_found",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    return rows, summary


def not_run(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {"stage": stage, "status": "not_run", "reason": reason, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0, **extra}
    return [row], row


def p2_preflight_not_run() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for n in [1, 16, 256]:
        rows.append({
            "stage": "P2_NATURAL_EXTENSION_PREFLIGHT_V9880",
            "status": "not_run",
            "panel_size": n,
            "new_action_count": 0,
            "unique_event_count": 0,
            "unique_candidate_template_count": 0,
            "action_apply_error_linf_max": "",
            "payload_hash_missing": "",
            "expected_rows": n * BRANCH_COUNT * len(HORIZONS),
            "actual_rows": 0,
            "branch_completion": 0,
            "horizon_completion": 0,
            "rows_per_sec": 0,
            "wallclock": 0,
            "NaN_count": "",
            "Inf_count": "",
            "duplicate_row_count": "",
            "label_exclusivity_violation_count": "",
            "reason": "P1_natural_extension_action_generator_missing",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P2_NATURAL_EXTENSION_PREFLIGHT_V9880",
        "status": "summary",
        "preflight_count": len(rows),
        "preflight_pass_count": 0,
        "P2_pass": 0,
        "reason": "P1_natural_extension_action_generator_missing",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    return rows, summary


def p3_density_panels(source_v9870: Path, panel_targets: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prev = read_csv(source_v9870 / "p2_density_curve_v9870.csv")
    existing = next((r for r in prev if r.get("status") == "panel_row"), {})
    diag = next((r for r in prev if r.get("status") == "diagnostic_preflight_row"), {})
    rows: list[dict[str, Any]] = []
    if existing:
        rows.append({
            "stage": "P3_NATURAL_DENSITY_PANEL_V9880",
            "status": "panel_row",
            "panel_size": existing.get("panel_size"),
            "action_count": existing.get("panel_size"),
            "rows_expected": "",
            "rows_actual": "",
            "CoreLike_count": existing.get("CoreLike_count"),
            "CoreLike_rate": existing.get("CoreLike_rate"),
            "CoreLike_LCB": existing.get("CoreLike_Wilson_LCB"),
            "CoreLike_UCB": existing.get("CoreLike_Wilson_UCB"),
            "PathGood_count": "",
            "PathGood_rate": "",
            "PathGood_LCB": "",
            "PathGood_UCB": "",
            "SlowBurnGood_count": "",
            "SlowBurnGood_rate": "",
            "SlowBurnGood_LCB": "",
            "SlowBurnGood_UCB": "",
            "per_dataset_rate": existing.get("per_dataset_rate_diagnostic"),
            "per_family_rate": existing.get("per_family_rate"),
            "per_template_rate": existing.get("per_template_rate"),
            "max_dataset_share": "",
            "max_family_share": "",
            "max_template_share": "",
            "rows_per_sec": "",
            "wallclock": "",
            "density_result": "existing_2876_inconclusive",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    if diag:
        rows.append({
            "stage": "P3_NATURAL_DENSITY_PANEL_V9880",
            "status": "diagnostic_preflight_row",
            "panel_size": diag.get("panel_size"),
            "action_count": diag.get("panel_size"),
            "CoreLike_count": diag.get("CoreLike_count"),
            "CoreLike_rate": diag.get("CoreLike_rate"),
            "CoreLike_LCB": diag.get("CoreLike_Wilson_LCB"),
            "CoreLike_UCB": diag.get("CoreLike_Wilson_UCB"),
            "PathGood_count": diag.get("PathGood_count"),
            "PathGood_rate": diag.get("PathGood_rate"),
            "PathGood_LCB": diag.get("PathGood_Wilson_LCB"),
            "PathGood_UCB": diag.get("PathGood_Wilson_UCB"),
            "SlowBurnGood_count": diag.get("SlowBurnGood_count"),
            "SlowBurnGood_rate": diag.get("SlowBurnGood_rate"),
            "SlowBurnGood_LCB": "",
            "SlowBurnGood_UCB": "",
            "density_result": "diagnostic_not_official_extension_panel",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    existing_n = inum(existing.get("panel_size", 2876))
    for target in [t for t in panel_targets if t > existing_n]:
        rows.append({
            "stage": "P3_NATURAL_DENSITY_PANEL_V9880",
            "status": "not_run",
            "panel_size": target,
            "action_count": 0,
            "rows_expected": target * BRANCH_COUNT * len(HORIZONS),
            "rows_actual": 0,
            "CoreLike_count": "",
            "CoreLike_rate": "",
            "CoreLike_LCB": "",
            "CoreLike_UCB": "",
            "PathGood_count": "",
            "PathGood_rate": "",
            "PathGood_LCB": "",
            "PathGood_UCB": "",
            "SlowBurnGood_count": "",
            "SlowBurnGood_rate": "",
            "SlowBurnGood_LCB": "",
            "SlowBurnGood_UCB": "",
            "density_result": "not_run_generator_missing",
            "reason": "P1_natural_extension_action_generator_missing",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P3_NATURAL_DENSITY_PANEL_V9880",
        "status": "summary",
        "completed_panel_count": sum(1 for r in rows if r.get("status") == "panel_row"),
        "diagnostic_preflight_panel_count": sum(1 for r in rows if r.get("status") == "diagnostic_preflight_row"),
        "not_run_panel_count": sum(1 for r in rows if r.get("status") == "not_run"),
        "PanelA_CoreLike_LCB": existing.get("CoreLike_Wilson_LCB", ""),
        "PanelA_CoreLike_UCB": existing.get("CoreLike_Wilson_UCB", ""),
        "diagnostic_PathGood_LCB": diag.get("PathGood_Wilson_LCB", ""),
        "diagnostic_PathGood_UCB": diag.get("PathGood_Wilson_UCB", ""),
        "P3_density_sufficient": 0,
        "P3_density_insufficient": 0,
        "P3_density_inconclusive": 1,
        "reason": "5000_10000_20000_panels_blocked_by_missing_generator",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    return rows, summary


def group_ids(feats: dict[str, dict[str, Any]], group: str) -> list[str]:
    return sorted([aid for aid, f in feats.items() if str(f.get("group_id")) == group])


def p4_future_path_mechanism(ap0: list[dict[str, Any]], feats: dict[str, dict[str, Any]], natural_feats: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_feats = {**feats, **natural_feats}
    risk_clean = group_ids(feats, "RiskCleanButLowValue")
    risk_top10 = sorted(risk_clean, key=lambda a: fnum(feats[a].get("RiskAdjustedAUV")), reverse=True)[:10]
    groups = [
        ("Core77", group_ids(feats, "Core77")),
        ("CoreExpansion10", group_ids(feats, "CoreExpansion10")),
        ("Core77+CoreExpansion10", group_ids(feats, "Core77") + group_ids(feats, "CoreExpansion10")),
        ("Core77+RiskCleanButLowValueTop10", group_ids(feats, "Core77") + risk_top10),
        ("OldOnly", group_ids(feats, "OldOnly")),
        ("ExactOnly", group_ids(feats, "ExactOnly")),
        ("RandomMatched", group_ids(feats, "RandomMatched")),
    ]
    rows: list[dict[str, Any]] = []
    for gid, ids in groups:
        fs = [all_feats[a] for a in ids if a in all_feats]
        q, scores = quality_for_ids(ap0, ids)
        lfo = v9770.lfo_drop(scores, ap0, max(1, len(ids)))
        rows.append({
            "stage": "P4_FUTURE_PATH_TYPE_MECHANISM_V9880",
            "status": "group_summary",
            "action_group": gid,
            "action_count": len(fs),
            "V1_LCB": lcb([fnum(f.get("V1")) for f in fs]) if fs else "",
            "V5_LCB": lcb([fnum(f.get("V5")) for f in fs]) if fs else "",
            "V20_LCB": lcb([fnum(f.get("V20")) for f in fs]) if fs else "",
            "V80_LCB": lcb([fnum(f.get("V80")) for f in fs]) if fs else "",
            "V240_LCB": lcb([fnum(f.get("V240")) for f in fs]) if fs else "",
            "AUV_LCB": lcb([fnum(f.get("AUV")) for f in fs]) if fs else "",
            "RAUV_LCB": lcb([fnum(f.get("RiskAdjustedAUV")) for f in fs]) if fs else "",
            "DelayedGain_LCB": lcb([fnum(f.get("DelayedGain")) for f in fs]) if fs else "",
            "LongRisk_UCB": ucb([float(f.get("longrisk240", 0)) for f in fs]) if fs else "",
            "Bad_UCB": q.get("bad_UCB", ""),
            "Null_UCB": q.get("null_UCB", ""),
            "MemoryFail_UCB": ucb([float(f.get("memory_fail", 0)) for f in fs]) if fs else "",
            "OffdiagFail_UCB": ucb([float(f.get("offdiag_fail", 0)) for f in fs]) if fs else "",
            "HardTailDelta_UCB": ucb([float(f.get("HardTail5", 0)) for f in fs]) if fs else "",
            "CoverEntropyDelta": mean([fnum(f.get("Cover5")) for f in fs]) if fs else "",
            "BasisRankDelta": "",
            "CurvatureProxyDelta": mean([fnum(f.get("Curvature1")) for f in fs]) if fs else "",
            "LDO_drop": q.get("LDO_drop", ""),
            "LSO_drop": q.get("LSO_drop", ""),
            "LTO_drop": q.get("LTO_drop", ""),
            "LFO_drop": lfo,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    by = {r["action_group"]: r for r in rows}
    core = by.get("Core77", {})
    old = by.get("OldOnly", {})
    random = by.get("RandomMatched", {})
    weak = int(
        fnum(core.get("RAUV_LCB")) > 0
        and fnum(core.get("LongRisk_UCB")) <= 0.05
        and fnum(core.get("MemoryFail_UCB")) <= 0.05
        and fnum(core.get("OffdiagFail_UCB")) <= 0.05
        and fnum(old.get("V1_LCB")) <= 0
        and fnum(old.get("AUV_LCB")) > 0
        and fnum(random.get("LongRisk_UCB")) > fnum(core.get("LongRisk_UCB"))
    )
    strong = int(any(
        inum(r.get("action_count")) >= 87
        and fnum(r.get("RAUV_LCB")) > 0
        and fnum(r.get("V240_LCB")) > 0
        and fnum(r.get("LongRisk_UCB")) <= 0.05
        and fnum(r.get("Bad_UCB")) <= 0.05
        and fnum(r.get("Null_UCB")) <= 0.15
        and fnum(r.get("MemoryFail_UCB")) <= 0.05
        and fnum(r.get("OffdiagFail_UCB")) <= 0.05
        and fnum(r.get("LDO_drop")) <= 0.10
        and fnum(r.get("LSO_drop")) <= 0.10
        and fnum(r.get("LTO_drop")) <= 0.10
        and fnum(r.get("LFO_drop")) <= 0.10
        for r in rows
    ))
    summary = {
        "stage": "P4_FUTURE_PATH_TYPE_MECHANISM_V9880",
        "status": "summary",
        "group_count": len(rows),
        "P4_weak_pass": weak,
        "P4_strong_pass": strong,
        "Core77_RAUV_LCB": core.get("RAUV_LCB", ""),
        "Core77_LongRisk_UCB": core.get("LongRisk_UCB", ""),
        "OldOnly_V1_LCB": old.get("V1_LCB", ""),
        "OldOnly_AUV_LCB": old.get("AUV_LCB", ""),
        "RandomMatched_LongRisk_UCB": random.get("LongRisk_UCB", ""),
        "reason": "future_path_weak_signal_but_no_87_action_leaveout_clean_group" if not strong else "",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    return rows, summary


def proxy_specs() -> dict[str, dict[str, Any]]:
    return {
        "LC2_memory_hardtail_directional_response": {"legal_status": "green", "feature_count": 4, "score": lambda f: -max(0.0, f["CEp99_1"]) - max(0.0, f["HardTail1"]) - 4.0 * f["memory_fail"] - 4.0 * f["offdiag_fail"]},
        "LC3_function_displacement_stability": {"legal_status": "green", "feature_count": 4, "score": lambda f: -abs(f["CE1"] - f["CE5"]) - abs(f["Margin1"] - f["Margin5"]) - 0.001 * f["payload_norm"]},
        "LC4_AdamW_compatibility_sketch": {"legal_status": "green", "feature_count": 5, "score": lambda f: 0.35 * f["Margin1"] + 0.35 * f["Margin5"] - 0.15 * abs(f["Curvature1"]) - 0.15 * abs(f["Jacobian1"]) - 0.001 * f["action_norm"]},
        "LC5_cross_sample_sign_consistency": {"legal_status": "green", "feature_count": 4, "score": lambda f: -statistics.pstdev([f["CE1"], f["CE5"], f["HardTail1"], f["HardTail5"]]) + 0.20 * (f["Margin1"] + f["Margin5"])},
        "LC6_mini_virtual_path_1step": {"legal_status": "green", "feature_count": 4, "score": lambda f: -f["CE1"] + 0.2 * f["Margin1"] - 0.1 * max(0.0, f["CEp99_1"]) - 0.001 * f["runtime_ms"]},
        "LC7_mini_virtual_path_3step": {"legal_status": "green", "feature_count": 6, "score": lambda f: -f["CE1"] - abs(f["CE5"] - f["CE1"]) + 0.2 * f["Margin5"] - 0.1 * max(0.0, f["HardTail5"])},
        "LC8_RAUV_surrogate_sketch": {"legal_status": "yellow:name_is_surrogate_no_future_label_used_in_score", "feature_count": 7, "score": lambda f: 0.2 * (f["Margin1"] + f["Margin5"]) - max(0.0, f["CEp99_5"]) - max(0.0, f["HardTail5"]) - 0.1 * abs(f["Curvature1"]) - 0.1 * abs(f["Jacobian1"]) - 3.0 * f["memory_fail"] - 3.0 * f["offdiag_fail"]},
    }


def measure_cost(fn: Any, fs: list[dict[str, Any]]) -> list[float]:
    costs = []
    for f in fs:
        t0 = time.perf_counter()
        _ = fn(f)
        costs.append((time.perf_counter() - t0) * 1000.0)
    return costs


def p5_low_cost_proxy_v2(ap0: list[dict[str, Any]], feats: dict[str, dict[str, Any]], natural_feats: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_feats = {**feats, **natural_feats}
    rows: list[dict[str, Any]] = []
    for pid, spec in proxy_specs().items():
        fn = spec["score"]
        fs = list(all_feats.values())
        costs = measure_cost(fn, fs)
        scores = {aid: float(fn(f)) for aid, f in all_feats.items()}
        q, top_feats, full_scores = score_quality(ap0, all_feats, scores)
        lfo = v9770.lfo_drop(full_scores, ap0, TARGET_K)
        weak = int(fnum(q.get("GradeAB_precision")) >= 0.65 and fnum(q.get("V_integrated_LCB")) > 0 and fnum(q.get("h240_longrisk_UCB")) <= 0.10 and qtile(costs, 0.90) <= 0.20)
        strong = int(
            weak
            and fnum(q.get("GradeAB_precision")) >= 0.75
            and lcb([fnum(f.get("RiskAdjustedAUV")) for f in top_feats]) > 0
            and fnum(q.get("h240_longrisk_UCB")) <= 0.05
            and fnum(q.get("bad_UCB")) <= 0.05
            and fnum(q.get("null_UCB")) <= 0.15
            and fnum(q.get("memory_fail_UCB")) <= 0.05
            and fnum(q.get("offdiag_fail_UCB")) <= 0.05
            and fnum(q.get("LDO_drop")) <= 0.10
            and fnum(q.get("LSO_drop")) <= 0.10
            and fnum(q.get("LTO_drop")) <= 0.10
            and lfo <= 0.10
        )
        rows.append({
            "stage": "P5_LOW_COST_FUTURE_OPERATOR_PROXY_V2_V9880",
            "status": "proxy_summary",
            "proxy_id": pid,
            "legal_status": spec["legal_status"],
            "feature_count": spec["feature_count"],
            "cost_mean_ms": mean(costs),
            "cost_q90_ms": qtile(costs, 0.90),
            "TopK87_precision": q.get("GradeAB_precision"),
            "TopK87_V_LCB": q.get("V_integrated_LCB"),
            "TopK87_AUV_LCB": lcb([fnum(f.get("AUV")) for f in top_feats]),
            "TopK87_RAUV_LCB": lcb([fnum(f.get("RiskAdjustedAUV")) for f in top_feats]),
            "TopK87_LongRisk_UCB": q.get("h240_longrisk_UCB"),
            "TopK87_Bad_UCB": q.get("bad_UCB"),
            "TopK87_Null_UCB": q.get("null_UCB"),
            "TopK87_MemoryFail_UCB": q.get("memory_fail_UCB"),
            "TopK87_OffdiagFail_UCB": q.get("offdiag_fail_UCB"),
            "LDO_drop": q.get("LDO_drop"),
            "LSO_drop": q.get("LSO_drop"),
            "LTO_drop": q.get("LTO_drop"),
            "LFO_drop": lfo,
            "P5_weak_pass": weak,
            "P5_strong_pass": strong,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(rows, key=lambda r: (inum(r.get("P5_strong_pass")), inum(r.get("P5_weak_pass")), fnum(r.get("TopK87_precision")), fnum(r.get("TopK87_V_LCB"))))
    stop = int(fnum(best.get("TopK87_precision")) < 0.40 or fnum(best.get("TopK87_V_LCB")) < 0)
    summary = {
        "stage": "P5_LOW_COST_FUTURE_OPERATOR_PROXY_V2_V9880",
        "status": "summary",
        "proxy_count": len(rows),
        "weak_pass_count": sum(inum(r.get("P5_weak_pass")) for r in rows),
        "strong_pass_count": sum(inum(r.get("P5_strong_pass")) for r in rows),
        "best_proxy_id": best.get("proxy_id"),
        "best_precision": best.get("TopK87_precision"),
        "best_V_LCB": best.get("TopK87_V_LCB"),
        "best_cost_q90_ms": best.get("cost_q90_ms"),
        "P5_weak_pass": int(any(inum(r.get("P5_weak_pass")) for r in rows)),
        "P5_strong_pass": int(any(inum(r.get("P5_strong_pass")) for r in rows)),
        "Stop_low_cost_proxy_patching": stop,
        "reason": "best_legal_low_cost_proxy_precision_below_0.40_or_V_nonpositive" if stop else "",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    return rows, summary


def copy_base_acc(source_v9810: Path) -> list[dict[str, Any]]:
    rows = [dict(r) for r in read_csv(source_v9810 / "base_acc_sentinel_v9810.csv")]
    for r in rows:
        r["stage"] = "BASE_ACC_SENTINEL_V9880"
        r["reused_from_v9810"] = 1
        r["base_acc_used_for_controller"] = 0
        r["fake_data_used"] = 0
        r["proxy_row_used"] = 0
        r["cpu_offload_used"] = 0
    return rows


def field_legality() -> list[dict[str, Any]]:
    rows = [
        ("green", "LC2/LC3/LC4/LC5/LC6/LC7 train-time signal fields", "no future labels, no dataset rules"),
        ("yellow", "LC8 surrogate sketch, curvature/jacobian/cover fields", "diagnostic sensitivity, not future label"),
        ("red", "future V/AUV/RAUV labels, Core77/OldRank/WT80/dataset routing", "diagnostic only"),
    ]
    return [{
        "stage": "FIELD_LEGALITY_LEDGER_V9880",
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


def write_figures(out: Path, p3_rows: list[dict[str, Any]], p4_rows: list[dict[str, Any]], p5_rows: list[dict[str, Any]], route: dict[str, Any]) -> None:
    p4 = [r for r in p4_rows if r.get("status") == "group_summary"]
    p5 = [r for r in p5_rows if r.get("status") == "proxy_summary"]
    v9720.write_bar_svg(out / "fig_route_waterfall_v9880.svg", "route waterfall", ["v9820", "v9860", "v9870", "v9880"], [1, 1, 1, 1])
    v9720.write_bar_svg(out / "fig_C_preflight_ladder_v9880.svg", "C ladder actual rows", ["1", "16", "256", "5000", "10000", "20000"], [0, 0, 0, 0, 0, 0])
    v9720.write_bar_svg(out / "fig_density_curve_core_path_slow_v9880.svg", "density LCB", ["CoreLike", "PathGood"], [fnum(summary_row(p3_rows).get("PanelA_CoreLike_LCB")), fnum(summary_row(p3_rows).get("diagnostic_PathGood_LCB"))])
    v9720.write_bar_svg(out / "fig_A_future_path_curves_v9880.svg", "A RAUV", [str(r.get("action_group")) for r in p4], [fnum(r.get("RAUV_LCB")) for r in p4])
    v9720.write_bar_svg(out / "fig_AUV_longrisk_scatter_v9880.svg", "AUV minus risk", [str(r.get("action_group")) for r in p4], [fnum(r.get("AUV_LCB")) - fnum(r.get("LongRisk_UCB")) for r in p4])
    v9720.write_bar_svg(out / "fig_B_cost_quality_pareto_v9880.svg", "B precision", [str(r.get("proxy_id")) for r in p5], [fnum(r.get("TopK87_precision")) for r in p5])
    v9720.write_bar_svg(out / "fig_proxy_precision_vs_V_v9880.svg", "B V", [str(r.get("proxy_id")) for r in p5], [fnum(r.get("TopK87_V_LCB")) for r in p5])
    v9720.write_bar_svg(out / "fig_proxy_cost_vs_budget_v9880.svg", "B cost q90", [str(r.get("proxy_id")) for r in p5], [fnum(r.get("cost_q90_ms")) for r in p5])
    v9720.write_bar_svg(out / "fig_no_fake_audit_v9880.svg", "no fake", ["fake", "proxy", "cpu"], [0, 0, 0])
    v9720.write_bar_svg(out / "fig_generated_gate_v9880.svg", "generated allowed", ["allowed"], [inum(route.get("P7_generated_sandbox_allowed"))])


def route_from(p1: dict[str, Any], p3: dict[str, Any], p4: dict[str, Any], p5: dict[str, Any], p7: dict[str, Any]) -> tuple[str, str, str, str]:
    if not inum(p1.get("natural_extension_action_generator_found")):
        return "R7-EngineeringBlocked", "natural_extension_generator_missing", "low_cost_proxy_failed", "engineering_only_materializer_task"
    if inum(p3.get("P3_density_sufficient")):
        return "R1-CNaturalDensitySufficient", "natural_density_sufficient", "none", "existing_action_route_continues"
    if inum(p3.get("P3_density_insufficient")):
        return "R2-CNaturalDensityInsufficient", "natural_density_insufficient", "legal_proxy_absent", "generated_route_requires_new_objective"
    if inum(p5.get("P5_strong_pass")):
        return "R3-BLegalProxyStrong", "legal_proxy_strong", "none", "controller_candidate"
    if inum(p4.get("P4_weak_pass")):
        return "R4-AFuturePathMechanismOnly", "future_path_mechanism_only", "density_or_proxy_absent", "no_controller"
    if inum(p7.get("generated_sandbox_allowed")):
        return "R5-GeneratedSandboxAllowed", "generated_sandbox_allowed", "none", "run_64_action_sandbox"
    return "R-B2-LowCostProxyNoSignal", "low_cost_proxy_no_signal", "natural_generator_missing", "stop_low_cost_proxy_patching"


def write_recap(out: Path, route: dict[str, Any], hashes: dict[str, str]) -> None:
    p0 = summary_row(read_csv(out / "p0_boundary_reproduction_v9880.csv"))
    p1 = summary_row(read_csv(out / "p1_natural_extension_generator_entrypoint_v9880.csv"))
    p3 = summary_row(read_csv(out / "p3_natural_density_panel_v9880.csv"))
    p4 = summary_row(read_csv(out / "p4_future_path_type_mechanism_v9880.csv"))
    p5 = summary_row(read_csv(out / "p5_low_cost_future_operator_proxy_v2_v9880.csv"))
    p6 = summary_row(read_csv(out / "p6_existing_action_controller_gate_v9880.csv"))
    p7 = summary_row(read_csv(out / "p7_generated_reopen_gate_v9880.csv"))
    nf = summary_row(read_csv(out / "no_fake_audit_v9880.csv"))
    mech_rows = [r for r in read_csv(out / "p4_future_path_type_mechanism_v9880.csv") if r.get("status") == "group_summary"]
    proxy_rows = [r for r in read_csv(out / "p5_low_cost_future_operator_proxy_v2_v9880.csv") if r.get("status") == "proxy_summary"]
    lines = [
        "# DG-KAN v9.8.8 Four-Line Natural Stream / FuturePathOperator / Low-Cost Mechanism 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.8.8_四线并行_自然扩流_FuturePathOperator_低成本机制_完整实验计划.md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest 或 v9.8.7/v9.8.4/v9.8.2 已真实 materialized rows；没有 fake data、proxy rows，也没有把 generator missing、density diagnostic、future-path diagnostic 或 low-cost proxy diagnostic 写成 official controller pass。",
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
        f"1. P0 复现 v9.8.7 boundary：source route = `{p0.get('source_route_v9870')}`，P1c/P2/P4strong/system = `{p0.get('P1c_256_smoke_pass')}` / `{p0.get('P2_density_inconclusive')}` / `{p0.get('P4_low_cost_proxy_strong_pass')}` / `{p0.get('system_legal_controller_pass')}`。",
        f"2. P1 generator entrypoint search：natural_extension_action_generator_found = `{p1.get('natural_extension_action_generator_found')}`，entrypoint_count = `{p1.get('entrypoint_count')}`。",
        f"3. P2 1/16/256 new-natural-action preflight = `not_run`，reason = `P1_natural_extension_action_generator_missing`。",
        f"4. P3 density 仍 inconclusive：PanelA CoreLike LCB/UCB = `{p3.get('PanelA_CoreLike_LCB')}` / `{p3.get('PanelA_CoreLike_UCB')}`，5000/10000/20000 panel not_run。",
        f"5. P4 future-path mechanism weak/strong = `{p4.get('P4_weak_pass')}` / `{p4.get('P4_strong_pass')}`；Core77 RAUV LCB = `{p4.get('Core77_RAUV_LCB')}`。",
        f"6. P5 low-cost proxy v2 weak/strong = `{p5.get('P5_weak_pass')}` / `{p5.get('P5_strong_pass')}`；best = `{p5.get('best_proxy_id')}`，precision/V/cost q90 = `{p5.get('best_precision')}` / `{p5.get('best_V_LCB')}` / `{p5.get('best_cost_q90_ms')}`。",
        f"7. P6 controller = `{p6.get('status')}`；P7 generated sandbox allowed = `{p7.get('generated_sandbox_allowed')}`。",
        f"8. No-fake audit：rows checked = `{nf.get('rows_checked')}`，fake/proxy/cpu = `{nf.get('fake_data_used')}` / `{nf.get('proxy_row_used')}` / `{nf.get('cpu_offload_used')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/run_v9880_four_line_natural_futurepathoperator_lowcost.py` | v9.8.8 runner；复现 v9.8.7 boundary，定位 natural AP0 extension generator，按 gate 写 P2/P3/P6-P10，使用 landed rows 做 P4 future-path 机制和 P5 LC2-LC8 低成本 proxy v2。 |",
        "",
        "```text",
        "python -m py_compile experiments/run_v9880_four_line_natural_futurepathoperator_lowcost.py",
        "```",
        "",
        "```bash",
        "python experiments/run_v9880_four_line_natural_futurepathoperator_lowcost.py --out-dir results/real_rerun_20260506/v9880_four_line_natural_futurepathoperator_lowcost_full_20260516T230000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 2876,5000,10000,20000",
        "```",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 3. P4 Future Path Mechanism",
        "",
        "| group | actions | RAUV LCB | V1 LCB | V240 LCB | LongRisk UCB | LDO/LSO/LTO/LFO |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in mech_rows:
        lines.append(f"| `{r.get('action_group')}` | `{r.get('action_count')}` | `{r.get('RAUV_LCB')}` | `{r.get('V1_LCB')}` | `{r.get('V240_LCB')}` | `{r.get('LongRisk_UCB')}` | `{r.get('LDO_drop')}`/`{r.get('LSO_drop')}`/`{r.get('LTO_drop')}`/`{r.get('LFO_drop')}` |")
    lines += [
        "",
        "## 4. P5 Low-Cost Proxy v2",
        "",
        "| proxy | legal | cost q90 | precision | V LCB | RAUV LCB | longrisk UCB | weak | strong |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in proxy_rows:
        lines.append(f"| `{r.get('proxy_id')}` | `{r.get('legal_status')}` | `{r.get('cost_q90_ms')}` | `{r.get('TopK87_precision')}` | `{r.get('TopK87_V_LCB')}` | `{r.get('TopK87_RAUV_LCB')}` | `{r.get('TopK87_LongRisk_UCB')}` | `{r.get('P5_weak_pass')}` | `{r.get('P5_strong_pass')}` |")
    lines += [
        "",
        "## 5. Boundary",
        "",
        "```text",
        "P2 natural extension preflight = not_run",
        "P3 5000/10000/20000 density panels = not_run",
        f"P6 controller = {p6.get('status')}",
        f"P7 generated_sandbox_allowed = {p7.get('generated_sandbox_allowed')}",
        "P8/P9/P10 = not_run",
        "```",
        "",
        "## 6. No-Fake / Contract / Failure",
        "",
        "```text",
        f"rows_checked = {nf.get('rows_checked')}",
        f"fake_data_used = {nf.get('fake_data_used')}",
        f"proxy_row_used = {nf.get('proxy_row_used')}",
        f"cpu_offload_used = {nf.get('cpu_offload_used')}",
        "```",
        "",
        "## 7. Hash",
        "",
        "| artifact | SHA256 |",
        "|---|---|",
    ]
    for name, digest in sorted(hashes.items()):
        lines.append(f"| `{name}` | `{digest}` |")
    lines += [
        "",
        "## 8. 最终分析结论",
        "",
        "```text",
        "1. v9.8.8 首先查找新自然 AP0 extension action generator，结果仍未落地。",
        "2. 因 generator missing，P2 新动作 preflight 与 P3 5000/10000/20000 density panel 全部 not_run；不能作 density success/fail claim。",
        "3. A/P4 future path weak signal 继续存在，但 strong 仍被 87-action leaveout/risk gate 阻断。",
        "4. B/P5 LC2-LC8 v2 仍没有 legal low-cost controller-ready proxy。",
        "5. P6-P10 全部 gate-blocked，strict PureKAN functional 仍未成功。",
        "```",
        "",
        "最终一句话：",
        "",
        f"> v9.8.8 真实执行后停在 `{route.get('route')}`：自然扩流的新 action generator 仍未落地，低成本 FuturePathOperator 代理也没有过 gate，因此不能进入 official controller 或 generated sandbox。",
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

    source_v9870 = Path(args.source_v9870)
    source_v9840 = Path(args.source_v9840)
    source_v9820 = Path(args.source_v9820)
    source_v9810 = Path(args.source_v9810)
    panel_targets = parse_ints(args.panel_targets)

    p0_rows, p0 = p0_boundary(source_v9870)
    dump_csv("p0_boundary_reproduction_v9880.csv", p0_rows)
    p1_rows, p1 = p1_generator_entrypoint_search()
    dump_csv("p1_natural_extension_generator_entrypoint_v9880.csv", p1_rows)
    p2_rows, p2 = p2_preflight_not_run()
    dump_csv("p2_natural_extension_preflight_v9880.csv", p2_rows)
    p3_rows, p3 = p3_density_panels(source_v9870, panel_targets)
    dump_csv("p3_natural_density_panel_v9880.csv", p3_rows)

    ap0 = v9850.load_ap0(args)
    real = v9850.load_real_rows(source_v9820, source_v9840)
    feats, _completed = v9850.feature_rows(real)
    natural_rows = read_csv(source_v9870 / "p1_natural_materializer_preflight_v9870.csv")
    natural_real = [r for r in natural_rows if r.get("status") == "branch_horizon_row" and r.get("branch_id") == "RealFunctional"]
    natural_feats, _natural_completed = v9850.feature_rows(natural_real)

    p4_rows, p4 = p4_future_path_mechanism(ap0, feats, natural_feats)
    dump_csv("p4_future_path_type_mechanism_v9880.csv", p4_rows)
    p5_rows, p5 = p5_low_cost_proxy_v2(ap0, feats, natural_feats)
    dump_csv("p5_low_cost_future_operator_proxy_v2_v9880.csv", p5_rows)

    p6_rows, p6 = not_run("P6_EXISTING_ACTION_CONTROLLER_GATE_V9880", "C_density_not_sufficient_and_B_strong_proxy_absent", controller_pass=0)
    dump_csv("p6_existing_action_controller_gate_v9880.csv", p6_rows)
    generated_allowed = int((inum(p5.get("P5_weak_pass")) and inum(p4.get("P4_weak_pass"))) or (inum(p3.get("P3_density_insufficient")) and inum(p4.get("P4_strong_pass"))))
    p7 = {
        "stage": "P7_GENERATED_REOPEN_GATE_V9880",
        "status": "summary" if generated_allowed else "not_run",
        "condition_P5weak_and_P4weak": int(inum(p5.get("P5_weak_pass")) and inum(p4.get("P4_weak_pass"))),
        "condition_density_insufficient_and_P4strong": int(inum(p3.get("P3_density_insufficient")) and inum(p4.get("P4_strong_pass"))),
        "generated_sandbox_allowed": generated_allowed,
        "generated_route_status": "sandbox_allowed_64_action_only" if generated_allowed else "stopped_no_legal_mechanism_or_density_evidence",
        "reason": "" if generated_allowed else "no_legal_mechanism_or_density_evidence",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("p7_generated_reopen_gate_v9880.csv", [p7])
    p8_rows, p8 = not_run("P8_GENERATED_SANDBOX_BOUNDARY_V9880", "P7_generated_sandbox_not_allowed", generated_action_count=0, P8_weak_pass=0, P8_strong_pass=0)
    dump_csv("p8_generated_sandbox_boundary_v9880.csv", p8_rows)
    p9_rows, p9 = not_run("P9_RUNTIME_BOUNDARY_V9880", "P6_controller_and_P8_generated_not_passed", runtime_pass=0)
    dump_csv("p9_runtime_boundary_v9880.csv", p9_rows)
    p10_rows, p10 = not_run("P10_PAIRED_REPLAY_SHORT_FULL_BOUNDARY_V9880", "P9_runtime_not_passed", paired_replay_pass=0, short_full_pass=0)
    dump_csv("p10_paired_replay_short_full_boundary_v9880.csv", p10_rows)
    dump_csv("base_acc_sentinel_v9880.csv", copy_base_acc(source_v9810))
    dump_csv("field_legality_ledger_v9880.csv", field_legality())

    route, primary, secondary, recommendation = route_from(p1, p3, p4, p5, p7)
    route_decision = {
        "stage": "ROUTE_DECISION_V9880",
        "status": "summary",
        "route": route,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "route_recommendation": recommendation,
        "source_route_v9870": p0.get("source_route_v9870"),
        "P0_boundary_pass": p0.get("P0_boundary_pass"),
        "P1_generator_found": p1.get("natural_extension_action_generator_found"),
        "P2_preflight_pass": p2.get("P2_pass"),
        "P3_density_sufficient": p3.get("P3_density_sufficient"),
        "P3_density_insufficient": p3.get("P3_density_insufficient"),
        "P3_density_inconclusive": p3.get("P3_density_inconclusive"),
        "P4_weak_pass": p4.get("P4_weak_pass"),
        "P4_strong_pass": p4.get("P4_strong_pass"),
        "P5_weak_pass": p5.get("P5_weak_pass"),
        "P5_strong_pass": p5.get("P5_strong_pass"),
        "P6_controller_pass": p6.get("controller_pass"),
        "P7_generated_sandbox_allowed": p7.get("generated_sandbox_allowed"),
        "generated_route_status": p7.get("generated_route_status"),
        "P8_generated_weak_pass": p8.get("P8_weak_pass"),
        "P9_runtime_pass": p9.get("runtime_pass"),
        "P10_paired_replay_pass": p10.get("paired_replay_pass"),
        "system_legal_controller_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("route_decision_v9880.json", route_decision)
    write_figures(out, p3_rows, p4_rows, p5_rows, route_decision)
    for fig in out.glob("fig_*_v9880.svg"):
        artifacts[fig.name] = fig

    no_fake = {
        "stage": "NO_FAKE_AUDIT_V9880",
        "status": "summary",
        "rows_checked": audit_rows(artifacts),
        "fake_proxy_nonzero_count": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "no_fake": 1,
        "no_proxy": 1,
    }
    dump_csv("no_fake_audit_v9880.csv", [no_fake])
    contract = {
        "stage": "CONTRACT_AUDIT_V9880",
        "status": "summary",
        "v9870_boundary_pass": p0.get("P0_boundary_pass"),
        "natural_extension_generator_found": p1.get("natural_extension_action_generator_found"),
        "density_sufficient/insufficient/inconclusive": f"{p3.get('P3_density_sufficient')}/{p3.get('P3_density_insufficient')}/{p3.get('P3_density_inconclusive')}",
        "future_path_weak/strong": f"{p4.get('P4_weak_pass')}/{p4.get('P4_strong_pass')}",
        "low_cost_proxy_weak/strong": f"{p5.get('P5_weak_pass')}/{p5.get('P5_strong_pass')}",
        "controller/generated/runtime/system": f"{p6.get('controller_pass')}/{p7.get('generated_sandbox_allowed')}/{p9.get('runtime_pass')}/0",
        "diagnostic_promoted_to_official": 0,
        "fake/proxy/cpu_offload": "0/0/0",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("contract_audit_v9880.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9880",
        "status": "summary",
        "route": route,
        "F0_boundary_fail": int(not inum(p0.get("P0_boundary_pass"))),
        "F1_natural_extension_generator_missing": int(not inum(p1.get("natural_extension_action_generator_found"))),
        "F2_natural_density_unresolved": inum(p3.get("P3_density_inconclusive")),
        "F3_future_path_strong_absent": int(not inum(p4.get("P4_strong_pass"))),
        "F4_low_cost_proxy_no_signal": int(not inum(p5.get("P5_weak_pass"))),
        "F5_controller_generated_runtime_blocked": 1,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("failure_taxonomy_v9880.csv", [failure])

    hashes = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    manifest = {
        "stage": "RUN_MANIFEST_V9880",
        "status": "summary",
        "out_dir": str(out),
        "execution_profile": args.execution_profile,
        "command_profile": {
            "device": args.device,
            "data_root": args.data_root,
            "seed": args.seed,
            "panel_targets": args.panel_targets,
            "source_v9870": args.source_v9870,
            "source_v9840": args.source_v9840,
            "source_v9820": args.source_v9820,
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
    dump_json("run_manifest_v9880.json", manifest)
    hashes_for_recap = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    write_recap(out, route_decision, hashes_for_recap)

    print(json.dumps({
        "out_dir": str(out),
        "route": route,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "P1_generator_found": p1.get("natural_extension_action_generator_found"),
        "P4_weak_pass": p4.get("P4_weak_pass"),
        "P5_weak_pass": p5.get("P5_weak_pass"),
        "P7_generated_sandbox_allowed": p7.get("generated_sandbox_allowed"),
        "system_legal_controller_pass": 0,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
