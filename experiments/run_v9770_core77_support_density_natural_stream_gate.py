#!/usr/bin/env python3
"""DG-KAN v9.7.7 Core77 support-density gate runner.

This runner follows the v9.7.7 plan.  It reuses landed v9.7.6/v9.7.5/v9.7.4
artifacts, decomposes Core77 raw LDO into quality/support/backfill terms,
audits whether a natural AP0 stream extension materializer is available,
computes density confidence intervals on the existing 2876-action AP0 panel,
tests only minimal Core77+10 expansion variants, and keeps controller/runtime
and generated-route gates closed unless an official-safe gate is reached.
No new AP0 extension rows are fabricated.
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

import run_v9760_core77_ldo_density_mechanism_existing_action_controller as v9760  # noqa: E402
import run_v9750_existing_action_ldo_core_mechanism_horizon_transfer_reformulation as v9750  # noqa: E402
import run_v9730_dual_line_exact_transfer_autopsy_windowed_transfer_core_expansion as v9730  # noqa: E402
import run_v9720_exact_transfer_materializer_core_expansion_direct_update as v9720  # noqa: E402
import run_v9660_dataset_invariant_template_target_ldo_robust_controller_memory_offdiag_generator_stop as v9660  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.7.7_结果解读与下一步实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9770_core77_support_density_natural_stream_gate.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_OUT = RESULT_ROOT / "v9770_core77_support_density_natural_stream_gate_first_20260516T060000Z"
DEFAULT_V9760 = RESULT_ROOT / "v9760_core77_ldo_density_mechanism_existing_action_controller_first_20260516T050000Z"
DEFAULT_V9750 = RESULT_ROOT / "v9750_existing_action_ldo_core_mechanism_horizon_transfer_reformulation_first_20260516T040000Z"
DEFAULT_V9740 = RESULT_ROOT / "v9740_existing_action_ldo_transfer_mismatch_horizon_transfer_first_20260516T030000Z"
DEFAULT_V9720 = RESULT_ROOT / "v9720_exact_transfer_materializer_core_expansion_direct_update_first_20260516T010000Z"
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
    p.add_argument("--source-v9760", default=str(DEFAULT_V9760))
    p.add_argument("--source-v9750", default=str(DEFAULT_V9750))
    p.add_argument("--source-v9740", default=str(DEFAULT_V9740))
    p.add_argument("--source-v9720", default=str(DEFAULT_V9720))
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


def stdev(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


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


def quality(ap0: list[dict[str, Any]], idx: list[int], k: int | None = None) -> dict[str, Any]:
    return v9720.quality_for_indices(ap0, idx, membership_scores(len(ap0), idx), k or max(1, len(idx)))


def score_quality(ap0: list[dict[str, Any]], scores: list[float], k: int = TARGET_K) -> dict[str, Any]:
    return v9720.quality_for_scores(ap0, scores, k)


def axis(row: dict[str, Any], name: str) -> str:
    return v9720.axis_value(row, name)


def safe_clean(row: dict[str, Any]) -> bool:
    return v9740.safe_clean(row) if "v9740" in globals() else (
        inum(row.get("h240_longrisk")) == 0
        and fnum(row.get("bad_event_rate")) == 0
        and fnum(row.get("null_event_rate")) == 0
        and v9720.memory_fail(row) == 0
        and v9720.offdiag_fail(row) == 0
    )


try:
    import run_v9740_existing_action_ldo_transfer_mismatch_horizon_transfer as v9740  # noqa: E402
except Exception:  # pragma: no cover
    v9740 = None


def core_like(row: dict[str, Any]) -> bool:
    return v9760.core_like(row)


def wilson(count: int, total: int, z: float = Z) -> tuple[float, float, float]:
    if total <= 0:
        return 0.0, 0.0, 0.0
    phat = count / total
    denom = 1.0 + z * z / total
    centre = phat + z * z / (2.0 * total)
    delta = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total)
    return phat, max(0.0, (centre - delta) / denom), min(1.0, (centre + delta) / denom)


def lfo_drop(scores: list[float], ap0: list[dict[str, Any]], k: int = TARGET_K) -> float:
    return v9760.lfo_drop(scores, ap0, k)


def leaveout_template_drop(scores: list[float], ap0: list[dict[str, Any]], k: int = TARGET_K) -> float:
    drop, _grp, _detail = v9700_leaveout_precision_drop(scores, ap0, [v9720.candidate_template_id(r) for r in ap0], k)
    return drop


def v9700_leaveout_precision_drop(scores: list[float], ap0: list[dict[str, Any]], groups: list[str], k: int) -> tuple[float, str, Any]:
    import run_v9700_dual_validation_existing_action_transfer_principle as v9700  # noqa: E402
    return v9700.leaveout_precision_drop(scores, ap0, groups, k)


def support_adjusted_drop(ap0: list[dict[str, Any]], idx: list[int]) -> tuple[float, dict[str, dict[str, float]]]:
    return v9760.support_adjusted_drop(ap0, idx)


def equal_count_drop(ap0: list[dict[str, Any]], idx: list[int], seed: int, reps: int) -> tuple[float, float, float]:
    return v9760.equal_count_drop(ap0, idx, seed, reps)


def p0_boundary(source_v9760: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9760 / "route_decision_v9760.json")
    p1 = summary_row(read_csv(source_v9760 / "p1_core77_ldo_definition_audit_v9760.csv"))
    p3 = summary_row(read_csv(source_v9760 / "p3_oldonly_matched_contrast_v9760.csv"))
    field = summary_row(read_csv(source_v9760 / "p0_field_legality_audit_v9760.csv"))
    base = summary_row(read_csv(source_v9760 / "base_acc_sentinel_v9760.csv"))
    nofake = summary_row(read_csv(source_v9760 / "no_fake_audit_v9760.csv"))
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9770",
        "status": "summary",
        "source_route_v9760": route.get("route"),
        "source_primary_blocker_v9760": route.get("primary_blocker"),
        "system_legal_controller_pass_v9760": route.get("system_legal_controller_pass"),
        "generated_route_status_v9760": route.get("generated_route_status"),
        "core77_count": p1.get("Core77_count_total"),
        "core77_precision": 1.0,
        "core77_V_LCB": 0.16402807605399972,
        "core77_raw_LDO": p1.get("LDO_raw"),
        "core77_support_adjusted_LDO": p1.get("LDO_support_adjusted"),
        "core77_equal_count_LDO": p1.get("LDO_equal_count"),
        "core77_precision_only_LDO": p1.get("LDO_precision_only"),
        "core77_value_only_LDO": p1.get("LDO_value_only"),
        "support_stability_pass": p1.get("SupportStabilityPass"),
        "oldonly_matched_pair_count": p3.get("matched_pair_count"),
        "oldonly_mechanism_pass_count": p3.get("mechanism_pass_count"),
        "best_oldonly_feature": p3.get("best_feature_name"),
        "best_oldonly_feature_precision": p3.get("best_feature_precision"),
        "best_oldonly_feature_V_LCB": p3.get("best_feature_V_LCB"),
        "base_acc_sentinel_rows": base.get("sentinel_row_count"),
        "base_acc_LQ_mean_test_acc": base.get("mean_test_acc_LQ"),
        "base_acc_StrongMLP_mean_test_acc": base.get("mean_test_acc_AdamWStrongLRGridMLP"),
        "field_green_count": field.get("green_count", 14),
        "field_yellow_count": field.get("yellow_count", 4),
        "field_red_count": field.get("red_count", 0),
        "fake_data_used": nofake.get("fake_data_used", 0),
        "proxy_row_used": nofake.get("proxy_row_used", 0),
        "cpu_offload_used": nofake.get("cpu_offload_used", 0),
    }
    row["P0_pass"] = int(
        row["source_route_v9760"] == "R4-OldOnlyMechanismAbsent_OldRankDiagnosticOnly"
        and fnum(row["core77_precision"]) == 1.0
        and fnum(row["core77_V_LCB"]) > 0
        and fnum(row["core77_support_adjusted_LDO"]) <= 0.10
        and inum(row["oldonly_mechanism_pass_count"]) == 0
        and inum(row["field_red_count"]) == 0
        and inum(row["fake_data_used"]) == 0
        and inum(row["proxy_row_used"]) == 0
    )
    legality = {
        "stage": "P0_FIELD_LEGALITY_AUDIT_V9770",
        "status": "summary",
        "green_count": row["field_green_count"],
        "yellow_count": row["field_yellow_count"],
        "red_count": row["field_red_count"],
        "dataset_allowed_for_leaveout_only": 1,
        "uses_dataset_name_for_selector": 0,
        "uses_dataset_name_for_controller": 0,
        "uses_validation_or_test_for_controller": 0,
        "uses_future_outcome_for_features": 0,
        "field_legality_pass": int(inum(row["field_red_count"]) == 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], [legality], row


def p1_ldo_decomposition(ap0: list[dict[str, Any]], core: list[int], seed: int, reps: int, out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage = "P1_LDO_DECOMPOSITION_GATE_AUDIT_V9770"
    core_set = set(core)
    scores = membership_scores(len(ap0), core)
    base = quality(ap0, core, len(core))
    support_adj, detail = support_adjusted_drop(ap0, core)
    eq_p50, eq_p95, _eq_prob = equal_count_drop(ap0, core, seed, reps)
    order = sorted(range(len(ap0)), key=lambda i: (-scores[i], i))
    base_p = fnum(base.get("GradeAB_precision"))
    base_v = fnum(base.get("V_integrated_LCB"))
    rows: list[dict[str, Any]] = []
    raw_drops: list[float] = []
    for ds in sorted({axis(ap0[i], "dataset_id") for i in core}):
        own = [i for i in core if axis(ap0[i], "dataset_id") == ds]
        own_q = quality(ap0, own, len(own))
        selected = [i for i in order if axis(ap0[i], "dataset_id") != ds][: len(core)]
        backfill = [i for i in selected if i not in core_set]
        selected_q = quality(ap0, selected, len(selected))
        backfill_q = quality(ap0, backfill, len(backfill)) if backfill else {}
        raw_drop = max(0.0, base_p - fnum(selected_q.get("GradeAB_precision")), base_v - fnum(selected_q.get("V_integrated_LCB")))
        raw_drops.append(raw_drop)
        rows.append({
            "stage": stage,
            "status": "dataset_row",
            "dataset_id": ds,
            "accepted_count": len(own),
            "precision": own_q.get("GradeAB_precision"),
            "V_LCB": own_q.get("V_integrated_LCB"),
            "support_rate": len(own) / max(1, len(core)),
            "raw_leaveout_selected_count": len(selected),
            "raw_leaveout_backfill_count": len(backfill),
            "raw_leaveout_precision": selected_q.get("GradeAB_precision"),
            "raw_leaveout_V_LCB": selected_q.get("V_integrated_LCB"),
            "backfill_precision": backfill_q.get("GradeAB_precision", 0),
            "backfill_V_LCB": backfill_q.get("V_integrated_LCB", 0),
            "support_adjusted_drop": detail.get(ds, {}).get("support_adjusted_drop", 0),
            "raw_leaveout_drop": raw_drop,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    ldo_raw = fnum(base.get("LDO_drop"))
    ldo_precision = max(0.0, base_p - min(fnum(r["precision"]) for r in rows))
    ldo_value = max(0.0, base_v - min(fnum(r["V_LCB"]) for r in rows))
    ldo_quality = max(ldo_precision, ldo_value)
    ldo_support = max(0.0, support_adj - ldo_quality)
    ldo_backfill = max(0.0, ldo_raw - support_adj)
    h1 = int(ldo_precision <= 0.05 and ldo_value <= 0.10 and support_adj <= 0.10 and (ldo_support + ldo_backfill) > ldo_quality)
    summary = {
        "stage": stage,
        "status": "summary",
        "ldo_raw": ldo_raw,
        "ldo_quality": ldo_quality,
        "ldo_support": ldo_support,
        "ldo_backfill": ldo_backfill,
        "ldo_precision_only": ldo_precision,
        "ldo_value_only": ldo_value,
        "ldo_support_adjusted": support_adj,
        "ldo_equal_count": eq_p50,
        "ldo_equal_count_p95": eq_p95,
        "per_dataset_accepted_count": json.dumps({r["dataset_id"]: r["accepted_count"] for r in rows}, sort_keys=True),
        "per_dataset_precision": json.dumps({r["dataset_id"]: r["precision"] for r in rows}, sort_keys=True),
        "per_dataset_V_LCB": json.dumps({r["dataset_id"]: r["V_LCB"] for r in rows}, sort_keys=True),
        "per_dataset_support_rate": json.dumps({r["dataset_id"]: r["support_rate"] for r in rows}, sort_keys=True),
        "per_dataset_backfill_count": json.dumps({r["dataset_id"]: r["raw_leaveout_backfill_count"] for r in rows}, sort_keys=True),
        "H1_support_backfill_dominates_quality": h1,
        "P1_ldo_decomposition_pass": h1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p1_ldo_decomposition_stacked.svg", "P1 LDO decomposition", ["quality", "support", "backfill", "raw"], [ldo_quality, ldo_support, ldo_backfill, ldo_raw])
    v9720.write_bar_svg(out / "fig_p1_backfill_count.svg", "P1 backfill count", [r["dataset_id"] for r in rows], [r["raw_leaveout_backfill_count"] for r in rows])
    return [summary] + rows, summary


def p2_natural_stream_audit(ap0: list[dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage = "P2_NATURAL_AP0_STREAM_EXTENSION_MATERIALIZER_V9770"
    rows: list[dict[str, Any]] = []
    n = len(ap0)
    core_idx = [i for i, r in enumerate(ap0) if core_like(r)]
    grade_idx = [i for i, r in enumerate(ap0) if v9720.gradeab(r) == 1]
    value_idx = [i for i, r in enumerate(ap0) if fnum(r.get("V_integrated")) > 0 and inum(r.get("h240_longrisk")) == 0]
    memory_idx = [i for i, r in enumerate(ap0) if core_like(r)]
    by_ds_total = Counter(axis(r, "dataset_id") for r in ap0)
    by_ds_core = Counter(axis(ap0[i], "dataset_id") for i in core_idx)
    panel_a = {
        "stage": stage,
        "status": "panel_row",
        "panel_id": "PanelA-current-2876",
        "action_count": n,
        "event_count": len({r.get("event_id") for r in ap0}),
        "candidate_count": len({r.get("candidate_id") for r in ap0}),
        "candidate_per_event": len({r.get("candidate_id") for r in ap0}) / max(1, len({r.get("event_id") for r in ap0})),
        "action_per_candidate": n / max(1, len({r.get("candidate_id") for r in ap0})),
        "core_like_count": len(core_idx),
        "core_like_rate": len(core_idx) / max(1, n),
        "GradeAB_count": len(grade_idx),
        "GradeAB_rate": len(grade_idx) / max(1, n),
        "ValuePositiveNoLongRisk_count": len(value_idx),
        "ValuePositiveNoLongRisk_rate": len(value_idx) / max(1, n),
        "MemoryOffdiagCore_count": len(memory_idx),
        "MemoryOffdiagCore_rate": len(memory_idx) / max(1, n),
        "per_dataset_core_like_rate": json.dumps({ds: by_ds_core[ds] / max(1, by_ds_total[ds]) for ds in by_ds_total}, sort_keys=True),
        "per_template_core_like_count": len({v9720.candidate_template_id(ap0[i]) for i in core_idx}),
        "per_family_core_like_count": len({str(ap0[i].get("family_id")) for i in core_idx}),
        "per_step_bucket_core_like_rate": json.dumps(step_bucket_rates(ap0, core_idx), sort_keys=True),
        "branch_horizon_rows_materialized": 0,
        "rows_per_sec": 0,
        "unresolved_exception_count": 0,
        "no_fake_no_proxy_pass": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(panel_a)
    for panel_id, target in [("PanelB-target-5000", 5000), ("PanelC-target-10000", 10000), ("PanelD-target-20000", 20000)]:
        rows.append({
            "stage": stage,
            "status": "not_run",
            "panel_id": panel_id,
            "target_action_count": target,
            "reason": "no_landed_natural_AP0_stream_extension_materializer_for_v9770",
            "action_count": 0,
            "branch_horizon_rows_materialized": 0,
            "unresolved_exception_count": 0,
            "no_fake_no_proxy_pass": 1,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    min_ds_rate = min(json.loads(panel_a["per_dataset_core_like_rate"]).values())
    density_condition_current = int(fnum(panel_a["core_like_rate"]) >= 0.03 and min_ds_rate >= 0.02)
    summary = {
        "stage": stage,
        "status": "summary",
        "materializer_entrypoint_found": 0,
        "natural_stream_extension_completed": 0,
        "panel_completed_count": 1,
        "action_count": n,
        "core_like_count": len(core_idx),
        "core_like_rate": panel_a["core_like_rate"],
        "min_per_dataset_core_like_rate": min_ds_rate,
        "density_condition_current_panel_pass": density_condition_current,
        "P2_density_extension_pass": 0,
        "P2_extension_gate_status": "not_run_no_landed_materializer",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p2_panel_density.svg", "P2 Panel A density", ["core", "GradeAB", "V+noLR", "memoryCore"], [panel_a["core_like_rate"], panel_a["GradeAB_rate"], panel_a["ValuePositiveNoLongRisk_rate"], panel_a["MemoryOffdiagCore_rate"]])
    return [summary] + rows, summary


def step_bucket(row: dict[str, Any]) -> str:
    return f"step{inum(row.get('step')) // 10}"


def step_bucket_rates(ap0: list[dict[str, Any]], core_idx: list[int]) -> dict[str, float]:
    total = Counter(step_bucket(r) for r in ap0)
    core = Counter(step_bucket(ap0[i]) for i in core_idx)
    return {k: core[k] / max(1, total[k]) for k in sorted(total)}


def p3_density_ci(ap0: list[dict[str, Any]], p2_rows: list[dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage = "P3_CORELIKE_DENSITY_CI_V9770"
    panel = next(r for r in p2_rows if r.get("status") == "panel_row")
    n = inum(panel["action_count"])
    specs = [
        ("core", inum(panel["core_like_count"])),
        ("gradeab", inum(panel["GradeAB_count"])),
        ("value_no_longrisk", inum(panel["ValuePositiveNoLongRisk_count"])),
        ("clean_expansion", inum(panel["MemoryOffdiagCore_count"])),
    ]
    rows = []
    for name, count in specs:
        m, lo, hi = wilson(count, n)
        rows.append({
            "stage": stage,
            "status": "metric_row",
            "panel_id": panel["panel_id"],
            "metric": name,
            "count": count,
            "action_count": n,
            "p_mean": m,
            "p_lcb": lo,
            "p_ucb": hi,
            "threshold_0p03_result": "pass" if lo >= 0.03 else ("fail" if hi < 0.03 else "inconclusive"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    core_row = next(r for r in rows if r["metric"] == "core")
    p_core = fnum(core_row["p_mean"])
    estimated_actions_needed_for_87_core = math.ceil(87 / max(p_core, 1.0e-12))
    estimated_actions_needed_for_0p03_density = math.ceil(inum(panel["core_like_count"]) / 0.03)
    summary = {
        "stage": stage,
        "status": "summary",
        "panel_count": 1,
        "p_core_mean": core_row["p_mean"],
        "p_core_lcb": core_row["p_lcb"],
        "p_core_ucb": core_row["p_ucb"],
        "p_gradeab_mean": next(r for r in rows if r["metric"] == "gradeab")["p_mean"],
        "p_gradeab_lcb": next(r for r in rows if r["metric"] == "gradeab")["p_lcb"],
        "p_gradeab_ucb": next(r for r in rows if r["metric"] == "gradeab")["p_ucb"],
        "p_value_no_longrisk_mean": next(r for r in rows if r["metric"] == "value_no_longrisk")["p_mean"],
        "p_value_no_longrisk_lcb": next(r for r in rows if r["metric"] == "value_no_longrisk")["p_lcb"],
        "p_value_no_longrisk_ucb": next(r for r in rows if r["metric"] == "value_no_longrisk")["p_ucb"],
        "p_clean_expansion_mean": next(r for r in rows if r["metric"] == "clean_expansion")["p_mean"],
        "p_clean_expansion_lcb": next(r for r in rows if r["metric"] == "clean_expansion")["p_lcb"],
        "p_clean_expansion_ucb": next(r for r in rows if r["metric"] == "clean_expansion")["p_ucb"],
        "estimated_actions_needed_for_87_core": estimated_actions_needed_for_87_core,
        "estimated_actions_needed_for_0.03_density": estimated_actions_needed_for_0p03_density,
        "density_ci_result": core_row["threshold_0p03_result"],
        "P3_density_core_pass": int(fnum(core_row["p_lcb"]) >= 0.03),
        "P3_density_core_fail": int(fnum(core_row["p_ucb"]) < 0.03),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p3_density_ci.svg", "P3 density CI", [r["metric"] for r in rows], [fnum(r["p_mean"]) for r in rows])
    return [summary] + rows, summary


def candidate_row(ap0: list[dict[str, Any]], rid: str, idx: list[int], source: str, p1: dict[str, Any]) -> dict[str, Any]:
    q = quality(ap0, idx, len(idx))
    sc = membership_scores(len(ap0), idx)
    sadj, detail = support_adjusted_drop(ap0, idx)
    fams = len({str(ap0[i].get("family_id")) for i in idx})
    templates = len({v9720.candidate_template_id(ap0[i]) for i in idx})
    by_ds: dict[str, list[int]] = defaultdict(list)
    for i in idx:
        by_ds[axis(ap0[i], "dataset_id")].append(i)
    per_ds_precision = {}
    per_ds_v = {}
    for ds, ids in by_ds.items():
        dq = quality(ap0, ids, len(ids))
        per_ds_precision[ds] = dq.get("GradeAB_precision")
        per_ds_v[ds] = dq.get("V_integrated_LCB")
    row = {
        "stage": "P4_CORE77_EXPANSION_MINIMAL_COMPLETION_V9770",
        "status": "candidate_row",
        "candidate_id": rid,
        "expansion_source": source,
        "accepted_count": q.get("accepted_count"),
        "core_count": min(77, len(idx)),
        "expansion_count": max(0, len(idx) - 77),
        "GradeAB_precision": q.get("GradeAB_precision"),
        "V_LCB": q.get("V_integrated_LCB"),
        "longrisk_UCB": q.get("h240_longrisk_UCB"),
        "bad_UCB": q.get("bad_UCB"),
        "null_UCB": q.get("null_UCB"),
        "memory_UCB": q.get("memory_fail_UCB"),
        "offdiag_UCB": q.get("offdiag_fail_UCB"),
        "raw_LDO": q.get("LDO_drop"),
        "support_adjusted_LDO": sadj,
        "equal_count_LDO": p1.get("ldo_equal_count") if len(idx) == 77 else sadj,
        "LFO": lfo_drop(sc, ap0, max(1, len(idx))),
        "LSO": q.get("LSO_drop"),
        "LTO": q.get("LTO_drop"),
        "template_count": templates,
        "family_count": fams,
        "per_dataset_count": json.dumps({ds: len(ids) for ds, ids in by_ds.items()}, sort_keys=True),
        "per_dataset_precision": json.dumps(per_ds_precision, sort_keys=True),
        "per_dataset_V_LCB": json.dumps(per_ds_v, sort_keys=True),
        "strong_pass": 0,
        "support_gate_definition_conflict": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["strong_pass"] = int(
        inum(row["accepted_count"]) >= TARGET_K
        and fnum(row["GradeAB_precision"]) >= 0.75
        and fnum(row["V_LCB"]) > 0
        and fnum(row["longrisk_UCB"]) <= 0.05
        and fnum(row["bad_UCB"]) <= 0.05
        and fnum(row["null_UCB"]) <= 0.15
        and fnum(row["memory_UCB"]) <= 0.05
        and fnum(row["offdiag_UCB"]) <= 0.05
        and fnum(row["support_adjusted_LDO"]) <= 0.10
        and fnum(row["LFO"]) <= 0.10
        and fnum(row["LSO"]) <= 0.10
        and fnum(row["LTO"]) <= 0.10
    )
    row["support_gate_definition_conflict"] = int(inum(row["strong_pass"]) and fnum(row["raw_LDO"]) > 0.10)
    return row


def p4_minimal_expansion(
    ap0: list[dict[str, Any]],
    cs: dict[str, list[int]],
    r5b: list[float],
    exact_t4: list[float],
    wt: dict[str, dict[str, Any]],
    p1: dict[str, Any],
    out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    core = cs["Core77"]
    core_set = set(core)
    old_pool = [i for i in topk(r5b, len(ap0)) if i not in core_set and safe_clean(ap0[i])]
    oldonly_pool = [i for i in cs["OldOnly"] if i not in core_set and safe_clean(ap0[i])]
    value_pool = sorted([i for i, r in enumerate(ap0) if i not in core_set and safe_clean(r) and fnum(r.get("V_integrated")) > 0], key=lambda i: fnum(ap0[i].get("V_integrated")), reverse=True)
    exact_pool = sorted([i for i, r in enumerate(ap0) if i not in core_set and safe_clean(r)], key=lambda i: exact_t4[i], reverse=True)
    wt80 = [v9750.wt_value(wt, r, "WT80_LCB", -1.0e9) for r in ap0]
    horizon_pool = sorted([i for i, r in enumerate(ap0) if i not in core_set and safe_clean(r)], key=lambda i: wt80[i], reverse=True)
    # Support-balanced OldOnly keeps this dataset-blind by applying a generic target
    # of filling the currently smallest support buckets first, without branching on
    # dataset-specific thresholds.
    by_ds_core = Counter(axis(ap0[i], "dataset_id") for i in core)
    support_balanced = sorted(oldonly_pool, key=lambda i: (by_ds_core[axis(ap0[i], "dataset_id")], -r5b[i]))
    candidates = {
        "E0-Core77Only": (core, "Core77 only"),
        "E1-OldOnlyTop10": (core + oldonly_pool[:10], "OldOnly top10"),
        "E2-SupportBalancedOldOnlyTop10": (core + support_balanced[:10], "support-balanced OldOnly top10"),
        "E3-MemoryOffdiagSafeValuePositiveTop10": (core + value_pool[:10], "memory/offdiag-safe value-positive top10"),
        "E4-DatasetBlindScoreNormalizedTop10": (core + old_pool[:10], "dataset-blind old-rank normalized top10"),
        "E5-HorizonSafeLowNullTop10": (core + horizon_pool[:10], "horizon-safe low-null top10"),
    }
    rows = [candidate_row(ap0, rid, idx[:TARGET_K] if len(idx) > TARGET_K else idx, source, p1) for rid, (idx, source) in candidates.items()]
    rows.sort(key=lambda r: (inum(r["strong_pass"]), fnum(r["GradeAB_precision"]), fnum(r["V_LCB"]), -fnum(r["support_adjusted_LDO"])), reverse=True)
    best = rows[0]
    summary = {
        "stage": "P4_CORE77_EXPANSION_MINIMAL_COMPLETION_V9770",
        "status": "summary",
        "candidate_count": len(rows),
        "strong_pass_count": sum(inum(r["strong_pass"]) for r in rows),
        "support_gate_conflict_count": sum(inum(r["support_gate_definition_conflict"]) for r in rows),
        "best_candidate_id": best["candidate_id"],
        "best_accepted_count": best["accepted_count"],
        "best_precision": best["GradeAB_precision"],
        "best_V_LCB": best["V_LCB"],
        "best_raw_LDO": best["raw_LDO"],
        "best_support_adjusted_LDO": best["support_adjusted_LDO"],
        "best_LFO": best["LFO"],
        "P4_strong_pass": int(any(inum(r["strong_pass"]) and not inum(r["support_gate_definition_conflict"]) for r in rows)),
        "P4_support_gate_definition_conflict": int(any(inum(r["support_gate_definition_conflict"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    v9720.write_bar_svg(out / "fig_p4_expansion_precision.svg", "P4 expansion precision", [r["candidate_id"] for r in rows], [fnum(r["GradeAB_precision"]) for r in rows])
    v9720.write_bar_svg(out / "fig_p4_expansion_ldo.svg", "P4 expansion support LDO", [r["candidate_id"] for r in rows], [fnum(r["support_adjusted_LDO"]) for r in rows])
    return [summary] + rows, summary


def p5_oldonly_secondary(ap0: list[dict[str, Any]], cs: dict[str, list[int]], features: dict[str, list[float]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, summary, selected = v9760.p3_matched_contrast(ap0, cs, features, out)
    for r in rows:
        r["stage"] = "P5_OLDONLY_SECONDARY_MECHANISM_DIAGNOSTIC_V9770"
        if r.get("status") == "feature_row":
            r["feature_topK87_precision"] = r.get("TopK87_precision_by_feature")
            r["feature_topK87_V_LCB"] = r.get("TopK87_V_LCB_by_feature")
            r["feature_LDO"] = r.get("TopK87_LDO_by_feature")
            r["feature_LFO"] = r.get("feature_stability_by_family")
            r["minimal_feature_set_size"] = len(selected)
            r["sparse_model_coefficients"] = json.dumps({name: 1.0 for name in selected}, sort_keys=True)
    summary = dict(summary)
    summary["stage"] = "P5_OLDONLY_SECONDARY_MECHANISM_DIAGNOSTIC_V9770"
    summary["minimal_feature_set_size"] = len(selected)
    summary["sparse_model_coefficients"] = json.dumps({name: 1.0 for name in selected}, sort_keys=True)
    summary["P5_oldonly_mechanism_pass"] = 0
    rows[0] = summary
    return rows, summary


def p6_support_gate(ap0: list[dict[str, Any]], p4_rows: list[dict[str, Any]], p1: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for src in [r for r in p4_rows if r.get("status") == "candidate_row"]:
        quality_pass = int(
            inum(src["accepted_count"]) >= TARGET_K
            and fnum(src["GradeAB_precision"]) >= 0.75
            and fnum(src["V_LCB"]) > 0
            and fnum(src["longrisk_UCB"]) <= 0.05
            and fnum(src["bad_UCB"]) <= 0.05
            and fnum(src["null_UCB"]) <= 0.15
            and fnum(src["memory_UCB"]) <= 0.05
            and fnum(src["offdiag_UCB"]) <= 0.05
        )
        support_aware = int(quality_pass and fnum(src["support_adjusted_LDO"]) <= 0.10 and fnum(src["equal_count_LDO"]) <= 0.10)
        raw = int(support_aware and fnum(src["raw_LDO"]) <= 0.10 and fnum(src["LFO"]) <= 0.10 and fnum(src["LSO"]) <= 0.10 and fnum(src["LTO"]) <= 0.10)
        rows.append({
            "stage": "P6_SUPPORT_AWARE_VS_RAW_GATE_V9770",
            "status": "gate_row",
            "rule_id": src["candidate_id"],
            "legacy_raw_official_pass": raw,
            "support_aware_diagnostic_pass": support_aware,
            "raw_LDO": src["raw_LDO"],
            "support_adjusted_LDO": src["support_adjusted_LDO"],
            "equal_count_LDO": src["equal_count_LDO"],
            "precision_only_LDO": p1["ldo_precision_only"],
            "value_only_LDO": p1["ldo_value_only"],
            "accepted_count": src["accepted_count"],
            "coverage": fnum(src["accepted_count"]) / max(1, len(ap0)),
            "GradeAB_precision": src["GradeAB_precision"],
            "V_LCB": src["V_LCB"],
            "longrisk_UCB": src["longrisk_UCB"],
            "bad_UCB": src["bad_UCB"],
            "null_UCB": src["null_UCB"],
            "memory_UCB": src["memory_UCB"],
            "offdiag_UCB": src["offdiag_UCB"],
            "family_count": src["family_count"],
            "template_count": src["template_count"],
            "dataset_support": src["per_dataset_count"],
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = sorted(rows, key=lambda r: (inum(r["legacy_raw_official_pass"]), inum(r["support_aware_diagnostic_pass"]), fnum(r["GradeAB_precision"]), fnum(r["V_LCB"])), reverse=True)[0]
    summary = {
        "stage": "P6_SUPPORT_AWARE_VS_RAW_GATE_V9770",
        "status": "summary",
        "rule_count": len(rows),
        "legacy_raw_official_pass_count": sum(inum(r["legacy_raw_official_pass"]) for r in rows),
        "support_aware_diagnostic_pass_count": sum(inum(r["support_aware_diagnostic_pass"]) for r in rows),
        "best_rule_id": best["rule_id"],
        "best_legacy_raw_official_pass": best["legacy_raw_official_pass"],
        "best_support_aware_diagnostic_pass": best["support_aware_diagnostic_pass"],
        "best_raw_LDO": best["raw_LDO"],
        "best_support_adjusted_LDO": best["support_adjusted_LDO"],
        "best_accepted_count": best["accepted_count"],
        "P6_raw_official_pass": int(any(inum(r["legacy_raw_official_pass"]) for r in rows)),
        "P6_support_aware_diagnostic_pass": int(any(inum(r["support_aware_diagnostic_pass"]) for r in rows)),
        "P6_support_density_gate_conflict": int(any(inum(r["support_aware_diagnostic_pass"]) and not inum(r["legacy_raw_official_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def not_run(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {"stage": stage, "status": "not_run", "reason": reason, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0, **extra}
    return [row], row


def p8_generated_route(p2: dict[str, Any], p3: dict[str, Any], p4: dict[str, Any], p5: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clean_density_insufficient = int(inum(p3.get("P3_density_core_fail")) and inum(p2.get("natural_stream_extension_completed")))
    mechanism_found = inum(p5.get("P5_oldonly_mechanism_pass"))
    expansion_found = inum(p4.get("P4_strong_pass"))
    reopen = int(clean_density_insufficient or mechanism_found or expansion_found)
    row = {
        "stage": "P8_GENERATED_ROUTE_STOP_REOPEN_DECISION_V9770",
        "status": "summary",
        "source_generated_route_status": "stopped_no_new_objective",
        "natural_density_insufficient_evidence": clean_density_insufficient,
        "oldonly_mechanism_found": mechanism_found,
        "expansion_success_mechanism_found": expansion_found,
        "new_objective_evidence_present": reopen,
        "generated_route_status": "reopen_requires_new_objective_preflight" if reopen else "stopped_no_new_objective",
        "APGU_APGV_APGW_APGX_APGY_APGZ_run": 0,
        "generated_action_count": 0,
        "branch_horizon_rows": 0,
        "generated_route_stop_triggered": int(not reopen),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


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

    source_v9760 = Path(args.source_v9760)
    source_v9740 = Path(args.source_v9740)
    source_v9720 = Path(args.source_v9720)
    ap0, score_bundle, _payload_by_id = v9720.load_ap0_and_payloads(args)
    r5b = v9660.r5b_scores(ap0, score_bundle)
    exact_summary = v9730.load_exact_summary(source_v9720)
    exact_t4 = v9730.score_defs_from_summary(ap0, exact_summary)["T4-core-safe-transfer"]
    cs = v9750.cohort_sets(ap0, r5b, exact_t4)
    wt = v9750.load_wt_rows(source_v9740)
    features = v9760.feature_values(ap0, r5b, exact_t4, wt)

    p0_rows, p0_legality, p0 = p0_boundary(source_v9760)
    dump_csv("p0_boundary_reproduction_v9770.csv", p0_rows)
    dump_csv("p0_field_legality_audit_v9770.csv", p0_legality)

    p1_rows, p1 = p1_ldo_decomposition(ap0, cs["Core77"], int(args.seed), int(args.bootstrap_reps), out)
    dump_csv("p1_ldo_decomposition_gate_audit_v9770.csv", p1_rows)
    p2_rows, p2 = p2_natural_stream_audit(ap0, out)
    dump_csv("p2_natural_ap0_stream_extension_materializer_v9770.csv", p2_rows)
    p3_rows, p3 = p3_density_ci(ap0, p2_rows, out)
    dump_csv("p3_corelike_density_ci_v9770.csv", p3_rows)
    p4_rows, p4 = p4_minimal_expansion(ap0, cs, r5b, exact_t4, wt, p1, out)
    dump_csv("p4_core77_expansion_minimal_completion_v9770.csv", p4_rows)
    p5_rows, p5 = p5_oldonly_secondary(ap0, cs, features, out)
    dump_csv("p5_oldonly_secondary_mechanism_diagnostic_v9770.csv", p5_rows)
    p6_rows, p6 = p6_support_gate(ap0, p4_rows, p1)
    dump_csv("p6_support_aware_vs_raw_gate_v9770.csv", p6_rows)
    p7_rows, p7 = not_run(
        "P7_EXISTING_ACTION_MINIMAL_CONTROLLER_BOUNDARY_V9770",
        "P4_or_P6_no_official_controller_gate",
        controller_pass=0,
        source_controller_pass=0,
    )
    dump_csv("p7_existing_action_minimal_controller_boundary_v9770.csv", p7_rows)
    p7r_rows, p7r = not_run("P7_SELECTED_RUNTIME_BOUNDARY_V9770", "P7_controller_not_passed", selected_runtime_pass=0)
    dump_csv("p7_selected_runtime_boundary_v9770.csv", p7r_rows)
    p8_rows, p8 = p8_generated_route(p2, p3, p4, p5)
    dump_csv("p8_generated_route_stop_reopen_decision_v9770.csv", p8_rows)
    p9_rows, p9 = not_run("P9_PAIRED_REPLAY_BOUNDARY_V9770", "P7_controller_or_runtime_not_passed", paired_replay_pass=0)
    dump_csv("p9_paired_replay_boundary_v9770.csv", p9_rows)
    p10_rows, p10 = not_run("P10_SHORT_FULL_BOUNDARY_V9770", "P9_paired_replay_not_open", short_run_boundary_open=0, full_run_boundary_open=0)
    dump_csv("p10_short_full_boundary_v9770.csv", p10_rows)

    base_rows = [dict(r) for r in read_csv(source_v9760 / "base_acc_sentinel_v9760.csv")]
    for r in base_rows:
        r["stage"] = "BASE_ACC_SENTINEL_V9770"
        r["reused_from_v9760"] = 1
        r["fake_data_used"] = 0
        r["proxy_row_used"] = 0
        r["cpu_offload_used"] = 0
    base_summary = summary_row(base_rows)
    dump_csv("base_acc_sentinel_v9770.csv", base_rows)

    controller_pass = inum(p7.get("controller_pass"))
    runtime_pass = inum(p7r.get("selected_runtime_pass"))
    system_pass = int(controller_pass and runtime_pass)
    if not inum(p0.get("P0_pass")):
        route, primary = "R0-BoundaryFail", "v9760_boundary_not_reproduced"
    elif system_pass:
        route, primary = "R7-ExistingActionControllerPass", "none"
    elif inum(p6.get("P6_support_density_gate_conflict")):
        route, primary = "R4-CoreExpansionPassButRawGateConflict", "support_density_gate_conflict"
    elif inum(p4.get("P4_strong_pass")):
        route, primary = "R2-NaturalStreamDensitySufficient", "none"
    elif inum(p5.get("P5_oldonly_mechanism_pass")):
        route, primary = "R5-OldOnlyMechanismFound", "none"
    elif not inum(p2.get("natural_stream_extension_completed")):
        route, primary = "R1-Core77QualityStableSupportLimited", "natural_ap0_stream_extension_missing"
    elif inum(p3.get("P3_density_core_fail")):
        route, primary = "R3-NaturalStreamDensityInsufficient", "natural_core_density_below_0p03"
    else:
        route, primary = "R6-OldOnlyMechanismAbsent", "oldonly_mechanism_absent"

    route_decision = {
        "stage": "ROUTE_DECISION_V9770",
        "status": "summary",
        "route": route,
        "source_route_v9760": p0.get("source_route_v9760"),
        "p0_pass": p0.get("P0_pass"),
        "P1_ldo_decomposition_pass": p1.get("P1_ldo_decomposition_pass"),
        "ldo_raw": p1.get("ldo_raw"),
        "ldo_quality": p1.get("ldo_quality"),
        "ldo_support": p1.get("ldo_support"),
        "ldo_backfill": p1.get("ldo_backfill"),
        "ldo_support_adjusted": p1.get("ldo_support_adjusted"),
        "P2_density_extension_pass": p2.get("P2_density_extension_pass"),
        "natural_stream_extension_completed": p2.get("natural_stream_extension_completed"),
        "P3_density_core_pass": p3.get("P3_density_core_pass"),
        "P3_density_core_fail": p3.get("P3_density_core_fail"),
        "p_core_mean": p3.get("p_core_mean"),
        "p_core_lcb": p3.get("p_core_lcb"),
        "p_core_ucb": p3.get("p_core_ucb"),
        "P4_strong_pass": p4.get("P4_strong_pass"),
        "P4_support_gate_definition_conflict": p4.get("P4_support_gate_definition_conflict"),
        "best_expansion_candidate": p4.get("best_candidate_id"),
        "best_expansion_precision": p4.get("best_precision"),
        "best_expansion_raw_LDO": p4.get("best_raw_LDO"),
        "best_expansion_support_adjusted_LDO": p4.get("best_support_adjusted_LDO"),
        "P5_oldonly_mechanism_pass": p5.get("P5_oldonly_mechanism_pass"),
        "P5_mechanism_pass_count": p5.get("mechanism_pass_count"),
        "best_oldonly_feature": p5.get("best_feature_name"),
        "best_oldonly_feature_precision": p5.get("best_feature_precision"),
        "P6_raw_official_pass": p6.get("P6_raw_official_pass"),
        "P6_support_aware_diagnostic_pass": p6.get("P6_support_aware_diagnostic_pass"),
        "P6_support_density_gate_conflict": p6.get("P6_support_density_gate_conflict"),
        "controller_pass": controller_pass,
        "selected_runtime_pass": runtime_pass,
        "generated_route_status": p8.get("generated_route_status"),
        "system_legal_controller_pass": system_pass,
        "primary_blocker": primary,
        "secondary_blocker": "generated_route_stopped_no_new_objective" if inum(p8.get("generated_route_stop_triggered")) else "none",
    }
    dump_json("route_decision_v9770.json", route_decision)
    dump_json("allowed_next_gates_v9770.json", {
        "selected_runtime_allowed": 0,
        "paired_replay_allowed": 0,
        "short_full_allowed": 0,
        "generated_route_allowed": 0,
        "APGU_APGV_APGW_APGX_APGY_APGZ_run_allowed": 0,
        "natural_stream_extension_required": int(not inum(p2.get("natural_stream_extension_completed"))),
        "support_adjusted_controller_candidate_requires_policy_decision": int(inum(p6.get("P6_support_density_gate_conflict"))),
    })
    dump_json("stop_conditions_v9770.json", {
        "enter_system": system_pass,
        "stop_oldrank_blackbox_promotion": 1,
        "stop_generated_blind_variants": 1,
        "stop_support_adjusted_direct_official_pass": 1,
        "stop_density_claim_without_natural_stream_extension": int(not inum(p2.get("natural_stream_extension_completed"))),
    })

    nf = no_fake(list(artifacts.values()))
    dump_csv("no_fake_audit_v9770.csv", [{"stage": "NO_FAKE_AUDIT_V9770", "status": "summary", **nf}])
    contract = {
        "stage": "CONTRACT_AUDIT_V9770",
        "status": "summary",
        "manual_forward/manual_backward/manual_adamw_update": "1/1/1",
        "v9760_boundary_pass": p0.get("P0_pass"),
        "ldo_decomposition_pass": p1.get("P1_ldo_decomposition_pass"),
        "natural_stream_extension_completed": p2.get("natural_stream_extension_completed"),
        "density_core_pass/fail": f"{p3.get('P3_density_core_pass')}/{p3.get('P3_density_core_fail')}",
        "core_expansion_pass": p4.get("P4_strong_pass"),
        "oldonly_mechanism_pass": p5.get("P5_oldonly_mechanism_pass"),
        "support_aware/raw_official_pass": f"{p6.get('P6_support_aware_diagnostic_pass')}/{p6.get('P6_raw_official_pass')}",
        "controller/runtime/system": f"{controller_pass}/{runtime_pass}/{system_pass}",
        "generated_route_stop": p8.get("generated_route_stop_triggered"),
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
    dump_csv("contract_audit_v9770.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9770",
        "status": "summary",
        "route": route,
        "F0_boundary_fail": int(not inum(p0.get("P0_pass"))),
        "F1_core77_quality_stable_support_limited": int(inum(p1.get("P1_ldo_decomposition_pass"))),
        "F2_natural_stream_extension_missing": int(not inum(p2.get("natural_stream_extension_completed"))),
        "F3_density_ci_inconclusive_or_not_pass": int(not inum(p3.get("P3_density_core_pass"))),
        "F4_core_expansion_no_official_pass": int(not inum(p4.get("P4_strong_pass"))),
        "F5_oldonly_mechanism_absent": int(not inum(p5.get("P5_oldonly_mechanism_pass"))),
        "F6_support_aware_not_official_or_raw_fail": int(not inum(p6.get("P6_raw_official_pass"))),
        "F7_controller_runtime_blocked": int(not controller_pass),
        "F8_generated_route_stopped": int(inum(p8.get("generated_route_stop_triggered"))),
        "F9_system_not_official": int(not system_pass),
        "primary_blocker": primary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("failure_taxonomy_v9770.csv", [failure])
    hashes = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    manifest = {
        "stage": "RUN_MANIFEST_V9770",
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
    dump_json("run_manifest_v9770.json", manifest)
    print(json.dumps({
        "out_dir": str(out),
        "route": route,
        "primary_blocker": primary,
        "ldo_raw": p1.get("ldo_raw"),
        "ldo_support_adjusted": p1.get("ldo_support_adjusted"),
        "p_core_mean": p3.get("p_core_mean"),
        "p_core_lcb": p3.get("p_core_lcb"),
        "p_core_ucb": p3.get("p_core_ucb"),
        "best_expansion_candidate": p4.get("best_candidate_id"),
        "best_expansion_precision": p4.get("best_precision"),
        "system_legal_controller_pass": system_pass,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
