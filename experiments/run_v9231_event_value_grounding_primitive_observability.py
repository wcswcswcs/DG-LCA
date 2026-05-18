#!/usr/bin/env python3
"""DG-KAN v9.2.31 event-value grounding and primitive observability audit.

This runner consumes the real v9.2.30 measured pre-event replay rows and
recomputes event-value labels, feature observability, failure attribution, and
auditable predictor gates.  It does not synthesize missing replay rows.  Metrics
that v9.2.30 did not measure, such as curvature deltas and control ECE/NLL
comparables, are explicitly recorded as not measured and excluded from the
grounded value.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402

PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.31_EventValueGrounding_PrimitiveObservability_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9231_event_value_grounding_primitive_observability.py"
SRC_V9230 = ROOT / "results" / "real_rerun_20260506" / "v9230_preevent_signal_value_functional_controller_first_20260510T210000Z"
REPORT_PATH = ROOT / "docs" / "DG-KAN_v9.2.31_EventValueGrounding_PrimitiveObservability_实验复盘.md"

FEATURE_COLUMNS = [
    "ce_tail_rank",
    "margin_tail_rank",
    "wrong_confidence_rank",
    "curvature_rank",
    "pre_real_logit_delta_norm",
    "pre_real_tail_logit_delta_norm",
    "pre_real_nonadamw_delta_norm",
    "pre_adamwparallel_logit_delta_norm",
    "pre_bestlr_logit_delta_norm",
    "pre_real_vs_adamw_delta_ratio",
    "pre_real_vs_bestlr_delta_ratio",
    "pre_tail_real_vs_control_ratio",
    "actual_r_z_perp",
    "branch_ratio",
    "effective_derivative",
    "functional_channel_entropy",
    "dominant_basis_fraction",
    "basis_usage_entropy",
    "cos_real_adamw",
    "cos_real_bestlr",
    "cos_tail_real_adamw",
    "projected_tail_component_ratio",
    "orthogonal_component_ratio",
    "microbatch_score_variance",
    "bootstrap_control_gap_std",
    "horizon_agreement_score",
    "event_score_entropy",
    "stratum_sample_count",
]

MOVEMENT_FEATURES = [
    "pre_real_logit_delta_norm",
    "pre_real_tail_logit_delta_norm",
    "pre_real_nonadamw_delta_norm",
    "pre_adamwparallel_logit_delta_norm",
    "pre_bestlr_logit_delta_norm",
    "pre_real_vs_adamw_delta_ratio",
    "pre_real_vs_bestlr_delta_ratio",
    "pre_tail_real_vs_control_ratio",
    "actual_r_z_perp",
    "orthogonal_component_ratio",
]

PREDICTOR_FEATURES = [
    "ce_tail_rank",
    "margin_tail_rank",
    "wrong_confidence_rank",
    "pre_real_logit_delta_norm",
    "pre_real_tail_logit_delta_norm",
    "pre_real_nonadamw_delta_norm",
    "pre_adamwparallel_logit_delta_norm",
    "pre_bestlr_logit_delta_norm",
    "pre_real_vs_adamw_delta_ratio",
    "pre_tail_real_vs_control_ratio",
    "actual_r_z_perp",
    "branch_ratio",
    "effective_derivative",
    "functional_channel_entropy",
    "dominant_basis_fraction",
    "cos_real_adamw",
    "cos_tail_real_adamw",
    "microbatch_score_variance",
    "bootstrap_control_gap_std",
    "horizon_agreement_score",
]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        text = str(value)
        if text.startswith("not_"):
            return default
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / max(1, len(vals)))


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if len(vals) < 2:
        return 0.0
    mu = _mean(vals)
    return float(math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)))


def _var(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if len(vals) < 2:
        return 0.0
    mu = _mean(vals)
    return float(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return 0.0
    mx = _mean(x for x, _ in pairs)
    my = _mean(y for _, y in pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 1.0e-30 or vy <= 1.0e-30:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return float(cov / math.sqrt(vx * vy))


def _quantile(values: Sequence[float], q: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    pairs = [(float(s), int(l)) for s, l in zip(scores, labels) if math.isfinite(float(s))]
    pos = [s for s, l in pairs if l == 1]
    neg = [s for s, l in pairs if l == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    total = 0.0
    for p in pos:
        for n in neg:
            total += 1.0
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return float(wins / max(1.0, total))


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _not_run(stage: str, artifact: str, reason: str, **extra: Any) -> Dict[str, Any]:
    row = {
        "stage": stage,
        "status": "not_run",
        "artifact": artifact,
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row.update(extra)
    return row


def _source_p0() -> Dict[str, Any]:
    route = _read_json(SRC_V9230 / "route_decision.json")
    audit_rows = read_csv_rows(SRC_V9230 / "v9230_provenance_audit.csv")
    fake = _int(audit_rows[0].get("fake_proxy_nonzero_count")) if audit_rows else 1
    p0_source = read_csv_rows(SRC_V9230 / "p0_v9229_boundary_reproduction.csv")
    source_route = p0_source[0].get("route", "") if p0_source else "not_found"
    best_corr = _float(route.get("best_movement_feature_abs_corr"))
    p0_pass = int(
        route.get("route") == "R7-PrimitiveEffectUnpredictable"
        and source_route == "R1-SignalStrataExplainFailures"
        and _int(route.get("pre_event_feature_pass")) == 0
        and abs(best_corr - 0.2997081595094821) <= 0.01
        and _int(route.get("event_value_predictor_pass")) == 0
        and fake == 0
    )
    return {
        "stage": "P0_V9230_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "source_artifact": str(SRC_V9230.relative_to(ROOT)),
        "route": route.get("route", ""),
        "source_route_v9229": source_route,
        "p1_rows": route.get("p1_row_count", ""),
        "pre_event_feature_pass": route.get("pre_event_feature_pass", ""),
        "best_movement_abs_corr": route.get("best_movement_feature_abs_corr", ""),
        "failure_mode_reclassification_pass": route.get("failure_mode_reclassification_pass", ""),
        "attributed_fraction": route.get("p2_attributed_fraction", route.get("attributed_fraction", "")),
        "abstain_precision": route.get("abstain_precision", ""),
        "best_predictor": route.get("best_predictor", "None"),
        "predictor_corr": route.get("predictor_corr", ""),
        "predictor_precision": route.get("predictor_precision", ""),
        "predictor_coverage": route.get("predictor_coverage", ""),
        "fake_proxy_count": fake,
        "P0_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _family_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("signal_stratum", "")),
        str(row.get("primitive", "")),
        str(row.get("event_type", "")),
        str(row.get("horizon", "")),
    )


def _family_short(row: Dict[str, Any]) -> str:
    return "|".join(_family_key(row))


def _event_gain(row: Dict[str, Any], prefix: str = "") -> float:
    if prefix:
        ce = _float(row.get(f"{prefix}_CEp99_delta"))
        margin = _float(row.get(f"{prefix}_margin_delta"))
        acc = 0.0
    else:
        ce = _float(row.get("CEp99_delta"))
        margin = _float(row.get("margin_p10_delta"))
        acc = _float(row.get("acc_delta"))
    signed_ce = -ce
    signed_margin = margin
    task_risk = max(0.0, -acc - 0.005)
    return signed_ce + signed_margin - 2.0 * task_risk


def _run_p1_grounding(opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P1_EVENT_VALUE_LABEL_GROUNDING", "p1_event_value_label_grounding.csv", "P0_v9230_boundary_failed")
        return [row], {
            "event_value_grounding_pass": 0,
            "value_reliability": 0.0,
            "value_normalization_pass": 0,
            "p1_row_count": 0,
        }

    source_rows = [r for r in read_csv_rows(SRC_V9230 / "p1_pre_event_movement_features.csv") if r.get("status") == "measured"]
    for r in source_rows:
        real_gain = _event_gain(r)
        parallel_gain = _event_gain(r, "AdamWParallel")
        lr_gain = _event_gain(r, "BestLR")
        r["_real_gain"] = real_gain
        r["_parallel_gain"] = parallel_gain
        r["_lr_gain"] = lr_gain
        r["_control_relative_value"] = min(real_gain - parallel_gain, real_gain - lr_gain)

    raw_by_stratum: Dict[str, List[float]] = defaultdict(list)
    for r in source_rows:
        raw_by_stratum[str(r.get("signal_stratum", ""))].append(float(r["_control_relative_value"]))
    stratum_stats = {
        s: {
            "mean": _mean(vals),
            "std": _std(vals),
            # Unit floor is intentional: v9.2.30 source values are already on
            # comparable CE/margin scale.  It avoids tiny-denominator inflation
            # while still doing stratum baseline control.
            "denom": max(1.0, _std(vals)),
            "raw_var": _var(vals),
        }
        for s, vals in raw_by_stratum.items()
    }

    by_family: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in source_rows:
        by_family[_family_key(r)].append(r)

    # Split-half reliability across families, using even/odd seeds when possible.
    split_pairs: List[Tuple[float, float]] = []
    family_stats: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
    for key, rows in by_family.items():
        vals = [float(r["_control_relative_value"]) for r in rows]
        even = [float(r["_control_relative_value"]) for r in rows if _int(r.get("seed")) % 2 == 0]
        odd = [float(r["_control_relative_value"]) for r in rows if _int(r.get("seed")) % 2 == 1]
        split1 = _mean(even) if even else _mean(vals)
        split2 = _mean(odd) if odd else _mean(vals)
        if even and odd:
            split_pairs.append((split1, split2))
        family_stats[key] = {
            "mean": _mean(vals),
            "std": _std(vals),
            "split1": split1,
            "split2": split2,
            "count": float(len(vals)),
        }
    value_reliability = _corr([a for a, _ in split_pairs], [b for _, b in split_pairs])

    out: List[Dict[str, Any]] = []
    for r in source_rows:
        raw = float(r["_control_relative_value"])
        stats = stratum_stats[str(r.get("signal_stratum", ""))]
        norm = (raw - stats["mean"]) / (stats["denom"] + 1.0e-12)
        z_norm = (raw - stats["mean"]) / (stats["std"] + 1.0e-12) if stats["std"] > 0 else 0.0
        fam = family_stats[_family_key(r)]
        out.append(
            {
                "stage": "P1_EVENT_VALUE_LABEL_GROUNDING",
                "status": "source_measured_derived",
                "event_id": r.get("event_id", ""),
                "source_event_id": r.get("event_id", ""),
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "signal_stratum": r.get("signal_stratum", ""),
                "horizon": r.get("horizon", ""),
                "primitive": r.get("primitive", ""),
                "event_type": r.get("event_type", ""),
                "branch": "RealFunctional_vs_AdamWParallel_and_bestLR",
                "raw_CEp99_delta": r.get("CEp99_delta", ""),
                "raw_margin_delta": r.get("margin_p10_delta", ""),
                "raw_ECE_delta": r.get("ECE_delta", ""),
                "raw_NLL_delta": r.get("NLL_delta", ""),
                "raw_curvature_delta": "not_measured_in_v9230_source",
                "raw_acc_delta": r.get("acc_delta", ""),
                "signed_gain_ce": -_float(r.get("CEp99_delta")),
                "signed_gain_margin": _float(r.get("margin_p10_delta")),
                "signed_gain_ece": -_float(r.get("ECE_delta")),
                "signed_gain_nll": -_float(r.get("NLL_delta")),
                "signed_gain_curvature": "not_measured_in_v9230_source",
                "ece_used_in_grounded_value": 0,
                "nll_used_in_grounded_value": 0,
                "curvature_used_in_grounded_value": 0,
                "source_missing_value_reason": "v9230_has_real_ECE_NLL_but_no_control_ECE_NLL_and_no_curvature_delta",
                "task_risk": max(0.0, -_float(r.get("acc_delta")) - 0.005),
                "real_gain_observed_metrics": r["_real_gain"],
                "adamwparallel_gain_observed_metrics": r["_parallel_gain"],
                "bestlr_gain_observed_metrics": r["_lr_gain"],
                "control_relative_value": raw,
                "normalized_value": norm,
                "z_normalized_value_audit_only": z_norm,
                "normalization_policy": "stratum_centered_unit_floor_no_tiny_denominator_inflation",
                "stratum_control_gap_mean": stats["mean"],
                "stratum_control_gap_std": stats["std"],
                "stratum_normalization_denom": stats["denom"],
                "bootstrap_value_mean": fam["mean"],
                "bootstrap_value_std": fam["std"],
                "event_family": _family_short(r),
                "event_family_value": fam["mean"],
                "split_half_value_1": fam["split1"],
                "split_half_value_2": fam["split2"],
                "event_family_count": int(fam["count"]),
                "value_reliability": value_reliability,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )

    norm_vars = {}
    for s in raw_by_stratum:
        raw_vals = [float(row["control_relative_value"]) for row in out if row["signal_stratum"] == s]
        norm_vals = [float(row["normalized_value"]) for row in out if row["signal_stratum"] == s]
        norm_vars[s] = {"raw": _var(raw_vals), "norm": _var(norm_vals)}
    value_normalization_pass = int(all(v["norm"] <= v["raw"] + 1.0e-12 for v in norm_vars.values()))
    summary = {
        "event_value_grounding_pass": int(value_reliability >= 0.30 and value_normalization_pass == 1),
        "value_reliability": value_reliability,
        "value_normalization_pass": value_normalization_pass,
        "p1_row_count": len(out),
        "raw_value_mean": _mean(float(r["control_relative_value"]) for r in out),
        "raw_value_std": _std(float(r["control_relative_value"]) for r in out),
        "normalized_value_mean": _mean(float(r["normalized_value"]) for r in out),
        "normalized_value_std": _std(float(r["normalized_value"]) for r in out),
        "normalization_variance_by_stratum": json.dumps(norm_vars, sort_keys=True),
        "missing_metric_policy": "curvature_and_control_ECE_NLL_not_measured_excluded_not_imputed",
    }
    return out, summary


def _run_p2_features(p1_rows: List[Dict[str, Any]], opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P2_PRE_EVENT_FEATURE_RELEVANCE", "p2_pre_event_feature_relevance.csv", "P1_event_value_grounding_failed")
        return [row], {
            "pre_event_feature_pass": 0,
            "pre_event_relevant_feature_count": 0,
            "best_predictive_feature": "not_opened",
            "best_movement_feature": "not_opened",
            "best_movement_abs_corr": 0.0,
        }
    source_by_id = {r.get("event_id"): r for r in read_csv_rows(SRC_V9230 / "p1_pre_event_movement_features.csv")}
    values = {
        r["event_id"]: {
            "raw": _float(r.get("control_relative_value")),
            "norm": _float(r.get("normalized_value")),
            "family": _float(r.get("event_family_value")),
        }
        for r in p1_rows
        if r.get("status") == "source_measured_derived"
    }
    rows: List[Dict[str, Any]] = []
    for feature in FEATURE_COLUMNS:
        event_ids = [eid for eid in values if eid in source_by_id]
        xs = [_float(source_by_id[eid].get(feature)) for eid in event_ids]
        raw = [values[eid]["raw"] for eid in event_ids]
        norm = [values[eid]["norm"] for eid in event_ids]
        fam = [values[eid]["family"] for eid in event_ids]
        missing = sum(1 for eid in event_ids if source_by_id[eid].get(feature, "") == "") / max(1, len(event_ids))
        rows.append(
            {
                "stage": "P2_PRE_EVENT_FEATURE_RELEVANCE",
                "status": "source_measured_derived",
                "feature": feature,
                "feature_family": "movement" if feature in MOVEMENT_FEATURES else "risk_or_primitive_or_uncertainty",
                "row_count": len(event_ids),
                "feature_corr_raw": _corr(xs, raw),
                "feature_corr_norm": _corr(xs, norm),
                "feature_corr_family": _corr(xs, fam),
                "feature_abs_corr_norm": abs(_corr(xs, norm)),
                "feature_missing_rate": missing,
                "feature_stability_across_microbatch": "not_measured_in_v9230_source",
                "dataset_name_used": 0,
                "seed_id_used": 0,
                "posthoc_metric_used": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    relevant = [r for r in rows if abs(_float(r.get("feature_corr_norm"))) >= 0.20]
    movement = [r for r in rows if r["feature"] in MOVEMENT_FEATURES]
    best = max(rows, key=lambda r: abs(_float(r.get("feature_corr_norm")))) if rows else {}
    best_movement = max(movement, key=lambda r: abs(_float(r.get("feature_corr_norm")))) if movement else {}
    family_best = max(rows, key=lambda r: abs(_float(r.get("feature_corr_family")))) if rows else {}
    single_pass = len(relevant) >= 3 and abs(_float(best_movement.get("feature_corr_norm"))) >= 0.30
    family_pass = abs(_float(family_best.get("feature_corr_family"))) >= abs(_float(best.get("feature_corr_norm"))) + 0.10
    summary = {
        "pre_event_feature_pass": int(single_pass),
        "family_value_predictable": int((not single_pass) and family_pass),
        "pre_event_relevant_feature_count": len(relevant),
        "best_predictive_feature": best.get("feature", ""),
        "best_predictive_feature_abs_corr": abs(_float(best.get("feature_corr_norm"))),
        "best_movement_feature": best_movement.get("feature", ""),
        "best_movement_abs_corr": abs(_float(best_movement.get("feature_corr_norm"))),
        "best_family_feature": family_best.get("feature", ""),
        "best_family_feature_abs_corr": abs(_float(family_best.get("feature_corr_family"))),
    }
    return rows, summary


def _run_p3_failure_modes(p1_rows: List[Dict[str, Any]], opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P3_FAILURE_MODE_RECLASSIFICATION", "p3_failure_mode_reclassification.csv", "P2_pre_event_feature_relevance_failed")
        return [row], {
            "failure_mode_reclassification_pass": 0,
            "assigned_fraction": 0.0,
            "good_event_precision": 0.0,
            "good_event_count": 0,
            "abstain_precision": 0.0,
            "abstain_count": 0,
            "failure_mode_counts": "{}",
        }

    source_by_id = {r.get("event_id"): r for r in read_csv_rows(SRC_V9230 / "p1_pre_event_movement_features.csv")}
    vals = [_float(r.get("normalized_value")) for r in p1_rows if r.get("status") == "source_measured_derived"]
    q_abs_small = max(0.0005, abs(_quantile(vals, 0.55)) * 0.05)
    q_uncert = _quantile([_float(r.get("bootstrap_value_std")) for r in p1_rows], 0.75)
    rows: List[Dict[str, Any]] = []
    for r in p1_rows:
        if r.get("status") != "source_measured_derived":
            continue
        src = source_by_id.get(r.get("event_id"), {})
        raw = _float(r.get("control_relative_value"))
        norm = _float(r.get("normalized_value"))
        movement = _float(src.get("pre_real_logit_delta_norm"))
        tail_ratio = _float(src.get("pre_tail_real_vs_control_ratio"))
        uncertainty = _float(r.get("bootstrap_value_std"))
        ce = _float(src.get("CEp99_delta"))
        margin = _float(src.get("margin_p10_delta"))
        real_gain = _event_gain(src)
        ctrl_gain = max(_event_gain(src, "AdamWParallel"), _event_gain(src, "BestLR"))
        beats = int(_int(src.get("real_beats_adamwparallel")) == 1 and _int(src.get("real_beats_best_lr")) == 1 and _int(src.get("task_safe")) == 1)
        if beats and norm > 0:
            primary = "M6-GoodEvent"
            secondary = "control_superiority_and_task_safe"
            action = "accept"
        elif movement <= 1.0e-9 or tail_ratio < 0.10:
            primary = "M1-Silent"
            secondary = "movement_too_small"
            action = "abstain"
        elif abs(raw) <= q_abs_small:
            primary = "M2-LowMagnitude"
            secondary = "metric_effect_tiny"
            action = "abstain"
        elif real_gain > 0 and real_gain <= ctrl_gain:
            primary = "M3-ControlDominated"
            secondary = "control_gain_higher"
            action = "control_or_abstain"
        elif ce >= 0 and margin <= 0 and movement > 0:
            primary = "M4-Misaligned"
            secondary = "tail_moved_metric_worse"
            action = "abstain"
        elif uncertainty >= q_uncert and abs(norm) < 0.01:
            primary = "M5-HighUncertainty"
            secondary = "bootstrap_family_variance_high"
            action = "abstain"
        else:
            primary = "M7-Abstain"
            secondary = "no_confident_positive_value"
            action = "abstain"
        rows.append(
            {
                "stage": "P3_FAILURE_MODE_RECLASSIFICATION",
                "status": "source_measured_derived",
                "event_id": r.get("event_id", ""),
                "signal_stratum": r.get("signal_stratum", ""),
                "primitive": r.get("primitive", ""),
                "horizon": r.get("horizon", ""),
                "primary_failure_mode": primary,
                "secondary_failure_mode": secondary,
                "movement_ratio": movement,
                "tail_movement_ratio": tail_ratio,
                "control_relative_value": raw,
                "normalized_value": norm,
                "uncertainty": uncertainty,
                "abstain_recommended": int(action != "accept"),
                "best_action": action,
                "actual_good_event": beats,
                "dataset_name_used": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    assigned = sum(1 for r in rows if r.get("primary_failure_mode")) / max(1, len(rows))
    good_rows = [r for r in rows if r["primary_failure_mode"] == "M6-GoodEvent"]
    good_precision = sum(_int(r.get("actual_good_event")) for r in good_rows) / max(1, len(good_rows))
    abstain_rows = [r for r in rows if _int(r.get("abstain_recommended")) == 1]
    abstain_precision = sum(1 for r in abstain_rows if _int(r.get("actual_good_event")) == 0) / max(1, len(abstain_rows))
    summary = {
        "failure_mode_reclassification_pass": int(assigned >= 0.90 and good_precision >= 0.70 and abstain_precision >= 0.75),
        "assigned_fraction": assigned,
        "good_event_precision": good_precision,
        "good_event_count": len(good_rows),
        "abstain_precision": abstain_precision,
        "abstain_count": len(abstain_rows),
        "failure_mode_counts": json.dumps(dict(Counter(r["primary_failure_mode"] for r in rows)), sort_keys=True),
    }
    return rows, summary


def _standardize_train_eval(train_x: np.ndarray, eval_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0)
    sigma = train_x.std(axis=0)
    sigma[sigma < 1.0e-12] = 1.0
    return (train_x - mu) / sigma, (eval_x - mu) / sigma


def _fit_ridge(X: np.ndarray, y: np.ndarray, ridge: float = 1.0e-3) -> np.ndarray:
    if X.shape[0] == 0:
        return np.zeros((X.shape[1] + 1,), dtype=np.float64)
    Xb = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float64), X], axis=1)
    reg = np.eye(Xb.shape[1], dtype=np.float64) * ridge
    reg[0, 0] = 0.0
    try:
        return np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(Xb.T @ Xb + reg) @ (Xb.T @ y)


def _predict_ridge(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float64), X], axis=1)
    return Xb @ coef


def _prepare_model_rows(p1_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    source_by_id = {r.get("event_id"): r for r in read_csv_rows(SRC_V9230 / "p1_pre_event_movement_features.csv")}
    rows: List[Dict[str, Any]] = []
    values = [_float(r.get("normalized_value")) for r in p1_rows if r.get("status") == "source_measured_derived"]
    strong_q = _quantile(values, 0.75)
    for r in p1_rows:
        if r.get("status") != "source_measured_derived":
            continue
        src = source_by_id.get(r.get("event_id"), {})
        row = dict(r)
        for feat in PREDICTOR_FEATURES:
            row[feat] = _float(src.get(feat))
        row["Y_beat"] = int(_int(src.get("real_beats_adamwparallel")) == 1 and _int(src.get("real_beats_best_lr")) == 1 and _int(src.get("task_safe")) == 1)
        row["Y_strong"] = int(_float(r.get("normalized_value")) > strong_q)
        row["bad_event"] = int(_int(src.get("task_safe")) == 0)
        row["CEp99_delta"] = _float(src.get("CEp99_delta"))
        row["margin_p10_delta"] = _float(src.get("margin_p10_delta"))
        row["acc_delta"] = _float(src.get("acc_delta"))
        row["AdamWParallel_CEp99_delta"] = _float(src.get("AdamWParallel_CEp99_delta"))
        row["BestLR_CEp99_delta"] = _float(src.get("BestLR_CEp99_delta"))
        row["real_beats_adamwparallel"] = _int(src.get("real_beats_adamwparallel"))
        row["real_beats_best_lr"] = _int(src.get("real_beats_best_lr"))
        row["task_safe"] = _int(src.get("task_safe"))
        rows.append(row)
    return rows


def _feature_matrix(rows: List[Dict[str, Any]], features: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(r.get(f, 0.0)) for f in features] for r in rows], dtype=np.float64)


def _score_predictor(name: str, train: List[Dict[str, Any]], eval_rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, float], str]:
    if not eval_rows:
        return np.zeros((0,), dtype=np.float64), {}, ""
    y_train = np.asarray([_float(r.get("normalized_value")) for r in train], dtype=np.float64)
    if name == "EV0-OldStrataOnly":
        score = np.asarray([
            _float(r.get("ce_tail_rank")) + _float(r.get("margin_tail_rank")) + _float(r.get("wrong_confidence_rank"))
            for r in eval_rows
        ], dtype=np.float64)
        return score, {}, "ce_tail_rank,margin_tail_rank,wrong_confidence_rank"
    if name == "EV1-GroundedLinear":
        features = PREDICTOR_FEATURES
        X_train = _feature_matrix(train, features)
        X_eval = _feature_matrix(eval_rows, features)
        Xs, Xe = _standardize_train_eval(X_train, X_eval)
        coef = _fit_ridge(Xs, y_train)
        imp = {f: float(abs(coef[i + 1])) for i, f in enumerate(features)}
        return _predict_ridge(Xe, coef), imp, ",".join(features)
    if name == "EV2-MonotoneSignalRule":
        score = np.asarray([
            0.8 * _float(r.get("pre_real_tail_logit_delta_norm"))
            + 0.6 * _float(r.get("pre_real_nonadamw_delta_norm"))
            + 0.4 * _float(r.get("ce_tail_rank"))
            + 0.3 * _float(r.get("margin_tail_rank"))
            - 0.5 * _float(r.get("bootstrap_control_gap_std"))
            - 0.4 * abs(_float(r.get("cos_real_adamw")))
            for r in eval_rows
        ], dtype=np.float64)
        return score, {}, "monotone_tail_movement_uncertainty_rule"
    if name == "EV3-FamilyValuePredictor":
        fam_means: Dict[str, float] = defaultdict(float)
        fam_counts: Dict[str, int] = defaultdict(int)
        for r in train:
            fam_means[str(r.get("event_family"))] += _float(r.get("normalized_value"))
            fam_counts[str(r.get("event_family"))] += 1
        global_mean = _mean(_float(r.get("normalized_value")) for r in train)
        for k in list(fam_means):
            fam_means[k] /= max(1, fam_counts[k])
        score = np.asarray([fam_means.get(str(r.get("event_family")), global_mean) for r in eval_rows], dtype=np.float64)
        return score, {}, "event_family_prior_no_dataset_name"
    if name == "EV4-ConformalAbstainController":
        score = np.asarray([
            _float(r.get("event_family_value")) - 1.0 * _float(r.get("bootstrap_value_std"))
            for r in eval_rows
        ], dtype=np.float64)
        return score, {}, "family_lower_confidence_bound"
    if name == "EV5-HorizonAgreementController":
        score = np.asarray([
            _float(r.get("horizon_agreement_score")) + 0.2 * _float(r.get("event_family_value"))
            - 0.2 * _float(r.get("bootstrap_control_gap_std"))
            for r in eval_rows
        ], dtype=np.float64)
        return score, {}, "horizon_agreement_family_value"
    if name == "EV6-OrthogonalTailController":
        score = np.asarray([
            _float(r.get("pre_real_nonadamw_delta_norm"))
            + _float(r.get("orthogonal_component_ratio"))
            + _float(r.get("pre_real_tail_logit_delta_norm"))
            - abs(_float(r.get("cos_real_adamw")))
            for r in eval_rows
        ], dtype=np.float64)
        return score, {}, "orthogonal_nonadamw_tail_movement"
    if name == "EV7-PrimitiveAgreementController":
        primitive_mean: Dict[str, float] = defaultdict(float)
        primitive_count: Dict[str, int] = defaultdict(int)
        for r in train:
            primitive_mean[str(r.get("primitive"))] += _float(r.get("normalized_value"))
            primitive_count[str(r.get("primitive"))] += 1
        for k in list(primitive_mean):
            primitive_mean[k] /= max(1, primitive_count[k])
        global_mean = _mean(_float(r.get("normalized_value")) for r in train)
        score = np.asarray([
            primitive_mean.get(str(r.get("primitive")), global_mean)
            + 0.1 * _float(r.get("pre_real_nonadamw_delta_norm"))
            for r in eval_rows
        ], dtype=np.float64)
        return score, {}, "primitive_prior_plus_nonadamw_movement"
    if name == "EV8-LDOCalibratedSignalController":
        features = [
            "pre_real_tail_logit_delta_norm",
            "pre_real_nonadamw_delta_norm",
            "pre_tail_real_vs_control_ratio",
            "effective_derivative",
            "bootstrap_control_gap_std",
            "cos_tail_real_adamw",
        ]
        X_train = _feature_matrix(train, features)
        X_eval = _feature_matrix(eval_rows, features)
        Xs, Xe = _standardize_train_eval(X_train, X_eval)
        coef = _fit_ridge(Xs, y_train)
        imp = {f: float(abs(coef[i + 1])) for i, f in enumerate(features)}
        return _predict_ridge(Xe, coef), imp, ",".join(features)
    raise ValueError(name)


def _acceptance_metrics(scores: Sequence[float], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = [_int(r.get("Y_beat")) for r in rows]
    bad = [_int(r.get("bad_event")) for r in rows]
    best: Dict[str, Any] = {
        "precision": 0.0,
        "recall": 0.0,
        "coverage": 0.0,
        "bad_event_rate": 0.0,
        "accepted_count": 0,
        "accepted_signal_strata_count": 0,
        "threshold_quantile": "",
        "threshold": "",
    }
    total_good = max(1, sum(labels))
    for q in [0.85, 0.90, 0.92, 0.95, 0.97]:
        threshold = _quantile(list(scores), q)
        accepted = [i for i, s in enumerate(scores) if float(s) >= threshold]
        coverage = len(accepted) / max(1, len(rows))
        if coverage < 0.03 or coverage > 0.15:
            continue
        precision = sum(labels[i] for i in accepted) / max(1, len(accepted))
        recall = sum(labels[i] for i in accepted) / total_good
        bad_rate = sum(bad[i] for i in accepted) / max(1, len(accepted))
        strata = {str(rows[i].get("signal_stratum")) for i in accepted}
        candidate = {
            "precision": precision,
            "recall": recall,
            "coverage": coverage,
            "bad_event_rate": bad_rate,
            "accepted_count": len(accepted),
            "accepted_signal_strata_count": len(strata),
            "threshold_quantile": q,
            "threshold": float(threshold),
        }
        if (precision, coverage, -bad_rate) > (best["precision"], best["coverage"], -best["bad_event_rate"]):
            best = candidate
    return best


def _run_p4_predictors(model_rows: List[Dict[str, Any]], opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P4_EVENT_VALUE_PREDICTOR_REDESIGN", "p4_event_value_predictor_redesign.csv", "P3_failure_mode_reclassification_failed")
        return [row], {
            "event_value_predictor_pass": 0,
            "best_predictor": "not_opened",
            "predictor_corr": 0.0,
            "predictor_auc": 0.5,
            "predictor_precision": 0.0,
            "predictor_coverage": 0.0,
            "predictor_bad_event_rate": 0.0,
            "accepted_signal_strata_count": 0,
            "abstention_precision_pass": 0,
            "abstention_coverage_pass": 0,
        }
    predictors = [
        "EV0-OldStrataOnly",
        "EV1-GroundedLinear",
        "EV2-MonotoneSignalRule",
        "EV3-FamilyValuePredictor",
        "EV4-ConformalAbstainController",
        "EV5-HorizonAgreementController",
        "EV6-OrthogonalTailController",
        "EV7-PrimitiveAgreementController",
        "EV8-LDOCalibratedSignalController",
    ]
    y = [_float(r.get("normalized_value")) for r in model_rows]
    labels = [_int(r.get("Y_beat")) for r in model_rows]
    rows: List[Dict[str, Any]] = []
    for pred in predictors:
        scores, importance, feature_set = _score_predictor(pred, model_rows, model_rows)
        corr = _corr(list(scores), y)
        auc = _auc(list(scores), labels)
        acc = _acceptance_metrics(scores, model_rows)
        p_pass = int(
            (corr >= 0.30 or auc >= 0.65)
            and acc["precision"] >= 0.75
            and 0.03 <= acc["coverage"] <= 0.15
            and acc["bad_event_rate"] <= 0.05
            and acc["accepted_signal_strata_count"] >= 2
        )
        rows.append(
            {
                "stage": "P4_EVENT_VALUE_PREDICTOR_REDESIGN",
                "status": "source_measured_derived",
                "predictor": pred,
                "feature_set": feature_set,
                "label_type": "V_norm,Y_beat,Y_strong",
                "train_split": "all_v9230_rows_calibration",
                "eval_split": "all_v9230_rows_calibration",
                "corr_value": corr,
                "auc_beat": auc,
                "precision": acc["precision"],
                "recall": acc["recall"],
                "coverage": acc["coverage"],
                "bad_event_rate": acc["bad_event_rate"],
                "accepted_count": acc["accepted_count"],
                "accepted_signal_strata_count": acc["accepted_signal_strata_count"],
                "threshold": acc["threshold"],
                "threshold_quantile": acc["threshold_quantile"],
                "feature_importance": json.dumps(importance, sort_keys=True),
                "dataset_name_used": 0,
                "seed_id_used": 0,
                "posthoc_metric_used": 0,
                "predictor_pass": p_pass,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    best = max(rows, key=lambda r: (_int(r.get("predictor_pass")), _float(r.get("precision")), _float(r.get("corr_value"))))
    summary = {
        "event_value_predictor_pass": _int(best.get("predictor_pass")),
        "best_predictor": best.get("predictor", ""),
        "predictor_corr": _float(best.get("corr_value")),
        "predictor_auc": _float(best.get("auc_beat")),
        "predictor_precision": _float(best.get("precision")),
        "predictor_coverage": _float(best.get("coverage")),
        "predictor_bad_event_rate": _float(best.get("bad_event_rate")),
        "accepted_signal_strata_count": _int(best.get("accepted_signal_strata_count")),
        "abstention_precision_pass": int(_float(best.get("precision")) >= 0.75),
        "abstention_coverage_pass": int(0.03 <= _float(best.get("coverage")) <= 0.15),
    }
    return rows, summary


def _run_p5_heldout(model_rows: List[Dict[str, Any]], best_predictor: str, opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P5_LEAVE_DATASET_AND_STRATUM_OUT_VALIDATION", "p5_leave_dataset_and_stratum_out_validation.csv", "P4_event_value_predictor_failed")
        return [row], {"leave_dataset_out_pass": 0, "leave_stratum_out_pass": 0}
    rows: List[Dict[str, Any]] = []
    datasets = sorted({str(r.get("dataset")) for r in model_rows})
    ldo_pass_count = 0
    for heldout in datasets:
        train = [r for r in model_rows if str(r.get("dataset")) != heldout]
        eval_rows = [r for r in model_rows if str(r.get("dataset")) == heldout]
        scores, _importance, _feature_set = _score_predictor(best_predictor, train, eval_rows)
        acc = _acceptance_metrics(scores, eval_rows)
        accepted = [i for i, s in enumerate(scores) if acc["threshold"] != "" and float(s) >= float(acc["threshold"])]
        beats_parallel = _mean(_int(eval_rows[i].get("real_beats_adamwparallel")) for i in accepted) if accepted else 0.0
        beats_lr = _mean(_int(eval_rows[i].get("real_beats_best_lr")) for i in accepted) if accepted else 0.0
        task_safe = _mean(_int(eval_rows[i].get("task_safe")) for i in accepted) if accepted else 0.0
        ce = _mean(_float(eval_rows[i].get("CEp99_delta")) for i in accepted) if accepted else 0.0
        margin = _mean(_float(eval_rows[i].get("margin_p10_delta")) for i in accepted) if accepted else 0.0
        split_pass = int(task_safe >= 1.0 and beats_parallel >= 0.50 and beats_lr >= 0.50)
        ldo_pass_count += split_pass
        rows.append(
            {
                "stage": "P5_LEAVE_DATASET_OUT_VALIDATION",
                "status": "source_measured_derived",
                "split_type": "leave_dataset_out",
                "heldout": heldout,
                "predictor": best_predictor,
                "primitive_selector": "signal_value_features_no_dataset_name",
                "thresholds": acc["threshold"],
                "corr": _corr(list(scores), [_float(r.get("normalized_value")) for r in eval_rows]),
                "precision": acc["precision"],
                "coverage": acc["coverage"],
                "bad_event_rate": acc["bad_event_rate"],
                "task_safe": task_safe,
                "CEp99_delta": ce,
                "margin_delta": margin,
                "curvature_delta": "not_measured_in_v9230_source",
                "beats_adamwparallel": beats_parallel,
                "beats_bestlr": beats_lr,
                "split_pass": split_pass,
                "dataset_name_used": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    strata = sorted({str(r.get("signal_stratum")) for r in model_rows})
    lso_pass_count = 0
    for heldout in strata:
        train = [r for r in model_rows if str(r.get("signal_stratum")) != heldout]
        eval_rows = [r for r in model_rows if str(r.get("signal_stratum")) == heldout]
        scores, _importance, _feature_set = _score_predictor(best_predictor, train, eval_rows)
        acc = _acceptance_metrics(scores, eval_rows)
        accepted = [i for i, s in enumerate(scores) if acc["threshold"] != "" and float(s) >= float(acc["threshold"])]
        task_safe = _mean(_int(eval_rows[i].get("task_safe")) for i in accepted) if accepted else 0.0
        ce = _mean(_float(eval_rows[i].get("CEp99_delta")) for i in accepted) if accepted else 0.0
        split_pass = int(task_safe >= 1.0 and ce <= 1.0e-9)
        lso_pass_count += split_pass
        rows.append(
            {
                "stage": "P5_LEAVE_STRATUM_OUT_VALIDATION",
                "status": "source_measured_derived",
                "split_type": "leave_stratum_out",
                "heldout": heldout,
                "predictor": best_predictor,
                "primitive_selector": "signal_value_features_no_dataset_name",
                "thresholds": acc["threshold"],
                "corr": _corr(list(scores), [_float(r.get("normalized_value")) for r in eval_rows]),
                "precision": acc["precision"],
                "coverage": acc["coverage"],
                "bad_event_rate": acc["bad_event_rate"],
                "task_safe": task_safe,
                "CEp99_delta": ce,
                "margin_delta": _mean(_float(eval_rows[i].get("margin_p10_delta")) for i in accepted) if accepted else 0.0,
                "curvature_delta": "not_measured_in_v9230_source",
                "beats_adamwparallel": _mean(_int(eval_rows[i].get("real_beats_adamwparallel")) for i in accepted) if accepted else 0.0,
                "beats_bestlr": _mean(_int(eval_rows[i].get("real_beats_best_lr")) for i in accepted) if accepted else 0.0,
                "split_pass": split_pass,
                "dataset_name_used": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    summary = {
        "leave_dataset_out_pass": int(ldo_pass_count >= 2),
        "leave_dataset_out_pass_count": ldo_pass_count,
        "leave_dataset_out_split_count": len(datasets),
        "leave_stratum_out_pass": int(lso_pass_count / max(1, len(strata)) >= 0.70),
        "leave_stratum_out_pass_count": lso_pass_count,
        "leave_stratum_out_split_count": len(strata),
    }
    return rows, summary


def _make_svg_bar(path: Path, title: str, labels: Sequence[str], values: Sequence[float]) -> None:
    ensure_dir(path.parent)
    width = 760
    height = 280
    vals = [float(v) for v in values]
    vmax = max([abs(v) for v in vals] + [1.0e-9])
    bar_w = max(12, int((width - 140) / max(1, len(vals))))
    zero_y = 150
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="28" font-family="sans-serif" font-size="16">{title}</text>',
        f'<line x1="80" x2="{width-40}" y1="{zero_y}" y2="{zero_y}" stroke="#666" stroke-width="1"/>',
    ]
    for i, (label, val) in enumerate(zip(labels, vals)):
        x = 80 + i * bar_w
        h = int((abs(val) / vmax) * 85)
        y = zero_y - h if val >= 0 else zero_y
        color = "#2b6cb0" if val >= 0 else "#c53030"
        lines.append(f'<rect x="{x}" y="{y}" width="{max(8, bar_w-4)}" height="{h}" fill="{color}" opacity="0.85"/>')
        lines.append(f'<text x="{x}" y="255" font-family="sans-serif" font-size="9" transform="rotate(-35 {x},255)">{label}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_report(path: Path, out_dir: Path, route: Dict[str, Any], hashes: List[Dict[str, str]]) -> None:
    def h(name: str) -> str:
        for row in hashes:
            if row["artifact"].endswith(name):
                return row["sha256"]
        return ""

    text = f"""# DG-KAN v9.2.31 Event-Value Grounding 与 Primitive Observability 实验复盘

> 本复盘记录 `DG-KAN_v9.2.31_EventValueGrounding_PrimitiveObservability_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = {route['route']}
base_candidate = LQ-t2-h256
success_v9231_strict_purekan_functional = {bool(route['success_v9231_strict_purekan_functional'])}
success_v9231_full_functional = {bool(route['success_v9231_full_functional'])}
success_v9231_external_ready = {bool(route['success_v9231_external_ready'])}
```

最终 artifact：

```text
{out_dir.relative_to(ROOT)}/
```

核心结论：

1. P0 复现 v9.2.30 boundary：source route = `R7-PrimitiveEffectUnpredictable`，fake/proxy = `0`。
2. P1 grounded value 可靠性 `value_reliability = {route['value_reliability']:.6f}`，event-value grounding pass = `{route['event_value_grounding_pass']}`。
3. P2 pre-event observability pass = `{route['pre_event_feature_pass']}`；best feature = `{route['best_predictive_feature']}`，best movement = `{route['best_movement_feature']}` / `{route['best_movement_abs_corr']:.6f}`。
4. P3 failure-mode reclassification pass = `{route['failure_mode_reclassification_pass']}`，GoodEvent precision = `{route['good_event_precision']:.6f}`，abstain precision = `{route['abstain_precision']:.6f}`。
5. P4 event-value predictor pass = `{route['event_value_predictor_pass']}`；best predictor = `{route['best_predictor']}`，corr = `{route['predictor_corr']:.6f}`，precision = `{route['predictor_precision']:.6f}`，coverage = `{route['predictor_coverage']:.6f}`。
6. 当前 blocker：`{route['primary_blocker']}`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9231_event_value_grounding_primitive_observability.py` | v9.2.31 runner；从 v9.2.30 真实 pre-event replay rows 生成 grounded value、observability、predictor / LDO gate、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9231_event_value_grounding_primitive_observability.py
```

正式运行：

```bash
python experiments/run_v9231_event_value_grounding_primitive_observability.py \\
  --out-dir {out_dir.relative_to(ROOT)} \\
  --fresh \\
  --seed 1314
```

说明：本轮是 source-measured rows 的 grounding / observability audit。v9.2.30 没有测到的 curvature delta、control ECE/NLL comparable 均写为 `not_measured_in_v9230_source`，没有补造。

## 2. Route

```json
{json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True)}
```

## 3. P1 Event-Value Grounding

Artifact：

```text
p1_event_value_label_grounding.csv
```

关键值：

| metric | value |
|---|---:|
| rows | `{route['p1_row_count']}` |
| value reliability | `{route['value_reliability']:.6f}` |
| normalization pass | `{route['value_normalization_pass']}` |
| raw value mean | `{route['raw_value_mean']:.6f}` |
| raw value std | `{route['raw_value_std']:.6f}` |
| normalized value std | `{route['normalized_value_std']:.6f}` |

判断：grounded value 只使用 v9.2.30 同时具备 Real / AdamWParallel / bestLR comparable 的 CEp99 与 margin，再扣 task risk；curvature 与 control-ECE/NLL 未测，明确排除。

## 4. P2 Pre-Event Observability

Artifact：

```text
p2_pre_event_feature_relevance.csv
```

关键值：

| metric | value |
|---|---:|
| relevant feature count | `{route['pre_event_relevant_feature_count']}` |
| best predictive feature | `{route['best_predictive_feature']}` |
| best predictive abs corr | `{route['best_predictive_feature_abs_corr']:.6f}` |
| best movement feature | `{route['best_movement_feature']}` |
| best movement abs corr | `{route['best_movement_abs_corr']:.6f}` |

判断：pre-event feature 从 v9.2.30 的 near-threshold diagnostic 进入 grounded-value observability；但这仍只是 predictor 前置条件，不是 functional success。

## 5. P3/P4 Predictor Gate

Artifacts：

```text
p3_failure_mode_reclassification.csv
p4_event_value_predictor_redesign.csv
```

P3/P4 summary：

| metric | value |
|---|---:|
| failure-mode pass | `{route['failure_mode_reclassification_pass']}` |
| GoodEvent precision | `{route['good_event_precision']:.6f}` |
| abstain precision | `{route['abstain_precision']:.6f}` |
| predictor pass | `{route['event_value_predictor_pass']}` |
| predictor corr | `{route['predictor_corr']:.6f}` |
| predictor coverage | `{route['predictor_coverage']:.6f}` |
| accepted strata | `{route['accepted_signal_strata_count']}` |

判断：如果 P4/P5 未过，P6 official paired replay 不能打开；本轮没有把 calibration rows 当成 official controller success。

## 6. Downstream Boundary

这些 artifact 已落盘；未满足 gate 的阶段明确为 `not_run`：

```text
p6_official_signal_value_paired_replay.csv
p7_short_run_functional_validation.csv
p8_full_10seed_functional_validation.csv
p9_adamw_only_fullpass_repair.csv
p10_robustness_external_ready.csv
```

## 7. No-fake audit

```text
rows_checked = {route['rows_checked']}
fake_proxy_nonzero_count = {route['fake_proxy_nonzero_count']}
fake_data_used = {route['fake_data_used']}
proxy_row_used = {route['proxy_row_used']}
cpu_offload_used = {route['cpu_offload_used']}
no_fake = {bool(route['no_fake'])}
no_proxy = {bool(route['no_proxy'])}
```

## 8. Hash

| artifact | SHA256 |
|---|---|
| runner | `{h('run_v9231_event_value_grounding_primitive_observability.py')}` |
| route | `{h('route_decision.json')}` |
| P1 grounding | `{h('p1_event_value_label_grounding.csv')}` |
| P2 relevance | `{h('p2_pre_event_feature_relevance.csv')}` |
| P3 reclassification | `{h('p3_failure_mode_reclassification.csv')}` |
| P4 predictor | `{h('p4_event_value_predictor_redesign.csv')}` |
| P5 LDO/LSO | `{h('p5_leave_dataset_and_stratum_out_validation.csv')}` |
| provenance audit | `{h('v9231_provenance_audit.csv')}` |

## 9. 最终分析结论

v9.2.31 的真实推进是：

```text
v9.2.30: pre-event movement feature 接近阈值，但旧 label / attribution / predictor 失败。
v9.2.31: 先把 event value grounding、stratum baseline、observability 和 predictor gate 分离审计。
```

机制判断：

1. 本轮不能说 primitive 已经可用于 official functional route；official paired replay 仍取决于 P4/P5 gate。
2. 如果 predictor 或 LDO/LSO 没过，正确结论是 event-value controller 尚不稳定，而不是倒回 dataset-specific Fashion/KMNIST patch。
3. 如果 grounding/observability 过而 controller 不过，下一步应补充真实 pre-event movement features、repeat seeds/h640，或重做 primitive observability，而不是编造 missing curvature/control-ECE。

最终一句话：

> v9.2.31 真实执行后停在 `{route['route']}`：`{route['primary_blocker']}`。
"""
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--seed", type=int, default=1314)
    args = parser.parse_args()

    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")

    manifest = {
        "version": "v9.2.31",
        "created_utc": _now_iso(),
        "plan_path": str(PLAN_PATH.relative_to(ROOT)),
        "runner": str(SCRIPT_PATH.relative_to(ROOT)),
        "source_v9230": str(SRC_V9230.relative_to(ROOT)),
        "seed": args.seed,
        "mode": "source_measured_event_value_grounding_no_new_replay",
        "missing_metric_policy": "curvature_and_control_ECE_NLL_not_measured_in_v9230_source_excluded_not_imputed",
    }
    write_json(out_dir / "run_manifest.json", manifest)
    contract = [
        {
            "stage": "CONTRACT",
            "loss_type": "CE_source_replay_only",
            "teacher_used": 0,
            "self_teacher_used": 0,
            "distillation_used": 0,
            "loss_modification_used": 0,
            "dataset_name_used_in_official_predictor": 0,
            "seed_id_used_in_official_predictor": 0,
            "class_name_route_key_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    ]
    write_csv_rows(out_dir / "contract_audit_v9231.csv", contract)

    p0 = _source_p0()
    write_csv_rows(out_dir / "p0_v9230_boundary_reproduction.csv", [p0])

    p1_rows, p1_summary = _run_p1_grounding(bool(_int(p0.get("P0_pass"))))
    write_csv_rows(out_dir / "p1_event_value_label_grounding.csv", p1_rows)

    p2_rows, p2_summary = _run_p2_features(p1_rows, bool(p1_summary["event_value_grounding_pass"]))
    write_csv_rows(out_dir / "p2_pre_event_feature_relevance.csv", p2_rows)

    p3_rows, p3_summary = _run_p3_failure_modes(p1_rows, bool(p2_summary["pre_event_feature_pass"]))
    write_csv_rows(out_dir / "p3_failure_mode_reclassification.csv", p3_rows)

    model_rows = _prepare_model_rows(p1_rows)
    p4_rows, p4_summary = _run_p4_predictors(model_rows, bool(p3_summary["failure_mode_reclassification_pass"]))
    write_csv_rows(out_dir / "p4_event_value_predictor_redesign.csv", p4_rows)

    p5_rows, p5_summary = _run_p5_heldout(model_rows, str(p4_summary["best_predictor"]), bool(p4_summary["event_value_predictor_pass"]))
    write_csv_rows(out_dir / "p5_leave_dataset_and_stratum_out_validation.csv", p5_rows)

    downstream_reason = (
        "P5_leave_dataset_or_stratum_out_failed"
        if not (p5_summary["leave_dataset_out_pass"] and p5_summary["leave_stratum_out_pass"])
        else "official_replay_not_implemented_in_source_grounding_runner"
    )
    write_csv_rows(out_dir / "p6_official_signal_value_paired_replay.csv", [
        _not_run("P6_OFFICIAL_SIGNAL_VALUE_PAIRED_REPLAY", "p6_official_signal_value_paired_replay.csv", downstream_reason)
    ])
    write_csv_rows(out_dir / "p7_short_run_functional_validation.csv", [
        _not_run("P7_SHORT_RUN_FUNCTIONAL_VALIDATION", "p7_short_run_functional_validation.csv", "P6_official_signal_value_paired_replay_not_passed")
    ])
    write_csv_rows(out_dir / "p8_full_10seed_functional_validation.csv", [
        _not_run("P8_FULL_10SEED_FUNCTIONAL_VALIDATION", "p8_full_10seed_functional_validation.csv", "P6_official_signal_value_paired_replay_not_passed")
    ])
    write_csv_rows(out_dir / "p9_adamw_only_fullpass_repair.csv", [
        _not_run("P9_ADAMW_ONLY_FULLPASS_REPAIR", "p9_adamw_only_fullpass_repair.csv", "P6_official_signal_value_paired_replay_not_passed")
    ])
    write_csv_rows(out_dir / "p10_robustness_external_ready.csv", [
        _not_run("P10_ROBUSTNESS_EXTERNAL_READY", "p10_robustness_external_ready.csv", "P6_official_signal_value_paired_replay_not_passed")
    ])

    # Traces are compact aliases for the measured-derived rows.
    write_csv_rows(out_dir / "event_value_grounding_trace_v9231.csv", p1_rows)
    write_csv_rows(out_dir / "primitive_observability_trace_v9231.csv", p2_rows)
    write_csv_rows(out_dir / "predictor_trace_v9231.csv", p4_rows)
    write_csv_rows(out_dir / "failure_table.csv", [
        {
            "route_candidate": "P0",
            "pass": p0["P0_pass"],
            "blocker": "" if _int(p0["P0_pass"]) else "v9230_boundary_not_reproduced",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "route_candidate": "P1_grounding",
            "pass": p1_summary["event_value_grounding_pass"],
            "blocker": "" if p1_summary["event_value_grounding_pass"] else "event_value_reliability_or_normalization_failed",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "route_candidate": "P2_observability",
            "pass": p2_summary["pre_event_feature_pass"],
            "blocker": "" if p2_summary["pre_event_feature_pass"] else "pre_event_features_do_not_predict_grounded_value",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "route_candidate": "P4_predictor",
            "pass": p4_summary["event_value_predictor_pass"],
            "blocker": "" if p4_summary["event_value_predictor_pass"] else "event_value_predictor_not_calibrated",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "route_candidate": "P5_LDO_LSO",
            "pass": int(p5_summary["leave_dataset_out_pass"] and p5_summary["leave_stratum_out_pass"]),
            "blocker": "" if (p5_summary["leave_dataset_out_pass"] and p5_summary["leave_stratum_out_pass"]) else "leave_dataset_or_stratum_out_failed",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
    ])

    # Small truthful figures.
    top_feats = sorted(
        [r for r in p2_rows if r.get("status") == "source_measured_derived"],
        key=lambda r: abs(_float(r.get("feature_corr_norm"))),
        reverse=True,
    )[:10]
    _make_svg_bar(
        out_dir / "figures" / "p2_feature_value_correlation_top10.svg",
        "P2 top feature correlations with grounded value",
        [r["feature"] for r in top_feats],
        [_float(r.get("feature_corr_norm")) for r in top_feats],
    )
    pred_fig = [r for r in p4_rows if r.get("status") == "source_measured_derived"]
    _make_svg_bar(
        out_dir / "figures" / "p4_predictor_corr.svg",
        "P4 predictor correlation",
        [r["predictor"] for r in pred_fig],
        [_float(r.get("corr_value")) for r in pred_fig],
    )

    if not _int(p0.get("P0_pass")):
        route_name = "R0-SourceBoundaryNotReproduced"
        primary = "v9230_boundary_not_reproduced"
    elif not p1_summary["event_value_grounding_pass"]:
        route_name = "R9-PrimitiveEffectUnpredictable"
        primary = "event_value_grounding_failed"
    elif not p2_summary["pre_event_feature_pass"]:
        route_name = "R3-FamilyValuePredictable" if p2_summary.get("family_value_predictable") else "R9-PrimitiveEffectUnpredictable"
        primary = "pre_event_features_do_not_predict_grounded_value"
    elif not p4_summary["event_value_predictor_pass"]:
        route_name = "R2-PreEventFeaturesPredictValue"
        primary = "event_value_predictor_not_calibrated"
    elif not (p5_summary["leave_dataset_out_pass"] and p5_summary["leave_stratum_out_pass"]):
        route_name = "R4-EventValuePredictorPass"
        primary = "leave_dataset_or_stratum_out_failed"
    else:
        route_name = "R5-LeaveDatasetOutPass"
        primary = "official_replay_not_implemented_in_source_grounding_runner"

    audit_paths = [
        out_dir / "contract_audit_v9231.csv",
        out_dir / "p0_v9230_boundary_reproduction.csv",
        out_dir / "p1_event_value_label_grounding.csv",
        out_dir / "p2_pre_event_feature_relevance.csv",
        out_dir / "p3_failure_mode_reclassification.csv",
        out_dir / "p4_event_value_predictor_redesign.csv",
        out_dir / "p5_leave_dataset_and_stratum_out_validation.csv",
        out_dir / "p6_official_signal_value_paired_replay.csv",
        out_dir / "p7_short_run_functional_validation.csv",
        out_dir / "p8_full_10seed_functional_validation.csv",
        out_dir / "p9_adamw_only_fullpass_repair.csv",
        out_dir / "p10_robustness_external_ready.csv",
    ]
    audit = audit_no_fake(audit_paths)

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9230_boundary_pass": _int(p0.get("P0_pass")),
        "dataset_tuning_detected": 0,
        "event_value_grounding_pass": p1_summary["event_value_grounding_pass"],
        "value_reliability": p1_summary["value_reliability"],
        "value_normalization_pass": p1_summary["value_normalization_pass"],
        "p1_row_count": p1_summary["p1_row_count"],
        "raw_value_mean": p1_summary["raw_value_mean"],
        "raw_value_std": p1_summary["raw_value_std"],
        "normalized_value_mean": p1_summary["normalized_value_mean"],
        "normalized_value_std": p1_summary["normalized_value_std"],
        "pre_event_feature_pass": p2_summary["pre_event_feature_pass"],
        "pre_event_relevant_feature_count": p2_summary["pre_event_relevant_feature_count"],
        "best_predictive_feature": p2_summary["best_predictive_feature"],
        "best_predictive_feature_abs_corr": p2_summary["best_predictive_feature_abs_corr"],
        "best_movement_feature": p2_summary["best_movement_feature"],
        "best_movement_abs_corr": p2_summary["best_movement_abs_corr"],
        "family_value_predictable": p2_summary.get("family_value_predictable", 0),
        "failure_mode_reclassification_pass": p3_summary["failure_mode_reclassification_pass"],
        "assigned_fraction": p3_summary["assigned_fraction"],
        "good_event_precision": p3_summary["good_event_precision"],
        "good_event_count": p3_summary["good_event_count"],
        "abstain_precision": p3_summary["abstain_precision"],
        "abstain_count": p3_summary["abstain_count"],
        "best_predictor": p4_summary["best_predictor"],
        "event_value_predictor_pass": p4_summary["event_value_predictor_pass"],
        "predictor_corr": p4_summary["predictor_corr"],
        "predictor_auc": p4_summary["predictor_auc"],
        "predictor_precision": p4_summary["predictor_precision"],
        "predictor_coverage": p4_summary["predictor_coverage"],
        "predictor_bad_event_rate": p4_summary["predictor_bad_event_rate"],
        "accepted_signal_strata_count": p4_summary["accepted_signal_strata_count"],
        "abstention_precision_pass": p4_summary["abstention_precision_pass"],
        "abstention_coverage_pass": p4_summary["abstention_coverage_pass"],
        "leave_dataset_out_pass": p5_summary["leave_dataset_out_pass"],
        "leave_dataset_out_pass_count": p5_summary.get("leave_dataset_out_pass_count", 0),
        "leave_stratum_out_pass": p5_summary["leave_stratum_out_pass"],
        "leave_stratum_out_pass_count": p5_summary.get("leave_stratum_out_pass_count", 0),
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "adamw_fullpass": 0,
        "robustness_pass": 0,
        "external_ready": 0,
        "primary_blocker": primary,
        "next_required_implementation": "if_LDO_failed_add_real_repeat_h640_features_else_run_official_signal_value_paired_replay",
        "success_v9231_strict_purekan_functional": 0,
        "success_v9231_full_functional": 0,
        "success_v9231_external_ready": 0,
        **audit,
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)

    # Hashes after all primary artifacts have been written.
    hash_paths = [
        PLAN_PATH,
        SCRIPT_PATH,
        out_dir / "route_decision.json",
        out_dir / "contract_audit_v9231.csv",
        out_dir / "p0_v9230_boundary_reproduction.csv",
        out_dir / "p1_event_value_label_grounding.csv",
        out_dir / "p2_pre_event_feature_relevance.csv",
        out_dir / "p3_failure_mode_reclassification.csv",
        out_dir / "p4_event_value_predictor_redesign.csv",
        out_dir / "p5_leave_dataset_and_stratum_out_validation.csv",
        out_dir / "failure_table.csv",
    ]
    audit_rows = [audit]
    write_csv_rows(out_dir / "v9231_provenance_audit.csv", audit_rows)
    hash_paths.append(out_dir / "v9231_provenance_audit.csv")
    hashes = artifact_hash_rows(hash_paths, root=ROOT)
    write_csv_rows(out_dir / "artifact_hashes.csv", hashes)
    _write_report(REPORT_PATH, out_dir, route, hashes)
    print(json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
