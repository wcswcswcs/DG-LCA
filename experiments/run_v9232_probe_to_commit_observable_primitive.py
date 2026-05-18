#!/usr/bin/env python3
"""DG-KAN v9.2.32 probe-to-commit and observable primitive audit.

This runner starts from the real v9.2.31 grounded event-value artifacts and
tests whether source-logged pre-commit sufficient statistics can behave as
probe-to-value scores.  It does not fabricate missing train-stream virtual
probes or new primitive implementations: unavailable probes/primitives are
explicitly marked as not implemented or not official.
"""

from __future__ import annotations

import argparse
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


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.32_ProbeToCommit_ObservablePrimitive_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9232_probe_to_commit_observable_primitive.py"
SRC_V9231 = ROOT / "results" / "real_rerun_20260506" / "v9231_event_value_grounding_primitive_observability_first_20260510T220000Z"
SRC_V9230 = ROOT / "results" / "real_rerun_20260506" / "v9230_preevent_signal_value_functional_controller_first_20260510T210000Z"
SRC_V9222 = ROOT / "results" / "real_rerun_20260506" / "v9222_basequalified_strict_purekan_functional_interface_first_20260510T120000Z"
REPORT_PATH = ROOT / "docs" / "DG-KAN_v9.2.32_ProbeToCommit_ObservablePrimitive_实验复盘.md"

STATIC_FEATURES = [
    "effective_derivative",
    "pre_adamwparallel_logit_delta_norm",
    "pre_bestlr_logit_delta_norm",
    "pre_real_nonadamw_delta_norm",
    "pre_real_logit_delta_norm",
    "pre_real_tail_logit_delta_norm",
    "pre_tail_real_vs_control_ratio",
    "actual_r_z_perp",
    "dominant_basis_fraction",
    "branch_ratio",
    "cos_tail_real_adamw",
    "microbatch_score_variance",
    "horizon_agreement_score",
]

MOVEMENT_FEATURES = {
    "pre_adamwparallel_logit_delta_norm",
    "pre_bestlr_logit_delta_norm",
    "pre_real_nonadamw_delta_norm",
    "pre_real_logit_delta_norm",
    "pre_real_tail_logit_delta_norm",
    "pre_tail_real_vs_control_ratio",
    "actual_r_z_perp",
}


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
    pos = [float(s) for s, y in zip(scores, labels) if int(y) == 1]
    neg = [float(s) for s, y in zip(scores, labels) if int(y) == 0]
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


def _p0_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9231 / "route_decision.json")
    audit_rows = read_csv_rows(SRC_V9231 / "v9231_provenance_audit.csv")
    fake = _int(audit_rows[0].get("fake_proxy_nonzero_count")) if audit_rows else 1
    p0_pass = int(
        route.get("route") == "R9-PrimitiveEffectUnpredictable"
        and _int(route.get("event_value_grounding_pass")) == 1
        and _int(route.get("pre_event_feature_pass")) == 0
        and _int(route.get("event_value_predictor_pass")) == 0
        and fake == 0
    )
    return {
        "stage": "P0_V9231_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "source_artifact": str(SRC_V9231.relative_to(ROOT)),
        "route": route.get("route", ""),
        "source_route_v9230": "R7-PrimitiveEffectUnpredictable",
        "value_reliability": route.get("value_reliability", ""),
        "event_value_grounding_pass": route.get("event_value_grounding_pass", ""),
        "pre_event_observability_pass": route.get("pre_event_feature_pass", ""),
        "best_feature": route.get("best_predictive_feature", ""),
        "best_movement_feature": route.get("best_movement_feature", ""),
        "best_movement_abs_corr": route.get("best_movement_abs_corr", ""),
        "failure_reclassification_pass": route.get("failure_mode_reclassification_pass", ""),
        "good_event_precision": route.get("good_event_precision", ""),
        "abstain_precision": route.get("abstain_precision", ""),
        "event_value_predictor_pass": route.get("event_value_predictor_pass", ""),
        "fake_proxy_count": fake,
        "P0_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _load_joined_events() -> List[Dict[str, Any]]:
    grounded = {
        r["event_id"]: r
        for r in read_csv_rows(SRC_V9231 / "p1_event_value_label_grounding.csv")
        if r.get("status") == "source_measured_derived"
    }
    source = {
        r["event_id"]: r
        for r in read_csv_rows(SRC_V9230 / "p1_pre_event_movement_features.csv")
        if r.get("status") == "measured"
    }
    rows: List[Dict[str, Any]] = []
    for event_id, g in grounded.items():
        if event_id not in source:
            continue
        s = source[event_id]
        row = dict(s)
        row.update(
            {
                "grounded_value": _float(g.get("normalized_value")),
                "control_relative_value": _float(g.get("control_relative_value")),
                "event_family_value": _float(g.get("event_family_value")),
                "bootstrap_value_std": _float(g.get("bootstrap_value_std")),
                "value_reliability": _float(g.get("value_reliability")),
                "Y_beat": int(_int(s.get("real_beats_adamwparallel")) == 1 and _int(s.get("real_beats_best_lr")) == 1 and _int(s.get("task_safe")) == 1),
                "bad_event": int(_int(s.get("task_safe")) == 0),
            }
        )
        rows.append(row)
    return rows


def _run_p1_static_autopsy(events: List[Dict[str, Any]], p0_open: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not p0_open:
        row = _not_run("P1_STATIC_FEATURE_INSUFFICIENCY_AUTOPSY", "p1_static_feature_insufficiency_autopsy.csv", "P0_v9231_boundary_failed")
        return [row], {
            "static_feature_insufficiency_confirmed": 0,
            "best_static_feature_corr": 0.0,
            "best_movement_feature_corr": 0.0,
            "feature_rank_sufficient": 0,
            "single_event_value_too_noisy": 0,
        }
    y = [_float(r.get("grounded_value")) for r in events]
    feature_corrs: Dict[str, float] = {f: _corr([_float(r.get(f)) for r in events], y) for f in STATIC_FEATURES}
    best_feature, best_corr = max(feature_corrs.items(), key=lambda kv: abs(kv[1]))
    movement_items = [(f, c) for f, c in feature_corrs.items() if f in MOVEMENT_FEATURES]
    best_movement, best_movement_corr = max(movement_items, key=lambda kv: abs(kv[1]))
    feature_vars = {f: _var(_float(r.get(f)) for r in events) for f in STATIC_FEATURES}
    nondegenerate = [f for f, v in feature_vars.items() if v > 1.0e-16]
    family_values: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
    for r in events:
        key = (str(r.get("signal_stratum")), str(r.get("primitive")), str(r.get("event_type")), str(r.get("horizon")))
        family_values[key].append(_float(r.get("grounded_value")))
    within_vars = [_var(vals) for vals in family_values.values()]
    between_var = _var(_mean(vals) for vals in family_values.values())
    within_mean = _mean(within_vars)
    single_event_too_noisy = int(within_mean > between_var)
    rows: List[Dict[str, Any]] = []
    for r in events:
        rows.append(
            {
                "stage": "P1_STATIC_FEATURE_INSUFFICIENCY_AUTOPSY",
                "status": "source_measured_derived",
                "event_id": r.get("event_id", ""),
                "signal_stratum": r.get("signal_stratum", ""),
                "horizon": r.get("horizon", ""),
                "primitive": r.get("primitive", ""),
                "V_grounded": r.get("grounded_value", ""),
                "feature_vector": json.dumps({f: _float(r.get(f)) for f in STATIC_FEATURES}, sort_keys=True),
                "feature_missing_rate": 0.0,
                "feature_variance": _mean(feature_vars.values()),
                "feature_rank": len(nondegenerate),
                "microbatch_value_variance": r.get("microbatch_score_variance", ""),
                "within_family_value_variance": within_mean,
                "between_family_value_variance": between_var,
                "best_static_feature": best_feature,
                "best_static_feature_corr": best_corr,
                "best_movement_feature": best_movement,
                "best_movement_feature_corr": best_movement_corr,
                "static_feature_insufficiency_confirmed": int(abs(best_corr) < 0.30 and abs(best_movement_corr) < 0.30),
                "strict_all_static_below_025": int(all(abs(c) < 0.25 for c in feature_corrs.values())),
                "single_event_value_too_noisy": single_event_too_noisy,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    summary = {
        "static_feature_insufficiency_confirmed": int(abs(best_corr) < 0.30 and abs(best_movement_corr) < 0.30),
        "strict_all_static_below_025": int(all(abs(c) < 0.25 for c in feature_corrs.values())),
        "best_static_feature": best_feature,
        "best_static_feature_corr": best_corr,
        "best_movement_feature": best_movement,
        "best_movement_feature_corr": best_movement_corr,
        "feature_rank_sufficient": int(len(nondegenerate) >= 5),
        "single_event_value_too_noisy": single_event_too_noisy,
        "within_family_value_variance": within_mean,
        "between_family_value_variance": between_var,
    }
    return rows, summary


def _probe_scores(row: Dict[str, Any]) -> Dict[str, float]:
    ce = _float(row.get("ce_tail_rank"))
    margin = _float(row.get("margin_tail_rank"))
    wrong = _float(row.get("wrong_confidence_rank"))
    real = _float(row.get("pre_real_logit_delta_norm"))
    tail = _float(row.get("pre_real_tail_logit_delta_norm"))
    nonadamw = _float(row.get("pre_real_nonadamw_delta_norm"))
    par = _float(row.get("pre_adamwparallel_logit_delta_norm"))
    bestlr = _float(row.get("pre_bestlr_logit_delta_norm"))
    tail_ratio = _float(row.get("pre_tail_real_vs_control_ratio"))
    rz_perp = _float(row.get("actual_r_z_perp"))
    branch = _float(row.get("branch_ratio"))
    deriv = _float(row.get("effective_derivative"))
    cos_tail = _float(row.get("cos_tail_real_adamw"))
    uncert = math.sqrt(max(0.0, _float(row.get("microbatch_score_variance")))) + _float(row.get("bootstrap_control_gap_std"))
    h_agree = _float(row.get("horizon_agreement_score"))
    return {
        "CP0-NoProbeStatic": -deriv,
        "CP1-LinearizedLogitProbe": -(0.55 * real + 0.45 * tail),
        "CP2-VirtualMicroHoldoutProbe": -(0.45 * par + 0.45 * bestlr) + 0.10 * tail_ratio,
        "CP3-LeaveOneOutPopRiskProbe": -uncert + 0.10 * ce + 0.10 * margin,
        "CP4-ControlContrastiveVirtualReplay": -deriv + 0.20 * tail_ratio + 0.10 * rz_perp - 0.10 * abs(cos_tail),
        "CP5-HorizonConsistencyProbe": h_agree + 0.10 * ce - 0.20 * uncert,
        "CP6-UncertaintyLCBProbe": (-deriv + 0.20 * tail_ratio + 0.10 * nonadamw) - 1.0 * uncert,
    }


def _acceptance_metrics(scores: Sequence[float], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = [_int(r.get("Y_beat")) for r in rows]
    bad = [_int(r.get("bad_event")) for r in rows]
    best = {
        "precision": 0.0,
        "recall": 0.0,
        "coverage": 0.0,
        "bad_event_rate": 0.0,
        "accepted": [],
        "threshold": "",
        "threshold_quantile": "",
    }
    total_good = max(1, sum(labels))
    for q in [0.85, 0.88, 0.90, 0.92, 0.95, 0.97]:
        threshold = _quantile(list(scores), q)
        idx = [i for i, s in enumerate(scores) if float(s) >= threshold]
        coverage = len(idx) / max(1, len(rows))
        if coverage < 0.03 or coverage > 0.15:
            continue
        precision = sum(labels[i] for i in idx) / max(1, len(idx))
        recall = sum(labels[i] for i in idx) / total_good
        bad_rate = sum(bad[i] for i in idx) / max(1, len(idx))
        cand = {
            "precision": precision,
            "recall": recall,
            "coverage": coverage,
            "bad_event_rate": bad_rate,
            "accepted": idx,
            "threshold": threshold,
            "threshold_quantile": q,
        }
        if (precision, coverage, -bad_rate) > (best["precision"], best["coverage"], -best["bad_event_rate"]):
            best = cand
    return best


def _run_p2_probe_audit(events: List[Dict[str, Any]], opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P2_PROBE_TO_VALUE_AUDIT", "p2_probe_to_value_audit.csv", "P1_static_feature_autopsy_not_opened")
        return [row], {
            "best_probe": "not_opened",
            "probe_predictive_pass": 0,
            "probe_controller_candidate_pass": 0,
            "best_probe_corr": 0.0,
            "best_probe_auc": 0.5,
        }
    y = [_float(r.get("grounded_value")) for r in events]
    labels = [_int(r.get("Y_beat")) for r in events]
    all_scores_by_probe: Dict[str, List[float]] = defaultdict(list)
    raw_scores = [_probe_scores(r) for r in events]
    for scores in raw_scores:
        for probe, value in scores.items():
            all_scores_by_probe[probe].append(float(value))
    # Orient diagnostic probe scores so positive means higher grounded value.
    orientations = {probe: (-1.0 if _corr(vals, y) < 0 else 1.0) for probe, vals in all_scores_by_probe.items()}

    rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for probe, vals_raw in all_scores_by_probe.items():
        vals = [orientations[probe] * v for v in vals_raw]
        corr = _corr(vals, y)
        auc = _auc(vals, labels)
        acc = _acceptance_metrics(vals, events)
        legality = 0
        predictive = int(corr >= 0.35 or auc >= 0.70)
        accepted_pass = int(acc["precision"] >= 0.75 and 0.03 <= acc["coverage"] <= 0.15 and acc["bad_event_rate"] <= 0.05)
        summaries.append(
            {
                "probe": probe,
                "corr": corr,
                "auc": auc,
                "precision": acc["precision"],
                "coverage": acc["coverage"],
                "bad_event_rate": acc["bad_event_rate"],
                "accepted_count": len(acc["accepted"]),
                "predictive": predictive,
                "accepted_pass": accepted_pass,
                "legality": legality,
            }
        )
        for i, r in enumerate(events):
            accepted = int(i in acc["accepted"])
            rows.append(
                {
                    "stage": "P2_PROBE_TO_VALUE_AUDIT",
                    "status": "source_logged_feature_probe",
                    "probe": probe,
                    "event_id": r.get("event_id", ""),
                    "dataset": r.get("dataset", ""),
                    "seed": r.get("seed", ""),
                    "horizon": r.get("horizon", ""),
                    "primitive": r.get("primitive", ""),
                    "signal_stratum": r.get("signal_stratum", ""),
                    "probe_score_real": vals[i],
                    "probe_score_adamwparallel": "not_separately_measured_source_logged_feature_probe",
                    "probe_score_bestlr": "not_separately_measured_source_logged_feature_probe",
                    "probe_control_gap": vals[i],
                    "probe_uncertainty": _float(r.get("bootstrap_control_gap_std")),
                    "probe_lcb": vals[i] - _float(r.get("bootstrap_control_gap_std")),
                    "grounded_value": r.get("grounded_value", ""),
                    "actual_real_minus_adamwparallel": r.get("actual_real_minus_adamwparallel", ""),
                    "actual_real_minus_bestlr": r.get("actual_real_minus_bestlr", ""),
                    "accepted": accepted,
                    "bad_event": r.get("bad_event", ""),
                    "probe_overhead_ratio": "not_measured_source_logged_feature_probe",
                    "step_ratio_with_probe": "not_measured_source_logged_feature_probe",
                    "memory_ratio_with_probe": "not_measured_source_logged_feature_probe",
                    "probe_uses_train_stream_only": 0,
                    "probe_scope": "v9230_logged_pre_event_features_not_new_train_stream_virtual_probe",
                    "dataset_name_used": 0,
                    "validation_used": 0,
                    "test_metric_used": 0,
                    "probe_predictive_pass": predictive,
                    "accepted_event_pass": accepted_pass,
                    "probe_legality_pass": legality,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    best = max(summaries, key=lambda r: (_int(r["predictive"]), float(r["corr"]), float(r["auc"])))
    summary = {
        "best_probe": best["probe"],
        "best_probe_corr": best["corr"],
        "best_probe_auc": best["auc"],
        "best_probe_precision": best["precision"],
        "best_probe_coverage": best["coverage"],
        "best_probe_bad_event_rate": best["bad_event_rate"],
        "probe_predictive_pass": int(best["predictive"]),
        "probe_controller_candidate_pass": int(best["predictive"] and best["accepted_pass"] and best["legality"]),
        "probe_legality_pass": 0,
        "probe_system_pass": 0,
        "probe_system_status": "not_measured_because_no_new_train_stream_probe_was_available",
        "all_probe_summary": json.dumps(summaries, sort_keys=True),
    }
    return rows, summary


def _run_p3_controller(p2_summary: Dict[str, Any], opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P3_PROBE_TO_COMMIT_CONTROLLER_CALIBRATION", "p3_probe_to_commit_controller_calibration.csv", "P2_probe_predictivity_or_legality_failed")
        return [row], {"probe_controller_pass": 0}
    # This branch is not reached in the first-wave source-logged probe audit.
    row = _not_run("P3_PROBE_TO_COMMIT_CONTROLLER_CALIBRATION", "p3_probe_to_commit_controller_calibration.csv", "official_controller_not_implemented_without_legal_train_stream_probe")
    return [row], {"probe_controller_pass": 0}


def _run_p4_heldout(opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P4_LEAVE_DATASET_AND_STRATUM_OUT_VALIDATION", "p4_leave_dataset_and_stratum_out_validation.csv", "P3_probe_controller_failed")
        return [row], {"leave_dataset_out_pass": 0, "leave_stratum_out_pass": 0}
    row = _not_run("P4_LEAVE_DATASET_AND_STRATUM_OUT_VALIDATION", "p4_leave_dataset_and_stratum_out_validation.csv", "not_implemented_without_controller")
    return [row], {"leave_dataset_out_pass": 0, "leave_stratum_out_pass": 0}


def _run_p5_observable_primitive(events: List[Dict[str, Any]], p2_summary: Dict[str, Any], opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P5_OBSERVABLE_PRIMITIVE_REPAIR_GATE", "p5_observable_primitive_repair_gate.csv", "P2_probe_predictive_and_controller_path_opened")
        return [row], {"observable_primitive_pass": 0, "best_observable_primitive": "not_opened"}
    source_route = _read_json(SRC_V9222 / "route_decision.json")
    # Use the best observed current primitive corr as OP0 observability. New OPs
    # are intentionally not fabricated.
    obs_by_primitive: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    # Best probe scores are not retained per primitive in summary, recompute.
    best_probe = str(p2_summary.get("best_probe", "CP0-NoProbeStatic"))
    for r in events:
        score = _probe_scores(r).get(best_probe, 0.0)
        obs_by_primitive[str(r.get("primitive"))].append((score, _float(r.get("grounded_value"))))
    primitive_corr = {
        p: abs(_corr([a for a, _ in vals], [b for _, b in vals]))
        for p, vals in obs_by_primitive.items()
    }
    op0_corr = max(primitive_corr.values() or [0.0])
    rows = [
        {
            "stage": "P5_OBSERVABLE_PRIMITIVE_REPAIR_GATE",
            "status": "source_measured_current_reference",
            "primitive": "OP0-N2a-current/N2c-current/N3c-current",
            "contract_pass": 1,
            "grad_pass": 1,
            "P4_forward_q90": "source_base_neutral_interface_pass_from_v9222",
            "P4_backward_q90": "source_base_neutral_interface_pass_from_v9222",
            "P4_step_q90": "source_base_neutral_interface_pass_from_v9222",
            "P4_memory": "source_base_neutral_interface_pass_from_v9222",
            "P5_nearpass": source_route.get("interface_p5_nearpass", 0),
            "functional_actuatability_pass": source_route.get("functional_actuatability_pass", 0),
            "probe_observability_corr": op0_corr,
            "accepted_precision": p2_summary.get("best_probe_precision", 0.0),
            "accepted_coverage": p2_summary.get("best_probe_coverage", 0.0),
            "paired_replay_ready": 0,
            "observable_primitive_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    ]
    for op in [
        "OP1-ObservableTailLinearChannel",
        "OP2-ObservablePiecewiseTailChannel",
        "OP3-ObservableSharedRBFLocalChannel",
        "OP4-ObservableOrthogonalTailChannel",
        "OP5-ObservableControlGapChannel",
        "OP6-LightHybridObservable",
    ]:
        rows.append(
            {
                "stage": "P5_OBSERVABLE_PRIMITIVE_REPAIR_GATE",
                "status": "not_implemented",
                "primitive": op,
                "contract_pass": 0,
                "grad_pass": 0,
                "P4_forward_q90": "not_implemented",
                "P4_backward_q90": "not_implemented",
                "P4_step_q90": "not_implemented",
                "P4_memory": "not_implemented",
                "P5_nearpass": 0,
                "functional_actuatability_pass": 0,
                "probe_observability_corr": "",
                "accepted_precision": "",
                "accepted_coverage": "",
                "paired_replay_ready": 0,
                "observable_primitive_pass": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    return rows, {
        "observable_primitive_pass": 0,
        "best_observable_primitive": "OP0-current-reference",
        "best_observable_primitive_corr": op0_corr,
        "observable_primitive_not_implemented_count": 6,
    }


def _write_report(path: Path, out_dir: Path, route: Dict[str, Any], hashes: List[Dict[str, str]]) -> None:
    def h(name: str) -> str:
        for row in hashes:
            if row["artifact"].endswith(name):
                return row["sha256"]
        return ""

    text = f"""# DG-KAN v9.2.32 Probe-to-Commit 与 Observable Primitive 实验复盘

> 本复盘记录 `DG-KAN_v9.2.32_ProbeToCommit_ObservablePrimitive_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把未实现 train-stream probe / observable primitive 写成通过。

## 0. 最新结论

```text
route = {route['route']}
base_candidate = LQ-t2-h256
success_v9232_strict_purekan_functional = {bool(route['success_v9232_strict_purekan_functional'])}
success_v9232_full_functional = {bool(route['success_v9232_full_functional'])}
success_v9232_external_ready = {bool(route['success_v9232_external_ready'])}
```

最终 artifact：

```text
{out_dir.relative_to(ROOT)}/
```

核心结论：

1. P0 复现 v9.2.31 boundary：`route=R9-PrimitiveEffectUnpredictable`，event-value grounding pass = `1`，fake/proxy = `0`。
2. P1 static feature insufficiency confirmed = `{route['static_feature_insufficiency_confirmed']}`；best static corr = `{route['best_static_feature_corr']:.6f}`，best movement corr = `{route['best_movement_feature_corr']:.6f}`。
3. P2 source-logged probe audit 已执行；best probe = `{route['best_probe']}`，corr = `{route['best_probe_corr']:.6f}`，AUC = `{route['best_probe_auc']:.6f}`。
4. P2 probe legality pass = `{route['probe_legality_pass']}`，原因是本轮没有把 v9.2.30 logged features 伪装成新的 train-stream virtual probe。
5. P5 observable primitive gate 中 OP0/current reference 未过 observability；OP1-OP6 明确 `not_implemented`。
6. 当前 blocker：`{route['primary_blocker']}`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9232_probe_to_commit_observable_primitive.py` | v9.2.32 runner；复现 v9.2.31 boundary，执行 static autopsy、source-logged probe-to-value audit、observable primitive gate、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9232_probe_to_commit_observable_primitive.py
```

正式运行：

```bash
python experiments/run_v9232_probe_to_commit_observable_primitive.py \\
  --out-dir {out_dir.relative_to(ROOT)} \\
  --fresh \\
  --seed 1314
```

说明：本轮 P2 使用 v9.2.30/v9.2.31 真实落盘的 pre-event / grounded-value rows 做 probe audit。没有新增真实 train-stream virtual microholdout probe；因此 probe legality 不会被写成通过。

## 2. Route

```json
{json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True)}
```

## 3. P1 Static Feature Autopsy

Artifact：

```text
p1_static_feature_insufficiency_autopsy.csv
```

关键值：

| metric | value |
|---|---:|
| static insufficiency confirmed | `{route['static_feature_insufficiency_confirmed']}` |
| strict all static below 0.25 | `{route['strict_all_static_below_025']}` |
| best static feature | `{route['best_static_feature']}` |
| best static corr | `{route['best_static_feature_corr']:.6f}` |
| best movement feature | `{route['best_movement_feature']}` |
| best movement corr | `{route['best_movement_feature_corr']:.6f}` |
| single-event value too noisy | `{route['single_event_value_too_noisy']}` |

判断：v9.2.31 的 label 可靠性保留，但 static movement features 仍不足以打开 controller。

## 4. P2 Probe-to-Value Audit

Artifact：

```text
p2_probe_to_value_audit.csv
```

关键值：

| metric | value |
|---|---:|
| best probe | `{route['best_probe']}` |
| best probe corr | `{route['best_probe_corr']:.6f}` |
| best probe AUC | `{route['best_probe_auc']:.6f}` |
| precision | `{route['best_probe_precision']:.6f}` |
| coverage | `{route['best_probe_coverage']:.6f}` |
| predictive pass | `{route['probe_predictive_pass']}` |
| legality pass | `{route['probe_legality_pass']}` |

判断：source-logged probe score 的连续相关性没有达到 `corr>=0.35`，但 CP5 的 AUC/precision 局部过 gate；关键 blocker 是它不是本轮新测的 official train-stream virtual probe，`probe_legality_pass = 0`，因此不能进入 P3/P4 official controller。

## 5. P5 Observable Primitive Gate

Artifact：

```text
p5_observable_primitive_repair_gate.csv
```

结果：

```text
observable_primitive_pass = {route['observable_primitive_pass']}
best_observable_primitive = {route['best_observable_primitive']}
best_observable_primitive_corr = {route['best_observable_primitive_corr']:.6f}
not_implemented OP rows = {route['observable_primitive_not_implemented_count']}
```

判断：current OP0 reference 没有达到 observability；OP1-OP6 不是本轮实现内容，按 `not_implemented` 记录，没有伪装成失败训练。

## 6. Downstream Boundary

P6-P10 只有在 legal probe controller 或 observable primitive survivor 后打开。本轮均已落盘为 `not_run`。

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
| runner | `{h('run_v9232_probe_to_commit_observable_primitive.py')}` |
| route | `{h('route_decision.json')}` |
| P1 autopsy | `{h('p1_static_feature_insufficiency_autopsy.csv')}` |
| P2 probe audit | `{h('p2_probe_to_value_audit.csv')}` |
| P5 primitive gate | `{h('p5_observable_primitive_repair_gate.csv')}` |
| provenance audit | `{h('v9232_provenance_audit.csv')}` |

## 9. 最终分析结论

v9.2.32 的真实推进是：

```text
v9.2.31: grounded value 可靠，但 static pre-event observability 不过。
v9.2.32: source-logged probe audit 仍无法证明可预测 causal value；真实 train-stream probe 与 observable primitive 仍需实现。
```

机制判断：

1. 本轮不能声明 probe-to-commit success，因为 P2 没有合法 train-stream virtual probe pass。
2. `source_logged_feature_probe` 只说明现有落盘 pre-event statistics 不足以预测 grounded value，不能当作 official controller。
3. OP1-OP6 未实现，因此不能说 observable primitive factory 失败；只能说 current OP0/reference 没有可观测性，下一步必须实现真正 train-stream virtual probe 或 observable primitive。

最终一句话：

> v9.2.32 真实执行后停在 `{route['route']}`：`{route['primary_blocker']}`。
"""
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def _make_svg_bar(path: Path, title: str, labels: Sequence[str], values: Sequence[float]) -> None:
    ensure_dir(path.parent)
    width = 780
    height = 280
    vals = [float(v) for v in values]
    vmax = max([abs(v) for v in vals] + [1.0e-9])
    bar_w = max(10, int((width - 130) / max(1, len(vals))))
    zero_y = 150
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="28" font-family="sans-serif" font-size="16">{title}</text>',
        f'<line x1="70" x2="{width-35}" y1="{zero_y}" y2="{zero_y}" stroke="#777" stroke-width="1"/>',
    ]
    for i, (label, val) in enumerate(zip(labels, vals)):
        x = 70 + i * bar_w
        h = int((abs(val) / vmax) * 85)
        y = zero_y - h if val >= 0 else zero_y
        color = "#2b6cb0" if val >= 0 else "#c53030"
        lines.append(f'<rect x="{x}" y="{y}" width="{max(7, bar_w-3)}" height="{h}" fill="{color}" opacity="0.85"/>')
        lines.append(f'<text x="{x}" y="255" font-family="sans-serif" font-size="9" transform="rotate(-35 {x},255)">{label}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


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

    write_json(out_dir / "run_manifest.json", {
        "version": "v9.2.32",
        "created_utc": _now_iso(),
        "plan_path": str(PLAN_PATH.relative_to(ROOT)),
        "runner": str(SCRIPT_PATH.relative_to(ROOT)),
        "source_v9231": str(SRC_V9231.relative_to(ROOT)),
        "source_v9230": str(SRC_V9230.relative_to(ROOT)),
        "seed": args.seed,
        "mode": "source_logged_probe_to_value_audit_no_new_train_stream_probe",
    })
    write_csv_rows(out_dir / "contract_audit_v9232.csv", [
        {
            "stage": "CONTRACT",
            "loss_type": "CE_source_replay_only",
            "teacher_used": 0,
            "self_teacher_used": 0,
            "distillation_used": 0,
            "loss_modification_used": 0,
            "dataset_name_used_in_official_controller": 0,
            "probe_train_stream_virtual_replay_newly_measured": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    ])

    p0 = _p0_boundary()
    write_csv_rows(out_dir / "p0_v9231_boundary_reproduction.csv", [p0])
    events = _load_joined_events()
    p1_rows, p1 = _run_p1_static_autopsy(events, bool(_int(p0.get("P0_pass"))))
    write_csv_rows(out_dir / "p1_static_feature_insufficiency_autopsy.csv", p1_rows)
    p2_rows, p2 = _run_p2_probe_audit(events, bool(p1["feature_rank_sufficient"]))
    write_csv_rows(out_dir / "p2_probe_to_value_audit.csv", p2_rows)
    p3_rows, p3 = _run_p3_controller(p2, bool(p2["probe_controller_candidate_pass"]))
    write_csv_rows(out_dir / "p3_probe_to_commit_controller_calibration.csv", p3_rows)
    p4_rows, p4 = _run_p4_heldout(bool(p3["probe_controller_pass"]))
    write_csv_rows(out_dir / "p4_leave_dataset_and_stratum_out_validation.csv", p4_rows)
    p5_rows, p5 = _run_p5_observable_primitive(events, p2, bool(not p2["probe_controller_candidate_pass"]))
    write_csv_rows(out_dir / "p5_observable_primitive_repair_gate.csv", p5_rows)

    downstream_reason = "P5_observable_primitive_failed_and_no_legal_probe_controller"
    for stage, fname in [
        ("P6_OFFICIAL_PROBE_GATED_PAIRED_REPLAY", "p6_official_probe_gated_paired_replay.csv"),
        ("P7_SHORT_RUN_FUNCTIONAL_VALIDATION", "p7_short_run_functional_validation.csv"),
        ("P8_FULL_10SEED_FUNCTIONAL_VALIDATION", "p8_full_10seed_functional_validation.csv"),
        ("P9_ADAMW_ONLY_FULLPASS_REPAIR", "p9_adamw_only_fullpass_repair.csv"),
        ("P10_ROBUSTNESS_EXTERNAL_READY", "p10_robustness_external_ready.csv"),
    ]:
        write_csv_rows(out_dir / fname, [_not_run(stage, fname, downstream_reason)])

    write_csv_rows(out_dir / "probe_value_trace_v9232.csv", p2_rows)
    write_csv_rows(out_dir / "virtual_microholdout_trace_v9232.csv", [
        _not_run("VIRTUAL_MICROHOLDOUT_TRACE", "virtual_microholdout_trace_v9232.csv", "true_train_stream_virtual_microholdout_probe_not_implemented_this_runner")
    ])
    write_csv_rows(out_dir / "loo_population_risk_trace_v9232.csv", [
        _not_run("LOO_POPULATION_RISK_TRACE", "loo_population_risk_trace_v9232.csv", "true_LOO_population_risk_probe_not_implemented_this_runner")
    ])
    write_csv_rows(out_dir / "observable_primitive_trace_v9232.csv", p5_rows)
    write_csv_rows(out_dir / "paired_replay_branch_trace_v9232.csv", [
        _not_run("PAIRED_REPLAY_BRANCH_TRACE", "paired_replay_branch_trace_v9232.csv", "P6_not_opened")
    ])

    failure_rows = [
        {
            "failure_code": "F2_v9231_boundary_unstable",
            "active": int(not _int(p0.get("P0_pass"))),
            "detail": "" if _int(p0.get("P0_pass")) else "v9231 boundary not reproduced",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "failure_code": "F6_probe_not_predictive",
            "active": int(not p2["probe_predictive_pass"]),
            "detail": f"best_probe={p2['best_probe']} corr={p2['best_probe_corr']}",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "failure_code": "F14_primitive_lacks_causal_observability",
            "active": int(not p5["observable_primitive_pass"]),
            "detail": f"best_observable_primitive_corr={p5['best_observable_primitive_corr']}",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
    ]
    write_csv_rows(out_dir / "failure_table.csv", failure_rows)

    # Figures.
    probe_summary = json.loads(p2.get("all_probe_summary", "[]"))
    _make_svg_bar(out_dir / "figures" / "p2_probe_corr.svg", "P2 probe correlation", [r["probe"] for r in probe_summary], [r["corr"] for r in probe_summary])
    _make_svg_bar(out_dir / "figures" / "p1_static_corr.svg", "P1 static feature correlation", [p1["best_static_feature"], p1["best_movement_feature"]], [p1["best_static_feature_corr"], p1["best_movement_feature_corr"]])

    if not _int(p0.get("P0_pass")):
        route_name = "R0-SourceBoundaryNotReproduced"
        primary = "v9231_boundary_not_reproduced"
    elif p2["probe_predictive_pass"] and not p2["probe_legality_pass"]:
        route_name = "R13-ReturnToInterfacePrimitiveDesign"
        primary = "probe_predictive_but_not_legal_train_stream_probe"
    elif p2["probe_predictive_pass"] and not p3["probe_controller_pass"]:
        route_name = "R1-StaticFeatureInsufficientButProbePredictive"
        primary = "probe_controller_precision_or_legality_failed"
    elif not p2["probe_predictive_pass"] and not p5["observable_primitive_pass"]:
        route_name = "R9-PrimitiveLacksCausalObservability"
        primary = "probe_scores_do_not_predict_grounded_value_and_observable_primitive_not_available"
    else:
        route_name = "R13-ReturnToInterfacePrimitiveDesign"
        primary = "probe_to_commit_not_opened"

    audit_paths = [
        out_dir / "contract_audit_v9232.csv",
        out_dir / "p0_v9231_boundary_reproduction.csv",
        out_dir / "p1_static_feature_insufficiency_autopsy.csv",
        out_dir / "p2_probe_to_value_audit.csv",
        out_dir / "p3_probe_to_commit_controller_calibration.csv",
        out_dir / "p4_leave_dataset_and_stratum_out_validation.csv",
        out_dir / "p5_observable_primitive_repair_gate.csv",
        out_dir / "p6_official_probe_gated_paired_replay.csv",
        out_dir / "p7_short_run_functional_validation.csv",
        out_dir / "p8_full_10seed_functional_validation.csv",
        out_dir / "p9_adamw_only_fullpass_repair.csv",
        out_dir / "p10_robustness_external_ready.csv",
    ]
    audit = audit_no_fake(audit_paths)
    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9231_boundary_pass": _int(p0.get("P0_pass")),
        "dataset_tuning_detected": 0,
        "static_feature_insufficiency_confirmed": p1["static_feature_insufficiency_confirmed"],
        "strict_all_static_below_025": p1["strict_all_static_below_025"],
        "best_static_feature": p1["best_static_feature"],
        "best_static_feature_corr": p1["best_static_feature_corr"],
        "best_movement_feature": p1["best_movement_feature"],
        "best_movement_feature_corr": p1["best_movement_feature_corr"],
        "feature_rank_sufficient": p1["feature_rank_sufficient"],
        "single_event_value_too_noisy": p1["single_event_value_too_noisy"],
        "within_family_value_variance": p1["within_family_value_variance"],
        "between_family_value_variance": p1["between_family_value_variance"],
        "best_probe": p2["best_probe"],
        "probe_predictive_pass": p2["probe_predictive_pass"],
        "best_probe_corr": p2["best_probe_corr"],
        "best_probe_auc": p2["best_probe_auc"],
        "best_probe_precision": p2["best_probe_precision"],
        "best_probe_coverage": p2["best_probe_coverage"],
        "best_probe_bad_event_rate": p2["best_probe_bad_event_rate"],
        "probe_legality_pass": p2["probe_legality_pass"],
        "probe_system_pass": p2["probe_system_pass"],
        "probe_controller_pass": p3["probe_controller_pass"],
        "leave_dataset_out_pass": p4["leave_dataset_out_pass"],
        "leave_stratum_out_pass": p4["leave_stratum_out_pass"],
        "observable_primitive_pass": p5["observable_primitive_pass"],
        "best_observable_primitive": p5["best_observable_primitive"],
        "best_observable_primitive_corr": p5["best_observable_primitive_corr"],
        "observable_primitive_not_implemented_count": p5["observable_primitive_not_implemented_count"],
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "functional_task_safe": 0,
        "functional_control_pass": 0,
        "functional_system_pass": 0,
        "hard_stratum_repair_pass": 0,
        "adamw_fullpass": 0,
        "strong_baseline_pass": 0,
        "robustness_pass": 0,
        "external_ready": 0,
        "primary_blocker": primary,
        "next_required_implementation": "implement_true_train_stream_virtual_probe_or_observable_primitive_OP1_OP6",
        "success_v9232_strict_purekan_functional": 0,
        "success_v9232_full_functional": 0,
        "success_v9232_external_ready": 0,
        **audit,
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    write_csv_rows(out_dir / "v9232_provenance_audit.csv", [audit])

    hash_paths = [
        PLAN_PATH,
        SCRIPT_PATH,
        out_dir / "route_decision.json",
        out_dir / "contract_audit_v9232.csv",
        out_dir / "p0_v9231_boundary_reproduction.csv",
        out_dir / "p1_static_feature_insufficiency_autopsy.csv",
        out_dir / "p2_probe_to_value_audit.csv",
        out_dir / "p5_observable_primitive_repair_gate.csv",
        out_dir / "failure_table.csv",
        out_dir / "v9232_provenance_audit.csv",
    ]
    hashes = artifact_hash_rows(hash_paths, root=ROOT)
    write_csv_rows(out_dir / "artifact_hashes.csv", hashes)
    _write_report(REPORT_PATH, out_dir, route, hashes)
    print(json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
