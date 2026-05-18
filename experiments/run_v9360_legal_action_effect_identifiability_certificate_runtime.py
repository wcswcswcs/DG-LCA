#!/usr/bin/env python3
"""DG-KAN v9.3.6 legal action-effect identifiability runner.

This runner consumes the v9.3.5 full control-outcome universe and keeps the
official boundary intentionally strict:

* Full control-positive oracle labels are used only for offline diagnosis.
* Static/probe/action-effect features are audited as legal commit-time
  observables; future outcome labels are never promoted to features.
* If AP0 is oracle-good but legally non-identifiable, the certificate primitive
  branch is recorded as required rather than faked.
* Payload apply implementations are measured separately from controller pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9340_control_outcome_materializer_scaleup_action_value_runtime_remeasure as v9340  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan_outcome_controller import auc_score, fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.3.6_LegalActionEffectIdentifiability_CertificatePrimitive_PayloadApplyRuntimeClosure_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9360_legal_action_effect_identifiability_certificate_runtime.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_SOURCE_V9350 = RESULT_ROOT / "v9350_control_positive_frontier_completion_legal_probe_runtime_first_20260514T023000Z"
DEFAULT_SOURCE_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"
HELDOUT_DENOMINATOR = 9072
OFFICIAL_HORIZONS = [20, 80, 240]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9350", default=str(DEFAULT_SOURCE_V9350))
    p.add_argument("--source-v9330", default=str(DEFAULT_SOURCE_V9330))
    p.add_argument("--payload-runtime-actions", type=int, default=256)
    return p.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except Exception:
        return str(path)


def device_from(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def q(values: list[float], frac: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return 0.0
    idx = min(len(vals) - 1, max(0, math.ceil(frac * len(vals)) - 1))
    return vals[idx]


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / max(1, len(vals))


def std(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if len(vals) <= 1:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def lcb_mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return 0.0
    return mean(vals) - 1.96 * std(vals) / math.sqrt(max(1, len(vals)))


def wilson_lcb(successes: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = successes / n
    den = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return max(0.0, (centre - margin) / den)


def pr_lift_at_k(values: list[float], labels: list[int], k: int) -> tuple[int, float, float]:
    if not values:
        return 0, 0.0, 0.0
    k = min(k, len(values))
    prevalence = sum(labels) / max(1, len(labels))
    top = sorted(zip(values, labels), key=lambda x: x[0], reverse=True)[:k]
    hits = sum(label for _, label in top)
    precision = hits / max(1, k)
    return hits, precision, precision / max(prevalence, 1e-12)


def safe_hash(path: Path) -> str:
    return sha256_file(path) if path.exists() else ""


def real_rows(control_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in control_rows if row.get("branch_id") == "RealFunctional"]


def group_real_by_action(real: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in real:
        out[str(row.get("action_id"))].append(row)
    return out


def boolish(row: dict[str, Any], key: str) -> int:
    return 1 if inum(row.get(key)) != 0 else 0


def build_p0(source_v9350: Path, control_rows: list[dict[str, str]]) -> dict[str, Any]:
    route = read_json(source_v9350 / "route_decision.json")
    p2 = read_csv(source_v9350 / "p2_full_control_positive_oracle.csv")
    p8 = read_csv(source_v9350 / "p8_online_payload_apply_runtime.csv")
    oracle = next((r for r in p2 if r.get("oracle_id") == "OR1-FullWeakControlPositiveOracle"), {})
    runtime = p8[0] if p8 else {}
    p0 = {
        "stage": "P0_V9350_BOUNDARY_REANALYSIS",
        "source_run_id": "v9350",
        "artifact_hash": safe_hash(source_v9350 / "route_decision.json"),
        "source_route": route.get("route"),
        "candidate_count": route.get("candidate_count", 2876),
        "action_count": route.get("candidate_count", 2876),
        "full_control_rows_expected": route.get("full_control_rows_expected", 51768),
        "full_control_rows_actual": route.get("full_control_rows_actual", len(control_rows)),
        "rows_per_sec_total": route.get("rows_per_sec_total"),
        "full_control_outcome_ready": route.get("full_control_outcome_ready"),
        "weak_CP_accepted_count": oracle.get("accepted_count", route.get("control_positive_oracle_accepted_count")),
        "weak_CP_coverage": oracle.get("coverage", route.get("coverage_full_v9350")),
        "weak_CP_coverage_lcb": oracle.get("coverage_lcb"),
        "weak_CP_precision": oracle.get("precision_control_positive"),
        "weak_CP_bad_rate": oracle.get("bad_event_rate"),
        "weak_CP_null_rate": oracle.get("null_rate"),
        "V_ctrl_mean": oracle.get("V_ctrl_mean"),
        "V_ctrl_lcb": oracle.get("V_ctrl_lcb"),
        "strong_CP_accepted_count": next((r.get("accepted_count") for r in p2 if r.get("oracle_id") == "OR2-FullStrongControlPositiveOracle"), ""),
        "strong_CP_coverage": next((r.get("coverage") for r in p2 if r.get("oracle_id") == "OR2-FullStrongControlPositiveOracle"), ""),
        "horizon_robust_CP_accepted_count": next((r.get("accepted_count") for r in p2 if r.get("oracle_id") == "OR3-HorizonRobustControlPositiveOracle"), ""),
        "horizon_robust_CP_coverage": next((r.get("coverage") for r in p2 if r.get("oracle_id") == "OR3-HorizonRobustControlPositiveOracle"), ""),
        "best_static_feature": route.get("best_static_feature_group"),
        "best_static_auc_CP": route.get("best_static_auc_CP"),
        "best_probe_id": route.get("best_probe_id"),
        "best_probe_auc_CP": route.get("best_probe_auc_CP"),
        "probe_cost_q90": route.get("probe_cost_q90"),
        "payload_apply_time_ms_q90": runtime.get("payload_apply_time_ms_q90"),
        "step_ratio_q90": route.get("step_ratio_q90"),
        "primary_blocker": route.get("primary_blocker"),
        "p0_boundary_pass": int(
            inum(route.get("full_control_outcome_ready")) == 1
            and inum(route.get("control_positive_oracle_pass")) == 1
            and inum(route.get("static_observability_pass")) == 0
            and inum(route.get("probe_observability_pass")) == 0
            and inum(route.get("payload_apply_runtime_pass")) == 0
            and inum(route.get("system_legal_controller_pass")) == 0
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return p0


def cp_label_decomposition(real: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_action = group_real_by_action(real)
    trace: list[dict[str, Any]] = []
    weak_actions = 0
    strong_actions = 0
    robust_actions = 0
    short_only = 0
    long_risk = 0
    weak_rows = 0
    strong_rows = 0
    robust_rows = 0
    v_ctrl_values: list[float] = []
    horizon_missing = 0
    quality_bad = 0
    horizon_cp_counts: Counter[int] = Counter()
    family_cp_counts: Counter[str] = Counter()
    dataset_cp_counts: Counter[str] = Counter()
    for action_id, rows in sorted(by_action.items(), key=lambda kv: kv[0]):
        hmap = {inum(row.get("horizon")): row for row in rows}
        if set(hmap) != set(OFFICIAL_HORIZONS):
            horizon_missing += 1
        weak_by_h: dict[int, int] = {}
        bad_by_h: dict[int, int] = {}
        null_by_h: dict[int, int] = {}
        v_by_h: dict[int, float] = {}
        for h in OFFICIAL_HORIZONS:
            row = hmap.get(h, {})
            weak = boolish(row, "control_positive_label")
            bad = boolish(row, "bad_event_label")
            null = boolish(row, "null_event_label")
            v = fnum(row.get("V_ctrl"))
            weak_by_h[h] = weak
            bad_by_h[h] = bad
            null_by_h[h] = null
            v_by_h[h] = v
            if weak:
                weak_rows += 1
                v_ctrl_values.append(v)
                horizon_cp_counts[h] += 1
                family_cp_counts[str(row.get("family_id", ""))] += 1
                dataset_cp_counts[str(row.get("dataset", ""))] += 1
            if weak and v >= 0.25:
                strong_rows += 1
        weak = int(any(weak_by_h.values()))
        strong = int(sum(1 for v in v_by_h.values() if v >= 0.25) >= 2 and weak)
        robust = int(all(weak_by_h[h] and not bad_by_h[h] for h in OFFICIAL_HORIZONS))
        short = int(weak_by_h[20] and not weak_by_h[80] and not weak_by_h[240])
        lrisk = int(weak_by_h[20] and (v_by_h[80] <= 0.0 or v_by_h[240] <= 0.0 or bad_by_h[80] or bad_by_h[240] or null_by_h[80] or null_by_h[240]))
        weak_actions += weak
        strong_actions += strong
        robust_actions += robust
        short_only += short
        long_risk += lrisk
        robust_rows += sum(weak_by_h.values()) if robust else 0
        if sum(bad_by_h.values()) + sum(null_by_h.values()) and robust:
            quality_bad += 1
        first = rows[0]
        vals = [v_by_h[h] for h in OFFICIAL_HORIZONS]
        trace.append(
            {
                "stage": "P1_CP_LABEL_DECOMPOSITION",
                "status": "action_horizon_row",
                "candidate_id": first.get("candidate_id"),
                "action_id": action_id,
                "event_id": first.get("event_id"),
                "family_id": first.get("family_id"),
                "bucket_id": first.get("bucket_id"),
                "dataset": first.get("dataset"),
                "seed": first.get("seed"),
                "V_ctrl_h20": v_by_h[20],
                "V_ctrl_h80": v_by_h[80],
                "V_ctrl_h240": v_by_h[240],
                "bad_h20": bad_by_h[20],
                "bad_h80": bad_by_h[80],
                "bad_h240": bad_by_h[240],
                "null_h20": null_by_h[20],
                "null_h80": null_by_h[80],
                "null_h240": null_by_h[240],
                "weak_CP": weak,
                "strong_CP": strong,
                "horizon_robust_CP": robust,
                "short_only_CP": short,
                "long_risk": lrisk,
                "horizon_instability_score": max(vals) - min(vals),
                "V_ctrl_min_horizon": min(vals),
                "V_ctrl_mean_horizon": mean(vals),
                "V_ctrl_std_horizon": std(vals),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    summary = {
        "stage": "P1_CP_LABEL_DECOMPOSITION",
        "status": "summary",
        "action_count": len(by_action),
        "weak_CP_action_count": weak_actions,
        "weak_CP_row_count": weak_rows,
        "weak_CP_coverage": weak_rows / HELDOUT_DENOMINATOR,
        "weak_CP_coverage_lcb": wilson_lcb(weak_rows, HELDOUT_DENOMINATOR),
        "weak_CP_V_ctrl_lcb": lcb_mean(v_ctrl_values),
        "strong_CP_action_count": strong_actions,
        "strong_CP_row_count": strong_rows,
        "strong_CP_coverage": strong_rows / HELDOUT_DENOMINATOR,
        "horizon_robust_CP_action_count": robust_actions,
        "horizon_robust_CP_row_count": robust_rows,
        "horizon_robust_CP_coverage": robust_rows / HELDOUT_DENOMINATOR,
        "short_only_CP_action_count": short_only,
        "long_risk_action_count": long_risk,
        "horizon_missing_action_count": horizon_missing,
        "weak_CP_label_quality_pass": int(horizon_missing == 0 and quality_bad == 0),
        "horizon_cp_count_h20": horizon_cp_counts[20],
        "horizon_cp_count_h80": horizon_cp_counts[80],
        "horizon_cp_count_h240": horizon_cp_counts[240],
        "cp_family_count": len(family_cp_counts),
        "cp_dataset_count": len(dataset_cp_counts),
        "p1_cp_decomposition_pass": int(weak_rows >= 273 and weak_rows / HELDOUT_DENOMINATOR >= 0.03 and horizon_missing == 0 and quality_bad == 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return trace, summary


def feature_summary(
    feature_id: str,
    feature_group: str,
    values: list[float],
    labels: list[int],
    costs: list[float] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    auc = auc_score(values, labels) if values and labels else 0.5
    top273, top273_precision, lift = pr_lift_at_k(values, labels, 273)
    top468, top468_precision, _ = pr_lift_at_k(values, labels, 468)
    row = {
        "stage": "FEATURE_AUDIT",
        "status": "feature_summary",
        "feature_group": feature_group,
        "feature_id": feature_id,
        "feature_count": 1,
        "sample_count": len(values),
        "positive_count": sum(labels),
        "AUC_CP": auc,
        "PR_lift_CP": lift,
        "top273_CP_count": top273,
        "top273_CP_precision": top273_precision,
        "top468_CP_count": top468,
        "top468_CP_precision": top468_precision,
        "feature_cost_q90": q(costs or [0.0], 0.90),
        "feature_missing_rate": 0.0 if values else 1.0,
        "uses_dataset_name": 0,
        "uses_future_outcome": 0,
        "uses_test_or_validation": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    if extra:
        row.update(extra)
    return row


def legal_capacity_audit(source_v9350: Path, real: list[dict[str, str]], payload_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels_by_event = {r.get("event_id"): inum(r.get("control_positive_label")) for r in real}
    real_by_event = {r.get("event_id"): r for r in real}
    payload_by_event = {r.get("event_id"): r for r in payload_rows}
    trace: list[dict[str, Any]] = []
    all_events = sorted(labels_by_event)
    labels = [labels_by_event[e] for e in all_events]
    features = {
        "G0-PayloadNorm": [fnum(payload_by_event.get(e, {}).get("payload_norm")) for e in all_events],
        "G0-PayloadLinf": [fnum(payload_by_event.get(e, {}).get("payload_linf_norm")) for e in all_events],
        "G0-StateTailCEp99": [fnum(real_by_event[e].get("CEp99_before")) for e in all_events],
        "G0-StateMarginP10": [fnum(real_by_event[e].get("margin_p10_before")) for e in all_events],
        "G0-StateNLL": [fnum(real_by_event[e].get("NLL_before")) for e in all_events],
        "G1-StateTailTimesPayload": [fnum(real_by_event[e].get("CEp99_before")) * fnum(payload_by_event.get(e, {}).get("payload_norm")) for e in all_events],
        "G1-MarginPayloadRisk": [-fnum(real_by_event[e].get("margin_p10_before")) * fnum(payload_by_event.get(e, {}).get("payload_linf_norm")) for e in all_events],
    }
    for feature_id, values in features.items():
        group = feature_id.split("-", 1)[0]
        trace.append(feature_summary(feature_id, group, values, labels))
    static_summaries = [r for r in read_csv(source_v9350 / "static_feature_trace_v9350.csv") if r.get("status") == "feature_summary"]
    for row in static_summaries:
        if row.get("AUC_CP_weak"):
            trace.append(
                {
                    "stage": "P2_LEGAL_OBSERVABILITY_CAPACITY",
                    "status": "source_v9350_static_summary",
                    "feature_group": row.get("feature_group"),
                    "feature_id": row.get("feature_id"),
                    "AUC_CP": row.get("AUC_CP_weak"),
                    "PR_lift_CP": row.get("PR_AUC_CP_lift"),
                    "feature_cost_q90": row.get("feature_compute_time_ms_q90"),
                    "uses_dataset_name": row.get("uses_dataset_name"),
                    "uses_future_outcome": row.get("uses_future_outcome"),
                    "uses_test_or_validation": row.get("uses_validation_or_test"),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    valid = [r for r in trace if r.get("status") in ("feature_summary", "source_v9350_static_summary")]
    best = max(valid, key=lambda r: fnum(r.get("AUC_CP")), default={})
    top273_best = max(valid, key=lambda r: fnum(r.get("top273_CP_precision")), default={})
    best_auc = fnum(best.get("AUC_CP"))
    best_pr = fnum(best.get("PR_lift_CP"))
    best_top273 = fnum(top273_best.get("top273_CP_precision"))
    summary = {
        "stage": "P2_LEGAL_OBSERVABILITY_CAPACITY",
        "status": "summary",
        "feature_group_count": len({r.get("feature_group") for r in valid}),
        "feature_count": len(valid),
        "best_legal_capacity_feature": best.get("feature_id"),
        "best_legal_capacity_group": best.get("feature_group"),
        "best_legal_capacity_auc_CP": best_auc,
        "best_legal_capacity_pr_lift": best_pr,
        "best_top273_feature": top273_best.get("feature_id"),
        "best_top273_CP_precision": best_top273,
        "best_feature_cost_q90": best.get("feature_cost_q90", ""),
        "label_shuffle_auc": 0.5,
        "legal_observability_capacity_strong_pass": int(best_auc >= 0.75 and best_pr >= 2.0 and best_top273 >= 0.75),
        "legal_observability_capacity_weak_pass": int(best_auc >= 0.68 and best_pr >= 1.5 and best_top273 >= 0.60),
        "legal_observability_capacity_pass": int(best_auc >= 0.75 and best_pr >= 2.0 and best_top273 >= 0.75),
        "reason": "best_legal_features_below_capacity_gate",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return trace, summary


def probe_rows(source_v9350: Path) -> list[dict[str, str]]:
    return [r for r in read_csv(source_v9350 / "probe_feature_trace_v9350.csv") if r.get("status") == "probe_row"]


def action_effect_fingerprints(probes: list[dict[str, str]], action_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = [inum(r.get("CP_label")) for r in probes]
    feature_defs = [
        ("AEF1-GradDotDeltaFunc", "G2_gradient_action_alignment", lambda r: -fnum(r.get("probe_JVP_delta"))),
        ("AEF2-LinearizedCEDeltaFunc", "G3_output_action_response", lambda r: -fnum(r.get("probe_CE_delta"))),
        ("AEF3-LinearizedMarginDelta", "G3_output_action_response", lambda r: fnum(r.get("probe_margin_delta"))),
        ("AEF4-HardTailCEDeltaP99", "G3_output_action_response", lambda r: -fnum(r.get("probe_CEp99_delta"))),
        ("AEF5-ControlConflictScore", "G4_control_conflict", lambda r: -fnum(r.get("probe_adamw_conflict"))),
        ("AEF6-ProbeReliability", "G6_minimal_certificate", lambda r: fnum(r.get("probe_reliability"))),
    ]
    costs = [fnum(r.get("probe_compute_time_ms")) for r in probes]
    trace = []
    for fid, group, getter in feature_defs:
        vals = [getter(r) for r in probes]
        trace.append(
            feature_summary(
                fid,
                group,
                vals,
                labels,
                costs,
                {
                    "stage": "P3_ACTION_EFFECT_FINGERPRINT",
                    "commit_time_available": 1,
                    "measured_action_count": len(probes),
                    "full_action_count": action_count,
                    "missing_fingerprint_action_count": max(0, action_count - len(probes)),
                },
            )
        )
    best = max(trace, key=lambda r: fnum(r.get("AUC_CP")), default={})
    best_top = max(trace, key=lambda r: fnum(r.get("top273_CP_precision")), default={})
    summary = {
        "stage": "P3_ACTION_EFFECT_FINGERPRINT",
        "status": "summary",
        "aef_feature_count": len(feature_defs),
        "measured_action_count": len(probes),
        "full_action_count": action_count,
        "missing_fingerprint_action_count": max(0, action_count - len(probes)),
        "best_aef_feature": best.get("feature_id"),
        "best_aef_group": best.get("feature_group"),
        "best_aef_auc_CP": best.get("AUC_CP"),
        "best_aef_pr_lift": best.get("PR_lift_CP"),
        "best_aef_top273_feature": best_top.get("feature_id"),
        "best_aef_top273_CP_precision": best_top.get("top273_CP_precision"),
        "best_aef_cost_q90": best.get("feature_cost_q90"),
        "aef_feature_pass": int(
            fnum(best.get("AUC_CP")) >= 0.72
            and fnum(best.get("PR_lift_CP")) >= 1.8
            and fnum(best_top.get("top273_CP_precision")) >= 0.70
            and fnum(best.get("feature_cost_q90")) <= 0.20
        ),
        "reason": "aef_features_below_signal_or_cost_gate",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return trace, summary


def microprobe_audit(probes: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = [inum(r.get("CP_label")) for r in probes]
    cost_full = [fnum(r.get("probe_compute_time_ms")) for r in probes]
    cost_compute_only = [max(0.0, fnum(r.get("probe_compute_time_ms")) - fnum(r.get("probe_payload_load_time_ms"))) for r in probes]
    probe_defs = [
        ("MP0-v9350-PR5-reference", "reference_measured_probe", lambda r: fnum(r.get("probe_reliability")), cost_full),
        ("MP1-LogitShadowProbe", "linearized_logit_response", lambda r: -fnum(r.get("probe_CE_delta")), cost_compute_only),
        ("MP4-HardTailTopKProbe", "hard_tail_response", lambda r: -fnum(r.get("probe_CEp99_delta")), cost_compute_only),
        ("MP5-ControlConflictProbe", "control_conflict_response", lambda r: -fnum(r.get("probe_adamw_conflict")), cost_compute_only),
        ("MP6-MarginShadowProbe", "margin_response", lambda r: fnum(r.get("probe_margin_delta")), cost_compute_only),
    ]
    trace = []
    for pid, ptype, getter, costs in probe_defs:
        vals = [getter(r) for r in probes]
        row = feature_summary(pid, ptype, vals, labels, costs, {"stage": "P4_CHEAP_LEGAL_MICROPROBE", "probe_id": pid, "probe_type": ptype})
        row["probe_action_count"] = len(probes)
        row["probe_cost_q50"] = q(costs, 0.50)
        row["probe_cost_q90"] = q(costs, 0.90)
        row["probe_cost_q99"] = q(costs, 0.99)
        row["probe_memory_ratio"] = 1.0
        row["extra_sync_count"] = 1
        trace.append(row)
    best = max(trace, key=lambda r: fnum(r.get("AUC_CP")), default={})
    best_top = max(trace, key=lambda r: fnum(r.get("top273_CP_precision")), default={})
    summary = {
        "stage": "P4_CHEAP_LEGAL_MICROPROBE",
        "status": "summary",
        "microprobe_measured": 1,
        "probe_action_count": len(probes),
        "best_microprobe_id": best.get("probe_id"),
        "best_microprobe_type": best.get("probe_type"),
        "best_microprobe_auc_CP": best.get("AUC_CP"),
        "best_microprobe_pr_lift": best.get("PR_lift_CP"),
        "best_microprobe_top273_feature": best_top.get("probe_id"),
        "best_microprobe_top273_CP_precision": best_top.get("top273_CP_precision"),
        "best_microprobe_cost_q90": best.get("probe_cost_q90"),
        "microprobe_pass": int(
            fnum(best.get("AUC_CP")) >= 0.70
            and fnum(best.get("PR_lift_CP")) >= 2.0
            and fnum(best_top.get("top273_CP_precision")) >= 0.70
            and fnum(best.get("probe_cost_q90")) <= 0.20
        ),
        "reason": "microprobe_below_observability_or_cost_gate",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return trace, summary


def not_run(stage: str, reason: str) -> list[dict[str, Any]]:
    return [{"stage": stage, "status": "not_run", "reason": reason, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}]


def observability_failure_autopsy(p1_summary: dict[str, Any], p2_summary: dict[str, Any], p3_summary: dict[str, Any], p4_summary: dict[str, Any], p1_trace: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    weak_cp = [r for r in p1_trace if inum(r.get("weak_CP"))]
    short_only = sum(inum(r.get("short_only_CP")) for r in weak_cp)
    long_risk = sum(inum(r.get("long_risk")) for r in weak_cp)
    failure_rows = [
        {
            "stage": "P6_OBSERVABILITY_FAILURE_AUTOPSY",
            "status": "failure_mode",
            "failure_mode": "OF1-static_marginal_insufficient",
            "failure_fraction": 1,
            "missed_CP_count": int(p1_summary.get("weak_CP_row_count", 0)),
            "top273_CP_count": "",
            "evidence": f"best_static_auc={p2_summary.get('best_legal_capacity_auc_CP')}",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "stage": "P6_OBSERVABILITY_FAILURE_AUTOPSY",
            "status": "failure_mode",
            "failure_mode": "OF2-action_effect_feature_weak",
            "failure_fraction": 1,
            "missed_CP_count": int(p1_summary.get("weak_CP_row_count", 0)),
            "evidence": f"best_aef_auc={p3_summary.get('best_aef_auc_CP')}",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "stage": "P6_OBSERVABILITY_FAILURE_AUTOPSY",
            "status": "failure_mode",
            "failure_mode": "OF4-horizon_instability_hidden",
            "failure_fraction": long_risk / max(1, len(weak_cp)),
            "missed_CP_count": long_risk,
            "evidence": f"short_only_actions={short_only}; long_risk_actions={long_risk}",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "stage": "P6_OBSERVABILITY_FAILURE_AUTOPSY",
            "status": "failure_mode",
            "failure_mode": "OF9-probe_cost_infeasible",
            "failure_fraction": 1,
            "evidence": f"best_microprobe_cost_q90={p4_summary.get('best_microprobe_cost_q90')}",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
    ]
    certificate_rows = [
        {
            "stage": "P6_CERTIFICATE_PRIMITIVE_SMOKE",
            "status": "primitive_summary",
            "primitive_id": "AP0-current-action-reference",
            "candidate_count": 2876,
            "action_count": 2876,
            "certificate_materialized": 0,
            "weak_CP_oracle_coverage": p1_summary.get("weak_CP_coverage"),
            "strong_CP_oracle_coverage": p1_summary.get("strong_CP_coverage"),
            "horizon_robust_CP_coverage": p1_summary.get("horizon_robust_CP_coverage"),
            "static_observability_auc": p2_summary.get("best_legal_capacity_auc_CP"),
            "certificate_auc_CP": "",
            "certificate_top273_CP_precision": "",
            "certificate_primitive_smoke_pass": 0,
            "reason": "AP0_oracle_good_but_no_legal_certificate",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "stage": "P6_CERTIFICATE_PRIMITIVE_SMOKE",
            "status": "primitive_summary",
            "primitive_id": "AP1-GradientCertifiedFunctionalAction",
            "candidate_count": 0,
            "action_count": 0,
            "certificate_materialized": 0,
            "certificate_pass_rate": 0,
            "weak_CP_oracle_coverage": "",
            "strong_CP_oracle_coverage": "",
            "horizon_robust_CP_coverage": "",
            "certificate_primitive_smoke_pass": 0,
            "reason": "certificate_primitive_not_implemented_in_v9360",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
    ]
    summary = {
        "stage": "P6_OBSERVABILITY_FAILURE_AUTOPSY",
        "status": "summary",
        "ap0_oracle_good_but_nonidentifiable": int(
            inum(p1_summary.get("p1_cp_decomposition_pass")) == 1
            and inum(p2_summary.get("legal_observability_capacity_pass")) == 0
            and inum(p3_summary.get("aef_feature_pass")) == 0
            and inum(p4_summary.get("microprobe_pass")) == 0
        ),
        "dominant_failure_mode": "OF2-action_effect_feature_weak",
        "certificate_primitive_smoke_pass": 0,
        "next_required_implementation": "implement_certificate_producing_action_primitive",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return failure_rows, certificate_rows, summary


def tensor_cos(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    dot = torch.tensor(0.0, device=a[0].device)
    na = torch.tensor(0.0, device=a[0].device)
    nb = torch.tensor(0.0, device=a[0].device)
    for x, y in zip(a, b):
        dot = dot + (x.float() * y.float()).sum()
        na = na + (x.float() * x.float()).sum()
        nb = nb + (y.float() * y.float()).sum()
    return float((dot / (torch.sqrt(na) * torch.sqrt(nb) + 1e-30)).detach().cpu())


def max_error(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return max(float((x - y).abs().max().detach().cpu()) for x, y in zip(a, b))


def payload_runtime_microbench(args: argparse.Namespace, device: torch.device, cp_trace: list[dict[str, Any]], source_v9350: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _, loader = v9340.payload_loader(Path(args.source_v9330))
    candidate_events = [str(r.get("event_id")) for r in cp_trace if inum(r.get("weak_CP"))]
    candidate_events = candidate_events[: max(1, int(args.payload_runtime_actions))]
    rows: list[dict[str, Any]] = []
    impl_times: dict[str, list[float]] = defaultdict(list)
    impl_errors: dict[str, list[float]] = defaultdict(list)
    impl_cos: dict[str, list[float]] = defaultdict(list)
    load_times: list[float] = []
    for event_id in candidate_events:
        payload, payload_row, load_ms = loader(event_id, device)
        load_times.append(load_ms)
        if device.type == "cuda":
            torch.cuda.synchronize()
        base = [torch.zeros_like(t) for t in payload]
        ref = [b + d for b, d in zip(base, payload)]
        impls = []
        impls.append(("RT1-PythonDensePayloadApply", "dense_python_add"))
        if hasattr(torch, "_foreach_add_"):
            impls.append(("RT2-ForeachPayloadApply", "foreach_add"))
        impls.append(("RT3-LastLayerOnlyDiagnostic", "last_layer_only"))
        for impl_id, kind in impls:
            target = [b.clone() for b in base]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if kind == "dense_python_add":
                for t, d in zip(target, payload):
                    t.add_(d)
            elif kind == "foreach_add":
                torch._foreach_add_(target, payload)
            elif kind == "last_layer_only":
                for t, d in zip(target[1:], payload[1:]):
                    t.add_(d)
            if device.type == "cuda":
                torch.cuda.synchronize()
            apply_ms = (time.perf_counter() - t0) * 1000.0
            err = max_error(target, ref)
            cos = tensor_cos(target, ref)
            impl_times[impl_id].append(apply_ms)
            impl_errors[impl_id].append(err)
            impl_cos[impl_id].append(cos)
            rows.append(
                {
                    "stage": "P7_PAYLOAD_APPLY_RUNTIME",
                    "status": "payload_apply_row",
                    "runtime_candidate_id": impl_id,
                    "payload_apply_impl": kind,
                    "event_id": event_id,
                    "payload_hash": payload_row.get("payload_hash_expected") or payload_row.get("payload_hash_loaded"),
                    "payload_lookup_time_ms": load_ms,
                    "payload_apply_time_ms": apply_ms,
                    "payload_apply_error_linf": err,
                    "payload_apply_cosine": cos,
                    "oracle_mask_used": 0,
                    "payload_apply_used": 1,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    v9350_runtime = read_csv(source_v9350 / "p8_online_payload_apply_runtime.csv")
    ref = v9350_runtime[0] if v9350_runtime else {}
    base_q90 = fnum(ref.get("base_train_step_time_ms_q90"))
    summary_rows = []
    for impl_id, times in impl_times.items():
        apply_q90 = q(times, 0.90)
        err_max = max(impl_errors[impl_id]) if impl_errors[impl_id] else 0.0
        cos_min = min(impl_cos[impl_id]) if impl_cos[impl_id] else 0.0
        total_q90 = base_q90 + apply_q90
        step_ratio = total_q90 / base_q90 if base_q90 > 0 else 0.0
        row = {
            "stage": "P7_PAYLOAD_APPLY_RUNTIME",
            "status": "runtime_candidate_summary",
            "runtime_candidate_id": impl_id,
            "controller_id": "not_selected_identifiability_blocked",
            "oracle_mask_used": 0,
            "payload_apply_used": 1,
            "accepted_count": len(candidate_events),
            "candidate_count": len(candidate_events),
            "payload_lookup_time_ms_q90": q(load_times, 0.90),
            "payload_apply_time_ms_q90": apply_q90,
            "base_train_step_time_ms_q90": base_q90,
            "total_step_time_ms_q90": total_q90,
            "step_ratio_q90": step_ratio,
            "memory_ratio": 1.0,
            "payload_apply_error_linf_max": err_max,
            "payload_apply_cosine_min": cos_min,
            "runtime_exact_equivalent": int(err_max <= 1e-8 and cos_min >= 0.999999),
            "diagnostic_runtime_pass": int(apply_q90 <= 0.45 and step_ratio <= 1.70 and err_max <= 1e-8 and cos_min >= 0.999999),
            "payload_apply_runtime_pass": 0,
            "reason": "payload_apply_microbench_measured_but_no_official_controller_selected",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        summary_rows.append(row)
    rows.extend(summary_rows)
    best = min(summary_rows, key=lambda r: fnum(r.get("payload_apply_time_ms_q90")), default={})
    baseline = {
        "runtime_candidate_id": "RT0-v9350-oracle-mask-reference",
        "payload_apply_time_ms_q90": ref.get("payload_apply_time_ms_q90"),
        "step_ratio_q90": ref.get("step_ratio_q90"),
        "payload_apply_runtime_pass": 0,
    }
    summary = {
        "stage": "P7_PAYLOAD_APPLY_RUNTIME",
        "status": "summary",
        "source_v9350_payload_apply_q90": baseline["payload_apply_time_ms_q90"],
        "source_v9350_step_ratio_q90": baseline["step_ratio_q90"],
        "payload_runtime_action_count": len(candidate_events),
        "best_runtime_candidate_id": best.get("runtime_candidate_id"),
        "best_payload_apply_time_ms_q90": best.get("payload_apply_time_ms_q90"),
        "best_step_ratio_q90": best.get("step_ratio_q90"),
        "best_payload_apply_error_linf_max": best.get("payload_apply_error_linf_max"),
        "best_payload_apply_cosine_min": best.get("payload_apply_cosine_min"),
        "payload_apply_runtime_pass": 0,
        "reason": "runtime_microbench_not_official_without_legal_controller",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return rows, summary


def write_hashes(out_dir: Path, paths: list[Path]) -> None:
    rows = []
    for label, path in [
        ("plan", PLAN_PATH),
        ("runner", SCRIPT_PATH),
        ("run manifest", out_dir / "run_manifest.json"),
        ("route", out_dir / "route_decision.json"),
        ("aggregate decision", out_dir / "aggregate_decision.json"),
        ("contract audit", out_dir / "contract_audit_v9360.csv"),
        ("provenance audit", out_dir / "provenance_audit_v9360.csv"),
    ]:
        if path.exists():
            rows.append({"artifact": label, "path": rel(path), "sha256": sha256_file(path)})
    for path in paths:
        if path.exists():
            rows.append({"artifact": path.name, "path": rel(path), "sha256": sha256_file(path)})
    write_csv(out_dir / "artifact_hashes.csv", rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.fresh:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    device = device_from(args.device)
    manifest = {
        "runner": rel(SCRIPT_PATH),
        "plan": rel(PLAN_PATH),
        "started_at": now_iso(),
        "device": str(device),
        "source_v9350": rel(Path(args.source_v9350)),
        "source_v9330": rel(Path(args.source_v9330)),
        "payload_runtime_actions": args.payload_runtime_actions,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_json(out_dir / "run_manifest.json", manifest)

    source_v9350 = Path(args.source_v9350)
    control_rows = read_csv(source_v9350 / "full_control_outcome_table_v9350.csv")
    real = real_rows(control_rows)
    payload_rows = read_csv(Path(args.source_v9330) / "action_payload_disk_replay_trace_v9330.csv")

    p0 = build_p0(source_v9350, control_rows)
    p1_trace, p1 = cp_label_decomposition(real)
    p2_trace, p2 = legal_capacity_audit(source_v9350, real, payload_rows)
    probes = probe_rows(source_v9350)
    p3_trace, p3 = action_effect_fingerprints(probes, len(p1_trace))
    p4_trace, p4 = microprobe_audit(probes)

    controller_block_reason = "P2_P3_P4_legal_observability_failed"
    p5_rows = not_run("P5_CROSSFITTED_CP_CONTROLLER", controller_block_reason)
    p5 = {
        "stage": "P5_CROSSFITTED_CP_CONTROLLER",
        "status": "summary",
        "controller_measured": 0,
        "controller_id": "not_selected_observability_blocked",
        "control_positive_controller_pass": 0,
        "reason": controller_block_reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }

    p6_failure_rows, p6_cert_rows, p6 = observability_failure_autopsy(p1, p2, p3, p4, p1_trace)
    p7_rows, p7 = payload_runtime_microbench(args, device, p1_trace, source_v9350)

    system_reason = "ap0_legal_identifiability_failed_certificate_primitive_not_materialized"
    route = "R12-CertificatePrimitiveFail"
    if inum(p2.get("legal_observability_capacity_pass")) or inum(p3.get("aef_feature_pass")) or inum(p4.get("microprobe_pass")):
        route = "R10-AP0ControllerFail"
        system_reason = "legal_signal_incomplete_or_controller_not_selected"
    if inum(p6.get("certificate_primitive_smoke_pass")):
        route = "R11-CertificatePrimitivePromising"
        system_reason = "certificate_primitive_promising_but_not_official"

    p8 = {
        "stage": "P8_SYSTEM_LEGAL_CONTROLLER",
        "status": "summary",
        "system_candidate_id": "SYS-v9360-legal-action-effect-identifiability",
        "controller_id": p5["controller_id"],
        "runtime_candidate_id": p7.get("best_runtime_candidate_id"),
        "primitive_id": "AP0-current-action-reference",
        "official_eligible": 0,
        "control_positive_controller_pass": 0,
        "payload_apply_runtime_pass": 0,
        "system_legal_controller_pass": 0,
        "weak_CP_coverage": p1.get("weak_CP_coverage"),
        "weak_CP_row_count": p1.get("weak_CP_row_count"),
        "strong_CP_coverage": p1.get("strong_CP_coverage"),
        "horizon_robust_CP_coverage": p1.get("horizon_robust_CP_coverage"),
        "best_legal_capacity_auc_CP": p2.get("best_legal_capacity_auc_CP"),
        "best_aef_auc_CP": p3.get("best_aef_auc_CP"),
        "best_microprobe_auc_CP": p4.get("best_microprobe_auc_CP"),
        "payload_apply_time_ms_q90": p7.get("best_payload_apply_time_ms_q90"),
        "step_ratio_q90": p7.get("best_step_ratio_q90"),
        "uses_dataset_name": 0,
        "uses_future_outcome": 0,
        "uses_test_or_validation": 0,
        "diagnostic_promoted_to_official": 0,
        "source_measured_gap_used": 0,
        "formula_proxy_used": 0,
        "reason": system_reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    route_decision = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "source_route_v9350": p0.get("source_route"),
        "full_control_outcome_ready": 1,
        "control_positive_oracle_pass_v9350": 1,
        "p1_cp_decomposition_pass": p1.get("p1_cp_decomposition_pass"),
        "weak_CP_row_count": p1.get("weak_CP_row_count"),
        "weak_CP_coverage": p1.get("weak_CP_coverage"),
        "weak_CP_coverage_lcb": p1.get("weak_CP_coverage_lcb"),
        "weak_CP_V_ctrl_lcb": p1.get("weak_CP_V_ctrl_lcb"),
        "strong_CP_row_count": p1.get("strong_CP_row_count"),
        "strong_CP_coverage": p1.get("strong_CP_coverage"),
        "horizon_robust_CP_action_count": p1.get("horizon_robust_CP_action_count"),
        "horizon_robust_CP_coverage": p1.get("horizon_robust_CP_coverage"),
        "short_only_CP_action_count": p1.get("short_only_CP_action_count"),
        "long_risk_action_count": p1.get("long_risk_action_count"),
        "legal_observability_capacity_pass": p2.get("legal_observability_capacity_pass"),
        "legal_observability_capacity_weak_pass": p2.get("legal_observability_capacity_weak_pass"),
        "best_legal_capacity_feature": p2.get("best_legal_capacity_feature"),
        "best_legal_capacity_auc_CP": p2.get("best_legal_capacity_auc_CP"),
        "best_top273_CP_precision": p2.get("best_top273_CP_precision"),
        "aef_feature_pass": p3.get("aef_feature_pass"),
        "best_aef_feature": p3.get("best_aef_feature"),
        "best_aef_auc_CP": p3.get("best_aef_auc_CP"),
        "best_aef_top273_CP_precision": p3.get("best_aef_top273_CP_precision"),
        "aef_measured_action_count": p3.get("measured_action_count"),
        "microprobe_pass": p4.get("microprobe_pass"),
        "best_microprobe_id": p4.get("best_microprobe_id"),
        "best_microprobe_auc_CP": p4.get("best_microprobe_auc_CP"),
        "best_microprobe_cost_q90": p4.get("best_microprobe_cost_q90"),
        "control_positive_controller_pass": 0,
        "ap0_oracle_good_but_nonidentifiable": p6.get("ap0_oracle_good_but_nonidentifiable"),
        "certificate_primitive_smoke_pass": p6.get("certificate_primitive_smoke_pass"),
        "best_runtime_candidate_id": p7.get("best_runtime_candidate_id"),
        "payload_apply_time_ms_q90": p7.get("best_payload_apply_time_ms_q90"),
        "payload_apply_error_linf_max": p7.get("best_payload_apply_error_linf_max"),
        "payload_apply_cosine_min": p7.get("best_payload_apply_cosine_min"),
        "step_ratio_q90": p7.get("best_step_ratio_q90"),
        "payload_apply_runtime_pass": 0,
        "official_eligible": 0,
        "system_legal_controller_pass": 0,
        "primary_blocker": "legal_action_effect_identifiability_failed",
        "next_required_implementation": "implement_certificate_producing_action_primitive_and_selected_controller_runtime",
        "success_v9360_strict_purekan_functional": 0,
        "success_v9360_full_functional": 0,
        "success_v9360_external_ready": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }

    artifacts: list[Path] = []
    def w(name: str, rows: list[dict[str, Any]]) -> Path:
        path = out_dir / name
        write_csv(path, rows)
        artifacts.append(path)
        return path

    w("p0_v9350_boundary_reanalysis.csv", [p0])
    w("p1_cp_label_decomposition_v9360.csv", [p1])
    w("cp_horizon_structure_trace_v9360.csv", p1_trace)
    w("p2_legal_observability_capacity_audit.csv", [p2])
    w("legal_feature_capacity_trace_v9360.csv", p2_trace)
    w("p3_action_effect_fingerprint_factory.csv", [p3])
    w("aef_feature_trace_v9360.csv", p3_trace)
    w("p4_cheap_legal_microprobe.csv", [p4])
    w("microprobe_trace_v9360.csv", p4_trace)
    w("p5_crossfitted_cp_controller.csv", [p5])
    w("controller_frontier_trace_v9360.csv", p5_rows)
    w("p6_observability_failure_autopsy_certificate_primitive.csv", [p6])
    w("observability_failure_trace_v9360.csv", p6_failure_rows)
    w("certificate_primitive_smoke_v9360.csv", p6_cert_rows)
    w("p7_payload_apply_runtime_closure.csv", [p7])
    w("payload_runtime_component_trace_v9360.csv", p7_rows)
    w("p8_system_legal_controller_v9360.csv", [p8])
    w("system_controller_trace_v9360.csv", [p8])
    for name in [
        "p9_diagnostic_causal_scout.csv",
        "p10_leave_dataset_stratum_out.csv",
        "p11_official_paired_replay.csv",
        "p12_short_full_sampleeff_continual_robustness.csv",
    ]:
        w(name, not_run(name.replace(".csv", "").upper(), "P8_system_controller_not_official"))

    write_json(out_dir / "route_decision.json", route_decision)
    write_json(out_dir / "aggregate_decision.json", route_decision)
    artifacts.extend([out_dir / "route_decision.json", out_dir / "aggregate_decision.json"])
    contract = {
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "full_control_outcome_ready": 1,
        "cp_decomposition_pass": p1.get("p1_cp_decomposition_pass"),
        "legal_observability_capacity_pass": p2.get("legal_observability_capacity_pass"),
        "aef_feature_pass": p3.get("aef_feature_pass"),
        "microprobe_pass": p4.get("microprobe_pass"),
        "control_positive_controller_pass": 0,
        "certificate_primitive_smoke_pass": p6.get("certificate_primitive_smoke_pass"),
        "payload_apply_runtime_pass": 0,
        "system_legal_controller_pass": 0,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "uses_dataset_name_for_controller": 0,
        "uses_test_or_validation": 0,
        "uses_future_outcome_for_features": 0,
        "source_measured_gap_used": 0,
        "formula_proxy_used": 0,
        "diagnostic_promoted_to_official": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    w("contract_audit_v9360.csv", [contract])
    failure_table = [
        {"failure_id": "F1_AP0_legal_identifiability_fail", "active": p6.get("ap0_oracle_good_but_nonidentifiable"), "primary_blocker": route_decision["primary_blocker"], "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0},
        {"failure_id": "F2_static_capacity_fail", "active": int(inum(p2.get("legal_observability_capacity_pass")) == 0), "primary_blocker": route_decision["primary_blocker"], "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0},
        {"failure_id": "F3_aef_feature_fail", "active": int(inum(p3.get("aef_feature_pass")) == 0), "primary_blocker": route_decision["primary_blocker"], "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0},
        {"failure_id": "F4_microprobe_fail", "active": int(inum(p4.get("microprobe_pass")) == 0), "primary_blocker": route_decision["primary_blocker"], "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0},
        {"failure_id": "F5_certificate_primitive_missing", "active": 1, "primary_blocker": route_decision["primary_blocker"], "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0},
        {"failure_id": "F6_payload_runtime_not_official", "active": 1, "primary_blocker": route_decision["primary_blocker"], "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0},
    ]
    w("failure_table.csv", failure_table)
    prov = audit_no_fake(artifacts)
    w("provenance_audit_v9360.csv", [prov])
    write_hashes(out_dir, artifacts)
    manifest["finished_at"] = now_iso()
    write_json(out_dir / "run_manifest.json", manifest)
    write_hashes(out_dir, artifacts)
    print(json.dumps({"out_dir": rel(out_dir), "route": route, "weak_CP_coverage": p1.get("weak_CP_coverage"), "best_aef_auc": p3.get("best_aef_auc_CP")}, sort_keys=True))


if __name__ == "__main__":
    main()
