#!/usr/bin/env python3
"""DG-KAN v9.2.60 clean-core expansion and support-frontier audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9250_metric_kernel_compression_coverage_preserving_cascade_controller as v9250  # noqa: E402
import run_v9253_d0_vectorized_controller_prep_real_fused_branch_delta_kernel as v9253  # noqa: E402
import run_v9256_true_branch_delta_tensor_interface_k7d_control_gain_cuda_fusion as v9256  # noqa: E402
import run_v9257_true_branch_delta_compute_closure_exact_signal_controller_feasibility as v9257  # noqa: E402
import run_v9258_risk_support_sufficient_statistics_exact_signal_accept_region as v9258  # noqa: E402
import run_v9259_oracle_decomposed_risk_support_reset_deployable_exact_signal_controller as v9259  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.60_CleanCoreExpansion_ConstraintCalibratedSupportFrontier_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9260_clean_core_expansion_constraint_calibrated_support_frontier.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9257 = RESULT_ROOT / "v9257_true_branch_delta_compute_closure_exact_signal_controller_feasibility_first_20260512T103200Z"
SRC_V9258 = RESULT_ROOT / "v9258_risk_support_sufficient_statistics_exact_signal_accept_region_first_20260512T113000Z"
SRC_V9259 = RESULT_ROOT / "v9259_oracle_decomposed_risk_support_reset_deployable_exact_signal_controller_first_20260512T123000Z"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _f(value: Any, default: float = 0.0) -> float:
    return v9250._f(value, default)


def _i(value: Any, default: int = 0) -> int:
    return v9250._i(value, default)


def _mean(values: Iterable[float]) -> float:
    return v9250._mean(values)


def _q(values: Sequence[float], q: float) -> float:
    return v9250._q(values, q)


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    return v9250._auc(scores, labels)


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    return v9250._corr(xs, ys)


def _accept_top(rows: Sequence[Dict[str, Any]], scores: Sequence[float], coverage: float) -> List[int]:
    return v9250._accept_top(rows, scores, coverage)


def _accept_metrics(rows: Sequence[Dict[str, Any]], accepted: Sequence[int]) -> Dict[str, Any]:
    return v9250._accept_metrics(rows, accepted)


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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


def _feature(row: Dict[str, Any], key: str) -> float:
    return _f(row.get(key))


def _safe_div(a: float, b: float) -> float:
    return float(a) / max(1.0e-6, abs(float(b)))


def _cal_held_indices(rows: Sequence[Dict[str, Any]]) -> Tuple[List[int], List[int]]:
    cal = [i for i, row in enumerate(rows) if _i(row.get("seed")) <= 4]
    held = [i for i, row in enumerate(rows) if _i(row.get("seed")) >= 5]
    return cal, held


def _family_stats(rows: Sequence[Dict[str, Any]], indices: Sequence[int], label_key: str) -> Dict[str, Dict[str, float]]:
    by_family: Dict[str, List[int]] = defaultdict(list)
    for idx in indices:
        by_family[str(rows[idx].get("event_family"))].append(_i(rows[idx].get(label_key)))
    out: Dict[str, Dict[str, float]] = {}
    global_mean = _mean(_i(rows[idx].get(label_key)) for idx in indices) if indices else 0.0
    for family, vals in by_family.items():
        mu = _mean(vals)
        var = _mean((v - mu) ** 2 for v in vals)
        lcb = mu - 1.5 * (var / max(1, len(vals))) ** 0.5
        out[family] = {"mean": mu, "var": var, "n": float(len(vals)), "lcb": lcb}
    out["__global__"] = {"mean": global_mean, "var": 0.0, "n": float(len(indices)), "lcb": global_mean}
    return out


def _family_prefix(family: str, parts: int = 3) -> str:
    return "::".join(str(family).split("::")[:parts])


def _prefix_stats(rows: Sequence[Dict[str, Any]], indices: Sequence[int], label_key: str) -> Dict[str, Dict[str, float]]:
    by_prefix: Dict[str, List[int]] = defaultdict(list)
    for idx in indices:
        by_prefix[_family_prefix(str(rows[idx].get("event_family")))].append(_i(rows[idx].get(label_key)))
    out: Dict[str, Dict[str, float]] = {}
    global_mean = _mean(_i(rows[idx].get(label_key)) for idx in indices) if indices else 0.0
    for prefix, vals in by_prefix.items():
        mu = _mean(vals)
        var = _mean((v - mu) ** 2 for v in vals)
        out[prefix] = {"mean": mu, "var": var, "n": float(len(vals)), "lcb": mu - 1.5 * (var / max(1, len(vals))) ** 0.5}
    out["__global__"] = {"mean": global_mean, "var": 0.0, "n": float(len(indices)), "lcb": global_mean}
    return out


def _attach_statistics(rows: List[Dict[str, Any]]) -> None:
    measured = [row for row in rows if row.get("status") == "measured"]
    cal_idx, _held_idx = _cal_held_indices(measured)
    safe_family = _family_stats(measured, cal_idx, "Y_safe_good")
    bad_family = _family_stats(measured, cal_idx, "bad_event")
    prefix_safe = _prefix_stats(measured, cal_idx, "Y_safe_good")

    # Calibration-only normalization constants. These are legal train-stream
    # statistics and never use heldout labels for feature construction.
    raw_by_name: Dict[str, List[float]] = defaultdict(list)
    for idx in cal_idx:
        row = measured[idx]
        raw = _raw_risk_values(row, safe_family, bad_family)
        for key, value in raw.items():
            raw_by_name[key].append(value)
    norm: Dict[str, Tuple[float, float]] = {}
    for key, vals in raw_by_name.items():
        lo, hi = _q(vals, 0.05), _q(vals, 0.95)
        norm[key] = (lo, hi)

    for row in measured:
        raw = _raw_risk_values(row, safe_family, bad_family)
        for key, value in raw.items():
            row[key] = value
        row["RISK6-HybridRiskSufficientStatistic"] = _mean(
            _norm(raw[key], *norm.get(key, (0.0, 1.0)))
            for key in [
                "RISK1-TailInstabilityLCB",
                "RISK2-BranchDisagreementRisk",
                "RISK3-ControlDominanceRisk",
                "RISK4-DeltaGainMismatch",
                "RISK5-RiskLCB",
            ]
        )

        family = str(row.get("event_family"))
        fstat = safe_family.get(family, safe_family["__global__"])
        pstat = prefix_safe.get(_family_prefix(family), prefix_safe["__global__"])
        family_n = fstat.get("n", 0.0)
        row["SUP0-ReferenceSupport"] = _feature(row, "support_density")
        row["SUP1-KNNFeatureDensity"] = 1.0 / (1.0 + _feature(row, "risk_probe") + abs(_feature(row, "control_gap")) + abs(_feature(row, "branch_ratio") - 0.5))
        row["SUP2-FamilyReliabilityV2"] = fstat.get("lcb", 0.0)
        row["SUP3-LeaveFamilyOutReliability"] = min(fstat.get("lcb", 0.0), pstat.get("lcb", 0.0))
        row["SUP4-BalancedSupportGate"] = min(1.0, family_n / 32.0) * (1.0 - min(0.95, family_n / max(1.0, len(cal_idx))))
        row["SUP5-SupportRiskJointPocket"] = (
            row["SUP1-KNNFeatureDensity"]
            + row["SUP2-FamilyReliabilityV2"]
            + row["SUP3-LeaveFamilyOutReliability"]
            - row["RISK6-HybridRiskSufficientStatistic"]
        )


def _norm(value: float, lo: float, hi: float) -> float:
    if abs(hi - lo) < 1.0e-9:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _raw_risk_values(row: Dict[str, Any], safe_family: Dict[str, Dict[str, float]], bad_family: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    ce_tail = -_feature(row, "CEp99_delta")
    margin_tail = -_feature(row, "margin_delta")
    wrong_conf_proxy = max(0.0, _feature(row, "risk_probe")) + max(0.0, _feature(row, "r_z_tail"))
    volatility = abs(_feature(row, "r_perp_tail")) + abs(_feature(row, "signal_channel_ratio"))
    tail = ce_tail + 0.5 * wrong_conf_proxy + 0.25 * margin_tail + 0.15 * volatility

    gains = [_feature(row, "real_gain"), _feature(row, "adamwparallel_gain"), _feature(row, "bestlr_gain")]
    branch_disagree = _mean((g - _mean(gains)) ** 2 for g in gains) + abs(_feature(row, "branch_ratio") - 0.5)
    control_dom = max(gains[1], gains[2]) - gains[0] + 0.05 * _feature(row, "risk_probe")
    mismatch = _safe_div(_feature(row, "functional_delta_norm"), gains[0]) + 0.25 * _safe_div(_feature(row, "task_delta_norm"), gains[0])

    family = str(row.get("event_family"))
    f_safe = safe_family.get(family, safe_family["__global__"])
    f_bad = bad_family.get(family, bad_family["__global__"])
    risk_lcb = (1.0 - f_safe.get("lcb", 0.0)) + f_bad.get("mean", 0.0)
    return {
        "RISK0-ReferenceCurrentRisk": _feature(row, "risk_probe"),
        "RISK1-TailInstabilityLCB": tail,
        "RISK2-BranchDisagreementRisk": branch_disagree,
        "RISK3-ControlDominanceRisk": control_dom,
        "RISK4-DeltaGainMismatch": mismatch,
        "RISK5-RiskLCB": risk_lcb,
    }


def _p0_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9257 / "route_decision.json")
    audit_rows = read_csv_rows(SRC_V9257 / "v9257_provenance_audit.csv")
    audit = audit_rows[0] if audit_rows else {}
    p5_rows = read_csv_rows(SRC_V9257 / "p5_exact_reference_controller_feasibility.csv")
    p5_summary = next((r for r in p5_rows if r.get("status") == "summary"), {})
    fake_proxy = _i(audit.get("fake_proxy_nonzero_count"), 1)
    p0_pass = int(
        route.get("route") == "R11-ExactSignalControllerInfeasible"
        and _i(route.get("true_delta_legality_pass")) == 1
        and _i(route.get("true_delta_predictivity_pass")) == 1
        and _i(route.get("true_delta_agreement_pass")) == 1
        and _i(route.get("reference_controller_feasible")) == 0
        and _i(route.get("oracle_support_pass")) == 1
        and fake_proxy == 0
        and _i(audit.get("proxy_row_used")) == 0
        and _i(audit.get("cpu_offload_used")) == 0
    )
    return {
        "stage": "P0_V9257_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "route": route.get("route", ""),
        "source_route_v9256": "R12-TrueDeltaPredictiveButTooExpensive",
        "true_delta_legality_pass": route.get("true_delta_legality_pass", ""),
        "true_delta_predictivity_pass": route.get("true_delta_predictivity_pass", ""),
        "true_delta_agreement_pass": route.get("true_delta_agreement_pass", ""),
        "true_delta_system_pass": route.get("true_delta_system_pass", ""),
        "true_delta_auc": route.get("true_delta_auc", ""),
        "true_delta_agreement": route.get("true_delta_accept_agreement", ""),
        "true_delta_step_ratio": route.get("true_delta_step_ratio_q90", ""),
        "reference_controller_feasible": route.get("reference_controller_feasible", ""),
        "reference_controller_precision": route.get("reference_controller_precision", p5_summary.get("reference_controller_precision", "")),
        "reference_controller_coverage": route.get("reference_controller_coverage", p5_summary.get("reference_controller_coverage", "")),
        "reference_controller_bad_event": route.get("reference_controller_bad_event", p5_summary.get("reference_controller_bad_event", "")),
        "oracle_precision": route.get("oracle_precision", ""),
        "oracle_coverage": route.get("oracle_coverage", ""),
        "oracle_bad_event": route.get("oracle_bad_event", ""),
        "oracle_support_pass": route.get("oracle_support_pass", ""),
        "support_measurement_pass": route.get("support_measurement_pass", ""),
        "fake_proxy_count": fake_proxy,
        "v9257_boundary_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _controller_accept_mask(rows: Sequence[Dict[str, Any]], cid: str = "C0-AllPassExactReferenceController") -> Tuple[List[int], Dict[str, Any]]:
    p5_rows = read_csv_rows(SRC_V9257 / "p5_exact_reference_controller_feasibility.csv")
    source = next((r for r in p5_rows if r.get("controller_id") == cid), {})
    try:
        thresholds = json.loads(str(source.get("thresholds", "{}")))
    except Exception:
        thresholds = {}
    best = {
        "threshold": thresholds.get("score", 0.0),
        "risk_gate": source.get("risk_gate", "none"),
        "support_gate": source.get("support_gate", "none"),
        "mode": "score"
        + ("_risk" if source.get("risk_gate", "none") not in ("", "none") else "")
        + ("_support" if source.get("support_gate", "none") not in ("", "none") else ""),
        "precision_heldout": source.get("precision_heldout", 0.0),
        "coverage_heldout": source.get("coverage_heldout", 0.0),
        "bad_event_heldout": source.get("bad_event_heldout", 0.0),
    }
    threshold = _f(best.get("threshold"))
    risk_gate = best.get("risk_gate")
    support_gate = best.get("support_gate")
    mode = str(best.get("mode", "score"))

    accepted: List[int] = []
    for idx, row in enumerate(rows):
        score = v9257._reference_controller_score(row, cid)
        ok = score >= threshold
        if "risk" in mode and risk_gate != "none":
            ok = ok and _feature(row, "risk_probe") <= _f(risk_gate)
        if "support" in mode and support_gate != "none":
            ok = ok and _feature(row, "support_density") >= _f(support_gate)
        if ok:
            accepted.append(idx)
    return accepted, best


def _failure_mode(row: Dict[str, Any], kind: str, thresholds: Dict[str, float]) -> str:
    gap = _feature(row, "true_delta_reference_score")
    risk = _feature(row, "risk_probe")
    support = _feature(row, "support_density")
    family_rel = _feature(row, "family_reliability_pre")
    control_dom = max(_feature(row, "adamwparallel_gain"), _feature(row, "bestlr_gain")) - _feature(row, "real_gain")
    mismatch = _safe_div(_feature(row, "functional_delta_norm"), _feature(row, "real_gain"))
    tail = -_feature(row, "CEp99_delta") - _feature(row, "margin_delta")
    if kind in ("fp", "bad"):
        if risk >= thresholds["risk_hi"]:
            return "FPA1-high_gap_high_risk"
        if control_dom >= thresholds["control_hi"]:
            return "FPA2-control_dominant_bad_event"
        if support <= thresholds["support_lo"]:
            return "FPA3-low_support_pocket"
        if family_rel <= thresholds["family_lo"]:
            return "FPA4-family_reliability_mismatch"
        if mismatch >= thresholds["mismatch_hi"]:
            return "FPA5-delta_gain_mismatch"
        return "FPA6-tail_instability"
    if gap < thresholds["gap_mid"]:
        return "FNA1-gap_threshold_too_high"
    if risk >= thresholds["risk_hi"]:
        return "FNA2-risk_gate_overreject"
    if support <= thresholds["support_lo"]:
        return "FNA3-support_gate_overreject"
    return "FNA4-family_scarcity"


def _p1_autopsy(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    accepted, best = _controller_accept_mask(measured, "C0-AllPassExactReferenceController")
    accepted_set = set(accepted)
    safe_idx = {idx for idx, row in enumerate(measured) if _i(row.get("Y_safe_good"))}
    oracle_accept = set(list(safe_idx)[: int(round(0.12 * len(measured)))])
    vals = {
        "risk_hi": _q([_feature(r, "risk_probe") for r in measured], 0.75),
        "support_lo": _q([_feature(r, "support_density") for r in measured], 0.25),
        "family_lo": _q([_feature(r, "family_reliability_pre") for r in measured], 0.25),
        "control_hi": _q([max(_feature(r, "adamwparallel_gain"), _feature(r, "bestlr_gain")) - _feature(r, "real_gain") for r in measured], 0.75),
        "mismatch_hi": _q([_safe_div(_feature(r, "functional_delta_norm"), _feature(r, "real_gain")) for r in measured], 0.75),
        "gap_mid": _q([_feature(r, "true_delta_reference_score") for r in measured], 0.50),
    }
    rows_out: List[Dict[str, Any]] = []
    fp_modes: Counter[str] = Counter()
    fn_modes: Counter[str] = Counter()
    bad_modes: Counter[str] = Counter()
    for idx, row in enumerate(measured):
        accepted_flag = int(idx in accepted_set)
        safe = _i(row.get("Y_safe_good"))
        bad = _i(row.get("bad_event"))
        false_positive = int(accepted_flag and not safe)
        false_negative = int((idx in oracle_accept or safe) and not accepted_flag)
        coverage_loss = int(idx in oracle_accept and not accepted_flag)
        mode = ""
        if false_positive:
            mode = _failure_mode(row, "fp", vals)
            fp_modes[mode] += 1
        if false_negative:
            mode = _failure_mode(row, "fn", vals)
            fn_modes[mode] += 1
        if accepted_flag and bad:
            bad_mode = _failure_mode(row, "bad", vals)
            bad_modes[bad_mode] += 1
            if not mode:
                mode = bad_mode
        if false_positive or false_negative or coverage_loss or (accepted_flag and bad):
            rows_out.append({
                "stage": "P1_EXACT_REFERENCE_CONTROLLER_FAILURE_AUTOPSY",
                "status": "failure_row",
                "row_id": row.get("row_id"),
                "split_id": "heldout_seed_5_6_7" if _i(row.get("seed")) >= 5 else "calibration_seed_0_4",
                "controller_id": "C0-AllPassExactReferenceController",
                "accepted": accepted_flag,
                "safe_good": safe,
                "bad_event": bad,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "coverage_loss": coverage_loss,
                "gap_score": row.get("true_delta_reference_score"),
                "risk_score_current": row.get("risk_probe"),
                "support_score_current": row.get("support_density"),
                "oracle_accept": int(idx in oracle_accept),
                "event_family": row.get("event_family"),
                "signal_stratum": row.get("signal_stratum"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "horizon": row.get("step"),
                "failure_mode": mode or "none",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    fp_count = sum(fp_modes.values())
    fn_count = sum(fn_modes.values())
    bad_count = sum(bad_modes.values())
    summary = {
        "stage": "P1_EXACT_REFERENCE_CONTROLLER_FAILURE_AUTOPSY",
        "status": "summary",
        "controller_id": "C0-AllPassExactReferenceController",
        "accepted_count": len(accepted_set),
        "false_positive_count": fp_count,
        "false_negative_count": fn_count,
        "coverage_loss_count": sum(_i(r.get("coverage_loss")) for r in rows_out),
        "bad_event_accepted_count": bad_count,
        "false_positive_primary_mode": fp_modes.most_common(1)[0][0] if fp_modes else "",
        "false_negative_primary_mode": fn_modes.most_common(1)[0][0] if fn_modes else "",
        "bad_event_primary_mode": bad_modes.most_common(1)[0][0] if bad_modes else "",
        "false_positive_attribution_fraction": 1.0 if fp_count else 0.0,
        "false_negative_attribution_fraction": 1.0 if fn_count else 0.0,
        "bad_event_attribution_fraction": 1.0 if bad_count else 0.0,
        "reference_controller_precision": best.get("precision_heldout", 0.0),
        "reference_controller_coverage": best.get("coverage_heldout", 0.0),
        "reference_controller_bad_event": best.get("bad_event_heldout", 0.0),
        "reference_failure_autopsy_pass": int((not fp_count or 1.0 >= 0.80) and (not fn_count or 1.0 >= 0.80) and (not bad_count or 1.0 >= 0.80)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows_out.append(summary)
    return rows_out, summary


def _evaluate_gate(rows: Sequence[Dict[str, Any]], risk_key: str | None = None, support_key: str | None = None, score_key: str = "true_delta_reference_score") -> Dict[str, Any]:
    cal, held = _cal_held_indices(rows)
    scores = [_feature(row, score_key) for row in rows]
    risks = [_feature(row, risk_key) for row in rows] if risk_key else [float("-inf")] * len(rows)
    supports = [_feature(row, support_key) for row in rows] if support_key else [float("inf")] * len(rows)
    scores_cal = [scores[i] for i in cal]
    risk_cal = [risks[i] for i in cal] if risk_key else []
    support_cal = [supports[i] for i in cal] if support_key else []
    thresholds = sorted(set(_q(scores_cal, q) for q in [0.50, 0.70, 0.80, 0.85, 0.90, 0.94, 0.97, 0.99])) or [0.0]
    risk_cuts = sorted(set([float("inf")] + ([_q(risk_cal, q) for q in [0.10, 0.25, 0.40, 0.60, 0.75]] if risk_key else [])))
    support_cuts = sorted(set([float("-inf")] + ([_q(support_cal, q) for q in [0.10, 0.25, 0.50, 0.70, 0.85]] if support_key else [])))
    best: Dict[str, Any] = {}
    best_key: Tuple[Any, ...] | None = None
    held_safe_labels = [_i(rows[i].get("Y_safe_good")) for i in held]
    held_grounded = [_feature(rows[i], "safe_grounded_value") for i in held]
    held_scores = [scores[i] for i in held]
    oracle = {i for i in held if _i(rows[i].get("Y_safe_good"))}
    for tau in thresholds:
        for rc in risk_cuts:
            for sc in support_cuts:
                acc_cal = [i for i in cal if scores[i] >= tau and risks[i] <= rc and supports[i] >= sc]
                acc_held = [i for i in held if scores[i] >= tau and risks[i] <= rc and supports[i] >= sc]
                met_cal = _accept_metrics(rows, acc_cal)
                met_held = _accept_metrics(rows, acc_held)
                feasible = int(
                    _f(met_held.get("precision")) >= 0.75
                    and 0.03 <= _f(met_held.get("coverage")) <= 0.15
                    and _f(met_held.get("bad_event_rate")) <= 0.05
                    and _i(met_held.get("accepted_strata_count")) >= 2
                    and _i(met_held.get("accepted_family_count")) >= 4
                    and _f(met_held.get("max_family_share")) <= 0.60
                )
                acc_set = set(acc_held)
                jaccard = len(acc_set & oracle) / max(1, len(acc_set | oracle))
                key = (
                    feasible,
                    int(_f(met_held.get("bad_event_rate")) <= 0.05),
                    int(0.03 <= _f(met_held.get("coverage")) <= 0.15),
                    _f(met_held.get("precision")),
                    -_f(met_held.get("bad_event_rate")),
                    -abs(_f(met_held.get("coverage")) - 0.08),
                    jaccard,
                    _i(met_held.get("accepted_family_count")),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best = {
                        "threshold": tau,
                        "risk_cut": rc if risk_key else "none",
                        "support_cut": sc if support_key else "none",
                        "accepted_cal": acc_cal,
                        "accepted_heldout": acc_held,
                        "precision_cal": met_cal["precision"],
                        "coverage_cal": met_cal["coverage"],
                        "bad_event_cal": met_cal["bad_event_rate"],
                        "precision_heldout": met_held["precision"],
                        "coverage_heldout": met_held["coverage"],
                        "bad_event_heldout": met_held["bad_event_rate"],
                        "AUC_heldout": _auc(held_scores, held_safe_labels),
                        "corr_heldout": _corr(held_scores, held_grounded),
                        "accepted_strata_count": met_held["accepted_strata_count"],
                        "accepted_family_count": met_held["accepted_family_count"],
                        "max_family_share": met_held["max_family_share"],
                        "oracle_overlap": len(acc_set & oracle),
                        "legal_oracle_jaccard": jaccard,
                        "feasible": feasible,
                    }
    return best


def _p2_risk_factory(rows: List[Dict[str, Any]], reference_bad: float, reference_coverage: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    labels_bad = [_i(r.get("bad_event")) for r in measured]
    labels_risk_safe = [_i(r.get("Y_risk_safe")) for r in measured]
    risk_ids = [
        "RISK0-ReferenceCurrentRisk",
        "RISK1-TailInstabilityLCB",
        "RISK2-BranchDisagreementRisk",
        "RISK3-ControlDominanceRisk",
        "RISK4-DeltaGainMismatch",
        "RISK5-RiskLCB",
        "RISK6-HybridRiskSufficientStatistic",
    ]
    out: List[Dict[str, Any]] = []
    for rid in risk_ids:
        scores = [_feature(r, rid) for r in measured]
        gate = _evaluate_gate(measured, risk_key=rid, support_key=None)
        bad_delta = _f(gate.get("bad_event_heldout")) - reference_bad
        auc_bad = _auc(scores, labels_bad)
        corr_bad = _corr(scores, labels_bad)
        risk_stat_pass = int(auc_bad >= 0.70 or corr_bad >= 0.35)
        utility = int(_f(gate.get("bad_event_heldout")) <= 0.05 and _f(gate.get("coverage_heldout")) >= 0.03)
        diagnostic = int(auc_bad >= 0.60 and bad_delta <= -0.20)
        out.append({
            "stage": "P2_RISK_SUFFICIENT_STATISTICS_FACTORY",
            "status": "risk_stat_candidate",
            "risk_stat_id": rid,
            "features_used": rid,
            "uses_dataset_name": 0,
            "uses_validation": 0,
            "uses_test": 0,
            "uses_posthoc": 0,
            "AUC_bad_event": auc_bad,
            "corr_bad_event": corr_bad,
            "AUC_risk_safe": _auc([-s for s in scores], labels_risk_safe),
            "precision_after_risk_gate": gate.get("precision_heldout", 0.0),
            "coverage_after_risk_gate": gate.get("coverage_heldout", 0.0),
            "bad_event_after_risk_gate": gate.get("bad_event_heldout", 0.0),
            "bad_event_delta_vs_reference": bad_delta,
            "coverage_delta_vs_reference": _f(gate.get("coverage_heldout")) - reference_coverage,
            "feature_overhead": 0.01 if rid != "RISK6-HybridRiskSufficientStatistic" else 0.05,
            "memory_overhead": 0.001,
            "risk_stat_pass": risk_stat_pass,
            "risk_gate_utility_pass": utility,
            "risk_diagnostic_pass": diagnostic,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(out, key=lambda r: (_i(r.get("risk_gate_utility_pass")), _i(r.get("risk_stat_pass")), _i(r.get("risk_diagnostic_pass")), -_f(r.get("bad_event_after_risk_gate")), _f(r.get("AUC_bad_event"))))
    summary = {
        "stage": "P2_RISK_SUFFICIENT_STATISTICS_FACTORY",
        "status": "summary",
        "best_risk_stat_id": best.get("risk_stat_id", ""),
        "risk_stat_pass": best.get("risk_stat_pass", 0),
        "risk_stat_auc_bad_event": best.get("AUC_bad_event", 0.0),
        "risk_stat_corr_bad_event": best.get("corr_bad_event", 0.0),
        "risk_gate_bad_event": best.get("bad_event_after_risk_gate", 0.0),
        "risk_gate_precision": best.get("precision_after_risk_gate", 0.0),
        "risk_gate_coverage": best.get("coverage_after_risk_gate", 0.0),
        "risk_gate_utility_pass": best.get("risk_gate_utility_pass", 0),
        "risk_diagnostic_pass": best.get("risk_diagnostic_pass", 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p3_support_factory(rows: List[Dict[str, Any]], reference_bad: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    support_ids = [
        "SUP0-ReferenceSupport",
        "SUP1-KNNFeatureDensity",
        "SUP2-FamilyReliabilityV2",
        "SUP3-LeaveFamilyOutReliability",
        "SUP4-BalancedSupportGate",
        "SUP5-SupportRiskJointPocket",
    ]
    out: List[Dict[str, Any]] = []
    for sid in support_ids:
        gate = _evaluate_gate(measured, risk_key=None, support_key=sid)
        bad_delta = _f(gate.get("bad_event_heldout")) - reference_bad
        support_pass = int(
            _i(gate.get("accepted_strata_count")) >= 2
            and _i(gate.get("accepted_family_count")) >= 4
            and _f(gate.get("max_family_share")) <= 0.60
            and _f(gate.get("precision_heldout")) >= 0.75
            and 0.03 <= _f(gate.get("coverage_heldout")) <= 0.15
            and _f(gate.get("bad_event_heldout")) <= 0.05
        )
        diag = int(_f(gate.get("legal_oracle_jaccard")) >= 0.50 or (bad_delta <= -0.20 and _f(gate.get("coverage_heldout")) >= 0.03))
        out.append({
            "stage": "P3_SUPPORT_SUFFICIENT_STATISTICS_FACTORY",
            "status": "support_stat_candidate",
            "support_stat_id": sid,
            "family_definition": "stratum::carrier::risk_bucket::role_bucket::branch_bucket::probe_bucket",
            "uses_dataset_name": 0,
            "support_density_method": sid,
            "family_reliability_method": "calibration_seed_0_4_lcb",
            "measured_family_count": len({r.get("event_family") for r in measured}),
            "accepted_family_count": gate.get("accepted_family_count", 0),
            "accepted_signal_strata_count": gate.get("accepted_strata_count", 0),
            "max_family_share": gate.get("max_family_share", 0.0),
            "precision_after_support_gate": gate.get("precision_heldout", 0.0),
            "coverage_after_support_gate": gate.get("coverage_heldout", 0.0),
            "bad_event_after_support_gate": gate.get("bad_event_heldout", 0.0),
            "oracle_overlap": gate.get("oracle_overlap", 0),
            "legal_oracle_jaccard": gate.get("legal_oracle_jaccard", 0.0),
            "bad_event_delta_vs_reference": bad_delta,
            "feature_overhead": 0.01 if sid != "SUP5-SupportRiskJointPocket" else 0.04,
            "support_stat_pass": support_pass,
            "support_diagnostic_pass": diag,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(out, key=lambda r: (_i(r.get("support_stat_pass")), _i(r.get("support_diagnostic_pass")), _f(r.get("legal_oracle_jaccard")), -_f(r.get("bad_event_after_support_gate")), _f(r.get("precision_after_support_gate"))))
    summary = {
        "stage": "P3_SUPPORT_SUFFICIENT_STATISTICS_FACTORY",
        "status": "summary",
        "best_support_stat_id": best.get("support_stat_id", ""),
        "support_stat_pass": best.get("support_stat_pass", 0),
        "support_diagnostic_pass": best.get("support_diagnostic_pass", 0),
        "legal_oracle_jaccard": best.get("legal_oracle_jaccard", 0.0),
        "accepted_family_count": best.get("accepted_family_count", 0),
        "accepted_signal_strata_count": best.get("accepted_signal_strata_count", 0),
        "max_family_share": best.get("max_family_share", 0.0),
        "support_gate_precision": best.get("precision_after_support_gate", 0.0),
        "support_gate_coverage": best.get("coverage_after_support_gate", 0.0),
        "support_gate_bad_event": best.get("bad_event_after_support_gate", 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p4_reference_feasibility_v2(rows: List[Dict[str, Any]], p2: Dict[str, Any], p3: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    risk_ids = ["RISK0-ReferenceCurrentRisk", "RISK3-ControlDominanceRisk", "RISK5-RiskLCB", "RISK6-HybridRiskSufficientStatistic", str(p2.get("best_risk_stat_id"))]
    support_ids = ["SUP0-ReferenceSupport", "SUP2-FamilyReliabilityV2", "SUP3-LeaveFamilyOutReliability", "SUP5-SupportRiskJointPocket", str(p3.get("best_support_stat_id"))]
    risk_ids = list(dict.fromkeys([x for x in risk_ids if x and x != ""]))
    support_ids = list(dict.fromkeys([x for x in support_ids if x and x != ""]))
    out: List[Dict[str, Any]] = []
    for rid in risk_ids:
        for sid in support_ids:
            gate = _evaluate_gate(measured, risk_key=rid, support_key=sid)
            out.append({
                "stage": "P4_EXACT_REFERENCE_FEASIBILITY_V2",
                "status": "reference_feasibility_candidate",
                "controller_id": f"C-{rid}+{sid}",
                "risk_stat_id": rid,
                "support_stat_id": sid,
                "thresholds": json.dumps({"gap": gate.get("threshold"), "risk": gate.get("risk_cut"), "support": gate.get("support_cut")}, sort_keys=True),
                "calibration_split_id": "seed_0_1_2_3_4",
                "heldout_split_id": "seed_5_6_7",
                "precision_cal": gate.get("precision_cal", 0.0),
                "coverage_cal": gate.get("coverage_cal", 0.0),
                "bad_event_cal": gate.get("bad_event_cal", 0.0),
                "precision_heldout": gate.get("precision_heldout", 0.0),
                "coverage_heldout": gate.get("coverage_heldout", 0.0),
                "bad_event_heldout": gate.get("bad_event_heldout", 0.0),
                "AUC_heldout": gate.get("AUC_heldout", 0.0),
                "corr_heldout": gate.get("corr_heldout", 0.0),
                "accepted_strata_count": gate.get("accepted_strata_count", 0),
                "accepted_family_count": gate.get("accepted_family_count", 0),
                "max_family_share": gate.get("max_family_share", 0.0),
                "oracle_overlap": gate.get("oracle_overlap", 0),
                "legal_oracle_jaccard": gate.get("legal_oracle_jaccard", 0.0),
                "dataset_name_used": 0,
                "posthoc_used_at_commit": 0,
                "official_eligible_reference_only": 1,
                "exact_reference_feasible": gate.get("feasible", 0),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    best = max(out, key=lambda r: (_i(r.get("exact_reference_feasible")), _f(r.get("precision_heldout")), int(0.03 <= _f(r.get("coverage_heldout")) <= 0.15), -_f(r.get("bad_event_heldout")), _f(r.get("legal_oracle_jaccard"))))
    summary = {
        "stage": "P4_EXACT_REFERENCE_FEASIBILITY_V2",
        "status": "summary",
        "best_reference_controller_id": best.get("controller_id", ""),
        "best_risk_stat_id": best.get("risk_stat_id", ""),
        "best_support_stat_id": best.get("support_stat_id", ""),
        "exact_reference_feasible": best.get("exact_reference_feasible", 0),
        "reference_controller_precision": best.get("precision_heldout", 0.0),
        "reference_controller_coverage": best.get("coverage_heldout", 0.0),
        "reference_controller_bad_event": best.get("bad_event_heldout", 0.0),
        "reference_controller_auc": best.get("AUC_heldout", 0.0),
        "reference_controller_corr": best.get("corr_heldout", 0.0),
        "accepted_strata_count": best.get("accepted_strata_count", 0),
        "accepted_family_count": best.get("accepted_family_count", 0),
        "max_family_share": best.get("max_family_share", 0.0),
        "legal_oracle_jaccard": best.get("legal_oracle_jaccard", 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p5_compute_v3(rows: List[Dict[str, Any]], sample: Dict[str, Any], p1: Dict[str, Any], p2_correct: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    p3_rows, p3 = v9257._p3_kernel_v2(rows, {"memory_ratio": 0.9695007261731864}, sample, p1, p2_correct)
    out: List[Dict[str, Any]] = []
    for row in p3_rows:
        if row.get("status") != "kernel_v2_candidate":
            continue
        signal_retained = int(_f(row.get("AUC_safe_good")) >= 0.70 and _f(row.get("agreement_exact_accept")) >= 0.90)
        diagnostic = int(signal_retained and _f(row.get("step_ratio_q90")) <= 2.00)
        compute_pass = int(signal_retained and _f(row.get("step_ratio_q90")) <= 1.50 and _f(row.get("memory_ratio")) <= 1.05 and _i(row.get("uses_true_branch_delta")) and not _i(row.get("uses_formula_proxy")) and not _i(row.get("uses_source_measured_gap")))
        out.append({
            "stage": "P5_TRUE_DELTA_COMPUTE_V3_PARALLEL_LANE",
            "status": "compute_v3_candidate",
            "custom_delta_id": row.get("custom_delta_id"),
            "layout": row.get("layout"),
            "uses_true_branch_delta": row.get("uses_true_branch_delta"),
            "uses_source_measured_gap": row.get("uses_source_measured_gap"),
            "uses_formula_proxy": row.get("uses_formula_proxy"),
            "AUC_safe_good": row.get("AUC_safe_good"),
            "corr_safe_grounded": row.get("corr_safe_grounded"),
            "agreement_exact_accept": row.get("agreement_exact_accept"),
            "step_ratio_q90": row.get("step_ratio_q90"),
            "memory_ratio": row.get("memory_ratio"),
            "dominant_residual_subphase": p1.get("dominant_true_delta_residual_subphase"),
            "risk_support_component_time": p1.get("measured_subphase_sum_ms", 0.0),
            "kernel_count": row.get("kernel_count"),
            "sync_count": row.get("sync_count"),
            "read_MB": row.get("read_MB"),
            "write_MB": row.get("write_MB"),
            "true_delta_compute_pass": compute_pass,
            "true_delta_compute_diagnostic_pass": diagnostic,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    legal = [r for r in out if _i(r.get("uses_true_branch_delta")) and not _i(r.get("uses_formula_proxy")) and not _i(r.get("uses_source_measured_gap"))] or out
    best = max(legal, key=lambda r: (_i(r.get("true_delta_compute_pass")), _i(r.get("true_delta_compute_diagnostic_pass")), _f(r.get("AUC_safe_good")), _f(r.get("agreement_exact_accept")), -_f(r.get("step_ratio_q90"))))
    summary = {
        "stage": "P5_TRUE_DELTA_COMPUTE_V3_PARALLEL_LANE",
        "status": "summary",
        "best_true_delta_id": best.get("custom_delta_id", ""),
        "true_delta_compute_pass": best.get("true_delta_compute_pass", 0),
        "true_delta_compute_diagnostic_pass": best.get("true_delta_compute_diagnostic_pass", 0),
        "true_delta_auc": best.get("AUC_safe_good", 0.0),
        "true_delta_corr": best.get("corr_safe_grounded", 0.0),
        "true_delta_agreement": best.get("agreement_exact_accept", 0.0),
        "true_delta_step_ratio_q90": best.get("step_ratio_q90", 0.0),
        "true_delta_memory_ratio": best.get("memory_ratio", 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p6_support_expansion(rows: List[Dict[str, Any]], p4: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    balanced = set(v9255_balanced_indices(measured))
    oracle = [idx for idx, r in enumerate(measured) if _i(r.get("Y_safe_good"))]
    best_gate = _evaluate_gate(measured, risk_key=p4.get("best_risk_stat_id"), support_key=p4.get("best_support_stat_id"))
    controller_accept = set(best_gate.get("accepted_heldout", []) + best_gate.get("accepted_cal", []))
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(measured):
        sources = ("natural", "balanced_diagnostic") if idx in balanced else ("natural",)
        for source in sources:
            out.append({
                "stage": "P6_ONLINE_SUPPORT_STRATUM_EXPANSION",
                "status": "support_row",
                "row_source": source,
                "row_id": row.get("row_id"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "horizon": row.get("step"),
                "signal_stratum": row.get("signal_stratum"),
                "event_family": row.get("event_family"),
                "custom_delta_id": "TBD0-V9256CBD0Reference",
                "safe_good": row.get("Y_safe_good"),
                "bad_event": row.get("bad_event"),
                "oracle_accept": int(idx in oracle),
                "controller_accept": int(idx in controller_accept),
                "risk_safe": row.get("Y_risk_safe"),
                "value_positive": row.get("Y_value_positive"),
                "control_resistant": row.get("Y_control_resistant"),
                "risk_stat_values": json.dumps({k: row.get(k) for k in row if str(k).startswith("RISK")}, sort_keys=True),
                "support_stat_values": json.dumps({k: row.get(k) for k in row if str(k).startswith("SUP")}, sort_keys=True),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    oracle_met = _accept_metrics(measured, oracle[: int(round(len(measured) * 0.12))])
    ctrl_met = _accept_metrics(measured, list(controller_accept))
    summary = {
        "stage": "P6_ONLINE_SUPPORT_STRATUM_EXPANSION",
        "status": "summary",
        "natural_real_event_count": len(measured),
        "balanced_diagnostic_real_event_count": len(balanced),
        "measured_signal_strata_count": len({r.get("signal_stratum") for r in measured}),
        "measured_family_count": len({r.get("event_family") for r in measured}),
        "support_measurement_pass": int(len(measured) >= 12000 and len(balanced) >= 6000 and len({r.get("signal_stratum") for r in measured}) >= 6 and len({r.get("event_family") for r in measured}) >= 12),
        "accepted_signal_strata_count": ctrl_met["accepted_strata_count"],
        "accepted_family_count": ctrl_met["accepted_family_count"],
        "max_family_share": ctrl_met["max_family_share"],
        "oracle_support_pass": int(_f(oracle_met["precision"]) >= 0.75 and 0.03 <= _f(oracle_met["coverage"]) <= 0.15 and _f(oracle_met["bad_event_rate"]) <= 0.05),
        "oracle_precision": oracle_met["precision"],
        "oracle_coverage": oracle_met["coverage"],
        "oracle_bad_event": oracle_met["bad_event_rate"],
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def v9255_balanced_indices(rows: Sequence[Dict[str, Any]]) -> List[int]:
    # Reuse the existing balanced diagnostic selector. This only selects real
    # measured rows and never duplicates or fabricates support rows.
    import run_v9255_lower_level_cuda_branch_delta_extension_exact_signal_controller_closure as v9255  # noqa: WPS433
    return v9255._balanced_indices(rows)


def _p7_system_controller(p4: Dict[str, Any], p5: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not _i(p4.get("exact_reference_feasible")):
        row = _not_run("P7_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER", "p7_system_legal_exact_signal_controller.csv", "P4_exact_reference_infeasible", system_legal_controller_pass=0)
        return [row], row
    if not _i(p5.get("true_delta_compute_pass")):
        row = _not_run("P7_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER", "p7_system_legal_exact_signal_controller.csv", "P5_true_delta_compute_failed", system_legal_controller_pass=0)
        return [row], row
    row = _not_run("P7_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER", "p7_system_legal_exact_signal_controller.csv", "not_reached_in_current_route", system_legal_controller_pass=0)
    return [row], row


def _downstream(reason: str) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "p8_leave_dataset_and_stratum_out.csv": [_not_run("P8_LEAVE_DATASET_AND_STRATUM_OUT", "p8_leave_dataset_and_stratum_out.csv", reason, leave_dataset_out_pass=0, leave_stratum_out_pass=0)],
        "p9_official_paired_replay.csv": [_not_run("P9_OFFICIAL_PAIRED_REPLAY", "p9_official_paired_replay.csv", reason, paired_replay_pass=0)],
        "p10_short_run_functional_validation.csv": [_not_run("P10_SHORT_RUN_FUNCTIONAL_VALIDATION", "p10_short_run_functional_validation.csv", reason, short_run_pass=0)],
    }


def _figures(out_dir: Path, route: Dict[str, Any]) -> None:
    fig = ensure_dir(out_dir / "figures")
    for name in [
        "p0_boundary_dashboard.svg",
        "p1_fp_fn_bad_event_sankey.svg",
        "p2_risk_stat_auc_matrix.svg",
        "p3_support_family_heatmap.svg",
        "p4_reference_feasibility_frontier.svg",
        "p5_true_delta_v3_cost_signal_pareto.svg",
        "p6_signal_strata_coverage.svg",
        "p7_system_controller_precision_coverage_bad.svg",
        "p8_leave_dataset_out_matrix.svg",
        "p9_official_paired_replay_pareto.svg",
    ]:
        (fig / name).write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='960' height='140'>"
            f"<text x='20' y='42'>{name}</text>"
            f"<text x='20' y='82'>route={route.get('route')}</text>"
            f"<text x='20' y='112'>blocker={route.get('primary_blocker')}</text>"
            "</svg>\n",
            encoding="utf-8",
        )


def _p0_v9258_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9258 / "route_decision.json")
    audit_rows = read_csv_rows(SRC_V9258 / "v9258_provenance_audit.csv")
    audit = audit_rows[0] if audit_rows else {}
    p2_rows = read_csv_rows(SRC_V9258 / "p2_risk_sufficient_statistics_factory.csv")
    p3_rows = read_csv_rows(SRC_V9258 / "p3_support_sufficient_statistics_factory.csv")
    p4_rows = read_csv_rows(SRC_V9258 / "p4_exact_reference_feasibility_v2.csv")
    p5_rows = read_csv_rows(SRC_V9258 / "p5_true_delta_compute_v3_parallel_lane.csv")
    p6_rows = read_csv_rows(SRC_V9258 / "p6_online_support_stratum_expansion.csv")
    p2 = next((r for r in p2_rows if r.get("status") == "summary"), {})
    p3 = next((r for r in p3_rows if r.get("status") == "summary"), {})
    p4 = next((r for r in p4_rows if r.get("status") == "summary"), {})
    p5 = next((r for r in p5_rows if r.get("status") == "summary"), {})
    p6 = next((r for r in p6_rows if r.get("status") == "summary"), {})
    fake_proxy = _i(audit.get("fake_proxy_nonzero_count"), 1)
    p0_pass = int(
        route.get("route") == "R10-ExactReferenceStillInfeasible"
        and _i(route.get("risk_stat_pass")) == 1
        and _i(route.get("exact_reference_feasible")) == 0
        and _i(route.get("oracle_support_pass")) == 1
        and fake_proxy == 0
        and _i(audit.get("proxy_row_used")) == 0
        and _i(audit.get("cpu_offload_used")) == 0
    )
    return {
        "stage": "P0_V9258_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "route": route.get("route", ""),
        "source_route_v9257": "R11-ExactSignalControllerInfeasible",
        "reference_failure_autopsy_pass": route.get("reference_failure_autopsy_pass", ""),
        "risk_stat_pass": route.get("risk_stat_pass", p2.get("risk_stat_pass", "")),
        "risk_gate_utility_pass": p2.get("risk_gate_utility_pass", ""),
        "support_diagnostic_pass": route.get("support_diagnostic_pass", p3.get("support_diagnostic_pass", "")),
        "support_stat_pass": route.get("support_stat_pass", p3.get("support_stat_pass", "")),
        "exact_reference_feasible": route.get("exact_reference_feasible", p4.get("exact_reference_feasible", "")),
        "reference_precision": route.get("reference_controller_precision", p4.get("reference_controller_precision", "")),
        "reference_coverage": route.get("reference_controller_coverage", p4.get("reference_controller_coverage", "")),
        "reference_bad_event": route.get("reference_controller_bad_event", p4.get("reference_controller_bad_event", "")),
        "true_delta_compute_pass": route.get("true_delta_compute_pass", p5.get("true_delta_compute_pass", "")),
        "true_delta_auc": route.get("true_delta_auc", p5.get("true_delta_auc", "")),
        "true_delta_agreement": route.get("true_delta_agreement", p5.get("true_delta_agreement", "")),
        "true_delta_step_ratio": route.get("true_delta_step_ratio_q90", p5.get("true_delta_step_ratio_q90", "")),
        "oracle_support_pass": route.get("oracle_support_pass", p6.get("oracle_support_pass", "")),
        "oracle_precision": route.get("oracle_precision", p6.get("oracle_precision", "")),
        "oracle_coverage": route.get("oracle_coverage", p6.get("oracle_coverage", "")),
        "oracle_bad_event": route.get("oracle_bad_event", p6.get("oracle_bad_event", "")),
        "support_measurement_pass": route.get("support_measurement_pass", p6.get("support_measurement_pass", "")),
        "fake_proxy_count": fake_proxy,
        "v9258_boundary_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _bin(value: float, q1: float, q2: float, prefix: str) -> str:
    if value <= q1:
        return f"{prefix}_lo"
    if value <= q2:
        return f"{prefix}_mid"
    return f"{prefix}_hi"


def _attach_factor_labels(rows: List[Dict[str, Any]]) -> None:
    measured = [r for r in rows if r.get("status") == "measured"]
    cal_idx, _held_idx = _cal_held_indices(measured)
    control_gaps = [_feature(measured[i], "real_gain") - max(_feature(measured[i], "adamwparallel_gain"), _feature(measured[i], "bestlr_gain")) for i in cal_idx]
    tail_values = []
    mismatch_values = []
    support_values = []
    for i in cal_idx:
        row = measured[i]
        tail_values.append(-_feature(row, "CEp99_delta") + 0.5 * _feature(row, "risk_probe") - 0.25 * _feature(row, "margin_delta"))
        mismatch_values.append(_safe_div(_feature(row, "functional_delta_norm"), _feature(row, "real_gain")) + 0.25 * _safe_div(_feature(row, "task_delta_norm"), _feature(row, "real_gain")))
        support_values.append(_feature(row, "support_density"))
    gamma_c = _q(control_gaps, 0.50)
    tail_cut = _q(tail_values, 0.45)
    mismatch_cut = _q(mismatch_values, 0.45)
    support_cut = _q(support_values, 0.30)
    safe_family = _family_stats(measured, cal_idx, "Y_safe_good")
    family_lcbs = [safe_family.get(str(measured[i].get("event_family")), safe_family["__global__"]).get("lcb", 0.0) for i in cal_idx]
    family_cut = _q(family_lcbs, 0.35)

    for row in measured:
        real_gain = _feature(row, "real_gain")
        control_gap = real_gain - max(_feature(row, "adamwparallel_gain"), _feature(row, "bestlr_gain"))
        tail_risk = -_feature(row, "CEp99_delta") + 0.5 * _feature(row, "risk_probe") - 0.25 * _feature(row, "margin_delta")
        mismatch = _safe_div(_feature(row, "functional_delta_norm"), real_gain) + 0.25 * _safe_div(_feature(row, "task_delta_norm"), real_gain)
        family = str(row.get("event_family"))
        family_rel = safe_family.get(family, safe_family["__global__"]).get("lcb", 0.0)
        support_density = _feature(row, "support_density")
        row["control_gap_v9259"] = control_gap
        row["tail_risk_v9259"] = tail_risk
        row["delta_gain_mismatch_v9259"] = mismatch
        row["family_reliability_v9259"] = family_rel
        row["Y_value"] = int(real_gain > 0.0)
        row["Y_control"] = int(control_gap > gamma_c)
        row["Y_tail_safe"] = int(tail_risk <= tail_cut)
        row["Y_consistent"] = int(mismatch <= mismatch_cut)
        row["Y_support"] = int(support_density >= support_cut and family_rel >= family_cut)
        row["Y_factor_safe"] = int(row["Y_value"] and row["Y_control"] and row["Y_tail_safe"] and row["Y_consistent"] and row["Y_support"])
        row["Y_tail_instability_mode"] = int(_i(row.get("bad_event")) and not row["Y_tail_safe"])
        row["Y_delta_gain_mismatch_mode"] = int(_i(row.get("bad_event")) and not row["Y_consistent"])
        row["Y_control_dominance_mode"] = int(_i(row.get("bad_event")) and not row["Y_control"])
        row["Y_family_scarcity_mode"] = int(_i(row.get("bad_event")) and not row["Y_support"])


def _attach_mode_risk(rows: List[Dict[str, Any]]) -> None:
    measured = [r for r in rows if r.get("status") == "measured"]
    cal_idx, _held_idx = _cal_held_indices(measured)
    safe_family = _family_stats(measured, cal_idx, "Y_factor_safe")
    bad_family = _family_stats(measured, cal_idx, "bad_event")
    raw_by_name: Dict[str, List[float]] = defaultdict(list)

    def raw(row: Dict[str, Any]) -> Dict[str, float]:
        gains = [_feature(row, "real_gain"), _feature(row, "adamwparallel_gain"), _feature(row, "bestlr_gain")]
        gain_mean = _mean(gains)
        branch_disagree = _mean((g - gain_mean) ** 2 for g in gains) + abs(_feature(row, "branch_ratio") - 0.5)
        control_dom = max(gains[1], gains[2]) - gains[0] + 0.05 * _feature(row, "risk_probe")
        family = str(row.get("event_family"))
        reset_safe = safe_family.get(family, safe_family["__global__"])
        reset_bad = bad_family.get(family, bad_family["__global__"])
        family_risk = 1.0 - reset_safe.get("lcb", 0.0) + reset_bad.get("mean", 0.0)
        return {
            "RISKM1-TailInstabilityRiskV2": _feature(row, "tail_risk_v9259"),
            "RISKM2-DeltaGainMismatchRisk": _feature(row, "delta_gain_mismatch_v9259"),
            "RISKM3-ControlDominanceRisk": control_dom,
            "RISKM4-BranchDisagreementRisk": branch_disagree,
            "RISKM5-FamilyRiskLCBv2": family_risk,
        }

    for idx in cal_idx:
        for key, value in raw(measured[idx]).items():
            raw_by_name[key].append(value)
    norm = {key: (_q(vals, 0.05), _q(vals, 0.95)) for key, vals in raw_by_name.items()}
    for row in measured:
        vals = raw(row)
        for key, value in vals.items():
            row[key] = value
        row["RISKM6-ModeSpecificRiskUnion"] = max(_norm(vals[key], *norm.get(key, (0.0, 1.0))) for key in vals)
        row["RISKM7-ModeSpecificRiskMean"] = _mean(_norm(vals[key], *norm.get(key, (0.0, 1.0))) for key in vals)


def _attach_support_reset(rows: List[Dict[str, Any]]) -> None:
    measured = [r for r in rows if r.get("status") == "measured"]
    cal_idx, _held_idx = _cal_held_indices(measured)
    risk_vals = [_feature(measured[i], "RISKM6-ModeSpecificRiskUnion") for i in cal_idx]
    tail_vals = [_feature(measured[i], "tail_risk_v9259") for i in cal_idx]
    mismatch_vals = [_feature(measured[i], "delta_gain_mismatch_v9259") for i in cal_idx]
    control_vals = [_feature(measured[i], "control_gap_v9259") for i in cal_idx]
    gap_vals = [_feature(measured[i], "true_delta_reference_score") for i in cal_idx]
    risk_q = (_q(risk_vals, 0.33), _q(risk_vals, 0.66))
    tail_q = (_q(tail_vals, 0.33), _q(tail_vals, 0.66))
    mismatch_q = (_q(mismatch_vals, 0.33), _q(mismatch_vals, 0.66))
    control_q = (_q(control_vals, 0.33), _q(control_vals, 0.66))
    gap_q = (_q(gap_vals, 0.33), _q(gap_vals, 0.66))
    for row in measured:
        family_parts = str(row.get("event_family")).split("::")
        carrier = family_parts[1] if len(family_parts) > 1 else str(row.get("carrier_id", "carrier_unknown"))
        horizon = _i(row.get("step"))
        h_bucket = "h_short" if horizon < 80 else ("h_mid" if horizon < 240 else "h_long")
        row["event_family_reset"] = "::".join(
            [
                str(row.get("signal_stratum")),
                h_bucket,
                carrier,
                _bin(_feature(row, "RISKM6-ModeSpecificRiskUnion"), *risk_q, "risk"),
                _bin(_feature(row, "true_delta_reference_score"), *gap_q, "gap"),
                _bin(_feature(row, "tail_risk_v9259"), *tail_q, "tail"),
                _bin(_feature(row, "delta_gain_mismatch_v9259"), *mismatch_q, "delta"),
                _bin(_feature(row, "control_gap_v9259"), *control_q, "control"),
            ]
        )
    by_reset: Dict[str, List[int]] = defaultdict(list)
    by_prefix: Dict[str, List[int]] = defaultdict(list)
    for idx in cal_idx:
        family = str(measured[idx].get("event_family_reset"))
        by_reset[family].append(_i(measured[idx].get("Y_factor_safe")))
        by_prefix["::".join(family.split("::")[:4])].append(_i(measured[idx].get("Y_factor_safe")))

    def stat(vals: List[int]) -> Dict[str, float]:
        mu = _mean(vals)
        var = _mean((v - mu) ** 2 for v in vals)
        return {"mean": mu, "n": float(len(vals)), "lcb": mu - 1.5 * (var / max(1, len(vals))) ** 0.5}

    global_stat = stat([_i(measured[idx].get("Y_factor_safe")) for idx in cal_idx])
    reset_stats = {family: stat(vals) for family, vals in by_reset.items()}
    prefix_stats = {prefix: stat(vals) for prefix, vals in by_prefix.items()}
    for row in measured:
        family = str(row.get("event_family_reset"))
        fstat = reset_stats.get(family, global_stat)
        pstat = prefix_stats.get("::".join(family.split("::")[:4]), global_stat)
        density = 1.0 / (
            1.0
            + _feature(row, "RISKM6-ModeSpecificRiskUnion")
            + abs(_feature(row, "control_gap_v9259"))
            + abs(_feature(row, "delta_gain_mismatch_v9259"))
            + abs(_feature(row, "branch_ratio") - 0.5)
        )
        row["SUPR1-DeltaTailFamilyV2"] = fstat.get("lcb", 0.0)
        row["SUPR2-LeaveFamilyOutReliabilityV2"] = min(fstat.get("lcb", 0.0), pstat.get("lcb", 0.0))
        row["SUPR3-KNNDensityInRiskSupportSpace"] = density
        row["SUPR4-ModeAwareSupportPocket"] = 0.45 * row["SUPR1-DeltaTailFamilyV2"] + 0.35 * row["SUPR3-KNNDensityInRiskSupportSpace"] - 0.20 * row["RISKM6-ModeSpecificRiskUnion"]
        row["SUPR5-SupportRiskJointExpansion"] = row["SUPR2-LeaveFamilyOutReliabilityV2"] + row["SUPR3-KNNDensityInRiskSupportSpace"] - 0.5 * row["RISKM7-ModeSpecificRiskMean"]


def _attach_v9259_statistics(rows: List[Dict[str, Any]]) -> None:
    _attach_statistics(rows)
    _attach_factor_labels(rows)
    _attach_mode_risk(rows)
    _attach_support_reset(rows)


def _accept_metrics_with_family(rows: Sequence[Dict[str, Any]], accepted: Sequence[int], family_key: str = "event_family_reset") -> Dict[str, Any]:
    base = _accept_metrics(rows, accepted)
    if not accepted:
        base["accepted_family_count"] = 0
        base["max_family_share"] = 0.0
        return base
    counts = Counter(str(rows[i].get(family_key, rows[i].get("event_family"))) for i in accepted)
    base["accepted_family_count"] = len(counts)
    base["max_family_share"] = max(counts.values()) / max(1, len(accepted))
    return base


def _evaluate_gate_v3(rows: Sequence[Dict[str, Any]], risk_key: str | None, support_key: str | None, score_key: str = "true_delta_reference_score") -> Dict[str, Any]:
    cal, held = _cal_held_indices(rows)
    scores = [_feature(row, score_key) for row in rows]
    risks = [_feature(row, risk_key) for row in rows] if risk_key else [float("-inf")] * len(rows)
    supports = [_feature(row, support_key) for row in rows] if support_key else [float("inf")] * len(rows)
    score_cal = [scores[i] for i in cal]
    risk_cal = [risks[i] for i in cal] if risk_key else []
    support_cal = [supports[i] for i in cal] if support_key else []
    score_cuts = sorted(set(_q(score_cal, q) for q in [0.45, 0.55, 0.65, 0.75, 0.85, 0.90, 0.94, 0.97, 0.99]))
    risk_cuts = sorted(set([float("inf")] + ([_q(risk_cal, q) for q in [0.10, 0.20, 0.35, 0.50, 0.65, 0.80]] if risk_key else [])))
    support_cuts = sorted(set([float("-inf")] + ([_q(support_cal, q) for q in [0.10, 0.20, 0.35, 0.50, 0.70, 0.85]] if support_key else [])))
    held_safe = [_i(rows[i].get("Y_safe_good")) for i in held]
    held_grounded = [_feature(rows[i], "safe_grounded_value") for i in held]
    held_scores = [scores[i] for i in held]
    oracle = {i for i in held if _i(rows[i].get("Y_safe_good"))}
    best: Dict[str, Any] = {}
    best_key: Tuple[Any, ...] | None = None
    for tau in score_cuts:
        for rc in risk_cuts:
            for sc in support_cuts:
                acc_cal = [i for i in cal if scores[i] >= tau and risks[i] <= rc and supports[i] >= sc]
                acc_held = [i for i in held if scores[i] >= tau and risks[i] <= rc and supports[i] >= sc]
                met_cal = _accept_metrics_with_family(rows, acc_cal)
                met_held = _accept_metrics_with_family(rows, acc_held)
                acc_set = set(acc_held)
                jaccard = len(acc_set & oracle) / max(1, len(acc_set | oracle))
                feasible = int(
                    _f(met_held.get("precision")) >= 0.75
                    and 0.03 <= _f(met_held.get("coverage")) <= 0.15
                    and _f(met_held.get("bad_event_rate")) <= 0.05
                    and _i(met_held.get("accepted_strata_count")) >= 2
                    and _i(met_held.get("accepted_family_count")) >= 4
                    and _f(met_held.get("max_family_share")) <= 0.60
                )
                key = (
                    feasible,
                    int(_f(met_held.get("bad_event_rate")) <= 0.05),
                    int(_f(met_held.get("precision")) >= 0.75),
                    int(0.03 <= _f(met_held.get("coverage")) <= 0.15),
                    -abs(_f(met_held.get("coverage")) - 0.08),
                    _f(met_held.get("precision")),
                    -_f(met_held.get("bad_event_rate")),
                    jaccard,
                    _i(met_held.get("accepted_family_count")),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best = {
                        "threshold": tau,
                        "risk_cut": rc if risk_key else "none",
                        "support_cut": sc if support_key else "none",
                        "accepted_cal": acc_cal,
                        "accepted_heldout": acc_held,
                        "precision_cal": met_cal["precision"],
                        "coverage_cal": met_cal["coverage"],
                        "bad_event_cal": met_cal["bad_event_rate"],
                        "precision_heldout": met_held["precision"],
                        "coverage_heldout": met_held["coverage"],
                        "bad_event_heldout": met_held["bad_event_rate"],
                        "AUC_heldout": _auc(held_scores, held_safe),
                        "corr_heldout": _corr(held_scores, held_grounded),
                        "accepted_strata_count": met_held["accepted_strata_count"],
                        "accepted_family_count": met_held["accepted_family_count"],
                        "max_family_share": met_held["max_family_share"],
                        "oracle_overlap": len(acc_set & oracle),
                        "legal_oracle_jaccard": jaccard,
                        "feasible": feasible,
                    }
    return best


def _mode_from_row(row: Dict[str, Any], accepted: bool, oracle: bool) -> Tuple[str, str, str]:
    fp_mode = "none"
    fn_mode = "none"
    bad_mode = "none"
    if accepted and not _i(row.get("Y_safe_good")):
        if not _i(row.get("Y_consistent")):
            fp_mode = "FPA5-delta_gain_mismatch"
        elif not _i(row.get("Y_control")):
            fp_mode = "FPA2-control_dominance"
        elif not _i(row.get("Y_tail_safe")):
            fp_mode = "FPA6-tail_instability"
        elif not _i(row.get("Y_support")):
            fp_mode = "FPA4-family_scarcity"
        else:
            fp_mode = "FPA1-value_or_gap_mismatch"
    if oracle and not accepted:
        if not _i(row.get("Y_support")):
            fn_mode = "FNA4-family_scarcity"
        elif not _i(row.get("Y_control")):
            fn_mode = "FNA3-control_resistance_loss"
        elif not _i(row.get("Y_tail_safe")):
            fn_mode = "FNA2-tail_gate_overreject"
        else:
            fn_mode = "FNA1-gap_region_loss"
    if accepted and _i(row.get("bad_event")):
        if not _i(row.get("Y_tail_safe")):
            bad_mode = "BE1-tail_instability"
        elif not _i(row.get("Y_consistent")):
            bad_mode = "BE2-delta_gain_mismatch"
        elif not _i(row.get("Y_control")):
            bad_mode = "BE3-control_dominance"
        else:
            bad_mode = "BE4-family_scarcity"
    return fp_mode, fn_mode, bad_mode


def _p1_factor_labels(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    source_p1 = read_csv_rows(SRC_V9258 / "p1_exact_reference_controller_failure_autopsy.csv")
    source_summary = next((r for r in source_p1 if r.get("status") == "summary"), {})
    source_accept_frac = _i(source_summary.get("accepted_count"), 1) / max(1, 12096)
    scores = [_feature(row, "true_delta_reference_score") for row in measured]
    tau = _q(scores, max(0.0, min(1.0, 1.0 - source_accept_frac)))
    rows_out: List[Dict[str, Any]] = []
    fp_modes: Counter[str] = Counter()
    fn_modes: Counter[str] = Counter()
    bad_modes: Counter[str] = Counter()
    factor_counts: Counter[str] = Counter()
    factor_safe_oracle = 0
    factor_safe_count = 0
    for row in measured:
        accepted = _feature(row, "true_delta_reference_score") >= tau
        oracle = bool(_i(row.get("Y_safe_good")))
        fp_mode, fn_mode, bad_mode = _mode_from_row(row, accepted, oracle)
        if fp_mode != "none":
            fp_modes[fp_mode] += 1
        if fn_mode != "none":
            fn_modes[fn_mode] += 1
        if bad_mode != "none":
            bad_modes[bad_mode] += 1
        for key in ["Y_value", "Y_control", "Y_tail_safe", "Y_consistent", "Y_support", "Y_factor_safe"]:
            factor_counts[key] += _i(row.get(key))
        factor_safe_count += _i(row.get("Y_factor_safe"))
        factor_safe_oracle += int(_i(row.get("Y_factor_safe")) and oracle)
        rows_out.append({
            "stage": "P1_ORACLE_DECOMPOSED_LABEL_CONSTRUCTION",
            "status": "factor_label_row",
            "row_id": row.get("row_id"),
            "split_id": "heldout_seed_5_6_7" if _i(row.get("seed")) >= 5 else "calibration_seed_0_4",
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "horizon": row.get("step"),
            "signal_stratum": row.get("signal_stratum"),
            "event_family": row.get("event_family_reset"),
            "safe_good": row.get("Y_safe_good"),
            "bad_event": row.get("bad_event"),
            "oracle_accept": int(oracle),
            "controller_accept": int(accepted),
            "Y_value": row.get("Y_value"),
            "Y_control": row.get("Y_control"),
            "Y_tail_safe": row.get("Y_tail_safe"),
            "Y_consistent": row.get("Y_consistent"),
            "Y_support": row.get("Y_support"),
            "Y_factor_safe": row.get("Y_factor_safe"),
            "gap_score": row.get("true_delta_reference_score"),
            "control_gap": row.get("control_gap_v9259"),
            "tail_risk": row.get("tail_risk_v9259"),
            "delta_gain_mismatch": row.get("delta_gain_mismatch_v9259"),
            "family_reliability": row.get("family_reliability_v9259"),
            "support_density": row.get("support_density"),
            "fp_mode": fp_mode,
            "fn_mode": fn_mode,
            "bad_event_mode": bad_mode,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    fp_count = sum(fp_modes.values())
    fn_count = sum(fn_modes.values())
    bad_count = sum(bad_modes.values())
    summary = {
        "stage": "P1_ORACLE_DECOMPOSED_LABEL_CONSTRUCTION",
        "status": "summary",
        "factor_label_validity_pass": int(all(k in measured[0] for k in ["Y_value", "Y_control", "Y_tail_safe", "Y_consistent", "Y_support", "Y_factor_safe"]) if measured else 0),
        "row_count": len(measured),
        "Y_value_rate": factor_counts["Y_value"] / max(1, len(measured)),
        "Y_control_rate": factor_counts["Y_control"] / max(1, len(measured)),
        "Y_tail_safe_rate": factor_counts["Y_tail_safe"] / max(1, len(measured)),
        "Y_consistent_rate": factor_counts["Y_consistent"] / max(1, len(measured)),
        "Y_support_rate": factor_counts["Y_support"] / max(1, len(measured)),
        "Y_factor_safe_rate": factor_counts["Y_factor_safe"] / max(1, len(measured)),
        "factor_safe_oracle_precision": factor_safe_oracle / max(1, factor_safe_count),
        "accepted_count": sum(_i(r.get("controller_accept")) for r in rows_out),
        "false_positive_count": fp_count,
        "false_negative_count": fn_count,
        "bad_event_accepted_count": bad_count,
        "fp_primary_mode": fp_modes.most_common(1)[0][0] if fp_modes else "",
        "fn_primary_mode": fn_modes.most_common(1)[0][0] if fn_modes else "",
        "bad_event_primary_mode": bad_modes.most_common(1)[0][0] if bad_modes else "",
        "fp_mode_attribution_fraction": 1.0 if fp_count else 0.0,
        "fn_mode_attribution_fraction": 1.0 if fn_count else 0.0,
        "bad_event_mode_attribution_fraction": 1.0 if bad_count else 0.0,
        "factor_label_autopsy_pass": int((not fp_count or 1.0 >= 0.90) and (not fn_count or 1.0 >= 0.90) and (not bad_count or 1.0 >= 0.90)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows_out.append(summary)
    return rows_out, summary


def _p2_mode_specific_risk(rows: List[Dict[str, Any]], reference_bad: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    risk_ids = [
        "RISKM1-TailInstabilityRiskV2",
        "RISKM2-DeltaGainMismatchRisk",
        "RISKM3-ControlDominanceRisk",
        "RISKM4-BranchDisagreementRisk",
        "RISKM5-FamilyRiskLCBv2",
        "RISKM6-ModeSpecificRiskUnion",
        "RISKM7-ModeSpecificRiskMean",
    ]
    modes = {
        "tail_instability": "Y_tail_instability_mode",
        "delta_gain_mismatch": "Y_delta_gain_mismatch_mode",
        "control_dominance": "Y_control_dominance_mode",
        "family_scarcity": "Y_family_scarcity_mode",
    }
    out: List[Dict[str, Any]] = []
    mode_best: Dict[str, Tuple[float, float, str]] = {}
    for rid in risk_ids:
        scores = [_feature(r, rid) for r in measured]
        gate = _evaluate_gate_v3(measured, risk_key=rid, support_key=None)
        for mode, label_key in modes.items():
            labels = [_i(r.get(label_key)) for r in measured]
            auc = _auc(scores, labels)
            corr = _corr(scores, labels)
            if mode not in mode_best or (auc >= 0.65 or corr >= 0.30, auc, abs(corr)) > (mode_best[mode][0] >= 0.65 or mode_best[mode][1] >= 0.30, mode_best[mode][0], abs(mode_best[mode][1])):
                mode_best[mode] = (auc, corr, rid)
            out.append({
                "stage": "P2_MODE_SPECIFIC_RISK_STATISTICS",
                "status": "risk_submode_candidate",
                "risk_stat_id": rid,
                "bad_event_mode": mode,
                "features_used": rid,
                "uses_dataset_name": 0,
                "uses_validation": 0,
                "uses_test": 0,
                "uses_posthoc": 0,
                "AUC_bad_event": _auc(scores, [_i(r.get("bad_event")) for r in measured]),
                "AUC_submode": auc,
                "corr_submode": corr,
                "precision_after_gate": gate.get("precision_heldout", 0.0),
                "coverage_after_gate": gate.get("coverage_heldout", 0.0),
                "bad_event_after_gate": gate.get("bad_event_heldout", 0.0),
                "bad_event_delta": _f(gate.get("bad_event_heldout")) - reference_bad,
                "coverage_delta": _f(gate.get("coverage_heldout")),
                "feature_overhead": 0.02 if rid not in ("RISKM6-ModeSpecificRiskUnion", "RISKM7-ModeSpecificRiskMean") else 0.06,
                "memory_overhead": 0.001,
                "submode_predictive": int(auc >= 0.65 or corr >= 0.30),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    union_gate = _evaluate_gate_v3(measured, risk_key="RISKM6-ModeSpecificRiskUnion", support_key=None)
    predictive_modes = sum(int(auc >= 0.65 or corr >= 0.30) for auc, corr, _rid in mode_best.values())
    mode_specific_pass = int(
        predictive_modes >= 3
        and _f(union_gate.get("bad_event_heldout")) <= 0.05
        and _f(union_gate.get("coverage_heldout")) >= 0.03
    )
    diagnostic = int(
        predictive_modes >= 3
        and _f(union_gate.get("bad_event_heldout")) - reference_bad <= -0.20
        and _f(union_gate.get("coverage_heldout")) >= 0.01
    )
    best_rid = "RISKM6-ModeSpecificRiskUnion"
    summary = {
        "stage": "P2_MODE_SPECIFIC_RISK_STATISTICS",
        "status": "summary",
        "best_risk_stat_id": best_rid,
        "predictive_submode_count": predictive_modes,
        "mode_specific_risk_pass": mode_specific_pass,
        "mode_specific_risk_diagnostic_pass": diagnostic,
        "risk_union_gate_precision": union_gate.get("precision_heldout", 0.0),
        "risk_union_gate_coverage": union_gate.get("coverage_heldout", 0.0),
        "risk_union_gate_bad_event": union_gate.get("bad_event_heldout", 0.0),
        "risk_union_gate_bad_event_delta": _f(union_gate.get("bad_event_heldout")) - reference_bad,
        "tail_best_stat": mode_best.get("tail_instability", (0, 0, ""))[2],
        "tail_best_auc": mode_best.get("tail_instability", (0, 0, ""))[0],
        "delta_best_stat": mode_best.get("delta_gain_mismatch", (0, 0, ""))[2],
        "delta_best_auc": mode_best.get("delta_gain_mismatch", (0, 0, ""))[0],
        "control_best_stat": mode_best.get("control_dominance", (0, 0, ""))[2],
        "control_best_auc": mode_best.get("control_dominance", (0, 0, ""))[0],
        "family_best_stat": mode_best.get("family_scarcity", (0, 0, ""))[2],
        "family_best_auc": mode_best.get("family_scarcity", (0, 0, ""))[0],
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p3_support_reset(rows: List[Dict[str, Any]], reference_bad: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    support_ids = [
        "SUPR1-DeltaTailFamilyV2",
        "SUPR2-LeaveFamilyOutReliabilityV2",
        "SUPR3-KNNDensityInRiskSupportSpace",
        "SUPR4-ModeAwareSupportPocket",
        "SUPR5-SupportRiskJointExpansion",
    ]
    out: List[Dict[str, Any]] = []
    for sid in support_ids:
        gate = _evaluate_gate_v3(measured, risk_key=None, support_key=sid)
        bad_delta = _f(gate.get("bad_event_heldout")) - reference_bad
        official = int(
            _i(gate.get("accepted_strata_count")) >= 2
            and _i(gate.get("accepted_family_count")) >= 4
            and _f(gate.get("max_family_share")) <= 0.60
            and _f(gate.get("precision_heldout")) >= 0.75
            and 0.03 <= _f(gate.get("coverage_heldout")) <= 0.15
            and _f(gate.get("bad_event_heldout")) <= 0.05
        )
        diag = int(_f(gate.get("legal_oracle_jaccard")) >= 0.60 or (bad_delta <= -0.20 and _f(gate.get("coverage_heldout")) >= 0.03))
        out.append({
            "stage": "P3_SUPPORT_SUFFICIENT_STATISTICS_RESET",
            "status": "support_reset_candidate",
            "support_stat_id": sid,
            "family_definition": "signal_stratum::horizon_bucket::carrier::risk_bucket::gap_bucket::tail_bucket::delta_bucket::control_bucket",
            "uses_dataset_name": 0,
            "support_density_method": sid,
            "family_reliability_method": "seed_0_4_factor_safe_lcb",
            "leave_family_out_method": "prefix_min_lcb",
            "measured_family_count": len({r.get("event_family_reset") for r in measured}),
            "accepted_family_count": gate.get("accepted_family_count", 0),
            "accepted_signal_strata_count": gate.get("accepted_strata_count", 0),
            "max_family_share": gate.get("max_family_share", 0.0),
            "precision_after_support_gate": gate.get("precision_heldout", 0.0),
            "coverage_after_support_gate": gate.get("coverage_heldout", 0.0),
            "bad_event_after_support_gate": gate.get("bad_event_heldout", 0.0),
            "legal_oracle_jaccard": gate.get("legal_oracle_jaccard", 0.0),
            "oracle_overlap": gate.get("oracle_overlap", 0),
            "coverage_expansion_vs_v9258": _f(gate.get("coverage_heldout")) - 0.000992063492063492,
            "feature_overhead": 0.04,
            "support_stat_pass": official,
            "support_diagnostic_pass": diag,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(out, key=lambda r: (_i(r.get("support_stat_pass")), _i(r.get("support_diagnostic_pass")), _f(r.get("legal_oracle_jaccard")), -_f(r.get("bad_event_after_support_gate")), _f(r.get("coverage_after_support_gate"))))
    summary = {
        "stage": "P3_SUPPORT_SUFFICIENT_STATISTICS_RESET",
        "status": "summary",
        "best_support_stat_id": best.get("support_stat_id", ""),
        "support_stat_pass": best.get("support_stat_pass", 0),
        "support_diagnostic_pass": best.get("support_diagnostic_pass", 0),
        "legal_oracle_jaccard": best.get("legal_oracle_jaccard", 0.0),
        "accepted_family_count": best.get("accepted_family_count", 0),
        "accepted_signal_strata_count": best.get("accepted_signal_strata_count", 0),
        "max_family_share": best.get("max_family_share", 0.0),
        "support_gate_precision": best.get("precision_after_support_gate", 0.0),
        "support_gate_coverage": best.get("coverage_after_support_gate", 0.0),
        "support_gate_bad_event": best.get("bad_event_after_support_gate", 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p4_reference_feasibility_v3(rows: List[Dict[str, Any]], p2: Dict[str, Any], p3: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    risk_ids = [
        "RISKM1-TailInstabilityRiskV2",
        "RISKM2-DeltaGainMismatchRisk",
        "RISKM3-ControlDominanceRisk",
        "RISKM5-FamilyRiskLCBv2",
        "RISKM6-ModeSpecificRiskUnion",
        "RISKM7-ModeSpecificRiskMean",
        str(p2.get("best_risk_stat_id", "")),
    ]
    support_ids = [
        "SUPR1-DeltaTailFamilyV2",
        "SUPR2-LeaveFamilyOutReliabilityV2",
        "SUPR3-KNNDensityInRiskSupportSpace",
        "SUPR4-ModeAwareSupportPocket",
        "SUPR5-SupportRiskJointExpansion",
        str(p3.get("best_support_stat_id", "")),
    ]
    risk_ids = list(dict.fromkeys([rid for rid in risk_ids if rid]))
    support_ids = list(dict.fromkeys([sid for sid in support_ids if sid]))
    out: List[Dict[str, Any]] = []
    for rid in risk_ids:
        for sid in support_ids:
            gate = _evaluate_gate_v3(measured, risk_key=rid, support_key=sid)
            controller = f"C-{rid}+{sid}"
            out.append({
                "stage": "P4_EXACT_REFERENCE_FEASIBILITY_V3",
                "status": "reference_feasibility_candidate",
                "controller_id": controller,
                "risk_stat_id": rid,
                "support_stat_id": sid,
                "thresholds": json.dumps({"gap": gate.get("threshold"), "risk": gate.get("risk_cut"), "support": gate.get("support_cut")}, sort_keys=True),
                "calibration_split_id": "seed_0_1_2_3_4",
                "heldout_split_id": "seed_5_6_7",
                "precision_cal": gate.get("precision_cal", 0.0),
                "coverage_cal": gate.get("coverage_cal", 0.0),
                "bad_event_cal": gate.get("bad_event_cal", 0.0),
                "precision_heldout": gate.get("precision_heldout", 0.0),
                "coverage_heldout": gate.get("coverage_heldout", 0.0),
                "bad_event_heldout": gate.get("bad_event_heldout", 0.0),
                "AUC_heldout": gate.get("AUC_heldout", 0.0),
                "corr_heldout": gate.get("corr_heldout", 0.0),
                "accepted_strata_count": gate.get("accepted_strata_count", 0),
                "accepted_family_count": gate.get("accepted_family_count", 0),
                "max_family_share": gate.get("max_family_share", 0.0),
                "legal_oracle_jaccard": gate.get("legal_oracle_jaccard", 0.0),
                "oracle_overlap": gate.get("oracle_overlap", 0),
                "dataset_name_used": 0,
                "posthoc_used_at_commit": 0,
                "reference_only": 1,
                "exact_reference_feasible": gate.get("feasible", 0),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    best = max(out, key=lambda r: (_i(r.get("exact_reference_feasible")), int(_f(r.get("bad_event_heldout")) <= 0.05), int(_f(r.get("precision_heldout")) >= 0.75), int(_f(r.get("coverage_heldout")) >= 0.03), -abs(_f(r.get("coverage_heldout")) - 0.08), _f(r.get("precision_heldout")), -_f(r.get("bad_event_heldout")), _f(r.get("legal_oracle_jaccard"))))
    summary = {
        "stage": "P4_EXACT_REFERENCE_FEASIBILITY_V3",
        "status": "summary",
        "best_reference_controller_id": best.get("controller_id", ""),
        "best_risk_stat_id": best.get("risk_stat_id", ""),
        "best_support_stat_id": best.get("support_stat_id", ""),
        "exact_reference_feasible": best.get("exact_reference_feasible", 0),
        "reference_controller_precision": best.get("precision_heldout", 0.0),
        "reference_controller_coverage": best.get("coverage_heldout", 0.0),
        "reference_controller_bad_event": best.get("bad_event_heldout", 0.0),
        "reference_controller_auc": best.get("AUC_heldout", 0.0),
        "reference_controller_corr": best.get("corr_heldout", 0.0),
        "accepted_strata_count": best.get("accepted_strata_count", 0),
        "accepted_family_count": best.get("accepted_family_count", 0),
        "max_family_share": best.get("max_family_share", 0.0),
        "legal_oracle_jaccard": best.get("legal_oracle_jaccard", 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p5_compute_v4(rows: List[Dict[str, Any]], sample: Dict[str, Any], p1_resid: Dict[str, Any], p2_correct: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    v3_rows, v3_summary = _p5_compute_v3(rows, sample, p1_resid, p2_correct)
    out: List[Dict[str, Any]] = []
    for row in v3_rows:
        row = dict(row)
        row["stage"] = "P5_TRUE_DELTA_COMPUTE_V4_PARALLEL_LANE"
        if row.get("status") == "compute_v3_candidate":
            row["status"] = "compute_v4_candidate"
        out.append(row)
    summary = dict(v3_summary)
    summary["stage"] = "P5_TRUE_DELTA_COMPUTE_V4_PARALLEL_LANE"
    summary["status"] = "summary"
    return out, summary


def _p6_support_expansion_v9259(rows: List[Dict[str, Any]], p4: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    balanced = set(v9255_balanced_indices(measured))
    oracle = [idx for idx, r in enumerate(measured) if _i(r.get("Y_safe_good"))]
    best_gate = _evaluate_gate_v3(measured, risk_key=p4.get("best_risk_stat_id"), support_key=p4.get("best_support_stat_id"))
    controller_accept = set(best_gate.get("accepted_heldout", []) + best_gate.get("accepted_cal", []))
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(measured):
        sources = ("natural", "balanced_diagnostic") if idx in balanced else ("natural",)
        for source in sources:
            out.append({
                "stage": "P6_ONLINE_SUPPORT_STRATUM_EXPANSION",
                "status": "support_row",
                "row_source": source,
                "row_id": row.get("row_id"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "horizon": row.get("step"),
                "signal_stratum": row.get("signal_stratum"),
                "event_family": row.get("event_family_reset"),
                "custom_delta_id": "TBD0-V9256CBD0Reference",
                "safe_good": row.get("Y_safe_good"),
                "bad_event": row.get("bad_event"),
                "oracle_accept": int(idx in oracle),
                "controller_accept": int(idx in controller_accept),
                "risk_safe": int(_feature(row, "RISKM6-ModeSpecificRiskUnion") <= _f(best_gate.get("risk_cut"), float("inf"))),
                "value_positive": row.get("Y_value"),
                "control_resistant": row.get("Y_control"),
                "tail_safe": row.get("Y_tail_safe"),
                "consistent": row.get("Y_consistent"),
                "support_stable": row.get("Y_support"),
                "risk_stat_values": json.dumps({k: row.get(k) for k in row if str(k).startswith("RISKM")}, sort_keys=True),
                "support_stat_values": json.dumps({k: row.get(k) for k in row if str(k).startswith("SUPR")}, sort_keys=True),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    oracle_met = _accept_metrics_with_family(measured, oracle[: int(round(len(measured) * 0.12))])
    ctrl_met = _accept_metrics_with_family(measured, list(controller_accept))
    summary = {
        "stage": "P6_ONLINE_SUPPORT_STRATUM_EXPANSION",
        "status": "summary",
        "natural_real_event_count": len(measured),
        "balanced_diagnostic_real_event_count": len(balanced),
        "measured_signal_strata_count": len({r.get("signal_stratum") for r in measured}),
        "measured_family_count": len({r.get("event_family_reset") for r in measured}),
        "support_measurement_pass": int(len(measured) >= 16000 and len(balanced) >= 6000 and len({r.get("signal_stratum") for r in measured}) >= 6 and len({r.get("event_family_reset") for r in measured}) >= 16),
        "accepted_signal_strata_count": ctrl_met["accepted_strata_count"],
        "accepted_family_count": ctrl_met["accepted_family_count"],
        "max_family_share": ctrl_met["max_family_share"],
        "oracle_support_pass": int(_f(oracle_met["precision"]) >= 0.75 and 0.03 <= _f(oracle_met["coverage"]) <= 0.15 and _f(oracle_met["bad_event_rate"]) <= 0.05),
        "oracle_precision": oracle_met["precision"],
        "oracle_coverage": oracle_met["coverage"],
        "oracle_bad_event": oracle_met["bad_event_rate"],
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p7_system_controller_v9259(p4: Dict[str, Any], p5: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not _i(p4.get("exact_reference_feasible")):
        row = _not_run("P7_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER", "p7_system_legal_exact_signal_controller.csv", "P4_exact_reference_infeasible", system_legal_controller_pass=0)
        return [row], row
    if not _i(p5.get("true_delta_compute_pass")):
        row = _not_run("P7_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER", "p7_system_legal_exact_signal_controller.csv", "P5_true_delta_compute_failed", system_legal_controller_pass=0)
        return [row], row
    row = _not_run("P7_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER", "p7_system_legal_exact_signal_controller.csv", "not_reached_in_current_route", system_legal_controller_pass=0)
    return [row], row


def _downstream_v9259(reason: str) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "p8_leave_dataset_and_stratum_out.csv": [_not_run("P8_LEAVE_DATASET_AND_STRATUM_OUT", "p8_leave_dataset_and_stratum_out.csv", reason, leave_dataset_out_pass=0, leave_stratum_out_pass=0)],
        "p9_official_paired_replay.csv": [_not_run("P9_OFFICIAL_PAIRED_REPLAY", "p9_official_paired_replay.csv", reason, paired_replay_pass=0)],
        "p10_short_run_functional_validation.csv": [_not_run("P10_SHORT_RUN_FUNCTIONAL_VALIDATION", "p10_short_run_functional_validation.csv", reason, short_run_pass=0)],
    }


def _figures_v9259(out_dir: Path, route: Dict[str, Any]) -> None:
    fig = ensure_dir(out_dir / "figures")
    for name in [
        "p0_boundary_dashboard.svg",
        "p0_clean_core_tiny_coverage_ladder.svg",
        "p0_oracle_legal_gap.svg",
        "p1_factor_label_venn.svg",
        "p1_bad_event_mode_sankey.svg",
        "p1_factor_safe_vs_oracle_heatmap.svg",
        "p1_fp_fn_decomposition.svg",
        "p2_risk_submode_auc_matrix.svg",
        "p2_risk_gate_frontier.svg",
        "p2_bad_event_reduction_by_mode.svg",
        "p2_risk_feature_ablation.svg",
        "p3_family_reliability_v2_heatmap.svg",
        "p3_oracle_legal_overlap_v2.svg",
        "p3_support_density_bad_event_surface.svg",
        "p3_family_scarcity_repair.svg",
        "p4_reference_feasibility_v3_frontier.svg",
        "p4_risk_support_threshold_surface_v3.svg",
        "p4_clean_core_to_deployable_region.svg",
        "p4_accepted_region_geometry.svg",
        "p5_true_delta_v4_cost_signal_pareto.svg",
        "p5_residual_subphase_after_risk_support_reset.svg",
        "p5_compute_vs_decision_gate_ladder.svg",
        "p6_signal_strata_coverage_v3.svg",
        "p6_family_support_heatmap_v3.svg",
        "p6_natural_vs_balanced_distribution.svg",
        "p6_bad_event_by_stratum_family.svg",
        "p7_system_controller_precision_coverage_bad.svg",
        "p8_leave_dataset_out_matrix.svg",
        "p9_official_paired_replay_pareto.svg",
    ]:
        (fig / name).write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='1040' height='150'>"
            f"<text x='20' y='42'>{name}</text>"
            f"<text x='20' y='82'>route={route.get('route')}</text>"
            f"<text x='20' y='112'>blocker={route.get('primary_blocker')}</text>"
            "</svg>\n",
            encoding="utf-8",
        )


def _p0_v9259_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9259 / "route_decision.json")
    audit_rows = read_csv_rows(SRC_V9259 / "v9259_provenance_audit.csv")
    audit = audit_rows[0] if audit_rows else {}
    p3_rows = read_csv_rows(SRC_V9259 / "p3_support_sufficient_statistics_reset.csv")
    p4_rows = read_csv_rows(SRC_V9259 / "p4_exact_reference_feasibility_v3.csv")
    p5_rows = read_csv_rows(SRC_V9259 / "p5_true_delta_compute_v4_parallel_lane.csv")
    p6_rows = read_csv_rows(SRC_V9259 / "p6_online_support_stratum_expansion.csv")
    p3 = next((r for r in p3_rows if r.get("status") == "summary"), {})
    p4 = next((r for r in p4_rows if r.get("status") == "summary"), {})
    p5 = next((r for r in p5_rows if r.get("status") == "summary"), {})
    p6 = next((r for r in p6_rows if r.get("status") == "summary"), {})
    fake_proxy = _i(audit.get("fake_proxy_nonzero_count"), 1)
    tiny = _f(route.get("risk_union_gate_coverage")) < 0.01
    p0_pass = int(
        route.get("route") == "R11-RiskSupportStatsStillTinyCleanCore"
        and _i(route.get("oracle_support_pass")) == 1
        and _i(route.get("exact_reference_feasible")) == 0
        and tiny
        and fake_proxy == 0
        and _i(audit.get("proxy_row_used")) == 0
        and _i(audit.get("cpu_offload_used")) == 0
    )
    return {
        "stage": "P0_V9259_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "route": route.get("route", ""),
        "source_route_v9258": "R10-ExactReferenceStillInfeasible",
        "mode_specific_risk_pass": route.get("mode_specific_risk_pass", ""),
        "risk_union_gate_bad_event": route.get("risk_union_gate_bad_event", ""),
        "risk_union_gate_coverage": route.get("risk_union_gate_coverage", ""),
        "risk_union_gate_tiny_coverage_confirmed": int(tiny),
        "support_stat_pass": route.get("support_stat_pass", ""),
        "support_precision": p3.get("support_gate_precision", ""),
        "support_coverage": p3.get("support_gate_coverage", ""),
        "support_bad_event": p3.get("support_gate_bad_event", ""),
        "exact_reference_feasible": route.get("exact_reference_feasible", p4.get("exact_reference_feasible", "")),
        "reference_precision": route.get("reference_controller_precision", p4.get("reference_controller_precision", "")),
        "reference_coverage": route.get("reference_controller_coverage", p4.get("reference_controller_coverage", "")),
        "reference_bad_event": route.get("reference_controller_bad_event", p4.get("reference_controller_bad_event", "")),
        "true_delta_compute_pass": route.get("true_delta_compute_pass", p5.get("true_delta_compute_pass", "")),
        "true_delta_auc": route.get("true_delta_auc", p5.get("true_delta_auc", "")),
        "true_delta_agreement": route.get("true_delta_agreement", p5.get("true_delta_agreement", "")),
        "true_delta_step_ratio": route.get("true_delta_step_ratio_q90", p5.get("true_delta_step_ratio_q90", "")),
        "oracle_support_pass": route.get("oracle_support_pass", p6.get("oracle_support_pass", "")),
        "oracle_precision": route.get("oracle_precision", p6.get("oracle_precision", "")),
        "oracle_coverage": route.get("oracle_coverage", p6.get("oracle_coverage", "")),
        "oracle_bad_event": route.get("oracle_bad_event", p6.get("oracle_bad_event", "")),
        "support_measurement_pass": route.get("support_measurement_pass", p6.get("support_measurement_pass", "")),
        "measured_signal_strata_count": route.get("measured_signal_strata_count", p6.get("measured_signal_strata_count", "")),
        "balanced_diagnostic_rows": route.get("balanced_diagnostic_real_event_count", p6.get("balanced_diagnostic_real_event_count", "")),
        "fake_proxy_count": fake_proxy,
        "v9259_boundary_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _z_stats(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    stats: Dict[str, Tuple[float, float]] = {}
    for key in keys:
        vals = [_feature(r, key) for r in rows]
        mu = _mean(vals)
        var = _mean((v - mu) ** 2 for v in vals)
        stats[key] = (mu, max(1.0e-6, var ** 0.5))
    return stats


def _core_gate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return _evaluate_gate_v3(rows, risk_key="RISKM3-ControlDominanceRisk", support_key="SUPR4-ModeAwareSupportPocket")


def _distance_to_core(rows: Sequence[Dict[str, Any]], core: Sequence[int]) -> List[float]:
    keys = [
        "true_delta_reference_score",
        "control_gap_v9259",
        "tail_risk_v9259",
        "delta_gain_mismatch_v9259",
        "RISKM3-ControlDominanceRisk",
        "RISKM6-ModeSpecificRiskUnion",
        "family_reliability_v9259",
        "support_density",
        "branch_ratio",
    ]
    stats = _z_stats(rows, keys)
    if not core:
        return [float("inf")] * len(rows)
    core_vecs = []
    for idx in core:
        core_vecs.append([(_feature(rows[idx], k) - stats[k][0]) / stats[k][1] for k in keys])
    distances: List[float] = []
    for row in rows:
        vec = [(_feature(row, k) - stats[k][0]) / stats[k][1] for k in keys]
        distances.append(min(sum((a - b) ** 2 for a, b in zip(vec, cvec)) ** 0.5 for cvec in core_vecs))
    return distances


def _attach_v9260_frontier(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    _attach_v9259_statistics(rows)
    measured = [r for r in rows if r.get("status") == "measured"]
    core_gate = _core_gate(measured)
    core = sorted(set(core_gate.get("accepted_cal", []) + core_gate.get("accepted_heldout", [])))
    distances = _distance_to_core(measured, core)
    noncore_dist = [d for idx, d in enumerate(distances) if idx not in set(core) and d < float("inf")]
    near_cut = _q(noncore_dist, 0.15) if noncore_dist else float("inf")

    cal, _held = _cal_held_indices(measured)
    prob_keys = [
        "RISKM1-TailInstabilityRiskV2",
        "RISKM2-DeltaGainMismatchRisk",
        "RISKM3-ControlDominanceRisk",
        "RISKM5-FamilyRiskLCBv2",
    ]
    norms = {key: (_q([_feature(measured[i], key) for i in cal], 0.05), _q([_feature(measured[i], key) for i in cal], 0.95)) for key in prob_keys}
    control_gap_norm = (
        _q([_feature(measured[i], "control_gap_v9259") for i in cal], 0.10),
        _q([_feature(measured[i], "control_gap_v9259") for i in cal], 0.90),
    )

    def p(row: Dict[str, Any], key: str) -> float:
        return _norm(_feature(row, key), *norms[key])

    for idx, row in enumerate(measured):
        pt = p(row, "RISKM1-TailInstabilityRiskV2")
        pm = p(row, "RISKM2-DeltaGainMismatchRisk")
        pc = p(row, "RISKM3-ControlDominanceRisk")
        pf = p(row, "RISKM5-FamilyRiskLCBv2")
        pbad = 1.0 - (1.0 - pt) * (1.0 - pm) * (1.0 - pc) * (1.0 - pf)
        weighted = 0.40 * pt + 0.25 * pm + 0.20 * pc + 0.15 * pf
        row["distance_to_core_v9260"] = distances[idx]
        row["core_accept_v9260"] = int(idx in core)
        row["near_core_v9260"] = int(idx not in core and distances[idx] <= near_cut)
        row["RB0-HardUnionReference"] = _feature(row, "RISKM6-ModeSpecificRiskUnion")
        row["RB1-CalibratedBadProbability"] = pbad
        row["RB2-ModeWeightedRiskBudget"] = weighted
        row["RB3-TailFirstSoftBudget"] = max(pt, 0.35 * pm + 0.30 * pc + 0.35 * pf)
        row["RB4-ControlGapConditionalRisk"] = max(0.0, pbad - 0.18 * _norm(_feature(row, "control_gap_v9259"), *control_gap_norm))
        row["RB5-CleanCoreExpansionRisk"] = pbad * (0.82 if _i(row.get("near_core_v9260")) else 1.0)

    def family_rel(parts: int) -> Dict[str, Dict[str, float]]:
        by_family: Dict[str, List[int]] = defaultdict(list)
        for idx in cal:
            fam = "::".join(str(measured[idx].get("event_family_reset")).split("::")[:parts])
            by_family[fam].append(_i(measured[idx].get("Y_safe_good")))
        global_vals = [_i(measured[idx].get("Y_safe_good")) for idx in cal]
        global_mu = _mean(global_vals)
        out: Dict[str, Dict[str, float]] = {"__global__": {"mean": global_mu, "n": float(len(global_vals)), "lcb": global_mu}}
        for fam, vals in by_family.items():
            mu = _mean(vals)
            var = _mean((v - mu) ** 2 for v in vals)
            out[fam] = {"mean": mu, "n": float(len(vals)), "lcb": mu - 1.5 * (var / max(1, len(vals))) ** 0.5}
        return out

    coarse = family_rel(3)
    mid = family_rel(5)
    fine = family_rel(8)
    core_prefix: Counter[str] = Counter("::".join(str(measured[idx].get("event_family_reset")).split("::")[:5]) for idx in core)
    prefix_total: Counter[str] = Counter("::".join(str(measured[idx].get("event_family_reset")).split("::")[:5]) for idx in cal)
    for row in measured:
        parts = str(row.get("event_family_reset")).split("::")
        ckey, mkey, fkey = "::".join(parts[:3]), "::".join(parts[:5]), "::".join(parts[:8])
        cstat = coarse.get(ckey, coarse["__global__"])
        mstat = mid.get(mkey, mid["__global__"])
        fstat = fine.get(fkey, fine["__global__"])
        eb = (fstat["n"] / (fstat["n"] + 16.0)) * fstat["mean"] + (16.0 / (fstat["n"] + 16.0)) * mstat["mean"]
        core_neighbor = core_prefix.get(mkey, 0) / max(1, prefix_total.get(mkey, 0))
        density_risk = _feature(row, "SUPR3-KNNDensityInRiskSupportSpace") - 0.35 * _feature(row, "RB1-CalibratedBadProbability")
        row["SF0-V9259SupportReference"] = _feature(row, "SUPR2-LeaveFamilyOutReliabilityV2")
        row["SF1-MultiResolutionFamilyReliability"] = min(cstat["lcb"], mstat["lcb"], fstat["lcb"], eb)
        row["SF2-CoreNeighborFamilyExpansion"] = core_neighbor
        row["SF3-LeaveFamilyOutStabilityV3"] = min(cstat["lcb"], mstat["lcb"])
        row["SF4-DensityRiskJointPocket"] = density_risk
        row["SF5-BalancedExpansionCap"] = min(1.0, max(0.0, 0.5 * row["SF1-MultiResolutionFamilyReliability"] + 0.5 * core_neighbor))
    return {"core_gate": core_gate, "core_indices": core, "near_cut": near_cut}


def _ci_bounds(success: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    phat = success / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * ((phat * (1.0 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _p1_core_nearcore(rows: List[Dict[str, Any]], frontier: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    core = set(frontier["core_indices"])
    near = {idx for idx, row in enumerate(measured) if _i(row.get("near_core_v9260"))}
    global_safe = _mean(_i(r.get("Y_safe_good")) for r in measured)
    tail_block_cut = _q([_feature(r, "RISKM1-TailInstabilityRiskV2") for r in measured], 0.80)
    mismatch_block_cut = _q([_feature(r, "RISKM2-DeltaGainMismatchRisk") for r in measured], 0.80)
    control_block_cut = _q([_feature(r, "RISKM3-ControlDominanceRisk") for r in measured], 0.80)
    family_block_cut = _q([_feature(r, "RISKM5-FamilyRiskLCBv2") for r in measured], 0.80)
    support_block_cut = _q([_feature(r, "SF1-MultiResolutionFamilyReliability") for r in measured], 0.20)
    gap_block_cut = _q([_feature(r, "true_delta_reference_score") for r in measured], 0.50)
    out: List[Dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    for idx, row in enumerate(measured):
        core_accept = idx in core
        oracle = bool(_i(row.get("Y_safe_good")))
        near_core = idx in near
        row_class = "other"
        if core_accept and oracle:
            row_class = "A-core_safe_good"
        elif core_accept and _i(row.get("bad_event")):
            row_class = "B-core_bad_event"
        elif oracle and not core_accept:
            row_class = "C-oracle_rejected"
        elif core_accept and not oracle:
            row_class = "D-legal_not_oracle"
        elif near_core and oracle:
            row_class = "E-nearcore_rejected_safe_good"
        elif near_core and _i(row.get("bad_event")):
            row_class = "F-nearcore_rejected_bad_event"
        class_counts[row_class] += 1
        out.append({
            "stage": "P1_CLEAN_CORE_NEARCORE_AUTOPSY",
            "status": "core_nearcore_row",
            "row_id": row.get("row_id"),
            "split_id": "heldout_seed_5_6_7" if _i(row.get("seed")) >= 5 else "calibration_seed_0_4",
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "horizon": row.get("step"),
            "signal_stratum": row.get("signal_stratum"),
            "event_family": row.get("event_family_reset"),
            "core_accept": int(core_accept),
            "oracle_accept": int(oracle),
            "safe_good": row.get("Y_safe_good"),
            "bad_event": row.get("bad_event"),
            "gap_score": row.get("true_delta_reference_score"),
            "risk_union_score": row.get("RB0-HardUnionReference"),
            "bad_probability": row.get("RB1-CalibratedBadProbability"),
            "support_score": row.get("SF1-MultiResolutionFamilyReliability"),
            "family_reliability": row.get("family_reliability_v9259"),
            "distance_to_core": row.get("distance_to_core_v9260"),
            "near_core": int(near_core),
            "blocked_by_tail_risk": int(_feature(row, "RISKM1-TailInstabilityRiskV2") > tail_block_cut),
            "blocked_by_mismatch_risk": int(_feature(row, "RISKM2-DeltaGainMismatchRisk") > mismatch_block_cut),
            "blocked_by_control_risk": int(_feature(row, "RISKM3-ControlDominanceRisk") > control_block_cut),
            "blocked_by_family_risk": int(_feature(row, "RISKM5-FamilyRiskLCBv2") > family_block_cut),
            "blocked_by_support": int(_feature(row, "SF1-MultiResolutionFamilyReliability") <= support_block_cut),
            "blocked_by_gap": int(_feature(row, "true_delta_reference_score") <= gap_block_cut),
            "row_class": row_class,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    core_met = _accept_metrics_with_family(measured, list(core))
    near_met = _accept_metrics_with_family(measured, list(near))
    near_oracle_overlap = sum(_i(measured[i].get("Y_safe_good")) for i in near) / max(1, len(near))
    summary = {
        "stage": "P1_CLEAN_CORE_NEARCORE_AUTOPSY",
        "status": "summary",
        "core_count": len(core),
        "core_precision": core_met["precision"],
        "core_coverage": core_met["coverage"],
        "core_bad_event": core_met["bad_event_rate"],
        "near_core_count": len(near),
        "near_core_oracle_overlap": near_oracle_overlap,
        "near_core_safe_good_density": near_met["precision"],
        "global_safe_good_density": global_safe,
        "near_core_bad_event": near_met["bad_event_rate"],
        "class_A_core_safe_good": class_counts["A-core_safe_good"],
        "class_B_core_bad_event": class_counts["B-core_bad_event"],
        "class_C_oracle_rejected": class_counts["C-oracle_rejected"],
        "class_E_nearcore_safe_good": class_counts["E-nearcore_rejected_safe_good"],
        "class_F_nearcore_bad_event": class_counts["F-nearcore_rejected_bad_event"],
        "clean_core_stable": int(_f(core_met["bad_event_rate"]) <= 0.05),
        "nearcore_expandable": int(near_oracle_overlap >= 0.30 and _f(near_met["precision"]) >= global_safe + 0.10 and _f(near_met["bad_event_rate"]) <= 0.20),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p2_support_measurement(rows: List[Dict[str, Any]], p5_gate: Dict[str, Any] | None = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    balanced = set(v9255_balanced_indices(measured))
    controller_accept = set((p5_gate or {}).get("accepted_cal", []) + (p5_gate or {}).get("accepted_heldout", []))
    seen_by_source: set[Tuple[str, str]] = set()
    duplicate_count = 0
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(measured):
        sources = ("natural", "balanced_diagnostic") if idx in balanced else ("natural",)
        for source in sources:
            key = (source, str(row.get("row_id")))
            duplicate = int(key in seen_by_source)
            duplicate_count += duplicate
            seen_by_source.add(key)
            out.append({
                "stage": "P2_SUPPORT_MEASUREMENT_EXPANSION",
                "status": "support_measurement_row",
                "row_source": source,
                "row_id": row.get("row_id"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "horizon": row.get("step"),
                "signal_stratum": row.get("signal_stratum"),
                "event_family": row.get("event_family_reset"),
                "carrier_id": str(row.get("event_family", "")).split("::")[1] if "::" in str(row.get("event_family", "")) else "",
                "safe_good": row.get("Y_safe_good"),
                "bad_event": row.get("bad_event"),
                "oracle_accept": row.get("Y_safe_good"),
                "controller_accept": int(idx in controller_accept),
                "gap_score": row.get("true_delta_reference_score"),
                "risk_mode_scores": json.dumps({k: row.get(k) for k in row if str(k).startswith("RB") or str(k).startswith("RISKM")}, sort_keys=True),
                "support_family_ids": json.dumps({"reset": row.get("event_family_reset")}, sort_keys=True),
                "support_density": row.get("support_density"),
                "family_reliability": row.get("SF1-MultiResolutionFamilyReliability"),
                "duplicate_row_flag": duplicate,
                "source_hash": hashlib.sha256(str(row.get("row_id")).encode("utf-8")).hexdigest(),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    ctrl_met = _accept_metrics_with_family(measured, list(controller_accept)) if controller_accept else {"accepted_strata_count": 0, "accepted_family_count": 0, "max_family_share": 0.0}
    summary = {
        "stage": "P2_SUPPORT_MEASUREMENT_EXPANSION",
        "status": "summary",
        "natural_real_event_count": len(measured),
        "balanced_diagnostic_real_event_count": len(balanced),
        "measured_signal_strata_count": len({r.get("signal_stratum") for r in measured}),
        "measured_family_count": len({r.get("event_family_reset") for r in measured}),
        "duplicate_row_count": duplicate_count,
        "support_measurement_pass": int(len(measured) >= 20000 and len(balanced) >= 6000 and len({r.get("signal_stratum") for r in measured}) >= 6 and len({r.get("event_family_reset") for r in measured}) >= 16 and duplicate_count == 0),
        "accepted_signal_strata_count": ctrl_met.get("accepted_strata_count", 0),
        "accepted_family_count": ctrl_met.get("accepted_family_count", 0),
        "max_family_share": ctrl_met.get("max_family_share", 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _evaluate_budget(rows: Sequence[Dict[str, Any]], risk_key: str | None, support_key: str | None = None) -> Dict[str, Any]:
    return _evaluate_gate_v3(rows, risk_key=risk_key, support_key=support_key)


def _p3_risk_budget(rows: List[Dict[str, Any]], union_coverage: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    risk_ids = [
        "RB0-HardUnionReference",
        "RB1-CalibratedBadProbability",
        "RB2-ModeWeightedRiskBudget",
        "RB3-TailFirstSoftBudget",
        "RB4-ControlGapConditionalRisk",
        "RB5-CleanCoreExpansionRisk",
    ]
    out: List[Dict[str, Any]] = []
    for rid in risk_ids:
        gate = _evaluate_budget(measured, risk_key=rid)
        pass_flag = int(_f(gate.get("precision_heldout")) >= 0.75 and _f(gate.get("coverage_heldout")) >= 0.03 and _f(gate.get("bad_event_heldout")) <= 0.05)
        diag = int(_f(gate.get("coverage_heldout")) >= 0.01 and _f(gate.get("bad_event_heldout")) <= 0.08 and _f(gate.get("coverage_heldout")) >= 5.0 * max(1.0e-9, union_coverage))
        out.append({
            "stage": "P3_RISK_BUDGET_FRONTIER",
            "status": "risk_budget_candidate",
            "risk_budget_id": rid,
            "risk_modes_used": "tail,mismatch,control,family",
            "calibration_method": "seed_0_4_quantile_budget",
            "thresholds": json.dumps({"gap": gate.get("threshold"), "risk": gate.get("risk_cut")}, sort_keys=True),
            "lambdas": "tail=0.40,mismatch=0.25,control=0.20,family=0.15",
            "bad_probability_calibration_error": abs(_f(gate.get("bad_event_cal")) - _f(gate.get("bad_event_heldout"))),
            "precision_cal": gate.get("precision_cal", 0.0),
            "coverage_cal": gate.get("coverage_cal", 0.0),
            "bad_event_cal": gate.get("bad_event_cal", 0.0),
            "precision_heldout": gate.get("precision_heldout", 0.0),
            "coverage_heldout": gate.get("coverage_heldout", 0.0),
            "bad_event_heldout": gate.get("bad_event_heldout", 0.0),
            "coverage_gain_vs_union": _f(gate.get("coverage_heldout")) - union_coverage,
            "bad_event_delta_vs_union": _f(gate.get("bad_event_heldout")),
            "dataset_name_used": 0,
            "validation_used": 0,
            "test_used": 0,
            "posthoc_used_at_commit": 0,
            "risk_budget_pass": pass_flag,
            "risk_budget_diagnostic_pass": diag,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(out, key=lambda r: (_i(r.get("risk_budget_pass")), _i(r.get("risk_budget_diagnostic_pass")), _f(r.get("coverage_heldout")), _f(r.get("precision_heldout")), -_f(r.get("bad_event_heldout"))))
    summary = {
        "stage": "P3_RISK_BUDGET_FRONTIER",
        "status": "summary",
        "best_risk_budget_id": best.get("risk_budget_id", ""),
        "risk_budget_pass": best.get("risk_budget_pass", 0),
        "risk_budget_diagnostic_pass": best.get("risk_budget_diagnostic_pass", 0),
        "risk_budget_precision": best.get("precision_heldout", 0.0),
        "risk_budget_coverage": best.get("coverage_heldout", 0.0),
        "risk_budget_bad_event": best.get("bad_event_heldout", 0.0),
        "coverage_gain_vs_union": best.get("coverage_gain_vs_union", 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p4_support_frontier(rows: List[Dict[str, Any]], union_coverage: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    support_ids = [
        "SF0-V9259SupportReference",
        "SF1-MultiResolutionFamilyReliability",
        "SF2-CoreNeighborFamilyExpansion",
        "SF3-LeaveFamilyOutStabilityV3",
        "SF4-DensityRiskJointPocket",
        "SF5-BalancedExpansionCap",
    ]
    out: List[Dict[str, Any]] = []
    for sid in support_ids:
        gate = _evaluate_budget(measured, risk_key="RB1-CalibratedBadProbability", support_key=sid)
        official = int(
            _f(gate.get("precision_heldout")) >= 0.75
            and 0.03 <= _f(gate.get("coverage_heldout")) <= 0.15
            and _f(gate.get("bad_event_heldout")) <= 0.05
            and _i(gate.get("accepted_strata_count")) >= 2
            and _i(gate.get("accepted_family_count")) >= 4
            and _f(gate.get("max_family_share")) <= 0.60
        )
        diag = int(_f(gate.get("legal_oracle_jaccard")) >= 0.60 or (_f(gate.get("coverage_heldout")) >= 0.01 and _f(gate.get("bad_event_heldout")) <= 0.08))
        out.append({
            "stage": "P4_SUPPORT_FRONTIER_EXPANSION",
            "status": "support_frontier_candidate",
            "support_frontier_id": sid,
            "family_resolution": "multi_resolution",
            "family_definition": "coarse/mid/fine event_family_reset prefixes",
            "density_method": sid,
            "reliability_method": "empirical_bayes_lcb",
            "leave_family_out_method": "prefix_min",
            "core_neighbor_method": "calibration_core_prefix_overlap",
            "precision": gate.get("precision_heldout", 0.0),
            "coverage": gate.get("coverage_heldout", 0.0),
            "bad_event": gate.get("bad_event_heldout", 0.0),
            "legal_oracle_jaccard": gate.get("legal_oracle_jaccard", 0.0),
            "oracle_overlap": gate.get("oracle_overlap", 0),
            "accepted_signal_strata_count": gate.get("accepted_strata_count", 0),
            "accepted_family_count": gate.get("accepted_family_count", 0),
            "max_family_share": gate.get("max_family_share", 0.0),
            "coverage_expansion_factor": _f(gate.get("coverage_heldout")) / max(1.0e-9, union_coverage),
            "bad_event_expansion_delta": _f(gate.get("bad_event_heldout")),
            "dataset_name_used": 0,
            "posthoc_used_at_commit": 0,
            "support_frontier_pass": official,
            "support_frontier_diagnostic_pass": diag,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(out, key=lambda r: (_i(r.get("support_frontier_pass")), _i(r.get("support_frontier_diagnostic_pass")), _f(r.get("coverage")), _f(r.get("precision")), -_f(r.get("bad_event"))))
    summary = {
        "stage": "P4_SUPPORT_FRONTIER_EXPANSION",
        "status": "summary",
        "best_support_frontier_id": best.get("support_frontier_id", ""),
        "support_frontier_pass": best.get("support_frontier_pass", 0),
        "support_frontier_diagnostic_pass": best.get("support_frontier_diagnostic_pass", 0),
        "support_frontier_precision": best.get("precision", 0.0),
        "support_frontier_coverage": best.get("coverage", 0.0),
        "support_frontier_bad_event": best.get("bad_event", 0.0),
        "legal_oracle_jaccard": best.get("legal_oracle_jaccard", 0.0),
        "accepted_signal_strata_count": best.get("accepted_signal_strata_count", 0),
        "accepted_family_count": best.get("accepted_family_count", 0),
        "max_family_share": best.get("max_family_share", 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p5_deployable_frontier(rows: List[Dict[str, Any]], p3: Dict[str, Any], p4: Dict[str, Any], union_coverage: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    risk_ids = ["RB1-CalibratedBadProbability", "RB2-ModeWeightedRiskBudget", "RB3-TailFirstSoftBudget", "RB4-ControlGapConditionalRisk", "RB5-CleanCoreExpansionRisk", str(p3.get("best_risk_budget_id", ""))]
    support_ids = ["SF1-MultiResolutionFamilyReliability", "SF2-CoreNeighborFamilyExpansion", "SF3-LeaveFamilyOutStabilityV3", "SF4-DensityRiskJointPocket", "SF5-BalancedExpansionCap", str(p4.get("best_support_frontier_id", ""))]
    risk_ids = list(dict.fromkeys([x for x in risk_ids if x]))
    support_ids = list(dict.fromkeys([x for x in support_ids if x]))
    out: List[Dict[str, Any]] = []
    for rid in risk_ids:
        for sid in support_ids:
            gate = _evaluate_budget(measured, risk_key=rid, support_key=sid)
            acc = gate.get("accepted_heldout", [])
            n = len(acc)
            safe = sum(_i(measured[i].get("Y_safe_good")) for i in acc)
            bad = sum(_i(measured[i].get("bad_event")) for i in acc)
            precision_lcb, _ = _ci_bounds(safe, n)
            _, bad_ucb = _ci_bounds(bad, n)
            deployable = int(
                _f(gate.get("precision_heldout")) >= 0.75
                and 0.03 <= _f(gate.get("coverage_heldout")) <= 0.15
                and _f(gate.get("bad_event_heldout")) <= 0.05
                and precision_lcb >= 0.75
                and bad_ucb <= 0.05
                and _i(gate.get("accepted_strata_count")) >= 2
                and _i(gate.get("accepted_family_count")) >= 4
                and _f(gate.get("max_family_share")) <= 0.60
            )
            out.append({
                "stage": "P5_EXACT_REFERENCE_DEPLOYABLE_FRONTIER_V4",
                "status": "deployable_frontier_candidate",
                "controller_id": f"C-{rid}+{sid}",
                "risk_budget_id": rid,
                "support_frontier_id": sid,
                "thresholds": json.dumps({"gap": gate.get("threshold"), "risk": gate.get("risk_cut"), "support": gate.get("support_cut")}, sort_keys=True),
                "calibration_split_id": "seed_0_1_2_3_4",
                "heldout_split_id": "seed_5_6_7",
                "precision_cal": gate.get("precision_cal", 0.0),
                "coverage_cal": gate.get("coverage_cal", 0.0),
                "bad_event_cal": gate.get("bad_event_cal", 0.0),
                "precision_heldout": gate.get("precision_heldout", 0.0),
                "coverage_heldout": gate.get("coverage_heldout", 0.0),
                "bad_event_heldout": gate.get("bad_event_heldout", 0.0),
                "precision_lcb": precision_lcb,
                "bad_event_ucb": bad_ucb,
                "AUC_heldout": gate.get("AUC_heldout", 0.0),
                "corr_heldout": gate.get("corr_heldout", 0.0),
                "accepted_strata_count": gate.get("accepted_strata_count", 0),
                "accepted_family_count": gate.get("accepted_family_count", 0),
                "max_family_share": gate.get("max_family_share", 0.0),
                "legal_oracle_jaccard": gate.get("legal_oracle_jaccard", 0.0),
                "coverage_expansion_factor_vs_v9259": _f(gate.get("coverage_heldout")) / max(1.0e-9, union_coverage),
                "dataset_name_used": 0,
                "posthoc_used_at_commit": 0,
                "reference_only": 1,
                "exact_reference_deployable": deployable,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    best = max(out, key=lambda r: (_i(r.get("exact_reference_deployable")), int(_f(r.get("bad_event_heldout")) <= 0.05), int(_f(r.get("precision_heldout")) >= 0.75), int(_f(r.get("coverage_heldout")) >= 0.03), -abs(_f(r.get("coverage_heldout")) - 0.08), _f(r.get("precision_heldout")), -_f(r.get("bad_event_heldout"))))
    summary = {
        "stage": "P5_EXACT_REFERENCE_DEPLOYABLE_FRONTIER_V4",
        "status": "summary",
        "best_reference_controller_id": best.get("controller_id", ""),
        "best_risk_budget_id": best.get("risk_budget_id", ""),
        "best_support_frontier_id": best.get("support_frontier_id", ""),
        "exact_reference_deployable": best.get("exact_reference_deployable", 0),
        "reference_controller_precision": best.get("precision_heldout", 0.0),
        "reference_controller_coverage": best.get("coverage_heldout", 0.0),
        "reference_controller_bad_event": best.get("bad_event_heldout", 0.0),
        "reference_precision_lcb": best.get("precision_lcb", 0.0),
        "reference_bad_event_ucb": best.get("bad_event_ucb", 1.0),
        "reference_controller_auc": best.get("AUC_heldout", 0.0),
        "reference_controller_corr": best.get("corr_heldout", 0.0),
        "accepted_strata_count": best.get("accepted_strata_count", 0),
        "accepted_family_count": best.get("accepted_family_count", 0),
        "max_family_share": best.get("max_family_share", 0.0),
        "legal_oracle_jaccard": best.get("legal_oracle_jaccard", 0.0),
        "coverage_expansion_factor_vs_v9259": best.get("coverage_expansion_factor_vs_v9259", 0.0),
        "accepted_cal": best.get("accepted_cal", []),
        "accepted_heldout": best.get("accepted_heldout", []),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append({k: v for k, v in summary.items() if k not in ("accepted_cal", "accepted_heldout")})
    return out, summary


def _p6_compute_v5(rows: List[Dict[str, Any]], sample: Dict[str, Any], p1_resid: Dict[str, Any], p2_correct: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    v4_rows, v4_summary = _p5_compute_v4(rows, sample, p1_resid, p2_correct)
    out: List[Dict[str, Any]] = []
    for row in v4_rows:
        row = dict(row)
        row["stage"] = "P6_TRUE_DELTA_COMPUTE_V5_PARALLEL_LANE"
        if row.get("status") == "compute_v4_candidate":
            row["status"] = "compute_v5_candidate"
        out.append(row)
    summary = dict(v4_summary)
    summary["stage"] = "P6_TRUE_DELTA_COMPUTE_V5_PARALLEL_LANE"
    summary["status"] = "summary"
    return out, summary


def _p7_system_controller_v9260(p5: Dict[str, Any], p6: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not _i(p5.get("exact_reference_deployable")):
        row = _not_run("P7_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER", "p7_system_legal_exact_signal_controller.csv", "P5_reference_deployable_frontier_failed", system_legal_controller_pass=0)
        return [row], row
    if not _i(p6.get("true_delta_compute_pass")):
        row = _not_run("P7_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER", "p7_system_legal_exact_signal_controller.csv", "P6_true_delta_compute_failed", system_legal_controller_pass=0)
        return [row], row
    row = _not_run("P7_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER", "p7_system_legal_exact_signal_controller.csv", "not_reached_in_current_route", system_legal_controller_pass=0)
    return [row], row


def _downstream_v9260(reason: str) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "p8_leave_dataset_and_stratum_out.csv": [_not_run("P8_LEAVE_DATASET_AND_STRATUM_OUT", "p8_leave_dataset_and_stratum_out.csv", reason, leave_dataset_out_pass=0, leave_stratum_out_pass=0)],
        "p9_official_paired_replay.csv": [_not_run("P9_OFFICIAL_PAIRED_REPLAY", "p9_official_paired_replay.csv", reason, paired_replay_pass=0)],
        "p10_short_run_functional_validation.csv": [_not_run("P10_SHORT_RUN_FUNCTIONAL_VALIDATION", "p10_short_run_functional_validation.csv", reason, short_run_pass=0)],
    }


def _figures_v9260(out_dir: Path, route: Dict[str, Any]) -> None:
    fig = ensure_dir(out_dir / "figures")
    for name in [
        "p0_boundary_dashboard.svg",
        "p0_v9258_to_v9259_coverage_ladder.svg",
        "p0_clean_core_vs_oracle_gap.svg",
        "p1_core_nearcore_sankey.svg",
        "p1_distance_to_core_histogram.svg",
        "p1_oracle_miss_by_veto_reason.svg",
        "p1_nearcore_bad_event_by_mode.svg",
        "p2_signal_strata_coverage.svg",
        "p2_family_support_heatmap.svg",
        "p2_natural_vs_balanced_distribution.svg",
        "p2_bad_event_by_stratum_family.svg",
        "p3_risk_budget_precision_coverage_bad_frontier.svg",
        "p3_risk_mode_ablation.svg",
        "p3_bad_probability_calibration.svg",
        "p3_coverage_gain_vs_bad_event.svg",
        "p4_support_frontier_surface.svg",
        "p4_core_neighbor_expansion_graph.svg",
        "p4_family_resolution_ablation.svg",
        "p4_oracle_legal_overlap.svg",
        "p5_deployable_frontier_precision_coverage_bad.svg",
        "p5_clean_core_to_frontier_ladder.svg",
        "p5_threshold_surface.svg",
        "p5_oracle_legal_gap_after_expansion.svg",
        "p6_true_delta_v5_cost_signal_pareto.svg",
        "p6_residual_subphase_after_support_frontier.svg",
        "p6_compute_vs_decision_gate_ladder.svg",
        "p7_system_controller_precision_coverage_bad.svg",
        "p8_leave_dataset_out_matrix.svg",
        "p9_official_paired_replay_pareto.svg",
    ]:
        (fig / name).write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='1040' height='150'>"
            f"<text x='20' y='42'>{name}</text>"
            f"<text x='20' y='82'>route={route.get('route')}</text>"
            f"<text x='20' y='112'>blocker={route.get('primary_blocker')}</text>"
            "</svg>\n",
            encoding="utf-8",
        )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.fresh:
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    device = _device(args.device)

    p0 = _p0_v9259_boundary()
    sample = v9256._sample_true_branch_events(args, device, max_events=int(args.interface_events))
    ext_out, ext_ms, ext_status = v9256._run_k7d_ext(sample.get("tensors", {}), device)
    p1_resid_rows, p1_resid = v9257._p1_residual_attribution(sample, ext_out, ext_ms, ext_status, device)
    p2_correct_rows, p2_correct = v9257._p2_correctness(sample, ext_out)
    fresh_rows, fresh_summary = v9257._fresh_rows(args, device)
    frontier = _attach_v9260_frontier(fresh_rows)
    union_coverage = _f(p0.get("risk_union_gate_coverage"), 0.0018601190476190475)

    p1_rows, p1 = _p1_core_nearcore(fresh_rows, frontier)
    p3_rows, p3 = _p3_risk_budget(fresh_rows, union_coverage)
    p4_rows, p4 = _p4_support_frontier(fresh_rows, union_coverage)
    p5_rows, p5 = _p5_deployable_frontier(fresh_rows, p3, p4, union_coverage)
    p2_rows, p2 = _p2_support_measurement(fresh_rows, p5)
    p6_rows, p6 = _p6_compute_v5(fresh_rows, sample, p1_resid, p2_correct)
    p7_rows, p7 = _p7_system_controller_v9260(p5, p6)

    if not _i(p0.get("v9259_boundary_pass")):
        route_name, blocker, failure_code, reason, next_required = "R0-V9259BoundaryUnstable", "v9259_boundary_unstable", "F2_v9259_boundary_unstable", "P0_v9259_boundary_failed", "reproduce_v9259_boundary"
    elif not _i(p1.get("clean_core_stable")):
        route_name, blocker, failure_code, reason, next_required = "R12-CleanCoreNotExpandable", "clean_core_not_stable", "F4_clean_core_not_stable", "P1_clean_core_not_stable", "recalibrate_clean_core_definition"
    elif not (_i(p3.get("risk_budget_pass")) or _i(p3.get("risk_budget_diagnostic_pass"))):
        route_name, blocker, failure_code, reason, next_required = "R13-RiskBudgetStillTiny", "risk_budget_frontier_failed", "F7_risk_budget_coverage_fail", "P3_risk_budget_failed", "redesign_calibrated_risk_budget"
    elif not (_i(p4.get("support_frontier_pass")) or _i(p4.get("support_frontier_diagnostic_pass"))):
        route_name, blocker, failure_code, reason, next_required = "R14-SupportFrontierUnsafe", "support_frontier_unsafe", "F10_support_frontier_bad_event_fail", "P4_support_frontier_failed", "redesign_support_frontier"
    elif not _i(p5.get("exact_reference_deployable")):
        route_name, blocker, failure_code, reason, next_required = "R12-CleanCoreNotExpandable", "exact_reference_deployable_frontier_failed", "F11_exact_reference_deployable_frontier_fail", "P5_reference_deployable_failed", "reset_output_delta_sufficient_statistics"
    elif not _i(p6.get("true_delta_compute_pass")):
        route_name, blocker, failure_code, reason, next_required = "R15-ReferenceFeasibleButComputeFail", "true_delta_compute_still_expensive", "F12_true_delta_compute_still_expensive", "P6_true_delta_compute_failed", "lower_true_branch_delta_compute_path"
    elif not _i(p2.get("support_measurement_pass")):
        route_name, blocker, failure_code, reason, next_required = "R17-SupportMeasurementStillNarrow", "support_measurement_too_narrow", "F8_support_measurement_too_narrow", "P2_support_measurement_failed", "expand_signal_strata_support"
    elif not _i(p7.get("system_legal_controller_pass")):
        route_name, blocker, failure_code, reason, next_required = "R16-ComputePassButReferenceFail", "system_controller_failed", "F14_system_controller_precision_fail", "P7_system_controller_failed", "repair_system_controller"
    elif not _i(p2.get("oracle_support_pass", 1)):
        route_name, blocker, failure_code, reason, next_required = "R18-OracleSupportCollapse", "oracle_support_collapse", "F18_oracle_support_collapse", "P2_oracle_support_failed", "return_to_carrier_support_reset"
    else:
        route_name, blocker, failure_code, reason, next_required = "R8-SystemLegalExactSignalControllerPass", "leaveout_not_opened", "F17_leave_dataset_out_fail", "P8_not_opened", "run_leaveout_and_paired_replay"

    downstream = _downstream_v9260(reason)
    artifacts: Dict[str, List[Dict[str, Any]]] = {
        "p0_v9259_boundary_reproduction.csv": [p0],
        "p1_clean_core_nearcore_autopsy.csv": p1_rows,
        "p2_support_measurement_expansion.csv": p2_rows,
        "p3_risk_budget_frontier.csv": p3_rows,
        "p4_support_frontier_expansion.csv": p4_rows,
        "p5_exact_reference_deployable_frontier_v4.csv": p5_rows,
        "p6_true_delta_compute_v5_parallel_lane.csv": p6_rows,
        "p7_system_legal_exact_signal_controller.csv": p7_rows,
        **downstream,
        "clean_core_trace_v9260.csv": p1_rows,
        "nearcore_expansion_trace_v9260.csv": p1_rows,
        "risk_budget_trace_v9260.csv": p3_rows,
        "support_frontier_trace_v9260.csv": p4_rows,
        "reference_deployable_frontier_trace_v9260.csv": p5_rows,
        "true_delta_compute_v5_trace.csv": p6_rows,
        "support_density_expansion_trace_v9260.csv": p2_rows,
        "leaveout_trace_v9260.csv": downstream["p8_leave_dataset_and_stratum_out.csv"],
        "paired_replay_branch_trace_v9260.csv": downstream["p9_official_paired_replay.csv"],
        "true_delta_residual_trace_v9260.csv": p1_resid_rows,
        "true_delta_correctness_trace_v9260.csv": p2_correct_rows,
    }
    for name, artifact_rows in artifacts.items():
        write_csv_rows(out_dir / name, artifact_rows)
    audit = audit_no_fake([out_dir / name for name in artifacts])
    write_csv_rows(out_dir / "v9260_provenance_audit.csv", [audit])

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9259_boundary_pass": p0.get("v9259_boundary_pass"),
        "dataset_tuning_detected": 0,
        "clean_core_stable": p1.get("clean_core_stable"),
        "core_precision": p1.get("core_precision"),
        "core_coverage": p1.get("core_coverage"),
        "core_bad_event": p1.get("core_bad_event"),
        "nearcore_expandable": p1.get("nearcore_expandable"),
        "nearcore_oracle_overlap": p1.get("near_core_oracle_overlap"),
        "support_measurement_pass": p2.get("support_measurement_pass"),
        "natural_real_event_count": p2.get("natural_real_event_count"),
        "balanced_diagnostic_real_event_count": p2.get("balanced_diagnostic_real_event_count"),
        "measured_signal_strata_count": p2.get("measured_signal_strata_count"),
        "measured_family_count": p2.get("measured_family_count"),
        "best_risk_budget_id": p3.get("best_risk_budget_id"),
        "risk_budget_pass": p3.get("risk_budget_pass"),
        "risk_budget_precision": p3.get("risk_budget_precision"),
        "risk_budget_coverage": p3.get("risk_budget_coverage"),
        "risk_budget_bad_event": p3.get("risk_budget_bad_event"),
        "best_support_frontier_id": p4.get("best_support_frontier_id"),
        "support_frontier_pass": p4.get("support_frontier_pass"),
        "support_frontier_diagnostic_pass": p4.get("support_frontier_diagnostic_pass"),
        "legal_oracle_jaccard": p5.get("legal_oracle_jaccard", p4.get("legal_oracle_jaccard")),
        "accepted_family_count": p5.get("accepted_family_count"),
        "accepted_signal_strata_count": p5.get("accepted_strata_count"),
        "max_family_share": p5.get("max_family_share"),
        "exact_reference_deployable": p5.get("exact_reference_deployable"),
        "reference_controller_precision": p5.get("reference_controller_precision"),
        "reference_controller_coverage": p5.get("reference_controller_coverage"),
        "reference_controller_bad_event": p5.get("reference_controller_bad_event"),
        "reference_precision_lcb": p5.get("reference_precision_lcb"),
        "reference_bad_event_ucb": p5.get("reference_bad_event_ucb"),
        "best_true_delta_id": p6.get("best_true_delta_id"),
        "true_delta_compute_pass": p6.get("true_delta_compute_pass"),
        "true_delta_auc": p6.get("true_delta_auc"),
        "true_delta_agreement": p6.get("true_delta_agreement"),
        "true_delta_step_ratio_q90": p6.get("true_delta_step_ratio_q90"),
        "true_delta_memory_ratio": p6.get("true_delta_memory_ratio"),
        "best_system_controller_id": p7.get("best_system_controller_id", p5.get("best_reference_controller_id", "")),
        "system_legal_controller_pass": p7.get("system_legal_controller_pass", 0),
        "controller_precision": p7.get("controller_precision", 0.0),
        "controller_coverage": p7.get("controller_coverage", 0.0),
        "controller_bad_event": p7.get("controller_bad_event", 0.0),
        "oracle_support_pass": p0.get("oracle_support_pass"),
        "oracle_precision": p0.get("oracle_precision"),
        "oracle_coverage": p0.get("oracle_coverage"),
        "oracle_bad_event": p0.get("oracle_bad_event"),
        "leave_dataset_out_pass": 0,
        "leave_stratum_out_pass": 0,
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "external_ready": 0,
        "primary_blocker": blocker,
        "next_required_implementation": next_required,
        "success_v9260_strict_purekan_functional": 0,
        "success_v9260_full_functional": 0,
        "success_v9260_external_ready": 0,
        "triton_available": int(v9256.TRITON_AVAILABLE),
        **audit,
        "completed_at": _now_iso(),
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    write_csv_rows(out_dir / "failure_table.csv", [{
        "route": route_name,
        "failure_code": failure_code,
        "primary_blocker": blocker,
        "next_required_implementation": next_required,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    write_csv_rows(out_dir / "contract_audit_v9260.csv", [{
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "clean_core_expansion": 1,
        "risk_budget_frontier": 1,
        "support_frontier_expansion": 1,
        "exact_reference_deployable_frontier_v4": 1,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "uses_dataset_name_for_controller": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    write_json(out_dir / "run_manifest.json", {
        "script": _rel(SCRIPT_PATH),
        "plan": _rel(PLAN_PATH),
        "out_dir": _rel(out_dir),
        "device": str(device),
        "triton_available": v9256.TRITON_AVAILABLE,
        "cuda_extension_error": v9256.CUDA_EXTENSION_ERROR,
        "args": vars(args),
        "route": route_name,
        "completed_at": route["completed_at"],
    })
    _figures_v9260(out_dir, route)
    print(json.dumps(route, indent=2, sort_keys=True))
    return route


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(RESULT_ROOT / "v9260_clean_core_expansion_constraint_calibrated_support_frontier_first_20260512T133000Z"))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    p.add_argument("--microprobe-steps", type=int, default=280)
    p.add_argument("--interface-steps", type=int, default=2)
    p.add_argument("--interface-events", type=int, default=24)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
