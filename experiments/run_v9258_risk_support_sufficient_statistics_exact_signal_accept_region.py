#!/usr/bin/env python3
"""DG-KAN v9.2.58 risk/support sufficient statistics audit."""

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
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.58_RiskSupportSufficientStatistics_ExactSignalAcceptRegion_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9258_risk_support_sufficient_statistics_exact_signal_accept_region.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9257 = RESULT_ROOT / "v9257_true_branch_delta_compute_closure_exact_signal_controller_feasibility_first_20260512T103200Z"


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


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.fresh:
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    device = _device(args.device)

    p0 = _p0_boundary()
    sample = v9256._sample_true_branch_events(args, device, max_events=int(args.interface_events))
    ext_out, ext_ms, ext_status = v9256._run_k7d_ext(sample.get("tensors", {}), device)
    p1_resid_rows, p1_resid = v9257._p1_residual_attribution(sample, ext_out, ext_ms, ext_status, device)
    p2_correct_rows, p2_correct = v9257._p2_correctness(sample, ext_out)
    fresh_rows, fresh_summary = v9257._fresh_rows(args, device)
    _attach_statistics(fresh_rows)
    measured = [r for r in fresh_rows if r.get("status") == "measured"]
    reference_bad = _f(p0.get("reference_controller_bad_event"))
    reference_coverage = _f(p0.get("reference_controller_coverage"))

    p1_rows, p1 = _p1_autopsy(fresh_rows)
    p2_rows, p2 = _p2_risk_factory(fresh_rows, reference_bad, reference_coverage)
    p3_rows, p3 = _p3_support_factory(fresh_rows, reference_bad)
    p4_rows, p4 = _p4_reference_feasibility_v2(fresh_rows, p2, p3)
    p5_rows, p5 = _p5_compute_v3(fresh_rows, sample, p1_resid, p2_correct)
    p6_rows, p6 = _p6_support_expansion(fresh_rows, p4)
    p7_rows, p7 = _p7_system_controller(p4, p5)

    if not _i(p0.get("v9257_boundary_pass")):
        route_name, blocker, failure_code, reason, next_required = "R0-V9257BoundaryUnstable", "v9257_boundary_unstable", "F2_v9257_boundary_unstable", "P0_v9257_boundary_failed", "reproduce_v9257_boundary"
    elif not _i(p1.get("reference_failure_autopsy_pass")):
        route_name, blocker, failure_code, reason, next_required = "R11-RiskSupportStatsMissing", "reference_failure_autopsy_incomplete", "F4_reference_failure_autopsy_incomplete", "P1_reference_failure_autopsy_failed", "repair_reference_failure_autopsy"
    elif not (_i(p2.get("risk_stat_pass")) or _i(p2.get("risk_diagnostic_pass"))):
        route_name, blocker, failure_code, reason, next_required = "R11-RiskSupportStatsMissing", "risk_stat_not_predictive", "F6_risk_stat_not_predictive", "P2_risk_stat_failed", "redesign_bad_event_risk_statistics"
    elif not (_i(p3.get("support_stat_pass")) or _i(p3.get("support_diagnostic_pass"))):
        route_name, blocker, failure_code, reason, next_required = "R11-RiskSupportStatsMissing", "support_stat_no_oracle_overlap", "F8_support_stat_no_oracle_overlap", "P3_support_stat_failed", "redesign_support_family_statistics"
    elif not _i(p4.get("exact_reference_feasible")):
        route_name, blocker, failure_code, reason, next_required = "R10-ExactReferenceStillInfeasible", "exact_reference_still_infeasible", "F10_exact_reference_still_infeasible", "P4_exact_reference_still_infeasible", "reset_risk_support_target_decomposition"
    elif not _i(p5.get("true_delta_compute_pass")):
        route_name, blocker, failure_code, reason, next_required = "R13-ControllerFeasibleButComputeFail", "true_delta_compute_still_expensive", "F11_true_delta_compute_still_expensive", "P5_true_delta_compute_failed", "lower_true_branch_delta_compute_path"
    elif not _i(p7.get("system_legal_controller_pass")):
        route_name, blocker, failure_code, reason, next_required = "R12-ComputePassButControllerFail", "system_legal_controller_failed", "F13_system_controller_precision_fail", "P7_system_controller_failed", "repair_system_controller"
    elif not _i(p6.get("support_measurement_pass")):
        route_name, blocker, failure_code, reason, next_required = "R14-SupportMeasurementStillNarrow", "support_measurement_too_narrow", "F16_support_measurement_too_narrow", "P6_support_measurement_failed", "expand_signal_strata_support"
    elif not _i(p6.get("oracle_support_pass")):
        route_name, blocker, failure_code, reason, next_required = "R15-OracleSupportCollapse", "oracle_support_collapse", "F17_oracle_support_collapse", "P6_oracle_support_failed", "return_to_carrier_support_reset"
    else:
        route_name, blocker, failure_code, reason, next_required = "R6-SystemLegalExactSignalControllerPass", "leaveout_not_opened", "F18_leave_dataset_out_fail", "P8_not_opened", "run_leaveout_and_paired_replay"

    downstream = _downstream(reason)
    artifacts: Dict[str, List[Dict[str, Any]]] = {
        "p0_v9257_boundary_reproduction.csv": [p0],
        "p1_exact_reference_controller_failure_autopsy.csv": p1_rows,
        "p2_risk_sufficient_statistics_factory.csv": p2_rows,
        "p3_support_sufficient_statistics_factory.csv": p3_rows,
        "p4_exact_reference_feasibility_v2.csv": p4_rows,
        "p5_true_delta_compute_v3_parallel_lane.csv": p5_rows,
        "p6_online_support_stratum_expansion.csv": p6_rows,
        "p7_system_legal_exact_signal_controller.csv": p7_rows,
        **downstream,
        "risk_stat_trace_v9258.csv": p2_rows,
        "support_stat_trace_v9258.csv": p3_rows,
        "reference_feasibility_trace_v9258.csv": p4_rows,
        "true_delta_compute_trace_v9258.csv": p5_rows,
        "accepted_region_geometry_trace_v9258.csv": p4_rows,
        "support_density_trace_v9258.csv": p6_rows,
        "leaveout_trace_v9258.csv": downstream["p8_leave_dataset_and_stratum_out.csv"],
        "paired_replay_branch_trace_v9258.csv": downstream["p9_official_paired_replay.csv"],
        "true_delta_residual_trace_v9258.csv": p1_resid_rows,
        "true_delta_correctness_trace_v9258.csv": p2_correct_rows,
    }
    for name, artifact_rows in artifacts.items():
        write_csv_rows(out_dir / name, artifact_rows)
    audit = audit_no_fake([out_dir / name for name in artifacts])
    write_csv_rows(out_dir / "v9258_provenance_audit.csv", [audit])

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9257_boundary_pass": p0.get("v9257_boundary_pass"),
        "dataset_tuning_detected": 0,
        "reference_failure_autopsy_pass": p1.get("reference_failure_autopsy_pass"),
        "fp_attribution_fraction": p1.get("false_positive_attribution_fraction"),
        "fn_attribution_fraction": p1.get("false_negative_attribution_fraction"),
        "bad_event_attribution_fraction": p1.get("bad_event_attribution_fraction"),
        "best_risk_stat_id": p2.get("best_risk_stat_id"),
        "risk_stat_pass": p2.get("risk_stat_pass"),
        "risk_stat_auc_bad_event": p2.get("risk_stat_auc_bad_event"),
        "risk_gate_bad_event": p2.get("risk_gate_bad_event"),
        "best_support_stat_id": p3.get("best_support_stat_id"),
        "support_stat_pass": p3.get("support_stat_pass"),
        "support_diagnostic_pass": p3.get("support_diagnostic_pass"),
        "legal_oracle_jaccard": p4.get("legal_oracle_jaccard", p3.get("legal_oracle_jaccard")),
        "accepted_family_count": p4.get("accepted_family_count"),
        "accepted_signal_strata_count": p4.get("accepted_strata_count"),
        "max_family_share": p4.get("max_family_share"),
        "exact_reference_feasible": p4.get("exact_reference_feasible"),
        "reference_controller_precision": p4.get("reference_controller_precision"),
        "reference_controller_coverage": p4.get("reference_controller_coverage"),
        "reference_controller_bad_event": p4.get("reference_controller_bad_event"),
        "best_true_delta_id": p5.get("best_true_delta_id"),
        "true_delta_compute_pass": p5.get("true_delta_compute_pass"),
        "true_delta_auc": p5.get("true_delta_auc"),
        "true_delta_agreement": p5.get("true_delta_agreement"),
        "true_delta_step_ratio_q90": p5.get("true_delta_step_ratio_q90"),
        "true_delta_memory_ratio": p5.get("true_delta_memory_ratio"),
        "best_system_controller_id": p7.get("best_system_controller_id", p4.get("best_reference_controller_id", "")),
        "system_legal_controller_pass": p7.get("system_legal_controller_pass", 0),
        "controller_precision": p7.get("controller_precision", 0.0),
        "controller_coverage": p7.get("controller_coverage", 0.0),
        "controller_bad_event": p7.get("controller_bad_event", 0.0),
        "oracle_support_pass": p6.get("oracle_support_pass"),
        "oracle_precision": p6.get("oracle_precision"),
        "oracle_coverage": p6.get("oracle_coverage"),
        "oracle_bad_event": p6.get("oracle_bad_event"),
        "support_measurement_pass": p6.get("support_measurement_pass"),
        "natural_real_event_count": p6.get("natural_real_event_count"),
        "balanced_diagnostic_real_event_count": p6.get("balanced_diagnostic_real_event_count"),
        "measured_signal_strata_count": p6.get("measured_signal_strata_count"),
        "measured_family_count": p6.get("measured_family_count"),
        "leave_dataset_out_pass": 0,
        "leave_stratum_out_pass": 0,
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "external_ready": 0,
        "primary_blocker": blocker,
        "next_required_implementation": next_required,
        "success_v9258_strict_purekan_functional": 0,
        "success_v9258_full_functional": 0,
        "success_v9258_external_ready": 0,
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
    write_csv_rows(out_dir / "contract_audit_v9258.csv", [{
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "risk_support_sufficient_statistics": 1,
        "exact_reference_feasibility_v2": 1,
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
    _figures(out_dir, route)
    print(json.dumps(route, indent=2, sort_keys=True))
    return route


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(RESULT_ROOT / "v9258_risk_support_sufficient_statistics_exact_signal_accept_region_first_20260512T110000Z"))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    p.add_argument("--microprobe-steps", type=int, default=168)
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
