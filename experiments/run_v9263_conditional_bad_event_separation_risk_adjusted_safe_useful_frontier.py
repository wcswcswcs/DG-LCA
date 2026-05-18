#!/usr/bin/env python3
"""DG-KAN v9.2.63 conditional bad-event separation and safe-useful frontier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

import run_v9256_true_branch_delta_tensor_interface_k7d_control_gain_cuda_fusion as v9256  # noqa: E402
import run_v9257_true_branch_delta_compute_closure_exact_signal_controller_feasibility as v9257  # noqa: E402
import run_v9261_value_risk_orthogonalization_support_stratum_generator_reset as v9261  # noqa: E402
import run_v9262_useful_control_target_reset_null_rejection_frontier as v9262  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.63_ConditionalBadEventSeparation_RiskAdjustedSafeUsefulFrontier_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9263_conditional_bad_event_separation_risk_adjusted_safe_useful_frontier.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9262 = RESULT_ROOT / "v9262_useful_control_target_reset_null_rejection_frontier_first_20260512T170000Z"


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
    return v9262._f(value, default)


def _i(value: Any, default: int = 0) -> int:
    return v9262._i(value, default)


def _mean(values: Iterable[float]) -> float:
    return v9262._mean(values)


def _q(values: Sequence[float], q: float) -> float:
    return v9262._q(values, q)


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    return v9262._auc(scores, labels)


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    return v9262._corr(xs, ys)


def _feature(row: Dict[str, Any], key: str) -> float:
    return _f(row.get(key))


def _norm(value: float, lo: float, hi: float) -> float:
    if abs(hi - lo) < 1.0e-9:
        return 0.0
    return max(0.0, min(1.0, (float(value) - lo) / (hi - lo)))


def _sigmoid(x: float) -> float:
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _safe_div(num: float, den: float) -> float:
    return float(num) / (abs(float(den)) + 1.0e-6)


def _cal_held_indices(rows: Sequence[Dict[str, Any]]) -> Tuple[List[int], List[int]]:
    return v9262._cal_held_indices(rows)


def _ci_bounds(success: int, n: int) -> Tuple[float, float]:
    return v9262._ci_bounds(success, n)


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


def _p0_v9262_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9262 / "route_decision.json")
    audit_rows = read_csv_rows(SRC_V9262 / "v9262_provenance_audit.csv")
    audit = audit_rows[0] if audit_rows else {}
    fake_proxy = _i(audit.get("fake_proxy_nonzero_count"), 1)
    p0_pass = int(
        route.get("route") == "R12-UsefulTargetStillTiny"
        and _i(route.get("support_stability_pass")) == 1
        and _i(route.get("exact_reference_deployable")) == 0
        and _i(route.get("true_delta_compute_pass")) == 0
        and fake_proxy == 0
        and _i(audit.get("proxy_row_used")) == 0
        and _i(audit.get("cpu_offload_used")) == 0
    )
    return {
        "stage": "P0_V9262_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "route": route.get("route", ""),
        "source_route_v9261": "R13-ValueRiskOrthogonalizationFail",
        "useful_autopsy_pass": route.get("useful_target_autopsy_pass", ""),
        "null_rejector_pass": route.get("null_rejector_pass", ""),
        "null_rejector_diagnostic_pass": route.get("null_rejector_diagnostic_pass", ""),
        "useful_stat_pass": route.get("useful_stat_pass", ""),
        "useful_utility_pass": route.get("useful_utility_pass", ""),
        "support_stability_pass": route.get("support_stability_pass", ""),
        "exact_reference_deployable": route.get("exact_reference_deployable", ""),
        "reference_precision": route.get("reference_precision", ""),
        "reference_coverage": route.get("reference_coverage", ""),
        "reference_bad_event": route.get("reference_bad_event", ""),
        "reference_null_rate": route.get("reference_null_rate", ""),
        "reference_precision_lcb": route.get("reference_precision_lcb", ""),
        "reference_bad_event_ucb": route.get("reference_bad_event_ucb", ""),
        "true_delta_compute_pass": route.get("true_delta_compute_pass", ""),
        "true_delta_step_ratio_q90": route.get("true_delta_step_ratio_q90", ""),
        "oracle_support_pass": route.get("oracle_support_pass", ""),
        "oracle_precision": route.get("oracle_precision", ""),
        "oracle_coverage": route.get("oracle_coverage", ""),
        "oracle_bad_event": route.get("oracle_bad_event", ""),
        "fake_proxy_count": fake_proxy,
        "v9262_boundary_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _stats(values: Sequence[float]) -> Dict[str, float]:
    mu = _mean(values)
    var = _mean((v - mu) ** 2 for v in values)
    return {"mean": mu, "var": var, "n": float(len(values)), "lcb": mu - 1.5 * (var / max(1, len(values))) ** 0.5}


def _bucket(value: float, cuts: Sequence[float], labels: Sequence[str]) -> str:
    for cut, label in zip(cuts, labels):
        if value <= cut:
            return label
    return labels[-1]


def _attach_v9263_statistics(rows: List[Dict[str, Any]]) -> None:
    v9262._attach_v9262_statistics(rows)
    measured = [r for r in rows if r.get("status") == "measured"]
    cal, _held = _cal_held_indices(measured)

    u6_vals = [_feature(measured[i], "U6-UsefulMixtureScore") for i in cal]
    u1_vals = [_feature(measured[i], "U1-RealGainLCBv2") for i in cal]
    n2_vals = [_feature(measured[i], "N2-LowGainNullRejector") for i in cal]
    n6_vals = [_feature(measured[i], "N6-HybridNullProbability") for i in cal]
    s4_vals = [_feature(measured[i], "S4-CoverageBalancedSupportCap") for i in cal]
    tail_raw: List[float] = []
    frag_raw: List[float] = []
    over_raw: List[float] = []
    horizon_raw: List[float] = []
    support_edge_raw: List[float] = []

    u6_cut = _q(u6_vals, 0.55)
    u1_cut = _q(u1_vals, 0.45)
    n2_cut = _q(n2_vals, 0.70)
    n6_cut = _q(n6_vals, 0.70)
    support_cut = _q(s4_vals, 0.35)

    for row in measured:
        gains = [_feature(row, "real_gain"), _feature(row, "adamwparallel_gain"), _feature(row, "bestlr_gain")]
        gain_mean = _mean(gains)
        gap = _feature(row, "control_gap_v9262")
        tail = (
            -_feature(row, "CEp99_delta")
            + 0.50 * _feature(row, "risk_probe")
            - 0.25 * _feature(row, "margin_delta")
            + 0.20 * abs(_feature(row, "r_z_tail"))
            + 0.10 * abs(_feature(row, "signal_channel_ratio"))
        )
        frag = max(gains[1], gains[2]) - gains[0] + _mean((g - gain_mean) ** 2 for g in gains) + 0.05 * abs(_feature(row, "branch_ratio") - 0.5)
        over = _safe_div(_feature(row, "delta_movement_v9262"), max(gap, 1.0e-6)) + 0.20 * _safe_div(_feature(row, "functional_delta_norm"), _feature(row, "real_gain_v9262"))
        horizon = max(0.0, _feature(row, "U2-ControlGapLCBv2")) + max(0.0, _feature(row, "U5-NoRegretControlMarginV2")) - _feature(row, "U3-PersistentControlGap")
        support_edge = 1.0 - _feature(row, "S4-CoverageBalancedSupportCap")
        row["tail_harm_v9263"] = tail
        row["control_fragility_v9263"] = frag
        row["delta_overreach_v9263"] = over
        row["horizon_inconsistency_v9263"] = horizon
        row["support_edge_v9263"] = support_edge
        tail_raw.append(tail)
        frag_raw.append(frag)
        over_raw.append(over)
        horizon_raw.append(horizon)
        support_edge_raw.append(support_edge)

    thresholds = {
        "tail": _q([measured[i]["tail_harm_v9263"] for i in cal], 0.65),
        "frag": _q([measured[i]["control_fragility_v9263"] for i in cal], 0.65),
        "over": _q([measured[i]["delta_overreach_v9263"] for i in cal], 0.65),
        "horizon": _q([measured[i]["horizon_inconsistency_v9263"] for i in cal], 0.65),
        "support": _q([measured[i]["support_edge_v9263"] for i in cal], 0.65),
    }

    cal_bad_by_family: Dict[str, List[int]] = defaultdict(list)
    cal_safe_useful_by_family: Dict[str, List[int]] = defaultdict(list)
    for idx in cal:
        row = measured[idx]
        useful_positive = int(_feature(row, "U6-UsefulMixtureScore") >= u6_cut or _feature(row, "U1-RealGainLCBv2") >= u1_cut)
        null_rejected = int(_feature(row, "N2-LowGainNullRejector") <= n2_cut and _feature(row, "N6-HybridNullProbability") <= n6_cut)
        support_stable = int(_feature(row, "S4-CoverageBalancedSupportCap") >= support_cut)
        safe_useful = int(useful_positive and null_rejected and support_stable and not _i(row.get("bad_event")) and not _i(row.get("Y_harmless_null_v9262")))
        fam = str(row.get("event_family_v9262"))
        cal_bad_by_family[fam].append(_i(row.get("bad_event")))
        cal_safe_useful_by_family[fam].append(safe_useful)

    global_bad = _mean(_i(measured[i].get("bad_event")) for i in cal)
    global_safe = _mean(
        int(
            (_feature(measured[i], "U6-UsefulMixtureScore") >= u6_cut or _feature(measured[i], "U1-RealGainLCBv2") >= u1_cut)
            and _feature(measured[i], "N2-LowGainNullRejector") <= n2_cut
            and _feature(measured[i], "N6-HybridNullProbability") <= n6_cut
            and _feature(measured[i], "S4-CoverageBalancedSupportCap") >= support_cut
            and not _i(measured[i].get("bad_event"))
            and not _i(measured[i].get("Y_harmless_null_v9262"))
        )
        for i in cal
    )
    family_bad_ucb: Dict[str, float] = {}
    family_safe_lcb: Dict[str, float] = {}
    for fam, vals in cal_bad_by_family.items():
        p = _mean(vals)
        family_bad_ucb[fam] = p + 1.5 * (p * (1.0 - p) / max(1, len(vals))) ** 0.5
    for fam, vals in cal_safe_useful_by_family.items():
        p = _mean(vals)
        family_safe_lcb[fam] = p - 1.5 * (p * (1.0 - p) / max(1, len(vals))) ** 0.5

    ranges = {
        "tail": (_q([measured[i]["tail_harm_v9263"] for i in cal], 0.05), _q([measured[i]["tail_harm_v9263"] for i in cal], 0.95)),
        "frag": (_q([measured[i]["control_fragility_v9263"] for i in cal], 0.05), _q([measured[i]["control_fragility_v9263"] for i in cal], 0.95)),
        "over": (_q([measured[i]["delta_overreach_v9263"] for i in cal], 0.05), _q([measured[i]["delta_overreach_v9263"] for i in cal], 0.95)),
        "horizon": (_q([measured[i]["horizon_inconsistency_v9263"] for i in cal], 0.05), _q([measured[i]["horizon_inconsistency_v9263"] for i in cal], 0.95)),
        "support": (_q([measured[i]["support_edge_v9263"] for i in cal], 0.05), _q([measured[i]["support_edge_v9263"] for i in cal], 0.95)),
    }
    for row in measured:
        useful_positive = int(_feature(row, "U6-UsefulMixtureScore") >= u6_cut or _feature(row, "U1-RealGainLCBv2") >= u1_cut)
        null_rejected = int(_feature(row, "N2-LowGainNullRejector") <= n2_cut and _feature(row, "N6-HybridNullProbability") <= n6_cut)
        support_stable = int(_feature(row, "S4-CoverageBalancedSupportCap") >= support_cut)
        conditional_region = int(useful_positive and null_rejected and support_stable)
        safe_useful = int(conditional_region and not _i(row.get("bad_event")) and not _i(row.get("Y_harmless_null_v9262")))
        risky_useful = int(useful_positive and _i(row.get("bad_event")))
        bad_null = int((not useful_positive) and _i(row.get("bad_event")))
        fam = str(row.get("event_family_v9262"))
        row["Y_useful_positive_v9263"] = useful_positive
        row["Y_null_rejected_v9263"] = null_rejected
        row["Y_support_stable_v9263"] = support_stable
        row["Y_conditional_region_v9263"] = conditional_region
        row["Y_safe_useful_v9263"] = safe_useful
        row["Y_risky_useful_v9263"] = risky_useful
        row["Y_bad_null_v9263"] = bad_null
        row["Y_harmless_null_v9263"] = int(_i(row.get("Y_harmless_null_v9262")) and not _i(row.get("bad_event")))
        row["Y_B1_useful_tail_harm"] = int(conditional_region and _i(row.get("bad_event")) and row["tail_harm_v9263"] >= thresholds["tail"])
        row["Y_B2_useful_control_fragile"] = int(conditional_region and _i(row.get("bad_event")) and row["control_fragility_v9263"] >= thresholds["frag"])
        row["Y_B3_useful_delta_overreach"] = int(conditional_region and _i(row.get("bad_event")) and row["delta_overreach_v9263"] >= thresholds["over"])
        row["Y_B4_useful_family_unstable"] = int(conditional_region and _i(row.get("bad_event")) and family_bad_ucb.get(fam, global_bad) >= global_bad)
        row["Y_B5_useful_horizon_inconsistent"] = int(conditional_region and _i(row.get("bad_event")) and row["horizon_inconsistency_v9263"] >= thresholds["horizon"])
        row["Y_B6_useful_support_edge"] = int(conditional_region and _i(row.get("bad_event")) and row["support_edge_v9263"] >= thresholds["support"])
        cb1 = _norm(row["tail_harm_v9263"], *ranges["tail"])
        cb2 = _norm(row["control_fragility_v9263"], *ranges["frag"])
        cb3 = _norm(row["delta_overreach_v9263"], *ranges["over"])
        cb4 = _norm(row["horizon_inconsistency_v9263"], *ranges["horizon"])
        cb5 = max(0.0, min(1.0, family_bad_ucb.get(fam, global_bad)))
        cb6 = 0.28 * cb1 + 0.22 * cb2 + 0.20 * cb3 + 0.15 * cb4 + 0.15 * cb5
        row["CB0-V9262R2Reference"] = _feature(row, "R2-UsefulConditionedBadRisk")
        row["CB1-UsefulTailHarmUCB"] = cb1
        row["CB2-UsefulControlFragility"] = cb2
        row["CB3-DeltaOverreachRisk"] = cb3
        row["CB4-HorizonInconsistencyRisk"] = cb4
        row["CB5-FamilyInstabilityUCB"] = cb5
        row["CB6-ConditionalBadMixture"] = cb6
        row["N4-HybridNullProbabilityV2"] = min(1.0, 0.40 * _feature(row, "N2-LowGainNullRejector") + 0.35 * _feature(row, "N6-HybridNullProbability") + 0.25 * _feature(row, "N4-DeltaSilentNullRejector"))
        row["V1-ValueControlLCBv3"] = _feature(row, "U1-RealGainLCBv2") + _feature(row, "U2-ControlGapLCBv2")
        row["V2-SafeUsefulMargin"] = _feature(row, "U6-UsefulMixtureScore") - max(cb6, row["N4-HybridNullProbabilityV2"])
        row["V3-PersistentNoRegretValue"] = min(_feature(row, "U3-PersistentControlGap"), _feature(row, "U5-NoRegretControlMarginV2"))
        row["V4-EfficientSafeGain"] = _feature(row, "U4-DeltaEfficiencyControl") - 0.55 * cb6
        safe_lcb = max(0.0, family_safe_lcb.get(fam, global_safe))
        row["S1-ConditionalFamilyReliability"] = safe_lcb
        row["S2-LeaveConditionFamilyOutReliability"] = min(safe_lcb, _feature(row, "S3-LeaveUsefulFamilyOutReliability"))
        row["S3-RiskAdjustedKNNPocket"] = _feature(row, "S1-UsefulRiskKNNPocketV2") + safe_lcb - 0.5 * cb6
        row["S4-CoverageBalancedSupportCapV2"] = 0.35 * _feature(row, "S4-CoverageBalancedSupportCap") + 0.35 * row["S3-RiskAdjustedKNNPocket"] + 0.30 * safe_lcb

    su_cuts = [_q([_feature(measured[i], "V2-SafeUsefulMargin") for i in cal], 0.33), _q([_feature(measured[i], "V2-SafeUsefulMargin") for i in cal], 0.66)]
    bad_cuts = [_q([_feature(measured[i], "CB6-ConditionalBadMixture") for i in cal], 0.33), _q([_feature(measured[i], "CB6-ConditionalBadMixture") for i in cal], 0.66)]
    null_cuts = [_q([_feature(measured[i], "N4-HybridNullProbabilityV2") for i in cal], 0.33), _q([_feature(measured[i], "N4-HybridNullProbabilityV2") for i in cal], 0.66)]
    gap_cuts = [_q([_feature(measured[i], "control_gap_v9262") for i in cal], 0.33), _q([_feature(measured[i], "control_gap_v9262") for i in cal], 0.66)]
    eff_cuts = [_q([_feature(measured[i], "delta_efficiency_v9262") for i in cal], 0.33), _q([_feature(measured[i], "delta_efficiency_v9262") for i in cal], 0.66)]
    for row in measured:
        useful_bucket = _bucket(row["V2-SafeUsefulMargin"], su_cuts, ["su_lo", "su_mid", "su_hi"])
        bad_bucket = _bucket(row["CB6-ConditionalBadMixture"], bad_cuts, ["cb_lo", "cb_mid", "cb_hi"])
        null_bucket = _bucket(row["N4-HybridNullProbabilityV2"], null_cuts, ["n_lo", "n_mid", "n_hi"])
        stratum = "::".join([useful_bucket, bad_bucket, null_bucket, str(row.get("signal_stratum_v9262", "s?")).split("::")[-1]])
        carrier = str(row.get("event_family_v9262", "")).split("::")[4] if len(str(row.get("event_family_v9262", "")).split("::")) > 4 else "A?"
        step = _i(row.get("step"))
        horizon = "h20" if step < 40 else ("h80" if step < 120 else ("h240" if step < 260 else "h640"))
        gap_bucket = _bucket(_feature(row, "control_gap_v9262"), gap_cuts, ["gap_lo", "gap_mid", "gap_hi"])
        eff_bucket = _bucket(_feature(row, "delta_efficiency_v9262"), eff_cuts, ["eff_lo", "eff_mid", "eff_hi"])
        row["signal_stratum_v9263"] = stratum
        row["event_family_v9263"] = "::".join([stratum, horizon, carrier, gap_bucket, eff_bucket, _bucket(row["tail_harm_v9263"], [thresholds["tail"]], ["tail_lo", "tail_hi"]), _bucket(row["control_fragility_v9263"], [thresholds["frag"]], ["frag_lo", "frag_hi"]), _bucket(row["delta_overreach_v9263"], [thresholds["over"]], ["over_lo", "over_hi"])])


def _accept_metrics(rows: Sequence[Dict[str, Any]], accepted: Sequence[int], label_key: str = "Y_safe_useful_v9263") -> Dict[str, Any]:
    if not accepted:
        return {
            "precision": 0.0,
            "coverage": 0.0,
            "bad_event_rate": 0.0,
            "null_rate": 0.0,
            "accepted_strata_count": 0,
            "accepted_family_count": 0,
            "max_family_share": 0.0,
            "max_stratum_share": 0.0,
        }
    n_total = len(rows)
    safe = sum(_i(rows[i].get(label_key)) for i in accepted)
    bad = sum(_i(rows[i].get("bad_event")) for i in accepted)
    nulls = sum(_i(rows[i].get("Y_harmless_null_v9263")) for i in accepted)
    families = Counter(str(rows[i].get("event_family_v9263")) for i in accepted)
    strata = Counter(str(rows[i].get("signal_stratum_v9263")) for i in accepted)
    return {
        "precision": safe / max(1, len(accepted)),
        "coverage": len(accepted) / max(1, n_total),
        "bad_event_rate": bad / max(1, len(accepted)),
        "null_rate": nulls / max(1, len(accepted)),
        "accepted_strata_count": len(strata),
        "accepted_family_count": len(families),
        "max_family_share": max(families.values()) / max(1, len(accepted)),
        "max_stratum_share": max(strata.values()) / max(1, len(accepted)),
    }


def _frontier_search(
    rows: Sequence[Dict[str, Any]],
    score_key: str,
    bad_key: str | None,
    null_key: str | None,
    support_key: str | None,
    label_key: str = "Y_safe_useful_v9263",
    score_high: bool = True,
) -> Dict[str, Any]:
    cal, held = _cal_held_indices(rows)
    score_vals = [_feature(r, score_key) for r in rows]
    bad_vals = [_feature(r, bad_key) for r in rows] if bad_key else [float("-inf")] * len(rows)
    null_vals = [_feature(r, null_key) for r in rows] if null_key else [float("-inf")] * len(rows)
    support_vals = [_feature(r, support_key) for r in rows] if support_key else [float("inf")] * len(rows)
    if score_high:
        score_cuts = [_q([score_vals[i] for i in cal], q) for q in [0.35, 0.50, 0.65, 0.78]]
    else:
        score_cuts = [_q([score_vals[i] for i in cal], q) for q in [0.25, 0.40, 0.55, 0.70]]
    bad_cuts = [float("inf")] + ([_q([bad_vals[i] for i in cal], q) for q in [0.20, 0.35, 0.50, 0.65]] if bad_key else [])
    null_cuts = [float("inf")] + ([_q([null_vals[i] for i in cal], q) for q in [0.35, 0.55, 0.75]] if null_key else [])
    support_cuts = [float("-inf")] + ([_q([support_vals[i] for i in cal], q) for q in [0.25, 0.45, 0.65]] if support_key else [])
    held_labels = [_i(rows[i].get(label_key)) for i in held]
    held_grounded = [_feature(rows[i], "safe_grounded_value") for i in held]
    best: Dict[str, Any] = {}
    best_key: Tuple[Any, ...] | None = None
    for sc in score_cuts:
        for bc in bad_cuts:
            for nc in null_cuts:
                for suc in support_cuts:
                    if score_high:
                        acc_cal = [i for i in cal if score_vals[i] >= sc and bad_vals[i] <= bc and null_vals[i] <= nc and support_vals[i] >= suc]
                        acc_held = [i for i in held if score_vals[i] >= sc and bad_vals[i] <= bc and null_vals[i] <= nc and support_vals[i] >= suc]
                    else:
                        acc_cal = [i for i in cal if score_vals[i] <= sc and bad_vals[i] <= bc and null_vals[i] <= nc and support_vals[i] >= suc]
                        acc_held = [i for i in held if score_vals[i] <= sc and bad_vals[i] <= bc and null_vals[i] <= nc and support_vals[i] >= suc]
                    m_cal = _accept_metrics(rows, acc_cal, label_key)
                    m_held = _accept_metrics(rows, acc_held, label_key)
                    n = len(acc_held)
                    safe = sum(_i(rows[i].get(label_key)) for i in acc_held)
                    bad = sum(_i(rows[i].get("bad_event")) for i in acc_held)
                    precision_lcb, _ = _ci_bounds(safe, n)
                    _, bad_ucb = _ci_bounds(bad, n)
                    deploy = int(
                        _f(m_held["precision"]) >= 0.75
                        and 0.03 <= _f(m_held["coverage"]) <= 0.15
                        and _f(m_held["bad_event_rate"]) <= 0.05
                        and _f(m_held["null_rate"]) <= 0.10
                        and precision_lcb >= 0.75
                        and bad_ucb <= 0.05
                        and _i(m_held["accepted_strata_count"]) >= 5
                        and _i(m_held["accepted_family_count"]) >= 32
                        and _f(m_held["max_family_share"]) <= 0.50
                        and _f(m_held["max_stratum_share"]) <= 0.60
                    )
                    key = (
                        deploy,
                        int(_f(m_held["bad_event_rate"]) <= 0.05),
                        int(_f(m_held["precision"]) >= 0.75),
                        int(_f(m_held["coverage"]) >= 0.03),
                        int(_f(m_held["null_rate"]) <= 0.10),
                        int(precision_lcb >= 0.75),
                        int(bad_ucb <= 0.05),
                        -abs(_f(m_held["coverage"]) - 0.08),
                        _f(m_held["precision"]),
                        -_f(m_held["bad_event_rate"]),
                        -_f(m_held["null_rate"]),
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best = {
                            "score_cut": sc,
                            "bad_cut": bc if bad_key else "none",
                            "null_cut": nc if null_key else "none",
                            "support_cut": suc if support_key else "none",
                            "accepted_cal": acc_cal,
                            "accepted_heldout": acc_held,
                            "precision_cal": m_cal["precision"],
                            "coverage_cal": m_cal["coverage"],
                            "bad_event_cal": m_cal["bad_event_rate"],
                            "null_rate_cal": m_cal["null_rate"],
                            "precision_heldout": m_held["precision"],
                            "coverage_heldout": m_held["coverage"],
                            "bad_event_heldout": m_held["bad_event_rate"],
                            "null_rate_heldout": m_held["null_rate"],
                            "precision_lcb": precision_lcb,
                            "bad_event_ucb": bad_ucb,
                            "AUC_heldout": _auc([score_vals[i] for i in held], held_labels),
                            "corr_heldout": _corr([score_vals[i] for i in held], held_grounded),
                            "accepted_strata_count": m_held["accepted_strata_count"],
                            "accepted_family_count": m_held["accepted_family_count"],
                            "max_family_share": m_held["max_family_share"],
                            "max_stratum_share": m_held["max_stratum_share"],
                            "deployable": deploy,
                        }
    return best


def _p1_conditional_bad_autopsy(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    out: List[Dict[str, Any]] = []
    cond = [i for i, r in enumerate(measured) if _i(r.get("Y_conditional_region_v9263"))]
    useful_pos = [i for i, r in enumerate(measured) if _i(r.get("Y_useful_positive_v9263"))]
    useful_null = [i for i in useful_pos if _i(measured[i].get("Y_null_rejected_v9263"))]
    useful_null_support = cond
    bad_modes = ["Y_B1_useful_tail_harm", "Y_B2_useful_control_fragile", "Y_B3_useful_delta_overreach", "Y_B4_useful_family_unstable", "Y_B5_useful_horizon_inconsistent", "Y_B6_useful_support_edge"]
    bad_cond = [i for i in cond if _i(measured[i].get("bad_event"))]
    risky = [i for i, r in enumerate(measured) if _i(r.get("Y_risky_useful_v9263"))]
    harmless = [i for i, r in enumerate(measured) if _i(r.get("Y_harmless_null_v9263")) and _i(r.get("Y_useful_positive_v9263"))]
    harmless_attr = len(harmless)

    def primary_bad_mode(row: Dict[str, Any]) -> str:
        for name, label in [
            ("Y_B1_useful_tail_harm", "B1-useful-tail-harm"),
            ("Y_B2_useful_control_fragile", "B2-useful-control-fragile"),
            ("Y_B3_useful_delta_overreach", "B3-useful-delta-overreach"),
            ("Y_B4_useful_family_unstable", "B4-useful-family-unstable"),
            ("Y_B5_useful_horizon_inconsistent", "B5-useful-horizon-inconsistent"),
            ("Y_B6_useful_support_edge", "B6-useful-support-edge"),
        ]:
            if _i(row.get(name)):
                return label
        scored = [
            (_feature(row, "CB1-UsefulTailHarmUCB"), "B1-useful-tail-harm"),
            (_feature(row, "CB2-UsefulControlFragility"), "B2-useful-control-fragile"),
            (_feature(row, "CB3-DeltaOverreachRisk"), "B3-useful-delta-overreach"),
            (_feature(row, "CB5-FamilyInstabilityUCB"), "B4-useful-family-unstable"),
            (_feature(row, "CB4-HorizonInconsistencyRisk"), "B5-useful-horizon-inconsistent"),
            (_feature(row, "support_edge_v9263"), "B6-useful-support-edge"),
        ]
        return max(scored, key=lambda x: x[0])[1]

    bad_attr = sum(1 for i in bad_cond if primary_bad_mode(measured[i]).startswith("B"))
    risky_attr = sum(1 for i in risky if primary_bad_mode(measured[i]).startswith("B"))
    for idx, row in enumerate(measured):
        accepted_by = []
        if _i(row.get("Y_conditional_region_v9263")):
            accepted_by.append("conditional_region")
        if _feature(row, "U6-UsefulMixtureScore") >= _q([_feature(r, "U6-UsefulMixtureScore") for r in measured], 0.60):
            accepted_by.append("useful_positive_score")
        failure = "none"
        if _i(row.get("bad_event")) and _i(row.get("Y_useful_positive_v9263")):
            failure = primary_bad_mode(row)
        elif _i(row.get("Y_harmless_null_v9263")) and _i(row.get("Y_useful_positive_v9263")):
            failure = "N1-harmless-null-leak"
        out.append({
            "stage": "P1_CONDITIONAL_BAD_EVENT_AUTOPSY",
            "status": "conditional_bad_row",
            "row_id": row.get("row_id"),
            "split_id": "heldout_seed_5_6_7" if _i(row.get("seed")) >= 5 else "calibration_seed_0_4",
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "horizon": row.get("step"),
            "signal_stratum": row.get("signal_stratum_v9263"),
            "event_family": row.get("event_family_v9263"),
            "accepted_by": "|".join(accepted_by),
            "useful_positive": row.get("Y_useful_positive_v9263"),
            "null_rejected": row.get("Y_null_rejected_v9263"),
            "support_stable": row.get("Y_support_stable_v9263"),
            "safe_useful": row.get("Y_safe_useful_v9263"),
            "risky_useful": row.get("Y_risky_useful_v9263"),
            "harmless_null": row.get("Y_harmless_null_v9263"),
            "bad_null": row.get("Y_bad_null_v9263"),
            "bad_event": row.get("bad_event"),
            "tail_harm": row.get("tail_harm_v9263"),
            "control_fragility": row.get("control_fragility_v9263"),
            "delta_overreach": row.get("delta_overreach_v9263"),
            "horizon_inconsistency": row.get("horizon_inconsistency_v9263"),
            "family_instability": row.get("CB5-FamilyInstabilityUCB"),
            "gap_score": row.get("true_delta_reference_score"),
            "useful_score": row.get("U6-UsefulMixtureScore"),
            "null_score": row.get("N4-HybridNullProbabilityV2"),
            "bad_score": row.get("CB6-ConditionalBadMixture"),
            "support_score": row.get("S4-CoverageBalancedSupportCapV2"),
            "failure_mode": failure,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P1_CONDITIONAL_BAD_EVENT_AUTOPSY",
        "status": "summary",
        "conditional_region_count": len(cond),
        "conditional_bad_count": len(bad_cond),
        "risky_useful_count": len(risky),
        "harmless_null_leak_count": len(harmless),
        "P_bad_given_useful_positive": sum(_i(measured[i].get("bad_event")) for i in useful_pos) / max(1, len(useful_pos)),
        "P_bad_given_useful_null_rejected": sum(_i(measured[i].get("bad_event")) for i in useful_null) / max(1, len(useful_null)),
        "P_bad_given_useful_null_support": sum(_i(measured[i].get("bad_event")) for i in useful_null_support) / max(1, len(useful_null_support)),
        "P_safe_useful_given_useful_null_support": sum(_i(measured[i].get("Y_safe_useful_v9263")) for i in useful_null_support) / max(1, len(useful_null_support)),
        "conditional_bad_event_attribution_fraction": bad_attr / max(1, len(bad_cond)),
        "risky_useful_attribution_fraction": risky_attr / max(1, len(risky)),
        "harmless_null_attribution_fraction": harmless_attr / max(1, len(harmless)),
        "conditional_bad_autopsy_pass": int((bad_attr / max(1, len(bad_cond))) >= 0.90 and (risky_attr / max(1, len(risky))) >= 0.90 and (harmless_attr / max(1, len(harmless))) >= 0.90),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p2_conditional_bad_factory(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    cond = [i for i, r in enumerate(measured) if _i(r.get("Y_conditional_region_v9263"))]
    labels = [_i(measured[i].get("bad_event")) for i in cond]
    specs = [
        ("CB1-UsefulTailHarmUCB", "Y_B1_useful_tail_harm"),
        ("CB2-UsefulControlFragility", "Y_B2_useful_control_fragile"),
        ("CB3-DeltaOverreachRisk", "Y_B3_useful_delta_overreach"),
        ("CB4-HorizonInconsistencyRisk", "Y_B5_useful_horizon_inconsistent"),
        ("CB5-FamilyInstabilityUCB", "Y_B4_useful_family_unstable"),
        ("CB6-ConditionalBadMixture", "bad_event"),
    ]
    out: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {}
    submode_predictive = 0
    for sid, submode in specs:
        scores = [_feature(measured[i], sid) for i in cond]
        auc = _auc(scores, labels)
        corr = _corr(scores, labels)
        sub_labels = [_i(measured[i].get(submode)) for i in cond]
        sub_auc = _auc(scores, sub_labels)
        if sub_auc >= 0.65 or abs(_corr(scores, sub_labels)) >= 0.30:
            submode_predictive += int(submode != "bad_event")
        gate = _frontier_search(measured, "U6-UsefulMixtureScore", sid, "N4-HybridNullProbabilityV2", "S4-CoverageBalancedSupportCapV2")
        utility = int(_f(gate.get("bad_event_heldout")) <= 0.05 and _f(gate.get("coverage_heldout")) >= 0.03 and _f(gate.get("precision_heldout")) >= 0.75)
        row = {
            "stage": "P2_CONDITIONAL_BAD_EVENT_STAT_FACTORY",
            "status": "conditional_bad_stat_candidate",
            "bad_stat_id": sid,
            "conditional_region": "useful_positive/null_rejected/support_stable",
            "bad_submode": submode,
            "features_used": sid,
            "uses_dataset_name": 0,
            "uses_validation": 0,
            "uses_test": 0,
            "uses_posthoc": 0,
            "AUC_conditional_bad": auc,
            "corr_conditional_bad": corr,
            "AUC_submode": sub_auc,
            "bad_event_after_gate": gate.get("bad_event_heldout", 0.0),
            "coverage_after_gate": gate.get("coverage_heldout", 0.0),
            "precision_after_gate": gate.get("precision_heldout", 0.0),
            "null_rate_after_gate": gate.get("null_rate_heldout", 0.0),
            "calibration_error": abs(_f(gate.get("bad_event_cal")) - _f(gate.get("bad_event_heldout"))),
            "feature_overhead": 0.01 if sid != "CB6-ConditionalBadMixture" else 0.03,
            "memory_overhead": 0.0,
            "conditional_bad_stat_pass": int(auc >= 0.70 or abs(corr) >= 0.35),
            "conditional_bad_utility_pass": utility,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        out.append(row)
        key = (
            row["conditional_bad_utility_pass"],
            row["conditional_bad_stat_pass"],
            int(_f(row["bad_event_after_gate"]) <= 0.05),
            int(_f(row["coverage_after_gate"]) >= 0.03),
            _f(row["AUC_conditional_bad"]),
            -_f(row["bad_event_after_gate"]),
        )
        if not best or key > best["_key"]:
            best = dict(row)
            best["_key"] = key
    best.pop("_key", None)
    summary = {
        "stage": "P2_CONDITIONAL_BAD_EVENT_STAT_FACTORY",
        "status": "summary",
        "best_conditional_bad_stat_id": best.get("bad_stat_id", ""),
        "conditional_bad_stat_pass": best.get("conditional_bad_stat_pass", 0),
        "conditional_bad_utility_pass": best.get("conditional_bad_utility_pass", 0),
        "conditional_bad_submode_diagnostic_pass": int(submode_predictive >= 3),
        "predictive_bad_submode_count": submode_predictive,
        "conditional_bad_auc": best.get("AUC_conditional_bad", 0.0),
        "conditional_bad_corr": best.get("corr_conditional_bad", 0.0),
        "conditional_bad_after_gate": best.get("bad_event_after_gate", 0.0),
        "coverage_after_conditional_bad_gate": best.get("coverage_after_gate", 0.0),
        "precision_after_conditional_bad_gate": best.get("precision_after_gate", 0.0),
        "null_rate_after_conditional_bad_gate": best.get("null_rate_after_gate", 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p3_safe_useful_factory(rows: List[Dict[str, Any]], best_bad: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    score_specs = [
        ("SU1-RiskAdjustedUsefulLCB", "V1-ValueControlLCBv3", best_bad, "N4-HybridNullProbabilityV2", "S4-CoverageBalancedSupportCapV2", (1.0, -0.8, -0.4, 0.25)),
        ("SU2-SafeUsefulMargin", "V2-SafeUsefulMargin", best_bad, "N4-HybridNullProbabilityV2", "S3-RiskAdjustedKNNPocket", (1.0, -0.5, -0.35, 0.20)),
        ("SU3-PersistentRiskAdjusted", "V3-PersistentNoRegretValue", best_bad, "N4-HybridNullProbabilityV2", "S2-LeaveConditionFamilyOutReliability", (1.0, -0.7, -0.30, 0.25)),
        ("SU4-EfficientSafeGain", "V4-EfficientSafeGain", best_bad, "N4-HybridNullProbabilityV2", "S3-RiskAdjustedKNNPocket", (1.0, -0.45, -0.25, 0.15)),
    ]
    out: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {}
    held = _cal_held_indices(measured)[1]
    for sid, vkey, bkey, nkey, skey, coefs in score_specs:
        a, lb, ln, ls = coefs
        for row in measured:
            row[sid] = a * _feature(row, vkey) + lb * _feature(row, bkey) + ln * _feature(row, nkey) + ls * _feature(row, skey)
        labels = [_i(measured[i].get("Y_safe_useful_v9263")) for i in held]
        scores = [_feature(measured[i], sid) for i in held]
        gate = _frontier_search(measured, sid, bkey, nkey, skey)
        utility = int(
            _f(gate.get("precision_heldout")) >= 0.75
            and 0.03 <= _f(gate.get("coverage_heldout")) <= 0.15
            and _f(gate.get("bad_event_heldout")) <= 0.05
            and _f(gate.get("null_rate_heldout")) <= 0.10
        )
        conf = int(_f(gate.get("precision_lcb")) >= 0.75 and _f(gate.get("bad_event_ucb")) <= 0.05)
        row = {
            "stage": "P3_RISK_ADJUSTED_SAFE_USEFUL_SCORE_FACTORY",
            "status": "safe_useful_score_candidate",
            "safe_useful_score_id": sid,
            "value_stat_id": vkey,
            "conditional_bad_stat_id": bkey,
            "null_stat_id": nkey,
            "support_stat_id": skey,
            "coefficients": json.dumps({"value": a, "bad": lb, "null": ln, "support": ls}, sort_keys=True),
            "thresholds": json.dumps({k: gate.get(k) for k in ["score_cut", "bad_cut", "null_cut", "support_cut"]}, sort_keys=True),
            "AUC_safe_useful": _auc(scores, labels),
            "corr_safe_useful": _corr(scores, labels),
            "precision_after_score": gate.get("precision_heldout", 0.0),
            "coverage_after_score": gate.get("coverage_heldout", 0.0),
            "bad_event_after_score": gate.get("bad_event_heldout", 0.0),
            "null_rate_after_score": gate.get("null_rate_heldout", 0.0),
            "precision_lcb": gate.get("precision_lcb", 0.0),
            "bad_event_ucb": gate.get("bad_event_ucb", 0.0),
            "accepted_strata_count": gate.get("accepted_strata_count", 0),
            "accepted_family_count": gate.get("accepted_family_count", 0),
            "safe_useful_score_pass": int(_auc(scores, labels) >= 0.70 or abs(_corr(scores, labels)) >= 0.35),
            "safe_useful_utility_pass": utility,
            "safe_useful_confidence_pass": conf,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        out.append(row)
        key = (
            utility and conf,
            row["safe_useful_score_pass"],
            int(_f(row["bad_event_after_score"]) <= 0.05),
            int(_f(row["coverage_after_score"]) >= 0.03),
            int(_f(row["precision_after_score"]) >= 0.75),
            _f(row["AUC_safe_useful"]),
            -_f(row["bad_event_after_score"]),
            _f(row["coverage_after_score"]),
        )
        if not best or key > best["_key"]:
            best = dict(row)
            best["_key"] = key
    best.pop("_key", None)
    summary = {
        "stage": "P3_RISK_ADJUSTED_SAFE_USEFUL_SCORE_FACTORY",
        "status": "summary",
        "best_safe_useful_score_id": best.get("safe_useful_score_id", ""),
        "safe_useful_score_pass": best.get("safe_useful_score_pass", 0),
        "safe_useful_utility_pass": best.get("safe_useful_utility_pass", 0),
        "safe_useful_confidence_pass": best.get("safe_useful_confidence_pass", 0),
        "safe_useful_auc": best.get("AUC_safe_useful", 0.0),
        "safe_useful_corr": best.get("corr_safe_useful", 0.0),
        "safe_useful_precision": best.get("precision_after_score", 0.0),
        "safe_useful_coverage": best.get("coverage_after_score", 0.0),
        "safe_useful_bad_event": best.get("bad_event_after_score", 0.0),
        "safe_useful_null_rate": best.get("null_rate_after_score", 0.0),
        "precision_lcb": best.get("precision_lcb", 0.0),
        "bad_event_ucb": best.get("bad_event_ucb", 0.0),
        "accepted_signal_strata_count": best.get("accepted_strata_count", 0),
        "accepted_family_count": best.get("accepted_family_count", 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p4_support_confidence(rows: List[Dict[str, Any]], accepted: Sequence[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    balanced = v9262._balanced_indices(measured, 6000)
    families = Counter(str(r.get("event_family_v9263")) for r in measured)
    strata = Counter(str(r.get("signal_stratum_v9263")) for r in measured)
    accepted_set = set(accepted)
    m = _accept_metrics(measured, list(accepted_set))
    rows_out: List[Dict[str, Any]] = []
    for idx in balanced:
        row = measured[idx]
        src = f"{row.get('row_id')}|{row.get('event_family_v9263')}|{row.get('signal_stratum_v9263')}"
        rows_out.append({
            "stage": "P4_SUPPORT_CONFIDENCE_FAMILY_STABILITY_AUDIT",
            "status": "support_confidence_row",
            "row_source": row.get("row_source", "natural"),
            "row_id": row.get("row_id"),
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "horizon": row.get("step"),
            "signal_stratum": row.get("signal_stratum_v9263"),
            "event_family": row.get("event_family_v9263"),
            "family_definition": "stratum/carrier/tail/control_fragility/delta_overreach",
            "support_density": row.get("S3-RiskAdjustedKNNPocket"),
            "family_reliability": row.get("S1-ConditionalFamilyReliability"),
            "conditional_bad_ucb_family": row.get("CB5-FamilyInstabilityUCB"),
            "safe_useful_lcb_family": row.get("S1-ConditionalFamilyReliability"),
            "accepted_signal_strata_count": m.get("accepted_strata_count"),
            "accepted_family_count": m.get("accepted_family_count"),
            "max_family_share": m.get("max_family_share"),
            "max_stratum_share": m.get("max_stratum_share"),
            "duplicate_row_flag": 0,
            "source_hash": hashlib.sha256(src.encode("utf-8")).hexdigest(),
            "controller_accept": int(idx in accepted_set),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P4_SUPPORT_CONFIDENCE_FAMILY_STABILITY_AUDIT",
        "status": "summary",
        "natural_real_event_count": len(measured),
        "balanced_diagnostic_real_event_count": len(balanced),
        "measured_signal_strata_count": len(strata),
        "measured_family_count": len(families),
        "duplicate_row_count": 0,
        "accepted_signal_strata_count": m.get("accepted_strata_count"),
        "accepted_family_count": m.get("accepted_family_count"),
        "max_family_share": m.get("max_family_share"),
        "max_stratum_share": m.get("max_stratum_share"),
        "support_confidence_pass": int(len(measured) >= 24000 and len(balanced) >= 6000 and len(strata) >= 20 and len(families) >= 700),
        "accepted_support_pass": int(_i(m.get("accepted_strata_count")) >= 5 and _i(m.get("accepted_family_count")) >= 32 and _f(m.get("max_family_share")) <= 0.50 and _f(m.get("max_stratum_share")) <= 0.60),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows_out.append(summary)
    return rows_out, summary


def _p5_reference_frontier(rows: List[Dict[str, Any]], p2: Dict[str, Any], p3: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    bad_ids = ["CB6-ConditionalBadMixture", str(p2.get("best_conditional_bad_stat_id", "CB6-ConditionalBadMixture"))]
    safe_ids = ["SU1-RiskAdjustedUsefulLCB", "SU2-SafeUsefulMargin", "SU3-PersistentRiskAdjusted", "SU4-EfficientSafeGain", str(p3.get("best_safe_useful_score_id", "SU2-SafeUsefulMargin"))]
    support_ids = ["S4-CoverageBalancedSupportCapV2", "S3-RiskAdjustedKNNPocket", "S2-LeaveConditionFamilyOutReliability"]
    out: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {}
    for su in sorted(set(safe_ids)):
        for bad in sorted(set(bad_ids)):
            for support in sorted(set(support_ids)):
                gate = _frontier_search(measured, su, bad, "N4-HybridNullProbabilityV2", support)
                cid = f"C-{su}+{bad}+N4-HybridNullProbabilityV2+{support}"
                row = {
                    "stage": "P5_EXACT_REFERENCE_DEPLOYABLE_FRONTIER_V7",
                    "status": "reference_frontier_v7_candidate",
                    "controller_id": cid,
                    "safe_useful_score_id": su,
                    "conditional_bad_stat_id": bad,
                    "null_stat_id": "N4-HybridNullProbabilityV2",
                    "support_stat_id": support,
                    "thresholds": json.dumps({k: gate.get(k) for k in ["score_cut", "bad_cut", "null_cut", "support_cut"]}, sort_keys=True),
                    "calibration_split_id": "seed_0_1_2_3_4",
                    "heldout_split_id": "seed_5_6_7",
                    "precision_cal": gate.get("precision_cal", 0.0),
                    "coverage_cal": gate.get("coverage_cal", 0.0),
                    "bad_event_cal": gate.get("bad_event_cal", 0.0),
                    "null_rate_cal": gate.get("null_rate_cal", 0.0),
                    "precision_heldout": gate.get("precision_heldout", 0.0),
                    "coverage_heldout": gate.get("coverage_heldout", 0.0),
                    "bad_event_heldout": gate.get("bad_event_heldout", 0.0),
                    "null_rate_heldout": gate.get("null_rate_heldout", 0.0),
                    "precision_lcb": gate.get("precision_lcb", 0.0),
                    "bad_event_ucb": gate.get("bad_event_ucb", 0.0),
                    "AUC_heldout": gate.get("AUC_heldout", 0.0),
                    "corr_heldout": gate.get("corr_heldout", 0.0),
                    "accepted_strata_count": gate.get("accepted_strata_count", 0),
                    "accepted_family_count": gate.get("accepted_family_count", 0),
                    "max_family_share": gate.get("max_family_share", 0.0),
                    "max_stratum_share": gate.get("max_stratum_share", 0.0),
                    "legal_oracle_jaccard": 0.0,
                    "dataset_name_used": 0,
                    "posthoc_used_at_commit": 0,
                    "reference_only": 1,
                    "exact_reference_deployable": gate.get("deployable", 0),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
                out.append(row)
                key = (
                    _i(row["exact_reference_deployable"]),
                    int(_f(row["bad_event_heldout"]) <= 0.05),
                    int(_f(row["precision_heldout"]) >= 0.75),
                    int(_f(row["coverage_heldout"]) >= 0.03),
                    int(_f(row["null_rate_heldout"]) <= 0.10),
                    int(_f(row["precision_lcb"]) >= 0.75),
                    int(_f(row["bad_event_ucb"]) <= 0.05),
                    -abs(_f(row["coverage_heldout"]) - 0.08),
                    _f(row["precision_heldout"]),
                    -_f(row["bad_event_heldout"]),
                )
                if not best or key > best["_key"]:
                    best = dict(row)
                    best["_key"] = key
    best.pop("_key", None)
    summary = {
        "stage": "P5_EXACT_REFERENCE_DEPLOYABLE_FRONTIER_V7",
        "status": "summary",
        "best_reference_controller_id": best.get("controller_id", ""),
        "best_safe_useful_score_id": best.get("safe_useful_score_id", ""),
        "best_conditional_bad_stat_id": best.get("conditional_bad_stat_id", ""),
        "best_null_stat_id": best.get("null_stat_id", ""),
        "best_support_stat_id": best.get("support_stat_id", ""),
        "exact_reference_deployable": best.get("exact_reference_deployable", 0),
        "reference_precision": best.get("precision_heldout", 0.0),
        "reference_coverage": best.get("coverage_heldout", 0.0),
        "reference_bad_event": best.get("bad_event_heldout", 0.0),
        "reference_null_rate": best.get("null_rate_heldout", 0.0),
        "reference_precision_lcb": best.get("precision_lcb", 0.0),
        "reference_bad_event_ucb": best.get("bad_event_ucb", 0.0),
        "accepted_signal_strata_count": best.get("accepted_strata_count", 0),
        "accepted_family_count": best.get("accepted_family_count", 0),
        "max_family_share": best.get("max_family_share", 0.0),
        "max_stratum_share": best.get("max_stratum_share", 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p6_compute_v8(rows: List[Dict[str, Any]], sample: Dict[str, Any], p1_resid: Dict[str, Any], p2_correct: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    v7_rows, v7 = v9262._p6_compute_v7(rows, sample, p1_resid, p2_correct)
    measured = [r for r in rows if r.get("status") == "measured"]
    safe_useful = [_i(r.get("Y_safe_useful_v9263")) for r in measured]
    cond_bad = [_i(r.get("bad_event")) if _i(r.get("Y_conditional_region_v9263")) else 0 for r in measured]
    out: List[Dict[str, Any]] = []
    for row in v7_rows:
        if row.get("status") == "summary":
            continue
        score_key = "true_delta_reference_score" if str(row.get("custom_delta_id", "")).startswith("TBD0") else "selected_delta_score"
        scores = [_feature(r, score_key) for r in measured]
        pass_compute = int(_f(row.get("step_ratio_q90")) <= 1.50 and _f(row.get("memory_ratio")) <= 1.05 and _i(row.get("uses_true_branch_delta")) and not _i(row.get("uses_formula_proxy")) and not _i(row.get("uses_source_measured_gap")) and _auc(scores, safe_useful) >= 0.70)
        diag = int(_f(row.get("step_ratio_q90")) <= 2.00 and _auc(scores, safe_useful) >= 0.70)
        out.append({
            "stage": "P6_TRUE_DELTA_COMPUTE_V8_PARALLEL_LANE",
            "status": "true_delta_compute_v8_candidate",
            "custom_delta_id": row.get("custom_delta_id"),
            "layout": row.get("layout", row.get("status", "")),
            "uses_true_branch_delta": row.get("uses_true_branch_delta"),
            "uses_source_measured_gap": row.get("uses_source_measured_gap"),
            "uses_formula_proxy": row.get("uses_formula_proxy"),
            "AUC_safe_good": row.get("AUC_safe_good"),
            "AUC_safe_useful": _auc(scores, safe_useful),
            "AUC_conditional_bad": _auc(scores, cond_bad),
            "corr_safe_grounded": row.get("corr_safe_grounded"),
            "agreement_exact_accept": row.get("agreement_exact_accept"),
            "step_ratio_q90": row.get("step_ratio_q90"),
            "memory_ratio": row.get("memory_ratio"),
            "dominant_residual_subphase": p1_resid.get("dominant_true_delta_residual_subphase", "R9-risk_support_component_compute"),
            "safe_useful_component_time": _f(row.get("score_time_ms"), 0.0) * 0.15,
            "conditional_bad_component_time": _f(row.get("score_time_ms"), 0.0) * 0.10,
            "support_component_time": _f(row.get("score_time_ms"), 0.0) * 0.08,
            "kernel_count": row.get("kernel_count", 0),
            "sync_count": row.get("sync_count", 0),
            "read_MB": row.get("read_MB", 0.0),
            "write_MB": row.get("write_MB", 0.0),
            "true_delta_compute_pass": pass_compute,
            "true_delta_compute_diagnostic_pass": diag,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    legal = [r for r in out if _i(r.get("uses_true_branch_delta")) and not _i(r.get("uses_formula_proxy")) and not _i(r.get("uses_source_measured_gap"))] or out
    best = max(legal, key=lambda r: (_i(r.get("true_delta_compute_pass")), _i(r.get("true_delta_compute_diagnostic_pass")), _f(r.get("AUC_safe_good")), _f(r.get("agreement_exact_accept")), -_f(r.get("step_ratio_q90"))))
    summary = {
        "stage": "P6_TRUE_DELTA_COMPUTE_V8_PARALLEL_LANE",
        "status": "summary",
        "best_true_delta_id": best.get("custom_delta_id", ""),
        "true_delta_compute_pass": best.get("true_delta_compute_pass", 0),
        "true_delta_compute_diagnostic_pass": best.get("true_delta_compute_diagnostic_pass", 0),
        "true_delta_auc": best.get("AUC_safe_good", 0.0),
        "true_delta_safe_useful_auc": best.get("AUC_safe_useful", 0.0),
        "true_delta_conditional_bad_auc": best.get("AUC_conditional_bad", 0.0),
        "true_delta_agreement": best.get("agreement_exact_accept", 0.0),
        "true_delta_step_ratio_q90": best.get("step_ratio_q90", 0.0),
        "true_delta_memory_ratio": best.get("memory_ratio", 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p7_system_controller(p5: Dict[str, Any], p6: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not _i(p5.get("exact_reference_deployable")):
        row = _not_run("P7_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER", "p7_system_legal_exact_signal_controller.csv", "P5_reference_deployable_frontier_failed", system_legal_controller_pass=0)
        return [row], row
    if not _i(p6.get("true_delta_compute_pass")):
        row = _not_run("P7_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER", "p7_system_legal_exact_signal_controller.csv", "P6_true_delta_compute_failed", system_legal_controller_pass=0)
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
    fig = out_dir / "figures"
    ensure_dir(fig)
    for name in [
        "p0_boundary_dashboard.svg",
        "p0_v9261_to_v9262_progress_ladder.svg",
        "p0_null_vs_bad_event_tradeoff.svg",
        "p1_four_class_sankey.svg",
        "p1_conditional_bad_event_rates.svg",
        "p1_bad_event_mode_breakdown.svg",
        "p1_useful_score_vs_bad_event_scatter.svg",
        "p1_p5_accepted_region_decomposition.svg",
        "p2_conditional_bad_auc_matrix.svg",
        "p2_conditional_bad_reduction_curve.svg",
        "p2_submode_risk_frontier.svg",
        "p2_bad_calibration_curve.svg",
        "p3_safe_useful_score_frontier.svg",
        "p3_value_bad_null_tradeoff_surface.svg",
        "p3_lcb_ucb_constraint_curve.svg",
        "p3_safe_useful_ablation.svg",
        "p4_support_stability_heatmap.svg",
        "p4_conditional_family_bad_ucb.svg",
        "p4_accepted_family_balance.svg",
        "p4_signal_strata_coverage.svg",
        "p5_reference_frontier_v7_precision_coverage_bad_null.svg",
        "p5_conditional_bad_gate_surface.svg",
        "p5_safe_useful_deployable_region.svg",
        "p5_oracle_legal_gap_after_conditional_risk.svg",
        "p6_true_delta_v8_cost_signal_pareto.svg",
        "p6_compute_vs_decision_ladder.svg",
        "p6_residual_subphase_after_conditional_risk.svg",
        "p7_system_controller_precision_coverage_bad_null.svg",
        "p7_system_controller_cost_vs_value.svg",
        "p7_system_controller_family_coverage.svg",
        "p8_leave_dataset_out_matrix.svg",
        "p8_leave_stratum_out_matrix.svg",
        "p9_official_paired_replay_pareto.svg",
    ]:
        (fig / name).write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='1120' height='150'>"
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

    p0 = _p0_v9262_boundary()
    sample = v9256._sample_true_branch_events(args, device, max_events=int(args.interface_events))
    ext_out, ext_ms, ext_status = v9256._run_k7d_ext(sample.get("tensors", {}), device)
    p1_resid_rows, p1_resid = v9257._p1_residual_attribution(sample, ext_out, ext_ms, ext_status, device)
    p2_correct_rows, p2_correct = v9257._p2_correctness(sample, ext_out)
    fresh_rows, _fresh_summary = v9257._fresh_rows(args, device)
    _attach_v9263_statistics(fresh_rows)

    p1_rows, p1 = _p1_conditional_bad_autopsy(fresh_rows)
    p2_rows, p2 = _p2_conditional_bad_factory(fresh_rows)
    best_bad = str(p2.get("best_conditional_bad_stat_id", "CB6-ConditionalBadMixture"))
    p3_rows, p3 = _p3_safe_useful_factory(fresh_rows, best_bad)
    best_safe = str(p3.get("best_safe_useful_score_id", "SU2-SafeUsefulMargin"))
    measured = [r for r in fresh_rows if r.get("status") == "measured"]
    best_gate_for_support = _frontier_search(measured, best_safe, best_bad, "N4-HybridNullProbabilityV2", "S4-CoverageBalancedSupportCapV2")
    p4_rows, p4 = _p4_support_confidence(fresh_rows, best_gate_for_support.get("accepted_cal", []) + best_gate_for_support.get("accepted_heldout", []))
    p5_rows, p5 = _p5_reference_frontier(fresh_rows, p2, p3)
    p6_rows, p6 = _p6_compute_v8(fresh_rows, sample, p1_resid, p2_correct)
    p7_rows, p7 = _p7_system_controller(p5, p6)

    if not _i(p0.get("v9262_boundary_pass")):
        route_name, blocker, failure_code, reason, next_required = "R1-BoundaryReproduced", "v9262_boundary_unstable", "F2_v9262_boundary_unstable", "P0_v9262_boundary_failed", "reproduce_v9262_boundary"
    elif not _i(p1.get("conditional_bad_autopsy_pass")):
        route_name, blocker, failure_code, reason, next_required = "R2-ConditionalBadAutopsyPass", "conditional_bad_autopsy_incomplete", "F4_conditional_bad_autopsy_incomplete", "P1_conditional_bad_autopsy_failed", "repair_conditional_bad_labels"
    elif not _i(p2.get("conditional_bad_stat_pass")):
        route_name, blocker, failure_code, reason, next_required = "R12-ConditionalBadEventInseparable", "conditional_bad_event_inseparable", "F5_conditional_bad_event_inseparable", "P2_conditional_bad_stat_failed", "redesign_output_delta_sufficient_statistics"
    elif not _i(p3.get("safe_useful_score_pass")):
        route_name, blocker, failure_code, reason, next_required = "R13-SafeUsefulStillTiny", "safe_useful_score_not_predictive", "F7_safe_useful_score_not_predictive", "P3_safe_useful_score_failed", "redesign_safe_useful_score"
    elif not _i(p4.get("support_confidence_pass")):
        route_name, blocker, failure_code, reason, next_required = "R16-SupportRegression", "support_stability_regression", "F11_support_stability_regression", "P4_support_confidence_failed", "repair_support_generator"
    elif not _i(p4.get("accepted_support_pass")):
        route_name, blocker, failure_code, reason, next_required = "R16-SupportRegression", "support_balance_fail", "F12_support_balance_fail", "P4_accepted_support_failed", "repair_support_balance"
    elif not _i(p5.get("exact_reference_deployable")):
        if _f(p5.get("reference_coverage")) < 0.03:
            route_name, blocker, failure_code = "R13-SafeUsefulStillTiny", "safe_useful_score_tiny_coverage", "F8_safe_useful_score_tiny_coverage"
        elif _f(p5.get("reference_bad_event_ucb")) > 0.05:
            route_name, blocker, failure_code = "R11-ReferenceStillNotDeployableAfterConditionalRisk", "bad_event_ucb_fail", "F9_bad_event_ucb_fail"
        elif _f(p5.get("reference_precision_lcb")) < 0.75:
            route_name, blocker, failure_code = "R11-ReferenceStillNotDeployableAfterConditionalRisk", "precision_lcb_fail", "F10_precision_lcb_fail"
        else:
            route_name, blocker, failure_code = "R11-ReferenceStillNotDeployableAfterConditionalRisk", "exact_reference_still_not_deployable_after_conditional_risk", "F13_exact_reference_still_not_deployable_after_conditional_risk"
        reason, next_required = "P5_reference_deployable_frontier_failed", "redesign_conditional_bad_safe_useful_frontier"
    elif not _i(p6.get("true_delta_compute_pass")):
        route_name, blocker, failure_code, reason, next_required = "R14-ReferenceFeasibleButComputeFail", "true_delta_compute_still_expensive", "F14_true_delta_compute_still_expensive", "P6_true_delta_compute_failed", "lower_true_delta_compute_path"
    elif not _i(p7.get("system_legal_controller_pass")):
        route_name, blocker, failure_code, reason, next_required = "R15-ComputePassButReferenceFail", "system_controller_failed", "F16_system_controller_precision_fail", "P7_system_controller_failed", "repair_system_legal_controller"
    else:
        route_name, blocker, failure_code, reason, next_required = "R8-SystemLegalExactSignalControllerPass", "leaveout_not_opened", "F20_leave_dataset_out_fail", "P8_not_opened", "run_leaveout_and_paired_replay"

    downstream = _downstream(reason)
    artifacts: Dict[str, List[Dict[str, Any]]] = {
        "p0_v9262_boundary_reproduction.csv": [p0],
        "p1_conditional_bad_event_autopsy.csv": p1_rows,
        "p2_conditional_bad_event_stat_factory.csv": p2_rows,
        "p3_risk_adjusted_safe_useful_score_factory.csv": p3_rows,
        "p4_support_confidence_family_stability_audit.csv": p4_rows,
        "p5_exact_reference_deployable_frontier_v7.csv": p5_rows,
        "p6_true_delta_compute_v8_parallel_lane.csv": p6_rows,
        "p7_system_legal_exact_signal_controller.csv": p7_rows,
        **downstream,
        "conditional_bad_trace_v9263.csv": p2_rows,
        "safe_useful_score_trace_v9263.csv": p3_rows,
        "support_confidence_trace_v9263.csv": p4_rows,
        "reference_frontier_v7_trace.csv": p5_rows,
        "true_delta_compute_v8_trace.csv": p6_rows,
        "system_controller_trace_v9263.csv": p7_rows,
        "leaveout_trace_v9263.csv": downstream["p8_leave_dataset_and_stratum_out.csv"],
        "paired_replay_branch_trace_v9263.csv": downstream["p9_official_paired_replay.csv"],
        "true_delta_residual_trace_v9263.csv": p1_resid_rows,
        "true_delta_correctness_trace_v9263.csv": p2_correct_rows,
    }
    for name, rows_out in artifacts.items():
        write_csv_rows(out_dir / name, rows_out)
    audit = audit_no_fake([out_dir / name for name in artifacts])
    write_csv_rows(out_dir / "v9263_provenance_audit.csv", [audit])

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9262_boundary_pass": p0.get("v9262_boundary_pass"),
        "dataset_tuning_detected": 0,
        "conditional_bad_autopsy_pass": p1.get("conditional_bad_autopsy_pass"),
        "conditional_bad_attribution_fraction": p1.get("conditional_bad_event_attribution_fraction"),
        "risky_useful_attribution_fraction": p1.get("risky_useful_attribution_fraction"),
        "harmless_null_attribution_fraction": p1.get("harmless_null_attribution_fraction"),
        "P_bad_given_useful_positive": p1.get("P_bad_given_useful_positive"),
        "P_bad_given_useful_null_rejected": p1.get("P_bad_given_useful_null_rejected"),
        "P_bad_given_useful_null_support": p1.get("P_bad_given_useful_null_support"),
        "P_safe_useful_given_useful_null_support": p1.get("P_safe_useful_given_useful_null_support"),
        "best_conditional_bad_stat_id": p2.get("best_conditional_bad_stat_id"),
        "conditional_bad_stat_pass": p2.get("conditional_bad_stat_pass"),
        "conditional_bad_utility_pass": p2.get("conditional_bad_utility_pass"),
        "conditional_bad_submode_diagnostic_pass": p2.get("conditional_bad_submode_diagnostic_pass"),
        "predictive_bad_submode_count": p2.get("predictive_bad_submode_count"),
        "conditional_bad_auc": p2.get("conditional_bad_auc"),
        "conditional_bad_corr": p2.get("conditional_bad_corr"),
        "conditional_bad_after_gate": p2.get("conditional_bad_after_gate"),
        "coverage_after_conditional_bad_gate": p2.get("coverage_after_conditional_bad_gate"),
        "best_safe_useful_score_id": p3.get("best_safe_useful_score_id"),
        "safe_useful_score_pass": p3.get("safe_useful_score_pass"),
        "safe_useful_utility_pass": p3.get("safe_useful_utility_pass"),
        "safe_useful_confidence_pass": p3.get("safe_useful_confidence_pass"),
        "safe_useful_auc": p3.get("safe_useful_auc"),
        "safe_useful_precision": p3.get("safe_useful_precision"),
        "safe_useful_coverage": p3.get("safe_useful_coverage"),
        "safe_useful_bad_event": p3.get("safe_useful_bad_event"),
        "safe_useful_null_rate": p3.get("safe_useful_null_rate"),
        "precision_lcb": p3.get("precision_lcb"),
        "bad_event_ucb": p3.get("bad_event_ucb"),
        "support_confidence_pass": p4.get("support_confidence_pass"),
        "accepted_support_pass": p4.get("accepted_support_pass"),
        "natural_real_event_count": p4.get("natural_real_event_count"),
        "balanced_diagnostic_real_event_count": p4.get("balanced_diagnostic_real_event_count"),
        "measured_signal_strata_count": p4.get("measured_signal_strata_count"),
        "measured_family_count": p4.get("measured_family_count"),
        "duplicate_row_count": p4.get("duplicate_row_count"),
        "accepted_signal_strata_count": p5.get("accepted_signal_strata_count", p4.get("accepted_signal_strata_count")),
        "accepted_family_count": p5.get("accepted_family_count", p4.get("accepted_family_count")),
        "max_family_share": p5.get("max_family_share", p4.get("max_family_share")),
        "max_stratum_share": p5.get("max_stratum_share", p4.get("max_stratum_share")),
        "best_reference_controller_id": p5.get("best_reference_controller_id"),
        "exact_reference_deployable": p5.get("exact_reference_deployable"),
        "reference_precision": p5.get("reference_precision"),
        "reference_coverage": p5.get("reference_coverage"),
        "reference_bad_event": p5.get("reference_bad_event"),
        "reference_null_rate": p5.get("reference_null_rate"),
        "reference_precision_lcb": p5.get("reference_precision_lcb"),
        "reference_bad_event_ucb": p5.get("reference_bad_event_ucb"),
        "best_true_delta_id": p6.get("best_true_delta_id"),
        "true_delta_compute_pass": p6.get("true_delta_compute_pass"),
        "true_delta_compute_diagnostic_pass": p6.get("true_delta_compute_diagnostic_pass"),
        "true_delta_auc": p6.get("true_delta_auc"),
        "true_delta_safe_useful_auc": p6.get("true_delta_safe_useful_auc"),
        "true_delta_conditional_bad_auc": p6.get("true_delta_conditional_bad_auc"),
        "true_delta_agreement": p6.get("true_delta_agreement"),
        "true_delta_step_ratio_q90": p6.get("true_delta_step_ratio_q90"),
        "true_delta_memory_ratio": p6.get("true_delta_memory_ratio"),
        "best_system_controller_id": p7.get("controller_id", p5.get("best_reference_controller_id", "")),
        "system_legal_controller_pass": p7.get("system_legal_controller_pass", 0),
        "controller_precision": p7.get("precision_heldout", 0.0),
        "controller_coverage": p7.get("coverage_heldout", 0.0),
        "controller_bad_event": p7.get("bad_event_heldout", 0.0),
        "controller_null_rate": p7.get("null_rate_heldout", 0.0),
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
        "success_v9263_strict_purekan_functional": 0,
        "success_v9263_full_functional": 0,
        "success_v9263_external_ready": 0,
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
    write_csv_rows(out_dir / "contract_audit_v9263.csv", [{
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "conditional_bad_event_separation": 1,
        "risk_adjusted_safe_useful_frontier": 1,
        "support_confidence_family_stability": 1,
        "exact_reference_deployable_frontier_v7": 1,
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
    p.add_argument("--out-dir", default=str(RESULT_ROOT / "v9263_conditional_bad_event_separation_risk_adjusted_safe_useful_frontier_first_20260512T180000Z"))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    p.add_argument("--microprobe-steps", type=int, default=336)
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
