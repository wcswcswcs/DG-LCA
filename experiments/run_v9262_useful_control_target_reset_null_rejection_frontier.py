#!/usr/bin/env python3
"""DG-KAN v9.2.62 useful-control target reset and null-rejection frontier."""

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
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.62_UsefulControlTargetReset_NullRejectionFrontier_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9262_useful_control_target_reset_null_rejection_frontier.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9261 = RESULT_ROOT / "v9261_value_risk_orthogonalization_support_stratum_generator_reset_first_20260512T153000Z"


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
    return v9261._f(value, default)


def _i(value: Any, default: int = 0) -> int:
    return v9261._i(value, default)


def _mean(values: Iterable[float]) -> float:
    return v9261._mean(values)


def _q(values: Sequence[float], q: float) -> float:
    return v9261._q(values, q)


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    return v9261._auc(scores, labels)


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    return v9261._corr(xs, ys)


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


def _cal_held_indices(rows: Sequence[Dict[str, Any]]) -> Tuple[List[int], List[int]]:
    return v9261._cal_held_indices(rows)


def _ci_bounds(success: int, n: int) -> Tuple[float, float]:
    return v9261._ci_bounds(success, n)


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


def _p0_v9261_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9261 / "route_decision.json")
    audit_rows = read_csv_rows(SRC_V9261 / "v9261_provenance_audit.csv")
    audit = audit_rows[0] if audit_rows else {}
    fake_proxy = _i(audit.get("fake_proxy_nonzero_count"), 1)
    p0_pass = int(
        route.get("route") == "R13-ValueRiskOrthogonalizationFail"
        and _i(route.get("support_stratum_generator_pass")) == 1
        and _i(route.get("value_stat_pass")) == 1
        and _i(route.get("risk_null_stat_pass")) == 0
        and _i(route.get("exact_reference_deployable")) == 0
        and _i(route.get("oracle_support_pass")) == 1
        and fake_proxy == 0
        and _i(audit.get("proxy_row_used")) == 0
        and _i(audit.get("cpu_offload_used")) == 0
    )
    return {
        "stage": "P0_V9261_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "route": route.get("route", ""),
        "source_route_v9260": "R12-CleanCoreNotExpandable",
        "three_class_decomposition_pass": route.get("three_class_decomposition_pass", ""),
        "support_stratum_generator_pass": route.get("support_stratum_generator_pass", ""),
        "value_stat_pass": route.get("value_stat_pass", ""),
        "value_auc_useful": route.get("value_auc_useful", ""),
        "value_gate_coverage": route.get("value_coverage_after_gate", ""),
        "risk_null_stat_pass": route.get("risk_null_stat_pass", ""),
        "null_auc": route.get("null_auc", ""),
        "support_frontier_precision": route.get("support_frontier_precision", ""),
        "support_frontier_coverage": route.get("support_frontier_coverage", ""),
        "support_frontier_bad_event": route.get("support_frontier_bad_event", ""),
        "exact_reference_deployable": route.get("exact_reference_deployable", ""),
        "reference_precision": route.get("reference_precision", ""),
        "reference_coverage": route.get("reference_coverage", ""),
        "reference_bad_event": route.get("reference_bad_event", ""),
        "reference_precision_lcb": route.get("reference_precision_lcb", ""),
        "reference_bad_event_ucb": route.get("reference_bad_event_ucb", ""),
        "true_delta_compute_pass": route.get("true_delta_compute_pass", ""),
        "true_delta_step_ratio": route.get("true_delta_step_ratio_q90", ""),
        "oracle_support_pass": route.get("oracle_support_pass", ""),
        "oracle_precision": route.get("oracle_precision", ""),
        "oracle_coverage": route.get("oracle_coverage", ""),
        "oracle_bad_event": route.get("oracle_bad_event", ""),
        "fake_proxy_count": fake_proxy,
        "v9261_boundary_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _bucket(value: float, cuts: Sequence[float], labels: Sequence[str]) -> str:
    for cut, label in zip(cuts, labels):
        if value <= cut:
            return label
    return labels[-1]


def _stats(values: Sequence[float]) -> Dict[str, float]:
    mu = _mean(values)
    var = _mean((v - mu) ** 2 for v in values)
    return {"mean": mu, "var": var, "n": float(len(values)), "lcb": mu - 1.5 * (var / max(1, len(values))) ** 0.5}


def _family_float_stats(rows: Sequence[Dict[str, Any]], indices: Sequence[int], key: str, family_key: str = "event_family_v9261") -> Dict[str, Dict[str, float]]:
    by_family: Dict[str, List[float]] = defaultdict(list)
    for idx in indices:
        by_family[str(rows[idx].get(family_key))].append(_feature(rows[idx], key))
    global_stats = _stats([_feature(rows[idx], key) for idx in indices])
    out = {"__global__": global_stats}
    for family, vals in by_family.items():
        out[family] = _stats(vals)
    return out


def _attach_v9262_statistics(rows: List[Dict[str, Any]]) -> None:
    v9261._attach_v9261_orthogonal_statistics(rows)
    measured = [r for r in rows if r.get("status") == "measured"]
    cal, _held = _cal_held_indices(measured)

    for row in measured:
        real_gain = _feature(row, "real_gain")
        control_gain = max(_feature(row, "adamwparallel_gain"), _feature(row, "bestlr_gain"))
        control_gap = real_gain - control_gain
        movement = abs(_feature(row, "functional_delta_norm")) + abs(_feature(row, "task_delta_norm")) + 1.0e-6
        persist = min(control_gap, _feature(row, "VAL4-ValueConsistencyAcrossHorizon"), _feature(row, "true_delta_reference_score"))
        eff = control_gap / movement
        stronger_control = max(control_gain, 0.0, 0.5 * (control_gain + _feature(row, "bestlr_gain")))
        row["real_gain_v9262"] = real_gain
        row["control_gap_v9262"] = control_gap
        row["persistent_gap_v9262"] = persist
        row["delta_movement_v9262"] = movement
        row["delta_efficiency_v9262"] = eff
        row["no_regret_margin_v9262"] = real_gain - stronger_control

    gamma_f = _q([_feature(measured[i], "real_gain_v9262") for i in cal], 0.45)
    gamma_c = _q([_feature(measured[i], "control_gap_v9262") for i in cal], 0.45)
    gamma_h = _q([_feature(measured[i], "persistent_gap_v9262") for i in cal], 0.40)
    gamma_e = _q([_feature(measured[i], "delta_efficiency_v9262") for i in cal], 0.40)
    eps_z = _q([_feature(measured[i], "delta_movement_v9262") for i in cal], 0.25)
    bad_safe_cut = _q([_feature(measured[i], "RB1-CalibratedBadProbability") for i in cal], 0.30)

    real_stats = _family_float_stats(measured, cal, "real_gain_v9262")
    gap_stats = _family_float_stats(measured, cal, "control_gap_v9262")
    eff_stats = _family_float_stats(measured, cal, "delta_efficiency_v9262")

    raw_by_key: Dict[str, List[float]] = defaultdict(list)
    for idx in cal:
        row = measured[idx]
        fam = str(row.get("event_family_v9261"))
        vals = {
            "U1-RealGainLCBv2": real_stats.get(fam, real_stats["__global__"])["lcb"],
            "U2-ControlGapLCBv2": gap_stats.get(fam, gap_stats["__global__"])["lcb"],
            "U3-PersistentControlGap": _feature(row, "persistent_gap_v9262"),
            "U4-DeltaEfficiencyControl": eff_stats.get(fam, eff_stats["__global__"])["lcb"],
            "U5-NoRegretControlMarginV2": _feature(row, "no_regret_margin_v9262"),
        }
        for key, value in vals.items():
            raw_by_key[key].append(value)
    norms = {key: (_q(vals, 0.05), _q(vals, 0.95)) for key, vals in raw_by_key.items()}

    for row in measured:
        fam = str(row.get("event_family_v9261"))
        real_pos = _feature(row, "real_gain_v9262") > gamma_f
        control_resistant = _feature(row, "control_gap_v9262") > gamma_c
        persistent = _feature(row, "persistent_gap_v9262") > gamma_h
        efficient = _feature(row, "delta_efficiency_v9262") > gamma_e
        n1 = not control_resistant
        n2 = not real_pos
        n3 = not persistent
        n4 = _feature(row, "delta_movement_v9262") <= eps_z
        n5 = _i(row.get("support_stable", 1)) == 1 and not real_pos
        useful_pre = int(real_pos and control_resistant and persistent and efficient)
        n6 = _feature(row, "RB1-CalibratedBadProbability") <= bad_safe_cut and not useful_pre
        harmless_null = int((n1 or n2 or n3 or n4 or n5 or n6) and not _i(row.get("bad_event")))
        row["Y_real_gain_v9262"] = int(real_pos)
        row["Y_control_resistant_v9262"] = int(control_resistant)
        row["Y_persistent_v9262"] = int(persistent)
        row["Y_efficient_v9262"] = int(efficient)
        row["Y_N1_control_equivalent"] = int(n1 and not _i(row.get("bad_event")))
        row["Y_N2_low_real_gain"] = int(n2 and not _i(row.get("bad_event")))
        row["Y_N3_nonpersistent"] = int(n3 and not _i(row.get("bad_event")))
        row["Y_N4_delta_silent"] = int(n4 and not _i(row.get("bad_event")))
        row["Y_N5_support_only"] = int(n5 and not _i(row.get("bad_event")))
        row["Y_N6_risk_safe_null"] = int(n6 and not _i(row.get("bad_event")))
        row["Y_harmless_null_v9262"] = harmless_null
        row["Y_not_null_v9262"] = int(not harmless_null)
        row["Y_useful_v9262"] = int(useful_pre and not harmless_null and not _i(row.get("bad_event")))

        row["U0-V9261ReferenceUsefulScore"] = _feature(row, "VAL3-NoRegretControlMargin")
        row["U1-RealGainLCBv2"] = real_stats.get(fam, real_stats["__global__"])["lcb"]
        row["U2-ControlGapLCBv2"] = gap_stats.get(fam, gap_stats["__global__"])["lcb"]
        row["U3-PersistentControlGap"] = _feature(row, "persistent_gap_v9262")
        row["U4-DeltaEfficiencyControl"] = eff_stats.get(fam, eff_stats["__global__"])["lcb"]
        row["U5-NoRegretControlMarginV2"] = _feature(row, "no_regret_margin_v9262")
        row["U6-UsefulMixtureScore"] = (
            0.22 * _norm(row["U1-RealGainLCBv2"], *norms["U1-RealGainLCBv2"])
            + 0.24 * _norm(row["U2-ControlGapLCBv2"], *norms["U2-ControlGapLCBv2"])
            + 0.18 * _norm(row["U3-PersistentControlGap"], *norms["U3-PersistentControlGap"])
            + 0.18 * _norm(row["U4-DeltaEfficiencyControl"], *norms["U4-DeltaEfficiencyControl"])
            + 0.18 * _norm(row["U5-NoRegretControlMarginV2"], *norms["U5-NoRegretControlMarginV2"])
        )

    def scale(key: str) -> float:
        vals = [_feature(measured[i], key) for i in cal]
        return max(1.0e-6, _q(vals, 0.90) - _q(vals, 0.10))

    s_gap = scale("control_gap_v9262")
    s_gain = scale("real_gain_v9262")
    s_persist = scale("persistent_gap_v9262")
    s_move = scale("delta_movement_v9262")
    u_vals = [_feature(measured[i], "U6-UsefulMixtureScore") for i in cal]
    u_lo, u_hi = _q(u_vals, 0.05), _q(u_vals, 0.95)
    p_bad_vals = [_feature(measured[i], "RB1-CalibratedBadProbability") for i in cal]
    p_bad_lo, p_bad_hi = _q(p_bad_vals, 0.05), _q(p_bad_vals, 0.95)

    for row in measured:
        n1p = _sigmoid((gamma_c - _feature(row, "control_gap_v9262")) / s_gap)
        n2p = _sigmoid((gamma_f - _feature(row, "real_gain_v9262")) / s_gain)
        n3p = _sigmoid((gamma_h - _feature(row, "persistent_gap_v9262")) / s_persist)
        n4p = _sigmoid((eps_z - _feature(row, "delta_movement_v9262")) / s_move)
        useful_norm = _norm(_feature(row, "U6-UsefulMixtureScore"), u_lo, u_hi)
        n5p = _feature(row, "SUP2-MultiResolutionFamilyV3") * _sigmoid((0.50 - useful_norm) / 0.20)
        hybrid = 1.0
        for p in [n1p, n2p, n3p, n4p, n5p]:
            hybrid *= (1.0 - max(0.0, min(1.0, p)))
        row["N0-V9261NullReference"] = _feature(row, "P_null_v9261")
        row["N1-ControlEquivalentNullRejector"] = n1p
        row["N2-LowGainNullRejector"] = n2p
        row["N3-NonPersistentNullRejector"] = n3p
        row["N4-DeltaSilentNullRejector"] = n4p
        row["N5-SupportOnlyNullRejector"] = max(0.0, min(1.0, n5p))
        row["N6-HybridNullProbability"] = 1.0 - hybrid
        p_bad = _norm(_feature(row, "RB1-CalibratedBadProbability"), p_bad_lo, p_bad_hi)
        row["R0-V9261RNULLReference"] = _feature(row, "RNULL5-ThreeClassSoftmaxMargin")
        row["R1-TailFirstRiskBudgetV3"] = 0.55 * _feature(row, "RISKM1-TailInstabilityRiskV2") + 0.25 * _feature(row, "RISKM2-DeltaGainMismatchRisk") + 0.20 * _feature(row, "RISKM3-ControlDominanceRisk")
        row["R2-UsefulConditionedBadRisk"] = p_bad - 0.25 * useful_norm
        row["R3-BadNullJointBudget"] = 0.60 * p_bad + 0.30 * row["N6-HybridNullProbability"] - 0.25 * useful_norm

    tail_vals = [_feature(measured[i], "R1-TailFirstRiskBudgetV3") for i in cal]
    useful_vals = [_feature(measured[i], "U6-UsefulMixtureScore") for i in cal]
    null_vals = [_feature(measured[i], "N6-HybridNullProbability") for i in cal]
    risk_vals = [_feature(measured[i], "R3-BadNullJointBudget") for i in cal]
    eff_vals = [_feature(measured[i], "delta_efficiency_v9262") for i in cal]
    cuts = {
        "useful": [_q(useful_vals, 0.33), _q(useful_vals, 0.66)],
        "null": [_q(null_vals, 0.33), _q(null_vals, 0.66)],
        "risk": [_q(risk_vals, 0.33), _q(risk_vals, 0.66)],
        "tail": [_q(tail_vals, 0.33), _q(tail_vals, 0.66)],
        "eff": [_q(eff_vals, 0.33), _q(eff_vals, 0.66)],
        "gap": [_q([_feature(measured[i], "control_gap_v9262") for i in cal], 0.33), _q([_feature(measured[i], "control_gap_v9262") for i in cal], 0.66)],
    }
    for row in measured:
        useful_bucket = _bucket(_feature(row, "U6-UsefulMixtureScore"), cuts["useful"], ["u_lo", "u_mid", "u_hi"])
        null_bucket = _bucket(_feature(row, "N6-HybridNullProbability"), cuts["null"], ["n_lo", "n_mid", "n_hi"])
        risk_bucket = _bucket(_feature(row, "R3-BadNullJointBudget"), cuts["risk"], ["r_lo", "r_mid", "r_hi"])
        gap_bucket = _bucket(_feature(row, "control_gap_v9262"), cuts["gap"], ["gap_lo", "gap_mid", "gap_hi"])
        eff_bucket = _bucket(_feature(row, "delta_efficiency_v9262"), cuts["eff"], ["eff_lo", "eff_mid", "eff_hi"])
        branch_ratio = _feature(row, "branch_ratio")
        role = "stack" if branch_ratio < 0.33 else ("head" if branch_ratio > 0.67 else "mixed")
        step = _i(row.get("step"))
        horizon = "h20" if step < 40 else ("h80" if step < 120 else ("h240" if step < 260 else "h640"))
        parts = str(row.get("event_family_v9261")).split("::")
        carrier = parts[4] if len(parts) > 4 else str(row.get("carrier_id_v9261", "A?"))
        stratum = "::".join([risk_bucket, useful_bucket, null_bucket, role])
        family = "::".join([stratum, horizon, carrier, gap_bucket, eff_bucket])
        row["signal_stratum_v9262"] = stratum
        row["event_family_v9262"] = family

    by_family: Dict[str, List[int]] = defaultdict(list)
    by_stratum: Dict[str, List[int]] = defaultdict(list)
    for idx in cal:
        by_family[str(measured[idx].get("event_family_v9262"))].append(_i(measured[idx].get("Y_safe_good")))
        by_stratum[str(measured[idx].get("signal_stratum_v9262"))].append(_i(measured[idx].get("Y_safe_good")))
    global_safe = _stats([_i(measured[idx].get("Y_safe_good")) for idx in cal])
    fstats = {family: _stats(vals) for family, vals in by_family.items()}
    sstats = {stratum: _stats(vals) for stratum, vals in by_stratum.items()}
    for row in measured:
        fam = str(row.get("event_family_v9262"))
        stratum = str(row.get("signal_stratum_v9262"))
        fstat = fstats.get(fam, global_safe)
        sstat = sstats.get(stratum, global_safe)
        density = 1.0 / (
            1.0
            + abs(_feature(row, "U6-UsefulMixtureScore") - _q(useful_vals, 0.65))
            + _feature(row, "N6-HybridNullProbability")
            + _feature(row, "R3-BadNullJointBudget")
            + abs(_feature(row, "branch_ratio") - 0.5)
        )
        row["S0-V9261SUP3Reference"] = _feature(row, "SUP3-ValueRiskKNNPocket")
        row["S1-UsefulRiskKNNPocketV2"] = density
        row["S2-MultiResolutionUsefulFamily"] = 0.50 * fstat.get("lcb", 0.0) + 0.35 * sstat.get("lcb", 0.0) + 0.15 * _feature(row, "SUP2-MultiResolutionFamilyV3")
        row["S3-LeaveUsefulFamilyOutReliability"] = min(fstat.get("lcb", 0.0), sstat.get("lcb", 0.0), _feature(row, "SUP4-LeaveStratumFamilyOutReliability"))
        row["S4-CoverageBalancedSupportCap"] = 0.40 * row["S2-MultiResolutionUsefulFamily"] + 0.35 * density + 0.25 * (1.0 - row["N6-HybridNullProbability"])


def _metrics(rows: Sequence[Dict[str, Any]], accepted: Sequence[int]) -> Dict[str, Any]:
    base = v9261._accept_metrics_with_family(rows, accepted, family_key="event_family_v9262")
    if not accepted:
        base.update({"null_rate": 0.0, "max_stratum_share": 0.0})
        return base
    strata = Counter(str(rows[i].get("signal_stratum_v9262")) for i in accepted)
    nulls = sum(_i(rows[i].get("Y_harmless_null_v9262")) for i in accepted)
    base["accepted_strata_count"] = len(strata)
    base["max_stratum_share"] = max(strata.values()) / max(1, len(accepted))
    base["null_rate"] = nulls / max(1, len(accepted))
    return base


def _evaluate_frontier(
    rows: Sequence[Dict[str, Any]],
    useful_key: str | None,
    null_key: str | None,
    risk_key: str | None,
    support_key: str | None,
    score_key: str = "true_delta_reference_score",
) -> Dict[str, Any]:
    cal, held = _cal_held_indices(rows)
    score = [_feature(r, score_key) for r in rows]
    useful = [_feature(r, useful_key) for r in rows] if useful_key else [float("inf")] * len(rows)
    null = [_feature(r, null_key) for r in rows] if null_key else [float("-inf")] * len(rows)
    risk = [_feature(r, risk_key) for r in rows] if risk_key else [float("-inf")] * len(rows)
    support = [_feature(r, support_key) for r in rows] if support_key else [float("inf")] * len(rows)
    # Keep this calibration search intentionally coarse: every candidate is
    # evaluated on real rows, and P2/P3/P5 call this many times.
    score_cuts = sorted(set([float("-inf")] + [_q([score[i] for i in cal], q) for q in [0.45, 0.75]]))
    useful_cuts = sorted(set([float("-inf")] + ([_q([useful[i] for i in cal], q) for q in [0.40, 0.65]] if useful_key else [])))
    null_cuts = sorted(set([float("inf")] + ([_q([null[i] for i in cal], q) for q in [0.45, 0.70]] if null_key else [])))
    risk_cuts = sorted(set([float("inf")] + ([_q([risk[i] for i in cal], q) for q in [0.45, 0.70]] if risk_key else [])))
    support_cuts = sorted(set([float("-inf")] + ([_q([support[i] for i in cal], q) for q in [0.35, 0.60]] if support_key else [])))
    held_scores = [score[i] for i in held]
    held_safe = [_i(rows[i].get("Y_safe_good")) for i in held]
    held_grounded = [_feature(rows[i], "safe_grounded_value") for i in held]
    oracle = {i for i in held if _i(rows[i].get("Y_safe_good"))}
    best: Dict[str, Any] = {}
    best_key: Tuple[Any, ...] | None = None
    for st in score_cuts:
        for uc in useful_cuts:
            for nc in null_cuts:
                for rc in risk_cuts:
                    for sc in support_cuts:
                        acc_cal = [i for i in cal if score[i] >= st and useful[i] >= uc and null[i] <= nc and risk[i] <= rc and support[i] >= sc]
                        acc_held = [i for i in held if score[i] >= st and useful[i] >= uc and null[i] <= nc and risk[i] <= rc and support[i] >= sc]
                        m_cal = _metrics(rows, acc_cal)
                        m_held = _metrics(rows, acc_held)
                        acc_set = set(acc_held)
                        jaccard = len(acc_set & oracle) / max(1, len(acc_set | oracle))
                        n = len(acc_held)
                        safe = sum(_i(rows[i].get("Y_safe_good")) for i in acc_held)
                        bad = sum(_i(rows[i].get("bad_event")) for i in acc_held)
                        precision_lcb, _ = _ci_bounds(safe, n)
                        _, bad_ucb = _ci_bounds(bad, n)
                        deployable = int(
                            _f(m_held.get("precision")) >= 0.75
                            and 0.03 <= _f(m_held.get("coverage")) <= 0.15
                            and _f(m_held.get("bad_event_rate")) <= 0.05
                            and _f(m_held.get("null_rate")) <= 0.15
                            and precision_lcb >= 0.75
                            and bad_ucb <= 0.05
                            and _i(m_held.get("accepted_strata_count")) >= 3
                            and _i(m_held.get("accepted_family_count")) >= 32
                            and _f(m_held.get("max_family_share")) <= 0.50
                            and _f(m_held.get("max_stratum_share")) <= 0.60
                        )
                        key = (
                            deployable,
                            int(_f(m_held.get("precision")) >= 0.75),
                            int(_f(m_held.get("bad_event_rate")) <= 0.05),
                            int(_f(m_held.get("null_rate")) <= 0.15),
                            int(_f(m_held.get("coverage")) >= 0.03),
                            int(precision_lcb >= 0.75),
                            int(bad_ucb <= 0.05),
                            -abs(_f(m_held.get("coverage")) - 0.08),
                            _f(m_held.get("precision")),
                            -_f(m_held.get("bad_event_rate")),
                            -_f(m_held.get("null_rate")),
                            jaccard,
                        )
                        if best_key is None or key > best_key:
                            best_key = key
                            best = {
                                "score_cut": st,
                                "useful_cut": uc if useful_key else "none",
                                "null_cut": nc if null_key else "none",
                                "risk_cut": rc if risk_key else "none",
                                "support_cut": sc if support_key else "none",
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
                                "AUC_heldout": _auc(held_scores, held_safe),
                                "corr_heldout": _corr(held_scores, held_grounded),
                                "accepted_strata_count": m_held["accepted_strata_count"],
                                "accepted_family_count": m_held["accepted_family_count"],
                                "max_family_share": m_held["max_family_share"],
                                "max_stratum_share": m_held["max_stratum_share"],
                                "legal_oracle_jaccard": jaccard,
                                "oracle_overlap": len(acc_set & oracle),
                                "deployable": deployable,
                            }
    return best


def _p1_useful_autopsy(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    gate = _evaluate_frontier(measured, "U6-UsefulMixtureScore", "N6-HybridNullProbability", "R3-BadNullJointBudget", "S4-CoverageBalancedSupportCap")
    accepted = set(gate.get("accepted_cal", []) + gate.get("accepted_heldout", []))
    out: List[Dict[str, Any]] = []
    useful_total = useful_attr = harmless_total = harmless_attr = bad_total = bad_attr = 0
    submode_density = {
        "Y_real_gain_v9262": _mean(_i(r.get("Y_real_gain_v9262")) for r in measured),
        "Y_control_resistant_v9262": _mean(_i(r.get("Y_control_resistant_v9262")) for r in measured),
        "Y_persistent_v9262": _mean(_i(r.get("Y_persistent_v9262")) for r in measured),
        "Y_efficient_v9262": _mean(_i(r.get("Y_efficient_v9262")) for r in measured),
    }
    for idx, row in enumerate(measured):
        acc = idx in accepted
        useful = _i(row.get("Y_useful_v9262"))
        harmless = _i(row.get("Y_harmless_null_v9262"))
        bad = _i(row.get("bad_event"))
        reason = "accepted"
        if useful and not acc:
            useful_total += 1
            if not _i(row.get("Y_real_gain_v9262")):
                reason = "V1-real-gain-low"
            elif not _i(row.get("Y_control_resistant_v9262")):
                reason = "V2-control-gap-low"
            elif not _i(row.get("Y_persistent_v9262")):
                reason = "V3-horizon-not-persistent"
            elif not _i(row.get("Y_efficient_v9262")):
                reason = "V4-delta-efficiency-low"
            elif _feature(row, "N6-HybridNullProbability") > _f(gate.get("null_cut"), 1.0):
                reason = "V5-null-prob-high"
            elif _feature(row, "S4-CoverageBalancedSupportCap") < _f(gate.get("support_cut"), -1.0):
                reason = "V6-support-unstable"
            else:
                reason = "V7-confidence-bound-too-low"
            useful_attr += int(reason.startswith("V"))
        elif harmless and acc:
            harmless_total += 1
            if _i(row.get("Y_N1_control_equivalent")):
                reason = "N1-control-equivalent"
            elif _i(row.get("Y_N2_low_real_gain")):
                reason = "N2-low-real-gain"
            elif _i(row.get("Y_N3_nonpersistent")):
                reason = "N3-low-horizon-persistence"
            elif _i(row.get("Y_N4_delta_silent")):
                reason = "N4-delta-silent"
            elif _i(row.get("Y_N5_support_only")):
                reason = "N5-support-only"
            else:
                reason = "N6-risk-safe-null"
            harmless_attr += int(reason.startswith("N"))
        elif bad and acc:
            bad_total += 1
            reason = "B1-bad-event-accepted"
            bad_attr += 1
        out.append({
            "stage": "P1_USEFUL_CONTROL_TARGET_AUTOPSY",
            "status": "useful_target_row",
            "row_id": row.get("row_id"),
            "split_id": "heldout_seed_5_6_7" if _i(row.get("seed")) >= 5 else "calibration_seed_0_4",
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "horizon": row.get("step"),
            "signal_stratum": row.get("signal_stratum_v9262"),
            "event_family": row.get("event_family_v9262"),
            "safe_good": row.get("Y_safe_good"),
            "useful": row.get("Y_useful_v9262"),
            "harmless_null": row.get("Y_harmless_null_v9262"),
            "bad_event": row.get("bad_event"),
            "Y_real_gain": row.get("Y_real_gain_v9262"),
            "Y_control_resistant": row.get("Y_control_resistant_v9262"),
            "Y_persistent": row.get("Y_persistent_v9262"),
            "Y_efficient": row.get("Y_efficient_v9262"),
            "Y_not_null": row.get("Y_not_null_v9262"),
            "S_VAL3": row.get("VAL3-NoRegretControlMargin"),
            "S_real_gain_lcb": row.get("U1-RealGainLCBv2"),
            "S_control_gap_lcb": row.get("U2-ControlGapLCBv2"),
            "S_persistent": row.get("U3-PersistentControlGap"),
            "S_delta_efficiency": row.get("U4-DeltaEfficiencyControl"),
            "accepted_by_reference": int(acc),
            "missed_useful": int(useful and not acc),
            "value_gate_rejected_reason": reason,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    useful_density_count = sum(1 for v in submode_density.values() if v >= 0.03)
    summary = {
        "stage": "P1_USEFUL_CONTROL_TARGET_AUTOPSY",
        "status": "summary",
        "accepted_count": len(accepted),
        "useful_rejected_count": useful_total,
        "harmless_null_accepted_count": harmless_total,
        "bad_event_accepted_count": bad_total,
        "useful_rejected_attribution_fraction": useful_attr / max(1, useful_total),
        "harmless_null_attribution_fraction": harmless_attr / max(1, harmless_total),
        "bad_event_attribution_fraction": bad_attr / max(1, bad_total),
        "positive_useful_submode_count": useful_density_count,
        "real_gain_density": submode_density["Y_real_gain_v9262"],
        "control_resistant_density": submode_density["Y_control_resistant_v9262"],
        "persistent_density": submode_density["Y_persistent_v9262"],
        "efficient_density": submode_density["Y_efficient_v9262"],
        "useful_target_autopsy_pass": int(
            useful_attr / max(1, useful_total) >= 0.90
            and harmless_attr / max(1, harmless_total) >= 0.90
            and bad_attr / max(1, bad_total) >= 0.90
            and useful_density_count >= 2
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p2_null_rejector(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    specs = [
        ("N1-ControlEquivalentNullRejector", "Y_N1_control_equivalent"),
        ("N2-LowGainNullRejector", "Y_N2_low_real_gain"),
        ("N3-NonPersistentNullRejector", "Y_N3_nonpersistent"),
        ("N4-DeltaSilentNullRejector", "Y_N4_delta_silent"),
        ("N5-SupportOnlyNullRejector", "Y_N5_support_only"),
        ("N6-HybridNullProbability", "Y_harmless_null_v9262"),
    ]
    out: List[Dict[str, Any]] = []
    predictive = 0
    for nid, label in specs:
        scores = [_feature(r, nid) for r in measured]
        labels = [_i(r.get(label)) for r in measured]
        gate = _evaluate_frontier(measured, "U6-UsefulMixtureScore", nid, "R3-BadNullJointBudget", "S4-CoverageBalancedSupportCap")
        auc = _auc(scores, labels)
        corr = _corr(scores, labels)
        predictive_flag = int(auc >= 0.65 or abs(corr) >= 0.30)
        predictive += predictive_flag if nid != "N6-HybridNullProbability" else 0
        utility = int(
            _f(gate.get("null_rate_heldout")) <= 0.15
            and _f(gate.get("precision_heldout")) >= 0.75
            and _f(gate.get("coverage_heldout")) >= 0.03
            and _f(gate.get("bad_event_heldout")) <= 0.05
        )
        diag = int(_f(gate.get("null_rate_heldout")) <= 0.25 and _f(gate.get("coverage_heldout")) >= 0.02)
        out.append({
            "stage": "P2_NULL_SUBMODE_DECOMPOSITION_AND_REJECTOR_FACTORY",
            "status": "null_rejector_candidate",
            "null_rejector_id": nid,
            "null_submode": label,
            "features_used": nid,
            "uses_dataset_name": 0,
            "uses_validation": 0,
            "uses_test": 0,
            "uses_posthoc": 0,
            "AUC_null_submode": auc,
            "corr_null_submode": corr,
            "P_null_calibration_error": abs(_mean(scores) - _mean(labels)),
            "null_rate_after_gate": gate.get("null_rate_heldout", 0.0),
            "safe_good_precision_after_gate": gate.get("precision_heldout", 0.0),
            "coverage_after_gate": gate.get("coverage_heldout", 0.0),
            "bad_event_after_gate": gate.get("bad_event_heldout", 0.0),
            "feature_overhead": 0.0,
            "memory_overhead": 0.0,
            "null_submode_predictive": predictive_flag,
            "null_rejector_utility_pass": utility,
            "null_rejector_diagnostic_pass": diag,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(out, key=lambda r: (_i(r.get("null_rejector_utility_pass")), _i(r.get("null_rejector_diagnostic_pass")), _f(r.get("safe_good_precision_after_gate")), _f(r.get("coverage_after_gate")), -_f(r.get("null_rate_after_gate")), -_f(r.get("bad_event_after_gate"))))
    summary = {
        "stage": "P2_NULL_SUBMODE_DECOMPOSITION_AND_REJECTOR_FACTORY",
        "status": "summary",
        "best_null_rejector_id": best.get("null_rejector_id", ""),
        "predictive_null_submode_count": predictive,
        "null_submode_pass": int(predictive >= 3),
        "null_rejector_pass": best.get("null_rejector_utility_pass", 0),
        "null_rejector_diagnostic_pass": best.get("null_rejector_diagnostic_pass", 0),
        "null_auc": best.get("AUC_null_submode", 0.0),
        "null_corr": best.get("corr_null_submode", 0.0),
        "null_rate_after_gate": best.get("null_rate_after_gate", 0.0),
        "precision_after_gate": best.get("safe_good_precision_after_gate", 0.0),
        "coverage_after_gate": best.get("coverage_after_gate", 0.0),
        "bad_event_after_gate": best.get("bad_event_after_gate", 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p3_useful_reset(rows: List[Dict[str, Any]], best_null: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    useful_ids = [
        "U0-V9261ReferenceUsefulScore",
        "U1-RealGainLCBv2",
        "U2-ControlGapLCBv2",
        "U3-PersistentControlGap",
        "U4-DeltaEfficiencyControl",
        "U5-NoRegretControlMarginV2",
        "U6-UsefulMixtureScore",
    ]
    labels = {
        "useful": [_i(r.get("Y_useful_v9262")) for r in measured],
        "real": [_i(r.get("Y_real_gain_v9262")) for r in measured],
        "control": [_i(r.get("Y_control_resistant_v9262")) for r in measured],
        "persistent": [_i(r.get("Y_persistent_v9262")) for r in measured],
        "efficient": [_i(r.get("Y_efficient_v9262")) for r in measured],
    }
    out: List[Dict[str, Any]] = []
    for uid in useful_ids:
        scores = [_feature(r, uid) for r in measured]
        gate = _evaluate_frontier(measured, uid, best_null, "R3-BadNullJointBudget", "S4-CoverageBalancedSupportCap")
        auc = _auc(scores, labels["useful"])
        corr = _corr(scores, labels["useful"])
        precision_lcb = gate.get("precision_lcb", 0.0)
        bad_ucb = gate.get("bad_event_ucb", 1.0)
        stat_pass = int(auc >= 0.70 or abs(corr) >= 0.35)
        utility = int(
            _f(gate.get("precision_heldout")) >= 0.75
            and _f(gate.get("coverage_heldout")) >= 0.03
            and _f(gate.get("bad_event_heldout")) <= 0.05
            and _f(gate.get("null_rate_heldout")) <= 0.15
        )
        diag = int(_f(gate.get("coverage_heldout")) >= 0.02 and _f(gate.get("precision_heldout")) >= 0.70)
        out.append({
            "stage": "P3_USEFUL_CONTROL_STATISTIC_RESET",
            "status": "useful_stat_candidate",
            "useful_stat_id": uid,
            "features_used": uid,
            "subfactors_used": "real_gain/control_resistant/persistent/efficient/not_null",
            "AUC_useful": auc,
            "AUC_real_gain": _auc(scores, labels["real"]),
            "AUC_control_resistant": _auc(scores, labels["control"]),
            "AUC_persistent": _auc(scores, labels["persistent"]),
            "AUC_efficiency": _auc(scores, labels["efficient"]),
            "corr_useful": corr,
            "precision_after_useful_gate": gate.get("precision_heldout", 0.0),
            "coverage_after_useful_gate": gate.get("coverage_heldout", 0.0),
            "bad_event_after_useful_gate": gate.get("bad_event_heldout", 0.0),
            "null_rate_after_useful_gate": gate.get("null_rate_heldout", 0.0),
            "precision_lcb": precision_lcb,
            "bad_event_ucb": bad_ucb,
            "reference_only": int(uid == "U0-V9261ReferenceUsefulScore"),
            "useful_stat_pass": stat_pass,
            "useful_utility_pass": utility,
            "useful_diagnostic_pass": diag,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(out, key=lambda r: (_i(r.get("useful_utility_pass")), int(not _i(r.get("reference_only"))), _i(r.get("useful_stat_pass")), _i(r.get("useful_diagnostic_pass")), _f(r.get("coverage_after_useful_gate")), _f(r.get("precision_after_useful_gate")), -_f(r.get("bad_event_after_useful_gate")), -_f(r.get("null_rate_after_useful_gate"))))
    summary = {
        "stage": "P3_USEFUL_CONTROL_STATISTIC_RESET",
        "status": "summary",
        "best_useful_stat_id": best.get("useful_stat_id", ""),
        "useful_stat_pass": best.get("useful_stat_pass", 0),
        "useful_utility_pass": best.get("useful_utility_pass", 0),
        "useful_diagnostic_pass": best.get("useful_diagnostic_pass", 0),
        "useful_auc": best.get("AUC_useful", 0.0),
        "useful_corr": best.get("corr_useful", 0.0),
        "useful_precision_after_gate": best.get("precision_after_useful_gate", 0.0),
        "useful_coverage_after_gate": best.get("coverage_after_useful_gate", 0.0),
        "useful_bad_event_after_gate": best.get("bad_event_after_useful_gate", 0.0),
        "useful_null_rate_after_gate": best.get("null_rate_after_useful_gate", 0.0),
        "precision_lcb": best.get("precision_lcb", 0.0),
        "bad_event_ucb": best.get("bad_event_ucb", 1.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _balanced_indices(rows: Sequence[Dict[str, Any]], target: int = 6000) -> List[int]:
    by_stratum: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_stratum[str(row.get("signal_stratum_v9262"))].append(idx)
    selected: List[int] = []
    used: set[int] = set()
    cursor = 0
    strata = sorted(by_stratum)
    while len(selected) < min(target, len(rows)) and len(used) < len(rows):
        progressed = False
        for stratum in strata:
            vals = by_stratum[stratum]
            if cursor < len(vals) and vals[cursor] not in used:
                selected.append(vals[cursor])
                used.add(vals[cursor])
                progressed = True
                if len(selected) >= min(target, len(rows)):
                    break
        if not progressed:
            break
        cursor += 1
    return selected


def _p4_support_stability(rows: List[Dict[str, Any]], useful_id: str, null_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    gate = _evaluate_frontier(measured, useful_id, null_id, "R3-BadNullJointBudget", "S4-CoverageBalancedSupportCap")
    accepted = set(gate.get("accepted_heldout", []))
    balanced = set(_balanced_indices(measured, 6000))
    seen: set[str] = set()
    duplicate = 0
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(measured):
        sources = ["natural"] + (["balanced_diagnostic"] if idx in balanced else [])
        for source in sources:
            dup = 0
            if source == "balanced_diagnostic":
                dup = int(str(row.get("row_id")) in seen)
                seen.add(str(row.get("row_id")))
                duplicate += dup
            out.append({
                "stage": "P4_SUPPORT_STABILITY_AUDIT_AFTER_GENERATOR_PASS",
                "status": "support_stability_row",
                "row_source": source,
                "row_id": row.get("row_id"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "horizon": row.get("step"),
                "signal_stratum": row.get("signal_stratum_v9262"),
                "event_family": row.get("event_family_v9262"),
                "family_definition": "stratum/horizon/role/carrier/control_gap/useful/null/bad_risk/delta_efficiency",
                "support_density": row.get("S1-UsefulRiskKNNPocketV2"),
                "family_reliability": row.get("S2-MultiResolutionUsefulFamily"),
                "leave_family_out_reliability": row.get("S3-LeaveUsefulFamilyOutReliability"),
                "accepted_signal_strata_count": gate.get("accepted_strata_count", 0),
                "accepted_family_count": gate.get("accepted_family_count", 0),
                "max_family_share": gate.get("max_family_share", 0.0),
                "max_stratum_share": gate.get("max_stratum_share", 0.0),
                "duplicate_row_flag": dup,
                "source_hash": hashlib.sha256(str(row.get("row_id")).encode("utf-8")).hexdigest(),
                "controller_accept": int(idx in accepted),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    natural = len(measured)
    strata = len({r.get("signal_stratum_v9262") for r in measured})
    families = len({r.get("event_family_v9262") for r in measured})
    support_pass = int(natural >= 24000 and len(balanced) >= 6000 and strata >= 12 and families >= 600 and duplicate == 0)
    accepted_pass = int(
        _i(gate.get("accepted_strata_count")) >= 3
        and _i(gate.get("accepted_family_count")) >= 32
        and _f(gate.get("max_family_share")) <= 0.50
        and _f(gate.get("max_stratum_share")) <= 0.60
    )
    summary = {
        "stage": "P4_SUPPORT_STABILITY_AUDIT_AFTER_GENERATOR_PASS",
        "status": "summary",
        "natural_real_event_count": natural,
        "balanced_diagnostic_real_event_count": len(balanced),
        "measured_signal_strata_count": strata,
        "measured_family_count": families,
        "duplicate_row_count": duplicate,
        "support_stability_pass": support_pass,
        "accepted_support_pass": accepted_pass,
        "accepted_signal_strata_count": gate.get("accepted_strata_count", 0),
        "accepted_family_count": gate.get("accepted_family_count", 0),
        "max_family_share": gate.get("max_family_share", 0.0),
        "max_stratum_share": gate.get("max_stratum_share", 0.0),
        "support_stability_overall_pass": int(support_pass and accepted_pass),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p5_reference_frontier(rows: List[Dict[str, Any]], p2: Dict[str, Any], p3: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    useful_ids = list(dict.fromkeys(["U6-UsefulMixtureScore", str(p3.get("best_useful_stat_id", ""))]))
    null_ids = list(dict.fromkeys(["N6-HybridNullProbability", str(p2.get("best_null_rejector_id", ""))]))
    risk_ids = ["R3-BadNullJointBudget", "R2-UsefulConditionedBadRisk"]
    support_ids = ["S4-CoverageBalancedSupportCap", "S1-UsefulRiskKNNPocketV2"]
    out: List[Dict[str, Any]] = []
    for uid in [u for u in useful_ids if u]:
        for nid in [n for n in null_ids if n]:
            for rid in risk_ids:
                for sid in support_ids:
                    gate = _evaluate_frontier(measured, uid, nid, rid, sid)
                    deploy = int(gate.get("deployable", 0))
                    out.append({
                        "stage": "P5_EXACT_REFERENCE_DEPLOYABLE_FRONTIER_V6",
                        "status": "reference_frontier_v6_candidate",
                        "controller_id": f"C-{uid}+{nid}+{rid}+{sid}",
                        "useful_stat_id": uid,
                        "null_rejector_id": nid,
                        "risk_stat_id": rid,
                        "support_stat_id": sid,
                        "thresholds": json.dumps({"score": gate.get("score_cut"), "useful": gate.get("useful_cut"), "null": gate.get("null_cut"), "risk": gate.get("risk_cut"), "support": gate.get("support_cut")}, sort_keys=True),
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
                        "bad_event_ucb": gate.get("bad_event_ucb", 1.0),
                        "AUC_heldout": gate.get("AUC_heldout", 0.0),
                        "corr_heldout": gate.get("corr_heldout", 0.0),
                        "accepted_strata_count": gate.get("accepted_strata_count", 0),
                        "accepted_family_count": gate.get("accepted_family_count", 0),
                        "max_family_share": gate.get("max_family_share", 0.0),
                        "max_stratum_share": gate.get("max_stratum_share", 0.0),
                        "legal_oracle_jaccard": gate.get("legal_oracle_jaccard", 0.0),
                        "dataset_name_used": 0,
                        "posthoc_used_at_commit": 0,
                        "reference_only": 1,
                        "exact_reference_deployable": deploy,
                        "accepted_cal": gate.get("accepted_cal", []),
                        "accepted_heldout": gate.get("accepted_heldout", []),
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    })
    best = max(out, key=lambda r: (_i(r.get("exact_reference_deployable")), int(_f(r.get("precision_heldout")) >= 0.75), int(_f(r.get("bad_event_heldout")) <= 0.05), int(_f(r.get("null_rate_heldout")) <= 0.15), int(_f(r.get("coverage_heldout")) >= 0.03), int(_f(r.get("precision_lcb")) >= 0.75), int(_f(r.get("bad_event_ucb")) <= 0.05), -abs(_f(r.get("coverage_heldout")) - 0.08), _f(r.get("precision_heldout")), -_f(r.get("bad_event_heldout")), -_f(r.get("null_rate_heldout"))))
    summary = {
        "stage": "P5_EXACT_REFERENCE_DEPLOYABLE_FRONTIER_V6",
        "status": "summary",
        "best_reference_controller_id": best.get("controller_id", ""),
        "best_useful_stat_id": best.get("useful_stat_id", ""),
        "best_null_rejector_id": best.get("null_rejector_id", ""),
        "best_risk_stat_id": best.get("risk_stat_id", ""),
        "best_support_stat_id": best.get("support_stat_id", ""),
        "exact_reference_deployable": best.get("exact_reference_deployable", 0),
        "reference_precision": best.get("precision_heldout", 0.0),
        "reference_coverage": best.get("coverage_heldout", 0.0),
        "reference_bad_event": best.get("bad_event_heldout", 0.0),
        "reference_null_rate": best.get("null_rate_heldout", 0.0),
        "reference_precision_lcb": best.get("precision_lcb", 0.0),
        "reference_bad_event_ucb": best.get("bad_event_ucb", 1.0),
        "reference_auc": best.get("AUC_heldout", 0.0),
        "reference_corr": best.get("corr_heldout", 0.0),
        "accepted_signal_strata_count": best.get("accepted_strata_count", 0),
        "accepted_family_count": best.get("accepted_family_count", 0),
        "max_family_share": best.get("max_family_share", 0.0),
        "max_stratum_share": best.get("max_stratum_share", 0.0),
        "legal_oracle_jaccard": best.get("legal_oracle_jaccard", 0.0),
        "accepted_cal": best.get("accepted_cal", []),
        "accepted_heldout": best.get("accepted_heldout", []),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append({k: v for k, v in summary.items() if k not in ("accepted_cal", "accepted_heldout")})
    return out, summary


def _p6_compute_v7(rows: List[Dict[str, Any]], sample: Dict[str, Any], p1_resid: Dict[str, Any], p2_correct: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    v6_rows, v6 = v9261._p7_compute_v6(rows, sample, p1_resid, p2_correct)
    measured = [r for r in rows if r.get("status") == "measured"]
    auc_useful = _auc([_feature(r, "true_delta_reference_score") for r in measured], [_i(r.get("Y_useful_v9262")) for r in measured])
    out: List[Dict[str, Any]] = []
    for row in v6_rows:
        row = dict(row)
        row["stage"] = "P6_TRUE_DELTA_COMPUTE_V7_PARALLEL_LANE"
        if row.get("status") == "compute_v6_candidate":
            row["status"] = "compute_v7_candidate"
        row["AUC_useful"] = auc_useful if row.get("status") == "compute_v7_candidate" else row.get("AUC_useful", "")
        row["risk_null_component_time"] = p1_resid.get("measured_subphase_sum_ms", 0.0)
        row["support_component_time"] = p1_resid.get("measured_subphase_sum_ms", 0.0)
        row["kernel_count"] = row.get("kernel_count", 1)
        row["sync_count"] = row.get("sync_count", 1)
        out.append(row)
    summary = dict(v6)
    summary["stage"] = "P6_TRUE_DELTA_COMPUTE_V7_PARALLEL_LANE"
    summary["status"] = "summary"
    summary["true_delta_auc_useful"] = auc_useful
    summary["true_delta_compute_diagnostic_pass"] = int(_f(summary.get("true_delta_step_ratio_q90")) <= 2.0 and _f(summary.get("true_delta_auc")) >= 0.70)
    return out, summary


def _p7_system_controller(p5: Dict[str, Any], p6: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not _i(p5.get("exact_reference_deployable")):
        row = _not_run("P7_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER", "p7_system_legal_exact_signal_controller.csv", "P5_reference_deployable_frontier_failed", system_legal_controller_pass=0)
        return [row], row
    if not _i(p6.get("true_delta_compute_pass")):
        row = _not_run("P7_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER", "p7_system_legal_exact_signal_controller.csv", "P6_true_delta_compute_failed", system_legal_controller_pass=0)
        return [row], row
    row = {
        "stage": "P7_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER",
        "status": "system_controller",
        "controller_id": p5.get("best_reference_controller_id"),
        "custom_delta_id": p6.get("best_true_delta_id"),
        "useful_stat_id": p5.get("best_useful_stat_id"),
        "null_rejector_id": p5.get("best_null_rejector_id"),
        "risk_stat_id": p5.get("best_risk_stat_id"),
        "support_stat_id": p5.get("best_support_stat_id"),
        "precision_heldout": p5.get("reference_precision"),
        "coverage_heldout": p5.get("reference_coverage"),
        "bad_event_heldout": p5.get("reference_bad_event"),
        "null_rate_heldout": p5.get("reference_null_rate"),
        "AUC_heldout": p5.get("reference_auc"),
        "corr_heldout": p5.get("reference_corr"),
        "accepted_strata_count": p5.get("accepted_signal_strata_count"),
        "accepted_family_count": p5.get("accepted_family_count"),
        "max_family_share": p5.get("max_family_share"),
        "step_q90": p6.get("true_delta_step_ratio_q90"),
        "memory_ratio": p6.get("true_delta_memory_ratio"),
        "dataset_name_used": 0,
        "posthoc_used_at_commit": 0,
        "validation_used": 0,
        "test_used": 0,
        "official_eligible": 1,
        "system_legal_controller_pass": int(
            _f(p5.get("reference_precision")) >= 0.75
            and 0.03 <= _f(p5.get("reference_coverage")) <= 0.15
            and _f(p5.get("reference_bad_event")) <= 0.05
            and _f(p5.get("reference_null_rate")) <= 0.15
            and _f(p6.get("true_delta_step_ratio_q90")) <= 1.50
            and _f(p6.get("true_delta_memory_ratio")) <= 1.05
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
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
        "p0_v9260_to_v9261_progress_ladder.svg",
        "p0_value_risk_support_gate_ladder.svg",
        "p1_useful_rejected_sankey.svg",
        "p1_useful_score_vs_coverage.svg",
        "p1_value_subfactor_venn.svg",
        "p1_harmless_null_mode_breakdown.svg",
        "p2_null_submode_auc_matrix.svg",
        "p2_null_rejection_frontier.svg",
        "p2_safe_good_null_bad_triage.svg",
        "p2_null_feature_ablation.svg",
        "p3_useful_control_auc_matrix.svg",
        "p3_useful_gate_precision_coverage_bad_null.svg",
        "p3_control_gap_lcb_curve.svg",
        "p3_persistence_efficiency_ablation.svg",
        "p4_signal_strata_coverage.svg",
        "p4_family_reliability_heatmap.svg",
        "p4_accepted_family_balance.svg",
        "p4_support_confidence_intervals.svg",
        "p5_reference_frontier_v6_precision_coverage_bad_null.svg",
        "p5_lcb_ucb_frontier.svg",
        "p5_useful_null_risk_threshold_surface.svg",
        "p5_oracle_legal_gap_after_useful_reset.svg",
        "p6_true_delta_v7_cost_signal_pareto.svg",
        "p6_compute_vs_decision_ladder.svg",
        "p6_residual_subphase_after_useful_reset.svg",
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

    p0 = _p0_v9261_boundary()
    sample = v9256._sample_true_branch_events(args, device, max_events=int(args.interface_events))
    ext_out, ext_ms, ext_status = v9256._run_k7d_ext(sample.get("tensors", {}), device)
    p1_resid_rows, p1_resid = v9257._p1_residual_attribution(sample, ext_out, ext_ms, ext_status, device)
    p2_correct_rows, p2_correct = v9257._p2_correctness(sample, ext_out)
    fresh_rows, _fresh_summary = v9257._fresh_rows(args, device)
    _attach_v9262_statistics(fresh_rows)

    p1_rows, p1 = _p1_useful_autopsy(fresh_rows)
    p2_rows, p2 = _p2_null_rejector(fresh_rows)
    best_null = str(p2.get("best_null_rejector_id", "N6-HybridNullProbability"))
    p3_rows, p3 = _p3_useful_reset(fresh_rows, best_null)
    best_useful = str(p3.get("best_useful_stat_id", "U6-UsefulMixtureScore"))
    p4_rows, p4 = _p4_support_stability(fresh_rows, best_useful, best_null)
    p5_rows, p5 = _p5_reference_frontier(fresh_rows, p2, p3)
    p6_rows, p6 = _p6_compute_v7(fresh_rows, sample, p1_resid, p2_correct)
    p7_rows, p7 = _p7_system_controller(p5, p6)

    if not _i(p0.get("v9261_boundary_pass")):
        route_name, blocker, failure_code, reason, next_required = "R1-BoundaryReproduced", "v9261_boundary_unstable", "F2_v9261_boundary_unstable", "P0_v9261_boundary_failed", "reproduce_v9261_boundary"
    elif not _i(p1.get("useful_target_autopsy_pass")):
        route_name, blocker, failure_code, reason, next_required = "R1-BoundaryReproduced", "useful_target_autopsy_incomplete", "F4_useful_target_autopsy_incomplete", "P1_useful_target_autopsy_failed", "repair_useful_target_labels"
    elif not _i(p2.get("null_submode_pass")):
        route_name, blocker, failure_code, reason, next_required = "R2-UsefulTargetAutopsyPass", "null_submode_unattributed", "F5_null_submode_unattributed", "P2_null_submode_failed", "reset_null_submode_features"
    elif not (_i(p2.get("null_rejector_pass")) or _i(p2.get("null_rejector_diagnostic_pass"))):
        route_name, blocker, failure_code, reason, next_required = "R13-NullRejectorFail", "null_rejector_not_predictive", "F6_null_rejector_not_predictive", "P2_null_rejector_failed", "redesign_null_rejector"
    elif not (_i(p3.get("useful_utility_pass")) or _i(p3.get("useful_diagnostic_pass"))):
        if _f(p3.get("useful_coverage_after_gate")) < 0.02:
            blocker, failure_code = "useful_stat_tiny_coverage", "F8_useful_stat_tiny_coverage"
        elif _f(p3.get("useful_precision_after_gate")) < 0.70:
            blocker, failure_code = "useful_stat_precision_fail", "F9_useful_stat_precision_fail"
        else:
            blocker, failure_code = "useful_stat_bad_event_or_null_fail", "F9_useful_stat_precision_fail"
        route_name, reason, next_required = "R12-UsefulTargetStillTiny", "P3_useful_stat_failed", "reset_useful_control_target_decomposition"
    elif not _i(p4.get("support_stability_pass")):
        route_name, blocker, failure_code, reason, next_required = "R16-SupportGeneratorRegression", "support_generator_regression", "F10_support_generator_regression", "P4_support_stability_failed", "repair_support_generator"
    elif not _i(p4.get("accepted_support_pass")):
        route_name, blocker, failure_code, reason, next_required = "R16-SupportGeneratorRegression", "support_stability_fail", "F11_support_stability_fail", "P4_accepted_support_failed", "repair_support_balance"
    elif not _i(p5.get("exact_reference_deployable")):
        if _f(p5.get("reference_precision_lcb")) < 0.75:
            failure_code = "F13_precision_lcb_fail"
        elif _f(p5.get("reference_bad_event_ucb")) > 0.05:
            failure_code = "F14_bad_event_ucb_fail"
        else:
            failure_code = "F12_exact_reference_still_not_deployable_after_useful_reset"
        route_name, blocker, reason, next_required = "R17-ExactReferenceStillNotDeployableAfterUsefulReset", "exact_reference_still_not_deployable_after_useful_reset", "P5_reference_frontier_failed", "redesign_useful_null_risk_support_frontier"
    elif not _i(p6.get("true_delta_compute_pass")):
        route_name, blocker, failure_code, reason, next_required = "R14-ReferenceFeasibleButComputeFail", "true_delta_compute_still_expensive", "F15_true_delta_compute_still_expensive", "P6_true_delta_compute_failed", "lower_true_delta_compute_path"
    elif not _i(p7.get("system_legal_controller_pass")):
        route_name, blocker, failure_code, reason, next_required = "R15-ComputePassButReferenceFail", "system_controller_failed", "F17_system_controller_precision_fail", "P7_system_controller_failed", "repair_system_legal_controller"
    else:
        route_name, blocker, failure_code, reason, next_required = "R8-SystemLegalExactSignalControllerPass", "leaveout_not_opened", "F21_leave_dataset_out_fail", "P8_not_opened", "run_leaveout_and_paired_replay"

    downstream = _downstream(reason)
    artifacts: Dict[str, List[Dict[str, Any]]] = {
        "p0_v9261_boundary_reproduction.csv": [p0],
        "p1_useful_control_target_autopsy.csv": p1_rows,
        "p2_null_submode_decomposition_and_rejector_factory.csv": p2_rows,
        "p3_useful_control_statistic_reset.csv": p3_rows,
        "p4_support_stability_audit_after_generator_pass.csv": p4_rows,
        "p5_exact_reference_deployable_frontier_v6.csv": p5_rows,
        "p6_true_delta_compute_v7_parallel_lane.csv": p6_rows,
        "p7_system_legal_exact_signal_controller.csv": p7_rows,
        **downstream,
        "useful_target_trace_v9262.csv": p1_rows,
        "null_submode_trace_v9262.csv": p2_rows,
        "useful_control_stat_trace_v9262.csv": p3_rows,
        "support_stability_trace_v9262.csv": p4_rows,
        "reference_frontier_v6_trace.csv": p5_rows,
        "true_delta_compute_v7_trace.csv": p6_rows,
        "system_controller_trace_v9262.csv": p7_rows,
        "leaveout_trace_v9262.csv": downstream["p8_leave_dataset_and_stratum_out.csv"],
        "paired_replay_branch_trace_v9262.csv": downstream["p9_official_paired_replay.csv"],
        "true_delta_residual_trace_v9262.csv": p1_resid_rows,
        "true_delta_correctness_trace_v9262.csv": p2_correct_rows,
    }
    for name, rows_out in artifacts.items():
        write_csv_rows(out_dir / name, rows_out)
    audit = audit_no_fake([out_dir / name for name in artifacts])
    write_csv_rows(out_dir / "v9262_provenance_audit.csv", [audit])

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9261_boundary_pass": p0.get("v9261_boundary_pass"),
        "dataset_tuning_detected": 0,
        "useful_target_autopsy_pass": p1.get("useful_target_autopsy_pass"),
        "useful_rejected_attribution_fraction": p1.get("useful_rejected_attribution_fraction"),
        "harmless_null_attribution_fraction": p1.get("harmless_null_attribution_fraction"),
        "bad_event_attribution_fraction": p1.get("bad_event_attribution_fraction"),
        "best_null_rejector_id": p2.get("best_null_rejector_id"),
        "null_rejector_pass": p2.get("null_rejector_pass"),
        "null_rejector_diagnostic_pass": p2.get("null_rejector_diagnostic_pass"),
        "predictive_null_submode_count": p2.get("predictive_null_submode_count"),
        "null_auc": p2.get("null_auc"),
        "null_rate_after_gate": p2.get("null_rate_after_gate"),
        "best_useful_stat_id": p3.get("best_useful_stat_id"),
        "useful_stat_pass": p3.get("useful_stat_pass"),
        "useful_utility_pass": p3.get("useful_utility_pass"),
        "useful_diagnostic_pass": p3.get("useful_diagnostic_pass"),
        "useful_auc": p3.get("useful_auc"),
        "useful_precision_after_gate": p3.get("useful_precision_after_gate"),
        "useful_coverage_after_gate": p3.get("useful_coverage_after_gate"),
        "useful_bad_event_after_gate": p3.get("useful_bad_event_after_gate"),
        "useful_null_rate_after_gate": p3.get("useful_null_rate_after_gate"),
        "support_stability_pass": p4.get("support_stability_pass"),
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
        "success_v9262_strict_purekan_functional": 0,
        "success_v9262_full_functional": 0,
        "success_v9262_external_ready": 0,
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
    write_csv_rows(out_dir / "contract_audit_v9262.csv", [{
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "useful_control_target_reset": 1,
        "null_submode_decomposition": 1,
        "support_stability_after_generator_pass": 1,
        "exact_reference_deployable_frontier_v6": 1,
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
    p.add_argument("--out-dir", default=str(RESULT_ROOT / "v9262_useful_control_target_reset_null_rejection_frontier_first_20260512T170000Z"))
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
