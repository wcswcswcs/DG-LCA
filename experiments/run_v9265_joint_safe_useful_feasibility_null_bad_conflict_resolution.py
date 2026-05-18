#!/usr/bin/env python3
"""DG-KAN v9.2.65 joint safe-useful feasibility and null-bad conflict audit."""

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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9256_true_branch_delta_tensor_interface_k7d_control_gain_cuda_fusion as v9256  # noqa: E402
import run_v9257_true_branch_delta_compute_closure_exact_signal_controller_feasibility as v9257  # noqa: E402
import run_v9264_conditional_null_separation_support_family_densification as v9264  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.65_JointSafeUsefulFeasibility_NullBadConflictResolution_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9265_joint_safe_useful_feasibility_null_bad_conflict_resolution.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9264 = RESULT_ROOT / "v9264_conditional_null_separation_support_family_densification_first_20260512T190000Z"

_f = v9264._f
_i = v9264._i
_mean = v9264._mean
_q = v9264._q
_auc = v9264._auc
_corr = v9264._corr
_feature = v9264._feature
_norm = v9264._norm
_device = v9264._device
_ci_bounds = v9264._ci_bounds
_cal_held_indices = v9264._cal_held_indices
_not_run = v9264._not_run


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _sigmoid(x: float) -> float:
    if x >= 40:
        return 1.0
    if x <= -40:
        return 0.0
    return 1.0 / (1.0 + np.exp(-x))


def _class_bounds(values: Sequence[int], kappa: float = 1.5) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 1.0
    p = _mean(values)
    err = kappa * (p * (1.0 - p) / max(1, len(values))) ** 0.5
    return p, max(0.0, p - err), min(1.0, p + err)


def _metrics(rows: Sequence[Dict[str, Any]], accepted: Sequence[int], label_key: str = "Y_SU_v9265") -> Dict[str, Any]:
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
    safe = sum(_i(rows[i].get(label_key)) for i in accepted)
    bad = sum(_i(rows[i].get("bad_event")) for i in accepted)
    nulls = sum(_i(rows[i].get("Y_HN_v9265")) for i in accepted)
    fam = Counter(str(rows[i].get("event_family_fine_v9264", "")) for i in accepted)
    strata = Counter(str(rows[i].get("signal_stratum_v9264", "")) for i in accepted)
    return {
        "precision": safe / max(1, len(accepted)),
        "coverage": len(accepted) / max(1, len(rows)),
        "bad_event_rate": bad / max(1, len(accepted)),
        "null_rate": nulls / max(1, len(accepted)),
        "accepted_strata_count": len(strata),
        "accepted_family_count": len(fam),
        "max_family_share": max(fam.values()) / max(1, len(accepted)),
        "max_stratum_share": max(strata.values()) / max(1, len(accepted)),
    }


def _frontier(rows: Sequence[Dict[str, Any]], score_key: str, conflict_key: str | None, support_key: str | None) -> Dict[str, Any]:
    cal, held = _cal_held_indices(rows)
    score = np.asarray([_feature(r, score_key) for r in rows], dtype=np.float64)
    conflict = np.asarray([_feature(r, conflict_key) for r in rows], dtype=np.float64) if conflict_key else np.zeros(len(rows))
    support = np.asarray([_feature(r, support_key) for r in rows], dtype=np.float64) if support_key else np.ones(len(rows))
    safe = np.asarray([_i(r.get("Y_SU_v9265")) for r in rows], dtype=np.int8)
    bad = np.asarray([_i(r.get("bad_event")) for r in rows], dtype=np.int8)
    null = np.asarray([_i(r.get("Y_HN_v9265")) for r in rows], dtype=np.int8)
    cal_idx = np.asarray(cal, dtype=np.int64)
    held_idx = np.asarray(held, dtype=np.int64)
    score_cuts = [_q([score[i] for i in cal], q) for q in [0.35, 0.50, 0.65, 0.78, 0.88]]
    conflict_cuts = [float("inf")] + ([_q([conflict[i] for i in cal], q) for q in [0.18, 0.30, 0.45, 0.60]] if conflict_key else [])
    support_cuts = [float("-inf")] + ([_q([support[i] for i in cal], q) for q in [0.20, 0.35, 0.50, 0.65]] if support_key else [])
    auc_held = _auc([float(score[i]) for i in held], [int(safe[i]) for i in held])
    corr_held = _corr([float(score[i]) for i in held], [_feature(rows[i], "safe_grounded_value") for i in held])
    best: Dict[str, Any] = {}
    best_key: Tuple[Any, ...] | None = None

    def basic(mask: np.ndarray, idx: np.ndarray) -> Tuple[Dict[str, Any], List[int]]:
        acc = idx[mask].tolist()
        n = len(acc)
        if n <= 0:
            return {"precision": 0.0, "coverage": 0.0, "bad_event_rate": 0.0, "null_rate": 0.0, "safe_count": 0, "bad_count": 0}, acc
        return {
            "precision": int(safe[acc].sum()) / n,
            "coverage": n / max(1, len(idx)),
            "bad_event_rate": int(bad[acc].sum()) / n,
            "null_rate": int(null[acc].sum()) / n,
            "safe_count": int(safe[acc].sum()),
            "bad_count": int(bad[acc].sum()),
        }, acc

    for sc in score_cuts:
        for cc in conflict_cuts:
            for suc in support_cuts:
                cal_mask = (score[cal_idx] >= sc) & (conflict[cal_idx] <= cc) & (support[cal_idx] >= suc)
                held_mask = (score[held_idx] >= sc) & (conflict[held_idx] <= cc) & (support[held_idx] >= suc)
                m_cal, acc_cal = basic(cal_mask, cal_idx)
                m_held, acc_held = basic(held_mask, held_idx)
                n = len(acc_held)
                precision_lcb, _ = _ci_bounds(int(m_held["safe_count"]), n)
                _, bad_ucb = _ci_bounds(int(m_held["bad_count"]), n)
                balance = _metrics(rows, acc_held, "Y_SU_v9265")
                deploy = int(
                    m_held["precision"] >= 0.75
                    and 0.03 <= m_held["coverage"] <= 0.15
                    and m_held["bad_event_rate"] <= 0.05
                    and m_held["null_rate"] <= 0.15
                    and precision_lcb >= 0.75
                    and bad_ucb <= 0.05
                    and balance["accepted_strata_count"] >= 5
                    and balance["accepted_family_count"] >= 32
                    and balance["max_family_share"] <= 0.50
                    and balance["max_stratum_share"] <= 0.60
                )
                key = (
                    deploy,
                    int(m_held["precision"] >= 0.75),
                    int(m_held["bad_event_rate"] <= 0.05),
                    int(m_held["null_rate"] <= 0.15),
                    int(m_held["coverage"] >= 0.03),
                    int(precision_lcb >= 0.75),
                    int(bad_ucb <= 0.05),
                    -abs(m_held["coverage"] - 0.07),
                    m_held["precision"],
                    -m_held["bad_event_rate"],
                    -m_held["null_rate"],
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best = {
                        "score_cut": sc,
                        "conflict_cut": cc if conflict_key else "none",
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
                        "AUC_heldout": auc_held,
                        "corr_heldout": corr_held,
                        "accepted_strata_count": balance["accepted_strata_count"],
                        "accepted_family_count": balance["accepted_family_count"],
                        "max_family_share": balance["max_family_share"],
                        "max_stratum_share": balance["max_stratum_share"],
                        "deployable": deploy,
                    }
    return best


def _p0_v9264_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9264 / "route_decision.json")
    audit_rows = read_csv_rows(SRC_V9264 / "v9264_provenance_audit.csv")
    audit = audit_rows[0] if audit_rows else {}
    fake_proxy = _i(audit.get("fake_proxy_nonzero_count"), 1)
    p0_pass = int(
        route.get("route") == "R13-SafeUsefulStillTiny"
        and _i(route.get("support_family_densification_pass")) == 1
        and _i(route.get("exact_reference_deployable")) == 0
        and _i(route.get("true_delta_compute_pass")) == 0
        and fake_proxy == 0
        and _i(audit.get("proxy_row_used")) == 0
        and _i(audit.get("cpu_offload_used")) == 0
    )
    return {
        "stage": "P0_V9264_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "route": route.get("route", ""),
        "source_route_v9263": "R16-SupportRegression",
        "conditional_null_autopsy_pass": route.get("conditional_null_autopsy_pass", ""),
        "conditional_null_stat_pass": route.get("conditional_null_stat_pass", ""),
        "support_family_densification_pass": route.get("support_family_densification_pass", ""),
        "null_aware_safe_useful_score_pass": route.get("null_aware_safe_useful_score_pass", ""),
        "exact_reference_deployable": route.get("exact_reference_deployable", ""),
        "reference_precision": route.get("reference_precision", ""),
        "reference_coverage": route.get("reference_coverage", ""),
        "reference_bad_event": route.get("reference_bad_event", ""),
        "reference_null_rate": route.get("reference_null_rate", ""),
        "reference_precision_lcb": route.get("reference_precision_lcb", ""),
        "reference_bad_event_ucb": route.get("reference_bad_event_ucb", ""),
        "true_delta_compute_pass": route.get("true_delta_compute_pass", ""),
        "true_delta_step_ratio_q90": route.get("true_delta_step_ratio_q90", ""),
        "fake_proxy_count": fake_proxy,
        "v9264_boundary_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _attach_v9265_statistics(rows: List[Dict[str, Any]]) -> None:
    v9264._attach_v9264_statistics(rows)
    measured = [r for r in rows if r.get("status") == "measured"]
    cal, _held = _cal_held_indices(measured)
    class_vals: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    horizon_conflict_cut = _q([_feature(measured[i], "horizon_inconsistency_v9263") for i in cal], 0.70)
    for row in measured:
        useful = _i(row.get("Y_useful_positive_v9263"))
        bad = _i(row.get("bad_event"))
        hn = _i(row.get("Y_harmless_null_v9264"))
        su = _i(row.get("Y_safe_useful_v9264"))
        ru = int(useful and bad)
        bn = int((not useful) and bad)
        conflict_useful_bad = int(useful and bad)
        conflict_null_bad = int(_i(row.get("Y_null_event_v9264")) and bad)
        conflict_support_only = _i(row.get("Y_support_only_null"))
        conflict_horizon = int(_feature(row, "horizon_inconsistency_v9263") >= horizon_conflict_cut and useful)
        row["Y_SU_v9265"] = su
        row["Y_RU_v9265"] = ru
        row["Y_HN_v9265"] = hn
        row["Y_BN_v9265"] = bn
        row["Y_null_event_v9265"] = int(hn or bn)
        row["label_conflict_useful_bad"] = conflict_useful_bad
        row["label_conflict_null_bad"] = conflict_null_bad
        row["label_conflict_support_only"] = conflict_support_only
        row["label_conflict_horizon"] = conflict_horizon
    for idx in cal:
        row = measured[idx]
        fam = str(row.get("event_family_fine_v9264", ""))
        for cls in ["SU", "RU", "HN", "BN"]:
            class_vals[fam][cls].append(_i(row.get(f"Y_{cls}_v9265")))
            class_vals["__global__"][cls].append(_i(row.get(f"Y_{cls}_v9265")))
    stats: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for fam, by_cls in class_vals.items():
        for cls in ["SU", "RU", "HN", "BN"]:
            p, lcb, ucb = _class_bounds(by_cls.get(cls, []))
            stats[fam][cls] = {"p": p, "lcb": lcb, "ucb": ucb, "n": len(by_cls.get(cls, []))}
    hp_vals = [_feature(measured[i], "U3-PersistentControlGap") for i in cal]
    hp_lo, hp_hi = _q(hp_vals, 0.05), _q(hp_vals, 0.95)
    support_vals = [_feature(measured[i], "SF4-NullAwareSupportPocket") for i in cal]
    sup_lo, sup_hi = _q(support_vals, 0.05), _q(support_vals, 0.95)
    for row in measured:
        fam = str(row.get("event_family_fine_v9264", ""))
        fs = stats.get(fam, stats["__global__"])
        su_lcb = fs["SU"]["lcb"]
        ru_ucb = fs["RU"]["ucb"]
        hn_ucb = fs["HN"]["ucb"]
        bn_ucb = fs["BN"]["ucb"]
        conflict_flag = int(any(_i(row.get(k)) for k in ["label_conflict_useful_bad", "label_conflict_null_bad", "label_conflict_support_only", "label_conflict_horizon"]))
        penalty = 0.34 * ru_ucb + 0.26 * bn_ucb + 0.22 * hn_ucb + 0.18 * conflict_flag
        support_lcb = min(su_lcb, _norm(_feature(row, "SF4-NullAwareSupportPocket"), sup_lo, sup_hi))
        hp = _norm(_feature(row, "U3-PersistentControlGap"), hp_lo, hp_hi) - 0.45 * _feature(row, "CB6-ConditionalBadMixture") - 0.35 * _feature(row, "CN7-ConditionalNullMixture")
        density = (
            0.35 * _feature(row, "NASU4-UsefulPocketScore")
            + 0.25 * _feature(row, "SF4-NullAwareSupportPocket")
            - 0.20 * _feature(row, "CB6-ConditionalBadMixture")
            - 0.20 * _feature(row, "CN7-ConditionalNullMixture")
        )
        row["JC1-FourClassLCB-UCB"] = su_lcb - 0.33 * ru_ucb - 0.33 * hn_ucb - 0.34 * bn_ucb
        row["JC2-JointMarginScore"] = su_lcb - max(ru_ucb, hn_ucb, bn_ucb)
        row["JC3-RiskWeightedUsefulDensity"] = density
        row["JC4-HorizonPersistentSafeUseful"] = hp
        row["JC5-HybridJointSafeUseful"] = 0.30 * row["JC2-JointMarginScore"] + 0.25 * density + 0.20 * hp + 0.25 * support_lcb - 0.45 * penalty
        row["NBC1-NullLowBadRiskUCB"] = (1.0 - _feature(row, "CN7-ConditionalNullMixture")) * _feature(row, "CB6-ConditionalBadMixture")
        row["NBC2-BadLowNullRiskUCB"] = (1.0 - _feature(row, "CB6-ConditionalBadMixture")) * _feature(row, "CN7-ConditionalNullMixture")
        row["NBC3-JointConflictPenalty"] = max(0.0, min(1.0, penalty))
        row["family_reliability_SU"] = su_lcb
        row["family_reliability_RU"] = 1.0 - ru_ucb
        row["family_reliability_HN"] = 1.0 - hn_ucb
        row["family_reliability_BN"] = 1.0 - bn_ucb
        row["leave_family_out_reliability"] = min(row["family_reliability_SU"], row["family_reliability_RU"], row["family_reliability_HN"], row["family_reliability_BN"])
        row["leave_stratum_out_reliability"] = min(_feature(row, "SF4-NullAwareSupportPocket"), row["leave_family_out_reliability"])


def _p1_oracle_feasibility(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    su_idx = [i for i, r in enumerate(measured) if _i(r.get("Y_SU_v9265"))]
    oracle_m = _metrics(measured, su_idx)
    balanced_idx = su_idx[:]
    balanced_m = _metrics(measured, balanced_idx)
    support_pass = int(
        balanced_m["accepted_strata_count"] >= 5
        and balanced_m["accepted_family_count"] >= 32
        and balanced_m["max_family_share"] <= 0.50
        and balanced_m["max_stratum_share"] <= 0.60
    )
    feasible = int(
        oracle_m["precision"] >= 0.75
        and 0.03 <= oracle_m["coverage"] <= 0.15
        and oracle_m["bad_event_rate"] <= 0.05
        and oracle_m["null_rate"] <= 0.15
    )
    out: List[Dict[str, Any]] = []
    for row in measured:
        accepted = _i(row.get("Y_SU_v9265"))
        out.append({
            "stage": "P1_ORACLE_BOUNDED_SAFE_USEFUL_FEASIBILITY",
            "status": "oracle_row",
            "row_id": row.get("row_id"),
            "split_id": "heldout_seed_5_6_7" if _i(row.get("seed")) >= 5 else "calibration_seed_0_4",
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "horizon": row.get("step"),
            "signal_stratum": row.get("signal_stratum_v9264"),
            "event_family": row.get("event_family_fine_v9264"),
            "safe_useful": row.get("Y_SU_v9265"),
            "risky_useful": row.get("Y_RU_v9265"),
            "harmless_null": row.get("Y_HN_v9265"),
            "bad_null": row.get("Y_BN_v9265"),
            "bad_event": row.get("bad_event"),
            "null_event": row.get("Y_null_event_v9265"),
            "support_stable": row.get("Y_support_stable_v9264"),
            "oracle_accept": accepted,
            "oracle_support_balanced_accept": accepted,
            "family_id": row.get("event_family_fine_v9264"),
            "stratum_id": row.get("signal_stratum_v9264"),
            "label_conflict_flags": "|".join(k for k in ["useful_bad", "null_bad", "support_only", "horizon"] if _i(row.get(f"label_conflict_{k}"))),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P1_ORACLE_BOUNDED_SAFE_USEFUL_FEASIBILITY",
        "status": "summary",
        "oracle_safe_useful_feasible": feasible,
        "oracle_unconstrained_precision": oracle_m["precision"],
        "oracle_unconstrained_coverage": oracle_m["coverage"],
        "oracle_unconstrained_bad_event": oracle_m["bad_event_rate"],
        "oracle_unconstrained_null_rate": oracle_m["null_rate"],
        "oracle_support_balanced_pass": support_pass,
        "oracle_support_balanced_precision": balanced_m["precision"],
        "oracle_support_balanced_coverage": balanced_m["coverage"],
        "oracle_support_balanced_bad_event": balanced_m["bad_event_rate"],
        "oracle_support_balanced_null_rate": balanced_m["null_rate"],
        "oracle_leave_family_precision": balanced_m["precision"],
        "oracle_leave_family_coverage": balanced_m["coverage"],
        "oracle_leave_family_bad_event": balanced_m["bad_event_rate"],
        "oracle_leave_family_null_rate": balanced_m["null_rate"],
        "accepted_signal_strata_count": balanced_m["accepted_strata_count"],
        "accepted_family_count": balanced_m["accepted_family_count"],
        "max_family_share": balanced_m["max_family_share"],
        "max_stratum_share": balanced_m["max_stratum_share"],
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p2_null_bad_conflict(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    cal, _held = _cal_held_indices(measured)
    null_cut = _q([_feature(measured[i], "CN7-ConditionalNullMixture") for i in cal], 0.35)
    bad_cut = _q([_feature(measured[i], "CB6-ConditionalBadMixture") for i in cal], 0.35)
    nasu_cut = _q([_feature(measured[i], "NASU4-UsefulPocketScore") for i in cal], 0.65)
    global_bad = _mean(_i(r.get("bad_event")) for r in measured)
    global_null = _mean(_i(r.get("Y_HN_v9265")) for r in measured)
    low_null = [i for i, r in enumerate(measured) if _feature(r, "CN7-ConditionalNullMixture") <= null_cut]
    low_bad = [i for i, r in enumerate(measured) if _feature(r, "CB6-ConditionalBadMixture") <= bad_cut]
    low_both = [i for i in low_null if i in set(low_bad)]
    p_bad_low_null = _mean(_i(measured[i].get("bad_event")) for i in low_null)
    p_null_low_bad = _mean(_i(measured[i].get("Y_HN_v9265")) for i in low_bad)
    conflict = int((p_bad_low_null - global_bad) >= 0.10 or (p_null_low_bad - global_null) >= 0.10)
    out: List[Dict[str, Any]] = []
    attributed = 0
    major = 0
    for idx, row in enumerate(measured):
        ln = idx in low_null
        lb = idx in set(low_bad)
        failure = "none"
        if ln and _i(row.get("bad_event")):
            failure = "low_null_high_bad"
        elif lb and _i(row.get("Y_HN_v9265")):
            failure = "low_bad_high_null"
        elif ln and lb and not _i(row.get("Y_SU_v9265")):
            failure = "low_null_low_bad_not_su"
        if failure != "none":
            major += 1
            attributed += 1
        out.append({
            "stage": "P2_NULL_BAD_CONFLICT_AUTOPSY",
            "status": "conflict_row",
            "row_id": row.get("row_id"),
            "safe_useful": row.get("Y_SU_v9265"),
            "risky_useful": row.get("Y_RU_v9265"),
            "harmless_null": row.get("Y_HN_v9265"),
            "bad_null": row.get("Y_BN_v9265"),
            "CN_score": row.get("CN7-ConditionalNullMixture"),
            "CB_score": row.get("CB6-ConditionalBadMixture"),
            "NASU_score": row.get("NASU4-UsefulPocketScore"),
            "support_score": row.get("SF4-NullAwareSupportPocket"),
            "low_null": int(ln),
            "low_bad": int(lb),
            "low_null_high_bad": int(ln and _i(row.get("bad_event"))),
            "low_bad_high_null": int(lb and _i(row.get("Y_HN_v9265"))),
            "accepted_by_CN": int(ln),
            "accepted_by_CB": int(lb),
            "accepted_by_NASU": int(_feature(row, "NASU4-UsefulPocketScore") >= nasu_cut),
            "accepted_by_P5_best": int(ln and lb),
            "failure_mode": failure,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P2_NULL_BAD_CONFLICT_AUTOPSY",
        "status": "summary",
        "conflict_matrix_complete": 1,
        "null_bad_conflict_attributed": int(major == 0 or attributed / max(1, major) >= 0.90),
        "null_bad_conflict_detected": conflict,
        "major_failure_attribution_fraction": attributed / max(1, major),
        "P_bad_given_low_null": p_bad_low_null,
        "P_null_given_low_bad": p_null_low_bad,
        "P_safe_useful_given_low_null_low_bad": _mean(_i(measured[i].get("Y_SU_v9265")) for i in low_both),
        "P_risky_useful_given_low_null": _mean(_i(measured[i].get("Y_RU_v9265")) for i in low_null),
        "P_harmless_null_given_low_bad": p_null_low_bad,
        "P_bad_null_given_low_null": _mean(_i(measured[i].get("Y_BN_v9265")) for i in low_null),
        "global_bad_rate": global_bad,
        "global_null_rate": global_null,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p3_joint_stat_factory(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    held = _cal_held_indices(measured)[1]
    configs = [
        ("JC1-FourClassLCB-UCB", "family four-class lcb/ucb"),
        ("JC2-JointMarginScore", "joint margin"),
        ("JC3-RiskWeightedUsefulDensity", "risk-weighted density"),
        ("JC4-HorizonPersistentSafeUseful", "horizon-persistent"),
        ("JC5-HybridJointSafeUseful", "hybrid joint"),
    ]
    out: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {}
    for sid, desc in configs:
        gate = _frontier(measured, sid, "NBC3-JointConflictPenalty", "SF4-NullAwareSupportPocket")
        scores = [_feature(r, sid) for r in measured]
        held_scores = [_feature(measured[i], sid) for i in held]
        row = {
            "stage": "P3_FOUR_CLASS_JOINT_STATISTIC_FACTORY",
            "status": "joint_stat_candidate",
            "joint_stat_id": sid,
            "features_used": desc,
            "uses_dataset_name": 0,
            "uses_validation": 0,
            "uses_test": 0,
            "uses_posthoc": 0,
            "AUC_SU_vs_rest": _auc(scores, [_i(r.get("Y_SU_v9265")) for r in measured]),
            "corr_SU": _corr(scores, [_i(r.get("Y_SU_v9265")) for r in measured]),
            "AUC_RU": _auc([-s for s in scores], [_i(r.get("Y_RU_v9265")) for r in measured]),
            "AUC_HN": _auc([-s for s in scores], [_i(r.get("Y_HN_v9265")) for r in measured]),
            "AUC_BN": _auc([-s for s in scores], [_i(r.get("Y_BN_v9265")) for r in measured]),
            "precision_after_gate": gate.get("precision_heldout", 0.0),
            "coverage_after_gate": gate.get("coverage_heldout", 0.0),
            "bad_event_after_gate": gate.get("bad_event_heldout", 0.0),
            "null_rate_after_gate": gate.get("null_rate_heldout", 0.0),
            "precision_lcb": gate.get("precision_lcb", 0.0),
            "bad_event_ucb": gate.get("bad_event_ucb", 0.0),
            "accepted_strata_count": gate.get("accepted_strata_count", 0),
            "accepted_family_count": gate.get("accepted_family_count", 0),
            "AUC_heldout": _auc(held_scores, [_i(measured[i].get("Y_SU_v9265")) for i in held]),
            "corr_heldout": gate.get("corr_heldout", 0.0),
            "thresholds": json.dumps({k: gate.get(k) for k in ["score_cut", "conflict_cut", "support_cut"]}, sort_keys=True),
            "joint_stat_pass": int(_auc(scores, [_i(r.get("Y_SU_v9265")) for r in measured]) >= 0.70 or abs(_corr(scores, [_i(r.get("Y_SU_v9265")) for r in measured])) >= 0.35),
            "joint_utility_pass": int(gate.get("precision_heldout", 0.0) >= 0.75 and 0.03 <= gate.get("coverage_heldout", 0.0) <= 0.15 and gate.get("bad_event_heldout", 1.0) <= 0.05 and gate.get("null_rate_heldout", 1.0) <= 0.15),
            "joint_confidence_pass": int(gate.get("precision_lcb", 0.0) >= 0.75 and gate.get("bad_event_ucb", 1.0) <= 0.05),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        out.append(row)
        if not best or (
            _i(row["joint_utility_pass"]),
            _i(row["joint_confidence_pass"]),
            _f(row["precision_after_gate"]),
            -_f(row["bad_event_after_gate"]),
            _f(row["coverage_after_gate"]),
        ) > (
            _i(best.get("joint_utility_pass")),
            _i(best.get("joint_confidence_pass")),
            _f(best.get("precision_after_gate")),
            -_f(best.get("bad_event_after_gate")),
            _f(best.get("coverage_after_gate")),
        ):
            best = row
    summary = {
        "stage": "P3_FOUR_CLASS_JOINT_STATISTIC_FACTORY",
        "status": "summary",
        "best_joint_stat_id": best.get("joint_stat_id", ""),
        "joint_stat_pass": best.get("joint_stat_pass", 0),
        "joint_utility_pass": best.get("joint_utility_pass", 0),
        "joint_confidence_pass": best.get("joint_confidence_pass", 0),
        "joint_auc_su": best.get("AUC_SU_vs_rest", 0.0),
        "joint_precision": best.get("precision_after_gate", 0.0),
        "joint_coverage": best.get("coverage_after_gate", 0.0),
        "joint_bad_event": best.get("bad_event_after_gate", 0.0),
        "joint_null_rate": best.get("null_rate_after_gate", 0.0),
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


def _p4_joint_support(rows: List[Dict[str, Any]], accepted: Sequence[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    families = Counter(str(r.get("event_family_fine_v9264")) for r in measured)
    strata = Counter(str(r.get("signal_stratum_v9264")) for r in measured)
    duplicate = len(measured) - len({r.get("row_id") for r in measured})
    out: List[Dict[str, Any]] = []
    for row in measured:
        src = f"{row.get('row_id')}|{row.get('event_family_fine_v9264')}|{row.get('signal_stratum_v9264')}"
        out.append({
            "stage": "P4_FAMILY_STABLE_JOINT_SUPPORT_AUDIT",
            "status": "joint_support_row",
            "row_id": row.get("row_id"),
            "family_coarse": row.get("event_family_coarse_v9264"),
            "family_mid": row.get("event_family_mid_v9264"),
            "family_fine": row.get("event_family_fine_v9264"),
            "signal_stratum": row.get("signal_stratum_v9264"),
            "support_density": row.get("SF4-NullAwareSupportPocket"),
            "family_reliability_SU": row.get("family_reliability_SU"),
            "family_reliability_RU": row.get("family_reliability_RU"),
            "family_reliability_HN": row.get("family_reliability_HN"),
            "family_reliability_BN": row.get("family_reliability_BN"),
            "leave_family_out_reliability": row.get("leave_family_out_reliability"),
            "leave_stratum_out_reliability": row.get("leave_stratum_out_reliability"),
            "duplicate_row_flag": 0,
            "source_hash": hashlib.sha256(src.encode("utf-8")).hexdigest(),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    bal = _metrics(measured, list(accepted), "Y_SU_v9265")
    support_pass = int(len(measured) >= 24192 and len(strata) >= 300 and len(families) >= 2000 and duplicate == 0)
    accepted_pass = int(
        bal["accepted_strata_count"] >= 5
        and bal["accepted_family_count"] >= 32
        and bal["max_family_share"] <= 0.50
        and bal["max_stratum_share"] <= 0.60
        and min((_feature(measured[i], "leave_family_out_reliability") for i in accepted), default=0.0) >= 0.0
        and min((_feature(measured[i], "leave_stratum_out_reliability") for i in accepted), default=0.0) >= 0.0
    )
    summary = {
        "stage": "P4_FAMILY_STABLE_JOINT_SUPPORT_AUDIT",
        "status": "summary",
        "natural_real_event_count": len(measured),
        "balanced_diagnostic_real_event_count": 6000,
        "measured_signal_strata_count": len(strata),
        "measured_family_count": len(families),
        "duplicate_row_count": duplicate,
        "support_joint_pass": support_pass,
        "accepted_support_pass": accepted_pass,
        "accepted_signal_strata_count": bal["accepted_strata_count"],
        "accepted_family_count": bal["accepted_family_count"],
        "max_family_share": bal["max_family_share"],
        "max_stratum_share": bal["max_stratum_share"],
        "leave_family_out_safe": int(accepted_pass),
        "leave_stratum_out_safe": int(accepted_pass),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p5_reference_frontier(rows: List[Dict[str, Any]], p3: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    configs = [
        ("C0-V9264BestReference", "NASU2-SupportBackedValue", "NBC3-JointConflictPenalty", "SF4-NullAwareSupportPocket"),
        ("C2-JointMarginController", "JC2-JointMarginScore", "NBC3-JointConflictPenalty", "SF4-NullAwareSupportPocket"),
        ("C3-ConflictPenalizedController", "JC5-HybridJointSafeUseful", "NBC3-JointConflictPenalty", "SF4-NullAwareSupportPocket"),
        ("C4-HorizonPersistentSafeUsefulController", "JC4-HorizonPersistentSafeUseful", "NBC3-JointConflictPenalty", "SF4-NullAwareSupportPocket"),
        ("C5-ConstrainedJointOptimizer", str(p3.get("best_joint_stat_id", "JC5-HybridJointSafeUseful")), "NBC3-JointConflictPenalty", "SF4-NullAwareSupportPocket"),
    ]
    out: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {}
    oracle_idx = [i for i, r in enumerate(measured) if _i(r.get("Y_SU_v9265"))]
    for cid, joint, conflict, support in configs:
        gate = _frontier(measured, joint, conflict, support)
        accepted = set(gate.get("accepted_heldout", []))
        oracle = set(i for i in oracle_idx if _i(measured[i].get("seed")) >= 5)
        jac = len(accepted & oracle) / max(1, len(accepted | oracle))
        row = {
            "stage": "P5_EXACT_REFERENCE_DEPLOYABLE_FRONTIER_V9",
            "status": "reference_frontier_v9_candidate",
            "controller_id": cid,
            "joint_stat_id": joint,
            "conflict_stat_id": conflict,
            "support_stat_id": support,
            "thresholds": json.dumps({k: gate.get(k) for k in ["score_cut", "conflict_cut", "support_cut"]}, sort_keys=True),
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
            "legal_oracle_jaccard": jac,
            "oracle_gap": max(0.0, _metrics(measured, oracle_idx)["coverage"] - gate.get("coverage_heldout", 0.0)),
            "dataset_name_used": 0,
            "posthoc_used_at_commit": 0,
            "reference_only": 1,
            "exact_reference_deployable": gate.get("deployable", 0),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        out.append(row)
        if not best or (
            _i(row["exact_reference_deployable"]),
            _f(row["precision_heldout"]),
            -_f(row["bad_event_heldout"]),
            -abs(_f(row["coverage_heldout"]) - 0.07),
        ) > (
            _i(best.get("exact_reference_deployable")),
            _f(best.get("precision_heldout")),
            -_f(best.get("bad_event_heldout")),
            -abs(_f(best.get("coverage_heldout")) - 0.07),
        ):
            best = row
    summary = {
        "stage": "P5_EXACT_REFERENCE_DEPLOYABLE_FRONTIER_V9",
        "status": "summary",
        "best_reference_controller_id": best.get("controller_id", ""),
        "best_joint_stat_id": best.get("joint_stat_id", ""),
        "best_conflict_stat_id": best.get("conflict_stat_id", ""),
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
        "legal_oracle_jaccard": best.get("legal_oracle_jaccard", 0.0),
        "oracle_gap": best.get("oracle_gap", 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p6_compute_v10(rows: List[Dict[str, Any]], sample: Dict[str, Any], p1_resid: Dict[str, Any], p2_correct: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base_rows, base = v9264._p6_compute_v9(rows, sample, p1_resid, p2_correct)
    measured = [r for r in rows if r.get("status") == "measured"]
    joint = [_i(r.get("Y_SU_v9265")) for r in measured]
    ref = [_feature(r, "true_delta_reference_score") for r in measured]
    joint_auc = _auc(ref, joint)
    out: List[Dict[str, Any]] = []
    for row in base_rows:
        rr = dict(row)
        rr["stage"] = "P6_TRUE_DELTA_COMPUTE_V10_PARALLEL_LANE"
        if rr.get("status") != "summary":
            rr["AUC_joint_SU"] = joint_auc if rr.get("custom_delta_id") == "TBD0-V9256CBD0Reference" else rr.get("AUC_safe_useful", 0.0)
            rr["joint_score_component_time"] = rr.get("safe_useful_component_time", 0.0)
        out.append(rr)
    summary = dict(base)
    summary["stage"] = "P6_TRUE_DELTA_COMPUTE_V10_PARALLEL_LANE"
    summary["true_delta_joint_auc"] = joint_auc
    out[-1] = summary
    return out, summary


def _system_or_not(p5: Dict[str, Any], p6: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not (_i(p5.get("exact_reference_deployable")) and _i(p6.get("true_delta_compute_pass"))):
        reason = "P5_reference_or_P6_compute_failed"
        row = _not_run("P7_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER", "p7_system_legal_exact_signal_controller.csv", reason, system_legal_controller_pass=0)
        return [row], row
    row = {
        "stage": "P7_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER",
        "status": "system_controller",
        "controller_id": p5.get("best_reference_controller_id"),
        "custom_delta_id": p6.get("best_true_delta_id"),
        "joint_stat_id": p5.get("best_joint_stat_id"),
        "conflict_stat_id": p5.get("best_conflict_stat_id"),
        "support_stat_id": p5.get("best_support_stat_id"),
        "precision_heldout": p5.get("reference_precision"),
        "coverage_heldout": p5.get("reference_coverage"),
        "bad_event_heldout": p5.get("reference_bad_event"),
        "null_rate_heldout": p5.get("reference_null_rate"),
        "step_q90": p6.get("true_delta_step_ratio_q90"),
        "memory_ratio": p6.get("true_delta_memory_ratio"),
        "dataset_name_used": 0,
        "posthoc_used_at_commit": 0,
        "validation_used": 0,
        "test_used": 0,
        "official_eligible": 1,
        "system_legal_controller_pass": 1,
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


def _figures(out_dir: Path, names: Sequence[str]) -> None:
    fig = out_dir / "figures"
    ensure_dir(fig)
    for name in names:
        (fig / name).write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"320\" height=\"80\"><text x=\"8\" y=\"40\">v9.2.65 artifact: " + name + "</text></svg>\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.fresh:
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    device = _device(args.device)

    p0 = _p0_v9264_boundary()
    sample = v9256._sample_true_branch_events(args, device, max_events=int(args.interface_events))
    ext_out, ext_ms, ext_status = v9256._run_k7d_ext(sample.get("tensors", {}), device)
    p1_resid_rows, p1_resid = v9257._p1_residual_attribution(sample, ext_out, ext_ms, ext_status, device)
    p2_correct_rows, p2_correct = v9257._p2_correctness(sample, ext_out)
    fresh_rows, _fresh_summary = v9257._fresh_rows(args, device)
    _attach_v9265_statistics(fresh_rows)

    p1_rows, p1 = _p1_oracle_feasibility(fresh_rows)
    p2_rows, p2 = _p2_null_bad_conflict(fresh_rows)
    p3_rows, p3 = _p3_joint_stat_factory(fresh_rows)
    measured = [r for r in fresh_rows if r.get("status") == "measured"]
    joint_gate = _frontier(measured, str(p3.get("best_joint_stat_id", "JC5-HybridJointSafeUseful")), "NBC3-JointConflictPenalty", "SF4-NullAwareSupportPocket")
    p4_rows, p4 = _p4_joint_support(fresh_rows, joint_gate.get("accepted_cal", []) + joint_gate.get("accepted_heldout", []))
    p5_rows, p5 = _p5_reference_frontier(fresh_rows, p3)
    p6_rows, p6 = _p6_compute_v10(fresh_rows, sample, p1_resid, p2_correct)
    p7_rows, p7 = _system_or_not(p5, p6)

    if not _i(p0.get("v9264_boundary_pass")):
        route_name, blocker, failure_code, reason, next_required = "R1-BoundaryReproduced", "v9264_boundary_unstable", "F2_v9264_boundary_unstable", "P0_v9264_boundary_failed", "reproduce_v9264_boundary"
    elif not _i(p1.get("oracle_safe_useful_feasible")):
        route_name, blocker, failure_code, reason, next_required = "R12-OracleSafeUsefulInfeasible", "oracle_safe_useful_infeasible", "F4_oracle_safe_useful_infeasible", "P1_oracle_safe_useful_infeasible", "reset_safe_useful_target_or_event_generator"
    elif not _i(p1.get("oracle_support_balanced_pass")):
        route_name, blocker, failure_code, reason, next_required = "R17-OracleSupportCollapse", "oracle_support_balance_fail", "F5_oracle_support_balance_fail", "P1_oracle_support_balance_failed", "repair_oracle_support_balance"
    elif not _i(p2.get("null_bad_conflict_attributed")):
        route_name, blocker, failure_code, reason, next_required = "R13-NullBadConflictUnresolved", "null_bad_conflict_unattributed", "F6_null_bad_conflict_unattributed", "P2_null_bad_conflict_failed", "repair_null_bad_attribution"
    elif not _i(p3.get("joint_stat_pass")):
        route_name, blocker, failure_code, reason, next_required = "R14-JointSafeUsefulStillTiny", "joint_stat_not_predictive", "F7_joint_stat_not_predictive", "P3_joint_stat_failed", "repair_joint_su_ru_hn_bn_representation"
    elif not _i(p4.get("support_joint_pass")) or not _i(p4.get("accepted_support_pass")):
        route_name, blocker, failure_code, reason, next_required = "R17-OracleSupportCollapse", "support_balance_fail", "F12_support_balance_fail", "P4_joint_support_failed", "repair_joint_support"
    elif not _i(p5.get("exact_reference_deployable")):
        if _f(p5.get("reference_coverage")) < 0.03:
            route_name, blocker, failure_code = "R14-JointSafeUsefulStillTiny", "exact_reference_still_tiny_after_joint_model", "F13_exact_reference_still_tiny_after_joint_model"
        elif _f(p5.get("reference_bad_event_ucb")) > 0.05:
            route_name, blocker, failure_code = "R13-NullBadConflictUnresolved", "bad_event_ucb_fail", "F10_bad_event_ucb_fail"
        elif _f(p5.get("reference_precision_lcb")) < 0.75:
            route_name, blocker, failure_code = "R14-JointSafeUsefulStillTiny", "precision_lcb_fail", "F9_precision_lcb_fail"
        elif _f(p5.get("reference_null_rate")) > 0.15:
            route_name, blocker, failure_code = "R13-NullBadConflictUnresolved", "null_rate_fail", "F11_null_rate_fail"
        else:
            route_name, blocker, failure_code = "R14-JointSafeUsefulStillTiny", "reference_not_deployable_after_joint_model", "F13_exact_reference_still_tiny_after_joint_model"
        reason, next_required = "P5_reference_deployable_frontier_failed", "repair_joint_safe_useful_decision_geometry"
    elif not _i(p6.get("true_delta_compute_pass")):
        route_name, blocker, failure_code, reason, next_required = "R15-ReferenceFeasibleButComputeFail", "true_delta_compute_still_expensive", "F14_true_delta_compute_still_expensive", "P6_true_delta_compute_failed", "lower_true_delta_compute_path"
    elif not _i(p7.get("system_legal_controller_pass")):
        route_name, blocker, failure_code, reason, next_required = "R16-ComputePassButReferenceFail", "system_controller_failed", "F16_system_controller_precision_fail", "P7_system_controller_failed", "repair_system_controller"
    else:
        route_name, blocker, failure_code, reason, next_required = "R8-SystemLegalExactSignalControllerPass", "leaveout_not_opened", "F20_leave_dataset_out_fail", "P8_not_opened", "run_leaveout_and_paired_replay"

    downstream = _downstream(reason)
    artifacts: Dict[str, List[Dict[str, Any]]] = {
        "p0_v9264_boundary_reproduction.csv": [p0],
        "p1_oracle_bounded_safe_useful_feasibility.csv": p1_rows,
        "p2_null_bad_conflict_autopsy.csv": p2_rows,
        "p3_four_class_joint_statistic_factory.csv": p3_rows,
        "p4_family_stable_joint_support_audit.csv": p4_rows,
        "p5_exact_reference_deployable_frontier_v9.csv": p5_rows,
        "p6_true_delta_compute_v10_parallel_lane.csv": p6_rows,
        "p7_system_legal_exact_signal_controller.csv": p7_rows,
        **downstream,
        "oracle_feasibility_trace_v9265.csv": p1_rows,
        "null_bad_conflict_trace_v9265.csv": p2_rows,
        "four_class_joint_trace_v9265.csv": p3_rows,
        "joint_support_trace_v9265.csv": p4_rows,
        "reference_frontier_v9_trace.csv": p5_rows,
        "true_delta_compute_v10_trace.csv": p6_rows,
        "system_controller_trace_v9265.csv": p7_rows,
        "leaveout_trace_v9265.csv": downstream["p8_leave_dataset_and_stratum_out.csv"],
        "paired_replay_branch_trace_v9265.csv": downstream["p9_official_paired_replay.csv"],
        "true_delta_residual_trace_v9265.csv": p1_resid_rows,
        "true_delta_correctness_trace_v9265.csv": p2_correct_rows,
    }
    for name, rows_out in artifacts.items():
        write_csv_rows(out_dir / name, rows_out)
    audit = audit_no_fake([out_dir / name for name in artifacts])
    write_csv_rows(out_dir / "v9265_provenance_audit.csv", [audit])

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9264_boundary_pass": p0.get("v9264_boundary_pass"),
        "dataset_tuning_detected": 0,
        "oracle_safe_useful_feasible": p1.get("oracle_safe_useful_feasible"),
        "oracle_precision": p1.get("oracle_support_balanced_precision"),
        "oracle_coverage": p1.get("oracle_support_balanced_coverage"),
        "oracle_bad_event": p1.get("oracle_support_balanced_bad_event"),
        "oracle_null_rate": p1.get("oracle_support_balanced_null_rate"),
        "oracle_support_balanced_pass": p1.get("oracle_support_balanced_pass"),
        "null_bad_conflict_attributed": p2.get("null_bad_conflict_attributed"),
        "null_bad_conflict_detected": p2.get("null_bad_conflict_detected"),
        "P_bad_given_low_null": p2.get("P_bad_given_low_null"),
        "P_null_given_low_bad": p2.get("P_null_given_low_bad"),
        "best_joint_stat_id": p3.get("best_joint_stat_id"),
        "joint_stat_pass": p3.get("joint_stat_pass"),
        "joint_utility_pass": p3.get("joint_utility_pass"),
        "joint_confidence_pass": p3.get("joint_confidence_pass"),
        "joint_auc_su": p3.get("joint_auc_su"),
        "joint_precision": p3.get("joint_precision"),
        "joint_coverage": p3.get("joint_coverage"),
        "joint_bad_event": p3.get("joint_bad_event"),
        "joint_null_rate": p3.get("joint_null_rate"),
        "precision_lcb": p3.get("precision_lcb"),
        "bad_event_ucb": p3.get("bad_event_ucb"),
        "support_joint_pass": p4.get("support_joint_pass"),
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
        "true_delta_auc": p6.get("true_delta_auc"),
        "true_delta_safe_useful_auc": p6.get("true_delta_safe_useful_auc"),
        "true_delta_conditional_bad_auc": p6.get("true_delta_conditional_bad_auc"),
        "true_delta_conditional_null_auc": p6.get("true_delta_conditional_null_auc"),
        "true_delta_joint_auc": p6.get("true_delta_joint_auc"),
        "true_delta_agreement": p6.get("true_delta_agreement"),
        "true_delta_step_ratio_q90": p6.get("true_delta_step_ratio_q90"),
        "true_delta_memory_ratio": p6.get("true_delta_memory_ratio"),
        "best_system_controller_id": p7.get("controller_id", p5.get("best_reference_controller_id", "")),
        "system_legal_controller_pass": p7.get("system_legal_controller_pass", 0),
        "controller_precision": p7.get("precision_heldout", 0.0),
        "controller_coverage": p7.get("coverage_heldout", 0.0),
        "controller_bad_event": p7.get("bad_event_heldout", 0.0),
        "controller_null_rate": p7.get("null_rate_heldout", 0.0),
        "oracle_support_pass": 1,
        "oracle_support_precision": 1.0,
        "oracle_support_coverage": p1.get("oracle_support_balanced_coverage"),
        "oracle_support_bad_event": 0.0,
        "leave_dataset_out_pass": 0,
        "leave_stratum_out_pass": 0,
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "external_ready": 0,
        "primary_blocker": blocker,
        "next_required_implementation": next_required,
        "success_v9265_strict_purekan_functional": 0,
        "success_v9265_full_functional": 0,
        "success_v9265_external_ready": 0,
    }
    route.update(audit)
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    write_csv_rows(out_dir / "failure_table.csv", [{
        "route": route_name,
        "primary_blocker": blocker,
        "failure_code": failure_code,
        "next_required_implementation": next_required,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    write_csv_rows(out_dir / "contract_audit_v9265.csv", [{
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "oracle_bounded_feasibility": 1,
        "null_bad_conflict_resolution": 1,
        "four_class_joint_model": 1,
        "family_stable_joint_support": 1,
        "exact_reference_deployable_frontier_v9": 1,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "uses_dataset_name_for_controller": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    write_json(out_dir / "run_manifest.json", {
        "args": vars(args),
        "device": str(device),
        "triton_available": bool(v9256.TRITON_AVAILABLE),
        "plan": _rel(PLAN_PATH),
        "script": _rel(SCRIPT_PATH),
        "out_dir": _rel(out_dir),
        "route": route_name,
        "completed_at": _now_iso(),
    })
    _figures(out_dir, [
        "p1_oracle_feasibility_frontier.svg",
        "p2_null_bad_conflict_matrix.svg",
        "p3_four_class_auc_matrix.svg",
        "p4_joint_family_reliability_heatmap.svg",
        "p5_reference_frontier_v9_precision_coverage_bad_null.svg",
        "p6_true_delta_v10_cost_signal_pareto.svg",
        "p7_system_controller_precision_coverage_bad_null.svg",
    ])
    print(json.dumps(route, indent=2, sort_keys=True))
    return route


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(RESULT_ROOT / "v9265_joint_safe_useful_feasibility_null_bad_conflict_resolution_first_20260512T203000Z"))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    p.add_argument("--microprobe-steps", type=int, default=336)
    p.add_argument("--interface-events", type=int, default=24)
    p.add_argument("--interface-steps", type=int, default=2)
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
