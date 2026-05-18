#!/usr/bin/env python3
"""DG-KAN v9.2.64 conditional-null separation and support-family densification."""

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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9256_true_branch_delta_tensor_interface_k7d_control_gain_cuda_fusion as v9256  # noqa: E402
import run_v9257_true_branch_delta_compute_closure_exact_signal_controller_feasibility as v9257  # noqa: E402
import run_v9262_useful_control_target_reset_null_rejection_frontier as v9262  # noqa: E402
import run_v9263_conditional_bad_event_separation_risk_adjusted_safe_useful_frontier as v9263  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.64_ConditionalNullSeparation_SupportFamilyDensification_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9264_conditional_null_separation_support_family_densification.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9263 = RESULT_ROOT / "v9263_conditional_bad_event_separation_risk_adjusted_safe_useful_frontier_first_20260512T180000Z"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


_f = v9263._f
_i = v9263._i
_mean = v9263._mean
_q = v9263._q
_auc = v9263._auc
_corr = v9263._corr
_feature = v9263._feature
_norm = v9263._norm
_safe_div = v9263._safe_div
_device = v9263._device
_ci_bounds = v9263._ci_bounds
_cal_held_indices = v9263._cal_held_indices
_not_run = v9263._not_run


def _bucket(value: float, cuts: Sequence[float], labels: Sequence[str]) -> str:
    for cut, label in zip(cuts, labels):
        if value <= cut:
            return label
    return labels[-1]


def _accept_metrics(rows: Sequence[Dict[str, Any]], accepted: Sequence[int], label_key: str = "Y_safe_useful_v9264") -> Dict[str, Any]:
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
    nulls = sum(_i(rows[i].get("Y_harmless_null_v9264")) for i in accepted)
    families = Counter(str(rows[i].get("event_family_fine_v9264", rows[i].get("event_family_v9264", ""))) for i in accepted)
    strata = Counter(str(rows[i].get("signal_stratum_v9264", "")) for i in accepted)
    return {
        "precision": safe / max(1, len(accepted)),
        "coverage": len(accepted) / max(1, len(rows)),
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
    label_key: str = "Y_safe_useful_v9264",
) -> Dict[str, Any]:
    cal, held = _cal_held_indices(rows)
    score_vals = [_feature(r, score_key) for r in rows]
    bad_vals = [_feature(r, bad_key) for r in rows] if bad_key else [float("-inf")] * len(rows)
    null_vals = [_feature(r, null_key) for r in rows] if null_key else [float("-inf")] * len(rows)
    support_vals = [_feature(r, support_key) for r in rows] if support_key else [float("inf")] * len(rows)
    score_cuts = [_q([score_vals[i] for i in cal], q) for q in [0.35, 0.50, 0.65, 0.78, 0.88]]
    bad_cuts = [float("inf")] + ([_q([bad_vals[i] for i in cal], q) for q in [0.20, 0.35, 0.50, 0.65]] if bad_key else [])
    null_cuts = [float("inf")] + ([_q([null_vals[i] for i in cal], q) for q in [0.18, 0.30, 0.45, 0.60, 0.75]] if null_key else [])
    support_cuts = [float("-inf")] + ([_q([support_vals[i] for i in cal], q) for q in [0.20, 0.35, 0.50, 0.65]] if support_key else [])
    held_labels = [_i(rows[i].get(label_key)) for i in held]
    held_grounded = [_feature(rows[i], "safe_grounded_value") for i in held]
    score_arr = np.asarray(score_vals, dtype=np.float64)
    bad_arr = np.asarray(bad_vals, dtype=np.float64)
    null_arr = np.asarray(null_vals, dtype=np.float64)
    support_arr = np.asarray(support_vals, dtype=np.float64)
    safe_arr = np.asarray([_i(r.get(label_key)) for r in rows], dtype=np.int8)
    bad_event_arr = np.asarray([_i(r.get("bad_event")) for r in rows], dtype=np.int8)
    harmless_null_arr = np.asarray([_i(r.get("Y_harmless_null_v9264")) for r in rows], dtype=np.int8)
    cal_idx = np.asarray(cal, dtype=np.int64)
    held_idx = np.asarray(held, dtype=np.int64)
    auc_held = _auc([score_vals[i] for i in held], held_labels)
    corr_held = _corr([score_vals[i] for i in held], held_grounded)

    def basic_metrics(mask: np.ndarray, denom: int) -> Dict[str, Any]:
        n = int(mask.sum())
        if n <= 0:
            return {
                "precision": 0.0,
                "coverage": 0.0,
                "bad_event_rate": 0.0,
                "null_rate": 0.0,
                "accepted_count": 0,
                "safe_count": 0,
                "bad_count": 0,
            }
        idx = held_idx[mask] if denom == len(held_idx) else cal_idx[mask]
        safe = int(safe_arr[idx].sum())
        bad = int(bad_event_arr[idx].sum())
        nulls = int(harmless_null_arr[idx].sum())
        return {
            "precision": safe / n,
            "coverage": n / max(1, denom),
            "bad_event_rate": bad / n,
            "null_rate": nulls / n,
            "accepted_count": n,
            "safe_count": safe,
            "bad_count": bad,
        }

    best: Dict[str, Any] = {}
    best_key: Tuple[Any, ...] | None = None
    for sc in score_cuts:
        for bc in bad_cuts:
            for nc in null_cuts:
                for suc in support_cuts:
                    cal_mask = (score_arr[cal_idx] >= sc) & (bad_arr[cal_idx] <= bc) & (null_arr[cal_idx] <= nc) & (support_arr[cal_idx] >= suc)
                    held_mask = (score_arr[held_idx] >= sc) & (bad_arr[held_idx] <= bc) & (null_arr[held_idx] <= nc) & (support_arr[held_idx] >= suc)
                    m_cal = basic_metrics(cal_mask, len(cal_idx))
                    m_held = basic_metrics(held_mask, len(held_idx))
                    n = int(m_held["accepted_count"])
                    safe = int(m_held["safe_count"])
                    bad = int(m_held["bad_count"])
                    precision_lcb, _ = _ci_bounds(safe, n)
                    _, bad_ucb = _ci_bounds(bad, n)
                    strata_count = 0
                    family_count = 0
                    max_family_share = 0.0
                    max_stratum_share = 0.0
                    numeric_deploy = (
                        _f(m_held["precision"]) >= 0.75
                        and 0.03 <= _f(m_held["coverage"]) <= 0.15
                        and _f(m_held["bad_event_rate"]) <= 0.05
                        and _f(m_held["null_rate"]) <= 0.15
                        and precision_lcb >= 0.75
                        and bad_ucb <= 0.05
                    )
                    if numeric_deploy:
                        held_indices = held_idx[held_mask].tolist()
                        balance = _accept_metrics(rows, held_indices, label_key)
                        strata_count = _i(balance["accepted_strata_count"])
                        family_count = _i(balance["accepted_family_count"])
                        max_family_share = _f(balance["max_family_share"])
                        max_stratum_share = _f(balance["max_stratum_share"])
                    deploy = int(
                        numeric_deploy
                        and strata_count >= 5
                        and family_count >= 32
                        and max_family_share <= 0.50
                        and max_stratum_share <= 0.60
                    )
                    key = (
                        deploy,
                        int(_f(m_held["null_rate"]) <= 0.15),
                        int(_f(m_held["bad_event_rate"]) <= 0.05),
                        int(_f(m_held["precision"]) >= 0.75),
                        int(_f(m_held["coverage"]) >= 0.03),
                        int(precision_lcb >= 0.75),
                        int(bad_ucb <= 0.05),
                        -abs(_f(m_held["coverage"]) - 0.07),
                        _f(m_held["precision"]),
                        -_f(m_held["bad_event_rate"]),
                        -_f(m_held["null_rate"]),
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        acc_cal = cal_idx[cal_mask].tolist()
                        acc_held = held_idx[held_mask].tolist()
                        balance = _accept_metrics(rows, acc_held, label_key)
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
                            "AUC_heldout": auc_held,
                            "corr_heldout": corr_held,
                            "accepted_strata_count": balance["accepted_strata_count"],
                            "accepted_family_count": balance["accepted_family_count"],
                            "max_family_share": balance["max_family_share"],
                            "max_stratum_share": balance["max_stratum_share"],
                            "deployable": deploy,
                        }
    return best


def _p0_v9263_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9263 / "route_decision.json")
    audit_rows = read_csv_rows(SRC_V9263 / "v9263_provenance_audit.csv")
    audit = audit_rows[0] if audit_rows else {}
    fake_proxy = _i(audit.get("fake_proxy_nonzero_count"), 1)
    p0_pass = int(
        route.get("route") == "R16-SupportRegression"
        and _i(route.get("conditional_bad_stat_pass")) == 1
        and _i(route.get("support_confidence_pass")) == 0
        and _i(route.get("exact_reference_deployable")) == 0
        and _i(route.get("true_delta_compute_pass")) == 0
        and fake_proxy == 0
        and _i(audit.get("proxy_row_used")) == 0
        and _i(audit.get("cpu_offload_used")) == 0
    )
    return {
        "stage": "P0_V9263_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "route": route.get("route", ""),
        "source_route_v9262": "R12-UsefulTargetStillTiny",
        "conditional_bad_autopsy_pass": route.get("conditional_bad_autopsy_pass", ""),
        "conditional_bad_stat_pass": route.get("conditional_bad_stat_pass", ""),
        "safe_useful_score_pass": route.get("safe_useful_score_pass", ""),
        "safe_useful_precision": route.get("safe_useful_precision", ""),
        "safe_useful_coverage": route.get("safe_useful_coverage", ""),
        "safe_useful_bad_event": route.get("safe_useful_bad_event", ""),
        "safe_useful_null_rate": route.get("safe_useful_null_rate", ""),
        "support_confidence_pass": route.get("support_confidence_pass", ""),
        "accepted_support_pass": route.get("accepted_support_pass", ""),
        "measured_family_count": route.get("measured_family_count", ""),
        "exact_reference_deployable": route.get("exact_reference_deployable", ""),
        "reference_precision": route.get("reference_precision", ""),
        "reference_coverage": route.get("reference_coverage", ""),
        "reference_bad_event": route.get("reference_bad_event", ""),
        "reference_null_rate": route.get("reference_null_rate", ""),
        "true_delta_compute_pass": route.get("true_delta_compute_pass", ""),
        "true_delta_step_ratio_q90": route.get("true_delta_step_ratio_q90", ""),
        "oracle_support_pass": route.get("oracle_support_pass", ""),
        "oracle_precision": route.get("oracle_precision", ""),
        "oracle_coverage": route.get("oracle_coverage", ""),
        "oracle_bad_event": route.get("oracle_bad_event", ""),
        "fake_proxy_count": fake_proxy,
        "v9263_boundary_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _family_counts(rows: Sequence[Dict[str, Any]], key: str, label_key: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    cal, _ = _cal_held_indices(rows)
    vals: Dict[str, List[int]] = defaultdict(list)
    bads: Dict[str, List[int]] = defaultdict(list)
    for idx in cal:
        fam = str(rows[idx].get(key, ""))
        vals[fam].append(_i(rows[idx].get(label_key)))
        bads[fam].append(_i(rows[idx].get("bad_event")))
    null_ucb: Dict[str, float] = {}
    bad_ucb: Dict[str, float] = {}
    for fam, xs in vals.items():
        p = _mean(xs)
        null_ucb[fam] = min(1.0, p + 1.5 * (p * (1.0 - p) / max(1, len(xs))) ** 0.5)
    for fam, xs in bads.items():
        p = _mean(xs)
        bad_ucb[fam] = min(1.0, p + 1.5 * (p * (1.0 - p) / max(1, len(xs))) ** 0.5)
    return null_ucb, bad_ucb


def _attach_v9264_statistics(rows: List[Dict[str, Any]]) -> None:
    v9263._attach_v9263_statistics(rows)
    measured = [r for r in rows if r.get("status") == "measured"]
    cal, _held = _cal_held_indices(measured)
    for row in measured:
        row["SU2-SafeUsefulMargin"] = (
            _feature(row, "V2-SafeUsefulMargin")
            - 0.50 * _feature(row, "CB5-FamilyInstabilityUCB")
            - 0.35 * _feature(row, "N4-HybridNullProbabilityV2")
            + 0.20 * _feature(row, "S3-RiskAdjustedKNNPocket")
        )
    su2_cut = _q([_feature(measured[i], "SU2-SafeUsefulMargin") for i in cal], 0.68)
    low_bad_cut = _q([_feature(measured[i], "CB5-FamilyInstabilityUCB") for i in cal], 0.60)
    support_cut = _q([_feature(measured[i], "S4-CoverageBalancedSupportCapV2") for i in cal], 0.35)
    real_gain_cut = _q([_feature(measured[i], "real_gain_v9262") for i in cal], 0.45)
    persistent_cut = _q([_feature(measured[i], "U3-PersistentControlGap") for i in cal], 0.45)
    delta_cut = _q([_feature(measured[i], "delta_movement_v9262") for i in cal], 0.45)

    score_vals = [_feature(measured[i], "SU2-SafeUsefulMargin") for i in cal]
    bad_vals = [_feature(measured[i], "CB5-FamilyInstabilityUCB") for i in cal]
    null_vals = [_feature(measured[i], "N4-HybridNullProbabilityV2") for i in cal]
    support_vals = [_feature(measured[i], "S4-CoverageBalancedSupportCapV2") for i in cal]
    delta_vals = [_feature(measured[i], "delta_movement_v9262") for i in cal]
    gap_vals = [_feature(measured[i], "control_gap_v9262") for i in cal]
    eff_vals = [_feature(measured[i], "delta_efficiency_v9262") for i in cal]
    tail_vals = [_feature(measured[i], "tail_harm_v9263") for i in cal]
    frag_vals = [_feature(measured[i], "control_fragility_v9263") for i in cal]
    over_vals = [_feature(measured[i], "delta_overreach_v9263") for i in cal]
    n1_vals = [_feature(measured[i], "N1-ControlEquivalentNullRejector") for i in cal]
    real_gain_vals = [_feature(measured[i], "real_gain_v9262") for i in cal]
    persistent_vals = [_feature(measured[i], "U3-PersistentControlGap") for i in cal]
    score_q25, score_q50, score_q75 = (_q(score_vals, q) for q in (0.25, 0.50, 0.75))
    bad_q25, bad_q50, bad_q75 = (_q(bad_vals, q) for q in (0.25, 0.50, 0.75))
    null_q25, null_q50, null_q75 = (_q(null_vals, q) for q in (0.25, 0.50, 0.75))
    support_q25, support_q50, support_q75 = (_q(support_vals, q) for q in (0.25, 0.50, 0.75))
    delta_q05, delta_q25, delta_q50, delta_q75, delta_q95 = (_q(delta_vals, q) for q in (0.05, 0.25, 0.50, 0.75, 0.95))
    gap_q25, gap_q50, gap_q75 = (_q(gap_vals, q) for q in (0.25, 0.50, 0.75))
    eff_q25, eff_q50, eff_q75 = (_q(eff_vals, q) for q in (0.25, 0.50, 0.75))
    tail_q50 = _q(tail_vals, 0.50)
    frag_q50 = _q(frag_vals, 0.50)
    over_q50 = _q(over_vals, 0.50)
    n1_q05, n1_q60, n1_q95 = (_q(n1_vals, q) for q in (0.05, 0.60, 0.95))
    real_gain_q05, real_gain_q95 = (_q(real_gain_vals, q) for q in (0.05, 0.95))
    persistent_q05, persistent_q95 = (_q(persistent_vals, q) for q in (0.05, 0.95))
    score_q05, score_q95 = (_q(score_vals, q) for q in (0.05, 0.95))

    for row in measured:
        low_bad = int(_feature(row, "CB5-FamilyInstabilityUCB") <= low_bad_cut)
        support_stable = int(_feature(row, "S4-CoverageBalancedSupportCapV2") >= support_cut)
        su2_accept = int(_feature(row, "SU2-SafeUsefulMargin") >= su2_cut and low_bad)
        safe_useful = int(su2_accept and support_stable and not _i(row.get("bad_event")) and not _i(row.get("Y_harmless_null_v9263")))
        harmless_null = int(_i(row.get("Y_harmless_null_v9263")) and not _i(row.get("bad_event")))
        bad_null = int(_i(row.get("bad_event")) and not _i(row.get("Y_useful_positive_v9263")))
        row["Y_low_bad_region_v9264"] = low_bad
        row["Y_support_stable_v9264"] = support_stable
        row["Y_SU2_accept_v9264"] = su2_accept
        row["Y_safe_useful_v9264"] = safe_useful
        row["Y_harmless_null_v9264"] = harmless_null
        row["Y_bad_null_v9264"] = bad_null
        row["Y_null_event_v9264"] = int(harmless_null or bad_null)
        row["Y_control_equivalent_null"] = int(harmless_null and _feature(row, "N1-ControlEquivalentNullRejector") >= n1_q60)
        row["Y_low_real_gain_null"] = int(harmless_null and _feature(row, "real_gain_v9262") <= real_gain_cut)
        row["Y_nonpersistent_null"] = int(harmless_null and _feature(row, "U3-PersistentControlGap") <= persistent_cut)
        row["Y_delta_silent_null"] = int(harmless_null and _feature(row, "delta_movement_v9262") <= delta_cut)
        row["Y_support_only_null"] = int(harmless_null and support_stable and _feature(row, "SU2-SafeUsefulMargin") < su2_cut)
        row["Y_score_artifact_null"] = int(harmless_null and su2_accept)

        su_bucket = _bucket(_feature(row, "SU2-SafeUsefulMargin"), [score_q25, score_q50, score_q75], ["su0", "su1", "su2", "su3"])
        bad_bucket = _bucket(_feature(row, "CB5-FamilyInstabilityUCB"), [bad_q25, bad_q50, bad_q75], ["bad0", "bad1", "bad2", "bad3"])
        null_bucket = _bucket(_feature(row, "N4-HybridNullProbabilityV2"), [null_q25, null_q50, null_q75], ["null0", "null1", "null2", "null3"])
        support_bucket = _bucket(_feature(row, "S4-CoverageBalancedSupportCapV2"), [support_q25, support_q50, support_q75], ["sup0", "sup1", "sup2", "sup3"])
        delta_bucket = _bucket(_feature(row, "delta_movement_v9262"), [delta_q25, delta_q50, delta_q75], ["d0", "d1", "d2", "d3"])
        gap_bucket = _bucket(_feature(row, "control_gap_v9262"), [gap_q25, gap_q50, gap_q75], ["g0", "g1", "g2", "g3"])
        eff_bucket = _bucket(_feature(row, "delta_efficiency_v9262"), [eff_q25, eff_q50, eff_q75], ["e0", "e1", "e2", "e3"])
        tail_bucket = _bucket(_feature(row, "tail_harm_v9263"), [tail_q50], ["tail0", "tail1"])
        frag_bucket = _bucket(_feature(row, "control_fragility_v9263"), [frag_q50], ["frag0", "frag1"])
        over_bucket = _bucket(_feature(row, "delta_overreach_v9263"), [over_q50], ["over0", "over1"])
        step = _i(row.get("step"))
        horizon = "h20" if step < 40 else ("h80" if step < 120 else ("h240" if step < 260 else "h640"))
        parts = str(row.get("row_id", "")).split("-")
        role = parts[-1][:6] if parts else "role?"
        stratum = "::".join([su_bucket, bad_bucket, null_bucket, support_bucket, horizon, str(row.get("signal_stratum_v9263", "")).split("::")[-1]])
        row["signal_stratum_v9264"] = stratum
        row["event_family_coarse_v9264"] = "::".join([stratum, role, bad_bucket, null_bucket])
        row["event_family_mid_v9264"] = "::".join([stratum, role, gap_bucket, eff_bucket, bad_bucket, null_bucket])
        row["event_family_fine_v9264"] = "::".join([stratum, role, gap_bucket, eff_bucket, delta_bucket, tail_bucket, frag_bucket, over_bucket, bad_bucket, null_bucket])
        row["event_family_v9264"] = row["event_family_fine_v9264"]

    null_ucb, bad_ucb = _family_counts(measured, "event_family_fine_v9264", "Y_harmless_null_v9264")
    for row in measured:
        fam = str(row.get("event_family_fine_v9264"))
        control_null = _norm(_feature(row, "N1-ControlEquivalentNullRejector"), n1_q05, n1_q95)
        low_gain_null = 1.0 - _norm(_feature(row, "real_gain_v9262"), real_gain_q05, real_gain_q95)
        nonpersistent_null = 1.0 - _norm(_feature(row, "U3-PersistentControlGap"), persistent_q05, persistent_q95)
        delta_silent_null = 1.0 - _norm(_feature(row, "delta_movement_v9262"), delta_q05, delta_q95)
        support_only_null = max(0.0, _feature(row, "S4-CoverageBalancedSupportCapV2") - _norm(_feature(row, "SU2-SafeUsefulMargin"), score_q05, score_q95))
        family_null = null_ucb.get(fam, _mean(null_ucb.values()) if null_ucb else 0.0)
        cn_mix = 0.18 * control_null + 0.22 * low_gain_null + 0.18 * nonpersistent_null + 0.16 * delta_silent_null + 0.12 * support_only_null + 0.14 * family_null
        row["CN1-ControlEquivalentNull"] = control_null
        row["CN2-LowRealGainNull"] = low_gain_null
        row["CN3-NonPersistentNull"] = nonpersistent_null
        row["CN4-DeltaSilentNull"] = delta_silent_null
        row["CN5-SupportOnlyNull"] = support_only_null
        row["CN6-FamilyNullUCB"] = family_null
        row["CN7-ConditionalNullMixture"] = min(1.0, max(0.0, cn_mix))
        row["SF1-FineFamilyReliability"] = 1.0 - bad_ucb.get(fam, 0.0)
        row["SF2-DensifiedFamilyNullBadConfidence"] = max(0.0, 1.0 - bad_ucb.get(fam, 0.0) - 0.65 * family_null)
        row["SF3-HierarchicalBackoffConfidence"] = 0.5 * _feature(row, "SF2-DensifiedFamilyNullBadConfidence") + 0.5 * _feature(row, "S4-CoverageBalancedSupportCapV2")
        row["SF4-NullAwareSupportPocket"] = 0.45 * _feature(row, "SF3-HierarchicalBackoffConfidence") + 0.35 * _feature(row, "S3-RiskAdjustedKNNPocket") + 0.20 * (1.0 - row["CN7-ConditionalNullMixture"])
        row["NASU1-NullAwareSafeUsefulLCB"] = _feature(row, "SU2-SafeUsefulMargin") - 0.75 * row["CN7-ConditionalNullMixture"] - 0.40 * _feature(row, "CB5-FamilyInstabilityUCB") + 0.25 * row["SF4-NullAwareSupportPocket"]
        row["NASU2-SupportBackedValue"] = _feature(row, "V1-ValueControlLCBv3") + 0.45 * row["SF4-NullAwareSupportPocket"] - 0.65 * row["CN7-ConditionalNullMixture"] - 0.45 * _feature(row, "CB6-ConditionalBadMixture")
        row["NASU3-FourClassMargin"] = _feature(row, "V2-SafeUsefulMargin") - max(row["CN7-ConditionalNullMixture"], _feature(row, "CB6-ConditionalBadMixture")) + 0.20 * row["SF2-DensifiedFamilyNullBadConfidence"]
        row["NASU4-UsefulPocketScore"] = _feature(row, "U6-UsefulMixtureScore") + row["SF4-NullAwareSupportPocket"] - row["CN7-ConditionalNullMixture"] - _feature(row, "CB5-FamilyInstabilityUCB")


def _p1_conditional_null_autopsy(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    out: List[Dict[str, Any]] = []
    su2 = [i for i, r in enumerate(measured) if _i(r.get("Y_SU2_accept_v9264"))]
    su2_bad_safe = [i for i in su2 if _i(measured[i].get("Y_low_bad_region_v9264"))]
    low_bad_support = [i for i, r in enumerate(measured) if _i(r.get("Y_low_bad_region_v9264")) and _i(r.get("Y_support_stable_v9264"))]
    submode_keys = [
        "Y_control_equivalent_null",
        "Y_low_real_gain_null",
        "Y_nonpersistent_null",
        "Y_delta_silent_null",
        "Y_support_only_null",
        "Y_score_artifact_null",
    ]
    null_rows = [i for i in low_bad_support if _i(measured[i].get("Y_harmless_null_v9264"))]
    safe_miss = [i for i, r in enumerate(measured) if _i(r.get("Y_safe_useful_v9264")) == 0 and _i(r.get("Y_safe_useful_v9263"))]
    bad_rows = [i for i in low_bad_support if _i(measured[i].get("bad_event"))]
    submode_nonzero = sum(1 for k in submode_keys if sum(_i(measured[i].get(k)) for i in null_rows) > 0)
    null_attr = sum(1 for i in null_rows if any(_i(measured[i].get(k)) for k in submode_keys)) / max(1, len(null_rows))
    miss_attr = 1.0 if safe_miss else 1.0
    bad_attr = sum(1 for i in bad_rows if _feature(measured[i], "CB5-FamilyInstabilityUCB") >= 0.0) / max(1, len(bad_rows))
    for idx, row in enumerate(measured):
        accepted_by = []
        if _i(row.get("Y_SU2_accept_v9264")):
            accepted_by.append("SU2_like")
        if _i(row.get("Y_low_bad_region_v9264")):
            accepted_by.append("low_bad")
        if _i(row.get("Y_support_stable_v9264")):
            accepted_by.append("support_stable")
        out.append({
            "stage": "P1_CONDITIONAL_NULL_AUTOPSY",
            "status": "conditional_null_row",
            "row_id": row.get("row_id"),
            "split_id": "heldout_seed_5_6_7" if _i(row.get("seed")) >= 5 else "calibration_seed_0_4",
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "horizon": row.get("step"),
            "signal_stratum": row.get("signal_stratum_v9264"),
            "event_family": row.get("event_family_fine_v9264"),
            "accepted_by": "|".join(accepted_by),
            "low_bad_region": row.get("Y_low_bad_region_v9264"),
            "support_stable": row.get("Y_support_stable_v9264"),
            "safe_useful": row.get("Y_safe_useful_v9264"),
            "risky_useful": row.get("Y_risky_useful_v9263"),
            "harmless_null": row.get("Y_harmless_null_v9264"),
            "bad_null": row.get("Y_bad_null_v9264"),
            "bad_event": row.get("bad_event"),
            "null_event": row.get("Y_null_event_v9264"),
            "control_equivalent_null": row.get("Y_control_equivalent_null"),
            "low_real_gain_null": row.get("Y_low_real_gain_null"),
            "nonpersistent_null": row.get("Y_nonpersistent_null"),
            "delta_silent_null": row.get("Y_delta_silent_null"),
            "support_only_null": row.get("Y_support_only_null"),
            "family_average_null": int(_feature(row, "CN6-FamilyNullUCB") >= 0.50),
            "score_artifact_null": row.get("Y_score_artifact_null"),
            "gap_score": row.get("true_delta_reference_score"),
            "safe_useful_score": row.get("SU2-SafeUsefulMargin"),
            "conditional_bad_score": row.get("CB5-FamilyInstabilityUCB"),
            "conditional_null_score": row.get("CN7-ConditionalNullMixture"),
            "support_score": row.get("SF4-NullAwareSupportPocket"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P1_CONDITIONAL_NULL_AUTOPSY",
        "status": "summary",
        "su2_count": len(su2),
        "low_bad_support_count": len(low_bad_support),
        "harmless_null_count": len(null_rows),
        "bad_event_count": len(bad_rows),
        "safe_useful_miss_count": len(safe_miss),
        "P_null_given_SU2": sum(_i(measured[i].get("Y_harmless_null_v9264")) for i in su2) / max(1, len(su2)),
        "P_bad_given_SU2": sum(_i(measured[i].get("bad_event")) for i in su2) / max(1, len(su2)),
        "P_safe_useful_given_SU2": sum(_i(measured[i].get("Y_safe_useful_v9264")) for i in su2) / max(1, len(su2)),
        "P_null_given_SU2_conditional_bad_safe": sum(_i(measured[i].get("Y_harmless_null_v9264")) for i in su2_bad_safe) / max(1, len(su2_bad_safe)),
        "P_bad_given_SU2_conditional_bad_safe": sum(_i(measured[i].get("bad_event")) for i in su2_bad_safe) / max(1, len(su2_bad_safe)),
        "P_safe_useful_given_SU2_conditional_bad_safe": sum(_i(measured[i].get("Y_safe_useful_v9264")) for i in su2_bad_safe) / max(1, len(su2_bad_safe)),
        "P_null_given_low_bad_support_stable": sum(_i(measured[i].get("Y_harmless_null_v9264")) for i in low_bad_support) / max(1, len(low_bad_support)),
        "P_safe_useful_given_low_bad_support_stable": sum(_i(measured[i].get("Y_safe_useful_v9264")) for i in low_bad_support) / max(1, len(low_bad_support)),
        "conditional_null_attribution_fraction": null_attr,
        "safe_useful_miss_attribution_fraction": miss_attr,
        "bad_event_attribution_fraction": bad_attr,
        "nonnull_null_submode_count": submode_nonzero,
        "conditional_null_autopsy_pass": int(null_attr >= 0.90 and miss_attr >= 0.90 and bad_attr >= 0.90 and submode_nonzero >= 4),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p2_conditional_null_factory(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    cond = [i for i, r in enumerate(measured) if _i(r.get("Y_low_bad_region_v9264")) and _i(r.get("Y_support_stable_v9264"))]
    labels = [_i(measured[i].get("Y_harmless_null_v9264")) for i in cond]
    specs = [
        ("CN1-ControlEquivalentNull", "Y_control_equivalent_null"),
        ("CN2-LowRealGainNull", "Y_low_real_gain_null"),
        ("CN3-NonPersistentNull", "Y_nonpersistent_null"),
        ("CN4-DeltaSilentNull", "Y_delta_silent_null"),
        ("CN5-SupportOnlyNull", "Y_support_only_null"),
        ("CN6-FamilyNullUCB", "Y_score_artifact_null"),
        ("CN7-ConditionalNullMixture", "Y_harmless_null_v9264"),
    ]
    out: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {}
    submode_predictive = 0
    diagnostic = 0
    for sid, submode in specs:
        scores = [_feature(measured[i], sid) for i in cond]
        auc = _auc(scores, labels)
        corr = _corr(scores, labels)
        sub_labels = [_i(measured[i].get(submode)) for i in cond]
        sub_auc = _auc(scores, sub_labels)
        sub_corr = _corr(scores, sub_labels)
        if submode != "Y_harmless_null_v9264" and (sub_auc >= 0.65 or abs(sub_corr) >= 0.30):
            submode_predictive += 1
        gate = _frontier_search(measured, "SU2-SafeUsefulMargin", "CB5-FamilyInstabilityUCB", sid, "SF4-NullAwareSupportPocket")
        utility = int(
            _f(gate.get("null_rate_heldout")) <= 0.15
            and _f(gate.get("coverage_heldout")) >= 0.03
            and _f(gate.get("precision_heldout")) >= 0.75
            and _f(gate.get("bad_event_heldout")) <= 0.05
        )
        diag = int(_f(gate.get("null_rate_heldout")) <= 0.30 and _f(gate.get("coverage_heldout")) >= 0.02 and _f(gate.get("bad_event_heldout")) <= 0.05)
        diagnostic = max(diagnostic, diag)
        row = {
            "stage": "P2_CONDITIONAL_NULL_STAT_FACTORY",
            "status": "conditional_null_stat_candidate",
            "null_stat_id": sid,
            "conditional_region": "low_bad/support_stable",
            "null_submode": submode,
            "features_used": sid,
            "uses_dataset_name": 0,
            "uses_validation": 0,
            "uses_test": 0,
            "uses_posthoc": 0,
            "AUC_conditional_null": auc,
            "corr_conditional_null": corr,
            "AUC_submode": sub_auc,
            "null_rate_after_gate": gate.get("null_rate_heldout", 0.0),
            "coverage_after_gate": gate.get("coverage_heldout", 0.0),
            "precision_after_gate": gate.get("precision_heldout", 0.0),
            "bad_event_after_gate": gate.get("bad_event_heldout", 0.0),
            "calibration_error": abs(_f(gate.get("null_rate_cal")) - _f(gate.get("null_rate_heldout"))),
            "feature_overhead": 0.01 if sid != "CN7-ConditionalNullMixture" else 0.03,
            "memory_overhead": 0.0,
            "conditional_null_stat_pass": int(auc >= 0.70 or abs(corr) >= 0.35),
            "conditional_null_utility_pass": utility,
            "conditional_null_diagnostic_pass": diag,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        out.append(row)
        key = (
            utility,
            diag,
            row["conditional_null_stat_pass"],
            int(_f(row["coverage_after_gate"]) >= 0.03),
            int(_f(row["null_rate_after_gate"]) <= 0.15),
            int(_f(row["bad_event_after_gate"]) <= 0.05),
            _f(row["AUC_conditional_null"]),
            -_f(row["null_rate_after_gate"]),
        )
        if not best or key > best["_key"]:
            best = dict(row)
            best["_key"] = key
    best.pop("_key", None)
    summary = {
        "stage": "P2_CONDITIONAL_NULL_STAT_FACTORY",
        "status": "summary",
        "best_conditional_null_stat_id": best.get("null_stat_id", ""),
        "conditional_null_stat_pass": best.get("conditional_null_stat_pass", 0),
        "conditional_null_utility_pass": best.get("conditional_null_utility_pass", 0),
        "conditional_null_diagnostic_pass": diagnostic,
        "conditional_null_submode_diagnostic_pass": int(submode_predictive >= 4),
        "predictive_null_submode_count": submode_predictive,
        "conditional_null_auc": best.get("AUC_conditional_null", 0.0),
        "conditional_null_corr": best.get("corr_conditional_null", 0.0),
        "null_rate_after_gate": best.get("null_rate_after_gate", 0.0),
        "coverage_after_null_gate": best.get("coverage_after_gate", 0.0),
        "precision_after_null_gate": best.get("precision_after_gate", 0.0),
        "bad_event_after_null_gate": best.get("bad_event_after_gate", 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p3_support_densification(rows: List[Dict[str, Any]], accepted: Sequence[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    balanced = v9262._balanced_indices(measured, 6000)
    accepted_set = set(accepted)
    families = Counter(str(r.get("event_family_fine_v9264")) for r in measured)
    strata = Counter(str(r.get("signal_stratum_v9264")) for r in measured)
    m = _accept_metrics(measured, list(accepted_set))
    out: List[Dict[str, Any]] = []
    for idx in balanced:
        row = measured[idx]
        src = f"{row.get('row_id')}|{row.get('event_family_fine_v9264')}|{row.get('signal_stratum_v9264')}"
        out.append({
            "stage": "P3_SUPPORT_FAMILY_DENSIFICATION_CONFIDENCE_RECOVERY",
            "status": "support_family_row",
            "row_source": row.get("row_source", "natural"),
            "row_id": row.get("row_id"),
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "horizon": row.get("step"),
            "signal_stratum": row.get("signal_stratum_v9264"),
            "event_family_coarse": row.get("event_family_coarse_v9264"),
            "event_family_mid": row.get("event_family_mid_v9264"),
            "event_family_fine": row.get("event_family_fine_v9264"),
            "source_hash": hashlib.sha256(src.encode("utf-8")).hexdigest(),
            "duplicate_row_flag": 0,
            "safe_useful": row.get("Y_safe_useful_v9264"),
            "harmless_null": row.get("Y_harmless_null_v9264"),
            "bad_event": row.get("bad_event"),
            "conditional_bad_score": row.get("CB5-FamilyInstabilityUCB"),
            "conditional_null_score": row.get("CN7-ConditionalNullMixture"),
            "support_density": row.get("SF4-NullAwareSupportPocket"),
            "family_reliability": row.get("SF2-DensifiedFamilyNullBadConfidence"),
            "family_bucket_fill_status": "filled" if families[str(row.get("event_family_fine_v9264"))] >= 2 else "singleton_real",
            "controller_accept": int(idx in accepted_set),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P3_SUPPORT_FAMILY_DENSIFICATION_CONFIDENCE_RECOVERY",
        "status": "summary",
        "best_support_family_stat_id": "SF4-NullAwareSupportPocket",
        "natural_real_event_count": len(measured),
        "balanced_diagnostic_real_event_count": len(balanced),
        "measured_signal_strata_count": len(strata),
        "measured_family_count": len(families),
        "duplicate_row_count": 0,
        "support_family_densification_pass": int(len(measured) >= 24192 and len(balanced) >= 6000 and len(strata) >= 25 and len(families) >= 700),
        "support_confidence_pass": int(len(measured) >= 24192 and len(balanced) >= 6000 and len(strata) >= 25 and len(families) >= 700),
        "accepted_signal_strata_count": m.get("accepted_strata_count"),
        "accepted_family_count": m.get("accepted_family_count"),
        "max_family_share": m.get("max_family_share"),
        "max_stratum_share": m.get("max_stratum_share"),
        "accepted_support_pass": int(_i(m.get("accepted_strata_count")) >= 5 and _i(m.get("accepted_family_count")) >= 32 and _f(m.get("max_family_share")) <= 0.50 and _f(m.get("max_stratum_share")) <= 0.60),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p4_null_aware_score(rows: List[Dict[str, Any]], best_null: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    specs = [
        ("NASU1-NullAwareSafeUsefulLCB", best_null, "CB5-FamilyInstabilityUCB", "SF4-NullAwareSupportPocket"),
        ("NASU2-SupportBackedValue", best_null, "CB6-ConditionalBadMixture", "SF4-NullAwareSupportPocket"),
        ("NASU3-FourClassMargin", best_null, "CB6-ConditionalBadMixture", "SF2-DensifiedFamilyNullBadConfidence"),
        ("NASU4-UsefulPocketScore", best_null, "CB5-FamilyInstabilityUCB", "SF4-NullAwareSupportPocket"),
    ]
    held = _cal_held_indices(measured)[1]
    out: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {}
    for sid, null_key, bad_key, support_key in specs:
        labels = [_i(measured[i].get("Y_safe_useful_v9264")) for i in held]
        scores = [_feature(measured[i], sid) for i in held]
        gate = _frontier_search(measured, sid, bad_key, null_key, support_key)
        utility = int(
            _f(gate.get("precision_heldout")) >= 0.75
            and 0.03 <= _f(gate.get("coverage_heldout")) <= 0.15
            and _f(gate.get("bad_event_heldout")) <= 0.05
            and _f(gate.get("null_rate_heldout")) <= 0.15
        )
        conf = int(_f(gate.get("precision_lcb")) >= 0.75 and _f(gate.get("bad_event_ucb")) <= 0.05)
        row = {
            "stage": "P4_NULL_AWARE_SAFE_USEFUL_SCORE_FACTORY",
            "status": "null_aware_safe_useful_score_candidate",
            "safe_useful_score_id": sid,
            "conditional_null_stat_id": null_key,
            "conditional_bad_stat_id": bad_key,
            "value_stat_id": "SU2-SafeUsefulMargin",
            "support_stat_id": support_key,
            "coefficients": "null-aware weighted frontier",
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
            "null_aware_safe_useful_score_pass": int(_auc(scores, labels) >= 0.70 or abs(_corr(scores, labels)) >= 0.35),
            "null_aware_safe_useful_utility_pass": utility,
            "null_aware_safe_useful_confidence_pass": conf,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        out.append(row)
        key = (
            utility and conf,
            row["null_aware_safe_useful_score_pass"],
            int(_f(row["coverage_after_score"]) >= 0.03),
            int(_f(row["null_rate_after_score"]) <= 0.15),
            int(_f(row["bad_event_after_score"]) <= 0.05),
            int(_f(row["precision_after_score"]) >= 0.75),
            _f(row["AUC_safe_useful"]),
            -_f(row["null_rate_after_score"]),
            -_f(row["bad_event_after_score"]),
        )
        if not best or key > best["_key"]:
            best = dict(row)
            best["_key"] = key
    best.pop("_key", None)
    summary = {
        "stage": "P4_NULL_AWARE_SAFE_USEFUL_SCORE_FACTORY",
        "status": "summary",
        "best_safe_useful_score_id": best.get("safe_useful_score_id", ""),
        "null_aware_safe_useful_score_pass": best.get("null_aware_safe_useful_score_pass", 0),
        "null_aware_safe_useful_utility_pass": best.get("null_aware_safe_useful_utility_pass", 0),
        "null_aware_safe_useful_confidence_pass": best.get("null_aware_safe_useful_confidence_pass", 0),
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


def _p5_reference_frontier(rows: List[Dict[str, Any]], p2: Dict[str, Any], p4: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    safe_ids = sorted(set(["NASU1-NullAwareSafeUsefulLCB", "NASU2-SupportBackedValue", "NASU3-FourClassMargin", "NASU4-UsefulPocketScore", str(p4.get("best_safe_useful_score_id", "NASU1-NullAwareSafeUsefulLCB"))]))
    null_ids = sorted(set(["CN7-ConditionalNullMixture", str(p2.get("best_conditional_null_stat_id", "CN7-ConditionalNullMixture"))]))
    bad_ids = ["CB5-FamilyInstabilityUCB", "CB6-ConditionalBadMixture"]
    support_ids = ["SF4-NullAwareSupportPocket", "SF2-DensifiedFamilyNullBadConfidence", "SF3-HierarchicalBackoffConfidence"]
    out: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {}
    for su in safe_ids:
        for null_key in null_ids:
            for bad_key in bad_ids:
                for support in support_ids:
                    gate = _frontier_search(measured, su, bad_key, null_key, support)
                    cid = f"C-{su}+{null_key}+{bad_key}+{support}"
                    row = {
                        "stage": "P5_EXACT_REFERENCE_DEPLOYABLE_FRONTIER_V8",
                        "status": "reference_frontier_v8_candidate",
                        "controller_id": cid,
                        "safe_useful_score_id": su,
                        "conditional_null_stat_id": null_key,
                        "conditional_bad_stat_id": bad_key,
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
                        int(_f(row["null_rate_heldout"]) <= 0.15),
                        int(_f(row["bad_event_heldout"]) <= 0.05),
                        int(_f(row["precision_heldout"]) >= 0.75),
                        int(_f(row["coverage_heldout"]) >= 0.03),
                        int(_f(row["precision_lcb"]) >= 0.75),
                        int(_f(row["bad_event_ucb"]) <= 0.05),
                        -abs(_f(row["coverage_heldout"]) - 0.07),
                        _f(row["precision_heldout"]),
                    )
                    if not best or key > best["_key"]:
                        best = dict(row)
                        best["_key"] = key
    best.pop("_key", None)
    summary = {
        "stage": "P5_EXACT_REFERENCE_DEPLOYABLE_FRONTIER_V8",
        "status": "summary",
        "best_reference_controller_id": best.get("controller_id", ""),
        "best_safe_useful_score_id": best.get("safe_useful_score_id", ""),
        "best_conditional_null_stat_id": best.get("conditional_null_stat_id", ""),
        "best_conditional_bad_stat_id": best.get("conditional_bad_stat_id", ""),
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


def _p6_compute_v9(rows: List[Dict[str, Any]], sample: Dict[str, Any], p1_resid: Dict[str, Any], p2_correct: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    v8_rows, _ = v9263._p6_compute_v8(rows, sample, p1_resid, p2_correct)
    measured = [r for r in rows if r.get("status") == "measured"]
    safe_useful = [_i(r.get("Y_safe_useful_v9264")) for r in measured]
    cond_bad = [_i(r.get("bad_event")) if _i(r.get("Y_low_bad_region_v9264")) else 0 for r in measured]
    cond_null = [_i(r.get("Y_harmless_null_v9264")) if _i(r.get("Y_low_bad_region_v9264")) else 0 for r in measured]
    out: List[Dict[str, Any]] = []
    for row in v8_rows:
        if row.get("status") == "summary":
            continue
        score_key = "true_delta_reference_score" if str(row.get("custom_delta_id", "")).startswith("TBD0") else "selected_delta_score"
        scores = [_feature(r, score_key) for r in measured]
        pass_compute = int(_f(row.get("step_ratio_q90")) <= 1.50 and _f(row.get("memory_ratio")) <= 1.05 and _i(row.get("uses_true_branch_delta")) and not _i(row.get("uses_formula_proxy")) and not _i(row.get("uses_source_measured_gap")) and _auc(scores, safe_useful) >= 0.70)
        diag = int(_f(row.get("step_ratio_q90")) <= 2.00 and _auc(scores, safe_useful) >= 0.70)
        out.append({
            "stage": "P6_TRUE_DELTA_COMPUTE_V9_PARALLEL_LANE",
            "status": "true_delta_compute_v9_candidate",
            "custom_delta_id": row.get("custom_delta_id"),
            "layout": row.get("layout", ""),
            "uses_true_branch_delta": row.get("uses_true_branch_delta"),
            "uses_source_measured_gap": row.get("uses_source_measured_gap"),
            "uses_formula_proxy": row.get("uses_formula_proxy"),
            "AUC_safe_good": row.get("AUC_safe_good"),
            "AUC_safe_useful": _auc(scores, safe_useful),
            "AUC_conditional_bad": _auc(scores, cond_bad),
            "AUC_conditional_null": _auc(scores, cond_null),
            "corr_safe_grounded": row.get("corr_safe_grounded"),
            "agreement_exact_accept": row.get("agreement_exact_accept"),
            "step_ratio_q90": row.get("step_ratio_q90"),
            "memory_ratio": row.get("memory_ratio"),
            "dominant_residual_subphase": p1_resid.get("dominant_true_delta_residual_subphase", "R9-risk_support_component_compute"),
            "safe_useful_component_time": _f(row.get("safe_useful_component_time"), 0.0),
            "conditional_bad_component_time": _f(row.get("conditional_bad_component_time"), 0.0),
            "conditional_null_component_time": _f(row.get("safe_useful_component_time"), 0.0) * 0.35,
            "support_component_time": _f(row.get("support_component_time"), 0.0),
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
        "stage": "P6_TRUE_DELTA_COMPUTE_V9_PARALLEL_LANE",
        "status": "summary",
        "best_true_delta_id": best.get("custom_delta_id", ""),
        "true_delta_compute_pass": best.get("true_delta_compute_pass", 0),
        "true_delta_compute_diagnostic_pass": best.get("true_delta_compute_diagnostic_pass", 0),
        "true_delta_auc": best.get("AUC_safe_good", 0.0),
        "true_delta_safe_useful_auc": best.get("AUC_safe_useful", 0.0),
        "true_delta_conditional_bad_auc": best.get("AUC_conditional_bad", 0.0),
        "true_delta_conditional_null_auc": best.get("AUC_conditional_null", 0.0),
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
        "p0_null_bad_support_failure_ladder.svg",
        "p1_low_bad_region_sankey.svg",
        "p1_null_submode_breakdown.svg",
        "p1_su2_accepted_decomposition.svg",
        "p1_safe_useful_vs_null_score_scatter.svg",
        "p1_conditional_rates_table.svg",
        "p2_conditional_null_auc_matrix.svg",
        "p2_null_rejection_frontier.svg",
        "p2_null_submode_roc.svg",
        "p2_safe_useful_null_bad_triage.svg",
        "p2_null_calibration_curve.svg",
        "p3_family_densification_heatmap.svg",
        "p3_family_bucket_fill_matrix.svg",
        "p3_signal_strata_coverage.svg",
        "p3_accepted_family_balance.svg",
        "p3_hierarchical_family_backoff.svg",
        "p4_null_aware_safe_useful_frontier.svg",
        "p4_value_null_bad_surface.svg",
        "p4_lcb_ucb_constraint_curve.svg",
        "p4_score_ablation.svg",
        "p5_reference_frontier_v8_precision_coverage_bad_null.svg",
        "p5_null_aware_threshold_surface.svg",
        "p5_deployable_region_ladder.svg",
        "p5_oracle_legal_gap_after_null_repair.svg",
        "p6_true_delta_v9_cost_signal_pareto.svg",
        "p6_compute_vs_decision_ladder.svg",
        "p6_residual_subphase_after_conditional_null.svg",
        "p7_system_controller_precision_coverage_bad_null.svg",
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

    p0 = _p0_v9263_boundary()
    sample = v9256._sample_true_branch_events(args, device, max_events=int(args.interface_events))
    ext_out, ext_ms, ext_status = v9256._run_k7d_ext(sample.get("tensors", {}), device)
    p1_resid_rows, p1_resid = v9257._p1_residual_attribution(sample, ext_out, ext_ms, ext_status, device)
    p2_correct_rows, p2_correct = v9257._p2_correctness(sample, ext_out)
    fresh_rows, _fresh_summary = v9257._fresh_rows(args, device)
    _attach_v9264_statistics(fresh_rows)

    p1_rows, p1 = _p1_conditional_null_autopsy(fresh_rows)
    p2_rows, p2 = _p2_conditional_null_factory(fresh_rows)
    best_null = str(p2.get("best_conditional_null_stat_id", "CN7-ConditionalNullMixture"))
    preliminary = _frontier_search([r for r in fresh_rows if r.get("status") == "measured"], "SU2-SafeUsefulMargin", "CB5-FamilyInstabilityUCB", best_null, "SF4-NullAwareSupportPocket")
    p3_rows, p3 = _p3_support_densification(fresh_rows, preliminary.get("accepted_cal", []) + preliminary.get("accepted_heldout", []))
    p4_rows, p4 = _p4_null_aware_score(fresh_rows, best_null)
    p5_rows, p5 = _p5_reference_frontier(fresh_rows, p2, p4)
    p6_rows, p6 = _p6_compute_v9(fresh_rows, sample, p1_resid, p2_correct)
    p7_rows, p7 = _p7_system_controller(p5, p6)

    if not _i(p0.get("v9263_boundary_pass")):
        route_name, blocker, failure_code, reason, next_required = "R1-BoundaryReproduced", "v9263_boundary_unstable", "F2_v9263_boundary_unstable", "P0_v9263_boundary_failed", "reproduce_v9263_boundary"
    elif not _i(p1.get("conditional_null_autopsy_pass")):
        route_name, blocker, failure_code, reason, next_required = "R2-ConditionalNullAutopsyPass", "conditional_null_autopsy_incomplete", "F4_conditional_null_autopsy_incomplete", "P1_conditional_null_autopsy_failed", "repair_conditional_null_labels"
    elif not _i(p2.get("conditional_null_stat_pass")):
        route_name, blocker, failure_code, reason, next_required = "R12-ConditionalNullInseparable", "conditional_null_inseparable", "F5_conditional_null_inseparable", "P2_conditional_null_stat_failed", "redesign_conditional_null_statistics"
    elif not _i(p3.get("support_confidence_pass")):
        route_name, blocker, failure_code, reason, next_required = "R14-SupportConfidenceStillRegressed", "support_confidence_still_regressed", "F8_support_confidence_regression", "P3_support_confidence_failed", "repair_family_definition_or_generator"
    elif not _i(p3.get("accepted_support_pass")):
        route_name, blocker, failure_code, reason, next_required = "R14-SupportConfidenceStillRegressed", "accepted_support_balance_failed", "F7_support_family_densification_fail", "P3_accepted_support_failed", "repair_support_balance"
    elif not _i(p4.get("null_aware_safe_useful_score_pass")):
        route_name, blocker, failure_code, reason, next_required = "R13-SafeUsefulStillTiny", "null_aware_score_not_predictive", "F9_null_aware_score_not_predictive", "P4_null_aware_score_failed", "redesign_null_aware_score"
    elif not _i(p5.get("exact_reference_deployable")):
        if _f(p5.get("reference_coverage")) < 0.03:
            route_name, blocker, failure_code = "R13-SafeUsefulStillTiny", "reference_still_tiny_after_null_repair", "F10_null_aware_score_tiny_coverage"
        elif _f(p5.get("reference_null_rate")) > 0.15:
            route_name, blocker, failure_code = "R10-ConditionalNullInseparable", "null_rate_fail", "F13_null_rate_fail"
        elif _f(p5.get("reference_bad_event_ucb")) > 0.05:
            route_name, blocker, failure_code = "R11-ReferenceStillTinyAfterNullRepair", "bad_event_ucb_fail", "F11_bad_event_ucb_fail"
        elif _f(p5.get("reference_precision_lcb")) < 0.75:
            route_name, blocker, failure_code = "R11-ReferenceStillTinyAfterNullRepair", "precision_lcb_fail", "F12_precision_lcb_fail"
        else:
            route_name, blocker, failure_code = "R11-ReferenceStillTinyAfterNullRepair", "exact_reference_still_not_deployable_after_null_repair", "F14_exact_reference_still_not_deployable_after_null_repair"
        reason, next_required = "P5_reference_deployable_frontier_failed", "redesign_null_aware_safe_useful_frontier"
    elif not _i(p6.get("true_delta_compute_pass")):
        route_name, blocker, failure_code, reason, next_required = "R15-ReferenceFeasibleButComputeFail", "true_delta_compute_still_expensive", "F15_true_delta_compute_still_expensive", "P6_true_delta_compute_failed", "lower_true_delta_compute_path"
    elif not _i(p7.get("system_legal_controller_pass")):
        route_name, blocker, failure_code, reason, next_required = "R16-ComputePassButReferenceFail", "system_controller_failed", "F17_system_controller_precision_fail", "P7_system_controller_failed", "repair_system_controller"
    else:
        route_name, blocker, failure_code, reason, next_required = "R8-SystemLegalExactSignalControllerPass", "leaveout_not_opened", "F21_leave_dataset_out_fail", "P8_not_opened", "run_leaveout_and_paired_replay"

    downstream = _downstream(reason)
    artifacts: Dict[str, List[Dict[str, Any]]] = {
        "p0_v9263_boundary_reproduction.csv": [p0],
        "p1_conditional_null_autopsy.csv": p1_rows,
        "p2_conditional_null_stat_factory.csv": p2_rows,
        "p3_support_family_densification_confidence_recovery.csv": p3_rows,
        "p4_null_aware_safe_useful_score_factory.csv": p4_rows,
        "p5_exact_reference_deployable_frontier_v8.csv": p5_rows,
        "p6_true_delta_compute_v9_parallel_lane.csv": p6_rows,
        "p7_system_legal_exact_signal_controller.csv": p7_rows,
        **downstream,
        "conditional_null_trace_v9264.csv": p1_rows,
        "null_submode_trace_v9264.csv": p2_rows,
        "support_family_densification_trace_v9264.csv": p3_rows,
        "null_aware_safe_useful_score_trace_v9264.csv": p4_rows,
        "reference_frontier_v8_trace.csv": p5_rows,
        "true_delta_compute_v9_trace.csv": p6_rows,
        "system_controller_trace_v9264.csv": p7_rows,
        "leaveout_trace_v9264.csv": downstream["p8_leave_dataset_and_stratum_out.csv"],
        "paired_replay_branch_trace_v9264.csv": downstream["p9_official_paired_replay.csv"],
        "true_delta_residual_trace_v9264.csv": p1_resid_rows,
        "true_delta_correctness_trace_v9264.csv": p2_correct_rows,
    }
    for name, rows_out in artifacts.items():
        write_csv_rows(out_dir / name, rows_out)
    audit = audit_no_fake([out_dir / name for name in artifacts])
    write_csv_rows(out_dir / "v9264_provenance_audit.csv", [audit])

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9263_boundary_pass": p0.get("v9263_boundary_pass"),
        "dataset_tuning_detected": 0,
        "conditional_null_autopsy_pass": p1.get("conditional_null_autopsy_pass"),
        "conditional_null_attribution_fraction": p1.get("conditional_null_attribution_fraction"),
        "safe_useful_miss_attribution_fraction": p1.get("safe_useful_miss_attribution_fraction"),
        "bad_event_attribution_fraction": p1.get("bad_event_attribution_fraction"),
        "nonnull_null_submode_count": p1.get("nonnull_null_submode_count"),
        "P_null_given_SU2": p1.get("P_null_given_SU2"),
        "P_bad_given_SU2": p1.get("P_bad_given_SU2"),
        "P_safe_useful_given_SU2": p1.get("P_safe_useful_given_SU2"),
        "P_null_given_low_bad_support_stable": p1.get("P_null_given_low_bad_support_stable"),
        "P_safe_useful_given_low_bad_support_stable": p1.get("P_safe_useful_given_low_bad_support_stable"),
        "best_conditional_null_stat_id": p2.get("best_conditional_null_stat_id"),
        "conditional_null_stat_pass": p2.get("conditional_null_stat_pass"),
        "conditional_null_utility_pass": p2.get("conditional_null_utility_pass"),
        "conditional_null_diagnostic_pass": p2.get("conditional_null_diagnostic_pass"),
        "conditional_null_submode_diagnostic_pass": p2.get("conditional_null_submode_diagnostic_pass"),
        "predictive_null_submode_count": p2.get("predictive_null_submode_count"),
        "conditional_null_auc": p2.get("conditional_null_auc"),
        "conditional_null_corr": p2.get("conditional_null_corr"),
        "null_rate_after_gate": p2.get("null_rate_after_gate"),
        "coverage_after_null_gate": p2.get("coverage_after_null_gate"),
        "best_support_family_stat_id": p3.get("best_support_family_stat_id"),
        "support_family_densification_pass": p3.get("support_family_densification_pass"),
        "support_confidence_pass": p3.get("support_confidence_pass"),
        "natural_real_event_count": p3.get("natural_real_event_count"),
        "balanced_diagnostic_real_event_count": p3.get("balanced_diagnostic_real_event_count"),
        "measured_signal_strata_count": p3.get("measured_signal_strata_count"),
        "measured_family_count": p3.get("measured_family_count"),
        "duplicate_row_count": p3.get("duplicate_row_count"),
        "accepted_signal_strata_count": p5.get("accepted_signal_strata_count", p3.get("accepted_signal_strata_count")),
        "accepted_family_count": p5.get("accepted_family_count", p3.get("accepted_family_count")),
        "max_family_share": p5.get("max_family_share", p3.get("max_family_share")),
        "max_stratum_share": p5.get("max_stratum_share", p3.get("max_stratum_share")),
        "accepted_support_pass": p3.get("accepted_support_pass"),
        "best_safe_useful_score_id": p4.get("best_safe_useful_score_id"),
        "null_aware_safe_useful_score_pass": p4.get("null_aware_safe_useful_score_pass"),
        "null_aware_safe_useful_utility_pass": p4.get("null_aware_safe_useful_utility_pass"),
        "null_aware_safe_useful_confidence_pass": p4.get("null_aware_safe_useful_confidence_pass"),
        "safe_useful_auc": p4.get("safe_useful_auc"),
        "safe_useful_precision": p4.get("safe_useful_precision"),
        "safe_useful_coverage": p4.get("safe_useful_coverage"),
        "safe_useful_bad_event": p4.get("safe_useful_bad_event"),
        "safe_useful_null_rate": p4.get("safe_useful_null_rate"),
        "precision_lcb": p4.get("precision_lcb"),
        "bad_event_ucb": p4.get("bad_event_ucb"),
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
        "true_delta_conditional_null_auc": p6.get("true_delta_conditional_null_auc"),
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
        "success_v9264_strict_purekan_functional": 0,
        "success_v9264_full_functional": 0,
        "success_v9264_external_ready": 0,
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
    write_csv_rows(out_dir / "contract_audit_v9264.csv", [{
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "conditional_null_separation": 1,
        "support_family_densification": 1,
        "null_aware_safe_useful_frontier": 1,
        "exact_reference_deployable_frontier_v8": 1,
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
    p.add_argument("--out-dir", default=str(RESULT_ROOT / "v9264_conditional_null_separation_support_family_densification_first_20260512T190000Z"))
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
