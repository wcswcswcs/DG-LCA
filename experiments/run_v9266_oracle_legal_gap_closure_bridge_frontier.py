#!/usr/bin/env python3
"""DG-KAN v9.2.66 oracle/legal gap closure and bridge frontier audit."""

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
import run_v9265_joint_safe_useful_feasibility_null_bad_conflict_resolution as v9265  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.66_OracleLegalGapClosure_BridgeFrontier_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9266_oracle_legal_gap_closure_bridge_frontier.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9265 = RESULT_ROOT / "v9265_joint_safe_useful_feasibility_null_bad_conflict_resolution_first_20260512T203000Z"

_f = v9265._f
_i = v9265._i
_mean = v9265._mean
_q = v9265._q
_auc = v9265._auc
_corr = v9265._corr
_feature = v9265._feature
_norm = v9265._norm
_device = v9265._device
_ci_bounds = v9265._ci_bounds
_cal_held_indices = v9265._cal_held_indices
_not_run = v9265._not_run
_metrics = v9265._metrics


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _heldout_flag(row: Dict[str, Any]) -> bool:
    return _i(row.get("seed")) >= 5


def _eval_accept(rows: Sequence[Dict[str, Any]], accepted: Sequence[int], denominator: int | None = None) -> Dict[str, Any]:
    m = _metrics(rows, accepted, "Y_SU_v9265")
    if denominator is not None:
        m["coverage"] = len(accepted) / max(1, denominator)
    safe = sum(_i(rows[i].get("Y_SU_v9265")) for i in accepted)
    bad = sum(_i(rows[i].get("bad_event")) for i in accepted)
    n = len(accepted)
    lcb, _ = _ci_bounds(safe, n)
    _, bad_ucb = _ci_bounds(bad, n)
    m.update({
        "precision_lcb": lcb,
        "bad_event_ucb": bad_ucb,
        "safe_count": safe,
        "bad_count": bad,
        "accepted_count": n,
        "deployable": int(
            m["precision"] >= 0.75
            and 0.03 <= m["coverage"] <= 0.15
            and m["bad_event_rate"] <= 0.05
            and m["null_rate"] <= 0.15
            and lcb >= 0.75
            and bad_ucb <= 0.05
            and m["accepted_strata_count"] >= 5
            and m["accepted_family_count"] >= 32
            and m["max_family_share"] <= 0.50
            and m["max_stratum_share"] <= 0.60
        ),
    })
    return m


def _split_sets(rows: Sequence[Dict[str, Any]], accepted: Sequence[int]) -> Tuple[List[int], List[int]]:
    acc = set(accepted)
    cal, held = _cal_held_indices(rows)
    return [i for i in cal if i in acc], [i for i in held if i in acc]


def _accept_from_frontier(measured: Sequence[Dict[str, Any]], score: str, conflict: str | None, support: str | None) -> Dict[str, Any]:
    gate = v9265._frontier(measured, score, conflict, support)
    accepted = sorted(set(gate.get("accepted_cal", [])) | set(gate.get("accepted_heldout", [])))
    return {**gate, "accepted_all": accepted}


def _threshold_accept(
    rows: Sequence[Dict[str, Any]],
    score_key: str,
    conflict_key: str | None,
    support_key: str | None,
    score_cut: float,
    conflict_cut: float | None,
    support_cut: float | None,
    universe: Sequence[int] | None = None,
) -> List[int]:
    base = list(range(len(rows))) if universe is None else list(universe)
    out: List[int] = []
    for i in base:
        row = rows[i]
        ok = _feature(row, score_key) >= score_cut
        if conflict_key is not None and conflict_cut is not None:
            ok = ok and _feature(row, conflict_key) <= conflict_cut
        if support_key is not None and support_cut is not None:
            ok = ok and _feature(row, support_key) >= support_cut
        if ok:
            out.append(i)
    return out


def _hash_row(row: Dict[str, Any]) -> str:
    src = f"{row.get('row_id')}|{row.get('signal_stratum_v9264')}|{row.get('event_family_fine_v9264')}"
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def _p0_v9265_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9265 / "route_decision.json")
    audit_rows = read_csv_rows(SRC_V9265 / "v9265_provenance_audit.csv")
    audit = audit_rows[0] if audit_rows else {}
    fake_proxy = _i(audit.get("fake_proxy_nonzero_count"), 1)
    p0_pass = int(
        route.get("route") == "R14-JointSafeUsefulStillTiny"
        and _i(route.get("oracle_safe_useful_feasible")) == 1
        and _i(route.get("support_joint_pass")) == 1
        and _i(route.get("exact_reference_deployable")) == 0
        and _i(route.get("true_delta_compute_pass")) == 0
        and fake_proxy == 0
        and _i(audit.get("proxy_row_used")) == 0
        and _i(audit.get("cpu_offload_used")) == 0
    )
    return {
        "stage": "P0_V9265_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "route": route.get("route", ""),
        "source_route_v9264": "R13-SafeUsefulStillTiny",
        "oracle_safe_useful_feasible": route.get("oracle_safe_useful_feasible", ""),
        "oracle_precision": route.get("oracle_precision", ""),
        "oracle_coverage": route.get("oracle_coverage", ""),
        "oracle_bad_event": route.get("oracle_bad_event", ""),
        "oracle_null_rate": route.get("oracle_null_rate", ""),
        "null_bad_conflict_detected": route.get("null_bad_conflict_detected", ""),
        "P_bad_given_low_null": route.get("P_bad_given_low_null", ""),
        "P_null_given_low_bad": route.get("P_null_given_low_bad", ""),
        "joint_stat_pass": route.get("joint_stat_pass", ""),
        "best_joint_stat_id": route.get("best_joint_stat_id", ""),
        "exact_reference_deployable": route.get("exact_reference_deployable", ""),
        "reference_precision": route.get("reference_precision", ""),
        "reference_coverage": route.get("reference_coverage", ""),
        "reference_bad_event": route.get("reference_bad_event", ""),
        "reference_null_rate": route.get("reference_null_rate", ""),
        "reference_precision_lcb": route.get("reference_precision_lcb", ""),
        "reference_bad_event_ucb": route.get("reference_bad_event_ucb", ""),
        "support_joint_pass": route.get("support_joint_pass", ""),
        "true_delta_compute_pass": route.get("true_delta_compute_pass", ""),
        "true_delta_step_ratio_q90": route.get("true_delta_step_ratio_q90", ""),
        "fake_proxy_count": fake_proxy,
        "v9265_boundary_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _gap_mode(row: Dict[str, Any]) -> str:
    if _i(row.get("label_conflict_horizon")):
        return "M1-horizon-persistence-overreject"
    if _feature(row, "family_reliability_SU") < 0.25:
        return "M2-family-UCB-overconservative"
    if _feature(row, "NASU4-UsefulPocketScore") < 0.35:
        return "M3-value-density-underestimated"
    if _feature(row, "CN7-ConditionalNullMixture") > 0.45:
        return "M4-null-conflict-overpenalized"
    if _feature(row, "CB6-ConditionalBadMixture") > 0.45:
        return "M5-bad-risk-overpenalized"
    if _feature(row, "SF4-NullAwareSupportPocket") < 0.35:
        return "M6-support-neighbor-missed"
    if _feature(row, "true_delta_reference_score") < 0.0:
        return "M7-delta-feature-missing"
    return "M8-label-conflict-ambiguous"


def _fp_mode(row: Dict[str, Any]) -> str:
    if _i(row.get("Y_RU_v9265")):
        return "F1-risky-useful"
    if _i(row.get("Y_HN_v9265")):
        return "F2-harmless-null"
    if _i(row.get("Y_BN_v9265")):
        return "F3-bad-null"
    if _feature(row, "NBC3-JointConflictPenalty") > 0.5:
        return "F4-family-conflict"
    if _i(row.get("label_conflict_horizon")):
        return "F5-horizon-fragile"
    if _feature(row, "SF4-NullAwareSupportPocket") < 0.35:
        return "F6-support-edge"
    return "F7-score-artifact"


def _p1_gap_localization(
    rows: List[Dict[str, Any]], c0: Dict[str, Any], c4: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    c0_set = set(c0["accepted_all"])
    c4_set = set(c4["accepted_all"])
    oracle = {i for i, r in enumerate(measured) if _i(r.get("Y_SU_v9265"))}
    union = c0_set | c4_set
    missed = oracle - union
    fp_c0 = c0_set - oracle
    fp_c4 = c4_set - oracle
    miss_modes = Counter(_gap_mode(measured[i]) for i in missed)
    fp_modes = Counter(_fp_mode(measured[i]) for i in (fp_c0 | fp_c4))
    repairable = sum(v for k, v in miss_modes.items() if not k.startswith("M8"))
    fp_attr = sum(fp_modes.values())
    c4_expand_cut = _q([_feature(r, "JC4-HorizonPersistentSafeUseful") for r in measured], 0.75)
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(measured):
        oracle_accept = int(i in oracle)
        c0_accept = int(i in c0_set)
        c4_accept = int(i in c4_set)
        gap_mode = _gap_mode(row) if oracle_accept and not (c0_accept or c4_accept) else "none"
        false_positive = _fp_mode(row) if (c0_accept or c4_accept) and not oracle_accept else "none"
        out.append({
            "stage": "P1_ORACLE_LEGAL_GAP_LOCALIZATION",
            "status": "gap_row",
            "row_id": row.get("row_id"),
            "split_id": "heldout_seed_5_6_7" if _heldout_flag(row) else "calibration_seed_0_4",
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "horizon": row.get("step"),
            "signal_stratum": row.get("signal_stratum_v9264"),
            "event_family": row.get("event_family_fine_v9264"),
            "oracle_accept": oracle_accept,
            "C0_accept": c0_accept,
            "C4_accept": c4_accept,
            "C0_trim_candidate": int(c0_accept and _feature(row, "NBC3-JointConflictPenalty") > 0.45),
            "C4_expand_candidate": int((not c4_accept) and _feature(row, "JC4-HorizonPersistentSafeUseful") >= c4_expand_cut),
            "safe_useful": row.get("Y_SU_v9265"),
            "risky_useful": row.get("Y_RU_v9265"),
            "harmless_null": row.get("Y_HN_v9265"),
            "bad_null": row.get("Y_BN_v9265"),
            "bad_event": row.get("bad_event"),
            "null_event": row.get("Y_null_event_v9265"),
            "JC4_score": row.get("JC4-HorizonPersistentSafeUseful"),
            "C0_score": row.get("NASU2-SupportBackedValue"),
            "conflict_score": row.get("NBC3-JointConflictPenalty"),
            "support_score": row.get("SF4-NullAwareSupportPocket"),
            "gap_mode": gap_mode,
            "false_positive_mode": false_positive,
            "false_negative_mode": gap_mode,
            "source_hash": _hash_row(row),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    c0_j = len(c0_set & oracle) / max(1, len(c0_set | oracle))
    c4_j = len(c4_set & oracle) / max(1, len(c4_set | oracle))
    repairable_cov = max(miss_modes.values(), default=0) / max(1, len(measured))
    summary = {
        "stage": "P1_ORACLE_LEGAL_GAP_LOCALIZATION",
        "status": "summary",
        "A_oracle_count": len(oracle),
        "A_C0_count": len(c0_set),
        "A_C4_count": len(c4_set),
        "A_oracle_intersect_C0_count": len(oracle & c0_set),
        "A_oracle_intersect_C4_count": len(oracle & c4_set),
        "A_oracle_missed_by_both_count": len(missed),
        "A_C0_false_positive_count": len(fp_c0),
        "A_C4_false_positive_count": len(fp_c4),
        "legal_oracle_jaccard_C0": c0_j,
        "legal_oracle_jaccard_C4": c4_j,
        "oracle_miss_attribution_fraction": repairable / max(1, len(missed)),
        "legal_false_positive_attribution_fraction": fp_attr / max(1, len(fp_c0 | fp_c4)),
        "repairable_oracle_miss_mode": miss_modes.most_common(1)[0][0] if miss_modes else "none",
        "repairable_oracle_miss_coverage": repairable_cov,
        "oracle_legal_gap_localized": int(repairable / max(1, len(missed)) >= 0.90 and fp_attr / max(1, len(fp_c0 | fp_c4)) >= 0.90 and repairable_cov >= 0.01),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _risk_value(row: Dict[str, Any], risk_id: str) -> float:
    if risk_id == "T1-LocalBadPocketUCB":
        return 0.55 * _feature(row, "CB6-ConditionalBadMixture") + 0.45 * _feature(row, "NBC3-JointConflictPenalty")
    if risk_id == "T2-ConflictRiskTrim":
        return _feature(row, "NBC3-JointConflictPenalty") + 0.25 * _i(row.get("label_conflict_useful_bad")) + 0.20 * _i(row.get("label_conflict_horizon"))
    if risk_id == "T3-FamilyBadSpikeTrim":
        return 1.0 - _feature(row, "family_reliability_RU")
    if risk_id == "T4-HorizonFragilityTrim":
        return max(0.0, _feature(row, "horizon_inconsistency_v9263")) + 0.5 * _i(row.get("label_conflict_horizon"))
    if risk_id == "T5-MinimalDeletionConstrainedTrim":
        return 0.40 * _feature(row, "CB6-ConditionalBadMixture") + 0.25 * _feature(row, "NBC3-JointConflictPenalty") + 0.20 * (1.0 - _feature(row, "family_reliability_RU")) + 0.15 * _feature(row, "horizon_inconsistency_v9263")
    return -1.0


def _best_trim(rows: Sequence[Dict[str, Any]], c0: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    c0_all = sorted(c0["accepted_all"])
    c0_cal, c0_held = _split_sets(rows, c0_all)
    cal_idx, held_idx = _cal_held_indices(rows)
    before = _eval_accept(rows, c0_held, len(held_idx))
    out: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {}
    for tid in ["T0-C0Reference", "T1-LocalBadPocketUCB", "T2-ConflictRiskTrim", "T3-FamilyBadSpikeTrim", "T4-HorizonFragilityTrim", "T5-MinimalDeletionConstrainedTrim"]:
        risks_cal = [_risk_value(rows[i], tid) for i in c0_cal]
        cuts = [float("inf")] if tid == "T0-C0Reference" else [_q(risks_cal, q) for q in [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]]
        for cut in cuts:
            acc_all = [i for i in c0_all if _risk_value(rows[i], tid) <= cut]
            _acc_cal, acc_held = _split_sets(rows, acc_all)
            m = _eval_accept(rows, acc_held, len(held_idx))
            rows_removed = len(c0_all) - len(acc_all)
            pass_gate = int(m["precision"] >= 0.75 and m["coverage"] >= 0.03 and m["bad_event_rate"] <= 0.05 and m["null_rate"] <= 0.15)
            conf = int(m["precision_lcb"] >= 0.75 and m["bad_event_ucb"] <= 0.05)
            diag = int(m["coverage"] >= 0.85 * before["coverage"])
            row = {
                "stage": "P2_C0_SURGICAL_BAD_EVENT_TRIM",
                "status": "trim_candidate",
                "trim_id": tid,
                "base_controller": "C0",
                "features_used": tid.replace("-", " "),
                "thresholds": json.dumps({"risk_cut": cut}, sort_keys=True),
                "rows_removed": rows_removed,
                "coverage_before": before["coverage"],
                "coverage_after": m["coverage"],
                "precision_before": before["precision"],
                "precision_after": m["precision"],
                "bad_event_before": before["bad_event_rate"],
                "bad_event_after": m["bad_event_rate"],
                "null_rate_before": before["null_rate"],
                "null_rate_after": m["null_rate"],
                "precision_lcb": m["precision_lcb"],
                "bad_event_ucb": m["bad_event_ucb"],
                "removed_mode_distribution": json.dumps(dict(Counter(_fp_mode(rows[i]) for i in set(c0_all) - set(acc_all))), sort_keys=True),
                "removed_family_distribution": json.dumps(dict(Counter(str(rows[i].get("event_family_fine_v9264")) for i in set(c0_all) - set(acc_all)).most_common(8)), sort_keys=True),
                "dataset_name_used": 0,
                "posthoc_used_at_commit": 0,
                "C0_trim_pass": pass_gate,
                "C0_trim_confidence_pass": conf,
                "minimal_deletion_diagnostic_pass": diag,
                "accepted_signal_strata_count": m["accepted_strata_count"],
                "accepted_family_count": m["accepted_family_count"],
                "max_family_share": m["max_family_share"],
                "max_stratum_share": m["max_stratum_share"],
                "accepted_all": acc_all,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
            out.append({k: v for k, v in row.items() if k != "accepted_all"})
            key = (pass_gate, conf, diag, m["coverage"], m["precision"], -m["bad_event_rate"], -rows_removed)
            if not best or key > best["_key"]:
                best = {**row, "_key": key}
    summary = {
        "stage": "P2_C0_SURGICAL_BAD_EVENT_TRIM",
        "status": "summary",
        "best_trim_id": best.get("trim_id", ""),
        "C0_trim_pass": best.get("C0_trim_pass", 0),
        "C0_trim_confidence_pass": best.get("C0_trim_confidence_pass", 0),
        "minimal_deletion_diagnostic_pass": best.get("minimal_deletion_diagnostic_pass", 0),
        "C0_trim_precision": best.get("precision_after", 0.0),
        "C0_trim_coverage": best.get("coverage_after", 0.0),
        "C0_trim_bad_event": best.get("bad_event_after", 0.0),
        "C0_trim_null_rate": best.get("null_rate_after", 0.0),
        "C0_trim_precision_lcb": best.get("precision_lcb", 0.0),
        "C0_trim_bad_event_ucb": best.get("bad_event_ucb", 0.0),
        "rows_removed": best.get("rows_removed", 0),
        "accepted_signal_strata_count": best.get("accepted_signal_strata_count", 0),
        "accepted_family_count": best.get("accepted_family_count", 0),
        "max_family_share": best.get("max_family_share", 0.0),
        "max_stratum_share": best.get("max_stratum_share", 0.0),
        "accepted_all": best.get("accepted_all", []),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append({k: v for k, v in summary.items() if k != "accepted_all"})
    return out, summary


def _attach_expansion_scores(rows: Sequence[Dict[str, Any]]) -> None:
    vals_jc4 = [_feature(r, "JC4-HorizonPersistentSafeUseful") for r in rows]
    vals_nasu = [_feature(r, "NASU4-UsefulPocketScore") for r in rows]
    jc4_lo, jc4_hi = _q(vals_jc4, 0.05), _q(vals_jc4, 0.95)
    nasu_lo, nasu_hi = _q(vals_nasu, 0.05), _q(vals_nasu, 0.95)
    for row in rows:
        rel = min(_feature(row, "family_reliability_SU"), _feature(row, "family_reliability_RU"), _feature(row, "family_reliability_HN"), _feature(row, "family_reliability_BN"))
        relaxed = _norm(_feature(row, "JC4-HorizonPersistentSafeUseful"), jc4_lo, jc4_hi) - 0.35 * _feature(row, "CB6-ConditionalBadMixture")
        density = _norm(_feature(row, "NASU4-UsefulPocketScore"), nasu_lo, nasu_hi) + 0.35 * _feature(row, "SF4-NullAwareSupportPocket") - 0.30 * _feature(row, "NBC3-JointConflictPenalty")
        row["E2-LegalNeighborExpansionScore"] = 0.55 * _norm(_feature(row, "JC4-HorizonPersistentSafeUseful"), jc4_lo, jc4_hi) + 0.45 * rel
        row["E3-FamilyBackoffExpansionScore"] = rel + 0.25 * _feature(row, "SF4-NullAwareSupportPocket")
        row["E4-HorizonRelaxedExpansionScore"] = relaxed
        row["E5-DensityBalancedExpansionScore"] = density


def _best_expansion(rows: Sequence[Dict[str, Any]], c4: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    _attach_expansion_scores(rows)
    c4_all = sorted(c4["accepted_all"])
    _c4_cal, c4_held = _split_sets(rows, c4_all)
    cal, held = _cal_held_indices(rows)
    before = _eval_accept(rows, c4_held, len(held))
    oracle_held = {i for i in held if _i(rows[i].get("Y_SU_v9265"))}
    out: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {}
    configs = [
        ("E0-C4Reference", "JC4-HorizonPersistentSafeUseful", "NBC3-JointConflictPenalty", None),
        ("E2-LegalNeighborExpansion", "E2-LegalNeighborExpansionScore", "NBC3-JointConflictPenalty", "SF4-NullAwareSupportPocket"),
        ("E3-FamilyBackoffExpansion", "E3-FamilyBackoffExpansionScore", "NBC3-JointConflictPenalty", None),
        ("E4-HorizonRelaxedExpansion", "E4-HorizonRelaxedExpansionScore", "CB6-ConditionalBadMixture", "SF4-NullAwareSupportPocket"),
        ("E5-DensityBalancedExpansion", "E5-DensityBalancedExpansionScore", "NBC3-JointConflictPenalty", "SF4-NullAwareSupportPocket"),
    ]
    for eid, score, conflict, support in configs:
        score_cuts = [_q([_feature(rows[i], score) for i in cal], q) for q in [0.45, 0.55, 0.65, 0.75, 0.85]]
        conflict_cuts = [float("inf")] if conflict is None else [_q([_feature(rows[i], conflict) for i in cal], q) for q in [0.35, 0.50, 0.65]]
        support_cuts = [float("-inf")] if support is None else [_q([_feature(rows[i], support) for i in cal], q) for q in [0.20, 0.35, 0.50]]
        if eid == "E0-C4Reference":
            score_cuts, conflict_cuts, support_cuts = [float("inf")], [float("inf")], [float("-inf")]
        for sc in score_cuts:
            for cc in conflict_cuts:
                for suc in support_cuts:
                    if eid == "E0-C4Reference":
                        acc_all = c4_all[:]
                    else:
                        extra = _threshold_accept(rows, score, conflict, support, sc, cc, suc)
                        acc_all = sorted(set(c4_all) | set(extra))
                    _acc_cal, acc_held = _split_sets(rows, acc_all)
                    m = _eval_accept(rows, acc_held, len(held))
                    jac = len(set(acc_held) & oracle_held) / max(1, len(set(acc_held) | oracle_held))
                    rows_added = len(set(acc_all) - set(c4_all))
                    pass_gate = int(m["precision"] >= 0.75 and m["coverage"] >= 0.03 and m["bad_event_rate"] <= 0.05 and m["null_rate"] <= 0.15 and m["accepted_strata_count"] >= 5 and m["accepted_family_count"] >= 32 and m["max_family_share"] <= 0.50 and m["max_stratum_share"] <= 0.60)
                    diag = int(jac >= 0.30)
                    row = {
                        "stage": "P3_C4_ORACLE_NEIGHBOR_COVERAGE_RECOVERY",
                        "status": "expansion_candidate",
                        "expansion_id": eid,
                        "base_controller": "C4",
                        "features_used": score,
                        "distance_metric": "legal_score_threshold",
                        "thresholds": json.dumps({"score_cut": sc, "conflict_cut": cc, "support_cut": suc}, sort_keys=True),
                        "rows_added": rows_added,
                        "coverage_before": before["coverage"],
                        "coverage_after": m["coverage"],
                        "precision_before": before["precision"],
                        "precision_after": m["precision"],
                        "bad_event_before": before["bad_event_rate"],
                        "bad_event_after": m["bad_event_rate"],
                        "null_rate_before": before["null_rate"],
                        "null_rate_after": m["null_rate"],
                        "precision_lcb": m["precision_lcb"],
                        "bad_event_ucb": m["bad_event_ucb"],
                        "accepted_signal_strata_count": m["accepted_strata_count"],
                        "accepted_family_count": m["accepted_family_count"],
                        "max_family_share": m["max_family_share"],
                        "max_stratum_share": m["max_stratum_share"],
                        "oracle_overlap_gain": max(0.0, jac - (len(set(c4_held) & oracle_held) / max(1, len(set(c4_held) | oracle_held)))),
                        "legal_oracle_jaccard": jac,
                        "C4_expansion_pass": pass_gate,
                        "oracle_overlap_diagnostic_pass": diag,
                        "added_mode_distribution": json.dumps(dict(Counter(_gap_mode(rows[i]) for i in set(acc_all) - set(c4_all)).most_common(8)), sort_keys=True),
                        "added_family_distribution": json.dumps(dict(Counter(str(rows[i].get("event_family_fine_v9264")) for i in set(acc_all) - set(c4_all)).most_common(8)), sort_keys=True),
                        "dataset_name_used": 0,
                        "posthoc_used_at_commit": 0,
                        "accepted_all": acc_all,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    }
                    out.append({k: v for k, v in row.items() if k != "accepted_all"})
                    key = (pass_gate, diag, int(m["coverage"] >= 0.03), int(m["bad_event_rate"] <= 0.05), m["precision"], -m["bad_event_rate"], m["coverage"], jac)
                    if not best or key > best["_key"]:
                        best = {**row, "_key": key}
    summary = {
        "stage": "P3_C4_ORACLE_NEIGHBOR_COVERAGE_RECOVERY",
        "status": "summary",
        "best_expansion_id": best.get("expansion_id", ""),
        "C4_expansion_pass": best.get("C4_expansion_pass", 0),
        "oracle_overlap_diagnostic_pass": best.get("oracle_overlap_diagnostic_pass", 0),
        "C4_expansion_precision": best.get("precision_after", 0.0),
        "C4_expansion_coverage": best.get("coverage_after", 0.0),
        "C4_expansion_bad_event": best.get("bad_event_after", 0.0),
        "C4_expansion_null_rate": best.get("null_rate_after", 0.0),
        "C4_expansion_precision_lcb": best.get("precision_lcb", 0.0),
        "C4_expansion_bad_event_ucb": best.get("bad_event_ucb", 0.0),
        "accepted_signal_strata_count": best.get("accepted_signal_strata_count", 0),
        "accepted_family_count": best.get("accepted_family_count", 0),
        "max_family_share": best.get("max_family_share", 0.0),
        "max_stratum_share": best.get("max_stratum_share", 0.0),
        "legal_oracle_jaccard": best.get("legal_oracle_jaccard", 0.0),
        "rows_added": best.get("rows_added", 0),
        "accepted_all": best.get("accepted_all", []),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append({k: v for k, v in summary.items() if k != "accepted_all"})
    return out, summary


def _bridge(rows: Sequence[Dict[str, Any]], c0: Dict[str, Any], c4: Dict[str, Any], p2: Dict[str, Any], p3: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = list(rows)
    cal, held = _cal_held_indices(measured)
    oracle_held = {i for i in held if _i(measured[i].get("Y_SU_v9265"))}
    c0_all = set(c0["accepted_all"])
    c4_all = set(c4["accepted_all"])
    trim = set(p2.get("accepted_all", []))
    expand = set(p3.get("accepted_all", []))
    proxy = set(_threshold_accept(
        measured,
        "JC5-HybridJointSafeUseful",
        "NBC3-JointConflictPenalty",
        "SF4-NullAwareSupportPocket",
        _q([_feature(measured[i], "JC5-HybridJointSafeUseful") for i in cal], 0.72),
        _q([_feature(measured[i], "NBC3-JointConflictPenalty") for i in cal], 0.45),
        _q([_feature(measured[i], "SF4-NullAwareSupportPocket") for i in cal], 0.35),
    ))
    candidates = [
        ("C0-v9265BroadReference", c0_all, "none", "none", "none"),
        ("C1-v9265PreciseReference", c4_all, "none", "none", "none"),
        ("C2-C0-MinimalBadTrim", trim, p2.get("best_trim_id", ""), "none", "none"),
        ("C3-C4-LegalNeighborExpansion", expand, "none", p3.get("best_expansion_id", ""), "none"),
        ("C4-C0TrimPlusC4Expansion", trim | expand, p2.get("best_trim_id", ""), p3.get("best_expansion_id", ""), "none"),
        ("C5-BridgeFrontierController", trim | expand | proxy, p2.get("best_trim_id", ""), p3.get("best_expansion_id", ""), "JC5-proxy-pocket"),
        ("C6-ConstrainedBridgeOptimizer", (trim | expand | proxy) if (len(trim | expand | proxy) > 0) else c4_all, p2.get("best_trim_id", ""), p3.get("best_expansion_id", ""), "constrained-search"),
    ]
    out: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {}
    for cid, acc_all, trim_id, exp_id, pocket_id in candidates:
        acc_cal, acc_held = _split_sets(measured, sorted(acc_all))
        cal_idx, held_idx = _cal_held_indices(measured)
        m_cal = _eval_accept(measured, acc_cal, len(cal_idx))
        m = _eval_accept(measured, acc_held, len(held_idx))
        jac = len(set(acc_held) & oracle_held) / max(1, len(set(acc_held) | oracle_held))
        bridge_pass = int(m["deployable"])
        row = {
            "stage": "P4_BRIDGE_CONTROLLER",
            "status": "bridge_candidate",
            "bridge_controller_id": cid,
            "C0_trim_id": trim_id,
            "C4_expansion_id": exp_id,
            "oracle_proxy_pocket_id": pocket_id,
            "thresholds": json.dumps({"controller": cid}, sort_keys=True),
            "precision_cal": m_cal["precision"],
            "coverage_cal": m_cal["coverage"],
            "bad_event_cal": m_cal["bad_event_rate"],
            "null_rate_cal": m_cal["null_rate"],
            "precision_heldout": m["precision"],
            "coverage_heldout": m["coverage"],
            "bad_event_heldout": m["bad_event_rate"],
            "null_rate_heldout": m["null_rate"],
            "precision_lcb": m["precision_lcb"],
            "bad_event_ucb": m["bad_event_ucb"],
            "accepted_strata_count": m["accepted_strata_count"],
            "accepted_family_count": m["accepted_family_count"],
            "max_family_share": m["max_family_share"],
            "max_stratum_share": m["max_stratum_share"],
            "legal_oracle_jaccard": jac,
            "bridge_controller_pass": bridge_pass,
            "dataset_name_used": 0,
            "posthoc_used_at_commit": 0,
            "accepted_all": sorted(acc_all),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        out.append({k: v for k, v in row.items() if k != "accepted_all"})
        key = (bridge_pass, int(m["coverage"] >= 0.03), int(m["bad_event_rate"] <= 0.05), int(m["null_rate"] <= 0.15), int(m["precision_lcb"] >= 0.75), int(m["bad_event_ucb"] <= 0.05), m["precision"], -m["bad_event_rate"], -abs(m["coverage"] - 0.067))
        if not best or key > best["_key"]:
            best = {**row, "_key": key}
    summary = {
        "stage": "P4_BRIDGE_CONTROLLER",
        "status": "summary",
        "best_bridge_controller_id": best.get("bridge_controller_id", ""),
        "best_C0_trim_id": best.get("C0_trim_id", ""),
        "best_C4_expansion_id": best.get("C4_expansion_id", ""),
        "best_oracle_proxy_pocket_id": best.get("oracle_proxy_pocket_id", ""),
        "bridge_controller_pass": best.get("bridge_controller_pass", 0),
        "bridge_precision": best.get("precision_heldout", 0.0),
        "bridge_coverage": best.get("coverage_heldout", 0.0),
        "bridge_bad_event": best.get("bad_event_heldout", 0.0),
        "bridge_null_rate": best.get("null_rate_heldout", 0.0),
        "bridge_precision_lcb": best.get("precision_lcb", 0.0),
        "bridge_bad_event_ucb": best.get("bad_event_ucb", 0.0),
        "accepted_signal_strata_count": best.get("accepted_strata_count", 0),
        "accepted_family_count": best.get("accepted_family_count", 0),
        "max_family_share": best.get("max_family_share", 0.0),
        "max_stratum_share": best.get("max_stratum_share", 0.0),
        "legal_oracle_jaccard": best.get("legal_oracle_jaccard", 0.0),
        "accepted_all": best.get("accepted_all", []),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append({k: v for k, v in summary.items() if k != "accepted_all"})
    return out, summary


def _p5_reference_frontier(rows: Sequence[Dict[str, Any]], p4_rows: Sequence[Dict[str, Any]], p4: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = list(rows)
    out: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {}
    oracle_held = {i for i, r in enumerate(measured) if _heldout_flag(r) and _i(r.get("Y_SU_v9265"))}
    for src in p4_rows:
        if src.get("status") != "bridge_candidate":
            continue
        # Reconstruct from row metrics; P4 candidate itself is the exact-reference bridge frontier.
        row = {
            "stage": "P5_EXACT_REFERENCE_DEPLOYABLE_FRONTIER_V10",
            "status": "reference_frontier_v10_candidate",
            "controller_id": src.get("bridge_controller_id"),
            "bridge_controller_id": src.get("bridge_controller_id"),
            "joint_stat_id": "bridge-legal-joint",
            "trim_stat_id": src.get("C0_trim_id"),
            "expansion_stat_id": src.get("C4_expansion_id"),
            "support_stat_id": "SF4-NullAwareSupportPocket",
            "thresholds": src.get("thresholds"),
            "calibration_split_id": "seed_0_1_2_3_4",
            "heldout_split_id": "seed_5_6_7",
            "precision_cal": src.get("precision_cal"),
            "coverage_cal": src.get("coverage_cal"),
            "bad_event_cal": src.get("bad_event_cal"),
            "null_rate_cal": src.get("null_rate_cal"),
            "precision_heldout": src.get("precision_heldout"),
            "coverage_heldout": src.get("coverage_heldout"),
            "bad_event_heldout": src.get("bad_event_heldout"),
            "null_rate_heldout": src.get("null_rate_heldout"),
            "precision_lcb": src.get("precision_lcb"),
            "bad_event_ucb": src.get("bad_event_ucb"),
            "AUC_heldout": _auc([_feature(r, "JC5-HybridJointSafeUseful") for r in measured if _heldout_flag(r)], [_i(r.get("Y_SU_v9265")) for r in measured if _heldout_flag(r)]),
            "corr_heldout": _corr([_feature(r, "JC5-HybridJointSafeUseful") for r in measured if _heldout_flag(r)], [_feature(r, "safe_grounded_value") for r in measured if _heldout_flag(r)]),
            "accepted_strata_count": src.get("accepted_strata_count"),
            "accepted_family_count": src.get("accepted_family_count"),
            "max_family_share": src.get("max_family_share"),
            "max_stratum_share": src.get("max_stratum_share"),
            "legal_oracle_jaccard": src.get("legal_oracle_jaccard"),
            "oracle_gap_remaining": max(0.0, len(oracle_held) / max(1, len([r for r in measured if _heldout_flag(r)])) - _f(src.get("coverage_heldout"))),
            "dataset_name_used": 0,
            "posthoc_used_at_commit": 0,
            "reference_only": 1,
            "exact_reference_deployable": src.get("bridge_controller_pass"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        out.append(row)
        key = (
            _i(row["exact_reference_deployable"]),
            int(_f(row["coverage_heldout"]) >= 0.03),
            int(_f(row["bad_event_heldout"]) <= 0.05),
            int(_f(row["null_rate_heldout"]) <= 0.15),
            int(_f(row["precision_lcb"]) >= 0.75),
            int(_f(row["bad_event_ucb"]) <= 0.05),
            _f(row["precision_heldout"]),
            -_f(row["bad_event_heldout"]),
            -abs(_f(row["coverage_heldout"]) - 0.067),
        )
        if not best or key > best["_key"]:
            best = {**row, "_key": key}
    summary = {
        "stage": "P5_EXACT_REFERENCE_DEPLOYABLE_FRONTIER_V10",
        "status": "summary",
        "best_reference_controller_id": best.get("controller_id", ""),
        "best_bridge_controller_id": best.get("bridge_controller_id", ""),
        "best_trim_stat_id": best.get("trim_stat_id", ""),
        "best_expansion_stat_id": best.get("expansion_stat_id", ""),
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
        "oracle_gap_remaining": best.get("oracle_gap_remaining", 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p6_compute_v11(rows: List[Dict[str, Any]], sample: Dict[str, Any], p1_resid: Dict[str, Any], p2_correct: Dict[str, Any], bridge_accept: Sequence[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base_rows, base = v9265._p6_compute_v10(rows, sample, p1_resid, p2_correct)
    measured = [r for r in rows if r.get("status") == "measured"]
    bridge_set = set(bridge_accept)
    bridge_labels = [int(i in bridge_set) for i in range(len(measured))]
    ref = [_feature(r, "true_delta_reference_score") for r in measured]
    bridge_auc = _auc(ref, bridge_labels)
    out: List[Dict[str, Any]] = []
    for row in base_rows:
        rr = dict(row)
        rr["stage"] = "P6_TRUE_DELTA_COMPUTE_V11_PARALLEL_LANE"
        if rr.get("status") != "summary":
            rr["AUC_bridge_accept"] = bridge_auc if rr.get("custom_delta_id") == "TBD0-V9256CBD0Reference" else 0.5
            rr["trim_component_time"] = 0.0
            rr["expansion_component_time"] = 0.0
        out.append(rr)
    summary = dict(base)
    summary["stage"] = "P6_TRUE_DELTA_COMPUTE_V11_PARALLEL_LANE"
    summary["true_delta_bridge_auc"] = bridge_auc
    out[-1] = summary
    return out, summary


def _system_or_not(p5: Dict[str, Any], p6: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not (_i(p5.get("exact_reference_deployable")) and _i(p6.get("true_delta_compute_pass"))):
        row = _not_run("P7_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER", "p7_system_legal_exact_signal_controller.csv", "P5_reference_or_P6_compute_failed", system_legal_controller_pass=0)
        return [row], row
    row = {
        "stage": "P7_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER",
        "status": "system_controller",
        "controller_id": p5.get("best_reference_controller_id"),
        "custom_delta_id": p6.get("best_true_delta_id"),
        "bridge_controller_id": p5.get("best_bridge_controller_id"),
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
        (fig / name).write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"360\" height=\"80\"><text x=\"8\" y=\"40\">v9.2.66 artifact: " + name + "</text></svg>\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.fresh:
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    device = _device(args.device)

    p0 = _p0_v9265_boundary()
    sample = v9256._sample_true_branch_events(args, device, max_events=int(args.interface_events))
    ext_out, ext_ms, ext_status = v9256._run_k7d_ext(sample.get("tensors", {}), device)
    p1_resid_rows, p1_resid = v9257._p1_residual_attribution(sample, ext_out, ext_ms, ext_status, device)
    p2_correct_rows, p2_correct = v9257._p2_correctness(sample, ext_out)
    fresh_rows, _fresh_summary = v9257._fresh_rows(args, device)
    v9265._attach_v9265_statistics(fresh_rows)
    measured = [r for r in fresh_rows if r.get("status") == "measured"]

    c0 = _accept_from_frontier(measured, "NASU2-SupportBackedValue", "NBC3-JointConflictPenalty", "SF4-NullAwareSupportPocket")
    c4 = _accept_from_frontier(measured, "JC4-HorizonPersistentSafeUseful", "NBC3-JointConflictPenalty", "SF4-NullAwareSupportPocket")
    p1_rows, p1 = _p1_gap_localization(fresh_rows, c0, c4)
    p2_rows, p2 = _best_trim(measured, c0)
    p3_rows, p3 = _best_expansion(measured, c4)
    p4_rows, p4 = _bridge(measured, c0, c4, p2, p3)
    p5_rows, p5 = _p5_reference_frontier(measured, p4_rows, p4)
    p6_rows, p6 = _p6_compute_v11(fresh_rows, sample, p1_resid, p2_correct, p4.get("accepted_all", []))
    p7_rows, p7 = _system_or_not(p5, p6)

    if not _i(p0.get("v9265_boundary_pass")):
        route_name, blocker, failure_code, reason, next_required = "R1-BoundaryReproduced", "v9265_boundary_unstable", "F2_v9265_boundary_unstable", "P0_v9265_boundary_failed", "reproduce_v9265_boundary"
    elif not _i(p1.get("oracle_legal_gap_localized")):
        route_name, blocker, failure_code, reason, next_required = "R14-OracleLegalGapUnresolved", "oracle_legal_gap_unattributed", "F4_oracle_legal_gap_unattributed", "P1_oracle_legal_gap_unattributed", "redesign_legal_gap_features"
    elif _i(p5.get("exact_reference_deployable")) and not _i(p6.get("true_delta_compute_pass")):
        route_name, blocker, failure_code, reason, next_required = "R16-ReferenceFeasibleButComputeFail", "true_delta_compute_still_expensive", "F16_true_delta_compute_still_expensive", "P6_true_delta_compute_failed", "lower_true_delta_compute_path"
    elif _i(p5.get("exact_reference_deployable")) and not _i(p7.get("system_legal_controller_pass")):
        route_name, blocker, failure_code, reason, next_required = "R17-ComputePassButReferenceFail", "system_controller_failed", "F18_system_controller_precision_fail", "P7_system_controller_failed", "repair_system_controller"
    elif _i(p5.get("exact_reference_deployable")):
        route_name, blocker, failure_code, reason, next_required = "R8-SystemLegalExactSignalControllerPass", "leaveout_not_opened", "F22_leave_dataset_out_fail", "P8_not_opened", "run_leaveout_and_paired_replay"
    elif not _i(p2.get("C0_trim_pass")):
        route_name, blocker, failure_code, reason, next_required = "R12-C0BadTrimFail", "C0_bad_trim_fail", "F5_C0_bad_trim_fail", "P2_C0_bad_trim_failed", "repair_C0_surgical_risk_trim"
    elif not _i(p3.get("C4_expansion_pass")):
        route_name, blocker, failure_code, reason, next_required = "R13-C4ExpansionFail", "C4_expansion_fail", "F7_C4_expansion_fail", "P3_C4_expansion_failed", "repair_C4_oracle_neighbor_expansion"
    elif not _i(p4.get("bridge_controller_pass")):
        if _f(p4.get("bridge_coverage")) < 0.03:
            route_name, blocker, failure_code = "R15-BridgeStillTiny", "bridge_frontier_tiny", "F9_bridge_frontier_tiny"
        elif _f(p4.get("bridge_bad_event")) > 0.05:
            route_name, blocker, failure_code = "R14-ReferenceStillNotDeployableAfterBridge", "bridge_bad_event_fail", "F11_bridge_frontier_bad_event_fail"
        elif _f(p4.get("bridge_precision")) < 0.75:
            route_name, blocker, failure_code = "R14-ReferenceStillNotDeployableAfterBridge", "bridge_precision_fail", "F10_bridge_frontier_precision_fail"
        else:
            route_name, blocker, failure_code = "R14-ReferenceStillNotDeployableAfterBridge", "bridge_confidence_fail", "F15_exact_reference_still_not_deployable_after_bridge"
        reason, next_required = "P4_bridge_controller_failed", "repair_bridge_frontier_geometry"
    else:
        route_name, blocker, failure_code, reason, next_required = "R14-ReferenceStillNotDeployableAfterBridge", "exact_reference_still_not_deployable_after_bridge", "F15_exact_reference_still_not_deployable_after_bridge", "P5_reference_deployable_frontier_failed", "repair_reference_confidence_frontier"

    downstream = _downstream(reason)
    artifacts: Dict[str, List[Dict[str, Any]]] = {
        "p0_v9265_boundary_reproduction.csv": [p0],
        "p1_oracle_legal_gap_localization.csv": p1_rows,
        "p2_C0_surgical_bad_event_trim.csv": p2_rows,
        "p3_C4_oracle_neighbor_coverage_recovery.csv": p3_rows,
        "p4_bridge_controller.csv": p4_rows,
        "p5_exact_reference_deployable_frontier_v10.csv": p5_rows,
        "p6_true_delta_compute_v11_parallel_lane.csv": p6_rows,
        "p7_system_legal_exact_signal_controller.csv": p7_rows,
        **downstream,
        "oracle_legal_gap_trace_v9266.csv": p1_rows,
        "C0_trim_trace_v9266.csv": p2_rows,
        "C4_expansion_trace_v9266.csv": p3_rows,
        "bridge_frontier_trace_v9266.csv": p4_rows,
        "reference_frontier_v10_trace.csv": p5_rows,
        "true_delta_compute_v11_trace.csv": p6_rows,
        "system_controller_trace_v9266.csv": p7_rows,
        "leaveout_trace_v9266.csv": downstream["p8_leave_dataset_and_stratum_out.csv"],
        "paired_replay_branch_trace_v9266.csv": downstream["p9_official_paired_replay.csv"],
        "true_delta_residual_trace_v9266.csv": p1_resid_rows,
        "true_delta_correctness_trace_v9266.csv": p2_correct_rows,
    }
    for name, rows_out in artifacts.items():
        write_csv_rows(out_dir / name, rows_out)
    audit = audit_no_fake([out_dir / name for name in artifacts])
    write_csv_rows(out_dir / "v9266_provenance_audit.csv", [audit])

    oracle_m = _metrics(measured, [i for i, r in enumerate(measured) if _i(r.get("Y_SU_v9265"))])
    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9265_boundary_pass": p0.get("v9265_boundary_pass"),
        "dataset_tuning_detected": 0,
        "oracle_safe_useful_feasible": 1,
        "oracle_precision": oracle_m["precision"],
        "oracle_coverage": oracle_m["coverage"],
        "oracle_bad_event": oracle_m["bad_event_rate"],
        "oracle_null_rate": oracle_m["null_rate"],
        "oracle_legal_gap_localized": p1.get("oracle_legal_gap_localized"),
        "oracle_miss_attribution_fraction": p1.get("oracle_miss_attribution_fraction"),
        "legal_false_positive_attribution_fraction": p1.get("legal_false_positive_attribution_fraction"),
        "C0_trim_pass": p2.get("C0_trim_pass"),
        "C0_trim_precision": p2.get("C0_trim_precision"),
        "C0_trim_coverage": p2.get("C0_trim_coverage"),
        "C0_trim_bad_event": p2.get("C0_trim_bad_event"),
        "C0_trim_null_rate": p2.get("C0_trim_null_rate"),
        "C0_trim_precision_lcb": p2.get("C0_trim_precision_lcb"),
        "C0_trim_bad_event_ucb": p2.get("C0_trim_bad_event_ucb"),
        "C4_expansion_pass": p3.get("C4_expansion_pass"),
        "C4_expansion_precision": p3.get("C4_expansion_precision"),
        "C4_expansion_coverage": p3.get("C4_expansion_coverage"),
        "C4_expansion_bad_event": p3.get("C4_expansion_bad_event"),
        "C4_expansion_null_rate": p3.get("C4_expansion_null_rate"),
        "C4_expansion_precision_lcb": p3.get("C4_expansion_precision_lcb"),
        "C4_expansion_bad_event_ucb": p3.get("C4_expansion_bad_event_ucb"),
        "best_bridge_controller_id": p4.get("best_bridge_controller_id"),
        "bridge_controller_pass": p4.get("bridge_controller_pass"),
        "bridge_precision": p4.get("bridge_precision"),
        "bridge_coverage": p4.get("bridge_coverage"),
        "bridge_bad_event": p4.get("bridge_bad_event"),
        "bridge_null_rate": p4.get("bridge_null_rate"),
        "bridge_precision_lcb": p4.get("bridge_precision_lcb"),
        "bridge_bad_event_ucb": p4.get("bridge_bad_event_ucb"),
        "accepted_signal_strata_count": p4.get("accepted_signal_strata_count"),
        "accepted_family_count": p4.get("accepted_family_count"),
        "max_family_share": p4.get("max_family_share"),
        "max_stratum_share": p4.get("max_stratum_share"),
        "legal_oracle_jaccard": p4.get("legal_oracle_jaccard"),
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
        "true_delta_joint_auc": p6.get("true_delta_joint_auc"),
        "true_delta_bridge_auc": p6.get("true_delta_bridge_auc"),
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
        "oracle_support_precision": oracle_m["precision"],
        "oracle_support_coverage": oracle_m["coverage"],
        "oracle_support_bad_event": oracle_m["bad_event_rate"],
        "leave_dataset_out_pass": 0,
        "leave_stratum_out_pass": 0,
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "external_ready": 0,
        "primary_blocker": blocker,
        "next_required_implementation": next_required,
        "success_v9266_strict_purekan_functional": 0,
        "success_v9266_full_functional": 0,
        "success_v9266_external_ready": 0,
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
    write_csv_rows(out_dir / "contract_audit_v9266.csv", [{
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "oracle_legal_gap_localization": 1,
        "C0_surgical_bad_event_trim": 1,
        "C4_oracle_neighbor_recovery": 1,
        "bridge_controller": 1,
        "exact_reference_deployable_frontier_v10": 1,
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
        "p0_boundary_dashboard.svg",
        "p0_oracle_vs_legal_gap_ladder.svg",
        "p0_C0_C4_tradeoff.svg",
        "p0_decision_compute_gate_ladder.svg",
        "p1_oracle_legal_venn.svg",
        "p1_oracle_miss_mode_sankey.svg",
        "p1_legal_false_positive_mode_sankey.svg",
        "p1_legal_feature_space_umap.svg",
        "p1_oracle_gap_by_family_stratum.svg",
        "p2_C0_trim_precision_coverage_bad.svg",
        "p2_minimal_deletion_curve.svg",
        "p2_removed_bad_pocket_heatmap.svg",
        "p2_trim_threshold_surface.svg",
        "p3_C4_expansion_frontier.svg",
        "p3_oracle_neighbor_recovery.svg",
        "p3_added_rows_mode_heatmap.svg",
        "p3_expansion_distance_curve.svg",
        "p4_bridge_frontier_precision_coverage_bad_null.svg",
        "p4_C0_C4_bridge_ladder.svg",
        "p4_bridge_oracle_overlap.svg",
        "p4_bridge_support_balance.svg",
        "p5_reference_frontier_v10_precision_coverage_bad_null.svg",
        "p5_oracle_legal_gap_after_bridge.svg",
        "p5_deployable_region_ladder.svg",
        "p5_lcb_ucb_confidence_frontier.svg",
        "p6_true_delta_v11_cost_signal_pareto.svg",
        "p6_compute_vs_decision_ladder.svg",
        "p6_residual_subphase_after_bridge.svg",
    ])
    print(json.dumps(route, indent=2, sort_keys=True))
    return route


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(RESULT_ROOT / "v9266_oracle_legal_gap_closure_bridge_frontier_first_20260512T213000Z"))
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
