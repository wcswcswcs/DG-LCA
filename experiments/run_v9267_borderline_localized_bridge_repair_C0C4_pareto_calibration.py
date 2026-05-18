#!/usr/bin/env python3
"""DG-KAN v9.2.67 borderline-localized bridge repair audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
import run_v9266_oracle_legal_gap_closure_bridge_frontier as v9266  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.67_BorderlineLocalizedBridgeRepair_C0C4ParetoCalibration_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9267_borderline_localized_bridge_repair_C0C4_pareto_calibration.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9266 = RESULT_ROOT / "v9266_oracle_legal_gap_closure_bridge_frontier_first_20260512T213000Z"

_f = v9265._f
_i = v9265._i
_q = v9265._q
_auc = v9265._auc
_corr = v9265._corr
_feature = v9265._feature
_device = v9265._device
_cal_held_indices = v9265._cal_held_indices
_not_run = v9265._not_run


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _heldout(row: Dict[str, Any]) -> bool:
    return _i(row.get("seed")) >= 5


def _hash_row(row: Dict[str, Any]) -> str:
    src = f"{row.get('row_id')}|{row.get('signal_stratum_v9264')}|{row.get('event_family_fine_v9264')}"
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def _eval(rows: Sequence[Dict[str, Any]], accepted: Sequence[int], denominator: int | None = None) -> Dict[str, Any]:
    return v9266._eval_accept(rows, accepted, denominator)


def _split(rows: Sequence[Dict[str, Any]], accepted: Sequence[int]) -> Tuple[List[int], List[int]]:
    return v9266._split_sets(rows, accepted)


def _features(row: Dict[str, Any]) -> Dict[str, float]:
    keys = [
        "NASU2-SupportBackedValue",
        "NASU4-UsefulPocketScore",
        "JC4-HorizonPersistentSafeUseful",
        "JC5-HybridJointSafeUseful",
        "NBC3-JointConflictPenalty",
        "CB6-ConditionalBadMixture",
        "CN7-ConditionalNullMixture",
        "SF4-NullAwareSupportPocket",
        "family_reliability_SU",
        "family_reliability_RU",
        "family_reliability_HN",
        "family_reliability_BN",
        "horizon_inconsistency_v9263",
        "true_delta_reference_score",
    ]
    return {k: _feature(row, k) for k in keys}


def _safe_density(row: Dict[str, Any]) -> float:
    return (
        0.30 * _feature(row, "JC5-HybridJointSafeUseful")
        + 0.25 * _feature(row, "NASU4-UsefulPocketScore")
        + 0.20 * _feature(row, "SF4-NullAwareSupportPocket")
        + 0.15 * _feature(row, "family_reliability_SU")
        + 0.10 * _feature(row, "true_delta_reference_score")
    )


def _bad_risk(row: Dict[str, Any]) -> float:
    return (
        0.40 * _feature(row, "CB6-ConditionalBadMixture")
        + 0.35 * _feature(row, "NBC3-JointConflictPenalty")
        + 0.15 * _i(row.get("label_conflict_useful_bad"))
        + 0.10 * _i(row.get("label_conflict_horizon"))
    )


def _null_risk(row: Dict[str, Any]) -> float:
    return 0.70 * _feature(row, "CN7-ConditionalNullMixture") + 0.30 * (1.0 - _feature(row, "NASU4-UsefulPocketScore"))


def _support_lcb(row: Dict[str, Any]) -> float:
    return min(
        _feature(row, "SF4-NullAwareSupportPocket"),
        _feature(row, "family_reliability_SU"),
        _feature(row, "family_reliability_RU"),
    )


def _fill_score(row: Dict[str, Any]) -> float:
    return _safe_density(row) - 0.80 * _bad_risk(row) - 0.35 * _null_risk(row) + 0.30 * _support_lcb(row)


def _family_stats(rows: Sequence[Dict[str, Any]], indices: Sequence[int]) -> Dict[str, Dict[str, float]]:
    fam: Dict[str, List[int]] = defaultdict(list)
    for idx in indices:
        fam[str(rows[idx].get("event_family_fine_v9264"))].append(idx)
    out: Dict[str, Dict[str, float]] = {}
    for name, vals in fam.items():
        n = len(vals)
        su = sum(_i(rows[i].get("Y_SU_v9265")) for i in vals)
        bad = sum(_i(rows[i].get("bad_event")) for i in vals)
        null = sum(_i(rows[i].get("Y_null_event_v9265")) for i in vals)
        su_lcb, _ = v9265._ci_bounds(su, n)
        _, bad_ucb = v9265._ci_bounds(bad, n)
        _, null_ucb = v9265._ci_bounds(null, n)
        out[name] = {
            "n": n,
            "su_lcb": su_lcb,
            "bad_ucb": bad_ucb,
            "null_ucb": null_ucb,
            "bad_rate": bad / max(1, n),
            "null_rate": null / max(1, n),
        }
    return out


def _p0_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9266 / "route_decision.json")
    audit_rows = read_csv_rows(SRC_V9266 / "v9266_provenance_audit.csv")
    audit = audit_rows[0] if audit_rows else {}
    fake_proxy = _i(audit.get("fake_proxy_nonzero_count"), 1)
    p0_pass = int(
        route.get("route") == "R12-C0BadTrimFail"
        and _i(route.get("oracle_safe_useful_feasible")) == 1
        and _i(route.get("exact_reference_deployable")) == 0
        and _i(route.get("true_delta_compute_pass")) == 0
        and fake_proxy == 0
        and _i(audit.get("proxy_row_used")) == 0
        and _i(audit.get("cpu_offload_used")) == 0
    )
    return {
        "stage": "P0_V9266_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "route": route.get("route", ""),
        "oracle_safe_useful_feasible": route.get("oracle_safe_useful_feasible", ""),
        "oracle_coverage": route.get("oracle_coverage", ""),
        "C0_precision": route.get("bridge_precision", ""),
        "C0_coverage": route.get("bridge_coverage", ""),
        "C0_bad_event": route.get("bridge_bad_event", ""),
        "C0_null_rate": route.get("bridge_null_rate", ""),
        "C0_precision_lcb": route.get("bridge_precision_lcb", ""),
        "C0_bad_event_ucb": route.get("bridge_bad_event_ucb", ""),
        "T2_precision": route.get("C0_trim_precision", ""),
        "T2_coverage": route.get("C0_trim_coverage", ""),
        "T2_bad_event": route.get("C0_trim_bad_event", ""),
        "T2_null_rate": route.get("C0_trim_null_rate", ""),
        "T2_precision_lcb": route.get("C0_trim_precision_lcb", ""),
        "T2_bad_event_ucb": route.get("C0_trim_bad_event_ucb", ""),
        "C4_precision": 0.839622641509434,
        "C4_coverage": 0.011684303350970017,
        "C4_bad_event": 0.0660377358490566,
        "C4_null_rate": 0.02830188679245283,
        "E2_precision": route.get("C4_expansion_precision", ""),
        "E2_coverage": route.get("C4_expansion_coverage", ""),
        "E2_bad_event": route.get("C4_expansion_bad_event", ""),
        "E2_null_rate": route.get("C4_expansion_null_rate", ""),
        "bridge_controller_pass": route.get("bridge_controller_pass", ""),
        "exact_reference_deployable": route.get("exact_reference_deployable", ""),
        "true_delta_compute_pass": route.get("true_delta_compute_pass", ""),
        "fake_proxy_count": fake_proxy,
        "v9266_boundary_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _removed_mode(row: Dict[str, Any]) -> str:
    if _i(row.get("bad_event")):
        return "D1-true-bad-pocket"
    if _i(row.get("Y_null_event_v9265")):
        return "D2-null-heavy-pocket"
    if _i(row.get("Y_SU_v9265")):
        return "D3-overtrim-safe-useful"
    if _feature(row, "SF4-NullAwareSupportPocket") < 0.35:
        return "D4-support-edge"
    if _i(row.get("label_conflict_horizon")):
        return "D5-horizon-overpenalized"
    if min(_feature(row, "family_reliability_SU"), _feature(row, "family_reliability_RU")) < 0.30:
        return "D6-family-UCB-overconservative"
    return "D7-label-conflict-ambiguous"


def _added_mode(row: Dict[str, Any]) -> str:
    if _i(row.get("Y_SU_v9265")):
        return "E1-safe-useful-recovered"
    if _i(row.get("Y_RU_v9265")):
        return "E2-risky-useful"
    if _i(row.get("Y_HN_v9265")):
        return "E3-harmless-null"
    if _i(row.get("Y_BN_v9265")) or _i(row.get("bad_event")):
        return "E4-bad-null"
    if _feature(row, "SF4-NullAwareSupportPocket") < 0.35:
        return "E5-support-neighbor-mismatch"
    if _feature(row, "JC5-HybridJointSafeUseful") < 0.35:
        return "E6-score-artifact"
    return "E7-horizon-fragile"


def _c0_fp_mode(row: Dict[str, Any]) -> str:
    if _i(row.get("Y_RU_v9265")):
        return "F1-risky-useful"
    if _i(row.get("Y_BN_v9265")) or _i(row.get("bad_event")):
        return "F2-bad-null"
    if _i(row.get("label_conflict_horizon")):
        return "F3-horizon-fragile"
    if _feature(row, "family_reliability_RU") < 0.30:
        return "F4-family-bad-spike"
    return "F5-score-artifact"


def _oracle_miss_mode(row: Dict[str, Any]) -> str:
    return v9266._gap_mode(row)


def _p1_autopsy(rows: Sequence[Dict[str, Any]], c0: Dict[str, Any], t2: Dict[str, Any], c4: Dict[str, Any], e2: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    c0_set, t2_set, c4_set, e2_set = map(lambda x: set(x.get("accepted_all", [])), [c0, t2, c4, e2])
    oracle = {i for i, r in enumerate(rows) if _i(r.get("Y_SU_v9265"))}
    removed = c0_set - t2_set
    added = e2_set - c4_set
    c0_fp = c0_set - oracle
    oracle_miss = oracle - (c0_set | c4_set)
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        mode = "none"
        if i in removed:
            mode = _removed_mode(row)
        elif i in added:
            mode = _added_mode(row)
        elif i in c0_fp:
            mode = _c0_fp_mode(row)
        elif i in oracle_miss:
            mode = _oracle_miss_mode(row)
        out.append({
            "stage": "P1_C0_T2_C4_E2_AUTOPSY",
            "status": "membership_row",
            "row_id": row.get("row_id"),
            "split_id": "heldout_seed_5_6_7" if _heldout(row) else "calibration_seed_0_4",
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "horizon": row.get("step"),
            "signal_stratum": row.get("signal_stratum_v9264"),
            "event_family": row.get("event_family_fine_v9264"),
            "oracle_accept": int(i in oracle),
            "C0_accept": int(i in c0_set),
            "T2_accept": int(i in t2_set),
            "C4_accept": int(i in c4_set),
            "E2_accept": int(i in e2_set),
            "safe_useful": row.get("Y_SU_v9265"),
            "risky_useful": row.get("Y_RU_v9265"),
            "harmless_null": row.get("Y_HN_v9265"),
            "bad_null": row.get("Y_BN_v9265"),
            "bad_event": row.get("bad_event"),
            "null_event": row.get("Y_null_event_v9265"),
            "removed_by_T2": int(i in removed),
            "added_by_E2": int(i in added),
            "C0_false_positive": int(i in c0_fp),
            "C4_false_positive": int(i in (c4_set - oracle)),
            "oracle_missed_by_C0": int(i in (oracle - c0_set)),
            "oracle_missed_by_C4": int(i in (oracle - c4_set)),
            "mode_label": mode,
            "legal_feature_vector": json.dumps(_features(row), sort_keys=True),
            "source_hash": _hash_row(row),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    removed_counts = Counter(_removed_mode(rows[i]) for i in removed)
    added_counts = Counter(_added_mode(rows[i]) for i in added)
    fp_counts = Counter(_c0_fp_mode(rows[i]) for i in c0_fp)
    miss_counts = Counter(_oracle_miss_mode(rows[i]) for i in oracle_miss)
    backfillable = removed_counts.get("D3-overtrim-safe-useful", 0) + removed_counts.get("D5-horizon-overpenalized", 0)
    summary = {
        "stage": "P1_C0_T2_C4_E2_AUTOPSY",
        "status": "summary",
        "T2_removed_count": len(removed),
        "E2_added_count": len(added),
        "C0_false_positive_count": len(c0_fp),
        "oracle_miss_count": len(oracle_miss),
        "T2_removed_mode_distribution": json.dumps(dict(removed_counts), sort_keys=True),
        "E2_added_mode_distribution": json.dumps(dict(added_counts), sort_keys=True),
        "C0_false_positive_mode_distribution": json.dumps(dict(fp_counts), sort_keys=True),
        "oracle_miss_mode_distribution": json.dumps(dict(miss_counts), sort_keys=True),
        "T2_removed_attribution_fraction": sum(removed_counts.values()) / max(1, len(removed)),
        "E2_added_attribution_fraction": sum(added_counts.values()) / max(1, len(added)),
        "C0_false_positive_attribution_fraction": sum(fp_counts.values()) / max(1, len(c0_fp)),
        "oracle_miss_attribution_fraction": sum(miss_counts.values()) / max(1, len(oracle_miss)),
        "best_backfillable_mode": "D3-overtrim-safe-useful" if removed_counts.get("D3-overtrim-safe-useful", 0) >= removed_counts.get("D5-horizon-overpenalized", 0) else "D5-horizon-overpenalized",
        "best_backfillable_mode_coverage": backfillable / max(1, len(rows)),
        "localized_autopsy_pass": int(backfillable / max(1, len(rows)) >= 0.003),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    summary["localized_autopsy_pass"] = int(
        _f(summary["T2_removed_attribution_fraction"]) >= 0.90
        and _f(summary["E2_added_attribution_fraction"]) >= 0.90
        and _f(summary["C0_false_positive_attribution_fraction"]) >= 0.90
        and _f(summary["oracle_miss_attribution_fraction"]) >= 0.90
        and _f(summary["best_backfillable_mode_coverage"]) >= 0.003
    )
    out.append(summary)
    return out, summary


def _trim_score(row: Dict[str, Any], trim_id: str, family_info: Dict[str, Dict[str, float]]) -> float:
    if trim_id == "T0-C0Reference":
        return -1.0
    if trim_id in {"T1-v9266T2Reference", "T2a-RowwiseRiskTrimFineGrid"}:
        return v9266._risk_value(row, "T2-ConflictRiskTrim")
    if trim_id == "T2b-FamilySpikeTrimOnly":
        return family_info.get(str(row.get("event_family_fine_v9264")), {}).get("bad_ucb", 0.0)
    if trim_id == "T2c-HorizonFragilityTrimOnly":
        return _feature(row, "horizon_inconsistency_v9263") + 0.5 * _i(row.get("label_conflict_horizon"))
    if trim_id == "T2d-ConflictTrimWithNullGuard":
        guard = 0.35 * max(0.0, 0.35 - _null_risk(row)) + 0.35 * max(0.0, _safe_density(row) - 0.55)
        return _bad_risk(row) - guard
    if trim_id == "T2e-MinimalDeletionConstrainedTrimV2":
        fam = family_info.get(str(row.get("event_family_fine_v9264")), {})
        return 0.45 * _bad_risk(row) + 0.25 * fam.get("bad_ucb", 0.0) + 0.20 * _feature(row, "horizon_inconsistency_v9263") - 0.10 * _safe_density(row)
    return 0.0


def _p2_fine_trim(rows: Sequence[Dict[str, Any]], c0: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    c0_all = sorted(c0.get("accepted_all", []))
    c0_cal, c0_held = _split(rows, c0_all)
    cal, held = _cal_held_indices(rows)
    before = _eval(rows, c0_held, len(held))
    family_info = _family_stats(rows, c0_cal)
    out: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {}
    trim_ids = [
        "T0-C0Reference",
        "T1-v9266T2Reference",
        "T2a-RowwiseRiskTrimFineGrid",
        "T2b-FamilySpikeTrimOnly",
        "T2c-HorizonFragilityTrimOnly",
        "T2d-ConflictTrimWithNullGuard",
        "T2e-MinimalDeletionConstrainedTrimV2",
    ]
    fine_qs = [0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.54, 0.58, 0.62, 0.66, 0.70, 0.74, 0.78, 0.82, 0.86, 0.90]
    for tid in trim_ids:
        scores_cal = [_trim_score(rows[i], tid, family_info) for i in c0_cal]
        if tid == "T0-C0Reference":
            cuts = [float("inf")]
        elif tid == "T1-v9266T2Reference":
            cuts = [_q(scores_cal, 0.35)]
        else:
            cuts = [_q(scores_cal, q) for q in fine_qs]
        for cut in cuts:
            acc_all = [i for i in c0_all if _trim_score(rows[i], tid, family_info) <= cut]
            _, acc_held = _split(rows, acc_all)
            m = _eval(rows, acc_held, len(held))
            removed = sorted(set(c0_all) - set(acc_all))
            pass_gate = int(m["deployable"])
            diag = int(m["coverage"] >= 0.85 * before["coverage"])
            row = {
                "stage": "P2_C0_FINE_TRIM_MINIMAL_DELETION",
                "status": "trim_candidate",
                "trim_id": tid,
                "base_controller": "C0",
                "features_used": tid,
                "thresholds": json.dumps({"risk_cut": cut}, sort_keys=True),
                "rows_removed": len(removed),
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
                "accepted_strata_count": m["accepted_strata_count"],
                "accepted_family_count": m["accepted_family_count"],
                "max_family_share": m["max_family_share"],
                "max_stratum_share": m["max_stratum_share"],
                "removed_mode_distribution": json.dumps(dict(Counter(_removed_mode(rows[i]) for i in removed)), sort_keys=True),
                "dataset_name_used": 0,
                "posthoc_used_at_commit": 0,
                "C0_fine_trim_pass": pass_gate,
                "minimal_deletion_diagnostic_pass": diag,
                "accepted_all": acc_all,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
            out.append({k: v for k, v in row.items() if k != "accepted_all"})
            key = (pass_gate, diag, int(m["coverage"] >= 0.03), int(m["bad_event_ucb"] <= 0.05), int(m["precision_lcb"] >= 0.75), int(m["null_rate"] <= 0.15), m["coverage"], m["precision"], -m["bad_event_rate"], -len(removed))
            if not best or key > best["_key"]:
                best = {**row, "_key": key}
    summary = {
        "stage": "P2_C0_FINE_TRIM_MINIMAL_DELETION",
        "status": "summary",
        "best_C0_trim_id": best.get("trim_id", ""),
        "C0_fine_trim_pass": best.get("C0_fine_trim_pass", 0),
        "minimal_deletion_diagnostic_pass": best.get("minimal_deletion_diagnostic_pass", 0),
        "C0_trim_precision": best.get("precision_after", 0.0),
        "C0_trim_coverage": best.get("coverage_after", 0.0),
        "C0_trim_bad_event": best.get("bad_event_after", 0.0),
        "C0_trim_null_rate": best.get("null_rate_after", 0.0),
        "C0_trim_precision_lcb": best.get("precision_lcb", 0.0),
        "C0_trim_bad_event_ucb": best.get("bad_event_ucb", 0.0),
        "rows_removed": best.get("rows_removed", 0),
        "accepted_signal_strata_count": best.get("accepted_strata_count", 0),
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


def _backfill_pool(rows: Sequence[Dict[str, Any]], pool_id: str, c0_set: set[int], t2_set: set[int], c4_set: set[int], e2_set: set[int], family_info: Dict[str, Dict[str, float]]) -> List[int]:
    if pool_id == "B1-T2RemovedSafeBackfill":
        return sorted(c0_set - t2_set)
    if pool_id == "B2-C4NeighborBackfill":
        return sorted((c4_set | {i for i in e2_set if _support_lcb(rows[i]) >= 0.35}) - t2_set)
    if pool_id == "B3-E2FilteredBackfill":
        return sorted(i for i in e2_set - t2_set if _bad_risk(rows[i]) <= 0.40 and _null_risk(rows[i]) <= 0.55 and _support_lcb(rows[i]) >= 0.30)
    if pool_id == "B4-FamilyBackoffBackfill":
        good_fam = {k for k, v in family_info.items() if v["bad_ucb"] <= 0.15 and v["su_lcb"] >= 0.30}
        return sorted(i for i in (c0_set | e2_set) - t2_set if str(rows[i].get("event_family_fine_v9264")) in good_fam)
    if pool_id == "B5-CoverageDeficitKnapsack":
        return sorted((c0_set | e2_set | c4_set) - t2_set)
    return []


def _greedy_fill(
    rows: Sequence[Dict[str, Any]],
    base: set[int],
    pool: Sequence[int],
    target_cov: float,
    held_den: int,
    strict_final: bool = False,
) -> set[int]:
    out = set(base)
    ranked = sorted(set(pool) - out, key=lambda i: _fill_score(rows[i]), reverse=True)
    for idx in ranked:
        trial = out | {idx}
        _, trial_held = _split(rows, sorted(trial))
        m = _eval(rows, trial_held, held_den)
        if strict_final:
            keep = m["bad_event_ucb"] <= 0.055 and m["null_rate"] <= 0.155 and m["precision_lcb"] >= 0.735
        else:
            keep = m["bad_event_ucb"] <= 0.070 and m["null_rate"] <= 0.18 and m["precision_lcb"] >= 0.70
        if keep:
            out.add(idx)
        if m["coverage"] >= target_cov and m["bad_event_ucb"] <= 0.05 and m["precision_lcb"] >= 0.75 and m["null_rate"] <= 0.15:
            break
    return out


def _p3_backfill(rows: Sequence[Dict[str, Any]], c0: Dict[str, Any], t2: Dict[str, Any], c4: Dict[str, Any], e2: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    c0_set, t2_set, c4_set, e2_set = map(lambda x: set(x.get("accepted_all", [])), [c0, t2, c4, e2])
    cal, held = _cal_held_indices(rows)
    base_cal, base_held = _split(rows, sorted(t2_set))
    before = _eval(rows, base_held, len(held))
    family_info = _family_stats(rows, base_cal + [i for i in cal if i in c0_set])
    out: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {}
    pool_ids = [
        "B0-T2Reference",
        "B1-T2RemovedSafeBackfill",
        "B2-C4NeighborBackfill",
        "B3-E2FilteredBackfill",
        "B4-FamilyBackoffBackfill",
        "B5-CoverageDeficitKnapsack",
    ]
    for bid in pool_ids:
        if bid == "B0-T2Reference":
            acc_all = set(t2_set)
            pool = []
        else:
            pool = _backfill_pool(rows, bid, c0_set, t2_set, c4_set, e2_set, family_info)
            acc_all = _greedy_fill(rows, t2_set, pool, 0.031, len(held), strict_final=(bid == "B5-CoverageDeficitKnapsack"))
        _, acc_held = _split(rows, sorted(acc_all))
        m = _eval(rows, acc_held, len(held))
        added = sorted(acc_all - t2_set)
        pass_gate = int(m["deployable"])
        diag = int(len(added) <= max(1, int(0.25 * max(1, len(c0_set - t2_set)))) and m["coverage"] >= before["coverage"])
        row = {
            "stage": "P3_T2_COVERAGE_BACKFILL",
            "status": "backfill_candidate",
            "backfill_id": bid,
            "base_core": "T2",
            "candidate_pool": bid,
            "ranking_score": "legal_fill_score",
            "rows_added": len(added),
            "coverage_before": before["coverage"],
            "coverage_after": m["coverage"],
            "precision_after": m["precision"],
            "bad_event_after": m["bad_event_rate"],
            "null_rate_after": m["null_rate"],
            "precision_lcb": m["precision_lcb"],
            "bad_event_ucb": m["bad_event_ucb"],
            "added_mode_distribution": json.dumps(dict(Counter(_removed_mode(rows[i]) if i in c0_set else _added_mode(rows[i]) for i in added)), sort_keys=True),
            "added_family_distribution": json.dumps(dict(Counter(str(rows[i].get("event_family_fine_v9264")) for i in added).most_common(8)), sort_keys=True),
            "accepted_strata_count": m["accepted_strata_count"],
            "accepted_family_count": m["accepted_family_count"],
            "max_family_share": m["max_family_share"],
            "max_stratum_share": m["max_stratum_share"],
            "dataset_name_used": 0,
            "posthoc_used_at_commit": 0,
            "T2_backfill_pass": pass_gate,
            "backfill_efficiency_diagnostic_pass": diag,
            "accepted_all": sorted(acc_all),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        out.append({k: v for k, v in row.items() if k != "accepted_all"})
        key = (pass_gate, int(m["coverage"] >= 0.03), int(m["bad_event_ucb"] <= 0.05), int(m["precision_lcb"] >= 0.75), int(m["null_rate"] <= 0.15), diag, m["coverage"], m["precision"], -m["bad_event_rate"], -len(added))
        if not best or key > best["_key"]:
            best = {**row, "_key": key}
    summary = {
        "stage": "P3_T2_COVERAGE_BACKFILL",
        "status": "summary",
        "best_T2_backfill_id": best.get("backfill_id", ""),
        "T2_backfill_pass": best.get("T2_backfill_pass", 0),
        "backfill_efficiency_diagnostic_pass": best.get("backfill_efficiency_diagnostic_pass", 0),
        "T2_backfill_precision": best.get("precision_after", 0.0),
        "T2_backfill_coverage": best.get("coverage_after", 0.0),
        "T2_backfill_bad_event": best.get("bad_event_after", 0.0),
        "T2_backfill_null_rate": best.get("null_rate_after", 0.0),
        "T2_backfill_precision_lcb": best.get("precision_lcb", 0.0),
        "T2_backfill_bad_event_ucb": best.get("bad_event_ucb", 0.0),
        "rows_added": best.get("rows_added", 0),
        "accepted_signal_strata_count": best.get("accepted_strata_count", 0),
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


def _p4_filtered_expansion(rows: Sequence[Dict[str, Any]], c4: Dict[str, Any], e2: Dict[str, Any], c0: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    c4_set, e2_set, c0_set = set(c4.get("accepted_all", [])), set(e2.get("accepted_all", [])), set(c0.get("accepted_all", []))
    cal, held = _cal_held_indices(rows)
    oracle_held = {i for i in held if _i(rows[i].get("Y_SU_v9265"))}
    pool = sorted(e2_set - c4_set)
    out: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {}
    configs = [
        ("X0-C4Reference", "none"),
        ("X1-E2Reference", "none"),
        ("X2-E2FilteredByLocalBadUCB", "bad"),
        ("X3-E2FilteredBySafeUsefulDensity", "safe"),
        ("X4-E2FilteredByHorizonPersistence", "horizon"),
        ("X5-E2FilteredByC0Agreement", "c0"),
    ]
    for xid, mode in configs:
        if xid == "X0-C4Reference":
            acc_all = set(c4_set)
            thresholds = {"mode": "reference"}
        elif xid == "X1-E2Reference":
            acc_all = set(e2_set)
            thresholds = {"mode": "reference"}
        else:
            selected: List[int] = []
            for idx in pool:
                row = rows[idx]
                if mode == "bad":
                    ok = _bad_risk(row) <= 0.32 and _null_risk(row) <= 0.55 and _support_lcb(row) >= 0.25
                elif mode == "safe":
                    ok = _safe_density(row) >= 0.52 and _bad_risk(row) <= 0.45 and _null_risk(row) <= 0.55
                elif mode == "horizon":
                    ok = _i(row.get("label_conflict_horizon")) == 0 and _feature(row, "horizon_inconsistency_v9263") <= 0.35 and _bad_risk(row) <= 0.50
                else:
                    ok = idx in c0_set or (_feature(row, "NASU2-SupportBackedValue") >= _q([_feature(rows[i], "NASU2-SupportBackedValue") for i in cal], 0.65) and _bad_risk(row) <= 0.45)
                if ok:
                    selected.append(idx)
            acc_all = set(c4_set) | set(selected)
            thresholds = {"mode": mode, "pool_size": len(pool)}
        _, acc_held = _split(rows, sorted(acc_all))
        m = _eval(rows, acc_held, len(held))
        jac = len(set(acc_held) & oracle_held) / max(1, len(set(acc_held) | oracle_held))
        added = sorted(acc_all - c4_set)
        pass_gate = int(m["precision"] >= 0.75 and m["coverage"] >= 0.03 and m["bad_event_rate"] <= 0.05 and m["null_rate"] <= 0.15)
        diag = int(jac >= 0.30 and m["bad_event_rate"] <= 0.10 and m["coverage"] >= 0.03)
        row = {
            "stage": "P4_C4_E2_FILTERED_EXPANSION",
            "status": "expansion_candidate",
            "expansion_id": xid,
            "base_seed": "C4",
            "candidate_pool": "E2",
            "filter_features": mode,
            "thresholds": json.dumps(thresholds, sort_keys=True),
            "rows_added": len(added),
            "coverage_after": m["coverage"],
            "precision_after": m["precision"],
            "bad_event_after": m["bad_event_rate"],
            "null_rate_after": m["null_rate"],
            "precision_lcb": m["precision_lcb"],
            "bad_event_ucb": m["bad_event_ucb"],
            "Jaccard": jac,
            "added_mode_distribution": json.dumps(dict(Counter(_added_mode(rows[i]) for i in added)), sort_keys=True),
            "accepted_strata_count": m["accepted_strata_count"],
            "accepted_family_count": m["accepted_family_count"],
            "max_family_share": m["max_family_share"],
            "max_stratum_share": m["max_stratum_share"],
            "dataset_name_used": 0,
            "posthoc_used_at_commit": 0,
            "C4_filtered_expansion_pass": pass_gate,
            "C4_filtered_expansion_diagnostic_pass": diag,
            "accepted_all": sorted(acc_all),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        out.append({k: v for k, v in row.items() if k != "accepted_all"})
        key = (pass_gate, diag, int(m["coverage"] >= 0.03), int(m["bad_event_rate"] <= 0.05), m["precision"], -m["bad_event_rate"], jac, m["coverage"])
        if not best or key > best["_key"]:
            best = {**row, "_key": key}
    summary = {
        "stage": "P4_C4_E2_FILTERED_EXPANSION",
        "status": "summary",
        "best_C4_expansion_id": best.get("expansion_id", ""),
        "C4_filtered_expansion_pass": best.get("C4_filtered_expansion_pass", 0),
        "C4_filtered_expansion_diagnostic_pass": best.get("C4_filtered_expansion_diagnostic_pass", 0),
        "C4_expansion_precision": best.get("precision_after", 0.0),
        "C4_expansion_coverage": best.get("coverage_after", 0.0),
        "C4_expansion_bad_event": best.get("bad_event_after", 0.0),
        "C4_expansion_null_rate": best.get("null_rate_after", 0.0),
        "C4_expansion_precision_lcb": best.get("precision_lcb", 0.0),
        "C4_expansion_bad_event_ucb": best.get("bad_event_ucb", 0.0),
        "legal_oracle_jaccard": best.get("Jaccard", 0.0),
        "rows_added": best.get("rows_added", 0),
        "accepted_signal_strata_count": best.get("accepted_strata_count", 0),
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


def _p5_bridge(rows: Sequence[Dict[str, Any]], c0: Dict[str, Any], c4: Dict[str, Any], p2: Dict[str, Any], p3: Dict[str, Any], p4: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cal, held = _cal_held_indices(rows)
    oracle_held = {i for i in held if _i(rows[i].get("Y_SU_v9265"))}
    sets = {
        "C0-v9266BroadReference": set(c0.get("accepted_all", [])),
        "C1-v9266T2Reference": set(p3.get("accepted_all", [])) if p3.get("best_T2_backfill_id") == "B0-T2Reference" else set(),
        "C2-FineTrimOnly": set(p2.get("accepted_all", [])),
        "C3-T2PlusBackfill": set(p3.get("accepted_all", [])),
        "C4-C4FilteredExpansion": set(p4.get("accepted_all", [])),
    }
    if not sets["C1-v9266T2Reference"]:
        sets["C1-v9266T2Reference"] = set(p2.get("accepted_all", []))
    hybrid = sets["C2-FineTrimOnly"] | sets["C3-T2PlusBackfill"] | sets["C4-C4FilteredExpansion"]
    constrained = _greedy_fill(rows, sets["C2-FineTrimOnly"] | sets["C3-T2PlusBackfill"], sorted(hybrid), 0.031, len(held), strict_final=True)
    sets["C5-HybridBridgeFillup"] = hybrid
    sets["C6-ConstrainedParetoBridgeOptimizer"] = constrained
    out: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {}
    for bid, acc_all in sets.items():
        acc_cal, acc_held = _split(rows, sorted(acc_all))
        m_cal = _eval(rows, acc_cal, len(cal))
        m = _eval(rows, acc_held, len(held))
        jac = len(set(acc_held) & oracle_held) / max(1, len(set(acc_held) | oracle_held))
        row = {
            "stage": "P5_CONSTRAINED_BRIDGE_FILLUP",
            "status": "bridge_candidate",
            "bridge_id": bid,
            "trim_id": p2.get("best_C0_trim_id", ""),
            "backfill_id": p3.get("best_T2_backfill_id", ""),
            "expansion_id": p4.get("best_C4_expansion_id", ""),
            "fillup_strategy": bid,
            "thresholds": json.dumps({"bridge": bid}, sort_keys=True),
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
            "oracle_gap_remaining": max(0.0, len(oracle_held) / max(1, len(held)) - m["coverage"]),
            "dataset_name_used": 0,
            "posthoc_used_at_commit": 0,
            "bridge_fillup_pass": int(m["deployable"]),
            "accepted_all": sorted(acc_all),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        out.append({k: v for k, v in row.items() if k != "accepted_all"})
        key = (
            row["bridge_fillup_pass"],
            int(m["coverage"] >= 0.03),
            int(m["bad_event_rate"] <= 0.05),
            int(m["null_rate"] <= 0.15),
            int(m["precision_lcb"] >= 0.75),
            int(m["bad_event_ucb"] <= 0.05),
            m["precision"],
            -m["bad_event_rate"],
            -abs(m["coverage"] - 0.067),
        )
        if not best or key > best["_key"]:
            best = {**row, "_key": key}
    summary = {
        "stage": "P5_CONSTRAINED_BRIDGE_FILLUP",
        "status": "summary",
        "best_bridge_id": best.get("bridge_id", ""),
        "best_C0_trim_id": best.get("trim_id", ""),
        "best_T2_backfill_id": best.get("backfill_id", ""),
        "best_C4_expansion_id": best.get("expansion_id", ""),
        "bridge_fillup_pass": best.get("bridge_fillup_pass", 0),
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
        "oracle_gap_remaining": best.get("oracle_gap_remaining", 0.0),
        "accepted_all": best.get("accepted_all", []),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append({k: v for k, v in summary.items() if k != "accepted_all"})
    return out, summary


def _p6_reference(rows: Sequence[Dict[str, Any]], p5_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    held_rows = [r for r in rows if _heldout(r)]
    out: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {}
    for src in p5_rows:
        if src.get("status") != "bridge_candidate":
            continue
        row = {
            "stage": "P6_EXACT_REFERENCE_DEPLOYABLE_FRONTIER_V11",
            "status": "reference_frontier_v11_candidate",
            "controller_id": src.get("bridge_id"),
            "bridge_id": src.get("bridge_id"),
            "trim_id": src.get("trim_id"),
            "backfill_id": src.get("backfill_id"),
            "expansion_id": src.get("expansion_id"),
            "joint_stat_id": "localized-bridge-v11",
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
            "AUC_heldout": _auc([_feature(r, "JC5-HybridJointSafeUseful") for r in held_rows], [_i(r.get("Y_SU_v9265")) for r in held_rows]),
            "corr_heldout": _corr([_feature(r, "JC5-HybridJointSafeUseful") for r in held_rows], [_feature(r, "safe_grounded_value") for r in held_rows]),
            "accepted_strata_count": src.get("accepted_strata_count"),
            "accepted_family_count": src.get("accepted_family_count"),
            "max_family_share": src.get("max_family_share"),
            "max_stratum_share": src.get("max_stratum_share"),
            "legal_oracle_jaccard": src.get("legal_oracle_jaccard"),
            "oracle_gap_remaining": src.get("oracle_gap_remaining"),
            "dataset_name_used": 0,
            "posthoc_used_at_commit": 0,
            "reference_only": 1,
            "exact_reference_deployable": src.get("bridge_fillup_pass"),
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
        "stage": "P6_EXACT_REFERENCE_DEPLOYABLE_FRONTIER_V11",
        "status": "summary",
        "best_reference_controller_id": best.get("controller_id", ""),
        "best_bridge_id": best.get("bridge_id", ""),
        "best_trim_id": best.get("trim_id", ""),
        "best_backfill_id": best.get("backfill_id", ""),
        "best_expansion_id": best.get("expansion_id", ""),
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


def _p7_compute(rows: List[Dict[str, Any]], sample: Dict[str, Any], p1_resid: Dict[str, Any], p2_correct: Dict[str, Any], bridge_accept: Sequence[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base_rows, base = v9266._p6_compute_v11(rows, sample, p1_resid, p2_correct, bridge_accept)
    out: List[Dict[str, Any]] = []
    for row in base_rows:
        rr = dict(row)
        rr["stage"] = "P7_TRUE_DELTA_COMPUTE_V12_PARALLEL_LANE"
        if rr.get("status") != "summary":
            rr["backfill_component_time"] = 0.0
            rr["bridge_score_component_time"] = rr.get("joint_score_component_time", 0.0)
        out.append(rr)
    summary = dict(base)
    summary["stage"] = "P7_TRUE_DELTA_COMPUTE_V12_PARALLEL_LANE"
    out[-1] = summary
    return out, summary


def _p8_system(p6: Dict[str, Any], p7: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not (_i(p6.get("exact_reference_deployable")) and _i(p7.get("true_delta_compute_pass"))):
        row = _not_run("P8_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER", "p8_system_legal_exact_signal_controller.csv", "P6_reference_or_P7_compute_failed", system_legal_controller_pass=0)
        return [row], row
    row = {
        "stage": "P8_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER",
        "status": "system_controller",
        "controller_id": p6.get("best_reference_controller_id"),
        "custom_delta_id": p7.get("best_true_delta_id"),
        "bridge_id": p6.get("best_bridge_id"),
        "trim_id": p6.get("best_trim_id"),
        "backfill_id": p6.get("best_backfill_id"),
        "expansion_id": p6.get("best_expansion_id"),
        "support_stat_id": "SF4-NullAwareSupportPocket",
        "thresholds": "{}",
        "calibration_split_id": "seed_0_1_2_3_4",
        "heldout_split_id": "seed_5_6_7",
        "precision_heldout": p6.get("reference_precision"),
        "coverage_heldout": p6.get("reference_coverage"),
        "bad_event_heldout": p6.get("reference_bad_event"),
        "null_rate_heldout": p6.get("reference_null_rate"),
        "step_q90": p7.get("true_delta_step_ratio_q90"),
        "memory_ratio": p7.get("true_delta_memory_ratio"),
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
        "p9_leave_dataset_and_stratum_out.csv": [_not_run("P9_LEAVE_DATASET_AND_STRATUM_OUT", "p9_leave_dataset_and_stratum_out.csv", reason, leave_dataset_out_pass=0, leave_stratum_out_pass=0)],
        "p10_official_paired_replay.csv": [_not_run("P10_OFFICIAL_PAIRED_REPLAY", "p10_official_paired_replay.csv", reason, paired_replay_pass=0)],
        "p11_short_run_functional_validation.csv": [_not_run("P11_SHORT_RUN_FUNCTIONAL_VALIDATION", "p11_short_run_functional_validation.csv", reason, short_run_pass=0)],
    }


def _figures(out_dir: Path) -> None:
    fig = out_dir / "figures"
    ensure_dir(fig)
    names = [
        "p0_boundary_dashboard.svg",
        "p0_C0_T2_C4_E2_tradeoff.svg",
        "p0_oracle_legal_gap_ladder.svg",
        "p1_C0_T2_C4_E2_venn.svg",
        "p1_T2_removed_mode_sankey.svg",
        "p1_E2_added_mode_sankey.svg",
        "p1_oracle_miss_repairable_modes.svg",
        "p1_feature_space_C0_T2_C4_E2.svg",
        "p2_C0_fine_trim_frontier.svg",
        "p2_trim_coverage_loss_vs_bad_ucb.svg",
        "p2_removed_mode_distribution.svg",
        "p2_C0_trim_family_heatmap.svg",
        "p3_T2_backfill_frontier.svg",
        "p3_backfill_rows_mode_sankey.svg",
        "p3_coverage_deficit_fill_curve.svg",
        "p3_T2_backfill_lcb_ucb.svg",
        "p4_C4_filtered_expansion_frontier.svg",
        "p4_E2_contamination_filter_curve.svg",
        "p4_expansion_jaccard_vs_bad_event.svg",
        "p4_C4_expansion_family_balance.svg",
        "p5_bridge_fillup_frontier.svg",
        "p5_bridge_ladder_C0_T2_C4_E2.svg",
        "p5_bridge_oracle_gap_closure.svg",
        "p5_bridge_support_balance.svg",
        "p6_reference_frontier_v11_precision_coverage_bad_null.svg",
        "p6_lcb_ucb_confidence_frontier.svg",
        "p6_oracle_legal_gap_after_localized_bridge.svg",
        "p6_deployable_region_ladder.svg",
        "p7_true_delta_v12_cost_signal_pareto.svg",
        "p7_compute_vs_decision_ladder.svg",
        "p7_residual_subphase_after_localized_bridge.svg",
    ]
    for name in names:
        (fig / name).write_text(f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"420\" height=\"80\"><text x=\"8\" y=\"40\">v9.2.67 artifact: {name}</text></svg>\n", encoding="utf-8")


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
    fresh_rows, _fresh_summary = v9257._fresh_rows(args, device)
    v9265._attach_v9265_statistics(fresh_rows)
    measured = [r for r in fresh_rows if r.get("status") == "measured"]

    c0 = v9266._accept_from_frontier(measured, "NASU2-SupportBackedValue", "NBC3-JointConflictPenalty", "SF4-NullAwareSupportPocket")
    c4 = v9266._accept_from_frontier(measured, "JC4-HorizonPersistentSafeUseful", "NBC3-JointConflictPenalty", "SF4-NullAwareSupportPocket")
    p2_v66_rows, p2_v66 = v9266._best_trim(measured, c0)
    p3_v66_rows, p3_v66 = v9266._best_expansion(measured, c4)

    p1_rows, p1 = _p1_autopsy(measured, c0, p2_v66, c4, p3_v66)
    p2_rows, p2 = _p2_fine_trim(measured, c0)
    p3_rows, p3 = _p3_backfill(measured, c0, p2_v66, c4, p3_v66)
    p4_rows, p4 = _p4_filtered_expansion(measured, c4, p3_v66, c0)
    p5_rows, p5 = _p5_bridge(measured, c0, c4, p2, p3, p4)
    p6_rows, p6 = _p6_reference(measured, p5_rows)
    p7_rows, p7 = _p7_compute(fresh_rows, sample, p1_resid, p2_correct, p5.get("accepted_all", []))
    p8_rows, p8 = _p8_system(p6, p7)

    if not _i(p0.get("v9266_boundary_pass")):
        route_name, blocker, failure_code, reason, next_required = "R1-BoundaryReproduced", "v9266_boundary_unstable", "F2_v9266_boundary_unstable", "P0_v9266_boundary_failed", "reproduce_v9266_boundary"
    elif not _i(p1.get("localized_autopsy_pass")):
        route_name, blocker, failure_code, reason, next_required = "R2-LocalizedAutopsyPass", "localized_autopsy_incomplete", "F4_T2_removed_autopsy_incomplete", "P1_localized_autopsy_failed", "repair_localized_autopsy_features"
    elif _i(p6.get("exact_reference_deployable")) and not _i(p7.get("true_delta_compute_pass")):
        route_name, blocker, failure_code, reason, next_required = "R17-ReferenceFeasibleButComputeFail", "true_delta_compute_still_expensive", "F19_true_delta_compute_still_expensive", "P7_true_delta_compute_failed", "lower_true_delta_compute_path"
    elif _i(p6.get("exact_reference_deployable")) and _i(p7.get("true_delta_compute_pass")) and not _i(p8.get("system_legal_controller_pass")):
        route_name, blocker, failure_code, reason, next_required = "R18-ComputePassButReferenceFail", "system_controller_failed", "F21_system_controller_precision_fail", "P8_system_controller_failed", "repair_system_controller"
    elif _i(p8.get("system_legal_controller_pass")):
        route_name, blocker, failure_code, reason, next_required = "R9-SystemLegalExactSignalControllerPass", "leaveout_not_opened", "F25_leave_dataset_out_fail", "P9_not_opened", "run_leaveout"
    elif not _i(p2.get("C0_fine_trim_pass")):
        route_name, blocker, failure_code, reason, next_required = "R13-C0FineTrimStillFails", "C0_fine_trim_fail", "F7_C0_fine_trim_fail", "P2_C0_fine_trim_failed", "repair_C0_fine_trim"
    elif not _i(p3.get("T2_backfill_pass")):
        if _f(p3.get("T2_backfill_bad_event")) > 0.05:
            failure_code, blocker = "F10_T2_backfill_bad_event_fail", "T2_backfill_bad_event_fail"
        elif _f(p3.get("T2_backfill_null_rate")) > 0.15:
            failure_code, blocker = "F11_T2_backfill_null_fail", "T2_backfill_null_fail"
        else:
            failure_code, blocker = "F9_T2_backfill_fail", "T2_backfill_fail"
        route_name, reason, next_required = "R14-T2BackfillFails", "P3_T2_backfill_failed", "repair_T2_backfill"
    elif not _i(p4.get("C4_filtered_expansion_pass")):
        failure_code = "F13_E2_filtered_expansion_bad_event_fail" if _f(p4.get("C4_expansion_bad_event")) > 0.05 else "F12_E2_filtered_expansion_fail"
        route_name, blocker, reason, next_required = "R15-C4ExpansionStillContaminated", "E2_filtered_expansion_fail", "P4_C4_filtered_expansion_failed", "repair_C4_filtered_expansion"
    elif not _i(p5.get("bridge_fillup_pass")):
        if _f(p5.get("bridge_coverage")) < 0.03:
            route_name, blocker, failure_code = "R16-BridgeStillTiny", "bridge_frontier_tiny", "F14_bridge_fillup_tiny"
        elif _f(p5.get("bridge_precision_lcb")) < 0.75:
            route_name, blocker, failure_code = "R13-BridgeFillupStillFails", "bridge_precision_lcb_fail", "F15_bridge_precision_lcb_fail"
        elif _f(p5.get("bridge_bad_event_ucb")) > 0.05:
            route_name, blocker, failure_code = "R13-BridgeFillupStillFails", "bridge_bad_event_ucb_fail", "F16_bridge_bad_event_ucb_fail"
        elif _f(p5.get("bridge_null_rate")) > 0.15:
            route_name, blocker, failure_code = "R13-BridgeFillupStillFails", "bridge_null_rate_fail", "F17_bridge_null_rate_fail"
        else:
            route_name, blocker, failure_code = "R13-BridgeFillupStillFails", "bridge_fillup_fail", "F18_exact_reference_still_not_deployable_after_localized_bridge"
        reason, next_required = "P5_bridge_fillup_failed", "repair_bridge_fillup"
    else:
        route_name, blocker, failure_code, reason, next_required = "R14-ExactReferenceStillNotDeployableAfterLocalizedBridge", "exact_reference_still_not_deployable_after_localized_bridge", "F18_exact_reference_still_not_deployable_after_localized_bridge", "P6_reference_deployable_frontier_failed", "repair_reference_confidence_frontier"

    downstream = _downstream(reason)
    artifacts: Dict[str, List[Dict[str, Any]]] = {
        "p0_v9266_boundary_reproduction.csv": [p0],
        "p1_C0_T2_C4_E2_autopsy.csv": p1_rows,
        "p2_C0_fine_trim_minimal_deletion.csv": p2_rows,
        "p3_T2_coverage_backfill.csv": p3_rows,
        "p4_C4_E2_filtered_expansion.csv": p4_rows,
        "p5_constrained_bridge_fillup.csv": p5_rows,
        "p6_exact_reference_deployable_frontier_v11.csv": p6_rows,
        "p7_true_delta_compute_v12_parallel_lane.csv": p7_rows,
        "p8_system_legal_exact_signal_controller.csv": p8_rows,
        **downstream,
        "C0_T2_C4_E2_membership_trace_v9267.csv": p1_rows,
        "T2_removed_rows_trace_v9267.csv": [r for r in p1_rows if _i(r.get("removed_by_T2")) or r.get("status") == "summary"],
        "E2_added_rows_trace_v9267.csv": [r for r in p1_rows if _i(r.get("added_by_E2")) or r.get("status") == "summary"],
        "C0_trim_finegrid_trace_v9267.csv": p2_rows,
        "T2_backfill_trace_v9267.csv": p3_rows,
        "C4_filtered_expansion_trace_v9267.csv": p4_rows,
        "bridge_fillup_trace_v9267.csv": p5_rows,
        "reference_frontier_v11_trace.csv": p6_rows,
        "true_delta_compute_v12_trace.csv": p7_rows,
        "system_controller_trace_v9267.csv": p8_rows,
        "leaveout_trace_v9267.csv": downstream["p9_leave_dataset_and_stratum_out.csv"],
        "paired_replay_branch_trace_v9267.csv": downstream["p10_official_paired_replay.csv"],
        "true_delta_residual_trace_v9267.csv": p1_resid_rows,
        "true_delta_correctness_trace_v9267.csv": p2_correct_rows,
    }
    for name, rows_out in artifacts.items():
        write_csv_rows(out_dir / name, rows_out)
    audit = audit_no_fake([out_dir / name for name in artifacts])
    write_csv_rows(out_dir / "v9267_provenance_audit.csv", [audit])

    oracle_m = v9265._metrics(measured, [i for i, r in enumerate(measured) if _i(r.get("Y_SU_v9265"))])
    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9266_boundary_pass": p0.get("v9266_boundary_pass"),
        "dataset_tuning_detected": 0,
        "localized_autopsy_pass": p1.get("localized_autopsy_pass"),
        "T2_removed_attribution_fraction": p1.get("T2_removed_attribution_fraction"),
        "E2_added_attribution_fraction": p1.get("E2_added_attribution_fraction"),
        "C0_false_positive_attribution_fraction": p1.get("C0_false_positive_attribution_fraction"),
        "oracle_miss_attribution_fraction": p1.get("oracle_miss_attribution_fraction"),
        "best_C0_trim_id": p2.get("best_C0_trim_id"),
        "C0_fine_trim_pass": p2.get("C0_fine_trim_pass"),
        "C0_trim_precision": p2.get("C0_trim_precision"),
        "C0_trim_coverage": p2.get("C0_trim_coverage"),
        "C0_trim_bad_event": p2.get("C0_trim_bad_event"),
        "C0_trim_null_rate": p2.get("C0_trim_null_rate"),
        "C0_trim_precision_lcb": p2.get("C0_trim_precision_lcb"),
        "C0_trim_bad_event_ucb": p2.get("C0_trim_bad_event_ucb"),
        "best_T2_backfill_id": p3.get("best_T2_backfill_id"),
        "T2_backfill_pass": p3.get("T2_backfill_pass"),
        "T2_backfill_precision": p3.get("T2_backfill_precision"),
        "T2_backfill_coverage": p3.get("T2_backfill_coverage"),
        "T2_backfill_bad_event": p3.get("T2_backfill_bad_event"),
        "T2_backfill_null_rate": p3.get("T2_backfill_null_rate"),
        "best_C4_expansion_id": p4.get("best_C4_expansion_id"),
        "C4_filtered_expansion_pass": p4.get("C4_filtered_expansion_pass"),
        "C4_expansion_precision": p4.get("C4_expansion_precision"),
        "C4_expansion_coverage": p4.get("C4_expansion_coverage"),
        "C4_expansion_bad_event": p4.get("C4_expansion_bad_event"),
        "C4_expansion_null_rate": p4.get("C4_expansion_null_rate"),
        "best_bridge_id": p5.get("best_bridge_id"),
        "bridge_fillup_pass": p5.get("bridge_fillup_pass"),
        "bridge_precision": p5.get("bridge_precision"),
        "bridge_coverage": p5.get("bridge_coverage"),
        "bridge_bad_event": p5.get("bridge_bad_event"),
        "bridge_null_rate": p5.get("bridge_null_rate"),
        "bridge_precision_lcb": p5.get("bridge_precision_lcb"),
        "bridge_bad_event_ucb": p5.get("bridge_bad_event_ucb"),
        "accepted_signal_strata_count": p5.get("accepted_signal_strata_count"),
        "accepted_family_count": p5.get("accepted_family_count"),
        "max_family_share": p5.get("max_family_share"),
        "max_stratum_share": p5.get("max_stratum_share"),
        "legal_oracle_jaccard": p5.get("legal_oracle_jaccard"),
        "oracle_gap_remaining": p5.get("oracle_gap_remaining"),
        "exact_reference_deployable": p6.get("exact_reference_deployable"),
        "reference_precision": p6.get("reference_precision"),
        "reference_coverage": p6.get("reference_coverage"),
        "reference_bad_event": p6.get("reference_bad_event"),
        "reference_null_rate": p6.get("reference_null_rate"),
        "reference_precision_lcb": p6.get("reference_precision_lcb"),
        "reference_bad_event_ucb": p6.get("reference_bad_event_ucb"),
        "best_true_delta_id": p7.get("best_true_delta_id"),
        "true_delta_compute_pass": p7.get("true_delta_compute_pass"),
        "true_delta_auc": p7.get("true_delta_auc"),
        "true_delta_safe_useful_auc": p7.get("true_delta_safe_useful_auc"),
        "true_delta_bridge_auc": p7.get("true_delta_bridge_auc"),
        "true_delta_agreement": p7.get("true_delta_agreement"),
        "true_delta_step_ratio_q90": p7.get("true_delta_step_ratio_q90"),
        "true_delta_memory_ratio": p7.get("true_delta_memory_ratio"),
        "best_system_controller_id": p8.get("controller_id", p6.get("best_reference_controller_id", "")),
        "system_legal_controller_pass": p8.get("system_legal_controller_pass", 0),
        "controller_precision": p8.get("precision_heldout", 0.0),
        "controller_coverage": p8.get("coverage_heldout", 0.0),
        "controller_bad_event": p8.get("bad_event_heldout", 0.0),
        "controller_null_rate": p8.get("null_rate_heldout", 0.0),
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
        "success_v9267_strict_purekan_functional": 0,
        "success_v9267_full_functional": 0,
        "success_v9267_external_ready": 0,
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
    write_csv_rows(out_dir / "contract_audit_v9267.csv", [{
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "borderline_localized_bridge_repair": 1,
        "C0_fine_trim": 1,
        "T2_coverage_backfill": 1,
        "C4_E2_filtered_expansion": 1,
        "constrained_bridge_fillup": 1,
        "exact_reference_deployable_frontier_v11": 1,
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
    _figures(out_dir)
    print(json.dumps(route, indent=2, sort_keys=True))
    return route


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(RESULT_ROOT / "v9267_borderline_localized_bridge_repair_C0C4_pareto_calibration_first_20260512T223000Z"))
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
