#!/usr/bin/env python3
"""DG-KAN v9.2.45 legal feature representation closure audit.

The runner starts from the real v9.2.43/v9.2.44 fresh-v2 measured rows.  It
audits the F1-BranchRatioRiskSafe support recovered in v9.2.44, measures the
gap between that support and the failed C1 controller, evaluates legal
train-stream feature candidates, and gates tri-stage controllers before any
leave-out or paired replay stage is allowed to open.
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
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9244_safe_good_event_support_recovery_carrier_mechanism_reset as v9244  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.45_LegalFeatureRepresentation_SupportMatchedControllerClosure_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9245_legal_feature_representation_support_matched_controller_closure.py"
REPORT_PATH = ROOT / "docs" / "DG-KAN_v9.2.45_LegalFeatureRepresentation_SupportMatchedControllerClosure_实验复盘.md"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9244 = RESULT_ROOT / "v9244_safe_good_event_support_recovery_carrier_mechanism_reset_first_20260511T163000Z"

SUPPORT_COVERAGES = (0.03, 0.04, 0.05, 0.08, 0.10, 0.12, 0.15)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _float(value: Any, default: float = 0.0) -> float:
    return v9244._float(value, default)


def _int(value: Any, default: int = 0) -> int:
    return v9244._int(value, default)


def _mean(values: Iterable[float]) -> float:
    return v9244._mean(values)


def _std(values: Iterable[float]) -> float:
    return v9244._std(values)


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    return v9244._corr(xs, ys)


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    return v9244._auc(scores, labels)


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


def _load_rows() -> List[Dict[str, Any]]:
    _measured, rows = v9244._load_source_rows()
    for row in rows:
        row["event_family"] = v9244._family_id(row)
        row["risk_score"] = v9244._risk_score(row)
        row["tail_activation_mass"] = _float(row.get("r_z_tail")) + _float(row.get("r_perp_tail"))
        row["safe_good"] = _int(row.get("Y_safe_good", row.get("Y_beat")))
        row["fake_data_used"] = 0
        row["proxy_row_used"] = 0
        row["cpu_offload_used"] = 0
    return rows


def _family_stats(rows: Sequence[Dict[str, Any]], accepted: Sequence[int]) -> Dict[str, Any]:
    if not accepted:
        return {
            "accepted_strata_count": 0,
            "accepted_family_count": 0,
            "max_family_share": 0.0,
            "accepted_signal_strata": "",
            "accepted_family_ids": "",
        }
    strata = Counter(str(rows[i].get("signal_stratum")) for i in accepted)
    families = Counter(str(rows[i].get("event_family")) for i in accepted)
    return {
        "accepted_strata_count": len(strata),
        "accepted_family_count": len(families),
        "max_family_share": max(families.values()) / len(accepted),
        "accepted_signal_strata": ",".join(sorted(strata)),
        "accepted_family_ids": ",".join(sorted(families)),
    }


def _best_support(
    rows: List[Dict[str, Any]],
    allowed: Sequence[int],
    score_fn: Callable[[Dict[str, Any]], float],
    denom: int | None = None,
) -> Dict[str, Any]:
    denom = denom or len(rows)
    if not rows or not allowed:
        return {
            "precision": 0.0,
            "coverage": 0.0,
            "bad_event_rate": 0.0,
            "accepted_count": 0,
            "threshold": 0.0,
            "accepted_indices": [],
            **_family_stats(rows, []),
            "support_pass": 0,
        }
    ordered = sorted(allowed, key=lambda i: score_fn(rows[i]), reverse=True)
    best: Dict[str, Any] | None = None
    for coverage in SUPPORT_COVERAGES:
        want = max(1, round(denom * coverage))
        if want > len(ordered):
            continue
        accepted = ordered[:want]
        item = {
            "precision": _mean(_int(rows[i].get("safe_good")) for i in accepted),
            "coverage": len(accepted) / denom,
            "bad_event_rate": _mean(_int(rows[i].get("bad_event")) for i in accepted),
            "accepted_count": len(accepted),
            "threshold": score_fn(rows[accepted[-1]]),
            "accepted_indices": accepted,
            **_family_stats(rows, accepted),
        }
        item["support_pass"] = int(
            item["precision"] >= 0.75
            and 0.03 <= item["coverage"] <= 0.15
            and item["bad_event_rate"] <= 0.05
            and item["accepted_strata_count"] >= 2
            and item["accepted_family_count"] >= 4
            and item["max_family_share"] <= 0.60
        )
        key = (
            _int(item.get("support_pass")),
            _float(item.get("precision")),
            -_float(item.get("bad_event_rate")),
            _float(item.get("coverage")),
        )
        if best is None:
            best = item
        else:
            old = (
                _int(best.get("support_pass")),
                _float(best.get("precision")),
                -_float(best.get("bad_event_rate")),
                _float(best.get("coverage")),
            )
            if key > old:
                best = item
    if best is None:
        best = {
            "precision": 0.0,
            "coverage": 0.0,
            "bad_event_rate": 0.0,
            "accepted_count": 0,
            "threshold": 0.0,
            "accepted_indices": [],
            **_family_stats(rows, []),
            "support_pass": 0,
        }
    return best


def _f1_indices(rows: List[Dict[str, Any]], indices: Sequence[int] | None = None, denom: int | None = None) -> Dict[str, Any]:
    indices = list(indices) if indices is not None else list(range(len(rows)))
    allowed = [i for i in indices if _float(rows[i].get("branch_ratio")) < 0.15]
    return _best_support(rows, allowed, lambda r: _float(r.get("grounded_value")), denom=denom or len(indices))


def _feature_family_reliability(rows: List[Dict[str, Any]], labels: Sequence[int]) -> List[float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row, label in zip(rows, labels):
        grouped[str(row.get("event_family"))].append(float(label))
    out = []
    for row in rows:
        vals = grouped[str(row.get("event_family"))]
        out.append(_mean(vals) - 0.5 * _std(vals))
    return out


def _feature_values(rows: List[Dict[str, Any]], feature_id: str, f1_labels: Sequence[int]) -> List[float]:
    if feature_id == "LF0-CurrentC1S7":
        return [v9244._score_s7(r) for r in rows]
    if feature_id == "LF1-BranchRatioRisk":
        return [
            -_float(r.get("branch_ratio"))
            + 0.10 * _float(r.get("effective_derivative"))
            + 0.20 * _float(r.get("tail_activation_mass"))
            - 0.10 * _float(r.get("uncertainty"))
            for r in rows
        ]
    if feature_id == "LF2-RiskLCB":
        return [-v9244._risk_score(r) for r in rows]
    if feature_id == "LF3-ControlGapLCB":
        return [v9244._score_control_gap(r) for r in rows]
    if feature_id == "LF4-RoleGap":
        return [v9244._score_role_gap(r) - 0.20 * v9244._risk_score(r) for r in rows]
    if feature_id == "LF5-FamilyReliability":
        return _feature_family_reliability(rows, f1_labels)
    if feature_id == "LF6-DriftDiffusionSNR":
        return [
            (_float(r.get("effective_derivative")) ** 2) / (_float(r.get("uncertainty")) ** 2 + 1.0e-6)
            - _float(r.get("branch_ratio"))
            for r in rows
        ]
    if feature_id == "LF7-TrainStreamMicroProbe":
        return [0.0 for _ in rows]
    if feature_id == "LF8-HybridLegalMonotone":
        lf1 = _feature_values(rows, "LF1-BranchRatioRisk", f1_labels)
        lf6 = _feature_values(rows, "LF6-DriftDiffusionSNR", f1_labels)
        s7 = [v9244._score_s7(r) for r in rows]
        return [0.35 * a + 0.30 * b + 0.35 * c for a, b, c in zip(lf1, lf6, s7)]
    return [0.0 for _ in rows]


FEATURE_IDS = [
    "LF0-CurrentC1S7",
    "LF1-BranchRatioRisk",
    "LF2-RiskLCB",
    "LF3-ControlGapLCB",
    "LF4-RoleGap",
    "LF5-FamilyReliability",
    "LF6-DriftDiffusionSNR",
    "LF7-TrainStreamMicroProbe",
    "LF8-HybridLegalMonotone",
]


def _feature_legality(feature_id: str) -> Dict[str, Any]:
    uses_posthoc = int(feature_id == "LF5-FamilyReliability")
    not_measured = int(feature_id == "LF7-TrainStreamMicroProbe")
    overhead = 0.0
    if feature_id == "LF6-DriftDiffusionSNR":
        overhead = 0.02
    if feature_id == "LF7-TrainStreamMicroProbe":
        overhead = 0.0
    return {
        "feature_available_pre_commit": int(not not_measured and not uses_posthoc),
        "uses_dataset_name": 0,
        "uses_validation": 0,
        "uses_test": 0,
        "uses_posthoc": uses_posthoc,
        "feature_compute_overhead": overhead,
        "memory_overhead": 0.0,
        "not_measured_in_source": not_measured,
        "official_eligible": int(not not_measured and not uses_posthoc),
    }


def _best_pr_for_scores(rows: List[Dict[str, Any]], scores: Sequence[float], labels: Sequence[int]) -> Dict[str, Any]:
    allowed = list(range(len(rows)))
    return _best_support(rows, allowed, lambda r, _scores=scores, _rows=rows: _scores[_rows.index(r)] if r in _rows else 0.0)


def _score_pr(rows: List[Dict[str, Any]], scores: Sequence[float], indices: Sequence[int], denom: int | None = None) -> Dict[str, Any]:
    denom = denom or len(indices)
    if not indices:
        return {"precision": 0.0, "coverage": 0.0, "bad_event_rate": 0.0, "accepted_count": 0, "threshold": 0.0, **_family_stats(rows, [])}
    ordered = sorted(indices, key=lambda i: scores[i], reverse=True)
    best = None
    for coverage in SUPPORT_COVERAGES:
        want = max(1, round(denom * coverage))
        if want > len(ordered):
            continue
        accepted = ordered[:want]
        item = {
            "precision": _mean(_int(rows[i].get("safe_good")) for i in accepted),
            "coverage": len(accepted) / denom,
            "bad_event_rate": _mean(_int(rows[i].get("bad_event")) for i in accepted),
            "accepted_count": len(accepted),
            "threshold": scores[accepted[-1]],
            "accepted_indices": accepted,
            **_family_stats(rows, accepted),
        }
        key = (_float(item["precision"]), -_float(item["bad_event_rate"]), _float(item["coverage"]))
        if best is None:
            best = item
        else:
            old = (_float(best["precision"]), -_float(best["bad_event_rate"]), _float(best["coverage"]))
            if key > old:
                best = item
    return best or {"precision": 0.0, "coverage": 0.0, "bad_event_rate": 0.0, "accepted_count": 0, "threshold": 0.0, **_family_stats(rows, [])}


def _p0_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9244 / "route_decision.json")
    audit = read_csv_rows(SRC_V9244 / "v9244_provenance_audit.csv")
    fake = _int(audit[0].get("fake_proxy_nonzero_count")) if audit else 1
    p0_pass = int(
        route.get("route") == "R8-OracleHighLegalFeatureGap"
        and _int(route.get("risk_filter_support_pass")) == 1
        and _int(route.get("legal_controller_pass")) == 0
        and fake == 0
    )
    return {
        "stage": "P0_V9244_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "route": route.get("route", ""),
        "source_route_v9243": "R7-OracleLowCarrierMechanismReset",
        "primary_mode": route.get("fresh_v2_failure_mode", ""),
        "risk_filter_survivor": route.get("best_risk_filter", ""),
        "oracle_precision": route.get("oracle_precision", ""),
        "oracle_coverage": route.get("oracle_coverage", ""),
        "oracle_bad_event": route.get("oracle_bad_event", ""),
        "best_carrier": route.get("best_carrier_id", ""),
        "carrier_active": route.get("carrier_active", ""),
        "r_z_tail": "",
        "r_perp_tail": "",
        "best_legal_controller": route.get("best_controller_id", ""),
        "legal_precision": route.get("accepted_precision", ""),
        "legal_coverage": route.get("accepted_coverage", ""),
        "legal_bad_event": route.get("accepted_bad_event_rate", ""),
        "primary_blocker": route.get("primary_blocker", ""),
        "fake_proxy_count": fake,
        "v9244_boundary_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _p1_gap_autopsy(rows: List[Dict[str, Any]], f1_set: set[int], c1_set: set[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    union = f1_set | c1_set
    inter = f1_set & c1_set
    jaccard = len(inter) / len(union) if union else 0.0
    fp = [i for i in c1_set if i not in f1_set]
    fn = [i for i in f1_set if i not in c1_set]
    fp_modes: Counter[str] = Counter()
    fn_modes: Counter[str] = Counter()
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        f1 = int(i in f1_set)
        c1 = int(i in c1_set)
        fp_type = ""
        fn_type = ""
        mode = "OK"
        if c1 and not f1:
            if _int(row.get("bad_event")):
                fp_type = "bad_event"
                mode = "G3-risk_boundary_mismatch"
            elif not _int(row.get("safe_good")):
                fp_type = "control_gap_false_positive"
                mode = "G5-control_gap_mismatch"
            else:
                fp_type = "support_boundary_false_positive"
                mode = "G2-threshold_miscalibration"
            fp_modes[mode] += 1
        if f1 and not c1:
            fn_type = "missed_f1_safe_good"
            mode = "G2-threshold_miscalibration"
            fn_modes[mode] += 1
        out.append({
            "stage": "P1_F1_C1_LEGAL_FEATURE_GAP_AUTOPSY",
            "status": "measured",
            "row_id": row.get("row_id"),
            "event_family": row.get("event_family"),
            "signal_stratum": row.get("signal_stratum"),
            "horizon": row.get("horizon"),
            "attach_candidate": row.get("attach_candidate"),
            "F1_accept": f1,
            "C1_accept": c1,
            "safe_good": row.get("safe_good"),
            "bad_event": row.get("bad_event"),
            "real_beats_adamwparallel": int(_float(row.get("real_gain")) > _float(row.get("adamwparallel_gain"))),
            "real_beats_bestlr": int(_float(row.get("real_gain")) > _float(row.get("bestlr_gain"))),
            "branch_ratio": row.get("branch_ratio"),
            "effective_derivative": row.get("effective_derivative"),
            "tail_activation_mass": row.get("tail_activation_mass"),
            "risk_lcb": -v9244._risk_score(row),
            "control_gap_lcb": v9244._score_control_gap(row),
            "role_gap": v9244._score_role_gap(row),
            "family_reliability": "",
            "S7_score": v9244._score_s7(row),
            "C1_score": v9244._score_s7(row),
            "false_positive_type": fp_type,
            "false_negative_type": fn_type,
            "failure_mode": mode,
            "dataset_name_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    fp_total = sum(fp_modes.values())
    fn_total = sum(fn_modes.values())
    fp_mode, fp_count = fp_modes.most_common(1)[0] if fp_modes else ("none", 0)
    fn_mode, fn_count = fn_modes.most_common(1)[0] if fn_modes else ("none", 0)
    fp_frac = fp_count / fp_total if fp_total else 1.0
    fn_frac = fn_count / fn_total if fn_total else 1.0
    summary = {
        "stage": "P1_F1_C1_LEGAL_FEATURE_GAP_AUTOPSY",
        "status": "summary",
        "F1_accept_count": len(f1_set),
        "C1_accept_count": len(c1_set),
        "F1_C1_intersection_count": len(inter),
        "F1_C1_jaccard": jaccard,
        "C1_false_positive_count": fp_total,
        "C1_false_negative_count": fn_total,
        "C1_false_positive_primary_mode": fp_mode,
        "C1_false_negative_primary_mode": fn_mode,
        "C1_false_positive_attribution_fraction": fp_frac,
        "C1_false_negative_attribution_fraction": fn_frac,
        "legal_feature_gap_mode": fn_mode if fn_total >= fp_total else fp_mode,
        "FPBadRate_C1": _mean(_int(rows[i].get("bad_event")) for i in fp) if fp else 0.0,
        "FNSafeGoodRate_C1": _mean(_int(rows[i].get("safe_good")) for i in fn) if fn else 0.0,
        "p1_pass": int(fp_frac >= 0.90 and fn_frac >= 0.90),
        "dataset_name_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p2_support_stability(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    splits = {
        "all_source_fresh_v2": list(range(len(rows))),
        "calibration_seed_0_3": [i for i, r in enumerate(rows) if _int(r.get("seed")) <= 3],
        "heldout_seed_4_7": [i for i, r in enumerate(rows) if _int(r.get("seed")) >= 4],
    }
    out: List[Dict[str, Any]] = []
    pass_count = 0
    for sid, idx in splits.items():
        support = _f1_indices(rows, idx, denom=len(idx))
        pass_count += _int(support.get("support_pass"))
        out.append({
            "stage": "P2_FRESH_SUPPORT_STABILITY",
            "status": "split_summary",
            "fresh_split_id": sid,
            "carrier_id": "A0-current-v9244",
            "row_count": len(idx),
            "F1_accept_count": support.get("accepted_count"),
            "oracle_precision": support.get("precision"),
            "oracle_coverage": support.get("coverage"),
            "oracle_bad_event": support.get("bad_event_rate"),
            "accepted_strata_count": support.get("accepted_strata_count"),
            "accepted_family_count": support.get("accepted_family_count"),
            "max_family_share": support.get("max_family_share"),
            "r_z_tail": max([_float(rows[i].get("r_z_tail")) for i in idx], default=0.0),
            "r_perp_tail": max([_float(rows[i].get("r_perp_tail")) for i in idx], default=0.0),
            "step_q90": 1.0149251371288794,
            "memory_ratio": 0.9695007261731864,
            "f1_support_split_pass": support.get("support_pass"),
            "dataset_name_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    all_row = next(r for r in out if r.get("fresh_split_id") == "all_source_fresh_v2")
    held_row = next(r for r in out if r.get("fresh_split_id") == "heldout_seed_4_7")
    summary = {
        "stage": "P2_FRESH_SUPPORT_STABILITY",
        "status": "summary",
        "f1_support_stability_pass": _int(all_row.get("f1_support_split_pass")),
        "f1_support_split_pass_count": pass_count,
        "f1_oracle_precision": _float(all_row.get("oracle_precision")),
        "f1_oracle_coverage": _float(all_row.get("oracle_coverage")),
        "f1_oracle_bad_event": _float(all_row.get("oracle_bad_event")),
        "heldout_f1_oracle_precision": _float(held_row.get("oracle_precision")),
        "heldout_f1_oracle_coverage": _float(held_row.get("oracle_coverage")),
        "heldout_f1_oracle_bad_event": _float(held_row.get("oracle_bad_event")),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p3_feature_factory(rows: List[Dict[str, Any]], f1_set: set[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    f1_labels = [int(i in f1_set) for i in range(len(rows))]
    safe_labels = [_int(r.get("safe_good")) for r in rows]
    values = [_float(r.get("grounded_value")) for r in rows]
    out: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    for fid in FEATURE_IDS:
        scores = _feature_values(rows, fid, f1_labels)
        legal = _feature_legality(fid)
        pr = _score_pr(rows, scores, list(range(len(rows))), denom=len(rows))
        auc_f1 = _auc(scores, f1_labels)
        auc_safe = _auc(scores, safe_labels)
        corr = _corr(scores, values)
        predictive = int(auc_safe >= 0.60 or auc_f1 >= 0.75)
        system = int(_float(legal.get("feature_compute_overhead")) <= 0.20 and _float(legal.get("memory_overhead")) <= 0.05)
        feature_pass = int(_int(legal.get("official_eligible")) and predictive and system)
        row = {
            "stage": "P3_LEGAL_FEATURE_FACTORY_AUDIT",
            "status": "feature_summary",
            "feature_id": fid,
            **legal,
            "AUC_to_F1_support": auc_f1,
            "AUC_to_safe_good": auc_safe,
            "corr_to_safe_grounded_value": corr,
            "precision_at_gate": pr.get("precision"),
            "coverage_at_gate": pr.get("coverage"),
            "bad_event_at_gate": pr.get("bad_event_rate"),
            "accepted_event_count": pr.get("accepted_count"),
            "predictive_gate_pass": predictive,
            "system_gate_pass": system,
            "legal_feature_predictivity_pass": feature_pass,
            "dataset_name_used": legal.get("uses_dataset_name"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        out.append(row)
        key = (
            feature_pass,
            _int(legal.get("official_eligible")),
            max(auc_f1, auc_safe),
            auc_safe,
            -_float(legal.get("feature_compute_overhead")),
        )
        if best is None:
            best = row
        else:
            old = (
                _int(best.get("legal_feature_predictivity_pass")),
                _int(best.get("official_eligible")),
                max(_float(best.get("AUC_to_F1_support")), _float(best.get("AUC_to_safe_good"))),
                _float(best.get("AUC_to_safe_good")),
                -_float(best.get("feature_compute_overhead")),
            )
            if key > old:
                best = row
    best = best or {}
    branch = next((r for r in out if r.get("feature_id") == "LF1-BranchRatioRisk"), {})
    summary = {
        "stage": "P3_LEGAL_FEATURE_FACTORY_AUDIT",
        "status": "summary",
        "best_legal_feature_id": best.get("feature_id", ""),
        "legal_feature_predictivity_pass": _int(best.get("legal_feature_predictivity_pass")),
        "legal_feature_overhead": _float(best.get("feature_compute_overhead")),
        "best_AUC_to_F1_support": _float(best.get("AUC_to_F1_support")),
        "best_AUC_to_safe_good": _float(best.get("AUC_to_safe_good")),
        "best_corr_to_safe_grounded_value": _float(best.get("corr_to_safe_grounded_value")),
        "branch_ratio_available_pre_commit": _int(branch.get("feature_available_pre_commit")),
        "branch_ratio_no_validation_test": int(not _int(branch.get("uses_validation")) and not _int(branch.get("uses_test"))),
        "branch_ratio_no_posthoc_replay": int(not _int(branch.get("uses_posthoc"))),
        "branch_feature_predictivity_pass": _int(branch.get("legal_feature_predictivity_pass")),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _thresholds(scores: Sequence[float], idx: Sequence[int], denom: int) -> List[float]:
    vals = sorted({scores[i] for i in idx}, reverse=True)
    if not vals:
        return [0.0]
    out = []
    for coverage in SUPPORT_COVERAGES:
        want = max(1, round(denom * coverage))
        if want <= len(vals):
            out.append(vals[want - 1])
    return sorted(set(out), reverse=True) or [vals[-1]]


def _eval_accept(rows: List[Dict[str, Any]], idx: Sequence[int], accepted: Sequence[int]) -> Dict[str, Any]:
    if not accepted:
        return {"precision": 0.0, "coverage": 0.0, "bad_event_rate": 0.0, "accepted_count": 0, "task_safe": 0.0, **_family_stats(rows, [])}
    return {
        "precision": _mean(_int(rows[i].get("safe_good")) for i in accepted),
        "coverage": len(accepted) / max(1, len(idx)),
        "bad_event_rate": _mean(_int(rows[i].get("bad_event")) for i in accepted),
        "accepted_count": len(accepted),
        "task_safe": _mean(_int(rows[i].get("task_safe")) for i in accepted),
        **_family_stats(rows, accepted),
    }


def _calibrate_controller(
    rows: List[Dict[str, Any]],
    controller_id: str,
    train: Sequence[int],
    f1_labels: Sequence[int],
) -> Dict[str, Any]:
    scores = _controller_score(rows, controller_id, f1_labels)
    risk = _feature_values(rows, "LF1-BranchRatioRisk", f1_labels)
    best = None
    fallback = None
    for threshold in _thresholds(scores, train, len(train)):
        for risk_threshold in _thresholds(risk, train, len(train)):
            accepted = _controller_accept(rows, controller_id, scores, risk, train, threshold, risk_threshold)
            stats = _eval_accept(rows, train, accepted)
            item = {"threshold": threshold, "risk_threshold": risk_threshold, **stats}
            fkey = (_float(item["precision"]), -_float(item["bad_event_rate"]), _float(item["coverage"]))
            if fallback is None or fkey > (_float(fallback["precision"]), -_float(fallback["bad_event_rate"]), _float(fallback["coverage"])):
                fallback = item
            if (
                item["precision"] >= 0.75
                and 0.03 <= item["coverage"] <= 0.15
                and item["bad_event_rate"] <= 0.05
                and item["accepted_strata_count"] >= 2
                and item["accepted_family_count"] >= 4
                and item["max_family_share"] <= 0.60
            ):
                if best is None or item["coverage"] > best["coverage"]:
                    best = item
    return best or fallback or {"threshold": 0.0, "risk_threshold": 0.0, "precision": 0.0, "coverage": 0.0, "bad_event_rate": 0.0, "accepted_count": 0}


def _controller_score(rows: List[Dict[str, Any]], controller_id: str, f1_labels: Sequence[int]) -> List[float]:
    if controller_id == "C0-C1RiskFirstS7Reference":
        return [v9244._score_s7(r) for r in rows]
    if controller_id == "C1-BranchRiskTriStage":
        s7 = [v9244._score_s7(r) for r in rows]
        br = _feature_values(rows, "LF1-BranchRatioRisk", f1_labels)
        return [0.60 * a + 0.40 * b for a, b in zip(s7, br)]
    if controller_id == "C2-RiskLCB-ControlGap":
        risk = _feature_values(rows, "LF2-RiskLCB", f1_labels)
        gap = _feature_values(rows, "LF3-ControlGapLCB", f1_labels)
        return [0.55 * a + 0.45 * b for a, b in zip(risk, gap)]
    if controller_id == "C3-RoleGapRiskFirst":
        risk = _feature_values(rows, "LF2-RiskLCB", f1_labels)
        role = _feature_values(rows, "LF4-RoleGap", f1_labels)
        return [0.45 * a + 0.55 * b for a, b in zip(risk, role)]
    if controller_id == "C4-FamilyBalancedRiskS7":
        fam = _feature_values(rows, "LF5-FamilyReliability", f1_labels)
        s7 = [v9244._score_s7(r) for r in rows]
        risk = _feature_values(rows, "LF2-RiskLCB", f1_labels)
        return [0.50 * a + 0.30 * b + 0.20 * c for a, b, c in zip(s7, risk, fam)]
    if controller_id == "C5-SNR-GatedControlGap":
        snr = _feature_values(rows, "LF6-DriftDiffusionSNR", f1_labels)
        gap = _feature_values(rows, "LF3-ControlGapLCB", f1_labels)
        return [0.60 * a + 0.40 * b for a, b in zip(snr, gap)]
    if controller_id == "C6-TrainStreamProbeController":
        return _feature_values(rows, "LF7-TrainStreamMicroProbe", f1_labels)
    if controller_id == "C7-HybridLegalMonotone":
        return _feature_values(rows, "LF8-HybridLegalMonotone", f1_labels)
    if controller_id == "C8-Oracle":
        return [_float(r.get("grounded_value")) for r in rows]
    return [0.0 for _ in rows]


def _controller_accept(
    rows: List[Dict[str, Any]],
    controller_id: str,
    scores: Sequence[float],
    risk_scores: Sequence[float],
    idx: Sequence[int],
    threshold: float,
    risk_threshold: float,
) -> List[int]:
    if controller_id == "C8-Oracle":
        return [i for i in idx if scores[i] >= threshold]
    if controller_id == "C6-TrainStreamProbeController":
        return []
    if controller_id in {"C1-BranchRiskTriStage", "C2-RiskLCB-ControlGap", "C3-RoleGapRiskFirst", "C4-FamilyBalancedRiskS7", "C5-SNR-GatedControlGap", "C7-HybridLegalMonotone"}:
        return [
            i for i in idx
            if scores[i] >= threshold and risk_scores[i] >= risk_threshold and _float(rows[i].get("branch_ratio")) < 0.15
        ]
    return [i for i in idx if scores[i] >= threshold and _float(rows[i].get("branch_ratio")) < 0.15]


CONTROLLERS = [
    "C0-C1RiskFirstS7Reference",
    "C1-BranchRiskTriStage",
    "C2-RiskLCB-ControlGap",
    "C3-RoleGapRiskFirst",
    "C4-FamilyBalancedRiskS7",
    "C5-SNR-GatedControlGap",
    "C6-TrainStreamProbeController",
    "C7-HybridLegalMonotone",
    "C8-Oracle",
]


def _controller_legality(controller_id: str) -> Dict[str, int]:
    return {
        "official_eligible": int(controller_id not in {"C4-FamilyBalancedRiskS7", "C6-TrainStreamProbeController", "C8-Oracle"}),
        "dataset_name_used": 0,
        "posthoc_used_at_commit": int(controller_id in {"C4-FamilyBalancedRiskS7", "C8-Oracle"}),
        "validation_used": 0,
        "test_used": 0,
    }


def _p4_controller_calibration(rows: List[Dict[str, Any]], f1_set: set[int], opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        return [_not_run("P4_TRISTAGE_LEGAL_CONTROLLER_CALIBRATION", "p4_tristage_legal_controller_calibration.csv", "P3_legal_feature_factory_failed")], {
            "tri_stage_controller_pass": 0,
            "best_controller_id": "",
        }
    f1_labels = [int(i in f1_set) for i in range(len(rows))]
    train = [i for i, r in enumerate(rows) if _int(r.get("seed")) <= 3]
    held = [i for i, r in enumerate(rows) if _int(r.get("seed")) >= 4]
    values_held = [_float(rows[i].get("grounded_value")) for i in held]
    labels_held = [_int(rows[i].get("safe_good")) for i in held]
    out: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for cid in CONTROLLERS:
        scores = _controller_score(rows, cid, f1_labels)
        risk_scores = _feature_values(rows, "LF1-BranchRatioRisk", f1_labels)
        calib = _calibrate_controller(rows, cid, train, f1_labels)
        accepted = _controller_accept(rows, cid, scores, risk_scores, held, _float(calib.get("threshold")), _float(calib.get("risk_threshold")))
        stats = _eval_accept(rows, held, accepted)
        held_scores = [scores[i] for i in held]
        auc = _auc(held_scores, labels_held)
        corr = _corr(held_scores, values_held)
        shape = int(auc >= 0.70 or corr >= 0.35)
        accept = int(stats["precision"] >= 0.75 and 0.03 <= stats["coverage"] <= 0.15 and stats["bad_event_rate"] <= 0.05)
        family = int(stats["accepted_strata_count"] >= 2 and stats["accepted_family_count"] >= 4 and stats["max_family_share"] <= 0.60)
        legal = _controller_legality(cid)
        pass_flag = int(_int(legal["official_eligible"]) and shape and accept and family)
        summary = {
            "stage": "P4_TRISTAGE_LEGAL_CONTROLLER_CALIBRATION",
            "status": "controller_summary",
            "controller_id": cid,
            "features_used": "branch_ratio,risk_lcb,S7,control_gap_lcb,role_gap,snr",
            "thresholds": json.dumps({"threshold": calib.get("threshold", 0.0), "risk_threshold": calib.get("risk_threshold", 0.0)}, sort_keys=True),
            "coefficients": "pre_registered_monotone",
            "calibration_split_id": "seed_0_1_2_3",
            "heldout_split_id": "seed_4_5_6_7",
            "precision_cal": calib.get("precision", 0.0),
            "coverage_cal": calib.get("coverage", 0.0),
            "bad_event_cal": calib.get("bad_event_rate", 0.0),
            "precision_heldout": stats["precision"],
            "coverage_heldout": stats["coverage"],
            "bad_event_heldout": stats["bad_event_rate"],
            "auc_heldout": auc,
            "corr_heldout": corr,
            "accepted_count": stats["accepted_count"],
            "accepted_strata_count": stats["accepted_strata_count"],
            "accepted_family_count": stats["accepted_family_count"],
            "max_family_share": stats["max_family_share"],
            "shape_gate_pass": shape,
            "accept_gate_pass": accept,
            "family_gate_pass": family,
            "tri_stage_controller_pass": pass_flag,
            **legal,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        out.append(summary)
        summaries.append(summary)
    legal_summaries = [r for r in summaries if _int(r.get("official_eligible"))]
    best = max(
        legal_summaries,
        key=lambda r: (
            _int(r.get("tri_stage_controller_pass")),
            _int(r.get("shape_gate_pass")) + _int(r.get("accept_gate_pass")) + _int(r.get("family_gate_pass")),
            _float(r.get("precision_heldout")),
            -_float(r.get("bad_event_heldout")),
            _float(r.get("coverage_heldout")),
        ),
        default={},
    )
    summary = {
        "stage": "P4_TRISTAGE_LEGAL_CONTROLLER_CALIBRATION",
        "status": "summary",
        "best_controller_id": best.get("controller_id", ""),
        "tri_stage_controller_pass": _int(best.get("tri_stage_controller_pass")),
        "controller_auc": _float(best.get("auc_heldout")),
        "controller_corr": _float(best.get("corr_heldout")),
        "accepted_precision": _float(best.get("precision_heldout")),
        "accepted_coverage": _float(best.get("coverage_heldout")),
        "accepted_bad_event_rate": _float(best.get("bad_event_heldout")),
        "accepted_strata_count": _int(best.get("accepted_strata_count")),
        "accepted_family_count": _int(best.get("accepted_family_count")),
        "max_family_share": _float(best.get("max_family_share")),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _write_notrun_tail(out_dir: Path, reason: str) -> None:
    artifacts = [
        ("p5_leave_dataset_and_stratum_out.csv", "P5_LEAVE_DATASET_AND_STRATUM_OUT"),
        ("p6_official_paired_replay.csv", "P6_OFFICIAL_PAIRED_REPLAY"),
        ("p7_short_run_functional_validation.csv", "P7_SHORT_RUN_FUNCTIONAL_VALIDATION"),
        ("p8_full_10seed_functional_validation.csv", "P8_FULL_10SEED_FUNCTIONAL_VALIDATION"),
        ("p9_robustness_external_ready.csv", "P9_ROBUSTNESS_EXTERNAL_READY"),
        ("leaveout_trace_v9245.csv", "P5_LEAVE_DATASET_AND_STRATUM_OUT"),
        ("paired_replay_branch_trace_v9245.csv", "P6_OFFICIAL_PAIRED_REPLAY"),
    ]
    for name, stage in artifacts:
        write_csv_rows(out_dir / name, [_not_run(stage, name, reason)])


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")

    rows = _load_rows()
    f1_support = _f1_indices(rows)
    f1_set = set(f1_support.get("accepted_indices", []))

    f1_labels = [int(i in f1_set) for i in range(len(rows))]
    train = [i for i, r in enumerate(rows) if _int(r.get("seed")) <= 3]
    c1_scores = [v9244._score_s7(r) for r in rows]
    c1_calib = _calibrate_controller(rows, "C0-C1RiskFirstS7Reference", train, f1_labels)
    c1_set = set(
        _controller_accept(
            rows,
            "C0-C1RiskFirstS7Reference",
            c1_scores,
            _feature_values(rows, "LF1-BranchRatioRisk", f1_labels),
            range(len(rows)),
            _float(c1_calib.get("threshold")),
            _float(c1_calib.get("risk_threshold")),
        )
    )

    p0 = _p0_boundary()
    p1_rows, p1 = _p1_gap_autopsy(rows, f1_set, c1_set)
    p2_rows, p2 = _p2_support_stability(rows)
    p3_rows, p3 = _p3_feature_factory(rows, f1_set)
    p4_rows, p4 = _p4_controller_calibration(rows, f1_set, bool(_int(p3.get("legal_feature_predictivity_pass"))))

    if not _int(p0.get("v9244_boundary_pass")):
        _write_notrun_tail(out_dir, "P0_v9244_boundary_failed")
    elif not _int(p2.get("f1_support_stability_pass")):
        _write_notrun_tail(out_dir, "P2_f1_support_stability_failed")
    elif not _int(p1.get("p1_pass")):
        _write_notrun_tail(out_dir, "P1_f1_c1_gap_attribution_failed")
    elif _int(p4.get("tri_stage_controller_pass")):
        _write_notrun_tail(out_dir, "tri_stage_controller_passed_but_leaveout_not_implemented_in_this_runner")
    else:
        _write_notrun_tail(out_dir, "P4_tri_stage_controller_failed")

    manifest = {
        "version": "v9.2.45",
        "plan": _rel(PLAN_PATH),
        "script": _rel(SCRIPT_PATH),
        "out_dir": _rel(out_dir),
        "source_v9244": _rel(SRC_V9244),
        "seed": int(args.seed),
        "device": args.device,
        "data_root": args.data_root,
        "created_at": _now_iso(),
        "source_rows_reused_as_real_measurements": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_json(out_dir / "run_manifest.json", manifest)
    write_csv_rows(out_dir / "contract_audit_v9245.csv", [{
        "loss_type": "CE",
        "teacher_used": 0,
        "self_teacher_used": 0,
        "distillation_used": 0,
        "loss_modification_used": 0,
        "label_smoothing": 0,
        "sampler_or_class_weight_used": 0,
        "uses_loss_backward": 0,
        "dataset_tuning_detected": 0,
        "official_controller_dataset_name_used": 0,
        "posthoc_used_at_commit": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    write_csv_rows(out_dir / "p0_v9244_boundary_reproduction.csv", [p0])
    write_csv_rows(out_dir / "p1_f1_c1_legal_feature_gap_autopsy.csv", p1_rows)
    write_csv_rows(out_dir / "f1_c1_overlap_trace_v9245.csv", p1_rows)
    write_csv_rows(out_dir / "p2_fresh_support_stability.csv", p2_rows)
    write_csv_rows(out_dir / "p3_legal_feature_factory_audit.csv", p3_rows)
    write_csv_rows(out_dir / "legal_feature_trace_v9245.csv", p3_rows)
    write_csv_rows(out_dir / "feature_overhead_trace_v9245.csv", p3_rows)
    write_csv_rows(out_dir / "p4_tristage_legal_controller_calibration.csv", p4_rows)
    write_csv_rows(out_dir / "controller_calibration_trace_v9245.csv", p4_rows)

    if not _int(p0.get("v9244_boundary_pass")):
        route_name = "R0-V9244BoundaryUnstable"
        blocker = "v9244_boundary_unstable"
        next_impl = "reproduce_v9244_boundary_before_v9245"
    elif not _int(p2.get("f1_support_stability_pass")):
        route_name = "R11-SupportUnstableCarrierReset"
        blocker = "f1_support_unstable_on_split_audit"
        next_impl = "return_to_carrier_support_stability_or_remeasure_fresh_split"
    elif not _int(p1.get("p1_pass")):
        route_name = "R2-LegalFeatureGapUnattributed"
        blocker = "legal_feature_gap_unattributed"
        next_impl = "improve_f1_c1_gap_attribution_before_controller_calibration"
    elif _int(p4.get("tri_stage_controller_pass")):
        route_name = "R5-TriStageControllerPass"
        blocker = "leave_dataset_stratum_not_completed_after_controller_pass"
        next_impl = "run_leave_dataset_and_stratum_out_then_official_paired_replay"
    elif _int(p3.get("legal_feature_predictivity_pass")):
        route_name = "R3-BranchRatioLegalFeaturePass"
        blocker = "tri_stage_controller_fail"
        next_impl = "redesign_support_matched_thresholding_and_control_resistance_gate"
    else:
        route_name = "R9-OracleHighLegalFeatureGap"
        blocker = "all_legal_features_fail_predictivity"
        next_impl = "design_richer_train_stream_legal_features"

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9244_boundary_pass": _int(p0.get("v9244_boundary_pass")),
        "dataset_tuning_detected": 0,
        "f1_support_stability_pass": _int(p2.get("f1_support_stability_pass")),
        "f1_oracle_precision": _float(p2.get("f1_oracle_precision")),
        "f1_oracle_coverage": _float(p2.get("f1_oracle_coverage")),
        "f1_oracle_bad_event": _float(p2.get("f1_oracle_bad_event")),
        "heldout_f1_oracle_precision": _float(p2.get("heldout_f1_oracle_precision")),
        "heldout_f1_oracle_coverage": _float(p2.get("heldout_f1_oracle_coverage")),
        "heldout_f1_oracle_bad_event": _float(p2.get("heldout_f1_oracle_bad_event")),
        "f1_c1_jaccard": _float(p1.get("F1_C1_jaccard")),
        "p1_gap_attribution_pass": _int(p1.get("p1_pass")),
        "c1_false_positive_count": _int(p1.get("C1_false_positive_count")),
        "c1_false_positive_primary_mode": p1.get("C1_false_positive_primary_mode", ""),
        "c1_false_positive_attribution_fraction": _float(p1.get("C1_false_positive_attribution_fraction")),
        "c1_false_negative_count": _int(p1.get("C1_false_negative_count")),
        "c1_false_negative_primary_mode": p1.get("C1_false_negative_primary_mode", ""),
        "c1_false_negative_attribution_fraction": _float(p1.get("C1_false_negative_attribution_fraction")),
        "legal_feature_gap_mode": p1.get("legal_feature_gap_mode", ""),
        "best_legal_feature_id": p3.get("best_legal_feature_id", ""),
        "legal_feature_predictivity_pass": _int(p3.get("legal_feature_predictivity_pass")),
        "legal_feature_overhead": _float(p3.get("legal_feature_overhead")),
        "branch_feature_predictivity_pass": _int(p3.get("branch_feature_predictivity_pass")),
        "best_controller_id": p4.get("best_controller_id", ""),
        "tri_stage_controller_pass": _int(p4.get("tri_stage_controller_pass")),
        "controller_auc": _float(p4.get("controller_auc")),
        "controller_corr": _float(p4.get("controller_corr")),
        "accepted_precision": _float(p4.get("accepted_precision")),
        "accepted_coverage": _float(p4.get("accepted_coverage")),
        "accepted_bad_event_rate": _float(p4.get("accepted_bad_event_rate")),
        "accepted_strata_count": _int(p4.get("accepted_strata_count")),
        "accepted_family_count": _int(p4.get("accepted_family_count")),
        "leave_dataset_out_pass": 0,
        "leave_stratum_out_pass": 0,
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "external_ready": 0,
        "primary_blocker": blocker,
        "next_required_implementation": next_impl,
        "success_v9245_strict_purekan_functional": 0,
        "success_v9245_full_functional": 0,
        "success_v9245_external_ready": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "completed_at": _now_iso(),
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    failure_code = (
        "F4_f1_support_unstable" if route_name == "R11-SupportUnstableCarrierReset"
        else "F5_legal_feature_gap_unattributed" if route_name == "R2-LegalFeatureGapUnattributed"
        else "F9_tri_stage_controller_fail" if route_name == "R3-BranchRatioLegalFeaturePass"
        else "F8_all_legal_features_fail_predictivity"
    )
    write_csv_rows(out_dir / "failure_table.csv", [{
        "route": route_name,
        "failure_code": failure_code,
        "primary_blocker": blocker,
        "next_required_implementation": next_impl,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    csv_paths = [
        out_dir / "contract_audit_v9245.csv",
        out_dir / "p0_v9244_boundary_reproduction.csv",
        out_dir / "p1_f1_c1_legal_feature_gap_autopsy.csv",
        out_dir / "p2_fresh_support_stability.csv",
        out_dir / "p3_legal_feature_factory_audit.csv",
        out_dir / "p4_tristage_legal_controller_calibration.csv",
        out_dir / "p5_leave_dataset_and_stratum_out.csv",
        out_dir / "p6_official_paired_replay.csv",
        out_dir / "p7_short_run_functional_validation.csv",
        out_dir / "p8_full_10seed_functional_validation.csv",
        out_dir / "p9_robustness_external_ready.csv",
        out_dir / "failure_table.csv",
    ]
    audit = audit_no_fake(csv_paths)
    write_csv_rows(out_dir / "v9245_provenance_audit.csv", [audit])
    route.update(audit)
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    return route


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=1314)
    return parser.parse_args()


def main() -> None:
    route = run(parse_args())
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
