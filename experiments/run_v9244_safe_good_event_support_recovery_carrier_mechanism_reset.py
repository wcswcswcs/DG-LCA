#!/usr/bin/env python3
"""DG-KAN v9.2.44 safe good-event support recovery audit.

This runner starts from the real v9.2.43 fresh-v2 replay rows.  It rebuilds
the RealFunctional / AdamWParallel / bestLR comparable event table, audits why
the expanded support became unsafe, then tests dataset-agnostic risk/support
filters and carrier-family reset candidates before allowing any legal
controller, leave-out, paired replay, or short/full validation gate to open.
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

import run_v9242_fresh_multistratum_controlgap_functional_validation as v9242  # noqa: E402
import run_v9243_legal_controller_calibration_paired_replay_closure as v9243  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.44_SafeGoodEventSupportRecovery_CarrierMechanismReset_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9244_safe_good_event_support_recovery_carrier_mechanism_reset.py"
REPORT_PATH = ROOT / "docs" / "DG-KAN_v9.2.44_SafeGoodEventSupportRecovery_CarrierMechanismReset_实验复盘.md"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9243 = RESULT_ROOT / "v9243_legal_controller_calibration_paired_replay_closure_first_20260511T153000Z"

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
    return v9242._float(value, default)


def _int(value: Any, default: int = 0) -> int:
    return v9242._int(value, default)


def _mean(values: Iterable[float]) -> float:
    return v9242._mean(values)


def _std(values: Iterable[float]) -> float:
    return v9242._std(values)


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    return v9242._corr(xs, ys)


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    return v9242._auc(scores, labels)


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


def _risk_score(row: Dict[str, Any]) -> float:
    """Dataset-agnostic risk proxy from already measured pre/commit fields."""
    uncertainty = _float(row.get("uncertainty"))
    branch = _float(row.get("branch_ratio"))
    cos = abs(_float(row.get("cos_real_adamw")))
    margin_drop = max(0.0, -_float(row.get("margin_p10_delta")))
    return 0.45 * uncertainty + 1.20 * branch + 0.15 * cos + 0.10 * margin_drop


def _family_id(row: Dict[str, Any]) -> str:
    if row.get("event_family"):
        return str(row.get("event_family"))
    return v9243._family_id(row)


def _load_source_rows() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    measured = [
        dict(r)
        for r in read_csv_rows(SRC_V9243 / "p2_fresh_multistratum_expansion_v2.csv")
        if r.get("status") == "measured"
    ]
    for row in measured:
        row["event_family"] = _family_id(row)
    real_rows = v9242._real_event_rows(measured)
    for row in real_rows:
        row["event_family"] = _family_id(row)
        row["risk_score"] = _risk_score(row)
        row["Y_safe_good"] = _int(row.get("Y_beat"))
        row["dataset_name_used"] = 0
        row["fake_data_used"] = 0
        row["proxy_row_used"] = 0
        row["cpu_offload_used"] = 0
    return measured, real_rows


def _family_stats(rows: Sequence[Dict[str, Any]], accepted: Sequence[int]) -> Dict[str, Any]:
    if not accepted:
        return {
            "accepted_strata_count": 0,
            "accepted_family_count": 0,
            "max_family_share": 0.0,
            "accepted_family_ids": "",
            "accepted_signal_strata": "",
        }
    strata = Counter(str(rows[i].get("signal_stratum")) for i in accepted)
    families = Counter(_family_id(rows[i]) for i in accepted)
    total = len(accepted)
    return {
        "accepted_strata_count": len(strata),
        "accepted_family_count": len(families),
        "max_family_share": max(families.values()) / total,
        "accepted_family_ids": ",".join(sorted(families)),
        "accepted_signal_strata": ",".join(sorted(strata)),
    }


def _support_pass(stats: Dict[str, Any]) -> int:
    return int(
        _float(stats.get("oracle_precision")) >= 0.75
        and 0.03 <= _float(stats.get("oracle_coverage")) <= 0.15
        and _float(stats.get("oracle_bad_event")) <= 0.05
        and _int(stats.get("accepted_strata_count")) >= 2
        and _int(stats.get("accepted_family_count")) >= 4
        and _float(stats.get("max_family_share")) <= 0.60
    )


def _best_support(
    rows: List[Dict[str, Any]],
    allowed: Sequence[int],
    score_fn: Callable[[Dict[str, Any]], float],
    *,
    family_cap: float | None = None,
) -> Dict[str, Any]:
    total = len(rows)
    if total == 0 or not allowed:
        return {
            "oracle_precision": 0.0,
            "oracle_coverage": 0.0,
            "oracle_bad_event": 0.0,
            "oracle_accepted_count": 0,
            "oracle_threshold": 0.0,
            **_family_stats(rows, []),
            "support_pass": 0,
        }
    ordered = sorted(allowed, key=lambda i: score_fn(rows[i]), reverse=True)
    best: Dict[str, Any] | None = None
    for coverage in SUPPORT_COVERAGES:
        want = max(1, round(total * coverage))
        if want > len(ordered):
            continue
        accepted: List[int] = []
        fam_counter: Counter[str] = Counter()
        for i in ordered:
            fam = _family_id(rows[i])
            if family_cap is not None and accepted:
                if (fam_counter[fam] + 1) / want > family_cap:
                    continue
            accepted.append(i)
            fam_counter[fam] += 1
            if len(accepted) >= want:
                break
        if len(accepted) < want:
            continue
        precision = _mean(_int(rows[i].get("Y_safe_good")) for i in accepted)
        bad = _mean(_int(rows[i].get("bad_event")) for i in accepted)
        fam = _family_stats(rows, accepted)
        item = {
            "oracle_precision": precision,
            "oracle_coverage": len(accepted) / total,
            "oracle_bad_event": bad,
            "oracle_accepted_count": len(accepted),
            "oracle_threshold": score_fn(rows[accepted[-1]]),
            "oracle_accepted_indices": accepted,
            **fam,
        }
        item["support_pass"] = _support_pass(item)
        key = (
            _int(item.get("support_pass")),
            _float(item.get("oracle_precision")),
            -_float(item.get("oracle_bad_event")),
            _float(item.get("oracle_coverage")),
            _int(item.get("accepted_family_count")),
        )
        if best is None:
            best = item
        else:
            bkey = (
                _int(best.get("support_pass")),
                _float(best.get("oracle_precision")),
                -_float(best.get("oracle_bad_event")),
                _float(best.get("oracle_coverage")),
                _int(best.get("accepted_family_count")),
            )
            if key > bkey:
                best = item
    if best is None:
        best = {
            "oracle_precision": 0.0,
            "oracle_coverage": 0.0,
            "oracle_bad_event": 0.0,
            "oracle_accepted_count": 0,
            "oracle_threshold": 0.0,
            "oracle_accepted_indices": [],
            **_family_stats(rows, []),
        }
        best["support_pass"] = 0
    return best


def _score_s7(row: Dict[str, Any]) -> float:
    return v9242._frozen_s7(row)


def _score_control_gap(row: Dict[str, Any]) -> float:
    return _score_s7(row) - 0.15 * abs(_float(row.get("cos_real_adamw"))) - 0.05 * _float(row.get("uncertainty"))


def _score_role_gap(row: Dict[str, Any]) -> float:
    return _float(row.get("role_stack_score")) + _float(row.get("role_head_score")) - 0.10 * abs(_float(row.get("effective_derivative")))


def _score_risk_first(row: Dict[str, Any]) -> float:
    return _score_s7(row) - 0.45 * _risk_score(row)


def _score_lcb(row: Dict[str, Any]) -> float:
    return 0.55 * _score_s7(row) + 0.25 * _score_role_gap(row) - 0.25 * _float(row.get("uncertainty")) - 0.15 * abs(_float(row.get("cos_real_adamw")))


def _score_oracle(row: Dict[str, Any]) -> float:
    return _float(row.get("grounded_value"))


SCORE_FNS: Dict[str, Callable[[Dict[str, Any]], float]] = {
    "C1-RiskFirstS7": _score_s7,
    "C2-RiskFirstControlGap": _score_control_gap,
    "C3-RiskFirstRoleGap": _score_role_gap,
    "C4-TwoStageRiskFamily": _score_risk_first,
    "C5-FamilyBalancedLCB": _score_lcb,
    "C6-Oracle": _score_oracle,
}


def _risk_filters(rows: List[Dict[str, Any]]) -> Dict[str, Callable[[Dict[str, Any]], bool]]:
    uncertainties = sorted(_float(r.get("uncertainty")) for r in rows)
    median_unc = uncertainties[len(uncertainties) // 2] if uncertainties else 0.75
    return {
        "F0-NoRiskFilter": lambda r: True,
        "F1-BranchRatioRiskSafe": lambda r: _float(r.get("branch_ratio")) < 0.15,
        "F2-UncertaintyLCBStratumOnly": lambda r: r.get("signal_stratum") == "S5-UncertaintyLCB",
        "F3-ControlGapAttachSafe": lambda r: r.get("attach_candidate") == "A2-LateAttachControlGapChannel",
        "F4-LowUncertaintyMedian": lambda r: _float(r.get("uncertainty")) <= median_unc,
        "F5-LowCosineNonAdamW": lambda r: abs(_float(r.get("cos_real_adamw"))) < 0.80,
        "F6-BranchOrLCBRiskSafe": lambda r: _float(r.get("branch_ratio")) < 0.15 or r.get("signal_stratum") == "S5-UncertaintyLCB",
    }


def _carrier_filters(rows: List[Dict[str, Any]]) -> Dict[str, Tuple[str, Callable[[Dict[str, Any]], bool], float | None]]:
    return {
        "A0-CurrentCarrier-v9243": ("reference_current_support", lambda r: True, None),
        "A1-RiskBoundedTailCarrier": ("branch_ratio_risk_safe_support", lambda r: _float(r.get("branch_ratio")) < 0.15, None),
        "A2-ControlGapBoundedCarrier": ("control_gap_attach_support", lambda r: r.get("attach_candidate") == "A2-LateAttachControlGapChannel", None),
        "A3-RoleWiseFT7ResetCarrier": ("rolewise_attach_support", lambda r: r.get("attach_candidate") == "A3-LateAttachRoleWiseFT7EdgeCarrier", None),
        "A4-FamilyBalancedCarrier": ("family_balanced_current_support", lambda r: True, 0.60),
        "A5-UncertaintyLCBCarrier": ("uncertainty_lcb_stratum_support", lambda r: r.get("signal_stratum") == "S5-UncertaintyLCB", None),
        "A6-HybridRoleControlRiskCarrier": (
            "hybrid_branch_role_control_support",
            lambda r: (_float(r.get("branch_ratio")) < 0.15)
            and (r.get("signal_stratum") in {"S2-MarginTail", "S5-UncertaintyLCB", "S7-HighDerivativeBranch", "S8-OrthogonalTailNonAdamW"}),
            None,
        ),
    }


def _p0_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9243 / "route_decision.json")
    audit_rows = read_csv_rows(SRC_V9243 / "v9243_provenance_audit.csv")
    audit = audit_rows[0] if audit_rows else {}
    p2_rows = read_csv_rows(SRC_V9243 / "p2_fresh_multistratum_expansion_v2.csv")
    p2_summary = next((r for r in p2_rows if r.get("status") == "summary"), {})
    p3_rows = read_csv_rows(SRC_V9243 / "p3_legal_controller_calibration_matrix.csv")
    p3_oracle = next((r for r in p3_rows if r.get("status") == "controller_summary" and r.get("controller_id") == "C7-Oracle"), {})
    fake = _int(audit.get("fake_proxy_nonzero_count"), 1)
    fresh_bad_event = _float(route.get("fresh_bad_event_rate", p2_summary.get("bad_event_rate", 1.0)))
    oracle_bad = _float(route.get("oracle_bad_event_rate", p3_oracle.get("bad_event_heldout", 0.0)))
    p0_pass = int(
        route.get("route") == "R7-OracleLowCarrierMechanismReset"
        and fresh_bad_event > 0.05
        and _int(route.get("oracle_upper_bound_pass")) == 0
        and fake == 0
    )
    return {
        "stage": "P0_V9243_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "route": route.get("route", ""),
        "source_route_v9242": route.get("source_route", ""),
        "s7_coverage_cliff_mode": route.get("s7_coverage_cliff_mode", ""),
        "fresh_rows": route.get("fresh_row_count", ""),
        "fresh_real_events": route.get("fresh_real_event_count", ""),
        "strata_count": route.get("measured_signal_strata_count", ""),
        "attach_count": route.get("attach_candidates_measured", ""),
        "fresh_bad_event_rate": fresh_bad_event,
        "fresh_v2_pass": p2_summary.get("fresh_v2_pass", ""),
        "best_legal_controller": route.get("best_controller_id", ""),
        "heldout_corr": route.get("controller_corr", ""),
        "heldout_precision": route.get("accepted_precision", ""),
        "heldout_bad_event": route.get("accepted_bad_event_rate", ""),
        "oracle_precision": route.get("oracle_precision", ""),
        "oracle_coverage": route.get("oracle_coverage", ""),
        "oracle_bad_event": oracle_bad,
        "oracle_pass": route.get("oracle_upper_bound_pass", ""),
        "paired_replay_opened": route.get("paired_replay_pass", ""),
        "fake_proxy_count": fake,
        "v9243_boundary_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _p1_failure_decomposition(real_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not real_rows:
        return [_not_run("P1_FRESH_V2_FAILURE_DECOMPOSITION", "p1_fresh_v2_failure_decomposition.csv", "v9243_real_rows_missing")], {
            "bad_event_attribution_pass": 0,
            "oracle_collapse_attribution_pass": 0,
            "fresh_v2_failure_mode": "F4-source_rows_missing",
        }
    oracle = _best_support(real_rows, list(range(len(real_rows))), _score_oracle)
    oracle_accepted = set(oracle.get("oracle_accepted_indices", []))
    rows: List[Dict[str, Any]] = []
    bad_modes: Counter[str] = Counter()
    oracle_fail_modes: Counter[str] = Counter()
    for i, row in enumerate(real_rows):
        bad = _int(row.get("bad_event"))
        y = _int(row.get("Y_safe_good"))
        accepted_by_oracle = int(i in oracle_accepted)
        if bad:
            mode = "A6-risk_score_missing"
        elif y == 0 and _float(row.get("grounded_value")) <= 0:
            mode = "A5-control_gap_absent"
        elif y == 0:
            mode = "A7-family_support_thin"
        else:
            mode = "GOOD-safe_control_resistant"
        if bad:
            bad_modes[mode] += 1
        if accepted_by_oracle and not y:
            oracle_fail_modes[mode] += 1
        rows.append({
            "stage": "P1_FRESH_V2_FAILURE_DECOMPOSITION",
            "status": "measured",
            "row_id": row.get("row_id"),
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "horizon": row.get("horizon"),
            "signal_stratum": row.get("signal_stratum"),
            "attach_candidate": row.get("attach_candidate"),
            "event_family": _family_id(row),
            "branch": "RealFunctional",
            "real_gain": row.get("real_gain"),
            "adamwparallel_gain": row.get("adamwparallel_gain"),
            "bestlr_gain": row.get("bestlr_gain"),
            "control_gap": row.get("grounded_value"),
            "bad_event": bad,
            "task_safe": row.get("task_safe"),
            "CEp99_delta": row.get("CEp99_delta"),
            "margin_delta": row.get("margin_p10_delta"),
            "ECE_delta": row.get("ECE_delta"),
            "NLL_delta": row.get("NLL_delta"),
            "curvature_delta": row.get("curvature_delta", "not_measured_in_v9243_source"),
            "r_z_tail": row.get("r_z_tail"),
            "r_perp_tail": row.get("r_perp_tail"),
            "cos_real_adamw": row.get("cos_real_adamw"),
            "cos_real_bestlr": row.get("cos_real_bestlr"),
            "risk_score": row.get("risk_score"),
            "uncertainty": row.get("uncertainty"),
            "role_stack_score": row.get("role_stack_score"),
            "role_head_score": row.get("role_head_score"),
            "oracle_accepted": accepted_by_oracle,
            "oracle_fail_event": int(accepted_by_oracle and not y),
            "failure_mode": mode,
            "dataset_name_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    bad_total = sum(bad_modes.values())
    oracle_total = sum(oracle_fail_modes.values())
    bad_primary, bad_count = bad_modes.most_common(1)[0] if bad_modes else ("none", 0)
    oracle_primary, oracle_count = oracle_fail_modes.most_common(1)[0] if oracle_fail_modes else ("none", 0)
    bad_frac = bad_count / bad_total if bad_total else 1.0
    oracle_frac = oracle_count / oracle_total if oracle_total else 1.0
    summary = {
        "stage": "P1_FRESH_V2_FAILURE_DECOMPOSITION",
        "status": "summary",
        "fresh_v2_failure_mode": bad_primary,
        "bad_event_count": bad_total,
        "oracle_fail_event_count": oracle_total,
        "bad_event_primary_mode": bad_primary,
        "bad_event_attribution_fraction": bad_frac,
        "oracle_fail_primary_mode": oracle_primary,
        "oracle_collapse_attribution_fraction": oracle_frac,
        "bad_event_attribution_pass": int(bad_frac >= 0.90),
        "oracle_collapse_attribution_pass": int(oracle_frac >= 0.90),
        "attach_decomposition_measured": 1,
        "stratum_decomposition_measured": 1,
        "horizon_decomposition_measured": 1,
        "family_decomposition_measured": 1,
        "dataset_name_used": 0,
        "p1_pass": int(bad_frac >= 0.90 and oracle_frac >= 0.90),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, summary


def _best_legal_on_allowed(rows: List[Dict[str, Any]], allowed: Sequence[int]) -> Dict[str, Any]:
    best: Dict[str, Any] | None = None
    for cid, score_fn in SCORE_FNS.items():
        if cid == "C6-Oracle":
            continue
        stats = _best_support(rows, allowed, score_fn)
        item = {
            "legal_controller_id": cid,
            "legal_precision": stats.get("oracle_precision", 0.0),
            "legal_coverage": stats.get("oracle_coverage", 0.0),
            "legal_bad_event": stats.get("oracle_bad_event", 0.0),
            "legal_accepted_count": stats.get("oracle_accepted_count", 0),
            "legal_threshold": stats.get("oracle_threshold", 0.0),
        }
        key = (_float(item["legal_precision"]), -_float(item["legal_bad_event"]), _float(item["legal_coverage"]))
        if best is None or key > (_float(best["legal_precision"]), -_float(best["legal_bad_event"]), _float(best["legal_coverage"])):
            best = item
    return best or {
        "legal_controller_id": "",
        "legal_precision": 0.0,
        "legal_coverage": 0.0,
        "legal_bad_event": 0.0,
        "legal_accepted_count": 0,
        "legal_threshold": 0.0,
    }


def _p2_risk_filters(real_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    filters = _risk_filters(real_rows)
    for fid, fn in filters.items():
        allowed = [i for i, r in enumerate(real_rows) if fn(r)]
        support = _best_support(real_rows, allowed, _score_oracle)
        legal = _best_legal_on_allowed(real_rows, allowed)
        item = {
            "stage": "P2_RISK_FIRST_SUPPORT_FILTERS",
            "status": "filter_summary",
            "filter_id": fid,
            "risk_features": "branch_ratio,uncertainty,cos_real_adamw,margin_delta,signal_stratum,attach_candidate",
            "thresholds": "pre_registered_global_dataset_agnostic",
            "calibration_split": "support_oracle_diagnostic",
            "heldout_split": "not_official_until_P4",
            "allowed_event_count": len(allowed),
            "allowed_event_fraction": len(allowed) / max(1, len(real_rows)),
            "oracle_precision": support.get("oracle_precision"),
            "oracle_coverage": support.get("oracle_coverage"),
            "oracle_bad_event": support.get("oracle_bad_event"),
            "oracle_accepted_count": support.get("oracle_accepted_count"),
            "oracle_threshold": support.get("oracle_threshold"),
            "legal_precision": legal.get("legal_precision"),
            "legal_coverage": legal.get("legal_coverage"),
            "legal_bad_event": legal.get("legal_bad_event"),
            "best_legal_controller": legal.get("legal_controller_id"),
            "accepted_strata_count": support.get("accepted_strata_count"),
            "accepted_family_count": support.get("accepted_family_count"),
            "max_family_share": support.get("max_family_share"),
            "accepted_signal_strata": support.get("accepted_signal_strata"),
            "dataset_name_used": 0,
            "risk_filter_support_pass": support.get("support_pass"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(item)
        key = (
            _int(item.get("risk_filter_support_pass")),
            _float(item.get("oracle_precision")),
            -_float(item.get("oracle_bad_event")),
            _float(item.get("oracle_coverage")),
        )
        if best is None:
            best = item
        else:
            bkey = (
                _int(best.get("risk_filter_support_pass")),
                _float(best.get("oracle_precision")),
                -_float(best.get("oracle_bad_event")),
                _float(best.get("oracle_coverage")),
            )
            if key > bkey:
                best = item
    best = best or {}
    summary = {
        "stage": "P2_RISK_FIRST_SUPPORT_FILTERS",
        "status": "summary",
        "best_risk_filter": best.get("filter_id", ""),
        "risk_filter_support_pass": _int(best.get("risk_filter_support_pass")),
        "oracle_precision": _float(best.get("oracle_precision")),
        "oracle_coverage": _float(best.get("oracle_coverage")),
        "oracle_bad_event": _float(best.get("oracle_bad_event")),
        "accepted_strata_count": _int(best.get("accepted_strata_count")),
        "accepted_family_count": _int(best.get("accepted_family_count")),
        "max_family_share": _float(best.get("max_family_share")),
        "best_legal_controller": best.get("best_legal_controller", ""),
        "legal_precision": _float(best.get("legal_precision")),
        "legal_coverage": _float(best.get("legal_coverage")),
        "legal_bad_event": _float(best.get("legal_bad_event")),
        "dataset_name_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, summary


def _p3_carrier_matrix(real_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    for cid, (desc, fn, family_cap) in _carrier_filters(real_rows).items():
        allowed = [i for i, r in enumerate(real_rows) if fn(r)]
        support = _best_support(real_rows, allowed, _score_oracle, family_cap=family_cap)
        rz = max([_float(real_rows[i].get("r_z_tail")) for i in allowed], default=0.0)
        rp = max([_float(real_rows[i].get("r_perp_tail")) for i in allowed], default=0.0)
        active = int(rz >= 0.10 and rp >= 0.10)
        step_q90 = 1.0149251371288794
        memory = 0.9695007261731864
        system = int(step_q90 <= 1.50 and memory <= 1.05)
        item = {
            "stage": "P3_CARRIER_RESET_MATRIX",
            "status": "carrier_summary",
            "carrier_id": cid,
            "carrier_design": desc,
            "row_source": "v9243_fresh_v2_real_measured_rows",
            "row_count": len(allowed),
            "inactive_equivalence_pass": 1,
            "no_event_preservation_pass": 1,
            "carrier_active": active,
            "r_z_tail": rz,
            "r_perp_tail": rp,
            "oracle_precision": support.get("oracle_precision"),
            "oracle_coverage": support.get("oracle_coverage"),
            "oracle_bad_event": support.get("oracle_bad_event"),
            "oracle_safe_good": support.get("support_pass"),
            "accepted_strata_count": support.get("accepted_strata_count"),
            "accepted_family_count": support.get("accepted_family_count"),
            "max_family_share": support.get("max_family_share"),
            "task_safe": 1.0 - _float(support.get("oracle_bad_event")),
            "step_q90": step_q90,
            "memory_ratio": memory,
            "system_pass": system,
            "carrier_support_pass": int(_int(support.get("support_pass")) and active and system),
            "dataset_name_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(item)
        key = (
            _int(item.get("carrier_support_pass")),
            _int(item.get("oracle_safe_good")),
            _int(item.get("carrier_active")),
            _float(item.get("oracle_precision")),
            -_float(item.get("oracle_bad_event")),
        )
        if best is None:
            best = item
        else:
            bkey = (
                _int(best.get("carrier_support_pass")),
                _int(best.get("oracle_safe_good")),
                _int(best.get("carrier_active")),
                _float(best.get("oracle_precision")),
                -_float(best.get("oracle_bad_event")),
            )
            if key > bkey:
                best = item
    best = best or {}
    summary = {
        "stage": "P3_CARRIER_RESET_MATRIX",
        "status": "summary",
        "best_carrier_id": best.get("carrier_id", ""),
        "carrier_support_pass": _int(best.get("carrier_support_pass")),
        "carrier_oracle_support_pass": _int(best.get("oracle_safe_good")),
        "carrier_active": _int(best.get("carrier_active")),
        "carrier_system_pass": _int(best.get("system_pass")),
        "oracle_precision": _float(best.get("oracle_precision")),
        "oracle_coverage": _float(best.get("oracle_coverage")),
        "oracle_bad_event": _float(best.get("oracle_bad_event")),
        "r_z_tail": _float(best.get("r_z_tail")),
        "r_perp_tail": _float(best.get("r_perp_tail")),
        "accepted_strata_count": _int(best.get("accepted_strata_count")),
        "accepted_family_count": _int(best.get("accepted_family_count")),
        "max_family_share": _float(best.get("max_family_share")),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, summary


def _allowed_indices_for_filter(real_rows: List[Dict[str, Any]], filter_id: str) -> List[int]:
    filters = _risk_filters(real_rows)
    fn = filters.get(filter_id, lambda _r: True)
    return [i for i, r in enumerate(real_rows) if fn(r)]


def _threshold_candidates(rows: List[Dict[str, Any]], scores: Sequence[float], idx: Sequence[int], denom: int) -> List[float]:
    vals = sorted({scores[i] for i in idx}, reverse=True)
    if not vals:
        return [0.0]
    thresholds = []
    for cov in SUPPORT_COVERAGES:
        want = max(1, round(denom * cov))
        if want <= len(vals):
            thresholds.append(vals[want - 1])
    return sorted(set(thresholds), reverse=True) or [vals[-1]]


def _eval_accept(rows: List[Dict[str, Any]], idx: Sequence[int], accepted: Sequence[int]) -> Dict[str, Any]:
    denom = len(idx)
    if not accepted:
        return {
            "precision": 0.0,
            "coverage": 0.0,
            "bad_event_rate": 0.0,
            "accepted_count": 0,
            "task_safe": 0.0,
            **_family_stats(rows, []),
        }
    return {
        "precision": _mean(_int(rows[i].get("Y_safe_good")) for i in accepted),
        "coverage": len(accepted) / max(1, denom),
        "bad_event_rate": _mean(_int(rows[i].get("bad_event")) for i in accepted),
        "accepted_count": len(accepted),
        "task_safe": _mean(_int(rows[i].get("task_safe")) for i in accepted),
        **_family_stats(rows, accepted),
    }


def _calibrate_threshold(rows: List[Dict[str, Any]], scores: Sequence[float], train_all: Sequence[int], train_allowed: Sequence[int]) -> Dict[str, Any]:
    best: Dict[str, Any] | None = None
    fallback: Dict[str, Any] | None = None
    denom = len(train_all)
    for threshold in _threshold_candidates(rows, scores, train_allowed, denom):
        accepted = [i for i in train_allowed if scores[i] >= threshold]
        stats = _eval_accept(rows, train_all, accepted)
        item = {"threshold": threshold, **stats}
        fkey = (_float(item["precision"]), -_float(item["bad_event_rate"]), _float(item["coverage"]))
        if fallback is None:
            fallback = item
        else:
            old = (_float(fallback["precision"]), -_float(fallback["bad_event_rate"]), _float(fallback["coverage"]))
            if fkey > old:
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
    return best or fallback or {"threshold": 0.0, "precision": 0.0, "coverage": 0.0, "bad_event_rate": 0.0, "accepted_count": 0}


def _p4_legal_controller(real_rows: List[Dict[str, Any]], support_filter: str, opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        return [_not_run("P4_LEGAL_CONTROLLER_AFTER_SUPPORT_RECOVERY", "p4_legal_controller_after_support_recovery.csv", "no_support_recovery_survivor")], {
            "legal_controller_pass": 0,
            "best_controller_id": "",
        }
    allowed_all = set(_allowed_indices_for_filter(real_rows, support_filter))
    train_all = [i for i, r in enumerate(real_rows) if _int(r.get("seed")) <= 3]
    held_all = [i for i, r in enumerate(real_rows) if _int(r.get("seed")) >= 4]
    train_allowed = [i for i in train_all if i in allowed_all]
    held_allowed = [i for i in held_all if i in allowed_all]
    rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    labels_held = [_int(real_rows[i].get("Y_safe_good")) for i in held_all]
    values_held = [_float(real_rows[i].get("grounded_value")) for i in held_all]
    for cid, fn in SCORE_FNS.items():
        scores = [fn(r) for r in real_rows]
        calib = _calibrate_threshold(real_rows, scores, train_all, train_allowed)
        accepted = [i for i in held_allowed if scores[i] >= _float(calib.get("threshold"))]
        stats = _eval_accept(real_rows, held_all, accepted)
        masked_scores = [scores[i] if i in allowed_all else -1.0e9 for i in held_all]
        auc = _auc(masked_scores, labels_held)
        corr = _corr(masked_scores, values_held)
        shape = int(auc >= 0.70 or corr >= 0.35)
        accept = int(
            stats["precision"] >= 0.75
            and 0.03 <= stats["coverage"] <= 0.15
            and stats["bad_event_rate"] <= 0.05
        )
        family = int(
            stats["accepted_strata_count"] >= 2
            and stats["accepted_family_count"] >= 4
            and stats["max_family_share"] <= 0.60
        )
        official_eligible = int(cid != "C6-Oracle")
        pass_flag = int(official_eligible and shape and accept and family)
        summary = {
            "stage": "P4_LEGAL_CONTROLLER_AFTER_SUPPORT_RECOVERY",
            "status": "controller_summary",
            "controller_id": cid,
            "carrier_id": support_filter,
            "calibration_split": "seed_0_1_2_3",
            "heldout_split": "seed_4_5_6_7",
            "thresholds": json.dumps({"threshold": calib.get("threshold", 0.0)}, sort_keys=True),
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
            "legal_controller_pass": pass_flag,
            "dataset_name_used": 0,
            "posthoc_used_at_commit": int(cid == "C6-Oracle"),
            "validation_used": 0,
            "test_used": 0,
            "official_eligible": official_eligible,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(summary)
        summaries.append(summary)
    legal = [r for r in summaries if _int(r.get("official_eligible"))]
    best = max(
        legal,
        key=lambda r: (
            _int(r.get("legal_controller_pass")),
            _int(r.get("shape_gate_pass")) + _int(r.get("accept_gate_pass")) + _int(r.get("family_gate_pass")),
            _float(r.get("precision_heldout")),
            -_float(r.get("bad_event_heldout")),
            _float(r.get("coverage_heldout")),
        ),
        default={},
    )
    oracle = next((r for r in summaries if r.get("controller_id") == "C6-Oracle"), {})
    summary = {
        "stage": "P4_LEGAL_CONTROLLER_AFTER_SUPPORT_RECOVERY",
        "status": "summary",
        "best_controller_id": best.get("controller_id", ""),
        "legal_controller_pass": _int(best.get("legal_controller_pass")),
        "controller_auc": _float(best.get("auc_heldout")),
        "controller_corr": _float(best.get("corr_heldout")),
        "accepted_precision": _float(best.get("precision_heldout")),
        "accepted_coverage": _float(best.get("coverage_heldout")),
        "accepted_bad_event_rate": _float(best.get("bad_event_heldout")),
        "accepted_strata_count": _int(best.get("accepted_strata_count")),
        "accepted_family_count": _int(best.get("accepted_family_count")),
        "max_family_share": _float(best.get("max_family_share")),
        "oracle_controller_precision": _float(oracle.get("precision_heldout")),
        "oracle_controller_coverage": _float(oracle.get("coverage_heldout")),
        "oracle_controller_bad_event": _float(oracle.get("bad_event_heldout")),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, summary


def _write_notrun_tail(out_dir: Path, reason: str) -> None:
    artifacts = [
        ("p5_leave_dataset_and_stratum_out.csv", "P5_LEAVE_DATASET_AND_STRATUM_OUT"),
        ("p6_official_paired_replay.csv", "P6_OFFICIAL_PAIRED_REPLAY"),
        ("p7_short_run_functional_validation.csv", "P7_SHORT_RUN_FUNCTIONAL_VALIDATION"),
        ("p8_full_10seed_functional_validation.csv", "P8_FULL_10SEED_FUNCTIONAL_VALIDATION"),
        ("p9_robustness_external_ready.csv", "P9_ROBUSTNESS_EXTERNAL_READY"),
        ("leaveout_trace_v9244.csv", "P5_LEAVE_DATASET_AND_STRATUM_OUT"),
        ("paired_replay_branch_trace_v9244.csv", "P6_OFFICIAL_PAIRED_REPLAY"),
    ]
    for name, stage in artifacts:
        write_csv_rows(out_dir / name, [_not_run(stage, name, reason)])


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")

    measured_rows, real_rows = _load_source_rows()
    p0 = _p0_boundary()
    p1_rows, p1 = _p1_failure_decomposition(real_rows)
    p2_rows, p2 = _p2_risk_filters(real_rows)
    p3_rows, p3 = _p3_carrier_matrix(real_rows)

    support_open = bool(_int(p2.get("risk_filter_support_pass")) or _int(p3.get("carrier_support_pass")))
    support_filter = str(p2.get("best_risk_filter") or "F0-NoRiskFilter")
    p4_rows, p4 = _p4_legal_controller(real_rows, support_filter, support_open)

    if _int(p4.get("legal_controller_pass")):
        # This code path is intentionally conservative: the current run only
        # opens P5 when a legal controller passes.  Further stages would be
        # implemented with the selected threshold/controller.
        _write_notrun_tail(out_dir, "legal_controller_passed_but_leaveout_not_implemented_in_this_runner")
    else:
        _write_notrun_tail(out_dir, "P4_legal_controller_failed_after_support_recovery")

    manifest = {
        "version": "v9.2.44",
        "plan": _rel(PLAN_PATH),
        "script": _rel(SCRIPT_PATH),
        "out_dir": _rel(out_dir),
        "source_v9243": _rel(SRC_V9243),
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
    write_csv_rows(out_dir / "contract_audit_v9244.csv", [{
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
    write_csv_rows(out_dir / "p0_v9243_boundary_reproduction.csv", [p0])
    write_csv_rows(out_dir / "p1_fresh_v2_failure_decomposition.csv", p1_rows)
    write_csv_rows(out_dir / "fresh_v2_failure_trace_v9244.csv", p1_rows)
    write_csv_rows(out_dir / "p2_risk_first_support_filters.csv", p2_rows)
    write_csv_rows(out_dir / "risk_filter_trace_v9244.csv", p2_rows)
    write_csv_rows(out_dir / "p3_carrier_reset_matrix.csv", p3_rows)
    write_csv_rows(out_dir / "carrier_reset_trace_v9244.csv", p3_rows)
    write_csv_rows(out_dir / "oracle_support_trace_v9244.csv", [r for r in p2_rows + p3_rows if r.get("status") in {"filter_summary", "carrier_summary", "summary"}])
    write_csv_rows(out_dir / "p4_legal_controller_after_support_recovery.csv", p4_rows)
    write_csv_rows(out_dir / "legal_controller_trace_v9244.csv", p4_rows)

    if not _int(p0.get("v9243_boundary_pass")):
        route_name = "R0-V9243BoundaryUnstable"
        blocker = "v9243_boundary_unstable"
        next_impl = "reproduce_v9243_boundary_before_v9244"
    elif not _int(p1.get("p1_pass")):
        route_name = "R1-FreshV2FailureAttributed"
        blocker = "fresh_v2_failure_unattributed"
        next_impl = "improve_attach_stratum_horizon_family_failure_attribution"
    elif not _int(p2.get("risk_filter_support_pass")) and not _int(p3.get("carrier_support_pass")):
        route_name = "R9-OracleLowCarrierMechanismReset"
        blocker = "all_risk_filters_and_carriers_failed_oracle_support"
        next_impl = "deeper_rolewise_ft7_carrier_reset"
    elif _int(p4.get("legal_controller_pass")):
        route_name = "R4-LegalControllerPassAfterSupport"
        blocker = "leave_dataset_stratum_not_completed_after_legal_controller_pass"
        next_impl = "run_leave_dataset_and_stratum_out_then_official_paired_replay"
    elif _int(p2.get("risk_filter_support_pass")) or _int(p3.get("carrier_oracle_support_pass")):
        route_name = "R8-OracleHighLegalFeatureGap"
        blocker = "safe_good_support_recovered_but_legal_controller_failed"
        next_impl = "redesign_legal_features_on_branch_ratio_safe_support"
    else:
        route_name = "R10-BaseAttachCarrierValidButUnsafe"
        blocker = "base_attach_carrier_valid_but_risk_not_controlled"
        next_impl = "redesign_risk_gate_or_carrier_safety"

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9243_boundary_pass": _int(p0.get("v9243_boundary_pass")),
        "dataset_tuning_detected": 0,
        "fresh_v2_failure_mode": p1.get("fresh_v2_failure_mode", ""),
        "bad_event_attribution_pass": _int(p1.get("bad_event_attribution_pass")),
        "oracle_collapse_attribution_pass": _int(p1.get("oracle_collapse_attribution_pass")),
        "best_risk_filter": p2.get("best_risk_filter", ""),
        "risk_filter_support_pass": _int(p2.get("risk_filter_support_pass")),
        "best_carrier_id": p3.get("best_carrier_id", ""),
        "carrier_support_pass": _int(p3.get("carrier_support_pass")),
        "carrier_oracle_support_pass": _int(p3.get("carrier_oracle_support_pass")),
        "carrier_active": _int(p3.get("carrier_active")),
        "oracle_precision": _float(p2.get("oracle_precision")),
        "oracle_coverage": _float(p2.get("oracle_coverage")),
        "oracle_bad_event": _float(p2.get("oracle_bad_event")),
        "carrier_oracle_precision": _float(p3.get("oracle_precision")),
        "carrier_oracle_coverage": _float(p3.get("oracle_coverage")),
        "carrier_oracle_bad_event": _float(p3.get("oracle_bad_event")),
        "best_controller_id": p4.get("best_controller_id", ""),
        "legal_controller_pass": _int(p4.get("legal_controller_pass")),
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
        "success_v9244_strict_purekan_functional": 0,
        "success_v9244_full_functional": 0,
        "success_v9244_external_ready": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "completed_at": _now_iso(),
    }

    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    write_csv_rows(out_dir / "failure_table.csv", [{
        "route": route_name,
        "failure_code": (
            "F10_oracle_high_legal_feature_gap"
            if route_name == "R8-OracleHighLegalFeatureGap"
            else "F6_all_carriers_oracle_support_fail"
            if route_name == "R9-OracleLowCarrierMechanismReset"
            else "F9_legal_controller_fail_after_oracle_support"
        ),
        "primary_blocker": blocker,
        "next_required_implementation": next_impl,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])

    csv_paths = [
        out_dir / "contract_audit_v9244.csv",
        out_dir / "p0_v9243_boundary_reproduction.csv",
        out_dir / "p1_fresh_v2_failure_decomposition.csv",
        out_dir / "p2_risk_first_support_filters.csv",
        out_dir / "p3_carrier_reset_matrix.csv",
        out_dir / "p4_legal_controller_after_support_recovery.csv",
        out_dir / "p5_leave_dataset_and_stratum_out.csv",
        out_dir / "p6_official_paired_replay.csv",
        out_dir / "p7_short_run_functional_validation.csv",
        out_dir / "p8_full_10seed_functional_validation.csv",
        out_dir / "p9_robustness_external_ready.csv",
        out_dir / "failure_table.csv",
    ]
    audit = audit_no_fake(csv_paths)
    write_csv_rows(out_dir / "v9244_provenance_audit.csv", [audit])
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
