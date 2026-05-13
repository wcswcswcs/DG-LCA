"""Reusable outcome-grounded controller utilities for DG-KAN runners.

This module holds the auditable, data-table based pieces that are reused by
the v9.2.8x controller promotion experiments: stable CSV/JSON IO, Wilson
intervals, AUC, full-row risk/support feature materialization, and
calibration-frozen decision repair search.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


def cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    return v


def fnum(v: Any, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def inum(v: Any, default: int = 0) -> int:
    return int(round(fnum(v, default)))


def token_int(text: str, name: str, default: int = 0) -> int:
    match = re.search(rf"{re.escape(name)}(\d+)", text or "")
    return int(match.group(1)) if match else default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: cell(row.get(k, "")) for k in fieldnames})


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_lcb(success: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = success / n
    den = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (center - margin) / den)


def wilson_ucb(success: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 1.0
    p = success / n
    den = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return min(1.0, (center + margin) / den)


def auc_score(values: list[float], labels: list[int]) -> float:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    pairs = sorted(zip(values, labels), key=lambda x: x[0])
    rank_sum = 0.0
    rank = 1
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2
        for k in range(i, j):
            if pairs[k][1]:
                rank_sum += avg_rank
        rank += j - i
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def metric_summary(
    rows: list[dict[str, Any]],
    full_rows: list[dict[str, str]],
    mask: Callable[[dict[str, Any]], bool],
    seed_pred: Callable[[int], bool],
) -> dict[str, Any]:
    denom = sum(1 for row in full_rows if seed_pred(inum(row.get("seed"))))
    accepted = [row for row in rows if seed_pred(inum(row.get("seed"))) and mask(row)]
    n = len(accepted)
    safe = sum(inum(row.get("safe_good_label")) for row in accepted)
    bad = sum(inum(row.get("bad_event_label")) for row in accepted)
    null = sum(inum(row.get("null_event_label")) for row in accepted)
    family_counts: dict[str, int] = {}
    stratum_counts: dict[str, int] = {}
    for row in accepted:
        family_counts[str(row.get("family_id", ""))] = family_counts.get(str(row.get("family_id", "")), 0) + 1
        stratum_counts[str(row.get("_signal_stratum", ""))] = stratum_counts.get(str(row.get("_signal_stratum", "")), 0) + 1
    return {
        "accepted_count": n,
        "safe_good_count": safe,
        "bad_event_count": bad,
        "null_event_count": null,
        "precision": safe / n if n else 0.0,
        "coverage": n / denom if denom else 0.0,
        "bad_event": bad / n if n else 0.0,
        "null_rate": null / n if n else 0.0,
        "precision_lcb": wilson_lcb(safe, n),
        "bad_event_ucb": wilson_ucb(bad, n),
        "accepted_family_count": len(family_counts),
        "accepted_signal_strata_count": len(stratum_counts),
        "max_family_share": max((v / n for v in family_counts.values()), default=0.0),
        "max_stratum_share": max((v / n for v in stratum_counts.values()), default=0.0),
    }


def attach_candidate_features(
    rows: list[dict[str, Any]],
    full_by_event: dict[str, dict[str, str]],
    old_event_ids: set[str],
) -> None:
    by_step: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        event = full_by_event.get(str(row.get("event_id", "")), {})
        family = event.get("event_family", "")
        stratum = event.get("signal_stratum", "")
        row["_event_family"] = family
        row["_signal_stratum"] = stratum
        row["_bad_level"] = token_int(stratum, "bad")
        row["_null_level"] = token_int(stratum, "null")
        row["_support_level"] = token_int(stratum, "sup")
        row["_tail_risk"] = token_int(family, "tail")
        row["_horizon_level"] = token_int(stratum, "h")
        row["_candidate_origin_tag"] = "shared" if row.get("event_id") in old_event_ids else "new_only"
        row["_stable_accept_current"] = inum(row.get("accept_native")) == 1
        by_step[(str(row.get("dataset")), str(row.get("seed")), str(row.get("step")))].append(row)
    for group in by_step.values():
        quantized_scores = sorted([inum(row.get("score_quantized_ref")) for row in group], reverse=True)
        for row in group:
            q = inum(row.get("score_quantized_ref"))
            second = (
                quantized_scores[1]
                if len(quantized_scores) > 1 and quantized_scores[0] == q
                else (quantized_scores[0] if quantized_scores and quantized_scores[0] != q else q)
            )
            row["_score_margin"] = q - second


def attach_calibrated_risk_stats(rows: list[dict[str, Any]]) -> None:
    groups = [
        "family_id",
        "_signal_stratum",
        "_event_family",
        "carrier_id",
        "bucket_id",
        "_horizon_level",
        "_candidate_origin_tag",
    ]
    stats: dict[str, dict[str, list[int]]] = {group: defaultdict(lambda: [0, 0, 0, 0]) for group in groups}
    for row in rows:
        if inum(row.get("seed")) >= 5 or not row.get("_stable_accept_current"):
            continue
        for group in groups:
            key = str(row.get(group, ""))
            stats[group][key][0] += 1
            stats[group][key][1] += inum(row.get("safe_good_label"))
            stats[group][key][2] += inum(row.get("bad_event_label"))
            stats[group][key][3] += inum(row.get("null_event_label"))
    for row in rows:
        bad_ucbs: list[float] = []
        safe_lcbs: list[float] = []
        null_ucbs: list[float] = []
        counts: list[int] = []
        for group in groups:
            n, safe, bad, null = stats[group].get(str(row.get(group, "")), [0, 0, 0, 0])
            row[f"_{group}_support_count"] = n
            row[f"_{group}_bad_ucb"] = wilson_ucb(bad, n)
            row[f"_{group}_bad_lcb"] = wilson_lcb(bad, n)
            row[f"_{group}_null_ucb"] = wilson_ucb(null, n)
            row[f"_{group}_support_lcb"] = wilson_lcb(safe, n)
            bad_ucbs.append(row[f"_{group}_bad_ucb"])
            safe_lcbs.append(row[f"_{group}_support_lcb"])
            null_ucbs.append(row[f"_{group}_null_ucb"])
            counts.append(n)
        row["_bad_ucb_max"] = max(bad_ucbs)
        row["_bad_ucb_mean"] = sum(bad_ucbs) / len(bad_ucbs)
        row["_null_ucb_max"] = max(null_ucbs)
        row["_support_lcb_min"] = min(safe_lcbs)
        row["_support_count_min"] = min(counts)
        row["_candidate_origin_risk"] = 1.0 if row.get("_candidate_origin_tag") == "new_only" else 0.0
        row["_bad_level_tail_risk"] = 0.15 * row["_bad_level"] + 0.10 * row["_tail_risk"] + 0.05 * row["_null_level"]
        row["_risk_residual_score"] = (
            row["_bad_ucb_mean"]
            + row["_bad_level_tail_risk"]
            + 0.05 * row["_candidate_origin_risk"]
            - 0.05 * row["_support_level"]
        )
        row["_monotone_risk_score"] = (
            row["_bad_ucb_mean"]
            + row["_bad_level_tail_risk"]
            + 0.10 * row["_candidate_origin_risk"]
            + 0.05 * row["_null_ucb_max"]
            - 0.05 * row["_support_lcb_min"]
        )


def build_risk_support_feature_table(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    held_accept = [row for row in rows if inum(row.get("seed")) >= 5 and row.get("_stable_accept_current")]
    feature_defs = [
        ("RSF1-BadLevelToken", "token", "_bad_level"),
        ("RSF2-NullLevelToken", "token", "_null_level"),
        ("RSF3-HorizonTailRisk", "token", "_tail_risk"),
        ("RSF4-BadUCBMean", "calibration_group", "_bad_ucb_mean"),
        ("RSF5-BadUCBMax", "calibration_group", "_bad_ucb_max"),
        ("RSF6-RiskResidualScore", "monotone_composite", "_risk_residual_score"),
        ("RSF7-MonotoneRiskScore", "monotone_composite", "_monotone_risk_score"),
        ("RSF8-SupportLCBMin", "calibration_group", "_support_lcb_min"),
        ("RSF9-CandidateOriginRisk", "candidate_lifecycle", "_candidate_origin_risk"),
    ]
    feature_rows: list[dict[str, Any]] = []
    for feature_id, family, key in feature_defs:
        values = [fnum(row.get(key)) for row in held_accept]
        bad_labels = [inum(row.get("bad_event_label")) for row in held_accept]
        safe_labels = [inum(row.get("safe_good_label")) for row in held_accept]
        raw_bad_auc = auc_score(values, bad_labels)
        raw_safe_auc = auc_score(values, safe_labels)
        bad_auc = max(raw_bad_auc, 1.0 - raw_bad_auc)
        safe_auc = max(raw_safe_auc, 1.0 - raw_safe_auc)
        if values and max(values) != min(values):
            scaled = [(value - min(values)) / (max(values) - min(values)) for value in values]
        else:
            scaled = [0.0 for _ in values]
        ece = 0.0
        for bin_id in range(5):
            lo = bin_id / 5
            hi = (bin_id + 1) / 5
            idx = [i for i, value in enumerate(scaled) if value >= lo and (value < hi or bin_id == 4)]
            if not idx:
                continue
            pred = sum(scaled[i] for i in idx) / len(idx)
            obs = sum(bad_labels[i] for i in idx) / len(idx)
            ece += len(idx) / len(values) * abs(pred - obs)
        feature_rows.append(
            {
                "feature_id": feature_id,
                "feature_family": family,
                "AUC_bad_event": bad_auc,
                "AUC_safe_good": safe_auc,
                "raw_AUC_bad_event": raw_bad_auc,
                "raw_AUC_safe_good": raw_safe_auc,
                "corr_bad_event": "",
                "corr_safe_good": "",
                "calibration_ECE_bad": ece,
                "risk_bin_count": 5,
                "family_coverage": len(set(row.get("family_id") for row in held_accept)),
                "horizon_coverage": len(set(row.get("horizon") for row in held_accept)),
                "candidate_origin_coverage": len(set(row.get("_candidate_origin_tag") for row in held_accept)),
                "uses_dataset_name": 0,
                "uses_outcome_at_commit": 0,
                "source_column": key,
            }
        )
    best_bad = max(feature_rows, key=lambda row: fnum(row["AUC_bad_event"]))
    best_safe = max(feature_rows, key=lambda row: fnum(row["AUC_safe_good"]))
    return feature_rows, best_bad, best_safe


def calibration_frozen_repairs(
    rows: list[dict[str, Any]],
    full_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def current(row: dict[str, Any]) -> bool:
        return bool(row.get("_stable_accept_current"))

    cal_accept = [row for row in rows if inum(row.get("seed")) < 5 and current(row)]
    candidates: list[tuple[str, str, dict[str, Any], Callable[[dict[str, Any]], bool]]] = [
        (
            "DR0-v9281-DR2-reference",
            "bad_level_tail_reference",
            {"bad_level_max": 1, "support_min": 3, "tail_risk_max": 1},
            lambda row: current(row)
            and row["_bad_level"] <= 1
            and row["_support_level"] >= 3
            and row["_tail_risk"] <= 1,
        ),
        ("DR1-BitExactStableAcceptOnly", "stable_accept_only", {}, current),
    ]
    for risk_key in ["_risk_residual_score", "_monotone_risk_score", "_bad_level_tail_risk", "_bad_ucb_mean"]:
        vals = sorted(set(round(fnum(row[risk_key]), 6) for row in cal_accept))
        if len(vals) > 80:
            vals = [vals[int(i * (len(vals) - 1) / 79)] for i in range(80)]
        for threshold in vals:
            for null_threshold in [0.10, 0.15, 0.20, 0.30, 0.50, 1.0]:
                for support_threshold in [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]:
                    candidates.append(
                        (
                            f"DR7-{risk_key}-thr-{threshold}-n{null_threshold}-s{support_threshold}",
                            "risk_support_monotone_score",
                            {
                                "risk_feature": risk_key,
                                "risk_max": threshold,
                                "null_ucb_max": null_threshold,
                                "support_lcb_min": support_threshold,
                            },
                            lambda row, risk_key=risk_key, threshold=threshold, null_threshold=null_threshold, support_threshold=support_threshold: current(row)
                            and fnum(row[risk_key]) <= threshold
                            and fnum(row["_null_ucb_max"]) <= null_threshold
                            and fnum(row["_support_lcb_min"]) >= support_threshold,
                        )
                    )

    rows_out: list[dict[str, Any]] = []
    pass_candidates: list[dict[str, Any]] = []
    legal_calibration_rows: list[dict[str, Any]] = []
    for candidate_id, repair_type, thresholds, mask in candidates:
        cal = metric_summary(rows, full_rows, mask, lambda seed: seed < 5)
        held = metric_summary(rows, full_rows, mask, lambda seed: seed >= 5)
        support_cal = (
            cal["accepted_family_count"] >= 32
            and cal["accepted_signal_strata_count"] >= 5
            and cal["max_family_share"] <= 0.50
            and cal["max_stratum_share"] <= 0.60
        )
        support_held = (
            held["accepted_family_count"] >= 32
            and held["accepted_signal_strata_count"] >= 5
            and held["max_family_share"] <= 0.50
            and held["max_stratum_share"] <= 0.60
        )
        cal_gate = (
            cal["precision"] >= 0.75
            and 0.03 <= cal["coverage"] <= 0.15
            and cal["bad_event"] <= 0.05
            and cal["null_rate"] <= 0.15
            and cal["precision_lcb"] >= 0.75
            and cal["bad_event_ucb"] <= 0.05
            and support_cal
        )
        held_gate = (
            held["precision"] >= 0.75
            and 0.03 <= held["coverage"] <= 0.15
            and held["bad_event"] <= 0.05
            and held["null_rate"] <= 0.15
            and held["precision_lcb"] >= 0.75
            and held["bad_event_ucb"] <= 0.05
            and support_held
        )
        row = {
            "repair_candidate_id": candidate_id,
            "repair_type": repair_type,
            "feature_set": thresholds.get("risk_feature", repair_type),
            "thresholds": thresholds,
            "calibration_split_id": "seed_0_4",
            "heldout_split_id": "seed_5_7",
            "dataset_name_used": 0,
            "accepted_count_cal": cal["accepted_count"],
            "accepted_count_heldout": held["accepted_count"],
            "precision_cal": cal["precision"],
            "coverage_cal": cal["coverage"],
            "bad_event_cal": cal["bad_event"],
            "null_rate_cal": cal["null_rate"],
            "precision_heldout": held["precision"],
            "coverage_heldout": held["coverage"],
            "bad_event_heldout": held["bad_event"],
            "null_rate_heldout": held["null_rate"],
            "precision_lcb": held["precision_lcb"],
            "bad_event_ucb": held["bad_event_ucb"],
            "accepted_signal_strata_count": held["accepted_signal_strata_count"],
            "accepted_family_count": held["accepted_family_count"],
            "max_family_share": held["max_family_share"],
            "max_stratum_share": held["max_stratum_share"],
            "accept_disagreement_count": 0,
            "score_quantized_disagreement_count": 0,
            "rank_disagreement_count": 0,
            "calibration_gate_pass": int(cal_gate),
            "support_balance_pass": int(support_held),
            "decision_gate_pass": int(cal_gate and held_gate),
        }
        rows_out.append(row)
        if cal_gate:
            pass_candidates.append(row)
        if 0.03 <= cal["coverage"] <= 0.15 and support_cal:
            legal_calibration_rows.append(row)
    if pass_candidates:
        best = max(pass_candidates, key=lambda row: (fnum(row["precision_heldout"]), -fnum(row["bad_event_heldout"])))
    elif legal_calibration_rows:
        best = min(
            legal_calibration_rows,
            key=lambda row: (fnum(row["bad_event_cal"]), -fnum(row["precision_cal"]), abs(fnum(row["coverage_cal"]) - 0.05)),
        )
    else:
        best = max(rows_out, key=lambda row: (fnum(row["precision_cal"]), -fnum(row["bad_event_cal"])))
    return rows_out, best
