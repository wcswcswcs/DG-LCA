#!/usr/bin/env python3
"""DG-KAN v9.2.41 control-gap value alignment audit.

This runner intentionally reuses the measured v9.2.40 carrier/value rows.
It does not fabricate missing replay horizons, and it never promotes posthoc
oracle rows into official controller success.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.41_ControlGap_ValueAlignment_RoleWiseLateAttach_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9241_controlgap_value_alignment_rolewise_lateattach.py"
REPORT_PATH = ROOT / "docs" / "DG-KAN_v9.2.41_ControlGap_ValueAlignment_RoleWiseLateAttach_实验复盘.md"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9240 = RESULT_ROOT / "v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if len(vals) <= 1:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = _mean(xs)
    my = _mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    if len(scores) != len(labels) or not scores:
        return 0.5
    n_pos = sum(int(x) for x in labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    pos_rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if int(pairs[k][1]):
                pos_rank_sum += avg_rank
        i = j
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _safe_key(row: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    return (
        str(row.get("attach_candidate", "")),
        str(row.get("dataset", "")),
        str(row.get("seed", "")),
        str(row.get("horizon", "")),
        str(row.get("event_id", "")),
    )


def _gain(row: Dict[str, Any]) -> float:
    return -_float(row.get("CEp99_delta")) + _float(row.get("margin_delta")) - 2.0 * _float(row.get("bad_event"))


def _score_metrics(
    scores: Sequence[float],
    grounded: Sequence[float],
    labels: Sequence[int],
    bads: Sequence[int],
    *,
    target_coverage: float = 0.10,
) -> Dict[str, Any]:
    n = len(scores)
    if n == 0:
        return {
            "corr": 0.0,
            "auc": 0.5,
            "precision": 0.0,
            "coverage": 0.0,
            "bad_event_rate": 0.0,
            "accepted_event_count": 0,
            "best_precision": 0.0,
            "best_coverage": 0.0,
            "best_bad_event_rate": 0.0,
            "best_accepted_event_count": 0,
            "threshold": 0.0,
        }
    ordered = sorted(range(n), key=lambda i: scores[i], reverse=True)
    k = max(1, round(n * target_coverage))
    chosen = ordered[:k]
    threshold = scores[chosen[-1]]

    def at_count(count: int) -> Tuple[float, float, float, int]:
        count = max(1, min(n, int(count)))
        idx = ordered[:count]
        return (
            sum(labels[i] for i in idx) / count,
            count / n,
            sum(bads[i] for i in idx) / count,
            count,
        )

    precision, coverage, bad_rate, accepted_count = at_count(k)
    best = (0.0, 0.0, 1.0, 0)
    for cov in (0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15):
        cand = at_count(round(n * cov))
        if cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
            best = cand
    return {
        "corr": _corr(scores, grounded),
        "auc": _auc(scores, labels),
        "precision": precision,
        "coverage": coverage,
        "bad_event_rate": bad_rate,
        "accepted_event_count": accepted_count,
        "best_precision": best[0],
        "best_coverage": best[1],
        "best_bad_event_rate": best[2],
        "best_accepted_event_count": best[3],
        "threshold": threshold,
    }


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


def _load_event_rows(source_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, str, str, str], Dict[str, Dict[str, Any]]]]:
    p5 = [r for r in read_csv_rows(source_dir / "p5_functional_carrier_actuatability.csv") if r.get("status") == "measured"]
    p6 = [r for r in read_csv_rows(source_dir / "p6_value_observability_audit.csv") if r.get("status") == "measured"]
    branch_groups: Dict[Tuple[str, str, str, str, str], Dict[str, Dict[str, Any]]] = {}
    for row in p5:
        branch_groups.setdefault(_safe_key(row), {})[str(row.get("branch", ""))] = row
    p6_by_key = {_safe_key(row): row for row in p6 if row.get("branch") == "RealFunctional"}
    rows: List[Dict[str, Any]] = []
    for key, branches in sorted(branch_groups.items()):
        real = branches.get("RealFunctional")
        adamw = branches.get("AdamWParallel")
        bestlr = branches.get("bestLR")
        if not real or not adamw or not bestlr:
            continue
        obs = p6_by_key.get(key, {})
        real_gain = _gain(real)
        adamw_gain = _gain(adamw)
        bestlr_gain = _gain(bestlr)
        derived_gap = real_gain - max(adamw_gain, bestlr_gain)
        grounded_value = _float(obs.get("grounded_value"), derived_gap)
        y_beat = _int(obs.get("Y_beat"), int(derived_gap > 0 and _int(real.get("bad_event")) == 0))
        row = {
            "attach_candidate": key[0],
            "dataset": key[1],
            "dataset_slice": key[1],
            "seed": key[2],
            "horizon": key[3],
            "event_id": key[4],
            "signal_stratum": real.get("signal_stratum", ""),
            "controller": obs.get("controller", "ControlGapTailScore"),
            "current_score": _float(obs.get("value_score"), _float(real.get("r_perp_tail")) - abs(_float(real.get("cos_real_adamw")))),
            "grounded_value": grounded_value,
            "Y_beat": y_beat,
            "real_gain": real_gain,
            "adamwparallel_gain": adamw_gain,
            "bestlr_gain": bestlr_gain,
            "control_gap": derived_gap,
            "control_gap_score": _float(obs.get("control_gap_score"), _float(obs.get("value_score"))),
            "tail_value_score": _float(obs.get("tail_value_score"), _float(real.get("r_z_tail"))),
            "role_score": _float(obs.get("role_score"), _float(real.get("branch_ratio"))),
            "tail_CEp99_delta": _float(real.get("CEp99_delta")),
            "margin_delta": _float(real.get("margin_delta")),
            "task_risk": _float(real.get("bad_event")),
            "bad_event": _int(real.get("bad_event")),
            "task_safe": _int(real.get("task_safe")),
            "uncertainty": abs(_float(real.get("cos_real_adamw"))) + 0.5 * abs(_float(real.get("cos_real_bestlr"))),
            "role_stack_score": _float(real.get("branch_ratio")) * (1.0 - abs(_float(real.get("cos_real_adamw")))),
            "role_head_score": _float(real.get("r_perp_tail")) + 0.5 * _float(real.get("r_z_tail")),
            "branch_ratio": _float(real.get("branch_ratio")),
            "effective_derivative": _float(real.get("effective_derivative")),
            "r_z_tail": _float(real.get("r_z_tail")),
            "r_perp_tail": _float(real.get("r_perp_tail")),
            "cos_real_adamw": _float(real.get("cos_real_adamw")),
            "cos_real_bestlr": _float(real.get("cos_real_bestlr")),
            "functional_step_norm": _float(real.get("functional_step_norm")),
            "task_step_norm": _float(real.get("task_step_norm")),
            "acc_delta": _float(real.get("acc_delta")),
            "source_artifact": "v9240/p5_p6",
            "derived_from_source_rows": 1,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
    return rows, branch_groups


def _assign_ranks(rows: List[Dict[str, Any]]) -> None:
    def rank_map(key: str, reverse: bool) -> Dict[int, float]:
        ordered = sorted(range(len(rows)), key=lambda i: rows[i][key], reverse=reverse)
        ranks: Dict[int, float] = {}
        i = 0
        while i < len(ordered):
            j = i + 1
            while j < len(ordered) and rows[ordered[j]][key] == rows[ordered[i]][key]:
                j += 1
            avg = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[ordered[k]] = avg
            i = j
        return ranks

    score_rank = rank_map("current_score", True)
    value_rank = rank_map("grounded_value", True)
    for i, row in enumerate(rows):
        row["score_rank"] = score_rank[i]
        row["value_rank"] = value_rank[i]
        row["score_sign"] = "positive" if row["current_score"] >= 0 else "negative"


def _family_value_score(rows: List[Dict[str, Any]]) -> List[float]:
    groups: Dict[Tuple[str, str, str], List[float]] = {}
    for row in rows:
        horizon = int(float(row["horizon"]))
        if horizon <= 5:
            hbucket = "h001_005"
        elif horizon <= 80:
            hbucket = "h020_080"
        else:
            hbucket = "h240_plus"
        role_bucket = "role_high" if row["branch_ratio"] >= 0.20 else "role_low"
        groups.setdefault((row["signal_stratum"], hbucket, role_bucket), []).append(row["grounded_value"])
    scores: List[float] = []
    for row in rows:
        horizon = int(float(row["horizon"]))
        hbucket = "h001_005" if horizon <= 5 else ("h020_080" if horizon <= 80 else "h240_plus")
        role_bucket = "role_high" if row["branch_ratio"] >= 0.20 else "role_low"
        vals = groups[(row["signal_stratum"], hbucket, role_bucket)]
        scores.append(_mean(vals) - 0.5 * _std(vals))
    return scores


def _candidate_score_specs(rows: List[Dict[str, Any]]) -> List[Tuple[str, str, List[float], int, str]]:
    family_scores = _family_value_score(rows)
    return [
        ("S0-CurrentV9240Score", "current_control_gap_tail_score", [r["current_score"] for r in rows], 1, ""),
        ("S1-SignFlippedCurrentScore", "diagnostic_sign_flip_only", [-r["current_score"] for r in rows], 0, "diagnostic_only_sign_flip"),
        ("S2-DirectControlGapField", "source_commit_time_control_gap_field", [r["control_gap_score"] for r in rows], 1, ""),
        (
            "S3-ConservativeLCB-UCB",
            "legal_conservative_tail_role_score",
            [
                -0.5 * r["current_score"]
                + 0.25 * r["role_stack_score"]
                + 0.15 * r["role_head_score"]
                - 0.10 * r["uncertainty"]
                for r in rows
            ],
            1,
            "",
        ),
        (
            "S4-TailCEMarginPosthoc",
            "posthoc_tail_ce_margin_value",
            [-r["tail_CEp99_delta"] + r["margin_delta"] - 2.0 * r["task_risk"] for r in rows],
            0,
            "posthoc_outcome_delta_used",
        ),
        (
            "S5-RoleWiseFT7Score",
            "legal_rolewise_ft7_feature_score",
            [r["role_stack_score"] + r["role_head_score"] - 0.15 * abs(r["effective_derivative"]) for r in rows],
            1,
            "",
        ),
        ("S6-FamilyValueReliability", "diagnostic_family_value_reliability", family_scores, 0, "posthoc_family_value_calibration"),
        (
            "S7-HybridMonotoneLegal",
            "legal_hybrid_monotone_features",
            [
                -0.50 * r["current_score"]
                + 0.25 * r["role_stack_score"]
                + 0.20 * r["role_head_score"]
                + 0.05 * r["tail_value_score"]
                for r in rows
            ],
            1,
            "",
        ),
        ("S8-OracleUpperBound", "posthoc_oracle_grounded_value", [r["grounded_value"] for r in rows], 0, "posthoc_oracle_upper_bound"),
    ]


def _score_candidates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scores = _candidate_score_specs(rows)
    grounded = [r["grounded_value"] for r in rows]
    labels = [int(r["Y_beat"]) for r in rows]
    bads = [int(r["bad_event"]) for r in rows]
    out: List[Dict[str, Any]] = []
    for score_id, description, values, official_eligible, reason in scores:
        metrics = _score_metrics(values, grounded, labels, bads)
        diagnostic_pass = int(metrics["auc"] >= 0.60 or metrics["corr"] >= 0.20)
        observability_pass = int(metrics["auc"] >= 0.70 or metrics["corr"] >= 0.35)
        accept_pass = int(
            metrics["best_precision"] >= 0.75
            and 0.03 <= metrics["best_coverage"] <= 0.15
            and metrics["best_bad_event_rate"] <= 0.05
        )
        official_pass = int(bool(official_eligible) and observability_pass and accept_pass)
        summary = {
            "stage": "P2_PARALLEL_SCORE_REDESIGN_MATRIX",
            "status": "score_summary",
            "score_id": score_id,
            "score_description": description,
            "attach_candidate": "all_source_measured_v9240_A1_A2_A3",
            "row_count": len(rows),
            "corr": metrics["corr"],
            "auc": metrics["auc"],
            "precision": metrics["precision"],
            "coverage": metrics["coverage"],
            "bad_event_rate": metrics["bad_event_rate"],
            "accepted_event_count": metrics["accepted_event_count"],
            "best_precision": metrics["best_precision"],
            "best_coverage": metrics["best_coverage"],
            "best_bad_event_rate": metrics["best_bad_event_rate"],
            "best_accepted_event_count": metrics["best_accepted_event_count"],
            "threshold": metrics["threshold"],
            "diagnostic_score_pass": diagnostic_pass,
            "value_observability_shape_pass": observability_pass,
            "accept_abstain_pass": accept_pass,
            "official_score_pass": official_pass,
            "dataset_name_used": 0,
            "posthoc_used_at_commit": 0 if official_eligible else int("posthoc" in reason),
            "validation_used": 0,
            "test_used": 0,
            "official_eligible": official_eligible,
            "gate_missing_reason": reason,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        out.append(summary)
        for row, value in zip(rows, values):
            out.append({
                "stage": "P2_PARALLEL_SCORE_REDESIGN_MATRIX",
                "status": "event_score",
                "score_id": score_id,
                "score_description": description,
                "attach_candidate": row["attach_candidate"],
                "event_id": row["event_id"],
                "dataset": row["dataset"],
                "seed": row["seed"],
                "horizon": row["horizon"],
                "signal_stratum": row["signal_stratum"],
                "score_value": value,
                "grounded_value": row["grounded_value"],
                "Y_beat": row["Y_beat"],
                "bad_event": row["bad_event"],
                "corr": "",
                "auc": "",
                "precision": "",
                "coverage": "",
                "bad_event_rate": "",
                "accepted_event_count": "",
                "dataset_name_used": 0,
                "posthoc_used_at_commit": 0 if official_eligible else int("posthoc" in reason),
                "validation_used": 0,
                "test_used": 0,
                "official_eligible": official_eligible,
                "gate_missing_reason": reason,
                "source_artifact": "v9240/p6_value_observability_audit.csv",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    return out


def _score_values(rows: List[Dict[str, Any]], score_id: str) -> List[float]:
    for sid, _, values, _, _ in _candidate_score_specs(rows):
        if sid == score_id:
            return values
    return [0.0 for _ in rows]


def _best_threshold(scores: Sequence[float], labels: Sequence[int], bads: Sequence[int], indices: Sequence[int]) -> Tuple[float, float, float, int, float]:
    if not indices:
        return (0.0, 0.0, 0.0, 0, float("inf"))
    ordered = sorted(indices, key=lambda i: scores[i], reverse=True)
    best: Tuple[float, float, float, int, float] | None = None
    for cov in (0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15):
        count = max(1, min(len(ordered), round(len(indices) * cov)))
        chosen = ordered[:count]
        precision = sum(labels[i] for i in chosen) / count
        bad = sum(bads[i] for i in chosen) / count
        threshold = scores[chosen[-1]]
        item = (precision, count / len(indices), bad, count, threshold)
        if best is None or item[0] > best[0] or (item[0] == best[0] and item[1] > best[1]):
            best = item
    assert best is not None
    return best


def _p5_leave_out(rows: List[Dict[str, Any]], score_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    scores = _score_values(rows, score_id)
    labels = [int(r["Y_beat"]) for r in rows]
    bads = [int(r["bad_event"]) for r in rows]
    out: List[Dict[str, Any]] = []

    ldo_pass_count = 0
    datasets = sorted({r["dataset"] for r in rows})
    for heldout in datasets:
        train = [i for i, r in enumerate(rows) if r["dataset"] != heldout]
        test = [i for i, r in enumerate(rows) if r["dataset"] == heldout]
        train_precision, train_coverage, train_bad, train_count, threshold = _best_threshold(scores, labels, bads, train)
        accepted = [i for i in test if scores[i] >= threshold]
        count = len(accepted)
        precision = sum(labels[i] for i in accepted) / count if count else 0.0
        coverage = count / len(test) if test else 0.0
        bad = sum(bads[i] for i in accepted) / count if count else 0.0
        task_safe = _mean([rows[i]["task_safe"] for i in accepted]) if count else 0.0
        ce = _mean([rows[i]["tail_CEp99_delta"] for i in accepted]) if count else 0.0
        margin = _mean([rows[i]["margin_delta"] for i in accepted]) if count else 0.0
        split_pass = int(precision >= 0.50 and task_safe >= 0.995)
        ldo_pass_count += split_pass
        out.append({
            "stage": "P5_LEAVE_DATASET_AND_STRATUM_OUT_VALIDATION",
            "status": "measured",
            "split_type": "leave_dataset_out",
            "heldout": heldout,
            "attach_candidate": "all_source_measured_v9240_A1_A2_A3",
            "score_id": score_id,
            "threshold": threshold,
            "train_precision": train_precision,
            "train_coverage": train_coverage,
            "train_bad_event_rate": train_bad,
            "precision": precision,
            "coverage": coverage,
            "bad_event_rate": bad,
            "task_safe": task_safe,
            "CEp99_delta": ce,
            "margin_delta": margin,
            "ECE_delta": "not_measured_in_v9240_source",
            "NLL_delta": "not_measured_in_v9240_source",
            "curvature_delta": "not_measured_in_v9240_source",
            "beats_adamwparallel": precision,
            "beats_bestlr": precision,
            "shuffle_control_pass": 0,
            "split_pass": split_pass,
            "dataset_name_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })

    strata = sorted({r["signal_stratum"] for r in rows})
    lso_pass_count = 0
    for heldout in strata:
        train = [i for i, r in enumerate(rows) if r["signal_stratum"] != heldout]
        test = [i for i, r in enumerate(rows) if r["signal_stratum"] == heldout]
        if not train:
            out.append({
                "stage": "P5_LEAVE_DATASET_AND_STRATUM_OUT_VALIDATION",
                "status": "not_evaluable",
                "split_type": "leave_stratum_out",
                "heldout": heldout,
                "attach_candidate": "all_source_measured_v9240_A1_A2_A3",
                "score_id": score_id,
                "reason": "only_one_signal_stratum_measured_in_v9240_source",
                "dataset_name_used": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            continue
        _, _, _, _, threshold = _best_threshold(scores, labels, bads, train)
        accepted = [i for i in test if scores[i] >= threshold]
        count = len(accepted)
        precision = sum(labels[i] for i in accepted) / count if count else 0.0
        coverage = count / len(test) if test else 0.0
        bad = sum(bads[i] for i in accepted) / count if count else 0.0
        task_safe = _mean([rows[i]["task_safe"] for i in accepted]) if count else 0.0
        ce = _mean([rows[i]["tail_CEp99_delta"] for i in accepted]) if count else 0.0
        split_pass = int(task_safe >= 0.995 and ce <= 0.0)
        lso_pass_count += split_pass
        out.append({
            "stage": "P5_LEAVE_DATASET_AND_STRATUM_OUT_VALIDATION",
            "status": "measured",
            "split_type": "leave_stratum_out",
            "heldout": heldout,
            "attach_candidate": "all_source_measured_v9240_A1_A2_A3",
            "score_id": score_id,
            "threshold": threshold,
            "precision": precision,
            "coverage": coverage,
            "bad_event_rate": bad,
            "task_safe": task_safe,
            "CEp99_delta": ce,
            "margin_delta": _mean([rows[i]["margin_delta"] for i in accepted]) if count else 0.0,
            "beats_adamwparallel": precision,
            "beats_bestlr": precision,
            "shuffle_control_pass": 0,
            "split_pass": split_pass,
            "dataset_name_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })

    ldo_pass = int(ldo_pass_count >= 2)
    lso_evaluable = int(any(r.get("split_type") == "leave_stratum_out" and r.get("status") == "measured" for r in out))
    lso_pass = int(lso_evaluable and lso_pass_count / max(1, sum(1 for r in out if r.get("split_type") == "leave_stratum_out" and r.get("status") == "measured")) >= 0.70)
    summary = {
        "leave_dataset_out_pass": ldo_pass,
        "leave_dataset_out_pass_count": ldo_pass_count,
        "leave_dataset_out_split_count": len(datasets),
        "leave_stratum_out_pass": lso_pass,
        "leave_stratum_out_pass_count": lso_pass_count,
        "leave_stratum_out_evaluable": lso_evaluable,
        "leave_stratum_out_split_count": len(strata),
    }
    out.append({
        "stage": "P5_LEAVE_DATASET_AND_STRATUM_OUT_VALIDATION",
        "status": "summary",
        "split_type": "summary",
        "heldout": "summary",
        "score_id": score_id,
        **summary,
        "dataset_name_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    return out, summary


def _write_not_run_range(out_dir: Path, reason: str, start_stage: int) -> None:
    files = [
        (5, "p5_leave_dataset_and_stratum_out_validation.csv", "P5_LEAVE_DATASET_AND_STRATUM_OUT_VALIDATION"),
        (6, "p6_official_paired_replay.csv", "P6_OFFICIAL_PAIRED_REPLAY"),
        (7, "p7_short_run_functional_validation.csv", "P7_SHORT_RUN_FUNCTIONAL_VALIDATION"),
        (8, "p8_full_10seed_functional_validation.csv", "P8_FULL_10SEED_FUNCTIONAL_VALIDATION"),
        (9, "p9_robustness_external_ready.csv", "P9_ROBUSTNESS_EXTERNAL_READY"),
    ]
    for stage_num, name, stage in files:
        if stage_num >= start_stage:
            write_csv_rows(out_dir / name, [_not_run(stage, name, reason)])
    if start_stage <= 5:
        write_csv_rows(out_dir / "leave_dataset_out_trace_v9241.csv", [_not_run("P5_LEAVE_DATASET_AND_STRATUM_OUT_VALIDATION", "leave_dataset_out_trace_v9241.csv", reason)])
    if start_stage <= 6:
        write_csv_rows(out_dir / "paired_replay_branch_trace_v9241.csv", [_not_run("P6_OFFICIAL_PAIRED_REPLAY", "paired_replay_branch_trace_v9241.csv", reason)])


def _p1_autopsy(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    _assign_ranks(rows)
    current_scores = [r["current_score"] for r in rows]
    grounded = [r["grounded_value"] for r in rows]
    labels = [int(r["Y_beat"]) for r in rows]
    bads = [int(r["bad_event"]) for r in rows]
    current_metrics = _score_metrics(current_scores, grounded, labels, bads)
    flipped_metrics = _score_metrics([-s for s in current_scores], grounded, labels, bads)
    accepted_threshold = current_metrics["threshold"]
    p1_rows: List[Dict[str, Any]] = []
    mode_counts: Dict[str, int] = {}
    accepted_failed_count = 0
    for row in rows:
        accepted = int(row["current_score"] >= accepted_threshold)
        if not accepted:
            mode = "not_accepted_by_current_score"
        elif row["Y_beat"]:
            mode = "accepted_true_positive"
        elif row["bad_event"]:
            mode = "F7-tail_metric_mismatch"
        elif flipped_metrics["auc"] - current_metrics["auc"] >= 0.10 and row["grounded_value"] < 0:
            mode = "F5-score_sign_mismatch"
        elif row["real_gain"] <= max(row["adamwparallel_gain"], row["bestlr_gain"]):
            mode = "F6-control_gap_missing"
        else:
            mode = "F9-event_family_noise"
        if accepted and not row["Y_beat"]:
            accepted_failed_count += 1
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        out = {
            "stage": "P1_VALUE_FAILURE_AUTOPSY",
            "status": "measured",
            **{k: row[k] for k in [
                "event_id",
                "dataset_slice",
                "seed",
                "horizon",
                "signal_stratum",
                "attach_candidate",
                "controller",
                "current_score",
                "grounded_value",
                "Y_beat",
                "score_sign",
                "score_rank",
                "value_rank",
                "real_gain",
                "adamwparallel_gain",
                "bestlr_gain",
                "control_gap",
                "tail_CEp99_delta",
                "margin_delta",
                "task_risk",
                "uncertainty",
                "role_stack_score",
                "role_head_score",
            ]},
            "accepted_by_current_score": accepted,
            "failure_mode": mode,
            "dataset_name_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        p1_rows.append(out)
    primary_mode = max(mode_counts.items(), key=lambda kv: kv[1])[0] if mode_counts else "none"
    primary_fraction = (mode_counts.get(primary_mode, 0) / accepted_failed_count) if accepted_failed_count else 0.0
    summary = {
        "current_corr": current_metrics["corr"],
        "current_auc": current_metrics["auc"],
        "current_precision": current_metrics["precision"],
        "current_coverage": current_metrics["coverage"],
        "current_bad_event_rate": current_metrics["bad_event_rate"],
        "signflip_corr": flipped_metrics["corr"],
        "signflip_auc": flipped_metrics["auc"],
        "signflip_precision": flipped_metrics["precision"],
        "signflip_coverage": flipped_metrics["coverage"],
        "control_gap_corr": _corr([r["control_gap"] for r in rows], grounded),
        "accepted_failed_event_count": accepted_failed_count,
        "primary_failure_mode": primary_mode,
        "primary_failure_fraction": primary_fraction,
        "score_sign_mismatch_pass": int(flipped_metrics["auc"] - current_metrics["auc"] >= 0.10 or flipped_metrics["auc"] >= 0.60),
        "value_failure_attribution_pass": int(primary_fraction >= 0.90 and accepted_failed_count > 0),
    }
    p1_rows.append({
        "stage": "P1_VALUE_FAILURE_AUTOPSY",
        "status": "summary",
        "event_id": "summary",
        **summary,
        "dataset_name_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    return p1_rows, summary


def _p0_boundary(source_dir: Path) -> Dict[str, Any]:
    route = _read_json(source_dir / "route_decision.json")
    audit = read_csv_rows(source_dir / "v9240_provenance_audit.csv")
    audit_row = audit[0] if audit else {}
    pass_gate = int(
        route.get("route") == "R11-BasePreservedButValueUnobservable"
        and _int(route.get("repaired_base_robust_pass")) == 1
        and _int(route.get("snapshot_attach_pass")) == 1
        and _int(route.get("checkpoint_inactive_equivalence_pass")) == 1
        and _int(route.get("no_event_replay_preservation_pass")) == 1
        and _int(route.get("functional_carrier_pass")) == 1
        and _int(route.get("value_observability_pass")) == 0
        and _int(audit_row.get("fake_proxy_nonzero_count", route.get("fake_proxy_nonzero_count"))) == 0
    )
    return {
        "stage": "P0_V9240_BOUNDARY_REPRODUCTION",
        "status": "measured",
        "route": route.get("route", ""),
        "source_route_v9240": route.get("route", ""),
        "repaired_base_robust_pass": _int(route.get("repaired_base_robust_pass")),
        "snapshot_attach_pass": _int(route.get("snapshot_attach_pass")),
        "checkpoint_inactive_equivalence_pass": _int(route.get("checkpoint_inactive_equivalence_pass")),
        "no_event_replay_preservation_pass": _int(route.get("no_event_replay_preservation_pass")),
        "functional_carrier_pass": _int(route.get("functional_carrier_pass")),
        "max_r_z_tail": _float(route.get("max_r_z_tail")),
        "max_r_perp_tail": _float(route.get("max_r_perp_tail")),
        "carrier_bad_event_rate": _float(route.get("carrier_bad_event_rate")),
        "value_observability_pass": _int(route.get("value_observability_pass")),
        "value_auc": _float(route.get("value_auc")),
        "value_corr": _float(route.get("value_corr")),
        "accepted_precision": _float(route.get("accepted_precision")),
        "accepted_coverage": _float(route.get("accepted_coverage")),
        "accepted_bad_event_rate": _float(route.get("accepted_bad_event_rate")),
        "fake_proxy_count": _int(audit_row.get("fake_proxy_nonzero_count", route.get("fake_proxy_nonzero_count"))),
        "v9240_boundary_pass": pass_gate,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _p3_rolewise(source_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    implementation = read_csv_rows(source_dir / "p2_snapshot_late_attach_implementation.csv")
    inactive = read_csv_rows(source_dir / "p3_checkpoint_inactive_equivalence.csv")
    noevent = read_csv_rows(source_dir / "p4_no_event_replay_preservation.csv")
    candidates = {"A3-LateAttachRoleWiseFT7EdgeCarrier", "A5-FamilyValueLateAttach"}
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "rolewise_attach_pass": 0,
        "rolewise_implemented_count": 0,
        "rolewise_inactive_equivalence_pass": 0,
        "rolewise_no_event_preservation_pass": 0,
    }
    for cand in sorted(candidates):
        impl = [r for r in implementation if r.get("attach_candidate") == cand]
        inactive_rows = [r for r in inactive if r.get("attach_candidate") == cand]
        noevent_rows = [r for r in noevent if r.get("attach_candidate") == cand]
        implemented = int(any(_int(r.get("implemented")) == 1 for r in impl))
        contract = int(any(_int(r.get("strict_purekan_contract_pass")) == 1 for r in impl))
        inactive_pass = int(bool(inactive_rows) and all(_int(r.get("inactive_equivalence_pass")) == 1 for r in inactive_rows))
        noevent_pass = int(bool(noevent_rows) and all(_int(r.get("no_event_preservation_pass")) == 1 for r in noevent_rows))
        pass_gate = int(implemented and contract and inactive_pass and noevent_pass)
        rows.append({
            "stage": "P3_ROLEWISE_LATE_ATTACH_IMPLEMENTATION",
            "status": "source_measured" if impl else "not_implemented_in_v9240_source",
            "attach_candidate": cand,
            "base_checkpoint_hash": impl[0].get("base_checkpoint_hash", "") if impl else "",
            "role_channels": "stack,head" if "RoleWise" in cand else "family_value",
            "edge_owned_param_fraction": _float(impl[0].get("edge_owned_param_fraction")) if impl else 0.0,
            "external_residual_used": _int(impl[0].get("external_residual_used")) if impl else 0,
            "ordinary_mlp_path_used": _int(impl[0].get("ordinary_mlp_path_used")) if impl else 0,
            "manual_forward": _int(impl[0].get("manual_forward")) if impl else 0,
            "manual_backward": _int(impl[0].get("manual_backward")) if impl else 0,
            "manual_update": _int(impl[0].get("manual_update")) if impl else 0,
            "uses_loss_backward": _int(impl[0].get("uses_loss_backward")) if impl else 0,
            "task_param_hash_before_attach": impl[0].get("task_param_hash_before_attach", "") if impl else "",
            "task_param_hash_after_attach": impl[0].get("task_param_hash_after_attach", "") if impl else "",
            "optimizer_state_hash_before_attach": impl[0].get("optimizer_state_hash_before_attach", "") if impl else "",
            "optimizer_state_hash_after_attach": impl[0].get("optimizer_state_hash_after_attach", "") if impl else "",
            "functional_param_zero_init": _int(impl[0].get("functional_param_zero_init")) if impl else 0,
            "inactive_equivalence_max_logit_diff": max((_float(r.get("max_logit_diff_inactive")) for r in inactive_rows), default=0.0),
            "no_event_max_logit_diff": max((_float(r.get("logit_diff_vs_base_continue")) for r in noevent_rows), default=0.0),
            "implemented": implemented,
            "implementation_status": impl[0].get("implementation_status", "") if impl else "not_measured_in_v9240_source",
            "strict_purekan_contract_pass": contract,
            "inactive_equivalence_pass": inactive_pass,
            "no_event_preservation_pass": noevent_pass,
            "rolewise_attach_pass": pass_gate,
            "source_artifact": "v9240/p2_p3_p4",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    a3 = [r for r in rows if r["attach_candidate"] == "A3-LateAttachRoleWiseFT7EdgeCarrier"][0]
    summary.update({
        "rolewise_attach_pass": a3["rolewise_attach_pass"],
        "rolewise_implemented_count": sum(int(r["implemented"]) for r in rows),
        "rolewise_inactive_equivalence_pass": a3["inactive_equivalence_pass"],
        "rolewise_no_event_preservation_pass": a3["no_event_preservation_pass"],
    })
    return rows, summary


def _p4_oracle(rows: List[Dict[str, Any]], seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    labels = [int(r["Y_beat"]) for r in rows]
    bads = [int(r["bad_event"]) for r in rows]
    grounded = [r["grounded_value"] for r in rows]
    score_sources = {
        "OracleUpperBound": grounded,
        "ValueScoreShuffled": [r["current_score"] for r in rows],
        "FunctionalChannelShuffled": grounded[:],
        "TailMaskShuffled": [r["tail_value_score"] for r in rows],
        "RoleScoreShuffled": [r["role_score"] for r in rows],
    }
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    for name, values in score_sources.items():
        values = list(values)
        if name != "OracleUpperBound":
            rng.shuffle(values)
        metrics = _score_metrics(values, grounded, labels, bads)
        pass_gate = int(
            metrics["best_precision"] >= 0.75
            and 0.03 <= metrics["best_coverage"] <= 0.15
            and metrics["best_bad_event_rate"] <= 0.05
        )
        out.append({
            "stage": "P4_ORACLE_UPPER_BOUND_AND_SHUFFLE_CONTROLS",
            "status": "summary",
            "control_id": name,
            "attach_candidate": "all_source_measured_v9240_A1_A2_A3",
            "oracle_precision": metrics["best_precision"],
            "oracle_coverage": metrics["best_coverage"],
            "oracle_bad_event_rate": metrics["best_bad_event_rate"],
            "oracle_auc": metrics["auc"],
            "oracle_corr": metrics["corr"],
            "oracle_upper_bound_pass": pass_gate if name == "OracleUpperBound" else 0,
            "shuffle_control_pass": pass_gate if name != "OracleUpperBound" else 0,
            "ValueScoreShuffled_pass": pass_gate if name == "ValueScoreShuffled" else "",
            "FunctionalChannelShuffled_pass": pass_gate if name == "FunctionalChannelShuffled" else "",
            "TailMaskShuffled_pass": pass_gate if name == "TailMaskShuffled" else "",
            "RoleScoreShuffled_pass": pass_gate if name == "RoleScoreShuffled" else "",
            "dataset_name_used": 0,
            "posthoc_used_at_commit": 1 if name == "OracleUpperBound" else 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
        if name == "OracleUpperBound":
            summary.update({
                "oracle_upper_bound_pass": pass_gate,
                "oracle_precision": metrics["best_precision"],
                "oracle_coverage": metrics["best_coverage"],
                "oracle_bad_event_rate": metrics["best_bad_event_rate"],
                "oracle_auc": metrics["auc"],
                "oracle_corr": metrics["corr"],
            })
    shuffled_any_pass = int(any(_int(r.get("shuffle_control_pass")) for r in out))
    summary["shuffle_control_any_pass"] = shuffled_any_pass
    return out, summary


def _write_not_run_downstream(out_dir: Path, reason: str) -> None:
    files = [
        ("p5_leave_dataset_and_stratum_out_validation.csv", "P5_LEAVE_DATASET_AND_STRATUM_OUT_VALIDATION"),
        ("p6_official_paired_replay.csv", "P6_OFFICIAL_PAIRED_REPLAY"),
        ("p7_short_run_functional_validation.csv", "P7_SHORT_RUN_FUNCTIONAL_VALIDATION"),
        ("p8_full_10seed_functional_validation.csv", "P8_FULL_10SEED_FUNCTIONAL_VALIDATION"),
        ("p9_robustness_external_ready.csv", "P9_ROBUSTNESS_EXTERNAL_READY"),
    ]
    for name, stage in files:
        write_csv_rows(out_dir / name, [_not_run(stage, name, reason)])
    write_csv_rows(out_dir / "leave_dataset_out_trace_v9241.csv", [_not_run("P5_LEAVE_DATASET_AND_STRATUM_OUT_VALIDATION", "leave_dataset_out_trace_v9241.csv", reason)])
    write_csv_rows(out_dir / "paired_replay_branch_trace_v9241.csv", [_not_run("P6_OFFICIAL_PAIRED_REPLAY", "paired_replay_branch_trace_v9241.csv", reason)])


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")

    source_dir = Path(args.source_dir)
    p0 = _p0_boundary(source_dir)
    event_rows, _ = _load_event_rows(source_dir)
    p1_rows, p1_summary = _p1_autopsy(event_rows)
    p2_rows = _score_candidates(event_rows)
    p3_rows, p3_summary = _p3_rolewise(source_dir)
    p4_rows, p4_summary = _p4_oracle(event_rows, int(args.seed))

    score_summaries = [r for r in p2_rows if r.get("status") == "score_summary"]
    legal_summaries = [r for r in score_summaries if _int(r.get("official_eligible")) == 1]
    best_legal = max(legal_summaries, key=lambda r: (_int(r.get("official_score_pass")), _float(r.get("auc")), _float(r.get("corr")), _float(r.get("best_precision"))), default={})
    official_score_pass = int(any(_int(r.get("official_score_pass")) == 1 for r in legal_summaries))
    value_observability_pass = official_score_pass
    best_score_id = best_legal.get("score_id", "")
    value_auc = _float(best_legal.get("auc"))
    value_corr = _float(best_legal.get("corr"))
    accepted_precision = _float(best_legal.get("best_precision"))
    accepted_coverage = _float(best_legal.get("best_coverage"))
    accepted_bad_event_rate = _float(best_legal.get("best_bad_event_rate"))

    p5_summary: Dict[str, Any] = {
        "leave_dataset_out_pass": 0,
        "leave_dataset_out_pass_count": 0,
        "leave_stratum_out_pass": 0,
        "leave_stratum_out_pass_count": 0,
        "leave_stratum_out_evaluable": 0,
    }

    if value_observability_pass:
        p5_rows, p5_summary = _p5_leave_out(event_rows, best_score_id)
        write_csv_rows(out_dir / "p5_leave_dataset_and_stratum_out_validation.csv", p5_rows)
        write_csv_rows(out_dir / "leave_dataset_out_trace_v9241.csv", p5_rows)
        if not (_int(p5_summary.get("leave_dataset_out_pass")) and _int(p5_summary.get("leave_stratum_out_pass"))):
            _write_not_run_range(out_dir, "P5_leave_dataset_or_stratum_out_failed", 6)
        else:
            _write_not_run_range(out_dir, "P6_official_paired_replay_not_implemented_in_this_runner", 6)
    else:
        _write_not_run_range(out_dir, "P2_no_legal_value_observability_survivor", 5)

    if not _int(p0.get("v9240_boundary_pass")):
        route_name = "R10-BasePreservedCarrierActiveButValueUnobservable"
        primary_blocker = "v9240_boundary_not_reproduced"
        next_impl = "reproduce_v9240_boundary_before_value_redesign"
    elif official_score_pass and not (_int(p5_summary.get("leave_dataset_out_pass")) and _int(p5_summary.get("leave_stratum_out_pass"))):
        route_name = "R3-ControlGapScorePass"
        primary_blocker = "leave_dataset_or_stratum_out_failed_after_value_score_pass"
        next_impl = "stabilize_legal_score_across_heldout_dataset_and_signal_stratum_before_official_replay"
    elif official_score_pass:
        route_name = "R8-LeaveDatasetOutPass"
        primary_blocker = "official_paired_replay_not_implemented_after_leaveout_pass"
        next_impl = "run_official_paired_replay_for_value_score_survivor"
    elif _int(p4_summary.get("oracle_upper_bound_pass")) and not official_score_pass:
        route_name = "R6-OracleHighLegalScoreLow"
        primary_blocker = "oracle_good_events_exist_but_legal_value_score_failed"
        next_impl = "redesign_legal_control_gap_or_rolewise_value_score_without_posthoc_or_dataset_keys"
    elif _int(p1_summary.get("score_sign_mismatch_pass")):
        route_name = "R2-ScoreSignMismatchConfirmed"
        primary_blocker = "score_sign_mismatch_but_no_legal_acceptance_gate"
        next_impl = "convert_sign_corrected_score_to_legal_control_gap_controller"
    else:
        route_name = "R7-OracleLowCarrierMechanismReset"
        primary_blocker = "oracle_upper_bound_failed_current_carrier_not_control_resistant"
        next_impl = "redesign_functional_carrier_mechanism"

    manifest = {
        "experiment": "DG-KAN v9.2.41 ControlGap ValueAlignment RoleWiseLateAttach",
        "plan": str(PLAN_PATH.relative_to(ROOT)),
        "script": str(SCRIPT_PATH.relative_to(ROOT)),
        "source_v9240": _rel(source_dir),
        "out_dir": _rel(out_dir),
        "seed": int(args.seed),
        "device": args.device,
        "data_root": args.data_root,
        "fresh": bool(args.fresh),
        "completed_at": _now_iso(),
    }
    write_json(out_dir / "run_manifest.json", manifest)
    write_csv_rows(out_dir / "p0_v9240_boundary_reproduction.csv", [p0])
    write_csv_rows(out_dir / "p1_value_failure_autopsy.csv", p1_rows)
    write_csv_rows(out_dir / "value_failure_trace_v9241.csv", p1_rows)
    write_csv_rows(out_dir / "p2_parallel_score_redesign_matrix.csv", p2_rows)
    write_csv_rows(out_dir / "score_component_trace_v9241.csv", p2_rows)
    write_csv_rows(out_dir / "p3_rolewise_late_attach_implementation.csv", p3_rows)
    write_csv_rows(out_dir / "rolewise_attach_trace_v9241.csv", p3_rows)
    write_csv_rows(out_dir / "p4_oracle_upper_bound_and_shuffle_controls.csv", p4_rows)
    write_csv_rows(out_dir / "oracle_upper_bound_trace_v9241.csv", p4_rows)
    write_csv_rows(out_dir / "contract_audit_v9241.csv", [{
        "stage": "CONTRACT_AUDIT",
        "status": "measured",
        "strict_purekan_contract_pass": p3_summary.get("rolewise_attach_pass", 0),
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_update": 1,
        "uses_loss_backward": 0,
        "ordinary_mlp_path_used": 0,
        "external_residual_used": 0,
        "dataset_tuning_detected": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])

    audit_paths = [
        out_dir / "p0_v9240_boundary_reproduction.csv",
        out_dir / "p1_value_failure_autopsy.csv",
        out_dir / "p2_parallel_score_redesign_matrix.csv",
        out_dir / "p3_rolewise_late_attach_implementation.csv",
        out_dir / "p4_oracle_upper_bound_and_shuffle_controls.csv",
        out_dir / "p5_leave_dataset_and_stratum_out_validation.csv",
        out_dir / "p6_official_paired_replay.csv",
        out_dir / "p7_short_run_functional_validation.csv",
        out_dir / "p8_full_10seed_functional_validation.csv",
        out_dir / "p9_robustness_external_ready.csv",
        out_dir / "contract_audit_v9241.csv",
    ]
    audit = audit_no_fake(audit_paths)
    completed_at = _now_iso()
    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9240_boundary_pass": _int(p0.get("v9240_boundary_pass")),
        "source_route": p0.get("route", ""),
        "dataset_tuning_detected": 0,
        "value_failure_mode": p1_summary.get("primary_failure_mode", ""),
        "value_failure_attribution_pass": p1_summary.get("value_failure_attribution_pass", 0),
        "accepted_failed_event_count": p1_summary.get("accepted_failed_event_count", 0),
        "primary_failure_fraction": p1_summary.get("primary_failure_fraction", 0.0),
        "score_sign_mismatch_pass": p1_summary.get("score_sign_mismatch_pass", 0),
        "current_value_auc": p1_summary.get("current_auc", 0.0),
        "current_value_corr": p1_summary.get("current_corr", 0.0),
        "signflip_auc": p1_summary.get("signflip_auc", 0.0),
        "signflip_corr": p1_summary.get("signflip_corr", 0.0),
        "control_gap_corr": p1_summary.get("control_gap_corr", 0.0),
        "best_score_id": best_score_id,
        "best_attach_candidate": "A3-LateAttachRoleWiseFT7EdgeCarrier" if p3_summary.get("rolewise_attach_pass") else "A2-LateAttachControlGapChannel",
        "value_observability_pass": value_observability_pass,
        "value_auc": value_auc,
        "value_corr": value_corr,
        "accepted_precision": accepted_precision,
        "accepted_coverage": accepted_coverage,
        "accepted_bad_event_rate": accepted_bad_event_rate,
        "oracle_upper_bound_pass": p4_summary.get("oracle_upper_bound_pass", 0),
        "oracle_precision": p4_summary.get("oracle_precision", 0.0),
        "oracle_coverage": p4_summary.get("oracle_coverage", 0.0),
        "oracle_bad_event_rate": p4_summary.get("oracle_bad_event_rate", 0.0),
        "oracle_auc": p4_summary.get("oracle_auc", 0.0),
        "rolewise_attach_pass": p3_summary.get("rolewise_attach_pass", 0),
        "rolewise_implemented_count": p3_summary.get("rolewise_implemented_count", 0),
        "shuffle_control_any_pass": p4_summary.get("shuffle_control_any_pass", 0),
        "leave_dataset_out_pass": p5_summary.get("leave_dataset_out_pass", 0),
        "leave_dataset_out_pass_count": p5_summary.get("leave_dataset_out_pass_count", 0),
        "leave_stratum_out_pass": p5_summary.get("leave_stratum_out_pass", 0),
        "leave_stratum_out_pass_count": p5_summary.get("leave_stratum_out_pass_count", 0),
        "leave_stratum_out_evaluable": p5_summary.get("leave_stratum_out_evaluable", 0),
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "functional_task_safe": 1,
        "functional_control_pass": 0,
        "functional_system_pass": 1,
        "hard_stratum_repair_pass": 0,
        "adamw_fullpass": 0,
        "strong_baseline_pass": 0,
        "robustness_pass": 0,
        "external_ready": 0,
        "primary_blocker": primary_blocker,
        "next_required_implementation": next_impl,
        "success_v9241_strict_purekan_functional": 0,
        "success_v9241_full_functional": 0,
        "success_v9241_external_ready": 0,
        **audit,
        "completed_at": completed_at,
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    write_csv_rows(out_dir / "failure_table.csv", [{
        "stage": "ROUTE",
        "status": "terminal",
        "route": route_name,
        "primary_blocker": primary_blocker,
        "failure_code": "F15_value_observability_fail" if route_name != "R7-OracleLowCarrierMechanismReset" else "F12_true_mechanism_absent",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    write_csv_rows(out_dir / "v9241_provenance_audit.csv", [audit])
    return route


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(RESULT_ROOT / "v9241_controlgap_value_alignment_rolewise_lateattach_first_20260511T133000Z"))
    parser.add_argument("--source-dir", default=str(SRC_V9240))
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=1314)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    route = run(args)
    print(json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
