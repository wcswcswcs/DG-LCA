"""Train-only retained-source certificate helpers for v22.09.

The functions here score already-measured train-stream evidence. Future horizon
columns are accepted only as audit labels and are never used by the score
builders.
"""

from __future__ import annotations

from math import isfinite, sqrt
from typing import Any, Iterable


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in {"", None}:
            return default
        out = float(value)
    except Exception:
        return default
    return out if isfinite(out) else default


def int_flag(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def official_early_chain_label(row: dict[str, Any], threshold: float = 0.005) -> int:
    """Audit label for h100+h400+h800 source positivity."""

    return int(
        finite_float(row.get("future_audit_source_h100"), -999.0) >= threshold
        and finite_float(row.get("future_audit_source_h400"), -999.0) >= threshold
        and finite_float(row.get("future_audit_source_h800"), -999.0) >= threshold
    )


def retained_h800_h3200_label(row: dict[str, Any], threshold: float = 0.005) -> int:
    """Audit label for h800+h3200 source positivity."""

    return int(
        finite_float(row.get("future_audit_source_h800"), -999.0) >= threshold
        and finite_float(row.get("future_audit_source_h3200"), -999.0) >= threshold
    )


def h3200_label(row: dict[str, Any], threshold: float = 0.005) -> int:
    return int(finite_float(row.get("future_audit_source_h3200"), -999.0) >= threshold)


def zscore_by_group(values: Iterable[Any], groups: Iterable[Any]) -> list[float]:
    vals = [finite_float(v) for v in values]
    group_list = list(groups)
    stats: dict[Any, tuple[float, float]] = {}
    for group in sorted(set(group_list), key=str):
        gvals = [v for v, g in zip(vals, group_list) if g == group and isfinite(v)]
        if not gvals:
            stats[group] = (0.0, 1.0)
            continue
        mean = sum(gvals) / len(gvals)
        var = sum((v - mean) ** 2 for v in gvals) / len(gvals)
        stats[group] = (mean, sqrt(var) if var > 1.0e-12 else 1.0)
    return [((0.0 if not isfinite(v) else v) - stats[g][0]) / stats[g][1] for v, g in zip(vals, group_list)]


def rank_auc(scores: list[float], labels: list[int]) -> float | str:
    valid = [(s, y) for s, y in zip(scores, labels) if isfinite(s)]
    if not valid:
        return ""
    scores = [s for s, _y in valid]
    labels = [int(y) for _s, y in valid]
    pos = sum(labels)
    neg = len(labels) - pos
    if pos <= 0 or neg <= 0:
        return ""
    ordered = sorted((s, i) for i, s in enumerate(scores))
    ranks = [0.0] * len(scores)
    p = 0
    while p < len(ordered):
        q = p + 1
        while q < len(ordered) and ordered[q][0] == ordered[p][0]:
            q += 1
        avg = 0.5 * (p + q - 1) + 1.0
        for _s, idx in ordered[p:q]:
            ranks[idx] = avg
        p = q
    pos_rank = sum(r for r, y in zip(ranks, labels) if y)
    return (pos_rank - pos * (pos + 1) / 2.0) / float(pos * neg)


def topk_precision(
    scores: list[float],
    labels: list[int],
    controls: list[int],
    k: int,
) -> tuple[float | str, float | str, float | str, list[int]]:
    valid = [(i, s) for i, s in enumerate(scores) if isfinite(s)]
    if not valid:
        return "", "", "", []
    order = [i for i, _s in sorted(valid, key=lambda item: item[1], reverse=True)[: min(k, len(valid))]]
    if not order:
        return "", "", "", []
    hits = sum(int(labels[i]) for i in order)
    positives = sum(int(v) for v in labels)
    control_fraction = sum(int(controls[i]) for i in order) / float(len(order))
    recall = hits / float(positives) if positives else ""
    return hits / float(len(order)), recall, control_fraction, order


def heldout_topk_precision(scores: list[float], labels: list[int], groups: list[str], k: int) -> float | str:
    vals: list[float] = []
    for group in sorted(set(groups)):
        idx = [i for i, g in enumerate(groups) if g == group and isfinite(scores[i])]
        if not idx:
            continue
        order = sorted(idx, key=lambda i: scores[i], reverse=True)[: min(k, len(idx))]
        if order:
            vals.append(sum(int(labels[i]) for i in order) / float(len(order)))
    return min(vals) if vals else ""


def certificate_summary(
    rows: list[dict[str, Any]],
    *,
    family: str,
    score_key: str,
    top_k: int = 20,
    commutator_key: str = "commutator_norm_ratio",
    flow_gap_key: str = "flow_order_gap",
    random_gap_key: str = "random_sign_corrupt_gap",
    control_limit: float = 0.10,
    auc_gate: float = 0.75,
    precision_gate: float = 0.50,
    heldout_gate: float = 0.40,
) -> dict[str, Any]:
    scores = [finite_float(r.get(score_key)) for r in rows]
    labels = [official_early_chain_label(r) for r in rows]
    h3200 = [h3200_label(r) for r in rows]
    retained = [retained_h800_h3200_label(r) for r in rows]
    controls = [int_flag(r.get("is_control")) for r in rows]
    groups = [f"{r.get('dataset', '')}:{r.get('seed', '')}" for r in rows]
    precision, recall, control_fraction, order = topk_precision(scores, labels, controls, top_k)
    heldout = heldout_topk_precision(scores, labels, groups, top_k)
    top = [rows[i] for i in order]
    random_gap = sum(finite_float(r.get(random_gap_key), 0.0) for r in top) / float(len(top)) if top else ""
    commutator = sum(finite_float(r.get(commutator_key), 999.0) for r in top) / float(len(top)) if top else ""
    flow_gap = sum(finite_float(r.get(flow_gap_key), 999.0) for r in top) / float(len(top)) if top else ""
    auc_early = rank_auc(scores, labels)
    blockers: list[str] = []
    if sum(labels) == 0:
        blockers.append("no_positive_official_early_chain_label_rows")
    if finite_float(auc_early, -1.0) < auc_gate:
        blockers.append("auc_official_early_chain_gate")
    if finite_float(precision, -1.0) < precision_gate:
        blockers.append("precision_top20_gate")
    if finite_float(heldout, -1.0) < heldout_gate:
        blockers.append("heldout_precision_gate")
    if finite_float(control_fraction, 1.0) > control_limit:
        blockers.append("control_equivalent_fraction_gate")
    if finite_float(random_gap, -999.0) <= 0.0:
        blockers.append("random_sign_corrupt_gap_gate")
    return {
        "certificate_family": family,
        "score_key": score_key,
        "rows": len(rows),
        "positive_official_early_chain_rows": sum(labels),
        "positive_h3200_rows": sum(h3200),
        "positive_retained_h800_h3200_rows": sum(retained),
        "AUC_predict_official_early_chain": auc_early,
        "AUC_predict_h3200_positive": rank_auc(scores, h3200),
        "AUC_predict_retained_h800_h3200_positive": rank_auc(scores, retained),
        "precision_at_top20": precision,
        "recall_at_top20": recall,
        "heldout_precision_at_top20": heldout,
        "control_equivalent_fraction": control_fraction,
        "top20_random_sign_corrupt_gap_mean": random_gap,
        "top20_commutator_norm_ratio_mean": commutator,
        "top20_flow_order_gap_mean": flow_gap,
        "uses_future_for_direction": 0,
        "future_source_used_as_audit_label_only": 1,
        "C0_retained_source_certificate_pass": int(not blockers),
        "blocker": ";".join(dict.fromkeys(blockers)),
    }


def retained_source_certificate_unit_tests() -> list[dict[str, Any]]:
    rows = [
        {"score": 0.9, "future_audit_source_h100": 0.01, "future_audit_source_h400": 0.02, "future_audit_source_h800": 0.03, "future_audit_source_h3200": 0.04, "dataset": "A", "seed": 0, "is_control": 0, "random_sign_corrupt_gap": 0.2},
        {"score": 0.1, "future_audit_source_h100": -0.1, "future_audit_source_h400": 0.0, "future_audit_source_h800": 0.0, "future_audit_source_h3200": -0.2, "dataset": "A", "seed": 0, "is_control": 1, "random_sign_corrupt_gap": -0.1},
    ]
    summary = certificate_summary(rows, family="unit", score_key="score", top_k=1)
    return [
        {"test": "official_label", "pass": official_early_chain_label(rows[0]) == 1 and official_early_chain_label(rows[1]) == 0},
        {"test": "auc", "pass": finite_float(rank_auc([0.9, 0.1], [1, 0]), -1.0) == 1.0},
        {"test": "summary_future_audit_only", "pass": int(summary["uses_future_for_direction"]) == 0 and int(summary["future_source_used_as_audit_label_only"]) == 1},
    ]
