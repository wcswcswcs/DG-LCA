#!/usr/bin/env python3
"""Post-process v22.17 bridge rows into source/risk/control smoke matrices."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_17_common import OUT_ROOT, append_exec, ensure_out, finite_float, read_rows, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--horizons", default="")
    p.add_argument("--merge-window", type=int, default=10)
    p.add_argument("--tau-u-margin", type=float, default=0.0)
    p.add_argument("--tau-l", type=float, default=1.0e-8)
    p.add_argument("--tau-i", type=float, default=1.0e-8)
    return p


def _auc(labels: list[int], scores: list[float]) -> float | str:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return ""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    rank_sum = 0.0
    for idx, (_score, label) in enumerate(pairs, start=1):
        if label:
            rank_sum += idx
    return float((rank_sum - pos * (pos + 1) / 2) / (pos * neg))


def _median(values: list[float]) -> float | str:
    if not values:
        return ""
    vals = sorted(values)
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def _precision_at_top_fraction(labels: list[int], scores: list[float], frac: float = 0.20) -> float | str:
    if not labels or not scores or len(labels) != len(scores):
        return ""
    k = max(1, int(math.ceil(len(labels) * frac)))
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)[:k]
    return float(sum(label for _score, label in pairs) / max(1, len(pairs)))


def _recall_at_fpr(labels: list[int], scores: list[float], max_fpr: float = 0.30) -> float | str:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return ""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    tp = fp = 0
    best = 0.0
    for _score, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
        fpr = fp / neg
        if fpr <= max_fpr:
            best = max(best, tp / pos)
    return float(best)


FEATURE_COLUMNS = [
    "U_task_usefulness",
    "random_control_U",
    "source_loss_t",
    "source_norm",
    "controller_source_norm",
    "linear_task_gain",
    "NDS",
    "TailRisk_proxy",
    "ControlProjection",
    "Staleness",
    "controller_source_loss_gain",
    "split_consensus_cosine",
    "predicted_gain_base",
    "predicted_gain_guided",
    "virtual_loss_base",
    "virtual_loss_guided",
    "source_age",
    "risk_score",
    "lambda_t",
]


def _feature_vector(row: dict[str, Any]) -> list[float]:
    vec = [finite_float(row.get(col), 0.0) for col in FEATURE_COLUMNS]
    u = finite_float(row.get("U_task_usefulness"), 0.0)
    rnd = finite_float(row.get("random_control_U"), 0.0)
    base = finite_float(row.get("predicted_gain_base"), 0.0)
    guided = finite_float(row.get("predicted_gain_guided"), 0.0)
    vbase = finite_float(row.get("virtual_loss_base"), 0.0)
    vguided = finite_float(row.get("virtual_loss_guided"), 0.0)
    vec.extend([u - rnd, guided - base, vbase - vguided])
    # Action-candidate identity is known at the current step and does not use
    # future labels; it lets C model source/action actionability rather than
    # source-only washout.
    for action_name in (
        "DGMLP_FU",
        "DGMLP_FU_GATED",
        "DGMLP_FU_ACTIONABLE",
        "DGMLP_FU_RELEASE",
        "DGMLP_FU_SPLIT_ACTIONABLE",
        "DGMLP_FU_SPLIT_RELEASE",
        "DGMLP_FU_SPLIT_VIRTUAL",
        "DGKAN_DFOU_FU_SPLIT_VIRTUAL",
        "DGKAN_DCHE_FU_SPLIT_VIRTUAL",
    ):
        vec.append(1.0 if row.get("intervention_model") == action_name else 0.0)
    vec.append(1.0 if row.get("model_family") == "MLP" else 0.0)
    vec.append(1.0 if row.get("model_family") == "KAN" else 0.0)
    return vec


def _fit_predict(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], neutral_train_rows: list[dict[str, Any]] | None = None) -> list[float] | None:
    neutral_train_rows = neutral_train_rows or []
    y_train = [int(r["y_actionable_matched_diagnostic"]) for r in train_rows] + [0 for _r in neutral_train_rows]
    if len(set(y_train)) < 2:
        return None
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return None
    x_train = [_feature_vector(r) for r in train_rows] + [_feature_vector(r) for r in neutral_train_rows]
    x_test = [_feature_vector(r) for r in test_rows]
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
    )
    clf.fit(x_train, y_train)
    return [float(x) for x in clf.predict_proba(x_test)[:, 1]]


def _group_cv_scores(rows: list[dict[str, Any]], key_fields: tuple[str, ...], neutral_rows: list[dict[str, Any]] | None = None) -> list[float] | None:
    neutral_rows = neutral_rows or []
    if len({int(r["y_actionable_matched_diagnostic"]) for r in rows}) < 2:
        return None
    groups: dict[tuple[str, ...], list[int]] = {}
    neutral_groups: dict[tuple[str, ...], list[int]] = {}
    for idx, row in enumerate(rows):
        groups.setdefault(tuple(str(row.get(k, "")) for k in key_fields), []).append(idx)
    for idx, row in enumerate(neutral_rows):
        neutral_groups.setdefault(tuple(str(row.get(k, "")) for k in key_fields), []).append(idx)
    scores: list[float | None] = [None] * len(rows)
    for group_key, indices in groups.items():
        held = set(indices)
        train = [row for idx, row in enumerate(rows) if idx not in held]
        neutral_train = [row for key, idxs in neutral_groups.items() for idx, row in enumerate(neutral_rows) if key != group_key and idx in idxs]
        test = [row for idx, row in enumerate(rows) if idx in held]
        pred = _fit_predict(train, test, neutral_train)
        if pred is None:
            continue
        for idx, score in zip(indices, pred):
            scores[idx] = score
    if any(score is None for score in scores):
        return None
    return [float(score) for score in scores if score is not None]


def _group_cv_scores_with_extra(rows: list[dict[str, Any]], extra_rows: list[dict[str, Any]], key_fields: tuple[str, ...]) -> tuple[list[float] | None, list[float] | None]:
    if len({int(r["y_actionable_matched_diagnostic"]) for r in rows}) < 2:
        return None, None
    groups: dict[tuple[str, ...], list[int]] = {}
    extra_groups: dict[tuple[str, ...], list[int]] = {}
    for idx, row in enumerate(rows):
        groups.setdefault(tuple(str(row.get(k, "")) for k in key_fields), []).append(idx)
    for idx, row in enumerate(extra_rows):
        extra_groups.setdefault(tuple(str(row.get(k, "")) for k in key_fields), []).append(idx)
    primary_scores: list[float | None] = [None] * len(rows)
    extra_scores: list[float] = []
    for group_key, indices in groups.items():
        held = set(indices)
        train = [row for idx, row in enumerate(rows) if idx not in held]
        extra_train = [row for key, idxs in extra_groups.items() for idx, row in enumerate(extra_rows) if key != group_key and idx in idxs]
        test = [row for idx, row in enumerate(rows) if idx in held]
        pred = _fit_predict(train, test, extra_train)
        if pred is None:
            continue
        for idx, score in zip(indices, pred):
            primary_scores[idx] = score
        extra_indices = extra_groups.get(group_key, [])
        if extra_indices:
            extra_test = [extra_rows[idx] for idx in extra_indices]
            extra_pred = _fit_predict(train, extra_test, extra_train)
            if extra_pred is not None:
                extra_scores.extend(extra_pred)
    if any(score is None for score in primary_scores):
        return None, None
    return [float(score) for score in primary_scores if score is not None], extra_scores if extra_scores else None


def _nearest_future(series: dict[tuple[str, str, str], dict[int, dict[str, Any]]], dataset: str, seed: str, model: str, target_step: int, merge_window: int) -> dict[str, Any] | None:
    steps = series.get((dataset, seed, model), {})
    if target_step in steps:
        return steps[target_step]
    best_step = None
    best_dist = None
    for step in steps:
        dist = abs(step - target_step)
        if dist <= merge_window and (best_dist is None or dist < best_dist):
            best_step = step
            best_dist = dist
    if best_step is None:
        return None
    return steps[best_step]


def _risk_summary(rows: list[dict[str, Any]], horizons: list[int], scope: str, subset_rule: str, neutral_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    neutral_rows = neutral_rows or []
    valid_by_h = {h: [r for r in rows if int(r.get("horizon", -1)) == h and r.get("y_actionable_matched_diagnostic") != ""] for h in horizons}
    neutral_by_h = {h: [r for r in neutral_rows if int(r.get("horizon", -1)) == h] for h in horizons}

    def labels_and_scores(horizon: int) -> tuple[list[int], list[float], list[dict[str, Any]]]:
        rs = valid_by_h.get(horizon, [])
        labels = [int(r["y_actionable_matched_diagnostic"]) for r in rs]
        scores = [finite_float(r.get("U_task_usefulness"), 0.0) for r in rs]
        return labels, scores, rs

    metrics: dict[int, dict[str, Any]] = {}
    trained_any = 0
    for horizon in horizons:
        y_h, u_h, rows_h = labels_and_scores(horizon)
        cv_scores = _group_cv_scores(rows_h, ("dataset", "seed"), neutral_by_h.get(horizon, []))
        if cv_scores is not None:
            trained_any = 1
        scores_for_operating = cv_scores if cv_scores is not None else u_h
        metrics[horizon] = {
            "valid_rows": len(rows_h),
            "positive_label_rate": sum(y_h) / max(1, len(y_h)) if rows_h else "",
            "auc_proxy": _auc(y_h, u_h) if rows_h else "",
            "auc_model": _auc(y_h, cv_scores) if cv_scores is not None else "",
            "precision_top20": _precision_at_top_fraction(y_h, scores_for_operating) if rows_h else "",
            "recall_fpr30": _recall_at_fpr(y_h, scores_for_operating) if rows_h else "",
        }

    primary_h = 100 if 100 in horizons else horizons[0]
    y_vals, u_scores, primary_rows = labels_and_scores(primary_h)
    primary_scores, neutral_scores = _group_cv_scores_with_extra(primary_rows, neutral_by_h.get(primary_h, []), ("dataset", "seed"))
    if primary_scores is None:
        primary_scores = u_scores
    dataset_scores = _group_cv_scores(primary_rows, ("dataset",), neutral_by_h.get(primary_h, []))
    seed_scores = _group_cv_scores(primary_rows, ("seed",), neutral_by_h.get(primary_h, []))
    naive_vals = [int(r.get("y_naive_washout") or 0) for r in primary_rows]
    false_alarm_neutral: float | str = ""
    if primary_scores:
        top_k = max(1, int(math.ceil(len(primary_scores) * 0.20)))
        cutoff = sorted(primary_scores, reverse=True)[top_k - 1]
        if neutral_scores:
            false_alarm_neutral = float(sum(score >= cutoff for score in neutral_scores) / len(neutral_scores))
    positive_leads = [float(r.get("horizon", primary_h)) for r in rows if r.get("y_actionable_matched_diagnostic") != "" and int(r.get("y_actionable_matched_diagnostic") or 0) == 1]
    auc_h100 = metrics.get(100, {}).get("auc_model", "")
    auc_h200 = metrics.get(200, {}).get("auc_model", "")
    cross_dataset_auc = _auc(y_vals, dataset_scores) if dataset_scores is not None else ""
    cross_seed_auc = _auc(y_vals, seed_scores) if seed_scores is not None else ""
    official_pass = int(
        scope == "all"
        and auc_h100 != ""
        and auc_h200 != ""
        and cross_dataset_auc != ""
        and cross_seed_auc != ""
        and float(auc_h100) >= 0.75
        and float(auc_h200) >= 0.70
        and float(cross_dataset_auc) >= 0.65
        and float(cross_seed_auc) >= 0.65
        and false_alarm_neutral != ""
        and float(false_alarm_neutral) <= 0.30
    )
    if not primary_rows:
        blocker = "ActionableRiskInsufficientHorizon"
    elif sum(y_vals) == 0:
        blocker = "ActionableRiskSparse"
    elif trained_any == 0:
        blocker = "RiskPredictorNotTrainable"
    elif official_pass:
        blocker = ""
    else:
        blocker = "RiskPredictorBelowOfficialThreshold"
    return {
        "risk_model": "logistic_actionable_source_features_control_negative_augmented" if trained_any and neutral_rows else ("logistic_actionable_source_features" if trained_any else "source_usefulness_proxy_no_trained_predictor"),
        "risk_scope": scope,
        "subset_rule": subset_rule,
        "horizon": ",".join(str(h) for h in horizons),
        "AUC_actionable_proxy": _auc(y_vals, u_scores) if primary_rows else "",
        "AUC_actionable_H50": metrics.get(50, {}).get("auc_model", ""),
        "AUC_actionable_H100": auc_h100,
        "AUC_actionable_H200": auc_h200,
        "AUC_actionable_H400": metrics.get(400, {}).get("auc_model", ""),
        "AUC_naive_washout_H100": _auc(naive_vals, u_scores) if primary_rows else "",
        "positive_label_rate_actionable": sum(y_vals) / max(1, len(y_vals)) if primary_rows else "",
        "positive_label_rate_actionable_H50": metrics.get(50, {}).get("positive_label_rate", ""),
        "positive_label_rate_actionable_H100": metrics.get(100, {}).get("positive_label_rate", ""),
        "positive_label_rate_actionable_H200": metrics.get(200, {}).get("positive_label_rate", ""),
        "positive_label_rate_actionable_H400": metrics.get(400, {}).get("positive_label_rate", ""),
        "positive_label_rate_naive": sum(naive_vals) / max(1, len(naive_vals)) if primary_rows else "",
        "precision_at_top20_actionable": metrics.get(primary_h, {}).get("precision_top20", ""),
        "recall_at_FPR30_actionable": metrics.get(primary_h, {}).get("recall_fpr30", ""),
        "median_lead_time_actionable": _median(positive_leads),
        "false_alarm_rate_on_stale_source": "",
        "false_alarm_rate_on_task_neutral_source": false_alarm_neutral,
        "task_neutral_eval_rows_H100": len(neutral_by_h.get(100, [])),
        "task_neutral_eval_rows_H200": len(neutral_by_h.get(200, [])),
        "cross_dataset_AUC": cross_dataset_auc,
        "cross_seed_AUC": cross_seed_auc,
        "valid_label_rows_H50": metrics.get(50, {}).get("valid_rows", ""),
        "valid_label_rows_H100": metrics.get(100, {}).get("valid_rows", ""),
        "valid_label_rows_H200": metrics.get(200, {}).get("valid_rows", ""),
        "valid_label_rows_H400": metrics.get(400, {}).get("valid_rows", ""),
        "risk_model_trained": trained_any,
        "official_pass": official_pass,
        "blocker": blocker,
    }


def _intervention_models(base_model: str, all_models: set[str]) -> list[str]:
    if base_model == "DGMLP":
        return sorted(m for m in all_models if m.startswith("DGMLP_FU"))
    if base_model == "DGKAN_DFOU":
        return sorted(m for m in all_models if m.startswith("DGKAN_DFOU_FU"))
    if base_model == "DGKAN_DCHE":
        return sorted(m for m in all_models if m.startswith("DGKAN_DCHE_FU"))
    return []


def _source_usefulness_official_rows(
    source_rows: list[dict[str, Any]],
    neutral_source_rows: list[dict[str, Any]],
    horizons: list[int],
    merge_window: int,
    tau_u_margin: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_series: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = {}
    neutral_series: dict[tuple[str, str, str, str], dict[int, dict[str, Any]]] = {}
    for row in source_rows:
        by_series.setdefault((row.get("dataset", ""), row.get("seed", ""), row.get("model_name", "")), {})[
            int(float(row.get("step", 0) or 0))
        ] = row
    for row in neutral_source_rows:
        neutral_series.setdefault(
            (row.get("dataset", ""), row.get("seed", ""), row.get("model_name", ""), row.get("neutral_control_source_type", "")),
            {},
        )[int(float(row.get("step", 0) or 0))] = row

    def nearest(series: dict[int, dict[str, Any]], target_step: int) -> dict[str, Any] | None:
        if target_step in series:
            return series[target_step]
        best_step = None
        best_dist = None
        for step in series:
            dist = abs(step - target_step)
            if dist <= merge_window and (best_dist is None or dist < best_dist):
                best_step = step
                best_dist = dist
        return series.get(best_step) if best_step is not None else None

    def median(values: list[float]) -> float | str:
        return _median(values)

    source_future_rows: list[dict[str, Any]] = []
    control_future_rows: list[dict[str, Any]] = []
    for row in source_rows:
        series = by_series.get((row.get("dataset", ""), row.get("seed", ""), row.get("model_name", "")), {})
        step = int(float(row.get("step", 0) or 0))
        for horizon in horizons:
            fut = nearest(series, step + int(horizon))
            current_loss = finite_float(row.get("source_loss_t"), 0.0)
            future_loss = finite_float(fut.get("source_loss_t") if fut else "", float("nan"))
            current_tail = finite_float(row.get("TailRisk_proxy"), 0.0)
            future_tail = finite_float(fut.get("TailRisk_proxy") if fut else "", float("nan"))
            has_future = fut is not None and future_loss == future_loss
            source_future_rows.append(
                {
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    "model_family": row.get("model_family", ""),
                    "model_name": row.get("model_name", ""),
                    "source_candidate_type": row.get("source_candidate_type", ""),
                    "step": step,
                    "horizon": horizon,
                    "target_step": step + int(horizon),
                    "merge_window": merge_window,
                    "U_minus_random": finite_float(row.get("U_task_usefulness"), 0.0) - finite_float(row.get("random_control_U"), 0.0),
                    "source_loss_t": current_loss,
                    "source_loss_after_h": "" if not has_future else future_loss,
                    "source_loss_delta_after_h": "" if not has_future else future_loss - current_loss,
                    "source_func_after_h": "" if fut is None else fut.get("source_func_t", ""),
                    "tail_risk_delta_after_h": "" if fut is None else future_tail - current_tail,
                    "source_loss_sign_flip": "" if not has_future else int(current_loss * future_loss < 0.0),
                    "valid_future": int(has_future),
                    "analysis_only_future_label_used_for_runtime_direction": 0,
                }
            )
    for row in neutral_source_rows:
        series = neutral_series.get(
            (row.get("dataset", ""), row.get("seed", ""), row.get("model_name", ""), row.get("neutral_control_source_type", "")),
            {},
        )
        step = int(float(row.get("step", 0) or 0))
        for horizon in horizons:
            fut = nearest(series, step + int(horizon))
            current_loss = finite_float(row.get("source_loss_t"), 0.0)
            future_loss = finite_float(fut.get("source_loss_t") if fut else "", float("nan"))
            current_tail = finite_float(row.get("TailRisk_proxy"), 0.0)
            future_tail = finite_float(fut.get("TailRisk_proxy") if fut else "", float("nan"))
            has_future = fut is not None and future_loss == future_loss
            control_future_rows.append(
                {
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    "model_family": row.get("model_family", ""),
                    "model_name": row.get("model_name", ""),
                    "neutral_control_source_type": row.get("neutral_control_source_type", ""),
                    "step": step,
                    "horizon": horizon,
                    "target_step": step + int(horizon),
                    "merge_window": merge_window,
                    "U_minus_random": finite_float(row.get("U_task_usefulness"), 0.0) - finite_float(row.get("random_control_U"), 0.0),
                    "source_loss_t": current_loss,
                    "source_loss_after_h": "" if not has_future else future_loss,
                    "source_loss_delta_after_h": "" if not has_future else future_loss - current_loss,
                    "tail_risk_delta_after_h": "" if fut is None else future_tail - current_tail,
                    "source_loss_sign_flip": "" if not has_future else int(current_loss * future_loss < 0.0),
                    "valid_future": int(has_future),
                    "analysis_only_future_label_used_for_runtime_direction": 0,
                }
            )

    grouped: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = {}
    control_grouped: dict[tuple[str, str, str, str, str, int], list[dict[str, Any]]] = {}
    for row in source_future_rows:
        if int(row.get("valid_future", 0)):
            grouped.setdefault(
                (row.get("model_family", ""), row.get("model_name", ""), row.get("dataset", ""), row.get("seed", ""), int(row.get("horizon", 0) or 0)),
                [],
            ).append(row)
    for row in control_future_rows:
        if int(row.get("valid_future", 0)):
            control_grouped.setdefault(
                (
                    row.get("model_family", ""),
                    row.get("model_name", ""),
                    row.get("neutral_control_source_type", ""),
                    row.get("dataset", ""),
                    row.get("seed", ""),
                    int(row.get("horizon", 0) or 0),
                ),
                [],
            ).append(row)

    group_rows: list[dict[str, Any]] = []
    control_group_rows: list[dict[str, Any]] = []
    for (family, model, dataset, seed, horizon), rows in sorted(grouped.items()):
        margins = [finite_float(r.get("U_minus_random"), 0.0) for r in rows]
        future_losses = [finite_float(r.get("source_loss_after_h"), float("nan")) for r in rows]
        tail_deltas = [finite_float(r.get("tail_risk_delta_after_h"), float("nan")) for r in rows]
        flips = [int(r.get("source_loss_sign_flip") or 0) for r in rows]
        future_losses = [v for v in future_losses if v == v]
        tail_deltas = [v for v in tail_deltas if v == v]
        group_pass = int(
            bool(rows)
            and median(margins) != ""
            and float(median(margins)) > float(tau_u_margin)
            and future_losses
            and float(median(future_losses)) >= 0.0
            and tail_deltas
            and float(median(tail_deltas)) <= 0.0
            and (sum(flips) / max(1, len(flips))) <= 0.30
        )
        group_rows.append(
            {
                "model_family": family,
                "model_name": model,
                "dataset": dataset,
                "seed": seed,
                "horizon": horizon,
                "valid_rows": len(rows),
                "median_U_minus_random": median(margins),
                "median_source_loss_after_h": median(future_losses),
                "median_tail_risk_delta_after_h": median(tail_deltas),
                "source_loss_flip_rate": sum(flips) / max(1, len(flips)),
                "group_pass": group_pass,
            }
        )
    for (family, model, control_type, dataset, seed, horizon), rows in sorted(control_grouped.items()):
        margins = [finite_float(r.get("U_minus_random"), 0.0) for r in rows]
        future_losses = [finite_float(r.get("source_loss_after_h"), float("nan")) for r in rows]
        tail_deltas = [finite_float(r.get("tail_risk_delta_after_h"), float("nan")) for r in rows]
        flips = [int(r.get("source_loss_sign_flip") or 0) for r in rows]
        future_losses = [v for v in future_losses if v == v]
        tail_deltas = [v for v in tail_deltas if v == v]
        group_pass = int(
            bool(rows)
            and median(margins) != ""
            and float(median(margins)) > float(tau_u_margin)
            and future_losses
            and float(median(future_losses)) >= 0.0
            and tail_deltas
            and float(median(tail_deltas)) <= 0.0
            and (sum(flips) / max(1, len(flips))) <= 0.30
        )
        control_group_rows.append(
            {
                "model_family": family,
                "model_name": model,
                "neutral_control_source_type": control_type,
                "dataset": dataset,
                "seed": seed,
                "horizon": horizon,
                "valid_rows": len(rows),
                "median_U_minus_random": median(margins),
                "median_source_loss_after_h": median(future_losses),
                "median_tail_risk_delta_after_h": median(tail_deltas),
                "source_loss_flip_rate": sum(flips) / max(1, len(flips)),
                "control_group_pass": group_pass,
            }
        )

    official_rows: list[dict[str, Any]] = []
    for family, model in sorted({(r.get("model_family", ""), r.get("model_name", "")) for r in group_rows}):
        source_h200 = [r for r in group_rows if r.get("model_family") == family and r.get("model_name") == model and int(r.get("horizon", 0) or 0) == 200]
        controls_h200 = [
            r
            for r in control_group_rows
            if r.get("model_family") == family and r.get("model_name") == model and int(r.get("horizon", 0) or 0) == 200
        ]
        pass_groups = [r for r in source_h200 if int(r.get("group_pass", 0))]
        pass_datasets = {r.get("dataset", "") for r in pass_groups}
        pass_seeds = {r.get("seed", "") for r in pass_groups}
        control_pass_groups = [r for r in controls_h200 if int(r.get("control_group_pass", 0))]
        controls_fail = int(len(control_pass_groups) == 0)
        official = int(family == "MLP" and len(pass_groups) >= 4 and len(pass_datasets) >= 2 and len(pass_seeds) >= 2 and controls_fail)
        official_rows.append(
            {
                "model_family": family,
                "model_name": model,
                "horizon": 200,
                "source_group_pass_count": len(pass_groups),
                "source_group_total": len(source_h200),
                "pass_dataset_count": len(pass_datasets),
                "pass_seed_count": len(pass_seeds),
                "control_group_pass_count": len(control_pass_groups),
                "controls_fail": controls_fail,
                "official_source_usefulness_pass": official,
                "pass_note": "official_B_pass_MLP_H200_controls_fail" if official else "B_below_official_or_non_MLP_or_controls_not_failed",
            }
        )
    return source_future_rows, group_rows, control_group_rows, official_rows


def main() -> None:
    args = parser().parse_args()
    ensure_out()
    horizons = [int(x.strip()) for x in str(args.horizons or args.horizon).split(",") if x.strip()]
    if not horizons:
        horizons = [int(args.horizon)]
    source_rows = read_rows(OUT_ROOT / "v22_17_task_useful_source_matrix.csv")
    neutral_source_rows = read_rows(OUT_ROOT / "v22_17_task_neutral_control_source_matrix.csv")
    labels: list[dict[str, Any]] = []
    neutral_eval_rows: list[dict[str, Any]] = []
    counter: list[dict[str, Any]] = []
    by_series: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = {}
    for r in source_rows:
        by_series.setdefault((r.get("dataset", ""), r.get("seed", ""), r.get("model_name", "")), {})[
            int(float(r.get("step", 0) or 0))
        ] = r
    source_future_rows, source_group_rows, source_control_group_rows, source_official_rows = _source_usefulness_official_rows(
        source_rows,
        neutral_source_rows,
        horizons,
        int(args.merge_window),
        float(args.tau_u_margin),
    )
    official_by_model = {
        (r.get("model_family", ""), r.get("model_name", "")): r
        for r in source_official_rows
    }
    all_models = {r.get("model_name", "") for r in source_rows}
    for row in neutral_source_rows:
        base_model = row.get("model_name", "")
        interventions = _intervention_models(base_model, all_models)
        if not interventions:
            continue
        for intervention_model in interventions:
            for horizon in horizons:
                neutral_eval_rows.append(
                    {
                        **row,
                        "base_model": base_model,
                        "intervention_model": intervention_model,
                        "horizon": horizon,
                        "target_step": int(float(row.get("step", 0) or 0)) + int(horizon),
                        "merge_window": args.merge_window,
                        "y_actionable_matched_diagnostic": 0,
                        "y_naive_washout": 0,
                        "official_actionable_label": 0,
                        "task_neutral_eval_only": 1,
                        "analysis_only_future_label_used_for_runtime_direction": 0,
                    }
                )
    for row in source_rows:
        base_model = row.get("model_name", "")
        interventions = _intervention_models(base_model, all_models)
        if not interventions:
            continue
        dataset = row.get("dataset", "")
        seed = row.get("seed", "")
        step = int(float(row.get("step", 0) or 0))
        for intervention_model in interventions:
            for horizon in horizons:
                target_step = step + int(horizon)
                base_future = _nearest_future(by_series, dataset, seed, base_model, target_step, int(args.merge_window))
                fu_future = _nearest_future(by_series, dataset, seed, intervention_model, target_step, int(args.merge_window))
                current_loss = finite_float(row.get("source_loss_t"), 0.0)
                base_future_loss = finite_float(base_future.get("source_loss_t") if base_future else "", float("nan"))
                fu_future_loss = finite_float(fu_future.get("source_loss_t") if fu_future else "", float("nan"))
                u_value = finite_float(row.get("U_task_usefulness"), 0.0)
                random_u = finite_float(row.get("random_control_U"), 0.0)
                has_future = int(base_future is not None and fu_future is not None)
                y = ""
                y_naive = ""
                reason = ""
                if has_future:
                    useful = u_value > random_u + float(args.tau_u_margin)
                    base_destroyed = (base_future_loss - current_loss) < -float(args.tau_l)
                    intervention_helped = (fu_future_loss - base_future_loss) > float(args.tau_i)
                    y = int(useful and base_destroyed and intervention_helped)
                    y_naive = int(useful and base_destroyed)
                else:
                    reason = "insufficient_future_window_for_matched_intervention"
                enriched = {
                    **row,
                    "dataset": dataset,
                    "seed": seed,
                    "model_family": row.get("model_family", ""),
                    "base_model": base_model,
                    "intervention_model": intervention_model,
                    "step": step,
                    "horizon": horizon,
                    "target_step": target_step,
                    "merge_window": args.merge_window,
                    "U_task_usefulness": u_value,
                    "random_control_U": random_u,
                    "source_loss_t": current_loss,
                    "source_loss_base_future": "" if not has_future else base_future_loss,
                    "source_loss_intervention_future": "" if not has_future else fu_future_loss,
                    "base_future_minus_current": "" if not has_future else base_future_loss - current_loss,
                    "intervention_minus_base_future": "" if not has_future else fu_future_loss - base_future_loss,
                    "y_actionable_matched_diagnostic": y,
                    "y_naive_washout": y_naive,
                    "official_actionable_label": int(has_future),
                    "analysis_only_future_label_used_for_runtime_direction": 0,
                    "deferred_reason": reason,
                }
                labels.append(enriched)
                counter.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "model_family": row.get("model_family", ""),
                        "base_model": base_model,
                        "intervention_model": intervention_model,
                        "step": step,
                        "horizon": horizon,
                        "target_step": target_step,
                        "merge_window": args.merge_window,
                        "matched_intervention_available": has_future,
                        "base_future_minus_current": "" if not has_future else base_future_loss - current_loss,
                        "intervention_minus_base_future": "" if not has_future else fu_future_loss - base_future_loss,
                        "diagnostic_only": 1,
                    }
                )

    valid = [r for r in labels if r.get("y_actionable_matched_diagnostic") != ""]
    risk_rows = [_risk_summary(valid, horizons, "all", "all_valid_matched_labels", neutral_eval_rows)]
    for family in sorted({r.get("model_family", "") for r in valid if r.get("model_family", "")}):
        risk_rows.append(
            _risk_summary(
                [r for r in valid if r.get("model_family") == family],
                horizons,
                f"family:{family}",
                f"model_family == {family}",
                [r for r in neutral_eval_rows if r.get("model_family") == family],
            )
        )
    if valid:
        margins = [finite_float(r.get("U_task_usefulness"), 0.0) - finite_float(r.get("random_control_U"), 0.0) for r in valid]
        for quantile in (0.50, 0.70, 0.90):
            threshold = sorted(margins)[int((len(margins) - 1) * quantile)]
            subset = [r for r in valid if finite_float(r.get("U_task_usefulness"), 0.0) - finite_float(r.get("random_control_U"), 0.0) >= threshold]
            neutral_subset = [r for r in neutral_eval_rows if finite_float(r.get("U_task_usefulness"), 0.0) - finite_float(r.get("random_control_U"), 0.0) >= threshold]
            risk_rows.append(_risk_summary(subset, horizons, f"highU:q{int(quantile * 100)}", f"U-random >= {threshold}", neutral_subset))

    condition_rows: list[dict[str, Any]] = []
    groups_cond: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in valid:
        groups_cond.setdefault(
            (row.get("horizon", ""), row.get("model_family", ""), row.get("base_model", ""), row.get("intervention_model", "")),
            [],
        ).append(row)
    for (horizon, family, base, intervention), rows in sorted(groups_cond.items()):
        useful = [
            finite_float(r.get("U_task_usefulness"), 0.0) > finite_float(r.get("random_control_U"), 0.0) + float(args.tau_u_margin)
            for r in rows
        ]
        destroyed = [finite_float(r.get("base_future_minus_current"), 0.0) < -float(args.tau_l) for r in rows]
        helped = [finite_float(r.get("intervention_minus_base_future"), 0.0) > float(args.tau_i) for r in rows]
        labels_pos = [int(r.get("y_actionable_matched_diagnostic") or 0) for r in rows]
        condition_rows.append(
            {
                "horizon": horizon,
                "model_family": family,
                "base_model": base,
                "intervention_model": intervention,
                "rows": len(rows),
                "useful_count": sum(int(x) for x in useful),
                "base_destroyed_count": sum(int(x) for x in destroyed),
                "intervention_helped_count": sum(int(x) for x in helped),
                "useful_and_destroyed_count": sum(int(a and b) for a, b in zip(useful, destroyed)),
                "actionable_positive_count": sum(labels_pos),
                "actionable_positive_rate": sum(labels_pos) / max(1, len(labels_pos)),
                "mean_U_minus_random": sum(finite_float(r.get("U_task_usefulness"), 0.0) - finite_float(r.get("random_control_U"), 0.0) for r in rows) / max(1, len(rows)),
                "mean_base_future_minus_current": sum(finite_float(r.get("base_future_minus_current"), 0.0) for r in rows) / max(1, len(rows)),
                "mean_intervention_minus_base_future": sum(finite_float(r.get("intervention_minus_base_future"), 0.0) for r in rows) / max(1, len(rows)),
            }
        )
    ablation: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in source_rows:
        groups.setdefault((row.get("model_family", ""), row.get("model_name", "")), []).append(row)
    for (family, model), rows in groups.items():
        u = [finite_float(r.get("U_task_usefulness"), 0.0) for r in rows]
        rnd = [finite_float(r.get("random_control_U"), 0.0) for r in rows]
        sign = [finite_float(r.get("signflip_U"), 0.0) for r in rows]
        official = official_by_model.get((family, model), {})
        ablation.append(
            {
                "model_family": family,
                "model_name": model,
                "rows": len(rows),
                "mean_U_task_usefulness": sum(u) / max(1, len(u)),
                "mean_random_control_U": sum(rnd) / max(1, len(rnd)),
                "mean_signflip_U": sum(sign) / max(1, len(sign)),
                "U_beats_random_fraction": sum(int(a > b) for a, b in zip(u, rnd)) / max(1, len(u)),
                "source_group_pass_count_H200": official.get("source_group_pass_count", ""),
                "source_group_total_H200": official.get("source_group_total", ""),
                "pass_dataset_count_H200": official.get("pass_dataset_count", ""),
                "pass_seed_count_H200": official.get("pass_seed_count", ""),
                "control_group_pass_count_H200": official.get("control_group_pass_count", ""),
                "controls_fail_H200": official.get("controls_fail", ""),
                "official_source_usefulness_pass": official.get("official_source_usefulness_pass", 0),
                "pass_note": official.get("pass_note", "B_official_matrix_missing"),
            }
        )

    write_rows(OUT_ROOT / "v22_17_actionable_washout_labels.csv", labels)
    write_rows(OUT_ROOT / "v22_17_counterfactual_intervention_diagnostic.csv", counter)
    write_rows(OUT_ROOT / "v22_17_risk_prediction_matrix.csv", risk_rows)
    write_rows(OUT_ROOT / "v22_17_risk_feature_ablation.csv", risk_rows)
    write_rows(OUT_ROOT / "v22_17_actionable_label_condition_matrix.csv", condition_rows)
    write_rows(OUT_ROOT / "v22_17_task_neutral_control_eval_matrix.csv", neutral_eval_rows)
    write_rows(OUT_ROOT / "v22_17_source_loss_boundary_matrix.csv", source_rows)
    write_rows(OUT_ROOT / "v22_17_source_usefulness_future_matrix.csv", source_future_rows)
    write_rows(OUT_ROOT / "v22_17_source_usefulness_group_matrix.csv", source_group_rows)
    write_rows(OUT_ROOT / "v22_17_source_usefulness_control_group_matrix.csv", source_control_group_rows)
    write_rows(OUT_ROOT / "v22_17_source_usefulness_official_matrix.csv", source_official_rows)
    write_rows(OUT_ROOT / "v22_17_source_candidate_ablation.csv", ablation)
    cmd = f"{sys.executable} experiments/run_v22_17_mechanism_smoke.py --horizons {','.join(str(h) for h in horizons)} --merge-window {args.merge_window} --tau-u-margin {args.tau_u_margin} --tau-l {args.tau_l} --tau-i {args.tau_i}"
    append_exec(
        cmd,
        task_id="B-C-D-mechanism-postprocess",
        status="pass" if source_rows else "blocked",
        gpu="1",
        exit_code=0,
        files=(
            "results/v22_17/v22_17_actionable_washout_labels.csv, "
            "results/v22_17/v22_17_risk_prediction_matrix.csv, "
            "results/v22_17/v22_17_actionable_label_condition_matrix.csv, "
            "results/v22_17/v22_17_task_neutral_control_eval_matrix.csv, "
            "results/v22_17/v22_17_source_usefulness_official_matrix.csv, "
            "results/v22_17/v22_17_source_candidate_ablation.csv"
        ),
        note="matched-run labels are offline analysis only; future labels are not used as runtime direction",
    )


if __name__ == "__main__":
    main()
