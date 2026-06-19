#!/usr/bin/env python3
"""Train and validate v22.18 offline benefit-conditioned policies."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import shlex
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.fu.benefit_policy import fit_ridge, safe_float, validation_metrics  # noqa: E402
from dgkan.fu.source_action_bank import RUNTIME_FEATURES  # noqa: E402
from experiments.run_v22_18_common import OUT_ROOT, append_exec, ensure_out, read_rows, write_rows  # noqa: E402
from experiments.run_v22_18_common import write_json  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--features-csv", default="results/v22_18/v22_18_action_features_runtime_only.csv")
    p.add_argument("--labels-csv", default="results/v22_18/v22_18_action_benefit_labels_H50_H100_H200.csv")
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--accept-threshold", type=float, default=0.0)
    p.add_argument(
        "--target-mode",
        choices=[
            "benefit_utility",
            "benefit_control_penalty",
            "benefit_positive_binary",
            "true_improvement",
            "benefit_reweighted_components",
        ],
        default="benefit_utility",
    )
    p.add_argument("--positive-label", choices=["benefit_positive", "true_improvement_positive"], default="benefit_positive")
    p.add_argument("--control-penalty", type=float, default=0.0)
    p.add_argument("--weight-nll", type=float, default=1.0)
    p.add_argument("--weight-auc", type=float, default=0.20)
    p.add_argument("--weight-source", type=float, default=0.05)
    p.add_argument("--weight-tail", type=float, default=0.10)
    p.add_argument("--weight-ece", type=float, default=0.10)
    p.add_argument("--weight-cost", type=float, default=0.05)
    p.add_argument("--feature-set", choices=["full", "no_action_numeric"], default="full")
    p.add_argument("--ridge", type=float, default=1.0e-3)
    p.add_argument("--output-suffix", default="")
    return p


def _path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def _out(name: str, suffix: str) -> Path:
    clean = suffix.strip().strip("_")
    path = OUT_ROOT / name
    if clean:
        return path.with_name(f"{path.stem}_{clean}{path.suffix}")
    return path


def _key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (row.get("dataset", ""), str(row.get("seed", "")), row.get("model_name", ""), row.get("action_id", ""))


def _joined(features: list[dict[str, str]], labels: list[dict[str, str]], horizon: int) -> list[dict[str, Any]]:
    by_key = {_key(row): row for row in features}
    rows: list[dict[str, Any]] = []
    for label in labels:
        if int(float(label.get("horizon") or 0)) != int(horizon):
            continue
        feat = by_key.get(_key(label))
        if not feat:
            continue
        rows.append({**feat, **{f"label_{k}": v for k, v in label.items()}})
    return rows


def _eval_split(
    rows: list[dict[str, Any]],
    hold_field: str,
    hold_value: str,
    *,
    target_mode: str,
    positive_label: str,
    control_penalty: float,
    weights: dict[str, float],
    ridge: float,
    feature_names: list[str],
) -> dict[str, Any]:
    train = [r for r in rows if str(r.get(hold_field, "")) != str(hold_value)]
    test = [r for r in rows if str(r.get(hold_field, "")) == str(hold_value)]
    y_train = [_target_value(r, target_mode, control_penalty, weights) for r in train]
    y_test = [_target_value(r, target_mode, control_penalty, weights) for r in test]
    pos_test = [_positive_value(r, positive_label) for r in test]
    policy = fit_ridge(train, y_train, feature_names, ridge=float(ridge))
    metrics = validation_metrics(test, y_test, pos_test, policy)
    metrics.update({"split": f"leave-one-{hold_field}", "held_out": hold_value, "train_rows": len(train), "test_rows": len(test)})
    return metrics


def _target_value(row: dict[str, Any], target_mode: str, control_penalty: float, weights: dict[str, float] | None = None) -> float:
    if target_mode == "true_improvement":
        return safe_float(row.get("label_true_improvement_positive"))
    if target_mode == "benefit_positive_binary":
        value = safe_float(row.get("label_benefit_positive"))
        if int(float(row.get("action_is_control") or 0)):
            value -= float(control_penalty)
        return value
    if target_mode == "benefit_reweighted_components":
        w = weights or {}
        value = (
            -float(w.get("nll", 1.0)) * safe_float(row.get("label_delta_NLL"))
            - float(w.get("auc", 0.20)) * safe_float(row.get("label_delta_AUC_loss_time"))
            + float(w.get("source", 0.05)) * safe_float(row.get("label_delta_source_loss"))
            - float(w.get("tail", 0.10)) * safe_float(row.get("label_delta_tail_q99"))
            - float(w.get("ece", 0.10)) * safe_float(row.get("label_delta_ECE"))
            - float(w.get("cost", 0.05)) * safe_float(row.get("label_controller_cost"))
        )
    else:
        value = safe_float(row.get("label_benefit_utility"))
    if target_mode in {"benefit_control_penalty", "benefit_reweighted_components"} and int(float(row.get("action_is_control") or 0)):
        value -= float(control_penalty)
    return value


def _positive_value(row: dict[str, Any], positive_label: str) -> int:
    return int(float(row.get(f"label_{positive_label}") or 0))


def _feature_names(feature_set: str) -> list[str]:
    if feature_set == "no_action_numeric":
        return [name for name in RUNTIME_FEATURES if name != "action_family_numeric"]
    return list(RUNTIME_FEATURES)


def main() -> None:
    args = parser().parse_args()
    ensure_out()
    features = read_rows(_path(args.features_csv))
    labels = read_rows(_path(args.labels_csv))
    rows = _joined(features, labels, int(args.horizon))
    feature_names = _feature_names(args.feature_set)
    weights = {
        "nll": float(args.weight_nll),
        "auc": float(args.weight_auc),
        "source": float(args.weight_source),
        "tail": float(args.weight_tail),
        "ece": float(args.weight_ece),
        "cost": float(args.weight_cost),
    }
    y = [_target_value(r, args.target_mode, float(args.control_penalty), weights) for r in rows]
    positive = [_positive_value(r, args.positive_label) for r in rows]
    policy = fit_ridge(rows, y, feature_names, ridge=float(args.ridge))
    scores = policy.predict(rows)

    policy_rows: list[dict[str, Any]] = []
    for row, score in zip(rows, scores):
        policy_rows.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "model_name": row.get("model_name", ""),
                "action_id": row.get("action_id", ""),
                "action_name": row.get("action_name", ""),
                "action_is_control": row.get("action_is_control", ""),
                "policy_variant": f"P3 benefit-risk ridge variant selector / {args.target_mode}",
                "predicted_benefit": score,
                "target_value_used_for_fit": _target_value(row, args.target_mode, float(args.control_penalty), weights),
                "realized_benefit_utility_H": row.get("label_benefit_utility", ""),
                "benefit_positive": row.get("label_benefit_positive", ""),
                "true_improvement_positive": row.get("label_true_improvement_positive", ""),
                "delta_NLL": row.get("label_delta_NLL", ""),
                "delta_AUC_loss_time": row.get("label_delta_AUC_loss_time", ""),
                "runtime_per_step_policy_integrated": 0,
                "offline_variant_selector_only": 1,
            }
        )
    base_metrics = validation_metrics(rows, y, positive, policy)
    threshold = float(args.accept_threshold)
    control_rows = [r for r, score in zip(rows, scores) if int(float(r.get("action_is_control") or 0)) and score > threshold]
    all_control_rows = [r for r in rows if int(float(r.get("action_is_control") or 0))]
    action_dist = Counter(r.get("action_id", "") for r, score in zip(rows, scores) if score > threshold)
    validation_rows: list[dict[str, Any]] = [
        {
            "split": "all-data-in-sample",
            "held_out": "",
            **base_metrics,
            "false_positive_benefit_rate_controls": len(control_rows) / max(1, len(all_control_rows)),
            "policy_noop_rate": sum(1 for score in scores if score <= threshold) / max(1, len(scores)),
            "policy_action_accept_rate": sum(1 for score in scores if score > threshold) / max(1, len(scores)),
            "policy_entropy_proxy_distinct_positive_actions": len(action_dist),
            "accept_threshold": threshold,
            "runtime_overhead_from_policy_table_only": 0.0,
            "runtime_per_step_policy_integrated": 0,
            "target_mode": args.target_mode,
            "positive_label": args.positive_label,
            "control_penalty": args.control_penalty,
            "weight_nll": args.weight_nll,
            "weight_auc": args.weight_auc,
            "weight_source": args.weight_source,
            "weight_tail": args.weight_tail,
            "weight_ece": args.weight_ece,
            "weight_cost": args.weight_cost,
            "feature_set": args.feature_set,
            "ridge": args.ridge,
        }
    ]
    for dataset in sorted({r.get("dataset", "") for r in rows}):
        if dataset:
            validation_rows.append(
                _eval_split(
                    rows,
                    "dataset",
                    dataset,
                    target_mode=args.target_mode,
                    positive_label=args.positive_label,
                    control_penalty=float(args.control_penalty),
                    weights=weights,
                    ridge=float(args.ridge),
                    feature_names=feature_names,
                )
            )
    for seed in sorted({str(r.get("seed", "")) for r in rows}):
        if seed:
            validation_rows.append(
                _eval_split(
                    rows,
                    "seed",
                    seed,
                    target_mode=args.target_mode,
                    positive_label=args.positive_label,
                    control_penalty=float(args.control_penalty),
                    weights=weights,
                    ridge=float(args.ridge),
                    feature_names=feature_names,
                )
            )

    leave_dataset = [r for r in validation_rows if r.get("split") == "leave-one-dataset"]
    leave_seed = [r for r in validation_rows if r.get("split") == "leave-one-seed"]
    c_exploration = int(
        base_metrics["benefit_AUC_binary_positive"] >= 0.65
        and base_metrics["benefit_spearman"] >= 0.25
        and validation_rows[0]["false_positive_benefit_rate_controls"] <= 0.30
        and 0.20 <= validation_rows[0]["policy_noop_rate"] <= 0.90
    )
    c_official = int(
        leave_dataset
        and leave_seed
        and min(float(r.get("benefit_AUC_binary_positive") or 0.0) for r in leave_dataset) >= 0.62
        and min(float(r.get("benefit_AUC_binary_positive") or 0.0) for r in leave_seed) >= 0.62
        and validation_rows[0]["runtime_per_step_policy_integrated"] == 1
    )
    validation_rows[0]["C_exploration_pass"] = c_exploration
    validation_rows[0]["C_official_pass"] = c_official
    validation_rows[0]["official_blocker"] = (
        "offline variant selector only; not integrated as runtime per-step benefit policy"
        if not c_official
        else ""
    )
    min_leave_dataset_auc = min((float(r.get("benefit_AUC_binary_positive") or 0.0) for r in leave_dataset), default=0.0)
    min_leave_seed_auc = min((float(r.get("benefit_AUC_binary_positive") or 0.0) for r in leave_seed), default=0.0)
    policy_export = {
        "policy_variant": f"P3 benefit-risk ridge variant selector / {args.target_mode}",
        "feature_names": list(policy.feature_names),
        "coef": [float(x) for x in policy.coef.tolist()],
        "mean": [float(x) for x in policy.mean.tolist()],
        "scale": [float(x) for x in policy.scale.tolist()],
        "intercept": float(policy.intercept),
        "horizon": int(args.horizon),
        "target_mode": args.target_mode,
        "positive_label": args.positive_label,
        "control_penalty": float(args.control_penalty),
        "weights": weights,
        "feature_set": args.feature_set,
        "ridge": float(args.ridge),
        "accept_threshold": threshold,
        "training_rows": len(rows),
        "C_exploration_pass": int(c_exploration),
        "C_official_pass": int(c_official),
        "min_leave_one_dataset_auc": min_leave_dataset_auc,
        "min_leave_one_seed_auc": min_leave_seed_auc,
        "runtime_per_step_policy_integrated": 0,
        "offline_variant_selector_only": 1,
    }

    write_rows(_out("v22_18_benefit_policy_matrix.csv", args.output_suffix), policy_rows)
    write_rows(_out("v22_18_policy_offline_validation.csv", args.output_suffix), validation_rows)
    export_out = _out("v22_18_benefit_policy_export.json", args.output_suffix)
    write_json(export_out, policy_export)
    write_rows(
        _out("v22_18_policy_action_distribution.csv", args.output_suffix),
        [{"action_id": action_id, "predicted_positive_count": count} for action_id, count in sorted(action_dist.items())],
    )
    cmd = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    append_exec(
        cmd,
        task_id="C-benefit-policy",
        status="pass" if c_exploration else "partial",
        exit_code=0,
        files=(
            f"{_out('v22_18_benefit_policy_matrix.csv', args.output_suffix).relative_to(ROOT)}, "
            f"{_out('v22_18_policy_offline_validation.csv', args.output_suffix).relative_to(ROOT)}, "
            f"{export_out.relative_to(ROOT)}"
        ),
        note=(
            f"rows={len(rows)}; C_exploration_pass={c_exploration}; C_official_pass={c_official}; "
            f"min_lodo_auc={min_leave_dataset_auc:.6g}; min_loseed_auc={min_leave_seed_auc:.6g}"
        ),
    )


if __name__ == "__main__":
    main()
