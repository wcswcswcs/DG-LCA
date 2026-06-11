#!/usr/bin/env python3
"""v22 F46 train-only selector held-out validation.

The C4 observability audit can find train-only diagnostics that separate
row-level source-chain outcomes. This script checks whether those diagnostics
hold out across phase-reset repair families and whether selected rows form a
grouped continuous/h4800 source-chain candidate. It is offline-only and does
not promote by itself.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.fu.source_chain import RETENTION_EPS, classify_source_chain  # noqa: E402
from experiments.run_v22_common import (  # noqa: E402
    PYTHON,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    read_rows,
    write_json,
    write_rows,
)


DEFAULT_FIT_PREFIXES = (
    "v2200_f40_",
    "v2200_f41_",
    "v2200_f42_",
    "v2200_f43_",
)
DEFAULT_HOLDOUT_PREFIXES = (
    "v2200_f44_",
    "v2200_f45_",
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--artifact-prefix", default="v22_f46_train_loss_selector")
    p.add_argument("--fit-prefixes", nargs="*", default=list(DEFAULT_FIT_PREFIXES))
    p.add_argument("--holdout-prefixes", nargs="*", default=list(DEFAULT_HOLDOUT_PREFIXES))
    return p


def is_candidate(row: dict[str, Any]) -> bool:
    return str(row.get("v22_id", "")).startswith(("F", "KSW", "MLP"))


def score(row: dict[str, Any], predictor: str, direction: str) -> float:
    if predictor == "train_loss_drop_h100_h800":
        value = finite_float(row.get("train_loss_h100")) - finite_float(row.get("train_loss_h800"))
    else:
        value = finite_float(row.get(predictor))
    if not math.isfinite(value):
        return float("nan")
    return -value if direction == "higher_bad" else value


def raw_threshold(score_threshold: float, direction: str) -> float:
    return -score_threshold if direction == "higher_bad" else score_threshold


def best_threshold(rows: list[dict[str, Any]], predictor: str, direction: str, label: str) -> dict[str, Any]:
    pairs = sorted(
        (score(row, predictor, direction), int_flag(row.get(label)))
        for row in rows
        if math.isfinite(score(row, predictor, direction))
    )
    if not pairs:
        return {
            "threshold_score": "",
            "threshold_raw": "",
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "precision": "",
            "recall": "",
            "f1": "",
            "youden": "",
        }
    best: tuple[float, float, int, int, int, int] | None = None
    for threshold in sorted({s for s, _y in pairs}):
        tp = sum(1 for s, y in pairs if s >= threshold and y)
        fp = sum(1 for s, y in pairs if s >= threshold and not y)
        fn = sum(1 for s, y in pairs if s < threshold and y)
        tn = sum(1 for s, y in pairs if s < threshold and not y)
        tpr = tp / (tp + fn) if tp + fn else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        youden = tpr - fpr
        if best is None or youden > best[1]:
            best = (threshold, youden, tp, fp, fn, tn)
    assert best is not None
    threshold, youden, tp, fp, fn, tn = best
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "threshold_score": threshold,
        "threshold_raw": raw_threshold(threshold, direction),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "youden": youden,
    }


def selected(rows: list[dict[str, Any]], predictor: str, direction: str, threshold: float) -> list[dict[str, Any]]:
    return [row for row in rows if math.isfinite(score(row, predictor, direction)) and score(row, predictor, direction) >= threshold]


def mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [finite_float(row.get(key)) for row in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def summarize_split(split: str, predictor: str, rows: list[dict[str, Any]], sel: list[dict[str, Any]]) -> dict[str, Any]:
    h = {f"h{step}": mean(sel, f"source_h{step}") for step in (100, 400, 800, 1600, 3200, 4800)}
    decision = classify_source_chain(h["h100"], h["h400"], h["h800"], h["h1600"], h["h3200"], h["h4800"])
    return {
        "split": split,
        "predictor": predictor,
        "rows": len(rows),
        "early_rows_all": sum(int_flag(row.get("early_source_chain")) for row in rows),
        "continuous_rows_all": sum(int_flag(row.get("continuous_retention_chain")) for row in rows),
        "selected_rows": len(sel),
        "selected_early_rows": sum(int_flag(row.get("early_source_chain")) for row in sel),
        "selected_continuous_rows": sum(int_flag(row.get("continuous_retention_chain")) for row in sel),
        "selected_h100": h["h100"],
        "selected_h400": h["h400"],
        "selected_h800": h["h800"],
        "selected_h1600": h["h1600"],
        "selected_h3200": h["h3200"],
        "selected_h4800": h["h4800"],
        "selected_h1600_retention_ratio": decision.h1600_retention_ratio,
        "selected_h3200_retention_ratio": decision.h3200_retention_ratio,
        "selected_h4800_retention_ratio": decision.h4800_retention_ratio,
        "selected_group_early_chain": decision.early_source_chain,
        "selected_group_continuous": decision.continuous_retention_chain,
        "selected_group_h4800": int(decision.continuous_retention_chain and math.isfinite(h["h4800"]) and h["h4800"] >= RETENTION_EPS),
        "selected_group_blocker": decision.blocker,
    }


def summarize_groups(split: str, predictor: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("carrier", "")), str(row.get("v22_id", "")))].append(row)
    out = []
    for (carrier, v22_id), group in sorted(grouped.items()):
        h = {f"h{step}": mean(group, f"source_h{step}") for step in (100, 400, 800, 1600, 3200, 4800)}
        decision = classify_source_chain(h["h100"], h["h400"], h["h800"], h["h1600"], h["h3200"], h["h4800"])
        out.append(
            {
                "split": split,
                "predictor": predictor,
                "carrier": carrier,
                "v22_id": v22_id,
                "rows": len(group),
                "early_rows": sum(int_flag(row.get("early_source_chain")) for row in group),
                "continuous_rows": sum(int_flag(row.get("continuous_retention_chain")) for row in group),
                **h,
                "h1600_retention_ratio": decision.h1600_retention_ratio,
                "h3200_retention_ratio": decision.h3200_retention_ratio,
                "h4800_retention_ratio": decision.h4800_retention_ratio,
                "early_source_chain_group": decision.early_source_chain,
                "continuous_retention_group": decision.continuous_retention_chain,
                "productive_h4800_group": int(decision.continuous_retention_chain and math.isfinite(h["h4800"]) and h["h4800"] >= RETENTION_EPS),
                "source_chain_blocker": decision.blocker,
            }
        )
    return out


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    cmd = (
        f"{PYTHON} experiments/run_v22_f46_train_loss_selector_holdout.py "
        f"--artifact-prefix {args.artifact_prefix}"
    )
    append_exec(out_dir, cmd, status="started")

    matrix = read_rows(out_dir / "v22_source_chain_matrix.csv")
    scan = read_rows(out_dir / "v22_source_observability_predictor_scan.csv")
    passing = [
        row
        for row in scan
        if str(row.get("kind", "")) == "train_only"
        and int_flag(row.get("pass"))
        and str(row.get("predictor", "")).startswith("train_loss")
    ]
    direction_by_predictor = {
        "train_loss_h100": "higher_bad",
        "train_loss_h400": "higher_bad",
        "train_loss_h800": "higher_bad",
        "train_loss_drop_h100_h800": "higher_good",
    }
    fit_rows = [
        row
        for row in matrix
        if is_candidate(row) and str(row.get("run_label", "")).startswith(tuple(args.fit_prefixes))
    ]
    holdout_rows = [
        row
        for row in matrix
        if is_candidate(row) and str(row.get("run_label", "")).startswith(tuple(args.holdout_prefixes))
    ]

    threshold_rows = []
    holdout_summary = []
    selected_group_rows = []
    best_decision: dict[str, Any] | None = None

    for pred_row in passing:
        predictor = str(pred_row.get("predictor", ""))
        direction = direction_by_predictor.get(predictor, "higher_good")
        fit = best_threshold(fit_rows, predictor, direction, "continuous_retention_chain")
        threshold = finite_float(fit.get("threshold_score"))
        threshold_record = {
            "predictor": predictor,
            "direction": direction,
            "fit_rows": len(fit_rows),
            "fit_early_rows": sum(int_flag(row.get("early_source_chain")) for row in fit_rows),
            "fit_continuous_rows": sum(int_flag(row.get("continuous_retention_chain")) for row in fit_rows),
            **fit,
        }
        threshold_rows.append(threshold_record)
        if not math.isfinite(threshold):
            continue
        fit_selected = selected(fit_rows, predictor, direction, threshold)
        holdout_selected = selected(holdout_rows, predictor, direction, threshold)
        all_rows = fit_rows + holdout_rows
        all_selected = selected(all_rows, predictor, direction, threshold)
        holdout_summary.extend(
            [
                summarize_split("fit_f40_f43", predictor, fit_rows, fit_selected),
                summarize_split("holdout_f44_f45", predictor, holdout_rows, holdout_selected),
                summarize_split("all_f40_f45", predictor, all_rows, all_selected),
            ]
        )
        selected_group_rows.extend(summarize_groups("fit_f40_f43", predictor, fit_selected))
        selected_group_rows.extend(summarize_groups("holdout_f44_f45", predictor, holdout_selected))
        selected_group_rows.extend(summarize_groups("all_f40_f45", predictor, all_selected))
        holdout_split = summarize_split("holdout_f44_f45", predictor, holdout_rows, holdout_selected)
        candidate = {
            "best_predictor": predictor,
            "holdout_selected_rows": holdout_split["selected_rows"],
            "holdout_selected_early_rows": holdout_split["selected_early_rows"],
            "holdout_selected_continuous_rows": holdout_split["selected_continuous_rows"],
            "holdout_selected_h100": holdout_split["selected_h100"],
            "holdout_selected_h400": holdout_split["selected_h400"],
            "holdout_selected_h800": holdout_split["selected_h800"],
            "holdout_selected_h1600": holdout_split["selected_h1600"],
            "holdout_selected_h3200": holdout_split["selected_h3200"],
            "holdout_selected_h4800": holdout_split["selected_h4800"],
            "holdout_group_early_chain": holdout_split["selected_group_early_chain"],
            "holdout_group_continuous": holdout_split["selected_group_continuous"],
            "holdout_group_h4800": holdout_split["selected_group_h4800"],
            "threshold_score": threshold,
            "threshold_raw": fit.get("threshold_raw", ""),
        }
        if best_decision is None:
            best_decision = candidate
        else:
            current_key = (
                int(candidate["holdout_group_early_chain"]),
                int(candidate["holdout_group_h4800"]),
                int(candidate["holdout_group_continuous"]),
                int(candidate["holdout_selected_continuous_rows"]),
                finite_float(candidate["holdout_selected_h4800"]),
            )
            best_key = (
                int(best_decision["holdout_group_early_chain"]),
                int(best_decision["holdout_group_h4800"]),
                int(best_decision["holdout_group_continuous"]),
                int(best_decision["holdout_selected_continuous_rows"]),
                finite_float(best_decision["holdout_selected_h4800"]),
            )
            if current_key > best_key:
                best_decision = candidate

    write_rows(out_dir / f"{args.artifact_prefix}_threshold.csv", threshold_rows)
    write_rows(out_dir / f"{args.artifact_prefix}_holdout.csv", holdout_summary)
    write_rows(out_dir / f"{args.artifact_prefix}_selected_groups.csv", selected_group_rows)

    best_decision = best_decision or {
        "best_predictor": "",
        "holdout_selected_rows": 0,
        "holdout_selected_early_rows": 0,
        "holdout_selected_continuous_rows": 0,
        "holdout_group_early_chain": 0,
        "holdout_group_continuous": 0,
        "holdout_group_h4800": 0,
        "threshold_score": "",
        "threshold_raw": "",
    }
    decision = {
        "fit_split": "+".join(args.fit_prefixes),
        "holdout_split": "+".join(args.holdout_prefixes),
        "fit_rows": len(fit_rows),
        "holdout_rows": len(holdout_rows),
        "passing_train_loss_predictors": len(passing),
        **best_decision,
        "promotion_allowed": 0,
    }
    if not passing:
        decision["decision"] = "F46NoTrainLossSelectorToValidate"
        decision["next_repair_recommended"] = 0
    elif int_flag(best_decision.get("holdout_group_early_chain")) and int_flag(best_decision.get("holdout_group_h4800")):
        decision["decision"] = "F46TrainLossSelectorHoldsOutAsRepairHypothesis"
        decision["next_repair_recommended"] = 1
    elif int_flag(best_decision.get("holdout_group_early_chain")) and int_flag(best_decision.get("holdout_group_continuous")):
        decision["decision"] = "F46TrainLossSelectorContinuousButNoH4800"
        decision["next_repair_recommended"] = 1
    else:
        decision["decision"] = "F46TrainLossSelectorNotActionable"
        decision["next_repair_recommended"] = 0
    write_json(out_dir / f"{args.artifact_prefix}_decision.json", decision)
    write_rows(out_dir / f"{args.artifact_prefix}_decision.csv", [decision])
    append_exec(
        out_dir,
        cmd,
        status="completed",
        note=f"decision={decision['decision']} best={decision.get('best_predictor')} holdout_h4800={decision.get('holdout_group_h4800')}",
    )


if __name__ == "__main__":
    main()
