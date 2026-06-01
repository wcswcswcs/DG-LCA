#!/usr/bin/env python3
"""Post-hoc v13.11 cover-objective alignment diagnostic.

This script intentionally uses only existing v13.11 Line V artifacts. It does
not run training and must not be interpreted as promotion evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


RUNS = ("official_v1311", "linev_repeat_seed1_v1311")
METRICS = (
    "cover_purity",
    "cover_churn",
    "cover_specialization_entropy",
    "cover_load_gini",
    "signal_to_cover_score",
)
SOURCE_FIELD = "source_vs_best_control"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        default="results/v13_11_cover_objective_validation_cover_preserving_substrate",
        help="Directory containing official_v1311 and repeat Line V artifacts.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to <base-dir>/objective_reset_diagnostic_v1311.",
    )
    return parser.parse_args()


def fnum(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def median(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def pearson(xs: list[float | None], ys: list[float | None]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    mean_x = sum(x for x, _ in pairs) / len(pairs)
    mean_y = sum(y for _, y in pairs) / len(pairs)
    var_x = sum((x - mean_x) ** 2 for x, _ in pairs)
    var_y = sum((y - mean_y) ** 2 for _, y in pairs)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    return cov / math.sqrt(var_x * var_y)


def ranks(values: list[float]) -> list[float]:
    indexed = sorted((value, index) for index, value in enumerate(values))
    output = [0.0] * len(values)
    pos = 0
    while pos < len(indexed):
        next_pos = pos + 1
        while next_pos < len(indexed) and indexed[next_pos][0] == indexed[pos][0]:
            next_pos += 1
        rank = (pos + next_pos - 1) / 2.0 + 1.0
        for _, index in indexed[pos:next_pos]:
            output[index] = rank
        pos = next_pos
    return output


def spearman(xs: list[float | None], ys: list[float | None]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    return pearson(ranks([x for x, _ in pairs]), ranks([y for _, y in pairs]))


def truthy(value: Any) -> bool:
    return str(value) in {"1", "1.0", "True", "true"}


def qsplit(rows: list[dict[str, Any]], metric: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    values = [(fnum(row.get(metric)), row) for row in rows]
    values = [(value, row) for value, row in values if value is not None]
    if len(values) < 4:
        return [], []
    values.sort(key=lambda item: item[0])
    count = max(1, int(math.ceil(len(values) * 0.25)))
    return [row for _, row in values[-count:]], [row for _, row in values[:count]]


def read_rows(base_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    files = (
        ("oracle", "v1311_oracle_cover_results.csv"),
        ("control", "v1311_random_permuted_cover_controls.csv"),
    )
    for run in RUNS:
        for row_kind, filename in files:
            path = base_dir / run / filename
            if not path.exists():
                continue
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    copied = dict(row)
                    copied["run"] = run
                    copied["row_kind"] = row_kind
                    copied["value_positive"] = int(
                        (fnum(copied.get(SOURCE_FIELD)) or -1e9) >= 0.005
                        and (fnum(copied.get("AUC_time_ratio")) or 1e9) <= 1.0
                        and truthy(copied.get("LineC_nonharm"))
                        and truthy(copied.get("tail_nonharm"))
                    )
                    rows.append(copied)
    return rows


def build_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, list[dict[str, Any]]]] = []
    for run in (*RUNS, "combined"):
        for row_kind in ("oracle", "control", "all"):
            subset = [
                row
                for row in rows
                if (run == "combined" or row["run"] == run)
                and (row_kind == "all" or row["row_kind"] == row_kind)
            ]
            if subset:
                scopes.append((run, row_kind, subset))
    for run, row_kind, subset in scopes:
        sources = [fnum(row.get(SOURCE_FIELD)) for row in subset]
        pass_rows = sum(truthy(row.get("task_family_pass")) for row in subset)
        value_positive_rows = sum(int(row["value_positive"]) for row in subset)
        for metric in METRICS:
            xs = [fnum(row.get(metric)) for row in subset]
            top, bottom = qsplit(subset, metric)
            top_sources = [fnum(row.get(SOURCE_FIELD)) for row in top]
            bottom_sources = [fnum(row.get(SOURCE_FIELD)) for row in bottom]
            top_median = median(top_sources)
            bottom_median = median(bottom_sources)
            output.append(
                {
                    "run": run,
                    "row_kind": row_kind,
                    "metric": metric,
                    "rows": len(subset),
                    "task_family_pass_rows": pass_rows,
                    "value_positive_rows": value_positive_rows,
                    "pearson_vs_source": pearson(xs, sources),
                    "spearman_vs_source": spearman(xs, sources),
                    "metric_median": median(xs),
                    "source_median": median(sources),
                    "top_quartile_rows": len(top),
                    "top_quartile_source_median": top_median,
                    "bottom_quartile_source_median": bottom_median,
                    "top_quartile_value_positive_rows": sum(int(row["value_positive"]) for row in top),
                    "bottom_quartile_value_positive_rows": sum(int(row["value_positive"]) for row in bottom),
                    "top_minus_bottom_source_median": (
                        top_median - bottom_median
                        if top_median is not None and bottom_median is not None
                        else None
                    ),
                }
            )
    return output


def build_method_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run in (*RUNS, "combined"):
        subset = [row for row in rows if run == "combined" or row["run"] == run]
        for method in sorted({row["cover_objective_method"] for row in subset}):
            method_rows = [row for row in subset if row["cover_objective_method"] == method]
            sources = [fnum(row.get(SOURCE_FIELD)) for row in method_rows]
            present_sources = [value for value in sources if value is not None]
            output.append(
                {
                    "run": run,
                    "cover_objective_method": method,
                    "rows": len(method_rows),
                    "task_family_pass_rows": sum(truthy(row.get("task_family_pass")) for row in method_rows),
                    "value_positive_rows": sum(int(row["value_positive"]) for row in method_rows),
                    "max_source_vs_best_control": max(present_sources, default=None),
                    "median_source_vs_best_control": median(sources),
                    "median_cover_purity": median([fnum(row.get("cover_purity")) for row in method_rows]),
                    "median_AUC_time_ratio": median([fnum(row.get("AUC_time_ratio")) for row in method_rows]),
                    "median_noise_delta": median([fnum(row.get("NoiseSignalLeak_delta")) for row in method_rows]),
                    "median_reservoir_delta": median(
                        [fnum(row.get("RealSignalReservoirRatio_delta")) for row in method_rows]
                    ),
                }
            )
    return output


def lookup_metric(metric_rows: list[dict[str, Any]], run: str, row_kind: str, metric: str) -> float | None:
    for row in metric_rows:
        if row["run"] == run and row["row_kind"] == row_kind and row["metric"] == metric:
            return row["spearman_vs_source"]
    return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir) if args.out_dir else base_dir / "objective_reset_diagnostic_v1311"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(base_dir)
    metric_rows = build_metric_rows(rows)
    method_rows = build_method_rows(rows)

    run_spearman = [
        lookup_metric(metric_rows, run, "oracle", "cover_purity")
        for run in RUNS
    ]
    alignment_pass = int(
        all(value is not None for value in run_spearman)
        and min(value for value in run_spearman if value is not None) >= 0.35
    )
    route = {
        "diagnostic_route": "D1-ObjectiveResetNeeded_CurrentCoverMetricsWeakValueAlignment",
        "official_route_unchanged": "R2-CoverObjectiveInvalid",
        "promotion_allowed": 0,
        "official_success_reached": 0,
        "uses_existing_artifacts_only": 1,
        "new_training_executed": 0,
        "line_a_k_architecture_search_executed": 0,
        "cover_metric_alignment_pass": alignment_pass,
        "combined_rows": len(rows),
        "combined_oracle_rows": sum(row["row_kind"] == "oracle" for row in rows),
        "combined_control_rows": sum(row["row_kind"] == "control" for row in rows),
        "combined_value_positive_rows": sum(int(row["value_positive"]) for row in rows),
        "combined_task_family_pass_rows": sum(truthy(row.get("task_family_pass")) for row in rows),
        "official_cover_purity_spearman_oracle": lookup_metric(
            metric_rows, "official_v1311", "oracle", "cover_purity"
        ),
        "repeat_cover_purity_spearman_oracle": lookup_metric(
            metric_rows, "linev_repeat_seed1_v1311", "oracle", "cover_purity"
        ),
        "combined_cover_purity_spearman_oracle": lookup_metric(
            metric_rows, "combined", "oracle", "cover_purity"
        ),
        "combined_signal_to_cover_spearman_oracle": lookup_metric(
            metric_rows, "combined", "oracle", "signal_to_cover_score"
        ),
        "recommendation": (
            "Do not continue A-CPF/K search inside v13.11; define a new "
            "value-cover objective with legal precommit train-stream proxy before substrate search."
        ),
    }

    write_csv(out_dir / "v1311_objective_reset_metric_alignment.csv", metric_rows)
    write_csv(out_dir / "v1311_objective_reset_method_summary.csv", method_rows)
    (out_dir / "v1311_objective_reset_route.json").write_text(
        json.dumps(route, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
