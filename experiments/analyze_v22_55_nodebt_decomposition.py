"""Summarize no-debt component failures for v22.55 repair matrices."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_SPECS = [
    (
        "adamw_reference",
        "chunks/v22_55_v22_55_repair_scaled_h128_s300_reference_adamw_*_summary.csv",
    ),
    (
        "poet_official_reference",
        "chunks/v22_55_v22_55_repair_scaled_h128_s300_reference_poet_official_*_summary.csv",
    ),
    (
        "pion_sphere_reference",
        "chunks/v22_55_v22_55_repair_scaled_h128_s300_pion_reference_pion_oet_sphere_official_*_summary.csv",
    ),
    (
        "pion_local_reference",
        "chunks/v22_55_v22_55_repair_scaled_h128_s300_pion_reference_pion_oet_local_*_summary.csv",
    ),
    (
        "adamw_wgo_residual_deleak",
        "chunks/v22_55_v22_55_repair_metricfix_scaled_h128_s300_i300_lastlayer_residual_deleak_wgo_residual_deleak_diag_*_summary.csv",
    ),
    (
        "adamw_wgo_residual_safety",
        "chunks/v22_55_v22_55_repair_metricfix_scaled_h128_s300_i300_lastlayer_residual_safety_wgo_residual_safety_diag_*_summary.csv",
    ),
    (
        "poet_wgo_residual_deleak",
        "chunks/v22_55_v22_55_repair_poet_scaled_h128_s300_i300_lastlayer_residual_wgo_residual_deleak_diag_*_summary.csv",
    ),
    (
        "poet_wgo_residual_safety",
        "chunks/v22_55_v22_55_repair_poet_scaled_h128_s300_i300_lastlayer_residual_wgo_residual_safety_diag_*_summary.csv",
    ),
    (
        "pion_sphere_wgo_residual_deleak",
        "chunks/v22_55_v22_55_repair_pionofficial_scaled_h128_s300_i300_lastlayer_residual_wgo_residual_deleak_diag_*_summary.csv",
    ),
    (
        "pion_sphere_wgo_residual_safety",
        "chunks/v22_55_v22_55_repair_pionofficial_scaled_h128_s300_i300_lastlayer_residual_wgo_residual_safety_diag_*_summary.csv",
    ),
    (
        "pion_local_wgo_residual_deleak",
        "chunks/v22_55_v22_55_repair_pionlocal_scaled_h128_s300_i300_lastlayer_residual_wgo_residual_deleak_diag_*_summary.csv",
    ),
    (
        "pion_local_wgo_residual_safety",
        "chunks/v22_55_v22_55_repair_pionlocal_scaled_h128_s300_i300_lastlayer_residual_wgo_residual_safety_diag_*_summary.csv",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/v22_55")
    parser.add_argument(
        "--prefix",
        default="v22_55_repair_continuation6_nodebt_decomposition",
    )
    parser.add_argument("--debt-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def fnum(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return math.nan
    return float(value)


def flag(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value in ("", None):
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 1 if str(value).lower() in {"true", "pass"} else 0


def finite_mean(records: list[dict[str, Any]], key: str) -> float:
    vals = [float(record[key]) for record in records if not math.isnan(float(record[key]))]
    return mean(vals) if vals else math.nan


def read_rows(pattern: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(glob.glob(pattern)):
        with open(path, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = sorted({key for record in records for key in record.keys()})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    row_records: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []

    for analysis_label, rel_pattern in DEFAULT_SPECS:
        rows = read_rows(str(results_dir / rel_pattern))
        if not rows:
            summary.append({"analysis_label": analysis_label, "rows": 0, "missing": 1})
            continue

        records: list[dict[str, Any]] = []
        for row in rows:
            record = {
                "analysis_label": analysis_label,
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "method": row.get("method"),
                "reference_method": row.get("reference_method"),
                "final_NLL": fnum(row, "final_NLL"),
                "test_NLL": fnum(row, "test_NLL"),
                "held_ECE_delta": fnum(row, "held_ECE_delta"),
                "held_Brier_delta": fnum(row, "held_Brier_delta"),
                "held_tail_q95_delta": fnum(row, "held_tail_q95_delta"),
                "no_ECE_Brier_tail_debt": flag(row, "no_ECE_Brier_tail_debt"),
                "controller_overhead_ratio": fnum(row, "controller_overhead_ratio"),
                "standard_loop_hard_gate_pass": flag(row, "standard_loop_hard_gate_pass"),
            }
            record["ece_debt"] = int(record["held_ECE_delta"] > args.debt_tolerance)
            record["brier_debt"] = int(record["held_Brier_delta"] > args.debt_tolerance)
            record["tail_q95_debt"] = int(record["held_tail_q95_delta"] > args.debt_tolerance)
            records.append(record)
            row_records.append(record)

        summary.append(
            {
                "analysis_label": analysis_label,
                "rows": len(records),
                "missing": 0,
                "mean_final_NLL": finite_mean(records, "final_NLL"),
                "mean_test_NLL": finite_mean(records, "test_NLL"),
                "mean_held_ECE_delta": finite_mean(records, "held_ECE_delta"),
                "mean_held_Brier_delta": finite_mean(records, "held_Brier_delta"),
                "mean_held_tail_q95_delta": finite_mean(records, "held_tail_q95_delta"),
                "no_debt_rows": sum(r["no_ECE_Brier_tail_debt"] for r in records),
                "ece_debt_rows": sum(r["ece_debt"] for r in records),
                "brier_debt_rows": sum(r["brier_debt"] for r in records),
                "tail_q95_debt_rows": sum(r["tail_q95_debt"] for r in records),
                "overhead_le_035_rows": sum(
                    0
                    if math.isnan(r["controller_overhead_ratio"])
                    else int(r["controller_overhead_ratio"] <= 0.35)
                    for r in records
                ),
                "mean_controller_overhead_ratio": finite_mean(
                    records, "controller_overhead_ratio"
                ),
                "standard_loop_pass_rows": sum(
                    r["standard_loop_hard_gate_pass"] for r in records
                ),
            }
        )

    summary_json = results_dir / f"{args.prefix}_summary.json"
    summary_csv = results_dir / f"{args.prefix}_summary.csv"
    rows_csv = results_dir / f"{args.prefix}_rows.csv"

    with summary_json.open("w") as fh:
        json.dump(
            {
                "target_achieved": 0,
                "route": "R3-ReweightingExplained_NoFU",
                "part_f_status": "ExternalOETDominates_CurrentFUInsufficient",
                "analysis": (
                    "No-debt decomposition over completed scaled h128/s300 reference "
                    "and WGO-on-AdamW/POET/Pion runs; positive delta above tolerance "
                    "is counted as debt component."
                ),
                "debt_tolerance": args.debt_tolerance,
                "summary": summary,
            },
            fh,
            indent=2,
            sort_keys=True,
        )
    write_csv(summary_csv, summary)
    write_csv(rows_csv, row_records)

    print(
        json.dumps(
            {
                "summary_json": str(summary_json),
                "summary_csv": str(summary_csv),
                "rows_csv": str(rows_csv),
                "methods": len(summary),
                "rows": len(row_records),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
