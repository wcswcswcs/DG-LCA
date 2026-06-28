#!/usr/bin/env python3
"""Summarize v22.66 KAN+MCGA scan rows by label and architecture."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REFERENCE_METHODS = {"adamw", "cautious_adamw", "schedule_free_adamw_local"}
DEFAULT_CONTROL_METHODS = [
    "same_functional_spectrum_random_coordinate",
    "same_generator_descent_energy_random",
    "same_C_skew_spectrum_random",
]


def norm_seed(value: Any) -> str:
    try:
        return str(int(float(value)))
    except Exception:
        return str(value)


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def debt(row: dict[str, Any]) -> float:
    vals = [safe_float(row.get(key)) for key in ["ECE", "Brier", "tail_loss_q95", "tail_loss_q99"]]
    return sum(vals) if all(math.isfinite(v) for v in vals) else math.nan


def mean(values: list[float]) -> float | str:
    vals = [v for v in values if math.isfinite(v)]
    return "" if not vals else sum(vals) / len(vals)


def load_rows(chunks_dir: Path, label: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(chunks_dir.glob("*.csv")):
        if path.name.endswith("_metric_rows.csv"):
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("run_label") == label and row.get("run_status") == "completed":
                    row["chunk_path"] = str(path)
                    rows.append(row)
    return rows


def summarize(rows: list[dict[str, str]], *, label: str, architecture: str, controls: list[str]) -> list[dict[str, Any]]:
    by = {
        (r.get("architecture_key"), r.get("method"), r.get("dataset"), norm_seed(r.get("seed"))): r
        for r in rows
    }
    conditions = sorted(
        {(r.get("dataset", ""), norm_seed(r.get("seed"))) for r in rows if r.get("architecture_key") == architecture},
        key=lambda item: (item[0], int(item[1])),
    )
    methods = sorted(
        {
            r.get("method", "")
            for r in rows
            if r.get("architecture_key") == architecture
            and r.get("method") not in REFERENCE_METHODS
            and r.get("method") not in controls
        }
    )
    out: list[dict[str, Any]] = []
    for method in methods:
        candidates = [by.get((architecture, method, dataset, seed)) for dataset, seed in conditions]
        candidates = [r for r in candidates if r]
        if not candidates:
            continue
        stats = {
            "KAN_improves_own_rows": 0,
            "beats_best_control_rows": 0,
            "beats_same_functional_spectrum_rows": 0,
            "beats_same_generator_rows": 0,
            "beats_same_C_skew_rows": 0,
            "no_debt_rows": 0,
            "overhead_le_035_rows": 0,
            "active_Gram_drift_le_005_rows": 0,
            "functional_spectrum_drift_le_050_rows": 0,
            "generator_positive_rows": 0,
            "standard_loop_pass_rows": 0,
            "bank_oet_used_rows": 0,
        }
        vals: dict[str, list[float]] = {
            "nll": [],
            "delta_own": [],
            "delta_control": [],
            "fs": [],
            "overhead": [],
            "linearization_error": [],
        }
        for row in candidates:
            dataset = row.get("dataset", "")
            seed = norm_seed(row.get("seed"))
            nll = safe_float(row.get("held_NLL"))
            own_best = min(
                [
                    value
                    for value in [
                        safe_float(by.get((architecture, ref, dataset, seed), {}).get("held_NLL"))
                        for ref in REFERENCE_METHODS
                    ]
                    if math.isfinite(value)
                ],
                default=math.nan,
            )
            control_vals = {
                control: safe_float(by.get((architecture, control, dataset, seed), {}).get("held_NLL"))
                for control in controls
            }
            control_best = min([v for v in control_vals.values() if math.isfinite(v)], default=math.nan)
            own_debt = min(
                [
                    value
                    for value in [debt(by.get((architecture, ref, dataset, seed), {})) for ref in REFERENCE_METHODS]
                    if math.isfinite(value)
                ],
                default=math.nan,
            )
            cand_debt = debt(row)
            stats["KAN_improves_own_rows"] += int(math.isfinite(nll) and math.isfinite(own_best) and nll < own_best)
            stats["beats_best_control_rows"] += int(math.isfinite(nll) and math.isfinite(control_best) and nll < control_best)
            named_controls = {
                "beats_same_functional_spectrum_rows": "same_functional_spectrum_random_coordinate",
                "beats_same_generator_rows": "same_generator_descent_energy_random",
                "beats_same_C_skew_rows": "same_C_skew_spectrum_random",
            }
            for stat_key, control in named_controls.items():
                stats[stat_key] += int(math.isfinite(nll) and math.isfinite(control_vals.get(control, math.nan)) and nll < control_vals[control])
            stats["no_debt_rows"] += int(math.isfinite(cand_debt) and math.isfinite(own_debt) and cand_debt <= own_debt + 1.0e-9)
            overhead = safe_float(row.get("controller_overhead_ratio"), 999.0)
            fs_drift = safe_float(row.get("functional_spectrum_drift_mean"), 999.0)
            stats["overhead_le_035_rows"] += int(overhead <= 0.35)
            stats["active_Gram_drift_le_005_rows"] += int(safe_float(row.get("active_Gram_drift_mean"), 999.0) <= 0.05)
            stats["functional_spectrum_drift_le_050_rows"] += int(fs_drift <= 0.50)
            stats["generator_positive_rows"] += int(safe_float(row.get("generator_descent_fraction"), 0.0) > 1.0e-4)
            stats["standard_loop_pass_rows"] += int(safe_float(row.get("standard_loop_runtime_trace_pass"), 0.0) == 1.0)
            stats["bank_oet_used_rows"] += int(safe_float(row.get("kan_bank_oet_used"), 0.0) > 0.5)
            vals["nll"].append(nll)
            vals["fs"].append(fs_drift)
            vals["overhead"].append(overhead)
            vals["linearization_error"].append(safe_float(row.get("kan_readout_linearization_max_abs_error"), 0.0))
            if math.isfinite(own_best):
                vals["delta_own"].append(nll - own_best)
            if math.isfinite(control_best):
                vals["delta_control"].append(nll - control_best)
        out.append(
            {
                "label": label,
                "architecture": architecture,
                "method": method,
                "rows": len(candidates),
                "mean_held_NLL": mean(vals["nll"]),
                **stats,
                "mean_delta_vs_own": mean(vals["delta_own"]),
                "mean_delta_vs_best_control": mean(vals["delta_control"]),
                "mean_functional_spectrum_drift": mean(vals["fs"]),
                "mean_overhead_ratio": mean(vals["overhead"]),
                "max_kan_readout_linearization_abs_error": max(vals["linearization_error"], default=""),
            }
        )
    out.sort(
        key=lambda row: (
            -int(row["KAN_improves_own_rows"]),
            -int(row["beats_best_control_rows"]),
            -int(row["functional_spectrum_drift_le_050_rows"]),
            float(row["mean_held_NLL"]),
        )
    )
    return out


def write_outputs(out_dir: Path, label: str, architecture: str, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{label}_scan_summary"
    fields = list(rows[0].keys()) if rows else ["label", "architecture", "method", "rows"]
    with base.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    base.with_suffix(".json").write_text(
        json.dumps({"label": label, "architecture": architecture, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", default="results/v22_66/chunks")
    parser.add_argument("--out-dir", default="results/v22_66")
    parser.add_argument("--label", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--control-methods", default=",".join(DEFAULT_CONTROL_METHODS))
    args = parser.parse_args()

    controls = [item.strip() for item in str(args.control_methods).split(",") if item.strip()]
    rows = summarize(load_rows(Path(args.chunks_dir), str(args.label)), label=str(args.label), architecture=str(args.architecture), controls=controls)
    write_outputs(Path(args.out_dir), str(args.label), str(args.architecture), rows)
    print(json.dumps({"label": args.label, "architecture": args.architecture, "methods": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
