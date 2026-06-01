#!/usr/bin/env python3
"""Aggregate v12.23 Line D hardening CSV shards into auditable artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def finite_values(values: Sequence[Any]) -> list[float]:
    return [v for v in (finite_float(x) for x in values) if math.isfinite(v)]


def mean(values: Sequence[Any]) -> float:
    vals = finite_values(values)
    return sum(vals) / len(vals) if vals else float("nan")


def read_rows(paths: Sequence[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    paths = sorted(out_dir.glob(args.pattern))
    rows = read_rows(paths)
    summary_rows: list[dict[str, Any]] = []
    keys = sorted({(r.get("family", ""), r.get("dataset", "")) for r in rows})
    for family, dataset in keys:
        group = [r for r in rows if r.get("family") == family and r.get("dataset") == dataset]
        val_acc = finite_values([r.get("val_acc") for r in group])
        summary_rows.append(
            {
                "stage": "V1223_LINE_D_HARDENING_AGGREGATE_SUMMARY",
                "family": family,
                "dataset": dataset,
                "rows": len(group),
                "family_near_pass_rows": sum(1 for r in group if r.get("status") == "FamilyNearPass"),
                "kernel_blocked_rows": sum(1 for r in group if r.get("status") == "KernelBlocked"),
                "mean_val_acc": mean([r.get("val_acc") for r in group]),
                "min_val_acc": min(val_acc) if val_acc else float("nan"),
                "max_val_acc": max(val_acc) if val_acc else float("nan"),
                "mean_NLL": mean([r.get("NLL") for r in group]),
                "mean_linec_CouplingR2": mean([r.get("linec_CouplingR2") for r in group]),
                "mean_linec_NoiseSignalLeak": mean([r.get("linec_NoiseSignalLeak") for r in group]),
                "mean_linec_RealSignalReservoirRatio": mean([r.get("linec_RealSignalReservoirRatio") for r in group]),
                "promotion_allowed": 0,
                "no_fake": 1,
            }
        )
    best_by_val = max(rows, key=lambda r: finite_float(r.get("val_acc")), default={})
    best_by_linec = max(rows, key=lambda r: finite_float(r.get("linec_CouplingR2")), default={})
    route = {
        "stage": "V1223_LINE_D_HARDENING_AGGREGATE_ROUTE",
        "input_files": [str(p) for p in paths],
        "rows": len(rows),
        "summary_rows": len(summary_rows),
        "family_near_pass_rows": sum(1 for r in rows if r.get("status") == "FamilyNearPass"),
        "kernel_blocked_rows": sum(1 for r in rows if r.get("status") == "KernelBlocked"),
        "best_by_val_acc": best_by_val,
        "best_by_linec_CouplingR2": best_by_linec,
        "promotion_allowed": 0,
        "route_impact": "Line D hardening evidence only; not an official S1-S5 promotion route in v12.23",
        "no_fake": 1,
    }
    write_csv(out_dir / f"{args.artifact_prefix}.csv", rows)
    write_csv(out_dir / f"{args.artifact_prefix}_summary.csv", summary_rows)
    (out_dir / f"{args.artifact_prefix}_summary.json").write_text(json.dumps(route, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(route, indent=2, ensure_ascii=False, sort_keys=True))
    return route


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pattern", default="v1223_line_d_hardening_*_e8.csv")
    parser.add_argument("--artifact-prefix", default="v1223_line_d_hardening_aggregate")
    return parser


if __name__ == "__main__":
    aggregate(build_argparser().parse_args())
