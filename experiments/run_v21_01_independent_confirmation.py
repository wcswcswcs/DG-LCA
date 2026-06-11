#!/usr/bin/env python3
"""Summarize v21.01 independent source-retention confirmation runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v21_01_common import (  # noqa: E402
    HORIZON_STEPS,
    PYTHON,
    append_exec,
    ensure_out,
    finite_float,
    retention_ratio,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--carrier", default="D-FOU")
    p.add_argument("--variant", default="FOU-R4-k4-triton-no-materialize")
    p.add_argument("--v21-id", default="KSW2-lowdegree-lowfreq-source-bank")
    p.add_argument("--prefix", default="v21_01_dfou_ksw2_independent_confirmation")
    return p


def summarize(args: argparse.Namespace) -> None:
    out_dir = ensure_out(args.out_dir)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v21_01_independent_confirmation.py --carrier {args.carrier} --v21-id {args.v21_id}",
        status="started",
    )
    import csv

    matrix = out_dir / "v21_01_source_retention_matrix.csv"
    with matrix.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if (
            str(row.get("carrier")) == str(args.carrier)
            and str(row.get("basis_repair_variant")) == str(args.variant)
            and str(row.get("v21_id")) == str(args.v21_id)
        ):
            buckets[(str(row.get("run_label")), str(row.get("init_seed_offset", "0")))].append(row)

    summary = []
    for (run_label, offset), group in sorted(buckets.items(), key=lambda kv: (int(kv[0][1]) if str(kv[0][1]).isdigit() else -1, kv[0][0])):
        item: dict[str, object] = {
            "run_label": run_label,
            "init_seed_offset": offset,
            "carrier": args.carrier,
            "basis_repair_variant": args.variant,
            "v21_id": args.v21_id,
            "rows": len(group),
        }
        for step in HORIZON_STEPS:
            vals = [finite_float(r.get(f"source_h{step}")) for r in group]
            vals = [v for v in vals if v == v]
            item[f"source_h{step}_mean"] = sum(vals) / len(vals) if vals else ""
            item[f"source_h{step}_pass_count"] = sum(1 for v in vals if v >= 0.005)
        item["retention_h3200_over_h1600"] = retention_ratio(item.get("source_h3200_mean"), item.get("source_h1600_mean"))
        item["retention_h4800_over_h3200"] = retention_ratio(item.get("source_h4800_mean"), item.get("source_h3200_mean"))
        item["retention_h6400_over_h4800"] = retention_ratio(item.get("source_h6400_mean"), item.get("source_h4800_mean"))
        h800 = finite_float(item.get("source_h800_mean"), -999.0)
        h1600 = finite_float(item.get("source_h1600_mean"), -999.0)
        h3200 = finite_float(item.get("source_h3200_mean"), -999.0)
        h4800 = finite_float(item.get("source_h4800_mean"), -999.0)
        h6400 = finite_float(item.get("source_h6400_mean"), -999.0)
        item["productive_h3200_candidate"] = int(h800 >= 0.005 and h1600 >= 0.005 and h3200 >= 0.005 and finite_float(item.get("retention_h3200_over_h1600"), 0.0) >= 0.50)
        item["productive_h4800_candidate"] = int(item["productive_h3200_candidate"] and h4800 >= 0.005 and finite_float(item.get("retention_h4800_over_h3200"), 0.0) >= 0.50)
        item["productive_h6400_candidate"] = int(item["productive_h4800_candidate"] and h6400 >= 0.005 and finite_float(item.get("retention_h6400_over_h4800"), 0.0) >= 0.50)
        item["late_rebound_group"] = int(h800 < 0.005 and h3200 >= 0.005)
        summary.append(item)

    independent = [r for r in summary if str(r.get("init_seed_offset")) != "0"]
    reproduced_h3200 = sum(1 for r in independent if int(r.get("productive_h3200_candidate", 0)))
    reproduced_h6400 = sum(1 for r in independent if int(r.get("productive_h6400_candidate", 0)))
    decision = {
        "carrier": args.carrier,
        "basis_repair_variant": args.variant,
        "v21_id": args.v21_id,
        "offset_groups": len(independent),
        "reproduced_h3200_groups": reproduced_h3200,
        "reproduced_h6400_groups": reproduced_h6400,
        "independent_confirmation_pass": int(reproduced_h3200 == len(independent) and reproduced_h6400 == len(independent) and len(independent) > 0),
        "decision": "independent_confirmation_failed" if reproduced_h3200 < len(independent) or reproduced_h6400 < len(independent) else "independent_confirmation_passed",
    }
    write_rows(out_dir / f"{args.prefix}_summary.csv", summary)
    write_rows(out_dir / f"{args.prefix}_decision.csv", [decision])
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v21_01_independent_confirmation.py --carrier {args.carrier} --v21-id {args.v21_id}",
        status="completed",
        note=str(decision),
    )


def main() -> None:
    summarize(parser().parse_args())


if __name__ == "__main__":
    main()
