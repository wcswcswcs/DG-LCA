#!/usr/bin/env python3
"""Aggregate v22.03 source-preservation continuations from fresh matrices."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.fu.diffeomorphic_target import summarize_info_volume  # noqa: E402
from dgkan.fu.terminal_retention import classify_terminal_erosion, classify_v22_03_source_chain  # noqa: E402
from experiments.run_v22_03_common import (  # noqa: E402
    PYTHON,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    mean,
    read_rows,
    simple_svg,
    write_json,
    write_rows,
)


HORIZONS = [100, 400, 800, 1600, 2400, 3200, 4000, 4800, 6400]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", required=True)
    p.add_argument("--tag", default="f145_f147_source_preserve")
    p.add_argument("--label", default="F145-F147 source-preservation oldshape full")
    return p


def _row_source(row: dict[str, Any], step: int) -> float:
    return finite_float(row.get(f"source_h{step}"))


def _candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if str(r.get("mechanism", "")).startswith("M") and not str(r.get("v21_id", "")).startswith("CTRL")]


def _group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("carrier", "")), str(row.get("basis_repair_variant", "")), str(row.get("v21_id", "")))


def summarize(source_dir: Path, label: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_rows(source_dir / "v21_01_source_retention_matrix.csv")
    traces = read_rows(source_dir / "v21_01_source_retention_raw_traces.csv")
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[_group_key(row)].append(row)
    trace_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in traces:
        trace_buckets[str(row.get("v21_id", ""))].append(row)

    summary: list[dict[str, Any]] = []
    dataset_localization: list[dict[str, Any]] = []
    row_localization: list[dict[str, Any]] = []
    for (carrier, variant, v22_id), group in sorted(buckets.items()):
        item: dict[str, Any] = {
            "continuation": label,
            "source_dir": str(source_dir),
            "carrier": carrier,
            "variant": variant,
            "v22_id": v22_id,
            "rows": len(group),
            "mechanism": group[0].get("mechanism", "") if group else "",
            "hypothesis": group[0].get("hypothesis", "") if group else "",
        }
        for step in HORIZONS:
            item[f"h{step}"] = mean(group, f"source_h{step}")
        row_h4800_positive_count = sum(1 for r in group if _row_source(r, 4800) >= 0.005)
        item["row_h4800_positive_count"] = row_h4800_positive_count
        decision = classify_v22_03_source_chain(
            item.get("h100"),
            item.get("h400"),
            item.get("h800"),
            item.get("h1600"),
            item.get("h2400"),
            item.get("h3200"),
            item.get("h4000"),
            item.get("h4800"),
            item.get("h6400"),
            control_equivalent=int(str(v22_id).startswith("CTRL")),
            row_h4800_positive_count=row_h4800_positive_count,
        )
        item.update(
            {
                "early_source_chain_group": decision.early_source_chain,
                "continuous_h3200_group": decision.continuous_h3200_chain,
                "productive_h4800_group": decision.productive_h4800_chain,
                "terminal_erosion_group": decision.terminal_erosion,
                "terminal_collapse_group": decision.terminal_collapse,
                "h1600_retention_ratio": decision.h1600_retention_ratio,
                "h3200_retention_ratio": decision.h3200_retention_ratio,
                "h4800_retention_ratio": decision.h4800_retention_ratio,
                "h6400_retention_ratio": decision.h6400_retention_ratio,
                "terminal_erosion_class": classify_terminal_erosion(item),
                "source_chain_blocker": decision.blocker,
            }
        )
        info_volume = summarize_info_volume(trace_buckets.get(v22_id, []), step=4000)
        info_volume["info_volume_rows"] = info_volume.pop("rows", 0)
        item.update(info_volume)
        summary.append(item)

        by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_dataset[str(row.get("dataset", ""))].append(row)
            row_decision = classify_v22_03_source_chain(
                row.get("source_h100"),
                row.get("source_h400"),
                row.get("source_h800"),
                row.get("source_h1600"),
                row.get("source_h2400"),
                row.get("source_h3200"),
                row.get("source_h4000"),
                row.get("source_h4800"),
                row.get("source_h6400"),
                control_equivalent=int(str(v22_id).startswith("CTRL")),
                row_h4800_positive_count=1 if _row_source(row, 4800) >= 0.005 else 0,
                row_positive_min=1,
            )
            row_localization.append(
                {
                    "v22_id": v22_id,
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    **{f"source_h{step}": row.get(f"source_h{step}", "") for step in HORIZONS},
                    "v22_03_early_source_chain": row_decision.early_source_chain,
                    "v22_03_continuous_h3200_chain": row_decision.continuous_h3200_chain,
                    "v22_03_productive_h4800_chain": row_decision.productive_h4800_chain,
                    "v22_03_terminal_erosion": row_decision.terminal_erosion,
                    "v22_03_source_chain_blocker": row_decision.blocker,
                }
            )
        for dataset, drows in sorted(by_dataset.items()):
            drow = {
                "v22_id": v22_id,
                "dataset": dataset,
                "rows": len(drows),
                **{f"mean_h{step}": mean(drows, f"source_h{step}") for step in HORIZONS},
                "h4800_positive_rows": sum(1 for r in drows if _row_source(r, 4800) >= 0.005),
            }
            ddec = classify_v22_03_source_chain(
                drow.get("mean_h100"),
                drow.get("mean_h400"),
                drow.get("mean_h800"),
                drow.get("mean_h1600"),
                drow.get("mean_h2400"),
                drow.get("mean_h3200"),
                drow.get("mean_h4000"),
                drow.get("mean_h4800"),
                drow.get("mean_h6400"),
                control_equivalent=int(str(v22_id).startswith("CTRL")),
                row_h4800_positive_count=drow["h4800_positive_rows"],
                row_positive_min=max(1, min(3, len(drows))),
            )
            drow.update(
                {
                    "early_rows": sum(int_flag(r.get("v22_03_early_source_chain")) for r in row_localization if r.get("v22_id") == v22_id and r.get("dataset") == dataset),
                    "continuous_rows": sum(int_flag(r.get("v22_03_continuous_h3200_chain")) for r in row_localization if r.get("v22_id") == v22_id and r.get("dataset") == dataset),
                    "h4800_rows": sum(int_flag(r.get("v22_03_productive_h4800_chain")) for r in row_localization if r.get("v22_id") == v22_id and r.get("dataset") == dataset),
                    "blocker": ddec.blocker,
                }
            )
            dataset_localization.append(drow)
    return summary, dataset_localization, row_localization


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir)
    summary, dataset_localization, row_localization = summarize(source_dir, args.label)
    tag = str(args.tag).strip() or "source_preservation"
    audit_path = out_dir / f"v22_03_{tag}_audit.csv"
    dataset_path = out_dir / f"v22_03_{tag}_dataset_localization.csv"
    row_path = out_dir / f"v22_03_{tag}_row_localization.csv"
    route_path = out_dir / f"v22_03_{tag}_route.csv"
    write_rows(audit_path, summary)
    write_rows(dataset_path, dataset_localization)
    write_rows(row_path, row_localization)
    candidates = _candidate_rows(summary)
    route = {
        "continuation": args.label,
        "source_dir": str(source_dir),
        "source_chain_rows": len(read_rows(source_dir / "v21_01_source_retention_matrix.csv")),
        "candidate_groups": len(candidates),
        "candidate_early_chain": sum(int_flag(r.get("early_source_chain_group")) for r in candidates),
        "candidate_continuous_h3200": sum(int_flag(r.get("continuous_h3200_group")) for r in candidates),
        "candidate_h4800": sum(int_flag(r.get("productive_h4800_group")) for r in candidates),
        "terminal_erosion_groups": sum(int_flag(r.get("terminal_erosion_group")) for r in candidates),
        "terminal_collapse_groups": sum(int_flag(r.get("terminal_collapse_group")) for r in candidates),
        "best_v22_id": "",
        "best_h4800": "",
        "best_h4800_retention_ratio": "",
        "promotion_allowed": 0,
        "decision": "NoH4800Candidate",
    }
    if candidates:
        best = sorted(candidates, key=lambda r: finite_float(r.get("h4800_retention_ratio"), -1.0), reverse=True)[0]
        route.update(
            {
                "best_v22_id": best.get("v22_id", ""),
                "best_h4800": best.get("h4800", ""),
                "best_h4800_retention_ratio": best.get("h4800_retention_ratio", ""),
                "decision": "HasH4800Candidate" if any(int_flag(r.get("productive_h4800_group")) for r in candidates) else "NoH4800Candidate",
            }
        )
    write_rows(route_path, [route])
    write_json(out_dir / f"v22_03_{tag}_route.json", route)
    simple_svg(out_dir / "figures" / f"v22_03_{tag}_h4800.svg", f"v22.03 {tag} h4800", candidates, "h4800")
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_03_source_preservation.py --source-dir {source_dir} --tag {tag}", status="completed", note=f"candidate_groups={route['candidate_groups']} h4800={route['candidate_h4800']}")


if __name__ == "__main__":
    main()
