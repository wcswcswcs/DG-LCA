#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v22_68"


def flag(value: Any) -> int:
    return int(str(value).strip() in {"1", "true", "True", "yes"})


def fnum(value: Any, default: float = math.inf) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def method_sort_key(row: dict[str, Any]) -> tuple[int, int, int, int, int, float]:
    return (
        int(row.get("repair_gate_pass", 0)),
        int(row.get("beats_best_control_NLL_rows", 0)),
        int(row.get("beats_external_OET_NLL_rows", 0)),
        int(row.get("beats_same_generator_controls_rows", 0)),
        int(row.get("beats_strongest_NLL_rows", 0)),
        -fnum(row.get("mean_Delta_NLL_vs_best_external"), 0.0),
    )


def collect_campaign() -> dict[str, Any]:
    json_files = sorted(
        p
        for p in OUT.glob("v22_68_part_b_*_repair_analysis.json")
        if p.name not in {
            "v22_68_part_b_repair_campaign_summary.json",
            "v22_68_part_b_control_gap_cross_method_analysis.json",
            "v22_68_part_b_control_gap_cross_method_analysis_updated.json",
        }
    )
    rows: list[dict[str, Any]] = []
    for path in json_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data.get("summary_rows", []):
            out = dict(row)
            out["analysis_file"] = path.name
            out["repair_run_label_prefix"] = data.get("repair_run_label_prefix", "")
            out["conclusion"] = data.get("conclusion", "")
            rows.append(out)
    rows_sorted = sorted(rows, key=method_sort_key, reverse=True)
    summary = {
        "artifact": "v22_68_part_b_repair_campaign_summary",
        "generated_at_sg": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %z"),
        "analysis_files": [p.name for p in json_files],
        "total_methods": len(rows),
        "passed_methods": sum(flag(r.get("repair_gate_pass")) for r in rows),
        "best_by_gate_then_control_external_same_strongest": rows_sorted[:20],
    }
    write_rows(OUT / "v22_68_part_b_repair_campaign_summary.csv", rows_sorted)
    (OUT / "v22_68_part_b_repair_campaign_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def collect_control_gap() -> dict[str, Any]:
    detail_files = sorted(OUT.glob("v22_68_part_b_*_repair_analysis_detail.csv"))
    rows: list[dict[str, Any]] = []
    for path in detail_files:
        for row in read_rows(path):
            out: dict[str, Any] = dict(row)
            out["analysis_file"] = path.name
            rows.append(out)

    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("steps", "")))
        method = str(row.get("candidate_method", ""))
        by_key[key].append(row)
        by_method[method].append(row)

    key_summary: list[dict[str, Any]] = []
    no_control_rows: list[dict[str, Any]] = []
    for key, group in sorted(by_key.items(), key=lambda kv: (kv[0][0], int(kv[0][1]), int(kv[0][2]))):
        best = min(group, key=lambda r: fnum(r.get("Delta_NLL_vs_best_control")))
        item = {
            "dataset": key[0],
            "seed": key[1],
            "steps": key[2],
            "method_count": len({str(r.get("candidate_method", "")) for r in group}),
            "any_control_pass": int(any(flag(r.get("beats_best_control_NLL")) for r in group)),
            "any_external_pass": int(any(flag(r.get("beats_external_OET_NLL")) for r in group)),
            "any_same_generator_pass": int(any(flag(r.get("beats_same_generator_controls_NLL")) for r in group)),
            "best_delta_control": fnum(best.get("Delta_NLL_vs_best_control"), 0.0),
            "best_control_margin_method": best.get("candidate_method", ""),
            "best_control_method": best.get("best_control_method", ""),
            "best_method_external_pass": flag(best.get("beats_external_OET_NLL")),
            "best_method_same_pass": flag(best.get("beats_same_generator_controls_NLL")),
            "best_method_failure_mode": best.get("failure_mode", ""),
        }
        key_summary.append(item)
        if not item["any_control_pass"]:
            no_control_rows.append(item)

    method_summary: list[dict[str, Any]] = []
    for method, group in by_method.items():
        method_summary.append(
            {
                "candidate_method": method,
                "analysis_files": sorted({str(r.get("analysis_file", "")) for r in group}),
                "rows": len(group),
                "control_pass": sum(flag(r.get("beats_best_control_NLL")) for r in group),
                "same_generator_pass": sum(flag(r.get("beats_same_generator_controls_NLL")) for r in group),
                "external_pass": sum(flag(r.get("beats_external_OET_NLL")) for r in group),
                "strongest_pass": sum(flag(r.get("beats_strongest_NLL")) for r in group),
                "mean_delta_external": sum(fnum(r.get("Delta_NLL_vs_best_external"), 0.0) for r in group) / max(1, len(group)),
                "mean_delta_control": sum(fnum(r.get("Delta_NLL_vs_best_control"), 0.0) for r in group) / max(1, len(group)),
            }
        )
    method_summary.sort(
        key=lambda r: (
            int(r["control_pass"]),
            int(r["external_pass"]),
            int(r["same_generator_pass"]),
            int(r["strongest_pass"]),
            -float(r["mean_delta_external"]),
        ),
        reverse=True,
    )

    summary = {
        "artifact": "v22_68_part_b_control_gap_cross_method_analysis_updated",
        "generated_at_sg": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %z"),
        "source_detail_files": [p.name for p in detail_files],
        "total_detail_rows": len(rows),
        "unique_methods": len(by_method),
        "unique_row_keys": len(by_key),
        "diagnostic_only_no_rowwise_promotion": 1,
        "rowwise_any_control_pass_rows": sum(int(r["any_control_pass"]) for r in key_summary),
        "rowwise_no_existing_method_controls_pass_count": len(no_control_rows),
        "rowwise_no_existing_method_controls_pass_rows": no_control_rows,
        "method_summary_top_by_control_external": method_summary[:20],
    }
    write_rows(OUT / "v22_68_part_b_control_gap_cross_method_analysis_updated_key_summary.csv", key_summary)
    write_rows(OUT / "v22_68_part_b_control_gap_cross_method_analysis_updated_method_summary.csv", method_summary)
    (OUT / "v22_68_part_b_control_gap_cross_method_analysis_updated.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    campaign = collect_campaign()
    control = collect_control_gap()
    print(
        json.dumps(
            {
                "campaign_summary": "results/v22_68/v22_68_part_b_repair_campaign_summary.json",
                "control_gap_summary": "results/v22_68/v22_68_part_b_control_gap_cross_method_analysis_updated.json",
                "total_methods": campaign["total_methods"],
                "passed_methods": campaign["passed_methods"],
                "unique_methods": control["unique_methods"],
                "rowwise_no_existing_method_controls_pass_count": control["rowwise_no_existing_method_controls_pass_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
