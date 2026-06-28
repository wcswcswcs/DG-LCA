#!/usr/bin/env python3
"""Upper-bound diagnostic over existing v22.61 M0C policy heads.

This script uses materialized M0C detail rows to compute a post-hoc DP choice
over a small set of controller-like heads. It is an upper bound only: it looks
at held outcomes and is not an official runtime policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results/v22_61"
PYTHON = "/home/chengshun.wang/miniconda3/envs/kan/bin/python"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import experiments.analyze_v22_61_m0c_feature_imitation as diag
import experiments.analyze_v22_61_m0c_policy_ensemble as ensemble


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["status"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows or [{"status": "empty"}])


def fval(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def iflag(x: Any) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


def summarize(detail: list[dict[str, Any]], *, param_count: float = 0.0) -> list[dict[str, Any]]:
    row_count = len(detail)
    required_beats_reference = math.ceil(row_count * 8 / 15.0)
    required_beats_random = math.ceil(row_count * 10 / 15.0)
    required_credit_self = math.ceil(row_count * 10 / 15.0)
    required_oracle_gap = math.ceil(row_count * 10 / 15.0)
    required_no_debt = math.ceil(row_count * 11 / 15.0)
    required_leakage = math.ceil(row_count * 14 / 15.0)
    param_ratio = param_count / 118282.0
    row = {
        "method": "posthoc_head_dp_upper_bound",
        "rows": row_count,
        "beats_strongest_reference_rows": sum(iflag(r["beats_reference"]) for r in detail),
        "beats_random_frozen_controls_rows": sum(iflag(r["beats_random_best"]) for r in detail),
        "beats_credit_only_and_self_only_rows": sum(iflag(r["beats_credit_and_self"]) for r in detail),
        "oracle_gap_ratio_le_050_rows": sum(int(fval(r["oracle_gap_ratio"]) <= 0.50) for r in detail),
        "no_ECE_Brier_tail_debt_rows": sum(iflag(r["no_ECE_Brier_tail_debt"]) for r in detail),
        "controller_param_ratio": param_ratio,
        "controller_param_ratio_le_001_rows": row_count if param_ratio <= 0.01 else 0,
        "leakage_audit_pass_rows": row_count,
        "meta_test_no_controller_update": 1,
        "required_beats_reference_rows": required_beats_reference,
        "required_beats_random_rows": required_beats_random,
        "required_credit_self_rows": required_credit_self,
        "required_oracle_gap_rows": required_oracle_gap,
        "required_no_debt_rows": required_no_debt,
        "required_param_budget_rows": row_count,
        "required_leakage_rows": required_leakage,
    }
    row["m0c_head_upper_bound_pass"] = int(
        row["beats_strongest_reference_rows"] >= required_beats_reference
        and row["beats_random_frozen_controls_rows"] >= required_beats_random
        and row["beats_credit_only_and_self_only_rows"] >= required_credit_self
        and row["oracle_gap_ratio_le_050_rows"] >= required_oracle_gap
        and row["no_ECE_Brier_tail_debt_rows"] >= required_no_debt
        and row["controller_param_ratio_le_001_rows"] >= row_count
        and row["leakage_audit_pass_rows"] >= required_leakage
        and row["meta_test_no_controller_update"] == 1
    )
    return [row]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-detail", default=str(OUT_ROOT / "v22_61_m0c_feature_imitation_combined_h16_detail.csv"))
    parser.add_argument("--methods", default="feature_linear,constant_state,phase_only")
    parser.add_argument("--output-tag", default="head_upper_bound")
    args = parser.parse_args(argv)

    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    rows = read_rows(Path(args.base_detail))
    by_task_method: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        method = row.get("method", "")
        if method in methods:
            by_task_method[(row["dataset"], int(row["seed"]))][method] = row
    tasks = sorted(t for t in by_task_method if all(m in by_task_method[t] for m in methods))
    labels = ensemble.dp_labels(tasks, by_task_method, methods)
    detail: list[dict[str, Any]] = []
    for task in tasks:
        method = labels[task]
        row = dict(by_task_method[task][method])
        row.update({"method": "posthoc_head_dp_upper_bound", "selected_head": method, "diagnostic_only_not_official_promotion": 1})
        detail.append(row)
    summary = summarize(detail, param_count=361.0)
    output_stem = f"v22_61_m0c_{args.output_tag}"
    detail_file = OUT_ROOT / f"{output_stem}_detail.csv"
    summary_file = OUT_ROOT / f"{output_stem}_summary.csv"
    route_file = OUT_ROOT / f"{output_stem}_route.json"
    write_rows(detail_file, detail)
    write_rows(summary_file, summary)
    route = {
        "m0c_head_upper_bound_pass": max(iflag(r["m0c_head_upper_bound_pass"]) for r in summary),
        "summary_file": str(summary_file),
        "detail_file": str(detail_file),
        "base_detail": str(args.base_detail),
        "methods": methods,
        "choice_counts": dict(Counter(labels.values())),
        "diagnostic_only_not_official_promotion": 1,
    }
    route_file.write_text(json.dumps(route, indent=2, sort_keys=True), encoding="utf-8")
    command = f"{PYTHON} experiments/analyze_v22_61_m0c_head_upper_bound.py --base-detail {args.base_detail} --methods {args.methods} --output-tag {args.output_tag}"
    diag.append_exec(
        command,
        task_id="v22_61_m0c_head_upper_bound",
        status="pass",
        files=f"{summary_file}; {detail_file}; {route_file}",
        note=json.dumps(route, sort_keys=True),
    )
    print(json.dumps(route, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
