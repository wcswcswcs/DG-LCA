#!/usr/bin/env python3
"""Failure-mode summaries for v22.61 M0C diagnostics."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results/v22_61"
CHUNK_ROOT = OUT_ROOT / "chunks"
ORACLE_DATASETS = ("MNIST", "FashionMNIST", "KMNIST", "Wine", "Spam")
PREFIXES = ("v22_61_m0o_objective_features", "v22_61_m0o_objective_features_holdout")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
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


def label_matches(label: str, prefix: str) -> bool:
    return any(label.startswith(f"{prefix}_{dataset}_s") for dataset in ORACLE_DATASETS)


def score(row: dict[str, str]) -> float:
    return fval(row.get("oracle_selection_score"), fval(row.get("Delta_NLL_vs_reference")))


def best_score(rows: list[dict[str, str]]) -> dict[str, str]:
    return min(rows, key=score)


def best_delta(rows: list[dict[str, str]]) -> dict[str, str]:
    return min(rows, key=lambda r: fval(r.get("Delta_NLL_vs_reference")))


def load_oracle_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for prefix in PREFIXES:
        for path in sorted(CHUNK_ROOT.glob(f"v22_61_{prefix}*_oracle_detail.csv")):
            for row in read_rows(path):
                if not label_matches(row.get("run_label", ""), prefix):
                    continue
                key = (
                    row.get("run_label", ""),
                    row.get("phase_bucket", ""),
                    row.get("checkpoint_step", ""),
                    row.get("mixture_name", ""),
                    row.get("mixture_kind", ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    return rows


def summarize_gate_deficits() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    metric_pairs = [
        ("beats_strongest_reference_rows", "required_beats_reference_rows"),
        ("beats_random_frozen_controls_rows", "required_beats_random_rows"),
        ("beats_credit_only_and_self_only_rows", "required_credit_self_rows"),
        ("oracle_gap_ratio_le_050_rows", "required_oracle_gap_rows"),
        ("no_ECE_Brier_tail_debt_rows", "required_no_debt_rows"),
        ("controller_param_ratio_le_001_rows", "required_param_budget_rows"),
        ("leakage_audit_pass_rows", "required_leakage_rows"),
    ]
    for tag in ("h16", "h32", "h64"):
        path = OUT_ROOT / f"v22_61_m0c_feature_imitation_combined_{tag}_summary.csv"
        if not path.exists():
            continue
        for row in read_rows(path):
            deficits = {}
            for actual_col, required_col in metric_pairs:
                deficits[actual_col.replace("_rows", "_deficit")] = max(0, int(fval(row.get(required_col))) - int(fval(row.get(actual_col))))
            bottleneck = max(deficits.items(), key=lambda item: item[1])
            out.append(
                {
                    "tag": tag,
                    "method": row.get("method", ""),
                    "pass": row.get("m0c_feature_imitation_pass", ""),
                    "rows": row.get("rows", ""),
                    "controller_param_ratio": row.get("controller_param_ratio", ""),
                    "bottleneck": bottleneck[0],
                    "bottleneck_deficit": bottleneck[1],
                    **deficits,
                }
            )
    return out


def summarize_oracle_targets(oracle_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_task_phase: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in oracle_rows:
        by_task_phase[(row.get("dataset", ""), row.get("seed", ""), row.get("phase_bucket", ""))].append(row)
    counts: Counter[tuple[str, str]] = Counter()
    for (_, _, phase), rows in by_task_phase.items():
        target = best_score([r for r in rows if r.get("mixture_kind") != "random_dirichlet"])
        counts[(phase, target.get("mixture_name", ""))] += 1
    return [
        {"phase_bucket": phase, "oracle_target_mixture": mixture, "count": count}
        for (phase, mixture), count in sorted(counts.items(), key=lambda item: (item[0][0], -item[1], item[0][1]))
    ]


def summarize_feature_linear_failures(oracle_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    detail_path = OUT_ROOT / "v22_61_m0c_feature_imitation_combined_h16_detail.csv"
    if not detail_path.exists():
        return []
    oracle_by_task: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in oracle_rows:
        oracle_by_task[(row.get("dataset", ""), row.get("seed", ""))].append(row)
    out: list[dict[str, Any]] = []
    for row in read_rows(detail_path):
        if row.get("method") != "feature_linear" or iflag(row.get("beats_random_best")):
            continue
        task_rows = oracle_by_task[(row.get("dataset", ""), row.get("seed", ""))]
        random_best = best_delta([r for r in task_rows if r.get("mixture_kind") == "random_dirichlet"])
        oracle = best_score([r for r in task_rows if r.get("mixture_kind") != "random_dirichlet"])
        out.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "selected_mixture": row.get("selected_mixture", ""),
                "selected_phase": row.get("selected_phase", ""),
                "selected_delta": row.get("Delta_NLL_vs_reference", ""),
                "random_best_delta": random_best.get("Delta_NLL_vs_reference", ""),
                "oracle_mixture": oracle.get("mixture_name", ""),
                "oracle_delta": oracle.get("Delta_NLL_vs_reference", ""),
                "oracle_gap_ratio": row.get("oracle_gap_ratio", ""),
                "no_debt": row.get("no_ECE_Brier_tail_debt", ""),
            }
        )
    return out


def main() -> int:
    oracle_rows = load_oracle_rows()
    gate_rows = summarize_gate_deficits()
    target_rows = summarize_oracle_targets(oracle_rows)
    failure_rows = summarize_feature_linear_failures(oracle_rows)
    gate_file = OUT_ROOT / "v22_61_m0c_failure_gate_deficits.csv"
    target_file = OUT_ROOT / "v22_61_m0c_oracle_target_distribution.csv"
    failure_file = OUT_ROOT / "v22_61_m0c_feature_linear_beats_random_failures.csv"
    write_rows(gate_file, gate_rows)
    write_rows(target_file, target_rows)
    write_rows(failure_file, failure_rows)
    route = {
        "gate_deficits_file": str(gate_file),
        "oracle_target_distribution_file": str(target_file),
        "feature_linear_beats_random_failures_file": str(failure_file),
        "feature_linear_beats_random_failures": len(failure_rows),
        "oracle_target_unique_phase_mixtures": len(target_rows),
    }
    (OUT_ROOT / "v22_61_m0c_failure_modes_route.json").write_text(json.dumps(route, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(route, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
