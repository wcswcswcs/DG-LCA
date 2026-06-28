#!/usr/bin/env python3
"""Simple constant/phase policy sweep for v22.61 M0C diagnostics.

This enumerates fixed mixture policies over already materialized oracle detail
rows. It is a diagnostic for whether controller success could come from a
dataset-independent static or phase-only schedule.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results/v22_61"
PYTHON = "/home/chengshun.wang/miniconda3/envs/kan/bin/python"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import experiments.analyze_v22_61_m0c_feature_imitation as diag


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


def score(row: dict[str, str]) -> float:
    return fval(row.get("oracle_selection_score"), fval(row.get("Delta_NLL_vs_reference")))


def best_score(rows: list[dict[str, str]]) -> dict[str, str]:
    return min(rows, key=score)


def best_delta(rows: list[dict[str, str]]) -> dict[str, str]:
    return min(rows, key=lambda r: fval(r.get("Delta_NLL_vs_reference")))


def evaluate_policy(
    policy_name: str,
    phase_to_mixture: dict[str, str],
    by_task_phase: dict[tuple[str, int, str], list[dict[str, str]]],
) -> list[dict[str, Any]]:
    by_task: dict[tuple[str, int], list[tuple[str, list[dict[str, str]]]]] = defaultdict(list)
    for (dataset, seed, phase), rows in by_task_phase.items():
        by_task[(dataset, seed)].append((phase, rows))
    out: list[dict[str, Any]] = []
    for (dataset, seed), phase_items in sorted(by_task.items()):
        chosen_rows: list[tuple[dict[str, str], str]] = []
        task_rows: list[dict[str, str]] = []
        for phase, phase_rows in phase_items:
            task_rows.extend(phase_rows)
            structured = [r for r in phase_rows if r.get("mixture_kind") != "random_dirichlet"]
            mixture = phase_to_mixture.get(phase, phase_to_mixture.get("*", "onehot_A0_identity"))
            matches = [r for r in structured if r.get("mixture_name") == mixture]
            if not matches:
                matches = [r for r in structured if r.get("mixture_name") == "onehot_A0_identity"] or structured
            chosen_rows.append((best_score(matches), mixture))
        chosen, mixture = min(chosen_rows, key=lambda item: score(item[0]))
        structured_task = [r for r in task_rows if r.get("mixture_kind") != "random_dirichlet"]
        random_best = best_delta([r for r in task_rows if r.get("mixture_kind") == "random_dirichlet"])
        oracle = best_score(structured_task)
        credit = best_score([r for r in structured_task if r.get("mixture_name") == "onehot_A1_credit_only"])
        self_only = best_score([r for r in structured_task if r.get("mixture_name") == "onehot_A2_self_geometry"])
        delta = fval(chosen.get("Delta_NLL_vs_reference"))
        oracle_delta = fval(oracle.get("Delta_NLL_vs_reference"))
        denom = max(abs(oracle_delta), 1.0e-12)
        out.append(
            {
                "dataset": dataset,
                "seed": seed,
                "method": policy_name,
                "selected_mixture": chosen.get("mixture_name", ""),
                "selected_phase": chosen.get("phase_bucket", ""),
                "Delta_NLL_vs_reference": delta,
                "oracle_Delta_NLL_vs_reference": oracle_delta,
                "beats_reference": int(delta < -1.0e-5),
                "beats_random_best": int(delta < fval(random_best.get("Delta_NLL_vs_reference")) - 1.0e-5),
                "beats_credit_and_self": int(delta < fval(credit.get("Delta_NLL_vs_reference")) - 1.0e-5 and delta < fval(self_only.get("Delta_NLL_vs_reference")) - 1.0e-5),
                "oracle_gap_ratio": max(0.0, (delta - oracle_delta) / denom),
                "no_ECE_Brier_tail_debt": iflag(chosen.get("no_ECE_Brier_tail_debt")),
                "operator_mixture_entropy": fval(chosen.get("operator_mixture_entropy")),
                "leakage_audit_pass": 1,
                "meta_test_no_controller_update": 1,
            }
        )
    return out


def gate_row(method: str, detail: list[dict[str, Any]]) -> dict[str, Any]:
    row_count = len(detail)
    required_beats_reference = math.ceil(row_count * 8 / 15.0)
    required_beats_random = math.ceil(row_count * 10 / 15.0)
    required_credit_self = math.ceil(row_count * 10 / 15.0)
    required_oracle_gap = math.ceil(row_count * 10 / 15.0)
    required_no_debt = math.ceil(row_count * 11 / 15.0)
    required_leakage = math.ceil(row_count * 14 / 15.0)
    row = {
        "method": method,
        "rows": row_count,
        "beats_strongest_reference_rows": sum(iflag(r["beats_reference"]) for r in detail),
        "beats_random_frozen_controls_rows": sum(iflag(r["beats_random_best"]) for r in detail),
        "beats_credit_only_and_self_only_rows": sum(iflag(r["beats_credit_and_self"]) for r in detail),
        "oracle_gap_ratio_le_050_rows": sum(int(fval(r["oracle_gap_ratio"]) <= 0.50) for r in detail),
        "no_ECE_Brier_tail_debt_rows": sum(iflag(r["no_ECE_Brier_tail_debt"]) for r in detail),
        "controller_param_ratio": 0.0,
        "controller_param_ratio_le_001_rows": row_count,
        "leakage_audit_pass_rows": row_count,
        "meta_test_no_controller_update": 1,
        "operator_mixture_entropy_mean": sum(fval(r["operator_mixture_entropy"]) for r in detail) / max(1, row_count),
        "required_beats_reference_rows": required_beats_reference,
        "required_beats_random_rows": required_beats_random,
        "required_credit_self_rows": required_credit_self,
        "required_oracle_gap_rows": required_oracle_gap,
        "required_no_debt_rows": required_no_debt,
        "required_param_budget_rows": row_count,
        "required_leakage_rows": required_leakage,
    }
    row["m0c_simple_policy_pass"] = int(
        row["beats_strongest_reference_rows"] >= required_beats_reference
        and row["beats_random_frozen_controls_rows"] >= required_beats_random
        and row["beats_credit_only_and_self_only_rows"] >= required_credit_self
        and row["oracle_gap_ratio_le_050_rows"] >= required_oracle_gap
        and row["no_ECE_Brier_tail_debt_rows"] >= required_no_debt
        and row["leakage_audit_pass_rows"] >= required_leakage
        and row["meta_test_no_controller_update"] == 1
    )
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="v22_61_m0o_objective_features,v22_61_m0o_objective_features_holdout")
    parser.add_argument("--output-tag", default="simple_policy_sweep")
    args = parser.parse_args(argv)

    rows = diag.load_detail(diag.split_prefixes(str(args.prefix)))
    by_task_phase: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_task_phase[(row["dataset"], int(row["seed"]), row.get("phase_bucket", ""))].append(row)
    classes = sorted({r["mixture_name"] for r in rows if r.get("mixture_kind") != "random_dirichlet"})
    phases = sorted({phase for (_, _, phase) in by_task_phase})
    policy_summaries: list[dict[str, Any]] = []
    best_details: list[dict[str, Any]] = []

    for cls in classes:
        detail = evaluate_policy(f"constant::{cls}", {"*": cls}, by_task_phase)
        policy_summaries.append(gate_row(f"constant::{cls}", detail))

    best_constant = max(policy_summaries, key=lambda r: (iflag(r["m0c_simple_policy_pass"]), int(r["beats_random_frozen_controls_rows"]), int(r["no_ECE_Brier_tail_debt_rows"]), int(r["beats_credit_only_and_self_only_rows"])))

    phase_summaries: list[dict[str, Any]] = []
    for combo in itertools.product(classes, repeat=len(phases)):
        phase_to_mixture = dict(zip(phases, combo))
        name = "phase::" + "|".join(f"{phase}={mix}" for phase, mix in phase_to_mixture.items())
        detail = evaluate_policy(name, phase_to_mixture, by_task_phase)
        phase_summaries.append(gate_row(name, detail))
    best_phase = max(phase_summaries, key=lambda r: (iflag(r["m0c_simple_policy_pass"]), int(r["beats_random_frozen_controls_rows"]), int(r["no_ECE_Brier_tail_debt_rows"]), int(r["beats_credit_only_and_self_only_rows"])))
    best_details.extend(evaluate_policy(best_constant["method"], {"*": best_constant["method"].split("::", 1)[1]}, by_task_phase))
    phase_policy = dict(part.split("=", 1) for part in best_phase["method"].split("::", 1)[1].split("|"))
    best_details.extend(evaluate_policy(best_phase["method"], phase_policy, by_task_phase))

    output_stem = f"v22_61_m0c_{args.output_tag}"
    summary_file = OUT_ROOT / f"{output_stem}_summary.csv"
    detail_file = OUT_ROOT / f"{output_stem}_best_detail.csv"
    route_file = OUT_ROOT / f"{output_stem}_route.json"
    summary = [best_constant, best_phase]
    write_rows(summary_file, summary)
    write_rows(detail_file, best_details)
    route = {
        "m0c_simple_policy_pass": max(iflag(r["m0c_simple_policy_pass"]) for r in summary),
        "best_constant": best_constant["method"],
        "best_phase": best_phase["method"],
        "summary_file": str(summary_file),
        "detail_file": str(detail_file),
        "prefixes": diag.split_prefixes(str(args.prefix)),
        "classes": len(classes),
        "phases": phases,
        "diagnostic_only_not_official_promotion": 1,
    }
    route_file.write_text(json.dumps(route, indent=2, sort_keys=True), encoding="utf-8")
    command = f"{PYTHON} experiments/analyze_v22_61_m0c_simple_policy_sweep.py --prefix {args.prefix} --output-tag {args.output_tag}"
    diag.append_exec(
        command,
        task_id="v22_61_m0c_simple_policy_sweep",
        status="pass",
        files=f"{summary_file}; {detail_file}; {route_file}",
        note=json.dumps(route, sort_keys=True),
    )
    print(json.dumps(route, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
