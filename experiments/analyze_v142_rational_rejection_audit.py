#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("results/v14_2_functional_first_all_basis_parallel")
OUT_DIR = ROOT / "rational_rejection_audit_v142"

RUNS = {
    "rat28_interval40": ROOT / "repair_v142_rat28_amortized40",
    "rat28_interval80_phase": ROOT / "repair_v142_rat28_interval80_phase",
    "rat28_r6r7r8": ROOT / "repair_v142_rat28_r6r7r8",
    "rat28_r6_loose_tight": ROOT / "repair_v142_rat28_r6_loose_tight",
    "rat28_r9_rolesep": ROOT / "repair_v142_rat28_r9_rolesep",
    "rat28_r10_agreement": ROOT / "repair_v142_rat28_r10_agreement",
    "rat28_r10_agreement160": ROOT / "repair_v142_rat28_r10_agreement160",
}


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def sint(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def role_from_state_key(key: str) -> str:
    low = str(key).lower()
    if "denominator" in low or "denom" in low or "den:" in low:
        return "denominator"
    if "readout" in low or low.endswith(":w2") or low == "w2":
        return "readout"
    if "numerator" in low or "num" in low:
        return "numerator"
    if "basis" in low or "freq" in low or "center" in low or "width" in low:
        return "basis"
    if "hidden" in low or "w1" in low:
        return "hidden"
    if "input" in low or "w0" in low:
        return "input"
    return "other"


def failure_reasons(row: dict[str, str]) -> list[str]:
    if sint(row.get("control_method"), 0) == 1:
        return ["control_row"]
    reasons: list[str] = []
    if fnum(row.get("source_vs_best_control"), -999.0) < 0.002:
        reasons.append("source_vs_best_control")
    if fnum(row.get("AUCtime_ratio_vs_best_control"), 9.0) > 1.05:
        reasons.append("AUCtime_ratio")
    if fnum(row.get("CEp99_delta_vs_adamw"), 999.0) > 0.10:
        reasons.append("CEp99_tail")
    if fnum(row.get("NLL_delta_vs_adamw"), 999.0) > 0.05:
        reasons.append("NLL_tail")
    if fnum(row.get("ECE_delta_vs_adamw"), 999.0) > 0.05:
        reasons.append("ECE_tail")
    if sint(row.get("LineC_majority_pass"), 0) != 1:
        reasons.append("LineC")
    if fnum(row.get("step_time_ratio_vs_adamw"), 9.0) > 1.15:
        reasons.append("step_time")
    return reasons or ["pass"]


def median(values: list[float]) -> float:
    vals = sorted(v for v in values if not math.isnan(v) and not math.isinf(v))
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rejection_rows: list[dict[str, Any]] = []
    method_source: dict[tuple[str, str], list[float]] = defaultdict(list)
    method_auc: dict[tuple[str, str], list[float]] = defaultdict(list)
    method_pass_tasks: dict[tuple[str, str], set[str]] = defaultdict(set)
    reason_counts: Counter[tuple[str, str]] = Counter()
    role_acc: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    role_key_sets: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    route_rows: list[dict[str, Any]] = []

    for run_name, run_dir in RUNS.items():
        route_path = run_dir / "v142_route_decision.json"
        route = json.loads(route_path.read_text(encoding="utf-8")) if route_path.exists() else {}
        route_rows.append(
            {
                "run_name": run_name,
                "run_dir": str(run_dir),
                "route": route.get("route", "missing"),
                "mlp_fms_task_pass_count": route.get("mlp_fms_task_pass_count", ""),
                "rational_fms_task_pass_count": route.get("rational_fms_task_pass_count", ""),
                "promotion_allowed": route.get("promotion_allowed", 0),
                "required_artifact_missing_count": route.get("required_artifact_missing_count", ""),
            }
        )
        rows = read_rows(run_dir / "v142_basis_fms_results.csv")
        for row in rows:
            if row.get("family") != "D-RAT":
                continue
            method = row.get("method", "")
            reasons = failure_reasons(row)
            method_key = (run_name, method)
            method_source[method_key].append(fnum(row.get("source_vs_best_control"), 0.0))
            method_auc[method_key].append(fnum(row.get("AUCtime_ratio_vs_best_control"), 9.0))
            if sint(row.get("synthetic_gate_pass"), 0) == 1:
                method_pass_tasks[method_key].add(str(row.get("task", "")))
            for reason in reasons:
                reason_counts[(method, reason)] += 1
            rejection_rows.append(
                {
                    "stage": "V142_RATIONAL_REJECTION_AUDIT",
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "family": row.get("family"),
                    "candidate_id": row.get("candidate_id"),
                    "task": row.get("task"),
                    "seed": row.get("seed"),
                    "loss_interface": row.get("loss_interface"),
                    "method": method,
                    "synthetic_gate_pass": row.get("synthetic_gate_pass"),
                    "failure_reasons": ";".join(reasons),
                    "source_vs_best_control": row.get("source_vs_best_control"),
                    "AUCtime_ratio_vs_best_control": row.get("AUCtime_ratio_vs_best_control"),
                    "CEp99_delta_vs_adamw": row.get("CEp99_delta_vs_adamw"),
                    "NLL_delta_vs_adamw": row.get("NLL_delta_vs_adamw"),
                    "ECE_delta_vs_adamw": row.get("ECE_delta_vs_adamw"),
                    "LineC_majority_pass": row.get("LineC_majority_pass"),
                    "step_time_ratio_vs_adamw": row.get("step_time_ratio_vs_adamw"),
                    "promotion_allowed": 0,
                }
            )

        for row in read_rows(run_dir / "v142_functional_metric_state_trace.csv"):
            if row.get("family") != "D-RAT":
                continue
            method = row.get("method", "")
            state_key = row.get("state_key", "")
            role = role_from_state_key(state_key)
            key = (run_name, method, role)
            role_key_sets[key].add(state_key)
            role_acc[key]["row_count"] += 1.0
            role_acc[key]["abs_state_sum"] += abs(fnum(row.get("state_value"), 0.0))
            role_acc[key]["scale_sum"] += fnum(row.get("scale_value"), 1.0)
            role_acc[key]["scale_min"] = min(role_acc[key].get("scale_min", 999.0), fnum(row.get("scale_value"), 1.0))
            role_acc[key]["scale_max"] = max(role_acc[key].get("scale_max", -999.0), fnum(row.get("scale_value"), 1.0))

    summary_rows: list[dict[str, Any]] = []
    for (run_name, method), sources in sorted(method_source.items()):
        summary_rows.append(
            {
                "stage": "V142_RATIONAL_REJECTION_SUMMARY",
                "run_name": run_name,
                "method": method,
                "rows": len(sources),
                "task_pass_count": len(method_pass_tasks.get((run_name, method), set())),
                "median_source_vs_best_control": median(sources),
                "max_source_vs_best_control": max(sources) if sources else 0.0,
                "median_AUCtime_ratio_vs_best_control": median(method_auc[(run_name, method)]),
                "source_positive_rows": sum(1 for v in sources if v >= 0.002),
                "promotion_allowed": 0,
            }
        )

    reason_rows = [
        {
            "stage": "V142_RATIONAL_REJECTION_REASON_COUNT",
            "method": method,
            "failure_reason": reason,
            "row_count": count,
            "promotion_allowed": 0,
        }
        for (method, reason), count in sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    role_rows: list[dict[str, Any]] = []
    for (run_name, method, role), acc in sorted(role_acc.items()):
        row_count = max(1.0, acc.get("row_count", 0.0))
        role_rows.append(
            {
                "stage": "V142_ROLE_SIGNAL_MASS_AUDIT",
                "run_name": run_name,
                "method": method,
                "role": role,
                "state_key_count": len(role_key_sets[(run_name, method, role)]),
                "trace_row_count": int(acc.get("row_count", 0.0)),
                "mean_abs_state_value": acc.get("abs_state_sum", 0.0) / row_count,
                "mean_scale_value": acc.get("scale_sum", 0.0) / row_count,
                "min_scale_value": acc.get("scale_min", 0.0),
                "max_scale_value": acc.get("scale_max", 0.0),
                "state_keys_sample": ";".join(sorted(role_key_sets[(run_name, method, role)])[:8]),
                "promotion_allowed": 0,
            }
        )

    total_rows = len(rejection_rows)
    pass_rows = sum(1 for r in rejection_rows if sint(r.get("synthetic_gate_pass"), 0) == 1)
    linec_fail = sum(1 for r in rejection_rows if "LineC" in str(r.get("failure_reasons", "")).split(";"))
    time_fail = sum(1 for r in rejection_rows if "step_time" in str(r.get("failure_reasons", "")).split(";"))
    tail_fail = sum(
        1
        for r in rejection_rows
        if any(reason in str(r.get("failure_reasons", "")).split(";") for reason in ["CEp99_tail", "NLL_tail", "ECE_tail"])
    )
    source_positive_rejected = sum(
        1
        for r in rejection_rows
        if fnum(r.get("source_vs_best_control"), 0.0) >= 0.002 and sint(r.get("synthetic_gate_pass"), 0) == 0
    )
    route = {
        "stage": "V142_RATIONAL_REJECTION_AUDIT_ROUTE",
        "diagnostic_route": "D2-RationalLineCTailOverheadRejectedLocalSource"
        if source_positive_rejected > 0
        else "D1-RationalNoSource",
        "official_route_unchanged": "R4-FMSNoGoCurrentDefinition",
        "best_repair_route_observed": "R3-GenericFunctionalOnlyKANSpecificNotEstablished",
        "new_training_executed": 0,
        "uses_existing_artifacts_only": 1,
        "promotion_allowed": 0,
        "total_rational_rows_audited": total_rows,
        "synthetic_gate_pass_rows": pass_rows,
        "source_positive_rejected_rows": source_positive_rejected,
        "linec_rejection_rows": linec_fail,
        "tail_rejection_rows": tail_fail,
        "step_time_rejection_rows": time_fail,
        "route_rows": route_rows,
        "output_dir": str(OUT_DIR),
    }

    write_rows(OUT_DIR / "v142_rational_rejection_audit.csv", rejection_rows)
    write_rows(OUT_DIR / "v142_rational_rejection_summary.csv", summary_rows)
    write_rows(OUT_DIR / "v142_rational_rejection_reason_counts.csv", reason_rows)
    write_rows(OUT_DIR / "v142_role_signal_mass_audit.csv", role_rows)
    write_json(OUT_DIR / "v142_rational_rejection_route.json", route)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
