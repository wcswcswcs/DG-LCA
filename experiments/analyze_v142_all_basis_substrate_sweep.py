#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


SOURCE = Path("results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235/v1235_basis_substrate_health.csv")
OUT_DIR = Path("results/v14_2_functional_first_all_basis_parallel/substrate_sweep_v142")


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if out != out or out in (float("inf"), float("-inf")):
            return default
        return out
    except Exception:
        return default


def sint(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


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


def main() -> None:
    rows = list(csv.DictReader(SOURCE.open(newline="", encoding="utf-8")))
    out: list[dict[str, Any]] = []
    for r in rows:
        step = fnum(r.get("step_ratio_vs_mlp"), 9.0)
        memory = max(fnum(r.get("raw_memory_ratio_vs_mlp"), 9.0), fnum(r.get("incremental_memory_ratio_vs_mlp"), 9.0))
        mean_delta = fnum(r.get("mean_delta_vs_mlp"), -999.0)
        worst_delta = fnum(r.get("worst_delta_vs_mlp"), -999.0)
        auc = fnum(r.get("AUC_time_ratio_vs_mlp"), 9.0)
        linec = fnum(r.get("LineC_pass_rate"), 0.0)
        telemetry = sint(r.get("telemetry_available"), 0)
        blockers = []
        if step > 1.75:
            blockers.append("step_ratio")
        if memory > 1.75:
            blockers.append("memory_ratio")
        if mean_delta < -0.05:
            blockers.append("mean_delta")
        if worst_delta < -0.10:
            blockers.append("worst_delta")
        if auc > 2.0:
            blockers.append("auc_time")
        if linec < 0.30:
            blockers.append("linec")
        if telemetry != 1:
            blockers.append("telemetry")
        gate = int(not blockers)
        out.append(
            {
                "stage": "V142_ALL_BASIS_SUBSTRATE_SWEEP",
                "family": r.get("family"),
                "candidate_id": r.get("candidate_id"),
                "v142_substrate_gate_pass": gate,
                "step_ratio_vs_mlp": step,
                "memory_ratio_vs_mlp": memory,
                "mean_delta_vs_mlp": mean_delta,
                "worst_delta_vs_mlp": worst_delta,
                "AUC_time_ratio_vs_mlp": auc,
                "LineC_pass_rate": linec,
                "telemetry_available": telemetry,
                "failure_reason": "pass" if gate else ";".join(blockers),
                "promotion_allowed": 0,
                "source_artifact": str(SOURCE),
            }
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_rows(OUT_DIR / "v142_all_basis_substrate_sweep.csv", out)
    family_rows: list[dict[str, Any]] = []
    for family in sorted({str(r["family"]) for r in out}):
        fr = [r for r in out if str(r["family"]) == family]
        pass_rows = [r for r in fr if sint(r["v142_substrate_gate_pass"]) == 1]
        family_rows.append(
            {
                "stage": "V142_ALL_BASIS_SUBSTRATE_SWEEP_SUMMARY",
                "family": family,
                "rows": len(fr),
                "pass_rows": len(pass_rows),
                "best_candidate": pass_rows[0]["candidate_id"] if pass_rows else sorted(fr, key=lambda x: (fnum(x["mean_delta_vs_mlp"], -999.0), -fnum(x["memory_ratio_vs_mlp"], 999.0)), reverse=True)[0]["candidate_id"],
                "promotion_allowed": 0,
            }
        )
    write_rows(OUT_DIR / "v142_all_basis_substrate_sweep_summary.csv", family_rows)
    route = {
        "stage": "V142_ALL_BASIS_SUBSTRATE_SWEEP_ROUTE",
        "source_rows": len(rows),
        "v142_pass_rows": sum(sint(r["v142_substrate_gate_pass"]) for r in out),
        "families_with_pass": sorted({str(r["family"]) for r in out if sint(r["v142_substrate_gate_pass"]) == 1}),
        "nonrat_families_with_pass": sorted({str(r["family"]) for r in out if sint(r["v142_substrate_gate_pass"]) == 1 and str(r["family"]) != "D-RAT"}),
        "promotion_allowed": 0,
        "new_training_executed": 0,
    }
    (OUT_DIR / "v142_all_basis_substrate_sweep_route.json").write_text(json.dumps(route, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
