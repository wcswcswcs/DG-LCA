#!/usr/bin/env python3
"""Audit cross-plan Non-RAT substrate repairs against v14.2 strict gates.

This script reads existing v12.34.2 foreach-off Non-RAT repair artifacts and
re-evaluates them under the v14.2 substrate-entry constraints. It does not run
training, instantiate models, or promote any row.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("results/v14_2_functional_first_all_basis_parallel")
OUT = ROOT / "nonrat_crossplan_substrate_repair_v142"
SRC = Path("results/v12_34_2_all_basis_substrate_functional_repair/official_v12342")

WORKSPACE = SRC / "v12342_nonrat_lifetime_foreachoff_workspace_truth.csv"
HARDENING = SRC / "v12342_nonrat_lifetime_foreachoff_hardening.csv"
FUNCTIONAL_P3 = SRC / "v12342_basis_functional_nonrat_foreachoff_p3.csv"
FUNCTIONAL_P4 = SRC / "v12342_basis_functional_nonrat_foreachoff_p4.csv"

V142_STEP_GATE = 1.75
V142_RAW_MEMORY_GATE = 1.75
V142_INCREMENTAL_MEMORY_GATE = 1.75


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
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def failure_reasons(row: dict[str, str]) -> list[str]:
    raw = fnum(row.get("raw_memory_ratio_vs_mlp"), 9.0)
    inc = fnum(row.get("incremental_memory_ratio_vs_mlp"), 9.0)
    step = fnum(row.get("step_ratio_vs_mlp"), 9.0)
    reasons: list[str] = []
    if raw > V142_RAW_MEMORY_GATE:
        reasons.append("raw_memory_ratio")
    if inc > V142_INCREMENTAL_MEMORY_GATE:
        reasons.append("incremental_memory_ratio")
    if step > V142_STEP_GATE:
        reasons.append("step_ratio")
    if sint(row.get("exact_kernel_implemented"), 0) != 1:
        reasons.append("exact_kernel")
    if sint(row.get("materializes_basis_tensor"), 1) != 0:
        reasons.append("basis_materialization")
    if sint(row.get("materializes_derivative_tensor"), 1) != 0:
        reasons.append("derivative_materialization")
    if sint(row.get("uses_label_or_ce_for_direction"), 1) != 0:
        reasons.append("forbidden_direction_source")
    if sint(row.get("forbidden_token_present"), 1) != 0:
        reasons.append("forbidden_token")
    return reasons


def v142_strict_workspace_pass(row: dict[str, str]) -> int:
    return int(not failure_reasons(row))


def summarize_p3(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    buckets: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "p3_rows": 0,
            "p3_executed_rows": 0,
            "p3_pass_rows": 0,
            "p3_source_positive_rows": 0,
            "p3_best_source_vs_best_control": -999.0,
        }
    )
    for row in rows:
        key = (row.get("family", ""), row.get("base_candidate_id", ""))
        bucket = buckets[key]
        bucket["p3_rows"] += 1
        executed = sint(row.get("executed"), 0)
        p3_pass = sint(row.get("p3_pass"), 0)
        source = fnum(row.get("source_vs_best_control"), -999.0)
        bucket["p3_executed_rows"] += executed
        bucket["p3_pass_rows"] += p3_pass
        if executed and source > 0.0:
            bucket["p3_source_positive_rows"] += 1
        bucket["p3_best_source_vs_best_control"] = max(bucket["p3_best_source_vs_best_control"], source)
    return buckets


def summarize_p4(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    buckets: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"p4_rows": 0, "p4_executed_rows": 0, "p4_pass_rows": 0}
    )
    for row in rows:
        key = (row.get("family", ""), row.get("base_candidate_id", ""))
        bucket = buckets[key]
        bucket["p4_rows"] += 1
        bucket["p4_executed_rows"] += sint(row.get("executed"), 0)
        bucket["p4_pass_rows"] += sint(row.get("p4_pass"), 0)
    return buckets


def main() -> None:
    workspace_rows = read_rows(WORKSPACE)
    hardening_rows = read_rows(HARDENING)
    p3_rows = read_rows(FUNCTIONAL_P3)
    p4_rows = read_rows(FUNCTIONAL_P4)
    p3_by_candidate = summarize_p3(p3_rows)
    p4_by_candidate = summarize_p4(p4_rows)

    repair_rows: list[dict[str, Any]] = []
    for row in workspace_rows:
        family = row.get("family", "")
        candidate_id = row.get("candidate_id", "")
        reasons = failure_reasons(row)
        strict_pass = int(not reasons)
        original_workspace_pass = sint(row.get("workspace_gate_pass"), 0)
        p3 = p3_by_candidate.get((family, candidate_id), {})
        p4 = p4_by_candidate.get((family, candidate_id), {})
        if strict_pass:
            status = "V142StrictWorkspacePassNeedsFreshV142FunctionalProof"
        elif original_workspace_pass and "incremental_memory_ratio" in reasons:
            status = "V12342WorkspacePassButV142IncrementalMemoryFail"
        elif "incremental_memory_ratio" in reasons:
            status = "WorkspaceRepairStillFailsV142IncrementalMemory"
        else:
            status = "WorkspaceRepairStillFailsV142Gate"
        repair_rows.append(
            {
                "stage": "V142_NONRAT_CROSSPLAN_SUBSTRATE_REPAIR_AUDIT",
                "source_stage": row.get("stage"),
                "family": family,
                "candidate_id": candidate_id,
                "mapped_method_id": row.get("mapped_method_id"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "v12342_workspace_gate_pass": original_workspace_pass,
                "v12342_workspace_strong_gate_pass": sint(row.get("workspace_strong_gate_pass"), 0),
                "v142_strict_workspace_pass": strict_pass,
                "raw_memory_ratio_vs_mlp": fnum(row.get("raw_memory_ratio_vs_mlp"), 9.0),
                "incremental_memory_ratio_vs_mlp": fnum(row.get("incremental_memory_ratio_vs_mlp"), 9.0),
                "step_ratio_vs_mlp": fnum(row.get("step_ratio_vs_mlp"), 9.0),
                "exact_kernel_implemented": sint(row.get("exact_kernel_implemented"), 0),
                "uses_dense_basis_materialization": sint(row.get("uses_dense_basis_materialization"), 1),
                "materializes_basis_tensor": sint(row.get("materializes_basis_tensor"), 1),
                "materializes_derivative_tensor": sint(row.get("materializes_derivative_tensor"), 1),
                "uses_label_or_ce_for_direction": sint(row.get("uses_label_or_ce_for_direction"), 1),
                "forbidden_token_present": sint(row.get("forbidden_token_present"), 1),
                "p3_rows": p3.get("p3_rows", 0),
                "p3_executed_rows": p3.get("p3_executed_rows", 0),
                "p3_pass_rows": p3.get("p3_pass_rows", 0),
                "p3_source_positive_rows": p3.get("p3_source_positive_rows", 0),
                "p3_best_source_vs_best_control": p3.get("p3_best_source_vs_best_control", ""),
                "p4_rows": p4.get("p4_rows", 0),
                "p4_executed_rows": p4.get("p4_executed_rows", 0),
                "p4_pass_rows": p4.get("p4_pass_rows", 0),
                "failure_reasons": "pass" if strict_pass else ";".join(reasons),
                "status": status,
                "promotion_allowed": 0,
                "new_training_executed": 0,
                "source_artifact": str(WORKSPACE),
            }
        )

    family_rows: list[dict[str, Any]] = []
    for family in sorted({row.get("family", "") for row in repair_rows}):
        rows = [row for row in repair_rows if row.get("family") == family]
        p3_family = [row for row in p3_rows if row.get("family") == family]
        p4_family = [row for row in p4_rows if row.get("family") == family]
        hardening_family = [row for row in hardening_rows if row.get("family") == family]
        best_inc = min((fnum(row.get("incremental_memory_ratio_vs_mlp"), 9.0) for row in rows), default=9.0)
        best_raw = min((fnum(row.get("raw_memory_ratio_vs_mlp"), 9.0) for row in rows), default=9.0)
        best_step = min((fnum(row.get("step_ratio_vs_mlp"), 9.0) for row in rows), default=9.0)
        exact_no_materialize = sum(
            1
            for row in rows
            if sint(row.get("exact_kernel_implemented"), 0) == 1
            and sint(row.get("materializes_basis_tensor"), 1) == 0
            and sint(row.get("materializes_derivative_tensor"), 1) == 0
        )
        family_rows.append(
            {
                "stage": "V142_NONRAT_CROSSPLAN_SUBSTRATE_REPAIR_FAMILY_SUMMARY",
                "family": family,
                "workspace_rows": len(rows),
                "v12342_workspace_gate_pass_rows": sum(sint(row.get("v12342_workspace_gate_pass"), 0) for row in rows),
                "v12342_workspace_strong_gate_pass_rows": sum(
                    sint(row.get("v12342_workspace_strong_gate_pass"), 0) for row in rows
                ),
                "v142_strict_workspace_pass_rows": sum(sint(row.get("v142_strict_workspace_pass"), 0) for row in rows),
                "exact_no_materialize_rows": exact_no_materialize,
                "best_raw_memory_ratio_vs_mlp": best_raw,
                "best_incremental_memory_ratio_vs_mlp": best_inc,
                "best_step_ratio_vs_mlp": best_step,
                "hardening_executed_rows": sum(sint(row.get("executed"), 0) for row in hardening_family),
                "hardening_family_near_pass_rows": sum(sint(row.get("family_near_pass"), 0) for row in hardening_family),
                "p3_rows": len(p3_family),
                "p3_executed_rows": sum(sint(row.get("executed"), 0) for row in p3_family),
                "p3_pass_rows": sum(sint(row.get("p3_pass"), 0) for row in p3_family),
                "p3_source_positive_rows": sum(
                    1
                    for row in p3_family
                    if sint(row.get("executed"), 0) and fnum(row.get("source_vs_best_control"), -999.0) > 0.0
                ),
                "p4_rows": len(p4_family),
                "p4_executed_rows": sum(sint(row.get("executed"), 0) for row in p4_family),
                "p4_pass_rows": sum(sint(row.get("p4_pass"), 0) for row in p4_family),
                "status": "V142StrictSubstratePass"
                if sum(sint(row.get("v142_strict_workspace_pass"), 0) for row in rows) > 0
                else "FailClosedNoV142StrictSubstratePass",
                "promotion_allowed": 0,
            }
        )

    functional_rows: list[dict[str, Any]] = []
    for family in sorted({row.get("family", "") for row in p3_rows + p4_rows}):
        p3_family = [row for row in p3_rows if row.get("family") == family]
        p4_family = [row for row in p4_rows if row.get("family") == family]
        functional_rows.append(
            {
                "stage": "V142_NONRAT_CROSSPLAN_FUNCTIONAL_REPAIR_SUMMARY",
                "family": family,
                "p3_rows": len(p3_family),
                "p3_executed_rows": sum(sint(row.get("executed"), 0) for row in p3_family),
                "p3_pass_rows": sum(sint(row.get("p3_pass"), 0) for row in p3_family),
                "p3_source_positive_rows": sum(
                    1
                    for row in p3_family
                    if sint(row.get("executed"), 0) and fnum(row.get("source_vs_best_control"), -999.0) > 0.0
                ),
                "p4_rows": len(p4_family),
                "p4_executed_rows": sum(sint(row.get("executed"), 0) for row in p4_family),
                "p4_pass_rows": sum(sint(row.get("p4_pass"), 0) for row in p4_family),
                "promotion_allowed": 0,
            }
        )

    v142_strict_pass_rows = sum(sint(row.get("v142_strict_workspace_pass"), 0) for row in repair_rows)
    route = {
        "stage": "V142_NONRAT_CROSSPLAN_SUBSTRATE_REPAIR_ROUTE",
        "diagnostic_route": "D4-NonRATCrossPlanRepairStillFailsV142SubstrateGate"
        if v142_strict_pass_rows == 0
        else "D4-NonRATCrossPlanRepairHasStrictSubstrateCandidateNeedsFreshV142Proof",
        "official_route_unchanged": "R4-FMSNoGoCurrentDefinition",
        "best_repair_route_observed": "R3-GenericFunctionalOnlyKANSpecificNotEstablished",
        "uses_existing_artifacts_only": 1,
        "new_training_executed": 0,
        "promotion_allowed": 0,
        "real_short_run_open_allowed": 0,
        "source_v12342_workspace_rows": len(workspace_rows),
        "source_v12342_workspace_gate_pass_rows": sum(sint(row.get("workspace_gate_pass"), 0) for row in workspace_rows),
        "source_v12342_workspace_strong_gate_pass_rows": sum(
            sint(row.get("workspace_strong_gate_pass"), 0) for row in workspace_rows
        ),
        "v142_strict_workspace_pass_rows": v142_strict_pass_rows,
        "nonrat_families_with_v142_strict_workspace_pass": sorted(
            {row.get("family") for row in repair_rows if sint(row.get("v142_strict_workspace_pass"), 0)}
        ),
        "source_p3_rows": len(p3_rows),
        "source_p3_executed_rows": sum(sint(row.get("executed"), 0) for row in p3_rows),
        "source_p3_pass_rows": sum(sint(row.get("p3_pass"), 0) for row in p3_rows),
        "source_p4_rows": len(p4_rows),
        "source_p4_executed_rows": sum(sint(row.get("executed"), 0) for row in p4_rows),
        "source_p4_pass_rows": sum(sint(row.get("p4_pass"), 0) for row in p4_rows),
        "output_dir": str(OUT),
    }

    write_rows(OUT / "v142_nonrat_crossplan_workspace_repair.csv", repair_rows)
    write_rows(OUT / "v142_nonrat_crossplan_family_summary.csv", family_rows)
    write_rows(OUT / "v142_nonrat_crossplan_functional_summary.csv", functional_rows)
    write_json(OUT / "v142_nonrat_crossplan_route.json", route)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
