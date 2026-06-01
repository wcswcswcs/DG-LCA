#!/usr/bin/env python
"""Audit v14.2 planned Rational g8/g16 group-granularity coverage.

This is a post-hoc availability audit.  It reads existing substrate/FMS
artifacts and the v12.35 substrate registry; it does not instantiate new
models, run training, or promote any row.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dgkan.diagnostics.basis_workspace import V1235_BASIS_CANDIDATES


ROOT = Path("results/v14_2_functional_first_all_basis_parallel")
OUT = ROOT / "group_granularity_availability_v142"
SUBSTRATE_SWEEP = ROOT / "substrate_sweep_v142" / "v142_all_basis_substrate_sweep.csv"


PLANNED_VARIANTS = [
    ("RAT-A", "GroupRational-Horner-g8", "G8", "horner"),
    ("RAT-B", "GroupRational-Horner-g16", "G16", "horner"),
    ("RAT-C", "GroupRational-TritonEval-g8", "G8", "triton"),
    ("RAT-D", "GroupRational-TritonEval-g16", "G16", "triton"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def group_token(method_id: str) -> str:
    low = method_id.lower()
    for pattern in (r"flashgroup[-_]?g(\d+)", r"flashkat_g(\d+)"):
        match = re.search(pattern, low)
        if match:
            return f"G{match.group(1)}"
    return "UNKNOWN"


def eval_token(candidate: object) -> str:
    return "triton" if int(getattr(candidate, "uses_triton", 0)) else "horner"


def collect_candidate_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate_id, candidate in V1235_BASIS_CANDIDATES.items():
        if getattr(candidate, "family", "") != "D-RAT":
            continue
        method_id = getattr(candidate, "method_id", "")
        rows.append(
            {
                "candidate_id": candidate_id,
                "family": getattr(candidate, "family", ""),
                "method_id": method_id,
                "group_granularity": group_token(method_id),
                "eval_backend": eval_token(candidate),
                "uses_triton": int(getattr(candidate, "uses_triton", 0)),
                "exact_kernel_implemented": int(getattr(candidate, "exact_kernel_implemented", 0)),
                "materializes_basis_tensor": int(getattr(candidate, "materializes_basis_tensor", 0)),
                "actual_symbol": getattr(candidate, "actual_symbol", ""),
            }
        )
    return rows


def collect_substrate_status() -> dict[str, dict[str, str]]:
    status: dict[str, dict[str, str]] = {}
    for row in read_csv(SUBSTRATE_SWEEP):
        status[row.get("candidate_id", "")] = row
    return status


def collect_fms_observations(candidate_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    candidate_meta = {str(row["candidate_id"]): row for row in candidate_rows}
    buckets: dict[tuple[str, str, str], dict[str, object]] = {}
    for path in sorted(ROOT.glob("*/v142_basis_fms_results.csv")):
        run_name = path.parent.name
        for row in read_csv(path):
            candidate_id = row.get("candidate_id", "")
            if candidate_id not in candidate_meta:
                continue
            key = (run_name, candidate_id, row.get("method", ""))
            bucket = buckets.setdefault(
                key,
                {
                    "run_name": run_name,
                    "candidate_id": candidate_id,
                    "group_granularity": candidate_meta[candidate_id]["group_granularity"],
                    "eval_backend": candidate_meta[candidate_id]["eval_backend"],
                    "method": row.get("method", ""),
                    "rows": 0,
                    "synthetic_gate_pass_rows": 0,
                    "task_pass_set": set(),
                    "source_positive_rows": 0,
                    "linec_fail_rows": 0,
                    "tail_fail_rows": 0,
                    "time_fail_rows": 0,
                },
            )
            bucket["rows"] = int(bucket["rows"]) + 1
            gate_pass = int(float(row.get("synthetic_gate_pass", "0") or 0))
            bucket["synthetic_gate_pass_rows"] = int(bucket["synthetic_gate_pass_rows"]) + gate_pass
            if gate_pass:
                bucket["task_pass_set"].add(row.get("task", ""))
            source = float(row.get("source_vs_best_control", "0") or 0)
            if source > 0.0:
                bucket["source_positive_rows"] = int(bucket["source_positive_rows"]) + 1
            if int(float(row.get("LineC_majority_pass", "0") or 0)) == 0:
                bucket["linec_fail_rows"] = int(bucket["linec_fail_rows"]) + 1
            if (
                float(row.get("CEp99_delta_vs_adamw", "0") or 0) > 0.0
                or float(row.get("NLL_delta_vs_adamw", "0") or 0) > 0.0
                or float(row.get("ECE_delta_vs_adamw", "0") or 0) > 0.0
            ):
                bucket["tail_fail_rows"] = int(bucket["tail_fail_rows"]) + 1
            if float(row.get("step_time_ratio_vs_adamw", "0") or 0) > 1.10:
                bucket["time_fail_rows"] = int(bucket["time_fail_rows"]) + 1

    out_rows: list[dict[str, object]] = []
    for bucket in buckets.values():
        task_pass_set = bucket.pop("task_pass_set")
        bucket["task_pass_count"] = len(task_pass_set)
        bucket["task_passes"] = ",".join(sorted(task_pass_set))
        out_rows.append(bucket)
    out_rows.sort(key=lambda row: (-int(row["task_pass_count"]), row["run_name"], row["method"]))
    return out_rows


def build_variant_rows(
    candidate_rows: list[dict[str, object]],
    substrate_status: dict[str, dict[str, str]],
    fms_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant_id, planned_name, group, backend in PLANNED_VARIANTS:
        matches = [
            row
            for row in candidate_rows
            if row["group_granularity"] == group and row["eval_backend"] == backend
        ]
        candidate_ids = [str(row["candidate_id"]) for row in matches]
        substrate_pass_ids = [
            cid
            for cid in candidate_ids
            if int(float(substrate_status.get(cid, {}).get("v142_substrate_gate_pass", "0") or 0)) == 1
        ]
        fms_matches = [row for row in fms_rows if row["candidate_id"] in set(candidate_ids)]
        best_task_pass = max([int(row["task_pass_count"]) for row in fms_matches], default=0)
        best_method = ""
        best_run = ""
        if fms_matches:
            best = max(fms_matches, key=lambda row: int(row["task_pass_count"]))
            best_method = str(best["method"])
            best_run = str(best["run_name"])
        if not candidate_ids:
            status = "UnavailableInCurrentV1235SubstrateMap"
        elif not substrate_pass_ids:
            status = "AvailableButNoV142SubstratePass"
        elif best_task_pass < 5:
            status = "SubstrateCoveredButNoKANSpecificFMS"
        else:
            status = "CoveredAndWouldRequireOfficialConfirmation"
        rows.append(
            {
                "planned_variant_id": variant_id,
                "planned_variant": planned_name,
                "group_granularity": group,
                "eval_backend": backend,
                "available_candidate_count": len(candidate_ids),
                "candidate_ids": ",".join(candidate_ids),
                "v142_substrate_pass_count": len(substrate_pass_ids),
                "v142_substrate_pass_candidate_ids": ",".join(substrate_pass_ids),
                "best_existing_fms_task_pass_count": best_task_pass,
                "best_existing_fms_run": best_run,
                "best_existing_fms_method": best_method,
                "status": status,
                "promotion_allowed": 0,
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidate_rows = collect_candidate_rows()
    substrate_status = collect_substrate_status()
    fms_rows = collect_fms_observations(candidate_rows)
    variant_rows = build_variant_rows(candidate_rows, substrate_status, fms_rows)

    write_csv(
        OUT / "v142_group_granularity_candidate_inventory.csv",
        candidate_rows,
        [
            "candidate_id",
            "family",
            "method_id",
            "group_granularity",
            "eval_backend",
            "uses_triton",
            "exact_kernel_implemented",
            "materializes_basis_tensor",
            "actual_symbol",
        ],
    )
    write_csv(
        OUT / "v142_group_granularity_variant_coverage.csv",
        variant_rows,
        [
            "planned_variant_id",
            "planned_variant",
            "group_granularity",
            "eval_backend",
            "available_candidate_count",
            "candidate_ids",
            "v142_substrate_pass_count",
            "v142_substrate_pass_candidate_ids",
            "best_existing_fms_task_pass_count",
            "best_existing_fms_run",
            "best_existing_fms_method",
            "status",
            "promotion_allowed",
        ],
    )
    write_csv(
        OUT / "v142_group_granularity_existing_fms_summary.csv",
        fms_rows,
        [
            "run_name",
            "candidate_id",
            "group_granularity",
            "eval_backend",
            "method",
            "rows",
            "synthetic_gate_pass_rows",
            "task_pass_count",
            "task_passes",
            "source_positive_rows",
            "linec_fail_rows",
            "tail_fail_rows",
            "time_fail_rows",
        ],
    )

    g8_available = any(row["group_granularity"] == "G8" for row in candidate_rows)
    g16_triton = [row for row in candidate_rows if row["group_granularity"] == "G16" and row["eval_backend"] == "triton"]
    g16_substrate_pass = sum(
        int(float(substrate_status.get(str(row["candidate_id"]), {}).get("v142_substrate_gate_pass", "0") or 0))
        for row in g16_triton
    )
    best_existing_fms_task_pass = max([int(row["task_pass_count"]) for row in fms_rows], default=0)
    route = {
        "diagnostic_route": "D3-GroupGranularityG8UnavailableG16TritonCoveredNoKANSpecificFMS",
        "official_route_unchanged": "R4-FMSNoGoCurrentDefinition",
        "best_repair_route_observed": "R3-GenericFunctionalOnlyKANSpecificNotEstablished",
        "uses_existing_artifacts_only": 1,
        "new_training_executed": 0,
        "line_a_k_architecture_search_executed": 0,
        "promotion_allowed": 0,
        "g8_available_in_v1235_substrate_map": int(g8_available),
        "g16_triton_candidate_count": len(g16_triton),
        "g16_triton_v142_substrate_pass_count": g16_substrate_pass,
        "best_existing_g16_fms_task_pass_count": best_existing_fms_task_pass,
        "planned_variant_rows": len(variant_rows),
        "candidate_inventory_rows": len(candidate_rows),
        "existing_fms_summary_rows": len(fms_rows),
        "notes": [
            "RAT-A/RAT-C g8 variants are not present in the current v12.35 substrate registry used by v14.2.",
            "RAT-D g16 TritonEval is covered by existing D-RAT G16 candidates, but existing FMS repairs remain below the >=5/7 KAN-specific gate.",
            "No promotion or real short-run can be opened by this availability audit.",
        ],
    }
    (OUT / "v142_group_granularity_route.json").write_text(json.dumps(route, indent=2) + "\n")


if __name__ == "__main__":
    main()
