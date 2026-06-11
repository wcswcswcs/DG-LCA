#!/usr/bin/env python3
"""Summarize v22.03 D-RAT/D-RBF limited existing-carrier smoke.

The source smoke is intentionally conservative: it is produced by the
existing PrimitiveKAN carrier path in ``run_v21_01_source_retention.py``.
It is therefore a readback of whether the current functional carrier path
opens source retention, not proof that the active micro-kernel repair has
been wired as a production official fused runner.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_03_common import (  # noqa: E402
    PYTHON,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    read_rows,
    write_json,
    write_rows,
)


SOURCE_SUMMARY = "v21_01_source_retention_summary.csv"
SOURCE_MATRIX = "v21_01_source_retention_matrix.csv"
LIMITED_SCOPE = "existing_primitivekan_carrier_path"
PASS_EPS = 0.005


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", required=True)
    p.add_argument("--tag", default="drat_drbf_limited_smoke")
    p.add_argument("--label", default="D-RAT/D-RBF existing-carrier limited functional smoke")
    return p


def _is_candidate(row: dict[str, Any]) -> bool:
    return not str(row.get("v21_id", "")).startswith("CTRL")


def _positive(value: Any) -> int:
    return int(finite_float(value, -1.0e9) >= PASS_EPS)


def _row_decision(row: dict[str, Any]) -> tuple[str, str]:
    h800 = _positive(row.get("source_h800_mean"))
    h1600 = _positive(row.get("source_h1600_mean"))
    if not h800:
        return "ExistingCarrierSmokeNoH800Source", "h800_source_missing"
    if not h1600:
        return "ExistingCarrierSmokeNoH1600Retention", "h1600_retention_missing"
    return "ExistingCarrierSmokeEarlySourceOnly", "official_fused_missing;micro_kernel_path_not_used"


def summarize(source_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = read_rows(source_dir / SOURCE_SUMMARY)
    matrix_rows = read_rows(source_dir / SOURCE_MATRIX)
    smoke_rows: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("carrier", "")) not in {"D-RAT", "D-RBF"}:
            continue
        decision, blocker = _row_decision(row) if _is_candidate(row) else ("ControlRow", "control")
        smoke_rows.append(
            {
                "carrier": row.get("carrier", ""),
                "limited_smoke_scope": LIMITED_SCOPE,
                "basis_repair_variant": row.get("basis_repair_variant", ""),
                "v21_id": row.get("v21_id", ""),
                "mechanism": row.get("mechanism", ""),
                "hypothesis": row.get("hypothesis", ""),
                "rows": row.get("rows", ""),
                "source_h100_mean": row.get("source_h100_mean", ""),
                "source_h100_pass_count": row.get("source_h100_pass_count", ""),
                "source_h400_mean": row.get("source_h400_mean", ""),
                "source_h400_pass_count": row.get("source_h400_pass_count", ""),
                "source_h800_mean": row.get("source_h800_mean", ""),
                "source_h800_pass_count": row.get("source_h800_pass_count", ""),
                "source_h1600_mean": row.get("source_h1600_mean", ""),
                "source_h1600_pass_count": row.get("source_h1600_pass_count", ""),
                "weak_h1600_candidate": row.get("weak_h1600_candidate", ""),
                "productive_h3200_candidate": row.get("productive_h3200_candidate", ""),
                "productive_h4800_candidate": row.get("productive_h4800_candidate", ""),
                "decision": decision,
                "blocker": blocker,
                "uses_micro_kernel_path": 0,
                "official_fused_kernel_complete": 0,
                "promotion_allowed": 0,
            }
        )

    route_rows: list[dict[str, Any]] = []
    overall_h800 = 0
    overall_h1600 = 0
    for carrier in ["D-RAT", "D-RBF"]:
        carrier_rows = [r for r in smoke_rows if r.get("carrier") == carrier]
        candidates = [r for r in carrier_rows if not str(r.get("v21_id", "")).startswith("CTRL")]
        h800_candidates = sum(_positive(r.get("source_h800_mean")) for r in candidates)
        h1600_candidates = sum(_positive(r.get("source_h1600_mean")) for r in candidates)
        overall_h800 += h800_candidates
        overall_h1600 += h1600_candidates
        best_h800 = max(candidates, key=lambda r: finite_float(r.get("source_h800_mean"), -1.0e9), default={})
        positive_early = [
            r
            for r in candidates
            if _positive(r.get("source_h100_mean")) or _positive(r.get("source_h400_mean"))
        ]
        best_early = max(
            positive_early,
            key=lambda r: max(finite_float(r.get("source_h100_mean"), -1.0e9), finite_float(r.get("source_h400_mean"), -1.0e9)),
            default={},
        )
        if h800_candidates == 0:
            decision = "ExistingCarrierSmokeNoH800Source"
            blocker = "h800_source_missing;micro_kernel_path_not_used;official_fused_missing"
        elif h1600_candidates == 0:
            decision = "ExistingCarrierSmokeNoH1600Retention"
            blocker = "h1600_retention_missing;micro_kernel_path_not_used;official_fused_missing"
        else:
            decision = "ExistingCarrierSmokeEarlySourceOnly"
            blocker = "micro_kernel_path_not_used;official_fused_missing"
        route_rows.append(
            {
                "carrier": carrier,
                "source_dir": str(source_dir),
                "limited_smoke_scope": LIMITED_SCOPE,
                "rows": len(carrier_rows),
                "candidate_rows": len(candidates),
                "control_rows": len(carrier_rows) - len(candidates),
                "candidate_h100_positive": sum(_positive(r.get("source_h100_mean")) for r in candidates),
                "candidate_h400_positive": sum(_positive(r.get("source_h400_mean")) for r in candidates),
                "candidate_h800_positive": h800_candidates,
                "candidate_h1600_positive": h1600_candidates,
                "best_h800_v21_id": best_h800.get("v21_id", ""),
                "best_h800": best_h800.get("source_h800_mean", ""),
                "best_h1600_for_best_h800": best_h800.get("source_h1600_mean", ""),
                "best_positive_early_v21_id": best_early.get("v21_id", ""),
                "best_positive_early_h100": best_early.get("source_h100_mean", ""),
                "best_positive_early_h400": best_early.get("source_h400_mean", ""),
                "decision": decision,
                "blocker": blocker,
                "uses_micro_kernel_path": 0,
                "official_fused_kernel_complete": 0,
                "promotion_allowed": 0,
            }
        )

    if overall_h800 == 0:
        overall_decision = "ExistingCarrierSmokeNoH800Source"
    elif overall_h1600 == 0:
        overall_decision = "ExistingCarrierSmokeNoH1600Retention"
    else:
        overall_decision = "ExistingCarrierSmokeEarlySourceOnly"
    route = {
        "source_dir": str(source_dir),
        "limited_smoke_scope": LIMITED_SCOPE,
        "source_summary_rows": len(rows),
        "source_matrix_rows": len(matrix_rows),
        "carrier_rows": len(smoke_rows),
        "candidate_h800_positive": overall_h800,
        "candidate_h1600_positive": overall_h1600,
        "decision": overall_decision,
        "blocker": "h800_source_missing;micro_kernel_path_not_used;official_fused_missing"
        if overall_h800 == 0
        else "h1600_retention_missing;micro_kernel_path_not_used;official_fused_missing",
        "uses_micro_kernel_path": 0,
        "official_fused_kernel_complete": 0,
        "promotion_allowed": 0,
    }
    return smoke_rows, route_rows, route


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir)
    smoke_rows, route_rows, route = summarize(source_dir)
    route["label"] = str(args.label)
    tag = str(args.tag).strip() or "drat_drbf_limited_smoke"
    write_rows(out_dir / f"v22_03_{tag}_summary.csv", smoke_rows)
    write_rows(out_dir / f"v22_03_{tag}_route.csv", route_rows)
    write_json(out_dir / f"v22_03_{tag}_route.json", route)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_03_drat_drbf_limited_smoke_summary.py --out-dir {out_dir} --source-dir {source_dir} --tag {tag}",
        status="completed",
        note=f"decision={route['decision']} carrier_rows={route['carrier_rows']}",
    )


if __name__ == "__main__":
    main()
