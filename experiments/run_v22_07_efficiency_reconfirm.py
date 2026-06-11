#!/usr/bin/env python3
"""v22.07 D-CHE/D-FOU full-loop efficiency reconfirm readback."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_07_common import (  # noqa: E402
    PYTHON,
    V2204_OFFICIAL,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    read_rows,
    simple_svg,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--v2204-dir", default=str(V2204_OFFICIAL))
    return p


def _pass(row: dict[str, Any]) -> tuple[int, str]:
    forward = finite_float(row.get("forward_ratio_vs_mlp"), 999.0)
    step = finite_float(row.get("step_ratio_vs_mlp"), 999.0)
    memory = finite_float(row.get("memory_ratio_vs_mlp"), 999.0)
    same_kernel = int_flag(row.get("same_kernel_functional_runner_proof"))
    fallback = int_flag(row.get("fallback_kernel_used"))
    official = int_flag(row.get("official_fused_kernel_complete"))
    blockers = []
    if forward > 1.25:
        blockers.append("forward_ratio")
    if step > 1.25:
        blockers.append("step_ratio")
    if memory > 1.05:
        blockers.append("memory_ratio")
    if not same_kernel:
        blockers.append("same_kernel_functional_runner_proof_missing")
    if fallback:
        blockers.append("fallback_kernel_used")
    if not official:
        blockers.append("official_fused_kernel_incomplete")
    return int(not blockers), ";".join(blockers)


def _normalize(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["variant_family"] = "CHE21-R2/R4" if row.get("carrier") == "D-CHE" else "FOU21-R3"
    out["forward_ratio_vs_mlp"] = row.get("best_forward_ratio", row.get("forward_ratio_vs_mlp", ""))
    out["backward_ratio_vs_mlp"] = row.get("best_backward_ratio", row.get("backward_ratio_vs_mlp", ""))
    out["step_ratio_vs_mlp"] = row.get("best_step_ratio", row.get("step_ratio_vs_mlp", ""))
    out["memory_ratio_vs_mlp"] = row.get("best_memory_ratio", row.get("memory_ratio_vs_mlp", ""))
    out["functional_direction_ms"] = row.get("functional_direction_ms", "")
    out["metric_solver_ms"] = row.get("metric_solver_ms", "")
    out["LineC_audit_ms"] = row.get("mean_linec_audit_ms", "")
    out["horizon_readback_ms"] = row.get("mean_horizon_readback_ms", "")
    out["same_kernel_functional_runner_proof"] = int_flag(row.get("v22_04_same_kernel_runner_proof")) or int_flag(row.get("v22_03_same_kernel_functional_runner_proof"))
    out["official_fused_kernel_complete"] = int_flag(row.get("full_loop_official_closure"))
    out["fallback_kernel_used"] = int_flag(row.get("fallback_kernel_used"))
    out["no_materialize_complete"] = row.get("no_materialize_complete", "")
    out["v22_07_source"] = "v22_04_measured_full_loop_summary_readback"
    passed, blocker = _pass(out)
    out["v22_07_S1_pass"] = passed
    out["v22_07_decision"] = "OfficialEfficientCarrier" if passed else "EfficiencyBlocked"
    out["v22_07_blocker"] = blocker
    return out


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source = Path(args.v2204_dir)
    summary = read_rows(source / "v22_04_efficiency_full_loop_summary.csv")
    rows = [_normalize(r) for r in summary if str(r.get("carrier", "")) in {"D-CHE", "D-FOU"}]
    route = {
        "D-CHE_S1_rows": sum(1 for r in rows if r.get("carrier") == "D-CHE" and int_flag(r.get("v22_07_S1_pass"))),
        "D-FOU_S1_rows": sum(1 for r in rows if r.get("carrier") == "D-FOU" and int_flag(r.get("v22_07_S1_pass"))),
        "D-CHE_S1_pass": int(any(r.get("carrier") == "D-CHE" and int_flag(r.get("v22_07_S1_pass")) for r in rows)),
        "D-FOU_S1_pass": int(any(r.get("carrier") == "D-FOU" and int_flag(r.get("v22_07_S1_pass")) for r in rows)),
        "source_v2204": str(source),
        "missing_v22_07_new_timing_fields": ";".join(
            sorted(
                {
                    key
                    for key in ["functional_direction_ms", "metric_solver_ms", "backward_ratio_vs_mlp", "no_materialize_complete"]
                    if any(str(r.get(key, "")).strip() == "" for r in rows)
                }
            )
        ),
        "decision": "ReadbackFromMeasuredFullLoopArtifacts",
    }
    write_rows(out_dir / "v22_07_efficiency_full_loop_reconfirm.csv", rows)
    write_json(out_dir / "v22_07_efficiency_route.json", route)
    simple_svg(out_dir / "figures/D-CHE_D-FOU_efficiency_dashboard.svg", "v22.07 D-CHE/D-FOU efficiency", rows, "step_ratio_vs_mlp")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_07_efficiency_reconfirm.py --v2204-dir {source} --out-dir {out_dir}",
        status="completed",
        note=f"rows={len(rows)} D-CHE={route['D-CHE_S1_pass']} D-FOU={route['D-FOU_S1_pass']} missing={route['missing_v22_07_new_timing_fields']}",
    )


if __name__ == "__main__":
    main()

