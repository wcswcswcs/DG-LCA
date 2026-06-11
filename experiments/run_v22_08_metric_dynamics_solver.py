#!/usr/bin/env python3
"""v22.08 Metric-as-dynamics solver gate after retained-source observer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_08_common import (  # noqa: E402
    PYTHON,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    read_json,
    read_rows,
    simple_svg,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default="")
    return p


def _summarize_solver(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    out: list[dict[str, Any]] = []
    for role in sorted({str(r.get("block_role", "")) for r in rows}):
        group = [r for r in rows if str(r.get("block_role", "")) == role]
        pass_rows = sum(
            int(
                finite_float(r.get("projection_residual_Gf"), 999.0) <= 0.35
                and finite_float(r.get("ActuationR2"), -999.0) >= 0.60
                and finite_float(r.get("B2_transfer_gain"), -999.0) >= 0.005
                and finite_float(r.get("B3_safety_gain"), -999.0) >= -0.005
            )
            for r in group
        )
        out.append(
            {
                "block_role": role,
                "rows": len(group),
                "C2_solver_gate_pass_rows": pass_rows,
                "projection_residual_Gf_min": min(finite_float(r.get("projection_residual_Gf"), 999.0) for r in group),
                "ActuationR2_max": max(finite_float(r.get("ActuationR2"), -999.0) for r in group),
                "B2_transfer_gain_max": max(finite_float(r.get("B2_transfer_gain"), -999.0) for r in group),
                "blocker": "" if pass_rows else "projection_residual_or_actuation_or_transfer_gate_failed",
            }
        )
    return out


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    observer_route = read_json(source_dir / "v22_08_retained_source_observer_route.json")
    observer_pass = int_flag(observer_route.get("C0_retained_source_observer_pass_rows"))
    c2_tests = read_rows(source_dir / "v22_08_c2_recompute_tests.csv")
    solver_summary = _summarize_solver(c2_tests)
    c2_pass_rows = sum(int_flag(r.get("C2_solver_gate_pass_rows")) for r in solver_summary)

    if not observer_pass:
        route = {
            "route": "R-ObserverLocalNoGoBoundary",
            "C0_retained_source_observer_pass_rows": observer_pass,
            "C2_recompute_rows_available": len(c2_tests),
            "C2_true_block_solver_pass_rows": c2_pass_rows,
            "C3_source_formation_attempted": 0,
            "C3_source_formation_pass_rows": 0,
            "promotion_allowed": 0,
            "blocker": "retained_source_observer_gate_failed",
            "next_codex_action": "do not continue target scale/cap/floor tuning; define a new train-only retained-source certificate before Line D",
        }
        matrix: list[dict[str, Any]] = []
    else:
        route = {
            "route": "R-SolverProjectionBlocked" if not c2_pass_rows else "S3-MLPSourceFormationOpened-PendingFreshC3",
            "C0_retained_source_observer_pass_rows": observer_pass,
            "C2_recompute_rows_available": len(c2_tests),
            "C2_true_block_solver_pass_rows": c2_pass_rows,
            "C3_source_formation_attempted": int(bool(c2_pass_rows)),
            "C3_source_formation_pass_rows": 0,
            "promotion_allowed": 0,
            "blocker": "" if c2_pass_rows else "C2_solver_gate_failed_after_recompute",
            "next_codex_action": "run fresh selected C3 source-state integration" if c2_pass_rows else "repair true block solver with rank/damping/CG before horizon run",
        }
        matrix = c2_tests

    write_rows(out_dir / "v22_08_c2_true_block_solver_matrix.csv", matrix)
    write_rows(out_dir / "v22_08_c2_true_block_solver_summary.csv", solver_summary)
    write_json(out_dir / "v22_08_metric_dynamics_solver_route.json", route)
    simple_svg(out_dir / "figures/v22_08_solver_projection_residual_vs_actuation.svg", "v22.08 solver projection residual", c2_tests, "projection_residual_Gf")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_08_metric_dynamics_solver.py --source-dir {source_dir} --out-dir {out_dir}",
        status="completed",
        note=f"observer_pass={observer_pass} c2_recompute_rows={len(c2_tests)} c2_pass_rows={c2_pass_rows} route={route['route']}",
    )


if __name__ == "__main__":
    main()

