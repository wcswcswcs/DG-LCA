#!/usr/bin/env python3
"""v22.09 terminal preservation gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_09_common import PYTHON, append_exec, ensure_out, int_flag, read_json, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default="")
    return p


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    solver = read_json(source_dir / "v22_09_metric_dynamics_solver_route.json")
    c3_pass = int_flag(solver.get("C3_source_formation_pass_rows"))
    if not c3_pass:
        rows: list[dict[str, Any]] = [
            {
                "terminal_preservation_status": "blocked_before_C4",
                "source_h3200": "",
                "source_h4000": "",
                "source_h4800": "",
                "source_h6400": "",
                "R4800_over_3200": "",
                "R6400_over_3200": "",
                "row_h4800_positive_count": 0,
                "C4_terminal_preservation_pass": 0,
                "blocker": "C3_source_formation_gate_failed",
            }
        ]
        route = {
            "route": "D0D1BlockedBeforeTerminalPreservation_v22.09",
            "C4_terminal_preservation_entered": 0,
            "C4_terminal_preservation_pass_rows": 0,
            "decision": "blocked_before_C4",
            "blocker": "C3_source_formation_gate_failed",
        }
    else:
        rows = [
            {
                "terminal_preservation_status": "not_executed_pending_h4800_runner",
                "C4_terminal_preservation_pass": 0,
                "blocker": "terminal_h4800_runner_not_claimed",
            }
        ]
        route = {
            "route": "TerminalPreservationPending_v22.09",
            "C4_terminal_preservation_entered": 1,
            "C4_terminal_preservation_pass_rows": 0,
            "decision": "pending",
            "blocker": "terminal_h4800_runner_not_claimed",
        }
    write_rows(out_dir / "v22_09_terminal_preservation_matrix.csv", rows)
    write_json(out_dir / "v22_09_terminal_preservation_route.json", route)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_09_terminal_preservation.py --source-dir {source_dir} --out-dir {out_dir}",
        status="completed",
        note=f"c3_pass={c3_pass} route={route['route']}",
    )


if __name__ == "__main__":
    main()
