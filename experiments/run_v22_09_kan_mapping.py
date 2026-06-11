#!/usr/bin/env python3
"""v22.09 KAN source-channel mapping gate."""

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
    terminal = read_json(source_dir / "v22_09_terminal_preservation_route.json")
    c3_pass = int_flag(solver.get("C3_source_formation_pass_rows"))
    c4_pass = int_flag(terminal.get("C4_terminal_preservation_pass_rows"))
    if not c3_pass:
        rows: list[dict[str, Any]] = [
            {
                "mapping_status": "blocked_before_KAN_mapping",
                "blocker": "MLP_C3_source_formation_gate_failed",
                "best_metric_v21_id": "",
                "KAN_source_channel_decision": "KANMappingNotEntered",
                "KAN_specific_delta_vs_MLP_same_metric": "",
            }
        ]
        route = {"route": "KANMappingNotEntered", "decision": "KANMappingNotEntered", "blocker": "MLP_C3_source_formation_gate_failed"}
    elif not c4_pass:
        rows = [
            {
                "mapping_status": "blocked_before_KAN_mapping",
                "blocker": "MLP_C4_terminal_preservation_gate_failed",
                "KAN_source_channel_decision": "KANMappingNotEntered",
            }
        ]
        route = {"route": "KANMappingNotEntered", "decision": "KANMappingNotEntered", "blocker": "MLP_C4_terminal_preservation_gate_failed"}
    else:
        rows = [
            {
                "mapping_status": "not_executed_pending_KAN_runner",
                "blocker": "KAN_mapping_runner_not_claimed",
                "KAN_source_channel_decision": "pending",
            }
        ]
        route = {"route": "KANSourceChannelMappingPending", "decision": "pending", "blocker": "KAN_mapping_runner_not_claimed"}
    write_rows(out_dir / "v22_09_kan_mapping_matrix.csv", rows)
    write_json(out_dir / "v22_09_kan_mapping_route.json", route)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_09_kan_mapping.py --source-dir {source_dir} --out-dir {out_dir}",
        status="completed",
        note=f"c3_pass={c3_pass} c4_pass={c4_pass} route={route['route']}",
    )


if __name__ == "__main__":
    main()
