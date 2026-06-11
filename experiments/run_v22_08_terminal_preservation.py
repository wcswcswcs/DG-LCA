#!/usr/bin/env python3
"""v22.08 terminal preservation gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_08_common import PYTHON, append_exec, ensure_out, int_flag, read_json, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default="")
    return p


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    metric_route = read_json(source_dir / "v22_08_metric_dynamics_solver_route.json")
    c3_pass = int_flag(metric_route.get("C3_source_formation_pass_rows"))
    if c3_pass:
        decision = "TerminalPreservationRequiresFreshH4800Run"
        blocker = "terminal_h4800_run_not_implemented_in_this_gate"
    else:
        decision = "D0D1BlockedBeforeTerminalPreservation"
        blocker = metric_route.get("blocker", "C3_source_formation_gate")
    route = {
        "decision": decision,
        "C3_source_formation_pass_rows": c3_pass,
        "C4_terminal_preservation_entered": int(bool(c3_pass)),
        "C5_productive_terminal_source": 0,
        "promotion_allowed": 0,
        "blocker": blocker,
    }
    summary = [
        {
            "source_h3200": "",
            "source_h4800": "",
            "R4800_over_3200": "",
            "row_h4800_positive_count": "",
            "C5_productive_terminal_source": 0,
            "preservation_status": "blocked_before_D0_D1" if not c3_pass else "not_run_fresh_h4800",
            "blocker": blocker,
        }
    ]
    write_rows(out_dir / "v22_08_terminal_preservation_summary.csv", summary)
    write_json(out_dir / "v22_08_terminal_preservation_route.json", route)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_08_terminal_preservation.py --source-dir {source_dir} --out-dir {out_dir}",
        status="completed",
        note=f"decision={decision} blocker={blocker}",
    )


if __name__ == "__main__":
    main()

