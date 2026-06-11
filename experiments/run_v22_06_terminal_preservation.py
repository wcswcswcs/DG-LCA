#!/usr/bin/env python3
"""v22.06 terminal-preservation gate summary.

This script does not invent terminal data.  It enters C5 only if the metric
solver summary contains h3200 source rows; otherwise it records the blocker.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_06_common import PYTHON, append_exec, ensure_out, finite_float, int_flag, read_json, read_rows, simple_svg, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default="")
    return p


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    route = read_json(source_dir / "v22_06_metric_solver_route.json")
    rows = read_rows(source_dir / "v22_06_metric_solver_summary.csv")
    eligible = [r for r in rows if int_flag(r.get("C4_h3200_source_pass"))]
    out_rows = []
    for r in eligible:
        out_rows.append(
            {
                "v21_id": r.get("v21_id", ""),
                "source_h3200": r.get("source_h3200_mean", ""),
                "source_h4800": r.get("source_h4800_mean", ""),
                "R4800_over_3200": r.get("retention_h4800_over_h3200", r.get("R4800_over_3200", "")),
                "C5_productive_terminal_source": int(
                    finite_float(r.get("source_h4800_mean"), -999.0) >= 0.005
                    and finite_float(r.get("retention_h4800_over_h3200", r.get("R4800_over_3200")), 0.0) >= 0.50
                ),
                "erase_energy_h3200_to_h4800": "",
                "removed_destructive_component_norm": "",
                "preservation_status": "terminal_data_readback",
            }
        )
    if not eligible:
        out_rows.append(
            {
                "v21_id": route.get("best_metric_v21_id", ""),
                "source_h3200": route.get("best_source_h3200", ""),
                "C5_productive_terminal_source": 0,
                "preservation_status": "blocked_before_C5",
                "blocker": "h3200_source_missing",
            }
        )
    c5_route = {
        "entered_C5_rows": len(eligible),
        "productive_terminal_rows": sum(int_flag(r.get("C5_productive_terminal_source")) for r in out_rows),
        "decision": "C5ProductiveTerminalSource" if any(int_flag(r.get("C5_productive_terminal_source")) for r in out_rows) else "C5BlockedBeforeTerminalPreservation",
        "blocker": "" if eligible else "h3200_source_missing",
    }
    write_rows(out_dir / "v22_06_terminal_preservation_summary.csv", out_rows)
    write_json(out_dir / "v22_06_terminal_preservation_route.json", c5_route)
    simple_svg(out_dir / "figures/terminal_preservation_projection_trace.svg", "v22.06 terminal preservation", out_rows, "source_h3200")
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_06_terminal_preservation.py --source-dir {source_dir} --out-dir {out_dir}", status="completed", note=f"entered_C5={len(eligible)} decision={c5_route['decision']}")


if __name__ == "__main__":
    main()
