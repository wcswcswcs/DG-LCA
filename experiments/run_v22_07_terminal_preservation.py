#!/usr/bin/env python3
"""v22.07 terminal-preservation gate summary."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_07_common import PYTHON, append_exec, ensure_out, finite_float, int_flag, read_json, read_rows, simple_svg, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default="")
    return p


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    route = read_json(source_dir / "v22_07_metric_dynamics_route.json")
    rows = read_rows(source_dir / "v22_07_c3_source_formation_matrix.csv")
    eligible = [r for r in rows if int_flag(r.get("C3_source_formation_pass"))]
    out_rows = []
    for r in eligible:
        s3200 = finite_float(r.get("source_vs_best_control_h3200"))
        s4800 = finite_float(r.get("source_vs_best_control_h4800"))
        ratio = max(0.0, s4800) / max(1.0e-12, s3200) if s3200 == s3200 and s3200 > 0 and s4800 == s4800 else ""
        out_rows.append(
            {
                "v21_id": r.get("v21_id", ""),
                "source_h3200": r.get("source_vs_best_control_h3200", ""),
                "source_h4800": r.get("source_vs_best_control_h4800", ""),
                "R4800_over_3200": ratio,
                "row_h4800_positive_count": r.get("row_positive_count_h4800", ""),
                "C5_productive_terminal_source": int(s4800 == s4800 and s4800 >= 0.005 and finite_float(ratio, 0.0) >= 0.50),
                "optimizer_cumulative_projection_on_source": "",
                "row_norm_drift": "",
                "row_angular_velocity": "",
                "radial_update_fraction": "",
                "preservation_status": "terminal_data_readback" if s4800 == s4800 else "blocked_missing_h4800_artifact",
                "blocker": "" if s4800 == s4800 else "h4800_terminal_artifact_missing",
            }
        )
    if not eligible:
        out_rows.append(
            {
                "v21_id": "",
                "source_h3200": "",
                "source_h4800": "",
                "R4800_over_3200": "",
                "C5_productive_terminal_source": 0,
                "preservation_status": "blocked_before_D0_D1",
                "blocker": route.get("blocker", "C3_source_formation_missing"),
            }
        )
    decision = "D1ProductiveTerminalSource" if any(int_flag(r.get("C5_productive_terminal_source")) for r in out_rows) else "D0D1BlockedBeforeTerminalPreservation"
    terminal_route = {
        "entered_D0_rows": len(eligible),
        "productive_terminal_rows": sum(int_flag(r.get("C5_productive_terminal_source")) for r in out_rows),
        "decision": decision,
        "blocker": "" if eligible else route.get("blocker", "C3_source_formation_missing"),
    }
    write_rows(out_dir / "v22_07_terminal_preservation_summary.csv", out_rows)
    write_json(out_dir / "v22_07_terminal_preservation_route.json", terminal_route)
    simple_svg(out_dir / "figures/terminal_erosion_autopsy.svg", "v22.07 terminal erosion autopsy", out_rows, "source_h3200")
    simple_svg(out_dir / "figures/optimizer_projection_on_source.svg", "v22.07 optimizer projection on source", out_rows, "optimizer_cumulative_projection_on_source")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_07_terminal_preservation.py --source-dir {source_dir} --out-dir {out_dir}",
        status="completed",
        note=f"entered_D0={len(eligible)} decision={terminal_route['decision']} blocker={terminal_route['blocker']}",
    )


if __name__ == "__main__":
    main()

