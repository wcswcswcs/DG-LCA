#!/usr/bin/env python3
"""v22.07 KAN source-channel mapping gate summary."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_07_common import PYTHON, append_exec, ensure_out, int_flag, read_json, read_rows, simple_svg, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default="")
    return p


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    metric_route = read_json(source_dir / "v22_07_metric_dynamics_route.json")
    terminal_route = read_json(source_dir / "v22_07_terminal_preservation_route.json")
    c3_rows = read_rows(source_dir / "v22_07_c3_source_formation_matrix.csv")
    mlp_open = any(int_flag(r.get("C3_source_formation_pass")) for r in c3_rows)
    terminal_open = str(terminal_route.get("decision", "")) == "D1ProductiveTerminalSource"
    if not (mlp_open or terminal_open):
        rows = [
            {
                "mapping_status": "blocked_before_KAN_mapping",
                "blocker": metric_route.get("blocker", "MLP_C3_or_D1_source_missing"),
                "best_metric_v21_id": "",
                "KAN_source_channel_decision": "KANMappingNotEntered",
                "KAN_specific_delta_vs_MLP_same_metric": "",
            }
        ]
    else:
        best = sorted(c3_rows, key=lambda r: (int_flag(r.get("C3_source_formation_pass")), float(r.get("source_vs_best_control_h3200") or -999)), reverse=True)[0]
        rows = [
            {
                "mapping_status": "pending_mapping_run",
                "blocker": "KAN_mapping_runner_not_implemented_for_v22_07_after_MLP_gate",
                "best_metric_v21_id": best.get("v21_id", ""),
                "KAN_source_channel_decision": "KANSourceChannelWriterMissing",
                "KAN_specific_delta_vs_MLP_same_metric": "",
            }
        ]
    route = {
        "mapping_rows": len(rows),
        "decision": rows[0]["KAN_source_channel_decision"],
        "blocker": rows[0]["blocker"],
        "promotion_allowed": 0,
    }
    write_rows(out_dir / "v22_07_kan_source_mapping_summary.csv", rows)
    write_json(out_dir / "v22_07_kan_source_mapping_route.json", route)
    simple_svg(out_dir / "figures/KAN_source_channel_mapping_heatmap.svg", "v22.07 KAN source-channel mapping", rows, "")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_07_kan_source_mapping.py --source-dir {source_dir} --out-dir {out_dir}",
        status="completed",
        note=f"decision={route['decision']} blocker={route['blocker']}",
    )


if __name__ == "__main__":
    main()

