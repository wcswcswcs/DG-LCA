#!/usr/bin/env python3
"""v22.06 KAN source-channel mapping gate summary."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_06_common import PYTHON, append_exec, ensure_out, int_flag, read_json, read_rows, simple_svg, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default="")
    return p


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    metric_route = read_json(source_dir / "v22_06_metric_solver_route.json")
    metric_rows = read_rows(source_dir / "v22_06_metric_solver_summary.csv")
    c4_open = any(int_flag(r.get("C4_h3200_source_pass")) for r in metric_rows)
    if not c4_open:
        rows = [
            {
                "mapping_status": "blocked_before_KAN_mapping",
                "blocker": "MLP_h3200_source_missing",
                "best_metric_v21_id": metric_route.get("best_metric_v21_id", ""),
                "KAN_source_channel_decision": "KANMappingNotEntered",
            }
        ]
    else:
        rows = [
            {
                "mapping_status": "pending_mapping_run",
                "blocker": "KAN_mapping_runner_not_launched_in_this_gate",
                "best_metric_v21_id": metric_route.get("best_metric_v21_id", ""),
                "KAN_source_channel_decision": "KANSourceChannelWriterMissing",
            }
        ]
    route = {
        "mapping_rows": len(rows),
        "decision": rows[0]["KAN_source_channel_decision"],
        "blocker": rows[0]["blocker"],
        "promotion_allowed": 0,
    }
    write_rows(out_dir / "v22_06_kan_source_mapping_summary.csv", rows)
    write_json(out_dir / "v22_06_kan_source_mapping_route.json", route)
    simple_svg(out_dir / "figures/KAN_source_channel_mapping_heatmap.svg", "v22.06 KAN source mapping", rows, "")
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_06_kan_source_mapping.py --source-dir {source_dir} --out-dir {out_dir}", status="completed", note=f"decision={route['decision']} blocker={route['blocker']}")


if __name__ == "__main__":
    main()
