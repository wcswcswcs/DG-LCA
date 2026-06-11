#!/usr/bin/env python3
"""v22.08 KAN source-channel mapping gate."""

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
        decision = "KANMappingRequiresFreshMLPSourceDecomposition"
        blocker = "MLP_source_decomposition_not_run"
    else:
        decision = "KANMappingNotEntered"
        blocker = metric_route.get("blocker", "C3_source_formation_gate")
    route = {
        "mapping_status": "blocked_before_KAN_mapping" if not c3_pass else "not_run",
        "decision": decision,
        "MLP_C3_source_formation_pass_rows": c3_pass,
        "KAN_source_channel_entered": int(bool(c3_pass)),
        "promotion_allowed": 0,
        "blocker": blocker,
    }
    summary = [
        {
            "mapping_status": route["mapping_status"],
            "blocker": blocker,
            "best_metric_v21_id": "",
            "KAN_source_channel_decision": decision,
            "KAN_specific_delta_vs_MLP_same_metric": "",
        }
    ]
    write_rows(out_dir / "v22_08_kan_source_mapping_summary.csv", summary)
    write_json(out_dir / "v22_08_kan_source_mapping_route.json", route)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_08_kan_source_mapping.py --source-dir {source_dir} --out-dir {out_dir}",
        status="completed",
        note=f"decision={decision} blocker={blocker}",
    )


if __name__ == "__main__":
    main()

