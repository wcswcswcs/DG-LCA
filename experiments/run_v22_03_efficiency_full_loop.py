#!/usr/bin/env python3
"""v22.03 full-loop efficiency readback from measured artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_03_common import PYTHON, V2202_OFFICIAL, append_exec, ensure_out, int_flag, read_rows, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default=str(V2202_OFFICIAL))
    return p


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir)
    truth = read_rows(source_dir / "v22_02_efficiency_truth_table.csv")
    full_loop = read_rows(source_dir / "v22_02_efficiency_full_loop_table.csv")
    summary = read_rows(source_dir / "v22_02_efficiency_officialization_summary.csv")
    out_summary: list[dict[str, Any]] = []
    for row in summary:
        carrier = str(row.get("carrier", ""))
        closure = int_flag(row.get("full_loop_official_closure"))
        out_summary.append(
            {
                **row,
                "v22_03_same_kernel_functional_runner_proof": closure,
                "v22_03_decision": "OfficialEfficientCarrier" if carrier in {"D-CHE", "D-FOU"} and closure else row.get("decision", "EfficiencyBlocked"),
            }
        )
    write_rows(out_dir / "v22_03_efficiency_truth_table.csv", truth)
    write_rows(out_dir / "v22_03_efficiency_full_loop_table.csv", full_loop)
    write_rows(out_dir / "v22_03_efficiency_full_loop_summary.csv", out_summary)
    write_json(
        out_dir / "v22_03_efficiency_full_loop_decision.json",
        {
            "D-CHE_full_loop_official_closure": max((int_flag(r.get("full_loop_official_closure")) for r in out_summary if r.get("carrier") == "D-CHE"), default=0),
            "D-FOU_full_loop_official_closure": max((int_flag(r.get("full_loop_official_closure")) for r in out_summary if r.get("carrier") == "D-FOU"), default=0),
            "source": str(source_dir),
            "decision": "ReadbackFromV2202MeasuredFullLoop",
        },
    )
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_03_efficiency_full_loop.py --source-dir {source_dir}", status="completed", note=f"rows={len(out_summary)}")


if __name__ == "__main__":
    main()

