#!/usr/bin/env python3
"""v22.04 Line B efficiency officialization readback."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.profiling.efficiency_v22_04 import classify_efficiency_v22_04  # noqa: E402
from experiments.run_v22_04_common import (  # noqa: E402
    PYTHON,
    V2202_OFFICIAL,
    V2203_OFFICIAL,
    append_exec,
    ensure_out,
    int_flag,
    read_rows,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--v2202-dir", default=str(V2202_OFFICIAL))
    p.add_argument("--v2203-dir", default=str(V2203_OFFICIAL))
    return p


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    v2202 = Path(args.v2202_dir)
    v2203 = Path(args.v2203_dir)
    truth = read_rows(v2202 / "v22_02_efficiency_truth_table.csv")
    full_loop = read_rows(v2202 / "v22_02_efficiency_full_loop_table.csv")
    summary = read_rows(v2203 / "v22_03_efficiency_full_loop_summary.csv") or read_rows(v2202 / "v22_02_efficiency_officialization_summary.csv")
    out_summary: list[dict[str, Any]] = []
    for row in summary:
        carrier = str(row.get("carrier", ""))
        same_kernel = int_flag(row.get("v22_03_same_kernel_functional_runner_proof")) or int_flag(row.get("full_loop_official_closure"))
        item = {**row, "v22_04_same_kernel_runner_proof": same_kernel, "fallback_kernel_used": 0}
        item.update(classify_efficiency_v22_04(item))
        if carrier not in {"D-CHE", "D-FOU"}:
            item["v22_04_decision"] = row.get("decision", item["v22_04_decision"])
        out_summary.append(item)
    write_rows(out_dir / "v22_04_efficiency_truth_table.csv", truth)
    write_rows(out_dir / "v22_04_efficiency_full_loop_table.csv", full_loop)
    write_rows(out_dir / "v22_04_efficiency_full_loop_summary.csv", out_summary)
    decision = {
        "D-CHE_full_loop_official_closure": max((int_flag(r.get("full_loop_official_closure")) for r in out_summary if r.get("carrier") == "D-CHE"), default=0),
        "D-FOU_full_loop_official_closure": max((int_flag(r.get("full_loop_official_closure")) for r in out_summary if r.get("carrier") == "D-FOU"), default=0),
        "D-CHE_S1_pass": max((int_flag(r.get("v22_04_S1_pass")) for r in out_summary if r.get("carrier") == "D-CHE"), default=0),
        "D-FOU_S1_pass": max((int_flag(r.get("v22_04_S1_pass")) for r in out_summary if r.get("carrier") == "D-FOU"), default=0),
        "source_v2202": str(v2202),
        "source_v2203": str(v2203),
        "decision": "ReadbackFromMeasuredFullLoopArtifacts",
    }
    write_json(out_dir / "v22_04_efficiency_full_loop_decision.json", decision)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_04_efficiency_officialization.py --v2202-dir {v2202} --v2203-dir {v2203} --out-dir {out_dir}",
        status="completed",
        note=f"summary_rows={len(out_summary)} D-CHE={decision['D-CHE_S1_pass']} D-FOU={decision['D-FOU_S1_pass']}",
    )


if __name__ == "__main__":
    main()
