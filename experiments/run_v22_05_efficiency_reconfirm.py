#!/usr/bin/env python3
"""v22.05 D-CHE/D-FOU efficiency reconfirm readback."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.profiling.efficiency_v22_05 import classify_efficiency_v22_05  # noqa: E402
from experiments.run_v22_05_common import PYTHON, V2204_OFFICIAL, append_exec, ensure_out, int_flag, read_rows, simple_svg, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--v2204-dir", default=str(V2204_OFFICIAL))
    return p


def _normalize(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["same_kernel_functional_runner_proof"] = int_flag(row.get("v22_04_same_kernel_runner_proof")) or int_flag(row.get("v22_03_same_kernel_functional_runner_proof"))
    out["official_fused_kernel_complete"] = int_flag(row.get("full_loop_official_closure"))
    out["fallback_kernel_used"] = int_flag(row.get("fallback_kernel_used"))
    out["functional_overhead_ratio"] = row.get("functional_overhead_ratio", 1.0)
    out["forward_ratio_vs_mlp"] = row.get("best_forward_ratio", row.get("forward_ratio_vs_mlp", ""))
    out["step_ratio_vs_mlp"] = row.get("best_step_ratio", row.get("step_ratio_vs_mlp", ""))
    out["memory_ratio_vs_mlp"] = row.get("best_memory_ratio", row.get("memory_ratio_vs_mlp", ""))
    out["v22_05_reconfirm_source"] = "v22_04_measured_full_loop_readback"
    out.update(classify_efficiency_v22_05(out))
    return out


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source = Path(args.v2204_dir)
    summary = read_rows(source / "v22_04_efficiency_full_loop_summary.csv")
    rows = [_normalize(r) for r in summary if str(r.get("carrier", "")) in {"D-CHE", "D-FOU"}]
    write_rows(out_dir / "v22_05_efficiency_full_loop_reconfirm.csv", rows)
    decision = {
        "D-CHE_S1_rows": sum(1 for r in rows if r.get("carrier") == "D-CHE" and int_flag(r.get("v22_05_S1_pass"))),
        "D-FOU_S1_rows": sum(1 for r in rows if r.get("carrier") == "D-FOU" and int_flag(r.get("v22_05_S1_pass"))),
        "D-CHE_S1_pass": int(any(r.get("carrier") == "D-CHE" and int_flag(r.get("v22_05_S1_pass")) for r in rows)),
        "D-FOU_S1_pass": int(any(r.get("carrier") == "D-FOU" and int_flag(r.get("v22_05_S1_pass")) for r in rows)),
        "source_v2204": str(source),
        "decision": "ReadbackFromMeasuredFullLoopArtifacts",
    }
    write_json(out_dir / "v22_05_efficiency_route.json", decision)
    simple_svg(out_dir / "figures/efficiency_forward_backward_step_memory_dashboard.svg", "v22.05 efficiency reconfirm", rows, "step_ratio_vs_mlp")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_05_efficiency_reconfirm.py --v2204-dir {source} --out-dir {out_dir}",
        status="completed",
        note=f"rows={len(rows)} D-CHE={decision['D-CHE_S1_pass']} D-FOU={decision['D-FOU_S1_pass']}",
    )


if __name__ == "__main__":
    main()
