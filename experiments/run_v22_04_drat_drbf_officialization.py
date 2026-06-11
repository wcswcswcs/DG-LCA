#!/usr/bin/env python3
"""v22.04 D-RAT/D-RBF active officialization wrapper.

The active benchmark is delegated to the v22.03 micro-kernel benchmark and
then relabeled under v22.04. No micro-kernel result is promoted as an
official fused production runner.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v22_03_drat_drbf_repair as repair  # noqa: E402
from experiments.run_v22_04_common import PYTHON, append_exec, ensure_out, read_rows, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-sizes", default="512,2048")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--warmup", type=int, default=8)
    return p


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_carrier: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_carrier.setdefault(str(row.get("carrier", "")), []).append(row)
    out = []
    for carrier, group in sorted(by_carrier.items()):
        best_forward = min(float(r["forward_ratio_vs_mlp"]) for r in group)
        best_step = min(float(r["step_ratio_vs_mlp"]) for r in group)
        best_memory = min(float(r["memory_ratio_vs_mlp"]) for r in group)
        micro_near = sum(int(r.get("micro_near_E1", 0)) for r in group)
        grad_rows = sum(int(r.get("gradcheck_pass", 0)) for r in group)
        official_rows = sum(int(r.get("official_fused_kernel_complete", 0)) for r in group)
        decision = "MicroNearE1OfficialFusedBlocked" if micro_near else ("ForwardKernelBlocked" if carrier == "D-RAT" else "MaterializationBlocked")
        blocker = []
        if not micro_near:
            blocker.append("micro_near_E1_missing")
        if not official_rows:
            blocker.append("official_fused_missing")
        out.append(
            {
                "carrier": carrier,
                "profile_rows": len(group),
                "near_E1_rows": 0,
                "micro_near_E1_rows": micro_near,
                "best_forward_ratio": best_forward,
                "best_step_ratio": best_step,
                "best_memory_ratio": best_memory,
                "gradcheck_pass_rows": grad_rows,
                "official_fused_rows": official_rows,
                "decision": decision,
                "blocker": ";".join(blocker),
            }
        )
    return out


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    device = repair._device(args.device)
    batch_sizes = [int(x) for x in str(args.batch_sizes).split(",") if x.strip()]
    rows = repair._rat_rows(device, batch_sizes, args.hidden, args.warmup, args.iters)
    rows.extend(repair._rbf_rows(device, batch_sizes, args.hidden, args.warmup, args.iters))
    for row in rows:
        row["v22_04_active_repair"] = 1
        row["official_promotion_allowed"] = 0
    write_rows(out_dir / "v22_04_drat_drbf_active_repair.csv", rows)
    write_rows(out_dir / "v22_04_drat_repair_table.csv", [r for r in rows if r.get("carrier") == "D-RAT"])
    write_rows(out_dir / "v22_04_drbf_repair_table.csv", [r for r in rows if r.get("carrier") == "D-RBF"])
    component = [
        {
            "carrier": r.get("carrier", ""),
            "component_variant": r.get("component_variant", ""),
            "batch_size": r.get("batch_size", ""),
            "forward_ratio_vs_mlp": r.get("forward_ratio_vs_mlp", ""),
            "step_ratio_vs_mlp": r.get("step_ratio_vs_mlp", ""),
            "memory_ratio_vs_mlp": r.get("memory_ratio_vs_mlp", ""),
            "micro_near_E1": r.get("micro_near_E1", ""),
            "official_fused_kernel_complete": r.get("official_fused_kernel_complete", ""),
            "blocker": "official_fused_missing",
        }
        for r in rows
    ]
    write_rows(out_dir / "v22_04_drat_drbf_component_waterfall.csv", component)
    summary = _summary(rows)
    write_rows(out_dir / "v22_04_drat_drbf_active_repair_summary.csv", summary)
    write_rows(out_dir / "v22_04_drat_drbf_repair_decision.csv", summary)
    write_json(out_dir / "v22_04_drat_drbf_repair_decision.json", {r["carrier"]: r for r in summary})
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_04_drat_drbf_officialization.py --out-dir {out_dir} --device {args.device} --batch-sizes {args.batch_sizes} --hidden {args.hidden} --iters {args.iters} --warmup {args.warmup}",
        status="completed",
        note=f"active micro-kernel rows={len(rows)}; official fused promotion remains blocked",
    )


if __name__ == "__main__":
    main()
