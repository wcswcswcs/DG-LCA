#!/usr/bin/env python3
"""Damped fresh repair for the v22.09 FSMR fast/slow memory attempt.

This is not a new official gate. It reuses the FSMR selected candidates and
tests whether the failed fresh horizon was caused by over-aggressive slow-memory
injection rather than by the certificate principle itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_09_common import (  # noqa: E402
    PYTHON,
    V2206_COMBINED_SOURCE,
    append_exec,
    ensure_out,
    int_flag,
    read_json,
    read_rows,
    simple_svg,
    write_json,
    write_rows,
)
from experiments.run_v22_09_post_fast_slow_memory_resonance_certificate import (  # noqa: E402
    _fresh_summary,
    _train_fresh,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default=str(V2206_COMBINED_SOURCE))
    p.add_argument("--selected-file", default=None)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--data-root", default="data")
    p.add_argument("--train-size", type=int, default=96)
    p.add_argument("--val-size", type=int, default=48)
    p.add_argument("--input-size", type=int, default=8)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--param-budget", type=int, default=12000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--fresh-top-k", type=int, default=2)
    p.add_argument("--steps", type=int, default=3200)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--fu-lr", type=float, default=0.000025)
    p.add_argument("--memory-beta", type=float, default=0.97)
    p.add_argument("--refresh-interval", type=int, default=16)
    return p


def _load_selected(args: argparse.Namespace, out_dir: Path) -> list[dict[str, Any]]:
    selected_path = Path(args.selected_file) if args.selected_file else out_dir / "v22_09_fast_slow_memory_resonance_selected_candidates.csv"
    rows = read_rows(selected_path)
    return rows[: max(0, int(args.fresh_top_k))]


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    prev_route = read_json(out_dir / "v22_09_fast_slow_memory_resonance_route.json")
    selected = _load_selected(args, out_dir)
    fresh_rows = []
    for row in selected:
        for run_kind in [
            "FSMR-AdamWPlusFastSlowMemoryFU",
            "CTRL-AdamW",
            "CTRL-SGD",
            "CTRL-NoOp",
            "CTRL-RandomMatchedFastSlowMemoryFU",
        ]:
            fresh_rows.append(_train_fresh(args, row, run_kind))
    fresh_enriched, fresh_pack = _fresh_summary(fresh_rows)
    fresh_summary = fresh_pack["summary"]
    pass_rows = int(fresh_pack["route"]["fresh_C3_fast_slow_memory_pass_rows"])
    blockers = []
    blockers.append("official_observer_gate_failed")
    if not pass_rows:
        blockers.append("damped_repair_fresh_C3_source_horizon_failed")
    blockers.append("official_C3_gate_not_claimed")
    route = {
        "route": "PostFastSlowMemoryDampedRepairFreshC3Opened" if pass_rows else "PostFastSlowMemoryDampedRepairBlocked",
        "previous_route": prev_route.get("route", ""),
        "certificate_family": "FSMR_Damped_Fast_Slow_Source_Memory_Repair",
        "selected_candidates": len(selected),
        "fresh_rows": len(fresh_summary),
        "fu_lr": float(args.fu_lr),
        "memory_beta": float(args.memory_beta),
        "refresh_interval": int(args.refresh_interval),
        "fresh_C3_damped_repair_pass_rows": pass_rows,
        "official_C3_pass_rows": 0,
        "promotion_allowed": 0,
        "blocker": ";".join(dict.fromkeys(blockers)),
        "next_codex_action": "FSMR damped repair failed fresh/official gate; record local source-memory boundary or change principle again",
    }
    write_rows(out_dir / "v22_09_fast_slow_memory_damped_repair_fresh_c3_matrix.csv", fresh_enriched)
    write_rows(out_dir / "v22_09_fast_slow_memory_damped_repair_fresh_c3_summary.csv", fresh_summary)
    write_json(out_dir / "v22_09_fast_slow_memory_damped_repair_route.json", route)
    simple_svg(out_dir / "figures/v22_09_fast_slow_memory_damped_repair_h3200.svg", "v22.09 FSMR damped repair h3200", fresh_summary, "source_vs_best_control_h3200")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_09_post_fast_slow_memory_damped_repair.py --source-dir {args.source_dir} --device {args.device} --fresh-top-k {int(args.fresh_top_k)} --steps {int(args.steps)} --fu-lr {float(args.fu_lr)} --memory-beta {float(args.memory_beta)} --refresh-interval {int(args.refresh_interval)} --out-dir {out_dir}",
        status="completed",
        note=f"selected={len(selected)} fresh_pass={pass_rows} route={route['route']} fu_lr={float(args.fu_lr)} beta={float(args.memory_beta)} refresh={int(args.refresh_interval)}",
    )


if __name__ == "__main__":
    main()
