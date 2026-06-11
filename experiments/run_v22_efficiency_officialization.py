#!/usr/bin/env python3
"""v22 D-CHE/D-FOU efficiency officialization wrapper."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.profiling.efficiency_v22 import v22_efficiency_gate  # noqa: E402
from experiments import run_v21_common as v21_common  # noqa: E402
from experiments import run_v21_efficiency_officialization as v21_eff  # noqa: E402
from experiments.run_v22_common import (  # noqa: E402
    PYTHON,
    V22_EXEC_DOC,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    read_rows,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data-root", default="data")
    p.add_argument("--families", default="D-CHE,D-FOU")
    p.add_argument("--batches", default="128,256")
    p.add_argument("--train-size", type=int, default=256)
    p.add_argument("--val-size", type=int, default=64)
    p.add_argument("--profiler-repeats", type=int, default=2)
    p.add_argument("--profiler-warmup", type=int, default=1)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--merge-only", action="store_true")
    return p


def call_v21_eff(args: argparse.Namespace) -> None:
    out_dir = ensure_out(args.out_dir)
    v21_common.V21_EXEC_DOC = V22_EXEC_DOC
    argv = [
        "run_v21_efficiency_officialization.py",
        "--out-dir",
        str(out_dir),
        "--device",
        str(args.device),
        "--data-root",
        str(args.data_root),
        "--families",
        str(args.families),
        "--batches",
        str(args.batches),
        "--train-size",
        str(args.train_size),
        "--val-size",
        str(args.val_size),
        "--profiler-repeats",
        str(args.profiler_repeats),
        "--profiler-warmup",
        str(args.profiler_warmup),
        "--shard-count",
        str(args.shard_count),
        "--shard-index",
        str(args.shard_index),
    ]
    if args.merge_only:
        argv.append("--merge-only")
    command = f"{PYTHON} experiments/run_v22_efficiency_officialization.py " + " ".join(argv[1:])
    append_exec(out_dir, command, status="started", note="delegates to v21 efficiency profiler and applies v22 gates")
    old = sys.argv[:]
    try:
        sys.argv = argv
        v21_eff.main()
    finally:
        sys.argv = old
    append_exec(out_dir, command, status="completed", note="raw v21 efficiency artifacts generated in v22 official dir")


def summarize_v22(out_dir: Path) -> None:
    rows = []
    for row in read_rows(out_dir / "v21_efficiency_truth_table.csv"):
        item = dict(row)
        item.update(v22_efficiency_gate(item))
        rows.append(item)
    write_rows(out_dir / "v22_efficiency_truth_table.csv", rows)
    waterfall = read_rows(out_dir / "v21_efficiency_waterfall.csv")
    grad = read_rows(out_dir / "v21_kernel_gradcheck.csv")
    write_rows(out_dir / "v22_efficiency_waterfall.csv", waterfall)
    write_rows(out_dir / "v22_kernel_gradcheck.csv", grad)
    by_carrier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_carrier[str(row.get("carrier", ""))].append(row)
    summary = []
    for carrier, cr in sorted(by_carrier.items()):
        fwd = [finite_float(r.get("forward_ratio_vs_mlp")) for r in cr if finite_float(r.get("forward_ratio_vs_mlp")) == finite_float(r.get("forward_ratio_vs_mlp"))]
        step = [finite_float(r.get("step_ratio_vs_mlp")) for r in cr if finite_float(r.get("step_ratio_vs_mlp")) == finite_float(r.get("step_ratio_vs_mlp"))]
        summary.append(
            {
                "carrier": carrier,
                "rows": len(cr),
                "E1_pass_rows": sum(int_flag(r.get("v22_E1_exploration_gate")) for r in cr),
                "S1_pass_rows": sum(int_flag(r.get("v22_S1_official_like_gate")) for r in cr),
                "E2_pass_rows": sum(int_flag(r.get("v22_E2_margin_gate")) for r in cr),
                "best_forward_ratio": min(fwd or [999.0]),
                "best_step_ratio": min(step or [999.0]),
            }
        )
    write_rows(out_dir / "v22_efficiency_officialization_summary.csv", summary)
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_efficiency_officialization.py --merge-only", status="completed", note=f"v22 efficiency rows={len(rows)}")


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    call_v21_eff(args)
    if args.merge_only:
        summarize_v22(out_dir)


if __name__ == "__main__":
    main()
