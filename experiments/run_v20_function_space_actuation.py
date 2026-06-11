#!/usr/bin/env python3
"""v20 function-space actuation route readback."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v20_common import PYTHON, V19_OFFICIAL, append_exec, ensure_out, finite_float, read_rows, write_rows  # noqa: E402
from dgkan.fu.function_space_actuation import actuation_rows_from_traces, actuation_summary  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--include-v19", action="store_true", default=True)
    return p


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    append_exec(out_dir, f"{PYTHON} experiments/run_v20_function_space_actuation.py --out-dir {out_dir}", status="started")
    traces = []
    for path in sorted(out_dir.glob("v20_*_traces.csv")):
        traces.extend(read_rows(path))
    if args.include_v19:
        for name in [
            "v19_h4_actuation_solver_smoke.csv",
            "v19_functional_continuation_traces.csv",
            "v19_h12_m32_postadamw_smoke_traces.csv",
        ]:
            traces.extend(read_rows(V19_OFFICIAL / name))
    act_rows = actuation_rows_from_traces(traces)
    write_rows(out_dir / "v20_function_space_actuation_raw_diagnostics.csv", act_rows)
    source_rows = read_rows(out_dir / "v20_kan_source_writer_matrix.csv") + read_rows(out_dir / "v20_mlp_m16_anatomy.csv")
    summary = actuation_summary(act_rows, source_rows)
    for row in summary:
        r2 = finite_float(row.get("ActuationR2_max"), -1.0)
        source_h1600 = finite_float(row.get("source_h1600_mean"), -999.0)
        row["route_precedence"] = (
            "high_actuation_but_source_not_retained"
            if r2 >= 0.70 and source_h1600 < 0.005
            else ("actuation_and_source_candidate" if r2 >= 0.70 and source_h1600 >= 0.005 else "actuation_insufficient_or_not_measured")
        )
        row["promotion_allowed"] = 0
    write_rows(out_dir / "v20_function_space_actuation_matrix.csv", summary)
    append_exec(out_dir, f"{PYTHON} experiments/run_v20_function_space_actuation.py", status="completed", note=f"actuation_rows={len(act_rows)} summary={len(summary)}")


if __name__ == "__main__":
    main()
