#!/usr/bin/env python3
"""v21 function-space target reset readback and decision artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v21_common import PYTHON, append_exec, ensure_out, finite_float, int_flag, read_rows, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    return p


def aggregate_trace(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, float]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (str(row.get("carrier")), str(row.get("basis_repair_variant")), str(row.get("v21_id")))
        grouped.setdefault(key, []).append(row)
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    for key, group in grouped.items():
        vals = [finite_float(r.get("ActuationR2")) for r in group]
        vals = [v for v in vals if v == v]
        b1 = [finite_float(r.get("B1_gain")) for r in group]
        b2 = [finite_float(r.get("B2_transfer_gain")) for r in group]
        b3 = [finite_float(r.get("B3_safety_gain")) for r in group]
        out[key] = {
            "trace_rows": float(len(group)),
            "ActuationR2_max": max(vals) if vals else float("nan"),
            "ActuationR2_mean": sum(vals) / len(vals) if vals else float("nan"),
            "B1_gain_mean": sum(v for v in b1 if v == v) / max(1, sum(1 for v in b1 if v == v)),
            "B2_transfer_gain_mean": sum(v for v in b2 if v == v) / max(1, sum(1 for v in b2 if v == v)),
            "B3_safety_gain_mean": sum(v for v in b3 if v == v) / max(1, sum(1 for v in b3 if v == v)),
        }
    return out


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    append_exec(out_dir, f"{PYTHON} experiments/run_v21_function_space_target_reset.py --out-dir {out_dir}", status="started")
    summary = read_rows(out_dir / "v21_kan_source_writer_matrix.csv")
    traces = read_rows(out_dir / "v21_kan_source_writer_traces.csv")
    trace_stats = aggregate_trace(traces)
    rows = []
    for row in summary:
        key = (str(row.get("carrier")), str(row.get("basis_repair_variant")), str(row.get("v21_id")))
        stats = trace_stats.get(key, {})
        act_max = stats.get("ActuationR2_max", float("nan"))
        source_h1600 = finite_float(row.get("source_h1600_mean"))
        source_h3200 = finite_float(row.get("source_h3200_mean"))
        source_success = int(int_flag(row.get("retained_h3200_candidate")) or (source_h1600 == source_h1600 and source_h1600 >= 0.005 and source_h3200 == source_h3200 and source_h3200 >= 0.005))
        if "M23" not in str(row.get("continuation_id")) and "function" not in str(row.get("hypothesis")).lower() and "source-bank" not in str(row.get("v21_id")).lower():
            continue
        rows.append(
            {
                "carrier": row.get("carrier", ""),
                "basis_repair_variant": row.get("basis_repair_variant", ""),
                "v21_id": row.get("v21_id", ""),
                "continuation_id": row.get("continuation_id", ""),
                "mechanism": row.get("mechanism", ""),
                "rows": row.get("rows", ""),
                "ActuationR2_max": act_max if act_max == act_max else "",
                "ActuationR2_mean": stats.get("ActuationR2_mean", ""),
                "B1_gain_mean": stats.get("B1_gain_mean", ""),
                "B2_transfer_gain_mean": stats.get("B2_transfer_gain_mean", ""),
                "B3_safety_gain_mean": stats.get("B3_safety_gain_mean", ""),
                "source_h800_mean": row.get("source_h800_mean", ""),
                "source_h1600_mean": row.get("source_h1600_mean", ""),
                "source_h3200_mean": row.get("source_h3200_mean", ""),
                "source_h4800_mean": row.get("source_h4800_mean", ""),
                "retained_h3200_candidate": row.get("retained_h3200_candidate", ""),
                "source_success": source_success,
                "route_precedence": "high_actuation_source_success" if act_max == act_max and act_max >= 0.70 and source_success else "high_actuation_but_source_not_retained" if act_max == act_max and act_max >= 0.70 else "actuation_insufficient_or_not_measured",
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v21_function_space_target_reset_matrix.csv", rows)
    write_rows(
        out_dir / "v21_function_space_target_reset_diagnostics.csv",
        [
            {
                "rows": len(rows),
                "high_actuation_rows": sum(1 for r in rows if finite_float(r.get("ActuationR2_max"), -1.0) >= 0.70),
                "source_success_rows": sum(int_flag(r.get("source_success")) for r in rows),
                "promotion_allowed": 0,
            }
        ],
    )
    append_exec(out_dir, f"{PYTHON} experiments/run_v21_function_space_target_reset.py --out-dir {out_dir}", status="completed", note=f"rows={len(rows)}")


if __name__ == "__main__":
    main()
