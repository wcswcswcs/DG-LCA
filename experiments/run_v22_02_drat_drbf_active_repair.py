#!/usr/bin/env python3
"""v22.02 D-RAT/D-RBF active-repair gate readback."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.profiling.efficiency_v22_02 import v22_02_efficiency_gate  # noqa: E402
from experiments.run_v22_02_common import (  # noqa: E402
    PYTHON,
    V2201_OFFICIAL,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    read_rows,
    sha256_file,
    simple_svg,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default=str(V2201_OFFICIAL))
    return p


def best(rows: list[dict[str, Any]], key: str) -> float:
    vals = [finite_float(r.get(key)) for r in rows]
    vals = [v for v in vals if v == v]
    return min(vals) if vals else float("nan")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_carrier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_carrier[str(row.get("carrier", ""))].append(row)
    out = []
    for carrier, cr in sorted(by_carrier.items()):
        near_key = "v22_02_NearE1_RBF_gate" if carrier == "D-RBF" else "v22_02_NearE1_gate"
        near_rows = sum(int_flag(r.get(near_key)) for r in cr)
        blockers = sorted({str(r.get("v22_02_efficiency_blocker", "")) for r in cr if str(r.get("v22_02_efficiency_blocker", ""))})
        if near_rows:
            decision = "NearEfficientCarrier"
        elif carrier == "D-RAT":
            decision = "ForwardKernelBlocked"
        else:
            decision = "MaterializationBlocked"
        out.append(
            {
                "carrier": carrier,
                "profile_rows": len(cr),
                "near_E1_rows": near_rows,
                "best_forward_ratio": best(cr, "forward_ratio_vs_mlp"),
                "best_step_ratio": best(cr, "step_ratio_vs_mlp"),
                "best_memory_ratio": best(cr, "memory_ratio_vs_mlp"),
                "gradcheck_pass_rows": sum(int_flag(r.get("gradcheck_pass")) for r in cr),
                "decision": decision,
                "blocker": ";".join(blockers),
            }
        )
    return out


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir)
    source_path = source_dir / "v22_01_drat_drbf_active_repair.csv"
    raw = [dict(r) for r in read_rows(source_path) if str(r.get("carrier")) in {"D-RAT", "D-RBF"}]
    rows = []
    for row in raw:
        item = dict(row)
        item["v22_02_source_artifact"] = str(source_path)
        item["v22_02_source_artifact_sha256"] = sha256_file(source_path) if source_path.exists() else ""
        item["safety_pass"] = int_flag(item.get("denominator_safety_pass", item.get("width_safety_pass", 1)))
        item["dense_reference_bytes"] = finite_float(item.get("basis_activation_bytes"), 0.0)
        item.update(v22_02_efficiency_gate(item))
        rows.append(item)
    write_rows(out_dir / "v22_02_drat_drbf_active_repair.csv", rows)
    write_rows(out_dir / "v22_02_drat_repair_table.csv", [r for r in rows if str(r.get("carrier")) == "D-RAT"])
    write_rows(out_dir / "v22_02_drbf_repair_table.csv", [r for r in rows if str(r.get("carrier")) == "D-RBF"])
    waterfall = []
    for row in rows:
        waterfall.append(
            {
                "carrier": row.get("carrier", ""),
                "component_variant": row.get("component_variant", ""),
                "batch_size": row.get("batch_size", ""),
                "num_eval_ms": row.get("numerator_eval_ms", ""),
                "den_eval_ms": row.get("denominator_eval_ms", ""),
                "reciprocal_ms": row.get("reciprocal_ms", ""),
                "rational_forward_ms": row.get("component_full_eval_ms", row.get("forward_only_ms", "")),
                "basis_materialized_bytes": row.get("basis_materialized_bytes", ""),
                "active_center_fraction": row.get("active_center_fraction", ""),
                "avg_local_k": row.get("local_k_effective", ""),
                "gaussian_exp_ms": row.get("rbf_exp_eval_ms", ""),
                "local_gather_ms": row.get("local_gather_ms", ""),
                "local_backward_ms": row.get("local_backward_ms", ""),
                "forward_ratio_vs_mlp": row.get("forward_ratio_vs_mlp", ""),
                "step_ratio_vs_mlp": row.get("step_ratio_vs_mlp", ""),
                "near_E1": row.get("v22_02_NearE1_gate", ""),
                "near_E1_RBF": row.get("v22_02_NearE1_RBF_gate", ""),
                "blocker": row.get("v22_02_efficiency_blocker", ""),
            }
        )
    write_rows(out_dir / "v22_02_drat_drbf_component_waterfall.csv", waterfall)
    summary = summarize(rows)
    write_rows(out_dir / "v22_02_drat_drbf_active_repair_summary.csv", summary)
    write_json(out_dir / "v22_02_drat_drbf_active_repair_decision.json", {r["carrier"]: r for r in summary})
    simple_svg(out_dir / "figures" / "fig_drat_forward_breakdown.svg", "D-RAT forward breakdown", [r for r in rows if r.get("carrier") == "D-RAT"], "forward_ratio_vs_mlp")
    simple_svg(out_dir / "figures" / "fig_drbf_local_support_breakdown.svg", "D-RBF local support breakdown", [r for r in rows if r.get("carrier") == "D-RBF"], "basis_materialized_bytes")
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_02_drat_drbf_active_repair.py --source-dir {source_dir}", status="completed", note=f"rows={len(rows)} source_artifact={source_path}")


if __name__ == "__main__":
    main()
