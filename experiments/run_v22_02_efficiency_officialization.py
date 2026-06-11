#!/usr/bin/env python3
"""v22.02 D-CHE/D-FOU full-loop efficiency officialization readback."""

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
    finite_float,
    ensure_out,
    int_flag,
    mean,
    read_rows,
    sha256_file,
    simple_svg,
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


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir)
    source_path = source_dir / "v22_01_efficiency_truth_table.csv"
    raw = [dict(r) for r in read_rows(source_path) if str(r.get("carrier")) in {"D-CHE", "D-FOU"}]
    rows = []
    for row in raw:
        item = dict(row)
        item["v22_02_source_artifact"] = str(source_path)
        item["v22_02_source_artifact_sha256"] = sha256_file(source_path) if source_path.exists() else ""
        item["functional_runner_uses_same_kernel"] = int_flag(item.get("functional_runner_uses_same_kernel", 0))
        item["fallback_kernel_used"] = int_flag(item.get("fallback_kernel_used", 0))
        item.update(v22_02_efficiency_gate(item))
        rows.append(item)
    write_rows(out_dir / "v22_02_efficiency_truth_table.csv", rows)
    write_rows(
        out_dir / "v22_02_efficiency_full_loop_table.csv",
        [
            {
                "carrier": r.get("carrier", ""),
                "variant": r.get("v22_01_variant", r.get("kernel_variant", "")),
                "batch_size": r.get("batch_size", ""),
                "forward_only_ms": r.get("forward_only_ms", ""),
                "backward_grad_ms": r.get("backward_grad_ms", ""),
                "optimizer_update_ms": r.get("optimizer_update_ms", ""),
                "functional_direction_ms": r.get("functional_direction_ms", ""),
                "functional_commit_ms": r.get("functional_commit_ms", ""),
                "linec_audit_ms": r.get("linec_audit_ms", ""),
                "horizon_readback_ms": r.get("horizon_readback_ms", ""),
                "training_step_ms": r.get("step_training_only_ms", ""),
                "full_loop_step_ms": r.get("step_with_audit_ms", ""),
                "forward_ratio_vs_mlp": r.get("forward_ratio_vs_mlp", ""),
                "step_ratio_vs_mlp": r.get("step_ratio_vs_mlp", ""),
                "memory_ratio_vs_mlp": r.get("memory_ratio_vs_mlp", ""),
                "functional_runner_uses_same_kernel": r.get("functional_runner_uses_same_kernel", ""),
                "official_fused_kernel_complete": r.get("official_fused_kernel_complete", ""),
                "no_materialize_complete": r.get("no_materialize_complete", ""),
                "fallback_kernel_used": r.get("fallback_kernel_used", ""),
                "v22_02_S1_official_like_gate": r.get("v22_02_S1_official_like_gate", ""),
                "blocker": r.get("v22_02_efficiency_blocker", ""),
            }
            for r in rows
        ],
    )

    by_carrier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_carrier[str(row.get("carrier", ""))].append(row)
    summary = []
    for carrier, cr in sorted(by_carrier.items()):
        batch_s1 = {str(r.get("batch_size")) for r in cr if int_flag(r.get("v22_02_S1_official_like_gate"))}
        variant_s1 = {str(r.get("v22_01_variant", r.get("kernel_variant", ""))) for r in cr if int_flag(r.get("v22_02_S1_official_like_gate"))}
        required_batches = 3
        required_variants = 2 if carrier == "D-CHE" else 1
        closure = int(len(batch_s1) >= required_batches and len(variant_s1) >= required_variants)
        summary.append(
            {
                "carrier": carrier,
                "rows": len(cr),
                "E1_pass_rows": sum(int_flag(r.get("v22_02_E1_exploration_gate")) for r in cr),
                "S1_pass_rows": sum(int_flag(r.get("v22_02_S1_official_like_gate")) for r in cr),
                "S1_pass_batch_sizes": len(batch_s1),
                "S1_pass_variants": len(variant_s1),
                "best_forward_ratio": best(cr, "forward_ratio_vs_mlp"),
                "best_step_ratio": best(cr, "step_ratio_vs_mlp"),
                "best_memory_ratio": best(cr, "memory_ratio_vs_mlp"),
                "mean_linec_audit_ms": mean(cr, "linec_audit_ms"),
                "mean_horizon_readback_ms": mean(cr, "horizon_readback_ms"),
                "full_loop_official_closure": closure,
                "decision": "OfficialEfficientCarrier" if closure else "NearOfficialOrBlocked",
            }
        )
    write_rows(out_dir / "v22_02_efficiency_officialization_summary.csv", summary)
    status = []
    for row in rows:
        official = int_flag(row.get("official_fused_kernel_complete"))
        status.append(
            {
                "carrier": row.get("carrier", ""),
                "variant": row.get("v22_01_variant", row.get("kernel_variant", "")),
                "batch_size": row.get("batch_size", ""),
                "implementation_path": row.get("manual_kernel_variant", ""),
                "official_fused_kernel_complete": official,
                "gradcheck_pass": row.get("gradcheck_pass", ""),
                "no_materialize_complete": row.get("no_materialize_complete", ""),
                "functional_runner_uses_same_kernel": row.get("functional_runner_uses_same_kernel", ""),
                "status_consistent": int((not official) or (int_flag(row.get("gradcheck_pass")) and int_flag(row.get("no_materialize_complete")))),
            }
        )
    write_rows(out_dir / "v22_02_official_fused_status_matrix.csv", status)
    simple_svg(out_dir / "figures" / "fig_efficiency_full_loop_waterfall.svg", "v22.02 efficiency full-loop waterfall", rows, "step_ratio_vs_mlp")
    simple_svg(out_dir / "figures" / "fig_dche_dfou_s1_stability.svg", "D-CHE/D-FOU S1 stability", rows, "v22_02_S1_official_like_gate")
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_02_efficiency_officialization.py --source-dir {source_dir}", status="completed", note=f"rows={len(rows)} summary={len(summary)} source_artifact={source_path}")


if __name__ == "__main__":
    main()
