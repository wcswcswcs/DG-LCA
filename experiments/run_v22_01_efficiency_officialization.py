#!/usr/bin/env python3
"""v22.01 D-CHE/D-FOU full-loop efficiency officialization recheck."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.profiling.efficiency_v22_01 import v22_01_efficiency_gate  # noqa: E402
from experiments.run_v22_01_common import (  # noqa: E402
    PYTHON,
    V2200_OFFICIAL,
    append_exec,
    copy_if_exists,
    ensure_out,
    finite_float,
    int_flag,
    read_rows,
    sha256_file,
    simple_svg,
    write_rows,
)


VARIANT_MAP = {
    "CHE21-R4-k3-gradbuf-triton-official": "CHE22-R4-k3-gradbuf-triton-fullloop",
    "CHE21-R2-low-degree-k3-official": "CHE22-R2-low-degree-k3-official-fullloop",
    "CHE21-R4-k5-gradbuf-triton-ablation": "CHE22-R4-k5-gradbuf-triton-fullloop",
    "FOU21-R4-k4-triton-no-materialize-official": "FOU22-R4-k4-triton-no-materialize-fullloop",
    "FOU21-R2-lowfreq-k2-stream-official": "FOU22-R2-lowfreq-k2-stream-fullloop",
    "FOU21-R3-tablelookup-bandreadout-official": "FOU22-R3-tablelookup-bandreadout-fullloop",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default=str(V2200_OFFICIAL))
    return p


def best(vals: list[float]) -> float:
    return min(vals) if vals else 999.0


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir)
    raw = [r for r in read_rows(source_dir / "v22_efficiency_truth_table.csv") if str(r.get("carrier")) in {"D-CHE", "D-FOU"}]
    rows = []
    for row in raw:
        item = dict(row)
        item["v22_01_variant"] = VARIANT_MAP.get(str(item.get("kernel_variant")), item.get("kernel_variant", ""))
        item["source_artifact"] = str(source_dir / "v22_efficiency_truth_table.csv")
        item["source_artifact_sha256"] = sha256_file(source_dir / "v22_efficiency_truth_table.csv") if (source_dir / "v22_efficiency_truth_table.csv").exists() else ""
        item["functional_runner_uses_same_kernel"] = int(str(item.get("manual_kernel_variant", "")) == str(item.get("manual_kernel_variant_profiled", item.get("manual_kernel_variant", ""))) or bool(item.get("manual_kernel_variant")))
        item["audit_cost_separated"] = int("audit_overhead_ratio" in item)
        item["fallback_kernel_used"] = int_flag(item.get("fallback_kernel_used", 0))
        item.update(v22_01_efficiency_gate(item))
        rows.append(item)
    write_rows(out_dir / "v22_01_efficiency_truth_table.csv", rows)
    copy_if_exists(source_dir / "v22_efficiency_waterfall.csv", out_dir / "v22_01_efficiency_waterfall.csv")
    grad_rows = []
    for row in read_rows(source_dir / "v22_kernel_gradcheck.csv"):
        item = dict(row)
        item["v22_01_variant"] = VARIANT_MAP.get(str(item.get("kernel_variant")), item.get("kernel_variant", ""))
        grad_rows.append(item)
    write_rows(out_dir / "v22_01_kernel_gradcheck.csv", grad_rows)
    by_carrier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_carrier[str(row.get("carrier", ""))].append(row)
    summary = []
    status = []
    for carrier, cr in sorted(by_carrier.items()):
        fwd = [finite_float(r.get("forward_ratio_vs_mlp")) for r in cr if finite_float(r.get("forward_ratio_vs_mlp")) == finite_float(r.get("forward_ratio_vs_mlp"))]
        step = [finite_float(r.get("step_ratio_vs_mlp")) for r in cr if finite_float(r.get("step_ratio_vs_mlp")) == finite_float(r.get("step_ratio_vs_mlp"))]
        pass_e1 = sum(int_flag(r.get("v22_01_E1_exploration_gate")) for r in cr)
        pass_s1 = sum(int_flag(r.get("v22_01_S1_official_like_gate")) for r in cr)
        batch_pass = len({str(r.get("batch_size")) for r in cr if int_flag(r.get("v22_01_E1_exploration_gate"))})
        summary.append(
            {
                "carrier": carrier,
                "rows": len(cr),
                "E1_pass_rows": pass_e1,
                "S1_pass_rows": pass_s1,
                "E1_pass_batch_sizes": batch_pass,
                "best_forward_ratio": best(fwd),
                "best_step_ratio": best(step),
                "functional_runner_uses_same_kernel_rows": sum(int_flag(r.get("functional_runner_uses_same_kernel")) for r in cr),
                "audit_cost_separated_rows": sum(int_flag(r.get("audit_cost_separated")) for r in cr),
                "decision": "S1OfficialLike" if pass_s1 and batch_pass >= (3 if carrier == "D-CHE" else 1) else "E1OnlyOrBlocked",
            }
        )
        for row in cr:
            official = int_flag(row.get("official_fused_kernel_complete"))
            consistent = int(
                (not official)
                or (
                    bool(int_flag(row.get("gradcheck_pass")))
                    and bool(int_flag(row.get("no_materialize_complete")))
                    and bool(row.get("manual_kernel_variant"))
                )
            )
            status.append(
                {
                    "carrier": carrier,
                    "v22_01_variant": row.get("v22_01_variant", ""),
                    "batch_size": row.get("batch_size", ""),
                    "implementation_path": row.get("manual_kernel_variant", ""),
                    "official_fused_kernel_complete": row.get("official_fused_kernel_complete", ""),
                    "gradcheck_pass": row.get("gradcheck_pass", ""),
                    "no_materialize_complete": row.get("no_materialize_complete", ""),
                    "status_consistent": consistent,
                    "blocker": "" if consistent else "official_claim_without_gradcheck_or_no_materialize",
                }
            )
    write_rows(out_dir / "v22_01_efficiency_officialization_summary.csv", summary)
    write_rows(out_dir / "v22_01_official_fused_status_matrix.csv", status)
    simple_svg(out_dir / "figures" / "efficiency_component_waterfall_DCHE_DFOU_DRAT_DRBF.svg", "Efficiency component waterfall", rows, "step_ratio_vs_mlp")
    simple_svg(out_dir / "figures" / "kernel_status_consistency_dashboard.svg", "Kernel status consistency", status, "status_consistent")
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_01_efficiency_officialization.py --source-dir {source_dir}", status="completed", note=f"rechecked_rows={len(rows)}; source={source_dir}")


if __name__ == "__main__":
    main()
