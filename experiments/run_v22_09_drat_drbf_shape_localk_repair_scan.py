#!/usr/bin/env python3
"""v22.09 D-RAT/D-RBF shape and local-K repair scan.

This is an audit-only scan after the official h128/K4 D-RAT/D-RBF gate remains
blocked. It searches whether lower hidden width or D-RBF K2 can meet the same
v22.09 robust ratios, but does not promote those rows to official closure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v22_05_drat_drbf_repair as v2205  # noqa: E402
from experiments import run_v22_07_drat_drbf_multibatch as v2207  # noqa: E402
from experiments import run_v22_09_basis_efficiency_closure as closure  # noqa: E402
from experiments.run_v22_09_common import (  # noqa: E402
    PYTHON,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--official-transition-batch-sizes", default="128,256,512,1024")
    p.add_argument("--hidden-grid", default="64,96,128")
    p.add_argument("--data-root", default="data")
    p.add_argument("--input-size", type=int, default=8)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--param-budget", type=int, default=12000)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--val-size", type=int, default=128)
    p.add_argument("--profiler-repeats", type=int, default=2)
    p.add_argument("--profiler-warmup", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.003)
    return p


def _shape_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups = sorted({(str(r.get("carrier", "")), str(r.get("repair_variant", "")), str(r.get("scan_hidden", ""))) for r in rows})
    for carrier, variant, hidden in groups:
        group = [r for r in rows if str(r.get("carrier", "")) == carrier and str(r.get("repair_variant", "")) == variant and str(r.get("scan_hidden", "")) == hidden]
        robust_rows = sum(int_flag(r.get("robust_production_pass")) for r in group)
        near_rows = sum(int_flag(r.get("near_E1_v22_09")) for r in group)
        blocker_parts: list[str] = []
        for row in group:
            for part in str(row.get("v22_09_official_blocker", "")).split(";"):
                part = part.strip()
                if part and part not in blocker_parts:
                    blocker_parts.append(part)
        out.append(
            {
                "carrier": carrier,
                "repair_variant": variant,
                "scan_hidden": hidden,
                "profile_rows": len(group),
                "robust_production_pass_rows": robust_rows,
                "near_E1_rows": near_rows,
                "best_forward_ratio": min(finite_float(r.get("forward_ratio_vs_mlp"), 999.0) for r in group),
                "best_backward_ratio": min(finite_float(r.get("backward_ratio_vs_mlp"), 999.0) for r in group),
                "best_step_ratio": min(finite_float(r.get("step_ratio_vs_mlp"), 999.0) for r in group),
                "best_memory_ratio": min(finite_float(r.get("memory_ratio_vs_mlp"), 999.0) for r in group),
                "shape_localk_candidate": int(robust_rows >= 3),
                "official_closure_claimed": 0,
                "blocker": ";".join(blocker_parts),
            }
        )
    return out


def _measure_group(args: argparse.Namespace, device: Any, *, carrier: str, hidden: int, repair_variant: str, component_variant: str, manual_token: str) -> list[dict[str, Any]]:
    local = argparse.Namespace(**vars(args))
    local.hidden = int(hidden)
    rows, _waterfall = v2205._official_transition_rows(  # type: ignore[attr-defined]
        local,
        device,
        carrier=carrier,
        repair_variant=repair_variant,
        component_variant=component_variant,
        manual_token=manual_token,
    )
    for row in rows:
        row["scan_hidden"] = int(hidden)
        row["shape_localk_scan"] = 1
        row["promotion_allowed"] = 0
        v2207._augment_component_fields(row)
        closure._normalize_components(row)
        row.update(closure._classify_v2209(row))
        row["official_closure_claimed"] = 0
    return rows


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    device = v2207.v2205_repair.repair._device(args.device)
    hidden_grid = [int(x) for x in str(args.hidden_grid).split(",") if x.strip()]

    rows: list[dict[str, Any]] = []
    for hidden in hidden_grid:
        rows.extend(
            _measure_group(
                args,
                device,
                carrier="D-RAT",
                hidden=hidden,
                repair_variant="RAT22.05-official-rational-k4-triton",
                component_variant="RAT22.05-shape-scan-rational-k4-triton-trainpath",
                manual_token="rational_k4_triton_l3_matmul",
            )
        )
        rows.extend(
            _measure_group(
                args,
                device,
                carrier="D-RBF",
                hidden=hidden,
                repair_variant="RBF22.03-R1-compact-local-k4-no-dense",
                component_variant="RBF22.09-shape-scan-rbf-k4-triton-trainpath",
                manual_token="rbf_k4_triton_l3_matmul",
            )
        )
        rows.extend(
            _measure_group(
                args,
                device,
                carrier="D-RBF",
                hidden=hidden,
                repair_variant="RBF22.03-R1-compact-local-low-k2-no-dense",
                component_variant="RBF22.09-localK2-shape-scan-rbf-k2-triton-trainpath",
                manual_token="rbf_k2_triton_l3_matmul",
            )
        )

    summary = _shape_summary(rows)
    candidate_groups = sum(int_flag(r.get("shape_localk_candidate")) for r in summary)
    route = {
        "route": "D-RAT_D-RBFShapeLocalKRepairCandidateOfficialBlocked" if candidate_groups else "D-RAT_D-RBFShapeLocalKRepairBlocked",
        "scan_rows": len(rows),
        "summary_rows": len(summary),
        "shape_localk_candidate_groups": candidate_groups,
        "official_closure_claimed": 0,
        "promotion_allowed": 0,
        "blocker": "official_h128_k4_gate_not_replaced;D-RAT_D-RBF_main_multibatch_gate_still_required",
    }
    write_rows(out_dir / "v22_09_drat_drbf_shape_localk_repair_scan.csv", rows)
    write_rows(out_dir / "v22_09_drat_drbf_shape_localk_repair_scan_summary.csv", summary)
    write_json(out_dir / "v22_09_drat_drbf_shape_localk_repair_scan_route.json", route)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_09_drat_drbf_shape_localk_repair_scan.py --device {args.device} --official-transition-batch-sizes {args.official_transition_batch_sizes} --hidden-grid {args.hidden_grid} --out-dir {out_dir}",
        status="completed",
        note=f"shape_localK_route={route['route']} scan_rows={len(rows)} candidate_groups={candidate_groups}",
    )


if __name__ == "__main__":
    main()
