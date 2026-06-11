#!/usr/bin/env python3
"""v22.13 kernel-native arbitrary-cotangent efficiency runner."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.profiling.efficiency_v22_13 import NativeEfficiencyConfig, run_native_efficiency_v22_13  # noqa: E402
from experiments.run_v22_13_common import PYTHON, append_exec, ensure_out, int_flag, simple_svg, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-sizes", default="128,256,512")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=2213)
    return p


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    batch_sizes = [int(x) for x in str(args.batch_sizes).split(",") if x.strip()]
    rows, summary, gradcheck = run_native_efficiency_v22_13(
        device_name=args.device,
        batch_sizes=batch_sizes,
        seed=int(args.seed),
        cfg=NativeEfficiencyConfig(hidden=int(args.hidden), repeats=int(args.repeats), warmup=int(args.warmup)),
    )
    write_rows(out_dir / "v22_13_native_efficiency_truth_table.csv", rows)
    write_rows(out_dir / "v22_13_operator_step_efficiency.csv", rows)
    write_rows(out_dir / "v22_13_component_timing_waterfall.csv", rows)
    write_rows(out_dir / "v22_13_manual_vs_native_vjp_comparison.csv", rows)
    write_rows(out_dir / "v22_13_native_kernel_gradcheck.csv", gradcheck)
    write_rows(out_dir / "v22_13_official_fused_status_matrix.csv", summary)
    write_rows(out_dir / "v22_13_DFOU_officialization_matrix.csv", [r for r in rows if r.get("carrier") == "D-FOU"])
    write_rows(out_dir / "v22_13_DCHE_officialization_matrix.csv", [r for r in rows if r.get("carrier") == "D-CHE"])
    official_rows = sum(int_flag(r.get("official_fused_kernel_complete")) for r in rows)
    carrier_pass = [r for r in summary if int_flag(r.get("carrier_specific_efficiency_pass"))]
    route = {
        "route": "S1-NativeOfficialRowsPresent" if official_rows > 0 else "R3-KernelNativeEfficiencyBlocked",
        "profile_rows": len(rows),
        "official_fused_kernel_complete_rows": official_rows,
        "manual_upstream_vjp_rows": sum(int_flag(r.get("manual_upstream_vjp_used")) for r in rows),
        "D-FOU_pass": int(any(r.get("carrier") == "D-FOU" and int_flag(r.get("carrier_specific_efficiency_pass")) for r in summary)),
        "D-CHE_pass": int(any(r.get("carrier") == "D-CHE" and int_flag(r.get("carrier_specific_efficiency_pass")) for r in summary)),
        "carrier_specific_efficiency_pass_rows": len(carrier_pass),
        "promotion_allowed": 0,
        "blocker": "" if official_rows else "arbitrary_cotangent_fused_backward_contract_missing",
    }
    write_json(out_dir / "v22_13_efficiency_native_route.json", route)
    simple_svg(out_dir / "figures/v22_13_DFOU_native_efficiency.svg", "v22.13 D-FOU native efficiency", [r for r in rows if r.get("carrier") == "D-FOU"], "operator_step_ratio_vs_mlp")
    simple_svg(out_dir / "figures/v22_13_DCHE_corrected_layout_efficiency.svg", "v22.13 D-CHE native efficiency", [r for r in rows if r.get("carrier") == "D-CHE"], "operator_step_ratio_vs_mlp")
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_13_efficiency_native.py --device {args.device} --batch-sizes {args.batch_sizes} --hidden {int(args.hidden)} --repeats {int(args.repeats)} --warmup {int(args.warmup)} --seed {int(args.seed)} --out-dir {out_dir}", status="completed" if official_rows > 0 else "blocked", note=f"route={route['route']} fused_complete={official_rows} blocker={route.get('blocker')}")


if __name__ == "__main__":
    main()

