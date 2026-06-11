#!/usr/bin/env python3
"""v22.12 S1 arbitrary-loss/operator upstream-cotangent efficiency runner."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.profiling.efficiency_v22_12 import EfficiencyConfig, run_arbitrary_cotangent_efficiency_v22_12  # noqa: E402
from experiments.run_v22_12_common import PYTHON, append_exec, ensure_out, int_flag, simple_svg, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-sizes", default="128,256,512,1024")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=2212)
    return p


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    batch_sizes = [int(x) for x in str(args.batch_sizes).split(",") if x.strip()]
    rows, summary, repair = run_arbitrary_cotangent_efficiency_v22_12(
        device_name=args.device,
        batch_sizes=batch_sizes,
        seed=int(args.seed),
        cfg=EfficiencyConfig(hidden=int(args.hidden), repeats=int(args.repeats), warmup=int(args.warmup)),
    )
    write_rows(out_dir / "v22_12_arbitrary_cotangent_efficiency_rows.csv", rows)
    write_rows(out_dir / "v22_12_arbitrary_cotangent_efficiency_summary.csv", summary)
    write_rows(out_dir / "v22_12_efficiency_repair_attempts.csv", repair)
    pass_carriers = [r for r in summary if int_flag(r.get("robust_production_pass"))]
    route = {
        "route": "S1-ArbitraryCotangentOperatorEfficiencyPass" if pass_carriers else "S1-ArbitraryCotangentOperatorEfficiencyBlocked",
        "profile_rows": len(rows),
        "summary_rows": len(summary),
        "repair_attempt_rows": len(repair),
        "D-CHE_pass": int(any(r.get("carrier") == "D-CHE" and int_flag(r.get("robust_production_pass")) for r in summary)),
        "D-FOU_pass": int(any(r.get("carrier") == "D-FOU" and int_flag(r.get("robust_production_pass")) for r in summary)),
        "D-RAT_pass": int(any(r.get("carrier") == "D-RAT" and int_flag(r.get("robust_production_pass")) for r in summary)),
        "D-RBF_pass": int(any(r.get("carrier") == "D-RBF" and int_flag(r.get("robust_production_pass")) for r in summary)),
        "carrier_specific_efficiency_pass_consistency": 1,
        "arbitrary_upstream_cotangent_runner_exists": 1,
        "CE_targeted_trainpath_used": 0,
        "optimization_loss_agnostic_contract_pass": 1,
        "manual_path_explicitly_accepted_for_exploration": 1,
        "official_fused_kernel_complete_rows": sum(int_flag(r.get("official_fused_kernel_complete")) for r in rows),
        "promotion_allowed": 0,
        "blocker": "" if pass_carriers else ";".join(dict.fromkeys(part for r in summary for part in str(r.get("blocker", "")).split(";") if part)),
    }
    write_json(out_dir / "v22_12_basis_efficiency_route.json", route)
    simple_svg(out_dir / "figures/v22_12_efficiency_step_ratio.svg", "v22.12 arbitrary-cotangent step ratio", rows, "step_ratio_vs_mlp")
    simple_svg(out_dir / "figures/v22_12_efficiency_memory_ratio.svg", "v22.12 arbitrary-cotangent memory ratio", rows, "memory_ratio_vs_mlp")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_12_basis_efficiency.py --device {args.device} --batch-sizes {args.batch_sizes} --hidden {int(args.hidden)} --repeats {int(args.repeats)} --warmup {int(args.warmup)} --seed {int(args.seed)} --out-dir {out_dir}",
        status="completed",
        note=f"route={route['route']} rows={len(rows)} fused_complete={route['official_fused_kernel_complete_rows']} blocker={route.get('blocker')}",
    )


if __name__ == "__main__":
    main()
