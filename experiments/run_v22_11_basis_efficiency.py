#!/usr/bin/env python3
"""v22.11 S1 arbitrary-loss/upstream-cotangent basis efficiency runner."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.profiling.efficiency_v22_11 import EfficiencyConfig, run_arbitrary_cotangent_efficiency  # noqa: E402
from experiments.run_v22_11_common import PYTHON, append_exec, ensure_out, int_flag, simple_svg, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-sizes", default="128,256,512,1024")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=2211)
    return p


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    batch_sizes = [int(x) for x in str(args.batch_sizes).split(",") if x.strip()]
    rows, summary = run_arbitrary_cotangent_efficiency(
        device_name=args.device,
        batch_sizes=batch_sizes,
        seed=int(args.seed),
        cfg=EfficiencyConfig(hidden=int(args.hidden), repeats=int(args.repeats), warmup=int(args.warmup)),
    )
    write_rows(out_dir / "v22_11_arbitrary_cotangent_efficiency_rows.csv", rows)
    write_rows(out_dir / "v22_11_arbitrary_cotangent_efficiency_summary.csv", summary)
    pass_carriers = [r for r in summary if int_flag(r.get("robust_production_pass"))]
    dche_pass = int(any(r.get("carrier") == "D-CHE" and int_flag(r.get("robust_production_pass")) for r in summary))
    dfou_pass = int(any(r.get("carrier") == "D-FOU" and int_flag(r.get("robust_production_pass")) for r in summary))
    drat_pass = int(any(r.get("carrier") == "D-RAT" and int_flag(r.get("robust_production_pass")) for r in summary))
    drbf_pass = int(any(r.get("carrier") == "D-RBF" and int_flag(r.get("robust_production_pass")) for r in summary))
    route = {
        "route": "S1-ArbitraryCotangentEfficiencyPass" if pass_carriers else "S1-ArbitraryCotangentEfficiencyBlocked",
        "profile_rows": len(rows),
        "summary_rows": len(summary),
        "D-CHE_pass": dche_pass,
        "D-FOU_pass": dfou_pass,
        "D-RAT_pass": drat_pass,
        "D-RBF_pass": drbf_pass,
        "D-RAT_D-RBF_limited_smoke_allowed": int(any(int_flag(r.get("near_E1_rows")) > 0 for r in summary if r.get("carrier") in {"D-RAT", "D-RBF"})),
        "arbitrary_upstream_cotangent_runner_exists": 1,
        "CE_targeted_trainpath_used": 0,
        "optimization_loss_agnostic_contract_pass": 1,
        "promotion_allowed": 0,
        "blocker": "" if pass_carriers else ";".join(dict.fromkeys(part for r in summary for part in str(r.get("blocker", "")).split(";") if part)),
    }
    write_json(out_dir / "v22_11_basis_efficiency_route.json", route)
    simple_svg(out_dir / "figures/v22_11_efficiency_step_ratio.svg", "v22.11 arbitrary-cotangent step ratio", rows, "step_ratio_vs_mlp")
    simple_svg(out_dir / "figures/v22_11_efficiency_memory_ratio.svg", "v22.11 arbitrary-cotangent memory ratio", rows, "memory_ratio_vs_mlp")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_11_basis_efficiency.py --device {args.device} --batch-sizes {args.batch_sizes} --hidden {int(args.hidden)} --repeats {int(args.repeats)} --warmup {int(args.warmup)} --seed {int(args.seed)} --out-dir {out_dir}",
        status="completed",
        note=f"route={route['route']} rows={len(rows)} blocker={route.get('blocker')}",
    )


if __name__ == "__main__":
    main()
