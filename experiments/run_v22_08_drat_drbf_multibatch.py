#!/usr/bin/env python3
"""v22.08 D-RAT/D-RBF robust multibatch officialization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v22_07_drat_drbf_multibatch as v2207  # noqa: E402
from experiments.run_v22_08_common import PYTHON, append_exec, ensure_out, finite_float, simple_svg, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--official-transition-batch-sizes", default="128,256,512,1024")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--data-root", default="data")
    p.add_argument("--input-size", type=int, default=8)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--param-budget", type=int, default=12000)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--val-size", type=int, default=128)
    p.add_argument("--profiler-repeats", type=int, default=2)
    p.add_argument("--profiler-warmup", type=int, default=1)
    p.add_argument("--repair-iters", type=int, default=12)
    p.add_argument("--repair-warmup", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.003)
    return p


def _f(value: Any, default: float = 0.0) -> float:
    return finite_float(value, default)


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    device = v2207.v2205_repair.repair._device(args.device)
    drat_rows, drat_waterfall = v2207.v2205_repair._drat_official_transition_rows(args, device)
    drbf_rows, drbf_waterfall = v2207.v2205_repair._drbf_official_transition_rows(args, device)
    rows = drat_rows + drbf_rows
    for row in rows:
        v2207._augment_component_fields(row)
        row.update(v2207._classify(row))
        row["v22_08_multibatch_probe"] = 1
    summary = v2207._summary(rows)
    batch_sizes = [int(x) for x in str(args.official_transition_batch_sizes).split(",") if x.strip()]
    needs_repair = any(
        str(r.get("decision")) != "RobustProductionPass"
        or int(r.get("component_telemetry_complete_rows", 0)) < int(r.get("profile_rows", 0))
        for r in summary
    )
    repair_rows: list[dict[str, Any]] = []
    repair_summary: list[dict[str, Any]] = []
    if needs_repair:
        repair_rows, repair_summary = v2207._attempt_repair(args, device, batch_sizes)
    v2207._merge_repair_summary(summary, repair_summary)
    route = {
        "D-RAT_robust_pass": int(any(r.get("carrier") == "D-RAT" and str(r.get("decision")) == "RobustProductionPass" for r in summary)),
        "D-RBF_robust_pass": int(any(r.get("carrier") == "D-RBF" and str(r.get("decision")) == "RobustProductionPass" for r in summary)),
        "D-RAT_near_E1_rows": next((r.get("near_E1_rows") for r in summary if r.get("carrier") == "D-RAT"), 0),
        "D-RBF_near_E1_rows": next((r.get("near_E1_rows") for r in summary if r.get("carrier") == "D-RBF"), 0),
        "multibatch_status_closed": int(all(str(r.get("decision")) == "RobustProductionPass" for r in summary)),
        "limited_smoke_allowed": int(any(int(r.get("near_E1_rows", 0)) >= 3 for r in summary)),
        "repair_attempted": int(bool(repair_summary)),
        "repair_attempt_rows": sum(int(_f(r.get("repair_attempt_rows"), 0.0)) for r in repair_summary),
        "D-RAT_repair_micro_near_E1_rows": next((r.get("repair_micro_near_E1_rows") for r in repair_summary if r.get("carrier") == "D-RAT"), 0),
        "D-RBF_repair_micro_near_E1_rows": next((r.get("repair_micro_near_E1_rows") for r in repair_summary if r.get("carrier") == "D-RBF"), 0),
        "repair_official_closure_claimed": 0,
    }
    write_rows(out_dir / "v22_08_drat_drbf_multibatch_officialization.csv", rows)
    write_rows(out_dir / "v22_08_drat_component_waterfall.csv", drat_waterfall)
    write_rows(out_dir / "v22_08_drbf_component_waterfall.csv", drbf_waterfall)
    write_rows(out_dir / "v22_08_drat_drbf_repair_attempts.csv", repair_rows)
    write_rows(out_dir / "v22_08_drat_drbf_repair_attempts_summary.csv", repair_summary)
    write_rows(out_dir / "v22_08_drat_drbf_multibatch_summary.csv", summary)
    write_json(out_dir / "v22_08_drat_drbf_multibatch_route.json", route)
    simple_svg(out_dir / "figures/v22_08_D-RAT_D-RBF_multibatch_efficiency_dashboard.svg", "v22.08 D-RAT/D-RBF multibatch", rows, "forward_ratio_vs_mlp")
    simple_svg(out_dir / "figures/v22_08_D-RAT_component_waterfall.svg", "v22.08 D-RAT component waterfall", [r for r in rows if r.get("carrier") == "D-RAT"], "forward_ratio_vs_mlp")
    simple_svg(out_dir / "figures/v22_08_D-RBF_component_waterfall.svg", "v22.08 D-RBF component waterfall", [r for r in rows if r.get("carrier") == "D-RBF"], "forward_ratio_vs_mlp")
    simple_svg(out_dir / "figures/v22_08_D-RAT_D-RBF_repair_attempt_dashboard.svg", "v22.08 D-RAT/D-RBF repair attempts", repair_rows, "forward_ratio_vs_mlp")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_08_drat_drbf_multibatch.py --device {args.device} --official-transition-batch-sizes {args.official_transition_batch_sizes} --out-dir {out_dir}",
        status="completed",
        note=f"rows={len(rows)} robust={sum(int(r.get('robust_production_pass',0)) for r in rows)} near={sum(int(r.get('near_E1_v22_07',0)) for r in rows)} telemetry_complete={sum(int(r.get('component_telemetry_complete',0)) for r in rows)} repair_rows={len(repair_rows)} repair_micro_near={sum(int(_f(r.get('micro_near_E1'),0)) for r in repair_rows)}",
    )


if __name__ == "__main__":
    main()

