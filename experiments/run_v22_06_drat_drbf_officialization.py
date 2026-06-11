#!/usr/bin/env python3
"""v22.06 D-RAT/D-RBF production officialization wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.profiling.efficiency_v22_06 import classify_drat_drbf_v22_06  # noqa: E402
from experiments import run_v22_05_drat_drbf_repair as v2205_repair  # noqa: E402
from experiments.run_v22_06_common import PYTHON, append_exec, ensure_out, finite_float, simple_svg, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--official-transition-batch-sizes", default="512")
    p.add_argument("--hidden", type=int, default=128)
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


def _f(value: Any, default: float = 999.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except Exception:
        return default


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for carrier in sorted({str(r.get("carrier", "")) for r in rows}):
        group = [r for r in rows if str(r.get("carrier", "")) == carrier]
        pass_rows = sum(int(r.get("production_fused_official_pass", 0)) for r in group)
        out.append(
            {
                "carrier": carrier,
                "profile_rows": len(group),
                "production_fused_official_rows": pass_rows,
                "near_E1_rows": sum(int(r.get("near_E1", 0)) for r in group),
                "best_forward_ratio": min(_f(r.get("forward_ratio_vs_mlp")) for r in group),
                "best_step_ratio": min(_f(r.get("step_ratio_vs_mlp")) for r in group),
                "best_memory_ratio": min(_f(r.get("memory_ratio_vs_mlp")) for r in group),
                "decision": "ProductionFusedPass" if pass_rows else "OfficialFusedBlocked",
                "blocker": "" if pass_rows else ";".join(sorted({str(r.get("v22_06_official_blocker", "")) for r in group if str(r.get("v22_06_official_blocker", ""))})),
            }
        )
    return out


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    device = v2205_repair.repair._device(args.device)
    drat_rows, drat_waterfall = v2205_repair._drat_official_transition_rows(args, device)
    drbf_rows, drbf_waterfall = v2205_repair._drbf_official_transition_rows(args, device)
    rows = drat_rows + drbf_rows
    for row in rows:
        if str(row.get("carrier")) == "D-RAT":
            den_min = finite_float(row.get("rational_den_min"), 1.0)
            den_p01 = finite_float(row.get("rational_den_p01"), 1.0)
            row["den_safety_pass"] = int((den_min != den_min or den_min > 1.0e-6) and (den_p01 != den_p01 or den_p01 > 1.0e-6))
        else:
            row["den_safety_pass"] = 1
        row.update(classify_drat_drbf_v22_06(row))
    summary = _summary(rows)
    write_rows(out_dir / "v22_06_drat_drbf_officialization.csv", rows)
    write_rows(out_dir / "v22_06_drat_component_waterfall.csv", drat_waterfall)
    write_rows(out_dir / "v22_06_drbf_component_waterfall.csv", drbf_waterfall)
    write_rows(out_dir / "v22_06_drat_drbf_officialization_summary.csv", summary)
    write_json(out_dir / "v22_06_drat_drbf_officialization_route.json", {r["carrier"]: r for r in summary})
    simple_svg(out_dir / "figures/D-RAT_component_waterfall.svg", "v22.06 D-RAT component waterfall", [r for r in rows if r.get("carrier") == "D-RAT"], "forward_ratio_vs_mlp")
    simple_svg(out_dir / "figures/D-RBF_component_waterfall.svg", "v22.06 D-RBF component waterfall", [r for r in rows if r.get("carrier") == "D-RBF"], "forward_ratio_vs_mlp")
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_06_drat_drbf_officialization.py --device {args.device} --official-transition-batch-sizes {args.official_transition_batch_sizes} --out-dir {out_dir}", status="completed", note=f"rows={len(rows)} pass={sum(int(r.get('production_fused_official_pass',0)) for r in rows)}")


if __name__ == "__main__":
    main()
