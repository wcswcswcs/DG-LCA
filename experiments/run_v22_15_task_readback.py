#!/usr/bin/env python3
"""Part D v22.15 task readback gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_15_common import PYTHON, append_exec, ensure_out, init_docs, int_flag, read_json, write_json, write_rows  # noqa: E402


DATASETS = ["MNIST", "FashionMNIST", "KMNIST"]
SEEDS = [0, 1, 2]
VARIANTS = [
    "MLP+AdamW",
    "MLP+SGD",
    "MLP+AdaptiveFU",
    "KAN+AdamW",
    "KAN+AdaptiveFU-readout-diagnostic",
    "KAN+AdaptiveFU-basis-official",
    "KAN+AdaptiveFU-coupled-basis-readout",
    "KAN+AdaptiveFU-source-manifold-diagnostic",
    "KAN+AdaptiveFU-basis-manifold-official",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    return p


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    init_docs()
    command = f"{PYTHON} experiments/run_v22_15_task_readback.py --out-dir {out_dir}"
    s0 = read_json(out_dir / "v22_15_code_truth_route.json")
    eff = read_json(out_dir / "v22_15_efficiency_route.json")
    mlp = read_json(out_dir / "v22_15_mlp_adaptive_route.json")
    geom = read_json(out_dir / "v22_15_loss_geometry_route.json")
    kan = read_json(out_dir / "v22_15_kan_basis_route.json")
    sm = read_json(out_dir / "v22_15_source_manifold_route.json")
    gates = {
        "S0_pass": int_flag(s0.get("S0_pass")),
        "adaptive_efficiency_pass": int_flag(eff.get("adaptive_efficiency_pass")),
        "C1_mlp_adaptive_pass": int_flag(mlp.get("C1_mlp_adaptive_pass")),
        "C3_pairwise_or_pointwise_route": int(int_flag(geom.get("ranking_pairwise_exploration_pass")) or int_flag(geom.get("pointwise_pass"))),
        "C5_KAN_basis_exploration_pass": int_flag(kan.get("KAN_basis_exploration_pass")),
        "C6_source_manifold_gate_if_claimed": int(int_flag(sm.get("MLP_source_manifold_pass")) or int_flag(sm.get("KAN_basis_manifold_pass")) or not sm),
        "controls_fail": int(int_flag(mlp.get("controls_fail")) and int_flag(geom.get("controls_fail")) and int_flag(kan.get("controls_fail")) and int_flag(sm.get("controls_fail", 1))),
    }
    mechanism_ready = int(all(gates.values()))
    rows: list[dict[str, Any]] = []
    if mechanism_ready:
        for dataset in DATASETS:
            for seed in SEEDS:
                for variant in VARIANTS:
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "variant": variant,
                            "status": "not_run",
                            "blocker": "mechanism gates passed but real dataset task runner was not executed in this v22.15 run; no task metric fabricated",
                        }
                    )
        route_name = "TaskReadbackPending_NoTaskMetricFabricated"
    else:
        blocker = ";".join(k for k, v in gates.items() if not v)
        for dataset in DATASETS:
            for seed in SEEDS:
                for variant in VARIANTS:
                    rows.append({"dataset": dataset, "seed": seed, "variant": variant, "status": "not_run", "blocker": f"mechanism_gate_not_passed:{blocker}"})
        route_name = "TaskReadbackDeferredByMechanismGate"
    write_rows(out_dir / "v22_15_task_eval_matrix.csv", rows)
    write_rows(out_dir / "v22_15_convergence_speed_matrix.csv", rows)
    write_rows(out_dir / "v22_15_forgetting_readback_matrix.csv", rows)
    write_rows(out_dir / "v22_15_expression_metrics_matrix.csv", rows)
    write_rows(out_dir / "v22_15_calibration_debt_matrix.csv", rows)
    route = {"route": route_name, "mechanism_ready_for_task": mechanism_ready, **gates}
    write_json(out_dir / "v22_15_task_readback_route.json", route)
    append_exec(out_dir, command, status="completed", gpu="n/a", task_id="D-task-readback", files="v22_15_task_eval_matrix.csv", note=f"route={route_name} mechanism_ready={mechanism_ready}")


if __name__ == "__main__":
    main()
