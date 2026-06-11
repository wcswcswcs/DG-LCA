#!/usr/bin/env python3
"""v22.09 metric-as-dynamics solver and source-formation gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dgkan.fu.metric_solver import solve_metric_readout_update  # noqa: E402
from experiments.run_v22_09_common import (  # noqa: E402
    PYTHON,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    read_json,
    simple_svg,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default="")
    return p


def _c2_matrix() -> list[dict[str, Any]]:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(5, 6)
            self.w2 = torch.nn.Parameter(torch.randn(6, 3) * 0.02)

        def frozen_readout_features(self, xb: torch.Tensor) -> torch.Tensor:
            return torch.tanh(self.fc1(xb))

        def forward(self, xb: torch.Tensor) -> torch.Tensor:
            return self.frozen_readout_features(xb) @ self.w2

    torch.manual_seed(2209)
    model = Tiny()
    x = torch.randn(18, 5)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    cases = [
        ("T0_loss_cotangent_all_x1", "T0-LossCotangent", 1.0, "all"),
        ("T3_train_flow_all_x8", "T10-CompensatedBlockSourceChannel", 8.0, "all"),
        ("T4_block_source_hidden_x8", "T10-CompensatedBlockSourceChannel", 8.0, "hidden_only"),
        ("T4_block_source_readout_x8", "T10-CompensatedBlockSourceChannel", 8.0, "readout_only"),
    ]
    rows: list[dict[str, Any]] = []
    for name, target_family, scale, role in cases:
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x).float(), y).backward()
        update = solve_metric_readout_update(
            model,
            x,
            y,
            mechanism="M268-V2206MetricSolverT10G0CompensatedHiddenBlockFU",
            target_family=target_family,
            metric_family="G0-L2",
            target_scale=scale,
            block_role=role,
            seed=2209,
        )
        diag = dict(update.diagnostics or {})
        residual = finite_float(diag.get("projection_residual_Gf"), 999.0)
        actuation = finite_float(diag.get("ActuationR2"), -999.0)
        b2 = finite_float(diag.get("B2_transfer_gain"), -999.0)
        solve_ms = finite_float(diag.get("solve_time_ms"), 999999.0)
        pass_flag = int(residual <= 0.50 and actuation >= 0.20 and b2 >= 0.02 and solve_ms <= 250.0)
        rows.append(
            {
                "case": name,
                "target_family": target_family,
                "target_scale": scale,
                "block_role": role,
                "solver_level": diag.get("solver_level", "S1/S2-readout-block"),
                "projection_residual_Gf": diag.get("projection_residual_Gf", ""),
                "ActuationR2": diag.get("ActuationR2", ""),
                "B2_transfer_gain": diag.get("B2_transfer_gain", ""),
                "B3_safety_gain": diag.get("B3_safety_gain", ""),
                "JVP_count": diag.get("JVP_count", ""),
                "VJP_count": diag.get("VJP_count", ""),
                "CG_iterations": diag.get("CG_iterations", ""),
                "condition_estimate": diag.get("condition_estimate", ""),
                "solve_time_ms": diag.get("solve_time_ms", ""),
                "parameter_update_norm": diag.get("parameter_update_norm", ""),
                "function_displacement_norm": diag.get("target_norm_L2", ""),
                "metric_energy_L2": diag.get("metric_energy_L2", ""),
                "metric_energy_Fisher": diag.get("metric_energy_Fisher", ""),
                "metric_energy_Sobolev": diag.get("metric_energy_Sobolev", ""),
                "metric_energy_RKHS": diag.get("metric_energy_RKHS", ""),
                "NDS": diag.get("NDS", diag.get("target_NDS", "")),
                "uses_future_or_validation": diag.get("uses_future_or_validation", 0),
                "C2_solver_gate_pass": pass_flag,
                "blocker": "" if pass_flag else "projection_residual_or_actuation_or_transfer_or_solve_time_gate_failed",
            }
        )
    return rows


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for role in sorted({str(r.get("block_role", "")) for r in rows}):
        group = [r for r in rows if str(r.get("block_role", "")) == role]
        out.append(
            {
                "block_role": role,
                "rows": len(group),
                "C2_solver_gate_pass_rows": sum(int_flag(r.get("C2_solver_gate_pass")) for r in group),
                "projection_residual_Gf_min": min(finite_float(r.get("projection_residual_Gf"), 999.0) for r in group),
                "ActuationR2_max": max(finite_float(r.get("ActuationR2"), -999.0) for r in group),
                "B2_transfer_gain_max": max(finite_float(r.get("B2_transfer_gain"), -999.0) for r in group),
                "blocker": "" if any(int_flag(r.get("C2_solver_gate_pass")) for r in group) else "projection_residual_or_actuation_or_transfer_gate_failed",
            }
        )
    return out


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    observer_route = read_json(source_dir / "v22_09_retained_source_observer_route.json")
    observer_pass = int_flag(observer_route.get("C0_retained_source_observer_pass_rows"))
    matrix = _c2_matrix()
    summary = _summary(matrix)
    c2_pass_rows = sum(int_flag(r.get("C2_solver_gate_pass")) for r in matrix)

    if not observer_pass:
        source_matrix = [
            {
                "source_formation_status": "blocked_before_C3",
                "reason": "retained_source_certificate_gate_failed",
                "C3_source_formation_pass": 0,
                "official_C3_pass": 0,
            }
        ]
        route = {
            "route": "RetainedSourceObserverLocalNoGo_v22.09",
            "C0_retained_source_observer_pass_rows": observer_pass,
            "C2_matrix_rows": len(matrix),
            "C2_solver_gate_pass_rows": c2_pass_rows,
            "C2_true_block_solver_pass_rows": 0,
            "C3_source_formation_attempted": 0,
            "C3_source_formation_pass_rows": 0,
            "official_C3_pass_rows": 0,
            "promotion_allowed": 0,
            "blocker": "retained_source_certificate_gate_failed",
            "next_codex_action": "stop C-O13/C-O14/C-O15 variants and record observer boundary; do not enter C3 without C0",
        }
    elif c2_pass_rows <= 0:
        source_matrix = [
            {
                "source_formation_status": "blocked_before_C3",
                "reason": "C2_solver_gate_failed",
                "C3_source_formation_pass": 0,
                "official_C3_pass": 0,
            }
        ]
        route = {
            "route": "R-SolverProjectionBlocked_v22.09",
            "C0_retained_source_observer_pass_rows": observer_pass,
            "C2_matrix_rows": len(matrix),
            "C2_solver_gate_pass_rows": c2_pass_rows,
            "C2_true_block_solver_pass_rows": 0,
            "C3_source_formation_attempted": 0,
            "C3_source_formation_pass_rows": 0,
            "official_C3_pass_rows": 0,
            "promotion_allowed": 0,
            "blocker": "C2_solver_gate_failed_after_recompute",
            "next_codex_action": "reduce target rank, add damping, and use block-restricted solve before C3",
        }
    else:
        source_matrix = [
            {
                "source_formation_status": "not_executed_pending_selected_fresh_runner",
                "reason": "C0_and_C2_opened_but_no_C3_runner_claimed_in_this_script",
                "C3_source_formation_pass": 0,
                "official_C3_pass": 0,
            }
        ]
        route = {
            "route": "S3PendingFreshSourceFormation_v22.09",
            "C0_retained_source_observer_pass_rows": observer_pass,
            "C2_matrix_rows": len(matrix),
            "C2_solver_gate_pass_rows": c2_pass_rows,
            "C2_true_block_solver_pass_rows": c2_pass_rows,
            "C3_source_formation_attempted": 0,
            "C3_source_formation_pass_rows": 0,
            "official_C3_pass_rows": 0,
            "promotion_allowed": 0,
            "blocker": "fresh_C3_runner_not_claimed",
            "next_codex_action": "compare direct commit vs optimizer-state/slow-source/block-memory injection",
        }

    write_rows(out_dir / "v22_09_metric_solver_matrix.csv", matrix)
    write_rows(out_dir / "v22_09_metric_solver_summary.csv", summary)
    write_rows(out_dir / "v22_09_source_formation_matrix.csv", source_matrix)
    write_json(out_dir / "v22_09_metric_dynamics_solver_route.json", route)
    simple_svg(out_dir / "figures/v22_09_solver_projection_residual_vs_actuation.svg", "v22.09 solver projection residual", matrix, "projection_residual_Gf")
    simple_svg(out_dir / "figures/v22_09_source_horizon_trajectory.svg", "v22.09 source horizon trajectory", source_matrix, "source_h3200")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_09_metric_dynamics_solver.py --source-dir {source_dir} --out-dir {out_dir}",
        status="completed",
        note=f"observer_pass={observer_pass} c2_rows={len(matrix)} c2_pass_rows={c2_pass_rows} c3_pass=0 route={route['route']}",
    )


if __name__ == "__main__":
    main()
