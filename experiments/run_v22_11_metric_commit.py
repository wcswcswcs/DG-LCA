#!/usr/bin/env python3
"""v22.11 S4 metric-as-dynamics commit and optimizer-state integration probe."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dgkan.fu.constructive_commit import solve_external_target_commit  # noqa: E402
from experiments.run_v22_11_common import PYTHON, append_exec, ensure_out, int_flag, read_json, simple_svg, write_json, write_rows  # noqa: E402
from experiments.run_v22_11_source_atom_generation import TinyArbitraryLossMLP  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default="")
    return p


def _load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    s3_route = read_json(source_dir / "v22_11_variational_source_route.json")
    rows: list[dict[str, Any]] = []
    if int_flag(s3_route.get("S3_variational_source_solve_pass_rows")) <= 0:
        route = {
            "route": "S4-BlockedBeforeMetricDynamicsCommit",
            "S3_variational_source_solve_pass_rows": int_flag(s3_route.get("S3_variational_source_solve_pass_rows")),
            "S4_metric_dynamics_commit_pass_rows": 0,
            "loss_agnostic_contract_pass": int_flag(s3_route.get("loss_agnostic_contract_pass")),
            "uses_labels_for_direction": 0,
            "uses_loss_for_direction": 0,
            "promotion_allowed": 0,
            "blocker": "S3_variational_source_solve_gate_failed",
        }
    else:
        payload = _load(source_dir / "v22_11_variational_source_target.pt")
        config = dict(payload.get("model_config", {}))
        model = TinyArbitraryLossMLP(int(config.get("input_dim", 8)), int(config.get("hidden", 64)), int(config.get("classes", 5)))
        model.load_state_dict(payload["model_state"])
        x = payload["x"].detach().float()
        target = payload["target_delta"].detach().float()
        attempts = [
            ("S4.1-readout-exact", "readout_only", 1.0e-3, 0.0, "split_A"),
            ("S4.2-readout-all-train-fit", "readout_only", 1.0e-3, 0.0, "all_train_stream"),
            ("S4.3-hidden-readout-block", "hidden_readout", 1.0e-3, 0.0, "split_A"),
            ("S4.4-optimizer-state-only-write", "optimizer_state_only", 1.0e-3, 1.0, "split_A"),
            ("S4.5-source-state-integrated-all-train", "readout_only", 5.0e-4, 0.35, "all_train_stream"),
        ]
        selected_update: torch.Tensor | None = None
        for solver_level, block_role, damping, state_write, fit_scope in attempts:
            update, diag = solve_external_target_commit(
                model,
                x,
                None,
                target,
                solver_level=solver_level,
                block_role=block_role,
                damping=damping,
                optimizer_state_write_fraction=state_write,
                fit_scope=fit_scope,
                seed=int(payload.get("seed", 2211)),
            )
            row = {
                "solver_level": solver_level,
                "damping": damping,
                "source_state_alignment": diag.get("function_displacement_cos_with_target", ""),
                "source_state_decay_rate": 0.0,
                "base_update_destructive_projection_fraction": 0.0,
                "uses_CE_specific_formula": 0,
                **diag,
            }
            rows.append(row)
            if int_flag(row.get("S4_metric_dynamics_commit_pass")) and selected_update is None:
                selected_update = update.detach().float()
                torch.save({**payload, "update_vec": selected_update, "commit_row": row}, out_dir / "v22_11_metric_commit_payload.pt")
        pass_rows = sum(int_flag(r.get("S4_metric_dynamics_commit_pass")) for r in rows)
        route = {
            "route": "S4-MetricDynamicsCommitPass" if pass_rows else "S4-MetricDynamicsCommitNoGo",
            "S3_variational_source_solve_pass_rows": int_flag(s3_route.get("S3_variational_source_solve_pass_rows")),
            "S4_metric_dynamics_commit_pass_rows": pass_rows,
            "selected_solver_level": next((r.get("solver_level", "") for r in rows if int_flag(r.get("S4_metric_dynamics_commit_pass"))), ""),
            "loss_agnostic_contract_pass": int(all(int_flag(r.get("loss_agnostic_contract_pass", 0)) for r in rows)),
            "uses_labels_for_direction": int(any(int_flag(r.get("uses_labels_for_direction", 0)) for r in rows)),
            "uses_loss_for_direction": int(any(int_flag(r.get("uses_loss_for_direction", 0)) for r in rows)),
            "uses_CE_specific_formula": 0,
            "promotion_allowed": 0,
            "blocker": "" if pass_rows else ";".join(dict.fromkeys(part for r in rows for part in str(r.get("blocker", "")).split(";") if part)),
        }
    write_rows(out_dir / "v22_11_metric_commit_matrix.csv", rows)
    write_json(out_dir / "v22_11_metric_commit_route.json", route)
    simple_svg(out_dir / "figures/v22_11_projection_residual_vs_actuation.svg", "v22.11 S4 projection residual", rows, "projection_residual_Gf")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_11_metric_commit.py --source-dir {source_dir} --out-dir {out_dir}",
        status="completed",
        note=f"route={route['route']} pass_rows={route['S4_metric_dynamics_commit_pass_rows']} blocker={route.get('blocker')}",
    )


if __name__ == "__main__":
    main()
