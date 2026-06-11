#!/usr/bin/env python3
"""v22.10 S3 retained-source variational solve."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dgkan.fu.variational_source_solver import run_loss_agnostic_variational_repair_ladder  # noqa: E402
from experiments.run_v22_10_common import PYTHON, append_exec, ensure_out, int_flag, read_json, read_rows, simple_svg, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default="")
    return p


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    source_route = read_json(source_dir / "v22_10_source_atom_route.json")
    if int_flag(source_route.get("S2_source_atom_pass_rows")) <= 0:
        route = {
            "route": "S3-BlockedBeforeVariationalSolve",
            "S2_source_atom_pass_rows": int_flag(source_route.get("S2_source_atom_pass_rows")),
            "S3_variational_source_solve_pass_rows": 0,
            "promotion_allowed": 0,
            "blocker": "S2_source_atom_generation_gate_failed",
        }
        write_rows(out_dir / "v22_10_variational_solve_matrix.csv", [])
        write_json(out_dir / "v22_10_variational_source_route.json", route)
    else:
        payload = _load_payload(source_dir / "v22_10_source_atom_payload.pt")
        selected_attempt = dict(payload.get("selected_attempt", {}))
        all_rows = read_rows(source_dir / "v22_10_source_atom_rows.csv")
        atom_rows = [dict(r) for r in all_rows if str(r.get("attempt")) == str(selected_attempt.get("attempt"))]
        tensors = {str(k): v.detach().float() for k, v in dict(payload.get("source_atom_tensors", {})).items()}
        logits = payload["logits"].detach().float()
        solve_rows, combo, route = run_loss_agnostic_variational_repair_ladder(
            atom_rows,
            tensors,
            logits,
            norm_scale=float(selected_attempt.get("norm_scale", 0.02)),
            split_count=int(selected_attempt.get("split_count", 4)),
            seed=int(payload.get("seed", 2210)),
        )
        route.update(
            {
                "S2_selected_attempt": selected_attempt.get("attempt", ""),
                "S2_source_atom_pass_rows": int_flag(source_route.get("S2_source_atom_pass_rows")),
                "promotion_allowed": 0,
            }
        )
        if combo is not None:
            torch.save(
                {
                    "seed": int(payload.get("seed", 2210)),
                    "model_config": payload.get("model_config", {}),
                    "model_state": payload.get("model_state", {}),
                    "x": payload.get("x"),
                    "logits": payload.get("logits"),
                    "target_delta": combo.detach().float(),
                    "loss_agnostic_contract_pass": int_flag(route.get("loss_agnostic_contract_pass")),
                    "selected_attempt": route.get("selected_attempt", ""),
                    "selected_atoms": route.get("selected_atoms", ""),
                    "S3_variational_source_solve_pass": int_flag(route.get("S3_variational_source_solve_pass_rows")),
                },
                out_dir / "v22_10_variational_source_target.pt",
            )
        write_rows(out_dir / "v22_10_variational_solve_matrix.csv", solve_rows)
        write_json(out_dir / "v22_10_variational_source_route.json", route)
        simple_svg(out_dir / "figures/v22_10_variational_objective_decomposition.svg", "v22.10 variational solve", solve_rows, "DDR")
        simple_svg(out_dir / "figures/v22_10_variational_control_projection.svg", "v22.10 control projection", solve_rows, "control_projection_fraction")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_10_variational_source_solve.py --source-dir {source_dir} --out-dir {out_dir}",
        status="completed",
        note=f"route={read_json(out_dir / 'v22_10_variational_source_route.json').get('route')} pass_rows={read_json(out_dir / 'v22_10_variational_source_route.json').get('S3_variational_source_solve_pass_rows')}",
    )


if __name__ == "__main__":
    main()
