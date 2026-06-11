#!/usr/bin/env python3
"""v22.11 S3 retained-source variational solve."""

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
from experiments.run_v22_11_common import PYTHON, append_exec, ensure_out, int_flag, read_json, read_rows, simple_svg, write_json, write_rows  # noqa: E402


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
    s2_route = read_json(source_dir / "v22_11_source_atom_route.json")
    rows: list[dict[str, Any]] = []
    if int_flag(s2_route.get("S2_source_atom_pass_rows")) <= 0:
        route = {
            "route": "S3-BlockedBeforeVariationalSourceSolve",
            "S2_source_atom_pass_rows": int_flag(s2_route.get("S2_source_atom_pass_rows")),
            "S3_variational_source_solve_pass_rows": 0,
            "loss_agnostic_contract_pass": int_flag(s2_route.get("loss_agnostic_contract_pass")),
            "uses_labels_for_direction": 0,
            "uses_loss_for_direction": 0,
            "promotion_allowed": 0,
            "blocker": "S2_source_atom_gate_failed",
        }
    else:
        payload = _load(source_dir / "v22_11_source_atom_payload.pt")
        atom_rows = read_rows(source_dir / "v22_11_source_atom_rows.csv")
        for row in atom_rows:
            row["S2_source_atom_pass"] = row.get("S2_v22_11_source_atom_pass", row.get("S2_source_atom_pass", "0"))
        tensors = payload.get("source_atom_tensors", {})
        logits = payload["logits"].detach().float()
        attempt = dict(payload.get("selected_attempt", {}))
        rows, combo, route = run_loss_agnostic_variational_repair_ladder(
            atom_rows,
            tensors,
            logits,
            norm_scale=float(attempt.get("norm_scale", 0.08) or 0.08),
            split_count=int(attempt.get("split_count", 4) or 4),
            seed=int(payload.get("seed", 2211)),
        )
        route["S2_source_atom_pass_rows"] = int_flag(s2_route.get("S2_source_atom_pass_rows"))
        route["uses_loss_for_direction"] = int_flag(s2_route.get("uses_loss_for_direction"))
        route["uses_loss_interface_cotangent_for_direction"] = int_flag(s2_route.get("uses_loss_interface_cotangent_for_direction"))
        route["uses_loss_formula_specific_direction"] = int_flag(s2_route.get("uses_loss_formula_specific_direction"))
        if combo is not None:
            torch.save({**payload, "target_delta": combo.detach().float(), "variational_route": route}, out_dir / "v22_11_variational_source_target.pt")
        invariance_by_atom = {
            str(row.get("atom_id")): row.get("loss_adapter_invariance_score", "")
            for row in atom_rows
            if row.get("loss_adapter_invariance_score", "") != ""
        }
        for row in rows:
            atom_ids = [part for part in str(row.get("selected_atoms", "")).split(";") if part]
            values = []
            for atom_id in atom_ids:
                try:
                    values.append(float(invariance_by_atom.get(atom_id, "")))
                except Exception:
                    pass
            row["loss_adapter_invariance_score"] = sum(values) / len(values) if values else row.get("loss_adapter_invariance_score", "")
            row["uses_loss_for_direction"] = int_flag(s2_route.get("uses_loss_for_direction"))
            row["uses_loss_interface_cotangent_for_direction"] = int_flag(s2_route.get("uses_loss_interface_cotangent_for_direction"))
            row["uses_loss_formula_specific_direction"] = int_flag(s2_route.get("uses_loss_formula_specific_direction"))
    write_rows(out_dir / "v22_11_variational_source_solve.csv", rows)
    write_json(out_dir / "v22_11_variational_source_route.json", route)
    simple_svg(out_dir / "figures/v22_11_variational_control_projection.svg", "v22.11 variational control projection", rows, "control_projection_fraction")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_11_variational_source_solve.py --source-dir {source_dir} --out-dir {out_dir}",
        status="completed",
        note=f"route={route['route']} pass_rows={route.get('S3_variational_source_solve_pass_rows')} blocker={route.get('blocker')}",
    )


if __name__ == "__main__":
    main()
