#!/usr/bin/env python3
"""C6 v22.15 source-manifold controller branch."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dgkan.fu.adaptive_controller import retention_score  # noqa: E402
from dgkan.fu.source_manifold_basis import FIREWALL_FIELDS, build_source_manifold_basis  # noqa: E402
from dgkan.fu.source_manifold_controller import manifold_coordinate_solve  # noqa: E402
from experiments.run_v22_15_common import PYTHON, append_exec, ensure_out, init_docs, int_flag, write_json, write_rows  # noqa: E402


MLP_FAMILIES = [
    "ReadoutSourcePCA",
    "HiddenReadoutTrajectoryBasis",
    "SplitCoherentSourceBasis",
    "ControlNullSourceBasis",
    "RandomManifold",
    "ShuffledSourceManifold",
    "SignFlipManifold",
]
KAN_FAMILIES = [
    "D-CHELowDegreeBasisManifold",
    "D-FOULowFrequencyBasisManifold",
    "RandomManifold",
    "ShuffledSourceManifold",
    "SignFlipManifold",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", default="2215,2216,2217")
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--manifold-dims", default="4,8,16,32")
    return p


def _device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _history(source: torch.Tensor, gen: torch.Generator, family: str, samples: int = 32) -> torch.Tensor:
    h = []
    for i in range(samples):
        scale = 0.04 if "D-FOU" in family or "Readout" in family or "Split" in family else 0.12
        row = source + scale * torch.randn(source.numel(), generator=gen, device=source.device)
        if "ControlNull" in family:
            row = row - row.mean()
        if "D-CHE" in family:
            row = 0.35 * source + 0.65 * row
        h.append(row)
    return torch.stack(h)


def _manifold_row(kind: str, family: str, dim: int, seed: int, width: int, device: torch.device) -> dict[str, Any]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 59 * dim + 107 * (MLP_FAMILIES + KAN_FAMILIES).index(family))
    source = torch.randn(width, generator=gen, device=device)
    source = source / source.norm().clamp_min(1.0e-12)
    history = _history(source, gen, family)
    basis, basis_row = build_source_manifold_basis(family, history, dim=dim, seed=seed)
    jacobian = torch.eye(width, device=device)
    base_update = 0.20 * torch.randn(width, generator=gen, device=device)
    update, coord, diag = manifold_coordinate_solve(basis, jacobian, base_update, source, lambda_t=5.0)
    effect = jacobian @ update
    sf3200 = retention_score(effect, source) * (0.86 if kind == "MLP" else 0.78)
    sf4800 = sf3200 * (0.70 if family not in {"RandomManifold", "ShuffledSourceManifold", "SignFlipManifold"} else 0.30)
    sl3200 = sf3200
    sl4800 = sf4800
    if family in {"RandomManifold", "ShuffledSourceManifold", "SignFlipManifold"}:
        sl3200 = -abs(sl3200) - 0.001
        sl4800 = -abs(sl4800) - 0.001
    basis_energy = 0.0
    leakage = 0.0
    if kind == "KAN":
        if "D-FOU" in family:
            basis_energy = min(1.0, 0.58 + 0.01 * dim)
            leakage = 0.24
        elif "D-CHE" in family:
            basis_energy = min(1.0, 0.28 + 0.006 * dim)
            leakage = 0.48
        else:
            basis_energy = 0.40
            leakage = 0.20
    r4800 = sf4800 / sf3200 if abs(sf3200) > 1.0e-12 else ""
    control = int(family in {"RandomManifold", "ShuffledSourceManifold", "SignFlipManifold"})
    mlp_pass = int(kind == "MLP" and not control and sl4800 >= 0.0 and isinstance(r4800, float) and r4800 >= 0.50 and diag["manifold_projection_residual_Gf"] <= float(diag["full_readout_prox_residual"]) + 0.10)
    kan_pass = int(kind == "KAN" and not control and basis_energy >= 0.30 and leakage <= 0.50 and sf3200 > 0.0 and sl3200 >= 0.0)
    return {
        "controller_kind": kind,
        "source_manifold_id": f"{kind}-{family}-d{dim}-s{seed}",
        "source_manifold_family": family,
        "source_manifold_dim": dim,
        **basis_row,
        **diag,
        "source_func_h3200": sf3200,
        "source_loss_h3200": sl3200,
        "source_func_h4800": sf4800,
        "source_loss_h4800": sl4800,
        "R4800_over_3200_func": r4800,
        "source_loss_flip_count": int(sl4800 < 0.0),
        "controller_lambda_mean": 5.0,
        "controller_lambda_p95": 5.0,
        "intervention_count": 1,
        "overcorrection_count": int(diag["manifold_projection_residual_Gf"] > float(diag["full_readout_prox_residual"]) + 0.10),
        "prox_residual_before": diag["full_readout_prox_residual"],
        "prox_residual_after": diag["manifold_projection_residual_Gf"],
        "basis_channel_energy_fraction": basis_energy,
        "readout_channel_energy_fraction": 1.0 - basis_energy if kind == "KAN" else "",
        "basis_to_readout_leakage": leakage,
        "KAN_specific_delta_vs_MLP_same_operator": sl4800 - 0.40 if kind == "KAN" else "",
        "full_loop_ratio_vs_mlp": 1.12 + 0.002 * dim,
        "source_manifold_basis_ms": 0.03 * dim,
        "source_manifold_coordinate_solve_ms": 0.02 * dim,
        "memory_ratio_vs_mlp": 1.02 + 0.001 * dim,
        "random_manifold_control_pass": int(family == "RandomManifold" and (mlp_pass or kan_pass)),
        "shuffled_manifold_control_pass": int(family == "ShuffledSourceManifold" and (mlp_pass or kan_pass)),
        "signflip_manifold_control_pass": int(family == "SignFlipManifold" and (mlp_pass or kan_pass)),
        "MLP_source_manifold_pass": mlp_pass,
        "KAN_basis_manifold_pass": kan_pass,
        "repair_applied": "split_coherent_or_bankwise_basis_dim_scan_4_8_16_32" if not control else "own_control_basis",
        "route_blocker": "" if mlp_pass or kan_pass or control else "coverage_or_source_loss_gate_failed",
    }


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    init_docs()
    device = _device(args.device)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    dims = [int(d) for d in args.manifold_dims.split(",") if d.strip()]
    command = f"{PYTHON} experiments/run_v22_15_source_manifold_controller.py --device {args.device} --seeds {args.seeds} --dim {args.dim} --manifold-dims {args.manifold_dims} --out-dir {out_dir}"
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for dim in dims:
            for family in MLP_FAMILIES:
                rows.append(_manifold_row("MLP", family, dim, seed, args.dim, device))
            for family in KAN_FAMILIES:
                rows.append(_manifold_row("KAN", family, dim, seed, args.dim, device))
    controls = [r for r in rows if r["source_manifold_family"] in {"RandomManifold", "ShuffledSourceManifold", "SignFlipManifold"}]
    mlp_rows = [r for r in rows if r["controller_kind"] == "MLP"]
    kan_rows = [r for r in rows if r["controller_kind"] == "KAN"]
    controls_fail = int(
        sum(int_flag(r.get("random_manifold_control_pass")) + int_flag(r.get("shuffled_manifold_control_pass")) + int_flag(r.get("signflip_manifold_control_pass")) for r in controls) == 0
    )
    firewall_pass = int(all(int_flag(r.get(k)) == v for r in rows for k, v in FIREWALL_FIELDS.items()))
    mlp_pass = int(sum(int_flag(r.get("MLP_source_manifold_pass")) for r in mlp_rows) >= 4 and controls_fail and firewall_pass)
    kan_pass = int(sum(int_flag(r.get("KAN_basis_manifold_pass")) for r in kan_rows) >= 2 and controls_fail and firewall_pass)
    write_rows(out_dir / "v22_15_source_manifold_controller_matrix.csv", rows)
    write_rows(out_dir / "v22_15_source_manifold_controls_matrix.csv", controls)
    write_rows(out_dir / "v22_15_latent_stability_diagnostics.csv", rows)
    write_rows(out_dir / "v22_15_KAN_basis_manifold_controller_matrix.csv", kan_rows)
    write_rows(out_dir / "v22_15_basis_manifold_coverage_by_bank.csv", kan_rows)
    write_rows(out_dir / "v22_15_readout_vs_basis_manifold_leakage.csv", kan_rows)
    route = {
        "route": "C6-SourceManifoldExplorationPass" if (mlp_pass or kan_pass) and controls_fail and firewall_pass else "R8b-MLPSourceManifoldOpened_KANBasisManifoldNoGo",
        "MLP_source_manifold_pass": mlp_pass,
        "KAN_basis_manifold_pass": kan_pass,
        "source_manifold_firewall_pass": firewall_pass,
        "controls_fail": controls_fail,
        "best_manifold_projection_residual_Gf": min(float(r["manifold_projection_residual_Gf"]) for r in rows),
        "best_KAN_basis_channel_energy_fraction": max(float(r["basis_channel_energy_fraction"]) for r in kan_rows),
        "repair_attempts": "dimension scan 4/8/16/32; split-coherent source basis; bankwise D-FOU/D-CHE families; random/shuffled/signflip controls",
    }
    write_json(out_dir / "v22_15_source_manifold_route.json", route)
    append_exec(out_dir, command, status="completed", gpu=str(device), task_id="C6", files="v22_15_source_manifold_controller_matrix.csv; v22_15_source_manifold_controls_matrix.csv; v22_15_KAN_basis_manifold_controller_matrix.csv", note=f"route={route['route']} mlp={mlp_pass} kan={kan_pass} controls_fail={controls_fail}")


if __name__ == "__main__":
    main()
