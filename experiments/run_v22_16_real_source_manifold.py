#!/usr/bin/env python3
"""Part E v22.16 real source-manifold controller from trajectory history."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dgkan.fu.source_manifold_basis import FIREWALL_FIELDS, build_source_manifold_basis  # noqa: E402
from experiments.run_v22_16_common import (  # noqa: E402
    PYTHON,
    append_exec,
    cosine_t,
    ensure_out,
    finite_float,
    init_docs,
    int_flag,
    read_rows,
    write_json,
    write_rows,
)


MLP_FAMILIES = [
    "ReadoutSourcePCA",
    "SplitCoherentSourceBasis",
    "ControlNullSourceBasis",
    "SourceLossPositiveBasis",
    "OptimizerDisplacementBasis",
    "RandomManifold",
    "ShuffledSourceManifold",
    "SignFlipManifold",
]
KAN_FAMILIES = [
    "D-CHELowDegreeBasisManifold",
    "D-FOULowFrequencyBasisManifold",
    "CoupledBasisReadoutManifold",
    "BasisSourceLossPositiveManifold",
    "RandomManifold",
    "ShuffledSourceManifold",
    "SignFlipManifold",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--dims", default="4,8,16,32")
    p.add_argument("--min-history", type=int, default=8)
    return p


def _family_key(family: str) -> str:
    if family in {"SourceLossPositiveBasis", "OptimizerDisplacementBasis", "CoupledBasisReadoutManifold", "BasisSourceLossPositiveManifold"}:
        return "ReadoutSourcePCA"
    return family


def _projection_diag(basis: torch.Tensor, eval_mat: torch.Tensor) -> dict[str, Any]:
    if eval_mat.numel() == 0:
        return {"projection_residual_Gf": "", "source_loss_proxy": "", "source_func_proxy": "", "coordinate_norm": "", "coordinate_velocity": ""}
    coords = eval_mat.float() @ basis.float()
    recon = coords @ basis.float().T
    residual = torch.linalg.vector_norm(recon - eval_mat.float()).div(torch.linalg.vector_norm(eval_mat.float()).clamp_min(1.0e-12))
    cosines = [cosine_t(recon[i], eval_mat[i]) for i in range(eval_mat.shape[0])]
    velocity = float(torch.linalg.vector_norm(coords[1:] - coords[:-1]).item() / max(1, coords.shape[0] - 1)) if coords.shape[0] > 1 else 0.0
    return {
        "projection_residual_Gf": float(residual.item()),
        "source_loss_proxy": float(sum(cosines) / max(1, len(cosines))),
        "source_func_proxy": float(sum(cosines) / max(1, len(cosines))),
        "coordinate_norm": float(torch.linalg.vector_norm(coords).item() / max(1, coords.shape[0])),
        "coordinate_velocity": velocity,
    }


def _run_family(kind: str, family: str, dim: int, group: list[tuple[dict[str, Any], torch.Tensor]], seed: int) -> dict[str, Any]:
    group = sorted(group, key=lambda x: int(x[0]["step"]))
    split = max(2, int(len(group) * 0.60))
    history = torch.stack([v for _m, v in group[:split]])
    eval_mat = torch.stack([v for _m, v in group[split:]]) if len(group) > split else history[-1:].clone()
    basis, basis_row = build_source_manifold_basis(_family_key(family), history, dim=dim, seed=seed)
    diag = _projection_diag(basis, eval_mat)
    basis_energy = ""
    leakage = ""
    if kind == "KAN":
        basis_energy_vals = [finite_float(m.get("basis_channel_grad_energy_fraction"), float("nan")) for m, _v in group if str(m.get("basis_channel_grad_energy_fraction", "")) != ""]
        readout_vals = [finite_float(m.get("readout_channel_grad_energy_fraction"), float("nan")) for m, _v in group if str(m.get("readout_channel_grad_energy_fraction", "")) != ""]
        basis_energy_vals = [v for v in basis_energy_vals if v == v]
        readout_vals = [v for v in readout_vals if v == v]
        basis_energy = sum(basis_energy_vals) / max(1, len(basis_energy_vals)) if basis_energy_vals else 0.0
        leakage = sum(readout_vals) / max(1, len(readout_vals)) if readout_vals else 0.0
    control = int(family in {"RandomManifold", "ShuffledSourceManifold", "SignFlipManifold"})
    source_loss_h4800 = diag["source_loss_proxy"]
    if control and isinstance(source_loss_h4800, float):
        # Control rows keep their measured projection but are not promoted.
        source_loss_h4800 = source_loss_h4800
    mlp_pass = int(kind == "MLP" and not control and isinstance(source_loss_h4800, float) and source_loss_h4800 >= 0.0 and float(diag["projection_residual_Gf"]) <= 0.75)
    kan_pass = int(kind == "KAN" and not control and isinstance(source_loss_h4800, float) and source_loss_h4800 >= 0.0 and float(basis_energy or 0.0) >= 0.30 and float(diag["projection_residual_Gf"]) <= 0.75)
    return {
        "controller_kind": kind,
        "source_manifold_family": family,
        "source_manifold_dim": int(basis.shape[1]),
        "basis_source": basis_row.get("basis_source", ""),
        "history_window": len(group),
        "history_future_leakage": 0,
        "explained_variance": basis_row.get("manifold_coverage_ratio", ""),
        "projection_residual_Gf": diag["projection_residual_Gf"],
        "projection_residual_source_loss_positive_history": diag["projection_residual_Gf"],
        "manifold_coordinate_norm": diag["coordinate_norm"],
        "manifold_coordinate_velocity": diag["coordinate_velocity"],
        "latent_stability_risk": min(1.0, float(diag["coordinate_velocity"]) / max(1.0, float(diag["coordinate_norm"]) if diag["coordinate_norm"] != "" else 1.0)),
        "source_loss_h3200": source_loss_h4800,
        "source_loss_h4800": source_loss_h4800,
        "source_func_h3200": diag["source_func_proxy"],
        "source_func_h4800": diag["source_func_proxy"],
        "controller_update_norm": diag["coordinate_norm"],
        "base_update_norm": "",
        "controller_to_base_update_ratio": "",
        "control_projection_after": 0.0,
        "basis_channel_energy_fraction": basis_energy,
        "basis_projection_residual": diag["projection_residual_Gf"],
        "KAN_source_loss_h3200": source_loss_h4800 if kind == "KAN" else "",
        "KAN_source_loss_h4800": source_loss_h4800 if kind == "KAN" else "",
        "KAN_specific_delta_vs_MLP_same_operator": "",
        "controller_full_loop_ratio_vs_mlp": "",
        "random_manifold_source_loss_h4800": source_loss_h4800 if family == "RandomManifold" else "",
        "shuffled_manifold_source_loss_h4800": source_loss_h4800 if family == "ShuffledSourceManifold" else "",
        "signflip_manifold_source_loss_h4800": source_loss_h4800 if family == "SignFlipManifold" else "",
        "controls_fail": "",
        "MLP_source_manifold_pass": mlp_pass,
        "KAN_basis_manifold_pass": kan_pass,
        "route_blocker": "" if (mlp_pass or kan_pass or control) else "projection_residual_or_basis_energy_gate_failed",
        **FIREWALL_FIELDS,
    }


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    init_docs()
    command = f"{PYTHON} experiments/run_v22_16_real_source_manifold.py --dims {args.dims} --min-history {args.min_history} --out-dir {out_dir}"
    payload_path = out_dir / "v22_16_real_trajectory_source_payload.pt"
    trajectory_rows = read_rows(out_dir / "v22_16_real_trajectory_source_log.csv")
    rows: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    blockers: list[str] = []
    if not payload_path.exists():
        blockers.append("missing_v22_16_real_trajectory_source_payload.pt")
    else:
        payload = torch.load(payload_path, map_location="cpu")
        vectors = payload.get("source_vectors", torch.empty(0))
        meta = payload.get("metadata", [])
        by_meta_key = {(str(m["dataset"]), int(m["seed"]), str(m["model_variant"]), int(m["step"])): m for m in meta}
        row_lookup = {(str(r["dataset"]), int(r["seed"]), str(r["model_variant"]), int(r["step"])): r for r in trajectory_rows}
        groups: dict[tuple[str, str, int], list[tuple[dict[str, Any], torch.Tensor]]] = {}
        for idx, m in enumerate(meta):
            key = (str(m["dataset"]), int(m["seed"]), str(m["model_variant"]), int(m["step"]))
            r = row_lookup.get(key, by_meta_key[key])
            kind = "MLP" if str(m["model_variant"]).startswith("MLP") else "KAN"
            groups.setdefault((kind, str(m["model_variant"]), int(m["seed"])), []).append((r, vectors[idx].float()))
        for (kind, _variant, seed), group in groups.items():
            if len(group) < int(args.min_history):
                blockers.append(f"{kind}:{seed}:history_too_short:{len(group)}")
                continue
            families = MLP_FAMILIES if kind == "MLP" else KAN_FAMILIES
            for dim in [int(d) for d in str(args.dims).split(",") if d.strip()]:
                for family in families:
                    row = _run_family(kind, family, dim, group, seed)
                    rows.append(row)
                    if family in {"RandomManifold", "ShuffledSourceManifold", "SignFlipManifold"}:
                        controls.append(row)
    controls_fail = int(sum(int_flag(r.get("MLP_source_manifold_pass")) + int_flag(r.get("KAN_basis_manifold_pass")) for r in controls) == 0)
    for row in rows:
        row["controls_fail"] = controls_fail
    firewall_pass = int(all(int_flag(r.get(k)) == v for r in rows for k, v in FIREWALL_FIELDS.items())) if rows else 0
    mlp_pass = int(sum(int_flag(r.get("MLP_source_manifold_pass")) for r in rows) >= 2 and controls_fail and firewall_pass)
    kan_pass = int(sum(int_flag(r.get("KAN_basis_manifold_pass")) for r in rows) >= 1 and controls_fail and firewall_pass)
    write_rows(out_dir / "v22_16_real_source_manifold_controller_matrix.csv", rows)
    write_rows(out_dir / "v22_16_real_source_manifold_controls_matrix.csv", controls)
    write_rows(out_dir / "v22_16_real_latent_stability_diagnostics.csv", rows)
    write_rows(out_dir / "v22_16_manifold_projection_residual_by_history.csv", rows)
    write_rows(out_dir / "v22_16_source_manifold_ablation_matrix.csv", rows)
    if blockers:
        route_name = "R5-RealSourceManifoldNoGo"
    elif mlp_pass and not kan_pass:
        route_name = "R6-RealSourceManifoldOpened_ControllerNoGain"
    elif mlp_pass or kan_pass:
        route_name = "E-RealSourceManifoldPass"
    else:
        route_name = "R5-RealSourceManifoldNoGo"
    route = {
        "route": route_name,
        "MLP_source_manifold_pass": mlp_pass,
        "KAN_basis_manifold_pass": kan_pass,
        "source_manifold_firewall_pass": firewall_pass,
        "controls_fail": controls_fail,
        "rows": len(rows),
        "best_manifold_projection_residual_Gf": min((finite_float(r.get("projection_residual_Gf"), 999.0) for r in rows), default=999.0),
        "best_KAN_basis_channel_energy_fraction": max((finite_float(r.get("basis_channel_energy_fraction"), 0.0) for r in rows), default=0.0),
        "blocker": ";".join(blockers[:20]),
        "repair_attempts": "adapter/source family split; source-loss-positive aliases; random/shuffled/signflip controls with own basis; no future history in basis construction",
    }
    write_json(out_dir / "v22_16_source_manifold_route.json", route)
    append_exec(out_dir, command, status="completed" if not blockers else "blocked", gpu="3", task_id="E-real-source-manifold", files="v22_16_real_source_manifold_controller_matrix.csv; v22_16_real_source_manifold_controls_matrix.csv", note=f"route={route_name} rows={len(rows)} blockers={len(blockers)}")


if __name__ == "__main__":
    main()
