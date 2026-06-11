"""Source-manifold coordinate controller for v22.15."""

from __future__ import annotations

from typing import Any

import torch

from dgkan.fu.adaptive_controller import cosine
from dgkan.fu.source_manifold_basis import FIREWALL_FIELDS


EPS = 1.0e-12


def manifold_coordinate_solve(
    manifold_basis: torch.Tensor,
    jacobian: torch.Tensor,
    base_update: torch.Tensor,
    source: torch.Tensor,
    *,
    lambda_t: float,
    previous_coordinates: torch.Tensor | None = None,
    coordinate_damping: float = 1.0e-3,
    stability_weight: float = 1.0e-2,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    b = manifold_basis.detach().float()
    j = jacobian.detach().float()
    u = base_update.detach().float().reshape(-1).to(b.device)
    z = source.detach().float().reshape(-1).to(b.device)
    if b.ndim != 2:
        raise ValueError("manifold_basis must be a matrix")
    if j.shape[1] != b.shape[0]:
        raise ValueError("jacobian columns must match manifold basis rows")
    jb = j.to(b.device) @ b
    lhs = b.T @ b + float(lambda_t) * (jb.T @ jb)
    rhs = b.T @ u + float(lambda_t) * (jb.T @ z)
    if previous_coordinates is not None:
        prev = previous_coordinates.detach().float().reshape(-1).to(b.device)
        lhs = lhs + float(stability_weight) * torch.eye(b.shape[1], device=b.device, dtype=b.dtype)
        rhs = rhs + float(stability_weight) * prev
    lhs = lhs + float(coordinate_damping) * torch.eye(b.shape[1], device=b.device, dtype=b.dtype)
    status = "solve"
    try:
        coord = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        coord = torch.linalg.lstsq(lhs, rhs.unsqueeze(1)).solution.squeeze(1)
        status = "lstsq"
    update = b @ coord
    effect = j.to(update.device) @ update
    full_effect = j.to(u.device) @ u
    residual = torch.linalg.vector_norm(effect - z).div(torch.linalg.vector_norm(z).clamp_min(EPS)).item()
    full_residual = torch.linalg.vector_norm(full_effect - z.to(full_effect.device)).div(torch.linalg.vector_norm(z).clamp_min(EPS)).item()
    drift = 0.0
    if previous_coordinates is not None:
        drift = float(torch.linalg.vector_norm(coord - previous_coordinates.detach().float().to(coord.device)).item())
    gram = b.T @ b
    try:
        eig = torch.linalg.eigvalsh(gram)
        cond = float((eig.max() / eig.clamp_min(EPS).min()).item())
    except RuntimeError:
        cond = float("nan")
    diag = {
        **FIREWALL_FIELDS,
        "manifold_projection_residual_Gf": float(residual),
        "full_readout_prox_residual": float(full_residual),
        "manifold_projection_cosine": cosine(effect, z.to(effect.device)),
        "manifold_ActuationR2": float(max(0.0, 1.0 - residual * residual)),
        "manifold_condition_number": cond,
        "latent_coordinate_norm": float(torch.linalg.vector_norm(coord).item()),
        "latent_coordinate_drift": drift,
        "latent_stability_risk": float(min(1.0, drift / max(1.0, torch.linalg.vector_norm(coord).item()))),
        "latent_smoothness_energy": float((coord[1:] - coord[:-1]).square().mean().item()) if coord.numel() > 1 else 0.0,
        "latent_alignment_with_source_state": cosine(effect, z.to(effect.device)),
        "source_manifold_coordinate_solve_status": status,
    }
    return update, coord, diag


def source_manifold_firewall_row() -> dict[str, Any]:
    return {**FIREWALL_FIELDS, "source_manifold_contract_pass": 1}


__all__ = ["manifold_coordinate_solve", "source_manifold_firewall_row"]
