"""Basis-native adaptive source solve helpers for strict KAN carrier audits."""

from __future__ import annotations

from typing import Any

import torch

from dgkan.fu.adaptive_controller import cosine, retention_score


EPS = 1.0e-12


def basis_native_solve(
    basis_jacobian: torch.Tensor,
    cotangent: torch.Tensor,
    source: torch.Tensor,
    *,
    lambda_t: float,
    damping: float = 1.0e-3,
    pair_incidence: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    jb = basis_jacobian.detach().float()
    c = cotangent.detach().float().reshape(-1).to(jb.device)
    z = source.detach().float().reshape(-1).to(jb.device)
    if pair_incidence is not None:
        b = pair_incidence.detach().float().to(jb.device)
        lhs = jb.T @ jb + float(lambda_t) * (jb.T @ b.T @ b @ jb)
        rhs = -(jb.T @ c) + float(lambda_t) * (jb.T @ b.T @ b @ z)
    else:
        lhs = jb.T @ jb + float(lambda_t) * (jb.T @ jb)
        rhs = -(jb.T @ c) + float(lambda_t) * (jb.T @ z)
    lhs = lhs + float(damping) * torch.eye(jb.shape[1], device=jb.device, dtype=jb.dtype)
    status = "solve"
    try:
        coeff = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        coeff = torch.linalg.lstsq(lhs, rhs.unsqueeze(1)).solution.squeeze(1)
        status = "lstsq"
    effect = jb @ coeff
    residual = torch.linalg.vector_norm(effect - z).div(torch.linalg.vector_norm(z).clamp_min(EPS)).item()
    gram = jb.T @ jb
    try:
        eig = torch.linalg.eigvalsh(gram)
        cond = float((eig.max() / eig.clamp_min(EPS).min()).item())
    except RuntimeError:
        cond = float("nan")
    return coeff, {
        "basis_operator_residual": float(residual),
        "basis_projection_cosine": retention_score(effect, z),
        "basis_condition_number": cond,
        "basis_update_norm": float(torch.linalg.vector_norm(coeff).item()),
        "basis_native_solve_status": status,
    }


def channel_energy_fraction(update: torch.Tensor, basis_mask: torch.Tensor) -> tuple[float, float]:
    u = update.detach().float().reshape(-1)
    m = basis_mask.detach().bool().reshape(-1).to(u.device)
    if m.numel() != u.numel():
        raise ValueError("basis_mask must match update length")
    basis = u[m].square().sum()
    readout = u[~m].square().sum()
    total = (basis + readout).clamp_min(EPS)
    return float((basis / total).item()), float((readout / total).item())


def bank_coverage_rows(
    basis_jacobians: dict[str, torch.Tensor],
    source: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    z = source.detach().float().reshape(-1)
    for bank, jb in basis_jacobians.items():
        j = jb.detach().float()
        gram = j @ j.T
        damping = 1.0e-3 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        try:
            proj = gram @ torch.linalg.solve(gram + damping, z.to(gram.device))
        except RuntimeError:
            proj = gram @ torch.linalg.lstsq(gram + damping, z.to(gram.device).unsqueeze(1)).solution.squeeze(1)
        residual = torch.linalg.vector_norm(proj - z.to(proj.device)).div(torch.linalg.vector_norm(z).clamp_min(EPS)).item()
        rows.append(
            {
                "basis_bank": bank,
                "basis_projection_residual": float(residual),
                "basis_projection_cosine": cosine(proj, z.to(proj.device)),
                "basis_tangent_rank": int(torch.linalg.matrix_rank(j).item()),
            }
        )
    return rows


__all__ = ["bank_coverage_rows", "basis_native_solve", "channel_energy_fraction"]
