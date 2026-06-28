"""Metric-weighted quotient-horizontal projector for KAN edge coordinates."""

from __future__ import annotations

import torch


def horizontal_projector(metric: torch.Tensor, vertical_basis: torch.Tensor, ridge: float = 1.0e-8) -> tuple[torch.Tensor, dict[str, float]]:
    g = 0.5 * (metric.detach().to(dtype=torch.float64) + metric.detach().to(dtype=torch.float64).transpose(0, 1))
    n = int(g.shape[0])
    eye = torch.eye(n, dtype=g.dtype, device=g.device)
    if int(vertical_basis.numel()) == 0:
        return eye, {"vertical_basis_condition": 0.0, "projector_idempotence_error": 0.0}
    v = vertical_basis.detach().to(dtype=torch.float64)
    gram_v = v.transpose(0, 1) @ g @ v
    gram_v = 0.5 * (gram_v + gram_v.transpose(0, 1))
    gram_v = gram_v + float(ridge) * torch.eye(int(gram_v.shape[0]), dtype=g.dtype, device=g.device)
    evals = torch.linalg.eigvalsh(gram_v)
    cond = float((evals.max() / evals.min().clamp_min(1.0e-12)).detach().cpu().item())
    solve = torch.linalg.solve(gram_v, v.transpose(0, 1) @ g)
    p = eye - v @ solve
    idem = float(torch.linalg.norm(p @ p - p).detach().cpu().item())
    return p, {"vertical_basis_condition": cond, "projector_idempotence_error": idem}


def project_horizontal(vec: torch.Tensor, metric: torch.Tensor, vertical_basis: torch.Tensor, ridge: float = 1.0e-8) -> tuple[torch.Tensor, dict[str, float]]:
    p, diag = horizontal_projector(metric, vertical_basis, ridge=ridge)
    v = vec.detach().to(dtype=torch.float64).reshape(-1, 1)
    out = p @ v
    orth = vertical_basis.detach().to(dtype=torch.float64).transpose(0, 1) @ metric.detach().to(dtype=torch.float64) @ out
    diag = dict(diag)
    diag["horizontal_orthogonality_error"] = float(torch.linalg.norm(orth).detach().cpu().item())
    return out.reshape(-1), diag


def quotient_projector_unit_case(seed: int = 0, dim: int = 8) -> dict[str, float]:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    raw = torch.randn(dim, dim, generator=gen, dtype=torch.float64)
    metric = raw.transpose(0, 1) @ raw + 0.5 * torch.eye(dim, dtype=torch.float64)
    vertical = torch.zeros(dim, 2, dtype=torch.float64)
    vertical[0, 0] = 1.0
    vertical[1, 1] = 1.0
    gauge = vertical[:, 0]
    obs = torch.zeros(dim, dtype=torch.float64)
    obs[3] = 1.0
    gauge_proj, d0 = project_horizontal(gauge, metric, vertical)
    obs_proj, d1 = project_horizontal(obs, metric, vertical)
    p, pd = horizontal_projector(metric, vertical)
    return {
        "pure_gauge_projection_residual": float(torch.linalg.norm(gauge_proj).detach().cpu().item() / torch.linalg.norm(gauge).clamp_min(1.0e-12).detach().cpu().item()),
        "observable_projection_retention": float(torch.linalg.norm(obs_proj).detach().cpu().item() / torch.linalg.norm(obs).clamp_min(1.0e-12).detach().cpu().item()),
        "horizontal_orthogonality_error": max(float(d0["horizontal_orthogonality_error"]), float(d1["horizontal_orthogonality_error"])),
        "projector_idempotence_error": float(pd["projector_idempotence_error"]),
        "vertical_basis_condition": float(pd["vertical_basis_condition"]),
    }
