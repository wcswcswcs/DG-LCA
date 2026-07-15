"""Basis-covariant edge-function residual flow utilities for v23.15."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from dgkan.fu.covariant_edge_preconditioner import (
    exact_residual_inverse,
    make_block_preconditioner,
    normal_system,
    pcg,
    pcg_residual_inverse,
)
from dgkan.fu.edge_basis_chart import expanded_metric


@dataclass
class FlowResult:
    delta: torch.Tensor
    diagnostics: dict[str, float | str]


def local_exact_flow(
    phi: torch.Tensor,
    residual: torch.Tensor,
    edge_metric: torch.Tensor,
    *,
    in_dim: int,
    lam: float,
) -> FlowResult:
    metric = expanded_metric(edge_metric.to(device=phi.device, dtype=torch.float64), int(in_dim))
    delta, diag = exact_residual_inverse(phi, residual, metric, float(lam))
    return FlowResult(delta=delta, diagnostics=diag)


def local_bc15_pcg_flow(
    phi: torch.Tensor,
    residual: torch.Tensor,
    edge_metric: torch.Tensor,
    *,
    in_dim: int,
    k: int,
    lam: float,
    rho: float,
    max_iter: int = 4,
    tol: float = 1.0e-8,
    collect_preconditioner_diagnostics: bool = True,
) -> FlowResult:
    metric = expanded_metric(edge_metric.to(device=phi.device, dtype=torch.float64), int(in_dim))
    precond, pdiag = make_block_preconditioner(
        phi,
        edge_metric.to(device=phi.device, dtype=torch.float64),
        in_dim=int(in_dim),
        k=int(k),
        rho=float(rho),
        collect_diagnostics=bool(collect_preconditioner_diagnostics),
    )
    delta, diag = pcg_residual_inverse(phi, residual, metric, float(lam), preconditioner=precond, max_iter=int(max_iter), tol=float(tol))
    diag.update(pdiag)
    diag["solver_type"] = f"bc15_pcg{int(max_iter)}"
    return FlowResult(delta=delta, diagnostics=diag)


def local_unpreconditioned_pcg_flow(
    phi: torch.Tensor,
    residual: torch.Tensor,
    edge_metric: torch.Tensor,
    *,
    in_dim: int,
    lam: float,
    max_iter: int = 4,
    tol: float = 1.0e-8,
) -> FlowResult:
    metric = expanded_metric(edge_metric.to(device=phi.device, dtype=torch.float64), int(in_dim))
    delta, diag = pcg_residual_inverse(phi, residual, metric, float(lam), preconditioner=None, max_iter=int(max_iter), tol=float(tol))
    diag["solver_type"] = f"c15_unpreconditioned_pcg{int(max_iter)}"
    return FlowResult(delta=delta, diagnostics=diag)


def local_fixed_identity_ridge_flow(
    phi: torch.Tensor,
    residual: torch.Tensor,
    *,
    lam: float,
    max_iter: int | None = None,
) -> FlowResult:
    metric = torch.eye(int(phi.shape[1]), device=phi.device, dtype=torch.float64)
    if max_iter is None:
        delta, diag = exact_residual_inverse(phi, residual, metric, float(lam))
        diag["solver_type"] = "fixed_identity_ridge_exact"
    else:
        normal, rhs = normal_system(phi, residual, metric, float(lam))
        delta, diag = pcg(normal, rhs, max_iter=int(max_iter), tol=1.0e-8, preconditioner=None)
        diag["solver_type"] = f"fixed_identity_ridge_pcg{int(max_iter)}"
    return FlowResult(delta=delta, diagnostics=diag)


def apply_layer_delta(model: Any, layer_id: int, delta: torch.Tensor, alpha: float = 1.0) -> None:
    with torch.no_grad():
        coeff = model.coeffs[int(layer_id)]
        update = delta.to(device=coeff.device, dtype=coeff.dtype).reshape(int(coeff.shape[0]), int(coeff.shape[2]), int(coeff.shape[1])).permute(0, 2, 1)
        coeff.add_(float(alpha) * update)
