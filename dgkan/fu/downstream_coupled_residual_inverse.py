"""Downstream-coupled edge residual inverse helpers for DG-KAN v23.08.

The functions here operate only on existing PureKAN edge coefficients.  They do
not add model structure; they build linearized operators around a frozen model
state so experiment runners can solve residual inverse problems in output space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def sym(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.T)


def build_edge_design_matrix_or_operator(model: Any, activation: torch.Tensor, layer_id: int) -> torch.Tensor:
    return model.layer_phi(activation, int(layer_id)).to(dtype=torch.float64)


def apply_edge_metric(edge_metric: torch.Tensor, coeff_delta: torch.Tensor) -> torch.Tensor:
    return edge_metric.to(device=coeff_delta.device, dtype=torch.float64) @ coeff_delta.to(dtype=torch.float64)


def whiten_edge_coeff(edge_metric: torch.Tensor, coeff: torch.Tensor, jitter: float = 1.0e-10) -> tuple[torch.Tensor, torch.Tensor]:
    metric = sym(edge_metric.to(device=coeff.device, dtype=torch.float64))
    eye = torch.eye(int(metric.shape[0]), device=metric.device, dtype=torch.float64)
    chol = torch.linalg.cholesky(metric + float(jitter) * eye)
    return chol.T @ coeff.to(dtype=torch.float64), chol


def unwhiten_edge_coeff(cholesky_factor: torch.Tensor, white_coeff: torch.Tensor) -> torch.Tensor:
    return torch.linalg.solve(cholesky_factor.T, white_coeff.to(dtype=torch.float64))


def solve_ridge_normal_exact(normal: torch.Tensor, rhs: torch.Tensor, jitter: float = 1.0e-10) -> tuple[torch.Tensor, dict[str, float]]:
    mat = sym(normal.to(dtype=torch.float64))
    b = rhs.to(device=mat.device, dtype=torch.float64)
    eye = torch.eye(int(mat.shape[0]), device=mat.device, dtype=torch.float64)
    used = 0.0
    for attempt in range(8):
        try:
            chol = torch.linalg.cholesky(mat + used * eye)
            sol = torch.cholesky_solve(b, chol)
            resid = ((mat + used * eye) @ sol - b).norm() / b.norm().clamp_min(1.0e-12)
            return sol, {"solve_residual": float(resid.detach().cpu().item()), "jitter_used": used, "solve_attempts": float(attempt + 1)}
        except RuntimeError:
            used = float(jitter) if used == 0.0 else used * 10.0
    sol = torch.linalg.solve(mat + used * eye, b)
    resid = ((mat + used * eye) @ sol - b).norm() / b.norm().clamp_min(1.0e-12)
    return sol, {"solve_residual": float(resid.detach().cpu().item()), "jitter_used": used, "solve_attempts": 8.0}


def solve_ridge_normal_cg(normal: torch.Tensor, rhs: torch.Tensor, tol: float = 1.0e-6, max_iter: int = 512) -> tuple[torch.Tensor, dict[str, float]]:
    mat = sym(normal.to(dtype=torch.float64))
    b = rhs.to(device=mat.device, dtype=torch.float64)
    x = torch.zeros_like(b)
    r = b - mat @ x
    p = r.clone()
    rsold = (r * r).sum(dim=0).clamp_min(1.0e-30)
    rhs_norm = b.norm(dim=0).clamp_min(1.0e-12)
    it = 0
    for it in range(1, int(max_iter) + 1):
        ap = mat @ p
        alpha = rsold / (p * ap).sum(dim=0).clamp_min(1.0e-30)
        x = x + p * alpha.reshape(1, -1)
        r = r - ap * alpha.reshape(1, -1)
        rsnew = (r * r).sum(dim=0).clamp_min(1.0e-30)
        rel = (rsnew.sqrt() / rhs_norm).max()
        if float(rel.detach().cpu().item()) <= float(tol):
            rsold = rsnew
            break
        beta = rsnew / rsold
        p = r + p * beta.reshape(1, -1)
        rsold = rsnew
    resid = (mat @ x - b).norm() / b.norm().clamp_min(1.0e-12)
    return x, {"cg_iterations": float(it), "cg_residual": float(resid.detach().cpu().item())}


def downstream_forward_from_activation(model: Any, activation: torch.Tensor, start_layer_id: int) -> torch.Tensor:
    h = activation
    for layer_id in range(int(start_layer_id), len(model.coeffs)):
        coeff = model.coeffs[int(layer_id)]
        basis = model.basis(h) / (float(max(1, int(model.dims[int(layer_id)]))) ** 0.5)
        h = torch.einsum("bik,iok->bo", basis, coeff)
    return h


def downstream_jacobian(model: Any, activation_after_layer: torch.Tensor, start_layer_id: int) -> torch.Tensor:
    h = activation_after_layer.detach().clone().requires_grad_(True)
    out = downstream_forward_from_activation(model, h, int(start_layer_id))
    grads = []
    for out_idx in range(int(out.shape[1])):
        grad = torch.autograd.grad(out[:, out_idx].sum(), h, retain_graph=True, create_graph=False)[0]
        grads.append(grad.detach().to(dtype=torch.float64))
    return torch.stack(grads, dim=1)


@dataclass
class DownstreamOperator:
    layer_id: int
    phi: torch.Tensor
    downstream_jac: torch.Tensor

    @property
    def sample_count(self) -> int:
        return int(self.phi.shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.downstream_jac.shape[1])

    @property
    def layer_output_dim(self) -> int:
        return int(self.downstream_jac.shape[2])

    @property
    def basis_dim(self) -> int:
        return int(self.phi.shape[1])

    @property
    def coeff_dim(self) -> int:
        return self.basis_dim * self.layer_output_dim

    def explicit_matrix(self) -> torch.Tensor:
        n, c, q = int(self.sample_count), int(self.output_dim), int(self.layer_output_dim)
        p = int(self.basis_dim)
        h = torch.zeros((n * c, p * q), device=self.phi.device, dtype=torch.float64)
        row = torch.arange(n, device=self.phi.device)
        for out_idx in range(c):
            rows = row * c + out_idx
            for hidden_idx in range(q):
                cols = torch.arange(p, device=self.phi.device) * q + hidden_idx
                h[rows[:, None], cols[None, :]] = self.phi * self.downstream_jac[:, out_idx, hidden_idx].reshape(-1, 1)
        return h

    def apply_H(self, coeff_delta: torch.Tensor) -> torch.Tensor:
        delta = coeff_delta.to(device=self.phi.device, dtype=torch.float64).reshape(self.basis_dim, self.layer_output_dim)
        hidden_move = self.phi @ delta
        out = torch.einsum("nch,nh->nc", self.downstream_jac, hidden_move)
        return out

    def apply_HT(self, output_vector: torch.Tensor) -> torch.Tensor:
        y = output_vector.to(device=self.phi.device, dtype=torch.float64).reshape(self.sample_count, self.output_dim)
        hidden_adj = torch.einsum("nch,nc->nh", self.downstream_jac, y)
        return self.phi.T @ hidden_adj


def make_downstream_operator(model: Any, activations: list[torch.Tensor], layer_id: int) -> DownstreamOperator:
    layer = int(layer_id)
    phi = build_edge_design_matrix_or_operator(model, activations[layer], layer)
    jac = downstream_jacobian(model, activations[layer + 1], layer + 1)
    return DownstreamOperator(layer, phi, jac)


def expanded_edge_metric(edge_metric: torch.Tensor, layer_output_dim: int) -> torch.Tensor:
    eye = torch.eye(int(layer_output_dim), device=edge_metric.device, dtype=torch.float64)
    return torch.kron(edge_metric.to(dtype=torch.float64), eye)


def solve_downstream_ridge_cg(
    operator: DownstreamOperator,
    output_residual: torch.Tensor,
    edge_metric: torch.Tensor,
    lam: float,
    solver: str = "exact",
    tol: float = 1.0e-6,
    max_iter: int = 512,
) -> tuple[torch.Tensor, dict[str, float]]:
    h = operator.explicit_matrix()
    r = output_residual.to(device=h.device, dtype=torch.float64).reshape(-1, 1)
    metric = expanded_edge_metric(edge_metric.to(device=h.device, dtype=torch.float64), operator.layer_output_dim)
    normal = h.T @ h + float(lam) * metric
    rhs = h.T @ r
    if str(solver).lower() == "cg":
        flat, diag = solve_ridge_normal_cg(normal, rhs, tol=tol, max_iter=max_iter)
        diag["solve_residual"] = diag.get("cg_residual", 0.0)
    else:
        flat, diag = solve_ridge_normal_exact(normal, rhs)
        diag["cg_iterations"] = 0.0
    vals = torch.linalg.eigvalsh(sym(normal))
    finite_vals = vals[torch.isfinite(vals)]
    mn = float(finite_vals.min().detach().cpu().item()) if int(finite_vals.numel()) else 0.0
    mx = float(finite_vals.max().detach().cpu().item()) if int(finite_vals.numel()) else 0.0
    diag.update(
        {
            "condition_number_after_ridge": float(mx / max(mn, 1.0e-12)) if mn > 0 else float("inf"),
            "rank_H": float(torch.linalg.matrix_rank(h, tol=1.0e-8).detach().cpu().item()),
            "H_rows": float(h.shape[0]),
            "H_cols": float(h.shape[1]),
        }
    )
    return flat.reshape(operator.basis_dim, operator.layer_output_dim), diag


def sketch_downstream_operator(operator: DownstreamOperator, sketch_rank: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    h = operator.explicit_matrix()
    gen = torch.Generator(device=h.device).manual_seed(int(seed))
    q, _ = torch.linalg.qr(torch.randn((int(h.shape[0]), int(sketch_rank)), device=h.device, dtype=torch.float64, generator=gen))
    return q.T @ h, q


class LastLayerResidualFlow:
    pass


class DownstreamCoupledResidualFlow:
    pass


class TopDownResidualFlow:
    pass


class HybridResidualFlow:
    pass
