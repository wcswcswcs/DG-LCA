"""Covariant edge normal-equation and PCG helpers for v23.15."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


EPS = 1.0e-12


def sym(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.T)


def eig_condition(mat: torch.Tensor) -> float:
    vals = torch.linalg.eigvalsh(sym(mat.to(dtype=torch.float64)))
    vals = vals[torch.isfinite(vals)]
    if int(vals.numel()) == 0:
        return float("inf")
    return float((vals.max().clamp_min(EPS) / vals.min().clamp_min(EPS)).detach().cpu().item())


def solve_spd(mat: torch.Tensor, rhs: torch.Tensor, jitter: float = 1.0e-10) -> tuple[torch.Tensor, dict[str, float]]:
    a = sym(mat.to(dtype=torch.float64))
    b = rhs.to(device=a.device, dtype=torch.float64)
    eye = torch.eye(int(a.shape[0]), device=a.device, dtype=torch.float64)
    used = 0.0
    for attempt in range(8):
        try:
            chol = torch.linalg.cholesky(a + used * eye)
            sol = torch.cholesky_solve(b, chol)
            resid = (a @ sol - b).norm().div(b.norm().clamp_min(EPS))
            return sol, {"solve_residual": float(resid.detach().cpu().item()), "jitter_used": used, "solve_attempts": float(attempt + 1)}
        except RuntimeError:
            used = float(jitter) if used == 0.0 else used * 10.0
    sol = torch.linalg.solve(a + used * eye, b)
    resid = (a @ sol - b).norm().div(b.norm().clamp_min(EPS))
    return sol, {"solve_residual": float(resid.detach().cpu().item()), "jitter_used": used, "solve_attempts": 8.0}


def normal_system(phi: torch.Tensor, residual: torch.Tensor, metric: torch.Tensor, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    p = phi.to(dtype=torch.float64)
    r = residual.to(device=p.device, dtype=torch.float64)
    n = float(max(1, int(p.shape[0])))
    normal = p.T @ p / n + float(lam) * metric.to(device=p.device, dtype=torch.float64)
    rhs = p.T @ r / n
    return sym(normal), rhs


def exact_residual_inverse(phi: torch.Tensor, residual: torch.Tensor, metric: torch.Tensor, lam: float) -> tuple[torch.Tensor, dict[str, float]]:
    normal, rhs = normal_system(phi, residual, metric, lam)
    sol, diag = solve_spd(normal, rhs)
    diag.update(
        {
            "condition_number": eig_condition(normal),
            "solver_type": "exact_dense",
            "solver_actual_iter": 0.0,
            "solver_residual": diag["solve_residual"],
        }
    )
    return sol, diag


@dataclass
class BlockPreconditioner:
    factors: list[torch.Tensor]
    block_size: int
    repeats_per_input: int = 1
    inverse_factors: list[torch.Tensor] | None = None

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        xx = x.to(dtype=torch.float64)
        if xx.ndim == 1:
            xx = xx.reshape(-1, 1)
        out = torch.empty_like(xx)
        k = int(self.block_size)
        block_count = int(xx.shape[0]) // k
        if block_count <= 0:
            return out.reshape_as(x.to(dtype=torch.float64))
        rhs = xx[: block_count * k].reshape(block_count, k, int(xx.shape[1]))
        if len(self.factors) == 1:
            packed = rhs.permute(1, 0, 2).reshape(k, block_count * int(xx.shape[1]))
            if self.inverse_factors:
                inv = self.inverse_factors[0].to(device=xx.device, dtype=torch.float64)
                solved = (inv @ packed).reshape(k, block_count, int(xx.shape[1])).permute(1, 0, 2)
            else:
                chol = self.factors[0].to(device=xx.device, dtype=torch.float64)
                solved = torch.cholesky_solve(packed, chol).reshape(k, block_count, int(xx.shape[1])).permute(1, 0, 2)
        else:
            if self.inverse_factors:
                invs = torch.stack([c.to(device=xx.device, dtype=torch.float64) for c in self.inverse_factors[:block_count]], dim=0)
                solved = torch.bmm(invs, rhs)
            else:
                chols = torch.stack([c.to(device=xx.device, dtype=torch.float64) for c in self.factors[:block_count]], dim=0)
                solved = torch.cholesky_solve(rhs, chols)
        out[: block_count * k] = solved.reshape(block_count * k, int(xx.shape[1]))
        if block_count * k < int(xx.shape[0]):
            out[block_count * k :] = xx[block_count * k :]
        return out.reshape_as(x.to(dtype=torch.float64))


def source_gram_blocks(phi: torch.Tensor, in_dim: int, k: int) -> list[torch.Tensor]:
    p = phi.to(dtype=torch.float64)
    out: list[torch.Tensor] = []
    kk = int(k)
    for in_idx in range(int(in_dim)):
        block = p[:, in_idx * kk : (in_idx + 1) * kk]
        out.append(sym(block.T @ block / float(max(1, int(block.shape[0])))))
    return out


def make_block_preconditioner(
    phi: torch.Tensor,
    G: torch.Tensor,
    *,
    in_dim: int,
    k: int,
    rho: float,
    jitter: float = 1.0e-8,
    aggregate_blocks: bool = True,
    collect_diagnostics: bool = True,
) -> tuple[BlockPreconditioner, dict[str, float]]:
    source_blocks = source_gram_blocks(phi, int(in_dim), int(k))
    if bool(aggregate_blocks) and source_blocks:
        blocks = [sym(torch.stack(source_blocks, dim=0).mean(dim=0))]
    else:
        blocks = source_blocks
    factors: list[torch.Tensor] = []
    inverse_factors: list[torch.Tensor] = []
    cond_before: list[float] = []
    cond_after: list[float] = []
    g = G.to(device=phi.device, dtype=torch.float64)
    eye = torch.eye(int(k), device=phi.device, dtype=torch.float64)
    for C in blocks:
        mat = sym(C + float(rho) * g)
        used = float(jitter)
        for _ in range(8):
            try:
                chol = torch.linalg.cholesky(mat + used * eye)
                break
            except RuntimeError:
                used *= 10.0
        else:
            chol = torch.linalg.cholesky(mat + used * eye)
        factors.append(chol)
        inverse_factors.append(torch.cholesky_inverse(chol))
        if bool(collect_diagnostics):
            cond_before.append(eig_condition(C + 1.0e-8 * g))
            inv = inverse_factors[-1]
            cond_after.append(eig_condition(inv @ (C + g) @ inv.T))
    diag = {
        "factor_refresh_count": float(len(factors)),
        "factor_reuse_count": float(max(0, int(in_dim) - len(factors))),
    }
    if bool(collect_diagnostics):
        diag.update(
            {
                "source_block_condition_median": float(torch.tensor([eig_condition(C + 1.0e-8 * g) for C in source_blocks]).median().item()) if source_blocks else 0.0,
                "aggregated_block_condition": float(torch.tensor(cond_before).median().item()) if cond_before else 0.0,
                "preconditioned_block_condition_proxy_median": float(torch.tensor(cond_after).median().item()) if cond_after else 0.0,
            }
        )
    return BlockPreconditioner(factors=factors, inverse_factors=inverse_factors, block_size=int(k)), diag


def pcg(
    mat: torch.Tensor,
    rhs: torch.Tensor,
    *,
    max_iter: int,
    tol: float = 1.0e-8,
    preconditioner: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    a = sym(mat.to(dtype=torch.float64))
    b = rhs.to(device=a.device, dtype=torch.float64)
    x = torch.zeros_like(b)
    r = b - a @ x
    z = preconditioner(r) if preconditioner is not None else r.clone()
    p = z.clone()
    rz_old = (r * z).sum(dim=0).clamp_min(EPS)
    rhs_norm = b.norm(dim=0).clamp_min(EPS)
    it = 0
    rel = torch.full((int(b.shape[1]),), float("inf"), device=a.device, dtype=torch.float64)
    for it in range(1, int(max_iter) + 1):
        ap = a @ p
        alpha = rz_old / (p * ap).sum(dim=0).clamp_min(EPS)
        x = x + p * alpha.reshape(1, -1)
        r = r - ap * alpha.reshape(1, -1)
        rel = r.norm(dim=0) / rhs_norm
        if float(rel.max().detach().cpu().item()) <= float(tol):
            break
        z = preconditioner(r) if preconditioner is not None else r.clone()
        rz_new = (r * z).sum(dim=0).clamp_min(EPS)
        beta = rz_new / rz_old
        p = z + p * beta.reshape(1, -1)
        rz_old = rz_new
    resid = (a @ x - b).norm().div(b.norm().clamp_min(EPS))
    return x, {"solver_actual_iter": float(it), "solver_residual": float(resid.detach().cpu().item()), "pcg_relative_residual_max": float(rel.max().detach().cpu().item())}


def pcg_residual_inverse(
    phi: torch.Tensor,
    residual: torch.Tensor,
    metric: torch.Tensor,
    lam: float,
    *,
    preconditioner: BlockPreconditioner | None = None,
    max_iter: int = 4,
    tol: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    normal, rhs = normal_system(phi, residual, metric, lam)
    sol, diag = pcg(normal, rhs, max_iter=int(max_iter), tol=float(tol), preconditioner=(preconditioner.apply if preconditioner else None))
    diag.update({"condition_number": eig_condition(normal), "solver_type": f"pcg{int(max_iter)}"})
    return sol, diag
