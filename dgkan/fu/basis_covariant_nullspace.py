"""Basis-covariant dense nullspace utilities for v23.17.

The routines in this file deliberately work on dense small-model matrices.
They are used for exact unit audits and synthetic mechanism tests before any
large matrix-free implementation is justified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


EPS = 1.0e-12


@dataclass
class NullSolveResult:
    vector: torch.Tensor
    mu: float
    null_ratio: float
    feasible: bool
    solve_residual: float
    jitter_used: float
    bisection_steps: int


def sym(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.T)


def as_weighted_jtj(j: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    jj = j.to(dtype=torch.float64)
    if w is None:
        return jj.T @ jj
    ww = w.to(device=jj.device, dtype=torch.float64)
    if int(ww.ndim) == 1:
        return jj.T @ (jj * ww.reshape(-1, 1))
    return jj.T @ ww @ jj


def g_inner(a: torch.Tensor, b: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    aa = a.reshape(-1, 1).to(device=g.device, dtype=torch.float64)
    bb = b.reshape(-1, 1).to(device=g.device, dtype=torch.float64)
    gg = g.to(dtype=torch.float64)
    return (aa.T @ gg @ bb).reshape(())


def g_norm(a: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return g_inner(a, a, g).clamp_min(0.0).sqrt()


def output_norm(y: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    yy = y.reshape(-1, 1).to(dtype=torch.float64)
    if w is None:
        return yy.square().sum().clamp_min(0.0).sqrt()
    ww = w.to(device=yy.device, dtype=torch.float64)
    if int(ww.ndim) == 1:
        return (yy.reshape(-1).square() * ww.reshape(-1)).sum().clamp_min(0.0).sqrt()
    return (yy.T @ ww @ yy).reshape(()).clamp_min(0.0).sqrt()


def null_ratio(j: torch.Tensor, v: torch.Tensor, g: torch.Tensor, w: torch.Tensor | None = None) -> float:
    vv = v.reshape(-1).to(device=j.device, dtype=torch.float64)
    num = output_norm(j.to(dtype=torch.float64) @ vv.reshape(-1, 1), w)
    den = g_norm(vv, g.to(device=j.device, dtype=torch.float64)).clamp_min(EPS)
    return float((num / den).detach().cpu().item())


def solve_spd(mat: torch.Tensor, rhs: torch.Tensor, *, jitter: float = 1.0e-10, attempts: int = 8) -> tuple[torch.Tensor, dict[str, float]]:
    aa = sym(mat.to(dtype=torch.float64))
    bb = rhs.to(device=aa.device, dtype=torch.float64)
    eye = torch.eye(int(aa.shape[0]), device=aa.device, dtype=aa.dtype)
    used = 0.0
    for attempt in range(int(attempts)):
        try:
            chol = torch.linalg.cholesky(aa + used * eye)
            sol = torch.cholesky_solve(bb, chol)
            resid = float(((aa + used * eye) @ sol - bb).norm().div(bb.norm().clamp_min(EPS)).detach().cpu().item())
            return sol, {"jitter_used": float(used), "solve_residual": resid, "cholesky_success": 1.0, "solve_attempts": float(attempt + 1)}
        except RuntimeError:
            used = float(jitter) if used == 0.0 else used * 10.0
    sol = torch.linalg.solve(aa + used * eye, bb)
    resid = float(((aa + used * eye) @ sol - bb).norm().div(bb.norm().clamp_min(EPS)).detach().cpu().item())
    return sol, {"jitter_used": float(used), "solve_residual": resid, "cholesky_success": 0.0, "solve_attempts": float(attempts)}


def soft_null_solve(
    g: torch.Tensor,
    j: torch.Tensor,
    h: torch.Tensor,
    *,
    mu: float,
    w: torch.Tensor | None = None,
    ridge: float = 1.0e-10,
) -> tuple[torch.Tensor, dict[str, float]]:
    gg = sym(g.to(dtype=torch.float64))
    jj = j.to(device=gg.device, dtype=torch.float64)
    hh = h.reshape(-1, 1).to(device=gg.device, dtype=torch.float64)
    normal = gg + float(mu) * as_weighted_jtj(jj, w).to(device=gg.device, dtype=torch.float64)
    normal = normal + float(ridge) * torch.eye(int(normal.shape[0]), device=normal.device, dtype=normal.dtype)
    sol, diag = solve_spd(normal, hh)
    diag["mu"] = float(mu)
    diag["normal_condition"] = float("inf")
    try:
        vals = torch.linalg.eigvalsh(sym(normal))
        mn = float(vals.min().detach().cpu().item())
        mx = float(vals.max().detach().cpu().item())
        diag["normal_min_eig"] = mn
        diag["normal_max_eig"] = mx
        diag["normal_condition"] = float(mx / max(mn, EPS)) if mn > 0.0 else float("inf")
    except Exception:
        pass
    return sol.reshape(-1), diag


def select_soft_null_mu(
    g: torch.Tensor,
    j: torch.Tensor,
    h: torch.Tensor,
    *,
    tau: float = 0.05,
    mu_min: float = 1.0,
    mu_max: float = 1.0e4,
    bisection_steps: int = 4,
    w: torch.Tensor | None = None,
    ridge: float = 1.0e-10,
) -> NullSolveResult:
    hh = h.reshape(-1).to(device=g.device, dtype=torch.float64)
    if float(hh.norm().detach().cpu().item()) <= EPS:
        return NullSolveResult(
            vector=torch.zeros_like(hh),
            mu=float(mu_min),
            null_ratio=0.0,
            feasible=True,
            solve_residual=0.0,
            jitter_used=0.0,
            bisection_steps=int(bisection_steps),
        )
    hi_vec, hi_diag = soft_null_solve(g, j, hh, mu=float(mu_max), w=w, ridge=ridge)
    hi_ratio = null_ratio(j, hi_vec, g, w)
    if hi_ratio > float(tau):
        return NullSolveResult(
            vector=hi_vec,
            mu=float(mu_max),
            null_ratio=float(hi_ratio),
            feasible=False,
            solve_residual=float(hi_diag.get("solve_residual", float("inf"))),
            jitter_used=float(hi_diag.get("jitter_used", 0.0)),
            bisection_steps=int(bisection_steps),
        )

    lo = float(mu_min)
    hi = float(mu_max)
    best_vec = hi_vec
    best_diag = hi_diag
    best_ratio = hi_ratio
    for _ in range(int(bisection_steps)):
        mid = (lo * hi) ** 0.5
        vec, diag = soft_null_solve(g, j, hh, mu=mid, w=w, ridge=ridge)
        ratio = null_ratio(j, vec, g, w)
        if ratio <= float(tau):
            hi = mid
            best_vec, best_diag, best_ratio = vec, diag, ratio
        else:
            lo = mid
    return NullSolveResult(
        vector=best_vec,
        mu=float(hi),
        null_ratio=float(best_ratio),
        feasible=True,
        solve_residual=float(best_diag.get("solve_residual", float("inf"))),
        jitter_used=float(best_diag.get("jitter_used", 0.0)),
        bisection_steps=int(bisection_steps),
    )


def exact_g_null_project(g: torch.Tensor, j: torch.Tensor, v: torch.Tensor, *, rcond: float = 1.0e-10) -> torch.Tensor:
    gg = sym(g.to(dtype=torch.float64))
    jj = j.to(device=gg.device, dtype=torch.float64)
    vv = v.reshape(-1, 1).to(device=gg.device, dtype=torch.float64)
    if int(jj.numel()) == 0 or int(jj.shape[0]) == 0:
        return vv.reshape(-1)
    ginv_jt, _diag = solve_spd(gg, jj.T)
    middle = jj @ ginv_jt
    correction = ginv_jt @ (torch.linalg.pinv(middle, rcond=float(rcond)) @ (jj @ vv))
    return (vv - correction).reshape(-1)


def conjugate_gradient_spd(
    mat: torch.Tensor,
    rhs: torch.Tensor,
    *,
    max_iter: int = 128,
    tol: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    aa = sym(mat.to(dtype=torch.float64)).detach()
    bb = rhs.reshape(-1, 1).to(device=aa.device, dtype=torch.float64).detach()
    x = torch.zeros_like(bb)
    r = bb - aa @ x
    p = r.clone()
    rr_old = (r * r).sum()
    rhs_norm = bb.norm().clamp_min(EPS)
    rel = torch.tensor(float("inf"), device=aa.device, dtype=torch.float64)
    it = 0
    for it in range(1, int(max_iter) + 1):
        ap = aa @ p
        denom = (p * ap).sum()
        if float(denom.detach().cpu().item()) <= EPS:
            break
        alpha = rr_old / denom
        x = x + alpha * p
        r = r - alpha * ap
        rel = r.norm() / rhs_norm
        if float(rel.detach().cpu().item()) <= float(tol):
            break
        rr_new = (r * r).sum()
        if float(rr_new.detach().cpu().item()) <= EPS:
            break
        p = r + (rr_new / rr_old) * p
        rr_old = rr_new
    resid = float((aa @ x - bb).norm().div(rhs_norm).detach().cpu().item())
    return x.reshape(-1), {"cg_iterations": float(it), "cg_residual": resid, "cg_relative_residual": float(rel.detach().cpu().item())}


def dense_pcg_soft_null(
    g: torch.Tensor,
    j: torch.Tensor,
    h: torch.Tensor,
    *,
    mu: float,
    w: torch.Tensor | None = None,
    ridge: float = 1.0e-10,
    max_iter: int = 256,
    tol: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    normal = sym(g.to(dtype=torch.float64)) + float(mu) * as_weighted_jtj(j, w).to(device=g.device, dtype=torch.float64)
    normal = normal + float(ridge) * torch.eye(int(normal.shape[0]), device=normal.device, dtype=normal.dtype)
    sol, diag = conjugate_gradient_spd(normal, h, max_iter=max_iter, tol=tol)
    diag["mu"] = float(mu)
    return sol, diag


def transform_covariant_objects(g: torch.Tensor, j: torch.Tensor, h: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ss = s.to(device=g.device, dtype=torch.float64)
    gg = g.to(dtype=torch.float64)
    jj = j.to(device=g.device, dtype=torch.float64)
    hh = h.reshape(-1, 1).to(device=g.device, dtype=torch.float64)
    return ss.T @ gg @ ss, jj @ ss, (ss.T @ hh).reshape(-1)


def summarize_matrix_health(g: torch.Tensor, j: torch.Tensor) -> dict[str, float]:
    vals = torch.linalg.eigvalsh(sym(g.to(dtype=torch.float64)))
    return {
        "g_min_eig": float(vals.min().detach().cpu().item()),
        "g_max_eig": float(vals.max().detach().cpu().item()),
        "j_rank": float(torch.linalg.matrix_rank(j.to(dtype=torch.float64)).detach().cpu().item()),
        "parameter_dim": float(int(g.shape[0])),
        "output_dim": float(int(j.shape[0])),
    }
