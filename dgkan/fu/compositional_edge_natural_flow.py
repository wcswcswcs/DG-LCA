"""Natural residual-flow solvers for the v23.16 compositional edge tangent."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import torch

from dgkan.fu.compositional_edge_tangent import (
    CompositionalTangentCache,
    apply_j,
    apply_jt,
    coeff_inner,
    coeff_layer_slices,
    explicit_jacobian,
    flatten_coeffs,
    unflatten_coeffs,
)


EPS = 1.0e-12


@dataclass
class NaturalFlowResult:
    delta: list[torch.Tensor]
    diagnostics: dict[str, float | str]


def zeros_like(xs: list[torch.Tensor]) -> list[torch.Tensor]:
    return [torch.zeros_like(x, dtype=torch.float64) for x in xs]


def add(a: list[torch.Tensor], b: list[torch.Tensor], *, alpha: float = 1.0) -> list[torch.Tensor]:
    return [aa.to(dtype=torch.float64) + float(alpha) * bb.to(device=aa.device, dtype=torch.float64) for aa, bb in zip(a, b)]


def scale(a: list[torch.Tensor], alpha: float) -> list[torch.Tensor]:
    return [aa.to(dtype=torch.float64) * float(alpha) for aa in a]


def metric_apply(delta: list[torch.Tensor], grams: list[torch.Tensor]) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for d, g in zip(delta, grams):
        gg = g.to(device=d.device, dtype=torch.float64)
        out.append(torch.einsum("kl,iol->iok", gg, d.to(dtype=torch.float64)))
    return out


def metric_norm_sq(delta: list[torch.Tensor], grams: list[torch.Tensor]) -> torch.Tensor:
    gd = metric_apply(delta, grams)
    return coeff_inner(delta, gd)


def metric_matrix_for_model(model: Any, grams: list[torch.Tensor]) -> torch.Tensor:
    blocks: list[torch.Tensor] = []
    for coeff, gram in zip(model.coeffs, grams):
        repeat = int(coeff.shape[0]) * int(coeff.shape[1])
        eye = torch.eye(repeat, device=coeff.device, dtype=torch.float64)
        blocks.append(torch.kron(eye, gram.to(device=coeff.device, dtype=torch.float64)))
    return torch.block_diag(*blocks) if blocks else torch.empty((0, 0), dtype=torch.float64)


def apply_normal(
    model: Any,
    cache: CompositionalTangentCache,
    delta: list[torch.Tensor],
    grams: list[torch.Tensor],
    lam: float,
    *,
    sample_weight: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    out = apply_j(model, cache, delta).to(dtype=torch.float64)
    if sample_weight is not None:
        out = out * sample_weight.to(device=out.device, dtype=torch.float64).reshape(-1, 1)
    n = float(max(1, int(out.shape[0])))
    back = apply_jt(model, cache, out)
    back = scale(back, 1.0 / n)
    reg = metric_apply(delta, grams)
    return add(back, reg, alpha=float(lam))


def rhs_from_residual(
    model: Any,
    cache: CompositionalTangentCache,
    residual: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    r = residual.to(device=cache.logits.device, dtype=torch.float64)
    if sample_weight is not None:
        r = r * sample_weight.to(device=r.device, dtype=torch.float64).reshape(-1, 1)
    return scale(apply_jt(model, cache, r), 1.0 / float(max(1, int(r.shape[0]))))


def pcg_solve(
    matvec: Callable[[list[torch.Tensor]], list[torch.Tensor]],
    rhs: list[torch.Tensor],
    *,
    max_iter: int,
    tol: float = 1.0e-8,
    preconditioner: Callable[[list[torch.Tensor]], list[torch.Tensor]] | None = None,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    x = zeros_like(rhs)
    r = add(rhs, matvec(x), alpha=-1.0)
    z = preconditioner(r) if preconditioner is not None else [rr.clone() for rr in r]
    p = [zz.clone() for zz in z]
    rz_old = coeff_inner(r, z).clamp_min(EPS)
    rhs_norm = torch.sqrt(coeff_inner(rhs, rhs)).clamp_min(EPS)
    rel = torch.tensor(float("inf"), dtype=torch.float64, device=rhs[0].device if rhs else torch.device("cpu"))
    it = 0
    for it in range(1, int(max_iter) + 1):
        ap = matvec(p)
        denom = coeff_inner(p, ap).clamp_min(EPS)
        alpha = rz_old / denom
        x = add(x, p, alpha=float(alpha.detach().cpu().item()))
        r = add(r, ap, alpha=-float(alpha.detach().cpu().item()))
        rel = torch.sqrt(coeff_inner(r, r)) / rhs_norm
        if float(rel.detach().cpu().item()) <= float(tol):
            break
        z = preconditioner(r) if preconditioner is not None else [rr.clone() for rr in r]
        rz_new = coeff_inner(r, z).clamp_min(EPS)
        beta = rz_new / rz_old
        p = add(z, p, alpha=float(beta.detach().cpu().item()))
        rz_old = rz_new
    residual = torch.sqrt(coeff_inner(add(matvec(x), rhs, alpha=-1.0), add(matvec(x), rhs, alpha=-1.0))) / rhs_norm
    return x, {
        "PCG_iterations": float(it),
        "PCG_residual": float(residual.detach().cpu().item()),
        "PCG_relative_residual": float(rel.detach().cpu().item()),
    }


def make_vertical_block_preconditioner(
    model: Any,
    cache: CompositionalTangentCache,
    grams: list[torch.Tensor],
    lam: float,
    *,
    jitter: float = 1.0e-9,
) -> Callable[[list[torch.Tensor]], list[torch.Tensor]]:
    factors: list[torch.Tensor] = []
    ns: list[tuple[int, int, int]] = []
    for layer_idx, coeff in enumerate(model.coeffs):
        basis = cache.bases[layer_idx].to(dtype=torch.float64)
        n = float(max(1, int(basis.shape[0])))
        in_dim, out_dim, k = int(coeff.shape[0]), int(coeff.shape[1]), int(coeff.shape[2])
        phi = basis.reshape(int(basis.shape[0]), in_dim * k)
        eye_in = torch.eye(in_dim, device=phi.device, dtype=torch.float64)
        gexp = torch.kron(eye_in, grams[layer_idx].to(device=phi.device, dtype=torch.float64))
        normal = phi.T @ phi / n + float(lam) * gexp
        normal = 0.5 * (normal + normal.T)
        eye = torch.eye(int(normal.shape[0]), device=normal.device, dtype=torch.float64)
        used = float(jitter)
        for _ in range(8):
            try:
                factors.append(torch.linalg.cholesky(normal + used * eye))
                break
            except RuntimeError:
                used *= 10.0
        else:
            factors.append(torch.linalg.cholesky(normal + used * eye))
        ns.append((in_dim, out_dim, k))

    def apply(vecs: list[torch.Tensor]) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for r, chol, (in_dim, out_dim, k) in zip(vecs, factors, ns):
            mat = r.to(dtype=torch.float64).permute(0, 2, 1).reshape(in_dim * k, out_dim)
            sol = torch.cholesky_solve(mat, chol)
            out.append(sol.reshape(in_dim, k, out_dim).permute(0, 2, 1).contiguous())
        return out

    return apply


def global_pcg_flow(
    model: Any,
    cache: CompositionalTangentCache,
    residual: torch.Tensor,
    grams: list[torch.Tensor],
    *,
    lam: float,
    iterations: int = 4,
    preconditioner: str = "vertical_block",
    sample_weight: torch.Tensor | None = None,
) -> NaturalFlowResult:
    rhs = rhs_from_residual(model, cache, residual, sample_weight=sample_weight)
    prec = None
    if preconditioner == "vertical_block":
        prec = make_vertical_block_preconditioner(model, cache, grams, float(lam))
    matvec = lambda d: apply_normal(model, cache, d, grams, float(lam), sample_weight=sample_weight)
    delta, diag = pcg_solve(matvec, rhs, max_iter=int(iterations), preconditioner=prec)
    pred = apply_j(model, cache, delta).to(dtype=torch.float64)
    r = residual.to(device=pred.device, dtype=torch.float64)
    denom = r.square().sum().clamp_min(EPS)
    fit = 1.0 - float((r - pred).square().sum().div(denom).detach().cpu().item())
    diag.update(
        {
            "solver_type": f"global_pcg{int(iterations)}",
            "preconditioner": preconditioner,
            "source_residual_fit_improvement": fit,
            "function_G_step_norm": float(torch.sqrt(metric_norm_sq(delta, grams).clamp_min(0.0)).detach().cpu().item()),
        }
    )
    return NaturalFlowResult(delta=delta, diagnostics=diag)


def gradient_flow(
    model: Any,
    cache: CompositionalTangentCache,
    residual: torch.Tensor,
    grams: list[torch.Tensor],
    *,
    sample_weight: torch.Tensor | None = None,
) -> NaturalFlowResult:
    """Ordinary coefficient-gradient control direction."""
    delta = rhs_from_residual(model, cache, residual, sample_weight=sample_weight)
    pred = apply_j(model, cache, delta).to(dtype=torch.float64)
    r = residual.to(device=pred.device, dtype=torch.float64)
    denom = r.square().sum().clamp_min(EPS)
    return NaturalFlowResult(
        delta=delta,
        diagnostics={
            "solver_type": "ordinary_gradient",
            "source_residual_fit_improvement": 1.0 - float((r - pred).square().sum().div(denom).detach().cpu().item()),
            "function_G_step_norm": float(torch.sqrt(metric_norm_sq(delta, grams).clamp_min(0.0)).detach().cpu().item()),
        },
    )


def exact_global_flow(
    model: Any,
    cache: CompositionalTangentCache,
    residual: torch.Tensor,
    grams: list[torch.Tensor],
    *,
    lam: float,
    sample_weight: torch.Tensor | None = None,
) -> NaturalFlowResult:
    j = explicit_jacobian(model, cache)
    n = float(max(1, int(cache.logits.shape[0])))
    if sample_weight is not None:
        w = sample_weight.to(device=j.device, dtype=torch.float64).repeat_interleave(int(cache.logits.shape[1]))
        jw = j * w.reshape(-1, 1)
    else:
        jw = j
    g = metric_matrix_for_model(model, grams).to(device=j.device, dtype=torch.float64)
    normal = jw.T @ j / n + float(lam) * g
    normal = 0.5 * (normal + normal.T)
    rhs = jw.T @ residual.to(device=j.device, dtype=torch.float64).reshape(-1, 1) / n
    eye = torch.eye(int(normal.shape[0]), device=j.device, dtype=torch.float64)
    used = 0.0
    for attempt in range(8):
        try:
            chol = torch.linalg.cholesky(normal + used * eye)
            sol = torch.cholesky_solve(rhs, chol).reshape(-1)
            break
        except RuntimeError:
            used = 1.0e-10 if used == 0.0 else used * 10.0
    else:
        sol = torch.linalg.solve(normal + used * eye, rhs).reshape(-1)
    delta = unflatten_coeffs(sol, [c.detach().to(dtype=torch.float64) for c in model.coeffs])
    pred = (j @ sol).reshape_as(residual).to(dtype=torch.float64)
    r = residual.to(device=pred.device, dtype=torch.float64)
    denom = r.square().sum().clamp_min(EPS)
    vals = torch.linalg.eigvalsh(normal)
    min_eig = float(vals.min().detach().cpu().item())
    max_eig = float(vals.max().detach().cpu().item())
    resid = float((normal @ sol.reshape(-1, 1) - rhs).norm().div(rhs.norm().clamp_min(EPS)).detach().cpu().item())
    return NaturalFlowResult(
        delta=delta,
        diagnostics={
            "solver_type": "exact_global_dense",
            "jitter_used": float(used),
            "solve_residual": resid,
            "condition_number": float(max_eig / max(min_eig, EPS)) if min_eig > 0 else float("inf"),
            "source_residual_fit_improvement": 1.0 - float((r - pred).square().sum().div(denom).detach().cpu().item()),
            "function_G_step_norm": float(torch.sqrt(metric_norm_sq(delta, grams).clamp_min(0.0)).detach().cpu().item()),
        },
    )


def blocktridiag_exact_flow(
    model: Any,
    cache: CompositionalTangentCache,
    residual: torch.Tensor,
    grams: list[torch.Tensor],
    *,
    lam: float,
    sample_weight: torch.Tensor | None = None,
) -> NaturalFlowResult:
    j = explicit_jacobian(model, cache)
    n = float(max(1, int(cache.logits.shape[0])))
    if sample_weight is not None:
        w = sample_weight.to(device=j.device, dtype=torch.float64).repeat_interleave(int(cache.logits.shape[1]))
        jw = j * w.reshape(-1, 1)
    else:
        jw = j
    g = metric_matrix_for_model(model, grams).to(device=j.device, dtype=torch.float64)
    full = jw.T @ j / n + float(lam) * g
    normal = torch.zeros_like(full)
    slices = coeff_layer_slices(model)
    for i, sli in enumerate(slices):
        for jdx, slj in enumerate(slices):
            if abs(i - jdx) <= 1:
                normal[sli, slj] = full[sli, slj]
    normal = 0.5 * (normal + normal.T)
    rhs = jw.T @ residual.to(device=j.device, dtype=torch.float64).reshape(-1, 1) / n
    eye = torch.eye(int(normal.shape[0]), device=j.device, dtype=torch.float64)
    used = 0.0
    for _ in range(8):
        try:
            chol = torch.linalg.cholesky(normal + used * eye)
            sol = torch.cholesky_solve(rhs, chol).reshape(-1)
            break
        except RuntimeError:
            used = 1.0e-10 if used == 0.0 else used * 10.0
    else:
        sol = torch.linalg.solve(normal + used * eye, rhs).reshape(-1)
    delta = unflatten_coeffs(sol, [c.detach().to(dtype=torch.float64) for c in model.coeffs])
    pred = (j @ sol).reshape_as(residual).to(dtype=torch.float64)
    r = residual.to(device=pred.device, dtype=torch.float64)
    denom = r.square().sum().clamp_min(EPS)
    vals = torch.linalg.eigvalsh(normal + used * eye)
    min_eig = float(vals.min().detach().cpu().item())
    max_eig = float(vals.max().detach().cpu().item())
    resid = float((normal @ sol.reshape(-1, 1) - rhs).norm().div(rhs.norm().clamp_min(EPS)).detach().cpu().item())
    return NaturalFlowResult(
        delta=delta,
        diagnostics={
            "solver_type": "blocktridiag_exact",
            "jitter_used": float(used),
            "solve_residual": resid,
            "condition_number": float(max_eig / max(min_eig, EPS)) if min_eig > 0 else float("inf"),
            "source_residual_fit_improvement": 1.0 - float((r - pred).square().sum().div(denom).detach().cpu().item()),
            "function_G_step_norm": float(torch.sqrt(metric_norm_sq(delta, grams).clamp_min(0.0)).detach().cpu().item()),
        },
    )


def blockdiag_exact_flow(
    model: Any,
    cache: CompositionalTangentCache,
    residual: torch.Tensor,
    grams: list[torch.Tensor],
    *,
    lam: float,
) -> NaturalFlowResult:
    j = explicit_jacobian(model, cache)
    n = float(max(1, int(cache.logits.shape[0])))
    r = residual.to(device=j.device, dtype=torch.float64).reshape(-1, 1)
    g = metric_matrix_for_model(model, grams).to(device=j.device, dtype=torch.float64)
    rhs = j.T @ r / n
    sol = torch.zeros((int(j.shape[1]), 1), device=j.device, dtype=torch.float64)
    for sl in coeff_layer_slices(model):
        jl = j[:, sl]
        gl = g[sl, sl]
        normal = 0.5 * (jl.T @ jl / n + float(lam) * gl + (jl.T @ jl / n + float(lam) * gl).T)
        eye = torch.eye(int(normal.shape[0]), device=j.device, dtype=torch.float64)
        chol = torch.linalg.cholesky(normal + 1.0e-9 * eye)
        sol[sl] = torch.cholesky_solve(rhs[sl], chol)
    delta = unflatten_coeffs(sol.reshape(-1), [c.detach().to(dtype=torch.float64) for c in model.coeffs])
    pred = (j @ sol).reshape_as(residual).to(dtype=torch.float64)
    denom = residual.to(device=j.device, dtype=torch.float64).square().sum().clamp_min(EPS)
    return NaturalFlowResult(
        delta=delta,
        diagnostics={
            "solver_type": "blockdiag_exact",
            "source_residual_fit_improvement": 1.0 - float((residual.to(device=j.device, dtype=torch.float64) - pred).square().sum().div(denom).detach().cpu().item()),
            "function_G_step_norm": float(torch.sqrt(metric_norm_sq(delta, grams).clamp_min(0.0)).detach().cpu().item()),
        },
    )


def offdiag_energy_ratio(model: Any, cache: CompositionalTangentCache, grams: list[torch.Tensor], lam: float) -> tuple[float, float, float]:
    j = explicit_jacobian(model, cache)
    n = float(max(1, int(cache.logits.shape[0])))
    g = metric_matrix_for_model(model, grams).to(device=j.device, dtype=torch.float64)
    normal = 0.5 * (j.T @ j / n + float(lam) * g + (j.T @ j / n + float(lam) * g).T)
    block = torch.zeros_like(normal)
    for sl in coeff_layer_slices(model):
        block[sl, sl] = normal[sl, sl]
    off = normal - block
    total = float(normal.norm().detach().cpu().item())
    ratio = float(off.norm().detach().cpu().item()) / max(total, EPS)
    adjacent = 0.0
    nonadjacent = 0.0
    slices = coeff_layer_slices(model)
    for i, sli in enumerate(slices):
        for jdx, slj in enumerate(slices):
            if i == jdx:
                continue
            val = float(normal[sli, slj].norm().detach().cpu().item())
            if abs(i - jdx) == 1:
                adjacent += val
            else:
                nonadjacent += val
    denom = adjacent + nonadjacent + EPS
    return ratio, adjacent / denom, nonadjacent / denom
