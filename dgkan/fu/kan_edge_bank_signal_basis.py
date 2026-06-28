"""Edge-bank signal and basis-redesign utilities for v22.78.

The objects in this file are train-only gradient transformers and diagnostics.
They do not read validation/test data and they do not choose row-wise actions
from candidate outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from dgkan.fu.kan_conditional_edge_signal_metric import EPS, flat64, weighted_project


def _metric(vector: torch.Tensor, metric_diag: torch.Tensor | None) -> torch.Tensor:
    v = flat64(vector)
    if metric_diag is None or int(metric_diag.numel()) != int(v.numel()):
        return torch.ones_like(v)
    return metric_diag.detach().to(device=v.device, dtype=torch.float64).reshape(-1).clamp_min(0.0) + EPS


def metric_energy(vector: torch.Tensor, metric_diag: torch.Tensor | None = None) -> torch.Tensor:
    v = flat64(vector)
    m = _metric(v, metric_diag)
    return (m * v * v).sum().clamp_min(0.0)


def param_bank_groups(shape: torch.Size | tuple[int, ...], device: torch.device) -> list[torch.Tensor]:
    """Return output-node bank groups for flattened ``w2`` style tensors."""

    dims = tuple(int(x) for x in shape)
    total = 1
    for dim in dims:
        total *= max(1, dim)
    ar = torch.arange(total, device=device, dtype=torch.long).reshape(dims)
    if len(dims) == 3:
        return [ar[:, cls, :].reshape(-1) for cls in range(dims[1])]
    if len(dims) == 2:
        return [ar[:, cls].reshape(-1) for cls in range(dims[1])]
    return [ar.reshape(-1)]


def quantile_basis_grid(n: int, device: torch.device, *, family: str, rank: int = 4) -> torch.Tensor:
    s = torch.linspace(0.0, 1.0, int(n), device=device, dtype=torch.float64)
    cols = [torch.ones_like(s)]
    if family == "density_spline":
        centers = torch.linspace(0.0, 1.0, max(2, int(rank)), device=device, dtype=torch.float64)
        width = 1.0 / max(2.0, float(rank) - 1.0)
        cols.extend([torch.clamp(1.0 - (s - c).abs() / max(width, EPS), min=0.0) for c in centers])
        cols.extend([s - 0.5, (s - 0.5).square() - (s - 0.5).square().mean()])
    elif family == "orthogonal_poly":
        x = 2.0 * s - 1.0
        cols.extend([x, x.square() - x.square().mean(), x.pow(3) - x.pow(3).mean(), x.pow(4) - x.pow(4).mean()])
    elif family == "lowfreq_bump":
        for k in range(1, max(2, int(rank)) + 1):
            cols.append(torch.sin(2.0 * torch.pi * float(k) * s))
            cols.append(torch.cos(2.0 * torch.pi * float(k) * s))
        centers = torch.linspace(0.1, 0.9, max(3, int(rank)), device=device, dtype=torch.float64)
        width = 0.12
        cols.extend([torch.exp(-0.5 * ((s - c) / width).square()) for c in centers])
    else:
        cols.extend([s - 0.5, torch.sin(2.0 * torch.pi * s), torch.cos(2.0 * torch.pi * s)])
    mat = torch.stack(cols, dim=1)
    mat = mat - mat.mean(dim=0, keepdim=True)
    norms = mat.norm(dim=0, keepdim=True).clamp_min(EPS)
    keep = (norms.reshape(-1) > 10.0 * EPS)
    return mat[:, keep] / norms[:, keep]


def project_grouped_quantile_basis(
    vector: torch.Tensor,
    metric_diag: torch.Tensor | None,
    shape: torch.Size | tuple[int, ...],
    *,
    family: str,
    rank: int = 4,
    ridge: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Project a vector onto train-only quantile-basis groups."""

    v = flat64(vector)
    m = _metric(v, metric_diag)
    out = torch.zeros_like(v)
    groups = param_bank_groups(tuple(int(x) for x in shape), v.device)
    condition_numbers = []
    lowfreq_energy = 0.0
    local_energy = 0.0
    density_balances = []
    coverage = []
    for group in groups:
        idx = group.to(device=v.device)
        if int(idx.numel()) == 0:
            continue
        dens = m[idx]
        order = torch.argsort(dens)
        sorted_idx = idx[order]
        y = v[sorted_idx]
        w = m[sorted_idx].clamp_min(EPS)
        basis = quantile_basis_grid(int(sorted_idx.numel()), v.device, family=family, rank=rank)
        wb = w.reshape(-1, 1) * basis
        gram = basis.transpose(0, 1) @ wb
        gram = gram + float(max(ridge, 0.0)) * torch.eye(int(gram.shape[0]), device=v.device, dtype=torch.float64)
        rhs = basis.transpose(0, 1) @ (w * y)
        coeff = torch.linalg.solve(gram, rhs)
        fitted = basis @ coeff
        out[sorted_idx] = fitted
        eig = torch.linalg.eigvalsh(gram).clamp_min(EPS)
        condition_numbers.append(float((eig.max() / eig.min()).detach().cpu().item()))
        if "lowfreq" in family:
            cut = min(int(coeff.numel()), 1 + 2 * max(2, int(rank)))
            lowfreq_energy += float(coeff[:cut].square().sum().detach().cpu().item())
            local_energy += float(coeff[cut:].square().sum().detach().cpu().item())
        chunks = torch.chunk(dens.detach().float(), max(2, min(8, int(dens.numel()))))
        means = torch.tensor([c.mean() for c in chunks if int(c.numel()) > 0], device=v.device)
        if int(means.numel()) > 0:
            density_balances.append(float((means.min() / means.max().clamp_min(EPS)).detach().cpu().item()))
            coverage.append(float((dens > dens.median()).float().mean().detach().cpu().item()))
    before = metric_energy(v, m).clamp_min(EPS)
    after = metric_energy(out, m).clamp_min(0.0)
    residual = metric_energy(v - out, m).clamp_min(0.0)
    total_basis_energy = max(EPS, lowfreq_energy + local_energy)
    return out.reshape_as(vector).to(dtype=vector.dtype), {
        "basis_projection_energy_fraction": float((after / before).detach().cpu().item()),
        "basis_projection_residual_fraction": float((residual / before).detach().cpu().item()),
        "basis_condition_number": float(max(condition_numbers) if condition_numbers else 0.0),
        "knot_density_balance": float(sum(density_balances) / max(1, len(density_balances))),
        "low_frequency_energy_fraction": float(lowfreq_energy / total_basis_energy),
        "local_bump_energy_fraction": float(local_energy / total_basis_energy),
        "bump_domain_density_coverage": float(sum(coverage) / max(1, len(coverage))),
        "task_orthogonal_poly_rank": float(min(max(1, int(rank)), 5)),
        "poly_condition_number": float(max(condition_numbers) if condition_numbers else 0.0),
    }


def bank_additive_projection(
    vector: torch.Tensor,
    metric_diag: torch.Tensor | None,
    shape: torch.Size | tuple[int, ...],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Project each output-node bank onto a row+basis additive structure."""

    dims = tuple(int(x) for x in shape)
    v = flat64(vector)
    m = _metric(v, metric_diag)
    if len(dims) != 3:
        total = metric_energy(v, m).clamp_min(EPS)
        return vector.detach().clone(), {
            "edge_bank_anova_explained": 1.0,
            "interaction_residual_fraction": 0.0,
            "bank_condition_number": 1.0,
            "source_witness_bank_coherence_LCB": 0.0,
        }
    block = v.reshape(dims)
    mb = m.reshape(dims)
    fitted = torch.zeros_like(block)
    condition_numbers = []
    for cls in range(dims[1]):
        y = block[:, cls, :]
        w = mb[:, cls, :].clamp_min(EPS)
        grand = (w * y).sum() / w.sum().clamp_min(EPS)
        row_mean = (w * y).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True).clamp_min(EPS)
        col_mean = (w * y).sum(dim=0, keepdim=True) / w.sum(dim=0, keepdim=True).clamp_min(EPS)
        fitted[:, cls, :] = row_mean + col_mean - grand
        row_scale = float(w.sum(dim=1).max().detach().cpu().item() / w.sum(dim=1).min().clamp_min(EPS).detach().cpu().item())
        col_scale = float(w.sum(dim=0).max().detach().cpu().item() / w.sum(dim=0).min().clamp_min(EPS).detach().cpu().item())
        condition_numbers.append(max(row_scale, col_scale))
    total = metric_energy(v, m).clamp_min(EPS)
    explained = metric_energy(fitted.reshape(-1), m).clamp_min(0.0) / total
    residual = metric_energy((block - fitted).reshape(-1), m).clamp_min(0.0) / total
    return fitted.reshape_as(vector).to(dtype=vector.dtype), {
        "edge_bank_anova_explained": float(explained.detach().cpu().item()),
        "interaction_residual_fraction": float(residual.detach().cpu().item()),
        "bank_condition_number": float(max(condition_numbers) if condition_numbers else 1.0),
    }


def bank_coherence(a: torch.Tensor, b: torch.Tensor, metric_diag: torch.Tensor | None, shape: torch.Size | tuple[int, ...]) -> float:
    ap, _ = bank_additive_projection(a, metric_diag, shape)
    bp, _ = bank_additive_projection(b, metric_diag, shape)
    av = flat64(ap)
    bv = flat64(bp).to(device=av.device)
    m = _metric(av, metric_diag)
    denom = ((m * av * av).sum().sqrt() * (m * bv * bv).sum().sqrt()).clamp_min(EPS)
    return float(((m * av * bv).sum() / denom).detach().cpu().item())


def edge_signal_sparsity(vector: torch.Tensor, metric_diag: torch.Tensor | None, *, threshold_fraction: float = 0.05) -> float:
    v = flat64(vector).abs()
    if int(v.numel()) == 0:
        return 0.0
    m = _metric(v, metric_diag)
    score = v * m.sqrt()
    threshold = score.max().clamp_min(EPS) * float(threshold_fraction)
    return float((score <= threshold).float().mean().detach().cpu().item())


def sign_cancellation_rate(vector: torch.Tensor, shape: torch.Size | tuple[int, ...]) -> float:
    dims = tuple(int(x) for x in shape)
    v = flat64(vector)
    if len(dims) != 3:
        return 0.0
    block = v.reshape(dims)
    rates = []
    for cls in range(dims[1]):
        y = block[:, cls, :]
        pos = float((y > 0).float().mean().detach().cpu().item())
        neg = float((y < 0).float().mean().detach().cpu().item())
        rates.append(min(pos, neg) * 2.0)
    return float(sum(rates) / max(1, len(rates)))


@dataclass
class EdgeBankSignalBasisState:
    domain_basis: torch.Tensor | None = None
    control_basis: torch.Tensor | None = None
    bank_basis: torch.Tensor | None = None
    metric_diag: torch.Tensor | None = None
    param_shape: tuple[int, ...] | None = None
    transform_scale: float = 1.0
    conditional_alpha: float = 1.0
    ridge: float = 1.0e-6
    enabled: bool = True


class EdgeBankSignalBasisOptimizer:
    """Optimizer wrapper for v22.78 boundary checks."""

    def __init__(
        self,
        base_optimizer: object,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        states: dict[str, EdgeBankSignalBasisState] | None = None,
    ) -> None:
        self.base_optimizer = base_optimizer
        self.named_parameters = [(name, param) for name, param in named_parameters]
        self.states = states or {}
        self.transform_calls = 0
        self.transformed_gradient_tensors = 0
        self.last_diagnostics: dict[str, float] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)  # type: ignore[attr-defined]

    def step(self, closure=None):  # type: ignore[no-untyped-def]
        self.transform_calls += 1
        merged: dict[str, float] = {}
        for name, param in self.named_parameters:
            if param.grad is None:
                continue
            state = self.states.get(name)
            if state is None or not state.enabled:
                continue
            grad = param.grad.detach()
            domain_res, ddiag = weighted_project(grad, state.domain_basis, state.metric_diag, ridge=state.ridge)
            control_res, cdiag = weighted_project(domain_res, state.control_basis, state.metric_diag, ridge=state.ridge)
            transformed = control_res.detach().to(dtype=param.grad.dtype) * float(state.transform_scale) * float(state.conditional_alpha)
            param.grad = transformed
            self.transformed_gradient_tensors += 1
            merged.update({f"{name}.domain_{k}": v for k, v in ddiag.items()})
            merged.update({f"{name}.control_{k}": v for k, v in cdiag.items()})
            merged[f"{name}.domain_nuisance_projection_applied"] = float(ddiag.get("projection_applied", 0.0))
            merged[f"{name}.control_contrastive_metric_applied"] = float(cdiag.get("projection_applied", 0.0))
        out = self.base_optimizer.step(closure)  # type: ignore[attr-defined]
        self.last_diagnostics = merged
        return out

    def state_dict(self):  # type: ignore[no-untyped-def]
        return self.base_optimizer.state_dict()  # type: ignore[attr-defined]

    def diagnostics(self) -> dict[str, float]:
        out = dict(self.last_diagnostics)
        out["optimizer_owned_gradient_transform_pass"] = float(self.transform_calls > 0)
        out["optimizer_transform_calls"] = float(self.transform_calls)
        out["transformed_gradient_tensors"] = float(self.transformed_gradient_tensors)
        out["edge_conditional_metric_applied"] = float(bool(self.states))
        out["edge_bank_anova_metric_applied"] = float(bool(self.states))
        out["basis_redesign_family_applied"] = float(bool(self.states))
        out["domain_nuisance_projection_applied"] = float(
            any(v > 0.5 for k, v in self.last_diagnostics.items() if k.endswith("domain_nuisance_projection_applied"))
        )
        out["control_contrastive_metric_applied"] = float(
            any(v > 0.5 for k, v in self.last_diagnostics.items() if k.endswith("control_contrastive_metric_applied"))
        )
        return out


__all__ = [
    "EdgeBankSignalBasisOptimizer",
    "EdgeBankSignalBasisState",
    "bank_additive_projection",
    "bank_coherence",
    "edge_signal_sparsity",
    "metric_energy",
    "param_bank_groups",
    "project_grouped_quantile_basis",
    "quantile_basis_grid",
    "sign_cancellation_rate",
]
