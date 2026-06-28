"""Conditional edge-signal metric utilities for v22.77.

The optimizer wrapper only transforms gradients inside ``step``. It receives
train-only sketches prepared before training and does not read validation/test
data, teacher targets, or candidate outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


EPS = 1.0e-12


def flat64(x: torch.Tensor) -> torch.Tensor:
    return x.detach().reshape(-1).to(dtype=torch.float64)


def weighted_project(
    vector: torch.Tensor,
    basis: torch.Tensor | None,
    metric_diag: torch.Tensor | None = None,
    *,
    ridge: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Project ``vector`` away from ``basis`` under a diagonal metric."""

    v = flat64(vector)
    if basis is None or int(v.numel()) == 0:
        energy = float((v * v).sum().detach().cpu().item())
        return v.reshape_as(vector).to(dtype=vector.dtype), {
            "projection_applied": 0.0,
            "energy_before": energy,
            "energy_after": energy,
            "projected_energy": 0.0,
            "residual_energy_fraction": 1.0,
            "projected_energy_fraction": 0.0,
        }
    b = basis.detach().to(device=v.device, dtype=torch.float64)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if int(b.shape[0]) != int(v.numel()) or int(b.shape[1]) == 0:
        energy = float((v * v).sum().detach().cpu().item())
        return v.reshape_as(vector).to(dtype=vector.dtype), {
            "projection_applied": 0.0,
            "energy_before": energy,
            "energy_after": energy,
            "projected_energy": 0.0,
            "residual_energy_fraction": 1.0,
            "projected_energy_fraction": 0.0,
        }
    if metric_diag is None or int(metric_diag.numel()) != int(v.numel()):
        m = torch.ones_like(v)
    else:
        m = metric_diag.detach().to(device=v.device, dtype=torch.float64).reshape(-1).clamp_min(0.0) + EPS
    mb = m.reshape(-1, 1) * b
    gram = b.transpose(0, 1) @ mb
    gram = gram + float(max(ridge, 0.0)) * torch.eye(int(gram.shape[0]), device=v.device, dtype=v.dtype)
    rhs = b.transpose(0, 1) @ (m * v)
    coeff = torch.linalg.solve(gram, rhs)
    projected = b @ coeff
    residual = v - projected
    energy_before = (m * v * v).sum().clamp_min(EPS)
    energy_after = (m * residual * residual).sum().clamp_min(0.0)
    projected_energy = (m * projected * projected).sum().clamp_min(0.0)
    return residual.reshape_as(vector).to(dtype=vector.dtype), {
        "projection_applied": 1.0,
        "energy_before": float(energy_before.detach().cpu().item()),
        "energy_after": float(energy_after.detach().cpu().item()),
        "projected_energy": float(projected_energy.detach().cpu().item()),
        "residual_energy_fraction": float((energy_after / energy_before).detach().cpu().item()),
        "projected_energy_fraction": float((projected_energy / energy_before).detach().cpu().item()),
    }


def normalize_columns(cols: list[torch.Tensor], dim: int, device: torch.device) -> torch.Tensor | None:
    out = []
    for col in cols:
        v = col.detach().reshape(-1).to(device=device, dtype=torch.float64)
        if int(v.numel()) != int(dim):
            continue
        norm = v.norm()
        if float(norm.detach().cpu().item()) <= EPS:
            continue
        out.append(v / norm.clamp_min(EPS))
    if not out:
        return None
    return torch.stack(out, dim=1)


def polynomial_nuisance_basis(dim: int, device: torch.device, *, include_density: bool = False) -> torch.Tensor:
    grid = torch.linspace(-1.0, 1.0, int(dim), device=device, dtype=torch.float64)
    cols = [
        torch.ones_like(grid),
        grid,
        grid.square() - grid.square().mean(),
    ]
    if include_density:
        cols.extend(
            [
                torch.sin(torch.pi * grid),
                torch.cos(torch.pi * grid),
                torch.sign(grid) * grid.abs().sqrt(),
            ]
        )
    return normalize_columns(cols, int(dim), device)  # type: ignore[return-value]


def coherence_score(a: torch.Tensor, b: torch.Tensor, metric_diag: torch.Tensor | None = None) -> float:
    av = flat64(a)
    bv = flat64(b).to(device=av.device)
    if metric_diag is None or int(metric_diag.numel()) != int(av.numel()):
        m = torch.ones_like(av)
    else:
        m = metric_diag.detach().to(device=av.device, dtype=torch.float64).reshape(-1).clamp_min(0.0) + EPS
    denom = ((m * av * av).sum().sqrt() * (m * bv * bv).sum().sqrt()).clamp_min(EPS)
    return float(((m * av * bv).sum() / denom).detach().cpu().item())


def shrink_from_lcb(lcb: float, *, threshold: float = 0.05) -> float:
    raw = (float(lcb) - float(threshold)) / max(EPS, 1.0 - float(threshold))
    return float(min(1.0, max(0.0, raw)))


@dataclass
class ConditionalEdgeSignalState:
    domain_basis: torch.Tensor | None = None
    control_basis: torch.Tensor | None = None
    own_basis: torch.Tensor | None = None
    metric_diag: torch.Tensor | None = None
    transform_scale: float = 1.0
    conditional_alpha: float = 1.0
    ridge: float = 1.0e-6
    enabled: bool = True


class ConditionalEdgeSignalOptimizer:
    """Wrap an optimizer and transform edge gradients inside ``step``."""

    def __init__(
        self,
        base_optimizer: object,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        states: dict[str, ConditionalEdgeSignalState] | None = None,
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
            own_res, odiag = weighted_project(control_res, state.own_basis, state.metric_diag, ridge=state.ridge)
            transformed = own_res.detach().to(dtype=param.grad.dtype) * float(state.transform_scale) * float(state.conditional_alpha)
            param.grad = transformed
            self.transformed_gradient_tensors += 1
            merged.update({f"{name}.domain_{k}": v for k, v in ddiag.items()})
            merged.update({f"{name}.control_{k}": v for k, v in cdiag.items()})
            merged.update({f"{name}.own_{k}": v for k, v in odiag.items()})
            merged[f"{name}.domain_nuisance_projection_applied"] = float(ddiag.get("projection_applied", 0.0))
            merged[f"{name}.control_contrastive_metric_applied"] = float(cdiag.get("projection_applied", 0.0))
            merged[f"{name}.conditional_alpha"] = float(state.conditional_alpha)
        out = self.base_optimizer.step(closure)  # type: ignore[attr-defined]
        if hasattr(self.base_optimizer, "diagnostics"):
            merged.update({f"base.{k}": float(v) for k, v in self.base_optimizer.diagnostics().items()})  # type: ignore[attr-defined]
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
        out["domain_nuisance_projection_applied"] = float(
            any(v > 0.5 for k, v in self.last_diagnostics.items() if k.endswith("domain_nuisance_projection_applied"))
        )
        out["control_contrastive_metric_applied"] = float(
            any(v > 0.5 for k, v in self.last_diagnostics.items() if k.endswith("control_contrastive_metric_applied"))
        )
        return out


__all__ = [
    "ConditionalEdgeSignalOptimizer",
    "ConditionalEdgeSignalState",
    "coherence_score",
    "normalize_columns",
    "polynomial_nuisance_basis",
    "shrink_from_lcb",
    "weighted_project",
]
