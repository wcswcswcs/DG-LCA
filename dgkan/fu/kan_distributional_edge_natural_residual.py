"""Distributional edge-natural residual operators for KAN optimizer steps.

This module extends the v22.72 edge-natural residual wrapper with a train-only
distributional debt cone.  The public optimizer still transforms gradients only
inside ``optimizer.step``; it does not add auxiliary losses, samplers, class
weights, validation/test directions, or runtime candidate selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch

from dgkan.fu.kan_edge_natural_residual import (
    EPS,
    debt_cone_project,
    own_reference_residual_project,
    raw_readout_multiplier,
)


def _flat64(x: torch.Tensor) -> torch.Tensor:
    return x.detach().reshape(-1).to(dtype=torch.float64)


def _unit_columns(mat: torch.Tensor) -> torch.Tensor:
    work = mat.detach().to(dtype=torch.float64)
    if int(work.numel()) == 0:
        rows = int(work.shape[0]) if work.ndim > 0 else 0
        return work.reshape(rows, 0)
    if work.ndim == 1:
        work = work.reshape(-1, 1)
    keep = torch.linalg.norm(work, dim=0) > 1.0e-12
    if int(keep.sum().item()) == 0:
        return work[:, :0]
    q, _r = torch.linalg.qr(work[:, keep], mode="reduced")
    return q


@dataclass
class DistributionalDebtConstraint:
    """One train-only query-cohort debt constraint in flattened parameter space."""

    direction: torch.Tensor
    mode: str
    cohort: str = "query"
    curvature: float = 0.0
    slack: float = 0.0
    budget: float = 0.0
    alpha: float = 1.0


def distributional_debt_predict(
    update: torch.Tensor,
    constraint: DistributionalDebtConstraint,
) -> float:
    """Predict debt delta for one constraint from linear, curvature and slack terms."""

    u = _flat64(update)
    d = _flat64(constraint.direction).to(device=u.device)
    if int(d.numel()) != int(u.numel()) or int(u.numel()) == 0:
        return float("inf")
    lin = torch.dot(d, u)
    curvature = max(float(constraint.curvature), 0.0)
    quad = 0.5 * float(constraint.alpha) * curvature * float(torch.dot(u, u).detach().cpu().item())
    return float(lin.detach().cpu().item()) + quad + float(constraint.slack) - float(constraint.budget)


def distributional_debt_cone_project(
    candidate: torch.Tensor,
    constraints: Iterable[DistributionalDebtConstraint] | None,
    *,
    max_passes: int = 6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Project ``candidate`` into the approximate distributional debt cone.

    The projection uses a conservative half-space approximation per constraint:
    ``<d, u> + 0.5 * alpha * max(curvature, 0) * ||u||^2 + slack <= budget``.
    The quadratic term is reevaluated after each pass and never treated as a
    negative reward.
    """

    out = _flat64(candidate)
    cons = list(constraints or [])
    usable = [c for c in cons if int(c.direction.numel()) == int(out.numel()) and int(out.numel()) > 0]
    if not usable:
        return out.reshape_as(candidate).to(dtype=torch.float64), {
            "distributional_debt_constraints": 0.0,
            "distributional_debt_violation_before": 0.0,
            "distributional_debt_violation_after": 0.0,
            "distributional_debt_active_fraction": 0.0,
        }

    dirs = torch.stack([_flat64(c.direction).to(device=out.device) for c in usable], dim=1)
    norms = dirs.square().sum(dim=0).clamp_min(EPS)

    def violations(vec: torch.Tensor) -> torch.Tensor:
        dots = dirs.transpose(0, 1) @ vec
        norm2 = float(torch.dot(vec, vec).detach().cpu().item())
        rhs = []
        for c in usable:
            curve = max(float(c.curvature), 0.0)
            allowance = float(c.budget) - float(c.slack) - 0.5 * float(c.alpha) * curve * norm2
            rhs.append(allowance)
        rhs_t = torch.tensor(rhs, device=vec.device, dtype=vec.dtype)
        return (dots - rhs_t).clamp_min(0.0)

    def linear_violations(vec: torch.Tensor) -> torch.Tensor:
        dots = dirs.transpose(0, 1) @ vec
        rhs = [float(c.budget) - float(c.slack) for c in usable]
        rhs_t = torch.tensor(rhs, device=vec.device, dtype=vec.dtype)
        return (dots - rhs_t).clamp_min(0.0)

    before = violations(out)
    active = before > 1.0e-12
    zero = torch.zeros_like(out)
    zero_safe = float(violations(zero).max().detach().cpu().item()) <= 1.0e-10
    for _ in range(max(1, int(max_passes))):
        full_viol = violations(out)
        if float(full_viol.max().detach().cpu().item()) <= 1.0e-10:
            break
        lin_viol = linear_violations(out)
        if float(lin_viol.max().detach().cpu().item()) > 1.0e-10:
            out = out - dirs @ (lin_viol / norms)
        full_viol = violations(out)
        if float(full_viol.max().detach().cpu().item()) <= 1.0e-10:
            break
        if zero_safe:
            base = out
            lo, hi = 0.0, 1.0
            for _bisect in range(40):
                mid = 0.5 * (lo + hi)
                mid_viol = violations(base * mid)
                if float(mid_viol.max().detach().cpu().item()) <= 1.0e-10:
                    lo = mid
                else:
                    hi = mid
            out = base * lo
            if float(violations(out).max().detach().cpu().item()) <= 1.0e-10:
                break
        else:
            break
    after = violations(out)
    return out.reshape_as(candidate).to(dtype=torch.float64), {
        "distributional_debt_constraints": float(len(usable)),
        "distributional_debt_violation_before": float(before.max().detach().cpu().item()),
        "distributional_debt_violation_after": float(after.max().detach().cpu().item()),
        "distributional_debt_active_fraction": float(active.to(dtype=torch.float64).mean().detach().cpu().item()),
    }


def functional_residual_fraction(
    candidate_displacement: torch.Tensor,
    own_displacements: torch.Tensor | None,
) -> tuple[float, float]:
    """Return residual energy and post-projection overlap in output-function space."""

    cand = _flat64(candidate_displacement)
    denom = torch.linalg.norm(cand).square().clamp_min(EPS)
    if own_displacements is None or int(own_displacements.numel()) == 0:
        return 1.0, 0.0
    basis = own_displacements.detach().to(dtype=torch.float64)
    if basis.ndim == 1:
        basis = basis.reshape(-1, 1)
    if int(basis.shape[0]) != int(cand.numel()):
        return 1.0, 0.0
    q = _unit_columns(basis)
    if int(q.numel()) == 0:
        return 1.0, 0.0
    coeff = q.transpose(0, 1) @ cand
    residual = cand - q @ coeff
    res_norm = torch.linalg.norm(residual).square()
    overlap = torch.linalg.norm(q.transpose(0, 1) @ residual) / torch.linalg.norm(residual).clamp_min(EPS)
    return float((res_norm / denom).clamp(0.0, 1.0).detach().cpu().item()), float(overlap.detach().cpu().item())


@dataclass
class DistributionalEdgeNaturalResidualState:
    """Per-parameter state for distributional edge-natural residual transforms."""

    raw_multiplier: torch.Tensor | None = None
    own_basis: torch.Tensor | None = None
    distributional_constraints: list[DistributionalDebtConstraint] = field(default_factory=list)
    transform_scale: float = 1.0
    enabled: bool = True


class DistributionalEdgeNaturalResidualOptimizer:
    """AdamW wrapper that applies distributional edge transforms inside ``step``."""

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        lr: float,
        weight_decay: float,
        edge_states: dict[str, DistributionalEdgeNaturalResidualState] | None = None,
    ) -> None:
        self.named_parameters = [(name, param) for name, param in named_parameters]
        self.base_optimizer = torch.optim.AdamW(
            [param for _name, param in self.named_parameters],
            lr=float(lr),
            weight_decay=float(weight_decay),
        )
        self.edge_states = edge_states or {}
        self.transform_calls = 0
        self.transformed_gradient_tensors = 0
        self.last_diagnostics: dict[str, float] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _transform_one(self, name: str, grad: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        state = self.edge_states.get(name)
        if state is None or not state.enabled:
            return grad, {}
        work = grad.detach().to(dtype=torch.float64)
        diag: dict[str, float] = {}
        if state.raw_multiplier is not None and int(state.raw_multiplier.numel()) == int(work.numel()):
            mult = state.raw_multiplier.to(device=work.device, dtype=work.dtype).reshape_as(work)
            work = work * mult
            diag["raw_multiplier_applied"] = 1.0
        residual, rdiag = own_reference_residual_project(
            work,
            state.own_basis.to(work.device) if state.own_basis is not None else None,
        )
        projected, ddiag = distributional_debt_cone_project(
            residual,
            [
                DistributionalDebtConstraint(
                    direction=c.direction.to(work.device),
                    mode=c.mode,
                    cohort=c.cohort,
                    curvature=c.curvature,
                    slack=c.slack,
                    budget=c.budget,
                    alpha=c.alpha,
                )
                for c in state.distributional_constraints
            ],
        )
        diag.update(rdiag)
        diag.update(ddiag)
        scale = float(state.transform_scale)
        diag["transform_scale"] = scale
        return (projected * scale).to(dtype=grad.dtype), diag

    def step(self, closure=None):  # type: ignore[no-untyped-def]
        self.transform_calls += 1
        merged: dict[str, float] = {}
        for name, param in self.named_parameters:
            if param.grad is None:
                continue
            new_grad, diag = self._transform_one(name, param.grad)
            if diag:
                self.transformed_gradient_tensors += 1
                param.grad = new_grad
                merged.update({f"{name}.{key}": value for key, value in diag.items()})
        self.last_diagnostics = merged
        return self.base_optimizer.step(closure)

    def state_dict(self):  # type: ignore[no-untyped-def]
        return self.base_optimizer.state_dict()

    def diagnostics(self) -> dict[str, float]:
        out = dict(self.last_diagnostics)
        out["optimizer_owned_gradient_transform_pass"] = float(self.transform_calls > 0)
        out["optimizer_transform_calls"] = float(self.transform_calls)
        out["transformed_gradient_tensors"] = float(self.transformed_gradient_tensors)
        return out


__all__ = [
    "DistributionalDebtConstraint",
    "DistributionalEdgeNaturalResidualOptimizer",
    "DistributionalEdgeNaturalResidualState",
    "debt_cone_project",
    "distributional_debt_cone_project",
    "distributional_debt_predict",
    "functional_residual_fraction",
    "raw_readout_multiplier",
]
