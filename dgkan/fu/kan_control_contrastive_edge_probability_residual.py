"""Control-contrastive edge-probability residual utilities for v22.76.

These helpers are optimizer-side only. They residualize gradients against
train-only control/own-reference sketches and never read validation/test data,
choose among candidate actions, or add auxiliary losses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch


EPS = 1.0e-12


def _flat64(x: torch.Tensor) -> torch.Tensor:
    return x.detach().reshape(-1).to(dtype=torch.float64)


def metric_residual_project(
    vector: torch.Tensor,
    basis: torch.Tensor | None,
    metric_diag: torch.Tensor | None = None,
    *,
    ridge: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Project ``vector`` away from ``basis`` under a diagonal metric."""

    v = _flat64(vector)
    if basis is None or int(v.numel()) == 0:
        return v.reshape_as(vector).to(dtype=vector.dtype), {
            "projection_applied": 0.0,
            "energy_before": float((v * v).sum().detach().cpu().item()),
            "energy_after": float((v * v).sum().detach().cpu().item()),
            "residual_energy_fraction": 1.0,
        }
    nmat = basis.detach().to(device=v.device, dtype=torch.float64)
    if nmat.ndim == 1:
        nmat = nmat.reshape(-1, 1)
    if int(nmat.shape[0]) != int(v.numel()) or int(nmat.shape[1]) == 0:
        return v.reshape_as(vector).to(dtype=vector.dtype), {
            "projection_applied": 0.0,
            "energy_before": float((v * v).sum().detach().cpu().item()),
            "energy_after": float((v * v).sum().detach().cpu().item()),
            "residual_energy_fraction": 1.0,
        }
    if metric_diag is None or int(metric_diag.numel()) != int(v.numel()):
        m = torch.ones_like(v)
    else:
        m = metric_diag.detach().to(device=v.device, dtype=torch.float64).reshape(-1).clamp_min(0.0) + EPS
    mn = m.reshape(-1, 1) * nmat
    gram = nmat.transpose(0, 1) @ mn
    gram = gram + float(max(ridge, 0.0)) * torch.eye(int(gram.shape[0]), device=v.device, dtype=v.dtype)
    rhs = nmat.transpose(0, 1) @ (m * v)
    coeff = torch.linalg.solve(gram, rhs)
    proj = nmat @ coeff
    residual = v - proj
    energy_before = (m * v * v).sum().clamp_min(EPS)
    energy_after = (m * residual * residual).sum().clamp_min(0.0)
    ctrl_energy = (m * proj * proj).sum().clamp_min(0.0)
    return residual.reshape_as(vector).to(dtype=vector.dtype), {
        "projection_applied": 1.0,
        "energy_before": float(energy_before.detach().cpu().item()),
        "energy_after": float(energy_after.detach().cpu().item()),
        "projected_control_energy": float(ctrl_energy.detach().cpu().item()),
        "residual_energy_fraction": float((energy_after / energy_before).detach().cpu().item()),
        "control_energy_fraction": float((ctrl_energy / energy_before).detach().cpu().item()),
    }


@dataclass
class ControlContrastiveResidualState:
    control_basis: torch.Tensor | None = None
    own_basis: torch.Tensor | None = None
    metric_diag: torch.Tensor | None = None
    transform_scale: float = 1.0
    ridge: float = 1.0e-6
    enabled: bool = True


class ControlContrastiveResidualOptimizer:
    """Wrap an optimizer and residualize gradients inside ``step``."""

    def __init__(
        self,
        base_optimizer: object,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        states: dict[str, ControlContrastiveResidualState] | None = None,
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
            work = param.grad.detach()
            res_ctrl, cdiag = metric_residual_project(work, state.control_basis, state.metric_diag, ridge=state.ridge)
            res_own, odiag = metric_residual_project(res_ctrl, state.own_basis, state.metric_diag, ridge=state.ridge)
            transformed = res_own.detach().to(dtype=param.grad.dtype) * float(state.transform_scale)
            param.grad = transformed
            self.transformed_gradient_tensors += 1
            merged.update({f"{name}.control_{k}": v for k, v in cdiag.items()})
            merged.update({f"{name}.own_{k}": v for k, v in odiag.items()})
            merged[f"{name}.control_residual_projection_applied"] = float(cdiag.get("projection_applied", 0.0))
            merged[f"{name}.own_reference_residual_applied"] = float(odiag.get("projection_applied", 0.0))
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
        out["edge_control_contrastive_metric_applied"] = float(bool(self.states))
        out["control_residual_projection_applied"] = float(
            any(v > 0.5 for k, v in self.last_diagnostics.items() if k.endswith("control_residual_projection_applied"))
        )
        out["own_reference_residual_applied"] = float(
            any(v > 0.5 for k, v in self.last_diagnostics.items() if k.endswith("own_reference_residual_applied"))
        )
        return out


__all__ = [
    "ControlContrastiveResidualOptimizer",
    "ControlContrastiveResidualState",
    "metric_residual_project",
]
