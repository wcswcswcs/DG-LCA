"""Edge-native residual and debt-cone operators for KAN optimizer steps.

The operators here are target-free at runtime: they transform existing task
gradients inside ``optimizer.step``.  They do not add auxiliary losses, class
weights, samplers, validation/test signals, or candidate-action selectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


EPS = 1.0e-12


def _as_flat64(vec: torch.Tensor) -> torch.Tensor:
    return vec.detach().reshape(-1).to(dtype=torch.float64)


def _unit_columns(mat: torch.Tensor) -> torch.Tensor:
    work = mat.detach().to(dtype=torch.float64)
    if int(work.numel()) == 0:
        return work.reshape(int(work.shape[0]) if work.ndim > 0 else 0, 0)
    if work.ndim == 1:
        work = work.reshape(-1, 1)
    keep = torch.linalg.norm(work, dim=0) > 1.0e-12
    if int(keep.sum().item()) == 0:
        return work[:, :0]
    work = work[:, keep]
    q, _r = torch.linalg.qr(work, mode="reduced")
    return q


def own_reference_residual_project(
    candidate: torch.Tensor,
    own_basis: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Remove the component of ``candidate`` lying in the own-reference span."""

    cand = _as_flat64(candidate)
    before_norm = torch.linalg.norm(cand).clamp_min(EPS)
    if own_basis is None or int(own_basis.numel()) == 0:
        return cand.reshape_as(candidate).to(dtype=torch.float64), {
            "own_overlap_before": 0.0,
            "own_overlap_after": 0.0,
            "own_residual_energy_fraction": 1.0,
        }
    q = _unit_columns(own_basis)
    if int(q.numel()) == 0:
        return cand.reshape_as(candidate).to(dtype=torch.float64), {
            "own_overlap_before": 0.0,
            "own_overlap_after": 0.0,
            "own_residual_energy_fraction": 1.0,
        }
    coeff = q.transpose(0, 1) @ cand
    overlap_before = torch.linalg.norm(coeff) / before_norm
    residual = cand - q @ coeff
    after_norm = torch.linalg.norm(residual).clamp_min(EPS)
    coeff_after = q.transpose(0, 1) @ residual
    overlap_after = torch.linalg.norm(coeff_after) / after_norm
    return residual.reshape_as(candidate).to(dtype=torch.float64), {
        "own_overlap_before": float(overlap_before.detach().cpu().item()),
        "own_overlap_after": float(overlap_after.detach().cpu().item()),
        "own_residual_energy_fraction": float((after_norm.square() / before_norm.square()).clamp(0.0, 1.0).detach().cpu().item()),
    }


def debt_cone_project(
    candidate: torch.Tensor,
    debt_directions: torch.Tensor | None,
    *,
    max_passes: int = 4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Project onto the linearized cone ``<candidate, debt_dir> <= 0``."""

    out = _as_flat64(candidate)
    if debt_directions is None or int(debt_directions.numel()) == 0:
        return out.reshape_as(candidate).to(dtype=torch.float64), {
            "debt_cone_violation_before": 0.0,
            "debt_cone_violation_after": 0.0,
            "debt_cone_active_fraction": 0.0,
        }
    dirs = debt_directions.detach().to(dtype=torch.float64)
    if dirs.ndim == 1:
        dirs = dirs.reshape(-1, 1)
    norms = dirs.square().sum(dim=0).clamp_min(EPS)
    before = (dirs.transpose(0, 1) @ out).clamp_min(0.0)
    active = before > 1.0e-12
    for _ in range(max(1, int(max_passes))):
        dots = dirs.transpose(0, 1) @ out
        violations = dots.clamp_min(0.0)
        if float(violations.max().detach().cpu().item()) <= 1.0e-10:
            break
        correction = dirs @ (violations / norms)
        out = out - correction
    after = (dirs.transpose(0, 1) @ out).clamp_min(0.0)
    return out.reshape_as(candidate).to(dtype=torch.float64), {
        "debt_cone_violation_before": float(before.max().detach().cpu().item()),
        "debt_cone_violation_after": float(after.max().detach().cpu().item()),
        "debt_cone_active_fraction": float(active.to(dtype=torch.float64).mean().detach().cpu().item()),
    }


def lower_cvar_tensor(values: torch.Tensor, frac: float = 0.25) -> float:
    flat = values.detach().reshape(-1).to(dtype=torch.float64)
    flat = flat[torch.isfinite(flat)]
    if int(flat.numel()) == 0:
        return 0.0
    take = max(1, int(torch.ceil(torch.tensor(float(frac) * int(flat.numel()))).item()))
    ordered = torch.sort(flat).values
    return float(ordered[:take].mean().detach().cpu().item())


def raw_readout_multiplier(
    raw_energy: torch.Tensor,
    target_score: torch.Tensor | None = None,
    *,
    strength: float = 2.0,
    floor: float = 0.15,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build a raw-readout-sensitive diagonal gradient multiplier."""

    raw = raw_energy.detach().reshape(-1).to(dtype=torch.float64)
    if int(raw.numel()) == 0:
        return raw, {"raw_readout_multiplier_active_fraction": 0.0, "raw_readout_multiplier_cvar25": 0.0}
    raw_norm = raw / raw.max().clamp_min(EPS)
    if target_score is not None and int(target_score.numel()) == int(raw.numel()):
        tgt = target_score.detach().reshape(-1).to(dtype=torch.float64)
        tgt = tgt / tgt.max().clamp_min(EPS)
        score = 0.65 * raw_norm + 0.35 * tgt
    else:
        score = raw_norm
    multiplier = 1.0 + float(strength) * score
    low = raw_norm < float(floor)
    multiplier = torch.where(low, multiplier * 0.35, multiplier)
    return multiplier, {
        "raw_readout_multiplier_active_fraction": float((raw_norm >= float(floor)).to(dtype=torch.float64).mean().detach().cpu().item()),
        "raw_readout_multiplier_cvar25": lower_cvar_tensor(raw_norm, 0.25),
        "raw_readout_multiplier_min": float(multiplier.min().detach().cpu().item()),
        "raw_readout_multiplier_max": float(multiplier.max().detach().cpu().item()),
    }


@dataclass
class EdgeNaturalResidualState:
    """Per-parameter state for edge-natural residual gradient transforms."""

    raw_multiplier: torch.Tensor | None = None
    own_basis: torch.Tensor | None = None
    debt_directions: torch.Tensor | None = None
    enabled: bool = True


class EdgeNaturalResidualOptimizer:
    """Optimizer wrapper that transforms gradients inside ``step``."""

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        lr: float,
        weight_decay: float,
        edge_states: dict[str, EdgeNaturalResidualState] | None = None,
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
        residual, rdiag = own_reference_residual_project(work, state.own_basis.to(work.device) if state.own_basis is not None else None)
        projected, ddiag = debt_cone_project(residual, state.debt_directions.to(work.device) if state.debt_directions is not None else None)
        diag.update(rdiag)
        diag.update(ddiag)
        return projected.to(dtype=grad.dtype), diag

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
