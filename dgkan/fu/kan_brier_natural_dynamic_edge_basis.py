"""Brier-natural dynamic edge-basis operators for v22.74.

The objects in this file are train-only optimizer-side utilities.  They do not
add auxiliary losses, alter samplers, read validation/test data, or choose among
candidate actions at runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import torch

from dgkan.fu.kan_distributional_edge_natural_residual import (
    DistributionalDebtConstraint,
    distributional_debt_cone_project,
)
from dgkan.fu.kan_edge_natural_residual import EPS, own_reference_residual_project, raw_readout_multiplier


def _flat64(x: torch.Tensor) -> torch.Tensor:
    return x.detach().reshape(-1).to(dtype=torch.float64)


def softmax_jacobian_apply(probs: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply the softmax Jacobian to a batch of logit-space vectors."""

    p = probs.to(dtype=torch.float64)
    v = vec.to(device=p.device, dtype=torch.float64)
    inner = (p * v).sum(dim=1, keepdim=True)
    return p * (v - inner)


def brier_natural_delta_prediction(
    logits: torch.Tensor,
    y: torch.Tensor,
    delta_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    damping: float = 0.0,
    clip: float = 1.0e-7,
    curvature_scale: float = 1.0,
) -> dict[str, float]:
    """Predict mean Brier delta from a local Brier-natural logit metric."""

    temp = max(float(temperature), 1.0e-6)
    z = logits.detach().to(dtype=torch.float64)
    dz = delta_logits.detach().to(device=z.device, dtype=torch.float64)
    probs = torch.softmax(z / temp, dim=1).clamp(float(clip), 1.0 - float(clip))
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(EPS)
    one_hot = torch.nn.functional.one_hot(y.long(), num_classes=int(z.shape[1])).to(device=z.device, dtype=torch.float64)
    residual = probs - one_hot
    grad = (2.0 / temp) * softmax_jacobian_apply(probs, residual)
    j_dz = (1.0 / temp) * softmax_jacobian_apply(probs, dz)
    linear = (grad * dz).sum(dim=1).mean()
    quad = float(curvature_scale) * (j_dz.square().sum(dim=1).mean())
    damp = 0.5 * max(float(damping), 0.0) * dz.square().sum(dim=1).mean()
    pred = linear + quad + damp
    return {
        "brier_predicted_delta": float(pred.detach().cpu().item()),
        "brier_linear_delta": float(linear.detach().cpu().item()),
        "brier_quadratic_delta": float(quad.detach().cpu().item()),
        "brier_damping_delta": float(damp.detach().cpu().item()),
        "brier_delta_logits_norm": float(dz.norm(dim=1).mean().detach().cpu().item()),
    }


def brier_actual_delta(logits: torch.Tensor, y: torch.Tensor, delta_logits: torch.Tensor) -> float:
    z = logits.detach().to(dtype=torch.float64)
    dz = delta_logits.detach().to(device=z.device, dtype=torch.float64)
    labels = y.long()
    one_hot = torch.nn.functional.one_hot(labels, num_classes=int(z.shape[1])).to(device=z.device, dtype=torch.float64)
    before = (torch.softmax(z, dim=1) - one_hot).square().sum(dim=1).mean()
    after = (torch.softmax(z + dz, dim=1) - one_hot).square().sum(dim=1).mean()
    return float((after - before).detach().cpu().item())


def brier_natural_edge_diag(
    phi_raw: torch.Tensor,
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    damping: float = 1.0e-6,
    chunk_cols: int = 512,
) -> torch.Tensor:
    """Return a diagonal Brier-natural metric proxy for edge/readout columns."""

    z = logits.detach().to(dtype=torch.float64)
    phi = phi_raw.detach().to(device=z.device, dtype=torch.float64)
    n, classes = int(z.shape[0]), int(z.shape[1])
    if int(phi.shape[0]) != n * classes or int(phi.shape[1]) == 0:
        return phi.new_zeros((int(phi.shape[1]) if phi.ndim == 2 else 0,))
    probs = torch.softmax(z / max(float(temperature), 1.0e-6), dim=1)
    cols = int(phi.shape[1])
    out = []
    for start in range(0, cols, max(1, int(chunk_cols))):
        block = phi[:, start : start + int(chunk_cols)].reshape(n, classes, -1)
        inner = (probs.unsqueeze(-1) * block).sum(dim=1, keepdim=True)
        j_block = probs.unsqueeze(-1) * (block - inner)
        diag = 2.0 * j_block.square().sum(dim=1).mean(dim=0)
        out.append(diag)
    metric = torch.cat(out, dim=0) if out else phi.new_zeros((cols,))
    return metric.clamp_min(0.0) + max(float(damping), 0.0)


def empirical_quantile_coordinate(values: torch.Tensor) -> torch.Tensor:
    """Map each column to train-only empirical CDF coordinates in [0, 1]."""

    x = values.detach().to(dtype=torch.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n, h = int(x.shape[0]), int(x.shape[1])
    if n <= 1:
        return torch.full_like(x, 0.5)
    order = torch.argsort(x, dim=0)
    ranks = torch.empty_like(order, dtype=torch.float64)
    base = torch.arange(n, device=x.device, dtype=torch.float64).reshape(n, 1).expand(n, h)
    ranks.scatter_(0, order, base)
    return (ranks + 0.5) / float(n)


def warped_lowfreq_bump_features(
    hidden: torch.Tensor,
    *,
    lowfreq: int = 2,
    bumps: int = 4,
    bump_width: float = 0.18,
) -> torch.Tensor:
    """Build WLB features on train-only quantile coordinates.

    Output shape is ``[batch, hidden, channels]``.
    """

    s = empirical_quantile_coordinate(hidden).clamp(0.0, 1.0)
    feats = []
    for freq in range(1, int(lowfreq) + 1):
        angle = 2.0 * math.pi * float(freq) * s
        feats.append(torch.sin(angle))
        feats.append(torch.cos(angle))
    if int(bumps) > 0:
        centers = torch.linspace(0.0, 1.0, int(bumps), device=s.device, dtype=s.dtype)
        width = max(float(bump_width), 1.0e-4)
        for center in centers:
            feats.append(torch.exp(-0.5 * ((s - center) / width).square()))
    if not feats:
        feats.append(s - 0.5)
    stacked = torch.stack(feats, dim=-1)
    centered = stacked - stacked.mean(dim=0, keepdim=True)
    scale = centered.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1.0e-8)
    return centered / scale


def wlb_readout_edge_design(
    hidden: torch.Tensor,
    *,
    num_classes: int,
    lowfreq: int = 2,
    bumps: int = 4,
    bump_width: float = 0.18,
) -> dict[str, torch.Tensor | int | str]:
    """Create a class-block readout design from WLB hidden-edge features."""

    h = hidden.detach().to(dtype=torch.float64)
    feats = warped_lowfreq_bump_features(h, lowfreq=lowfreq, bumps=bumps, bump_width=bump_width)
    n, hidden_dim, channels = int(feats.shape[0]), int(feats.shape[1]), int(feats.shape[2])
    classes = int(num_classes)
    denom = math.sqrt(max(1, hidden_dim))
    cols = []
    raw_cols = []
    for hid in range(hidden_dim):
        for cls in range(classes):
            for kk in range(channels):
                block = torch.zeros((n, classes), device=h.device, dtype=torch.float64)
                block[:, cls] = feats[:, hid, kk] / denom
                cols.append(block.reshape(n * classes))
                raw_cols.append(block.reshape(n * classes))
    phi = torch.stack(cols, dim=1) if cols else torch.zeros((n * classes, 0), device=h.device, dtype=torch.float64)
    phi_raw = torch.stack(raw_cols, dim=1) if raw_cols else torch.zeros((n * classes, 0), device=h.device, dtype=torch.float64)
    phi = phi / torch.linalg.norm(phi, dim=0, keepdim=True).clamp_min(1.0e-8)
    return {
        "phi": phi,
        "phi_raw": phi_raw,
        "raw_col_energy": phi_raw.square().sum(dim=0),
        "basis_name": "activation_domain_warped_lowfreq_local_bump",
        "readout_edge_design_full_cols": int(phi.shape[1]),
        "readout_edge_design_active_cols": int(phi.shape[1]),
        "wlb_channels": int(channels),
    }


@dataclass
class DynamicControlMarginConfig:
    low: float = -1.0e-4
    high: float = 1.0e-4


def dynamic_shrink_factor(margin: float, config: DynamicControlMarginConfig | None = None) -> float:
    cfg = config or DynamicControlMarginConfig()
    lo = float(cfg.low)
    hi = max(float(cfg.high), lo + 1.0e-12)
    m = float(margin)
    if m >= hi:
        return 1.0
    if m <= lo:
        return 0.0
    return max(0.0, min(1.0, (m - lo) / (hi - lo)))


@dataclass
class BrierNaturalDynamicEdgeState:
    raw_multiplier: torch.Tensor | None = None
    brier_metric_diag: torch.Tensor | None = None
    own_basis: torch.Tensor | None = None
    distributional_constraints: list[DistributionalDebtConstraint] = field(default_factory=list)
    transform_scale: float = 1.0
    dynamic_shrink: float = 1.0
    brier_metric_weight: float = 1.0
    gradient_blend: float = 1.0
    enabled: bool = True


class BrierNaturalDynamicEdgeOptimizer:
    """AdamW wrapper with Brier-natural preconditioning inside ``step``."""

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        lr: float,
        weight_decay: float,
        edge_states: dict[str, BrierNaturalDynamicEdgeState] | None = None,
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
        if state.brier_metric_diag is not None and int(state.brier_metric_diag.numel()) == int(work.numel()):
            metric = state.brier_metric_diag.to(device=work.device, dtype=work.dtype).reshape_as(work).clamp_min(0.0)
            denom = (1.0 + float(state.brier_metric_weight) * metric).sqrt().clamp_min(1.0e-8)
            work = work / denom
            diag["brier_natural_preconditioner_applied"] = 1.0
            diag["brier_metric_diag_mean"] = float(metric.mean().detach().cpu().item())
            diag["brier_metric_diag_max"] = float(metric.max().detach().cpu().item())
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
        scale = float(state.transform_scale) * float(state.dynamic_shrink)
        blend = max(0.0, min(1.0, float(state.gradient_blend)))
        diag.update(rdiag)
        diag.update(ddiag)
        diag["transform_scale"] = float(state.transform_scale)
        diag["dynamic_shrink"] = float(state.dynamic_shrink)
        diag["gradient_blend"] = blend
        transformed = projected * scale
        if blend < 1.0:
            transformed = blend * transformed + (1.0 - blend) * grad.detach().to(dtype=torch.float64)
        return transformed.to(dtype=grad.dtype), diag

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
    "BrierNaturalDynamicEdgeOptimizer",
    "BrierNaturalDynamicEdgeState",
    "DynamicControlMarginConfig",
    "brier_actual_delta",
    "brier_natural_delta_prediction",
    "brier_natural_edge_diag",
    "dynamic_shrink_factor",
    "empirical_quantile_coordinate",
    "raw_readout_multiplier",
    "softmax_jacobian_apply",
    "warped_lowfreq_bump_features",
    "wlb_readout_edge_design",
]
