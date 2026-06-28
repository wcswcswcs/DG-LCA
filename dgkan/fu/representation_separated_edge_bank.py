"""Representation-separated edge-bank utilities for v22.79.

The optimizer wrapper in this module only transforms gradients inside
``step``.  It is intended for strict FC-PureKAN train-only experiments where an
upstream representation-separating transform on ``w1`` is coupled to a
downstream edge-bank transform on ``w2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from dgkan.fu.kan_conditional_edge_signal_metric import EPS, flat64, weighted_project
from dgkan.fu.kan_edge_bank_signal_basis import bank_additive_projection, metric_energy


def _metric(vector: torch.Tensor, metric_diag: torch.Tensor | None) -> torch.Tensor:
    v = flat64(vector)
    if metric_diag is None or int(metric_diag.numel()) != int(v.numel()):
        return torch.ones_like(v)
    return metric_diag.detach().to(device=v.device, dtype=torch.float64).reshape(-1).clamp_min(0.0) + EPS


def same_energy_orthogonal_control(vector: torch.Tensor, *, shift_fraction: int = 7) -> torch.Tensor:
    """Return a deterministic same-norm control with candidate axis removed."""

    v = flat64(vector)
    if int(v.numel()) == 0:
        return v
    shift = max(1, int(v.numel()) // max(2, int(shift_fraction)))
    ctrl = torch.roll(v, shifts=shift)
    coeff = (ctrl * v).sum() / v.square().sum().clamp_min(EPS)
    ctrl = ctrl - coeff * v
    return ctrl / ctrl.norm().clamp_min(EPS) * v.norm().clamp_min(EPS)


def separation_lift_to_downstream(
    upstream_vector: torch.Tensor,
    upstream_shape: torch.Size | tuple[int, ...],
    downstream_shape: torch.Size | tuple[int, ...],
    downstream_signal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Lift a ``w1`` separation direction into the downstream ``w2`` bank shape.

    This is a first-order train-only proxy: the upstream direction is compressed
    over incoming coordinates to hidden/basis coordinates, then coupled with the
    downstream class/basis signal.  It does not use validation/test data or
    candidate outcomes.
    """

    u_dims = tuple(int(x) for x in upstream_shape)
    d_dims = tuple(int(x) for x in downstream_shape)
    u = flat64(upstream_vector)
    total_u = 1
    for dim in u_dims:
        total_u *= max(1, dim)
    total_d = 1
    for dim in d_dims:
        total_d *= max(1, dim)
    if int(u.numel()) != total_u or len(u_dims) < 2 or len(d_dims) < 2:
        return torch.zeros(total_d, device=u.device, dtype=torch.float64)
    if len(u_dims) == 3:
        hidden_basis = u.reshape(u_dims).mean(dim=0)
    else:
        hidden_basis = u.reshape(u_dims)
    hidden = min(int(hidden_basis.shape[0]), int(d_dims[0]))
    basis = int(d_dims[-1]) if len(d_dims) >= 3 else 1
    if hidden <= 0:
        return torch.zeros(total_d, device=u.device, dtype=torch.float64)
    if int(hidden_basis.ndim) == 1:
        hidden_basis = hidden_basis.reshape(-1, 1)
    if int(hidden_basis.shape[1]) != basis:
        if int(hidden_basis.shape[1]) > basis:
            hidden_basis = hidden_basis[:, :basis]
        else:
            reps = int(torch.ceil(torch.tensor(float(basis) / float(max(1, hidden_basis.shape[1])))).item())
            hidden_basis = hidden_basis.repeat(1, reps)[:, :basis]
    out = torch.zeros(d_dims, device=u.device, dtype=torch.float64)
    hb = hidden_basis[:hidden].to(device=u.device, dtype=torch.float64)
    if len(d_dims) == 3:
        if downstream_signal is not None and int(flat64(downstream_signal).numel()) == total_d:
            ds = flat64(downstream_signal).to(device=u.device).reshape(d_dims)
            class_scale = ds[:hidden].mean(dim=(0, 2))
            if float(class_scale.abs().max().detach().cpu().item()) <= EPS:
                class_scale = torch.ones(int(d_dims[1]), device=u.device, dtype=torch.float64)
        else:
            class_scale = torch.ones(int(d_dims[1]), device=u.device, dtype=torch.float64)
        out[:hidden, :, :] = hb[:, None, :] * class_scale.reshape(1, -1, 1)
    else:
        out[:hidden, :] = hb[:hidden, : int(d_dims[1])]
    return out.reshape(-1)


def representation_separation_bank_stats(
    downstream_signal: torch.Tensor,
    separation_lift: torch.Tensor,
    metric_diag: torch.Tensor | None,
    downstream_shape: torch.Size | tuple[int, ...],
    *,
    blend: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Measure whether an upstream lift makes downstream bank signal additive."""

    sig = flat64(downstream_signal)
    lift = flat64(separation_lift).to(device=sig.device)
    if int(lift.numel()) != int(sig.numel()):
        lift = torch.zeros_like(sig)
    lift = lift / lift.norm().clamp_min(EPS) * sig.norm().clamp_min(EPS)
    pre_bank, pre = bank_additive_projection(sig, metric_diag, downstream_shape)
    post_signal = sig + float(blend) * lift
    post_bank, post = bank_additive_projection(post_signal, metric_diag, downstream_shape)
    pre_r2 = float(pre.get("edge_bank_anova_explained", 0.0))
    post_r2 = float(post.get("edge_bank_anova_explained", 0.0))
    pre_int = float(pre.get("interaction_residual_fraction", 1.0))
    post_int = float(post.get("interaction_residual_fraction", 1.0))
    control = same_energy_orthogonal_control(lift)
    control_signal = sig + float(blend) * control
    _ctrl_bank, ctrl = bank_additive_projection(control_signal, metric_diag, downstream_shape)
    ctrl_r2 = float(ctrl.get("edge_bank_anova_explained", 0.0))
    return flat64(post_bank), {
        "pre_bank_R2": pre_r2,
        "post_bank_R2": post_r2,
        "bank_R2_gain": post_r2 - pre_r2,
        "pre_interaction_residual": pre_int,
        "post_interaction_residual": post_int,
        "interaction_residual_reduction": pre_int - post_int,
        "same_separation_energy_random_R2": ctrl_r2,
        "same_separation_energy_random_gap": post_r2 - ctrl_r2,
        "separation_lift_energy": float(metric_energy(lift, _metric(sig, metric_diag)).detach().cpu().item()),
        "bank_condition_number": float(post.get("bank_condition_number", 0.0)),
    }


@dataclass
class RepresentationSeparatedEdgeBankState:
    domain_basis: torch.Tensor | None = None
    control_basis: torch.Tensor | None = None
    metric_diag: torch.Tensor | None = None
    transform_scale: float = 1.0
    conditional_alpha: float = 1.0
    ridge: float = 1.0e-6
    role: str = "downstream_edge_bank"
    enabled: bool = True


class RepresentationSeparatedEdgeBankOptimizer:
    """Wrap an optimizer and transform upstream/downstream gradients in step."""

    def __init__(
        self,
        base_optimizer: object,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        states: dict[str, RepresentationSeparatedEdgeBankState] | None = None,
    ) -> None:
        self.base_optimizer = base_optimizer
        self.named_parameters = [(name, param) for name, param in named_parameters]
        self.states = states or {}
        self.transform_calls = 0
        self.transformed_gradient_tensors = 0
        self.upstream_tensors = 0
        self.downstream_tensors = 0
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
            if "upstream" in state.role:
                self.upstream_tensors += 1
            if "downstream" in state.role or "bank" in state.role:
                self.downstream_tensors += 1
            merged.update({f"{name}.domain_{k}": v for k, v in ddiag.items()})
            merged.update({f"{name}.control_{k}": v for k, v in cdiag.items()})
            merged[f"{name}.role_upstream"] = float("upstream" in state.role)
            merged[f"{name}.role_downstream"] = float("downstream" in state.role or "bank" in state.role)
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
        out["two_layer_gradient_transform_tensors"] = float(self.upstream_tensors + self.downstream_tensors)
        out["upstream_separation_metric_applied"] = float(self.upstream_tensors > 0)
        out["edge_bank_anova_metric_applied"] = float(self.downstream_tensors > 0)
        out["domain_nuisance_projection_applied"] = float(
            any(v > 0.5 for k, v in self.last_diagnostics.items() if k.endswith("domain_nuisance_projection_applied"))
        )
        out["control_contrastive_metric_applied"] = float(
            any(v > 0.5 for k, v in self.last_diagnostics.items() if k.endswith("control_contrastive_metric_applied"))
        )
        return out


__all__ = [
    "RepresentationSeparatedEdgeBankOptimizer",
    "RepresentationSeparatedEdgeBankState",
    "representation_separation_bank_stats",
    "same_energy_orthogonal_control",
    "separation_lift_to_downstream",
]
