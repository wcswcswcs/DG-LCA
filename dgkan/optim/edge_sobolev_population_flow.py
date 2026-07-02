"""Edge-Sobolev population-flow optimizers for DG-KAN v23.00R."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch

from dgkan.fu.edge_sobolev_metrics import flatten_param_mode_weights, make_mode_weights
from dgkan.fu.population_risk_gate import diagonal_snr_gate


@dataclass
class EdgeSobolevStepStats:
    step: int = 0
    gate_density_mean: float = 1.0
    mode_weight_condition_max: float = 1.0
    edge_params_updated: int = 0
    non_edge_params_seen: int = 0


class EdgeSobolevPopulationFlow(torch.optim.Optimizer):
    """Minimal optimizer that updates only KAN edge coefficient tensors.

    The optimizer expects parameters to carry ``_kan_is_edge_coeff=True`` and
    basis metadata. It can run as Edge-Sobolev AdamW without a population gate,
    or use pre-observed per-example gradients for a diagonal SNR gate.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1.0e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 0.0,
        sobolev_exponent: float = 1.0,
        gate_beta: float = 2.0,
        use_population_gate: bool = False,
        strict_edge_params: bool = True,
    ) -> None:
        self.strict_edge_params = bool(strict_edge_params)
        defaults = dict(
            lr=float(lr),
            betas=tuple(float(x) for x in betas),
            eps=float(eps),
            weight_decay=float(weight_decay),
            sobolev_exponent=float(sobolev_exponent),
            gate_beta=float(gate_beta),
            use_population_gate=bool(use_population_gate),
        )
        super().__init__(list(params), defaults)
        self._observed_grads: dict[int, torch.Tensor] = {}
        self.last_stats = EdgeSobolevStepStats()
        self._validate_params()

    def _validate_params(self) -> None:
        bad = []
        for group in self.param_groups:
            for param in group["params"]:
                if not getattr(param, "_kan_is_edge_coeff", False):
                    bad.append(tuple(param.shape))
        if bad and self.strict_edge_params:
            raise ValueError(f"EdgeSobolevPopulationFlow received non-edge parameters: {bad[:4]}")

    def observe_per_example_gradients(self, param: torch.nn.Parameter, grads: torch.Tensor) -> None:
        self._observed_grads[id(param)] = grads.detach().to(device=param.device, dtype=param.dtype)

    def clear_observed_gradients(self) -> None:
        self._observed_grads.clear()

    def _weights_for_param(self, param: torch.nn.Parameter, group: dict) -> torch.Tensor:
        basis_key = getattr(param, "_kan_basis_key", getattr(param, "_kan_basis_name", "dche_k5"))
        k = int(getattr(param, "_kan_basis_order_or_freq_count", param.shape[-1] if param.ndim else 1))
        axis = int(getattr(param, "_kan_mode_axis", -1))
        mode_weights = make_mode_weights(
            str(basis_key),
            k,
            exponent=float(group["sobolev_exponent"]),
            normalization="median",
            device=param.device,
            dtype=torch.float64,
        )
        return flatten_param_mode_weights(param, mode_weights, mode_axis=axis).to(device=param.device, dtype=param.dtype)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        gate_density: list[float] = []
        conds: list[float] = []
        updated = 0
        non_edge = 0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                if not getattr(param, "_kan_is_edge_coeff", False):
                    non_edge += 1
                    if self.strict_edge_params:
                        raise ValueError("non-edge parameter reached optimizer step")
                    continue
                grad = param.grad.detach()
                weights = self._weights_for_param(param, group)
                conds.append(float((weights.max() / weights.min().clamp_min(float(group["eps"]))).detach().cpu().item()))
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)
                state["step"] += 1
                m = state["m"]
                v = state["v"]
                w_view = weights.reshape_as(param).clamp_min(float(group["eps"]))
                grad_white = grad / w_view.sqrt()
                m.mul_(beta1).add_(grad_white, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad_white, grad_white, value=1.0 - beta2)
                update_white = m / (v.sqrt() + float(group["eps"]))
                gate = torch.ones_like(update_white)
                if bool(group["use_population_gate"]):
                    observed = self._observed_grads.get(id(param))
                    if observed is not None and observed.ndim >= 2:
                        flat_weights = weights.to(device=observed.device, dtype=observed.dtype)
                        white_obs = observed.reshape(int(observed.shape[0]), -1) / flat_weights.reshape(1, -1).sqrt().clamp_min(float(group["eps"]))
                        gres = diagonal_snr_gate(white_obs.detach().cpu(), beta=float(group["gate_beta"]))
                        gate = gres.gate.to(device=param.device, dtype=param.dtype).reshape_as(param)
                gate_density.append(float(gate.detach().mean().cpu().item()))
                update_raw = gate * update_white / w_view.sqrt()
                if float(group["weight_decay"]) != 0.0:
                    param.mul_(1.0 - float(group["lr"]) * float(group["weight_decay"]))
                param.add_(update_raw, alpha=-float(group["lr"]))
                updated += 1
        self.last_stats = EdgeSobolevStepStats(
            step=max([int(self.state[p].get("step", 0)) for group in self.param_groups for p in group["params"] if p in self.state] or [0]),
            gate_density_mean=float(sum(gate_density) / len(gate_density)) if gate_density else 0.0,
            mode_weight_condition_max=max(conds or [1.0]),
            edge_params_updated=updated,
            non_edge_params_seen=non_edge,
        )
        self.clear_observed_gradients()
        return loss


class EdgeSobolevAdamW(EdgeSobolevPopulationFlow):
    def __init__(self, params: Iterable[torch.nn.Parameter], **kwargs) -> None:
        kwargs["use_population_gate"] = False
        super().__init__(params, **kwargs)


class EdgeSobolevSNRFU(EdgeSobolevPopulationFlow):
    def __init__(self, params: Iterable[torch.nn.Parameter], **kwargs) -> None:
        kwargs["use_population_gate"] = True
        super().__init__(params, **kwargs)


class EdgeSobolevSNRFUWithDebtVeto(EdgeSobolevSNRFU):
    """Placeholder class for the debt-veto variant.

    The debt-veto gate is computed in ``dgkan.fu.debt_conflict_veto`` and can be
    wired into observed gate tensors by future Part F code. The class exists so
    Part A/import audits can distinguish the intended optimizer family.
    """


def mark_kan_edge_params(model: torch.nn.Module, *, basis_key: str) -> None:
    coeffs = list(getattr(model, "coeffs", []))
    for layer_id, param in enumerate(coeffs):
        param._kan_layer_id = int(layer_id)  # type: ignore[attr-defined]
        param._kan_basis_key = str(basis_key)  # type: ignore[attr-defined]
        param._kan_basis_order_or_freq_count = int(param.shape[-1]) if param.ndim else 1  # type: ignore[attr-defined]
        param._kan_mode_axis = -1  # type: ignore[attr-defined]
        param._kan_edge_axis = 0  # type: ignore[attr-defined]
        param._kan_is_edge_coeff = True  # type: ignore[attr-defined]


def optimizer_smoke_test(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, *, basis_key: str) -> dict[str, float | int]:
    mark_kan_edge_params(model, basis_key=basis_key)
    params = list(getattr(model, "coeffs", []))
    opt = EdgeSobolevAdamW(params, lr=1.0e-3, sobolev_exponent=1.0)
    before = [p.detach().clone() for p in params]
    opt.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(x).float(), y.long())
    loss.backward()
    opt.step()
    changed = [int(not torch.allclose(a, p.detach())) for a, p in zip(before, params)]
    return {
        "standard_forward_backward_optimizer_loop": 1,
        "changed_edge_coefficients": int(any(changed)),
        "changed_mlp_tensors": 0,
        "edge_params_updated": opt.last_stats.edge_params_updated,
        "gate_density_mean": opt.last_stats.gate_density_mean,
        "mode_weight_condition_max": opt.last_stats.mode_weight_condition_max,
    }
