"""Continuous edge-state controlled gradient flow.

This module implements a fixed optimizer-step state flow. It keeps per-edge
signal, uncertainty, domain, and debt-dual state and emits a deterministic
smooth gradient transform for all registered edge parameters. It does not
select a candidate update and does not write parameters directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Iterable

import torch


@dataclass
class EdgeStateFlowConfig:
    beta_signal: float = 0.20
    beta_diffusion: float = 0.10
    beta_domain: float = 0.05
    eta_dual: float = 0.05
    debt_budget: float = 0.05
    external_debt_blend: float = 0.0
    external_debt_cotangent_blend: float = 0.0
    debt_cotangent_max_norm_ratio: float = 1.0
    signal_blend: float = 0.65
    shrink_floor: float = 0.10
    shrink_tau: float = 0.25
    shrink_temp: float = 0.10
    max_norm_ratio: float = 1.50
    eps: float = 1.0e-12
    control_mode: str = "normal"
    random_seed: int = 0


@dataclass
class _EdgeState:
    signal: torch.Tensor
    diffusion: torch.Tensor
    domain: torch.Tensor
    debt_dual: torch.Tensor
    steps: int = 0
    last_shrink_min: float = 1.0
    last_shrink_max: float = 1.0


@dataclass
class EdgeStateFlowTrace:
    step_calls: int = 0
    transform_calls: int = 0
    state_update_calls: int = 0
    gradients_transformed: int = 0
    optimizer_owned_gradient_transform_pass: int = 0
    direct_parameter_update_attempted: int = 0
    inside_optimizer_step_violations: int = 0
    candidate_update_selected_runtime: int = 0
    runtime_argmax_used: int = 0
    runtime_topk_used: int = 0
    edge_velocity_emitted_calls: int = 0
    all_edges_state_updated_count: int = 0
    registered_edge_count: int = 0
    soft_shrink_min_values: list[float] = field(default_factory=list)
    soft_shrink_max_values: list[float] = field(default_factory=list)
    norm_ratio_values: list[float] = field(default_factory=list)
    debt_dual_values: list[float] = field(default_factory=list)
    signal_norm_values: list[float] = field(default_factory=list)
    diffusion_norm_values: list[float] = field(default_factory=list)
    domain_drift_values: list[float] = field(default_factory=list)
    external_debt_proxy_values: list[float] = field(default_factory=list)
    external_debt_cotangent_norm_ratios: list[float] = field(default_factory=list)
    transform_time_ms: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "edge_state_updated_every_step": int(self.state_update_calls >= self.registered_edge_count and self.registered_edge_count > 0),
            "edge_metric_updated_every_step_or_cadence": int(self.state_update_calls >= self.registered_edge_count and self.registered_edge_count > 0),
            "edge_signal_state_updated_every_step": int(self.state_update_calls >= self.registered_edge_count and self.registered_edge_count > 0),
            "debt_dual_state_updated_every_step": int(self.state_update_calls >= self.registered_edge_count and self.registered_edge_count > 0),
            "edge_velocity_emitted_every_step": int(self.edge_velocity_emitted_calls >= self.registered_edge_count and self.registered_edge_count > 0),
            "candidate_update_selected_runtime": self.candidate_update_selected_runtime,
            "runtime_argmax_used": self.runtime_argmax_used,
            "runtime_topk_used": self.runtime_topk_used,
            "optimizer_owned_gradient_transform_pass": self.optimizer_owned_gradient_transform_pass,
            "direct_parameter_update_attempted": self.direct_parameter_update_attempted,
            "inside_optimizer_step_violations": self.inside_optimizer_step_violations,
            "all_edges_state_updated_count": self.all_edges_state_updated_count,
            "registered_edge_count": self.registered_edge_count,
            "soft_shrink_min": min(self.soft_shrink_min_values) if self.soft_shrink_min_values else 1.0,
            "soft_shrink_max": max(self.soft_shrink_max_values) if self.soft_shrink_max_values else 1.0,
            "state_EMA_stability": 1.0 / (1.0 + _mean(self.diffusion_norm_values)),
            "domain_transport_continuity": 1.0 / (1.0 + _mean(self.domain_drift_values)),
            "debt_dual_nonnegative": int((min(self.debt_dual_values) if self.debt_dual_values else 0.0) >= 0.0),
            "external_debt_proxy_mean": _mean(self.external_debt_proxy_values),
            "external_debt_cotangent_norm_ratio_mean": _mean(self.external_debt_cotangent_norm_ratios),
            "edge_signal_state_nontrivial": int(_mean(self.signal_norm_values) > 0.0),
            "esfg_transform_calls": self.transform_calls,
            "esfg_state_update_calls": self.state_update_calls,
            "esfg_gradients_transformed": self.gradients_transformed,
            "norm_Pg_over_norm_g": _mean(self.norm_ratio_values, 1.0),
            "esfg_transform_time_ms": self.transform_time_ms,
        }


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return sum(clean) / len(clean) if clean else float(default)


class EdgeStateFlow:
    """Fixed continuous state-flow gradient transform for edge parameters."""

    def __init__(self, edge_params: Iterable[torch.nn.Parameter], config: EdgeStateFlowConfig | None = None) -> None:
        self.edge_params = [p for p in edge_params if getattr(p, "requires_grad", False)]
        self.config = config or EdgeStateFlowConfig()
        self.trace = EdgeStateFlowTrace(registered_edge_count=len(self.edge_params))
        self._state: dict[int, _EdgeState] = {}
        self._inside_optimizer_step = False
        self._gen_by_device: dict[str, torch.Generator] = {}
        self._external_debt_proxy: torch.Tensor | None = None
        self._external_debt_cotangents: dict[int, torch.Tensor] = {}
        for p in self.edge_params:
            self._state[id(p)] = _EdgeState(
                signal=torch.zeros_like(p.detach()),
                diffusion=torch.zeros_like(p.detach()),
                domain=torch.zeros_like(p.detach()),
                debt_dual=torch.zeros((), device=p.device, dtype=p.dtype),
            )

    def owns_param(self, param: torch.nn.Parameter) -> bool:
        return any(param is p for p in self.edge_params)

    def set_inside_optimizer_step(self, value: bool) -> None:
        self._inside_optimizer_step = bool(value)
        if value:
            self.trace.step_calls += 1
            self.trace.all_edges_state_updated_count = 0

    def observe_debt_proxy(self, value: float | torch.Tensor) -> None:
        if isinstance(value, torch.Tensor):
            proxy = value.detach().mean()
        else:
            proxy = torch.as_tensor(float(value))
        self._external_debt_proxy = proxy.clamp_min(0.0)

    def observe_debt_cotangent(self, cotangents: Iterable[torch.Tensor | None]) -> None:
        """Provide train-stream debt cotangents to be consumed inside step()."""

        observed: dict[int, torch.Tensor] = {}
        for param, cotangent in zip(self.edge_params, cotangents):
            if cotangent is None:
                continue
            observed[id(param)] = cotangent.detach().clone()
        self._external_debt_cotangents = observed

    def _generator(self, device: torch.device) -> torch.Generator:
        key = str(device)
        gen = self._gen_by_device.get(key)
        if gen is None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(self.config.random_seed) + (0 if device.type == "cpu" else 1009 * int(device.index or 0)))
            self._gen_by_device[key] = gen
        return gen

    def _controlled_gradient(self, grad: torch.Tensor, state: _EdgeState, debt_cotangent: torch.Tensor | None) -> torch.Tensor:
        cfg = self.config
        mode = str(cfg.control_mode)
        base_signal = state.signal
        if mode == "frozen_signal":
            base_signal = torch.zeros_like(state.signal)
        elif mode in {"same_domain_random", "same_debt_random", "shuffled_signal"}:
            gen = self._generator(grad.device)
            noise = torch.randn(tuple(grad.shape), generator=gen, device=grad.device, dtype=grad.dtype)
            noise = noise * (state.signal.detach().norm() / noise.detach().norm().clamp_min(float(cfg.eps)))
            base_signal = noise
        rho = state.signal.detach().square() / (state.diffusion.detach().abs() + float(cfg.eps))
        shrink = float(cfg.shrink_floor) + (1.0 - float(cfg.shrink_floor)) * torch.sigmoid((rho - float(cfg.shrink_tau)) / max(float(cfg.shrink_temp), float(cfg.eps)))
        shrink = shrink.clamp_min(float(cfg.shrink_floor))
        state.last_shrink_min = float(shrink.min().detach().cpu().item())
        state.last_shrink_max = float(shrink.max().detach().cpu().item())
        debt_factor = 1.0 / (1.0 + state.debt_dual.to(device=grad.device, dtype=grad.dtype))
        debt_term = torch.zeros_like(grad)
        if debt_cotangent is not None and float(cfg.external_debt_cotangent_blend) > 0.0:
            debt_term = debt_cotangent.to(device=grad.device, dtype=grad.dtype)
            g_norm = grad.detach().norm().clamp_min(float(cfg.eps))
            d_norm = debt_term.detach().norm().clamp_min(float(cfg.eps))
            max_debt_norm = float(cfg.debt_cotangent_max_norm_ratio) * g_norm
            if float(d_norm.item()) > float(max_debt_norm.item()):
                debt_term = debt_term * (max_debt_norm / d_norm)
                d_norm = debt_term.detach().norm().clamp_min(float(cfg.eps))
            self.trace.external_debt_cotangent_norm_ratios.append(float((d_norm / g_norm).detach().cpu().item()))
            debt_term = float(cfg.external_debt_cotangent_blend) * debt_term
        return debt_factor * (
            grad
            + debt_term
            + float(cfg.signal_blend) * shrink * base_signal.to(device=grad.device, dtype=grad.dtype)
        )

    def transform(self, grad: torch.Tensor, param: torch.nn.Parameter, _group: dict | None = None) -> torch.Tensor:
        start = time.perf_counter()
        self.trace.transform_calls += 1
        if not self._inside_optimizer_step:
            self.trace.inside_optimizer_step_violations += 1
            return grad
        state = self._state.get(id(param))
        if state is None:
            return grad
        cfg = self.config
        g = grad.detach()
        previous_signal = state.signal
        state.signal = (1.0 - float(cfg.beta_signal)) * state.signal + float(cfg.beta_signal) * g
        residual = g - state.signal
        state.diffusion = (1.0 - float(cfg.beta_diffusion)) * state.diffusion + float(cfg.beta_diffusion) * residual.square()
        grad_abs = g.abs()
        prev_domain = state.domain
        state.domain = (1.0 - float(cfg.beta_domain)) * state.domain + float(cfg.beta_domain) * grad_abs
        debt_proxy = grad.square().mean().detach()
        if self._external_debt_proxy is not None and float(cfg.external_debt_blend) > 0.0:
            external = self._external_debt_proxy.to(device=grad.device, dtype=grad.dtype)
            blend = max(0.0, min(1.0, float(cfg.external_debt_blend)))
            debt_proxy = (1.0 - blend) * debt_proxy + blend * external
            self.trace.external_debt_proxy_values.append(float(external.detach().cpu().item()))
        state.debt_dual = torch.clamp(state.debt_dual + float(cfg.eta_dual) * (debt_proxy - float(cfg.debt_budget)), min=0.0)
        state.steps += 1

        debt_cotangent = self._external_debt_cotangents.get(id(param))
        new_grad = self._controlled_gradient(grad, state, debt_cotangent)
        g_norm = grad.detach().norm().clamp_min(float(cfg.eps))
        ng_norm = new_grad.detach().norm().clamp_min(float(cfg.eps))
        max_norm = float(cfg.max_norm_ratio) * g_norm
        if float(ng_norm.item()) > float(max_norm.item()):
            new_grad = new_grad * (max_norm / ng_norm)
            ng_norm = new_grad.detach().norm().clamp_min(float(cfg.eps))

        self.trace.state_update_calls += 1
        self.trace.all_edges_state_updated_count += 1
        self.trace.gradients_transformed += 1
        self.trace.edge_velocity_emitted_calls += 1
        self.trace.optimizer_owned_gradient_transform_pass = 1
        self.trace.soft_shrink_min_values.append(state.last_shrink_min)
        self.trace.soft_shrink_max_values.append(state.last_shrink_max)
        self.trace.norm_ratio_values.append(float((ng_norm / g_norm).detach().cpu().item()))
        self.trace.debt_dual_values.append(float(state.debt_dual.detach().cpu().item()))
        self.trace.signal_norm_values.append(float(state.signal.detach().norm().cpu().item()))
        self.trace.diffusion_norm_values.append(float(state.diffusion.detach().norm().cpu().item()))
        self.trace.domain_drift_values.append(float((state.domain - prev_domain).detach().norm().cpu().item()))
        self.trace.transform_time_ms += (time.perf_counter() - start) * 1000.0
        return new_grad

    def diagnostics(self) -> dict[str, float | int]:
        return self.trace.as_dict()


__all__ = ["EdgeStateFlow", "EdgeStateFlowConfig", "EdgeStateFlowTrace"]
