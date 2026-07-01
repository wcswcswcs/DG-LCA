"""Controlled composite metric tube primitives for v22.90.

This module extends the v22.89 layer-composite metric machinery from exact
metric preservation to a bounded spectral tube around the initial composite
metric ``R0 = A.T @ C @ A``.  The implementation is intentionally explicit so
the experiment runner can audit each projection, radial solve and retraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Iterable

import torch

from dgkan.fu.layer_composite_metric import (
    EPS,
    composite_metric,
    condition_number,
    matrix_inv_sqrt,
    matrix_sqrt,
    matrix_to_w1,
    natural_tangent_velocity,
    relative_fro_error,
    solve_sylvester_spd,
    sym,
    w1_to_matrix,
)


@dataclass
class CompositeMetricTubeConfig:
    delta_spec: float = 0.40
    s_min: float = 0.60
    s_max: float = 1.80
    kappa_max: float = 1.0e4
    radial_velocity_fraction_cap: float = 0.25
    metric_state_beta: float = 0.05
    metric_update_interval: int = 10
    rho: float = 1.0e-8
    eps: float = EPS
    max_norm_ratio: float = 20.0
    random_seed: int = 0
    control_mode: str = "task"
    debt_cotangent_blend: float = 0.0
    debt_dual_lr: float = 0.05
    debt_budget: float = 0.0
    anti_windup_max: float = 10.0
    debt_cotangent_norm_cap: float = 1.0


@dataclass
class TubeProjectionResult:
    r_projected: torch.Tensor
    log_spec_distance: float
    trace_ratio: float
    condition: float
    clip_fraction: float
    trace_scale: float
    ridge_added: float
    projection_count: int


def _relative_eigs(center: torch.Tensor, r: torch.Tensor, eps: float = EPS) -> tuple[torch.Tensor, torch.Tensor]:
    c = sym(center).to(dtype=torch.float64)
    rr = sym(r).to(device=c.device, dtype=torch.float64)
    inv_sqrt = matrix_inv_sqrt(c, eps)
    m = sym(inv_sqrt @ rr @ inv_sqrt)
    vals, vecs = torch.linalg.eigh(m)
    vals = vals.clamp_min(float(eps))
    return vals, vecs


def log_spec_distance(center: torch.Tensor, r: torch.Tensor, eps: float = EPS) -> float:
    vals, _ = _relative_eigs(center, r, eps)
    return float(vals.log().abs().max().detach().cpu().item())


def project_spd_to_tube(
    center: torch.Tensor,
    r: torch.Tensor,
    config: CompositeMetricTubeConfig | None = None,
) -> TubeProjectionResult:
    cfg = config or CompositeMetricTubeConfig()
    device = center.device
    c0 = sym(center).to(device=device, dtype=torch.float64)
    rr = sym(r).to(device=device, dtype=torch.float64)
    sqrt_c0 = matrix_sqrt(c0, float(cfg.eps))
    vals, vecs = _relative_eigs(c0, rr, float(cfg.eps))
    logs = vals.log()
    clipped = logs.clamp(-float(cfg.delta_spec), float(cfg.delta_spec))
    clip_fraction = float((logs.ne(clipped)).to(dtype=torch.float64).mean().detach().cpu().item())
    out = sym(sqrt_c0 @ (vecs * clipped.exp().unsqueeze(0)) @ vecs.T @ sqrt_c0)
    trace0 = torch.trace(c0).clamp_min(float(cfg.eps))
    trace_ratio = torch.trace(out) / trace0
    scale = torch.ones((), device=device, dtype=torch.float64)
    if float(trace_ratio.detach().cpu().item()) < float(cfg.s_min):
        scale = torch.as_tensor(float(cfg.s_min), device=device, dtype=torch.float64) / trace_ratio.clamp_min(float(cfg.eps))
    elif float(trace_ratio.detach().cpu().item()) > float(cfg.s_max):
        scale = torch.as_tensor(float(cfg.s_max), device=device, dtype=torch.float64) / trace_ratio.clamp_min(float(cfg.eps))
    out = sym(out * scale)
    ridge_added = 0.0
    dim = int(out.shape[0])
    eye = torch.eye(dim, device=device, dtype=torch.float64)
    trace_unit = float((torch.trace(out).abs() / max(1, dim)).detach().cpu().item())
    for attempt in range(8):
        cond = condition_number(out, float(cfg.eps))
        if cond <= float(cfg.kappa_max) * 1.01:
            break
        ridge = max(trace_unit, float(cfg.eps)) * (10.0 ** (attempt - 8))
        ridge_added += float(ridge)
        out = sym(out + float(ridge) * eye)
    return TubeProjectionResult(
        r_projected=out,
        log_spec_distance=log_spec_distance(c0, out, float(cfg.eps)),
        trace_ratio=float((torch.trace(out) / trace0).detach().cpu().item()),
        condition=condition_number(out, float(cfg.eps)),
        clip_fraction=clip_fraction,
        trace_scale=float(scale.detach().cpu().item()),
        ridge_added=float(ridge_added),
        projection_count=1,
    )


def c_norm(delta: torch.Tensor, c: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    d = delta.to(dtype=torch.float64)
    cc = c.to(device=d.device, dtype=torch.float64)
    return torch.sqrt(torch.sum(d * (cc @ d)).clamp_min(float(eps)))


def radial_velocity(
    c: torch.Tensor,
    a: torch.Tensor,
    r_dot_target: torch.Tensor,
    c_dot: torch.Tensor | None = None,
    eps: float = EPS,
) -> tuple[torch.Tensor, dict[str, float]]:
    c64 = sym(c).to(device=a.device, dtype=torch.float64)
    a64 = a.to(dtype=torch.float64)
    target = sym(r_dot_target).to(device=a.device, dtype=torch.float64)
    if c_dot is not None:
        target = sym(target - a64.T @ c_dot.to(device=a.device, dtype=torch.float64) @ a64)
    r = composite_metric(a64, c64)
    s = solve_sylvester_spd(r, target, eps)
    delta = a64 @ s
    realized = a64.T @ c64 @ delta + delta.T @ c64 @ a64
    residual = float(((realized - target).norm() / target.norm().clamp_min(float(eps))).detach().cpu().item()) if float(target.norm().detach().cpu().item()) > 0.0 else 0.0
    return delta, {
        "radial_residual": residual,
        "radial_velocity_norm": float(delta.norm().detach().cpu().item()),
        "radial_c_norm": float(c_norm(delta, c64, eps).detach().cpu().item()),
        "radial_target_norm": float(target.norm().detach().cpu().item()),
    }


def tube_retract(
    a_tilde: torch.Tensor,
    c: torch.Tensor,
    tube_target_r: torch.Tensor,
    config: CompositeMetricTubeConfig | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    cfg = config or CompositeMetricTubeConfig()
    c64 = sym(c).to(device=a_tilde.device, dtype=torch.float64)
    rt = sym(tube_target_r).to(device=a_tilde.device, dtype=torch.float64)
    dim = int(rt.shape[0])
    eye = torch.eye(dim, device=a_tilde.device, dtype=torch.float64)
    r_tilde = composite_metric(a_tilde.to(dtype=torch.float64), c64)
    a_new = a_tilde.to(dtype=torch.float64) @ matrix_inv_sqrt(r_tilde + float(cfg.rho) * eye, float(cfg.eps)) @ matrix_sqrt(rt + float(cfg.rho) * eye, float(cfg.eps))
    r_new = composite_metric(a_new, c64)
    return a_new, {
        "metric_retraction_error": relative_fro_error(r_new, rt, float(cfg.eps)),
        "metric_drift_before_tube_retraction": relative_fro_error(r_tilde, rt, float(cfg.eps)),
        "A_update_finite": int(torch.isfinite(a_new).all().detach().cpu().item()),
    }


@dataclass
class _TubeParamState:
    param: torch.nn.Parameter
    c: torch.Tensor
    name: str
    c_inv: torch.Tensor
    center_r: torch.Tensor | None = None
    target_r: torch.Tensor | None = None
    pre_r: torch.Tensor | None = None
    last_delta: torch.Tensor | None = None
    debt_dual: torch.Tensor | None = None
    should_retract_this_step: bool = False
    transformed_this_step: bool = False
    last_diag: dict[str, float] = field(default_factory=dict)


@dataclass
class CompositeMetricTubeTrace:
    step_calls: int = 0
    transform_calls: int = 0
    retraction_calls: int = 0
    state_update_calls: int = 0
    velocity_emitted_calls: int = 0
    registered_edge_count: int = 0
    optimizer_owned_gradient_transform_pass: int = 0
    inside_optimizer_step_violations: int = 0
    transform_time_ms: float = 0.0
    retraction_time_ms: float = 0.0
    tangent_residual_values: list[float] = field(default_factory=list)
    radial_residual_values: list[float] = field(default_factory=list)
    retraction_error_values: list[float] = field(default_factory=list)
    log_spec_values: list[float] = field(default_factory=list)
    trace_ratio_values: list[float] = field(default_factory=list)
    condition_values: list[float] = field(default_factory=list)
    radial_fraction_values: list[float] = field(default_factory=list)
    tangent_fraction_values: list[float] = field(default_factory=list)
    debt_dual_values: list[float] = field(default_factory=list)
    debt_cotangent_norm_ratio_values: list[float] = field(default_factory=list)
    debt_cotangent_observe_count: int = 0
    projection_count: int = 0
    tube_saturation_count: int = 0

    def as_dict(self) -> dict[str, float | int]:
        def mean(vals: list[float], default: float = 0.0) -> float:
            clean = [float(v) for v in vals if math.isfinite(float(v))]
            return sum(clean) / len(clean) if clean else float(default)

        def maxv(vals: list[float], default: float = 0.0) -> float:
            clean = [float(v) for v in vals if math.isfinite(float(v))]
            return max(clean) if clean else float(default)

        return {
            "optimizer_owned_gradient_transform_pass": int(self.optimizer_owned_gradient_transform_pass),
            "state_updated_every_step_or_cadence": int(self.state_update_calls > 0),
            "metric_state_updated_every_step_or_cadence": int(self.state_update_calls > 0),
            "tube_velocity_emitted": int(self.velocity_emitted_calls > 0),
            "tube_projection_trace_written": int(self.projection_count > 0),
            "runtime_metric_winner_selection_used": 0,
            "runtime_highfreq_lowfreq_switch_used": 0,
            "runtime_topk_metric_mode_used": 0,
            "runtime_trust_threshold_metric_gate_used": 0,
            "runtime_dataset_seed_branch_used": 0,
            "candidate_update_selected_runtime": 0,
            "inside_optimizer_step_violations": int(self.inside_optimizer_step_violations),
            "tube_projection_count": int(self.projection_count),
            "R_log_spec_distance_median": mean(self.log_spec_values),
            "R_log_spec_distance_max": maxv(self.log_spec_values),
            "R_trace_ratio_median": mean(self.trace_ratio_values, 1.0),
            "R_condition_median": mean(self.condition_values, 1.0),
            "radial_velocity_fraction_median": mean(self.radial_fraction_values),
            "tangent_velocity_fraction_median": mean(self.tangent_fraction_values, 1.0),
            "debt_controller_used": int(self.debt_cotangent_observe_count > 0),
            "debt_dual_mean": mean(self.debt_dual_values),
            "debt_cotangent_norm_ratio_mean": mean(self.debt_cotangent_norm_ratio_values),
            "tube_saturation_rate": float(self.tube_saturation_count) / max(1, int(self.projection_count)),
            "max_tangent_residual": maxv(self.tangent_residual_values),
            "max_radial_residual": maxv(self.radial_residual_values),
            "max_retraction_error": maxv(self.retraction_error_values),
            "transform_time_ms": float(self.transform_time_ms),
            "retraction_time_ms": float(self.retraction_time_ms),
        }


class ControlledCompositeMetricTubeFlow:
    """Optimizer-owned tangent+radial flow constrained by a spectral metric tube."""

    def __init__(
        self,
        edge_params: Iterable[torch.nn.Parameter],
        c_matrices: Iterable[torch.Tensor],
        config: CompositeMetricTubeConfig | None = None,
        names: Iterable[str] | None = None,
    ) -> None:
        params = [p for p in edge_params if getattr(p, "requires_grad", False)]
        cs = list(c_matrices)
        ns = list(names) if names is not None else [f"edge_{idx}" for idx in range(len(params))]
        if len(cs) != len(params):
            raise ValueError("c_matrices length must match trainable edge_params length")
        self.config = config or CompositeMetricTubeConfig()
        self.trace = CompositeMetricTubeTrace(registered_edge_count=len(params))
        self._states: dict[int, _TubeParamState] = {}
        self._inside_optimizer_step = False
        self._external_debt_cotangents: dict[int, torch.Tensor] = {}
        self._external_debt_proxy: float = 0.0
        for idx, param in enumerate(params):
            c = sym(cs[idx]).detach().to(device=param.device, dtype=torch.float64)
            a = w1_to_matrix(param.detach()).to(device=param.device)
            center = composite_metric(a, c).detach()
            state = _TubeParamState(
                param=param,
                c=c,
                name=ns[idx],
                c_inv=torch.linalg.inv(c),
                center_r=center.clone(),
                target_r=center.clone(),
            )
            state.debt_dual = torch.zeros((), device=param.device, dtype=torch.float64)
            self._states[id(param)] = state

    def owns_param(self, param: torch.nn.Parameter) -> bool:
        return id(param) in self._states

    def set_inside_optimizer_step(self, value: bool) -> None:
        self._inside_optimizer_step = bool(value)
        if value:
            self.trace.step_calls += 1
            for state in self._states.values():
                state.transformed_this_step = False

    def observe_debt_proxy(self, value: float | torch.Tensor) -> None:
        if isinstance(value, torch.Tensor):
            self._external_debt_proxy = float(value.detach().mean().cpu().item())
        else:
            self._external_debt_proxy = float(value)

    def observe_debt_cotangent(self, cotangents: Iterable[torch.Tensor | None]) -> None:
        observed: dict[int, torch.Tensor] = {}
        for state, cotangent in zip(self._states.values(), cotangents):
            if cotangent is None:
                continue
            observed[id(state.param)] = cotangent.detach().clone()
        self._external_debt_cotangents = observed
        self.trace.debt_cotangent_observe_count += 1

    def _metric_state_target(self, state: _TubeParamState, a: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        assert state.center_r is not None
        current = composite_metric(a, state.c)
        task_psd = sym(g.T @ g)
        task_psd = task_psd * (torch.trace(current).abs() / torch.trace(task_psd).abs().clamp_min(float(self.config.eps)))
        beta = float(self.config.metric_state_beta)
        raw_target = sym((1.0 - beta) * current + beta * task_psd)
        projected = project_spd_to_tube(state.center_r, raw_target, self.config)
        state.target_r = projected.r_projected.detach().clone()
        diag = {
            "R_log_spec_distance": projected.log_spec_distance,
            "R_trace_ratio": projected.trace_ratio,
            "R_condition": projected.condition,
            "tube_projection_clip_fraction": projected.clip_fraction,
            "tube_projection_trace_scale": projected.trace_scale,
            "tube_projection_ridge_added": projected.ridge_added,
        }
        return state.target_r, diag

    def transform(self, grad: torch.Tensor, param: torch.nn.Parameter, _group: dict[str, Any] | None = None) -> torch.Tensor:
        start = time.perf_counter()
        self.trace.transform_calls += 1
        if not self._inside_optimizer_step:
            self.trace.inside_optimizer_step_violations += 1
            return grad
        state = self._states.get(id(param))
        if state is None:
            return grad
        interval = max(1, int(self.config.metric_update_interval))
        should_update_metric = state.target_r is None or (int(self.trace.step_calls) % interval == 0)
        a = w1_to_matrix(param.detach()).to(device=param.device)
        g = w1_to_matrix(grad.detach()).to(device=param.device)
        if float(self.config.debt_cotangent_blend) > 0.0:
            cot = self._external_debt_cotangents.get(id(param))
            debt_g = w1_to_matrix(cot).to(device=param.device) if cot is not None else None
            if state.debt_dual is not None:
                state.debt_dual = (
                    state.debt_dual
                    + float(self.config.debt_dual_lr) * (float(self._external_debt_proxy) - float(self.config.debt_budget))
                ).clamp(0.0, float(self.config.anti_windup_max))
                self.trace.debt_dual_values.append(float(state.debt_dual.detach().cpu().item()))
            if debt_g is not None:
                task_norm = g.norm().clamp_min(float(self.config.eps))
                debt_norm = debt_g.norm().clamp_min(float(self.config.eps))
                max_debt_norm = float(self.config.debt_cotangent_norm_cap) * task_norm
                if float(debt_norm.detach().cpu().item()) > float(max_debt_norm.detach().cpu().item()):
                    debt_g = debt_g * (max_debt_norm / debt_norm)
                    debt_norm = debt_g.norm().clamp_min(float(self.config.eps))
                debt_scale = 1.0 + (state.debt_dual.to(device=g.device, dtype=g.dtype) if state.debt_dual is not None else 0.0)
                g = g + float(self.config.debt_cotangent_blend) * debt_scale * debt_g
                self.trace.debt_cotangent_norm_ratio_values.append(float((debt_norm / task_norm).detach().cpu().item()))
        if str(self.config.control_mode) in {"same_tube_random", "same_tangent_energy_random", "same_radial_energy_random"}:
            gen = torch.Generator(device=param.device)
            gen.manual_seed(int(self.config.random_seed) + int(self.trace.transform_calls) * 1009)
            noise = torch.randn(g.shape, generator=gen, device=param.device, dtype=torch.float64)
            g = noise * (g.norm().clamp_min(float(self.config.eps)) / noise.norm().clamp_min(float(self.config.eps)))
        tan, tan_diag = natural_tangent_velocity(state.c, a, g, float(self.config.eps), state.c_inv)
        if should_update_metric:
            target_r, target_diag = self._metric_state_target(state, a, g)
            self.trace.state_update_calls += 1
            self.trace.projection_count += 1
            if float(target_diag.get("tube_projection_clip_fraction", 0.0)) > 0.0:
                self.trace.tube_saturation_count += 1
        else:
            assert state.target_r is not None
            target_r = state.target_r
            target_diag = {
                "R_log_spec_distance": float(state.last_diag.get("R_log_spec_distance", 0.0)),
                "R_trace_ratio": float(state.last_diag.get("R_trace_ratio", 1.0)),
                "R_condition": float(state.last_diag.get("R_condition", 1.0)),
                "tube_projection_clip_fraction": 0.0,
                "tube_projection_trace_scale": 1.0,
                "tube_projection_ridge_added": 0.0,
            }
        current = composite_metric(a, state.c)
        r_dot = sym(target_r - current)
        rad, rad_diag = radial_velocity(state.c, a, r_dot, None, float(self.config.eps))
        tan_norm = c_norm(tan, state.c, float(self.config.eps))
        rad_norm = c_norm(rad, state.c, float(self.config.eps))
        cap = min(1.0, float(self.config.radial_velocity_fraction_cap) * float(tan_norm.detach().cpu().item()) / max(float(rad_norm.detach().cpu().item()), float(self.config.eps)))
        delta = tan + float(cap) * rad
        max_norm = float(self.config.max_norm_ratio) * g.norm().clamp_min(float(self.config.eps))
        if float(delta.norm().detach().cpu().item()) > float(max_norm.detach().cpu().item()):
            delta = delta * (max_norm / delta.norm().clamp_min(float(self.config.eps)))
        state.pre_r = current.detach().clone()
        state.last_delta = delta.detach().clone()
        state.should_retract_this_step = should_update_metric
        state.transformed_this_step = True
        state.last_diag = {**tan_diag, **rad_diag, **target_diag, "radial_velocity_fraction": float(cap), "tangent_velocity_fraction": 1.0}
        self.trace.velocity_emitted_calls += 1
        self.trace.optimizer_owned_gradient_transform_pass = 1
        self.trace.tangent_residual_values.append(float(tan_diag.get("tangent_residual", 0.0)))
        self.trace.radial_residual_values.append(float(rad_diag.get("radial_residual", 0.0)))
        self.trace.radial_fraction_values.append(float(cap))
        self.trace.tangent_fraction_values.append(1.0)
        self.trace.log_spec_values.append(float(target_diag["R_log_spec_distance"]))
        self.trace.trace_ratio_values.append(float(target_diag["R_trace_ratio"]))
        self.trace.condition_values.append(float(target_diag["R_condition"]))
        self.trace.transform_time_ms += (time.perf_counter() - start) * 1000.0
        return matrix_to_w1(-delta, grad.shape, dtype=grad.dtype, device=grad.device)

    def after_base_step(self) -> None:
        if not self._inside_optimizer_step:
            self.trace.inside_optimizer_step_violations += 1
            return
        start = time.perf_counter()
        for state in self._states.values():
            if not state.transformed_this_step or state.target_r is None or not state.should_retract_this_step:
                continue
            param = state.param
            a_tilde = w1_to_matrix(param.detach()).to(device=param.device)
            projected = project_spd_to_tube(state.center_r, state.target_r, self.config)
            a_new, diag = tube_retract(a_tilde, state.c, projected.r_projected, self.config)
            with torch.no_grad():
                param.copy_(matrix_to_w1(a_new, param.shape, dtype=param.dtype, device=param.device))
            self.trace.retraction_calls += 1
            self.trace.retraction_error_values.append(float(diag["metric_retraction_error"]))
            state.last_diag.update(diag)
        self.trace.retraction_time_ms += (time.perf_counter() - start) * 1000.0

    def diagnostics(self) -> dict[str, Any]:
        out: dict[str, Any] = dict(self.trace.as_dict())
        for state in self._states.values():
            for key, value in state.last_diag.items():
                out[f"{state.name}_{key}"] = value
        return out


__all__ = [
    "CompositeMetricTubeConfig",
    "ControlledCompositeMetricTubeFlow",
    "TubeProjectionResult",
    "c_norm",
    "log_spec_distance",
    "project_spd_to_tube",
    "radial_velocity",
    "tube_retract",
]
