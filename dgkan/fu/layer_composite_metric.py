"""Layer-level edge-bank composite metric utilities for v22.89R.

The code in this module is deliberately small and explicit.  It provides the
matrix operations needed by the experiment runner and an optimizer-owned
gradient transform that operates on persistent ``w1`` edge-coordinate tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Iterable

import torch


EPS = 1.0e-12


def sym(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.transpose(-1, -2))


def _clean_eigh(x: torch.Tensor, eps: float = EPS) -> tuple[torch.Tensor, torch.Tensor]:
    vals, vecs = torch.linalg.eigh(sym(x).to(dtype=torch.float64))
    vals = vals.clamp_min(float(eps))
    return vals, vecs


def matrix_sqrt(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    vals, vecs = _clean_eigh(x, eps)
    return (vecs * vals.sqrt().unsqueeze(0)) @ vecs.T


def matrix_inv_sqrt(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    vals, vecs = _clean_eigh(x, eps)
    return (vecs * vals.rsqrt().unsqueeze(0)) @ vecs.T


def condition_number(x: torch.Tensor, eps: float = EPS) -> float:
    vals, _ = _clean_eigh(x, eps)
    return float((vals.max() / vals.min().clamp_min(float(eps))).detach().cpu().item())


def effective_rank(x: torch.Tensor, eps: float = EPS) -> float:
    vals, _ = _clean_eigh(x, eps)
    probs = vals / vals.sum().clamp_min(float(eps))
    entropy = -(probs * probs.clamp_min(float(eps)).log()).sum()
    return float(entropy.exp().detach().cpu().item())


def ridge_condition(
    c: torch.Tensor,
    *,
    target_condition: float = 1.0e5,
    max_ridge_rel: float = 1.0e-2,
    eps: float = EPS,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Add isotropic ridge until the SPD matrix reaches the condition budget."""

    c64 = sym(c).to(dtype=torch.float64)
    dim = int(c64.shape[0])
    eye = torch.eye(dim, device=c64.device, dtype=c64.dtype)
    trace_scale = float((torch.trace(c64).abs() / max(1, dim)).detach().cpu().item())
    base = max(trace_scale, float(eps))
    cond_before = condition_number(c64, eps)
    ridge = 0.0
    out = c64
    for power in range(10):
        cond = condition_number(out, eps)
        if cond <= float(target_condition):
            break
        candidate = base * (10.0 ** (power - 8))
        ridge = min(max(float(candidate), ridge * 10.0 if ridge > 0.0 else float(candidate)), float(max_ridge_rel) * base)
        out = c64 + ridge * eye
    cond_after = condition_number(out, eps)
    return out, {
        "ridge_added": float(ridge),
        "C_condition_before_ridge": float(cond_before),
        "C_condition_after_ridge": float(cond_after),
    }


def metric_matrix(phi: torch.Tensor, weights: torch.Tensor | None = None, smooth_diag: torch.Tensor | None = None, ridge: float = 1.0e-6) -> torch.Tensor:
    x = phi.to(dtype=torch.float64)
    n = max(1, int(x.shape[0]))
    if weights is None:
        c = (x.T @ x) / float(n)
    else:
        w = weights.detach().to(device=x.device, dtype=x.dtype).reshape(-1).clamp_min(0.0)
        w = w / w.mean().clamp_min(EPS)
        c = x.T @ (x * w.unsqueeze(1)) / float(n)
    if smooth_diag is not None:
        sd = smooth_diag.detach().to(device=x.device, dtype=x.dtype).reshape(-1)
        c = c + torch.diag(sd)
    if ridge > 0.0:
        c = c + float(ridge) * torch.eye(int(c.shape[0]), device=x.device, dtype=x.dtype)
    return sym(c)


def w1_to_matrix(param: torch.Tensor) -> torch.Tensor:
    """Map [input, output, basis] or [input, hidden, basis] to [input*basis, output]."""

    if param.ndim == 2:
        return param.detach().to(dtype=torch.float64)
    if param.ndim != 3:
        raise ValueError(f"expected 2D or 3D edge tensor, got shape {tuple(param.shape)}")
    return param.detach().permute(0, 2, 1).reshape(int(param.shape[0]) * int(param.shape[2]), int(param.shape[1])).to(dtype=torch.float64)


def matrix_to_w1(a: torch.Tensor, shape: torch.Size | tuple[int, ...], *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if len(shape) == 2:
        return a.to(device=device, dtype=dtype).reshape(shape)
    if len(shape) != 3:
        raise ValueError(f"expected 2D or 3D edge tensor shape, got {tuple(shape)}")
    d, h, k = int(shape[0]), int(shape[1]), int(shape[2])
    return a.to(device=device, dtype=dtype).reshape(d, k, h).permute(0, 2, 1).contiguous()


def basis_smoothness_diag(basis_name: str, input_dim: int, k: int, scale: float = 1.0e-4, device: torch.device | None = None) -> torch.Tensor:
    name = str(basis_name)
    vals: list[float] = []
    if name == "chebyshev":
        vals = [float(i * i) for i in range(int(k))]
    elif name == "fourier_lowfreq":
        vals = [1.0]
        freq = 1
        while len(vals) < int(k):
            vals.append(float(freq * freq))
            if len(vals) < int(k):
                vals.append(float(freq * freq))
            freq += 1
    else:
        vals = [float((i + 1) * (i + 1)) for i in range(int(k))]
    one = torch.tensor(vals[: int(k)], device=device, dtype=torch.float64) * float(scale)
    return one.repeat(int(input_dim))


def primitive_layer1_phi(model: Any, x: torch.Tensor) -> torch.Tensor:
    basis = model.layer1_basis(x)
    denom = math.sqrt(max(1, int(getattr(model, "input_dim", basis.shape[1]))))
    return (basis / denom).reshape(int(basis.shape[0]), -1).to(dtype=torch.float64)


def additive_phi(model: Any, x: torch.Tensor) -> torch.Tensor:
    basis = model.basis(x)
    denom = math.sqrt(max(1, int(getattr(model, "input_dim", basis.shape[1]))))
    return (basis / denom).reshape(int(basis.shape[0]), -1).to(dtype=torch.float64)


def make_target_metric(out_dim: int, target_type: str = "variance_critical", target_variance: float = 1.0, device: torch.device | None = None) -> torch.Tensor:
    scale = float(target_variance)
    if target_type == "flat":
        scale = 1.0
    if target_type == "derivative":
        scale = max(0.5, float(target_variance))
    if target_type == "lowfreq":
        scale = float(target_variance)
    return torch.eye(int(out_dim), device=device, dtype=torch.float64) * scale


def euclidean_orthonormal_basis(c: torch.Tensor, out_dim: int, *, mode: str = "random", smooth_diag: torch.Tensor | None = None, seed: int = 0) -> torch.Tensor:
    dim = int(c.shape[0])
    cols = int(out_dim)
    if mode in {"lowfreq", "highfreq"} and smooth_diag is not None:
        order_all = torch.argsort(smooth_diag.detach().to(device=c.device, dtype=torch.float64), descending=(mode == "highfreq"))
        order = order_all[:cols]
        u = torch.zeros(dim, cols, device=c.device, dtype=torch.float64)
        u[order, torch.arange(cols, device=c.device)] = 1.0
        q, _ = torch.linalg.qr(u, mode="reduced")
        return q[:, :cols]
    gen = torch.Generator(device=c.device)
    gen.manual_seed(int(seed))
    raw = torch.randn(dim, cols, generator=gen, device=c.device, dtype=torch.float64)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q[:, :cols]


def factorized_initialization(c: torch.Tensor, r_star: torch.Tensor, u: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    inv_sqrt_c = matrix_inv_sqrt(c, eps)
    sqrt_r = matrix_sqrt(r_star, eps)
    return inv_sqrt_c @ u.to(device=c.device, dtype=torch.float64) @ sqrt_r


def composite_metric(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return sym(a.T @ c.to(device=a.device, dtype=torch.float64) @ a)


def relative_fro_error(x: torch.Tensor, y: torch.Tensor, eps: float = EPS) -> float:
    return float(((x - y).norm() / y.norm().clamp_min(float(eps))).detach().cpu().item())


def composite_effect_fraction(phi: torch.Tensor, delta_a: torch.Tensor, input_dim: int, k: int, eps: float = EPS) -> tuple[float, float]:
    x = phi.to(device=delta_a.device, dtype=torch.float64)
    da = delta_a.to(dtype=torch.float64)
    full = (x @ da).square().sum()
    denom = torch.zeros((), device=x.device, dtype=torch.float64)
    for idx in range(int(input_dim)):
        sl = slice(idx * int(k), (idx + 1) * int(k))
        denom = denom + (x[:, sl] @ da[sl, :]).square().sum()
    frac = float((full / denom.clamp_min(float(eps))).detach().cpu().item())
    return frac, 1.0 - frac


def solve_sylvester_spd(r: torch.Tensor, b: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    vals, vecs = _clean_eigh(r, eps)
    bt = vecs.T @ b.to(device=r.device, dtype=torch.float64) @ vecs
    denom = vals.unsqueeze(1) + vals.unsqueeze(0)
    st = bt / denom.clamp_min(float(eps))
    return vecs @ st @ vecs.T


def tangent_projection(c: torch.Tensor, a: torch.Tensor, z: torch.Tensor, eps: float = EPS) -> tuple[torch.Tensor, dict[str, float]]:
    c64 = c.to(device=a.device, dtype=torch.float64)
    a64 = a.to(dtype=torch.float64)
    z64 = z.to(dtype=torch.float64)
    r = composite_metric(a64, c64)
    b = a64.T @ c64 @ z64 + z64.T @ c64 @ a64
    s = solve_sylvester_spd(r, b, eps)
    delta = z64 - a64 @ s
    tangent = a64.T @ c64 @ delta + delta.T @ c64 @ a64
    sylv = r @ s + s @ r - b
    return delta, {
        "tangent_residual": float((tangent.norm() / r.norm().clamp_min(float(eps))).detach().cpu().item()),
        "Sylvester_residual": float((sylv.norm() / b.norm().clamp_min(float(eps))).detach().cpu().item()) if float(b.norm().detach().cpu().item()) > 0.0 else 0.0,
        "R_condition": condition_number(r, eps),
    }


def natural_tangent_velocity(
    c: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    eps: float = EPS,
    c_inv: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    c64 = c.to(device=a.device, dtype=torch.float64)
    g64 = g.to(device=a.device, dtype=torch.float64)
    if c_inv is None:
        z = -torch.linalg.solve(c64, g64)
    else:
        z = -(c_inv.to(device=a.device, dtype=torch.float64) @ g64)
    delta, diag = tangent_projection(c64, a, z, eps)
    preserved = ((-g64) * delta).sum() / (g64.norm() * delta.norm()).clamp_min(float(eps))
    solve_res = (c64 @ z + g64).norm() / g64.norm().clamp_min(float(eps))
    diag.update(
        {
            "task_gradient_preserved_fraction": float(preserved.detach().cpu().item()),
            "C_solve_residual": float(solve_res.detach().cpu().item()),
            "projected_velocity_norm": float(delta.norm().detach().cpu().item()),
            "task_gradient_norm": float(g64.norm().detach().cpu().item()),
        }
    )
    return delta, diag


def random_tangent_velocity(c: torch.Tensor, a: torch.Tensor, reference_delta: torch.Tensor, seed: int, eps: float = EPS) -> tuple[torch.Tensor, dict[str, float]]:
    gen = torch.Generator(device=a.device)
    gen.manual_seed(int(seed))
    noise = torch.randn(tuple(reference_delta.shape), generator=gen, device=a.device, dtype=torch.float64)
    delta, diag = tangent_projection(c, a, noise, eps)
    scale = reference_delta.norm().clamp_min(float(eps)) / delta.norm().clamp_min(float(eps))
    delta = delta * scale
    diag["task_gradient_preserved_fraction"] = 0.0
    diag["projected_velocity_norm"] = float(delta.norm().detach().cpu().item())
    return delta, diag


def retract_to_metric(a_tilde: torch.Tensor, c: torch.Tensor, r_target: torch.Tensor, rho: float = 1.0e-8, eps: float = EPS) -> tuple[torch.Tensor, dict[str, float]]:
    c64 = c.to(device=a_tilde.device, dtype=torch.float64)
    rt = sym(r_target.to(device=a_tilde.device, dtype=torch.float64))
    dim = int(rt.shape[0])
    eye = torch.eye(dim, device=a_tilde.device, dtype=torch.float64)
    r_tilde = composite_metric(a_tilde, c64)
    a_new = a_tilde @ matrix_inv_sqrt(r_tilde + float(rho) * eye, eps) @ matrix_sqrt(rt + float(rho) * eye, eps)
    r_new = composite_metric(a_new, c64)
    rhat_new = r_new / (torch.trace(r_new) / max(1, dim)).clamp_min(float(eps))
    rhat_target = rt / (torch.trace(rt) / max(1, dim)).clamp_min(float(eps))
    return a_new, {
        "metric_drift_before_retraction": relative_fro_error(r_tilde, rt, eps),
        "metric_drift_after_retraction": relative_fro_error(r_new, rt, eps),
        "retraction_error": relative_fro_error(r_new, rt, eps),
        "shape_drift": relative_fro_error(rhat_new, rhat_target, eps),
        "scale_change": float((torch.trace(r_new) / torch.trace(rt).clamp_min(float(eps))).detach().cpu().item()),
    }


def quantile_coordinates(values: torch.Tensor) -> torch.Tensor:
    x = values.detach()
    order = torch.argsort(x, dim=0)
    ranks = torch.empty_like(order, dtype=torch.float64)
    base = torch.linspace(0.0, 1.0, int(x.shape[0]), device=x.device, dtype=torch.float64).unsqueeze(1).expand_as(ranks)
    ranks.scatter_(0, order, base)
    return ranks.to(dtype=x.dtype)


def transport_diagnostic_from_basis(old_u: torch.Tensor, new_u: torch.Tensor, basis_fn: Any) -> dict[str, float]:
    raw_old = basis_fn(old_u).reshape(int(old_u.shape[0]), -1).to(dtype=torch.float64)
    raw_new = basis_fn(new_u).reshape(int(new_u.shape[0]), -1).to(dtype=torch.float64)
    s_old = quantile_coordinates(old_u)
    s_new = quantile_coordinates(new_u)
    q_old = basis_fn(2.0 * s_old - 1.0).reshape(int(old_u.shape[0]), -1).to(dtype=torch.float64)
    q_new = basis_fn(2.0 * s_new - 1.0).reshape(int(new_u.shape[0]), -1).to(dtype=torch.float64)
    c_raw_old = metric_matrix(raw_old)
    c_raw_new = metric_matrix(raw_new)
    c_q_old = metric_matrix(q_old)
    c_q_new = metric_matrix(q_new)
    raw_drift = relative_fro_error(c_raw_new, c_raw_old)
    q_drift = relative_fro_error(c_q_new, c_q_old)
    cosine = torch.nn.functional.cosine_similarity(q_old.flatten(), q_new.flatten(), dim=0)
    return {
        "transport_drift": q_drift,
        "no_transport_drift": raw_drift,
        "transport_reduced_source_guard_R_drift_fraction": float(((raw_drift - q_drift) / max(raw_drift, EPS))),
        "basis_identity_cosine_pre_post_transport": float(cosine.detach().cpu().item()),
        "edge_extrapolation_rate_delta": 0.0,
    }


@dataclass
class CompositeMetricFlowConfig:
    variant: str = "shape"
    rho: float = 1.0e-8
    eps: float = EPS
    random_seed: int = 0
    control_mode: str = "task"
    max_norm_ratio: float = 2.0
    scale_band: float = 0.05
    debt_cotangent_blend: float = 0.0
    debt_dual_lr: float = 0.05
    debt_budget: float = 0.0
    anti_windup_max: float = 10.0
    population_cotangent_blend: float = 0.0
    barrier_projection: bool = False
    barrier_alpha: float = 0.10
    barrier_max_correction_ratio: float = 2.0


@dataclass
class _ParamMetricState:
    param: torch.nn.Parameter
    c: torch.Tensor
    name: str
    c_inv: torch.Tensor | None = None
    initial_c: torch.Tensor | None = None
    reference_r: torch.Tensor | None = None
    pre_r: torch.Tensor | None = None
    transformed_this_step: bool = False
    last_delta: torch.Tensor | None = None
    last_diag: dict[str, float] = field(default_factory=dict)
    refresh_diag: dict[str, float] = field(default_factory=dict)
    debt_dual: torch.Tensor | None = None


@dataclass
class CompositeMetricFlowTrace:
    step_calls: int = 0
    transform_calls: int = 0
    state_update_calls: int = 0
    edge_velocity_emitted_calls: int = 0
    retraction_calls: int = 0
    registered_edge_count: int = 0
    optimizer_owned_gradient_transform_pass: int = 0
    direct_parameter_update_attempted: int = 0
    inside_optimizer_step_violations: int = 0
    candidate_update_selected_runtime: int = 0
    runtime_argmax_used: int = 0
    runtime_topk_used: int = 0
    changed_w2_readout_tensors: int = 0
    transform_time_ms: float = 0.0
    metric_drift_after_values: list[float] = field(default_factory=list)
    tangent_residual_values: list[float] = field(default_factory=list)
    sylvester_residual_values: list[float] = field(default_factory=list)
    task_gradient_fraction_values: list[float] = field(default_factory=list)
    projected_norm_values: list[float] = field(default_factory=list)
    debt_dual_values: list[float] = field(default_factory=list)
    barrier_active_values: list[float] = field(default_factory=list)
    barrier_kkt_residual_values: list[float] = field(default_factory=list)
    barrier_task_preserved_values: list[float] = field(default_factory=list)
    barrier_correction_ratio_values: list[float] = field(default_factory=list)
    population_observe_count: int = 0
    population_cotangent_blend_values: list[float] = field(default_factory=list)
    population_correction_ratio_values: list[float] = field(default_factory=list)
    raw_scale_change_values: list[float] = field(default_factory=list)
    scale_band_violation_count: int = 0
    c_refresh_count: int = 0
    c_refresh_reset_reference_count: int = 0
    c_refresh_rel_drift_values: list[float] = field(default_factory=list)
    c_refresh_rel_drift_vs_initial_values: list[float] = field(default_factory=list)
    r_drift_oldc_vs_old_reference_values: list[float] = field(default_factory=list)
    r_drift_newc_vs_old_reference_values: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, float | int]:
        def mean(vals: list[float], default: float = 0.0) -> float:
            clean = [float(v) for v in vals if math.isfinite(float(v))]
            return sum(clean) / len(clean) if clean else float(default)

        def max_clean(vals: list[float], default: float = 0.0) -> float:
            clean = [float(v) for v in vals if math.isfinite(float(v))]
            return max(clean) if clean else float(default)

        return {
            "state_updated_every_step": int(self.state_update_calls >= self.registered_edge_count and self.registered_edge_count > 0),
            "velocity_emitted_every_step": int(self.edge_velocity_emitted_calls >= self.registered_edge_count and self.registered_edge_count > 0),
            "all_edges_receive_state_update": int(self.state_update_calls >= self.registered_edge_count and self.registered_edge_count > 0),
            "optimizer_owned_transform": int(self.optimizer_owned_gradient_transform_pass),
            "optimizer_owned_gradient_transform_pass": int(self.optimizer_owned_gradient_transform_pass),
            "direct_parameter_update_attempted": int(self.direct_parameter_update_attempted),
            "inside_optimizer_step_violations": int(self.inside_optimizer_step_violations),
            "candidate_update_selected_runtime": int(self.candidate_update_selected_runtime),
            "runtime_argmax_used": int(self.runtime_argmax_used),
            "runtime_topk_used": int(self.runtime_topk_used),
            "changed_w2_readout_tensors": int(self.changed_w2_readout_tensors),
            "registered_edge_count": int(self.registered_edge_count),
            "composite_transform_calls": int(self.transform_calls),
            "composite_retraction_calls": int(self.retraction_calls),
            "metric_drift_after_retraction_mean": mean(self.metric_drift_after_values),
            "metric_drift_after_retraction_max": max(self.metric_drift_after_values) if self.metric_drift_after_values else 0.0,
            "tangent_residual_mean": mean(self.tangent_residual_values),
            "Sylvester_residual_mean": mean(self.sylvester_residual_values),
            "task_gradient_preserved_fraction_mean": mean(self.task_gradient_fraction_values),
            "projected_velocity_norm_mean": mean(self.projected_norm_values),
            "debt_dual_mean": mean(self.debt_dual_values),
            "barrier_projection_used": int(len(self.barrier_active_values) > 0),
            "barrier_active_fraction": mean(self.barrier_active_values),
            "barrier_KKT_residual_mean": mean(self.barrier_kkt_residual_values),
            "barrier_task_velocity_preserved_fraction_mean": mean(self.barrier_task_preserved_values, 1.0),
            "barrier_correction_norm_ratio_mean": mean(self.barrier_correction_ratio_values),
            "barrier_correction_norm_ratio_max": max_clean(self.barrier_correction_ratio_values),
            "population_diffusion_used": int(len(self.population_cotangent_blend_values) > 0),
            "population_diffusion_observation_count": int(self.population_observe_count),
            "population_diffusion_transform_count": int(len(self.population_cotangent_blend_values)),
            "population_cotangent_blend_mean": mean(self.population_cotangent_blend_values),
            "population_correction_norm_ratio_mean": mean(self.population_correction_ratio_values),
            "population_correction_norm_ratio_max": max_clean(self.population_correction_ratio_values),
            "raw_scale_change_mean": mean(self.raw_scale_change_values, 1.0),
            "raw_scale_change_max": max(self.raw_scale_change_values) if self.raw_scale_change_values else 1.0,
            "raw_scale_change_min": min(self.raw_scale_change_values) if self.raw_scale_change_values else 1.0,
            "scale_band_violation_count": int(self.scale_band_violation_count),
            "scale_controller_stability_pass": int(self.scale_band_violation_count == 0),
            "composite_transform_time_ms": float(self.transform_time_ms),
            "C_refresh_used": int(self.c_refresh_count > 0),
            "C_refresh_count": int(self.c_refresh_count),
            "C_refresh_reset_reference_count": int(self.c_refresh_reset_reference_count),
            "C_refresh_rel_drift_mean": mean(self.c_refresh_rel_drift_values),
            "C_refresh_rel_drift_max": max_clean(self.c_refresh_rel_drift_values),
            "C_refresh_rel_drift_vs_initial_mean": mean(self.c_refresh_rel_drift_vs_initial_values),
            "C_refresh_rel_drift_vs_initial_max": max_clean(self.c_refresh_rel_drift_vs_initial_values),
            "R_drift_oldC_vs_old_reference_mean": mean(self.r_drift_oldc_vs_old_reference_values),
            "R_drift_oldC_vs_old_reference_max": max_clean(self.r_drift_oldc_vs_old_reference_values),
            "R_drift_newC_vs_old_reference_mean": mean(self.r_drift_newc_vs_old_reference_values),
            "R_drift_newC_vs_old_reference_max": max_clean(self.r_drift_newc_vs_old_reference_values),
        }


class LayerCompositeMetricFlow:
    """Optimizer-owned composite metric-preserving transform for edge tensors."""

    def __init__(
        self,
        edge_params: Iterable[torch.nn.Parameter],
        c_matrices: Iterable[torch.Tensor],
        config: CompositeMetricFlowConfig | None = None,
        names: Iterable[str] | None = None,
    ) -> None:
        params = [p for p in edge_params if getattr(p, "requires_grad", False)]
        cs = list(c_matrices)
        ns = list(names) if names is not None else [f"edge_{idx}" for idx in range(len(params))]
        if len(cs) != len(params):
            raise ValueError("c_matrices length must match trainable edge_params length")
        self.config = config or CompositeMetricFlowConfig()
        self.trace = CompositeMetricFlowTrace(registered_edge_count=len(params))
        self._states: dict[int, _ParamMetricState] = {}
        self._inside_optimizer_step = False
        self._external_debt_cotangents: dict[int, torch.Tensor] = {}
        self._external_population_cotangents: dict[int, torch.Tensor] = {}
        self._external_population_diag: dict[int, dict[str, float]] = {}
        self._external_debt_proxy: float = 0.0
        for idx, param in enumerate(params):
            c = sym(cs[idx]).detach().to(device=param.device, dtype=torch.float64)
            state = _ParamMetricState(param=param, c=c, name=ns[idx], c_inv=torch.linalg.inv(c), initial_c=c.clone())
            state.debt_dual = torch.zeros((), device=param.device, dtype=torch.float64)
            self._states[id(param)] = state

    def owns_param(self, param: torch.nn.Parameter) -> bool:
        return id(param) in self._states

    def set_inside_optimizer_step(self, value: bool) -> None:
        self._inside_optimizer_step = bool(value)
        if value:
            self.trace.step_calls += 1
            self.trace.state_update_calls = 0
            self.trace.edge_velocity_emitted_calls = 0
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

    def observe_population_cotangent(
        self,
        cotangents: Iterable[torch.Tensor | None],
        diagnostics: Iterable[dict[str, float] | None] | None = None,
    ) -> None:
        observed: dict[int, torch.Tensor] = {}
        observed_diag: dict[int, dict[str, float]] = {}
        diag_list = list(diagnostics) if diagnostics is not None else []
        for idx, (state, cotangent) in enumerate(zip(self._states.values(), cotangents)):
            if cotangent is None:
                continue
            observed[id(state.param)] = cotangent.detach().clone()
            if idx < len(diag_list) and diag_list[idx] is not None:
                observed_diag[id(state.param)] = dict(diag_list[idx] or {})
        self._external_population_cotangents = observed
        self._external_population_diag = observed_diag
        self.trace.population_observe_count += 1

    def refresh_metric(self, param: torch.nn.Parameter, c_new: torch.Tensor, *, reset_reference: bool = True) -> dict[str, float]:
        """Refresh the composite metric matrix for a moving activation domain.

        This updates only optimizer/operator state.  It does not mutate model
        parameters, and it is intended for fixed-cadence train-only domain
        transport diagnostics.
        """

        state = self._states.get(id(param))
        if state is None:
            return {}
        old_c = state.c
        new_c = sym(c_new).detach().to(device=param.device, dtype=torch.float64)
        a = w1_to_matrix(param.detach()).to(device=param.device)
        old_r = composite_metric(a, old_c)
        old_reference = state.reference_r if state.reference_r is not None else (state.pre_r if state.pre_r is not None else old_r)
        old_reference = old_reference.detach().to(device=param.device, dtype=torch.float64)
        new_r = composite_metric(a, new_c)
        initial_c = state.initial_c if state.initial_c is not None else old_c
        diag = {
            "C_refresh_used": 1.0,
            "C_refresh_rel_drift": relative_fro_error(new_c, old_c, float(self.config.eps)),
            "C_refresh_rel_drift_vs_initial": relative_fro_error(new_c, initial_c, float(self.config.eps)),
            "C_refresh_condition": condition_number(new_c, float(self.config.eps)),
            "R_drift_oldC_vs_old_reference": relative_fro_error(old_r, old_reference, float(self.config.eps)),
            "R_drift_newC_vs_old_reference": relative_fro_error(new_r, old_reference, float(self.config.eps)),
            "C_refresh_reset_reference": float(bool(reset_reference)),
        }
        state.c = new_c
        state.c_inv = torch.linalg.inv(new_c)
        if reset_reference:
            state.reference_r = new_r.detach().clone()
            state.pre_r = new_r.detach().clone()
            diag["R_drift_newC_vs_refresh_reference"] = 0.0
            self.trace.c_refresh_reset_reference_count += 1
        state.refresh_diag = diag
        self.trace.c_refresh_count += 1
        self.trace.c_refresh_rel_drift_values.append(float(diag["C_refresh_rel_drift"]))
        self.trace.c_refresh_rel_drift_vs_initial_values.append(float(diag["C_refresh_rel_drift_vs_initial"]))
        self.trace.r_drift_oldc_vs_old_reference_values.append(float(diag["R_drift_oldC_vs_old_reference"]))
        self.trace.r_drift_newc_vs_old_reference_values.append(float(diag["R_drift_newC_vs_old_reference"]))
        return diag

    def _target_r(self, state: _ParamMetricState, a: torch.Tensor) -> torch.Tensor:
        current = composite_metric(a, state.c)
        if state.pre_r is None:
            return current
        if self.config.variant == "strict":
            return state.pre_r
        dim = int(state.pre_r.shape[0])
        old_shape = state.pre_r / (torch.trace(state.pre_r) / max(1, dim)).clamp_min(float(self.config.eps))
        scale = torch.trace(current) / max(1, dim)
        if self.config.variant in {"shape_scale", "shape_scale_controlled", "scale_controlled"}:
            reference = state.reference_r if state.reference_r is not None else state.pre_r
            ref_scale = (torch.trace(reference) / max(1, dim)).clamp_min(float(self.config.eps))
            lower = ref_scale * max(0.0, 1.0 - float(self.config.scale_band))
            upper = ref_scale * (1.0 + float(self.config.scale_band))
            clamped = scale.clamp(float(lower.detach().cpu().item()), float(upper.detach().cpu().item()))
            ratio = float((scale / ref_scale).detach().cpu().item())
            state.last_diag["raw_scale_change_vs_reference"] = ratio
            if ratio < 1.0 - float(self.config.scale_band) or ratio > 1.0 + float(self.config.scale_band):
                state.last_diag["scale_band_violation_count"] = float(state.last_diag.get("scale_band_violation_count", 0.0)) + 1.0
            scale = clamped
        return old_shape * scale.clamp_min(float(self.config.eps))

    def transform(self, grad: torch.Tensor, param: torch.nn.Parameter, _group: dict[str, Any] | None = None) -> torch.Tensor:
        start = time.perf_counter()
        self.trace.transform_calls += 1
        if not self._inside_optimizer_step:
            self.trace.inside_optimizer_step_violations += 1
            return grad
        state = self._states.get(id(param))
        if state is None:
            return grad
        cfg = self.config
        a = w1_to_matrix(param.detach()).to(device=param.device)
        g = w1_to_matrix(grad.detach()).to(device=param.device)
        if float(cfg.debt_cotangent_blend) > 0.0:
            cot = self._external_debt_cotangents.get(id(param))
            if cot is not None:
                g = g + float(cfg.debt_cotangent_blend) * w1_to_matrix(cot).to(device=param.device)
            if state.debt_dual is not None:
                state.debt_dual = (state.debt_dual + float(cfg.debt_dual_lr) * (float(self._external_debt_proxy) - float(cfg.debt_budget))).clamp(0.0, float(cfg.anti_windup_max))
                g = g * (1.0 + state.debt_dual.to(device=g.device, dtype=g.dtype))
                self.trace.debt_dual_values.append(float(state.debt_dual.detach().cpu().item()))
        if float(cfg.population_cotangent_blend) > 0.0:
            cot = self._external_population_cotangents.get(id(param))
            if cot is not None:
                blend = min(1.0, max(0.0, float(cfg.population_cotangent_blend)))
                pop_g = w1_to_matrix(cot).to(device=param.device)
                g = (1.0 - blend) * g + blend * pop_g
                pop_diag = dict(self._external_population_diag.get(id(param), {}))
                pop_diag["population_cotangent_blend"] = blend
                state.refresh_diag.update(pop_diag)
                self.trace.population_cotangent_blend_values.append(blend)
                self.trace.population_correction_ratio_values.append(float(pop_diag.get("population_correction_norm_ratio", 0.0)))
        delta, diag = natural_tangent_velocity(state.c, a, g, float(cfg.eps), state.c_inv)
        if str(cfg.control_mode) == "same_composite_tangent_random":
            delta, random_diag = random_tangent_velocity(state.c, a, delta, int(cfg.random_seed) + self.trace.transform_calls, float(cfg.eps))
            diag.update(random_diag)
        g_norm = g.norm().clamp_min(float(cfg.eps))
        max_norm = float(cfg.max_norm_ratio) * g_norm
        if float(delta.norm().detach().cpu().item()) > float(max_norm.detach().cpu().item()):
            delta = delta * (max_norm / delta.norm().clamp_min(float(cfg.eps)))
            diag["projected_velocity_norm"] = float(delta.norm().detach().cpu().item())
        if bool(cfg.barrier_projection):
            barrier_active = 0.0
            barrier_residual = 0.0
            barrier_preserved = 1.0
            barrier_correction_ratio = 0.0
            cot = self._external_debt_cotangents.get(id(param))
            if cot is not None:
                debt_grad = w1_to_matrix(cot).to(device=param.device)
                debt_proxy = max(0.0, float(self._external_debt_proxy))
                target = -float(cfg.barrier_alpha) * debt_proxy
                pre_lhs = float((debt_grad * delta).sum().detach().cpu().item())
                violation = pre_lhs - target
                if violation > float(cfg.eps):
                    debt_delta, _debt_diag = natural_tangent_velocity(state.c, a, debt_grad, float(cfg.eps), state.c_inv)
                    descent = -float((debt_grad * debt_delta).sum().detach().cpu().item())
                    if descent > float(cfg.eps):
                        coeff = violation / descent
                        correction = float(coeff) * debt_delta
                        max_corr = float(cfg.barrier_max_correction_ratio) * delta.norm().clamp_min(float(cfg.eps))
                        if float(correction.norm().detach().cpu().item()) > float(max_corr.detach().cpu().item()):
                            correction = correction * (max_corr / correction.norm().clamp_min(float(cfg.eps)))
                        before = delta
                        delta = delta + correction
                        barrier_active = 1.0
                        post_lhs = float((debt_grad * delta).sum().detach().cpu().item())
                        barrier_residual = max(0.0, post_lhs - target)
                        barrier_correction_ratio = float(
                            (correction.norm() / before.norm().clamp_min(float(cfg.eps))).detach().cpu().item()
                        )
                        barrier_preserved = float(
                            ((before * delta).sum() / (before.norm() * delta.norm()).clamp_min(float(cfg.eps))).detach().cpu().item()
                        )
                    else:
                        barrier_residual = float(violation)
                else:
                    barrier_residual = 0.0
            self.trace.barrier_active_values.append(float(barrier_active))
            self.trace.barrier_kkt_residual_values.append(float(barrier_residual))
            self.trace.barrier_task_preserved_values.append(float(barrier_preserved))
            self.trace.barrier_correction_ratio_values.append(float(barrier_correction_ratio))
            diag.update(
                {
                    "barrier_projection_used": 1.0,
                    "barrier_active": float(barrier_active),
                    "barrier_KKT_residual": float(barrier_residual),
                    "barrier_task_velocity_preserved_fraction": float(barrier_preserved),
                    "barrier_correction_norm_ratio": float(barrier_correction_ratio),
                    "projected_velocity_norm": float(delta.norm().detach().cpu().item()),
                }
            )
        current_r = composite_metric(a, state.c).detach()
        if state.reference_r is None:
            state.reference_r = current_r.clone()
        state.pre_r = current_r
        state.last_delta = delta.detach()
        state.last_diag = diag
        state.transformed_this_step = True
        self.trace.state_update_calls += 1
        self.trace.edge_velocity_emitted_calls += 1
        self.trace.optimizer_owned_gradient_transform_pass = 1
        self.trace.tangent_residual_values.append(float(diag.get("tangent_residual", 0.0)))
        self.trace.sylvester_residual_values.append(float(diag.get("Sylvester_residual", 0.0)))
        self.trace.task_gradient_fraction_values.append(float(diag.get("task_gradient_preserved_fraction", 0.0)))
        self.trace.projected_norm_values.append(float(diag.get("projected_velocity_norm", 0.0)))
        self.trace.transform_time_ms += (time.perf_counter() - start) * 1000.0
        return matrix_to_w1(-delta, grad.shape, dtype=grad.dtype, device=grad.device)

    def after_base_step(self) -> None:
        if not self._inside_optimizer_step:
            self.trace.inside_optimizer_step_violations += 1
            return
        for state in self._states.values():
            if not state.transformed_this_step or state.pre_r is None:
                continue
            param = state.param
            a_tilde = w1_to_matrix(param.detach()).to(device=param.device)
            target = self._target_r(state, a_tilde)
            a_new, diag = retract_to_metric(a_tilde, state.c, target, float(self.config.rho), float(self.config.eps))
            if "raw_scale_change_vs_reference" in state.last_diag:
                raw_scale = float(state.last_diag.get("raw_scale_change_vs_reference", 1.0))
                self.trace.raw_scale_change_values.append(raw_scale)
                self.trace.scale_band_violation_count += int(state.last_diag.get("scale_band_violation_count", 0.0) > 0.0)
            with torch.no_grad():
                param.copy_(matrix_to_w1(a_new, param.shape, dtype=param.dtype, device=param.device))
            self.trace.retraction_calls += 1
            self.trace.metric_drift_after_values.append(float(diag.get("metric_drift_after_retraction", 0.0)))
            state.last_diag.update(diag)

    def diagnostics(self) -> dict[str, Any]:
        out: dict[str, Any] = dict(self.trace.as_dict())
        for idx, state in enumerate(self._states.values()):
            for key, value in state.last_diag.items():
                out[f"{state.name}_{key}"] = value
            for key, value in state.refresh_diag.items():
                out[f"{state.name}_{key}"] = value
            out[f"{state.name}_C_condition"] = condition_number(state.c, float(self.config.eps))
            out[f"{state.name}_C_effective_rank"] = effective_rank(state.c, float(self.config.eps))
        return out


__all__ = [
    "CompositeMetricFlowConfig",
    "LayerCompositeMetricFlow",
    "additive_phi",
    "basis_smoothness_diag",
    "composite_effect_fraction",
    "composite_metric",
    "condition_number",
    "effective_rank",
    "euclidean_orthonormal_basis",
    "factorized_initialization",
    "make_target_metric",
    "matrix_inv_sqrt",
    "matrix_sqrt",
    "metric_matrix",
    "natural_tangent_velocity",
    "primitive_layer1_phi",
    "quantile_coordinates",
    "relative_fro_error",
    "retract_to_metric",
    "ridge_condition",
    "tangent_projection",
    "transport_diagnostic_from_basis",
    "w1_to_matrix",
    "matrix_to_w1",
]
