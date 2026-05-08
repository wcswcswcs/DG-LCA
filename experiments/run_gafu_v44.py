#!/usr/bin/env python3
"""DG-KAN v4.4 runner: functional-coordinate optimizer redesign probes."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgkan_core import (
    PureKANClassifier,
    RBFDense,
    RuntimeState,
    TrainConfig,
    _get_sobolev_gram,
    coefficient_named_params,
    ensure_dir,
    estimate_geometry,
    evaluate,
    get_device,
    iter_minibatches,
    load_vision_bundle,
    make_pure_norm,
    parse_int_list,
    parse_str_list,
    rbf_basis_occupancy_audit,
    read_csv,
    set_seed,
    write_csv,
)
from run_gafu_v3 import add_args as add_v3_args, dataset_name
from run_gafu_v43 import (
    AdamState,
    V43Params,
    V43RunState,
    _adam_updates,
    _apply_updates,
    _effective_rank,
    _ekfng_updates,
    _feature_audit,
    _flat_updates,
    _fpa_updates,
    _functional_updates,
    _mean,
    _method_updates as _v43_method_updates,
    _restore,
    _role_for_name,
    _role_group,
    _safe_cos,
    _snapshot,
)


DATASETS = ["MNIST", "Fashion-MNIST", "KMNIST"]

P0_METHODS = [
    "PureKAN-AdamW",
    "D0-allFullSobolev",
    "D6-allTaskAware",
    "F4-FNG-leftFullRight",
    "FCAdam-L2-smoke",
    "FCAdam-H1-smoke",
    "SFD-direct-smoke",
    "GFK-output-only-smoke",
    "GFK-all-lowrank-smoke",
    "PureKAN-AB-RBF-AdamW-smoke",
    "PureKAN-AB-RBF-FCAdam-smoke",
]

P1_METHODS = [
    "PureKAN-AdamW-one-step",
    "D0-allFullSobolev",
    "D6-allTaskAware",
    "F4-FNG-leftFullRight",
    "FPA-edge-prox",
    "EKFNG-leftRightFull-lowSob",
    "FCAdam-L2",
    "FCAdam-H1-low",
    "FCAdam-dataSob",
    "SFD-direct",
    "SFD-prox",
    "SFD-residual",
    "GFK-output-only",
    "GFK-block-output",
    "GFK-all-lowrank",
    "AB-RBF-FCAdam-H1-low",
]

P2_METHODS = P1_METHODS


@dataclass
class V44Params:
    train_size: int = 1024
    val_size: int = 512
    test_size: int = 512
    batch_size: int = 128
    eval_batch_size: int = 512
    audit_batch_size: int = 64
    hidden_dim: int = 64
    depth: int = 4
    basis_count: int = 16
    alpha_init: float = 1.0
    adam_lr: float = 1e-3
    raw_lr: float = 2e-4
    fc_lr: float = 8e-4
    fc_beta1: float = 0.9
    fc_beta2: float = 0.999
    fc_eps: float = 1e-8
    fc_rho: float = 1e-3
    fc_h1_alpha: float = 1.0
    fc_h1_low_alpha: float = 0.10
    fc_data_sob_alpha: float = 0.03
    sfd_lambda: float = 2e-3
    sfd_ridge: float = 2e-2
    sfd_tau: float = 1.0
    sfd_trust: float = 0.12
    gfk_tau: float = 0.05
    gfk_ridge: float = 3e-2
    gfk_sob_lambda: float = 1e-3
    gfk_trust: float = 0.10
    p2_steps: int = 5


@dataclass
class FCAdamState:
    m: Dict[str, torch.Tensor] = field(default_factory=dict)
    v: Dict[str, torch.Tensor] = field(default_factory=dict)
    t: Dict[str, int] = field(default_factory=dict)


@dataclass
class V44RunState:
    adam: AdamState = field(default_factory=AdamState)
    shadow_adam: AdamState = field(default_factory=AdamState)
    fc: FCAdamState = field(default_factory=FCAdamState)


class ABRBFDense(RBFDense):
    """RBF edge basis plus explicit constant and linear basis terms."""

    def __init__(self, in_dim: int, out_dim: int, rbf_basis_count: int, *, bias: bool = False) -> None:
        super().__init__(in_dim, out_dim, rbf_basis_count, bias=bias)
        scale = 1.0 / math.sqrt(max(1, in_dim * (rbf_basis_count + 2)))
        self.coeff = nn.Parameter(torch.randn(out_dim, in_dim, rbf_basis_count + 2) * scale)
        self.rbf_basis_count = int(rbf_basis_count)

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        z = (x.unsqueeze(-1) - self.centers) / self.width
        rbf = torch.exp(-0.5 * z.square())
        const = torch.ones_like(x).unsqueeze(-1)
        linear = x.unsqueeze(-1)
        return torch.cat([const, linear, rbf], dim=-1)

    def basis_and_derivative(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = (x.unsqueeze(-1) - self.centers) / self.width
        rbf = torch.exp(-0.5 * z.square())
        rbf_deriv = -((x.unsqueeze(-1) - self.centers) / (self.width**2)) * rbf
        const = torch.ones_like(x).unsqueeze(-1)
        linear = x.unsqueeze(-1)
        deriv = torch.cat([torch.zeros_like(const), torch.ones_like(linear), rbf_deriv], dim=-1)
        return torch.cat([const, linear, rbf], dim=-1), deriv


class ABPureResidualKANBlock(nn.Module):
    def __init__(self, dim: int, basis_count: int, *, norm_mode: str = "fixed") -> None:
        super().__init__()
        self.norm = make_pure_norm(dim, norm_mode)
        self.kan = ABRBFDense(dim, dim, basis_count, bias=False)
        self.register_buffer("alpha", torch.tensor(1.0))
        self.branch_scale = 1.0

    def forward(self, h: torch.Tensor, *, disable_kan: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if disable_kan:
            return h, torch.zeros_like(h)
        branch = self.alpha * float(self.branch_scale) * self.kan(self.norm(h))
        return h + branch, branch


class ABPureKANClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, *, hidden_dim: int, depth: int, basis_count: int) -> None:
        super().__init__()
        self.norm_mode = "fixed"
        self.input_kan = ABRBFDense(input_dim, hidden_dim, basis_count, bias=False)
        self.blocks = nn.ModuleList([ABPureResidualKANBlock(hidden_dim, basis_count) for _ in range(depth)])
        self.output_norm = make_pure_norm(hidden_dim, "fixed")
        self.output_kan = ABRBFDense(hidden_dim, num_classes, basis_count, bias=False)

    def forward(self, x: torch.Tensor, *, disable_kan: bool = False, return_branch: bool = False) -> Any:
        h = self.input_kan(x)
        ratios: List[torch.Tensor] = []
        for block in self.blocks:
            prev = h
            h, branch = block(h, disable_kan=disable_kan)
            ratios.append(branch.norm(dim=-1).mean() / prev.norm(dim=-1).mean().clamp_min(1e-6))
        logits = self.output_kan(self.output_norm(h))
        if return_branch:
            return logits, torch.stack(ratios).mean() if ratios else torch.tensor(0.0, device=x.device)
        return logits

    def kan_layers(self) -> Iterable[RBFDense]:
        yield self.input_kan
        for block in self.blocks:
            yield block.kan
        yield self.output_kan


def _make_model(bundle: Any, params: V44Params, device: torch.device, *, ab: bool = False) -> nn.Module:
    if ab:
        return ABPureKANClassifier(
            bundle.input_dim,
            bundle.num_classes,
            hidden_dim=params.hidden_dim,
            depth=params.depth,
            basis_count=params.basis_count,
        ).to(device)
    return PureKANClassifier(
        bundle.input_dim,
        bundle.num_classes,
        hidden_dim=params.hidden_dim,
        depth=params.depth,
        basis_count=params.basis_count,
        alpha_init=params.alpha_init,
        alpha_mode="fixed1",
        norm_mode="fixed",
    ).to(device)


def _base_cfg(dataset: str, seed: int, method: str, params: V44Params) -> TrainConfig:
    return TrainConfig(
        dataset=dataset,
        method=method,
        optimizer_method="purekan_ufull",
        seed=seed,
        train_size=params.train_size,
        val_size=params.val_size,
        test_size=params.test_size,
        epochs=1,
        batch_size=params.batch_size,
        eval_batch_size=params.eval_batch_size,
        audit_batch_size=params.audit_batch_size,
        hidden_dim=params.hidden_dim,
        depth=params.depth,
        basis_count=params.basis_count,
        alpha_init=params.alpha_init,
        alpha_mode="fixed1",
        pure_norm_mode="fixed",
        model_type="pure_kan",
        gafu_v3_enabled=True,
        v3_phase_mode="hard",
        v3_metric_active="full_sobolev_gram",
        v3_metric_transition="full_sobolev_gram",
        v3_metric_geometry="full_sobolev_gram",
        v3_gram_rho=params.fc_rho,
        coeff_lr=0.03,
        trust_radius=0.50,
        geometry_min_epochs=999,
    )


def _metric_matrix(layer: RBFDense, kind: str, params: V44Params, device: torch.device) -> Tuple[torch.Tensor, Dict[str, float]]:
    k = int(layer.coeff.shape[-1])
    rbf_k = int(layer.centers.numel())
    cfg = _base_cfg("MNIST", 0, f"metric-{kind}", params)
    state = RuntimeState(phase="GEOMETRY", current_branch_scale=1.0, branch_switched=True)
    alpha = 0.0
    if "h1" in kind:
        alpha = params.fc_h1_low_alpha if "low" in kind else params.fc_h1_alpha
    if "data" in kind and layer.last_input is not None:
        basis = layer.basis(layer.last_input.to(device)).detach().float()
        local = basis.reshape(-1, k)
        mat = (local.T @ local) / max(1, local.shape[0])
        if "sob" in kind:
            sob = _metric_matrix(layer, "h1-low", params, device)[0]
            mat = mat + params.fc_data_sob_alpha * sob
        mat = mat + params.fc_rho * torch.eye(k, device=device)
    else:
        if isinstance(layer, ABRBFDense):
            gram = _get_sobolev_gram(
                layer,
                cfg,
                state,
                alpha=alpha,
                beta=0.0,
                rho=params.fc_rho,
                device=device,
                dtype=layer.coeff.dtype,
            )["A"].detach().float()
            mat = torch.zeros(k, k, device=device)
            mat[:2, :2] = torch.eye(2, device=device)
            mat[2 : 2 + rbf_k, 2 : 2 + rbf_k] = gram
            mat = mat + params.fc_rho * torch.eye(k, device=device)
        else:
            gram = _get_sobolev_gram(
                layer,
                cfg,
                state,
                alpha=alpha,
                beta=0.0,
                rho=params.fc_rho,
                device=device,
                dtype=layer.coeff.dtype,
            )
            mat = gram["A"].detach().float().to(device)
    mat = 0.5 * (mat + mat.T)
    eig = torch.linalg.eigvalsh(mat).clamp_min(1e-12)
    return mat, {
        "metric_condition": float((eig[-1] / eig[0]).detach().cpu()),
        "metric_eig_min": float(eig[0].detach().cpu()),
        "metric_eig_max": float(eig[-1].detach().cpu()),
    }


def _fcadam_updates(
    model: nn.Module,
    state: FCAdamState,
    params: V44Params,
    *,
    metric_kind: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    updates: Dict[str, torch.Tensor] = {}
    conds: List[float] = []
    recon_errors: List[float] = []
    u_grad_norms: List[float] = []
    u_update_norms: List[float] = []
    a_update_norms: List[float] = []
    fn_norms: List[float] = []
    for name, p in coefficient_named_params(model):
        if p.grad is None:
            continue
        layer = _layer_for_name(model, name)
        mat, stats = _metric_matrix(layer, metric_kind, params, p.device)
        chol = torch.linalg.cholesky(mat.float())
        grad = p.grad.detach().float()
        flat = grad.reshape(-1, grad.shape[-1])
        grad_u = torch.linalg.solve_triangular(chol, flat.T, upper=False).T.reshape_as(grad)
        t = state.t.get(name, 0) + 1
        state.t[name] = t
        m = state.m.get(name, torch.zeros_like(grad_u))
        v = state.v.get(name, torch.zeros_like(grad_u))
        m = params.fc_beta1 * m + (1.0 - params.fc_beta1) * grad_u
        v = params.fc_beta2 * v + (1.0 - params.fc_beta2) * grad_u.square()
        state.m[name] = m.detach()
        state.v[name] = v.detach()
        mhat = m / max(1e-12, 1.0 - params.fc_beta1**t)
        vhat = v / max(1e-12, 1.0 - params.fc_beta2**t)
        du = -params.fc_lr * mhat / (vhat.sqrt() + params.fc_eps)
        flat_du = du.reshape(-1, du.shape[-1])
        da = torch.linalg.solve_triangular(chol.T, flat_du.T, upper=True).T.reshape_as(p).to(dtype=p.dtype)
        coeff_flat = p.detach().reshape(-1, p.shape[-1]).double()
        chol64 = chol.double()
        # Row-vector convention: a = u @ L^{-1}, therefore u = a @ L.
        u = torch.matmul(coeff_flat, chol64)
        recon = torch.linalg.solve_triangular(chol64.T, u.T, upper=True).T
        recon_errors.append(float(((recon - coeff_flat).norm() / coeff_flat.norm().clamp_min(1e-12)).detach().cpu()))
        u_grad_norms.append(float(grad_u.norm().detach().cpu()))
        u_update_norms.append(float(du.norm().detach().cpu()))
        a_update_norms.append(float(da.float().norm().detach().cpu()))
        fn_norms.append(float(torch.sqrt((da.float().reshape(-1, da.shape[-1]) @ mat * da.float().reshape(-1, da.shape[-1])).sum().clamp_min(1e-24)).detach().cpu()))
        conds.append(stats["metric_condition"])
        updates[name] = da
    return updates, {
        "optimizer_family": "fcadam",
        "optimizer_state_type": "adam_in_function_coordinate",
        "adam_state_in_function_coordinate": 1,
        "metric_kind": metric_kind,
        "metric_condition_mean": _mean(conds, float("nan")),
        "coordinate_reconstruction_error": _mean(recon_errors, float("nan")),
        "u_grad_norm": _mean(u_grad_norms, float("nan")),
        "u_update_norm": _mean(u_update_norms, float("nan")),
        "a_update_norm": _mean(a_update_norms, float("nan")),
        "functional_update_norm": _mean(fn_norms, float("nan")),
    }


def _layer_for_name(model: nn.Module, name: str) -> RBFDense:
    layers = list(model.kan_layers())  # type: ignore[attr-defined]
    names = [n for n, _ in coefficient_named_params(model)]
    return layers[names.index(name)]


def _target_fit_layer(
    layer: RBFDense,
    target: torch.Tensor,
    params: V44Params,
    *,
    sob_lambda: float,
    ridge: float,
    trust: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if layer.last_input is None:
        return torch.zeros_like(layer.coeff), {"fit_r2": float("nan"), "fit_residual": float("nan"), "condition": float("nan")}
    basis = layer.basis(layer.last_input.to(layer.coeff.device)).detach().float()
    target = target.detach().float().to(layer.coeff.device)
    k = int(layer.coeff.shape[-1])
    mat, _ = _metric_matrix(layer, "h1-low", params, layer.coeff.device)
    prior = sob_lambda * mat + ridge * torch.eye(k, device=layer.coeff.device)
    prior = 0.5 * (prior + prior.T)
    chol_prior = torch.linalg.cholesky(prior.float())
    bsz, in_dim, _ = basis.shape
    rhs_by_in = basis.permute(1, 2, 0).contiguous()
    chol_batch = chol_prior.unsqueeze(0).expand(in_dim, -1, -1).contiguous()
    rinv_phi_t = torch.cholesky_solve(rhs_by_in, chol_batch)
    kernel = torch.einsum("bik,ikc->bc", basis, rinv_phi_t)
    kernel = 0.5 * (kernel + kernel.T)
    system = kernel + torch.eye(bsz, device=kernel.device)
    eig = torch.linalg.eigvalsh(system.float()).clamp_min(1e-12)
    chol_system = torch.linalg.cholesky(system.float())
    alpha = torch.cholesky_solve(target.float(), chol_system)
    delta_iko = torch.einsum("ikb,bo->iko", rinv_phi_t, alpha)
    delta = delta_iko.permute(2, 0, 1).contiguous().to(dtype=layer.coeff.dtype)
    pred = torch.einsum("bik,oik->bo", basis.to(dtype=delta.dtype), delta).float()
    ratio = float((pred.norm() / target.norm().clamp_min(1e-12)).detach().cpu())
    shrink = 1.0
    if trust > 0 and ratio > trust:
        shrink = trust / max(1e-12, ratio)
        delta = delta * shrink
        pred = pred * shrink
    residual = pred - target
    r2 = 1.0 - float((residual.square().sum() / target.square().sum().clamp_min(1e-12)).detach().cpu())
    return delta, {
        "fit_r2": r2,
        "fit_residual": float((residual.norm() / target.norm().clamp_min(1e-12)).detach().cpu()),
        "condition": float((eig[-1] / eig[0]).detach().cpu()),
        "trust_shrink": shrink,
    }


def _sfd_updates(
    model: nn.Module,
    shadow: AdamState,
    params: V44Params,
    *,
    mode: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    adam_update, _ = _adam_updates(model, shadow, lr=params.adam_lr, mutate=True)
    updates: Dict[str, torch.Tensor] = {}
    r2s: List[float] = []
    residuals: List[float] = []
    conds: List[float] = []
    shrinks: List[float] = []
    for name, p in coefficient_named_params(model):
        layer = _layer_for_name(model, name)
        d_adam = adam_update.get(name)
        if d_adam is None or layer.last_input is None:
            continue
        with torch.no_grad():
            basis = layer.basis(layer.last_input.to(layer.coeff.device))
            target = torch.einsum("bik,oik->bo", basis, d_adam.to(layer.coeff.device)).detach().float()
            if mode == "residual":
                target = 0.5 * target
            if mode == "prox":
                target = target * 0.75
        delta, stats = _target_fit_layer(
            layer,
            params.sfd_tau * target,
            params,
            sob_lambda=params.sfd_lambda,
            ridge=params.sfd_ridge,
            trust=params.sfd_trust if mode in {"prox", "residual"} else params.sfd_trust * 1.5,
        )
        updates[name] = delta
        r2s.append(stats["fit_r2"])
        residuals.append(stats["fit_residual"])
        conds.append(stats["condition"])
        shrinks.append(stats["trust_shrink"])
    return updates, {
        "optimizer_family": "sfd",
        "shadow_adam_state_exists": 1,
        "sfd_mode": mode,
        "teacher_student_function_r2": _mean(r2s, float("nan")),
        "SFD_fit_residual": _mean(residuals, float("nan")),
        "SFD_step_shrink_rate": 1.0 - _mean(shrinks, 1.0),
        "metric_condition_mean": _mean(conds, float("nan")),
    }


def _gfk_updates(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    params: V44Params,
    *,
    mode: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    logits = model(xb)
    prob = logits.softmax(dim=-1).detach()
    onehot = F.one_hot(yb, num_classes=logits.shape[-1]).float()
    output_target = params.gfk_tau * (onehot - prob)
    updates: Dict[str, torch.Tensor] = {}
    r2s: List[float] = []
    conds: List[float] = []
    roles: List[str] = []
    for name, p in coefficient_named_params(model):
        role = _role_group(_role_for_name(name))
        if mode == "output-only" and role != "output":
            continue
        if mode == "block-output" and role not in {"block", "output"}:
            continue
        layer = _layer_for_name(model, name)
        if role == "output":
            target = output_target
        elif layer.last_output_grad is not None:
            target = -params.gfk_tau * layer.last_output_grad.detach().float()
        else:
            continue
        delta, stats = _target_fit_layer(
            layer,
            target,
            params,
            sob_lambda=params.gfk_sob_lambda,
            ridge=params.gfk_ridge,
            trust=params.gfk_trust,
        )
        updates[name] = delta
        r2s.append(stats["fit_r2"])
        conds.append(stats["condition"])
        roles.append(role)
    return updates, {
        "optimizer_family": "gfk",
        "gfk_mode": mode,
        "kernel_condition": _mean(conds, float("nan")),
        "predicted_logit_improvement": _mean(r2s, float("nan")),
        "roles_updated": ",".join(sorted(set(roles))),
    }


def _compute_loss_and_grad(model: nn.Module, xb: torch.Tensor, yb: torch.Tensor) -> float:
    model.train()
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(xb), yb)
    loss.backward()
    return float(loss.detach().cpu())


def _method_updates(
    model: nn.Module,
    dataset: str,
    seed: int,
    method: str,
    params: V44Params,
    run_state: V44RunState,
    xb: torch.Tensor,
    yb: torch.Tensor,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    key = method.lower().replace("-", "_")
    if key in {"purekan_adamw", "purekan_adamw_one_step", "purekan_ab_rbf_adamw_smoke"}:
        updates, rms = _adam_updates(model, run_state.adam, lr=params.adam_lr, mutate=True)
        return updates, {"optimizer_family": "adamw", "optimizer_state_type": "raw_adamw", "adam_rms_mean": _mean([float(v.mean().cpu()) for v in rms.values()], float("nan"))}
    if key in {"d0_allfullsobolev", "d6_alltaskaware", "f4_fng_leftfullright", "fpa_edge_prox", "ekfng_leftrightfull_lowsob"}:
        v43_params = V43Params(train_size=params.train_size, val_size=params.val_size, test_size=params.test_size, batch_size=params.batch_size, hidden_dim=params.hidden_dim, depth=params.depth, basis_count=params.basis_count)
        updates, stats = _v43_method_updates(model, dataset, seed, method.replace("EKFNG", "EKFNG"), v43_params, V43RunState())
        stats["optimizer_family"] = stats.get("optimizer_family", "v43")
        return updates, stats
    if key in {"fcadam_l2", "fcadam_l2_smoke"}:
        return _fcadam_updates(model, run_state.fc, params, metric_kind="l2")
    if key in {"fcadam_h1", "fcadam_h1_smoke"}:
        return _fcadam_updates(model, run_state.fc, params, metric_kind="h1")
    if key in {"fcadam_h1_low", "ab_rbf_fcadam_h1_low", "purekan_ab_rbf_fcadam_smoke"}:
        return _fcadam_updates(model, run_state.fc, params, metric_kind="h1-low")
    if key == "fcadam_datasob":
        return _fcadam_updates(model, run_state.fc, params, metric_kind="dataSob")
    if key in {"sfd_direct", "sfd_direct_smoke"}:
        return _sfd_updates(model, run_state.shadow_adam, params, mode="direct")
    if key == "sfd_prox":
        return _sfd_updates(model, run_state.shadow_adam, params, mode="prox")
    if key == "sfd_residual":
        return _sfd_updates(model, run_state.shadow_adam, params, mode="residual")
    if key in {"gfk_output_only", "gfk_output_only_smoke"}:
        return _gfk_updates(model, xb, yb, params, mode="output-only")
    if key == "gfk_block_output":
        return _gfk_updates(model, xb, yb, params, mode="block-output")
    if key in {"gfk_all_lowrank", "gfk_all_lowrank_smoke"}:
        return _gfk_updates(model, xb, yb, params, mode="all-lowrank")
    raise ValueError(f"unknown v4.4 method: {method}")


def _estimate_geometry_any(model: nn.Module, x: torch.Tensor, *, device: torch.device, batch_size: int) -> Dict[str, float]:
    if isinstance(model, PureKANClassifier):
        return estimate_geometry(model, x, device=device, batch_size=batch_size)
    model.eval()
    xb = x[:batch_size].to(device)
    h = model.input_kan(xb)  # type: ignore[attr-defined]
    phi_values: List[torch.Tensor] = []
    cond_values: List[torch.Tensor] = []
    curvature = torch.tensor(0.0, device=device)
    for block in model.blocks:  # type: ignore[attr-defined]
        z = block.norm(h)
        basis, deriv = block.kan.basis_and_derivative(z)
        coeff = block.kan.coeff
        dy_dx = torch.einsum("bik,oik->boi", deriv, coeff)
        if dy_dx.shape[-1] == dy_dx.shape[-2]:
            eye = torch.eye(dy_dx.shape[-1], device=device).unsqueeze(0)
            jac = eye + float(block.alpha.detach()) * dy_dx
            cond_values.append(torch.linalg.cond(jac.float()).clamp(max=1e6))
        phi_values.append(dy_dx.abs().flatten())
        branch = block.alpha * torch.einsum("bik,oik->bo", basis, coeff)
        h = h + branch
        curvature = curvature + coeff.square().mean()
    phi = torch.cat(phi_values) if phi_values else torch.tensor([0.0], device=device)
    cond_cat = torch.cat(cond_values) if cond_values else torch.tensor([1.0], device=device)
    return {
        "phi_prime_p95": float(torch.quantile(phi, 0.95).detach().cpu()),
        "phi_prime_max": float(phi.max().detach().cpu()),
        "max_jac_condition": float(cond_cat.max().detach().cpu()),
        "credit_amplification_p95": float(torch.quantile(cond_cat, 0.95).detach().cpu()),
        "curvature_energy": float(curvature.detach().cpu()),
    }


def _function_displacement(
    model: nn.Module,
    updates: Dict[str, torch.Tensor],
    xb: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    named = coefficient_named_params(model)
    before = model(xb).detach()
    snap = _snapshot(named)
    _apply_updates(named, updates)
    after = model(xb).detach()
    _restore(named, snap)
    delta = (after - before).detach().flatten().float()
    drift = float(delta.norm().detach().cpu() / before.detach().flatten().float().norm().clamp_min(1e-12).cpu())
    return delta, drift


def _r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.numel() == 0 or target.numel() == 0:
        return float("nan")
    n = min(pred.numel(), target.numel())
    pred = pred[:n].float()
    target = target[:n].float()
    return 1.0 - float(((pred - target).square().sum() / target.square().sum().clamp_min(1e-12)).detach().cpu())


def _nonkan_count(model: nn.Module) -> int:
    coeff_ids = {id(p) for _, p in coefficient_named_params(model)}
    return sum(p.numel() for p in model.parameters() if p.requires_grad and id(p) not in coeff_ids)


def _coverage_row(model: nn.Module) -> Dict[str, float]:
    named = coefficient_named_params(model)
    total = sum(p.numel() for _, p in named)
    return {
        "functional_coverage": 1.0 if total else 0.0,
        "input_coeff_seen": 1.0 if any(n.startswith("input_kan.") for n, _ in named) else 0.0,
        "block_coeff_seen": 1.0 if any(n.startswith("blocks.") for n, _ in named) else 0.0,
        "output_coeff_seen": 1.0 if any(n.startswith("output_kan.") for n, _ in named) else 0.0,
    }


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = []
    device = get_device(args.device)
    params = V44Params(train_size=512, val_size=128, test_size=128)
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=512, val_size=128, test_size=128, seed=0, download=not args.no_download, allow_fake_data=args.allow_fake_data)
        idx = next(iter(iter_minibatches(len(bundle.x_train), params.batch_size, 0)))
        xb = bundle.x_train[idx].to(device)
        yb = bundle.y_train[idx].to(device)
        for method in P0_METHODS:
            set_seed(0)
            ab = "AB-RBF" in method
            model = _make_model(bundle, params, device, ab=ab)
            named = coefficient_named_params(model)
            snap = _snapshot(named)
            start = time.perf_counter()
            err = ""
            stats: Dict[str, Any] = {}
            try:
                _compute_loss_and_grad(model, xb, yb)
                updates, stats = _method_updates(model, dataset, 0, method, params, V44RunState(), xb, yb)
                _apply_updates(named, updates)
            except Exception as exc:
                err = repr(exc)
            rollback_error = 0.0
            try:
                _restore(named, snap)
                rollback_error = max(float((p.detach() - snap[n]).abs().max().cpu()) for n, p in named) if named else 0.0
            except Exception as exc:
                err = err or repr(exc)
            step_ms = 1000.0 * (time.perf_counter() - start)
            conds = [float(v) for k, v in stats.items() if "condition" in k and isinstance(v, (float, int)) and math.isfinite(float(v))]
            cov = _coverage_row(model)
            row = {
                "stage": "P0",
                "dataset": dataset,
                "method": method,
                "learnable_nonKAN_params": _nonkan_count(model),
                "raw_nonKAN_params": _nonkan_count(model),
                "basis_type": "AB-RBF" if ab else "RBF",
                "basis_count": params.basis_count + (2 if ab else 0),
                "has_affine_basis": int(ab),
                "alpha_trainable": 0,
                "norm_mode": "fixed",
                "optimizer_state_type": stats.get("optimizer_state_type", stats.get("optimizer_family", "")),
                "adam_state_in_function_coordinate": stats.get("adam_state_in_function_coordinate", 0),
                "shadow_adam_state_exists": stats.get("shadow_adam_state_exists", 0),
                "metric_condition_input": _mean(conds, float("nan")),
                "metric_condition_block_mean": _mean(conds, float("nan")),
                "metric_condition_output": _mean(conds, float("nan")),
                "coordinate_reconstruction_error": stats.get("coordinate_reconstruction_error", 0.0 if "FCAdam" not in method else float("nan")),
                "rollback_error_for_shadow_step": rollback_error,
                "step_time_ms": step_ms,
                "error": err,
            }
            row.update(cov)
            rows.append(row)
            print(f"P0 {dataset} {method} nonKAN={row['learnable_nonKAN_params']} cov={row['functional_coverage']:.1f} err={err}")
    write_csv(out_dir / "p0_invariants.csv", rows)
    return rows


def _one_step_row(
    *,
    model: nn.Module,
    bundle: Any,
    dataset: str,
    seed: int,
    method: str,
    params: V44Params,
    device: torch.device,
    idx: np.ndarray,
) -> Dict[str, Any]:
    xb = bundle.x_train[idx].to(device)
    yb = bundle.y_train[idx].to(device)
    hb = bundle.x_val[: len(idx)].to(device)
    yh = bundle.y_val[: len(idx)].to(device)
    named = coefficient_named_params(model)
    train_before = F.cross_entropy(model(xb), yb).item()
    hold_before = F.cross_entropy(model(hb), yh).item()
    feat_before = _feature_audit(model, xb, yb)
    geom_before = _estimate_geometry_any(model, hb, device=device, batch_size=min(params.audit_batch_size, len(hb)))
    _compute_loss_and_grad(model, xb, yb)
    adam_ref, _ = _adam_updates(model, AdamState(), lr=params.adam_lr, mutate=True)
    adam_fun, adam_drift = _function_displacement(model, adam_ref, xb)
    updates, stats = _method_updates(model, dataset, seed, method, params, V44RunState(), xb, yb)
    cand_fun, cand_drift = _function_displacement(model, updates, xb)
    _apply_updates(named, updates)
    train_after = F.cross_entropy(model(xb), yb).item()
    hold_after = F.cross_entropy(model(hb), yh).item()
    feat_after = _feature_audit(model, xb, yb)
    geom_after = _estimate_geometry_any(model, hb, device=device, batch_size=min(params.audit_batch_size, len(hb)))
    update_flat = _flat_updates(updates)
    adam_flat = _flat_updates(adam_ref)
    role_norms: Dict[str, List[float]] = {}
    adam_role_norms: Dict[str, List[float]] = {}
    for name, upd in updates.items():
        role_norms.setdefault(_role_group(_role_for_name(name)), []).append(float(upd.detach().float().norm().cpu()))
    for name, upd in adam_ref.items():
        adam_role_norms.setdefault(_role_group(_role_for_name(name)), []).append(float(upd.detach().float().norm().cpu()))
    return {
        "stage": "P1",
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "train_loss_before": train_before,
        "train_loss_after": train_after,
        "holdout_loss_before": hold_before,
        "holdout_loss_after": hold_after,
        "actual_train_descent": train_before - train_after,
        "actual_holdout_descent": hold_before - hold_after,
        "bad_train_step": int(train_after > train_before),
        "cos_coeff_with_adam": _safe_cos(update_flat, adam_flat),
        "cos_function_with_adam": _safe_cos(cand_fun, adam_fun),
        "function_displacement_r2_vs_adam": _r2(cand_fun, adam_fun),
        "projection_error_vs_adam": float((cand_fun - adam_fun).norm().cpu() / adam_fun.norm().clamp_min(1e-12).cpu()) if cand_fun.numel() else float("nan"),
        "candidate_logit_drift": cand_drift,
        "adam_logit_drift": adam_drift,
        "adam_delta_coeff_norm_input": _mean(adam_role_norms.get("input", []), 0.0),
        "adam_delta_coeff_norm_block": _mean(adam_role_norms.get("block", []), 0.0),
        "adam_delta_coeff_norm_output": _mean(adam_role_norms.get("output", []), 0.0),
        "candidate_delta_coeff_norm_input": _mean(role_norms.get("input", []), 0.0),
        "candidate_delta_coeff_norm_block": _mean(role_norms.get("block", []), 0.0),
        "candidate_delta_coeff_norm_output": _mean(role_norms.get("output", []), 0.0),
        "margin_change_candidate": feat_after["margin_mean"] - feat_before["margin_mean"],
        "rank_change_candidate": feat_after["block_feature_effective_rank"] - feat_before["block_feature_effective_rank"],
        "margin_change_error_vs_adam": float("nan"),
        "rank_change_error_vs_adam": float("nan"),
        "feature_rank_before": feat_before["block_feature_effective_rank"],
        "feature_rank_after": feat_after["block_feature_effective_rank"],
        "margin_p10_before": feat_before["margin_p10"],
        "margin_p10_after": feat_after["margin_p10"],
        "phi_prime_p95_before": geom_before["phi_prime_p95"],
        "phi_prime_p95_after": geom_after["phi_prime_p95"],
        "jacobian_condition_before": geom_before["max_jac_condition"],
        "jacobian_condition_after": geom_after["max_jac_condition"],
        "metric_condition": stats.get("metric_condition_mean", stats.get("metric_condition_right", stats.get("kernel_condition", float("nan")))),
        "coordinate_reconstruction_error": stats.get("coordinate_reconstruction_error", float("nan")),
        "teacher_student_function_r2": stats.get("teacher_student_function_r2", float("nan")),
        "u_update_norm": stats.get("u_update_norm", float("nan")),
        "a_update_norm": stats.get("a_update_norm", float("nan")),
        "error": stats.get("error", ""),
    }


def run_p1(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = []
    device = get_device(args.device)
    params = V44Params(train_size=512, val_size=128, test_size=128)
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=512, val_size=128, test_size=128, seed=0, download=not args.no_download, allow_fake_data=args.allow_fake_data)
        idx = next(iter(iter_minibatches(len(bundle.x_train), params.batch_size, 13)))
        for method in P1_METHODS:
            set_seed(0)
            model = _make_model(bundle, params, device, ab=("AB-RBF" in method))
            try:
                row = _one_step_row(model=model, bundle=bundle, dataset=dataset, seed=0, method=method, params=params, device=device, idx=idx)
            except Exception as exc:
                row = {"stage": "P1", "dataset": dataset, "seed": 0, "method": method, "error": repr(exc)}
                if not args.continue_on_error:
                    raise
            rows.append(row)
            print(f"P1 {dataset} {method} trainΔ={row.get('actual_train_descent','')} holdΔ={row.get('actual_holdout_descent','')} r2={row.get('function_displacement_r2_vs_adam','')}")
    write_csv(out_dir / "p1_adam_functional_trajectory.csv", rows)
    return rows


def _run_horizon(
    args: argparse.Namespace,
    *,
    dataset: str,
    seed: int,
    method: str,
    params: V44Params,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    device = get_device(args.device)
    bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=params.train_size, val_size=params.val_size, test_size=params.test_size, seed=seed, download=not args.no_download, allow_fake_data=args.allow_fake_data)
    set_seed(seed)
    model = _make_model(bundle, params, device, ab=("AB-RBF" in method))
    run_state = V44RunState()
    idxs = list(iter_minibatches(len(bundle.x_train), params.batch_size, seed + 77))[: params.p2_steps]
    xb0 = bundle.x_train[idxs[0]].to(device)
    yb0 = bundle.y_train[idxs[0]].to(device)
    hb = bundle.x_val[: params.batch_size].to(device)
    yh = bundle.y_val[: params.batch_size].to(device)
    train_start = F.cross_entropy(model(xb0), yb0).item()
    hold_start = F.cross_entropy(model(hb), yh).item()
    feat_start = _feature_audit(model, hb, yh)
    geom_start = _estimate_geometry_any(model, hb, device=device, batch_size=params.audit_batch_size)
    trace: List[Dict[str, Any]] = []
    bad = 0
    cos_vals: List[float] = []
    r2_vals: List[float] = []
    train_after_1 = hold_after_1 = float("nan")
    start = time.perf_counter()
    for step, idx in enumerate(idxs):
        xb = bundle.x_train[idx].to(device)
        yb = bundle.y_train[idx].to(device)
        train_before = F.cross_entropy(model(xb), yb).item()
        hold_before = F.cross_entropy(model(hb), yh).item()
        _compute_loss_and_grad(model, xb, yb)
        adam_ref, _ = _adam_updates(model, AdamState(), lr=params.adam_lr, mutate=True)
        adam_fun, _ = _function_displacement(model, adam_ref, xb)
        updates, stats = _method_updates(model, dataset, seed, method, params, run_state, xb, yb)
        cand_fun, cand_drift = _function_displacement(model, updates, xb)
        cos = _safe_cos(_flat_updates(updates), _flat_updates(adam_ref))
        r2v = _r2(cand_fun, adam_fun)
        cos_vals.append(cos)
        r2_vals.append(r2v)
        _apply_updates(coefficient_named_params(model), updates)
        train_after = F.cross_entropy(model(xb), yb).item()
        hold_after = F.cross_entropy(model(hb), yh).item()
        if train_after > train_before:
            bad += 1
        if step == 0:
            train_after_1 = train_after
            hold_after_1 = hold_after
        trace.append(
            {
                "stage": "P2",
                "dataset": dataset,
                "seed": seed,
                "method": method,
                "step": step,
                "train_loss_before": train_before,
                "train_loss_after": train_after,
                "holdout_loss_before": hold_before,
                "holdout_loss_after": hold_after,
                "cos_to_adam": cos,
                "function_r2_to_adam": r2v,
                "logit_drift": cand_drift,
                "metric_condition": stats.get("metric_condition_mean", stats.get("metric_condition_right", stats.get("kernel_condition", float("nan")))),
            }
        )
    wall = time.perf_counter() - start
    train_end = F.cross_entropy(model(xb0), yb0).item()
    hold_end = F.cross_entropy(model(hb), yh).item()
    feat_end = _feature_audit(model, hb, yh)
    geom_end = _estimate_geometry_any(model, hb, device=device, batch_size=params.audit_batch_size)
    occ = rbf_basis_occupancy_audit(model)
    test = evaluate(model, bundle.x_test, bundle.y_test, device=device, batch_size=params.eval_batch_size)
    row = {
        "stage": "P2",
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "steps": params.p2_steps,
        "test_acc_after_5step": test["acc"],
        "train_loss_descent_1step": train_start - train_after_1,
        "holdout_loss_descent_1step": hold_start - hold_after_1,
        "train_loss_descent_5step": train_start - train_end,
        "holdout_loss_descent_5step": hold_start - hold_end,
        "bad_step_rate": bad / max(1, params.p2_steps),
        "cumulative_margin_change": feat_end["margin_mean"] - feat_start["margin_mean"],
        "margin_p10_after": feat_end["margin_p10"],
        "cumulative_rank_change": feat_end["block_feature_effective_rank"] - feat_start["block_feature_effective_rank"],
        "feature_rank_after": feat_end["block_feature_effective_rank"],
        "cumulative_logit_drift": _mean([float(r["logit_drift"]) for r in trace], float("nan")),
        "cos_to_adam_mean": _mean(cos_vals, float("nan")),
        "function_r2_to_adam_mean": _mean(r2_vals, float("nan")),
        "phi_prime_p95_change": geom_end["phi_prime_p95"] - geom_start["phi_prime_p95"],
        "phi_prime_p95_after": geom_end["phi_prime_p95"],
        "jacobian_change": geom_end["max_jac_condition"] - geom_start["max_jac_condition"],
        "jacobian_after": geom_end["max_jac_condition"],
        "basis_occupancy_entropy_change": occ.get("rbf_basis_mean", float("nan")),
        "step_time_ms": 1000.0 * wall / max(1, params.p2_steps),
        "error": "",
    }
    return row, trace


def run_p2(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows = [] if args.fresh else read_csv(out_dir / "p2_horizon_audit.csv")
    trace_rows = [] if args.fresh else read_csv(out_dir / "optimizer_dynamics_trace.csv")
    params = V44Params()
    datasets = [dataset_name(d) for d in parse_str_list(args.datasets)]
    seeds = parse_int_list(args.seeds)
    if args.seeds == add_args().get_default("seeds"):
        seeds = [0, 1, 2]
    wanted = set(parse_str_list(args.methods))
    methods = [m for m in P2_METHODS if not wanted or m in wanted]
    done = {(r.get("dataset"), int(r.get("seed", -1)), r.get("method")) for r in rows if not r.get("error")}
    for dataset in datasets:
        for method in methods:
            for seed in seeds:
                if (dataset, seed, method) in done:
                    continue
                try:
                    row, trace = _run_horizon(args, dataset=dataset, seed=seed, method=method, params=params)
                    rows.append(row)
                    trace_rows.extend(trace)
                    print(f"P2 {dataset} {method} seed={seed} hold5={row['holdout_loss_descent_5step']:.4g} r2={row['function_r2_to_adam_mean']:.3f}")
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    rows.append({"stage": "P2", "dataset": dataset, "seed": seed, "method": method, "error": repr(exc)})
                    print(f"P2 ERROR {dataset} {method} seed={seed}: {exc!r}")
                write_csv(out_dir / "p2_horizon_audit.csv", rows)
                write_csv(out_dir / "optimizer_dynamics_trace.csv", trace_rows)
    return rows


def _blank_later_files(out_dir: Path, reason: str) -> None:
    for name in [
        "p3_short_horizon_train_scorecard.csv",
        "p4_fcadam_scorecard.csv",
        "p5_sfd_scorecard.csv",
        "p6_gfk_scorecard.csv",
        "p7_candidate_selection.csv",
        "p8_confirm5.csv",
        "p9_confirm10.csv",
    ]:
        write_csv(out_dir / name, [{"stage": name.split("_")[0].upper(), "status": "not_run", "reason": reason}])


def add_args() -> argparse.ArgumentParser:
    p = add_v3_args()
    p.description = __doc__
    p.set_defaults(packages="V4_4_P0_SMOKE", datasets="MNIST,Fashion-MNIST,KMNIST", out_dir=Path("results/v4_4"), seeds="0,1,2")
    return p


def main() -> int:
    args = add_args().parse_args()
    all_rows: List[Dict[str, Any]] = []
    for package in parse_str_list(args.packages):
        key = package.strip().upper().replace("-", "_")
        if key == "V4_4_P0_SMOKE":
            all_rows = run_p0(args)
        elif key == "V4_4_P1_TRAJECTORY":
            all_rows = run_p1(args)
        elif key == "V4_4_P2_HORIZON":
            all_rows = run_p2(args)
            _blank_later_files(ensure_dir(args.out_dir), "pending analyzer gate decision")
        else:
            raise ValueError(f"unknown v4.4 package: {package}")
    return 0 if all_rows is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
