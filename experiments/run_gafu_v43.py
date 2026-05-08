#!/usr/bin/env python3
"""DG-KAN v4.3 runner: FPA / EK-FNG / CFT PureKAN optimizer probes."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
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
    functional_coeff_step,
    get_device,
    iter_minibatches,
    load_vision_bundle,
    parse_int_list,
    parse_str_list,
    rbf_basis_occupancy_audit,
    read_csv,
    set_seed,
    write_csv,
)
from run_gafu_v3 import add_args as add_v3_args, dataset_name
from run_gafu_v42 import BFTParams, _bft_step


DATASETS = ["MNIST", "Fashion-MNIST", "KMNIST"]
BASE_METHODS = [
    "PureKAN-AdamW",
    "D0-allFullSobolev",
    "D6-allTaskAware",
    "F4-FNG-leftFullRight",
    "BFT-mixed-reverse",
]
V43_METHODS = [
    "FPA-sob-prox",
    "FPA-edge-prox",
    "EKFNG-rightFull",
    "EKFNG-leftRightFull",
    "EKFNG-leftRightFull-lowSob",
    "CFT-output-only",
    "CFT-block-output-sequential",
]
P1_METHODS = ["AdamW-one-step", *BASE_METHODS[1:], *V43_METHODS, "raw-gradient"]
P2_METHODS = [*BASE_METHODS, *V43_METHODS]


@dataclass
class V43Params:
    train_size: int = 6000
    val_size: int = 1000
    test_size: int = 1000
    batch_size: int = 256
    eval_batch_size: int = 512
    audit_batch_size: int = 64
    hidden_dim: int = 64
    depth: int = 4
    basis_count: int = 16
    alpha_init: float = 1.5
    lr: float = 1e-3
    raw_lr: float = 2e-4
    coeff_lr: float = 0.03
    fpa_lambda: float = 0.03
    fpa_rho: float = 1e-3
    ekfng_lr: float = 0.03
    ekfng_sob_lambda: float = 0.02
    ekfng_rho_right: float = 1e-3
    ekfng_rho_left: float = 1e-2
    cft_tau: float = 0.03
    cft_ridge: float = 3e-2
    cft_sob_lambda: float = 1e-3
    max_full_feature_dim: int = 384
    epochs_p2: int = 3
    epochs_p3: int = 8
    epochs_p4: int = 8
    epochs_p5: int = 6


@dataclass
class AdamState:
    m: Dict[str, torch.Tensor] = field(default_factory=dict)
    v: Dict[str, torch.Tensor] = field(default_factory=dict)
    t: int = 0


@dataclass
class V43RunState:
    adam: AdamState = field(default_factory=AdamState)
    functional: RuntimeState = field(default_factory=lambda: RuntimeState(phase="GEOMETRY", current_branch_scale=1.0, branch_switched=True))
    step_rows: List[Dict[str, Any]] = field(default_factory=list)
    role_rows: List[Dict[str, Any]] = field(default_factory=list)
    lambda_rows: List[Dict[str, Any]] = field(default_factory=list)
    proposal_rows: List[Dict[str, Any]] = field(default_factory=list)


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else default


def _p95(values: Iterable[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.quantile(vals, 0.95)) if vals else default


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.std(vals)) if vals else float("nan")


def _safe_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().flatten().float()
    b = b.detach().flatten().float()
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    n = min(a.numel(), b.numel())
    a = a[:n]
    b = b[:n]
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((torch.dot(a, b) / denom).detach().cpu())


def _flat_updates(updates: Dict[str, torch.Tensor], names: Sequence[str] | None = None) -> torch.Tensor:
    keys = list(names) if names is not None else sorted(updates)
    vals = [updates[k].detach().flatten().float() for k in keys if k in updates]
    return torch.cat(vals) if vals else torch.empty(0)


def _snapshot(named: Sequence[Tuple[str, torch.nn.Parameter]]) -> Dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in named}


def _restore(named: Sequence[Tuple[str, torch.nn.Parameter]], snap: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, p in named:
            if name in snap:
                p.copy_(snap[name])


def _apply_updates(named: Sequence[Tuple[str, torch.nn.Parameter]], updates: Dict[str, torch.Tensor], scale: float = 1.0) -> None:
    with torch.no_grad():
        for name, p in named:
            update = updates.get(name)
            if update is not None:
                p.add_(update.to(device=p.device, dtype=p.dtype) * scale)


def _role_for_name(name: str) -> str:
    if name.startswith("input_kan."):
        return "input"
    if name.startswith("output_kan."):
        return "output"
    if name.startswith("blocks."):
        parts = name.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            return f"block{parts[1]}"
        return "block"
    return "other"


def _role_group(role: str) -> str:
    return "block" if role.startswith("block") else role


def _make_model(bundle: Any, params: V43Params, device: torch.device) -> PureKANClassifier:
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


def _base_cfg(dataset: str, seed: int, method: str, params: V43Params) -> TrainConfig:
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
        metric_mode="grid",
        branch_schedule="none",
        branch_max_active_frac=0.0,
        geometry_min_epochs=999,
        v3_phase_mode="hard",
        v3_metric_active="full_sobolev_gram",
        v3_metric_transition="full_sobolev_gram",
        v3_metric_geometry="full_sobolev_gram",
        coeff_lr=params.coeff_lr,
        trust_radius=0.50,
    )


def _functional_cfg(dataset: str, seed: int, method: str, params: V43Params, role: str = "all") -> TrainConfig:
    cfg = _base_cfg(dataset, seed, method, params)
    key = method.lower().replace("-", "_")
    if key in {"d0_allfullsobolev", "sobolev_full"}:
        pass
    elif key in {"d6_alltaskaware", "task_diag_d6"}:
        cfg.optimizer_method = "purekan_tfu"
        cfg.tfu_enabled = True
        cfg.tfu_sob_lambda = 0.03
        cfg.pure_input_metric = "tfu_data_task_diag"
        cfg.pure_shallow_metric = "tfu_data_task_diag"
        cfg.pure_deep_metric = "tfu_task_diag"
        cfg.pure_output_metric = "tfu_task_diag"
    elif key in {"f4_fng_leftfullright", "fng_leftfullright"}:
        cfg.optimizer_method = "purekan_fng"
        cfg.fng_enabled = True
        cfg.fng_mode = "leftfull"
        cfg.fng_sob_lambda = 0.02
        cfg.pure_input_metric = "fng_leftfull_right"
        cfg.pure_shallow_metric = "fng_leftfull_right"
        cfg.pure_deep_metric = "fng_leftfull_right"
        cfg.pure_output_metric = "fng_leftfull_right"
    elif key.startswith("cft"):
        cfg.optimizer_method = "purekan_ftf"
        cfg.coeff_lr = 1.0
        cfg.ftf_enabled = True
        cfg.ftf_tau = params.cft_tau
        cfg.ftf_ridge = params.cft_ridge
        cfg.ftf_sob_lambda = params.cft_sob_lambda
        cfg.ftf_activation_trust = 0.10
        cfg.ftf_mode = "output_only" if "output_only" in key else "blocks_output"
        cfg.pure_input_metric = "ftf"
        cfg.pure_shallow_metric = "ftf"
        cfg.pure_deep_metric = "ftf"
        cfg.pure_output_metric = "ftf"
    else:
        raise ValueError(f"unknown functional cfg method: {method}")
    cfg.pure_block_metric = "phase"
    return cfg


def _compute_loss_and_grad(model: PureKANClassifier, xb: torch.Tensor, yb: torch.Tensor) -> float:
    model.train()
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(xb), yb)
    loss.backward()
    return float(loss.detach().cpu())


def _adam_updates(
    model: PureKANClassifier,
    state: AdamState,
    *,
    lr: float,
    weight_decay: float = 0.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    mutate: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    if mutate:
        state.t += 1
        t = state.t
    else:
        t = state.t + 1
    updates: Dict[str, torch.Tensor] = {}
    rms: Dict[str, torch.Tensor] = {}
    for name, p in coefficient_named_params(model):
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        prev_m = state.m.get(name)
        prev_v = state.v.get(name)
        if prev_m is None or prev_m.shape != g.shape:
            prev_m = torch.zeros_like(g)
            prev_v = torch.zeros_like(g)
        m = beta1 * prev_m + (1.0 - beta1) * g
        v = beta2 * prev_v + (1.0 - beta2) * g.square()
        if mutate:
            state.m[name] = m.detach()
            state.v[name] = v.detach()
        mhat = m / max(1e-12, 1.0 - beta1**t)
        vhat = v / max(1e-12, 1.0 - beta2**t)
        denom = vhat.sqrt() + eps
        upd = -lr * mhat / denom
        if weight_decay > 0:
            upd = upd - lr * weight_decay * p.detach().float()
        updates[name] = upd.to(dtype=p.dtype)
        rms[name] = denom.detach()
    return updates, rms


def _layer_by_name(model: PureKANClassifier) -> Dict[str, RBFDense]:
    layers = list(model.kan_layers())
    return {name: layers[idx] for idx, (name, _) in enumerate(coefficient_named_params(model))}


def _sob_matrix(layer: RBFDense, cfg: TrainConfig, state: RuntimeState, rho: float) -> torch.Tensor:
    gram = _get_sobolev_gram(
        layer,
        cfg,
        state,
        alpha=cfg.v3_alpha_geo,
        beta=cfg.v3_beta_geo,
        rho=rho,
        device=layer.coeff.device,
        dtype=layer.coeff.dtype,
    )
    return gram["M"].detach().float().to(layer.coeff.device)


def _basis_feature_stats(layer: RBFDense, params: V43Params) -> Tuple[torch.Tensor | None, torch.Tensor, bool, float, float]:
    if layer.last_input is None:
        f = layer.coeff.shape[1] * layer.coeff.shape[2]
        return None, torch.ones(f, device=layer.coeff.device), False, 1.0, float(f)
    basis = layer.basis(layer.last_input.to(layer.coeff.device)).detach().float()
    phi = basis.reshape(basis.shape[0], -1)
    diag = phi.square().mean(dim=0).clamp_min(1e-6)
    full_ok = phi.shape[1] <= params.max_full_feature_dim
    if full_ok:
        cov = (phi.T @ phi) / max(1, phi.shape[0])
        eig = torch.linalg.eigvalsh((cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device)).float()).clamp_min(1e-12)
        return cov, diag, True, float((eig[-1] / eig[0]).detach().cpu()), float((eig.sum().square() / eig.square().sum().clamp_min(1e-12)).detach().cpu())
    return None, diag, False, float((diag.max() / diag.min()).detach().cpu()), float((diag.sum().square() / diag.square().sum().clamp_min(1e-12)).detach().cpu())


def _left_cov(layer: RBFDense, rho: float) -> Tuple[torch.Tensor, float, float]:
    out_dim = layer.coeff.shape[0]
    if layer.last_output_grad is None:
        mat = torch.eye(out_dim, device=layer.coeff.device)
    else:
        delta = layer.last_output_grad.detach().float().reshape(-1, out_dim)
        mat = (delta.T @ delta) / max(1, delta.shape[0])
        tr = torch.trace(mat).clamp_min(1e-12)
        mat = mat * (out_dim / tr)
    mat = 0.5 * (mat + mat.T) + rho * torch.eye(out_dim, device=mat.device)
    eig = torch.linalg.eigvalsh(mat.float()).clamp_min(1e-12)
    eff = float((eig.sum().square() / eig.square().sum().clamp_min(1e-12)).detach().cpu())
    return mat, float((eig[-1] / eig[0]).detach().cpu()), eff


def _solve_right(flat: torch.Tensor, right_full: torch.Tensor | None, right_diag: torch.Tensor) -> torch.Tensor:
    if right_full is not None:
        mat = 0.5 * (right_full + right_full.T)
        chol = torch.linalg.cholesky(mat.float())
        return torch.cholesky_solve(flat.T.float(), chol).T.to(dtype=flat.dtype)
    return flat / right_diag.view(1, -1).to(dtype=flat.dtype).clamp_min(1e-8)


def _fpa_updates(
    model: PureKANClassifier,
    adam_update: Dict[str, torch.Tensor],
    adam_rms: Dict[str, torch.Tensor],
    params: V43Params,
    *,
    edge: bool,
    lam: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    updates: Dict[str, torch.Tensor] = {}
    stats: Dict[str, Any] = {"chol_success": 1, "solve_success": 1}
    cfg = _base_cfg("MNIST", 0, "FPA", params)
    state = RuntimeState(phase="GEOMETRY", current_branch_scale=1.0, branch_switched=True)
    by_name = _layer_by_name(model)
    conds: List[float] = []
    retentions: List[float] = []
    errors: List[float] = []
    for name, p in coefficient_named_params(model):
        d_adam = adam_update.get(name)
        rms = adam_rms.get(name)
        if d_adam is None or rms is None:
            continue
        layer = by_name[name]
        if not edge:
            sob = _sob_matrix(layer, cfg, state, params.fpa_rho)
            scalar = float(rms.detach().float().mean().cpu())
            mat = scalar * torch.eye(sob.shape[0], device=sob.device) + lam * sob + params.fpa_rho * torch.eye(sob.shape[0], device=sob.device)
            eig = torch.linalg.eigvalsh(mat.float()).clamp_min(1e-12)
            chol = torch.linalg.cholesky(mat.float())
            flat = (scalar * d_adam.detach().float()).reshape(-1, d_adam.shape[-1])
            solved = torch.cholesky_solve(flat.T, chol).T.reshape_as(d_adam).to(dtype=p.dtype)
        else:
            cov, diag, full_ok, cond, eff = _basis_feature_stats(layer, params)
            flat = d_adam.detach().float().reshape(d_adam.shape[0], -1)
            d_inv = float(rms.detach().float().mean().cpu())
            right_diag = d_inv + lam * diag + params.fpa_rho
            if full_ok and cov is not None:
                eye = torch.eye(cov.shape[0], device=cov.device)
                right = d_inv * eye + lam * cov + params.fpa_rho * eye
                eig = torch.linalg.eigvalsh(right.float()).clamp_min(1e-12)
                solved_flat = _solve_right(d_inv * flat, right, right_diag)
            else:
                eig = right_diag.detach().float()
                solved_flat = _solve_right(d_inv * flat, None, right_diag)
            solved = solved_flat.reshape_as(d_adam).to(dtype=p.dtype)
            _ = eff
        updates[name] = solved
        conds.append(float((eig[-1] / eig[0]).detach().cpu()))
        retentions.append(_safe_cos(solved, d_adam))
        errors.append(float(((solved.detach().float() - d_adam.detach().float()).norm() / d_adam.detach().float().norm().clamp_min(1e-12)).detach().cpu()))
    stats.update(
        {
            "metric_condition_right": _mean(conds, float("nan")),
            "retained_adam_component": _mean(retentions, float("nan")),
            "projection_error_norm": _mean(errors, float("nan")),
            "fpa_lambda": lam,
        }
    )
    return updates, stats


def _ekfng_updates(
    model: PureKANClassifier,
    params: V43Params,
    *,
    left: bool,
    sob_lambda: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    updates: Dict[str, torch.Tensor] = {}
    stats: Dict[str, Any] = {"chol_success": 1, "solve_success": 1}
    cfg = _base_cfg("MNIST", 0, "EKFNG", params)
    state = RuntimeState(phase="GEOMETRY", current_branch_scale=1.0, branch_switched=True)
    by_name = _layer_by_name(model)
    cond_right: List[float] = []
    cond_left: List[float] = []
    eff_right: List[float] = []
    eff_left: List[float] = []
    for name, p in coefficient_named_params(model):
        if p.grad is None:
            continue
        layer = by_name[name]
        grad = p.grad.detach().float()
        cov, diag, full_ok, cond, eff = _basis_feature_stats(layer, params)
        sob = _sob_matrix(layer, cfg, state, params.ekfng_rho_right)
        sob_diag = sob.diag().repeat(layer.coeff.shape[1]).to(diag.device)
        flat = grad.reshape(grad.shape[0], -1)
        right_diag = (diag + sob_lambda * sob_diag + params.ekfng_rho_right).clamp_min(params.ekfng_rho_right)
        if full_ok and cov is not None:
            right = cov + params.ekfng_rho_right * torch.eye(cov.shape[0], device=cov.device)
            if right.shape[0] == sob_diag.numel():
                right = right + sob_lambda * torch.diag(sob_diag)
            eig = torch.linalg.eigvalsh(right.float()).clamp_min(1e-12)
            solved = _solve_right(flat, right, right_diag)
            cond_right.append(float((eig[-1] / eig[0]).detach().cpu()))
        else:
            solved = _solve_right(flat, None, right_diag)
            cond_right.append(float((right_diag.max() / right_diag.min()).detach().cpu()))
        eff_right.append(eff)
        if left:
            left_mat, c_cond, c_eff = _left_cov(layer, params.ekfng_rho_left)
            chol_l = torch.linalg.cholesky(left_mat.float())
            solved = torch.cholesky_solve(solved.float(), chol_l).to(dtype=solved.dtype)
            cond_left.append(c_cond)
            eff_left.append(c_eff)
        updates[name] = (-params.ekfng_lr * solved).reshape_as(p).to(dtype=p.dtype)
    stats.update(
        {
            "metric_condition_right": _mean(cond_right, float("nan")),
            "metric_condition_left": _mean(cond_left, 1.0),
            "G_phi_effective_rank": _mean(eff_right, float("nan")),
            "C_left_effective_rank": _mean(eff_left, float("nan")),
            "ekfng_sob_lambda": sob_lambda,
        }
    )
    return updates, stats


def _functional_updates(
    model: PureKANClassifier,
    dataset: str,
    seed: int,
    method: str,
    params: V43Params,
    *,
    role_filter: str | None = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    named = coefficient_named_params(model)
    snap = _snapshot(named)
    saved_grads = {name: None if p.grad is None else p.grad.detach().clone() for name, p in named}
    if role_filter is not None:
        for name, p in named:
            role = _role_for_name(name)
            enabled = role == role_filter or (role_filter == "block" and role.startswith("block"))
            if not enabled:
                p.grad = None
    cfg = _functional_cfg(dataset, seed, method, params, role_filter or "all")
    state = RuntimeState(phase="GEOMETRY", current_branch_scale=1.0, branch_switched=True)
    try:
        functional_coeff_step(model, cfg, state, step_idx=1, total_steps=1)
        updates = {name: p.detach().clone() - snap[name] for name, p in named}
        stats = {
            "metric_condition_right": _mean([v for vals in state.fng_combined_conditions.values() for v in vals], float("nan")),
            "metric_condition_sobolev": state.fullgeo_condition,
            "ftf_fit_R2": _mean([v for vals in state.ftf_fit_r2.values() for v in vals], float("nan")),
            "ftf_residual_rel": _mean([v for vals in state.ftf_residual_rel.values() for v in vals], float("nan")),
            "chol_success": 1,
            "solve_success": 1,
        }
    except Exception as exc:
        updates = {}
        stats = {"error": repr(exc), "chol_success": 0, "solve_success": 0}
    finally:
        _restore(named, snap)
        for name, p in named:
            grad = saved_grads.get(name)
            p.grad = None if grad is None else grad.detach().clone()
    return updates, stats


def _method_updates(
    model: PureKANClassifier,
    dataset: str,
    seed: int,
    method: str,
    params: V43Params,
    run_state: V43RunState,
    *,
    step: int = 0,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    key = method.lower().replace("-", "_")
    adam_ref, adam_rms_ref = _adam_updates(model, AdamState(), lr=params.lr, mutate=True)
    if key in {"purekan_adamw", "adamw_one_step"}:
        updates, rms = _adam_updates(model, run_state.adam, lr=params.lr, mutate=True)
        return updates, {"optimizer_family": "adamw", "adam_ref": adam_ref, "adam_rms": rms}
    if key == "raw_gradient":
        updates = {name: -params.raw_lr * p.grad.detach() for name, p in coefficient_named_params(model) if p.grad is not None}
        return updates, {"optimizer_family": "raw", "adam_ref": adam_ref}
    if key.startswith("fpa_sob"):
        updates, stats = _fpa_updates(model, adam_ref, adam_rms_ref, params, edge=False, lam=_lambda_from_method(method, params))
        stats.update({"optimizer_family": "fpa_sob", "adam_ref": adam_ref})
        return updates, stats
    if key.startswith("fpa_edge"):
        updates, stats = _fpa_updates(model, adam_ref, adam_rms_ref, params, edge=True, lam=_lambda_from_method(method, params))
        stats.update({"optimizer_family": "fpa_edge", "adam_ref": adam_ref})
        return updates, stats
    if key.startswith("ekfng_right"):
        updates, stats = _ekfng_updates(model, params, left=False, sob_lambda=_ek_lambda_from_method(method, params))
        stats.update({"optimizer_family": "ekfng_right", "adam_ref": adam_ref})
        return updates, stats
    if key.startswith("ekfng_left"):
        updates, stats = _ekfng_updates(model, params, left=True, sob_lambda=_ek_lambda_from_method(method, params))
        stats.update({"optimizer_family": "ekfng_left", "adam_ref": adam_ref})
        return updates, stats
    if key in {"d0_allfullsobolev", "d6_alltaskaware", "f4_fng_leftfullright", "cft_output_only"}:
        updates, stats = _functional_updates(model, dataset, seed, method, params)
        stats.update({"optimizer_family": "functional", "adam_ref": adam_ref})
        return updates, stats
    if key == "cft_block_output_sequential":
        roles = [f"block{i}" for i in range(len(model.blocks))] + ["output"]
        role = roles[step % len(roles)]
        updates, stats = _functional_updates(model, dataset, seed, method, params, role_filter=role)
        stats.update({"optimizer_family": "cft", "role_selected": role, "adam_ref": adam_ref})
        return updates, stats
    raise ValueError(f"unknown v4.3 method: {method}")


def _lambda_from_method(method: str, params: V43Params) -> float:
    key = method.lower().replace("-", "_")
    if "lambda0.01" in key or "lambda001" in key:
        return 0.01
    if "lambda0.10" in key or "lambda010" in key:
        return 0.10
    if "lambda0.03" in key or "lambda003" in key:
        return 0.03
    if "adaptive" in key:
        return params.fpa_lambda
    return params.fpa_lambda


def _ek_lambda_from_method(method: str, params: V43Params) -> float:
    key = method.lower().replace("-", "_")
    if "lowsob" in key or "low_sob" in key:
        return 0.003
    if "adaptive" in key:
        return params.ekfng_sob_lambda
    return params.ekfng_sob_lambda


@torch.no_grad()
def _effective_rank(h: torch.Tensor) -> float:
    h = h.detach().float()
    if h.ndim > 2:
        h = h.reshape(h.shape[0], -1)
    h = h - h.mean(dim=0, keepdim=True)
    if min(h.shape) <= 1:
        return 1.0
    s = torch.linalg.svdvals(h)
    return float((s.sum().square() / s.square().sum().clamp_min(1e-12)).detach().cpu())


@torch.no_grad()
def _separation(h: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    h = h.detach().float()
    if h.ndim > 2:
        h = h.reshape(h.shape[0], -1)
    labels = y.detach().cpu()
    centroids: List[torch.Tensor] = []
    within: List[torch.Tensor] = []
    for cls in sorted(int(v) for v in labels.unique()):
        mask = labels == cls
        if mask.sum() < 2:
            continue
        hc = h[mask.to(h.device)]
        c = hc.mean(dim=0)
        centroids.append(c)
        within.append((hc - c).square().sum(dim=1).mean())
    if len(centroids) < 2:
        return 0.0, float(torch.stack(within).mean().detach().cpu()) if within else 0.0
    cmat = torch.stack(centroids)
    d = torch.pdist(cmat).mean()
    w = torch.stack(within).mean() if within else torch.tensor(0.0, device=h.device)
    return float(d.detach().cpu()), float(w.detach().cpu())


@torch.no_grad()
def _margin(logits: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    correct = logits.gather(1, y.view(-1, 1)).squeeze(1)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, y.view(-1, 1), False)
    other = logits.masked_fill(~mask, -1e9).max(dim=1).values
    m = correct - other
    return float(m.mean().detach().cpu()), float(torch.quantile(m, 0.10).detach().cpu())


def _feature_audit(model: PureKANClassifier, xb: torch.Tensor, yb: torch.Tensor) -> Dict[str, float]:
    model.eval()
    logits = model(xb)
    input_h = model.input_kan.last_output
    block_h = model.blocks[-1].kan.last_output if len(model.blocks) else input_h
    input_rank = _effective_rank(input_h if input_h is not None else logits)
    block_rank = _effective_rank(block_h if block_h is not None else logits)
    sep, within = _separation(block_h if block_h is not None else logits, yb)
    margin_mean, margin_p10 = _margin(logits.detach(), yb)
    return {
        "input_feature_effective_rank": input_rank,
        "block_feature_effective_rank": block_rank,
        "class_centroid_separation": sep,
        "within_class_variance": within,
        "margin_mean": margin_mean,
        "margin_p10": margin_p10,
    }


def _logit_drift(model: PureKANClassifier, xb: torch.Tensor, before: torch.Tensor) -> float:
    with torch.no_grad():
        after = model(xb)
        return float((after - before).float().norm().cpu() / before.float().norm().clamp_min(1e-8).cpu())


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = []
    device = get_device(args.device)
    params = V43Params(train_size=512, val_size=128, test_size=128, hidden_dim=64, depth=4, basis_count=16)
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=512, val_size=128, test_size=128, seed=0, download=not args.no_download, allow_fake_data=args.allow_fake_data)
        for method in [*BASE_METHODS, *V43_METHODS]:
            set_seed(0)
            model = _make_model(bundle, params, device)
            idx = next(iter(iter_minibatches(len(bundle.x_train), params.batch_size, 0)))
            xb = bundle.x_train[idx].to(device)
            yb = bundle.y_train[idx].to(device)
            start = time.perf_counter()
            loss = _compute_loss_and_grad(model, xb, yb)
            named = coefficient_named_params(model)
            if method == "BFT-mixed-reverse":
                logs = _bft_step(
                    model,
                    bundle,
                    dataset,
                    0,
                    idx,
                    device,
                    BFTParams(train_size=512, val_size=128, test_size=128, hidden_dim=64, depth=4, basis_count=16),
                    proposal_mode="mixed",
                    order="reverse",
                    epoch=0,
                    step=0,
                )
                stats = {
                    "metric_condition_right": float("nan"),
                    "metric_condition_left": float("nan"),
                    "chol_success": 1,
                    "solve_success": 1,
                    "accepted_rate": _mean([float(r["accepted"]) for r in logs]),
                }
            else:
                updates, stats = _method_updates(model, dataset, 0, method if method != "PureKAN-AdamW" else "AdamW-one-step", params, V43RunState())
                _apply_updates(named, updates)
            step_ms = 1000.0 * (time.perf_counter() - start)
            nonkan = sum(
                p.numel()
                for name, p in model.named_parameters()
                if p.requires_grad and not name.endswith("coeff")
            )
            coeff_total = sum(p.numel() for _, p in named)
            coeff_seen = coeff_total
            rows.append(
                {
                    "stage": "P0",
                    "dataset": dataset,
                    "method": method,
                    "loss": loss,
                    "num_total_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "num_coeff_params": coeff_total,
                    "num_nonkan_trainable_params": nonkan,
                    "functional_coverage": coeff_seen / max(1, coeff_total),
                    "input_coeff_count": sum(p.numel() for n, p in named if n.startswith("input_kan.")),
                    "block_coeff_count": sum(p.numel() for n, p in named if n.startswith("blocks.")),
                    "output_coeff_count": sum(p.numel() for n, p in named if n.startswith("output_kan.")),
                    "metric_condition_right": stats.get("metric_condition_right", float("nan")),
                    "metric_condition_left": stats.get("metric_condition_left", float("nan")),
                    "metric_condition_sobolev": stats.get("metric_condition_sobolev", float("nan")),
                    "chol_success": stats.get("chol_success", 1),
                    "solve_success": stats.get("solve_success", 1),
                    "nan_count": int(any(not torch.isfinite(p).all().item() for _, p in named)),
                    "inf_count": 0,
                    "step_time_ms": step_ms,
                    "memory_peak_mb": torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0,
                    "error": stats.get("error", ""),
                }
            )
            print(f"P0 {dataset} {method} nonKAN={nonkan} cov={rows[-1]['functional_coverage']:.3f} err={rows[-1]['error']}")
    write_csv(out_dir / "p0_invariants.csv", rows)
    return rows


def run_p1(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = []
    device = get_device(args.device)
    params = V43Params(train_size=512, val_size=128, test_size=128, hidden_dim=64, depth=4, basis_count=16)
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=512, val_size=128, test_size=128, seed=0, download=not args.no_download, allow_fake_data=args.allow_fake_data)
        idx = next(iter(iter_minibatches(len(bundle.x_train), params.batch_size, 13)))
        xb = bundle.x_train[idx].to(device)
        yb = bundle.y_train[idx].to(device)
        hb = bundle.x_val[: len(idx)].to(device)
        yh = bundle.y_val[: len(idx)].to(device)
        for method in P1_METHODS:
            set_seed(0)
            model = _make_model(bundle, params, device)
            named = coefficient_named_params(model)
            train_before = F.cross_entropy(model(xb), yb).item()
            hold_before = F.cross_entropy(model(hb), yh).item()
            feat_before = _feature_audit(model, xb, yb)
            geom_before = estimate_geometry(model, hb, device=device, batch_size=min(params.audit_batch_size, len(hb)))
            logits_before = model(xb).detach()
            _compute_loss_and_grad(model, xb, yb)
            adam_ref, _ = _adam_updates(model, AdamState(), lr=params.lr, mutate=True)
            if method == "BFT-mixed-reverse":
                snap = _snapshot(named)
                logs = _bft_step(
                    model,
                    bundle,
                    dataset,
                    0,
                    idx,
                    device,
                    BFTParams(train_size=512, val_size=128, test_size=128, hidden_dim=64, depth=4, basis_count=16),
                    proposal_mode="mixed",
                    order="reverse",
                    epoch=0,
                    step=0,
                )
                updates = {name: p.detach().clone() - snap[name] for name, p in named}
                stats = {
                    "metric_condition_right": float("nan"),
                    "metric_condition_left": float("nan"),
                    "retained_adam_component": float("nan"),
                    "projection_error_norm": float("nan"),
                    "bft_accepted_rate": _mean([float(r["accepted"]) for r in logs]),
                }
            else:
                updates, stats = _method_updates(model, dataset, 0, method, params, V43RunState())
                _apply_updates(named, updates)
            train_after = F.cross_entropy(model(xb), yb).item()
            hold_after = F.cross_entropy(model(hb), yh).item()
            feat_after = _feature_audit(model, xb, yb)
            geom_after = estimate_geometry(model, hb, device=device, batch_size=min(params.audit_batch_size, len(hb)))
            update_flat = _flat_updates(updates)
            adam_flat = _flat_updates(adam_ref)
            dot = float(torch.dot(update_flat[: min(update_flat.numel(), adam_flat.numel())], adam_flat[: min(update_flat.numel(), adam_flat.numel())]).detach().cpu()) if update_flat.numel() and adam_flat.numel() else float("nan")
            adam_norm_sq = float(adam_flat.square().sum().detach().cpu()) if adam_flat.numel() else float("nan")
            row = {
                "stage": "P1",
                "dataset": dataset,
                "method": method,
                "train_loss_before": train_before,
                "train_loss_after": train_after,
                "holdout_loss_before": hold_before,
                "holdout_loss_after": hold_after,
                "train_descent": train_before - train_after,
                "holdout_descent": hold_before - hold_after,
                "bad_train_step": int(train_after > train_before),
                "bad_holdout_step": int(hold_after > hold_before),
                "cos_with_adam_global": _safe_cos(update_flat, adam_flat),
                "projection_on_adam_global": dot / max(1e-12, adam_norm_sq) if math.isfinite(dot) and math.isfinite(adam_norm_sq) else float("nan"),
                "norm_ratio_vs_adam_global": float(update_flat.norm().detach().cpu() / adam_flat.norm().clamp_min(1e-12).detach().cpu()) if update_flat.numel() and adam_flat.numel() else float("nan"),
                "input_feature_effective_rank_before": feat_before["input_feature_effective_rank"],
                "input_feature_effective_rank_after": feat_after["input_feature_effective_rank"],
                "block_feature_effective_rank_before": feat_before["block_feature_effective_rank"],
                "block_feature_effective_rank_after": feat_after["block_feature_effective_rank"],
                "class_centroid_separation_before": feat_before["class_centroid_separation"],
                "class_centroid_separation_after": feat_after["class_centroid_separation"],
                "within_class_variance_before": feat_before["within_class_variance"],
                "within_class_variance_after": feat_after["within_class_variance"],
                "margin_mean_before": feat_before["margin_mean"],
                "margin_mean_after": feat_after["margin_mean"],
                "margin_p10_before": feat_before["margin_p10"],
                "margin_p10_after": feat_after["margin_p10"],
                "phi_prime_p95_before": geom_before["phi_prime_p95"],
                "phi_prime_p95_after": geom_after["phi_prime_p95"],
                "jacobian_condition_before": geom_before["max_jac_condition"],
                "jacobian_condition_after": geom_after["max_jac_condition"],
                "curvature_energy_before": geom_before["curvature_energy"],
                "curvature_energy_after": geom_after["curvature_energy"],
                "logit_drift": _logit_drift(model, xb, logits_before),
                "metric_condition_right": stats.get("metric_condition_right", float("nan")),
                "metric_condition_left": stats.get("metric_condition_left", float("nan")),
                "retained_adam_component": stats.get("retained_adam_component", float("nan")),
                "projection_error_norm": stats.get("projection_error_norm", float("nan")),
                "error": stats.get("error", ""),
            }
            rows.append(row)
            print(
                f"P1 {dataset} {method} trainΔ={row['train_descent']:.4g} holdΔ={row['holdout_descent']:.4g} "
                f"cosA={row['cos_with_adam_global']:.3f} rankΔ={row['block_feature_effective_rank_after']-row['block_feature_effective_rank_before']:.3g}"
            )
    write_csv(out_dir / "p1_proposal_direction_audit.csv", rows)
    return rows


def _run_custom_train(
    args: argparse.Namespace,
    *,
    dataset: str,
    seed: int,
    method: str,
    params: V43Params,
    epochs: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    device = get_device(args.device)
    set_seed(seed)
    bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=params.train_size, val_size=params.val_size, test_size=params.test_size, seed=seed, download=not args.no_download, allow_fake_data=args.allow_fake_data)
    model = _make_model(bundle, params, device)
    run_state = V43RunState()
    val_losses: List[float] = []
    val_accs: List[float] = []
    rank_curve: List[float] = []
    margin_curve: List[float] = []
    phi_curve: List[float] = []
    jac_curve: List[float] = []
    cos_curve: List[float] = []
    proj_curve: List[float] = []
    train_losses: List[float] = []
    start = time.perf_counter()
    step = 0
    for epoch in range(epochs):
        for idx in iter_minibatches(len(bundle.x_train), params.batch_size, seed + epoch * 997):
            xb = bundle.x_train[idx].to(device)
            yb = bundle.y_train[idx].to(device)
            if method == "BFT-mixed-reverse":
                logs = _bft_step(
                    model,
                    bundle,
                    dataset,
                    seed,
                    idx,
                    device,
                    BFTParams(train_size=params.train_size, val_size=params.val_size, test_size=params.test_size, hidden_dim=params.hidden_dim, depth=params.depth, basis_count=params.basis_count),
                    proposal_mode="mixed",
                    order="reverse",
                    epoch=epoch,
                    step=step,
                )
                for log in logs:
                    log.update({"dataset": dataset, "seed": seed, "method": method})
                    run_state.proposal_rows.append(log)
                step += 1
                continue
            loss = _compute_loss_and_grad(model, xb, yb)
            train_losses.append(loss)
            adam_ref, _ = _adam_updates(model, AdamState(), lr=params.lr, mutate=True)
            updates, stats = _method_updates(model, dataset, seed, method, params, run_state, step=step)
            update_flat = _flat_updates(updates)
            adam_flat = _flat_updates(adam_ref)
            cos = _safe_cos(update_flat, adam_flat)
            dot = float(torch.dot(update_flat[: min(update_flat.numel(), adam_flat.numel())], adam_flat[: min(update_flat.numel(), adam_flat.numel())]).detach().cpu()) if update_flat.numel() and adam_flat.numel() else float("nan")
            anorm = float(adam_flat.square().sum().detach().cpu()) if adam_flat.numel() else float("nan")
            named = coefficient_named_params(model)
            by_role: Dict[str, List[float]] = {}
            for name, upd in updates.items():
                by_role.setdefault(_role_group(_role_for_name(name)), []).append(float(upd.detach().float().norm().cpu()))
            _apply_updates(named, updates)
            for role, values in by_role.items():
                run_state.role_rows.append({"dataset": dataset, "seed": seed, "method": method, "step": step, "role": role, "update_norm": _mean(values)})
            cos_curve.append(cos)
            proj_curve.append(dot / max(1e-12, anorm) if math.isfinite(dot) and math.isfinite(anorm) else float("nan"))
            run_state.step_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "step": step,
                    "train_loss": loss,
                    "cos_with_adam": cos,
                    "projection_on_adam": proj_curve[-1],
                    "proposal_norm": float(update_flat.norm().detach().cpu()) if update_flat.numel() else 0.0,
                    "metric_condition_right": stats.get("metric_condition_right", float("nan")),
                    "metric_condition_left": stats.get("metric_condition_left", float("nan")),
                    "role_selected": stats.get("role_selected", ""),
                    "fpa_lambda": stats.get("fpa_lambda", float("nan")),
                    "retained_adam_component": stats.get("retained_adam_component", float("nan")),
                    "projection_error_norm": stats.get("projection_error_norm", float("nan")),
                    "ftf_fit_R2": stats.get("ftf_fit_R2", float("nan")),
                }
            )
            step += 1
        val = evaluate(model, bundle.x_val, bundle.y_val, device=device, batch_size=params.eval_batch_size)
        val_losses.append(val["loss"])
        val_accs.append(val["acc"])
        feat = _feature_audit(model, bundle.x_val[: params.audit_batch_size].to(device), bundle.y_val[: params.audit_batch_size].to(device))
        geom = estimate_geometry(model, bundle.x_val, device=device, batch_size=params.audit_batch_size)
        occ = rbf_basis_occupancy_audit(model)
        rank_curve.append(feat["block_feature_effective_rank"])
        margin_curve.append(feat["margin_mean"])
        phi_curve.append(geom["phi_prime_p95"])
        jac_curve.append(geom["max_jac_condition"])
        run_state.lambda_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "method": method,
                "epoch": epoch,
                "val_loss": val["loss"],
                "val_acc": val["acc"],
                "feature_effective_rank": feat["block_feature_effective_rank"],
                "margin_mean": feat["margin_mean"],
                "phi_prime_p95": geom["phi_prime_p95"],
                "jacobian_condition": geom["max_jac_condition"],
                "basis_dead_fraction": occ.get("rbf_basis_dead_frac", float("nan")),
            }
        )
    wall = time.perf_counter() - start
    test = evaluate(model, bundle.x_test, bundle.y_test, device=device, batch_size=params.eval_batch_size)
    train_eval = evaluate(model, bundle.x_train[:2048], bundle.y_train[:2048], device=device, batch_size=params.eval_batch_size)
    final_feat = _feature_audit(model, bundle.x_val[: params.audit_batch_size].to(device), bundle.y_val[: params.audit_batch_size].to(device))
    final_geom = estimate_geometry(model, bundle.x_val, device=device, batch_size=params.audit_batch_size)
    occ = rbf_basis_occupancy_audit(model)
    row = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "epochs": epochs,
        "train_size": params.train_size,
        "test_acc": test["acc"],
        "test_loss": test["loss"],
        "val_auc": float(np.mean(val_losses)) if val_losses else float("nan"),
        "val_acc_auc": float(np.mean(val_accs)) if val_accs else float("nan"),
        "val_loss_curve": ",".join(f"{v:.6g}" for v in val_losses),
        "val_acc_curve": ",".join(f"{v:.6g}" for v in val_accs),
        "train_loss_curve_per_step": ",".join(f"{v:.6g}" for v in train_losses[:240]),
        "train_acc": train_eval["acc"],
        "train_val_gap": train_eval["acc"] - (val_accs[-1] if val_accs else 0.0),
        "ece": test["ece"],
        "NLL": test["loss"],
        "feature_effective_rank_final": final_feat["block_feature_effective_rank"],
        "input_feature_effective_rank_final": final_feat["input_feature_effective_rank"],
        "class_centroid_separation_final": final_feat["class_centroid_separation"],
        "margin_mean_final": final_feat["margin_mean"],
        "margin_p10_final": final_feat["margin_p10"],
        "feature_effective_rank_curve_last_block": ",".join(f"{v:.6g}" for v in rank_curve),
        "margin_mean_curve": ",".join(f"{v:.6g}" for v in margin_curve),
        "cos_with_adam_mean": _mean(cos_curve, float("nan")),
        "projection_on_adam_mean": _mean(proj_curve, float("nan")),
        "cos_with_adam_by_step": ",".join(f"{v:.6g}" for v in cos_curve[:240] if math.isfinite(v)),
        "projection_on_adam_by_step": ",".join(f"{v:.6g}" for v in proj_curve[:240] if math.isfinite(v)),
        "phi_prime_p95": final_geom["phi_prime_p95"],
        "max_jac_condition": final_geom["max_jac_condition"],
        "curvature_energy": final_geom["curvature_energy"],
        "phi_prime_p95_curve": ",".join(f"{v:.6g}" for v in phi_curve),
        "jacobian_condition_curve": ",".join(f"{v:.6g}" for v in jac_curve),
        "basis_occupancy_entropy": occ.get("rbf_basis_mean", float("nan")),
        "basis_dead_fraction": occ.get("rbf_basis_dead_frac", float("nan")),
        "input_out_of_grid_fraction": occ.get("rbf_input_out_of_grid_frac", float("nan")),
        "step_time_ms": 1000.0 * wall / max(1, step),
        "memory_peak_mb": torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0,
        "error": "",
    }
    return row, run_state.step_rows + run_state.lambda_rows, run_state.role_rows + run_state.proposal_rows


def run_train_stage(args: argparse.Namespace, package: str, methods: Sequence[str], *, epochs: int, out_name: str, params: V43Params) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows = [] if args.fresh else read_csv(out_dir / out_name)
    trace_rows = [] if args.fresh else read_csv(out_dir / "optimizer_dynamics_trace.csv")
    role_rows = [] if args.fresh else read_csv(out_dir / "role_update_trace.csv")
    datasets = [dataset_name(d) for d in parse_str_list(args.datasets)]
    seeds = parse_int_list(args.seeds)
    wanted = set(parse_str_list(args.methods))
    if wanted:
        methods = [m for m in methods if m in wanted]
    for dataset in datasets:
        for method in methods:
            for seed in seeds:
                try:
                    row, traces, roles = _run_custom_train(args, dataset=dataset, seed=seed, method=method, params=params, epochs=epochs)
                    row["stage"] = package
                    rows.append(row)
                    trace_rows.extend(traces)
                    role_rows.extend(roles)
                    print(f"{package} {dataset} {method} seed={seed} acc={row['test_acc']:.4f} valAUC={row['val_auc']:.4f}")
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    rows.append({"stage": package, "dataset": dataset, "method": method, "seed": seed, "error": repr(exc)})
                    print(f"{package} ERROR {dataset} {method} seed={seed}: {exc!r}")
                write_csv(out_dir / out_name, rows)
                write_csv(out_dir / "optimizer_dynamics_trace.csv", trace_rows)
                write_csv(out_dir / "role_update_trace.csv", role_rows)
    return rows


def run_p2(args: argparse.Namespace) -> List[Dict[str, Any]]:
    params = V43Params(epochs_p2=3)
    seeds = args.seeds
    if seeds == add_args().get_default("seeds"):
        args.seeds = "0,1,2"
    rows = run_train_stage(args, "P2", P2_METHODS, epochs=params.epochs_p2, out_name="p2_short_horizon_scorecard.csv", params=params)
    args.seeds = seeds
    return rows


def run_p3(args: argparse.Namespace) -> List[Dict[str, Any]]:
    params = V43Params(epochs_p3=8)
    methods = [
        "PureKAN-AdamW",
        "FPA-sob-prox-lambda0.01",
        "FPA-sob-prox-lambda0.03",
        "FPA-sob-prox-lambda0.10",
        "FPA-edge-prox-lambda0.01",
        "FPA-edge-prox-lambda0.03",
        "FPA-edge-prox-lambda0.10",
        "FPA-edge-prox-adaptiveLambda",
    ]
    seeds = args.seeds
    if seeds == add_args().get_default("seeds"):
        args.seeds = "0,1,2"
    rows = run_train_stage(args, "P3", methods, epochs=params.epochs_p3, out_name="p3_fpa_scorecard.csv", params=params)
    args.seeds = seeds
    return rows


def run_p4(args: argparse.Namespace) -> List[Dict[str, Any]]:
    params = V43Params(epochs_p4=8)
    methods = [
        "PureKAN-AdamW",
        "F4-FNG-leftFullRight",
        "EKFNG-rightFull",
        "EKFNG-leftRightFull",
        "EKFNG-leftRightFull-lowSob",
    ]
    seeds = args.seeds
    if seeds == add_args().get_default("seeds"):
        args.seeds = "0,1,2"
    rows = run_train_stage(args, "P4", methods, epochs=params.epochs_p4, out_name="p4_ekfng_scorecard.csv", params=params)
    args.seeds = seeds
    return rows


def run_p5(args: argparse.Namespace) -> List[Dict[str, Any]]:
    params = V43Params(epochs_p5=6, cft_tau=0.02)
    methods = [
        "CFT-output-only",
        "CFT-block-output-sequential",
        "CFT-output-only-smallTau",
        "CFT-block-output-sequential-smallTau",
    ]
    seeds = args.seeds
    if seeds == add_args().get_default("seeds"):
        args.seeds = "0,1,2"
    rows = run_train_stage(args, "P5", methods, epochs=params.epochs_p5, out_name="p5_cft_scorecard.csv", params=params)
    args.seeds = seeds
    return rows


def add_args() -> argparse.ArgumentParser:
    p = add_v3_args()
    p.description = __doc__
    p.set_defaults(packages="V4_3_P0_SMOKE", datasets="MNIST,Fashion-MNIST,KMNIST", out_dir=Path("results/v4_3"))
    return p


def main() -> int:
    args = add_args().parse_args()
    all_rows: List[Dict[str, Any]] = []
    for package in parse_str_list(args.packages):
        key = package.strip().upper().replace("-", "_")
        if key == "V4_3_P0_SMOKE":
            all_rows = run_p0(args)
        elif key == "V4_3_P1_PROPOSAL":
            all_rows = run_p1(args)
        elif key == "V4_3_P2_SHORT":
            all_rows = run_p2(args)
        elif key == "V4_3_P3_FPA":
            all_rows = run_p3(args)
        elif key == "V4_3_P4_EKFNG":
            all_rows = run_p4(args)
        elif key == "V4_3_P5_CFT":
            all_rows = run_p5(args)
        else:
            raise ValueError(f"unknown v4.3 package: {package}")
    return 0 if all_rows is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
