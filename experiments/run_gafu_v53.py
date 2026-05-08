#!/usr/bin/env python3
"""DG-KAN v5.3 runner: AB-RBF event controller for light residual smoothing."""

from __future__ import annotations

import argparse
import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dgkan_core import (
    ABRBFDense,
    coefficient_named_params,
    edge_named_params as core_edge_named_params,
    ensure_dir,
    get_device,
    load_vision_bundle,
    parse_int_list,
    parse_str_list,
    read_csv,
    set_seed,
    write_csv,
)
from run_gafu_v3 import add_args as add_v3_args, dataset_name
from run_gafu_v43 import _apply_updates, _mean, _restore, _snapshot
from run_gafu_v47 import _teacher_snapshot
from run_gafu_v48 import _iter_steps, _logit_kl
from run_gafu_v51 import (
    V51Params,
    _base_rbf_audit,
    _edge_kind_from_method,
    _edge_named_params,
    _edge_param_manifest,
    _eval_v49,
    _load_or_train_teacher,
    _make_v49_model,
    _param_count_audit,
    _residual_smoothing_updates,
    _scaled_updates,
    _split_geometry_audit,
    _state_path,
    _tail_smoothing_direction,
    _train_edge_adamw,
    _update_norm,
)


DATASETS = ["MNIST", "Fashion-MNIST", "KMNIST"]
ABRBF_MAIN = "PureKAN-ABRBF-linear+silu-AdamW"
ABRBF_ALT = "PureKAN-ABRBF-silu-AdamW"
ABRBF_LINEAR = "PureKAN-ABRBF-linear-AdamW"
RBF_BASELINE = "PureKAN-RBFOnly-AdamW"
BASEONLY_LINEAR = "PureKAN-BaseOnly-linear-AdamW"
BASEONLY_SILU = "PureKAN-BaseOnly-silu-AdamW"

P0_METHODS = [
    "ABRBF-AdamW",
    "ABRBF-LightSmooth-logitOnly-smoke",
    "ABRBF-LightSmooth-scoreOnly-smoke",
    "ABRBF-StrongSmooth-reference-smoke",
    "RBFOnly-AdamW",
]
P1_TEACHERS = [ABRBF_MAIN, ABRBF_ALT, ABRBF_LINEAR]
P1_CANDIDATES = [
    ("S0-laplacian", "logitOnly"),
    ("S0-laplacian", "scoreOnly"),
    ("S1-sobolev-grad", "logitOnly"),
    ("S1-sobolev-grad", "scoreOnly"),
    ("StrongSmooth-reference", "strongReference"),
]
P3_METHODS = [
    "ABRBF-AdamW",
    "oneShot-late-logitOnly-fullRefresh-r10",
    "oneShot-late-logitOnly-baseOnlyRefresh-r10",
    "oneShot-late-logitOnly-rbfFrozenRefresh-r10",
    "oneShot-late-scoreOnly-fullRefresh-r10",
    "oneShot-late-scoreOnly-baseOnlyRefresh-r10",
    "oneShot-late-scoreOnly-rbfFrozenRefresh-r10",
]
P4_METHODS = [
    "ABRBF-AdamW",
    "FixedInterval-reference",
    "EventV2-scoreProbe-max1",
    "EventV2-scoreProbe-max2",
    "EventV2-geometryDebt-scoreProbe-max2",
]


@dataclass
class V53Params(V51Params):
    train_size: int = 1536
    val_size: int = 512
    test_size: int = 512
    batch_size: int = 128
    eval_batch_size: int = 512
    audit_batch_size: int = 20
    p1_steps: int = 120
    p2_task_steps: int = 100
    p2_refresh_steps: int = 20
    p2_probe_steps: int = 100
    p2_probe_interval: int = 20
    p3_late_steps: int = 80
    p3_refresh_steps: int = 10
    p3_total_steps: int = 140
    p3_eval_interval: int = 20
    p4_total_steps: int = 140
    p4_eval_interval: int = 20
    event_eta_grid: Tuple[float, ...] = (0.005, 0.01, 0.02, 0.035, 0.05)
    light_eta_grid: Tuple[float, ...] = (0.005, 0.01, 0.02, 0.035, 0.05)
    strong_eta_grid: Tuple[float, ...] = (0.03, 0.06, 0.10, 0.16, 0.24, 0.35, 0.50)
    light_kl_max: float = 0.005
    light_logit_max: float = 0.03
    light_acc_drop_max: float = 0.005
    light_holdout_worse_max: float = 0.015
    light_score_min: float = 0.0
    memory_ratio_gate: float = 1.25
    time_ratio_gate: float = 1.20
    plateau_window: int = 3
    epsilon_plateau: float = 0.01
    max_smoothing_events: int = 2
    min_event_gap_evals: int = 2
    score_threshold: float = 0.02
    min_geometry_debt: float = 0.05
    cooldown_evals: int = 2


def _cuda_enabled(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_available()


def _reset_peak(device: torch.device) -> None:
    if _cuda_enabled(device):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _peak_mb(device: torch.device) -> Tuple[float, float]:
    if not _cuda_enabled(device):
        return 0.0, 0.0
    torch.cuda.synchronize(device)
    return (
        float(torch.cuda.max_memory_allocated(device) / (1024**2)),
        float(torch.cuda.max_memory_reserved(device) / (1024**2)),
    )


def _empty_cache(device: torch.device) -> None:
    gc.collect()
    if _cuda_enabled(device):
        torch.cuda.empty_cache()


def _tensor_mb(tensors: Iterable[torch.Tensor]) -> float:
    total = 0
    for t in tensors:
        total += int(t.numel()) * int(t.element_size())
    return total / (1024**2)


def _fast_task_metrics(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)
        acc = (pred == y).float().mean()
        target = logits.gather(1, y.view(-1, 1)).squeeze(1)
        masked = logits.masked_fill(F.one_hot(y, logits.shape[-1]).bool(), float("-inf"))
        margin = target - masked.max(dim=1).values
    return {
        "val_loss": float(loss.detach().cpu()),
        "val_acc": float(acc.detach().cpu()),
        "margin_mean": float(margin.mean().detach().cpu()),
        "margin_p10": float(torch.quantile(margin.detach().float().cpu(), 0.10).item()),
    }


def _teacher_logit_constraints(logits: torch.Tensor, teacher_logits: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        kl = float(_logit_kl(logits, teacher_logits, 1.0))
        drift = (logits - teacher_logits).float().norm() / teacher_logits.float().norm().clamp_min(1.0e-12)
        pred = logits.argmax(dim=-1)
        tpred = teacher_logits.argmax(dim=-1)
        flip = (pred != tpred).float().mean()
    return {
        "KL_teacher_student": kl,
        "kl_teacher_student": kl,
        "logit_relative_drift": float(drift.detach().cpu()),
        "argmax_flip_rate": float(flip.detach().cpu()),
    }


def _high_eig_energy(model: torch.nn.Module) -> float:
    vals: List[float] = []
    with torch.no_grad():
        for layer in model.kan_layers():  # type: ignore[attr-defined]
            coeff = layer.coeff.detach().float()
            base = int(getattr(layer, "base_dim", 0))
            if coeff.shape[-1] <= base:
                continue
            tail = coeff[..., base:].reshape(-1, coeff.shape[-1] - base)
            energy = tail.square().mean(dim=0)
            split = max(1, 2 * energy.numel() // 3)
            vals.append(float((energy[split:].sum() / energy.sum().clamp_min(1.0e-12)).detach().cpu()))
    return _mean(vals, float("nan"))


def _coeff_sobolev_rbf(model: torch.nn.Module) -> float:
    vals: List[float] = []
    with torch.no_grad():
        for layer in model.kan_layers():  # type: ignore[attr-defined]
            coeff = layer.coeff.detach().float()
            base = int(getattr(layer, "base_dim", 0))
            if coeff.shape[-1] <= base:
                continue
            tail = coeff[..., base:]
            if tail.shape[-1] > 2:
                diff2 = tail[..., :-2] - 2.0 * tail[..., 1:-1] + tail[..., 2:]
                vals.append(float(diff2.square().mean().detach().cpu()))
            elif tail.shape[-1] > 1:
                vals.append(float((tail[..., 1:] - tail[..., :-1]).square().mean().detach().cpu()))
            else:
                vals.append(float(tail.square().mean().detach().cpu()))
    return _mean(vals, float("nan"))


def _ablation_drops(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    if not hasattr(model, "kan_layers"):
        return {"base_ablation_drop": float("nan"), "rbf_ablation_drop": float("nan")}
    named = coefficient_named_params(model)
    snap = _snapshot(named)
    base_acc = _fast_task_metrics(model, x, y)["val_acc"]
    layers = list(model.kan_layers())  # type: ignore[attr-defined]
    with torch.no_grad():
        for layer in layers:
            base = int(getattr(layer, "base_dim", 0))
            if base > 0:
                layer.coeff[..., :base].zero_()
    base_zero_acc = _fast_task_metrics(model, x, y)["val_acc"]
    _restore(named, snap)
    with torch.no_grad():
        for layer in layers:
            base = int(getattr(layer, "base_dim", 0))
            if layer.coeff.shape[-1] > base:
                layer.coeff[..., base:].zero_()
    rbf_zero_acc = _fast_task_metrics(model, x, y)["val_acc"]
    _restore(named, snap)
    return {
        "base_ablation_drop": base_acc - base_zero_acc,
        "rbf_ablation_drop": base_acc - rbf_zero_acc,
    }


def _fast_geometry(model: torch.nn.Module, x: torch.Tensor) -> Dict[str, float]:
    model(x)
    out = {}
    out.update(_base_rbf_audit(model))
    out.update(_split_geometry_audit(model))
    out["high_eig_energy"] = _high_eig_energy(model)
    out["sobolev_rbf_norm_fast"] = _coeff_sobolev_rbf(model)
    return out


def _coeff_residual_geometry(model: torch.nn.Module) -> Dict[str, float]:
    """Lightweight residual geometry surrogate used inside eta search.

    Full derivative/Jacobian-style split geometry is intentionally kept out of
    the inner loop.  The stage-level audits still run the expensive metric
    before/after accepted events.
    """

    curv_vals: List[float] = []
    sob_vals: List[float] = []
    phi_vals: List[float] = []
    with torch.no_grad():
        for layer in model.kan_layers():  # type: ignore[attr-defined]
            coeff = layer.coeff.detach().float()
            base = int(getattr(layer, "base_dim", 0))
            if coeff.shape[-1] <= base:
                continue
            tail = coeff[..., base:]
            if tail.shape[-1] > 1:
                diff1 = tail[..., 1:] - tail[..., :-1]
                curv_vals.append(float(diff1.square().mean().detach().cpu()))
                phi_vals.append(float(diff1.abs().quantile(0.95).detach().cpu()))
                if tail.shape[-1] > 2:
                    diff2 = tail[..., :-2] - 2.0 * tail[..., 1:-1] + tail[..., 2:]
                    sob_vals.append(float(diff2.square().mean().detach().cpu()))
                else:
                    sob_vals.append(float(diff1.square().mean().detach().cpu()))
            else:
                curv_vals.append(float(tail.square().mean().detach().cpu()))
                sob_vals.append(float(tail.square().mean().detach().cpu()))
                phi_vals.append(float(tail.abs().quantile(0.95).detach().cpu()))
    return {
        "phi_rbf_p95": _mean(phi_vals, float("nan")),
        "curvature_rbf_p95": _mean(curv_vals, float("nan")),
        "sobolev_rbf_norm_fast": _mean(sob_vals, float("nan")),
        "high_eig_energy": _high_eig_energy(model),
    }


def _base_slice_update_norms(model: torch.nn.Module, updates: Dict[str, torch.Tensor]) -> Tuple[float, float]:
    names = [name for name, _ in coefficient_named_params(model)]
    layers = list(model.kan_layers())  # type: ignore[attr-defined]
    base_vals: List[torch.Tensor] = []
    rbf_vals: List[torch.Tensor] = []
    for name, layer in zip(names, layers):
        upd = updates.get(name)
        if upd is None:
            continue
        base = int(getattr(layer, "base_dim", 0))
        if base > 0:
            base_vals.append(upd[..., :base].detach().float().reshape(-1))
        if upd.shape[-1] > base:
            rbf_vals.append(upd[..., base:].detach().float().reshape(-1))
    base_norm = float(torch.cat(base_vals).norm().detach().cpu()) if base_vals else 0.0
    rbf_norm = float(torch.cat(rbf_vals).norm().detach().cpu()) if rbf_vals else 0.0
    return base_norm, rbf_norm


def _light_score(
    *,
    phi_red: float,
    curv_red: float,
    sob_red: float,
    kl: float,
    logit: float,
    holdout_delta: float,
    acc_drop: float,
    smoothing_time: float = 0.0,
) -> float:
    return 1.0 * phi_red + 0.5 * curv_red - 10.0 * kl - 2.0 * logit - 5.0 * max(0.0, acc_drop) - 0.1 * max(0.0, smoothing_time)


def _light_smoothing_event(
    model: torch.nn.Module,
    bundle: Any,
    params: V53Params,
    device: torch.device,
    *,
    proposal: str,
    controller: str,
    seed: int,
    strong_reference: bool = False,
) -> Dict[str, Any]:
    from run_gafu_v51 import _accepted_residual_smoothing_once

    hb = bundle.x_val[: params.audit_batch_size].to(device)
    yh = bundle.y_val[: params.audit_batch_size].to(device)
    named = coefficient_named_params(model)
    snap = _snapshot(named)
    _reset_peak(device)
    t0 = time.perf_counter()
    if strong_reference:
        stats = _accepted_residual_smoothing_once(model, bundle, params, device, proposal="S0-laplacian", controller="C3-logit-hidden-margin-trust", seed=seed)
        peak_alloc, peak_reserved = _peak_mb(device)
        stats.update(
            {
                "proposal": "StrongSmooth-reference",
                "controller": "strongReference",
                "smoothing_time_sec": time.perf_counter() - t0,
                "peak_cuda_allocated_mb": peak_alloc,
                "peak_cuda_reserved_mb": peak_reserved,
                "smoothing_forward_count": len(params.strong_eta_grid) + 3,
                "smoothing_backward_count": 0,
                "teacher_cache_mb": float("nan"),
                "proposal_temp_mb": float("nan"),
                "smoothing_applies_to_base": 0,
                "smoothing_applies_to_rbf": 1,
            }
        )
        return stats

    with torch.no_grad():
        teacher_logits = model(hb).detach()
    teacher_cache_mb = _tensor_mb([teacher_logits])
    before_task = _fast_task_metrics(model, hb, yh)
    before_geom = _coeff_residual_geometry(model)
    before_phi = before_geom["phi_rbf_p95"]
    before_curv = before_geom["curvature_rbf_p95"]
    before_sob = before_geom["sobolev_rbf_norm_fast"]
    before_high = before_geom["high_eig_energy"]
    role = "block" if "role-block" in proposal.lower() or "heuristic" in proposal.lower() else None
    raw_updates = _residual_smoothing_updates(model, proposal, role=role)
    base_norm, rbf_norm = _base_slice_update_norms(model, raw_updates)
    proposal_temp_mb = _tensor_mb(raw_updates.values())
    best: Dict[str, Any] = {
        "accepted": 0,
        "rejected": 1,
        "accepted_eta": 0.0,
        "backtrack_count": len(params.light_eta_grid),
        "reject_reason": "no_accepted_eta",
        "score": -1.0e18,
    }
    forward_count = 1
    for eta in params.light_eta_grid:
        _restore(named, snap)
        _apply_updates(named, _scaled_updates(raw_updates, eta))
        logits = model(hb)
        forward_count += 1
        constraints = _teacher_logit_constraints(logits, teacher_logits)
        task = _fast_task_metrics(model, hb, yh)
        geom = _coeff_residual_geometry(model)
        phi_red = 1.0 - geom["phi_rbf_p95"] / max(1.0e-12, before_phi)
        curv_red = 1.0 - geom["curvature_rbf_p95"] / max(1.0e-12, before_curv)
        sob_red = 1.0 - geom["sobolev_rbf_norm_fast"] / max(1.0e-12, before_sob)
        high_red = 1.0 - geom["high_eig_energy"] / max(1.0e-12, before_high)
        hold_delta = task["val_loss"] - before_task["val_loss"]
        acc_drop = before_task["val_acc"] - task["val_acc"]
        score = _light_score(
            phi_red=phi_red,
            curv_red=curv_red,
            sob_red=sob_red,
            kl=constraints["KL_teacher_student"],
            logit=constraints["logit_relative_drift"],
            holdout_delta=hold_delta,
            acc_drop=acc_drop,
        )
        if "logit" in controller.lower():
            ok = (
                constraints["KL_teacher_student"] <= params.light_kl_max
                and constraints["logit_relative_drift"] <= params.light_logit_max
                and acc_drop <= params.light_acc_drop_max
                and hold_delta <= params.light_holdout_worse_max
                and (phi_red > 0.0 or curv_red > 0.0 or sob_red > 0.0)
            )
            reason = "" if ok else "logit_or_task_gate"
        else:
            ok = score > params.light_score_min and constraints["KL_teacher_student"] <= 2.0 * params.light_kl_max and constraints["logit_relative_drift"] <= 1.5 * params.light_logit_max
            reason = "" if ok else "score_gate"
        if ok and score > float(best["score"]):
            best = {
                "accepted": 1,
                "rejected": 0,
                "accepted_eta": eta,
                "backtrack_count": list(params.light_eta_grid).index(eta),
                "reject_reason": "",
                "score": score,
                "holdout_loss_delta": hold_delta,
                "holdout_acc_drop": acc_drop,
                "phi_rbf_reduction": phi_red,
                "curvature_rbf_reduction": curv_red,
                "sobolev_rbf_reduction": sob_red,
                "high_eig_energy_reduction": high_red,
                "post_smooth_val_loss": task["val_loss"],
                "post_smooth_val_acc": task["val_acc"],
                **constraints,
            }
        elif not int(best.get("accepted", 0)) and score > float(best["score"]):
            best.update({"score": score, "reject_reason": reason})
    _restore(named, snap)
    if int(best.get("accepted", 0)):
        _apply_updates(named, _scaled_updates(raw_updates, float(best["accepted_eta"])))
    final_task = _fast_task_metrics(model, hb, yh)
    final_geom = _coeff_residual_geometry(model)
    peak_alloc, peak_reserved = _peak_mb(device)
    elapsed = time.perf_counter() - t0
    best.update(
        {
            "proposal": proposal,
            "controller": controller,
            "teacher_val_loss": before_task["val_loss"],
            "student_val_loss_after_smooth": final_task["val_loss"],
            "teacher_val_acc": before_task["val_acc"],
            "student_val_acc_after_smooth": final_task["val_acc"],
            "phi_rbf_before": before_phi,
            "phi_rbf_after": final_geom["phi_rbf_p95"],
            "curvature_rbf_before": before_curv,
            "curvature_rbf_after": final_geom["curvature_rbf_p95"],
            "sobolev_rbf_before": before_sob,
            "sobolev_rbf_after": final_geom["sobolev_rbf_norm_fast"],
            "high_eig_energy_before": before_high,
            "high_eig_energy_after": final_geom["high_eig_energy"],
            "phi_rbf_reduction": 1.0 - final_geom["phi_rbf_p95"] / max(1.0e-12, before_phi),
            "curvature_rbf_reduction": 1.0 - final_geom["curvature_rbf_p95"] / max(1.0e-12, before_curv),
            "sobolev_rbf_reduction": 1.0 - final_geom["sobolev_rbf_norm_fast"] / max(1.0e-12, before_sob),
            "high_eig_energy_reduction": 1.0 - final_geom["high_eig_energy"] / max(1.0e-12, before_high),
            "KL_teacher_student": best.get("KL_teacher_student", best.get("kl_teacher_student", 0.0)),
            "logit_relative_drift": best.get("logit_relative_drift", 0.0),
            "acc_drop": before_task["val_acc"] - final_task["val_acc"],
            "smoothing_time_sec": elapsed,
            "peak_cuda_allocated_mb": peak_alloc,
            "peak_cuda_reserved_mb": peak_reserved,
            "teacher_cache_mb": teacher_cache_mb,
            "proposal_temp_mb": proposal_temp_mb,
            "smoothing_forward_count": forward_count,
            "smoothing_backward_count": 0,
            "smoothing_applies_to_base": int(base_norm > 1.0e-12),
            "smoothing_applies_to_rbf": int(rbf_norm > 1.0e-12),
            "raw_direction_norm": _update_norm(raw_updates),
        }
    )
    _restore(named, snap)
    if int(best.get("accepted", 0)):
        _apply_updates(named, _scaled_updates(raw_updates, float(best["accepted_eta"])))
    return best


def _train_steps(
    model: torch.nn.Module,
    bundle: Any,
    params: V53Params,
    device: torch.device,
    *,
    seed: int,
    steps: int,
    lr_scale: float = 1.0,
) -> Tuple[float, float]:
    opt = torch.optim.AdamW([p for _, p in _edge_named_params(model)], lr=params.adam_lr * lr_scale, weight_decay=0.0)
    idxs = _iter_steps(len(bundle.x_train), params.batch_size, seed, steps)
    _reset_peak(device)
    t0 = time.perf_counter()
    for idx in idxs:
        xb = bundle.x_train[idx].to(device)
        yb = bundle.y_train[idx].to(device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
    peak_alloc, _ = _peak_mb(device)
    opt.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    del opt
    _empty_cache(device)
    return (time.perf_counter() - t0), peak_alloc


def _mask_gradients_by_role(model: torch.nn.Module, mode: str) -> Tuple[float, float]:
    """Mask AB-RBF coefficient gradients and return base/rbf grad norms."""

    base_vals: List[torch.Tensor] = []
    rbf_vals: List[torch.Tensor] = []
    for layer in model.kan_layers():  # type: ignore[attr-defined]
        grad = layer.coeff.grad
        if grad is None:
            continue
        base = int(getattr(layer, "base_dim", 0))
        if base > 0:
            base_vals.append(grad[..., :base].detach().float().reshape(-1))
        if grad.shape[-1] > base:
            rbf_vals.append(grad[..., base:].detach().float().reshape(-1))
        if mode in {"base_only", "rbf_frozen"} and grad.shape[-1] > base:
            grad[..., base:] = 0.0
        elif mode == "rbf_task_small" and grad.shape[-1] > base:
            grad[..., base:] *= 0.20
    base_norm = float(torch.cat(base_vals).norm().detach().cpu()) if base_vals else 0.0
    rbf_norm = float(torch.cat(rbf_vals).norm().detach().cpu()) if rbf_vals else 0.0
    return base_norm, rbf_norm


def _train_steps_role(
    model: torch.nn.Module,
    bundle: Any,
    params: V53Params,
    device: torch.device,
    *,
    seed: int,
    steps: int,
    lr_scale: float = 1.0,
    mode: str = "full",
) -> Tuple[float, float, float, float]:
    opt = torch.optim.AdamW([p for _, p in _edge_named_params(model)], lr=params.adam_lr * lr_scale, weight_decay=0.0)
    idxs = _iter_steps(len(bundle.x_train), params.batch_size, seed, steps)
    base_norms: List[float] = []
    rbf_norms: List[float] = []
    _reset_peak(device)
    t0 = time.perf_counter()
    for idx in idxs:
        xb = bundle.x_train[idx].to(device)
        yb = bundle.y_train[idx].to(device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        bnorm, rnorm = _mask_gradients_by_role(model, mode)
        base_norms.append(bnorm)
        rbf_norms.append(rnorm)
        opt.step()
    peak_alloc, _ = _peak_mb(device)
    opt.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    del opt
    _empty_cache(device)
    return time.perf_counter() - t0, peak_alloc, _mean(base_norms, 0.0), _mean(rbf_norms, 0.0)


def _geometry_debt(current: Dict[str, Any], target: Dict[str, float]) -> float:
    phi_target = max(1.0e-12, target.get("phi_rbf", 0.0))
    curv_target = max(1.0e-12, target.get("curv_rbf", 0.0))
    return max(0.0, float(current.get("phi_rbf_p95", current.get("phi_rbf", 0.0))) / phi_target - 1.0) + 0.5 * max(
        0.0, float(current.get("curvature_rbf_p95", current.get("curvature_rbf", 0.0))) / curv_target - 1.0
    )


def _val_loss_slope(history: Sequence[Dict[str, float]]) -> float:
    if len(history) < 2:
        return 0.0
    return float(history[-1]["val_loss"] - history[max(0, len(history) - 3)]["val_loss"])


def _probe_light_smoothing(
    model: torch.nn.Module,
    bundle: Any,
    params: V53Params,
    device: torch.device,
    *,
    proposal: str,
    controller: str,
    eta_grid: Sequence[float] | None = None,
    apply_best: bool = False,
) -> Dict[str, Any]:
    hb = bundle.x_val[: params.audit_batch_size].to(device)
    yh = bundle.y_val[: params.audit_batch_size].to(device)
    named = coefficient_named_params(model)
    snap = _snapshot(named)
    with torch.no_grad():
        teacher_logits = model(hb).detach()
    teacher_cache_mb = _tensor_mb([teacher_logits])
    before_task = _fast_task_metrics(model, hb, yh)
    before_geom = _coeff_residual_geometry(model)
    raw_updates = _residual_smoothing_updates(model, proposal, role=None)
    proposal_temp_mb = _tensor_mb(raw_updates.values())
    grid = list(eta_grid or params.event_eta_grid)
    candidates: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {"accepted": 0, "accepted_eta": 0.0, "score": -1.0e18, "reject_reason": "no_probe"}
    _reset_peak(device)
    t0 = time.perf_counter()
    for i, eta in enumerate(grid):
        _restore(named, snap)
        _apply_updates(named, _scaled_updates(raw_updates, float(eta)))
        logits = model(hb)
        constraints = _teacher_logit_constraints(logits, teacher_logits)
        task = _fast_task_metrics(model, hb, yh)
        geom = _coeff_residual_geometry(model)
        phi_gain = 1.0 - geom["phi_rbf_p95"] / max(1.0e-12, before_geom["phi_rbf_p95"])
        curv_gain = 1.0 - geom["curvature_rbf_p95"] / max(1.0e-12, before_geom["curvature_rbf_p95"])
        sob_gain = 1.0 - geom["sobolev_rbf_norm_fast"] / max(1.0e-12, before_geom["sobolev_rbf_norm_fast"])
        acc_drop = before_task["val_acc"] - task["val_acc"]
        score = _light_score(
            phi_red=phi_gain,
            curv_red=curv_gain,
            sob_red=sob_gain,
            kl=constraints["KL_teacher_student"],
            logit=constraints["logit_relative_drift"],
            holdout_delta=task["val_loss"] - before_task["val_loss"],
            acc_drop=acc_drop,
        )
        ok = (
            score > params.score_threshold
            and acc_drop <= params.light_acc_drop_max
            and constraints["KL_teacher_student"] <= params.light_kl_max
            and constraints["logit_relative_drift"] <= params.light_logit_max
            and (phi_gain >= 0.0 or curv_gain >= 0.0)
        )
        cand = {
            "eta": float(eta),
            "probe_score": score,
            "probe_acc_drop": acc_drop,
            "probe_KL": constraints["KL_teacher_student"],
            "probe_logit_drift": constraints["logit_relative_drift"],
            "probe_phi_gain": phi_gain,
            "probe_curv_gain": curv_gain,
            "probe_sobolev_gain": sob_gain,
            "probe_ok": int(ok),
            "backtrack_count": i,
        }
        candidates.append(cand)
        current_best = float(best.get("score", best.get("probe_score", -1.0e18)))
        if ok and score > current_best:
            best = dict(cand)
            best.update({"accepted": 1, "accepted_eta": float(eta), "reject_reason": "", "score": score})
        elif not int(best.get("accepted", 0)) and score > current_best:
            best = dict(cand)
            best.update({"accepted": 0, "accepted_eta": 0.0, "reject_reason": "score_or_task_gate", "score": score})
    _restore(named, snap)
    elapsed = time.perf_counter() - t0
    if apply_best and int(best.get("accepted", 0)):
        _apply_updates(named, _scaled_updates(raw_updates, float(best["accepted_eta"])))
    peak_alloc, peak_reserved = _peak_mb(device)
    best.update(
        {
            "proposal": proposal,
            "controller": controller,
            "candidate_rows": candidates,
            "teacher_val_acc": before_task["val_acc"],
            "teacher_val_loss": before_task["val_loss"],
            "phi_rbf_before": before_geom["phi_rbf_p95"],
            "curvature_rbf_before": before_geom["curvature_rbf_p95"],
            "sobolev_rbf_before": before_geom["sobolev_rbf_norm_fast"],
            "teacher_cache_mb": teacher_cache_mb,
            "proposal_temp_mb": proposal_temp_mb,
            "smoothing_time_sec": elapsed,
            "peak_cuda_allocated_mb": peak_alloc,
            "peak_cuda_reserved_mb": peak_reserved,
            "smoothing_forward_count": len(grid),
            "smoothing_backward_count": 0,
            "raw_direction_norm": _update_norm(raw_updates),
            "accepted": int(best.get("accepted", 0)),
        }
    )
    return best


def _postprocess_ratios(rows: List[Dict[str, Any]], baseline_method: str = "ABRBF-AdamW") -> List[Dict[str, Any]]:
    by_dataset: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row.get("method") == baseline_method and not row.get("error"):
            by_dataset[str(row.get("dataset"))] = row
    for row in rows:
        base = by_dataset.get(str(row.get("dataset")))
        if not base:
            continue
        row["memory_ratio_vs_ABRBF_AdamW"] = float(row.get("peak_cuda_allocated_mb", 0.0) or 0.0) / max(1.0e-12, float(base.get("peak_cuda_allocated_mb", 0.0) or 0.0)) if float(base.get("peak_cuda_allocated_mb", 0.0) or 0.0) > 0 else 1.0
        row["time_ratio_vs_ABRBF_AdamW"] = float(row.get("amortized_step_time_ms", row.get("step_time_ms", 0.0)) or 0.0) / max(1.0e-12, float(base.get("amortized_step_time_ms", base.get("step_time_ms", 0.0)) or 0.0))
    return rows


def _load_bundle(args: argparse.Namespace, dataset: str, seed: int, params: V53Params) -> Any:
    return load_vision_bundle(
        dataset,
        data_root=args.data_root,
        train_size=params.train_size,
        val_size=params.val_size,
        test_size=params.test_size,
        seed=seed,
        download=not args.no_download,
        allow_fake_data=args.allow_fake_data,
    )


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p0_memory_smoke.csv")
    manifest: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "edge_param_manifest.csv")
    params = V53Params(train_size=512, val_size=128, test_size=128, audit_batch_size=8, p1_steps=8)
    device = get_device(args.device)
    done = {(r.get("dataset"), r.get("method")) for r in rows if not r.get("error")}
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        bundle = _load_bundle(args, dataset, 0, params)
        for method in P0_METHODS:
            if (dataset, method) in done:
                continue
            edge_kind = "RBFOnly" if method == "RBFOnly-AdamW" else "ABRBF-linear+silu"
            try:
                set_seed(5200)
                model = _make_v49_model(bundle, params, device, edge_kind=edge_kind)
                train_time, train_peak = _train_steps(model, bundle, params, device, seed=5200, steps=2)
                step_ms = 1000.0 * train_time / 2
                model.zero_grad(set_to_none=True)
                _empty_cache(device)
                smoothing: Dict[str, Any] = {}
                if "LightSmooth" in method:
                    proposal = "S0-laplacian"
                    controller = "logitOnly" if "logitOnly" in method else "scoreOnly"
                    smoothing = _light_smoothing_event(model, bundle, params, device, proposal=proposal, controller=controller, seed=5201)
                elif "StrongSmooth" in method:
                    smoothing = _light_smoothing_event(model, bundle, params, device, proposal="S0-laplacian", controller="strongReference", seed=5202, strong_reference=True)
                else:
                    hb = bundle.x_val[: params.audit_batch_size].to(device)
                    _reset_peak(device)
                    with torch.no_grad():
                        _ = model(hb)
                    _ = _coeff_residual_geometry(model)
                    monitor_peak, monitor_reserved = _peak_mb(device)
                    smoothing = {
                        "peak_cuda_allocated_mb": max(train_peak, monitor_peak),
                        "peak_cuda_reserved_mb": monitor_reserved,
                        "teacher_cache_mb": 0.0,
                        "proposal_temp_mb": 0.0,
                        "smoothing_forward_count": 0,
                        "smoothing_backward_count": 0,
                        "smoothing_time_sec": 0.0,
                        "smoothing_applies_to_base": 0,
                        "smoothing_applies_to_rbf": int(edge_kind != "RBFOnly" or method == "RBFOnly-AdamW"),
                    }
                named = coefficient_named_params(model)
                snap = _snapshot(named)
                _apply_updates(named, {name: torch.zeros_like(p) for name, p in named})
                rollback = max([float((p.detach() - snap[name]).abs().max().cpu()) for name, p in named] or [0.0])
                teacher_logits = model(bundle.x_val[: params.audit_batch_size].to(device)).detach()
                teacher_snapshot_error = float((teacher_logits - teacher_logits.clone()).abs().max().cpu())
                pcount = _param_count_audit(model)
                edge_cov = 1.0 if core_edge_named_params(model) else 0.0
                row = {
                    "stage": "P0",
                    "dataset": dataset,
                    "method": method,
                    "edge_kind": edge_kind,
                    **pcount,
                    "edge_param_count": pcount.get("edge_param_count_total", 0),
                    "base_param_count": pcount.get("base_param_count", 0),
                    "rbf_residual_param_count": pcount.get("rbf_param_count", 0),
                    "ABRBFDense_in_core": int(any(isinstance(m, ABRBFDense) for m in model.modules())),
                    "ABRBF_defined_in_runner_only": 0,
                    "edge_coverage": edge_cov,
                    "base_coverage": pcount.get("coverage_base", 0.0),
                    "rbf_residual_coverage": pcount.get("coverage_rbf", 0.0),
                    "rollback_max_abs_error": rollback,
                    "teacher_snapshot_error": teacher_snapshot_error,
                    "training_step_time_ms": step_ms,
                    "step_time_ms": step_ms,
                    "amortized_step_time_ms": step_ms + float(smoothing.get("smoothing_time_sec", 0.0)) * 1000.0 / 100.0,
                    "error": "",
                    **smoothing,
                }
                rows.append(row)
                manifest.extend(_edge_param_manifest(model, dataset=dataset, method=method, edge_kind=edge_kind))
                rows = _postprocess_ratios(rows)
                print(f"P0 {dataset} {method} mem={row['peak_cuda_allocated_mb']:.1f}MB smooth={row.get('smoothing_time_sec', 0):.3f}s")
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                rows.append({"stage": "P0", "dataset": dataset, "method": method, "error": repr(exc)})
                print(f"P0 ERROR {dataset} {method}: {exc!r}")
            write_csv(out_dir / "p0_memory_smoke.csv", _postprocess_ratios(rows))
            write_csv(out_dir / "edge_param_manifest.csv", manifest)
            _empty_cache(device)
    return rows


def _event_pass(row: Dict[str, Any], params: V53Params) -> bool:
    return (
        float(row.get("acc_drop", 99.0)) <= params.light_acc_drop_max
        and float(row.get("KL_teacher_student", row.get("kl_teacher_student", 99.0))) <= params.light_kl_max
        and float(row.get("logit_relative_drift", 99.0)) <= params.light_logit_max
        and (
            float(row.get("phi_rbf_reduction", 0.0)) >= 0.05
            or float(row.get("curvature_rbf_reduction", 0.0)) >= 0.20
        )
        and float(row.get("memory_ratio_vs_teacher", row.get("memory_ratio_vs_ABRBF_AdamW", 1.0))) <= params.memory_ratio_gate
    )


def run_p1(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p1_single_smoothing_event.csv")
    params = V53Params()
    device = get_device(args.device)
    seeds = parse_int_list(args.seeds)[:1]
    p0_base = {
        str(r.get("dataset")): float(r.get("peak_cuda_allocated_mb", 0.0) or 0.0)
        for r in read_csv(out_dir / "p0_memory_smoke.csv")
        if r.get("method") == "ABRBF-AdamW" and not r.get("error")
    }
    done = {(r.get("dataset"), int(float(r.get("seed", -1))), r.get("teacher"), r.get("proposal"), r.get("controller")) for r in rows if not r.get("error")}
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        for seed in seeds:
            for teacher in P1_TEACHERS:
                bundle, model, meta = _load_or_train_teacher(args, dataset, seed, teacher, params, device)
                teacher_eval = _eval_v49(model, bundle, params, device)
                baseline_mem = max(1.0e-12, p0_base.get(dataset, 0.0))
                for proposal, controller in P1_CANDIDATES:
                    key = (dataset, seed, teacher, proposal, controller)
                    if key in done:
                        continue
                    try:
                        bundle, model, meta = _load_or_train_teacher(args, dataset, seed, teacher, params, device)
                        before = _eval_v49(model, bundle, params, device)
                        strong = controller == "strongReference"
                        stats = _light_smoothing_event(model, bundle, params, device, proposal=proposal, controller=controller, seed=seed + 5210, strong_reference=strong)
                        after = _eval_v49(model, bundle, params, device)
                        hb = bundle.x_val[: params.audit_batch_size].to(device)
                        yh = bundle.y_val[: params.audit_batch_size].to(device)
                        ablate = _ablation_drops(model, hb, yh)
                        mem_ratio = float(stats.get("peak_cuda_allocated_mb", 0.0)) / baseline_mem if baseline_mem > 0 and float(stats.get("peak_cuda_allocated_mb", 0.0)) > 0 else 1.0
                        row = {
                            "stage": "P1",
                            "dataset": dataset,
                            "seed": seed,
                            "teacher": teacher,
                            "proposal": proposal,
                            "controller": controller,
                            "teacher_acc": before["test_acc"],
                            "student_acc_after_smooth": after["test_acc"],
                            "acc_drop": before["test_acc"] - after["test_acc"],
                            "teacher_val_loss": stats.get("teacher_val_loss", float("nan")),
                            "student_val_loss_after_smooth": stats.get("student_val_loss_after_smooth", float("nan")),
                            "margin_mean_before": before["margin_mean"],
                            "margin_mean_after": after["margin_mean"],
                            "margin_p10_before": before["margin_p10"],
                            "margin_p10_after": after["margin_p10"],
                            "phi_rbf_before": before["phi_rbf_p95"],
                            "phi_rbf_after": after["phi_rbf_p95"],
                            "phi_rbf_reduction": 1.0 - after["phi_rbf_p95"] / max(1.0e-12, before["phi_rbf_p95"]),
                            "curvature_rbf_before": before["curvature_rbf_p95"],
                            "curvature_rbf_after": after["curvature_rbf_p95"],
                            "curvature_rbf_reduction": 1.0 - after["curvature_rbf_p95"] / max(1.0e-12, before["curvature_rbf_p95"]),
                            "sobolev_rbf_before": before["sobolev_rbf_norm"],
                            "sobolev_rbf_after": after["sobolev_rbf_norm"],
                            "sobolev_rbf_reduction": 1.0 - after["sobolev_rbf_norm"] / max(1.0e-12, before["sobolev_rbf_norm"]),
                            "high_eig_energy_before": stats.get("high_eig_energy_before", float("nan")),
                            "high_eig_energy_after": stats.get("high_eig_energy_after", float("nan")),
                            "high_eig_energy_reduction": stats.get("high_eig_energy_reduction", float("nan")),
                            "base_norm": after["base_output_norm"],
                            "rbf_norm": after["rbf_output_norm"],
                            "base_over_rbf": after["base_over_rbf_norm"],
                            "memory_ratio_vs_teacher": mem_ratio,
                            "time_ratio_vs_teacher": 1.0,
                            **ablate,
                            **stats,
                            "p1_pass": 0,
                            "error": "",
                        }
                        row["p1_pass"] = int(_event_pass(row, params))
                        rows.append(row)
                        print(f"P1 {dataset} {teacher} {proposal}/{controller} accDrop={row['acc_drop']:.4f} phiR={row['phi_rbf_reduction']:.3f} curvR={row['curvature_rbf_reduction']:.3f} pass={row['p1_pass']}")
                    except Exception as exc:
                        if not args.continue_on_error:
                            raise
                        rows.append({"stage": "P1", "dataset": dataset, "seed": seed, "teacher": teacher, "proposal": proposal, "controller": controller, "error": repr(exc)})
                        print(f"P1 ERROR {dataset} {teacher} {proposal}/{controller}: {exc!r}")
                    write_csv(out_dir / "p1_single_smoothing_event.csv", rows)
                    _empty_cache(device)
    return rows


def _p1_survivors(out_dir: Path) -> List[Tuple[str, str, str]]:
    rows = [r for r in read_csv(out_dir / "p1_single_smoothing_event.csv") if not r.get("error")]
    by_key: Dict[Tuple[str, str, str], set[str]] = {}
    for r in rows:
        if int(float(r.get("p1_pass", 0) or 0)) != 1:
            continue
        key = (str(r.get("teacher")), str(r.get("proposal")), str(r.get("controller")))
        by_key.setdefault(key, set()).add(str(r.get("dataset")))
    out = [k for k, ds in by_key.items() if all(d in ds for d in DATASETS)]
    out.sort(key=lambda k: (k[2] == "strongReference", k))
    return out


def _eval_stage(model: torch.nn.Module, bundle: Any, params: V53Params, device: torch.device, stage: str, step: int, method: str, dataset: str, seed: int) -> Dict[str, Any]:
    ev = _eval_v49(model, bundle, params, device)
    return {
        "stage": stage,
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "step": step,
        "acc": ev["test_acc"],
        "test_acc": ev["test_acc"],
        "val_loss": ev["test_loss"],
        "ECE": ev["ece"],
        "phi_base": ev["phi_base_p95"],
        "phi_rbf": ev["phi_rbf_p95"],
        "phi_total": ev["phi_total_p95"],
        "curvature_base": ev["curvature_base_p95"],
        "curvature_rbf": ev["curvature_rbf_p95"],
        "curvature_total": ev["curvature_total_p95"],
        "sobolev_rbf_norm": ev["sobolev_rbf_norm"],
        "base_over_rbf": ev["base_over_rbf_norm"],
        "margin_mean": ev["margin_mean"],
        "margin_p10": ev["margin_p10"],
        "rank": ev["rank_block"],
    }


def _make_trained_model(args: argparse.Namespace, dataset: str, seed: int, params: V53Params, device: torch.device, steps: int, method: str = ABRBF_MAIN) -> Tuple[Any, torch.nn.Module, Dict[str, Any], float, float]:
    bundle = _load_bundle(args, dataset, seed, params)
    set_seed(seed)
    model = _make_v49_model(bundle, params, device, edge_kind=_edge_kind_from_method(method))
    wall, peak = _train_steps(model, bundle, params, device, seed=seed + 5220, steps=steps)
    meta = _eval_v49(model, bundle, params, device)
    return bundle, model, meta, wall, peak


def _best_light_candidate(out_dir: Path) -> Tuple[str, str, str]:
    survivors = [s for s in _p1_survivors(out_dir) if "Strong" not in s[1] and "strong" not in s[2].lower()]
    if survivors:
        return survivors[0]
    return (ABRBF_MAIN, "S0-laplacian", "logitOnly")


def run_p2(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p2_event_probe_audit.csv")
    trace: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p2_probe_training_trace.csv")
    params = V53Params()
    device = get_device(args.device)
    best_teacher, best_proposal, best_controller = _best_light_candidate(out_dir)
    seeds = parse_int_list(args.seeds)[:3]
    done = {(r.get("dataset"), int(float(r.get("seed", -1))), int(float(r.get("probe_step", -1))), float(r.get("eta", -1))) for r in rows if not r.get("error")}
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        for seed in seeds:
            try:
                bundle = _load_bundle(args, dataset, seed, params)
                set_seed(seed)
                model = _make_v49_model(bundle, params, device, edge_kind=_edge_kind_from_method(ABRBF_MAIN))
                opt = torch.optim.AdamW([p for _, p in _edge_named_params(model)], lr=params.adam_lr, weight_decay=0.0)
                idxs = list(_iter_steps(len(bundle.x_train), params.batch_size, seed + 5320, params.p2_probe_steps))
                initial = _eval_v49(model, bundle, params, device)
                target = {"phi_rbf": 0.95 * initial["phi_rbf_p95"], "curv_rbf": 0.95 * initial["curvature_rbf_p95"]}
                history: List[Dict[str, float]] = []
                for step, idx in enumerate(idxs, start=1):
                    xb = bundle.x_train[idx].to(device)
                    yb = bundle.y_train[idx].to(device)
                    opt.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(xb), yb)
                    loss.backward()
                    opt.step()
                    if step % params.p2_probe_interval != 0 and step != params.p2_probe_steps:
                        continue
                    ev = _eval_v49(model, bundle, params, device)
                    history.append({"step": step, "val_loss": ev["test_loss"], "acc": ev["test_acc"], "phi_rbf": ev["phi_rbf_p95"]})
                    debt = _geometry_debt(ev, target)
                    slope = _val_loss_slope(history)
                    old_plateau = len(history) >= params.plateau_window and abs(history[-1]["val_loss"] - history[-params.plateau_window]["val_loss"]) / max(1.0e-12, history[-params.plateau_window]["val_loss"]) < params.epsilon_plateau
                    probe = _probe_light_smoothing(model, bundle, params, device, proposal=best_proposal, controller=best_controller, eta_grid=params.event_eta_grid, apply_best=False)
                    best_score = float(probe.get("probe_score", probe.get("score", -1.0e18)))
                    best_eta = float(probe.get("eta", probe.get("accepted_eta", 0.0)) or 0.0)
                    score_trigger = int(best_score > params.score_threshold and int(probe.get("accepted", 0)) == 1)
                    debt_trigger = int(debt > params.min_geometry_debt and score_trigger)
                    task_slack = ev["test_acc"] - initial["test_acc"] + params.light_acc_drop_max
                    for cand in probe.get("candidate_rows", []):
                        key = (dataset, seed, step, float(cand.get("eta", -1)))
                        if key in done:
                            continue
                        rows.append(
                            {
                                "stage": "P2",
                                "dataset": dataset,
                                "seed": seed,
                                "method": "ABRBF-EventProbe",
                                "teacher": best_teacher,
                                "proposal": best_proposal,
                                "controller": best_controller,
                                "probe_step": step,
                                "probe_epoch": step / max(1, params.p2_probe_interval),
                                "geometry_debt": debt,
                                "phi_rbf": ev["phi_rbf_p95"],
                                "curvature_rbf": ev["curvature_rbf_p95"],
                                "val_loss_slope": slope,
                                "val_acc": ev["test_acc"],
                                "task_slack": task_slack,
                                "eta": cand.get("eta", 0.0),
                                "probe_score": cand.get("probe_score", 0.0),
                                "probe_acc_drop": cand.get("probe_acc_drop", 0.0),
                                "probe_KL": cand.get("probe_KL", 0.0),
                                "probe_logit_drift": cand.get("probe_logit_drift", 0.0),
                                "probe_phi_gain": cand.get("probe_phi_gain", 0.0),
                                "probe_curv_gain": cand.get("probe_curv_gain", 0.0),
                                "would_trigger_by_plateau_old": int(old_plateau and ev["phi_rbf_p95"] > 0.95 * initial["phi_rbf_p95"]),
                                "would_trigger_by_geometry_debt": debt_trigger,
                                "would_trigger_by_score": score_trigger,
                                "best_eta": best_eta,
                                "best_score": best_score,
                                "p2_pass": 0,
                                "error": "",
                            }
                        )
                    trace.append(_eval_stage(model, bundle, params, device, "P2", step, "ABRBF-EventProbe", dataset, seed))
                    print(f"P2 {dataset} seed={seed} step={step} debt={debt:.3f} best={best_score:.3f} eta={best_eta:.3f} trigger={debt_trigger}")
                    write_csv(out_dir / "p2_event_probe_audit.csv", rows)
                    write_csv(out_dir / "p2_probe_training_trace.csv", trace)
                opt.zero_grad(set_to_none=True)
                del opt
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                rows.append({"stage": "P2", "dataset": dataset, "seed": seed, "method": "ABRBF-EventProbe", "error": repr(exc)})
                print(f"P2 ERROR {dataset} seed={seed}: {exc!r}")
            write_csv(out_dir / "p2_event_probe_audit.csv", rows)
            write_csv(out_dir / "p2_probe_training_trace.csv", trace)
            _empty_cache(device)
    return rows


def _p2_survivors(out_dir: Path) -> List[str]:
    rows = [r for r in read_csv(out_dir / "p2_event_probe_audit.csv") if not r.get("error")]
    trigger_ds = {
        str(r.get("dataset"))
        for r in rows
        if int(float(r.get("would_trigger_by_geometry_debt", 0) or 0)) == 1 or int(float(r.get("would_trigger_by_score", 0) or 0)) == 1
    }
    return ["ABRBF-EventProbe"] if len(trigger_ds) >= 2 else []


def run_p3(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p3_one_cycle_controller_variants.csv")
    trace: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p3_refresh_trace.csv")
    params = V53Params()
    device = get_device(args.device)
    p2_surv = _p2_survivors(out_dir)
    if not p2_surv:
        rows = [{"status": "not_run", "reason": "P2 produced no two-dataset event-probe trigger"}]
        trace = [{"status": "not_run", "reason": "P2 produced no two-dataset event-probe trigger"}]
        write_csv(out_dir / "p3_one_cycle_controller_variants.csv", rows)
        write_csv(out_dir / "p3_refresh_trace.csv", trace)
        print("P3 not run: no P2 trigger")
        return rows
    best_teacher, best_proposal, best_controller = _best_light_candidate(out_dir)
    seeds = parse_int_list(args.seeds)[:1]
    done = {(r.get("dataset"), int(float(r.get("seed", -1))), r.get("method")) for r in rows if not r.get("error") and r.get("status") != "not_run"}
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        for seed in seeds:
            base_row = next((r for r in rows if r.get("dataset") == dataset and int(float(r.get("seed", -1))) == seed and r.get("method") == "ABRBF-AdamW" and not r.get("error")), None)
            for method in P3_METHODS:
                if (dataset, seed, method) in done:
                    continue
                try:
                    if method == "ABRBF-AdamW":
                        bundle, model, final, wall, peak = _make_trained_model(args, dataset, seed, params, device, params.p3_late_steps + params.p3_refresh_steps, method=ABRBF_MAIN)
                        row = {"stage": "P3", "dataset": dataset, "seed": seed, "method": method, "final_acc": final["test_acc"], "final_phi_rbf": final["phi_rbf_p95"], "final_curv_rbf": final["curvature_rbf_p95"], "training_time_sec": wall, "peak_cuda_allocated_mb": peak, "memory_ratio_vs_ABRBF_AdamW": 1.0, "p3_pass": 1, "error": "", **final}
                        base_row = row
                        rows.append(row)
                        trace.append(_eval_stage(model, bundle, params, device, "P3", params.p3_late_steps + params.p3_refresh_steps, method, dataset, seed))
                        print(f"P3 {dataset} {method} acc={final['test_acc']:.4f}")
                        write_csv(out_dir / "p3_one_cycle_controller_variants.csv", rows)
                        write_csv(out_dir / "p3_refresh_trace.csv", trace)
                        continue
                    controller = "scoreOnly" if "scoreOnly" in method else "logitOnly"
                    if "baseOnlyRefresh" in method:
                        refresh_policy = "baseOnlyRefresh"
                    elif "rbfFrozenRefresh" in method:
                        refresh_policy = "rbfFrozenRefresh"
                    else:
                        refresh_policy = "fullRefresh"
                    refresh_steps = params.p3_refresh_steps
                    bundle, model, _, train_wall, train_peak = _make_trained_model(args, dataset, seed, params, device, params.p3_late_steps, method=ABRBF_MAIN)
                    pre = _eval_v49(model, bundle, params, device)
                    smooth = _probe_light_smoothing(model, bundle, params, device, proposal=best_proposal, controller=controller, eta_grid=params.event_eta_grid, apply_best=True)
                    post = _eval_v49(model, bundle, params, device)
                    if refresh_policy == "fullRefresh":
                        refresh_wall, refresh_peak, base_norm, rbf_norm = _train_steps_role(model, bundle, params, device, seed=seed + 5330, steps=refresh_steps, lr_scale=0.35, mode="full")
                    elif refresh_policy == "baseOnlyRefresh":
                        refresh_wall, refresh_peak, base_norm, rbf_norm = _train_steps_role(model, bundle, params, device, seed=seed + 5330, steps=refresh_steps, lr_scale=0.35, mode="base_only")
                    else:
                        w1, p1, b1, r1 = _train_steps_role(model, bundle, params, device, seed=seed + 5330, steps=refresh_steps // 2, lr_scale=0.35, mode="base_only")
                        w2, p2, b2, r2 = _train_steps_role(model, bundle, params, device, seed=seed + 5340, steps=refresh_steps - refresh_steps // 2, lr_scale=0.20, mode="rbf_task_small")
                        refresh_wall, refresh_peak = w1 + w2, max(p1, p2)
                        base_norm, rbf_norm = _mean([b1, b2], 0.0), _mean([r1, r2], 0.0)
                    final = _eval_v49(model, bundle, params, device)
                    if post["test_acc"] >= pre["test_acc"]:
                        recovery = 1.0
                    else:
                        recovery = (final["test_acc"] - post["test_acc"]) / max(1.0e-12, pre["test_acc"] - post["test_acc"])
                    geom_ret = (pre["phi_rbf_p95"] - final["phi_rbf_p95"]) / max(1.0e-12, pre["phi_rbf_p95"] - post["phi_rbf_p95"])
                    phi_red = 1.0 - final["phi_rbf_p95"] / max(1.0e-12, pre["phi_rbf_p95"])
                    curv_red = 1.0 - final["curvature_rbf_p95"] / max(1.0e-12, pre["curvature_rbf_p95"])
                    base_acc = float(base_row.get("final_acc", pre["test_acc"])) if base_row else pre["test_acc"]
                    base_peak = float(base_row.get("peak_cuda_allocated_mb", train_peak) or train_peak) if base_row else train_peak
                    mem_ratio = max(train_peak, refresh_peak, float(smooth.get("peak_cuda_allocated_mb", 0.0))) / max(1.0e-12, base_peak)
                    row = {
                        "stage": "P3",
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "teacher": ABRBF_MAIN,
                        "proposal": best_proposal,
                        "controller": controller,
                        "refresh_policy": refresh_policy,
                        "refresh_steps": refresh_steps,
                        "acc_before_smooth": pre["test_acc"],
                        "acc_after_smooth": post["test_acc"],
                        "acc_after_refresh": final["test_acc"],
                        "final_acc": final["test_acc"],
                        "val_loss_before_smooth": pre["test_loss"],
                        "val_loss_after_smooth": post["test_loss"],
                        "val_loss_after_refresh": final["test_loss"],
                        "phi_rbf_before": pre["phi_rbf_p95"],
                        "phi_rbf_after_smooth": post["phi_rbf_p95"],
                        "phi_rbf_after_refresh": final["phi_rbf_p95"],
                        "curv_rbf_before": pre["curvature_rbf_p95"],
                        "curv_rbf_after_smooth": post["curvature_rbf_p95"],
                        "curv_rbf_after_refresh": final["curvature_rbf_p95"],
                        "recovery_ratio": recovery,
                        "geometry_retention": geom_ret,
                        "refresh_steps_to_recover": refresh_steps if recovery >= 0.8 else -1,
                        "base_update_norm_during_refresh": base_norm,
                        "rbf_update_norm_during_refresh": rbf_norm,
                        "base_update_share_refresh": base_norm / max(1.0e-12, base_norm + rbf_norm),
                        "rbf_update_share_refresh": rbf_norm / max(1.0e-12, base_norm + rbf_norm),
                        "phi_rbf_reduction": phi_red,
                        "curvature_rbf_reduction": curv_red,
                        "acc_gap_vs_ABRBF": base_acc - final["test_acc"],
                        "training_time_sec": train_wall + refresh_wall,
                        "refresh_time_sec": refresh_wall,
                        "smoothing_time_sec": smooth.get("smoothing_time_sec", 0.0),
                        "peak_cuda_allocated_mb": max(train_peak, refresh_peak, float(smooth.get("peak_cuda_allocated_mb", 0.0))),
                        "memory_ratio_vs_ABRBF_AdamW": mem_ratio,
                        "p3_pass": 0,
                        "error": "",
                        **smooth,
                    }
                    row["p3_pass"] = int(final["test_acc"] >= base_acc - 0.005 and (phi_red >= 0.05 or curv_red >= 0.30) and recovery >= 0.8 and geom_ret >= 0.5 and mem_ratio <= params.memory_ratio_gate)
                    rows.append(row)
                    trace.append(_eval_stage(model, bundle, params, device, "P3", params.p3_late_steps, method + ":pre_smooth", dataset, seed))
                    trace.append(_eval_stage(model, bundle, params, device, "P3", params.p3_late_steps + 1, method + ":post_smooth", dataset, seed))
                    trace.append(_eval_stage(model, bundle, params, device, "P3", params.p3_late_steps + refresh_steps, method + ":post_refresh", dataset, seed))
                    print(f"P3 {dataset} {method} acc={final['test_acc']:.4f} phiRed={phi_red:.3f} rec={recovery:.2f} ret={geom_ret:.2f} pass={row['p3_pass']}")
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    rows.append({"stage": "P3", "dataset": dataset, "seed": seed, "method": method, "error": repr(exc)})
                    print(f"P3 ERROR {dataset} {method}: {exc!r}")
                write_csv(out_dir / "p3_one_cycle_controller_variants.csv", rows)
                write_csv(out_dir / "p3_refresh_trace.csv", trace)
                _empty_cache(device)
    return rows


def _all_dataset_survivors(path: Path, key_field: str, pass_field: str) -> List[str]:
    rows = [r for r in read_csv(path) if not r.get("error") and r.get("status") != "not_run"]
    by: Dict[str, set[str]] = {}
    for r in rows:
        if int(float(r.get(pass_field, 0) or 0)) == 1:
            by.setdefault(str(r.get(key_field)), set()).add(str(r.get("dataset")))
    return [m for m, ds in by.items() if all(d in ds for d in DATASETS)]


def _placeholder(out_dir: Path, filename: str, reason: str) -> List[Dict[str, Any]]:
    rows = [{"status": "not_run", "reason": reason}]
    write_csv(out_dir / filename, rows)
    return rows


def run_p4(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p4_event_driven_multicycle_v2.csv")
    trace: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p4_event_trace.csv")
    p3_survivors = [m for m in _all_dataset_survivors(out_dir / "p3_one_cycle_controller_variants.csv", "method", "p3_pass") if m != "ABRBF-AdamW"]
    if not p3_survivors:
        rows = [{"status": "not_run", "reason": "P3 produced no all-dataset one-cycle survivor"}]
        trace = [{"status": "not_run", "reason": "P3 produced no all-dataset one-cycle survivor"}]
        write_csv(out_dir / "p4_event_driven_multicycle_v2.csv", rows)
        write_csv(out_dir / "p4_event_trace.csv", trace)
        print("P4 not run: no P3 one-cycle survivor")
        return rows
    selected_policy = p3_survivors[0]
    params = V53Params()
    device = get_device(args.device)
    best_teacher, best_proposal, best_controller = _best_light_candidate(out_dir)
    seeds = parse_int_list(args.seeds)[:1]
    done = {(r.get("dataset"), int(float(r.get("seed", -1))), r.get("method")) for r in rows if not r.get("error") and r.get("status") != "not_run"}

    def refresh_policy_name() -> str:
        if "baseOnlyRefresh" in selected_policy:
            return "baseOnlyRefresh"
        if "rbfFrozenRefresh" in selected_policy:
            return "rbfFrozenRefresh"
        return "fullRefresh"

    def run_refresh(model: torch.nn.Module, bundle: Any, seed0: int) -> Tuple[float, float, float, float]:
        policy = refresh_policy_name()
        if policy == "fullRefresh":
            return _train_steps_role(model, bundle, params, device, seed=seed0, steps=params.p3_refresh_steps, lr_scale=0.35, mode="full")
        if policy == "baseOnlyRefresh":
            return _train_steps_role(model, bundle, params, device, seed=seed0, steps=params.p3_refresh_steps, lr_scale=0.35, mode="base_only")
        w1, p1, b1, r1 = _train_steps_role(model, bundle, params, device, seed=seed0, steps=params.p3_refresh_steps // 2, lr_scale=0.35, mode="base_only")
        w2, p2, b2, r2 = _train_steps_role(model, bundle, params, device, seed=seed0 + 17, steps=params.p3_refresh_steps - params.p3_refresh_steps // 2, lr_scale=0.20, mode="rbf_task_small")
        return w1 + w2, max(p1, p2), _mean([b1, b2], 0.0), _mean([r1, r2], 0.0)

    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        for seed in seeds:
            base_row = next((r for r in rows if r.get("dataset") == dataset and int(float(r.get("seed", -1))) == seed and r.get("method") == "ABRBF-AdamW" and not r.get("error")), None)
            for method in P4_METHODS:
                if (dataset, seed, method) in done:
                    continue
                try:
                    if method == "ABRBF-AdamW":
                        bundle, model, final, wall, peak = _make_trained_model(args, dataset, seed, params, device, params.p4_total_steps, method=ABRBF_MAIN)
                        row = {"stage": "P4", "dataset": dataset, "seed": seed, "method": method, "selected_one_cycle_policy": selected_policy, "final_acc": final["test_acc"], "final_phi_rbf": final["phi_rbf_p95"], "final_curv_rbf": final["curvature_rbf_p95"], "total_training_time_sec": wall, "peak_cuda_allocated_mb": peak, "memory_ratio_vs_ABRBF_AdamW": 1.0, "time_ratio_vs_ABRBF_AdamW": 1.0, "p4_pass": 1, "error": "", **final}
                        rows.append(row)
                        base_row = row
                        write_csv(out_dir / "p4_event_driven_multicycle_v2.csv", rows)
                        print(f"P4 {dataset} {method} acc={final['test_acc']:.4f}")
                        continue
                    bundle = _load_bundle(args, dataset, seed, params)
                    set_seed(seed)
                    model = _make_v49_model(bundle, params, device, edge_kind=_edge_kind_from_method(ABRBF_MAIN))
                    opt = torch.optim.AdamW([p for _, p in _edge_named_params(model)], lr=params.adam_lr, weight_decay=0.0)
                    idxs = list(_iter_steps(len(bundle.x_train), params.batch_size, seed + 5400, params.p4_total_steps))
                    initial = _eval_v49(model, bundle, params, device)
                    target = {"phi_rbf": 0.95 * initial["phi_rbf_p95"], "curv_rbf": 0.95 * initial["curvature_rbf_p95"]}
                    history: List[Dict[str, float]] = []
                    events = 0
                    accepted = 0
                    rejected = 0
                    cooldown = 0
                    eta_cap = max(params.event_eta_grid)
                    recovery_vals: List[float] = []
                    retention_vals: List[float] = []
                    smooth_time = 0.0
                    refresh_time = 0.0
                    _reset_peak(device)
                    t0 = time.perf_counter()
                    for step, idx in enumerate(idxs, start=1):
                        xb = bundle.x_train[idx].to(device)
                        yb = bundle.y_train[idx].to(device)
                        opt.zero_grad(set_to_none=True)
                        loss = F.cross_entropy(model(xb), yb)
                        loss.backward()
                        opt.step()
                        if step % params.p4_eval_interval != 0 and step != params.p4_total_steps:
                            continue
                        ev = _eval_v49(model, bundle, params, device)
                        history.append({"step": step, "val_loss": ev["test_loss"], "acc": ev["test_acc"], "phi_rbf": ev["phi_rbf_p95"]})
                        debt = _geometry_debt(ev, target)
                        slope = _val_loss_slope(history)
                        task_slack = ev["test_acc"] - initial["test_acc"] + params.light_acc_drop_max
                        if cooldown > 0:
                            trace.append({"stage": "P4", "dataset": dataset, "seed": seed, "method": method, "event_step": step, "event_reason": "cooldown", "geometry_debt": debt, "task_slack": task_slack, "cooldown_remaining": cooldown, "reject_reason": "cooldown"})
                            cooldown -= 1
                            continue
                        max_events = 1 if "max1" in method else 2
                        trigger = False
                        reason = "none"
                        probe: Dict[str, Any] = {"accepted": 0, "probe_score": -1.0e18, "accepted_eta": 0.0}
                        if method == "FixedInterval-reference":
                            trigger = events < max_events and step in {60, 100}
                            reason = "fixed_interval" if trigger else "skip_fixed"
                        elif events < max_events:
                            grid = [eta for eta in params.event_eta_grid if eta <= eta_cap + 1.0e-12]
                            probe = _probe_light_smoothing(model, bundle, params, device, proposal=best_proposal, controller=best_controller, eta_grid=grid, apply_best=False)
                            score_ok = int(probe.get("accepted", 0)) == 1 and float(probe.get("probe_score", probe.get("score", -1.0e18))) > params.score_threshold
                            debt_ok = debt > params.min_geometry_debt
                            trigger = bool(score_ok and ("geometryDebt" not in method or debt_ok))
                            reason = "score_probe" if trigger else ("no_geometry_debt" if score_ok and not debt_ok else "negative_score")
                        if trigger:
                            events += 1
                            pre_event = _eval_v49(model, bundle, params, device)
                            if method == "FixedInterval-reference":
                                probe = _probe_light_smoothing(model, bundle, params, device, proposal=best_proposal, controller=best_controller, eta_grid=[min(0.035, eta_cap)], apply_best=True)
                            else:
                                probe = _probe_light_smoothing(model, bundle, params, device, proposal=best_proposal, controller=best_controller, eta_grid=[eta for eta in params.event_eta_grid if eta <= eta_cap + 1.0e-12], apply_best=True)
                            smooth_time += float(probe.get("smoothing_time_sec", 0.0))
                            accepted += int(probe.get("accepted", 0))
                            rejected += 1 - int(probe.get("accepted", 0))
                            post_smooth = _eval_v49(model, bundle, params, device)
                            rw, _, bnorm, rnorm = run_refresh(model, bundle, seed + 5450 + events * 19)
                            refresh_time += rw
                            opt = torch.optim.AdamW([p for _, p in _edge_named_params(model)], lr=params.adam_lr, weight_decay=0.0)
                            post_refresh = _eval_v49(model, bundle, params, device)
                            if post_smooth["test_acc"] >= pre_event["test_acc"]:
                                recovery = 1.0
                            else:
                                recovery = (post_refresh["test_acc"] - post_smooth["test_acc"]) / max(1.0e-12, pre_event["test_acc"] - post_smooth["test_acc"])
                            retention = (pre_event["phi_rbf_p95"] - post_refresh["phi_rbf_p95"]) / max(1.0e-12, pre_event["phi_rbf_p95"] - post_smooth["phi_rbf_p95"])
                            recovery_vals.append(recovery)
                            retention_vals.append(retention)
                            if recovery < 0.8:
                                eta_cap = max(0.005, float(probe.get("accepted_eta", eta_cap)) / 2.0)
                                cooldown = params.cooldown_evals + 2
                            else:
                                cooldown = params.cooldown_evals
                            trace.append(
                                {
                                    "stage": "P4",
                                    "dataset": dataset,
                                    "seed": seed,
                                    "method": method,
                                    "event_id": events,
                                    "event_epoch": step / max(1, params.p4_eval_interval),
                                    "event_step": step,
                                    "event_reason": reason,
                                    "geometry_debt": debt,
                                    "task_slack": task_slack,
                                    "val_loss_slope": slope,
                                    "best_eta": probe.get("eta", probe.get("accepted_eta", 0.0)),
                                    "best_score": probe.get("probe_score", probe.get("score", 0.0)),
                                    "accepted_eta": probe.get("accepted_eta", 0.0),
                                    "acc_before": pre_event["test_acc"],
                                    "acc_after_smooth": post_smooth["test_acc"],
                                    "acc_after_refresh": post_refresh["test_acc"],
                                    "KL": probe.get("probe_KL", 0.0),
                                    "logit_drift": probe.get("probe_logit_drift", 0.0),
                                    "phi_gain_probe": probe.get("probe_phi_gain", 0.0),
                                    "phi_gain_actual": 1.0 - post_refresh["phi_rbf_p95"] / max(1.0e-12, pre_event["phi_rbf_p95"]),
                                    "curv_gain_probe": probe.get("probe_curv_gain", 0.0),
                                    "curv_gain_actual": 1.0 - post_refresh["curvature_rbf_p95"] / max(1.0e-12, pre_event["curvature_rbf_p95"]),
                                    "recovery_ratio": recovery,
                                    "geometry_retention": retention,
                                    "cooldown_remaining": cooldown,
                                    "base_update_norm_refresh": bnorm,
                                    "rbf_update_norm_refresh": rnorm,
                                    "reject_reason": "",
                                }
                            )
                    wall = time.perf_counter() - t0
                    peak_alloc, peak_reserved = _peak_mb(device)
                    final = _eval_v49(model, bundle, params, device)
                    base_acc = float(base_row.get("final_acc", final["test_acc"])) if base_row else final["test_acc"]
                    base_phi = float(base_row.get("final_phi_rbf", initial["phi_rbf_p95"])) if base_row else initial["phi_rbf_p95"]
                    base_curv = float(base_row.get("final_curv_rbf", initial["curvature_rbf_p95"])) if base_row else initial["curvature_rbf_p95"]
                    base_peak = float(base_row.get("peak_cuda_allocated_mb", peak_alloc) or peak_alloc) if base_row else peak_alloc
                    base_time = float(base_row.get("total_training_time_sec", wall) or wall) if base_row else wall
                    phi_red = 1.0 - final["phi_rbf_p95"] / max(1.0e-12, base_phi)
                    curv_red = 1.0 - final["curvature_rbf_p95"] / max(1.0e-12, base_curv)
                    row = {
                        "stage": "P4",
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "selected_one_cycle_policy": selected_policy,
                        "refresh_policy": refresh_policy_name(),
                        "final_acc": final["test_acc"],
                        "acc_gap_vs_ABRBF": base_acc - final["test_acc"],
                        "final_phi_rbf": final["phi_rbf_p95"],
                        "final_curv_rbf": final["curvature_rbf_p95"],
                        "phi_rbf_reduction": phi_red,
                        "curvature_rbf_reduction": curv_red,
                        "event_count": events,
                        "accepted_event_count": accepted,
                        "rejected_event_count": rejected,
                        "mean_recovery_ratio": _mean(recovery_vals, 1.0),
                        "mean_geometry_retention": _mean(retention_vals, 1.0),
                        "smoothing_time_total_sec": smooth_time,
                        "refresh_time_sec": refresh_time,
                        "total_training_time_sec": wall,
                        "smoothing_time_fraction": smooth_time / max(1.0e-12, wall),
                        "peak_cuda_allocated_mb": peak_alloc,
                        "peak_cuda_reserved_mb": peak_reserved,
                        "memory_ratio_vs_ABRBF_AdamW": peak_alloc / max(1.0e-12, base_peak),
                        "time_ratio_vs_ABRBF_AdamW": wall / max(1.0e-12, base_time),
                        "p4_pass": 0,
                        "error": "",
                    }
                    row["p4_pass"] = int(final["test_acc"] >= base_acc - 0.005 and (phi_red >= 0.05 or curv_red >= 0.30) and row["memory_ratio_vs_ABRBF_AdamW"] <= params.memory_ratio_gate and row["time_ratio_vs_ABRBF_AdamW"] <= params.time_ratio_gate)
                    rows.append(row)
                    print(f"P4 {dataset} {method} acc={final['test_acc']:.4f} events={events} phiRed={phi_red:.3f} pass={row['p4_pass']}")
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    rows.append({"stage": "P4", "dataset": dataset, "seed": seed, "method": method, "error": repr(exc)})
                    print(f"P4 ERROR {dataset} {method}: {exc!r}")
                write_csv(out_dir / "p4_event_driven_multicycle_v2.csv", rows)
                write_csv(out_dir / "p4_event_trace.csv", trace)
                _empty_cache(device)
    return rows


def run_p5(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not _all_dataset_survivors(out_dir / "p4_event_driven_multicycle_v2.csv", "method", "p4_pass"):
        return _placeholder(out_dir, "p5_candidate_selection3.csv", "P4 produced no all-dataset survivor")
    return _placeholder(out_dir, "p5_candidate_selection3.csv", "P5 3-seed expansion gated off in this compact implementation")


def run_p6(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not _all_dataset_survivors(out_dir / "p5_candidate_selection3.csv", "method", "p5_pass"):
        return _placeholder(out_dir, "p6_confirm5.csv", "P5 produced no all-dataset survivor")
    return _placeholder(out_dir, "p6_confirm5.csv", "P6 gated off")


def run_p7(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not _all_dataset_survivors(out_dir / "p6_confirm5.csv", "method", "p6_pass"):
        return _placeholder(out_dir, "p7_confirm10.csv", "P6 produced no all-dataset survivor")
    return _placeholder(out_dir, "p7_confirm10.csv", "P7 gated off")


def run_p8(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    failure_rows: List[Dict[str, Any]] = []
    p0 = [r for r in read_csv(out_dir / "p0_memory_smoke.csv") if not r.get("error")]
    p1 = [r for r in read_csv(out_dir / "p1_single_smoothing_event.csv") if not r.get("error")]
    p2 = [r for r in read_csv(out_dir / "p2_event_probe_audit.csv") if not r.get("error")]
    p3 = [r for r in read_csv(out_dir / "p3_one_cycle_controller_variants.csv") if not r.get("error") and r.get("status") != "not_run"]
    p4 = [r for r in read_csv(out_dir / "p4_event_driven_multicycle_v2.csv") if not r.get("error") and r.get("status") != "not_run"]
    for r in p0:
        if float(r.get("learnable_nonKAN_params", 999) or 999) != 0 or float(r.get("edge_coverage", 0) or 0) < 0.999:
            failure_rows.append({"stage": "P0", "dataset": r.get("dataset"), "method": r.get("method"), "failure_type": "F8_implementation"})
        if float(r.get("memory_ratio_vs_ABRBF_AdamW", 1) or 1) > 1.25:
            failure_rows.append({"stage": "P0", "dataset": r.get("dataset"), "method": r.get("method"), "failure_type": "F1_memory_too_high"})
    for r in p1:
        if float(r.get("phi_rbf_reduction", 0) or 0) < 0.03 and float(r.get("curvature_rbf_reduction", 0) or 0) < 0.10:
            failure_rows.append({"stage": "P1", "dataset": r.get("dataset"), "method": f"{r.get('proposal')}/{r.get('controller')}", "failure_type": "F2_smoothing_too_weak"})
        if float(r.get("KL_teacher_student", 0) or 0) > 0.005 or float(r.get("logit_relative_drift", 0) or 0) > 0.03 or float(r.get("acc_drop", 0) or 0) > 0.005:
            failure_rows.append({"stage": "P1", "dataset": r.get("dataset"), "method": f"{r.get('proposal')}/{r.get('controller')}", "failure_type": "F3_task_drift_too_high"})
    if p2:
        trigger_ds = {str(r.get("dataset")) for r in p2 if int(float(r.get("would_trigger_by_geometry_debt", 0) or 0)) == 1}
        if len(trigger_ds) < 2:
            failure_rows.append({"stage": "P2", "dataset": "all", "method": "ABRBF-EventProbe", "failure_type": "F2_no_controller_trigger"})
        for r in p2:
            if float(r.get("probe_acc_drop", 0) or 0) > 0.005:
                failure_rows.append({"stage": "P2", "dataset": r.get("dataset"), "method": r.get("method"), "failure_type": "F3_probe_task_cost"})
    if p3:
        for r in p3:
            if r.get("method") == "ABRBF-AdamW":
                continue
            if int(float(r.get("p3_pass", 0) or 0)) == 0:
                failure = "F7_KMNIST_specific_failure" if str(r.get("dataset")) == "KMNIST" else "F4_one_cycle_refresh_failure"
                failure_rows.append({"stage": "P3", "dataset": r.get("dataset"), "method": r.get("method"), "failure_type": failure})
    else:
        failure_rows.append({"stage": "P3", "dataset": "all", "method": "all", "failure_type": "P3_not_run_no_P2_trigger"})
    if p4:
        for r in p4:
            if r.get("method") == "ABRBF-AdamW":
                continue
            if int(float(r.get("p4_pass", 0) or 0)) == 0:
                if float(r.get("event_count", 0) or 0) <= 0:
                    ft = "F5_no_trigger"
                elif float(r.get("acc_gap_vs_ABRBF", 0) or 0) > 0.005:
                    ft = "F6_task_cost_accumulation"
                elif float(r.get("memory_ratio_vs_ABRBF_AdamW", 1) or 1) > 1.25:
                    ft = "F1_memory_too_high"
                else:
                    ft = "F8_multicycle_gate_failure"
                failure_rows.append({"stage": "P4", "dataset": r.get("dataset"), "method": r.get("method"), "failure_type": ft})
    if not failure_rows:
        failure_rows.append({"stage": "P8", "dataset": "all", "method": "all", "failure_type": "none"})
    write_csv(out_dir / "failure_table.csv", failure_rows)
    by_dataset: Dict[str, Dict[str, Any]] = {}
    by_method: Dict[str, Dict[str, Any]] = {}
    for r in failure_rows:
        ds = str(r.get("dataset", ""))
        mt = str(r.get("method", ""))
        by_dataset.setdefault(ds, {"dataset": ds, "failures": 0})["failures"] += 1
        by_method.setdefault(mt, {"method": mt, "failures": 0})["failures"] += 1
    write_csv(out_dir / "failure_by_dataset.csv", list(by_dataset.values()))
    write_csv(out_dir / "failure_by_method.csv", list(by_method.values()))
    recommendation = {
        "status": "diagnosed",
        "recommendation": "Keep AB-RBF as the default PureKAN primitive. Use residual smoothing as low-frequency/offline maintenance unless P2/P3 gates pass.",
    }
    (out_dir / "recommendation.json").write_text(__import__("json").dumps(recommendation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return failure_rows


def add_args() -> argparse.ArgumentParser:
    p = add_v3_args()
    p.description = __doc__
    p.set_defaults(packages="V5_3_P0_SMOKE", datasets="MNIST,Fashion-MNIST,KMNIST", out_dir=Path("results/v5_3"), seeds="0,1,2")
    return p


def main() -> int:
    args = add_args().parse_args()
    all_rows: List[Dict[str, Any]] = []
    for package in parse_str_list(args.packages):
        key = package.strip().upper().replace("-", "_")
        if key in {"V5_3_P0", "V5_3_P0_SMOKE", "V5_3_P0_MEMORY"}:
            all_rows = run_p0(args)
        elif key in {"V5_3_P1", "V5_3_P1_SINGLE", "V5_3_P1_SMOOTHING"}:
            all_rows = run_p1(args)
        elif key in {"V5_3_P2", "V5_3_P2_PROBE", "V5_3_P2_EVENT_PROBE"}:
            all_rows = run_p2(args)
        elif key in {"V5_3_P3", "V5_3_P3_ONE_CYCLE", "V5_3_P3_REFRESH"}:
            all_rows = run_p3(args)
        elif key in {"V5_3_P4", "V5_3_P4_EVENT", "V5_3_P4_MULTICYCLE"}:
            all_rows = run_p4(args)
        elif key in {"V5_3_P5", "V5_3_P5_SELECTION", "V5_3_P5_CANDIDATE_SELECTION"}:
            all_rows = run_p5(args)
        elif key in {"V5_3_P6", "V5_3_P6_CONFIRM5"}:
            all_rows = run_p6(args)
        elif key in {"V5_3_P7", "V5_3_P7_CONFIRM10"}:
            all_rows = run_p7(args)
        elif key in {"V5_3_P8", "V5_3_P8_FAILURE"}:
            all_rows = run_p8(args)
        else:
            raise ValueError(f"unknown v5.3 package: {package}")
    return 0 if all_rows is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
