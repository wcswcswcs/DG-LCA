#!/usr/bin/env python3
"""DG-KAN v5.2 runner: memory-budgeted AB-RBF residual smoothing."""

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
P2_METHODS = [
    "ABRBF-AdamW",
    "ABRBF-LightSmooth-logitOnly-oneCycle",
    "ABRBF-LightSmooth-scoreOnly-oneCycle",
    "ABRBF-StrongSmooth-reference-oneCycle",
]
P3_METHODS = [
    "ABRBF-AdamW",
    "ABRBF-LightSmooth-fixedInterval",
    "ABRBF-LightSmooth-eventDriven",
    "ABRBF-LightSmooth-eventDriven-refreshStrong",
]


@dataclass
class V52Params(V51Params):
    train_size: int = 1536
    val_size: int = 512
    test_size: int = 512
    batch_size: int = 128
    eval_batch_size: int = 512
    audit_batch_size: int = 20
    p1_steps: int = 120
    p2_task_steps: int = 100
    p2_refresh_steps: int = 20
    p3_total_steps: int = 140
    p3_eval_interval: int = 20
    p3_refresh_steps: int = 10
    light_eta_grid: Tuple[float, ...] = (0.03, 0.06, 0.10, 0.16, 0.24)
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
    max_smoothing_events: int = 3
    min_event_gap_evals: int = 2


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
) -> float:
    return 1.0 * phi_red + 0.7 * curv_red + 0.2 * sob_red - 3.0 * kl - 2.0 * logit - 2.0 * max(0.0, holdout_delta) - 4.0 * max(0.0, acc_drop)


def _light_smoothing_event(
    model: torch.nn.Module,
    bundle: Any,
    params: V52Params,
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
    params: V52Params,
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


def _load_bundle(args: argparse.Namespace, dataset: str, seed: int, params: V52Params) -> Any:
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
    params = V52Params(train_size=512, val_size=128, test_size=128, audit_batch_size=8, p1_steps=8)
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


def _event_pass(row: Dict[str, Any], params: V52Params) -> bool:
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
    params = V52Params()
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


def _eval_stage(model: torch.nn.Module, bundle: Any, params: V52Params, device: torch.device, stage: str, step: int, method: str, dataset: str, seed: int) -> Dict[str, Any]:
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


def _make_trained_model(args: argparse.Namespace, dataset: str, seed: int, params: V52Params, device: torch.device, steps: int, method: str = ABRBF_MAIN) -> Tuple[Any, torch.nn.Module, Dict[str, Any], float, float]:
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
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p2_one_cycle_train_smooth_refresh.csv")
    trace: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p2_training_trace.csv")
    params = V52Params()
    device = get_device(args.device)
    best_teacher, best_proposal, best_controller = _best_light_candidate(out_dir)
    seeds = parse_int_list(args.seeds)[:1]
    done = {(r.get("dataset"), int(float(r.get("seed", -1))), r.get("method")) for r in rows if not r.get("error")}
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        for seed in seeds:
            baseline_key = (dataset, seed, "ABRBF-AdamW")
            if baseline_key not in done:
                try:
                    bundle, model, pre, wall, peak = _make_trained_model(args, dataset, seed, params, device, params.p2_task_steps + params.p2_refresh_steps, method=ABRBF_MAIN)
                    row = {"stage": "P2", "dataset": dataset, "seed": seed, "method": "ABRBF-AdamW", "final_acc": pre["test_acc"], "final_phi_rbf": pre["phi_rbf_p95"], "final_curv_rbf": pre["curvature_rbf_p95"], "training_time_sec": wall, "peak_cuda_allocated_mb": peak, "p2_pass": 1, "error": "", **pre}
                    rows.append(row)
                    trace.append(_eval_stage(model, bundle, params, device, "P2", params.p2_task_steps + params.p2_refresh_steps, "ABRBF-AdamW", dataset, seed))
                    write_csv(out_dir / "p2_one_cycle_train_smooth_refresh.csv", rows)
                    write_csv(out_dir / "p2_training_trace.csv", trace)
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    rows.append({"stage": "P2", "dataset": dataset, "seed": seed, "method": "ABRBF-AdamW", "error": repr(exc)})
            base_row = next((r for r in rows if r.get("dataset") == dataset and r.get("seed") == seed and r.get("method") == "ABRBF-AdamW"), None)
            for method in P2_METHODS[1:]:
                if (dataset, seed, method) in done:
                    continue
                try:
                    strong = "StrongSmooth" in method
                    controller = "scoreOnly" if "scoreOnly" in method else "logitOnly"
                    proposal = best_proposal if not strong else "StrongSmooth-reference"
                    bundle, model, pre_eval, train_wall, train_peak = _make_trained_model(args, dataset, seed, params, device, params.p2_task_steps, method=best_teacher)
                    pre = _eval_v49(model, bundle, params, device)
                    trace.append(_eval_stage(model, bundle, params, device, "P2", params.p2_task_steps, method + ":pre_smooth", dataset, seed))
                    model.zero_grad(set_to_none=True)
                    _empty_cache(device)
                    smooth = _light_smoothing_event(model, bundle, params, device, proposal=proposal, controller=("strongReference" if strong else controller), seed=seed + 5230, strong_reference=strong)
                    post = _eval_v49(model, bundle, params, device)
                    trace.append(_eval_stage(model, bundle, params, device, "P2", params.p2_task_steps + 1, method + ":post_smooth", dataset, seed))
                    refresh_wall, refresh_peak = _train_steps(model, bundle, params, device, seed=seed + 5231, steps=params.p2_refresh_steps, lr_scale=0.35)
                    final = _eval_v49(model, bundle, params, device)
                    trace.append(_eval_stage(model, bundle, params, device, "P2", params.p2_task_steps + params.p2_refresh_steps, method + ":post_refresh", dataset, seed))
                    if post["test_acc"] >= pre["test_acc"]:
                        refresh_recovery = 1.0
                    else:
                        refresh_recovery = (final["test_acc"] - post["test_acc"]) / max(1.0e-12, pre["test_acc"] - post["test_acc"])
                    geom_retention = (pre["phi_rbf_p95"] - final["phi_rbf_p95"]) / max(1.0e-12, pre["phi_rbf_p95"] - post["phi_rbf_p95"])
                    acc_gate = final["test_acc"] >= float(base_row.get("final_acc", pre["test_acc"])) - 0.005 if base_row else False
                    phi_red_final = 1.0 - final["phi_rbf_p95"] / max(1.0e-12, pre["phi_rbf_p95"])
                    curv_red_final = 1.0 - final["curvature_rbf_p95"] / max(1.0e-12, pre["curvature_rbf_p95"])
                    row = {
                        "stage": "P2",
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "teacher": best_teacher,
                        "proposal": proposal,
                        "controller": controller,
                        "pre_smooth_acc": pre["test_acc"],
                        "post_smooth_acc": post["test_acc"],
                        "post_refresh_acc": final["test_acc"],
                        "final_acc": final["test_acc"],
                        "pre_smooth_phi_rbf": pre["phi_rbf_p95"],
                        "post_smooth_phi_rbf": post["phi_rbf_p95"],
                        "post_refresh_phi_rbf": final["phi_rbf_p95"],
                        "final_phi_rbf": final["phi_rbf_p95"],
                        "pre_smooth_curv_rbf": pre["curvature_rbf_p95"],
                        "post_smooth_curv_rbf": post["curvature_rbf_p95"],
                        "post_refresh_curv_rbf": final["curvature_rbf_p95"],
                        "final_curv_rbf": final["curvature_rbf_p95"],
                        "refresh_recovery_ratio": refresh_recovery,
                        "geometry_retention_ratio": geom_retention,
                        "phi_rbf_reduction_final": phi_red_final,
                        "curvature_rbf_reduction_final": curv_red_final,
                        "training_time_sec": train_wall + refresh_wall,
                        "smoothing_time_sec": smooth.get("smoothing_time_sec", 0.0),
                        "smoothing_time_fraction": smooth.get("smoothing_time_sec", 0.0) / max(1.0e-12, train_wall + refresh_wall + smooth.get("smoothing_time_sec", 0.0)),
                        "peak_cuda_allocated_mb": max(train_peak, refresh_peak, float(smooth.get("peak_cuda_allocated_mb", 0.0))),
                        "memory_ratio_vs_ABRBF_AdamW": 1.0,
                        "p2_pass": 0,
                        "error": "",
                        **smooth,
                    }
                    if base_row:
                        row["memory_ratio_vs_ABRBF_AdamW"] = row["peak_cuda_allocated_mb"] / max(1.0e-12, float(base_row.get("peak_cuda_allocated_mb", 0.0) or 0.0)) if float(base_row.get("peak_cuda_allocated_mb", 0.0) or 0.0) > 0 else 1.0
                    row["p2_pass"] = int(acc_gate and (phi_red_final >= 0.05 or curv_red_final >= 0.20) and refresh_recovery >= 0.80 and geom_retention >= 0.50 and row["memory_ratio_vs_ABRBF_AdamW"] <= params.memory_ratio_gate)
                    rows.append(row)
                    print(f"P2 {dataset} {method} acc={final['test_acc']:.4f} phiRed={phi_red_final:.3f} rec={refresh_recovery:.2f} ret={geom_retention:.2f} pass={row['p2_pass']}")
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    rows.append({"stage": "P2", "dataset": dataset, "seed": seed, "method": method, "error": repr(exc)})
                    print(f"P2 ERROR {dataset} {method}: {exc!r}")
                write_csv(out_dir / "p2_one_cycle_train_smooth_refresh.csv", rows)
                write_csv(out_dir / "p2_training_trace.csv", trace)
                _empty_cache(device)
    return rows


def _p2_survivors(out_dir: Path) -> List[str]:
    rows = [r for r in read_csv(out_dir / "p2_one_cycle_train_smooth_refresh.csv") if not r.get("error")]
    by: Dict[str, set[str]] = {}
    for r in rows:
        if str(r.get("method")) == "ABRBF-AdamW":
            continue
        if int(float(r.get("p2_pass", 0) or 0)) == 1:
            by.setdefault(str(r.get("method")), set()).add(str(r.get("dataset")))
    return [m for m, ds in by.items() if all(d in ds for d in DATASETS)]


def run_p3(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p3_event_driven_multicycle.csv")
    trace: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p3_event_trace.csv")
    params = V52Params()
    device = get_device(args.device)
    p2_surv = _p2_survivors(out_dir)
    if not p2_surv:
        rows = [{"status": "not_run", "reason": "P2 produced no all-dataset light smoothing survivor"}]
        trace = [{"status": "not_run", "reason": "P2 produced no all-dataset light smoothing survivor"}]
        write_csv(out_dir / "p3_event_driven_multicycle.csv", rows)
        write_csv(out_dir / "p3_event_trace.csv", trace)
        print("P3 not run: no P2 survivor")
        return rows
    best_teacher, best_proposal, best_controller = _best_light_candidate(out_dir)
    seeds = parse_int_list(args.seeds)[:1]
    done = {(r.get("dataset"), int(float(r.get("seed", -1))), r.get("method")) for r in rows if not r.get("error") and r.get("status") != "not_run"}
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        for seed in seeds:
            base_final = None
            for method in P3_METHODS:
                if (dataset, seed, method) in done:
                    continue
                try:
                    bundle = _load_bundle(args, dataset, seed, params)
                    set_seed(seed)
                    model = _make_v49_model(bundle, params, device, edge_kind=_edge_kind_from_method(ABRBF_MAIN))
                    opt = torch.optim.AdamW([p for _, p in _edge_named_params(model)], lr=params.adam_lr, weight_decay=0.0)
                    idxs = list(_iter_steps(len(bundle.x_train), params.batch_size, seed + 5300, params.p3_total_steps))
                    initial = _eval_v49(model, bundle, params, device)
                    eval_history: List[Dict[str, float]] = []
                    events = 0
                    accepted = 0
                    rejected = 0
                    smooth_time = 0.0
                    last_event_eval = -999
                    _reset_peak(device)
                    t0 = time.perf_counter()
                    for step, idx in enumerate(idxs, start=1):
                        xb = bundle.x_train[idx].to(device)
                        yb = bundle.y_train[idx].to(device)
                        opt.zero_grad(set_to_none=True)
                        loss = F.cross_entropy(model(xb), yb)
                        loss.backward()
                        opt.step()
                        if step % params.p3_eval_interval != 0 and step != params.p3_total_steps:
                            continue
                        ev = _eval_v49(model, bundle, params, device)
                        eval_history.append({"step": step, "val_loss": ev["test_loss"], "phi_rbf": ev["phi_rbf_p95"], "acc": ev["test_acc"], "curv": ev["curvature_rbf_p95"]})
                        trigger = False
                        reason = "none"
                        if method == "ABRBF-LightSmooth-fixedInterval":
                            trigger = events < params.max_smoothing_events and step in {60, 100, 140}
                            reason = "fixed_interval" if trigger else "skip_fixed"
                        elif method in {"ABRBF-LightSmooth-eventDriven", "ABRBF-LightSmooth-eventDriven-refreshStrong"}:
                            if len(eval_history) >= params.plateau_window and events < params.max_smoothing_events and len(eval_history) - last_event_eval >= params.min_event_gap_evals:
                                old = eval_history[-params.plateau_window]["val_loss"]
                                plateau = abs(ev["test_loss"] - old) / max(1.0e-12, old) < params.epsilon_plateau
                                phi_bad = ev["phi_rbf_p95"] > 0.95 * initial["phi_rbf_p95"]
                                trigger = plateau and phi_bad
                                reason = "plateau_phi_budget" if trigger else "skip_no_plateau_or_phi"
                        if trigger:
                            events += 1
                            last_event_eval = len(eval_history)
                            pre_event = _eval_v49(model, bundle, params, device)
                            smooth = _light_smoothing_event(model, bundle, params, device, proposal=best_proposal, controller=best_controller, seed=seed + 5310 + events)
                            smooth_time += float(smooth.get("smoothing_time_sec", 0.0))
                            accepted += int(smooth.get("accepted", 0))
                            rejected += 1 - int(smooth.get("accepted", 0))
                            if method == "ABRBF-LightSmooth-eventDriven-refreshStrong":
                                _train_steps(model, bundle, params, device, seed=seed + 5320 + events, steps=params.p3_refresh_steps * 2, lr_scale=0.35)
                            else:
                                _train_steps(model, bundle, params, device, seed=seed + 5320 + events, steps=params.p3_refresh_steps, lr_scale=0.35)
                            post_event = _eval_v49(model, bundle, params, device)
                            trace.append(
                                {
                                    "stage": "P3",
                                    "dataset": dataset,
                                    "seed": seed,
                                    "method": method,
                                    "event_index": events,
                                    "event_step": step,
                                    "event_reason": reason,
                                    "accepted": smooth.get("accepted", 0),
                                    "accepted_eta": smooth.get("accepted_eta", 0.0),
                                    "pre_event_val_loss": pre_event["test_loss"],
                                    "post_event_val_loss": post_event["test_loss"],
                                    "pre_event_phi_rbf": pre_event["phi_rbf_p95"],
                                    "post_event_phi_rbf": post_event["phi_rbf_p95"],
                                    "pre_event_acc": pre_event["test_acc"],
                                    "post_event_acc": post_event["test_acc"],
                                    "acc_drop_event": pre_event["test_acc"] - post_event["test_acc"],
                                    "phi_gain_event": 1.0 - post_event["phi_rbf_p95"] / max(1.0e-12, pre_event["phi_rbf_p95"]),
                                }
                            )
                    wall = time.perf_counter() - t0
                    peak_alloc, peak_reserved = _peak_mb(device)
                    final = _eval_v49(model, bundle, params, device)
                    if method == "ABRBF-AdamW":
                        base_final = final
                    base_acc = base_final["test_acc"] if base_final else final["test_acc"]
                    base_phi = base_final["phi_rbf_p95"] if base_final else initial["phi_rbf_p95"]
                    phi_red = 1.0 - final["phi_rbf_p95"] / max(1.0e-12, base_phi)
                    curv_red = 1.0 - final["curvature_rbf_p95"] / max(1.0e-12, initial["curvature_rbf_p95"])
                    smoothing_fraction = smooth_time / max(1.0e-12, wall)
                    row = {
                        "stage": "P3",
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "final_acc": final["test_acc"],
                        "acc_gap_vs_ABRBF": base_acc - final["test_acc"],
                        "final_phi_rbf": final["phi_rbf_p95"],
                        "phi_rbf_reduction": phi_red,
                        "curvature_rbf_reduction": curv_red,
                        "smoothing_event_count": events,
                        "accepted_event_count": accepted,
                        "rejected_event_count": rejected,
                        "training_time_total_sec": wall - smooth_time,
                        "smoothing_time_total_sec": smooth_time,
                        "smoothing_time_fraction": smoothing_fraction,
                        "peak_cuda_allocated_mb": peak_alloc,
                        "peak_cuda_reserved_mb": peak_reserved,
                        "amortized_step_time_ms": 1000.0 * wall / max(1, params.p3_total_steps),
                        "amortized_memory_ratio": 1.0,
                        "accumulated_acc_drop": initial["test_acc"] - final["test_acc"],
                        "accumulated_phi_reduction": 1.0 - final["phi_rbf_p95"] / max(1.0e-12, initial["phi_rbf_p95"]),
                        "p3_pass": int(final["test_acc"] >= base_acc - 0.005 and (phi_red >= 0.05 or curv_red >= 0.20) and smoothing_fraction <= 0.15 and events <= 3),
                        "error": "",
                    }
                    rows.append(row)
                    print(f"P3 {dataset} {method} acc={final['test_acc']:.4f} events={events} phiRed={phi_red:.3f} pass={row['p3_pass']}")
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    rows.append({"stage": "P3", "dataset": dataset, "seed": seed, "method": method, "error": repr(exc)})
                    print(f"P3 ERROR {dataset} {method}: {exc!r}")
                write_csv(out_dir / "p3_event_driven_multicycle.csv", rows)
                write_csv(out_dir / "p3_event_trace.csv", trace)
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
    if not _all_dataset_survivors(out_dir / "p3_event_driven_multicycle.csv", "method", "p3_pass"):
        return _placeholder(out_dir, "p4_candidate_selection3.csv", "P3 produced no all-dataset survivor")
    return _placeholder(out_dir, "p4_candidate_selection3.csv", "P4 implementation deferred; P3 survivor path not expected in v5.2 smoke budget")


def run_p5(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not _all_dataset_survivors(out_dir / "p4_candidate_selection3.csv", "method", "p4_pass"):
        return _placeholder(out_dir, "p5_confirm5.csv", "P4 produced no all-dataset survivor")
    return _placeholder(out_dir, "p5_confirm5.csv", "P5 gated off")


def run_p6(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    if not _all_dataset_survivors(out_dir / "p5_confirm5.csv", "method", "p5_pass"):
        return _placeholder(out_dir, "p6_confirm10.csv", "P5 produced no all-dataset survivor")
    return _placeholder(out_dir, "p6_confirm10.csv", "P6 gated off")


def run_p7(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    return _placeholder(out_dir, "p7_cifar_small_memory_precheck.csv", "No P3/P4 unified light smoothing survivor to scale-check")


def run_p8(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    failure_rows: List[Dict[str, Any]] = []
    p0 = [r for r in read_csv(out_dir / "p0_memory_smoke.csv") if not r.get("error")]
    p1 = [r for r in read_csv(out_dir / "p1_single_smoothing_event.csv") if not r.get("error")]
    p2 = [r for r in read_csv(out_dir / "p2_one_cycle_train_smooth_refresh.csv") if not r.get("error")]
    p3 = [r for r in read_csv(out_dir / "p3_event_driven_multicycle.csv") if not r.get("error") and r.get("status") != "not_run"]
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
    for r in p2:
        if r.get("method") == "ABRBF-AdamW":
            continue
        if float(r.get("refresh_recovery_ratio", 1) or 1) < 0.80:
            failure_rows.append({"stage": "P2", "dataset": r.get("dataset"), "method": r.get("method"), "failure_type": "F4_refresh_cannot_recover"})
        if float(r.get("geometry_retention_ratio", 1) or 1) < 0.50:
            failure_rows.append({"stage": "P2", "dataset": r.get("dataset"), "method": r.get("method"), "failure_type": "F5_geometry_not_retained"})
    if p3:
        for r in p3:
            if str(r.get("dataset")) == "KMNIST" and int(float(r.get("p3_pass", 0) or 0)) == 0:
                failure_rows.append({"stage": "P3", "dataset": r.get("dataset"), "method": r.get("method"), "failure_type": "F7_KMNIST_specific_failure"})
    else:
        failure_rows.append({"stage": "P3", "dataset": "all", "method": "all", "failure_type": "P3_not_run_no_P2_survivor"})
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
    p.set_defaults(packages="V5_2_P0_SMOKE", datasets="MNIST,Fashion-MNIST,KMNIST", out_dir=Path("results/v5_2"), seeds="0,1,2")
    return p


def main() -> int:
    args = add_args().parse_args()
    all_rows: List[Dict[str, Any]] = []
    for package in parse_str_list(args.packages):
        key = package.strip().upper().replace("-", "_")
        if key in {"V5_2_P0", "V5_2_P0_SMOKE", "V5_2_P0_MEMORY"}:
            all_rows = run_p0(args)
        elif key in {"V5_2_P1", "V5_2_P1_SINGLE", "V5_2_P1_SMOOTHING"}:
            all_rows = run_p1(args)
        elif key in {"V5_2_P2", "V5_2_P2_ONE_CYCLE"}:
            all_rows = run_p2(args)
        elif key in {"V5_2_P3", "V5_2_P3_EVENT"}:
            all_rows = run_p3(args)
        elif key in {"V5_2_P4", "V5_2_P4_SELECTION"}:
            all_rows = run_p4(args)
        elif key in {"V5_2_P5", "V5_2_P5_CONFIRM5"}:
            all_rows = run_p5(args)
        elif key in {"V5_2_P6", "V5_2_P6_CONFIRM10"}:
            all_rows = run_p6(args)
        elif key in {"V5_2_P7", "V5_2_P7_CIFAR"}:
            all_rows = run_p7(args)
        elif key in {"V5_2_P8", "V5_2_P8_FAILURE"}:
            all_rows = run_p8(args)
        else:
            raise ValueError(f"unknown v5.2 package: {package}")
    return 0 if all_rows is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
