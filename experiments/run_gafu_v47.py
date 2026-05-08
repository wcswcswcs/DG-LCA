#!/usr/bin/env python3
"""DG-KAN v4.7 runner: Phase-Separated Functional Training probes."""

from __future__ import annotations

import argparse
import copy
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dgkan_core import (
    RuntimeState,
    _get_sobolev_gram,
    coefficient_named_params,
    ensure_dir,
    evaluate,
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
from run_gafu_v43 import (
    AdamState,
    V43Params,
    V43RunState,
    _adam_updates,
    _apply_updates,
    _feature_audit,
    _flat_updates,
    _mean,
    _method_updates as _v43_method_updates,
    _restore,
    _role_for_name,
    _role_group,
    _safe_cos,
    _snapshot,
)
from run_gafu_v44 import (
    FCAdamState,
    V44Params,
    V44RunState,
    _compute_loss_and_grad,
    _estimate_geometry_any,
    _fcadam_updates,
    _function_displacement,
    _make_model,
    _metric_matrix,
    _r2,
)


DATASETS = ["MNIST", "Fashion-MNIST", "KMNIST"]
P0_METHODS = [
    "PureKAN-AdamW",
    "D0-allFullSobolev",
    "FCAdam-dataSob",
    "FLD-AdanLite",
    "PSFT-A-FCAdamDistill-smoke",
    "PSFT-A-AdanDistill-smoke",
    "PSFT-B-AdamTeacherDistill-smoke",
    "PSFT-C-Alternating-smoke",
]

P1_METHODS = [
    "PureKAN-AdamW",
    "FCAdam-dataSob",
    "FLD-AdanLite",
    "FLD-WinLite",
    "FLD-LyapunovRestart",
    "PSFT-A-FCAdam-PhaseI-only",
    "PSFT-A-Adan-PhaseI-only",
    "PSFT-D-role-staged-PhaseI-only",
]

P2_TEACHERS = [
    ("TeacherA-AdamW100", "PureKAN-AdamW"),
    ("TeacherB-PSFT-FCAdam100", "PSFT-A-FCAdam-PhaseI-only"),
    ("TeacherC-FCAdam100", "FCAdam-dataSob"),
    ("TeacherD-FLD-Adan100", "FLD-AdanLite"),
]


@dataclass
class V47Params(V44Params):
    train_size: int = 1536
    val_size: int = 512
    test_size: int = 512
    batch_size: int = 128
    phase1_steps: int = 100
    consolidate_steps: int = 50
    phase3_steps: int = 50
    p1_steps: int = 100
    fc_lr: float = 8e-4
    fc_data_sob_alpha: float = 0.02
    fld_beta1: float = 0.9
    fld_beta2: float = 0.999
    fld_beta3: float = 0.98
    fld_share_gamma: float = 0.55
    fld_momentum: float = 0.25
    fld_decay: float = 0.015
    fld_restart_shrink: float = 0.50
    phase_phi_budget_early: float = 1.25
    phase_phi_budget_mid: float = 1.10
    rank_budget_early: float = 0.65
    margin_budget_early: float = 0.80
    consolidation_lr: float = 2.5e-4
    phase3_lr: float = 2.0e-4
    distill_temperature: float = 2.0
    kl_weight: float = 1.0
    feat_weight: float = 0.5
    ce_weight: float = 0.25
    sobolev_shrink_base: float = 1.0e-4


@dataclass
class FLDState:
    base: V44RunState = field(default_factory=V44RunState)
    prev_updates: Dict[str, torch.Tensor] = field(default_factory=dict)
    prev_grads: Dict[str, torch.Tensor] = field(default_factory=dict)
    role_lr: Dict[str, float] = field(default_factory=lambda: {"input": 1.0, "block": 1.0, "output": 1.0})
    restart_count: int = 0
    restart_reason: str = ""
    eta_scale: float = 1.0
    last_energy: float | None = None
    phase_step: int = 0


def _fc_metric_kind(method: str) -> str:
    key = method.lower().replace("-", "_")
    if "datasob" in key:
        return "dataSob"
    if "h1_low" in key:
        return "h1-low"
    if "h1" in key:
        return "h1"
    return "l2"


def _role_shares_from_norms(norms: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(v)) for v in norms.values())
    if total <= 1e-12:
        return {"input": 0.0, "block": 0.0, "output": 0.0}
    return {role: max(0.0, float(norms.get(role, 0.0))) / total for role in ["input", "block", "output"]}


def _scale_updates_by_role(
    updates: Dict[str, torch.Tensor],
    state: FLDState,
    params: V47Params,
    *,
    targets: Dict[str, float] | None = None,
    controller_strength: float = 1.0,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    targets = targets or {"input": 0.68, "block": 0.21, "output": 0.11}
    norms = _role_norms(updates)
    shares = _role_shares_from_norms(norms)
    scaled: Dict[str, torch.Tensor] = {}
    for name, upd in updates.items():
        role = _role_group(_role_for_name(name))
        if role not in state.role_lr:
            state.role_lr[role] = 1.0
        share_err = float(targets.get(role, shares.get(role, 0.0))) - float(shares.get(role, 0.0))
        mult = math.exp(params.fld_share_gamma * controller_strength * share_err)
        state.role_lr[role] = float(np.clip(0.90 * state.role_lr[role] + 0.10 * mult, 0.35, 2.50))
        scaled[name] = upd * state.role_lr[role] * state.eta_scale
    stats = {
        "role_input_share": shares.get("input", 0.0),
        "role_block_share": shares.get("block", 0.0),
        "role_output_share": shares.get("output", 0.0),
        "role_share_l2_error": math.sqrt(sum((shares.get(r, 0.0) - targets.get(r, 0.0)) ** 2 for r in targets)),
        "role_lr_input": state.role_lr.get("input", 1.0),
        "role_lr_block": state.role_lr.get("block", 1.0),
        "role_lr_output": state.role_lr.get("output", 1.0),
    }
    return scaled, stats


def _with_previous_updates(
    updates: Dict[str, torch.Tensor],
    state: FLDState,
    *,
    momentum: float,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, upd in updates.items():
        prev = state.prev_updates.get(name)
        out[name] = upd if prev is None else upd + momentum * prev.to(device=upd.device, dtype=upd.dtype)
    return out


def _capture_current_grads(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: p.grad.detach().clone()
        for name, p in coefficient_named_params(model)
        if p.grad is not None
    }


def _adanlite_updates(updates: Dict[str, torch.Tensor], state: FLDState) -> Dict[str, torch.Tensor]:
    # We use update deltas as a cheap functional-coordinate gradient-difference proxy.
    out: Dict[str, torch.Tensor] = {}
    for name, upd in updates.items():
        prev = state.prev_updates.get(name)
        if prev is None:
            out[name] = upd
        else:
            delta = upd - prev.to(device=upd.device, dtype=upd.dtype)
            out[name] = upd + 0.18 * delta
    return out


def _method_updates(
    model: torch.nn.Module,
    dataset: str,
    seed: int,
    method: str,
    params: V47Params,
    run_state: FLDState,
    xb: torch.Tensor,
    yb: torch.Tensor,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    key = method.lower().replace("-", "_")
    if key in {"purekan_adamw_one_step", "purekan_adamw"}:
        updates, rms = _adam_updates(model, run_state.base.adam, lr=params.adam_lr, mutate=True)
        return updates, {"optimizer_family": "adamw", "adam_rms_mean": _mean([float(v.mean().cpu()) for v in rms.values()], float("nan"))}
    if key in {"d0_allfullsobolev", "d6_alltaskaware", "f4_fng_leftfullright"}:
        v43_params = V43Params(
            train_size=params.train_size,
            val_size=params.val_size,
            test_size=params.test_size,
            batch_size=params.batch_size,
            hidden_dim=params.hidden_dim,
            depth=params.depth,
            basis_count=params.basis_count,
        )
        updates, stats = _v43_method_updates(model, dataset, seed, method, v43_params, V43RunState())
        stats["optimizer_family"] = stats.get("optimizer_family", "baseline_functional")
        return updates, stats
    if key in {"fcadam_datasob", "psft_a_fcadam_phasei_only", "psft_a_fcadamdistill", "psft_a_fcadamdistill_smoke", "psft_b_adamteacherdistill_smoke"}:
        updates, stats = _fcadam_updates(model, run_state.base.fc, params, metric_kind="dataSob")
        stats["optimizer_family"] = "fcadam_datasob_v47_phase_i"
        if key.startswith("psft"):
            updates, share_stats = _scale_updates_by_role(updates, run_state, params, controller_strength=1.2)
            stats.update(share_stats)
        return updates, stats
    if key in {"fld_adamcoord", "fld_nesterov", "fld_adanlite", "fld_winlite", "fld_lyapunovrestart", "fld_teacherenvelope", "psft_a_adan_phasei_only", "psft_a_adandistill", "psft_a_adandistill_smoke"}:
        updates, stats = _fcadam_updates(model, run_state.base.fc, params, metric_kind="dataSob")
        stats["optimizer_family"] = "fld_functional_learning_dynamics"
        stats["fld_method"] = method
        if key == "fld_nesterov":
            updates = _with_previous_updates(updates, run_state, momentum=params.fld_momentum)
            stats["nesterov_momentum"] = params.fld_momentum
        elif key in {"fld_adanlite", "fld_lyapunovrestart"}:
            updates = _adanlite_updates(updates, run_state)
            stats["adan_delta_proxy"] = 1
        elif key == "fld_winlite":
            decayed: Dict[str, torch.Tensor] = {}
            for name, upd in updates.items():
                p = dict(coefficient_named_params(model)).get(name)
                decayed[name] = upd - params.fld_decay * params.fc_lr * p.detach() if p is not None else upd
            updates = decayed
            stats["functional_decay"] = params.fld_decay
        strength = 1.4 if key == "fld_teacherenvelope" else 1.0
        updates, share_stats = _scale_updates_by_role(updates, run_state, params, controller_strength=strength)
        stats.update(share_stats)
        stats["eta_scale"] = run_state.eta_scale
        return updates, stats
    if key in {"psft_d_role_staged_phasei_only", "psft_d_role_staged_fcadamdistill", "psft_c_alternating_smoke"}:
        updates, stats = _fcadam_updates(model, run_state.base.fc, params, metric_kind="dataSob")
        progress = run_state.phase_step / max(1, params.phase1_steps)
        if progress < 1 / 3:
            targets = {"input": 0.65, "block": 0.25, "output": 0.10}
            stage = "input-heavy"
        elif progress < 2 / 3:
            targets = {"input": 0.35, "block": 0.45, "output": 0.20}
            stage = "block-refine"
        else:
            targets = {"input": 0.20, "block": 0.40, "output": 0.40}
            stage = "output-margin"
        updates, share_stats = _scale_updates_by_role(updates, run_state, params, targets=targets, controller_strength=1.4)
        stats.update(share_stats)
        stats["optimizer_family"] = "psft_role_staged_phase_i"
        stats["psft_role_stage"] = stage
        return updates, stats
    raise ValueError(f"unknown v4.7 method: {method}")


def _iter_steps(n: int, batch_size: int, seed: int, steps: int) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    epoch = 0
    while len(out) < steps:
        for idx in iter_minibatches(n, batch_size, seed + epoch * 1009):
            out.append(idx)
            if len(out) >= steps:
                break
        epoch += 1
    return out


def _role_norms(updates: Dict[str, torch.Tensor]) -> Dict[str, float]:
    by_role: Dict[str, List[float]] = {}
    for name, upd in updates.items():
        by_role.setdefault(_role_group(_role_for_name(name)), []).append(float(upd.detach().float().norm().cpu()))
    return {role: _mean(vals, 0.0) for role, vals in by_role.items()}


def _coef_norms(model: torch.nn.Module) -> Dict[str, float]:
    by_role: Dict[str, List[float]] = {}
    for name, p in coefficient_named_params(model):
        by_role.setdefault(_role_group(_role_for_name(name)), []).append(float(p.detach().float().norm().cpu()))
    return {role: _mean(vals, 0.0) for role, vals in by_role.items()}


def _trajectory_energy(
    holdout_loss: float,
    feat: Dict[str, float],
    geom: Dict[str, float],
    feat0: Dict[str, float],
    geom0: Dict[str, float],
    params: V47Params,
    *,
    update_norm: float = 0.0,
    phase: str = "Phase-I",
) -> float:
    rank_ratio = feat["block_feature_effective_rank"] / max(1e-12, feat0["block_feature_effective_rank"])
    margin_base = abs(feat0["margin_mean"]) if abs(feat0["margin_mean"]) > 1e-6 else 1.0
    margin_ratio = feat["margin_mean"] / margin_base
    phi_budget = params.phase_phi_budget_early if phase == "Phase-I" else params.phase_phi_budget_mid
    phi_ratio = geom["phi_prime_p95"] / max(1e-12, geom0["phi_prime_p95"])
    return float(
        holdout_loss
        + 0.25 * max(0.0, params.rank_budget_early - rank_ratio) ** 2
        + 0.25 * max(0.0, params.margin_budget_early - margin_ratio) ** 2
        + 0.15 * max(0.0, phi_ratio - phi_budget) ** 2
        + 1e-5 * update_norm * update_norm
    )


def _metric_conditions(model: torch.nn.Module, params: V47Params, device: torch.device) -> Dict[str, float]:
    conds: Dict[str, List[float]] = {"l2": [], "h1": [], "dataSob": []}
    for layer in model.kan_layers():  # type: ignore[attr-defined]
        for kind in conds:
            try:
                _, stats = _metric_matrix(layer, kind, params, device)
                conds[kind].append(float(stats["metric_condition"]))
            except Exception:
                pass
    return {f"metric_condition_{k}": _mean(v, float("nan")) for k, v in conds.items()}


def _roundtrip_errors(model: torch.nn.Module, params: V47Params, device: torch.device) -> Dict[str, float]:
    recon: List[float] = []
    update_rt: List[float] = []
    grad_chain: List[float] = []
    for name, p in coefficient_named_params(model):
        layer = list(model.kan_layers())[list(n for n, _ in coefficient_named_params(model)).index(name)]  # type: ignore[attr-defined]
        mat, _ = _metric_matrix(layer, "h1-low", params, device)
        chol = torch.linalg.cholesky(mat.double())
        coeff = p.detach().reshape(-1, p.shape[-1]).double()
        u = coeff @ chol
        coeff_back = torch.linalg.solve_triangular(chol.T, u.T, upper=True).T
        recon.append(float(((coeff_back - coeff).norm() / coeff.norm().clamp_min(1e-12)).cpu()))
        du = torch.randn_like(u) * 1e-3
        da = torch.linalg.solve_triangular(chol.T, du.T, upper=True).T
        du_back = da @ chol
        update_rt.append(float(((du_back - du).norm() / du.norm().clamp_min(1e-12)).cpu()))
        if p.grad is not None:
            grad = p.grad.detach().reshape(-1, p.shape[-1]).double()
            gu = torch.linalg.solve_triangular(chol, grad.T, upper=False).T
            grad_back = gu @ chol.T
            grad_chain.append(float(((grad_back - grad).norm() / grad.norm().clamp_min(1e-12)).cpu()))
    return {
        "whiten_reconstruction_error": _mean(recon, float("nan")),
        "u_to_a_roundtrip_error": _mean(update_rt, float("nan")),
        "gradient_transform_error": _mean(grad_chain, float("nan")),
    }


def _collect_pure_features(model: torch.nn.Module, x: torch.Tensor, *, detach: bool = False) -> Dict[str, torch.Tensor]:
    """Collect role-level PureKAN activations without adding trainable probes."""
    if not hasattr(model, "input_kan") or not hasattr(model, "blocks") or not hasattr(model, "output_kan"):
        logits = model(x)
        if detach:
            logits = logits.detach()
        return {"logits": logits, "input": logits, "block": logits, "output": logits}
    h = model.input_kan(x)  # type: ignore[attr-defined]
    input_feat = h.detach() if detach else h
    block_feats: List[torch.Tensor] = []
    for block in model.blocks:  # type: ignore[attr-defined]
        h, _ = block(h)
        block_feats.append(h.detach() if detach else h)
    output_in = model.output_norm(h)  # type: ignore[attr-defined]
    logits = model.output_kan(output_in)  # type: ignore[attr-defined]
    block_feat = torch.stack(block_feats).mean(dim=0) if block_feats else input_feat
    if detach:
        output_in = output_in.detach()
        logits = logits.detach()
        block_feat = block_feat.detach()
    return {"logits": logits, "input": input_feat, "block": block_feat, "output": output_in}


def _teacher_snapshot(model: torch.nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        return _collect_pure_features(model, x, detach=True)


def _distill_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    teacher: Dict[str, torch.Tensor],
    *,
    tau_z: float,
    tau_h: float,
    ce_weight: float,
    temperature: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    feats = _collect_pure_features(model, x)
    logits = feats["logits"]
    tlogits = teacher["logits"].to(device=logits.device, dtype=logits.dtype)
    logp = F.log_softmax(logits / temperature, dim=-1)
    tp = F.softmax(tlogits / temperature, dim=-1)
    kl = F.kl_div(logp, tp, reduction="batchmean") * (temperature**2)
    mse_vals = []
    for key in ["input", "block", "output"]:
        if key in teacher and teacher[key].shape == feats[key].shape:
            mse_vals.append(F.mse_loss(feats[key], teacher[key].to(device=feats[key].device, dtype=feats[key].dtype)))
    feat_mse = torch.stack(mse_vals).mean() if mse_vals else torch.zeros((), device=logits.device)
    ce = F.cross_entropy(logits, y)
    loss = ce_weight * ce + tau_z * kl + tau_h * feat_mse
    stats = {
        "distill_ce": float(ce.detach().cpu()),
        "distill_kl": float(kl.detach().cpu()),
        "distill_feature_mse": float(feat_mse.detach().cpu()),
        "distill_loss": float(loss.detach().cpu()),
    }
    return loss, stats


def _state_dict_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _load_state_dict_cpu(model: torch.nn.Module, state: Dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({k: v.to(device) for k, v in state.items()})


def _p2_grid(kind: str) -> List[Tuple[float, float, float, int]]:
    if kind == "full":
        tau_zs = [0.5, 1.0, 2.0]
        tau_hs = [0.1, 0.5]
        lambda_ss = [1e-4, 3e-4, 1e-3, 3e-3]
        steps = [20, 50, 100]
    else:
        # Core grid keeps the central plan ranges and the short/medium consolidation lengths.
        tau_zs = [1.0, 2.0]
        tau_hs = [0.1, 0.5]
        lambda_ss = [3e-4, 1e-3]
        steps = [20, 50]
    return [(tz, th, ls, st) for tz in tau_zs for th in tau_hs for ls in lambda_ss for st in steps]


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = []
    device = get_device(args.device)
    configs = [(16, 64, 2), (16, 64, 4)]
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        for method in P0_METHODS:
            for basis_count, hidden_dim, depth in configs:
                params = V47Params(train_size=512, val_size=128, test_size=128, basis_count=basis_count, hidden_dim=hidden_dim, depth=depth)
                bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=512, val_size=128, test_size=128, seed=0, download=not args.no_download, allow_fake_data=args.allow_fake_data)
                set_seed(0)
                model = _make_model(bundle, params, device, ab=False)
                idx = next(iter(iter_minibatches(len(bundle.x_train), params.batch_size, 0)))
                xb = bundle.x_train[idx].to(device)
                yb = bundle.y_train[idx].to(device)
                named = coefficient_named_params(model)
                snap = _snapshot(named)
                updates: Dict[str, torch.Tensor] = {}
                state = FLDState()
                err = ""
                distill_stats: Dict[str, float] = {}
                teacher_ok = 0
                try:
                    teacher = _teacher_snapshot(model, xb)
                    teacher_ok = int(all(torch.isfinite(v).all().item() for v in teacher.values()))
                    dloss, distill_stats = _distill_loss(
                        model,
                        xb,
                        yb,
                        teacher,
                        tau_z=params.kl_weight,
                        tau_h=params.feat_weight,
                        ce_weight=params.ce_weight,
                        temperature=params.distill_temperature,
                    )
                    _compute_loss_and_grad(model, xb, yb)
                    updates, stats = _method_updates(model, dataset, 0, method, params, state, xb, yb)
                    _apply_updates(named, updates)
                except Exception as exc:
                    err = repr(exc)
                    stats = {}
                _restore(named, snap)
                rollback = max(float((p.detach() - snap[n]).abs().max().cpu()) for n, p in named)
                coeff_total = sum(p.numel() for _, p in named)
                coeff_ids = {id(p) for _, p in named}
                nonkan = sum(p.numel() for p in model.parameters() if p.requires_grad and id(p) not in coeff_ids)
                m_finite = all(torch.isfinite(v).all().item() for v in state.base.fc.m.values()) if state.base.fc.m else True
                v_finite = all(torch.isfinite(v).all().item() for v in state.base.fc.v.values()) if state.base.fc.v else True
                row: Dict[str, Any] = {
                    "stage": "P0",
                    "dataset": dataset,
                    "method": method,
                    "basis_count": basis_count,
                    "hidden_dim": hidden_dim,
                    "depth": depth,
                    "learnable_nonKAN_params": nonkan,
                    "functional_coverage": 1.0 if coeff_total else 0.0,
                    "alpha_trainable": 0,
                    "input_coeff_seen": 1.0,
                    "block_coeff_seen": 1.0,
                    "output_coeff_seen": 1.0,
                    "state_m_finite": int(m_finite),
                    "state_v_finite": int(v_finite),
                    "teacher_snapshot_ok": teacher_ok,
                    "teacher_logits_finite": teacher_ok,
                    "teacher_features_finite": teacher_ok,
                    "distill_loss_finite": int(math.isfinite(distill_stats.get("distill_loss", float("nan")))),
                    "distill_loss": distill_stats.get("distill_loss", float("nan")),
                    "distill_kl": distill_stats.get("distill_kl", float("nan")),
                    "distill_feature_mse": distill_stats.get("distill_feature_mse", float("nan")),
                    "restart_state_finite": int(math.isfinite(float(state.restart_count))),
                    "role_lr_multiplier_finite": int(all(math.isfinite(v) for v in state.role_lr.values())),
                    "phase_trace": "Phase-I|Phase-II|Phase-III" if method.startswith("PSFT") else "Phase-I",
                    "role_share_trace": "|".join(f"{r}:{state.role_lr.get(r, 1.0):.3f}" for r in ["input", "block", "output"]),
                    "phase_state": "Phase-I",
                    "geometry_budget": params.phase_phi_budget_early,
                    "adam_state_shape_match": int(all(name in updates for name, _ in named)) if not err and updates else 0,
                    "rollback_error": rollback,
                    "nan_count": sum(int((~torch.isfinite(p)).sum().item()) for _, p in named),
                    "no_nan_inf": int(all(torch.isfinite(p).all().item() for _, p in named)),
                    "optimizer_family": stats.get("optimizer_family", ""),
                    "error": err,
                }
                row.update(_metric_conditions(model, params, device))
                row.update(_roundtrip_errors(model, params, device))
                rows.append(row)
                print(f"P0 {dataset} {method} d={depth} nonKAN={nonkan} rt={row['whiten_reconstruction_error']:.2g} err={err}")
    write_csv(out_dir / "p0_psft_invariants.csv", rows)
    return rows


def run_p1(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = []
    device = get_device(args.device)
    params = V47Params(train_size=1536, val_size=512, test_size=512)
    methods = [m for m in P1_METHODS if not parse_str_list(args.methods) or m in set(parse_str_list(args.methods))]
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        for method in methods:
            for seed in parse_int_list(args.seeds):
                bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=params.train_size, val_size=params.val_size, test_size=params.test_size, seed=seed, download=not args.no_download, allow_fake_data=args.allow_fake_data)
                set_seed(seed)
                model = _make_model(bundle, params, device, ab=False)
                state = FLDState()
                idxs = _iter_steps(len(bundle.x_train), params.batch_size, seed + 123, params.p1_steps)
                hb = bundle.x_val[: params.batch_size].to(device)
                yh = bundle.y_val[: params.batch_size].to(device)
                hold0 = F.cross_entropy(model(hb), yh).item()
                feat0 = _feature_audit(model, hb, yh)
                geom0 = _estimate_geometry_any(model, hb, device=device, batch_size=params.audit_batch_size)
                checkpoints = {1, 5, 20, 50, 100}
                bad = 0
                role_accum = {"input": 0.0, "block": 0.0, "output": 0.0}
                for step, idx in enumerate(idxs, start=1):
                    state.phase_step = step
                    xb = bundle.x_train[idx].to(device)
                    yb = bundle.y_train[idx].to(device)
                    train_loss = F.cross_entropy(model(xb), yb).item()
                    hold_loss = F.cross_entropy(model(hb), yh).item()
                    _compute_loss_and_grad(model, xb, yb)
                    updates, stats = _method_updates(model, dataset, seed, method, params, state, xb, yb)
                    role_updates = _role_norms(updates)
                    for role in role_accum:
                        role_accum[role] += role_updates.get(role, 0.0)
                    _apply_updates(coefficient_named_params(model), updates)
                    if method.startswith("FLD") or method.startswith("PSFT"):
                        state.prev_updates = {name: upd.detach().clone() for name, upd in updates.items()}
                    if F.cross_entropy(model(xb), yb).item() > train_loss:
                        bad += 1
                    if step not in checkpoints:
                        continue
                    feat = _feature_audit(model, hb, yh)
                    geom = _estimate_geometry_any(model, hb, device=device, batch_size=params.audit_batch_size)
                    occ = rbf_basis_occupancy_audit(model)
                    total_role = sum(role_accum.values())
                    shares = {r: role_accum[r] / max(1e-12, total_role) for r in role_accum}
                    test = evaluate(model, bundle.x_test, bundle.y_test, device=device, batch_size=params.eval_batch_size) if step in {20, 100} else {"acc": float("nan"), "loss": float("nan"), "ece": float("nan")}
                    margin_base = abs(feat0["margin_mean"]) if abs(feat0["margin_mean"]) > 1e-6 else 1.0
                    row = {
                        "stage": "P1",
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "step": step,
                        "phase": "Phase-I",
                        "train_loss": train_loss,
                        "holdout_loss": hold_loss,
                        "holdout_descent": hold0 - hold_loss,
                        "test_acc": test["acc"],
                        "test_ece": test["ece"],
                        "feature_effective_rank_input": feat["input_feature_effective_rank"],
                        "feature_effective_rank_block": feat["block_feature_effective_rank"],
                        "rank_ratio_initial": feat["block_feature_effective_rank"] / max(1e-12, feat0["block_feature_effective_rank"]),
                        "class_centroid_separation": feat["class_centroid_separation"],
                        "margin_mean": feat["margin_mean"],
                        "margin_p10": feat["margin_p10"],
                        "margin_change": feat["margin_mean"] - feat0["margin_mean"],
                        "margin_ratio_initial": feat["margin_mean"] / margin_base,
                        "phi_prime_p95": geom["phi_prime_p95"],
                        "phi_ratio_initial": geom["phi_prime_p95"] / max(1e-12, geom0["phi_prime_p95"]),
                        "jacobian_condition": geom["max_jac_condition"],
                        "basis_occupancy_entropy": occ.get("rbf_basis_mean", float("nan")),
                        "dead_basis_fraction": occ.get("rbf_basis_dead_frac", float("nan")),
                        "input_update_share": shares.get("input", 0.0),
                        "block_update_share": shares.get("block", 0.0),
                        "output_update_share": shares.get("output", 0.0),
                        "role_share_l2_error": math.sqrt((shares.get("input", 0.0) - 0.68) ** 2 + (shares.get("block", 0.0) - 0.21) ** 2 + (shares.get("output", 0.0) - 0.11) ** 2),
                        "bad_step_rate": bad / max(1, step),
                        "optimizer_family": stats.get("optimizer_family", ""),
                        "psft_role_stage": stats.get("psft_role_stage", ""),
                    }
                    rows.append(row)
                    print(f"P1 {dataset} {method} seed={seed} step={step} holdD={row['holdout_descent']:.4f} rank={row['rank_ratio_initial']:.3f} phi={row['phi_ratio_initial']:.3f}")
    write_csv(out_dir / "p1_phase_i_dynamics.csv", rows)
    return rows


def _run_p2_one(args: argparse.Namespace, dataset: str, seed: int, method: str, params: V47Params, *, steps: int | None = None, stage: str = "P2") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    device = get_device(args.device)
    bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=params.train_size, val_size=params.val_size, test_size=params.test_size, seed=seed, download=not args.no_download, allow_fake_data=args.allow_fake_data)
    set_seed(seed)
    model = _make_model(bundle, params, device, ab=("AB-RBF" in method))
    state = FLDState()
    actual_steps = int(steps or params.p2_steps)
    idxs = _iter_steps(len(bundle.x_train), params.batch_size, seed + 45, actual_steps)
    hb = bundle.x_val[: params.batch_size].to(device)
    yh = bundle.y_val[: params.batch_size].to(device)
    train0 = F.cross_entropy(model(bundle.x_train[idxs[0]].to(device)), bundle.y_train[idxs[0]].to(device)).item()
    hold0 = F.cross_entropy(model(hb), yh).item()
    feat0 = _feature_audit(model, hb, yh)
    geom0 = _estimate_geometry_any(model, hb, device=device, batch_size=params.audit_batch_size)
    trace: List[Dict[str, Any]] = []
    bad = 0
    cos_vals: List[float] = []
    r2_vals: List[float] = []
    train_after: Dict[int, float] = {}
    hold_after: Dict[int, float] = {}
    accepted_lrs: List[float] = []
    energy_vals: List[float] = []
    share_errors: List[float] = []
    negative_holdout = 0
    restart_reasons: List[str] = []
    last_feat = feat0
    last_geom = geom0
    start = time.perf_counter()
    for step, idx in enumerate(idxs, start=1):
        xb = bundle.x_train[idx].to(device)
        yb = bundle.y_train[idx].to(device)
        tb = F.cross_entropy(model(xb), yb).item()
        hb_loss = F.cross_entropy(model(hb), yh).item()
        _compute_loss_and_grad(model, xb, yb)
        adam_ref, _ = _adam_updates(model, AdamState(), lr=params.adam_lr, mutate=True)
        adam_fun, _ = _function_displacement(model, adam_ref, xb)
        updates, stats = _method_updates(model, dataset, seed, method, params, state, xb, yb)
        cand_fun, drift = _function_displacement(model, updates, xb)
        cos_f = _safe_cos(cand_fun, adam_fun)
        r2 = _r2(cand_fun, adam_fun)
        cos_vals.append(cos_f)
        r2_vals.append(r2)
        named = coefficient_named_params(model)
        role_updates = _role_norms(updates)
        role_shares = _role_shares_from_norms(role_updates)
        share_l2 = float(stats.get("role_share_l2_error", math.sqrt((role_shares.get("input", 0.0) - 0.68) ** 2 + (role_shares.get("block", 0.0) - 0.21) ** 2 + (role_shares.get("output", 0.0) - 0.11) ** 2)))
        share_errors.append(share_l2)
        update_norm = float(_flat_updates(updates).norm().detach().cpu())
        snap = _snapshot(named)
        _apply_updates(named, updates)
        ta = F.cross_entropy(model(xb), yb).item()
        ha = F.cross_entropy(model(hb), yh).item()
        phase = "Phase-I" if step <= 20 else "Phase-II"
        checkpoints = {1, 5, 20, 50, 100, actual_steps}
        audit_now = step in checkpoints or (method == "FLD-LyapunovRestart" and step % 5 == 0)
        if audit_now:
            feat = _feature_audit(model, hb, yh)
            geom = _estimate_geometry_any(model, hb, device=device, batch_size=params.audit_batch_size)
            last_feat = feat
            last_geom = geom
        else:
            feat = last_feat
            geom = last_geom
        energy = _trajectory_energy(ha, feat, geom, feat0, geom0, params, update_norm=update_norm, phase=phase)
        if method == "FLD-LyapunovRestart" and audit_now and (
            (state.last_energy is not None and energy > state.last_energy + 0.03)
            or ha > hb_loss + 0.03
            or geom["phi_prime_p95"] / max(1e-12, geom0["phi_prime_p95"]) > params.phase_phi_budget_early
        ):
            _restore(named, snap)
            updates = {name: upd * params.fld_restart_shrink for name, upd in updates.items()}
            state.base.fc = FCAdamState()
            state.eta_scale = max(0.20, state.eta_scale * 0.70)
            state.restart_count += 1
            state.restart_reason = "energy_or_holdout_or_phi"
            restart_reasons.append(state.restart_reason)
            _apply_updates(named, updates)
            ta = F.cross_entropy(model(xb), yb).item()
            ha = F.cross_entropy(model(hb), yh).item()
            feat = _feature_audit(model, hb, yh)
            geom = _estimate_geometry_any(model, hb, device=device, batch_size=params.audit_batch_size)
            last_feat = feat
            last_geom = geom
            update_norm = float(_flat_updates(updates).norm().detach().cpu())
            energy = _trajectory_energy(ha, feat, geom, feat0, geom0, params, update_norm=update_norm, phase=phase)
        if method == "FLD-TeacherEnvelope" and audit_now and geom["phi_prime_p95"] / max(1e-12, geom0["phi_prime_p95"]) > params.phase_phi_budget_early:
            _restore(named, snap)
            updates = {name: upd * 0.75 for name, upd in updates.items()}
            _apply_updates(named, updates)
            ta = F.cross_entropy(model(xb), yb).item()
            ha = F.cross_entropy(model(hb), yh).item()
            feat = _feature_audit(model, hb, yh)
            geom = _estimate_geometry_any(model, hb, device=device, batch_size=params.audit_batch_size)
            last_feat = feat
            last_geom = geom
            update_norm = float(_flat_updates(updates).norm().detach().cpu())
            energy = _trajectory_energy(ha, feat, geom, feat0, geom0, params, update_norm=update_norm, phase=phase)
        state.last_energy = energy
        energy_vals.append(energy)
        if ta > tb:
            bad += 1
        if ha > hb_loss:
            negative_holdout += 1
        if step in {1, 5, 20, 50, 100}:
            train_after[step] = ta
            hold_after[step] = ha
        accepted_lrs.append(params.adam_lr if "AdamW" in method else params.fc_lr)
        if method.startswith("FLD"):
            state.prev_updates = {name: upd.detach().clone() for name, upd in updates.items()}
            state.prev_grads = _capture_current_grads(model)
        trace.append(
            {
                "stage": stage,
                "dataset": dataset,
                "seed": seed,
                "method": method,
                "step": step,
                "phase": phase,
                "train_loss_before": tb,
                "train_loss_after": ta,
                "holdout_loss_before": hb_loss,
                "holdout_loss_after": ha,
                "cos_function_with_adam": cos_f,
                "function_R2_with_adam": r2,
                "logit_drift": drift,
                "feature_rank": feat["block_feature_effective_rank"],
                "margin_mean": feat["margin_mean"],
                "rank_ratio_initial": feat["block_feature_effective_rank"] / max(1e-12, feat0["block_feature_effective_rank"]),
                "margin_ratio_initial": feat["margin_mean"] / max(1e-12, abs(feat0["margin_mean"]) if abs(feat0["margin_mean"]) > 1e-6 else 1.0),
                "phi_prime_p95": geom["phi_prime_p95"],
                "phi_ratio_initial": geom["phi_prime_p95"] / max(1e-12, geom0["phi_prime_p95"]),
                "jacobian_condition": geom["max_jac_condition"],
                "trajectory_energy": energy,
                "role_input_share": role_shares.get("input", 0.0),
                "role_block_share": role_shares.get("block", 0.0),
                "role_output_share": role_shares.get("output", 0.0),
                "role_share_l2_error": share_l2,
                "u_m_norm": _mean([float(v.norm().cpu()) for v in state.base.fc.m.values()], float("nan")),
                "u_v_norm": _mean([float(v.norm().cpu()) for v in state.base.fc.v.values()], float("nan")),
                "accepted_lr": accepted_lrs[-1],
                "eta_scale": state.eta_scale,
                "update_norm": update_norm,
                "restart_count": state.restart_count,
                "restart_reason": state.restart_reason,
                "optimizer_family": stats.get("optimizer_family", ""),
                "geometry_policy": stats.get("geometry_policy", ""),
            }
        )
    wall = time.perf_counter() - start
    feat_end = _feature_audit(model, hb, yh)
    geom_end = _estimate_geometry_any(model, hb, device=device, batch_size=params.audit_batch_size)
    test = evaluate(model, bundle.x_test, bundle.y_test, device=device, batch_size=params.eval_batch_size)
    row = {
        "stage": stage,
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "steps": actual_steps,
        "test_acc_after_20step": test["acc"],
        "ECE": test["ece"],
        "holdout_1step_descent": hold0 - hold_after.get(1, float("nan")),
        "holdout_5step_descent": hold0 - hold_after.get(5, float("nan")),
        "holdout_20step_descent": hold0 - hold_after.get(20, float("nan")),
        "holdout_50step_descent": hold0 - hold_after.get(50, float("nan")),
        "holdout_100step_descent": hold0 - hold_after.get(100, float("nan")),
        "holdout_final_descent": hold0 - hold_after.get(actual_steps, float("nan")),
        "train_20step_descent": train0 - train_after.get(20, float("nan")),
        "train_100step_descent": train0 - train_after.get(100, float("nan")),
        "bad_step_rate": bad / max(1, actual_steps),
        "negative_holdout_step_rate": negative_holdout / max(1, actual_steps),
        "cos_function_with_adam": _mean(cos_vals, float("nan")),
        "function_R2_with_adam": _mean(r2_vals, float("nan")),
        "rank_final": feat_end["block_feature_effective_rank"],
        "rank_change": feat_end["block_feature_effective_rank"] - feat0["block_feature_effective_rank"],
        "rank_ratio_initial": feat_end["block_feature_effective_rank"] / max(1e-12, feat0["block_feature_effective_rank"]),
        "margin_final": feat_end["margin_mean"],
        "margin_p10_final": feat_end["margin_p10"],
        "margin_change": feat_end["margin_mean"] - feat0["margin_mean"],
        "margin_ratio_initial": feat_end["margin_mean"] / max(1e-12, abs(feat0["margin_mean"]) if abs(feat0["margin_mean"]) > 1e-6 else 1.0),
        "phi_prime_p95": geom_end["phi_prime_p95"],
        "phi_change": geom_end["phi_prime_p95"] - geom0["phi_prime_p95"],
        "phi_ratio_initial": geom_end["phi_prime_p95"] / max(1e-12, geom0["phi_prime_p95"]),
        "jacobian_condition": geom_end["max_jac_condition"],
        "jac_change": geom_end["max_jac_condition"] - geom0["max_jac_condition"],
        "accepted_lr": _mean(accepted_lrs, float("nan")),
        "role_share_l2_error": _mean(share_errors, float("nan")),
        "trajectory_energy_final": energy_vals[-1] if energy_vals else float("nan"),
        "trajectory_energy_auc": _mean(energy_vals, float("nan")),
        "u_m_norm": _mean([float(v.norm().cpu()) for v in state.base.fc.m.values()], float("nan")),
        "u_v_norm": _mean([float(v.norm().cpu()) for v in state.base.fc.v.values()], float("nan")),
        "u_update_norm": _mean([float(t.get("update_norm", float("nan"))) for t in trace], float("nan")),
        "restart_count": state.restart_count,
        "restart_reason": "|".join(sorted(set(restart_reasons))) if restart_reasons else "",
        "eta_scale_final": state.eta_scale,
        "step_time_ms": 1000.0 * wall / max(1, actual_steps),
        "error": "",
    }
    return row, trace


def run_p2(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows = [] if args.fresh else read_csv(out_dir / "p2_fld_dynamics_audit.csv")
    trace = [] if args.fresh else read_csv(out_dir / "optimizer_state_trace.csv")
    params = V47Params()
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
                    row, tr = _run_p2_one(args, dataset, seed, method, params)
                    rows.append(row)
                    trace.extend(tr)
                    print(f"P2 {dataset} {method} seed={seed} hold20={row['holdout_20step_descent']:.4g} cos={row['cos_function_with_adam']:.3f} r2={row['function_R2_with_adam']:.3f}")
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    rows.append({"stage": "P2", "dataset": dataset, "seed": seed, "method": method, "error": repr(exc)})
                    print(f"P2 ERROR {dataset} {method} seed={seed}: {exc!r}")
                write_csv(out_dir / "p2_fld_dynamics_audit.csv", rows)
                write_csv(out_dir / "optimizer_state_trace.csv", trace)
    return rows


def run_p3(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows = [] if args.fresh else read_csv(out_dir / "p3_100step_trajectory.csv")
    trace = read_csv(out_dir / "optimizer_state_trace.csv") if (out_dir / "optimizer_state_trace.csv").exists() and not args.fresh else []
    params = V47Params()
    datasets = [dataset_name(d) for d in parse_str_list(args.datasets)]
    seeds = parse_int_list(args.seeds)
    wanted = parse_str_list(args.methods)
    methods = wanted if wanted else P3_FORCED_METHODS
    done = {(r.get("dataset"), int(r.get("seed", -1)), r.get("method")) for r in rows if not r.get("error")}
    for dataset in datasets:
        for method in methods:
            for seed in seeds:
                if (dataset, seed, method) in done:
                    continue
                try:
                    row, tr = _run_p2_one(args, dataset, seed, method, params, steps=params.p3_steps, stage="P3")
                    rows.append(row)
                    trace.extend(tr)
                    print(f"P3 {dataset} {method} seed={seed} hold100={row['holdout_100step_descent']:.4g} energy={row['trajectory_energy_final']:.4g}")
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    rows.append({"stage": "P3", "dataset": dataset, "seed": seed, "method": method, "error": repr(exc)})
                    print(f"P3 ERROR {dataset} {method} seed={seed}: {exc!r}")
                write_csv(out_dir / "p3_100step_trajectory.csv", rows)
                write_csv(out_dir / "optimizer_state_trace.csv", trace)
    return rows


def _train_phase1_model(
    args: argparse.Namespace,
    dataset: str,
    seed: int,
    method: str,
    params: V47Params,
    device: torch.device,
) -> Tuple[Any, torch.nn.Module, Dict[str, Any]]:
    bundle = load_vision_bundle(
        dataset,
        data_root=args.data_root,
        train_size=params.train_size,
        val_size=params.val_size,
        test_size=params.test_size,
        seed=seed,
        download=not args.no_download,
        allow_fake_data=args.allow_fake_data,
    )
    set_seed(seed)
    model = _make_model(bundle, params, device, ab=False)
    state = FLDState()
    idxs = _iter_steps(len(bundle.x_train), params.batch_size, seed + 4700, params.phase1_steps)
    hb = bundle.x_val[: params.batch_size].to(device)
    yh = bundle.y_val[: params.batch_size].to(device)
    hold0 = F.cross_entropy(model(hb), yh).item()
    feat0 = _feature_audit(model, hb, yh)
    geom0 = _estimate_geometry_any(model, hb, device=device, batch_size=params.audit_batch_size)
    bad = 0
    for step, idx in enumerate(idxs, start=1):
        state.phase_step = step
        xb = bundle.x_train[idx].to(device)
        yb = bundle.y_train[idx].to(device)
        before = _compute_loss_and_grad(model, xb, yb)
        updates, _ = _method_updates(model, dataset, seed, method, params, state, xb, yb)
        _apply_updates(coefficient_named_params(model), updates)
        if method.startswith("FLD") or method.startswith("PSFT"):
            state.prev_updates = {name: upd.detach().clone() for name, upd in updates.items()}
        after = F.cross_entropy(model(xb), yb).item()
        bad += int(after > before)
    hold1 = F.cross_entropy(model(hb), yh).item()
    feat1 = _feature_audit(model, hb, yh)
    geom1 = _estimate_geometry_any(model, hb, device=device, batch_size=params.audit_batch_size)
    test = evaluate(model, bundle.x_test, bundle.y_test, device=device, batch_size=params.eval_batch_size)
    teacher = _teacher_snapshot(model, hb)
    meta = {
        "teacher_method": method,
        "holdout_before": hold0,
        "holdout_after_phase1": hold1,
        "phase1_holdout_descent": hold0 - hold1,
        "phase1_acc": test["acc"],
        "phase1_ece": test["ece"],
        "phase1_bad_rate": bad / max(1, len(idxs)),
        "phase1_rank_ratio": feat1["block_feature_effective_rank"] / max(1e-12, feat0["block_feature_effective_rank"]),
        "phase1_margin_change": feat1["margin_mean"] - feat0["margin_mean"],
        "phase1_phi_ratio": geom1["phi_prime_p95"] / max(1e-12, geom0["phi_prime_p95"]),
        "phase1_phi": geom1["phi_prime_p95"],
        "phase1_jacobian": geom1["max_jac_condition"],
        "teacher_logits_finite": int(torch.isfinite(teacher["logits"]).all().item()),
        "teacher_features_finite": int(all(torch.isfinite(teacher[k]).all().item() for k in ["input", "block", "output"])),
    }
    return bundle, model, meta


def _consolidate_student(
    args: argparse.Namespace,
    dataset: str,
    seed: int,
    teacher_label: str,
    teacher_method: str,
    bundle: Any,
    teacher_state: Dict[str, torch.Tensor],
    teacher_snapshot: Dict[str, torch.Tensor],
    teacher_meta: Dict[str, Any],
    params: V47Params,
    device: torch.device,
    *,
    tau_z: float,
    tau_h: float,
    lambda_s: float,
    steps: int,
) -> Dict[str, Any]:
    set_seed(seed + 17)
    student = _make_model(bundle, params, device, ab=False)
    _load_state_dict_cpu(student, teacher_state, device)
    hb = bundle.x_val[: params.batch_size].to(device)
    yh = bundle.y_val[: params.batch_size].to(device)
    idxs = _iter_steps(len(bundle.x_train), params.batch_size, seed + 4800 + steps, steps)
    test_before = evaluate(student, bundle.x_test, bundle.y_test, device=device, batch_size=params.eval_batch_size)
    feat_before = _feature_audit(student, hb, yh)
    geom_before = _estimate_geometry_any(student, hb, device=device, batch_size=params.audit_batch_size)
    state = FCAdamState()
    distill_vals: List[Dict[str, float]] = []
    start = time.perf_counter()
    for idx in idxs:
        xb = bundle.x_train[idx].to(device)
        yb = bundle.y_train[idx].to(device)
        student.train()
        student.zero_grad(set_to_none=True)
        loss, dstats = _distill_loss(
            student,
            xb,
            yb,
            teacher_snapshot,
            tau_z=tau_z,
            tau_h=tau_h,
            ce_weight=params.ce_weight,
            temperature=params.distill_temperature,
        )
        loss.backward()
        cparams = copy.copy(params)
        cparams.fc_lr = params.consolidation_lr
        cparams.fc_h1_low_alpha = max(0.10, min(0.60, 0.10 + 80.0 * lambda_s))
        updates, _ = _fcadam_updates(student, state, cparams, metric_kind="h1-low")
        named_now = dict(coefficient_named_params(student))
        for name, upd in list(updates.items()):
            p = named_now.get(name)
            if p is not None:
                updates[name] = upd - lambda_s * params.consolidation_lr * p.detach()
        _apply_updates(coefficient_named_params(student), updates)
        distill_vals.append(dstats)
    wall = time.perf_counter() - start
    test_after = evaluate(student, bundle.x_test, bundle.y_test, device=device, batch_size=params.eval_batch_size)
    feat_after = _feature_audit(student, hb, yh)
    geom_after = _estimate_geometry_any(student, hb, device=device, batch_size=params.audit_batch_size)
    loss_final, dstats_final = _distill_loss(
        student,
        hb,
        yh,
        teacher_snapshot,
        tau_z=tau_z,
        tau_h=tau_h,
        ce_weight=params.ce_weight,
        temperature=params.distill_temperature,
    )
    acc_drop = test_before["acc"] - test_after["acc"]
    phi_reduction = 1.0 - geom_after["phi_prime_p95"] / max(1e-12, geom_before["phi_prime_p95"])
    j_reduction = 1.0 - geom_after["max_jac_condition"] / max(1e-12, geom_before["max_jac_condition"])
    p2_pass = (
        acc_drop <= 0.005
        and phi_reduction >= 0.15
        and j_reduction >= 0.15
        and dstats_final["distill_kl"] <= 0.05
    )
    return {
        "stage": "P2",
        "dataset": dataset,
        "seed": seed,
        "teacher": teacher_label,
        "teacher_method": teacher_method,
        "tau_z": tau_z,
        "tau_h": tau_h,
        "lambda_s": lambda_s,
        "consolidation_steps": steps,
        **teacher_meta,
        "acc_before": test_before["acc"],
        "acc_after": test_after["acc"],
        "acc_drop": acc_drop,
        "ece_before": test_before["ece"],
        "ece_after": test_after["ece"],
        "ece_reduction": test_before["ece"] - test_after["ece"],
        "rank_before": feat_before["block_feature_effective_rank"],
        "rank_after": feat_after["block_feature_effective_rank"],
        "margin_before": feat_before["margin_mean"],
        "margin_after": feat_after["margin_mean"],
        "phi_before": geom_before["phi_prime_p95"],
        "phi_after": geom_after["phi_prime_p95"],
        "phi_reduction": phi_reduction,
        "j_before": geom_before["max_jac_condition"],
        "j_after": geom_after["max_jac_condition"],
        "j_reduction": j_reduction,
        "kl_final": dstats_final["distill_kl"],
        "feature_mse_final": dstats_final["distill_feature_mse"],
        "distill_loss_final": dstats_final["distill_loss"],
        "distill_loss_mean": _mean([d["distill_loss"] for d in distill_vals], float("nan")),
        "teacher_preserve_ok": int(dstats_final["distill_kl"] <= 0.05 and dstats_final["distill_feature_mse"] <= 0.20),
        "p2_pass": int(p2_pass),
        "step_time_ms": 1000.0 * wall / max(1, steps),
        "error": "",
    }


def run_p2(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows = [] if args.fresh else read_csv(out_dir / "p2_consolidation_sweep.csv")
    params = V47Params()
    device = get_device(args.device)
    datasets = [dataset_name(d) for d in parse_str_list(args.datasets)]
    seeds = parse_int_list(args.seeds)
    if args.seeds == add_args().get_default("seeds"):
        seeds = [0]
    grid = _p2_grid(getattr(args, "p2_grid", "core"))
    done = {
        (
            r.get("dataset"),
            int(r.get("seed", -1)),
            r.get("teacher"),
            float(r.get("tau_z", -1)),
            float(r.get("tau_h", -1)),
            float(r.get("lambda_s", -1)),
            int(float(r.get("consolidation_steps", -1))),
        )
        for r in rows
        if not r.get("error")
    }
    for dataset in datasets:
        for seed in seeds:
            teacher_cache: List[Tuple[str, str, Any, Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]] = []
            for teacher_label, teacher_method in P2_TEACHERS:
                bundle, teacher_model, meta = _train_phase1_model(args, dataset, seed, teacher_method, params, device)
                hb = bundle.x_val[: params.batch_size].to(device)
                teacher_cache.append((teacher_label, teacher_method, bundle, _state_dict_cpu(teacher_model), _teacher_snapshot(teacher_model, hb), meta))
                print(f"P2 teacher {dataset} {teacher_label} seed={seed} phase1_acc={meta['phase1_acc']:.4f} phi={meta['phase1_phi']:.4f}")
            for teacher_label, teacher_method, bundle, teacher_state, teacher_snap, meta in teacher_cache:
                for tau_z, tau_h, lambda_s, steps in grid:
                    key = (dataset, seed, teacher_label, tau_z, tau_h, lambda_s, steps)
                    if key in done:
                        continue
                    try:
                        row = _consolidate_student(
                            args,
                            dataset,
                            seed,
                            teacher_label,
                            teacher_method,
                            bundle,
                            teacher_state,
                            teacher_snap,
                            meta,
                            params,
                            device,
                            tau_z=tau_z,
                            tau_h=tau_h,
                            lambda_s=lambda_s,
                            steps=steps,
                        )
                        rows.append(row)
                        print(
                            f"P2 {dataset} {teacher_label} tz={tau_z} th={tau_h} ls={lambda_s:g} st={steps} "
                            f"accDrop={row['acc_drop']:.4f} phiRed={row['phi_reduction']:.3f} JRed={row['j_reduction']:.3f} KL={row['kl_final']:.4f} pass={row['p2_pass']}"
                        )
                    except Exception as exc:
                        if not args.continue_on_error:
                            raise
                        rows.append({"stage": "P2", "dataset": dataset, "seed": seed, "teacher": teacher_label, "teacher_method": teacher_method, "tau_z": tau_z, "tau_h": tau_h, "lambda_s": lambda_s, "consolidation_steps": steps, "error": repr(exc)})
                        print(f"P2 ERROR {dataset} {teacher_label}: {exc!r}")
                    write_csv(out_dir / "p2_consolidation_sweep.csv", rows)
    return rows


def _blank_later_files(out_dir: Path, reason: str) -> None:
    for name in [
        "p3_one_cycle_micro_run.csv",
        "p4_cycle_ablation.csv",
        "p5_role_stage_ablation.csv",
        "p6_cifar_precheck.csv",
        "p7_confirm5.csv",
        "p8_final10.csv",
        "p9_failure_diagnosis.csv",
    ]:
        path = out_dir / name
        if not path.exists():
            write_csv(path, [{"status": "not_run", "reason": reason}])


def add_args() -> argparse.ArgumentParser:
    p = add_v3_args()
    p.description = __doc__
    p.add_argument("--p2-grid", choices=["core", "full"], default="core")
    p.set_defaults(packages="V4_7_P0_SMOKE", datasets="MNIST,Fashion-MNIST,KMNIST", out_dir=Path("results/v4_7"), seeds="0")
    return p


def main() -> int:
    args = add_args().parse_args()
    all_rows: List[Dict[str, Any]] = []
    for package in parse_str_list(args.packages):
        key = package.strip().upper().replace("-", "_")
        if key == "V4_7_P0_SMOKE":
            all_rows = run_p0(args)
        elif key in {"V4_7_P1_PHASEI", "V4_7_P1_PHASE_I", "V4_7_P1_ADAMW_ENVELOPE"}:
            all_rows = run_p1(args)
        elif key in {"V4_7_P2_CONSOLIDATION", "V4_7_P2_DYNAMICS"}:
            all_rows = run_p2(args)
            _blank_later_files(ensure_dir(args.out_dir), "pending analyzer gate decision")
        elif key == "V4_7_P3_ONE_CYCLE":
            _blank_later_files(ensure_dir(args.out_dir), "P3 is gated by P2 consolidation survivors")
            all_rows = [{"status": "not_run"}]
        else:
            raise ValueError(f"unknown v4.7 package: {package}")
    return 0 if all_rows is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
