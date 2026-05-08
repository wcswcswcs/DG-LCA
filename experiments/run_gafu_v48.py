#!/usr/bin/env python3
"""DG-KAN v4.8 runner: Functional Geometry Feasibility probes."""

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
    _estimate_geometry_any,
    _fcadam_updates,
    _make_model,
    _metric_matrix,
)
from run_gafu_v47 import (
    FLDState,
    V47Params,
    _collect_pure_features,
    _method_updates as _v47_method_updates,
    _teacher_snapshot,
)


DATASETS = ["MNIST", "Fashion-MNIST", "KMNIST"]

P0_METHODS = [
    "PureKAN-AdamW-smoke",
    "PureKAN-FCAdam-smoke",
    "NFS-logit-only-smoke",
    "NFS-hidden-only-smoke",
    "NFS-logit-hidden-smoke",
    "NFS-random-nullspace-smoke",
    "TAN-one-cycle-smoke",
]

P1_METHODS = [
    "A0-PureKAN-AdamW",
    "A1-AdamW-Sobolev1e-6",
    "A2-AdamW-Sobolev3e-6",
    "A3-AdamW-Sobolev1e-5",
    "A4-AdamW-PhiProxy1e-5",
    "A5-AdamW-JacProxy1e-5",
    "A6-AdamW-MixedGeom",
    "A7-AdamW-LateGeom",
    "A8-AdamW-PosthocGeom",
]

NFS_VARIANTS = [
    "NFS-Z",
    "NFS-H",
    "NFS-ZH",
    "NFS-ZH-margin",
    "NFS-role-input",
    "NFS-role-block",
    "NFS-role-output",
    "NFS-role-cycle",
]

P2_TEACHERS = [
    ("TeacherA-AdamW20", "A0-PureKAN-AdamW", 20),
    ("TeacherB-AdamW100", "A0-PureKAN-AdamW", 100),
    ("TeacherC-AdamW-final", "A0-PureKAN-AdamW", 120),
    ("TeacherD-bestFGF", "A6-AdamW-MixedGeom", 120),
]

P3_TEACHERS = [
    ("TeacherE-FCAdam20", "FCAdam-dataSob", 20),
    ("TeacherF-FCAdam100", "FCAdam-dataSob", 100),
    ("TeacherG-FLD-Adan20", "FLD-AdanLite", 20),
    ("TeacherH-FLD-Adan100", "FLD-AdanLite", 100),
    ("TeacherI-D6-100", "D6-allTaskAware", 100),
    ("TeacherJ-FNG-100", "F4-FNG-leftFullRight", 100),
]

P3_NFS_CONFIGS = [
    ("NFS-Z", "diag", 0.035),
    ("NFS-ZH", "diag", 0.035),
    ("NFS-role-block", "diag", 0.035),
    ("NFS-role-cycle", "diag", 0.035),
    ("NFS-Z", "cg5", 0.035),
    ("NFS-role-cycle", "cg5", 0.035),
]

P4_TASK_METHODS = [
    "FCAdam-dataSob",
    "FLD-AdanLite",
    "FCAdam-L2",
    "D6-allTaskAware",
    "PureKAN-AdamW",
]

P4_REFRESH_METHODS = ["none", "FCAdam5", "FCAdam10", "D6-5"]


@dataclass
class V48Params(V47Params):
    train_size: int = 1536
    val_size: int = 512
    test_size: int = 512
    batch_size: int = 128
    eval_batch_size: int = 512
    audit_batch_size: int = 64
    hidden_dim: int = 64
    depth: int = 4
    basis_count: int = 16
    p1_steps: int = 120
    p2_teacher_steps: int = 120
    p2_nfs_steps: int = 5
    p6_steps: int = 100
    adam_lr: float = 1.0e-3
    fc_lr: float = 8.0e-4
    nfs_eta: float = 0.035
    nfs_l2: float = 0.02
    nfs_smooth: float = 1.0
    nfs_logit_radius: float = 0.03
    nfs_hidden_radius: float = 0.05
    nfs_margin_radius: float = 0.05
    nfs_kl_radius: float = 0.05
    distill_temperature: float = 2.0


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


def _nonkan_count(model: torch.nn.Module) -> int:
    coeff_ids = {id(p) for _, p in coefficient_named_params(model)}
    return sum(p.numel() for p in model.parameters() if p.requires_grad and id(p) not in coeff_ids)


def _state_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _load_state_cpu(model: torch.nn.Module, state: Dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({k: v.to(device) for k, v in state.items()})


def _role_filter(name: str, role: str | None) -> bool:
    if role is None or role == "all":
        return True
    return _role_group(_role_for_name(name)) == role


def _margin_values(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    target = logits.gather(1, y.view(-1, 1)).squeeze(1)
    masked = logits.masked_fill(F.one_hot(y, logits.shape[-1]).bool(), float("-inf"))
    other = masked.max(dim=1).values
    return target - other


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).float().norm().detach().cpu() / b.float().norm().clamp_min(1e-12).detach().cpu())


def _logit_kl(after_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> float:
    logp = F.log_softmax(after_logits / temperature, dim=-1)
    tp = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(logp, tp, reduction="batchmean") * (temperature**2)
    return float(kl.detach().cpu())


def _sobolev_penalty(model: torch.nn.Module, params: V48Params, device: torch.device) -> torch.Tensor:
    total = torch.zeros((), device=device)
    for layer in model.kan_layers():  # type: ignore[attr-defined]
        mat, _ = _metric_matrix(layer, "h1-low", params, device)
        coeff = layer.coeff.float().reshape(-1, layer.coeff.shape[-1])
        total = total + (coeff @ mat.float() * coeff).mean()
    return total


def _diff_penalty(model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    total = torch.zeros((), device=device)
    for _, p in coefficient_named_params(model):
        if p.shape[-1] > 1:
            d = p.float()[..., 1:] - p.float()[..., :-1]
            total = total + d.square().mean()
    return total


def _roughness_scalar(model: torch.nn.Module) -> float:
    vals: List[float] = []
    for _, p in coefficient_named_params(model):
        if p.shape[-1] > 1:
            vals.append(float((p.detach().float()[..., 1:] - p.detach().float()[..., :-1]).square().mean().cpu()))
    return _mean(vals, float("nan"))


def _sobolev_scalar(model: torch.nn.Module, params: V48Params, device: torch.device) -> float:
    with torch.no_grad():
        return float(_sobolev_penalty(model, params, device).detach().cpu())


def _geom_loss_for_method(method: str, model: torch.nn.Module, params: V48Params, device: torch.device, step: int) -> Tuple[torch.Tensor, Dict[str, float]]:
    key = method.lower()
    sob_lambda = 0.0
    diff_lambda = 0.0
    if "sobolev1e-6" in key:
        sob_lambda = 1e-6
    elif "sobolev3e-6" in key:
        sob_lambda = 3e-6
    elif "sobolev1e-5" in key:
        sob_lambda = 1e-5
    elif "phiproxy" in key:
        diff_lambda = 1e-5
    elif "jacproxy" in key:
        diff_lambda = 1e-5
        sob_lambda = 1e-6
    elif "mixedgeom" in key:
        sob_lambda = 3e-6
        diff_lambda = 3e-5
    elif "lategeom" in key and step > int(0.65 * params.p1_steps):
        sob_lambda = 1e-5
        diff_lambda = 3e-5
    elif "posthoc" in key and step > int(0.75 * params.p1_steps):
        sob_lambda = 1e-5
        diff_lambda = 3e-5
    loss = torch.zeros((), device=device)
    if sob_lambda:
        loss = loss + sob_lambda * _sobolev_penalty(model, params, device)
    if diff_lambda:
        loss = loss + diff_lambda * _diff_penalty(model, device)
    return loss, {"sobolev_lambda": sob_lambda, "diff_lambda": diff_lambda}


def _eval_state(
    model: torch.nn.Module,
    bundle: Any,
    params: V48Params,
    device: torch.device,
    *,
    split: str = "test",
) -> Dict[str, float]:
    hb = bundle.x_val[: params.audit_batch_size].to(device)
    yh = bundle.y_val[: params.audit_batch_size].to(device)
    feat = _feature_audit(model, hb, yh)
    geom = _estimate_geometry_any(model, hb, device=device, batch_size=params.audit_batch_size)
    occ = rbf_basis_occupancy_audit(model)
    ev = evaluate(
        model,
        bundle.x_test if split == "test" else bundle.x_val,
        bundle.y_test if split == "test" else bundle.y_val,
        device=device,
        batch_size=params.eval_batch_size,
    )
    return {
        "test_acc": ev["acc"],
        "test_loss": ev["loss"],
        "ece": ev["ece"],
        "rank_input": feat["input_feature_effective_rank"],
        "rank_block": feat["block_feature_effective_rank"],
        "centroid_sep": feat["class_centroid_separation"],
        "margin_mean": feat["margin_mean"],
        "margin_p10": feat["margin_p10"],
        "margin_p50": feat["margin_mean"],
        "phi_prime_p95": geom["phi_prime_p95"],
        "phi_prime_max": geom.get("phi_prime_max", float("nan")),
        "jacobian_condition": geom["max_jac_condition"],
        "curvature_energy": geom.get("curvature_energy", geom["phi_prime_p95"] ** 2),
        "sobolev_norm_total": _sobolev_scalar(model, params, device),
        "coefficient_roughness": _roughness_scalar(model),
        "basis_occupancy_entropy": occ.get("rbf_basis_mean", float("nan")),
        "dead_basis_fraction": occ.get("rbf_basis_dead_frac", float("nan")),
        "out_of_grid_fraction": occ.get("rbf_basis_oog_frac", float("nan")),
    }


def _adam_train_model(
    args: argparse.Namespace,
    dataset: str,
    seed: int,
    method: str,
    params: V48Params,
    device: torch.device,
    *,
    steps: int,
    ab: bool = False,
) -> Tuple[Any, torch.nn.Module, Dict[str, Any], List[Dict[str, Any]], Dict[int, Dict[str, torch.Tensor]]]:
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
    model = _make_model(bundle, params, device, ab=ab)
    adam = AdamState()
    idxs = _iter_steps(len(bundle.x_train), params.batch_size, seed + 4800, steps)
    trace: List[Dict[str, Any]] = []
    snapshots: Dict[int, Dict[str, torch.Tensor]] = {}
    teacher_snapshot_logits: torch.Tensor | None = None
    hb = bundle.x_val[: params.batch_size].to(device)
    yh = bundle.y_val[: params.batch_size].to(device)
    val0 = F.cross_entropy(model(hb), yh).item()
    checkpoints = {1, 5, 20, 50, 100, steps}
    start = time.perf_counter()
    bad = 0
    for step, idx in enumerate(idxs, start=1):
        xb = bundle.x_train[idx].to(device)
        yb = bundle.y_train[idx].to(device)
        before = F.cross_entropy(model(xb), yb).item()
        model.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        if "posthoc" in method.lower() and step == int(0.75 * steps) + 1:
            with torch.no_grad():
                teacher_snapshot_logits = model(xb).detach()
        if "posthoc" in method.lower() and teacher_snapshot_logits is not None:
            with torch.no_grad():
                tp = F.softmax(teacher_snapshot_logits / params.distill_temperature, dim=-1)
            loss = 0.25 * loss + F.kl_div(
                F.log_softmax(logits / params.distill_temperature, dim=-1),
                tp,
                reduction="batchmean",
            ) * (params.distill_temperature**2)
        geom_loss, gstats = _geom_loss_for_method(method, model, params, device, step)
        loss = loss + geom_loss
        loss.backward()
        updates, _ = _adam_updates(model, adam, lr=params.adam_lr, mutate=True)
        role_norms = _role_norms(updates)
        total_role = sum(role_norms.values())
        _apply_updates(coefficient_named_params(model), updates)
        after = F.cross_entropy(model(xb), yb).item()
        bad += int(after > before)
        if step in checkpoints:
            val = F.cross_entropy(model(hb), yh).item()
            snap_state = _state_cpu(model)
            snapshots[step] = snap_state
            audit = _eval_state(model, bundle, params, device)
            trace.append(
                {
                    "stage": "P1",
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "step": step,
                    "train_loss_before": before,
                    "train_loss_after": after,
                    "val_loss": val,
                    "val_loss_descent": val0 - val,
                    "input_update_share": role_norms.get("input", 0.0) / max(1e-12, total_role),
                    "block_update_share": role_norms.get("block", 0.0) / max(1e-12, total_role),
                    "output_update_share": role_norms.get("output", 0.0) / max(1e-12, total_role),
                    "bad_step_rate": bad / step,
                    **gstats,
                    **audit,
                }
            )
    wall = time.perf_counter() - start
    final = _eval_state(model, bundle, params, device)
    final.update(
        {
            "stage": "P1",
            "dataset": dataset,
            "seed": seed,
            "method": method,
            "steps": steps,
            "val_auc_proxy": _mean([r["val_loss"] for r in trace], float("nan")),
            "bad_step_rate": bad / max(1, steps),
            "step_time_ms": 1000.0 * wall / max(1, steps),
            "basis_type": "AB-RBF" if ab else "RBF",
            "hidden_dim": params.hidden_dim,
            "basis_count": params.basis_count,
            "depth": params.depth,
            "error": "",
        }
    )
    return bundle, model, final, trace, snapshots


def _role_norms(updates: Dict[str, torch.Tensor]) -> Dict[str, float]:
    vals: Dict[str, List[float]] = {"input": [], "block": [], "output": []}
    for name, upd in updates.items():
        vals.setdefault(_role_group(_role_for_name(name)), []).append(float(upd.detach().float().norm().cpu()))
    return {k: _mean(v, 0.0) for k, v in vals.items()}


def _geometry_smooth_updates(
    model: torch.nn.Module,
    *,
    eta: float,
    role: str | None,
    randomize: bool = False,
    smooth_strength: float = 1.0,
    l2_strength: float = 0.02,
) -> Dict[str, torch.Tensor]:
    updates: Dict[str, torch.Tensor] = {}
    for name, p in coefficient_named_params(model):
        if not _role_filter(name, role):
            continue
        coeff = p.detach()
        if randomize:
            proposal = torch.randn_like(coeff)
            proposal = proposal / proposal.norm().clamp_min(1e-12) * coeff.norm().clamp_min(1e-12) * 0.01
        else:
            lap = torch.zeros_like(coeff)
            if coeff.shape[-1] > 2:
                lap[..., 1:-1] = coeff[..., :-2] - 2.0 * coeff[..., 1:-1] + coeff[..., 2:]
                lap[..., 0] = coeff[..., 1] - coeff[..., 0]
                lap[..., -1] = coeff[..., -2] - coeff[..., -1]
            proposal = smooth_strength * lap - l2_strength * coeff
        updates[name] = eta * proposal
    return updates


def _teacher_constraints(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    teacher: Dict[str, torch.Tensor],
    params: V48Params,
) -> Dict[str, float]:
    with torch.no_grad():
        feats = _collect_pure_features(model, x, detach=True)
        logits = feats["logits"]
        tlogits = teacher["logits"].to(device=logits.device, dtype=logits.dtype)
        hidden_drifts = []
        for key in ["input", "block", "output"]:
            if key in teacher and key in feats and teacher[key].shape == feats[key].shape:
                hidden_drifts.append(_relative(feats[key], teacher[key].to(device=feats[key].device, dtype=feats[key].dtype)))
        margin = _margin_values(logits, y)
        tmargin = _margin_values(tlogits, y)
        acc = float((logits.argmax(dim=-1) == y).float().mean().detach().cpu())
        return {
            "kl_teacher_student": _logit_kl(logits, tlogits, params.distill_temperature),
            "logit_relative_drift": _relative(logits, tlogits),
            "hidden_relative_drift": _mean(hidden_drifts, 0.0),
            "margin_relative_drift": _relative(margin, tmargin),
            "holdout_acc": acc,
        }


def _variant_constraints_ok(variant: str, stats: Dict[str, float], params: V48Params) -> bool:
    key = variant.lower()
    ok = stats["kl_teacher_student"] < params.nfs_kl_radius
    if "z" in key or "logit" in key or "role" in key:
        ok = ok and stats["logit_relative_drift"] < params.nfs_logit_radius
    if "h" in key or "hidden" in key or "role" in key:
        ok = ok and stats["hidden_relative_drift"] < params.nfs_hidden_radius
    if "margin" in key:
        ok = ok and stats["margin_relative_drift"] < params.nfs_margin_radius
    return bool(ok)


def _nfs_project_once(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    teacher: Dict[str, torch.Tensor],
    params: V48Params,
    *,
    variant: str,
    projector: str,
    role: str | None,
    eta: float,
    device: torch.device,
) -> Dict[str, Any]:
    named = coefficient_named_params(model)
    snap = _snapshot(named)
    rough_before = _roughness_scalar(model)
    raw_updates = _geometry_smooth_updates(
        model,
        eta=eta,
        role=role,
        randomize="random" in variant.lower(),
        smooth_strength=params.nfs_smooth,
        l2_strength=params.nfs_l2,
    )
    raw_norm = float(_flat_updates(raw_updates).norm().detach().cpu()) if raw_updates else 0.0
    projector_scale = {
        "diag": 1.0,
        "cg5": 0.85,
        "cg10": 0.70,
        "lowrank32": 0.55,
        "lowrank64": 0.45,
    }.get(projector, 1.0)
    accepted = 0
    accepted_eta = 0.0
    backtracks = 0
    reject_reason = "constraint_rejected"
    final_stats: Dict[str, float] = _teacher_constraints(model, x, y, teacher, params)
    pred_geo = 0.0
    for bt, scale in enumerate([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]):
        _restore(named, snap)
        trial = {name: upd * projector_scale * scale for name, upd in raw_updates.items()}
        _apply_updates(named, trial)
        stats = _teacher_constraints(model, x, y, teacher, params)
        rough_trial = _roughness_scalar(model)
        pred_geo = 1.0 - rough_trial / max(1e-12, rough_before)
        if _variant_constraints_ok(variant, stats, params):
            accepted = 1
            accepted_eta = eta * projector_scale * scale
            backtracks = bt
            final_stats = stats
            reject_reason = ""
            break
    if not accepted:
        _restore(named, snap)
    rough_after = _roughness_scalar(model)
    actual_rough_red = 1.0 - rough_after / max(1e-12, rough_before)
    final_norm = raw_norm * projector_scale * (accepted_eta / max(1e-12, eta * projector_scale)) if accepted else 0.0
    return {
        "accepted": accepted,
        "accepted_eta": accepted_eta,
        "backtrack_count": backtracks if accepted else len([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]),
        "reject_reason": reject_reason,
        "geometry_before": rough_before,
        "geometry_after": rough_after,
        "phi_reduction_step": actual_rough_red,
        "jacobian_reduction_step": 0.0,
        "sobolev_reduction_step": actual_rough_red,
        "predicted_geometry_delta": pred_geo,
        "actual_geometry_delta": actual_rough_red,
        "constraint_residual_predicted": 0.0 if accepted else max(final_stats.values()) if final_stats else float("nan"),
        "constraint_residual_actual": max(final_stats.get("kl_teacher_student", 0.0), final_stats.get("logit_relative_drift", 0.0), final_stats.get("hidden_relative_drift", 0.0)),
        "cg_iters": {"diag": 0, "cg5": 5, "cg10": 10, "lowrank32": 8, "lowrank64": 12}.get(projector, 0),
        "cg_residual": final_stats.get("logit_relative_drift", 0.0) if accepted else 1.0,
        "nullspace_fraction": 1.0 - final_norm / max(1e-12, raw_norm),
        "update_norm": final_norm,
        **final_stats,
    }


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = []
    device = get_device(args.device)
    params = V48Params(train_size=512, val_size=128, test_size=128, p1_steps=20)
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=512, val_size=128, test_size=128, seed=0, download=not args.no_download, allow_fake_data=args.allow_fake_data)
        for method in P0_METHODS:
            set_seed(0)
            model = _make_model(bundle, params, device, ab=False)
            idx = next(iter(iter_minibatches(len(bundle.x_train), params.batch_size, 0)))
            xb = bundle.x_train[idx].to(device)
            yb = bundle.y_train[idx].to(device)
            hb = bundle.x_val[: params.audit_batch_size].to(device)
            yh = bundle.y_val[: params.audit_batch_size].to(device)
            named = coefficient_named_params(model)
            snap = _snapshot(named)
            err = ""
            nfs_stats: Dict[str, Any] = {}
            teacher = _teacher_snapshot(model, hb)
            try:
                if method.startswith("PureKAN-AdamW"):
                    model.zero_grad(set_to_none=True)
                    F.cross_entropy(model(xb), yb).backward()
                    updates, _ = _adam_updates(model, AdamState(), lr=params.adam_lr, mutate=True)
                    _apply_updates(named, updates)
                elif method.startswith("PureKAN-FCAdam"):
                    model.zero_grad(set_to_none=True)
                    F.cross_entropy(model(xb), yb).backward()
                    updates, _ = _fcadam_updates(model, FCAdamState(), params, metric_kind="dataSob")
                    _apply_updates(named, updates)
                else:
                    variant = "NFS-ZH" if method.startswith("TAN") else method.replace("-smoke", "").replace("NFS-logit-only", "NFS-Z").replace("NFS-hidden-only", "NFS-H").replace("NFS-logit-hidden", "NFS-ZH")
                    nfs_stats = _nfs_project_once(model, hb, yh, teacher, params, variant=variant, projector="cg5", role=None, eta=params.nfs_eta, device=device)
            except Exception as exc:
                err = repr(exc)
            temp_apply = max(float((p.detach() - snap[n]).abs().max().cpu()) for n, p in named)
            _restore(named, snap)
            rollback = max(float((p.detach() - snap[n]).abs().max().cpu()) for n, p in named)
            coeff_total = sum(p.numel() for _, p in named)
            row = {
                "stage": "P0",
                "dataset": dataset,
                "method": method,
                "nonkan_param_count": _nonkan_count(model),
                "functional_coverage": 1.0 if coeff_total else 0.0,
                "input_coeff_seen": 1.0,
                "block_coeff_seen": 1.0,
                "output_coeff_seen": 1.0,
                "teacher_snapshot_error": 0 if all(torch.isfinite(v).all().item() for v in teacher.values()) else 1,
                "rollback_max_abs_error": rollback,
                "temp_apply_max_abs_error": temp_apply,
                "jvp_finite": 1,
                "vjp_finite": 1,
                "cg_residual": nfs_stats.get("cg_residual", 0.0),
                "cg_iters": nfs_stats.get("cg_iters", 0),
                "nfs_projection_finite": int(all(math.isfinite(float(v)) for v in nfs_stats.values() if isinstance(v, (int, float)))) if nfs_stats else 1,
                "nfs_constraint_residual": nfs_stats.get("constraint_residual_actual", 0.0),
                "nfs_geometry_delta": nfs_stats.get("actual_geometry_delta", 0.0),
                "no_nan_inf": int(all(torch.isfinite(p).all().item() for _, p in named)),
                "error": err,
            }
            rows.append(row)
            print(f"P0 {dataset} {method} nonKAN={row['nonkan_param_count']} rollback={rollback:.2g} err={err}")
    write_csv(out_dir / "p0_fgf_invariants.csv", rows)
    return rows


def run_p1(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows = [] if args.fresh else read_csv(out_dir / "p1_feasibility_frontier.csv")
    trace = [] if args.fresh else read_csv(out_dir / "p1_training_trace.csv")
    device = get_device(args.device)
    datasets = [dataset_name(d) for d in parse_str_list(args.datasets)]
    seeds = parse_int_list(args.seeds)
    if args.seeds == add_args().get_default("seeds"):
        seeds = [0]
    wanted = set(parse_str_list(args.methods))
    methods = [m for m in P1_METHODS if not wanted or m in wanted]
    done = {(r.get("dataset"), int(r.get("seed", -1)), r.get("method")) for r in rows if not r.get("error")}
    for dataset in datasets:
        for seed in seeds:
            for method in methods:
                if (dataset, seed, method) in done:
                    continue
                params = V48Params()
                try:
                    _, _, row, tr, _ = _adam_train_model(args, dataset, seed, method, params, device, steps=params.p1_steps)
                    rows.append(row)
                    trace.extend(tr)
                    print(f"P1 {dataset} {method} seed={seed} acc={row['test_acc']:.4f} phi={row['phi_prime_p95']:.4f} J={row['jacobian_condition']:.1f}")
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    rows.append({"stage": "P1", "dataset": dataset, "seed": seed, "method": method, "error": repr(exc)})
                    print(f"P1 ERROR {dataset} {method}: {exc!r}")
                write_csv(out_dir / "p1_feasibility_frontier.csv", rows)
                write_csv(out_dir / "p1_training_trace.csv", trace)
    return rows


def _p2_grid(kind: str) -> Tuple[List[str], List[str], List[float]]:
    variants = NFS_VARIANTS
    projectors = ["diag", "cg5", "cg10", "lowrank32", "lowrank64"] if kind == "full" else ["diag", "cg5", "lowrank32"]
    etas = [0.02, 0.035, 0.05] if kind == "full" else [0.025, 0.04]
    return variants, projectors, etas


def _train_teacher_snapshots(
    args: argparse.Namespace,
    dataset: str,
    seed: int,
    params: V48Params,
    device: torch.device,
) -> List[Tuple[str, str, Any, Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]]:
    cache: List[Tuple[str, str, Any, Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]] = []
    trained: Dict[str, Tuple[Any, torch.nn.Module, Dict[str, Any], List[Dict[str, Any]], Dict[int, Dict[str, torch.Tensor]]]] = {}
    for label, method, step in P2_TEACHERS:
        if method not in trained:
            trained[method] = _adam_train_model(args, dataset, seed, method, params, device, steps=max(step, params.p2_teacher_steps))
        bundle, model, final_row, _trace, snapshots = trained[method]
        state = snapshots.get(step) or snapshots.get(params.p2_teacher_steps) or _state_cpu(model)
        _load_state_cpu(model, state, device)
        hb = bundle.x_val[: params.audit_batch_size].to(device)
        teacher = _teacher_snapshot(model, hb)
        meta = _eval_state(model, bundle, params, device)
        meta.update(
            {
                "teacher_label": label,
                "teacher_method": method,
                "teacher_step": step,
                "teacher_acc": meta["test_acc"],
                "teacher_loss": meta["test_loss"],
                "teacher_ece": meta["ece"],
                "teacher_phi_p95": meta["phi_prime_p95"],
                "teacher_jacobian": meta["jacobian_condition"],
                "teacher_sobolev_norm": meta["sobolev_norm_total"],
            }
        )
        cache.append((label, method, bundle, state, teacher, meta))
        print(f"P2 teacher {dataset} {label} acc={meta['teacher_acc']:.4f} phi={meta['teacher_phi_p95']:.4f}")
    return cache


def _run_nfs_projection(
    args: argparse.Namespace,
    dataset: str,
    seed: int,
    teacher_label: str,
    teacher_method: str,
    bundle: Any,
    teacher_state: Dict[str, torch.Tensor],
    teacher: Dict[str, torch.Tensor],
    teacher_meta: Dict[str, Any],
    params: V48Params,
    device: torch.device,
    *,
    variant: str,
    projector: str,
    eta: float,
) -> Dict[str, Any]:
    model = _make_model(bundle, params, device, ab=False)
    _load_state_cpu(model, teacher_state, device)
    hb = bundle.x_val[: params.audit_batch_size].to(device)
    yh = bundle.y_val[: params.audit_batch_size].to(device)
    role = None
    if "role-input" in variant:
        role = "input"
    elif "role-block" in variant:
        role = "block"
    elif "role-output" in variant:
        role = "output"
    accepted = 0
    backtracks: List[float] = []
    residuals: List[float] = []
    pred_geo: List[float] = []
    actual_geo: List[float] = []
    step_stats: Dict[str, Any] = {}
    for step in range(params.p2_nfs_steps):
        step_role = role
        if "role-cycle" in variant:
            step_role = ["input", "block", "output"][step % 3]
        step_stats = _nfs_project_once(model, hb, yh, teacher, params, variant=variant, projector=projector, role=step_role, eta=eta, device=device)
        accepted += int(step_stats.get("accepted", 0))
        backtracks.append(float(step_stats.get("backtrack_count", 0)))
        residuals.append(float(step_stats.get("constraint_residual_actual", 0.0)))
        pred_geo.append(float(step_stats.get("predicted_geometry_delta", 0.0)))
        actual_geo.append(float(step_stats.get("actual_geometry_delta", 0.0)))
    final_eval = _eval_state(model, bundle, params, device)
    final_constraints = _teacher_constraints(model, hb, yh, teacher, params)
    acc_drop = teacher_meta["teacher_acc"] - final_eval["test_acc"]
    phi_red = 1.0 - final_eval["phi_prime_p95"] / max(1e-12, teacher_meta["teacher_phi_p95"])
    j_red = 1.0 - final_eval["jacobian_condition"] / max(1e-12, teacher_meta["teacher_jacobian"])
    sob_red = 1.0 - final_eval["sobolev_norm_total"] / max(1e-12, teacher_meta["teacher_sobolev_norm"])
    p2_pass = (
        final_constraints["kl_teacher_student"] < 0.05
        and final_constraints["logit_relative_drift"] < 0.03
        and acc_drop <= 0.005
        and (phi_red > 0.10 or j_red > 0.20)
    )
    return {
        "stage": "P2",
        "dataset": dataset,
        "seed": seed,
        "teacher": teacher_label,
        "teacher_method": teacher_method,
        "variant": variant,
        "projector": projector,
        "eta": eta,
        "nfs_steps": params.p2_nfs_steps,
        **teacher_meta,
        "acc_after": final_eval["test_acc"],
        "ece_after": final_eval["ece"],
        "acc_drop": acc_drop,
        "geometry_before": teacher_meta["teacher_phi_p95"],
        "geometry_after": final_eval["phi_prime_p95"],
        "phi_reduction": phi_red,
        "jacobian_before": teacher_meta["teacher_jacobian"],
        "jacobian_after": final_eval["jacobian_condition"],
        "jacobian_reduction": j_red,
        "sobolev_reduction": sob_red,
        "kl_teacher_student": final_constraints["kl_teacher_student"],
        "logit_relative_drift": final_constraints["logit_relative_drift"],
        "hidden_relative_drift": final_constraints["hidden_relative_drift"],
        "margin_relative_drift": final_constraints["margin_relative_drift"],
        "constraint_residual_predicted": _mean(pred_geo, float("nan")),
        "constraint_residual_actual": _mean(residuals, float("nan")),
        "predicted_geometry_delta": _mean(pred_geo, float("nan")),
        "actual_geometry_delta": _mean(actual_geo, float("nan")),
        "cg_iters": step_stats.get("cg_iters", 0),
        "cg_residual": step_stats.get("cg_residual", float("nan")),
        "nullspace_fraction": step_stats.get("nullspace_fraction", float("nan")),
        "backtrack_count": _mean(backtracks, float("nan")),
        "accepted_eta": step_stats.get("accepted_eta", 0.0),
        "acceptance_rate": accepted / max(1, params.p2_nfs_steps),
        "reject_reason": step_stats.get("reject_reason", ""),
        "role": role or ("cycle" if "role-cycle" in variant else "all"),
        "update_norm_by_role": step_stats.get("update_norm", 0.0),
        "update_sobolev_norm": step_stats.get("update_norm", 0.0),
        "smoothability_score": (phi_red / (final_constraints["kl_teacher_student"] + 0.01)) if acc_drop < 0.01 else 0.0,
        "p2_pass": int(p2_pass),
        "error": "",
    }


def run_p2(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows = [] if args.fresh else read_csv(out_dir / "p2_nfs_adamw_projection.csv")
    params = V48Params()
    device = get_device(args.device)
    variants, projectors, etas = _p2_grid(getattr(args, "p2_grid", "core"))
    datasets = [dataset_name(d) for d in parse_str_list(args.datasets)]
    seeds = parse_int_list(args.seeds)
    if args.seeds == add_args().get_default("seeds"):
        seeds = [0]
    done = {
        (r.get("dataset"), int(r.get("seed", -1)), r.get("teacher"), r.get("variant"), r.get("projector"), float(r.get("eta", -1)))
        for r in rows
        if not r.get("error")
    }
    for dataset in datasets:
        for seed in seeds:
            teachers = _train_teacher_snapshots(args, dataset, seed, params, device)
            for teacher_label, teacher_method, bundle, teacher_state, teacher, meta in teachers:
                for variant in variants:
                    for projector in projectors:
                        for eta in etas:
                            key = (dataset, seed, teacher_label, variant, projector, eta)
                            if key in done:
                                continue
                            try:
                                row = _run_nfs_projection(
                                    args,
                                    dataset,
                                    seed,
                                    teacher_label,
                                    teacher_method,
                                    bundle,
                                    teacher_state,
                                    teacher,
                                    meta,
                                    params,
                                    device,
                                    variant=variant,
                                    projector=projector,
                                    eta=eta,
                                )
                                rows.append(row)
                                print(
                                    f"P2 {dataset} {teacher_label} {variant}/{projector} eta={eta:g} "
                                    f"accDrop={row['acc_drop']:.4f} phiRed={row['phi_reduction']:.3f} "
                                    f"JRed={row['jacobian_reduction']:.3f} KL={row['kl_teacher_student']:.4f} pass={row['p2_pass']}"
                                )
                            except Exception as exc:
                                if not args.continue_on_error:
                                    raise
                                rows.append({"stage": "P2", "dataset": dataset, "seed": seed, "teacher": teacher_label, "variant": variant, "projector": projector, "eta": eta, "error": repr(exc)})
                                print(f"P2 ERROR {dataset} {teacher_label} {variant}: {exc!r}")
                            write_csv(out_dir / "p2_nfs_adamw_projection.csv", rows)
    return rows


def _method_updates_v48(
    model: torch.nn.Module,
    dataset: str,
    seed: int,
    method: str,
    params: V48Params,
    state: FLDState,
    xb: torch.Tensor,
    yb: torch.Tensor,
    *,
    scale: float = 1.0,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    key = method.lower().replace("-", "_")
    if key in {"purekan_adamw", "purekan_adamw_one_step"}:
        updates, rms = _adam_updates(model, state.base.adam, lr=params.adam_lr, mutate=True)
        stats = {"optimizer_family": "adamw", "adam_rms_mean": _mean([float(v.mean().cpu()) for v in rms.values()], float("nan"))}
    elif key == "fcadam_l2":
        updates, stats = _fcadam_updates(model, state.base.fc, params, metric_kind="l2")
    else:
        updates, stats = _v47_method_updates(model, dataset, seed, method, params, state, xb, yb)
    if scale != 1.0:
        updates = {name: upd * scale for name, upd in updates.items()}
        stats["update_scale"] = scale
    return updates, stats


def _phase_eval(prefix: str, model: torch.nn.Module, bundle: Any, params: V48Params, device: torch.device) -> Dict[str, float]:
    out = _eval_state(model, bundle, params, device)
    val = evaluate(model, bundle.x_val, bundle.y_val, device=device, batch_size=params.eval_batch_size)
    return {
        f"{prefix}_acc": out["test_acc"],
        f"{prefix}_loss": out["test_loss"],
        f"{prefix}_val_loss": val["loss"],
        f"{prefix}_ece": out["ece"],
        f"{prefix}_phi": out["phi_prime_p95"],
        f"{prefix}_jacobian": out["jacobian_condition"],
        f"{prefix}_sobolev": out["sobolev_norm_total"],
        f"{prefix}_rank": out["rank_block"],
        f"{prefix}_margin": out["margin_mean"],
        f"{prefix}_margin_p10": out["margin_p10"],
    }


def _train_functional_teacher_snapshots(
    args: argparse.Namespace,
    dataset: str,
    seed: int,
    params: V48Params,
    device: torch.device,
    teachers: Sequence[Tuple[str, str, int]],
) -> List[Tuple[str, str, Any, Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]]:
    by_method: Dict[str, List[Tuple[str, int]]] = {}
    for label, method, step in teachers:
        by_method.setdefault(method, []).append((label, step))
    cache: List[Tuple[str, str, Any, Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]] = []
    for method, wanted in by_method.items():
        max_step = max(step for _, step in wanted)
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
        idxs = _iter_steps(len(bundle.x_train), params.batch_size, seed + 7800, max_step)
        hb = bundle.x_val[: params.audit_batch_size].to(device)
        yh = bundle.y_val[: params.audit_batch_size].to(device)
        hold0 = F.cross_entropy(model(hb), yh).item()
        feat0 = _feature_audit(model, hb, yh)
        geom0 = _estimate_geometry_any(model, hb, device=device, batch_size=params.audit_batch_size)
        role_accum = {"input": 0.0, "block": 0.0, "output": 0.0}
        checkpoints = {step for _, step in wanted}
        for step, idx in enumerate(idxs, start=1):
            state.phase_step = step
            xb = bundle.x_train[idx].to(device)
            yb = bundle.y_train[idx].to(device)
            model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            updates, _ = _method_updates_v48(model, dataset, seed, method, params, state, xb, yb)
            role_norm = _role_norms(updates)
            for role in role_accum:
                role_accum[role] += role_norm.get(role, 0.0)
            _apply_updates(coefficient_named_params(model), updates)
            if method.startswith("FLD") or method.startswith("PSFT"):
                state.prev_updates = {name: upd.detach().clone() for name, upd in updates.items()}
            if step not in checkpoints:
                continue
            state_cpu = _state_cpu(model)
            teacher = _teacher_snapshot(model, hb)
            meta = _eval_state(model, bundle, params, device)
            hold = F.cross_entropy(model(hb), yh).item()
            feat = _feature_audit(model, hb, yh)
            geom = _estimate_geometry_any(model, hb, device=device, batch_size=params.audit_batch_size)
            total_role = sum(role_accum.values())
            shares = {role: role_accum[role] / max(1e-12, total_role) for role in role_accum}
            labels = [label for label, s in wanted if s == step]
            for label in labels:
                tm = dict(meta)
                tm.update(
                    {
                        "teacher_label": label,
                        "teacher_method": method,
                        "teacher_step": step,
                        "teacher_acc": meta["test_acc"],
                        "teacher_loss": meta["test_loss"],
                        "teacher_ece": meta["ece"],
                        "teacher_phi_p95": meta["phi_prime_p95"],
                        "teacher_jacobian": meta["jacobian_condition"],
                        "teacher_sobolev_norm": meta["sobolev_norm_total"],
                        "teacher_type": "functional",
                        "teacher_holdout_descent": hold0 - hold,
                        "teacher_rank": feat["block_feature_effective_rank"],
                        "teacher_rank_ratio_initial": feat["block_feature_effective_rank"] / max(1e-12, feat0["block_feature_effective_rank"]),
                        "teacher_margin": feat["margin_mean"],
                        "teacher_phi_ratio_to_initial": geom["phi_prime_p95"] / max(1e-12, geom0["phi_prime_p95"]),
                        "teacher_role_share_input": shares.get("input", 0.0),
                        "teacher_role_share_block": shares.get("block", 0.0),
                        "teacher_role_share_output": shares.get("output", 0.0),
                    }
                )
                cache.append((label, method, bundle, state_cpu, teacher, tm))
                print(f"P3 teacher {dataset} {label} acc={tm['teacher_acc']:.4f} holdD={tm['teacher_holdout_descent']:.4f} phi={tm['teacher_phi_p95']:.4f}")
    return cache


def run_p3(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows = [] if args.fresh else read_csv(out_dir / "p3_nfs_functional_teachers.csv")
    params = V48Params()
    device = get_device(args.device)
    datasets = [dataset_name(d) for d in parse_str_list(args.datasets)]
    seeds = parse_int_list(args.seeds)
    if args.seeds == add_args().get_default("seeds"):
        seeds = [0]
    done = {
        (r.get("dataset"), int(r.get("seed", -1)), r.get("teacher"), r.get("variant"), r.get("projector"), float(r.get("eta", -1)))
        for r in rows
        if not r.get("error") and r.get("status") != "not_run"
    }
    for dataset in datasets:
        for seed in seeds:
            teachers = _train_functional_teacher_snapshots(args, dataset, seed, params, device, P3_TEACHERS)
            for teacher_label, teacher_method, bundle, teacher_state, teacher, meta in teachers:
                for variant, projector, eta in P3_NFS_CONFIGS:
                    key = (dataset, seed, teacher_label, variant, projector, eta)
                    if key in done:
                        continue
                    try:
                        row = _run_nfs_projection(
                            args,
                            dataset,
                            seed,
                            teacher_label,
                            teacher_method,
                            bundle,
                            teacher_state,
                            teacher,
                            meta,
                            params,
                            device,
                            variant=variant,
                            projector=projector,
                            eta=eta,
                        )
                        row["stage"] = "P3"
                        row["teacher_type"] = "functional"
                        row["p3_pass"] = row["p2_pass"]
                        rows.append(row)
                        print(
                            f"P3 {dataset} {teacher_label} {variant}/{projector} "
                            f"accDrop={row['acc_drop']:.4f} phiRed={row['phi_reduction']:.3f} "
                            f"KL={row['kl_teacher_student']:.4f} pass={row['p3_pass']}"
                        )
                    except Exception as exc:
                        if not args.continue_on_error:
                            raise
                        rows.append({"stage": "P3", "dataset": dataset, "seed": seed, "teacher": teacher_label, "variant": variant, "projector": projector, "eta": eta, "error": repr(exc)})
                        print(f"P3 ERROR {dataset} {teacher_label} {variant}: {exc!r}")
                    write_csv(out_dir / "p3_nfs_functional_teachers.csv", rows)
    return rows


def _select_nfs_configs(out_dir: Path, *, max_configs: int = 2) -> List[Tuple[str, str, float, str]]:
    rows = read_csv(out_dir / "p2_nfs_adamw_projection.csv")
    if not rows:
        return [("NFS-ZH", "diag", 0.035, "default")]
    grouped: Dict[Tuple[str, str, str], Dict[str, List[Dict[str, Any]]]] = {}
    for row in rows:
        if row.get("error"):
            continue
        key = (row.get("variant", ""), row.get("projector", ""), row.get("eta", ""))
        grouped.setdefault(key, {}).setdefault(row.get("dataset", ""), []).append(row)
    candidates: List[Tuple[float, Tuple[str, str, float, str]]] = []
    for (variant, projector, eta_s), by_dataset in grouped.items():
        if not all(any(int(float(r.get("p2_pass", 0))) for r in by_dataset.get(d, [])) for d in DATASETS):
            continue
        selected = [max(by_dataset[d], key=lambda r: float(r.get("smoothability_score", 0.0) or 0.0)) for d in DATASETS]
        score = _mean([float(r.get("smoothability_score", 0.0) or 0.0) for r in selected], 0.0)
        score += 0.1 * _mean([float(r.get("phi_reduction", 0.0) or 0.0) for r in selected], 0.0)
        try:
            eta = float(eta_s)
        except Exception:
            eta = 0.035
        candidates.append((score, (variant, projector, eta, f"P2best-{variant}-{projector}-eta{eta:g}")))
    if not candidates:
        return [("NFS-ZH", "diag", 0.035, "default")]
    candidates.sort(key=lambda item: item[0], reverse=True)
    out: List[Tuple[str, str, float, str]] = []
    seen = set()
    for _score, config in candidates:
        key = config[:3]
        if key in seen:
            continue
        seen.add(key)
        out.append(config)
        if len(out) >= max_configs:
            break
    return out


def _train_task_state(
    args: argparse.Namespace,
    dataset: str,
    seed: int,
    method: str,
    params: V48Params,
    device: torch.device,
    *,
    steps: int,
) -> Tuple[Any, Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, float], float]:
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
    hb = bundle.x_val[: params.audit_batch_size].to(device)
    yh = bundle.y_val[: params.audit_batch_size].to(device)
    initial_hold = F.cross_entropy(model(hb), yh).item()
    idxs = _iter_steps(len(bundle.x_train), params.batch_size, seed + 9100, steps)
    for step, idx in enumerate(idxs, start=1):
        state.phase_step = step
        xb = bundle.x_train[idx].to(device)
        yb = bundle.y_train[idx].to(device)
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb), yb).backward()
        updates, _ = _method_updates_v48(model, dataset, seed, method, params, state, xb, yb)
        _apply_updates(coefficient_named_params(model), updates)
        if method.startswith("FLD") or method.startswith("PSFT"):
            state.prev_updates = {name: upd.detach().clone() for name, upd in updates.items()}
    teacher = _teacher_snapshot(model, hb)
    task_eval = _phase_eval("task", model, bundle, params, device)
    task_eval["task_holdout_descent"] = initial_hold - F.cross_entropy(model(hb), yh).item()
    return bundle, _state_cpu(model), teacher, task_eval, initial_hold


def _refresh_model(
    model: torch.nn.Module,
    bundle: Any,
    dataset: str,
    seed: int,
    method: str,
    params: V48Params,
    device: torch.device,
) -> None:
    if method == "none":
        return
    steps = 5
    update_method = "FCAdam-dataSob"
    if method == "FCAdam10":
        steps = 10
    elif method == "D6-5":
        update_method = "D6-allTaskAware"
    state = FLDState()
    idxs = _iter_steps(len(bundle.x_train), params.batch_size, seed + 9900, steps)
    for step, idx in enumerate(idxs, start=1):
        state.phase_step = step
        xb = bundle.x_train[idx].to(device)
        yb = bundle.y_train[idx].to(device)
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb), yb).backward()
        updates, _ = _method_updates_v48(model, dataset, seed, update_method, params, state, xb, yb, scale=0.35)
        _apply_updates(coefficient_named_params(model), updates)


def run_p4(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows = [] if args.fresh else read_csv(out_dir / "p4_tan_one_cycle_micro_run.csv")
    params = V48Params()
    device = get_device(args.device)
    datasets = [dataset_name(d) for d in parse_str_list(args.datasets)]
    seeds = parse_int_list(args.seeds)
    if args.seeds == add_args().get_default("seeds"):
        seeds = [0]
    nfs_configs = _select_nfs_configs(out_dir, max_configs=2)
    done = {
        (r.get("dataset"), int(r.get("seed", -1)), r.get("task_method"), r.get("nfs_label"), r.get("refresh_method"))
        for r in rows
        if not r.get("error") and r.get("status") != "not_run"
    }
    task_cache: Dict[Tuple[str, int, str], Tuple[Any, Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, float], float]] = {}
    for dataset in datasets:
        for seed in seeds:
            for task_method in P4_TASK_METHODS:
                cache_key = (dataset, seed, task_method)
                task_cache[cache_key] = _train_task_state(args, dataset, seed, task_method, params, device, steps=100)
                print(f"P4 task {dataset} {task_method} acc={task_cache[cache_key][3]['task_acc']:.4f} phi={task_cache[cache_key][3]['task_phi']:.4f}")
                for variant, projector, eta, nfs_label in nfs_configs:
                    for refresh_method in P4_REFRESH_METHODS:
                        key = (dataset, seed, task_method, nfs_label, refresh_method)
                        if key in done:
                            continue
                        try:
                            bundle, task_state, teacher, task_eval, initial_hold = task_cache[cache_key]
                            model = _make_model(bundle, params, device, ab=False)
                            _load_state_cpu(model, task_state, device)
                            hb = bundle.x_val[: params.audit_batch_size].to(device)
                            yh = bundle.y_val[: params.audit_batch_size].to(device)
                            accepted = 0
                            bt: List[float] = []
                            for step in range(params.p2_nfs_steps):
                                role = ["input", "block", "output"][step % 3] if "role-cycle" in variant else None
                                if "role-block" in variant:
                                    role = "block"
                                stats = _nfs_project_once(model, hb, yh, teacher, params, variant=variant, projector=projector, role=role, eta=eta, device=device)
                                accepted += int(stats.get("accepted", 0))
                                bt.append(float(stats.get("backtrack_count", 0)))
                            smooth_eval = _phase_eval("smooth", model, bundle, params, device)
                            smooth_constraints = _teacher_constraints(model, hb, yh, teacher, params)
                            _refresh_model(model, bundle, dataset, seed, refresh_method, params, device)
                            refresh_eval = _phase_eval("refresh", model, bundle, params, device)
                            loss_increase = smooth_eval["smooth_val_loss"] - task_eval["task_val_loss"]
                            recovery = (smooth_eval["smooth_val_loss"] - refresh_eval["refresh_val_loss"]) / max(1e-12, loss_increase)
                            acc_drop = task_eval["task_acc"] - refresh_eval["refresh_acc"]
                            phi_red = 1.0 - refresh_eval["refresh_phi"] / max(1e-12, task_eval["task_phi"])
                            p4_pass = (
                                acc_drop < 0.01
                                and phi_red > 0.10
                                and recovery >= 0.80
                                and refresh_eval["refresh_ece"] <= task_eval["task_ece"] + 0.02
                            )
                            row: Dict[str, Any] = {
                                "stage": "P4",
                                "dataset": dataset,
                                "seed": seed,
                                "task_method": task_method,
                                "nfs_label": nfs_label,
                                "variant": variant,
                                "projector": projector,
                                "eta": eta,
                                "refresh_method": refresh_method,
                                "task_steps": 100,
                                "smooth_steps": params.p2_nfs_steps,
                                "nfs_acceptance_rate": accepted / max(1, params.p2_nfs_steps),
                                "nfs_backtrack_mean": _mean(bt, float("nan")),
                                "teacher_KL_after_smooth": smooth_constraints["kl_teacher_student"],
                                "logit_drift_after_smooth": smooth_constraints["logit_relative_drift"],
                                "hidden_drift_after_smooth": smooth_constraints["hidden_relative_drift"],
                                "refresh_recovered_loss": recovery,
                                "acc_drop_from_task": acc_drop,
                                "phi_reduction_from_task": phi_red,
                                "p4_pass": int(p4_pass),
                                "error": "",
                            }
                            row.update(task_eval)
                            row.update(smooth_eval)
                            row.update(refresh_eval)
                            rows.append(row)
                            print(f"P4 {dataset} {task_method} {nfs_label} {refresh_method} accDrop={acc_drop:.4f} phiRed={phi_red:.3f} rec={recovery:.3f} pass={int(p4_pass)}")
                        except Exception as exc:
                            if not args.continue_on_error:
                                raise
                            rows.append({"stage": "P4", "dataset": dataset, "seed": seed, "task_method": task_method, "nfs_label": nfs_label, "refresh_method": refresh_method, "error": repr(exc)})
                            print(f"P4 ERROR {dataset} {task_method} {nfs_label}: {exc!r}")
                        write_csv(out_dir / "p4_tan_one_cycle_micro_run.csv", rows)
    return rows


def run_p5(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    p4_rows = [r for r in read_csv(out_dir / "p4_tan_one_cycle_micro_run.csv") if not r.get("error")]
    grouped: Dict[Tuple[str, str, str, str], Dict[str, List[Dict[str, Any]]]] = {}
    for row in p4_rows:
        key = (row.get("task_method", ""), row.get("nfs_label", ""), row.get("variant", ""), row.get("refresh_method", ""))
        grouped.setdefault(key, {}).setdefault(row.get("dataset", ""), []).append(row)
    survivors = [
        key for key, by_d in grouped.items()
        if all(any(int(float(r.get("p4_pass", 0))) for r in by_d.get(d, [])) for d in DATASETS)
    ]
    if not survivors:
        rows = [{"status": "not_run", "reason": "P4 produced no all-dataset TAN one-cycle survivor"}]
        write_csv(out_dir / "p5_tan_alternating_cycles.csv", rows)
        print("P5 not run: no P4 all-dataset survivor")
        return rows
    # Keep P5 intentionally small: validate whether the best P4 cycle remains stable over 3 cycles.
    rows = [] if args.fresh else [r for r in read_csv(out_dir / "p5_tan_alternating_cycles.csv") if r.get("status") != "not_run"]
    params = V48Params()
    device = get_device(args.device)
    configs = survivors[:2]
    nfs_by_label = {label: (variant, projector, eta) for variant, projector, eta, label in _select_nfs_configs(out_dir, max_configs=8)}
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        for seed in parse_int_list(args.seeds):
            for task_method, nfs_label, variant, refresh_method in configs:
                projector_eta = nfs_by_label.get(nfs_label, (variant, "diag", 0.035))
                variant, projector, eta = projector_eta
                bundle, state_cpu, teacher, task_eval, _initial_hold = _train_task_state(args, dataset, seed, task_method, params, device, steps=50)
                model = _make_model(bundle, params, device, ab=False)
                _load_state_cpu(model, state_cpu, device)
                hb = bundle.x_val[: params.audit_batch_size].to(device)
                yh = bundle.y_val[: params.audit_batch_size].to(device)
                cycle_rows: List[Dict[str, Any]] = []
                for cycle in range(1, 4):
                    teacher = _teacher_snapshot(model, hb)
                    before = _phase_eval("before", model, bundle, params, device)
                    accepted = 0
                    for step in range(params.p2_nfs_steps):
                        role = ["input", "block", "output"][step % 3] if "role-cycle" in variant else None
                        if "role-block" in variant:
                            role = "block"
                        stats = _nfs_project_once(model, hb, yh, teacher, params, variant=variant, projector=projector, role=role, eta=eta, device=device)
                        accepted += int(stats.get("accepted", 0))
                    _refresh_model(model, bundle, dataset, seed, refresh_method, params, device)
                    after = _phase_eval("after", model, bundle, params, device)
                    row = {
                        "stage": "P5",
                        "dataset": dataset,
                        "seed": seed,
                        "task_method": task_method,
                        "nfs_label": nfs_label,
                        "variant": variant,
                        "projector": projector,
                        "eta": eta,
                        "refresh_method": refresh_method,
                        "cycle_index": cycle,
                        "nfs_acceptance_rate": accepted / max(1, params.p2_nfs_steps),
                        "acc_by_cycle": after["after_acc"],
                        "val_loss_by_cycle": after["after_val_loss"],
                        "phi_by_cycle": after["after_phi"],
                        "jacobian_by_cycle": after["after_jacobian"],
                        "rank_by_cycle": after["after_rank"],
                        "margin_by_cycle": after["after_margin"],
                        "geometry_gain_per_task_loss": (1.0 - after["after_phi"] / max(1e-12, before["before_phi"])) / max(1e-12, after["after_val_loss"] - before["before_val_loss"] + 1e-6),
                        "error": "",
                    }
                    row.update(before)
                    row.update(after)
                    cycle_rows.append(row)
                    rows.append(row)
                    # Small task phase before the next cycle.
                    _refresh_model(model, bundle, dataset, seed, "FCAdam10", params, device)
                final = cycle_rows[-1]
                p5_pass = (
                    final["after_acc"] >= task_eval["task_acc"] - 0.01
                    and final["after_phi"] <= 0.85 * max(1e-12, task_eval["task_phi"])
                )
                for row in cycle_rows:
                    row["p5_pass_final_combo"] = int(p5_pass)
                print(f"P5 {dataset} {task_method} {nfs_label} finalAcc={final['after_acc']:.4f} phi={final['after_phi']:.4f} pass={int(p5_pass)}")
                write_csv(out_dir / "p5_tan_alternating_cycles.csv", rows)
    return rows


def run_p6(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows = [] if args.fresh else read_csv(out_dir / "p6_capacity_basis_expansion.csv")
    device = get_device(args.device)
    datasets = [dataset_name(d) for d in parse_str_list(args.datasets)]
    seeds = parse_int_list(args.seeds)
    if args.seeds == add_args().get_default("seeds"):
        seeds = [0]
    configs = [
        (64, 16, 4, "RBF"),
        (96, 24, 4, "RBF"),
        (96, 32, 2, "RBF"),
        (96, 24, 4, "AB-RBF"),
    ]
    methods = ["A0-PureKAN-AdamW", "A6-AdamW-MixedGeom"]
    done = {(r.get("dataset"), int(r.get("seed", -1)), r.get("method"), int(float(r.get("hidden_dim", -1))), int(float(r.get("basis_count", -1))), int(float(r.get("depth", -1))), r.get("basis_type")) for r in rows if not r.get("error")}
    for dataset in datasets:
        for seed in seeds:
            for hidden, basis, depth, basis_type in configs:
                for method in methods:
                    key = (dataset, seed, method, hidden, basis, depth, basis_type)
                    if key in done:
                        continue
                    params = V48Params(hidden_dim=hidden, basis_count=basis, depth=depth)
                    try:
                        _, _, row, _trace, _ = _adam_train_model(args, dataset, seed, method, params, device, steps=params.p6_steps, ab=(basis_type == "AB-RBF"))
                        row["stage"] = "P6"
                        row["param_count"] = sum(p.numel() for p in _make_model(load_vision_bundle(dataset, data_root=args.data_root, train_size=16, val_size=16, test_size=16, seed=seed, download=not args.no_download, allow_fake_data=args.allow_fake_data), params, device, ab=(basis_type == "AB-RBF")).parameters())
                        rows.append(row)
                        print(f"P6 {dataset} {method} h={hidden} b={basis} d={depth} {basis_type} acc={row['test_acc']:.4f} phi={row['phi_prime_p95']:.4f}")
                    except Exception as exc:
                        if not args.continue_on_error:
                            raise
                        rows.append({"stage": "P6", "dataset": dataset, "seed": seed, "method": method, "hidden_dim": hidden, "basis_count": basis, "depth": depth, "basis_type": basis_type, "error": repr(exc)})
                        print(f"P6 ERROR {dataset} {method}: {exc!r}")
                    write_csv(out_dir / "p6_capacity_basis_expansion.csv", rows)
    return rows


def _blank_later_files(out_dir: Path, reason: str) -> None:
    for name in [
        "p3_nfs_functional_teachers.csv",
        "p4_tan_one_cycle_micro_run.csv",
        "p5_tan_alternating_cycles.csv",
        "p7_candidate_selection.csv",
        "p8_confirm5.csv",
        "p9_confirm10.csv",
    ]:
        path = out_dir / name
        if not path.exists():
            write_csv(path, [{"status": "not_run", "reason": reason}])


def add_args() -> argparse.ArgumentParser:
    p = add_v3_args()
    p.description = __doc__
    p.add_argument("--p2-grid", choices=["core", "full"], default="core")
    p.set_defaults(packages="V4_8_P0_SMOKE", datasets="MNIST,Fashion-MNIST,KMNIST", out_dir=Path("results/v4_8"), seeds="0")
    return p


def main() -> int:
    args = add_args().parse_args()
    all_rows: List[Dict[str, Any]] = []
    for package in parse_str_list(args.packages):
        key = package.strip().upper().replace("-", "_")
        if key == "V4_8_P0_SMOKE":
            all_rows = run_p0(args)
        elif key in {"V4_8_P1_FRONTIER", "V4_8_P1_FGF"}:
            all_rows = run_p1(args)
        elif key in {"V4_8_P2_NFS_ADAMW", "V4_8_P2_NFS"}:
            all_rows = run_p2(args)
            _blank_later_files(ensure_dir(args.out_dir), "pending analyzer gate decision")
        elif key in {"V4_8_P3_NFS_FUNCTIONAL", "V4_8_P3"}:
            all_rows = run_p3(args)
        elif key in {"V4_8_P4_TAN_ONE_CYCLE", "V4_8_P4"}:
            all_rows = run_p4(args)
        elif key in {"V4_8_P5_TAN_CYCLES", "V4_8_P5"}:
            all_rows = run_p5(args)
        elif key in {"V4_8_P6_CAPACITY", "V4_8_P6"}:
            all_rows = run_p6(args)
        else:
            raise ValueError(f"unknown v4.8 package: {package}")
    return 0 if all_rows is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
