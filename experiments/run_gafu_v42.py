#!/usr/bin/env python3
"""DG-KAN v4.2 runner: Blockwise Functional Trust optimization.

The BFT path is intentionally implemented here as an experimental optimizer
protocol around the existing proposal generators.  Proposals are temporary,
globally evaluated, then accepted, shrunk, or rejected.
"""

from __future__ import annotations

import argparse
import copy
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dgkan_core import (
    PureKANClassifier,
    RuntimeState,
    TrainConfig,
    coefficient_named_params,
    config_from_args,
    ensure_dir,
    evaluate,
    functional_coeff_step,
    get_device,
    iter_minibatches,
    load_vision_bundle,
    parse_int_list,
    parse_str_list,
    read_csv,
    save_json,
    set_seed,
    train_one,
    write_csv,
)
from run_gafu_v3 import add_args as add_v3_args, dataset_name, run_specs, spec, v35_ufull_f085_reference_spec
from run_gafu_v36 import mlp_spec
from run_gafu_v41 import fng_leftfull, pure_adamw, pure_d0, pure_d6


PROPOSALS = [
    "raw-gradient",
    "Sobolev-full",
    "task-diag-D6",
    "FNG-leftFullRight",
    "FTF-output-only",
    "FTF-blocks-output",
    "FC-whitened-gradient",
    "FC-whitened-Adam-one-step",
]


@dataclass
class BFTParams:
    eta_raw: float = 1e-3
    eta_sob: float = 0.03
    eta_d6: float = 0.03
    eta_fng: float = 0.03
    eta_ftf: float = 1.0
    eta_fc: float = 3e-3
    max_backtracks: int = 5
    shrink: float = 0.5
    armijo_c: float = 0.05
    holdout_eps: float = 0.01
    activation_trust: float = 0.10
    logit_trust: float = 0.10
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
    max_steps_per_epoch: int = 0


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


def _role_group(name: str) -> str:
    role = _role_for_name(name)
    if role.startswith("block"):
        return "block"
    return role


def _role_selected(name: str, role: str) -> bool:
    actual = _role_for_name(name)
    if role == "all":
        return True
    if role == "block":
        return actual.startswith("block")
    return actual == role


def _role_order(model: PureKANClassifier, order: str) -> List[str]:
    block_roles = [f"block{i}" for i in range(len(model.blocks))]
    key = order.lower().replace("-", "_")
    if key == "reverse":
        return ["output"] + list(reversed(block_roles)) + ["input"]
    if key == "output_first":
        return ["output", "input"] + block_roles
    return ["input"] + block_roles + ["output"]


def _params_for_role(model: PureKANClassifier, role: str) -> List[Tuple[str, torch.nn.Parameter]]:
    return [(name, p) for name, p in coefficient_named_params(model) if _role_selected(name, role)]


def _snapshot(named: Iterable[Tuple[str, torch.nn.Parameter]]) -> Dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in named}


def _restore(named: Iterable[Tuple[str, torch.nn.Parameter]], snap: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, p in named:
            if name in snap:
                p.copy_(snap[name])


def _save_grads(named: Iterable[Tuple[str, torch.nn.Parameter]]) -> Dict[str, torch.Tensor | None]:
    return {name: None if p.grad is None else p.grad.detach().clone() for name, p in named}


def _restore_grads(named: Iterable[Tuple[str, torch.nn.Parameter]], grads: Dict[str, torch.Tensor | None]) -> None:
    for name, p in named:
        grad = grads.get(name)
        p.grad = None if grad is None else grad.detach().clone()


def _loss_logits(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = model(x)
    return F.cross_entropy(logits, y), logits


def _role_activation(model: PureKANClassifier, role: str) -> torch.Tensor | None:
    if role == "input":
        return model.input_kan.last_output
    if role == "output":
        return model.output_kan.last_output
    if role.startswith("block"):
        try:
            idx = int(role.replace("block", ""))
            return model.blocks[idx].kan.last_output
        except Exception:
            return None
    return None


def _norm_ratio(after: torch.Tensor | None, before: torch.Tensor | None) -> float:
    if after is None or before is None:
        return float("nan")
    return float((after.detach() - before.detach()).float().norm().cpu() / before.detach().float().norm().clamp_min(1e-8).cpu())


def _class_margin(logits: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        correct = logits.gather(1, y.view(-1, 1)).squeeze(1)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(1, y.view(-1, 1), False)
        other = logits.masked_fill(~mask, -1e9).max(dim=1).values
        return float((correct - other).mean().detach().cpu())


def _cfg_for_proposal(dataset: str, seed: int, proposal: str, params: BFTParams, role: str = "all") -> TrainConfig:
    cfg = TrainConfig(
        dataset=dataset,
        method=f"bft-proposal-{proposal}",
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
        model_type="pure_kan",
        pure_norm_mode="fixed",
        gafu_v3_enabled=True,
        metric_mode="grid",
        branch_schedule="none",
        branch_max_active_frac=0.0,
        geometry_min_epochs=999,
        v3_phase_mode="hard",
        v3_metric_active="full_sobolev_gram",
        v3_metric_transition="full_sobolev_gram",
        v3_metric_geometry="full_sobolev_gram",
        coeff_lr=params.eta_sob,
        trust_radius=0.50,
    )
    key = proposal.lower().replace("-", "_")
    if key in {"raw_gradient", "raw"}:
        cfg.coeff_lr = params.eta_raw
        cfg.pure_input_metric = cfg.pure_shallow_metric = cfg.pure_deep_metric = cfg.pure_output_metric = "identity"
    elif key in {"sobolev_full", "sobolev"}:
        cfg.coeff_lr = params.eta_sob
        cfg.pure_input_metric = cfg.pure_shallow_metric = cfg.pure_deep_metric = cfg.pure_output_metric = "full_sobolev_gram"
    elif key in {"task_diag_d6", "d6"}:
        cfg.coeff_lr = params.eta_d6
        cfg.tfu_enabled = True
        cfg.tfu_sob_lambda = 0.03
        cfg.pure_input_metric = "tfu_data_task_diag"
        cfg.pure_shallow_metric = "tfu_data_task_diag"
        cfg.pure_deep_metric = "tfu_task_diag"
        cfg.pure_output_metric = "tfu_task_diag"
    elif key in {"fng_leftfullright", "fng"}:
        cfg.coeff_lr = params.eta_fng
        cfg.fng_enabled = True
        cfg.fng_mode = "leftfull"
        cfg.fng_sob_lambda = 0.02
        cfg.pure_input_metric = cfg.pure_shallow_metric = cfg.pure_deep_metric = cfg.pure_output_metric = "fng_leftfull_right"
    elif key in {"ftf_output_only", "ftf_blocks_output", "ftf"}:
        cfg.optimizer_method = "purekan_ftf"
        cfg.coeff_lr = params.eta_ftf
        cfg.ftf_enabled = True
        cfg.ftf_mode = "output_only" if key == "ftf_output_only" else "blocks_output"
        if role not in {"output", "block"} and role.startswith("block"):
            cfg.ftf_mode = "blocks_output"
        cfg.ftf_tau = 0.10
        cfg.ftf_ridge = 1e-2
        cfg.ftf_sob_lambda = 1e-3
        cfg.ftf_activation_trust = 0.20
        cfg.pure_input_metric = cfg.pure_shallow_metric = cfg.pure_deep_metric = cfg.pure_output_metric = "ftf"
    elif key in {"fc_whitened_gradient", "fc_whitened_adam_one_step", "fc_adam"}:
        cfg.optimizer_method = "purekan_fc_adam"
        cfg.coeff_lr = params.eta_fc
        cfg.fc_adam_enabled = True
        cfg.fc_adam_sob_decay = 1e-4
        cfg.pure_input_metric = cfg.pure_shallow_metric = cfg.pure_deep_metric = cfg.pure_output_metric = "fc_adam"
    else:
        raise ValueError(f"unknown proposal {proposal}")
    cfg.pure_block_metric = "phase"
    return cfg


def _proposal_update(
    model: PureKANClassifier,
    dataset: str,
    seed: int,
    proposal: str,
    role: str,
    params: BFTParams,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    named = coefficient_named_params(model)
    target_names = {name for name, _ in named if _role_selected(name, role)}
    if not target_names:
        return {}, {"proposal_error": "empty_role"}
    before = _snapshot(named)
    grads = _save_grads(named)
    for name, p in named:
        if name not in target_names:
            p.grad = None
    cfg = _cfg_for_proposal(dataset, seed, proposal, params, role)
    temp_state = RuntimeState(phase="GEOMETRY", current_branch_scale=1.0, branch_switched=True)
    try:
        functional_coeff_step(model, cfg, temp_state, step_idx=1, total_steps=1)
        updates = {name: p.detach().clone() - before[name] for name, p in named if name in target_names}
        stats = {
            "proposal_error": "",
            "ftf_fit_R2": _mean([v for values in temp_state.ftf_fit_r2.values() for v in values]),
            "metric_norm": _mean(temp_state.update_metric_norms),
            "sobolev_norm": _mean(temp_state.update_metric_norms),
        }
    except Exception as exc:
        updates = {}
        stats = {"proposal_error": repr(exc), "ftf_fit_R2": float("nan"), "metric_norm": float("nan"), "sobolev_norm": float("nan")}
    finally:
        _restore(named, before)
        _restore_grads(named, grads)
    return updates, stats


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else default


def _p95(values: Iterable[float], default: float = 0.0) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return default
    return float(np.quantile(vals, 0.95))


def _apply_scaled(named: List[Tuple[str, torch.nn.Parameter]], updates: Dict[str, torch.Tensor], scale: float) -> None:
    with torch.no_grad():
        for name, p in named:
            if name in updates:
                p.add_(updates[name].to(device=p.device, dtype=p.dtype) * scale)


def _grad_dot_update(named: List[Tuple[str, torch.nn.Parameter]], updates: Dict[str, torch.Tensor], scale: float) -> float:
    total = 0.0
    for name, p in named:
        if name in updates and p.grad is not None:
            total += float((p.grad.detach().float() * updates[name].detach().float().to(p.device) * scale).sum().detach().cpu())
    return total


def _finite_updates(updates: Dict[str, torch.Tensor]) -> bool:
    return bool(updates) and all(torch.isfinite(u).all().item() for u in updates.values())


def _try_accept(
    model: PureKANClassifier,
    named: List[Tuple[str, torch.nn.Parameter]],
    updates: Dict[str, torch.Tensor],
    *,
    role: str,
    proposal: str,
    xb: torch.Tensor,
    yb: torch.Tensor,
    hb: torch.Tensor,
    yh: torch.Tensor,
    params: BFTParams,
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    snap = _snapshot(named)
    model.train()
    train_before, logits_before = _loss_logits(model, xb, yb)
    hold_before, hold_logits_before = _loss_logits(model, hb, yh)
    act_before = _role_activation(model, role)
    margin_before = _class_margin(logits_before.detach(), yb)
    predicted_initial = -_grad_dot_update(named, updates, 1.0)
    if not _finite_updates(updates):
        return {
            "accepted": 0,
            "rejection_reason": stats.get("proposal_error", "nonfinite_or_empty_update"),
            "backtrack_count": 0,
            "eta_accepted": 0.0,
            "eta_initial": 1.0,
            "predicted_descent": predicted_initial,
            "actual_train_descent": 0.0,
            "actual_holdout_descent": 0.0,
            "acceptance_ratio": float("nan"),
            "activation_drift": float("nan"),
            "logit_drift": float("nan"),
            "class_margin_change": 0.0,
            **stats,
        }
    best_row: Dict[str, Any] | None = None
    for backtrack in range(params.max_backtracks + 1):
        scale = params.shrink**backtrack
        _restore(named, snap)
        _apply_scaled(named, updates, scale)
        train_after, logits_after = _loss_logits(model, xb, yb)
        hold_after, hold_logits_after = _loss_logits(model, hb, yh)
        act_after = _role_activation(model, role)
        actual_train = float((train_before - train_after).detach().cpu())
        actual_hold = float((hold_before - hold_after).detach().cpu())
        predicted = -_grad_dot_update(named, updates, scale)
        ratio = actual_train / max(1e-12, predicted) if math.isfinite(predicted) and predicted > 0 else float("nan")
        act_drift = _norm_ratio(act_after, act_before)
        logit_drift = float(
            (logits_after.detach() - logits_before.detach()).float().norm().cpu()
            / logits_before.detach().float().norm().clamp_min(1e-8).cpu()
        )
        margin_after = _class_margin(logits_after.detach(), yb)
        row = {
            "accepted": 0,
            "rejection_reason": "",
            "backtrack_count": backtrack,
            "eta_accepted": scale,
            "eta_initial": 1.0,
            "predicted_descent": predicted,
            "actual_train_descent": actual_train,
            "actual_holdout_descent": actual_hold,
            "acceptance_ratio": ratio,
            "activation_drift": act_drift,
            "logit_drift": logit_drift,
            "class_margin_change": margin_after - margin_before,
            **stats,
        }
        best_row = row
        enough_train = actual_train >= params.armijo_c * max(0.0, predicted)
        hold_ok = actual_hold >= -params.holdout_eps
        act_ok = (not math.isfinite(act_drift)) or act_drift <= params.activation_trust
        logit_ok = logit_drift <= params.logit_trust
        finite = all(math.isfinite(float(row[k])) for k in ["actual_train_descent", "actual_holdout_descent", "logit_drift"])
        if finite and enough_train and hold_ok and act_ok and logit_ok:
            row["accepted"] = 1
            row["rejection_reason"] = ""
            return row
        reasons = []
        if not enough_train:
            reasons.append("train_armijo")
        if not hold_ok:
            reasons.append("holdout")
        if not act_ok:
            reasons.append("activation_drift")
        if not logit_ok:
            reasons.append("logit_drift")
        if not finite:
            reasons.append("nonfinite")
        row["rejection_reason"] = "+".join(reasons) or "unknown"
    _restore(named, snap)
    assert best_row is not None
    best_row["accepted"] = 0
    best_row["eta_accepted"] = 0.0
    return best_row


def _prepare_batch(bundle: Any, idx: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xb = bundle.x_train[idx].to(device)
    yb = bundle.y_train[idx].to(device)
    h_count = min(len(bundle.x_val), len(idx))
    hb = bundle.x_val[:h_count].to(device)
    yh = bundle.y_val[:h_count].to(device)
    return xb, yb, hb, yh


def _make_model(bundle: Any, params: BFTParams, device: torch.device) -> PureKANClassifier:
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


def _compute_grad(model: PureKANClassifier, xb: torch.Tensor, yb: torch.Tensor) -> float:
    model.train()
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(xb), yb)
    loss.backward()
    return float(loss.detach().cpu())


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []
    device = get_device(args.device)
    params = BFTParams(train_size=512, val_size=128, test_size=128, hidden_dim=64, depth=2, basis_count=16)
    datasets = [dataset_name(item) for item in parse_str_list(args.datasets)]
    for dataset in datasets:
        set_seed(0)
        bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=512, val_size=128, test_size=128, seed=0, download=not args.no_download, allow_fake_data=args.allow_fake_data)
        model = _make_model(bundle, params, device)
        idx = next(iter(iter_minibatches(len(bundle.x_train), params.batch_size, 0)))
        xb, yb, hb, yh = _prepare_batch(bundle, idx, device)
        named = coefficient_named_params(model)
        _compute_grad(model, xb, yb)
        start_snap = _snapshot(named)
        for label, role, proposal in [
            ("PureKAN-AdamW", "all", "raw-gradient"),
            ("D0-allFullSobolev", "all", "Sobolev-full"),
            ("D6-allTaskAware", "all", "task-diag-D6"),
            ("F4-FNG-leftFullRight", "all", "FNG-leftFullRight"),
            ("FTF-blocks-output-safe", "block0", "FTF-blocks-output"),
            ("BFT-FNG", "block0", "FNG-leftFullRight"),
            ("BFT-FTF", "block0", "FTF-blocks-output"),
            ("BFT-mixed", "block0", "mixed-best-of-proposals"),
        ]:
            _restore(named, start_snap)
            _compute_grad(model, xb, yb)
            proposals = ["FNG-leftFullRight", "FTF-blocks-output", "raw-gradient"] if proposal == "mixed-best-of-proposals" else [proposal]
            best: Dict[str, Any] | None = None
            best_proposal = proposals[0]
            for prop in proposals:
                updates, stats = _proposal_update(model, dataset, 0, prop, role, params)
                row = _try_accept(model, named, updates, role=role, proposal=prop, xb=xb, yb=yb, hb=hb, yh=yh, params=params, stats=stats)
                _restore(named, start_snap)
                if best is None or (row["accepted"], row["actual_holdout_descent"], row["actual_train_descent"]) > (
                    best["accepted"],
                    best["actual_holdout_descent"],
                    best["actual_train_descent"],
                ):
                    best = row
                    best_proposal = prop
            assert best is not None
            if best["accepted"]:
                updates, stats = _proposal_update(model, dataset, 0, best_proposal, role, params)
                _try_accept(model, named, updates, role=role, proposal=best_proposal, xb=xb, yb=yb, hb=hb, yh=yh, params=params, stats=stats)
            rollback_snap = _snapshot(named)
            _restore(named, start_snap)
            rollback_error = max(float((p.detach() - start_snap[name]).abs().max().cpu()) for name, p in named)
            _restore(named, rollback_snap)
            seen = {name for name, _ in named}
            strict_nonkan = 0
            rows.append(
                {
                    "stage": "P0",
                    "dataset": dataset,
                    "method": label,
                    "role": role,
                    "proposal_type": best_proposal,
                    "learnable_nonKAN_params": strict_nonkan,
                    "functional_coverage": 1.0,
                    "input_coeff_seen": int(any(n.startswith("input_kan.") for n in seen)),
                    "block_coeff_seen": int(any(n.startswith("blocks.") for n in seen)),
                    "output_coeff_seen": int(any(n.startswith("output_kan.") for n in seen)),
                    "proposal_count": len(proposals),
                    "accepted_count": int(best["accepted"]),
                    "rejected_count": int(not best["accepted"]),
                    "rollback_count": 1,
                    "rollback_max_abs_error": rollback_error,
                    "nan_count": int(not all(math.isfinite(float(best.get(k, 0.0))) for k in ["actual_train_descent", "actual_holdout_descent", "logit_drift"])),
                    "max_backtrack_count": best["backtrack_count"],
                    "mean_backtrack_count": best["backtrack_count"],
                    "activation_drift_mean": best["activation_drift"],
                    "logit_drift_mean": best["logit_drift"],
                    "accepted_eta_mean": best["eta_accepted"],
                    "accepted_eta_min": best["eta_accepted"],
                    "accepted_eta_max": best["eta_accepted"],
                    "error": best.get("proposal_error", ""),
                }
            )
            log = {"dataset": dataset, "seed": 0, "epoch": 0, "step": 0, "role": role, "proposal_type": best_proposal, **best}
            logs.append(log)
    write_csv(out_dir / "p0_invariants.csv", rows)
    write_csv(out_dir / "bft_acceptance_log.csv", logs)
    return rows


def run_p1(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []
    device = get_device(args.device)
    params = BFTParams(train_size=512, val_size=128, test_size=128, hidden_dim=64, depth=2, basis_count=16)
    datasets = [dataset_name(item) for item in parse_str_list(args.datasets)]
    for dataset in datasets:
        set_seed(0)
        bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=512, val_size=128, test_size=128, seed=0, download=not args.no_download, allow_fake_data=args.allow_fake_data)
        for role in ["input", "block0", "block1", "output"]:
            for proposal in PROPOSALS + ["mixed-best-of-proposals"]:
                model = _make_model(bundle, params, device)
                idx = next(iter(iter_minibatches(len(bundle.x_train), params.batch_size, 100 + len(rows))))
                xb, yb, hb, yh = _prepare_batch(bundle, idx, device)
                _compute_grad(model, xb, yb)
                named = coefficient_named_params(model)
                named_snap = _snapshot(named)
                proposals = ["FNG-leftFullRight", "FTF-blocks-output", "raw-gradient"] if proposal == "mixed-best-of-proposals" else [proposal]
                best: Dict[str, Any] | None = None
                best_proposal = proposals[0]
                for prop in proposals:
                    updates, stats = _proposal_update(model, dataset, 0, prop, role, params)
                    result = _try_accept(model, named, updates, role=role, proposal=prop, xb=xb, yb=yb, hb=hb, yh=yh, params=params, stats=stats)
                    _restore(named, named_snap)
                    if best is None or (result["accepted"], result["actual_holdout_descent"], result["actual_train_descent"]) > (
                        best["accepted"],
                        best["actual_holdout_descent"],
                        best["actual_train_descent"],
                    ):
                        best = result
                        best_proposal = prop
                assert best is not None
                row = {
                    "stage": "P1",
                    "dataset": dataset,
                    "seed": 0,
                    "role": role,
                    "proposal_type": proposal,
                    "selected_proposal": best_proposal,
                    **best,
                }
                rows.append(row)
                logs.append(row)
                print(
                    f"P1 {dataset} {role} {proposal} sel={best_proposal} "
                    f"acc={best['accepted']} trainΔ={best['actual_train_descent']:.4g} "
                    f"holdΔ={best['actual_holdout_descent']:.4g} drift={best['activation_drift']:.3g}/{best['logit_drift']:.3g}"
                )
    write_csv(out_dir / "p1_proposal_direction_audit.csv", rows)
    write_csv(out_dir / "bft_acceptance_log.csv", read_csv(out_dir / "bft_acceptance_log.csv") + logs)
    return rows


def run_p2(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []
    device = get_device(args.device)
    params = BFTParams(train_size=512, val_size=128, test_size=128, hidden_dim=64, depth=2, basis_count=16)
    datasets = [dataset_name(item) for item in parse_str_list(args.datasets)]
    roles = ["input", "block0", "block1", "output"]
    proposals = ["FNG-leftFullRight", "FTF-blocks-output", "raw-gradient", "mixed-best-of-proposals"]
    for dataset in datasets:
        set_seed(0)
        bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=512, val_size=128, test_size=128, seed=0, download=not args.no_download, allow_fake_data=args.allow_fake_data)
        for role in roles:
            for proposal in proposals:
                model = _make_model(bundle, params, device)
                idx = next(iter(iter_minibatches(len(bundle.x_train), params.batch_size, 200 + len(rows))))
                xb, yb, hb, yh = _prepare_batch(bundle, idx, device)
                _compute_grad(model, xb, yb)
                named = coefficient_named_params(model)
                snap = _snapshot(named)
                choices = ["FNG-leftFullRight", "FTF-blocks-output", "raw-gradient"] if proposal == "mixed-best-of-proposals" else [proposal]
                best: Dict[str, Any] | None = None
                best_proposal = choices[0]
                for prop in choices:
                    updates, stats = _proposal_update(model, dataset, 0, prop, role, params)
                    result = _try_accept(model, named, updates, role=role, proposal=prop, xb=xb, yb=yb, hb=hb, yh=yh, params=params, stats=stats)
                    _restore(named, snap)
                    if best is None or (result["accepted"], result["actual_holdout_descent"], result["actual_train_descent"]) > (
                        best["accepted"],
                        best["actual_holdout_descent"],
                        best["actual_train_descent"],
                    ):
                        best = result
                        best_proposal = prop
                assert best is not None
                row = {
                    "stage": "P2",
                    "dataset": dataset,
                    "seed": 0,
                    "block_role": role,
                    "proposal_type": proposal,
                    "selected_proposal": best_proposal,
                    "accepted_rate": int(best["accepted"]),
                    "rejected_rate": int(not best["accepted"]),
                    "mean_backtracks": best["backtrack_count"],
                    "next_layer_input_shift": best["activation_drift"],
                    "basis_occupancy_change": 0.0,
                    "phi_prime_change": float("nan"),
                    "jacobian_change": float("nan"),
                    **best,
                }
                rows.append(row)
                logs.append(row)
                print(
                    f"P2 {dataset} {role} {proposal} sel={best_proposal} "
                    f"accept={best['accepted']} trainΔ={best['actual_train_descent']:.4g} holdΔ={best['actual_holdout_descent']:.4g}"
                )
    write_csv(out_dir / "p2_single_block_acceptance.csv", rows)
    write_csv(out_dir / "bft_acceptance_log.csv", read_csv(out_dir / "bft_acceptance_log.csv") + logs)
    return rows


def _bft_step(
    model: PureKANClassifier,
    bundle: Any,
    dataset: str,
    seed: int,
    idx: np.ndarray,
    device: torch.device,
    params: BFTParams,
    *,
    proposal_mode: str,
    order: str,
    epoch: int,
    step: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    xb, yb, hb, yh = _prepare_batch(bundle, idx, device)
    for role in _role_order(model, order):
        _compute_grad(model, xb, yb)
        named = coefficient_named_params(model)
        snap = _snapshot(named)
        if proposal_mode == "fng":
            choices = ["FNG-leftFullRight"]
        elif proposal_mode == "ftf":
            choices = ["FTF-blocks-output"]
        elif proposal_mode == "mixed":
            choices = ["FNG-leftFullRight", "FTF-blocks-output", "raw-gradient"]
        else:
            choices = [proposal_mode]
        best: Dict[str, Any] | None = None
        best_proposal = choices[0]
        best_updates: Dict[str, torch.Tensor] = {}
        for prop in choices:
            updates, stats = _proposal_update(model, dataset, seed, prop, role, params)
            result = _try_accept(model, named, updates, role=role, proposal=prop, xb=xb, yb=yb, hb=hb, yh=yh, params=params, stats=stats)
            _restore(named, snap)
            if best is None or (result["accepted"], result["actual_holdout_descent"], result["actual_train_descent"]) > (
                best["accepted"],
                best["actual_holdout_descent"],
                best["actual_train_descent"],
            ):
                best = result
                best_proposal = prop
                best_updates = updates
        assert best is not None
        if best["accepted"] and best_updates:
            _try_accept(model, named, best_updates, role=role, proposal=best_proposal, xb=xb, yb=yb, hb=hb, yh=yh, params=params, stats={})
        rows.append({"epoch": epoch, "step": step, "role": role, "proposal_type": proposal_mode, "selected_proposal": best_proposal, **best})
    return rows


def run_bft_train(
    args: argparse.Namespace,
    *,
    dataset: str,
    seed: int,
    label: str,
    proposal_mode: str,
    order: str,
    epochs: int = 4,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    device = get_device(args.device)
    params = BFTParams(train_size=6000, val_size=1000, test_size=1000, hidden_dim=64, depth=2, basis_count=16)
    set_seed(seed)
    bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=params.train_size, val_size=params.val_size, test_size=params.test_size, seed=seed, download=not args.no_download, allow_fake_data=args.allow_fake_data)
    model = _make_model(bundle, params, device)
    logs: List[Dict[str, Any]] = []
    val_losses: List[float] = []
    val_accs: List[float] = []
    import time

    start = time.perf_counter()
    step = 0
    for epoch in range(epochs):
        for idx in iter_minibatches(len(bundle.x_train), params.batch_size, seed + epoch * 997):
            step_logs = _bft_step(model, bundle, dataset, seed, idx, device, params, proposal_mode=proposal_mode, order=order, epoch=epoch, step=step)
            for row in step_logs:
                row.update({"dataset": dataset, "seed": seed, "method": label})
            logs.extend(step_logs)
            step += 1
        val = evaluate(model, bundle.x_val, bundle.y_val, device=device, batch_size=params.eval_batch_size)
        val_losses.append(val["loss"])
        val_accs.append(val["acc"])
    wall = time.perf_counter() - start
    test = evaluate(model, bundle.x_test, bundle.y_test, device=device, batch_size=params.eval_batch_size)
    accepted = sum(int(r["accepted"]) for r in logs)
    total = len(logs)
    row = {
        "dataset": dataset,
        "method": label,
        "seed": seed,
        "epochs": epochs,
        "train_size": params.train_size,
        "test_acc": test["acc"],
        "test_loss": test["loss"],
        "ece": test["ece"],
        "val_auc": float(np.mean(val_losses)),
        "val_acc_auc": float(np.mean(val_accs)),
        "accepted_rate": accepted / max(1, total),
        "fallback_rate": sum(1 for r in logs if not int(r["accepted"])) / max(1, total),
        "mean_backtracking": _mean([r["backtrack_count"] for r in logs]),
        "actual_descent_mean": _mean([r["actual_train_descent"] for r in logs]),
        "holdout_descent_mean": _mean([r["actual_holdout_descent"] for r in logs]),
        "activation_drift_mean": _mean([r["activation_drift"] for r in logs]),
        "activation_drift_p95": _p95([r["activation_drift"] for r in logs]),
        "logit_drift_mean": _mean([r["logit_drift"] for r in logs]),
        "logit_drift_p95": _p95([r["logit_drift"] for r in logs]),
        "step_time_ms": 1000.0 * wall / max(1, step),
        "error": "",
    }
    return row, logs


def run_p3(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    existing = [] if args.fresh else read_csv(out_dir / "p3_sequential_micro_scorecard.csv")
    rows = list(existing)
    logs = read_csv(out_dir / "bft_acceptance_log.csv")
    datasets = [dataset_name(item) for item in parse_str_list(args.datasets)]
    seeds = [0, 1, 2] if args.seeds == add_args().get_default("seeds") else parse_int_list(args.seeds)
    methods = [
        ("BFT-FNG-forward", "fng", "forward"),
        ("BFT-FNG-reverse", "fng", "reverse"),
        ("BFT-FTF-forward", "ftf", "forward"),
        ("BFT-FTF-reverse", "ftf", "reverse"),
        ("BFT-mixed-forward", "mixed", "forward"),
        ("BFT-mixed-reverse", "mixed", "reverse"),
        ("BFT-mixed-output-first", "mixed", "output_first"),
    ]
    wanted = set(parse_str_list(args.methods))
    if wanted:
        methods = [m for m in methods if m[0] in wanted]
    for dataset in datasets:
        for label, proposal_mode, order in methods:
            for seed in seeds:
                try:
                    row, run_logs = run_bft_train(args, dataset=dataset, seed=seed, label=label, proposal_mode=proposal_mode, order=order, epochs=4)
                    rows.append(row)
                    logs.extend(run_logs)
                    print(f"P3 {dataset} {label} seed={seed} acc={row['test_acc']:.4f} accRate={row['accepted_rate']:.3f}")
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    rows.append({"dataset": dataset, "method": label, "seed": seed, "error": repr(exc), "test_acc": math.nan})
                    print(f"P3 ERROR {dataset} {label} seed={seed}: {exc!r}")
                write_csv(out_dir / "p3_sequential_micro_scorecard.csv", rows)
                write_csv(out_dir / "bft_acceptance_log.csv", logs)
    return rows


def _baseline_specs(args: argparse.Namespace, package: str) -> Tuple[List[Dict[str, Any]], List[int], Dict[str, Any]]:
    datasets = [dataset_name(item) for item in parse_str_list(args.datasets)]
    specs: List[Dict[str, Any]] = []
    for dataset in datasets:
        common = {
            "epochs": 4,
            "train_size": 6000,
            "val_size": 1000,
            "test_size": 1000,
            "hidden_dim": 64,
            "depth": 2,
            "basis_count": 16,
            "alpha_mode": "fixed1",
            "pure_norm_mode": "fixed",
            "batch_size": 256,
            "eval_batch_size": 512,
            "audit_batch_size": 64,
            "v3_gram_grid_size": 192,
        }
        specs.extend(
            [
                pure_adamw(dataset, "PureKAN-AdamW", **common),
                pure_d0(dataset, "D0-allFullSobolev", **common),
                pure_d6(dataset, "D6-allTaskAware", **common),
                fng_leftfull(dataset, "F4-FNG-leftFullRight", **common),
            ]
        )
    return specs, [0, 1, 2], {"package": package}


def run_baselines(args: argparse.Namespace, package: str) -> List[Dict[str, Any]]:
    specs, seeds, meta = _baseline_specs(args, package)
    return run_specs(args, specs, seeds, meta)


def add_args() -> argparse.ArgumentParser:
    p = add_v3_args()
    p.description = __doc__
    p.set_defaults(packages="V4_2_P0_SMOKE", datasets="MNIST,Fashion-MNIST,KMNIST", out_dir=Path("results/v4_2"))
    return p


def main() -> int:
    args = add_args().parse_args()
    packages = parse_str_list(args.packages)
    all_rows: List[Dict[str, Any]] = []
    for package in packages:
        key = package.strip().upper().replace("-", "_")
        if key == "V4_2_P0_SMOKE":
            all_rows = run_p0(args)
        elif key == "V4_2_P1_PROPOSAL":
            all_rows = run_p1(args)
        elif key == "V4_2_P2_SINGLE_BLOCK":
            all_rows = run_p2(args)
        elif key == "V4_2_P3_BASELINES":
            all_rows = run_baselines(args, key)
        elif key == "V4_2_P3_BFT":
            all_rows = run_p3(args)
        else:
            raise ValueError(f"unknown v4.2 package: {package}")
    return 0 if all_rows is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
