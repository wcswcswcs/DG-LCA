#!/usr/bin/env python3
"""DG-KAN v3.7 runner: PureKAN-UFULL code audit, schedules, and role-aware updates."""

from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    load_vision_bundle,
    parse_int_list,
    parse_str_list,
    save_json,
    set_seed,
    write_csv,
)
from run_gafu_v3 import (
    Spec,
    add_args as add_v3_args,
    dataset_name,
    run_specs,
    spec,
    v35_ufull_f085_reference_spec,
)
from run_gafu_v36 import mlp_spec


def v37_epochs(dataset: str) -> int:
    return 30 if dataset == "Fashion-MNIST" else 20


def v37_common(dataset: str, **extra: Any) -> Dict[str, Any]:
    out = {
        "epochs": v37_epochs(dataset),
        "train_size": 6000,
        "val_size": 1000,
        "test_size": 1000,
        "hidden_dim": 96,
        "depth": 4,
        "basis_count": 24,
        "alpha_mode": "fixed1",
        "alpha_init": 1.5,
        "coeff_lr": 0.03,
        "lr": 1e-3,
        "batch_size": 256,
        "eval_batch_size": 512,
        "audit_batch_size": 64,
        "v3_gram_grid_size": 256,
        "trust_radius": 0.50,
    }
    out.update(extra)
    return out


def pure_adamw(dataset: str, label: str = "PureKAN-AdamW-alphaFixed1", **extra: Any) -> Spec:
    overrides = {**v37_common(dataset), "model_type": "pure_kan"}
    overrides.update(extra)
    return spec(dataset, label, "purekan_adamw", **overrides)


def pure_ufull(
    dataset: str,
    label: str,
    *,
    active: str = "full_sobolev_gram",
    transition: str = "full_sobolev_gram",
    geometry: str = "full_sobolev_gram",
    max_active_frac: float = 0.0,
    phase_mode: str = "hard",
    pure_input_metric: str = "phase",
    pure_block_metric: str = "phase",
    pure_output_metric: str = "phase",
    pure_input_lr_mult: float = 1.0,
    pure_block_lr_mult: float = 1.0,
    pure_output_lr_mult: float = 1.0,
    coeff_lr: float = 0.03,
    **extra: Any,
) -> Spec:
    overrides = {
        **v37_common(dataset, coeff_lr=coeff_lr),
        "model_type": "pure_kan",
        "gafu_v3_enabled": True,
        "metric_mode": "grid",
        "branch_schedule": "none",
        "warmup_frac": 0.0,
        "v3_phase_mode": phase_mode,
        "v3_metric_active": active,
        "v3_metric_transition": transition,
        "v3_metric_geometry": geometry,
        "v3_alpha_fast": 0.0,
        "v3_beta_fast": 0.0,
        "v3_alpha_geo": 0.15,
        "v3_beta_geo": 0.02,
        "v3_gram_rho": 1e-3,
        "branch_boost": 1.0,
        "branch_final_scale": 1.0,
        "coeff_lr_boost": 1.0,
        "coeff_lr_final_mult": 1.0,
        "branch_max_active_frac": max_active_frac,
        # PureKAN schedule audits need max_active_frac to define the warmup
        # duration.  A large min-epoch value prevents epoch-end geometry
        # triggers from collapsing 0.10/0.20 warmups into the same trace.
        "geometry_min_epochs": 999,
        "unified_optimizer_mode": "hybrid",
        "pure_input_metric": pure_input_metric,
        "pure_block_metric": pure_block_metric,
        "pure_output_metric": pure_output_metric,
        "pure_input_lr_mult": pure_input_lr_mult,
        "pure_block_lr_mult": pure_block_lr_mult,
        "pure_output_lr_mult": pure_output_lr_mult,
    }
    overrides.update(extra)
    return spec(dataset, label, "purekan_ufull", **overrides)


def hybrid_ufull(dataset: str, label: str = "Hybrid-DGKAN-UFULL-f085-alphaFixed1", **extra: Any) -> Spec:
    overrides = v37_common(dataset)
    overrides.update(extra)
    return v35_ufull_f085_reference_spec(dataset, label=label, **overrides)


def p0_common() -> Dict[str, Any]:
    return {
        "train_size": 512,
        "val_size": 128,
        "test_size": 128,
        "epochs": 1,
        "hidden_dim": 96,
        "depth": 4,
        "basis_count": 24,
        "batch_size": 256,
        "eval_batch_size": 256,
        "audit_batch_size": 64,
        "v3_gram_grid_size": 128,
    }


def package_specs(args: argparse.Namespace, package: str) -> Tuple[List[Spec], List[int], Dict[str, Any]]:
    key = package.strip().upper().replace("-", "_")
    datasets = [dataset_name(item) for item in parse_str_list(args.datasets)]
    specs: List[Spec] = []
    seeds = parse_int_list(args.seeds)

    if key == "V3_7_P0_CODE_AUDIT":
        seeds = [0]
        for dataset in [d for d in datasets if d in {"MNIST", "Fashion-MNIST", "KMNIST"}]:
            common = p0_common()
            specs.extend(
                [
                    pure_adamw(dataset, "PureKAN-AdamW-smoke-alphaFixed1", **common),
                    pure_ufull(dataset, "PureKAN-UFULL-full-smoke-alphaFixed1", **common),
                    pure_ufull(
                        dataset,
                        "PureKAN-UFULL-diagwarmup-smoke-alphaFixed1",
                        active="basis_diag_gram",
                        max_active_frac=0.20,
                        **common,
                    ),
                    pure_ufull(
                        dataset,
                        "PureKAN-UFULL-identitywarmup-smoke-alphaFixed1",
                        active="identity",
                        max_active_frac=0.20,
                        **common,
                    ),
                    hybrid_ufull(dataset, "Hybrid-DGKAN-UFULL-f085-smoke-alphaFixed1", **common),
                    mlp_spec(dataset, "MLP-AdamW-smoke-alphaFixed1", **common),
                ]
            )
        return specs, seeds, {"package": key}

    if key == "V3_7_P2_SCHEDULE_REPAIR":
        seeds = [0, 1, 2]
        for dataset in [d for d in datasets if d in {"MNIST", "Fashion-MNIST", "KMNIST"}]:
            specs.extend(
                [
                    pure_adamw(dataset),
                    pure_ufull(dataset, "PureKAN-UFULL-full-alphaFixed1"),
                    pure_ufull(dataset, "PureKAN-UFULL-diagwarmup-0p10-alphaFixed1", active="basis_diag_gram", max_active_frac=0.10),
                    pure_ufull(dataset, "PureKAN-UFULL-diagwarmup-0p20-alphaFixed1", active="basis_diag_gram", max_active_frac=0.20),
                    pure_ufull(dataset, "PureKAN-UFULL-identitywarmup-0p10-alphaFixed1", active="identity", max_active_frac=0.10),
                    pure_ufull(dataset, "PureKAN-UFULL-identitywarmup-0p20-alphaFixed1", active="identity", max_active_frac=0.20),
                    pure_ufull(
                        dataset,
                        "PureKAN-UFULL-diag-to-full-smooth-0p20-alphaFixed1",
                        active="basis_diag_gram",
                        transition="diag_to_full_sobolev",
                        geometry="full_sobolev_gram",
                        max_active_frac=0.20,
                        phase_mode="smooth",
                    ),
                ]
            )
        return specs, seeds, {"package": key}

    if key == "V3_7_P3_ROLE_SWEEP":
        seeds = [0, 1, 2]
        for dataset in [d for d in datasets if d in {"MNIST", "Fashion-MNIST", "KMNIST"}]:
            specs.extend(
                [
                    pure_ufull(dataset, "PureKAN-UFULL-role-allfull-alphaFixed1"),
                    pure_ufull(
                        dataset,
                        "PureKAN-UFULL-role-ioDiag-blockFull-o2-alphaFixed1",
                        pure_input_metric="basis_diag_gram",
                        pure_block_metric="full_sobolev_gram",
                        pure_output_metric="basis_diag_gram",
                        pure_output_lr_mult=2.0,
                    ),
                    pure_ufull(
                        dataset,
                        "PureKAN-UFULL-role-ioDiag-blockFull-i2o2-alphaFixed1",
                        pure_input_metric="basis_diag_gram",
                        pure_block_metric="full_sobolev_gram",
                        pure_output_metric="basis_diag_gram",
                        pure_input_lr_mult=2.0,
                        pure_output_lr_mult=2.0,
                    ),
                    pure_ufull(
                        dataset,
                        "PureKAN-UFULL-role-ioDiag-blockFull-o4-alphaFixed1",
                        pure_input_metric="basis_diag_gram",
                        pure_block_metric="full_sobolev_gram",
                        pure_output_metric="basis_diag_gram",
                        pure_output_lr_mult=4.0,
                    ),
                    pure_ufull(
                        dataset,
                        "PureKAN-UFULL-role-iId-blockFull-oDiag-o2-alphaFixed1",
                        pure_input_metric="identity",
                        pure_block_metric="full_sobolev_gram",
                        pure_output_metric="basis_diag_gram",
                        pure_output_lr_mult=2.0,
                    ),
                    pure_ufull(
                        dataset,
                        "PureKAN-UFULL-role-iId-blockFull-oDiag-o4-alphaFixed1",
                        pure_input_metric="identity",
                        pure_block_metric="full_sobolev_gram",
                        pure_output_metric="basis_diag_gram",
                        pure_output_lr_mult=4.0,
                    ),
                    pure_ufull(
                        dataset,
                        "PureKAN-UFULL-role-iDiag-blockFull-oId-alphaFixed1",
                        pure_input_metric="basis_diag_gram",
                        pure_block_metric="full_sobolev_gram",
                        pure_output_metric="identity",
                        pure_input_lr_mult=2.0,
                    ),
                    pure_ufull(
                        dataset,
                        "PureKAN-UFULL-role-allDiag-o2-alphaFixed1",
                        pure_input_metric="basis_diag_gram",
                        pure_block_metric="basis_diag_gram",
                        pure_output_metric="basis_diag_gram",
                        pure_output_lr_mult=2.0,
                    ),
                ]
            )
        return specs, seeds, {"package": key}

    if key == "V3_7_P5_CANDIDATE_SELECTION":
        seeds = [0, 1, 2]
        for dataset in [d for d in datasets if d in {"MNIST", "Fashion-MNIST", "KMNIST"}]:
            specs.extend(
                [
                    mlp_spec(dataset, "MLP-AdamW-alphaFixed1", **v37_common(dataset)),
                    hybrid_ufull(dataset),
                    pure_adamw(dataset),
                    pure_ufull(dataset, "PureKAN-UFULL-full-alphaFixed1"),
                    pure_ufull(
                        dataset,
                        "PureKAN-UFULL-role-ioDiag-blockFull-o2-alphaFixed1",
                        pure_input_metric="basis_diag_gram",
                        pure_block_metric="full_sobolev_gram",
                        pure_output_metric="basis_diag_gram",
                        pure_output_lr_mult=2.0,
                    ),
                    pure_ufull(
                        dataset,
                        "PureKAN-UFULL-role-iId-blockFull-oDiag-o2-alphaFixed1",
                        pure_input_metric="identity",
                        pure_block_metric="full_sobolev_gram",
                        pure_output_metric="basis_diag_gram",
                        pure_output_lr_mult=2.0,
                    ),
                    pure_ufull(dataset, "PureKAN-UFULL-diagwarmup-0p20-alphaFixed1", active="basis_diag_gram", max_active_frac=0.20),
                ]
            )
        return specs, seeds, {"package": key}

    if key == "V3_7_P6_CONFIRM5":
        seeds = [0, 1, 2, 3, 4]
        for dataset in [d for d in datasets if d in {"MNIST", "Fashion-MNIST", "KMNIST"}]:
            specs.extend(
                [
                    mlp_spec(dataset, "MLP-AdamW-alphaFixed1", **v37_common(dataset)),
                    hybrid_ufull(dataset),
                    pure_adamw(dataset),
                    pure_ufull(
                        dataset,
                        "PureKAN-UFULL-final-candidate-alphaFixed1",
                        pure_input_metric="basis_diag_gram",
                        pure_block_metric="full_sobolev_gram",
                        pure_output_metric="basis_diag_gram",
                        pure_output_lr_mult=2.0,
                    ),
                ]
            )
        return specs, seeds, {"package": key}

    raise ValueError(f"unknown v3.7 package: {package}")


def _shadow_specs(dataset: str) -> List[Tuple[str, Dict[str, Any]]]:
    return [
        ("AdamW-one-step", {"kind": "adamw"}),
        ("identity-functional", {"v3_metric_geometry": "identity"}),
        ("basis-diag-gram", {"v3_metric_geometry": "basis_diag_gram"}),
        ("full-sobolev-gram", {"v3_metric_geometry": "full_sobolev_gram"}),
        (
            "role-inputDiag-blockFull-outputDiag",
            {
                "v3_metric_geometry": "full_sobolev_gram",
                "pure_input_metric": "basis_diag_gram",
                "pure_block_metric": "full_sobolev_gram",
                "pure_output_metric": "basis_diag_gram",
            },
        ),
        (
            "role-inputIdentity-blockFull-outputIdentity",
            {
                "v3_metric_geometry": "full_sobolev_gram",
                "pure_input_metric": "identity",
                "pure_block_metric": "full_sobolev_gram",
                "pure_output_metric": "identity",
            },
        ),
    ]


def _flat_update(before: Dict[str, torch.Tensor], model: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    grads: List[torch.Tensor] = []
    updates: List[torch.Tensor] = []
    for name, p in coefficient_named_params(model):
        if p.grad is not None:
            grads.append(p.grad.detach().flatten().float().cpu())
        else:
            grads.append(torch.zeros(p.numel()))
        updates.append((p.detach().cpu() - before[name]).flatten().float())
    return torch.cat(grads) if grads else torch.empty(0), torch.cat(updates) if updates else torch.empty(0)


def run_shadow(args: argparse.Namespace, package: str) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    run_dir = ensure_dir(out_dir / "shadow")
    rows: List[Dict[str, Any]] = []
    datasets = [dataset_name(item) for item in parse_str_list(args.datasets)]
    seeds = parse_int_list(args.seeds) or [0]
    device = get_device(args.device)
    for dataset in [d for d in datasets if d in {"MNIST", "Fashion-MNIST", "KMNIST"}]:
        for seed in seeds[:1]:
            set_seed(seed)
            bundle = load_vision_bundle(
                dataset,
                data_root=args.data_root,
                train_size=512,
                val_size=128,
                test_size=128,
                seed=seed,
                download=not args.no_download,
                allow_fake_data=args.allow_fake_data,
            )
            base = PureKANClassifier(
                bundle.input_dim,
                bundle.num_classes,
                hidden_dim=96,
                depth=4,
                basis_count=24,
                alpha_init=1.5,
                alpha_mode="fixed1",
            ).to(device)
            idx = torch.arange(min(256, len(bundle.x_train)))
            xb = bundle.x_train[idx].to(device)
            yb = bundle.y_train[idx].to(device)
            val_x = bundle.x_val.to(device)
            val_y = bundle.y_val.to(device)
            with torch.no_grad():
                train_before = float(F.cross_entropy(base(xb), yb).detach().cpu())
                val_before = float(F.cross_entropy(base(val_x), val_y).detach().cpu())
            for label, overrides in _shadow_specs(dataset):
                model = copy.deepcopy(base).to(device)
                cfg = TrainConfig(
                    dataset=dataset,
                    method=f"shadow-{label}",
                    seed=seed,
                    device=args.device,
                    data_root=args.data_root,
                    download=not args.no_download,
                    train_size=512,
                    val_size=128,
                    test_size=128,
                    epochs=1,
                    hidden_dim=96,
                    depth=4,
                    basis_count=24,
                    alpha_mode="fixed1",
                    model_type="pure_kan",
                    gafu_v3_enabled=True,
                    branch_schedule="none",
                    branch_max_active_frac=0.0,
                    v3_phase_mode="hard",
                    v3_metric_active=overrides.get("v3_metric_geometry", "full_sobolev_gram"),
                    v3_metric_transition=overrides.get("v3_metric_geometry", "full_sobolev_gram"),
                    v3_metric_geometry=overrides.get("v3_metric_geometry", "full_sobolev_gram"),
                    v3_gram_grid_size=128,
                    coeff_lr=0.03,
                    pure_input_metric=overrides.get("pure_input_metric", "phase"),
                    pure_block_metric=overrides.get("pure_block_metric", "phase"),
                    pure_output_metric=overrides.get("pure_output_metric", "phase"),
                )
                before = {name: p.detach().cpu().clone() for name, p in coefficient_named_params(model)}
                model.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                if overrides.get("kind") == "adamw":
                    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                    opt.step()
                    state = RuntimeState()
                else:
                    state = RuntimeState(phase="GEOMETRY", current_branch_scale=1.0, branch_switched=True)
                    functional_coeff_step(model, cfg, state, step_idx=1, total_steps=1)
                grad, update = _flat_update(before, model)
                with torch.no_grad():
                    train_after = float(F.cross_entropy(model(xb), yb).detach().cpu())
                    val_after = float(F.cross_entropy(model(val_x), val_y).detach().cpu())
                pred_descent = float(-(grad * update).sum().detach().cpu()) if grad.numel() else float("nan")
                row = {
                    "stage": "DG-KAN-v3.7",
                    "package": package,
                    "dataset": dataset,
                    "method": label,
                    "seed": seed,
                    "shadow_train_loss_before": train_before,
                    "shadow_train_loss_after": train_after,
                    "shadow_val_loss_before": val_before,
                    "shadow_val_loss_after": val_after,
                    "shadow_predicted_descent": pred_descent,
                    "shadow_actual_descent": train_before - train_after,
                    "shadow_val_descent": val_before - val_after,
                    "shadow_bad_step_bool": int(train_after > train_before),
                    "shadow_update_norm": float(update.norm().detach().cpu()) if update.numel() else 0.0,
                    "shadow_update_over_param": float(update.norm().detach().cpu()) / max(
                        1e-12, float(torch.cat([v.flatten().float() for v in before.values()]).norm())
                    ),
                    "input_update_norm": _mean_state(state.pure_role_update_norms.get("input", [])),
                    "block_update_norm_mean": _mean_state(state.pure_role_update_norms.get("block", [])),
                    "output_update_norm": _mean_state(state.pure_role_update_norms.get("output", [])),
                    "input_cos_raw_precond": _mean_state(state.pure_role_cos_raw_precond.get("input", []), float("nan")),
                    "block_cos_raw_precond_mean": _mean_state(state.pure_role_cos_raw_precond.get("block", []), float("nan")),
                    "output_cos_raw_precond": _mean_state(state.pure_role_cos_raw_precond.get("output", []), float("nan")),
                    "metric_condition_input": _mean_state(state.pure_role_metric_conditions.get("input", []), float("nan")),
                    "metric_condition_block": _mean_state(state.pure_role_metric_conditions.get("block", []), float("nan")),
                    "metric_condition_output": _mean_state(state.pure_role_metric_conditions.get("output", []), float("nan")),
                    "error": "",
                }
                rows.append(row)
                print(
                    f"{package} {dataset} {label} seed={seed} "
                    f"trainΔ={row['shadow_actual_descent']:.4g} valΔ={row['shadow_val_descent']:.4g} "
                    f"bad={row['shadow_bad_step_bool']}"
                )
                save_json(run_dir / f"{dataset}_{label}_seed{seed}.json", row)
                write_csv(out_dir / "shadow_runs.csv", rows)
    write_csv(out_dir / "runs.csv", rows)
    save_json(out_dir / "aggregate_summary.json", {"package": package, "new_runs": len(rows)})
    return rows


def _mean_state(values: List[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else default


def add_args() -> argparse.ArgumentParser:
    p = add_v3_args()
    p.description = __doc__
    p.set_defaults(packages="V3_7_P0_CODE_AUDIT", out_dir=Path("results/gafu_v3_7"))
    return p


def main() -> int:
    args = add_args().parse_args()
    packages = parse_str_list(args.packages)
    original_fresh = args.fresh
    first = True
    all_rows: List[Dict[str, Any]] = []
    for package in packages:
        key = package.strip().upper().replace("-", "_")
        args.fresh = original_fresh and first
        if key == "V3_7_P1_SHADOW_STEP":
            all_rows = run_shadow(args, key)
        else:
            specs, seeds, meta = package_specs(args, package)
            all_rows = run_specs(args, specs, seeds, meta)
        first = False
    return 0 if all_rows is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
