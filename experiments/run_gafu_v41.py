#!/usr/bin/env python3
"""DG-KAN v4.1 runner: redesigned PureKAN functional updates.

This runner follows docs/DG-KAN_v4.1_FunctionalUpdate_Redesign_DeepPlan.md.
It intentionally keeps the heavy confirm phases conditional: P2 should be run
only for P1 survivors, and later phases only for P2/P3 survivors.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from dgkan_core import (
    PureKANClassifier,
    RuntimeState,
    TrainConfig,
    coefficient_named_params,
    ensure_dir,
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
from run_gafu_v38 import d6_global
from run_gafu_v39 import fng_spec as v39_fng_spec


def v41_epochs(dataset: str) -> int:
    return 12 if dataset == "Fashion-MNIST" else 8


def v41_common(dataset: str, **extra: Any) -> Dict[str, Any]:
    out = {
        "epochs": v41_epochs(dataset),
        "train_size": 6000,
        "val_size": 1000,
        "test_size": 1000,
        "hidden_dim": 64,
        "depth": 4,
        "basis_count": 16,
        "alpha_init": 1.5,
        "alpha_mode": "fixed1",
        "model_type": "pure_kan",
        "pure_norm_mode": "fixed",
        "norm_update_method": "none",
        "coeff_lr": 0.03,
        "lr": 1e-3,
        "batch_size": 256,
        "eval_batch_size": 512,
        "audit_batch_size": 64,
        "v3_gram_grid_size": 192,
        "trust_radius": 0.50,
    }
    out.update(extra)
    return out


def p0_common() -> Dict[str, Any]:
    return {
        "epochs": 1,
        "train_size": 512,
        "val_size": 128,
        "test_size": 128,
        "hidden_dim": 32,
        "depth": 2,
        "basis_count": 8,
        "batch_size": 256,
        "eval_batch_size": 256,
        "audit_batch_size": 64,
        "v3_gram_grid_size": 96,
    }


def pure_adamw(dataset: str, label: str = "PureKAN-AdamW", **extra: Any) -> Spec:
    overrides = v41_common(dataset)
    overrides.update(extra)
    return spec(dataset, label, "purekan_adamw", **overrides)


def pure_d0(dataset: str, label: str = "D0-allFullSobolev", **extra: Any) -> Spec:
    overrides = {
        **v41_common(dataset),
        "gafu_v3_enabled": True,
        "metric_mode": "grid",
        "branch_schedule": "none",
        "warmup_frac": 0.0,
        "v3_phase_mode": "hard",
        "v3_metric_active": "full_sobolev_gram",
        "v3_metric_transition": "full_sobolev_gram",
        "v3_metric_geometry": "full_sobolev_gram",
        "branch_boost": 1.0,
        "branch_final_scale": 1.0,
        "coeff_lr_boost": 1.0,
        "coeff_lr_final_mult": 1.0,
        "branch_max_active_frac": 0.0,
        "geometry_min_epochs": 999,
        "unified_optimizer_mode": "hybrid",
    }
    overrides.update(extra)
    return spec(dataset, label, "purekan_ufull", **overrides)


def pure_d6(dataset: str, label: str = "D6-allTaskAware", **extra: Any) -> Spec:
    overrides = v41_common(dataset)
    overrides.update(extra)
    return d6_global(dataset, label, **overrides)


def fng_leftfull(dataset: str, label: str = "F4-FNG-leftFull-right", **extra: Any) -> Spec:
    overrides = v41_common(dataset)
    overrides.update(extra)
    return v39_fng_spec(dataset, label, mode="leftfull", sob_lambda=0.02, **overrides)


def ftf_spec(
    dataset: str,
    label: str,
    *,
    ftf_mode: str,
    tau: float = 0.10,
    ridge: float = 1e-2,
    sob_lambda: float = 1e-3,
    activation_trust: float = 0.20,
    coeff_lr: float = 1.0,
    output_mult: float = 1.0,
    block_mult: float = 1.0,
    input_mult: float = 1.0,
    **extra: Any,
) -> Spec:
    overrides = {
        **v41_common(dataset, coeff_lr=coeff_lr),
        "gafu_v3_enabled": True,
        "metric_mode": "grid",
        "branch_schedule": "none",
        "warmup_frac": 0.0,
        "v3_phase_mode": "hard",
        "v3_metric_active": "full_sobolev_gram",
        "v3_metric_transition": "full_sobolev_gram",
        "v3_metric_geometry": "full_sobolev_gram",
        "branch_boost": 1.0,
        "branch_final_scale": 1.0,
        "coeff_lr_boost": 1.0,
        "coeff_lr_final_mult": 1.0,
        "branch_max_active_frac": 0.0,
        "geometry_min_epochs": 999,
        "unified_optimizer_mode": "hybrid",
        "ftf_enabled": True,
        "ftf_mode": ftf_mode,
        "ftf_tau": tau,
        "ftf_ridge": ridge,
        "ftf_sob_lambda": sob_lambda,
        "ftf_activation_trust": activation_trust,
        "ftf_output_lr_mult": output_mult,
        "ftf_block_lr_mult": block_mult,
        "ftf_input_lr_mult": input_mult,
        "pure_input_metric": "ftf",
        "pure_shallow_metric": "ftf",
        "pure_deep_metric": "ftf",
        "pure_block_metric": "phase",
        "pure_output_metric": "ftf",
    }
    overrides.update(extra)
    return spec(dataset, label, "purekan_ftf", **overrides)


def fc_adam_spec(dataset: str, label: str = "FC-Adam-one-step", **extra: Any) -> Spec:
    overrides = {
        **v41_common(dataset, coeff_lr=3e-3),
        "gafu_v3_enabled": True,
        "metric_mode": "grid",
        "branch_schedule": "none",
        "warmup_frac": 0.0,
        "v3_phase_mode": "hard",
        "v3_metric_active": "full_sobolev_gram",
        "v3_metric_transition": "full_sobolev_gram",
        "v3_metric_geometry": "full_sobolev_gram",
        "branch_boost": 1.0,
        "branch_final_scale": 1.0,
        "branch_max_active_frac": 0.0,
        "geometry_min_epochs": 999,
        "fc_adam_enabled": True,
        "fc_adam_sob_decay": 1e-4,
        "pure_input_metric": "fc_adam",
        "pure_shallow_metric": "fc_adam",
        "pure_deep_metric": "fc_adam",
        "pure_block_metric": "phase",
        "pure_output_metric": "fc_adam",
    }
    overrides.update(extra)
    return spec(dataset, label, "purekan_fc_adam", **overrides)


def ftr_spec(dataset: str, label: str = "FTR-CG-small", **extra: Any) -> Spec:
    overrides = {
        **v41_common(dataset, coeff_lr=0.03),
        "gafu_v3_enabled": True,
        "metric_mode": "grid",
        "branch_schedule": "none",
        "warmup_frac": 0.0,
        "v3_phase_mode": "hard",
        "v3_metric_active": "full_sobolev_gram",
        "v3_metric_transition": "full_sobolev_gram",
        "v3_metric_geometry": "full_sobolev_gram",
        "branch_boost": 1.0,
        "branch_final_scale": 1.0,
        "branch_max_active_frac": 0.0,
        "geometry_min_epochs": 999,
        "ftr_enabled": True,
        "ftr_cg_iters": 5,
        "ftr_rho": 1e-2,
        "ftr_sob_lambda": 1e-4,
        "ftr_trust_radius": 0.10,
        "pure_input_metric": "ftr_cg_small",
        "pure_shallow_metric": "ftr_cg_small",
        "pure_deep_metric": "ftr_cg_small",
        "pure_block_metric": "phase",
        "pure_output_metric": "ftr_cg_small",
    }
    overrides.update(extra)
    return spec(dataset, label, "purekan_ftr", **overrides)


def hybrid_ufull(dataset: str, label: str = "Hybrid-DGKAN-UFULL-f085", **extra: Any) -> Spec:
    overrides = v41_common(dataset)
    overrides.update(extra)
    return v35_ufull_f085_reference_spec(dataset, label=label, **overrides)


def _candidate_specs(dataset: str, **extra: Any) -> List[Spec]:
    return [
        pure_adamw(dataset, **extra),
        pure_d0(dataset, **extra),
        pure_d6(dataset, **extra),
        fng_leftfull(dataset, **extra),
        ftf_spec(dataset, "FTF-output-only", ftf_mode="output_only", **extra),
        ftf_spec(dataset, "FTF-blocks-output", ftf_mode="blocks_output", **extra),
        ftf_spec(dataset, "FTF-all-simultaneous", ftf_mode="all_simultaneous", **extra),
        ftf_spec(dataset, "FTF-all-sequential", ftf_mode="all_sequential", **extra),
        fc_adam_spec(dataset, **extra),
        ftr_spec(dataset, **extra),
    ]


def package_specs(args: argparse.Namespace, package: str) -> Tuple[List[Spec], List[int], Dict[str, Any]]:
    key = package.strip().upper().replace("-", "_")
    datasets = [dataset_name(item) for item in parse_str_list(args.datasets)]
    datasets = [d for d in datasets if d in {"MNIST", "Fashion-MNIST", "KMNIST"}]
    seeds = parse_int_list(args.seeds)
    specs: List[Spec] = []

    if key == "V4_1_P0_SMOKE":
        seeds = [0]
        for dataset in datasets:
            specs.extend(_candidate_specs(dataset, **p0_common()))
        return specs, seeds, {"package": key}

    if key == "V4_1_P2_MICRO":
        seeds = [0, 1, 2] if args.seeds == add_args().get_default("seeds") else seeds
        wanted = set(parse_str_list(args.methods))
        for dataset in datasets:
            all_specs = [
                pure_adamw(dataset),
                pure_d0(dataset),
                pure_d6(dataset),
                fng_leftfull(dataset),
                ftf_spec(dataset, "FTF-output-only", ftf_mode="output_only"),
                ftf_spec(dataset, "FTF-blocks-output", ftf_mode="blocks_output"),
                ftf_spec(dataset, "FTF-all-simultaneous", ftf_mode="all_simultaneous"),
                ftf_spec(dataset, "FTF-all-sequential", ftf_mode="all_sequential"),
                fc_adam_spec(dataset),
                ftr_spec(dataset),
                hybrid_ufull(dataset),
                mlp_spec(dataset, epochs=v41_epochs(dataset), hidden_dim=64, depth=4),
            ]
            if wanted:
                specs.extend([item for item in all_specs if item["label"] in wanted or item["base_method"] in wanted])
            else:
                specs.extend(all_specs)
        return specs, seeds, {"package": key}

    if key == "V4_1_P3_REFINEMENT":
        seeds = [0, 1, 2]
        wanted = set(parse_str_list(args.methods))
        for dataset in datasets:
            refined = [
                ftf_spec(dataset, "FTF-all-seq-tau005-ridge003", ftf_mode="all_sequential", tau=0.05, ridge=0.03),
                ftf_spec(dataset, "FTF-blocks-output-tau005", ftf_mode="blocks_output", tau=0.05, ridge=0.03),
                fc_adam_spec(dataset, "FC-Adam-lr001", coeff_lr=1e-3),
                ftr_spec(dataset, "FTR-CG-small-r005", ftr_trust_radius=0.05),
            ]
            if wanted:
                specs.extend([item for item in refined if item["label"] in wanted or item["base_method"] in wanted])
            else:
                specs.extend(refined)
        return specs, seeds, {"package": key}

    raise ValueError(f"unknown v4.1 package: {package}")


def _shadow_specs() -> List[Tuple[str, Dict[str, Any]]]:
    return [
        ("AdamW-one-step", {"kind": "adamw"}),
        ("D0-allFullSobolev", {"metric": "full_sobolev_gram"}),
        ("D6-allTaskAware", {"tfu": True}),
        ("F4-FNG-leftFull-right", {"fng": "leftfull"}),
        ("FTF-output-only", {"ftf": "output_only"}),
        ("FTF-blocks-output", {"ftf": "blocks_output"}),
        ("FTF-all-simultaneous", {"ftf": "all_simultaneous"}),
        ("FTF-all-sequential", {"ftf": "all_sequential"}),
        ("FC-Adam-one-step", {"fc_adam": True}),
        ("FTR-CG-small", {"ftr": True}),
    ]


def _mean(values: List[float], default: float = float("nan")) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else default


def _max(values: List[float], default: float = float("nan")) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return max(vals) if vals else default


def _flat_coeff_update(before: Dict[str, torch.Tensor], model: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    grads: List[torch.Tensor] = []
    updates: List[torch.Tensor] = []
    for name, p in coefficient_named_params(model):
        grads.append((p.grad.detach().flatten().float().cpu() if p.grad is not None else torch.zeros(p.numel())))
        updates.append((p.detach().cpu() - before[name]).flatten().float())
    return torch.cat(grads) if grads else torch.empty(0), torch.cat(updates) if updates else torch.empty(0)


def _cfg_for_shadow(dataset: str, seed: int, label: str, overrides: Dict[str, Any], args: argparse.Namespace) -> TrainConfig:
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
        batch_size=256,
        eval_batch_size=256,
        audit_batch_size=64,
        hidden_dim=64,
        depth=4,
        basis_count=16,
        alpha_init=1.5,
        alpha_mode="fixed1",
        model_type="pure_kan",
        pure_norm_mode="fixed",
        norm_update_method="none",
        gafu_v3_enabled=True,
        metric_mode="grid",
        branch_schedule="none",
        branch_max_active_frac=0.0,
        geometry_min_epochs=999,
        v3_phase_mode="hard",
        v3_metric_active=overrides.get("metric", "full_sobolev_gram"),
        v3_metric_transition=overrides.get("metric", "full_sobolev_gram"),
        v3_metric_geometry=overrides.get("metric", "full_sobolev_gram"),
        v3_gram_grid_size=128,
        coeff_lr=float(overrides.get("coeff_lr", 0.03)),
        trust_radius=0.50,
    )
    if overrides.get("tfu"):
        cfg.tfu_enabled = True
        cfg.tfu_sob_lambda = 0.03
        cfg.pure_input_metric = "tfu_data_task_diag"
        cfg.pure_shallow_metric = "tfu_data_task_diag"
        cfg.pure_deep_metric = "tfu_task_diag"
        cfg.pure_block_metric = "phase"
        cfg.pure_output_metric = "tfu_task_diag"
    if overrides.get("fng"):
        cfg.fng_enabled = True
        cfg.fng_mode = str(overrides.get("fng"))
        cfg.fng_sob_lambda = 0.02
        metric = "fng_leftfull_right"
        cfg.pure_input_metric = metric
        cfg.pure_shallow_metric = metric
        cfg.pure_deep_metric = metric
        cfg.pure_block_metric = "phase"
        cfg.pure_output_metric = metric
    if overrides.get("ftf"):
        cfg.ftf_enabled = True
        cfg.ftf_mode = str(overrides.get("ftf"))
        cfg.ftf_tau = 0.10
        cfg.ftf_sob_lambda = 1e-3
        cfg.ftf_ridge = 1e-2
        cfg.ftf_activation_trust = 0.20
        cfg.coeff_lr = 1.0
        cfg.pure_input_metric = "ftf"
        cfg.pure_shallow_metric = "ftf"
        cfg.pure_deep_metric = "ftf"
        cfg.pure_block_metric = "phase"
        cfg.pure_output_metric = "ftf"
    if overrides.get("fc_adam"):
        cfg.fc_adam_enabled = True
        cfg.coeff_lr = 3e-3
        cfg.fc_adam_sob_decay = 1e-4
        cfg.pure_input_metric = "fc_adam"
        cfg.pure_shallow_metric = "fc_adam"
        cfg.pure_deep_metric = "fc_adam"
        cfg.pure_block_metric = "phase"
        cfg.pure_output_metric = "fc_adam"
    if overrides.get("ftr"):
        cfg.ftr_enabled = True
        cfg.coeff_lr = 0.03
        cfg.ftr_cg_iters = 5
        cfg.ftr_rho = 1e-2
        cfg.ftr_sob_lambda = 1e-4
        cfg.ftr_trust_radius = 0.10
        cfg.pure_input_metric = "ftr_cg_small"
        cfg.pure_shallow_metric = "ftr_cg_small"
        cfg.pure_deep_metric = "ftr_cg_small"
        cfg.pure_block_metric = "phase"
        cfg.pure_output_metric = "ftr_cg_small"
    return cfg


def run_shadow(args: argparse.Namespace, package: str) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    run_dir = ensure_dir(out_dir / "shadow")
    rows: List[Dict[str, Any]] = []
    datasets = [dataset_name(item) for item in parse_str_list(args.datasets)]
    datasets = [d for d in datasets if d in {"MNIST", "Fashion-MNIST", "KMNIST"}]
    seeds = parse_int_list(args.seeds) or [0]
    device = get_device(args.device)
    for dataset in datasets:
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
            for label, overrides in _shadow_specs():
                model = PureKANClassifier(
                    bundle.input_dim,
                    bundle.num_classes,
                    hidden_dim=64,
                    depth=4,
                    basis_count=16,
                    alpha_init=1.5,
                    alpha_mode="fixed1",
                    norm_mode="fixed",
                ).to(device)
                idx = torch.arange(min(256, len(bundle.x_train)))
                xb = bundle.x_train[idx].to(device)
                yb = bundle.y_train[idx].to(device)
                val_x = bundle.x_val.to(device)
                val_y = bundle.y_val.to(device)
                with torch.no_grad():
                    train_before = float(F.cross_entropy(model(xb), yb).detach().cpu())
                    val_before = float(F.cross_entropy(model(val_x), val_y).detach().cpu())

                cfg = _cfg_for_shadow(dataset, seed, label, overrides, args)
                before = {name: p.detach().cpu().clone() for name, p in coefficient_named_params(model)}
                model.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                state = RuntimeState(phase="GEOMETRY", current_branch_scale=1.0, branch_switched=True)
                if overrides.get("kind") == "adamw":
                    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                    opt.step()
                    state = RuntimeState()
                else:
                    functional_coeff_step(model, cfg, state, step_idx=1, total_steps=1)

                grad, update = _flat_coeff_update(before, model)
                with torch.no_grad():
                    train_after = float(F.cross_entropy(model(xb), yb).detach().cpu())
                    val_after = float(F.cross_entropy(model(val_x), val_y).detach().cpu())
                pred_descent = float(-(grad * update).sum().detach().cpu()) if grad.numel() else float("nan")
                actual_descent = train_before - train_after
                val_descent = val_before - val_after
                fit_values: List[float] = []
                delta_values: List[float] = []
                for values in state.ftf_fit_r2.values():
                    fit_values.extend(values)
                for values in state.ftf_delta_norm_ratios.values():
                    delta_values.extend(values)
                row = {
                    "stage": "DG-KAN-v4.1",
                    "package": package,
                    "dataset": dataset,
                    "method": label,
                    "seed": seed,
                    "shadow_train_loss_before": train_before,
                    "shadow_train_loss_after": train_after,
                    "shadow_val_loss_before": val_before,
                    "shadow_val_loss_after": val_after,
                    "shadow_predicted_descent": pred_descent,
                    "shadow_actual_descent": actual_descent,
                    "shadow_val_descent": val_descent,
                    "shadow_bad_step_bool": int(train_after > train_before),
                    "shadow_update_norm": float(update.norm().detach().cpu()) if update.numel() else 0.0,
                    "shadow_raw_update_cos": float(torch.nn.functional.cosine_similarity(-grad, update, dim=0).detach().cpu())
                    if grad.numel() and update.norm() > 0
                    else float("nan"),
                    "ftf_fit_R2_mean": _mean(fit_values),
                    "ftf_fit_R2_min": min(fit_values) if fit_values else float("nan"),
                    "ftf_delta_norm_ratio_max": _max(delta_values),
                    "ftf_trust_clip_rate": state.ftf_trust_clip_count / max(1, state.ftf_update_count),
                    "fc_adam_reconstruction_error_mean": _mean(state.fc_adam_reconstruction_errors),
                    "fc_adam_grad_chain_error_mean": _mean(state.fc_adam_grad_chain_errors),
                    "ftr_cg_residual_mean": _mean(state.ftr_cg_residuals),
                    "ftr_predicted_change_norm_mean": _mean(state.ftr_predicted_change_norms),
                    "error": "",
                }
                rows.append(row)
                print(
                    f"{package} {dataset} {label} seed={seed} "
                    f"trainΔ={actual_descent:.4g} valΔ={val_descent:.4g} pred={pred_descent:.4g} "
                    f"R2={row['ftf_fit_R2_mean']:.3g} bad={row['shadow_bad_step_bool']}"
                )
                save_json(run_dir / f"{dataset}_{label}_seed{seed}.json", row)
                write_csv(out_dir / "runs.csv", rows)
    save_json(out_dir / "aggregate_summary.json", {"package": package, "new_runs": len(rows)})
    return rows


def add_args() -> argparse.ArgumentParser:
    p = add_v3_args()
    p.description = __doc__
    p.set_defaults(packages="V4_1_P0_SMOKE", datasets="MNIST,Fashion-MNIST,KMNIST", out_dir=Path("results/v4_1"))
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
        if key == "V4_1_P1_SHADOW":
            all_rows = run_shadow(args, key)
        else:
            specs, seeds, meta = package_specs(args, package)
            all_rows = run_specs(args, specs, seeds, meta)
        first = False
    return 0 if all_rows is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
