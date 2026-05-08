#!/usr/bin/env python3
"""DG-KAN v3.9 runner: normalization and FNG/KFAC-style PureKAN updates."""

from __future__ import annotations

import argparse
import copy
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
    config_from_args,
    ensure_dir,
    functional_coeff_step,
    functional_norm_step,
    get_device,
    load_vision_bundle,
    norm_named_params,
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


def v39_epochs(dataset: str) -> int:
    return 30 if dataset == "Fashion-MNIST" else 20


def v39_common(dataset: str, **extra: Any) -> Dict[str, Any]:
    out = {
        "epochs": v39_epochs(dataset),
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
        "norm_lr": 0.01,
        "batch_size": 256,
        "eval_batch_size": 512,
        "audit_batch_size": 64,
        "v3_gram_grid_size": 256,
        "trust_radius": 0.50,
        "pure_norm_mode": "fixed",
        "norm_update_method": "none",
    }
    out.update(extra)
    return out


def pure_adamw(dataset: str, label: str = "PureKAN-FixedNorm-AdamW", **extra: Any) -> Spec:
    overrides = {**v39_common(dataset), "model_type": "pure_kan"}
    overrides.update(extra)
    return spec(dataset, label, "purekan_adamw", **overrides)


def pure_d0(dataset: str, label: str = "PureKAN-FixedNorm-D0-allFullSobolev", **extra: Any) -> Spec:
    overrides = {
        **v39_common(dataset),
        "model_type": "pure_kan",
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


def pure_d6(dataset: str, label: str = "PureKAN-FixedNorm-D6-allTaskAware", **extra: Any) -> Spec:
    overrides = v39_common(dataset)
    overrides.update(extra)
    return d6_global(dataset, label, **overrides)


def fng_spec(
    dataset: str,
    label: str,
    *,
    mode: str,
    sob_lambda: float = 0.02,
    norm_mode: str = "fixed",
    norm_update: str = "none",
    momentum_beta: float = 0.0,
    step_scale: float = 1.0,
    **extra: Any,
) -> Spec:
    metric = {
        "right": "fng_right",
        "leftdiag": "fng_leftdiag_right",
        "leftfull": "fng_leftfull_right",
        "leftlowrank": "fng_leftlowrank_right",
    }[mode]
    overrides = {
        **v39_common(dataset),
        "model_type": "pure_kan",
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
        "fng_enabled": True,
        "fng_mode": mode,
        "fng_sob_lambda": sob_lambda,
        "fng_step_scale": step_scale,
        "fng_direction_momentum_beta": momentum_beta,
        "fng_metric_normalized_momentum": momentum_beta > 0,
        "pure_input_metric": metric,
        "pure_shallow_metric": metric,
        "pure_deep_metric": metric,
        "pure_output_metric": metric,
        "pure_norm_mode": norm_mode,
        "norm_update_method": norm_update,
    }
    overrides.update(extra)
    return spec(dataset, label, "purekan_fng", **overrides)


def hybrid_ufull(dataset: str, label: str = "Hybrid-DGKAN-UFULL-f085", **extra: Any) -> Spec:
    overrides = v39_common(dataset)
    overrides.update(extra)
    return v35_ufull_f085_reference_spec(dataset, label=label, **overrides)


def p0_common() -> Dict[str, Any]:
    return {
        "train_size": 512,
        "val_size": 128,
        "test_size": 128,
        "epochs": 1,
        "hidden_dim": 32,
        "depth": 2,
        "basis_count": 8,
        "batch_size": 256,
        "eval_batch_size": 256,
        "audit_batch_size": 64,
        "v3_gram_grid_size": 128,
    }


def package_specs(args: argparse.Namespace, package: str) -> Tuple[List[Spec], List[int], Dict[str, Any]]:
    key = package.strip().upper().replace("-", "_")
    datasets = [dataset_name(item) for item in parse_str_list(args.datasets)]
    datasets = [d for d in datasets if d in {"MNIST", "Fashion-MNIST", "KMNIST"}]
    specs: List[Spec] = []
    seeds = parse_int_list(args.seeds)

    if key == "V3_9_P0_SMOKE":
        seeds = [0]
        for dataset in datasets:
            common = p0_common()
            specs.extend(
                [
                    pure_adamw(dataset, "PureKAN-FixedNorm-AdamW", **common),
                    pure_d0(dataset, **common),
                    pure_d6(dataset, **common),
                    pure_adamw(
                        dataset,
                        "PureKAN-AffineNorm-AdamW-LN",
                        pure_norm_mode="affine_channel",
                        norm_update_method="adamw",
                        **common,
                    ),
                    pure_d6(
                        dataset,
                        "PureKAN-AffineNorm-FDiag-LN",
                        pure_norm_mode="affine_channel",
                        norm_update_method="fdiag",
                        **common,
                    ),
                    pure_d6(
                        dataset,
                        "PureKAN-ScalarGainNorm-FDiag",
                        pure_norm_mode="scalar_gain",
                        norm_update_method="fdiag",
                        **common,
                    ),
                    fng_spec(dataset, "PureKAN-FixedNorm-FNG-right", mode="right", **common),
                    fng_spec(dataset, "PureKAN-FixedNorm-FNG-leftDiagRight", mode="leftdiag", **common),
                    fng_spec(dataset, "PureKAN-FixedNorm-FNG-leftFullRight", mode="leftfull", **common),
                ]
            )
        return specs, seeds, {"package": key}

    if key == "V3_9_P2_NORM":
        seeds = [0, 1, 2] if args.seeds == add_args().get_default("seeds") else seeds
        for dataset in datasets:
            specs.extend(
                [
                    pure_adamw(dataset, "PureKAN-FixedNorm-AdamW"),
                    pure_adamw(
                        dataset,
                        "PureKAN-AffineNorm-AdamW-LN",
                        pure_norm_mode="affine_channel",
                        norm_update_method="adamw",
                    ),
                    pure_adamw(dataset, "PureKAN-AffineNorm-FDiag-LN", pure_norm_mode="affine_channel", norm_update_method="fdiag"),
                    pure_adamw(dataset, "PureKAN-ScalarGainNorm-FDiag", pure_norm_mode="scalar_gain", norm_update_method="fdiag"),
                    pure_adamw(dataset, "PureKAN-NoNorm-AdamW", pure_norm_mode="none", epochs=5),
                    pure_d6(dataset),
                    pure_d6(
                        dataset,
                        "PureKAN-AffineNorm-AdamW-LN-D6",
                        pure_norm_mode="affine_channel",
                        norm_update_method="adamw",
                    ),
                    pure_d6(dataset, "PureKAN-AffineNorm-FDiag-LN-D6", pure_norm_mode="affine_channel", norm_update_method="fdiag"),
                    fng_spec(dataset, "PureKAN-FixedNorm-FNG-leftFullRight", mode="leftfull"),
                    fng_spec(
                        dataset,
                        "PureKAN-AffineNorm-AdamW-LN-FNG-leftFullRight",
                        mode="leftfull",
                        norm_mode="affine_channel",
                        norm_update="adamw",
                    ),
                    fng_spec(
                        dataset,
                        "PureKAN-AffineNorm-FDiag-LN-FNG-leftFullRight",
                        mode="leftfull",
                        norm_mode="affine_channel",
                        norm_update="fdiag",
                    ),
                ]
            )
        return specs, seeds, {"package": key}

    if key == "V3_9_P3_FNG":
        seeds = [0, 1, 2] if args.seeds == add_args().get_default("seeds") else seeds
        wanted = set(parse_str_list(args.methods))
        for dataset in datasets:
            all_specs = [
                pure_adamw(dataset, "PureKAN-AdamW"),
                pure_d0(dataset, "D0-allFullSobolev"),
                pure_d6(dataset, "D6-allTaskAware"),
                fng_spec(dataset, "F2-FNG-right-only", mode="right"),
                fng_spec(dataset, "F3-FNG-leftDiag-right", mode="leftdiag"),
                fng_spec(dataset, "F4-FNG-leftFull-right", mode="leftfull"),
                fng_spec(dataset, "F4-FNG-leftFull-right-noSob", mode="leftfull", sob_lambda=0.0),
                fng_spec(dataset, "F4-FNG-leftFull-right-lowSob", mode="leftfull", sob_lambda=0.005),
                fng_spec(dataset, "F5-FNG-leftLowRank-right", mode="leftlowrank"),
            ]
            if wanted:
                specs.extend([s for s in all_specs if s["label"] in wanted or s["base_method"] in wanted])
            else:
                specs.extend(all_specs)
        return specs, seeds, {"package": key}

    if key == "V3_9_P4_DYNAMICS":
        seeds = [0, 1, 2]
        base_methods = parse_str_list(args.methods)
        for dataset in datasets:
            for method in base_methods:
                if method == "F4-FNG-leftFull-right":
                    specs.extend(
                        [
                            fng_spec(dataset, "F4-FNG-leftFull-right", mode="leftfull"),
                            fng_spec(dataset, "F4-FNG-leftFull-right-ema08", mode="leftfull", momentum_beta=0.8),
                            fng_spec(dataset, "F4-FNG-leftFull-right-ema09", mode="leftfull", momentum_beta=0.9),
                        ]
                    )
                elif method == "F3-FNG-leftDiag-right":
                    specs.extend(
                        [
                            fng_spec(dataset, "F3-FNG-leftDiag-right", mode="leftdiag"),
                            fng_spec(dataset, "F3-FNG-leftDiag-right-ema08", mode="leftdiag", momentum_beta=0.8),
                            fng_spec(dataset, "F3-FNG-leftDiag-right-ema09", mode="leftdiag", momentum_beta=0.9),
                        ]
                    )
        return specs, seeds, {"package": key}

    raise ValueError(f"unknown v3.9 package: {package}")


def _shadow_specs() -> List[Tuple[str, Dict[str, Any]]]:
    return [
        ("AdamW-one-step", {"kind": "adamw"}),
        ("D0-allFullSobolev", {"metric": "full_sobolev_gram"}),
        ("D6-allTaskAware", {"tfu": True}),
        ("F2-FNG-right-only", {"fng": "right"}),
        ("F3-FNG-leftDiag-right", {"fng": "leftdiag"}),
        ("F4-FNG-leftFull-right", {"fng": "leftfull"}),
        ("F4-FNG-leftFull-right-noSob", {"fng": "leftfull", "sob": 0.0}),
        ("F4-FNG-leftFull-right-lowSob", {"fng": "leftfull", "sob": 0.005}),
        ("F4-FNG-leftFull-right+AffineNorm-FDiag", {"fng": "leftfull", "norm": "affine_channel", "norm_update": "fdiag"}),
        ("F4-FNG-leftFull-right+AffineNorm-AdamW-LN", {"fng": "leftfull", "norm": "affine_channel", "norm_update": "adamw"}),
    ]


def _flat_update(before: Dict[str, torch.Tensor], model: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    grads: List[torch.Tensor] = []
    updates: List[torch.Tensor] = []
    tracked = dict(coefficient_named_params(model) + norm_named_params(model))
    for name, p in tracked.items():
        grads.append((p.grad.detach().flatten().float().cpu() if p.grad is not None else torch.zeros(p.numel())))
        updates.append((p.detach().cpu() - before[name]).flatten().float())
    return torch.cat(grads) if grads else torch.empty(0), torch.cat(updates) if updates else torch.empty(0)


def _mean_state(values: List[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else default


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
                norm_mode = overrides.get("norm", "fixed")
                model = PureKANClassifier(
                    bundle.input_dim,
                    bundle.num_classes,
                    hidden_dim=96,
                    depth=4,
                    basis_count=24,
                    alpha_init=1.5,
                    alpha_mode="fixed1",
                    norm_mode=norm_mode,
                ).to(device)
                idx = torch.arange(min(256, len(bundle.x_train)))
                xb = bundle.x_train[idx].to(device)
                yb = bundle.y_train[idx].to(device)
                val_x = bundle.x_val.to(device)
                val_y = bundle.y_val.to(device)
                with torch.no_grad():
                    train_before = float(F.cross_entropy(model(xb), yb).detach().cpu())
                    val_before = float(F.cross_entropy(model(val_x), val_y).detach().cpu())

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
                    pure_norm_mode=norm_mode,
                    norm_update_method=overrides.get("norm_update", "none"),
                    gafu_v3_enabled=True,
                    branch_schedule="none",
                    branch_max_active_frac=0.0,
                    v3_phase_mode="hard",
                    v3_metric_active=overrides.get("metric", "full_sobolev_gram"),
                    v3_metric_transition=overrides.get("metric", "full_sobolev_gram"),
                    v3_metric_geometry=overrides.get("metric", "full_sobolev_gram"),
                    v3_gram_grid_size=128,
                    coeff_lr=0.03,
                    tfu_enabled=bool(overrides.get("tfu", False)),
                    fng_enabled=bool(overrides.get("fng", "")),
                    fng_mode=str(overrides.get("fng", "none")),
                    fng_sob_lambda=float(overrides.get("sob", 0.02)),
                )
                if cfg.tfu_enabled:
                    cfg.pure_input_metric = "tfu_data_task_diag"
                    cfg.pure_shallow_metric = "tfu_data_task_diag"
                    cfg.pure_deep_metric = "tfu_task_diag"
                    cfg.pure_output_metric = "tfu_task_diag"
                if cfg.fng_enabled:
                    metric = {
                        "right": "fng_right",
                        "leftdiag": "fng_leftdiag_right",
                        "leftfull": "fng_leftfull_right",
                    }[str(overrides.get("fng"))]
                    cfg.pure_input_metric = metric
                    cfg.pure_shallow_metric = metric
                    cfg.pure_deep_metric = metric
                    cfg.pure_output_metric = metric

                tracked = dict(coefficient_named_params(model) + norm_named_params(model))
                before = {name: p.detach().cpu().clone() for name, p in tracked.items()}
                model.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                state = RuntimeState(phase="GEOMETRY", current_branch_scale=1.0, branch_switched=True)
                if overrides.get("kind") == "adamw":
                    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                    opt.step()
                    state = RuntimeState()
                else:
                    if cfg.norm_update_method.lower().replace("-", "_") == "adamw":
                        norm_params = [p for _, p in norm_named_params(model)]
                        if norm_params:
                            opt = torch.optim.AdamW(norm_params, lr=1e-3, weight_decay=1e-4)
                            opt.step()
                    else:
                        functional_norm_step(model, cfg, state, progress=0.0)
                    functional_coeff_step(model, cfg, state, step_idx=1, total_steps=1)

                grad, update = _flat_update(before, model)
                with torch.no_grad():
                    train_after = float(F.cross_entropy(model(xb), yb).detach().cpu())
                    val_after = float(F.cross_entropy(model(val_x), val_y).detach().cpu())
                row = {
                    "stage": "DG-KAN-v3.9",
                    "package": package,
                    "dataset": dataset,
                    "method": label,
                    "seed": seed,
                    "shadow_train_loss_before": train_before,
                    "shadow_train_loss_after": train_after,
                    "shadow_val_loss_before": val_before,
                    "shadow_val_loss_after": val_after,
                    "shadow_predicted_descent": float(-(grad * update).sum().detach().cpu()) if grad.numel() else float("nan"),
                    "shadow_actual_descent": train_before - train_after,
                    "shadow_val_descent": val_before - val_after,
                    "shadow_bad_step_bool": int(train_after > train_before),
                    "shadow_update_norm": float(update.norm().detach().cpu()) if update.numel() else 0.0,
                    "input_cos_raw_precond": _mean_state(state.pure_role_cos_raw_precond.get("input", []), float("nan")),
                    "block_cos_raw_precond_mean": _mean_state(state.pure_role_cos_raw_precond.get("block", []), float("nan")),
                    "output_cos_raw_precond": _mean_state(state.pure_role_cos_raw_precond.get("output", []), float("nan")),
                    "fng_input_combined_condition": _mean_state(state.fng_combined_conditions.get("input", []), float("nan")),
                    "fng_block_combined_condition": _mean_state(state.fng_combined_conditions.get("block", []), float("nan")),
                    "fng_output_combined_condition": _mean_state(state.fng_combined_conditions.get("output", []), float("nan")),
                    "norm_mode": norm_mode,
                    "norm_update_method": cfg.norm_update_method,
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


def add_args() -> argparse.ArgumentParser:
    p = add_v3_args()
    p.description = __doc__
    p.set_defaults(packages="V3_9_P0_SMOKE", datasets="MNIST,Fashion-MNIST,KMNIST", out_dir=Path("results/gafu_v3_9"))
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
        if key == "V3_9_P1_SHADOW":
            all_rows = run_shadow(args, key)
        else:
            specs, seeds, meta = package_specs(args, package)
            all_rows = run_specs(args, specs, seeds, meta)
        first = False
    return 0 if all_rows is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
