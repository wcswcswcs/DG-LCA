#!/usr/bin/env python3
"""DG-KAN v3.8 runner: GlobalTFU and depth-wise PureKAN functional updates."""

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


def v38_epochs(dataset: str) -> int:
    return 30 if dataset == "Fashion-MNIST" else 20


def v38_common(dataset: str, **extra: Any) -> Dict[str, Any]:
    out = {
        "epochs": v38_epochs(dataset),
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
    overrides = {**v38_common(dataset), "model_type": "pure_kan"}
    overrides.update(extra)
    return spec(dataset, label, "purekan_adamw", **overrides)


def pure_full_sobolev(dataset: str, label: str = "D0-allFullSobolev-alphaFixed1", **extra: Any) -> Spec:
    overrides = {
        **v38_common(dataset),
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
        "tfu_enabled": False,
    }
    overrides.update(extra)
    return spec(dataset, label, "purekan_ufull", **overrides)


def pure_tfu(
    dataset: str,
    label: str,
    *,
    input_metric: str,
    shallow_metric: str,
    deep_metric: str,
    output_metric: str,
    sob_lambda: float = 0.03,
    input_sob_lambda: float = -1.0,
    shallow_sob_lambda: float = -1.0,
    deep_sob_lambda: float = -1.0,
    output_sob_lambda: float = -1.0,
    coeff_lr: float = 0.03,
    input_lr_mult: float = 1.0,
    shallow_lr_mult: float = 1.0,
    deep_lr_mult: float = 1.0,
    output_lr_mult: float = 1.0,
    safeguard: str = "off",
    **extra: Any,
) -> Spec:
    overrides = {
        **v38_common(dataset, coeff_lr=coeff_lr),
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
        "tfu_enabled": True,
        "tfu_sob_lambda": sob_lambda,
        "tfu_input_sob_lambda": input_sob_lambda,
        "tfu_shallow_sob_lambda": shallow_sob_lambda,
        "tfu_deep_sob_lambda": deep_sob_lambda,
        "tfu_output_sob_lambda": output_sob_lambda,
        "tfu_rho": 1e-3,
        "tfu_metric_clamp_min": 0.05,
        "tfu_metric_clamp_max": 20.0,
        "tfu_safeguard_mode": safeguard,
        "tfu_safeguard_cos_min": 0.05,
        "pure_input_metric": input_metric,
        "pure_shallow_metric": shallow_metric,
        "pure_deep_metric": deep_metric,
        "pure_block_metric": "phase",
        "pure_output_metric": output_metric,
        "pure_input_lr_mult": input_lr_mult,
        "pure_shallow_lr_mult": shallow_lr_mult,
        "pure_deep_lr_mult": deep_lr_mult,
        "pure_output_lr_mult": output_lr_mult,
    }
    overrides.update(extra)
    return spec(dataset, label, "purekan_tfu", **overrides)


def hybrid_ufull(dataset: str, label: str = "Hybrid-DGKAN-UFULL-f085-alphaFixed1", **extra: Any) -> Spec:
    overrides = v38_common(dataset)
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


def d6_global(
    dataset: str,
    label: str = "D6-allTaskAware-alphaFixed1",
    *,
    sob_lambda: float = 0.03,
    safeguard: str = "off",
    **extra: Any,
) -> Spec:
    return pure_tfu(
        dataset,
        label,
        input_metric="tfu_data_task_diag",
        shallow_metric="tfu_data_task_diag",
        deep_metric="tfu_task_diag",
        output_metric="tfu_task_diag",
        sob_lambda=sob_lambda,
        safeguard=safeguard,
        **extra,
    )


def d1_front_task(dataset: str, label: str = "D1-frontTask-backSob-alphaFixed1", **extra: Any) -> Spec:
    return pure_tfu(
        dataset,
        label,
        input_metric="tfu_data_task_diag",
        shallow_metric="tfu_data_task_diag",
        deep_metric="full_sobolev_gram",
        output_metric="full_sobolev_gram",
        sob_lambda=0.03,
        **extra,
    )


def d2_back_task(dataset: str, label: str = "D2-frontSob-backTask-alphaFixed1", **extra: Any) -> Spec:
    return pure_tfu(
        dataset,
        label,
        input_metric="basis_diag_gram",
        shallow_metric="full_sobolev_gram",
        deep_metric="tfu_task_diag",
        output_metric="tfu_task_diag",
        sob_lambda=0.03,
        **extra,
    )


def d3_io_task(dataset: str, label: str = "D3-taskSobTask-alphaFixed1", **extra: Any) -> Spec:
    return pure_tfu(
        dataset,
        label,
        input_metric="tfu_data_task_diag",
        shallow_metric="full_sobolev_gram",
        deep_metric="full_sobolev_gram",
        output_metric="tfu_task_diag",
        sob_lambda=0.03,
        **extra,
    )


def d7_io_task_middle_sob(dataset: str, label: str = "D7-inputOutputTask-middleSob-alphaFixed1", **extra: Any) -> Spec:
    return pure_tfu(
        dataset,
        label,
        input_metric="tfu_data_task_diag",
        shallow_metric="full_sobolev_gram",
        deep_metric="full_sobolev_gram",
        output_metric="tfu_task_diag",
        sob_lambda=0.03,
        **extra,
    )


def d8_mixed(dataset: str, label: str = "D8-inputDiag-blockTask-outputFisher-alphaFixed1", **extra: Any) -> Spec:
    return pure_tfu(
        dataset,
        label,
        input_metric="basis_diag_gram",
        shallow_metric="tfu_task_diag",
        deep_metric="tfu_task_diag",
        output_metric="tfu_task_diag",
        sob_lambda=0.03,
        **extra,
    )


def package_specs(args: argparse.Namespace, package: str) -> Tuple[List[Spec], List[int], Dict[str, Any]]:
    key = package.strip().upper().replace("-", "_")
    datasets = [dataset_name(item) for item in parse_str_list(args.datasets)]
    datasets = [d for d in datasets if d in {"MNIST", "Fashion-MNIST", "KMNIST"}]
    specs: List[Spec] = []
    seeds = parse_int_list(args.seeds)

    if key == "V3_8_P0_SMOKE":
        seeds = [0]
        for dataset in datasets:
            common = p0_common()
            specs.extend(
                [
                    pure_adamw(dataset, "PureKAN-AdamW-smoke-alphaFixed1", **common),
                    pure_full_sobolev(dataset, "D0-allFullSobolev-smoke-alphaFixed1", **common),
                    d6_global(dataset, "D6-allTaskAware-smoke-alphaFixed1", **common),
                    d1_front_task(dataset, "D1-frontTask-backSob-smoke-alphaFixed1", **common),
                    d2_back_task(dataset, "D2-frontSob-backTask-smoke-alphaFixed1", **common),
                    d3_io_task(dataset, "D3-taskSobTask-smoke-alphaFixed1", **common),
                    d7_io_task_middle_sob(dataset, "D7-inputOutputTask-middleSob-smoke-alphaFixed1", **common),
                ]
            )
        return specs, seeds, {"package": key}

    if key == "V3_8_P2_GLOBAL_TFU":
        seeds = [0, 1, 2] if args.seeds == add_args().get_default("seeds") else seeds
        for dataset in datasets:
            specs.extend(
                [
                    pure_adamw(dataset),
                    pure_full_sobolev(dataset),
                    d6_global(dataset),
                    d6_global(dataset, "D6-noSob-alphaFixed1", sob_lambda=0.0),
                    d6_global(dataset, "D6-lowSob-alphaFixed1", sob_lambda=0.01),
                    d6_global(dataset, "D6-withSafeguard-alphaFixed1", safeguard="cosine"),
                    hybrid_ufull(dataset),
                    mlp_spec(dataset, "MLP-AdamW-alphaFixed1", **v38_common(dataset)),
                ]
            )
        return specs, seeds, {"package": key}

    if key == "V3_8_P3_DEPTHWISE":
        seeds = [0, 1, 2] if args.seeds == add_args().get_default("seeds") else seeds
        for dataset in datasets:
            specs.extend(
                [
                    pure_adamw(dataset),
                    pure_full_sobolev(dataset),
                    d6_global(dataset),
                    d1_front_task(dataset),
                    d2_back_task(dataset),
                    d3_io_task(dataset),
                    d7_io_task_middle_sob(dataset),
                    d8_mixed(dataset),
                ]
            )
        return specs, seeds, {"package": key}

    if key == "V3_8_P4_CONFIRM3":
        seeds = [0, 1, 2]
        methods = parse_str_list(args.methods)
        for dataset in datasets:
            base = [pure_adamw(dataset), pure_full_sobolev(dataset), hybrid_ufull(dataset)]
            candidates = {
                "D6-allTaskAware-alphaFixed1": d6_global(dataset),
                "D6-withSafeguard-alphaFixed1": d6_global(dataset, "D6-withSafeguard-alphaFixed1", safeguard="cosine"),
                "D8-inputDiag-blockTask-outputFisher-alphaFixed1": d8_mixed(dataset),
                "D1-frontTask-backSob-alphaFixed1": d1_front_task(dataset),
                "D2-frontSob-backTask-alphaFixed1": d2_back_task(dataset),
                "D3-taskSobTask-alphaFixed1": d3_io_task(dataset),
                "D7-inputOutputTask-middleSob-alphaFixed1": d7_io_task_middle_sob(dataset),
            }
            selected = [candidates[m] for m in methods if m in candidates]
            specs.extend(base + selected)
        return specs, seeds, {"package": key}

    raise ValueError(f"unknown v3.8 package: {package}")


def _shadow_specs(dataset: str) -> List[Tuple[str, Dict[str, Any]]]:
    return [
        ("AdamW-one-step", {"kind": "adamw"}),
        ("D0-allFullSobolev", {"v3_metric_geometry": "full_sobolev_gram"}),
        (
            "D6-allTaskAware",
            {
                "tfu_enabled": True,
                "pure_input_metric": "tfu_data_task_diag",
                "pure_shallow_metric": "tfu_data_task_diag",
                "pure_deep_metric": "tfu_task_diag",
                "pure_output_metric": "tfu_task_diag",
            },
        ),
        (
            "D1-frontTask-backSob",
            {
                "tfu_enabled": True,
                "pure_input_metric": "tfu_data_task_diag",
                "pure_shallow_metric": "tfu_data_task_diag",
                "pure_deep_metric": "full_sobolev_gram",
                "pure_output_metric": "full_sobolev_gram",
            },
        ),
        (
            "D2-frontSob-backTask",
            {
                "tfu_enabled": True,
                "pure_input_metric": "basis_diag_gram",
                "pure_shallow_metric": "full_sobolev_gram",
                "pure_deep_metric": "tfu_task_diag",
                "pure_output_metric": "tfu_task_diag",
            },
        ),
        (
            "D3-taskSobTask",
            {
                "tfu_enabled": True,
                "pure_input_metric": "tfu_data_task_diag",
                "pure_shallow_metric": "full_sobolev_gram",
                "pure_deep_metric": "full_sobolev_gram",
                "pure_output_metric": "tfu_task_diag",
            },
        ),
        (
            "D7-inputOutputTask-middleSob",
            {
                "tfu_enabled": True,
                "pure_input_metric": "tfu_data_task_diag",
                "pure_shallow_metric": "full_sobolev_gram",
                "pure_deep_metric": "full_sobolev_gram",
                "pure_output_metric": "tfu_task_diag",
            },
        ),
        (
            "D8-inputDiag-blockTask-outputFisher",
            {
                "tfu_enabled": True,
                "pure_input_metric": "basis_diag_gram",
                "pure_shallow_metric": "tfu_task_diag",
                "pure_deep_metric": "tfu_task_diag",
                "pure_output_metric": "tfu_task_diag",
            },
        ),
    ]


def _flat_update(before: Dict[str, torch.Tensor], model: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    grads: List[torch.Tensor] = []
    updates: List[torch.Tensor] = []
    for name, p in coefficient_named_params(model):
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
                    tfu_enabled=bool(overrides.get("tfu_enabled", False)),
                    tfu_sob_lambda=float(overrides.get("tfu_sob_lambda", 0.03)),
                    pure_input_metric=overrides.get("pure_input_metric", "phase"),
                    pure_shallow_metric=overrides.get("pure_shallow_metric", "phase"),
                    pure_deep_metric=overrides.get("pure_deep_metric", "phase"),
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
                    "stage": "DG-KAN-v3.8",
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
                    "input_cos_raw_precond": _mean_state(state.pure_role_cos_raw_precond.get("input", []), float("nan")),
                    "shallow_cos_raw_precond_mean": _mean_state(state.pure_role_cos_raw_precond.get("shallow", []), float("nan")),
                    "deep_cos_raw_precond_mean": _mean_state(state.pure_role_cos_raw_precond.get("deep", []), float("nan")),
                    "block_cos_raw_precond_mean": _mean_state(state.pure_role_cos_raw_precond.get("block", []), float("nan")),
                    "output_cos_raw_precond": _mean_state(state.pure_role_cos_raw_precond.get("output", []), float("nan")),
                    "metric_condition_input": _mean_state(state.pure_role_metric_conditions.get("input", []), float("nan")),
                    "metric_condition_shallow": _mean_state(state.pure_role_metric_conditions.get("shallow", []), float("nan")),
                    "metric_condition_deep": _mean_state(state.pure_role_metric_conditions.get("deep", []), float("nan")),
                    "metric_condition_output": _mean_state(state.pure_role_metric_conditions.get("output", []), float("nan")),
                    "tfu_safeguard_fail_rate": state.tfu_safeguard_fail_count / max(1, state.tfu_safeguard_check_count),
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
    p.set_defaults(packages="V3_8_P0_SMOKE", datasets="MNIST,Fashion-MNIST,KMNIST", out_dir=Path("results/gafu_v3_8"))
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
        if key == "V3_8_P1_SHADOW":
            all_rows = run_shadow(args, key)
        else:
            specs, seeds, meta = package_specs(args, package)
            all_rows = run_specs(args, specs, seeds, meta)
        first = False
    return 0 if all_rows is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
