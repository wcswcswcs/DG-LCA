#!/usr/bin/env python3
"""DG-KAN v3.6 runner: Rational tangent U-FULL and PureKAN probes."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from dgkan_core import (
    _rational_tangent_jacobian,
    build_polynomial_sobolev_gram,
    ensure_dir,
    parse_int_list,
    parse_str_list,
    rational_denominator_values,
    rational_forward_from_params,
    save_json,
    write_csv,
)
from run_gafu_v3 import (
    Spec,
    add_args as add_v3_args,
    dataset_name,
    run_specs,
    spec,
    v33_dataset_specific_best,
    v35_convstem_adamw_spec,
    v35_convstem_ufull_spec,
    v35_rational_adamw_spec,
    v35_rational_ufull_spec,
    v35_ufull_f085_reference_spec,
    v35_ufull_f100_spec,
)


def v36_epochs(dataset: str) -> int:
    if dataset == "Fashion-MNIST":
        return 30
    return 20


def _small_common(**extra: Any) -> Dict[str, Any]:
    out = {
        "train_size": 512,
        "val_size": 128,
        "test_size": 128,
        "epochs": 1,
        "hidden_dim": 32,
        "depth": 2,
        "basis_count": 8,
        "rational_groups": 4,
        "batch_size": 256,
        "eval_batch_size": 256,
        "audit_batch_size": 64,
        "v3_gram_grid_size": 128,
    }
    out.update(extra)
    return out


def rational_metric_spec(
    dataset: str,
    label: str,
    metric: str,
    *,
    epochs: int | None = None,
    coeff_lr: float = 0.02,
    branch_functional: bool = False,
    backend: str = "triton",
    **extra: Any,
) -> Spec:
    overrides = {
        "epochs": epochs if epochs is not None else v36_epochs(dataset),
        "model_type": "rational_dgkan",
        "alpha_mode": "fixed1",
        "branch_final_scale": 1.0,
        "branch_boost": 1.0,
        "coeff_lr_boost": 1.0,
        "coeff_lr_final_mult": 1.0,
        "branch_max_active_frac": 0.0,
        "branch_schedule": "none",
        "v3_phase_mode": "hard",
        "v3_metric_active": metric,
        "v3_metric_transition": metric,
        "v3_metric_geometry": metric,
        "v3_alpha_fast": 0.0,
        "v3_beta_fast": 0.0,
        "v3_alpha_geo": 0.05 if "h1" in metric.lower() else 0.0,
        "v3_beta_geo": 0.0,
        "v3_gram_rho": 1.0 if "tangent" in metric.lower() else 1e-3,
        "gafu_v3_enabled": True,
        "unified_optimizer_mode": "hybrid",
        "coeff_lr": coeff_lr,
        "rest_lr": 1e-3,
        "rational_backend": backend,
        "rational_groups": 8,
        "rational_mode": "swish",
        "rational_branch_functional": branch_functional,
    }
    overrides.update(extra)
    return spec(dataset, label, "rational_ufull", **overrides)


def purekan_spec(dataset: str, label: str, method: str, *, epochs: int | None = None, **extra: Any) -> Spec:
    overrides = {
        "epochs": epochs if epochs is not None else v36_epochs(dataset),
        "model_type": "pure_kan",
        "alpha_mode": "fixed1",
        "alpha_init": 1.0,
        "hidden_dim": 64,
        "depth": 2,
        "basis_count": 16,
        "coeff_lr": 0.03,
        "branch_final_scale": 1.0,
        "v3_gram_grid_size": 256,
    }
    overrides.update(extra)
    return spec(dataset, label, method, **overrides)


def mlp_spec(dataset: str, label: str = "MLP-AdamW-alphaFixed1", *, epochs: int | None = None, **extra: Any) -> Spec:
    overrides = {
        "epochs": epochs if epochs is not None else v36_epochs(dataset),
        "model_type": "mlp",
        "alpha_mode": "fixed1",
    }
    overrides.update(extra)
    return spec(dataset, label, "mlp_adamw", **overrides)


def package_specs(args: argparse.Namespace, package: str) -> Tuple[List[Spec], List[int], Dict[str, Any]]:
    key = package.strip().upper().replace("-", "_")
    datasets = [dataset_name(item) for item in parse_str_list(args.datasets)]
    seeds = parse_int_list(args.seeds)
    specs: List[Spec] = []

    if key == "V3_6_P0_IMPL_SMOKE":
        seeds = [0]
        for dataset in [d for d in datasets if d in {"Fashion-MNIST", "KMNIST"}]:
            common = _small_common()
            specs.extend(
                [
                    v35_rational_adamw_spec(dataset, backend="triton", **common),
                    rational_metric_spec(
                        dataset,
                        "Rational-UFULL-poly-separate-smoke-alphaFixed1",
                        "rational_poly_separate",
                        coeff_lr=0.02,
                        **common,
                    ),
                    rational_metric_spec(
                        dataset,
                        "Rational-UFULL-tangent-joint-smoke-alphaFixed1",
                        "rational_tangent_joint_l2",
                        coeff_lr=0.02,
                        **common,
                    ),
                    rational_metric_spec(
                        dataset,
                        "Rational-UFULL-branch-smoke-alphaFixed1",
                        "rational_tangent_joint_l2",
                        coeff_lr=0.02,
                        branch_functional=True,
                        **common,
                    ),
                    purekan_spec(dataset, "PureKAN-AdamW-smoke-alphaFixed1", "purekan_adamw", **common),
                    purekan_spec(dataset, "PureKAN-UFULL-smoke-alphaFixed1", "purekan_ufull", **common),
                    v35_ufull_f085_reference_spec(dataset, label="Hybrid-UFULL-smoke-alphaFixed1", **common),
                ]
            )
        return specs, seeds, {"package": key}

    if key == "V3_6_P1_RATIONAL_METRIC_AUDIT":
        seeds = [0, 1, 2]
        for dataset in [d for d in datasets if d in {"Fashion-MNIST", "KMNIST"}]:
            epochs = v36_epochs(dataset)
            common = {"hidden_dim": 96, "depth": 4, "rational_groups": 8, "v3_gram_grid_size": 256}
            specs.extend(
                [
                    v35_rational_adamw_spec(dataset, epochs=epochs, backend="triton", **common),
                    rational_metric_spec(
                        dataset,
                        "Rational-UFULL-poly-separate-current-alphaFixed1",
                        "rational_poly_separate",
                        coeff_lr=0.02,
                        epochs=epochs,
                        **common,
                    ),
                    rational_metric_spec(dataset, "Rational-UFULL-poly-diag-alphaFixed1", "rational_poly_diag", epochs=epochs, **common),
                    rational_metric_spec(
                        dataset,
                        "Rational-UFULL-poly-rho1e-2-alphaFixed1",
                        "rational_poly_separate",
                        epochs=epochs,
                        v3_gram_rho=1e-2,
                        **common,
                    ),
                    rational_metric_spec(
                        dataset,
                        "Rational-UFULL-poly-rho1e-1-alphaFixed1",
                        "rational_poly_separate",
                        epochs=epochs,
                        v3_gram_rho=1e-1,
                        **common,
                    ),
                    rational_metric_spec(dataset, "Rational-UFULL-identity-alphaFixed1", "rational_identity", epochs=epochs, **common),
                    rational_metric_spec(
                        dataset,
                        "Rational-UFULL-legendre-separate-alphaFixed1",
                        "rational_legendre_separate",
                        epochs=epochs,
                        **common,
                    ),
                ]
            )
        return specs, seeds, {"package": key}

    if key == "V3_6_P3_RATIONAL_DGKAN_TANGENT3":
        seeds = [0, 1, 2]
        for dataset in [d for d in datasets if d in {"Fashion-MNIST", "KMNIST"}]:
            epochs = v36_epochs(dataset)
            common = {"hidden_dim": 96, "depth": 4, "rational_groups": 8, "v3_gram_grid_size": 256}
            specs.extend(
                [
                    v35_rational_adamw_spec(dataset, epochs=epochs, backend="triton", **common),
                    rational_metric_spec(dataset, "Rational-UFULL-poly-current-alphaFixed1", "rational_poly_separate", epochs=epochs, **common),
                    rational_metric_spec(dataset, "Rational-UFULL-tangent-act-L2-alphaFixed1", "rational_tangent_joint_l2", epochs=epochs, **common),
                    rational_metric_spec(dataset, "Rational-UFULL-tangent-act-H1-alphaFixed1", "rational_tangent_joint_h1", epochs=epochs, **common),
                    rational_metric_spec(
                        dataset,
                        "Rational-UFULL-tangent-branch-L2-alphaFixed1",
                        "rational_tangent_joint_l2",
                        branch_functional=True,
                        epochs=epochs,
                        **common,
                    ),
                    rational_metric_spec(
                        dataset,
                        "Rational-UFULL-tangent-branch-H1-alphaFixed1",
                        "rational_tangent_joint_h1",
                        branch_functional=True,
                        epochs=epochs,
                        **common,
                    ),
                    rational_metric_spec(dataset, "Rational-UFULL-tangent-diag-alphaFixed1", "rational_tangent_diag_l2", epochs=epochs, **common),
                    v35_ufull_f085_reference_spec(dataset, label="RBF-UFULL-f085-reference-alphaFixed1", epochs=epochs),
                ]
            )
        return specs, seeds, {"package": key}

    if key == "V3_6_P4_RATIONAL_DGKAN_CONFIRM":
        seeds = [0, 1, 2, 3, 4]
        for dataset in [d for d in datasets if d in {"Fashion-MNIST", "KMNIST"}]:
            epochs = v36_epochs(dataset)
            common = {"hidden_dim": 96, "depth": 4, "rational_groups": 8, "v3_gram_grid_size": 256}
            specs.extend(
                [
                    v35_rational_adamw_spec(dataset, epochs=epochs, backend="triton", **common),
                    rational_metric_spec(dataset, "Rational-UFULL-tangent-act-L2-alphaFixed1", "rational_tangent_joint_l2", epochs=epochs, **common),
                    rational_metric_spec(
                        dataset,
                        "Rational-UFULL-tangent-branch-L2-alphaFixed1",
                        "rational_tangent_joint_l2",
                        branch_functional=True,
                        epochs=epochs,
                        **common,
                    ),
                ]
            )
        return specs, seeds, {"package": key}

    if key == "V3_6_P5_UFULL_F100_FAILURE_AUDIT":
        seeds = [0, 1, 2, 3, 4]
        for dataset in [d for d in datasets if d in {"Fashion-MNIST", "KMNIST"}]:
            epochs = v36_epochs(dataset)
            specs.extend(
                [
                    spec(dataset, "AdamW-alphaFixed1-alphaFixed1", "adamw", epochs=epochs, alpha_mode="fixed1"),
                    v35_ufull_f085_reference_spec(dataset, epochs=epochs),
                    v35_ufull_f100_spec(dataset, epochs=epochs),
                    v33_dataset_specific_best(dataset, epochs=epochs),
                ]
            )
        return specs, seeds, {"package": key}

    if key == "V3_6_P6_CIFAR_BRANCH_AUDIT":
        seeds = [0, 1, 2]
        common = {
            "train_size": 10000,
            "val_size": 2000,
            "test_size": 2000,
            "epochs": 20,
            "hidden_dim": 96,
            "depth": 4,
            "basis_count": 24,
            "audit_batch_size": 64,
        }
        dataset = "CIFAR10"
        specs.extend(
            [
                v35_convstem_adamw_spec(dataset, **common),
                v35_convstem_ufull_spec(dataset, branch_final_scale=0.85, **common),
                v35_convstem_ufull_spec(dataset, branch_final_scale=1.0, **common),
                spec(
                    dataset,
                    "ConvStem-DGKAN-UFULL-diag-warmup-probe-alphaFixed1",
                    "convstem_dgkan_ufull",
                    model_type="convstem_dgkan",
                    alpha_mode="fixed1",
                    branch_final_scale=0.85,
                    v3_metric_active="basis_diag_gram",
                    v3_metric_transition="full_sobolev_gram",
                    v3_metric_geometry="full_sobolev_gram",
                    branch_boost=1.0,
                    **common,
                ),
                spec(
                    dataset,
                    "ConvStem-DGKAN-UFULL-branchboost-probe-alphaFixed1",
                    "convstem_dgkan_ufull",
                    model_type="convstem_dgkan",
                    alpha_mode="fixed1",
                    branch_final_scale=0.85,
                    branch_boost=1.25,
                    coeff_lr_boost=1.1,
                    **common,
                ),
            ]
        )
        return specs, seeds, {"package": key}

    if key == "V3_6_P7_PUREKAN_SMOKE":
        seeds = [0]
        for dataset in [d for d in datasets if d in {"Fashion-MNIST", "KMNIST"}]:
            common = _small_common()
            specs.extend(
                [
                    purekan_spec(dataset, "PureKAN-AdamW-smoke-alphaFixed1", "purekan_adamw", **common),
                    purekan_spec(dataset, "PureKAN-UFULL-smoke-alphaFixed1", "purekan_ufull", **common),
                    v35_ufull_f085_reference_spec(dataset, label="Hybrid-UFULL-smoke-alphaFixed1", **common),
                    mlp_spec(dataset, "MLP-AdamW-smoke-alphaFixed1", **common),
                ]
            )
        return specs, seeds, {"package": key}

    if key == "V3_6_P8_PUREKAN_3SEED":
        seeds = [0, 1, 2]
        for dataset in [d for d in datasets if d in {"MNIST", "Fashion-MNIST", "KMNIST"}]:
            epochs = v36_epochs(dataset)
            common = {"hidden_dim": 64, "depth": 2, "basis_count": 16, "train_size": 6000, "val_size": 1000, "test_size": 1000}
            specs.extend(
                [
                    mlp_spec(dataset, "MLP-AdamW-alphaFixed1", epochs=epochs, **common),
                    v35_ufull_f085_reference_spec(dataset, label="Hybrid-DGKAN-UFULL-f085-alphaFixed1", epochs=epochs, **common),
                    purekan_spec(dataset, "PureKAN-AdamW-alphaFixed1", "purekan_adamw", epochs=epochs, **common),
                    purekan_spec(dataset, "PureKAN-UFULL-alphaFixed1", "purekan_ufull", epochs=epochs, **common),
                    purekan_spec(
                        dataset,
                        "PureKAN-UFULL-diagwarmup-alphaFixed1",
                        "purekan_ufull",
                        epochs=epochs,
                        v3_metric_active="basis_diag_gram",
                        v3_metric_transition="full_sobolev_gram",
                        v3_metric_geometry="full_sobolev_gram",
                        **common,
                    ),
                ]
            )
        return specs, seeds, {"package": key}

    if key == "V3_6_P9_PUREKAN_5SEED":
        seeds = [0, 1, 2, 3, 4]
        for dataset in [d for d in datasets if d in {"MNIST", "Fashion-MNIST", "KMNIST"}]:
            epochs = v36_epochs(dataset)
            common = {"hidden_dim": 64, "depth": 2, "basis_count": 16, "train_size": 6000, "val_size": 1000, "test_size": 1000}
            specs.extend(
                [
                    mlp_spec(dataset, "MLP-AdamW-alphaFixed1", epochs=epochs, **common),
                    purekan_spec(dataset, "PureKAN-UFULL-alphaFixed1", "purekan_ufull", epochs=epochs, **common),
                    purekan_spec(dataset, "PureKAN-UFULL-diagwarmup-alphaFixed1", "purekan_ufull", epochs=epochs, v3_metric_active="basis_diag_gram", **common),
                ]
            )
        return specs, seeds, {"package": key}

    raise ValueError(f"unknown v3.6 package: {package}")


def _toy_target(name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "silu":
        return x * torch.sigmoid(x)
    if name == "tanh_sin":
        return torch.tanh(x) + 0.2 * torch.sin(3.0 * x)
    if name == "smoothstep":
        u = ((x + 2.5) / 5.0).clamp(0.0, 1.0)
        return 2.0 * (u * u * (3.0 - 2.0 * u)) - 1.0
    raise ValueError(name)


def _toy_joint_gram(num: torch.Tensor, den: torch.Tensor, *, h1: bool, rho: float, grid_size: int = 256) -> Dict[str, torch.Tensor]:
    grid = torch.linspace(-2.5, 2.5, grid_size, device=num.device, dtype=num.dtype)
    dt = 5.0 / max(1, grid_size - 1)
    j = _rational_tangent_jacobian(grid, num, den).float()
    gram = dt * (j.T @ j)
    if h1:
        eps = 1e-2
        jp = _rational_tangent_jacobian((grid + eps).clamp(-2.5, 2.5), num, den).float()
        jm = _rational_tangent_jacobian((grid - eps).clamp(-2.5, 2.5), num, den).float()
        jd = (jp - jm) / (2.0 * eps)
        gram = gram + 0.05 * dt * (jd.T @ jd)
    damped = gram + rho * torch.eye(gram.shape[0], device=gram.device)
    eig = torch.linalg.eigvalsh(damped).clamp_min(1e-12)
    return {"A": damped, "chol": torch.linalg.cholesky(damped), "condition": eig[-1] / eig[0]}


def run_toy(args: argparse.Namespace, package: str) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else []
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    methods = [
        "adam",
        "poly-separate-current",
        "poly-diag",
        "tangent-joint-L2",
        "tangent-joint-H1",
        "tangent-diag",
    ]
    seeds = parse_int_list(args.seeds) or [0, 1, 2]
    targets = ["silu", "tanh_sin", "smoothstep"]
    total = len(seeds) * len(targets) * len(methods)
    done = 0
    for seed in seeds:
        torch.manual_seed(seed)
        for target in targets:
            x = torch.linspace(-2.5, 2.5, 256, device=device)
            y = _toy_target(target, x)
            for method in methods:
                done += 1
                torch.manual_seed(seed)
                num = torch.nn.Parameter(torch.tensor([0.0, 0.9, 0.0, 0.05, 0.0, 0.0], device=device))
                den = torch.nn.Parameter(torch.full((4,), 0.05, device=device))
                opt = torch.optim.Adam([num, den], lr=0.02) if method == "adam" else None
                losses: List[float] = []
                bad = 0
                cond = float("nan")
                start = time.perf_counter()
                for step in range(300):
                    pred = rational_forward_from_params(x, num, den)
                    loss = F.mse_loss(pred, y)
                    prev = float(loss.detach().cpu())
                    if opt is not None:
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                    else:
                        for p in [num, den]:
                            if p.grad is not None:
                                p.grad = None
                        loss.backward()
                        g = torch.cat([num.grad.detach().flatten(), den.grad.detach().flatten()]).float()
                        if method.startswith("poly"):
                            g_num = g[: num.numel()]
                            g_den = g[num.numel() :]
                            rho = 1e-2
                            gram_num = build_polynomial_sobolev_gram(num.numel(), 0.05, 0.0, rho, device=device, dtype=num.dtype)
                            gram_den = build_polynomial_sobolev_gram(
                                den.numel(), 0.05, 0.0, rho, kind="abs_power", device=device, dtype=den.dtype
                            )
                            if "diag" in method:
                                dnum = -g_num / torch.diag(gram_num["A"]).to(device).clamp_min(rho)
                                dden = -g_den / torch.diag(gram_den["A"]).to(device).clamp_min(rho)
                            else:
                                dnum = -torch.cholesky_solve(g_num.view(-1, 1), gram_num["chol"].to(device)).view(-1)
                                dden = -torch.cholesky_solve(g_den.view(-1, 1), gram_den["chol"].to(device)).view(-1)
                            direction = torch.cat([dnum, dden])
                            cond = max(float(gram_num["condition"]), float(gram_den["condition"]))
                        else:
                            gram = _toy_joint_gram(num.detach(), den.detach(), h1="H1" in method, rho=1e-2)
                            if "diag" in method:
                                direction = -g / torch.diag(gram["A"]).to(g.device).clamp_min(1e-2)
                            else:
                                direction = -torch.cholesky_solve(g.view(-1, 1), gram["chol"].to(g.device)).view(-1)
                            cond = float(gram["condition"].detach().cpu())
                        update = 0.03 * direction.to(device)
                        limit = 0.20 * torch.cat([num.detach().flatten(), den.detach().flatten()]).norm().clamp_min(1e-6)
                        norm = update.norm()
                        if norm > limit:
                            update = update * (limit / norm.clamp_min(1e-12))
                        with torch.no_grad():
                            num.add_(update[: num.numel()].reshape_as(num))
                            den.add_(update[num.numel() :].reshape_as(den))
                    new_loss = float(F.mse_loss(rational_forward_from_params(x, num, den), y).detach().cpu())
                    if new_loss > prev * 1.20 or not math.isfinite(new_loss):
                        bad += 1
                    losses.append(new_loss)
                den_values = rational_denominator_values(x, den.detach().view(1, -1)).flatten().float()
                row = {
                    "stage": "GA-FU-v3.6",
                    "package": package,
                    "target": target,
                    "dataset": "ToyRational",
                    "method": method,
                    "seed": seed,
                    "steps": 300,
                    "final_mse": losses[-1],
                    "loss_auc": float(sum(losses) / max(1, len(losses))),
                    "bad_step_rate": bad / 300,
                    "den_actual_min_grid": float(den_values.min().detach().cpu()),
                    "den_actual_p01_grid": float(torch.quantile(den_values, 0.01).detach().cpu()),
                    "metric_condition": cond,
                    "wall_sec": time.perf_counter() - start,
                    "error": "",
                }
                rows.append(row)
                print(
                    f"[{done}/{total}] {package} toy {target} {method} seed={seed} "
                    f"mse={row['final_mse']:.4g} auc={row['loss_auc']:.4g} bad={row['bad_step_rate']:.3f} cond={cond:.1f}"
                )
                write_csv(out_dir / "toy_runs.csv", rows)
    write_csv(out_dir / "runs.csv", rows)
    save_json(out_dir / "aggregate_summary.json", {"package": package, "new_runs": len(rows), "toy_rows": len(rows)})
    return rows


def add_args() -> argparse.ArgumentParser:
    p = add_v3_args()
    p.description = __doc__
    p.set_defaults(packages="V3_6_P0_IMPL_SMOKE", out_dir=Path("results/gafu_v3_6"))
    return p


def main() -> int:
    args = add_args().parse_args()
    packages = parse_str_list(args.packages)
    original_fresh = args.fresh
    all_rows: List[Dict[str, Any]] = []
    first = True
    for package in packages:
        key = package.strip().upper().replace("-", "_")
        args.fresh = original_fresh and first
        if key == "V3_6_P2_RATIONAL_TANGENT_TOY":
            all_rows = run_toy(args, key)
        else:
            specs, seeds, meta = package_specs(args, package)
            all_rows = run_specs(args, specs, seeds, meta)
        first = False
    return 0 if all_rows is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
