#!/usr/bin/env python3
"""GA-FU-v3 phase-aligned optimizer runner.

This runner follows docs/DG-KAN_Optimizer_v3_完整计划.md.  It keeps v3
separate from the older consolidation runner so the new phase/metric
experiments do not blur the v2 weak-pass records.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

from dgkan_core import (
    RBFDense,
    RuntimeState,
    TrainConfig,
    _gram_diag,
    _solve_with_gram,
    _transition_soft,
    _v3_precondition_direction,
    _v3_transition_length,
    add_baseline_comparisons,
    add_common_train_args,
    add_failure_types,
    build_rbf_sobolev_gram,
    config_from_args,
    ensure_dir,
    parse_int_list,
    parse_str_list,
    read_csv,
    save_json,
    summarize_runs,
    train_one,
    write_csv,
)


Spec = Dict[str, Any]


def dataset_name(name: str) -> str:
    key = name.strip().lower().replace("_", "-")
    if key in {"fashion", "fashion-mnist", "fmnist"}:
        return "Fashion-MNIST"
    if key in {"kmnist", "kuzushiji-mnist"}:
        return "KMNIST"
    if key == "mnist":
        return "MNIST"
    if key in {"cifar", "cifar10", "cifar-10"}:
        return "CIFAR10"
    raise ValueError(f"unknown dataset {name!r}")


def spec(dataset: str, label: str, base_method: str, **overrides: Any) -> Spec:
    return {
        "dataset": dataset,
        "label": label,
        "base_method": base_method,
        "overrides": overrides,
    }


def alpha_suffix(alpha_mode: str) -> str:
    return {
        "learnable": "alphaLearn",
        "fixed1": "alphaFixed1",
        "fixed_init": "alphaFixedInit",
    }.get(alpha_mode, alpha_mode)


def alpha_spec(dataset: str, label: str, base_method: str, alpha_mode: str, **overrides: Any) -> Spec:
    tagged = f"{label}-{alpha_suffix(alpha_mode)}"
    return spec(dataset, tagged, base_method, alpha_mode=alpha_mode, **overrides)


def tuned_v2_overrides(dataset: str) -> Dict[str, Any]:
    if dataset == "Fashion-MNIST":
        return {
            "coeff_lr": 0.05,
            "rest_lr": 0.001,
            "branch_boost": 1.2,
            "coeff_lr_boost": 1.0,
            "branch_max_active_frac": 0.35,
            "branch_final_scale": 0.8,
            "geometry_min_epochs": 0,
            "rest_lr_schedule": "cosine",
            "rest_lr_final_mult": 0.3,
            "coeff_lr_schedule": "cosine",
            "coeff_lr_decay_final_mult": 0.5,
        }
    if dataset == "KMNIST":
        return {
            "coeff_lr": 0.14,
            "rest_lr": 0.003,
            "branch_boost": 1.2,
            "coeff_lr_boost": 1.0,
            "branch_max_active_frac": 0.35,
            "branch_final_scale": 0.8,
            "geometry_min_epochs": 0,
        }
    return {
        "coeff_lr": 0.10,
        "rest_lr": 0.003,
        "branch_boost": 1.2,
        "coeff_lr_boost": 1.0,
        "branch_max_active_frac": 0.35,
        "branch_final_scale": 0.8,
        "geometry_min_epochs": 0,
    }


def v3_base_overrides(dataset: str, *, phase_mode: str = "smooth") -> Dict[str, Any]:
    out = {
        **tuned_v2_overrides(dataset),
        "gafu_v3_enabled": True,
        "metric_mode": "grid",
        "branch_schedule": "geometry_aware",
        "warmup_frac": 0.0,
        "v3_phase_mode": phase_mode,
        "v3_alpha_fast": 0.0,
        "v3_beta_fast": 0.0,
        "v3_alpha_geo": 0.15,
        "v3_beta_geo": 0.02,
        "v3_gram_rho": 1e-3,
        "v3_transition_frac": 0.10,
    }
    return out


def v3_legacy_phase_overrides(dataset: str, *, phase_mode: str) -> Dict[str, Any]:
    return {
        **v3_base_overrides(dataset, phase_mode=phase_mode),
        "v3_metric_active": "grid_index_diag_legacy",
        "v3_metric_transition": "grid_index_diag_legacy",
        "v3_metric_geometry": "grid_index_diag_legacy",
    }


def v3_metric_overrides(
    dataset: str,
    *,
    label: str,
    active: str,
    transition: str,
    geometry: str,
    phase_mode: str = "smooth",
    alpha_geo: float = 0.15,
    beta_geo: float = 0.02,
) -> Spec:
    return spec(
        dataset,
        label,
        "geometry_current",
        **{
            **v3_base_overrides(dataset, phase_mode=phase_mode),
            "v3_metric_active": active,
            "v3_metric_transition": transition,
            "v3_metric_geometry": geometry,
            "v3_alpha_geo": alpha_geo,
            "v3_beta_geo": beta_geo,
        },
    )


def full_from_start_spec(dataset: str, label: str, **extra: Any) -> Spec:
    overrides = {
        **v3_base_overrides(dataset, phase_mode="hard"),
        "v3_metric_active": "full_sobolev_gram",
        "v3_metric_transition": "full_sobolev_gram",
        "v3_metric_geometry": "full_sobolev_gram",
        "v3_alpha_geo": 0.15,
        "v3_beta_geo": 0.02,
        "v3_gram_rho": 1e-3,
    }
    overrides.update(extra)
    return spec(dataset, label, "geometry_current", **overrides)


def full_from_start_alpha_spec(dataset: str, label: str, alpha_mode: str, **extra: Any) -> Spec:
    item = full_from_start_spec(dataset, f"{label}-{alpha_suffix(alpha_mode)}", **extra)
    item["overrides"]["alpha_mode"] = alpha_mode
    return item


def hard_diag_to_full_spec(dataset: str, label: str, **extra: Any) -> Spec:
    overrides = {
        **v3_base_overrides(dataset, phase_mode="hard"),
        "v3_metric_active": "basis_diag_gram",
        "v3_metric_transition": "full_sobolev_gram",
        "v3_metric_geometry": "full_sobolev_gram",
        "v3_alpha_fast": 0.0,
        "v3_beta_fast": 0.0,
        "v3_alpha_geo": 0.15,
        "v3_beta_geo": 0.02,
        "v3_gram_rho": 1e-3,
    }
    overrides.update(extra)
    return spec(dataset, label, "geometry_current", **overrides)


def hard_diag_to_full_alpha_spec(dataset: str, label: str, alpha_mode: str, **extra: Any) -> Spec:
    item = hard_diag_to_full_spec(dataset, f"{label}-{alpha_suffix(alpha_mode)}", **extra)
    item["overrides"]["alpha_mode"] = alpha_mode
    return item


def v33_dataset_base(dataset: str, *, epochs: int | None = None) -> Dict[str, Any]:
    out = {
        **tuned_v2_overrides(dataset),
        "alpha_mode": "fixed1",
        "branch_boost": 1.2,
        "coeff_lr_boost": 1.0,
        "branch_max_active_frac": 0.35,
        "branch_final_scale": 0.8,
        "geometry_min_epochs": 0,
    }
    if dataset == "Fashion-MNIST":
        out.update(
            {
                "coeff_lr": 0.05,
                "rest_lr": 0.001,
                "rest_lr_schedule": "cosine",
                "rest_lr_final_mult": 0.3,
                "coeff_lr_schedule": "cosine",
                "coeff_lr_decay_final_mult": 0.5,
            }
        )
    elif dataset == "KMNIST":
        out.update(
            {
                "coeff_lr": 0.14,
                "rest_lr": 0.003,
                "rest_lr_schedule": "none",
                "rest_lr_final_mult": 1.0,
                "coeff_lr_schedule": "none",
                "coeff_lr_decay_final_mult": 1.0,
            }
        )
    if epochs is not None:
        out["epochs"] = epochs
    return out


def v33_epochs(dataset: str, *, smoke: bool = False) -> int:
    if smoke:
        return 10
    return 30 if dataset == "Fashion-MNIST" else 20


def ufull_alpha_spec(dataset: str, label: str, **extra: Any) -> Spec:
    epochs = int(extra.pop("epochs", v33_epochs(dataset)))
    overrides = {
        **v33_dataset_base(dataset, epochs=epochs),
        "gafu_v3_enabled": True,
        "metric_mode": "grid",
        "branch_schedule": "geometry_aware",
        "warmup_frac": 0.0,
        "v3_phase_mode": "hard",
        "v3_metric_active": "full_sobolev_gram",
        "v3_metric_transition": "full_sobolev_gram",
        "v3_metric_geometry": "full_sobolev_gram",
        "v3_alpha_fast": 0.15,
        "v3_beta_fast": 0.02,
        "v3_alpha_geo": 0.15,
        "v3_beta_geo": 0.02,
        "v3_gram_rho": 1e-3,
        "unified_optimizer_mode": "hybrid",
    }
    overrides.update(extra)
    return spec(dataset, label, "geometry_current", **overrides)


def v35_ufull_f100_spec(dataset: str, label: str = "U-FULL-f100-alphaFixed1", *, epochs: int | None = None, **extra: Any) -> Spec:
    e = epochs if epochs is not None else v33_epochs(dataset)
    overrides = {
        "branch_final_scale": 1.0,
        "branch_boost": 1.0,
        "coeff_lr_boost": 1.0,
        "coeff_lr_final_mult": 1.0,
        "branch_schedule": "none",
        "branch_max_active_frac": 0.0,
        "geometry_min_epochs": 0,
        "v3_phase_mode": "hard",
        "unified_optimizer_mode": "hybrid",
    }
    overrides.update(extra)
    return ufull_alpha_spec(dataset, label, epochs=e, **overrides)


def v35_ufull_f085_reference_spec(
    dataset: str,
    label: str = "U-FULL-f085-reference-alphaFixed1",
    *,
    epochs: int | None = None,
    **extra: Any,
) -> Spec:
    e = epochs if epochs is not None else v33_epochs(dataset)
    overrides = {"branch_final_scale": 0.85}
    overrides.update(extra)
    return ufull_alpha_spec(dataset, label, epochs=e, **overrides)


def v35_rational_adamw_spec(dataset: str, *, epochs: int | None = None, backend: str = "triton", **extra: Any) -> Spec:
    e = epochs if epochs is not None else v33_epochs(dataset)
    overrides = {
        "epochs": e,
        "model_type": "rational_dgkan",
        "alpha_mode": "fixed1",
        "branch_final_scale": 1.0,
        "rational_backend": backend,
        "rational_groups": 8,
        "rational_mode": "swish",
    }
    overrides.update(extra)
    return spec(dataset, f"Rational-DGKAN-AdamW-{backend}-alphaFixed1", "rational_adamw", **overrides)


def v35_rational_ufull_spec(dataset: str, *, epochs: int | None = None, backend: str = "triton", coeff_lr: float = 0.02, **extra: Any) -> Spec:
    e = epochs if epochs is not None else v33_epochs(dataset)
    overrides = {
        "epochs": e,
        "model_type": "rational_dgkan",
        "alpha_mode": "fixed1",
        "branch_final_scale": 1.0,
        "branch_boost": 1.0,
        "coeff_lr_boost": 1.0,
        "coeff_lr_final_mult": 1.0,
        "branch_max_active_frac": 0.0,
        "branch_schedule": "none",
        "v3_phase_mode": "hard",
        "v3_metric_active": "full_sobolev_gram",
        "v3_metric_transition": "full_sobolev_gram",
        "v3_metric_geometry": "full_sobolev_gram",
        "gafu_v3_enabled": True,
        "unified_optimizer_mode": "hybrid",
        "coeff_lr": coeff_lr,
        "rest_lr": 1e-3,
        "rational_backend": backend,
        "rational_groups": 8,
        "rational_mode": "swish",
        "notes": "rational numerator/denominator coefficients use full-Sobolev functional update",
    }
    overrides.update(extra)
    lr_tag = str(coeff_lr).replace(".", "p")
    return spec(dataset, f"Rational-DGKAN-U-FULL-{backend}-c{lr_tag}-alphaFixed1", "rational_ufull", **overrides)


def v35_convstem_adamw_spec(dataset: str = "CIFAR10", *, epochs: int = 20, **extra: Any) -> Spec:
    overrides = {"epochs": epochs, "model_type": "convstem_dgkan", "alpha_mode": "fixed1", "branch_final_scale": 1.0}
    overrides.update(extra)
    return spec(dataset, "ConvStem-DGKAN-AdamW-alphaFixed1", "convstem_dgkan_adamw", **overrides)


def v35_convstem_ufull_spec(dataset: str = "CIFAR10", *, epochs: int = 20, branch_final_scale: float = 0.85, **extra: Any) -> Spec:
    overrides = {
        "epochs": epochs,
        "model_type": "convstem_dgkan",
        "alpha_mode": "fixed1",
        "branch_final_scale": branch_final_scale,
    }
    overrides.update(extra)
    return spec(dataset, f"ConvStem-DGKAN-U-FULL-f{int(branch_final_scale * 100):03d}-alphaFixed1", "convstem_dgkan_ufull", **overrides)


def v33_ufull_matrix(dataset: str, *, epochs: int | None = None, prefix: str = "U-FULL") -> List[Spec]:
    base = v33_dataset_base(dataset)
    e = epochs if epochs is not None else v33_epochs(dataset)
    return [
        ufull_alpha_spec(dataset, f"{prefix}-base-alphaFixed1", epochs=e),
        ufull_alpha_spec(dataset, f"{prefix}-f085-alphaFixed1", branch_final_scale=0.85, epochs=e),
        ufull_alpha_spec(dataset, f"{prefix}-f090-alphaFixed1", branch_final_scale=0.90, epochs=e),
        ufull_alpha_spec(dataset, f"{prefix}-cplus-alphaFixed1", coeff_lr=float(base["coeff_lr"]) * 1.10, epochs=e),
        ufull_alpha_spec(
            dataset,
            f"{prefix}-f085-cplus-alphaFixed1",
            branch_final_scale=0.85,
            coeff_lr=float(base["coeff_lr"]) * 1.10,
            epochs=e,
        ),
        ufull_alpha_spec(
            dataset,
            f"{prefix}-softgeo-alphaFixed1",
            v3_alpha_fast=0.10,
            v3_beta_fast=0.01,
            v3_alpha_geo=0.10,
            v3_beta_geo=0.01,
            epochs=e,
        ),
    ]


def v33_dataset_specific_best(dataset: str, *, epochs: int | None = None) -> Spec:
    if dataset == "Fashion-MNIST":
        e = epochs if epochs is not None else 30
        return hard_diag_to_full_alpha_spec(
            dataset,
            "F-V3-HARD-base30-v32best",
            "fixed1",
            branch_final_scale=0.8,
            coeff_lr=0.05,
            rest_lr=0.001,
            rest_lr_schedule="cosine",
            rest_lr_final_mult=0.3,
            coeff_lr_schedule="cosine",
            coeff_lr_decay_final_mult=0.5,
            epochs=e,
        )
    e = epochs if epochs is not None else 20
    return full_from_start_alpha_spec(dataset, "K-V3-FULL-base-v32best", "fixed1", epochs=e)


def v33_gafu_v2_spec(dataset: str, *, epochs: int | None = None) -> Spec:
    e = epochs if epochs is not None else v33_epochs(dataset)
    overrides = {**v33_dataset_base(dataset, epochs=e), "unified_optimizer_mode": "hybrid"}
    return spec(dataset, "GA-FU-v2-alphaFixed1", "geometry_current", **overrides)


def uo_spec(dataset: str, label: str, mode: str, *, epochs: int | None = None, **extra: Any) -> Spec:
    normalize = mode in {"norm_pgadam", "uo_norm_pgadam"}
    overrides = {
        "unified_optimizer_mode": mode,
        "kan_grad_transform": (
            "sobolev_post_adam"
            if "postadam" in mode
            else "sobolev_pre_adam_norm"
            if normalize
            else "sobolev_pre_adam"
            if "pgadam" in mode
            else "none"
        ),
        "kan_precondition_before_adam": int("pgadam" in mode),
        "kan_precondition_after_adam": int("postadam" in mode),
        "kan_precond_grad_normalize": int(normalize),
        "uo_record_moment_audit": True,
    }
    overrides.update(extra)
    return ufull_alpha_spec(dataset, label, epochs=epochs if epochs is not None else v33_epochs(dataset), **overrides)


def afu_hybrid_spec(dataset: str, label: str = "AFU-0-Hybrid-alphaFixed1", *, epochs: int | None = None, **extra: Any) -> Spec:
    overrides = {"branch_final_scale": 0.85}
    overrides.update(extra)
    return ufull_alpha_spec(dataset, label, epochs=epochs if epochs is not None else v33_epochs(dataset), **overrides)


def afu_spec(dataset: str, label: str, *, epochs: int | None = None, **extra: Any) -> Spec:
    overrides = {
        "branch_final_scale": 0.85,
        "unified_optimizer_mode": "afu",
        "kan_grad_transform": "all_functional_group_metrics",
        "afu_cov_ema_beta": 0.95,
        "afu_head_rho": 1e-2,
        "afu_stem_rho": 1e-2,
        "afu_ln_rho": 1e-2,
        "afu_bias_rho": 1e-2,
        "afu_max_update_ratio": 0.05,
    }
    overrides.update(extra)
    return ufull_alpha_spec(dataset, label, epochs=epochs if epochs is not None else v33_epochs(dataset), **overrides)


def v34_afu_core_specs(dataset: str, *, epochs: int | None = None) -> List[Spec]:
    e = epochs if epochs is not None else v33_epochs(dataset)
    return [
        afu_hybrid_spec(dataset, epochs=e),
        afu_spec(dataset, "AFU-1-BranchLocal-alphaFixed1", afu_kan_bias=True, epochs=e),
        afu_spec(dataset, "AFU-2-HeadCov-alphaFixed1", afu_kan_bias=True, afu_head=True, epochs=e),
        afu_spec(
            dataset,
            "AFU-3-HeadCov-LNHead-alphaFixed1",
            afu_kan_bias=True,
            afu_head=True,
            afu_ln_scope="head",
            epochs=e,
        ),
        afu_spec(
            dataset,
            "AFU-4-StemDiag-alphaFixed1",
            afu_kan_bias=True,
            afu_head=True,
            afu_stem=True,
            epochs=e,
        ),
        afu_spec(
            dataset,
            "AFU-5-FullDiagLite-alphaFixed1",
            afu_kan_bias=True,
            afu_head=True,
            afu_stem=True,
            afu_ln_scope="all",
            epochs=e,
        ),
        afu_spec(
            dataset,
            "AFU-6-FullKFACLite-alphaFixed1",
            afu_kan_bias=True,
            afu_head=True,
            afu_stem=True,
            afu_ln_scope="all",
            afu_cov_ema_beta=0.98,
            epochs=e,
        ),
    ]


def v34_single_component_specs(dataset: str, *, epochs: int | None = None) -> List[Spec]:
    e = epochs if epochs is not None else v33_epochs(dataset)
    return [
        afu_hybrid_spec(dataset, epochs=e),
        afu_spec(dataset, "AFU-1-BranchLocal-alphaFixed1", afu_kan_bias=True, epochs=e),
        afu_spec(dataset, "AFU-2-HeadCov-alphaFixed1", afu_kan_bias=True, afu_head=True, epochs=e),
        afu_spec(dataset, "AFU-HeadOnly-alphaFixed1", afu_head=True, epochs=e),
        afu_spec(dataset, "AFU-StemOnly-alphaFixed1", afu_stem=True, epochs=e),
        afu_spec(dataset, "AFU-LNOnly-alphaFixed1", afu_ln_scope="all", epochs=e),
    ]


def package_specs(args: argparse.Namespace, package: str) -> Tuple[List[Spec], List[int], Dict[str, Any]]:
    datasets = [dataset_name(x) for x in parse_str_list(args.datasets)]
    pkg = package.strip().upper().replace("-", "_")
    seeds = parse_int_list(args.seeds)
    specs: List[Spec] = []
    if pkg == "V3_E0":
        seeds = [0] if args.seeds == add_args().get_default("seeds") else seeds
        for dataset in datasets:
            specs.extend(
                [
                    spec(dataset, "AdamW", "AdamW"),
                    spec(dataset, "GA-FU-v2-current", "geometry_current", **tuned_v2_overrides(dataset)),
                    v3_metric_overrides(
                        dataset,
                        label="GA-FU-v3-smoke",
                        active="basis_diag_gram",
                        transition="diag_to_full_sobolev",
                        geometry="full_sobolev_gram",
                    ),
                ]
            )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_R1_SMOKE_FIXED":
        seeds = [0]
        for dataset in datasets:
            specs.extend(
                [
                    spec(dataset, "AdamW", "AdamW"),
                    spec(dataset, "GA-FU-v2-current", "geometry_current", **tuned_v2_overrides(dataset)),
                    v3_metric_overrides(
                        dataset,
                        label="GA-FU-v3-smoke-fixed",
                        active="basis_diag_gram",
                        transition="diag_to_full_sobolev",
                        geometry="full_sobolev_gram",
                    ),
                ]
            )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_2_P0_SMOKE":
        seeds = [0]
        for dataset in datasets:
            for alpha_mode in ["learnable", "fixed1"]:
                specs.extend(
                    [
                        alpha_spec(dataset, "AdamW", "AdamW", alpha_mode),
                        alpha_spec(
                            dataset,
                            "GA-FU-v2-current",
                            "geometry_current",
                            alpha_mode,
                            **tuned_v2_overrides(dataset),
                        ),
                    ]
                )
                item = v3_metric_overrides(
                    dataset,
                    label=f"GA-FU-v3-smoke-fixed-transition-{alpha_suffix(alpha_mode)}",
                    active="basis_diag_gram",
                    transition="diag_to_full_sobolev",
                    geometry="full_sobolev_gram",
                )
                item["overrides"]["alpha_mode"] = alpha_mode
                specs.append(item)
        return specs, seeds, {"package": pkg}
    if pkg == "V3_2_P1_SANITY":
        seeds = [0]
        for dataset in datasets:
            epochs = 30 if dataset == "Fashion-MNIST" else 20
            for alpha_mode in ["learnable", "fixed1"]:
                specs.append(alpha_spec(dataset, "AdamW", "AdamW", alpha_mode, epochs=epochs))
                if dataset == "Fashion-MNIST":
                    v2 = {
                        **tuned_v2_overrides(dataset),
                        "branch_final_scale": 0.8,
                        "coeff_lr": 0.05,
                        "coeff_lr_decay_final_mult": 0.5,
                        "epochs": epochs,
                    }
                    specs.append(alpha_spec(dataset, "GA-FU-v2-bothcos", "geometry_current", alpha_mode, **v2))
                    specs.append(
                        hard_diag_to_full_alpha_spec(
                            dataset,
                            "F-V3-HARD-base30",
                            alpha_mode,
                            branch_final_scale=0.8,
                            coeff_lr=0.05,
                            rest_lr=0.001,
                            rest_lr_schedule="cosine",
                            rest_lr_final_mult=0.3,
                            coeff_lr_schedule="cosine",
                            coeff_lr_decay_final_mult=0.5,
                            epochs=epochs,
                        )
                    )
                    specs.append(
                        hard_diag_to_full_alpha_spec(
                            dataset,
                            "F-V3-HARD-c006-softcoeff",
                            alpha_mode,
                            branch_final_scale=0.85,
                            coeff_lr=0.06,
                            rest_lr=0.001,
                            rest_lr_schedule="cosine",
                            rest_lr_final_mult=0.3,
                            coeff_lr_schedule="cosine",
                            coeff_lr_decay_final_mult=0.7,
                            epochs=epochs,
                        )
                    )
                elif dataset == "KMNIST":
                    specs.append(
                        alpha_spec(dataset, "GA-FU-v2-tuned", "geometry_current", alpha_mode, epochs=epochs, **tuned_v2_overrides(dataset))
                    )
                    specs.append(full_from_start_alpha_spec(dataset, "K-V3-FULL-base", alpha_mode, epochs=epochs))
                    specs.append(
                        full_from_start_alpha_spec(
                            dataset,
                            "K-V3-FULL-restcos07",
                            alpha_mode,
                            rest_lr_schedule="cosine",
                            rest_lr_final_mult=0.7,
                            epochs=epochs,
                        )
                    )
                    specs.append(
                        full_from_start_alpha_spec(
                            dataset,
                            "K-V3-FULL-bothcos07",
                            alpha_mode,
                            rest_lr_schedule="cosine",
                            rest_lr_final_mult=0.7,
                            coeff_lr_schedule="cosine",
                            coeff_lr_decay_final_mult=0.7,
                            epochs=epochs,
                        )
                    )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_2_P2_FASHION_CONFIRM":
        for dataset in datasets:
            if dataset != "Fashion-MNIST":
                continue
            epochs = 30
            v2 = {
                **tuned_v2_overrides(dataset),
                "branch_final_scale": 0.8,
                "coeff_lr": 0.05,
                "coeff_lr_decay_final_mult": 0.5,
                "epochs": epochs,
            }
            specs.extend(
                [
                    alpha_spec(dataset, "AdamW", "AdamW", "learnable", epochs=epochs),
                    alpha_spec(dataset, "GA-FU-v2-bothcos", "geometry_current", "learnable", **v2),
                    hard_diag_to_full_alpha_spec(
                        dataset,
                        "F-V3-HARD-base30",
                        "learnable",
                        branch_final_scale=0.8,
                        coeff_lr=0.05,
                        rest_lr=0.001,
                        rest_lr_schedule="cosine",
                        rest_lr_final_mult=0.3,
                        coeff_lr_schedule="cosine",
                        coeff_lr_decay_final_mult=0.5,
                        epochs=epochs,
                    ),
                    alpha_spec(dataset, "AdamW", "AdamW", "fixed1", epochs=epochs),
                    alpha_spec(dataset, "GA-FU-v2-bothcos", "geometry_current", "fixed1", **v2),
                    hard_diag_to_full_alpha_spec(
                        dataset,
                        "F-V3-HARD-base30",
                        "fixed1",
                        branch_final_scale=0.8,
                        coeff_lr=0.05,
                        rest_lr=0.001,
                        rest_lr_schedule="cosine",
                        rest_lr_final_mult=0.3,
                        coeff_lr_schedule="cosine",
                        coeff_lr_decay_final_mult=0.5,
                        epochs=epochs,
                    ),
                    hard_diag_to_full_alpha_spec(
                        dataset,
                        "F-V3-HARD-c006-softcoeff",
                        "fixed1",
                        branch_final_scale=0.85,
                        coeff_lr=0.06,
                        rest_lr=0.001,
                        rest_lr_schedule="cosine",
                        rest_lr_final_mult=0.3,
                        coeff_lr_schedule="cosine",
                        coeff_lr_decay_final_mult=0.7,
                        epochs=epochs,
                    ),
                ]
            )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_2_P3_KMNIST_CONFIRM":
        for dataset in datasets:
            if dataset != "KMNIST":
                continue
            epochs = 20
            specs.extend(
                [
                    alpha_spec(dataset, "AdamW", "AdamW", "learnable", epochs=epochs),
                    alpha_spec(dataset, "GA-FU-v2-tuned", "geometry_current", "learnable", epochs=epochs, **tuned_v2_overrides(dataset)),
                    full_from_start_alpha_spec(
                        dataset,
                        "K-V3-FULL-restcos07",
                        "learnable",
                        rest_lr_schedule="cosine",
                        rest_lr_final_mult=0.7,
                        epochs=epochs,
                    ),
                    full_from_start_alpha_spec(
                        dataset,
                        "K-V3-FULL-bothcos07",
                        "learnable",
                        rest_lr_schedule="cosine",
                        rest_lr_final_mult=0.7,
                        coeff_lr_schedule="cosine",
                        coeff_lr_decay_final_mult=0.7,
                        epochs=epochs,
                    ),
                    alpha_spec(dataset, "AdamW", "AdamW", "fixed1", epochs=epochs),
                    alpha_spec(dataset, "GA-FU-v2-tuned", "geometry_current", "fixed1", epochs=epochs, **tuned_v2_overrides(dataset)),
                    full_from_start_alpha_spec(dataset, "K-V3-FULL-base", "fixed1", epochs=epochs),
                ]
            )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_2_P5_FINAL_FIXED":
        for dataset in datasets:
            if dataset == "Fashion-MNIST":
                epochs = 30
                v2 = {
                    **tuned_v2_overrides(dataset),
                    "branch_final_scale": 0.8,
                    "coeff_lr": 0.05,
                    "coeff_lr_decay_final_mult": 0.5,
                    "epochs": epochs,
                }
                specs.extend(
                    [
                        alpha_spec(dataset, "AdamW-final", "AdamW", "fixed1", epochs=epochs),
                        alpha_spec(dataset, "GA-FU-v2-final", "geometry_current", "fixed1", **v2),
                        hard_diag_to_full_alpha_spec(
                            dataset,
                            "F-V3-HARD-base30-final",
                            "fixed1",
                            branch_final_scale=0.8,
                            coeff_lr=0.05,
                            rest_lr=0.001,
                            rest_lr_schedule="cosine",
                            rest_lr_final_mult=0.3,
                            coeff_lr_schedule="cosine",
                            coeff_lr_decay_final_mult=0.5,
                            epochs=epochs,
                        ),
                    ]
                )
            elif dataset == "KMNIST":
                epochs = 20
                specs.extend(
                    [
                        alpha_spec(dataset, "AdamW-final", "AdamW", "fixed1", epochs=epochs),
                        alpha_spec(dataset, "GA-FU-v2-final", "geometry_current", "fixed1", epochs=epochs, **tuned_v2_overrides(dataset)),
                        full_from_start_alpha_spec(dataset, "K-V3-FULL-base-final", "fixed1", epochs=epochs),
                    ]
                )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_3_P0_SMOKE":
        seeds = [0]
        for dataset in datasets:
            common = {
                "epochs": 1,
                "train_size": 512,
                "val_size": 128,
                "test_size": 128,
                "hidden_dim": 32,
                "depth": 2,
                "basis_count": 8,
                "audit_batch_size": 64,
            }
            specs.extend(
                [
                    alpha_spec(dataset, "AdamW-alphaFixed1", "AdamW", "fixed1", **common),
                    ufull_alpha_spec(dataset, "U-FULL-smoke-alphaFixed1", **common),
                    uo_spec(dataset, "UO-PGAdam-smoke-alphaFixed1", "pgadam", **common),
                    uo_spec(dataset, "UO-NormPGAdam-smoke-alphaFixed1", "norm_pgadam", **common),
                    uo_spec(dataset, "AllFunctionalRestSGD-smoke-alphaFixed1", "all_functional_rest_sgd", **common),
                ]
            )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_3_P1_UFULL_SANITY":
        seeds = [0]
        for dataset in datasets:
            epochs = v33_epochs(dataset)
            specs.extend(
                [
                    alpha_spec(dataset, "AdamW-alphaFixed1", "AdamW", "fixed1", epochs=epochs),
                    v33_gafu_v2_spec(dataset, epochs=epochs),
                    v33_dataset_specific_best(dataset, epochs=epochs),
                ]
            )
            specs.extend(v33_ufull_matrix(dataset, epochs=epochs))
        return specs, seeds, {"package": pkg}
    if pkg == "V3_3_P2_UFULL_EXPR3":
        seeds = [0, 1, 2] if args.seeds == add_args().get_default("seeds") else seeds
        for dataset in datasets:
            epochs = v33_epochs(dataset)
            specs.extend(
                [
                    alpha_spec(dataset, "AdamW-alphaFixed1", "AdamW", "fixed1", epochs=epochs),
                    v33_gafu_v2_spec(dataset, epochs=epochs),
                    v33_dataset_specific_best(dataset, epochs=epochs),
                ]
            )
            specs.extend(v33_ufull_matrix(dataset, epochs=epochs))
        return specs, seeds, {"package": pkg}
    if pkg == "V3_3_P3_UO_SMOKE":
        seeds = [0]
        for dataset in datasets:
            epochs = v33_epochs(dataset, smoke=True)
            specs.extend(
                [
                    alpha_spec(dataset, "AdamW-alphaFixed1", "AdamW", "fixed1", epochs=epochs),
                    ufull_alpha_spec(dataset, "Hybrid-U-FULL-base-alphaFixed1", epochs=epochs),
                    uo_spec(dataset, "UO-PGAdam-alphaFixed1", "pgadam", epochs=epochs),
                    uo_spec(dataset, "UO-NormPGAdam-alphaFixed1", "norm_pgadam", epochs=epochs),
                    uo_spec(dataset, "UO-PostAdam-alphaFixed1", "postadam", epochs=epochs),
                    uo_spec(
                        dataset,
                        "UO-PostAdam-trust030-alphaFixed1",
                        "postadam_trust030",
                        uo_trust_radius=0.30,
                        epochs=epochs,
                    ),
                    uo_spec(dataset, "AllFunctionalRestSGD-negative-alphaFixed1", "all_functional_rest_sgd", epochs=epochs),
                ]
            )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_3_P4_UO_EXPR3":
        seeds = [0, 1, 2] if args.seeds == add_args().get_default("seeds") else seeds
        for dataset in datasets:
            epochs = v33_epochs(dataset)
            specs.extend(
                [
                    alpha_spec(dataset, "AdamW-alphaFixed1", "AdamW", "fixed1", epochs=epochs),
                    ufull_alpha_spec(dataset, "Hybrid-U-FULL-base-alphaFixed1", epochs=epochs),
                    uo_spec(dataset, "UO-PGAdam-alphaFixed1", "pgadam", epochs=epochs),
                    uo_spec(dataset, "UO-NormPGAdam-alphaFixed1", "norm_pgadam", epochs=epochs),
                    uo_spec(dataset, "UO-PostAdam-alphaFixed1", "postadam", epochs=epochs),
                ]
            )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_3_P5_CONFIRM5":
        seeds = [0, 1, 2, 3, 4] if args.seeds == add_args().get_default("seeds") else seeds
        for dataset in datasets:
            epochs = v33_epochs(dataset)
            specs.extend(
                [
                    alpha_spec(dataset, "AdamW-alphaFixed1", "AdamW", "fixed1", epochs=epochs),
                    v33_gafu_v2_spec(dataset, epochs=epochs),
                    v33_dataset_specific_best(dataset, epochs=epochs),
                    ufull_alpha_spec(dataset, "U-FULL-base-alphaFixed1", epochs=epochs),
                    ufull_alpha_spec(dataset, "U-FULL-f085-alphaFixed1", branch_final_scale=0.85, epochs=epochs),
                    ufull_alpha_spec(
                        dataset,
                        "U-FULL-f085-cplus-alphaFixed1",
                        branch_final_scale=0.85,
                        coeff_lr=float(v33_dataset_base(dataset)["coeff_lr"]) * 1.10,
                        epochs=epochs,
                    ),
                    uo_spec(dataset, "UO-NormPGAdam-alphaFixed1", "norm_pgadam", epochs=epochs),
                ]
            )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_3_P6_CONFIRM10":
        seeds = list(range(10)) if args.seeds == add_args().get_default("seeds") else seeds
        for dataset in datasets:
            epochs = v33_epochs(dataset)
            specs.extend(
                [
                    alpha_spec(dataset, "AdamW-alphaFixed1", "AdamW", "fixed1", epochs=epochs),
                    v33_dataset_specific_best(dataset, epochs=epochs),
                    ufull_alpha_spec(dataset, "U-FULL-base-alphaFixed1", epochs=epochs),
                    ufull_alpha_spec(dataset, "U-FULL-f085-alphaFixed1", branch_final_scale=0.85, epochs=epochs),
                    uo_spec(dataset, "UO-NormPGAdam-alphaFixed1", "norm_pgadam", epochs=epochs),
                ]
            )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_4_P0_SMOKE":
        seeds = [0]
        for dataset in datasets:
            common = {
                "epochs": 1,
                "train_size": 512,
                "val_size": 128,
                "test_size": 128,
                "hidden_dim": 32,
                "depth": 2,
                "basis_count": 8,
                "audit_batch_size": 64,
            }
            specs.extend(
                [
                    alpha_spec(dataset, "AdamW-alphaFixed1", "AdamW", "fixed1", **common),
                    afu_hybrid_spec(dataset, **common),
                    afu_spec(dataset, "AFU-1-BranchLocal-alphaFixed1", afu_kan_bias=True, **common),
                    afu_spec(dataset, "AFU-2-HeadCov-alphaFixed1", afu_kan_bias=True, afu_head=True, **common),
                    afu_spec(
                        dataset,
                        "AFU-4-StemDiag-alphaFixed1",
                        afu_kan_bias=True,
                        afu_head=True,
                        afu_stem=True,
                        **common,
                    ),
                    afu_spec(
                        dataset,
                        "AFU-5-FullDiagLite-alphaFixed1",
                        afu_kan_bias=True,
                        afu_head=True,
                        afu_stem=True,
                        afu_ln_scope="all",
                        **common,
                    ),
                ]
            )
            if dataset == "Fashion-MNIST":
                specs.append(
                    afu_spec(
                        dataset,
                        "AFU-6-FullKFACLite-alphaFixed1",
                        afu_kan_bias=True,
                        afu_head=True,
                        afu_stem=True,
                        afu_ln_scope="all",
                        afu_cov_ema_beta=0.98,
                        **common,
                    )
                )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_4_P2_COMPONENT3":
        seeds = [0, 1, 2] if args.seeds == add_args().get_default("seeds") else seeds
        for dataset in datasets:
            epochs = v33_epochs(dataset)
            specs.extend(
                [
                    alpha_spec(dataset, "AdamW-alphaFixed1", "AdamW", "fixed1", epochs=epochs),
                    v33_dataset_specific_best(dataset, epochs=epochs),
                ]
            )
            specs.extend(v34_single_component_specs(dataset, epochs=epochs))
        return specs, seeds, {"package": pkg}
    if pkg == "V3_4_P3_CUMULATIVE3":
        seeds = [0, 1, 2] if args.seeds == add_args().get_default("seeds") else seeds
        for dataset in datasets:
            epochs = v33_epochs(dataset)
            specs.extend(
                [
                    alpha_spec(dataset, "AdamW-alphaFixed1", "AdamW", "fixed1", epochs=epochs),
                    v33_dataset_specific_best(dataset, epochs=epochs),
                ]
            )
            specs.extend(v34_afu_core_specs(dataset, epochs=epochs))
        return specs, seeds, {"package": pkg}
    if pkg == "V3_4_P4_EXPR_RESCUE":
        seeds = [0, 1, 2] if args.seeds == add_args().get_default("seeds") else seeds
        for dataset in datasets:
            if dataset != "KMNIST":
                continue
            epochs = v33_epochs(dataset)
            specs.extend(
                [
                    alpha_spec(dataset, "AdamW-alphaFixed1", "AdamW", "fixed1", epochs=epochs),
                    afu_hybrid_spec(dataset, "AFU-0-Hybrid-f0875-alphaFixed1", branch_final_scale=0.875, epochs=epochs),
                    afu_hybrid_spec(dataset, "AFU-0-Hybrid-f090-alphaFixed1", branch_final_scale=0.90, epochs=epochs),
                    afu_spec(
                        dataset,
                        "AFU-2-HeadCov-f0875-alphaFixed1",
                        afu_kan_bias=True,
                        afu_head=True,
                        branch_final_scale=0.875,
                        epochs=epochs,
                    ),
                    afu_spec(
                        dataset,
                        "AFU-2-HeadCov-f090-alphaFixed1",
                        afu_kan_bias=True,
                        afu_head=True,
                        branch_final_scale=0.90,
                        epochs=epochs,
                    ),
                ]
            )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_4_P5_CONFIRM5":
        seeds = [0, 1, 2, 3, 4] if args.seeds == add_args().get_default("seeds") else seeds
        for dataset in datasets:
            epochs = v33_epochs(dataset)
            specs.extend(
                [
                    alpha_spec(dataset, "AdamW-alphaFixed1", "AdamW", "fixed1", epochs=epochs),
                    v33_dataset_specific_best(dataset, epochs=epochs),
                    afu_hybrid_spec(dataset, epochs=epochs),
                    afu_spec(dataset, "AFU-LNOnly-alphaFixed1", afu_ln_scope="all", epochs=epochs),
                    afu_spec(dataset, "AFU-2-HeadCov-alphaFixed1", afu_kan_bias=True, afu_head=True, epochs=epochs),
                    afu_spec(
                        dataset,
                        "AFU-5-FullDiagLite-alphaFixed1",
                        afu_kan_bias=True,
                        afu_head=True,
                        afu_stem=True,
                        afu_ln_scope="all",
                        epochs=epochs,
                    ),
                ]
            )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_4_P6_CONFIRM10":
        seeds = list(range(10)) if args.seeds == add_args().get_default("seeds") else seeds
        for dataset in datasets:
            epochs = v33_epochs(dataset)
            specs.extend(
                [
                    alpha_spec(dataset, "AdamW-alphaFixed1", "AdamW", "fixed1", epochs=epochs),
                    afu_hybrid_spec(dataset, epochs=epochs),
                    afu_spec(dataset, "AFU-2-HeadCov-alphaFixed1", afu_kan_bias=True, afu_head=True, epochs=epochs),
                ]
            )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_5_P0_CONFIG":
        seeds = [0]
        for dataset in datasets:
            common = {
                "epochs": 1,
                "train_size": 512,
                "val_size": 128,
                "test_size": 128,
                "hidden_dim": 32,
                "depth": 2,
                "basis_count": 8,
                "audit_batch_size": 64,
            }
            specs.append(v35_ufull_f100_spec(dataset, **common))
        return specs, seeds, {"package": pkg}
    if pkg == "V3_5_P1_SEED0":
        seeds = [0]
        for dataset in datasets:
            epochs = v33_epochs(dataset)
            specs.extend(
                [
                    alpha_spec(dataset, "AdamW-alphaFixed1", "AdamW", "fixed1", epochs=epochs),
                    v35_ufull_f085_reference_spec(dataset, epochs=epochs),
                    v35_ufull_f100_spec(dataset, epochs=epochs),
                    v33_dataset_specific_best(dataset, epochs=epochs),
                ]
            )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_5_P2_CONFIRM5":
        seeds = [0, 1, 2, 3, 4] if args.seeds == add_args().get_default("seeds") else seeds
        for dataset in datasets:
            epochs = v33_epochs(dataset)
            specs.extend(
                [
                    alpha_spec(dataset, "AdamW-alphaFixed1", "AdamW", "fixed1", epochs=epochs),
                    v35_ufull_f085_reference_spec(dataset, epochs=epochs),
                    v35_ufull_f100_spec(dataset, epochs=epochs),
                    v33_dataset_specific_best(dataset, epochs=epochs),
                ]
            )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_5_P3_CONFIRM10":
        seeds = list(range(10)) if args.seeds == add_args().get_default("seeds") else seeds
        for dataset in datasets:
            epochs = v33_epochs(dataset)
            specs.extend(
                [
                    alpha_spec(dataset, "AdamW-alphaFixed1", "AdamW", "fixed1", epochs=epochs),
                    v35_ufull_f085_reference_spec(dataset, epochs=epochs),
                    v35_ufull_f100_spec(dataset, epochs=epochs),
                    v33_dataset_specific_best(dataset, epochs=epochs),
                ]
            )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_5_P4_SMALL_DATA":
        seeds = [0, 1, 2, 3, 4] if args.seeds == add_args().get_default("seeds") else seeds
        for dataset in datasets:
            for train_size, epochs in [(500, 50), (1000, 40), (2000, 30), (6000, v33_epochs(dataset))]:
                common = {"train_size": train_size, "epochs": epochs}
                specs.extend(
                    [
                        alpha_spec(dataset, "AdamW-alphaFixed1", "AdamW", "fixed1", **common),
                        v35_ufull_f085_reference_spec(dataset, **common),
                        v35_ufull_f100_spec(dataset, **common),
                    ]
                )
                if dataset == "Fashion-MNIST":
                    specs.append(v33_dataset_specific_best(dataset, epochs=epochs))
        return specs, seeds, {"package": pkg}
    if pkg == "V3_5_P5_LABEL_NOISE":
        seeds = [0, 1, 2, 3, 4] if args.seeds == add_args().get_default("seeds") else seeds
        for dataset in datasets:
            epochs = v33_epochs(dataset)
            for noise in [0.2, 0.4]:
                common = {"label_noise": noise, "epochs": epochs}
                specs.extend(
                    [
                        alpha_spec(dataset, f"AdamW-alphaFixed1-noise{noise:g}", "AdamW", "fixed1", **common),
                        v35_ufull_f085_reference_spec(
                            dataset, label=f"U-FULL-f085-reference-alphaFixed1-noise{noise:g}", **common
                        ),
                        v35_ufull_f100_spec(dataset, label=f"U-FULL-f100-alphaFixed1-noise{noise:g}", **common),
                    ]
                )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_5_P8_RATIONAL":
        seeds = [0, 1, 2] if args.seeds == add_args().get_default("seeds") else seeds
        for dataset in datasets:
            epochs = v33_epochs(dataset)
            specs.extend(
                [
                    alpha_spec(dataset, "MLP-AdamW-alphaFixed1", "mlp_adamw", "fixed1", epochs=epochs),
                    v35_ufull_f085_reference_spec(dataset, epochs=epochs),
                    v35_ufull_f100_spec(dataset, epochs=epochs),
                    v35_rational_adamw_spec(dataset, epochs=epochs, backend="triton"),
                    v35_rational_adamw_spec(dataset, epochs=epochs, backend="torch"),
                    v35_rational_ufull_spec(dataset, epochs=epochs, backend="triton", coeff_lr=0.02),
                    v35_rational_ufull_spec(dataset, epochs=epochs, backend="torch", coeff_lr=0.02),
                ]
            )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_5_P6_CIFAR_SMALL":
        seeds = [0, 1, 2] if args.seeds == add_args().get_default("seeds") else seeds
        for dataset in datasets:
            if dataset != "CIFAR10":
                continue
            common = {"epochs": 20, "train_size": 10000, "val_size": 5000, "test_size": 10000}
            specs.extend(
                [
                    v35_convstem_adamw_spec(dataset, **common),
                    v35_convstem_ufull_spec(dataset, branch_final_scale=0.85, **common),
                    v35_convstem_ufull_spec(dataset, branch_final_scale=1.0, **common),
                ]
            )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_E2":
        for dataset in datasets:
            specs.extend(
                [
                    spec(dataset, "AdamW", "AdamW"),
                    spec(dataset, "GA-FU-v2-tuned", "geometry_current", **tuned_v2_overrides(dataset)),
                    spec(dataset, "V3-phase-aligned-hard", "geometry_current", **v3_legacy_phase_overrides(dataset, phase_mode="hard")),
                    spec(dataset, "V3-phase-aligned-smooth", "geometry_current", **v3_legacy_phase_overrides(dataset, phase_mode="smooth")),
                ]
            )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_E3":
        for dataset in datasets:
            specs.extend(
                [
                    spec(dataset, "AdamW", "AdamW"),
                    spec(dataset, "GA-FU-v2-tuned", "geometry_current", **tuned_v2_overrides(dataset)),
                    v3_metric_overrides(
                        dataset,
                        label="V3-diag-gram",
                        active="basis_diag_gram",
                        transition="basis_diag_gram",
                        geometry="basis_diag_gram",
                    ),
                    v3_metric_overrides(
                        dataset,
                        label="V3-full-from-start",
                        active="full_sobolev_gram",
                        transition="full_sobolev_gram",
                        geometry="full_sobolev_gram",
                    ),
                    v3_metric_overrides(
                        dataset,
                        label="V3-diag-to-full-hard",
                        active="basis_diag_gram",
                        transition="full_sobolev_gram",
                        geometry="full_sobolev_gram",
                        phase_mode="hard",
                    ),
                    v3_metric_overrides(
                        dataset,
                        label="V3-diag-to-full-smooth",
                        active="basis_diag_gram",
                        transition="diag_to_full_sobolev",
                        geometry="full_sobolev_gram",
                    ),
                    v3_metric_overrides(
                        dataset,
                        label="V3-diag-to-full-smooth-softgeo",
                        active="basis_diag_gram",
                        transition="diag_to_full_sobolev",
                        geometry="full_sobolev_gram",
                        alpha_geo=0.05,
                        beta_geo=0.0,
                    ),
                ]
            )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_E4":
        for dataset in datasets:
            specs.append(spec(dataset, "AdamW", "AdamW"))
            if dataset == "Fashion-MNIST":
                matrix = [
                    ("none", "none", 1.0, "none", 1.0),
                    ("restcos", "cosine", 0.3, "none", 1.0),
                    ("coeffcos", "none", 1.0, "cosine", 0.5),
                    ("bothcos", "cosine", 0.3, "cosine", 0.5),
                    ("restcos-softcoeff", "cosine", 0.3, "cosine", 0.7),
                ]
            else:
                matrix = [
                    ("none", "none", 1.0, "none", 1.0),
                    ("restcos07", "cosine", 0.7, "none", 1.0),
                    ("coeffcos07", "none", 1.0, "cosine", 0.7),
                    ("bothcos07", "cosine", 0.7, "cosine", 0.7),
                    ("restcos05", "cosine", 0.5, "none", 1.0),
                ]
            for tag, rest_sched, rest_final, coeff_sched, coeff_final in matrix:
                specs.append(
                    v3_metric_overrides(
                        dataset,
                        label=f"V3-schedule-{tag}",
                        active="basis_diag_gram",
                        transition="diag_to_full_sobolev",
                        geometry="full_sobolev_gram",
                    )
                )
                specs[-1]["overrides"].update(
                    {
                        "rest_lr_schedule": rest_sched,
                        "rest_lr_final_mult": rest_final,
                        "coeff_lr_schedule": coeff_sched,
                        "coeff_lr_decay_final_mult": coeff_final,
                    }
                )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_R3_E3_FIXED":
        seeds = [0] if args.seeds == add_args().get_default("seeds") else seeds
        for dataset in datasets:
            specs.extend(
                [
                    spec(dataset, "AdamW", "AdamW"),
                    spec(dataset, "GA-FU-v2-tuned", "geometry_current", **tuned_v2_overrides(dataset)),
                    v3_metric_overrides(
                        dataset,
                        label="V3-diag-gram",
                        active="basis_diag_gram",
                        transition="basis_diag_gram",
                        geometry="basis_diag_gram",
                    ),
                    full_from_start_spec(dataset, "V3-full-from-start"),
                    hard_diag_to_full_spec(dataset, "V3-diag-to-full-hard"),
                    v3_metric_overrides(
                        dataset,
                        label="V3-diag-to-full-smooth-fixed",
                        active="basis_diag_gram",
                        transition="diag_to_full_sobolev",
                        geometry="full_sobolev_gram",
                    ),
                    v3_metric_overrides(
                        dataset,
                        label="V3-diag-to-full-smooth-fixed-longtr",
                        active="basis_diag_gram",
                        transition="diag_to_full_sobolev",
                        geometry="full_sobolev_gram",
                    ),
                ]
            )
            specs[-1]["overrides"].update(
                {
                    "v3_transition_frac": 0.20,
                    "v3_transition_min_steps": 100,
                    "v3_transition_max_steps": 500,
                }
            )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_R4_KMNIST_FULL_EXPAND":
        for dataset in datasets:
            if dataset != "KMNIST":
                continue
            specs.extend(
                [
                    spec(dataset, "AdamW", "AdamW"),
                    spec(dataset, "GA-FU-v2-tuned", "geometry_current", **tuned_v2_overrides(dataset)),
                    full_from_start_spec(dataset, "K-V3-FULL-base"),
                    full_from_start_spec(
                        dataset,
                        "K-V3-FULL-restcos07",
                        rest_lr_schedule="cosine",
                        rest_lr_final_mult=0.7,
                    ),
                    full_from_start_spec(
                        dataset,
                        "K-V3-FULL-bothcos07",
                        rest_lr_schedule="cosine",
                        rest_lr_final_mult=0.7,
                        coeff_lr_schedule="cosine",
                        coeff_lr_decay_final_mult=0.7,
                    ),
                ]
            )
        return specs, seeds, {"package": pkg}
    if pkg == "V3_R5_FASHION_RESCUE":
        for dataset in datasets:
            if dataset != "Fashion-MNIST":
                continue
            v2 = {
                **tuned_v2_overrides(dataset),
                "branch_final_scale": 0.8,
                "coeff_lr": 0.05,
                "coeff_lr_decay_final_mult": 0.5,
            }
            specs.extend(
                [
                    spec(dataset, "AdamW", "AdamW"),
                    spec(dataset, "GA-FU-v2-tuned-bothcos", "geometry_current", **v2),
                    hard_diag_to_full_spec(
                        dataset,
                        "F-V3-HARD-base30",
                        branch_final_scale=0.8,
                        coeff_lr=0.05,
                        rest_lr=0.001,
                        rest_lr_schedule="cosine",
                        rest_lr_final_mult=0.3,
                        coeff_lr_schedule="cosine",
                        coeff_lr_decay_final_mult=0.5,
                    ),
                    hard_diag_to_full_spec(
                        dataset,
                        "F-V3-HARD-f085",
                        branch_final_scale=0.85,
                        coeff_lr=0.05,
                        rest_lr=0.001,
                        rest_lr_schedule="cosine",
                        rest_lr_final_mult=0.3,
                        coeff_lr_schedule="cosine",
                        coeff_lr_decay_final_mult=0.5,
                    ),
                    hard_diag_to_full_spec(
                        dataset,
                        "F-V3-HARD-c006-softcoeff",
                        branch_final_scale=0.85,
                        coeff_lr=0.06,
                        rest_lr=0.001,
                        rest_lr_schedule="cosine",
                        rest_lr_final_mult=0.3,
                        coeff_lr_schedule="cosine",
                        coeff_lr_decay_final_mult=0.7,
                    ),
                    full_from_start_spec(
                        dataset,
                        "F-V3-FULL-f085",
                        branch_final_scale=0.85,
                        coeff_lr=0.05,
                        rest_lr=0.001,
                        rest_lr_schedule="cosine",
                        rest_lr_final_mult=0.3,
                        coeff_lr_schedule="cosine",
                        coeff_lr_decay_final_mult=0.7,
                    ),
                ]
            )
        return specs, seeds, {"package": pkg}
    raise ValueError(f"unknown package {package!r}")


def pass_level(row: Dict[str, Any], dataset: str) -> str:
    gap = float(row.get("acc_gap_vs_adamw_mean", math.nan))
    auc = float(row.get("val_auc_improvement_vs_adamw_mean", math.nan))
    phi = float(row.get("phi_prime_reduction_vs_adamw_mean", math.nan))
    jac = float(row.get("jac_reduction_vs_adamw_mean", math.nan))
    branch = float(row.get("branch_over_adamw_mean", math.nan))
    ece = float(row.get("ece_reduction_vs_adamw_mean", math.nan))
    if not all(math.isfinite(x) for x in [gap, auc, phi, jac, branch]):
        return "none"
    weak = gap < 0.010 and auc > 0.06 and phi > 0.20 and jac > 0.15 and 0.45 < branch < 0.95
    medium_acc = gap <= 0.0 if dataset == "Fashion-MNIST" else gap < 0.005
    medium = medium_acc and auc > 0.08 and phi > 0.25 and jac > 0.20
    strong = gap < 0.003 and auc > 0.10 and phi > 0.30 and jac > 0.30 and ece > 0.10
    if strong:
        return "strong"
    if medium:
        return "medium"
    if weak:
        return "weak"
    return "none"


def decision_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    compared = add_baseline_comparisons(rows)
    summary = summarize_runs(compared, ["dataset", "method", "label_noise", "train_size", "epochs"])
    levels: Dict[str, Dict[str, str]] = {}
    for item in summary:
        method = str(item.get("method"))
        dataset = str(item.get("dataset"))
        if method == "AdamW":
            continue
        levels.setdefault(method, {})[dataset] = pass_level(item, dataset)

    def rank(level: str) -> int:
        return {"none": 0, "weak": 1, "medium": 2, "strong": 3}.get(level, 0)

    v3_methods = [m for m in levels if m.startswith("GA-FU-v3") or m.startswith("V3-")]
    best_method = ""
    best_min = 0
    for method in v3_methods:
        min_level = min((rank(levels[method].get(ds, "none")) for ds in ["Fashion-MNIST", "KMNIST"]), default=0)
        if min_level > best_min:
            best_method = method
            best_min = min_level
    return {
        "pass_levels": levels,
        "v3_core_pass_level": {0: "none", 1: "weak", 2: "medium", 3: "strong"}.get(best_min, "none"),
        "best_v3_method_by_min_dataset_level": best_method,
    }


def finite_smoke_pass(rows: List[Dict[str, Any]]) -> bool:
    for row in rows:
        if row.get("error"):
            return False
        for key in ["test_acc", "val_auc", "v3_metric_condition_active", "v3_metric_condition_geometry"]:
            value = row.get(key)
            if value in ("", None):
                continue
            try:
                f = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(f) and not int(row.get("gafu_v3_enabled", 0)):
                continue
            if not math.isfinite(f):
                return False
            if "condition" in key and f != 0.0 and not (1.0 < f < 1.0e4):
                return False
    return True


def wandb_available() -> Any:
    try:
        import wandb  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("wandb is not installed; run `pip install wandb` first") from exc
    return wandb


def wandb_log_row(args: argparse.Namespace, row: Dict[str, Any], *, package: str, safe_name: str) -> None:
    if not args.wandb:
        return
    wandb = wandb_available()
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        group=args.wandb_group or package,
        name=safe_name,
        job_type="train",
        config={k: v for k, v in row.items() if isinstance(v, (str, int, float, bool))},
        reinit=True,
    )
    wandb.log({k: v for k, v in row.items() if isinstance(v, (int, float)) and math.isfinite(float(v))})
    run.finish()


def wandb_upload_artifact(args: argparse.Namespace, out_dir: Path) -> None:
    if not args.wandb:
        return
    wandb = wandb_available()
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        group=args.wandb_group or "gafu-v3",
        name=f"upload-{out_dir.name}",
        job_type="upload-results",
        reinit=True,
    )
    artifact = wandb.Artifact(out_dir.name, type="gafu-v3-results")
    for path in out_dir.rglob("*"):
        if path.is_file() and path.suffix in {".csv", ".json", ".md"}:
            artifact.add_file(str(path), name=str(path.relative_to(out_dir)))
    run.log_artifact(artifact)
    run.finish()


def _parse_curve(text: Any) -> List[float]:
    if text in ("", None):
        return []
    if isinstance(text, (int, float)):
        return [float(text)]
    vals: List[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(float(part))
        except ValueError:
            continue
    return vals


def write_run_curve_files(path: Path, row: Dict[str, Any]) -> None:
    ensure_dir(path)
    val_loss = _parse_curve(row.get("val_loss_curve"))
    val_acc = _parse_curve(row.get("val_acc_curve"))
    epoch_time = _parse_curve(row.get("epoch_time_sec_curve"))
    curve_rows: List[Dict[str, Any]] = []
    for idx, loss in enumerate(val_loss):
        curve_rows.append(
            {
                "epoch": idx + 1,
                "val_loss": loss,
                "val_acc": val_acc[idx] if idx < len(val_acc) else "",
                "epoch_time_sec": epoch_time[idx] if idx < len(epoch_time) else "",
                "phase_final": row.get("v3_phase_final", ""),
                "switch_epoch": row.get("v3_phase_switch_epoch", ""),
                "metric_mix_auc": row.get("v3_metric_mix_auc", ""),
                "branch_scale_final": row.get("branch_scale_final", ""),
                "coeff_lr_multiplier_final": row.get("coeff_lr_multiplier_final", ""),
            }
        )
    write_csv(path / "curves.csv", curve_rows)
    write_csv(
        path / "direction_audit.csv",
        [
            {
                "phase": row.get("v3_phase_final", ""),
                "metric_active": row.get("v3_metric_active", ""),
                "metric_transition": row.get("v3_metric_transition", ""),
                "metric_geometry": row.get("v3_metric_geometry", ""),
                "metric_mix_auc": row.get("v3_metric_mix_auc", ""),
                "raw_grad_norm": row.get("update_raw_grad_norm_mean", ""),
                "current_direction_norm": row.get("update_precond_direction_norm_mean", ""),
                "cos_diagfast_fullgeo": row.get("v3_direction_cos_diagfast_fullgeo_mean", ""),
                "norm_fullgeo_over_diagfast": row.get("v3_direction_norm_ratio_fullgeo_diagfast_mean", ""),
                "norm_current_over_diagfast": row.get("v3_direction_norm_ratio_current_diagfast_mean", ""),
                "metric_norm_diagfast": row.get("v3_direction_metric_norm_diagfast_mean", ""),
                "metric_norm_fullgeo": row.get("v3_direction_metric_norm_fullgeo_mean", ""),
                "metric_norm_current": row.get("v3_direction_metric_norm_current_mean", ""),
                "update_over_coeff_norm": row.get("trust_update_over_coeff_norm_mean", ""),
            }
        ],
    )
    branch_keys = sorted(k for k in row if k.startswith("branch_layer_") and k.endswith("_ratio_final"))
    branch_row = {
        "branch_output_norm_ratio": row.get("branch_output_norm_ratio", ""),
        "branch_output_norm_ratio_final": row.get("branch_output_norm_ratio_final", ""),
        "no_kan_drop": row.get("no_kan_drop", ""),
        "no_kan_test_acc": row.get("no_kan_test_acc", ""),
        "no_kan_loss_increase": row.get("no_kan_loss_increase", ""),
        "kan_logit_delta_norm_mean": row.get("kan_logit_delta_norm_mean", ""),
        "kan_margin_contribution_mean": row.get("kan_margin_contribution_mean", ""),
    }
    for key in branch_keys:
        branch_row[key] = row.get(key, "")
    write_csv(path / "branch_audit.csv", [branch_row])
    write_csv(
        path / "moment_audit.csv",
        [
            {
                "unified_optimizer_mode": row.get("unified_optimizer_mode", ""),
                "optimizer_path": row.get("optimizer_path", ""),
                "kan_grad_transform": row.get("kan_grad_transform", ""),
                "kan_raw_grad_norm": row.get("optimizer_kan_raw_grad_norm_mean", ""),
                "kan_precond_grad_norm": row.get("optimizer_kan_precond_grad_norm_mean", ""),
                "kan_precond_over_raw_norm": row.get("optimizer_kan_precond_over_raw_norm_mean", ""),
                "rest_grad_norm": row.get("optimizer_rest_grad_norm_mean", ""),
                "kan_update_norm": row.get("optimizer_kan_update_norm_mean", ""),
                "rest_update_norm": row.get("optimizer_rest_update_norm_mean", ""),
                "kan_update_over_param_norm": row.get("optimizer_kan_update_over_param_norm_mean", ""),
                "rest_update_over_param_norm": row.get("optimizer_rest_update_over_param_norm_mean", ""),
                "kan_update_over_rest_update": row.get("optimizer_kan_update_over_rest_update_mean", ""),
                "adam_m_norm_kan": row.get("optimizer_adam_m_norm_kan_mean", ""),
                "adam_v_norm_kan": row.get("optimizer_adam_v_norm_kan_mean", ""),
                "moment_cos_raw_precond": row.get("optimizer_moment_cos_raw_precond_mean", ""),
                "moment_cos_precond_current": row.get("optimizer_moment_cos_precond_current_mean", ""),
                "moment_cos_raw_current": row.get("optimizer_moment_cos_raw_current_mean", ""),
                "moment_amplification_ratio": row.get("optimizer_moment_amplification_ratio_mean", ""),
                "bad_step_count": row.get("optimizer_bad_step_count", ""),
                "bad_step_rate": row.get("optimizer_bad_step_rate", ""),
                "trust_clip_rate": row.get("trust_clip_rate", ""),
            }
        ],
    )
    group_rows: List[Dict[str, Any]] = []
    for group in ["kan_bias", "head", "stem", "ln"]:
        group_rows.append(
            {
                "group": group,
                "update_norm_mean": row.get(f"afu_{group}_update_norm_mean", ""),
                "update_over_param_mean": row.get(f"afu_{group}_update_over_param_mean", ""),
                "update_over_param_p95": row.get(f"afu_{group}_update_over_param_p95", ""),
                "raw_grad_norm_mean": row.get(f"afu_{group}_raw_grad_norm_mean", ""),
                "precond_grad_norm_mean": row.get(f"afu_{group}_precond_grad_norm_mean", ""),
                "cos_raw_precond_mean": row.get(f"afu_{group}_cos_raw_precond_mean", ""),
                "metric_condition_mean": row.get(f"afu_{group}_metric_condition_mean", ""),
                "metric_min_mean": row.get(f"afu_{group}_metric_min_mean", ""),
                "metric_max_mean": row.get(f"afu_{group}_metric_max_mean", ""),
                "update_share_mean": row.get(f"afu_{group}_update_share_mean", ""),
            }
        )
    write_csv(path / "group_update_audit.csv", group_rows)
    write_csv(
        path / "metric_condition_audit.csv",
        [
            {
                "kan_gram_condition": row.get("v3_metric_condition_geometry", ""),
                "head_cov_condition": row.get("afu_head_metric_condition_mean", ""),
                "stem_cov_diag_condition": row.get("afu_stem_metric_condition_mean", ""),
                "ln_fisher_condition": row.get("afu_ln_metric_condition_mean", ""),
                "kan_bias_diag_condition": row.get("afu_kan_bias_metric_condition_mean", ""),
                "afu_clip_rate": row.get("afu_clip_rate", ""),
                "afu_head_cov_time_ms": row.get("afu_head_cov_time_ms", ""),
                "afu_head_solve_time_ms": row.get("afu_head_solve_time_ms", ""),
                "afu_stem_metric_time_ms": row.get("afu_stem_metric_time_ms", ""),
                "afu_ln_metric_time_ms": row.get("afu_ln_metric_time_ms", ""),
            }
        ],
    )


def summarize_failures(rows: List[Dict[str, Any]], group_keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in group_keys)
        item = groups.setdefault(key, {k: v for k, v in zip(group_keys, key)})
        item["failures"] = int(item.get("failures", 0)) + 1
        failure = str(row.get("failure_type", "") or "run_failed")
        if failure:
            item[f"failure_{failure}"] = int(item.get(f"failure_{failure}", 0)) + 1
    return list(groups.values())


def run_r0_checks(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    torch.manual_seed(0)
    layer = RBFDense(4, 3, 8)
    grad = torch.randn(3, 4, 8)
    cfg = TrainConfig(gafu_v3_enabled=True, v3_gram_grid_size=args.v3_gram_grid_size)
    fast = build_rbf_sobolev_gram(
        layer.centers,
        layer.width,
        cfg.v3_alpha_fast,
        cfg.v3_beta_fast,
        cfg.v3_gram_rho,
        grid_size=cfg.v3_gram_grid_size,
        device=grad.device,
        dtype=grad.dtype,
    )
    geo = build_rbf_sobolev_gram(
        layer.centers,
        layer.width,
        cfg.v3_alpha_geo,
        cfg.v3_beta_geo,
        cfg.v3_gram_rho,
        grid_size=cfg.v3_gram_grid_size,
        device=grad.device,
        dtype=grad.dtype,
    )
    diagfast_expected = -grad / _gram_diag(fast, grad).clamp_min(cfg.v3_gram_rho)
    fullgeo_expected = -_solve_with_gram(grad, geo)

    state0 = RuntimeState(current_metric_mix=0.0, phase="TRANSITION")
    actual0, _ = _v3_precondition_direction(
        grad, layer, cfg, state0, metric_mode="diag_to_full_sobolev", phase="TRANSITION"
    )
    state1 = RuntimeState(current_metric_mix=1.0, phase="TRANSITION")
    actual1, _ = _v3_precondition_direction(
        grad, layer, cfg, state1, metric_mode="diag_to_full_sobolev", phase="TRANSITION"
    )

    rel0 = float((actual0 - diagfast_expected).norm() / diagfast_expected.norm().clamp_min(1e-12))
    rel1 = float((actual1 - fullgeo_expected).norm() / fullgeo_expected.norm().clamp_min(1e-12))
    length = _v3_transition_length(cfg, 100)
    mix_first = _transition_soft(cfg, 1 / max(1, length))
    mix_mid = _transition_soft(cfg, 0.5)
    mix_last = _transition_soft(cfg, 1.0)
    cond_fast = float(fast["condition"])
    cond_geo = float(geo["condition"])
    row = {
        "stage": "GA-FU-v3",
        "package": "V3_R0_CHECKS",
        "endpoint_relerr_mix0": rel0,
        "endpoint_relerr_mix1": rel1,
        "cond_fast": cond_fast,
        "cond_geo": cond_geo,
        "eig_min_fast": float(fast["eig_min"]),
        "eig_min_geo": float(geo["eig_min"]),
        "eig_max_fast": float(fast["eig_max"]),
        "eig_max_geo": float(geo["eig_max"]),
        "mix_first": mix_first,
        "mix_mid": mix_mid,
        "mix_last": mix_last,
        "transition_length_steps": length,
        "direction_cos_diagfast_fullgeo": state1.direction_cos_diagfast_fullgeo[0],
        "direction_norm_ratio_fullgeo_diagfast": state1.direction_norm_ratio_fullgeo_diagfast[0],
    }
    row["pass"] = (
        rel0 < 1e-6
        and rel1 < 1e-6
        and all(math.isfinite(float(row[k])) for k in ["cond_fast", "cond_geo", "mix_first", "mix_last"])
        and abs(cond_fast - cond_geo) > 1e-6
        and mix_first > 0.0
        and abs(mix_last - 1.0) < 1e-12
    )
    save_json(out_dir / "r0_checks.json", row)
    write_csv(out_dir / "r0_checks.csv", [row])
    save_json(out_dir / "aggregate_summary.json", {"new_runs": 1, "r0_checks": row})
    wandb_upload_artifact(args, out_dir)
    print(
        "R0 checks "
        f"pass={row['pass']} rel0={rel0:.3e} rel1={rel1:.3e} "
        f"cond_fast={cond_fast:.2f} cond_geo={cond_geo:.2f} mix_first={mix_first:.4f}"
    )
    return [row]


def add_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packages", default="V3_E0")
    p.add_argument("--datasets", default="Fashion-MNIST,KMNIST")
    p.add_argument("--methods", default="")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--out-dir", type=Path, default=Path("results/gafu_v3"))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="DG-KAN")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--wandb-group", default="")
    add_common_train_args(p)
    return p


def run_specs(args: argparse.Namespace, specs: List[Spec], seeds: List[int], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    run_dir = ensure_dir(out_dir / "runs")
    if args.methods:
        wanted = set(parse_str_list(args.methods))
        specs = [
            item
            for item in specs
            if str(item.get("label", "")) in wanted or str(item.get("base_method", "")) in wanted
        ]
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "runs.csv")
    new_rows: List[Dict[str, Any]] = []
    package = str(meta.get("package", "V3"))
    completed = {
        (
            str(row.get("package", "")),
            str(row.get("dataset", "")),
            str(row.get("method", "")),
            str(row.get("alpha_mode", "")),
            int(row.get("seed", -1)) if str(row.get("seed", "")).strip() else -1,
            int(float(row.get("train_size", -1))) if str(row.get("train_size", "")).strip() else -1,
            int(float(row.get("epochs", -1))) if str(row.get("epochs", "")).strip() else -1,
            float(row.get("label_noise", 0.0) or 0.0),
        )
        for row in rows
        if row.get("error", "") == ""
    }
    total = len(specs) * len(seeds)
    done = 0
    for item in specs:
        for seed in seeds:
            done += 1
            dataset = item["dataset"]
            label = item["label"]
            base_method = item["base_method"]
            overrides = dict(item.get("overrides", {}))
            alpha_mode = str(overrides.get("alpha_mode", getattr(args, "alpha_mode", "")))
            run_train_size = int(overrides.get("train_size", args.train_size))
            run_epochs = int(overrides.get("epochs", args.epochs))
            run_label_noise = float(overrides.get("label_noise", 0.0))
            key = (package, dataset, label, alpha_mode, int(seed), run_train_size, run_epochs, run_label_noise)
            if key in completed:
                print(f"[{done}/{total}] SKIP existing {package} {dataset} {label} alpha={alpha_mode} seed={seed}")
                continue
            try:
                cfg = config_from_args(args, dataset, base_method, seed, **overrides)
                cfg.method = label
                cfg.notes = f"gafu_v3 package={package} base_method={base_method}"
                row = train_one(cfg)
                row["stage"] = "GA-FU-v3"
                row["package"] = package
                row["base_method"] = base_method
                print(
                    f"[{done}/{total}] {package} {dataset} {label} alpha={row.get('alpha_mode')} seed={seed} "
                    f"acc={row['test_acc']:.4f} auc={row['val_auc']:.4f} "
                    f"phase={row.get('v3_phase_final')} mix={row.get('v3_metric_mix_auc', 0.0):.3f} "
                    f"condA={row.get('v3_metric_condition_active', math.nan):.2f} "
                    f"condG={row.get('v3_metric_condition_geometry', math.nan):.2f} "
                    f"clip={row.get('trust_clip_rate', 0.0):.3f}"
                )
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                row = {
                    "stage": "GA-FU-v3",
                    "package": package,
                    "dataset": dataset,
                    "method": label,
                    "base_method": base_method,
                    "seed": seed,
                    "error": repr(exc),
                    "test_acc": math.nan,
                }
                print(f"[{done}/{total}] ERROR {package} {dataset} {label} seed={seed}: {exc!r}")

            rows.append(row)
            new_rows.append(row)
            completed.add(key)
            safe_name = f"{package}_{dataset}_{label}_{row.get('alpha_mode', alpha_mode)}_seed{seed}_e{row.get('epochs', args.epochs)}"
            safe_name = safe_name.replace("/", "_").replace(" ", "_")
            save_json(run_dir / f"{safe_name}.json", row)
            write_run_curve_files(run_dir / safe_name, row)
            wandb_log_row(args, row, package=package, safe_name=safe_name)
            write_csv(out_dir / "runs.csv", add_baseline_comparisons(rows))

    compared = add_baseline_comparisons(rows)
    summary_method = summarize_runs(compared, ["package", "dataset", "alpha_mode", "method", "label_noise", "train_size", "epochs"])
    summary_dataset = summarize_runs(compared, ["package", "dataset", "alpha_mode", "label_noise", "train_size", "epochs"])
    report = decision_report(compared)
    failure_tagged = add_failure_types(compared)
    failure_table = [row for row in failure_tagged if row.get("error") or row.get("failure_type")]
    write_csv(out_dir / "summary_by_method.csv", summary_method)
    write_csv(out_dir / "summary_by_dataset.csv", summary_dataset)
    write_csv(out_dir / "failure_table.csv", failure_table)
    write_csv(out_dir / "failure_summary_by_method.csv", summarize_failures(failure_table, ["package", "dataset", "alpha_mode", "method"]))
    write_csv(out_dir / "failure_summary_by_dataset.csv", summarize_failures(failure_table, ["package", "dataset", "alpha_mode"]))
    save_json(
        out_dir / "aggregate_summary.json",
        {
            "new_runs": len(new_rows),
            "total_rows": len(compared),
            "smoke_finite_pass": finite_smoke_pass(new_rows),
            "summary_by_method": summary_method,
            "summary_by_dataset": summary_dataset,
            "decision_report": report,
        },
    )
    wandb_upload_artifact(args, out_dir)
    print(f"Wrote {len(new_rows)} new rows to {out_dir}")
    print(f"V3 best min-dataset pass level: {report['v3_core_pass_level']} ({report['best_v3_method_by_min_dataset_level']})")
    return compared


def main() -> int:
    args = add_args().parse_args()
    all_rows: List[Dict[str, Any]] = []
    packages = parse_str_list(args.packages)
    first = True
    original_fresh = args.fresh
    for package in packages:
        if package.strip().upper().replace("-", "_") == "V3_R0_CHECKS":
            args.fresh = original_fresh and first
            all_rows = run_r0_checks(args)
            first = False
            continue
        specs, seeds, meta = package_specs(args, package)
        args.fresh = original_fresh and first
        all_rows = run_specs(args, specs, seeds, meta)
        first = False
    return 0 if all_rows is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
