#!/usr/bin/env python3
"""GA-FU consolidation runner.

This implements the executable surface in
docs/DG-KAN_GA-FU_统一Optimizer下一轮实验计划.md.  GA-FU is not a new component
stack here: it is the frozen geometry-aware functional update profile with
dataset-specific branch activation and late geometry control.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from dgkan_core import (
    add_common_train_args,
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
    raise ValueError(f"unknown dataset {name!r}")


def method_name(name: str) -> str:
    key = name.strip().lower().replace("_", "-")
    mapping = {
        "adamw": "AdamW",
        "dgkan-adamw": "AdamW",
        "static": "StaticFunctional",
        "static-func": "StaticFunctional",
        "staticfunctional": "StaticFunctional",
        "gafu": "GA-FU",
        "ga-fu": "GA-FU",
        "gafu-km-acc": "GA-FU-KM-acc",
        "ga-fu-km-acc": "GA-FU-KM-acc",
        "gafu-km-boost1.2": "GA-FU-KM-boost1.2",
        "ga-fu-km-boost1.2": "GA-FU-KM-boost1.2",
        "gafu-km-clr0.1-rest0.001": "GA-FU-KM-clr0.1-rest0.001",
        "ga-fu-km-clr0.1-rest0.001": "GA-FU-KM-clr0.1-rest0.001",
        "gafu-km-clr0.2-balanced": "GA-FU-KM-clr0.2-balanced",
        "ga-fu-km-clr0.2-balanced": "GA-FU-KM-clr0.2-balanced",
        "gafu-km-final0.8": "GA-FU-KM-final0.8",
        "ga-fu-km-final0.8": "GA-FU-KM-final0.8",
        "gafu-fm-clr0.05-rest0.001": "GA-FU-FM-clr0.05-rest0.001",
        "ga-fu-fm-clr0.05-rest0.001": "GA-FU-FM-clr0.05-rest0.001",
        "gafu-fm-boost1.2-clr0.05-active0.35": "GA-FU-FM-boost1.2-clr0.05-active0.35",
        "ga-fu-fm-boost1.2-clr0.05-active0.35": "GA-FU-FM-boost1.2-clr0.05-active0.35",
    }
    return mapping.get(key, name.strip())


def spec(dataset: str, label: str, base_method: str, **overrides: Any) -> Spec:
    return {
        "dataset": dataset,
        "label": label,
        "base_method": base_method,
        "overrides": overrides,
    }


def gafu_overrides(dataset: str, *, coeff_lr: float = 0.3, rest_lr: float = 0.003,
                   final_scale: float | None = None) -> Dict[str, Any]:
    if dataset == "Fashion-MNIST":
        return {
            "coeff_lr": coeff_lr,
            "rest_lr": rest_lr,
            "branch_boost": 2.0,
            "coeff_lr_boost": 1.5,
            "branch_max_active_frac": 0.50,
            "branch_final_scale": 0.8 if final_scale is None else final_scale,
            "geometry_phi_high": 0.052,
            "geometry_jac_high": 4.5,
            "geometry_branch_high": 0.30,
        }
    if dataset == "KMNIST":
        return {
            "coeff_lr": coeff_lr,
            "rest_lr": rest_lr,
            "branch_boost": 1.5,
            "coeff_lr_boost": 1.2,
            "branch_max_active_frac": 0.35,
            "branch_final_scale": 0.9 if final_scale is None else final_scale,
            "geometry_phi_high": 0.055,
            "geometry_jac_high": 4.5,
            "geometry_branch_high": 0.30,
        }
    return {
        "coeff_lr": coeff_lr,
        "rest_lr": rest_lr,
        "branch_boost": 1.5,
        "coeff_lr_boost": 1.2,
        "branch_max_active_frac": 0.35,
        "branch_final_scale": 0.9 if final_scale is None else final_scale,
        "geometry_phi_high": 0.055,
        "geometry_jac_high": 4.5,
        "geometry_branch_high": 0.30,
    }


def static_overrides() -> Dict[str, Any]:
    return {"coeff_lr": 0.3, "rest_lr": 0.003}


def specs_for_methods(datasets: Sequence[str], methods: Sequence[str]) -> List[Spec]:
    out: List[Spec] = []
    for dataset in datasets:
        for method in methods:
            method = method_name(method)
            if method == "AdamW":
                out.append(spec(dataset, "AdamW", "AdamW"))
            elif method == "StaticFunctional":
                out.append(spec(dataset, "StaticFunctional", "static_top", **static_overrides()))
            elif method == "GA-FU":
                out.append(spec(dataset, "GA-FU", "geometry_current", **gafu_overrides(dataset)))
            elif method == "GA-FU-KM-acc":
                if dataset == "KMNIST":
                    out.append(spec(dataset, "GA-FU-KM-acc", "geometry_current", **gafu_overrides(dataset, final_scale=1.0)))
            elif method == "GA-FU-KM-final0.8":
                if dataset == "KMNIST":
                    out.append(spec(dataset, "GA-FU-KM-final0.8", "geometry_current", **gafu_overrides(dataset, final_scale=0.8)))
            elif method == "GA-FU-KM-boost1.2":
                if dataset == "KMNIST":
                    out.append(spec(dataset, "GA-FU-KM-boost1.2", "geometry_current", **{
                        **gafu_overrides(dataset, coeff_lr=0.2),
                        "branch_boost": 1.2,
                    }))
            elif method == "GA-FU-KM-clr0.1-rest0.001":
                if dataset == "KMNIST":
                    out.append(spec(dataset, "GA-FU-KM-clr0.1-rest0.001", "geometry_current", **gafu_overrides(dataset, coeff_lr=0.1, rest_lr=0.001)))
            elif method == "GA-FU-KM-clr0.2-balanced":
                if dataset == "KMNIST":
                    out.append(spec(dataset, "GA-FU-KM-clr0.2-balanced", "geometry_current", **gafu_overrides(dataset, coeff_lr=0.2)))
            elif method == "GA-FU-FM-clr0.05-rest0.001":
                if dataset == "Fashion-MNIST":
                    out.append(spec(dataset, "GA-FU-FM-clr0.05-rest0.001", "geometry_current", **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.001)))
            elif method == "GA-FU-FM-boost1.2-clr0.05-active0.35":
                if dataset == "Fashion-MNIST":
                    out.append(spec(dataset, "GA-FU-FM-boost1.2-clr0.05-active0.35", "geometry_current", **{
                        **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.001),
                        "branch_boost": 1.2,
                        "coeff_lr_boost": 1.0,
                        "branch_max_active_frac": 0.35,
                    }))
            else:
                raise ValueError(f"unknown method {method!r}")
    return out


def tune_specs(datasets: Sequence[str]) -> List[Spec]:
    out: List[Spec] = []
    for dataset in datasets:
        out.append(spec(dataset, "AdamW", "AdamW"))
        out.append(spec(dataset, "StaticFunctional", "static_top", **static_overrides()))
        out.append(spec(dataset, "GA-FU", "geometry_current", **gafu_overrides(dataset)))
        if dataset == "Fashion-MNIST":
            out.extend(
                [
                    spec(dataset, "GA-FU-FM-clr0.05-rest0.001", "geometry_current", **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.001)),
                    spec(dataset, "GA-FU-FM-clr0.05-active0.25", "geometry_current", **{
                        **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.001),
                        "branch_max_active_frac": 0.25,
                    }),
                    spec(dataset, "GA-FU-FM-clr0.05-active0.35", "geometry_current", **{
                        **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.001),
                        "branch_max_active_frac": 0.35,
                    }),
                    spec(dataset, "GA-FU-FM-boost1.5-clr0.05-active0.35", "geometry_current", **{
                        **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.001),
                        "branch_boost": 1.5,
                        "coeff_lr_boost": 1.2,
                        "branch_max_active_frac": 0.35,
                    }),
                    spec(dataset, "GA-FU-FM-boost1.2-clr0.05-active0.35", "geometry_current", **{
                        **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.001),
                        "branch_boost": 1.2,
                        "coeff_lr_boost": 1.0,
                        "branch_max_active_frac": 0.35,
                    }),
                    spec(dataset, "GA-FU-FM-clr0.1-rest0.001", "geometry_current", **gafu_overrides(dataset, coeff_lr=0.1, rest_lr=0.001)),
                    spec(dataset, "GA-FU-FM-clr0.2-rest0.001", "geometry_current", **gafu_overrides(dataset, coeff_lr=0.2, rest_lr=0.001)),
                    spec(dataset, "GA-FU-FM-clr0.2", "geometry_current", **gafu_overrides(dataset, coeff_lr=0.2)),
                    spec(dataset, "GA-FU-FM-final0.9", "geometry_current", **gafu_overrides(dataset, final_scale=0.9)),
                    spec(dataset, "GA-FU-FM-clr0.4", "geometry_current", **gafu_overrides(dataset, coeff_lr=0.4)),
                    spec(dataset, "GA-FU-FM-rest0.004", "geometry_current", **gafu_overrides(dataset, rest_lr=0.004)),
                ]
            )
        elif dataset == "KMNIST":
            out.extend(
                [
                    spec(dataset, "GA-FU-KM-acc", "geometry_current", **gafu_overrides(dataset, final_scale=1.0)),
                    spec(dataset, "GA-FU-KM-final0.8", "geometry_current", **gafu_overrides(dataset, final_scale=0.8)),
                    spec(dataset, "GA-FU-KM-clr0.05-rest0.001", "geometry_current", **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.001)),
                    spec(dataset, "GA-FU-KM-clr0.1-rest0.001", "geometry_current", **gafu_overrides(dataset, coeff_lr=0.1, rest_lr=0.001)),
                    spec(dataset, "GA-FU-KM-clr0.2-rest0.001", "geometry_current", **gafu_overrides(dataset, coeff_lr=0.2, rest_lr=0.001)),
                    spec(dataset, "GA-FU-KM-clr0.2-balanced", "geometry_current", **gafu_overrides(dataset, coeff_lr=0.2)),
                    spec(dataset, "GA-FU-KM-clr0.2-acc", "geometry_current", **gafu_overrides(dataset, coeff_lr=0.2, final_scale=1.0)),
                    spec(dataset, "GA-FU-KM-clr0.1-balanced", "geometry_current", **gafu_overrides(dataset, coeff_lr=0.1)),
                    spec(dataset, "GA-FU-KM-boost1.2", "geometry_current", **{
                        **gafu_overrides(dataset, coeff_lr=0.2),
                        "branch_boost": 1.2,
                    }),
                    spec(dataset, "GA-FU-KM-clr0.4-balanced", "geometry_current", **gafu_overrides(dataset, coeff_lr=0.4)),
                    spec(dataset, "GA-FU-KM-clr0.4-acc", "geometry_current", **gafu_overrides(dataset, coeff_lr=0.4, final_scale=1.0)),
                    spec(dataset, "GA-FU-KM-rest0.004-acc", "geometry_current", **gafu_overrides(dataset, rest_lr=0.004, final_scale=1.0)),
                    spec(dataset, "GA-FU-KM-clr0.5-acc", "geometry_current", **gafu_overrides(dataset, coeff_lr=0.5, final_scale=1.0)),
                ]
            )
    return out


def auc_tune_specs(datasets: Sequence[str]) -> List[Spec]:
    """Small package for GA-FU variants that target validation-loss AUC."""
    out: List[Spec] = []
    for dataset in datasets:
        out.append(spec(dataset, "AdamW", "AdamW"))
        if dataset == "Fashion-MNIST":
            out.extend(
                [
                    spec(dataset, "GA-FU-FM-auc-b1.0-c0.03-r0.001-f0.9-g0", "geometry_current", **{
                        **gafu_overrides(dataset, coeff_lr=0.03, rest_lr=0.001, final_scale=0.9),
                        "branch_boost": 1.0,
                        "coeff_lr_boost": 1.0,
                        "geometry_min_epochs": 0,
                    }),
                    spec(dataset, "GA-FU-FM-auc-b1.0-c0.05-r0.001-f0.9-g0", "geometry_current", **{
                        **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.001, final_scale=0.9),
                        "branch_boost": 1.0,
                        "coeff_lr_boost": 1.0,
                        "geometry_min_epochs": 0,
                    }),
                    spec(dataset, "GA-FU-FM-auc-b1.1-c0.05-r0.001-f0.9-g0", "geometry_current", **{
                        **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.001, final_scale=0.9),
                        "branch_boost": 1.1,
                        "coeff_lr_boost": 1.0,
                        "geometry_min_epochs": 0,
                    }),
                    spec(dataset, "GA-FU-FM-auc-b1.2-c0.05-r0.001-f0.8-g0", "geometry_current", **{
                        **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.001, final_scale=0.8),
                        "branch_boost": 1.2,
                        "coeff_lr_boost": 1.0,
                        "branch_max_active_frac": 0.35,
                        "geometry_min_epochs": 0,
                    }),
                    spec(dataset, "GA-FU-FM-auc-b1.2-c0.05-r0.001-f0.9-g0", "geometry_current", **{
                        **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.001, final_scale=0.9),
                        "branch_boost": 1.2,
                        "coeff_lr_boost": 1.0,
                        "branch_max_active_frac": 0.35,
                        "geometry_min_epochs": 0,
                    }),
                    spec(dataset, "GA-FU-FM-auc-b1.2-c0.05-r0.001-f1.0-g0", "geometry_current", **{
                        **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.001, final_scale=1.0),
                        "branch_boost": 1.2,
                        "coeff_lr_boost": 1.0,
                        "branch_max_active_frac": 0.35,
                        "geometry_min_epochs": 0,
                    }),
                    spec(dataset, "GA-FU-FM-auc-b1.0-c0.05-r0.003-f0.9-g0", "geometry_current", **{
                        **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.003, final_scale=0.9),
                        "branch_boost": 1.0,
                        "coeff_lr_boost": 1.0,
                        "geometry_min_epochs": 0,
                    }),
                ]
            )
        elif dataset == "KMNIST":
            out.extend(
                [
                    spec(dataset, "GA-FU-KM-auc-b1.0-c0.05-r0.001-f0.9-g0", "geometry_current", **{
                        **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.001, final_scale=0.9),
                        "branch_boost": 1.0,
                        "coeff_lr_boost": 1.0,
                        "geometry_min_epochs": 0,
                    }),
                    spec(dataset, "GA-FU-KM-auc-b1.0-c0.10-r0.001-f0.9-g0", "geometry_current", **{
                        **gafu_overrides(dataset, coeff_lr=0.10, rest_lr=0.001, final_scale=0.9),
                        "branch_boost": 1.0,
                        "coeff_lr_boost": 1.0,
                        "geometry_min_epochs": 0,
                    }),
                    spec(dataset, "GA-FU-KM-auc-b1.1-c0.10-r0.001-f0.9-g0", "geometry_current", **{
                        **gafu_overrides(dataset, coeff_lr=0.10, rest_lr=0.001, final_scale=0.9),
                        "branch_boost": 1.1,
                        "coeff_lr_boost": 1.0,
                        "geometry_min_epochs": 0,
                    }),
                    spec(dataset, "GA-FU-KM-auc-b1.2-c0.10-r0.001-f0.8-g0", "geometry_current", **{
                        **gafu_overrides(dataset, coeff_lr=0.10, rest_lr=0.001, final_scale=0.8),
                        "branch_boost": 1.2,
                        "coeff_lr_boost": 1.0,
                        "geometry_min_epochs": 0,
                    }),
                    spec(dataset, "GA-FU-KM-auc-b1.1-c0.20-r0.003-f0.9-g0", "geometry_current", **{
                        **gafu_overrides(dataset, coeff_lr=0.20, rest_lr=0.003, final_scale=0.9),
                        "branch_boost": 1.1,
                        "coeff_lr_boost": 1.0,
                        "geometry_min_epochs": 0,
                    }),
                ]
            )
    return out


def capacity_tune_specs(datasets: Sequence[str]) -> List[Spec]:
    out: List[Spec] = []
    for dataset in datasets:
        out.append(spec(dataset, "AdamW", "AdamW"))
        if dataset == "Fashion-MNIST":
            out.extend(
                [
                    spec(dataset, "GA-FU-FM-cap-b1.0-c0.05-r0.001-f0.9-g0", "geometry_current", **{
                        **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.001, final_scale=0.9),
                        "branch_boost": 1.0,
                        "coeff_lr_boost": 1.0,
                        "geometry_min_epochs": 0,
                    }),
                    spec(dataset, "GA-FU-FM-cap-b1.2-c0.05-r0.001-f0.8-g0", "geometry_current", **{
                        **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.001, final_scale=0.8),
                        "branch_boost": 1.2,
                        "coeff_lr_boost": 1.0,
                        "branch_max_active_frac": 0.35,
                        "geometry_min_epochs": 0,
                    }),
                ]
            )
        elif dataset == "KMNIST":
            out.extend(
                [
                    spec(dataset, "GA-FU-KM-cap-b1.1-c0.10-r0.001-f0.9-g0", "geometry_current", **{
                        **gafu_overrides(dataset, coeff_lr=0.10, rest_lr=0.001, final_scale=0.9),
                        "branch_boost": 1.1,
                        "coeff_lr_boost": 1.0,
                        "geometry_min_epochs": 0,
                    }),
                    spec(dataset, "GA-FU-KM-cap-b1.2-c0.10-r0.001-f0.8-g0", "geometry_current", **{
                        **gafu_overrides(dataset, coeff_lr=0.10, rest_lr=0.001, final_scale=0.8),
                        "branch_boost": 1.2,
                        "coeff_lr_boost": 1.0,
                        "geometry_min_epochs": 0,
                    }),
                    spec(dataset, "GA-FU-KM-cap-b1.1-c0.20-r0.003-f0.9-g0", "geometry_current", **{
                        **gafu_overrides(dataset, coeff_lr=0.20, rest_lr=0.003, final_scale=0.9),
                        "branch_boost": 1.1,
                        "coeff_lr_boost": 1.0,
                        "geometry_min_epochs": 0,
                    }),
                ]
            )
    return out


def fashion_fast_tune_specs(datasets: Sequence[str]) -> List[Spec]:
    out: List[Spec] = []
    for dataset in datasets:
        if dataset != "Fashion-MNIST":
            continue
        out.append(spec(dataset, "AdamW", "AdamW"))
        out.extend(
            [
                spec(dataset, "GA-FU-FM-fast-b1.0-c0.05-cb1.5-r0.001-f0.9-g0", "geometry_current", **{
                    **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.001, final_scale=0.9),
                    "branch_boost": 1.0,
                    "coeff_lr_boost": 1.5,
                    "geometry_min_epochs": 0,
                }),
                spec(dataset, "GA-FU-FM-fast-b1.0-c0.05-cb2.0-r0.001-f0.9-g0", "geometry_current", **{
                    **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.001, final_scale=0.9),
                    "branch_boost": 1.0,
                    "coeff_lr_boost": 2.0,
                    "geometry_min_epochs": 0,
                }),
                spec(dataset, "GA-FU-FM-fast-b1.1-c0.05-cb1.5-r0.001-f0.9-g0", "geometry_current", **{
                    **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.001, final_scale=0.9),
                    "branch_boost": 1.1,
                    "coeff_lr_boost": 1.5,
                    "geometry_min_epochs": 0,
                }),
                spec(dataset, "GA-FU-FM-fast-b1.2-c0.05-cb1.5-r0.001-f0.8-g0", "geometry_current", **{
                    **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.001, final_scale=0.8),
                    "branch_boost": 1.2,
                    "coeff_lr_boost": 1.5,
                    "branch_max_active_frac": 0.35,
                    "geometry_min_epochs": 0,
                }),
                spec(dataset, "GA-FU-FM-fast-b1.0-c0.08-cb1.5-r0.001-f0.9-g0", "geometry_current", **{
                    **gafu_overrides(dataset, coeff_lr=0.08, rest_lr=0.001, final_scale=0.9),
                    "branch_boost": 1.0,
                    "coeff_lr_boost": 1.5,
                    "geometry_min_epochs": 0,
                }),
            ]
        )
    return out


def fashion_length_tune_specs(datasets: Sequence[str]) -> List[Spec]:
    out: List[Spec] = []
    for dataset in datasets:
        if dataset != "Fashion-MNIST":
            continue
        out.append(spec(dataset, "AdamW", "AdamW"))
        out.extend(
            [
                spec(dataset, "GA-FU-FM-e30-b1.2-c0.05-r0.001-f0.8-g0", "geometry_current", **{
                    **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.001, final_scale=0.8),
                    "branch_boost": 1.2,
                    "coeff_lr_boost": 1.0,
                    "branch_max_active_frac": 0.35,
                    "geometry_min_epochs": 0,
                }),
                spec(dataset, "GA-FU-FM-e30-b1.0-c0.05-cb2.0-r0.001-f0.9-g0", "geometry_current", **{
                    **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.001, final_scale=0.9),
                    "branch_boost": 1.0,
                    "coeff_lr_boost": 2.0,
                    "geometry_min_epochs": 0,
                }),
            ]
        )
    return out


def kmnist_next_tune_specs(datasets: Sequence[str]) -> List[Spec]:
    out: List[Spec] = []
    for dataset in datasets:
        if dataset != "KMNIST":
            continue
        out.extend(
            [
                spec(dataset, "GA-FU-KM-next-b1.1-c0.20-r0.003-f0.8-g0", "geometry_current", **{
                    **gafu_overrides(dataset, coeff_lr=0.20, rest_lr=0.003, final_scale=0.8),
                    "branch_boost": 1.1,
                    "coeff_lr_boost": 1.0,
                    "geometry_min_epochs": 0,
                }),
                spec(dataset, "GA-FU-KM-next-b1.2-c0.20-r0.003-f0.8-g0", "geometry_current", **{
                    **gafu_overrides(dataset, coeff_lr=0.20, rest_lr=0.003, final_scale=0.8),
                    "branch_boost": 1.2,
                    "coeff_lr_boost": 1.0,
                    "geometry_min_epochs": 0,
                }),
                spec(dataset, "GA-FU-KM-next-b1.2-c0.15-r0.003-f0.8-g0", "geometry_current", **{
                    **gafu_overrides(dataset, coeff_lr=0.15, rest_lr=0.003, final_scale=0.8),
                    "branch_boost": 1.2,
                    "coeff_lr_boost": 1.0,
                    "geometry_min_epochs": 0,
                }),
            ]
        )
    return out


def kmnist_fine_tune_specs(datasets: Sequence[str]) -> List[Spec]:
    out: List[Spec] = []
    for dataset in datasets:
        if dataset != "KMNIST":
            continue
        for coeff_lr in (0.12, 0.14, 0.16):
            label = f"GA-FU-KM-fine-b1.2-c{coeff_lr:.2f}-r0.003-f0.8-g0"
            out.append(spec(dataset, label, "geometry_current", **{
                **gafu_overrides(dataset, coeff_lr=coeff_lr, rest_lr=0.003, final_scale=0.8),
                "branch_boost": 1.2,
                "coeff_lr_boost": 1.0,
                "geometry_min_epochs": 0,
            }))
    return out


def fashion_scheduler_tune_specs(datasets: Sequence[str]) -> List[Spec]:
    out: List[Spec] = []
    for dataset in datasets:
        if dataset != "Fashion-MNIST":
            continue
        out.append(spec(dataset, "AdamW", "AdamW"))
        base = {
            **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.001, final_scale=0.8),
            "branch_boost": 1.2,
            "coeff_lr_boost": 1.0,
            "branch_max_active_frac": 0.35,
            "geometry_min_epochs": 0,
        }
        out.extend(
            [
                spec(dataset, "GA-FU-FM-sched-b1.2-c0.05-r0.001-f0.8-restcos0.3", "geometry_current", **{
                    **base,
                    "rest_lr_schedule": "cosine",
                    "rest_lr_final_mult": 0.3,
                }),
                spec(dataset, "GA-FU-FM-sched-b1.2-c0.05-r0.001-f0.8-coeffcos0.5", "geometry_current", **{
                    **base,
                    "coeff_lr_schedule": "cosine",
                    "coeff_lr_decay_final_mult": 0.5,
                }),
                spec(dataset, "GA-FU-FM-sched-b1.2-c0.05-r0.001-f0.8-bothcos", "geometry_current", **{
                    **base,
                    "rest_lr_schedule": "cosine",
                    "rest_lr_final_mult": 0.3,
                    "coeff_lr_schedule": "cosine",
                    "coeff_lr_decay_final_mult": 0.5,
                }),
                spec(dataset, "GA-FU-FM-sched-b1.2-c0.05-r0.001-f0.8-restlin0.3", "geometry_current", **{
                    **base,
                    "rest_lr_schedule": "linear",
                    "rest_lr_final_mult": 0.3,
                }),
            ]
        )
    return out


def fashion_scheduler_confirm_specs(datasets: Sequence[str]) -> List[Spec]:
    out: List[Spec] = []
    for dataset in datasets:
        if dataset != "Fashion-MNIST":
            continue
        out.append(spec(dataset, "AdamW", "AdamW"))
        base = {
            **gafu_overrides(dataset, coeff_lr=0.05, rest_lr=0.001, final_scale=0.8),
            "branch_boost": 1.2,
            "coeff_lr_boost": 1.0,
            "branch_max_active_frac": 0.35,
            "geometry_min_epochs": 0,
        }
        out.extend(
            [
                spec(dataset, "GA-FU-FM-sched-b1.2-c0.05-r0.001-f0.8-restcos0.3", "geometry_current", **{
                    **base,
                    "rest_lr_schedule": "cosine",
                    "rest_lr_final_mult": 0.3,
                }),
                spec(dataset, "GA-FU-FM-sched-b1.2-c0.05-r0.001-f0.8-bothcos", "geometry_current", **{
                    **base,
                    "rest_lr_schedule": "cosine",
                    "rest_lr_final_mult": 0.3,
                    "coeff_lr_schedule": "cosine",
                    "coeff_lr_decay_final_mult": 0.5,
                }),
            ]
        )
    return out


def package_specs(args: argparse.Namespace, package: str) -> Tuple[List[Spec], List[int], Dict[str, Any]]:
    key = package.strip().upper()
    datasets = [dataset_name(x) for x in parse_str_list(args.datasets)]
    methods = [method_name(x) for x in parse_str_list(args.methods)]
    meta: Dict[str, Any] = {}
    if key == "P0":
        meta = {"train_size": 512, "val_size": 128, "test_size": 128, "epochs": 1}
        return specs_for_methods(datasets, ["AdamW", "StaticFunctional", "GA-FU"]), [0], meta
    if key in {"P1", "P2", "P3"}:
        return specs_for_methods(datasets, methods), parse_int_list(args.seeds), meta
    if key in {"TUNE", "P1-TUNE", "TUNING"}:
        return tune_specs(datasets), parse_int_list(args.seeds), meta
    if key in {"AUCTUNE", "AUC-TUNE", "P1-AUCTUNE"}:
        return auc_tune_specs(datasets), parse_int_list(args.seeds), meta
    if key in {"CAPTUNE", "CAP-TUNE", "CAPACITY-TUNE"}:
        return capacity_tune_specs(datasets), parse_int_list(args.seeds), meta
    if key in {"FASTTUNE", "FAST-TUNE", "FASHION-FAST"}:
        return fashion_fast_tune_specs(datasets), parse_int_list(args.seeds), meta
    if key in {"FASHION30", "FASHION-LENGTH"}:
        return fashion_length_tune_specs(datasets), parse_int_list(args.seeds), meta
    if key in {"KMNEXT", "KMNIST-NEXT"}:
        return kmnist_next_tune_specs(datasets), parse_int_list(args.seeds), meta
    if key in {"KMFINE", "KMNIST-FINE"}:
        return kmnist_fine_tune_specs(datasets), parse_int_list(args.seeds), meta
    if key in {"FASHSCHED", "FASHION-SCHED", "FASHION-SCHEDULE"}:
        return fashion_scheduler_tune_specs(datasets), parse_int_list(args.seeds), meta
    if key in {"FASHCONFIRM", "FASHION-SCHED-CONFIRM"}:
        return fashion_scheduler_confirm_specs(datasets), parse_int_list(args.seeds), meta
    raise ValueError(f"unknown package {package!r}")


def _group_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (row.get("dataset"), row.get("epochs"), row.get("label_noise", 0.0), row.get("train_size"))


def add_gafu_comparisons(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compared = [dict(row) for row in rows]
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in compared:
        groups.setdefault(_group_key(row), []).append(row)
    for group in groups.values():
        adam = [row for row in group if str(row.get("method")) == "AdamW" and not row.get("error")]
        if not adam:
            continue
        adam_time = float(np.mean([float(row["step_time_ms"]) for row in adam]))
        adam_ece = float(np.mean([float(row["ece"]) for row in adam]))
        adam_acc = float(np.mean([float(row["test_acc"]) for row in adam]))
        adam_auc = float(np.mean([float(row["val_auc"]) for row in adam]))
        adam_phi = float(np.mean([float(row["phi_prime_p95"]) for row in adam]))
        adam_j = float(np.mean([float(row["max_jac_condition"]) for row in adam]))
        adam_branch = float(np.mean([float(row["branch_output_norm_ratio"]) for row in adam]))
        for row in group:
            if row.get("error"):
                continue
            row["acc_gap_vs_adamw"] = adam_acc - float(row["test_acc"])
            row["val_auc_improvement_vs_adamw"] = (adam_auc - float(row["val_auc"])) / max(1e-9, adam_auc)
            row["phi_prime_reduction_vs_adamw"] = (adam_phi - float(row["phi_prime_p95"])) / max(1e-9, adam_phi)
            row["jac_reduction_vs_adamw"] = (adam_j - float(row["max_jac_condition"])) / max(1e-9, adam_j)
            row["branch_over_adamw"] = float(row["branch_output_norm_ratio"]) / max(1e-9, adam_branch)
            row["time_ratio_vs_adamw"] = float(row["step_time_ms"]) / max(1e-9, adam_time)
            row["ece_reduction_vs_adamw"] = (adam_ece - float(row["ece"])) / max(1e-9, adam_ece)
    for row in compared:
        row["gafu_failure_type"] = classify_gafu_failure(row)
    return compared


def classify_gafu_failure(row: Dict[str, Any]) -> str:
    if row.get("error"):
        return "run_failed"
    try:
        gap = float(row.get("acc_gap_vs_adamw", 0.0))
        auc = float(row.get("val_auc_improvement_vs_adamw", 0.0))
        phi = float(row.get("phi_prime_reduction_vs_adamw", 0.0))
        jac = float(row.get("jac_reduction_vs_adamw", 0.0))
        branch = float(row.get("branch_over_adamw", 1.0))
        ece = float(row.get("ece_reduction_vs_adamw", 0.0))
    except (TypeError, ValueError):
        return ""
    method = str(row.get("method", "")).lower()
    if method == "adamw":
        return ""
    if branch < 0.45 and gap > 0.005:
        return "branch_underactive"
    if branch > 1.10:
        return "branch_overactive"
    if gap > 0.015:
        return "accuracy_gap_too_large"
    if auc <= 0:
        return "no_convergence_gain"
    if phi < 0.20 or jac < 0.10:
        return "geometry_not_preserved"
    if ece < 0:
        return "calibration_worse"
    return ""


def pass_level(row: Dict[str, Any], dataset: str) -> str:
    gap = float(row.get("acc_gap_vs_adamw_mean", math.nan))
    auc = float(row.get("val_auc_improvement_vs_adamw_mean", math.nan))
    phi = float(row.get("phi_prime_reduction_vs_adamw_mean", math.nan))
    jac = float(row.get("jac_reduction_vs_adamw_mean", math.nan))
    branch = float(row.get("branch_over_adamw_mean", math.nan))
    ece = float(row.get("ece_reduction_vs_adamw_mean", math.nan))
    if not all(math.isfinite(x) for x in [gap, auc, phi, jac, branch]):
        return "none"
    weak = gap < 0.015 and auc > 0.05 and phi > 0.20 and jac > 0.10 and 0.45 < branch < 0.95
    medium_acc = gap <= 0.0 if dataset == "Fashion-MNIST" else gap < 0.010
    medium = medium_acc and auc > 0.08 and phi > 0.25 and jac > 0.15
    strong = gap < 0.005 and auc > 0.10 and phi > 0.25 and jac > 0.20 and ece > 0.10
    if strong:
        return "strong"
    if medium:
        return "medium"
    if weak:
        return "weak"
    return "none"


def decision_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    compared = add_gafu_comparisons(rows)
    summary = summarize_runs(compared, ["dataset", "method", "label_noise", "train_size", "epochs"])
    levels: Dict[str, Dict[str, str]] = {}
    for item in summary:
        method = str(item.get("method"))
        if "GA-FU" not in method:
            continue
        dataset = str(item.get("dataset"))
        levels.setdefault(method, {})[dataset] = pass_level(item, dataset)

    def rank(level: str) -> int:
        return {"none": 0, "weak": 1, "medium": 2, "strong": 3}.get(level, 0)

    default_levels = levels.get("GA-FU", {})
    default_min = min((rank(default_levels.get(ds, "none")) for ds in ["Fashion-MNIST", "KMNIST"]), default=0)
    if default_min >= 2:
        recommendation = "Stage I clean default = GA-FU"
    elif default_min >= 1:
        recommendation = "GA-FU = geometry-stable Pareto optimizer; AdamW remains accuracy baseline"
    else:
        recommendation = "GA-FU did not pass P1 gate; tune fixed control/capacity before new optimizer components"

    return {
        "pass_levels": levels,
        "GA_FU_weak_pass": default_min >= 1,
        "GA_FU_medium_pass": default_min >= 2,
        "GA_FU_strong_pass": default_min >= 3,
        "recommended_stage_i_default": recommendation,
    }


def add_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packages", default="P0")
    p.add_argument("--datasets", default="Fashion-MNIST,KMNIST")
    p.add_argument("--methods", default="AdamW,StaticFunctional,GA-FU")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--out-dir", type=Path, default=Path("results/gafu_consolidation"))
    p.add_argument("--fresh", action="store_true")
    add_common_train_args(p)
    return p


def run_specs(args: argparse.Namespace, specs: List[Spec], seeds: List[int], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    run_dir = ensure_dir(out_dir / "runs")
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "gafu_runs.csv")
    new_rows: List[Dict[str, Any]] = []
    total = len(specs) * len(seeds)
    done = 0
    for item in specs:
        for seed in seeds:
            done += 1
            dataset = item["dataset"]
            label = item["label"]
            base_method = item["base_method"]
            overrides = dict(item.get("overrides", {}))
            overrides.update(meta)
            try:
                cfg = config_from_args(args, dataset, base_method, seed, **overrides)
                cfg.method = label
                cfg.notes = f"gafu_consolidation base_method={base_method}"
                row = train_one(cfg)
                row["stage"] = "GA-FU"
                row["base_method"] = base_method
                print(
                    f"[{done}/{total}] {dataset} {label} seed={seed} "
                    f"acc={row['test_acc']:.4f} val_auc={row['val_auc']:.4f} "
                    f"branch={row['branch_output_norm_ratio']:.3f} "
                    f"phi={row['phi_prime_p95']:.4f} J={row['max_jac_condition']:.2f} "
                    f"ece={row['ece']:.4f} switch={row['branch_switch_reason']}"
                )
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                row = {
                    "stage": "GA-FU",
                    "dataset": dataset,
                    "method": label,
                    "base_method": base_method,
                    "seed": seed,
                    "error": repr(exc),
                    "test_acc": math.nan,
                }
                print(f"[{done}/{total}] ERROR {dataset} {label} seed={seed}: {exc!r}")

            rows.append(row)
            new_rows.append(row)
            safe_name = f"{dataset}_{label}_seed{seed}_e{row.get('epochs', meta.get('epochs', args.epochs))}"
            safe_name = safe_name.replace("/", "_").replace(" ", "_")
            save_json(run_dir / f"{safe_name}.json", row)
            compared = add_gafu_comparisons(rows)
            write_csv(out_dir / "gafu_runs.csv", compared)

    compared = add_gafu_comparisons(rows)
    summary_method = summarize_runs(compared, ["dataset", "method", "label_noise", "train_size", "epochs"])
    summary_dataset = summarize_runs(compared, ["dataset", "label_noise", "train_size", "epochs"])
    failures = summarize_runs(compared, ["dataset", "method", "gafu_failure_type"])
    failure_table = [row for row in compared if row.get("gafu_failure_type")]
    report = decision_report(compared)
    write_csv(out_dir / "gafu_summary_by_method.csv", summary_method)
    write_csv(out_dir / "gafu_summary_by_dataset.csv", summary_dataset)
    write_csv(out_dir / "gafu_failure_summary.csv", failures)
    write_csv(out_dir / "gafu_failure_table.csv", failure_table)
    save_json(
        out_dir / "aggregate_summary.json",
        {
            "new_runs": len(new_rows),
            "total_rows": len(compared),
            "summary_by_method": summary_method,
            "summary_by_dataset": summary_dataset,
            "failure_summary": failures,
            "decision_report": report,
        },
    )
    print(f"Wrote {len(new_rows)} new rows to {out_dir}")
    print(f"GA-FU weak/medium/strong: {report['GA_FU_weak_pass']}/{report['GA_FU_medium_pass']}/{report['GA_FU_strong_pass']}")
    print(report["recommended_stage_i_default"])
    return compared


def main() -> int:
    args = add_args().parse_args()
    all_rows: List[Dict[str, Any]] = []
    packages = parse_str_list(args.packages)
    first = True
    original_fresh = args.fresh
    for package in packages:
        specs, seeds, meta = package_specs(args, package)
        args.fresh = original_fresh and first
        all_rows = run_specs(args, specs, seeds, meta)
        first = False
    return 0 if all_rows is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
