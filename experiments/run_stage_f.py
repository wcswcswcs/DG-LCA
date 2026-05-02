#!/usr/bin/env python3
"""Recovered Stage F runner.

This script restores the experiment surface described in docs/log.md:
functional-update optimization advantage, branch activation schedules, online
geometry triggers, and the F6 KMNIST threshold confirmation.
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dgkan_core import (
    add_common_train_args,
    add_failure_types,
    config_from_args,
    ensure_dir,
    parse_float_list,
    parse_int_list,
    parse_str_list,
    read_csv,
    save_json,
    summarize_runs,
    train_one,
    write_csv,
)


Spec = Dict[str, Any]


def spec(dataset: str, label: str, base_method: str, **overrides: Any) -> Spec:
    return {
        "dataset": dataset,
        "label": label,
        "base_method": base_method,
        "overrides": overrides,
    }


def phase_f0(datasets: List[str], methods: List[str]) -> List[Spec]:
    labels = methods or ["AdamW", "static_top", "FirstOrder", "IdentityAdj", "geometry_current"]
    return [spec(dataset, method, method) for dataset in datasets for method in labels]


def phase_f4_schedule() -> List[Spec]:
    return [
        spec("Fashion-MNIST", "AdamW", "AdamW"),
        spec("Fashion-MNIST", "loss_aware_b2_c1.5_m0.50_f0.8", "static_top",
             branch_schedule="loss_aware", branch_boost=2.0, coeff_lr_boost=1.5,
             branch_max_active_frac=0.50, branch_final_scale=0.8),
        spec("Fashion-MNIST", "early_active_cos_b2_c1.5_a0.35_f0.8", "static_top",
             branch_schedule="early_active_cosine", branch_boost=2.0, coeff_lr_boost=1.5,
             branch_active_frac=0.35, branch_final_scale=0.8),
        spec("KMNIST", "AdamW", "AdamW"),
        spec("KMNIST", "loss_aware_mild_b1.5_c1.2_m0.35_f1.0", "static_top",
             branch_schedule="loss_aware", branch_boost=1.5, coeff_lr_boost=1.2,
             branch_max_active_frac=0.35, branch_final_scale=1.0),
        spec("KMNIST", "loss_aware_aggr_b2_c2_m0.50_f0.8", "static_top",
             branch_schedule="loss_aware", branch_boost=2.0, coeff_lr_boost=2.0,
             branch_max_active_frac=0.50, branch_final_scale=0.8),
    ]


def phase_f4_geometry() -> List[Spec]:
    return [
        spec("Fashion-MNIST", "AdamW", "AdamW"),
        spec("Fashion-MNIST", "geometry_aware_b2_c1.5_m0.50_f0.8", "static_top",
             branch_schedule="geometry_aware", branch_boost=2.0, coeff_lr_boost=1.5,
             branch_max_active_frac=0.50, branch_final_scale=0.8,
             geometry_phi_high=0.052, geometry_jac_high=4.5, geometry_branch_high=0.30),
        spec("KMNIST", "AdamW", "AdamW"),
        spec("KMNIST", "geometry_aware_b1.5_c1.2_m0.35_f1.0", "static_top",
             branch_schedule="geometry_aware", branch_boost=1.5, coeff_lr_boost=1.2,
             branch_max_active_frac=0.35, branch_final_scale=1.0,
             geometry_phi_high=0.055, geometry_jac_high=4.5, geometry_branch_high=0.30),
        spec("KMNIST", "geometry_aware_b1.5_c1.2_m0.35_f0.8", "static_top",
             branch_schedule="geometry_aware", branch_boost=1.5, coeff_lr_boost=1.2,
             branch_max_active_frac=0.35, branch_final_scale=0.8,
             geometry_phi_high=0.055, geometry_jac_high=4.5, geometry_branch_high=0.30),
    ]


def phase_f4_geometry_threshold_sweep(jac_highs: List[float], final_scales: List[float]) -> List[Spec]:
    specs = [spec("KMNIST", "AdamW", "AdamW")]
    for jac_high, final_scale in itertools.product(jac_highs, final_scales):
        label = f"geometry_aware_j{jac_high:g}_f{final_scale:g}"
        specs.append(
            spec("KMNIST", label, "static_top",
                 branch_schedule="geometry_aware", branch_boost=1.5, coeff_lr_boost=1.2,
                 branch_max_active_frac=0.35, branch_final_scale=final_scale,
                 geometry_phi_high=0.055, geometry_jac_high=jac_high, geometry_branch_high=0.30)
        )
    return specs


def phase_f4_geometry_threshold_confirm() -> List[Spec]:
    return [
        spec("KMNIST", "AdamW", "AdamW"),
        spec("KMNIST", "loss_aware_mild_b1.5_c1.2_m0.35_f1.0", "static_top",
             branch_schedule="loss_aware", branch_boost=1.5, coeff_lr_boost=1.2,
             branch_max_active_frac=0.35, branch_final_scale=1.0),
        spec("KMNIST", "geometry_aware_j4.5_f0.9", "static_top",
             branch_schedule="geometry_aware", branch_boost=1.5, coeff_lr_boost=1.2,
             branch_max_active_frac=0.35, branch_final_scale=0.9,
             geometry_phi_high=0.055, geometry_jac_high=4.5, geometry_branch_high=0.30),
    ]


def specs_for_phase(args: argparse.Namespace, phase: str) -> List[Spec]:
    phase_key = phase.strip().lower()
    datasets = parse_str_list(args.datasets)
    methods = parse_str_list(args.methods) if args.methods else []
    if phase_key in {"f0", "stage_f0"}:
        return phase_f0(datasets, methods)
    if phase_key in {"f4", "f4_schedule", "schedule"}:
        return phase_f4_schedule()
    if phase_key in {"f5", "f4_geometry", "geometry"}:
        return phase_f4_geometry()
    if phase_key in {"f6", "f4_geometry_threshold_confirm", "threshold_confirm"}:
        return phase_f4_geometry_threshold_confirm()
    if phase_key in {"f4_geometry_threshold_sweep", "threshold_sweep"}:
        return phase_f4_geometry_threshold_sweep(parse_float_list(args.jac_highs), parse_float_list(args.final_scales))
    if phase_key == "all":
        return (
            phase_f0(datasets, methods)
            + phase_f4_schedule()
            + phase_f4_geometry()
            + phase_f4_geometry_threshold_confirm()
        )
    raise ValueError(f"unknown phase {phase!r}")


def filter_specs_by_dataset(specs: Iterable[Spec], datasets: List[str]) -> List[Spec]:
    wanted = {name.lower().replace("_", "-") for name in datasets}
    if not wanted:
        return list(specs)
    out: List[Spec] = []
    for item in specs:
        dataset_key = str(item["dataset"]).lower().replace("_", "-")
        if dataset_key in wanted:
            out.append(item)
    return out


def add_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phases", default="F4_geometry_threshold_confirm")
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--methods", default="")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--label-noise", type=float, default=0.0)
    p.add_argument("--jac-highs", default="3.5,4.5,5.5")
    p.add_argument("--final-scales", default="0.8,0.9,1.0")
    p.add_argument("--out-dir", type=Path, default=Path("results/stage_f_recovered"))
    p.add_argument("--fresh", action="store_true")
    add_common_train_args(p)
    return p


def run_specs(args: argparse.Namespace, specs: List[Spec], seeds: List[int]) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    run_dir = ensure_dir(out_dir / "runs")
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "stage_f_runs.csv")
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
            overrides.setdefault("label_noise", args.label_noise)
            try:
                cfg = config_from_args(args, dataset, base_method, seed, **overrides)
                cfg.method = label
                cfg.notes = f"stage_f_recovered base_method={base_method}"
                row = train_one(cfg)
                row["stage"] = "F"
                row["base_method"] = base_method
                print(
                    f"[{done}/{total}] {dataset} {label} seed={seed} "
                    f"acc={row['test_acc']:.4f} val_auc={row['val_auc']:.4f} "
                    f"branch={row['branch_output_norm_ratio']:.3f} "
                    f"phi={row['phi_prime_p95']:.4f} J={row['max_jac_condition']:.2f} "
                    f"switch={row['branch_switch_reason']}"
                )
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                row = {
                    "stage": "F",
                    "dataset": dataset,
                    "method": label,
                    "base_method": base_method,
                    "seed": seed,
                    "label_noise": args.label_noise,
                    "error": repr(exc),
                    "test_acc": math.nan,
                }
                print(f"[{done}/{total}] ERROR {dataset} {label} seed={seed}: {exc!r}")

            rows.append(row)
            new_rows.append(row)
            safe_name = f"{dataset}_{label}_seed{seed}".replace("/", "_").replace(" ", "_")
            save_json(run_dir / f"{safe_name}.json", row)
            compared = add_failure_types(rows)
            write_csv(out_dir / "stage_f_runs.csv", compared)

    compared = add_failure_types(rows)
    summary_method = summarize_runs(compared, ["dataset", "method", "label_noise"])
    summary_failure = summarize_runs(compared, ["dataset", "failure_type"])
    write_csv(out_dir / "stage_f_summary_by_method.csv", summary_method)
    write_csv(out_dir / "stage_f_failure_summary.csv", summary_failure)
    save_json(
        out_dir / "stage_f_summary.json",
        {
            "new_runs": len(new_rows),
            "total_rows": len(compared),
            "summary_by_method": summary_method,
            "failure_summary": summary_failure,
        },
    )
    print(f"Wrote {len(new_rows)} new rows to {out_dir}")
    return compared


def main() -> int:
    args = add_args().parse_args()
    phases = parse_str_list(args.phases)
    specs: List[Spec] = []
    for phase in phases:
        specs.extend(specs_for_phase(args, phase))
    specs = filter_specs_by_dataset(specs, parse_str_list(args.datasets))
    seeds = parse_int_list(args.seeds)
    run_specs(args, specs, seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
