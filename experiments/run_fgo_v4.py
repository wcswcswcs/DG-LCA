#!/usr/bin/env python3
"""Recovered FGO-v4 mini experiment runner.

The v4 document is a next-step plan, not a completed log. This runner encodes
that plan as executable packages:

* V4-0: reproduce current clean baselines.
* V4-mini-clean: KMNIST data-pulse / re-anchor search.
* V4-mini-noise: Fashion noisy-label SNR-lite cost search.
* V4-mini-stress: small-data stress checks for the leading clean/noise ideas.
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Any, Dict, List

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


def phase_v4_0(datasets: List[str]) -> List[Spec]:
    methods = ["AdamW", "geometry_current", "no_grid", "fgo_v3_clean"]
    return [spec(dataset, method, method) for dataset in datasets for method in methods]


def phase_v4_clean(args: argparse.Namespace) -> List[Spec]:
    out: List[Spec] = [
        spec("KMNIST", "AdamW", "AdamW"),
        spec("KMNIST", "geometry_current", "geometry_current"),
        spec("KMNIST", "no_grid", "no_grid"),
        spec("KMNIST", "fgo_v3_clean", "fgo_v3_clean"),
    ]
    combos = itertools.product(
        parse_float_list(args.lambda_earlies),
        parse_float_list(args.lambda_finals),
        parse_float_list(args.pulse_fracs),
        parse_float_list(args.momentum_mus),
        parse_float_list(args.prox_lambdas),
    )
    for lambda_early, lambda_final, pulse_frac, momentum_mu, prox_lambda in combos:
        label = (
            "fgo_v4_clean"
            f"_le{lambda_early:g}_lf{lambda_final:g}"
            f"_p{pulse_frac:g}_m{momentum_mu:g}_prox{prox_lambda:g}"
        )
        out.append(
            spec("KMNIST", label, "fgo_v4_clean",
                 lambda_early=lambda_early, lambda_final=lambda_final,
                 pulse_frac=pulse_frac, momentum_mu=momentum_mu, prox_lambda=prox_lambda,
                 trust_radius=args.v4_trust_radius)
        )
    return out


def phase_v4_noise(args: argparse.Namespace) -> List[Spec]:
    out: List[Spec] = []
    for noise in parse_float_list(args.noise_levels):
        out.extend(
            [
                spec("Fashion-MNIST", f"AdamW_noise{noise:g}", "AdamW", label_noise=noise),
                spec("Fashion-MNIST", f"geometry_current_noise{noise:g}", "geometry_current", label_noise=noise),
                spec("Fashion-MNIST", f"snr_previous_best_noise{noise:g}", "snr_previous_best", label_noise=noise),
            ]
        )
        for grouping, interval in itertools.product(["layer", "output_channel", "edge"], parse_int_list(args.snr_intervals)):
            label = f"snr_lite_{grouping}_i{interval}_noise{noise:g}"
            base = "snr_lite_layer" if grouping == "layer" else "snr_lite_output"
            out.append(
                spec("Fashion-MNIST", label, base,
                     label_noise=noise, snr_grouping=grouping,
                     snr_update_interval=interval, snr_sample_frac=args.snr_sample_frac)
            )
    return out


def phase_v4_stress(args: argparse.Namespace) -> List[Spec]:
    small_train = max(200, int(args.train_size * args.small_data_frac))
    out: List[Spec] = []
    for dataset in ["Fashion-MNIST", "KMNIST"]:
        out.extend(
            [
                spec(dataset, "AdamW_small", "AdamW", train_size=small_train),
                spec(dataset, "geometry_current_small", "geometry_current", train_size=small_train),
                spec(dataset, "fgo_v4_clean_best_small", "fgo_v4_clean",
                     train_size=small_train, lambda_early=1.0, lambda_final=0.05,
                     pulse_frac=0.25, momentum_mu=0.8, prox_lambda=3e-5),
            ]
        )
    for noise in parse_float_list(args.noise_levels):
        out.extend(
            [
                spec("Fashion-MNIST", f"geometry_current_small_noise{noise:g}", "geometry_current",
                     train_size=small_train, label_noise=noise),
                spec("Fashion-MNIST", f"snr_lite_layer_small_noise{noise:g}", "snr_lite_layer",
                     train_size=small_train, label_noise=noise,
                     snr_update_interval=4, snr_sample_frac=args.snr_sample_frac),
            ]
        )
    return out


def specs_for_phase(args: argparse.Namespace, phase: str) -> List[Spec]:
    key = phase.strip().lower()
    if key in {"v4-0", "v4_0", "baseline"}:
        return phase_v4_0(parse_str_list(args.datasets))
    if key in {"v4-mini-clean", "v4_clean", "clean"}:
        return phase_v4_clean(args)
    if key in {"v4-mini-noise", "v4_noise", "noise"}:
        return phase_v4_noise(args)
    if key in {"v4-mini-stress", "v4_stress", "stress"}:
        return phase_v4_stress(args)
    if key == "all":
        return (
            phase_v4_0(parse_str_list(args.datasets))
            + phase_v4_clean(args)
            + phase_v4_noise(args)
            + phase_v4_stress(args)
        )
    raise ValueError(f"unknown phase {phase!r}")


def mean_for(rows: List[Dict[str, Any]], dataset: str, method_contains: str, metric: str) -> float:
    vals = []
    needle = method_contains.lower()
    for row in rows:
        if str(row.get("dataset")) != dataset:
            continue
        if needle not in str(row.get("method", "")).lower():
            continue
        try:
            value = float(row.get(metric, math.nan))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            vals.append(value)
    if not vals:
        return math.nan
    return sum(vals) / len(vals)


def decision_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    compared = add_failure_types(rows)
    clean = {
        "kmnist_acc_gap_vs_adamw": mean_for(compared, "KMNIST", "fgo_v4_clean", "acc_gap_vs_adamw"),
        "kmnist_auc_improvement_vs_adamw": mean_for(compared, "KMNIST", "fgo_v4_clean", "val_auc_improvement_vs_adamw"),
        "kmnist_phi_reduction_vs_adamw": mean_for(compared, "KMNIST", "fgo_v4_clean", "phi_prime_reduction_vs_adamw"),
        "kmnist_jac_reduction_vs_adamw": mean_for(compared, "KMNIST", "fgo_v4_clean", "jac_reduction_vs_adamw"),
        "fashion_gain_vs_geometry": mean_for(compared, "Fashion-MNIST", "fgo_v4_clean", "gain_vs_geometry"),
    }
    clean["passes_default_gate"] = (
        math.isfinite(clean["kmnist_acc_gap_vs_adamw"])
        and clean["kmnist_acc_gap_vs_adamw"] < 0.005
        and clean["kmnist_auc_improvement_vs_adamw"] > 0.10
        and clean["kmnist_phi_reduction_vs_adamw"] > 0.20
        and clean["kmnist_jac_reduction_vs_adamw"] > 0.10
        and (math.isnan(clean["fashion_gain_vs_geometry"]) or clean["fashion_gain_vs_geometry"] >= -0.003)
    )

    noise = {
        "fashion_gain_vs_geometry": mean_for(compared, "Fashion-MNIST", "snr_lite", "gain_vs_geometry"),
        "fashion_time_ratio_vs_geometry": mean_for(compared, "Fashion-MNIST", "snr_lite", "time_ratio_vs_geometry"),
        "fashion_ece_reduction_vs_geometry": mean_for(compared, "Fashion-MNIST", "snr_lite", "ece_reduction_vs_geometry"),
    }
    noise["passes_default_gate"] = (
        math.isfinite(noise["fashion_gain_vs_geometry"])
        and noise["fashion_gain_vs_geometry"] > 0.01
        and noise["fashion_time_ratio_vs_geometry"] < 1.4
        and noise["fashion_ece_reduction_vs_geometry"] > 0.25
    )
    return {"clean_default": clean, "noise_default": noise}


def add_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phases", default="V4-mini-clean")
    p.add_argument("--datasets", default="Fashion-MNIST,KMNIST")
    p.add_argument("--seeds", default="0")
    p.add_argument("--out-dir", type=Path, default=Path("results/fgo_v4_recovered"))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--limit-configs", type=int, default=0)

    p.add_argument("--lambda-earlies", default="0.5,1.0")
    p.add_argument("--lambda-finals", default="0,0.05")
    p.add_argument("--pulse-fracs", default="0.15,0.25")
    p.add_argument("--momentum-mus", default="0,0.8")
    p.add_argument("--prox-lambdas", default="0,3e-5")
    p.add_argument("--v4-trust-radius", type=float, default=0.20)
    p.add_argument("--noise-levels", default="0.2,0.4")
    p.add_argument("--snr-intervals", default="4,8")
    p.add_argument("--snr-sample-frac", type=float, default=0.5)
    p.add_argument("--small-data-frac", type=float, default=0.25)
    add_common_train_args(p)
    return p


def main() -> int:
    args = add_args().parse_args()
    phases = parse_str_list(args.phases)
    specs: List[Spec] = []
    for phase in phases:
        specs.extend(specs_for_phase(args, phase))
    if args.limit_configs > 0:
        specs = specs[: args.limit_configs]

    seeds = parse_int_list(args.seeds)
    out_dir = ensure_dir(args.out_dir)
    run_dir = ensure_dir(out_dir / "runs")
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "fgo_v4_runs.csv")
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
            try:
                cfg = config_from_args(args, dataset, base_method, seed, **overrides)
                cfg.method = label
                cfg.notes = f"fgo_v4_recovered base_method={base_method}"
                row = train_one(cfg)
                row["stage"] = "FGO-v4"
                row["base_method"] = base_method
                print(
                    f"[{done}/{total}] {dataset} {label} seed={seed} "
                    f"acc={row['test_acc']:.4f} val_auc={row['val_auc']:.4f} "
                    f"branch={row['branch_output_norm_ratio']:.3f} "
                    f"phi={row['phi_prime_p95']:.4f} J={row['max_jac_condition']:.2f} "
                    f"time={row['step_time_ms']:.1f}ms"
                )
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                row = {
                    "stage": "FGO-v4",
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
            safe_name = f"{dataset}_{label}_seed{seed}".replace("/", "_").replace(" ", "_")
            save_json(run_dir / f"{safe_name}.json", row)
            write_csv(out_dir / "fgo_v4_runs.csv", add_failure_types(rows))

    compared = add_failure_types(rows)
    summary = summarize_runs(compared, ["dataset", "method", "label_noise"])
    failures = summarize_runs(compared, ["dataset", "failure_type"])
    report = decision_report(compared)
    write_csv(out_dir / "fgo_v4_summary_by_method.csv", summary)
    write_csv(out_dir / "fgo_v4_failure_summary.csv", failures)
    save_json(
        out_dir / "fgo_v4_summary.json",
        {
            "new_runs": len(new_rows),
            "total_rows": len(compared),
            "summary_by_method": summary,
            "failure_summary": failures,
            "decision_report": report,
        },
    )
    save_json(out_dir / "fgo_v4_decision_report.json", report)
    print(f"Wrote {len(new_rows)} new rows to {out_dir}")
    print(f"Clean gate pass: {report['clean_default']['passes_default_gate']}")
    print(f"Noise gate pass: {report['noise_default']['passes_default_gate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
