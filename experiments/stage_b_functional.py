#!/usr/bin/env python3
"""Recovered Stage B functional-update runner.

Stage B isolates the hybrid update used throughout the later docs: KAN edge
coefficients receive a functional-space preconditioned step, while the stem,
normalization, alpha, and head parameters keep ordinary AdamW updates.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from dgkan_core import (
    add_common_train_args,
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


def optional_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def add_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--methods", default="AdamW,static_top,geometry_current,no_grid")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--label-noise", type=float, default=0.0)
    p.add_argument("--out-dir", type=Path, default=Path("results/stage_b_recovered"))
    p.add_argument("--fresh", action="store_true")

    p.add_argument("--metric-mode", choices=["grid", "fixed_data", "no_grid", "pulse"], default=None)
    p.add_argument(
        "--branch-schedule",
        choices=["none", "early_active_linear", "early_active_cosine", "loss_aware", "geometry_aware"],
        default=None,
    )
    p.add_argument("--warmup-frac", type=optional_float, default=None)
    p.add_argument("--sobolev-alpha", type=optional_float, default=None)
    p.add_argument("--sobolev-beta", type=optional_float, default=None)
    p.add_argument("--rho", type=optional_float, default=None)
    p.add_argument("--branch-boost", type=optional_float, default=None)
    p.add_argument("--coeff-lr-boost", type=optional_float, default=None)
    p.add_argument("--coeff-lr-final-mult", type=optional_float, default=None)
    p.add_argument("--branch-final-scale", type=optional_float, default=None)
    p.add_argument("--branch-active-frac", type=optional_float, default=None)
    p.add_argument("--branch-max-active-frac", type=optional_float, default=None)
    p.add_argument("--lambda-early", type=optional_float, default=None)
    p.add_argument("--lambda-final", type=optional_float, default=None)
    p.add_argument("--pulse-frac", type=optional_float, default=None)
    p.add_argument("--anchor-decay", choices=["linear", "cosine"], default=None)
    p.add_argument("--momentum-mu", type=optional_float, default=None)
    p.add_argument("--trust-radius", type=optional_float, default=None)
    p.add_argument("--prox-lambda", type=optional_float, default=None)
    p.add_argument("--credit-mode", choices=["analytic", "first_order", "identity"], default=None)
    add_common_train_args(p)
    return p


def override_dict(args: argparse.Namespace) -> Dict[str, Any]:
    pairs = {
        "metric_mode": args.metric_mode,
        "branch_schedule": args.branch_schedule,
        "warmup_frac": args.warmup_frac,
        "sobolev_alpha": args.sobolev_alpha,
        "sobolev_beta": args.sobolev_beta,
        "rho": args.rho,
        "branch_boost": args.branch_boost,
        "coeff_lr_boost": args.coeff_lr_boost,
        "coeff_lr_final_mult": args.coeff_lr_final_mult,
        "branch_final_scale": args.branch_final_scale,
        "branch_active_frac": args.branch_active_frac,
        "branch_max_active_frac": args.branch_max_active_frac,
        "lambda_early": args.lambda_early,
        "lambda_final": args.lambda_final,
        "pulse_frac": args.pulse_frac,
        "anchor_decay": args.anchor_decay,
        "momentum_mu": args.momentum_mu,
        "trust_radius": args.trust_radius,
        "prox_lambda": args.prox_lambda,
        "credit_mode": args.credit_mode,
        "label_noise": args.label_noise,
    }
    return {key: value for key, value in pairs.items() if value is not None}


def main() -> int:
    args = add_args().parse_args()
    out_dir = ensure_dir(args.out_dir)
    run_dir = ensure_dir(out_dir / "runs")
    datasets = parse_str_list(args.datasets)
    methods = parse_str_list(args.methods)
    seeds = parse_int_list(args.seeds)
    overrides = override_dict(args)

    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "stage_b_runs.csv")
    total = len(datasets) * len(methods) * len(seeds)
    done = 0
    new_rows: List[Dict[str, Any]] = []

    for dataset in datasets:
        for method in methods:
            for seed in seeds:
                done += 1
                try:
                    cfg = config_from_args(args, dataset, method, seed, **overrides)
                    cfg.notes = "stage_b_recovered functional-update sweep"
                    row = train_one(cfg)
                    row["stage"] = "B"
                    print(
                        f"[{done}/{total}] {dataset} {method} seed={seed} "
                        f"acc={row['test_acc']:.4f} val_auc={row['val_auc']:.4f} "
                        f"branch={row['branch_output_norm_ratio']:.3f} "
                        f"phi={row['phi_prime_p95']:.4f} J={row['max_jac_condition']:.2f}"
                    )
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    row = {
                        "stage": "B",
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "label_noise": args.label_noise,
                        "error": repr(exc),
                        "test_acc": math.nan,
                    }
                    print(f"[{done}/{total}] ERROR {dataset} {method} seed={seed}: {exc!r}")

                rows.append(row)
                new_rows.append(row)
                safe_name = f"{dataset}_{method}_seed{seed}".replace("/", "_").replace(" ", "_")
                save_json(run_dir / f"{safe_name}.json", row)
                write_csv(out_dir / "stage_b_runs.csv", rows)

    summary = summarize_runs(rows, ["dataset", "method", "label_noise"])
    write_csv(out_dir / "stage_b_summary_by_method.csv", summary)
    save_json(out_dir / "stage_b_summary.json", {"new_runs": len(new_rows), "summary": summary})
    print(f"Wrote {len(new_rows)} new rows to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
