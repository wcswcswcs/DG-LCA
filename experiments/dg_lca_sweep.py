#!/usr/bin/env python3
"""Systematic DG-LCA toy sweeps.

The goal is to turn one-off gate runs into experiment tables:

* Gate 1 rank/staleness sweep for structured bottleneck residual blocks.
* Gate 2 Sobolev-preconditioner grid over target functions and hyperparameters.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dg_lca_toy import (
    build_parser as build_toy_parser,
    ensure_dir,
    get_device,
    now_tag,
    parse_float_list,
    run_gate1,
    save_json,
    train_kan_like,
)


def parse_int_list(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_str_list(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def toy_defaults() -> argparse.Namespace:
    return build_toy_parser().parse_args([])


def gate1_rank_sweep(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    ranks = parse_int_list(args.ranks)
    rows: List[Dict[str, Any]] = []
    for rank in ranks:
        run_args = toy_defaults()
        run_args.gate = "gate1"
        run_args.device = args.device
        run_args.seed = args.seed
        run_args.router_rank = rank
        run_args.router_steps = args.router_steps
        run_args.router_hidden = args.router_hidden
        run_args.router_batch_size = args.router_batch_size
        run_args.router_eval_batches = args.router_eval_batches
        run_args.router_lr = args.router_lr
        run_args.gate1_block = args.gate1_block
        run_args.gate1_bottleneck = args.gate1_bottleneck
        run_args.block_scale = args.block_scale
        run_args.stale_noise_levels = args.stale_noise_levels
        run_args.out_dir = out_dir / f"gate1_{args.gate1_block}_rank{rank}"
        ensure_dir(run_args.out_dir)

        summary = run_gate1(run_args, run_args.out_dir)
        metrics = summary["metrics"]
        row: Dict[str, Any] = {
            "rank": rank,
            "block": summary["block_type"],
            "bottleneck": summary["block_bottleneck"],
            "status": summary["gate1_status"],
            "cosine": metrics["router_cosine"],
            "relerr": metrics["router_relerr"],
            "norm_ratio": metrics["router_norm_ratio"],
            "identity_cosine": metrics["identity_cosine"],
            "identity_relerr": metrics["identity_relerr"],
            "linearity_residual": metrics["linearity_residual"],
            "train_loss_last_20_mean": summary["train_loss_last_20_mean"],
        }
        for stale_row in summary["stale_metrics"]:
            noise = stale_row["noise_scale"]
            row[f"stale_cos_noise_{noise:g}"] = stale_row["router_cosine"]
            row[f"stale_relerr_noise_{noise:g}"] = stale_row["router_relerr"]
        rows.append(row)

    rows = sorted(rows, key=lambda r: (r["relerr"], -r["cosine"]))
    result = {
        "mode": "gate1_rank_sweep",
        "block": args.gate1_block,
        "bottleneck": args.gate1_bottleneck,
        "ranks": ranks,
        "rows": rows,
        "best_by_relerr": rows[0] if rows else None,
    }
    save_json(out_dir / "gate1_rank_sweep.json", result)
    write_csv(out_dir / "gate1_rank_sweep.csv", rows)
    return result


def gate2_grid(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    device = get_device(args.device)
    targets = parse_str_list(args.targets)
    lrs = parse_float_list(args.precond_lrs)
    alphas = parse_float_list(args.alphas)
    betas = parse_float_list(args.betas)

    rows: List[Dict[str, Any]] = []
    baselines: Dict[str, Dict[str, Any]] = {}
    for target in targets:
        adam = train_kan_like(
            "adam",
            seed=args.seed,
            n_basis=args.kan_basis,
            steps=args.kan_steps,
            target=target,
            device=device,
            lr=args.kan_adam_lr,
            alpha=0.0,
            beta=0.0,
            rho=args.rho,
        )
        adam_row = asdict(adam)
        adam_row.update({"target": target, "alpha": None, "beta": None, "lr": args.kan_adam_lr})
        baselines[target] = adam_row
        rows.append(adam_row)

        for lr in lrs:
            for alpha in alphas:
                for beta in betas:
                    pre = train_kan_like(
                        "sobolev_precond",
                        seed=args.seed,
                        n_basis=args.kan_basis,
                        steps=args.kan_steps,
                        target=target,
                        device=device,
                        lr=lr,
                        alpha=alpha,
                        beta=beta,
                        rho=args.rho,
                    )
                    pre_row = asdict(pre)
                    pre_row.update({"target": target, "alpha": alpha, "beta": beta, "lr": lr})
                    pre_row["val_ratio_vs_adam"] = pre_row["val_loss"] / adam_row["val_loss"]
                    pre_row["slope_ratio_vs_adam"] = pre_row["max_abs_slope"] / adam_row["max_abs_slope"]
                    pre_row["curv_ratio_vs_adam"] = pre_row["curvature_energy"] / adam_row["curvature_energy"]
                    rows.append(pre_row)

    precond_rows = [row for row in rows if row["method"] == "sobolev_precond"]
    best_by_target: Dict[str, Dict[str, Any]] = {}
    smoothest_by_target: Dict[str, Dict[str, Any]] = {}
    for target in targets:
        target_rows = [row for row in precond_rows if row["target"] == target]
        best_by_target[target] = min(target_rows, key=lambda r: r["val_loss"])
        smoothest_by_target[target] = min(target_rows, key=lambda r: r["curvature_energy"])

    result = {
        "mode": "gate2_grid",
        "targets": targets,
        "kan_steps": args.kan_steps,
        "kan_basis": args.kan_basis,
        "baselines": baselines,
        "best_by_target": best_by_target,
        "smoothest_by_target": smoothest_by_target,
        "rows": rows,
    }
    save_json(out_dir / "gate2_grid.json", result)
    write_csv(out_dir / "gate2_grid.csv", rows)
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["gate1-rank", "gate2-grid", "all"], default="all")
    p.add_argument("--out-dir", type=Path, default=Path("results/dg_lca_sweeps"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--ranks", default="1,2,4,8,16,32")
    p.add_argument("--gate1-block", choices=["residual", "mlp", "bottleneck_residual"], default="bottleneck_residual")
    p.add_argument("--gate1-bottleneck", type=int, default=16)
    p.add_argument("--block-scale", type=float, default=1.0)
    p.add_argument("--router-steps", type=int, default=800)
    p.add_argument("--router-hidden", type=int, default=128)
    p.add_argument("--router-batch-size", type=int, default=256)
    p.add_argument("--router-eval-batches", type=int, default=8)
    p.add_argument("--router-lr", type=float, default=2e-3)
    p.add_argument("--stale-noise-levels", default="0,0.01,0.03,0.1")

    p.add_argument("--targets", default="sin5,smooth_step")
    p.add_argument("--kan-steps", type=int, default=350)
    p.add_argument("--kan-basis", type=int, default=32)
    p.add_argument("--kan-adam-lr", type=float, default=0.03)
    p.add_argument("--precond-lrs", default="0.15,0.25")
    p.add_argument("--alphas", default="0.01,0.05")
    p.add_argument("--betas", default="0,0.0005,0.005")
    p.add_argument("--rho", type=float, default=1e-2)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_dir = ensure_dir(args.out_dir / now_tag())
    results: Dict[str, Any] = {"run_dir": str(run_dir), "mode": args.mode}
    save_json(run_dir / "config.json", vars(args))

    if args.mode in {"gate1-rank", "all"}:
        results["gate1_rank"] = gate1_rank_sweep(args, run_dir)
    if args.mode in {"gate2-grid", "all"}:
        results["gate2_grid"] = gate2_grid(args, run_dir)

    save_json(run_dir / "summary.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
