#!/usr/bin/env python3
"""Recovered Stage A DG-KAN baselines.

Stage A in the docs asks a narrow question: is the residual DG-KAN
architecture itself trainable, and how does it compare with a plain MLP /
all-parameter AdamW baseline before adding functional-update machinery?

This runner intentionally uses the shared recovered core so later Stage B/F
scripts measure the same model, metrics, and geometry audits.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

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


METHOD_ALIASES = {
    "mlp": "mlp_adamw",
    "mlp_adam": "mlp_adamw",
    "mlp_adamw": "mlp_adamw",
    "residual_mlp": "mlp_adamw",
    "adam": "AdamW",
    "adamw": "AdamW",
    "fullbp": "AdamW",
    "fullbp_dgkan": "AdamW",
    "dgkan": "AdamW",
    "dgkan_adamw": "AdamW",
    "kan_coeff_adam": "static_top",
    "kan_coeff_adamw": "static_top",
    "dgkan_coeff_adam": "static_top",
    "dgkan_coeff_adamw": "static_top",
    "functional_static": "static_top",
    "functional_geometry": "geometry_current",
}


def canonical_method(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    return METHOD_ALIASES.get(key, name.strip())


def add_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--methods", default="mlp_adamw,AdamW,static_top,geometry_current")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--out-dir", type=Path, default=Path("results/stage_a_recovered"))
    p.add_argument("--fresh", action="store_true", help="Ignore previous CSV rows in out-dir.")
    add_common_train_args(p)
    return p


def main() -> int:
    args = add_args().parse_args()
    out_dir = ensure_dir(args.out_dir)
    run_dir = ensure_dir(out_dir / "runs")
    datasets = parse_str_list(args.datasets)
    requested_methods = parse_str_list(args.methods)
    seeds = parse_int_list(args.seeds)

    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "stage_a_runs.csv")
    new_rows: List[Dict[str, Any]] = []
    total = len(datasets) * len(requested_methods) * len(seeds)
    done = 0

    for dataset in datasets:
        for requested in requested_methods:
            method = canonical_method(requested)
            for seed in seeds:
                done += 1
                try:
                    cfg = config_from_args(args, dataset, method, seed)
                    cfg.notes = f"stage_a_recovered requested_method={requested}"
                    row = train_one(cfg)
                    row["requested_method"] = requested
                    row["stage"] = "A"
                    print(
                        f"[{done}/{total}] {dataset} {requested} seed={seed} "
                        f"acc={row['test_acc']:.4f} loss={row['test_loss']:.4f} "
                        f"phi={row['phi_prime_p95']:.4f} J={row['max_jac_condition']:.2f}"
                    )
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    row = {
                        "stage": "A",
                        "dataset": dataset,
                        "method": method,
                        "requested_method": requested,
                        "seed": seed,
                        "error": repr(exc),
                        "test_acc": math.nan,
                    }
                    print(f"[{done}/{total}] ERROR {dataset} {requested} seed={seed}: {exc!r}")

                rows.append(row)
                new_rows.append(row)
                safe_name = f"{dataset}_{requested}_seed{seed}".replace("/", "_").replace(" ", "_")
                save_json(run_dir / f"{safe_name}.json", row)
                write_csv(out_dir / "stage_a_runs.csv", rows)

    summary = summarize_runs(rows, ["dataset", "method"])
    summary_requested = summarize_runs(rows, ["dataset", "requested_method"])
    write_csv(out_dir / "stage_a_summary_by_method.csv", summary)
    write_csv(out_dir / "stage_a_summary_by_requested_method.csv", summary_requested)
    save_json(
        out_dir / "stage_a_summary.json",
        {
            "new_runs": len(new_rows),
            "total_rows": len(rows),
            "summary_by_method": summary,
            "summary_by_requested_method": summary_requested,
        },
    )
    print(f"Wrote {len(new_rows)} new rows to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
