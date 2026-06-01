#!/usr/bin/env python3
"""Targeted v12.23 Line D hardening driver.

This script reuses the v12.23 official Line D primitive smoke implementation,
but lets the continuation run selected classic families, datasets, and seeds in
parallel per GPU.  It writes top-level v1223_* artifacts so the official v12.23
code-review packet can include the hardening evidence without special casing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO_ROOT / "experiments"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

import run_v1223_failclosed_explore_open2_functional_rebuild as v1223  # noqa: E402


FAMILY_SPECS = {
    "Rational": "B7b-RationalKAT-flashgroup-G16-h112-linearres-tritonL3",
    "Fourier": "B4g-FourierKAN-lowfreq-K2-tritonL3-matmulTile",
}


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def mean(values: Sequence[Any]) -> float:
    vals = [finite_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def build_line_d_base_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        out_dir=str(args.out_dir),
        data_root=args.data_root,
        no_download=bool(args.no_download),
        line_d_datasets=args.datasets,
        line_d_seeds=args.seeds,
        line_d_train_size=int(args.train_size),
        line_d_val_size=int(args.val_size),
        line_d_batch_size=int(args.batch_size),
        line_d_epochs=int(args.epochs),
        line_d_linec_batch=int(args.linec_batch_size),
        line_d_sketch_dim=int(args.linec_sketch_dim),
    )


def run_hardening(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    v1223.ensure_dir(out_dir)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("v12.23 Line D hardening requires CUDA; CPU-offload is not allowed")
    torch.cuda.set_device(device)

    datasets = [v1223.v120._canonical_dataset(x) for x in parse_csv(args.datasets)]
    seeds = parse_csv_ints(args.seeds)
    families = parse_csv(args.families)
    unknown = sorted(set(families) - set(FAMILY_SPECS))
    if unknown:
        raise ValueError(f"unknown families: {unknown}; available={sorted(FAMILY_SPECS)}")
    if not datasets or not seeds or not families:
        raise ValueError("families, datasets, and seeds must all be non-empty")

    base_args = build_line_d_base_args(args)
    probe_args = v1223.line_d_args(base_args, str(device))
    probe_args.seed = int(seeds[0])
    data = v1223.v120._load_vision_split(
        probe_args,
        datasets[0],
        train_size=int(args.train_size),
        val_size=int(args.val_size),
        test_size=int(args.val_size),
    )
    input_dim, output_dim = int(data[6]), int(data[7])
    _, budget = v1223.v124._param_budget(input_dim, output_dim)
    specs = {s.candidate_id: s for s in v1223.prim.primitive_specs(budget, input_dim, output_dim)}

    rows: list[dict[str, Any]] = []
    started_at = v1223.now_iso()
    for dataset in datasets:
        for seed in seeds:
            for family in families:
                method_id = FAMILY_SPECS[family]
                status = "KernelBlocked"
                error = ""
                metrics: dict[str, Any] = {}
                t0 = time.perf_counter()
                try:
                    row = v1223.classic_family_smoke_one(base_args, dataset, int(seed), method_id, specs[method_id], device)
                    metrics = row
                    valid_task = math.isfinite(finite_float(row.get("val_acc")))
                    valid_linec = math.isfinite(finite_float(row.get("linec_CouplingR2")))
                    status = "FamilyNearPass" if valid_task and valid_linec else ("TaskBlocked" if valid_task else "KernelBlocked")
                except Exception as exc:  # noqa: BLE001 - blocker text is an audit artifact.
                    error = f"{type(exc).__name__}: {exc}"
                    if "out of memory" in error.lower():
                        status = "KernelBlocked"
                    elif "linec" in error.lower() or "classification" in error.lower():
                        status = "TaskBlocked"
                    else:
                        status = "KernelBlocked"
                rows.append(
                    {
                        "stage": "V1223_LINE_D_HARDENING",
                        "run_id": args.run_id,
                        "family": family,
                        "candidate_id": method_id,
                        "dataset": dataset,
                        "seed": int(seed),
                        "device": str(device),
                        "train_size": int(args.train_size),
                        "val_size": int(args.val_size),
                        "epochs": int(args.epochs),
                        "status": status,
                        "elapsed_sec": time.perf_counter() - t0,
                        "val_acc": metrics.get("val_acc", ""),
                        "NLL": metrics.get("NLL", ""),
                        "ECE": metrics.get("ECE", ""),
                        "CEp99": metrics.get("CEp99", ""),
                        "step_time_q90_ms": metrics.get("step_time_q90_ms", ""),
                        "linec_CouplingR2": metrics.get("linec_CouplingR2", ""),
                        "linec_NoiseSignalLeak": metrics.get("linec_NoiseSignalLeak", ""),
                        "linec_RealSignalReservoirRatio": metrics.get("linec_RealSignalReservoirRatio", ""),
                        "linec_error": metrics.get("linec_error", ""),
                        "blocker_or_note": error or metrics.get("linec_error", ""),
                        "promotion_allowed": 0,
                        "no_fake": 1,
                    }
                )
                torch.cuda.empty_cache()

    summary_rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(str(row["family"]), str(row["dataset"]))].append(row)
    for (family, dataset), group in sorted(by_key.items()):
        summary_rows.append(
            {
                "stage": "V1223_LINE_D_HARDENING_SUMMARY",
                "run_id": args.run_id,
                "family": family,
                "dataset": dataset,
                "rows": len(group),
                "family_near_pass_rows": sum(1 for r in group if r.get("status") == "FamilyNearPass"),
                "kernel_blocked_rows": sum(1 for r in group if r.get("status") == "KernelBlocked"),
                "mean_val_acc": mean([r.get("val_acc") for r in group]),
                "mean_NLL": mean([r.get("NLL") for r in group]),
                "mean_linec_CouplingR2": mean([r.get("linec_CouplingR2") for r in group]),
                "mean_linec_NoiseSignalLeak": mean([r.get("linec_NoiseSignalLeak") for r in group]),
                "mean_linec_RealSignalReservoirRatio": mean([r.get("linec_RealSignalReservoirRatio") for r in group]),
                "promotion_allowed": 0,
                "no_fake": 1,
            }
        )

    prefix = args.artifact_prefix
    v1223.write_csv_rows(out_dir / f"{prefix}.csv", rows)
    v1223.write_csv_rows(out_dir / f"{prefix}_summary.csv", summary_rows)
    summary = {
        "stage": "V1223_LINE_D_HARDENING_ROUTE",
        "run_id": args.run_id,
        "generated_at": v1223.now_iso(),
        "started_at": started_at,
        "artifact_prefix": prefix,
        "device": str(device),
        "families": families,
        "datasets": datasets,
        "seeds": seeds,
        "rows": len(rows),
        "summary_rows": len(summary_rows),
        "family_near_pass_rows": sum(1 for r in rows if r.get("status") == "FamilyNearPass"),
        "kernel_blocked_rows": sum(1 for r in rows if r.get("status") == "KernelBlocked"),
        "promotion_allowed": 0,
        "route_impact": "Line D hardening evidence only; not an official S1-S5 promotion route in v12.23",
        "no_fake": 1,
    }
    v1223.write_json(out_dir / f"{prefix}_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", default=str(v1223.DEFAULT_OUT_DIR))
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--families", default="Rational,Fourier")
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--linec-batch-size", type=int, default=64)
    parser.add_argument("--linec-sketch-dim", type=int, default=24)
    return parser


if __name__ == "__main__":
    run_hardening(build_argparser().parse_args())
