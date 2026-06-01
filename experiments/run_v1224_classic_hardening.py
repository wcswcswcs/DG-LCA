#!/usr/bin/env python
"""v12.24 formal Rational/Fourier hardening with an MLP reference."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v1223_failclosed_explore_open2_functional_rebuild as v1223
from dgkan.diagnostics.classic_basis import (
    CLASSIC_FAMILY_SPECS as FAMILY_SPECS,
    fnum,
    incremental_peak_bytes,
    memory_ratio as compute_memory_ratio,
    parse_csv,
    parse_ints,
    train_mlp_reference,
)


def run() -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="v1224_classic_hardening")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--artifact-prefix", default="v1224_classic_hardening")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--families", default="Rational,Fourier")
    ap.add_argument("--datasets", default="MNIST,Fashion-MNIST")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--train-size", type=int, default=1024)
    ap.add_argument("--val-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--linec-batch-size", type=int, default=64)
    ap.add_argument("--linec-sketch-dim", type=int, default=24)
    ap.add_argument("--mlp-hidden", type=int, default=160)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    v1223.ensure_dir(out_dir)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("v12.24 classic hardening requires CUDA")
    torch.cuda.set_device(device)

    families = parse_csv(args.families)
    datasets = [v1223.v120._canonical_dataset(d) for d in parse_csv(args.datasets)]
    seeds = parse_ints(args.seeds)
    base_args = argparse.Namespace(
        out_dir=str(out_dir),
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
    rows: list[dict[str, Any]] = []
    mlp_cache: dict[tuple[str, int], dict[str, Any]] = {}
    for dataset in datasets:
        for seed in seeds:
            load_args = argparse.Namespace(data_root=args.data_root, no_download=bool(args.no_download), seed=int(seed))
            data = v1223.v120._load_vision_split(load_args, dataset, train_size=int(args.train_size), val_size=int(args.val_size), test_size=int(args.val_size))
            x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, _xt, _yt, input_dim, output_dim, _protocol = data
            x_train = x_train_cpu.to(device=device, dtype=torch.float32)
            y_train = y_train_cpu.to(device=device)
            x_val = x_val_cpu.to(device=device, dtype=torch.float32)
            y_val = y_val_cpu.to(device=device)
            mlp_cache[(dataset, seed)] = train_mlp_reference(args, dataset, int(seed), int(input_dim), int(output_dim), x_train, y_train, x_val, y_val, device)
            torch.cuda.empty_cache()
            for family in families:
                method_id = FAMILY_SPECS[family]
                if device.type == "cuda":
                    family_baseline_alloc = float(torch.cuda.memory_allocated(device))
                    torch.cuda.reset_peak_memory_stats(device)
                else:
                    family_baseline_alloc = float("nan")
                t0 = time.perf_counter()
                try:
                    _, budget = v1223.v124._param_budget(int(input_dim), int(output_dim))
                    specs = {s.candidate_id: s for s in v1223.prim.primitive_specs(budget, int(input_dim), int(output_dim))}
                    raw = v1223.classic_family_smoke_one(base_args, dataset, int(seed), method_id, specs[method_id], device)
                    status = raw.get("status", "executed")
                    error = raw.get("linec_error", "")
                except Exception as exc:  # noqa: BLE001
                    raw = {}
                    status = "blocked"
                    error = f"{type(exc).__name__}: {exc}"
                family_peak = float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else float("nan")
                family_incr_peak = incremental_peak_bytes(family_peak, family_baseline_alloc)
                mlp = mlp_cache[(dataset, seed)]
                val_acc = fnum(raw.get("val_acc"))
                mlp_acc = fnum(mlp.get("mlp_val_acc"))
                step_ratio = fnum(raw.get("step_time_q90_ms")) / fnum(mlp.get("mlp_step_time_q90_ms")) if fnum(mlp.get("mlp_step_time_q90_ms")) > 0 else float("nan")
                memory_ratio_raw = compute_memory_ratio(family_peak, fnum(mlp.get("mlp_peak_memory_bytes")))
                memory_ratio = compute_memory_ratio(family_incr_peak, fnum(mlp.get("mlp_incremental_peak_memory_bytes")), memory_ratio_raw)
                linec_pass = int(
                    fnum(raw.get("linec_CouplingR2")) >= 0.15
                    and fnum(raw.get("linec_NoiseSignalLeak")) <= 0.20
                    and fnum(raw.get("linec_RealSignalReservoirRatio")) <= 0.70
                )
                exploration_pass = int(
                    linec_pass
                    and val_acc - mlp_acc >= -0.03
                    and step_ratio <= 1.25
                    and memory_ratio <= 1.0
                )
                rows.append(
                    {
                        "stage": "V1224_CLASSIC_HARDENING",
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
                        "val_acc": raw.get("val_acc", ""),
                        "mlp_val_acc": mlp.get("mlp_val_acc", ""),
                        "mean_delta_vs_MLP": val_acc - mlp_acc if math.isfinite(val_acc) and math.isfinite(mlp_acc) else "",
                        "NLL": raw.get("NLL", ""),
                        "ECE": raw.get("ECE", ""),
                        "CEp99": raw.get("CEp99", ""),
                        "step_time_q90_ms": raw.get("step_time_q90_ms", ""),
                        "mlp_step_time_q90_ms": mlp.get("mlp_step_time_q90_ms", ""),
                        "step_ratio_vs_mlp": step_ratio,
                        "family_peak_memory_bytes": family_peak,
                        "mlp_peak_memory_bytes": mlp.get("mlp_peak_memory_bytes", ""),
                        "family_memory_baseline_bytes": family_baseline_alloc,
                        "mlp_memory_baseline_bytes": mlp.get("mlp_memory_baseline_bytes", ""),
                        "family_incremental_peak_memory_bytes": family_incr_peak,
                        "mlp_incremental_peak_memory_bytes": mlp.get("mlp_incremental_peak_memory_bytes", ""),
                        "memory_ratio_raw_vs_mlp": memory_ratio_raw,
                        "memory_ratio_vs_mlp": memory_ratio,
                        "linec_CouplingR2": raw.get("linec_CouplingR2", ""),
                        "linec_NoiseSignalLeak": raw.get("linec_NoiseSignalLeak", ""),
                        "linec_RealSignalReservoirRatio": raw.get("linec_RealSignalReservoirRatio", ""),
                        "linec_pass": linec_pass,
                        "exploration_gate_pass": exploration_pass,
                        "blocker_or_note": error,
                        "promotion_allowed": 0,
                        "no_fake": 1,
                    }
                )
                torch.cuda.empty_cache()
    csv_path = out_dir / f"{args.artifact_prefix}.csv"
    v1223.write_csv_rows(csv_path, rows)
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(str(row["family"]), []).append(row)
    summary_rows = []
    for family, group in sorted(by_family.items()):
        summary_rows.append(
            {
                "stage": "V1224_CLASSIC_HARDENING_SUMMARY",
                "family": family,
                "rows": len(group),
                "exploration_pass_rows": sum(v1223.safe_int(r.get("exploration_gate_pass"), 0) for r in group),
                "mean_delta_vs_MLP": sum(fnum(r.get("mean_delta_vs_MLP"), 0.0) for r in group) / max(1, len(group)),
                "mean_linec_CouplingR2": sum(fnum(r.get("linec_CouplingR2"), 0.0) for r in group) / max(1, len(group)),
                "mean_linec_NoiseSignalLeak": sum(fnum(r.get("linec_NoiseSignalLeak"), 0.0) for r in group) / max(1, len(group)),
                "mean_linec_RealSignalReservoirRatio": sum(fnum(r.get("linec_RealSignalReservoirRatio"), 0.0) for r in group) / max(1, len(group)),
            }
        )
    summary_csv = out_dir / f"{args.artifact_prefix}_summary.csv"
    summary_json = out_dir / f"{args.artifact_prefix}_summary.json"
    v1223.write_csv_rows(summary_csv, summary_rows)
    result = {
        "stage": "V1224_CLASSIC_HARDENING_AGGREGATE",
        "artifact_csv": v1223.rel(csv_path),
        "summary_csv": v1223.rel(summary_csv),
        "rows": len(rows),
        "summary_rows": len(summary_rows),
        "exploration_pass_rows": sum(v1223.safe_int(r.get("exploration_gate_pass"), 0) for r in rows),
        "promotion_allowed": 0,
        "no_fake": 1,
    }
    v1223.write_json(summary_json, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    run()
