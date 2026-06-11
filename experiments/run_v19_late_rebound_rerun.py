#!/usr/bin/env python3
"""Independent v19 D-CHE/M2 late-rebound rerun matrix."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v17_common import (
    CONTROL_MECHANISMS,
    append_log,
    carrier_model,
    finite_float,
    load_dataset,
    now_sg,
    resolve_device,
    train_one,
    write_rows,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data-root", default="data")
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--reruns", type=int, default=3)
    p.add_argument("--train-size", type=int, default=64)
    p.add_argument("--val-size", type=int, default=48)
    p.add_argument("--input-size", type=int, default=8)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--param-budget", type=int, default=12000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--horizon-steps", type=int, default=4800)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--fu-lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.001)
    p.add_argument("--alt-period", type=int, default=10)
    p.add_argument("--sanity-eps", type=float, default=1.0e-3)
    p.add_argument("--shard-count", type=int, default=4)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--basis-repair-variant", default="CHE-R0-current")
    return p


def trace_at(traces: list[dict[str, object]], step: int) -> dict[str, object]:
    for row in traces:
        if int(finite_float(row.get("step"), -1)) == int(step):
            return row
    return {}


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    append_log(out_dir, f"## {now_sg()} v19 late rebound rerun shard {args.shard_index}/{args.shard_count}")
    device = resolve_device(args.device)
    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    mechanisms = [
        "M2-SGDMomentumPrimaryFU",
        "CTRL-AdamW",
        "CTRL-SGD",
        "CTRL-NoOpMatchedOverhead",
        "CTRL-RandomMatchedNorm",
        "CTRL-RecoveryOnly",
    ]
    jobs = []
    for rerun_id in range(int(args.reruns)):
        for dataset in datasets:
            for seed in seeds:
                for mechanism in mechanisms:
                    jobs.append({"rerun_id": rerun_id, "dataset": dataset, "seed": seed, "mechanism": mechanism})
    selected = [job for idx, job in enumerate(jobs) if idx % int(args.shard_count) == int(args.shard_index)]
    matrix_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    for job_index, job in enumerate(selected):
        started = time.time()
        row = {
            "job_index": job_index,
            "global_job_count": len(jobs),
            "shard_count": int(args.shard_count),
            "shard_index": int(args.shard_index),
            "rerun_id": job["rerun_id"],
            "dataset": job["dataset"],
            "seed": job["seed"],
            "carrier": "D-CHE",
            "mechanism": job["mechanism"],
            "control_mechanism": int(str(job["mechanism"]) in CONTROL_MECHANISMS),
            "horizon_steps": int(args.horizon_steps),
            "basis_repair_variant": args.basis_repair_variant,
            "execution_status": "started",
        }
        try:
            x_train, y_train, x_val, y_val = load_dataset(
                str(job["dataset"]),
                Path(args.data_root),
                int(args.train_size),
                int(args.val_size),
                int(job["seed"]) + 19_000 + 101 * int(job["rerun_id"]),
                device,
                int(args.input_size),
            )
            local_args = argparse.Namespace(**vars(args))
            local_args.steps = int(args.horizon_steps)
            local_args.basis_repair_variant = args.basis_repair_variant
            model_seed = 190_000 + 1009 * int(job["rerun_id"]) + 37 * int(job["seed"]) + sum(ord(c) for c in str(job["mechanism"]))
            model = carrier_model("D-CHE", x_train, model_seed, local_args, device)
            final, traces, _ = train_one(
                model,
                str(job["mechanism"]),
                x_train,
                y_train,
                x_val,
                y_val,
                local_args,
                seed=model_seed,
            )
            for h in [100, 400, 800, 1600, 2400, 3200, 4800]:
                tr = trace_at(traces, h)
                row[f"val_loss_h{h}"] = tr.get("val_loss", "")
                row[f"val_acc_h{h}"] = tr.get("val_acc", "")
                row[f"CEp99_h{h}"] = tr.get("CEp99", "")
                row[f"NLL_h{h}"] = tr.get("NLL", "")
                row[f"ECE_h{h}"] = tr.get("ECE", "")
                row[f"Brier_h{h}"] = tr.get("Brier", "")
                row[f"LineC_channel_h{h}"] = tr.get("LineC_channel_CouplingR2", "")
                row[f"LineC_fast_h{h}"] = tr.get("LineC_fast_CouplingR2", "")
            row.update(
                {
                    "final_val_loss": final.get("val_loss", ""),
                    "final_val_acc": final.get("val_acc", ""),
                    "final_CEp99": final.get("CEp99", ""),
                    "final_NLL": final.get("NLL", ""),
                    "final_ECE": final.get("ECE", ""),
                    "final_Brier": final.get("Brier", ""),
                    "runtime_sec": time.time() - started,
                    "execution_status": "measured",
                    "error_type": "",
                    "error_message": "",
                }
            )
            for tr in traces:
                trace = dict(tr)
                trace.update(
                    {
                        "rerun_id": job["rerun_id"],
                        "dataset": job["dataset"],
                        "seed": job["seed"],
                        "carrier": "D-CHE",
                        "mechanism": job["mechanism"],
                        "shard_index": int(args.shard_index),
                    }
                )
                trace_rows.append(trace)
        except Exception as exc:
            row.update(
                {
                    "runtime_sec": time.time() - started,
                    "execution_status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                }
            )
        matrix_rows.append(row)
        write_rows(out_dir / f"v19_late_rebound_rerun_matrix_r{args.shard_index}.csv", matrix_rows)
        write_rows(out_dir / f"v19_late_rebound_traces_r{args.shard_index}.csv", trace_rows)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
