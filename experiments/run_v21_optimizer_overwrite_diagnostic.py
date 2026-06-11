#!/usr/bin/env python3
"""v21 FU-vs-AdamW overwrite diagnostic.

This is a first-step audit only. It does not promote any source candidate.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v17_common import carrier_model, first_step_diagnostics, load_dataset, resolve_device  # noqa: E402
from experiments.run_v21_common import PYTHON, append_exec, ensure_out, v21_functional_specs, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--scope", default="mlp_source", choices=["mlp_source", "kan_writer"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data-root", default="data")
    p.add_argument("--carriers", default="MLP")
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--train-size", type=int, default=64)
    p.add_argument("--val-size", type=int, default=48)
    p.add_argument("--input-size", type=int, default=8)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--param-budget", type=int, default=12000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=4800)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--fu-lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.001)
    p.add_argument("--alt-period", type=int, default=10)
    p.add_argument("--source-warmup-steps", type=int, default=0)
    p.add_argument("--basis-repair-variant", default="R0-current")
    p.add_argument("--basis-repair-variant-fou", default="FOU-R4-k4-triton-no-materialize")
    p.add_argument("--init-seed-offset", type=int, default=0)
    p.add_argument("--sanity-eps", type=float, default=1.0e-3)
    p.add_argument("--spec-ids", default="")
    p.add_argument("--run-label", default="v21_optimizer_overwrite")
    return p


def selected_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = v21_functional_specs(str(args.scope))
    wanted = {x.strip() for x in str(args.spec_ids).split(",") if x.strip()}
    if wanted:
        specs = [
            s
            for s in specs
            if str(s.get("v21_id")) in wanted
            or str(s.get("continuation_id")) in wanted
            or str(s.get("mechanism")) in wanted
        ]
    return specs


def jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    carriers = [x.strip() for x in str(args.carriers).split(",") if x.strip()]
    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    out: list[dict[str, Any]] = []
    idx = 0
    for carrier in carriers:
        for dataset in datasets:
            for seed in seeds:
                for spec in selected_specs(args):
                    repair = str(args.basis_repair_variant)
                    if carrier == "D-FOU":
                        repair = str(args.basis_repair_variant_fou)
                    if carrier == "MLP":
                        repair = "R0-current"
                    out.append(
                        {
                            "job_index": idx,
                            "run_label": str(args.run_label),
                            "scope": str(args.scope),
                            "dataset": dataset,
                            "seed": seed,
                            "carrier": carrier,
                            "basis_repair_variant": repair,
                            "init_seed_offset": int(args.init_seed_offset),
                            **spec,
                        }
                    )
                    idx += 1
    return out


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("carrier")), str(row.get("basis_repair_variant")), str(row.get("v21_id"))), []).append(row)
    out: list[dict[str, Any]] = []
    for (carrier, variant, v21_id), group in sorted(groups.items()):
        def vals(key: str) -> list[float]:
            ans = []
            for item in group:
                try:
                    ans.append(float(item.get(key)))
                except Exception:
                    pass
            return ans

        cos_vals = vals("cos_FU_AdamW")
        proj_vals = vals("optimizer_overwrite_projection")
        out.append(
            {
                "carrier": carrier,
                "basis_repair_variant": variant,
                "v21_id": v21_id,
                "rows": len(group),
                "mechanism": group[0].get("mechanism", "") if group else "",
                "cos_FU_AdamW_mean": sum(cos_vals) / len(cos_vals) if cos_vals else "",
                "optimizer_overwrite_projection_mean": sum(proj_vals) / len(proj_vals) if proj_vals else "",
                "adamw_blocks_fu_direction": int(bool(proj_vals) and sum(proj_vals) / len(proj_vals) <= -0.20),
                "diagnosis": "AdamWOverwriteLikely" if bool(proj_vals) and sum(proj_vals) / len(proj_vals) <= -0.20 else "NoAdamWOverwriteBlockerAtFirstStep",
            }
        )
    return out


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    device = resolve_device(args.device)
    selected = jobs(args)
    append_exec(out_dir, f"{PYTHON} experiments/run_v21_optimizer_overwrite_diagnostic.py --scope {args.scope} --device {args.device}", status="started", note=f"jobs={len(selected)}")
    rows: list[dict[str, Any]] = []
    for job in selected:
        started = time.perf_counter()
        try:
            x_train, y_train, _x_val, _y_val = load_dataset(str(job["dataset"]), Path(args.data_root), int(args.train_size), int(args.val_size), int(job["seed"]), device, int(args.input_size))
            local = deepcopy(args)
            local.fu_lr = float(job["fu_lr"])
            local.alt_period = int(job["alt_period"])
            local.source_warmup_steps = int(job.get("source_warmup_steps", 0))
            local.basis_repair_variant = str(job["basis_repair_variant"])
            train_seed = 210_000 + int(job["job_index"]) + int(job.get("init_seed_offset", 0))
            model = carrier_model(str(job["carrier"]), x_train, train_seed, local, device)
            sign, overwrite, update = first_step_diagnostics(model, str(job["mechanism"]), x_train[: min(32, len(x_train))], y_train[: min(32, len(y_train))], local, train_seed)
            rows.append(
                {
                    **job,
                    **overwrite,
                    "update_kind": update.kind,
                    "update_space": update.space,
                    "update_source": update.source,
                    "update_norm": update.to_metadata().get("update_norm", ""),
                    "small_step_status": sign.get("status", ""),
                    "small_step_loss_sanity": sign.get("small_step_loss_sanity", ""),
                    "runtime_sec": time.perf_counter() - started,
                    "execution_status": "measured",
                    "blocker": "",
                    "promotion_allowed": 0,
                }
            )
        except Exception as exc:
            rows.append({**job, "runtime_sec": time.perf_counter() - started, "execution_status": f"blocked:{type(exc).__name__}", "blocker": str(exc)[:500], "promotion_allowed": 0})
    write_rows(out_dir / "v21_adamw_overwrite_diagnostic.csv", rows)
    write_rows(out_dir / "v21_adamw_overwrite_summary.csv", summarize(rows))
    append_exec(out_dir, f"{PYTHON} experiments/run_v21_optimizer_overwrite_diagnostic.py --scope {args.scope}", status="completed", note=f"rows={len(rows)} grouped={len(summarize(rows))}")


if __name__ == "__main__":
    main()
