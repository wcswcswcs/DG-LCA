#!/usr/bin/env python3
"""v21 MLP source-retention dynamics runner."""

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

from experiments.run_v17_common import CONTROL_MECHANISMS, carrier_model, load_dataset, resolve_device, train_one  # noqa: E402
from experiments.run_v21_common import (  # noqa: E402
    PYTHON,
    append_exec,
    ensure_out,
    finite_float,
    merge_csvs,
    slug,
    v21_functional_specs,
    write_rows,
)
from dgkan.fu.source_channel import grouped_source_retention  # noqa: E402


def parser(default_scope: str = "mlp_source") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--scope", default=default_scope, choices=["mlp_source", "kan_writer"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data-root", default="data")
    p.add_argument("--carriers", default="MLP" if default_scope == "mlp_source" else "D-CHE,D-FOU")
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
    p.add_argument("--basis-repair-variant", default="R0-current" if default_scope == "mlp_source" else "CHE-R4-k3-gradbuf-triton")
    p.add_argument("--basis-repair-variant-fou", default="FOU-R4-k4-triton-no-materialize")
    p.add_argument("--init-seed-offset", type=int, default=0)
    p.add_argument("--sanity-eps", type=float, default=1.0e-3)
    p.add_argument("--spec-ids", default="")
    p.add_argument("--run-label", default="")
    p.add_argument("--shard-count", type=int, default=4)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--merge-only", action="store_true")
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
                            "run_label": str(args.run_label or args.scope),
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


def prefix(scope: str) -> str:
    return "v21_mlp_source_dynamics" if scope == "mlp_source" else "v21_kan_source_writer"


def enrich_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str, str, str, str], float] = {}
    for row in rows:
        if str(row.get("mechanism")) not in CONTROL_MECHANISMS:
            continue
        for h in ["h800", "h1600", "h3200", "h4800"]:
            val = finite_float(row.get(f"val_loss_{h}"))
            if val == val:
                key = (str(row.get("carrier")), str(row.get("basis_repair_variant")), str(row.get("dataset")), str(row.get("seed")), h)
                best[key] = min(best.get(key, float("inf")), val)
    out = []
    for row in rows:
        item = dict(row)
        for h in ["h800", "h1600", "h3200", "h4800"]:
            val = finite_float(item.get(f"val_loss_{h}"))
            base = best.get((str(item.get("carrier")), str(item.get("basis_repair_variant")), str(item.get("dataset")), str(item.get("seed")), h), float("nan"))
            item[f"source_{h}"] = base - val if val == val and base == base else ""
        out.append(item)
    return out


def run_shard(args: argparse.Namespace) -> None:
    out_dir = ensure_out(args.out_dir)
    device = resolve_device(args.device)
    selected = [j for j in jobs(args) if int(j["job_index"]) % int(args.shard_count) == int(args.shard_index)]
    append_exec(out_dir, f"{PYTHON} experiments/run_v21_mlp_source_dynamics.py --scope {args.scope} --shard-index {args.shard_index} --device {args.device}", status="started", note=f"jobs={len(selected)}")
    rows = []
    trace_rows = []
    for job in selected:
        start = time.perf_counter()
        try:
            x_train, y_train, x_val, y_val = load_dataset(str(job["dataset"]), Path(args.data_root), int(args.train_size), int(args.val_size), int(job["seed"]), device, int(args.input_size))
            local = deepcopy(args)
            local.fu_lr = float(job["fu_lr"])
            local.alt_period = int(job["alt_period"])
            local.source_warmup_steps = int(job.get("source_warmup_steps", 0))
            local.basis_repair_variant = str(job["basis_repair_variant"])
            train_seed = 210_000 + int(job["job_index"]) + int(job.get("init_seed_offset", 0))
            model = carrier_model(str(job["carrier"]), x_train, train_seed, local, device)
            final, traces, _first_update = train_one(model, str(job["mechanism"]), x_train, y_train, x_val, y_val, local, seed=train_seed)
            row = {**job, **final, "runtime_sec": time.perf_counter() - start, "execution_status": "measured", "blocker": "", "promotion_allowed": 0}
            rows.append(row)
            for tr in traces:
                trace_rows.append({**job, **tr})
        except Exception as exc:
            rows.append({**job, "runtime_sec": time.perf_counter() - start, "execution_status": f"blocked:{type(exc).__name__}", "blocker": str(exc)[:500], "promotion_allowed": 0})
    pfx = prefix(str(args.scope))
    safe = slug(str(args.run_label or args.scope))
    suffix = f"_{safe}_fc{int(args.shard_index)}"
    write_rows(out_dir / f"{pfx}_matrix{suffix}.csv", rows)
    write_rows(out_dir / f"{pfx}_traces{suffix}.csv", trace_rows)
    append_exec(out_dir, f"{PYTHON} experiments/run_v21_mlp_source_dynamics.py --scope {args.scope} --shard-index {args.shard_index}", status="completed", note=f"rows={len(rows)} traces={len(trace_rows)}")


def merge(args: argparse.Namespace) -> None:
    out_dir = ensure_out(args.out_dir)
    pfx = prefix(str(args.scope))
    rows = merge_csvs(out_dir, f"{pfx}_matrix*_fc*.csv", f"{pfx}_raw_matrix.csv")
    traces = merge_csvs(out_dir, f"{pfx}_traces*_fc*.csv", f"{pfx}_raw_traces.csv")
    trace_by_job: dict[str, dict[str, dict[str, Any]]] = {}
    for tr in traces:
        step = str(tr.get("step"))
        if step in {"800", "1600", "3200", "4800"}:
            trace_by_job.setdefault(f"{tr.get('run_label')}:{tr.get('job_index')}", {})[step] = tr
    enriched = []
    for row in rows:
        item = dict(row)
        hrows = trace_by_job.get(f"{row.get('run_label')}:{row.get('job_index')}", {})
        for step in ["800", "1600", "3200", "4800"]:
            tr = hrows.get(step, {})
            for key in [
                "val_loss",
                "train_loss",
                "CEp99",
                "NLL",
                "ECE",
                "Brier",
                "LineC_channel_loss",
                "LineC_fast_loss",
                "ActuationR2",
                "B1_gain",
                "B2_transfer_gain",
                "B3_safety_gain",
                "source_state_gate_accept",
                "source_state_current_cos",
                "generalization_gate_accept",
            ]:
                if key in tr:
                    item[f"{key}_h{step}"] = tr.get(key)
        enriched.append(item)
    enriched = enrich_sources(enriched)
    write_rows(out_dir / f"{pfx}_matrix.csv", enriched)
    write_rows(out_dir / f"{pfx}_traces.csv", traces)
    summary_input = [r for r in enriched if "smoke" not in str(r.get("run_label", "")).lower()]
    summary = grouped_source_retention(summary_input, group_keys=("carrier", "basis_repair_variant", "v21_id"))
    by_id = {str(r.get("v21_id")): r for r in enriched}
    for item in summary:
        sample = by_id.get(str(item.get("v21_id")), {})
        item["continuation_id"] = sample.get("continuation_id", "")
        item["mechanism"] = sample.get("mechanism", item.get("mechanism", ""))
        item["hypothesis"] = sample.get("hypothesis", "")
    target = f"{pfx}.csv" if args.scope == "mlp_source" else "v21_kan_source_writer_matrix.csv"
    write_rows(out_dir / target, summary)
    if args.scope == "mlp_source":
        write_rows(out_dir / "v21_optimizer_washout_matrix.csv", [r for r in summary if "M2" in str(r.get("v21_id")) or "F9" in str(r.get("v21_id")) or "F10" in str(r.get("v21_id")) or "Anchor" in str(r.get("v21_id"))])
    else:
        write_rows(out_dir / "v21_matrix_block_fu_matrix.csv", [r for r in summary if "matrix" in str(r.get("hypothesis")).lower() or "KSW3" in str(r.get("v21_id"))])
        target_raw = [r for r in summary_input if str(r.get("v21_id", "")).startswith("F3-T")]
        target_smoke_raw = [r for r in enriched if str(r.get("v21_id", "")).startswith("F3-T") and "smoke" in str(r.get("run_label", "")).lower()]
        target_summary = [r for r in summary if str(r.get("v21_id", "")).startswith("F3-T")]
        target_smoke_summary = grouped_source_retention(target_smoke_raw, group_keys=("carrier", "basis_repair_variant", "v21_id"))
        write_rows(out_dir / "v21_function_space_target_source_raw_matrix.csv", target_raw)
        write_rows(out_dir / "v21_function_space_target_source_matrix.csv", target_summary)
        write_rows(out_dir / "v21_function_space_target_source_smoke_raw_matrix.csv", target_smoke_raw)
        write_rows(out_dir / "v21_function_space_target_source_smoke_matrix.csv", target_smoke_summary)
    append_exec(out_dir, f"{PYTHON} experiments/run_v21_mlp_source_dynamics.py --scope {args.scope} --merge-only", status="completed", note=f"rows={len(enriched)} grouped={len(summary)}")


def main_with_scope(default_scope: str) -> None:
    args = parser(default_scope).parse_args()
    if args.merge_only:
        merge(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main_with_scope("mlp_source")
