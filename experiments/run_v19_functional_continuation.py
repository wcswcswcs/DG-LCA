#!/usr/bin/env python3
"""Targeted v19 functional continuation for MLP source-retention washout."""

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

from experiments.run_v17_common import (  # noqa: E402
    CONTROL_MECHANISMS,
    append_log,
    finite_float,
    load_dataset,
    now_sg,
    resolve_device,
    train_one,
    carrier_model,
    write_rows,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
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
    p.add_argument("--init-seed-offset", type=int, default=0)
    p.add_argument("--sanity-eps", type=float, default=1.0e-3)
    p.add_argument("--shard-count", type=int, default=4)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--run-label", default="")
    p.add_argument("--spec-ids", default="")
    return p


def continuation_specs() -> list[dict[str, Any]]:
    controls = [
        {"continuation_id": f"CTRL-{name}", "mechanism": name, "fu_lr": 0.001, "alt_period": 10}
        for name in ["CTRL-AdamW", "CTRL-SGD", "CTRL-RecoveryOnly", "CTRL-NoOpMatchedOverhead", "CTRL-RandomMatchedNorm", "CTRL-RandomSameRankBlock"]
    ]
    probes = [
        {"continuation_id": "M2-fu0p0005", "mechanism": "M2-SGDMomentumPrimaryFU", "fu_lr": 0.0005, "alt_period": 10},
        {"continuation_id": "M2-fu0p00025", "mechanism": "M2-SGDMomentumPrimaryFU", "fu_lr": 0.00025, "alt_period": 10},
        {"continuation_id": "M5-alt20-fu0p0005", "mechanism": "M5-AlternatingFUGradient", "fu_lr": 0.0005, "alt_period": 20},
        {"continuation_id": "M5-alt50-fu0p0005", "mechanism": "M5-AlternatingFUGradient", "fu_lr": 0.0005, "alt_period": 50},
        {"continuation_id": "M5-alt100-fu0p0005", "mechanism": "M5-AlternatingFUGradient", "fu_lr": 0.0005, "alt_period": 100},
        {"continuation_id": "M5-alt100-fu0p00025", "mechanism": "M5-AlternatingFUGradient", "fu_lr": 0.00025, "alt_period": 100},
        {"continuation_id": "M5-alt200-fu0p0005", "mechanism": "M5-AlternatingFUGradient", "fu_lr": 0.0005, "alt_period": 200},
        {"continuation_id": "M6-slowstate-fu0p0005", "mechanism": "M6-SlowStateFU", "fu_lr": 0.0005, "alt_period": 10},
        {"continuation_id": "M6-slowstate-fu0p0001", "mechanism": "M6-SlowStateFU", "fu_lr": 0.0001, "alt_period": 10},
        {"continuation_id": "M15-linec-alt50-fu0p0005", "mechanism": "M15-LineCFilteredAlternatingFU", "fu_lr": 0.0005, "alt_period": 50},
        {"continuation_id": "M15-linec-alt100-fu0p0005", "mechanism": "M15-LineCFilteredAlternatingFU", "fu_lr": 0.0005, "alt_period": 100},
        {"continuation_id": "M16-warm800-alt50-fu0p0001", "mechanism": "M16-TwoPhaseMomentumThenLineCFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800},
        {"continuation_id": "M16-warm800-alt100-fu0p0001", "mechanism": "M16-TwoPhaseMomentumThenLineCFU", "fu_lr": 0.0001, "alt_period": 100, "source_warmup_steps": 800},
        {"continuation_id": "M16-warm1200-alt100-fu0p0001", "mechanism": "M16-TwoPhaseMomentumThenLineCFU", "fu_lr": 0.0001, "alt_period": 100, "source_warmup_steps": 1200},
        {"continuation_id": "M16-warm800-alt100-fu0p00005", "mechanism": "M16-TwoPhaseMomentumThenLineCFU", "fu_lr": 0.00005, "alt_period": 100, "source_warmup_steps": 800},
        {"continuation_id": "M17-readout-warm800-alt50-fu0p0001", "mechanism": "M17-ReadoutCarrierTwoPhaseLineCFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800},
        {"continuation_id": "M17-readout-warm800-alt100-fu0p0001", "mechanism": "M17-ReadoutCarrierTwoPhaseLineCFU", "fu_lr": 0.0001, "alt_period": 100, "source_warmup_steps": 800},
        {"continuation_id": "M17-readout-warm800-alt50-fu0p00005", "mechanism": "M17-ReadoutCarrierTwoPhaseLineCFU", "fu_lr": 0.00005, "alt_period": 50, "source_warmup_steps": 800},
        {"continuation_id": "M17-readout-warm1200-alt50-fu0p0001", "mechanism": "M17-ReadoutCarrierTwoPhaseLineCFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 1200},
        {"continuation_id": "M13-lowrank-r4-fu0p0005", "mechanism": "M13-LowRankMatrixBlockFU", "fu_lr": 0.0005, "alt_period": 10},
        {"continuation_id": "M13-lowrank-r4-fu0p0001", "mechanism": "M13-LowRankMatrixBlockFU", "fu_lr": 0.0001, "alt_period": 10},
        {"continuation_id": "M18-lowrank-r4-fu0p0001", "mechanism": "M18-CarrierLowRankBlockFU", "fu_lr": 0.0001, "alt_period": 10},
        {"continuation_id": "M18-lowrank-r4-fu0p00005", "mechanism": "M18-CarrierLowRankBlockFU", "fu_lr": 0.00005, "alt_period": 10},
        {"continuation_id": "M18-lowrank-r4-fu0p00025", "mechanism": "M18-CarrierLowRankBlockFU", "fu_lr": 0.00025, "alt_period": 10},
        {"continuation_id": "M19-split-lowrank-r4-fu0p0001", "mechanism": "M19-SplitConsensusLowRankBlockFU", "fu_lr": 0.0001, "alt_period": 10},
        {"continuation_id": "M19-split-lowrank-r4-fu0p00005", "mechanism": "M19-SplitConsensusLowRankBlockFU", "fu_lr": 0.00005, "alt_period": 10},
        {"continuation_id": "M19-split-lowrank-r4-fu0p00025", "mechanism": "M19-SplitConsensusLowRankBlockFU", "fu_lr": 0.00025, "alt_period": 10},
        {"continuation_id": "M20-h4-actuation-r4-fu0p0001", "mechanism": "M20-TrainSplitFunctionSpaceActuationFU", "fu_lr": 0.0001, "alt_period": 10},
        {"continuation_id": "M20-h4-actuation-r4-fu0p00005", "mechanism": "M20-TrainSplitFunctionSpaceActuationFU", "fu_lr": 0.00005, "alt_period": 10},
        {"continuation_id": "M20-h4-actuation-r4-fu0p00025", "mechanism": "M20-TrainSplitFunctionSpaceActuationFU", "fu_lr": 0.00025, "alt_period": 10},
        {"continuation_id": "M21-exact-readout-fu0p0001", "mechanism": "M21-ExactReadoutFunctionSpaceActuationFU", "fu_lr": 0.0001, "alt_period": 10},
        {"continuation_id": "M22-exact-readout-highcap-fu0p0001", "mechanism": "M22-ExactReadoutHighCapScheduledFU", "fu_lr": 0.0001, "alt_period": 10},
        {"continuation_id": "M23-exact-readout-ultracap-fu0p0001", "mechanism": "M23-ExactReadoutUltraCapScheduledFU", "fu_lr": 0.0001, "alt_period": 10},
        {"continuation_id": "M22-exact-readout-highcap-alt50-fu0p0001", "mechanism": "M22-ExactReadoutHighCapScheduledFU", "fu_lr": 0.0001, "alt_period": 50},
        {"continuation_id": "M23-exact-readout-ultracap-alt50-fu0p0001", "mechanism": "M23-ExactReadoutUltraCapScheduledFU", "fu_lr": 0.0001, "alt_period": 50},
        {"continuation_id": "M24-warm400-ultracap-alt50-fu0p0001", "mechanism": "M24-WarmupExactReadoutUltraCapFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400},
        {"continuation_id": "M24-warm800-ultracap-alt50-fu0p0001", "mechanism": "M24-WarmupExactReadoutUltraCapFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800},
        {"continuation_id": "M25-adamw-ultracap-alt50-fu0p0001", "mechanism": "M25-AdamWExactReadoutUltraCapFU", "fu_lr": 0.0001, "alt_period": 50},
        {"continuation_id": "M25-adamw-ultracap-alt100-fu0p0001", "mechanism": "M25-AdamWExactReadoutUltraCapFU", "fu_lr": 0.0001, "alt_period": 100},
        {"continuation_id": "M26-gated-adamw-ultracap-alt100-fu0p0001", "mechanism": "M26-GatedAdamWExactReadoutFU", "fu_lr": 0.0001, "alt_period": 100},
        {"continuation_id": "M27-multibatch-gated-alt50-fu0p0001", "mechanism": "M27-MultiBatchGeneralizationGatedReadoutFU", "fu_lr": 0.0001, "alt_period": 50},
        {"continuation_id": "M27-multibatch-gated-alt100-fu0p0001", "mechanism": "M27-MultiBatchGeneralizationGatedReadoutFU", "fu_lr": 0.0001, "alt_period": 100},
        {"continuation_id": "M27-multibatch-gated-alt100-fu0p00005", "mechanism": "M27-MultiBatchGeneralizationGatedReadoutFU", "fu_lr": 0.00005, "alt_period": 100},
        {"continuation_id": "M27-multibatch-gated-alt200-fu0p0001", "mechanism": "M27-MultiBatchGeneralizationGatedReadoutFU", "fu_lr": 0.0001, "alt_period": 200},
        {"continuation_id": "M27-multibatch-gated-alt200-fu0p00005", "mechanism": "M27-MultiBatchGeneralizationGatedReadoutFU", "fu_lr": 0.00005, "alt_period": 200},
        {"continuation_id": "M28-adamw-multibatch-gated-alt100-fu0p0001", "mechanism": "M28-AdamWMultiBatchGatedReadoutFU", "fu_lr": 0.0001, "alt_period": 100},
        {"continuation_id": "M28-adamw-multibatch-gated-alt100-fu0p00005", "mechanism": "M28-AdamWMultiBatchGatedReadoutFU", "fu_lr": 0.00005, "alt_period": 100},
        {"continuation_id": "M28-adamw-multibatch-gated-alt200-fu0p0001", "mechanism": "M28-AdamWMultiBatchGatedReadoutFU", "fu_lr": 0.0001, "alt_period": 200},
        {"continuation_id": "M29-consensus-alt50-fu0p0001", "mechanism": "M29-ConsensusSourceStateFU", "fu_lr": 0.0001, "alt_period": 50},
        {"continuation_id": "M29-consensus-alt100-fu0p0001", "mechanism": "M29-ConsensusSourceStateFU", "fu_lr": 0.0001, "alt_period": 100},
        {"continuation_id": "M29-consensus-alt100-fu0p00005", "mechanism": "M29-ConsensusSourceStateFU", "fu_lr": 0.00005, "alt_period": 100},
        {"continuation_id": "M30-lowthreshold-consensus-alt50-fu0p0001", "mechanism": "M30-LowThresholdConsensusSourceStateFU", "fu_lr": 0.0001, "alt_period": 50},
        {"continuation_id": "M30-lowthreshold-consensus-alt100-fu0p0001", "mechanism": "M30-LowThresholdConsensusSourceStateFU", "fu_lr": 0.0001, "alt_period": 100},
        {"continuation_id": "M30-lowthreshold-consensus-alt100-fu0p00005", "mechanism": "M30-LowThresholdConsensusSourceStateFU", "fu_lr": 0.00005, "alt_period": 100},
        {"continuation_id": "M31-adamw-consensus-alt50-fu0p0001", "mechanism": "M31-AdamWLowThresholdConsensusResidualFU", "fu_lr": 0.0001, "alt_period": 50},
        {"continuation_id": "M31-adamw-consensus-alt100-fu0p0001", "mechanism": "M31-AdamWLowThresholdConsensusResidualFU", "fu_lr": 0.0001, "alt_period": 100},
        {"continuation_id": "M31-adamw-consensus-alt100-fu0p00005", "mechanism": "M31-AdamWLowThresholdConsensusResidualFU", "fu_lr": 0.00005, "alt_period": 100},
        {"continuation_id": "M32-postadamw-consensus-alt50-fu0p0001", "mechanism": "M32-PostAdamWConsensusResidualFU", "fu_lr": 0.0001, "alt_period": 50},
        {"continuation_id": "M32-postadamw-consensus-alt100-fu0p0001", "mechanism": "M32-PostAdamWConsensusResidualFU", "fu_lr": 0.0001, "alt_period": 100},
        {"continuation_id": "M32-postadamw-consensus-alt100-fu0p00005", "mechanism": "M32-PostAdamWConsensusResidualFU", "fu_lr": 0.00005, "alt_period": 100},
        {"continuation_id": "M2-fu0p0001", "mechanism": "M2-SGDMomentumPrimaryFU", "fu_lr": 0.0001, "alt_period": 10},
        {"continuation_id": "M2-fu0p00005", "mechanism": "M2-SGDMomentumPrimaryFU", "fu_lr": 0.00005, "alt_period": 10},
        {"continuation_id": "M5-alt20-fu0p00025", "mechanism": "M5-AlternatingFUGradient", "fu_lr": 0.00025, "alt_period": 20},
        {"continuation_id": "M11-dual-fu0p0005", "mechanism": "M11-DualMemorySlowStateFU", "fu_lr": 0.0005, "alt_period": 10},
        {"continuation_id": "M12-schedulefree-fu0p0005", "mechanism": "M12-ScheduleFreeAveragedFU", "fu_lr": 0.0005, "alt_period": 10},
        {"continuation_id": "M14-popriskslow-fu0p0005", "mechanism": "M14-SourceChannelPopRiskSlowFU", "fu_lr": 0.0005, "alt_period": 10},
    ]
    return controls + probes


def selected_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = continuation_specs()
    wanted = {x.strip() for x in str(getattr(args, "spec_ids", "")).split(",") if x.strip()}
    if wanted:
        specs = [
            s
            for s in specs
            if str(s.get("continuation_id")) in wanted or str(s.get("mechanism")) in wanted
        ]
    return specs


def jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    carriers = [x.strip() for x in str(getattr(args, "carriers", "MLP")).split(",") if x.strip()]
    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    out = []
    idx = 0
    for carrier in carriers:
        for dataset in datasets:
            for seed in seeds:
                for spec in selected_specs(args):
                    out.append(
                        {
                            "run_label": str(getattr(args, "run_label", "")),
                            "job_index": idx,
                            "dataset": dataset,
                            "seed": seed,
                            "carrier": carrier,
                            "basis_repair_variant": str(getattr(args, "basis_repair_variant", "R0-current")),
                            "init_seed_offset": int(getattr(args, "init_seed_offset", 0)),
                            **spec,
                        }
                    )
                    idx += 1
    return out


def enrich_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    control_best: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if str(row.get("mechanism")) not in CONTROL_MECHANISMS:
            continue
        key_base = (
            str(row.get("carrier", "MLP")),
            str(row.get("basis_repair_variant", "R0-current") or "R0-current"),
            str(row.get("dataset")),
            str(row.get("seed")),
        )
        for h in ["h800", "h1600", "h3200", "h4800"]:
            val = finite_float(row.get(f"val_loss_{h}"))
            if val == val:
                key = (*key_base, h)
                control_best[key] = min(control_best.get(key, float("inf")), val)

    enriched = []
    for row in rows:
        item = dict(row)
        for h in ["h800", "h1600", "h3200", "h4800"]:
            val = finite_float(item.get(f"val_loss_{h}"))
            best = control_best.get(
                (
                    str(item.get("carrier", "MLP")),
                    str(item.get("basis_repair_variant", "R0-current") or "R0-current"),
                    str(item.get("dataset")),
                    str(item.get("seed")),
                    h,
                ),
                float("nan"),
            )
            item[f"source_{h}"] = best - val if val == val and best == best else ""
        enriched.append(item)
    return enriched


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = enrich_sources(rows)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in enriched:
        if str(row.get("mechanism")) in CONTROL_MECHANISMS:
            continue
        grouped.setdefault(
            (
                str(row.get("carrier", "MLP")),
                str(row.get("basis_repair_variant", "R0-current") or "R0-current"),
                str(row.get("continuation_id")),
            ),
            [],
        ).append(row)
    summary = []
    for (carrier, repair_variant, cid), group in sorted(grouped.items()):
        item: dict[str, Any] = {"carrier": carrier, "basis_repair_variant": repair_variant, "continuation_id": cid, "mechanism": group[0].get("mechanism"), "rows": len(group)}
        for h in ["h800", "h1600", "h3200", "h4800"]:
            vals = [finite_float(r.get(f"source_{h}")) for r in group]
            vals = [v for v in vals if v == v]
            item[f"source_{h}_mean"] = sum(vals) / len(vals) if vals else ""
            item[f"source_{h}_pass_count"] = sum(1 for v in vals if v >= 0.005)
        h800 = finite_float(item.get("source_h800_mean"))
        h1600 = finite_float(item.get("source_h1600_mean"))
        h3200 = finite_float(item.get("source_h3200_mean"))
        h4800 = finite_float(item.get("source_h4800_mean"))
        item["retention_h1600_over_h800"] = max(0.0, h1600 / h800) if h800 == h800 and h800 > 0.0 and h1600 == h1600 else ""
        item["retention_h3200_over_h1600"] = max(0.0, h3200 / h1600) if h1600 == h1600 and h1600 > 0.0 and h3200 == h3200 else ""
        item["retention_h4800_over_h3200"] = max(0.0, h4800 / h3200) if h3200 == h3200 and h3200 > 0.0 and h4800 == h4800 else ""
        item["retained_candidate"] = int(
            finite_float(item.get("source_h800_mean"), -999.0) >= 0.005
            and finite_float(item.get("retention_h1600_over_h800"), 0.0) >= 0.50
            and finite_float(item.get("retention_h3200_over_h1600"), 0.0) >= 0.50
        )
        summary.append(item)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        rows: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []
        for p in sorted(out_dir.glob("v19_functional_continuation_matrix*_fc*.csv")):
            import csv

            with p.open(newline="", encoding="utf-8") as f:
                rows.extend(list(csv.DictReader(f)))
        for p in sorted(out_dir.glob("v19_functional_continuation_traces*_fc*.csv")):
            import csv

            with p.open(newline="", encoding="utf-8") as f:
                traces.extend(list(csv.DictReader(f)))
        trace_by_job: dict[str, dict[str, dict[str, Any]]] = {}
        for tr in traces:
            step = str(tr.get("step"))
            if step not in {"800", "1600", "3200", "4800"}:
                continue
            trace_by_job.setdefault(f"{tr.get('run_label', '')}:{tr.get('job_index')}", {})[step] = tr
        enriched = []
        for row in rows:
            item = dict(row)
            horizons = trace_by_job.get(f"{row.get('run_label', '')}:{row.get('job_index')}", {})
            for step in ["800", "1600", "3200", "4800"]:
                tr = horizons.get(step, {})
                for key in ["val_loss", "train_loss", "CEp99", "NLL", "ECE", "Brier", "LineC_channel_loss", "LineC_fast_loss"]:
                    if key in tr:
                        item[f"{key}_h{step}"] = tr.get(key)
            enriched.append(item)
        rows = enrich_sources(enriched)
        write_rows(out_dir / "v19_functional_continuation_matrix.csv", rows)
        write_rows(out_dir / "v19_functional_continuation_traces.csv", traces)
        write_rows(out_dir / "v19_functional_continuation_summary.csv", summarize(rows))
        append_log(out_dir, f"## {now_sg()} v19 functional continuation merge rows={len(rows)} traces={len(traces)}")
        return
    device = resolve_device(args.device)
    selected = [j for j in jobs(args) if int(j["job_index"]) % int(args.shard_count) == int(args.shard_index)]
    append_log(out_dir, f"## {now_sg()} v19 functional continuation shard {args.shard_index}/{args.shard_count} device={device} jobs={len(selected)}")
    rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for job in selected:
        started = time.perf_counter()
        try:
            x_train, y_train, x_val, y_val = load_dataset(str(job["dataset"]), Path(args.data_root), int(args.train_size), int(args.val_size), int(job["seed"]), device, int(args.input_size))
            local_args = deepcopy(args)
            local_args.fu_lr = float(job["fu_lr"])
            local_args.alt_period = int(job["alt_period"])
            local_args.source_warmup_steps = int(job.get("source_warmup_steps", getattr(args, "source_warmup_steps", 0)))
            local_args.basis_repair_variant = str(job.get("basis_repair_variant", getattr(args, "basis_repair_variant", "R0-current")))
            train_seed = 190_000 + int(job["job_index"]) + int(job.get("init_seed_offset", 0) or 0)
            model = carrier_model(str(job["carrier"]), x_train, train_seed, local_args, device)
            final, traces, _first_update = train_one(model, str(job["mechanism"]), x_train, y_train, x_val, y_val, local_args, seed=train_seed)
            row = {**job, **final, "runtime_sec": time.perf_counter() - started, "execution_status": "measured", "blocker": "", "promotion_allowed": 0}
            rows.append(row)
            for tr in traces:
                trace_rows.append({**job, **tr})
        except Exception as exc:
            rows.append({**job, "runtime_sec": time.perf_counter() - started, "execution_status": f"blocked:{type(exc).__name__}", "blocker": str(exc)[:500], "promotion_allowed": 0})
    label = str(getattr(args, "run_label", "")).strip()
    safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)
    suffix = f"_{safe_label}_fc{int(args.shard_index)}" if safe_label else f"_fc{int(args.shard_index)}"
    write_rows(out_dir / f"v19_functional_continuation_matrix{suffix}.csv", rows)
    write_rows(out_dir / f"v19_functional_continuation_traces{suffix}.csv", trace_rows)


if __name__ == "__main__":
    main()
