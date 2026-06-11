#!/usr/bin/env python3
"""v21 basis-kernel efficiency officialization profiling runner."""

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
    carrier_model,
    load_dataset,
    loss_value,
    make_update,
    resolve_device,
    same_param_mlp_factory,
)
from experiments.run_v21_common import (  # noqa: E402
    PYTHON,
    V21_EFFICIENCY_VARIANTS,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    merge_csvs,
    to_v19_repair_variant,
    write_rows,
)
from dgkan.profiling.efficiency_v20 import efficiency_waterfall_rows, profile_isolated  # noqa: E402
from dgkan.profiling.efficiency_v21 import v21_efficiency_gate  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data-root", default="data")
    p.add_argument("--families", default="D-CHE,D-FOU,LQ,D-RAT,D-RBF,D-WAV")
    p.add_argument("--batches", default="8,32,128,256")
    p.add_argument("--train-size", type=int, default=256)
    p.add_argument("--val-size", type=int, default=64)
    p.add_argument("--input-size", type=int, default=8)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--param-budget", type=int, default=12000)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--weight-decay", type=float, default=0.001)
    p.add_argument("--fu-lr", type=float, default=0.001)
    p.add_argument("--profiler-repeats", type=int, default=3)
    p.add_argument("--profiler-warmup", type=int, default=1)
    p.add_argument("--shard-count", type=int, default=4)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--merge-only", action="store_true")
    return p


def jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    families = [x.strip() for x in str(args.families).split(",") if x.strip()]
    batches = [int(x.strip()) for x in str(args.batches).split(",") if x.strip()]
    out = []
    idx = 0
    for family in families:
        for variant in V21_EFFICIENCY_VARIANTS.get(family, ["R0-current"]):
            for batch in batches:
                out.append({"job_index": idx, "carrier": family, "kernel_variant": variant, "repair_variant": to_v19_repair_variant(family, variant), "batch_size": batch})
                idx += 1
    return out


def manual_flags(row: dict[str, Any], variant: str) -> dict[str, Any]:
    manual_variant = str(row.get("manual_kernel_variant", "") or row.get("manual_kernel_variant_profiled", ""))
    manual_profiled = int_flag(row.get("manual_ce_train_stream_profiled"))
    manual_correct = int_flag(row.get("manual_correctness_pass")) if manual_profiled else 0
    official_fused = int("triton" in manual_variant and manual_correct)
    no_materialize = int(manual_profiled and ("triton" in manual_variant or "stream" in str(variant).lower() or "no-materialize" in str(variant).lower() or "gradbuf" in str(variant).lower()))
    return {
        "manual_ce_train_stream_profiled": manual_profiled,
        "manual_kernel_variant_profiled": row.get("manual_kernel_variant_profiled", ""),
        "gradcheck_pass": manual_correct,
        "official_fused_kernel_complete": official_fused,
        "no_materialize_complete": no_materialize,
        "full_loop_timing_pass": int_flag(row.get("phase_level_timing_present")),
        "audit_overhead_ratio": row.get("audit_overhead_ratio", ""),
    }


def run_shard(args: argparse.Namespace) -> None:
    out_dir = ensure_out(args.out_dir)
    device = resolve_device(args.device)
    selected = [j for j in jobs(args) if int(j["job_index"]) % int(args.shard_count) == int(args.shard_index)]
    append_exec(out_dir, f"{PYTHON} experiments/run_v21_efficiency_officialization.py --shard-index {args.shard_index} --shard-count {args.shard_count} --device {args.device}", status="started", note=f"jobs={len(selected)}")
    rows = []
    waterfall = []
    grad_rows = []
    for job in selected:
        start = time.perf_counter()
        try:
            load_n = max(int(args.train_size), int(job["batch_size"]))
            x_train, y_train, x_val, y_val = load_dataset("MNIST", Path(args.data_root), load_n, int(args.val_size), 21_100 + int(job["job_index"]), device, int(args.input_size))
            xb = x_train[: int(job["batch_size"])]
            yb = y_train[: int(job["batch_size"])]
            local = deepcopy(args)
            local.classes = int(args.classes)
            local.hidden = int(args.hidden)
            local.param_budget = int(args.param_budget)
            local.input_size = int(args.input_size)
            local.basis_repair_variant = str(job["repair_variant"])
            seed = 21_100 + int(job["job_index"])
            base = carrier_model(str(job["carrier"]), x_train, seed, local, device)
            same_hidden, mlp_factory = same_param_mlp_factory(base, int(x_train.shape[1]), int(args.classes), seed, device)

            def model_factory(family=str(job["carrier"]), s=seed):
                return carrier_model(family, x_train, s, local, device)

            def update_factory(model):
                return make_update(model, "M3-FUPrimary", xb, yb, seed=seed)

            def audit(model):
                return loss_value(model, x_val, y_val)

            manual_audit: dict[str, Any] = {}
            manual_variant = ""
            if hasattr(base, "manual_kernel_variant"):
                manual_variant = str(base.manual_kernel_variant())  # type: ignore[attr-defined]
            if hasattr(base, "manual_gradient_audit"):
                try:
                    manual_audit = base.manual_gradient_audit(xb[: min(16, int(xb.shape[0]))], yb[: min(16, int(yb.shape[0]))])  # type: ignore[attr-defined]
                except Exception as exc:
                    manual_audit = {"manual_correctness_blocker": f"{type(exc).__name__}:{exc}"}

            prof = profile_isolated(
                model_factory,
                mlp_factory,
                xb,
                yb,
                update_factory,
                audit,
                device=device,
                repeats=int(args.profiler_repeats),
                warmup=int(args.profiler_warmup),
                lr=float(args.lr),
                use_manual_ce=True,
            )
            grad_relerr = finite_float(manual_audit.get("grad_relerr_max"), 999.0)
            grad_cos = finite_float(manual_audit.get("grad_cos_min"), -1.0)
            output_err = finite_float(manual_audit.get("output_max_abs_error"), 999.0)
            row = {
                **job,
                **prof,
                "same_param_mlp_hidden": same_hidden,
                "manual_kernel_variant": manual_variant,
                "manual_grad_relerr_max": manual_audit.get("grad_relerr_max", ""),
                "manual_grad_cos_min": manual_audit.get("grad_cos_min", ""),
                "manual_output_max_abs_error": manual_audit.get("output_max_abs_error", ""),
                "manual_correctness_pass": int(grad_relerr < 1.0e-4 and grad_cos > 0.999 and output_err < 1.0e-4),
                "manual_correctness_blocker": manual_audit.get("manual_correctness_blocker", ""),
                "forward_ratio_vs_mlp": prof.get("forward_ratio", ""),
                "backward_ratio_vs_mlp": prof.get("backward_ratio", ""),
                "step_ratio_vs_mlp": prof.get("training_step_ratio", ""),
                "memory_ratio_vs_mlp": prof.get("memory_ratio", ""),
                "full_loop_ratio_vs_mlp": prof.get("full_loop_ratio", prof.get("training_step_ratio", "")),
                "execution_status": "measured",
                "runtime_sec": time.perf_counter() - start,
                "promotion_allowed": 0,
            }
            row.update(manual_flags(row, str(job["kernel_variant"])))
            row.update(v21_efficiency_gate(row))
            rows.append(row)
            waterfall.extend(efficiency_waterfall_rows(row))
            grad_rows.append(
                {
                    "carrier": job["carrier"],
                    "kernel_variant": job["kernel_variant"],
                    "batch_size": job["batch_size"],
                    "manual_kernel_variant": manual_variant,
                    "grad_relerr_max": row["manual_grad_relerr_max"],
                    "grad_cos_min": row["manual_grad_cos_min"],
                    "output_max_abs_error": row["manual_output_max_abs_error"],
                    "gradcheck_pass": row["gradcheck_pass"],
                    "blocker": row["manual_correctness_blocker"],
                }
            )
        except Exception as exc:
            rows.append({**job, "execution_status": f"blocked:{type(exc).__name__}", "blocker": str(exc)[:500], "runtime_sec": time.perf_counter() - start, "v21_exploration_gate": 0, "v21_official_like_gate": 0, "v21_efficiency_blocker": f"Exception:{type(exc).__name__}", "promotion_allowed": 0})
    suffix = f"_b{int(args.shard_index)}"
    write_rows(out_dir / f"v21_efficiency_truth_table{suffix}.csv", rows)
    write_rows(out_dir / f"v21_efficiency_waterfall{suffix}.csv", waterfall)
    write_rows(out_dir / f"v21_kernel_gradcheck{suffix}.csv", grad_rows)
    append_exec(out_dir, f"{PYTHON} experiments/run_v21_efficiency_officialization.py --shard-index {args.shard_index}", status="completed", note=f"rows={len(rows)}")


def merge(args: argparse.Namespace) -> None:
    out_dir = ensure_out(args.out_dir)
    rows = []
    for row in merge_csvs(out_dir, "v21_efficiency_truth_table_b*.csv", "v21_efficiency_truth_table.csv"):
        row.update(v21_efficiency_gate(row))
        rows.append(row)
    write_rows(out_dir / "v21_efficiency_truth_table.csv", rows)
    waterfall = merge_csvs(out_dir, "v21_efficiency_waterfall_b*.csv", "v21_efficiency_waterfall.csv")
    grad = merge_csvs(out_dir, "v21_kernel_gradcheck_b*.csv", "v21_kernel_gradcheck.csv")
    summary = []
    for family in sorted({str(r.get("carrier")) for r in rows}):
        fam_rows = [r for r in rows if str(r.get("carrier")) == family]
        fwd = [finite_float(r.get("forward_ratio_vs_mlp")) for r in fam_rows if finite_float(r.get("forward_ratio_vs_mlp")) == finite_float(r.get("forward_ratio_vs_mlp"))]
        step = [finite_float(r.get("step_ratio_vs_mlp")) for r in fam_rows if finite_float(r.get("step_ratio_vs_mlp")) == finite_float(r.get("step_ratio_vs_mlp"))]
        summary.append(
            {
                "carrier": family,
                "rows": len(fam_rows),
                "exploration_pass_rows": sum(int_flag(r.get("v21_exploration_gate")) for r in fam_rows),
                "official_like_pass_rows": sum(int_flag(r.get("v21_official_like_gate")) for r in fam_rows),
                "best_forward_ratio": min(fwd or [999.0]),
                "best_step_ratio": min(step or [999.0]),
            }
        )
    write_rows(out_dir / "v21_efficiency_officialization_summary.csv", summary)
    append_exec(out_dir, f"{PYTHON} experiments/run_v21_efficiency_officialization.py --merge-only", status="completed", note=f"rows={len(rows)} waterfall={len(waterfall)} grad={len(grad)}")


def main() -> None:
    args = parser().parse_args()
    if args.merge_only:
        merge(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
