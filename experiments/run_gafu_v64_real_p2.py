#!/usr/bin/env python3
"""DG-KAN v6.4 real-only P2 training runner.

This runner exists because the historical v6.4 script mixed a real P2 training
path with later proxy/derived stages.  This file keeps only the empirical part:
real dataset loading, real optimizer steps, and raw per-step traces.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from dgkan_core import ensure_dir, get_device, load_vision_bundle, parse_int_list, parse_str_list, write_csv
from run_gafu_v3 import dataset_name
from run_gafu_v64_real import P2_OPTIMIZERS
from run_gafu_v63 import V63Params, _decorate_task_rows, _train_manual_candidate, _train_mlp_reference, _wandb_finish, _wandb_init, _wandb_log_row


REAL_METHOD = "DWM2-poly2-compiled"


def _git_status() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], text=True).strip()
    except Exception as exc:
        return f"git_status_unavailable: {exc!r}"


def _write_manifest(out_dir: Path, args: argparse.Namespace, started: float, finished: float) -> None:
    command_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest: dict[str, Any] = {
        "provenance": "EMPIRICAL_REAL_TRAINING",
        "script": "experiments/run_gafu_v64_real_p2.py",
        "historical_note": "Replaces the real P2 portion of blocked mixed/proxy run_gafu_v64.py.",
        "command_args": command_args,
        "started_unix": started,
        "finished_unix": finished,
        "duration_sec": finished - started,
        "git_status_short": _git_status(),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(wandb=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/real_rerun_20260504/v64_p2_real"))
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-size", type=int, default=1536)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--optimizers", default=",".join(P2_OPTIMIZERS))
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--wandb-project", default="DG-KAN")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-group", default="v64-real-p2-diagnostic-20260504")
    parser.add_argument("--wandb-name-prefix", default="v64-real-p2-diagnostic")
    parser.add_argument("--no-wandb", action="store_false", dest="wandb")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    out_dir = ensure_dir(args.out_dir)
    _wandb_init(args)
    rows: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    device = get_device(args.device)
    params = V63Params(
        p3_steps=int(args.steps),
        train_size=int(args.train_size),
        val_size=int(args.val_size),
        test_size=int(args.test_size),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
    )
    datasets = parse_str_list(args.datasets)
    seeds = parse_int_list(args.seeds)
    optimizers = parse_str_list(args.optimizers)

    for ds in datasets:
        canonical = dataset_name(ds)
        for seed in seeds:
            bundle = load_vision_bundle(
                canonical,
                data_root=args.data_root,
                train_size=params.train_size,
                val_size=params.val_size,
                test_size=params.test_size,
                seed=seed,
                allow_fake_data=False,
            )
            try:
                mlp_final, mlp_trace = _train_mlp_reference(bundle, seed, params, device, canonical, "P2_REAL_DIAGNOSTIC", params.p3_steps, wandb_args=args)
                rows.append(mlp_final)
                trace.extend(mlp_trace)
                write_csv(out_dir / "p2_real_task_recipe.csv", rows)
                write_csv(out_dir / "p2_real_task_trace.csv", trace)
                print(f"P2_REAL {canonical} seed={seed} MLP test_acc={mlp_final['test_acc']:.4f}")

                for opt_name in optimizers:
                    final, tr = _train_manual_candidate(
                        REAL_METHOD,
                        opt_name,
                        bundle,
                        seed,
                        params,
                        device,
                        canonical,
                        "P2_REAL_DIAGNOSTIC",
                        params.p3_steps,
                        step_ratio=float("nan"),
                        bmem_ratio=float("nan"),
                        wandb_args=args,
                    )
                    final["optimizer"] = opt_name
                    final["test_acc_sanity"] = final.get("test_acc")
                    final["selection_uses_test"] = 0
                    for item in tr:
                        item["optimizer"] = opt_name
                        item["test_acc_sanity"] = item.get("test_acc")
                    rows.append(final)
                    trace.extend(tr)
                    write_csv(out_dir / "p2_real_task_recipe.csv", rows)
                    write_csv(out_dir / "p2_real_task_trace.csv", trace)
                    print(f"P2_REAL {canonical} seed={seed} {REAL_METHOD}/{opt_name} test_acc={final['test_acc']:.4f}")
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                rows.append({"stage": "P2_REAL_DIAGNOSTIC", "dataset": canonical, "seed": seed, "method": REAL_METHOD, "error": repr(exc)})
                write_csv(out_dir / "p2_real_task_recipe.csv", rows)

    _decorate_task_rows(rows, trace, "P2_REAL_DIAGNOSTIC")
    for row in rows:
        row["test_acc_sanity"] = row.get("test_acc", row.get("test_acc_sanity", ""))
        row["selection_uses_test"] = row.get("selection_uses_test", 0)
        _wandb_log_row(args, row, "summary/v64_p2_real_diagnostic")
    write_csv(out_dir / "p2_real_task_recipe.csv", rows)
    write_csv(out_dir / "p2_real_task_trace.csv", trace)
    _write_manifest(out_dir, args, started, time.time())
    _wandb_finish(args, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
