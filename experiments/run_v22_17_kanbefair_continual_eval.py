#!/usr/bin/env python3
"""Run KANbeFair exact Class_MNIST continual-learning baseline smoke."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_17_common import (  # noqa: E402
    OUT_ROOT,
    PYTHON,
    WORKTREE_ROOT,
    append_exec,
    ensure_out,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="MLP,KAN")
    p.add_argument("--seed", type=int, default=22017)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--test-batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--output-name", default="v22_17_kanbefair_continual_learning_matrix.csv")
    p.add_argument("--fresh", action="store_true", default=True)
    return p


def _layers_for(model: str) -> list[str]:
    if model == "KAN":
        return ["4", "4"]
    return ["32", "32"]


def _exp_dir(model: str, args: argparse.Namespace) -> Path:
    layers = _layers_for(model)
    base = (
        WORKTREE_ROOT
        / "logs"
        / "Class_MNIST"
        / model
        / f"{'_'.join(layers)}__False__gelu__{args.batch_size}__{args.epochs}__{args.lr}__{args.seed}"
    )
    if model == "KAN":
        return base / "5__3__zero__-1.0_1.0"
    return base / "default"


def _cmd_for(model: str, args: argparse.Namespace) -> list[str]:
    cmd = [
        PYTHON,
        "train_continual_learning.py",
        "--model",
        model,
        "--layers_width",
        *_layers_for(model),
        "--dataset",
        "Class_MNIST",
        "--batch-size",
        str(args.batch_size),
        "--test-batch-size",
        str(args.test_batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--activation_name",
        "gelu",
    ]
    if model == "KAN":
        cmd.extend(
            [
                "--kan_bspline_grid",
                "5",
                "--kan_bspline_order",
                "3",
                "--kan_shortcut_name",
                "zero",
                "--kan_grid_range",
                "-1",
                "1",
            ]
        )
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def _parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    row: dict[str, Any] = {
        "log_path": str(path.relative_to(ROOT)) if path.exists() else "",
        "average_accuracy": "",
        "average_backward_transfer": "",
        "total_training_time_sec": "",
        "average_training_time_per_epoch_sec": "",
    }
    acc: dict[tuple[int, int], float] = {}
    for match in re.finditer(r"continual accuracy (\d+) - (\d+): ([0-9.]+)", text):
        task_i = int(match.group(1))
        after_task = int(match.group(2))
        value = float(match.group(3))
        acc[(after_task, task_i)] = value
        row[f"ACC_after_task_{after_task}_on_task_{task_i}"] = value
    avg = re.search(r"average_accuracy ([0-9.]+), average_backward_transfer: ([\-0-9.]+)", text)
    if avg:
        row["average_accuracy"] = float(avg.group(1))
        row["average_backward_transfer"] = float(avg.group(2))
    timing = re.search(
        r"total training time: ([0-9.eE,+-]+) seconds; average training time per epoch: ([0-9.eE,+-]+) seconds",
        text,
    )
    if timing:
        row["total_training_time_sec"] = float(timing.group(1).replace(",", ""))
        row["average_training_time_per_epoch_sec"] = float(timing.group(2).replace(",", ""))
    if (0, 0) in acc and (2, 0) in acc:
        row["Forgetting_T0_after_T2"] = max(0.0, acc[(0, 0)] - acc[(2, 0)])
    if (1, 1) in acc and (2, 1) in acc:
        row["Forgetting_T1_after_T2"] = max(0.0, acc[(1, 1)] - acc[(2, 1)])
    return row


def main() -> None:
    args = parser().parse_args()
    ensure_out()
    src = WORKTREE_ROOT / "src"
    rows: list[dict[str, Any]] = []
    commands: list[str] = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        exp_dir = _exp_dir(model, args)
        if args.fresh and exp_dir.exists():
            shutil.rmtree(exp_dir)
        cmd = _cmd_for(model, args)
        commands.append(" ".join(cmd))
        start = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(src), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        stdout_path = OUT_ROOT / "logs" / f"v22_17_continual_{model}_stdout.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
        log_path = exp_dir / "log"
        parsed = _parse_log(log_path)
        parsed.update(
            {
                "dataset": "Class_MNIST",
                "protocol": "KANbeFair_exact_digits_0-2_3-5_6-8_num_classes_9",
                "model_name": model,
                "origin": "KANbeFair_original",
                "dry_run": int(bool(args.dry_run)),
                "epochs": args.epochs,
                "seed": args.seed,
                "exit_code": proc.returncode,
                "status": "pass" if proc.returncode == 0 else "fail",
                "stdout_log": str(stdout_path.relative_to(ROOT)),
                "elapsed_sec": time.perf_counter() - start,
                "dgkan_official_evidence": 0,
                "basis_native_claim_allowed": 0,
            }
        )
        rows.append(parsed)
    out_path = OUT_ROOT / args.output_name
    write_rows(out_path, rows)
    append_exec(
        "\n".join(commands),
        task_id="G4-continual-dryrun",
        status="pass" if rows and all(int(r["exit_code"]) == 0 for r in rows) else "fail",
        gpu="script device cuda if available; external CUDA_VISIBLE_DEVICES controls physical GPU",
        exit_code=0 if rows and all(int(r["exit_code"]) == 0 for r in rows) else 1,
        files=str(out_path.relative_to(ROOT)),
        note="KANbeFair exact Class_MNIST original baseline dry-run only; not DG-KAN/FU official evidence",
    )


if __name__ == "__main__":
    main()
