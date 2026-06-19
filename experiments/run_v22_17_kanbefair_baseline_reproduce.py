#!/usr/bin/env python3
"""Run KANbeFair original baseline dry-run reproduction rows."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
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
    read_rows,
    run_logged,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="MNIST")
    p.add_argument("--timeout", type=int, default=420)
    p.add_argument("--clean-logs", action="store_true", default=True)
    return p


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8", errors="replace").splitlines())


def _new_result_lines(path: Path, before: int) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[before:]


def _parse_result_line(line: str) -> dict[str, Any]:
    parts = line.split(",")
    if len(parts) < 10:
        return {"raw_results_line": line, "parse_ok": 0}
    row = {
        "timestamp": parts[0],
        "dataset": parts[1],
        "model_name": parts[2],
        "raw_results_line": line,
        "parse_ok": 1,
    }
    # Last six fields are kwargs emitted by train.py in insertion order.
    if len(parts) >= 9:
        tail = parts[-6:]
        keys = [
            "train_metric_from_original",
            "test_metric_from_original",
            "param_count",
            "flops",
            "total_training_time",
            "average_training_time_per_epoch",
        ]
        for key, value in zip(keys, tail):
            row[key] = value
    return row


def _latest_log_tail(dataset: str, model: str) -> str:
    root = WORKTREE_ROOT / "logs" / dataset / model
    logs = sorted(root.rglob("log")) if root.exists() else []
    if not logs:
        return ""
    return logs[-1].read_text(encoding="utf-8", errors="replace")[-2000:].replace("\n", "\\n")


def _run_train(task_id: str, args: list[str], timeout: int) -> tuple[int, list[str]]:
    result_path = WORKTREE_ROOT / "results/results.csv"
    before = _line_count(result_path)
    proc = run_logged(
        [PYTHON, "train.py", *args],
        task_id=task_id,
        cwd=WORKTREE_ROOT / "src",
        gpu="0",
        timeout=timeout,
        env={"CUDA_VISIBLE_DEVICES": "0"},
    )
    return proc.returncode, _new_result_lines(result_path, before)


def main() -> None:
    args = parser().parse_args()
    ensure_out()
    if args.clean_logs:
        for model in ["MLP", "KAN"]:
            path = WORKTREE_ROOT / "logs" / args.dataset / model
            if path.exists():
                shutil.rmtree(path)
    rows: list[dict[str, Any]] = []
    parse_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    mlp_args = [
        "--model",
        "MLP",
        "--layers_width",
        "32",
        "32",
        "--dataset",
        args.dataset,
        "--batch-size",
        "128",
        "--test-batch-size",
        "256",
        "--epochs",
        "1",
        "--lr",
        "0.001",
        "--seed",
        "0",
        "--activation_name",
        "gelu",
        "--dry-run",
    ]
    code, lines = _run_train("G1-baseline-MLP-dryrun", mlp_args, args.timeout)
    parsed = [_parse_result_line(line) for line in lines]
    parse_rows.extend(parsed)
    rows.append(
        {
            "baseline_origin": "KANbeFair_original",
            "dataset": args.dataset,
            "model_name": "MLP",
            "command": f"{PYTHON} train.py {' '.join(mlp_args)}",
            "exit_code": code,
            "dry_run": 1,
            "fallback_used": 0,
            "results_rows_added": len(lines),
            "log_tail": _latest_log_tail(args.dataset, "MLP"),
            **(parsed[-1] if parsed else {}),
        }
    )
    if code != 0:
        failure_rows.append({"stage": "baseline_mlp_dryrun", "failure_type": "exit_nonzero", "exit_code": code, "repair_attempt": "none_available_before_MLP_pass"})

    kan_attempts = [
        (
            "plan_grid20_order5",
            [
                "--model",
                "KAN",
                "--layers_width",
                "4",
                "4",
                "--dataset",
                args.dataset,
                "--batch-size",
                "128",
                "--test-batch-size",
                "256",
                "--epochs",
                "1",
                "--lr",
                "0.001",
                "--seed",
                "0",
                "--kan_bspline_grid",
                "20",
                "--kan_bspline_order",
                "5",
                "--kan_shortcut_name",
                "silu",
                "--kan_grid_range",
                "-4",
                "4",
                "--dry-run",
            ],
        ),
        (
            "fallback_grid5_order3",
            [
                "--model",
                "KAN",
                "--layers_width",
                "4",
                "4",
                "--dataset",
                args.dataset,
                "--batch-size",
                "128",
                "--test-batch-size",
                "256",
                "--epochs",
                "1",
                "--lr",
                "0.001",
                "--seed",
                "0",
                "--kan_bspline_grid",
                "5",
                "--kan_bspline_order",
                "3",
                "--kan_shortcut_name",
                "silu",
                "--kan_grid_range",
                "-2",
                "2",
                "--dry-run",
            ],
        ),
    ]
    kan_pass = False
    for idx, (attempt_name, cmd_args) in enumerate(kan_attempts):
        if idx > 0 and kan_pass:
            continue
        code, lines = _run_train(f"G1-baseline-KAN-dryrun-{attempt_name}", cmd_args, args.timeout)
        parsed = [_parse_result_line(line) for line in lines]
        parse_rows.extend(parsed)
        kan_pass = code == 0 and bool(parsed)
        rows.append(
            {
                "baseline_origin": "KANbeFair_original",
                "dataset": args.dataset,
                "model_name": "KAN",
                "command": f"{PYTHON} train.py {' '.join(cmd_args)}",
                "exit_code": code,
                "dry_run": 1,
                "attempt": attempt_name,
                "fallback_used": int(idx > 0),
                "results_rows_added": len(lines),
                "log_tail": _latest_log_tail(args.dataset, "KAN"),
                **(parsed[-1] if parsed else {}),
            }
        )
        if code != 0:
            failure_rows.append(
                {
                    "stage": f"baseline_kan_dryrun_{attempt_name}",
                    "failure_type": "timeout" if code == 124 else "exit_nonzero",
                    "exit_code": code,
                    "repair_attempt": "lower_grid_order" if idx == 0 else "record_smoke_blocked",
                }
            )

    write_rows(OUT_ROOT / "v22_17_kanbefair_baseline_reproduction.csv", rows)
    write_rows(OUT_ROOT / "v22_17_kanbefair_original_results_parse.csv", parse_rows)
    write_rows(OUT_ROOT / "v22_17_kanbefair_failure_taxonomy.csv", failure_rows)
    append_exec(
        f"{sys.executable} experiments/run_v22_17_kanbefair_baseline_reproduce.py --dataset {args.dataset} --timeout {args.timeout}",
        task_id="G1-baseline-reproduce",
        status="pass" if any(r.get("model_name") == "MLP" and int(r.get("exit_code", 1)) == 0 for r in rows) and kan_pass else "partial",
        gpu="0",
        exit_code=0,
        files=(
            "results/v22_17/v22_17_kanbefair_baseline_reproduction.csv, "
            "results/v22_17/v22_17_kanbefair_original_results_parse.csv"
        ),
        note="KAN fallback is only used if plan-grid attempt fails or times out",
    )


if __name__ == "__main__":
    main()

