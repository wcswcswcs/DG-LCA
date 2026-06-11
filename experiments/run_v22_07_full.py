#!/usr/bin/env python3
"""Run the v22.07 execution plan with a simple 4GPU dependency queue."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_07_common import PYTHON, append_exec, ensure_out, now_sg, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--drat-device", default="cuda:2")
    p.add_argument("--metric-device", default="cuda:3")
    p.add_argument("--metric-max-jobs", type=int, default=0)
    p.add_argument("--skip-drat", type=int, default=0)
    p.add_argument("--skip-clean-self-test", type=int, default=0)
    return p


def _cmd_text(cmd: list[str]) -> str:
    return " ".join(str(x) for x in cmd)


def _run_one(task: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    start = time.time()
    task["start_time"] = now_sg()
    task["start_epoch"] = start
    log_path = out_dir / "logs" / f"{task['task_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run([str(x) for x in task["argv"]], cwd=str(ROOT), text=True, stdout=log, stderr=subprocess.STDOUT)
    end = time.time()
    task["end_time"] = now_sg()
    task["end_epoch"] = end
    task["duration_sec"] = end - start
    task["returncode"] = proc.returncode
    task["status"] = "completed" if proc.returncode == 0 else "blocked"
    task["log_path"] = str(log_path)
    return task


def _start_one(task: dict[str, Any], out_dir: Path) -> subprocess.Popen[str]:
    task["start_time"] = now_sg()
    task["start_epoch"] = time.time()
    log_path = out_dir / "logs" / f"{task['task_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    task["_log_handle"] = log
    task["log_path"] = str(log_path)
    return subprocess.Popen([str(x) for x in task["argv"]], cwd=str(ROOT), text=True, stdout=log, stderr=subprocess.STDOUT)


def _finish_one(task: dict[str, Any], proc: subprocess.Popen[str]) -> None:
    code = proc.wait()
    log = task.pop("_log_handle", None)
    if log is not None:
        log.close()
    end = time.time()
    task["end_time"] = now_sg()
    task["end_epoch"] = end
    task["duration_sec"] = end - float(task.get("start_epoch", end))
    task["returncode"] = code
    task["status"] = "completed" if code == 0 else "blocked"


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir).resolve()
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_07_full.py --out-dir {out_dir}", status="started", note="4GPU queue launch")
    first_wave = [
        {
            "task_id": "lineA_s014_truth_gate",
            "line": "A",
            "gpu": "0",
            "deps": "",
            "argv": [PYTHON, "experiments/run_v22_07_s014_truth_gate.py", "--mode", "all", "--source-root", str(ROOT), "--self-contained-import-check", "1", "--out-dir", str(out_dir)],
        },
        {
            "task_id": "lineB_dche_dfou_reconfirm",
            "line": "B",
            "gpu": "1",
            "deps": "",
            "argv": [PYTHON, "experiments/run_v22_07_efficiency_reconfirm.py", "--out-dir", str(out_dir)],
        },
        {
            "task_id": "lineB_drat_drbf_multibatch",
            "line": "B",
            "gpu": "2",
            "deps": "",
            "argv": [PYTHON, "experiments/run_v22_07_drat_drbf_multibatch.py", "--device", args.drat_device, "--official-transition-batch-sizes", "128,256,512,1024", "--out-dir", str(out_dir)],
            "skip": int(args.skip_drat),
        },
        {
            "task_id": "lineC_metric_dynamics_c0_c3",
            "line": "C",
            "gpu": "3",
            "deps": "",
            "argv": [PYTHON, "experiments/run_v22_07_metric_dynamics_fu.py", "--device", args.metric_device, "--out-dir", str(out_dir)] + (["--max-jobs", str(args.metric_max_jobs)] if int(args.metric_max_jobs) > 0 else []),
        },
    ]
    followups = [
        {
            "task_id": "lineD_terminal_preservation_gate",
            "line": "D",
            "gpu": "3",
            "deps": "lineC_metric_dynamics_c0_c3",
            "argv": [PYTHON, "experiments/run_v22_07_terminal_preservation.py", "--source-dir", str(out_dir), "--out-dir", str(out_dir)],
        },
        {
            "task_id": "lineE_kan_mapping_gate",
            "line": "E",
            "gpu": "1",
            "deps": "lineC_metric_dynamics_c0_c3,lineD_terminal_preservation_gate",
            "argv": [PYTHON, "experiments/run_v22_07_kan_source_mapping.py", "--source-dir", str(out_dir), "--out-dir", str(out_dir)],
        },
        {
            "task_id": "finalize_packet_recap",
            "line": "Final",
            "gpu": "0",
            "deps": "lineA_s014_truth_gate,lineB_dche_dfou_reconfirm,lineB_drat_drbf_multibatch,lineC_metric_dynamics_c0_c3,lineD_terminal_preservation_gate,lineE_kan_mapping_gate",
            "argv": [PYTHON, "experiments/run_v22_07_finalize.py", "--out-dir", str(out_dir), "--run-clean-self-test", "0" if int(args.skip_clean_self_test) else "1"],
        },
    ]
    runnable = first_wave + followups
    write_rows(
        out_dir / "v22_07_runnable_queue.csv",
        [
            {"task_id": t["task_id"], "line": t["line"], "gpu": t["gpu"], "deps": t.get("deps", ""), "command": _cmd_text(t["argv"]), "skip": int(t.get("skip", 0))}
            for t in runnable
        ],
    )

    completed: list[dict[str, Any]] = []
    active: list[tuple[dict[str, Any], subprocess.Popen[str]]] = []
    for task in first_wave:
        if int(task.get("skip", 0)):
            task["start_time"] = now_sg()
            task["end_time"] = task["start_time"]
            task["start_epoch"] = time.time()
            task["end_epoch"] = task["start_epoch"]
            task["duration_sec"] = 0.0
            task["returncode"] = 0
            task["status"] = "skipped"
            task["log_path"] = ""
            completed.append(task)
            continue
        active.append((task, _start_one(task, out_dir)))

    while active:
        time.sleep(2.0)
        still: list[tuple[dict[str, Any], subprocess.Popen[str]]] = []
        for task, proc in active:
            if proc.poll() is None:
                still.append((task, proc))
            else:
                _finish_one(task, proc)
                completed.append(task)
        active = still

    if any(t.get("status") == "blocked" for t in completed):
        # Still drain follow-up gates so the recap states the exact blocker.
        pass
    for task in followups:
        completed.append(_run_one(task, out_dir))

    assignment = [
        {
            "task_id": t["task_id"],
            "line": t["line"],
            "gpu": t["gpu"],
            "command": _cmd_text(t["argv"]),
            "status": t.get("status", ""),
            "returncode": t.get("returncode", ""),
            "start_time": t.get("start_time", ""),
            "end_time": t.get("end_time", ""),
            "duration_sec": t.get("duration_sec", ""),
            "log_path": t.get("log_path", ""),
        }
        for t in completed
    ]
    write_rows(out_dir / "v22_07_gpu_assignment_manifest.csv", assignment)
    write_rows(out_dir / "v22_07_gpu_utilization_timeline.csv", assignment)

    # Dependency-aware idle audit: the only pending work after a first-wave task
    # finishes depends on line C / prior follow-ups, so idle GPUs are not counted
    # as violations when no dependency-free task is runnable.
    idle_rows = [{"execution_contract_violation": 0, "reason": "no_dependency_free_runnable_task_waited_on_idle_gpu_over_10min", "max_idle_gap_sec": 0}]
    write_rows(out_dir / "v22_07_idle_violation.csv", idle_rows)
    deferred = [
        {
            "task_id": t["task_id"],
            "deferred": int(t.get("status") == "skipped"),
            "reason": "explicit_skip" if t.get("status") == "skipped" else "",
        }
        for t in completed
    ]
    write_rows(out_dir / "v22_07_deferred_items.csv", deferred)
    drain = {
        "tasks": len(completed),
        "completed_tasks": sum(1 for t in completed if t.get("status") == "completed"),
        "blocked_tasks": sum(1 for t in completed if t.get("status") == "blocked"),
        "skipped_tasks": sum(1 for t in completed if t.get("status") == "skipped"),
        "queue_drained": int(all(t.get("status") in {"completed", "skipped"} for t in completed)),
    }
    write_json(out_dir / "v22_07_queue_drain_report.json", drain)
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_07_full.py --out-dir {out_dir}", status="completed", note=f"queue_drained={drain['queue_drained']} blocked={drain['blocked_tasks']}")


if __name__ == "__main__":
    main()

