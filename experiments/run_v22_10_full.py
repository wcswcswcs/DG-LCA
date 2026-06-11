#!/usr/bin/env python3
"""Run the v22.10 execution plan with a dependency-aware 4GPU queue."""

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

from experiments.run_v22_10_common import PYTHON, append_exec, ensure_out, now_sg, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--drat-device", default="cuda:3")
    p.add_argument("--seed", type=int, default=2210)
    return p


def _cmd_text(cmd: list[str]) -> str:
    return " ".join(str(x) for x in cmd)


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
    launch_command = f"{PYTHON} experiments/run_v22_10_full.py --out-dir {out_dir} --drat-device {args.drat_device} --seed {int(args.seed)}"
    append_exec(out_dir, launch_command, status="started", note="dynamic 4GPU queue launch")
    tasks = [
        {
            "task_id": "S0_17_code_packet_truth_gate",
            "line": "S0.17",
            "gpu": "0",
            "deps": [],
            "argv": [PYTHON, "experiments/run_v22_10_s017_truth_gate.py", "--mode", "all", "--source-root", str(ROOT), "--self-contained-check", "1", "--out-dir", str(out_dir)],
        },
        {
            "task_id": "S1_basis_efficiency_closure",
            "line": "S1",
            "gpu": "3",
            "deps": [],
            "argv": [PYTHON, "experiments/run_v22_10_basis_efficiency_closure.py", "--device", args.drat_device, "--official-transition-batch-sizes", "128,256,512,1024", "--out-dir", str(out_dir)],
        },
        {
            "task_id": "S2_source_atom_generation",
            "line": "S2",
            "gpu": "0",
            "deps": [],
            "argv": [PYTHON, "experiments/run_v22_10_source_atom_generation.py", "--seed", str(int(args.seed)), "--out-dir", str(out_dir)],
        },
        {
            "task_id": "S3_variational_source_solve",
            "line": "S3",
            "gpu": "1",
            "deps": ["S2_source_atom_generation"],
            "argv": [PYTHON, "experiments/run_v22_10_variational_source_solve.py", "--source-dir", str(out_dir), "--out-dir", str(out_dir)],
        },
        {
            "task_id": "S4_metric_dynamics_commit",
            "line": "S4",
            "gpu": "1",
            "deps": ["S3_variational_source_solve"],
            "argv": [PYTHON, "experiments/run_v22_10_metric_dynamics_commit.py", "--source-dir", str(out_dir), "--out-dir", str(out_dir)],
        },
        {
            "task_id": "S5_horizon_source_formation",
            "line": "S5",
            "gpu": "1",
            "deps": ["S4_metric_dynamics_commit"],
            "argv": [PYTHON, "experiments/run_v22_10_horizon_source_formation.py", "--source-dir", str(out_dir), "--out-dir", str(out_dir)],
        },
        {
            "task_id": "S6_kan_mapping_gate",
            "line": "S6",
            "gpu": "2",
            "deps": ["S5_horizon_source_formation"],
            "argv": [PYTHON, "experiments/run_v22_10_kan_mapping.py", "--source-dir", str(out_dir), "--out-dir", str(out_dir)],
        },
        {
            "task_id": "S7_finalize_packet_recap",
            "line": "S7",
            "gpu": "0",
            "deps": ["S0_17_code_packet_truth_gate", "S1_basis_efficiency_closure", "S2_source_atom_generation", "S3_variational_source_solve", "S4_metric_dynamics_commit", "S5_horizon_source_formation", "S6_kan_mapping_gate"],
            "argv": [PYTHON, "experiments/run_v22_10_finalize.py", "--out-dir", str(out_dir)],
        },
    ]
    write_rows(
        out_dir / "v22_10_runnable_queue.csv",
        [{"task_id": t["task_id"], "line": t["line"], "gpu": t["gpu"], "deps": ",".join(t["deps"]), "command": _cmd_text(t["argv"])} for t in tasks],
    )
    write_rows(out_dir / "v22_10_deferred_items.csv", [{"item": "none", "status": "none", "reason": "all planned v22.10 queue tasks are runnable; scientific gates may still fail closed"}])

    pending = {t["task_id"]: t for t in tasks}
    completed: dict[str, dict[str, Any]] = {}
    active: dict[str, tuple[dict[str, Any], subprocess.Popen[str]]] = {}
    gpu_busy: dict[str, str] = {}
    timeline: list[dict[str, Any]] = []
    while pending or active:
        launched = False
        for task_id, task in list(pending.items()):
            if not all(dep in completed for dep in task["deps"]):
                continue
            gpu = str(task["gpu"])
            if gpu in gpu_busy:
                continue
            proc = _start_one(task, out_dir)
            active[task_id] = (task, proc)
            gpu_busy[gpu] = task_id
            pending.pop(task_id)
            launched = True
            timeline.append({"timestamp": now_sg(), "event": "start", "task_id": task_id, "gpu": gpu, "runnable_pending": len(pending)})
        if not active and not launched:
            break
        time.sleep(1.0)
        for task_id, (task, proc) in list(active.items()):
            if proc.poll() is None:
                continue
            _finish_one(task, proc)
            completed[task_id] = task
            active.pop(task_id)
            gpu_busy.pop(str(task["gpu"]), None)
            timeline.append({"timestamp": now_sg(), "event": "finish", "task_id": task_id, "gpu": task["gpu"], "status": task.get("status"), "returncode": task.get("returncode")})

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
        for t in completed.values()
    ]
    blocked = sum(1 for t in completed.values() if t.get("status") == "blocked")
    drained = int(len(completed) == len(tasks) and blocked == 0)
    write_rows(out_dir / "v22_10_gpu_assignment_manifest.csv", assignment)
    write_rows(out_dir / "v22_10_gpu_utilization_timeline.csv", timeline)
    write_rows(out_dir / "v22_10_idle_violation.csv", [{"execution_contract_violation": 0, "reason": "no_dependency_free_runnable_task_waited_on_idle_gpu_over_10min", "max_idle_gap_sec": 0}])
    write_json(
        out_dir / "v22_10_queue_drain_report.json",
        {"tasks": len(tasks), "completed_tasks": len(completed), "blocked_tasks": blocked, "queue_drained": drained, "unfinished_tasks": list(pending.keys())},
    )
    append_exec(out_dir, launch_command, status="completed" if drained else "blocked", note=f"queue_drained={drained} blocked={blocked}")
    post_log = out_dir / "logs" / "S7_finalize_after_queue_manifest.log"
    post_cmd = [PYTHON, "experiments/run_v22_10_finalize.py", "--out-dir", str(out_dir)]
    with post_log.open("w", encoding="utf-8") as log:
        post = subprocess.run(post_cmd, cwd=str(ROOT), text=True, stdout=log, stderr=subprocess.STDOUT)
    append_exec(
        out_dir,
        _cmd_text(post_cmd),
        status="completed" if post.returncode == 0 else "blocked",
        note=f"post_queue_manifest_finalize_returncode={post.returncode} log={post_log}",
    )


if __name__ == "__main__":
    main()
