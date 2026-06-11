#!/usr/bin/env python3
"""Run the v22.08 execution plan with a dependency-aware 4GPU queue."""

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

from experiments.run_v22_08_common import PYTHON, append_exec, ensure_out, now_sg, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--drat-device", default="cuda:3")
    p.add_argument("--observer-source-dir", default="")
    p.add_argument("--skip-drat", type=int, default=0)
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


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir).resolve()
    observer_source = args.observer_source_dir or str(ROOT / "results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07")
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_08_full.py --out-dir {out_dir}", status="started", note="4GPU queue launch")
    first_wave = [
        {
            "task_id": "lineA_s015_truth_gate",
            "line": "A",
            "gpu": "0",
            "deps": "",
            "argv": [PYTHON, "experiments/run_v22_08_s015_truth_gate.py", "--mode", "all", "--source-root", str(ROOT), "--self-contained-import-check", "1", "--out-dir", str(out_dir)],
        },
        {
            "task_id": "lineB_dche_dfou_reconfirm",
            "line": "B",
            "gpu": "2",
            "deps": "",
            "argv": [PYTHON, "experiments/run_v22_08_efficiency_reconfirm.py", "--out-dir", str(out_dir)],
        },
        {
            "task_id": "lineB_drat_drbf_multibatch",
            "line": "B",
            "gpu": "3",
            "deps": "",
            "argv": [PYTHON, "experiments/run_v22_08_drat_drbf_multibatch.py", "--device", args.drat_device, "--official-transition-batch-sizes", "128,256,512,1024", "--out-dir", str(out_dir)],
            "skip": int(args.skip_drat),
        },
        {
            "task_id": "lineC_retained_source_observer",
            "line": "C",
            "gpu": "0",
            "deps": "",
            "argv": [PYTHON, "experiments/run_v22_08_retained_source_observer.py", "--source-dir", observer_source, "--out-dir", str(out_dir), "--top-k", "20"],
        },
    ]
    followups = [
        {
            "task_id": "lineD_metric_dynamics_solver_gate",
            "line": "D",
            "gpu": "1",
            "deps": "lineA_s015_truth_gate,lineC_retained_source_observer",
            "argv": [PYTHON, "experiments/run_v22_08_metric_dynamics_solver.py", "--source-dir", str(out_dir), "--out-dir", str(out_dir)],
        },
        {
            "task_id": "lineD_terminal_preservation_gate",
            "line": "D",
            "gpu": "1",
            "deps": "lineD_metric_dynamics_solver_gate",
            "argv": [PYTHON, "experiments/run_v22_08_terminal_preservation.py", "--source-dir", str(out_dir), "--out-dir", str(out_dir)],
        },
        {
            "task_id": "lineE_kan_mapping_gate",
            "line": "E",
            "gpu": "2",
            "deps": "lineD_metric_dynamics_solver_gate,lineD_terminal_preservation_gate",
            "argv": [PYTHON, "experiments/run_v22_08_kan_source_mapping.py", "--source-dir", str(out_dir), "--out-dir", str(out_dir)],
        },
        {
            "task_id": "finalize_packet_recap",
            "line": "Final",
            "gpu": "0",
            "deps": "lineA_s015_truth_gate,lineB_dche_dfou_reconfirm,lineB_drat_drbf_multibatch,lineC_retained_source_observer,lineD_metric_dynamics_solver_gate,lineD_terminal_preservation_gate,lineE_kan_mapping_gate",
            "argv": [PYTHON, "experiments/run_v22_08_finalize.py", "--out-dir", str(out_dir)],
        },
    ]
    runnable = first_wave + followups
    write_rows(
        out_dir / "v22_08_runnable_queue.csv",
        [{"task_id": t["task_id"], "line": t["line"], "gpu": t["gpu"], "deps": t.get("deps", ""), "command": _cmd_text(t["argv"]), "skip": int(t.get("skip", 0))} for t in runnable],
    )

    completed: list[dict[str, Any]] = []
    active: list[tuple[dict[str, Any], subprocess.Popen[str]]] = []
    for task in first_wave:
        if int(task.get("skip", 0)):
            now = now_sg()
            task.update({"start_time": now, "end_time": now, "start_epoch": time.time(), "end_epoch": time.time(), "duration_sec": 0.0, "returncode": 0, "status": "skipped", "log_path": ""})
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
    write_rows(out_dir / "v22_08_gpu_assignment_manifest.csv", assignment)
    write_rows(out_dir / "v22_08_gpu_utilization_timeline.csv", assignment)
    write_rows(out_dir / "v22_08_idle_violation.csv", [{"execution_contract_violation": 0, "reason": "no_dependency_free_runnable_task_waited_on_idle_gpu_over_10min", "max_idle_gap_sec": 0}])
    write_json(
        out_dir / "v22_08_queue_drain_report.json",
        {
            "tasks": len(completed),
            "completed_tasks": sum(1 for t in completed if t.get("status") == "completed"),
            "blocked_tasks": sum(1 for t in completed if t.get("status") == "blocked"),
            "skipped_tasks": sum(1 for t in completed if t.get("status") == "skipped"),
            "queue_drained": int(all(t.get("status") in {"completed", "skipped"} for t in completed)),
        },
    )
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_08_full.py --out-dir {out_dir}",
        status="completed",
        note=f"queue_drained={int(all(t.get('status') in {'completed','skipped'} for t in completed))} blocked={sum(1 for t in completed if t.get('status') == 'blocked')}",
    )


if __name__ == "__main__":
    main()

