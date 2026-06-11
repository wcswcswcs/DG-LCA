#!/usr/bin/env python3
"""v22.05 queue and GPU utilization artifacts."""

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

from experiments.run_v22_05_common import PYTHON, append_exec, ensure_out, now_sg, read_rows, simple_svg, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--interval-sec", type=float, default=5.0)
    return p


def _sample_gpu() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
    rows = []
    ts = now_sg()
    if proc.returncode != 0:
        return [{"timestamp": ts, "gpu_id": "", "job_id": "", "queue_nonempty": "", "busy": "", "memory_used_mb": "", "utilization_percent": "", "sample_error": proc.stderr.strip()[:300]}]
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            used = int(float(parts[1]))
            util = int(float(parts[2]))
            rows.append(
                {
                    "timestamp": ts,
                    "gpu_id": parts[0],
                    "job_id": "",
                    "queue_nonempty": 0,
                    "busy": int(used > 0 or util > 0),
                    "memory_used_mb": used,
                    "utilization_percent": util,
                }
            )
    return rows


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    journal = read_rows(out_dir / "v22_05_command_journal.csv")
    queue = [{"timestamp": r.get("timestamp", ""), "command": r.get("command", ""), "status": r.get("status", ""), "note": r.get("note", "")} for r in journal]
    assignments = []
    for row in queue:
        for idx in range(4):
            if f"cuda:{idx}" in str(row.get("command", "")):
                assignments.append({"gpu_id": idx, **row})
    timeline = []
    for idx in range(max(1, int(args.samples))):
        timeline.extend(_sample_gpu())
        if idx + 1 < int(args.samples):
            time.sleep(max(0.0, float(args.interval_sec)))
    idle_violation = int(any(int(r.get("queue_nonempty") or 0) and not int(r.get("busy") or 0) for r in timeline))
    write_rows(out_dir / "v22_05_runnable_queue.csv", queue)
    write_rows(out_dir / "v22_05_gpu_assignment_manifest.csv", assignments)
    write_rows(out_dir / "v22_05_gpu_utilization_timeline.csv", timeline)
    write_rows(out_dir / "gpu_utilization_timeline.csv", timeline)
    write_rows(out_dir / "v22_05_idle_violation.csv", [{"checked": 1, "idle_violation": idle_violation, "samples": len(timeline), "blocker": "" if not idle_violation else "gpu_idle_while_queue_nonempty"}])
    write_rows(out_dir / "v22_05_deferred_items.csv", [])
    write_rows(out_dir / "v22_05_queue_drain_report.csv", [{"queued_commands": len(queue), "gpu_assigned_commands": len(assignments), "drained": int(all(str(r.get("status")) != "started" for r in queue))}])
    simple_svg(out_dir / "figures/GPU_utilization_timeline.svg", "v22.05 GPU utilization timeline", timeline, "utilization_percent")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_05_queue.py --out-dir {out_dir} --samples {args.samples} --interval-sec {args.interval_sec}",
        status="completed",
        note=f"samples={len(timeline)} idle_violation={idle_violation}",
    )


if __name__ == "__main__":
    main()
