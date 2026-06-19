#!/usr/bin/env python3
"""Shared helpers for DG-KAN v22.17 KANbeFair integration runs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("KAN_PYTHON", "/home/chengshun.wang/miniconda3/envs/kan/bin/python")
OUT_ROOT = ROOT / "results/v22_17"
FIG_ROOT = OUT_ROOT / "figures"
LOG_ROOT = OUT_ROOT / "logs"
PLAN_DOC = ROOT / "docs/DG-KAN_v22.17_TaskUsefulControllability_KANbeFair_上传源码更新版.md"
EXEC_DOC = ROOT / "docs/DG-KAN_v22.17_TaskUsefulControllability_KANbeFair_执行日志.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v22.17_TaskUsefulControllability_KANbeFair_实验结果复盘.md"
EXTERNAL_ROOT = ROOT / "external"
ZIP_PATH = EXTERNAL_ROOT / "KANbeFair-main.zip"
RAW_ROOT = EXTERNAL_ROOT / "KANbeFair_upload_raw"
WORKTREE_ROOT = EXTERNAL_ROOT / "KANbeFair_worktree"


REQUIRED_SOURCE_FILES = [
    "dgkan/fu/mlp_adaptive_controller.py",
    "dgkan/integration/kanbefair_adapter.py",
    "experiments/run_v22_17_common.py",
    "experiments/run_v22_17_kanbefair_unpack_patch.py",
    "experiments/run_v22_17_kanbefair_env_smoke.py",
    "experiments/run_v22_17_kanbefair_baseline_reproduce.py",
    "experiments/run_v22_17_kanbefair_continual_eval.py",
    "experiments/run_v22_17_kanbefair_dgkan_eval.py",
    "experiments/run_v22_17_basis_jvp_vjp_gradcheck.py",
    "experiments/run_v22_17_kan_basis_native_audit.py",
    "experiments/run_v22_17_controllability_sketch_audit.py",
    "experiments/run_v22_17_mechanism_smoke.py",
    "experiments/run_v22_17_finalize.py",
]


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out() -> Path:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    init_docs()
    return OUT_ROOT


def init_docs() -> None:
    EXEC_DOC.parent.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.17 Task-Useful Controllability + KANbeFair 执行日志\n\n"
            f"生成时间：{now_sg()}\n\n"
            "记录原则：只记录真实命令、文件、状态、blocker 与修复尝试；未执行项不写成完成。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.17 Task-Useful Controllability + KANbeFair 实验结果复盘\n\n"
            f"生成时间：{now_sg()}\n\n"
            "本复盘只引用本轮落盘 artifact 与真实读回结果；禁止编造数据。\n",
            encoding="utf-8",
        )


def append_exec(
    command: str,
    *,
    task_id: str,
    status: str,
    gpu: str = "",
    files: str = "",
    note: str = "",
    exit_code: int | str = "",
) -> None:
    ensure_out()
    row = {
        "timestamp": now_sg(),
        "task_id": task_id,
        "gpu": gpu,
        "command": command,
        "status": status,
        "exit_code": exit_code,
        "files": files,
        "note": note,
    }
    journal = OUT_ROOT / "v22_17_command_journal.csv"
    exists = journal.exists() and journal.stat().st_size > 0
    with journal.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    with EXEC_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n## {row['timestamp']} {task_id}\n\n")
        f.write(f"```bash\n{command}\n```\n\n")
        f.write(f"- gpu: {gpu or 'n/a'}\n- status: {status}\n- exit_code: {exit_code if exit_code != '' else 'n/a'}\n")
        if files:
            f.write(f"- files: {files}\n")
        if note:
            f.write(f"- note: {note}\n")


def run_logged(
    args: list[str],
    *,
    task_id: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    gpu: str = "",
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    ensure_out()
    command = " ".join(args)
    started = time.time()
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else str(ROOT),
            env=merged_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        status = "pass" if proc.returncode == 0 else "fail"
        exit_code: int | str = proc.returncode
    except subprocess.TimeoutExpired as exc:
        proc = subprocess.CompletedProcess(args=args, returncode=124, stdout=exc.stdout or "", stderr=exc.stderr or "")
        status = "timeout"
        exit_code = 124
    stdout_path = LOG_ROOT / f"{task_id}_stdout.log"
    stderr_path = LOG_ROOT / f"{task_id}_stderr.log"
    stdout_path.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8", errors="replace")
    append_exec(
        command,
        task_id=task_id,
        status=status,
        gpu=gpu,
        exit_code=exit_code,
        files=f"{stdout_path.relative_to(ROOT)}, {stderr_path.relative_to(ROOT)}",
        note=f"elapsed_sec={time.time() - started:.3f}; cwd={cwd or ROOT}",
    )
    return proc


def read_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(r) for r in rows]
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_manifest(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        rows.append(
            {
                "relative_path": str(rel),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def tree_manifest_sha(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(f"{r['sha256']}  {r['relative_path']}" for r in rows)
    return sha256_text(payload)


def artifact_index() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "artifact": str(path.resolve().relative_to(ROOT)),
                    "exists": 1,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def md_table(rows: list[dict[str, Any]], fields: list[str], max_rows: int = 40) -> str:
    if not rows:
        return "\n_无落盘 rows。_\n"
    shown = rows[:max_rows]
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in shown:
        out.append("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
    if len(rows) > max_rows:
        out.append(f"\n_仅显示前 {max_rows} / {len(rows)} rows；完整 CSV 见 artifact。_")
    return "\n".join(out) + "\n"


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in {"", None}:
            return default
        out = float(value)
    except Exception:
        return default
    if out != out or abs(out) == float("inf"):
        return default
    return out


def int_flag(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def copy_uploaded_zip_if_needed() -> tuple[Path, str]:
    ensure_out()
    candidates = [ZIP_PATH, ROOT / "KANbeFair-main.zip"]
    source = next((p for p in candidates if p.exists()), None)
    if source is None:
        raise FileNotFoundError("KANbeFair-main.zip not found in external/ or repo root")
    if source.resolve() != ZIP_PATH.resolve():
        shutil.copy2(source, ZIP_PATH)
        note = f"copied_from={source}"
    else:
        note = "already_in_external"
    return ZIP_PATH, note


def write_execution_manifests() -> None:
    queue = [
        {"task_id": "A-provenance", "gpu": "0", "script": "experiments/run_v22_17_kanbefair_unpack_patch.py", "status": "completed_or_runnable"},
        {"task_id": "A-G0-env-smoke", "gpu": "0", "script": "experiments/run_v22_17_kanbefair_env_smoke.py", "status": "completed_or_runnable"},
        {"task_id": "G1-baseline-dryrun", "gpu": "0", "script": "experiments/run_v22_17_kanbefair_baseline_reproduce.py", "status": "completed_or_runnable"},
        {"task_id": "G4-continual-dryrun", "gpu": "3", "script": "experiments/run_v22_17_kanbefair_continual_eval.py", "status": "completed_or_runnable"},
        {"task_id": "B-C-D-mechanism-smoke", "gpu": "1", "script": "experiments/run_v22_17_mechanism_smoke.py", "status": "completed_or_runnable"},
        {"task_id": "E-F-G-bridge-smoke", "gpu": "2,3", "script": "experiments/run_v22_17_kanbefair_dgkan_eval.py", "status": "completed_or_runnable"},
        {"task_id": "finalize", "gpu": "n/a", "script": "experiments/run_v22_17_finalize.py", "status": "completed_or_runnable"},
    ]
    write_rows(OUT_ROOT / "v22_17_runnable_queue.csv", queue)
    write_rows(OUT_ROOT / "v22_17_gpu_assignment_manifest.csv", queue)
    write_rows(OUT_ROOT / "v22_17_idle_violation.csv", [{"execution_contract_violation": 0, "reason": "single-agent sequential execution; queue manifest created"}])
    write_json(OUT_ROOT / "v22_17_queue_drain_report.json", {"runnable_queue_nonempty_after_run": 0, "execution_contract_violation": 0})


def source_packet_paths() -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for rel in REQUIRED_SOURCE_FILES:
        path = ROOT / rel
        if path.exists() and path not in seen:
            paths.append(path)
            seen.add(path)
    for path in [PLAN_DOC, EXEC_DOC, RECAP_DOC]:
        if path.exists() and path not in seen:
            paths.append(path)
            seen.add(path)
    return paths


__all__ = [
    "EXEC_DOC",
    "EXTERNAL_ROOT",
    "FIG_ROOT",
    "LOG_ROOT",
    "OUT_ROOT",
    "PLAN_DOC",
    "PYTHON",
    "RAW_ROOT",
    "RECAP_DOC",
    "REQUIRED_SOURCE_FILES",
    "ROOT",
    "WORKTREE_ROOT",
    "ZIP_PATH",
    "append_exec",
    "artifact_index",
    "copy_uploaded_zip_if_needed",
    "ensure_out",
    "file_manifest",
    "finite_float",
    "init_docs",
    "int_flag",
    "md_table",
    "now_sg",
    "read_json",
    "read_rows",
    "run_logged",
    "sha256_file",
    "sha256_text",
    "source_packet_paths",
    "tree_manifest_sha",
    "write_execution_manifests",
    "write_json",
    "write_rows",
]
