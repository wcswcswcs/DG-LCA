#!/usr/bin/env python3
"""Shared helpers for DG-KAN v22.18 benefit-conditioned runs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("KAN_PYTHON", "/home/chengshun.wang/miniconda3/envs/kan/bin/python")
OUT_ROOT = ROOT / "results/v22_18"
FIG_ROOT = OUT_ROOT / "figures"
LOG_ROOT = OUT_ROOT / "logs"
PLAN_DOC = ROOT / "docs/DG-KAN_v22.18_BenefitConditionedControllability_完整计划.md"
EXEC_DOC = ROOT / "docs/DG-KAN_v22.18_BenefitConditionedControllability_执行日志.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v22.18_BenefitConditionedControllability_实验结果复盘.md"
WORKTREE_ROOT = ROOT / "external/KANbeFair_worktree"
V22_17_ROOT = ROOT / "results/v22_17"


REQUIRED_SOURCE_FILES = [
    "dgkan/fu/source_action_bank.py",
    "dgkan/fu/benefit_policy.py",
    "dgkan/fu/treatment_effect_labels.py",
    "dgkan/fu/kan_native_jvp.py",
    "dgkan/fu/continual_source_state.py",
    "experiments/run_v22_18_common.py",
    "experiments/run_v22_18_status_manifest.py",
    "experiments/run_v22_17_basis_jvp_vjp_gradcheck.py",
    "experiments/run_v22_17_kan_basis_native_audit.py",
    "experiments/run_v22_17_kanbefair_dgkan_eval.py",
    "experiments/run_v22_18_source_action_bank.py",
    "experiments/run_v22_18_benefit_policy.py",
    "experiments/run_v22_18_branch_label_readback.py",
    "experiments/run_v22_18_control_projected_labels.py",
    "experiments/run_v22_18_strict_label_bank.py",
    "experiments/run_v22_18_runtime_p3_no_go_audit.py",
    "experiments/run_v22_18_mlp_fu_benefit_mainline.py",
    "experiments/run_v22_18_kan_native_ladder.py",
    "experiments/run_v22_18_kanbefair_expanded.py",
    "experiments/run_v22_18_label_strictness_audit.py",
    "experiments/run_v22_18_continual.py",
    "experiments/run_v22_18_identity_audit.py",
    "experiments/run_v22_18_finalize.py",
]


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out() -> Path:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    init_docs()
    return OUT_ROOT


def init_docs() -> None:
    EXEC_DOC.parent.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.18 Benefit-Conditioned Controllability 执行日志\n\n"
            f"生成时间：{now_sg()}\n\n"
            "记录原则：只记录真实执行的命令、文件、状态、blocker、修复尝试；未执行项不写成完成。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.18 Benefit-Conditioned Controllability 实验结果复盘\n\n"
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
    journal = OUT_ROOT / "v22_18_command_journal.csv"
    exists = journal.exists() and journal.stat().st_size > 0
    with journal.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    with EXEC_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n## {row['timestamp']} {task_id}\n\n")
        f.write(f"```bash\n{command}\n```\n\n")
        f.write(f"- gpu: {gpu or 'n/a'}\n")
        f.write(f"- status: {status}\n")
        f.write(f"- exit_code: {exit_code if exit_code != '' else 'n/a'}\n")
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


def source_packet_paths() -> list[Path]:
    return [ROOT / rel for rel in REQUIRED_SOURCE_FILES]


def artifact_index() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not OUT_ROOT.exists():
        return rows
    for path in sorted(OUT_ROOT.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "artifact": str(path.relative_to(ROOT)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "nonempty": int(path.stat().st_size > 0),
                    "latest_status_timestamp": now_sg(),
                }
            )
    return rows


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in {float("inf"), float("-inf")}:
        return default
    return out


def int_flag(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def md_table(rows: list[dict[str, Any]], fields: list[str] | None = None, limit: int | None = None) -> str:
    if not rows:
        return "\n无记录。\n"
    show = rows[:limit] if limit else rows
    if fields is None:
        fields = list(show[0])
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in show:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    if limit and len(rows) > limit:
        lines.append(f"\n（仅显示前 {limit} 行，共 {len(rows)} 行。）")
    return "\n".join(lines) + "\n"
