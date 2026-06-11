#!/usr/bin/env python3
"""Shared helpers for v22 source-retention/kernel-officialization artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Iterable
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("KAN_PYTHON", "/home/chengshun.wang/miniconda3/envs/kan/bin/python")
V22_ROOT = ROOT / "results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu"
V22_OFFICIAL = V22_ROOT / "official_v22"
V22_EXEC_DOC = ROOT / "docs/DG-KAN_v22.0_EarlySourceRetentionFU_DCHE_DFOU_KernelOfficialization_4GPU_执行日志.md"
V22_RECAP_DOC = ROOT / "docs/DG-KAN_v22.0_EarlySourceRetentionFU_DCHE_DFOU_KernelOfficialization_4GPU_实验结果复盘.md"
V22_PLAN_DOC = ROOT / "docs/DG-KAN_v22.0_EarlySourceRetentionFU_DCHE_DFOU_KernelOfficialization_4GPU_完整计划.md"
HORIZON_STEPS = (100, 400, 800, 1600, 2400, 3200, 4800, 6400)


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out(out_dir: str | Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else V22_OFFICIAL
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return out


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
        for key in row.keys():
            if key not in fields:
                fields.append(str(key))
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def append_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(text)


def append_exec(out_dir: Path, command: str, *, status: str = "", note: str = "") -> None:
    row = {"timestamp": now_sg(), "command": command, "status": status, "note": note}
    journal_path = out_dir / "v22_command_journal.csv"
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    exists = journal_path.exists() and journal_path.stat().st_size > 0
    with journal_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "command", "status", "note"])
        if not exists:
            writer.writeheader()
        writer.writerow({k: str(v) for k, v in row.items()})
    append_text(V22_EXEC_DOC, f"\n## {row['timestamp']}\n\n```bash\n{command}\n```\n\n- status: {status or 'recorded'}\n")
    if note:
        append_text(V22_EXEC_DOC, f"- note: {note}\n")


def run_cmd(command: list[str], *, cwd: Path = ROOT, timeout: int = 300) -> tuple[int, str]:
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, timeout=timeout)
    return proc.returncode, "\n".join(
        ["$ " + " ".join(command), f"exit={proc.returncode}", "--- stdout ---", proc.stdout, "--- stderr ---", proc.stderr]
    )


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in {"", None}:
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def int_flag(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def placeholder_svg(path: Path, title: str, rows: list[dict[str, Any]], metric: str = "") -> None:
    vals = [finite_float(r.get(metric)) for r in rows] if metric else []
    vals = [v for v in vals if math.isfinite(v)]
    summary = f"rows={len(rows)}"
    if vals:
        summary += f" min={min(vals):.4g} mean={sum(vals)/len(vals):.4g} max={max(vals):.4g}"
    text = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='900' height='220'>"
        "<rect width='100%' height='100%' fill='#f7f7f5'/>"
        f"<text x='24' y='60' font-family='monospace' font-size='22'>{title}</text>"
        f"<text x='24' y='110' font-family='monospace' font-size='16'>{summary}</text>"
        "</svg>\n"
    )
    write_text(path, text)


def init_docs() -> None:
    if not V22_EXEC_DOC.exists():
        write_text(
            V22_EXEC_DOC,
            "# DG-KAN v22.0 EarlySourceRetentionFU DCHE/DFOU KernelOfficialization 4GPU 执行日志\n\n"
            f"生成时间：{now_sg()}\n\n"
            "记录原则：只记录真实执行过的命令、文件与状态；不把未执行内容写成结果。\n",
        )
    if not V22_RECAP_DOC.exists():
        write_text(
            V22_RECAP_DOC,
            "# DG-KAN v22.0 EarlySourceRetentionFU DCHE/DFOU KernelOfficialization 4GPU 实验结果复盘\n\n"
            f"生成时间：{now_sg()}\n\n"
            "本文件由 v22 runner/finalizer 从落盘 artifacts 汇总；禁止编造数据。\n",
        )


def build_packet(out_dir: Path) -> None:
    packet = out_dir / "v22_code_review_packet"
    if packet.exists():
        shutil.rmtree(packet)
    packet.mkdir(parents=True)
    dirs = [
        "00_README",
        "01_ENVIRONMENT",
        "02_SOURCE_TREE",
        "03_CODE_METRICS",
        "04_EFFICIENCY",
        "05_SOURCE_CHAIN",
        "06_TARGET_DISCOVERY",
        "07_ROUTE_AND_MANIFEST",
        "08_DOCS",
        "09_REPRO_COMMANDS",
    ]
    for d in dirs:
        (packet / d).mkdir(parents=True, exist_ok=True)
    write_text(packet / "00_README" / "README.md", "v22 packet: source, artifacts, route, logs, and reproduction commands.\n")
    for rel in [
        "dgkan/fu/source_chain.py",
        "dgkan/profiling/efficiency_v22.py",
        "experiments/run_v22_common.py",
        "experiments/run_v22_s07_truth_gate.py",
        "experiments/run_v22_efficiency_officialization.py",
        "experiments/run_v22_source_chain_dynamics.py",
        "experiments/run_v22_merge_finalize.py",
        "experiments/run_v22_mlp_source_lab.py",
        "experiments/run_v22_kan_source_writer.py",
        "experiments/run_v22_function_space_target_reset.py",
        "experiments/run_v22_source_observability_audit.py",
        "experiments/run_v22_f40_phase_reset_summary.py",
        "experiments/run_v22_f46_train_loss_selector_holdout.py",
        "dgkan/fu/mechanisms.py",
        "experiments/run_v17_common.py",
        "experiments/run_v21_01_source_retention.py",
    ]:
        copy_if_exists(ROOT / rel, packet / "02_SOURCE_TREE" / rel)
    for pattern in [
        "v22_*",
        "v21_efficiency_*",
        "v21_kernel_gradcheck*",
        "v21_01_source_retention*",
        "v21_01_functional*",
    ]:
        for path in sorted(out_dir.glob(pattern)):
            if path.is_file() and path.suffix.lower() not in {".zip"}:
                copy_if_exists(path, packet / "07_ROUTE_AND_MANIFEST" / path.name)
    copy_if_exists(V22_PLAN_DOC, packet / "08_DOCS" / V22_PLAN_DOC.name)
    copy_if_exists(V22_EXEC_DOC, packet / "08_DOCS" / V22_EXEC_DOC.name)
    copy_if_exists(V22_RECAP_DOC, packet / "08_DOCS" / V22_RECAP_DOC.name)
    for fig in sorted((out_dir / "figures").glob("*")):
        copy_if_exists(fig, packet / "04_EFFICIENCY" / fig.name)
    manifest = []
    for path in sorted(packet.rglob("*")):
        if path.is_file():
            manifest.append({"path": str(path.relative_to(packet)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_rows(packet / "packet_manifest.csv", manifest)
    zip_path = out_dir / "v22_code_review_packet.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(packet.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(packet))
    bundle = out_dir / "v22_results_bundle.zip"
    if bundle.exists():
        bundle.unlink()
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() != ".zip" and "v22_code_review_packet" not in str(path):
                z.write(path, path.relative_to(out_dir))


__all__ = [
    "ROOT",
    "PYTHON",
    "V22_ROOT",
    "V22_OFFICIAL",
    "V22_EXEC_DOC",
    "V22_RECAP_DOC",
    "V22_PLAN_DOC",
    "HORIZON_STEPS",
    "append_exec",
    "append_text",
    "build_packet",
    "copy_if_exists",
    "ensure_out",
    "finite_float",
    "init_docs",
    "int_flag",
    "now_sg",
    "placeholder_svg",
    "read_json",
    "read_rows",
    "run_cmd",
    "sha256_file",
    "write_json",
    "write_rows",
    "write_text",
]
