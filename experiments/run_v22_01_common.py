#!/usr/bin/env python3
"""Shared helpers for v22.01 artifacts and logs."""

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
V2200_OFFICIAL = ROOT / "results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22"
V2201_ROOT = ROOT / "results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu"
V2201_OFFICIAL = V2201_ROOT / "official_v22_01"
V2201_PLAN_DOC = ROOT / "docs/DG-KAN_v22.01_EarlySourceRetention_TerminalCollapse_KernelOfficialization_DRAT_DRBF_4GPU_完整计划.md"
V2201_EXEC_DOC = ROOT / "docs/DG-KAN_v22.01_EarlySourceRetention_TerminalCollapse_KernelOfficialization_DRAT_DRBF_4GPU_执行日志.md"
V2201_RECAP_DOC = ROOT / "docs/DG-KAN_v22.01_EarlySourceRetention_TerminalCollapse_KernelOfficialization_DRAT_DRBF_4GPU_实验结果复盘.md"
HORIZONS = (100, 400, 800, 1200, 1600, 2400, 3200, 4000, 4800, 6400)


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out(out_dir: str | Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else V2201_OFFICIAL
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return out


def init_docs() -> None:
    if not V2201_EXEC_DOC.exists():
        V2201_EXEC_DOC.write_text(
            "# DG-KAN v22.01 EarlySourceRetention TerminalCollapse KernelOfficialization DRAT/DRBF 4GPU 执行日志\n\n"
            f"生成时间：{now_sg()}\n\n"
            "记录原则：只记录真实执行过的命令、文件、状态和 blocker；不把未执行内容写成结果。\n",
            encoding="utf-8",
        )
    if not V2201_RECAP_DOC.exists():
        V2201_RECAP_DOC.write_text(
            "# DG-KAN v22.01 EarlySourceRetention TerminalCollapse KernelOfficialization DRAT/DRBF 4GPU 实验结果复盘\n\n"
            f"生成时间：{now_sg()}\n\n"
            "本文件由 v22.01 runner/finalizer 从落盘 artifacts 汇总；禁止编造数据。\n",
            encoding="utf-8",
        )


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


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    init_docs()
    row = {"timestamp": now_sg(), "command": command, "status": status, "note": note}
    journal = out_dir / "v22_01_command_journal.csv"
    exists = journal.exists() and journal.stat().st_size > 0
    with journal.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "command", "status", "note"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    append_text(V2201_EXEC_DOC, f"\n## {row['timestamp']}\n\n```bash\n{command}\n```\n\n- status: {status or 'recorded'}\n")
    if note:
        append_text(V2201_EXEC_DOC, f"- note: {note}\n")


def run_cmd(command: list[str], *, cwd: Path = ROOT, timeout: int = 300) -> tuple[int, str]:
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, timeout=timeout)
    return proc.returncode, "\n".join(["$ " + " ".join(command), f"exit={proc.returncode}", "--- stdout ---", proc.stdout, "--- stderr ---", proc.stderr])


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


def mean(rows: Iterable[dict[str, Any]], key: str) -> float:
    vals = [finite_float(r.get(key)) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


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


def simple_svg(path: Path, title: str, rows: list[dict[str, Any]], metric: str = "") -> None:
    vals = [finite_float(r.get(metric)) for r in rows] if metric else []
    vals = [v for v in vals if math.isfinite(v)]
    summary = f"rows={len(rows)}"
    if vals:
        summary += f" min={min(vals):.4g} mean={sum(vals)/len(vals):.4g} max={max(vals):.4g}"
    write_text(
        path,
        "<svg xmlns='http://www.w3.org/2000/svg' width='960' height='240'>"
        "<rect width='100%' height='100%' fill='#f7f7f5'/>"
        f"<text x='24' y='62' font-family='monospace' font-size='22'>{title}</text>"
        f"<text x='24' y='116' font-family='monospace' font-size='16'>{summary}</text>"
        "</svg>\n",
    )


def md_table(rows: list[dict[str, Any]], fields: list[str], max_rows: int = 30) -> str:
    if not rows:
        return "\n_无落盘 rows。_\n"
    shown = rows[:max_rows]
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in shown:
        out.append("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
    if len(rows) > max_rows:
        out.append(f"\n_仅显示前 {max_rows} / {len(rows)} rows；完整 CSV 见 artifact。_")
    return "\n".join(out) + "\n"


def build_packet(out_dir: Path) -> tuple[Path, Path]:
    packet = out_dir / "v22_01_code_review_packet"
    if packet.exists():
        shutil.rmtree(packet)
    for rel in [
        "00_README",
        "02_SOURCE_TREE/dgkan",
        "02_SOURCE_TREE/experiments",
        "03_IMPORT_CLOSURE",
        "04_LINEC_CORRECTNESS",
        "05_SOURCE_CHAIN_AND_RETENTION",
        "06_DEBT_ACCOUNTING",
        "07_FUNCTIONAL_MECHANISMS",
        "08_EFFICIENCY_KERNELS",
        "09_EXPERIMENT_RUNNERS",
        "10_GPU_QUEUE",
        "11_DOCS",
        "12_RESULTS",
    ]:
        (packet / rel).mkdir(parents=True, exist_ok=True)
    write_text(packet / "00_README" / "README.md", "v22.01 packet: source snapshots, truth-gate artifacts, route, logs, and reproduction commands.\n")
    source_files = [
        "dgkan/fu/source_chain.py",
        "dgkan/fu/debt_accounting.py",
        "dgkan/fu/mechanisms.py",
        "dgkan/fu/source_channel.py",
        "dgkan/fu/source_state.py",
        "dgkan/fu/function_space_actuation.py",
        "dgkan/fu/matrix_block.py",
        "dgkan/fu/poprisk_snr.py",
        "dgkan/kernels/cheby_fused.py",
        "dgkan/kernels/fourier_fused.py",
        "dgkan/kernels/rational_fused.py",
        "dgkan/kernels/rbf_sparse.py",
        "dgkan/profiling/efficiency_v20.py",
        "dgkan/profiling/efficiency_v21.py",
        "dgkan/profiling/efficiency_v22.py",
        "dgkan/profiling/efficiency_v22_01.py",
    ]
    runner_files = [
        "experiments/run_v22_01_common.py",
        "experiments/run_v22_01_s08_truth_gate.py",
        "experiments/run_v22_01_efficiency_officialization.py",
        "experiments/run_v22_01_drat_drbf_active_repair.py",
        "experiments/run_v22_01_terminal_collapse_autopsy.py",
        "experiments/run_v22_01_mlp_source_lab.py",
        "experiments/run_v22_01_function_space_target_reset.py",
        "experiments/run_v22_01_kan_source_writer.py",
        "experiments/run_v22_01_continue_terminal_repair.py",
        "experiments/run_v22_01_controls_and_finalize.py",
        "experiments/run_v21_efficiency_officialization.py",
        "experiments/run_v21_01_source_retention.py",
    ]
    for rel in source_files + runner_files:
        copy_if_exists(ROOT / rel, packet / "02_SOURCE_TREE" / rel)
    for path in sorted(out_dir.glob("v22_01_*")):
        if path.is_file() and path.suffix.lower() != ".zip":
            copy_if_exists(path, packet / "12_RESULTS" / path.name)
    for fig in sorted((out_dir / "figures").glob("*")):
        copy_if_exists(fig, packet / "12_RESULTS" / "figures" / fig.name)
    for doc in [V2201_PLAN_DOC, V2201_EXEC_DOC, V2201_RECAP_DOC]:
        copy_if_exists(doc, packet / "11_DOCS" / doc.name)
    manifest = []
    for path in sorted(packet.rglob("*")):
        if path.is_file():
            manifest.append({"path": str(path.relative_to(packet)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_rows(packet / "packet_manifest.csv", manifest)
    write_rows(packet / "packet_sha256_manifest.csv", manifest)
    zip_path = out_dir / "v22_01_code_review_packet.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(packet.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(packet))
    bundle = out_dir / "v22_01_results_bundle.zip"
    if bundle.exists():
        bundle.unlink()
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() != ".zip" and "v22_01_code_review_packet" not in str(path):
                z.write(path, path.relative_to(out_dir))
    return zip_path, bundle


__all__ = [
    "HORIZONS",
    "PYTHON",
    "ROOT",
    "V2200_OFFICIAL",
    "V2201_EXEC_DOC",
    "V2201_OFFICIAL",
    "V2201_PLAN_DOC",
    "V2201_RECAP_DOC",
    "append_exec",
    "append_text",
    "build_packet",
    "copy_if_exists",
    "ensure_out",
    "finite_float",
    "init_docs",
    "int_flag",
    "md_table",
    "mean",
    "now_sg",
    "read_json",
    "read_rows",
    "run_cmd",
    "sha256_file",
    "simple_svg",
    "write_json",
    "write_rows",
    "write_text",
]
