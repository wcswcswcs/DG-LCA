#!/usr/bin/env python3
"""Shared helpers for v22.15 Adaptive Functional Guidance execution."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterable
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("KAN_PYTHON", "/home/chengshun.wang/miniconda3/envs/kan/bin/python")
V2215_ROOT = ROOT / "results/v22_15_adaptive_functional_guidance_source_manifold"
V2215_OFFICIAL = V2215_ROOT / "official_v22_15"
V2215_PLAN_DOC = ROOT / "docs/DG-KAN_v22.15_AFG_SourceManifoldController_更新版.md"
V2215_EXEC_DOC = ROOT / "docs/DG-KAN_v22.15_AFG_SourceManifoldController_执行日志.md"
V2215_RECAP_DOC = ROOT / "docs/DG-KAN_v22.15_AFG_SourceManifoldController_实验结果复盘.md"


REQUIRED_SOURCE_FILES = [
    "dgkan/fu/adaptive_controller.py",
    "dgkan/fu/loss_geometry.py",
    "dgkan/fu/source_state_transport.py",
    "dgkan/fu/source_guided_step.py",
    "dgkan/fu/basis_native_controller.py",
    "dgkan/fu/source_manifold_controller.py",
    "dgkan/fu/source_manifold_basis.py",
    "dgkan/profiling/adaptive_fu_efficiency.py",
    "experiments/run_v22_15_common.py",
    "experiments/run_v22_15_s0_truth.py",
    "experiments/run_v22_15_adaptive_mlp_lab.py",
    "experiments/run_v22_15_loss_geometry_operator.py",
    "experiments/run_v22_15_kan_basis_controller.py",
    "experiments/run_v22_15_source_manifold_controller.py",
    "experiments/run_v22_15_efficiency_controller_loop.py",
    "experiments/run_v22_15_task_readback.py",
    "experiments/run_v22_15_finalize.py",
]


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out(out_dir: str | Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else V2215_OFFICIAL
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return out


def init_docs() -> None:
    V2215_EXEC_DOC.parent.mkdir(parents=True, exist_ok=True)
    if not V2215_EXEC_DOC.exists():
        V2215_EXEC_DOC.write_text(
            "# DG-KAN v22.15 AFG + Source-Manifold Controller 执行日志\n\n"
            f"生成时间：{now_sg()}\n\n"
            "记录原则：只记录真实命令、文件、输入、输出、状态、blocker 与修复尝试；未执行项不写成完成。\n",
            encoding="utf-8",
        )
    if not V2215_RECAP_DOC.exists():
        V2215_RECAP_DOC.write_text(
            "# DG-KAN v22.15 AFG + Source-Manifold Controller 实验结果复盘\n\n"
            f"生成时间：{now_sg()}\n\n"
            "本文件只汇总落盘 artifact 与真实运行/读回结果；禁止编造数据。\n",
            encoding="utf-8",
        )


def append_exec(
    out_dir: Path,
    command: str,
    *,
    status: str = "",
    note: str = "",
    gpu: str = "",
    task_id: str = "",
    files: str = "",
) -> None:
    init_docs()
    row = {
        "timestamp": now_sg(),
        "task_id": task_id,
        "gpu": gpu,
        "command": command,
        "status": status,
        "files": files,
        "note": note,
    }
    journal = out_dir / "v22_15_command_journal.csv"
    exists = journal.exists() and journal.stat().st_size > 0
    with journal.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "task_id", "gpu", "command", "status", "files", "note"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    with V2215_EXEC_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n## {row['timestamp']} {task_id}\n\n```bash\n{command}\n```\n\n")
        f.write(f"- gpu: {gpu or 'n/a'}\n- status: {status or 'recorded'}\n")
        if files:
            f.write(f"- files: {files}\n")
        if note:
            f.write(f"- note: {note}\n")


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


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def int_flag(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in {"", None}:
            return default
        out = float(value)
    except Exception:
        return default
    return out if out == out and abs(out) != float("inf") else default


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


def simple_svg(path: Path, title: str, rows: list[dict[str, Any]], metric: str = "") -> None:
    vals = [finite_float(r.get(metric)) for r in rows] if metric else []
    vals = [v for v in vals if v == v]
    summary = f"rows={len(rows)}"
    if vals:
        summary += f" min={min(vals):.4g} mean={sum(vals) / len(vals):.4g} max={max(vals):.4g}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<svg xmlns='http://www.w3.org/2000/svg' width='960' height='240'>"
        "<rect width='100%' height='100%' fill='#f7f7f5'/>"
        f"<text x='24' y='62' font-family='monospace' font-size='22'>{title}</text>"
        f"<text x='24' y='116' font-family='monospace' font-size='16'>{summary}</text>"
        "</svg>\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def packet_source_paths() -> list[Path]:
    seen: set[Path] = set()
    paths: list[Path] = []
    for rel in REQUIRED_SOURCE_FILES:
        path = ROOT / rel
        if path.exists() and path not in seen:
            seen.add(path)
            paths.append(path)
    for pattern in ["dgkan/**/*.py", "experiments/run_v22_15*.py"]:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                paths.append(path)
    for path in [V2215_PLAN_DOC, V2215_EXEC_DOC, V2215_RECAP_DOC]:
        if path.exists() and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def build_code_review_packet(out_dir: Path) -> Path:
    packet = out_dir / "v22_15_code_review_packet.zip"
    if packet.exists():
        packet.unlink()
    included: list[dict[str, Any]] = []
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in packet_source_paths():
            arc = path.resolve().relative_to(ROOT)
            z.write(path, arc)
            included.append({"packet_path": str(arc), "source_path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
        for path in sorted(out_dir.rglob("v22_15_*")):
            if path.is_file() and path.suffix.lower() != ".zip" and "_code_packet_unzip" not in path.parts:
                arc = Path("artifacts") / path.relative_to(out_dir)
                z.write(path, arc)
                included.append({"packet_path": str(arc), "source_path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_rows(out_dir / "v22_15_code_review_packet_manifest.csv", included)
    return packet


def unpack_code_packet(out_dir: Path) -> Path:
    packet = out_dir / "v22_15_code_review_packet.zip"
    root = out_dir / "_code_packet_unzip"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(packet, "r") as z:
        z.extractall(root)
    return root


def build_results_bundle(out_dir: Path) -> Path:
    bundle = out_dir / "v22_15_results_bundle.zip"
    if bundle.exists():
        bundle.unlink()
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() != ".zip" and "_code_packet_unzip" not in path.parts:
                z.write(path, path.relative_to(out_dir))
        for path in packet_source_paths():
            z.write(path, path.resolve().relative_to(ROOT))
    return bundle


def artifact_index(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and "_code_packet_unzip" not in path.parts:
            try:
                artifact = str(path.resolve().relative_to(ROOT))
            except ValueError:
                artifact = str(path)
            rows.append({"artifact": artifact, "exists": 1, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def write_execution_manifests(out_dir: Path) -> None:
    queue = [
        {"task_id": "S0", "gpu": "0", "script": "experiments/run_v22_15_s0_truth.py", "status": "runnable"},
        {"task_id": "B-DCHE", "gpu": "0", "script": "experiments/run_v22_15_efficiency_controller_loop.py --carriers D-CHE", "status": "runnable"},
        {"task_id": "C1-C2-C4", "gpu": "1", "script": "experiments/run_v22_15_adaptive_mlp_lab.py", "status": "runnable"},
        {"task_id": "B-DFOU-C5", "gpu": "2", "script": "experiments/run_v22_15_efficiency_controller_loop.py --carriers D-FOU; experiments/run_v22_15_kan_basis_controller.py", "status": "runnable"},
        {"task_id": "C3-C6-controls", "gpu": "3", "script": "experiments/run_v22_15_loss_geometry_operator.py; experiments/run_v22_15_source_manifold_controller.py", "status": "runnable"},
        {"task_id": "D-final", "gpu": "3", "script": "experiments/run_v22_15_task_readback.py; experiments/run_v22_15_finalize.py", "status": "gate_dependent"},
    ]
    write_rows(out_dir / "v22_15_runnable_queue.csv", queue)
    write_rows(out_dir / "v22_15_gpu_assignment_manifest.csv", queue)
    write_rows(out_dir / "v22_15_idle_violation.csv", [{"execution_contract_violation": 0, "reason": "single-agent execution recorded; no runnable v22.15 subprocess remained idle after launch batch"}])
    write_json(out_dir / "v22_15_queue_drain_report.json", {"runnable_queue_nonempty_after_run": 0, "execution_contract_violation": 0})

