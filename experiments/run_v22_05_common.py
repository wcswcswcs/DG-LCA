#!/usr/bin/env python3
"""Shared helpers for v22.05 execution, audit logs, and packets."""

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
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("KAN_PYTHON", "/home/chengshun.wang/miniconda3/envs/kan/bin/python")
V2204_OFFICIAL = ROOT / "results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04"
V2205_ROOT = ROOT / "results/v22_05_metric_first_functional_update_basis_efficiency_4gpu"
V2205_OFFICIAL = V2205_ROOT / "official_v22_05"
V2205_PLAN_DOC = ROOT / "docs/DG-KAN_v22.05_MetricFirst_FunctionalUpdate_BasisEfficiency_4GPU_完整计划.md"
V2205_EXEC_DOC = ROOT / "docs/DG-KAN_v22.05_MetricFirst_FunctionalUpdate_BasisEfficiency_4GPU_执行日志.md"
V2205_RECAP_DOC = ROOT / "docs/DG-KAN_v22.05_MetricFirst_FunctionalUpdate_BasisEfficiency_4GPU_实验结果复盘.md"


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out(out_dir: str | Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else V2205_OFFICIAL
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    return out


def init_docs() -> None:
    V2205_EXEC_DOC.parent.mkdir(parents=True, exist_ok=True)
    if not V2205_EXEC_DOC.exists():
        V2205_EXEC_DOC.write_text(
            "# DG-KAN v22.05 MetricFirst FunctionalUpdate BasisEfficiency 4GPU 执行日志\n\n"
            f"生成时间：{now_sg()}\n\n"
            "记录原则：只记录真实执行过的命令、输入文件、输出 artifact、状态和 blocker；不把未执行内容写成结果。\n",
            encoding="utf-8",
        )
    if not V2205_RECAP_DOC.exists():
        V2205_RECAP_DOC.write_text(
            "# DG-KAN v22.05 MetricFirst FunctionalUpdate BasisEfficiency 4GPU 实验结果复盘\n\n"
            f"生成时间：{now_sg()}\n\n"
            "本文件由 v22.05 runner/finalizer 从落盘 artifacts 汇总；禁止编造数据。\n",
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
    journal = out_dir / "v22_05_command_journal.csv"
    exists = journal.exists() and journal.stat().st_size > 0
    with journal.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "command", "status", "note"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    append_text(V2205_EXEC_DOC, f"\n## {row['timestamp']}\n\n```bash\n{command}\n```\n\n- status: {status or 'recorded'}\n")
    if note:
        append_text(V2205_EXEC_DOC, f"- note: {note}\n")


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
    return out if out == out and abs(out) != float("inf") else default


def int_flag(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def mean(rows: Iterable[dict[str, Any]], key: str) -> float:
    vals = [finite_float(r.get(key)) for r in rows]
    vals = [v for v in vals if v == v]
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
        summary += f" min={min(vals):.4g} mean={sum(vals)/len(vals):.4g} max={max(vals):.4g}"
    write_text(
        path,
        "<svg xmlns='http://www.w3.org/2000/svg' width='960' height='240'>"
        "<rect width='100%' height='100%' fill='#f7f7f5'/>"
        f"<text x='24' y='62' font-family='monospace' font-size='22'>{title}</text>"
        f"<text x='24' y='116' font-family='monospace' font-size='16'>{summary}</text>"
        "</svg>\n",
    )


def _ignore_for_packet(_dir: str, names: list[str]) -> set[str]:
    ignored = {"__pycache__", ".pytest_cache"}
    ignored.update(name for name in names if name.endswith(".pyc"))
    return ignored


def build_packet(out_dir: Path) -> tuple[Path, Path]:
    packet = out_dir / "v22_05_code_review_packet"
    if packet.exists():
        shutil.rmtree(packet)
    for rel in [
        "00_README.md",
        "02_SOURCE_TREE",
        "03_IMPORT_CLOSURE",
        "04_LINEC_CORRECTNESS",
        "05_SOURCE_CHAIN_RETENTION_DEBT",
        "06_MECHANISM_SEMANTIC_CONTRACTS",
        "07_FUNCTION_SPACE_METRICS",
        "08_BASIS_KERNELS_AND_PROFILERS",
        "09_GPU_QUEUE_CONTRACT",
        "10_RESULTS_POINTERS",
    ]:
        target = packet / rel
        if Path(rel).suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("v22.05 code review packet\n", encoding="utf-8")
        else:
            target.mkdir(parents=True, exist_ok=True)
    write_text(packet / "00_README.md", "v22.05 packet: self-contained source tree, truth-gate artifacts, logs, and reproduction commands.\n")
    source_tree = packet / "02_SOURCE_TREE"
    shutil.copytree(ROOT / "dgkan", source_tree / "dgkan", ignore=_ignore_for_packet)
    (source_tree / "experiments").mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / "experiments").glob("run_v*.py")):
        copy_if_exists(path, source_tree / "experiments" / path.name)
    if (ROOT / "tests").exists():
        shutil.copytree(ROOT / "tests", source_tree / "tests", ignore=_ignore_for_packet)
    for path in sorted(out_dir.glob("v22_05_*")):
        if path.is_file() and path.suffix.lower() != ".zip":
            section = "10_RESULTS_POINTERS"
            if "import" in path.name or "compile" in path.name or "clean_unzip" in path.name:
                section = "03_IMPORT_CLOSURE"
            elif "linec" in path.name:
                section = "04_LINEC_CORRECTNESS"
            elif "source" in path.name or "retention" in path.name or "debt" in path.name:
                section = "05_SOURCE_CHAIN_RETENTION_DEBT"
            elif "mechanism" in path.name or "semantic" in path.name:
                section = "06_MECHANISM_SEMANTIC_CONTRACTS"
            elif "metric" in path.name or "nds" in path.name.lower():
                section = "07_FUNCTION_SPACE_METRICS"
            elif "efficiency" in path.name or "drat" in path.name.lower() or "drbf" in path.name.lower():
                section = "08_BASIS_KERNELS_AND_PROFILERS"
            elif "gpu" in path.name or "queue" in path.name or "idle" in path.name:
                section = "09_GPU_QUEUE_CONTRACT"
            copy_if_exists(path, packet / section / path.name)
    for fig in sorted((out_dir / "figures").glob("*")):
        copy_if_exists(fig, packet / "10_RESULTS_POINTERS" / "figures" / fig.name)
    for doc in [V2205_PLAN_DOC, V2205_EXEC_DOC, V2205_RECAP_DOC]:
        copy_if_exists(doc, packet / "10_RESULTS_POINTERS" / "docs" / doc.name)
    write_text(
        packet / "10_RESULTS_POINTERS" / "clean_unzip_self_test.sh",
        f"{PYTHON} -m compileall -q dgkan experiments tests\n"
        f"{PYTHON} experiments/run_v22_05_s012_truth_gate.py --mode all --source-root . --self-contained-import-check 1\n",
    )
    manifest = []
    for path in sorted(packet.rglob("*")):
        if path.is_file():
            manifest.append({"path": str(path.relative_to(packet)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_rows(packet / "packet_manifest.csv", manifest)
    write_rows(packet / "packet_sha256_manifest.csv", manifest)
    zip_path = out_dir / "v22_05_code_review_packet.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(packet.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(packet))
    bundle = out_dir / "v22_05_results_bundle.zip"
    if bundle.exists():
        bundle.unlink()
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() != ".zip" and "v22_05_code_review_packet" not in str(path):
                z.write(path, path.relative_to(out_dir))
    return zip_path, bundle
