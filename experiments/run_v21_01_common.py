"""Shared helpers for v21.01 execution artifacts."""

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
V2101_ROOT = ROOT / "results/v21_01_source_retention_fu_kernel_officialization_4gpu"
V2101_OFFICIAL = V2101_ROOT / "official_v21_01"
V2101_EXEC_DOC = ROOT / "docs/DG-KAN_v21.01_SourceRetentionFU_KernelOfficialization_4GPU_执行日志.md"
V2101_RECAP_DOC = ROOT / "docs/DG-KAN_v21.01_SourceRetentionFU_KernelOfficialization_4GPU_实验结果复盘.md"
V2101_PLAN_DOC = ROOT / "docs/DG-KAN_v21.01_SourceRetentionFU_KernelOfficialization_4GPU_完整计划.md"
V21_OFFICIAL = ROOT / "results/v21_0_source_retention_kernel_officialization_4gpu/official_v21"

HORIZON_STEPS = (100, 400, 800, 1600, 2400, 3200, 4000, 4800, 6400)


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out(out_dir: str | Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else V2101_OFFICIAL
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
    init_docs()
    row = {"timestamp": now_sg(), "command": command, "status": status, "note": note}
    journal_path = out_dir / "v21_01_command_journal.csv"
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    exists = journal_path.exists() and journal_path.stat().st_size > 0
    with journal_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "command", "status", "note"])
        if not exists:
            writer.writeheader()
        writer.writerow({k: str(v) for k, v in row.items()})
    append_text(V2101_EXEC_DOC, f"\n## {row['timestamp']}\n\n```bash\n{command}\n```\n\n- status: {status or 'recorded'}\n")
    if note:
        append_text(V2101_EXEC_DOC, f"- note: {note}\n")


def run_cmd(command: list[str], *, cwd: Path = ROOT, timeout: int = 300) -> tuple[int, str]:
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, timeout=timeout)
    report = "\n".join(["$ " + " ".join(command), f"exit={proc.returncode}", "--- stdout ---", proc.stdout, "--- stderr ---", proc.stderr])
    return proc.returncode, report


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


def retention_ratio(current_source: Any, previous_source: Any, eps: float = 1.0e-12) -> float | str:
    previous = finite_float(previous_source)
    current = finite_float(current_source)
    if not math.isfinite(previous) or previous <= 0.0 or not math.isfinite(current):
        return ""
    return max(0.0, current) / max(float(eps), previous)


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text))


def merge_csvs(out_dir: Path, pattern: str, target: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(out_dir.glob(pattern)):
        if path.name == target:
            continue
        rows.extend(read_rows(path))
    write_rows(out_dir / target, rows)
    return rows


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
    if not V2101_EXEC_DOC.exists():
        write_text(
            V2101_EXEC_DOC,
            "# DG-KAN v21.01 SourceRetentionFU KernelOfficialization 4GPU 执行日志\n\n"
            f"生成时间：{now_sg()}\n\n"
            "本日志只记录真实执行命令、状态和 blocker；未运行项不写成完成。\n",
        )
    if not V2101_RECAP_DOC.exists():
        write_text(
            V2101_RECAP_DOC,
            "# DG-KAN v21.01 SourceRetentionFU KernelOfficialization 4GPU 实验结果复盘\n\n"
            f"生成时间：{now_sg()}\n\n"
            "## Route\n\n- route: `R0-InProgress`\n- promotion_allowed: 0\n\n"
            "## 审计边界\n\n- 本复盘只汇总落盘 artifact 与真实运行结果；不会把 v21.0 旧结果改名冒充 v21.01 fresh success。\n",
        )


def build_packet(out_dir: Path, required_artifacts: list[str]) -> None:
    packet = out_dir / "v21_01_code_review_packet"
    if packet.exists():
        shutil.rmtree(packet)
    for name in [
        "00_README.md",
        "01_ENVIRONMENT",
        "02_SOURCE_TREE",
        "03_IMPORT_CLOSURE",
        "04_LINEC_CORRECTNESS",
        "05_RETENTION_AND_DEBT",
        "06_UPDATE_SEMANTICS",
        "07_FUNCTIONAL_MECHANISMS",
        "08_EFFICIENCY_KERNELS",
        "09_PROFILER_CORRECTNESS",
        "10_EXPERIMENT_RUNNERS",
        "11_RESULTS",
        "12_REPRO_COMMANDS",
    ]:
        target = packet / name
        if Path(name).suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target.mkdir(parents=True, exist_ok=True)
    for cmd, name in [
        (["git", "status", "--short"], "git_status.txt"),
        (["git", "rev-parse", "HEAD"], "git_head.txt"),
        ([PYTHON, "--version"], "python_version.txt"),
        (["nvidia-smi"], "cuda_info.txt"),
    ]:
        _code, report = run_cmd(cmd, timeout=120)
        write_text(packet / "01_ENVIRONMENT" / name, report)
    source_roots = ["dgkan", "experiments", "tests"]
    for root_name in source_roots:
        root = ROOT / root_name
        if root.exists():
            for path in sorted(root.rglob("*.py")):
                copy_if_exists(path, packet / "02_SOURCE_TREE" / path.relative_to(ROOT))
    copy_if_exists(V2101_PLAN_DOC, packet / "02_SOURCE_TREE/docs" / V2101_PLAN_DOC.name)
    copy_if_exists(V2101_EXEC_DOC, packet / "12_REPRO_COMMANDS" / V2101_EXEC_DOC.name)
    copy_if_exists(V2101_RECAP_DOC, packet / "12_REPRO_COMMANDS" / V2101_RECAP_DOC.name)
    section_map = {
        "03_IMPORT_CLOSURE": ["v21_01_code_truth_gate.csv", "v21_01_import_closure.csv", "v21_01_import_error_details.csv", "v21_01_compileall.log", "v21_01_required_source_files.csv"],
        "04_LINEC_CORRECTNESS": ["v21_01_linec_fast_golden.csv", "v21_01_linec_channel_golden.csv"],
        "05_RETENTION_AND_DEBT": ["v21_01_retention_formula_results.csv", "v21_01_debt_accounting_results.csv", "v21_01_route_aggregation_results.csv"],
        "06_UPDATE_SEMANTICS": ["v21_01_update_sign_results.csv", "v21_01_update_space_kind_contract.csv", "v21_01_optimizer_coupling_contract.csv"],
        "07_FUNCTIONAL_MECHANISMS": ["v21_01_mechanism_semantic_contract.csv", "v21_01_mechanism_noncollapse_results.csv", "v21_01_source_retention_summary.csv", "v21_01_source_dynamics_matrix.csv"],
        "08_EFFICIENCY_KERNELS": ["v21_01_kernel_gradcheck_results.csv", "v21_01_official_fused_status_matrix.csv", "v21_01_efficiency_truth_table.csv", "v21_01_full_loop_efficiency_matrix.csv"],
        "09_PROFILER_CORRECTNESS": ["v21_01_profiler_phase_results.csv", "v21_01_profiler_vs_full_loop_results.csv"],
        "10_EXPERIMENT_RUNNERS": ["v21_01_command_journal.csv"],
        "11_RESULTS": required_artifacts,
    }
    for section, names in section_map.items():
        for name in names:
            copy_if_exists(out_dir / name, packet / section / Path(name).name)
    write_text(packet / "00_README.md", "\n".join(["# v21.01 Code Review Packet", f"- generated_at: {now_sg()}", f"- result_dir: {out_dir}", "- absent evidence is blocked, not promoted."]) + "\n")
    rows = []
    for path in sorted(p for p in packet.rglob("*") if p.is_file()):
        rel = path.relative_to(packet)
        if rel.name in {"packet_manifest.csv", "packet_sha256_manifest.csv"}:
            continue
        rows.append({"relative_path": str(rel), "sha256": sha256_file(path), "bytes": path.stat().st_size, "artifact_type": rel.parts[0], "required": 1})
    write_rows(packet / "packet_manifest.csv", rows)
    write_rows(packet / "packet_sha256_manifest.csv", [{"relative_path": r["relative_path"], "sha256": r["sha256"], "bytes": r["bytes"]} for r in rows])
    with zipfile.ZipFile(out_dir / "v21_01_code_review_packet.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in packet.rglob("*") if p.is_file()):
            zf.write(path, Path("v21_01_code_review_packet") / path.relative_to(packet))
    with zipfile.ZipFile(out_dir / "v21_01_results_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in out_dir.rglob("*") if p.is_file() and "v21_01_code_review_packet/" not in str(p.relative_to(out_dir)) and p.name != "v21_01_results_bundle.zip"):
            zf.write(path, path.relative_to(out_dir))
