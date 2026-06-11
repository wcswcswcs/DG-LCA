#!/usr/bin/env python3
"""Shared helpers for v22.09 execution, artifacts, packets, and logs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterable
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("KAN_PYTHON", "/home/chengshun.wang/miniconda3/envs/kan/bin/python")
V2209_ROOT = ROOT / "results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu"
V2209_OFFICIAL = V2209_ROOT / "official_v22_09"
V2209_PLAN_DOC = ROOT / "docs/DG-KAN_v22.09_RetainedSourceObservability_FunctionalUpdate_BasisEfficiency_4GPU完整计划.md"
V2209_EXEC_DOC = ROOT / "docs/DG-KAN_v22.09_RetainedSourceObservability_FunctionalUpdate_BasisEfficiency_4GPU_执行日志.md"
V2209_RECAP_DOC = ROOT / "docs/DG-KAN_v22.09_RetainedSourceObservability_FunctionalUpdate_BasisEfficiency_4GPU_实验结果复盘.md"
V2204_OFFICIAL = ROOT / "results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04"
V2206_OFFICIAL = ROOT / "results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06"
V2206_COMBINED_SOURCE = V2206_OFFICIAL / "metric_solver_source_smoke_combined_v9_to_v19_readback"
V2207_OFFICIAL = ROOT / "results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07"
V2208_OFFICIAL = ROOT / "results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08"


REQUIRED_SOURCE_FILES = [
    "dgkan/fu/core.py",
    "dgkan/fu/source_chain.py",
    "dgkan/fu/terminal_retention.py",
    "dgkan/fu/debt_accounting.py",
    "dgkan/fu/function_space_metrics.py",
    "dgkan/fu/metric_projection.py",
    "dgkan/fu/metric_solver.py",
    "dgkan/fu/jacobian_sketch.py",
    "dgkan/fu/sobolev_metric.py",
    "dgkan/fu/rkhs_metric.py",
    "dgkan/fu/fisher_metric.py",
    "dgkan/fu/basis_channel_metric.py",
    "dgkan/fu/retained_source_certificate.py",
    "dgkan/fu/train_flow_commutator.py",
    "dgkan/fu/source_state_dynamics.py",
    "dgkan/fu/optimizer_state_integration.py",
    "dgkan/metrics/linec.py",
    "dgkan/kernels/fused_chebyshev_k3.py",
    "dgkan/kernels/fused_fourier_k2.py",
    "dgkan/kernels/fused_rational_k4.py",
    "dgkan/kernels/fused_rbf.py",
    "dgkan/kernels/rational_fused.py",
    "dgkan/kernels/rbf_sparse.py",
    "dgkan/profiling/efficiency_v22_06.py",
    "dgkan/profiling/efficiency_v22_09.py",
    "experiments/run_v22_09_common.py",
    "experiments/run_v22_09_s016_truth_gate.py",
    "experiments/run_v22_09_basis_efficiency_closure.py",
    "experiments/run_v22_09_drat_drbf_shape_localk_repair_scan.py",
    "experiments/run_v22_09_retained_source_observer.py",
    "experiments/run_v22_09_post_boundary_flow_replay_certificate.py",
    "experiments/run_v22_09_post_optimizer_state_holonomy_certificate.py",
    "experiments/run_v22_09_post_row_orthogonal_source_transport_certificate.py",
    "experiments/run_v22_09_post_info_volume_signal_channel_certificate.py",
    "experiments/run_v22_09_post_fast_slow_memory_resonance_certificate.py",
    "experiments/run_v22_09_post_fast_slow_memory_damped_repair.py",
    "experiments/run_v22_09_post_time_reversal_adjoint_certificate.py",
    "experiments/run_v22_09_post_time_reversal_adjoint_damped_repair.py",
    "experiments/run_v22_09_metric_dynamics_solver.py",
    "experiments/run_v22_09_terminal_preservation.py",
    "experiments/run_v22_09_kan_mapping.py",
    "experiments/run_v22_09_finalize.py",
    "experiments/run_v22_09_full.py",
]


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out(out_dir: str | Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else V2209_OFFICIAL
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    return out


def init_docs() -> None:
    V2209_EXEC_DOC.parent.mkdir(parents=True, exist_ok=True)
    if not V2209_EXEC_DOC.exists():
        V2209_EXEC_DOC.write_text(
            "# DG-KAN v22.09 RetainedSourceObservability FunctionalUpdate BasisEfficiency 4GPU 执行日志\n\n"
            f"生成时间：{now_sg()}\n\n"
            "记录原则：只记录真实命令、输入、输出、状态、blocker 与修复尝试；未执行项不写成完成。\n",
            encoding="utf-8",
        )
    if not V2209_RECAP_DOC.exists():
        V2209_RECAP_DOC.write_text(
            "# DG-KAN v22.09 RetainedSourceObservability FunctionalUpdate BasisEfficiency 4GPU 实验结果复盘\n\n"
            f"生成时间：{now_sg()}\n\n"
            "本文件只汇总落盘 artifact 与真实运行/读回结果；禁止编造数据。\n",
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
    out_dir.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": now_sg(), "command": command, "status": status, "note": note}
    journal = out_dir / "v22_09_command_journal.csv"
    exists = journal.exists() and journal.stat().st_size > 0
    with journal.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "command", "status", "note"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    append_text(V2209_EXEC_DOC, f"\n## {row['timestamp']}\n\n```bash\n{command}\n```\n\n- status: {status or 'recorded'}\n")
    if note:
        append_text(V2209_EXEC_DOC, f"- note: {note}\n")


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
    vals = [v for v in vals if math.isfinite(v)]
    summary = f"rows={len(rows)}"
    if vals:
        summary += f" min={min(vals):.4g} mean={sum(vals) / len(vals):.4g} max={max(vals):.4g}"
    write_text(
        path,
        "<svg xmlns='http://www.w3.org/2000/svg' width='960' height='240'>"
        "<rect width='100%' height='100%' fill='#f7f7f5'/>"
        f"<text x='24' y='62' font-family='monospace' font-size='22'>{title}</text>"
        f"<text x='24' y='116' font-family='monospace' font-size='16'>{summary}</text>"
        "</svg>\n",
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_index(out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            try:
                artifact = str(path.resolve().relative_to(ROOT))
            except ValueError:
                artifact = str(path)
            rows.append({"artifact": artifact, "exists": 1, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def packet_source_paths() -> list[Path]:
    seen: set[Path] = set()
    paths: list[Path] = []
    for path in [ROOT / rel for rel in REQUIRED_SOURCE_FILES]:
        if path not in seen:
            seen.add(path)
            paths.append(path)
    for path in sorted((ROOT / "dgkan").glob("**/*.py")):
        if path not in seen:
            seen.add(path)
            paths.append(path)
    for path in sorted((ROOT / "experiments").glob("run_v*.py")):
        if path not in seen:
            seen.add(path)
            paths.append(path)
    paths.extend(sorted((ROOT / "tests").glob("**/*.py")))
    docs = [V2209_PLAN_DOC, V2209_EXEC_DOC, V2209_RECAP_DOC]
    return [p for p in paths + docs if p.exists() and p.is_file()]


def build_code_review_packet(out_dir: Path) -> Path:
    packet = out_dir / "v22_09_code_review_packet.zip"
    if packet.exists():
        packet.unlink()
    included: list[dict[str, Any]] = []
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in packet_source_paths():
            try:
                arcname = path.resolve().relative_to(ROOT)
            except ValueError:
                arcname = Path("artifacts") / path.name
            z.write(path, arcname)
            included.append(
                {
                    "packet_path": str(arcname),
                    "source_path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        for path in sorted(out_dir.rglob("v22_09_*")):
            if path.is_file() and path.suffix.lower() != ".zip":
                arcname = Path("artifacts") / path.relative_to(out_dir)
                z.write(path, arcname)
                included.append(
                    {
                        "packet_path": str(arcname),
                        "source_path": str(path.resolve()),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    write_rows(out_dir / "v22_09_packet_manifest.csv", included)
    write_rows(out_dir / "v22_09_packet_sha256_manifest.csv", included)
    write_rows(out_dir / "v22_09_code_review_packet_manifest.csv", included)
    return packet


def unpack_code_packet(out_dir: Path) -> Path:
    packet = out_dir / "v22_09_code_review_packet.zip"
    unzip_root = out_dir / "_code_packet_unzip"
    if unzip_root.exists():
        shutil.rmtree(unzip_root)
    unzip_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(packet, "r") as z:
        z.extractall(unzip_root)
    return unzip_root


def build_results_bundle(out_dir: Path) -> Path:
    bundle = out_dir / "v22_09_results_bundle.zip"
    if bundle.exists():
        bundle.unlink()
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() != ".zip" and "_code_packet_unzip" not in path.parts:
                z.write(path, path.relative_to(out_dir))
        for path in packet_source_paths():
            try:
                z.write(path, path.resolve().relative_to(ROOT))
            except ValueError:
                z.write(path, Path("external") / path.name)
    return bundle
