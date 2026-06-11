#!/usr/bin/env python3
"""Shared helpers for v22.08 execution, artifacts, and Markdown logs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("KAN_PYTHON", "/home/chengshun.wang/miniconda3/envs/kan/bin/python")
V2208_ROOT = ROOT / "results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu"
V2208_OFFICIAL = V2208_ROOT / "official_v22_08"
V2208_PLAN_DOC = ROOT / "docs/DG-KAN_v22.08_RetainedSourceDynamics_FunctionalUpdate_BasisEfficiency_4GPU_完整计划.md"
V2208_EXEC_DOC = ROOT / "docs/DG-KAN_v22.08_RetainedSourceDynamics_FunctionalUpdate_BasisEfficiency_4GPU_执行日志.md"
V2208_RECAP_DOC = ROOT / "docs/DG-KAN_v22.08_RetainedSourceDynamics_FunctionalUpdate_BasisEfficiency_4GPU_实验结果复盘.md"
V2204_OFFICIAL = ROOT / "results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04"
V2206_OFFICIAL = ROOT / "results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06"
V2206_COMBINED_SOURCE = V2206_OFFICIAL / "metric_solver_source_smoke_combined_v9_to_v19_readback"
V2207_OFFICIAL = ROOT / "results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07"


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out(out_dir: str | Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else V2208_OFFICIAL
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    return out


def init_docs() -> None:
    V2208_EXEC_DOC.parent.mkdir(parents=True, exist_ok=True)
    if not V2208_EXEC_DOC.exists():
        V2208_EXEC_DOC.write_text(
            "# DG-KAN v22.08 RetainedSourceDynamics FunctionalUpdate BasisEfficiency 4GPU 执行日志\n\n"
            f"生成时间：{now_sg()}\n\n"
            "记录原则：只记录真实命令、输入、输出、状态、blocker 与修复尝试；未执行项不写成完成。\n",
            encoding="utf-8",
        )
    if not V2208_RECAP_DOC.exists():
        V2208_RECAP_DOC.write_text(
            "# DG-KAN v22.08 RetainedSourceDynamics FunctionalUpdate BasisEfficiency 4GPU 实验结果复盘\n\n"
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
    journal = out_dir / "v22_08_command_journal.csv"
    exists = journal.exists() and journal.stat().st_size > 0
    with journal.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "command", "status", "note"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    append_text(V2208_EXEC_DOC, f"\n## {row['timestamp']}\n\n```bash\n{command}\n```\n\n- status: {status or 'recorded'}\n")
    if note:
        append_text(V2208_EXEC_DOC, f"- note: {note}\n")


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


def build_results_bundle(out_dir: Path) -> Path:
    bundle = out_dir / "v22_08_results_bundle.zip"
    if bundle.exists():
        bundle.unlink()
    code_paths = sorted((ROOT / "experiments").glob("run_v22_08_*.py"))
    code_paths.extend(
        [
            ROOT / "dgkan/fu/metric_solver.py",
            ROOT / "dgkan/fu/mechanisms.py",
            ROOT / "dgkan/fu/core.py",
        ]
    )
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() != ".zip":
                z.write(path, path.relative_to(out_dir))
        for doc in [V2208_PLAN_DOC, V2208_EXEC_DOC, V2208_RECAP_DOC]:
            if doc.exists():
                z.write(doc, Path("docs") / doc.name)
        for path in code_paths:
            if path.exists() and path.is_file():
                z.write(path, path.resolve().relative_to(ROOT))
    return bundle


def build_code_review_packet(out_dir: Path) -> Path:
    packet = out_dir / "v22_08_code_review_packet.zip"
    if packet.exists():
        packet.unlink()
    code_paths = sorted((ROOT / "experiments").glob("run_v22_08_*.py"))
    code_paths.extend(
        [
            ROOT / "dgkan/fu/metric_solver.py",
            ROOT / "dgkan/fu/mechanisms.py",
            ROOT / "dgkan/fu/core.py",
        ]
    )
    doc_paths = [V2208_PLAN_DOC, V2208_EXEC_DOC, V2208_RECAP_DOC]
    key_artifacts = [
        out_dir / "v22_08_final_route.json",
        out_dir / "v22_08_final_route.csv",
        out_dir / "v22_08_queue_drain_report.json",
        out_dir / "v22_08_gpu_assignment_manifest.csv",
        out_dir / "v22_08_command_journal.csv",
        out_dir / "v22_08_code_truth_gate.csv",
        out_dir / "v22_08_c2_recompute_tests.csv",
        out_dir / "v22_08_retained_source_observer_summary.csv",
        out_dir / "v22_08_retained_source_observer_top_candidates.csv",
        out_dir / "v22_08_metric_dynamics_solver_route.json",
        out_dir / "v22_08_post_nogo_source_state_route.json",
        out_dir / "v22_08_post_nogo_source_state_certificate_summary.csv",
        out_dir / "v22_08_post_nogo_source_state_selected_candidates.csv",
        out_dir / "v22_08_post_nogo_source_state_fresh_c3_summary.csv",
        out_dir / "v22_08_post_nogo_counterfactual_washout_route.json",
        out_dir / "v22_08_post_nogo_counterfactual_washout_summary.csv",
        out_dir / "v22_08_post_nogo_counterfactual_washout_selected_candidates.csv",
        out_dir / "v22_08_post_nogo_counterfactual_washout_fresh_c3_summary.csv",
        out_dir / "v22_08_post_nogo_ntk_channel_route.json",
        out_dir / "v22_08_post_nogo_ntk_channel_summary.csv",
        out_dir / "v22_08_post_nogo_ntk_channel_selected_candidates.csv",
        out_dir / "v22_08_post_nogo_ntk_channel_fresh_c3_summary.csv",
        out_dir / "v22_08_post_nogo_aug_tangent_route.json",
        out_dir / "v22_08_post_nogo_aug_tangent_summary.csv",
        out_dir / "v22_08_post_nogo_aug_tangent_selected_candidates.csv",
        out_dir / "v22_08_post_nogo_aug_tangent_fresh_c3_summary.csv",
        out_dir / "v22_08_post_nogo_bootstrap_basin_route.json",
        out_dir / "v22_08_post_nogo_bootstrap_basin_summary.csv",
        out_dir / "v22_08_post_nogo_bootstrap_basin_selected_candidates.csv",
        out_dir / "v22_08_post_nogo_bootstrap_basin_fresh_c3_summary.csv",
        out_dir / "v22_08_post_nogo_crossfit_influence_route.json",
        out_dir / "v22_08_post_nogo_crossfit_influence_summary.csv",
        out_dir / "v22_08_post_nogo_crossfit_influence_selected_candidates.csv",
        out_dir / "v22_08_post_nogo_crossfit_influence_fresh_c3_summary.csv",
        out_dir / "v22_08_post_nogo_train_flow_commutator_route.json",
        out_dir / "v22_08_post_nogo_train_flow_commutator_summary.csv",
        out_dir / "v22_08_post_nogo_train_flow_commutator_selected_candidates.csv",
        out_dir / "v22_08_post_nogo_train_flow_commutator_fresh_c3_summary.csv",
        out_dir / "v22_08_train_flow_commutator_verify_route.json",
        out_dir / "v22_08_train_flow_commutator_verify_repeat_summary.csv",
        out_dir / "v22_08_train_flow_commutator_verify_fresh_c3_summary.csv",
        out_dir / "v22_08_artifact_index.csv",
    ]
    included: list[dict[str, Any]] = []
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in code_paths + doc_paths + key_artifacts:
            if not path.exists() or not path.is_file():
                continue
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
    write_rows(out_dir / "v22_08_code_review_packet_manifest.csv", included)
    return packet
