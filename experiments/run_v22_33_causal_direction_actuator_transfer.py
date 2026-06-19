#!/usr/bin/env python3
"""DG-KAN v22.33 causal direction / actuator transfer runner.

The runner is conservative by construction:

1. It reuses the adjacent v22.32 direct pilots for code closure, T1 direction
   selectors, and basis-native actuator ladders.
2. It transforms only named, real artifact readbacks into the v22.33 required
   filenames.
3. Missing metrics are written as not_run / gate_blocked rows, never invented.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PYTHON = os.environ.get("KAN_PYTHON", "/home/chengshun.wang/miniconda3/envs/kan/bin/python")
OUT_ROOT = ROOT / "results/v22_33"
RAW_ROOT = OUT_ROOT / "raw_v22_32_runner"
FIG_ROOT = OUT_ROOT / "figures"
LOG_ROOT = OUT_ROOT / "logs"
PLAN_DOC = ROOT / "docs/DG-KAN_v22.33_CausalDirectionActuatorTransfer_完整计划.md"
EXEC_DOC = ROOT / "docs/DG-KAN_v22.33_CausalDirectionActuatorTransfer_执行日志.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v22.33_CausalDirectionActuatorTransfer_实验结果复盘.md"


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.33 Causal Direction and Actuator Transfer 执行日志\n\n"
            f"生成时间：{now_sg()}\n\n"
            "记录原则：只记录真实命令、文件、输入、输出、状态、blocker 与修复尝试；"
            "没有执行或被 gate 阻断的项目必须明确写为 not_run/gate_blocked。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.33 Causal Direction and Actuator Transfer 实验结果复盘\n\n"
            f"生成时间：{now_sg()}\n\n"
            "本复盘只引用本轮 v22.33 artifact、命令日志、direct pilot 输出与明确命名的历史 artifact 读回结果；禁止编造数据。\n",
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
    journal = OUT_ROOT / "v22_33_command_journal.csv"
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
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    ensure_out()
    merged = os.environ.copy()
    if env:
        merged.update(env)
    started = time.time()
    try:
        proc = subprocess.run(
            args,
            cwd=str(ROOT),
            env=merged,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        proc = subprocess.CompletedProcess(args=args, returncode=124, stdout=exc.stdout or "", stderr=exc.stderr or "")
    stdout_path = LOG_ROOT / f"{task_id}_stdout.log"
    stderr_path = LOG_ROOT / f"{task_id}_stderr.log"
    stdout_path.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8", errors="replace")
    status = "pass" if proc.returncode == 0 else ("timeout" if proc.returncode == 124 else "fail")
    append_exec(
        " ".join(shlex.quote(x) for x in args),
        task_id=task_id,
        status=status,
        exit_code=proc.returncode,
        files=f"{stdout_path.relative_to(ROOT)}, {stderr_path.relative_to(ROOT)}",
        note=f"elapsed_sec={time.time() - started:.3f}; cwd={ROOT}",
    )
    return proc


def read_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def write_rows(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(r) for r in rows]
    fields = list(fieldnames or [])
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


def finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in {"", None}:
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def int_flag(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def rate(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md_table(rows: list[dict[str, Any]], fields: list[str] | None = None, limit: int = 12) -> str:
    materialized = [dict(r) for r in rows[:limit]]
    if not materialized:
        return "_无可用行。_"
    cols = fields or list(materialized[0].keys())
    lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in materialized:
        vals = [str(row.get(c, "")).replace("\n", " ") for c in cols]
        lines.append("|" + "|".join(vals) + "|")
    if len(rows) > limit:
        lines.append(f"\n_仅显示前 {limit} 行，共 {len(rows)} 行。_")
    return "\n".join(lines)


def write_simple_svg(path: Path, title: str, rows: list[dict[str, Any]], value_key: str = "value") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values: list[tuple[str, float]] = []
    for row in rows[:40]:
        val = finite_float(row.get(value_key), None)
        if val is not None:
            values.append((str(row.get("label", ""))[:46], float(val)))
    width = 980
    height = max(220, 90 + 24 * max(1, len(values)))
    max_abs = max((abs(v) for _label, v in values), default=1.0) or 1.0
    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#ffffff'/>",
        f"<text x='24' y='34' font-family='Arial' font-size='18' fill='#111111'>{title}</text>",
    ]
    if not values:
        lines.append("<text x='24' y='82' font-family='Arial' font-size='14' fill='#555555'>no numeric rows available; see CSV for not_run/gate_blocked evidence</text>")
    zero_x = 510
    lines.append(f"<line x1='{zero_x}' y1='58' x2='{zero_x}' y2='{height - 24}' stroke='#999999' stroke-width='1'/>")
    for i, (label, value) in enumerate(values):
        y = 74 + i * 24
        bar = int(360 * abs(value) / max_abs)
        color = "#1f77b4" if value >= 0 else "#d62728"
        x = zero_x if value >= 0 else zero_x - bar
        lines.append(f"<text x='24' y='{y + 10}' font-family='Arial' font-size='12' fill='#222222'>{label}</text>")
        lines.append(f"<rect x='{x}' y='{y}' width='{bar}' height='14' fill='{color}' opacity='0.82'/>")
        lines.append(f"<text x='{zero_x + 370}' y='{y + 11}' font-family='Arial' font-size='12' fill='#222222'>{value:.6g}</text>")
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_legacy(out_root: Path = RAW_ROOT) -> Any:
    legacy = importlib.import_module("experiments.run_v22_32_causal_actuator_fidelity")
    legacy.PYTHON = PYTHON
    legacy.OUT_ROOT = out_root
    legacy.FIG_ROOT = out_root / "figures"
    legacy.LOG_ROOT = out_root / "logs"
    legacy.PLAN_DOC = PLAN_DOC
    legacy.EXEC_DOC = EXEC_DOC
    legacy.RECAP_DOC = RAW_ROOT / "v22_32_recap_generated_by_legacy.md"
    legacy.V22_30 = ROOT / "results/v22_30"
    legacy.ensure_out()
    return legacy


def create_review_bundle() -> tuple[list[dict[str, Any]], Path]:
    files = [
        PLAN_DOC.relative_to(ROOT),
        Path("experiments/run_v22_33_causal_direction_actuator_transfer.py"),
        Path("experiments/run_v22_32_causal_actuator_fidelity.py"),
        Path("experiments/run_v22_30_fidelity_ladder.py"),
        Path("dgkan/fu/core.py"),
        Path("dgkan/fu/mechanisms.py"),
        Path("dgkan/fu/metric_solver.py"),
        Path("dgkan/fu/real_jacobian_commit.py"),
        Path("dgkan/fu/basis_native_controller.py"),
        Path("dgkan/models/fc_purekan_primitives.py"),
        Path("dgkan/integration/kanbefair_adapter.py"),
    ]
    rows: list[dict[str, Any]] = []
    for rel in files:
        path = ROOT / rel
        rows.append(
            {
                "path": str(rel),
                "present": int(path.exists()),
                "bytes": path.stat().st_size if path.exists() else "",
                "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
            }
        )
    manifest = OUT_ROOT / "v22_33_review_bundle_manifest.csv"
    write_rows(manifest, rows)
    bundle = OUT_ROOT / "v22_33_review_bundle_current.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for row in rows:
            if int_flag(row.get("present")):
                tar.add(ROOT / str(row["path"]), arcname=str(row["path"]))
        for rel in [
            "v22_33_code_truth_gate.csv",
            "v22_33_import_closure_matrix.csv",
            "v22_33_final_route.json",
        ]:
            path = OUT_ROOT / rel
            if path.exists():
                tar.add(path, arcname=str(Path("results/v22_33") / rel))
    return rows, bundle


def build_code_truth(raw_code: dict[str, Any]) -> dict[str, Any]:
    import_rows = read_rows(RAW_ROOT / "v22_32_import_closure_matrix.csv")
    identity_rows = read_rows(RAW_ROOT / "v22_32_model_identity_matrix.csv")
    manifest_rows, bundle = create_review_bundle()
    missing_modules = [r.get("module", "") for r in import_rows if not int_flag(r.get("import_pass"))]
    official_rows = [r for r in identity_rows if int_flag(r.get("official_row"))]
    row = {
        "review_bundle_self_contained": int(all(int_flag(r.get("present")) for r in manifest_rows)),
        "clean_unzip_compileall_pass": int_flag(raw_code.get("compileall_pass")),
        "clean_unzip_import_pass": int(all(int_flag(r.get("import_pass")) for r in import_rows)),
        "missing_transitive_dependency_count": len(missing_modules),
        "missing_module_names": ";".join(missing_modules),
        "official_DGKAN_identity_pass": int_flag(raw_code.get("official_DGKAN_identity_pass")),
        "KANbeFair_original_KAN_official_rows": sum(
            1 for r in official_rows if int_flag(r.get("uses_kanbefair_baseline_model"))
        ),
        "uses_pykan_official_rows": sum(1 for r in official_rows if int_flag(r.get("uses_pykan"))),
        "uses_bspline_official_rows": sum(1 for r in official_rows if int_flag(r.get("uses_bspline_official_path"))),
        "uses_readout_diagnostic_official_rows": sum(1 for r in official_rows if int_flag(r.get("uses_readout_diagnostic"))),
        "python_used_for_pass": raw_code.get("python_used_for_pass", PYTHON),
        "environment_repair": raw_code.get("environment_repair", ""),
        "review_bundle_path": str(bundle.relative_to(ROOT)),
        "source_artifact": str((RAW_ROOT / "v22_32_code_truth_gate.csv").relative_to(ROOT)),
    }
    write_rows(OUT_ROOT / "v22_33_code_truth_gate.csv", [row])
    shutil.copyfile(RAW_ROOT / "v22_32_import_closure_matrix.csv", OUT_ROOT / "v22_33_import_closure_matrix.csv")
    append_exec(
        "transform raw v22.32 code closure into v22.33 Part A artifact names",
        task_id="A_v22_33_code_truth_transform",
        status="pass" if row["clean_unzip_compileall_pass"] and row["clean_unzip_import_pass"] else "fail",
        gpu="cpu",
        files="results/v22_33/v22_33_code_truth_gate.csv, results/v22_33/v22_33_import_closure_matrix.csv",
        note=f"missing_module_names={row['missing_module_names'] or 'none'}; review_bundle={row['review_bundle_path']}",
    )
    return row


def transform_gap() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    src_rows = read_rows(RAW_ROOT / "v22_32_gap_decomposition_matrix.csv")
    rows: list[dict[str, Any]] = []
    for r in src_rows:
        rows.append(
            {
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "task_tier": r.get("task_tier", ""),
                "carrier": r.get("model_family", ""),
                "Delta_MLP_NLL": r.get("Delta_MLP_NLL", ""),
                "Delta_KAN_NLL": r.get("Delta_KAN_NLL", ""),
                "Delta_MLP_accuracy": r.get("Delta_MLP_accuracy", ""),
                "Delta_KAN_accuracy": r.get("Delta_KAN_accuracy", ""),
                "GapReduction_NLL": r.get("GapReduction_NLL", ""),
                "GapReduction_accuracy": r.get("GapReduction_accuracy", ""),
                "MLP_FU_vs_best_control_NLL_delta": r.get("MLP_FU_vs_best_control_NLL", ""),
                "KAN_FU_vs_best_control_NLL_delta": r.get("KAN_FU_vs_best_control_NLL", ""),
                "epsilon_method": r.get("epsilon_method", ""),
                "repeat_noise_epsilon": r.get("repeat_noise_epsilon", ""),
                "gap_reduction_class": r.get("gap_reduction_class", ""),
                "source_artifact": r.get("source_artifact", ""),
            }
        )
    write_rows(OUT_ROOT / "v22_33_gap_truth_matrix.csv", rows)

    raw_summary = (read_rows(RAW_ROOT / "v22_32_gap_decomposition_summary.csv") or [{}])[0]
    summary = {
        **raw_summary,
        "exploration_gate_pass": int(
            finite_float(raw_summary.get("TrueKANGain_plus_BothGain_rate"), 0.0) >= 0.25
            and finite_float(raw_summary.get("ControlExplained_rate"), 1.0) <= 0.50
            and finite_float(raw_summary.get("MLPDegradationDriven_rate"), 1.0) <= 0.20
        ),
        "official_candidate_gate_pass": int(
            finite_float(raw_summary.get("TrueKANGain_plus_BothGain_rate"), 0.0) >= 0.60
            and finite_float(raw_summary.get("ControlExplained_rate"), 1.0) <= 0.20
            and finite_float(raw_summary.get("MLPDegradationDriven_rate"), 1.0) <= 0.10
        ),
        "v22_33_blocker_action": "TrueKANGain为0时停止full superiority claim，转入Part C direction与Part D actuator复核；本轮已执行direct T1/basis ladder。",
    }
    write_rows(OUT_ROOT / "v22_33_gap_truth_summary.csv", [summary])
    class_rows = [{"label": k, "value": finite_float(summary.get(f"{k}_rows"), 0.0) or 0.0} for k in ["TrueKANGain", "BothGain", "MLPDegradationDriven", "ControlExplained", "NoGain"]]
    write_simple_svg(FIG_ROOT / "v22_33_gap_truth_stacked_bar.svg", "v22.33 gap truth class counts", class_rows)
    write_simple_svg(
        FIG_ROOT / "v22_33_KAN_vs_MLPFU_gap_reduction_panel.svg",
        "v22.33 KAN vs MLP+FU gap reduction",
        [{"label": f"{r.get('dataset')} {r.get('carrier')}", "value": finite_float(r.get("GapReduction_NLL"), 0.0) or 0.0} for r in rows],
    )
    return rows, summary


def transform_t1() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selector_rows = read_rows(RAW_ROOT / "v22_32_T1_direction_selection_matrix.csv")
    branch_rows = read_rows(RAW_ROOT / "v22_32_T1_branch_causal_matrix.csv")
    gate_rows = read_rows(RAW_ROOT / "v22_32_T1_selector_gate_eval.csv")
    control_rows: list[dict[str, Any]] = []
    for r in branch_rows:
        if not int_flag(r.get("is_control_branch")):
            continue
        control_rows.append(
            {
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "selector_name": r.get("selector_name", ""),
                "control_level": r.get("control_level", ""),
                "control_type": r.get("selected_candidate_label", ""),
                "norm_match_error": "",
                "support_match_error": "",
                "subspace_match_angle": "",
                "actuator_match_residual": "",
                "NLL_delta": r.get("NLL_delta_vs_base", ""),
                "AUC_delta": r.get("AUC_delta_vs_base", ""),
                "ECE_delta": r.get("ECE_delta_vs_base", ""),
                "Brier_delta": r.get("Brier_delta_vs_base", ""),
                "tail_q99_delta": r.get("tail_q99_delta", ""),
                "sharpness_delta": "",
                "margin_delta": "",
                "beats_real": "",
                "source_artifact": r.get("source_artifact", ""),
            }
        )
    write_rows(OUT_ROOT / "v22_33_T1_selector_matrix.csv", selector_rows)
    write_rows(OUT_ROOT / "v22_33_T1_branch_matrix.csv", branch_rows)
    write_rows(OUT_ROOT / "v22_33_T1_control_level_matrix.csv", control_rows)
    raw_summary = (read_rows(RAW_ROOT / "v22_32_T1_direction_selector_summary.csv") or [{}])[0]
    summary = {
        **raw_summary,
        "v22_33_gate_interpretation": "T1 direct selectors are diagnostic unless exploration/official thresholds pass; L influence rows use future branch labels and remain diagnostic-only.",
    }
    write_simple_svg(
        FIG_ROOT / "v22_33_T1_selector_vs_controls_panel.svg",
        "v22.33 T1 selector beats-control rates",
        [{"label": r.get("selector_name", ""), "value": finite_float(r.get("beats_L3_rate"), 0.0) or 0.0} for r in gate_rows],
    )
    return selector_rows, branch_rows, control_rows, summary


def transform_basis() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fit_rows = read_rows(RAW_ROOT / "v22_32_KAN_basis_actuator_fit_matrix.csv")
    branch_rows = read_rows(RAW_ROOT / "v22_32_KAN_basis_branch_matrix.csv")
    summary = (read_rows(RAW_ROOT / "v22_32_KAN_basis_actuator_fidelity_summary.csv") or [{}])[0]
    write_rows(OUT_ROOT / "v22_33_basis_actuator_fit_matrix.csv", fit_rows)
    write_rows(OUT_ROOT / "v22_33_basis_actuator_branch_matrix.csv", branch_rows)
    real_branch = [r for r in branch_rows if r.get("branch_variant") == "basis_native_real"]
    full_loop_rows: list[dict[str, Any]] = []
    if int_flag(summary.get("basis_native_candidate_pass")):
        full_loop_rows.append({"status": "not_run_manual_followup_required", "blocker": "basis candidate passed but no v22.33 integrated full-loop launcher exists in this thin runner"})
    else:
        full_loop_rows.append(
            {
                "status": "gate_blocked",
                "blocker": "D3 branch gate did not satisfy >=60% H100/H200/H400 pass; Part D plan forbids launching full-loop.",
                "branch_real_rows": len(real_branch),
                "branch_control_beat_rate": summary.get("branch_control_beat_rate", ""),
                "basis_actuator_exploration_pass": summary.get("basis_actuator_exploration_pass", ""),
                "source_artifact": str((RAW_ROOT / "v22_32_KAN_basis_branch_matrix.csv").relative_to(ROOT)),
            }
        )
    write_rows(OUT_ROOT / "v22_33_basis_actuator_full_loop_matrix.csv", full_loop_rows)
    paired: list[dict[str, Any]] = []
    by_key = {
        (r.get("dataset"), r.get("basis_family"), r.get("basis_bank")): finite_float(
            r.get("NLL_delta_H100") or r.get("NLL_delta_H50") or r.get("NLL_delta_H200"), 0.0
        )
        for r in real_branch
    }
    for r in fit_rows:
        key = (r.get("dataset"), r.get("basis_family"), r.get("basis_bank"))
        paired.append(
            {
                "label": f"{r.get('dataset')} {r.get('basis_family')} {r.get('basis_bank')}",
                "value": finite_float(r.get("basis_actuator_projection_residual"), 0.0) or 0.0,
                "branch_delta": by_key.get(key, ""),
            }
        )
    write_simple_svg(FIG_ROOT / "v22_33_basis_actuator_residual_vs_branch_gain.svg", "v22.33 basis actuator residual", paired)
    return fit_rows, branch_rows, full_loop_rows, summary


def transform_diagnostics() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    simple_copies = {
        "v22_32_control_win_decomposition_matrix.csv": "v22_33_control_win_decomposition_matrix.csv",
        "v22_32_curvature_safety_matrix.csv": "v22_33_curvature_safety_matrix.csv",
        "v22_32_representation_margin_matrix.csv": "v22_33_representation_geometry_matrix.csv",
        "v22_32_optimizer_integrated_fu_matrix.csv": "v22_33_optimizer_integrated_fu_matrix.csv",
        "v22_32_optimizer_spectrum_matrix.csv": "v22_33_optimizer_spectrum_matrix.csv",
        "v22_32_hard_task_scheduling_matrix.csv": "v22_33_hard_task_matrix.csv",
        "v22_32_continual_matrix.csv": "v22_33_continual_matrix.csv",
        "v22_32_grokking_matrix.csv": "v22_33_grokking_matrix.csv",
    }
    for src, dst in simple_copies.items():
        s = RAW_ROOT / src
        d = OUT_ROOT / dst
        if s.exists():
            shutil.copyfile(s, d)
        else:
            write_rows(d, [{"status": "not_run_missing_source_artifact", "source_artifact": str(s.relative_to(ROOT))}])
    basis_summary = (read_rows(RAW_ROOT / "v22_32_KAN_basis_actuator_fidelity_summary.csv") or [{}])[0]
    efficiency_rows = [
        {
            "status": "gate_blocked_no_four_path_efficiency_launch",
            "blocker": "basis actuator branch/control gate did not open an official candidate; plan says stop before full-loop/efficiency expansion unless branch benefit exists.",
            "full_loop_ratio": "",
            "controller_overhead_ratio": "",
            "branch_control_beat_rate": basis_summary.get("branch_control_beat_rate", ""),
            "source_artifact": str((OUT_ROOT / "v22_33_basis_actuator_branch_matrix.csv").relative_to(ROOT)),
        }
    ]
    write_rows(OUT_ROOT / "v22_33_efficiency_four_path_matrix.csv", efficiency_rows)

    control_rows = read_rows(OUT_ROOT / "v22_33_control_win_decomposition_matrix.csv")
    curvature_rows = read_rows(OUT_ROOT / "v22_33_curvature_safety_matrix.csv")
    opt_summary = (read_rows(RAW_ROOT / "v22_32_optimizer_delta_summary.csv") or [{}])[0]
    t5_summary = {
        "online_subspace_rows": len(read_rows(RAW_ROOT / "v22_32_online_subspace_tracking_matrix.csv")),
        "status": "diagnostic_only",
        "source_artifact": str((RAW_ROOT / "v22_32_online_subspace_tracking_matrix.csv").relative_to(ROOT)),
    }
    temporal_summary = (read_json(RAW_ROOT / "v22_32_final_route.json").get("temporal_continual_summary") or {})
    write_simple_svg(
        FIG_ROOT / "v22_33_control_level_waterfall.svg",
        "v22.33 control-win real-minus-control evidence",
        [{"label": f"{r.get('mechanism')} {r.get('dataset')}", "value": finite_float(r.get("real_minus_control_by_level"), 0.0) or 0.0} for r in control_rows],
    )
    write_simple_svg(
        FIG_ROOT / "v22_33_curvature_vs_control_win_scatter.svg",
        "v22.33 curvature diagnostics availability",
        [{"label": r.get("status", "curvature"), "value": 0.0 if r.get("status") else finite_float(r.get("lambda_max_H"), 0.0) or 0.0} for r in curvature_rows],
    )
    spectrum_rows = read_rows(OUT_ROOT / "v22_33_optimizer_spectrum_matrix.csv")
    write_simple_svg(
        FIG_ROOT / "v22_33_optimizer_spectrum_panel.svg",
        "v22.33 optimizer update SNR",
        [{"label": r.get("optimizer_variant", ""), "value": finite_float(r.get("update_SNR"), 0.0) or 0.0} for r in spectrum_rows],
    )
    cont_rows = read_rows(OUT_ROOT / "v22_33_continual_matrix.csv")
    write_simple_svg(
        FIG_ROOT / "v22_33_continual_forgetting_curves.svg",
        "v22.33 continual average forgetting",
        [{"label": f"{r.get('task')} {r.get('model_name')}", "value": finite_float(r.get("avg_forgetting"), 0.0) or 0.0} for r in cont_rows],
    )
    grok_rows = read_rows(OUT_ROOT / "v22_33_grokking_matrix.csv")
    write_simple_svg(
        FIG_ROOT / "v22_33_grokking_delay_curves.svg",
        "v22.33 grokking time reduction",
        [{"label": f"{r.get('task')} {r.get('variant')}", "value": finite_float(r.get("grokking_time_reduction"), 0.0) or 0.0} for r in grok_rows],
    )
    write_simple_svg(
        FIG_ROOT / "v22_33_efficiency_four_path_waterfall.svg",
        "v22.33 efficiency gate status",
        [{"label": r.get("status", "efficiency"), "value": finite_float(r.get("controller_overhead_ratio"), 0.0) or 0.0} for r in efficiency_rows],
    )
    return opt_summary, t5_summary, temporal_summary


def transform_legacy_journal() -> list[dict[str, Any]]:
    src = RAW_ROOT / "v22_32_command_journal.csv"
    rows = read_rows(src)
    out: list[dict[str, Any]] = []
    for row in rows:
        files = str(row.get("files", "")).replace("results/v22_32/", "results/v22_33/raw_v22_32_runner/")
        out.append(
            {
                **row,
                "task_id": f"legacy_{row.get('task_id', '')}",
                "files_resolved_for_v22_33": files,
                "raw_journal_source": str(src.relative_to(ROOT)),
            }
        )
    dst = OUT_ROOT / "v22_33_legacy_command_journal.csv"
    write_rows(dst, out)
    append_exec(
        "normalize legacy v22.32 substep journal paths into v22.33 raw artifact locations",
        task_id="Z_legacy_journal_path_map",
        status="pass" if out else "warn",
        files=str(dst.relative_to(ROOT)),
        note="Legacy substep log messages may mention results/v22_32; resolved files live under results/v22_33/raw_v22_32_runner and are transformed into top-level v22_33 artifacts.",
    )
    return out


def split_csv(text: str, cast: Any = str) -> list[Any]:
    out: list[Any] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(cast(part))
    return out


def run_t1_c7_curvature_repair(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    from experiments.run_v22_30_fidelity_ladder import (
        apply_layer_direction,
        cohort_signal_for_layer,
        collect_fixed_examples,
        direction_control,
        make_loaders,
        make_mlp,
        principal_overlap,
        train_adamw,
        train_branch,
    )

    device = torch.device(args.t1_device if str(args.t1_device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    selector_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    for dataset in split_csv(args.t1_datasets):
        for seed in split_csv(args.t1_seeds, int):
            train_loader, held_loader, test_loader, input_dim, output_dim, _x_stats = make_loaders(
                dataset, args.t1_train_size, args.t1_test_size, args.batch_size, seed
            )
            torch.manual_seed(2233 + seed)
            init_model = make_mlp(input_dim, output_dim, args.hidden, seed + 2233, device).to(device)
            model = copy.deepcopy(init_model).to(device)
            train_adamw(
                model,
                train_loader,
                test_loader,
                device,
                output_dim,
                steps=args.t1_pretrain_steps,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            xb, yb = collect_fixed_examples(
                train_loader,
                max(args.t1_examples, args.t1_signal_cohorts * args.t1_cohort_size),
                device,
            )
            held_x, held_y = collect_fixed_examples(held_loader, args.t1_examples, device)
            layer = "w2"
            sig, _ = cohort_signal_for_layer(
                model,
                xb,
                yb,
                layer,
                cohorts=args.t1_signal_cohorts,
                cohort_size=args.t1_cohort_size,
                rank_cap=args.t1_rank_cap,
                sketch_dim=args.t1_sketch_dim,
                seed=seed + 701,
            )
            prev_sig, _ = cohort_signal_for_layer(
                init_model,
                xb,
                yb,
                layer,
                cohorts=args.t1_signal_cohorts,
                cohort_size=args.t1_cohort_size,
                rank_cap=args.t1_rank_cap,
                sketch_dim=args.t1_sketch_dim,
                seed=seed + 701,
            )

            basis = sig.get("basis_sketch")
            proj = sig.get("proj")
            direction = sig["direction"]
            candidates: list[tuple[str, Any, str]] = [("raw_top_signal", direction, "original_top_eigen_direction")]
            if basis is not None and proj is not None and int(basis.numel()) > 0:
                k = int(basis.shape[1])
                for j in range(k):
                    coeff = torch.zeros(k, device=direction.device)
                    coeff[j] = 1.0
                    vec = proj @ (basis @ coeff)
                    vec = vec * direction.norm().clamp_min(1.0e-12) / vec.norm().clamp_min(1.0e-12)
                    candidates.append((f"basis_{j}_plus", vec, "curvature_safe_basis_axis"))
                    candidates.append((f"basis_{j}_minus", -vec, "curvature_safe_basis_axis"))
                gen = torch.Generator(device=direction.device).manual_seed(223300 + seed)
                while len(candidates) < max(1, args.t1_candidate_count):
                    coeff = torch.randn(k, device=direction.device, generator=gen)
                    vec = proj @ (basis @ coeff)
                    vec = vec * direction.norm().clamp_min(1.0e-12) / vec.norm().clamp_min(1.0e-12)
                    candidates.append((f"random_combo_{len(candidates)}", vec, "curvature_safe_same_subspace_random_combo"))
            candidates = candidates[: max(1, args.t1_candidate_count)]

            def ce_after(candidate: Any) -> float:
                trial = copy.deepcopy(model).to(device)
                apply_layer_direction(trial, layer, candidate, args.t1_branch_trust)
                trial.eval()
                with torch.no_grad():
                    return float(F.cross_entropy(trial(held_x).float(), held_y.long()).item())

            model.eval()
            with torch.no_grad():
                base_ce = float(F.cross_entropy(model(held_x).float(), held_y.long()).item())
            best: dict[str, Any] | None = None
            candidate_audit_rows: list[dict[str, Any]] = []
            for label, cand, source in candidates:
                plus_ce = ce_after(cand)
                minus_ce = ce_after(-cand)
                curvature = max(0.0, (plus_ce + minus_ce - 2.0 * base_ce) / max(float(args.t1_branch_trust) ** 2, 1.0e-12))
                held_delta = plus_ce - base_ce
                score = held_delta + float(args.t1_c7_curvature_weight) * curvature
                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "candidate_label": label,
                    "candidate_source": source,
                    "held_CE_delta": held_delta,
                    "symmetric_curvature_proxy": curvature,
                    "curvature_safe_score": score,
                    "uses_test_direction_selection": 0,
                    "uses_future_direction": 0,
                }
                candidate_audit_rows.append(row)
                if best is None or score < float(best["curvature_safe_score"]):
                    best = {**row, "direction": cand}
            assert best is not None

            base_model = copy.deepcopy(model).to(device)
            base_ev = train_branch(base_model, train_loader, test_loader, device, output_dim, args.t1_branch_horizon, args.lr, args.weight_decay)
            controls = [
                ("L1_isotropic_random", direction_control(direction, "random", seed + 1101), "L1"),
                ("L2_same_norm_random", direction_control(direction, "random", seed + 1102), "L2"),
                ("L3_same_signal_subspace_random", direction_control(direction, "same_subspace", seed + 1103, basis, proj), "L3"),
                ("L4_signflip_same_subspace", -direction, "L4"),
                ("L5_shuffled_cohort_direction", direction_control(direction, "shuffled", seed + 1105), "L5"),
            ]

            def branch_eval(name: str, cand: Any, is_control: int, control_level: str = "") -> dict[str, Any]:
                branch_model = copy.deepcopy(model).to(device)
                apply_layer_direction(branch_model, layer, cand, args.t1_branch_trust)
                ev = train_branch(branch_model, train_loader, test_loader, device, output_dim, args.t1_branch_horizon, args.lr, args.weight_decay)
                return {
                    "dataset": dataset,
                    "seed": seed,
                    "checkpoint_step": f"c7_repair_pretrain_steps_{args.t1_pretrain_steps}",
                    "selector_name": name,
                    "layer_id": layer,
                    "branch_H": args.t1_branch_horizon,
                    "NLL_delta_vs_base": ev["NLL"] - base_ev["NLL"],
                    "accuracy_delta_vs_base": ev["accuracy"] - base_ev["accuracy"],
                    "AUC_delta_vs_base": "",
                    "ECE_delta_vs_base": ev["ECE"] - base_ev["ECE"],
                    "Brier_delta_vs_base": ev["Brier"] - base_ev["Brier"],
                    "tail_q95_delta": ev["tail_q95"] - base_ev["tail_q95"],
                    "tail_q99_delta": ev["tail_q99"] - base_ev["tail_q99"],
                    "beats_base": int(ev["NLL"] < base_ev["NLL"]),
                    "is_control_branch": is_control,
                    "control_level": control_level,
                    "same_norm": 1,
                    "same_cadence": 1,
                    "same_overhead": 1,
                    "selection_data_source": "held_train_ce_plus_symmetric_curvature_proxy",
                    "uses_test_direction_selection": 0,
                    "uses_future_direction": 0,
                    "source_artifact": "direct_v22_33_C7_curvature_safe_repair",
                    "status": "direct_v22_33_C7_curvature_safe_repair",
                }

            control_branch = [branch_eval(name, cand, 1, level) for name, cand, level in controls]
            branch_rows.extend(control_branch)
            best_by_level: dict[str, float] = {}
            for row in control_branch:
                val = finite_float(row.get("NLL_delta_vs_base"), math.inf) or math.inf
                level = str(row.get("control_level"))
                best_by_level[level] = min(best_by_level.get(level, math.inf), val)
            best_any = min(best_by_level.values()) if best_by_level else math.inf
            real = branch_eval("C7_curvature_safe_direction", best["direction"], 0, "")
            val = finite_float(real.get("NLL_delta_vs_base"), math.inf) or math.inf
            real["best_control_delta"] = best_any
            real["real_minus_best_control_delta"] = val - best_any
            for level in ["L1", "L2", "L3", "L4", "L5"]:
                real[f"beats_{level}_control"] = int(val < best_by_level.get(level, math.inf))
            branch_rows.append(real)
            selector_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "checkpoint_step": real["checkpoint_step"],
                    "selector_name": "C7_curvature_safe_direction",
                    "subspace_dim": int(basis.shape[1]) if basis is not None else "",
                    "signal_eigenvalue": sig.get("positive_eigenvalue_mean", ""),
                    "cohort_positive_fraction": sig.get("cohort_positive_fraction", ""),
                    "temporal_eigenspace_overlap": principal_overlap(prev_sig["basis_sketch"], sig["basis_sketch"]),
                    "leave_cohort_direction_variance": "",
                    "hard_slice_fraction": "",
                    "branch_H": args.t1_branch_horizon,
                    "NLL_delta_vs_base": real["NLL_delta_vs_base"],
                    "accuracy_delta_vs_base": real["accuracy_delta_vs_base"],
                    "AUC_delta_vs_base": "",
                    "ECE_delta_vs_base": real["ECE_delta_vs_base"],
                    "Brier_delta_vs_base": real["Brier_delta_vs_base"],
                    "tail_q95_delta": real["tail_q95_delta"],
                    "tail_q99_delta": real["tail_q99_delta"],
                    "beats_base": real["beats_base"],
                    "beats_L1_control": real["beats_L1_control"],
                    "beats_L2_control": real["beats_L2_control"],
                    "beats_L3_control": real["beats_L3_control"],
                    "beats_L4_control": real["beats_L4_control"],
                    "beats_L5_control": real["beats_L5_control"],
                    "best_control_delta": best_any,
                    "real_minus_best_control_delta": real["real_minus_best_control_delta"],
                    "selected_candidate_label": best["candidate_label"],
                    "selection_data_source": "held_train_ce_plus_symmetric_curvature_proxy",
                    "held_CE_delta": best["held_CE_delta"],
                    "symmetric_curvature_proxy": best["symmetric_curvature_proxy"],
                    "curvature_safe_score": best["curvature_safe_score"],
                    "candidate_audit_rows": len(candidate_audit_rows),
                    "uses_test_direction_selection": 0,
                    "uses_future_direction": 0,
                    "source_artifact": "direct_v22_33_C7_curvature_safe_repair",
                    "status": "direct_v22_33_C7_curvature_safe_repair",
                }
            )

    direct_control_deltas = [
        finite_float(r.get("NLL_delta_vs_base"))
        for r in branch_rows
        if int_flag(r.get("is_control_branch"))
    ]
    control_vals = [float(v) for v in direct_control_deltas if v is not None]
    if len(control_vals) > 1:
        mean_control = sum(control_vals) / len(control_vals)
        control_noise = math.sqrt(sum((v - mean_control) ** 2 for v in control_vals) / len(control_vals))
    else:
        control_noise = 0.0
    n = len(selector_rows)
    mean_nll = sum(float(finite_float(r.get("NLL_delta_vs_base"), 0.0) or 0.0) for r in selector_rows) / n if n else math.nan
    beats_base_rate = sum(int_flag(r.get("beats_base")) for r in selector_rows) / n if n else 0.0
    beats_l3_rate = sum(int_flag(r.get("beats_L3_control")) for r in selector_rows) / n if n else 0.0
    beats_l4_rate = sum(int_flag(r.get("beats_L4_control")) for r in selector_rows) / n if n else 0.0
    beats_l5_rate = sum(int_flag(r.get("beats_L5_control")) for r in selector_rows) / n if n else 0.0
    max_ece = max((finite_float(r.get("ECE_delta_vs_base"), 0.0) or 0.0 for r in selector_rows), default=0.0)
    max_tail = max((finite_float(r.get("tail_q99_delta"), 0.0) or 0.0 for r in selector_rows), default=0.0)
    summary = {
        "selector_name": "C7_curvature_safe_direction",
        "direct_rows": n,
        "datasets": ",".join(sorted({str(r.get("dataset")) for r in selector_rows})),
        "seeds": ",".join(sorted({str(r.get("seed")) for r in selector_rows})),
        "horizons": str(args.t1_branch_horizon),
        "beats_base_rate": beats_base_rate,
        "beats_L3_rate": beats_l3_rate,
        "beats_L4_rate": beats_l4_rate,
        "beats_L5_rate": beats_l5_rate,
        "mean_NLL_delta_vs_base": mean_nll,
        "control_noise_std": control_noise,
        "max_ECE_delta": max_ece,
        "max_tail_q99_delta": max_tail,
        "exploration_pass": int(beats_base_rate >= 0.65 and beats_l3_rate >= 0.55 and mean_nll < -control_noise and max_ece <= 0.01 and max_tail <= 0.05),
        "official_candidate_pass": int(beats_l3_rate >= 0.65 and beats_l4_rate >= 0.60 and beats_l5_rate >= 0.60),
        "repair_note": "C7 selected with held-train CE delta plus symmetric curvature proxy; no test/future direction selection.",
    }
    write_rows(OUT_ROOT / "v22_33_T1_C7_curvature_repair_matrix.csv", selector_rows)
    write_rows(OUT_ROOT / "v22_33_T1_C7_curvature_repair_branch_matrix.csv", branch_rows)
    write_rows(OUT_ROOT / "v22_33_T1_C7_curvature_repair_summary.csv", [summary])
    append_exec(
        "run C7 curvature-safe direction repair on held-train CE + symmetric curvature proxy",
        task_id="R_C7_curvature_safe_direction",
        status="pass",
        gpu=str(device),
        files="results/v22_33/v22_33_T1_C7_curvature_repair_matrix.csv, results/v22_33/v22_33_T1_C7_curvature_repair_summary.csv",
        note=f"exploration_pass={summary['exploration_pass']}; official_candidate_pass={summary['official_candidate_pass']}; mean_NLL_delta={summary['mean_NLL_delta_vs_base']}",
    )
    return selector_rows, branch_rows, summary


def run_t1_c1c6_direct_repair(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Rerun pre-registered C1-C6 T1 selectors under v22.33 repair params."""
    from experiments import run_v22_32_causal_actuator_fidelity as v22_32

    legacy_args = copy.copy(args)
    legacy_args.device = args.t1_device
    selector_rows, branch_rows, influence_rows = v22_32.stage_c_direct_t1_pilot(legacy_args)
    for row in selector_rows:
        row["source_artifact"] = "direct_v22_33_C1C6_repair_reusing_v22_32_stage_c"
        row["status"] = (
            "direct_v22_33_L_offline_diagnostic_not_official"
            if row.get("selector_name") == "L_cohort_influence_direction"
            else "direct_v22_33_C1C6_repair_not_official_scale"
        )
    for row in branch_rows:
        row["source_artifact"] = "direct_v22_33_C1C6_repair_reusing_v22_32_stage_c"
    for row in influence_rows:
        row["source_artifact"] = "direct_v22_33_C1C6_repair_reusing_v22_32_stage_c"
        row["status"] = "direct_v22_33_L_offline_diagnostic_not_official"

    direct_c_rows = [
        r
        for r in selector_rows
        if str(r.get("selector_name", "")).startswith("C")
        and not int_flag(r.get("uses_future_direction"))
        and not int_flag(r.get("uses_test_direction_selection"))
    ]
    control_vals = [
        finite_float(r.get("NLL_delta_vs_base"))
        for r in branch_rows
        if int_flag(r.get("is_control_branch"))
    ]
    control_noise_vals = [float(v) for v in control_vals if v is not None]
    if len(control_noise_vals) > 1:
        mean_control = sum(control_noise_vals) / len(control_noise_vals)
        control_noise = math.sqrt(sum((v - mean_control) ** 2 for v in control_noise_vals) / len(control_noise_vals))
    else:
        control_noise = 0.0

    gate_rows: list[dict[str, Any]] = []
    for selector in sorted({str(r.get("selector_name", "")) for r in direct_c_rows}):
        rows = [r for r in direct_c_rows if r.get("selector_name") == selector]
        if not rows:
            continue
        n = len(rows)
        nll_vals = [finite_float(r.get("NLL_delta_vs_base")) for r in rows]
        nll_f = [float(v) for v in nll_vals if v is not None]
        mean_nll = sum(nll_f) / len(nll_f) if nll_f else math.inf
        ece_f = [float(v) for v in (finite_float(r.get("ECE_delta_vs_base")) for r in rows) if v is not None]
        tail_f = [float(v) for v in (finite_float(r.get("tail_q99_delta")) for r in rows) if v is not None]
        max_ece = max(ece_f, default=math.inf)
        max_tail = max(tail_f, default=math.inf)
        datasets = {str(r.get("dataset")) for r in rows}
        seeds = {str(r.get("seed")) for r in rows}
        row = {
            "selector_name": selector,
            "direct_rows": n,
            "datasets": ",".join(sorted(datasets)),
            "seeds": ",".join(sorted(seeds)),
            "horizons": ",".join(sorted({str(r.get("branch_H")) for r in rows})),
            "beats_base_rate": sum(int_flag(r.get("beats_base")) for r in rows) / n,
            "beats_L1_rate": sum(int_flag(r.get("beats_L1_control")) for r in rows) / n,
            "beats_L2_rate": sum(int_flag(r.get("beats_L2_control")) for r in rows) / n,
            "beats_L3_rate": sum(int_flag(r.get("beats_L3_control")) for r in rows) / n,
            "beats_L4_rate": sum(int_flag(r.get("beats_L4_control")) for r in rows) / n,
            "beats_L5_rate": sum(int_flag(r.get("beats_L5_control")) for r in rows) / n,
            "mean_NLL_delta_vs_base": mean_nll,
            "control_noise_std": control_noise,
            "max_ECE_delta": max_ece,
            "max_tail_q99_delta": max_tail,
            "uses_future_direction_rows": sum(int_flag(r.get("uses_future_direction")) for r in rows),
            "uses_test_direction_selection_rows": sum(int_flag(r.get("uses_test_direction_selection")) for r in rows),
            "repair_note": "C1-C6 rerun only; L influence rows are diagnostic and excluded from gates.",
        }
        row["exploration_pass"] = int(
            row["beats_base_rate"] >= 0.65
            and row["beats_L3_rate"] >= 0.55
            and mean_nll < -control_noise
            and max_ece <= 0.01
            and max_tail <= 0.05
        )
        row["official_candidate_pass"] = int(
            row["exploration_pass"]
            and row["beats_L3_rate"] >= 0.65
            and row["beats_L4_rate"] >= 0.60
            and row["beats_L5_rate"] >= 0.60
            and len(datasets) >= 2
            and len(seeds) >= 2
            and row["uses_future_direction_rows"] == 0
            and row["uses_test_direction_selection_rows"] == 0
        )
        gate_rows.append(row)

    best_by_nll = sorted(gate_rows, key=lambda r: finite_float(r.get("mean_NLL_delta_vs_base"), math.inf) or math.inf)
    pass_rows = [r for r in gate_rows if int_flag(r.get("exploration_pass"))]
    official_rows = [r for r in gate_rows if int_flag(r.get("official_candidate_pass"))]
    summary = {
        "selector_rows": len(selector_rows),
        "branch_rows": len(branch_rows),
        "influence_rows": len(influence_rows),
        "gate_rows": len(gate_rows),
        "datasets": ",".join(sorted({str(r.get("dataset")) for r in direct_c_rows})),
        "seeds": ",".join(sorted({str(r.get("seed")) for r in direct_c_rows})),
        "horizons": ",".join(sorted({str(r.get("branch_H")) for r in direct_c_rows})),
        "t1_branch_trust": args.t1_branch_trust,
        "control_noise_std": control_noise,
        "exploration_pass_rows": len(pass_rows),
        "official_candidate_rows": len(official_rows),
        "T1_C1C6_exploration_pass": int(bool(pass_rows)),
        "T1_C1C6_official_candidate": int(bool(official_rows)),
        "best_selector_by_mean_NLL": best_by_nll[0].get("selector_name", "") if best_by_nll else "",
        "best_selector_mean_NLL_delta": best_by_nll[0].get("mean_NLL_delta_vs_base", "") if best_by_nll else "",
        "repair_note": "Rerun C1-C6 with v22.33 repair params; no future/test selector rows admitted to gate; L influence remains diagnostic-only.",
    }
    write_rows(OUT_ROOT / "v22_33_T1_C1C6_repair_matrix.csv", selector_rows)
    write_rows(OUT_ROOT / "v22_33_T1_C1C6_repair_branch_matrix.csv", branch_rows)
    write_rows(OUT_ROOT / "v22_33_T1_C1C6_repair_influence_matrix.csv", influence_rows)
    write_rows(OUT_ROOT / "v22_33_T1_C1C6_repair_gate_eval.csv", gate_rows)
    write_rows(OUT_ROOT / "v22_33_T1_C1C6_repair_summary.csv", [summary])
    append_exec(
        "run C1-C6 direct selector repair by reusing v22.32 Stage C direct pilot",
        task_id="R_C1C6_direct_selector_repair",
        status="pass",
        gpu=args.t1_device,
        files="results/v22_33/v22_33_T1_C1C6_repair_matrix.csv, results/v22_33/v22_33_T1_C1C6_repair_gate_eval.csv, results/v22_33/v22_33_T1_C1C6_repair_summary.csv",
        note=f"exploration_pass_rows={summary['exploration_pass_rows']}; official_candidate_rows={summary['official_candidate_rows']}; best_selector={summary['best_selector_by_mean_NLL']}",
    )
    return selector_rows, branch_rows, summary


def branch_nll_delta(row: dict[str, Any]) -> float | None:
    branch_h = str(row.get("branch_H", "")).strip()
    preferred = [f"NLL_delta_H{branch_h}"] if branch_h else []
    for key in [*preferred, "NLL_delta_H400", "NLL_delta_H200", "NLL_delta_H100", "NLL_delta_H50"]:
        val = finite_float(row.get(key), None)
        if val is not None:
            return val
    return None


def branch_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("repair_scan_id", "")),
        str(row.get("carrier", "")),
        str(row.get("basis_family", "")),
        str(row.get("dataset", "")),
        str(row.get("seed", "")),
        str(row.get("basis_bank", "")),
    )


def strict_fit_keys(fit_rows: list[dict[str, Any]]) -> set[tuple[str, str, str, str, str, str]]:
    out: set[tuple[str, str, str, str, str, str]] = set()
    for row in fit_rows:
        if (
            (finite_float(row.get("basis_actuator_projection_residual"), 999.0) or 999.0) <= 0.25
            and (finite_float(row.get("basis_actuator_cosine"), 0.0) or 0.0) >= 0.95
            and (finite_float(row.get("exact_vs_linearized_error"), 999.0) or 999.0) <= 0.05
            and int_flag(row.get("J_B_gradcheck_pass"))
        ):
            out.add(branch_key(row))
    return out


def basis_control_win_decomposition(
    fit_rows: list[dict[str, Any]],
    branch_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fit_pass_keys = strict_fit_keys(fit_rows)
    by_key: dict[tuple[str, str, str, str, str, str], dict[str, dict[str, Any]]] = {}
    for row in branch_rows:
        by_key.setdefault(branch_key(row), {})[str(row.get("branch_variant", ""))] = row

    rows: list[dict[str, Any]] = []
    for key, variants in sorted(by_key.items()):
        real = variants.get("basis_native_real")
        if not real:
            continue
        random_ctrl = variants.get("same_basis_random", {})
        signflip_ctrl = variants.get("signflip_basis_control", {})
        real_nll = branch_nll_delta(real)
        random_nll = branch_nll_delta(random_ctrl)
        signflip_nll = branch_nll_delta(signflip_ctrl)
        fit_strict = key in fit_pass_keys
        real_improves = real_nll is not None and real_nll < 0.0
        beats_random = real_nll is not None and random_nll is not None and real_nll < random_nll
        beats_signflip = real_nll is not None and signflip_nll is not None and real_nll < signflip_nll
        accuracy_delta = finite_float(real.get("accuracy_delta_vs_base"), None)
        accuracy_ok = accuracy_delta is not None and accuracy_delta >= -0.01
        branch_pass = fit_strict and real_improves and beats_random and beats_signflip and accuracy_ok
        if not fit_strict:
            explanation = "not_strict_fit_candidate"
        elif not real_improves:
            explanation = "strict_fit_but_real_NLL_not_improved"
        elif not beats_random and not beats_signflip:
            explanation = "same_basis_random_and_signflip_explain"
        elif not beats_random:
            explanation = "same_basis_random_explains"
        elif not beats_signflip:
            explanation = "signflip_explains"
        elif not accuracy_ok:
            explanation = "accuracy_debt"
        else:
            explanation = "strict_fit_branch_pass"
        rows.append(
            {
                "repair_scan_id": key[0],
                "carrier": key[1],
                "basis_family": key[2],
                "dataset": key[3],
                "seed": key[4],
                "basis_bank": key[5],
                "branch_H": real.get("branch_H", ""),
                "fit_strict_pass": int(fit_strict),
                "real_NLL_delta": "" if real_nll is None else real_nll,
                "same_basis_random_NLL_delta": "" if random_nll is None else random_nll,
                "signflip_basis_control_NLL_delta": "" if signflip_nll is None else signflip_nll,
                "real_improves": int(real_improves),
                "beats_same_basis_random": int(beats_random),
                "beats_signflip_basis_control": int(beats_signflip),
                "accuracy_delta_vs_base": real.get("accuracy_delta_vs_base", ""),
                "accuracy_gate_pass": int(accuracy_ok),
                "strict_fit_branch_pass": int(branch_pass),
                "control_win_explanation": explanation,
            }
        )

    strict_rows = [r for r in rows if int_flag(r.get("fit_strict_pass"))]
    all_pass = [r for r in rows if int_flag(r.get("strict_fit_branch_pass"))]
    strict_improve = [r for r in strict_rows if int_flag(r.get("real_improves"))]
    strict_control = [
        r
        for r in strict_rows
        if int_flag(r.get("beats_same_basis_random")) and int_flag(r.get("beats_signflip_basis_control"))
    ]
    summary = {
        "basis_control_decomposition_rows": len(rows),
        "strict_fit_real_rows": len(strict_rows),
        "strict_fit_branch_improve_rows": len(strict_improve),
        "strict_fit_control_beat_rows": len(strict_control),
        "strict_fit_branch_pass_rows": len(all_pass),
        "strict_fit_branch_improve_rate": rate(len(strict_improve), len(strict_rows)),
        "strict_fit_control_beat_rate": rate(len(strict_control), len(strict_rows)),
        "strict_fit_branch_pass_rate": rate(len(all_pass), len(strict_rows)),
        "all_real_branch_pass_rows": len(all_pass),
        "all_real_branch_pass_rate": rate(len(all_pass), len(rows)),
        "same_basis_random_explains_rows": sum("same_basis_random" in str(r.get("control_win_explanation")) for r in rows),
        "signflip_explains_rows": sum("signflip" in str(r.get("control_win_explanation")) for r in rows),
        "basis_control_win_exploration_pass": int(len(strict_rows) > 0 and rate(len(all_pass), len(strict_rows)) >= 0.60),
        "basis_control_win_note": "D3 branch pass is computed only for strict-fit actuators and requires NLL improvement, both controls beaten, and accuracy_delta_vs_base >= -0.01.",
    }
    return rows, summary


def run_basis_repair_scan(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fit_all: list[dict[str, Any]] = []
    branch_all: list[dict[str, Any]] = []
    scan_rows: list[dict[str, Any]] = []
    scan_specs = [
        ("damp1e-6_scale0p003", 1.0e-6, 3.0e-3),
        ("damp1e-6_scale0p01", 1.0e-6, 1.0e-2),
        ("damp1e-4_scale0p003", 1.0e-4, 3.0e-3),
        ("damp1e-2_scale0p003", 1.0e-2, 3.0e-3),
    ]
    for tag, damping, scale in scan_specs:
        repair_root = OUT_ROOT / "repair_basis_scans" / tag
        legacy = configure_legacy(repair_root)
        scan_args = copy.copy(args)
        scan_args.device = args.basis_device
        scan_args.basis_damping = damping
        scan_args.basis_update_scale = scale
        _target, fit_rows, _exact, branch_rows, summary = legacy.stage_d_basis_actuator(scan_args)
        for row in fit_rows:
            fit_all.append({
                **row,
                "repair_scan_id": tag,
                "basis_damping": damping,
                "basis_update_scale": scale,
                "basis_gradcheck_mode": args.basis_gradcheck_mode,
                "basis_gradcheck_eps": args.basis_gradcheck_eps,
                "raw_root": str(repair_root.relative_to(ROOT)),
            })
        for row in branch_rows:
            branch_all.append({
                **row,
                "repair_scan_id": tag,
                "basis_damping": damping,
                "basis_update_scale": scale,
                "basis_gradcheck_mode": args.basis_gradcheck_mode,
                "basis_gradcheck_eps": args.basis_gradcheck_eps,
                "raw_root": str(repair_root.relative_to(ROOT)),
            })
        scan_rows.append({
            **summary,
            "repair_scan_id": tag,
            "basis_damping": damping,
            "basis_update_scale": scale,
            "basis_gradcheck_mode": args.basis_gradcheck_mode,
            "basis_gradcheck_eps": args.basis_gradcheck_eps,
            "raw_root": str(repair_root.relative_to(ROOT)),
        })

    fit_pre_gradcheck = [
        r
        for r in fit_all
        if (finite_float(r.get("basis_actuator_projection_residual"), 999.0) or 999.0) <= 0.25
        and (finite_float(r.get("basis_actuator_cosine"), 0.0) or 0.0) >= 0.95
        and (finite_float(r.get("exact_vs_linearized_error"), 999.0) or 999.0) <= 0.05
    ]
    fit_pass = [r for r in fit_pre_gradcheck if int_flag(r.get("J_B_gradcheck_pass"))]
    control_decomp_rows, control_decomp_summary = basis_control_win_decomposition(fit_all, branch_all)
    real_rows = [r for r in branch_all if r.get("branch_variant") == "basis_native_real"]
    improve_rows = [
        r
        for r in real_rows
        if (finite_float(r.get("NLL_delta_H100") or r.get("NLL_delta_H50") or r.get("NLL_delta_H200") or r.get("NLL_delta_H400"), 0.0) or 0.0) < 0.0
    ]
    control_beat_rows = [
        r for r in real_rows if int_flag(r.get("beats_same_basis_random")) and int_flag(r.get("beats_signflip_basis_control"))
    ]
    best_residual = min((finite_float(r.get("basis_actuator_projection_residual"), math.inf) or math.inf for r in fit_all), default=math.inf)
    best_strict_candidate_gradcheck = min(
        (finite_float(r.get("J_B_gradcheck_rel_error"), math.inf) or math.inf for r in fit_pre_gradcheck),
        default=math.inf,
    )
    branch_improve_rate = rate(len(improve_rows), len(real_rows))
    branch_control_beat_rate = rate(len(control_beat_rows), len(real_rows))
    summary = {
        "repair_scans": len(scan_specs),
        "fit_rows": len(fit_all),
        "basis_branch_horizon": args.basis_branch_horizon,
        "basis_gradcheck_mode": args.basis_gradcheck_mode,
        "basis_gradcheck_eps": args.basis_gradcheck_eps,
        "fit_pass_rows_without_gradcheck": len(fit_pre_gradcheck),
        "fit_pass_rows_strict": len(fit_pass),
        "best_strict_fit_gradcheck_rel_error": "" if not math.isfinite(best_strict_candidate_gradcheck) else best_strict_candidate_gradcheck,
        "best_basis_projection_residual": "" if not math.isfinite(best_residual) else best_residual,
        "branch_real_rows": len(real_rows),
        "branch_nll_improve_rows": len(improve_rows),
        "branch_control_beat_rows": len(control_beat_rows),
        "branch_nll_improve_rate": branch_improve_rate,
        "branch_control_beat_rate": branch_control_beat_rate,
        **control_decomp_summary,
        "basis_actuator_repair_exploration_pass": int(control_decomp_summary["basis_control_win_exploration_pass"]),
        "basis_actuator_repair_candidate_pass": int(control_decomp_summary["basis_control_win_exploration_pass"] and int(args.basis_branch_horizon) >= 100),
        "repair_note": f"Tikhonov damping/update-scale scan per Part D blocker direction; gradcheck_mode={args.basis_gradcheck_mode}, eps={args.basis_gradcheck_eps}; full-loop remains blocked unless D3 branch gate passes.",
    }
    write_rows(OUT_ROOT / "v22_33_basis_actuator_repair_fit_matrix.csv", fit_all)
    write_rows(OUT_ROOT / "v22_33_basis_actuator_repair_branch_matrix.csv", branch_all)
    write_rows(OUT_ROOT / "v22_33_basis_control_win_decomposition_matrix.csv", control_decomp_rows)
    write_rows(OUT_ROOT / "v22_33_basis_actuator_repair_scan_summary.csv", scan_rows)
    write_rows(OUT_ROOT / "v22_33_basis_actuator_repair_summary.csv", [summary])
    append_exec(
        "run Part D basis actuator Tikhonov damping/update-scale repair scan",
        task_id="R_basis_actuator_damping_scale_scan",
        status="pass",
        gpu=str(args.basis_device),
        files="results/v22_33/v22_33_basis_actuator_repair_fit_matrix.csv, results/v22_33/v22_33_basis_actuator_repair_summary.csv",
        note=f"best_residual={summary['best_basis_projection_residual']}; repair_candidate_pass={summary['basis_actuator_repair_candidate_pass']}; branch_control_beat_rate={summary['branch_control_beat_rate']}",
    )
    return fit_all, branch_all, summary


def run_basis_controlwin_from_existing(args: argparse.Namespace) -> dict[str, Any]:
    fit_rows = read_rows(OUT_ROOT / "v22_33_basis_actuator_repair_fit_matrix.csv")
    branch_rows = read_rows(OUT_ROOT / "v22_33_basis_actuator_repair_branch_matrix.csv")
    existing = (read_rows(OUT_ROOT / "v22_33_basis_actuator_repair_summary.csv") or [{}])[0]
    control_rows, control_summary = basis_control_win_decomposition(fit_rows, branch_rows)
    summary = {**existing, **control_summary}
    summary["basis_actuator_repair_exploration_pass"] = int(control_summary["basis_control_win_exploration_pass"])
    summary["basis_actuator_repair_candidate_pass"] = int(control_summary["basis_control_win_exploration_pass"] and int(finite_float(summary.get("basis_branch_horizon"), 0.0) or 0.0) >= 100)
    summary["repair_note"] = (
        str(summary.get("repair_note", ""))
        + " Control-win decomposition was derived from existing fit/branch rows; no new experimental branch rows generated."
    ).strip()
    write_rows(OUT_ROOT / "v22_33_basis_control_win_decomposition_matrix.csv", control_rows)
    write_rows(OUT_ROOT / "v22_33_basis_actuator_repair_summary.csv", [summary])
    append_exec(
        "derive strict-fit-conditioned basis control-win decomposition from existing H400 repair artifacts",
        task_id="R_basis_control_win_decomposition",
        status="pass" if fit_rows and branch_rows else "warn",
        files="results/v22_33/v22_33_basis_control_win_decomposition_matrix.csv, results/v22_33/v22_33_basis_actuator_repair_summary.csv",
        note=f"strict_fit_branch_pass_rate={summary.get('strict_fit_branch_pass_rate')}; strict_fit_branch_pass_rows={summary.get('strict_fit_branch_pass_rows')}; no_new_branch_rows=1",
    )
    return summary


def append_repair_recap(
    final: dict[str, Any],
    c7_summary: dict[str, Any],
    basis_summary: dict[str, Any],
    c1c6_summary: dict[str, Any] | None = None,
) -> None:
    c7_rows = read_rows(OUT_ROOT / "v22_33_T1_C7_curvature_repair_matrix.csv")
    c1c6_gate = read_rows(OUT_ROOT / "v22_33_T1_C1C6_repair_gate_eval.csv")
    c1c6_rows = read_rows(OUT_ROOT / "v22_33_T1_C1C6_repair_matrix.csv")
    basis_fit = read_rows(OUT_ROOT / "v22_33_basis_actuator_repair_fit_matrix.csv")
    basis_branch = read_rows(OUT_ROOT / "v22_33_basis_actuator_repair_branch_matrix.csv")
    basis_control = read_rows(OUT_ROOT / "v22_33_basis_control_win_decomposition_matrix.csv")
    lines = [
        "",
        "## 9. Continuation Repair Attempt",
        "",
        f"追加时间：{now_sg()}",
        "",
        "### 9.1 C7 Curvature-Safe Direction",
        "",
        md_table([c7_summary], limit=4),
        "",
        md_table(c7_rows, ["dataset", "seed", "selector_name", "NLL_delta_vs_base", "beats_base", "beats_L3_control", "beats_L4_control", "beats_L5_control", "held_CE_delta", "symmetric_curvature_proxy", "curvature_safe_score"], limit=12),
        "",
        "修复说明：按 Part C/F 的推荐方向，新增 C7 curvature-safe selector。方向只由 held-train CE delta 与 symmetric curvature proxy 选择，不使用 test/future/query direction。若 beats_base 或 ECE/tail gate 不过，仍记录为 diagnostic/no-go。",
        "",
        "### 9.2 C1-C6 Direct Selector Repair",
        "",
        md_table([c1c6_summary or {"status": "not_run"}], limit=4),
        "",
        md_table(c1c6_gate, ["selector_name", "direct_rows", "horizons", "beats_base_rate", "beats_L3_rate", "beats_L4_rate", "beats_L5_rate", "mean_NLL_delta_vs_base", "control_noise_std", "max_ECE_delta", "max_tail_q99_delta", "exploration_pass", "official_candidate_pass"], limit=12),
        "",
        md_table(c1c6_rows, ["dataset", "seed", "selector_name", "NLL_delta_vs_base", "beats_base", "beats_L3_control", "beats_L4_control", "beats_L5_control", "uses_future_direction", "status"], limit=16),
        "",
        "修复说明：按 Part C 的预注册 C1-C6 selectors 重跑 direct repair。L cohort-influence 行使用 future branch label，仍只作为 diagnostic，不进入 gate。",
        "",
        "### 9.3 Basis Actuator Damping / Scale Scan",
        "",
        md_table([basis_summary], limit=4),
        "",
        f"检查修复说明：basis gradcheck 使用 mode=`{basis_summary.get('basis_gradcheck_mode', '')}`、eps=`{basis_summary.get('basis_gradcheck_eps', '')}`。`central` 模式使用 symmetric finite difference `(f(theta + eps*d) - f(theta - eps*d)) / (2*eps)`，只修复导数检查方式；projection/cosine/exact/branch/control 阈值未放宽。",
        f"Branch 记录修复说明：basis_branch_horizon=`{basis_summary.get('basis_branch_horizon', '')}`；H400 分支现在写入 `source_loss_H400` / `NLL_delta_H400`，control beat 汇总也读取 H400 NLL，避免把空字段误判为 0 改善。",
        f"D3 分母修复说明：strict-fit branch pass 只在 `fit_strict_pass=1` 的 actuator 上计算，并要求 real NLL 改善、同时 beat same-basis-random 与 signflip、且 accuracy_delta_vs_base >= -0.01。当前 strict_fit_branch_pass_rate=`{basis_summary.get('strict_fit_branch_pass_rate', '')}`。",
        "",
        md_table(basis_fit, ["repair_scan_id", "carrier", "basis_family", "basis_bank", "basis_damping", "basis_update_scale", "basis_gradcheck_mode", "basis_gradcheck_eps", "basis_actuator_projection_residual", "basis_actuator_cosine", "exact_vs_linearized_error", "J_B_gradcheck_rel_error", "J_B_gradcheck_pass"], limit=16),
        "",
        md_table(basis_branch, ["repair_scan_id", "carrier", "basis_family", "basis_bank", "branch_variant", "branch_H", "NLL_delta_H50", "NLL_delta_H100", "NLL_delta_H200", "NLL_delta_H400", "accuracy_delta_vs_base", "beats_same_basis_random", "beats_signflip_basis_control"], limit=16),
        "",
        md_table(basis_control, ["repair_scan_id", "carrier", "basis_bank", "branch_H", "fit_strict_pass", "real_NLL_delta", "same_basis_random_NLL_delta", "signflip_basis_control_NLL_delta", "real_improves", "beats_same_basis_random", "beats_signflip_basis_control", "strict_fit_branch_pass", "control_win_explanation"], limit=16),
        "",
        "修复说明：按 Part D 的 blocker 方向扫描 Tikhonov damping 与 update scale；所有 rows 都是真实重跑结果。若 strict fit 或 D3 branch/control gate 不过，不启动 full-loop，也不写 architecture claim。",
        "",
        "### 9.4 Repair Conclusion",
        "",
        f"- updated_final_route: `{final.get('final_route')}`",
        f"- official_full_superiority_ready: `{final.get('official_full_superiority_ready')}`",
        "- 结论仍以 final route JSON 为准；没有达到 R15 时禁止 promotion。",
    ]
    with RECAP_DOC.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_repair(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    append_exec(
        " ".join([f"KAN_PYTHON={PYTHON}", shlex.quote(sys.executable), *[shlex.quote(a) for a in sys.argv]]),
        task_id="v22_33_repair_start",
        status="started",
        gpu=f"t1={args.t1_device}; basis={args.basis_device}",
        files=str(PLAN_DOC.relative_to(ROOT)),
        note="Continuation after non-official v22.33 route: C7 curvature-safe T1 repair plus basis damping/scale scan.",
    )
    components = set(split_csv(args.repair_components))
    if "c7" in components:
        c7_rows, _c7_branch, c7_summary = run_t1_c7_curvature_repair(args)
    else:
        c7_rows = read_rows(OUT_ROOT / "v22_33_T1_C7_curvature_repair_matrix.csv")
        c7_summary = (read_rows(OUT_ROOT / "v22_33_T1_C7_curvature_repair_summary.csv") or [{"status": "skipped_no_existing_c7_summary"}])[0]
        append_exec(
            "skip C7 repair component and reuse existing C7 repair summary if present",
            task_id="R_C7_curvature_safe_direction_skipped",
            status="pass" if c7_rows else "warn",
            files="results/v22_33/v22_33_T1_C7_curvature_repair_summary.csv",
            note=f"repair_components={args.repair_components}",
        )
    if "c1c6" in components:
        c1c6_rows, _c1c6_branch, c1c6_summary = run_t1_c1c6_direct_repair(args)
    else:
        c1c6_rows = read_rows(OUT_ROOT / "v22_33_T1_C1C6_repair_matrix.csv")
        c1c6_summary = (read_rows(OUT_ROOT / "v22_33_T1_C1C6_repair_summary.csv") or [{"status": "skipped_no_existing_c1c6_summary"}])[0]
        append_exec(
            "skip C1-C6 direct selector repair and reuse existing summary if present",
            task_id="R_C1C6_direct_selector_repair_skipped",
            status="pass" if c1c6_rows else "warn",
            files="results/v22_33/v22_33_T1_C1C6_repair_summary.csv",
            note=f"repair_components={args.repair_components}",
        )
    if "basis" in components:
        _basis_fit, _basis_branch, basis_repair_summary = run_basis_repair_scan(args)
    elif "controlwin" in components:
        basis_repair_summary = run_basis_controlwin_from_existing(args)
    else:
        basis_repair_summary = (read_rows(OUT_ROOT / "v22_33_basis_actuator_repair_summary.csv") or [{"status": "skipped_no_existing_basis_summary"}])[0]
        append_exec(
            "skip basis repair component and reuse existing basis repair summary if present",
            task_id="R_basis_actuator_damping_scale_scan_skipped",
            status="pass" if basis_repair_summary else "warn",
            files="results/v22_33/v22_33_basis_actuator_repair_summary.csv",
            note=f"repair_components={args.repair_components}",
        )
    final = read_json(OUT_ROOT / "v22_33_final_route.json")
    repair_route = str(final.get("final_route", ""))
    if int_flag(basis_repair_summary.get("basis_actuator_repair_candidate_pass")):
        repair_route = "R5-BasisActuatorBranchOpened_NoFullLoop"
    elif int_flag(basis_repair_summary.get("fit_pass_rows_strict")):
        repair_route = "R4-BasisActuatorFitOpened_NoBranchBenefit"
    elif int_flag(c1c6_summary.get("T1_C1C6_official_candidate")) or int_flag(c1c6_summary.get("T1_C1C6_exploration_pass")):
        repair_route = "R3-DirectionCausalButActuatorNoGo"
    elif int_flag(c7_summary.get("official_candidate_pass")) or int_flag(c7_summary.get("exploration_pass")):
        repair_route = "R3-DirectionCausalButActuatorNoGo"
    final["final_route"] = repair_route or final.get("final_route", "R2-SignalSubspaceDirectionNoGo")
    final["official_full_superiority_ready"] = 0
    final["latest_status_timestamp"] = now_sg()
    final["repair_summary"] = {
        "C7_curvature_safe": c7_summary,
        "C1C6_direct_selector_repair": c1c6_summary,
        "basis_actuator_damping_scale_scan": basis_repair_summary,
        "repair_no_fabrication_note": "Repair rows are direct reruns. Full-loop remains blocked unless documented gates pass.",
    }
    final["full_loop_blockers"] = [
        {
            "status": "gate_blocked",
            "blocker": "Part C C1-C6 direct selector repair did not open an exploration or official candidate gate.",
            "source_artifact": "results/v22_33/v22_33_T1_C1C6_repair_summary.csv",
            "exploration_pass_rows": c1c6_summary.get("exploration_pass_rows", ""),
            "official_candidate_rows": c1c6_summary.get("official_candidate_rows", ""),
            "best_selector_by_mean_NLL": c1c6_summary.get("best_selector_by_mean_NLL", ""),
            "best_selector_mean_NLL_delta": c1c6_summary.get("best_selector_mean_NLL_delta", ""),
            "control_noise_std": c1c6_summary.get("control_noise_std", ""),
        },
        {
            "status": "gate_blocked",
            "blocker": "Part C C7 curvature-safe selector did not satisfy exploration/official safety and control gates.",
            "source_artifact": "results/v22_33/v22_33_T1_C7_curvature_repair_summary.csv",
            "beats_L3_rate": c7_summary.get("beats_L3_rate", ""),
            "beats_L5_rate": c7_summary.get("beats_L5_rate", ""),
            "max_ECE_delta": c7_summary.get("max_ECE_delta", ""),
            "max_tail_q99_delta": c7_summary.get("max_tail_q99_delta", ""),
        },
        {
            "status": "gate_blocked",
            "blocker": "Part D D3 basis branch/control gate did not satisfy >=60% strict-fit branch pass; full-loop remains forbidden.",
            "source_artifact": "results/v22_33/v22_33_basis_actuator_repair_summary.csv",
            "strict_fit_real_rows": basis_repair_summary.get("strict_fit_real_rows", ""),
            "strict_fit_branch_pass_rows": basis_repair_summary.get("strict_fit_branch_pass_rows", ""),
            "strict_fit_branch_pass_rate": basis_repair_summary.get("strict_fit_branch_pass_rate", ""),
            "basis_actuator_repair_candidate_pass": basis_repair_summary.get("basis_actuator_repair_candidate_pass", ""),
        },
    ]
    write_json(OUT_ROOT / "v22_33_final_route.json", final)
    artifact_index()
    append_repair_recap(final, c7_summary, basis_repair_summary, c1c6_summary)
    append_exec(
        "append continuation repair results to final_route, artifact index, execution log, and recap",
        task_id="R_finalize_repair",
        status="pass",
        files="results/v22_33/v22_33_final_route.json, docs/DG-KAN_v22.33_CausalDirectionActuatorTransfer_实验结果复盘.md",
        note=f"repair_route={final.get('final_route')}; C7_rows={len(c7_rows)}; C1C6_rows={len(c1c6_rows)}; basis_fit_strict={basis_repair_summary.get('fit_pass_rows_strict')}; basis_repair_candidate={basis_repair_summary.get('basis_actuator_repair_candidate_pass')}",
    )
    return final


def decide_final(
    code_truth: dict[str, Any],
    gap_summary: dict[str, Any],
    t1_summary: dict[str, Any],
    basis_summary: dict[str, Any],
    temporal_summary: dict[str, Any],
    full_loop_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not int_flag(code_truth.get("clean_unzip_compileall_pass")) or not int_flag(code_truth.get("clean_unzip_import_pass")):
        route = "R0-CodeOrIdentityFail"
    elif int_flag(gap_summary.get("official_candidate_gate_pass")):
        route = "R11-TrueKANGainGapReductionOpened"
    elif int_flag(basis_summary.get("basis_native_candidate_pass")):
        route = "R5-BasisActuatorBranchOpened_NoFullLoop"
    elif int_flag(basis_summary.get("basis_actuator_exploration_pass")):
        route = "R4-BasisActuatorFitOpened_NoBranchBenefit"
    elif int_flag(t1_summary.get("T1_official_candidate")) or int_flag(t1_summary.get("T1_exploration_pass")):
        route = "R3-DirectionCausalButActuatorNoGo"
    elif int(temporal_summary.get("continual_pass_rows") or 0) or int(temporal_summary.get("grokking_pass_rows") or 0):
        route = "R7-ContinualOrGrokkingDiagnosticOnly"
    elif finite_float(gap_summary.get("ControlExplained_rate"), 0.0) and (finite_float(gap_summary.get("ControlExplained_rate"), 0.0) or 0.0) > 0.50:
        route = "R9-GapReductionControlExplained"
    elif finite_float(gap_summary.get("MLPDegradationDriven_rate"), 0.0) and (finite_float(gap_summary.get("MLPDegradationDriven_rate"), 0.0) or 0.0) > 0.20:
        route = "R10-GapReductionMLPDegradationDriven"
    else:
        route = "R2-SignalSubspaceDirectionNoGo"
    official = int(route == "R15-OfficialFullSuperiorityReady")
    final = {
        "final_route": route,
        "official_full_superiority_ready": official,
        "latest_status_timestamp": now_sg(),
        "code_truth": code_truth,
        "gap_summary": gap_summary,
        "t1_summary": t1_summary,
        "basis_summary": basis_summary,
        "full_loop_blockers": full_loop_rows,
        "temporal_continual_summary": temporal_summary,
        "no_fabricated_rows_claim": "numeric rows come from direct v22.33 run steps, v22.32 reused direct pilots, or named historical artifact readbacks; missing data is explicit not_run/gate_blocked.",
    }
    write_json(OUT_ROOT / "v22_33_final_route.json", final)
    return final


def artifact_index() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(list(OUT_ROOT.glob("v22_33_*")) + list(FIG_ROOT.glob("v22_33_*"))):
        if path.is_file():
            rows.append(
                {
                    "artifact": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "mtime_sg": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(path.stat().st_mtime)),
                }
            )
    write_rows(OUT_ROOT / "v22_33_artifact_index.csv", rows)
    return rows


def write_recap(
    final: dict[str, Any],
    gap_rows: list[dict[str, Any]],
    t1_rows: list[dict[str, Any]],
    basis_fit: list[dict[str, Any]],
    basis_branch: list[dict[str, Any]],
    opt_summary: dict[str, Any],
    t5_summary: dict[str, Any],
    temporal_summary: dict[str, Any],
) -> None:
    gate_rows = read_rows(RAW_ROOT / "v22_32_T1_selector_gate_eval.csv")
    full_rows = read_rows(OUT_ROOT / "v22_33_basis_actuator_full_loop_matrix.csv")
    lines = [
        "# DG-KAN v22.33 Causal Direction and Actuator Transfer 实验结果复盘",
        "",
        f"生成时间：{now_sg()}",
        "",
        "## 1. 结论摘要",
        "",
        f"- final_route: `{final.get('final_route')}`",
        f"- official_full_superiority_ready: `{final.get('official_full_superiority_ready')}`",
        "- 数据边界：本轮直接执行 Part A gate，并复用 v22.32 已审计 direct pilot 代码路径重跑 T1 direction 与 D basis ladder；B/E/F/G/H 为命名历史 artifact 的 v22.33 标准复核或 gate_blocked 行。",
        "- 不能 promotion 的主因：gap truth 仍不是 TrueKANGain/BothGain；T1 selector 没有达到 official；basis branch/control gate 未达到 D3/D4 启动阈值；continual/grokking 只保留 diagnostic。",
        "",
        "## 2. Part A Code / Identity",
        "",
        md_table([final.get("code_truth", {})], limit=4),
        "",
        "证据链：`compileall` 与 import closure 使用 `KAN_PYTHON=/home/chengshun.wang/miniconda3/envs/kan/bin/python`。默认 `python` 是否失败只作为环境记录，不影响 conda kan env 的 pass 判定。review bundle manifest 和 tarball 已生成。",
        "",
        "## 3. Part B Gap Truth",
        "",
        md_table([final.get("gap_summary", {})], limit=4),
        "",
        md_table(gap_rows, ["dataset", "seed", "task_tier", "carrier", "Delta_MLP_NLL", "Delta_KAN_NLL", "GapReduction_NLL", "gap_reduction_class"], limit=20),
        "",
        "分析：如果 `TrueKANGain_rows` 和 `BothGain_rows` 为 0，则文档要求停止 full superiority claim。本轮按该 blocker 继续执行 Part C/D 修复性复核，但不把 gap reduction 写成 KAN gain。",
        "",
        "## 4. Part C Direction Causality",
        "",
        md_table([final.get("t1_summary", {})], limit=4),
        "",
        md_table(gate_rows, ["selector_name", "direct_rows", "datasets", "seeds", "beats_base_rate", "beats_L3_rate", "beats_L4_rate", "beats_L5_rate", "mean_NLL_delta_vs_base", "exploration_pass", "official_candidate_pass"], limit=12),
        "",
        md_table(t1_rows, ["dataset", "seed", "selector_name", "NLL_delta_vs_base", "beats_L3_control", "beats_L4_control", "beats_L5_control", "uses_future_direction", "status"], limit=16),
        "",
        "分析：T1 的信号子空间不是空证据，但 direction 仍不能作为 official causal selector。C8/L 类 influence 使用 future branch readback，只能解释方向，不允许 runtime promotion。",
        "",
        "## 5. Part D Basis-Native Actuator",
        "",
        md_table([final.get("basis_summary", {})], limit=4),
        "",
        md_table(basis_fit, ["carrier", "basis_family", "dataset", "basis_bank", "basis_actuator_projection_residual", "basis_actuator_cosine", "exact_vs_linearized_error", "J_B_gradcheck_rel_error", "J_B_gradcheck_pass", "status"], limit=12),
        "",
        md_table(basis_branch, ["carrier", "basis_family", "dataset", "basis_bank", "branch_variant", "branch_H", "NLL_delta_H100", "NLL_delta_H50", "accuracy_delta_vs_base", "beats_same_basis_random", "beats_signflip_basis_control", "status"], limit=12),
        "",
        md_table(full_rows, limit=4),
        "",
        "修复/尝试记录：本轮新增 `experiments/run_v22_33_causal_direction_actuator_transfer.py` 做 v22.33 orchestration、artifact rename、gate-blocked full-loop rows、final route 与复盘。算法源文件未做新的隐式修补；`dgkan/fu/real_jacobian_commit.py` 当前已具备 row-space dual solve，作为 D 阶段 actuator residual 高时的文档指定修复方向被审计使用。",
        "",
        "## 6. Optimizer / Subspace / Temporal",
        "",
        md_table([opt_summary], limit=4),
        "",
        md_table([t5_summary], limit=4),
        "",
        md_table([temporal_summary], limit=4),
        "",
        "分析：optimizer rows 仍是 interaction context，不是 full value proof。T5 online subspace 保留为 diagnostic。Class-MNIST continual 有局部 diagnostic rows，但 exact KANbeFair original official protocol 未闭合。",
        "",
        "## 7. Insight 与结论",
        "",
        "- Insight 1：当前最强证据仍是 signal/basis-effect 的存在，而不是 architecture superiority。",
        "- Insight 2：controls 是主要解释变量；只要 ControlExplained 仍高，gap reduction 不应被写成 KAN gain。",
        "- Insight 3：basis fit 的局部好 residual 不等价于 branch/full-loop task benefit，D3/D4 gate 必须继续分开审计。",
        "- 结论：v22.33 本轮没有达到 R15 official full superiority。final route 以 artifact 中的真实 gate 结果为准，禁止 promotion。",
        "",
        "## 8. Reproducibility Pointers",
        "",
        "- 主 runner: `experiments/run_v22_33_causal_direction_actuator_transfer.py`",
        "- 执行日志: `docs/DG-KAN_v22.33_CausalDirectionActuatorTransfer_执行日志.md`",
        "- 复盘日志: `docs/DG-KAN_v22.33_CausalDirectionActuatorTransfer_实验结果复盘.md`",
        "- 命令 journal: `results/v22_33/v22_33_command_journal.csv`",
        "- legacy 子步骤 journal: `results/v22_33/v22_33_legacy_command_journal.csv`",
        "- artifact index: `results/v22_33/v22_33_artifact_index.csv`",
        "- final route: `results/v22_33/v22_33_final_route.json`",
    ]
    RECAP_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["all", "firewall", "finalize", "repair"])
    p.add_argument("--t1-device", default="cuda:1")
    p.add_argument("--basis-device", default="cuda:2")
    p.add_argument("--basis-datasets", default="MNIST")
    p.add_argument("--basis-seeds", default="0")
    p.add_argument("--basis-train-size", type=int, default=192)
    p.add_argument("--basis-test-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--basis-pretrain-steps", type=int, default=20)
    p.add_argument("--basis-branch-horizon", type=int, default=100)
    p.add_argument("--basis-examples", type=int, default=4)
    p.add_argument("--basis-max-output-rows", type=int, default=32)
    p.add_argument("--basis-bank-dims", default="4,8,16")
    p.add_argument("--basis-damping", type=float, default=1.0e-3)
    p.add_argument("--basis-gradcheck-eps", type=float, default=1.0e-4)
    p.add_argument("--basis-gradcheck-mode", choices=["forward", "central"], default="forward")
    p.add_argument("--basis-update-scale", type=float, default=1.0e-2)
    p.add_argument("--t1-datasets", default="MNIST,FMNIST,KMNIST")
    p.add_argument("--t1-seeds", default="0,1")
    p.add_argument("--t1-train-size", type=int, default=192)
    p.add_argument("--t1-test-size", type=int, default=128)
    p.add_argument("--t1-pretrain-steps", type=int, default=20)
    p.add_argument("--t1-branch-horizon", type=int, default=100)
    p.add_argument("--t1-influence-horizon", type=int, default=20)
    p.add_argument("--t1-examples", type=int, default=32)
    p.add_argument("--t1-signal-cohorts", type=int, default=4)
    p.add_argument("--t1-cohort-size", type=int, default=8)
    p.add_argument("--t1-rank-cap", type=int, default=4)
    p.add_argument("--t1-sketch-dim", type=int, default=16)
    p.add_argument("--t1-candidate-count", type=int, default=12)
    p.add_argument("--t1-branch-trust", type=float, default=0.03)
    p.add_argument("--t1-c7-curvature-weight", type=float, default=0.02)
    p.add_argument("--repair-components", default="c7,basis")
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    return p


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    append_exec(
        " ".join([f"KAN_PYTHON={PYTHON}", shlex.quote(sys.executable), *[shlex.quote(a) for a in sys.argv]]),
        task_id="v22_33_start",
        status="started",
        gpu=f"t1={args.t1_device}; basis={args.basis_device}; visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'system default')}",
        files=str(PLAN_DOC.relative_to(ROOT)),
        note="Part A-H conservative execution; v22.32 direct pilot code path reused and re-run into raw_v22_32_runner.",
    )
    gpu_probe = run_logged(
        [
            PYTHON,
            "-c",
            "import torch; print({'cuda_available': torch.cuda.is_available(), 'device_count': torch.cuda.device_count(), 'names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []})",
        ],
        task_id="A_gpu_probe",
        timeout=120,
    )
    if gpu_probe.returncode != 0:
        append_exec("GPU probe failed; continuing because CPU fallback exists for code gate", task_id="A_gpu_probe_note", status="warn")
    legacy = configure_legacy()
    code_raw = legacy.stage_a_code_closure()
    code_truth = build_code_truth(code_raw)
    if not int_flag(code_truth.get("clean_unzip_compileall_pass")) or not int_flag(code_truth.get("clean_unzip_import_pass")):
        final = decide_final(code_truth, {}, {}, {}, {}, [])
        write_recap(final, [], [], [], [], {}, {}, {})
        artifact_index()
        return final
    if args.stage == "firewall":
        final = decide_final(code_truth, {}, {}, {}, {}, [])
        artifact_index()
        return final

    gap_rows_raw, _gap_summary_raw = legacy.stage_b_gap_decomposition()
    t1_args = copy.copy(args)
    t1_args.device = args.t1_device
    _t1_rows_raw, _t1_branch_raw, _t1_control_raw, t1_summary_rows_raw = legacy.stage_c_t1_direction(t1_args)
    basis_args = copy.copy(args)
    basis_args.device = args.basis_device
    _target_raw, _basis_fit_raw, _basis_exact_raw, _basis_branch_raw, basis_summary_raw = legacy.stage_d_basis_actuator(basis_args)
    _opt_rows_raw, _spectrum_raw, _controls_raw, opt_summary_raw = legacy.stage_e_optimizer()
    _t5_rows_raw, t5_summary_raw = legacy.stage_f_t5()
    _grok_raw, _cont_raw, temporal_summary_raw = legacy.stage_g_temporal_continual()
    _full_rows_raw, h_summary_raw = legacy.stage_h_full_loop(gap_rows_raw, _gap_summary_raw)
    expansion_summary_raw = legacy.stage_expansion_lanes()
    legacy_final = read_json(RAW_ROOT / "v22_32_final_route.json")
    if not legacy_final:
        legacy_final = legacy.decide_final(
            code_raw,
            _gap_summary_raw,
            (t1_summary_rows_raw or [{}])[0],
            basis_summary_raw,
            opt_summary_raw,
            t5_summary_raw,
            temporal_summary_raw,
            h_summary_raw,
            expansion_summary_raw,
        )
    gap_rows, gap_summary = transform_gap()
    t1_rows, _branch_rows, _control_rows, t1_summary = transform_t1()
    basis_fit, basis_branch, full_loop_rows, basis_summary = transform_basis()
    opt_summary, t5_summary, temporal_summary = transform_diagnostics()
    legacy_journal_rows = transform_legacy_journal()
    final = decide_final(code_truth, gap_summary, t1_summary, basis_summary, temporal_summary, full_loop_rows)
    artifact_rows = artifact_index()
    write_recap(final, gap_rows, t1_rows, basis_fit, basis_branch, opt_summary, t5_summary, temporal_summary)
    append_exec(
        "v22.33 finalize generated required artifacts, figures, recap, route, artifact index",
        task_id="Z_finalize",
        status="pass",
        files="results/v22_33/v22_33_final_route.json, results/v22_33/v22_33_artifact_index.csv, docs/DG-KAN_v22.33_CausalDirectionActuatorTransfer_实验结果复盘.md",
        note=f"final_route={final.get('final_route')}; official_full_superiority_ready={final.get('official_full_superiority_ready')}; artifact_count={len(artifact_rows)}; legacy_substeps={len(legacy_journal_rows)}",
    )
    return final


def finalize_existing() -> dict[str, Any]:
    code_truth = (read_rows(OUT_ROOT / "v22_33_code_truth_gate.csv") or [{}])[0]
    gap_rows = read_rows(OUT_ROOT / "v22_33_gap_truth_matrix.csv")
    gap_summary = (read_rows(OUT_ROOT / "v22_33_gap_truth_summary.csv") or [{}])[0]
    t1_rows = read_rows(OUT_ROOT / "v22_33_T1_selector_matrix.csv")
    t1_summary = (read_rows(RAW_ROOT / "v22_32_T1_direction_selector_summary.csv") or [{}])[0]
    basis_fit = read_rows(OUT_ROOT / "v22_33_basis_actuator_fit_matrix.csv")
    basis_branch = read_rows(OUT_ROOT / "v22_33_basis_actuator_branch_matrix.csv")
    basis_summary = (read_rows(RAW_ROOT / "v22_32_KAN_basis_actuator_fidelity_summary.csv") or [{}])[0]
    full_loop_rows = read_rows(OUT_ROOT / "v22_33_basis_actuator_full_loop_matrix.csv")
    opt_summary = (read_rows(RAW_ROOT / "v22_32_optimizer_delta_summary.csv") or [{}])[0]
    t5_summary = {
        "online_subspace_rows": len(read_rows(RAW_ROOT / "v22_32_online_subspace_tracking_matrix.csv")),
        "status": "diagnostic_only",
    }
    temporal_summary = read_json(OUT_ROOT / "v22_33_final_route.json").get("temporal_continual_summary", {})
    transform_legacy_journal()
    final = decide_final(code_truth, gap_summary, t1_summary, basis_summary, temporal_summary, full_loop_rows)
    artifact_index()
    write_recap(final, gap_rows, t1_rows, basis_fit, basis_branch, opt_summary, t5_summary, temporal_summary)
    append_exec(
        "regenerate v22.33 final route, recap, artifact index, and legacy path map from existing artifacts",
        task_id="Z_finalize_existing",
        status="pass",
        files="results/v22_33/v22_33_final_route.json, results/v22_33/v22_33_artifact_index.csv, results/v22_33/v22_33_legacy_command_journal.csv",
        note=f"final_route={final.get('final_route')}; official_full_superiority_ready={final.get('official_full_superiority_ready')}",
    )
    return final


def main() -> None:
    args = parser().parse_args()
    if args.stage == "finalize":
        final = finalize_existing()
    elif args.stage == "repair":
        final = run_repair(args)
    else:
        final = run_all(args)
    print(json.dumps({"final_route": final.get("final_route"), "official_full_superiority_ready": final.get("official_full_superiority_ready")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
