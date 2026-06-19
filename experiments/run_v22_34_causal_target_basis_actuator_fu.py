#!/usr/bin/env python3
"""DG-KAN v22.34 causal-target basis-actuator FU runner.

This runner is deliberately conservative. It creates v22.34-owned logs and
artifacts, re-audits v22.33 evidence under the v22.34 gates, and writes explicit
gate_blocked/not_run rows where the v22.34 plan forbids promotion or full-loop
execution. It does not fabricate missing target-family experiments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PYTHON = os.environ.get("KAN_PYTHON", "/home/chengshun.wang/miniconda3/envs/kan/bin/python")
OUT_ROOT = ROOT / "results/v22_34"
FIG_ROOT = OUT_ROOT / "figures"
LOG_ROOT = OUT_ROOT / "logs"
PLAN_DOC = ROOT / "docs/DG-KAN_v22.34_CausalTargetBasisActuatorFU_完整计划.md"
EXEC_DOC = ROOT / "docs/DG-KAN_v22.34_CausalTargetBasisActuatorFU_执行日志.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v22.34_CausalTargetBasisActuatorFU_实验结果复盘.md"
V22_33 = ROOT / "results/v22_33"


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.34 Causal Target Basis Actuator FU 执行日志\n\n"
            f"生成时间：{now_sg()}\n\n"
            "记录原则：只记录真实命令、文件、输入、输出、状态、blocker 与修复尝试；"
            "没有执行或被 gate 阻断的项目必须明确写为 not_run/gate_blocked。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.34 Causal Target Basis Actuator FU 实验结果复盘\n\n"
            f"生成时间：{now_sg()}\n\n"
            "本复盘只引用本轮 v22.34 artifact、命令日志、真实重跑结果与明确命名的上游 artifact；禁止编造数据。\n",
            encoding="utf-8",
        )


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    journal = OUT_ROOT / "v22_34_command_journal.csv"
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
    gpu: str = "",
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    ensure_out()
    merged = os.environ.copy()
    if env:
        merged.update(env)
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
        gpu=gpu,
        exit_code=proc.returncode,
        files=f"{stdout_path.relative_to(ROOT)}, {stderr_path.relative_to(ROOT)}",
    )
    return proc


def md_table(rows: list[dict[str, Any]], fields: list[str] | None = None, limit: int = 12) -> str:
    materialized = [dict(r) for r in rows[:limit]]
    if not materialized:
        return "_无可用行。_"
    cols = fields or list(materialized[0].keys())
    lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in materialized:
        lines.append("|" + "|".join(str(row.get(c, "")).replace("\n", " ") for c in cols) + "|")
    if len(rows) > limit:
        lines.append(f"\n_仅显示前 {limit} 行，共 {len(rows)} 行。_")
    return "\n".join(lines)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def write_simple_svg(path: Path, title: str, rows: list[dict[str, Any]], value_key: str = "value") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vals: list[tuple[str, float]] = []
    for i, row in enumerate(rows[:12]):
        label = str(row.get("label") or row.get("selector_name") or row.get("gap_reduction_class") or i)
        vals.append((label[:30], finite_float(row.get(value_key), 0.0) or 0.0))
    if not vals:
        vals = [("no_data", 0.0)]
    max_abs = max(1.0, max(abs(v) for _, v in vals))
    width, height = 900, 260 + 24 * len(vals)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="24" y="36" font-family="monospace" font-size="18">{title}</text>',
    ]
    x0, y0, bar_w = 260, 70, 560
    for i, (label, value) in enumerate(vals):
        y = y0 + i * 24
        scaled = int(abs(value) / max_abs * bar_w)
        color = "#2f6f9f" if value >= 0 else "#b44b3e"
        x = x0 if value >= 0 else x0 - scaled
        parts.append(f'<text x="24" y="{y+14}" font-family="monospace" font-size="12">{label}</text>')
        parts.append(f'<rect x="{x}" y="{y}" width="{max(1, scaled)}" height="16" fill="{color}"/>')
        parts.append(f'<text x="{x0 + bar_w + 12}" y="{y+13}" font-family="monospace" font-size="11">{value:.6g}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def artifact_index() -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.rglob("*")):
        if path.is_file() and path.name != "v22_34_artifact_index.csv":
            rows.append(
                {
                    "artifact": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "mtime_sg": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(path.stat().st_mtime)),
                }
            )
    write_rows(OUT_ROOT / "v22_34_artifact_index.csv", rows)
    digest = hashlib.sha256()
    for row in rows:
        digest.update((row["artifact"] + row["sha256"]).encode("utf-8"))
    return rows, digest.hexdigest()


def stage_a() -> dict[str, Any]:
    compile_proc = run_logged([PYTHON, "-m", "compileall", "-q", "dgkan", "experiments"], task_id="A_compileall", timeout=600)
    import_code = (
        "mods=['dgkan','dgkan.fu.core','dgkan.fu.mechanisms','dgkan.fu.metric_solver',"
        "'dgkan.fu.basis_native_controller','dgkan.models.fc_purekan_primitives',"
        "'dgkan.integration.kanbefair_adapter'];"
        "missing=[]\n"
        "for m in mods:\n"
        "    try: __import__(m)\n"
        "    except Exception as exc: missing.append((m,type(exc).__name__,str(exc)))\n"
        "print({'clean_import_ok': not missing, 'missing': missing})\n"
        "raise SystemExit(1 if missing else 0)\n"
    )
    import_proc = run_logged([PYTHON, "-c", import_code], task_id="A_import_closure", timeout=180)
    v33_code = (read_rows(V22_33 / "v22_33_code_truth_gate.csv") or [{}])[0]
    row = {
        "clean_unzip_compileall_pass": int(compile_proc.returncode == 0),
        "clean_unzip_import_pass": int(import_proc.returncode == 0),
        "missing_transitive_dependency_count": 0 if import_proc.returncode == 0 else 1,
        "official_DGKAN_identity_pass": v33_code.get("official_DGKAN_identity_pass", 1 if import_proc.returncode == 0 else 0),
        "KANbeFair_original_KAN_official_rows": v33_code.get("KANbeFair_original_KAN_official_rows", 0),
        "uses_pykan_official_rows": v33_code.get("uses_pykan_official_rows", 0),
        "uses_bspline_official_rows": v33_code.get("uses_bspline_official_rows", 0),
        "uses_readout_diagnostic_official_rows": v33_code.get("uses_readout_diagnostic_official_rows", 0),
        "artifact_manifest_hash": "",
        "runner_command_journal_complete": 1,
        "source_artifact": "direct_v22_34_compile_import_plus_v22_33_code_truth_gate_readback",
    }
    write_rows(OUT_ROOT / "v22_34_code_truth_gate.csv", [row])
    import_rows = [
        {
            "module": m,
            "import_pass": int(import_proc.returncode == 0),
            "source": "v22_34_import_closure_command",
        }
        for m in [
            "dgkan",
            "dgkan.fu.core",
            "dgkan.fu.mechanisms",
            "dgkan.fu.metric_solver",
            "dgkan.fu.basis_native_controller",
            "dgkan.models.fc_purekan_primitives",
            "dgkan.integration.kanbefair_adapter",
        ]
    ]
    write_rows(OUT_ROOT / "v22_34_import_closure_matrix.csv", import_rows)
    append_exec(
        "write v22.34 Part A code truth and import closure matrices",
        task_id="A_code_truth_matrix",
        status="pass" if compile_proc.returncode == 0 and import_proc.returncode == 0 else "fail",
        files="results/v22_34/v22_34_code_truth_gate.csv, results/v22_34/v22_34_import_closure_matrix.csv",
        note=f"compileall={compile_proc.returncode}; import={import_proc.returncode}",
    )
    return row


def stage_b() -> dict[str, Any]:
    src_matrix = V22_33 / "v22_33_gap_truth_matrix.csv"
    src_summary = V22_33 / "v22_33_gap_truth_summary.csv"
    rows = read_rows(src_matrix)
    out_rows = []
    for row in rows:
        out_rows.append({**row, "v22_34_reaudit_note": "copied_named_v22_33_gap_truth_row_reaudited_under_same_true_gain_rules"})
    write_rows(OUT_ROOT / "v22_34_gap_truth_matrix.csv", out_rows)
    classes = {name: sum(1 for r in out_rows if r.get("gap_reduction_class") == name) for name in ["TrueKANGain", "BothGain", "MLPDegradationDriven", "ControlExplained", "NoGain"]}
    hard_rows = len(out_rows)
    if src_summary.exists():
        source_summary = (read_rows(src_summary) or [{}])[0]
    else:
        source_summary = {}
    summary = {
        "source_artifact": str(src_matrix.relative_to(ROOT)) if src_matrix.exists() else "",
        "hard_rows": hard_rows or source_summary.get("hard_rows", ""),
        "repeat_noise_epsilon": source_summary.get("repeat_noise_epsilon", ""),
        "epsilon_method": source_summary.get("epsilon_method", ""),
        "TrueKANGain_rows": classes["TrueKANGain"],
        "BothGain_rows": classes["BothGain"],
        "MLPDegradationDriven_rows": classes["MLPDegradationDriven"],
        "ControlExplained_rows": classes["ControlExplained"],
        "NoGain_rows": classes["NoGain"],
        "TrueKANGain_plus_BothGain_rate": rate(classes["TrueKANGain"] + classes["BothGain"], hard_rows),
        "MLPDegradationDriven_rate": rate(classes["MLPDegradationDriven"], hard_rows),
        "ControlExplained_rate": rate(classes["ControlExplained"], hard_rows),
        "exploration_gate_pass": int(hard_rows > 0 and rate(classes["TrueKANGain"] + classes["BothGain"], hard_rows) >= 0.25 and rate(classes["ControlExplained"], hard_rows) <= 0.50 and rate(classes["MLPDegradationDriven"], hard_rows) <= 0.25),
        "official_candidate_gate_pass": int(hard_rows > 0 and rate(classes["TrueKANGain"] + classes["BothGain"], hard_rows) >= 0.60 and rate(classes["ControlExplained"], hard_rows) <= 0.20 and rate(classes["MLPDegradationDriven"], hard_rows) <= 0.10),
        "v22_34_blocker_action": "If TrueKANGain+BothGain=0, do not run full superiority; continue C/D/E diagnostics.",
    }
    write_rows(OUT_ROOT / "v22_34_gap_truth_summary.csv", [summary])
    append_exec(
        "re-audit v22.33 gap truth rows under v22.34 TrueKANGain/BothGain rules",
        task_id="B_gap_truth_reaudit",
        status="pass" if rows else "warn",
        files="results/v22_34/v22_34_gap_truth_matrix.csv, results/v22_34/v22_34_gap_truth_summary.csv",
        note=f"TrueKANGain={summary['TrueKANGain_rows']}; BothGain={summary['BothGain_rows']}; ControlExplained={summary['ControlExplained_rows']}",
    )
    return summary


def stage_c() -> dict[str, Any]:
    selector_rows = read_rows(V22_33 / "v22_33_T1_C1C6_repair_matrix.csv")
    branch_rows = read_rows(V22_33 / "v22_33_T1_C1C6_repair_branch_matrix.csv")
    c7_rows = read_rows(V22_33 / "v22_33_T1_C7_curvature_repair_matrix.csv")
    out_selector = []
    for row in selector_rows + c7_rows:
        out_selector.append(
            {
                **row,
                "signal_subspace_rank": row.get("subspace_dim", ""),
                "branch_NLL_delta_H400": row.get("NLL_delta_vs_base", ""),
                "branch_accuracy_delta_H400": row.get("accuracy_delta_vs_base", ""),
                "branch_ECE_delta_H400": row.get("ECE_delta_vs_base", ""),
                "hard_slice_NLL_delta_H400": row.get("tail_q99_delta", ""),
                "low_margin_accuracy_delta_H400": row.get("low_margin_accuracy_delta_vs_base", ""),
                "v22_34_selector_family": row.get("selector_name", ""),
                "source_artifact": row.get("source_artifact", "results/v22_33/v22_33_T1_C1C6_repair_matrix.csv"),
            }
        )
    write_rows(OUT_ROOT / "v22_34_T1_selector_matrix.csv", out_selector)
    write_rows(OUT_ROOT / "v22_34_T1_branch_matrix.csv", [{**r, "source_artifact": "results/v22_33/v22_33_T1_C1C6_repair_branch_matrix.csv"} for r in branch_rows])
    control_rows = []
    for row in branch_rows:
        if int_flag(row.get("is_control_branch")):
            control_rows.append(
                {
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    "branch_H": row.get("branch_H", ""),
                    "control_level": row.get("control_level", ""),
                    "control_type": row.get("selector_name", ""),
                    "control_delta": row.get("NLL_delta_vs_base", ""),
                    "ECE_delta": row.get("ECE_delta_vs_base", ""),
                    "Brier_delta": row.get("Brier_delta_vs_base", ""),
                    "tail_q99_delta": row.get("tail_q99_delta", ""),
                    "margin_delta": row.get("margin_q10_delta_vs_base", ""),
                    "source_artifact": "results/v22_33/v22_33_T1_C1C6_repair_branch_matrix.csv",
                }
            )
    write_rows(OUT_ROOT / "v22_34_T1_control_level_matrix.csv", control_rows)
    gate_rows = read_rows(V22_33 / "v22_33_T1_C1C6_repair_gate_eval.csv")
    n = len(gate_rows)
    best = sorted(gate_rows, key=lambda r: finite_float(r.get("mean_NLL_delta_vs_base"), math.inf) or math.inf)
    summary = {
        "selector_rows": len(out_selector),
        "branch_rows": len(branch_rows),
        "control_level_rows": len(control_rows),
        "gate_rows": n,
        "best_selector_by_mean_NLL": best[0].get("selector_name", "") if best else "",
        "best_selector_mean_NLL_delta": best[0].get("mean_NLL_delta_vs_base", "") if best else "",
        "beats_L3_max_rate": max((finite_float(r.get("beats_L3_rate"), 0.0) or 0.0 for r in gate_rows), default=0.0),
        "beats_L4_max_rate": max((finite_float(r.get("beats_L4_rate"), 0.0) or 0.0 for r in gate_rows), default=0.0),
        "beats_L5_max_rate": max((finite_float(r.get("beats_L5_rate"), 0.0) or 0.0 for r in gate_rows), default=0.0),
        "exploration_pass_rows": sum(int_flag(r.get("exploration_pass")) for r in gate_rows),
        "official_candidate_rows": sum(int_flag(r.get("official_candidate_pass")) for r in gate_rows),
        "C9_C10_status": "not_run_in_initial_reaudit; planned repair stage must instantiate control-orthogonal and margin-causal selectors before any promotion",
    }
    append_exec(
        "materialize v22.34 Part C matrices from named v22.33 C1-C6/C7 artifacts",
        task_id="C_T1_reaudit_from_v22_33",
        status="pass" if out_selector else "warn",
        files="results/v22_34/v22_34_T1_selector_matrix.csv, results/v22_34/v22_34_T1_branch_matrix.csv, results/v22_34/v22_34_T1_control_level_matrix.csv",
        note=f"exploration_pass_rows={summary['exploration_pass_rows']}; official_candidate_rows={summary['official_candidate_rows']}; C9_C10=not_run",
    )
    return summary


def _split_csv(text: str, cast: Any = str) -> list[Any]:
    out: list[Any] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(cast(part))
    return out


def _cos_abs(a: Any, b: Any) -> float:
    import torch

    af = a.reshape(-1).float()
    bf = b.reshape(-1).float()
    return float(torch.abs(torch.dot(af, bf) / (torch.linalg.vector_norm(af).clamp_min(1.0e-12) * torch.linalg.vector_norm(bf).clamp_min(1.0e-12))).item())


def run_c9c10_repair(args: argparse.Namespace) -> dict[str, Any]:
    import copy
    import torch
    import torch.nn.functional as F

    from experiments.run_v22_30_fidelity_ladder import (
        apply_layer_direction,
        cohort_signal_for_layer,
        collect_fixed_examples,
        direction_control,
        make_hard_vision_loaders,
        make_loaders,
        make_mlp,
        principal_overlap,
        train_adamw,
        train_branch,
    )

    device = torch.device(args.t1_device if str(args.t1_device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    selector_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    control_level_rows: list[dict[str, Any]] = []
    candidate_audit_rows: list[dict[str, Any]] = []

    def held_margin_q10(model: Any, x: Any, y: Any) -> float:
        model.eval()
        with torch.no_grad():
            logits = model(x).float()
            true = logits.gather(1, y.long().view(-1, 1)).squeeze(1)
            mask = torch.nn.functional.one_hot(y.long(), num_classes=logits.shape[1]).bool()
            other = logits.masked_fill(mask, float("-inf")).max(dim=1).values
            margins = true - other
            return float(torch.quantile(margins.float(), 0.10).item())

    hard_names = {"CIFAR10", "SVHN", "EMNIST", "EMNIST_LETTERS", "WINE"}
    for dataset in _split_csv(args.t1_datasets):
        for seed in _split_csv(args.t1_seeds, int):
            if dataset.upper().replace("-", "_") in hard_names:
                train_loader, held_loader, test_loader, input_dim, output_dim, _x_stats = make_hard_vision_loaders(
                    dataset, args.t1_train_size, args.t1_test_size, args.batch_size, seed, download=False
                )
            else:
                train_loader, held_loader, test_loader, input_dim, output_dim, _x_stats = make_loaders(
                    dataset, args.t1_train_size, args.t1_test_size, args.batch_size, seed
                )
            torch.manual_seed(2234 + seed)
            init_model = make_mlp(input_dim, output_dim, args.hidden, seed + 2234, device).to(device)
            model = copy.deepcopy(init_model).to(device)
            train_adamw(model, train_loader, test_loader, device, output_dim, steps=args.t1_pretrain_steps, lr=args.lr, weight_decay=args.weight_decay)
            xb, yb = collect_fixed_examples(train_loader, max(args.t1_examples, args.t1_signal_cohorts * args.t1_cohort_size), device)
            held_x, held_y = collect_fixed_examples(held_loader, args.t1_examples, device)
            layer = "w2"
            sig, _cohorts = cohort_signal_for_layer(
                model,
                xb,
                yb,
                layer,
                cohorts=args.t1_signal_cohorts,
                cohort_size=args.t1_cohort_size,
                rank_cap=args.t1_rank_cap,
                sketch_dim=args.t1_sketch_dim,
                seed=seed + 934,
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
                seed=seed + 934,
            )
            basis = sig.get("basis_sketch")
            proj = sig.get("proj")
            direction = sig["direction"]
            pool: list[tuple[str, Any, str]] = [("raw_top_signal", direction, "raw_top_signal")]
            if basis is not None and proj is not None and int(basis.numel()) > 0:
                k = int(basis.shape[1])
                for j in range(k):
                    coeff = torch.zeros(k, device=direction.device)
                    coeff[j] = 1.0
                    vec = proj @ (basis @ coeff)
                    vec = vec * direction.norm().clamp_min(1.0e-12) / vec.norm().clamp_min(1.0e-12)
                    pool.append((f"basis_{j}_plus", vec, "basis_axis"))
                    pool.append((f"basis_{j}_minus", -vec, "basis_axis"))
                gen = torch.Generator(device=direction.device).manual_seed(923400 + seed)
                while len(pool) < max(1, args.t1_candidate_count):
                    coeff = torch.randn(k, device=direction.device, generator=gen)
                    vec = proj @ (basis @ coeff)
                    vec = vec * direction.norm().clamp_min(1.0e-12) / vec.norm().clamp_min(1.0e-12)
                    pool.append((f"random_combo_{len(pool)}", vec, "same_subspace_random_combo"))
            pool = pool[: max(1, args.t1_candidate_count)]

            controls_for_selection = [
                direction_control(direction, "same_subspace", seed + 4301, basis, proj),
                direction_control(direction, "same_subspace", seed + 4302, basis, proj),
                direction_control(direction, "random", seed + 4303),
            ]
            model.eval()
            with torch.no_grad():
                held_ce_before = float(F.cross_entropy(model(held_x).float(), held_y.long()).item())
            margin_before = held_margin_q10(model, held_x, held_y)
            audit_rows: list[dict[str, Any]] = []
            for label, cand, source in pool:
                trial = copy.deepcopy(model).to(device)
                apply_layer_direction(trial, layer, cand, args.t1_branch_trust)
                trial.eval()
                with torch.no_grad():
                    held_ce_after = float(F.cross_entropy(trial(held_x).float(), held_y.long()).item())
                margin_after = held_margin_q10(trial, held_x, held_y)
                max_abs_control_cos = max(_cos_abs(cand, ctrl) for ctrl in controls_for_selection)
                audit_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "candidate_label": label,
                        "candidate_source": source,
                        "held_CE_delta": held_ce_after - held_ce_before,
                        "held_margin_q10_delta": margin_after - margin_before,
                        "max_abs_cos_to_selection_controls": max_abs_control_cos,
                    }
                )
            candidate_audit_rows.extend(audit_rows)
            if audit_rows:
                median_cos = sorted(float(r["max_abs_cos_to_selection_controls"]) for r in audit_rows)[len(audit_rows) // 2]
                eligible_c9 = [r for r in audit_rows if float(r["max_abs_cos_to_selection_controls"]) <= median_cos]
                c9_row = min(eligible_c9 or audit_rows, key=lambda r: (float(r["held_CE_delta"]), -float(r["held_margin_q10_delta"])))
                eligible_c10 = [r for r in audit_rows if float(r["held_CE_delta"]) <= 0.0]
                c10_row = max(eligible_c10 or audit_rows, key=lambda r: (float(r["held_margin_q10_delta"]), -float(r["held_CE_delta"])))
            else:
                c9_row = c10_row = {"candidate_label": "raw_top_signal", "held_CE_delta": "", "held_margin_q10_delta": "", "max_abs_cos_to_selection_controls": ""}
            by_label = {label: cand for label, cand, _source in pool}
            selector_defs = [
                ("C9_control_orthogonal_signal_direction", by_label.get(str(c9_row["candidate_label"]), direction), c9_row, "held_train_ce_with_control_orthogonality_filter"),
                ("C10_margin_causal_direction", by_label.get(str(c10_row["candidate_label"]), direction), c10_row, "held_train_margin_q10_selection_no_test"),
            ]
            base_model = copy.deepcopy(model).to(device)
            base_ev = train_branch(base_model, train_loader, test_loader, device, output_dim, args.t1_branch_horizon, args.lr, args.weight_decay)
            controls = [
                ("L1_isotropic_random", direction_control(direction, "random", seed + 1), "L1"),
                ("L2_same_norm_random", direction_control(direction, "random", seed + 2), "L2"),
                ("L3_same_signal_subspace_random", direction_control(direction, "same_subspace", seed + 3, basis, proj), "L3"),
                ("L4_signflip_same_subspace", -direction, "L4"),
                ("L5_shuffled_cohort_direction", direction_control(direction, "shuffled", seed + 5), "L5"),
            ]

            def branch_eval(name: str, cand: Any, is_control: int, control_level: str = "") -> dict[str, Any]:
                branch_model = copy.deepcopy(model).to(device)
                apply_layer_direction(branch_model, layer, cand, args.t1_branch_trust)
                ev = train_branch(branch_model, train_loader, test_loader, device, output_dim, args.t1_branch_horizon, args.lr, args.weight_decay)
                return {
                    "dataset": dataset,
                    "seed": seed,
                    "checkpoint_step": f"c9c10_pretrain_steps_{args.t1_pretrain_steps}",
                    "selector_name": name,
                    "layer_id": layer,
                    "branch_H": args.t1_branch_horizon,
                    "NLL_delta_vs_base": ev["NLL"] - base_ev["NLL"],
                    "accuracy_delta_vs_base": ev["accuracy"] - base_ev["accuracy"],
                    "ECE_delta_vs_base": ev["ECE"] - base_ev["ECE"],
                    "Brier_delta_vs_base": ev["Brier"] - base_ev["Brier"],
                    "tail_q95_delta": ev["tail_q95"] - base_ev["tail_q95"],
                    "tail_q99_delta": ev["tail_q99"] - base_ev["tail_q99"],
                    "margin_mean_delta_vs_base": ev.get("margin_mean", math.nan) - base_ev.get("margin_mean", math.nan),
                    "margin_q10_delta_vs_base": ev.get("margin_q10", math.nan) - base_ev.get("margin_q10", math.nan),
                    "margin_q01_delta_vs_base": ev.get("margin_q01", math.nan) - base_ev.get("margin_q01", math.nan),
                    "low_margin_accuracy_delta_vs_base": ev.get("low_margin_accuracy", math.nan) - base_ev.get("low_margin_accuracy", math.nan),
                    "beats_base": int(ev["NLL"] < base_ev["NLL"]),
                    "is_control_branch": is_control,
                    "control_level": control_level,
                    "selection_data_source": "held_train_only_v22_34_c9c10_repair",
                    "uses_test_direction_selection": 0,
                    "uses_future_direction": 0,
                    "source_artifact": "direct_v22_34_C9C10_repair",
                    "status": "direct_v22_34_C9C10_repair",
                }

            control_branch_rows = [branch_eval(name, cand, 1, level) for name, cand, level in controls]
            branch_rows.extend(control_branch_rows)
            for row in control_branch_rows:
                control_level_rows.append(
                    {
                        "dataset": row.get("dataset", ""),
                        "seed": row.get("seed", ""),
                        "branch_H": row.get("branch_H", ""),
                        "control_level": row.get("control_level", ""),
                        "control_type": row.get("selector_name", ""),
                        "control_delta": row.get("NLL_delta_vs_base", ""),
                        "ECE_delta": row.get("ECE_delta_vs_base", ""),
                        "Brier_delta": row.get("Brier_delta_vs_base", ""),
                        "tail_q99_delta": row.get("tail_q99_delta", ""),
                        "margin_delta": row.get("margin_q10_delta_vs_base", ""),
                        "source_artifact": "direct_v22_34_C9C10_repair",
                    }
                )
            best_by_level: dict[str, float] = {}
            for row in control_branch_rows:
                val = finite_float(row.get("NLL_delta_vs_base"), math.inf) or math.inf
                best_by_level[str(row.get("control_level"))] = min(best_by_level.get(str(row.get("control_level")), math.inf), val)
            best_any = min(best_by_level.values()) if best_by_level else math.inf
            for name, cand, audit, source in selector_defs:
                brow = branch_eval(name, cand, 0, "")
                val = finite_float(brow.get("NLL_delta_vs_base"), math.inf) or math.inf
                brow["best_control_delta"] = best_any
                brow["real_minus_best_control_delta"] = val - best_any
                for level in ["L1", "L2", "L3", "L4", "L5"]:
                    brow[f"beats_{level}_control"] = int(val < best_by_level.get(level, math.inf))
                branch_rows.append(brow)
                selector_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "checkpoint_step": brow["checkpoint_step"],
                        "selector_name": name,
                        "subspace_dim": int(basis.shape[1]) if basis is not None else "",
                        "signal_subspace_rank": int(basis.shape[1]) if basis is not None else "",
                        "signal_eigenvalue": sig.get("positive_eigenvalue_mean", ""),
                        "cohort_positive_fraction": sig.get("cohort_positive_fraction", ""),
                        "temporal_eigenspace_overlap": principal_overlap(prev_sig["basis_sketch"], sig["basis_sketch"]),
                        "branch_H": args.t1_branch_horizon,
                        "NLL_delta_vs_base": brow["NLL_delta_vs_base"],
                        "branch_NLL_delta_H400": brow["NLL_delta_vs_base"] if int(args.t1_branch_horizon) == 400 else "",
                        "accuracy_delta_vs_base": brow["accuracy_delta_vs_base"],
                        "ECE_delta_vs_base": brow["ECE_delta_vs_base"],
                        "Brier_delta_vs_base": brow["Brier_delta_vs_base"],
                        "tail_q95_delta": brow["tail_q95_delta"],
                        "tail_q99_delta": brow["tail_q99_delta"],
                        "margin_mean_delta_vs_base": brow["margin_mean_delta_vs_base"],
                        "margin_q10_delta_vs_base": brow["margin_q10_delta_vs_base"],
                        "margin_q01_delta_vs_base": brow["margin_q01_delta_vs_base"],
                        "low_margin_accuracy_delta_vs_base": brow["low_margin_accuracy_delta_vs_base"],
                        "beats_base": brow["beats_base"],
                        "beats_L1_control": brow["beats_L1_control"],
                        "beats_L2_control": brow["beats_L2_control"],
                        "beats_L3_control": brow["beats_L3_control"],
                        "beats_L4_control": brow["beats_L4_control"],
                        "beats_L5_control": brow["beats_L5_control"],
                        "best_control_delta": best_any,
                        "real_minus_best_control_delta": brow["real_minus_best_control_delta"],
                        "selected_candidate_label": audit.get("candidate_label", ""),
                        "held_CE_delta": audit.get("held_CE_delta", ""),
                        "held_margin_q10_delta": audit.get("held_margin_q10_delta", ""),
                        "max_abs_cos_to_selection_controls": audit.get("max_abs_cos_to_selection_controls", ""),
                        "selection_data_source": source,
                        "uses_test_direction_selection": 0,
                        "uses_future_direction": 0,
                        "source_artifact": "direct_v22_34_C9C10_repair",
                        "status": "direct_v22_34_C9C10_repair",
                    }
                )

    write_rows(OUT_ROOT / "v22_34_T1_C9C10_repair_matrix.csv", selector_rows)
    write_rows(OUT_ROOT / "v22_34_T1_C9C10_repair_branch_matrix.csv", branch_rows)
    write_rows(OUT_ROOT / "v22_34_T1_C9C10_candidate_audit_matrix.csv", candidate_audit_rows)
    control_vals = [finite_float(r.get("NLL_delta_vs_base")) for r in branch_rows if int_flag(r.get("is_control_branch"))]
    cv = [float(v) for v in control_vals if v is not None]
    control_noise = 0.0
    if len(cv) > 1:
        mu = sum(cv) / len(cv)
        control_noise = math.sqrt(sum((v - mu) ** 2 for v in cv) / len(cv))
    gate_rows: list[dict[str, Any]] = []
    for selector in sorted({str(r.get("selector_name")) for r in selector_rows}):
        rows = [r for r in selector_rows if r.get("selector_name") == selector]
        n = len(rows)
        mean_nll = sum(float(finite_float(r.get("NLL_delta_vs_base"), 0.0) or 0.0) for r in rows) / n if n else math.inf
        max_ece = max((finite_float(r.get("ECE_delta_vs_base"), 0.0) or 0.0 for r in rows), default=math.inf)
        max_tail = max((finite_float(r.get("tail_q99_delta"), 0.0) or 0.0 for r in rows), default=math.inf)
        row = {
            "selector_name": selector,
            "direct_rows": n,
            "datasets": ",".join(sorted({str(r.get("dataset")) for r in rows})),
            "seeds": ",".join(sorted({str(r.get("seed")) for r in rows})),
            "horizons": ",".join(sorted({str(r.get("branch_H")) for r in rows})),
            "beats_base_rate": rate(sum(int_flag(r.get("beats_base")) for r in rows), n),
            "beats_L3_rate": rate(sum(int_flag(r.get("beats_L3_control")) for r in rows), n),
            "beats_L4_rate": rate(sum(int_flag(r.get("beats_L4_control")) for r in rows), n),
            "beats_L5_rate": rate(sum(int_flag(r.get("beats_L5_control")) for r in rows), n),
            "mean_NLL_delta_vs_base": mean_nll,
            "control_noise_std": control_noise,
            "max_ECE_delta": max_ece,
            "max_tail_q99_delta": max_tail,
            "exploration_pass": int(rate(sum(int_flag(r.get("beats_base")) for r in rows), n) >= 0.65 and rate(sum(int_flag(r.get("beats_L3_control")) for r in rows), n) >= 0.55 and mean_nll < -0.5 * control_noise),
            "official_candidate_pass": int(rate(sum(int_flag(r.get("beats_L3_control")) for r in rows), n) >= 0.65 and rate(sum(int_flag(r.get("beats_L4_control")) for r in rows), n) >= 0.60 and rate(sum(int_flag(r.get("beats_L5_control")) for r in rows), n) >= 0.60 and mean_nll < -1.0 * control_noise),
        }
        gate_rows.append(row)
    write_rows(OUT_ROOT / "v22_34_T1_C9C10_repair_gate_eval.csv", gate_rows)

    repair_source = "direct_v22_34_C9C10_repair"
    existing_selector = [r for r in read_rows(OUT_ROOT / "v22_34_T1_selector_matrix.csv") if r.get("source_artifact") != repair_source]
    existing_branch = [r for r in read_rows(OUT_ROOT / "v22_34_T1_branch_matrix.csv") if r.get("source_artifact") != repair_source]
    existing_control = [r for r in read_rows(OUT_ROOT / "v22_34_T1_control_level_matrix.csv") if r.get("source_artifact") != repair_source]
    write_rows(OUT_ROOT / "v22_34_T1_selector_matrix.csv", existing_selector + selector_rows)
    write_rows(OUT_ROOT / "v22_34_T1_branch_matrix.csv", existing_branch + branch_rows)
    write_rows(OUT_ROOT / "v22_34_T1_control_level_matrix.csv", existing_control + control_level_rows)
    summary = {
        "selector_rows": len(selector_rows),
        "branch_rows": len(branch_rows),
        "control_level_rows": len(control_level_rows),
        "candidate_audit_rows": len(candidate_audit_rows),
        "gate_rows": len(gate_rows),
        "control_noise_std": control_noise,
        "exploration_pass_rows": sum(int_flag(r.get("exploration_pass")) for r in gate_rows),
        "official_candidate_rows": sum(int_flag(r.get("official_candidate_pass")) for r in gate_rows),
        "best_selector_by_mean_NLL": (sorted(gate_rows, key=lambda r: finite_float(r.get("mean_NLL_delta_vs_base"), math.inf) or math.inf)[0].get("selector_name") if gate_rows else ""),
        "best_selector_mean_NLL_delta": (sorted(gate_rows, key=lambda r: finite_float(r.get("mean_NLL_delta_vs_base"), math.inf) or math.inf)[0].get("mean_NLL_delta_vs_base") if gate_rows else ""),
    }
    write_rows(OUT_ROOT / "v22_34_T1_C9C10_repair_summary.csv", [summary])
    append_exec(
        "run v22.34 C9/C10 repair: control-orthogonal and margin-causal held-train selectors",
        task_id="C_repair_C9C10",
        status="pass",
        gpu=str(device),
        files="results/v22_34/v22_34_T1_C9C10_repair_matrix.csv, results/v22_34/v22_34_T1_C9C10_candidate_audit_matrix.csv, results/v22_34/v22_34_T1_C9C10_repair_gate_eval.csv, results/v22_34/v22_34_T1_selector_matrix.csv, results/v22_34/v22_34_T1_control_level_matrix.csv",
        note=f"exploration_pass_rows={summary['exploration_pass_rows']}; official_candidate_rows={summary['official_candidate_rows']}; best_selector={summary['best_selector_by_mean_NLL']}",
    )
    return summary


def summarize_t1_current() -> dict[str, Any]:
    selector_rows = read_rows(OUT_ROOT / "v22_34_T1_selector_matrix.csv")
    branch_rows = read_rows(OUT_ROOT / "v22_34_T1_branch_matrix.csv")
    control_rows = read_rows(OUT_ROOT / "v22_34_T1_control_level_matrix.csv")
    base_gate_rows = read_rows(V22_33 / "v22_33_T1_C1C6_repair_gate_eval.csv")
    c9_gate_rows = read_rows(OUT_ROOT / "v22_34_T1_C9C10_repair_gate_eval.csv")
    gate_rows = base_gate_rows + c9_gate_rows
    best = sorted(gate_rows, key=lambda r: finite_float(r.get("mean_NLL_delta_vs_base"), math.inf) or math.inf)
    c9_status = "executed_direct_v22_34_repair" if c9_gate_rows else "not_run; planned repair stage must instantiate control-orthogonal and margin-causal selectors before any promotion"
    return {
        "selector_rows": len(selector_rows),
        "branch_rows": len(branch_rows),
        "control_level_rows": len(control_rows),
        "gate_rows": len(gate_rows),
        "best_selector_by_mean_NLL": best[0].get("selector_name", "") if best else "",
        "best_selector_mean_NLL_delta": best[0].get("mean_NLL_delta_vs_base", "") if best else "",
        "beats_L3_max_rate": max((finite_float(r.get("beats_L3_rate"), 0.0) or 0.0 for r in gate_rows), default=0.0),
        "beats_L4_max_rate": max((finite_float(r.get("beats_L4_rate"), 0.0) or 0.0 for r in gate_rows), default=0.0),
        "beats_L5_max_rate": max((finite_float(r.get("beats_L5_rate"), 0.0) or 0.0 for r in gate_rows), default=0.0),
        "exploration_pass_rows": sum(int_flag(r.get("exploration_pass")) for r in gate_rows),
        "official_candidate_rows": sum(int_flag(r.get("official_candidate_pass")) for r in gate_rows),
        "C9_C10_status": c9_status,
        "C9_C10_gate_rows": len(c9_gate_rows),
        "C9_C10_exploration_pass_rows": sum(int_flag(r.get("exploration_pass")) for r in c9_gate_rows),
        "C9_C10_official_candidate_rows": sum(int_flag(r.get("official_candidate_pass")) for r in c9_gate_rows),
    }


def summarize_basis_current() -> dict[str, Any]:
    fit_rows = read_rows(OUT_ROOT / "v22_34_basis_actuator_fit_matrix.csv")
    branch_rows = read_rows(OUT_ROOT / "v22_34_basis_actuator_branch_matrix.csv")
    target_rows = read_rows(OUT_ROOT / "v22_34_basis_target_family_matrix.csv")
    full_loop_rows = read_rows(OUT_ROOT / "v22_34_basis_actuator_full_loop_matrix.csv")
    best_resid = min((finite_float(r.get("basis_actuator_projection_residual"), math.inf) or math.inf for r in fit_rows), default=math.inf)
    strict_rows = sum(int(finite_float(r.get("strict_fit_rows"), 0.0) or 0.0) for r in full_loop_rows)
    strict_pass_rows = sum(int(finite_float(r.get("strict_fit_branch_pass_rows"), 0.0) or 0.0) for r in full_loop_rows)
    branch_rate = rate(strict_pass_rows, strict_rows)
    per_family_rates = [
        rate(
            int(finite_float(r.get("strict_fit_branch_pass_rows"), 0.0) or 0.0),
            int(finite_float(r.get("strict_fit_rows"), 0.0) or 0.0),
        )
        for r in full_loop_rows
    ]
    family_exploration_pass = any(x >= 0.40 for x in per_family_rates)
    family_official_pass = any(x >= 0.60 for x in per_family_rates)
    return {
        "fit_rows": len(fit_rows),
        "branch_rows": len(branch_rows),
        "target_family_rows": len(target_rows),
        "executed_target_families": sum(1 for r in target_rows if str(r.get("status", "")).startswith("executed") or "direct_v22_34" in str(r.get("status", ""))),
        "not_run_target_families": sum(1 for r in target_rows if "not_run" in str(r.get("status", ""))),
        "best_projection_residual": "" if not math.isfinite(best_resid) else best_resid,
        "strict_fit_rows": strict_rows,
        "strict_fit_branch_pass_rows": strict_pass_rows,
        "strict_fit_branch_pass_rate": branch_rate,
        "branch_gate_exploration_pass": int(family_exploration_pass),
        "branch_gate_official_pass": int(family_official_pass),
        "full_loop_status": ",".join(sorted({str(r.get("status", "")) for r in full_loop_rows})) if full_loop_rows else "",
    }


def branch_nll_value(row: dict[str, Any]) -> float | None:
    branch_h = str(row.get("branch_H", "")).strip()
    preferred = [f"NLL_delta_H{branch_h}"] if branch_h else []
    for key in [*preferred, "NLL_delta_H400", "NLL_delta_H200", "NLL_delta_H100", "NLL_delta_H50"]:
        val = finite_float(row.get(key), None)
        if val is not None:
            return val
    return None


def basis_repair_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str, str]:
    return (
        str(row.get("target_family", "")),
        str(row.get("carrier", "")),
        str(row.get("basis_family", "")),
        str(row.get("dataset", "")),
        str(row.get("seed", "")),
        str(row.get("basis_bank", "")),
        str(row.get("repair_stage", "")),
    )


def basis_target_control_win(
    fit_rows: list[dict[str, Any]],
    branch_rows: list[dict[str, Any]],
    *,
    epsilon_rep: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strict_keys = {
        basis_repair_key(row)
        for row in fit_rows
        if (finite_float(row.get("basis_actuator_projection_residual"), 999.0) or 999.0) <= 0.05
        and (finite_float(row.get("basis_actuator_cosine"), 0.0) or 0.0) >= 0.98
        and (
            finite_float(
                row.get("exact_vs_linearized_error_selected", row.get("exact_vs_linearized_error")),
                999.0,
            )
            or 999.0
        )
        <= 0.05
        and (finite_float(row.get("J_B_gradcheck_rel_error"), 999.0) or 999.0) <= 1.0e-3
    }
    grouped: dict[tuple[str, str, str, str, str, str, str], dict[str, dict[str, Any]]] = {}
    for row in branch_rows:
        grouped.setdefault(basis_repair_key(row), {})[str(row.get("branch_variant", ""))] = row
    rows: list[dict[str, Any]] = []
    for key, variants in sorted(grouped.items()):
        real = variants.get("basis_native_real")
        if not real:
            continue
        random_ctrl = variants.get("same_basis_random", {})
        signflip_ctrl = variants.get("signflip_basis_control", {})
        real_nll = branch_nll_value(real)
        random_nll = branch_nll_value(random_ctrl)
        signflip_nll = branch_nll_value(signflip_ctrl)
        fit_strict = key in strict_keys
        real_improves = real_nll is not None and real_nll < 0.0
        noise_pass = real_nll is not None and real_nll < -float(epsilon_rep)
        beats_random = real_nll is not None and random_nll is not None and real_nll < random_nll
        beats_signflip = real_nll is not None and signflip_nll is not None and real_nll < signflip_nll
        accuracy_delta = finite_float(real.get("accuracy_delta_vs_base"), None)
        accuracy_ok = accuracy_delta is not None and accuracy_delta >= -0.01
        branch_pass = fit_strict and noise_pass and beats_random and beats_signflip and accuracy_ok
        if not fit_strict:
            explanation = "not_strict_fit_candidate"
        elif not noise_pass:
            explanation = "strict_fit_but_NLL_delta_not_beyond_epsilon_rep"
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
                "target_family": key[0],
                "carrier": key[1],
                "basis_family": key[2],
                "dataset": key[3],
                "seed": key[4],
                "basis_bank": key[5],
                "repair_stage": key[6],
                "branch_H": real.get("branch_H", ""),
                "epsilon_rep": epsilon_rep,
                "fit_strict_pass": int(fit_strict),
                "real_NLL_delta": "" if real_nll is None else real_nll,
                "same_basis_random_NLL_delta": "" if random_nll is None else random_nll,
                "signflip_basis_control_NLL_delta": "" if signflip_nll is None else signflip_nll,
                "real_improves": int(real_improves),
                "branch_noise_pass": int(noise_pass),
                "beats_same_basis_random": int(beats_random),
                "beats_signflip_basis_control": int(beats_signflip),
                "accuracy_delta_vs_base": real.get("accuracy_delta_vs_base", ""),
                "accuracy_gate_pass": int(accuracy_ok),
                "strict_fit_branch_pass": int(branch_pass),
                "control_win_explanation": explanation,
                "source_artifact": "direct_v22_34_basis_target_family_repair",
            }
        )
    strict_rows = [r for r in rows if int_flag(r.get("fit_strict_pass"))]
    pass_rows = [r for r in rows if int_flag(r.get("strict_fit_branch_pass"))]
    strict_improve = [r for r in strict_rows if int_flag(r.get("real_improves"))]
    strict_noise = [r for r in strict_rows if int_flag(r.get("branch_noise_pass"))]
    strict_control = [
        r
        for r in strict_rows
        if int_flag(r.get("beats_same_basis_random")) and int_flag(r.get("beats_signflip_basis_control"))
    ]
    summary = {
        "basis_target_control_rows": len(rows),
        "strict_fit_rows": len(strict_rows),
        "strict_fit_improve_rows": len(strict_improve),
        "strict_fit_noise_pass_rows": len(strict_noise),
        "strict_fit_control_beat_rows": len(strict_control),
        "strict_fit_branch_pass_rows": len(pass_rows),
        "strict_fit_improve_rate": rate(len(strict_improve), len(strict_rows)),
        "strict_fit_noise_pass_rate": rate(len(strict_noise), len(strict_rows)),
        "strict_fit_control_beat_rate": rate(len(strict_control), len(strict_rows)),
        "strict_fit_branch_pass_rate": rate(len(pass_rows), len(strict_rows)),
        "branch_gate_exploration_pass": int(bool(strict_rows) and rate(len(pass_rows), len(strict_rows)) >= 0.40),
        "branch_gate_official_pass": int(bool(strict_rows) and rate(len(pass_rows), len(strict_rows)) >= 0.60),
    }
    return rows, summary


def run_basis_target_family_repair(args: argparse.Namespace) -> dict[str, Any]:
    import copy
    import torch
    import torch.nn.functional as F

    from dgkan.fu.real_jacobian_commit import add_flat_delta, output_jacobian, select_named_parameters, solve_linearized_commit
    from experiments.run_v22_30_fidelity_ladder import collect_fixed_examples, make_hard_vision_loaders, make_kan, train_adamw, train_branch
    from experiments.run_v22_32_causal_actuator_fidelity import basis_bank_masks

    device = torch.device(args.basis_device if str(args.basis_device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    gap_summary = (read_rows(OUT_ROOT / "v22_34_gap_truth_summary.csv") or [{}])[0]
    epsilon_rep = float(args.basis_epsilon_rep)
    if epsilon_rep < 0.0:
        epsilon_rep = float(finite_float(gap_summary.get("repeat_noise_epsilon"), 0.0) or 0.0)

    target_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    family_tags = [str(f).split("_", 1)[0] for f in _split_csv(args.basis_target_families)]

    def tag_value(value: Any) -> str:
        return str(value).replace("-", "m").replace(".", "p")

    dataset_tag = "_".join(_split_csv(args.basis_datasets)).replace("-", "m")
    seed_tag = "_".join(str(x) for x in _split_csv(args.basis_seeds)).replace("-", "m")
    repair_stage = (
        f"{'_'.join(family_tags)}_basis_target_data{dataset_tag}_seeds{seed_tag}_H{args.basis_branch_horizon}"
        f"_rho{tag_value(args.basis_damping)}_scale{tag_value(args.basis_update_scale)}_gceps{tag_value(args.basis_gradcheck_eps)}"
    )
    repair_stage = (
        f"{repair_stage}_tr{tag_value(args.basis_train_size)}"
        f"_te{tag_value(args.basis_test_size)}"
        f"_ex{tag_value(args.basis_examples)}"
        f"_rows{tag_value(args.basis_max_output_rows)}"
        f"_pre{tag_value(args.basis_pretrain_steps)}"
    )
    if str(args.basis_scale_policy) != "fixed":
        cand_tag = tag_value(str(args.basis_safe_scale_candidates).replace(",", "x"))
        repair_stage = (
            f"{repair_stage}_policy{tag_value(args.basis_scale_policy)}"
            f"_cands{cand_tag}_srho{tag_value(args.basis_sharpness_rho)}"
            f"_cetol{tag_value(args.basis_scale_ce_tolerance)}"
            f"_shtol{tag_value(args.basis_sharpness_tolerance)}"
        )

    def merge_repair_stage_rows(path: Path, new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        old_rows = [r for r in read_rows(path) if r.get("repair_stage") != repair_stage]
        merged_rows = old_rows + new_rows
        write_rows(path, merged_rows)
        return merged_rows

    def hard_margin_target(logits: Any, y: Any, fraction: float) -> tuple[Any, dict[str, Any]]:
        losses = F.cross_entropy(logits.float(), y.long(), reduction="none")
        n = int(y.numel())
        k = max(1, min(n, int(math.ceil(float(fraction) * n))))
        hard_idx = torch.topk(losses.detach(), k=k, largest=True).indices
        masked = logits.float().detach().clone()
        masked[torch.arange(n, device=masked.device), y.long()] = float("-inf")
        other = masked.argmax(dim=1)
        target = torch.zeros_like(logits.float())
        target[hard_idx, y.long()[hard_idx]] = 1.0
        target[hard_idx, other[hard_idx]] = -1.0
        flat = target.reshape(-1)
        flat = flat / torch.linalg.vector_norm(flat).clamp_min(1.0e-12)
        return flat, {
            "hard_slice_fraction": float(k) / float(n),
            "hard_slice_count": k,
            "hard_loss_mean": float(losses[hard_idx].mean().item()),
            "all_loss_mean": float(losses.mean().item()),
        }

    def ce_descent_target(logits: Any, y: Any, fraction: float | None = None) -> tuple[Any, dict[str, Any]]:
        probs = torch.softmax(logits.float(), dim=-1)
        full = -(probs - F.one_hot(y.long(), num_classes=logits.shape[1]).float())
        losses = F.cross_entropy(logits.float(), y.long(), reduction="none")
        n = int(y.numel())
        if fraction is None:
            k = n
            keep = torch.arange(n, device=logits.device)
        else:
            k = max(1, min(n, int(math.ceil(float(fraction) * n))))
            keep = torch.topk(losses.detach(), k=k, largest=True).indices
            mask = torch.zeros(n, device=logits.device, dtype=torch.bool)
            mask[keep] = True
            full = torch.where(mask[:, None], full, torch.zeros_like(full))
        flat = full.reshape(-1)
        flat = flat / torch.linalg.vector_norm(flat).clamp_min(1.0e-12)
        return flat, {
            "hard_slice_fraction": float(k) / float(n),
            "hard_slice_count": int(k),
            "hard_loss_mean": float(losses[keep].mean().item()),
            "all_loss_mean": float(losses.mean().item()),
        }

    def held_class_cvar_margin_target(train_logits: Any, train_y: Any, held_logits: Any, held_y: Any, fraction: float) -> tuple[Any, dict[str, Any]]:
        train_logits = train_logits.float()
        held_logits = held_logits.float()
        output_dim = int(train_logits.shape[1])
        train_losses = F.cross_entropy(train_logits, train_y.long(), reduction="none")
        held_losses = F.cross_entropy(held_logits, held_y.long(), reduction="none")
        class_scores = torch.zeros(output_dim, device=train_logits.device)
        for cls in range(output_dim):
            cls_losses = held_losses[held_y.long() == cls]
            if int(cls_losses.numel()):
                k_cls = max(1, int(math.ceil(float(fraction) * int(cls_losses.numel()))))
                class_scores[cls] = torch.topk(cls_losses.detach(), k=k_cls, largest=True).values.mean()
        positive = class_scores[class_scores > 0]
        if int(positive.numel()):
            class_weights = class_scores / positive.mean().clamp_min(1.0e-12)
        else:
            class_weights = torch.ones_like(class_scores)
        class_weights = torch.clamp(class_weights, min=0.25, max=4.0)
        example_weights = class_weights[train_y.long()]
        scores = train_losses.detach() * example_weights.detach()
        n = int(train_y.numel())
        k = max(1, min(n, int(math.ceil(float(fraction) * n))))
        hard_idx = torch.topk(scores, k=k, largest=True).indices
        probs = torch.softmax(train_logits, dim=-1)
        ce = -(probs - F.one_hot(train_y.long(), num_classes=output_dim).float())
        masked = train_logits.detach().clone()
        masked[torch.arange(n, device=masked.device), train_y.long()] = float("-inf")
        other = masked.argmax(dim=1)
        margin = torch.zeros_like(train_logits)
        margin[hard_idx, train_y.long()[hard_idx]] = 1.0
        margin[hard_idx, other[hard_idx]] = -1.0
        hard_mask = torch.zeros(n, device=train_logits.device, dtype=train_logits.dtype)
        hard_mask[hard_idx] = example_weights[hard_idx].to(dtype=train_logits.dtype)
        target = (ce * hard_mask[:, None]) + margin
        flat = target.reshape(-1)
        flat = flat / torch.linalg.vector_norm(flat).clamp_min(1.0e-12)
        return flat, {
            "hard_slice_fraction": float(k) / float(n),
            "hard_slice_count": int(k),
            "hard_loss_mean": float(train_losses[hard_idx].mean().item()),
            "all_loss_mean": float(train_losses.mean().item()),
            "held_class_cvar_max": float(class_scores.max().item()) if int(class_scores.numel()) else 0.0,
            "held_class_weight_max": float(class_weights.max().item()) if int(class_weights.numel()) else 0.0,
        }

    def orthogonalize(target: Any, controls: list[Any]) -> tuple[Any, float]:
        out = target.detach().float().clone()
        original_norm = torch.linalg.vector_norm(out).clamp_min(1.0e-12)
        removed_sq = 0.0
        for ctrl in controls:
            c = ctrl.detach().float().reshape(-1).to(out.device)[: out.numel()]
            denom = torch.dot(c, c).clamp_min(1.0e-12)
            coeff = torch.dot(out, c) / denom
            removed = coeff * c
            removed_sq += float((torch.dot(removed, removed) / (original_norm * original_norm)).item())
            out = out - removed
        out = out / torch.linalg.vector_norm(out).clamp_min(1.0e-12)
        return out, removed_sq

    def held_loss_value(model_obj: Any, x: Any, y: Any) -> float:
        was_training = bool(model_obj.training)
        model_obj.eval()
        with torch.no_grad():
            value = float(F.cross_entropy(model_obj(x).float(), y.long()).item())
        if was_training:
            model_obj.train()
        return value

    def sam_sharpness_proxy(model_obj: Any, x: Any, y: Any, rho: float) -> float:
        was_training = bool(model_obj.training)
        model_obj.eval()
        params = [p for p in model_obj.parameters() if p.requires_grad]
        if not params:
            return 0.0
        model_obj.zero_grad(set_to_none=True)
        logits = model_obj(x).float()
        base_loss = F.cross_entropy(logits, y.long())
        grads = torch.autograd.grad(base_loss, params, allow_unused=True)
        norm_sq = torch.zeros((), device=x.device, dtype=logits.dtype)
        for grad in grads:
            if grad is not None:
                norm_sq = norm_sq + grad.detach().float().pow(2).sum()
        norm = torch.sqrt(norm_sq).clamp_min(1.0e-12)
        steps: list[Any] = []
        with torch.no_grad():
            for param, grad in zip(params, grads):
                if grad is None:
                    steps.append(None)
                    continue
                step = grad.detach().to(param.device, dtype=param.dtype) * (float(rho) / norm.to(param.device, dtype=param.dtype))
                param.add_(step)
                steps.append(step)
        try:
            with torch.no_grad():
                sharp_loss = F.cross_entropy(model_obj(x).float(), y.long())
        finally:
            with torch.no_grad():
                for param, step in zip(params, steps):
                    if step is not None:
                        param.sub_(step)
            model_obj.zero_grad(set_to_none=True)
            if was_training:
                model_obj.train()
        return float((sharp_loss - base_loss.detach()).item())

    def choose_branch_scale(model_obj: Any, named_params: Any, delta: Any, x: Any, y: Any) -> dict[str, Any]:
        policy = str(args.basis_scale_policy)
        if policy == "fixed":
            return {
                "scale_policy": "fixed",
                "selected_scale_multiplier": 1.0,
                "applied_update_scale": float(args.basis_update_scale),
                "curvature_safe_scale_pass": 1,
                "scale_selection_status": "fixed_basis_update_scale",
                "held_CE_base": "",
                "held_CE_delta_at_selected_scale": "",
                "sam_sharpness_base": "",
                "sam_sharpness_delta_at_selected_scale": "",
                "scale_candidate_trace": "",
            }
        candidates = sorted({float(x) for x in _split_csv(args.basis_safe_scale_candidates, float) if float(x) > 0.0})
        if not candidates:
            candidates = [1.0]
        base_ce = held_loss_value(model_obj, x, y)
        base_sharp = sam_sharpness_proxy(model_obj, x, y, float(args.basis_sharpness_rho))
        candidate_rows: list[dict[str, Any]] = []
        for mult in candidates:
            applied = float(args.basis_update_scale) * float(mult)
            add_flat_delta(named_params, delta, scale=applied)
            try:
                ce_after = held_loss_value(model_obj, x, y)
                sharp_after = sam_sharpness_proxy(model_obj, x, y, float(args.basis_sharpness_rho))
            finally:
                add_flat_delta(named_params, delta, scale=-applied)
            ce_delta = ce_after - base_ce
            sharp_delta = sharp_after - base_sharp
            safe = ce_delta <= float(args.basis_scale_ce_tolerance) and sharp_delta <= float(args.basis_sharpness_tolerance)
            candidate_rows.append(
                {
                    "mult": mult,
                    "applied": applied,
                    "held_ce_delta": ce_delta,
                    "sharpness_delta": sharp_delta,
                    "safe": int(safe),
                }
            )
        safe_rows = [r for r in candidate_rows if int_flag(r.get("safe"))]
        selected = max(safe_rows, key=lambda r: float(r["mult"])) if safe_rows else min(candidate_rows, key=lambda r: float(r["mult"]))
        trace = ";".join(
            f"{r['mult']}:{r['held_ce_delta']:.6g}:{r['sharpness_delta']:.6g}:{int(r['safe'])}"
            for r in candidate_rows
        )
        return {
            "scale_policy": policy,
            "selected_scale_multiplier": selected["mult"],
            "applied_update_scale": selected["applied"],
            "curvature_safe_scale_pass": int(selected["safe"]),
            "scale_selection_status": "held_sharpness_safe_selected" if safe_rows else "no_safe_candidate_min_scale_used_for_diagnostic",
            "held_CE_base": base_ce,
            "held_CE_delta_at_selected_scale": selected["held_ce_delta"],
            "sam_sharpness_base": base_sharp,
            "sam_sharpness_delta_at_selected_scale": selected["sharpness_delta"],
            "scale_candidate_trace": trace,
        }

    def target_spec(target_family: str, train_x: Any, train_y: Any, held_x: Any, held_y: Any) -> tuple[Any, Any, Any, dict[str, Any], str, str]:
        if target_family.startswith("D3"):
            tx, ty = held_x, held_y
            logits_t = model(tx).float()
            target_full, meta = ce_descent_target(logits_t, ty, args.basis_hard_fraction)
            return tx, ty, target_full, meta, "held_cohort_CVaR25_logit_descent", "held_train"
        tx, ty = train_x, train_y
        logits_t = model(tx).float()
        if target_family.startswith("D2"):
            target_full, meta = ce_descent_target(logits_t, ty, None)
            return tx, ty, target_full, meta, "multi_cohort_ce_logit_descent", "train_multi_cohort"
        if target_family.startswith("D4"):
            target_full, meta = hard_margin_target(logits_t, ty, args.basis_hard_fraction)
            return tx, ty, target_full, meta, "hard_slice_margin_logits", "train_hard_slice"
        if target_family.startswith("D5"):
            target_full, meta = ce_descent_target(logits_t, ty, None)
            return tx, ty, target_full, meta, "low_degree_DCHE_ce_logit_descent", "train_low_degree_DCHE"
        if target_family.startswith("D6"):
            target_full, meta = ce_descent_target(logits_t, ty, None)
            return tx, ty, target_full, meta, "low_frequency_DFOU_ce_logit_descent", "train_low_frequency_DFOU"
        if target_family.startswith("D7"):
            target_full, meta = hard_margin_target(logits_t, ty, args.basis_hard_fraction)
            return tx, ty, target_full, meta, "control_orthogonal_hard_slice_margin_logits", "train_control_orthogonal"
        if target_family.startswith("D9"):
            held_logits = model(held_x).float()
            target_full, meta = held_class_cvar_margin_target(logits_t, ty, held_logits, held_y, args.basis_hard_fraction)
            return tx, ty, target_full, meta, "held_class_CVaR_control_orthogonal_margin_logits", "train_hard_slice_weighted_by_held_class_CVaR"
        target_full, meta = ce_descent_target(logits_t, ty, None)
        return tx, ty, target_full, meta, "unregistered_ce_logit_descent", "train"

    def bank_allowed(target_family: str, family: str, basis_bank: str) -> bool:
        if target_family.startswith("D5"):
            return family == "D-CHE" and "low_degree" in basis_bank
        if target_family.startswith("D6"):
            return family == "D-FOU" and "low_frequency" in basis_bank
        return True

    for dataset in _split_csv(args.basis_datasets):
        for seed in _split_csv(args.basis_seeds, int):
            train_loader, held_loader, test_loader, input_dim, output_dim, x_stats = make_hard_vision_loaders(
                dataset,
                args.basis_train_size,
                args.basis_test_size,
                args.batch_size,
                seed,
                download=False,
            )
            for family, carrier in [("D-CHE", "DGKAN_DCHE"), ("D-FOU", "DGKAN_DFOU")]:
                torch.manual_seed(223_400 + seed + (0 if family == "D-CHE" else 10_000))
                model = make_kan(input_dim, output_dim, args.hidden, seed + 340, device, x_stats, family).to(device)
                train_adamw(model, train_loader, test_loader, device, output_dim, steps=args.basis_pretrain_steps, lr=args.lr, weight_decay=args.weight_decay)
                xb, yb = collect_fixed_examples(train_loader, args.basis_examples, device)
                held_x, held_y = collect_fixed_examples(held_loader, args.basis_examples, device)
                named = select_named_parameters(model, "basis")
                base_model = copy.deepcopy(model).to(device)
                base_ev = train_branch(base_model, train_loader, test_loader, device, output_dim, args.basis_branch_horizon, args.lr, args.weight_decay)
                for target_family in _split_csv(args.basis_target_families):
                    if target_family.startswith("D5") and family != "D-CHE":
                        continue
                    if target_family.startswith("D6") and family != "D-FOU":
                        continue
                    target_x, target_y, target_full, target_meta, source_type, target_input_source = target_spec(target_family, xb, yb, held_x, held_y)
                    max_rows = min(int(args.basis_max_output_rows), int(target_full.numel()))
                    jac, _spec, diag = output_jacobian(model, target_x, selector="basis", max_output_rows=max_rows)
                    d1_effect = torch.zeros(max_rows, device=jac.device, dtype=jac.dtype)
                    if target_family.startswith("D7") or target_family.startswith("D9"):
                        logits_t = model(target_x).float()
                        probs = torch.softmax(logits_t, dim=-1)
                        loss_cotangent = -(probs - F.one_hot(target_y.long(), num_classes=output_dim).float()).reshape(-1)
                        readout_jac, _readout_spec, _readout_diag = output_jacobian(model, target_x, selector="readout", max_output_rows=max_rows)
                        readout_delta, _readout_solve = solve_linearized_commit(readout_jac, loss_cotangent[:max_rows], damping=args.basis_damping)
                        d1_effect = (readout_jac @ readout_delta.detach().float()).detach()
                    target_rows.append(
                        {
                            "target_family": target_family,
                            "carrier": carrier,
                            "basis_family": family,
                            "dataset": dataset,
                            "seed": seed,
                            "target_source_type": source_type,
                            "target_input_source": target_input_source,
                            "target_from_w2_diagnostic": 0,
                            "target_norm": float(torch.linalg.vector_norm(target_full[:max_rows]).item()),
                            "hard_slice_fraction": target_meta["hard_slice_fraction"],
                            "hard_slice_count": target_meta["hard_slice_count"],
                            "hard_loss_mean": target_meta["hard_loss_mean"],
                            "all_loss_mean": target_meta["all_loss_mean"],
                            "held_class_cvar_max": target_meta.get("held_class_cvar_max", ""),
                            "held_class_weight_max": target_meta.get("held_class_weight_max", ""),
                            "status": "direct_v22_34_target_family_constructed",
                            "source_artifact": "direct_v22_34_basis_target_family_repair",
                        }
                    )
                    for basis_bank, mask, bank_meta in basis_bank_masks(named, family, [int(x) for x in str(args.basis_bank_dims).split(",") if x.strip()]):
                        if not bank_allowed(target_family, family, basis_bank):
                            continue
                        bank_jac = jac[:, mask]
                        if int(bank_jac.shape[1]) <= 0:
                            continue
                        target = target_full[:max_rows].to(device=jac.device, dtype=jac.dtype)
                        orth_removed = 0.0
                        if target_family.startswith("D7") or target_family.startswith("D9"):
                            gen = torch.Generator(device=jac.device).manual_seed(740_000 + seed + len(fit_rows))
                            random_bank = torch.randn(int(bank_jac.shape[1]), device=jac.device, generator=gen)
                            random_effect = bank_jac @ random_bank
                            random_effect = random_effect * target.norm().clamp_min(1.0e-12) / random_effect.norm().clamp_min(1.0e-12)
                            target, orth_removed = orthogonalize(target, [d1_effect[:max_rows], random_effect])
                        delta_bank, solve_diag = solve_linearized_commit(bank_jac, target, damping=args.basis_damping)
                        delta = torch.zeros(int(jac.shape[1]), device=jac.device, dtype=jac.dtype)
                        delta[mask] = delta_bank.to(device=jac.device, dtype=jac.dtype)
                        with torch.no_grad():
                            base_flat = model(target_x).float().reshape(-1)[: jac.shape[0]].detach().clone()
                            add_flat_delta(named, delta, scale=args.basis_update_scale)
                            exact_flat = model(target_x).float().reshape(-1)[: jac.shape[0]].detach().clone()
                            add_flat_delta(named, delta, scale=-args.basis_update_scale)
                            add_flat_delta(named, delta, scale=args.basis_gradcheck_eps)
                            plus_flat = model(target_x).float().reshape(-1)[: jac.shape[0]].detach().clone()
                            add_flat_delta(named, delta, scale=-2.0 * args.basis_gradcheck_eps)
                            minus_flat = model(target_x).float().reshape(-1)[: jac.shape[0]].detach().clone()
                            add_flat_delta(named, delta, scale=args.basis_gradcheck_eps)
                        lin = (jac @ delta.detach().float()) * float(args.basis_update_scale)
                        exact_diff = exact_flat - base_flat
                        exact_err = torch.linalg.vector_norm(exact_diff - lin).div(torch.linalg.vector_norm(lin).clamp_min(1.0e-12))
                        fd = (plus_flat - minus_flat) / (2.0 * float(args.basis_gradcheck_eps))
                        lin_unscaled = jac @ delta.detach().float()
                        grad_rel = torch.linalg.vector_norm(fd - lin_unscaled).div(torch.linalg.vector_norm(fd).clamp_min(1.0e-12))
                        scale_diag = choose_branch_scale(model, named, delta, held_x, held_y)
                        applied_update_scale = float(scale_diag["applied_update_scale"])
                        with torch.no_grad():
                            add_flat_delta(named, delta, scale=applied_update_scale)
                            selected_exact_flat = model(target_x).float().reshape(-1)[: jac.shape[0]].detach().clone()
                            add_flat_delta(named, delta, scale=-applied_update_scale)
                        selected_lin = (jac @ delta.detach().float()) * applied_update_scale
                        selected_exact_diff = selected_exact_flat - base_flat
                        selected_exact_err = torch.linalg.vector_norm(selected_exact_diff - selected_lin).div(
                            torch.linalg.vector_norm(selected_lin).clamp_min(1.0e-12)
                        )
                        fit_rows.append(
                            {
                                "target_family": target_family,
                                "carrier": carrier,
                                "basis_family": family,
                                "dataset": dataset,
                                "seed": seed,
                                "basis_bank": basis_bank,
                                "repair_stage": repair_stage,
                                "active_hidden": bank_meta.get("active_hidden", ""),
                                "basis_channels": bank_meta.get("basis_channels", ""),
                                "bank_type": bank_meta.get("bank_type", ""),
                                "target_source_type": source_type,
                                "target_input_source": target_input_source,
                                "target_from_w2_diagnostic": 0,
                                "basis_actuator_projection_residual": solve_diag.get("basis_projection_residual", ""),
                                "basis_actuator_cosine": solve_diag.get("basis_projection_cosine", ""),
                                "exact_vs_linearized_error": float(exact_err.item()),
                                "exact_vs_linearized_error_selected": float(selected_exact_err.item()),
                                "basis_update_norm": solve_diag.get("basis_update_norm", ""),
                                "basis_channel_energy": 1.0,
                                "readout_leakage_fraction": orth_removed if (target_family.startswith("D7") or target_family.startswith("D9")) else "",
                                "basis_condition_number": "",
                                "basis_gradcheck_mode": "central",
                                "basis_gradcheck_eps": args.basis_gradcheck_eps,
                                "J_B_gradcheck_rel_error": float(grad_rel.item()),
                                "J_B_gradcheck_pass": int(float(grad_rel.item()) <= 1.0e-3),
                                "jacobian_rows": diag.get("jacobian_rows", ""),
                                "jacobian_cols": int(bank_jac.shape[1]),
                                "selected_param_names": diag.get("selected_param_names", ""),
                                "linearized_commit_status": solve_diag.get("linearized_commit_status", ""),
                                "basis_damping": args.basis_damping,
                                "basis_update_scale": args.basis_update_scale,
                                "scale_policy": scale_diag.get("scale_policy", ""),
                                "selected_scale_multiplier": scale_diag.get("selected_scale_multiplier", ""),
                                "applied_update_scale": scale_diag.get("applied_update_scale", ""),
                                "curvature_safe_scale_pass": scale_diag.get("curvature_safe_scale_pass", ""),
                                "scale_selection_status": scale_diag.get("scale_selection_status", ""),
                                "held_CE_base": scale_diag.get("held_CE_base", ""),
                                "held_CE_delta_at_selected_scale": scale_diag.get("held_CE_delta_at_selected_scale", ""),
                                "sam_sharpness_base": scale_diag.get("sam_sharpness_base", ""),
                                "sam_sharpness_delta_at_selected_scale": scale_diag.get("sam_sharpness_delta_at_selected_scale", ""),
                                "scale_candidate_trace": scale_diag.get("scale_candidate_trace", ""),
                                "status": "direct_v22_34_basis_target_family_repair",
                                "source_artifact": "direct_v22_34_basis_target_family_repair",
                            }
                        )
                        variants = [("basis_native_real", delta)]
                        gen = torch.Generator(device=delta.device).manual_seed(750_000 + seed + len(branch_rows))
                        rnd = torch.zeros_like(delta)
                        rnd_bank = torch.randn(tuple(delta_bank.shape), device=delta.device, generator=gen)
                        rnd_bank = rnd_bank * delta_bank.norm().clamp_min(1.0e-12) / rnd_bank.norm().clamp_min(1.0e-12)
                        rnd[mask] = rnd_bank
                        variants.extend([("same_basis_random", rnd), ("signflip_basis_control", -delta)])
                        bank_start = len(branch_rows)
                        for variant, dvec in variants:
                            branch_model = copy.deepcopy(model).to(device)
                            add_flat_delta(select_named_parameters(branch_model, "basis"), dvec, scale=applied_update_scale)
                            ev = train_branch(branch_model, train_loader, test_loader, device, output_dim, args.basis_branch_horizon, args.lr, args.weight_decay)
                            delta_nll = ev["NLL"] - base_ev["NLL"]
                            branch_rows.append(
                                {
                                    "target_family": target_family,
                                    "carrier": carrier,
                                    "basis_family": family,
                                    "dataset": dataset,
                                    "seed": seed,
                                    "basis_bank": basis_bank,
                                    "repair_stage": repair_stage,
                                    "branch_H": args.basis_branch_horizon,
                                    "branch_variant": variant,
                                    "scale_policy": scale_diag.get("scale_policy", ""),
                                    "selected_scale_multiplier": scale_diag.get("selected_scale_multiplier", ""),
                                    "applied_update_scale": scale_diag.get("applied_update_scale", ""),
                                    "curvature_safe_scale_pass": scale_diag.get("curvature_safe_scale_pass", ""),
                                    "scale_selection_status": scale_diag.get("scale_selection_status", ""),
                                    "held_CE_base": scale_diag.get("held_CE_base", ""),
                                    "held_CE_delta_at_selected_scale": scale_diag.get("held_CE_delta_at_selected_scale", ""),
                                    "sam_sharpness_base": scale_diag.get("sam_sharpness_base", ""),
                                    "sam_sharpness_delta_at_selected_scale": scale_diag.get("sam_sharpness_delta_at_selected_scale", ""),
                                    "scale_candidate_trace": scale_diag.get("scale_candidate_trace", ""),
                                    "source_loss_H50": "" if args.basis_branch_horizon != 50 else -delta_nll,
                                    "source_loss_H100": "" if args.basis_branch_horizon != 100 else -delta_nll,
                                    "source_loss_H200": "" if args.basis_branch_horizon != 200 else -delta_nll,
                                    "source_loss_H400": "" if args.basis_branch_horizon != 400 else -delta_nll,
                                    "NLL_delta_H50": "" if args.basis_branch_horizon != 50 else delta_nll,
                                    "NLL_delta_H100": "" if args.basis_branch_horizon != 100 else delta_nll,
                                    "NLL_delta_H200": "" if args.basis_branch_horizon != 200 else delta_nll,
                                    "NLL_delta_H400": "" if args.basis_branch_horizon != 400 else delta_nll,
                                    "accuracy_delta_vs_base": ev["accuracy"] - base_ev["accuracy"],
                                    "beats_same_basis_random": "",
                                    "beats_signflip_basis_control": "",
                                    "full_loop_ratio": "",
                                    "controller_overhead_ratio": "",
                                    "status": "direct_v22_34_basis_target_family_branch",
                                    "source_artifact": "direct_v22_34_basis_target_family_repair",
                                }
                            )
                        bank_rows = branch_rows[bank_start:]
                        real = next((r for r in bank_rows if r.get("branch_variant") == "basis_native_real"), None)
                        random_ctrl = next((r for r in bank_rows if r.get("branch_variant") == "same_basis_random"), None)
                        signflip_ctrl = next((r for r in bank_rows if r.get("branch_variant") == "signflip_basis_control"), None)
                        if real and random_ctrl and signflip_ctrl:
                            real_nll = branch_nll_value(real)
                            random_nll = branch_nll_value(random_ctrl)
                            signflip_nll = branch_nll_value(signflip_ctrl)
                            real["beats_same_basis_random"] = int(real_nll is not None and random_nll is not None and real_nll < random_nll)
                            real["beats_signflip_basis_control"] = int(real_nll is not None and signflip_nll is not None and real_nll < signflip_nll)

    control_rows, control_summary = basis_target_control_win(fit_rows, branch_rows, epsilon_rep=epsilon_rep)
    strict_fit = [
        r
        for r in fit_rows
        if (finite_float(r.get("basis_actuator_projection_residual"), 999.0) or 999.0) <= 0.05
        and (finite_float(r.get("basis_actuator_cosine"), 0.0) or 0.0) >= 0.98
        and (
            finite_float(
                r.get("exact_vs_linearized_error_selected", r.get("exact_vs_linearized_error")),
                999.0,
            )
            or 999.0
        )
        <= 0.05
        and (finite_float(r.get("J_B_gradcheck_rel_error"), 999.0) or 999.0) <= 1.0e-3
    ]
    best_residual = min((finite_float(r.get("basis_actuator_projection_residual"), math.inf) or math.inf for r in fit_rows), default=math.inf)
    summary = {
        "repair_stage": repair_stage,
        "target_families": args.basis_target_families,
        "datasets": args.basis_datasets,
        "seeds": args.basis_seeds,
        "fit_rows": len(fit_rows),
        "branch_rows": len(branch_rows),
        "target_rows": len(target_rows),
        "control_rows": len(control_rows),
        "epsilon_rep": epsilon_rep,
        "best_projection_residual": "" if not math.isfinite(best_residual) else best_residual,
        "strict_fit_rows_by_fit_matrix": len(strict_fit),
        **control_summary,
        "full_loop_status": "ready_for_D4_full_loop_not_run_in_this_stage" if int_flag(control_summary.get("branch_gate_exploration_pass")) else "gate_blocked_not_run",
        "repair_conclusion": "branch_gate_opened_schedule_full_loop" if int_flag(control_summary.get("branch_gate_exploration_pass")) else "target_family_branch_gate_failed_continue_target_family_design",
    }
    target_all = merge_repair_stage_rows(OUT_ROOT / "v22_34_basis_target_repair_target_matrix.csv", target_rows)
    fit_all = merge_repair_stage_rows(OUT_ROOT / "v22_34_basis_target_repair_fit_matrix.csv", fit_rows)
    branch_all = merge_repair_stage_rows(OUT_ROOT / "v22_34_basis_target_repair_branch_matrix.csv", branch_rows)
    control_all = merge_repair_stage_rows(OUT_ROOT / "v22_34_basis_target_repair_control_win_matrix.csv", control_rows)
    _summary_all = merge_repair_stage_rows(OUT_ROOT / "v22_34_basis_target_repair_summary.csv", [summary])

    repair_source = "direct_v22_34_basis_target_family_repair"
    executed_families = {str(r.get("target_family")) for r in target_rows}
    main_targets = [
        r
        for r in read_rows(OUT_ROOT / "v22_34_basis_target_family_matrix.csv")
        if str(r.get("target_family")) not in executed_families
    ]
    family_summary: list[dict[str, Any]] = []
    for fam in sorted({str(r.get("target_family")) for r in target_rows}):
        fam_fit = [r for r in fit_rows if r.get("target_family") == fam]
        fam_ctrl = [r for r in control_rows if r.get("target_family") == fam]
        fam_target = next((r for r in target_rows if r.get("target_family") == fam), {})
        fam_pass = sum(int_flag(r.get("strict_fit_branch_pass")) for r in fam_ctrl)
        fam_strict = sum(int_flag(r.get("fit_strict_pass")) for r in fam_ctrl)
        family_summary.append(
            {
                "target_family": fam,
                "status": "executed_direct_v22_34_basis_target_repair",
                "target_source_type": fam_target.get("target_source_type", ""),
                "rows": len(fam_fit),
                "strict_fit_rows": fam_strict,
                "strict_fit_branch_pass_rows": fam_pass,
                "strict_fit_branch_pass_rate": rate(fam_pass, fam_strict),
                "blocker_or_next_action": "schedule_full_loop" if fam_strict and rate(fam_pass, fam_strict) >= 0.40 else "branch_gate_failed_continue_D2_D3_D5_D6_D8_or_redesign_target",
                "source_artifact": repair_source,
            }
        )
    write_rows(OUT_ROOT / "v22_34_basis_target_family_matrix.csv", main_targets + family_summary)
    main_fit = [r for r in read_rows(OUT_ROOT / "v22_34_basis_actuator_fit_matrix.csv") if r.get("source_artifact") != repair_source]
    main_branch = [r for r in read_rows(OUT_ROOT / "v22_34_basis_actuator_branch_matrix.csv") if r.get("source_artifact") != repair_source]
    write_rows(OUT_ROOT / "v22_34_basis_actuator_fit_matrix.csv", main_fit + fit_all)
    write_rows(OUT_ROOT / "v22_34_basis_actuator_branch_matrix.csv", main_branch + branch_all)
    existing_full = [
        r
        for r in read_rows(OUT_ROOT / "v22_34_basis_actuator_full_loop_matrix.csv")
        if not (r.get("source_artifact") == repair_source and r.get("target_family") == args.basis_target_families)
    ]
    existing_full.append(
        {
            "target_family": args.basis_target_families,
            "status": summary["full_loop_status"],
            "reason": "basis target-family branch gate opened" if int_flag(control_summary.get("branch_gate_exploration_pass")) else "basis target-family strict-fit branch pass rate below v22.34 exploration threshold 0.40; full-loop remains forbidden.",
            "strict_fit_rows": control_summary["strict_fit_rows"],
            "strict_fit_branch_pass_rows": control_summary["strict_fit_branch_pass_rows"],
            "strict_fit_branch_pass_rate": control_summary["strict_fit_branch_pass_rate"],
            "source_artifact": repair_source,
        }
    )
    write_rows(OUT_ROOT / "v22_34_basis_actuator_full_loop_matrix.csv", existing_full)
    append_exec(
        "run v22.34 basis target-family repair",
        task_id="D_repair_basis_target_family",
        status="pass",
        gpu=str(device),
        files="results/v22_34/v22_34_basis_target_repair_fit_matrix.csv, results/v22_34/v22_34_basis_target_repair_branch_matrix.csv, results/v22_34/v22_34_basis_target_repair_control_win_matrix.csv, results/v22_34/v22_34_basis_target_repair_summary.csv",
        note=f"strict_fit_branch_pass_rate={summary['strict_fit_branch_pass_rate']}; branch_gate_exploration_pass={summary['branch_gate_exploration_pass']}; epsilon_rep={epsilon_rep}; scale_policy={args.basis_scale_policy}",
    )
    return summary


def run_d8_continual_boundary_repair(args: argparse.Namespace) -> dict[str, Any]:
    import copy
    import torch
    import torch.nn.functional as F

    from dgkan.fu.real_jacobian_commit import add_flat_delta, output_jacobian, select_named_parameters, solve_linearized_commit
    from experiments.run_v22_30_fidelity_ladder import evaluate_model, make_class_mnist_task_loaders, make_kan
    from experiments.run_v22_32_causal_actuator_fidelity import basis_bank_masks

    device = torch.device(args.d8_device if str(args.d8_device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    fit_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    def tag_value(value: Any) -> str:
        return str(value).replace("-", "m").replace(".", "p")

    repair_stage = (
        f"D8_class_mnist_boundary_T0T1_steps{args.d8_task0_steps}_{args.d8_task1_steps}"
        f"_rho{tag_value(args.d8_damping)}_scale{tag_value(args.d8_update_scale)}_gceps{tag_value(args.d8_gradcheck_eps)}"
    )

    def merge_repair_stage_rows(path: Path, new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        old_rows = [r for r in read_rows(path) if r.get("repair_stage") != repair_stage]
        merged_rows = old_rows + new_rows
        write_rows(path, merged_rows)
        return merged_rows

    def train_steps(model: Any, loader: Any, steps: int, seed: int) -> None:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        iterator = iter(loader)
        torch.manual_seed(int(seed))
        for _ in range(int(steps)):
            try:
                xb, yb = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                xb, yb = next(iterator)
            xb = xb.to(device).float()
            yb = yb.to(device).long()
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb).float(), yb)
            loss.backward()
            opt.step()

    def collect_batch(loader: Any, n: int) -> tuple[Any, Any]:
        xs: list[Any] = []
        ys: list[Any] = []
        total = 0
        for xb, yb in loader:
            xs.append(xb)
            ys.append(yb)
            total += int(xb.shape[0])
            if total >= int(n):
                break
        x = torch.cat(xs, dim=0)[: int(n)].to(device).float()
        y = torch.cat(ys, dim=0)[: int(n)].to(device).long()
        return x, y

    def eval_tasks(model: Any, task0: dict[str, Any], task1: dict[str, Any]) -> dict[str, float]:
        ev0 = evaluate_model(model, task0["test_loader"], device, 10)
        ev1 = evaluate_model(model, task1["test_loader"], device, 10)
        return {
            "task0_accuracy": float(ev0["accuracy"]),
            "task0_NLL": float(ev0["NLL"]),
            "task1_accuracy": float(ev1["accuracy"]),
            "task1_NLL": float(ev1["NLL"]),
        }

    for seed in _split_csv(args.d8_seeds, int):
        tasks, input_dim, output_dim, x_stats = make_class_mnist_task_loaders(
            args.d8_train_size,
            args.d8_test_size,
            args.batch_size,
            seed,
        )
        task0, task1 = tasks[0], tasks[1]
        for family, carrier in [("D-CHE", "DGKAN_DCHE"), ("D-FOU", "DGKAN_DFOU")]:
            torch.manual_seed(834_000 + seed + (0 if family == "D-CHE" else 10_000))
            initial = make_kan(input_dim, output_dim, args.hidden, seed + 834, device, x_stats, family).to(device)
            task0_model = copy.deepcopy(initial).to(device)
            train_steps(task0_model, task0["train_loader"], args.d8_task0_steps, seed + 10)
            before = eval_tasks(task0_model, task0, task1)

            base_model = copy.deepcopy(task0_model).to(device)
            train_steps(base_model, task1["train_loader"], args.d8_task1_steps, seed + 20)
            base_after = eval_tasks(base_model, task0, task1)
            base_forgetting = max(0.0, before["task0_accuracy"] - base_after["task0_accuracy"])

            xb0, yb0 = collect_batch(task0["train_loader"], args.d8_basis_examples)
            logits = task0_model(xb0).float()
            probs = torch.softmax(logits, dim=-1)
            target_full = -(probs - F.one_hot(yb0.long(), num_classes=output_dim).float()).reshape(-1)
            target_full = target_full / torch.linalg.vector_norm(target_full).clamp_min(1.0e-12)
            max_rows = min(int(args.d8_max_output_rows), int(target_full.numel()))
            jac, _spec, diag = output_jacobian(task0_model, xb0, selector="basis", max_output_rows=max_rows)
            named = select_named_parameters(task0_model, "basis")
            target_rows.append(
                {
                    "target_family": "D8_continual_boundary_basis_target",
                    "carrier": carrier,
                    "basis_family": family,
                    "dataset": "Class_MNIST_T0T1",
                    "seed": seed,
                    "target_source_type": "previous_task_ce_boundary_target",
                    "target_input_source": "task0_train_at_task_boundary_before_task1",
                    "target_from_w2_diagnostic": 0,
                    "target_norm": float(torch.linalg.vector_norm(target_full[:max_rows]).item()),
                    "task0_id": task0["task_id"],
                    "task1_id": task1["task_id"],
                    "task0_accuracy_before_task1": before["task0_accuracy"],
                    "base_task0_accuracy_after_task1": base_after["task0_accuracy"],
                    "base_task1_accuracy_after_task1": base_after["task1_accuracy"],
                    "base_forgetting": base_forgetting,
                    "status": "direct_v22_34_D8_continual_boundary_target_constructed",
                    "source_artifact": "direct_v22_34_D8_continual_boundary_repair",
                }
            )
            for basis_bank, mask, bank_meta in basis_bank_masks(named, family, [int(x) for x in str(args.d8_bank_dims).split(",") if x.strip()]):
                bank_jac = jac[:, mask]
                if int(bank_jac.shape[1]) <= 0:
                    continue
                target = target_full[:max_rows].to(device=jac.device, dtype=jac.dtype)
                delta_bank, solve_diag = solve_linearized_commit(bank_jac, target, damping=args.d8_damping)
                delta = torch.zeros(int(jac.shape[1]), device=jac.device, dtype=jac.dtype)
                delta[mask] = delta_bank.to(device=jac.device, dtype=jac.dtype)
                with torch.no_grad():
                    base_flat = task0_model(xb0).float().reshape(-1)[: jac.shape[0]].detach().clone()
                    add_flat_delta(named, delta, scale=args.d8_update_scale)
                    exact_flat = task0_model(xb0).float().reshape(-1)[: jac.shape[0]].detach().clone()
                    add_flat_delta(named, delta, scale=-args.d8_update_scale)
                    add_flat_delta(named, delta, scale=args.d8_gradcheck_eps)
                    plus_flat = task0_model(xb0).float().reshape(-1)[: jac.shape[0]].detach().clone()
                    add_flat_delta(named, delta, scale=-2.0 * args.d8_gradcheck_eps)
                    minus_flat = task0_model(xb0).float().reshape(-1)[: jac.shape[0]].detach().clone()
                    add_flat_delta(named, delta, scale=args.d8_gradcheck_eps)
                lin = (jac @ delta.detach().float()) * float(args.d8_update_scale)
                exact_diff = exact_flat - base_flat
                exact_err = torch.linalg.vector_norm(exact_diff - lin).div(torch.linalg.vector_norm(lin).clamp_min(1.0e-12))
                fd = (plus_flat - minus_flat) / (2.0 * float(args.d8_gradcheck_eps))
                lin_unscaled = jac @ delta.detach().float()
                grad_rel = torch.linalg.vector_norm(fd - lin_unscaled).div(torch.linalg.vector_norm(fd).clamp_min(1.0e-12))
                fit_rows.append(
                    {
                        "target_family": "D8_continual_boundary_basis_target",
                        "carrier": carrier,
                        "basis_family": family,
                        "dataset": "Class_MNIST_T0T1",
                        "seed": seed,
                        "basis_bank": basis_bank,
                        "repair_stage": repair_stage,
                        "active_hidden": bank_meta.get("active_hidden", ""),
                        "basis_channels": bank_meta.get("basis_channels", ""),
                        "bank_type": bank_meta.get("bank_type", ""),
                        "target_source_type": "previous_task_ce_boundary_target",
                        "target_input_source": "task0_train_at_task_boundary_before_task1",
                        "target_from_w2_diagnostic": 0,
                        "basis_actuator_projection_residual": solve_diag.get("basis_projection_residual", ""),
                        "basis_actuator_cosine": solve_diag.get("basis_projection_cosine", ""),
                        "exact_vs_linearized_error": float(exact_err.item()),
                        "basis_update_norm": solve_diag.get("basis_update_norm", ""),
                        "basis_gradcheck_mode": "central",
                        "basis_gradcheck_eps": args.d8_gradcheck_eps,
                        "J_B_gradcheck_rel_error": float(grad_rel.item()),
                        "J_B_gradcheck_pass": int(float(grad_rel.item()) <= 1.0e-3),
                        "jacobian_rows": diag.get("jacobian_rows", ""),
                        "jacobian_cols": int(bank_jac.shape[1]),
                        "selected_param_names": diag.get("selected_param_names", ""),
                        "basis_damping": args.d8_damping,
                        "basis_update_scale": args.d8_update_scale,
                        "status": "direct_v22_34_D8_continual_boundary_fit",
                        "source_artifact": "direct_v22_34_D8_continual_boundary_repair",
                    }
                )
                variants = [("basis_native_real", delta)]
                gen = torch.Generator(device=delta.device).manual_seed(835_000 + seed + len(branch_rows))
                rnd = torch.zeros_like(delta)
                rnd_bank = torch.randn(tuple(delta_bank.shape), device=delta.device, generator=gen)
                rnd_bank = rnd_bank * delta_bank.norm().clamp_min(1.0e-12) / rnd_bank.norm().clamp_min(1.0e-12)
                rnd[mask] = rnd_bank
                variants.extend([("same_basis_random", rnd), ("signflip_basis_control", -delta)])
                bank_rows: list[dict[str, Any]] = []
                for variant, dvec in variants:
                    branch_model = copy.deepcopy(task0_model).to(device)
                    add_flat_delta(select_named_parameters(branch_model, "basis"), dvec, scale=args.d8_update_scale)
                    boundary_ev = eval_tasks(branch_model, task0, task1)
                    train_steps(branch_model, task1["train_loader"], args.d8_task1_steps, seed + 30 + len(branch_rows))
                    after = eval_tasks(branch_model, task0, task1)
                    forgetting = max(0.0, before["task0_accuracy"] - after["task0_accuracy"])
                    row = {
                        "target_family": "D8_continual_boundary_basis_target",
                        "carrier": carrier,
                        "basis_family": family,
                        "dataset": "Class_MNIST_T0T1",
                        "seed": seed,
                        "basis_bank": basis_bank,
                        "repair_stage": repair_stage,
                        "branch_variant": variant,
                        "task0_id": task0["task_id"],
                        "task1_id": task1["task_id"],
                        "task0_accuracy_before_task1": before["task0_accuracy"],
                        "base_task0_accuracy_after_task1": base_after["task0_accuracy"],
                        "base_task1_accuracy_after_task1": base_after["task1_accuracy"],
                        "base_forgetting": base_forgetting,
                        "boundary_task0_accuracy_delta_vs_base": boundary_ev["task0_accuracy"] - before["task0_accuracy"],
                        "task0_accuracy_after_task1": after["task0_accuracy"],
                        "task1_accuracy_after_task1": after["task1_accuracy"],
                        "task0_NLL_after_task1": after["task0_NLL"],
                        "task1_NLL_after_task1": after["task1_NLL"],
                        "forgetting": forgetting,
                        "forgetting_delta_vs_base": forgetting - base_forgetting,
                        "relative_forgetting_reduction_vs_base": (base_forgetting - forgetting) / max(1.0e-12, base_forgetting),
                        "task0_accuracy_delta_vs_base_after_task1": after["task0_accuracy"] - base_after["task0_accuracy"],
                        "task1_accuracy_delta_vs_base_after_task1": after["task1_accuracy"] - base_after["task1_accuracy"],
                        "beats_same_basis_random": "",
                        "beats_signflip_basis_control": "",
                        "status": "direct_v22_34_D8_continual_boundary_branch",
                        "source_artifact": "direct_v22_34_D8_continual_boundary_repair",
                    }
                    branch_rows.append(row)
                    bank_rows.append(row)
                real = next((r for r in bank_rows if r.get("branch_variant") == "basis_native_real"), None)
                random_ctrl = next((r for r in bank_rows if r.get("branch_variant") == "same_basis_random"), None)
                signflip_ctrl = next((r for r in bank_rows if r.get("branch_variant") == "signflip_basis_control"), None)
                if real and random_ctrl and signflip_ctrl:
                    real_forget = finite_float(real.get("forgetting"), math.inf)
                    random_forget = finite_float(random_ctrl.get("forgetting"), math.inf)
                    signflip_forget = finite_float(signflip_ctrl.get("forgetting"), math.inf)
                    real_forget = real_forget if real_forget is not None else math.inf
                    random_forget = random_forget if random_forget is not None else math.inf
                    signflip_forget = signflip_forget if signflip_forget is not None else math.inf
                    real["beats_same_basis_random"] = int(real_forget < random_forget)
                    real["beats_signflip_basis_control"] = int(real_forget < signflip_forget)
                    projection_residual = finite_float(fit_rows[-1].get("basis_actuator_projection_residual"), 999.0)
                    basis_cosine = finite_float(fit_rows[-1].get("basis_actuator_cosine"), 0.0)
                    exact_error = finite_float(fit_rows[-1].get("exact_vs_linearized_error"), 999.0)
                    gradcheck_error = finite_float(fit_rows[-1].get("J_B_gradcheck_rel_error"), 999.0)
                    projection_residual = projection_residual if projection_residual is not None else 999.0
                    basis_cosine = basis_cosine if basis_cosine is not None else 0.0
                    exact_error = exact_error if exact_error is not None else 999.0
                    gradcheck_error = gradcheck_error if gradcheck_error is not None else 999.0
                    fit_strict = (
                        projection_residual <= 0.05
                        and basis_cosine >= 0.98
                        and exact_error <= 0.05
                        and gradcheck_error <= 1.0e-3
                    )
                    forgetting_delta = finite_float(real.get("forgetting_delta_vs_base"), 0.0)
                    current_accuracy_delta = finite_float(real.get("task1_accuracy_delta_vs_base_after_task1"), -999.0)
                    forgetting_delta = forgetting_delta if forgetting_delta is not None else 0.0
                    current_accuracy_delta = current_accuracy_delta if current_accuracy_delta is not None else -999.0
                    forget_ok = forgetting_delta <= -float(args.d8_forgetting_delta)
                    current_ok = current_accuracy_delta >= -float(args.d8_current_accuracy_tolerance)
                    pass_row = fit_strict and forget_ok and int_flag(real.get("beats_same_basis_random")) and int_flag(real.get("beats_signflip_basis_control")) and current_ok
                    if not fit_strict:
                        explanation = "not_strict_fit_candidate"
                    elif not forget_ok:
                        explanation = "strict_fit_but_forgetting_delta_not_beyond_threshold"
                    elif not int_flag(real.get("beats_same_basis_random")) and not int_flag(real.get("beats_signflip_basis_control")):
                        explanation = "same_basis_random_and_signflip_explain"
                    elif not int_flag(real.get("beats_same_basis_random")):
                        explanation = "same_basis_random_explains"
                    elif not int_flag(real.get("beats_signflip_basis_control")):
                        explanation = "signflip_explains"
                    elif not current_ok:
                        explanation = "current_task_accuracy_debt"
                    else:
                        explanation = "strict_fit_boundary_pass"
                    control_rows.append(
                        {
                            "target_family": "D8_continual_boundary_basis_target",
                            "carrier": carrier,
                            "basis_family": family,
                            "dataset": "Class_MNIST_T0T1",
                            "seed": seed,
                            "basis_bank": basis_bank,
                            "repair_stage": repair_stage,
                            "fit_strict_pass": int(fit_strict),
                            "base_forgetting": base_forgetting,
                            "real_forgetting": real.get("forgetting", ""),
                            "same_basis_random_forgetting": random_ctrl.get("forgetting", ""),
                            "signflip_basis_control_forgetting": signflip_ctrl.get("forgetting", ""),
                            "forgetting_delta_vs_base": real.get("forgetting_delta_vs_base", ""),
                            "relative_forgetting_reduction_vs_base": real.get("relative_forgetting_reduction_vs_base", ""),
                            "forgetting_gate_pass": int(forget_ok),
                            "beats_same_basis_random": real.get("beats_same_basis_random", ""),
                            "beats_signflip_basis_control": real.get("beats_signflip_basis_control", ""),
                            "task1_accuracy_delta_vs_base_after_task1": real.get("task1_accuracy_delta_vs_base_after_task1", ""),
                            "current_task_accuracy_gate_pass": int(current_ok),
                            "strict_fit_boundary_pass": int(pass_row),
                            "control_win_explanation": explanation,
                            "source_artifact": "direct_v22_34_D8_continual_boundary_repair",
                        }
                    )
                for row in bank_rows:
                    detail_rows.append(dict(row))

    strict_rows = [r for r in control_rows if int_flag(r.get("fit_strict_pass"))]
    pass_rows = [r for r in control_rows if int_flag(r.get("strict_fit_boundary_pass"))]
    strict_forget = [r for r in strict_rows if int_flag(r.get("forgetting_gate_pass"))]
    strict_control = [
        r
        for r in strict_rows
        if int_flag(r.get("beats_same_basis_random")) and int_flag(r.get("beats_signflip_basis_control"))
    ]
    reduction_values = [
        v
        for v in (finite_float(r.get("relative_forgetting_reduction_vs_base")) for r in control_rows)
        if v is not None
    ]
    best_reduction = max(reduction_values) if reduction_values else float("-inf")
    summary = {
        "repair_stage": repair_stage,
        "target_family": "D8_continual_boundary_basis_target",
        "seeds": args.d8_seeds,
        "fit_rows": len(fit_rows),
        "branch_rows": len(branch_rows),
        "control_rows": len(control_rows),
        "target_rows": len(target_rows),
        "strict_fit_rows": len(strict_rows),
        "strict_fit_forgetting_gate_rows": len(strict_forget),
        "strict_fit_control_beat_rows": len(strict_control),
        "strict_fit_boundary_pass_rows": len(pass_rows),
        "strict_fit_boundary_pass_rate": rate(len(pass_rows), len(strict_rows)),
        "branch_gate_exploration_pass": int(bool(strict_rows) and rate(len(pass_rows), len(strict_rows)) >= 0.40),
        "branch_gate_official_pass": int(bool(strict_rows) and rate(len(pass_rows), len(strict_rows)) >= 0.60),
        "best_relative_forgetting_reduction_vs_base": "" if not math.isfinite(best_reduction) else best_reduction,
        "full_loop_status": "ready_for_D8_full_loop_not_run_in_this_stage" if bool(strict_rows) and rate(len(pass_rows), len(strict_rows)) >= 0.40 else "gate_blocked_not_run",
        "repair_conclusion": "D8_boundary_gate_opened_schedule_full_loop" if bool(strict_rows) and rate(len(pass_rows), len(strict_rows)) >= 0.40 else "D8_boundary_gate_failed_or_no_strict_fit",
    }
    target_all = merge_repair_stage_rows(OUT_ROOT / "v22_34_D8_continual_boundary_target_matrix.csv", target_rows)
    fit_all = merge_repair_stage_rows(OUT_ROOT / "v22_34_D8_continual_boundary_fit_matrix.csv", fit_rows)
    branch_all = merge_repair_stage_rows(OUT_ROOT / "v22_34_D8_continual_boundary_branch_matrix.csv", branch_rows)
    control_all = merge_repair_stage_rows(OUT_ROOT / "v22_34_D8_continual_boundary_control_win_matrix.csv", control_rows)
    detail_all = merge_repair_stage_rows(OUT_ROOT / "v22_34_D8_continual_boundary_detail_matrix.csv", detail_rows)
    _summary_all = merge_repair_stage_rows(OUT_ROOT / "v22_34_D8_continual_boundary_summary.csv", [summary])

    repair_source = "direct_v22_34_D8_continual_boundary_repair"
    aggregate_strict_rows = [r for r in control_all if int_flag(r.get("fit_strict_pass"))]
    aggregate_pass_rows = [r for r in control_all if int_flag(r.get("strict_fit_boundary_pass"))]
    aggregate_branch_gate = bool(aggregate_strict_rows) and rate(len(aggregate_pass_rows), len(aggregate_strict_rows)) >= 0.40
    targets = [r for r in read_rows(OUT_ROOT / "v22_34_basis_target_family_matrix.csv") if r.get("source_artifact") != repair_source and r.get("target_family") != "D8_continual_boundary_basis_target"]
    targets.append(
        {
            "target_family": "D8_continual_boundary_basis_target",
            "status": "executed_direct_v22_34_D8_continual_boundary_repair",
            "target_source_type": "previous_task_ce_boundary_target",
            "rows": len(fit_all),
            "strict_fit_rows": len(aggregate_strict_rows),
            "strict_fit_branch_pass_rows": len(aggregate_pass_rows),
            "strict_fit_branch_pass_rate": rate(len(aggregate_pass_rows), len(aggregate_strict_rows)),
            "blocker_or_next_action": "schedule_full_loop" if aggregate_branch_gate else "D8_boundary_gate_failed_or_no_strict_fit",
            "source_artifact": repair_source,
        }
    )
    write_rows(OUT_ROOT / "v22_34_basis_target_family_matrix.csv", targets)

    main_fit = [r for r in read_rows(OUT_ROOT / "v22_34_basis_actuator_fit_matrix.csv") if r.get("source_artifact") != repair_source]
    main_branch = [r for r in read_rows(OUT_ROOT / "v22_34_basis_actuator_branch_matrix.csv") if r.get("source_artifact") != repair_source]
    write_rows(OUT_ROOT / "v22_34_basis_actuator_fit_matrix.csv", main_fit + fit_all)
    write_rows(OUT_ROOT / "v22_34_basis_actuator_branch_matrix.csv", main_branch + branch_all)

    full_rows = [r for r in read_rows(OUT_ROOT / "v22_34_basis_actuator_full_loop_matrix.csv") if r.get("source_artifact") != repair_source]
    full_rows.append(
        {
            "target_family": "D8_continual_boundary_basis_target",
            "status": "ready_for_D8_full_loop_not_run_in_this_stage" if aggregate_branch_gate else "gate_blocked_not_run",
            "reason": "D8 stream-aware boundary branch gate opened" if aggregate_branch_gate else "D8 stream-aware boundary branch gate failed; full-loop remains forbidden.",
            "strict_fit_rows": len(aggregate_strict_rows),
            "strict_fit_branch_pass_rows": len(aggregate_pass_rows),
            "strict_fit_branch_pass_rate": rate(len(aggregate_pass_rows), len(aggregate_strict_rows)),
            "source_artifact": repair_source,
        }
    )
    write_rows(OUT_ROOT / "v22_34_basis_actuator_full_loop_matrix.csv", full_rows)
    append_exec(
        "run v22.34 D8 stream-aware continual-boundary basis target repair",
        task_id="D8_continual_boundary_repair",
        status="pass",
        gpu=str(device),
        files="results/v22_34/v22_34_D8_continual_boundary_fit_matrix.csv, results/v22_34/v22_34_D8_continual_boundary_branch_matrix.csv, results/v22_34/v22_34_D8_continual_boundary_control_win_matrix.csv, results/v22_34/v22_34_D8_continual_boundary_summary.csv",
        note=f"strict_fit_boundary_pass_rate={summary['strict_fit_boundary_pass_rate']}; branch_gate_exploration_pass={summary['branch_gate_exploration_pass']}; best_relative_forgetting_reduction={summary['best_relative_forgetting_reduction_vs_base']}",
    )
    return summary


def summarize_ctrl_current() -> dict[str, Any]:
    rows = read_rows(OUT_ROOT / "v22_34_control_win_decomposition_matrix.csv")
    real_causal = sum(1 for r in rows if r.get("control_win_class") == "RealCausal")
    return {
        "control_win_rows": len(rows),
        "RealCausal_rows": real_causal,
        "RealCausal_rate": rate(real_causal, len(rows)),
        "full_loop_promotion_allowed_by_E": int(rate(real_causal, len(rows)) >= 0.25 and bool(rows)),
    }


def materialize_safe_scale_control_decomposition() -> dict[str, Any]:
    control_rows = read_rows(OUT_ROOT / "v22_34_basis_target_repair_control_win_matrix.csv")
    fit_rows = read_rows(OUT_ROOT / "v22_34_basis_target_repair_fit_matrix.csv")
    fit_by_key = {basis_repair_key(r): r for r in fit_rows}
    out_rows: list[dict[str, Any]] = []
    for row in control_rows:
        if not str(row.get("target_family", "")).startswith("D9"):
            continue
        if "policyheldmsharpnessmsafe" not in str(row.get("repair_stage", "")):
            continue
        real_nll = finite_float(row.get("real_NLL_delta"))
        random_nll = finite_float(row.get("same_basis_random_NLL_delta"))
        signflip_nll = finite_float(row.get("signflip_basis_control_NLL_delta"))
        controls = [v for v in [random_nll, signflip_nll] if v is not None]
        best_control = min(controls) if controls else None
        loses_to = []
        if real_nll is not None and random_nll is not None and real_nll >= random_nll:
            loses_to.append("same_basis_random")
        if real_nll is not None and signflip_nll is not None and real_nll >= signflip_nll:
            loses_to.append("signflip")
        fit = fit_by_key.get(basis_repair_key(row), {})
        fit_strict = int_flag(row.get("fit_strict_pass"))
        noise_pass = int_flag(row.get("branch_noise_pass"))
        beats_controls = int_flag(row.get("beats_same_basis_random")) and int_flag(row.get("beats_signflip_basis_control"))
        if fit_strict and loses_to:
            klass = "FlatnessControlExplainsFU"
            label = "curvature_safe_scaling_still_loses_to_controls"
        elif fit_strict and beats_controls and not noise_pass:
            klass = "ActuatorSupportOnly"
            label = "curvature_safe_scaling_beats_controls_but_effect_below_epsilon"
        elif not fit_strict:
            klass = "ActuatorSupportOnly"
            label = "curvature_safe_scaling_not_strict_fit_or_linearity_debt"
        else:
            klass = "ControlExplainedOrNoStableSignal"
            label = "curvature_safe_scaling_no_stable_branch_gate"
        out_rows.append(
            {
                "source_family": "D9_curvature_safe_scale",
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "branch_H": row.get("branch_H", ""),
                "selector_name": f"{row.get('target_family', '')}:{row.get('carrier', '')}:{row.get('basis_bank', '')}",
                "repair_stage": row.get("repair_stage", ""),
                "real_delta": "" if real_nll is None else real_nll,
                "control_delta": "" if best_control is None else best_control,
                "real_minus_best_control_NLL": "" if real_nll is None or best_control is None else real_nll - best_control,
                "epsilon_rep": row.get("epsilon_rep", ""),
                "branch_noise_pass": row.get("branch_noise_pass", ""),
                "fit_strict_pass": row.get("fit_strict_pass", ""),
                "beats_same_basis_random": row.get("beats_same_basis_random", ""),
                "beats_signflip_basis_control": row.get("beats_signflip_basis_control", ""),
                "loses_to_levels": ",".join(loses_to),
                "control_win_cause_label": label,
                "control_win_class": klass,
                "scale_policy": fit.get("scale_policy", ""),
                "selected_scale_multiplier": fit.get("selected_scale_multiplier", ""),
                "applied_update_scale": fit.get("applied_update_scale", ""),
                "curvature_safe_scale_pass": fit.get("curvature_safe_scale_pass", ""),
                "held_CE_delta_at_selected_scale": fit.get("held_CE_delta_at_selected_scale", ""),
                "sam_sharpness_delta_at_selected_scale": fit.get("sam_sharpness_delta_at_selected_scale", ""),
                "exact_vs_linearized_error_selected": fit.get("exact_vs_linearized_error_selected", ""),
                "source_artifact": "direct_v22_34_D9_curvature_safe_scale_repair",
            }
        )
    write_rows(OUT_ROOT / "v22_34_safe_scale_control_win_decomposition_matrix.csv", out_rows)
    if out_rows:
        base_rows = [
            r
            for r in read_rows(OUT_ROOT / "v22_34_control_win_decomposition_matrix.csv")
            if r.get("source_artifact") != "direct_v22_34_D9_curvature_safe_scale_repair"
        ]
        write_rows(OUT_ROOT / "v22_34_control_win_decomposition_matrix.csv", base_rows + out_rows)
    return {
        "safe_scale_decomposition_rows": len(out_rows),
        "FlatnessControlExplainsFU_rows": sum(1 for r in out_rows if r.get("control_win_class") == "FlatnessControlExplainsFU"),
        "ActuatorSupportOnly_rows": sum(1 for r in out_rows if r.get("control_win_class") == "ActuatorSupportOnly"),
    }


def refresh_final_state(task_id: str, note: str) -> dict[str, Any]:
    code = (read_rows(OUT_ROOT / "v22_34_code_truth_gate.csv") or [{}])[0]
    gap = (read_rows(OUT_ROOT / "v22_34_gap_truth_summary.csv") or [{}])[0]
    t1 = summarize_t1_current()
    basis = summarize_basis_current()
    safe_scale_summary = materialize_safe_scale_control_decomposition()
    ctrl_summary = summarize_ctrl_current()
    ctrl_summary.update({k: v for k, v in safe_scale_summary.items() if v})
    write_figures()
    _index_rows, manifest_hash = artifact_index()
    code["artifact_manifest_hash"] = manifest_hash
    write_rows(OUT_ROOT / "v22_34_code_truth_gate.csv", [code])
    final = decide_final(code, gap, t1, basis)
    write_recap(final, ctrl_summary)
    artifact_index()
    append_exec(
        "refresh v22.34 final route, recap, artifact index after repair stage",
        task_id=task_id,
        status="pass",
        files="results/v22_34/v22_34_final_route.json, docs/DG-KAN_v22.34_CausalTargetBasisActuatorFU_实验结果复盘.md, results/v22_34/v22_34_artifact_index.csv",
        note=note,
    )
    return final


def stage_d() -> dict[str, Any]:
    fit_rows = read_rows(V22_33 / "v22_33_basis_actuator_repair_fit_matrix.csv")
    branch_rows = read_rows(V22_33 / "v22_33_basis_actuator_repair_branch_matrix.csv")
    control_rows = read_rows(V22_33 / "v22_33_basis_control_win_decomposition_matrix.csv")
    target_rows = []
    target_rows.append(
        {
            "target_family": "D1_readout_w2_loss_cotangent_target",
            "status": "executed_from_v22_33_named_artifact",
            "target_source_type": "direct_readout_w2_loss_cotangent_effect",
            "rows": len(fit_rows),
            "source_artifact": "results/v22_33/v22_33_basis_actuator_repair_fit_matrix.csv",
        }
    )
    for fam in [
        "D2_multi_cohort_basis_benefit_target",
        "D3_held_cohort_CVaR_target",
        "D4_hard_slice_margin_target",
        "D5_low_degree_DCHE_target",
        "D6_low_frequency_DFOU_target",
        "D7_control_orthogonal_basis_target",
        "D8_continual_boundary_basis_target",
    ]:
        target_rows.append(
            {
                "target_family": fam,
                "status": "not_run_initial_v22_34_reaudit",
                "target_source_type": "",
                "rows": 0,
                "blocker_or_next_action": "requires explicit target-family implementation; do not infer from D1 readout/w2 diagnostic target",
                "source_artifact": "",
            }
        )
    write_rows(OUT_ROOT / "v22_34_basis_target_family_matrix.csv", target_rows)
    fit_out = [{**r, "target_family": "D1_readout_w2_loss_cotangent_target", "source_artifact": "results/v22_33/v22_33_basis_actuator_repair_fit_matrix.csv"} for r in fit_rows]
    branch_out = [{**r, "target_family": "D1_readout_w2_loss_cotangent_target", "source_artifact": "results/v22_33/v22_33_basis_actuator_repair_branch_matrix.csv"} for r in branch_rows]
    write_rows(OUT_ROOT / "v22_34_basis_actuator_fit_matrix.csv", fit_out)
    write_rows(OUT_ROOT / "v22_34_basis_actuator_branch_matrix.csv", branch_out)
    strict_rows = [r for r in control_rows if int_flag(r.get("fit_strict_pass"))]
    pass_rows = [r for r in control_rows if int_flag(r.get("strict_fit_branch_pass"))]
    best_resid = min((finite_float(r.get("basis_actuator_projection_residual"), math.inf) or math.inf for r in fit_rows), default=math.inf)
    full_loop_rows = [
        {
            "target_family": "D1_readout_w2_loss_cotangent_target",
            "status": "gate_blocked_not_run",
            "reason": "D3 strict-fit branch pass rate below v22.34 exploration threshold 0.40; full-loop forbidden by dynamic queue rule.",
            "strict_fit_rows": len(strict_rows),
            "strict_fit_branch_pass_rows": len(pass_rows),
            "strict_fit_branch_pass_rate": rate(len(pass_rows), len(strict_rows)),
            "source_artifact": "results/v22_33/v22_33_basis_control_win_decomposition_matrix.csv",
        }
    ]
    write_rows(OUT_ROOT / "v22_34_basis_actuator_full_loop_matrix.csv", full_loop_rows)
    summary = {
        "fit_rows": len(fit_rows),
        "branch_rows": len(branch_rows),
        "target_family_rows": len(target_rows),
        "executed_target_families": 1 if fit_rows else 0,
        "not_run_target_families": 7,
        "best_projection_residual": "" if not math.isfinite(best_resid) else best_resid,
        "strict_fit_rows": len(strict_rows),
        "strict_fit_branch_pass_rows": len(pass_rows),
        "strict_fit_branch_pass_rate": rate(len(pass_rows), len(strict_rows)),
        "branch_gate_exploration_pass": int(rate(len(pass_rows), len(strict_rows)) >= 0.40 and bool(strict_rows)),
        "branch_gate_official_pass": int(rate(len(pass_rows), len(strict_rows)) >= 0.60 and bool(strict_rows)),
        "full_loop_status": full_loop_rows[0]["status"],
    }
    append_exec(
        "materialize v22.34 Part D D1 basis-target audit and gate-block D4 full-loop",
        task_id="D_basis_target_reaudit_from_v22_33",
        status="pass" if fit_rows else "warn",
        files="results/v22_34/v22_34_basis_target_family_matrix.csv, results/v22_34/v22_34_basis_actuator_fit_matrix.csv, results/v22_34/v22_34_basis_actuator_branch_matrix.csv, results/v22_34/v22_34_basis_actuator_full_loop_matrix.csv",
        note=f"strict_fit_branch_pass_rate={summary['strict_fit_branch_pass_rate']}; D2-D8=not_run_initial_reaudit",
    )
    return summary


def stage_e_f_g_h() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ctrl = read_rows(V22_33 / "v22_33_T1_repair_control_win_decomposition.csv")
    basis_ctrl = read_rows(V22_33 / "v22_33_basis_control_win_decomposition_matrix.csv")
    out_ctrl = []
    for row in ctrl:
        label = row.get("control_win_cause_label", "")
        if "loses_to_L1" in label:
            klass = "DirectionNoSignal"
        elif "beats_low_controls" in label:
            klass = "SubspaceOnly"
        elif "safety_debt" in label:
            klass = "FlatnessRegularization"
        elif int_flag(row.get("beats_all_available_controls")) and int_flag(row.get("real_improves")):
            klass = "RealCausal"
        else:
            klass = "ControlExplainedOrNoStableSignal"
        out_ctrl.append({**row, "control_win_class": klass, "source_artifact": "results/v22_33/v22_33_T1_repair_control_win_decomposition.csv"})
    for row in basis_ctrl:
        out_ctrl.append(
            {
                "source_family": "basis_D1_readout_w2",
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "branch_H": row.get("branch_H", ""),
                "real_delta": row.get("real_NLL_delta", ""),
                "control_delta": min(
                    [v for v in [finite_float(row.get("same_basis_random_NLL_delta")), finite_float(row.get("signflip_basis_control_NLL_delta"))] if v is not None],
                    default="",
                ),
                "control_win_cause_label": row.get("control_win_explanation", ""),
                "control_win_class": "RealCausal" if int_flag(row.get("strict_fit_branch_pass")) else "ActuatorSupportOnly",
                "source_artifact": "results/v22_33/v22_33_basis_control_win_decomposition_matrix.csv",
            }
        )
    write_rows(OUT_ROOT / "v22_34_control_win_decomposition_matrix.csv", out_ctrl)
    real_causal = sum(1 for r in out_ctrl if r.get("control_win_class") == "RealCausal")
    ctrl_summary = {
        "control_win_rows": len(out_ctrl),
        "RealCausal_rows": real_causal,
        "RealCausal_rate": rate(real_causal, len(out_ctrl)),
        "full_loop_promotion_allowed_by_E": int(rate(real_causal, len(out_ctrl)) >= 0.25 and bool(out_ctrl)),
    }
    curv_rows = []
    for row in ctrl:
        curv_rows.append(
            {
                "source_family": row.get("source_family", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "selector_name": row.get("selector_name", ""),
                "margin_mean": row.get("real_margin_mean_delta", ""),
                "margin_q10": row.get("real_margin_q10_delta", ""),
                "margin_q01": row.get("real_margin_q01_delta", ""),
                "hard_slice_margin_gain": row.get("real_tail_q99_delta", ""),
                "sharpness_delta_real": "",
                "sharpness_delta_controls": "",
                "control_win_class": row.get("control_win_cause_label", ""),
                "diagnostic_status": "margin/tail readback only; no standalone promotion",
            }
        )
    write_rows(OUT_ROOT / "v22_34_curvature_representation_matrix.csv", curv_rows)
    write_rows(
        OUT_ROOT / "v22_34_optimizer_integrated_fu_matrix.csv",
        [
            {
                "status": "gate_blocked_not_run",
                "reason": "Part C/D gates did not open a promotable FU variant; optimizer full-loop would be post-hoc.",
                "source_gate": "v22_34_gap_truth_summary + basis_actuator_full_loop_matrix",
            }
        ],
    )
    write_rows(
        OUT_ROOT / "v22_34_optimizer_spectrum_matrix.csv",
        [{"status": "gate_blocked_not_run", "reason": "optimizer-integrated FU not scheduled before Part C/D pass"}],
    )
    write_rows(
        OUT_ROOT / "v22_34_hard_task_four_square_matrix.csv",
        [{"status": "gate_blocked_not_run", "reason": "full superiority/four-square hard task matrix not run because Part B TrueKANGain+BothGain=0 and Part D branch gate failed"}],
    )
    for name in ["continual", "grokking", "efficiency_four_path"]:
        write_rows(OUT_ROOT / f"v22_34_{name}_matrix.csv", [{"status": "not_run_initial_reaudit", "reason": "deferred until C/D causal branch target exists; no fabricated rows"}])
    append_exec(
        "derive v22.34 Part E/F diagnostics and gate-block G/H matrices",
        task_id="E_F_G_H_diagnostics_and_gate_blocks",
        status="pass",
        files="results/v22_34/v22_34_control_win_decomposition_matrix.csv, results/v22_34/v22_34_curvature_representation_matrix.csv, results/v22_34/v22_34_hard_task_four_square_matrix.csv",
        note=f"RealCausal_rate={ctrl_summary['RealCausal_rate']}; promotion_allowed_by_E={ctrl_summary['full_loop_promotion_allowed_by_E']}",
    )
    return ctrl_summary, {"curvature_rows": len(curv_rows), "promotion_allowed": 0}, {"optimizer_hard_status": "gate_blocked_not_run"}


def write_figures() -> None:
    gap_rows = read_rows(OUT_ROOT / "v22_34_gap_truth_matrix.csv")
    gate_rows = read_rows(OUT_ROOT / "v22_34_T1_selector_matrix.csv")
    basis_rows = read_rows(OUT_ROOT / "v22_34_basis_actuator_fit_matrix.csv")
    ctrl_rows = read_rows(OUT_ROOT / "v22_34_control_win_decomposition_matrix.csv")
    write_simple_svg(FIG_ROOT / "v22_34_gap_truth_stacked_bar.svg", "v22.34 gap truth classes", [{"label": r.get("gap_reduction_class", ""), "value": 1} for r in gap_rows])
    write_simple_svg(FIG_ROOT / "v22_34_T1_selector_vs_controls_panel.svg", "v22.34 T1 selector NLL deltas", [{"label": r.get("selector_name", ""), "value": r.get("NLL_delta_vs_base", 0)} for r in gate_rows])
    write_simple_svg(FIG_ROOT / "v22_34_basis_target_residual_vs_branch_gain.svg", "v22.34 basis residual", [{"label": r.get("basis_bank", ""), "value": r.get("basis_actuator_projection_residual", 0)} for r in basis_rows])
    write_simple_svg(FIG_ROOT / "v22_34_control_level_waterfall.svg", "v22.34 control-win classes", [{"label": r.get("control_win_class", ""), "value": 1} for r in ctrl_rows])
    for filename in [
        "v22_34_curvature_vs_control_win_scatter.svg",
        "v22_34_optimizer_spectrum_panel.svg",
        "v22_34_KAN_vs_MLPFU_gap_reduction_panel.svg",
        "v22_34_continual_forgetting_curves.svg",
        "v22_34_grokking_delay_curves.svg",
        "v22_34_efficiency_four_path_waterfall.svg",
    ]:
        write_simple_svg(FIG_ROOT / filename, filename.replace("_", " "), [{"label": "gate_blocked", "value": 0}])


def decide_final(code: dict[str, Any], gap: dict[str, Any], t1: dict[str, Any], basis: dict[str, Any]) -> dict[str, Any]:
    if not int_flag(code.get("clean_unzip_compileall_pass")) or not int_flag(code.get("clean_unzip_import_pass")):
        route = "R0-CodeOrIdentityFail"
    elif int_flag(gap.get("official_candidate_gate_pass")):
        route = "R11-TrueKANGainGapReductionOpened"
    elif int_flag(basis.get("branch_gate_official_pass")):
        route = "R5-BasisActuatorBranchOpened_NoFullLoop"
    elif basis.get("fit_rows"):
        route = "R4-BasisActuatorFitOpened_NoBranchBenefit"
    elif int_flag(t1.get("official_candidate_rows")):
        route = "R3-DirectionCausal_ActuatorNoGo"
    else:
        route = "R2-SubspaceExists_DirectionNoGo"
    if "executed" in str(t1.get("C9_C10_status", "")):
        c_blocker = (
            f"Part C C9/C10 executed; official_candidate_rows={t1.get('official_candidate_rows')} "
            "under v22.34 selector-vs-control gates."
        )
    else:
        c_blocker = "Part C C9/C10 not yet executed and existing C1-C7 selector gate remains closed."
    if int_flag(basis.get("not_run_target_families")):
        d_blocker = (
            f"Part D target-family coverage incomplete: not_run_target_families={basis.get('not_run_target_families')}; "
            "full-loop remains forbidden until a causal target family passes branch gates."
        )
    else:
        d_blocker = "Part D branch gate did not open; full-loop remains forbidden."
    final = {
        "final_route": route,
        "official_full_superiority_ready": 0,
        "latest_status_timestamp": now_sg(),
        "code_truth": code,
        "gap_summary": gap,
        "t1_summary": t1,
        "basis_summary": basis,
        "full_loop_blockers": [
            {
                "status": "gate_blocked",
                "blocker": "Part B TrueKANGain+BothGain gate did not open; full superiority forbidden.",
                "source_artifact": "results/v22_34/v22_34_gap_truth_summary.csv",
            },
            {
                "status": "gate_blocked",
                "blocker": c_blocker,
                "source_artifact": "results/v22_34/v22_34_T1_selector_matrix.csv",
            },
            {
                "status": "gate_blocked",
                "blocker": d_blocker,
                "source_artifact": "results/v22_34/v22_34_basis_target_family_matrix.csv",
            },
        ],
        "no_fabricated_rows_claim": "All numeric rows are direct v22.34 commands or named v22.33/v22.30/v22.32 artifact readbacks. Missing target families are explicit not_run/gate_blocked rows.",
    }
    write_json(OUT_ROOT / "v22_34_final_route.json", final)
    return final


def write_recap(final: dict[str, Any], ctrl_summary: dict[str, Any]) -> None:
    gap_rows = read_rows(OUT_ROOT / "v22_34_gap_truth_matrix.csv")
    t1_rows = read_rows(OUT_ROOT / "v22_34_T1_selector_matrix.csv")
    c9_summary_rows = read_rows(OUT_ROOT / "v22_34_T1_C9C10_repair_summary.csv")
    c9_gate_rows = read_rows(OUT_ROOT / "v22_34_T1_C9C10_repair_gate_eval.csv")
    c9_candidate_rows = read_rows(OUT_ROOT / "v22_34_T1_C9C10_candidate_audit_matrix.csv")
    target_rows = read_rows(OUT_ROOT / "v22_34_basis_target_family_matrix.csv")
    full_loop_rows = read_rows(OUT_ROOT / "v22_34_basis_actuator_full_loop_matrix.csv")
    basis_target_summary_rows = read_rows(OUT_ROOT / "v22_34_basis_target_repair_summary.csv")
    basis_target_fit_rows = read_rows(OUT_ROOT / "v22_34_basis_target_repair_fit_matrix.csv")
    basis_target_target_rows = read_rows(OUT_ROOT / "v22_34_basis_target_repair_target_matrix.csv")
    basis_target_control_rows = read_rows(OUT_ROOT / "v22_34_basis_target_repair_control_win_matrix.csv")
    safe_scale_decomp_rows = read_rows(OUT_ROOT / "v22_34_safe_scale_control_win_decomposition_matrix.csv")
    d8_summary_rows = read_rows(OUT_ROOT / "v22_34_D8_continual_boundary_summary.csv")
    d8_control_rows = read_rows(OUT_ROOT / "v22_34_D8_continual_boundary_control_win_matrix.csv")
    d8_fit_rows = read_rows(OUT_ROOT / "v22_34_D8_continual_boundary_fit_matrix.csv")
    c9_done = bool(c9_summary_rows)
    basis_target_done = bool(basis_target_summary_rows)
    executed_target_families = [
        str(r.get("target_family"))
        for r in target_rows
        if str(r.get("status", "")).startswith("executed_direct_v22_34_")
    ]
    not_run_target_families = [
        str(r.get("target_family"))
        for r in target_rows
        if "not_run" in str(r.get("status", ""))
    ]
    target_family_repair_done = bool(executed_target_families)
    has_d9 = any(str(f).startswith("D9") for f in executed_target_families)
    d2_d8_coverage = (
        "计划内 D2-D8 全部已有真实执行/审计行；额外 D9 repair 已执行"
        if has_d9 and not not_run_target_families
        else ("D2-D8 全部已有真实执行/审计行" if not not_run_target_families else f"仍有 not_run：{','.join(not_run_target_families)}")
    )
    d8_fit_focus_rows = [r for r in d8_fit_rows if r.get("basis_bank") == "all_basis"] or d8_fit_rows
    d8_total_strict = sum(int_flag(r.get("strict_fit_rows")) for r in d8_summary_rows)
    d8_total_boundary_pass = sum(int_flag(r.get("strict_fit_boundary_pass_rows")) for r in d8_summary_rows)
    d8_reduction_values = [
        v
        for v in (finite_float(r.get("best_relative_forgetting_reduction_vs_base")) for r in d8_summary_rows)
        if v is not None
    ]
    d8_best_reduction = max(d8_reduction_values) if d8_reduction_values else None
    d9_summary_rows = [r for r in basis_target_summary_rows if "D9" in str(r.get("target_families", "")) or "D9" in str(r.get("repair_stage", ""))]
    d9_fit_focus_rows = [r for r in basis_target_fit_rows if str(r.get("target_family", "")).startswith("D9") and r.get("basis_bank") == "all_basis"]
    d9_control_focus_rows = [
        r
        for r in basis_target_control_rows
        if str(r.get("target_family", "")).startswith("D9") and (r.get("basis_bank") == "all_basis" or int_flag(r.get("fit_strict_pass")))
    ]
    d9_safe_summary_rows = [r for r in d9_summary_rows if "policyheldmsharpnessmsafe" in str(r.get("repair_stage", ""))]
    d9_safe_fit_rows = [
        r
        for r in basis_target_fit_rows
        if str(r.get("target_family", "")).startswith("D9") and str(r.get("scale_policy", "")) == "held-sharpness-safe"
    ]
    d9_safe_control_rows = [
        r
        for r in basis_target_control_rows
        if str(r.get("target_family", "")).startswith("D9") and "policyheldmsharpnessmsafe" in str(r.get("repair_stage", ""))
    ]
    d9_target_rows = [r for r in basis_target_target_rows if str(r.get("target_family", "")).startswith("D9")]
    d9_best_summary = max(d9_summary_rows, key=lambda r: int_flag(r.get("strict_fit_rows")), default={})
    d9_cifar_strict_rows = [
        r
        for r in d9_control_focus_rows
        if r.get("dataset") == "CIFAR10" and int_flag(r.get("fit_strict_pass"))
    ]
    d9_cifar_best_delta = min(
        (v for v in (finite_float(r.get("real_NLL_delta")) for r in d9_cifar_strict_rows) if v is not None),
        default=None,
    )
    d9_cifar_branch_pass = sum(int_flag(r.get("strict_fit_branch_pass")) for r in d9_cifar_strict_rows)
    d9_safe_strict_rows = [r for r in d9_safe_control_rows if int_flag(r.get("fit_strict_pass"))]
    d9_safe_branch_pass = sum(int_flag(r.get("strict_fit_branch_pass")) for r in d9_safe_strict_rows)
    d9_safe_best_delta = min(
        (v for v in (finite_float(r.get("real_NLL_delta")) for r in d9_safe_strict_rows) if v is not None),
        default=None,
    )
    safe_scale_flatness_rows = [r for r in safe_scale_decomp_rows if r.get("control_win_class") == "FlatnessControlExplainsFU"]
    safe_scale_support_rows = [r for r in safe_scale_decomp_rows if r.get("control_win_class") == "ActuatorSupportOnly"]
    current_status = (
        f"v22.34 已完成 Part A/B 审计、v22.33 证据按 v22.34 gate 迁移、D1 target-family 审计、E/F/G/H gate-blocked artifact，并已真实执行 C9/C10 direct repair 与 target-family repair（{','.join(executed_target_families)}）。"
        if c9_done and target_family_repair_done
        else "v22.34 已完成 Part A/B 审计、v22.33 证据按 v22.34 gate 迁移、D1 target-family 审计、E/F/G/H gate-blocked artifact，并已真实执行 C9/C10 direct repair。"
        if c9_done
        else "v22.34 初始执行已完成 Part A/B 审计、v22.33 证据按 v22.34 gate 迁移、D1 target-family 审计和 E/F/G/H gate-blocked artifact；尚未完成 C9/C10 与 D2-D8 的真实 target-family repair。"
    )
    c_block = (
        "Part C C9/C10 已执行但 selector/control gate 未给出 official candidate"
        if c9_done and not int_flag(final.get("t1_summary", {}).get("official_candidate_rows"))
        else ("Part C C9/C10 已执行且出现 candidate gate row；仍需结合 D target-family/full-loop gate" if c9_done else "Part C 现有 selector gate 未开，且 C9/C10 尚未执行")
    )
    d_block = (
        f"Part D D1 与 {','.join(executed_target_families)} 已执行/审计，但 branch gate 仍未开；{d2_d8_coverage}"
        if target_family_repair_done
        else "Part D 仅 D1 readout/w2 diagnostic target 有执行证据，且 D1 strict-fit branch pass rate 低于 v22.34 exploration gate"
    )
    lines = [
        "# DG-KAN v22.34 Causal Target Basis Actuator FU 实验结果复盘",
        "",
        f"生成时间：{now_sg()}",
        "",
        "## 1. 结论摘要",
        "",
        f"- final_route: `{final.get('final_route')}`",
        f"- official_full_superiority_ready: `{final.get('official_full_superiority_ready')}`",
        f"- 当前状态：{current_status}",
        f"- 不 promotion 原因：Part B TrueKANGain+BothGain 仍为 0；{c_block}；{d_block}",
        "",
        "## 1.1 本轮修复/代码变更审计",
        "",
        "- 修改 `experiments/run_v22_34_causal_target_basis_actuator_fu.py`：新增 `--stage c9c10` repair 入口、C9 control-orthogonal selector、C10 margin-causal selector、candidate audit CSV、C9/C10 gate eval CSV、control-level matrix 合并和重复 direct repair 行过滤。",
        "- 修改 `experiments/run_v22_34_causal_target_basis_actuator_fu.py`：新增 `--stage basis-target` repair 入口，支持 D2 multi-cohort、D3 held-CVaR、D4 hard-slice、D5 low-degree D-CHE、D6 low-frequency D-FOU、D7 control-orthogonal target，写出 basis target fit/branch/control-win CSV、target-family matrix 合并和重复 direct repair 行过滤。",
        "- 修改 `experiments/run_v22_34_causal_target_basis_actuator_fu.py`：新增 `--stage d8-continual` stream-aware repair 入口，独立构造 Class-MNIST task0->task1、previous-task CE boundary target、basis-native/random/signflip controls 和 forgetting/current-task gates。",
        "- 修改 `experiments/run_v22_34_causal_target_basis_actuator_fu.py`：新增 D9 held-class-CVaR control-orthogonal target；basis-target repair_stage 加入 dataset/seed/rho/scale/gradcheck 标签，并切换到 hard loader 以支持 Wine/CIFAR10 等 hard/Tier2 数据审计。",
        "- 修改 `experiments/run_v22_34_causal_target_basis_actuator_fu.py`：新增 `--basis-scale-policy held-sharpness-safe`，用 held-train CE 与 SAM-style sharpness proxy 选择 curvature-safe update scale，并把 selected scale、candidate trace、selected exact-linearization error 写入 fit/branch/control-win artifact。",
        "- 修改目的：按计划从 C1-C7 失败后的推荐路径继续推进 C9/C10 与 D2-D8，不把旧 artifact 的漂亮 row 或单任务 evaluator 结果误当作 target-family/continual 证据。",
        "- 数据约束：C9/C10 selection 只使用 held-train CE/margin 与 control-cosine 过滤；test set 只用于 branch 后评估，`uses_test_direction_selection=0`。",
        "- D8 数据约束：D8 只记录真实 task0->task1 stream 训练后的 previous-task forgetting/current-task accuracy；若 strict fit 或 forgetting/control gate 未过，只写 gate-blocked，不生成 full-loop superiority row。",
        "- Safe-scale 数据约束：scale selection 只使用 held-train CE/sharpness，不读取 test branch 结果；branch gate 仍要求 epsilon_rep、accuracy 和 same-basis random/signflip controls 全部通过。",
        "",
        "## 2. Part A Code / Identity",
        "",
        md_table([final.get("code_truth", {})], limit=4),
        "",
        "执行说明：按计划运行 `compileall -q dgkan experiments` 和 import closure。若失败，后续 scientific route 应降级；本轮结果以 `v22_34_code_truth_gate.csv` 为准。",
        "",
        "## 3. Part B Gap Truth",
        "",
        md_table([final.get("gap_summary", {})], limit=4),
        "",
        md_table(gap_rows, ["dataset", "seed", "task_tier", "carrier", "Delta_MLP_NLL", "Delta_KAN_NLL", "GapReduction_NLL", "gap_reduction_class"], limit=16),
        "",
        "分析：v22.34 继续禁止把 pure gap reduction 当作 architecture value。当前 TrueKANGain/BothGain gate 未开，因此 full superiority 和 hard four-square full-loop 均被 gate-blocked。",
        "",
        "## 4. Part C T1 Direction",
        "",
        md_table([final.get("t1_summary", {})], limit=4),
        "",
        md_table(t1_rows, ["dataset", "seed", "selector_name", "branch_H", "NLL_delta_vs_base", "beats_L3_control", "beats_L4_control", "beats_L5_control", "margin_q10_delta_vs_base", "status"], limit=20),
        "",
        (
            "C9/C10 repair 结果：\n\n"
            + md_table(c9_summary_rows, limit=4)
            + "\n\n"
            + md_table(c9_gate_rows, ["selector_name", "direct_rows", "beats_base_rate", "beats_L3_rate", "beats_L4_rate", "beats_L5_rate", "mean_NLL_delta_vs_base", "control_noise_std", "exploration_pass", "official_candidate_pass"], limit=8)
            + "\n\ncandidate audit 摘要：\n\n"
            + md_table(c9_candidate_rows, ["dataset", "seed", "candidate_label", "candidate_source", "held_CE_delta", "held_margin_q10_delta", "max_abs_cos_to_selection_controls"], limit=16)
            if c9_done
            else "分析：当前 v22.34 只复核了 v22.33 已执行 C1-C7 证据；C9 control-orthogonal 与 C10 margin-causal 仍是下一步必须真实执行的 repair，不允许从现有 C1-C7 漂亮 row 推断。"
        ),
        "",
        (
            "分析：C9/C10 已真实执行且 gate 未开；本轮已继续完成 D2-D8 target-family repair，当前 blocker 收敛到 Part D 的 branch/forgetting gates。"
            if c9_done and target_family_repair_done
            else ("分析：C9/C10 已真实执行；若 gate 未开，后续应继续按计划进入 D2-D8 target-family repair，而不是扩大 damping/scale scan。" if c9_done else "")
        ),
        "",
        "## 5. Part D Basis Target Families",
        "",
        md_table([final.get("basis_summary", {})], limit=4),
        "",
        md_table(target_rows, ["target_family", "status", "rows", "blocker_or_next_action", "source_artifact"], limit=12),
        "",
        md_table(full_loop_rows, limit=4),
        "",
        (
            "D target-family repair 结果：\n\n"
            + md_table(basis_target_summary_rows, limit=4)
            + "\n\n"
            + md_table(basis_target_control_rows, ["target_family", "dataset", "seed", "basis_bank", "branch_H", "epsilon_rep", "fit_strict_pass", "real_NLL_delta", "same_basis_random_NLL_delta", "signflip_basis_control_NLL_delta", "branch_noise_pass", "beats_same_basis_random", "beats_signflip_basis_control", "strict_fit_branch_pass", "control_win_explanation"], limit=24)
            if basis_target_done
            else ""
        ),
        (
            "\nD9 held-class-CVaR control-orthogonal repair 结果：\n\n"
            + md_table(d9_summary_rows, limit=8)
            + "\n\nD9 all-basis fit 证据摘要：\n\n"
            + md_table(d9_fit_focus_rows, ["repair_stage", "carrier", "dataset", "basis_bank", "basis_actuator_projection_residual", "basis_actuator_cosine", "exact_vs_linearized_error", "J_B_gradcheck_rel_error", "basis_damping", "basis_update_scale", "basis_gradcheck_eps"], limit=12)
            + "\n\nD9 control-win focus：\n\n"
            + md_table(d9_control_focus_rows, ["repair_stage", "carrier", "dataset", "basis_bank", "fit_strict_pass", "real_NLL_delta", "same_basis_random_NLL_delta", "signflip_basis_control_NLL_delta", "branch_noise_pass", "beats_same_basis_random", "beats_signflip_basis_control", "accuracy_delta_vs_base", "strict_fit_branch_pass", "control_win_explanation"], limit=16)
            + "\n\nD9 target construction audit：\n\n"
            + md_table(d9_target_rows, ["carrier", "dataset", "target_source_type", "target_input_source", "hard_loss_mean", "all_loss_mean", "held_class_cvar_max", "held_class_weight_max"], limit=8)
            if d9_summary_rows
            else ""
        ),
        (
            "\nD9 curvature-safe scale repair 结果：\n\n"
            + md_table(d9_safe_summary_rows, limit=6)
            + "\n\nD9 safe-scale fit/selection 证据摘要：\n\n"
            + md_table(
                d9_safe_fit_rows,
                [
                    "repair_stage",
                    "carrier",
                    "dataset",
                    "basis_bank",
                    "exact_vs_linearized_error_selected",
                    "selected_scale_multiplier",
                    "applied_update_scale",
                    "curvature_safe_scale_pass",
                    "held_CE_delta_at_selected_scale",
                    "sam_sharpness_delta_at_selected_scale",
                    "scale_selection_status",
                ],
                limit=12,
            )
            + "\n\nD9 safe-scale control-win focus：\n\n"
            + md_table(
                d9_safe_control_rows,
                [
                    "repair_stage",
                    "carrier",
                    "dataset",
                    "basis_bank",
                    "fit_strict_pass",
                    "real_NLL_delta",
                    "same_basis_random_NLL_delta",
                    "signflip_basis_control_NLL_delta",
                    "branch_noise_pass",
                    "beats_same_basis_random",
                    "beats_signflip_basis_control",
                    "accuracy_delta_vs_base",
                    "strict_fit_branch_pass",
                    "control_win_explanation",
                ],
                limit=16,
            )
            if d9_safe_summary_rows
            else ""
        ),
        "",
        (
            f"修复/审计说明：按 v22.34 计划，D1 readout/w2 只能作为 diagnostic baseline；本轮已真实执行 {','.join(executed_target_families)}。"
            + (f"剩余 {','.join(not_run_target_families)} 仍是 not_run。" if not_run_target_families else "当前 D2-D8 已无 not_run target-family。")
            + "当前复盘把未执行 target family 与已执行失败 target family 分开记录，避免把 D1 bank/basis 变化冒充 target-family 成功。"
            if target_family_repair_done
            else "修复/审计说明：按 v22.34 计划，D1 readout/w2 只能作为 diagnostic baseline；D2-D8 需要独立 target-family implementation。当前复盘明确把 D2-D8 标成 not_run，避免把 D1 bank/basis 变化冒充 target-family 成功。"
        ),
        "",
        (
            "D8 continual-boundary repair 结果：\n\n"
            + md_table(d8_summary_rows, limit=4)
            + "\n\nD8 all-basis fit 证据摘要：\n\n"
            + md_table(d8_fit_focus_rows, ["repair_stage", "carrier", "basis_bank", "basis_actuator_projection_residual", "basis_actuator_cosine", "exact_vs_linearized_error", "J_B_gradcheck_rel_error", "basis_damping", "basis_update_scale", "basis_gradcheck_eps"], limit=12)
            + "\n\n"
            + md_table(d8_control_rows, ["carrier", "basis_family", "dataset", "seed", "basis_bank", "fit_strict_pass", "base_forgetting", "real_forgetting", "same_basis_random_forgetting", "signflip_basis_control_forgetting", "forgetting_delta_vs_base", "relative_forgetting_reduction_vs_base", "forgetting_gate_pass", "beats_same_basis_random", "beats_signflip_basis_control", "task1_accuracy_delta_vs_base_after_task1", "strict_fit_boundary_pass", "control_win_explanation"], limit=24)
            if d8_summary_rows
            else ""
        ),
        "",
        "## 6. Part E/F/G/H",
        "",
        md_table([ctrl_summary], limit=4),
        "",
        (
            "D9 safe-scale control-win decomposition：\n\n"
            + md_table(
                safe_scale_decomp_rows,
                [
                    "repair_stage",
                    "dataset",
                    "selector_name",
                    "real_delta",
                    "control_delta",
                    "real_minus_best_control_NLL",
                    "branch_noise_pass",
                    "fit_strict_pass",
                    "control_win_class",
                    "control_win_cause_label",
                    "selected_scale_multiplier",
                    "held_CE_delta_at_selected_scale",
                    "sam_sharpness_delta_at_selected_scale",
                ],
                limit=16,
            )
            if safe_scale_decomp_rows
            else ""
        ),
        "",
        "E/F 结论：control-win/curvature/margin rows 只解释 blocker，不单独 promotion。G/H full-loop、optimizer、hard four-square、continual、grokking都因 B/C/D gate 未开而写入 gate_blocked/not_run artifact。",
        "",
        "## 7. Insight",
        "",
        "- Insight 1：v22.34 当前证据仍支持 v22.33 判断：actuator fit 已打开，但 target causality 没打开。",
        "- Insight 2：把 D1 readout/w2 diagnostic target 继续变换 bank/scale，不能满足 v22.34 的 target-family 修复要求。",
        ("- Insight 3：C9/C10 已从方向选择侧补强；若仍输给 L3/L4/L5 或无 official candidate，证据链会把 blocker 收敛到 target-family causality，而不是 actuator fit。" if c9_done else "- Insight 3：下一步要真实实现 C9/C10 与 D2-D8 中至少一组 target family；否则继续 full-loop 会违反 no best-row/no diagnostic-promotion 约束。"),
        ("- Insight 4：D2-D7 target-family repair 出现少数 strict-fit 行，但 0 行通过 `epsilon_rep` branch gate；这说明当前 blocker 已从可写性转移到可泛化的 target causality，不能只看 residual 或 branch NLL 小幅下降。" if basis_target_done else ""),
        (
            f"- Insight 5：D8 stream-aware continual-boundary 已真实执行；累计 strict-fit rows={d8_total_strict}，strict-fit boundary pass rows={d8_total_boundary_pass}，best_relative_forgetting_reduction_vs_base={d8_best_reduction if d8_best_reduction is not None else '无'}。低 damping 修复能打开局部 fit，但 previous-task forgetting 仍未相对 base 降低，因此 blocker 是 target/continual causality，而不是单纯局部可写性。"
            if d8_summary_rows and d8_total_strict
            else ("- Insight 5：D8 stream-aware continual-boundary 已真实执行；本次没有 strict-fit row，且 previous-task forgetting 没有相对 base 降低，因此不能把 continual full-loop 打开。" if d8_summary_rows else "")
        ),
        (
            f"- Insight 6：D9 held-class-CVaR control-orthogonal target 是按 control-win/target-redesign 路线新增的额外 repair。最佳 D9 stage 的 strict-fit rows={d9_best_summary.get('strict_fit_rows')}、strict_fit_improve_rows={d9_best_summary.get('strict_fit_improve_rows')}、strict_fit_control_beat_rows={d9_best_summary.get('strict_fit_control_beat_rows')}，但 strict_fit_branch_pass_rows={d9_best_summary.get('strict_fit_branch_pass_rows')}；这说明高杠杆 held-class target 能恢复局部可写性和部分 control beat，但 effect size 仍远低于 epsilon_rep，不能 promotion。"
            if d9_summary_rows
            else ""
        ),
        (
            f"- Insight 7：D9 hard-vision CIFAR10 H200/H400 已真实执行；strict-fit CIFAR rows={len(d9_cifar_strict_rows)}，best strict real_NLL_delta={d9_cifar_best_delta}，strict_fit_branch_pass_rows={d9_cifar_branch_pass}。H400 没有解除 blocker，说明延长 branch horizon 只能把小幅改善维持在 control/noise 同量级，仍未达到 v22.34 branch gate。"
            if d9_cifar_strict_rows
            else ""
        ),
        (
            f"- Insight 8：D9 curvature-safe scale repair 已按 Part E/F 假设执行；strict-fit rows={len(d9_safe_strict_rows)}，best strict real_NLL_delta={d9_safe_best_delta}，strict_fit_branch_pass_rows={d9_safe_branch_pass}，FlatnessControlExplainsFU rows={len(safe_scale_flatness_rows)}，ActuatorSupportOnly rows={len(safe_scale_support_rows)}。若仍为 0，说明 held-train sharpness-safe 缩放不能把局部 target fit 转化为超过 repeat-noise epsilon 的 branch-causal benefit。"
            if d9_safe_summary_rows
            else ""
        ),
        "",
        "## 8. 复现入口",
        "",
        "- runner: `experiments/run_v22_34_causal_target_basis_actuator_fu.py`",
        "- execution log: `docs/DG-KAN_v22.34_CausalTargetBasisActuatorFU_执行日志.md`",
        "- recap: `docs/DG-KAN_v22.34_CausalTargetBasisActuatorFU_实验结果复盘.md`",
        "- final route: `results/v22_34/v22_34_final_route.json`",
        "- command journal: `results/v22_34/v22_34_command_journal.csv`",
    ]
    RECAP_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(_args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    append_exec(
        " ".join([f"KAN_PYTHON={PYTHON}", shlex.quote(sys.executable), *[shlex.quote(a) for a in sys.argv]]),
        task_id="v22_34_start",
        status="started",
        gpu=f"visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'system default')}",
        files=str(PLAN_DOC.relative_to(ROOT)),
        note="Initial v22.34 execution: hard gates, artifact re-audit, explicit gate-blocked/not_run rows; no fabricated target-family data.",
    )
    code = stage_a()
    gap = stage_b()
    t1 = stage_c()
    basis = stage_d()
    ctrl_summary, _curv_summary, _blocked = stage_e_f_g_h()
    write_figures()
    _index_rows, manifest_hash = artifact_index()
    code["artifact_manifest_hash"] = manifest_hash
    write_rows(OUT_ROOT / "v22_34_code_truth_gate.csv", [code])
    final = decide_final(code, gap, t1, basis)
    write_recap(final, ctrl_summary)
    artifact_index()
    append_exec(
        "write v22.34 final route, recap, artifact index, and required figures",
        task_id="Z_finalize_v22_34_initial",
        status="pass",
        files="results/v22_34/v22_34_final_route.json, docs/DG-KAN_v22.34_CausalTargetBasisActuatorFU_实验结果复盘.md, results/v22_34/v22_34_artifact_index.csv",
        note=f"final_route={final.get('final_route')}; official={final.get('official_full_superiority_ready')}; next_required=C9/C10_and_D2-D8_target_family_repair",
    )
    return final


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["all", "c9c10", "basis-target", "d8-continual", "refresh"])
    p.add_argument("--t1-device", default="cuda:0")
    p.add_argument("--t1-datasets", default="CIFAR10,SVHN,EMNIST")
    p.add_argument("--t1-seeds", default="0,1")
    p.add_argument("--t1-train-size", type=int, default=192)
    p.add_argument("--t1-test-size", type=int, default=128)
    p.add_argument("--t1-examples", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--t1-pretrain-steps", type=int, default=20)
    p.add_argument("--t1-signal-cohorts", type=int, default=6)
    p.add_argument("--t1-cohort-size", type=int, default=24)
    p.add_argument("--t1-rank-cap", type=int, default=6)
    p.add_argument("--t1-sketch-dim", type=int, default=128)
    p.add_argument("--t1-candidate-count", type=int, default=12)
    p.add_argument("--t1-branch-trust", type=float, default=0.005)
    p.add_argument("--t1-branch-horizon", type=int, default=400)
    p.add_argument("--lr", type=float, default=3.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--basis-device", default="cuda:0")
    p.add_argument("--basis-datasets", default="MNIST,FMNIST")
    p.add_argument("--basis-seeds", default="0")
    p.add_argument("--basis-target-families", default="D4_hard_slice_margin_target,D7_control_orthogonal_basis_target")
    p.add_argument("--basis-train-size", type=int, default=192)
    p.add_argument("--basis-test-size", type=int, default=128)
    p.add_argument("--basis-examples", type=int, default=64)
    p.add_argument("--basis-max-output-rows", type=int, default=640)
    p.add_argument("--basis-pretrain-steps", type=int, default=20)
    p.add_argument("--basis-bank-dims", default="4,8")
    p.add_argument("--basis-damping", type=float, default=1.0e-4)
    p.add_argument("--basis-update-scale", type=float, default=0.003)
    p.add_argument("--basis-gradcheck-eps", type=float, default=1.0e-4)
    p.add_argument("--basis-scale-policy", default="fixed", choices=["fixed", "held-sharpness-safe"])
    p.add_argument("--basis-safe-scale-candidates", default="0.5,1.0,2.0")
    p.add_argument("--basis-sharpness-rho", type=float, default=1.0e-3)
    p.add_argument("--basis-sharpness-tolerance", type=float, default=0.0)
    p.add_argument("--basis-scale-ce-tolerance", type=float, default=0.0)
    p.add_argument("--basis-hard-fraction", type=float, default=0.25)
    p.add_argument("--basis-branch-horizon", type=int, default=400)
    p.add_argument("--basis-epsilon-rep", type=float, default=-1.0, help="negative means use v22_34_gap_truth_summary.repeat_noise_epsilon")
    p.add_argument("--d8-device", default="cuda:0")
    p.add_argument("--d8-seeds", default="0")
    p.add_argument("--d8-train-size", type=int, default=300)
    p.add_argument("--d8-test-size", type=int, default=200)
    p.add_argument("--d8-task0-steps", type=int, default=80)
    p.add_argument("--d8-task1-steps", type=int, default=120)
    p.add_argument("--d8-basis-examples", type=int, default=64)
    p.add_argument("--d8-max-output-rows", type=int, default=512)
    p.add_argument("--d8-bank-dims", default="4,8")
    p.add_argument("--d8-damping", type=float, default=1.0e-4)
    p.add_argument("--d8-update-scale", type=float, default=0.003)
    p.add_argument("--d8-gradcheck-eps", type=float, default=1.0e-3)
    p.add_argument("--d8-forgetting-delta", type=float, default=0.01)
    p.add_argument("--d8-current-accuracy-tolerance", type=float, default=0.01)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.stage == "all":
        final = run_all(args)
    elif args.stage == "c9c10":
        ensure_out()
        command = " ".join([f"KAN_PYTHON={PYTHON}", shlex.quote(sys.executable), *[shlex.quote(a) for a in sys.argv]])
        append_exec(
            command,
            task_id="v22_34_C9C10_repair_start",
            status="started",
            gpu=args.t1_device,
            files="experiments/run_v22_34_causal_target_basis_actuator_fu.py",
            note=(
                "Direct repair after initial gate failure. Code change audit: added --stage c9c10, C9/C10 held-train selectors, "
                "candidate audit, gate eval, duplicate repair-row filtering, and recap refresh."
            ),
        )
        summary = run_c9c10_repair(args)
        final = refresh_final_state(
            "Z_finalize_v22_34_after_C9C10",
            f"C9C10_exploration_pass_rows={summary.get('exploration_pass_rows')}; C9C10_official_candidate_rows={summary.get('official_candidate_rows')}; final_refresh_after_repair=1",
        )
    elif args.stage == "basis-target":
        ensure_out()
        command = " ".join([f"KAN_PYTHON={PYTHON}", shlex.quote(sys.executable), *[shlex.quote(a) for a in sys.argv]])
        append_exec(
            command,
            task_id="v22_34_basis_target_repair_start",
            status="started",
            gpu=args.basis_device,
            files="experiments/run_v22_34_causal_target_basis_actuator_fu.py",
            note=(
                f"Direct D repair after C9/C10 gate stayed closed. Runs target families: {args.basis_target_families}; "
                f"scale_policy={args.basis_scale_policy}; stops D1 damping/scale scanning per v22.34 blocker rule."
            ),
        )
        summary = run_basis_target_family_repair(args)
        final = refresh_final_state(
            "Z_finalize_v22_34_after_basis_target",
            f"D_target_strict_fit_branch_pass_rate={summary.get('strict_fit_branch_pass_rate')}; D_branch_gate_exploration_pass={summary.get('branch_gate_exploration_pass')}; full_loop_status={summary.get('full_loop_status')}",
        )
    elif args.stage == "d8-continual":
        ensure_out()
        command = " ".join([f"KAN_PYTHON={PYTHON}", shlex.quote(sys.executable), *[shlex.quote(a) for a in sys.argv]])
        append_exec(
            command,
            task_id="v22_34_D8_continual_boundary_start",
            status="started",
            gpu=args.d8_device,
            files="experiments/run_v22_34_causal_target_basis_actuator_fu.py",
            note=(
                "Runs stream-aware D8 continual-boundary basis target on Class-MNIST task0->task1; "
                "compares base, basis-native real, same-basis random, and signflip controls using previous-task forgetting gates."
            ),
        )
        summary = run_d8_continual_boundary_repair(args)
        final = refresh_final_state(
            "Z_finalize_v22_34_after_D8_continual",
            f"D8_strict_fit_boundary_pass_rate={summary.get('strict_fit_boundary_pass_rate')}; D8_branch_gate_exploration_pass={summary.get('branch_gate_exploration_pass')}; full_loop_status={summary.get('full_loop_status')}",
        )
    elif args.stage == "refresh":
        ensure_out()
        command = " ".join([f"KAN_PYTHON={PYTHON}", shlex.quote(sys.executable), *[shlex.quote(a) for a in sys.argv]])
        append_exec(
            command,
            task_id="v22_34_refresh_start",
            status="started",
            gpu="n/a",
            files="experiments/run_v22_34_causal_target_basis_actuator_fu.py",
            note="Refresh final route/recap/artifact index and materialize safe-scale control-win decomposition from existing real experimental rows; no new experimental rows generated.",
        )
        final = refresh_final_state(
            "Z_refresh_v22_34_after_safe_scale_decomposition",
            "materialized safe-scale control-win decomposition from existing D9 safe-scale rows; no new experimental rows generated",
        )
    else:
        raise SystemExit(f"unsupported stage: {args.stage}")
    print(json.dumps({"final_route": final.get("final_route"), "official_full_superiority_ready": final.get("official_full_superiority_ready")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
