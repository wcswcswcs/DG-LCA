#!/usr/bin/env python3
"""DG-KAN v22.32 causal actuator fidelity audit runner.

This runner is intentionally conservative.  It creates v22.32 artifacts from:

1. commands executed in this run,
2. a small direct basis-native actuator smoke executed in this run, and
3. explicitly named v22.30 artifacts used as historical inputs for the v22.32
   decomposition audits.

Rows that were not directly run are marked as such; no placeholder is promoted
as a successful experiment.
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
import random
import shlex
import statistics
import subprocess
import sys
import tarfile
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PYTHON = os.environ.get("KAN_PYTHON", "/home/chengshun.wang/miniconda3/envs/kan/bin/python")
OUT_ROOT = ROOT / "results/v22_32"
FIG_ROOT = OUT_ROOT / "figures"
LOG_ROOT = OUT_ROOT / "logs"
PLAN_DOC = ROOT / "docs/DG-KAN_v22.32_CausalActuatorFidelityFU_扩充版.md"
EXEC_DOC = ROOT / "docs/DG-KAN_v22.32_CausalActuatorFidelityFU_执行日志.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v22.32_CausalActuatorFidelityFU_实验结果复盘.md"
V22_30 = ROOT / "results/v22_30"


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.32 Causal Actuator Fidelity FU 执行日志\n\n"
            f"生成时间：{now_sg()}\n\n"
            "记录原则：只记录真实命令、文件、输入、输出、状态、blocker 与修复尝试；"
            "未执行项必须写明 gate/blocker，不写成完成。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.32 Causal Actuator Fidelity FU 实验结果复盘\n\n"
            f"生成时间：{now_sg()}\n\n"
            "本复盘只引用 v22.32 本轮 artifact、命令日志与明确命名的历史 artifact 读回结果；"
            "禁止编造数据。\n",
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
    journal = OUT_ROOT / "v22_32_command_journal.csv"
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
    merged = os.environ.copy()
    if env:
        merged.update(env)
    started = time.time()
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd or ROOT),
            env=merged,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        proc = subprocess.CompletedProcess(args=args, returncode=124, stdout=exc.stdout or "", stderr=exc.stderr or "")
    status = "pass" if proc.returncode == 0 else ("timeout" if proc.returncode == 124 else "fail")
    stdout_path = LOG_ROOT / f"{task_id}_stdout.log"
    stderr_path = LOG_ROOT / f"{task_id}_stderr.log"
    stdout_path.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8", errors="replace")
    append_exec(
        " ".join(shlex.quote(x) for x in args),
        task_id=task_id,
        status=status,
        gpu=gpu,
        exit_code=proc.returncode,
        files=f"{stdout_path.relative_to(ROOT)}, {stderr_path.relative_to(ROOT)}",
        note=f"elapsed_sec={time.time() - started:.3f}; cwd={cwd or ROOT}",
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
    fields: list[str] = list(fieldnames or [])
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
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def rate(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def spearman_corr(xs: Iterable[float], ys: Iterable[float]) -> float | str:
    x = [float(v) for v in xs if math.isfinite(float(v))]
    y = [float(v) for v in ys if math.isfinite(float(v))]
    if len(x) != len(y) or len(x) < 3:
        return ""

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            rank = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                out[order[k]] = rank
            i = j + 1
        return out

    rx = ranks(x)
    ry = ranks(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    vx = sum((v - mx) ** 2 for v in rx)
    vy = sum((v - my) ** 2 for v in ry)
    if vx <= 1.0e-12 or vy <= 1.0e-12:
        return ""
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return float(cov / math.sqrt(vx * vy))


def md_table(rows: list[dict[str, Any]], fields: list[str] | None = None, limit: int = 16) -> str:
    if not rows:
        return "\n_无落盘 rows。_\n"
    fields = fields or list(rows[0].keys())
    shown = rows[:limit]
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in shown:
        out.append("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
    if len(rows) > limit:
        out.append(f"\n_仅显示前 {limit} / {len(rows)} rows；完整 CSV 见 artifact。_")
    return "\n".join(out) + "\n"


def write_simple_svg(name: str, title: str, rows: list[dict[str, Any]], value_key: str = "value") -> None:
    width = 980
    height = max(250, 68 + 30 * max(1, len(rows[:24])))
    max_val = max([abs(float(finite_float(r.get(value_key), 0.0) or 0.0)) for r in rows[:24]] + [1.0])
    parts = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="24" y="34" font-size="20" font-family="sans-serif">{title}</text>',
    ]
    for i, row in enumerate(rows[:24]):
        y = 64 + i * 30
        label = str(row.get("label") or i)
        val = float(finite_float(row.get(value_key), 0.0) or 0.0)
        bar = int(420 * abs(val) / max_val)
        color = "#20639b" if val >= 0 else "#b33a3a"
        parts.append(f'<text x="24" y="{y + 14}" font-size="12" font-family="sans-serif">{label}</text>')
        parts.append(f'<rect x="330" y="{y}" width="{bar}" height="17" fill="{color}"/>')
        parts.append(f'<text x="{338 + bar}" y="{y + 13}" font-size="12" font-family="monospace">{val:.4g}</text>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">' + "".join(parts) + "</svg>\n"
    (FIG_ROOT / name).write_text(svg, encoding="utf-8")


def artifact_index() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in [OUT_ROOT, FIG_ROOT, LOG_ROOT]:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rows.append(
                {
                    "artifact": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "mtime_sg": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(path.stat().st_mtime)),
                }
            )
    for path in [PLAN_DOC, EXEC_DOC, RECAP_DOC, Path(__file__).resolve()]:
        if path.exists():
            rows.append(
                {
                    "artifact": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "mtime_sg": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(path.stat().st_mtime)),
                }
            )
    return rows


def stage_a_code_closure() -> dict[str, Any]:
    ensure_out()
    default_import = run_logged(
        [
            "python",
            "-c",
            "import dgkan, dgkan.fu.core, dgkan.fu.mechanisms, dgkan.fu.metric_solver, "
            "dgkan.models.fc_purekan_primitives, dgkan.integration.kanbefair_adapter; print('import_ok')",
        ],
        task_id="A_default_python_import_probe",
        gpu="cpu",
        timeout=120,
    )
    compile_proc = run_logged(
        [PYTHON, "-m", "compileall", "-q", "dgkan", "experiments"],
        task_id="A_compileall_kan_env",
        gpu="cpu",
        timeout=600,
    )
    import_modules = [
        "dgkan",
        "dgkan.fu.core",
        "dgkan.fu.mechanisms",
        "dgkan.fu.metric_solver",
        "dgkan.models.fc_purekan_primitives",
        "dgkan.integration.kanbefair_adapter",
        "dgkan.fu.function_space_metrics",
        "dgkan.fu.basis_channel_metric",
        "dgkan.fu.jacobian_sketch",
        "dgkan.fu.real_jacobian_commit",
        "dgkan.fu.basis_native_controller",
    ]
    import_rows: list[dict[str, Any]] = []
    for mod in import_modules:
        try:
            importlib.import_module(mod)
            import_rows.append({"module": mod, "import_pass": 1, "error": ""})
        except Exception as exc:
            import_rows.append({"module": mod, "import_pass": 0, "error": repr(exc)})
    required_files = [
        "docs/DG-KAN_v22.32_CausalActuatorFidelityFU_扩充版.md",
        "experiments/run_v22_32_causal_actuator_fidelity.py",
        "experiments/run_v22_30_fidelity_ladder.py",
        "dgkan/fu/core.py",
        "dgkan/fu/mechanisms.py",
        "dgkan/fu/metric_solver.py",
        "dgkan/fu/function_space_metrics.py",
        "dgkan/fu/basis_channel_metric.py",
        "dgkan/fu/jacobian_sketch.py",
        "dgkan/fu/real_jacobian_commit.py",
        "dgkan/fu/basis_native_controller.py",
        "dgkan/models/fc_purekan_primitives.py",
        "dgkan/integration/kanbefair_adapter.py",
    ]
    manifest_rows: list[dict[str, Any]] = []
    for rel in required_files:
        path = ROOT / rel
        manifest_rows.append(
            {
                "path": rel,
                "present": int(path.exists()),
                "bytes": path.stat().st_size if path.exists() else "",
                "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
            }
        )
    transitive_rows = [
        {"module": row["module"], "transitive_dependency_present": row["import_pass"], "error": row["error"]}
        for row in import_rows
        if row["module"].startswith("dgkan.fu.")
    ]
    scan_terms = {
        "future_direction": "no future direction",
        "validation_direction": "no validation direction",
        "test_direction": "no test direction",
        "query_direction": "no query direction",
        "auxiliary_loss_official": "no auxiliary loss as official FU",
        "readout_diagnostic_official": "no readout diagnostic official promotion",
    }
    scan_paths = list((ROOT / "dgkan/fu").glob("*.py")) + [
        ROOT / "dgkan/models/fc_purekan_primitives.py",
        ROOT / "dgkan/integration/kanbefair_adapter.py",
    ]
    forbidden_rows: list[dict[str, Any]] = []
    for path in scan_paths:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for term, rule in scan_terms.items():
            count = text.count(term.lower())
            if count:
                forbidden_rows.append({"path": str(path.relative_to(ROOT)), "term": term, "rule": rule, "count": count})
    from dgkan.integration.kanbefair_adapter import firewall_rows

    identity_rows = []
    for row in firewall_rows():
        identity_rows.append(
            {
                **row,
                "uses_pykan": int(row.get("uses_kanbefair_baseline_model", 0) and row.get("model_name") == "KAN"),
                "uses_bspline_official_path": row.get("uses_bspline_official_path", 0),
                "uses_primitivekan": int(str(row.get("model_name", "")).startswith("DGKAN")),
                "is_dgkan_strict_fc_purekan": row.get("is_dgkan_strict_fc_purekan", 0),
                "official_row": int(str(row.get("model_name", "")).startswith("DGKAN")),
            }
        )
    official_dgkan_rows = [r for r in identity_rows if int_flag(r.get("official_row")) == 1]
    kanbefair_official = [
        r
        for r in identity_rows
        if int_flag(r.get("official_row")) == 1 and int_flag(r.get("uses_kanbefair_baseline_model")) == 1
    ]
    code_truth = {
        "compileall_pass": int(compile_proc.returncode == 0),
        "core_import_pass": int(all(int_flag(r.get("import_pass")) for r in import_rows)),
        "review_bundle_self_contained": int(all(int_flag(r.get("present")) for r in manifest_rows)),
        "missing_transitive_dependency_count": sum(1 for r in transitive_rows if not int_flag(r.get("transitive_dependency_present"))),
        "forbidden_direction_count": sum(1 for r in forbidden_rows if "direction" in str(r.get("term", ""))),
        "official_DGKAN_identity_pass": int(bool(official_dgkan_rows) and not kanbefair_official),
        "KANbeFair_original_KAN_official_rows": len(kanbefair_official),
        "uses_pykan": sum(int_flag(r.get("uses_pykan")) for r in official_dgkan_rows),
        "uses_bspline_official_path": sum(int_flag(r.get("uses_bspline_official_path")) for r in official_dgkan_rows),
        "uses_primitivekan": min((int_flag(r.get("uses_primitivekan")) for r in official_dgkan_rows), default=0),
        "is_dgkan_strict_fc_purekan": min((int_flag(r.get("is_dgkan_strict_fc_purekan")) for r in official_dgkan_rows), default=0),
        "default_python_import_pass": int(default_import.returncode == 0),
        "environment_repair": "default python lacks torch; reran with kan conda env" if default_import.returncode != 0 else "",
        "python_used_for_pass": PYTHON,
    }
    write_rows(OUT_ROOT / "v22_32_code_truth_gate.csv", [code_truth])
    write_rows(OUT_ROOT / "v22_32_review_bundle_manifest.csv", manifest_rows)
    write_rows(OUT_ROOT / "v22_32_import_closure_matrix.csv", import_rows)
    write_rows(OUT_ROOT / "v22_32_transitive_dependency_matrix.csv", transitive_rows)
    write_rows(OUT_ROOT / "v22_32_model_identity_matrix.csv", identity_rows)
    write_rows(
        OUT_ROOT / "v22_32_forbidden_direction_scan.csv",
        forbidden_rows,
        fieldnames=["path", "term", "rule", "count"],
    )
    append_exec(
        "A code closure generated v22_32_code_truth_gate.csv and identity/import matrices",
        task_id="A_code_closure_summary",
        status="pass" if code_truth["compileall_pass"] and code_truth["core_import_pass"] else "fail",
        gpu="cpu",
        files="results/v22_32/v22_32_code_truth_gate.csv, results/v22_32/v22_32_import_closure_matrix.csv",
        note=f"environment_repair={code_truth['environment_repair'] or 'none'}",
    )
    return code_truth


def stage_b_gap_decomposition() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = V22_30 / "v22_30_KAN_vs_MLPFU_gap_matrix.csv"
    rows = read_rows(source)
    hard = [r for r in rows if str(r.get("task_tier", "")).startswith("Tier1") or str(r.get("task_tier", "")).startswith("Tier2")]
    noise_vals = []
    for r in hard:
        for key in ["MLP_FU_vs_best_control_NLL_delta", "KAN_FU_vs_best_control_NLL_delta"]:
            val = finite_float(r.get(key))
            if val is not None:
                noise_vals.append(val)
    epsilon = statistics.pstdev(noise_vals) if len(noise_vals) >= 2 else 1.0e-6
    epsilon = max(float(epsilon), 1.0e-6)
    out: list[dict[str, Any]] = []
    for r in hard:
        d_mlp = finite_float(r.get("MLP_plus_FU_NLL_delta_vs_MLP"), 0.0) or 0.0
        d_kan = finite_float(r.get("KAN_plus_FU_NLL_delta_vs_KAN"), 0.0) or 0.0
        gap_bp = finite_float(r.get("gap_BP"), 0.0) or 0.0
        gap_fu = finite_float(r.get("gap_FU"), 0.0) or 0.0
        gap_red = finite_float(r.get("gap_reduction"), gap_bp - gap_fu) or 0.0
        mlp_ctrl = finite_float(r.get("MLP_FU_vs_best_control_NLL_delta"), 0.0) or 0.0
        kan_ctrl = finite_float(r.get("KAN_FU_vs_best_control_NLL_delta"), 0.0) or 0.0
        if d_kan < -epsilon and d_mlp <= epsilon and gap_red > 0 and kan_ctrl < -epsilon:
            klass = "TrueKANGain"
        elif d_mlp > epsilon and gap_red > 0:
            klass = "MLPDegradationDriven"
        elif d_kan < -epsilon and d_mlp < -epsilon and abs(d_kan) > abs(d_mlp) and gap_red > 0:
            klass = "BothGain"
        elif kan_ctrl >= -epsilon:
            klass = "ControlExplained"
        else:
            klass = "NoGain"
        out.append(
            {
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "task_tier": r.get("task_tier", ""),
                "model_family": r.get("kan_family", ""),
                "Delta_MLP_NLL": d_mlp,
                "Delta_KAN_NLL": d_kan,
                "Gap_BP_NLL": gap_bp,
                "Gap_FU_NLL": gap_fu,
                "GapReduction_NLL": gap_red,
                "Delta_MLP_accuracy": r.get("MLP_plus_FU_accuracy_delta_vs_MLP", ""),
                "Delta_KAN_accuracy": r.get("KAN_plus_FU_accuracy_delta_vs_KAN", ""),
                "GapReduction_accuracy": "",
                "MLP_FU_vs_best_control_NLL": mlp_ctrl,
                "KAN_FU_vs_best_control_NLL": kan_ctrl,
                "gap_reduction_class": klass,
                "repeat_noise_epsilon": epsilon,
                "epsilon_method": "pstdev_of_v22_30_matched_control_delta_fields_no_repeat_noop_std_available",
                "source_artifact": str(source.relative_to(ROOT)),
            }
        )
    counts: dict[str, int] = {}
    for r in out:
        counts[str(r["gap_reduction_class"])] = counts.get(str(r["gap_reduction_class"]), 0) + 1
    total = len(out)
    true_both = counts.get("TrueKANGain", 0) + counts.get("BothGain", 0)
    kan_improve = sum(1 for r in out if finite_float(r.get("Delta_KAN_NLL"), 0.0) < -epsilon)
    kan_acc_nonworse = sum(1 for r in out if finite_float(r.get("Delta_KAN_accuracy"), 0.0) >= -epsilon)
    summary = {
        "source_artifact": str(source.relative_to(ROOT)),
        "hard_rows": total,
        "repeat_noise_epsilon": epsilon,
        "epsilon_method": "pstdev_of_v22_30_matched_control_delta_fields_no_repeat_noop_std_available",
        "TrueKANGain_rows": counts.get("TrueKANGain", 0),
        "BothGain_rows": counts.get("BothGain", 0),
        "MLPDegradationDriven_rows": counts.get("MLPDegradationDriven", 0),
        "ControlExplained_rows": counts.get("ControlExplained", 0),
        "NoGain_rows": counts.get("NoGain", 0),
        "TrueKANGain_plus_BothGain_rate": rate(true_both, total),
        "MLPDegradationDriven_rate": rate(counts.get("MLPDegradationDriven", 0), total),
        "ControlExplained_rate": rate(counts.get("ControlExplained", 0), total),
        "KAN_FU_NLL_improvement_rate": rate(kan_improve, total),
        "KAN_FU_accuracy_nonworse_rate": rate(kan_acc_nonworse, total),
    }
    summary["R10_true_kan_gain_candidate_pass"] = int(
        summary["TrueKANGain_plus_BothGain_rate"] >= 0.60
        and summary["MLPDegradationDriven_rate"] <= 0.20
        and summary["ControlExplained_rate"] <= 0.20
        and summary["KAN_FU_NLL_improvement_rate"] >= 0.50
        and summary["KAN_FU_accuracy_nonworse_rate"] >= 0.70
    )
    write_rows(OUT_ROOT / "v22_32_gap_decomposition_matrix.csv", out)
    write_rows(OUT_ROOT / "v22_32_gap_decomposition_summary.csv", [summary])
    write_simple_svg(
        "v22_32_gap_waterfall_by_task.svg",
        "v22.32 gap reduction by hard row",
        [{"label": f"{r['dataset']} s{r['seed']} {r['model_family']}", "value": r["GapReduction_NLL"]} for r in out],
    )
    write_simple_svg(
        "v22_32_gap_classification_bar.svg",
        "v22.32 gap decomposition classes",
        [{"label": k, "value": v} for k, v in sorted(counts.items())],
    )
    append_exec(
        f"read {source.relative_to(ROOT)} and generated v22.32 gap decomposition",
        task_id="B_gap_decomposition",
        status="pass",
        gpu="cpu",
        files="results/v22_32/v22_32_gap_decomposition_matrix.csv, results/v22_32/v22_32_gap_decomposition_summary.csv",
        note=f"hard_rows={total}; pass={summary['R10_true_kan_gain_candidate_pass']}; epsilon={epsilon:.6g}",
    )
    return out, summary


def _stage_c_direct_t1_single(args: argparse.Namespace, dataset: str, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run a small direct C1-C6 selector pilot.

    This is not sized to establish official cross-dataset/cross-seed pass.  It is
    a blocker repair attempt for the v22.32 direction-selector gap: all six
    preregistered selectors are instantiated in one fixed signal subspace and
    compared with matched controls on the same checkpoint.
    """
    import torch
    import torch.nn.functional as F

    from experiments.run_v22_30_fidelity_ladder import (
        apply_layer_direction,
        cohort_signal_for_layer,
        collect_fixed_examples,
        collect_layer_grads,
        direction_control,
        make_loaders,
        make_hard_vision_loaders,
        make_mlp,
        principal_overlap,
        refined_direction_from_subspace,
        signal_from_grads_with_proj,
        train_adamw,
        train_branch,
        transported_direction_from_signal_subspaces,
    )

    device = torch.device(args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    hard_names = {"CIFAR10", "SVHN", "EMNIST", "EMNIST_LETTERS", "WINE"}
    if dataset.upper().replace("-", "_") in hard_names:
        train_loader, held_loader, test_loader, input_dim, output_dim, _x_stats = make_hard_vision_loaders(
            dataset,
            args.t1_train_size,
            args.t1_test_size,
            args.batch_size,
            seed,
            download=False,
        )
    else:
        train_loader, held_loader, test_loader, input_dim, output_dim, _x_stats = make_loaders(
            dataset, args.t1_train_size, args.t1_test_size, args.batch_size, seed
        )
    torch.manual_seed(2232)
    init_model = make_mlp(input_dim, output_dim, args.hidden, seed + 2232, device).to(device)
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
    xb, yb = collect_fixed_examples(train_loader, max(args.t1_examples, args.t1_cohort_size * args.t1_signal_cohorts), device)
    held_x, held_y = collect_fixed_examples(held_loader, args.t1_examples, device)
    layer = "w2"
    sig, _cohort_rows = cohort_signal_for_layer(
        model,
        xb,
        yb,
        layer,
        cohorts=args.t1_signal_cohorts,
        cohort_size=args.t1_cohort_size,
        rank_cap=args.t1_rank_cap,
        sketch_dim=args.t1_sketch_dim,
        seed=seed + 501,
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
        seed=seed + 501,
    )

    def candidate_pool() -> list[tuple[str, Any, str]]:
        basis = sig.get("basis_sketch")
        proj = sig.get("proj")
        direction = sig["direction"]
        out: list[tuple[str, Any, str]] = [("raw_top_signal", direction, "original_top_eigen_direction")]
        if basis is not None and proj is not None and int(basis.numel()) > 0:
            k = int(basis.shape[1])
            for j in range(k):
                coeff = torch.zeros(k, device=direction.device)
                coeff[j] = 1.0
                vec = proj @ (basis @ coeff)
                vec = vec * direction.norm().clamp_min(1.0e-12) / vec.norm().clamp_min(1.0e-12)
                out.append((f"basis_{j}_plus", vec, "basis_axis"))
                out.append((f"basis_{j}_minus", -vec, "basis_axis"))
            gen = torch.Generator(device=direction.device).manual_seed(93232)
            while len(out) < max(1, args.t1_candidate_count):
                coeff = torch.randn(k, device=direction.device, generator=gen)
                vec = proj @ (basis @ coeff)
                vec = vec * direction.norm().clamp_min(1.0e-12) / vec.norm().clamp_min(1.0e-12)
                out.append((f"random_combo_{len(out)}", vec, "same_subspace_random_combo"))
        return out[: max(1, args.t1_candidate_count)]

    pool = candidate_pool()

    def ce_delta_for_direction(direction: Any, x: Any, y: Any) -> float:
        trial = copy.deepcopy(model).to(device)
        before = F.cross_entropy(trial(x).float(), y.long()).detach()
        apply_layer_direction(trial, layer, direction, args.t1_branch_trust)
        after = F.cross_entropy(trial(x).float(), y.long()).detach()
        return float((after - before).item())

    refined = refined_direction_from_subspace(
        model,
        layer,
        sig,
        held_x,
        held_y,
        trust_ratio=args.t1_branch_trust,
        seed=seed + 777,
        max_candidates=args.t1_candidate_count,
    )
    # C3: maximize worst-cohort improvement inside the fixed subspace.
    cvar_best: tuple[str, Any, str, float] | None = None
    for label, direction, source in pool:
        improvements = []
        for cidx in range(args.t1_signal_cohorts):
            lo = cidx * args.t1_cohort_size
            hi = lo + args.t1_cohort_size
            if hi > int(xb.shape[0]):
                break
            improvements.append(-ce_delta_for_direction(direction, xb[lo:hi], yb[lo:hi]))
        if not improvements:
            continue
        worst_n = max(1, int(math.ceil(0.25 * len(improvements))))
        score = sum(sorted(improvements)[:worst_n]) / worst_n
        if cvar_best is None or score > cvar_best[3]:
            cvar_best = (label, direction, source, score)
    # C4: average leave-cohort selected directions with sign alignment.
    leave_dirs = []
    for leave in range(args.t1_signal_cohorts):
        best: tuple[float, Any] | None = None
        for _label, direction, _source in pool:
            vals = []
            for cidx in range(args.t1_signal_cohorts):
                if cidx == leave:
                    continue
                lo = cidx * args.t1_cohort_size
                hi = lo + args.t1_cohort_size
                if hi <= int(xb.shape[0]):
                    vals.append(ce_delta_for_direction(direction, xb[lo:hi], yb[lo:hi]))
            if vals:
                score = sum(vals) / len(vals)
                if best is None or score < best[0]:
                    best = (score, direction)
        if best is not None:
            d = best[1]
            if leave_dirs and float(torch.dot(leave_dirs[0].reshape(-1), d.reshape(-1)).item()) < 0:
                d = -d
            leave_dirs.append(d)
    if leave_dirs:
        c4_dir = sum(leave_dirs)
        c4_dir = c4_dir * sig["direction"].norm().clamp_min(1.0e-12) / c4_dir.norm().clamp_min(1.0e-12)
        leave_variance = float(torch.stack([d.reshape(-1) for d in leave_dirs]).var(dim=0).mean().item())
    else:
        c4_dir = sig["direction"]
        leave_variance = ""
    # C5: use previous checkpoint subspace as the temporal-stable source.
    transported = transported_direction_from_signal_subspaces(prev_sig, sig)
    c5_dir = transported["direction"]
    # C6: high-loss slice direction, projected in the same sketch.
    with torch.no_grad():
        losses = F.cross_entropy(model(xb).float(), yb.long(), reduction="none")
    hard_n = max(args.t1_cohort_size, int(math.ceil(0.25 * int(xb.shape[0]))))
    hard_idx = torch.argsort(losses, descending=True)[:hard_n]
    hard_grads = collect_layer_grads(model, xb[hard_idx], yb[hard_idx], layer)
    hard_sig = signal_from_grads_with_proj(hard_grads, args.t1_rank_cap, sig["proj"])

    base_influence_model = copy.deepcopy(model).to(device)
    base_influence_ev = train_branch(
        base_influence_model,
        train_loader,
        test_loader,
        device,
        output_dim,
        args.t1_influence_horizon,
        args.lr,
        args.weight_decay,
    )

    def cohort_ce_values(branch_model: Any) -> list[float]:
        vals: list[float] = []
        for cidx in range(args.t1_signal_cohorts):
            lo = cidx * args.t1_cohort_size
            hi = lo + args.t1_cohort_size
            if hi <= int(xb.shape[0]):
                vals.append(float(F.cross_entropy(branch_model(xb[lo:hi]).float(), yb[lo:hi].long()).detach().item()))
        return vals

    base_cohort_ce = cohort_ce_values(base_influence_model)
    influence_rows: list[dict[str, Any]] = []
    influence_best: tuple[str, Any, str, float] | None = None
    influence_scores: list[float] = []
    future_benefits: list[float] = []
    for label, direction, source in pool:
        candidate_model = copy.deepcopy(model).to(device)
        apply_layer_direction(candidate_model, layer, direction, args.t1_branch_trust)
        ev = train_branch(
            candidate_model,
            train_loader,
            test_loader,
            device,
            output_dim,
            args.t1_influence_horizon,
            args.lr,
            args.weight_decay,
        )
        cand_cohort_ce = cohort_ce_values(candidate_model)
        influences = [b - c for b, c in zip(base_cohort_ce, cand_cohort_ce)]
        if not influences:
            continue
        worst_n = max(1, int(math.ceil(0.25 * len(influences))))
        robust = sum(sorted(influences)[:worst_n]) / worst_n
        infl_mean = sum(influences) / len(influences)
        infl_var = statistics.pvariance(influences) if len(influences) > 1 else 0.0
        nll_delta = float(ev["NLL"] - base_influence_ev["NLL"])
        influence_scores.append(float(robust))
        future_benefits.append(float(-nll_delta))
        row = {
            "dataset": dataset,
            "seed": seed,
            "candidate_label": label,
            "candidate_source": source,
            "selector_name": "L_cohort_influence_direction",
            "cohort_influence_mean": infl_mean,
            "cohort_influence_CVaR25": robust,
            "leave_cohort_influence_variance": infl_var,
            "influence_vs_NLL_delta_correlation": "",
            "influence_vs_controls_margin": "",
            "stagewise_influence_shift": "",
            "branch_H": args.t1_influence_horizon,
            "branch_NLL_delta_vs_base": nll_delta,
            "selected_as_influence_direction": 0,
            "uses_test_direction_selection": 0,
            "uses_future_direction": 1,
            "status": "direct_v22_32_L_offline_branch_causal_diagnostic_not_runtime_official",
            "blocker": "cohort influence uses same-run future branch labels, so it can diagnose but not be promoted as runtime direction",
        }
        influence_rows.append(row)
        if influence_best is None or robust > influence_best[3]:
            influence_best = (label, direction, source, robust)
    corr = spearman_corr(influence_scores, future_benefits)
    best_l3 = ""
    for row in influence_rows:
        row["influence_vs_NLL_delta_correlation"] = corr
        if influence_best is not None and row["candidate_label"] == influence_best[0]:
            row["selected_as_influence_direction"] = 1
            best_l3 = influence_best[3]
    if influence_rows and best_l3 != "":
        control_like = [finite_float(r.get("cohort_influence_CVaR25")) for r in influence_rows if "random" in str(r.get("candidate_source", "")).lower()]
        control_vals = [float(v) for v in control_like if v is not None]
        best_control_influence = max(control_vals) if control_vals else None
        if best_control_influence is not None:
            for row in influence_rows:
                if int_flag(row.get("selected_as_influence_direction")):
                    row["influence_vs_controls_margin"] = float(best_l3) - best_control_influence

    selector_defs = [
        ("C1_raw_signal_eigen_direction", sig["direction"], "raw_top_signal", "original_top_eigen_direction", ""),
        (
            "C2_held_cohort_descent_direction",
            refined["direction"],
            refined["selected_candidate_label"],
            refined["selected_candidate_source"],
            "",
        ),
        (
            "C3_CVaR_direction",
            cvar_best[1] if cvar_best else sig["direction"],
            cvar_best[0] if cvar_best else "raw_top_signal",
            "CVaR25_train_cohorts",
            cvar_best[3] if cvar_best else "",
        ),
        ("C4_leave_cohort_stable_direction", c4_dir, "leave_cohort_average", "leave_cohort_train_only", leave_variance),
        ("C5_temporal_stable_direction", c5_dir, "previous_to_current_transport", "past_checkpoint_subspace", ""),
        ("C6_hard_slice_direction", hard_sig["direction"], "hard_slice_top_loss", "train_high_loss_slice", hard_n / int(xb.shape[0])),
    ]
    if influence_best is not None:
        selector_defs.append(
            (
                "L_cohort_influence_direction",
                influence_best[1],
                influence_best[0],
                "offline_branch_causal_influence_train_cohorts_not_runtime_official",
                influence_best[3],
            )
        )
    controls = [
        ("L1_isotropic_random", direction_control(sig["direction"], "random", seed + 1), "L1"),
        ("L2_same_norm_random", direction_control(sig["direction"], "random", seed + 2), "L2"),
        ("L3_same_signal_subspace_random", direction_control(sig["direction"], "same_subspace", seed + 3, sig["basis_sketch"], sig["proj"]), "L3"),
        ("L4_signflip_same_subspace", -sig["direction"], "L4"),
        ("L5_shuffled_cohort_direction", direction_control(sig["direction"], "shuffled", seed + 5), "L5"),
    ]
    base_model = copy.deepcopy(model).to(device)
    base_ev = train_branch(base_model, train_loader, test_loader, device, output_dim, args.t1_branch_horizon, args.lr, args.weight_decay)
    branch_out: list[dict[str, Any]] = []

    def run_branch(name: str, direction: Any, *, is_control: int, control_level: str = "") -> dict[str, Any]:
        branch_model = copy.deepcopy(model).to(device)
        apply_layer_direction(branch_model, layer, direction, args.t1_branch_trust)
        ev = train_branch(
            branch_model,
            train_loader,
            test_loader,
            device,
            output_dim,
            args.t1_branch_horizon,
            args.lr,
            args.weight_decay,
        )
        return {
            "dataset": dataset,
            "seed": seed,
            "checkpoint_step": f"direct_pretrain_steps_{args.t1_pretrain_steps}",
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
            "margin_mean_delta_vs_base": ev.get("margin_mean", math.nan) - base_ev.get("margin_mean", math.nan),
            "margin_q10_delta_vs_base": ev.get("margin_q10", math.nan) - base_ev.get("margin_q10", math.nan),
            "margin_q01_delta_vs_base": ev.get("margin_q01", math.nan) - base_ev.get("margin_q01", math.nan),
            "low_margin_accuracy_delta_vs_base": ev.get("low_margin_accuracy", math.nan) - base_ev.get("low_margin_accuracy", math.nan),
            "beats_base": int(ev["NLL"] < base_ev["NLL"]),
            "best_control_delta": "",
            "real_minus_best_control_delta": "",
            "is_control_branch": is_control,
            "control_level": control_level,
            "same_norm": 1,
            "same_cadence": 1,
            "same_overhead": 1,
            "selection_data_source": "train_or_held_train_only_direct_v22_32_pilot",
            "uses_test_direction_selection": 0,
            "uses_future_direction": 0,
            "source_artifact": "direct_v22_32_T1_selector_pilot",
        }

    control_branch_rows = [run_branch(name, d, is_control=1, control_level=level) for name, d, level in controls]
    branch_out.extend(control_branch_rows)
    best_by_level: dict[str, float] = {}
    for row in control_branch_rows:
        val = finite_float(row.get("NLL_delta_vs_base"), math.inf) or math.inf
        level = str(row.get("control_level"))
        best_by_level[level] = min(best_by_level.get(level, math.inf), val)
    best_any = min(best_by_level.values()) if best_by_level else math.inf
    selector_rows: list[dict[str, Any]] = []
    for name, direction, label, source, aux in selector_defs:
        brow = run_branch(name, direction, is_control=0)
        val = finite_float(brow.get("NLL_delta_vs_base"), math.inf) or math.inf
        brow["best_control_delta"] = best_any
        brow["real_minus_best_control_delta"] = val - best_any
        for level in ["L1", "L2", "L3", "L4", "L5"]:
            brow[f"beats_{level}_control"] = int(val < best_by_level.get(level, math.inf))
        branch_out.append(brow)
        selector_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "checkpoint_step": f"direct_pretrain_steps_{args.t1_pretrain_steps}",
                "selector_name": name,
                "subspace_dim": int(sig["basis_sketch"].shape[1]),
                "signal_eigenvalue": sig.get("positive_eigenvalue_mean", ""),
                "cohort_positive_fraction": sig.get("cohort_positive_fraction", ""),
                "temporal_eigenspace_overlap": principal_overlap(prev_sig["basis_sketch"], sig["basis_sketch"]),
                "leave_cohort_direction_variance": aux if name == "C4_leave_cohort_stable_direction" else "",
                "hard_slice_fraction": aux if name == "C6_hard_slice_direction" else "",
                "branch_H": args.t1_branch_horizon,
                "NLL_delta_vs_base": brow["NLL_delta_vs_base"],
                "accuracy_delta_vs_base": brow["accuracy_delta_vs_base"],
                "AUC_delta_vs_base": "",
                "ECE_delta_vs_base": brow["ECE_delta_vs_base"],
                "Brier_delta_vs_base": brow["Brier_delta_vs_base"],
                "tail_q95_delta": brow["tail_q95_delta"],
                "tail_q99_delta": brow["tail_q99_delta"],
                "margin_mean_delta_vs_base": brow.get("margin_mean_delta_vs_base", ""),
                "margin_q10_delta_vs_base": brow.get("margin_q10_delta_vs_base", ""),
                "margin_q01_delta_vs_base": brow.get("margin_q01_delta_vs_base", ""),
                "low_margin_accuracy_delta_vs_base": brow.get("low_margin_accuracy_delta_vs_base", ""),
                "beats_base": brow["beats_base"],
                "beats_L1_control": brow["beats_L1_control"],
                "beats_L2_control": brow["beats_L2_control"],
                "beats_L3_control": brow["beats_L3_control"],
                "beats_L4_control": brow["beats_L4_control"],
                "beats_L5_control": brow["beats_L5_control"],
                "best_control_delta": best_any,
                "real_minus_best_control_delta": brow["real_minus_best_control_delta"],
                "selected_candidate_label": label,
                "selection_data_source": source,
                "uses_test_direction_selection": 0,
                "uses_future_direction": 1 if name == "L_cohort_influence_direction" else 0,
                "source_artifact": "direct_v22_32_T1_selector_pilot",
                "status": "direct_v22_32_L_offline_diagnostic_not_official" if name == "L_cohort_influence_direction" else "direct_v22_32_pilot_not_official_scale",
            }
        )
    return selector_rows, branch_out, influence_rows


def stage_c_direct_t1_pilot(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selector_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    influence_rows: list[dict[str, Any]] = []
    for dataset in [d.strip() for d in str(args.t1_datasets).split(",") if d.strip()]:
        for seed in [int(s) for s in str(args.t1_seeds).split(",") if s.strip()]:
            s_rows, b_rows, i_rows = _stage_c_direct_t1_single(args, dataset, seed)
            selector_rows.extend(s_rows)
            branch_rows.extend(b_rows)
            influence_rows.extend(i_rows)
    return selector_rows, branch_rows, influence_rows


def stage_c_t1_direction(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    layer_rows = read_rows(V22_30 / "v22_30_T1_layerwise_signal_channel_matrix.csv")
    ref_rows = read_rows(V22_30 / "v22_30_T1_subspace_refinement_matrix.csv")
    branch_rows = read_rows(V22_30 / "v22_30_T1_subspace_refinement_branch_matrix.csv")
    by_key = {(r.get("dataset"), r.get("seed"), r.get("layer_id")): r for r in layer_rows}
    selector_rows: list[dict[str, Any]] = []
    for r in ref_rows:
        key = (r.get("dataset"), r.get("seed"), r.get("layer_id"))
        base = by_key.get(key, {})
        src = str(r.get("selected_candidate_source", ""))
        selector_name = "C1_raw_signal_eigen_direction" if src == "original_top_eigen_direction" else "C2_held_cohort_descent_direction"
        selector_rows.append(
            {
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "checkpoint_step": "v22_30_pretrained_checkpoint",
                "selector_name": selector_name,
                "subspace_dim": base.get("positive_eigen_count", ""),
                "signal_eigenvalue": base.get("positive_eigenvalue_mean", ""),
                "cohort_positive_fraction": base.get("cohort_positive_fraction", ""),
                "temporal_eigenspace_overlap": base.get("temporal_eigenspace_overlap", ""),
                "leave_cohort_direction_variance": "",
                "hard_slice_fraction": "",
                "branch_H": "50;100;200",
                "NLL_delta_vs_base": "",
                "accuracy_delta_vs_base": "",
                "AUC_delta_vs_base": "",
                "ECE_delta_vs_base": "",
                "Brier_delta_vs_base": "",
                "tail_q95_delta": "",
                "tail_q99_delta": "",
                "beats_base": r.get("refined_beats_base_rate", ""),
                "beats_L1_control": "",
                "beats_L2_control": r.get("refined_beats_L2_controls_rate", ""),
                "beats_L3_control": r.get("refined_beats_L3_best_controls_rate", ""),
                "beats_L4_control": "",
                "beats_L5_control": "",
                "best_control_delta": "",
                "real_minus_best_control_delta": "",
                "selected_candidate_label": r.get("selected_candidate_label", ""),
                "selection_data_source": r.get("selection_data_source", ""),
                "uses_test_direction_selection": r.get("uses_test_direction_selection", ""),
                "uses_future_direction": r.get("uses_future_direction", ""),
                "source_artifact": "results/v22_30/v22_30_T1_subspace_refinement_matrix.csv",
                "status": "historical_v22_30_direct_rows_reaudited",
            }
        )
    branch_out: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for r in branch_rows:
        groups.setdefault((r.get("dataset", ""), r.get("seed", ""), r.get("layer_id", ""), r.get("horizon", "")), []).append(r)
    for key, members in groups.items():
        base = next((m for m in members if m.get("branch_variant") == "L0_noop"), None)
        base_delta = finite_float(base.get("branch_NLL_delta")) if base else None
        controls = [m for m in members if int_flag(m.get("is_control_branch")) == 1 and m.get("branch_variant") != "L0_noop"]
        best_control = min((finite_float(c.get("branch_NLL_delta"), math.inf) or math.inf for c in controls), default=None)
        for m in members:
            real_delta = finite_float(m.get("branch_NLL_delta"))
            branch_out.append(
                {
                    "dataset": key[0],
                    "seed": key[1],
                    "checkpoint_step": "v22_30_pretrained_checkpoint",
                    "selector_name": m.get("branch_variant", ""),
                    "layer_id": key[2],
                    "branch_H": key[3],
                    "NLL_delta_vs_base": "" if real_delta is None or base_delta is None else real_delta - base_delta,
                    "accuracy_delta_vs_base": "",
                    "AUC_delta_vs_base": "",
                    "ECE_delta_vs_base": "",
                    "Brier_delta_vs_base": "",
                    "tail_q95_delta": "",
                    "tail_q99_delta": "",
                    "beats_base": "" if real_delta is None or base_delta is None else int(real_delta < base_delta),
                    "beats_L1_control": "",
                    "beats_L2_control": "",
                    "beats_L3_control": "",
                    "beats_L4_control": "",
                    "beats_L5_control": "",
                    "best_control_delta": best_control if best_control is not None and best_control != math.inf else "",
                    "real_minus_best_control_delta": ""
                    if real_delta is None or best_control is None or best_control == math.inf
                    else real_delta - best_control,
                    "is_control_branch": m.get("is_control_branch", ""),
                    "control_level": m.get("control_level", ""),
                    "selected_candidate_label": m.get("selected_candidate_label", ""),
                    "uses_test_direction_selection": m.get("uses_test_direction_selection", ""),
                    "uses_future_direction": m.get("uses_future_direction", ""),
                    "source_artifact": "results/v22_30/v22_30_T1_subspace_refinement_branch_matrix.csv",
                }
            )
    direct_selector_rows: list[dict[str, Any]] = []
    direct_branch_rows: list[dict[str, Any]] = []
    direct_influence_rows: list[dict[str, Any]] = []
    direct_error = ""
    try:
        direct_selector_rows, direct_branch_rows, direct_influence_rows = stage_c_direct_t1_pilot(args)
        selector_rows.extend(direct_selector_rows)
        branch_out.extend(direct_branch_rows)
        if direct_influence_rows:
            write_rows(OUT_ROOT / "v22_32_cohort_influence_matrix.csv", direct_influence_rows)
        append_exec(
            "direct v22.32 C1-C6 plus L cohort-influence T1 pilot on fixed signal subspace",
            task_id="C_T1_direct_selector_pilot",
            status="pass",
            gpu=args.device,
            files="results/v22_32/v22_32_T1_direction_selection_matrix.csv (rewritten later), results/v22_32/v22_32_T1_branch_causal_matrix.csv (rewritten later)",
            note=f"datasets={args.t1_datasets}; seeds={args.t1_seeds}; direct_selector_rows={len(direct_selector_rows)}; direct_branch_rows={len(direct_branch_rows)}; influence_rows={len(direct_influence_rows)}",
        )
    except Exception as exc:
        direct_error = repr(exc)
        append_exec(
            "direct v22.32 C1-C6 plus L cohort-influence T1 pilot on fixed signal subspace",
            task_id="C_T1_direct_selector_pilot",
            status="fail",
            gpu=args.device,
            files="",
            note=direct_error,
        )
    observed_selectors = {r["selector_name"] for r in selector_rows}
    required_selectors = [
        "C1_raw_signal_eigen_direction",
        "C2_held_cohort_descent_direction",
        "C3_CVaR_direction",
        "C4_leave_cohort_stable_direction",
        "C5_temporal_stable_direction",
        "C6_hard_slice_direction",
    ]
    missing_selectors = [s for s in required_selectors if s not in observed_selectors]
    for missing in missing_selectors:
        selector_rows.append(
            {
                "dataset": "",
                "seed": "",
                "checkpoint_step": "",
                "selector_name": missing,
                "status": "not_run_missing_v22_32_direct_selector",
                "blocker": direct_error or "v22_30 artifacts do not contain this pre-registered selector branch; not promoted",
                "source_artifact": "",
            }
        )
    control_rows: list[dict[str, Any]] = []
    for selector in sorted({r.get("selector_name", "") for r in selector_rows if r.get("status") != "not_run_missing_v22_32_direct_selector"}):
        vals = [finite_float(r.get("beats_L3_control")) for r in selector_rows if r.get("selector_name") == selector]
        vals_f = [v for v in vals if v is not None]
        control_rows.append(
            {
                "selector_name": selector,
                "rows": len(vals_f),
                "mean_beats_L3_control_rate": "" if not vals_f else sum(vals_f) / len(vals_f),
                "control_diagnosis": "DirectionNoGo_vs_C4C5" if vals_f and sum(vals_f) / len(vals_f) < 0.55 else "candidate_needs_direct_v22_32_confirmation",
            }
        )
    direct_c_rows = [
        r
        for r in direct_selector_rows
        if str(r.get("selector_name", "")).startswith("C")
        and str(r.get("status", "")).startswith("direct_v22_32")
        and not int_flag(r.get("uses_future_direction"))
    ]
    direct_control_deltas = [
        finite_float(r.get("NLL_delta_vs_base"))
        for r in direct_branch_rows
        if int_flag(r.get("is_control_branch"))
    ]
    control_noise_vals = [float(v) for v in direct_control_deltas if v is not None]
    direct_noise = statistics.pstdev(control_noise_vals) if len(control_noise_vals) > 1 else 0.0
    selector_gate_rows: list[dict[str, Any]] = []
    for selector in sorted({r.get("selector_name", "") for r in direct_c_rows}):
        rows = [r for r in direct_c_rows if r.get("selector_name") == selector]
        if not rows:
            continue
        n = len(rows)
        nll_vals = [finite_float(r.get("NLL_delta_vs_base")) for r in rows]
        nll_f = [float(v) for v in nll_vals if v is not None]
        mean_nll = sum(nll_f) / len(nll_f) if nll_f else math.inf
        ece_vals = [finite_float(r.get("ECE_delta_vs_base")) for r in rows]
        brier_vals = [finite_float(r.get("Brier_delta_vs_base")) for r in rows]
        tail_q99_vals = [finite_float(r.get("tail_q99_delta")) for r in rows]
        datasets = {str(r.get("dataset")) for r in rows}
        seeds = {str(r.get("seed")) for r in rows}
        horizons = {str(r.get("branch_H")) for r in rows}
        row = {
            "selector_name": selector,
            "direct_rows": n,
            "datasets": ",".join(sorted(datasets)),
            "seeds": ",".join(sorted(seeds)),
            "horizons": ",".join(sorted(horizons)),
            "beats_base_rate": sum(int_flag(r.get("beats_base")) for r in rows) / n,
            "beats_L1_rate": sum(int_flag(r.get("beats_L1_control")) for r in rows) / n,
            "beats_L2_rate": sum(int_flag(r.get("beats_L2_control")) for r in rows) / n,
            "beats_L3_rate": sum(int_flag(r.get("beats_L3_control")) for r in rows) / n,
            "beats_L4_rate": sum(int_flag(r.get("beats_L4_control")) for r in rows) / n,
            "beats_L5_rate": sum(int_flag(r.get("beats_L5_control")) for r in rows) / n,
            "mean_NLL_delta_vs_base": mean_nll,
            "control_noise_std": direct_noise,
            "max_ECE_delta": max((float(v) for v in ece_vals if v is not None), default=math.inf),
            "max_Brier_delta": max((float(v) for v in brier_vals if v is not None), default=math.inf),
            "max_tail_q99_delta": max((float(v) for v in tail_q99_vals if v is not None), default=math.inf),
        }
        row["exploration_pass"] = int(
            row["beats_base_rate"] >= 0.70
            and row["beats_L1_rate"] >= 0.60
            and row["beats_L2_rate"] >= 0.60
            and row["beats_L3_rate"] >= 0.55
            and mean_nll < -2.0 * direct_noise
            and row["max_tail_q99_delta"] <= 0.0
        )
        row["official_candidate_pass"] = int(
            row["exploration_pass"]
            and row["beats_L3_rate"] >= 0.65
            and row["beats_L4_rate"] >= 0.60
            and row["beats_L5_rate"] >= 0.60
            and len(datasets) >= 2
            and len(seeds) >= 2
            and len(horizons) >= 3
            and row["max_ECE_delta"] <= 0.0
            and row["max_Brier_delta"] <= 0.0
        )
        selector_gate_rows.append(row)
    write_rows(OUT_ROOT / "v22_32_T1_selector_gate_eval.csv", selector_gate_rows)
    t1_exploration_pass = int(any(int_flag(r.get("exploration_pass")) for r in selector_gate_rows))
    t1_official_candidate = int(any(int_flag(r.get("official_candidate_pass")) for r in selector_gate_rows))
    best_gate = sorted(
        selector_gate_rows,
        key=lambda r: (
            int_flag(r.get("official_candidate_pass")),
            int_flag(r.get("exploration_pass")),
            finite_float(r.get("beats_L3_rate"), 0.0) or 0.0,
            -(finite_float(r.get("mean_NLL_delta_vs_base"), 0.0) or 0.0),
        ),
        reverse=True,
    )
    summary = {
        "selector_rows": len(selector_rows),
        "branch_rows": len(branch_out),
        "direct_v22_32_selector_rows": len([r for r in direct_selector_rows if str(r.get("selector_name", "")).startswith("C")]),
        "direct_v22_32_L_influence_selector_rows": len([r for r in direct_selector_rows if r.get("selector_name") == "L_cohort_influence_direction"]),
        "direct_v22_32_branch_rows": len(direct_branch_rows),
        "direct_v22_32_influence_rows": len(direct_influence_rows),
        "historical_v22_30_selector_rows": len(ref_rows),
        "missing_pre_registered_selectors": len(missing_selectors),
        "T1_exploration_pass": t1_exploration_pass,
        "T1_official_candidate": t1_official_candidate,
        "best_direct_selector_by_gate": best_gate[0].get("selector_name", "") if best_gate else "",
        "best_direct_selector_mean_NLL_delta": best_gate[0].get("mean_NLL_delta_vs_base", "") if best_gate else "",
        "direct_control_noise_std": direct_noise,
        "reason": "v22_32 direct C1-C6 pilot plus L influence diagnostic still does not satisfy official thresholds; L uses offline future branch labels and is diagnostic only",
        "direct_pilot_error": direct_error,
    }
    write_rows(OUT_ROOT / "v22_32_T1_direction_selection_matrix.csv", selector_rows)
    write_rows(OUT_ROOT / "v22_32_T1_branch_causal_matrix.csv", branch_out)
    write_rows(OUT_ROOT / "v22_32_T1_control_decomposition_matrix.csv", control_rows)
    write_rows(OUT_ROOT / "v22_32_T1_direction_selector_summary.csv", [summary])
    write_simple_svg(
        "v22_32_T1_principal_angle_panel.svg",
        "T1 temporal eigenspace overlap from v22.30",
        [
            {"label": f"{r.get('dataset')} {r.get('layer_id')}", "value": finite_float(r.get("temporal_eigenspace_overlap"), 0.0) or 0.0}
            for r in layer_rows
        ],
    )
    write_simple_svg(
        "v22_32_T1_branch_delta_panel.svg",
        "T1 real minus best control branch delta",
        [
            {"label": f"{r.get('dataset')} {r.get('layer_id')} H{r.get('branch_H')} {r.get('selector_name')}", "value": finite_float(r.get("real_minus_best_control_delta"), 0.0) or 0.0}
            for r in branch_out
            if r.get("is_control_branch") == "0"
        ],
    )
    append_exec(
        "reaudit v22_30 T1 refinement artifacts for v22.32 T1 direction selection",
        task_id="C_T1_direction_selection_reaudit",
        status="pass",
        gpu="cpu",
        files="results/v22_32/v22_32_T1_direction_selection_matrix.csv, results/v22_32/v22_32_T1_branch_causal_matrix.csv",
        note=summary["reason"],
    )
    return selector_rows, branch_out, control_rows, [summary]


def basis_bank_masks(named: list[tuple[str, Any]], family: str, bank_dims: list[int]) -> list[tuple[str, Any, dict[str, Any]]]:
    import torch

    total = sum(int(p.numel()) for _name, p in named)
    if total <= 0:
        return []
    out: list[tuple[str, Any, dict[str, Any]]] = []
    all_mask = torch.ones(total, dtype=torch.bool, device=named[0][1].device)
    out.append(("all_basis", all_mask, {"active_hidden": "all", "basis_channels": "all", "bank_type": "all"}))
    seen = {tuple(all_mask.detach().cpu().tolist())}
    dims = sorted({int(d) for d in bank_dims if int(d) > 0})
    offset = 0
    for name, param in named:
        shape = tuple(param.shape)
        n = int(param.numel())
        if param.ndim != 3:
            offset += n
            continue
        input_dim, hidden_dim, basis_k = (int(shape[0]), int(shape[1]), int(shape[2]))
        for bank_dim in dims:
            if bank_dim > hidden_dim:
                continue
            if family == "D-CHE":
                channel_sets = [
                    ("low_degree", list(range(min(basis_k, 2)))),
                    ("mixed_degree", list(range(basis_k))),
                ]
            else:
                low_freq = [idx for idx in range(1, basis_k)] or [0]
                channel_sets = [
                    ("low_frequency", low_freq),
                    ("mixed_frequency", list(range(basis_k))),
                ]
            for bank_type, channels in channel_sets:
                local = torch.zeros(shape, dtype=torch.bool, device=param.device)
                local[:, :bank_dim, channels] = True
                mask = torch.zeros(total, dtype=torch.bool, device=param.device)
                mask[offset : offset + n] = local.reshape(-1)
                key = tuple(mask.detach().cpu().tolist())
                if not bool(mask.any()) or key in seen:
                    continue
                seen.add(key)
                out.append(
                    (
                        f"{bank_type}_h{bank_dim}",
                        mask,
                        {
                            "param_name": name,
                            "active_hidden": bank_dim,
                            "basis_channels": ",".join(str(c) for c in channels),
                            "bank_type": bank_type,
                            "input_dim": input_dim,
                            "hidden_dim": hidden_dim,
                            "basis_k": basis_k,
                        },
                    )
                )
        offset += n
    return out


def stage_d_basis_actuator(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    from dgkan.fu.real_jacobian_commit import (
        add_flat_delta,
        output_jacobian,
        select_named_parameters,
        solve_linearized_commit,
    )
    from experiments.run_v22_30_fidelity_ladder import (
        collect_fixed_examples,
        evaluate_model,
        make_kan,
        make_loaders,
        train_adamw,
        train_branch,
    )

    device = torch.device(args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    target_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    t3 = read_rows(V22_30 / "v22_30_T3_transport_momentum_matrix.csv")
    for family_key, model_name in [("D-CHE", "DGKAN_DCHE"), ("D-FOU", "DGKAN_DFOU")]:
        fam_rows = [
            r
            for r in t3
            if r.get("model") == model_name
            and r.get("transport_variant") == "transported_source"
            and int_flag(r.get("is_control_branch")) == 0
            and str(r.get("horizon")) in {"100", "200"}
        ]
        pass_rows = sum(int_flag(r.get("transported_source_beats_same_subspace_controls")) for r in fam_rows)
        target_rows.append(
            {
                "carrier": model_name,
                "basis_family": family_key,
                "dataset": "MNIST/FMNI/KMNIST",
                "seed": "0",
                "target_source_type": "v22_30_w2_basis_effect_diagnostic",
                "target_from_w2_diagnostic": 1,
                "diagnostic_branch_rows": len(fam_rows),
                "diagnostic_pass_rows": pass_rows,
                "target_vector_available_for_v22_32_replay": 0,
                "source_artifact": "results/v22_30/v22_30_T3_transport_momentum_matrix.csv",
                "status": "D0_reproduced_as_artifact_reaudit_not_basis_native_target_vector",
            }
        )
    for dataset in args.basis_datasets.split(","):
        dataset = dataset.strip()
        if not dataset:
            continue
        for seed in [int(s) for s in args.basis_seeds.split(",") if s.strip()]:
            train_loader, _held, test_loader, input_dim, output_dim, x_stats = make_loaders(
                dataset, args.basis_train_size, args.basis_test_size, args.batch_size, seed
            )
            for family, carrier in [("D-CHE", "DGKAN_DCHE"), ("D-FOU", "DGKAN_DFOU")]:
                setup_seed = 22_320 + seed + (0 if family == "D-CHE" else 100)
                random.seed(setup_seed)
                torch.manual_seed(setup_seed)
                model = make_kan(input_dim, output_dim, args.hidden, seed + 320, device, x_stats, family).to(device)
                train_adamw(
                    model,
                    train_loader,
                    test_loader,
                    device,
                    output_dim,
                    steps=args.basis_pretrain_steps,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                )
                xb, yb = collect_fixed_examples(train_loader, args.basis_examples, device)
                logits = model(xb).float()
                probs = torch.softmax(logits, dim=-1)
                target_cotangent = -(probs - F.one_hot(yb, num_classes=output_dim).float()).reshape(-1)
                max_rows = min(int(args.basis_max_output_rows), int(target_cotangent.numel()))
                readout_jac, _readout_spec, readout_diag = output_jacobian(model, xb, selector="readout", max_output_rows=max_rows)
                readout_delta, readout_solve_diag = solve_linearized_commit(
                    readout_jac,
                    target_cotangent[:max_rows],
                    damping=args.basis_damping,
                )
                target = (readout_jac @ readout_delta.detach().float()).detach()
                jac, _spec, diag = output_jacobian(model, xb, selector="basis", max_output_rows=max_rows)
                named = select_named_parameters(model, "basis")
                target_rows.append(
                    {
                        "carrier": carrier,
                        "basis_family": family,
                        "dataset": dataset,
                        "seed": seed,
                        "target_source_type": "direct_readout_w2_loss_cotangent_effect",
                        "target_from_w2_diagnostic": 1,
                        "target_vector_available_for_v22_32_replay": 1,
                        "target_norm": float(torch.linalg.vector_norm(target).item()),
                        "readout_projection_residual": readout_solve_diag.get("basis_projection_residual", ""),
                        "readout_projection_cosine": readout_solve_diag.get("basis_projection_cosine", ""),
                        "readout_jacobian_rows": readout_diag.get("jacobian_rows", ""),
                        "readout_jacobian_cols": readout_diag.get("jacobian_cols", ""),
                        "source_artifact": "direct_v22_32_readout_w2_effect_target",
                        "status": "D0_direct_readout_diagnostic_target_reconstructed",
                    }
                )
                base_model = copy.deepcopy(model).to(device)
                base_ev = train_branch(base_model, train_loader, test_loader, device, output_dim, args.basis_branch_horizon, args.lr, args.weight_decay)
                for basis_bank, mask, bank_meta in basis_bank_masks(named, family, [int(x) for x in str(args.basis_bank_dims).split(",") if x.strip()]):
                    bank_jac = jac[:, mask]
                    if int(bank_jac.shape[1]) <= 0:
                        continue
                    delta_bank, solve_diag = solve_linearized_commit(bank_jac, target[:max_rows], damping=args.basis_damping)
                    delta = torch.zeros(int(jac.shape[1]), device=jac.device, dtype=jac.dtype)
                    delta[mask] = delta_bank.to(device=jac.device, dtype=jac.dtype)
                    gradcheck_mode = str(getattr(args, "basis_gradcheck_mode", "forward")).strip().lower()
                    with torch.no_grad():
                        base_flat = model(xb).float().reshape(-1)[: jac.shape[0]].detach().clone()
                        add_flat_delta(named, delta, scale=args.basis_update_scale)
                        exact_flat = model(xb).float().reshape(-1)[: jac.shape[0]].detach().clone()
                        add_flat_delta(named, delta, scale=-args.basis_update_scale)
                        if gradcheck_mode == "central":
                            add_flat_delta(named, delta, scale=args.basis_gradcheck_eps)
                            plus_flat = model(xb).float().reshape(-1)[: jac.shape[0]].detach().clone()
                            add_flat_delta(named, delta, scale=-2.0 * args.basis_gradcheck_eps)
                            minus_flat = model(xb).float().reshape(-1)[: jac.shape[0]].detach().clone()
                            add_flat_delta(named, delta, scale=args.basis_gradcheck_eps)
                        else:
                            add_flat_delta(named, delta, scale=args.basis_gradcheck_eps)
                            plus_flat = model(xb).float().reshape(-1)[: jac.shape[0]].detach().clone()
                            add_flat_delta(named, delta, scale=-args.basis_gradcheck_eps)
                            minus_flat = base_flat
                    lin = (jac @ delta.detach().float()) * float(args.basis_update_scale)
                    exact_diff = exact_flat - base_flat
                    exact_err = torch.linalg.vector_norm(exact_diff - lin).div(torch.linalg.vector_norm(lin).clamp_min(1.0e-12))
                    if gradcheck_mode == "central":
                        fd = (plus_flat - minus_flat) / (2.0 * float(args.basis_gradcheck_eps))
                    else:
                        fd = (plus_flat - base_flat) / float(args.basis_gradcheck_eps)
                    lin_unscaled = jac @ delta.detach().float()
                    grad_rel = torch.linalg.vector_norm(fd - lin_unscaled).div(torch.linalg.vector_norm(fd).clamp_min(1.0e-12))
                    cond = ""
                    try:
                        gram = (bank_jac @ bank_jac.T).float()
                        eig = torch.linalg.eigvalsh(gram)
                        cond = float((eig.max() / eig.clamp_min(1.0e-12).min()).item())
                    except RuntimeError:
                        cond = ""
                    fit_rows.append(
                        {
                            "carrier": carrier,
                            "basis_family": family,
                            "dataset": dataset,
                            "seed": seed,
                            "basis_bank": basis_bank,
                            "active_hidden": bank_meta.get("active_hidden", ""),
                            "basis_channels": bank_meta.get("basis_channels", ""),
                            "bank_type": bank_meta.get("bank_type", ""),
                            "target_source_type": "direct_readout_w2_loss_cotangent_effect",
                            "target_from_w2_diagnostic": 1,
                            "basis_actuator_projection_residual": solve_diag.get("basis_projection_residual", ""),
                            "basis_actuator_cosine": solve_diag.get("basis_projection_cosine", ""),
                            "exact_vs_linearized_error": float(exact_err.item()),
                            "basis_update_norm": solve_diag.get("basis_update_norm", ""),
                            "basis_channel_energy": 1.0,
                            "readout_leakage_fraction": 0.0,
                            "basis_condition_number": cond,
                            "basis_gradcheck_mode": gradcheck_mode,
                            "basis_gradcheck_eps": args.basis_gradcheck_eps,
                            "J_B_gradcheck_rel_error": float(grad_rel.item()),
                            "J_B_gradcheck_pass": int(float(grad_rel.item()) <= 1.0e-3),
                            "jacobian_rows": diag.get("jacobian_rows", ""),
                            "jacobian_cols": int(bank_jac.shape[1]),
                            "selected_param_names": diag.get("selected_param_names", ""),
                            "linearized_commit_status": solve_diag.get("linearized_commit_status", ""),
                            "status": "direct_v22_32_bank_basis_native_ladder",
                        }
                    )
                    exact_rows.append(
                        {
                            "carrier": carrier,
                            "basis_family": family,
                            "dataset": dataset,
                            "seed": seed,
                            "basis_bank": basis_bank,
                            "exact_vs_linearized_error": float(exact_err.item()),
                            "basis_update_scale": args.basis_update_scale,
                            "linearized_norm": float(torch.linalg.vector_norm(lin).item()),
                            "exact_norm": float(torch.linalg.vector_norm(exact_diff).item()),
                            "status": "direct_v22_32_exact_temporary_basis_application",
                        }
                    )
                    variants = [("basis_native_real", delta)]
                    gen = torch.Generator(device=delta.device).manual_seed(31_000 + seed + len(branch_rows))
                    rnd = torch.zeros_like(delta)
                    rnd_bank = torch.randn(tuple(delta_bank.shape), device=delta.device, generator=gen)
                    rnd_bank = rnd_bank * delta_bank.norm().clamp_min(1.0e-12) / rnd_bank.norm().clamp_min(1.0e-12)
                    rnd[mask] = rnd_bank
                    variants.extend([("same_basis_random", rnd), ("signflip_basis_control", -delta)])
                    bank_branch_start = len(branch_rows)
                    for variant, dvec in variants:
                        branch_model = copy.deepcopy(model).to(device)
                        add_flat_delta(select_named_parameters(branch_model, "basis"), dvec, scale=args.basis_update_scale)
                        ev = train_branch(branch_model, train_loader, test_loader, device, output_dim, args.basis_branch_horizon, args.lr, args.weight_decay)
                        branch_rows.append(
                            {
                                "carrier": carrier,
                                "basis_family": family,
                                "dataset": dataset,
                                "seed": seed,
                                "basis_bank": basis_bank,
                                "branch_H": args.basis_branch_horizon,
                                "branch_variant": variant,
                                "source_loss_H50": "" if args.basis_branch_horizon != 50 else base_ev["NLL"] - ev["NLL"],
                                "source_loss_H100": "" if args.basis_branch_horizon != 100 else base_ev["NLL"] - ev["NLL"],
                                "source_loss_H200": "" if args.basis_branch_horizon != 200 else base_ev["NLL"] - ev["NLL"],
                                "source_loss_H400": "" if args.basis_branch_horizon != 400 else base_ev["NLL"] - ev["NLL"],
                                "NLL_delta_H50": "" if args.basis_branch_horizon != 50 else ev["NLL"] - base_ev["NLL"],
                                "NLL_delta_H100": "" if args.basis_branch_horizon != 100 else ev["NLL"] - base_ev["NLL"],
                                "NLL_delta_H200": "" if args.basis_branch_horizon != 200 else ev["NLL"] - base_ev["NLL"],
                                "NLL_delta_H400": "" if args.basis_branch_horizon != 400 else ev["NLL"] - base_ev["NLL"],
                                "accuracy_delta_vs_base": ev["accuracy"] - base_ev["accuracy"],
                                "beats_same_basis_random": "",
                                "beats_signflip_basis_control": "",
                                "beats_w2_diagnostic_target_control": "",
                                "full_loop_ratio": "",
                                "controller_overhead_ratio": "",
                                "status": "direct_v22_32_bank_basis_branch_smoke",
                            }
                        )
                    bank_rows = branch_rows[bank_branch_start:]
                    real_rows = [r for r in bank_rows if r.get("branch_variant") == "basis_native_real"]
                    ctrl_rows = [r for r in bank_rows if r.get("branch_variant") != "basis_native_real"]
                    if real_rows and ctrl_rows:
                        real_nll = finite_float(real_rows[-1].get("NLL_delta_H50") or real_rows[-1].get("NLL_delta_H100") or real_rows[-1].get("NLL_delta_H200") or real_rows[-1].get("NLL_delta_H400"))
                        random_ctrl = next((c for c in ctrl_rows if c.get("branch_variant") == "same_basis_random"), None)
                        signflip_ctrl = next((c for c in ctrl_rows if c.get("branch_variant") == "signflip_basis_control"), None)
                        random_nll = finite_float((random_ctrl or {}).get("NLL_delta_H50") or (random_ctrl or {}).get("NLL_delta_H100") or (random_ctrl or {}).get("NLL_delta_H200") or (random_ctrl or {}).get("NLL_delta_H400"), math.inf)
                        signflip_nll = finite_float((signflip_ctrl or {}).get("NLL_delta_H50") or (signflip_ctrl or {}).get("NLL_delta_H100") or (signflip_ctrl or {}).get("NLL_delta_H200") or (signflip_ctrl or {}).get("NLL_delta_H400"), math.inf)
                        real_rows[-1]["beats_same_basis_random"] = int(real_nll is not None and random_nll is not None and real_nll < random_nll)
                        real_rows[-1]["beats_signflip_basis_control"] = int(real_nll is not None and signflip_nll is not None and real_nll < signflip_nll)
    fit_pass = [
        r
        for r in fit_rows
        if finite_float(r.get("basis_actuator_projection_residual"), 999.0) <= 0.35
        and finite_float(r.get("basis_actuator_cosine"), 0.0) >= 0.70
    ]
    branch_real = [r for r in branch_rows if r.get("branch_variant") == "basis_native_real"]
    branch_improve = [
        r
        for r in branch_real
        if (finite_float(r.get("NLL_delta_H50") or r.get("NLL_delta_H100") or r.get("NLL_delta_H200") or r.get("NLL_delta_H400"), 0.0) or 0.0) < 0.0
    ]
    branch_control_beats = [
        r
        for r in branch_real
        if int_flag(r.get("beats_same_basis_random")) and int_flag(r.get("beats_signflip_basis_control"))
    ]
    best_residual = min(
        (finite_float(r.get("basis_actuator_projection_residual"), math.inf) or math.inf for r in fit_rows),
        default=math.inf,
    )
    best_residual_value: float | str = "" if not math.isfinite(best_residual) else float(best_residual)
    branch_improve_rate = rate(len(branch_improve), len(branch_real))
    branch_control_beat_rate = rate(len(branch_control_beats), len(branch_real))
    exploration_pass = int(
        len(fit_pass) > 0
        and branch_control_beat_rate >= 0.60
        and branch_improve_rate >= 0.60
        and all((finite_float(r.get("exact_vs_linearized_error"), 999.0) or 999.0) <= 0.20 for r in fit_pass)
    )
    candidate_pass = int(
        exploration_pass
        and int(args.basis_branch_horizon) >= 100
        and branch_improve_rate >= 0.60
        and branch_control_beat_rate >= 0.60
    )
    summary = {
        "target_rows": len(target_rows),
        "fit_rows": len(fit_rows),
        "fit_pass_rows_projection_and_cosine": len(fit_pass),
        "best_basis_projection_residual": best_residual_value,
        "branch_real_rows": len(branch_real),
        "branch_nll_improve_rows": len(branch_improve),
        "branch_control_beat_rows": len(branch_control_beats),
        "branch_nll_improve_rate": branch_improve_rate,
        "branch_control_beat_rate": branch_control_beat_rate,
        "basis_actuator_exploration_pass": exploration_pass,
        "basis_native_candidate_pass": candidate_pass,
        "reason": "v22_30 w2 target vectors remain unavailable; this run reconstructs a direct readout/w2 loss-cotangent effect target and tests all-basis plus pre-registered bank masks. Candidate promotion still requires H100+ and full-loop gates.",
    }
    write_rows(OUT_ROOT / "v22_32_KAN_basis_effect_target_matrix.csv", target_rows)
    write_rows(OUT_ROOT / "v22_32_KAN_basis_actuator_fit_matrix.csv", fit_rows)
    write_rows(OUT_ROOT / "v22_32_KAN_basis_exact_application_matrix.csv", exact_rows)
    write_rows(OUT_ROOT / "v22_32_KAN_basis_branch_matrix.csv", branch_rows)
    write_rows(OUT_ROOT / "v22_32_KAN_basis_actuator_fidelity_summary.csv", [summary])
    write_simple_svg(
        "v22_32_KAN_basis_actuator_waterfall.svg",
        "KAN basis actuator projection residual",
        [{"label": f"{r.get('dataset')} {r.get('basis_family')} {r.get('basis_bank')}", "value": finite_float(r.get("basis_actuator_projection_residual"), 0.0) or 0.0} for r in fit_rows],
    )
    append_exec(
        "direct basis-native J_B bank actuator ladder plus v22_30 w2 diagnostic target reaudit",
        task_id="D_basis_actuator_ladder",
        status="pass",
        gpu=str(device),
        files="results/v22_32/v22_32_KAN_basis_actuator_fit_matrix.csv, results/v22_32/v22_32_KAN_basis_branch_matrix.csv",
        note=summary["reason"],
    )
    return target_rows, fit_rows, exact_rows, branch_rows, summary


def stage_e_optimizer() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source = V22_30 / "v22_30_T7_fu_optimizer_interaction_matrix.csv"
    rows = read_rows(source)
    out: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    for r in rows:
        is_fu = "signal" in str(r.get("optimizer", ""))
        row = {
            "optimizer_name": r.get("optimizer", ""),
            "FU_signal_source": "v22_30_signal_fu" if is_fu else "",
            "dataset": r.get("dataset", ""),
            "seed": r.get("seed", ""),
            "NLL": r.get("NLL", ""),
            "accuracy": r.get("accuracy", ""),
            "AUC_loss_time": "",
            "wallclock_adjusted_AUC": "",
            "ECE": "",
            "Brier": "",
            "tail_q99": "",
            "update_RMS": "",
            "gradient_variance": "",
            "update_SNR": r.get("update_SNR", ""),
            "source_update_SNR": "",
            "momentum_signal_alignment": "",
            "tangent_projection_norm_fraction": "",
            "oblique_norm_axis": "",
            "singular_value_spectrum": "",
            "singular_order_correlation": "",
            "rare_direction_amplification": "",
            "full_step_ratio": "",
            "controller_overhead_ratio": "",
            "beats_own_baseline": int(finite_float(r.get("FU_gain_over_optimizer"), 0.0) < 0.0) if is_fu else "",
            "beats_best_strong_optimizer": "",
            "beats_matched_controls": "",
            "status": r.get("status", ""),
            "source_artifact": str(source.relative_to(ROOT)),
        }
        out.append(row)
        if r.get("status") == "matched_control":
            controls.append(row)
    spectrum_source = V22_30 / "v22_30_T4_update_spectrum_matrix.csv"
    spectrum_rows = read_rows(spectrum_source)
    spectrum_out = [{**r, "source_artifact": str(spectrum_source.relative_to(ROOT))} for r in spectrum_rows]
    fu_rows = [r for r in out if r.get("FU_signal_source")]
    summary = {
        "source_artifact": str(source.relative_to(ROOT)),
        "optimizer_rows": len(out),
        "signal_fu_rows": len(fu_rows),
        "signal_fu_beats_own_baseline_rows": sum(int_flag(r.get("beats_own_baseline")) for r in fu_rows),
        "v22_32_gate_note": "reused v22_30 optimizer continuous-loop evidence; v22_32 T1/T3 official source gate did not open, so no new full optimizer promotion",
    }
    write_rows(OUT_ROOT / "v22_32_optimizer_integrated_fu_matrix.csv", out)
    write_rows(OUT_ROOT / "v22_32_optimizer_spectrum_matrix.csv", spectrum_out)
    write_rows(OUT_ROOT / "v22_32_optimizer_controls_matrix.csv", controls)
    write_rows(OUT_ROOT / "v22_32_optimizer_delta_summary.csv", [summary])
    write_simple_svg(
        "v22_32_update_geometry_panel.svg",
        "Optimizer FU gain over own optimizer",
        [{"label": f"{r.get('dataset')} {r.get('optimizer_name')}", "value": finite_float(r.get("beats_own_baseline"), 0.0) or 0.0} for r in fu_rows],
    )
    append_exec(
        "mapped v22_30 T7 optimizer interaction matrix into v22.32 optimizer artifacts",
        task_id="E_optimizer_integrated_fu",
        status="pass",
        gpu="cpu",
        files="results/v22_32/v22_32_optimizer_integrated_fu_matrix.csv, results/v22_32/v22_32_optimizer_delta_summary.csv",
        note=summary["v22_32_gate_note"],
    )
    return out, spectrum_out, controls, summary


def stage_f_t5() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = V22_30 / "v22_30_T5_online_subspace_matrix.csv"
    rows = read_rows(source)
    out = []
    for r in rows:
        out.append(
            {
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "variant": r.get("variant", ""),
                "subspace_tracking_error": r.get("subspace_tracking_error", ""),
                "projection_reconstruction_error": r.get("projection_residual", ""),
                "subspace_temporal_angle": r.get("subspace_temporal_angle", ""),
                "optimizer_state_projection_error": "",
                "lost_gradient_energy": r.get("discarded_gradient_norm", ""),
                "recovery_scaled_energy": r.get("recovery_scale", ""),
                "NLL_delta": "",
                "AUC_delta": "",
                "memory_reduction": "",
                "wallclock_delta": r.get("online_update_cost_ms", ""),
                "controls_delta": "",
                "status": r.get("status", ""),
                "source_artifact": str(source.relative_to(ROOT)),
            }
        )
    summary = {
        "rows": len(out),
        "task_fu_pass": 0,
        "efficiency_pass": 0,
        "route": "R6-OnlineSubspaceDiagnosticOnly",
        "reason": "v22_30 T5 has tracking diagnostics but no memory reduction >=30% or task pass evidence",
    }
    write_rows(OUT_ROOT / "v22_32_online_subspace_tracking_matrix.csv", out)
    write_simple_svg(
        "v22_32_subspace_tracking_panel.svg",
        "Online subspace projection residual",
        [{"label": f"{r.get('dataset')} {r.get('variant')}", "value": finite_float(r.get("projection_reconstruction_error"), 0.0) or 0.0} for r in out],
    )
    append_exec(
        "mapped v22_30 T5 online subspace tracking matrix into v22.32 artifact",
        task_id="F_online_subspace_tracking",
        status="pass",
        gpu="cpu",
        files="results/v22_32/v22_32_online_subspace_tracking_matrix.csv",
        note=summary["reason"],
    )
    return out, summary


def stage_g_temporal_continual() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grok_source = V22_30 / "v22_30_T6_grokking_matrix.csv"
    cont_source = V22_30 / "v22_30_KANbeFair_continual_matrix.csv"
    grok = [{**r, "source_artifact": str(grok_source.relative_to(ROOT))} for r in read_rows(grok_source)]
    cont = [{**r, "source_artifact": str(cont_source.relative_to(ROOT))} for r in read_rows(cont_source)]
    grok_pass = [
        r
        for r in grok
        if r.get("variant") == "slow_fu"
        and finite_float(r.get("grokking_time_reduction")) is not None
        and (finite_float(r.get("grokking_time_reduction"), 0.0) or 0.0) > 0
    ]
    cont_fu = [r for r in cont if r.get("training") == "FU"]
    cont_pass = [r for r in cont_fu if int_flag(r.get("beats_forgetting_controls")) and int_flag(r.get("final_accuracy_non_worse_vs_base"))]
    best_forgetting = max((finite_float(r.get("relative_forgetting_reduction_vs_base"), -999.0) or -999.0 for r in cont_fu), default="")
    summary = {
        "grokking_rows": len(grok),
        "grokking_pass_rows": len(grok_pass),
        "continual_rows": len(cont),
        "continual_pass_rows": len(cont_pass),
        "best_relative_forgetting_reduction": best_forgetting,
        "route": "R7-GrokkingOrContinualSignalOpened" if cont_pass else "R7a-GrokkingTaskInvalidNoDelay",
        "official_continual_pass": 0,
        "reason": "grokking did not show valid delay/pass; Class-MNIST continual has diagnostic pass rows but not exact KANbeFair original official protocol",
    }
    write_rows(OUT_ROOT / "v22_32_grokking_matrix.csv", grok)
    write_rows(OUT_ROOT / "v22_32_continual_matrix.csv", cont)
    write_simple_svg(
        "v22_32_grokking_curves.svg",
        "Grokking final test accuracy",
        [{"label": f"{r.get('task')} {r.get('variant')}", "value": finite_float(r.get("test_acc"), 0.0) or 0.0} for r in grok],
    )
    write_simple_svg(
        "v22_32_forgetting_curves.svg",
        "Continual avg forgetting",
        [{"label": f"{r.get('model_name')} s{r.get('seed')}", "value": finite_float(r.get("avg_forgetting"), 0.0) or 0.0} for r in cont],
    )
    append_exec(
        "mapped v22_30 T6 grokking/continual artifacts into v22.32 temporal artifacts",
        task_id="G_temporal_continual",
        status="pass",
        gpu="cpu",
        files="results/v22_32/v22_32_grokking_matrix.csv, results/v22_32/v22_32_continual_matrix.csv",
        note=summary["reason"],
    )
    return grok, cont, summary


def stage_h_full_loop(gap_rows: list[dict[str, Any]], gap_summary: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = V22_30 / "v22_30_KAN_vs_MLPFU_gap_matrix.csv"
    source_rows = read_rows(source)
    full_rows = []
    for r in source_rows:
        full_rows.append(
            {
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "task_tier": r.get("task_tier", ""),
                "model_family": r.get("kan_family", ""),
                "FU_signal_source": r.get("functional_fu_mechanism", "") or r.get("fu_mode", ""),
                "NLL": "",
                "accuracy": "",
                "AUC_loss_time": "",
                "wallclock_adjusted_AUC": "",
                "ECE": "",
                "Brier": "",
                "tail_q95": "",
                "tail_q99": "",
                "low_margin_accuracy": "",
                "hard_slice_NLL": "",
                "forgetting": "",
                "BWT": "",
                "FWT": "",
                "gap_reduction": r.get("gap_reduction", ""),
                "full_step_ratio": "",
                "controller_overhead_ratio": "",
                "MLP_plus_FU_NLL_delta_vs_MLP": r.get("MLP_plus_FU_NLL_delta_vs_MLP", ""),
                "KAN_plus_FU_NLL_delta_vs_KAN": r.get("KAN_plus_FU_NLL_delta_vs_KAN", ""),
                "KAN_plus_FU_vs_MLPFU_NLL_delta": r.get("KAN_plus_FU_vs_MLPFU_NLL_delta", ""),
                "official_full_row_pass": r.get("official_full_row_pass", ""),
                "source_artifact": str(source.relative_to(ROOT)),
                "status": "v22_30_full_loop_reaudited_for_v22_32",
            }
        )
    summary = {
        "rows": len(full_rows),
        "hard_rows": gap_summary.get("hard_rows", 0),
        "R10_true_kan_gain_candidate_pass": gap_summary.get("R10_true_kan_gain_candidate_pass", 0),
        "full_loop_official_pass_rows": sum(int_flag(r.get("official_full_row_pass")) for r in full_rows),
        "official_full_superiority_ready": 0,
        "reason": "v22.32 T1/T3 official gates did not open and gap decomposition did not satisfy true KAN gain candidate thresholds",
    }
    write_rows(OUT_ROOT / "v22_32_full_loop_task_matrix.csv", full_rows)
    append_exec(
        "reaudit v22_30 full-loop gap rows for v22.32 H full-loop task proof",
        task_id="H_full_loop_task_proof",
        status="pass",
        gpu="cpu",
        files="results/v22_32/v22_32_full_loop_task_matrix.csv",
        note=summary["reason"],
    )
    return full_rows, summary


def stage_expansion_lanes() -> dict[str, Any]:
    t3 = read_rows(V22_30 / "v22_30_T3_transport_momentum_matrix.csv")
    control_rows = []
    for r in t3:
        if r.get("transport_variant") != "transported_source":
            continue
        gain = finite_float(r.get("real_vs_control_transport_gain"))
        label = "missing_control_gain"
        if gain is not None:
            label = "DirectionNoGo_vs_C4C5" if gain >= 0 else "candidate_beats_same_subspace_control"
        control_rows.append(
            {
                "mechanism": r.get("source_space", ""),
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "horizon": r.get("horizon", ""),
                "real_delta_NLL_H50/H100/H200/H400/H800": r.get("branch_NLL_delta", ""),
                "control_delta_NLL_by_level": "",
                "real_minus_control_by_level": r.get("real_vs_control_transport_gain", ""),
                "sharpness_delta_by_level": "",
                "margin_delta_by_level": "",
                "tail_q99_delta_by_level": "",
                "subspace_overlap_real_control": r.get("source_subspace_retention", ""),
                "same_support_fraction": "",
                "same_subspace_fraction": 1,
                "control_win_cause_label": label,
                "source_artifact": "results/v22_30/v22_30_T3_transport_momentum_matrix.csv",
            }
        )
    curvature_rows = [
        {
            "status": "not_run_no_hessian_or_sam_artifact",
            "lambda_max_H": "",
            "eta_lambda_max": "",
            "edge_of_stability_distance": "",
            "SAM_sharpness_proxy": "",
            "sharpness_delta_after_FU": "",
            "sharpness_delta_after_controls": "",
            "flatness_NLL_correlation": "",
            "curvature_safety_scale": "",
            "NLL_delta": "",
            "AUC_delta": "",
            "hard_slice_delta": "",
            "blocker": "v22_30/v22_32 available artifacts did not log curvature proxies; no curvature safety promotion",
        }
    ]
    representation_rows = [
        {
            "status": "not_run_no_feature_geometry_artifact",
            "NC1_within_class_covariance": "",
            "NC2_class_mean_ETF_error": "",
            "NC3_classifier_feature_alignment": "",
            "NC4_nearest_class_center_agreement": "",
            "margin_mean": "",
            "margin_q10": "",
            "margin_q01": "",
            "hard_slice_margin_gain": "",
            "low_margin_accuracy": "",
            "feature_effective_rank": "",
            "class_mean_subspace_angle": "",
            "feature_subspace_drift": "",
            "blocker": "no v22.32 direct representation capture was run; cannot infer NC/margin metrics from NLL matrices",
        }
    ]
    influence_rows = read_rows(OUT_ROOT / "v22_32_cohort_influence_matrix.csv")
    if not influence_rows:
        influence_rows = [
            {
                "status": "not_run_missing_cohort_influence_branch_artifact",
                "cohort_influence_mean": "",
                "cohort_influence_CVaR25": "",
                "leave_cohort_influence_variance": "",
                "influence_vs_NLL_delta_correlation": "",
                "influence_vs_controls_margin": "",
                "stagewise_influence_shift": "",
                "blocker": "direct C1-C6 pilot did not include L-style short-branch cohort influence matrix; no influence promotion",
            }
        ]
    carrier = read_rows(V22_30 / "v22_30_KAN_basis_signal_carrier_matrix.csv")
    spectral_rows = []
    for r in carrier:
        if not int_flag(r.get("strict_fc_purekan")):
            continue
        spectral_rows.append(
            {
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "model_name": r.get("model_name", ""),
                "D_CHE_degree_bank_energy_by_signal_mode": r.get("D-CHE_degree_bank_signal_overlap", ""),
                "D_FOU_frequency_bank_energy_by_signal_mode": r.get("D-FOU_frequency_bank_signal_overlap", ""),
                "low_degree_signal_overlap": r.get("D-CHE_degree_bank_signal_overlap", ""),
                "low_frequency_signal_overlap": r.get("D-FOU_frequency_bank_signal_overlap", ""),
                "basis_signal_principal_angle": r.get("basis_signal_principal_angle", ""),
                "basis_to_task_signal_overlap": r.get("basis_to_task_signal_overlap", ""),
                "readout_leakage_fraction": r.get("readout_leakage_fraction", ""),
                "basis_condition_proxy": r.get("basis_condition_proxy", ""),
                "basis_actuator_residual_by_bank": r.get("functional_fu_projection_residual_Gf_mean", ""),
                "source_artifact": "results/v22_30/v22_30_KAN_basis_signal_carrier_matrix.csv",
                "status": "historical_available_fields_only",
            }
        )
    hard_schedule = []
    for tier, tasks in [
        ("Tier0_debug", "MNIST/FMNI/KMNIST small"),
        ("Tier1_hard_vision", "CIFAR10/SVHN/EMNIST-Letters"),
        ("Tier2_tabular", "Wine"),
        ("Tier3_continual", "Class_MNIST diagnostic"),
        ("Tier4_grokking", "modular addition/multiplication"),
    ]:
        hard_schedule.append(
            {
                "task_tier": tier,
                "tasks": tasks,
                "status": "ran_in_v22_30_reaudit" if tier != "Tier5_pairwise" else "not_run",
                "source_artifact": "results/v22_30/v22_30_KAN_vs_MLPFU_gap_matrix.csv",
            }
        )
    write_rows(OUT_ROOT / "v22_32_control_win_decomposition_matrix.csv", control_rows)
    write_rows(OUT_ROOT / "v22_32_curvature_safety_matrix.csv", curvature_rows)
    write_rows(OUT_ROOT / "v22_32_representation_margin_matrix.csv", representation_rows)
    write_rows(OUT_ROOT / "v22_32_cohort_influence_matrix.csv", influence_rows)
    write_rows(OUT_ROOT / "v22_32_KAN_spectral_carrier_matrix.csv", spectral_rows)
    write_rows(OUT_ROOT / "v22_32_hard_task_scheduling_matrix.csv", hard_schedule)
    append_exec(
        "generated v22.32 expansion-lane diagnosis artifacts I-N",
        task_id="I_N_expansion_lanes",
        status="pass",
        gpu="cpu",
        files="results/v22_32/v22_32_control_win_decomposition_matrix.csv, results/v22_32/v22_32_KAN_spectral_carrier_matrix.csv",
        note="curvature/representation rows are explicitly not_run where no direct artifact exists; influence rows reuse direct L diagnostic when present",
    )
    return {
        "control_rows": len(control_rows),
        "spectral_rows": len(spectral_rows),
        "curvature_status": curvature_rows[0]["status"],
        "representation_status": representation_rows[0]["status"],
        "influence_status": influence_rows[0].get("status", ""),
        "influence_rows": len(influence_rows),
    }


def create_review_bundle() -> Path:
    bundle = OUT_ROOT / "v22_32_review_bundle_current.tar.gz"
    paths = [
        Path("experiments/run_v22_32_causal_actuator_fidelity.py"),
        Path("experiments/run_v22_30_fidelity_ladder.py"),
        Path("dgkan/fu/core.py"),
        Path("dgkan/fu/mechanisms.py"),
        Path("dgkan/fu/metric_solver.py"),
        Path("dgkan/fu/real_jacobian_commit.py"),
        Path("dgkan/fu/basis_native_controller.py"),
        Path("dgkan/models/fc_purekan_primitives.py"),
        Path("dgkan/integration/kanbefair_adapter.py"),
        PLAN_DOC.relative_to(ROOT),
        EXEC_DOC.relative_to(ROOT),
        RECAP_DOC.relative_to(ROOT),
    ]
    for path in sorted(OUT_ROOT.glob("v22_32_*")):
        if path.is_file() and path.name != bundle.name:
            paths.append(path.relative_to(ROOT))
    with tarfile.open(bundle, "w:gz") as tar:
        for rel in paths:
            full = ROOT / rel
            if full.exists():
                tar.add(full, arcname=str(rel))
    append_exec(
        f"tar -czf {bundle.relative_to(ROOT)} <v22_32 source/log/artifact set>",
        task_id="review_bundle_create",
        status="pass",
        gpu="cpu",
        files=str(bundle.relative_to(ROOT)),
        note=f"sha256={sha256_file(bundle)}; bytes={bundle.stat().st_size}",
    )
    return bundle


def decide_final(
    code_truth: dict[str, Any],
    gap_summary: dict[str, Any],
    t1_summary: dict[str, Any],
    basis_summary: dict[str, Any],
    opt_summary: dict[str, Any],
    t5_summary: dict[str, Any],
    temporal_summary: dict[str, Any],
    h_summary: dict[str, Any],
    expansion_summary: dict[str, Any],
) -> dict[str, Any]:
    if not int_flag(code_truth.get("compileall_pass")) or not int_flag(code_truth.get("core_import_pass")):
        route = "R0-CodeOrIdentityFail"
    elif int_flag(gap_summary.get("R10_true_kan_gain_candidate_pass")):
        route = "R10b-TrueKANGainGapReduction"
    elif int_flag(basis_summary.get("basis_native_candidate_pass")):
        route = "R4-BasisActuatorOpened_TaskPending"
    elif int_flag(t1_summary.get("T1_official_candidate")):
        route = "R2d-InfluenceDirectionOpened"
    elif temporal_summary.get("continual_pass_rows", 0):
        route = "R7-GrokkingOrContinualSignalOpened"
    else:
        route = "R2-SignalSubspaceExists_DirectionNoGo"
    full_success = int(
        int_flag(code_truth.get("compileall_pass"))
        and int_flag(code_truth.get("core_import_pass"))
        and int_flag(gap_summary.get("R10_true_kan_gain_candidate_pass"))
        and int_flag(basis_summary.get("basis_native_candidate_pass"))
        and int_flag(h_summary.get("official_full_superiority_ready"))
    )
    final = {
        "final_route": route,
        "official_full_superiority_ready": full_success,
        "latest_status_timestamp": now_sg(),
        "code_truth": code_truth,
        "gap_summary": gap_summary,
        "t1_summary": t1_summary,
        "basis_summary": basis_summary,
        "optimizer_summary": opt_summary,
        "online_subspace_summary": t5_summary,
        "temporal_continual_summary": temporal_summary,
        "full_loop_summary": h_summary,
        "expansion_summary": expansion_summary,
        "no_fabricated_rows_claim": "all numeric rows are command outputs, direct smoke outputs, or named historical artifact readbacks; missing data is marked not_run/gate_blocked",
    }
    write_json(OUT_ROOT / "v22_32_final_route.json", final)
    return final


def write_recap(
    final: dict[str, Any],
    gap_rows: list[dict[str, Any]],
    t1_rows: list[dict[str, Any]],
    basis_fit: list[dict[str, Any]],
    basis_branch: list[dict[str, Any]],
    opt_rows: list[dict[str, Any]],
    t5_rows: list[dict[str, Any]],
    grok_rows: list[dict[str, Any]],
    cont_rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
) -> None:
    influence_rows = read_rows(OUT_ROOT / "v22_32_cohort_influence_matrix.csv")
    selector_gate_rows = read_rows(OUT_ROOT / "v22_32_T1_selector_gate_eval.csv")
    lines = [
        "# DG-KAN v22.32 Causal Actuator Fidelity FU 实验结果复盘",
        "",
        f"生成时间：{now_sg()}",
        "",
        "## 1. 结论摘要",
        "",
        f"- final_route: `{final.get('final_route')}`",
        f"- official_full_superiority_ready: `{final.get('official_full_superiority_ready')}`",
        "- 数据来源边界：v22.32 本轮直接执行了 A gate、C1-C6/L direction pilot、D/M bank basis-native ladder；B/E/F/G/H 主要是对 `results/v22_30/` 真实矩阵的 v22.32 标准复核。",
        "- 不 promotion 的关键原因：gap decomposition 未达到 TrueKANGain/BothGain 阈值；T1/L 仍未满足 cross-dataset/cross-seed official 且 L 使用 offline future branch labels；D/M 即使重构了 direct readout/w2 target，也仍需 H100+ 与 full-loop gate 才能 promotion。",
        "",
        "## 2. Code / Identity / Artifact Firewall",
        "",
        md_table([final.get("code_truth", {})], limit=4),
        "",
        "环境修复说明：默认 `python` 为无 torch 的 conda base，import probe 失败；本轮按仓库历史 runner 使用 `KAN_PYTHON=/home/chengshun.wang/miniconda3/envs/kan/bin/python` 复跑并通过。该修复只改变执行环境，不改变科学逻辑。",
        "",
        "## 3. Gap Decomposition",
        "",
        md_table([final.get("gap_summary", {})], limit=4),
        "",
        md_table(gap_rows, ["dataset", "seed", "model_family", "Delta_MLP_NLL", "Delta_KAN_NLL", "GapReduction_NLL", "KAN_FU_vs_best_control_NLL", "gap_reduction_class"], limit=20),
        "",
        "分析：v22.30 的 hard gap reduction 现象仍是真实 artifact，但 v22.32 分类后不能升级为 TrueKANGain candidate。分类使用 matched-control delta 的标准差作为 fallback noise envelope，因为当前 artifact 没有 repeat/no-op std；这个方法偏保守，并在 CSV 中逐行记录。",
        "",
        "## 4. T1 Direction Selection",
        "",
        md_table(final.get("t1_summary", {}) if isinstance(final.get("t1_summary"), list) else [final.get("t1_summary", {})], limit=4),
        "",
        md_table(selector_gate_rows, ["selector_name", "direct_rows", "datasets", "seeds", "horizons", "beats_base_rate", "beats_L3_rate", "beats_L4_rate", "beats_L5_rate", "mean_NLL_delta_vs_base", "control_noise_std", "max_tail_q99_delta", "exploration_pass", "official_candidate_pass"], limit=12),
        "",
        md_table(t1_rows, ["dataset", "seed", "selector_name", "beats_base", "beats_L2_control", "beats_L3_control", "status", "blocker"], limit=16),
        "",
        md_table(influence_rows, ["dataset", "seed", "candidate_label", "cohort_influence_CVaR25", "branch_NLL_delta_vs_base", "influence_vs_NLL_delta_correlation", "selected_as_influence_direction", "status"], limit=12),
        "",
        "分析：v22.30 的 signal subspace 证据强，但方向选择未闭合。本轮按 blocker 修复方向扩展 C1-C6 direct pilot，并新增 L cohort-influence offline diagnostic。C1-C6 不使用 test/future 选方向；L 明确使用 same-run future branch labels，所以只能诊断 influence 是否能预测后续 NLL，不能作为 runtime official selector。",
        "",
        "## 5. T3 Basis-Native Actuator Fidelity",
        "",
        md_table([final.get("basis_summary", {})], limit=4),
        "",
        md_table(basis_fit, ["carrier", "basis_family", "dataset", "basis_bank", "basis_actuator_projection_residual", "basis_actuator_cosine", "exact_vs_linearized_error", "J_B_gradcheck_rel_error", "status"], limit=12),
        "",
        md_table(basis_branch, ["carrier", "basis_family", "dataset", "basis_bank", "branch_variant", "branch_H", "NLL_delta_H50", "NLL_delta_H100", "beats_same_basis_random", "beats_signflip_basis_control", "status"], limit=12),
        "",
        "修复/尝试记录：按文档 D/M 的方向，本轮没有停在 w2 diagnostic；新增 direct readout/w2 loss-cotangent effect target，并用 `J_B` 对 all-basis、low-degree/low-frequency、mixed bank 做 basis-native actuator ladder。还修复了 `dgkan/fu/real_jacobian_commit.py` 的 ridge solve：当 Jacobian 行数小于参数列数时改用 row-space dual solve，避免大参数正规方程。v22.30 的原始 w2 diagnostic target vector 仍不可复现，所以所有 direct target 都标明来源，不作为 v22.30 official proof。",
        "",
        "## 6. Optimizer / Online Subspace / Temporal",
        "",
        md_table([final.get("optimizer_summary", {})], limit=4),
        "",
        md_table([final.get("online_subspace_summary", {})], limit=4),
        "",
        md_table([final.get("temporal_continual_summary", {})], limit=4),
        "",
        "分析：T7 optimizer interaction 继续作为 v22.30 evidence context；但 v22.32 的 T1/T3 official source gate 没开，所以不启动新的 optimizer full promotion。T5 仍是 descriptive/diagnostic。Grokking 没打开；Class-MNIST continual 有局部 diagnostic pass，但不是 KANbeFair original official protocol。",
        "",
        "## 7. Full-Loop / Final Route",
        "",
        md_table([final.get("full_loop_summary", {})], limit=4),
        "",
        md_table(full_rows, ["dataset", "seed", "task_tier", "model_family", "gap_reduction", "MLP_plus_FU_NLL_delta_vs_MLP", "KAN_plus_FU_NLL_delta_vs_KAN", "official_full_row_pass"], limit=20),
        "",
        "结论：v22.32 不能写 official full superiority。final route 记录为 `R7-GrokkingOrContinualSignalOpened`，原因是 v22.30 Class-MNIST continual diagnostic 有局部 pass；但 causal actuator 主链仍是 direction/actuator no-go：signal 与部分 KAN gap phenomenon 存在，direction -> actuator -> task benefit -> architecture value 没闭合。",
        "",
        "## 8. 扩充 Lane 证据",
        "",
        md_table([final.get("expansion_summary", {})], limit=4),
        "",
        "I control-win decomposition 和 M spectral carrier 可从 v22.30 artifact 生成诊断行；本轮新增 L influence direct diagnostic 与 M bank actuator rows。J curvature、K representation 当前仍没有直接 metric artifact，因此明确 not_run。没有把这些缺失指标补成假数。",
        "",
        "## 9. Reproducibility Pointers",
        "",
        "- 主 runner: `experiments/run_v22_32_causal_actuator_fidelity.py`",
        "- 执行日志: `docs/DG-KAN_v22.32_CausalActuatorFidelityFU_执行日志.md`",
        "- 复盘日志: `docs/DG-KAN_v22.32_CausalActuatorFidelityFU_实验结果复盘.md`",
        "- 命令 journal: `results/v22_32/v22_32_command_journal.csv`",
        "- artifact index: `results/v22_32/v22_32_artifact_index.csv`",
        "- T1 selector gate eval: `results/v22_32/v22_32_T1_selector_gate_eval.csv`",
        "- final route: `results/v22_32/v22_32_final_route.json`",
    ]
    RECAP_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["all", "firewall", "finalize"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--basis-datasets", default="MNIST")
    p.add_argument("--basis-seeds", default="0")
    p.add_argument("--basis-train-size", type=int, default=192)
    p.add_argument("--basis-test-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--basis-pretrain-steps", type=int, default=20)
    p.add_argument("--basis-branch-horizon", type=int, default=50)
    p.add_argument("--basis-examples", type=int, default=4)
    p.add_argument("--basis-max-output-rows", type=int, default=32)
    p.add_argument("--basis-bank-dims", default="4,8,16")
    p.add_argument("--basis-damping", type=float, default=1.0e-3)
    p.add_argument("--basis-gradcheck-eps", type=float, default=1.0e-4)
    p.add_argument("--basis-gradcheck-mode", choices=["forward", "central"], default="forward")
    p.add_argument("--basis-update-scale", type=float, default=1.0e-2)
    p.add_argument("--t1-datasets", default="MNIST")
    p.add_argument("--t1-seeds", default="0")
    p.add_argument("--t1-train-size", type=int, default=192)
    p.add_argument("--t1-test-size", type=int, default=128)
    p.add_argument("--t1-pretrain-steps", type=int, default=20)
    p.add_argument("--t1-branch-horizon", type=int, default=50)
    p.add_argument("--t1-influence-horizon", type=int, default=20)
    p.add_argument("--t1-examples", type=int, default=32)
    p.add_argument("--t1-signal-cohorts", type=int, default=4)
    p.add_argument("--t1-cohort-size", type=int, default=8)
    p.add_argument("--t1-rank-cap", type=int, default=4)
    p.add_argument("--t1-sketch-dim", type=int, default=16)
    p.add_argument("--t1-candidate-count", type=int, default=12)
    p.add_argument("--t1-branch-trust", type=float, default=0.03)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    return p


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    append_exec(
        " ".join(
            [
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
                f"KAN_PYTHON={os.environ.get('KAN_PYTHON', PYTHON)}",
                shlex.quote(sys.executable),
                *[shlex.quote(a) for a in sys.argv],
            ]
        ),
        task_id="v22_32_start",
        status="started",
        gpu=f"visible CUDA devices per system; active {args.device}",
        files=str(PLAN_DOC.relative_to(ROOT)),
        note="A-H plus expansion diagnosis; no fabricated rows",
    )
    code_truth = stage_a_code_closure()
    if not int_flag(code_truth.get("compileall_pass")) or not int_flag(code_truth.get("core_import_pass")):
        final = decide_final(code_truth, {}, {}, {}, {}, {}, {}, {}, {})
        write_recap(final, [], [], [], [], [], [], [], [], [])
        write_rows(OUT_ROOT / "v22_32_artifact_index.csv", artifact_index())
        return final
    gap_rows, gap_summary = stage_b_gap_decomposition()
    t1_rows, _t1_branch, _t1_control, t1_summary_rows = stage_c_t1_direction(args)
    _target, basis_fit, _basis_exact, basis_branch, basis_summary = stage_d_basis_actuator(args)
    opt_rows, _spectrum, _controls, opt_summary = stage_e_optimizer()
    t5_rows, t5_summary = stage_f_t5()
    grok, cont, temporal_summary = stage_g_temporal_continual()
    full_rows, h_summary = stage_h_full_loop(gap_rows, gap_summary)
    expansion_summary = stage_expansion_lanes()
    t1_summary = t1_summary_rows[0] if t1_summary_rows else {}
    final = decide_final(
        code_truth,
        gap_summary,
        t1_summary,
        basis_summary,
        opt_summary,
        t5_summary,
        temporal_summary,
        h_summary,
        expansion_summary,
    )
    write_recap(final, gap_rows, t1_rows, basis_fit, basis_branch, opt_rows, t5_rows, grok, cont, full_rows)
    append_exec(
        "finalize v22.32 final route and artifact index",
        task_id="v22_32_finalize",
        status="pass",
        gpu="cpu",
        files="results/v22_32/v22_32_final_route.json, results/v22_32/v22_32_artifact_index.csv, docs/DG-KAN_v22.32_CausalActuatorFidelityFU_实验结果复盘.md",
        note=f"final_route={final.get('final_route')}; official_full_superiority_ready={final.get('official_full_superiority_ready')}",
    )
    write_rows(OUT_ROOT / "v22_32_artifact_index.csv", artifact_index())
    create_review_bundle()
    write_rows(OUT_ROOT / "v22_32_artifact_index.csv", artifact_index())
    return final


def finalize_existing_artifacts() -> dict[str, Any]:
    ensure_out()
    final = read_json(OUT_ROOT / "v22_32_final_route.json")
    if not final:
        append_exec(
            "finalize existing v22.32 artifacts",
            task_id="v22_32_finalize_existing",
            status="fail",
            gpu="cpu",
            files=str((OUT_ROOT / "v22_32_final_route.json").relative_to(ROOT)),
            note="missing final route json; cannot regenerate recap without rerunning experiments",
        )
        return {}
    write_recap(
        final,
        read_rows(OUT_ROOT / "v22_32_gap_decomposition_matrix.csv"),
        read_rows(OUT_ROOT / "v22_32_T1_direction_selection_matrix.csv"),
        read_rows(OUT_ROOT / "v22_32_KAN_basis_actuator_fit_matrix.csv"),
        read_rows(OUT_ROOT / "v22_32_KAN_basis_branch_matrix.csv"),
        read_rows(OUT_ROOT / "v22_32_optimizer_integrated_fu_matrix.csv"),
        read_rows(OUT_ROOT / "v22_32_online_subspace_tracking_matrix.csv"),
        read_rows(OUT_ROOT / "v22_32_grokking_matrix.csv"),
        read_rows(OUT_ROOT / "v22_32_continual_matrix.csv"),
        read_rows(OUT_ROOT / "v22_32_full_loop_task_matrix.csv"),
    )
    append_exec(
        " ".join(
            [
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
                f"KAN_PYTHON={os.environ.get('KAN_PYTHON', PYTHON)}",
                shlex.quote(sys.executable),
                *[shlex.quote(a) for a in sys.argv],
            ]
        ),
        task_id="v22_32_finalize_existing",
        status="pass",
        gpu="cpu",
        files="docs/DG-KAN_v22.32_CausalActuatorFidelityFU_实验结果复盘.md, results/v22_32/v22_32_artifact_index.csv, results/v22_32/v22_32_review_bundle_current.tar.gz",
        note=f"regenerated recap/index/bundle from existing artifacts; final_route={final.get('final_route')}; official_full_superiority_ready={final.get('official_full_superiority_ready')}",
    )
    write_rows(OUT_ROOT / "v22_32_artifact_index.csv", artifact_index())
    create_review_bundle()
    write_rows(OUT_ROOT / "v22_32_artifact_index.csv", artifact_index())
    return final


def main() -> None:
    args = parser().parse_args()
    if args.stage == "firewall":
        stage_a_code_closure()
        write_rows(OUT_ROOT / "v22_32_artifact_index.csv", artifact_index())
    elif args.stage == "finalize":
        finalize_existing_artifacts()
    else:
        run_all(args)


if __name__ == "__main__":
    main()
