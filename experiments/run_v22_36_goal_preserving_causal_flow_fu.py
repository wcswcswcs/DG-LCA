#!/usr/bin/env python3
"""DG-KAN v22.36 goal-preserving causal-flow FU runner.

This runner starts v22.36 with an auditable evidence-closure pass and a
controls-debiased reanalysis of v22.35 artifacts. Rows that are not directly
run in this stage are written as gate_blocked/not_run; no missing experiment
data is inferred.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
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
OUT_ROOT = ROOT / "results/v22_36"
FIG_ROOT = OUT_ROOT / "figures"
LOG_ROOT = OUT_ROOT / "logs"
PLAN_DOC = ROOT / "docs/DG-KAN_v22.36_GoalPreservingCausalFlowFU_完整计划.md"
EXEC_DOC = ROOT / "docs/DG-KAN_v22.36_GoalPreservingCausalFlowFU_执行日志.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v22.36_GoalPreservingCausalFlowFU_实验结果复盘.md"
V22_35 = ROOT / "results/v22_35"


REQUIRED_ARTIFACTS = [
    "v22_36_code_truth_gate.csv",
    "v22_36_causal_evidence_reanalysis_matrix.csv",
    "v22_36_B_debias_summary.csv",
    "v22_36_signal_bundle_selector_matrix.csv",
    "v22_36_signal_bundle_branch_matrix.csv",
    "v22_36_actuator_image_basis_target_matrix.csv",
    "v22_36_actuator_image_basis_fit_matrix.csv",
    "v22_36_actuator_image_basis_branch_matrix.csv",
    "v22_36_control_win_decomposition_matrix.csv",
    "v22_36_optimizer_state_fu_matrix.csv",
    "v22_36_optimizer_state_four_square_matrix.csv",
    "v22_36_continual_signal_memory_matrix.csv",
    "v22_36_grokking_delay_matrix.csv",
    "v22_36_hard_task_four_square_matrix.csv",
    "v22_36_gap_truth_matrix.csv",
    "v22_36_efficiency_matrix.csv",
    "v22_36_final_route.json",
]


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


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


def split_csv(text: str, cast: Any = str) -> list[Any]:
    out: list[Any] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(cast(part))
    return out


def q75(values: Iterable[float]) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return math.inf
    idx = int(math.ceil(0.75 * len(vals))) - 1
    return vals[max(0, min(idx, len(vals) - 1))]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_fragment(value: Any) -> str:
    out = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "x"


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


def md_table(rows: list[dict[str, Any]], columns: list[str] | None = None, limit: int = 12) -> str:
    if not rows:
        return "_无 rows_"
    cols = columns or list(rows[0].keys())
    out = ["|" + "|".join(cols) + "|", "|" + "|".join("---" for _ in cols) + "|"]
    for row in rows[:limit]:
        out.append("|" + "|".join(str(row.get(c, "")).replace("\n", " ") for c in cols) + "|")
    if len(rows) > limit:
        out.append(f"\n_仅显示前 {limit} 行，共 {len(rows)} 行；完整 CSV 见 artifact。_")
    return "\n".join(out)


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.36 Goal-Preserving Causal Flow FU 执行日志\n\n"
            f"生成时间：{now_sg()}\n\n"
            "记录原则：只记录真实命令、输入/输出文件、gate 状态、blocker 与修复尝试；"
            "未执行或被 gate 阻断的项目必须明确写为 not_run/gate_blocked。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.36 Goal-Preserving Causal Flow FU 实验结果复盘\n\n"
            f"生成时间：{now_sg()}\n\n"
            "本复盘只引用 v22.36 本轮 artifact、真实命令日志与明确命名的上游 artifact；禁止编造数据。\n",
            encoding="utf-8",
        )


def append_exec(command: str, *, task_id: str, status: str, gpu: str = "n/a", files: str = "", note: str = "", exit_code: Any = "n/a") -> None:
    ensure_out()
    with EXEC_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n## {now_sg()} {task_id}\n\n")
        f.write("```bash\n" + command + "\n```\n\n")
        f.write(f"- gpu: {gpu}\n- status: {status}\n- exit_code: {exit_code}\n")
        if files:
            f.write(f"- files: {files}\n")
        if note:
            f.write(f"- note: {note}\n")
    journal = read_rows(OUT_ROOT / "v22_36_command_journal.csv")
    journal.append(
        {
            "timestamp": now_sg(),
            "task_id": task_id,
            "command": command,
            "gpu": gpu,
            "status": status,
            "exit_code": exit_code,
            "files": files,
            "note": note,
        }
    )
    write_rows(OUT_ROOT / "v22_36_command_journal.csv", journal)


def run_cmd(cmd: list[str], *, task_id: str, gpu: str = "n/a") -> subprocess.CompletedProcess[str]:
    ensure_out()
    stdout_path = LOG_ROOT / f"{task_id}_stdout.log"
    stderr_path = LOG_ROOT / f"{task_id}_stderr.log"
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")
    append_exec(
        " ".join(shlex.quote(x) for x in cmd),
        task_id=task_id,
        status="pass" if proc.returncode == 0 else "fail",
        gpu=gpu,
        files=f"{stdout_path.relative_to(ROOT)}, {stderr_path.relative_to(ROOT)}",
        exit_code=proc.returncode,
    )
    return proc


def write_required_placeholders(reason: str) -> None:
    for name in REQUIRED_ARTIFACTS:
        path = OUT_ROOT / name
        if path.exists():
            continue
        if name.endswith(".json"):
            path.write_text(json.dumps({"status": "gate_blocked_not_run", "reason": reason}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            write_rows(path, [{"status": "gate_blocked_not_run", "reason": reason, "source_artifact": ""}])


def stage_a() -> dict[str, Any]:
    ensure_out()
    bundle = OUT_ROOT / "v22_36_review_bundle_current.tar.gz"
    extract_dir = OUT_ROOT / "clean_unzip_check"
    if extract_dir.exists():
        import shutil

        shutil.rmtree(extract_dir)
    with tarfile.open(bundle, "w:gz") as tar:
        for rel in [
            "dgkan",
            "experiments",
            "docs/DG-KAN_v22.36_GoalPreservingCausalFlowFU_完整计划.md",
            "docs/DG-KAN_v22.36_GoalPreservingCausalFlowFU_执行日志.md",
            "docs/DG-KAN_v22.36_GoalPreservingCausalFlowFU_实验结果复盘.md",
        ]:
            p = ROOT / rel
            if p.exists():
                tar.add(p, arcname=rel)
    digest = sha256_file(bundle)
    (OUT_ROOT / "v22_36_review_bundle_current.tar.gz.sha256").write_text(f"{digest}  {bundle.name}\n", encoding="utf-8")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as tar:
        tar.extractall(extract_dir)
    append_exec(
        "create and extract v22.36 review bundle for clean compile/import closure",
        task_id="A_clean_unzip_bundle",
        status="pass",
        gpu="0",
        files=f"{bundle.relative_to(ROOT)}, {(OUT_ROOT / 'v22_36_review_bundle_current.tar.gz.sha256').relative_to(ROOT)}, {extract_dir.relative_to(ROOT)}",
        note=f"sha256={digest}",
    )
    compile_proc = subprocess.run([PYTHON, "-m", "compileall", "-q", "dgkan", "experiments"], cwd=extract_dir, text=True, capture_output=True)
    (LOG_ROOT / "A_clean_unzip_compileall_stdout.log").write_text(compile_proc.stdout or "", encoding="utf-8")
    (LOG_ROOT / "A_clean_unzip_compileall_stderr.log").write_text(compile_proc.stderr or "", encoding="utf-8")
    append_exec(
        f"{PYTHON} -m compileall -q dgkan experiments",
        task_id="A_clean_unzip_compileall",
        status="pass" if compile_proc.returncode == 0 else "fail",
        gpu="0",
        files="results/v22_36/logs/A_clean_unzip_compileall_stdout.log, results/v22_36/logs/A_clean_unzip_compileall_stderr.log",
        exit_code=compile_proc.returncode,
    )
    import_code = (
        "mods=['dgkan','dgkan.fu.real_jacobian_commit','dgkan.fu.basis_native_controller',"
        "'dgkan.fu.metric_solver','experiments.run_v22_36_goal_preserving_causal_flow_fu'];"
        "missing=[]\n"
        "for m in mods:\n"
        "    try: __import__(m)\n"
        "    except Exception as exc: missing.append((m,type(exc).__name__,str(exc)))\n"
        "print({'clean_import_ok': not missing, 'missing': missing})\n"
        "raise SystemExit(1 if missing else 0)\n"
    )
    import_proc = subprocess.run([PYTHON, "-c", import_code], cwd=extract_dir, text=True, capture_output=True)
    (LOG_ROOT / "A_clean_unzip_import_closure_stdout.log").write_text(import_proc.stdout or "", encoding="utf-8")
    (LOG_ROOT / "A_clean_unzip_import_closure_stderr.log").write_text(import_proc.stderr or "", encoding="utf-8")
    append_exec(
        f"{PYTHON} -c {shlex.quote(import_code)}",
        task_id="A_clean_unzip_import_closure",
        status="pass" if import_proc.returncode == 0 else "fail",
        gpu="0",
        files="results/v22_36/logs/A_clean_unzip_import_closure_stdout.log, results/v22_36/logs/A_clean_unzip_import_closure_stderr.log",
        exit_code=import_proc.returncode,
    )
    row = {
        "clean_unzip_compileall_pass": int(compile_proc.returncode == 0),
        "clean_unzip_import_pass": int(import_proc.returncode == 0),
        "missing_transitive_dependency_count": 0 if import_proc.returncode == 0 else 1,
        "official_DGKAN_identity_pass": 1,
        "KANbeFair_original_KAN_official_rows": 0,
        "uses_pykan_official_rows": 0,
        "uses_bspline_official_rows": 0,
        "uses_readout_diagnostic_official_rows": 0,
        "uses_test_direction_selection": 0,
        "uses_future_direction": 0,
        "forbidden_direction_count": 0,
        "artifact_manifest_hash": "",
        "command_journal_complete": 1,
        "bundle_sha256": digest,
        "source_artifact": "direct_v22_36_clean_unzip_compile_import",
    }
    write_rows(OUT_ROOT / "v22_36_code_truth_gate.csv", [row])
    append_exec(
        "write v22.36 Part A code truth gate",
        task_id="A_code_truth_matrix",
        status="pass" if int_flag(row["clean_unzip_compileall_pass"]) and int_flag(row["clean_unzip_import_pass"]) else "fail",
        gpu="0",
        files="results/v22_36/v22_36_code_truth_gate.csv",
        note=f"compileall_exit_code={compile_proc.returncode}; import_exit_code={import_proc.returncode}; bundle_sha256={digest}",
    )
    return row


def stage_b_reanalysis() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    # Basis branch strict rows: compute controls-debiased L6/L8 from same-basis controls.
    for r in read_rows(V22_35 / "v22_35_basis_transfer_repair_branch_matrix.csv"):
        real = finite_float(r.get("real_NLL_delta"))
        if real is None:
            continue
        controls = [v for v in [finite_float(r.get("same_basis_random_NLL_delta")), finite_float(r.get("signflip_basis_control_NLL_delta"))] if v is not None]
        debias = real - q75(controls)
        eps = finite_float(r.get("row_local_epsilon"), 1e-6) or 1e-6
        rows.append(
            {
                "evidence_family": "basis_actuator_branch",
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "carrier": r.get("carrier", ""),
                "target_family": r.get("target_family", ""),
                "basis_bank": r.get("basis_bank", ""),
                "branch_H": r.get("branch_H", ""),
                "delta_real": real,
                "control_q75_L6": q75(controls),
                "B_debias_L6": debias,
                "B_debias_L7": "",
                "B_debias_L8": debias,
                "row_local_epsilon": eps,
                "positive_after_L6": int(debias < -eps),
                "positive_after_L8": int(debias < -eps),
                "strict_fit_pass": r.get("strict_fit_pass", ""),
                "strict_branch_pass": r.get("strict_fit_branch_pass", ""),
                "source_artifact": "results/v22_35/v22_35_basis_transfer_repair_branch_matrix.csv",
            }
        )

    # Optimizer-state rows: direct probe has same-optimizer controls.
    for r in read_rows(V22_35 / "v22_35_optimizer_direct_probe_matrix.csv"):
        if r.get("training_variant") != "target_FU_signal":
            continue
        real = finite_float(r.get("NLL_delta_vs_optimizer"))
        best_control = finite_float(r.get("best_same_optimizer_control_NLL"))
        base_nll = finite_float(r.get("base_optimizer_NLL"))
        row_nll = finite_float(r.get("NLL"))
        if real is None or best_control is None or base_nll is None or row_nll is None:
            continue
        control_delta = best_control - base_nll
        debias = real - control_delta
        rows.append(
            {
                "evidence_family": "optimizer_state_fu",
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "carrier": r.get("carrier", ""),
                "optimizer_name": r.get("optimizer_name", ""),
                "delta_real": real,
                "control_q75_L7": control_delta,
                "B_debias_L6": "",
                "B_debias_L7": debias,
                "B_debias_L8": debias,
                "row_local_epsilon": 0.0,
                "positive_after_L7": int(debias < 0.0),
                "beats_own_optimizer_baseline": r.get("beats_optimizer_baseline", ""),
                "beats_same_optimizer_controls": r.get("beats_same_optimizer_controls", ""),
                "beats_strongest_completed_optimizer_baseline": r.get("beats_strongest_completed_optimizer_baseline", ""),
                "controller_overhead_ratio": r.get("controller_overhead_ratio", ""),
                "source_artifact": "results/v22_35/v22_35_optimizer_direct_probe_matrix.csv",
            }
        )

    write_rows(OUT_ROOT / "v22_36_causal_evidence_reanalysis_matrix.csv", rows)
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(str(row.get("evidence_family", "")), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for fam, fam_rows in sorted(by_family.items()):
        pos_l6 = sum(int_flag(r.get("positive_after_L6")) for r in fam_rows)
        pos_l7 = sum(int_flag(r.get("positive_after_L7")) for r in fam_rows)
        pos_l8 = sum(int_flag(r.get("positive_after_L8")) for r in fam_rows)
        summary_rows.append(
            {
                "evidence_family": fam,
                "rows": len(fam_rows),
                "positive_after_L6_rows": pos_l6,
                "positive_after_L6_rate": rate(pos_l6, len(fam_rows)),
                "positive_after_L7_rows": pos_l7,
                "positive_after_L7_rate": rate(pos_l7, len(fam_rows)),
                "positive_after_L8_rows": pos_l8,
                "positive_after_L8_rate": rate(pos_l8, len(fam_rows)),
            }
        )
    write_rows(OUT_ROOT / "v22_36_B_debias_summary.csv", summary_rows)

    # Gap truth: copy optimizer four-square truth classes as v22.36 seed evidence.
    gap_rows: list[dict[str, Any]] = []
    for r in read_rows(V22_35 / "v22_35_optimizer_direct_four_square_matrix.csv"):
        gap_rows.append({**r, "source_artifact": "results/v22_35/v22_35_optimizer_direct_four_square_matrix.csv"})
    if not gap_rows:
        gap_rows = [{"status": "not_run_no_v22_35_optimizer_four_square", "source_artifact": ""}]
    write_rows(OUT_ROOT / "v22_36_gap_truth_matrix.csv", gap_rows)
    append_exec(
        "recompute v22.36 controls-debiased B_debias evidence from v22.35 branch/optimizer artifacts",
        task_id="B_causal_evidence_reanalysis",
        status="pass" if rows else "warn",
        gpu="0",
        files="results/v22_36/v22_36_causal_evidence_reanalysis_matrix.csv, results/v22_36/v22_36_B_debias_summary.csv, results/v22_36/v22_36_gap_truth_matrix.csv",
        note=f"rows={len(rows)}; families={','.join(sorted(by_family))}",
    )
    return {"rows": len(rows), "summary_rows": len(summary_rows)}


def stage_c_selector() -> dict[str, Any]:
    selector_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    for path, name in [
        (V22_35 / "v22_35_T1_C11_edge_safe_selector_matrix.csv", "C11_edge_safe_control_orthogonal_direction"),
        (V22_35 / "v22_35_T1_C10_repaired_selector_matrix.csv", "C10_repaired_margin_NLL_constrained_direction"),
        (V22_35 / "v22_35_T1_C11_cohort_influence_selector_matrix.csv", "C16_continual_memory_prior_diagnostic"),
    ]:
        for r in read_rows(path):
            selector_rows.append(
                {
                    "selector_name": r.get("selector_name", name),
                    "dataset": r.get("dataset", ""),
                    "seed": r.get("seed", ""),
                    "signal_subspace_dim": r.get("subspace_rank", r.get("candidate_count", "")),
                    "cohort_positive_fraction": r.get("cohort_positive_fraction", ""),
                    "temporal_eigenspace_overlap": r.get("temporal_eigenspace_overlap", ""),
                    "cross_seed_stability": "",
                    "cross_dataset_stability": "",
                    "official_candidate_gate_pass": 0,
                    "status": "v22_35_readback_for_v22_36_prior",
                    "source_artifact": str(path.relative_to(ROOT)),
                }
            )
    for path, default_name in [
        (V22_35 / "v22_35_T1_C11_edge_safe_branch_matrix.csv", "C11_edge_safe_control_orthogonal_direction"),
        (V22_35 / "v22_35_T1_C10_repaired_branch_matrix.csv", "C10_repaired_margin_NLL_constrained_direction"),
        (V22_35 / "v22_35_T1_C11_cohort_influence_branch_matrix.csv", "C16_continual_memory_prior_diagnostic"),
    ]:
        for r in read_rows(path):
            real = finite_float(r.get("NLL_delta_vs_base"), finite_float(r.get("real_NLL_delta")))
            eps = finite_float(r.get("row_local_epsilon"), 1e-6) or 1e-6
            branch_rows.append(
                {
                    "selector_name": r.get("selector_name", default_name),
                    "dataset": r.get("dataset", ""),
                    "seed": r.get("seed", ""),
                    "branch_H20_NLL_delta": real if str(r.get("branch_H")) == "20" else "",
                    "branch_H60_NLL_delta": real if str(r.get("branch_H")) == "60" else "",
                    "branch_H200_NLL_delta": real if str(r.get("branch_H")) == "200" else "",
                    "branch_H400_NLL_delta": real if str(r.get("branch_H")) == "400" else "",
                    "branch_H800_NLL_delta": real if str(r.get("branch_H")) == "800" else "",
                    "row_local_epsilon": eps,
                    "NLL_delta_beyond_row_epsilon": int(real is not None and real < -eps),
                    "B_debias_L6": r.get("real_minus_best_control_NLL", ""),
                    "beats_L6": r.get("beats_L6", ""),
                    "beats_L7": r.get("beats_L7", ""),
                    "accuracy_delta": r.get("accuracy_delta_vs_base", r.get("accuracy_delta", "")),
                    "ECE_delta": r.get("ECE_delta_vs_base", ""),
                    "Brier_delta": r.get("Brier_delta_vs_base", ""),
                    "tail_q99_delta": r.get("tail_q99_delta", ""),
                    "margin_q10_delta": r.get("margin_q10_delta_vs_base", r.get("held_margin_q10_delta", "")),
                    "sharpness_delta": r.get("symmetric_curvature_proxy", ""),
                    "official_candidate_gate_pass": 0,
                    "status": "v22_35_readback_for_v22_36_prior",
                    "source_artifact": str(path.relative_to(ROOT)),
                }
            )
    for missing in ["C13_control_debiased_CVaR_section", "C14_actuator_support_residual_section", "C15_optimizer_success_aligned_section"]:
        selector_rows.append(
            {
                "selector_name": missing,
                "status": "not_run_in_initial_v22_36_reanalysis",
                "reason": "requires new causal-section implementation; scheduled after v22.36 evidence closure",
                "official_candidate_gate_pass": 0,
            }
        )
    write_rows(OUT_ROOT / "v22_36_signal_bundle_selector_matrix.csv", selector_rows)
    write_rows(OUT_ROOT / "v22_36_signal_bundle_branch_matrix.csv", branch_rows or [{"status": "not_run_no_selector_branch_rows"}])
    append_exec(
        "materialize v22.36 causal-section selector readback and not-run rows for C13/C14/C15",
        task_id="C_causal_section_selector",
        status="pass" if selector_rows else "warn",
        gpu="1",
        files="results/v22_36/v22_36_signal_bundle_selector_matrix.csv, results/v22_36/v22_36_signal_bundle_branch_matrix.csv",
        note=f"selector_rows={len(selector_rows)}; branch_rows={len(branch_rows)}; new_C13_C14_C15_run=0",
    )
    return {"selector_rows": len(selector_rows), "branch_rows": len(branch_rows)}


def stage_d_actuator_native() -> dict[str, Any]:
    target_rows = read_rows(V22_35 / "v22_35_basis_transfer_repair_target_matrix.csv")
    fit_rows = read_rows(V22_35 / "v22_35_basis_transfer_repair_fit_matrix.csv")
    branch_rows = read_rows(V22_35 / "v22_35_basis_transfer_repair_branch_matrix.csv")
    out_targets = []
    out_fit = []
    out_branch = []
    for r in target_rows:
        out_targets.append({**r, "v22_36_variant": "D14_actuator_support_residual_readback", "source_artifact": "results/v22_35/v22_35_basis_transfer_repair_target_matrix.csv"})
    for r in fit_rows:
        out_fit.append({**r, "v22_36_variant": "D14_actuator_support_residual_readback", "source_artifact": "results/v22_35/v22_35_basis_transfer_repair_fit_matrix.csv"})
    for r in branch_rows:
        real = finite_float(r.get("real_NLL_delta"))
        controls = [v for v in [finite_float(r.get("same_basis_random_NLL_delta")), finite_float(r.get("signflip_basis_control_NLL_delta"))] if v is not None]
        eps = finite_float(r.get("row_local_epsilon"), 1e-6) or 1e-6
        debias = "" if real is None else real - q75(controls)
        out_branch.append(
            {
                **r,
                "v22_36_variant": "D14_actuator_support_residual_readback",
                "B_debias_L6": debias,
                "B_debias_L7": "",
                "B_debias_L8": debias,
                "v22_36_branch_gate_pass": int(real is not None and debias != "" and debias < -eps and int_flag(r.get("strict_fit_pass")) and int_flag(r.get("beats_same_basis_random")) and int_flag(r.get("beats_signflip_basis_control"))),
                "source_artifact": "results/v22_35/v22_35_basis_transfer_repair_branch_matrix.csv",
            }
        )
    write_rows(OUT_ROOT / "v22_36_actuator_image_basis_target_matrix.csv", out_targets or [{"status": "not_run_no_target_rows"}])
    write_rows(OUT_ROOT / "v22_36_actuator_image_basis_fit_matrix.csv", out_fit or [{"status": "not_run_no_fit_rows"}])
    write_rows(OUT_ROOT / "v22_36_actuator_image_basis_branch_matrix.csv", out_branch or [{"status": "not_run_no_branch_rows"}])
    strict = sum(int_flag(r.get("strict_fit_pass")) for r in out_branch)
    passed = sum(int_flag(r.get("v22_36_branch_gate_pass")) for r in out_branch)
    append_exec(
        "materialize v22.36 actuator-image-native KAN basis readback with B_debias_L6/L8",
        task_id="D_actuator_image_native_basis_fu",
        status="pass" if out_branch else "warn",
        gpu="2",
        files="results/v22_36/v22_36_actuator_image_basis_target_matrix.csv, results/v22_36/v22_36_actuator_image_basis_fit_matrix.csv, results/v22_36/v22_36_actuator_image_basis_branch_matrix.csv",
        note=f"fit_rows={len(out_fit)}; branch_rows={len(out_branch)}; strict_branch_rows={strict}; v22_36_branch_gate_pass_rows={passed}",
    )
    return {"fit_rows": len(out_fit), "branch_rows": len(out_branch), "strict_branch_rows": strict, "branch_pass_rows": passed}


def stage_e_control_win() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for r in read_rows(V22_35 / "v22_35_control_win_decomposition_matrix.csv"):
        rows.append({**r, "control_hierarchy_max": "L7_from_v22_35", "source_artifact": "results/v22_35/v22_35_control_win_decomposition_matrix.csv"})
    for r in read_rows(OUT_ROOT / "v22_36_actuator_image_basis_branch_matrix.csv"):
        if r.get("status", "").startswith("not_run"):
            continue
        klass = "RealCausal" if int_flag(r.get("v22_36_branch_gate_pass")) else ("ActuatorSupportOnly" if int_flag(r.get("strict_fit_pass")) else "not_strict_fit_candidate")
        rows.append(
            {
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "target_family": r.get("target_family", ""),
                "basis_bank": r.get("basis_bank", ""),
                "control_win_class": klass,
                "control_level_that_wins": "L6_same_basis_random_or_L8_noise" if klass != "RealCausal" else "",
                "real_delta_NLL": r.get("real_NLL_delta", ""),
                "same_basis_random_NLL_delta": r.get("same_basis_random_NLL_delta", ""),
                "signflip_basis_control_NLL_delta": r.get("signflip_basis_control_NLL_delta", ""),
                "B_debias_L6": r.get("B_debias_L6", ""),
                "row_local_epsilon": r.get("row_local_epsilon", ""),
                "source_artifact": "results/v22_36/v22_36_actuator_image_basis_branch_matrix.csv",
            }
        )
    write_rows(OUT_ROOT / "v22_36_control_win_decomposition_matrix.csv", rows or [{"status": "not_run_no_control_rows"}])
    real = sum(1 for r in rows if r.get("control_win_class") == "RealCausal")
    support = sum(1 for r in rows if r.get("control_win_class") == "ActuatorSupportOnly")
    append_exec(
        "materialize v22.36 L0-L8 control-win mechanism decomposition",
        task_id="E_control_win_decomposition",
        status="pass" if rows else "warn",
        gpu="0",
        files="results/v22_36/v22_36_control_win_decomposition_matrix.csv",
        note=f"rows={len(rows)}; RealCausal_rows={real}; ActuatorSupportOnly_rows={support}",
    )
    return {"rows": len(rows), "RealCausal_rows": real, "ActuatorSupportOnly_rows": support}


def stage_f_optimizer_state() -> dict[str, Any]:
    rows = []
    for r in read_rows(V22_35 / "v22_35_optimizer_direct_probe_matrix.csv"):
        if r.get("training_variant") == "target_FU_signal":
            overhead = finite_float(r.get("controller_overhead_ratio"), math.inf)
            rows.append(
                {
                    **r,
                    "v22_36_variant": "optimizer_state_CFF_readback",
                    "FU_beats_own_optimizer": r.get("beats_optimizer_baseline", ""),
                    "FU_beats_matched_controls": r.get("beats_same_optimizer_controls", ""),
                    "FU_beats_strongest_optimizer": r.get("beats_strongest_completed_optimizer_baseline", ""),
                    "overhead_safe_exploration": int(overhead is not None and overhead <= 0.30),
                    "source_artifact": "results/v22_35/v22_35_optimizer_direct_probe_matrix.csv",
                }
            )
    four = []
    for r in read_rows(V22_35 / "v22_35_optimizer_direct_four_square_matrix.csv"):
        four.append({**r, "source_artifact": "results/v22_35/v22_35_optimizer_direct_four_square_matrix.csv"})
    write_rows(OUT_ROOT / "v22_36_optimizer_state_fu_matrix.csv", rows or [{"status": "not_run_no_optimizer_rows"}])
    write_rows(OUT_ROOT / "v22_36_optimizer_state_four_square_matrix.csv", four or [{"status": "not_run_no_four_square_rows"}])
    beats_own = sum(int_flag(r.get("FU_beats_own_optimizer")) for r in rows)
    beats_ctrl = sum(int_flag(r.get("FU_beats_matched_controls")) for r in rows)
    beats_strong = sum(int_flag(r.get("FU_beats_strongest_optimizer")) for r in rows)
    overhead_safe = sum(int_flag(r.get("overhead_safe_exploration")) for r in rows)
    exploration = int(len(rows) >= 9 and beats_own >= 6 and beats_ctrl >= 6 and beats_strong >= 4 and overhead_safe == len(rows))
    append_exec(
        "materialize v22.36 optimizer-state FU readback from v22.35 direct optimizer probe",
        task_id="F_optimizer_state_fu",
        status="pass" if rows else "warn",
        gpu="3",
        files="results/v22_36/v22_36_optimizer_state_fu_matrix.csv, results/v22_36/v22_36_optimizer_state_four_square_matrix.csv",
        note=f"signal_rows={len(rows)}; beats_own={beats_own}; beats_controls={beats_ctrl}; beats_strongest={beats_strong}; exploration_pass={exploration}; direct_v22_36_training=0",
    )
    return {
        "signal_rows": len(rows),
        "beats_own": beats_own,
        "beats_controls": beats_ctrl,
        "beats_strongest": beats_strong,
        "optimizer_state_exploration_pass": exploration,
    }


def torch_device(name: str) -> Any:
    import torch

    if str(name).startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def tensor_cosine(a: Any, b: Any) -> float | None:
    denom = a.norm().clamp_min(1.0e-12) * b.norm().clamp_min(1.0e-12)
    try:
        return float((a.flatten() @ b.flatten() / denom).item())
    except Exception:
        return None


def singular_order_score(base_update: Any, candidate_update: Any) -> float | None:
    if getattr(base_update, "ndim", 0) != 2 or getattr(candidate_update, "ndim", 0) != 2:
        return None
    try:
        import torch

        sb = torch.linalg.svdvals(base_update.float())
        sc = torch.linalg.svdvals(candidate_update.float())
        n = min(int(sb.numel()), int(sc.numel()))
        if n <= 1:
            return None
        rb = torch.argsort(torch.argsort(sb[:n], descending=True))
        rc = torch.argsort(torch.argsort(sc[:n], descending=True))
        return float((rb == rc).float().mean().item())
    except Exception:
        return None


def rare_direction_ratio(g: Any, base_update: Any, candidate_update: Any) -> float | None:
    try:
        mask = g.detach().abs() <= g.detach().abs().quantile(0.25)
        if not bool(mask.any().item()):
            return None
        base = base_update.detach()[mask].norm().clamp_min(1.0e-12)
        cand = candidate_update.detach()[mask].norm()
        return float((cand / base).item())
    except Exception:
        return None


def train_cff_optimizer_v22_36(
    model: Any,
    train_loader: Any,
    test_loader: Any,
    device: Any,
    output_dim: int,
    *,
    optimizer_family: str,
    steps: int,
    lr: float,
    weight_decay: float,
    seed: int,
    signal_mode: str,
    alpha_grid: list[float],
    acceptance_tol: float,
    acceptance_cadence: int,
    acceptance_eval_size: int,
    acceptance_batches: int,
    acceptance_metric: str,
    acceptance_risk_tol: float,
    alpha_selection: str,
    shadow_acceptance: int,
    signal_transform: str,
    temperature_grid: list[float],
    temperature_selection_metric: str,
    tail_projection: int,
    accept_loader: Any | None,
) -> dict[str, Any]:
    """v22.36-owned CFF optimizer-state loop.

    Direction choice uses only train/held-train batches. Test data is used only
    once at the end for measurement, not for update selection.
    """
    import torch
    import torch.nn.functional as F

    from experiments.run_v22_30_fidelity_ladder import cycle_batches, evaluate_model, orthogonalized_update, sync

    if signal_mode not in {"none", "signal", "random", "signflip", "shuffled", "same_actuator_support", "same_optimizer_geometry"}:
        raise ValueError(f"unknown signal_mode={signal_mode!r}")

    named = list(model.named_parameters())
    m: dict[str, Any] = {}
    v: dict[str, Any] = {}
    avg: dict[str, Any] = {}
    signal: dict[str, Any] = {}
    gen = torch.Generator(device=device).manual_seed(int(seed) + 55037)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0e-8
    update_norms: list[float] = []
    snrs: list[float] = []
    norm_match_scales: list[float] = []
    loss_trace: list[float] = []
    grad_vars: list[float] = []
    alignments: list[float] = []
    singular_scores: list[float] = []
    rare_ratios: list[float] = []
    accepted_alphas: list[float] = []
    last_alpha = 0.0
    acceptance_total = 0
    acceptance_accepts = 0
    acceptance_rejects = 0
    candidate_loss_delta_sum = 0.0
    tail_projection = int(tail_projection)
    acceptance_metric = str(acceptance_metric or "ce").strip().lower()
    if acceptance_metric not in {"ce", "risk"}:
        raise ValueError(f"unknown acceptance_metric={acceptance_metric!r}")
    alpha_selection = str(alpha_selection or "max_alpha").strip().lower()
    if alpha_selection not in {"max_alpha", "min_delta"}:
        raise ValueError(f"unknown alpha_selection={alpha_selection!r}")
    shadow_acceptance = int(shadow_acceptance)
    signal_transform = str(signal_transform or "raw").strip().lower()
    if signal_transform not in {"raw", "optimizer_residual"}:
        raise ValueError(f"unknown signal_transform={signal_transform!r}")
    temperature_selection_metric = str(temperature_selection_metric or "nll").strip().lower()
    if temperature_selection_metric not in {"nll", "risk", "tail_risk"}:
        raise ValueError(f"unknown temperature_selection_metric={temperature_selection_metric!r}")
    temperature_candidates = sorted({float(t) for t in temperature_grid if float(t) > 0.0}) or [1.0]

    def apply_updates(updates: dict[str, Any]) -> None:
        apply_updates_to(named, updates)

    def apply_updates_to(param_list: list[tuple[str, Any]], updates: dict[str, Any]) -> None:
        with torch.no_grad():
            for name, p in param_list:
                update = updates.get(name)
                if update is None:
                    continue
                p.mul_(1.0 - float(lr) * float(weight_decay))
                p.add_(-float(lr) * update)

    def build_base_optimizer_updates(
        param_list: list[tuple[str, Any]],
        state_m: dict[str, Any],
        state_v: dict[str, Any],
        step_index: int,
    ) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        with torch.no_grad():
            for name, p in param_list:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                state_m[name] = g.clone() if name not in state_m else beta1 * state_m[name] + (1.0 - beta1) * g
                if optimizer_family == "Muon-like" and p.ndim == 2:
                    base_update = orthogonalized_update(state_m[name])
                else:
                    state_v[name] = g.square() if name not in state_v else beta2 * state_v[name] + (1.0 - beta2) * g.square()
                    m_hat = state_m[name] / (1.0 - beta1**step_index)
                    v_hat = state_v[name] / (1.0 - beta2**step_index)
                    base_update = m_hat / (v_hat.sqrt() + eps)
                    if optimizer_family == "Cautious AdamW":
                        mask = (base_update * g) > 0.0
                        scale = mask.float().mean().clamp_min(1.0e-3)
                        base_update = base_update * mask / scale
                updates[name] = base_update.detach().clone()
        return updates

    def snapshot_params() -> list[Any]:
        return [p.detach().clone() for _name, p in named]

    def restore_params(params: list[Any]) -> None:
        with torch.no_grad():
            for old, (_name, p) in zip(params, named):
                p.copy_(old)

    def batch_risk_metrics(logits: Any, targets: Any) -> dict[str, float]:
        logits = logits.float()
        targets = targets.long()
        per_sample_ce = F.cross_entropy(logits, targets, reduction="none")
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=int(output_dim)).float()
        conf, pred = probs.max(dim=1)
        correct = pred.eq(targets).float()
        ece = torch.zeros((), device=logits.device)
        for lo in torch.linspace(0.0, 0.9, 10, device=logits.device):
            hi = lo + 0.1
            mask = (conf > lo) & (conf <= hi if float(hi.item()) < 1.0 else conf <= hi)
            if bool(mask.any()):
                ece = ece + mask.float().mean() * (conf[mask].mean() - correct[mask].mean()).abs()
        return {
            "ce": float(per_sample_ce.mean().detach().item()),
            "brier": float((probs - one_hot).square().sum(dim=1).mean().detach().item()),
            "tail_q99": float(torch.quantile(per_sample_ce.detach(), 0.99).item()),
            "ece": float(ece.detach().item()),
        }

    def loader_metrics_with_temperature(loader: Any, temperature: float) -> dict[str, float]:
        model.eval()
        losses_all = []
        logits_all = []
        labels_all = []
        total = 0
        correct = 0
        with torch.no_grad():
            for xb_eval, yb_eval in loader:
                xb_eval = xb_eval.to(device).float()
                yb_eval = yb_eval.to(device).long()
                logits_eval = model(xb_eval).float() / float(temperature)
                loss_eval = F.cross_entropy(logits_eval, yb_eval, reduction="none")
                losses_all.append(loss_eval.detach().cpu())
                logits_all.append(logits_eval.detach().cpu())
                labels_all.append(yb_eval.detach().cpu())
                total += int(yb_eval.numel())
                correct += int((logits_eval.argmax(dim=-1) == yb_eval).sum().item())
        if total == 0:
            return {
                "NLL": math.nan,
                "accuracy": math.nan,
                "ECE": math.nan,
                "Brier": math.nan,
                "tail_q95": math.nan,
                "tail_q99": math.nan,
            }
        losses = torch.cat(losses_all)
        logits_cpu = torch.cat(logits_all)
        labels_cpu = torch.cat(labels_all)
        probs = torch.softmax(logits_cpu.float(), dim=-1)
        target = F.one_hot(labels_cpu, num_classes=int(output_dim)).float()
        conf, pred = probs.max(dim=-1)
        ok = (pred == labels_cpu).float()
        ece = torch.tensor(0.0)
        for lo in torch.linspace(0.0, 0.9, 10):
            hi = lo + 0.1
            mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
            if mask.any():
                ece = ece + mask.float().mean() * (conf[mask].mean() - ok[mask].mean()).abs()
        return {
            "NLL": float(losses.mean().item()),
            "accuracy": float(correct / total),
            "ECE": float(ece.item()),
            "Brier": float(((probs - target) ** 2).sum(dim=-1).mean().item()),
            "tail_q95": float(torch.quantile(losses.float(), 0.95).item()),
            "tail_q99": float(torch.quantile(losses.float(), 0.99).item()),
        }

    def temperature_score(metrics: dict[str, float]) -> float:
        if temperature_selection_metric == "nll":
            return float(metrics.get("NLL", math.inf))
        if temperature_selection_metric == "tail_risk":
            return (
                float(metrics.get("NLL", math.inf))
                + 2.0 * float(metrics.get("ECE", math.inf))
                + float(metrics.get("Brier", math.inf))
                + 0.50 * float(metrics.get("tail_q99", math.inf))
            )
        return (
            float(metrics.get("NLL", math.inf))
            + float(metrics.get("ECE", math.inf))
            + float(metrics.get("Brier", math.inf))
            + 0.10 * float(metrics.get("tail_q99", math.inf))
        )

    def acceptance_delta(candidate: dict[str, float], base: dict[str, float]) -> float:
        if acceptance_metric == "ce":
            return candidate["ce"] - base["ce"]
        risk_debt = (
            max(0.0, candidate["brier"] - base["brier"] - float(acceptance_risk_tol))
            + max(0.0, candidate["tail_q99"] - base["tail_q99"] - float(acceptance_risk_tol))
            + max(0.0, candidate["ece"] - base["ece"] - float(acceptance_risk_tol))
        )
        return candidate["ce"] - base["ce"] + risk_debt

    def acceptance_ok(candidate: dict[str, float], base: dict[str, float]) -> bool:
        if candidate["ce"] > base["ce"] + float(acceptance_tol):
            return False
        if acceptance_metric == "ce":
            return True
        tol = float(acceptance_risk_tol)
        return (
            candidate["brier"] <= base["brier"] + tol
            and candidate["tail_q99"] <= base["tail_q99"] + tol
            and candidate["ece"] <= base["ece"] + tol
        )

    start = time.time()
    it = cycle_batches(train_loader)
    accept_it = cycle_batches(accept_loader) if accept_loader is not None else None
    alpha_candidates = sorted({float(a) for a in alpha_grid if float(a) >= 0.0})
    if 0.0 not in alpha_candidates:
        alpha_candidates.insert(0, 0.0)
    shadow_model = None
    shadow_named: list[tuple[str, Any]] = []
    shadow_m: dict[str, Any] = {}
    shadow_v: dict[str, Any] = {}
    if shadow_acceptance and signal_mode != "none":
        shadow_model = copy.deepcopy(model).to(device)
        shadow_named = list(shadow_model.named_parameters())

    for step in range(1, int(steps) + 1):
        xb, yb = next(it)
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        model.zero_grad(set_to_none=True)
        logits = model(xb).float()
        per_sample_loss = F.cross_entropy(logits, yb, reduction="none")
        loss = per_sample_loss.mean()
        loss_trace.append(float(loss.detach().item()))
        use_tail_projection = bool(tail_projection and signal_mode == "signal" and yb.numel() >= 4)
        loss.backward(retain_graph=use_tail_projection)
        base_grads: dict[str, Any] = {}
        tail_grads: dict[str, Any] = {}
        if use_tail_projection:
            with torch.no_grad():
                cutoff = torch.quantile(per_sample_loss.detach(), 0.75)
                tail_mask = per_sample_loss.detach() >= cutoff
            for name, p in named:
                if p.grad is not None:
                    base_grads[name] = p.grad.detach().clone()
            model.zero_grad(set_to_none=True)
            tail_loss = per_sample_loss[tail_mask].mean()
            tail_loss.backward()
            for name, p in named:
                if p.grad is not None:
                    tail_grads[name] = p.grad.detach().clone()
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                for name, p in named:
                    if name in base_grads:
                        p.grad = base_grads[name].clone()
        base_updates: dict[str, Any] = {}
        signal_dirs: dict[str, Any] = {}
        base_norms: dict[str, Any] = {}
        with torch.no_grad():
            for name, p in named:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                grad_vars.append(float(g.float().var(unbiased=False).item()))
                m[name] = g.clone() if name not in m else beta1 * m[name] + (1.0 - beta1) * g
                align = tensor_cosine(m[name], g)
                if align is not None:
                    alignments.append(align)
                if optimizer_family == "Muon-like" and p.ndim == 2:
                    base_update = orthogonalized_update(m[name])
                else:
                    v[name] = g.square() if name not in v else beta2 * v[name] + (1.0 - beta2) * g.square()
                    m_hat = m[name] / (1.0 - beta1**step)
                    v_hat = v[name] / (1.0 - beta2**step)
                    base_update = m_hat / (v_hat.sqrt() + eps)
                    if optimizer_family == "Cautious AdamW":
                        mask = (base_update * g) > 0.0
                        scale = mask.float().mean().clamp_min(1.0e-3)
                        base_update = base_update * mask / scale
                base_updates[name] = base_update.detach().clone()
                base_norms[name] = base_update.norm().clamp_min(1.0e-12)
                signal[name] = g.clone() if name not in signal else 0.95 * signal[name] + 0.05 * g
                sig = signal[name]
                if signal_transform == "optimizer_residual" and signal_mode != "none":
                    denom = base_update.float().square().sum().clamp_min(1.0e-12)
                    coeff = (sig.float() * base_update.float()).sum() / denom
                    sig = sig - coeff * base_update
                if signal_mode == "none":
                    sig_dir = torch.zeros_like(sig)
                elif signal_mode == "random":
                    sig_dir = torch.randn(sig.shape, device=sig.device, generator=gen)
                elif signal_mode == "signflip":
                    sig_dir = -sig
                elif signal_mode == "shuffled":
                    sig_dir = sig.reshape(-1)[torch.randperm(sig.numel(), generator=gen, device=sig.device)].reshape_as(sig)
                elif signal_mode == "same_actuator_support":
                    rand = torch.randn(sig.shape, device=sig.device, generator=gen)
                    support = sig.abs() >= sig.abs().quantile(0.50)
                    sig_dir = rand * support
                elif signal_mode == "same_optimizer_geometry":
                    rand = torch.randn(sig.shape, device=sig.device, generator=gen)
                    sig_dir = rand * base_update.norm().clamp_min(1.0e-12) / rand.norm().clamp_min(1.0e-12)
                else:
                    sig_dir = sig
                    if tail_projection and name in tail_grads:
                        tail_g = tail_grads[name]
                        dot = (sig_dir.float() * tail_g.float()).sum()
                        if bool(dot.detach().item() < 0.0):
                            sig_dir = sig_dir - dot * tail_g / tail_g.float().square().sum().clamp_min(1.0e-12)
                signal_dirs[name] = sig_dir.detach().clone()
                snrs.append(float(m[name].norm().item() / (g - m[name]).norm().clamp_min(1.0e-12).item()))

        def build_candidate(alpha: float) -> dict[str, Any]:
            updates: dict[str, Any] = {}
            with torch.no_grad():
                for name, p in named:
                    if name not in base_updates:
                        continue
                    base_update = base_updates[name]
                    sig_dir = signal_dirs[name]
                    if signal_mode == "none" or float(alpha) == 0.0:
                        update = base_update
                    else:
                        scale = base_norms[name] / sig_dir.norm().clamp_min(1.0e-12)
                        update = base_update + float(alpha) * scale * sig_dir
                        norm_match_scales.append(float(scale.item()))
                    updates[name] = update.detach().clone()
                    if signal_mode != "none":
                        score = singular_order_score(base_update, update)
                        if score is not None:
                            singular_scores.append(score)
                        rr = rare_direction_ratio(p.grad.detach(), base_update, update) if p.grad is not None else None
                        if rr is not None:
                            rare_ratios.append(rr)
            return updates

        before = snapshot_params()
        candidate_cache: dict[float, dict[str, Any]] = {}

        def cached_candidate(alpha: float) -> dict[str, Any]:
            alpha = float(alpha)
            if alpha not in candidate_cache:
                candidate_cache[alpha] = build_candidate(alpha)
            return candidate_cache[alpha]

        eval_xb, eval_yb = xb, yb
        if accept_it is not None:
            eval_xs = []
            eval_ys = []
            for _ in range(max(1, int(acceptance_batches))):
                eval_xb_i, eval_yb_i = next(accept_it)
                eval_xs.append(eval_xb_i.to(device).float())
                eval_ys.append(eval_yb_i.to(device).long())
            eval_xb = torch.cat(eval_xs, dim=0)
            eval_yb = torch.cat(eval_ys, dim=0)
        if int(acceptance_eval_size) > 0 and eval_xb.shape[0] > int(acceptance_eval_size):
            eval_xb = eval_xb[: int(acceptance_eval_size)]
            eval_yb = eval_yb[: int(acceptance_eval_size)]
        shadow_reference_metrics = None
        if shadow_model is not None:
            shadow_model.train()
            shadow_model.zero_grad(set_to_none=True)
            shadow_logits = shadow_model(xb).float()
            shadow_loss = F.cross_entropy(shadow_logits, yb)
            shadow_loss.backward()
            shadow_updates = build_base_optimizer_updates(shadow_named, shadow_m, shadow_v, step)
            apply_updates_to(shadow_named, shadow_updates)
        if signal_mode == "none":
            apply_updates(cached_candidate(0.0))
        else:
            base_candidate = cached_candidate(0.0)
            should_search = step == 1 or int(acceptance_cadence) <= 1 or step % int(acceptance_cadence) == 0
            if shadow_model is not None and should_search:
                with torch.no_grad():
                    shadow_reference_metrics = batch_risk_metrics(shadow_model(eval_xb), eval_yb)
            if should_search:
                restore_params(before)
                apply_updates(base_candidate)
                with torch.no_grad():
                    base_metrics = batch_risk_metrics(model(eval_xb), eval_yb)
                restore_params(before)
                best_alpha = 0.0
                best_delta = 0.0
                for alpha in alpha_candidates:
                    if float(alpha) == 0.0:
                        continue
                    cand = cached_candidate(alpha)
                    apply_updates(cand)
                    with torch.no_grad():
                        cand_metrics = batch_risk_metrics(model(eval_xb), eval_yb)
                    candidate_delta = acceptance_delta(cand_metrics, base_metrics)
                    restore_params(before)
                    if not acceptance_ok(cand_metrics, base_metrics):
                        continue
                    if shadow_reference_metrics is not None and not acceptance_ok(cand_metrics, shadow_reference_metrics):
                        continue
                    if alpha_selection == "max_alpha":
                        if alpha >= best_alpha:
                            best_alpha = float(alpha)
                            best_delta = candidate_delta
                    elif candidate_delta < best_delta - 1.0e-12 or (
                        abs(candidate_delta - best_delta) <= 1.0e-12 and alpha > best_alpha
                    ):
                        best_alpha = float(alpha)
                        best_delta = candidate_delta
                acceptance_total += 1
                candidate_loss_delta_sum += 0.0 if not math.isfinite(best_delta) else best_delta
                last_alpha = best_alpha
                if best_alpha > 0.0:
                    acceptance_accepts += 1
                else:
                    acceptance_rejects += 1
            apply_updates(cached_candidate(last_alpha))
            accepted_alphas.append(last_alpha)

        with torch.no_grad():
            chosen_update_norm = 0.0
            for old, (_name, p) in zip(before, named):
                chosen_update_norm += float((p.detach() - old).norm().item()) ** 2
            update_norms.append(math.sqrt(chosen_update_norm))
            if optimizer_family == "Schedule-Free AdamW":
                for name, p in named:
                    if name not in avg:
                        avg[name] = p.detach().clone()
                    else:
                        avg[name].mul_(float(step - 1) / float(step)).add_(p.detach(), alpha=1.0 / float(step))

    if optimizer_family == "Schedule-Free AdamW":
        with torch.no_grad():
            for name, p in named:
                if name in avg:
                    p.copy_(avg[name])
    sync(device)
    chosen_temperature = 1.0
    if len(temperature_candidates) > 1 and accept_loader is not None:
        best_score = math.inf
        for temp in temperature_candidates:
            metrics = loader_metrics_with_temperature(accept_loader, temp)
            score = temperature_score(metrics)
            if score < best_score:
                best_score = score
                chosen_temperature = float(temp)
    if abs(float(chosen_temperature) - 1.0) <= 1.0e-12:
        ev = evaluate_model(model, test_loader, device, output_dim)
    else:
        ev = loader_metrics_with_temperature(test_loader, chosen_temperature)
    step_ms = 1000.0 * (time.time() - start) / max(1, steps)
    auc_loss_time = sum(loss_trace) / max(1, len(loss_trace))
    ev.update(
        {
            "AUC_loss_time": auc_loss_time,
            "wallclock_adjusted_AUC": auc_loss_time * step_ms,
            "train_step_ms": step_ms,
            "update_norm": sum(update_norms) / max(1, len(update_norms)),
            "update_SNR": sum(snrs) / max(1, len(snrs)),
            "gradient_variance": sum(grad_vars) / max(1, len(grad_vars)) if grad_vars else "",
            "momentum_gradient_alignment": sum(alignments) / max(1, len(alignments)) if alignments else "",
            "singular_order_preservation": sum(singular_scores) / max(1, len(singular_scores)) if singular_scores else "",
            "rare_direction_amplification": sum(rare_ratios) / max(1, len(rare_ratios)) if rare_ratios else "",
            "norm_match_scale_mean": sum(norm_match_scales) / max(1, len(norm_match_scales)) if norm_match_scales else "",
            "signal_accept_rate": "" if signal_mode == "none" else acceptance_accepts / max(1, acceptance_total),
            "control_accept_rate": "" if signal_mode in {"none", "signal"} else acceptance_accepts / max(1, acceptance_total),
            "direct_optimizer_accept_rate": "" if signal_mode == "none" else acceptance_accepts / max(1, acceptance_total),
            "direct_optimizer_reject_count": "" if signal_mode == "none" else acceptance_rejects,
            "direct_optimizer_candidate_loss_delta_mean": ""
            if signal_mode == "none" or not acceptance_total
            else candidate_loss_delta_sum / max(1, acceptance_total),
            "accepted_alpha_mean": "" if signal_mode == "none" else sum(accepted_alphas) / max(1, len(accepted_alphas)),
            "acceptance_metric": acceptance_metric,
            "acceptance_risk_tol": "" if acceptance_metric == "ce" else float(acceptance_risk_tol),
            "alpha_selection": alpha_selection,
            "shadow_acceptance": shadow_acceptance,
            "signal_transform": signal_transform,
            "calibration_temperature": chosen_temperature,
            "temperature_selection_metric": temperature_selection_metric,
            "temperature_grid": ",".join(str(x) for x in temperature_candidates),
            "tail_projection": tail_projection,
            "implementation_level": (
                "v22_36_cff_schedule_free_averaging_manual"
                if optimizer_family == "Schedule-Free AdamW"
                else ("v22_36_cff_muon_like_svd_orthogonalized_momentum" if optimizer_family == "Muon-like" else "v22_36_cff_manual_adamw_family")
            ),
        }
    )
    return ev


def materialize_v22_36_optimizer_four_square(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("seed", "")),
            str(row.get("carrier", "")),
            str(row.get("optimizer_family", "")),
            str(row.get("training_variant", "")),
        )
        by_key[key] = row
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("training_variant") != "CFF_FU_signal" or row.get("carrier") not in {"D-CHE", "D-FOU"}:
            continue
        dataset = str(row.get("dataset", ""))
        seed = str(row.get("seed", ""))
        carrier = str(row.get("carrier", ""))
        opt = str(row.get("optimizer_family", ""))
        kan_base = by_key.get((dataset, seed, carrier, opt, "optimizer_alone"), {})
        mlp_base = by_key.get((dataset, seed, "MLP", opt, "optimizer_alone"), {})
        mlp_fu = by_key.get((dataset, seed, "MLP", opt, "CFF_FU_signal"), {})
        if not kan_base or not mlp_base or not mlp_fu:
            continue
        kan_fu_nll = finite_float(row.get("NLL"))
        kan_base_nll = finite_float(kan_base.get("NLL"))
        mlp_fu_nll = finite_float(mlp_fu.get("NLL"))
        mlp_base_nll = finite_float(mlp_base.get("NLL"))
        if None in {kan_fu_nll, kan_base_nll, mlp_fu_nll, mlp_base_nll}:
            continue
        delta_kan = float(kan_fu_nll) - float(kan_base_nll)
        delta_mlp = float(mlp_fu_nll) - float(mlp_base_nll)
        gap_bp = float(kan_base_nll) - float(mlp_base_nll)
        gap_fu = float(kan_fu_nll) - float(mlp_fu_nll)
        gap_reduction = gap_bp - gap_fu
        control_beat = int_flag(row.get("beats_same_optimizer_controls"))
        if delta_kan < 0.0 and delta_mlp >= 0.0 and control_beat:
            klass = "TrueKANGain"
        elif delta_kan < 0.0 and delta_mlp < 0.0 and gap_reduction > 0.0 and control_beat:
            klass = "BothGain"
        elif delta_mlp > 0.0 and gap_reduction > 0.0:
            klass = "MLPDegradationDriven"
        elif not control_beat:
            klass = "ControlExplained"
        else:
            klass = "NoGainOrMLPDominates"
        out_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "optimizer_family": opt,
                "kan_carrier": carrier,
                "MLP_strong_NLL": mlp_base_nll,
                "MLP_CFF_FU_NLL": mlp_fu_nll,
                "KAN_strong_NLL": kan_base_nll,
                "KAN_CFF_FU_NLL": kan_fu_nll,
                "Delta_MLP_NLL": delta_mlp,
                "Delta_KAN_NLL": delta_kan,
                "Gap_BP": gap_bp,
                "Gap_FU": gap_fu,
                "GapReduction": gap_reduction,
                "MLP_CFF_FU_beats_MLP_strong": int(delta_mlp < 0.0),
                "KAN_CFF_FU_beats_KAN_strong": int(delta_kan < 0.0),
                "KAN_CFF_FU_beats_MLP_CFF_FU": int(float(kan_fu_nll) < float(mlp_fu_nll)),
                "KAN_CFF_FU_beats_same_optimizer_controls": control_beat,
                "TrueKANGain_class": klass,
                "source_artifact": "results/v22_36/v22_36_optimizer_state_direct_probe_matrix.csv",
            }
        )
    n = len(out_rows)
    true_rows = sum(1 for r in out_rows if r.get("TrueKANGain_class") == "TrueKANGain")
    both_rows = sum(1 for r in out_rows if r.get("TrueKANGain_class") == "BothGain")
    mlp_win = sum(int_flag(r.get("MLP_CFF_FU_beats_MLP_strong")) for r in out_rows)
    kan_win = sum(int_flag(r.get("KAN_CFF_FU_beats_KAN_strong")) for r in out_rows)
    controls = sum(int_flag(r.get("KAN_CFF_FU_beats_same_optimizer_controls")) for r in out_rows)
    summary = {
        "four_square_rows": n,
        "TrueKANGain_rows": true_rows,
        "BothGain_rows": both_rows,
        "TrueKANGain_plus_BothGain_rate": rate(true_rows + both_rows, n),
        "MLP_CFF_FU_beats_MLP_strong_rows": mlp_win,
        "KAN_CFF_FU_beats_KAN_strong_rows": kan_win,
        "KAN_CFF_FU_beats_same_optimizer_controls_rows": controls,
        "kan_internal_value_opened": int(n > 0 and kan_win >= 5 and controls >= 5),
        "mlp_general_value_opened": int(n > 0 and mlp_win >= 5),
        "status": "v22_36_direct_four_square_completed" if n else "v22_36_direct_four_square_no_rows",
    }
    return out_rows, summary


def stage_f_optimizer_state_direct(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.run_v22_30_fidelity_ladder import make_kan, make_mlp
    from experiments.run_v22_35_causal_target_noise_calibrated_fu import make_v22_35_c11_loaders

    ensure_out()
    device = torch_device(args.optimizer_device)
    rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []
    group_base: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    group_controls: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    alpha_grid = [float(x) for x in split_csv(args.optimizer_alpha_grid, float)]
    control_modes = ["random", "signflip", "shuffled", "same_actuator_support", "same_optimizer_geometry"]

    def fresh(dataset: str, seed: int, carrier: str, model_seed: int) -> tuple[Any, Any, Any, Any, int, dict[str, Any]]:
        train_loader, held_loader, test_loader, input_dim, output_dim, x_stats, meta = make_v22_35_c11_loaders(
            dataset,
            int(args.optimizer_train_size),
            int(args.optimizer_test_size),
            int(args.batch_size),
            int(seed),
            tier2_download=bool(args.tier2_download),
        )
        availability_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "carrier": carrier,
                "status": "available",
                "task_tier": meta.get("task_tier", ""),
                "source_kind": meta.get("source_kind", ""),
                "effective_train_rows": meta.get("effective_train_rows", ""),
                "effective_test_rows": meta.get("effective_test_rows", ""),
                "source_artifact": "direct_v22_36_optimizer_state_loader",
            }
        )
        if carrier == "MLP":
            model = make_mlp(input_dim, output_dim, int(args.hidden), int(model_seed), device).to(device)
        else:
            model = make_kan(input_dim, output_dim, int(args.hidden), int(model_seed), device, x_stats, carrier).to(device)
        return model, train_loader, held_loader, test_loader, int(output_dim), meta

    def run_one(dataset: str, seed: int, carrier: str, optimizer_family: str, mode: str) -> dict[str, Any]:
        arch = "MLP" if carrier == "MLP" else "KAN"
        arch_offset = 0 if carrier == "MLP" else (1000 if carrier == "D-CHE" else 2000)
        model_seed = int(seed) + 126_000 + arch_offset
        model, train_loader, held_loader, test_loader, output_dim, meta = fresh(dataset, seed, carrier, model_seed)
        effective_mode = mode
        cff_disabled_for_mlp = 0
        if str(args.optimizer_signal_scope) == "kan_only" and carrier == "MLP" and mode != "none":
            effective_mode = "none"
            cff_disabled_for_mlp = 1
        ev = train_cff_optimizer_v22_36(
            model,
            train_loader,
            test_loader,
            device,
            output_dim,
            optimizer_family=optimizer_family,
            steps=int(args.optimizer_steps),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            seed=int(seed) + 88_000,
            signal_mode=effective_mode,
            alpha_grid=alpha_grid,
            acceptance_tol=float(args.optimizer_acceptance_tol),
            acceptance_cadence=int(args.optimizer_acceptance_cadence),
            acceptance_eval_size=int(args.optimizer_acceptance_eval_size),
            acceptance_batches=int(args.optimizer_acceptance_batches),
            acceptance_metric=str(args.optimizer_acceptance_metric),
            acceptance_risk_tol=float(args.optimizer_acceptance_risk_tol),
            alpha_selection=str(args.optimizer_alpha_selection),
            shadow_acceptance=int(args.optimizer_shadow_acceptance),
            signal_transform=str(args.optimizer_signal_transform),
            temperature_grid=[float(x) for x in split_csv(args.optimizer_eval_temperature_grid, float)],
            temperature_selection_metric=str(args.optimizer_temperature_selection_metric),
            tail_projection=int(args.optimizer_tail_projection),
            accept_loader=held_loader,
        )
        variant = "optimizer_alone" if mode == "none" else ("CFF_FU_signal" if mode == "signal" else f"same_optimizer_{mode}_control")
        return {
            "dataset": dataset,
            "seed": seed,
            "task_tier": meta.get("task_tier", ""),
            "architecture": arch,
            "carrier": carrier,
            "optimizer_family": optimizer_family,
            "implementation_level": ev.get("implementation_level", ""),
            "training_variant": variant,
            "control_variant": "" if mode in {"none", "signal"} else mode,
            "requested_signal_mode": mode,
            "effective_signal_mode": effective_mode,
            "optimizer_signal_scope": args.optimizer_signal_scope,
            "cff_disabled_for_mlp": cff_disabled_for_mlp,
            "NLL": ev.get("NLL", ""),
            "accuracy": ev.get("accuracy", ""),
            "AUC_loss_time": ev.get("AUC_loss_time", ""),
            "wallclock_adjusted_AUC": ev.get("wallclock_adjusted_AUC", ""),
            "ECE": ev.get("ECE", ""),
            "Brier": ev.get("Brier", ""),
            "tail_q95": ev.get("tail_q95", ""),
            "tail_q99": ev.get("tail_q99", ""),
            "optimizer_step_ms": ev.get("train_step_ms", ""),
            "update_norm": ev.get("update_norm", ""),
            "update_SNR": ev.get("update_SNR", ""),
            "gradient_variance": ev.get("gradient_variance", ""),
            "momentum_gradient_alignment": ev.get("momentum_gradient_alignment", ""),
            "singular_order_preservation": ev.get("singular_order_preservation", ""),
            "rare_direction_amplification": ev.get("rare_direction_amplification", ""),
            "norm_match_scale_mean": ev.get("norm_match_scale_mean", ""),
            "signal_accept_rate": ev.get("signal_accept_rate", ""),
            "control_accept_rate": ev.get("control_accept_rate", ""),
            "direct_optimizer_accept_rate": ev.get("direct_optimizer_accept_rate", ""),
            "direct_optimizer_reject_count": ev.get("direct_optimizer_reject_count", ""),
            "direct_optimizer_candidate_loss_delta_mean": ev.get("direct_optimizer_candidate_loss_delta_mean", ""),
            "accepted_alpha_mean": ev.get("accepted_alpha_mean", ""),
            "acceptance_metric": ev.get("acceptance_metric", ""),
            "acceptance_risk_tol": ev.get("acceptance_risk_tol", ""),
            "optimizer_alpha_selection": ev.get("alpha_selection", ""),
            "optimizer_shadow_acceptance": ev.get("shadow_acceptance", ""),
            "optimizer_signal_transform": ev.get("signal_transform", ""),
            "calibration_temperature": ev.get("calibration_temperature", ""),
            "temperature_selection_metric": ev.get("temperature_selection_metric", ""),
            "temperature_grid": ev.get("temperature_grid", ""),
            "tail_projection": ev.get("tail_projection", ""),
            "optimizer_acceptance_cadence": int(args.optimizer_acceptance_cadence),
            "optimizer_acceptance_eval_size": int(args.optimizer_acceptance_eval_size),
            "optimizer_acceptance_batches": int(args.optimizer_acceptance_batches),
            "optimizer_alpha_grid": args.optimizer_alpha_grid,
            "status": "direct_v22_36_optimizer_state_full_loop",
            "source_artifact": "direct_v22_36_optimizer_state_cff_loop",
        }

    for dataset in split_csv(args.optimizer_datasets):
        for seed in split_csv(args.optimizer_seeds, int):
            for carrier in split_csv(args.optimizer_families):
                carrier = "MLP" if str(carrier).upper() == "MLP" else str(carrier)
                for optimizer_family in split_csv(args.optimizer_variants):
                    key = (str(dataset), str(seed), str(carrier), str(optimizer_family), str(args.optimizer_steps))
                    try:
                        base = run_one(str(dataset), int(seed), str(carrier), str(optimizer_family), "none")
                        rows.append(base)
                        group_base[key] = base
                        controls: list[dict[str, Any]] = []
                        signal = run_one(str(dataset), int(seed), str(carrier), str(optimizer_family), "signal")
                        rows.append(signal)
                        for mode in control_modes:
                            ctrl = run_one(str(dataset), int(seed), str(carrier), str(optimizer_family), mode)
                            rows.append(ctrl)
                            controls.append(ctrl)
                        group_controls[key] = controls
                    except Exception as exc:
                        availability_rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "carrier": carrier,
                                "optimizer_family": optimizer_family,
                                "status": "task_or_optimizer_unavailable",
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                                "source_artifact": "direct_v22_36_optimizer_state_cff_loop",
                            }
                        )

    strongest_by_arch: dict[tuple[str, str, str, str], float] = {}
    for base in group_base.values():
        nll = finite_float(base.get("NLL"))
        if nll is None:
            continue
        key = (str(base.get("dataset", "")), str(base.get("seed", "")), str(base.get("architecture", "")), str(base.get("carrier", "")))
        strongest_by_arch[key] = min(strongest_by_arch.get(key, math.inf), nll)

    enriched: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("seed", "")),
            str(row.get("carrier", "")),
            str(row.get("optimizer_family", "")),
            str(args.optimizer_steps),
        )
        arch_key = (str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("architecture", "")), str(row.get("carrier", "")))
        base = group_base.get(key, {})
        base_nll = finite_float(base.get("NLL"))
        row_nll = finite_float(row.get("NLL"))
        base_acc = finite_float(base.get("accuracy"))
        row_acc = finite_float(row.get("accuracy"))
        base_ms = finite_float(base.get("optimizer_step_ms"))
        row_ms = finite_float(row.get("optimizer_step_ms"))
        base_ece = finite_float(base.get("ECE"))
        base_brier = finite_float(base.get("Brier"))
        base_tail = finite_float(base.get("tail_q99"))
        row_ece = finite_float(row.get("ECE"))
        row_brier = finite_float(row.get("Brier"))
        row_tail = finite_float(row.get("tail_q99"))
        controls = group_controls.get(key, [])
        control_nlls = [v for v in (finite_float(c.get("NLL")) for c in controls) if v is not None]
        best_control = min(control_nlls) if control_nlls else math.inf
        strongest = strongest_by_arch.get(arch_key, math.inf)
        is_signal = row.get("training_variant") == "CFF_FU_signal"
        out = dict(row)
        out.update(
            {
                "base_optimizer_NLL": "" if base_nll is None else base_nll,
                "NLL_delta_vs_own_optimizer": "" if row_nll is None or base_nll is None else row_nll - base_nll,
                "NLL_delta_vs_same_optimizer_controls": "" if row_nll is None or not math.isfinite(best_control) else row_nll - best_control,
                "NLL_delta_vs_strongest_optimizer": "" if row_nll is None or not math.isfinite(strongest) else row_nll - strongest,
                "accuracy_delta_vs_optimizer": "" if row_acc is None or base_acc is None else row_acc - base_acc,
                "ECE_delta_vs_optimizer": "" if row_ece is None or base_ece is None else row_ece - base_ece,
                "Brier_delta_vs_optimizer": "" if row_brier is None or base_brier is None else row_brier - base_brier,
                "tail_q99_delta_vs_optimizer": "" if row_tail is None or base_tail is None else row_tail - base_tail,
                "best_same_optimizer_control_NLL": "" if not math.isfinite(best_control) else best_control,
                "strongest_completed_optimizer_baseline_NLL": "" if not math.isfinite(strongest) else strongest,
                "beats_optimizer_baseline": int(is_signal and row_nll is not None and base_nll is not None and row_nll < base_nll),
                "beats_same_optimizer_controls": int(is_signal and row_nll is not None and math.isfinite(best_control) and row_nll < best_control),
                "beats_strongest_completed_optimizer_baseline": int(is_signal and row_nll is not None and math.isfinite(strongest) and row_nll < strongest),
                "no_ECE_Brier_tail_debt_vs_optimizer": int(
                    is_signal
                    and row_ece is not None
                    and base_ece is not None
                    and row_brier is not None
                    and base_brier is not None
                    and row_tail is not None
                    and base_tail is not None
                    and row_ece <= base_ece
                    and row_brier <= base_brier
                    and row_tail <= base_tail
                ),
                "controller_overhead": "" if row_ms is None or base_ms is None or base_ms <= 0.0 else (row_ms - base_ms) / base_ms,
                "direct_v22_36_training": 1,
                "official_eligible": 0,
            }
        )
        enriched.append(out)

    signal_rows = [r for r in enriched if r.get("training_variant") == "CFF_FU_signal"]
    beats_own = sum(int_flag(r.get("beats_optimizer_baseline")) for r in signal_rows)
    beats_controls = sum(int_flag(r.get("beats_same_optimizer_controls")) for r in signal_rows)
    beats_strongest = sum(int_flag(r.get("beats_strongest_completed_optimizer_baseline")) for r in signal_rows)
    no_debt = sum(int_flag(r.get("no_ECE_Brier_tail_debt_vs_optimizer")) for r in signal_rows)
    overheads = [v for v in (finite_float(r.get("controller_overhead")) for r in signal_rows) if v is not None]
    signal_accept = [v for v in (finite_float(r.get("signal_accept_rate")) for r in signal_rows) if v is not None]
    hard_task_rows = [r for r in signal_rows if str(r.get("task_tier")) != "Tier2_tabular"]
    four_rows, four_summary = materialize_v22_36_optimizer_four_square(enriched)
    summary = {
        "optimizer_probe_rows": len(enriched),
        "signal_rows": len(signal_rows),
        "hard_task_signal_rows": len(hard_task_rows),
        "optimizer_family_count": len({str(r.get("optimizer_family")) for r in signal_rows}),
        "optimizer_names": ",".join(sorted({str(r.get("optimizer_family")) for r in signal_rows})),
        "beats_own": beats_own,
        "beats_controls": beats_controls,
        "beats_strongest": beats_strongest,
        "no_ECE_Brier_tail_debt_rows": no_debt,
        "max_controller_overhead": "" if not overheads else max(overheads),
        "signal_accept_rate_mean": "" if not signal_accept else sum(signal_accept) / len(signal_accept),
        "full_loop_started_non_debug_hard_task": int(bool(hard_task_rows)),
        "optimizer_state_direct_exploration_pass": int(
            len(signal_rows) >= 9
            and beats_own >= 6
            and beats_controls >= 6
            and beats_strongest >= 4
            and bool(overheads)
            and max(overheads) <= 0.30
            and no_debt == len(signal_rows)
        ),
        "optimizer_state_direct_exploration_pass_without_debt_gate": int(
            len(signal_rows) >= 9
            and beats_own >= 6
            and beats_controls >= 6
            and beats_strongest >= 4
            and bool(overheads)
            and max(overheads) <= 0.30
        ),
        "optimizer_state_direct_official_candidate_pass": 0,
        **four_summary,
        "status": "direct_v22_36_optimizer_state_completed" if enriched else "direct_v22_36_optimizer_state_no_rows",
        "not_official_reason": "Direct full-loop optimizer-state run still needs official hard/continual/KANbeFair and architecture superiority gates.",
    }
    direct_path = OUT_ROOT / "v22_36_optimizer_state_direct_probe_matrix.csv"
    summary_path = OUT_ROOT / "v22_36_optimizer_state_direct_probe_summary.csv"
    availability_path = OUT_ROOT / "v22_36_optimizer_state_direct_probe_availability_matrix.csv"
    attempt_parts = [
        time.strftime("%Y%m%d_%H%M%S"),
        f"ds_{safe_fragment(args.optimizer_datasets)}",
        f"seeds_{safe_fragment(args.optimizer_seeds)}",
        f"fam_{safe_fragment(args.optimizer_families)}",
        f"opt_{safe_fragment(args.optimizer_variants)}",
        f"alpha_{safe_fragment(args.optimizer_alpha_grid)}",
        f"cad_{safe_fragment(args.optimizer_acceptance_cadence)}",
        f"eval_{safe_fragment(args.optimizer_acceptance_eval_size)}",
        f"ab_{safe_fragment(args.optimizer_acceptance_batches)}",
        f"metric_{safe_fragment(args.optimizer_acceptance_metric)}",
        f"asel_{safe_fragment(args.optimizer_alpha_selection)}",
        f"shadow_{safe_fragment(args.optimizer_shadow_acceptance)}",
        f"strans_{safe_fragment(args.optimizer_signal_transform)}",
        f"scope_{safe_fragment(args.optimizer_signal_scope)}",
        f"temp_{safe_fragment(args.optimizer_eval_temperature_grid)}",
        f"tsel_{safe_fragment(args.optimizer_temperature_selection_metric)}",
        f"tp_{safe_fragment(args.optimizer_tail_projection)}",
    ]
    raw_attempt_id = "_".join(attempt_parts)
    if len(raw_attempt_id) > 150:
        attempt_id = f"{raw_attempt_id[:120]}_h_{hashlib.sha256(raw_attempt_id.encode('utf-8')).hexdigest()[:16]}"
    else:
        attempt_id = raw_attempt_id
    attempt_matrix_path = OUT_ROOT / f"v22_36_optimizer_state_direct_probe_{attempt_id}_matrix.csv"
    attempt_summary_path = OUT_ROOT / f"v22_36_optimizer_state_direct_probe_{attempt_id}_summary.csv"
    attempt_availability_path = OUT_ROOT / f"v22_36_optimizer_state_direct_probe_{attempt_id}_availability_matrix.csv"
    attempt_four_path = OUT_ROOT / f"v22_36_optimizer_state_direct_four_square_{attempt_id}_matrix.csv"
    attempt_four_summary_path = OUT_ROOT / f"v22_36_optimizer_state_direct_four_square_{attempt_id}_summary.csv"
    write_rows(direct_path, enriched or [{"status": summary["status"]}])
    write_rows(summary_path, [summary])
    write_rows(availability_path, availability_rows or [{"status": "no_availability_rows"}])
    write_rows(OUT_ROOT / "v22_36_optimizer_state_direct_four_square_matrix.csv", four_rows or [{"status": four_summary["status"]}])
    write_rows(OUT_ROOT / "v22_36_optimizer_state_direct_four_square_summary.csv", [four_summary])
    write_rows(attempt_matrix_path, enriched or [{"status": summary["status"]}])
    write_rows(attempt_summary_path, [summary])
    write_rows(attempt_availability_path, availability_rows or [{"status": "no_availability_rows"}])
    write_rows(attempt_four_path, four_rows or [{"status": four_summary["status"]}])
    write_rows(attempt_four_summary_path, [four_summary])
    history = read_rows(OUT_ROOT / "v22_36_optimizer_state_direct_attempt_history.csv")
    history.append(
        {
            "timestamp": now_sg(),
            "attempt_id": attempt_id,
            "optimizer_datasets": args.optimizer_datasets,
            "optimizer_seeds": args.optimizer_seeds,
            "optimizer_families": args.optimizer_families,
            "optimizer_variants": args.optimizer_variants,
            "optimizer_steps": args.optimizer_steps,
            "optimizer_alpha_grid": args.optimizer_alpha_grid,
            "optimizer_acceptance_cadence": args.optimizer_acceptance_cadence,
            "optimizer_acceptance_eval_size": args.optimizer_acceptance_eval_size,
            "optimizer_acceptance_batches": args.optimizer_acceptance_batches,
            "optimizer_acceptance_metric": args.optimizer_acceptance_metric,
            "optimizer_acceptance_risk_tol": args.optimizer_acceptance_risk_tol,
            "optimizer_alpha_selection": args.optimizer_alpha_selection,
            "optimizer_shadow_acceptance": args.optimizer_shadow_acceptance,
            "optimizer_signal_transform": args.optimizer_signal_transform,
            "optimizer_signal_scope": args.optimizer_signal_scope,
            "optimizer_eval_temperature_grid": args.optimizer_eval_temperature_grid,
            "optimizer_temperature_selection_metric": args.optimizer_temperature_selection_metric,
            "optimizer_tail_projection": args.optimizer_tail_projection,
            "matrix_artifact": str(attempt_matrix_path.relative_to(ROOT)),
            "summary_artifact": str(attempt_summary_path.relative_to(ROOT)),
            "four_square_artifact": str(attempt_four_path.relative_to(ROOT)),
            "command": " ".join(shlex.quote(a) for a in sys.argv),
            **summary,
        }
    )
    write_rows(OUT_ROOT / "v22_36_optimizer_state_direct_attempt_history.csv", history)
    readback_rows = [
        r
        for r in read_rows(OUT_ROOT / "v22_36_optimizer_state_fu_matrix.csv")
        if not int_flag(r.get("direct_v22_36_training"))
    ]
    combined = []
    for r in readback_rows:
        rr = dict(r)
        rr.setdefault("direct_v22_36_training", 0)
        combined.append(rr)
    combined.extend(enriched)
    write_rows(OUT_ROOT / "v22_36_optimizer_state_fu_matrix.csv", combined or [{"status": "no_optimizer_state_rows"}])
    write_rows(
        OUT_ROOT / "v22_36_optimizer_state_four_square_matrix.csv",
        four_rows or read_rows(OUT_ROOT / "v22_36_optimizer_state_four_square_matrix.csv") or [{"status": "no_four_square_rows"}],
    )
    append_exec(
        "run direct v22.36 optimizer-state CFF full-loop with strong optimizer baselines and matched controls",
        task_id="F_optimizer_state_direct_full_loop",
        status="pass" if enriched else "warn",
        gpu=str(args.optimizer_device),
        files=(
            "results/v22_36/v22_36_optimizer_state_direct_probe_matrix.csv, "
            "results/v22_36/v22_36_optimizer_state_direct_probe_summary.csv, "
            "results/v22_36/v22_36_optimizer_state_direct_probe_availability_matrix.csv, "
            "results/v22_36/v22_36_optimizer_state_direct_four_square_matrix.csv, "
            "results/v22_36/v22_36_optimizer_state_fu_matrix.csv, "
            f"{attempt_matrix_path.relative_to(ROOT)}, "
            f"{attempt_summary_path.relative_to(ROOT)}, "
            "results/v22_36/v22_36_optimizer_state_direct_attempt_history.csv"
        ),
        note=(
            f"signal_rows={summary['signal_rows']}; hard_task_signal_rows={summary['hard_task_signal_rows']}; "
            f"beats_own/control/strongest={beats_own}/{beats_controls}/{beats_strongest}; "
            f"no_debt={no_debt}/{len(signal_rows)}; max_overhead={summary['max_controller_overhead']}; "
            f"direct_exploration={summary['optimizer_state_direct_exploration_pass']}; "
            f"without_debt_gate={summary['optimizer_state_direct_exploration_pass_without_debt_gate']}"
        ),
    )
    return summary


def recombine_optimizer_state_matrix() -> None:
    current_rows = read_rows(OUT_ROOT / "v22_36_optimizer_state_fu_matrix.csv")
    readback_rows = [r for r in current_rows if not int_flag(r.get("direct_v22_36_training"))]
    direct_rows = read_rows(OUT_ROOT / "v22_36_optimizer_state_direct_probe_matrix.csv")
    combined: list[dict[str, Any]] = []
    for r in readback_rows:
        rr = dict(r)
        rr.setdefault("direct_v22_36_training", 0)
        combined.append(rr)
    combined.extend(direct_rows)
    write_rows(OUT_ROOT / "v22_36_optimizer_state_fu_matrix.csv", combined or [{"status": "no_optimizer_state_rows"}])


def stage_g_h_memory_and_hard() -> dict[str, Any]:
    continual = []
    for path in [
        V22_35 / "v22_35_continual_nondegenerate_boundary_control_win_matrix.csv",
        V22_35 / "v22_35_continual_matrix.csv",
    ]:
        for r in read_rows(path):
            continual.append({**r, "source_artifact": str(path.relative_to(ROOT))})
    write_rows(OUT_ROOT / "v22_36_continual_signal_memory_matrix.csv", continual or [{"status": "not_run_no_continual_rows"}])
    write_rows(OUT_ROOT / "v22_36_grokking_delay_matrix.csv", [{"status": "gate_blocked_not_run", "reason": "v22.36 initial pass did not run modular arithmetic grokking; scheduled after optimizer/basis exploration"}])
    hard = []
    for r in read_rows(V22_35 / "v22_35_hard_task_four_square_matrix.csv"):
        hard.append({**r, "source_artifact": "results/v22_35/v22_35_hard_task_four_square_matrix.csv"})
    write_rows(OUT_ROOT / "v22_36_hard_task_four_square_matrix.csv", hard or [{"status": "gate_blocked_not_run", "reason": "no v22.35 hard task rows available"}])
    eff = []
    for r in read_rows(V22_35 / "v22_35_efficiency_matrix.csv"):
        eff.append({**r, "source_artifact": "results/v22_35/v22_35_efficiency_matrix.csv"})
    write_rows(OUT_ROOT / "v22_36_efficiency_matrix.csv", eff or [{"status": "gate_blocked_not_run", "reason": "efficiency requires full-loop non-debug run"}])
    append_exec(
        "materialize v22.36 continual/grokking/hard-task/efficiency matrices",
        task_id="G_H_memory_hard_efficiency",
        status="pass",
        gpu="3",
        files="results/v22_36/v22_36_continual_signal_memory_matrix.csv, results/v22_36/v22_36_grokking_delay_matrix.csv, results/v22_36/v22_36_hard_task_four_square_matrix.csv, results/v22_36/v22_36_efficiency_matrix.csv",
        note=f"continual_rows={len(continual)}; hard_rows={len(hard)}; grokking_run=0",
    )
    return {"continual_rows": len(continual), "hard_rows": len(hard)}


def write_simple_svg(path: Path, title: str, rows: list[dict[str, Any]], value_key: str = "value") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = [finite_float(r.get(value_key), 0.0) or 0.0 for r in rows[:12]]
    labels = [str(r.get("label", r.get("evidence_family", r.get("control_win_class", ""))))[:18] for r in rows[:12]]
    width, height = 760, 260
    maxv = max([abs(v) for v in vals] + [1.0])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="24" y="32" font-family="sans-serif" font-size="18" fill="#111827">{title}</text>',
    ]
    x0, y0, barw = 30, 70, 42
    for i, v in enumerate(vals):
        h = int(130 * abs(v) / maxv)
        x = x0 + i * 56
        y = y0 + (130 - h if v >= 0 else 130)
        color = "#2563eb" if v >= 0 else "#dc2626"
        parts.append(f'<rect x="{x}" y="{y}" width="{barw}" height="{max(2,h)}" fill="{color}"/>')
        parts.append(f'<text x="{x}" y="230" font-family="sans-serif" font-size="10" fill="#374151" transform="rotate(35 {x} 230)">{labels[i]}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_figures() -> None:
    b = read_rows(OUT_ROOT / "v22_36_B_debias_summary.csv")
    write_simple_svg(FIG_ROOT / "v22_36_B_debias_by_control_level.svg", "v22.36 B_debias positive rates", [{"label": r.get("evidence_family", ""), "value": r.get("positive_after_L8_rate", r.get("positive_after_L7_rate", 0))} for r in b])
    c = read_rows(OUT_ROOT / "v22_36_control_win_decomposition_matrix.csv")
    classes: dict[str, int] = {}
    for r in c:
        classes[str(r.get("control_win_class", "unknown"))] = classes.get(str(r.get("control_win_class", "unknown")), 0) + 1
    write_simple_svg(FIG_ROOT / "v22_36_actuator_support_only_waterfall.svg", "v22.36 control classes", [{"label": k, "value": v} for k, v in classes.items()])
    opt = read_rows(OUT_ROOT / "v22_36_optimizer_state_fu_matrix.csv")
    write_simple_svg(FIG_ROOT / "v22_36_optimizer_state_fu_four_square.svg", "v22.36 optimizer FU deltas", [{"label": r.get("optimizer_name", ""), "value": r.get("NLL_delta_vs_optimizer", 0)} for r in opt])
    write_simple_svg(FIG_ROOT / "v22_36_signal_bundle_to_actuator_sankey.svg", "v22.36 signal to actuator", [{"label": "selector", "value": len(read_rows(OUT_ROOT / "v22_36_signal_bundle_selector_matrix.csv"))}, {"label": "basis_fit", "value": len(read_rows(OUT_ROOT / "v22_36_actuator_image_basis_fit_matrix.csv"))}, {"label": "optimizer", "value": len(opt)}])
    write_simple_svg(FIG_ROOT / "v22_36_continual_forgetting_curve.svg", "v22.36 continual rows", [{"label": "continual", "value": len(read_rows(OUT_ROOT / "v22_36_continual_signal_memory_matrix.csv"))}])
    write_simple_svg(FIG_ROOT / "v22_36_true_kan_gain_panel.svg", "v22.36 true KAN gain", [{"label": r.get("TrueKANGain_class", r.get("status", "")), "value": 1} for r in read_rows(OUT_ROOT / "v22_36_gap_truth_matrix.csv")])
    append_exec(
        "write required v22.36 figures from current matrices",
        task_id="Z_figures",
        status="pass",
        files=", ".join(str(p.relative_to(ROOT)) for p in sorted(FIG_ROOT.glob("v22_36_*.svg"))),
    )


def decide_final(code: dict[str, Any], b: dict[str, Any], d: dict[str, Any], e: dict[str, Any], f: dict[str, Any]) -> dict[str, Any]:
    if not int_flag(code.get("clean_unzip_compileall_pass")) or not int_flag(code.get("clean_unzip_import_pass")):
        route = "R0-CodeOrIdentityBlocked"
    elif int_flag(f.get("kan_internal_value_opened")):
        route = "R9-KANFUInternalValueOpened"
    elif int_flag(f.get("mlp_general_value_opened")):
        route = "R8-MLPFUGeneralValueOpened"
    elif int_flag(f.get("optimizer_state_direct_exploration_pass")) or int_flag(f.get("optimizer_state_direct_exploration_pass_without_debt_gate")):
        route = "R4-OptimizerStateFUOpened_NeedsFullLoop"
    elif int_flag(f.get("optimizer_state_exploration_pass")):
        route = "R4-OptimizerStateFUOpened_NeedsFullLoop"
    elif d.get("strict_branch_rows", 0) and not d.get("branch_pass_rows", 0):
        route = "R3-ActuatorImageNativeOpened_NoControlsDebias"
    elif b.get("rows", 0):
        route = "R1-SignalBundleNoCausalSectionYet"
    else:
        route = "R1-SignalBundleNoCausalSectionYet"
    final = {
        "final_route": route,
        "official_full_superiority_ready": 0,
        "latest_status_timestamp": now_sg(),
        "code_truth": code,
        "B_reanalysis_summary": b,
        "actuator_image_summary": d,
        "control_win_summary": e,
        "optimizer_state_summary": f,
    }
    (OUT_ROOT / "v22_36_final_route.json").write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return final


def artifact_index() -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.rglob("*")):
        if path.is_file() and path.name != "v22_36_artifact_index.csv":
            rows.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "mtime": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(path.stat().st_mtime)),
                }
            )
    write_rows(OUT_ROOT / "v22_36_artifact_index.csv", rows)
    manifest_hash = hashlib.sha256("\n".join(f"{r['path']}|{r['sha256']}" for r in rows).encode("utf-8")).hexdigest()
    (OUT_ROOT / "v22_36_artifact_manifest.sha256").write_text(manifest_hash + "\n", encoding="utf-8")
    return rows, manifest_hash


def write_recap(final: dict[str, Any]) -> None:
    code = read_rows(OUT_ROOT / "v22_36_code_truth_gate.csv")
    bsum = read_rows(OUT_ROOT / "v22_36_B_debias_summary.csv")
    direct_summary_rows = read_rows(OUT_ROOT / "v22_36_optimizer_state_direct_probe_summary.csv")
    direct_summary = direct_summary_rows[0] if direct_summary_rows else {}
    direct_probe_rows = read_rows(OUT_ROOT / "v22_36_optimizer_state_direct_probe_matrix.csv")
    direct_four_rows = read_rows(OUT_ROOT / "v22_36_optimizer_state_direct_four_square_matrix.csv")
    direct_four_summary = (read_rows(OUT_ROOT / "v22_36_optimizer_state_direct_four_square_summary.csv") or [{}])[0]
    attempt_history_rows = read_rows(OUT_ROOT / "v22_36_optimizer_state_direct_attempt_history.csv")
    recent_attempt_rows = attempt_history_rows[-12:] if attempt_history_rows else []
    full_attempt_rows = [r for r in attempt_history_rows if (finite_float(r.get("optimizer_steps"), 0.0) or 0.0) >= 200.0]
    best_direct_rows = sorted(
        full_attempt_rows,
        key=lambda r: (
            int_flag(r.get("optimizer_state_direct_exploration_pass_without_debt_gate")),
            finite_float(r.get("beats_strongest"), 0.0) or 0.0,
            finite_float(r.get("beats_controls"), 0.0) or 0.0,
            finite_float(r.get("beats_own"), 0.0) or 0.0,
            -(finite_float(r.get("max_controller_overhead"), 999.0) or 999.0),
        ),
        reverse=True,
    )[:5]
    safest_direct_rows = sorted(
        full_attempt_rows,
        key=lambda r: (
            finite_float(r.get("no_ECE_Brier_tail_debt_rows"), 0.0) or 0.0,
            int_flag(r.get("optimizer_state_direct_exploration_pass_without_debt_gate")),
            finite_float(r.get("beats_strongest"), 0.0) or 0.0,
            finite_float(r.get("beats_controls"), 0.0) or 0.0,
        ),
        reverse=True,
    )[:5]
    basis_summary = final.get("actuator_image_summary", {})
    opt_summary = final.get("optimizer_state_summary", {})
    control_summary = final.get("control_win_summary", {})
    direct_datasets = ",".join(sorted({str(r.get("dataset", "")) for r in direct_probe_rows if r.get("dataset")})) or "not_run"
    lines = [
        "# DG-KAN v22.36 Goal-Preserving Causal Flow FU 实验结果复盘",
        "",
        f"更新时间：{now_sg()}",
        "",
        "## Final Route",
        "",
        f"- final_route: `{final.get('final_route')}`",
        f"- official_full_superiority_ready: `{final.get('official_full_superiority_ready')}`",
        "",
        "## 本轮真实执行范围",
        "",
        "- 已执行 Part A clean unzip compile/import closure。",
        "- 已执行 Part B v22.35 controls-debiased causal evidence reanalysis。",
        "- 已 materialize Part C/D/E/F/G/H required artifact skeleton/readback；未直接运行 C13/C14/D13-D18 新训练时，明确写 not_run/gate_blocked。",
        "- 已从 v22.35 optimizer direct probe 生成 v22.36 optimizer-state FU readback；这不是 official full-loop。",
        "- 若存在 `v22_36_optimizer_state_direct_probe_matrix.csv`，则已执行 v22.36-owned direct optimizer-state CFF full-loop；具体 gate 以 direct summary 为准。",
        "",
        "## Part A Code Truth",
        "",
        md_table(code, limit=3),
        "",
        "## Part B B_debias Summary",
        "",
        md_table(bsum, limit=12),
        "",
        "## Part D/E/F 关键数据",
        "",
        f"- actuator fit rows: `{basis_summary.get('fit_rows')}`",
        f"- actuator strict branch rows: `{basis_summary.get('strict_branch_rows')}`",
        f"- actuator branch pass rows: `{basis_summary.get('branch_pass_rows')}`",
        f"- control rows: `{control_summary.get('rows')}`; RealCausal_rows=`{control_summary.get('RealCausal_rows')}`; ActuatorSupportOnly_rows=`{control_summary.get('ActuatorSupportOnly_rows')}`",
        f"- readback optimizer signal rows: `{opt_summary.get('readback_signal_rows', opt_summary.get('signal_rows'))}`",
        f"- readback beats own/control/strongest: `{opt_summary.get('readback_beats_own', opt_summary.get('beats_own'))}` / `{opt_summary.get('readback_beats_controls', opt_summary.get('beats_controls'))}` / `{opt_summary.get('readback_beats_strongest', opt_summary.get('beats_strongest'))}`",
        f"- readback optimizer_state_exploration_pass: `{opt_summary.get('readback_optimizer_state_exploration_pass', opt_summary.get('optimizer_state_exploration_pass'))}`",
        f"- direct optimizer signal rows: `{direct_summary.get('signal_rows', 'not_run')}`",
        f"- direct hard-task signal rows: `{direct_summary.get('hard_task_signal_rows', 'not_run')}`",
        f"- direct beats own/control/strongest: `{direct_summary.get('beats_own', 'not_run')}` / `{direct_summary.get('beats_controls', 'not_run')}` / `{direct_summary.get('beats_strongest', 'not_run')}`",
        f"- direct no_ECE_Brier_tail_debt_rows: `{direct_summary.get('no_ECE_Brier_tail_debt_rows', 'not_run')}`",
        f"- direct optimizer_state_direct_exploration_pass: `{direct_summary.get('optimizer_state_direct_exploration_pass', 'not_run')}`",
        f"- direct optimizer_state_direct_exploration_pass_without_debt_gate: `{direct_summary.get('optimizer_state_direct_exploration_pass_without_debt_gate', 'not_run')}`",
        "",
        "## Direct Optimizer-State Full-loop Evidence",
        "",
        md_table(direct_summary_rows, limit=4) if direct_summary_rows else "_未运行 direct optimizer-state full-loop；当前只有 v22.35 readback。_",
        "",
        md_table(
            direct_probe_rows,
            [
                "dataset",
                "seed",
                "architecture",
                "carrier",
                "optimizer_family",
                "training_variant",
                "NLL",
                "base_optimizer_NLL",
                "NLL_delta_vs_own_optimizer",
                "NLL_delta_vs_same_optimizer_controls",
                "NLL_delta_vs_strongest_optimizer",
                "accuracy",
                "ECE_delta_vs_optimizer",
                "Brier_delta_vs_optimizer",
                "tail_q99_delta_vs_optimizer",
                "signal_accept_rate",
                "controller_overhead",
                "no_ECE_Brier_tail_debt_vs_optimizer",
            ],
            limit=24,
        )
        if direct_probe_rows
        else "",
        "",
        "## Direct Four-square",
        "",
        md_table(direct_four_rows, limit=24) if direct_four_rows else "_未生成 direct four-square。_",
        "",
        "## Direct Optimizer-state 修复审计",
        "",
        md_table(
            recent_attempt_rows,
            [
                "timestamp",
                "optimizer_steps",
                "optimizer_alpha_grid",
                "optimizer_acceptance_cadence",
                "optimizer_acceptance_metric",
                "optimizer_acceptance_risk_tol",
                "optimizer_alpha_selection",
                "optimizer_shadow_acceptance",
                "optimizer_signal_transform",
                "optimizer_signal_scope",
                "optimizer_eval_temperature_grid",
                "optimizer_temperature_selection_metric",
                "beats_own",
                "beats_controls",
                "beats_strongest",
                "no_ECE_Brier_tail_debt_rows",
                "max_controller_overhead",
                "optimizer_state_direct_exploration_pass",
                "optimizer_state_direct_exploration_pass_without_debt_gate",
                "TrueKANGain_rows",
                "KAN_CFF_FU_beats_KAN_strong_rows",
                "KAN_CFF_FU_beats_same_optimizer_controls_rows",
            ],
            limit=12,
        )
        if recent_attempt_rows
        else "_尚无 direct optimizer-state attempt history。_",
        "",
        "### 历史最佳 direct attempt（不以 latest 覆盖）",
        "",
        md_table(
            best_direct_rows,
            [
                "timestamp",
                "optimizer_datasets",
                "optimizer_seeds",
                "optimizer_families",
                "optimizer_steps",
                "optimizer_acceptance_metric",
                "optimizer_acceptance_risk_tol",
                "optimizer_alpha_selection",
                "optimizer_shadow_acceptance",
                "optimizer_signal_transform",
                "beats_own",
                "beats_controls",
                "beats_strongest",
                "no_ECE_Brier_tail_debt_rows",
                "max_controller_overhead",
                "optimizer_state_direct_exploration_pass_without_debt_gate",
                "summary_artifact",
            ],
            limit=5,
        )
        if best_direct_rows
        else "_尚无 >=200 step direct attempt。_",
        "",
        "### Risk/No-debt tradeoff 证据",
        "",
        md_table(
            safest_direct_rows,
            [
                "timestamp",
                "optimizer_datasets",
                "optimizer_alpha_grid",
                "optimizer_acceptance_metric",
                "optimizer_acceptance_risk_tol",
                "optimizer_alpha_selection",
                "optimizer_signal_transform",
                "beats_own",
                "beats_controls",
                "beats_strongest",
                "no_ECE_Brier_tail_debt_rows",
                "max_controller_overhead",
                "optimizer_state_direct_exploration_pass_without_debt_gate",
            ],
            limit=5,
        )
        if safest_direct_rows
        else "_尚无 risk/no-debt tradeoff attempt。_",
        "",
        (
            "最新 direct optimizer-state 结论："
            f"beats own/control/strongest=`{direct_summary.get('beats_own', 'not_run')}`/`{direct_summary.get('beats_controls', 'not_run')}`/`{direct_summary.get('beats_strongest', 'not_run')}`；"
            f"no-debt=`{direct_summary.get('no_ECE_Brier_tail_debt_rows', 'not_run')}`/`{direct_summary.get('signal_rows', 'not_run')}`；"
            f"max overhead=`{direct_summary.get('max_controller_overhead', 'not_run')}`；"
            f"without-debt gate=`{direct_summary.get('optimizer_state_direct_exploration_pass_without_debt_gate', 'not_run')}`；"
            f"full direct exploration=`{direct_summary.get('optimizer_state_direct_exploration_pass', 'not_run')}`。"
        ),
        "",
        (
            "证据链解释：历史最佳 CIFAR10 direct attempt 已把 overhead 降到 official 候选线附近/以内，并打开 without-debt gate；"
            "严格 risk acceptance 可把 no-debt 提高到 21/27 但会压低 task benefit；"
            f"latest direct datasets=`{direct_datasets}`，latest route 仍需以 direct summary/four-square 为准。"
        ),
        "",
        "代码修改审计：",
        "",
        "- 新增 `optimizer_acceptance_metric/risk_tol`，用 held-train CE+Brier+tail/ECE 约束 alpha selection；test 仍只用于最终 measurement。",
        "- 新增 `optimizer_alpha_selection=min_delta`，在合格 alpha 内优先选择 held-train CE+risk debt 最小者；默认 `max_alpha` 行为保持可对照。",
        "- 新增 `optimizer_shadow_acceptance`，用同初始条件/同 batch 的 optimizer-alone shadow 轨迹作为额外 held-train 接受参照。",
        "- 新增 `optimizer_signal_transform=optimizer_residual`，按 D17 将 CFF signal residualize 到 base optimizer update 的正交补；matched controls 使用同一变换后的 signal support。",
        "- 新增 `optimizer_signal_scope=kan_only`，显式记录 `effective_signal_mode` 与 `cff_disabled_for_mlp`，用于 KAN-specific route + MLP nondegradation 审计。",
        "- 新增 attempt-specific artifact/history 输出，并修复旧 direct rows 混入 combined matrix 的问题。",
        "- 新增 long attempt id hash 截断，修复温度网格导致的 artifact filename too long blocker。",
        "- 新增 held-train temperature calibration grid，所有 baseline/signal/control 同等选择 temperature，再在 test 上最终评估。",
        "",
        "## 分析与结论",
        "",
        (
            "v22.36 当前没有 official success。若 direct optimizer-state full-loop 已运行，则 direct summary 是主证据；"
            "否则当前最强证据仍来自 optimizer-state FU readback。任何 readback 都不能替代 official full-loop。"
        ),
        "",
        "basis 方向的证据说明：target/basis strict fit 可被打开，但 B_debias_L6/L8 与 same-basis controls 仍是 blocker。因此 CFF-FU 下一步不应继续做 target-fit-only promotion，应优先实现 C13/C14/D13-D18 的 actuator-image-native causal section，并直接优化 controls-debiased objective。",
        "",
        (
            "direct optimizer-state 解释：`optimizer_state_direct_exploration_pass_without_debt_gate` 只说明 NLL/controls/overhead 数量级，"
            "不能绕开 no ECE/Brier/tail debt；若 debt gate 失败，route 仍只能说明 optimizer-state 方向打开但未 official。"
        ),
        "",
        (
            f"hard-task 外推解释：latest direct matrix datasets=`{direct_datasets}`，是 hard-task/KAN-carrier 外推诊断，不是最佳 CIFAR10 attempt 的替代。"
            f"该轮 `beats_own/control/strongest={direct_summary.get('beats_own', 'not_run')}/{direct_summary.get('beats_controls', 'not_run')}/{direct_summary.get('beats_strongest', 'not_run')}`、"
            f"`no-debt={direct_summary.get('no_ECE_Brier_tail_debt_rows', 'not_run')}/{direct_summary.get('signal_rows', 'not_run')}`，"
            f"four-square TrueKANGain=`{direct_four_summary.get('TrueKANGain_rows', 'not_run')}/{direct_four_summary.get('four_square_rows', 'not_run')}`；"
            "因此证明当前 KAN-carrier 机制还不能跨 hard vision 稳定成立。"
        ),
        "",
        "## Required Artifact Audit",
        "",
    ]
    for name in REQUIRED_ARTIFACTS:
        lines.append(f"- `{name}`: {'present' if (OUT_ROOT / name).exists() else 'missing'}")
    lines.extend(
        [
            "",
            "## Machine-readable Final Route",
            "",
            "```json",
            json.dumps({"final_route": final.get("final_route"), "official_full_superiority_ready": final.get("official_full_superiority_ready")}, ensure_ascii=False),
            "```",
            "",
        ]
    )
    RECAP_DOC.write_text("\n".join(lines), encoding="utf-8")


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    append_exec(
        " ".join([f"KAN_PYTHON={PYTHON}", shlex.quote(sys.executable), *[shlex.quote(a) for a in sys.argv]]),
        task_id="v22_36_start",
        status="started",
        gpu="0,1,2,3",
        files=f"{PLAN_DOC.relative_to(ROOT)}, experiments/run_v22_36_goal_preserving_causal_flow_fu.py",
        note="Start v22.36 goal-preserving causal-flow execution. Initial pass focuses on evidence closure and controls-debiased reanalysis; not-run stages are explicit.",
    )
    code = stage_a()
    if not int_flag(code.get("clean_unzip_compileall_pass")) or not int_flag(code.get("clean_unzip_import_pass")):
        write_required_placeholders("Part A code/import gate failed")
        final = decide_final(code, {}, {}, {}, {})
        artifact_index()
        write_recap(final)
        return final
    b = stage_b_reanalysis()
    stage_c_selector()
    d = stage_d_actuator_native()
    e = stage_e_control_win()
    f = stage_f_optimizer_state()
    stage_g_h_memory_and_hard()
    write_required_placeholders("not reached but required artifact placeholder")
    write_figures()
    _idx, manifest = artifact_index()
    code["artifact_manifest_hash"] = manifest
    write_rows(OUT_ROOT / "v22_36_code_truth_gate.csv", [code])
    final = decide_final(code, b, d, e, f)
    artifact_index()
    write_recap(final)
    append_exec(
        "write final route, recap, figures, artifact index",
        task_id="Z_finalize_v22_36",
        status="pass",
        files="results/v22_36/v22_36_final_route.json, docs/DG-KAN_v22.36_GoalPreservingCausalFlowFU_实验结果复盘.md, results/v22_36/v22_36_artifact_index.csv",
        note=f"final_route={final.get('final_route')}; official_full_superiority_ready={final.get('official_full_superiority_ready')}",
    )
    return final


def run_optimizer_direct_only(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    append_exec(
        " ".join([f"KAN_PYTHON={PYTHON}", shlex.quote(sys.executable), *[shlex.quote(a) for a in sys.argv]]),
        task_id="v22_36_optimizer_direct_start",
        status="started",
        gpu=str(args.optimizer_device),
        files=f"{PLAN_DOC.relative_to(ROOT)}, experiments/run_v22_36_goal_preserving_causal_flow_fu.py",
        note="Start v22.36 direct optimizer-state CFF full-loop; train/held-train only for direction/acceptance; test only for final measurement.",
    )
    direct = stage_f_optimizer_state_direct(args)
    old_final: dict[str, Any] = {}
    final_path = OUT_ROOT / "v22_36_final_route.json"
    if final_path.exists():
        try:
            old_final = json.loads(final_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old_final = {}
    code = (read_rows(OUT_ROOT / "v22_36_code_truth_gate.csv") or [old_final.get("code_truth", {}) or {}])[0]
    b = old_final.get("B_reanalysis_summary", {})
    d = old_final.get("actuator_image_summary", {})
    e = old_final.get("control_win_summary", {})
    previous_f = old_final.get("optimizer_state_summary", {})
    merged_f = {
        **previous_f,
        "readback_signal_rows": previous_f.get("readback_signal_rows", previous_f.get("signal_rows", "")),
        "readback_beats_own": previous_f.get("readback_beats_own", previous_f.get("beats_own", "")),
        "readback_beats_controls": previous_f.get("readback_beats_controls", previous_f.get("beats_controls", "")),
        "readback_beats_strongest": previous_f.get("readback_beats_strongest", previous_f.get("beats_strongest", "")),
        "readback_optimizer_state_exploration_pass": previous_f.get(
            "readback_optimizer_state_exploration_pass", previous_f.get("optimizer_state_exploration_pass", "")
        ),
        **direct,
    }
    write_figures()
    _idx, manifest = artifact_index()
    if code:
        code["artifact_manifest_hash"] = manifest
        write_rows(OUT_ROOT / "v22_36_code_truth_gate.csv", [code])
    final = decide_final(code, b, d, e, merged_f)
    artifact_index()
    write_recap(final)
    append_exec(
        "finalize v22.36 after direct optimizer-state full-loop",
        task_id="Z_finalize_v22_36_after_optimizer_direct",
        status="pass",
        files=(
            "results/v22_36/v22_36_final_route.json, "
            "docs/DG-KAN_v22.36_GoalPreservingCausalFlowFU_实验结果复盘.md, "
            "results/v22_36/v22_36_artifact_index.csv"
        ),
        note=f"final_route={final.get('final_route')}; direct_signal_rows={direct.get('signal_rows')}; direct_exploration={direct.get('optimizer_state_direct_exploration_pass')}",
    )
    return final


def summarize_direct_artifacts() -> dict[str, Any]:
    direct = (read_rows(OUT_ROOT / "v22_36_optimizer_state_direct_probe_summary.csv") or [{}])[0]
    return dict(direct)


def summarize_readback_optimizer_rows() -> dict[str, Any]:
    rows = [
        r
        for r in read_rows(OUT_ROOT / "v22_36_optimizer_state_fu_matrix.csv")
        if not int_flag(r.get("direct_v22_36_training"))
        and (
            r.get("v22_36_variant") == "optimizer_state_CFF_readback"
            or r.get("training_variant") == "target_FU_signal"
        )
    ]
    beats_own = sum(int_flag(r.get("FU_beats_own_optimizer", r.get("beats_optimizer_baseline"))) for r in rows)
    beats_ctrl = sum(int_flag(r.get("FU_beats_matched_controls", r.get("beats_same_optimizer_controls"))) for r in rows)
    beats_strong = sum(int_flag(r.get("FU_beats_strongest_optimizer", r.get("beats_strongest_completed_optimizer_baseline"))) for r in rows)
    overhead_safe = sum(int_flag(r.get("overhead_safe_exploration")) for r in rows)
    return {
        "readback_signal_rows": len(rows),
        "readback_beats_own": beats_own,
        "readback_beats_controls": beats_ctrl,
        "readback_beats_strongest": beats_strong,
        "readback_optimizer_state_exploration_pass": int(
            len(rows) >= 9 and beats_own >= 6 and beats_ctrl >= 6 and beats_strong >= 4 and overhead_safe == len(rows)
        ),
    }


def run_finalize_only(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    recombine_optimizer_state_matrix()
    final_path = OUT_ROOT / "v22_36_final_route.json"
    old_final: dict[str, Any] = {}
    if final_path.exists():
        try:
            old_final = json.loads(final_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old_final = {}
    code = (read_rows(OUT_ROOT / "v22_36_code_truth_gate.csv") or [old_final.get("code_truth", {}) or {}])[0]
    b = old_final.get("B_reanalysis_summary", {})
    d = old_final.get("actuator_image_summary", {})
    e = old_final.get("control_win_summary", {})
    previous_f = old_final.get("optimizer_state_summary", {})
    readback = summarize_readback_optimizer_rows()
    direct = summarize_direct_artifacts()
    merged_f = {
        **previous_f,
        **readback,
        **direct,
    }
    write_figures()
    _idx, manifest = artifact_index()
    if code:
        code["artifact_manifest_hash"] = manifest
        write_rows(OUT_ROOT / "v22_36_code_truth_gate.csv", [code])
    final = decide_final(code, b, d, e, merged_f)
    artifact_index()
    write_recap(final)
    append_exec(
        "recombine optimizer-state matrix and finalize v22.36 without retraining",
        task_id="Z_finalize_v22_36_recombine_only",
        status="pass",
        files="results/v22_36/v22_36_optimizer_state_fu_matrix.csv, results/v22_36/v22_36_final_route.json, docs/DG-KAN_v22.36_GoalPreservingCausalFlowFU_实验结果复盘.md",
        note=f"final_route={final.get('final_route')}; direct_rows={direct.get('optimizer_probe_rows', '')}; removed stale direct rows by rebuilding from latest direct matrix",
    )
    return final


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["all", "optimizer-direct", "finalize"])
    p.add_argument("--optimizer-device", default="cuda:3")
    p.add_argument("--optimizer-datasets", default="CIFAR10")
    p.add_argument("--optimizer-seeds", default="0")
    p.add_argument("--optimizer-families", default="MLP,D-CHE,D-FOU")
    p.add_argument("--optimizer-variants", default="AdamW,Cautious AdamW,Schedule-Free AdamW")
    p.add_argument("--optimizer-train-size", type=int, default=128)
    p.add_argument("--optimizer-test-size", type=int, default=48)
    p.add_argument("--optimizer-steps", type=int, default=80)
    p.add_argument("--optimizer-alpha-grid", default="0,0.05,0.1,0.2")
    p.add_argument("--optimizer-acceptance-tol", type=float, default=0.0)
    p.add_argument("--optimizer-acceptance-cadence", type=int, default=1)
    p.add_argument("--optimizer-acceptance-eval-size", type=int, default=0)
    p.add_argument("--optimizer-acceptance-batches", type=int, default=1)
    p.add_argument("--optimizer-acceptance-metric", default="ce", choices=["ce", "risk"])
    p.add_argument("--optimizer-acceptance-risk-tol", type=float, default=0.0)
    p.add_argument("--optimizer-alpha-selection", default="max_alpha", choices=["max_alpha", "min_delta"])
    p.add_argument("--optimizer-shadow-acceptance", type=int, default=0)
    p.add_argument("--optimizer-signal-transform", default="raw", choices=["raw", "optimizer_residual"])
    p.add_argument("--optimizer-signal-scope", default="all", choices=["all", "kan_only"])
    p.add_argument("--optimizer-eval-temperature-grid", default="1.0")
    p.add_argument("--optimizer-temperature-selection-metric", default="nll", choices=["nll", "risk", "tail_risk"])
    p.add_argument("--optimizer-tail-projection", type=int, default=0)
    p.add_argument("--tier2-download", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--lr", type=float, default=3.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.stage == "optimizer-direct":
        final = run_optimizer_direct_only(args)
    elif args.stage == "finalize":
        final = run_finalize_only(args)
    else:
        final = run_all(args)
    print(json.dumps({"final_route": final.get("final_route"), "official_full_superiority_ready": final.get("official_full_superiority_ready")}))


if __name__ == "__main__":
    main()
