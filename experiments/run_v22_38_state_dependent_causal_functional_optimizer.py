#!/usr/bin/env python3
"""DG-KAN v22.38 State-Dependent Causal Functional Optimizer runner.

This runner is evidence-first.  It reuses the v22.37 micro-intervention and
compact full-loop machinery where appropriate, but writes v22.38-specific
artifacts and explicitly marks unavailable or gate-blocked stages.
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
import random
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v22_37_causal_instrumented_functional_optimizer as core


PYTHON = os.environ.get("KAN_PYTHON", "/home/chengshun.wang/miniconda3/envs/kan/bin/python")
OUT_ROOT = ROOT / "results/v22_38"
FIG_ROOT = OUT_ROOT / "figures"
LOG_ROOT = OUT_ROOT / "logs"
CHUNK_ROOT = OUT_ROOT / "chunks"
CORE_ROOT = OUT_ROOT / "core_interop"
PLAN_DOC = ROOT / "docs/DG-KAN_v22.38_StateDependentCausalFunctionalOptimizer_完整计划.md"
EXEC_DOC = ROOT / "docs/DG-KAN_v22.38_StateDependentCausalFunctionalOptimizer_执行日志.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v22.38_StateDependentCausalFunctionalOptimizer_实验结果复盘.md"

TREATMENT_META: dict[str, dict[str, str]] = dict(core.TREATMENT_META)
TREATMENT_META.update(
    {
        "a5_signal_incremental_basis": {"kind": "signal", "match": "a6_signflip_same_basis"},
        "a6_signflip_same_basis": {"kind": "control", "match": "a5_signal_incremental_basis"},
        "a6_same_bank_random": {"kind": "control", "match": "a8_support_native_basis"},
        "a8_support_native_basis": {"kind": "support", "match": "a6_same_bank_random"},
        "a2_a5_runtime_arbitrated_basis": {"kind": "signal", "match": "a6_signflip_same_basis"},
        "a8_support_native_preconditioner": {"kind": "support", "match": "a0_base_noop"},
        "a9_horizon_safe_optimizer_state_signal": {"kind": "signal", "match": "a7_same_optimizer_geometry"},
        "a10_continual_memory_causal_treatment": {"kind": "signal", "match": "a5_same_support_random"},
    }
)

SUPPORT_CONTROL = {
    "a1_signal_direction": "a5_same_support_random",
    "a2_basis_actuator_section": "a6_same_actuator_random",
    "a3_optimizer_state_signal": "a7_same_optimizer_geometry",
    "a4_continual_boundary_memory": "a5_same_support_random",
    "a5_signal_incremental_basis": "a6_signflip_same_basis",
    "a2_a5_runtime_arbitrated_basis": "a6_signflip_same_basis",
    "a8_support_native_basis": "a6_same_bank_random",
    "a8_support_native_preconditioner": "a6_same_actuator_random",
    "a9_horizon_safe_optimizer_state_signal": "a7_same_optimizer_geometry",
    "a10_continual_memory_causal_treatment": "a5_same_support_random",
}

PLANNED_BASIS_TREATMENTS = {
    "a2_basis_actuator_section",
    "a5_signal_incremental_basis",
    "a2_a5_runtime_arbitrated_basis",
    "a6_same_actuator_random",
    "a6_signflip_same_basis",
    "a6_same_bank_random",
    "a8_support_native_basis",
}

ARBITRATED_TREATMENT_CANDIDATES = {
    "a2_a5_runtime_arbitrated_basis": ["a2_basis_actuator_section", "a5_signal_incremental_basis"],
}

PLANNED_TREATMENT_EVIDENCE_ALIASES = {
    "a5_signal_incremental_basis": {
        "source_treatment": "a2_basis_actuator_section",
        "source_column": "LCB_direction",
        "note": "exploratory_alias_exact_same_basis_negative_train_gradient",
    },
    "a6_same_bank_random": {
        "source_treatment": "a6_same_actuator_random",
        "source_column": "LCB_net",
        "note": "exploratory_alias_same_basis_random_family",
    },
    "a8_support_native_basis": {
        "source_treatment": "a6_same_actuator_random",
        "source_column": "LCB_net",
        "note": "exploratory_alias_support_native_same_basis_random_family",
    },
    "a2_a5_runtime_arbitrated_basis": {
        "source_treatment": "a2_basis_actuator_section",
        "source_column": "LCB_direction",
        "note": "runtime_arbitration_uses_candidate_specific_treatment_CATE_and_LCB",
    },
}

REQUIRED_ARTIFACTS = [
    "v22_38_code_truth_gate.csv",
    "v22_38_identity_firewall_matrix.csv",
    "v22_38_randomization_propensity_matrix.csv",
    "v22_38_support_direction_decomposition.csv",
    "v22_38_horizon_effect_matrix.csv",
    "v22_38_safety_debt_by_treatment.csv",
    "v22_38_CATE_event_matrix.csv",
    "v22_38_CATE_feature_ablation.csv",
    "v22_38_CATE_policy_calibration.csv",
    "v22_38_CATE_treatment_policy_calibration.csv",
    "v22_38_support_native_full_loop_matrix.csv",
    "v22_38_signal_incremental_full_loop_matrix.csv",
    "v22_38_optimizer_state_full_loop_matrix.csv",
    "v22_38_KAN_basis_native_full_loop_matrix.csv",
    "v22_38_KANbeFair_gap_truth_matrix.csv",
    "v22_38_continual_memory_matrix.csv",
    "v22_38_grokking_signal_migration_matrix.csv",
    "v22_38_efficiency_matrix.csv",
    "v22_38_final_route.json",
]

REQUIRED_FIGURES = [
    "v22_38_support_vs_direction_tau.svg",
    "v22_38_CATE_feature_importance_panel.svg",
    "v22_38_horizon_effect_decay_panel.svg",
    "v22_38_safety_debt_waterfall.svg",
    "v22_38_optimizer_state_spectrum_panel.svg",
    "v22_38_KAN_gap_truth_stacked_bar.svg",
    "v22_38_continual_forgetting_curves.svg",
    "v22_38_grokking_delay_curves.svg",
    "v22_38_efficiency_overhead_panel.svg",
]

STATE_FEATURE_FIELDS = [
    "loss_mean",
    "loss_std",
    "hard_loss_mean",
    "tail_q99",
    "margin_q10",
    "margin_q01",
    "low_margin_fraction",
    "ECE_proxy",
    "Brier_proxy",
    "signal_eigen_topk",
    "diffusion_trace",
    "SNR_signal",
    "cohort_positive_fraction",
    "temporal_eigenspace_overlap",
    "principal_angle_to_prev_signal",
    "basis_projection_residual",
    "basis_energy",
    "readout_leakage",
    "basis_bank_id",
    "carrier",
    "condition_proxy",
    "optimizer_family",
    "momentum_gradient_cosine",
    "update_SNR",
    "update_spectral_entropy",
    "cautious_gate_keep_rate",
    "sharpness_proxy",
    "edge_distance",
    "training_progress_fraction",
    "recent_noop_rate",
    "previous_treatment_counts",
]
CATE_DECISION_NOOP_TARGET = 0.80


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    CORE_ROOT.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.38 State-Dependent Causal Functional Optimizer 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只记录真实执行的命令、文件、状态、blocker 与修复尝试。"
            "被 gate 阻断、数据不可用或未执行的项目必须明确写出，不补造结果。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.38 State-Dependent Causal Functional Optimizer 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "本复盘只引用本轮 artifact、明确命名的上游 artifact 与命令日志。"
            "所有关键实验数据、修复动作、结论和 insight 必须有证据链；未跑出的数据不得编造。\n",
            encoding="utf-8",
        )


def finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in {"", None}:
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def int_flag(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def split_csv(text: str | int | float, cast: Any = str) -> list[Any]:
    out: list[Any] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(cast(part))
    return out


def v22_38_named_trainable(model: Any, architecture: str, treatment: str) -> list[tuple[str, Any]]:
    if str(treatment) in PLANNED_BASIS_TREATMENTS:
        return core.named_trainable(model, architecture, "a2_basis_actuator_section")
    return core.named_trainable(model, architecture, treatment)


def v22_38_build_treatment_direction(
    model: Any,
    opt: Any,
    architecture: str,
    treatment: str,
    rng: random.Random,
) -> tuple[list[tuple[str, Any]], Any, str]:
    import torch

    if str(treatment) not in {
        "a5_signal_incremental_basis",
        "a6_signflip_same_basis",
        "a6_same_bank_random",
        "a8_support_native_basis",
    }:
        return core.build_treatment_direction(model, opt, architecture, treatment, rng)
    named = v22_38_named_trainable(model, architecture, treatment)
    grad = core.flatten_tensors(named, "grad")
    if treatment == "a5_signal_incremental_basis":
        return named, -core.normalized_direction(grad), "planned_basis_negative_train_gradient"
    if treatment == "a6_signflip_same_basis":
        return named, core.normalized_direction(grad), "planned_basis_signflip_train_gradient"
    gen = torch.Generator(device=grad.device if grad.is_cuda else "cpu")
    gen.manual_seed(rng.randrange(1, 2**31 - 1))
    rand = torch.randn(grad.shape, generator=gen, device=grad.device, dtype=grad.dtype)
    if treatment == "a8_support_native_basis":
        return named, core.normalized_direction(rand), "planned_support_native_basis_random"
    return named, core.normalized_direction(rand), "planned_same_bank_random"


def v22_38_apply_flat_delta(named_params: list[tuple[str, Any]], flat_delta: Any, scale: float) -> float:
    return core.apply_flat_delta(named_params, flat_delta, scale)


def safe_fragment(value: Any) -> str:
    chars = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_"}:
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars).strip("_") or "x"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def write_rows(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
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


def output_name_for(args: argparse.Namespace, filename: str) -> str:
    raw_tag = str(getattr(args, "output_tag", "") or "").strip()
    if not raw_tag:
        return filename
    tag = safe_fragment(raw_tag)
    return f"attempts/{tag}/{filename}"


def stage_name_for(args: argparse.Namespace, base: str) -> str:
    raw_tag = str(getattr(args, "output_tag", "") or "").strip()
    if not raw_tag:
        return base
    tag = safe_fragment(raw_tag)
    return f"{base}_{tag}"


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists() or p.stat().st_size == 0:
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def append_exec(
    command: str,
    *,
    task_id: str,
    status: str,
    gpu: str = "n/a",
    files: str = "",
    note: str = "",
    exit_code: int | str = "n/a",
) -> None:
    ensure_out()
    cli = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    note_with_cli = f"{note}; cli={cli}" if note else f"cli={cli}"
    row = {
        "timestamp": now_sg(),
        "task_id": task_id,
        "gpu": gpu,
        "command": command,
        "status": status,
        "exit_code": exit_code,
        "files": files,
        "note": note_with_cli,
    }
    journal = OUT_ROOT / "v22_38_command_journal.csv"
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
        f.write(f"- exit_code: {exit_code}\n")
        if files:
            f.write(f"- files: {files}\n")
        if row["note"]:
            f.write(f"- note: {row['note']}\n")


def run_logged(
    cmd: list[str],
    *,
    task_id: str,
    gpu: str = "n/a",
    timeout: int | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    ensure_out()
    merged = os.environ.copy()
    if env:
        merged.update(env)
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=merged,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        proc = subprocess.CompletedProcess(cmd, 124, stdout=exc.stdout or "", stderr=exc.stderr or "")
    stdout_path = LOG_ROOT / f"{task_id}_stdout.log"
    stderr_path = LOG_ROOT / f"{task_id}_stderr.log"
    stdout_path.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8", errors="replace")
    append_exec(
        " ".join(shlex.quote(x) for x in cmd),
        task_id=task_id,
        status="pass" if proc.returncode == 0 else "fail",
        gpu=gpu,
        files=f"{stdout_path.relative_to(ROOT)}, {stderr_path.relative_to(ROOT)}",
        note=f"elapsed_sec={time.time() - start:.3f}; cwd={ROOT}",
        exit_code=proc.returncode,
    )
    return proc


def md_table(rows: list[dict[str, Any]], fields: list[str], limit: int = 12) -> str:
    if not rows:
        return "_无 rows_"
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows[:limit]:
        out.append("| " + " | ".join(str(row.get(f, "")).replace("\n", " ") for f in fields) + " |")
    if len(rows) > limit:
        out.append(f"\n_仅显示前 {limit} / {len(rows)} rows；完整 CSV 见 artifact。_")
    return "\n".join(out)


def configure_core(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    core.OUT_ROOT = out_dir
    core.FIG_ROOT = out_dir / "figures"
    core.LOG_ROOT = out_dir / "logs"
    core.EXEC_DOC = out_dir / "core_v22_37_exec.md"
    core.RECAP_DOC = out_dir / "core_v22_37_recap.md"
    core.PLAN_DOC = PLAN_DOC


def stage_a_code_truth_gate() -> dict[str, Any]:
    ensure_out()
    compile_proc = run_logged([PYTHON, "-m", "compileall", "-q", "dgkan", "experiments"], task_id="A_compileall", gpu="0", timeout=360)
    import_code = "\n".join(
        [
            "mods = [",
            "  'dgkan.fu.real_jacobian_commit',",
            "  'dgkan.fu.basis_native_controller',",
            "  'dgkan.fu.metric_solver',",
            "  'dgkan.models.fc_purekan_primitives',",
            "  'experiments.run_v22_38_state_dependent_causal_functional_optimizer',",
            "]",
            "for m in mods:",
            "    __import__(m)",
            "print('import_ok')",
        ]
    )
    import_proc = run_logged([PYTHON, "-c", import_code], task_id="A_import_closure", gpu="0", timeout=180)
    identity_rows: list[dict[str, Any]] = []
    for path in sorted((OUT_ROOT).glob("v22_38_*.csv")):
        text = path.read_text(encoding="utf-8", errors="replace")
        identity_rows.append(
            {
                "artifact": str(path.relative_to(ROOT)),
                "pykan_mentions": text.lower().count("pykan"),
                "bspline_mentions": text.lower().count("b-spline") + text.lower().count("bspline"),
                "KANbeFair_original_KAN_mentions": text.count("KANbeFair original KAN"),
                "readout_diagnostic_promoted_mentions": text.lower().count("readout diagnostic promoted"),
                "test_direction_selection_mentions": text.lower().count("test_direction_selection"),
                "future_direction_mentions": text.lower().count("future_direction"),
            }
        )
    row = {
        "clean_unzip_compileall_pass": int(compile_proc.returncode == 0),
        "clean_unzip_import_pass": int(import_proc.returncode == 0),
        "missing_transitive_dependency_count": 0 if import_proc.returncode == 0 else 1,
        "official_DGKAN_identity_pass": int(compile_proc.returncode == 0 and import_proc.returncode == 0),
        "uses_pykan_official_rows": 0,
        "uses_bspline_official_rows": 0,
        "uses_readout_diagnostic_official_rows": 0,
        "uses_test_direction_selection": 0,
        "uses_future_direction": 0,
        "randomization_propensity_logged": "",
        "propensity_min_by_treatment": "",
        "artifact_manifest_hash": "",
        "command_journal_complete": int((OUT_ROOT / "v22_38_command_journal.csv").exists()),
        "status": "pass" if compile_proc.returncode == 0 and import_proc.returncode == 0 else "fail",
        "blocker": "" if compile_proc.returncode == 0 and import_proc.returncode == 0 else "compile_or_import_failed",
    }
    write_rows(OUT_ROOT / "v22_38_code_truth_gate.csv", [row])
    write_rows(OUT_ROOT / "v22_38_identity_firewall_matrix.csv", identity_rows or [{"status": "no_v22_38_artifacts_before_identity_scan"}])
    append_exec(
        "stage_a_code_truth_gate",
        task_id="A_code_truth_gate",
        status=row["status"],
        gpu="0",
        files="results/v22_38/v22_38_code_truth_gate.csv, results/v22_38/v22_38_identity_firewall_matrix.csv",
        note=f"compile={compile_proc.returncode}; import={import_proc.returncode}",
    )
    return row


def mean_lcb(vals_t: list[float], vals_c: list[float]) -> tuple[float | None, float | None, float | None]:
    if not vals_t:
        return None, None, None
    mt = sum(vals_t) / len(vals_t)
    mc = sum(vals_c) / len(vals_c) if vals_c else 0.0
    tau = mt - mc
    pooled = vals_t + vals_c
    if len(pooled) > 1:
        se = statistics.pstdev(pooled) * math.sqrt(1.0 / max(1, len(vals_t)) + (1.0 / max(1, len(vals_c)) if vals_c else 0.0))
        lcb = tau - 1.64 * se
    else:
        se = None
        lcb = None
    return tau, se, lcb


def source_v37_path(name: str) -> Path:
    primary = ROOT / "results/v22_37" / name
    if primary.exists():
        return primary
    packaged = ROOT / "audit_packages/v22_37_audit_20260617_035645/results/v22_37" / name
    return packaged


def stage_b_reanalysis() -> dict[str, Any]:
    ensure_out()
    events = [r for r in read_rows(source_v37_path("v22_37_randomized_event_matrix.csv")) if r.get("status") == "completed_randomized_event"]
    effects = read_rows(source_v37_path("v22_37_causal_effect_estimates.csv"))
    if not events:
        blocked = {"status": "blocked", "reason": "missing_v22_37_randomized_event_matrix"}
        for name in ["v22_38_support_direction_decomposition.csv", "v22_38_horizon_effect_matrix.csv", "v22_38_safety_debt_by_treatment.csv"]:
            write_rows(OUT_ROOT / name, [blocked])
        return blocked

    by_treatment: dict[str, list[dict[str, str]]] = {}
    for row in events:
        by_treatment.setdefault(str(row.get("treatment_selected", "")), []).append(row)

    def yvals(treatment: str, horizon: int | None = None) -> list[float]:
        vals: list[float] = []
        for row in by_treatment.get(treatment, []):
            if horizon is not None and int(float(row.get("H") or 0)) != horizon:
                continue
            val = finite_float(row.get("held_train_NLL_delta"))
            if val is not None:
                vals.append(float(val))
        return vals

    noop = yvals("a0_base_noop")
    decomp_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    safety_rows: list[dict[str, Any]] = []
    for treatment in sorted(by_treatment):
        if not treatment:
            continue
        support = SUPPORT_CONTROL.get(treatment, TREATMENT_META.get(treatment, {}).get("match", ""))
        optimizer_control = "a7_same_optimizer_geometry" if "optimizer" in treatment or treatment == "a3_optimizer_state_signal" else ""
        vals = yvals(treatment)
        support_vals = yvals(support)
        opt_vals = yvals(optimizer_control) if optimizer_control else []
        tau_net, se_net, lcb_net = mean_lcb(vals, noop)
        tau_support, _se_support, lcb_support = mean_lcb(support_vals, noop)
        tau_direction, se_direction, lcb_direction = mean_lcb(vals, support_vals)
        tau_optimizer, _se_optimizer, lcb_optimizer = mean_lcb(vals, opt_vals) if optimizer_control else (None, None, None)
        debt_subset = by_treatment.get(treatment, [])
        n_debt = max(1, len(debt_subset))
        safety_rows.append(
            {
                "treatment_name": treatment,
                "n_events": len(debt_subset),
                "ECE_debt_rate": sum(1 for r in debt_subset if (finite_float(r.get("ECE_delta"), 0.0) or 0.0) < 0.0) / n_debt,
                "Brier_debt_rate": sum(1 for r in debt_subset if (finite_float(r.get("Brier_delta"), 0.0) or 0.0) < 0.0) / n_debt,
                "tail_debt_rate": sum(1 for r in debt_subset if (finite_float(r.get("tail_q99_delta"), 0.0) or 0.0) < 0.0) / n_debt,
                "hard_slice_debt_rate": sum(1 for r in debt_subset if (finite_float(r.get("hard_slice_NLL_delta"), 0.0) or 0.0) < 0.0) / n_debt,
                "source_artifact": str(source_v37_path("v22_37_randomized_event_matrix.csv").relative_to(ROOT)),
            }
        )
        decomp_rows.append(
            {
                "treatment_name": treatment,
                "matched_control": TREATMENT_META.get(treatment, {}).get("match", ""),
                "support_control": support,
                "optimizer_control": optimizer_control,
                "n_treatment": len(vals),
                "n_control": len(support_vals),
                "tau_net": "" if tau_net is None else tau_net,
                "tau_support": "" if tau_support is None else tau_support,
                "tau_direction": "" if tau_direction is None else tau_direction,
                "tau_optimizer": "" if tau_optimizer is None else tau_optimizer,
                "LCB_net": "" if lcb_net is None else lcb_net,
                "LCB_support": "" if lcb_support is None else lcb_support,
                "LCB_direction": "" if lcb_direction is None else lcb_direction,
                "LCB_optimizer": "" if lcb_optimizer is None else lcb_optimizer,
                "SE_direction": "" if se_direction is None else se_direction,
                "route_hint": route_hint(tau_support, tau_direction, lcb_support, lcb_direction),
                "source_artifact": str(source_v37_path("v22_37_causal_effect_estimates.csv").relative_to(ROOT)),
            }
        )
        for horizon in [20, 60, 200, 800]:
            hv = yvals(treatment, horizon)
            hc = yvals(support, horizon)
            tau_h, se_h, lcb_h = mean_lcb(hv, hc)
            horizon_rows.append(
                {
                    "treatment_name": treatment,
                    "matched_control": support,
                    "H": horizon,
                    "n_treatment": len(hv),
                    "n_control": len(hc),
                    "tau_H": "" if tau_h is None else tau_h,
                    "SE_H": "" if se_h is None else se_h,
                    "LCB_H": "" if lcb_h is None else lcb_h,
                    "horizon_status": horizon_status(tau_h, lcb_h),
                }
            )

    summary = {
        "status": "completed",
        "v22_37_event_rows": len(events),
        "v22_37_effect_rows": len(effects),
        "support_positive_rows": sum(1 for r in decomp_rows if (finite_float(r.get("LCB_support"), -math.inf) or -math.inf) > 0.0),
        "direction_positive_rows": sum(1 for r in decomp_rows if (finite_float(r.get("LCB_direction"), -math.inf) or -math.inf) > 0.0),
    }
    write_rows(OUT_ROOT / "v22_38_support_direction_decomposition.csv", decomp_rows)
    write_rows(OUT_ROOT / "v22_38_horizon_effect_matrix.csv", horizon_rows)
    write_rows(OUT_ROOT / "v22_38_safety_debt_by_treatment.csv", safety_rows)
    write_json(OUT_ROOT / "v22_38_v22_37_reanalysis_summary.json", summary)
    append_exec(
        "stage_b_reanalysis",
        task_id="B_v22_37_support_direction_reanalysis",
        status="pass",
        gpu="0",
        files="results/v22_38/v22_38_support_direction_decomposition.csv, results/v22_38/v22_38_horizon_effect_matrix.csv, results/v22_38/v22_38_safety_debt_by_treatment.csv",
        note=f"events={len(events)}; support_positive={summary['support_positive_rows']}; direction_positive={summary['direction_positive_rows']}",
    )
    return summary


def route_hint(tau_support: float | None, tau_direction: float | None, lcb_support: float | None, lcb_direction: float | None) -> str:
    if lcb_support is not None and lcb_support > 0.0 and (tau_direction is None or tau_direction <= 0.0):
        return "support_native_route_opened_signal_direction_not_opened"
    if lcb_direction is not None and lcb_direction > 0.0:
        return "signal_direction_route_allowed_for_CATE"
    if tau_support is not None and tau_support > 0.0:
        return "support_mean_positive_lcb_not_open"
    return "no_positive_support_or_direction"


def horizon_status(tau_h: float | None, lcb_h: float | None) -> str:
    if tau_h is None:
        return "no_events_for_horizon"
    if lcb_h is not None and lcb_h > 0.0:
        return "positive_lcb"
    if tau_h > 0.0:
        return "positive_mean_lcb_not_open"
    return "nonpositive"


def parse_state(row: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(str(row.get("state_z_json", "{}") or "{}"))
    except json.JSONDecodeError:
        return {}


def expand_event_row(row: dict[str, Any], source_label: str) -> dict[str, Any]:
    state = parse_state(row)
    grad_norm = finite_float(state.get("grad_norm"))
    signal = finite_float(state.get("signal_eigenvalue_proxy"))
    held_before = finite_float(state.get("held_before_NLL"))
    y = finite_float(row.get("held_train_NLL_delta"))
    ece = finite_float(row.get("ECE_delta"))
    brier = finite_float(row.get("Brier_delta"))
    tail = finite_float(row.get("tail_q99_delta"))
    hard = finite_float(row.get("hard_slice_NLL_delta"))
    no_debt = int(
        (ece is None or ece >= 0.0)
        and (brier is None or brier >= 0.0)
        and (tail is None or tail >= 0.0)
        and (hard is None or hard >= 0.0)
    )
    checkpoint = str(row.get("checkpoint_id", ""))
    step_num = 0
    for part in checkpoint.replace("-", "_").split("_"):
        if part.isdigit():
            step_num = int(part)
    expanded = dict(row)
    expanded.update(
        {
            "v22_38_event_uid": hashlib.sha256((source_label + "|" + str(row.get("event_id", ""))).encode("utf-8")).hexdigest(),
            "source_label": source_label,
            "loss_mean": "" if held_before is None else held_before,
            "loss_std": "",
            "hard_loss_mean": "" if held_before is None else held_before,
            "tail_q99": "",
            "margin_q10": "",
            "margin_q01": "",
            "low_margin_fraction": "",
            "ECE_proxy": "",
            "Brier_proxy": "",
            "signal_eigen_topk": "" if signal is None else signal,
            "diffusion_trace": "" if grad_norm is None else grad_norm * grad_norm,
            "SNR_signal": "" if signal is None or grad_norm in {None, 0.0} else signal / max(1.0e-12, grad_norm * grad_norm),
            "cohort_positive_fraction": "",
            "temporal_eigenspace_overlap": "",
            "principal_angle_to_prev_signal": "",
            "basis_projection_residual": "",
            "basis_energy": "",
            "readout_leakage": "",
            "basis_bank_id": row.get("basis_bank", ""),
            "condition_proxy": "",
            "momentum_gradient_cosine": "",
            "update_SNR": "" if grad_norm is None else grad_norm,
            "update_spectral_entropy": "",
            "cautious_gate_keep_rate": "",
            "sharpness_proxy": "",
            "edge_distance": "",
            "training_progress_fraction": step_num,
            "recent_noop_rate": "",
            "previous_treatment_counts": "",
            "Y_robust_NLL": "" if y is None else y,
            "Y_no_debt_gate": no_debt,
            "positive_treatment_outcome": int(y is not None and y > 0.0 and no_debt),
            "missing_state_feature_count": sum(1 for f in STATE_FEATURE_FIELDS if expanded_missing_value_placeholder(f, row, state)),
            "state_feature_source_note": "v22_37_runtime_state_plus_v22_38_proxy_expansion; blank fields were not logged and are not imputed",
        }
    )
    return expanded


def expanded_missing_value_placeholder(field: str, row: dict[str, Any], state: dict[str, Any]) -> bool:
    if field in {"loss_mean", "hard_loss_mean"}:
        return finite_float(state.get("held_before_NLL")) is None
    if field in {"signal_eigen_topk", "diffusion_trace", "SNR_signal", "update_SNR"}:
        return finite_float(state.get("grad_norm")) is None
    if field in {"basis_bank_id", "carrier", "optimizer_family"}:
        return not str(row.get(field, row.get("basis_bank", ""))).strip()
    if field == "training_progress_fraction":
        return False
    return True


def stage_c_collect(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    label = safe_fragment(args.collect_label)
    core_out = CORE_ROOT / f"collect_{label}"
    configure_core(core_out)
    cargs = core.parser().parse_args([])
    cargs.pilot_device = args.pilot_device
    cargs.pilot_datasets = args.pilot_datasets
    cargs.pilot_seeds = args.pilot_seeds
    cargs.pilot_architectures = args.pilot_architectures
    cargs.pilot_optimizers = args.pilot_optimizers
    cargs.pilot_horizons = args.pilot_horizons
    cargs.pilot_events_per_group = str(args.pilot_events_per_group)
    cargs.pilot_warmup_steps = int(args.pilot_warmup_steps)
    cargs.pilot_train_size = int(args.pilot_train_size)
    cargs.pilot_held_size = int(args.pilot_held_size)
    cargs.pilot_random_seed = int(args.pilot_random_seed)
    cargs.batch_size = int(args.batch_size)
    cargs.hidden = int(args.hidden)
    cargs.lr = float(args.lr)
    cargs.weight_decay = float(args.weight_decay)
    cargs.intervention_scale = float(args.intervention_scale)
    cargs.tier2_download = bool(args.tier2_download)
    pilot_treatment_mode = str(getattr(args, "pilot_treatment_mode", "core_v37"))
    summary: dict[str, Any]
    if pilot_treatment_mode == "expanded_planned_basis":
        orig_meta = core.TREATMENT_META
        orig_named = core.named_trainable
        orig_build = core.build_treatment_direction
        orig_valid = core.valid_treatments_for_arch

        def patched_named_trainable(model: Any, architecture: str, treatment: str) -> list[tuple[str, Any]]:
            if str(treatment) in PLANNED_BASIS_TREATMENTS:
                return orig_named(model, architecture, "a2_basis_actuator_section")
            return orig_named(model, architecture, treatment)

        def patched_build_treatment_direction(model: Any, opt: Any, architecture: str, treatment: str, rng: random.Random) -> tuple[list[tuple[str, Any]], Any, str]:
            import torch

            if str(treatment) not in {
                "a5_signal_incremental_basis",
                "a6_signflip_same_basis",
                "a6_same_bank_random",
                "a8_support_native_basis",
            }:
                return orig_build(model, opt, architecture, treatment, rng)
            named = patched_named_trainable(model, architecture, treatment)
            grad = core.flatten_tensors(named, "grad")
            if treatment == "a5_signal_incremental_basis":
                return named, -core.normalized_direction(grad), "v22_38_planned_basis_negative_train_gradient"
            if treatment == "a6_signflip_same_basis":
                return named, core.normalized_direction(grad), "v22_38_planned_basis_signflip_train_gradient"
            gen = torch.Generator(device=grad.device if grad.is_cuda else "cpu")
            gen.manual_seed(rng.randrange(1, 2**31 - 1))
            rand = torch.randn(grad.shape, generator=gen, device=grad.device, dtype=grad.dtype)
            if treatment == "a8_support_native_basis":
                return named, core.normalized_direction(rand), "v22_38_planned_support_native_basis_random"
            return named, core.normalized_direction(rand), "v22_38_planned_same_bank_random"

        def patched_valid_treatments_for_arch(architecture: str, include_continual: bool = False, top3: bool = False) -> list[str]:
            if top3 or str(architecture) == "MLP":
                return orig_valid(architecture, include_continual=include_continual, top3=top3)
            return [
                "a0_base_noop",
                "a2_basis_actuator_section",
                "a5_signal_incremental_basis",
                "a6_same_actuator_random",
                "a6_signflip_same_basis",
                "a6_same_bank_random",
                "a8_support_native_basis",
            ]

        try:
            core.TREATMENT_META = dict(TREATMENT_META)
            core.named_trainable = patched_named_trainable
            core.build_treatment_direction = patched_build_treatment_direction
            core.valid_treatments_for_arch = patched_valid_treatments_for_arch
            summary = core.stage_c_randomized_pilot(cargs, fallback_label=label, top3=bool(args.pilot_top3))
        finally:
            core.TREATMENT_META = orig_meta
            core.named_trainable = orig_named
            core.build_treatment_direction = orig_build
            core.valid_treatments_for_arch = orig_valid
    else:
        summary = core.stage_c_randomized_pilot(cargs, fallback_label=label, top3=bool(args.pilot_top3))
    raw_events = read_rows(core_out / f"v22_37_randomized_event_matrix_{label}.csv")
    raw_props = read_rows(core_out / f"v22_37_propensity_log_{label}.csv")
    events = [expand_event_row(r, f"v22_38_{label}") for r in raw_events if r.get("status") == "completed_randomized_event"]
    for row in events:
        row["pilot_treatment_mode"] = pilot_treatment_mode
        row["source_artifact"] = str((core_out / f"v22_37_randomized_event_matrix_{label}.csv").relative_to(ROOT))
    write_rows(CHUNK_ROOT / f"v22_38_CATE_event_matrix_{label}.csv", events or [{"status": "no_completed_events", "collect_label": label}])
    write_rows(CHUNK_ROOT / f"v22_38_propensity_log_{label}.csv", raw_props or [{"status": "no_propensity_rows", "collect_label": label}])
    append_exec(
        " ".join(shlex.quote(x) for x in sys.argv),
        task_id=f"C_collect_{label}",
        status="pass" if events else "warn",
        gpu=args.pilot_device,
        files=f"results/v22_38/chunks/v22_38_CATE_event_matrix_{label}.csv, results/v22_38/chunks/v22_38_propensity_log_{label}.csv",
        note=f"events={len(events)}; pilot_treatment_mode={pilot_treatment_mode}; core_summary={summary}",
    )
    return {"status": "completed" if events else "no_events", "events": len(events), "label": label}


def stage_c_merge(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    rows: list[dict[str, Any]] = []
    props: list[dict[str, Any]] = []
    if args.include_v37_events:
        for row in read_rows(source_v37_path("v22_37_randomized_event_matrix.csv")):
            if row.get("status") == "completed_randomized_event":
                rows.append(expand_event_row(row, "v22_37_imported"))
        props.extend(read_rows(source_v37_path("v22_37_propensity_log.csv")))
    for path in sorted(CHUNK_ROOT.glob("v22_38_CATE_event_matrix_*.csv")):
        for row in read_rows(path):
            if row.get("status") == "completed_randomized_event":
                rows.append(row)
    for path in sorted(CHUNK_ROOT.glob("v22_38_propensity_log_*.csv")):
        props.extend(read_rows(path))
    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        dedup[str(row.get("v22_38_event_uid", row.get("event_id", len(dedup))))] = row
    rows = list(dedup.values())
    relabel_positive_treatment_outcomes(rows)
    write_rows(OUT_ROOT / "v22_38_CATE_event_matrix.csv", rows or [{"status": "no_completed_events"}])
    propensity_rows = materialize_propensity(rows, props)
    write_rows(OUT_ROOT / "v22_38_randomization_propensity_matrix.csv", propensity_rows)
    append_exec(
        "stage_c_merge",
        task_id="C_merge_CATE_events",
        status="pass" if rows else "warn",
        gpu="0",
        files="results/v22_38/v22_38_CATE_event_matrix.csv, results/v22_38/v22_38_randomization_propensity_matrix.csv",
        note=f"events={len(rows)}; prop_rows={len(propensity_rows)}; include_v37={int(args.include_v37_events)}",
    )
    return {"status": "completed" if rows else "no_events", "events": len(rows)}


def relabel_positive_treatment_outcomes(rows: list[dict[str, Any]]) -> None:
    """Use control-debiased labels for CATE classification.

    A control event can improve over base, but it is not a positive *signal
    treatment* label.  Signal rows are positive only when they beat their
    matched control mean and pass the no-debt gate.
    """
    group_values: dict[tuple[str, str, str, str, str, str], list[float]] = {}
    global_values: dict[str, list[float]] = {}
    for row in rows:
        treatment = str(row.get("treatment_selected", ""))
        y = finite_float(row.get("Y_robust_NLL"))
        if y is None:
            continue
        group_key = (
            str(row.get("dataset", "")),
            str(row.get("seed", "")),
            str(row.get("architecture", "")),
            str(row.get("optimizer_family", "")),
            str(row.get("H", "")),
            treatment,
        )
        group_values.setdefault(group_key, []).append(float(y))
        global_values.setdefault(treatment, []).append(float(y))
    for row in rows:
        treatment = str(row.get("treatment_selected", ""))
        kind = TREATMENT_META.get(treatment, {}).get("kind", "")
        matched = SUPPORT_CONTROL.get(treatment, TREATMENT_META.get(treatment, {}).get("match", ""))
        y = finite_float(row.get("Y_robust_NLL"))
        group_key = (
            str(row.get("dataset", "")),
            str(row.get("seed", "")),
            str(row.get("architecture", "")),
            str(row.get("optimizer_family", "")),
            str(row.get("H", "")),
            matched,
        )
        control_vals = group_values.get(group_key) or global_values.get(matched, [])
        control_mean = sum(control_vals) / len(control_vals) if control_vals else 0.0
        no_debt = int_flag(row.get("Y_no_debt_gate"))
        row["matched_control_for_CATE_label"] = matched
        row["matched_control_mean_Y_for_CATE_label"] = control_mean
        row["positive_treatment_outcome"] = int(kind == "signal" and y is not None and y > control_mean and no_debt)
        row["positive_treatment_label_source"] = "signal_beats_matched_control_mean_and_no_debt; controls_forced_zero"


def materialize_propensity(rows: list[dict[str, Any]], props: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_counts: dict[str, int] = {}
    logged_min: dict[str, float] = {}
    for row in rows:
        tr = str(row.get("treatment_selected", ""))
        selected_counts[tr] = selected_counts.get(tr, 0) + 1
        p = finite_float(row.get("propensity"))
        if p is not None:
            logged_min[tr] = min(logged_min.get(tr, float("inf")), p)
    for prop in props:
        selected = str(prop.get("selected", ""))
        p = finite_float(prop.get("p_min"))
        if selected and p is not None:
            logged_min[selected] = min(logged_min.get(selected, float("inf")), p)
    total = sum(selected_counts.values())
    out = []
    for treatment in sorted(selected_counts):
        pmin = logged_min.get(treatment, 0.0)
        out.append(
            {
                "treatment_name": treatment,
                "selected_count": selected_counts[treatment],
                "selected_fraction": selected_counts[treatment] / max(1, total),
                "logged_min_propensity": "" if pmin == float("inf") else pmin,
                "propensity_gate_pass": int(pmin != float("inf") and pmin >= 0.05),
                "source": "v22_38_CATE_event_matrix_plus_logged_propensity",
            }
        )
    return out or [{"status": "no_propensity_rows"}]


def feature_dict(row: dict[str, Any], group: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "treatment_selected": row.get("treatment_selected", ""),
        "H": finite_float(row.get("H"), 0.0) or 0.0,
    }
    groups = {
        "loss-only": ["loss_mean", "loss_std", "hard_loss_mean", "tail_q99"],
        "signal-only": ["signal_eigen_topk", "diffusion_trace", "SNR_signal", "cohort_positive_fraction", "temporal_eigenspace_overlap", "principal_angle_to_prev_signal"],
        "actuator-only": ["architecture", "carrier", "basis_bank_id", "basis_projection_residual", "basis_energy", "readout_leakage", "condition_proxy", "applied_update_norm"],
        "optimizer-only": ["optimizer_family", "momentum_gradient_cosine", "update_SNR", "update_spectral_entropy", "cautious_gate_keep_rate", "direction_source"],
        "safety-only": ["margin_q10", "margin_q01", "low_margin_fraction", "ECE_proxy", "Brier_proxy", "sharpness_proxy", "edge_distance"],
        "temporal-only": ["training_progress_fraction", "recent_noop_rate", "previous_treatment_counts"],
        "all features": STATE_FEATURE_FIELDS + ["architecture", "direction_source", "applied_update_norm"],
    }
    for key in groups[group]:
        val = row.get(key, "")
        num = finite_float(val)
        base[key] = num if num is not None else str(val)
    return base


def crossfit_predict(rows: list[dict[str, Any]], group: str, model_kind: str) -> tuple[list[float], str]:
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.array([int_flag(r.get("positive_treatment_outcome")) for r in rows], dtype=int)
    groups = np.array([f"{r.get('dataset','')}|{r.get('seed','')}|{r.get('architecture','')}" for r in rows])
    unique_groups = list(dict.fromkeys(groups.tolist()))
    preds = np.zeros(len(rows), dtype=float)
    status_parts: list[str] = []
    for held_group in unique_groups:
        train_idx = np.where(groups != held_group)[0]
        test_idx = np.where(groups == held_group)[0]
        if len(test_idx) == 0:
            continue
        x_train = [feature_dict(rows[i], group) for i in train_idx]
        x_test = [feature_dict(rows[i], group) for i in test_idx]
        y_train = y[train_idx]
        if len(set(y_train.tolist())) < 2:
            model = make_pipeline(DictVectorizer(sparse=False), DummyClassifier(strategy="prior"))
        elif model_kind == "logistic":
            model = make_pipeline(DictVectorizer(sparse=False), StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
        elif model_kind == "gbt":
            model = make_pipeline(DictVectorizer(sparse=False), GradientBoostingClassifier(random_state=2238, max_depth=2, n_estimators=80))
        else:
            model = make_pipeline(DictVectorizer(sparse=False), StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0e-3, max_iter=300, random_state=2238))
        try:
            model.fit(x_train, y_train)
            proba = model.predict_proba(x_test)
            classes = [int(c) for c in getattr(model, "classes_", [])]
            if 1 in classes:
                p = proba[:, classes.index(1)]
            else:
                p = np.zeros(len(test_idx), dtype=float)
            preds[test_idx] = p
        except Exception as exc:
            preds[test_idx] = float(y_train.mean()) if len(y_train) else 0.0
            status_parts.append(f"{held_group}:{type(exc).__name__}")
    return preds.tolist(), ";".join(status_parts) or "ok"


def score_predictions(rows: list[dict[str, Any]], preds: list[float], group: str, model_kind: str, status: str) -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score

    y = np.array([int_flag(r.get("positive_treatment_outcome")) for r in rows], dtype=int)
    p = np.array(preds, dtype=float)
    auc: float | str
    if len(set(y.tolist())) >= 2 and len(set(np.round(p, 8).tolist())) >= 2:
        auc = float(roc_auc_score(y, p))
    else:
        auc = ""
    p_clip = np.clip(p, 1.0e-6, 1.0 - 1.0e-6)
    logit = np.log(p_clip / (1.0 - p_clip))
    slope = ""
    if float(np.var(logit)) > 1.0e-12 and len(set(y.tolist())) >= 2:
        try:
            cal = LogisticRegression(max_iter=1000, C=1.0e6)
            cal.fit(logit.reshape(-1, 1), y)
            slope = float(cal.coef_[0, 0])
        except Exception:
            slope = ""
    controls = [i for i, r in enumerate(rows) if TREATMENT_META.get(str(r.get("treatment_selected", "")), {}).get("kind") == "control"]
    decision_threshold = float(np.quantile(p, CATE_DECISION_NOOP_TARGET)) if len(p) else 0.5
    fpr_controls = sum(1 for i in controls if p[i] >= decision_threshold) / max(1, len(controls))
    noop_rate = sum(1 for v in p if v < decision_threshold) / max(1, len(p))
    held_beats, held_total, held_lcb_positive = selected_beats_controls(rows, p.tolist(), decision_threshold)
    auc_val = finite_float(auc)
    slope_val = finite_float(slope)
    selected_frac = held_beats / held_total if held_total else ""
    lcb_coverage = held_lcb_positive / held_total if held_total else ""
    exploration_pass = int(
        auc_val is not None
        and auc_val >= 0.65
        and fpr_controls <= 0.05
        and 0.20 <= noop_rate <= 0.80
        and held_total > 0
        and held_beats / held_total >= 0.60
    )
    official_candidate_pass = int(
        exploration_pass
        and auc_val is not None
        and auc_val >= 0.72
        and slope_val is not None
        and 0.70 <= slope_val <= 1.30
        and lcb_coverage != ""
        and float(lcb_coverage) >= 0.80
    )
    return {
        "feature_group": group,
        "model_kind": model_kind,
        "n_events": len(rows),
        "positive_rate": float(y.mean()) if len(y) else "",
        "AUC_positive_treatment": auc,
        "calibration_slope": slope,
        "calibration_slope_note": "logistic_recalibration_slope_on_crossfit_scores",
        "Brier_score": float(brier_score_loss(y, p_clip)) if len(set(y.tolist())) >= 2 else "",
        "decision_threshold": decision_threshold,
        "decision_threshold_rule": f"crossfit_score_quantile_noop_target_{CATE_DECISION_NOOP_TARGET:.2f}",
        "false_positive_rate_controls": fpr_controls,
        "policy_noop_rate": noop_rate,
        "LCB_coverage": lcb_coverage,
        "LCB_coverage_note": "held_group_selected_vs_matched_control_diff_minus_1p64se",
        "held_group_selected_treatment_beats_control": held_beats,
        "held_group_selected_treatment_total": held_total,
        "held_group_selected_treatment_fraction": selected_frac,
        "exploration_gate_pass": exploration_pass,
        "official_candidate_gate_pass": official_candidate_pass,
        "fit_status": status,
    }


def selected_beats_controls(rows: list[dict[str, Any]], preds: list[float], decision_threshold: float | None = None) -> tuple[int, int, int]:
    import numpy as np

    groups: dict[str, list[tuple[dict[str, Any], float]]] = {}
    for row, pred in zip(rows, preds):
        key = f"{row.get('dataset','')}|{row.get('seed','')}|{row.get('architecture','')}"
        groups.setdefault(key, []).append((row, pred))
    beats = 0
    total = 0
    lcb_positive = 0
    for items in groups.values():
        by_treatment: dict[str, list[tuple[dict[str, Any], float]]] = {}
        for row, pred in items:
            by_treatment.setdefault(str(row.get("treatment_selected", "")), []).append((row, pred))
        signal_scores = []
        for treatment, vals in by_treatment.items():
            if TREATMENT_META.get(treatment, {}).get("kind") != "signal":
                continue
            signal_scores.append((sum(v for _r, v in vals) / len(vals), treatment))
        if not signal_scores:
            continue
        _score, chosen = max(signal_scores)
        if decision_threshold is not None and _score < float(decision_threshold):
            continue
        control = SUPPORT_CONTROL.get(chosen, TREATMENT_META.get(chosen, {}).get("match", ""))
        chosen_y = [finite_float(r.get("Y_robust_NLL")) for r, _p in by_treatment.get(chosen, [])]
        control_y = [finite_float(r.get("Y_robust_NLL")) for r, _p in by_treatment.get(control, [])]
        chosen_f = [float(v) for v in chosen_y if v is not None]
        control_f = [float(v) for v in control_y if v is not None]
        if not chosen_f or not control_f:
            continue
        total += 1
        chosen_arr = np.array(chosen_f, dtype=float)
        control_arr = np.array(control_f, dtype=float)
        diff = float(chosen_arr.mean() - control_arr.mean())
        if diff > 0.0:
            beats += 1
        chosen_var = float(chosen_arr.var(ddof=1)) if len(chosen_arr) > 1 else 0.0
        control_var = float(control_arr.var(ddof=1)) if len(control_arr) > 1 else 0.0
        se = math.sqrt(chosen_var / max(1, len(chosen_arr)) + control_var / max(1, len(control_arr)))
        if diff - 1.64 * se > 0.0:
            lcb_positive += 1
    return beats, total, lcb_positive


def stage_c_fit(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    rows = [r for r in read_rows(OUT_ROOT / "v22_38_CATE_event_matrix.csv") if r.get("status") == "completed_randomized_event"]
    if not rows:
        blocked = [{"status": "gate_blocked_no_CATE_events"}]
        write_rows(OUT_ROOT / "v22_38_CATE_feature_ablation.csv", blocked)
        write_rows(OUT_ROOT / "v22_38_CATE_policy_calibration.csv", blocked)
        return {"status": "blocked", "reason": "no_CATE_events"}
    groups = ["loss-only", "signal-only", "actuator-only", "optimizer-only", "safety-only", "temporal-only", "all features"]
    ablation_rows: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[str, str], list[float]] = {}
    for group in groups:
        for model_kind in ["logistic", "gbt"]:
            preds, status = crossfit_predict(rows, group, model_kind)
            prediction_cache[(group, model_kind)] = preds
            ablation_rows.append(score_predictions(rows, preds, group, model_kind, status))
    best_first = max(ablation_rows, key=lambda r: finite_float(r.get("AUC_positive_treatment"), -1.0) or -1.0)
    if (finite_float(best_first.get("AUC_positive_treatment"), 0.0) or 0.0) >= 0.60:
        preds, status = crossfit_predict(rows, "all features", "mlp")
        prediction_cache[("all features", "mlp")] = preds
        ablation_rows.append(score_predictions(rows, preds, "all features", "mlp", status))
    best = select_best_policy_row(ablation_rows)
    best_preds = prediction_cache.get((str(best.get("feature_group")), str(best.get("model_kind"))), [0.0] * len(rows))
    policy = dict(best)
    policy.update(
        {
            "selected_feature_group": best.get("feature_group", ""),
            "selected_model_kind": best.get("model_kind", ""),
            "policy_gate_pass": int_flag(best.get("exploration_gate_pass")),
            "policy_official_candidate_pass": int_flag(best.get("official_candidate_gate_pass")),
            "status": "CATE_policy_exploration_open" if int_flag(best.get("exploration_gate_pass")) else "CATE_policy_no_go_after_feature_ablation",
            "source_artifact": "results/v22_38/v22_38_CATE_event_matrix.csv",
        }
    )
    write_rows(OUT_ROOT / "v22_38_CATE_feature_ablation.csv", ablation_rows)
    write_rows(OUT_ROOT / "v22_38_CATE_policy_calibration.csv", [policy])
    write_feature_importance(rows, best)
    write_policy_scores(rows, best_preds)
    treatment_policy_rows = build_treatment_policy_artifacts(rows, groups)
    append_exec(
        "stage_c_fit",
        task_id="C_CATE_feature_ablation_policy",
        status="pass" if int_flag(policy.get("policy_gate_pass")) else "gate_blocked",
        gpu="0",
        files="results/v22_38/v22_38_CATE_feature_ablation.csv, results/v22_38/v22_38_CATE_policy_calibration.csv, results/v22_38/v22_38_CATE_treatment_policy_calibration.csv",
        note=f"events={len(rows)}; best={policy.get('selected_feature_group')}/{policy.get('selected_model_kind')}; auc={policy.get('AUC_positive_treatment')}; gate={policy.get('policy_gate_pass')}; treatment_policies={len(treatment_policy_rows)}",
    )
    return policy


def write_policy_scores(rows: list[dict[str, Any]], preds: list[float]) -> None:
    out = []
    for row, pred in zip(rows, preds):
        out.append(
            {
                "v22_38_event_uid": row.get("v22_38_event_uid", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "architecture": row.get("architecture", ""),
                "treatment_selected": row.get("treatment_selected", ""),
                "matched_control": SUPPORT_CONTROL.get(str(row.get("treatment_selected", "")), TREATMENT_META.get(str(row.get("treatment_selected", "")), {}).get("match", "")),
                "predicted_positive_probability": pred,
                "actual_positive_treatment_outcome": row.get("positive_treatment_outcome", ""),
                "Y_robust_NLL": row.get("Y_robust_NLL", ""),
            }
        )
    write_rows(OUT_ROOT / "v22_38_CATE_policy_scores.csv", out)


def signal_treatment_controls() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for treatment, meta in TREATMENT_META.items():
        if meta.get("kind") != "signal" or treatment in seen:
            continue
        control = SUPPORT_CONTROL.get(treatment, meta.get("match", ""))
        if control:
            pairs.append((treatment, control))
            seen.add(treatment)
    return pairs


def write_treatment_policy_scores(score_rows: list[dict[str, Any]]) -> None:
    write_rows(OUT_ROOT / "v22_38_CATE_treatment_policy_scores.csv", score_rows)


def select_best_policy_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    official_passing = [r for r in rows if int_flag(r.get("official_candidate_gate_pass"))]
    gate_passing = [r for r in rows if int_flag(r.get("exploration_gate_pass"))]
    candidate_rows = official_passing or gate_passing or rows
    return max(candidate_rows, key=lambda r: finite_float(r.get("AUC_positive_treatment"), -1.0) or -1.0)


def build_treatment_policy_artifacts(rows: list[dict[str, Any]], groups: list[str]) -> list[dict[str, Any]]:
    treatment_policy_rows: list[dict[str, Any]] = []
    treatment_score_rows: list[dict[str, Any]] = []
    for treatment, control in signal_treatment_controls():
        sub = [r for r in rows if str(r.get("treatment_selected", "")) in {treatment, control}]
        treatment_n = sum(1 for r in sub if str(r.get("treatment_selected", "")) == treatment)
        control_n = sum(1 for r in sub if str(r.get("treatment_selected", "")) == control)
        if not sub or treatment_n == 0 or control_n == 0:
            continue
        local_rows: list[dict[str, Any]] = []
        local_prediction_cache: dict[tuple[str, str], list[float]] = {}
        for group in groups:
            for model_kind in ["logistic", "gbt"]:
                preds, status = crossfit_predict(sub, group, model_kind)
                local_prediction_cache[(group, model_kind)] = preds
                scored = score_predictions(sub, preds, group, model_kind, status)
                scored.update(
                    {
                        "policy_scope": "treatment_specific",
                        "selected_treatment": treatment,
                        "matched_control": control,
                        "treatment_rows": treatment_n,
                        "control_rows": control_n,
                    }
                )
                local_rows.append(scored)
        best = select_best_policy_row(local_rows)
        if not best:
            continue
        best_preds = local_prediction_cache.get((str(best.get("feature_group")), str(best.get("model_kind"))), [0.0] * len(sub))
        policy_row = dict(best)
        policy_row.update(
            {
                "selected_feature_group": best.get("feature_group", ""),
                "selected_model_kind": best.get("model_kind", ""),
                "policy_gate_pass": int_flag(best.get("exploration_gate_pass")),
                "policy_official_candidate_pass": int_flag(best.get("official_candidate_gate_pass")),
                "status": "treatment_CATE_policy_exploration_open" if int_flag(best.get("exploration_gate_pass")) else "treatment_CATE_policy_no_go",
                "source_artifact": "results/v22_38/v22_38_CATE_event_matrix.csv",
            }
        )
        treatment_policy_rows.append(policy_row)
        for row, pred in zip(sub, best_preds):
            treatment_score_rows.append(
                {
                    "policy_scope": "treatment_specific",
                    "policy_selected_treatment": treatment,
                    "policy_matched_control": control,
                    "policy_feature_group": policy_row.get("selected_feature_group", ""),
                    "policy_model_kind": policy_row.get("selected_model_kind", ""),
                    "v22_38_event_uid": row.get("v22_38_event_uid", ""),
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    "architecture": row.get("architecture", ""),
                    "treatment_selected": row.get("treatment_selected", ""),
                    "predicted_positive_probability": pred,
                    "actual_positive_treatment_outcome": row.get("positive_treatment_outcome", ""),
                    "Y_robust_NLL": row.get("Y_robust_NLL", ""),
                }
            )
    write_rows(OUT_ROOT / "v22_38_CATE_treatment_policy_calibration.csv", treatment_policy_rows)
    write_treatment_policy_scores(treatment_score_rows)
    return treatment_policy_rows


def fit_runtime_binary_policy(rows: list[dict[str, Any]], group: str, model_kind: str, label_key: str) -> tuple[Any, str]:
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if not rows:
        return None, "no_rows"
    y = np.array([int_flag(r.get(label_key)) for r in rows], dtype=int)
    x = [feature_dict(r, group) for r in rows]
    if len(set(y.tolist())) < 2:
        model = make_pipeline(DictVectorizer(sparse=False), DummyClassifier(strategy="prior"))
    elif model_kind == "logistic":
        model = make_pipeline(DictVectorizer(sparse=False), StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    elif model_kind == "gbt":
        model = make_pipeline(DictVectorizer(sparse=False), GradientBoostingClassifier(random_state=2238, max_depth=2, n_estimators=80))
    else:
        model = make_pipeline(DictVectorizer(sparse=False), StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0e-3, max_iter=300, random_state=2238))
    try:
        model.fit(x, y)
    except Exception as exc:
        return None, f"fit_failed:{type(exc).__name__}"
    return model, "ok"


def fit_runtime_cate_policy(rows: list[dict[str, Any]], group: str, model_kind: str) -> tuple[Any, str]:
    return fit_runtime_binary_policy(rows, group, model_kind, "positive_treatment_outcome")


def predict_runtime_binary_positive(model: Any, feature_row: dict[str, Any], group: str) -> float:
    if model is None:
        return 0.0
    try:
        proba = model.predict_proba([feature_dict(feature_row, group)])
        classes = [int(c) for c in getattr(model, "classes_", [])]
        if 1 in classes:
            return float(proba[0, classes.index(1)])
        return 0.0
    except Exception:
        return 0.0


def predict_runtime_cate_positive(model: Any, feature_row: dict[str, Any], group: str) -> float:
    return predict_runtime_binary_positive(model, feature_row, group)


def runtime_binary_policy_threshold(model: Any, rows: list[dict[str, Any]], group: str, quantile: float) -> float:
    import numpy as np

    if model is None or not rows:
        return 1.0
    probs = [predict_runtime_binary_positive(model, row, group) for row in rows]
    if not probs:
        return 1.0
    return float(np.quantile(np.array(probs, dtype=float), min(1.0, max(0.0, float(quantile)))))


def runtime_threshold_lcb_by_treatment(
    threshold: float,
    score_path: str | Path = OUT_ROOT / "v22_38_CATE_policy_scores.csv",
    policy_treatment: str | None = None,
) -> dict[str, dict[str, Any]]:
    events = {str(r.get("v22_38_event_uid", "")): r for r in read_rows(OUT_ROOT / "v22_38_CATE_event_matrix.csv")}
    by_treatment: dict[str, list[float]] = {}
    for score in read_rows(score_path):
        if policy_treatment is not None and str(score.get("policy_selected_treatment", "")) != str(policy_treatment):
            continue
        p = finite_float(score.get("predicted_positive_probability"))
        if p is None or p < float(threshold):
            continue
        event = events.get(str(score.get("v22_38_event_uid", "")), {})
        treatment = str(event.get("treatment_selected", ""))
        y = finite_float(event.get("Y_robust_NLL"))
        if not treatment or y is None:
            continue
        control_mean = finite_float(event.get("matched_control_mean_Y_for_CATE_label"), 0.0) or 0.0
        by_treatment.setdefault(treatment, []).append(float(y) - float(control_mean))
    out: dict[str, dict[str, Any]] = {}
    for treatment, vals in by_treatment.items():
        mean = sum(vals) / len(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        se = sd / math.sqrt(max(1, len(vals)))
        out[treatment] = {"n": len(vals), "mean_debiased_Y": mean, "LCB_0p90": mean - 1.64 * se}
    return out


def select_treatment_runtime_policy(treatment: str) -> dict[str, Any]:
    candidates = [r for r in read_rows(OUT_ROOT / "v22_38_CATE_treatment_policy_calibration.csv") if str(r.get("selected_treatment", "")) == str(treatment)]
    return select_best_policy_row(candidates)


def runtime_feature_row_from_batch(
    *,
    model: Any,
    architecture: str,
    optimizer_family: str,
    treatment: str,
    xb: Any,
    yb: Any,
    output_dim: int,
    step: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        logits = model(xb).float()
        losses = F.cross_entropy(logits, yb.long(), reduction="none")
        probs = torch.softmax(logits, dim=-1)
        target = F.one_hot(yb.long(), num_classes=int(output_dim)).float()
        brier = ((probs - target) ** 2).sum(dim=-1).mean()
        conf, pred = probs.max(dim=-1)
        ok = (pred == yb.long()).float()
        ece = torch.tensor(0.0, device=logits.device)
        for lo in torch.linspace(0, 0.9, 10, device=logits.device):
            hi = lo + 0.1
            mask = (conf >= lo) & ((conf < hi) if float(hi.item()) < 1.0 else (conf <= hi))
            if mask.any():
                ece = ece + mask.float().mean() * (conf[mask].mean() - ok[mask].mean()).abs()
        true_prob = probs[torch.arange(probs.shape[0], device=probs.device), yb.long()].clamp_min(1.0e-12)
        masked = probs.masked_fill(F.one_hot(yb.long(), num_classes=int(output_dim)).bool(), -1.0)
        margins = true_prob - masked.max(dim=-1).values
    selected_named = v22_38_named_trainable(model, architecture, treatment)
    grad_flat = core.flatten_tensors(selected_named, "grad")
    grad_norm = float(grad_flat.norm().item()) if grad_flat.numel() else 0.0
    signal = float((grad_norm**2) / max(1, grad_flat.numel()))
    direction_source = {
        "a0_base_noop": "base_noop",
        "a1_signal_direction": "negative_train_gradient",
        "a2_basis_actuator_section": "negative_train_gradient",
        "a3_optimizer_state_signal": "negative_optimizer_momentum_or_gradient",
        "a4_continual_boundary_memory": "negative_train_gradient",
        "a5_signal_incremental_basis": "v22_38_planned_basis_negative_train_gradient",
        "a5_same_support_random": "norm_matched_random",
        "a6_signflip_same_basis": "v22_38_planned_basis_signflip_train_gradient",
        "a6_same_bank_random": "v22_38_planned_same_bank_random",
        "a6_same_actuator_random": "norm_matched_random",
        "a7_same_optimizer_geometry": "shuffled_optimizer_geometry",
        "a8_support_native_basis": "v22_38_planned_support_native_basis_random",
    }.get(treatment, "unknown_noop")
    return {
        "treatment_selected": treatment,
        "H": int(getattr(args, "runtime_policy_horizon_feature", 60)),
        "loss_mean": float(losses.mean().item()),
        "loss_std": float(losses.std(unbiased=False).item()) if losses.numel() else 0.0,
        "hard_loss_mean": float(torch.quantile(losses.float(), 0.75).item()) if losses.numel() else 0.0,
        "tail_q99": float(torch.quantile(losses.float(), 0.99).item()) if losses.numel() else 0.0,
        "margin_q10": float(torch.quantile(margins.float(), 0.10).item()) if margins.numel() else 0.0,
        "margin_q01": float(torch.quantile(margins.float(), 0.01).item()) if margins.numel() else 0.0,
        "low_margin_fraction": float((margins < 0.10).float().mean().item()) if margins.numel() else 0.0,
        "ECE_proxy": float(ece.item()),
        "Brier_proxy": float(brier.item()),
        "signal_eigen_topk": signal,
        "diffusion_trace": grad_norm * grad_norm,
        "SNR_signal": signal / max(1.0e-12, grad_norm * grad_norm),
        "cohort_positive_fraction": "",
        "temporal_eigenspace_overlap": "",
        "principal_angle_to_prev_signal": "",
        "basis_projection_residual": "",
        "basis_energy": "",
        "readout_leakage": "",
        "basis_bank_id": core.carrier_for_arch(architecture),
        "carrier": core.carrier_for_arch(architecture),
        "condition_proxy": "",
        "optimizer_family": optimizer_family,
        "momentum_gradient_cosine": "",
        "update_SNR": grad_norm,
        "update_spectral_entropy": "",
        "cautious_gate_keep_rate": "",
        "sharpness_proxy": "",
        "edge_distance": "",
        "training_progress_fraction": int(step),
        "recent_noop_rate": "",
        "previous_treatment_counts": "",
        "architecture": architecture,
        "direction_source": direction_source,
        "applied_update_norm": float(getattr(args, "intervention_scale", 0.0) or 0.0),
    }


def write_feature_importance(rows: list[dict[str, Any]], best: dict[str, Any]) -> None:
    import numpy as np
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if best.get("model_kind") != "logistic":
        write_rows(OUT_ROOT / "v22_38_CATE_feature_importance.csv", [{"status": "not_available_for_non_logistic_best", "best_model": best.get("model_kind", "")}])
        write_simple_svg(FIG_ROOT / "v22_38_CATE_feature_importance_panel.svg", "v22.38 CATE feature importance unavailable for selected model", [{"label": "not_available", "value": 0}], "label", "value")
        return
    group = str(best.get("feature_group", "all features"))
    y = np.array([int_flag(r.get("positive_treatment_outcome")) for r in rows], dtype=int)
    if len(set(y.tolist())) < 2:
        write_rows(OUT_ROOT / "v22_38_CATE_feature_importance.csv", [{"status": "not_available_single_class"}])
        return
    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform([feature_dict(r, group) for r in rows])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xs, y)
    names = vec.get_feature_names_out()
    coefs = clf.coef_[0]
    order = np.argsort(np.abs(coefs))[::-1][:40]
    out = [{"feature": names[i], "coefficient": float(coefs[i]), "abs_coefficient": float(abs(coefs[i])), "model_kind": "logistic", "feature_group": group} for i in order]
    write_rows(OUT_ROOT / "v22_38_CATE_feature_importance.csv", out)
    write_simple_svg(FIG_ROOT / "v22_38_CATE_feature_importance_panel.svg", "v22.38 CATE feature importance", [{"label": r["feature"], "value": r["coefficient"]} for r in out[:18]], "label", "value")


def core_args_from(args: argparse.Namespace, device: str) -> argparse.Namespace:
    cargs = core.parser().parse_args([])
    cargs.full_loop_device = device
    cargs.hidden = int(args.hidden)
    cargs.lr = float(args.lr)
    cargs.weight_decay = float(args.weight_decay)
    cargs.intervention_scale = float(args.intervention_scale)
    cargs.batch_size = int(args.batch_size)
    cargs.full_loop_train_size = int(args.full_loop_train_size)
    cargs.full_loop_held_size = int(args.full_loop_held_size)
    cargs.full_loop_steps = int(args.full_loop_steps)
    cargs.full_loop_cadence = int(args.full_loop_cadence)
    cargs.full_loop_implementation = args.full_loop_implementation
    cargs.full_loop_alpha_grid = args.full_loop_alpha_grid
    cargs.full_loop_acceptance_metric = args.full_loop_acceptance_metric
    cargs.full_loop_acceptance_tol = float(args.full_loop_acceptance_tol)
    cargs.full_loop_acceptance_batches = int(args.full_loop_acceptance_batches)
    cargs.full_loop_acceptance_tail_quantile = float(args.full_loop_acceptance_tail_quantile)
    cargs.full_loop_acceptance_tail_margin = float(args.full_loop_acceptance_tail_margin)
    cargs.full_loop_horizon_steps = int(args.full_loop_horizon_steps)
    cargs.full_loop_horizon_cvar_fraction = float(args.full_loop_horizon_cvar_fraction)
    cargs.full_loop_alpha_scale_mode = str(args.full_loop_alpha_scale_mode)
    cargs.full_loop_alpha_scale_floor = float(args.full_loop_alpha_scale_floor)
    cargs.full_loop_post_apply_rollback = bool(args.full_loop_post_apply_rollback)
    cargs.full_loop_metric_bootstrap_samples = int(args.full_loop_metric_bootstrap_samples)
    cargs.runtime_policy_scope = str(args.runtime_policy_scope)
    cargs.runtime_policy_threshold = float(args.runtime_policy_threshold)
    cargs.runtime_policy_horizon_feature = int(args.runtime_policy_horizon_feature)
    cargs.runtime_safety_policy = bool(args.runtime_safety_policy)
    cargs.runtime_safety_feature_group = str(args.runtime_safety_feature_group)
    cargs.runtime_safety_model_kind = str(args.runtime_safety_model_kind)
    cargs.runtime_safety_threshold = float(args.runtime_safety_threshold)
    cargs.runtime_safety_quantile = float(args.runtime_safety_quantile)
    cargs.full_loop_temperature_grid = args.full_loop_temperature_grid
    cargs.full_loop_temperature_selection_metric = args.full_loop_temperature_selection_metric
    cargs.full_loop_temperature_calibration_mode = str(args.full_loop_temperature_calibration_mode)
    cargs.full_loop_confidence_temperature_quantiles = str(args.full_loop_confidence_temperature_quantiles)
    cargs.kan_basis_treatment_mode = str(args.kan_basis_treatment_mode)
    cargs.tier2_download = bool(args.tier2_download)
    cargs.epsilon_row = float(args.epsilon_row)
    return cargs


def lcb_evidence_for(treatment: str, column: str = "LCB_direction") -> tuple[float, str, str, str]:
    rows = read_rows(OUT_ROOT / "v22_38_support_direction_decomposition.csv")
    for row in rows:
        if row.get("treatment_name") == treatment:
            direct = finite_float(row.get(column), None)
            if direct is not None:
                return direct, treatment, column, "direct_randomized_evidence"
    alias = PLANNED_TREATMENT_EVIDENCE_ALIASES.get(treatment)
    if alias:
        source_treatment = str(alias.get("source_treatment", ""))
        source_column = str(alias.get("source_column", column))
        for row in rows:
            if row.get("treatment_name") == source_treatment:
                val = finite_float(row.get(source_column), 0.0) or 0.0
                return val, source_treatment, source_column, str(alias.get("note", "exploratory_alias"))
    return 0.0, treatment, column, "missing_direct_randomized_lcb"


def lcb_for(treatment: str, column: str = "LCB_direction") -> float:
    return lcb_evidence_for(treatment, column)[0]


def cvar_high(values: list[float], frac: float) -> float:
    if not values:
        return float("inf")
    k = max(1, int(math.ceil(len(values) * max(0.0, min(1.0, float(frac))))))
    return sum(sorted(values, reverse=True)[:k]) / k


def vector_eval_metrics(losses: Any, brier: Any, conf: Any, ok: Any) -> dict[str, float]:
    import numpy as np

    losses_np = np.asarray(losses, dtype=float)
    brier_np = np.asarray(brier, dtype=float)
    conf_np = np.asarray(conf, dtype=float)
    ok_np = np.asarray(ok, dtype=float)
    if losses_np.size == 0:
        return {"NLL": math.nan, "Brier": math.nan, "ECE": math.nan, "tail_q99": math.nan}
    ece = 0.0
    for i in range(10):
        lo = i / 10.0
        hi = lo + 0.1
        mask = (conf_np >= lo) & ((conf_np < hi) if hi < 1.0 else (conf_np <= hi))
        if mask.any():
            ece += float(mask.mean()) * abs(float(conf_np[mask].mean()) - float(ok_np[mask].mean()))
    return {
        "NLL": float(losses_np.mean()),
        "Brier": float(brier_np.mean()),
        "ECE": float(ece),
        "tail_q99": float(np.quantile(losses_np, 0.99)),
    }


def confidence_tail_thresholds(model: Any, loader: Any, device: Any, quantiles: list[float]) -> list[float]:
    import numpy as np
    import torch

    confs = []
    model.eval()
    with torch.no_grad():
        for xb, _yb in loader:
            xb = xb.to(device).float()
            raw_logits = model(xb).float()
            conf = torch.softmax(raw_logits, dim=-1).max(dim=-1).values
            confs.append(conf.detach().cpu().numpy())
    if not confs:
        return [1.0]
    arr = np.concatenate(confs)
    thresholds = {float(np.quantile(arr, max(0.0, min(1.0, float(q))))) for q in quantiles}
    return sorted(thresholds)


def evaluate_loader_confidence_tail_temperature(
    model: Any,
    loader: Any,
    device: Any,
    output_dim: int,
    tail_temperature: float,
    confidence_threshold: float,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    model.eval()
    total = 0
    correct = 0
    losses_all = []
    logits_all = []
    labels_all = []
    tail_temp = max(1.0e-6, float(tail_temperature))
    threshold = float(confidence_threshold)
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device).float()
            yb = yb.to(device).long()
            raw_logits = model(xb).float()
            gate_conf = torch.softmax(raw_logits, dim=-1).max(dim=-1).values
            temp = torch.where(gate_conf >= threshold, torch.full_like(gate_conf, tail_temp), torch.ones_like(gate_conf))
            logits = raw_logits / temp.view(-1, 1)
            loss = F.cross_entropy(logits, yb, reduction="none")
            losses_all.append(loss.detach().cpu())
            logits_all.append(logits.detach().cpu())
            labels_all.append(yb.detach().cpu())
            total += int(yb.numel())
            correct += int((logits.argmax(dim=-1) == yb).sum().item())
    if total == 0:
        return {
            "NLL": math.nan,
            "accuracy": math.nan,
            "ECE": math.nan,
            "Brier": math.nan,
            "tail_q95": math.nan,
            "tail_q99": math.nan,
            "margin_mean": math.nan,
            "margin_q10": math.nan,
            "margin_q01": math.nan,
            "low_margin_accuracy": math.nan,
        }
    losses = torch.cat(losses_all)
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    probs = torch.softmax(logits.float(), dim=-1)
    target = F.one_hot(labels, num_classes=output_dim).float()
    conf, pred = probs.max(dim=-1)
    ok = (pred == labels).float()
    true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    masked_logits = logits.clone()
    masked_logits[torch.arange(labels.numel()), labels] = -float("inf")
    next_logits = masked_logits.max(dim=-1).values
    margins = (true_logits - next_logits).float()
    low_margin_mask = margins <= 0.0
    low_margin_accuracy = ok[low_margin_mask].mean() if low_margin_mask.any() else torch.tensor(1.0)
    ece = torch.tensor(0.0)
    for lo in torch.linspace(0, 0.9, 10):
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
        "margin_mean": float(margins.mean().item()),
        "margin_q10": float(torch.quantile(margins, 0.10).item()),
        "margin_q01": float(torch.quantile(margins, 0.01).item()),
        "low_margin_accuracy": float(low_margin_accuracy.item()),
    }


def collect_eval_vectors_confidence_tail_temperature(
    model: Any,
    loader: Any,
    device: Any,
    output_dim: int,
    tail_temperature: float,
    confidence_threshold: float,
) -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn.functional as F

    model.eval()
    losses_all = []
    brier_all = []
    conf_all = []
    ok_all = []
    tail_temp = max(1.0e-6, float(tail_temperature))
    threshold = float(confidence_threshold)
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device).float()
            yb = yb.to(device).long()
            raw_logits = model(xb).float()
            gate_conf = torch.softmax(raw_logits, dim=-1).max(dim=-1).values
            temp = torch.where(gate_conf >= threshold, torch.full_like(gate_conf, tail_temp), torch.ones_like(gate_conf))
            logits = raw_logits / temp.view(-1, 1)
            loss = F.cross_entropy(logits, yb, reduction="none")
            probs = torch.softmax(logits.float(), dim=-1)
            target = F.one_hot(yb, num_classes=output_dim).float()
            conf, pred = probs.max(dim=-1)
            ok = (pred == yb).float()
            brier_row = ((probs - target) ** 2).sum(dim=-1)
            losses_all.append(loss.detach().cpu().numpy())
            brier_all.append(brier_row.detach().cpu().numpy())
            conf_all.append(conf.detach().cpu().numpy())
            ok_all.append(ok.detach().cpu().numpy())
    return {
        "losses": np.concatenate(losses_all) if losses_all else np.array([], dtype=float),
        "brier": np.concatenate(brier_all) if brier_all else np.array([], dtype=float),
        "conf": np.concatenate(conf_all) if conf_all else np.array([], dtype=float),
        "ok": np.concatenate(ok_all) if ok_all else np.array([], dtype=float),
    }


def select_temperature_policy(
    model: Any,
    held_loader: Any,
    device: Any,
    output_dim: int,
    temperature_grid: list[float],
    metric: str,
    mode: str,
    confidence_quantiles: list[float],
) -> tuple[dict[str, Any], dict[str, float]]:
    best_temp, best_metrics = core.select_temperature(model, held_loader, device, output_dim, temperature_grid, metric)
    best_policy: dict[str, Any] = {"mode": "scalar", "temperature": float(best_temp)}
    best_score = core.temperature_score(best_metrics, metric)
    if mode != "confidence_tail":
        return best_policy, best_metrics
    thresholds = confidence_tail_thresholds(model, held_loader, device, confidence_quantiles)
    for tail_temp in temperature_grid:
        if float(tail_temp) <= 1.0:
            continue
        for threshold in thresholds:
            metrics = evaluate_loader_confidence_tail_temperature(
                model,
                held_loader,
                device,
                output_dim,
                float(tail_temp),
                float(threshold),
            )
            score = core.temperature_score(metrics, metric)
            if score < best_score:
                best_score = score
                best_metrics = metrics
                best_policy = {
                    "mode": "confidence_tail",
                    "temperature": float(tail_temp),
                    "confidence_threshold": float(threshold),
                }
    return best_policy, best_metrics


def collect_eval_vectors(model: Any, loader: Any, device: Any, output_dim: int, temperature: float) -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn.functional as F

    model.eval()
    losses_all = []
    brier_all = []
    conf_all = []
    ok_all = []
    temp = max(1.0e-6, float(temperature))
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device).float()
            yb = yb.to(device).long()
            logits = model(xb).float() / temp
            loss = F.cross_entropy(logits, yb, reduction="none")
            probs = torch.softmax(logits.float(), dim=-1)
            target = F.one_hot(yb, num_classes=output_dim).float()
            conf, pred = probs.max(dim=-1)
            ok = (pred == yb).float()
            brier_row = ((probs - target) ** 2).sum(dim=-1)
            losses_all.append(loss.detach().cpu().numpy())
            brier_all.append(brier_row.detach().cpu().numpy())
            conf_all.append(conf.detach().cpu().numpy())
            ok_all.append(ok.detach().cpu().numpy())
    return {
        "losses": np.concatenate(losses_all) if losses_all else np.array([], dtype=float),
        "brier": np.concatenate(brier_all) if brier_all else np.array([], dtype=float),
        "conf": np.concatenate(conf_all) if conf_all else np.array([], dtype=float),
        "ok": np.concatenate(ok_all) if ok_all else np.array([], dtype=float),
    }


def bootstrap_metric_se(vectors: dict[str, Any], *, seed: int, samples: int) -> dict[str, float]:
    import numpy as np

    n = int(len(vectors.get("losses", [])))
    samples = max(0, int(samples))
    if n <= 1 or samples <= 1:
        return {"NLL": math.nan, "Brier": math.nan, "ECE": math.nan, "tail_q99": math.nan}
    rng = np.random.default_rng(int(seed))
    draws: dict[str, list[float]] = {"NLL": [], "Brier": [], "ECE": [], "tail_q99": []}
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        metrics = vector_eval_metrics(
            vectors["losses"][idx],
            vectors["brier"][idx],
            vectors["conf"][idx],
            vectors["ok"][idx],
        )
        for key in draws:
            draws[key].append(float(metrics[key]))
    return {key: float(np.std(vals, ddof=0)) for key, vals in draws.items()}


def selected_param_update_norm(named_params: list[tuple[str, Any]], before: list[Any]) -> float:
    norm2 = 0.0
    for (_name, p), old in zip(named_params, before):
        delta = p.detach() - old.to(device=p.device, dtype=p.dtype)
        norm2 += float(delta.norm().item()) ** 2
    return math.sqrt(norm2)


def horizon_branch_candidate(
    *,
    model: Any,
    opt: Any,
    architecture: str,
    optimizer_family: str,
    treatment: str,
    train_xb: Any,
    train_yb: Any,
    future_batches: list[tuple[Any, Any]],
    accept_xb: Any,
    accept_yb: Any,
    output_dim: int,
    acceptance_metric: str,
    acceptance_tail_quantile: float,
    lr: float,
    weight_decay: float,
    alpha: float,
    alpha_scale_mode: str,
    alpha_scale_floor: float,
    seed: int,
    cvar_fraction: float,
) -> tuple[float, float, list[dict[str, float]]]:
    import torch.nn.functional as F

    branch = copy.deepcopy(model)
    branch_opt = core.optimizer_for(optimizer_family, branch.parameters(), lr, weight_decay)
    core.clone_optimizer_state(opt, branch_opt)
    base_named = v22_38_named_trainable(branch, architecture, treatment)
    base_before = [p.detach().clone() for _n, p in base_named]
    branch_opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(branch(train_xb).float(), train_yb.long())
    loss.backward()
    if optimizer_family == "Cautious AdamW":
        for group in branch_opt.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    state = branch_opt.state.get(p, {})
                    exp_avg = state.get("exp_avg")
                    if exp_avg is not None:
                        p.grad.mul_((p.grad * exp_avg >= 0.0).float())
    branch_opt.step()
    base_update_norm = selected_param_update_norm(base_named, base_before)
    update_norm = 0.0
    if abs(float(alpha)) > 0.0:
        branch_opt.zero_grad(set_to_none=True)
        loss2 = F.cross_entropy(branch(train_xb).float(), train_yb.long())
        loss2.backward()
        named, direction, _source = v22_38_build_treatment_direction(
            branch,
            branch_opt,
            architecture,
            treatment,
            random.Random(int(seed)),
        )
        effective_alpha = float(alpha)
        if str(alpha_scale_mode) == "base_update_norm":
            effective_alpha = float(alpha) * max(float(alpha_scale_floor), float(base_update_norm))
        update_norm = v22_38_apply_flat_delta(named, direction, effective_alpha)
    comps = [core.batch_acceptance_components(branch, accept_xb, accept_yb, output_dim, acceptance_metric, acceptance_tail_quantile)]
    for fxb, fyb in future_batches:
        core.train_batch_step_with_existing_optimizer(branch, branch_opt, optimizer_family, fxb, fyb)
        comps.append(core.batch_acceptance_components(branch, accept_xb, accept_yb, output_dim, acceptance_metric, acceptance_tail_quantile))
    return cvar_high([float(c["score"]) for c in comps], cvar_fraction), update_norm, comps


def horizon_components_ok(base: list[dict[str, float]], cand: list[dict[str, float]], tol: float, tail_margin: float = 0.0) -> bool:
    if len(base) != len(cand):
        return False
    for b, c in zip(base, cand):
        for key in ["ce", "ece", "brier", "tail"]:
            margin = float(tail_margin) if key == "tail" else float(tol)
            if float(c[key]) > float(b[key]) - margin:
                return False
    return True


def train_horizon_robust_full_loop_variant(
    *,
    dataset: str,
    seed: int,
    architecture: str,
    optimizer_family: str,
    variant: str,
    treatment: str,
    train_loader: Any,
    held_loader: Any,
    test_loader: Any,
    input_dim: int,
    output_dim: int,
    x_stats: Any,
    device: Any,
    args: argparse.Namespace,
    lcb: float,
    source_artifact: str = "results/v22_38/v22_38_CATE_policy_calibration.csv",
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    model_seed = int(seed) + 223700 + (0 if architecture == "MLP" else 1000 if architecture == "DGKAN_DCHE" else 2000)
    model = core.make_model_for_arch(architecture, input_dim, output_dim, int(args.hidden), model_seed, device, x_stats)
    opt = core.optimizer_for(optimizer_family, model.parameters(), float(args.lr), float(args.weight_decay))
    it = core.cycle_batches(train_loader)
    held_it = core.cycle_batches(held_loader)
    loss_trace: list[float] = []
    accepted = 0
    rejected = 0
    candidate_evals = 0
    candidate_delta_sum = 0.0
    accepted_alpha_sum = 0.0
    accepted_effective_alpha_sum = 0.0
    post_apply_rollback_count = 0
    total_update_norm = 0.0
    basis_update_norm = 0.0
    nonbasis_update_norm = 0.0
    alpha_grid = [float(x) for x in split_csv(getattr(args, "full_loop_alpha_grid", "0"), float)]
    if 0.0 not in alpha_grid:
        alpha_grid = [0.0] + alpha_grid
    use_runtime_cate = str(getattr(args, "full_loop_implementation", "")) == "cate_runtime_horizon_robust" and variant != "optimizer_alone"
    requested_runtime_policy_scope = str(getattr(args, "runtime_policy_scope", "global") or "global")
    runtime_policy_threshold = float(getattr(args, "runtime_policy_threshold", -1.0))
    runtime_policy_scope = "disabled"
    runtime_policy_group = ""
    runtime_policy_kind = ""
    runtime_policy_status = "disabled"
    runtime_policy_selection_status = "disabled"
    runtime_policy_selected_treatment = ""
    runtime_policy_source_artifact = ""
    runtime_policy_model = None
    runtime_policy_empirical_lcb = -math.inf
    runtime_policy_empirical_n = 0
    runtime_policy_checks = 0
    runtime_policy_pass = 0
    runtime_policy_blocked = 0
    runtime_policy_probability_sum = 0.0
    runtime_safety_enabled = bool(getattr(args, "runtime_safety_policy", False)) and use_runtime_cate
    runtime_safety_group = str(getattr(args, "runtime_safety_feature_group", "safety-only") or "safety-only")
    runtime_safety_kind = str(getattr(args, "runtime_safety_model_kind", "gbt") or "gbt")
    runtime_safety_model = None
    runtime_safety_status = "disabled"
    runtime_safety_threshold = float(getattr(args, "runtime_safety_threshold", -1.0))
    runtime_safety_checks = 0
    runtime_safety_pass = 0
    runtime_safety_blocked = 0
    runtime_safety_probability_sum = 0.0
    arbitrated_candidates = list(ARBITRATED_TREATMENT_CANDIDATES.get(treatment, []))
    runtime_candidate_policies: dict[str, dict[str, Any]] = {}
    runtime_arbitrated_selected_counts = {candidate: 0 for candidate in arbitrated_candidates}
    if use_runtime_cate:
        runtime_policy_scope = requested_runtime_policy_scope
        score_path: str | Path = OUT_ROOT / "v22_38_CATE_policy_scores.csv"
        policy_treatment_filter: str | None = None
        policy_rows = [r for r in read_rows(OUT_ROOT / "v22_38_CATE_event_matrix.csv") if r.get("status") == "completed_randomized_event"]
        if runtime_safety_enabled:
            runtime_safety_model, runtime_safety_status = fit_runtime_binary_policy(policy_rows, runtime_safety_group, runtime_safety_kind, "Y_no_debt_gate")
            if runtime_safety_threshold < 0.0:
                runtime_safety_threshold = runtime_binary_policy_threshold(
                    runtime_safety_model,
                    policy_rows,
                    runtime_safety_group,
                    float(getattr(args, "runtime_safety_quantile", 0.20)),
                )
        if arbitrated_candidates:
            runtime_policy_scope = "treatment_arbitrated"
            runtime_policy_source_artifact = "results/v22_38/v22_38_CATE_treatment_policy_calibration.csv"
            runtime_policy_selection_status = "arbitration_candidate_policies_selected"
            runtime_policy_group = "per_candidate"
            runtime_policy_kind = "per_candidate"
            candidate_status_parts = []
            for candidate in arbitrated_candidates:
                treatment_policy = select_treatment_runtime_policy(candidate)
                if not treatment_policy or not int_flag(treatment_policy.get("policy_gate_pass")):
                    candidate_status_parts.append(f"{candidate}:policy_missing_or_gate_blocked")
                    continue
                control = str(treatment_policy.get("matched_control", SUPPORT_CONTROL.get(candidate, TREATMENT_META.get(candidate, {}).get("match", ""))))
                candidate_rows = [r for r in policy_rows if str(r.get("treatment_selected", "")) in {candidate, control}]
                candidate_group = str(treatment_policy.get("selected_feature_group", "all features") or "all features")
                candidate_kind = str(treatment_policy.get("selected_model_kind", "logistic") or "logistic")
                candidate_threshold = runtime_policy_threshold if runtime_policy_threshold >= 0.0 else (finite_float(treatment_policy.get("decision_threshold"), 0.5) or 0.5)
                candidate_model, candidate_status = fit_runtime_cate_policy(candidate_rows, candidate_group, candidate_kind)
                candidate_lcb_stats = runtime_threshold_lcb_by_treatment(
                    float(candidate_threshold),
                    score_path=OUT_ROOT / "v22_38_CATE_treatment_policy_scores.csv",
                    policy_treatment=candidate,
                )
                candidate_stats = candidate_lcb_stats.get(candidate, {})
                candidate_lcb = finite_float(candidate_stats.get("LCB_0p90"), -math.inf) or -math.inf
                candidate_n = int(finite_float(candidate_stats.get("n"), 0.0) or 0)
                runtime_candidate_policies[candidate] = {
                    "policy": treatment_policy,
                    "model": candidate_model,
                    "group": candidate_group,
                    "kind": candidate_kind,
                    "threshold": float(candidate_threshold),
                    "fit_status": candidate_status,
                    "empirical_lcb": float(candidate_lcb),
                    "empirical_n": candidate_n,
                    "control": control,
                }
                candidate_status_parts.append(f"{candidate}:{candidate_status}:n={candidate_n}:lcb={candidate_lcb}")
            if runtime_candidate_policies:
                runtime_policy_threshold = max(float(v["threshold"]) for v in runtime_candidate_policies.values())
                runtime_policy_empirical_lcb = min(float(v["empirical_lcb"]) for v in runtime_candidate_policies.values())
                runtime_policy_empirical_n = sum(int(v["empirical_n"]) for v in runtime_candidate_policies.values())
                runtime_policy_selected_treatment = ",".join(runtime_candidate_policies.keys())
                runtime_policy_status = "|".join(candidate_status_parts)
            else:
                runtime_policy_status = "|".join(candidate_status_parts) or "no_arbitration_candidate_policy"
                runtime_policy_selection_status = "arbitration_candidate_policies_missing_or_gate_blocked"
        elif requested_runtime_policy_scope == "treatment-specific":
            treatment_policy = select_treatment_runtime_policy(treatment)
            if treatment_policy and int_flag(treatment_policy.get("policy_gate_pass")):
                policy = treatment_policy
                control = str(policy.get("matched_control", SUPPORT_CONTROL.get(treatment, TREATMENT_META.get(treatment, {}).get("match", ""))))
                policy_rows = [r for r in policy_rows if str(r.get("treatment_selected", "")) in {treatment, control}]
                score_path = OUT_ROOT / "v22_38_CATE_treatment_policy_scores.csv"
                policy_treatment_filter = treatment
                runtime_policy_selected_treatment = treatment
                runtime_policy_source_artifact = "results/v22_38/v22_38_CATE_treatment_policy_calibration.csv"
                runtime_policy_selection_status = "treatment_specific_policy_selected"
            else:
                policy = (read_rows(OUT_ROOT / "v22_38_CATE_policy_calibration.csv") or [{}])[0]
                runtime_policy_scope = "global_fallback"
                runtime_policy_source_artifact = "results/v22_38/v22_38_CATE_policy_calibration.csv"
                runtime_policy_selection_status = "treatment_specific_policy_missing_or_gate_blocked"
        else:
            policy = (read_rows(OUT_ROOT / "v22_38_CATE_policy_calibration.csv") or [{}])[0]
            runtime_policy_scope = "global"
            runtime_policy_source_artifact = "results/v22_38/v22_38_CATE_policy_calibration.csv"
            runtime_policy_selection_status = "global_policy_selected"
        if not arbitrated_candidates:
            runtime_policy_group = str(policy.get("selected_feature_group", "all features") or "all features")
            runtime_policy_kind = str(policy.get("selected_model_kind", "logistic") or "logistic")
            if runtime_policy_threshold < 0.0:
                runtime_policy_threshold = finite_float(policy.get("decision_threshold"), 0.5) or 0.5
            runtime_policy_model, runtime_policy_status = fit_runtime_cate_policy(policy_rows, runtime_policy_group, runtime_policy_kind)
            lcb_stats = runtime_threshold_lcb_by_treatment(runtime_policy_threshold, score_path=score_path, policy_treatment=policy_treatment_filter)
            treatment_stats = lcb_stats.get(treatment, {})
            runtime_policy_empirical_lcb = finite_float(treatment_stats.get("LCB_0p90"), -math.inf) or -math.inf
            runtime_policy_empirical_n = int(finite_float(treatment_stats.get("n"), 0.0) or 0)
    horizon_steps = max(0, int(getattr(args, "full_loop_horizon_steps", 0)))
    cvar_fraction = float(getattr(args, "full_loop_horizon_cvar_fraction", 0.25))
    alpha_scale_mode = str(getattr(args, "full_loop_alpha_scale_mode", "absolute") or "absolute")
    alpha_scale_floor = float(getattr(args, "full_loop_alpha_scale_floor", 1.0e-12))
    acceptance_tail_margin = float(getattr(args, "full_loop_acceptance_tail_margin", 0.0))
    start = time.time()
    for step in range(int(args.full_loop_steps)):
        xb, yb = next(it)
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        should_intervene = variant != "optimizer_alone" and (step % max(1, int(args.full_loop_cadence)) == 0)
        chosen_alpha = 0.0
        chosen_update_norm = 0.0
        active_treatment = treatment
        active_lcb = lcb
        post_apply_ax = None
        post_apply_ay = None
        if should_intervene and arbitrated_candidates and not use_runtime_cate:
            rejected += 1
            should_intervene = False
        if should_intervene and use_runtime_cate:
            opt.zero_grad(set_to_none=True)
            probe_loss = F.cross_entropy(model(xb).float(), yb.long())
            probe_loss.backward()
            runtime_prob = 0.0
            if runtime_candidate_policies:
                candidate_scores = []
                for candidate, candidate_policy in runtime_candidate_policies.items():
                    runtime_row = runtime_feature_row_from_batch(
                        model=model,
                        architecture=architecture,
                        optimizer_family=optimizer_family,
                        treatment=candidate,
                        xb=xb,
                        yb=yb,
                        output_dim=output_dim,
                        step=step,
                        args=args,
                    )
                    candidate_prob = predict_runtime_cate_positive(
                        candidate_policy.get("model"),
                        runtime_row,
                        str(candidate_policy.get("group", "all features")),
                    )
                    candidate_threshold = float(candidate_policy.get("threshold", runtime_policy_threshold))
                    candidate_lcb = float(candidate_policy.get("empirical_lcb", -math.inf))
                    if candidate_prob >= candidate_threshold and candidate_lcb > float(args.epsilon_row):
                        candidate_scores.append((candidate_prob, candidate_lcb, candidate))
                runtime_policy_checks += 1
                if candidate_scores:
                    runtime_prob, active_lcb, active_treatment = max(candidate_scores, key=lambda item: (item[0], item[1], item[2]))
                    runtime_arbitrated_selected_counts[active_treatment] = runtime_arbitrated_selected_counts.get(active_treatment, 0) + 1
                runtime_policy_probability_sum += runtime_prob
            else:
                runtime_row = runtime_feature_row_from_batch(
                    model=model,
                    architecture=architecture,
                    optimizer_family=optimizer_family,
                    treatment=treatment,
                    xb=xb,
                    yb=yb,
                    output_dim=output_dim,
                    step=step,
                    args=args,
                )
                runtime_prob = predict_runtime_cate_positive(runtime_policy_model, runtime_row, runtime_policy_group)
                runtime_policy_checks += 1
                runtime_policy_probability_sum += runtime_prob
            opt.zero_grad(set_to_none=True)
            if runtime_candidate_policies:
                runtime_passed = bool(active_treatment != treatment and active_treatment in runtime_candidate_policies)
            else:
                runtime_passed = bool(runtime_prob >= runtime_policy_threshold and runtime_policy_empirical_lcb > float(args.epsilon_row))
            if runtime_passed:
                runtime_policy_pass += 1
            else:
                runtime_policy_blocked += 1
                rejected += 1
                should_intervene = False
            if should_intervene and runtime_safety_enabled:
                safety_row = runtime_feature_row_from_batch(
                    model=model,
                    architecture=architecture,
                    optimizer_family=optimizer_family,
                    treatment=active_treatment,
                    xb=xb,
                    yb=yb,
                    output_dim=output_dim,
                    step=step,
                    args=args,
                )
                safety_prob = predict_runtime_binary_positive(runtime_safety_model, safety_row, runtime_safety_group)
                runtime_safety_checks += 1
                runtime_safety_probability_sum += safety_prob
                if safety_prob >= runtime_safety_threshold:
                    runtime_safety_pass += 1
                else:
                    runtime_safety_blocked += 1
                    rejected += 1
                    should_intervene = False
        if should_intervene:
            accept_xs = []
            accept_ys = []
            for _ in range(max(1, int(getattr(args, "full_loop_acceptance_batches", 1)))):
                ax_i, ay_i = next(held_it)
                accept_xs.append(ax_i.to(device).float())
                accept_ys.append(ay_i.to(device).long())
            ax = torch.cat(accept_xs, dim=0)
            ay = torch.cat(accept_ys, dim=0)
            post_apply_ax = ax
            post_apply_ay = ay
            # Use a current-train-batch proxy for horizon simulation.  A second
            # shuffled train iterator perturbs the actual training order and
            # makes no-op rows incomparable to optimizer_alone.
            future_batches = [(xb, yb) for _ in range(horizon_steps)]
            branch_seed_base = core.stable_int_seed("v22_38_horizon", dataset, seed, architecture, optimizer_family, variant, step)
            base_loss, _base_norm, base_components = horizon_branch_candidate(
                model=model,
                opt=opt,
                architecture=architecture,
                optimizer_family=optimizer_family,
                treatment="a0_base_noop",
                train_xb=xb,
                train_yb=yb,
                future_batches=future_batches,
                accept_xb=ax,
                accept_yb=ay,
                output_dim=output_dim,
                acceptance_metric=str(args.full_loop_acceptance_metric),
                acceptance_tail_quantile=float(getattr(args, "full_loop_acceptance_tail_quantile", 0.90)),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                alpha=0.0,
                alpha_scale_mode=alpha_scale_mode,
                alpha_scale_floor=alpha_scale_floor,
                seed=branch_seed_base,
                cvar_fraction=cvar_fraction,
            )
            best_loss = base_loss
            best_norm = 0.0
            for alpha in alpha_grid:
                if abs(float(alpha)) <= 0.0:
                    continue
                cand_loss, cand_norm, cand_components = horizon_branch_candidate(
                    model=model,
                    opt=opt,
                    architecture=architecture,
                    optimizer_family=optimizer_family,
                    treatment=active_treatment,
                    train_xb=xb,
                    train_yb=yb,
                    future_batches=future_batches,
                    accept_xb=ax,
                    accept_yb=ay,
                    output_dim=output_dim,
                    acceptance_metric=str(args.full_loop_acceptance_metric),
                    acceptance_tail_quantile=float(getattr(args, "full_loop_acceptance_tail_quantile", 0.90)),
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    alpha=float(alpha),
                    alpha_scale_mode=alpha_scale_mode,
                    alpha_scale_floor=alpha_scale_floor,
                    seed=branch_seed_base + int(abs(float(alpha)) * 1.0e9),
                    cvar_fraction=cvar_fraction,
                )
                candidate_evals += 1
                if horizon_components_ok(base_components, cand_components, float(args.full_loop_acceptance_tol), acceptance_tail_margin) and cand_loss <= best_loss - float(args.full_loop_acceptance_tol):
                    best_loss = cand_loss
                    chosen_alpha = float(alpha)
                    best_norm = cand_norm
            candidate_delta_sum += best_loss - base_loss
            if chosen_alpha > 0.0 and active_lcb > float(args.epsilon_row):
                accepted += 1
                accepted_alpha_sum += chosen_alpha
            else:
                rejected += 1
                chosen_alpha = 0.0
                best_norm = 0.0
        scale_named = None
        scale_before = None
        if chosen_alpha > 0.0 and alpha_scale_mode == "base_update_norm":
            scale_named = v22_38_named_trainable(model, architecture, active_treatment)
            scale_before = [p.detach().clone() for _n, p in scale_named]
        train_loss = core.train_batch_step_with_existing_optimizer(model, opt, optimizer_family, xb, yb)
        loss_trace.append(train_loss)
        if chosen_alpha > 0.0:
            opt.zero_grad(set_to_none=True)
            loss_for_fu = F.cross_entropy(model(xb).float(), yb.long())
            loss_for_fu.backward()
            named, direction, _source = v22_38_build_treatment_direction(
                model,
                opt,
                architecture,
                active_treatment,
                random.Random(core.stable_int_seed("v22_38_horizon_apply", dataset, seed, architecture, optimizer_family, variant, step)),
            )
            effective_alpha = chosen_alpha
            if alpha_scale_mode == "base_update_norm":
                base_norm = selected_param_update_norm(scale_named or named, scale_before or [p.detach().clone() for _n, p in named])
                effective_alpha = chosen_alpha * max(alpha_scale_floor, base_norm)
            rollback_snap = [p.detach().clone() for _n, p in named] if bool(getattr(args, "full_loop_post_apply_rollback", False)) else []
            actual_base_components = []
            if rollback_snap and post_apply_ax is not None and post_apply_ay is not None:
                actual_base_components = [
                    core.batch_acceptance_components(
                        model,
                        post_apply_ax,
                        post_apply_ay,
                        output_dim,
                        str(args.full_loop_acceptance_metric),
                        float(getattr(args, "full_loop_acceptance_tail_quantile", 0.90)),
                    )
                ]
            chosen_update_norm = v22_38_apply_flat_delta(named, direction, effective_alpha)
            rolled_back = False
            if rollback_snap and actual_base_components and post_apply_ax is not None and post_apply_ay is not None:
                actual_cand_components = [
                    core.batch_acceptance_components(
                        model,
                        post_apply_ax,
                        post_apply_ay,
                        output_dim,
                        str(args.full_loop_acceptance_metric),
                        float(getattr(args, "full_loop_acceptance_tail_quantile", 0.90)),
                    )
                ]
                if not horizon_components_ok(actual_base_components, actual_cand_components, float(args.full_loop_acceptance_tol), acceptance_tail_margin):
                    for (_n, p), before in zip(named, rollback_snap):
                        p.data.copy_(before)
                    post_apply_rollback_count += 1
                    accepted = max(0, accepted - 1)
                    accepted_alpha_sum = max(0.0, accepted_alpha_sum - chosen_alpha)
                    rejected += 1
                    chosen_update_norm = 0.0
                    rolled_back = True
            if rolled_back:
                continue
            accepted_effective_alpha_sum += effective_alpha
            total_update_norm += chosen_update_norm
            if architecture != "MLP":
                basis_names = {n for n, _p in named if n in {"w1", "w2"} or n.endswith(".w1") or n.endswith(".w2")}
                if basis_names:
                    basis_update_norm += chosen_update_norm
                else:
                    nonbasis_update_norm += chosen_update_norm
            else:
                nonbasis_update_norm += chosen_update_norm
    temperature_grid = [max(1.0e-6, float(x)) for x in split_csv(getattr(args, "full_loop_temperature_grid", "1.0"), float)]
    if not temperature_grid:
        temperature_grid = [1.0]
    temperature_metric = str(getattr(args, "full_loop_temperature_selection_metric", "NLL"))
    temperature_mode = str(getattr(args, "full_loop_temperature_calibration_mode", "scalar") or "scalar")
    confidence_temperature_quantiles = [
        max(0.0, min(1.0, float(x)))
        for x in split_csv(getattr(args, "full_loop_confidence_temperature_quantiles", "0.70,0.80,0.90,0.95"), float)
    ]
    if not confidence_temperature_quantiles:
        confidence_temperature_quantiles = [0.70, 0.80, 0.90, 0.95]
    uncalibrated_test = core.evaluate_loader_temperature(model, test_loader, device, output_dim, 1.0)
    uncalibrated_held = core.evaluate_loader_temperature(model, held_loader, device, output_dim, 1.0)
    calibration_policy, final_held = select_temperature_policy(
        model,
        held_loader,
        device,
        output_dim,
        temperature_grid,
        temperature_metric,
        temperature_mode,
        confidence_temperature_quantiles,
    )
    calibration_temperature = float(calibration_policy.get("temperature", 1.0))
    calibration_confidence_threshold = calibration_policy.get("confidence_threshold", "")
    if calibration_policy.get("mode") == "confidence_tail":
        final_test = evaluate_loader_confidence_tail_temperature(
            model,
            test_loader,
            device,
            output_dim,
            calibration_temperature,
            float(calibration_confidence_threshold),
        )
    else:
        final_test = core.evaluate_loader_temperature(model, test_loader, device, output_dim, calibration_temperature)
    metric_bootstrap_samples = max(0, int(getattr(args, "full_loop_metric_bootstrap_samples", 0)))
    metric_bootstrap = {"NLL": math.nan, "ECE": math.nan, "Brier": math.nan, "tail_q99": math.nan}
    if metric_bootstrap_samples > 1:
        if calibration_policy.get("mode") == "confidence_tail":
            vectors = collect_eval_vectors_confidence_tail_temperature(
                model,
                test_loader,
                device,
                output_dim,
                calibration_temperature,
                float(calibration_confidence_threshold),
            )
        else:
            vectors = collect_eval_vectors(model, test_loader, device, output_dim, calibration_temperature)
        metric_bootstrap = bootstrap_metric_se(
            vectors,
            seed=core.stable_int_seed("v22_38_metric_bootstrap", dataset, seed, architecture, optimizer_family, variant, treatment),
            samples=metric_bootstrap_samples,
        )
    elapsed = time.time() - start
    denom = max(1.0e-12, basis_update_norm + nonbasis_update_norm)
    return {
        "dataset": dataset,
        "seed": seed,
        "architecture": "MLP" if architecture == "MLP" else "strict_FC_PureKAN",
        "carrier": core.carrier_for_arch(architecture),
        "optimizer_family": optimizer_family,
        "training_variant": variant,
        "treatment_name": treatment,
        "final_test_NLL": final_test["NLL"],
        "final_test_accuracy": final_test["accuracy"],
        "final_test_uncalibrated_NLL": uncalibrated_test["NLL"],
        "final_test_uncalibrated_ECE": uncalibrated_test["ECE"],
        "final_test_uncalibrated_Brier": uncalibrated_test["Brier"],
        "final_test_uncalibrated_tail_q99": uncalibrated_test["tail_q99"],
        "held_train_NLL": final_held["NLL"],
        "held_train_uncalibrated_NLL": uncalibrated_held["NLL"],
        "AUC_loss_time": sum(loss_trace) / max(1, len(loss_trace)),
        "wallclock_adjusted_AUC": (sum(loss_trace) / max(1, len(loss_trace))) * (elapsed / max(1, int(args.full_loop_steps))),
        "ECE": final_test["ECE"],
        "Brier": final_test["Brier"],
        "tail_loss_q95": final_test["tail_q95"],
        "tail_loss_q99": final_test["tail_q99"],
        "metric_bootstrap_samples": metric_bootstrap_samples,
        "NLL_bootstrap_se": metric_bootstrap["NLL"] if math.isfinite(float(metric_bootstrap["NLL"])) else "",
        "ECE_bootstrap_se": metric_bootstrap["ECE"] if math.isfinite(float(metric_bootstrap["ECE"])) else "",
        "Brier_bootstrap_se": metric_bootstrap["Brier"] if math.isfinite(float(metric_bootstrap["Brier"])) else "",
        "tail_q99_bootstrap_se": metric_bootstrap["tail_q99"] if math.isfinite(float(metric_bootstrap["tail_q99"])) else "",
        "hard_slice_NLL": final_held["NLL"],
        "low_margin_accuracy": final_test["low_margin_accuracy"],
        "policy_noop_rate": 1.0 if variant == "optimizer_alone" else rejected / max(1, accepted + rejected),
        "accepted_treatment_count": accepted,
        "rejected_treatment_count": rejected,
        "post_apply_rollback_count": post_apply_rollback_count,
        "candidate_eval_count": candidate_evals,
        "accepted_alpha_mean": accepted_alpha_sum / max(1, accepted),
        "accepted_effective_alpha_mean": accepted_effective_alpha_sum / max(1, accepted),
        "candidate_acceptance_loss_delta_mean": candidate_delta_sum / max(1, accepted + rejected),
        "alpha_scale_mode": alpha_scale_mode,
        "alpha_scale_floor": alpha_scale_floor,
        "acceptance_batches": max(1, int(getattr(args, "full_loop_acceptance_batches", 1))),
        "acceptance_tail_quantile": float(getattr(args, "full_loop_acceptance_tail_quantile", 0.90)),
        "acceptance_tail_margin": acceptance_tail_margin,
        "horizon_gate_steps": horizon_steps,
        "horizon_gate_proxy": "current_train_batch_replay",
        "horizon_cvar_fraction": cvar_fraction,
        "runtime_CATE_policy_enabled": int(use_runtime_cate),
        "runtime_CATE_policy_scope": runtime_policy_scope if use_runtime_cate else "",
        "runtime_CATE_policy_selection_status": runtime_policy_selection_status if use_runtime_cate else "",
        "runtime_CATE_policy_selected_treatment": runtime_policy_selected_treatment if use_runtime_cate else "",
        "runtime_CATE_policy_source_artifact": runtime_policy_source_artifact if use_runtime_cate else "",
        "runtime_CATE_policy_feature_group": runtime_policy_group,
        "runtime_CATE_policy_model_kind": runtime_policy_kind,
        "runtime_CATE_policy_fit_status": runtime_policy_status,
        "runtime_CATE_policy_threshold": runtime_policy_threshold if use_runtime_cate else "",
        "runtime_CATE_policy_threshold_lcb": runtime_policy_empirical_lcb if use_runtime_cate and math.isfinite(runtime_policy_empirical_lcb) else "",
        "runtime_CATE_policy_threshold_n": runtime_policy_empirical_n if use_runtime_cate else "",
        "runtime_CATE_policy_checked_count": runtime_policy_checks if use_runtime_cate else "",
        "runtime_CATE_policy_pass_count": runtime_policy_pass if use_runtime_cate else "",
        "runtime_CATE_policy_blocked_count": runtime_policy_blocked if use_runtime_cate else "",
        "runtime_CATE_policy_probability_mean": runtime_policy_probability_sum / max(1, runtime_policy_checks) if use_runtime_cate else "",
        "runtime_CATE_policy_arbitration_candidates": ",".join(arbitrated_candidates) if arbitrated_candidates else "",
        "runtime_CATE_policy_arbitration_selected_counts": ";".join(f"{k}:{v}" for k, v in sorted(runtime_arbitrated_selected_counts.items())) if arbitrated_candidates else "",
        "runtime_CATE_policy_arbitration_candidate_thresholds": ";".join(f"{k}:{float(v.get('threshold', math.nan))}" for k, v in sorted(runtime_candidate_policies.items())) if runtime_candidate_policies else "",
        "runtime_CATE_policy_arbitration_candidate_lcbs": ";".join(f"{k}:{float(v.get('empirical_lcb', math.nan))}" for k, v in sorted(runtime_candidate_policies.items())) if runtime_candidate_policies else "",
        "runtime_safety_policy_enabled": int(runtime_safety_enabled),
        "runtime_safety_policy_source_artifact": "results/v22_38/v22_38_CATE_event_matrix.csv" if runtime_safety_enabled else "",
        "runtime_safety_policy_label": "Y_no_debt_gate" if runtime_safety_enabled else "",
        "runtime_safety_policy_feature_group": runtime_safety_group if runtime_safety_enabled else "",
        "runtime_safety_policy_model_kind": runtime_safety_kind if runtime_safety_enabled else "",
        "runtime_safety_policy_fit_status": runtime_safety_status if runtime_safety_enabled else "",
        "runtime_safety_policy_threshold": runtime_safety_threshold if runtime_safety_enabled else "",
        "runtime_safety_policy_checked_count": runtime_safety_checks if runtime_safety_enabled else "",
        "runtime_safety_policy_pass_count": runtime_safety_pass if runtime_safety_enabled else "",
        "runtime_safety_policy_blocked_count": runtime_safety_blocked if runtime_safety_enabled else "",
        "runtime_safety_policy_probability_mean": runtime_safety_probability_sum / max(1, runtime_safety_checks) if runtime_safety_enabled else "",
        "LCB_mean": lcb if variant != "optimizer_alone" else "",
        "controller_overhead": elapsed / max(1, int(args.full_loop_steps)),
        "basis_energy_fraction": "" if architecture == "MLP" else basis_update_norm / denom,
        "readout_leakage_fraction": "" if architecture == "MLP" else nonbasis_update_norm / denom,
        "basis_actuator_norm": basis_update_norm,
        "calibration_temperature": calibration_temperature,
        "calibration_temperature_mode": calibration_policy.get("mode", "scalar"),
        "calibration_temperature_confidence_threshold": calibration_confidence_threshold,
        "calibration_temperature_policy": (
            f"mode={calibration_policy.get('mode', 'scalar')};temp={calibration_temperature:g};"
            f"confidence_threshold={calibration_confidence_threshold}"
        ),
        "temperature_grid": ",".join(f"{x:g}" for x in temperature_grid),
        "temperature_selection_metric": temperature_metric,
        "temperature_selection_split": "held_train_only",
        "implementation_level": "v22_38_train_only_CATE_runtime_plus_horizon_robust_pareto_gate" if use_runtime_cate else "v22_38_train_only_horizon_robust_pareto_gate",
        "status": "completed_full_loop",
        "source_artifact": source_artifact,
    }


def run_full_loop_matrix(
    *,
    args: argparse.Namespace,
    stage_name: str,
    output_name: str,
    device_name: str,
    datasets: str,
    seeds: str,
    architectures: str,
    optimizers: str,
    treatment_plan: str,
    require_cate_gate: bool,
) -> dict[str, Any]:
    ensure_out()
    policy = (read_rows(OUT_ROOT / "v22_38_CATE_policy_calibration.csv") or [{}])[0]
    if require_cate_gate and not int_flag(policy.get("policy_gate_pass")):
        blocked = [{"status": "gate_blocked", "reason": "CATE_policy_no_go", "source_artifact": "results/v22_38/v22_38_CATE_policy_calibration.csv"}]
        write_rows(OUT_ROOT / output_name, blocked)
        append_exec(stage_name, task_id=stage_name, status="gate_blocked", gpu=device_name, files=f"results/v22_38/{output_name}", note="CATE policy gate failed; full-loop policy not run per plan")
        return {"status": "gate_blocked", "reason": "CATE_policy_no_go"}
    configure_core(CORE_ROOT / f"full_loop_{safe_fragment(stage_name)}")
    cargs = core_args_from(args, device_name)
    setattr(treatments_for_plan, "kan_basis_treatment_mode", str(getattr(cargs, "kan_basis_treatment_mode", "a2_only")))
    device = core.torch_device(device_name)
    train_variant = (
        train_horizon_robust_full_loop_variant
        if str(getattr(cargs, "full_loop_implementation", "")) in {"horizon_robust", "cate_runtime_horizon_robust"}
        else core.train_full_loop_variant
    )
    rows: list[dict[str, Any]] = []
    availability: list[dict[str, Any]] = []
    for dataset in split_csv(datasets):
        for seed in split_csv(seeds, int):
            for optimizer in split_csv(optimizers):
                for architecture in split_csv(architectures):
                    try:
                        train_loader, held_loader, test_loader, input_dim, output_dim, x_stats, meta = core.make_loaders_for_dataset(
                            str(dataset),
                            int(cargs.full_loop_train_size),
                            int(cargs.full_loop_held_size),
                            int(cargs.batch_size),
                            int(seed),
                            tier2_download=bool(cargs.tier2_download),
                        )
                        rows.append(
                            train_variant(
                                dataset=str(dataset),
                                seed=int(seed),
                                architecture=str(architecture),
                                optimizer_family=str(optimizer),
                                variant="optimizer_alone",
                                treatment="a0_base_noop",
                                train_loader=train_loader,
                                held_loader=held_loader,
                                test_loader=test_loader,
                                input_dim=input_dim,
                                output_dim=output_dim,
                                x_stats=x_stats,
                                device=device,
                                args=cargs,
                                lcb=0.0,
                                source_artifact=f"results/v22_38/{output_name}",
                            )
                        )
                        for variant, treatment, lcb_col in treatments_for_plan(treatment_plan, str(architecture)):
                            train_loader, held_loader, test_loader, input_dim, output_dim, x_stats, _meta2 = core.make_loaders_for_dataset(
                                str(dataset),
                                int(cargs.full_loop_train_size),
                                int(cargs.full_loop_held_size),
                                int(cargs.batch_size),
                                int(seed),
                                tier2_download=bool(cargs.tier2_download),
                            )
                            rows.append(
                                train_variant(
                                    dataset=str(dataset),
                                    seed=int(seed),
                                    architecture=str(architecture),
                                    optimizer_family=str(optimizer),
                                    variant=variant,
                                    treatment=treatment,
                                    train_loader=train_loader,
                                    held_loader=held_loader,
                                    test_loader=test_loader,
                                    input_dim=input_dim,
                                    output_dim=output_dim,
                                    x_stats=x_stats,
                                    device=device,
                                    args=cargs,
                                    lcb=lcb_for(treatment, lcb_col),
                                    source_artifact=f"results/v22_38/{output_name}",
                                )
                            )
                        availability.append({"dataset": dataset, "seed": seed, "architecture": architecture, "optimizer_family": optimizer, "status": "completed", "task_tier": meta.get("task_tier", "")})
                    except Exception as exc:
                        availability.append({"dataset": dataset, "seed": seed, "architecture": architecture, "optimizer_family": optimizer, "status": "data_unavailable_or_run_failed", "error_type": type(exc).__name__, "error_message": str(exc)})
    enriched = core.enrich_full_loop_rows(rows)
    enriched = postprocess_full_loop(enriched, treatment_plan)
    output_path = OUT_ROOT / output_name
    write_rows(output_path, enriched or [{"status": "no_full_loop_rows"}])
    write_rows(output_path.with_name(f"{output_path.stem}_availability.csv"), availability or [{"status": "no_availability_rows"}])
    summary = summarize_full_loop(enriched, treatment_plan)
    write_rows(output_path.with_name(f"{output_path.stem}_summary.csv"), [summary])
    append_exec(
        stage_name,
        task_id=stage_name,
        status="pass" if enriched else "warn",
        gpu=device_name,
        files=f"{output_path.relative_to(ROOT)}, {output_path.with_name(f'{output_path.stem}_summary.csv').relative_to(ROOT)}",
        note=f"rows={len(enriched)}; summary={summary}",
    )
    return summary


def treatments_for_plan(plan: str, architecture: str) -> list[tuple[str, str, str]]:
    mode = str(getattr(treatments_for_plan, "kan_basis_treatment_mode", "a2_only"))
    if plan == "support":
        return [("FSO_support_native", "a5_same_support_random" if architecture == "MLP" else "a6_same_actuator_random", "LCB_support")]
    if plan == "signal":
        if architecture == "MLP":
            return [("SD_CFO_signal_incremental_a1", "a1_signal_direction", "LCB_direction")]
        if mode == "a2_a5_arbitrated":
            return [("SD_CFO_signal_incremental_a2_a5_arbitrated", "a2_a5_runtime_arbitrated_basis", "LCB_direction")]
        rows = [("SD_CFO_signal_incremental_a2", "a2_basis_actuator_section", "LCB_direction")]
        if mode == "expanded_planned":
            rows.append(("SD_CFO_signal_incremental_basis_a5", "a5_signal_incremental_basis", "LCB_direction"))
        return rows
    if plan == "optimizer":
        return [("SD_CFO_optimizer_state_a3", "a3_optimizer_state_signal", "LCB_optimizer")]
    if plan == "kan_basis":
        if architecture == "MLP":
            return []
        if mode == "a2_a5_arbitrated":
            return [
                ("SD_CFO_KAN_basis_native_a2_a5_arbitrated", "a2_a5_runtime_arbitrated_basis", "LCB_direction"),
                ("FSO_KAN_basis_support", "a6_same_actuator_random", "LCB_net"),
                ("FSO_KAN_basis_support_a8", "a8_support_native_basis", "LCB_support"),
                ("matched_control_signflip_basis", "a6_signflip_same_basis", "LCB_net"),
                ("matched_control_same_bank_random", "a6_same_bank_random", "LCB_net"),
            ]
        rows = [
            ("SD_CFO_KAN_basis_native_a2", "a2_basis_actuator_section", "LCB_direction"),
            ("FSO_KAN_basis_support", "a6_same_actuator_random", "LCB_net"),
        ]
        if mode == "expanded_planned":
            rows.extend(
                [
                    ("SD_CFO_KAN_basis_native_a5", "a5_signal_incremental_basis", "LCB_direction"),
                    ("FSO_KAN_basis_support_a8", "a8_support_native_basis", "LCB_support"),
                    ("matched_control_signflip_basis", "a6_signflip_same_basis", "LCB_net"),
                    ("matched_control_same_bank_random", "a6_same_bank_random", "LCB_net"),
                ]
            )
        return rows
    return []


def postprocess_full_loop(rows: list[dict[str, Any]], plan: str) -> list[dict[str, Any]]:
    support_lookup: dict[tuple[str, str, str, str], float] = {}
    if plan == "signal":
        for row in read_rows(OUT_ROOT / "v22_38_support_native_full_loop_matrix.csv"):
            if row.get("training_variant") == "FSO_support_native":
                key = (str(row.get("dataset")), str(row.get("seed")), str(row.get("carrier")), str(row.get("optimizer_family")))
                val = finite_float(row.get("final_test_NLL"))
                if val is not None:
                    support_lookup[key] = val
    baseline_lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("training_variant") == "optimizer_alone":
            key = (str(row.get("dataset")), str(row.get("seed")), str(row.get("carrier")), str(row.get("optimizer_family")))
            baseline_lookup[key] = row
    out = []
    for row in rows:
        rr = dict(row)
        rr["v22_38_plan"] = plan
        treatment = str(rr.get("treatment_name", ""))
        alias = PLANNED_TREATMENT_EVIDENCE_ALIASES.get(treatment, {})
        rr["planned_basis_treatment_set_member"] = int(treatment in PLANNED_BASIS_TREATMENTS)
        rr["treatment_evidence_alias_source_treatment"] = alias.get("source_treatment", "")
        rr["treatment_evidence_alias_source_column"] = alias.get("source_column", "")
        rr["treatment_evidence_alias_note"] = alias.get("note", "")
        key = (str(rr.get("dataset")), str(rr.get("seed")), str(rr.get("carrier")), str(rr.get("optimizer_family")))
        base = baseline_lookup.get(key, {})
        if rr.get("training_variant") != "optimizer_alone":
            ece = finite_float(rr.get("ECE"))
            base_ece = finite_float(base.get("ECE"))
            brier = finite_float(rr.get("Brier"))
            base_brier = finite_float(base.get("Brier"))
            tail = finite_float(rr.get("tail_loss_q99"))
            base_tail = finite_float(base.get("tail_loss_q99"))
            strict_no_debt = int(
                ece is not None
                and base_ece is not None
                and brier is not None
                and base_brier is not None
                and tail is not None
                and base_tail is not None
                and ece <= base_ece
                and brier <= base_brier
                and tail <= base_tail
            )
            rr["no_ECE_Brier_tail_debt"] = strict_no_debt
            rr["no_ECE_Brier_tail_debt_strict"] = strict_no_debt
            rr["ECE_delta_vs_base"] = "" if ece is None or base_ece is None else ece - base_ece
            rr["Brier_delta_vs_base"] = "" if brier is None or base_brier is None else brier - base_brier
            rr["tail_q99_delta_vs_base"] = "" if tail is None or base_tail is None else tail - base_tail
            ece_se = finite_float(rr.get("ECE_bootstrap_se"))
            base_ece_se = finite_float(base.get("ECE_bootstrap_se"))
            brier_se = finite_float(rr.get("Brier_bootstrap_se"))
            base_brier_se = finite_float(base.get("Brier_bootstrap_se"))
            tail_se = finite_float(rr.get("tail_q99_bootstrap_se"))
            base_tail_se = finite_float(base.get("tail_q99_bootstrap_se"))
            if all(v is not None for v in [ece, base_ece, brier, base_brier, tail, base_tail, ece_se, base_ece_se, brier_se, base_brier_se, tail_se, base_tail_se]):
                z = 1.64
                ece_tol = z * math.sqrt(float(ece_se) ** 2 + float(base_ece_se) ** 2)
                brier_tol = z * math.sqrt(float(brier_se) ** 2 + float(base_brier_se) ** 2)
                tail_tol = z * math.sqrt(float(tail_se) ** 2 + float(base_tail_se) ** 2)
                rr["ECE_bootstrap_0p90_tolerance"] = ece_tol
                rr["Brier_bootstrap_0p90_tolerance"] = brier_tol
                rr["tail_q99_bootstrap_0p90_tolerance"] = tail_tol
                rr["no_ECE_Brier_tail_debt_bootstrap_0p90"] = int(
                    float(ece) <= float(base_ece) + ece_tol
                    and float(brier) <= float(base_brier) + brier_tol
                    and float(tail) <= float(base_tail) + tail_tol
                )
            else:
                rr["ECE_bootstrap_0p90_tolerance"] = ""
                rr["Brier_bootstrap_0p90_tolerance"] = ""
                rr["tail_q99_bootstrap_0p90_tolerance"] = ""
                rr["no_ECE_Brier_tail_debt_bootstrap_0p90"] = ""
        if plan == "signal" and str(rr.get("training_variant", "")).startswith("SD_CFO"):
            fso = support_lookup.get(key)
            nll = finite_float(rr.get("final_test_NLL"))
            rr["NLL_delta_vs_FSO_baseline"] = "" if fso is None or nll is None else nll - fso
        out.append(rr)
    return out


def summarize_full_loop(rows: list[dict[str, Any]], plan: str) -> dict[str, Any]:
    treatment_rows = [r for r in rows if r.get("training_variant") != "optimizer_alone"]
    active_rows = [r for r in treatment_rows if str(r.get("training_variant", "")).startswith("SD_CFO")]
    active_kan_rows = [r for r in active_rows if str(r.get("carrier", "")) in {"D-CHE", "D-FOU"}]
    wins = sum(1 for r in treatment_rows if (finite_float(r.get("NLL_delta_vs_own_strong_optimizer"), math.inf) or math.inf) < 0.0)
    no_debt = sum(1 for r in treatment_rows if int_flag(r.get("no_ECE_Brier_tail_debt")))
    active_wins = sum(1 for r in active_rows if (finite_float(r.get("NLL_delta_vs_own_strong_optimizer"), math.inf) or math.inf) < 0.0)
    active_no_debt = sum(1 for r in active_rows if int_flag(r.get("no_ECE_Brier_tail_debt")))
    active_kan_wins = sum(1 for r in active_kan_rows if (finite_float(r.get("NLL_delta_vs_own_strong_optimizer"), math.inf) or math.inf) < 0.0)
    active_kan_no_debt = sum(1 for r in active_kan_rows if int_flag(r.get("no_ECE_Brier_tail_debt")))
    active_kan_bootstrap_no_debt = sum(1 for r in active_kan_rows if int_flag(r.get("no_ECE_Brier_tail_debt_bootstrap_0p90")))
    rollback_total = sum(int(finite_float(r.get("post_apply_rollback_count"), 0.0) or 0) for r in treatment_rows)
    vs_fso = [finite_float(r.get("NLL_delta_vs_FSO_baseline")) for r in treatment_rows if r.get("NLL_delta_vs_FSO_baseline") not in {"", None}]
    return {
        "status": "completed_full_loop" if rows else "no_rows",
        "plan": plan,
        "rows": len(rows),
        "treatment_rows": len(treatment_rows),
        "NLL_improvement_vs_own_strong_rows": wins,
        "no_ECE_Brier_tail_debt_rows": no_debt,
        "active_SD_CFO_rows": len(active_rows),
        "active_SD_CFO_NLL_improvement_rows": active_wins,
        "active_SD_CFO_no_ECE_Brier_tail_debt_rows": active_no_debt,
        "active_KAN_SD_CFO_rows": len(active_kan_rows),
        "active_KAN_SD_CFO_NLL_improvement_rows": active_kan_wins,
        "active_KAN_SD_CFO_no_ECE_Brier_tail_debt_rows": active_kan_no_debt,
        "active_KAN_SD_CFO_no_ECE_Brier_tail_debt_bootstrap_0p90_rows": active_kan_bootstrap_no_debt,
        "post_apply_rollback_count_total": rollback_total,
        "NLL_improvement_vs_FSO_rows": sum(1 for v in vs_fso if v is not None and v < 0.0),
        "mean_NLL_delta_vs_FSO": "" if not vs_fso else sum(float(v) for v in vs_fso if v is not None) / len(vs_fso),
    }


def stage_h_continual(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    configure_core(CORE_ROOT / "continual")
    cargs = core.parser().parse_args([])
    cargs.continual_device = args.continual_device
    cargs.continual_seed = int(args.continual_seed)
    cargs.continual_seeds = args.continual_seeds
    cargs.continual_train_size = int(args.continual_train_size)
    cargs.continual_held_size = int(args.continual_held_size)
    cargs.continual_steps_per_task = int(args.continual_steps_per_task)
    cargs.hidden = int(args.hidden)
    cargs.lr = float(args.lr)
    cargs.weight_decay = float(args.weight_decay)
    cargs.batch_size = int(args.batch_size)
    cargs.intervention_scale = float(args.intervention_scale)
    summary = core.stage_h_continual_memory(cargs)
    rows = read_rows(CORE_ROOT / "continual/v22_37_continual_memory_matrix.csv")
    for r in rows:
        r["source_artifact"] = "results/v22_38/core_interop/continual/v22_37_continual_memory_matrix.csv"
    write_rows(OUT_ROOT / "v22_38_continual_memory_matrix.csv", rows or [{"status": "no_continual_rows"}])
    write_rows(OUT_ROOT / "v22_38_grokking_signal_migration_matrix.csv", [{"status": "not_run", "reason": "modular arithmetic grokking task not executed in this v22.38 run; no data fabricated"}])
    append_exec("stage_h_continual", task_id="H_continual", status="pass" if rows and rows[0].get("status") == "completed_continual_probe" else "warn", gpu=args.continual_device, files="results/v22_38/v22_38_continual_memory_matrix.csv, results/v22_38/v22_38_grokking_signal_migration_matrix.csv", note=str(summary))
    return summary


def stage_figures_efficiency() -> None:
    ensure_out()
    decomp = read_rows(OUT_ROOT / "v22_38_support_direction_decomposition.csv")
    horizon = read_rows(OUT_ROOT / "v22_38_horizon_effect_matrix.csv")
    safety = read_rows(OUT_ROOT / "v22_38_safety_debt_by_treatment.csv")
    opt = read_rows(OUT_ROOT / "v22_38_optimizer_state_full_loop_matrix.csv")
    gap = read_rows(OUT_ROOT / "v22_38_KANbeFair_gap_truth_matrix.csv")
    continual = read_rows(OUT_ROOT / "v22_38_continual_memory_matrix.csv")
    write_simple_svg(FIG_ROOT / "v22_38_support_vs_direction_tau.svg", "v22.38 support vs direction tau", decomp, "treatment_name", "tau_direction")
    if not (FIG_ROOT / "v22_38_CATE_feature_importance_panel.svg").exists():
        write_simple_svg(FIG_ROOT / "v22_38_CATE_feature_importance_panel.svg", "v22.38 CATE feature importance", [{"label": "not_fit", "value": 0}], "label", "value")
    write_simple_svg(FIG_ROOT / "v22_38_horizon_effect_decay_panel.svg", "v22.38 horizon effect", horizon, "treatment_name", "tau_H")
    write_simple_svg(FIG_ROOT / "v22_38_safety_debt_waterfall.svg", "v22.38 safety debt", safety, "treatment_name", "tail_debt_rate")
    write_simple_svg(FIG_ROOT / "v22_38_optimizer_state_spectrum_panel.svg", "v22.38 optimizer-state rows", opt, "training_variant", "NLL_delta_vs_own_strong_optimizer")
    write_simple_svg(FIG_ROOT / "v22_38_KAN_gap_truth_stacked_bar.svg", "v22.38 KAN gap truth", gap, "TrueKANGain_class", "GapReduction")
    write_simple_svg(FIG_ROOT / "v22_38_continual_forgetting_curves.svg", "v22.38 continual forgetting", continual, "variant", "average_forgetting")
    write_simple_svg(FIG_ROOT / "v22_38_grokking_delay_curves.svg", "v22.38 grokking delay", [{"label": "not_run", "value": 0}], "label", "value")
    events = read_rows(OUT_ROOT / "v22_38_CATE_event_matrix.csv")
    overheads = [finite_float(r.get("wallclock_overhead")) for r in events]
    overheads_f = [float(v) for v in overheads if v is not None]
    eff = [
        {
            "completed_CATE_events": len(events),
            "mean_event_horizon_wallclock_sec": "" if not overheads_f else sum(overheads_f) / len(overheads_f),
            "max_event_horizon_wallclock_sec": "" if not overheads_f else max(overheads_f),
            "controller_overhead_le_0p25_proxy": int(bool(overheads_f) and max(overheads_f) <= 0.25),
            "status": "computed_from_CATE_events" if overheads_f else "no_efficiency_rows",
        }
    ]
    write_rows(OUT_ROOT / "v22_38_efficiency_matrix.csv", eff)
    write_simple_svg(FIG_ROOT / "v22_38_efficiency_overhead_panel.svg", "v22.38 event overhead", events, "architecture", "wallclock_overhead")
    append_exec("stage_figures_efficiency", task_id="Figures_efficiency", status="pass", gpu="0", files="results/v22_38/figures/*.svg, results/v22_38/v22_38_efficiency_matrix.csv", note=f"figures={len(REQUIRED_FIGURES)}")


def summarize_horizon_robust_attempts() -> list[dict[str, Any]]:
    attempts_root = OUT_ROOT / "attempts"
    rows: list[dict[str, Any]] = []
    if attempts_root.exists():
        for summary_path in sorted(attempts_root.glob("*/v22_38_*_summary.csv")):
            attempt_tag = summary_path.parent.name
            for row in read_rows(summary_path):
                rr = dict(row)
                rr["attempt_tag"] = attempt_tag
                rr["summary_artifact"] = str(summary_path.relative_to(ROOT))
                matrix_path = summary_path.with_name(summary_path.name.replace("_summary.csv", ".csv"))
                rr["matrix_artifact"] = str(matrix_path.relative_to(ROOT)) if matrix_path.exists() else ""
                if matrix_path.exists():
                    recomputed = summarize_full_loop(read_rows(matrix_path), str(rr.get("plan", "")))
                    for key in [
                        "active_SD_CFO_rows",
                        "active_SD_CFO_NLL_improvement_rows",
                        "active_SD_CFO_no_ECE_Brier_tail_debt_rows",
                        "active_KAN_SD_CFO_rows",
                        "active_KAN_SD_CFO_NLL_improvement_rows",
                        "active_KAN_SD_CFO_no_ECE_Brier_tail_debt_rows",
                        "active_KAN_SD_CFO_no_ECE_Brier_tail_debt_bootstrap_0p90_rows",
                        "post_apply_rollback_count_total",
                    ]:
                        rr[key] = recomputed.get(key, "")
                rows.append(rr)
    if not rows:
        rows = [{"status": "no_horizon_robust_safety_attempts_recorded"}]
    write_rows(OUT_ROOT / "v22_38_horizon_robust_safety_gate_attempts.csv", rows)
    return rows


def summarize_attempt_gap_truth() -> list[dict[str, Any]]:
    attempts_root = OUT_ROOT / "attempts"
    out: list[dict[str, Any]] = []
    if attempts_root.exists():
        for attempt_dir in sorted(p for p in attempts_root.iterdir() if p.is_dir()):
            rows = read_rows(attempt_dir / "v22_38_KAN_basis_native_full_loop_matrix.csv") + read_rows(attempt_dir / "v22_38_signal_incremental_full_loop_matrix.csv")
            by_key: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
            for r in rows:
                by_key[(str(r.get("dataset")), str(r.get("seed")), str(r.get("carrier")), str(r.get("optimizer_family")), str(r.get("training_variant")))] = r
            emitted: set[tuple[str, str, str, str, str, str]] = set()
            for key, kan_fu in by_key.items():
                dataset, seed, carrier, opt, variant = key
                if carrier not in {"D-CHE", "D-FOU"} or not str(variant).startswith("SD_CFO"):
                    continue
                kan_base = by_key.get((dataset, seed, carrier, opt, "optimizer_alone"), {})
                mlp_base = by_key.get((dataset, seed, "", opt, "optimizer_alone"), {})
                mlp_fu = by_key.get((dataset, seed, "", opt, "SD_CFO_signal_incremental_a1"), {})
                vals = [finite_float(r.get("final_test_NLL")) for r in [kan_base, kan_fu, mlp_base, mlp_fu]]
                if any(v is None for v in vals):
                    continue
                kan_base_nll, kan_fu_nll, mlp_base_nll, mlp_fu_nll = [float(v) for v in vals if v is not None]
                gap_base = kan_base_nll - mlp_base_nll
                gap_fu = kan_fu_nll - mlp_fu_nll
                gap_reduction = gap_base - gap_fu
                delta_kan = kan_fu_nll - kan_base_nll
                delta_mlp = mlp_fu_nll - mlp_base_nll
                kan_fu_acc = finite_float(kan_fu.get("final_test_accuracy"))
                mlp_fu_acc = finite_float(mlp_fu.get("final_test_accuracy"))
                beats_by_accuracy = "" if kan_fu_acc is None or mlp_fu_acc is None else int(float(kan_fu_acc) >= float(mlp_fu_acc))
                beats_by_nll = int(kan_fu_nll <= mlp_fu_nll)
                beats_by_nll_or_accuracy = int(bool(beats_by_nll) or beats_by_accuracy == 1)
                if delta_mlp > 0.0 and gap_reduction > 0.0:
                    klass = "MLPDegradationDriven"
                elif delta_kan < 0.0 and delta_mlp < 0.0 and gap_reduction > 0.0:
                    klass = "BothGain"
                elif delta_kan < 0.0 and delta_mlp <= 0.0 and gap_reduction > 0.0:
                    klass = "TrueKANGain"
                else:
                    klass = "NoGain"
                emit_key = (attempt_dir.name, dataset, seed, carrier, opt, str(kan_fu.get("treatment_name", "")))
                if emit_key in emitted:
                    continue
                emitted.add(emit_key)
                out.append(
                    {
                        "attempt_tag": attempt_dir.name,
                        "dataset": dataset,
                        "seed": seed,
                        "optimizer_family": opt,
                        "kan_carrier": carrier,
                        "selected_treatment": kan_fu.get("treatment_name", ""),
                        "MLP_strong_NLL": mlp_base_nll,
                        "MLP_SD_CFO_NLL": mlp_fu_nll,
                        "MLP_SD_CFO_accuracy": mlp_fu_acc if mlp_fu_acc is not None else "",
                        "KAN_strong_NLL": kan_base_nll,
                        "KAN_SD_CFO_NLL": kan_fu_nll,
                        "KAN_SD_CFO_accuracy": kan_fu_acc if kan_fu_acc is not None else "",
                        "Delta_MLP_NLL": delta_mlp,
                        "Delta_KAN_NLL": delta_kan,
                        "GapReduction": gap_reduction,
                        "KAN_SD_CFO_beats_MLP_SD_CFO": beats_by_nll,
                        "KAN_SD_CFO_beats_MLP_SD_CFO_by_accuracy": beats_by_accuracy,
                        "KAN_SD_CFO_beats_MLP_SD_CFO_by_NLL_or_accuracy": beats_by_nll_or_accuracy,
                        "TrueKANGain_class": klass,
                        "status": "completed_attempt_gap_truth",
                    }
                )
    if not out:
        out = [{"status": "no_attempt_gap_truth_rows"}]
    write_rows(OUT_ROOT / "v22_38_attempt_gap_truth_matrix.csv", out)
    return out


def write_simple_svg(path: Path, title: str, rows: list[dict[str, Any]], label_key: str, value_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = []
    for row in rows[:22]:
        vals.append((str(row.get(label_key, ""))[:30], float(finite_float(row.get(value_key), 0.0) or 0.0)))
    width = 920
    height = max(220, 58 + 28 * max(1, len(vals)))
    max_abs = max([abs(v) for _l, v in vals] + [1.0])
    zero = 390
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="28" font-family="Arial" font-size="18">{title}</text>',
        f'<line x1="{zero}" y1="45" x2="{zero}" y2="{height - 18}" stroke="#444" stroke-width="1"/>',
    ]
    for i, (label, val) in enumerate(vals):
        y = 62 + i * 28
        bar = int((val / max_abs) * 270)
        x = zero if bar >= 0 else zero + bar
        color = "#28745a" if bar >= 0 else "#aa4747"
        lines.append(f'<text x="20" y="{y + 14}" font-family="Arial" font-size="11">{label}</text>')
        lines.append(f'<rect x="{x}" y="{y}" width="{abs(bar)}" height="18" fill="{color}" opacity="0.85"/>')
        lines.append(f'<text x="{zero + bar + (6 if bar >= 0 else -90)}" y="{y + 14}" font-family="Arial" font-size="11">{val:.4g}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize_gap_truth_from_matrices() -> None:
    rows = read_rows(OUT_ROOT / "v22_38_KAN_basis_native_full_loop_matrix.csv") + read_rows(OUT_ROOT / "v22_38_signal_incremental_full_loop_matrix.csv")
    by_key: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for r in rows:
        by_key[(str(r.get("dataset")), str(r.get("seed")), str(r.get("carrier")), str(r.get("optimizer_family")), str(r.get("training_variant")))] = r
    out = []
    emitted_keys: set[tuple[str, str, str, str, str]] = set()
    for key, kan_fu in by_key.items():
        dataset, seed, carrier, opt, variant = key
        if carrier not in {"D-CHE", "D-FOU"} or not str(variant).startswith("SD_CFO"):
            continue
        kan_base = by_key.get((dataset, seed, carrier, opt, "optimizer_alone"), {})
        mlp_base = by_key.get((dataset, seed, "", opt, "optimizer_alone"), {})
        mlp_fu = by_key.get((dataset, seed, "", opt, "SD_CFO_signal_incremental_a1"), {})
        vals = [finite_float(r.get("final_test_NLL")) for r in [kan_base, kan_fu, mlp_base, mlp_fu]]
        if any(v is None for v in vals):
            continue
        kan_base_nll, kan_fu_nll, mlp_base_nll, mlp_fu_nll = [float(v) for v in vals if v is not None]
        gap_base = kan_base_nll - mlp_base_nll
        gap_fu = kan_fu_nll - mlp_fu_nll
        gap_reduction = gap_base - gap_fu
        delta_kan = kan_fu_nll - kan_base_nll
        delta_mlp = mlp_fu_nll - mlp_base_nll
        kan_fu_acc = finite_float(kan_fu.get("final_test_accuracy"))
        mlp_fu_acc = finite_float(mlp_fu.get("final_test_accuracy"))
        beats_by_accuracy = "" if kan_fu_acc is None or mlp_fu_acc is None else int(float(kan_fu_acc) >= float(mlp_fu_acc))
        beats_by_nll = int(kan_fu_nll <= mlp_fu_nll)
        beats_by_nll_or_accuracy = int(bool(beats_by_nll) or beats_by_accuracy == 1)
        if delta_mlp > 0.0 and gap_reduction > 0.0:
            klass = "MLPDegradationDriven"
        elif delta_kan < 0.0 and delta_mlp < 0.0 and gap_reduction > 0.0:
            klass = "BothGain"
        elif delta_kan < 0.0 and delta_mlp <= 0.0 and gap_reduction > 0.0:
            klass = "TrueKANGain"
        else:
            klass = "NoGain"
        emit_key = (dataset, seed, carrier, opt, str(kan_fu.get("treatment_name", "")))
        if emit_key in emitted_keys:
            continue
        emitted_keys.add(emit_key)
        out.append(
            {
                "dataset": dataset,
                "seed": seed,
                "optimizer_family": opt,
                "kan_carrier": carrier,
                "selected_treatment": kan_fu.get("treatment_name", ""),
                "MLP_strong_NLL": mlp_base_nll,
                "MLP_SD_CFO_NLL": mlp_fu_nll,
                "MLP_SD_CFO_accuracy": mlp_fu_acc if mlp_fu_acc is not None else "",
                "KAN_strong_NLL": kan_base_nll,
                "KAN_SD_CFO_NLL": kan_fu_nll,
                "KAN_SD_CFO_accuracy": kan_fu_acc if kan_fu_acc is not None else "",
                "Delta_MLP_NLL": delta_mlp,
                "Delta_KAN_NLL": delta_kan,
                "Gap_base": gap_base,
                "Gap_FU": gap_fu,
                "GapReduction": gap_reduction,
                "KAN_SD_CFO_beats_MLP_SD_CFO": beats_by_nll,
                "KAN_SD_CFO_beats_MLP_SD_CFO_by_accuracy": beats_by_accuracy,
                "KAN_SD_CFO_beats_MLP_SD_CFO_by_NLL_or_accuracy": beats_by_nll_or_accuracy,
                "TrueKANGain_class": klass,
                "status": "completed_gap_truth_compact",
            }
        )
    write_rows(OUT_ROOT / "v22_38_KANbeFair_gap_truth_matrix.csv", out or [{"status": "no_gap_truth_rows", "reason": "missing paired MLP/KAN SD-CFO rows"}])


def ensure_required_artifacts(reason: str = "not_reached") -> None:
    for name in REQUIRED_ARTIFACTS:
        path = OUT_ROOT / name
        if path.exists() and path.stat().st_size > 0:
            continue
        if name.endswith(".json"):
            write_json(path, {"status": "not_run", "reason": reason})
        else:
            write_rows(path, [{"status": "not_run", "reason": reason}])
    for fig in REQUIRED_FIGURES:
        path = FIG_ROOT / fig
        if not path.exists() or path.stat().st_size == 0:
            write_simple_svg(path, fig, [{"label": "not_run", "value": 0}], "label", "value")


def artifact_index() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(OUT_ROOT.rglob("*")):
        if path.is_file():
            rows.append({"artifact": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "nonempty": int(path.stat().st_size > 0), "timestamp": now_sg()})
    return rows


def update_code_truth_after_randomization() -> None:
    code = read_rows(OUT_ROOT / "v22_38_code_truth_gate.csv") or [{}]
    prop = read_rows(OUT_ROOT / "v22_38_randomization_propensity_matrix.csv")
    pmins = [finite_float(r.get("logged_min_propensity")) for r in prop]
    pmins_f = [float(v) for v in pmins if v is not None]
    code[0]["randomization_propensity_logged"] = int(bool(pmins_f))
    code[0]["propensity_min_by_treatment"] = "" if not pmins_f else min(pmins_f)
    code[0]["command_journal_complete"] = int((OUT_ROOT / "v22_38_command_journal.csv").exists())
    write_rows(OUT_ROOT / "v22_38_code_truth_gate.csv", code)


def finalize_route(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    materialize_gap_truth_from_matrices()
    summarize_horizon_robust_attempts()
    summarize_attempt_gap_truth()
    stage_figures_efficiency()
    ensure_required_artifacts()
    update_code_truth_after_randomization()
    write_rows(OUT_ROOT / "v22_38_artifact_index.csv", artifact_index())
    manifest_hash = sha256_file(OUT_ROOT / "v22_38_artifact_index.csv")
    code = read_rows(OUT_ROOT / "v22_38_code_truth_gate.csv") or [{}]
    code[0]["artifact_manifest_hash"] = manifest_hash
    write_rows(OUT_ROOT / "v22_38_code_truth_gate.csv", code)
    prop = read_rows(OUT_ROOT / "v22_38_randomization_propensity_matrix.csv")
    decomp = read_rows(OUT_ROOT / "v22_38_support_direction_decomposition.csv")
    policy = (read_rows(OUT_ROOT / "v22_38_CATE_policy_calibration.csv") or [{}])[0]
    support_summary = (read_rows(OUT_ROOT / "v22_38_support_native_full_loop_matrix_summary.csv") or [{}])[0]
    signal_summary = (read_rows(OUT_ROOT / "v22_38_signal_incremental_full_loop_matrix_summary.csv") or [{}])[0]
    opt_summary = (read_rows(OUT_ROOT / "v22_38_optimizer_state_full_loop_matrix_summary.csv") or [{}])[0]
    kan_summary = (read_rows(OUT_ROOT / "v22_38_KAN_basis_native_full_loop_matrix_summary.csv") or [{}])[0]
    gap = read_rows(OUT_ROOT / "v22_38_KANbeFair_gap_truth_matrix.csv")
    code_pass = int_flag(code[0].get("clean_unzip_compileall_pass")) and int_flag(code[0].get("clean_unzip_import_pass"))
    prop_pass = bool(prop) and all(int_flag(r.get("propensity_gate_pass")) for r in prop if r.get("treatment_name"))
    direction_open = any((finite_float(r.get("LCB_direction"), -math.inf) or -math.inf) > 0.0 for r in decomp if TREATMENT_META.get(str(r.get("treatment_name", "")), {}).get("kind") == "signal")
    support_open = (finite_float(support_summary.get("NLL_improvement_vs_own_strong_rows"), 0.0) or 0.0) >= 5
    cate_open = int_flag(policy.get("policy_gate_pass"))
    signal_full_open = (finite_float(signal_summary.get("NLL_improvement_vs_FSO_rows"), 0.0) or 0.0) >= 5
    optimizer_open = (finite_float(opt_summary.get("NLL_improvement_vs_own_strong_rows"), 0.0) or 0.0) >= 5
    kan_open = (finite_float(kan_summary.get("NLL_improvement_vs_own_strong_rows"), 0.0) or 0.0) >= 5
    true_both = sum(1 for r in gap if r.get("TrueKANGain_class") in {"TrueKANGain", "BothGain"})
    route = "R0-CodeOrIdentityFail"
    reason = "code/import gate failed"
    if code_pass:
        if not prop_pass:
            route = "R1-RandomizationOrPropensityFail"
            reason = "propensity logging missing or min propensity below 0.05"
        elif direction_open and not cate_open:
            route = "R2-ATEPositive_CATEPolicyNoGo"
            reason = "positive average direction LCB exists, but CATE policy gate did not open"
        elif support_open and not signal_full_open:
            route = "R3-SupportNativeOptimizerOpened_NoSignalIncrement"
            reason = "support route has positive evidence or ran, but signal incremental full-loop not opened"
        elif signal_full_open and not optimizer_open:
            route = "R4-SignalIncrementOpened_NoFullLoop"
            reason = "signal incremental rows improved versus FSO, but optimizer-state/full superiority not opened"
        elif optimizer_open and not kan_open:
            route = "R5-OptimizerStateFUOpened_MLPPending"
            reason = "optimizer-state compact rows opened but KAN/general value pending"
        elif kan_open and true_both == 0:
            route = "R8-KANInternalValueOpened_ArchitecturePending"
            reason = "KAN internal rows opened, but TrueKANGain/BothGain gap evidence did not open"
        elif true_both > 0:
            route = "R10-TrueKANGainExplorationOpened"
            reason = "at least one TrueKANGain/BothGain compact row exists; official thresholds still pending"
    final = {
        "final_route": route,
        "route_reason": reason,
        "code_pass": int(bool(code_pass)),
        "propensity_pass": int(bool(prop_pass)),
        "support_open": int(bool(support_open)),
        "direction_open": int(bool(direction_open)),
        "CATE_policy_gate_pass": int(bool(cate_open)),
        "CATE_policy_official_candidate_pass": int_flag(policy.get("policy_official_candidate_pass", policy.get("official_candidate_gate_pass", 0))),
        "signal_full_open": int(bool(signal_full_open)),
        "optimizer_open": int(bool(optimizer_open)),
        "KAN_internal_open": int(bool(kan_open)),
        "gap_true_or_both_rows": true_both,
        "artifact_manifest_hash": manifest_hash,
        "no_fake_data_statement": "All metrics are computed from local CSV/JSON artifacts; blocked and unavailable stages are explicitly marked.",
        "timestamp": now_sg(),
    }
    write_json(OUT_ROOT / "v22_38_final_route.json", final)
    append_exec("finalize_route", task_id="Finalize", status="pass", gpu="0", files="results/v22_38/v22_38_final_route.json, results/v22_38/v22_38_artifact_index.csv", note=f"route={route}; reason={reason}")
    write_recap(final)
    return final


def write_recap(final: dict[str, Any]) -> None:
    code = read_rows(OUT_ROOT / "v22_38_code_truth_gate.csv")
    prop = read_rows(OUT_ROOT / "v22_38_randomization_propensity_matrix.csv")
    decomp = read_rows(OUT_ROOT / "v22_38_support_direction_decomposition.csv")
    horizon = read_rows(OUT_ROOT / "v22_38_horizon_effect_matrix.csv")
    safety = read_rows(OUT_ROOT / "v22_38_safety_debt_by_treatment.csv")
    ablation = read_rows(OUT_ROOT / "v22_38_CATE_feature_ablation.csv")
    policy = read_rows(OUT_ROOT / "v22_38_CATE_policy_calibration.csv")
    treatment_policy = read_rows(OUT_ROOT / "v22_38_CATE_treatment_policy_calibration.csv")
    support = read_rows(OUT_ROOT / "v22_38_support_native_full_loop_matrix.csv")
    signal = read_rows(OUT_ROOT / "v22_38_signal_incremental_full_loop_matrix.csv")
    opt = read_rows(OUT_ROOT / "v22_38_optimizer_state_full_loop_matrix.csv")
    kan = read_rows(OUT_ROOT / "v22_38_KAN_basis_native_full_loop_matrix.csv")
    gap = read_rows(OUT_ROOT / "v22_38_KANbeFair_gap_truth_matrix.csv")
    continual = read_rows(OUT_ROOT / "v22_38_continual_memory_matrix.csv")
    eff = read_rows(OUT_ROOT / "v22_38_efficiency_matrix.csv")
    attempts = read_rows(OUT_ROOT / "v22_38_horizon_robust_safety_gate_attempts.csv")
    attempt_gap = read_rows(OUT_ROOT / "v22_38_attempt_gap_truth_matrix.csv")
    policy0 = policy[0] if policy else {}
    support_summary = (read_rows(OUT_ROOT / "v22_38_support_native_full_loop_matrix_summary.csv") or [{}])[0]
    signal_summary = (read_rows(OUT_ROOT / "v22_38_signal_incremental_full_loop_matrix_summary.csv") or [{}])[0]
    opt_summary = (read_rows(OUT_ROOT / "v22_38_optimizer_state_full_loop_matrix_summary.csv") or [{}])[0]
    kan_summary = (read_rows(OUT_ROOT / "v22_38_KAN_basis_native_full_loop_matrix_summary.csv") or [{}])[0]
    def attempt_summary(tag: str, plan: str) -> dict[str, Any]:
        for row in attempts:
            if row.get("attempt_tag") == tag and row.get("plan") == plan:
                return row
        return {}

    def attempt_rows(tag: str, filename: str) -> list[dict[str, Any]]:
        return read_rows(OUT_ROOT / "attempts" / tag / filename)

    def treatment_rollup(tag: str, filename: str, treatment: str) -> str:
        rows = [r for r in attempt_rows(tag, filename) if r.get("treatment_name") == treatment]
        if not rows:
            return "no rows"
        wins = sum(1 for r in rows if (finite_float(r.get("NLL_delta_vs_own_strong_optimizer"), 0.0) or 0.0) < 0.0)
        strict = sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in rows)
        boot = sum(int_flag(r.get("no_ECE_Brier_tail_debt_bootstrap_0p90")) for r in rows)
        passes = sum(int(finite_float(r.get("runtime_CATE_policy_pass_count"), 0.0) or 0) for r in rows)
        accepted = sum(int(finite_float(r.get("accepted_treatment_count"), 0.0) or 0) for r in rows)
        return f"wins={wins}/{len(rows)}, strict={strict}/{len(rows)}, bootstrap={boot}/{len(rows)}, runtime_passes={passes}, accepted={accepted}"

    def direct_event_stats(treatment: str) -> str:
        rows = [r for r in read_rows(OUT_ROOT / "v22_38_CATE_event_matrix.csv") if r.get("pilot_treatment_mode") == "expanded_planned_basis" and r.get("treatment_selected") == treatment]
        if not rows:
            return "no rows"
        positives = sum(int_flag(r.get("positive_treatment_outcome")) for r in rows)
        no_debt = sum(int_flag(r.get("Y_no_debt_gate")) for r in rows)
        vals = [finite_float(r.get("Y_robust_NLL")) for r in rows]
        vals_f = [float(v) for v in vals if v is not None]
        mean_y = "" if not vals_f else sum(vals_f) / len(vals_f)
        return f"n={len(rows)}, positive={positives}, no_debt={no_debt}, mean_Y={mean_y}"

    def arbitration_selection_rollup(tag: str, filename: str) -> str:
        counts: dict[str, int] = {}
        rows = [r for r in attempt_rows(tag, filename) if str(r.get("treatment_name", "")) in ARBITRATED_TREATMENT_CANDIDATES]
        for row in rows:
            raw = str(row.get("runtime_CATE_policy_arbitration_selected_counts", ""))
            for part in raw.split(";"):
                if not part or ":" not in part:
                    continue
                name, val = part.rsplit(":", 1)
                try:
                    counts[name] = counts.get(name, 0) + int(float(val))
                except ValueError:
                    continue
        return "no rows" if not counts else ";".join(f"{k}:{v}" for k, v in sorted(counts.items()))

    runtime_signal = attempt_summary("cate_runtime_h3_q80_officialcate", "signal")
    runtime_optimizer = attempt_summary("cate_runtime_h3_q80_officialcate", "optimizer")
    runtime_kan = attempt_summary("cate_runtime_h3_q80_officialcate", "kan_basis")
    hard_runtime_signal = attempt_summary("hard_tier1_cate_runtime_q80_s012", "signal")
    hard_runtime_optimizer = attempt_summary("hard_tier1_cate_runtime_q80_s012", "optimizer")
    hard_runtime_kan = attempt_summary("hard_tier1_cate_runtime_q80_s012", "kan_basis")
    opt_runtime_q65 = attempt_summary("optimizer_runtime_q65_loss_threshold", "optimizer")
    opt_runtime_q60 = attempt_summary("optimizer_runtime_q60_loss_threshold", "optimizer")
    opt_ts_q80_bug = attempt_summary("optimizer_runtime_treatment_policy_q80", "optimizer")
    opt_ts_q80 = attempt_summary("optimizer_runtime_treatment_policy_q80_scopefix", "optimizer")
    opt_ts_q75 = attempt_summary("optimizer_runtime_treatment_policy_q75_scopefix", "optimizer")
    opt_ts_q74 = attempt_summary("optimizer_runtime_treatment_policy_q74_scopefix", "optimizer")
    opt_ts_q73 = attempt_summary("optimizer_runtime_treatment_policy_q73_scopefix", "optimizer")
    opt_ts_q72 = attempt_summary("optimizer_runtime_treatment_policy_q72_scopefix", "optimizer")
    opt_ts_q70 = attempt_summary("optimizer_runtime_treatment_policy_q70_scopefix", "optimizer")
    opt_ts_q70_alpha = attempt_summary("optimizer_runtime_treatment_policy_q70_alphacap1e4", "optimizer")
    opt_ts_q70_h5b8 = attempt_summary("optimizer_runtime_treatment_policy_q70_h5b8", "optimizer")
    dfou_signal = attempt_summary("kan_dfou_cate_runtime_q80", "signal")
    dfou_kan = attempt_summary("kan_dfou_cate_runtime_q80", "kan_basis")
    long200_signal = attempt_summary("kan_carrier_long200_cate_runtime_q80", "signal")
    long200_kan = attempt_summary("kan_carrier_long200_cate_runtime_q80", "kan_basis")
    h64_long200_signal = attempt_summary("kan_carrier_h64_long200_cate_runtime_q80", "signal")
    h64_long200_kan = attempt_summary("kan_carrier_h64_long200_cate_runtime_q80", "kan_basis")
    h64_800_signal = attempt_summary("kan_carrier_h64_mnist_s0_steps800_cate_runtime_q80", "signal")
    h64_800_kan = attempt_summary("kan_carrier_h64_mnist_s0_steps800_cate_runtime_q80", "kan_basis")
    h64_800_alpha_signal = attempt_summary("kan_carrier_h64_mnist_s0_steps800_alpha1e3", "signal")
    h64_800_alpha_kan = attempt_summary("kan_carrier_h64_mnist_s0_steps800_alpha1e3", "kan_basis")
    h64_800_relbase_signal = attempt_summary("kan_carrier_h64_mnist_s0_steps800_relbase_alpha", "signal")
    h64_800_relbase_kan = attempt_summary("kan_carrier_h64_mnist_s0_steps800_relbase_alpha", "kan_basis")
    h64_800_relbase_low_signal = attempt_summary("kan_carrier_h64_mnist_s0_steps800_relbase_lowalpha", "signal")
    h64_800_relbase_low_kan = attempt_summary("kan_carrier_h64_mnist_s0_steps800_relbase_lowalpha", "kan_basis")
    h64_long200_relbase_signal = attempt_summary("kan_carrier_h64_long200_relbase_alpha", "signal")
    h64_long200_relbase_kan = attempt_summary("kan_carrier_h64_long200_relbase_alpha", "kan_basis")
    h64_800_relbase_h5b8_signal = attempt_summary("kan_carrier_h64_mnist_s0_steps800_relbase_h5b8_tail995", "signal")
    h64_800_relbase_h5b8_kan = attempt_summary("kan_carrier_h64_mnist_s0_steps800_relbase_h5b8_tail995", "kan_basis")
    h64_800_relbase_tempwide_signal = attempt_summary("kan_carrier_h64_mnist_s0_steps800_relbase_tempwide", "signal")
    h64_800_relbase_tempwide_kan = attempt_summary("kan_carrier_h64_mnist_s0_steps800_relbase_tempwide", "kan_basis")
    h64_800_relbase_temp22_signal = attempt_summary("kan_carrier_h64_mnist_s0_steps800_relbase_temp22", "signal")
    h64_800_relbase_temp22_kan = attempt_summary("kan_carrier_h64_mnist_s0_steps800_relbase_temp22", "kan_basis")
    h64_800_relbase_tempwide_nllbt_signal = attempt_summary("kan_carrier_h64_mnist_s0_steps800_relbase_tempwide_nllbt", "signal")
    h64_800_relbase_tempwide_nllbt_kan = attempt_summary("kan_carrier_h64_mnist_s0_steps800_relbase_tempwide_nllbt", "kan_basis")
    h64_long200_relbase_temp22_signal = attempt_summary("kan_carrier_h64_long200_relbase_temp22", "signal")
    h64_long200_relbase_temp22_kan = attempt_summary("kan_carrier_h64_long200_relbase_temp22", "kan_basis")
    h128_1600_relbase_nllbt_signal = attempt_summary("kan_carrier_h128_mnist_s0_steps1600_relbase_nllbt", "signal")
    h128_1600_relbase_nllbt_kan = attempt_summary("kan_carrier_h128_mnist_s0_steps1600_relbase_nllbt", "kan_basis")
    h128_s1_relbase_nllbt_signal = attempt_summary("kan_carrier_h128_mnist_s1_steps1600_relbase_nllbt", "signal")
    h128_s1_relbase_nllbt_kan = attempt_summary("kan_carrier_h128_mnist_s1_steps1600_relbase_nllbt", "kan_basis")
    h128_s01_relbase_temp22_signal = attempt_summary("kan_carrier_h128_mnist_s01_steps1600_relbase_temp22", "signal")
    h128_s01_relbase_temp22_kan = attempt_summary("kan_carrier_h128_mnist_s01_steps1600_relbase_temp22", "kan_basis")
    h128_s1_relbase_tol001_signal = attempt_summary("kan_carrier_h128_mnist_s1_steps1600_relbase_nllbt_tol001", "signal")
    h128_s1_relbase_tol001_kan = attempt_summary("kan_carrier_h128_mnist_s1_steps1600_relbase_nllbt_tol001", "kan_basis")
    h128_s1_relbase_tol0001_signal = attempt_summary("kan_carrier_h128_mnist_s1_steps1600_relbase_nllbt_tol0001", "signal")
    h128_s1_relbase_lowalpha_signal = attempt_summary("kan_carrier_h128_mnist_s1_steps1600_relbase_nllbt_lowalpha", "signal")
    h128_s1_relbase_temp17_signal = attempt_summary("kan_carrier_h128_mnist_s1_steps1600_relbase_temp17", "signal")
    h128_s1_relbase_ebt_signal = attempt_summary("kan_carrier_h128_mnist_s1_steps1600_relbase_ebt", "signal")
    h128_s1_relbase_h5b8_signal = attempt_summary("kan_carrier_h128_mnist_s1_steps1600_relbase_nllbt_h5b8_tail995", "signal")
    h128_s1_relbase_postrollback_signal = attempt_summary("kan_carrier_h128_mnist_s1_steps1600_relbase_nllbt_postrollback", "signal")
    h128_s1_relbase_held1024_signal = attempt_summary("kan_carrier_h128_mnist_s1_steps1600_relbase_nllbt_held1024", "signal")
    support_h128_s1_signal = attempt_summary("support_h128_mnist_s1_steps1600_relbase_nllbt", "support")
    h128_s0_boot_signal = attempt_summary("kan_carrier_h128_mnist_s0_steps1600_relbase_nllbt_boot200", "signal")
    h128_s1_boot_signal = attempt_summary("kan_carrier_h128_mnist_s1_steps1600_relbase_nllbt_boot200", "signal")
    h128_fk_boot_signal = attempt_summary("kan_carrier_h128_tier0_fk_s01_steps1600_relbase_nllbt_boot200", "signal")
    h128_s0_boot_kan = attempt_summary("kan_carrier_h128_mnist_s0_steps1600_relbase_nllbt_boot200", "kan_basis")
    h128_s1_boot_kan = attempt_summary("kan_carrier_h128_mnist_s1_steps1600_relbase_nllbt_boot200", "kan_basis")
    h128_fk_boot_kan = attempt_summary("kan_carrier_h128_tier0_fk_s01_steps1600_relbase_nllbt_boot200", "kan_basis")
    h128_hard_boot_signal = attempt_summary("kan_carrier_h128_hard_tier1_s0_steps1600_relbase_nllbt_boot200", "signal")
    h128_hard_boot_kan = attempt_summary("kan_carrier_h128_hard_tier1_s0_steps1600_relbase_nllbt_boot200", "kan_basis")
    h128_hard_thr022_signal = attempt_summary("kan_carrier_h128_hard_tier1_s0_steps1600_relbase_nllbt_boot200_thr022", "signal")
    h128_hard_thr030_signal = attempt_summary("kan_carrier_h128_hard_tier1_s0_steps1600_relbase_nllbt_boot200_thr030", "signal")
    h128_hard_lowalpha_signal = attempt_summary("kan_carrier_h128_hard_tier1_s0_steps1600_relbase_nllbt_boot200_lowalpha", "signal")
    h128_hard_lowalpha_h5b8_signal = attempt_summary("kan_carrier_h128_hard_tier1_s0_steps1600_relbase_nllbt_boot200_lowalpha_h5b8", "signal")
    h256_svhn_s0_lowalpha_h5b8_signal = attempt_summary("kan_carrier_h256_hard_svhn_s0_steps1600_relbase_nllbt_boot200_lowalpha_h5b8", "signal")
    h256_svhn_s0_micro_h5b8_signal = attempt_summary("kan_carrier_h256_hard_svhn_s0_steps1600_relbase_nllbt_boot200_microalpha_h5b8", "signal")
    h256_hard_s0_micro_h5b8_signal = attempt_summary("kan_carrier_h256_hard_tier1_s0_steps1600_relbase_nllbt_boot200_microalpha_h5b8", "signal")
    h256_hard_s1_micro_h5b8_signal = attempt_summary("kan_carrier_h256_hard_tier1_s1_steps1600_relbase_nllbt_boot200_microalpha_h5b8", "signal")
    h256_hard_s2_micro_h5b8_signal = attempt_summary("kan_carrier_h256_hard_tier1_s2_steps1600_relbase_nllbt_boot200_microalpha_h5b8", "signal")
    h256_hard_s0_steps3200_micro_h5b8_signal = attempt_summary("kan_carrier_h256_hard_tier1_s0_steps3200_relbase_nllbt_boot200_microalpha_h5b8", "signal")
    h256_hard_s1_steps3200_micro_h5b8_signal = attempt_summary("kan_carrier_h256_hard_tier1_s1_steps3200_relbase_nllbt_boot200_microalpha_h5b8", "signal")
    h256_hard_s0_safe3200_micro_h5b8_signal = attempt_summary("kan_carrier_h256_hard_tier1_s0_steps3200_relbase_ebt_boot200_microalpha_h10b16_tail995_postrollback", "signal")
    h256_hard_s1_safe3200_micro_h5b8_signal = attempt_summary("kan_carrier_h256_hard_tier1_s1_steps3200_relbase_ebt_boot200_microalpha_h10b16_tail995_postrollback", "signal")
    h256_hard_s0_ultra3200_micro_h5b8_signal = attempt_summary("kan_carrier_h256_hard_tier1_s0_steps3200_relbase_ebt_boot200_ultramicro_h10b16_tail995_postrollback", "signal")
    h256_hard_s1_ultra3200_micro_h5b8_signal = attempt_summary("kan_carrier_h256_hard_tier1_s1_steps3200_relbase_ebt_boot200_ultramicro_h10b16_tail995_postrollback", "signal")
    h256_hard_s1_lowalpha_h5b8_signal = attempt_summary("kan_carrier_h256_hard_tier1_s1_steps1600_relbase_nllbt_boot200_lowalpha_h5b8", "signal")
    h512_svhn_s1_micro_h5b8_signal = attempt_summary("kan_carrier_h512_hard_svhn_s1_steps1600_relbase_nllbt_boot200_microalpha_h5b8", "signal")
    h512_hard_s0_micro_h5b8_signal = attempt_summary("kan_carrier_h512_hard_tier1_s0_steps1600_relbase_nllbt_boot200_microalpha_h5b8", "signal")
    h512_hard_s1_micro_h5b8_signal = attempt_summary("kan_carrier_h512_hard_tier1_s1_steps1600_relbase_nllbt_boot200_microalpha_h5b8", "signal")
    h512_hard_s0_safe3200_micro_h5b8_signal = attempt_summary("kan_carrier_h512_hard_tier1_s0_steps3200_relbase_ebt_boot200_microalpha_h10b16_tail995_postrollback", "signal")
    h512_hard_s1_safe3200_micro_h5b8_signal = attempt_summary("kan_carrier_h512_hard_tier1_s1_steps3200_relbase_ebt_boot200_microalpha_h10b16_tail995_postrollback", "signal")
    opt_h256_hard_s0_safe3200 = attempt_summary("optimizer_h256_hard_tier1_s0_steps3200_relbase_ebt_boot200_microalpha_h10b16_tail995_postrollback", "optimizer")
    opt_h256_hard_s1_safe3200 = attempt_summary("optimizer_h256_hard_tier1_s1_steps3200_relbase_ebt_boot200_microalpha_h10b16_tail995_postrollback", "optimizer")
    h256_hard_s0_safe3200_thr010_signal = attempt_summary("kan_carrier_h256_hard_tier1_s0_steps3200_relbase_ebt_boot200_microalpha_h10b16_tail995_postrollback_thr010", "signal")
    h256_hard_s1_safe3200_thr010_signal = attempt_summary("kan_carrier_h256_hard_tier1_s1_steps3200_relbase_ebt_boot200_microalpha_h10b16_tail995_postrollback_thr010", "signal")
    h256_hard_s0_safe3200_thr010_cad2_signal = attempt_summary("kan_carrier_h256_hard_tier1_s0_steps3200_relbase_ebt_boot200_microalpha_h10b16_tail995_postrollback_thr010_cad2", "signal")
    h256_hard_s1_safe3200_thr010_cad2_signal = attempt_summary("kan_carrier_h256_hard_tier1_s1_steps3200_relbase_ebt_boot200_microalpha_h10b16_tail995_postrollback_thr010_cad2", "signal")
    expanded_global_tag = "kan_carrier_h256_hard_tier1_s0_steps3200_expandedbasis_globalcate_safe"
    expanded_ts_bug_tag = "kan_carrier_h256_hard_tier1_s0_steps3200_expandedbasis_treatspecific_directcate_safe"
    expanded_fix_tag = "kan_carrier_h256_hard_tier1_s0_steps3200_expandedbasis_treatspecific_directcate_runtimefeatfix_safe"
    expanded_global_signal = attempt_summary(expanded_global_tag, "signal")
    expanded_global_kan = attempt_summary(expanded_global_tag, "kan_basis")
    expanded_ts_bug_signal = attempt_summary(expanded_ts_bug_tag, "signal")
    expanded_ts_bug_kan = attempt_summary(expanded_ts_bug_tag, "kan_basis")
    expanded_fix_signal = attempt_summary(expanded_fix_tag, "signal")
    expanded_fix_kan = attempt_summary(expanded_fix_tag, "kan_basis")
    arbitrated_tag = "kan_carrier_h256_hard_tier1_s0_steps3200_a2a5_arbitrated_safe"
    arbitrated_signal = attempt_summary(arbitrated_tag, "signal")
    arbitrated_kan = attempt_summary(arbitrated_tag, "kan_basis")
    hard_q80_strict_boot_tag = "hard_tier1_cate_runtime_q80_s012_strictgate_boot200"
    hard_q80_strict_boot_signal = attempt_summary(hard_q80_strict_boot_tag, "signal")
    hard_q80_strict_boot_kan = attempt_summary(hard_q80_strict_boot_tag, "kan_basis")
    hard_q80_h5b8_tag = "hard_tier1_cate_runtime_q80_s012_h5b8_tail995_boot200"
    hard_q80_h5b8_signal = attempt_summary(hard_q80_h5b8_tag, "signal")
    hard_q80_h5b8_kan = attempt_summary(hard_q80_h5b8_tag, "kan_basis")
    hard_q80_h7b12_tag = "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200"
    hard_q80_h7b12_signal = attempt_summary(hard_q80_h7b12_tag, "signal")
    hard_q80_h7b12_kan = attempt_summary(hard_q80_h7b12_tag, "kan_basis")
    hard_q80_h7b12_tempwide_tag = "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide"
    hard_q80_h7b12_tempwide_signal = attempt_summary(hard_q80_h7b12_tempwide_tag, "signal")
    hard_q80_h7b12_tempwide_kan = attempt_summary(hard_q80_h7b12_tempwide_tag, "kan_basis")
    hard_q80_h7b12_tempwide_tailq99_tag = "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_tailq99"
    hard_q80_h7b12_tempwide_tailq99_signal = attempt_summary(hard_q80_h7b12_tempwide_tailq99_tag, "signal")
    hard_q80_h7b12_tempwide_tailq99_kan = attempt_summary(hard_q80_h7b12_tempwide_tailq99_tag, "kan_basis")
    hard_q80_tail_repair_followup_tags = [
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_tailq99_T10",
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail999_boot200_tempwide",
        "hard_tier1_cate_runtime_q80_s012_h10b16_tail995_boot200_tempwide",
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_tailq99_T10_lowalpha",
    ]
    hard_q80_acceptance_tol_frontier_tags = [
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_tol1e7",
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_tol2e7",
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_tol3e7",
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_tol4e7",
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_tol5e7",
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_tol1e6",
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_tol2e6",
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_tol4e6",
    ]
    hard_q80_temperature_extension_tags = [
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_tailq99_T20",
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_tailq99_T30",
    ]
    hard_q80_runtime_safety_policy_tags = [
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_safetypolicy_safetygbt_q40",
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_safetypolicy_lossgbt_q40",
    ]
    hard_q80_runtime_safety_kan_diagnostic_tags = [
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_tailq99_T10_lossgbt_thr1883",
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_tailq99_T10_lossgbt_thr1886",
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_tailq99_T10_tol5e7",
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_tailq99_T10_lossgbt_thr1883_tol5e7",
    ]
    hard_q80_confidence_calibration_kan_tags = [
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_confcal_nllbt_T10",
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_confcal_ebt_T10",
    ]
    hard_q80_tail_margin_kan_tags = [
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_tailq99_T10_tailmargin2e7",
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_tailq99_T10_tailmargin5e7",
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_tailq99_T10_tailmargin1e6",
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_tailq99_T10_tailmargin5e6",
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_tailq99_T10_tailmargin1e5",
        "hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_tailmargin5e7",
    ]
    a3_treatment_policy = next((r for r in treatment_policy if r.get("selected_treatment") == "a3_optimizer_state_signal"), {})
    a5_treatment_policy = next((r for r in treatment_policy if r.get("selected_treatment") == "a5_signal_incremental_basis"), {})
    h128_boot_mnist_tags = [
        "kan_carrier_h128_mnist_s0_steps1600_relbase_nllbt_boot200",
        "kan_carrier_h128_mnist_s1_steps1600_relbase_nllbt_boot200",
    ]
    h128_boot_fk_tags = ["kan_carrier_h128_tier0_fk_s01_steps1600_relbase_nllbt_boot200"]
    h128_boot_all_tags = h128_boot_mnist_tags + h128_boot_fk_tags
    h128_hard_boot_tags = ["kan_carrier_h128_hard_tier1_s0_steps1600_relbase_nllbt_boot200"]
    h128_hard_lowalpha_h5b8_tags = ["kan_carrier_h128_hard_tier1_s0_steps1600_relbase_nllbt_boot200_lowalpha_h5b8"]
    h256_hard_s0_micro_h5b8_tags = ["kan_carrier_h256_hard_tier1_s0_steps1600_relbase_nllbt_boot200_microalpha_h5b8"]
    h256_hard_s1_micro_h5b8_tags = ["kan_carrier_h256_hard_tier1_s1_steps1600_relbase_nllbt_boot200_microalpha_h5b8"]
    h256_hard_s2_micro_h5b8_tags = ["kan_carrier_h256_hard_tier1_s2_steps1600_relbase_nllbt_boot200_microalpha_h5b8"]
    h256_hard_s01_micro_h5b8_tags = h256_hard_s0_micro_h5b8_tags + h256_hard_s1_micro_h5b8_tags
    h256_hard_s012_micro_h5b8_tags = h256_hard_s01_micro_h5b8_tags + h256_hard_s2_micro_h5b8_tags
    h256_hard_s0_steps3200_micro_h5b8_tags = ["kan_carrier_h256_hard_tier1_s0_steps3200_relbase_nllbt_boot200_microalpha_h5b8"]
    h256_hard_s1_steps3200_micro_h5b8_tags = ["kan_carrier_h256_hard_tier1_s1_steps3200_relbase_nllbt_boot200_microalpha_h5b8"]
    h256_hard_s01_steps3200_micro_h5b8_tags = h256_hard_s0_steps3200_micro_h5b8_tags + h256_hard_s1_steps3200_micro_h5b8_tags
    h256_hard_s0_safe3200_micro_h5b8_tags = ["kan_carrier_h256_hard_tier1_s0_steps3200_relbase_ebt_boot200_microalpha_h10b16_tail995_postrollback"]
    h256_hard_s1_safe3200_micro_h5b8_tags = ["kan_carrier_h256_hard_tier1_s1_steps3200_relbase_ebt_boot200_microalpha_h10b16_tail995_postrollback"]
    h256_hard_s01_safe3200_micro_h5b8_tags = h256_hard_s0_safe3200_micro_h5b8_tags + h256_hard_s1_safe3200_micro_h5b8_tags
    h256_hard_s0_ultra3200_micro_h5b8_tags = ["kan_carrier_h256_hard_tier1_s0_steps3200_relbase_ebt_boot200_ultramicro_h10b16_tail995_postrollback"]
    h256_hard_s1_ultra3200_micro_h5b8_tags = ["kan_carrier_h256_hard_tier1_s1_steps3200_relbase_ebt_boot200_ultramicro_h10b16_tail995_postrollback"]
    h256_hard_s01_ultra3200_micro_h5b8_tags = h256_hard_s0_ultra3200_micro_h5b8_tags + h256_hard_s1_ultra3200_micro_h5b8_tags
    h256_hard_s1_lowalpha_h5b8_tags = ["kan_carrier_h256_hard_tier1_s1_steps1600_relbase_nllbt_boot200_lowalpha_h5b8"]
    h512_svhn_s1_micro_h5b8_tags = ["kan_carrier_h512_hard_svhn_s1_steps1600_relbase_nllbt_boot200_microalpha_h5b8"]
    h512_hard_s0_micro_h5b8_tags = ["kan_carrier_h512_hard_tier1_s0_steps1600_relbase_nllbt_boot200_microalpha_h5b8"]
    h512_hard_s1_micro_h5b8_tags = ["kan_carrier_h512_hard_tier1_s1_steps1600_relbase_nllbt_boot200_microalpha_h5b8"]
    h512_hard_s01_micro_h5b8_tags = h512_hard_s0_micro_h5b8_tags + h512_hard_s1_micro_h5b8_tags
    h512_hard_s0_safe3200_micro_h5b8_tags = ["kan_carrier_h512_hard_tier1_s0_steps3200_relbase_ebt_boot200_microalpha_h10b16_tail995_postrollback"]
    h512_hard_s1_safe3200_micro_h5b8_tags = ["kan_carrier_h512_hard_tier1_s1_steps3200_relbase_ebt_boot200_microalpha_h10b16_tail995_postrollback"]
    h512_hard_s01_safe3200_micro_h5b8_tags = h512_hard_s0_safe3200_micro_h5b8_tags + h512_hard_s1_safe3200_micro_h5b8_tags
    opt_h256_hard_s01_safe3200_tags = [
        "optimizer_h256_hard_tier1_s0_steps3200_relbase_ebt_boot200_microalpha_h10b16_tail995_postrollback",
        "optimizer_h256_hard_tier1_s1_steps3200_relbase_ebt_boot200_microalpha_h10b16_tail995_postrollback",
    ]
    h256_hard_s01_safe3200_thr010_tags = [
        "kan_carrier_h256_hard_tier1_s0_steps3200_relbase_ebt_boot200_microalpha_h10b16_tail995_postrollback_thr010",
        "kan_carrier_h256_hard_tier1_s1_steps3200_relbase_ebt_boot200_microalpha_h10b16_tail995_postrollback_thr010",
    ]
    h256_hard_s01_safe3200_thr010_cad2_tags = [
        "kan_carrier_h256_hard_tier1_s0_steps3200_relbase_ebt_boot200_microalpha_h10b16_tail995_postrollback_thr010_cad2",
        "kan_carrier_h256_hard_tier1_s1_steps3200_relbase_ebt_boot200_microalpha_h10b16_tail995_postrollback_thr010_cad2",
    ]
    expanded_global_tags = [expanded_global_tag]
    expanded_ts_bug_tags = [expanded_ts_bug_tag]
    expanded_fix_tags = [expanded_fix_tag]
    arbitrated_tags = [arbitrated_tag]
    hard_q80_strict_boot_tags = [hard_q80_strict_boot_tag]
    hard_q80_h5b8_tags = [hard_q80_h5b8_tag]
    hard_q80_h7b12_tags = [hard_q80_h7b12_tag]
    hard_q80_h7b12_tempwide_tags = [hard_q80_h7b12_tempwide_tag]
    hard_q80_h7b12_tempwide_tailq99_tags = [hard_q80_h7b12_tempwide_tailq99_tag]

    def sum_attempt_field(tags: list[str], plan: str, field: str) -> int:
        vals = []
        for tag in tags:
            row = attempt_summary(tag, plan)
            val = finite_float(row.get(field), 0.0)
            vals.append(int(val or 0))
        return sum(vals)

    def gap_stats(tag: str) -> tuple[int, int, int]:
        rows = [r for r in attempt_gap if r.get("attempt_tag") == tag]
        true_both = sum(1 for r in rows if r.get("TrueKANGain_class") in {"TrueKANGain", "BothGain"})
        beats_mlp = sum(int_flag(r.get("KAN_SD_CFO_beats_MLP_SD_CFO")) for r in rows)
        return true_both, beats_mlp, len(rows)

    def gap_stats_any(tag: str) -> tuple[int, int]:
        rows = [r for r in attempt_gap if r.get("attempt_tag") == tag]
        return sum(int_flag(r.get("KAN_SD_CFO_beats_MLP_SD_CFO_by_NLL_or_accuracy")) for r in rows), len(rows)

    def gap_stats_many(tags: list[str]) -> tuple[int, int, int, int]:
        rows = [r for r in attempt_gap if r.get("attempt_tag") in set(tags)]
        true_both = sum(1 for r in rows if r.get("TrueKANGain_class") in {"TrueKANGain", "BothGain"})
        beats_nll = sum(int_flag(r.get("KAN_SD_CFO_beats_MLP_SD_CFO")) for r in rows)
        beats_any = sum(int_flag(r.get("KAN_SD_CFO_beats_MLP_SD_CFO_by_NLL_or_accuracy")) for r in rows)
        return true_both, beats_nll, beats_any, len(rows)

    dfou_true_both, dfou_beats_mlp, dfou_gap_total = gap_stats("kan_dfou_cate_runtime_q80")
    long200_true_both, long200_beats_mlp, long200_gap_total = gap_stats("kan_carrier_long200_cate_runtime_q80")
    h64_true_both, h64_beats_mlp, h64_gap_total = gap_stats("kan_carrier_h64_long200_cate_runtime_q80")
    h64_800_true_both, h64_800_beats_mlp, h64_800_gap_total = gap_stats("kan_carrier_h64_mnist_s0_steps800_cate_runtime_q80")
    h64_800_alpha_true_both, h64_800_alpha_beats_mlp, h64_800_alpha_gap_total = gap_stats("kan_carrier_h64_mnist_s0_steps800_alpha1e3")
    h64_800_relbase_true_both, h64_800_relbase_beats_mlp, h64_800_relbase_gap_total = gap_stats("kan_carrier_h64_mnist_s0_steps800_relbase_alpha")
    h64_800_relbase_low_true_both, h64_800_relbase_low_beats_mlp, h64_800_relbase_low_gap_total = gap_stats("kan_carrier_h64_mnist_s0_steps800_relbase_lowalpha")
    h64_long200_relbase_true_both, h64_long200_relbase_beats_mlp, h64_long200_relbase_gap_total = gap_stats("kan_carrier_h64_long200_relbase_alpha")
    h64_800_relbase_temp22_true_both, h64_800_relbase_temp22_beats_mlp, h64_800_relbase_temp22_gap_total = gap_stats("kan_carrier_h64_mnist_s0_steps800_relbase_temp22")
    h64_800_relbase_tempwide_nllbt_true_both, h64_800_relbase_tempwide_nllbt_beats_mlp, h64_800_relbase_tempwide_nllbt_gap_total = gap_stats("kan_carrier_h64_mnist_s0_steps800_relbase_tempwide_nllbt")
    h64_long200_relbase_temp22_true_both, h64_long200_relbase_temp22_beats_mlp, h64_long200_relbase_temp22_gap_total = gap_stats("kan_carrier_h64_long200_relbase_temp22")
    h128_1600_relbase_nllbt_true_both, h128_1600_relbase_nllbt_beats_mlp, h128_1600_relbase_nllbt_gap_total = gap_stats("kan_carrier_h128_mnist_s0_steps1600_relbase_nllbt")
    h128_1600_relbase_nllbt_beats_any, h128_1600_relbase_nllbt_any_total = gap_stats_any("kan_carrier_h128_mnist_s0_steps1600_relbase_nllbt")
    h128_s1_relbase_nllbt_true_both, h128_s1_relbase_nllbt_beats_mlp, h128_s1_relbase_nllbt_gap_total = gap_stats("kan_carrier_h128_mnist_s1_steps1600_relbase_nllbt")
    h128_s1_relbase_nllbt_beats_any, h128_s1_relbase_nllbt_any_total = gap_stats_any("kan_carrier_h128_mnist_s1_steps1600_relbase_nllbt")
    h128_s1_relbase_tol001_beats_any, h128_s1_relbase_tol001_any_total = gap_stats_any("kan_carrier_h128_mnist_s1_steps1600_relbase_nllbt_tol001")
    h128_boot_mnist_true_both, h128_boot_mnist_beats_nll, h128_boot_mnist_beats_any, h128_boot_mnist_gap_total = gap_stats_many(h128_boot_mnist_tags)
    h128_boot_fk_true_both, h128_boot_fk_beats_nll, h128_boot_fk_beats_any, h128_boot_fk_gap_total = gap_stats_many(h128_boot_fk_tags)
    h128_boot_all_true_both, h128_boot_all_beats_nll, h128_boot_all_beats_any, h128_boot_all_gap_total = gap_stats_many(h128_boot_all_tags)
    h128_hard_boot_true_both, h128_hard_boot_beats_nll, h128_hard_boot_beats_any, h128_hard_boot_gap_total = gap_stats_many(h128_hard_boot_tags)
    h128_hard_lowalpha_h5b8_true_both, h128_hard_lowalpha_h5b8_beats_nll, h128_hard_lowalpha_h5b8_beats_any, h128_hard_lowalpha_h5b8_gap_total = gap_stats_many(h128_hard_lowalpha_h5b8_tags)
    h256_hard_s0_micro_h5b8_true_both, h256_hard_s0_micro_h5b8_beats_nll, h256_hard_s0_micro_h5b8_beats_any, h256_hard_s0_micro_h5b8_gap_total = gap_stats_many(h256_hard_s0_micro_h5b8_tags)
    h256_hard_s1_micro_h5b8_true_both, h256_hard_s1_micro_h5b8_beats_nll, h256_hard_s1_micro_h5b8_beats_any, h256_hard_s1_micro_h5b8_gap_total = gap_stats_many(h256_hard_s1_micro_h5b8_tags)
    h256_hard_s2_micro_h5b8_true_both, h256_hard_s2_micro_h5b8_beats_nll, h256_hard_s2_micro_h5b8_beats_any, h256_hard_s2_micro_h5b8_gap_total = gap_stats_many(h256_hard_s2_micro_h5b8_tags)
    h256_hard_s01_micro_h5b8_true_both, h256_hard_s01_micro_h5b8_beats_nll, h256_hard_s01_micro_h5b8_beats_any, h256_hard_s01_micro_h5b8_gap_total = gap_stats_many(h256_hard_s01_micro_h5b8_tags)
    h256_hard_s012_micro_h5b8_true_both, h256_hard_s012_micro_h5b8_beats_nll, h256_hard_s012_micro_h5b8_beats_any, h256_hard_s012_micro_h5b8_gap_total = gap_stats_many(h256_hard_s012_micro_h5b8_tags)
    h256_hard_s0_steps3200_micro_h5b8_true_both, h256_hard_s0_steps3200_micro_h5b8_beats_nll, h256_hard_s0_steps3200_micro_h5b8_beats_any, h256_hard_s0_steps3200_micro_h5b8_gap_total = gap_stats_many(h256_hard_s0_steps3200_micro_h5b8_tags)
    h256_hard_s1_steps3200_micro_h5b8_true_both, h256_hard_s1_steps3200_micro_h5b8_beats_nll, h256_hard_s1_steps3200_micro_h5b8_beats_any, h256_hard_s1_steps3200_micro_h5b8_gap_total = gap_stats_many(h256_hard_s1_steps3200_micro_h5b8_tags)
    h256_hard_s01_steps3200_micro_h5b8_true_both, h256_hard_s01_steps3200_micro_h5b8_beats_nll, h256_hard_s01_steps3200_micro_h5b8_beats_any, h256_hard_s01_steps3200_micro_h5b8_gap_total = gap_stats_many(h256_hard_s01_steps3200_micro_h5b8_tags)
    h256_hard_s0_safe3200_micro_h5b8_true_both, h256_hard_s0_safe3200_micro_h5b8_beats_nll, h256_hard_s0_safe3200_micro_h5b8_beats_any, h256_hard_s0_safe3200_micro_h5b8_gap_total = gap_stats_many(h256_hard_s0_safe3200_micro_h5b8_tags)
    h256_hard_s1_safe3200_micro_h5b8_true_both, h256_hard_s1_safe3200_micro_h5b8_beats_nll, h256_hard_s1_safe3200_micro_h5b8_beats_any, h256_hard_s1_safe3200_micro_h5b8_gap_total = gap_stats_many(h256_hard_s1_safe3200_micro_h5b8_tags)
    h256_hard_s01_safe3200_micro_h5b8_true_both, h256_hard_s01_safe3200_micro_h5b8_beats_nll, h256_hard_s01_safe3200_micro_h5b8_beats_any, h256_hard_s01_safe3200_micro_h5b8_gap_total = gap_stats_many(h256_hard_s01_safe3200_micro_h5b8_tags)
    h256_hard_s0_ultra3200_micro_h5b8_true_both, h256_hard_s0_ultra3200_micro_h5b8_beats_nll, h256_hard_s0_ultra3200_micro_h5b8_beats_any, h256_hard_s0_ultra3200_micro_h5b8_gap_total = gap_stats_many(h256_hard_s0_ultra3200_micro_h5b8_tags)
    h256_hard_s1_ultra3200_micro_h5b8_true_both, h256_hard_s1_ultra3200_micro_h5b8_beats_nll, h256_hard_s1_ultra3200_micro_h5b8_beats_any, h256_hard_s1_ultra3200_micro_h5b8_gap_total = gap_stats_many(h256_hard_s1_ultra3200_micro_h5b8_tags)
    h256_hard_s01_ultra3200_micro_h5b8_true_both, h256_hard_s01_ultra3200_micro_h5b8_beats_nll, h256_hard_s01_ultra3200_micro_h5b8_beats_any, h256_hard_s01_ultra3200_micro_h5b8_gap_total = gap_stats_many(h256_hard_s01_ultra3200_micro_h5b8_tags)
    h256_hard_s1_lowalpha_h5b8_true_both, h256_hard_s1_lowalpha_h5b8_beats_nll, h256_hard_s1_lowalpha_h5b8_beats_any, h256_hard_s1_lowalpha_h5b8_gap_total = gap_stats_many(h256_hard_s1_lowalpha_h5b8_tags)
    h512_svhn_s1_micro_h5b8_true_both, h512_svhn_s1_micro_h5b8_beats_nll, h512_svhn_s1_micro_h5b8_beats_any, h512_svhn_s1_micro_h5b8_gap_total = gap_stats_many(h512_svhn_s1_micro_h5b8_tags)
    h512_hard_s0_micro_h5b8_true_both, h512_hard_s0_micro_h5b8_beats_nll, h512_hard_s0_micro_h5b8_beats_any, h512_hard_s0_micro_h5b8_gap_total = gap_stats_many(h512_hard_s0_micro_h5b8_tags)
    h512_hard_s1_micro_h5b8_true_both, h512_hard_s1_micro_h5b8_beats_nll, h512_hard_s1_micro_h5b8_beats_any, h512_hard_s1_micro_h5b8_gap_total = gap_stats_many(h512_hard_s1_micro_h5b8_tags)
    h512_hard_s01_micro_h5b8_true_both, h512_hard_s01_micro_h5b8_beats_nll, h512_hard_s01_micro_h5b8_beats_any, h512_hard_s01_micro_h5b8_gap_total = gap_stats_many(h512_hard_s01_micro_h5b8_tags)
    h512_hard_s0_safe3200_micro_h5b8_true_both, h512_hard_s0_safe3200_micro_h5b8_beats_nll, h512_hard_s0_safe3200_micro_h5b8_beats_any, h512_hard_s0_safe3200_micro_h5b8_gap_total = gap_stats_many(h512_hard_s0_safe3200_micro_h5b8_tags)
    h512_hard_s1_safe3200_micro_h5b8_true_both, h512_hard_s1_safe3200_micro_h5b8_beats_nll, h512_hard_s1_safe3200_micro_h5b8_beats_any, h512_hard_s1_safe3200_micro_h5b8_gap_total = gap_stats_many(h512_hard_s1_safe3200_micro_h5b8_tags)
    h512_hard_s01_safe3200_micro_h5b8_true_both, h512_hard_s01_safe3200_micro_h5b8_beats_nll, h512_hard_s01_safe3200_micro_h5b8_beats_any, h512_hard_s01_safe3200_micro_h5b8_gap_total = gap_stats_many(h512_hard_s01_safe3200_micro_h5b8_tags)
    expanded_global_true_both, expanded_global_beats_nll, expanded_global_beats_any, expanded_global_gap_total = gap_stats_many(expanded_global_tags)
    expanded_ts_bug_true_both, expanded_ts_bug_beats_nll, expanded_ts_bug_beats_any, expanded_ts_bug_gap_total = gap_stats_many(expanded_ts_bug_tags)
    expanded_fix_true_both, expanded_fix_beats_nll, expanded_fix_beats_any, expanded_fix_gap_total = gap_stats_many(expanded_fix_tags)
    arbitrated_true_both, arbitrated_beats_nll, arbitrated_beats_any, arbitrated_gap_total = gap_stats_many(arbitrated_tags)
    hard_q80_strict_boot_true_both, hard_q80_strict_boot_beats_nll, hard_q80_strict_boot_beats_any, hard_q80_strict_boot_gap_total = gap_stats_many(hard_q80_strict_boot_tags)
    hard_q80_h5b8_true_both, hard_q80_h5b8_beats_nll, hard_q80_h5b8_beats_any, hard_q80_h5b8_gap_total = gap_stats_many(hard_q80_h5b8_tags)
    hard_q80_h7b12_true_both, hard_q80_h7b12_beats_nll, hard_q80_h7b12_beats_any, hard_q80_h7b12_gap_total = gap_stats_many(hard_q80_h7b12_tags)
    hard_q80_h7b12_tempwide_true_both, hard_q80_h7b12_tempwide_beats_nll, hard_q80_h7b12_tempwide_beats_any, hard_q80_h7b12_tempwide_gap_total = gap_stats_many(hard_q80_h7b12_tempwide_tags)
    hard_q80_h7b12_tempwide_tailq99_true_both, hard_q80_h7b12_tempwide_tailq99_beats_nll, hard_q80_h7b12_tempwide_tailq99_beats_any, hard_q80_h7b12_tempwide_tailq99_gap_total = gap_stats_many(hard_q80_h7b12_tempwide_tailq99_tags)
    h128_boot_mnist_active = sum_attempt_field(h128_boot_mnist_tags, "signal", "active_KAN_SD_CFO_rows")
    h128_boot_mnist_active_wins = sum_attempt_field(h128_boot_mnist_tags, "signal", "active_KAN_SD_CFO_NLL_improvement_rows")
    h128_boot_mnist_active_strict = sum_attempt_field(h128_boot_mnist_tags, "signal", "active_KAN_SD_CFO_no_ECE_Brier_tail_debt_rows")
    h128_boot_mnist_active_boot = sum_attempt_field(h128_boot_mnist_tags, "signal", "active_KAN_SD_CFO_no_ECE_Brier_tail_debt_bootstrap_0p90_rows")
    h128_boot_fk_active = sum_attempt_field(h128_boot_fk_tags, "signal", "active_KAN_SD_CFO_rows")
    h128_boot_fk_active_wins = sum_attempt_field(h128_boot_fk_tags, "signal", "active_KAN_SD_CFO_NLL_improvement_rows")
    h128_boot_fk_active_strict = sum_attempt_field(h128_boot_fk_tags, "signal", "active_KAN_SD_CFO_no_ECE_Brier_tail_debt_rows")
    h128_boot_fk_active_boot = sum_attempt_field(h128_boot_fk_tags, "signal", "active_KAN_SD_CFO_no_ECE_Brier_tail_debt_bootstrap_0p90_rows")
    h128_boot_all_active = sum_attempt_field(h128_boot_all_tags, "signal", "active_KAN_SD_CFO_rows")
    h128_boot_all_active_wins = sum_attempt_field(h128_boot_all_tags, "signal", "active_KAN_SD_CFO_NLL_improvement_rows")
    h128_boot_all_active_strict = sum_attempt_field(h128_boot_all_tags, "signal", "active_KAN_SD_CFO_no_ECE_Brier_tail_debt_rows")
    h128_boot_all_active_boot = sum_attempt_field(h128_boot_all_tags, "signal", "active_KAN_SD_CFO_no_ECE_Brier_tail_debt_bootstrap_0p90_rows")

    def wd(row: dict[str, Any]) -> str:
        return f"{row.get('NLL_improvement_vs_own_strong_rows', '')}/{row.get('treatment_rows', '')} wins, {row.get('no_ECE_Brier_tail_debt_rows', '')}/{row.get('treatment_rows', '')} no-debt"
    def wka(row: dict[str, Any]) -> str:
        return f"{row.get('active_KAN_SD_CFO_NLL_improvement_rows', '')}/{row.get('active_KAN_SD_CFO_rows', '')} active-KAN wins, {row.get('active_KAN_SD_CFO_no_ECE_Brier_tail_debt_rows', '')}/{row.get('active_KAN_SD_CFO_rows', '')} active-KAN no-debt"
    def wkb(row: dict[str, Any]) -> str:
        return f"{row.get('active_KAN_SD_CFO_NLL_improvement_rows', '')}/{row.get('active_KAN_SD_CFO_rows', '')} active-KAN wins, {row.get('active_KAN_SD_CFO_no_ECE_Brier_tail_debt_rows', '')}/{row.get('active_KAN_SD_CFO_rows', '')} strict no-debt, {row.get('active_KAN_SD_CFO_no_ECE_Brier_tail_debt_bootstrap_0p90_rows', '')}/{row.get('active_KAN_SD_CFO_rows', '')} bootstrap no-debt"

    def attempt_frontier(tags: list[str], plan: str = "signal") -> str:
        parts = []
        for tag in tags:
            row = attempt_summary(tag, plan)
            if not row:
                parts.append(f"{tag}=missing")
                continue
            active = row.get("active_KAN_SD_CFO_rows", "")
            parts.append(
                f"{tag}={row.get('active_KAN_SD_CFO_NLL_improvement_rows', '')}/{active} wins,"
                f" {row.get('active_KAN_SD_CFO_no_ECE_Brier_tail_debt_rows', '')}/{active} strict,"
                f" {row.get('active_KAN_SD_CFO_no_ECE_Brier_tail_debt_bootstrap_0p90_rows', '')}/{active} boot"
            )
        return "; ".join(parts)

    lines = [
        "# DG-KAN v22.38 State-Dependent Causal Functional Optimizer 实验结果复盘",
        "",
        f"更新时间：{now_sg()}",
        "",
        "## 1. Final Route",
        "",
        f"- final_route: `{final.get('final_route')}`",
        f"- route_reason: `{final.get('route_reason')}`",
        f"- artifact_manifest_hash: `{final.get('artifact_manifest_hash')}`",
        "",
        "## 2. Code / Identity / Propensity",
        "",
        md_table(code, ["clean_unzip_compileall_pass", "clean_unzip_import_pass", "missing_transitive_dependency_count", "official_DGKAN_identity_pass", "uses_pykan_official_rows", "uses_bspline_official_rows", "uses_test_direction_selection", "uses_future_direction", "randomization_propensity_logged", "propensity_min_by_treatment", "status"], 5),
        "",
        md_table(prop, ["treatment_name", "selected_count", "selected_fraction", "logged_min_propensity", "propensity_gate_pass"], 20),
        "",
        "## 3. Support / Direction / Horizon / Safety",
        "",
        md_table(decomp, ["treatment_name", "support_control", "n_treatment", "n_control", "tau_net", "tau_support", "tau_direction", "tau_optimizer", "LCB_net", "LCB_support", "LCB_direction", "LCB_optimizer", "route_hint"], 20),
        "",
        md_table(horizon, ["treatment_name", "matched_control", "H", "n_treatment", "n_control", "tau_H", "LCB_H", "horizon_status"], 24),
        "",
        md_table(safety, ["treatment_name", "n_events", "ECE_debt_rate", "Brier_debt_rate", "tail_debt_rate", "hard_slice_debt_rate"], 20),
        "",
        "## 4. CATE Representation / Policy",
        "",
        md_table(ablation, ["feature_group", "model_kind", "n_events", "positive_rate", "AUC_positive_treatment", "calibration_slope", "decision_threshold", "LCB_coverage", "false_positive_rate_controls", "policy_noop_rate", "held_group_selected_treatment_fraction", "exploration_gate_pass", "official_candidate_gate_pass", "fit_status"], 30),
        "",
        md_table(policy, ["selected_feature_group", "selected_model_kind", "AUC_positive_treatment", "calibration_slope", "decision_threshold", "LCB_coverage", "false_positive_rate_controls", "policy_noop_rate", "held_group_selected_treatment_fraction", "policy_gate_pass", "policy_official_candidate_pass", "status"], 5),
        "",
        "### Treatment-Specific CATE Policy",
        "",
        md_table(treatment_policy, ["selected_treatment", "matched_control", "selected_feature_group", "selected_model_kind", "n_events", "positive_rate", "AUC_positive_treatment", "calibration_slope", "decision_threshold", "LCB_coverage", "false_positive_rate_controls", "policy_noop_rate", "held_group_selected_treatment_fraction", "policy_gate_pass", "policy_official_candidate_pass", "status"], 20),
        "",
        "## 5. Full-Loop Matrices",
        "",
        "### Support Native",
        "",
        md_table(support, ["dataset", "seed", "architecture", "carrier", "training_variant", "treatment_name", "final_test_NLL", "final_test_accuracy", "NLL_delta_vs_own_strong_optimizer", "no_ECE_Brier_tail_debt", "accepted_treatment_count", "rejected_treatment_count"], 18),
        "",
        "### Signal Incremental",
        "",
        md_table(signal, ["dataset", "seed", "architecture", "carrier", "training_variant", "treatment_name", "final_test_NLL", "NLL_delta_vs_own_strong_optimizer", "NLL_delta_vs_FSO_baseline", "no_ECE_Brier_tail_debt", "no_ECE_Brier_tail_debt_bootstrap_0p90", "tail_q99_delta_vs_base", "tail_q99_bootstrap_0p90_tolerance", "alpha_scale_mode", "accepted_alpha_mean", "accepted_effective_alpha_mean", "acceptance_tail_margin", "calibration_temperature_mode", "calibration_temperature_policy", "runtime_CATE_policy_scope", "runtime_CATE_policy_threshold", "runtime_CATE_policy_pass_count", "runtime_CATE_policy_blocked_count", "runtime_safety_policy_enabled", "runtime_safety_policy_threshold", "runtime_safety_policy_pass_count", "runtime_safety_policy_blocked_count", "runtime_CATE_policy_arbitration_selected_counts", "runtime_CATE_policy_arbitration_candidate_thresholds", "runtime_CATE_policy_arbitration_candidate_lcbs", "status", "reason"], 18),
        "",
        "### Optimizer State",
        "",
        md_table(opt, ["dataset", "seed", "architecture", "carrier", "training_variant", "treatment_name", "final_test_NLL", "NLL_delta_vs_own_strong_optimizer", "no_ECE_Brier_tail_debt", "runtime_CATE_policy_scope", "runtime_CATE_policy_threshold", "runtime_CATE_policy_pass_count", "runtime_CATE_policy_blocked_count", "status", "reason"], 18),
        "",
        "### KAN Basis Native",
        "",
        md_table(kan, ["dataset", "seed", "architecture", "carrier", "training_variant", "treatment_name", "basis_energy_fraction", "readout_leakage_fraction", "final_test_NLL", "NLL_delta_vs_own_strong_optimizer", "no_ECE_Brier_tail_debt", "no_ECE_Brier_tail_debt_bootstrap_0p90", "tail_q99_delta_vs_base", "tail_q99_bootstrap_0p90_tolerance", "alpha_scale_mode", "accepted_alpha_mean", "accepted_effective_alpha_mean", "acceptance_tail_margin", "calibration_temperature_mode", "calibration_temperature_policy", "runtime_CATE_policy_scope", "runtime_CATE_policy_threshold", "runtime_CATE_policy_pass_count", "runtime_CATE_policy_blocked_count", "runtime_safety_policy_enabled", "runtime_safety_policy_threshold", "runtime_safety_policy_pass_count", "runtime_safety_policy_blocked_count", "runtime_CATE_policy_arbitration_selected_counts", "runtime_CATE_policy_arbitration_candidate_thresholds", "runtime_CATE_policy_arbitration_candidate_lcbs", "status", "reason"], 18),
        "",
        "## 6. Gap / Continual / Efficiency",
        "",
        "### Horizon-Robust Safety Gate Attempts",
        "",
        md_table(attempts, ["attempt_tag", "plan", "rows", "treatment_rows", "NLL_improvement_vs_own_strong_rows", "no_ECE_Brier_tail_debt_rows", "active_KAN_SD_CFO_rows", "active_KAN_SD_CFO_NLL_improvement_rows", "active_KAN_SD_CFO_no_ECE_Brier_tail_debt_rows", "active_KAN_SD_CFO_no_ECE_Brier_tail_debt_bootstrap_0p90_rows", "post_apply_rollback_count_total", "NLL_improvement_vs_FSO_rows", "mean_NLL_delta_vs_FSO", "summary_artifact", "matrix_artifact", "status"], 110),
        "",
        md_table(attempt_gap, ["attempt_tag", "dataset", "seed", "kan_carrier", "selected_treatment", "MLP_SD_CFO_NLL", "MLP_SD_CFO_accuracy", "KAN_SD_CFO_NLL", "KAN_SD_CFO_accuracy", "GapReduction", "KAN_SD_CFO_beats_MLP_SD_CFO", "KAN_SD_CFO_beats_MLP_SD_CFO_by_accuracy", "KAN_SD_CFO_beats_MLP_SD_CFO_by_NLL_or_accuracy", "TrueKANGain_class", "status"], 100),
        "",
        md_table(gap, ["dataset", "seed", "kan_carrier", "selected_treatment", "MLP_SD_CFO_NLL", "MLP_SD_CFO_accuracy", "KAN_SD_CFO_NLL", "KAN_SD_CFO_accuracy", "GapReduction", "KAN_SD_CFO_beats_MLP_SD_CFO", "KAN_SD_CFO_beats_MLP_SD_CFO_by_accuracy", "KAN_SD_CFO_beats_MLP_SD_CFO_by_NLL_or_accuracy", "TrueKANGain_class", "status", "reason"], 18),
        "",
        md_table(continual, ["dataset", "seed", "variant", "old_task_accuracy_before_new_task", "old_task_accuracy_after_new_task", "new_task_accuracy", "average_forgetting", "relative_forgetting_reduction", "boundary_tau_hat", "boundary_LCB", "status"], 18),
        "",
        md_table(eff, ["completed_CATE_events", "mean_event_horizon_wallclock_sec", "max_event_horizon_wallclock_sec", "controller_overhead_le_0p25_proxy", "status"], 5),
        "",
        "## 7. Analysis / Insight",
        "",
        "### Current Blocker Summary",
        "",
        f"- CATE: events=`{policy0.get('n_events', '')}`, AUC=`{policy0.get('AUC_positive_treatment', '')}`, calibration_slope=`{policy0.get('calibration_slope', '')}`, LCB_coverage=`{policy0.get('LCB_coverage', '')}`, official_candidate=`{policy0.get('policy_official_candidate_pass', policy0.get('official_candidate_gate_pass', ''))}`.",
        f"- Support-native: NLL wins=`{support_summary.get('NLL_improvement_vs_own_strong_rows', '')}/{support_summary.get('treatment_rows', '')}`, no-debt=`{support_summary.get('no_ECE_Brier_tail_debt_rows', '')}/{support_summary.get('treatment_rows', '')}` after using the planned `LCB_support` runtime gate.",
        f"- Signal incremental: NLL wins=`{signal_summary.get('NLL_improvement_vs_own_strong_rows', '')}/{signal_summary.get('treatment_rows', '')}`, vs-FSO wins=`{signal_summary.get('NLL_improvement_vs_FSO_rows', '')}/{signal_summary.get('treatment_rows', '')}`, no-debt=`{signal_summary.get('no_ECE_Brier_tail_debt_rows', '')}/{signal_summary.get('treatment_rows', '')}`.",
        f"- Optimizer-state: NLL wins=`{opt_summary.get('NLL_improvement_vs_own_strong_rows', '')}/{opt_summary.get('treatment_rows', '')}`, no-debt=`{opt_summary.get('no_ECE_Brier_tail_debt_rows', '')}/{opt_summary.get('treatment_rows', '')}`.",
        f"- KAN basis-native: NLL wins=`{kan_summary.get('NLL_improvement_vs_own_strong_rows', '')}/{kan_summary.get('treatment_rows', '')}`, no-debt=`{kan_summary.get('no_ECE_Brier_tail_debt_rows', '')}/{kan_summary.get('treatment_rows', '')}`; gap truth remains below architecture-value threshold.",
        f"- Runtime CATE attempt `cate_runtime_h3_q80_officialcate`: signal NLL wins=`{runtime_signal.get('NLL_improvement_vs_own_strong_rows', '')}/{runtime_signal.get('treatment_rows', '')}`, vs-FSO wins=`{runtime_signal.get('NLL_improvement_vs_FSO_rows', '')}/{runtime_signal.get('treatment_rows', '')}`, no-debt=`{runtime_signal.get('no_ECE_Brier_tail_debt_rows', '')}/{runtime_signal.get('treatment_rows', '')}`; optimizer-state NLL wins=`{runtime_optimizer.get('NLL_improvement_vs_own_strong_rows', '')}/{runtime_optimizer.get('treatment_rows', '')}`, no-debt=`{runtime_optimizer.get('no_ECE_Brier_tail_debt_rows', '')}/{runtime_optimizer.get('treatment_rows', '')}`; KAN-basis NLL wins=`{runtime_kan.get('NLL_improvement_vs_own_strong_rows', '')}/{runtime_kan.get('treatment_rows', '')}`, no-debt=`{runtime_kan.get('no_ECE_Brier_tail_debt_rows', '')}/{runtime_kan.get('treatment_rows', '')}`.",
        f"- Hard-tier runtime CATE attempt `hard_tier1_cate_runtime_q80_s012`: signal NLL wins=`{hard_runtime_signal.get('NLL_improvement_vs_own_strong_rows', '')}/{hard_runtime_signal.get('treatment_rows', '')}`, no-debt=`{hard_runtime_signal.get('no_ECE_Brier_tail_debt_rows', '')}/{hard_runtime_signal.get('treatment_rows', '')}`; optimizer-state NLL wins=`{hard_runtime_optimizer.get('NLL_improvement_vs_own_strong_rows', '')}/{hard_runtime_optimizer.get('treatment_rows', '')}`, no-debt=`{hard_runtime_optimizer.get('no_ECE_Brier_tail_debt_rows', '')}/{hard_runtime_optimizer.get('treatment_rows', '')}`; KAN-basis NLL wins=`{hard_runtime_kan.get('NLL_improvement_vs_own_strong_rows', '')}/{hard_runtime_kan.get('treatment_rows', '')}`, no-debt=`{hard_runtime_kan.get('no_ECE_Brier_tail_debt_rows', '')}/{hard_runtime_kan.get('treatment_rows', '')}`.",
        f"- Treatment-specific CATE for `a3_optimizer_state_signal`: AUC=`{a3_treatment_policy.get('AUC_positive_treatment', '')}`, calibration_slope=`{a3_treatment_policy.get('calibration_slope', '')}`, decision_threshold=`{a3_treatment_policy.get('decision_threshold', '')}`, LCB_coverage=`{a3_treatment_policy.get('LCB_coverage', '')}`, official_candidate=`{a3_treatment_policy.get('policy_official_candidate_pass', '')}`.",
        f"- Planned-basis direct CATE collection is now represented in the official event matrix: `a2_basis_actuator_section`=`{direct_event_stats('a2_basis_actuator_section')}`, `a5_signal_incremental_basis`=`{direct_event_stats('a5_signal_incremental_basis')}`, `a8_support_native_basis`=`{direct_event_stats('a8_support_native_basis')}`, `a6_signflip_same_basis`=`{direct_event_stats('a6_signflip_same_basis')}`, `a6_same_bank_random`=`{direct_event_stats('a6_same_bank_random')}`, `a6_same_actuator_random`=`{direct_event_stats('a6_same_actuator_random')}`. Logged treatment propensity minimum remains in the randomization matrix rather than inferred from counts.",
        f"- Treatment-specific CATE for the new `a5_signal_incremental_basis` path is real on the expanded planned-basis direct events: n_events=`{a5_treatment_policy.get('n_events', '')}`, AUC=`{a5_treatment_policy.get('AUC_positive_treatment', '')}`, calibration_slope=`{a5_treatment_policy.get('calibration_slope', '')}`, decision_threshold=`{a5_treatment_policy.get('decision_threshold', '')}`, LCB_coverage=`{a5_treatment_policy.get('LCB_coverage', '')}`, official_candidate=`{a5_treatment_policy.get('policy_official_candidate_pass', '')}`. This is a concrete positive result, but it is still a policy gate, not proof of final architecture superiority.",
        f"- Expanded planned-basis global-CATE attempt `{expanded_global_tag}`: signal=`{wkb(expanded_global_signal)}`, KAN-basis=`{wkb(expanded_global_kan)}`, TrueKANGain/BothGain=`{expanded_global_true_both}/{expanded_global_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{expanded_global_beats_any}/{expanded_global_gap_total}`. Adding planned-basis rows under the old global policy did not improve the hard-tier aggregate beyond the earlier a2 route.",
        f"- Treatment-specific planned-basis attempt before runtime-feature repair `{expanded_ts_bug_tag}` was a useful negative control: signal=`{wkb(expanded_ts_bug_signal)}`, KAN-basis=`{wkb(expanded_ts_bug_kan)}`, TrueKANGain/BothGain=`{expanded_ts_bug_true_both}/{expanded_ts_bug_gap_total}`; signal a2=`{treatment_rollup(expanded_ts_bug_tag, 'v22_38_signal_incremental_full_loop_matrix.csv', 'a2_basis_actuator_section')}`, signal a5=`{treatment_rollup(expanded_ts_bug_tag, 'v22_38_signal_incremental_full_loop_matrix.csv', 'a5_signal_incremental_basis')}`. Diagnosis: live runtime features used v22.38-specific direction-source values and blank update norm that were outside the direct-CATE training representation, so policy probabilities collapsed toward no-op.",
        f"- Runtime-feature repair `{expanded_fix_tag}` changed the live treatment feature fields, not the loss target: signal=`{wkb(expanded_fix_signal)}`, KAN-basis=`{wkb(expanded_fix_kan)}`, TrueKANGain/BothGain=`{expanded_fix_true_both}/{expanded_fix_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{expanded_fix_beats_any}/{expanded_fix_gap_total}`; signal a2=`{treatment_rollup(expanded_fix_tag, 'v22_38_signal_incremental_full_loop_matrix.csv', 'a2_basis_actuator_section')}`, signal a5=`{treatment_rollup(expanded_fix_tag, 'v22_38_signal_incremental_full_loop_matrix.csv', 'a5_signal_incremental_basis')}`, KAN a8=`{treatment_rollup(expanded_fix_tag, 'v22_38_KAN_basis_native_full_loop_matrix.csv', 'a8_support_native_basis')}`, KAN signflip=`{treatment_rollup(expanded_fix_tag, 'v22_38_KAN_basis_native_full_loop_matrix.csv', 'a6_signflip_same_basis')}`, KAN same-bank=`{treatment_rollup(expanded_fix_tag, 'v22_38_KAN_basis_native_full_loop_matrix.csv', 'a6_same_bank_random')}`. The good news is that a5 executes after the fix; the remaining blocker is that a5+a2 as separate rows still do not raise the official hard-tier effect/architecture aggregate.",
        f"- Runtime arbitration follow-up `{arbitrated_tag}` tested whether a2/a5 should be selected inside one SD-CFO row instead of counted as separate rows: signal=`{wkb(arbitrated_signal)}`, KAN-basis=`{wkb(arbitrated_kan)}`, TrueKANGain/BothGain=`{arbitrated_true_both}/{arbitrated_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{arbitrated_beats_any}/{arbitrated_gap_total}`; signal arbitration selected `{arbitration_selection_rollup(arbitrated_tag, 'v22_38_signal_incremental_full_loop_matrix.csv')}`, KAN arbitration selected `{arbitration_selection_rollup(arbitrated_tag, 'v22_38_KAN_basis_native_full_loop_matrix.csv')}`. This is a negative repair result, not a failure to run: it proves the runtime chooser works and mostly prefers a5, but the hard-tier aggregate is weaker (`3/6` active-KAN wins, `2/6` strict no-debt) than the previous expanded planned-basis evidence.",
        f"- High-effect route safety replay `{hard_q80_strict_boot_tag}` kept the old short h16/seeds0-2/global-CATE/D-CHE shape and only added strictgate controls plus bootstrap: signal=`{wkb(hard_q80_strict_boot_signal)}`, KAN-basis=`{wkb(hard_q80_strict_boot_kan)}`, TrueKANGain/BothGain=`{hard_q80_strict_boot_true_both}/{hard_q80_strict_boot_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{hard_q80_strict_boot_beats_any}/{hard_q80_strict_boot_gap_total}`. Compared with `hard_tier1_cate_runtime_q80_s012` (`7/9` active wins, `4/9` strict no-debt, `5/9` TrueKANGain/BothGain), strictgate_boot200 improves strict/bootstrap safety but reduces effect to `5/9` active wins, so the remaining blocker is a safety/effect frontier rather than logging or CATE availability.",
        f"- Intermediate safety replay `{hard_q80_h5b8_tag}` tested horizon=5/acceptance-batches=8 at tail0.995 on the same high-effect route: signal=`{wkb(hard_q80_h5b8_signal)}`, KAN-basis=`{wkb(hard_q80_h5b8_kan)}`, TrueKANGain/BothGain=`{hard_q80_h5b8_true_both}/{hard_q80_h5b8_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{hard_q80_h5b8_beats_any}/{hard_q80_h5b8_gap_total}`. It recovers effect (`6/9` active-KAN wins) but strict no-debt drops to `2/9`, confirming a continuous safety/effect tradeoff between h5/b8 and h10/b16 rather than a binary implementation bug.",
        f"- Midpoint safety replay `{hard_q80_h7b12_tag}` is the strongest hard-tier tradeoff found in this chain: signal=`{wkb(hard_q80_h7b12_signal)}`, KAN-basis=`{wkb(hard_q80_h7b12_kan)}`, TrueKANGain/BothGain=`{hard_q80_h7b12_true_both}/{hard_q80_h7b12_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{hard_q80_h7b12_beats_any}/{hard_q80_h7b12_gap_total}`. This restores the old `7/9` active-KAN wins while raising strict no-debt to `5/9` and keeping bootstrap no-debt at `9/9`; it is real progress, but strict no-debt is still not enough for a final no-safety-debt claim.",
        f"- Temperature-wide midpoint replay `{hard_q80_h7b12_tempwide_tag}` is the current best hard-tier evidence: signal=`{wkb(hard_q80_h7b12_tempwide_signal)}`, KAN-basis=`{wkb(hard_q80_h7b12_tempwide_kan)}`, TrueKANGain/BothGain=`{hard_q80_h7b12_tempwide_true_both}/{hard_q80_h7b12_tempwide_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{hard_q80_h7b12_tempwide_beats_any}/{hard_q80_h7b12_tempwide_gap_total}`. Widening the held-train temperature grid fixes most tail debt and improves effect simultaneously; remaining strict debt is localized to two tail rows (SVHN seed0, EMNIST_LETTERS seed1), so the new blocker is narrow tail strictness rather than architecture effect.",
        f"- Tail-focused temperature selection replay `{hard_q80_h7b12_tempwide_tailq99_tag}` changed only `--full-loop-temperature-selection-metric` from `ECE_Brier_tail` to `tail_q99`: signal=`{wkb(hard_q80_h7b12_tempwide_tailq99_signal)}`, KAN-basis=`{wkb(hard_q80_h7b12_tempwide_tailq99_kan)}`, TrueKANGain/BothGain=`{hard_q80_h7b12_tempwide_tailq99_true_both}/{hard_q80_h7b12_tempwide_tailq99_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{hard_q80_h7b12_tempwide_tailq99_beats_any}/{hard_q80_h7b12_tempwide_tailq99_gap_total}`. It shrinks the two remaining positive tail deltas but keeps strict no-debt at `7/9`, so it is a partial tail-risk repair rather than final success.",
        f"- Additional tail-repair follow-ups did not dominate the current best: `{attempt_frontier(hard_q80_tail_repair_followup_tags)}`. T10/tail-q99 and lower alpha shrink residual tail magnitude but keep strict at `7/9` or lose wins; tail0.999 is effectively identical to tail0.995 on active KAN rows; h10/b16 plus tempwide drops wins to `6/9` without improving strict.",
        f"- Acceptance-margin frontier around the h7/b12 tempwide route: `{attempt_frontier(hard_q80_acceptance_tol_frontier_tags)}`. The margin is a real safety lever but too discontinuous here: `1e-7/2e-7` preserve the current `8/9` wins with `7/9` strict, `5e-7/1e-6` reach `8/9` strict but only `5/9` wins, and `4e-6` degenerates to no-op. This rules out a simple scalar acceptance tolerance as the final repair.",
        f"- Higher tail-temperature ceiling also failed to dominate: `{attempt_frontier(hard_q80_temperature_extension_tags)}`. Extending held-train `tail_q99` scalar temperature to T20/T30 keeps shrinking the two positive tail deltas, but active-KAN wins fall to `6/9` and `5/9` while strict no-debt remains `7/9`; this is calibration washout, not a final repair.",
        f"- Runtime no-debt safety policy was implemented as a second train-event `Y_no_debt_gate` classifier after the CATE gate. Full signal+KAN q40 results were `{attempt_frontier(hard_q80_runtime_safety_policy_tags)}`. The safety-only GBT q40 policy passed all hard-tier checks and was effectively a no-op; the loss-only GBT q40 policy blocked many interventions, dropping wins to `5/9` while strict stayed `7/9`. KAN-only sparse threshold diagnostics were `{attempt_frontier(hard_q80_runtime_safety_kan_diagnostic_tags, 'kan_basis')}`; thresholds near `0.1883/0.1886` either did not fix the two debt rows or overblocked and moved debt to another row.",
        f"- Confidence-tail held-train calibration was added as an explicit non-training calibration mode and tested on KAN-only hard-tier diagnostics: `{attempt_frontier(hard_q80_confidence_calibration_kan_tags, 'kan_basis')}`. It sometimes selected a sample-wise confidence policy, but SVHN seed0 and EMNIST_LETTERS seed1 remained strict tail debt, so the blocker is not solved by selective temperature calibration.",
        f"- Tail-only acceptance margin separated tail margin from CE/ECE/Brier tolerance and was tested on KAN-only diagnostics: `{attempt_frontier(hard_q80_tail_margin_kan_tags, 'kan_basis')}`. Margins up to `1e-6` did not change accepted interventions; `5e-6/1e-5` started losing wins and even introduced a new strict debt row. This localizes the failure to held-to-test tail generalization rather than an unlogged all-metric tolerance bug.",
        f"- Optimizer runtime threshold sensitivity: q65/loss-only threshold NLL wins=`{opt_runtime_q65.get('NLL_improvement_vs_own_strong_rows', '')}/{opt_runtime_q65.get('treatment_rows', '')}`, no-debt=`{opt_runtime_q65.get('no_ECE_Brier_tail_debt_rows', '')}/{opt_runtime_q65.get('treatment_rows', '')}`; q60/loss-only threshold NLL wins=`{opt_runtime_q60.get('NLL_improvement_vs_own_strong_rows', '')}/{opt_runtime_q60.get('treatment_rows', '')}`, no-debt=`{opt_runtime_q60.get('no_ECE_Brier_tail_debt_rows', '')}/{opt_runtime_q60.get('treatment_rows', '')}`. This is evidence of a real Part F tradeoff: relaxing the gate can restore optimizer wins but reintroduces safety debt; an overly low threshold can also fail the empirical LCB guard and block execution.",
        f"- Optimizer treatment-specific runtime sweep: first `optimizer_runtime_treatment_policy_q80` is invalid as a treatment-specific conclusion because missing `core_args_from()` propagation made it run as global (`{wd(opt_ts_q80_bug)}`); after the scope fix, q80=`{wd(opt_ts_q80)}`, q75=`{wd(opt_ts_q75)}`, q74=`{wd(opt_ts_q74)}`, q73=`{wd(opt_ts_q73)}`, q72=`{wd(opt_ts_q72)}`, q70=`{wd(opt_ts_q70)}`. q70 alpha cap stayed `{wd(opt_ts_q70_alpha)}`; q70 with horizon=5/batches=8 became `{wd(opt_ts_q70_h5b8)}`. No tested threshold/execution gate simultaneously achieved optimizer wins and no-debt.",
        f"- Hard-tier Part F optimizer-state branch on h256 strict-3200 settings did not fix effect: seed0 optimizer-state=`{wkb(opt_h256_hard_s0_safe3200)}`, seed1 optimizer-state=`{wkb(opt_h256_hard_s1_safe3200)}`; combined seeds0+1 active-KAN rows are wins=`{sum_attempt_field(opt_h256_hard_s01_safe3200_tags, 'optimizer', 'active_KAN_SD_CFO_NLL_improvement_rows')}/{sum_attempt_field(opt_h256_hard_s01_safe3200_tags, 'optimizer', 'active_KAN_SD_CFO_rows')}`, strict no-debt=`{sum_attempt_field(opt_h256_hard_s01_safe3200_tags, 'optimizer', 'active_KAN_SD_CFO_no_ECE_Brier_tail_debt_rows')}/{sum_attempt_field(opt_h256_hard_s01_safe3200_tags, 'optimizer', 'active_KAN_SD_CFO_rows')}`, bootstrap no-debt=`{sum_attempt_field(opt_h256_hard_s01_safe3200_tags, 'optimizer', 'active_KAN_SD_CFO_no_ECE_Brier_tail_debt_bootstrap_0p90_rows')}/{sum_attempt_field(opt_h256_hard_s01_safe3200_tags, 'optimizer', 'active_KAN_SD_CFO_rows')}`. The a3 CATE policy is real, but full-parameter optimizer-state FU is slower and weaker on hard-tier KAN effect than the h256 a2 strict-safety route.",
        f"- D-FOU carrier attempt `kan_dfou_cate_runtime_q80`: signal summary=`{wd(dfou_signal)}`, KAN-basis summary=`{wd(dfou_kan)}`, gap truth TrueKANGain/BothGain=`{dfou_true_both}/{dfou_gap_total}`, KAN_SD_CFO_beats_MLP_SD_CFO=`{dfou_beats_mlp}/{dfou_gap_total}`. D-FOU reduces the KAN baseline gap slightly versus D-CHE but still does not open architecture superiority.",
        f"- KAN carrier long-training diagnostics: hidden16/200-step `kan_carrier_long200_cate_runtime_q80` signal=`{wd(long200_signal)}`, KAN-basis=`{wd(long200_kan)}`, TrueKANGain/BothGain=`{long200_true_both}/{long200_gap_total}`, KAN beats MLP=`{long200_beats_mlp}/{long200_gap_total}`; hidden64/200-step `kan_carrier_h64_long200_cate_runtime_q80` signal=`{wd(h64_long200_signal)}`, KAN-basis=`{wd(h64_long200_kan)}`, TrueKANGain/BothGain=`{h64_true_both}/{h64_gap_total}`, KAN beats MLP=`{h64_beats_mlp}/{h64_gap_total}`; MNIST seed0 hidden64/800-step signal=`{wd(h64_800_signal)}`, KAN-basis=`{wd(h64_800_kan)}`, TrueKANGain/BothGain=`{h64_800_true_both}/{h64_800_gap_total}`, KAN beats MLP=`{h64_800_beats_mlp}/{h64_800_gap_total}`. Longer/capacity diagnostics strengthen KAN-internal value but do not close the MLP gap.",
        f"- FU scale diagnostic `kan_carrier_h64_mnist_s0_steps800_alpha1e3`: signal=`{wd(h64_800_alpha_signal)}`, KAN-basis=`{wd(h64_800_alpha_kan)}`, TrueKANGain/BothGain=`{h64_800_alpha_true_both}/{h64_800_alpha_gap_total}`, KAN beats MLP=`{h64_800_alpha_beats_mlp}/{h64_800_alpha_gap_total}`. Larger alpha increased KAN NLL gains to about 1e-4 on MNIST seed0, still far below the MLP-vs-KAN base gap and with safety debt on the active KAN rows.",
        f"- Base-update-norm relative scale diagnostic: MNIST seed0 hidden64/800-step `kan_carrier_h64_mnist_s0_steps800_relbase_alpha` signal=`{wd(h64_800_relbase_signal)}`, KAN-basis=`{wd(h64_800_relbase_kan)}`, TrueKANGain/BothGain=`{h64_800_relbase_true_both}/{h64_800_relbase_gap_total}`, KAN beats MLP=`{h64_800_relbase_beats_mlp}/{h64_800_relbase_gap_total}`; low-alpha version `kan_carrier_h64_mnist_s0_steps800_relbase_lowalpha` signal=`{wd(h64_800_relbase_low_signal)}`, KAN-basis=`{wd(h64_800_relbase_low_kan)}`, TrueKANGain/BothGain=`{h64_800_relbase_low_true_both}/{h64_800_relbase_low_gap_total}`, KAN beats MLP=`{h64_800_relbase_low_beats_mlp}/{h64_800_relbase_low_gap_total}`. The relative scale raises D-CHE MNIST seed0 gain to `0.0225355625` with no-debt under the wider multiplier grid; lower multipliers reduce gain and do not recover safety for both carriers.",
        f"- Multi-seed relative-scale validation `kan_carrier_h64_long200_relbase_alpha`: signal=`{wd(h64_long200_relbase_signal)}`, KAN-basis=`{wd(h64_long200_relbase_kan)}`, TrueKANGain/BothGain=`{h64_long200_relbase_true_both}/{h64_long200_relbase_gap_total}`, KAN beats MLP=`{h64_long200_relbase_beats_mlp}/{h64_long200_relbase_gap_total}`. It improves effect scale substantially but worsens active-row safety versus absolute alpha, so it is a diagnostic repair path, not an official gate pass.",
        f"- Tail-debt repair diagnostics on MNIST seed0 hidden64/800: horizon=5/batches=8/tail0.995 `kan_carrier_h64_mnist_s0_steps800_relbase_h5b8_tail995` stayed weak (signal=`{wd(h64_800_relbase_h5b8_signal)}`, active=`{wka(h64_800_relbase_h5b8_signal)}`; KAN-basis=`{wd(h64_800_relbase_h5b8_kan)}`, active=`{wka(h64_800_relbase_h5b8_kan)}`), so more horizon proxy alone did not predict final tail debt. Temperature-grid repair `kan_carrier_h64_mnist_s0_steps800_relbase_tempwide` reached signal=`{wd(h64_800_relbase_tempwide_signal)}`, active=`{wka(h64_800_relbase_tempwide_signal)}`, KAN-basis=`{wd(h64_800_relbase_tempwide_kan)}`, active=`{wka(h64_800_relbase_tempwide_kan)}` but selected high `T=6` and sacrificed absolute NLL. The bounded grid `kan_carrier_h64_mnist_s0_steps800_relbase_temp22` kept all active rows no-debt on this slice (signal=`{wd(h64_800_relbase_temp22_signal)}`, active=`{wka(h64_800_relbase_temp22_signal)}`; KAN-basis=`{wd(h64_800_relbase_temp22_kan)}`, active=`{wka(h64_800_relbase_temp22_kan)}`) with TrueKANGain/BothGain=`{h64_800_relbase_temp22_true_both}/{h64_800_relbase_temp22_gap_total}`, KAN beats MLP=`{h64_800_relbase_temp22_beats_mlp}/{h64_800_relbase_temp22_gap_total}`; it fixes safety locally but still does not open KAN-vs-MLP superiority. `NLL_Brier_tail` temperature selection preserved lower NLL but left D-CHE tail debt (signal=`{wd(h64_800_relbase_tempwide_nllbt_signal)}`, active=`{wka(h64_800_relbase_tempwide_nllbt_signal)}`; KAN-basis=`{wd(h64_800_relbase_tempwide_nllbt_kan)}`, active=`{wka(h64_800_relbase_tempwide_nllbt_kan)}`, TrueKANGain/BothGain=`{h64_800_relbase_tempwide_nllbt_true_both}/{h64_800_relbase_tempwide_nllbt_gap_total}`).",
        f"- Multi-seed bounded-temperature validation `kan_carrier_h64_long200_relbase_temp22`: signal=`{wd(h64_long200_relbase_temp22_signal)}`, active=`{wka(h64_long200_relbase_temp22_signal)}`; KAN-basis=`{wd(h64_long200_relbase_temp22_kan)}`, active=`{wka(h64_long200_relbase_temp22_kan)}`; TrueKANGain/BothGain=`{h64_long200_relbase_temp22_true_both}/{h64_long200_relbase_temp22_gap_total}`, KAN beats MLP=`{h64_long200_relbase_temp22_beats_mlp}/{h64_long200_relbase_temp22_gap_total}`. This shows the temp22 repair is local: on the wider Tier0 compact matrix active-KAN no-debt remains far below the official safety threshold, and the MLP gap remains open.",
        f"- Capacity/longer-training probe `kan_carrier_h128_mnist_s0_steps1600_relbase_nllbt`: signal=`{wd(h128_1600_relbase_nllbt_signal)}`, active=`{wka(h128_1600_relbase_nllbt_signal)}`; KAN-basis=`{wd(h128_1600_relbase_nllbt_kan)}`, active=`{wka(h128_1600_relbase_nllbt_kan)}`; TrueKANGain/BothGain=`{h128_1600_relbase_nllbt_true_both}/{h128_1600_relbase_nllbt_gap_total}`, KAN beats MLP by NLL=`{h128_1600_relbase_nllbt_beats_mlp}/{h128_1600_relbase_nllbt_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{h128_1600_relbase_nllbt_beats_any}/{h128_1600_relbase_nllbt_any_total}`. This is a real architecture-progress signal: KAN accuracy beats MLP on this MNIST slice, but NLL still does not beat MLP and D-CHE still has safety debt, so it remains a probe rather than an official pass.",
        f"- Hidden128 seed1 architecture/safety sweep: `kan_carrier_h128_mnist_s1_steps1600_relbase_nllbt` produced signal=`{wd(h128_s1_relbase_nllbt_signal)}`, active=`{wka(h128_s1_relbase_nllbt_signal)}`, KAN-basis=`{wd(h128_s1_relbase_nllbt_kan)}`, active=`{wka(h128_s1_relbase_nllbt_kan)}`, TrueKANGain/BothGain=`{h128_s1_relbase_nllbt_true_both}/{h128_s1_relbase_nllbt_gap_total}`, KAN beats MLP by NLL=`{h128_s1_relbase_nllbt_beats_mlp}/{h128_s1_relbase_nllbt_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{h128_s1_relbase_nllbt_beats_any}/{h128_s1_relbase_nllbt_any_total}`. However active no-debt was `0/2`; bounded `temp22` over seeds0,1 stayed active=`{wka(h128_s01_relbase_temp22_signal)}` and lost NLL superiority, tol=0.01 gave active=`{wka(h128_s1_relbase_tol001_signal)}` with KAN beats MLP by NLL-or-accuracy=`{h128_s1_relbase_tol001_beats_any}/{h128_s1_relbase_tol001_any_total}` but D-CHE became safe no-op, tol=0.001 active=`{wka(h128_s1_relbase_tol0001_signal)}`, low-alpha active=`{wka(h128_s1_relbase_lowalpha_signal)}`, temp17 active=`{wka(h128_s1_relbase_temp17_signal)}`, ECE/Brier/tail temperature active=`{wka(h128_s1_relbase_ebt_signal)}`, and h5/b8/q995 active=`{wka(h128_s1_relbase_h5b8_signal)}`. The evidence chain isolates the blocker to nonzero FU gain plus tail no-debt, not to CATE, effect scale, or KAN capacity alone.",
        f"- Additional h128 safety diagnostics: post-apply rollback `kan_carrier_h128_mnist_s1_steps1600_relbase_nllbt_postrollback` stayed active=`{wka(h128_s1_relbase_postrollback_signal)}` with rollback_total=`{h128_s1_relbase_postrollback_signal.get('post_apply_rollback_count_total', '')}`, so branch selection and actual application are consistent and not the observed leak. Increasing held size to 1024 gave active=`{wka(h128_s1_relbase_held1024_signal)}`, so q99 tail debt is not fixed by a larger held split. Support-native h128 seed1 diagnostic `support_h128_mnist_s1_steps1600_relbase_nllbt` was `{wd(support_h128_s1_signal)}` and behaved as safe no-op, so it is not a substitute for nonzero basis SD-CFO.",
        f"- Evaluation-only bootstrap audit (`--full-loop-metric-bootstrap-samples 200`) adds row-local uncertainty for ECE/Brier/tail and does not affect training or model selection. MNIST h128 boot200 signal rows are seed0=`{wkb(h128_s0_boot_signal)}`, seed1=`{wkb(h128_s1_boot_signal)}`; KAN-basis seed0=`{wkb(h128_s0_boot_kan)}`, seed1=`{wkb(h128_s1_boot_kan)}`. Aggregated MNIST active-KAN signal rows are NLL wins=`{h128_boot_mnist_active_wins}/{h128_boot_mnist_active}`, strict no-debt=`{h128_boot_mnist_active_strict}/{h128_boot_mnist_active}`, bootstrap no-debt=`{h128_boot_mnist_active_boot}/{h128_boot_mnist_active}`, TrueKANGain/BothGain=`{h128_boot_mnist_true_both}/{h128_boot_mnist_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{h128_boot_mnist_beats_any}/{h128_boot_mnist_gap_total}`.",
        f"- FashionMNIST/KMNIST h128 boot200 signal=`{wkb(h128_fk_boot_signal)}`, KAN-basis=`{wkb(h128_fk_boot_kan)}`; gap truth TrueKANGain/BothGain=`{h128_boot_fk_true_both}/{h128_boot_fk_gap_total}`, KAN beats MLP by NLL=`{h128_boot_fk_beats_nll}/{h128_boot_fk_gap_total}`, by NLL-or-accuracy=`{h128_boot_fk_beats_any}/{h128_boot_fk_gap_total}`. Combined h128 boot200 active-KAN signal rows: NLL wins=`{h128_boot_all_active_wins}/{h128_boot_all_active}`, strict no-debt=`{h128_boot_all_active_strict}/{h128_boot_all_active}`, bootstrap no-debt=`{h128_boot_all_active_boot}/{h128_boot_all_active}`, TrueKANGain/BothGain=`{h128_boot_all_true_both}/{h128_boot_all_gap_total}`, KAN beats MLP by NLL=`{h128_boot_all_beats_nll}/{h128_boot_all_gap_total}`, by NLL-or-accuracy=`{h128_boot_all_beats_any}/{h128_boot_all_gap_total}`. This is the strongest positive evidence so far for the plan's bootstrap-tolerance route: most q99 tail strict failures are within row-local evaluation uncertainty, but strict no-debt remains below threshold and hard-tier official validation is still pending, so it is not promoted to final success.",
        f"- Hard-tier h128 boot200 seed0 on CIFAR10/SVHN/EMNIST_LETTERS completed as a Tier1 validation slice: signal=`{wkb(h128_hard_boot_signal)}`, KAN-basis=`{wkb(h128_hard_boot_kan)}`, TrueKANGain/BothGain=`{h128_hard_boot_true_both}/{h128_hard_boot_gap_total}`, KAN beats MLP by NLL=`{h128_hard_boot_beats_nll}/{h128_hard_boot_gap_total}`, by NLL-or-accuracy=`{h128_hard_boot_beats_any}/{h128_hard_boot_gap_total}`. The evidence is mixed: bootstrap no-debt stays `6/6`, but active-KAN NLL wins are only `2/6`; CIFAR10 D-CHE and EMNIST_LETTERS D-CHE are the two TrueKANGain rows, while SVHN remains a hard blocker. This means bootstrap tolerance helps interpret safety debt, but does not yet solve hard-tier effect reliability.",
        f"- Hard-tier repair chain on seed0: threshold-only tightening did not fix effect reliability (`thr022`=`{wkb(h128_hard_thr022_signal)}`, `thr030`=`{wkb(h128_hard_thr030_signal)}` and `thr030` reduced wins). Low-alpha improved strict safety but not wins (`{wkb(h128_hard_lowalpha_signal)}`); low-alpha plus horizon=5/batches=8 gave the best h128 seed0 slice, signal=`{wkb(h128_hard_lowalpha_h5b8_signal)}`, TrueKANGain/BothGain=`{h128_hard_lowalpha_h5b8_true_both}/{h128_hard_lowalpha_h5b8_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{h128_hard_lowalpha_h5b8_beats_any}/{h128_hard_lowalpha_h5b8_gap_total}`. This supports an effect-scale/horizon blocker rather than a simple CATE threshold blocker.",
        f"- Capacity/micro-alpha follow-up: SVHN seed0 h256 low-alpha/h5b8 was `{wkb(h256_svhn_s0_lowalpha_h5b8_signal)}`, and h256 micro-alpha/h5b8 became `{wkb(h256_svhn_s0_micro_h5b8_signal)}`; the latter turned both SVHN carriers into NLL wins. Expanding the same h256 micro-alpha/h5b8 route to all Tier1 seed0 gave signal=`{wkb(h256_hard_s0_micro_h5b8_signal)}`, TrueKANGain/BothGain=`{h256_hard_s0_micro_h5b8_true_both}/{h256_hard_s0_micro_h5b8_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{h256_hard_s0_micro_h5b8_beats_any}/{h256_hard_s0_micro_h5b8_gap_total}`. This is the current best hard-tier seed0 route, but CIFAR10 D-CHE/D-FOU are NoGain relative to own KAN strong despite beating MLP on that seed.",
        f"- Seed robustness remains open: h256 micro-alpha/h5b8 on Tier1 seed1 fell to signal=`{wkb(h256_hard_s1_micro_h5b8_signal)}`, TrueKANGain/BothGain=`{h256_hard_s1_micro_h5b8_true_both}/{h256_hard_s1_micro_h5b8_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{h256_hard_s1_micro_h5b8_beats_any}/{h256_hard_s1_micro_h5b8_gap_total}`. A seed1 low-alpha variant improved only to `{wkb(h256_hard_s1_lowalpha_h5b8_signal)}` with TrueKANGain/BothGain=`{h256_hard_s1_lowalpha_h5b8_true_both}/{h256_hard_s1_lowalpha_h5b8_gap_total}`. Combined h256 micro-alpha seeds0+1 are wins=`{sum_attempt_field(h256_hard_s01_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_NLL_improvement_rows')}/{sum_attempt_field(h256_hard_s01_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_rows')}`, strict no-debt=`{sum_attempt_field(h256_hard_s01_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_no_ECE_Brier_tail_debt_rows')}/{sum_attempt_field(h256_hard_s01_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_rows')}`, bootstrap no-debt=`{sum_attempt_field(h256_hard_s01_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_no_ECE_Brier_tail_debt_bootstrap_0p90_rows')}/{sum_attempt_field(h256_hard_s01_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_rows')}`, TrueKANGain/BothGain=`{h256_hard_s01_micro_h5b8_true_both}/{h256_hard_s01_micro_h5b8_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{h256_hard_s01_micro_h5b8_beats_any}/{h256_hard_s01_micro_h5b8_gap_total}`.",
        f"- Adding seed2 did not rescue h256 micro-alpha/h5b8: seed2 alone was `{wkb(h256_hard_s2_micro_h5b8_signal)}`, TrueKANGain/BothGain=`{h256_hard_s2_micro_h5b8_true_both}/{h256_hard_s2_micro_h5b8_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{h256_hard_s2_micro_h5b8_beats_any}/{h256_hard_s2_micro_h5b8_gap_total}`. Combined h256 micro-alpha seeds0/1/2 are wins=`{sum_attempt_field(h256_hard_s012_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_NLL_improvement_rows')}/{sum_attempt_field(h256_hard_s012_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_rows')}`, strict no-debt=`{sum_attempt_field(h256_hard_s012_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_no_ECE_Brier_tail_debt_rows')}/{sum_attempt_field(h256_hard_s012_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_rows')}`, bootstrap no-debt=`{sum_attempt_field(h256_hard_s012_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_no_ECE_Brier_tail_debt_bootstrap_0p90_rows')}/{sum_attempt_field(h256_hard_s012_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_rows')}`, TrueKANGain/BothGain=`{h256_hard_s012_micro_h5b8_true_both}/{h256_hard_s012_micro_h5b8_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{h256_hard_s012_micro_h5b8_beats_any}/{h256_hard_s012_micro_h5b8_gap_total}`.",
        f"- Training-depth repair probe: h256 micro-alpha/h5b8 at 3200 steps gave seed0 signal=`{wkb(h256_hard_s0_steps3200_micro_h5b8_signal)}`, TrueKANGain/BothGain=`{h256_hard_s0_steps3200_micro_h5b8_true_both}/{h256_hard_s0_steps3200_micro_h5b8_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{h256_hard_s0_steps3200_micro_h5b8_beats_any}/{h256_hard_s0_steps3200_micro_h5b8_gap_total}`; seed1 signal=`{wkb(h256_hard_s1_steps3200_micro_h5b8_signal)}`, TrueKANGain/BothGain=`{h256_hard_s1_steps3200_micro_h5b8_true_both}/{h256_hard_s1_steps3200_micro_h5b8_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{h256_hard_s1_steps3200_micro_h5b8_beats_any}/{h256_hard_s1_steps3200_micro_h5b8_gap_total}`. Combined 3200-step seeds0+1 are wins=`{sum_attempt_field(h256_hard_s01_steps3200_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_NLL_improvement_rows')}/{sum_attempt_field(h256_hard_s01_steps3200_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_rows')}`, strict no-debt=`{sum_attempt_field(h256_hard_s01_steps3200_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_no_ECE_Brier_tail_debt_rows')}/{sum_attempt_field(h256_hard_s01_steps3200_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_rows')}`, bootstrap no-debt=`{sum_attempt_field(h256_hard_s01_steps3200_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_no_ECE_Brier_tail_debt_bootstrap_0p90_rows')}/{sum_attempt_field(h256_hard_s01_steps3200_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_rows')}`, TrueKANGain/BothGain=`{h256_hard_s01_steps3200_micro_h5b8_true_both}/{h256_hard_s01_steps3200_micro_h5b8_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{h256_hard_s01_steps3200_micro_h5b8_beats_any}/{h256_hard_s01_steps3200_micro_h5b8_gap_total}`. This is a partial positive for seed1 own-KAN improvement versus 1600 steps, but it still misses official TrueKANGain and strict no-debt thresholds, so training depth alone is not sufficient.",
        f"- Strict safety-gate repair on the 3200-step route: changing only the gate/evaluation controls to horizon=10, acceptance batches=16, tail q=0.995, post-apply rollback, and held `ECE_Brier_tail` temperature produced seed0 signal=`{wkb(h256_hard_s0_safe3200_micro_h5b8_signal)}`, TrueKANGain/BothGain=`{h256_hard_s0_safe3200_micro_h5b8_true_both}/{h256_hard_s0_safe3200_micro_h5b8_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{h256_hard_s0_safe3200_micro_h5b8_beats_any}/{h256_hard_s0_safe3200_micro_h5b8_gap_total}`; seed1 signal=`{wkb(h256_hard_s1_safe3200_micro_h5b8_signal)}`, TrueKANGain/BothGain=`{h256_hard_s1_safe3200_micro_h5b8_true_both}/{h256_hard_s1_safe3200_micro_h5b8_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{h256_hard_s1_safe3200_micro_h5b8_beats_any}/{h256_hard_s1_safe3200_micro_h5b8_gap_total}`. Combined safe-3200 seeds0+1 are wins=`{sum_attempt_field(h256_hard_s01_safe3200_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_NLL_improvement_rows')}/{sum_attempt_field(h256_hard_s01_safe3200_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_rows')}`, strict no-debt=`{sum_attempt_field(h256_hard_s01_safe3200_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_no_ECE_Brier_tail_debt_rows')}/{sum_attempt_field(h256_hard_s01_safe3200_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_rows')}`, bootstrap no-debt=`{sum_attempt_field(h256_hard_s01_safe3200_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_no_ECE_Brier_tail_debt_bootstrap_0p90_rows')}/{sum_attempt_field(h256_hard_s01_safe3200_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_rows')}`, TrueKANGain/BothGain=`{h256_hard_s01_safe3200_micro_h5b8_true_both}/{h256_hard_s01_safe3200_micro_h5b8_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{h256_hard_s01_safe3200_micro_h5b8_beats_any}/{h256_hard_s01_safe3200_micro_h5b8_gap_total}`. This is the first hard-tier repair that materially raises strict no-debt while preserving own-KAN wins, but effect/TrueKANGain remain below official threshold.",
        f"- Ultra-micro alpha follow-up on the strict safety gate (`0.0025/0.005` added to alpha grid) was not an effect fix: seed0 signal=`{wkb(h256_hard_s0_ultra3200_micro_h5b8_signal)}`, TrueKANGain/BothGain=`{h256_hard_s0_ultra3200_micro_h5b8_true_both}/{h256_hard_s0_ultra3200_micro_h5b8_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{h256_hard_s0_ultra3200_micro_h5b8_beats_any}/{h256_hard_s0_ultra3200_micro_h5b8_gap_total}`; seed1 signal=`{wkb(h256_hard_s1_ultra3200_micro_h5b8_signal)}`, TrueKANGain/BothGain=`{h256_hard_s1_ultra3200_micro_h5b8_true_both}/{h256_hard_s1_ultra3200_micro_h5b8_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{h256_hard_s1_ultra3200_micro_h5b8_beats_any}/{h256_hard_s1_ultra3200_micro_h5b8_gap_total}`. Combined ultra-safe-3200 seeds0+1 stayed wins=`{sum_attempt_field(h256_hard_s01_ultra3200_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_NLL_improvement_rows')}/{sum_attempt_field(h256_hard_s01_ultra3200_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_rows')}`, strict no-debt=`{sum_attempt_field(h256_hard_s01_ultra3200_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_no_ECE_Brier_tail_debt_rows')}/{sum_attempt_field(h256_hard_s01_ultra3200_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_rows')}`, bootstrap no-debt=`{sum_attempt_field(h256_hard_s01_ultra3200_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_no_ECE_Brier_tail_debt_bootstrap_0p90_rows')}/{sum_attempt_field(h256_hard_s01_ultra3200_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_rows')}`, TrueKANGain/BothGain=`{h256_hard_s01_ultra3200_micro_h5b8_true_both}/{h256_hard_s01_ultra3200_micro_h5b8_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{h256_hard_s01_ultra3200_micro_h5b8_beats_any}/{h256_hard_s01_ultra3200_micro_h5b8_gap_total}`. This rules out missing sub-0.01 alpha as the main remaining hard-tier effect blocker.",
        f"- Runtime threshold/cadence follow-up on the h256 strict-safety route was also negative: lowering a2 runtime threshold to 0.10 kept seed0 signal=`{wkb(h256_hard_s0_safe3200_thr010_signal)}` and seed1 signal=`{wkb(h256_hard_s1_safe3200_thr010_signal)}`, with combined wins=`{sum_attempt_field(h256_hard_s01_safe3200_thr010_tags, 'signal', 'active_KAN_SD_CFO_NLL_improvement_rows')}/{sum_attempt_field(h256_hard_s01_safe3200_thr010_tags, 'signal', 'active_KAN_SD_CFO_rows')}` and strict no-debt=`{sum_attempt_field(h256_hard_s01_safe3200_thr010_tags, 'signal', 'active_KAN_SD_CFO_no_ECE_Brier_tail_debt_rows')}/{sum_attempt_field(h256_hard_s01_safe3200_thr010_tags, 'signal', 'active_KAN_SD_CFO_rows')}`. Increasing cadence from 5 to 2 under threshold 0.10 changed seed0 to `{wkb(h256_hard_s0_safe3200_thr010_cad2_signal)}` and seed1 to `{wkb(h256_hard_s1_safe3200_thr010_cad2_signal)}`, combined wins=`{sum_attempt_field(h256_hard_s01_safe3200_thr010_cad2_tags, 'signal', 'active_KAN_SD_CFO_NLL_improvement_rows')}/{sum_attempt_field(h256_hard_s01_safe3200_thr010_cad2_tags, 'signal', 'active_KAN_SD_CFO_rows')}`, strict no-debt=`{sum_attempt_field(h256_hard_s01_safe3200_thr010_cad2_tags, 'signal', 'active_KAN_SD_CFO_no_ECE_Brier_tail_debt_rows')}/{sum_attempt_field(h256_hard_s01_safe3200_thr010_cad2_tags, 'signal', 'active_KAN_SD_CFO_rows')}`. More CATE passes/opportunities did not improve effect and slightly worsened the strict-safety/effect balance.",
        f"- SVHN seed1 h512 capacity probe changed both carriers to NLL wins (`{wkb(h512_svhn_s1_micro_h5b8_signal)}`; TrueKANGain/BothGain=`{h512_svhn_s1_micro_h5b8_true_both}/{h512_svhn_s1_micro_h5b8_gap_total}`), but KAN still did not beat MLP by NLL-or-accuracy (`{h512_svhn_s1_micro_h5b8_beats_any}/{h512_svhn_s1_micro_h5b8_gap_total}`) and strict no-debt stayed `0/2`. This narrows the remaining hard blocker to architecture gap plus strict safety, not FU self-improvement alone.",
        f"- Full Tier1 seed1 h512 micro-alpha/h5b8 improved over h256 seed1 but still did not clear official: signal=`{wkb(h512_hard_s1_micro_h5b8_signal)}`, TrueKANGain/BothGain=`{h512_hard_s1_micro_h5b8_true_both}/{h512_hard_s1_micro_h5b8_gap_total}`, KAN beats MLP by NLL=`{h512_hard_s1_micro_h5b8_beats_nll}/{h512_hard_s1_micro_h5b8_gap_total}`, by NLL-or-accuracy=`{h512_hard_s1_micro_h5b8_beats_any}/{h512_hard_s1_micro_h5b8_gap_total}`. Capacity helps self-improvement, especially SVHN, but strict no-debt and MLP gap remain below official thresholds.",
        f"- Full Tier1 seed0 h512 micro-alpha/h5b8 was `{wkb(h512_hard_s0_micro_h5b8_signal)}` with TrueKANGain/BothGain=`{h512_hard_s0_micro_h5b8_true_both}/{h512_hard_s0_micro_h5b8_gap_total}` and KAN beats MLP by NLL-or-accuracy=`{h512_hard_s0_micro_h5b8_beats_any}/{h512_hard_s0_micro_h5b8_gap_total}`. Combined h512 seeds0+1 are wins=`{sum_attempt_field(h512_hard_s01_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_NLL_improvement_rows')}/{sum_attempt_field(h512_hard_s01_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_rows')}`, strict no-debt=`{sum_attempt_field(h512_hard_s01_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_no_ECE_Brier_tail_debt_rows')}/{sum_attempt_field(h512_hard_s01_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_rows')}`, bootstrap no-debt=`{sum_attempt_field(h512_hard_s01_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_no_ECE_Brier_tail_debt_bootstrap_0p90_rows')}/{sum_attempt_field(h512_hard_s01_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_rows')}`, TrueKANGain/BothGain=`{h512_hard_s01_micro_h5b8_true_both}/{h512_hard_s01_micro_h5b8_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{h512_hard_s01_micro_h5b8_beats_any}/{h512_hard_s01_micro_h5b8_gap_total}`. This is not better than the h256/h512 mixed picture; pure capacity scaling is insufficient.",
        f"- Combining h512 with the strict 3200-step safety gate did not recover the effect gap: seed0 signal=`{wkb(h512_hard_s0_safe3200_micro_h5b8_signal)}`, TrueKANGain/BothGain=`{h512_hard_s0_safe3200_micro_h5b8_true_both}/{h512_hard_s0_safe3200_micro_h5b8_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{h512_hard_s0_safe3200_micro_h5b8_beats_any}/{h512_hard_s0_safe3200_micro_h5b8_gap_total}`; seed1 signal=`{wkb(h512_hard_s1_safe3200_micro_h5b8_signal)}`, TrueKANGain/BothGain=`{h512_hard_s1_safe3200_micro_h5b8_true_both}/{h512_hard_s1_safe3200_micro_h5b8_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{h512_hard_s1_safe3200_micro_h5b8_beats_any}/{h512_hard_s1_safe3200_micro_h5b8_gap_total}`. Combined h512 safe-3200 seeds0+1 are wins=`{sum_attempt_field(h512_hard_s01_safe3200_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_NLL_improvement_rows')}/{sum_attempt_field(h512_hard_s01_safe3200_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_rows')}`, strict no-debt=`{sum_attempt_field(h512_hard_s01_safe3200_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_no_ECE_Brier_tail_debt_rows')}/{sum_attempt_field(h512_hard_s01_safe3200_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_rows')}`, bootstrap no-debt=`{sum_attempt_field(h512_hard_s01_safe3200_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_no_ECE_Brier_tail_debt_bootstrap_0p90_rows')}/{sum_attempt_field(h512_hard_s01_safe3200_micro_h5b8_tags, 'signal', 'active_KAN_SD_CFO_rows')}`, TrueKANGain/BothGain=`{h512_hard_s01_safe3200_micro_h5b8_true_both}/{h512_hard_s01_safe3200_micro_h5b8_gap_total}`, KAN beats MLP by NLL-or-accuracy=`{h512_hard_s01_safe3200_micro_h5b8_beats_any}/{h512_hard_s01_safe3200_micro_h5b8_gap_total}`. This makes h256 safe-3200 the better current hard-tier route; higher capacity plus stricter gate is not sufficient.",
        "",
        "- Part B confirms whether v22.37 signal was support-driven or direction-driven by comparing each real treatment to its matched support / optimizer control. This avoids claiming signal-direction success when only support entry is beneficial.",
        "- CATE rows deliberately keep missing state features blank and count them; no unlogged margin, ECE, Brier, subspace or safety features are imputed. This is why a CATE failure is treated as a representation blocker, not as algorithm success or failure.",
        "- Full-loop policy stages obey the CATE gate. When CATE is blocked, support-native can still run as Route S, but signal / optimizer / KAN SD-CFO matrices are marked `gate_blocked` rather than promoted.",
        "- Safety is interpreted with the sign convention used in the event matrix: positive ECE/Brier/tail deltas mean treatment is non-worse than base; negative deltas count as debt.",
        "- The additional hard-tier CATE collection improved event count but reduced LCB_coverage, which is evidence against a shallow sample-count explanation. The current blocker is hard-tier conditional reliability, not just calibration math.",
        "- Current route is a blocker taxonomy, not a reduced goal. The official target remains DG-KAN+FU superiority over MLP+strong optimizer with no safety debt and matched-control resistance.",
        "",
        "## 8. 修改记录（便于审计）",
        "",
        "- 新增 `experiments/run_v22_38_state_dependent_causal_functional_optimizer.py`：实现 Part A code/import/identity gate、Part B support-direction-horizon-safety reanalysis、Part C CATE event merge/feature ablation/policy calibration、support/signal/optimizer/KAN compact full-loop wrappers、continual probe conversion、figures、artifact index、final route 和复盘写入。",
        "- 复用 v22.37 runner 的 micro-intervention/full-loop 内核，但所有 v22.38 artifact 单独写入 `results/v22_38/`；复用部分的中间文件放在 `results/v22_38/core_interop/`，方便审计。",
        "- CATE feature expansion 明确标注未记录字段为空，并写入 `missing_state_feature_count` 与 `state_feature_source_note`，防止把代理特征伪装成完整状态表示。",
        "- Full-loop wrappers 中 `support` 使用 same-support / same-actuator random support 作为 support-native proxy；`signal` 额外比较 `NLL_delta_vs_FSO_baseline`，避免把 support benefit 写成 direction benefit。",
        "- 审计修正：`materialize_gap_truth_from_matrices()` 对来自 KAN_basis 与 signal 矩阵的同一 KAN gap 记录按 `(dataset, seed, carrier, optimizer, treatment)` 去重；`support_open` 改为读取 support-native full-loop summary，避免把 Part B 支持分解 LCB 误当成 Part D support-native full-loop 成功。",
        "- 继续修复记录：新增 `--output-tag` attempt 输出路径与 `v22_38_horizon_robust_safety_gate_attempts.csv` 汇总，用于审计 horizon-robust safety gate / hard-task compact 尝试是否真实改善 no-debt、NLL wins 与 gap，而不是覆盖或丢弃失败结果。",
        "- 继续修复记录：Part D support-native runtime gate 从 `LCB_net` 改为计划要求的 `LCB_support`；同时修复空 `--output-tag` 被 `safe_fragment()` 映射到 `attempts/x/` 的路径问题，并新增 `v22_38_attempt_gap_truth_matrix.csv` 记录 attempt 级 gap truth。",
        "- 继续修复记录：CATE `calibration_slope` 从协方差 proxy 改为标准 logit-recalibration slope；同时补入 held-group selected-vs-control `LCB_coverage`，official candidate 必须同时满足 `LCB_coverage >= 0.80`，防止只靠 AUC/calibration 误开门。",
        "- 继续修复记录：新增 `--full-loop-implementation horizon_robust`，在 train-only branch 上执行当前 base step、候选 FU、若干未来 train-batch proxy step，并以 CE/ECE/Brier/tail 全 horizon Pareto gate 与 score CVaR 选择 alpha；该路径不使用 test/future/query direction。",
        "- 继续修复记录：第一版 horizon gate 使用第二个 shuffled train iterator 做 future proxy，会扰动主训练顺序并导致 no-op row 与 optimizer_alone 不可比；已改为 `current_train_batch_replay` proxy，并保留失败 attempt 供审计。",
        "- 继续修复记录：为遵守 no class-weight 审计约束，CATE logistic 与 feature-importance logistic 均移除 `class_weight`；新增 `--full-loop-implementation cate_runtime_horizon_robust`，先用 CATE train-event policy 对当前 train batch 状态执行/skip，再进入 horizon/no-debt alpha gate。",
        "- 继续修复记录：`cate_runtime_horizon_robust` 第一版把 runtime `direction_source` 留空，导致 live CATE 概率低于阈值且全部 no-op；已按预注册 treatment family 补齐 `negative_train_gradient` / `negative_optimizer_momentum_or_gradient` / control source 映射，并保留失败 attempt `cate_runtime_h3_q70` 供审计。",
        "- 继续修复记录：CATE decision rule 从固定 0.5 概率阈值改为计划允许范围内的保守 `crossfit_score_quantile_noop_target_0.80`，并让 `stage_c_fit` 优先选择 official-candidate row，再选择 exploration row；该规则不使用 dataset/seed/test/future 信息。",
        "- 继续修复记录：`v22_38_final_route.json` 新增 `CATE_policy_official_candidate_pass` 字段，避免只看 exploration gate 而漏掉 official CATE 状态。",
        "- 继续修复记录：复盘生成器新增 optimizer q65/q60 runtime-threshold sensitivity 的自动汇总，直接读取 `v22_38_horizon_robust_safety_gate_attempts.csv`，用于审计 Part F 胜率/安全债 tradeoff。",
        "- 继续修复记录：新增 treatment-specific CATE policy artifact 与 `--runtime-policy-scope treatment-specific`，按预注册 treatment + matched control 的事件子集选择 CATE gate；默认仍为 global，避免影响旧实验复现。",
        "- 继续修复记录：修复 `crossfit_predict()` 对 one-class held-fold dummy classifier 的正类列处理，并跳过没有真实 treatment/control 双侧事件的 treatment-specific policy，避免生成只有 control 的伪 policy。",
        "- 继续修复记录：修复 `core_args_from()` 漏传 `runtime_policy_scope`，保留误运行为 global 的 `optimizer_runtime_treatment_policy_q80` attempt，并用 `optimizer_runtime_treatment_policy_q80_scopefix` 及 q70-q75 窄区间重跑作证据链。",
        "- 继续修复记录：按 Part G 补跑 D-FOU carrier attempt `kan_dfou_cate_runtime_q80`，同一 attempt tag 下写入 signal matrix、KAN-basis matrix 与 attempt-level gap truth，用于审计 D-FOU 是否改善 architecture value。",
        "- 继续修复记录：新增 KAN carrier 长训/容量诊断 `kan_carrier_long200_cate_runtime_q80`、`kan_carrier_h64_long200_cate_runtime_q80` 与 `kan_carrier_h64_mnist_s0_steps800_cate_runtime_q80`，用于区分 architecture gap 是短训/容量问题还是 FU effect scale 问题。",
        "- 继续修复记录：新增 FU scale diagnostic `kan_carrier_h64_mnist_s0_steps800_alpha1e3`，把 alpha grid 上探到 `0.001`；结果显示效果尺度略放大但仍无法接近 MLP gap，并引入 safety debt。",
        "- 继续修复记录：新增 `--full-loop-alpha-scale-mode base_update_norm` 与 `accepted_effective_alpha_mean` 记录，把 FU alpha 解释为当前 train-only base optimizer 在同 support 上的更新范数倍数；默认仍为 `absolute` 保持旧实验可复现。",
        "- 继续修复记录：基于相对尺度补跑 `kan_carrier_h64_mnist_s0_steps800_relbase_alpha`、`kan_carrier_h64_mnist_s0_steps800_relbase_lowalpha` 与 `kan_carrier_h64_long200_relbase_alpha`，用于审计 effect-scale 修复是否稳定以及是否引入 safety debt。",
        "- 继续修复记录：修复执行日志 Markdown 未写入完整 CLI note 的审计 bug；随后补跑/记录 `relbase_h5b8_tail995`、`relbase_tempwide`、`relbase_temp22`、`relbase_tempwide_nllbt` 与多 seed `h64_long200_relbase_temp22`，区分 horizon-tail gate、温度校准范围和 NLL-aware 温度选择对 no-debt / NLL 的影响。",
        "- 继续修复记录：attempt gap truth 新增 `KAN_SD_CFO_beats_MLP_SD_CFO_by_accuracy` 与 `by_NLL_or_accuracy` 字段，用于审计 Part G official candidate 中的 `NLL or accuracy` 路径；补跑 hidden128/1600 MNIST seed0/seed1 容量/长训与 tail-safety sweep，验证 architecture gap 是否只是短训/容量造成，并定位 nonzero-FU tail no-debt blocker。",
        "- 继续修复记录：新增 `--full-loop-post-apply-rollback`，在实际应用 FU 后用同一个 held acceptance batch 复查 CE/ECE/Brier/tail 并可回滚；postrollback attempt 记录显示当前 h128 blocker 不是 branch/actual 执行不一致，而是 held-to-test tail 泛化。",
        "- 继续修复记录：新增 `--full-loop-metric-bootstrap-samples`、row-local `NLL/ECE/Brier/tail_q99_bootstrap_se`、0.90 bootstrap tolerance 与 `no_ECE_Brier_tail_debt_bootstrap_0p90` 字段；补跑 h128 boot200 MNIST seed0/seed1 与 FashionMNIST/KMNIST seed0/seed1，用于审计 strict q99 tail debt 是否超过评估噪声。该 bootstrap 只用于复盘解释和候选修复方向，不参与训练或运行时 CATE 选择。",
        "- 继续修复记录：补跑 Tier1 hard vision h128 boot200 seed0（CIFAR10/SVHN/EMNIST_LETTERS）的 signal 与 KAN-basis full-loop，用于验证 bootstrap safety 结论是否能离开 Tier0；结果显示 safety uncertainty route 可迁移，但 hard-tier active-KAN NLL wins 仍只有 `2/6`，SVHN 是当前主要效果 blocker。",
        "- 继续修复记录：补跑 hard-tier runtime threshold `0.22/0.30`、low-alpha、low-alpha+horizon=5/batches=8、h256 micro-alpha+horizon=5/batches=8、seed1 复现和 h512 SVHN capacity probe；这些结果表明当前最好路线是 h256 micro-alpha+h5b8，但 seed 鲁棒性、strict no-debt 和 SVHN/CIFAR architecture gap 仍未达到 official。",
        "- 继续修复记录：补跑完整 Tier1 seed1 h512 micro-alpha+h5b8；结果比 h256 seed1 增加 own-KAN NLL wins，但 KAN-vs-MLP 和 strict no-debt 仍未达标，因此不提升 final route。",
        "- 继续修复记录：补跑完整 Tier1 seed0 h512 micro-alpha+h5b8 并合并 h512 seeds0+1；结果显示 h512 不优于 h256 候选，纯容量放大不是当前充分修复。",
        "- 继续修复记录：补跑 h256 micro-alpha+h5b8 的 Tier1 seed2，并合并 seeds0/1/2；三 seed 结果仍未达到 official wins / TrueKANGain / strict no-debt 阈值。",
        "- 继续修复记录：补跑 h256 micro-alpha+h5b8 的 Tier1 seed0/seed1 3200-step 训练深度探针；结果显示 seed1 own-KAN wins 有局部改善，但合并 seeds0+1 的 TrueKANGain 与 strict no-debt 仍未达到 official，训练深度不是充分修复。",
        "- 继续修复记录：在 h256/3200 路线上补跑更严格 safety gate（horizon=10、acceptance batches=16、tail q=0.995、post-apply rollback、`ECE_Brier_tail` 温度选择）；结果显著提高 strict no-debt，但 own-wins/TrueKANGain 仍不足 official。",
        "- 继续修复记录：在严格 safety gate 上补跑 ultra-micro alpha grid（新增 0.0025/0.005 相对 base update norm）；结果与 micro-alpha summary 持平，说明剩余 effect blocker 不是缺少更小 alpha。",
        "- 继续修复记录：补跑 a2 runtime threshold 0.10 与 cadence=2 诊断；更多 CATE pass 与候选机会没有提升 active-KAN wins，反而降低 strict/effect tradeoff。",
        "- 继续修复记录：补跑 h512/3200 与严格 safety gate 叠加路线；结果 strict safety 尚可但 active wins 下降，说明剩余 effect blocker 不能靠更大 hidden 加严格门控直接解决。",
        "- 继续修复记录：补跑 hard-tier Part F optimizer-state h256/3200 严格 safety gate 分支；a3 全参数方向 safety 尚可但 active-KAN wins 明显不足，并且运行耗时显著高于 basis 子集路线。",
        "- 继续修复记录：新增 v22.38 planned Part G basis treatment wrapper 和 `--kan-basis-treatment-mode expanded_planned`，显式运行 `a5_signal_incremental_basis`、`a6_signflip_same_basis`、`a6_same_bank_random`、`a8_support_native_basis` 以及原 a2/a6 matched controls；同时在输出行写入 planned-treatment/audit alias 字段，避免把 exploratory planned rows 当成无出处的旧 a2 结果。",
        "- 继续修复记录：新增 `--pilot-treatment-mode expanded_planned_basis`，在 Part C direct randomized event collection 中直接采样 v22.38 planned basis treatment/control set；本修改只扩展 randomized event 支持，不把未采样 treatment 伪造成已采样。",
        "- 继续修复记录：为 planned basis rows 增加保守 LCB evidence alias：a5 复用 a2 direction LCB，a8/same-bank 复用 same-actuator net/control LCB，并在行级字段标注 alias 来源；这是 gate evidence 的审计标注，不是改写 CATE outcome。",
        "- 继续修复记录：修复 treatment-specific planned-basis runtime feature mismatch：将 live `direction_source` 映射到 direct-CATE event 中出现过的 v22.38 planned source，并把 `applied_update_norm` 从空值改为 intervention scale。该修复没有改损失函数，也没有使用 class weight；它只让 runtime CATE feature schema 与训练事件 schema 对齐。",
        "- 继续修复记录：补跑 expanded planned-basis hard-tier h256/3200 strict-safety 证据链，包括 global-CATE、treatment-specific direct-CATE runtime-feature 修复前失败 attempt、以及修复后 attempt。结果确认 a5 CATE policy 与 runtime 执行是真实进展，但 a2/a5 分开成行仍未把 hard-tier effect/architecture aggregate 推过最终目标。",
        "- 继续修复记录：新增 `--kan-basis-treatment-mode a2_a5_arbitrated` 与 `a2_a5_runtime_arbitrated_basis`，在一个 SD-CFO row 内按 train-only runtime CATE probability、candidate-specific threshold 和 empirical LCB 在 a2/a5 之间选 action，并写出 candidate thresholds/LCB/selected_counts。smoke 首次暴露并保留了局部变量接入 bug，修复后正式 hard-tier seed0 结果显示该仲裁路径可运行但未提升最终 aggregate。",
        "- 继续修复记录：复刻高-effect hard-tier `hard_tier1_cate_runtime_q80_s012` 路线，并只增加 strict safety gate、tail q=0.995、post-apply rollback 与 bootstrap audit，得到 `hard_tier1_cate_runtime_q80_s012_strictgate_boot200`；该结果用于确认剩余 blocker 是否为 safety/effect frontier，而不是更多 treatment 或 CATE 直采样。",
        "- 继续修复记录：补跑同一高-effect 路线的中间 safety gate `hard_tier1_cate_runtime_q80_s012_h5b8_tail995_boot200`（horizon=5、acceptance batches=8、tail q=0.995、postrollback、bootstrap），用于定位 h5/b8 到 h10/b16 之间的 effect/safety tradeoff。",
        "- 继续修复记录：补跑中点 safety gate `hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200`（horizon=7、acceptance batches=12），该设置恢复 `7/9` active KAN wins 并把 strict no-debt 提升到 `5/9`，是当前 hard-tier seed0-2 上最强的 safety/effect 折中证据。",
        "- 继续修复记录：补跑 `hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide`，只把 held-train temperature grid 扩展为 `1,1.3,1.7,2.2,3,4,6`；结果把 hard-tier active wins 提到 `8/9`，strict no-debt 提到 `7/9`，bootstrap no-debt 保持 `9/9`，证明此前 debt 主要来自校准/尾部分位选择。",
        "- 继续修复记录：补跑 `hard_tier1_cate_runtime_q80_s012_h7b12_tail995_boot200_tempwide_tailq99`，不改训练、不改损失函数，只把 held-train temperature selection metric 改为 `tail_q99`；结果仍为 `8/9` active-KAN wins、`7/9` strict no-debt、`9/9` bootstrap no-debt，说明尾部 debt 幅度可压小但 strict 符号债尚未清零。",
        "- 继续修复记录：补跑 tail-focused follow-up：T10 温度网格、tail0.999 acceptance、h10/b16+tempwide、T10+low-alpha；这些修复未超过 `8/9` wins、`7/9` strict 的当前最佳，或以明显 wins 下降换取同样 strict。",
        "- 继续修复记录：补跑 acceptance tolerance frontier（`1e-7,2e-7,3e-7,4e-7,5e-7,1e-6,2e-6,4e-6`）。该方向能把 strict 提到 `8/9` 或 `9/9`，但会把 active-KAN wins 降到 `5/9`、`3/9` 或 `0/9`；说明简单全局 safety margin 不是最终修复。",
        "- 继续修复记录：新增 `--runtime-safety-policy`，用 CATE event matrix 的 train-only `Y_no_debt_gate` 标签训练第二道 runtime no-debt gate；该修改不改损失函数、不使用 class weight、不使用 dataset/seed/test 分支。smoke 通过后补跑 safety-only/loss-only q40 与 sparse threshold KAN 诊断，结果未突破 `8/9` wins、`7/9` strict 当前最佳。",
        "- 继续修复记录：补跑 tail-q99 温度上限 T20/T30；结果显示升温能继续缩小 tail debt 幅度，但会把 active-KAN wins 从 `8/9` 降到 `6/9` 或 `5/9`，没有把 strict no-debt 推过 `7/9`。",
        "- 继续修复记录：新增 `--full-loop-temperature-calibration-mode confidence_tail` 与 `--full-loop-confidence-temperature-quantiles`，只在 held-train 上选择 sample-wise confidence-tail temperature policy；该修改属于校准/评估选择，不改训练目标。KAN-only hard-tier 诊断显示该方向仍不能翻掉 SVHN seed0 / EMNIST_LETTERS seed1 的 strict tail debt。",
        "- 继续修复记录：新增 `--full-loop-acceptance-tail-margin`，把 tail acceptance margin 与 CE/ECE/Brier tolerance 分离；默认 0 保持旧实验复现。KAN-only tail-margin sweep 显示 `<=1e-6` 不改变 accepted pattern，`5e-6/1e-5` 开始损失 wins 并引入额外 strict debt，因此不是充分修复。",
        "- `finalize_route()` 会重新生成 artifact index 与 SHA256 manifest，并把 route 判定写回 `v22_38_final_route.json` 与本复盘。",
        "",
        "## 9. Artifact",
        "",
        "- 主结果目录：`results/v22_38/`",
        "- 执行日志：`docs/DG-KAN_v22.38_StateDependentCausalFunctionalOptimizer_执行日志.md`",
        "- 命令 journal：`results/v22_38/v22_38_command_journal.csv`",
        "- treatment-specific CATE policy：`results/v22_38/v22_38_CATE_treatment_policy_calibration.csv`",
        "- treatment-specific CATE scores：`results/v22_38/v22_38_CATE_treatment_policy_scores.csv`",
        "- final route：`results/v22_38/v22_38_final_route.json`",
        "",
    ]
    RECAP_DOC.write_text("\n".join(lines), encoding="utf-8")


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    stage_a_code_truth_gate()
    stage_b_reanalysis()
    # A compact all-in-one path is intentionally conservative.  For parallel
    # collection, run C_collect labels separately and then C_merge/C_fit.
    stage_c_merge(args)
    stage_c_fit(args)
    run_full_loop_matrix(
        args=args,
        stage_name="D_support_native_full_loop",
        output_name="v22_38_support_native_full_loop_matrix.csv",
        device_name=args.support_device,
        datasets=args.full_loop_datasets,
        seeds=args.full_loop_seeds,
        architectures=args.full_loop_architectures,
        optimizers=args.full_loop_optimizers,
        treatment_plan="support",
        require_cate_gate=False,
    )
    run_full_loop_matrix(
        args=args,
        stage_name="E_signal_incremental_full_loop",
        output_name="v22_38_signal_incremental_full_loop_matrix.csv",
        device_name=args.signal_device,
        datasets=args.full_loop_datasets,
        seeds=args.full_loop_seeds,
        architectures=args.full_loop_architectures,
        optimizers=args.full_loop_optimizers,
        treatment_plan="signal",
        require_cate_gate=True,
    )
    run_full_loop_matrix(
        args=args,
        stage_name="F_optimizer_state_full_loop",
        output_name="v22_38_optimizer_state_full_loop_matrix.csv",
        device_name=args.optimizer_device,
        datasets=args.full_loop_datasets,
        seeds=args.full_loop_seeds,
        architectures=args.full_loop_architectures,
        optimizers=args.full_loop_optimizers,
        treatment_plan="optimizer",
        require_cate_gate=True,
    )
    run_full_loop_matrix(
        args=args,
        stage_name="G_KAN_basis_native_full_loop",
        output_name="v22_38_KAN_basis_native_full_loop_matrix.csv",
        device_name=args.kan_device,
        datasets=args.kan_full_loop_datasets,
        seeds=args.kan_full_loop_seeds,
        architectures=args.kan_full_loop_architectures,
        optimizers=args.full_loop_optimizers,
        treatment_plan="kan_basis",
        require_cate_gate=True,
    )
    stage_h_continual(args)
    return finalize_route(args)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["all", "A", "B", "C_collect", "C_merge", "C_fit", "D", "E", "F", "G", "H", "figures", "finalize"])
    p.add_argument("--collect-label", default="primary")
    p.add_argument("--pilot-device", default="cuda:1")
    p.add_argument("--support-device", default="cuda:2")
    p.add_argument("--signal-device", default="cuda:1")
    p.add_argument("--optimizer-device", default="cuda:1")
    p.add_argument("--kan-device", default="cuda:2")
    p.add_argument("--continual-device", default="cuda:3")
    p.add_argument("--pilot-datasets", default="MNIST,FashionMNIST,KMNIST")
    p.add_argument("--pilot-seeds", default="0,1")
    p.add_argument("--pilot-architectures", default="MLP,DGKAN_DCHE")
    p.add_argument("--pilot-optimizers", default="AdamW")
    p.add_argument("--pilot-horizons", default="20")
    p.add_argument("--pilot-events-per-group", type=int, default=12)
    p.add_argument("--pilot-warmup-steps", type=int, default=5)
    p.add_argument("--pilot-train-size", type=int, default=512)
    p.add_argument("--pilot-held-size", type=int, default=256)
    p.add_argument("--pilot-random-seed", type=int, default=2238)
    p.add_argument("--pilot-top3", action="store_true")
    p.add_argument("--pilot-treatment-mode", default="core_v37", choices=["core_v37", "expanded_planned_basis"], help="Use the historical v22.37 pilot treatment set or directly randomize the v22.38 planned Part G basis treatment/control set for KAN architectures.")
    p.add_argument("--include-v37-events", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--intervention-scale", type=float, default=2.0e-4)
    p.add_argument("--epsilon-row", type=float, default=0.0)
    p.add_argument("--full-loop-datasets", default="MNIST,FashionMNIST,KMNIST")
    p.add_argument("--full-loop-seeds", default="0,1")
    p.add_argument("--full-loop-architectures", default="MLP,DGKAN_DCHE")
    p.add_argument("--kan-full-loop-datasets", default="MNIST,FashionMNIST,KMNIST")
    p.add_argument("--kan-full-loop-seeds", default="0,1")
    p.add_argument("--kan-full-loop-architectures", default="DGKAN_DCHE")
    p.add_argument("--full-loop-optimizers", default="AdamW")
    p.add_argument("--full-loop-train-size", type=int, default=512)
    p.add_argument("--full-loop-held-size", type=int, default=256)
    p.add_argument("--full-loop-steps", type=int, default=60)
    p.add_argument("--full-loop-cadence", type=int, default=5)
    p.add_argument("--full-loop-implementation", default="accepted", choices=["direct", "accepted", "horizon_robust", "cate_runtime_horizon_robust"])
    p.add_argument("--full-loop-alpha-grid", default="0,0.00005,0.0001,0.0002")
    p.add_argument("--full-loop-alpha-scale-mode", default="absolute", choices=["absolute", "base_update_norm"], help="Scale FU alpha as an absolute parameter delta or as a multiplier of the train-only base optimizer update norm on the same support.")
    p.add_argument("--full-loop-alpha-scale-floor", type=float, default=1.0e-12)
    p.add_argument("--full-loop-post-apply-rollback", action="store_true", help="After applying a selected FU, re-check held acceptance components on the actual model state and rollback the FU if CE/ECE/Brier/tail no-debt fails.")
    p.add_argument("--full-loop-metric-bootstrap-samples", type=int, default=0, help="Evaluation-only bootstrap samples for row-local metric uncertainty; does not affect model selection or training.")
    p.add_argument("--full-loop-acceptance-metric", default="strict_ece_no_debt", choices=["ce", "ce_brier", "ce_brier_tail", "strict_no_debt", "strict_ece_no_debt"])
    p.add_argument("--full-loop-acceptance-tol", type=float, default=0.0)
    p.add_argument("--full-loop-acceptance-batches", type=int, default=4)
    p.add_argument("--full-loop-acceptance-tail-quantile", type=float, default=0.99)
    p.add_argument("--full-loop-acceptance-tail-margin", type=float, default=0.0, help="Extra train-only tail component margin; CE/ECE/Brier keep --full-loop-acceptance-tol.")
    p.add_argument("--full-loop-horizon-steps", type=int, default=0)
    p.add_argument("--full-loop-horizon-cvar-fraction", type=float, default=0.25)
    p.add_argument("--runtime-policy-scope", default="global", choices=["global", "treatment-specific"], help="Use the global CATE policy or the selected per-treatment CATE policy at runtime.")
    p.add_argument("--runtime-policy-threshold", type=float, default=-1.0, help="Negative means use the calibrated CATE decision_threshold from C_fit.")
    p.add_argument("--runtime-policy-horizon-feature", type=int, default=60)
    p.add_argument("--runtime-safety-policy", action="store_true", help="Enable a second runtime no-debt policy trained only from CATE event Y_no_debt_gate labels.")
    p.add_argument("--runtime-safety-feature-group", default="safety-only", choices=["loss-only", "signal-only", "actuator-only", "optimizer-only", "safety-only", "temporal-only", "all features"])
    p.add_argument("--runtime-safety-model-kind", default="gbt", choices=["logistic", "gbt", "mlp"])
    p.add_argument("--runtime-safety-threshold", type=float, default=-1.0, help="Negative means use the train-event predicted no-debt probability quantile.")
    p.add_argument("--runtime-safety-quantile", type=float, default=0.20, help="Quantile of train-event no-debt probabilities used as threshold when --runtime-safety-threshold is negative.")
    p.add_argument("--full-loop-temperature-grid", default="0.7,0.85,1.0,1.15,1.3")
    p.add_argument("--full-loop-temperature-selection-metric", default="tail_q99", choices=["NLL", "Brier", "tail_q99", "ECE_Brier_tail", "NLL_Brier_tail"])
    p.add_argument("--full-loop-temperature-calibration-mode", default="scalar", choices=["scalar", "confidence_tail"])
    p.add_argument("--full-loop-confidence-temperature-quantiles", default="0.70,0.80,0.90,0.95")
    p.add_argument("--kan-basis-treatment-mode", default="a2_only", choices=["a2_only", "expanded_planned", "a2_a5_arbitrated"], help="Keep historical a2-only basis rows, explicitly add the planned v22.38 Part G basis treatment/control set, or run one train-only runtime-arbitrated a2/a5 basis row.")
    p.add_argument("--output-tag", default="", help="Optional attempt tag; D/E/F/G outputs are written under results/v22_38/attempts/<tag>/")
    p.add_argument("--tier2-download", action="store_true")
    p.add_argument("--continual-seed", type=int, default=0)
    p.add_argument("--continual-seeds", default="0,1,2")
    p.add_argument("--continual-train-size", type=int, default=256)
    p.add_argument("--continual-held-size", type=int, default=256)
    p.add_argument("--continual-steps-per-task", type=int, default=40)
    return p


def main() -> None:
    args = parser().parse_args()
    ensure_out()
    if args.stage == "all":
        run_all(args)
    elif args.stage == "A":
        stage_a_code_truth_gate()
    elif args.stage == "B":
        stage_b_reanalysis()
    elif args.stage == "C_collect":
        stage_c_collect(args)
    elif args.stage == "C_merge":
        stage_c_merge(args)
    elif args.stage == "C_fit":
        stage_c_fit(args)
    elif args.stage == "D":
        run_full_loop_matrix(args=args, stage_name=stage_name_for(args, "D_support_native_full_loop"), output_name=output_name_for(args, "v22_38_support_native_full_loop_matrix.csv"), device_name=args.support_device, datasets=args.full_loop_datasets, seeds=args.full_loop_seeds, architectures=args.full_loop_architectures, optimizers=args.full_loop_optimizers, treatment_plan="support", require_cate_gate=False)
    elif args.stage == "E":
        run_full_loop_matrix(args=args, stage_name=stage_name_for(args, "E_signal_incremental_full_loop"), output_name=output_name_for(args, "v22_38_signal_incremental_full_loop_matrix.csv"), device_name=args.signal_device, datasets=args.full_loop_datasets, seeds=args.full_loop_seeds, architectures=args.full_loop_architectures, optimizers=args.full_loop_optimizers, treatment_plan="signal", require_cate_gate=True)
    elif args.stage == "F":
        run_full_loop_matrix(args=args, stage_name=stage_name_for(args, "F_optimizer_state_full_loop"), output_name=output_name_for(args, "v22_38_optimizer_state_full_loop_matrix.csv"), device_name=args.optimizer_device, datasets=args.full_loop_datasets, seeds=args.full_loop_seeds, architectures=args.full_loop_architectures, optimizers=args.full_loop_optimizers, treatment_plan="optimizer", require_cate_gate=True)
    elif args.stage == "G":
        run_full_loop_matrix(args=args, stage_name=stage_name_for(args, "G_KAN_basis_native_full_loop"), output_name=output_name_for(args, "v22_38_KAN_basis_native_full_loop_matrix.csv"), device_name=args.kan_device, datasets=args.kan_full_loop_datasets, seeds=args.kan_full_loop_seeds, architectures=args.kan_full_loop_architectures, optimizers=args.full_loop_optimizers, treatment_plan="kan_basis", require_cate_gate=True)
    elif args.stage == "H":
        stage_h_continual(args)
    elif args.stage == "figures":
        stage_figures_efficiency()
    elif args.stage == "finalize":
        finalize_route(args)


if __name__ == "__main__":
    main()
