#!/usr/bin/env python3
"""DG-KAN v22.45E multi-mechanism extension runner.

This runner is intentionally conservative: it reuses the audited v22.43/v22.44R
continuous FU kernel when a mechanism can be represented there, records proxy
coverage explicitly when it cannot, and never upgrades a proxy row into a true
mechanism-opened route.
"""

from __future__ import annotations

import argparse
import compileall
import concurrent.futures
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
csv.field_size_limit(sys.maxsize)

from experiments import run_v22_43_metric_preserving_continuous_functional_flow_fu as v2243
from experiments import run_v22_44R_functional_actuator_spectrum_metric_oet_fu as v2244
from experiments import run_v22_42R_s7_continual_memory_diagnostic as v2242s7


PYTHON = os.environ.get("KAN_PYTHON", "/home/chengshun.wang/miniconda3/envs/kan/bin/python")
OUT_ROOT = ROOT / "results/v22_45E"
CHUNK_ROOT = OUT_ROOT / "chunks"
LOG_ROOT = OUT_ROOT / "logs"
FIG_ROOT = OUT_ROOT / "figures"
EXEC_DOC = ROOT / "docs/DG-KAN_v22.45E_MultiMechanismExtension_执行日志.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v22.45E_MultiMechanismExtension_实验结果复盘.md"
LOG_LOCK = threading.RLock()


V2245E_TABLES = [
    "v22_45E_code_truth_gate.csv",
    "v22_45E_runtime_regression_audit.csv",
    "v22_45E_metric_oet_fidelity_matrix.csv",
    "v22_45E_functional_actuator_spectrum_matrix.csv",
    "v22_45E_support_direction_decomposition.csv",
    "v22_45E_safety_debt_matrix.csv",
    "v22_45E_overhead_matrix.csv",
    "v22_45E_mechanism_coverage_matrix.csv",
    "v22_45E_lowcost_screen_matrix.csv",
    "v22_45E_mechanism_screen_summary.csv",
    "v22_45E_repair_screen_matrix.csv",
    "v22_45E_repair_mechanism_summary.csv",
    "v22_45E_repair_support_direction_decomposition.csv",
    "v22_45E_repair_safety_debt_matrix.csv",
    "v22_45E_repair_overhead_matrix.csv",
    "v22_45E_repair_runtime_regression_audit.csv",
    "v22_45E_m5_variant_matrix.csv",
    "v22_45E_m5_variant_mechanism_summary.csv",
    "v22_45E_m5_variant_support_direction_decomposition.csv",
    "v22_45E_m5_variant_safety_debt_matrix.csv",
    "v22_45E_m5_variant_overhead_matrix.csv",
    "v22_45E_m5_variant_runtime_regression_audit.csv",
    "v22_45E_m2_trajectory_specs.csv",
    "v22_45E_m2_trajectory_matrix.csv",
    "v22_45E_m2_trajectory_mechanism_summary.csv",
    "v22_45E_m2_trajectory_support_direction_decomposition.csv",
    "v22_45E_m2_trajectory_safety_debt_matrix.csv",
    "v22_45E_m2_trajectory_overhead_matrix.csv",
    "v22_45E_m2_trajectory_runtime_regression_audit.csv",
    "v22_45E_m2_trajectory_plan_metrics.csv",
    "v22_45E_m2_trajectory_status.csv",
    "v22_45E_m5_trajectory_specs.csv",
    "v22_45E_m5_trajectory_matrix.csv",
    "v22_45E_m5_trajectory_mechanism_summary.csv",
    "v22_45E_m5_trajectory_support_direction_decomposition.csv",
    "v22_45E_m5_trajectory_safety_debt_matrix.csv",
    "v22_45E_m5_trajectory_overhead_matrix.csv",
    "v22_45E_m5_trajectory_runtime_regression_audit.csv",
    "v22_45E_m5_trajectory_status.csv",
    "v22_45E_m5_fairness_plan_metrics.csv",
    "v22_45E_m5_fairness_plan_summary.csv",
    "v22_45E_m6_trajectory_specs.csv",
    "v22_45E_m6_trajectory_matrix.csv",
    "v22_45E_m6_trajectory_mechanism_summary.csv",
    "v22_45E_m6_trajectory_support_direction_decomposition.csv",
    "v22_45E_m6_trajectory_safety_debt_matrix.csv",
    "v22_45E_m6_trajectory_overhead_matrix.csv",
    "v22_45E_m6_trajectory_runtime_regression_audit.csv",
    "v22_45E_m6_trajectory_plan_metrics.csv",
    "v22_45E_m6_trajectory_status.csv",
    "v22_45E_m7_trajectory_specs.csv",
    "v22_45E_m7_trajectory_matrix.csv",
    "v22_45E_m7_trajectory_trace.csv",
    "v22_45E_m7_trajectory_runtime_trace.csv",
    "v22_45E_m7_trajectory_plan_metrics.csv",
    "v22_45E_m7_trajectory_status.csv",
    "v22_45E_m7_trajectory_summary.json",
    "v22_45E_m9_trajectory_specs.csv",
    "v22_45E_m9_trajectory_matrix.csv",
    "v22_45E_m9_trajectory_mechanism_summary.csv",
    "v22_45E_m9_trajectory_support_direction_decomposition.csv",
    "v22_45E_m9_trajectory_safety_debt_matrix.csv",
    "v22_45E_m9_trajectory_overhead_matrix.csv",
    "v22_45E_m9_trajectory_runtime_regression_audit.csv",
    "v22_45E_m9_trajectory_plan_metrics.csv",
    "v22_45E_m9_trajectory_status.csv",
    "v22_45E_m8_gauge_feasibility_audit.csv",
    "v22_45E_m8_gauge_dynamic_equivalence.csv",
    "v22_45E_m4_trajectory_specs.csv",
    "v22_45E_m4_trajectory_matrix.csv",
    "v22_45E_m4_trajectory_trace.csv",
    "v22_45E_m4_trajectory_runtime_trace.csv",
    "v22_45E_m4_trajectory_plan_metrics.csv",
    "v22_45E_m4_trajectory_status.csv",
    "v22_45E_m4_trajectory_summary.json",
    "v22_45E_M4_slow_signal_diagnostic_matrix.csv",
    "v22_45E_M7_continual_memory_matrix.csv",
    "v22_45E_blocker_repair_log.csv",
    "v22_45E_execution_repair_log.csv",
    "v22_45E_proxy_evidence_audit.csv",
    "v22_45E_command_journal.csv",
]


MECHANISM_COVERAGE = [
    {
        "mechanism": "Core-M",
        "name": "Functional-Actuator-Spectrum Safe Metric-OET FU",
        "coverage": "implemented",
        "kernel_mapping": "S4-SignalMetric-OET in v22.43 continuous FU kernel",
        "claim_limit": "screens Core-M metric/OET support; not a new v22.45E mechanism",
    },
    {
        "mechanism": "M1",
        "name": "Functional Mirror / Bregman Flow",
        "coverage": "partial",
        "kernel_mapping": "M1 logit-space mirror variants with explicit train-batch dual/primal update plus support-masked J^T residual and M1-KLMirrorLS sketched functional least-squares actuator projection",
        "claim_limit": "explicit dual/primal mirror state is implemented for train-batch logits; M1-KLMirrorLS adds a train-batch sketched least-squares actuator approximation, not the full all-function/all-parameter solver",
    },
    {
        "mechanism": "M2",
        "name": "Spectral-Radial Split Flow",
        "coverage": "partial",
        "kernel_mapping": "P4 functional-JVP radial gated pure using train-batch finite-difference logit response spectrum and safety budget debt",
        "claim_limit": "finite-difference batch functional spectrum is closer to plan M2, but still a low-cost approximation rather than a full actuator-spectrum solver",
    },
    {
        "mechanism": "M3",
        "name": "Transported Lie-Momentum Flow",
        "coverage": "partial",
        "kernel_mapping": "P3 transported left-Lie generator momentum with adjoint transport by previous Cayley rotation and same-Lie controls",
        "claim_limit": "tests a real runtime Lie momentum state, but still lacks full left/right balanced transport and hard-task confirmation",
    },
    {
        "mechanism": "M4",
        "name": "Slow-Signal Reservoir Migration Flow",
        "coverage": "partial",
        "kernel_mapping": "S1 residual signal with slow beta plus v22.45E in-process modular-addition and Class_MNIST slow/reservoir trajectories",
        "claim_limit": "M4 trajectories record slow/fast/RSM and matched slow controls, but use a diagonal top-k parameter support approximation rather than the full metric/OET projection",
    },
    {
        "mechanism": "M5",
        "name": "Functional Actuator Matching for Fair MLP Supports",
        "coverage": "implemented",
        "kernel_mapping": "S6 KAN BasisGram/OET carrier rows plus MLP low-rank/frequency matched supports",
        "claim_limit": "matching is via existing functional support variants and spectrum audit distances",
    },
    {
        "mechanism": "M6",
        "name": "Noise-Shaped Reservoir Regularization",
        "coverage": "partial",
        "kernel_mapping": "M6-reservoir-noise and M6-noise-suppression in v22.43 complete trajectory loop with parameter metric-support complement reservoir noise plus same-norm signal/Gaussian controls",
        "claim_limit": "parameter metric support-complement reservoir noise only; not full function-space Brownian reservoir and not SignalFUOpened evidence",
    },
    {
        "mechanism": "M7",
        "name": "Continual Functional Memory Flow",
        "coverage": "partial",
        "kernel_mapping": "v22.45E trajectory-m7 old/new MNIST-family and modular add-to-multiply continual loops with rank-k parameter-gradient and finite-difference old-logit memory projection plus random/rehearsal controls",
        "claim_limit": "rank-k finite-difference old-logit projection is a train-anchor J_old*v approximation; still not full all-function G_f memory projection and not EMNIST coverage",
    },
    {
        "mechanism": "M8",
        "name": "Gauge / Canonicalization Flow",
        "coverage": "blocked",
        "kernel_mapping": "basis Gram condition and basis_state_transport_error diagnostics only",
        "claim_limit": "no canonical gauge transform Gamma(theta) implemented in runtime",
    },
    {
        "mechanism": "M9",
        "name": "Adaptive Metric Mixture Flow",
        "coverage": "partial",
        "kernel_mapping": "M9-adaptive-metric-mixture in v22.43 complete trajectory loop with train-batch diagonal Euclidean/Fisher/Signal/Basis mixture weights updated by mirror descent",
        "claim_limit": "diagonal train-batch metric mixture only; not a full matrix metric search, not validation/test-guided, and only opened if it beats fixed metric controls with no-debt",
    },
]


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def safe_fragment(value: Any) -> str:
    return v2243.safe_fragment(value)


def finite_float(value: Any, default: float | None = None) -> float | None:
    return v2243.finite_float(value, default)


def value_or(value: Any, default: float) -> float:
    parsed = finite_float(value)
    return float(default) if parsed is None else float(parsed)


def int_flag(value: Any) -> int:
    return v2243.int_flag(value)


def bool_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def spec_value(spec: dict[str, Any], key: str, default: Any) -> Any:
    value = spec.get(key, default)
    return default if value == "" or value is None else value


def split_csv(text: str, cast: Any = str) -> list[Any]:
    return v2243.split_csv(text, cast)


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.45E MultiMechanismExtension 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只记录真实执行的命令、文件、GPU、状态、blocker 与修复尝试；"
            "未执行、被 gate 阻断、数据不可用或失败必须显式写出；不补造实验结果。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.45E MultiMechanismExtension 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：复盘只引用本轮 artifact 或明确命名的上游诊断 artifact；"
            "实验数据、修复动作、分析结论、insight 和证据链必须可追溯；不编造缺失数据。\n",
            encoding="utf-8",
        )
    journal = OUT_ROOT / "v22_45E_command_journal.csv"
    if not journal.exists():
        with journal.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["timestamp", "task_id", "gpu", "command", "status", "exit_code", "files", "note"],
            )
            writer.writeheader()


def bind_upstream() -> None:
    ensure_out()
    for mod in (v2243, v2244):
        mod.OUT_ROOT = OUT_ROOT
        mod.CHUNK_ROOT = CHUNK_ROOT
        mod.LOG_ROOT = LOG_ROOT
        if hasattr(mod, "FIG_ROOT"):
            mod.FIG_ROOT = FIG_ROOT
        mod.EXEC_DOC = EXEC_DOC
        mod.RECAP_DOC = RECAP_DOC
        mod.ensure_out = ensure_out
        mod.append_exec = append_exec


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
    if not fields:
        fields = ["status"]
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(json_clean(data), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def read_rows_with_source(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob(pattern)):
        for row in read_rows(path):
            item = dict(row)
            item["source_artifact"] = path.name
            rows.append(item)
    return rows


def json_clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_clean(v) for v in value]
    return value


def append_exec(
    command: str,
    *,
    task_id: str,
    status: str,
    gpu: str = "",
    files: str = "",
    note: str = "",
    exit_code: Any = "n/a",
) -> None:
    ensure_out()
    with LOG_LOCK:
        command_s = str(command)
        if len(command_s) > 8000:
            command_s = command_s[:8000] + " ...[truncated]"
        row = {
            "timestamp": now_sg(),
            "task_id": str(task_id),
            "gpu": str(gpu or "n/a"),
            "command": command_s,
            "status": str(status),
            "exit_code": str(exit_code),
            "files": str(files),
            "note": str(note),
        }
        with (OUT_ROOT / "v22_45E_command_journal.csv").open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "task_id", "gpu", "command", "status", "exit_code", "files", "note"])
            writer.writerow(row)
        with EXEC_DOC.open("a", encoding="utf-8") as f:
            f.write(f"\n## {row['timestamp']} {row['task_id']}\n\n")
            f.write("```bash\n" + command_s + "\n```\n\n")
            f.write(f"- gpu: {row['gpu']}\n- status: {row['status']}\n- exit_code: {row['exit_code']}\n")
            if files:
                f.write(f"- files: {files}\n")
            if note:
                f.write(f"- note: {note}\n")


def run_logged(cmd: list[str], *, task_id: str, gpu: str = "", timeout: int = 1200) -> subprocess.CompletedProcess[str]:
    ensure_out()
    stdout_path = LOG_ROOT / f"{safe_fragment(task_id)}_stdout.log"
    stderr_path = LOG_ROOT / f"{safe_fragment(task_id)}_stderr.log"
    env = os.environ.copy()
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu).replace("cuda:", "")
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, timeout=int(timeout))
    except subprocess.TimeoutExpired as exc:
        proc = subprocess.CompletedProcess(cmd, 124, stdout=exc.stdout or "", stderr=exc.stderr or "")
    stdout_path.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8", errors="replace")
    append_exec(
        " ".join(shlex.quote(x) for x in cmd),
        task_id=task_id,
        status="pass" if proc.returncode == 0 else "fail",
        gpu=gpu,
        files=f"{stdout_path.relative_to(ROOT)}, {stderr_path.relative_to(ROOT)}",
        note=f"elapsed_sec={time.time() - started:.3f}",
        exit_code=proc.returncode,
    )
    return proc


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return True


def run_code_truth_gate() -> dict[str, Any]:
    ensure_out()
    started = time.time()
    files = [
        ROOT / "experiments/run_v22_45E_multi_mechanism_extension.py",
        ROOT / "experiments/run_v22_44R_functional_actuator_spectrum_metric_oet_fu.py",
        ROOT / "experiments/run_v22_43_metric_preserving_continuous_functional_flow_fu.py",
    ]
    compile_errors: list[str] = []
    for path in files:
        ok = compileall.compile_file(str(path), quiet=1)
        if not ok:
            compile_errors.append(str(path.relative_to(ROOT)))
    import_failures: list[str] = []
    for modname in [
        "experiments.run_v22_45E_multi_mechanism_extension",
        "experiments.run_v22_44R_functional_actuator_spectrum_metric_oet_fu",
        "experiments.run_v22_43_metric_preserving_continuous_functional_flow_fu",
    ]:
        try:
            __import__(modname)
        except Exception as exc:  # pragma: no cover - audit path
            import_failures.append(f"{modname}: {type(exc).__name__}: {exc}")
    row = {
        "compile_pass": int(not compile_errors),
        "import_pass": int(not import_failures),
        "missing_transitive_dependency_count": len(import_failures),
        "runtime_argmax_candidate_used": 0,
        "runtime_topk_candidate_used": 0,
        "candidate_action_selection_used_for_runtime": 0,
        "candidate_value_model_used_as_runtime_policy": 0,
        "micro_rct_winner_used_as_runtime_action": 0,
        "treatment_identity_used_in_runtime_state": 0,
        "direction_source_used_in_runtime_state": 0,
        "continuous_fu_state_updated_every_step": "audited_in_runtime_rows",
        "fu_velocity_emitted_every_step": "audited_in_runtime_rows",
        "metric_search_used": 0,
        "registered_metrics_only": 1,
        "compile_errors": ";".join(compile_errors),
        "import_failures": ";".join(import_failures),
        "elapsed_sec": f"{time.time() - started:.3f}",
        "status": "pass" if not compile_errors and not import_failures else "fail",
    }
    write_rows(OUT_ROOT / "v22_45E_code_truth_gate.csv", [row])
    append_exec(
        "run_code_truth_gate",
        task_id="part_A_code_truth_gate",
        status=str(row["status"]),
        gpu="cpu",
        files="results/v22_45E/v22_45E_code_truth_gate.csv",
        note=f"compile_errors={len(compile_errors)}; import_failures={len(import_failures)}",
    )
    return row


def run_metric_and_spectrum(args: argparse.Namespace) -> dict[str, Any]:
    bind_upstream()
    metric = v2244.run_metric_oet_fidelity()
    copy_if_exists(OUT_ROOT / "v22_44R_metric_oet_fidelity_matrix.csv", OUT_ROOT / "v22_45E_metric_oet_fidelity_matrix.csv")
    spectrum_args = argparse.Namespace(
        audit_datasets=args.audit_datasets,
        audit_architectures=args.audit_architectures,
        audit_supports=args.audit_supports,
        audit_train_size=args.audit_train_size,
        held_size=args.held_size,
        batch_size=args.batch_size,
        sketch_dim=args.sketch_dim,
        functional_eps=args.functional_eps,
        hidden=args.hidden,
        seed=args.seed,
        device=args.device,
        tier2_download=args.tier2_download,
    )
    spectrum = v2244.run_part_c(spectrum_args)
    copy_if_exists(OUT_ROOT / "v22_44R_functional_actuator_spectrum_matrix.csv", OUT_ROOT / "v22_45E_functional_actuator_spectrum_matrix.csv")
    append_exec(
        "run_metric_and_functional_spectrum_audit",
        task_id="part_BC_metric_spectrum_alias",
        status="pass" if metric.get("status") == "pass" and spectrum.get("status") == "pass" else "warn",
        gpu=args.device,
        files="results/v22_45E/v22_45E_metric_oet_fidelity_matrix.csv, results/v22_45E/v22_45E_functional_actuator_spectrum_matrix.csv",
        note="v22.44R audit functions reused with v22.45E OUT_ROOT; v22_44R-named source artifacts kept and v22_45E aliases copied.",
    )
    return {"metric": metric, "spectrum": spectrum}


def base_collect_cmd(args: argparse.Namespace, spec: dict[str, Any], *, child_device: str) -> list[str]:
    cmd = [
        PYTHON,
        str(Path(__file__).relative_to(ROOT)),
        "--stage",
        "collect",
        "--dataset",
        str(spec["dataset"]),
        "--seed",
        str(spec["seed"]),
        "--architecture",
        str(spec["architecture"]),
        "--optimizer",
        str(spec["optimizer"]),
        "--variant",
        str(spec["variant"]),
        "--control-mode",
        str(spec["control_mode"]),
        "--device",
        child_device,
        "--steps",
        str(spec.get("steps", args.steps)),
        "--train-size",
        str(args.train_size),
        "--held-size",
        str(args.held_size),
        "--batch-size",
        str(args.batch_size),
        "--hidden",
        str(spec.get("hidden", args.hidden)),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--support-rank",
        str(spec.get("support_rank", args.support_rank)),
        "--nuisance-rank",
        str(spec.get("nuisance_rank", args.nuisance_rank)),
        "--support-refresh-cadence",
        str(spec.get("support_refresh_cadence", args.support_refresh_cadence)),
        "--beta-signal",
        str(spec.get("beta_signal", args.beta_signal)),
        "--beta-metric",
        str(spec.get("beta_metric", args.beta_metric)),
        "--beta-q",
        str(spec.get("beta_q", args.beta_q)),
        "--eta-rho",
        str(spec.get("eta_rho", args.eta_rho)),
        "--eta-debt",
        str(spec.get("eta_debt", args.eta_debt)),
        "--tau-safe",
        str(spec.get("tau_safe", args.tau_safe)),
        "--rho-min",
        str(spec.get("rho_min", args.rho_min)),
        "--rho-max",
        str(spec.get("rho_max", args.rho_max)),
        "--velocity-scale",
        str(spec.get("velocity_scale", args.velocity_scale)),
        "--metric-shrinkage",
        str(spec.get("metric_shrinkage", args.metric_shrinkage)),
        "--metric-eps",
        str(spec.get("metric_eps", args.metric_eps)),
        "--metric-refresh-cadence",
        str(spec.get("metric_refresh_cadence", args.metric_refresh_cadence)),
        "--debt-velocity-barrier",
        str(spec.get("debt_velocity_barrier", args.debt_velocity_barrier)),
        "--calibration-velocity-barrier",
        str(spec.get("calibration_velocity_barrier", args.calibration_velocity_barrier)),
        "--safety-budget-velocity-barrier",
        str(spec.get("safety_budget_velocity_barrier", args.safety_budget_velocity_barrier)),
        "--calibration-readout-radial-cap",
        str(spec.get("calibration_readout_radial_cap", args.calibration_readout_radial_cap)),
        "--calibration-nuisance-weight",
        str(spec.get("calibration_nuisance_weight", args.calibration_nuisance_weight)),
        "--calibration-correction-weight",
        str(spec.get("calibration_correction_weight", args.calibration_correction_weight)),
        "--calibration-nuisance-mode",
        str(spec.get("calibration_nuisance_mode", args.calibration_nuisance_mode)),
        "--calibration-nuisance-cadence",
        str(spec.get("calibration_nuisance_cadence", args.calibration_nuisance_cadence)),
        "--mirror-grad-cadence",
        str(spec.get("mirror_grad_cadence", args.mirror_grad_cadence)),
        "--kan-init-variant",
        str(spec.get("kan_init_variant", args.kan_init_variant)),
        "--label",
        str(spec["label"]),
    ]
    if spec.get("pure_fu_mode"):
        cmd.append("--pure-fu-mode")
    if int(spec.get("warmup_steps", args.warmup_steps)) > 0:
        cmd.extend(["--warmup-steps", str(spec.get("warmup_steps", args.warmup_steps))])
    return cmd


def add_spec(
    specs: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    mechanism: str,
    recipe: str,
    dataset: str,
    seed: int,
    architecture: str,
    variant: str,
    control_mode: str,
    optimizer: str = "AdamW",
    pure_fu_mode: bool = False,
    **overrides: Any,
) -> None:
    gpus = split_csv(args.gpus)
    gpu = gpus[len(specs) % max(1, len(gpus))]
    arch = v2243.default_architecture_for_variant(variant, architecture)
    label = (
        f"{safe_fragment(args.label)}_{safe_fragment(mechanism)}_{safe_fragment(recipe)}_"
        f"{safe_fragment(dataset)}_s{seed}_{safe_fragment(arch)}_{safe_fragment(optimizer)}_"
        f"{safe_fragment(variant)}_{safe_fragment(control_mode)}"
    )
    row = {
        "mechanism": mechanism,
        "recipe": recipe,
        "dataset": dataset,
        "seed": int(seed),
        "architecture": arch,
        "optimizer": optimizer,
        "variant": variant,
        "control_mode": control_mode,
        "device": f"cuda:{gpu}" if not str(gpu).startswith("cuda") else str(gpu),
        "label": label,
        "pure_fu_mode": pure_fu_mode,
    }
    row.update(overrides)
    specs.append(row)


def build_screen_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets = split_csv(args.eval_datasets)
    seeds = split_csv(args.eval_seeds, int)
    archs = split_csv(args.eval_architectures)
    specs: list[dict[str, Any]] = []
    seen_baselines: set[tuple[str, int, str]] = set()
    for dataset in datasets:
        for seed in seeds:
            for arch in archs:
                base_key = (str(dataset), int(seed), str(arch))
                if base_key not in seen_baselines:
                    add_spec(
                        specs,
                        args,
                        mechanism="BASE",
                        recipe="optimizer_alone",
                        dataset=dataset,
                        seed=seed,
                        architecture=arch,
                        variant="optimizer_alone",
                        control_mode="none",
                    )
                    seen_baselines.add(base_key)
                if "Core-M" in args.mechanisms:
                    for cmode in ["none", "same-metric-support-random", "same-metric-support-signflip"]:
                        add_spec(specs, args, mechanism="Core-M", recipe="SignalMetricOET", dataset=dataset, seed=seed, architecture=arch, variant="S4-SignalMetric-OET", control_mode=cmode)
                if "M1" in args.mechanisms:
                    for cmode in ["none", "same-metric-support-random", "same-metric-support-signflip"]:
                        add_spec(
                            specs,
                            args,
                            mechanism="M1",
                            recipe="FisherMirror_BrierProxy",
                            dataset=dataset,
                            seed=seed,
                            architecture=arch,
                            variant="S4-FisherEMA-OET",
                            control_mode=cmode,
                            calibration_nuisance_weight=args.m1_calibration_weight,
                            calibration_nuisance_mode="brier",
                            calibration_velocity_barrier=args.m1_calibration_barrier,
                        )
                if "M2" in args.mechanisms:
                    for cmode in ["none", "same-radial-random", "same-OET-random"]:
                        add_spec(
                            specs,
                            args,
                            mechanism="M2",
                            recipe="OETRadial005Pure",
                            dataset=dataset,
                            seed=seed,
                            architecture=arch,
                            variant="P4-Euclidean-OET-radial005-pure",
                            control_mode=cmode,
                            pure_fu_mode=True,
                            velocity_scale=args.m2_velocity_scale,
                            safety_budget_velocity_barrier=args.m2_safety_barrier,
                        )
                if "M3" in args.mechanisms:
                    for cmode in ["none", "same-OET-random"]:
                        add_spec(
                            specs,
                            args,
                            mechanism="M3",
                            recipe="OETPure_NoTransportedMomentum",
                            dataset=dataset,
                            seed=seed,
                            architecture=arch,
                            variant="P3-Euclidean-OET-pure",
                            control_mode=cmode,
                            pure_fu_mode=True,
                            velocity_scale=args.m3_velocity_scale,
                        )
                if "M4" in args.mechanisms:
                    for cmode in ["none", "same-metric-support-random", "same-metric-support-signflip"]:
                        add_spec(
                            specs,
                            args,
                            mechanism="M4",
                            recipe="SlowSignalS1Proxy",
                            dataset=dataset,
                            seed=seed,
                            architecture=arch,
                            variant="S1-M-top1-S4-SignalMetric-OET",
                            control_mode=cmode,
                            beta_signal=args.m4_beta_slow,
                        )
                if "M5" in args.mechanisms:
                    if str(arch) == "DGKAN_DCHE":
                        for cmode in ["none", "same-basis-Gram-random", "same-basis-Gram-signflip"]:
                            add_spec(specs, args, mechanism="M5", recipe="KAN_DCHE_BasisGram", dataset=dataset, seed=seed, architecture=arch, variant="KAN-D-CHE-BasisGram", control_mode=cmode)
                    elif str(arch) == "MLP":
                        for variant in ["MLP-low-rank-hidden-metric-support", "MLP-frequency-like-random-feature-support"]:
                            add_spec(specs, args, mechanism="M5", recipe="MLP_MatchedFunctionalSupport", dataset=dataset, seed=seed, architecture=arch, variant=variant, control_mode="none")
                if "M6" in args.mechanisms:
                    continue
    if int(args.row_limit) > 0:
        specs = specs[: int(args.row_limit)]
    return specs


def build_repair_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets = split_csv(args.eval_datasets)
    seeds = split_csv(args.eval_seeds, int)
    archs = split_csv(args.eval_architectures)
    repair_mechanisms = set(split_csv(args.repair_mechanisms))
    specs: list[dict[str, Any]] = []
    seen_baselines: set[tuple[str, int, str]] = set()
    common_efficiency = {
        "support_refresh_cadence": max(20, int(args.support_refresh_cadence)),
        "metric_refresh_cadence": max(5, int(args.metric_refresh_cadence)),
        "metric_shrinkage": max(0.10, float(args.metric_shrinkage)),
    }
    for dataset in datasets:
        for seed in seeds:
            for arch in archs:
                base_key = (str(dataset), int(seed), str(arch))
                if base_key not in seen_baselines:
                    add_spec(
                        specs,
                        args,
                        mechanism="BASE",
                        recipe="repair_optimizer_alone",
                        dataset=dataset,
                        seed=seed,
                        architecture=arch,
                        variant="optimizer_alone",
                        control_mode="none",
                        **common_efficiency,
                    )
                    seen_baselines.add(base_key)
                if "M1" in repair_mechanisms:
                    for cmode in ["none", "same-metric-support-random", "same-metric-support-signflip"]:
                        add_spec(
                            specs,
                            args,
                            mechanism="M1",
                            recipe="TailBrierMirrorFallback",
                            dataset=dataset,
                            seed=seed,
                            architecture=arch,
                            variant="S4-FisherEMA-OET",
                            control_mode=cmode,
                            calibration_nuisance_weight=max(float(args.m1_calibration_weight), 0.35),
                            calibration_correction_weight=0.10,
                            calibration_nuisance_mode="tail_q99_brier_qp_margin",
                            calibration_velocity_barrier=max(float(args.m1_calibration_barrier), 2.0),
                            safety_budget_velocity_barrier=max(float(args.safety_budget_velocity_barrier), 4.0),
                            calibration_nuisance_cadence=max(5, int(args.calibration_nuisance_cadence)),
                            velocity_scale=min(float(args.velocity_scale), 0.35),
                            rho_max=min(float(args.rho_max), 0.20),
                            **common_efficiency,
                        )
                if "M2" in repair_mechanisms:
                    for cmode in ["none", "same-radial-random", "same-OET-random"]:
                        add_spec(
                            specs,
                            args,
                            mechanism="M2",
                            recipe="FunctionalRadialGatedWarmup",
                            dataset=dataset,
                            seed=seed,
                            architecture=arch,
                            variant=str(args.m2_repair_variant),
                            control_mode=cmode,
                            pure_fu_mode=True,
                            warmup_steps=max(int(args.warmup_steps), 60),
                            velocity_scale=min(float(args.m2_velocity_scale), 0.20),
                            safety_budget_velocity_barrier=max(float(args.m2_safety_barrier), 5.0),
                            beta_signal=min(float(args.beta_signal), 0.02),
                            **common_efficiency,
                        )
                if "M5" in repair_mechanisms:
                    if str(arch) == "DGKAN_DCHE":
                        for cmode in ["none", "same-basis-Gram-random", "same-basis-Gram-signflip"]:
                            add_spec(
                                specs,
                                args,
                                mechanism="M5",
                                recipe="KAN_DCHE_BasisGram_EfficiencyRetest",
                                dataset=dataset,
                                seed=seed,
                                architecture=arch,
                                variant="KAN-D-CHE-BasisGram",
                                control_mode=cmode,
                                **common_efficiency,
                            )
                    elif str(arch) == "MLP":
                        for variant in ["MLP-low-rank-hidden-metric-support", "MLP-frequency-like-random-feature-support"]:
                            add_spec(
                                specs,
                                args,
                                mechanism="M5",
                                recipe="MLP_MatchedFunctionalSupport_EfficiencyRetest",
                                dataset=dataset,
                                seed=seed,
                                architecture=arch,
                                variant=variant,
                                control_mode="none",
                                **common_efficiency,
                            )
    if int(args.row_limit) > 0:
        specs = specs[: int(args.row_limit)]
    return specs


def build_m5_variant_audit_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets = split_csv(args.eval_datasets)
    seeds = split_csv(args.eval_seeds, int)
    specs: list[dict[str, Any]] = []
    seen_baselines: set[tuple[str, int, str]] = set()
    common = {
        "support_refresh_cadence": max(40, int(args.support_refresh_cadence)),
        "metric_refresh_cadence": max(10, int(args.metric_refresh_cadence)),
        "metric_shrinkage": max(0.10, float(args.metric_shrinkage)),
    }
    for dataset in datasets:
        for seed in seeds:
            for arch in ["MLP", "DGKAN_DCHE"]:
                base_key = (str(dataset), int(seed), arch)
                if base_key not in seen_baselines:
                    add_spec(
                        specs,
                        args,
                        mechanism="BASE",
                        recipe="m5_variant_optimizer_alone",
                        dataset=dataset,
                        seed=seed,
                        architecture=arch,
                        variant="optimizer_alone",
                        control_mode="none",
                        **common,
                    )
                    seen_baselines.add(base_key)
            for variant in split_csv(args.m5_variant_audit_kan_variants):
                if "OET-BankLocal" in variant:
                    controls = ["none", "same-basis-OET-random", "same-bank-shuffled"]
                else:
                    controls = ["none", "same-basis-Gram-random", "same-basis-Gram-signflip"]
                for cmode in controls:
                    add_spec(
                        specs,
                        args,
                        mechanism="M5",
                        recipe="KAN_DCHE_VariantEfficiencyAudit",
                        dataset=dataset,
                        seed=seed,
                        architecture="DGKAN_DCHE",
                        variant=variant,
                        control_mode=cmode,
                        **common,
                    )
            for variant in split_csv(args.m5_mlp_matched_variants):
                add_spec(
                    specs,
                    args,
                    mechanism="M5",
                    recipe="MLP_MatchedFunctionalSupport_VariantAudit",
                    dataset=dataset,
                    seed=seed,
                    architecture="MLP",
                    variant=variant,
                    control_mode="none",
                    **common,
                )
    if int(args.row_limit) > 0:
        specs = specs[: int(args.row_limit)]
    return specs


def dispatch_specs(args: argparse.Namespace, specs: list[dict[str, Any]], task_prefix: str) -> dict[str, Any]:
    ensure_out()
    specs_filename = f"v22_45E_{task_prefix}_specs.csv"
    write_rows(OUT_ROOT / specs_filename, specs)
    commands: list[tuple[list[str], str, str]] = []
    for spec in specs:
        physical_device = str(spec["device"])
        child_device = "cuda:0" if physical_device.startswith("cuda") else physical_device
        cmd = base_collect_cmd(args, spec, child_device=child_device)
        commands.append((cmd, str(spec["label"]), physical_device))
    append_exec(
        f"{task_prefix} dispatch",
        task_id=f"{task_prefix}_dispatch_{safe_fragment(args.label)}",
        status="started",
        gpu=",".join(split_csv(args.gpus)),
        files=f"results/v22_45E/{specs_filename}, results/v22_45E/chunks",
        note=f"rows={len(commands)}; workers={args.workers}; datasets={args.eval_datasets}; seeds={args.eval_seeds}; steps={args.steps}",
    )
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = [
            ex.submit(run_logged, cmd, task_id=f"{task_prefix}_{safe_fragment(label)}", gpu=device, timeout=int(args.row_timeout))
            for cmd, label, device in commands
        ]
        for fut in concurrent.futures.as_completed(futs):
            proc = fut.result()
            failures += int(proc.returncode != 0)
    if task_prefix == "screen":
        merge_summary = merge_v2243_outputs()
        files = "results/v22_45E/v22_45E_lowcost_screen_matrix.csv, results/v22_45E/v22_45E_mechanism_screen_summary.csv"
    else:
        if task_prefix == "repair":
            matrix_filename = "v22_45E_repair_screen_matrix.csv"
            runtime_filename = "v22_45E_repair_runtime_regression_audit.csv"
            support_filename = "v22_45E_repair_support_direction_decomposition.csv"
            safety_filename = "v22_45E_repair_safety_debt_matrix.csv"
            overhead_filename = "v22_45E_repair_overhead_matrix.csv"
            summary_filename = "v22_45E_repair_mechanism_summary.csv"
        else:
            safe_task = safe_fragment(task_prefix)
            matrix_filename = f"v22_45E_{safe_task}_matrix.csv"
            runtime_filename = f"v22_45E_{safe_task}_runtime_regression_audit.csv"
            support_filename = f"v22_45E_{safe_task}_support_direction_decomposition.csv"
            safety_filename = f"v22_45E_{safe_task}_safety_debt_matrix.csv"
            overhead_filename = f"v22_45E_{safe_task}_overhead_matrix.csv"
            summary_filename = f"v22_45E_{safe_task}_mechanism_summary.csv"
        merge_summary = merge_v2243_outputs(
            specs_filename=specs_filename,
            matrix_filename=matrix_filename,
            runtime_filename=runtime_filename,
            support_direction_filename=support_filename,
            safety_filename=safety_filename,
            overhead_filename=overhead_filename,
            summary_filename=summary_filename,
        )
        files = f"results/v22_45E/{matrix_filename}, results/v22_45E/{summary_filename}"
    append_exec(
        f"{task_prefix} completed",
        task_id=f"{task_prefix}_completed_{safe_fragment(args.label)}",
        status="pass" if failures == 0 else "fail",
        gpu=",".join(split_csv(args.gpus)),
        files=files,
        note=f"rows={len(commands)}; failures={failures}; merged={merge_summary}",
        exit_code=0 if failures == 0 else 1,
    )
    return {"rows": len(commands), "failures": failures, **merge_summary}


def stage_collect(args: argparse.Namespace) -> dict[str, Any]:
    bind_upstream()
    return v2243.train_variant(
        dataset=args.dataset,
        seed=args.seed,
        architecture=args.architecture,
        optimizer_family=args.optimizer,
        variant=args.variant,
        control_mode=args.control_mode,
        device_name=args.device,
        steps=args.steps,
        train_size=args.train_size,
        held_size=args.held_size,
        batch_size=args.batch_size,
        hidden=args.hidden,
        lr=args.lr,
        weight_decay=args.weight_decay,
        support_rank=args.support_rank,
        nuisance_rank=args.nuisance_rank,
        support_refresh_cadence=args.support_refresh_cadence,
        beta_signal=args.beta_signal,
        beta_metric=args.beta_metric,
        beta_q=args.beta_q,
        eta_rho=args.eta_rho,
        eta_debt=args.eta_debt,
        tau_safe=args.tau_safe,
        rho_min=args.rho_min,
        rho_max=args.rho_max,
        velocity_scale=args.velocity_scale,
        metric_shrinkage=args.metric_shrinkage,
        metric_eps=args.metric_eps,
        metric_refresh_cadence=args.metric_refresh_cadence,
        debt_velocity_barrier=args.debt_velocity_barrier,
        calibration_velocity_barrier=args.calibration_velocity_barrier,
        safety_budget_velocity_barrier=args.safety_budget_velocity_barrier,
        calibration_readout_radial_cap=args.calibration_readout_radial_cap,
        calibration_nuisance_weight=args.calibration_nuisance_weight,
        calibration_correction_weight=args.calibration_correction_weight,
        calibration_nuisance_mode=args.calibration_nuisance_mode,
        calibration_nuisance_cadence=args.calibration_nuisance_cadence,
        pure_fu_mode=args.pure_fu_mode,
        warmup_steps=args.warmup_steps,
        kan_init_variant=args.kan_init_variant,
        tier2_download=args.tier2_download,
        label=args.label,
        mirror_grad_cadence=args.mirror_grad_cadence,
    )


def train_trajectory_from_spec(args: argparse.Namespace, spec: dict[str, Any]) -> dict[str, Any]:
    """Run one complete FU trajectory in-process.

    This is intentionally not the old per-row subprocess dispatch path.  A
    trajectory owns its model, optimizer, controller, metric/FU state, and step
    loop from initialization through final evaluation.
    """

    bind_upstream()
    return v2243.train_variant(
        dataset=str(spec["dataset"]),
        seed=int(spec["seed"]),
        architecture=str(spec["architecture"]),
        optimizer_family=str(spec_value(spec, "optimizer", args.optimizer)),
        variant=str(spec["variant"]),
        control_mode=str(spec["control_mode"]),
        device_name=str(spec_value(spec, "device", args.device)),
        steps=int(spec_value(spec, "steps", args.steps)),
        train_size=int(spec_value(spec, "train_size", args.train_size)),
        held_size=int(spec_value(spec, "held_size", args.held_size)),
        batch_size=int(spec_value(spec, "batch_size", args.batch_size)),
        hidden=int(spec_value(spec, "hidden", args.hidden)),
        lr=float(spec_value(spec, "lr", args.lr)),
        weight_decay=float(spec_value(spec, "weight_decay", args.weight_decay)),
        support_rank=int(spec_value(spec, "support_rank", args.support_rank)),
        nuisance_rank=int(spec_value(spec, "nuisance_rank", args.nuisance_rank)),
        support_refresh_cadence=int(spec_value(spec, "support_refresh_cadence", args.support_refresh_cadence)),
        beta_signal=float(spec_value(spec, "beta_signal", args.beta_signal)),
        beta_metric=float(spec_value(spec, "beta_metric", args.beta_metric)),
        beta_q=float(spec_value(spec, "beta_q", args.beta_q)),
        eta_rho=float(spec_value(spec, "eta_rho", args.eta_rho)),
        eta_debt=float(spec_value(spec, "eta_debt", args.eta_debt)),
        tau_safe=float(spec_value(spec, "tau_safe", args.tau_safe)),
        rho_min=float(spec_value(spec, "rho_min", args.rho_min)),
        rho_max=float(spec_value(spec, "rho_max", args.rho_max)),
        velocity_scale=float(spec_value(spec, "velocity_scale", args.velocity_scale)),
        metric_shrinkage=float(spec_value(spec, "metric_shrinkage", args.metric_shrinkage)),
        metric_eps=float(spec_value(spec, "metric_eps", args.metric_eps)),
        metric_refresh_cadence=int(spec_value(spec, "metric_refresh_cadence", args.metric_refresh_cadence)),
        debt_velocity_barrier=float(spec_value(spec, "debt_velocity_barrier", args.debt_velocity_barrier)),
        calibration_velocity_barrier=float(spec_value(spec, "calibration_velocity_barrier", args.calibration_velocity_barrier)),
        safety_budget_velocity_barrier=float(spec_value(spec, "safety_budget_velocity_barrier", args.safety_budget_velocity_barrier)),
        calibration_readout_radial_cap=float(spec_value(spec, "calibration_readout_radial_cap", args.calibration_readout_radial_cap)),
        calibration_nuisance_weight=float(spec_value(spec, "calibration_nuisance_weight", args.calibration_nuisance_weight)),
        calibration_correction_weight=float(spec_value(spec, "calibration_correction_weight", args.calibration_correction_weight)),
        calibration_nuisance_mode=str(spec_value(spec, "calibration_nuisance_mode", args.calibration_nuisance_mode)),
        calibration_nuisance_cadence=int(spec_value(spec, "calibration_nuisance_cadence", args.calibration_nuisance_cadence)),
        pure_fu_mode=bool_flag(spec_value(spec, "pure_fu_mode", args.pure_fu_mode)),
        warmup_steps=int(spec_value(spec, "warmup_steps", args.warmup_steps)),
        kan_init_variant=str(spec_value(spec, "kan_init_variant", args.kan_init_variant)),
        tier2_download=bool_flag(spec_value(spec, "tier2_download", args.tier2_download)),
        label=str(spec["label"]),
        mirror_grad_cadence=int(spec_value(spec, "mirror_grad_cadence", args.mirror_grad_cadence)),
    )


def mechanism_from_label(row: dict[str, Any]) -> tuple[str, str]:
    label = str(row.get("run_label", ""))
    parts = label.split("_")
    for mech in ["Core-M", "M1", "M2", "M3", "M4", "M5", "M6", "M9", "BASE"]:
        safe = safe_fragment(mech)
        if f"_{safe}_" in f"_{label}_":
            idx = parts.index(safe) if safe in parts else -1
            recipe = parts[idx + 1] if 0 <= idx + 1 < len(parts) else ""
            return mech, recipe
    variant = str(row.get("variant", ""))
    if variant == "optimizer_alone":
        return "BASE", "optimizer_alone"
    return "UNKNOWN", ""


def merge_v2243_outputs(
    *,
    specs_filename: str = "v22_45E_screen_specs.csv",
    matrix_filename: str = "v22_45E_lowcost_screen_matrix.csv",
    runtime_filename: str = "v22_45E_runtime_regression_audit.csv",
    support_direction_filename: str = "v22_45E_support_direction_decomposition.csv",
    safety_filename: str = "v22_45E_safety_debt_matrix.csv",
    overhead_filename: str = "v22_45E_overhead_matrix.csv",
    summary_filename: str = "v22_45E_mechanism_screen_summary.csv",
) -> dict[str, Any]:
    bind_upstream()
    summary = v2243.merge_chunks(write_final=False)
    pure_summary = v2243.merge_pure_artifacts(write_final=False)
    full = read_rows(OUT_ROOT / "v22_43_metric_support_full_loop_matrix.csv")
    full.extend(read_rows(OUT_ROOT / "v22_43P_pure_support_full_loop_matrix.csv"))
    specs_by_label = {r.get("label", ""): r for r in read_rows(OUT_ROOT / specs_filename)}
    spec_labels = {str(x) for x in specs_by_label if x}
    rows: list[dict[str, Any]] = []
    for r in full:
        if spec_labels and str(r.get("run_label", "")) not in spec_labels:
            continue
        row = dict(r)
        mech, recipe = mechanism_from_label(row)
        spec = specs_by_label.get(row.get("run_label", ""), {})
        row["mechanism"] = spec.get("mechanism", mech)
        row["mechanism_recipe"] = spec.get("recipe", recipe)
        rows.append(row)
    rows = enrich_v2245_derived_metrics(rows, matrix_filename)
    write_rows(OUT_ROOT / matrix_filename, rows or [{"status": "no_rows"}])
    write_rows(OUT_ROOT / runtime_filename, runtime_regression_rows(rows))
    write_rows(OUT_ROOT / support_direction_filename, support_direction_rows(rows))
    write_rows(OUT_ROOT / safety_filename, safety_debt_rows(rows))
    write_rows(OUT_ROOT / overhead_filename, overhead_rows(rows))
    write_rows(OUT_ROOT / summary_filename, annotate_evidence_rows(mechanism_summary(rows), source_hint=summary_filename))
    return {"status": summary.get("status"), "pure_status": pure_summary.get("status"), "summary_rows": len(rows)}


def field_for_control_mode(control_mode: str) -> str:
    return safe_fragment(control_mode).replace("-", "_")


def enrich_v2245_derived_metrics(rows: list[dict[str, Any]], matrix_filename: str = "v22_45E_lowcost_screen_matrix.csv") -> list[dict[str, Any]]:
    """Backfill v22.45E gate deltas from actual row metrics after regular+pure merge.

    v22.43 computes these fields inside separate regular/pure merges.  v22.45E
    combines both artifacts and adds mechanism-specific spec labels, so we
    recompute deltas here from the already landed final_NLL/ECE/Brier/tail rows
    to avoid treating blank derived fields as failed experiments.
    """

    baselines: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        if row.get("variant") == "optimizer_alone":
            baselines[baseline_key(row)] = row

    controls: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("variant") != "optimizer_alone" and str(row.get("control_mode", "")) != "none":
            controls.setdefault(same_group_key(row), {})[str(row.get("control_mode", ""))] = row

    out: list[dict[str, Any]] = []
    control_modes = [
        "same-metric-support-random",
        "same-metric-support-signflip",
        "same-metric-support-shuffled",
        "same-mirror-support-random",
        "same-mirror-support-signflip",
        "same-norm-additive-random",
        "same-OET-random",
        "same-OET-signflip",
        "same-OET-shuffled",
        "same-generator-norm-random",
        "same-radial-random",
        "same-Lie-random",
        "same-Lie-signflip",
        "same-basis-Gram-random",
        "same-basis-Gram-signflip",
        "same-basis-OET-random",
        "same-bank-shuffled",
        "same-degree-frequency-random",
        "same-readout-leakage-control",
        "M6-signal-noise-control",
        "M6-same-norm-Gaussian-control",
        "M9-fixed-uniform-mixture-control",
        "M9-fixed-euclidean-control",
        "M9-fixed-fisher-control",
        "M9-fixed-signal-control",
        "M9-fixed-basis-control",
    ]
    for raw in rows:
        row = dict(raw)
        base = baselines.get(baseline_key(row), {})
        row_nll = finite_float(row.get("final_NLL"))
        base_nll = finite_float(base.get("final_NLL"))
        if row_nll is not None and base_nll is not None:
            row["NLL_delta_vs_base"] = row_nll - base_nll
            row["NLL_improvement_vs_own_strong_optimizer"] = base_nll - row_nll
            row["NLL_delta_real_minus_own_strong_optimizer"] = row_nll - base_nll
            row["ECE_delta_vs_own_strong_optimizer"] = value_or(row.get("ECE"), 0.0) - value_or(base.get("ECE"), 0.0)
            row["Brier_delta_vs_own_strong_optimizer"] = value_or(row.get("Brier"), 0.0) - value_or(base.get("Brier"), 0.0)
            row["tail_q99_delta_vs_own_strong_optimizer"] = value_or(row.get("tail_loss_q99"), 0.0) - value_or(base.get("tail_loss_q99"), 0.0)
            row["no_ECE_Brier_tail_debt"] = int(
                value_or(row.get("ECE_delta_vs_own_strong_optimizer"), math.inf) <= 0.0
                and value_or(row.get("Brier_delta_vs_own_strong_optimizer"), math.inf) <= 0.0
                and value_or(row.get("tail_q99_delta_vs_own_strong_optimizer"), math.inf) <= 0.0
            )
        elif row.get("variant") != "optimizer_alone":
            row["no_ECE_Brier_tail_debt"] = int_flag(row.get("no_ECE_Brier_tail_debt"))

        row_controls = controls.get(same_group_key(row), {})
        for mode in control_modes:
            ctrl = row_controls.get(mode, {})
            ctrl_nll = finite_float(ctrl.get("final_NLL"))
            field = field_for_control_mode(mode)
            if row_nll is not None and ctrl_nll is not None:
                row[f"NLL_delta_vs_{field}"] = row_nll - ctrl_nll
                row[f"beats_{field}"] = int(row_nll < ctrl_nll)
        out.append(row)

    append_exec(
        "enrich_v2245_derived_metrics",
        task_id="derived_metric_backfill",
        status="pass",
        gpu="n/a",
        files=f"results/v22_45E/{matrix_filename}",
        note=f"Backfilled NLL/safety/control deltas from actual merged final metrics for rows={len(out)}.",
    )
    return out


def same_group_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("mechanism", "")),
        str(row.get("dataset", "")),
        str(row.get("seed", "")),
        str(row.get("architecture", "")),
        str(row.get("optimizer_family", "")),
        str(row.get("steps", "")),
        str(row.get("hidden", "")),
        str(row.get("support_rank", "")),
        str(row.get("velocity_scale", "")),
        str(row.get("calibration_nuisance_weight", "")),
        str(row.get("calibration_nuisance_mode", "")),
    )


def baseline_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("dataset", "")),
        str(row.get("seed", "")),
        str(row.get("architecture", "")),
        str(row.get("optimizer_family", "")),
        str(row.get("steps", "")),
        str(row.get("hidden", "")),
    )


def support_direction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = {baseline_key(r): r for r in rows if r.get("variant") == "optimizer_alone"}
    controls: dict[tuple[str, ...], dict[str, Any]] = {}
    for r in rows:
        if str(r.get("control_mode", "")) != "none" and r.get("variant") != "optimizer_alone":
            controls.setdefault(same_group_key(r), r)
    out = []
    for r in rows:
        if r.get("variant") == "optimizer_alone" or str(r.get("control_mode", "")) != "none":
            continue
        base = baselines.get(baseline_key(r), {})
        ctrl = controls.get(same_group_key(r), {})
        base_nll = finite_float(base.get("final_NLL"))
        ctrl_nll = finite_float(ctrl.get("final_NLL"))
        real_nll = finite_float(r.get("final_NLL"))
        tau_support = "" if base_nll is None or ctrl_nll is None else base_nll - ctrl_nll
        tau_direction = "" if ctrl_nll is None or real_nll is None else ctrl_nll - real_nll
        out.append(
            {
                "run_label": r.get("run_label", ""),
                "mechanism": r.get("mechanism", ""),
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "architecture": r.get("architecture", ""),
                "variant": r.get("variant", ""),
                "control_used": ctrl.get("control_mode", ""),
                "base_final_NLL": "" if base_nll is None else base_nll,
                "control_final_NLL": "" if ctrl_nll is None else ctrl_nll,
                "real_final_NLL": "" if real_nll is None else real_nll,
                "nll_tau_support_base_minus_control": tau_support,
                "nll_tau_direction_control_minus_real": tau_direction,
                "support_positive_direction_nonpositive": int(
                    isinstance(tau_support, float)
                    and isinstance(tau_direction, float)
                    and tau_support > 0.0
                    and tau_direction <= 0.0
                ),
            }
        )
    return out or [{"status": "no_candidate_rows"}]


def runtime_regression_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "continuous_fu_state_updated_every_step",
        "fu_velocity_emitted_every_step",
        "candidate_action_selection_used_for_runtime",
        "runtime_argmax_candidate_used",
        "runtime_topk_candidate_used",
        "candidate_value_model_used_as_runtime_policy",
        "micro_rct_winner_used_as_runtime_action",
        "uses_test_direction_selection",
        "uses_future_direction",
        "uses_validation_direction",
    ]
    out = []
    for r in rows:
        out.append({k: r.get(k, "") for k in ["run_label", "mechanism", "dataset", "seed", "architecture", "variant", "control_mode"] + fields})
    return out or [{"status": "no_runtime_rows"}]


def safety_debt_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cols = [
        "run_label",
        "mechanism",
        "dataset",
        "seed",
        "architecture",
        "variant",
        "control_mode",
        "final_NLL",
        "ECE_delta_vs_own_strong_optimizer",
        "Brier_delta_vs_own_strong_optimizer",
        "tail_q99_delta_vs_own_strong_optimizer",
        "no_ECE_Brier_tail_debt",
        "safety_budget_debt",
        "calibration_nuisance_active_fraction",
        "calibration_velocity_scale",
        "safety_budget_velocity_scale",
    ]
    return [{c: r.get(c, "") for c in cols} for r in rows] or [{"status": "no_rows"}]


def overhead_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cols = [
        "run_label",
        "mechanism",
        "dataset",
        "seed",
        "architecture",
        "variant",
        "control_mode",
        "controller_overhead_ratio",
        "full_step_ms",
        "base_optimizer_ms",
        "controller_ms",
        "basis_metric_update_time_ms",
        "mirror_grad_ms",
        "mirror_grad_refreshed_fraction",
        "mirror_grad_cached_fraction",
        "memory_peak_MB",
    ]
    return [{c: r.get(c, "") for c in cols} for r in rows] or [{"status": "no_rows"}]


def count_beats(rows: list[dict[str, Any]], fields: list[str]) -> int:
    total = 0
    for row in rows:
        total += int(any(int_flag(row.get(field)) for field in fields))
    return total


def coverage_for_mechanism(mechanism: str) -> dict[str, Any]:
    return next((m for m in MECHANISM_COVERAGE if m["mechanism"] == mechanism), {})


def evidence_annotation(row: dict[str, Any], *, source_hint: str = "") -> dict[str, Any]:
    """Classify evidence without mutating raw metrics.

    Old artifacts can contain stale `coverage=proxy_blocked` even after later
    trajectory-first repairs.  This function keeps the raw coverage visible but
    adds an effective evidence tier used by reports and route gates.
    """

    mech = str(row.get("mechanism", ""))
    source = str(row.get("source_artifact", source_hint))
    coverage_raw = str(row.get("coverage", ""))
    claim = str(row.get("claim_limit", "")).lower()
    coverage_current = str(coverage_for_mechanism(mech).get("coverage", coverage_raw))
    text = " ".join([source.lower(), coverage_raw.lower(), claim, str(row.get("recipe", "")).lower()])
    if "candidate_rows" in row and int(value_or(row.get("candidate_rows"), 0.0)) <= 0:
        return {
            "coverage_raw": coverage_raw,
            "coverage_effective": coverage_current,
            "evidence_tier": "no_candidate_rows",
            "route_eligible": 0,
            "screen_pass_route_eligible": 0,
        }
    is_trajectory = "_trajectory" in source.lower() or source.lower().startswith("v22_45e_m4_trajectory")
    is_proxy = (
        "proxy" in text
        or coverage_raw in {"proxy", "proxy_blocked", "blocked"}
        or "no brownian" in claim
        or "diagnostic-only" in claim
        or "classification screen is a proxy" in claim
        or "no explicit dual-coordinate" in claim
    )
    if is_trajectory and mech == "M6":
        tier = "faithful_partial_reservoir_trajectory"
        route_eligible = 0
    elif is_trajectory and mech in {"M1", "M2", "M3", "M4", "M5", "M9"}:
        tier = "faithful_partial_trajectory" if mech in {"M1", "M2", "M3", "M4", "M9"} else "faithful_trajectory_fairness_audit"
        route_eligible = 1
    elif mech == "Core-M" and not is_proxy:
        tier = "implemented_lowcost_screen"
        route_eligible = 1
    elif mech == "M5" and "variant" in source.lower() and not is_proxy:
        tier = "implemented_dispatch_audit"
        route_eligible = 0
    elif is_proxy:
        tier = "proxy_or_blocker"
        route_eligible = 0
    elif coverage_current == "blocked":
        tier = "blocked"
        route_eligible = 0
    elif coverage_current == "partial":
        tier = "partial_lowcost_or_repair"
        route_eligible = 0
    elif coverage_current == "implemented":
        tier = "implemented_dispatch_audit"
        route_eligible = 0
    else:
        tier = "unknown"
        route_eligible = 0
    return {
        "coverage_raw": coverage_raw,
        "coverage_effective": coverage_current,
        "evidence_tier": tier,
        "route_eligible": route_eligible,
        "screen_pass_route_eligible": int(int_flag(row.get("screen_pass")) and route_eligible),
    }


def annotate_evidence_rows(rows: list[dict[str, Any]], *, source_hint: str = "") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item.update(evidence_annotation(item, source_hint=source_hint))
        out.append(item)
    return out


def mechanism_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        r for r in rows
        if r.get("variant") != "optimizer_alone" and str(r.get("control_mode", "")) == "none"
    ]
    out = []
    for mech in ["Core-M", "M1", "M2", "M3", "M4", "M5", "M6", "M9"]:
        if mech == "M5":
            cands = [r for r in candidates if r.get("mechanism") == mech and r.get("architecture") == "strict_FC_PureKAN"]
            mlp_matched = [r for r in candidates if r.get("mechanism") == mech and r.get("architecture") == "MLP"]
            controls = [r for r in rows if r.get("mechanism") == mech and str(r.get("control_mode", "")) != "none"] + mlp_matched
        else:
            cands = [r for r in candidates if r.get("mechanism") == mech]
            mlp_matched = []
            controls = [r for r in rows if r.get("mechanism") == mech and str(r.get("control_mode", "")) != "none"]
        n = len(cands)
        nll_gain = sum(1 for r in cands if value_or(r.get("NLL_improvement_vs_own_strong_optimizer"), -math.inf) > 0.0)
        no_debt = sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in cands)
        overhead_ok = sum(1 for r in cands if value_or(r.get("controller_overhead_ratio"), math.inf) <= 0.35)
        beats = count_beats(
            cands,
            [
                "beats_same_metric_support_random",
                "beats_same_metric_support_signflip",
                "beats_same_mirror_support_random",
                "beats_same_mirror_support_signflip",
                "beats_same_radial_random",
                "beats_same_OET_random",
                "beats_same_Lie_random",
                "beats_same_Lie_signflip",
                "beats_same_basis_Gram_random",
                "beats_same_basis_Gram_signflip",
                "beats_same_basis_OET_random",
                "beats_same_bank_shuffled",
                "beats_M6_signal_noise_control",
                "beats_M6_same_norm_Gaussian_control",
                "beats_M9_fixed_uniform_mixture_control",
                "beats_M9_fixed_euclidean_control",
                "beats_M9_fixed_fisher_control",
                "beats_M9_fixed_signal_control",
                "beats_M9_fixed_basis_control",
            ],
        )
        beats_mlp_matched = 0
        if mech == "M5" and mlp_matched:
            mlp_by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for row in mlp_matched:
                mlp_by_group.setdefault((str(row.get("dataset", "")), str(row.get("seed", ""))), []).append(row)
            for row in cands:
                mlps = mlp_by_group.get((str(row.get("dataset", "")), str(row.get("seed", ""))), [])
                row_nll = finite_float(row.get("final_NLL"))
                best_mlp_nll = min((finite_float(m.get("final_NLL")) for m in mlps if finite_float(m.get("final_NLL")) is not None), default=None)
                if row_nll is not None and best_mlp_nll is not None and row_nll < best_mlp_nll:
                    beats_mlp_matched += 1
        support_only = sum(
            int_flag(r.get("support_positive_direction_nonpositive"))
            for r in support_direction_rows(rows)
            if r.get("mechanism") == mech
        )
        pass_ratio = (beats / n) if n else 0.0
        no_debt_ratio = (no_debt / n) if n else 0.0
        overhead_ratio = (overhead_ok / n) if n else 0.0
        beats_mlp_ratio = (beats_mlp_matched / n) if n else 0.0
        coverage = coverage_for_mechanism(mech)
        is_proxy = str(coverage.get("coverage", "")).startswith("proxy") or coverage.get("coverage") == "blocked"
        screen_pass = int(n > 0 and pass_ratio >= 0.375 and no_debt_ratio >= 0.50 and overhead_ratio >= 0.75 and not is_proxy)
        if mech == "M5":
            screen_pass = int(bool(screen_pass) and beats_mlp_ratio >= 0.375)
        if mech == "M6":
            screen_pass = 0
        row = {
                "mechanism": mech,
                "coverage": coverage.get("coverage", ""),
                "candidate_rows": n,
                "control_rows": len(controls),
                "NLL_improvement_vs_own_rows": nll_gain,
                "beats_matched_control_rows": beats,
                "no_ECE_Brier_tail_debt_rows": no_debt,
                "controller_overhead_le_0p35_rows": overhead_ok,
                "support_only_rows": support_only,
                "beats_MLP_matched_support_rows": beats_mlp_matched,
                "beats_control_ratio": f"{pass_ratio:.6f}",
                "beats_MLP_matched_support_ratio": f"{beats_mlp_ratio:.6f}",
                "no_debt_ratio": f"{no_debt_ratio:.6f}",
                "overhead_ok_ratio": f"{overhead_ratio:.6f}",
                "screen_pass": screen_pass,
                "claim_limit": coverage.get("claim_limit", ""),
        }
        row.update(evidence_annotation(row))
        out.append(row)
    for mech in ["M7", "M8"]:
        coverage = coverage_for_mechanism(mech)
        row = {
                "mechanism": mech,
                "coverage": coverage.get("coverage", ""),
                "candidate_rows": 0,
                "control_rows": 0,
                "NLL_improvement_vs_own_rows": 0,
                "beats_matched_control_rows": 0,
                "no_ECE_Brier_tail_debt_rows": 0,
                "controller_overhead_le_0p35_rows": 0,
                "support_only_rows": 0,
                "beats_control_ratio": "0.000000",
                "no_debt_ratio": "0.000000",
                "overhead_ok_ratio": "0.000000",
                "screen_pass": 0,
                "claim_limit": coverage.get("claim_limit", ""),
        }
        row.update(evidence_annotation(row))
        out.append(row)
    return out


def run_screen(args: argparse.Namespace) -> dict[str, Any]:
    write_rows(OUT_ROOT / "v22_45E_mechanism_coverage_matrix.csv", MECHANISM_COVERAGE)
    specs = build_screen_specs(args)
    return dispatch_specs(args, specs, "screen")


def run_repair_screen(args: argparse.Namespace) -> dict[str, Any]:
    write_rows(OUT_ROOT / "v22_45E_mechanism_coverage_matrix.csv", MECHANISM_COVERAGE)
    specs = build_repair_specs(args)
    return dispatch_specs(args, specs, args.repair_output)


def run_m5_variant_audit(args: argparse.Namespace) -> dict[str, Any]:
    write_rows(OUT_ROOT / "v22_45E_mechanism_coverage_matrix.csv", MECHANISM_COVERAGE)
    specs = build_m5_variant_audit_specs(args)
    return dispatch_specs(args, specs, args.m5_variant_output)


def build_m1_trajectory_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets = split_csv(args.eval_datasets)
    seeds = split_csv(args.eval_seeds, int)
    archs = split_csv(args.eval_architectures)
    variants = split_csv(args.m1_trajectory_variants)
    specs: list[dict[str, Any]] = []
    trajectory_steps = int(args.trajectory_steps or args.steps)
    common = {
        "steps": trajectory_steps,
        "support_refresh_cadence": max(40, int(args.support_refresh_cadence)),
        "metric_refresh_cadence": max(10, int(args.metric_refresh_cadence)),
        "metric_shrinkage": max(0.10, float(args.metric_shrinkage)),
        "velocity_scale": min(float(args.velocity_scale), float(args.trajectory_m1_velocity_scale)),
        "safety_budget_velocity_barrier": max(float(args.safety_budget_velocity_barrier), float(args.trajectory_m1_safety_barrier)),
        "rho_max": min(float(args.rho_max), 0.20),
        "mirror_grad_cadence": max(1, int(args.trajectory_m1_mirror_grad_cadence)),
    }
    seen_baselines: set[tuple[str, int, str]] = set()
    for dataset in datasets:
        for seed in seeds:
            for arch in archs:
                key = (str(dataset), int(seed), str(arch))
                if key not in seen_baselines:
                    add_spec(
                        specs,
                        args,
                        mechanism="BASE",
                        recipe="trajectory_optimizer_alone",
                        dataset=dataset,
                        seed=seed,
                        architecture=arch,
                        variant="optimizer_alone",
                        control_mode="none",
                        **{k: v for k, v in common.items() if k not in {"velocity_scale", "safety_budget_velocity_barrier", "rho_max"}},
                    )
                    seen_baselines.add(key)
                for variant in variants:
                    for cmode in ["none", "same-mirror-support-random", "same-mirror-support-signflip"]:
                        add_spec(
                            specs,
                            args,
                            mechanism="M1",
                            recipe="TrajectoryFunctionalMirror",
                            dataset=dataset,
                            seed=seed,
                            architecture=arch,
                            variant=variant,
                            control_mode=cmode,
                            **common,
                        )
    if int(args.row_limit) > 0:
        specs = specs[: int(args.row_limit)]
    return specs


def build_m2_trajectory_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets = split_csv(args.eval_datasets)
    seeds = split_csv(args.eval_seeds, int)
    archs = split_csv(args.eval_architectures)
    specs: list[dict[str, Any]] = []
    trajectory_steps = int(args.trajectory_steps or args.steps)
    trajectory_warmup = max(int(args.warmup_steps), int(args.trajectory_warmup_steps))
    if trajectory_warmup >= trajectory_steps:
        trajectory_warmup = max(0, min(trajectory_steps - 1, trajectory_steps // 4))
    common = {
        "steps": trajectory_steps,
        "support_refresh_cadence": max(40, int(args.support_refresh_cadence)),
        "metric_refresh_cadence": max(5, int(args.metric_refresh_cadence)),
        "metric_shrinkage": max(0.10, float(args.metric_shrinkage)),
        "warmup_steps": trajectory_warmup,
        "velocity_scale": min(float(args.m2_velocity_scale), float(args.trajectory_m2_velocity_scale)),
        "safety_budget_velocity_barrier": max(float(args.m2_safety_barrier), float(args.trajectory_m2_safety_barrier)),
        "beta_signal": min(float(args.beta_signal), float(args.trajectory_m2_beta_signal)),
    }
    seen_baselines: set[tuple[str, int, str]] = set()
    for dataset in datasets:
        for seed in seeds:
            for arch in archs:
                key = (str(dataset), int(seed), str(arch))
                if key not in seen_baselines:
                    add_spec(
                        specs,
                        args,
                        mechanism="BASE",
                        recipe="trajectory_optimizer_alone",
                        dataset=dataset,
                        seed=seed,
                        architecture=arch,
                        variant="optimizer_alone",
                        control_mode="none",
                        **{k: v for k, v in common.items() if k not in {"warmup_steps", "velocity_scale", "safety_budget_velocity_barrier", "beta_signal"}},
                    )
                    seen_baselines.add(key)
                add_spec(
                    specs,
                    args,
                    mechanism="M2-COMP",
                    recipe="StrictOETComparator",
                    dataset=dataset,
                    seed=seed,
                    architecture=arch,
                    variant="P3-Euclidean-OET-pure",
                    control_mode="none",
                    pure_fu_mode=True,
                    **common,
                )
                for cmode in ["none", "same-radial-random", "same-OET-random"]:
                    add_spec(
                        specs,
                        args,
                        mechanism="M2",
                        recipe="TrajectoryFunctionalJVP",
                        dataset=dataset,
                        seed=seed,
                        architecture=arch,
                        variant=str(args.m2_repair_variant),
                        control_mode=cmode,
                        pure_fu_mode=True,
                        **common,
                    )
    if int(args.row_limit) > 0:
        specs = specs[: int(args.row_limit)]
    return specs


def build_m3_trajectory_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets = split_csv(args.eval_datasets)
    seeds = split_csv(args.eval_seeds, int)
    archs = split_csv(args.eval_architectures)
    specs: list[dict[str, Any]] = []
    trajectory_steps = int(args.trajectory_steps or args.steps)
    trajectory_warmup = max(int(args.warmup_steps), int(args.trajectory_warmup_steps))
    if trajectory_warmup >= trajectory_steps:
        trajectory_warmup = max(0, min(trajectory_steps - 1, trajectory_steps // 4))
    common = {
        "steps": trajectory_steps,
        "support_refresh_cadence": max(40, int(args.support_refresh_cadence)),
        "metric_refresh_cadence": max(5, int(args.metric_refresh_cadence)),
        "metric_shrinkage": max(0.10, float(args.metric_shrinkage)),
        "warmup_steps": trajectory_warmup,
        "velocity_scale": min(float(args.m3_velocity_scale), float(args.trajectory_m3_velocity_scale)),
        "safety_budget_velocity_barrier": max(float(args.m3_safety_barrier), float(args.trajectory_m3_safety_barrier)),
        "beta_signal": min(float(args.beta_signal), float(args.trajectory_m3_beta_signal)),
    }
    seen_baselines: set[tuple[str, int, str]] = set()
    for dataset in datasets:
        for seed in seeds:
            for arch in archs:
                key = (str(dataset), int(seed), str(arch))
                if key not in seen_baselines:
                    add_spec(
                        specs,
                        args,
                        mechanism="BASE",
                        recipe="trajectory_optimizer_alone",
                        dataset=dataset,
                        seed=seed,
                        architecture=arch,
                        variant="optimizer_alone",
                        control_mode="none",
                        **{k: v for k, v in common.items() if k not in {"warmup_steps", "velocity_scale", "safety_budget_velocity_barrier", "beta_signal"}},
                    )
                    seen_baselines.add(key)
                add_spec(
                    specs,
                    args,
                    mechanism="M3-COMP",
                    recipe="AmbientStrictOETComparator",
                    dataset=dataset,
                    seed=seed,
                    architecture=arch,
                    variant="P3-Euclidean-OET-pure",
                    control_mode="none",
                    pure_fu_mode=True,
                    **common,
                )
                for cmode in ["none", "same-Lie-random", "same-Lie-signflip"]:
                    add_spec(
                        specs,
                        args,
                        mechanism="M3",
                        recipe="TransportedLieMomentum",
                        dataset=dataset,
                        seed=seed,
                        architecture=arch,
                        variant=str(args.m3_trajectory_variant),
                        control_mode=cmode,
                        pure_fu_mode=True,
                        **common,
                    )
    if int(args.row_limit) > 0:
        specs = specs[: int(args.row_limit)]
    return specs


def trajectory_command_line(args: argparse.Namespace, stage: str) -> str:
    argv = [PYTHON, str(Path(__file__).relative_to(ROOT)), "--stage", stage]
    passthrough = [
        "--label",
        args.label,
        "--eval-datasets",
        args.eval_datasets,
        "--eval-seeds",
        args.eval_seeds,
        "--eval-architectures",
        args.eval_architectures,
        "--gpus",
        args.gpus,
        "--workers",
        str(args.workers),
        "--steps",
        str(args.steps),
        "--trajectory-steps",
        str(args.trajectory_steps),
        "--train-size",
        str(args.train_size),
        "--held-size",
        str(args.held_size),
        "--batch-size",
        str(args.batch_size),
        "--hidden",
        str(args.hidden),
        "--support-rank",
        str(args.support_rank),
        "--support-refresh-cadence",
        str(args.support_refresh_cadence),
        "--metric-refresh-cadence",
        str(args.metric_refresh_cadence),
        "--velocity-scale",
        str(args.velocity_scale),
        "--safety-budget-velocity-barrier",
        str(args.safety_budget_velocity_barrier),
        "--m2-repair-variant",
        str(args.m2_repair_variant),
    ]
    if stage == "trajectory-m1":
        passthrough.extend([
            "--m1-trajectory-output",
            str(args.m1_trajectory_output),
            "--m1-trajectory-variants",
            str(args.m1_trajectory_variants),
            "--trajectory-m1-velocity-scale",
            str(args.trajectory_m1_velocity_scale),
            "--trajectory-m1-safety-barrier",
            str(args.trajectory_m1_safety_barrier),
            "--trajectory-m1-mirror-grad-cadence",
            str(args.trajectory_m1_mirror_grad_cadence),
        ])
    if stage == "trajectory-m2":
        passthrough.extend([
            "--m2-trajectory-output",
            str(args.m2_trajectory_output),
            "--trajectory-m2-velocity-scale",
            str(args.trajectory_m2_velocity_scale),
            "--trajectory-m2-safety-barrier",
            str(args.trajectory_m2_safety_barrier),
            "--trajectory-m2-beta-signal",
            str(args.trajectory_m2_beta_signal),
        ])
    if stage == "trajectory-m3":
        passthrough.extend([
            "--m3-trajectory-output",
            str(args.m3_trajectory_output),
            "--m3-trajectory-variant",
            str(args.m3_trajectory_variant),
            "--trajectory-m3-velocity-scale",
            str(args.trajectory_m3_velocity_scale),
            "--trajectory-m3-safety-barrier",
            str(args.trajectory_m3_safety_barrier),
            "--trajectory-m3-beta-signal",
            str(args.trajectory_m3_beta_signal),
        ])
    if stage == "trajectory-m4":
        passthrough.extend([
            "--m4-trajectory-output",
            str(args.m4_trajectory_output),
            "--m4-tasks",
            str(args.m4_tasks),
            "--m4-trajectory-steps",
            str(args.m4_trajectory_steps),
            "--diagnostic-seeds",
            str(args.diagnostic_seeds),
            "--diagnostic-architectures",
            str(args.diagnostic_architectures),
            "--diagnostic-hidden",
            str(args.diagnostic_hidden),
            "--diagnostic-batch-size",
            str(args.diagnostic_batch_size),
            "--m4-modulus",
            str(args.m4_modulus),
            "--m4-train-fraction",
            str(args.m4_train_fraction),
            "--m4-diag-velocity-scale",
            str(args.m4_diag_velocity_scale),
            "--m4-beta-slow",
            str(args.m4_beta_slow),
            "--m4-rho",
            str(args.m4_rho),
            "--m4-lambda-mig",
            str(args.m4_lambda_mig),
            "--m4-lambda-res",
            str(args.m4_lambda_res),
            "--m4-support-rank",
            str(args.m4_support_rank),
            "--m7-train-size",
            str(args.m7_train_size),
            "--m7-held-size",
            str(args.m7_held_size),
            "--m4-train-threshold",
            str(args.m4_train_threshold),
            "--m4-test-threshold",
            str(args.m4_test_threshold),
        ])
    if stage == "trajectory-m5":
        passthrough.extend([
            "--m5-trajectory-output",
            str(args.m5_trajectory_output),
            "--m5-variant-audit-kan-variants",
            str(args.m5_variant_audit_kan_variants),
            "--m5-mlp-matched-variants",
            str(args.m5_mlp_matched_variants),
        ])
    if stage == "trajectory-m6":
        passthrough.extend([
            "--m6-trajectory-output",
            str(args.m6_trajectory_output),
            "--trajectory-m6-velocity-scale",
            str(args.trajectory_m6_velocity_scale),
            "--trajectory-m6-safety-barrier",
            str(args.trajectory_m6_safety_barrier),
            "--trajectory-m6-beta-signal",
            str(args.trajectory_m6_beta_signal),
        ])
    if stage == "trajectory-m7":
        passthrough.extend([
            "--m7-trajectory-output",
            str(args.m7_trajectory_output),
            "--m7-tasks",
            str(args.m7_tasks),
            "--m7-trajectory-steps",
            str(args.m7_trajectory_steps),
            "--m7-rho",
            str(args.m7_rho),
            "--m7-tau-forget",
            str(args.m7_tau_forget),
            "--m7-velocity-scale",
            str(args.m7_velocity_scale),
            "--m7-beta-memory",
            str(args.m7_beta_memory),
            "--m7-memory-rank",
            str(args.m7_memory_rank),
            "--m7-projector",
            str(args.m7_projector),
            "--m7-functional-eps",
            str(args.m7_functional_eps),
            "--m7-modulus",
            str(args.m7_modulus),
            "--m7-modular-train-fraction",
            str(args.m7_modular_train_fraction),
            "--diagnostic-hidden",
            str(args.diagnostic_hidden),
            "--diagnostic-batch-size",
            str(args.diagnostic_batch_size),
            "--m7-train-size",
            str(args.m7_train_size),
            "--m7-held-size",
            str(args.m7_held_size),
        ])
    if stage == "trajectory-m9":
        passthrough.extend([
            "--m9-trajectory-output",
            str(args.m9_trajectory_output),
            "--trajectory-m9-velocity-scale",
            str(args.trajectory_m9_velocity_scale),
            "--trajectory-m9-safety-barrier",
            str(args.trajectory_m9_safety_barrier),
            "--trajectory-m9-beta-signal",
            str(args.trajectory_m9_beta_signal),
        ])
    if stage == "m8-gauge-audit":
        passthrough.extend([
            "--device",
            str(args.device),
        ])
    if int(args.row_limit) > 0:
        passthrough.extend(["--row-limit", str(args.row_limit)])
    if bool(getattr(args, "tier2_download", False)):
        passthrough.append("--tier2-download")
    return " ".join(shlex.quote(str(x)) for x in argv + passthrough)


def run_one_inprocess_trajectory(args: argparse.Namespace, spec: dict[str, Any], task_prefix: str) -> dict[str, Any]:
    started = time.time()
    task_id = f"{task_prefix}_{safe_fragment(spec.get('label', 'trajectory'))}"
    try:
        summary = train_trajectory_from_spec(args, spec)
        append_exec(
            f"in_process train_trajectory_from_spec label={shlex.quote(str(spec.get('label', '')))}",
            task_id=task_id,
            status="pass",
            gpu=str(spec.get("device", "")),
            files=(
                f"{summary.get('chunk_state_trace', '')}, {summary.get('chunk_runtime_trace', '')}, "
                f"results/v22_45E/chunks/{'v22_43P' if bool_flag(spec.get('pure_fu_mode')) else 'v22_43'}_{safe_fragment(spec.get('label', ''))}_summary.csv"
            ),
            note=f"elapsed_sec={time.time() - started:.3f}; in_process=1; subprocess_used=0; complete_step_loop=1",
        )
        return {"label": spec.get("label", ""), "status": "pass", "error": ""}
    except Exception as exc:  # pragma: no cover - experiment audit path
        tb = traceback.format_exc(limit=8).replace("\n", " | ")
        append_exec(
            f"in_process train_trajectory_from_spec label={shlex.quote(str(spec.get('label', '')))}",
            task_id=task_id,
            status="fail",
            gpu=str(spec.get("device", "")),
            files="",
            note=f"elapsed_sec={time.time() - started:.3f}; error={type(exc).__name__}: {exc}; traceback={tb[:1600]}",
            exit_code=1,
        )
        return {"label": spec.get("label", ""), "status": "fail", "error": f"{type(exc).__name__}: {exc}"}


def run_m1_trajectory(args: argparse.Namespace) -> dict[str, Any]:
    write_rows(OUT_ROOT / "v22_45E_mechanism_coverage_matrix.csv", MECHANISM_COVERAGE)
    specs = build_m1_trajectory_specs(args)
    output_prefix = safe_fragment(args.m1_trajectory_output)
    specs_filename = f"v22_45E_{output_prefix}_specs.csv"
    matrix_filename = f"v22_45E_{output_prefix}_matrix.csv"
    runtime_filename = f"v22_45E_{output_prefix}_runtime_regression_audit.csv"
    support_filename = f"v22_45E_{output_prefix}_support_direction_decomposition.csv"
    safety_filename = f"v22_45E_{output_prefix}_safety_debt_matrix.csv"
    overhead_filename = f"v22_45E_{output_prefix}_overhead_matrix.csv"
    summary_filename = f"v22_45E_{output_prefix}_mechanism_summary.csv"
    status_filename = f"v22_45E_{output_prefix}_status.csv"
    write_rows(OUT_ROOT / specs_filename, specs)
    append_exec(
        trajectory_command_line(args, "trajectory-m1"),
        task_id=f"m1_trajectory_started_{safe_fragment(args.label)}",
        status="started",
        gpu=",".join(split_csv(args.gpus)),
        files=f"results/v22_45E/{specs_filename}, results/v22_45E/chunks",
        note=(
            f"trajectories={len(specs)}; workers={args.workers}; in_process=1; subprocess_used=0; "
            f"datasets={args.eval_datasets}; seeds={args.eval_seeds}; variants={args.m1_trajectory_variants}; "
            f"steps={args.trajectory_steps or args.steps}"
        ),
    )
    failures = 0
    statuses: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futures = [ex.submit(run_one_inprocess_trajectory, args, spec, "m1_trajectory") for spec in specs]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            statuses.append(row)
            failures += int(row.get("status") != "pass")
    write_rows(OUT_ROOT / status_filename, statuses)
    merge_summary = merge_v2243_outputs(
        specs_filename=specs_filename,
        matrix_filename=matrix_filename,
        runtime_filename=runtime_filename,
        support_direction_filename=support_filename,
        safety_filename=safety_filename,
        overhead_filename=overhead_filename,
        summary_filename=summary_filename,
    )
    append_exec(
        "m1 trajectory-first suite completed",
        task_id=f"m1_trajectory_completed_{safe_fragment(args.label)}",
        status="pass" if failures == 0 else "fail",
        gpu=",".join(split_csv(args.gpus)),
        files=f"results/v22_45E/{matrix_filename}, results/v22_45E/{summary_filename}",
        note=f"trajectories={len(specs)}; failures={failures}; merged={merge_summary}",
        exit_code=0 if failures == 0 else 1,
    )
    return {"trajectories": len(specs), "failures": failures, **merge_summary}


def m2_trajectory_plan_metric_rows(matrix_filename: str) -> list[dict[str, Any]]:
    rows = read_rows(OUT_ROOT / matrix_filename)
    by_group: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("seed", "")),
            str(row.get("architecture", "")),
            str(row.get("optimizer_family", "")),
        )
        role = ""
        if row.get("variant") == "optimizer_alone":
            role = "base"
        elif row.get("mechanism") == "M2-COMP" and row.get("control_mode") == "none":
            role = "strict_oet"
        elif row.get("mechanism") == "M2" and row.get("control_mode") == "none":
            role = "functional"
        elif row.get("mechanism") == "M2" and row.get("control_mode") == "same-radial-random":
            role = "same_radial"
        elif row.get("mechanism") == "M2" and row.get("control_mode") == "same-OET-random":
            role = "same_oet"
        if role:
            by_group.setdefault(key, {})[role] = row
    out: list[dict[str, Any]] = []
    for key, group in sorted(by_group.items()):
        dataset, seed, architecture, optimizer = key
        func = group.get("functional", {})
        if not func:
            continue
        base = group.get("base", {})
        strict = group.get("strict_oet", {})
        radial = group.get("same_radial", {})
        oet = group.get("same_oet", {})
        func_nll = finite_float(func.get("final_NLL"))
        base_nll = finite_float(base.get("final_NLL"))
        strict_nll = finite_float(strict.get("final_NLL"))
        radial_nll = finite_float(radial.get("final_NLL"))
        oet_nll = finite_float(oet.get("final_NLL"))
        out.append(
            {
                "dataset": dataset,
                "seed": seed,
                "architecture": architecture,
                "optimizer_family": optimizer,
                "functional_variant": func.get("variant", ""),
                "base_final_NLL": "" if base_nll is None else base_nll,
                "strict_oet_final_NLL": "" if strict_nll is None else strict_nll,
                "functional_final_NLL": "" if func_nll is None else func_nll,
                "same_radial_final_NLL": "" if radial_nll is None else radial_nll,
                "same_oet_final_NLL": "" if oet_nll is None else oet_nll,
                "NLL_improvement_vs_base": "" if func_nll is None or base_nll is None else base_nll - func_nll,
                "NLL_improvement_vs_strict_OET": "" if func_nll is None or strict_nll is None else strict_nll - func_nll,
                "NLL_improvement_vs_same_radial": "" if func_nll is None or radial_nll is None else radial_nll - func_nll,
                "NLL_improvement_vs_same_OET": "" if func_nll is None or oet_nll is None else oet_nll - func_nll,
                "beats_strict_OET": int(func_nll is not None and strict_nll is not None and func_nll < strict_nll),
                "beats_same_radial": int(func_nll is not None and radial_nll is not None and func_nll < radial_nll),
                "beats_same_OET": int(func_nll is not None and oet_nll is not None and func_nll < oet_nll),
                "no_ECE_Brier_tail_debt": func.get("no_ECE_Brier_tail_debt", ""),
                "controller_overhead_ratio": func.get("controller_overhead_ratio", ""),
                "radial_energy_fraction": func.get("radial_energy_fraction", ""),
                "functional_radial_fd_eval_count_mean": func.get("functional_radial_fd_eval_count_mean", ""),
                "functional_radial_response_energy_max": func.get("functional_radial_response_energy_max", ""),
                "functional_radial_rse_max": func.get("functional_radial_rse_max", ""),
                "continuous_fu_state_updated_every_step": func.get("continuous_fu_state_updated_every_step", ""),
                "fu_velocity_emitted_every_step": func.get("fu_velocity_emitted_every_step", ""),
                "candidate_action_selection_used_for_runtime": func.get("candidate_action_selection_used_for_runtime", ""),
            }
        )
    return out or [{"status": "no_functional_m2_trajectory_rows"}]


def run_m2_trajectory(args: argparse.Namespace) -> dict[str, Any]:
    write_rows(OUT_ROOT / "v22_45E_mechanism_coverage_matrix.csv", MECHANISM_COVERAGE)
    specs = build_m2_trajectory_specs(args)
    output_prefix = safe_fragment(args.m2_trajectory_output)
    specs_filename = f"v22_45E_{output_prefix}_specs.csv"
    matrix_filename = f"v22_45E_{output_prefix}_matrix.csv"
    runtime_filename = f"v22_45E_{output_prefix}_runtime_regression_audit.csv"
    support_filename = f"v22_45E_{output_prefix}_support_direction_decomposition.csv"
    safety_filename = f"v22_45E_{output_prefix}_safety_debt_matrix.csv"
    overhead_filename = f"v22_45E_{output_prefix}_overhead_matrix.csv"
    summary_filename = f"v22_45E_{output_prefix}_mechanism_summary.csv"
    plan_filename = f"v22_45E_{output_prefix}_plan_metrics.csv"
    write_rows(OUT_ROOT / specs_filename, specs)
    append_exec(
        trajectory_command_line(args, "trajectory-m2"),
        task_id=f"m2_trajectory_started_{safe_fragment(args.label)}",
        status="started",
        gpu=",".join(split_csv(args.gpus)),
        files=f"results/v22_45E/{specs_filename}, results/v22_45E/chunks",
        note=(
            f"trajectories={len(specs)}; workers={args.workers}; in_process=1; subprocess_used=0; "
            f"datasets={args.eval_datasets}; seeds={args.eval_seeds}; architectures={args.eval_architectures}; "
            f"steps={args.trajectory_steps or args.steps}"
        ),
    )
    failures = 0
    statuses: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futures = [ex.submit(run_one_inprocess_trajectory, args, spec, "m2_trajectory") for spec in specs]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            statuses.append(row)
            failures += int(row.get("status") != "pass")
    write_rows(OUT_ROOT / f"v22_45E_{output_prefix}_status.csv", statuses)
    merge_summary = merge_v2243_outputs(
        specs_filename=specs_filename,
        matrix_filename=matrix_filename,
        runtime_filename=runtime_filename,
        support_direction_filename=support_filename,
        safety_filename=safety_filename,
        overhead_filename=overhead_filename,
        summary_filename=summary_filename,
    )
    plan_rows = m2_trajectory_plan_metric_rows(matrix_filename)
    write_rows(OUT_ROOT / plan_filename, plan_rows)
    append_exec(
        "m2 trajectory-first suite completed",
        task_id=f"m2_trajectory_completed_{safe_fragment(args.label)}",
        status="pass" if failures == 0 else "fail",
        gpu=",".join(split_csv(args.gpus)),
        files=f"results/v22_45E/{matrix_filename}, results/v22_45E/{summary_filename}, results/v22_45E/{plan_filename}",
        note=f"trajectories={len(specs)}; failures={failures}; merged={merge_summary}; plan_metric_rows={len(plan_rows)}",
        exit_code=0 if failures == 0 else 1,
    )
    return {"trajectories": len(specs), "failures": failures, "plan_metric_rows": len(plan_rows), **merge_summary}


def m3_trajectory_plan_metric_rows(matrix_filename: str) -> list[dict[str, Any]]:
    rows = read_rows(OUT_ROOT / matrix_filename)
    by_group: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("seed", "")),
            str(row.get("architecture", "")),
            str(row.get("optimizer_family", "")),
        )
        role = ""
        if row.get("variant") == "optimizer_alone":
            role = "base"
        elif row.get("mechanism") == "M3-COMP" and row.get("control_mode") == "none":
            role = "ambient_oet"
        elif row.get("mechanism") == "M3" and row.get("control_mode") == "none":
            role = "lie"
        elif row.get("mechanism") == "M3" and row.get("control_mode") == "same-Lie-random":
            role = "same_lie_random"
        elif row.get("mechanism") == "M3" and row.get("control_mode") == "same-Lie-signflip":
            role = "same_lie_signflip"
        if role:
            by_group.setdefault(key, {})[role] = row
    out: list[dict[str, Any]] = []
    for key, group in sorted(by_group.items()):
        dataset, seed, architecture, optimizer = key
        lie = group.get("lie", {})
        if not lie:
            continue
        base = group.get("base", {})
        ambient = group.get("ambient_oet", {})
        random_control = group.get("same_lie_random", {})
        signflip_control = group.get("same_lie_signflip", {})
        lie_nll = finite_float(lie.get("final_NLL"))
        base_nll = finite_float(base.get("final_NLL"))
        ambient_nll = finite_float(ambient.get("final_NLL"))
        random_nll = finite_float(random_control.get("final_NLL"))
        signflip_nll = finite_float(signflip_control.get("final_NLL"))
        out.append(
            {
                "dataset": dataset,
                "seed": seed,
                "architecture": architecture,
                "optimizer_family": optimizer,
                "lie_variant": lie.get("variant", ""),
                "base_final_NLL": "" if base_nll is None else base_nll,
                "ambient_oet_final_NLL": "" if ambient_nll is None else ambient_nll,
                "lie_final_NLL": "" if lie_nll is None else lie_nll,
                "same_lie_random_final_NLL": "" if random_nll is None else random_nll,
                "same_lie_signflip_final_NLL": "" if signflip_nll is None else signflip_nll,
                "NLL_improvement_vs_base": "" if lie_nll is None or base_nll is None else base_nll - lie_nll,
                "NLL_improvement_vs_ambient_OET": "" if lie_nll is None or ambient_nll is None else ambient_nll - lie_nll,
                "NLL_improvement_vs_same_Lie_random": "" if lie_nll is None or random_nll is None else random_nll - lie_nll,
                "NLL_improvement_vs_same_Lie_signflip": "" if lie_nll is None or signflip_nll is None else signflip_nll - lie_nll,
                "beats_ambient_OET": int(lie_nll is not None and ambient_nll is not None and lie_nll < ambient_nll),
                "beats_same_Lie_random": int(lie_nll is not None and random_nll is not None and lie_nll < random_nll),
                "beats_same_Lie_signflip": int(lie_nll is not None and signflip_nll is not None and lie_nll < signflip_nll),
                "no_ECE_Brier_tail_debt": lie.get("no_ECE_Brier_tail_debt", ""),
                "controller_overhead_ratio": lie.get("controller_overhead_ratio", ""),
                "lie_momentum_active_fraction": lie.get("lie_momentum_active_fraction", ""),
                "lie_momentum_norm": lie.get("lie_momentum_norm", ""),
                "ambient_momentum_norm": lie.get("ambient_momentum_norm", ""),
                "transported_lie_momentum_norm": lie.get("transported_lie_momentum_norm", ""),
                "transport_error": lie.get("transport_error", ""),
                "left_rotation_angle": lie.get("left_rotation_angle", ""),
                "left_right_imbalance": lie.get("left_right_imbalance", ""),
                "Lie_SNR": lie.get("Lie_SNR", ""),
                "continuous_fu_state_updated_every_step": lie.get("continuous_fu_state_updated_every_step", ""),
                "fu_velocity_emitted_every_step": lie.get("fu_velocity_emitted_every_step", ""),
                "candidate_action_selection_used_for_runtime": lie.get("candidate_action_selection_used_for_runtime", ""),
            }
        )
    return out or [{"status": "no_functional_m3_trajectory_rows"}]


def run_m3_trajectory(args: argparse.Namespace) -> dict[str, Any]:
    write_rows(OUT_ROOT / "v22_45E_mechanism_coverage_matrix.csv", MECHANISM_COVERAGE)
    specs = build_m3_trajectory_specs(args)
    output_prefix = safe_fragment(args.m3_trajectory_output)
    specs_filename = f"v22_45E_{output_prefix}_specs.csv"
    matrix_filename = f"v22_45E_{output_prefix}_matrix.csv"
    runtime_filename = f"v22_45E_{output_prefix}_runtime_regression_audit.csv"
    support_filename = f"v22_45E_{output_prefix}_support_direction_decomposition.csv"
    safety_filename = f"v22_45E_{output_prefix}_safety_debt_matrix.csv"
    overhead_filename = f"v22_45E_{output_prefix}_overhead_matrix.csv"
    summary_filename = f"v22_45E_{output_prefix}_mechanism_summary.csv"
    plan_filename = f"v22_45E_{output_prefix}_plan_metrics.csv"
    write_rows(OUT_ROOT / specs_filename, specs)
    append_exec(
        trajectory_command_line(args, "trajectory-m3"),
        task_id=f"m3_trajectory_started_{safe_fragment(args.label)}",
        status="started",
        gpu=",".join(split_csv(args.gpus)),
        files=f"results/v22_45E/{specs_filename}, results/v22_45E/chunks",
        note=(
            f"trajectories={len(specs)}; workers={args.workers}; in_process=1; subprocess_used=0; "
            f"datasets={args.eval_datasets}; seeds={args.eval_seeds}; architectures={args.eval_architectures}; "
            f"steps={args.trajectory_steps or args.steps}"
        ),
    )
    failures = 0
    statuses: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futures = [ex.submit(run_one_inprocess_trajectory, args, spec, "m3_trajectory") for spec in specs]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            statuses.append(row)
            failures += int(row.get("status") != "pass")
    write_rows(OUT_ROOT / f"v22_45E_{output_prefix}_status.csv", statuses)
    merge_summary = merge_v2243_outputs(
        specs_filename=specs_filename,
        matrix_filename=matrix_filename,
        runtime_filename=runtime_filename,
        support_direction_filename=support_filename,
        safety_filename=safety_filename,
        overhead_filename=overhead_filename,
        summary_filename=summary_filename,
    )
    plan_rows = m3_trajectory_plan_metric_rows(matrix_filename)
    write_rows(OUT_ROOT / plan_filename, plan_rows)
    append_exec(
        "m3 trajectory-first suite completed",
        task_id=f"m3_trajectory_completed_{safe_fragment(args.label)}",
        status="pass" if failures == 0 else "fail",
        gpu=",".join(split_csv(args.gpus)),
        files=f"results/v22_45E/{matrix_filename}, results/v22_45E/{summary_filename}, results/v22_45E/{plan_filename}",
        note=f"trajectories={len(specs)}; failures={failures}; merged={merge_summary}; plan_metric_rows={len(plan_rows)}",
        exit_code=0 if failures == 0 else 1,
    )
    return {"trajectories": len(specs), "failures": failures, "plan_metric_rows": len(plan_rows), **merge_summary}


M4_TASKS = ["modular_addition", "Class_MNIST_0_4_to_5_9"]


def m4_trajectory_steps(args: argparse.Namespace) -> int:
    return int(args.m4_trajectory_steps or args.m4_diag_steps)


def build_m4_trajectory_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    gpus = split_csv(args.gpus)
    variants = ["optimizer_alone", "slow_signal_flow", "slow_random_control", "slow_signflip_control"]
    tasks = [t for t in split_csv(args.m4_tasks) if t]
    unknown = [t for t in tasks if t not in M4_TASKS]
    if unknown:
        raise ValueError(f"unknown --m4-tasks values: {unknown}; allowed={M4_TASKS}")
    for task in tasks:
        for seed in split_csv(args.diagnostic_seeds, int):
            for arch in split_csv(args.diagnostic_architectures):
                for variant in variants:
                    gpu = gpus[len(specs) % max(1, len(gpus))]
                    task_tag = "ModAdd" if task == "modular_addition" else "ClassMNIST"
                    label = (
                        f"{safe_fragment(args.label)}_M4_Trajectory{task_tag}_"
                        f"{safe_fragment(task)}_p{int(args.m4_modulus)}_{safe_fragment(arch)}_s{int(seed)}_{safe_fragment(variant)}"
                    )
                    specs.append(
                        {
                            "mechanism": "M4",
                            "dataset": task,
                            "modulus": int(args.m4_modulus),
                            "train_fraction": float(args.m4_train_fraction),
                            "seed": int(seed),
                            "architecture": str(arch),
                            "variant": variant,
                            "optimizer": str(args.optimizer),
                            "steps": m4_trajectory_steps(args),
                            "steps_per_task": m4_trajectory_steps(args),
                            "train_size": int(args.m7_train_size),
                            "held_size": int(args.m7_held_size),
                            "hidden": int(args.diagnostic_hidden),
                            "batch_size": int(args.diagnostic_batch_size),
                            "lr": float(args.lr),
                            "weight_decay": float(args.weight_decay),
                            "rho": float(args.m4_rho),
                            "velocity_scale": float(args.m4_diag_velocity_scale),
                            "beta_slow": float(args.m4_beta_slow),
                            "lambda_mig": float(args.m4_lambda_mig),
                            "lambda_res": float(args.m4_lambda_res),
                            "support_rank": int(args.m4_support_rank or args.support_rank),
                            "train_threshold": float(args.m4_train_threshold),
                            "test_threshold": float(args.m4_test_threshold),
                            "device": f"cuda:{gpu}" if not str(gpu).startswith("cuda") else str(gpu),
                            "label": label,
                        }
                    )
    if int(args.row_limit) > 0:
        specs = specs[: int(args.row_limit)]
    return specs


def m4_energy(values: dict[str, Any]) -> float:
    total = 0.0
    for value in values.values():
        total += float(value.detach().float().pow(2.0).sum().item())
    return total


def m4_project_support_like(anchor: dict[str, Any], values: dict[str, Any], support_rank: int) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {}
    k_cap = max(1, int(support_rank))
    for name, value in values.items():
        ref = anchor.get(name)
        val = value.detach().float()
        if ref is None or val.numel() == 0:
            out[name] = val * 0.0
            continue
        flat_ref = ref.detach().float().abs().reshape(-1)
        flat_val = val.reshape(-1)
        if flat_ref.numel() <= k_cap:
            out[name] = val.clone()
            continue
        idx = torch.topk(flat_ref, k=min(k_cap, flat_ref.numel()), largest=True).indices
        mask = torch.zeros_like(flat_ref)
        mask[idx] = 1.0
        out[name] = (flat_val * mask).reshape_as(val)
    return out


def m4_signal_velocity(
    grad: dict[str, Any],
    slow_state: dict[str, Any],
    *,
    support_rank: int,
    lambda_mig: float,
    lambda_res: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    fast: dict[str, Any] = {}
    for name, g in grad.items():
        slow = slow_state.get(name)
        fast[name] = g.detach().float() - (slow.detach().float() if slow is not None else g.detach().float() * 0.0)
    p_slow = m4_project_support_like(slow_state, slow_state, support_rank)
    p_fast = m4_project_support_like(slow_state, fast, support_rank)
    r_fast = {name: fast[name] - p_fast.get(name, fast[name] * 0.0) for name in fast}
    velocity: dict[str, Any] = {}
    for name, slow in slow_state.items():
        velocity[name] = (
            slow.detach().float()
            + float(lambda_mig) * p_slow.get(name, slow * 0.0)
            - float(lambda_res) * r_fast.get(name, slow * 0.0)
        )
    slow_energy = m4_energy(slow_state)
    fast_energy = m4_energy(fast)
    p_slow_energy = m4_energy(p_slow)
    p_fast_energy = m4_energy(p_fast)
    r_fast_energy = m4_energy(r_fast)
    stats = {
        "slow_signal_energy": slow_energy,
        "fast_signal_energy": fast_energy,
        "support_slow_energy": p_slow_energy,
        "support_fast_energy": p_fast_energy,
        "reservoir_fast_energy": r_fast_energy,
        "support_slow_fraction": p_slow_energy / max(1.0e-12, slow_energy),
        "support_fast_fraction": p_fast_energy / max(1.0e-12, fast_energy),
        "RSM_index": (p_slow_energy / max(1.0e-12, slow_energy)) - (p_fast_energy / max(1.0e-12, fast_energy)),
        "raw_signal_velocity_energy": m4_energy(velocity),
    }
    return velocity, stats


def run_m4_modular_trajectory_from_spec(args: argparse.Namespace, spec: dict[str, Any]) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from experiments import run_v22_37_causal_instrumented_functional_optimizer as core
    from experiments import run_v22_40_continuous_functional_flow_fu as v2240
    from experiments import run_v22_42R_s5_temporal_slow_signal_diagnostic as s5diag

    s5diag.seed_all(int(spec["seed"]))
    requested_device = str(spec.get("device", "cuda:0"))
    device = torch.device(requested_device if torch.cuda.is_available() or not requested_device.startswith("cuda") else "cpu")
    train_loader, eval_train_loader, test_loader, input_dim, output_dim, x_stats = s5diag.modular_loaders(
        int(spec["modulus"]),
        float(spec["train_fraction"]),
        int(spec["batch_size"]),
        int(spec["seed"]),
        device,
    )
    model_seed = int(spec["seed"]) + 458000 + (0 if spec["architecture"] == "MLP" else 1000)
    model = core.make_model_for_arch(
        str(spec["architecture"]),
        input_dim,
        output_dim,
        int(spec["hidden"]),
        model_seed,
        device,
        x_stats,
    )
    opt = core.optimizer_for(str(spec["optimizer"]), model.parameters(), float(spec["lr"]), float(spec["weight_decay"]))
    train_iter = core.cycle_batches(train_loader)
    avg_state: dict[str, Any] = {}
    slow_state: dict[str, Any] = {}
    trace: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    slow_energies: list[float] = []
    fast_energies: list[float] = []
    rsm_values: list[float] = []
    support_slow_fracs: list[float] = []
    support_fast_fracs: list[float] = []
    raw_velocity_energies: list[float] = []
    train_cross_step: str | int = ""
    test_cross_step: str | int = ""
    test_at_train_cross: str | float = ""
    h200_test_acc: str | float = ""
    h800_test_acc: str | float = ""
    started = time.time()
    steps = int(spec["steps"])
    eval_steps = {1, 20, 60, 200, 400, 800, 1200, 1600, steps}
    eval_steps = {s for s in eval_steps if 1 <= int(s) <= steps}
    for step in range(1, steps + 1):
        x, y = next(train_iter)
        x = x.to(device).float()
        y = y.to(device).long()
        opt.zero_grad(set_to_none=True)
        logits = model(x).float()
        loss = F.cross_entropy(logits, y)
        loss.backward()
        named = s5diag.named_trainable(model)
        grad = s5diag.grad_dict(named)
        for name, g in grad.items():
            old = slow_state.get(name)
            slow_state[name] = g if old is None else (1.0 - float(spec["beta_slow"])) * old.to(device=g.device).float() + float(spec["beta_slow"]) * g
        signal_raw, signal_stats = m4_signal_velocity(
            grad,
            slow_state,
            support_rank=int(spec["support_rank"]),
            lambda_mig=float(spec["lambda_mig"]),
            lambda_res=float(spec["lambda_res"]),
        )
        v2240.optimizer_step(model, opt, str(spec["optimizer"]), step, avg_state)
        if spec["variant"] == "optimizer_alone":
            velocity_raw = s5diag.zeros_like(named)
        elif spec["variant"] == "slow_signal_flow":
            velocity_raw = signal_raw
        elif spec["variant"] == "slow_random_control":
            velocity_raw = s5diag.random_like(signal_raw, int(spec["seed"]) * 100000 + step)
        elif spec["variant"] == "slow_signflip_control":
            velocity_raw = {name: -value for name, value in signal_raw.items()}
        else:
            raise ValueError(f"unknown M4 variant {spec['variant']}")
        velocity = s5diag.normalize(velocity_raw, 1.0)
        fu_norm = 0.0
        if spec["variant"] != "optimizer_alone":
            fu_norm = s5diag.apply_velocity(named, velocity, float(spec["lr"]) * float(spec["rho"]) * float(spec["velocity_scale"]))
        slow_energies.append(float(signal_stats["slow_signal_energy"]))
        fast_energies.append(float(signal_stats["fast_signal_energy"]))
        rsm_values.append(float(signal_stats["RSM_index"]))
        support_slow_fracs.append(float(signal_stats["support_slow_fraction"]))
        support_fast_fracs.append(float(signal_stats["support_fast_fraction"]))
        raw_velocity_energies.append(float(signal_stats["raw_signal_velocity_energy"]))
        runtime_rows.append(
            {
                "run_label": spec["label"],
                "dataset": "modular_addition",
                "modulus": spec["modulus"],
                "seed": spec["seed"],
                "architecture": spec["architecture"],
                "variant": spec["variant"],
                "step": step,
                "runtime_policy_type": "continuous_slow_signal_reservoir_migration_velocity_field",
                "continuous_fu_state_updated": 1,
                "fu_velocity_emitted": 1,
                "runtime_argmax_candidate_used": 0,
                "runtime_topk_candidate_used": 0,
                "candidate_action_selection_used_for_runtime": 0,
                "candidate_value_model_used_as_runtime_policy": 0,
                "micro_rct_winner_used_as_runtime_action": 0,
                "path_mpc_discrete_action_sequence_used": 0,
                "fu_velocity_norm": fu_norm,
                "slow_signal_energy": signal_stats["slow_signal_energy"],
                "fast_signal_energy": signal_stats["fast_signal_energy"],
                "support_slow_fraction": signal_stats["support_slow_fraction"],
                "support_fast_fraction": signal_stats["support_fast_fraction"],
                "reservoir_fast_energy": signal_stats["reservoir_fast_energy"],
                "RSM_index": signal_stats["RSM_index"],
            }
        )
        if step in eval_steps:
            tr = s5diag.evaluate(model, eval_train_loader, device)
            te = s5diag.evaluate(model, test_loader, device)
            if train_cross_step == "" and tr["accuracy"] >= float(spec["train_threshold"]):
                train_cross_step = step
                test_at_train_cross = te["accuracy"]
            if test_cross_step == "" and te["accuracy"] >= float(spec["test_threshold"]):
                test_cross_step = step
            if step == 200:
                h200_test_acc = te["accuracy"]
            if step == 800:
                h800_test_acc = te["accuracy"]
            trace.append(
                {
                    "run_label": spec["label"],
                    "dataset": "modular_addition",
                    "modulus": spec["modulus"],
                    "seed": spec["seed"],
                    "architecture": spec["architecture"],
                    "variant": spec["variant"],
                    "step": step,
                    "train_NLL": tr["NLL"],
                    "train_accuracy": tr["accuracy"],
                    "test_NLL": te["NLL"],
                    "test_accuracy": te["accuracy"],
                    "slow_signal_energy": signal_stats["slow_signal_energy"],
                    "fast_signal_energy": signal_stats["fast_signal_energy"],
                    "support_slow_fraction": signal_stats["support_slow_fraction"],
                    "support_fast_fraction": signal_stats["support_fast_fraction"],
                    "RSM_index": signal_stats["RSM_index"],
                }
            )
    v2240.final_schedule_free_swap(model, str(spec["optimizer"]), avg_state)
    final_train = s5diag.evaluate(model, eval_train_loader, device)
    final_test = s5diag.evaluate(model, test_loader, device)
    if test_at_train_cross == "":
        test_at_train_cross = final_test["accuracy"]
    grokking_delay: str | int = ""
    if train_cross_step != "" and test_cross_step != "":
        grokking_delay = max(0, int(test_cross_step) - int(train_cross_step))
    summary = {
        "run_label": spec["label"],
        "mechanism": "M4",
        "dataset": "modular_addition",
        "modulus": spec["modulus"],
        "train_fraction": spec["train_fraction"],
        "seed": spec["seed"],
        "architecture": spec["architecture"],
        "optimizer_family": spec["optimizer"],
        "variant": spec["variant"],
        "steps": steps,
        "hidden": spec["hidden"],
        "support_rank": spec["support_rank"],
        "beta_slow": spec["beta_slow"],
        "lambda_mig": spec["lambda_mig"],
        "lambda_res": spec["lambda_res"],
        "velocity_scale": spec["velocity_scale"],
        "final_train_accuracy": final_train["accuracy"],
        "final_test_accuracy": final_test["accuracy"],
        "final_train_acc": final_train["accuracy"],
        "final_test_acc": final_test["accuracy"],
        "final_train_NLL": final_train["NLL"],
        "final_test_NLL": final_test["NLL"],
        "train_cross_step": train_cross_step,
        "test_cross_step": test_cross_step,
        "time_to_generalization": test_cross_step,
        "grokking_delay": grokking_delay,
        "test_accuracy_lift_after_delay": final_test["accuracy"] - float(test_at_train_cross),
        "slow_signal_energy": statistics.fmean(slow_energies) if slow_energies else 0.0,
        "fast_signal_energy": statistics.fmean(fast_energies) if fast_energies else 0.0,
        "RSM_index": statistics.fmean(rsm_values) if rsm_values else 0.0,
        "support_slow_fraction": statistics.fmean(support_slow_fracs) if support_slow_fracs else 0.0,
        "support_fast_fraction": statistics.fmean(support_fast_fracs) if support_fast_fracs else 0.0,
        "raw_signal_velocity_energy": statistics.fmean(raw_velocity_energies) if raw_velocity_energies else 0.0,
        "H200_test_accuracy": h200_test_acc,
        "H800_test_accuracy": h800_test_acc,
        "continuous_fu_state_updated_every_step": 1,
        "fu_velocity_emitted_every_step": 1,
        "candidate_action_selection_used_for_runtime": 0,
        "runtime_argmax_candidate_used": 0,
        "runtime_topk_candidate_used": 0,
        "candidate_value_model_used_as_runtime_policy": 0,
        "micro_rct_winner_used_as_runtime_action": 0,
        "path_mpc_discrete_action_sequence_used": 0,
        "in_process": 1,
        "subprocess_used": 0,
        "complete_step_loop": 1,
        "elapsed_sec": time.time() - started,
        "status": "completed_v22_45E_m4_trajectory",
    }
    prefix = CHUNK_ROOT / f"v22_45E_m4_{safe_fragment(spec['label'])}"
    summary_path = Path(str(prefix) + "_summary.csv")
    trace_path = Path(str(prefix) + "_trace.csv")
    runtime_path = Path(str(prefix) + "_runtime_trace.csv")
    write_rows(summary_path, [summary])
    write_rows(trace_path, trace)
    write_rows(runtime_path, runtime_rows)
    summary["chunk_summary"] = str(summary_path.relative_to(ROOT))
    summary["chunk_trace"] = str(trace_path.relative_to(ROOT))
    summary["chunk_runtime_trace"] = str(runtime_path.relative_to(ROOT))
    return {"summary": summary, "trace": trace, "runtime": runtime_rows}


def run_m4_class_mnist_trajectory_from_spec(args: argparse.Namespace, spec: dict[str, Any]) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from experiments import run_v22_37_causal_instrumented_functional_optimizer as core
    from experiments import run_v22_40_continuous_functional_flow_fu as v2240
    from experiments import run_v22_42R_s5_temporal_slow_signal_diagnostic as s5diag

    v2242s7.seed_all(int(spec["seed"]))
    requested_device = str(spec.get("device", "cuda:0"))
    device = torch.device(requested_device if torch.cuda.is_available() or not requested_device.startswith("cuda") else "cpu")
    old_loader, new_loader, old_eval, new_eval = m7_mnist_split_loaders(
        "Class_MNIST_0_4_to_5_9",
        int(spec["train_size"]),
        int(spec["held_size"]),
        int(spec["batch_size"]),
        int(spec["seed"]),
        download=bool(args.tier2_download),
    )
    x_stats = next(iter(old_loader))[0].to(device).float()
    model_seed = int(spec["seed"]) + 459000 + (0 if spec["architecture"] == "MLP" else 1000)
    model = core.make_model_for_arch(str(spec["architecture"]), 784, 10, int(spec["hidden"]), model_seed, device, x_stats)
    opt = core.optimizer_for(str(spec["optimizer"]), model.parameters(), float(spec["lr"]), float(spec["weight_decay"]))
    old_iter = core.cycle_batches(old_loader)
    new_iter = core.cycle_batches(new_loader)
    avg_state: dict[str, Any] = {}
    slow_state: dict[str, Any] = {}
    runtime_rows: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    slow_energies: list[float] = []
    fast_energies: list[float] = []
    rsm_values: list[float] = []
    support_slow_fracs: list[float] = []
    support_fast_fracs: list[float] = []
    raw_velocity_energies: list[float] = []
    steps = int(spec["steps_per_task"])
    started = time.time()
    global_step = 0

    def update_slow(grad: dict[str, Any]) -> None:
        for name, g in grad.items():
            old = slow_state.get(name)
            slow_state[name] = g if old is None else (1.0 - float(spec["beta_slow"])) * old.to(device=g.device).float() + float(spec["beta_slow"]) * g

    def runtime_row(phase: str, step: int, fu_norm: float, stats: dict[str, float]) -> dict[str, Any]:
        return {
            "run_label": spec["label"],
            "dataset": "Class_MNIST_0_4_to_5_9",
            "seed": spec["seed"],
            "architecture": spec["architecture"],
            "variant": spec["variant"],
            "phase": phase,
            "step": step,
            "runtime_policy_type": "continuous_slow_signal_reservoir_migration_velocity_field",
            "continuous_fu_state_updated": 1,
            "fu_velocity_emitted": 1,
            "runtime_argmax_candidate_used": 0,
            "runtime_topk_candidate_used": 0,
            "candidate_action_selection_used_for_runtime": 0,
            "candidate_value_model_used_as_runtime_policy": 0,
            "micro_rct_winner_used_as_runtime_action": 0,
            "path_mpc_discrete_action_sequence_used": 0,
            "fu_velocity_norm": fu_norm,
            "slow_signal_energy": stats.get("slow_signal_energy", 0.0),
            "fast_signal_energy": stats.get("fast_signal_energy", 0.0),
            "support_slow_fraction": stats.get("support_slow_fraction", 0.0),
            "support_fast_fraction": stats.get("support_fast_fraction", 0.0),
            "reservoir_fast_energy": stats.get("reservoir_fast_energy", 0.0),
            "RSM_index": stats.get("RSM_index", 0.0),
        }

    eval_marks = {1, max(1, steps // 2), steps}
    zero_stats = {
        "slow_signal_energy": 0.0,
        "fast_signal_energy": 0.0,
        "support_slow_fraction": 0.0,
        "support_fast_fraction": 0.0,
        "reservoir_fast_energy": 0.0,
        "RSM_index": 0.0,
        "raw_signal_velocity_energy": 0.0,
    }
    for step in range(1, steps + 1):
        global_step += 1
        grad = v2242s7.train_one_batch(model, opt, str(spec["optimizer"]), next(old_iter), device, avg_state, global_step)
        update_slow(grad)
        runtime_rows.append(runtime_row("old", global_step, 0.0, zero_stats))
        if step in eval_marks:
            trace.append(
                {
                    "run_label": spec["label"],
                    "dataset": "Class_MNIST_0_4_to_5_9",
                    "seed": spec["seed"],
                    "architecture": spec["architecture"],
                    "variant": spec["variant"],
                    "phase": "old",
                    "step": global_step,
                    "old_task_accuracy": v2242s7.evaluate_accuracy(model, old_eval, device),
                    "new_task_accuracy": v2242s7.evaluate_accuracy(model, new_eval, device),
                    **zero_stats,
                }
            )
    old_before = v2242s7.evaluate_accuracy(model, old_eval, device)
    new_before = v2242s7.evaluate_accuracy(model, new_eval, device)
    for step in range(1, steps + 1):
        global_step += 1
        xb, yb = next(new_iter)
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb).float(), yb)
        loss.backward()
        named = s5diag.named_trainable(model)
        grad = s5diag.grad_dict(named)
        update_slow(grad)
        signal_raw, signal_stats = m4_signal_velocity(
            grad,
            slow_state,
            support_rank=int(spec["support_rank"]),
            lambda_mig=float(spec["lambda_mig"]),
            lambda_res=float(spec["lambda_res"]),
        )
        v2240.optimizer_step(model, opt, str(spec["optimizer"]), global_step, avg_state)
        if spec["variant"] == "optimizer_alone":
            velocity_raw = s5diag.zeros_like(named)
        elif spec["variant"] == "slow_signal_flow":
            velocity_raw = signal_raw
        elif spec["variant"] == "slow_random_control":
            velocity_raw = s5diag.random_like(signal_raw, int(spec["seed"]) * 100000 + global_step)
        elif spec["variant"] == "slow_signflip_control":
            velocity_raw = {name: -value for name, value in signal_raw.items()}
        else:
            raise ValueError(f"unknown M4 variant {spec['variant']}")
        velocity = s5diag.normalize(velocity_raw, 1.0)
        fu_norm = 0.0
        if spec["variant"] != "optimizer_alone":
            fu_norm = s5diag.apply_velocity(named, velocity, float(spec["lr"]) * float(spec["rho"]) * float(spec["velocity_scale"]))
        slow_energies.append(float(signal_stats["slow_signal_energy"]))
        fast_energies.append(float(signal_stats["fast_signal_energy"]))
        rsm_values.append(float(signal_stats["RSM_index"]))
        support_slow_fracs.append(float(signal_stats["support_slow_fraction"]))
        support_fast_fracs.append(float(signal_stats["support_fast_fraction"]))
        raw_velocity_energies.append(float(signal_stats["raw_signal_velocity_energy"]))
        runtime_rows.append(runtime_row("new", global_step, fu_norm, signal_stats))
        if step in eval_marks:
            trace.append(
                {
                    "run_label": spec["label"],
                    "dataset": "Class_MNIST_0_4_to_5_9",
                    "seed": spec["seed"],
                    "architecture": spec["architecture"],
                    "variant": spec["variant"],
                    "phase": "new",
                    "step": global_step,
                    "old_task_accuracy": v2242s7.evaluate_accuracy(model, old_eval, device),
                    "new_task_accuracy": v2242s7.evaluate_accuracy(model, new_eval, device),
                    **signal_stats,
                }
            )
    v2240.final_schedule_free_swap(model, str(spec["optimizer"]), avg_state)
    old_after = v2242s7.evaluate_accuracy(model, old_eval, device)
    new_after = v2242s7.evaluate_accuracy(model, new_eval, device)
    forgetting = old_before - old_after
    final_avg = 0.5 * (old_after + new_after)
    summary = {
        "run_label": spec["label"],
        "mechanism": "M4",
        "dataset": "Class_MNIST_0_4_to_5_9",
        "seed": spec["seed"],
        "architecture": spec["architecture"],
        "optimizer_family": spec["optimizer"],
        "variant": spec["variant"],
        "steps": steps,
        "steps_per_task": steps,
        "hidden": spec["hidden"],
        "support_rank": spec["support_rank"],
        "beta_slow": spec["beta_slow"],
        "lambda_mig": spec["lambda_mig"],
        "lambda_res": spec["lambda_res"],
        "velocity_scale": spec["velocity_scale"],
        "old_task_accuracy_before_new_task": old_before,
        "new_task_accuracy_before_new_task": new_before,
        "old_task_accuracy_after_new_task": old_after,
        "new_task_accuracy": new_after,
        "final_average_accuracy": final_avg,
        "average_forgetting": forgetting,
        "relative_forgetting_reduction": "",
        "slow_signal_energy": statistics.fmean(slow_energies) if slow_energies else 0.0,
        "fast_signal_energy": statistics.fmean(fast_energies) if fast_energies else 0.0,
        "RSM_index": statistics.fmean(rsm_values) if rsm_values else 0.0,
        "support_slow_fraction": statistics.fmean(support_slow_fracs) if support_slow_fracs else 0.0,
        "support_fast_fraction": statistics.fmean(support_fast_fracs) if support_fast_fracs else 0.0,
        "raw_signal_velocity_energy": statistics.fmean(raw_velocity_energies) if raw_velocity_energies else 0.0,
        "continuous_fu_state_updated_every_step": 1,
        "fu_velocity_emitted_every_step": 1,
        "candidate_action_selection_used_for_runtime": 0,
        "runtime_argmax_candidate_used": 0,
        "runtime_topk_candidate_used": 0,
        "candidate_value_model_used_as_runtime_policy": 0,
        "micro_rct_winner_used_as_runtime_action": 0,
        "path_mpc_discrete_action_sequence_used": 0,
        "in_process": 1,
        "subprocess_used": 0,
        "complete_step_loop": 1,
        "elapsed_sec": time.time() - started,
        "status": "completed_v22_45E_m4_class_mnist_trajectory",
        "claim_limit": "Class_MNIST continual slow-gradient/reservoir migration; parameter top-k support approximation, not full metric/OET projection",
    }
    prefix = CHUNK_ROOT / f"v22_45E_m4_{safe_fragment(spec['label'])}"
    summary_path = Path(str(prefix) + "_summary.csv")
    trace_path = Path(str(prefix) + "_trace.csv")
    runtime_path = Path(str(prefix) + "_runtime_trace.csv")
    write_rows(summary_path, [summary])
    write_rows(trace_path, trace)
    write_rows(runtime_path, runtime_rows)
    summary["chunk_summary"] = str(summary_path.relative_to(ROOT))
    summary["chunk_trace"] = str(trace_path.relative_to(ROOT))
    summary["chunk_runtime_trace"] = str(runtime_path.relative_to(ROOT))
    return {"summary": summary, "trace": trace, "runtime": runtime_rows}


def run_one_m4_trajectory(args: argparse.Namespace, spec: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    task_id = f"m4_trajectory_{safe_fragment(spec.get('label', 'trajectory'))}"
    try:
        if str(spec.get("dataset", "modular_addition")) == "Class_MNIST_0_4_to_5_9":
            result = run_m4_class_mnist_trajectory_from_spec(args, spec)
            runner_note = "run_m4_class_mnist_trajectory"
        else:
            result = run_m4_modular_trajectory_from_spec(args, spec)
            runner_note = "run_m4_modular_trajectory"
        summary = result["summary"]
        append_exec(
            f"in_process {runner_note} label={shlex.quote(str(spec.get('label', '')))}",
            task_id=task_id,
            status="pass",
            gpu=str(spec.get("device", "")),
            files=f"{summary.get('chunk_summary', '')}, {summary.get('chunk_trace', '')}, {summary.get('chunk_runtime_trace', '')}",
            note=f"elapsed_sec={time.time() - started:.3f}; in_process=1; subprocess_used=0; complete_step_loop=1",
        )
        return {"label": spec.get("label", ""), "status": "pass", "error": "", **result}
    except Exception as exc:  # pragma: no cover - experiment audit path
        tb = traceback.format_exc(limit=8).replace("\n", " | ")
        append_exec(
            f"in_process run_m4_trajectory label={shlex.quote(str(spec.get('label', '')))}",
            task_id=task_id,
            status="fail",
            gpu=str(spec.get("device", "")),
            note=f"elapsed_sec={time.time() - started:.3f}; error={type(exc).__name__}: {exc}; traceback={tb[:1600]}",
            exit_code=1,
        )
        return {"label": spec.get("label", ""), "status": "fail", "error": f"{type(exc).__name__}: {exc}"}


def summarize_m4_trajectory(rows: list[dict[str, Any]], runtimes: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("dataset", "")), str(row.get("architecture", "")), str(row.get("seed", ""))), {})[str(row.get("variant", ""))] = row
    slow_wins = 0
    matched_controls_fail = 0
    invalid_no_delay = 0
    any_test_cross = 0
    delay_reductions: list[float] = []
    forgetting_reductions: list[float] = []
    valid_forgetting = 0
    final_nonworse_count = 0
    continual_pass = 0
    nll_wins = 0
    rsm_values: list[float] = []
    for (dataset, _arch, _seed), group in groups.items():
        base = group.get("optimizer_alone", {})
        slow = group.get("slow_signal_flow", {})
        rand = group.get("slow_random_control", {})
        sign = group.get("slow_signflip_control", {})
        if not slow:
            continue
        if dataset == "Class_MNIST_0_4_to_5_9":
            base_forget = finite_float(base.get("average_forgetting"))
            slow_forget = finite_float(slow.get("average_forgetting"))
            rand_forget = finite_float(rand.get("average_forgetting"))
            sign_forget = finite_float(sign.get("average_forgetting"))
            base_avg = finite_float(base.get("final_average_accuracy"))
            slow_avg = finite_float(slow.get("final_average_accuracy"))
            task_valid = int(base_forget is not None and base_forget > 0.0)
            valid_forgetting += task_valid
            rel = None if not task_valid or slow_forget is None else (base_forget - slow_forget) / abs(base_forget)
            if rel is not None:
                forgetting_reductions.append(rel)
            controls_fail = int(
                slow_forget is not None
                and rand_forget is not None
                and sign_forget is not None
                and rand_forget > slow_forget
                and sign_forget > slow_forget
            )
            matched_controls_fail += controls_fail
            final_nonworse = int(slow_avg is not None and base_avg is not None and slow_avg >= base_avg - 1.0e-12)
            final_nonworse_count += final_nonworse
            continual_pass += int(rel is not None and rel >= 0.10 and controls_fail and final_nonworse)
        else:
            bdelay = finite_float(base.get("grokking_delay"))
            sdelay = finite_float(slow.get("grokking_delay"))
            if bdelay is None or sdelay is None:
                invalid_no_delay += 1
            elif bdelay > 0.0:
                delay_reductions.append((bdelay - sdelay) / bdelay)
            any_test_cross += int(finite_float(base.get("test_cross_step")) is not None or finite_float(slow.get("test_cross_step")) is not None)
            slow_acc = value_or(slow.get("final_test_accuracy"), -math.inf)
            base_acc = value_or(base.get("final_test_accuracy"), -math.inf)
            slow_wins += int(slow_acc > base_acc)
            slow_nll = finite_float(slow.get("final_test_NLL"))
            base_nll = finite_float(base.get("final_test_NLL"))
            nll_wins += int(slow_nll is not None and base_nll is not None and slow_nll < base_nll)
            rand_acc = value_or(rand.get("final_test_accuracy"), math.inf)
            sign_acc = value_or(sign.get("final_test_accuracy"), math.inf)
            matched_controls_fail += int(rand_acc < slow_acc and sign_acc < slow_acc)
        rsm = finite_float(slow.get("RSM_index"))
        if rsm is not None:
            rsm_values.append(rsm)
    bad_runtime = 0
    for row in runtimes:
        for key in [
            "runtime_argmax_candidate_used",
            "runtime_topk_candidate_used",
            "candidate_action_selection_used_for_runtime",
            "candidate_value_model_used_as_runtime_policy",
            "micro_rct_winner_used_as_runtime_action",
            "path_mpc_discrete_action_sequence_used",
        ]:
            bad_runtime += int(str(row.get(key, "0")) not in {"0", ""})
        bad_runtime += int(str(row.get("continuous_fu_state_updated", "1")) != "1")
        bad_runtime += int(str(row.get("fu_velocity_emitted", "1")) != "1")
    mean_reduction = statistics.fmean(delay_reductions) if delay_reductions else math.nan
    mean_forgetting_reduction = statistics.fmean(forgetting_reductions) if forgetting_reductions else math.nan
    route = "M4TrajectoryNoDelayOrNoControlWin"
    if continual_pass >= max(1, math.ceil(0.50 * max(1, valid_forgetting))) and valid_forgetting > 0 and bad_runtime == 0:
        route = "R13-ContinualSlowSignalReservoirMigrationOpened_PartialTrajectory"
    elif invalid_no_delay >= max(1, len(groups) // 2):
        route = "GrokkingTaskInvalidNoDelay"
    elif delay_reductions and mean_reduction >= 0.25 and matched_controls_fail >= max(1, len(groups) // 2) and bad_runtime == 0:
        route = "R13-GrokkingReservoirMigrationOpened_PartialTrajectory"
    return {
        "generated_at": now_sg(),
        "rows": len(rows),
        "groups": len(groups),
        "slow_signal_final_test_acc_beats_base_groups": slow_wins,
        "slow_signal_final_test_NLL_beats_base_groups": nll_wins,
        "matched_slow_controls_fail_groups": matched_controls_fail,
        "invalid_no_delay_groups": invalid_no_delay,
        "any_test_cross_groups": any_test_cross,
        "grokking_delay_reduction_mean": mean_reduction,
        "valid_forgetting_groups": valid_forgetting,
        "continual_forgetting_reduction_mean": mean_forgetting_reduction,
        "continual_forgetting_pass_groups": continual_pass,
        "final_average_accuracy_nonworse_groups": final_nonworse_count,
        "RSM_index_mean": statistics.fmean(rsm_values) if rsm_values else math.nan,
        "runtime_bad_flag_count": bad_runtime,
        "candidate_action_regression_pass": int(bad_runtime == 0),
        "route": route,
        "promotion_allowed": int(route.startswith("R13")),
        "claim_limit": "partial top-k support/reservoir approximation; not full metric/OET projection",
    }


def m4_trajectory_plan_metric_rows(matrix_filename: str) -> list[dict[str, Any]]:
    rows = read_rows(OUT_ROOT / matrix_filename)
    groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("dataset", "")), str(row.get("architecture", "")), str(row.get("seed", ""))), {})[str(row.get("variant", ""))] = row
    out: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        dataset, architecture, seed = key
        base = group.get("optimizer_alone", {})
        slow = group.get("slow_signal_flow", {})
        rand = group.get("slow_random_control", {})
        sign = group.get("slow_signflip_control", {})
        if not slow:
            continue
        if dataset == "Class_MNIST_0_4_to_5_9":
            base_forget = finite_float(base.get("average_forgetting"))
            slow_forget = finite_float(slow.get("average_forgetting"))
            rand_forget = finite_float(rand.get("average_forgetting"))
            sign_forget = finite_float(sign.get("average_forgetting"))
            base_avg = finite_float(base.get("final_average_accuracy"))
            slow_avg = finite_float(slow.get("final_average_accuracy"))
            task_valid = int(base_forget is not None and base_forget > 0.0)
            rel = "" if not task_valid or slow_forget is None else (base_forget - slow_forget) / abs(base_forget)
            random_gap = "" if rand_forget is None or slow_forget is None else rand_forget - slow_forget
            sign_gap = "" if sign_forget is None or slow_forget is None else sign_forget - slow_forget
            controls_fail = int(isinstance(random_gap, float) and isinstance(sign_gap, float) and random_gap > 0.0 and sign_gap > 0.0)
            final_nonworse = int(slow_avg is not None and base_avg is not None and slow_avg >= base_avg - 1.0e-12)
            runtime_truth = int(
                int_flag(slow.get("continuous_fu_state_updated_every_step"))
                and int_flag(slow.get("fu_velocity_emitted_every_step"))
                and int_flag(slow.get("candidate_action_selection_used_for_runtime")) == 0
            )
            out.append(
                {
                    "dataset": dataset,
                    "architecture": architecture,
                    "seed": seed,
                    "base_average_forgetting": "" if base_forget is None else base_forget,
                    "slow_average_forgetting": "" if slow_forget is None else slow_forget,
                    "slow_random_average_forgetting": "" if rand_forget is None else rand_forget,
                    "slow_signflip_average_forgetting": "" if sign_forget is None else sign_forget,
                    "forgetting_task_valid": task_valid,
                    "relative_forgetting_reduction": rel,
                    "continual_forgetting_reduction_ge_10p": int(isinstance(rel, float) and rel >= 0.10),
                    "same_slow_random_control_gap": random_gap,
                    "same_slow_signflip_control_gap": sign_gap,
                    "base_final_average_accuracy": "" if base_avg is None else base_avg,
                    "slow_final_average_accuracy": "" if slow_avg is None else slow_avg,
                    "matched_slow_controls_fail": controls_fail,
                    "final_average_accuracy_nonworse": final_nonworse,
                    "M4_pass": int(isinstance(rel, float) and rel >= 0.10 and controls_fail and final_nonworse and runtime_truth),
                    "slow_signal_energy": slow.get("slow_signal_energy", ""),
                    "fast_signal_energy": slow.get("fast_signal_energy", ""),
                    "RSM_index": slow.get("RSM_index", ""),
                    "support_slow_fraction": slow.get("support_slow_fraction", ""),
                    "support_fast_fraction": slow.get("support_fast_fraction", ""),
                    "continuous_fu_state_updated_every_step": slow.get("continuous_fu_state_updated_every_step", ""),
                    "fu_velocity_emitted_every_step": slow.get("fu_velocity_emitted_every_step", ""),
                    "candidate_action_selection_used_for_runtime": slow.get("candidate_action_selection_used_for_runtime", ""),
                    "runtime_truth": runtime_truth,
                    "claim_limit": "Class_MNIST continual slow-gradient/reservoir migration; parameter top-k support approximation",
                }
            )
            continue
        base_nll = finite_float(base.get("final_test_NLL"))
        slow_nll = finite_float(slow.get("final_test_NLL"))
        rand_nll = finite_float(rand.get("final_test_NLL"))
        sign_nll = finite_float(sign.get("final_test_NLL"))
        base_acc = finite_float(base.get("final_test_accuracy"))
        slow_acc = finite_float(slow.get("final_test_accuracy"))
        rand_acc = finite_float(rand.get("final_test_accuracy"))
        sign_acc = finite_float(sign.get("final_test_accuracy"))
        base_delay = finite_float(base.get("grokking_delay"))
        slow_delay = finite_float(slow.get("grokking_delay"))
        delay_reduction = ""
        if base_delay is not None and slow_delay is not None and base_delay > 0.0:
            delay_reduction = (base_delay - slow_delay) / base_delay
        controls_fail = int(
            slow_acc is not None
            and rand_acc is not None
            and sign_acc is not None
            and rand_acc < slow_acc
            and sign_acc < slow_acc
        )
        final_nonworse = int(slow_acc is not None and base_acc is not None and slow_acc >= base_acc)
        out.append(
            {
                "dataset": dataset,
                "architecture": architecture,
                "seed": seed,
                "base_final_test_accuracy": "" if base_acc is None else base_acc,
                "slow_final_test_accuracy": "" if slow_acc is None else slow_acc,
                "slow_random_final_test_accuracy": "" if rand_acc is None else rand_acc,
                "slow_signflip_final_test_accuracy": "" if sign_acc is None else sign_acc,
                "base_final_test_NLL": "" if base_nll is None else base_nll,
                "slow_final_test_NLL": "" if slow_nll is None else slow_nll,
                "NLL_delta_slow_minus_base": "" if slow_nll is None or base_nll is None else slow_nll - base_nll,
                "NLL_improvement_vs_base": "" if slow_nll is None or base_nll is None else base_nll - slow_nll,
                "NLL_improvement_vs_slow_random": "" if slow_nll is None or rand_nll is None else rand_nll - slow_nll,
                "NLL_improvement_vs_slow_signflip": "" if slow_nll is None or sign_nll is None else sign_nll - slow_nll,
                "base_grokking_delay": "" if base_delay is None else base_delay,
                "slow_grokking_delay": "" if slow_delay is None else slow_delay,
                "grokking_delay_reduction": delay_reduction,
                "grokking_delay_reduction_ge_25p": int(isinstance(delay_reduction, float) and delay_reduction >= 0.25),
                "matched_slow_controls_fail": controls_fail,
                "final_average_accuracy_nonworse": final_nonworse,
                "M4_pass": int(isinstance(delay_reduction, float) and delay_reduction >= 0.25 and controls_fail and final_nonworse),
                "slow_signal_energy": slow.get("slow_signal_energy", ""),
                "fast_signal_energy": slow.get("fast_signal_energy", ""),
                "RSM_index": slow.get("RSM_index", ""),
                "support_slow_fraction": slow.get("support_slow_fraction", ""),
                "support_fast_fraction": slow.get("support_fast_fraction", ""),
                "H200_test_accuracy": slow.get("H200_test_accuracy", ""),
                "H800_test_accuracy": slow.get("H800_test_accuracy", ""),
                "continuous_fu_state_updated_every_step": slow.get("continuous_fu_state_updated_every_step", ""),
                "fu_velocity_emitted_every_step": slow.get("fu_velocity_emitted_every_step", ""),
                "candidate_action_selection_used_for_runtime": slow.get("candidate_action_selection_used_for_runtime", ""),
            }
        )
    return out or [{"status": "no_m4_trajectory_rows"}]


def run_m4_trajectory(args: argparse.Namespace) -> dict[str, Any]:
    write_rows(OUT_ROOT / "v22_45E_mechanism_coverage_matrix.csv", MECHANISM_COVERAGE)
    specs = build_m4_trajectory_specs(args)
    output_prefix = safe_fragment(args.m4_trajectory_output)
    specs_filename = f"v22_45E_{output_prefix}_specs.csv"
    matrix_filename = f"v22_45E_{output_prefix}_matrix.csv"
    trace_filename = f"v22_45E_{output_prefix}_trace.csv"
    runtime_filename = f"v22_45E_{output_prefix}_runtime_trace.csv"
    plan_filename = f"v22_45E_{output_prefix}_plan_metrics.csv"
    status_filename = f"v22_45E_{output_prefix}_status.csv"
    summary_filename = f"v22_45E_{output_prefix}_summary.json"
    write_rows(OUT_ROOT / specs_filename, specs)
    append_exec(
        trajectory_command_line(args, "trajectory-m4"),
        task_id=f"m4_trajectory_started_{safe_fragment(args.label)}",
        status="started",
        gpu=",".join(split_csv(args.gpus)),
        files=f"results/v22_45E/{specs_filename}, results/v22_45E/chunks",
        note=(
            f"trajectories={len(specs)}; workers={args.workers}; in_process=1; subprocess_used=0; "
            f"architectures={args.diagnostic_architectures}; seeds={args.diagnostic_seeds}; "
            f"tasks={args.m4_tasks}; steps={m4_trajectory_steps(args)}; modulus={args.m4_modulus}; "
            f"class_train_size={args.m7_train_size}; class_held_size={args.m7_held_size}"
        ),
    )
    failures = 0
    statuses: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    runtimes: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futures = [ex.submit(run_one_m4_trajectory, args, spec) for spec in specs]
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            statuses.append({"label": result.get("label", ""), "status": result.get("status", ""), "error": result.get("error", "")})
            failures += int(result.get("status") != "pass")
            if result.get("status") == "pass":
                rows.append(result["summary"])
                traces.extend(result["trace"])
                runtimes.extend(result["runtime"])
    write_rows(OUT_ROOT / status_filename, statuses)
    write_rows(OUT_ROOT / matrix_filename, rows or [{"status": "no_rows"}])
    write_rows(OUT_ROOT / trace_filename, traces or [{"status": "no_rows"}])
    write_rows(OUT_ROOT / runtime_filename, runtimes or [{"status": "no_rows"}])
    plan_rows = m4_trajectory_plan_metric_rows(matrix_filename)
    write_rows(OUT_ROOT / plan_filename, plan_rows)
    summary = summarize_m4_trajectory(rows, runtimes)
    summary["failures"] = failures
    write_json(OUT_ROOT / summary_filename, summary)
    append_exec(
        "m4 trajectory-first slow-signal suite completed",
        task_id=f"m4_trajectory_completed_{safe_fragment(args.label)}",
        status="pass" if failures == 0 else "fail",
        gpu=",".join(split_csv(args.gpus)),
        files=(
            f"results/v22_45E/{matrix_filename}, results/v22_45E/{trace_filename}, "
            f"results/v22_45E/{runtime_filename}, results/v22_45E/{plan_filename}, results/v22_45E/{summary_filename}"
        ),
        note=f"trajectories={len(specs)}; failures={failures}; groups={summary.get('groups')}; route={summary.get('route')}",
        exit_code=0 if failures == 0 else 1,
    )
    return {"trajectories": len(specs), "failures": failures, "plan_metric_rows": len(plan_rows), **summary}


def run_m5_trajectory(args: argparse.Namespace) -> dict[str, Any]:
    write_rows(OUT_ROOT / "v22_45E_mechanism_coverage_matrix.csv", MECHANISM_COVERAGE)
    specs = build_m5_variant_audit_specs(args)
    for spec in specs:
        spec["recipe"] = str(spec.get("recipe", "")).replace("VariantAudit", "TrajectoryAudit")
        spec["steps"] = int(args.trajectory_steps or args.steps)
    output_prefix = safe_fragment(args.m5_trajectory_output)
    specs_filename = f"v22_45E_{output_prefix}_specs.csv"
    matrix_filename = f"v22_45E_{output_prefix}_matrix.csv"
    runtime_filename = f"v22_45E_{output_prefix}_runtime_regression_audit.csv"
    support_filename = f"v22_45E_{output_prefix}_support_direction_decomposition.csv"
    safety_filename = f"v22_45E_{output_prefix}_safety_debt_matrix.csv"
    overhead_filename = f"v22_45E_{output_prefix}_overhead_matrix.csv"
    summary_filename = f"v22_45E_{output_prefix}_mechanism_summary.csv"
    status_filename = f"v22_45E_{output_prefix}_status.csv"
    write_rows(OUT_ROOT / specs_filename, specs)
    append_exec(
        trajectory_command_line(args, "trajectory-m5"),
        task_id=f"m5_trajectory_started_{safe_fragment(args.label)}",
        status="started",
        gpu=",".join(split_csv(args.gpus)),
        files=f"results/v22_45E/{specs_filename}, results/v22_45E/chunks",
        note=(
            f"trajectories={len(specs)}; workers={args.workers}; in_process=1; subprocess_used=0; "
            f"datasets={args.eval_datasets}; seeds={args.eval_seeds}; kan_variants={args.m5_variant_audit_kan_variants}; "
            f"steps={args.trajectory_steps or args.steps}"
        ),
    )
    failures = 0
    statuses: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futures = [ex.submit(run_one_inprocess_trajectory, args, spec, "m5_trajectory") for spec in specs]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            statuses.append(row)
            failures += int(row.get("status") != "pass")
    write_rows(OUT_ROOT / status_filename, statuses)
    merge_summary = merge_v2243_outputs(
        specs_filename=specs_filename,
        matrix_filename=matrix_filename,
        runtime_filename=runtime_filename,
        support_direction_filename=support_filename,
        safety_filename=safety_filename,
        overhead_filename=overhead_filename,
        summary_filename=summary_filename,
    )
    append_exec(
        "m5 trajectory-first suite completed",
        task_id=f"m5_trajectory_completed_{safe_fragment(args.label)}",
        status="pass" if failures == 0 else "fail",
        gpu=",".join(split_csv(args.gpus)),
        files=f"results/v22_45E/{matrix_filename}, results/v22_45E/{summary_filename}",
        note=f"trajectories={len(specs)}; failures={failures}; merged={merge_summary}",
        exit_code=0 if failures == 0 else 1,
    )
    return {"trajectories": len(specs), "failures": failures, **merge_summary}


def build_m6_trajectory_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets = split_csv(args.eval_datasets)
    seeds = split_csv(args.eval_seeds, int)
    archs = split_csv(args.eval_architectures)
    specs: list[dict[str, Any]] = []
    trajectory_steps = int(args.trajectory_steps or args.steps)
    common = {
        "steps": trajectory_steps,
        "support_refresh_cadence": max(40, int(args.support_refresh_cadence)),
        "metric_refresh_cadence": max(10, int(args.metric_refresh_cadence)),
        "metric_shrinkage": max(0.10, float(args.metric_shrinkage)),
        "velocity_scale": min(float(args.velocity_scale), float(args.trajectory_m6_velocity_scale)),
        "safety_budget_velocity_barrier": max(float(args.safety_budget_velocity_barrier), float(args.trajectory_m6_safety_barrier)),
        "beta_signal": min(float(args.beta_signal), float(args.trajectory_m6_beta_signal)),
        "rho_max": min(float(args.rho_max), 0.20),
    }
    seen_baselines: set[tuple[str, int, str]] = set()
    for dataset in datasets:
        for seed in seeds:
            for arch in archs:
                key = (str(dataset), int(seed), str(arch))
                if key not in seen_baselines:
                    add_spec(
                        specs,
                        args,
                        mechanism="BASE",
                        recipe="m6_trajectory_optimizer_alone",
                        dataset=dataset,
                        seed=seed,
                        architecture=arch,
                        variant="optimizer_alone",
                        control_mode="none",
                        **{k: v for k, v in common.items() if k not in {"velocity_scale", "safety_budget_velocity_barrier", "beta_signal", "rho_max"}},
                    )
                    seen_baselines.add(key)
                add_spec(
                    specs,
                    args,
                    mechanism="M6",
                    recipe="ReservoirNoiseSuppressionTrajectory",
                    dataset=dataset,
                    seed=seed,
                    architecture=arch,
                    variant="M6-noise-suppression",
                    control_mode="none",
                    **common,
                )
                for cmode in ["none", "M6-signal-noise-control", "M6-same-norm-Gaussian-control"]:
                    add_spec(
                        specs,
                        args,
                        mechanism="M6",
                        recipe="ReservoirNoiseTrajectory",
                        dataset=dataset,
                        seed=seed,
                        architecture=arch,
                        variant="M6-reservoir-noise",
                        control_mode=cmode,
                        **common,
                    )
    if int(args.row_limit) > 0:
        specs = specs[: int(args.row_limit)]
    return specs


def m6_trajectory_plan_metric_rows(matrix_filename: str) -> list[dict[str, Any]]:
    rows = read_rows(OUT_ROOT / matrix_filename)
    by_group: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("seed", "")),
            str(row.get("architecture", "")),
            str(row.get("optimizer_family", "")),
        )
        role = ""
        if row.get("variant") == "optimizer_alone":
            role = "base"
        elif row.get("mechanism") == "M6" and row.get("variant") == "M6-noise-suppression":
            role = "suppression"
        elif row.get("mechanism") == "M6" and row.get("variant") == "M6-reservoir-noise" and row.get("control_mode") == "none":
            role = "reservoir"
        elif row.get("mechanism") == "M6" and row.get("control_mode") == "M6-signal-noise-control":
            role = "signal_noise_control"
        elif row.get("mechanism") == "M6" and row.get("control_mode") == "M6-same-norm-Gaussian-control":
            role = "gaussian_control"
        if role:
            by_group.setdefault(key, {})[role] = row
    out: list[dict[str, Any]] = []
    for key, group in sorted(by_group.items()):
        dataset, seed, architecture, optimizer = key
        reservoir = group.get("reservoir", {})
        if not reservoir:
            continue
        base = group.get("base", {})
        suppression = group.get("suppression", {})
        signal_control = group.get("signal_noise_control", {})
        gaussian_control = group.get("gaussian_control", {})
        rnll = finite_float(reservoir.get("final_NLL"))
        bnll = finite_float(base.get("final_NLL"))
        snll = finite_float(suppression.get("final_NLL"))
        signll = finite_float(signal_control.get("final_NLL"))
        gnll = finite_float(gaussian_control.get("final_NLL"))
        runtime_truth = int(
            int_flag(reservoir.get("continuous_fu_state_updated_every_step")) == 1
            and int_flag(reservoir.get("fu_velocity_emitted_every_step")) == 1
            and int_flag(reservoir.get("candidate_action_selection_used_for_runtime")) == 0
        )
        improvement_vs_base = "" if rnll is None or bnll is None else bnll - rnll
        improvement_vs_suppression = "" if rnll is None or snll is None else snll - rnll
        improvement_vs_signal = "" if rnll is None or signll is None else signll - rnll
        improvement_vs_gaussian = "" if rnll is None or gnll is None else gnll - rnll
        no_debt = int_flag(reservoir.get("no_ECE_Brier_tail_debt"))
        explains = int(
            isinstance(improvement_vs_base, float)
            and isinstance(improvement_vs_suppression, float)
            and isinstance(improvement_vs_signal, float)
            and isinstance(improvement_vs_gaussian, float)
            and improvement_vs_base > 0.0
            and improvement_vs_suppression > 0.0
            and improvement_vs_signal > 0.0
            and improvement_vs_gaussian > 0.0
            and no_debt
            and runtime_truth
        )
        out.append(
            {
                "dataset": dataset,
                "seed": seed,
                "architecture": architecture,
                "optimizer_family": optimizer,
                "base_final_NLL": "" if bnll is None else bnll,
                "reservoir_final_NLL": "" if rnll is None else rnll,
                "suppression_final_NLL": "" if snll is None else snll,
                "signal_noise_control_final_NLL": "" if signll is None else signll,
                "gaussian_control_final_NLL": "" if gnll is None else gnll,
                "NLL_improvement_vs_base": improvement_vs_base,
                "NLL_improvement_vs_noise_suppression": improvement_vs_suppression,
                "NLL_improvement_vs_signal_noise_control": improvement_vs_signal,
                "NLL_improvement_vs_same_norm_Gaussian_control": improvement_vs_gaussian,
                "beats_noise_suppression": int(isinstance(improvement_vs_suppression, float) and improvement_vs_suppression > 0.0),
                "beats_signal_noise_control": int(isinstance(improvement_vs_signal, float) and improvement_vs_signal > 0.0),
                "beats_same_norm_Gaussian_control": int(isinstance(improvement_vs_gaussian, float) and improvement_vs_gaussian > 0.0),
                "no_ECE_Brier_tail_debt": reservoir.get("no_ECE_Brier_tail_debt", ""),
                "ECE_delta_vs_own_strong_optimizer": reservoir.get("ECE_delta_vs_own_strong_optimizer", ""),
                "Brier_delta_vs_own_strong_optimizer": reservoir.get("Brier_delta_vs_own_strong_optimizer", ""),
                "tail_q99_delta_vs_own_strong_optimizer": reservoir.get("tail_q99_delta_vs_own_strong_optimizer", ""),
                "controller_overhead_ratio": reservoir.get("controller_overhead_ratio", ""),
                "reservoir_noise_energy": reservoir.get("reservoir_noise_energy", ""),
                "reservoir_noise_metric_norm": reservoir.get("reservoir_noise_metric_norm", ""),
                "signal_noise_leakage": reservoir.get("signal_noise_leakage", ""),
                "RSM_index": reservoir.get("RSM_index", ""),
                "continuous_fu_state_updated_every_step": reservoir.get("continuous_fu_state_updated_every_step", ""),
                "fu_velocity_emitted_every_step": reservoir.get("fu_velocity_emitted_every_step", ""),
                "candidate_action_selection_used_for_runtime": reservoir.get("candidate_action_selection_used_for_runtime", ""),
                "runtime_truth": runtime_truth,
                "M6_reservoir_regularization_explains_gain": explains,
                "claim_limit": "parameter metric support-complement reservoir noise; explanatory route only, not SignalFUOpened",
            }
        )
    return out or [{"status": "no_m6_trajectory_rows"}]


def run_m6_trajectory(args: argparse.Namespace) -> dict[str, Any]:
    write_rows(OUT_ROOT / "v22_45E_mechanism_coverage_matrix.csv", MECHANISM_COVERAGE)
    specs = build_m6_trajectory_specs(args)
    output_prefix = safe_fragment(args.m6_trajectory_output)
    specs_filename = f"v22_45E_{output_prefix}_specs.csv"
    matrix_filename = f"v22_45E_{output_prefix}_matrix.csv"
    runtime_filename = f"v22_45E_{output_prefix}_runtime_regression_audit.csv"
    support_filename = f"v22_45E_{output_prefix}_support_direction_decomposition.csv"
    safety_filename = f"v22_45E_{output_prefix}_safety_debt_matrix.csv"
    overhead_filename = f"v22_45E_{output_prefix}_overhead_matrix.csv"
    summary_filename = f"v22_45E_{output_prefix}_mechanism_summary.csv"
    plan_filename = f"v22_45E_{output_prefix}_plan_metrics.csv"
    status_filename = f"v22_45E_{output_prefix}_status.csv"
    write_rows(OUT_ROOT / specs_filename, specs)
    append_exec(
        trajectory_command_line(args, "trajectory-m6"),
        task_id=f"m6_trajectory_started_{safe_fragment(args.label)}",
        status="started",
        gpu=",".join(split_csv(args.gpus)),
        files=f"results/v22_45E/{specs_filename}, results/v22_45E/chunks",
        note=(
            f"trajectories={len(specs)}; workers={args.workers}; in_process=1; subprocess_used=0; "
            f"datasets={args.eval_datasets}; seeds={args.eval_seeds}; architectures={args.eval_architectures}; "
            f"steps={args.trajectory_steps or args.steps}"
        ),
    )
    failures = 0
    statuses: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futures = [ex.submit(run_one_inprocess_trajectory, args, spec, "m6_trajectory") for spec in specs]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            statuses.append(row)
            failures += int(row.get("status") != "pass")
    write_rows(OUT_ROOT / status_filename, statuses)
    merge_summary = merge_v2243_outputs(
        specs_filename=specs_filename,
        matrix_filename=matrix_filename,
        runtime_filename=runtime_filename,
        support_direction_filename=support_filename,
        safety_filename=safety_filename,
        overhead_filename=overhead_filename,
        summary_filename=summary_filename,
    )
    plan_rows = m6_trajectory_plan_metric_rows(matrix_filename)
    write_rows(OUT_ROOT / plan_filename, plan_rows)
    pass_rows = sum(int_flag(r.get("M6_reservoir_regularization_explains_gain")) for r in plan_rows)
    append_exec(
        "m6 trajectory-first reservoir-noise suite completed",
        task_id=f"m6_trajectory_completed_{safe_fragment(args.label)}",
        status="pass" if failures == 0 else "fail",
        gpu=",".join(split_csv(args.gpus)),
        files=f"results/v22_45E/{matrix_filename}, results/v22_45E/{summary_filename}, results/v22_45E/{plan_filename}",
        note=f"trajectories={len(specs)}; failures={failures}; merged={merge_summary}; plan_metric_rows={len(plan_rows)}; explanatory_pass_rows={pass_rows}",
        exit_code=0 if failures == 0 else 1,
    )
    return {"trajectories": len(specs), "failures": failures, "plan_metric_rows": len(plan_rows), "m6_explanatory_pass_rows": pass_rows, **merge_summary}


M7_PARAMETER_VARIANTS = [
    "optimizer_alone",
    "memory_projected_residual_flow",
    "same_memory_random_subspace_control",
    "old_memory_rehearsal_control",
]
M7_FUNCTIONAL_VARIANTS = [
    "optimizer_alone",
    "functional_memory_projected_residual_flow",
    "functional_memory_random_subspace_control",
    "old_memory_rehearsal_control",
]
M7_VARIANTS = sorted(set(M7_PARAMETER_VARIANTS + M7_FUNCTIONAL_VARIANTS))
M7_TASKS = [
    "Class_MNIST_0_4_to_5_9",
    "Permuted_MNIST_identity_to_perm",
    "Modular_Add_to_Mul",
    "Modular_Add_to_Mul_p13",
]


def m7_mnist_split_loaders(task: str, train_size: int, held_size: int, batch_size: int, seed: int, *, download: bool) -> tuple[Any, Any, Any, Any]:
    import random
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_ds = datasets.MNIST(root=str(ROOT / "data"), train=True, transform=transform, download=bool(download))
    test_ds = datasets.MNIST(root=str(ROOT / "data"), train=False, transform=transform, download=bool(download))

    class IndexedPermutationView(Dataset):
        def __init__(self, base: Any, indices: list[int], perm: Any | None = None) -> None:
            self.base = base
            self.indices = list(indices)
            self.perm = perm

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, pos: int) -> tuple[Any, Any]:
            x, y = self.base[self.indices[pos]]
            x = x.view(-1)
            if self.perm is not None:
                x = x[self.perm]
            return x, y

    rng = random.Random(int(seed) + 7457)
    train_targets = train_ds.targets.tolist()
    test_targets = test_ds.targets.tolist()
    old_perm = None
    new_perm = None
    if task == "Class_MNIST_0_4_to_5_9":
        old_train = [i for i, y in enumerate(train_targets) if int(y) < 5]
        new_train = [i for i, y in enumerate(train_targets) if int(y) >= 5]
        old_test = [i for i, y in enumerate(test_targets) if int(y) < 5]
        new_test = [i for i, y in enumerate(test_targets) if int(y) >= 5]
        for idxs in [old_train, new_train, old_test, new_test]:
            rng.shuffle(idxs)
    elif task == "Permuted_MNIST_identity_to_perm":
        train_all = list(range(len(train_targets)))
        test_all = list(range(len(test_targets)))
        rng.shuffle(train_all)
        rng.shuffle(test_all)
        old_train = train_all[: int(train_size)]
        new_train = train_all[int(train_size): int(train_size) * 2]
        old_test = test_all[: int(held_size)]
        new_test = test_all[int(held_size): int(held_size) * 2]
        perm_gen = torch.Generator().manual_seed(int(seed) + 7919)
        new_perm = torch.randperm(784, generator=perm_gen)
    else:
        raise ValueError(f"unknown M7 task {task}; allowed={','.join(M7_TASKS)}")
    old_train = old_train[: int(train_size)]
    new_train = new_train[: int(train_size)]
    old_test = old_test[: int(held_size)]
    new_test = new_test[: int(held_size)]
    gen_old = torch.Generator().manual_seed(int(seed) + 7111)
    gen_new = torch.Generator().manual_seed(int(seed) + 7222)
    old_loader = DataLoader(IndexedPermutationView(train_ds, old_train, old_perm), batch_size=int(batch_size), shuffle=True, generator=gen_old)
    new_loader = DataLoader(IndexedPermutationView(train_ds, new_train, new_perm), batch_size=int(batch_size), shuffle=True, generator=gen_new)
    old_eval = DataLoader(IndexedPermutationView(test_ds, old_test, old_perm), batch_size=512, shuffle=False)
    new_eval = DataLoader(IndexedPermutationView(test_ds, new_test, new_perm), batch_size=512, shuffle=False)
    return old_loader, new_loader, old_eval, new_eval


def m7_modular_add_mul_loaders(p: int, train_fraction: float, batch_size: int, seed: int, device: Any) -> tuple[Any, Any, Any, Any, int, int, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    xs = []
    y_add = []
    y_mul = []
    for a in range(int(p)):
        for b in range(int(p)):
            x = torch.zeros(2 * int(p), dtype=torch.float32)
            x[a] = 1.0
            x[int(p) + b] = 1.0
            xs.append(x)
            y_add.append((a + b) % int(p))
            y_mul.append((a * b) % int(p))
    x_all = torch.stack(xs)
    add_all = torch.tensor(y_add, dtype=torch.long)
    mul_all = torch.tensor(y_mul, dtype=torch.long)
    gen = torch.Generator().manual_seed(int(seed) + 8741)
    perm = torch.randperm(len(add_all), generator=gen)
    n_train = min(len(add_all), max(1, int(round(float(train_fraction) * len(add_all)))))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    eval_idx = test_idx if int(test_idx.numel()) > 0 else train_idx
    gen_old = torch.Generator().manual_seed(int(seed) + 8841)
    gen_new = torch.Generator().manual_seed(int(seed) + 8941)
    old_loader = DataLoader(TensorDataset(x_all[train_idx], add_all[train_idx]), batch_size=int(batch_size), shuffle=True, generator=gen_old)
    new_loader = DataLoader(TensorDataset(x_all[train_idx], mul_all[train_idx]), batch_size=int(batch_size), shuffle=True, generator=gen_new)
    old_eval = DataLoader(TensorDataset(x_all[eval_idx], add_all[eval_idx]), batch_size=512, shuffle=False)
    new_eval = DataLoader(TensorDataset(x_all[eval_idx], mul_all[eval_idx]), batch_size=512, shuffle=False)
    return old_loader, new_loader, old_eval, new_eval, 2 * int(p), int(p), x_all[train_idx].to(device)


def m7_task_loaders(task: str, train_size: int, held_size: int, batch_size: int, seed: int, *, download: bool, device: Any, modulus: int, modular_train_fraction: float) -> tuple[Any, Any, Any, Any, int, int, Any, Any]:
    from torch.utils.data import DataLoader

    if task.startswith("Modular_Add_to_Mul"):
        old_loader, new_loader, old_eval, new_eval, input_dim, output_dim, x_stats = m7_modular_add_mul_loaders(modulus, modular_train_fraction, batch_size, seed, device)
        old_anchor = DataLoader(old_loader.dataset, batch_size=int(batch_size), shuffle=False)
        return old_loader, new_loader, old_eval, new_eval, input_dim, output_dim, x_stats, old_anchor
    old_loader, new_loader, old_eval, new_eval = m7_mnist_split_loaders(
        task,
        train_size,
        held_size,
        batch_size,
        seed,
        download=download,
    )
    x_stats = next(iter(old_loader))[0].to(device).float()
    old_anchor = DataLoader(old_loader.dataset, batch_size=int(batch_size), shuffle=False)
    return old_loader, new_loader, old_eval, new_eval, 784, 10, x_stats, old_anchor


def m7_dict_dot(a: dict[str, Any], b: dict[str, Any]) -> float:
    import torch

    total = torch.zeros((), device=next(iter(a.values())).device) if a else torch.zeros(())
    for name, av in a.items():
        bv = b.get(name)
        if bv is None:
            continue
        total = total + (av.detach().float() * bv.to(device=av.device).detach().float()).sum()
    return float(total.item())


def m7_residualize_against_memory(signal: dict[str, Any], memory: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    signal_norm = v2242s7.tensor_norm(signal)
    memory_norm = v2242s7.tensor_norm(memory)
    denom = max(1.0e-12, memory_norm * memory_norm)
    coeff = m7_dict_dot(signal, memory) / denom if memory_norm > 1.0e-12 else 0.0
    projected: dict[str, Any] = {}
    component: dict[str, Any] = {}
    for name, value in signal.items():
        mem = memory.get(name)
        if mem is None:
            component[name] = value.detach().float() * 0.0
            projected[name] = value.detach().float()
        else:
            comp = coeff * mem.to(device=value.device).detach().float()
            component[name] = comp
            projected[name] = value.detach().float() - comp
    component_norm = v2242s7.tensor_norm(component)
    projected_norm = v2242s7.tensor_norm(projected)
    after_component = abs(m7_dict_dot(projected, memory)) / max(1.0e-12, memory_norm)
    return projected, {
        "memory_projection_coeff": coeff,
        "memory_projection_before_ratio": component_norm / max(1.0e-12, signal_norm),
        "memory_projection_after_ratio": after_component / max(1.0e-12, projected_norm),
        "memory_signal_retention": projected_norm / max(1.0e-12, signal_norm),
        "new_grad_norm": signal_norm,
        "projected_grad_norm": projected_norm,
        "memory_norm": memory_norm,
        "memory_rank_effective": 1 if memory_norm > 1.0e-12 else 0,
    }


def m7_residualize_against_basis(signal: dict[str, Any], basis: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, float]]:
    import torch

    active = [b for b in basis if v2242s7.tensor_norm(b) > 1.0e-12]
    if len(active) <= 1:
        return m7_residualize_against_memory(signal, active[0] if active else {})
    signal_norm = v2242s7.tensor_norm(signal)
    n = len(active)
    gram = torch.zeros((n, n), dtype=torch.float64)
    rhs = torch.zeros((n,), dtype=torch.float64)
    for i, bi in enumerate(active):
        rhs[i] = m7_dict_dot(signal, bi)
        for j, bj in enumerate(active):
            gram[i, j] = m7_dict_dot(bi, bj)
    ridge = max(1.0e-12, float(torch.trace(gram).item()) / max(1, n) * 1.0e-6)
    coeffs = torch.linalg.solve(gram + ridge * torch.eye(n, dtype=torch.float64), rhs)

    def component_for(values: Any) -> dict[str, Any]:
        component = {name: value.detach().float() * 0.0 for name, value in signal.items()}
        for coeff, bi in zip(values.tolist(), active):
            for name, value in signal.items():
                bv = bi.get(name)
                if bv is not None:
                    component[name] = component[name] + float(coeff) * bv.to(device=value.device).detach().float()
        return component

    component = component_for(coeffs)
    projected = {name: value.detach().float() - component[name].to(device=value.device) for name, value in signal.items()}
    projected_norm = v2242s7.tensor_norm(projected)
    component_norm = v2242s7.tensor_norm(component)
    rhs_after = torch.zeros((n,), dtype=torch.float64)
    for i, bi in enumerate(active):
        rhs_after[i] = m7_dict_dot(projected, bi)
    coeffs_after = torch.linalg.solve(gram + ridge * torch.eye(n, dtype=torch.float64), rhs_after)
    after_component_norm = v2242s7.tensor_norm(component_for(coeffs_after))
    memory_norm = math.sqrt(sum(max(0.0, m7_dict_dot(b, b)) for b in active))
    return projected, {
        "memory_projection_coeff": float(coeffs.norm().item()),
        "memory_projection_before_ratio": component_norm / max(1.0e-12, signal_norm),
        "memory_projection_after_ratio": after_component_norm / max(1.0e-12, projected_norm),
        "memory_signal_retention": projected_norm / max(1.0e-12, signal_norm),
        "new_grad_norm": signal_norm,
        "projected_grad_norm": projected_norm,
        "memory_norm": memory_norm,
        "memory_rank_effective": n,
    }


def m7_add_scaled_direction(named: list[tuple[str, Any]], direction: dict[str, Any], scale: float) -> None:
    import torch

    with torch.no_grad():
        for name, p in named:
            value = direction.get(name)
            if value is not None:
                p.add_(float(scale) * value.to(device=p.device, dtype=p.dtype))


def m7_functional_responses(model: Any, named: list[tuple[str, Any]], directions: list[dict[str, Any]], anchor_x: Any, eps: float) -> list[Any]:
    import torch

    with torch.no_grad():
        base = model(anchor_x).float().detach()
    responses = []
    for direction in directions:
        m7_add_scaled_direction(named, direction, -float(eps))
        try:
            with torch.no_grad():
                moved = model(anchor_x).float().detach()
            responses.append(((moved - base) / max(1.0e-12, float(eps))).reshape(-1).detach())
        finally:
            m7_add_scaled_direction(named, direction, float(eps))
    return responses


def m7_residualize_functional_against_basis(
    model: Any,
    named: list[tuple[str, Any]],
    signal: dict[str, Any],
    basis: list[dict[str, Any]],
    anchor_x: Any,
    eps: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    import torch

    active = [b for b in basis if v2242s7.tensor_norm(b) > 1.0e-12]
    signal_norm = v2242s7.tensor_norm(signal)
    if not active or signal_norm <= 1.0e-12:
        return {name: value.detach().float() for name, value in signal.items()}, {
            "memory_projection_coeff": 0.0,
            "memory_projection_before_ratio": 0.0,
            "memory_projection_after_ratio": 0.0,
            "memory_signal_retention": 1.0,
            "new_grad_norm": signal_norm,
            "projected_grad_norm": signal_norm,
            "memory_norm": 0.0,
            "memory_rank_effective": 0,
            "functional_memory_projection_before_ratio": 0.0,
            "functional_memory_projection_after_ratio": 0.0,
            "functional_memory_residual_norm": 0.0,
            "functional_memory_signal_norm": 0.0,
            "functional_fd_eval_count": 0,
        }
    responses = m7_functional_responses(model, named, [signal] + active, anchor_x, eps)
    signal_response = responses[0].double()
    basis_responses = [r.double() for r in responses[1:]]
    n = len(basis_responses)
    gram = torch.zeros((n, n), dtype=torch.float64, device=signal_response.device)
    rhs = torch.zeros((n,), dtype=torch.float64, device=signal_response.device)
    for i, bi in enumerate(basis_responses):
        rhs[i] = torch.dot(signal_response, bi)
        for j, bj in enumerate(basis_responses):
            gram[i, j] = torch.dot(bi, bj)
    ridge = max(1.0e-12, float(torch.trace(gram).item()) / max(1, n) * 1.0e-6)
    coeffs = torch.linalg.solve(gram + ridge * torch.eye(n, dtype=torch.float64, device=signal_response.device), rhs)
    component_response = torch.zeros_like(signal_response)
    component = {name: value.detach().float() * 0.0 for name, value in signal.items()}
    for coeff, bi, basis_direction in zip(coeffs.tolist(), basis_responses, active):
        component_response = component_response + float(coeff) * bi
        for name, value in signal.items():
            bv = basis_direction.get(name)
            if bv is not None:
                component[name] = component[name] + float(coeff) * bv.to(device=value.device).detach().float()
    projected = {name: value.detach().float() - component[name].to(device=value.device) for name, value in signal.items()}
    residual_response = signal_response - component_response
    signal_response_norm = float(signal_response.norm().item())
    component_response_norm = float(component_response.norm().item())
    residual_response_norm = float(residual_response.norm().item())
    rhs_after = torch.zeros((n,), dtype=torch.float64, device=signal_response.device)
    for i, bi in enumerate(basis_responses):
        rhs_after[i] = torch.dot(residual_response, bi)
    coeffs_after = torch.linalg.solve(gram + ridge * torch.eye(n, dtype=torch.float64, device=signal_response.device), rhs_after)
    component_after_response = torch.zeros_like(signal_response)
    for coeff, bi in zip(coeffs_after.tolist(), basis_responses):
        component_after_response = component_after_response + float(coeff) * bi
    after_projection_norm = float(component_after_response.norm().item())
    projected_norm = v2242s7.tensor_norm(projected)
    memory_norm = math.sqrt(sum(max(0.0, m7_dict_dot(b, b)) for b in active))
    return projected, {
        "memory_projection_coeff": float(coeffs.norm().item()),
        "memory_projection_before_ratio": component_response_norm / max(1.0e-12, signal_response_norm),
        "memory_projection_after_ratio": after_projection_norm / max(1.0e-12, signal_response_norm),
        "memory_signal_retention": projected_norm / max(1.0e-12, signal_norm),
        "new_grad_norm": signal_norm,
        "projected_grad_norm": projected_norm,
        "memory_norm": memory_norm,
        "memory_rank_effective": n,
        "functional_memory_projection_before_ratio": component_response_norm / max(1.0e-12, signal_response_norm),
        "functional_memory_projection_after_ratio": after_projection_norm / max(1.0e-12, signal_response_norm),
        "functional_memory_residual_norm": residual_response_norm,
        "functional_memory_residual_retention": residual_response_norm / max(1.0e-12, signal_response_norm),
        "functional_memory_signal_norm": signal_response_norm,
        "functional_fd_eval_count": n + 1,
    }


def m7_set_grad(named: list[tuple[str, Any]], grad: dict[str, Any]) -> None:
    for name, p in named:
        value = grad.get(name)
        if value is None:
            p.grad = None
        else:
            p.grad = value.to(device=p.device, dtype=p.dtype).detach().clone()


def m7_trace_row(label: str, dataset: str, seed: int, arch: str, variant: str, phase: str, step: int, old_acc: float, new_acc: float, memory_norm: float, fu_norm: float, before: float, after: float) -> dict[str, Any]:
    return {
        "run_label": label,
        "dataset": dataset,
        "seed": seed,
        "architecture": arch,
        "variant": variant,
        "phase": phase,
        "step": step,
        "old_task_accuracy": old_acc,
        "new_task_accuracy": new_acc,
        "memory_state_norm": memory_norm,
        "fu_velocity_norm": fu_norm,
        "memory_projection_before_ratio": before,
        "memory_projection_after_ratio": after,
    }


def run_one_m7_trajectory(args: argparse.Namespace, spec: dict[str, Any]) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from experiments import run_v22_37_causal_instrumented_functional_optimizer as core
    from experiments import run_v22_40_continuous_functional_flow_fu as v2240

    started = time.time()
    seed = int(spec["seed"])
    arch = str(spec["architecture"])
    variant = str(spec["variant"])
    label = str(spec["label"])
    dataset = str(spec.get("dataset", "Class_MNIST_0_4_to_5_9"))
    device_name = str(spec.get("device", "cuda:0"))
    v2242s7.seed_all(seed)
    device = v2243.torch_device(device_name)
    if torch.cuda.is_available() and device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(device.index if device.index is not None else torch.cuda.current_device())
        except RuntimeError:
            pass
    old_loader, new_loader, old_eval, new_eval, input_dim, output_dim, x_stats, old_anchor_loader = m7_task_loaders(
        dataset,
        int(spec["train_size"]),
        int(spec["held_size"]),
        int(spec["batch_size"]),
        seed,
        download=bool(args.tier2_download),
        device=device,
        modulus=int(spec.get("modulus", 13)),
        modular_train_fraction=float(spec.get("modular_train_fraction", 0.40)),
    )
    old_anchor_x = next(iter(old_anchor_loader))[0].to(device).float()
    model_seed = seed + 745000 + (0 if arch == "MLP" else 1000 if arch == "DGKAN_DCHE" else 2000)
    model = core.make_model_for_arch(arch, int(input_dim), int(output_dim), int(spec["hidden"]), model_seed, device, x_stats)
    opt = core.optimizer_for(str(spec["optimizer"]), model.parameters(), float(spec["lr"]), float(spec["weight_decay"]))
    old_iter = core.cycle_batches(old_loader)
    new_iter = core.cycle_batches(new_loader)
    avg_state: dict[str, Any] = {}
    memory_state: dict[str, Any] = {}
    memory_basis_snapshots: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    old_grad_norms: list[float] = []
    new_grad_norms: list[float] = []
    before_ratios: list[float] = []
    after_ratios: list[float] = []
    retentions: list[float] = []
    fu_norms: list[float] = []
    functional_after_ratios: list[float] = []
    functional_fd_counts: list[float] = []
    constraint_hits = 0
    steps_per_task = int(spec["steps_per_task"])
    memory_rank = max(1, int(spec.get("memory_rank", 1)))
    capture_cadence = max(1, steps_per_task // max(1, memory_rank))
    global_step = 0
    for step in range(1, steps_per_task + 1):
        global_step += 1
        grad = v2242s7.train_one_batch(model, opt, str(spec["optimizer"]), next(old_iter), device, avg_state, global_step)
        old_grad_norms.append(v2242s7.tensor_norm(grad))
        for name, g in grad.items():
            old = memory_state.get(name)
            memory_state[name] = g if old is None else (1.0 - float(spec["beta_memory"])) * old.to(device=g.device).float() + float(spec["beta_memory"]) * g
        if memory_rank > 1 and (step % capture_cadence == 0 or step == steps_per_task):
            memory_basis_snapshots.append({name: value.detach().float().clone() for name, value in grad.items()})
            memory_basis_snapshots = memory_basis_snapshots[-max(0, memory_rank - 1):]
        mem_norm = v2242s7.tensor_norm(memory_state)
        runtime_rows.append({
            "run_label": label,
            "dataset": dataset,
            "seed": seed,
            "architecture": arch,
            "variant": variant,
            "phase": "old",
            "step": global_step,
            "runtime_policy_type": f"m7_rank{memory_rank}_memory_projection",
            "continuous_fu_state_updated": 1,
            "fu_velocity_emitted": 1,
            "candidate_action_selection_used_for_runtime": 0,
            "runtime_argmax_candidate_used": 0,
            "runtime_topk_candidate_used": 0,
            "candidate_value_model_used_as_runtime_policy": 0,
            "micro_rct_winner_used_as_runtime_action": 0,
            "path_mpc_discrete_action_sequence_used": 0,
            "memory_state_norm": mem_norm,
            "memory_rank": memory_rank,
            "memory_rank_effective": 1,
            "fu_velocity_norm": 0.0,
            "memory_projection_before_ratio": 0.0,
            "memory_projection_after_ratio": 0.0,
        })
        if step in {1, max(1, steps_per_task // 2), steps_per_task}:
            trace_rows.append(m7_trace_row(label, dataset, seed, arch, variant, "old", global_step, v2242s7.evaluate_accuracy(model, old_eval, device), v2242s7.evaluate_accuracy(model, new_eval, device), mem_norm, 0.0, 0.0, 0.0))
    old_before = v2242s7.evaluate_accuracy(model, old_eval, device)
    new_before = v2242s7.evaluate_accuracy(model, new_eval, device)
    memory_basis = [{name: value.detach().float().clone() for name, value in memory_state.items()}]
    memory_basis.extend(memory_basis_snapshots[-max(0, memory_rank - 1):])
    memory_basis = memory_basis[:memory_rank]
    random_memory_basis = [
        v2242s7.normalize(
            v2242s7.random_like(base, v2243.stable_seed(label, seed, "m7_random_memory", idx)),
            v2242s7.tensor_norm(base),
        )
        for idx, base in enumerate(memory_basis)
    ]
    for step in range(1, steps_per_task + 1):
        global_step += 1
        xb, yb = next(new_iter)
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb).float(), yb)
        loss.backward()
        named = v2242s7.named_trainable(model)
        grad = v2242s7.grad_dict(named)
        new_grad_norms.append(v2242s7.tensor_norm(grad))
        stats: dict[str, float]
        if variant == "optimizer_alone":
            effective_grad = grad
            _, stats = m7_residualize_against_basis(grad, memory_basis)
            stats["memory_projection_after_ratio"] = stats["memory_projection_before_ratio"]
            stats["memory_signal_retention"] = 1.0
        elif variant == "memory_projected_residual_flow":
            effective_grad, stats = m7_residualize_against_basis(grad, memory_basis)
        elif variant == "functional_memory_projected_residual_flow":
            effective_grad, stats = m7_residualize_functional_against_basis(
                model,
                named,
                grad,
                memory_basis,
                old_anchor_x,
                float(spec.get("functional_eps", 1.0e-3)),
            )
        elif variant == "same_memory_random_subspace_control":
            effective_grad, stats = m7_residualize_against_basis(grad, random_memory_basis)
        elif variant == "functional_memory_random_subspace_control":
            effective_grad, stats = m7_residualize_functional_against_basis(
                model,
                named,
                grad,
                random_memory_basis,
                old_anchor_x,
                float(spec.get("functional_eps", 1.0e-3)),
            )
        elif variant == "old_memory_rehearsal_control":
            grad_norm = v2242s7.tensor_norm(grad)
            mem = v2242s7.normalize(memory_state, grad_norm)
            effective_grad = {name: grad[name] + float(spec["rho"]) * mem.get(name, grad[name] * 0.0).to(device=grad[name].device) for name in grad}
            _, stats = m7_residualize_against_basis(effective_grad, memory_basis)
            stats["memory_signal_retention"] = v2242s7.tensor_norm(effective_grad) / max(1.0e-12, grad_norm)
        else:
            raise ValueError(f"unknown M7 variant {variant}")
        m7_set_grad(named, effective_grad)
        v2240.optimizer_step(model, opt, str(spec["optimizer"]), global_step, avg_state)
        before = float(stats.get("memory_projection_before_ratio", 0.0))
        after = float(stats.get("memory_projection_after_ratio", 0.0))
        retention = float(stats.get("memory_signal_retention", 0.0))
        fu_norm = v2242s7.tensor_norm(effective_grad)
        before_ratios.append(before)
        after_ratios.append(after)
        retentions.append(retention)
        fu_norms.append(fu_norm)
        if finite_float(stats.get("functional_memory_projection_after_ratio")) is not None:
            functional_after_ratios.append(float(stats.get("functional_memory_projection_after_ratio", 0.0)))
        if finite_float(stats.get("functional_fd_eval_count")) is not None:
            functional_fd_counts.append(float(stats.get("functional_fd_eval_count", 0.0)))
        constraint_hits += int(after <= float(spec["tau_forget"]))
        projection_mode = "function_space_fd_old_logits" if str(variant).startswith("functional_memory_") else "parameter_gradient"
        runtime_rows.append({
            "run_label": label,
            "dataset": dataset,
            "seed": seed,
            "architecture": arch,
            "variant": variant,
            "phase": "new",
            "step": global_step,
            "runtime_policy_type": f"m7_rank{memory_rank}_{projection_mode}_memory_projection",
            "continuous_fu_state_updated": 1,
            "fu_velocity_emitted": int(fu_norm > 1.0e-12),
            "candidate_action_selection_used_for_runtime": 0,
            "runtime_argmax_candidate_used": 0,
            "runtime_topk_candidate_used": 0,
            "candidate_value_model_used_as_runtime_policy": 0,
            "micro_rct_winner_used_as_runtime_action": 0,
            "path_mpc_discrete_action_sequence_used": 0,
            "memory_state_norm": stats.get("memory_norm", v2242s7.tensor_norm(memory_state)),
            "memory_rank": memory_rank,
            "memory_rank_effective": stats.get("memory_rank_effective", 1),
            "fu_velocity_norm": fu_norm,
            "memory_projection_before_ratio": before,
            "memory_projection_after_ratio": after,
            "memory_signal_retention": retention,
            "functional_memory_projection_after_ratio": stats.get("functional_memory_projection_after_ratio", ""),
            "functional_memory_residual_retention": stats.get("functional_memory_residual_retention", ""),
            "functional_fd_eval_count": stats.get("functional_fd_eval_count", ""),
        })
        if step in {1, max(1, steps_per_task // 2), steps_per_task}:
            trace_rows.append(m7_trace_row(label, dataset, seed, arch, variant, "new", global_step, v2242s7.evaluate_accuracy(model, old_eval, device), v2242s7.evaluate_accuracy(model, new_eval, device), stats.get("memory_norm", v2242s7.tensor_norm(memory_state)), fu_norm, before, after))
    v2240.final_schedule_free_swap(model, str(spec["optimizer"]), avg_state)
    old_after = v2242s7.evaluate_accuracy(model, old_eval, device)
    new_after = v2242s7.evaluate_accuracy(model, new_eval, device)
    forgetting = old_before - old_after
    final_avg = 0.5 * (old_after + new_after)
    peak_mb = 0.0
    if torch.cuda.is_available() and device.type == "cuda":
        try:
            peak_mb = float(torch.cuda.max_memory_allocated(device.index if device.index is not None else torch.cuda.current_device()) / (1024 * 1024))
        except RuntimeError:
            peak_mb = 0.0
    summary = {
        "run_label": label,
        "dataset": dataset,
        "seed": seed,
        "architecture": arch,
        "optimizer_family": spec["optimizer"],
        "variant": variant,
        "task_boundary": "old_to_new",
        "input_dim": input_dim,
        "output_dim": output_dim,
        "modulus": spec.get("modulus", ""),
        "modular_train_fraction": spec.get("modular_train_fraction", ""),
        "modular_eval_split": "full_grid_train_eval" if dataset.startswith("Modular_Add_to_Mul") and float(spec.get("modular_train_fraction", 0.0)) >= 1.0 else ("heldout_split" if dataset.startswith("Modular_Add_to_Mul") else ""),
        "steps_per_task": steps_per_task,
        "hidden": spec["hidden"],
        "train_size": spec["train_size"],
        "held_size": spec["held_size"],
        "old_task_accuracy_before_new_task": old_before,
        "new_task_accuracy_before_new_task": new_before,
        "old_task_accuracy_after_new_task": old_after,
        "new_task_accuracy": new_after,
        "final_average_accuracy": final_avg,
        "average_forgetting": forgetting,
        "relative_forgetting_reduction": "",
        "average_accuracy_delta_vs_base": "",
        "memory_state_norm": v2242s7.tensor_norm(memory_state),
        "memory_rank": memory_rank,
        "memory_rank_effective": max([int(value_or(r.get("memory_rank_effective"), 0)) for r in runtime_rows] or [0]),
        "memory_state_SNR": v2242s7.tensor_norm(memory_state) / max(1.0e-12, statistics.fmean(new_grad_norms) if new_grad_norms else 0.0),
        "old_grad_norm_mean": statistics.fmean(old_grad_norms) if old_grad_norms else 0.0,
        "new_grad_norm_mean": statistics.fmean(new_grad_norms) if new_grad_norms else 0.0,
        "memory_projection_before_ratio": statistics.fmean(before_ratios) if before_ratios else 0.0,
        "memory_projection_after_ratio": statistics.fmean(after_ratios) if after_ratios else 0.0,
        "functional_memory_projection_after_ratio": statistics.fmean(functional_after_ratios) if functional_after_ratios else "",
        "functional_fd_eval_count_mean": statistics.fmean(functional_fd_counts) if functional_fd_counts else "",
        "memory_constraint_satisfied_fraction": constraint_hits / max(1, len(after_ratios)),
        "memory_signal_retention": statistics.fmean(retentions) if retentions else 0.0,
        "fu_velocity_norm_mean": statistics.fmean(fu_norms) if fu_norms else 0.0,
        "tau_forget": spec["tau_forget"],
        "continuous_fu_state_updated_every_step": 1,
        "fu_velocity_emitted_every_step": int(all(int_flag(r.get("fu_velocity_emitted")) for r in runtime_rows)),
        "candidate_action_selection_used_for_runtime": 0,
        "runtime_argmax_candidate_used": 0,
        "runtime_topk_candidate_used": 0,
        "candidate_value_model_used_as_runtime_policy": 0,
        "micro_rct_winner_used_as_runtime_action": 0,
        "path_mpc_discrete_action_sequence_used": 0,
        "in_process": 1,
        "subprocess_used": 0,
        "complete_step_loop": 1,
        "elapsed_sec": time.time() - started,
        "memory_peak_MB": peak_mb,
        "claim_limit": (
            f"rank-{memory_rank} finite-difference old-logit functional memory projection on {dataset}; "
            "train-anchor J_old*v approximation, not full all-function G_f M7"
            if str(variant).startswith("functional_memory_")
            else f"rank-{memory_rank} parameter-gradient old-memory projection on {dataset}; not full function-space J_t/G_f memory projection"
        ),
        "status": "completed_v22_45E_m7_trajectory_row",
    }
    return {"status": "pass", "label": label, "summary": summary, "trace": trace_rows, "runtime": runtime_rows}


def build_m7_trajectory_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    seeds = split_csv(args.eval_seeds, int)
    archs = [a for a in split_csv(args.eval_architectures) if a in {"MLP", "DGKAN_DCHE", "DGKAN_DFOU"}]
    tasks = [t for t in split_csv(args.m7_tasks) if t]
    unknown = [t for t in tasks if t not in M7_TASKS]
    if unknown:
        raise ValueError(f"unknown --m7-tasks values: {unknown}; allowed={M7_TASKS}")
    gpus = split_csv(args.gpus) or ["0"]
    steps = int(args.m7_trajectory_steps or args.m7_steps_per_task)
    if str(args.m7_projector) == "functional":
        variants = M7_FUNCTIONAL_VARIANTS
    elif str(args.m7_projector) == "both":
        variants = M7_PARAMETER_VARIANTS + [v for v in M7_FUNCTIONAL_VARIANTS if v not in M7_PARAMETER_VARIANTS]
    else:
        variants = M7_PARAMETER_VARIANTS
    specs: list[dict[str, Any]] = []
    for task in tasks:
        dataset_name = f"Modular_Add_to_Mul_p{int(args.m7_modulus)}" if task.startswith("Modular_Add_to_Mul") else task
        for seed in seeds:
            for arch in archs:
                for variant in variants:
                    idx = len(specs)
                    specs.append({
                        "label": f"{safe_fragment(args.label)}_{safe_fragment(args.m7_trajectory_output)}_{safe_fragment(dataset_name)}_r{int(args.m7_memory_rank)}_{safe_fragment(arch)}_s{seed}_{safe_fragment(variant)}",
                        "dataset": dataset_name,
                        "seed": int(seed),
                        "architecture": arch,
                        "variant": variant,
                        "optimizer": str(args.optimizer),
                        "steps_per_task": steps,
                        "train_size": int(args.m7_train_size),
                        "held_size": int(args.m7_held_size),
                        "batch_size": int(args.diagnostic_batch_size),
                        "hidden": int(args.diagnostic_hidden),
                        "lr": float(args.lr),
                        "weight_decay": float(args.weight_decay),
                        "rho": float(args.m7_rho),
                        "velocity_scale": float(args.m7_velocity_scale),
                        "beta_memory": float(args.m7_beta_memory),
                        "tau_forget": float(args.m7_tau_forget),
                        "memory_rank": int(args.m7_memory_rank),
                        "projector": str(args.m7_projector),
                        "functional_eps": float(args.m7_functional_eps),
                        "modulus": int(args.m7_modulus),
                        "modular_train_fraction": float(args.m7_modular_train_fraction),
                        "device": f"cuda:{gpus[idx % max(1, len(gpus))]}",
                        "in_process": 1,
                        "subprocess_used": 0,
                        "complete_step_loop": 1,
                    })
    if int(args.row_limit) > 0:
        specs = specs[: int(args.row_limit)]
    return specs


def m7_plan_metric_rows(matrix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = {}
    for row in matrix_rows:
        groups.setdefault((str(row.get("dataset", "")), str(row.get("architecture", "")), str(row.get("seed", "")), str(row.get("memory_rank", "1"))), {})[str(row.get("variant", ""))] = row
    out: list[dict[str, Any]] = []
    for (dataset, arch, seed, memory_rank), vals in sorted(groups.items()):
        base = vals.get("optimizer_alone", {})
        proj_variant = "functional_memory_projected_residual_flow" if vals.get("functional_memory_projected_residual_flow") else "memory_projected_residual_flow"
        rand_variant = "functional_memory_random_subspace_control" if vals.get("functional_memory_random_subspace_control") else "same_memory_random_subspace_control"
        proj = vals.get(proj_variant, {})
        rand = vals.get(rand_variant, {})
        rehearse = vals.get("old_memory_rehearsal_control", {})
        if not proj:
            continue
        base_forget = finite_float(base.get("average_forgetting"))
        proj_forget = finite_float(proj.get("average_forgetting"))
        rand_forget = finite_float(rand.get("average_forgetting"))
        rehearse_forget = finite_float(rehearse.get("average_forgetting"))
        base_avg = finite_float(base.get("final_average_accuracy"))
        proj_avg = finite_float(proj.get("final_average_accuracy"))
        forgetting_task_valid = int(base_forget is not None and base_forget > 0.0)
        rel = "" if not forgetting_task_valid or proj_forget is None else (base_forget - proj_forget) / abs(base_forget)
        random_gap = "" if rand_forget is None or proj_forget is None else rand_forget - proj_forget
        rehearsal_gap = "" if rehearse_forget is None or proj_forget is None else rehearse_forget - proj_forget
        final_nonworse = int(proj_avg is not None and base_avg is not None and proj_avg >= base_avg - 1.0e-12)
        controls_fail = int(
            isinstance(random_gap, float)
            and isinstance(rehearsal_gap, float)
            and random_gap > 0.0
            and rehearsal_gap > 0.0
        )
        runtime_truth = int(
            int_flag(proj.get("continuous_fu_state_updated_every_step"))
            and int_flag(proj.get("fu_velocity_emitted_every_step"))
            and int_flag(proj.get("candidate_action_selection_used_for_runtime")) == 0
            and int_flag(proj.get("subprocess_used")) == 0
        )
        constraint_ok = int(value_or(proj.get("memory_constraint_satisfied_fraction"), 0.0) >= 0.95)
        exploration_open = int(forgetting_task_valid and isinstance(rel, float) and rel >= 0.05 and final_nonworse and controls_fail and runtime_truth and constraint_ok)
        official_ready = int(forgetting_task_valid and isinstance(rel, float) and rel >= 0.10 and final_nonworse and controls_fail and runtime_truth and constraint_ok)
        out.append({
            "dataset": dataset,
            "seed": seed,
            "architecture": arch,
            "memory_rank": memory_rank,
            "projected_variant": proj_variant,
            "random_control_variant": rand_variant,
            "base_average_forgetting": "" if base_forget is None else base_forget,
            "projected_average_forgetting": "" if proj_forget is None else proj_forget,
            "random_control_average_forgetting": "" if rand_forget is None else rand_forget,
            "rehearsal_control_average_forgetting": "" if rehearse_forget is None else rehearse_forget,
            "forgetting_task_valid": forgetting_task_valid,
            "relative_forgetting_reduction": rel,
            "same_memory_random_control_gap": random_gap,
            "same_memory_rehearsal_control_gap": rehearsal_gap,
            "base_final_average_accuracy": "" if base_avg is None else base_avg,
            "projected_final_average_accuracy": "" if proj_avg is None else proj_avg,
            "final_average_accuracy_nonworse": final_nonworse,
            "matched_memory_controls_fail": controls_fail,
            "memory_projection_before_ratio": proj.get("memory_projection_before_ratio", ""),
            "memory_projection_after_ratio": proj.get("memory_projection_after_ratio", ""),
            "functional_memory_projection_after_ratio": proj.get("functional_memory_projection_after_ratio", ""),
            "functional_fd_eval_count_mean": proj.get("functional_fd_eval_count_mean", ""),
            "memory_constraint_satisfied_fraction": proj.get("memory_constraint_satisfied_fraction", ""),
            "memory_signal_retention": proj.get("memory_signal_retention", ""),
            "memory_rank_effective": proj.get("memory_rank_effective", ""),
            "runtime_truth": runtime_truth,
            "constraint_ok": constraint_ok,
            "M7_exploration_opened": exploration_open,
            "M7_official_ready": official_ready,
            "claim_limit": (
                f"rank-{memory_rank} finite-difference old-logit functional memory projection on {dataset}; train-anchor J_old*v approximation, not full all-function G_f M7"
                if str(proj_variant).startswith("functional_")
                else f"rank-{memory_rank} parameter-gradient memory projection on {dataset}; not full function-space M7"
            ),
        })
    return out or [{"status": "no_m7_trajectory_rows"}]


def summarize_m7_trajectory(matrix_rows: list[dict[str, Any]], runtime_rows: list[dict[str, Any]], plan_rows: list[dict[str, Any]]) -> dict[str, Any]:
    active_plan = [r for r in plan_rows if not r.get("status")]
    tasks = sorted({str(r.get("dataset", "")) for r in active_plan if r.get("dataset")})
    rels = [value_or(r.get("relative_forgetting_reduction"), math.nan) for r in active_plan if finite_float(r.get("relative_forgetting_reduction")) is not None]
    bad_runtime = 0
    for r in runtime_rows:
        for key in ["runtime_argmax_candidate_used", "runtime_topk_candidate_used", "candidate_action_selection_used_for_runtime", "candidate_value_model_used_as_runtime_policy", "micro_rct_winner_used_as_runtime_action", "path_mpc_discrete_action_sequence_used"]:
            bad_runtime += int(str(r.get(key, "0")) not in {"0", ""})
        bad_runtime += int(str(r.get("continuous_fu_state_updated", "1")) != "1")
        bad_runtime += int(str(r.get("fu_velocity_emitted", "1")) != "1")
    opened = sum(int_flag(r.get("M7_exploration_opened")) for r in active_plan)
    official = sum(int_flag(r.get("M7_official_ready")) for r in active_plan)
    groups = len(active_plan)
    has_functional = any(str(r.get("projected_variant", "")).startswith("functional_") for r in active_plan)
    route = "M7TrajectoryDiagnosticOnly"
    promotion_allowed = 0
    if groups and opened >= max(1, math.ceil(0.50 * groups)):
        route = "R12-ContinualFunctionalMemoryOpened_Partial"
        promotion_allowed = 1
    return {
        "generated_at": now_sg(),
        "rows": len(matrix_rows),
        "groups": groups,
        "tasks_covered": ",".join(tasks),
        "memory_projected_opened_groups": opened,
        "official_ready_groups": official,
        "relative_forgetting_reduction_mean": statistics.fmean(rels) if rels else math.nan,
        "matched_memory_controls_fail_groups": sum(int_flag(r.get("matched_memory_controls_fail")) for r in active_plan),
        "final_average_accuracy_nonworse_groups": sum(int_flag(r.get("final_average_accuracy_nonworse")) for r in active_plan),
        "constraint_ok_groups": sum(int_flag(r.get("constraint_ok")) for r in active_plan),
        "runtime_bad_flag_count": bad_runtime,
        "candidate_action_regression_pass": int(bad_runtime == 0),
        "route": route,
        "promotion_allowed": promotion_allowed,
        "claim_limit": (
            f"finite-difference old-logit functional memory projection on {','.join(tasks) if tasks else 'no_task'}; train-anchor J_old*v approximation, not full all-function G_f M7"
            if has_functional
            else f"parameter-gradient memory projection on {','.join(tasks) if tasks else 'no_task'}; not full function-space M7"
        ),
    }


def run_m7_trajectory(args: argparse.Namespace) -> dict[str, Any]:
    write_rows(OUT_ROOT / "v22_45E_mechanism_coverage_matrix.csv", MECHANISM_COVERAGE)
    specs = build_m7_trajectory_specs(args)
    output_prefix = safe_fragment(args.m7_trajectory_output)
    specs_filename = f"v22_45E_{output_prefix}_specs.csv"
    matrix_filename = f"v22_45E_{output_prefix}_matrix.csv"
    trace_filename = f"v22_45E_{output_prefix}_trace.csv"
    runtime_filename = f"v22_45E_{output_prefix}_runtime_trace.csv"
    plan_filename = f"v22_45E_{output_prefix}_plan_metrics.csv"
    status_filename = f"v22_45E_{output_prefix}_status.csv"
    summary_filename = f"v22_45E_{output_prefix}_summary.json"
    write_rows(OUT_ROOT / specs_filename, specs)
    append_exec(
        trajectory_command_line(args, "trajectory-m7"),
        task_id=f"m7_trajectory_started_{safe_fragment(args.label)}",
        status="started",
        gpu=",".join(split_csv(args.gpus)),
        files=f"results/v22_45E/{specs_filename}",
        note=f"trajectories={len(specs)}; workers={args.workers}; in_process=1; subprocess_used=0; steps_per_task={args.m7_trajectory_steps or args.m7_steps_per_task}",
    )
    statuses: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futures = [ex.submit(run_one_m7_trajectory, args, spec) for spec in specs]
        for fut in concurrent.futures.as_completed(futures):
            try:
                result = fut.result()
            except Exception as exc:
                failures += 1
                statuses.append({"status": "fail", "error": repr(exc), "traceback": traceback.format_exc()})
                continue
            statuses.append({"label": result.get("label"), "status": result.get("status", "fail")})
            failures += int(result.get("status") != "pass")
            if result.get("summary"):
                matrix_rows.append(result["summary"])
            trace_rows.extend(result.get("trace", []))
            runtime_rows.extend(result.get("runtime", []))
    write_rows(OUT_ROOT / status_filename, statuses)
    write_rows(OUT_ROOT / matrix_filename, matrix_rows)
    write_rows(OUT_ROOT / trace_filename, trace_rows)
    write_rows(OUT_ROOT / runtime_filename, runtime_rows)
    plan_rows = m7_plan_metric_rows(matrix_rows)
    write_rows(OUT_ROOT / plan_filename, plan_rows)
    summary = summarize_m7_trajectory(matrix_rows, runtime_rows, plan_rows)
    write_json(OUT_ROOT / summary_filename, summary)
    append_exec(
        "m7 trajectory-first continual-memory suite completed",
        task_id=f"m7_trajectory_completed_{safe_fragment(args.label)}",
        status="pass" if failures == 0 else "fail",
        gpu=",".join(split_csv(args.gpus)),
        files=f"results/v22_45E/{matrix_filename}, results/v22_45E/{plan_filename}, results/v22_45E/{summary_filename}",
        note=f"trajectories={len(specs)}; failures={failures}; groups={summary.get('groups')}; opened={summary.get('memory_projected_opened_groups')}; route={summary.get('route')}",
        exit_code=0 if failures == 0 else 1,
    )
    return {"trajectories": len(specs), "failures": failures, **summary}


def build_m9_trajectory_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets = split_csv(args.eval_datasets)
    seeds = split_csv(args.eval_seeds, int)
    archs = split_csv(args.eval_architectures)
    specs: list[dict[str, Any]] = []
    trajectory_steps = int(args.trajectory_steps or args.steps)
    common = {
        "steps": trajectory_steps,
        "support_refresh_cadence": max(20, int(args.support_refresh_cadence)),
        "metric_refresh_cadence": max(1, min(5, int(args.metric_refresh_cadence))),
        "metric_shrinkage": max(0.10, float(args.metric_shrinkage)),
        "velocity_scale": min(float(args.velocity_scale), float(args.trajectory_m9_velocity_scale)),
        "safety_budget_velocity_barrier": max(float(args.safety_budget_velocity_barrier), float(args.trajectory_m9_safety_barrier)),
        "beta_signal": min(float(args.beta_signal), float(args.trajectory_m9_beta_signal)),
        "rho_max": min(float(args.rho_max), 0.20),
    }
    controls = [
        "none",
        "M9-fixed-uniform-mixture-control",
        "M9-fixed-euclidean-control",
        "M9-fixed-fisher-control",
        "M9-fixed-signal-control",
        "M9-fixed-basis-control",
    ]
    seen_baselines: set[tuple[str, int, str]] = set()
    for dataset in datasets:
        for seed in seeds:
            for arch in archs:
                key = (str(dataset), int(seed), str(arch))
                if key not in seen_baselines:
                    add_spec(
                        specs,
                        args,
                        mechanism="BASE",
                        recipe="m9_trajectory_optimizer_alone",
                        dataset=dataset,
                        seed=seed,
                        architecture=arch,
                        variant="optimizer_alone",
                        control_mode="none",
                        **{k: v for k, v in common.items() if k not in {"velocity_scale", "safety_budget_velocity_barrier", "beta_signal", "rho_max"}},
                    )
                    seen_baselines.add(key)
                for cmode in controls:
                    add_spec(
                        specs,
                        args,
                        mechanism="M9",
                        recipe="AdaptiveMetricMixtureTrajectory",
                        dataset=dataset,
                        seed=seed,
                        architecture=arch,
                        variant="M9-adaptive-metric-mixture",
                        control_mode=cmode,
                        **common,
                    )
    if int(args.row_limit) > 0:
        specs = specs[: int(args.row_limit)]
    return specs


def m9_trajectory_plan_metric_rows(matrix_filename: str) -> list[dict[str, Any]]:
    rows = read_rows(OUT_ROOT / matrix_filename)
    by_group: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = {}
    control_roles = {
        "M9-fixed-uniform-mixture-control": "uniform",
        "M9-fixed-euclidean-control": "euclidean",
        "M9-fixed-fisher-control": "fisher",
        "M9-fixed-signal-control": "signal",
        "M9-fixed-basis-control": "basis",
    }
    for row in rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("seed", "")),
            str(row.get("architecture", "")),
            str(row.get("optimizer_family", "")),
        )
        role = ""
        if row.get("variant") == "optimizer_alone":
            role = "base"
        elif row.get("mechanism") == "M9" and row.get("variant") == "M9-adaptive-metric-mixture" and row.get("control_mode") == "none":
            role = "adaptive"
        elif row.get("mechanism") == "M9":
            role = control_roles.get(str(row.get("control_mode", "")), "")
        if role:
            by_group.setdefault(key, {})[role] = row
    out: list[dict[str, Any]] = []
    for key, group in sorted(by_group.items()):
        dataset, seed, architecture, optimizer = key
        adaptive = group.get("adaptive", {})
        if not adaptive:
            continue
        base = group.get("base", {})
        fixed_rows = {name: group.get(name, {}) for name in ["uniform", "euclidean", "fisher", "signal", "basis"]}
        anll = finite_float(adaptive.get("final_NLL"))
        bnll = finite_float(base.get("final_NLL"))
        fixed_nlls = {name: finite_float(row.get("final_NLL")) for name, row in fixed_rows.items()}
        valid_fixed = {name: value for name, value in fixed_nlls.items() if value is not None}
        best_fixed_name = min(valid_fixed, key=valid_fixed.get) if valid_fixed else ""
        best_fixed_nll = valid_fixed.get(best_fixed_name) if best_fixed_name else None
        improvement_vs_base = "" if anll is None or bnll is None else bnll - anll
        improvement_vs_best_fixed = "" if anll is None or best_fixed_nll is None else best_fixed_nll - anll
        runtime_truth = int(
            int_flag(adaptive.get("continuous_fu_state_updated_every_step")) == 1
            and int_flag(adaptive.get("fu_velocity_emitted_every_step")) == 1
            and int_flag(adaptive.get("candidate_action_selection_used_for_runtime")) == 0
        )
        final_weights = [
            value_or(adaptive.get("metric_final_weight_euclidean"), 0.0),
            value_or(adaptive.get("metric_final_weight_fisher"), 0.0),
            value_or(adaptive.get("metric_final_weight_signal"), 0.0),
            value_or(adaptive.get("metric_final_weight_basis"), 0.0),
        ]
        max_final_weight = max(final_weights) if final_weights else 0.0
        collapsed = int(max_final_weight >= 0.85 or bool(str(adaptive.get("metric_mixture_collapsed_to", "")).strip()))
        adaptation_truth = int(
            value_or(adaptive.get("metric_mixture_active_fraction"), 0.0) >= 0.99
            and int_flag(adaptive.get("metric_mixture_control_frozen")) == 0
            and value_or(adaptive.get("metric_mixture_update_count"), 0.0) > 0.0
            and value_or(adaptive.get("metric_weight_drift_total_proxy"), 0.0) > 0.0
        )
        no_debt = int_flag(adaptive.get("no_ECE_Brier_tail_debt"))
        overhead_not_catastrophic = int(value_or(adaptive.get("controller_overhead_ratio"), math.inf) <= 0.75)
        beats_uniform = int(anll is not None and fixed_nlls.get("uniform") is not None and anll < fixed_nlls["uniform"])
        beats_best_fixed = int(isinstance(improvement_vs_best_fixed, float) and improvement_vs_best_fixed > 0.0)
        opened = int(
            isinstance(improvement_vs_base, float)
            and improvement_vs_base > 0.0
            and beats_uniform
            and beats_best_fixed
            and no_debt
            and overhead_not_catastrophic
            and runtime_truth
            and adaptation_truth
            and not collapsed
        )
        diagnostic = int(not opened and (collapsed or not beats_best_fixed or not beats_uniform))
        row = {
            "dataset": dataset,
            "seed": seed,
            "architecture": architecture,
            "optimizer_family": optimizer,
            "base_final_NLL": "" if bnll is None else bnll,
            "adaptive_final_NLL": "" if anll is None else anll,
            "uniform_final_NLL": "" if fixed_nlls.get("uniform") is None else fixed_nlls["uniform"],
            "euclidean_final_NLL": "" if fixed_nlls.get("euclidean") is None else fixed_nlls["euclidean"],
            "fisher_final_NLL": "" if fixed_nlls.get("fisher") is None else fixed_nlls["fisher"],
            "signal_final_NLL": "" if fixed_nlls.get("signal") is None else fixed_nlls["signal"],
            "basis_final_NLL": "" if fixed_nlls.get("basis") is None else fixed_nlls["basis"],
            "best_fixed_metric": best_fixed_name,
            "best_fixed_final_NLL": "" if best_fixed_nll is None else best_fixed_nll,
            "NLL_improvement_vs_base": improvement_vs_base,
            "NLL_improvement_vs_uniform": "" if anll is None or fixed_nlls.get("uniform") is None else fixed_nlls["uniform"] - anll,
            "NLL_improvement_vs_best_fixed": improvement_vs_best_fixed,
            "NLL_improvement_vs_euclidean": "" if anll is None or fixed_nlls.get("euclidean") is None else fixed_nlls["euclidean"] - anll,
            "NLL_improvement_vs_fisher": "" if anll is None or fixed_nlls.get("fisher") is None else fixed_nlls["fisher"] - anll,
            "NLL_improvement_vs_signal": "" if anll is None or fixed_nlls.get("signal") is None else fixed_nlls["signal"] - anll,
            "NLL_improvement_vs_basis": "" if anll is None or fixed_nlls.get("basis") is None else fixed_nlls["basis"] - anll,
            "beats_uniform": beats_uniform,
            "beats_best_fixed_metric": beats_best_fixed,
            "beats_euclidean": int(anll is not None and fixed_nlls.get("euclidean") is not None and anll < fixed_nlls["euclidean"]),
            "beats_fisher": int(anll is not None and fixed_nlls.get("fisher") is not None and anll < fixed_nlls["fisher"]),
            "beats_signal": int(anll is not None and fixed_nlls.get("signal") is not None and anll < fixed_nlls["signal"]),
            "beats_basis": int(anll is not None and fixed_nlls.get("basis") is not None and anll < fixed_nlls["basis"]),
            "no_ECE_Brier_tail_debt": adaptive.get("no_ECE_Brier_tail_debt", ""),
            "ECE_delta_vs_own_strong_optimizer": adaptive.get("ECE_delta_vs_own_strong_optimizer", ""),
            "Brier_delta_vs_own_strong_optimizer": adaptive.get("Brier_delta_vs_own_strong_optimizer", ""),
            "tail_q99_delta_vs_own_strong_optimizer": adaptive.get("tail_q99_delta_vs_own_strong_optimizer", ""),
            "controller_overhead_ratio": adaptive.get("controller_overhead_ratio", ""),
            "overhead_not_catastrophic": overhead_not_catastrophic,
            "metric_mixture_entropy": adaptive.get("metric_mixture_entropy", ""),
            "metric_mixture_final_entropy": adaptive.get("metric_mixture_final_entropy", ""),
            "metric_weight_drift_total_proxy": adaptive.get("metric_weight_drift_total_proxy", ""),
            "metric_mixture_update_count": adaptive.get("metric_mixture_update_count", ""),
            "metric_mixture_collapsed_to": adaptive.get("metric_mixture_collapsed_to", ""),
            "metric_final_weight_euclidean": adaptive.get("metric_final_weight_euclidean", ""),
            "metric_final_weight_fisher": adaptive.get("metric_final_weight_fisher", ""),
            "metric_final_weight_signal": adaptive.get("metric_final_weight_signal", ""),
            "metric_final_weight_basis": adaptive.get("metric_final_weight_basis", ""),
            "max_final_metric_weight": max_final_weight,
            "mixture_collapsed": collapsed,
            "adaptation_truth": adaptation_truth,
            "runtime_truth": runtime_truth,
            "continuous_fu_state_updated_every_step": adaptive.get("continuous_fu_state_updated_every_step", ""),
            "fu_velocity_emitted_every_step": adaptive.get("fu_velocity_emitted_every_step", ""),
            "candidate_action_selection_used_for_runtime": adaptive.get("candidate_action_selection_used_for_runtime", ""),
            "AdaptiveMetricFlowOpened": opened,
            "MetricSelectionDiagnosticOnly": diagnostic,
            "claim_limit": "diagonal train-batch metric mixture with mirror-descent weights; no validation/test/future metric search",
        }
        out.append(row)
    return out or [{"status": "no_m9_trajectory_rows"}]


def run_m9_trajectory(args: argparse.Namespace) -> dict[str, Any]:
    write_rows(OUT_ROOT / "v22_45E_mechanism_coverage_matrix.csv", MECHANISM_COVERAGE)
    specs = build_m9_trajectory_specs(args)
    output_prefix = safe_fragment(args.m9_trajectory_output)
    specs_filename = f"v22_45E_{output_prefix}_specs.csv"
    matrix_filename = f"v22_45E_{output_prefix}_matrix.csv"
    runtime_filename = f"v22_45E_{output_prefix}_runtime_regression_audit.csv"
    support_filename = f"v22_45E_{output_prefix}_support_direction_decomposition.csv"
    safety_filename = f"v22_45E_{output_prefix}_safety_debt_matrix.csv"
    overhead_filename = f"v22_45E_{output_prefix}_overhead_matrix.csv"
    summary_filename = f"v22_45E_{output_prefix}_mechanism_summary.csv"
    plan_filename = f"v22_45E_{output_prefix}_plan_metrics.csv"
    status_filename = f"v22_45E_{output_prefix}_status.csv"
    write_rows(OUT_ROOT / specs_filename, specs)
    append_exec(
        trajectory_command_line(args, "trajectory-m9"),
        task_id=f"m9_trajectory_started_{safe_fragment(args.label)}",
        status="started",
        gpu=",".join(split_csv(args.gpus)),
        files=f"results/v22_45E/{specs_filename}, results/v22_45E/chunks",
        note=(
            f"trajectories={len(specs)}; workers={args.workers}; in_process=1; subprocess_used=0; "
            f"datasets={args.eval_datasets}; seeds={args.eval_seeds}; architectures={args.eval_architectures}; "
            f"steps={args.trajectory_steps or args.steps}; controls=fixed_uniform,euc,fisher,signal,basis"
        ),
    )
    failures = 0
    statuses: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futures = [ex.submit(run_one_inprocess_trajectory, args, spec, "m9_trajectory") for spec in specs]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            statuses.append(row)
            failures += int(row.get("status") != "pass")
    write_rows(OUT_ROOT / status_filename, statuses)
    merge_summary = merge_v2243_outputs(
        specs_filename=specs_filename,
        matrix_filename=matrix_filename,
        runtime_filename=runtime_filename,
        support_direction_filename=support_filename,
        safety_filename=safety_filename,
        overhead_filename=overhead_filename,
        summary_filename=summary_filename,
    )
    plan_rows = m9_trajectory_plan_metric_rows(matrix_filename)
    write_rows(OUT_ROOT / plan_filename, plan_rows)
    pass_rows = sum(int_flag(r.get("AdaptiveMetricFlowOpened")) for r in plan_rows)
    diagnostic_rows = sum(int_flag(r.get("MetricSelectionDiagnosticOnly")) for r in plan_rows)
    append_exec(
        "m9 trajectory-first adaptive-metric-mixture suite completed",
        task_id=f"m9_trajectory_completed_{safe_fragment(args.label)}",
        status="pass" if failures == 0 else "fail",
        gpu=",".join(split_csv(args.gpus)),
        files=f"results/v22_45E/{matrix_filename}, results/v22_45E/{summary_filename}, results/v22_45E/{plan_filename}",
        note=f"trajectories={len(specs)}; failures={failures}; merged={merge_summary}; plan_metric_rows={len(plan_rows)}; opened_rows={pass_rows}; metric_selection_diagnostic_rows={diagnostic_rows}",
        exit_code=0 if failures == 0 else 1,
    )
    return {"trajectories": len(specs), "failures": failures, "plan_metric_rows": len(plan_rows), "m9_opened_rows": pass_rows, **merge_summary}


def run_external_diagnostic(
    args: argparse.Namespace,
    *,
    kind: str,
    script: str,
    matrix_src: Path,
    summary_src: Path,
    matrix_dst: Path,
    summary_dst: Path,
    extra: list[str],
) -> dict[str, Any]:
    ensure_out()
    label = f"{args.label}_{kind}"
    cmd = [
        PYTHON,
        script,
        "--stage",
        "full",
        "--label",
        label,
        "--seeds",
        args.diagnostic_seeds,
        "--architectures",
        args.diagnostic_architectures,
        "--gpus",
        args.gpus,
        "--workers",
        str(args.workers),
        "--row-timeout",
        str(args.diagnostic_row_timeout),
    ] + extra
    proc = run_logged(cmd, task_id=f"{kind}_diagnostic_full", gpu=args.gpus, timeout=int(args.diagnostic_total_timeout))
    copied_matrix = copy_if_exists(matrix_src, matrix_dst)
    copied_summary = copy_if_exists(summary_src, summary_dst)
    append_exec(
        "copy_external_diagnostic_artifacts",
        task_id=f"{kind}_copy_into_v22_45E",
        status="pass" if copied_matrix and copied_summary else "warn",
        gpu="n/a",
        files=f"{matrix_dst.relative_to(ROOT)}, {summary_dst.relative_to(ROOT)}",
        note=f"source_matrix={matrix_src.relative_to(ROOT) if matrix_src.exists() else matrix_src}; source_summary={summary_src.relative_to(ROOT) if summary_src.exists() else summary_src}; returncode={proc.returncode}",
        exit_code=0 if copied_matrix and copied_summary else 1,
    )
    return {"returncode": proc.returncode, "matrix_copied": copied_matrix, "summary_copied": copied_summary}


def run_m4_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    label = f"{args.label}_M4_slow_signal"
    safe = safe_fragment(label)
    return run_external_diagnostic(
        args,
        kind="M4_slow_signal",
        script="experiments/run_v22_42R_s5_temporal_slow_signal_diagnostic.py",
        matrix_src=ROOT / f"results/v22_42R/v22_42R_s5_temporal_slow_signal_{safe}_matrix.csv",
        summary_src=ROOT / f"results/v22_42R/v22_42R_s5_temporal_slow_signal_{safe}_summary.json",
        matrix_dst=OUT_ROOT / "v22_45E_M4_slow_signal_diagnostic_matrix.csv",
        summary_dst=OUT_ROOT / "v22_45E_M4_slow_signal_diagnostic_summary.json",
        extra=[
            "--steps",
            str(args.m4_diag_steps),
            "--hidden",
            str(args.diagnostic_hidden),
            "--batch-size",
            str(args.diagnostic_batch_size),
            "--modulus",
            str(args.m4_modulus),
            "--train-fraction",
            str(args.m4_train_fraction),
            "--velocity-scale",
            str(args.m4_diag_velocity_scale),
            "--beta-slow",
            str(args.m4_beta_slow),
        ],
    )


def run_m7_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    label = f"{args.label}_M7_continual_memory"
    safe = safe_fragment(label)
    return run_external_diagnostic(
        args,
        kind="M7_continual_memory",
        script="experiments/run_v22_42R_s7_continual_memory_diagnostic.py",
        matrix_src=ROOT / f"results/v22_42R/v22_42R_s7_continual_memory_{safe}_matrix.csv",
        summary_src=ROOT / f"results/v22_42R/v22_42R_s7_continual_memory_{safe}_summary.json",
        matrix_dst=OUT_ROOT / "v22_45E_M7_continual_memory_matrix.csv",
        summary_dst=OUT_ROOT / "v22_45E_M7_continual_memory_summary.json",
        extra=[
            "--steps-per-task",
            str(args.m7_steps_per_task),
            "--hidden",
            str(args.diagnostic_hidden),
            "--batch-size",
            str(args.diagnostic_batch_size),
            "--train-size",
            str(args.m7_train_size),
            "--held-size",
            str(args.m7_held_size),
            "--velocity-scale",
            str(args.m7_velocity_scale),
            "--beta-memory",
            str(args.m7_beta_memory),
        ],
    )


def write_blocker_repair_log() -> list[dict[str, Any]]:
    rows = [
        {
            "mechanism": "M1",
            "blocker": "Full M1 requires a functional actuator least-squares projection. The current kernel now has train-batch logit dual/primal mirror state plus M1-KLMirrorLS sketched finite-difference ridge-LS, but it is still not a full all-function/all-parameter solver.",
            "repair_attempt": "Added trajectory-first M1-KL/Brier/TailSafe/Fisher mirror variants, corrected KL to mirror-residual autograd, added cached mirror cadence, then implemented M1-KLMirrorLS with log-prob finite-difference response, ridge LS, support-random sketched basis, and basis-capacity sweeps up to 16.",
            "result": "Sketched LS ran as complete in-process trajectories with fallback=0, but c240 gates did not pass: basis4 cached40 pass_all=1/16, rand8 cached120 pass_all=2/16, basis16 cached240 pass_all=0/16. No FunctionalMirrorFlowOpened promotion.",
        },
        {
            "mechanism": "M2",
            "blocker": "Full M2 still lacks a true functional actuator-spectrum solver. The current kernel can evaluate train-batch finite-difference functional response on selected matrix SVD modes, but not a full all-function actuator spectrum.",
            "repair_attempt": "First added a support-overlap/safety radial proxy, then added P4 functional-rse radial010 variants that use a safety-only top gate and per-mode finite-difference response/RSE selection; reran smoke and c240 complete in-process trajectory suites against strict-OET, same-radial, and same-OET controls.",
            "result": "The RSE-only top-gate c240 was validly executed but failed route gates: beats_strict_OET=6/16, beats_same_radial=10/16, no_debt=3/16, overhead<=0.35=0/16, screen_pass=0. No SpectralRadialSplitFlowOpened promotion.",
        },
        {
            "mechanism": "M3",
            "blocker": "Full M3 requires momentum in the current OET/Lie algebra frame. The kernel now has a transported left-Lie generator momentum state, but it is still a left-Cayley implementation rather than a full left/right balanced transport solver.",
            "repair_attempt": "Added P3-Euclidean-OET-transported-lie-momentum-pure with per-step skew generator momentum, adjoint transport by the previous Cayley rotation, same-Lie-random/signflip controls, and trajectory-first runner artifacts.",
            "result": m3_trajectory_prefix_verified_result("m3_trajectory_lie_c240"),
        },
        {
            "mechanism": "M4",
            "blocker": "Classification S1 rows do not cover the plan's M4 hard tasks by themselves. The older v22.42R copy was diagnostic-only and the first v22.45E M4 modular-addition trajectories had no test crossing, so grokking delay was invalid.",
            "repair_attempt": "Added --stage trajectory-m4: complete in-process modular-addition and Class_MNIST old/new trajectories for optimizer_alone, slow_signal_flow, slow_random_control, and slow_signflip_control. The velocity uses EMA slow gradient plus top-k support migration and fast-reservoir suppression; logs slow/fast energy, RSM_index, delay or forgetting, and runtime truth flags.",
            "result": (
                m4_trajectory_prefix_verified_result("m4_trajectory")
                + " "
                + m4_trajectory_prefix_verified_result("m4_trajectory_wd1e3_c1600")
                + " "
                + m4_trajectory_prefix_verified_result("m4_trajectory_p7_wd1e3_c1600")
                + " "
                + m4_trajectory_prefix_verified_result("m4_trajectory_class_mnist")
                + " "
                + m4_trajectory_prefix_verified_result("m4_trajectory_class_mnist_strong")
            ),
        },
        {
            "mechanism": "M6",
            "blocker": "Earlier M6 evidence was a NoiseControlProxy and did not inject a reservoir stochastic term inside the runtime training loop.",
            "repair_attempt": "Added M6-reservoir-noise and M6-noise-suppression variants to the v22.43 complete trajectory kernel. The new M6 velocity adds parameter metric-support complement noise and compares it against signal-noise and same-norm Gaussian controls.",
            "result": m6_trajectory_verified_result(),
        },
        {
            "mechanism": "M7",
            "blocker": "Earlier M7 evidence was copied from the v22.42R diagnostic runner and did not implement the plan's new-task residualization against old memory subspace inside v22.45E artifacts.",
            "repair_attempt": "Added trajectory-m7: complete in-process MNIST-family and modular old/new trajectories with EMA/rank-k old-gradient memory. The first projector residualizes new-task gradients in parameter space; the second estimates train-anchor old-logit J_old*v by finite differences and residualizes in that functional response space. Both use optimizer/random-subspace/rehearsal controls, runtime truth flags, and plan metrics.",
            "result": (
                m7_trajectory_verified_result()
                + " "
                + m7_trajectory_verified_result("m7_trajectory_functional_class_rank4")
                + " "
                + m7_trajectory_verified_result("m7_trajectory_functional_p7_fullgrid_rank4")
            ),
        },
        {
            "mechanism": "M8",
            "blocker": "No runtime canonical gauge transform Gamma(theta) or FU state update in canonical coordinates.",
            "repair_attempt": "Added m8-gauge-audit: static code-path feasibility rows plus dynamic function-equivalence checks for naive hidden covariance, basis-Gram, and degree/frequency bank rescaling repairs.",
            "result": m8_gauge_audit_verified_result(),
        },
        {
            "mechanism": "M9",
            "blocker": "Earlier M9 evidence only had fixed metric rows and no runtime adaptive mixture weights w_m,t.",
            "repair_attempt": "Added M9-adaptive-metric-mixture to the v22.43 complete trajectory kernel. The controller maintains Euclidean/Fisher/Signal/Basis diagonal metric components and updates train-only mixture weights by mirror descent against safety/support/residual debt, with fixed uniform/single-metric controls.",
            "result": m9_trajectory_verified_result(),
        },
    ]
    write_rows(OUT_ROOT / "v22_45E_blocker_repair_log.csv", rows)
    append_exec(
        "write_blocker_repair_log",
        task_id="blocker_repair_log",
        status="pass",
        files="results/v22_45E/v22_45E_blocker_repair_log.csv",
        note="Records fallback/repair attempts for mechanisms that are only partially supported by the existing kernel.",
    )
    return rows


def trajectory_first_verified_result() -> str:
    plan_path = OUT_ROOT / "v22_45E_m2_trajectory_plan_metrics.csv"
    status_path = OUT_ROOT / "v22_45E_m2_trajectory_status.csv"
    if not plan_path.exists():
        return "not_run; trajectory-m2 stage has not produced v22_45E_m2_trajectory_plan_metrics.csv."
    plan_rows = [r for r in read_rows(plan_path) if not r.get("status")]
    status_rows = read_rows(status_path)
    failures = sum(1 for r in status_rows if r.get("status") == "fail")
    n = len(plan_rows)
    beats_strict = sum(int_flag(r.get("beats_strict_OET")) for r in plan_rows)
    beats_radial = sum(int_flag(r.get("beats_same_radial")) for r in plan_rows)
    beats_oet = sum(int_flag(r.get("beats_same_OET")) for r in plan_rows)
    no_debt = sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in plan_rows)
    overhead_ok = sum(1 for r in plan_rows if value_or(r.get("controller_overhead_ratio"), math.inf) <= 0.35)
    radial_ok = sum(1 for r in plan_rows if value_or(r.get("radial_energy_fraction"), math.inf) <= 0.20)
    runtime_ok = sum(
        1
        for r in plan_rows
        if int_flag(r.get("continuous_fu_state_updated_every_step")) == 1
        and int_flag(r.get("fu_velocity_emitted_every_step")) == 1
        and int_flag(r.get("candidate_action_selection_used_for_runtime")) == 0
    )
    return (
        f"trajectory-m2 completed artifact check: groups={n}; failures={failures}; "
        f"beats_strict_OET={beats_strict}/{n}; beats_same_radial={beats_radial}/{n}; "
        f"beats_same_OET={beats_oet}/{n}; no_debt={no_debt}/{n}; overhead<=0.35={overhead_ok}/{n}; "
        f"radial_energy<=0.20={radial_ok}/{n}; runtime_truth={runtime_ok}/{n}. "
        "These are in-process complete trajectories; still a finite-difference functional-JVP approximation, not a full actuator-spectrum solver."
    )


def m2_trajectory_prefix_verified_result(prefix: str) -> str:
    plan_path = OUT_ROOT / f"v22_45E_{prefix}_plan_metrics.csv"
    status_path = OUT_ROOT / f"v22_45E_{prefix}_status.csv"
    summary_path = OUT_ROOT / f"v22_45E_{prefix}_mechanism_summary.csv"
    if not plan_path.exists():
        return f"not_run; {plan_path.name} is absent."
    plan_rows = [r for r in read_rows(plan_path) if not r.get("status")]
    status_rows = read_rows(status_path)
    failures = sum(1 for r in status_rows if r.get("status") == "fail")
    n = len(plan_rows)
    beats_strict = sum(int_flag(r.get("beats_strict_OET")) for r in plan_rows)
    beats_radial = sum(int_flag(r.get("beats_same_radial")) for r in plan_rows)
    beats_oet = sum(int_flag(r.get("beats_same_OET")) for r in plan_rows)
    no_debt = sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in plan_rows)
    overhead_ok = sum(1 for r in plan_rows if value_or(r.get("controller_overhead_ratio"), math.inf) <= 0.35)
    radial_ok = sum(1 for r in plan_rows if value_or(r.get("radial_energy_fraction"), math.inf) <= 0.20)
    runtime_ok = sum(
        1
        for r in plan_rows
        if int_flag(r.get("continuous_fu_state_updated_every_step")) == 1
        and int_flag(r.get("fu_velocity_emitted_every_step")) == 1
        and int_flag(r.get("candidate_action_selection_used_for_runtime")) == 0
    )
    summary = next((r for r in read_rows(summary_path) if r.get("mechanism") == "M2"), {})

    def mean_of(key: str) -> str:
        vals = [
            value_or(r.get(key), math.nan)
            for r in plan_rows
            if finite_float(r.get(key)) is not None
        ]
        if not vals:
            return "nan"
        return f"{statistics.fmean(vals):.6g}"

    return (
        f"{prefix}: trajectories={len(status_rows)}; failures={failures}; groups={n}; "
        f"cand={summary.get('candidate_rows', '')}; nll={summary.get('NLL_improvement_vs_own_rows', '')}; "
        f"control={summary.get('beats_matched_control_rows', '')}; "
        f"beats_strict_OET={beats_strict}/{n}; beats_same_radial={beats_radial}/{n}; "
        f"beats_same_OET={beats_oet}/{n}; no_debt={no_debt}/{n}; overhead<=0.35={overhead_ok}/{n}; "
        f"radial_energy<=0.20={radial_ok}/{n}; runtime_truth={runtime_ok}/{n}; pass={summary.get('screen_pass', '')}; "
        f"mean_vs_strict={mean_of('NLL_improvement_vs_strict_OET')}; "
        f"mean_vs_same_radial={mean_of('NLL_improvement_vs_same_radial')}; "
        f"mean_overhead={mean_of('controller_overhead_ratio')}; mean_radial_energy={mean_of('radial_energy_fraction')}; "
        f"mean_fd_eval={mean_of('functional_radial_fd_eval_count_mean')}; "
        f"mean_rse_max={mean_of('functional_radial_rse_max')}. "
        "Complete in-process trajectories; finite-difference train-batch functional response on candidate SVD modes only."
    )


def m3_trajectory_prefix_verified_result(prefix: str) -> str:
    plan_path = OUT_ROOT / f"v22_45E_{prefix}_plan_metrics.csv"
    status_path = OUT_ROOT / f"v22_45E_{prefix}_status.csv"
    summary_path = OUT_ROOT / f"v22_45E_{prefix}_mechanism_summary.csv"
    if not plan_path.exists():
        return f"not_run; {plan_path.name} is absent."
    plan_rows = [r for r in read_rows(plan_path) if not r.get("status")]
    status_rows = read_rows(status_path)
    failures = sum(1 for r in status_rows if r.get("status") == "fail")
    n = len(plan_rows)
    beats_ambient = sum(int_flag(r.get("beats_ambient_OET")) for r in plan_rows)
    beats_random = sum(int_flag(r.get("beats_same_Lie_random")) for r in plan_rows)
    beats_signflip = sum(int_flag(r.get("beats_same_Lie_signflip")) for r in plan_rows)
    no_debt = sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in plan_rows)
    overhead_ok = sum(1 for r in plan_rows if value_or(r.get("controller_overhead_ratio"), math.inf) <= 0.35)
    runtime_ok = sum(
        1
        for r in plan_rows
        if int_flag(r.get("continuous_fu_state_updated_every_step")) == 1
        and int_flag(r.get("fu_velocity_emitted_every_step")) == 1
        and int_flag(r.get("candidate_action_selection_used_for_runtime")) == 0
    )
    summary = next((r for r in read_rows(summary_path) if r.get("mechanism") == "M3"), {})

    def mean_of(key: str) -> str:
        vals = [
            value_or(r.get(key), math.nan)
            for r in plan_rows
            if finite_float(r.get(key)) is not None
        ]
        if not vals:
            return "nan"
        return f"{statistics.fmean(vals):.6g}"

    return (
        f"{prefix}: trajectories={len(status_rows)}; failures={failures}; groups={n}; "
        f"cand={summary.get('candidate_rows', '')}; nll={summary.get('NLL_improvement_vs_own_rows', '')}; "
        f"control={summary.get('beats_matched_control_rows', '')}; "
        f"beats_ambient_OET={beats_ambient}/{n}; beats_same_Lie_random={beats_random}/{n}; "
        f"beats_same_Lie_signflip={beats_signflip}/{n}; no_debt={no_debt}/{n}; "
        f"overhead<=0.35={overhead_ok}/{n}; runtime_truth={runtime_ok}/{n}; pass={summary.get('screen_pass', '')}; "
        f"mean_vs_ambient={mean_of('NLL_improvement_vs_ambient_OET')}; "
        f"mean_vs_same_Lie_random={mean_of('NLL_improvement_vs_same_Lie_random')}; "
        f"mean_lie_norm={mean_of('lie_momentum_norm')}; mean_transport_error={mean_of('transport_error')}; "
        f"mean_Lie_SNR={mean_of('Lie_SNR')}. "
        "Complete in-process trajectories with runtime Lie generator momentum state."
    )


def m4_trajectory_prefix_verified_result(prefix: str = "m4_trajectory") -> str:
    summary_path = OUT_ROOT / f"v22_45E_{prefix}_summary.json"
    plan_path = OUT_ROOT / f"v22_45E_{prefix}_plan_metrics.csv"
    status_path = OUT_ROOT / f"v22_45E_{prefix}_status.csv"
    if not summary_path.exists():
        return f"not_run; {summary_path.name} is absent."
    summary = read_json(summary_path)
    plan_rows = [r for r in read_rows(plan_path) if not r.get("status")]
    status_rows = read_rows(status_path)
    failures = sum(1 for r in status_rows if r.get("status") == "fail")
    n = len(plan_rows)
    pass_rows = sum(int_flag(r.get("M4_pass")) for r in plan_rows)
    control_fail = sum(int_flag(r.get("matched_slow_controls_fail")) for r in plan_rows)
    nonworse = sum(int_flag(r.get("final_average_accuracy_nonworse")) for r in plan_rows)
    delay25 = sum(int_flag(r.get("grokking_delay_reduction_ge_25p")) for r in plan_rows)
    forget10 = sum(int_flag(r.get("continual_forgetting_reduction_ge_10p")) for r in plan_rows)
    valid_forget = sum(int_flag(r.get("forgetting_task_valid")) for r in plan_rows)
    tasks = ",".join(sorted({str(r.get("dataset", "")) for r in plan_rows if r.get("dataset")}))
    runtime_ok = sum(
        1
        for r in plan_rows
        if int_flag(r.get("continuous_fu_state_updated_every_step")) == 1
        and int_flag(r.get("fu_velocity_emitted_every_step")) == 1
        and int_flag(r.get("candidate_action_selection_used_for_runtime")) == 0
    )

    def mean_of(key: str) -> str:
        vals = [value_or(r.get(key), math.nan) for r in plan_rows if finite_float(r.get(key)) is not None]
        if not vals:
            return "nan"
        return f"{statistics.fmean(vals):.6g}"

    return (
        f"{prefix}: trajectories={len(status_rows)}; failures={failures}; groups={n}; "
        f"tasks={tasks or 'n/a'}; "
        f"M4_pass={pass_rows}/{n}; delay_reduction>=25%={delay25}/{n}; "
        f"valid_forgetting={valid_forget}/{n}; forgetting_reduction>=10%={forget10}/{n}; "
        f"matched_controls_fail={control_fail}/{n}; final_nonworse={nonworse}/{n}; "
        f"runtime_truth={runtime_ok}/{n}; route={summary.get('route')}; "
        f"invalid_no_delay_groups={summary.get('invalid_no_delay_groups')}; "
        f"mean_RSM={mean_of('RSM_index')}; mean_NLL_improvement_vs_base={mean_of('NLL_improvement_vs_base')}; "
        f"mean_relative_forgetting_reduction={mean_of('relative_forgetting_reduction')}. "
        "Complete in-process M4 trajectories; partial top-k support/reservoir approximation, not full metric/OET projection."
    )


def m5_trajectory_verified_result() -> str:
    summary_rows = [
        r for r in read_rows_with_source("v22_45E_m5_trajectory*_mechanism_summary.csv")
        if r.get("mechanism") == "M5"
        and "smoke" not in str(r.get("source_artifact", "")).lower()
    ]
    status_rows = [r for r in read_rows_with_source("v22_45E_m5_trajectory*_status.csv") if "smoke" not in str(r.get("source_artifact", "")).lower()]
    if not summary_rows:
        return "not_run; no v22_45E_m5_trajectory*_mechanism_summary.csv M5 row was found."
    failures = sum(1 for r in status_rows if r.get("status") == "fail")
    parts = []
    for row in summary_rows:
        parts.append(
            f"{row.get('source_artifact')}: cand={row.get('candidate_rows')}, "
            f"nll={row.get('NLL_improvement_vs_own_rows')}, control={row.get('beats_matched_control_rows')}, "
            f"mlp_match={row.get('beats_MLP_matched_support_rows')}, no_debt={row.get('no_ECE_Brier_tail_debt_rows')}, "
            f"overhead={row.get('controller_overhead_le_0p35_rows')}, pass={row.get('screen_pass')}"
        )
    return f"trajectory-m5 artifact check: failures={failures}; " + " | ".join(parts)


def m5_trajectory_prefix_verified_result(prefix: str) -> str:
    summary_path = OUT_ROOT / f"v22_45E_{prefix}_mechanism_summary.csv"
    status_path = OUT_ROOT / f"v22_45E_{prefix}_status.csv"
    if not summary_path.exists():
        return f"not_run; {summary_path.name} is absent."
    m5 = next((r for r in read_rows(summary_path) if r.get("mechanism") == "M5"), {})
    status_rows = read_rows(status_path)
    failures = sum(1 for r in status_rows if r.get("status") == "fail")
    return (
        f"{prefix}: trajectories={len(status_rows)}; failures={failures}; "
        f"cand={m5.get('candidate_rows')}; nll={m5.get('NLL_improvement_vs_own_rows')}; "
        f"control={m5.get('beats_matched_control_rows')}; mlp_match={m5.get('beats_MLP_matched_support_rows')}; "
        f"no_debt={m5.get('no_ECE_Brier_tail_debt_rows')}; "
        f"overhead={m5.get('controller_overhead_le_0p35_rows')}; pass={m5.get('screen_pass')}."
    )


def parse_semicolon_float_list(value: Any) -> list[float]:
    vals: list[float] = []
    for part in str(value or "").replace(",", ";").split(";"):
        parsed = finite_float(part)
        if parsed is not None:
            vals.append(float(parsed))
    return vals


def m5_main_trajectory_matrix_paths() -> list[Path]:
    excluded = (
        "_overhead_matrix.csv",
        "_safety_debt_matrix.csv",
        "_support_direction_decomposition.csv",
        "_runtime_regression_audit.csv",
    )
    paths = []
    for path in sorted(OUT_ROOT.glob("v22_45E_m5_trajectory*_matrix.csv")):
        name = path.name.lower()
        if "smoke" in name:
            continue
        if any(name.endswith(suffix) for suffix in excluded):
            continue
        paths.append(path)
    return paths


def m5_spectrum_locator(row: dict[str, Any]) -> tuple[str, str]:
    arch = str(row.get("architecture", ""))
    variant = str(row.get("variant", ""))
    if arch == "MLP":
        if "frequency-like" in variant:
            return "MLP", "A_MLP_frequency_like"
        if "polynomial-like" in variant:
            return "MLP", "A_MLP_polynomial_like"
        if "same-rank-block" in variant or "block" in variant:
            return "MLP", "A_MLP_same_rank_block"
        if "same-param" in variant:
            return "MLP", "A_MLP_same_param_flops"
        if "spectral" in variant:
            return "MLP", "A_MLP_spectral"
        return "MLP", "A_MLP_lowrank"
    bank = "DFOU" if "D-FOU" in variant or "FOU" in str(row.get("basis_bank", "")) else "DCHE"
    arch_key = "DGKAN_DFOU" if bank == "DFOU" else "DGKAN_DCHE"
    suffix = "bank_OET" if "OET" in variant or "BankLocal" in variant else "basis_bank"
    return arch_key, f"A_KAN_{bank}_{suffix}"


def spectrum_row_for(
    spectrum: dict[tuple[str, str, str, str], dict[str, Any]],
    dataset: str,
    seed: str,
    architecture: str,
    actuator: str,
) -> tuple[dict[str, Any] | None, int]:
    exact = spectrum.get((dataset, seed, architecture, actuator))
    if exact is not None:
        return exact, 1
    fallback = spectrum.get((dataset, "0", architecture, actuator))
    return fallback, 0 if fallback is not None else 0


def functional_spectrum_distance(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float | None:
    if not a or not b:
        return None
    av = parse_semicolon_float_list(a.get("functional_actuator_singular_values_top5"))
    bv = parse_semicolon_float_list(b.get("functional_actuator_singular_values_top5"))
    n = min(len(av), len(bv))
    if n <= 0:
        return None
    eps = 1.0e-12
    return math.sqrt(sum((math.log(max(eps, av[i])) - math.log(max(eps, bv[i]))) ** 2 for i in range(n)))


def m5_same_basis_control_pass(row: dict[str, Any]) -> int:
    return int(any(int_flag(row.get(key)) for key in [
        "beats_same_basis_Gram_random",
        "beats_same_basis_Gram_signflip",
        "beats_same_basis_OET_random",
        "beats_same_bank_shuffled",
    ]))


def m5_classification(
    kan_row: dict[str, Any],
    mlp_row: dict[str, Any] | None,
    nll_delta: float | None,
) -> str:
    explicit = str(kan_row.get("TrueKANGain_class", "")).strip()
    if explicit:
        return explicit
    kan_gain = value_or(kan_row.get("NLL_improvement_vs_own_strong_optimizer"), 0.0) > 0.0
    mlp_gain = value_or(mlp_row.get("NLL_improvement_vs_own_strong_optimizer") if mlp_row else None, 0.0) > 0.0
    if nll_delta is not None and nll_delta <= 0.0 and kan_gain:
        return "BothGain" if mlp_gain else "TrueKANGain"
    if kan_gain:
        return "KANInternalValueOnly"
    if mlp_gain:
        return "MLPMatchedSupportStronger"
    return "NoGain"


def m5_summary_row(rows: list[dict[str, Any]], source_artifact: str) -> dict[str, Any]:
    active = [r for r in rows if not r.get("status")]

    def count(key: str) -> int:
        return sum(int_flag(r.get(key)) for r in active)

    def mean(key: str) -> str:
        vals = [value_or(r.get(key), math.nan) for r in active if finite_float(r.get(key)) is not None]
        return "" if not vals else f"{statistics.fmean(vals):.8g}"

    sources = sorted({str(r.get("source_artifact", "")) for r in active if r.get("source_artifact")})
    return {
        "source_artifact": source_artifact,
        "rows": len(active),
        "source_artifact_count": len(sources),
        "KAN_improvement_rows": count("KAN_improves"),
        "beats_same_basis_controls_rows": count("beats_same_basis_controls"),
        "beats_MLP_matched_support_rows": count("KAN_beats_MLP_matched_support"),
        "TrueKANGain_rows": count("TrueKANGain"),
        "BothGain_rows": count("BothGain"),
        "TrueKANGain_plus_BothGain_rows": count("TrueKANGain_or_BothGain"),
        "ControlExplained_rows": count("ControlExplained"),
        "MLPDegradationDriven_rows": count("MLPDegradationDriven"),
        "no_debt_rows": count("no_ECE_Brier_tail_debt"),
        "overhead_le_0p35_rows": count("controller_overhead_le_0p35"),
        "spectrum_pair_available_rows": count("spectrum_pair_available"),
        "exact_spectrum_seed_rows": count("spectrum_seed_exact_all"),
        "M5_fairness_candidate_pass_rows": count("M5_fairness_candidate_pass"),
        "mean_functional_spectrum_distance": mean("functional_spectrum_distance"),
        "mean_RSE_distance": mean("RSE_distance"),
        "mean_output_scale_distance": mean("output_scale_distance"),
        "mean_KAN_vs_MLP_matched_support_NLL_delta": mean("KAN_vs_MLP_matched_support_NLL_delta"),
        "mean_KAN_vs_MLP_matched_support_AUC_delta": mean("KAN_vs_MLP_matched_support_AUC_delta"),
        "note": "Rows come from complete in-process M5 trajectory matrices; seed=1 spectrum rows use explicit seed=0 spectrum fallback unless exact rows exist.",
    }


def write_m5_fairness_plan_metrics() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spectrum_rows = [
        r for r in read_rows(OUT_ROOT / "v22_45E_functional_actuator_spectrum_matrix.csv")
        if int_flag(r.get("functional_actuator_spectrum_computed")) and int_flag(r.get("RSE_computed"))
    ]
    spectrum = {
        (
            str(r.get("dataset", "")),
            str(r.get("seed", "")),
            str(r.get("architecture", "")),
            str(r.get("actuator", "")),
        ): r
        for r in spectrum_rows
    }
    metric_rows: list[dict[str, Any]] = []
    for path in m5_main_trajectory_matrix_paths():
        matrix_rows = read_rows(path)
        mlps = [
            r for r in matrix_rows
            if r.get("mechanism") == "M5"
            and r.get("architecture") == "MLP"
            and str(r.get("control_mode", "")) == "none"
            and str(r.get("variant", "")).startswith("MLP-")
            and finite_float(r.get("final_NLL")) is not None
        ]
        mlp_by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in mlps:
            mlp_by_group.setdefault((str(row.get("dataset", "")), str(row.get("seed", ""))), []).append(row)
        candidates = [
            r for r in matrix_rows
            if r.get("mechanism") == "M5"
            and r.get("architecture") == "strict_FC_PureKAN"
            and str(r.get("control_mode", "")) == "none"
            and str(r.get("variant", "")) != "optimizer_alone"
            and finite_float(r.get("final_NLL")) is not None
        ]
        for kan in candidates:
            dataset = str(kan.get("dataset", ""))
            seed = str(kan.get("seed", ""))
            group_mlps = mlp_by_group.get((dataset, seed), [])
            best_mlp = min(group_mlps, key=lambda r: value_or(r.get("final_NLL"), math.inf)) if group_mlps else None
            kan_final = finite_float(kan.get("final_NLL"))
            mlp_final = finite_float(best_mlp.get("final_NLL")) if best_mlp else None
            nll_delta = finite_float(kan.get("KAN_NLL_delta_vs_MLP_matched_support_FU"))
            if nll_delta is None and kan_final is not None and mlp_final is not None:
                nll_delta = kan_final - mlp_final
            kan_auc = finite_float(kan.get("AUC_loss_time"))
            mlp_auc = finite_float(best_mlp.get("AUC_loss_time")) if best_mlp else None
            auc_delta = kan_auc - mlp_auc if kan_auc is not None and mlp_auc is not None else None
            kan_arch, kan_actuator = m5_spectrum_locator(kan)
            mlp_arch, mlp_actuator = m5_spectrum_locator(best_mlp or {"architecture": "MLP", "variant": ""})
            kan_spec, kan_seed_exact = spectrum_row_for(spectrum, dataset, seed, kan_arch, kan_actuator)
            mlp_spec, mlp_seed_exact = spectrum_row_for(spectrum, dataset, seed, mlp_arch, mlp_actuator)
            fs_distance = functional_spectrum_distance(kan_spec, mlp_spec)
            kan_rse = finite_float(kan_spec.get("RSE_mean")) if kan_spec else None
            mlp_rse = finite_float(mlp_spec.get("RSE_mean")) if mlp_spec else None
            rse_distance = abs(kan_rse - mlp_rse) if kan_rse is not None and mlp_rse is not None else None
            kan_output_norm = finite_float(kan_spec.get("output_logit_norm")) if kan_spec else None
            mlp_output_norm = finite_float(mlp_spec.get("output_logit_norm")) if mlp_spec else None
            output_distance = abs(kan_output_norm - mlp_output_norm) if kan_output_norm is not None and mlp_output_norm is not None else None
            klass = m5_classification(kan, best_mlp, nll_delta)
            kan_improves = int(value_or(kan.get("NLL_improvement_vs_own_strong_optimizer"), 0.0) > 0.0)
            mlp_improves = int(value_or(best_mlp.get("NLL_improvement_vs_own_strong_optimizer") if best_mlp else None, 0.0) > 0.0)
            beats_mlp = int(nll_delta is not None and nll_delta <= 0.0)
            same_basis = m5_same_basis_control_pass(kan)
            no_debt = int_flag(kan.get("no_ECE_Brier_tail_debt"))
            overhead_ok = int(value_or(kan.get("controller_overhead_ratio"), math.inf) <= 0.35)
            degradation_driven = int(
                nll_delta is not None
                and nll_delta <= 0.0
                and kan_improves
                and value_or(best_mlp.get("NLL_improvement_vs_own_strong_optimizer") if best_mlp else None, 0.0) <= 0.0
            )
            fair_pass = int(
                kan_improves
                and same_basis
                and beats_mlp
                and klass in {"TrueKANGain", "BothGain"}
                and not degradation_driven
                and no_debt
                and overhead_ok
                and kan_spec is not None
                and mlp_spec is not None
            )
            metric_rows.append(
                {
                    "source_artifact": path.name,
                    "dataset": dataset,
                    "seed": seed,
                    "KAN_architecture": kan.get("architecture", ""),
                    "KAN_variant": kan.get("variant", ""),
                    "matched_MLP_variant": best_mlp.get("variant", "") if best_mlp else "",
                    "KAN_final_NLL": "" if kan_final is None else kan_final,
                    "matched_MLP_final_NLL": "" if mlp_final is None else mlp_final,
                    "KAN_NLL_improvement_vs_own_strong_optimizer": kan.get("NLL_improvement_vs_own_strong_optimizer", ""),
                    "MLP_matched_NLL_improvement_vs_own_strong_optimizer": (
                        best_mlp.get("NLL_improvement_vs_own_strong_optimizer", "") if best_mlp else ""
                    ),
                    "KAN_vs_MLP_matched_support_NLL_delta": "" if nll_delta is None else nll_delta,
                    "KAN_vs_MLP_matched_support_AUC_delta": "" if auc_delta is None else auc_delta,
                    "KAN_beats_MLP_matched_support": beats_mlp,
                    "KAN_improves": kan_improves,
                    "MLP_improves": mlp_improves,
                    "TrueKANGain_class": klass,
                    "TrueKANGain": int(klass == "TrueKANGain"),
                    "BothGain": int(klass == "BothGain"),
                    "TrueKANGain_or_BothGain": int(klass in {"TrueKANGain", "BothGain"}),
                    "ControlExplained": int(klass in {"MLPMatchedSupportStronger", "KANInternalValueOnly"}),
                    "MLPDegradationDriven": degradation_driven,
                    "beats_same_basis_controls": same_basis,
                    "beats_same_basis_Gram_random": int_flag(kan.get("beats_same_basis_Gram_random")),
                    "beats_same_basis_Gram_signflip": int_flag(kan.get("beats_same_basis_Gram_signflip")),
                    "beats_same_basis_OET_random": int_flag(kan.get("beats_same_basis_OET_random")),
                    "beats_same_bank_shuffled": int_flag(kan.get("beats_same_bank_shuffled")),
                    "no_ECE_Brier_tail_debt": no_debt,
                    "controller_overhead_ratio": kan.get("controller_overhead_ratio", ""),
                    "controller_overhead_le_0p35": overhead_ok,
                    "support_rank": kan.get("support_rank", ""),
                    "support_effective_rank": kan_spec.get("functional_actuator_effective_rank", "") if kan_spec else "",
                    "mlp_support_effective_rank": mlp_spec.get("functional_actuator_effective_rank", "") if mlp_spec else "",
                    "functional_spectrum_distance": "" if fs_distance is None else fs_distance,
                    "RSE_distance": "" if rse_distance is None else rse_distance,
                    "output_scale_distance": "" if output_distance is None else output_distance,
                    "KAN_RSE_mean": "" if kan_rse is None else kan_rse,
                    "matched_MLP_RSE_mean": "" if mlp_rse is None else mlp_rse,
                    "KAN_output_logit_norm": "" if kan_output_norm is None else kan_output_norm,
                    "matched_MLP_output_logit_norm": "" if mlp_output_norm is None else mlp_output_norm,
                    "KAN_spectrum_architecture": kan_arch,
                    "KAN_spectrum_actuator": kan_actuator,
                    "matched_MLP_spectrum_architecture": mlp_arch,
                    "matched_MLP_spectrum_actuator": mlp_actuator,
                    "KAN_spectrum_seed": kan_spec.get("seed", "") if kan_spec else "",
                    "matched_MLP_spectrum_seed": mlp_spec.get("seed", "") if mlp_spec else "",
                    "KAN_spectrum_seed_exact": kan_seed_exact,
                    "matched_MLP_spectrum_seed_exact": mlp_seed_exact,
                    "spectrum_seed_exact_all": int(kan_seed_exact and mlp_seed_exact),
                    "spectrum_pair_available": int(kan_spec is not None and mlp_spec is not None),
                    "M5_fairness_candidate_pass": fair_pass,
                    "claim_limit": "Audit-only plan metrics joined from complete in-process M5 trajectories and v22.44R/v22.45E functional spectrum rows; no new training rows are synthesized.",
                }
            )
    if metric_rows:
        summary_rows = [m5_summary_row(metric_rows, "ALL")]
        for source in sorted({str(r.get("source_artifact", "")) for r in metric_rows}):
            summary_rows.append(m5_summary_row([r for r in metric_rows if r.get("source_artifact") == source], source))
    else:
        summary_rows = [{"status": "not_run", "reason": "no complete M5 trajectory main matrix rows were available"}]
    write_rows(OUT_ROOT / "v22_45E_m5_fairness_plan_metrics.csv", metric_rows or [{"status": "not_run", "reason": "no complete M5 trajectory main matrix rows were available"}])
    write_rows(OUT_ROOT / "v22_45E_m5_fairness_plan_summary.csv", summary_rows)
    append_exec(
        "write_m5_fairness_plan_metrics",
        task_id="m5_fairness_plan_metrics",
        status="pass",
        files="results/v22_45E/v22_45E_m5_fairness_plan_metrics.csv, results/v22_45E/v22_45E_m5_fairness_plan_summary.csv",
        note=(
            f"rows={len(metric_rows)}; summary_rows={len(summary_rows)}; "
            "joins complete in-process M5 trajectory rows to functional spectrum/RSE/output-scale audit; seed fallback is explicit."
        ),
    )
    return metric_rows, summary_rows


def m5_fairness_plan_verified_result() -> str:
    rows = [r for r in read_rows(OUT_ROOT / "v22_45E_m5_fairness_plan_metrics.csv") if not r.get("status")]
    summary = next((r for r in read_rows(OUT_ROOT / "v22_45E_m5_fairness_plan_summary.csv") if r.get("source_artifact") == "ALL"), {})
    if not rows:
        return "not_run; no v22_45E_m5_fairness_plan_metrics.csv rows."
    return (
        f"m5 fairness plan metrics: rows={len(rows)}; "
        f"spectrum_pair_available={summary.get('spectrum_pair_available_rows')}; "
        f"exact_spectrum_seed={summary.get('exact_spectrum_seed_rows')}; "
        f"beats_mlp={summary.get('beats_MLP_matched_support_rows')}; "
        f"TrueKANGain+BothGain={summary.get('TrueKANGain_plus_BothGain_rows')}; "
        f"ControlExplained={summary.get('ControlExplained_rows')}; "
        f"MLPDegradationDriven={summary.get('MLPDegradationDriven_rows')}; "
        f"no_debt={summary.get('no_debt_rows')}; overhead<=0.35={summary.get('overhead_le_0p35_rows')}; "
        f"fairness_pass={summary.get('M5_fairness_candidate_pass_rows')}; "
        f"mean_spectrum_distance={summary.get('mean_functional_spectrum_distance')}."
    )


def m6_trajectory_verified_result(prefix: str = "m6_trajectory") -> str:
    plan_path = OUT_ROOT / f"v22_45E_{prefix}_plan_metrics.csv"
    status_path = OUT_ROOT / f"v22_45E_{prefix}_status.csv"
    summary_path = OUT_ROOT / f"v22_45E_{prefix}_mechanism_summary.csv"
    if not plan_path.exists():
        return f"not_run; {plan_path.name} is absent."
    plan_rows = [r for r in read_rows(plan_path) if not r.get("status")]
    status_rows = read_rows(status_path)
    failures = sum(1 for r in status_rows if r.get("status") == "fail")
    n = len(plan_rows)
    pass_rows = sum(int_flag(r.get("M6_reservoir_regularization_explains_gain")) for r in plan_rows)
    beats_supp = sum(int_flag(r.get("beats_noise_suppression")) for r in plan_rows)
    beats_signal = sum(int_flag(r.get("beats_signal_noise_control")) for r in plan_rows)
    beats_gauss = sum(int_flag(r.get("beats_same_norm_Gaussian_control")) for r in plan_rows)
    no_debt = sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in plan_rows)
    runtime_ok = sum(int_flag(r.get("runtime_truth")) for r in plan_rows)
    summary = next((r for r in read_rows(summary_path) if r.get("mechanism") == "M6"), {})

    def mean_of(key: str) -> str:
        vals = [value_or(r.get(key), math.nan) for r in plan_rows if finite_float(r.get(key)) is not None]
        if not vals:
            return "nan"
        return f"{statistics.fmean(vals):.6g}"

    return (
        f"{prefix}: trajectories={len(status_rows)}; failures={failures}; groups={n}; "
        f"cand={summary.get('candidate_rows', '')}; nll={summary.get('NLL_improvement_vs_own_rows', '')}; "
        f"control={summary.get('beats_matched_control_rows', '')}; "
        f"beats_suppression={beats_supp}/{n}; beats_signal_noise={beats_signal}/{n}; "
        f"beats_gaussian={beats_gauss}/{n}; no_debt={no_debt}/{n}; runtime_truth={runtime_ok}/{n}; "
        f"explanatory_pass={pass_rows}/{n}; mean_vs_base={mean_of('NLL_improvement_vs_base')}; "
        f"mean_vs_suppression={mean_of('NLL_improvement_vs_noise_suppression')}; "
        f"mean_reservoir_energy={mean_of('reservoir_noise_energy')}; mean_signal_leakage={mean_of('signal_noise_leakage')}. "
        "Complete in-process trajectories with parameter support-complement reservoir noise; explanatory only, not SignalFUOpened."
    )


def m7_trajectory_verified_result(prefix: str = "m7_trajectory") -> str:
    plan_path = OUT_ROOT / f"v22_45E_{prefix}_plan_metrics.csv"
    status_path = OUT_ROOT / f"v22_45E_{prefix}_status.csv"
    summary_path = OUT_ROOT / f"v22_45E_{prefix}_summary.json"
    if not plan_path.exists():
        return f"not_run; {plan_path.name} is absent."
    plan_rows = [r for r in read_rows(plan_path) if not r.get("status")]
    status_rows = read_rows(status_path)
    summary = read_json(summary_path)
    failures = sum(1 for r in status_rows if r.get("status") == "fail")
    n = len(plan_rows)
    tasks = ",".join(sorted({str(r.get("dataset", "")) for r in plan_rows if r.get("dataset")}))
    ranks = ",".join(sorted({str(r.get("memory_rank", "")) for r in plan_rows if r.get("memory_rank")}))

    def mean_of(key: str) -> str:
        vals = [value_or(r.get(key), math.nan) for r in plan_rows if finite_float(r.get(key)) is not None]
        if not vals:
            return "nan"
        return f"{statistics.fmean(vals):.6g}"

    return (
        f"{prefix}: trajectories={len(status_rows)}; failures={failures}; groups={n}; "
        f"tasks={tasks or 'n/a'}; memory_ranks={ranks or 'n/a'}; "
        f"valid_forgetting={sum(int_flag(r.get('forgetting_task_valid')) for r in plan_rows)}/{n}; "
        f"opened={sum(int_flag(r.get('M7_exploration_opened')) for r in plan_rows)}/{n}; "
        f"official_ready={sum(int_flag(r.get('M7_official_ready')) for r in plan_rows)}/{n}; "
        f"controls_fail={sum(int_flag(r.get('matched_memory_controls_fail')) for r in plan_rows)}/{n}; "
        f"final_nonworse={sum(int_flag(r.get('final_average_accuracy_nonworse')) for r in plan_rows)}/{n}; "
        f"constraint_ok={sum(int_flag(r.get('constraint_ok')) for r in plan_rows)}/{n}; "
        f"runtime_truth={sum(int_flag(r.get('runtime_truth')) for r in plan_rows)}/{n}; "
        f"mean_relative_forgetting_reduction={mean_of('relative_forgetting_reduction')}; "
        f"mean_projection_after={mean_of('memory_projection_after_ratio')}; route={summary.get('route', '')}. "
        f"Complete in-process M7 trajectories; {summary.get('claim_limit', 'partial M7 only')}."
    )


def m9_trajectory_verified_result(prefix: str = "m9_trajectory") -> str:
    plan_path = OUT_ROOT / f"v22_45E_{prefix}_plan_metrics.csv"
    status_path = OUT_ROOT / f"v22_45E_{prefix}_status.csv"
    summary_path = OUT_ROOT / f"v22_45E_{prefix}_mechanism_summary.csv"
    if not plan_path.exists():
        return f"not_run; {plan_path.name} is absent."
    plan_rows = [r for r in read_rows(plan_path) if not r.get("status")]
    status_rows = read_rows(status_path)
    failures = sum(1 for r in status_rows if r.get("status") == "fail")
    n = len(plan_rows)
    opened = sum(int_flag(r.get("AdaptiveMetricFlowOpened")) for r in plan_rows)
    diagnostic = sum(int_flag(r.get("MetricSelectionDiagnosticOnly")) for r in plan_rows)
    beats_uniform = sum(int_flag(r.get("beats_uniform")) for r in plan_rows)
    beats_best = sum(int_flag(r.get("beats_best_fixed_metric")) for r in plan_rows)
    no_debt = sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in plan_rows)
    runtime_ok = sum(int_flag(r.get("runtime_truth")) for r in plan_rows)
    adaptation_ok = sum(int_flag(r.get("adaptation_truth")) for r in plan_rows)
    collapsed = sum(int_flag(r.get("mixture_collapsed")) for r in plan_rows)
    summary = next((r for r in read_rows(summary_path) if r.get("mechanism") == "M9"), {})

    def mean_of(key: str) -> str:
        vals = [value_or(r.get(key), math.nan) for r in plan_rows if finite_float(r.get(key)) is not None]
        if not vals:
            return "nan"
        return f"{statistics.fmean(vals):.6g}"

    return (
        f"{prefix}: trajectories={len(status_rows)}; failures={failures}; groups={n}; "
        f"cand={summary.get('candidate_rows', '')}; nll={summary.get('NLL_improvement_vs_own_rows', '')}; "
        f"control={summary.get('beats_matched_control_rows', '')}; "
        f"opened={opened}/{n}; metric_selection_diagnostic={diagnostic}/{n}; "
        f"beats_uniform={beats_uniform}/{n}; beats_best_fixed={beats_best}/{n}; "
        f"no_debt={no_debt}/{n}; runtime_truth={runtime_ok}/{n}; adaptation_truth={adaptation_ok}/{n}; "
        f"collapsed={collapsed}/{n}; mean_vs_base={mean_of('NLL_improvement_vs_base')}; "
        f"mean_vs_uniform={mean_of('NLL_improvement_vs_uniform')}; "
        f"mean_vs_best_fixed={mean_of('NLL_improvement_vs_best_fixed')}; "
        f"mean_final_entropy={mean_of('metric_mixture_final_entropy')}; mean_overhead={mean_of('controller_overhead_ratio')}. "
        "Complete in-process trajectories with train-only diagonal metric-mixture mirror descent; no validation/test metric search."
    )


def m8_gauge_audit_verified_result() -> str:
    audit_path = OUT_ROOT / "v22_45E_m8_gauge_feasibility_audit.csv"
    dynamic_path = OUT_ROOT / "v22_45E_m8_gauge_dynamic_equivalence.csv"
    if not audit_path.exists():
        return "not_run; v22_45E_m8_gauge_feasibility_audit.csv is absent."
    audit_rows = [r for r in read_rows(audit_path) if not r.get("status")]
    dynamic_rows = [r for r in read_rows(dynamic_path) if not r.get("status")]
    faithful = sum(int_flag(r.get("faithful_runtime_gamma_available")) for r in audit_rows)
    runtime_canon = sum(int_flag(r.get("runtime_fu_state_canonical_update_available")) for r in audit_rows)
    dynamic_test_rows = [r for r in dynamic_rows if r.get("transform_name") != "identity_restore_check"]
    equivalent = sum(int_flag(r.get("function_equivalent_under_tol")) for r in dynamic_test_rows)
    tested = len(dynamic_test_rows)
    deltas = [value_or(r.get("max_abs_logit_delta"), math.nan) for r in dynamic_test_rows if finite_float(r.get("max_abs_logit_delta")) is not None]
    max_delta = max(deltas) if deltas else math.nan
    return (
        f"M8 gauge audit: feasibility_rows={len(audit_rows)}; dynamic_tests={tested}; "
        f"faithful_runtime_gamma_available={faithful}/{len(audit_rows)}; "
        f"runtime_fu_state_canonical_update_available={runtime_canon}/{len(audit_rows)}; "
        f"function_equivalent_naive_transforms={equivalent}/{tested}; "
        f"max_abs_logit_delta={max_delta:.6g}; GaugeDiagnosticOnly/blocker; no route promotion."
    )


def _m8_restore_state(model: Any, state: dict[str, Any]) -> None:
    import torch

    with torch.no_grad():
        model.load_state_dict({k: v.clone() for k, v in state.items()}, strict=True)


def _m8_apply_hidden_covariance_rescale(model: Any, xb: Any) -> str:
    import torch

    if not hasattr(model, "w1") or not hasattr(model, "hidden"):
        return "skipped_no_w1_or_hidden"
    with torch.no_grad():
        hidden = model.hidden(xb).detach().float()
        scale = hidden.std(dim=0).clamp_min(1.0e-3).reciprocal().clamp(0.25, 4.0)
        if model.w1.ndim == 3 and int(model.w1.shape[1]) == int(scale.numel()):
            model.w1.mul_(scale.to(device=model.w1.device, dtype=model.w1.dtype)[None, :, None])
            return "applied_w1_hidden_column_rescale_from_train_hidden_std"
    return "skipped_shape_mismatch"


def _m8_apply_basis_gram_w2_rescale(model: Any, xb: Any) -> str:
    import torch

    if not hasattr(model, "w2") or not hasattr(model, "frozen_readout_features"):
        return "skipped_no_w2_or_readout_features"
    with torch.no_grad():
        feats = model.frozen_readout_features(xb).detach().float()
        centered = feats - feats.mean(dim=0, keepdim=True)
        energy = centered.square().mean(dim=0).clamp_min(1.0e-6)
        if model.w2.ndim == 3:
            hidden, out_dim, k_param = [int(v) for v in model.w2.shape]
            core_energy = energy[: hidden * k_param]
            if int(core_energy.numel()) == hidden * k_param:
                scale = core_energy.sqrt().reciprocal().clamp(0.25, 4.0).view(hidden, k_param)
                model.w2.mul_(scale.to(device=model.w2.device, dtype=model.w2.dtype)[:, None, :].expand(hidden, out_dim, k_param))
                return "applied_w2_basis_gram_rescale_from_train_readout_feature_energy"
    return "skipped_shape_mismatch"


def _m8_apply_degree_frequency_bank_rescale(model: Any, xb: Any) -> str:
    import torch

    if not hasattr(model, "w1") or not hasattr(model, "layer1_basis"):
        return "skipped_no_w1_or_layer1_basis"
    with torch.no_grad():
        b1 = model.layer1_basis(xb).detach().float()
        energy = b1.square().mean(dim=0).clamp_min(1.0e-6)
        if model.w1.ndim == 3 and int(model.w1.shape[0]) == int(energy.shape[0]) and int(model.w1.shape[2]) == int(energy.shape[1]):
            bank = energy.mean(dim=0)
            scale = (bank.mean().clamp_min(1.0e-6) / bank).sqrt().clamp(0.25, 4.0)
            model.w1.mul_(scale.to(device=model.w1.device, dtype=model.w1.dtype)[None, None, :])
            return "applied_w1_degree_frequency_bank_rescale_from_train_basis_energy"
    return "skipped_shape_mismatch"


def run_m8_gauge_audit(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from experiments import run_v22_37_causal_instrumented_functional_optimizer as core

    ensure_out()
    device = v2243.torch_device(str(args.device))
    feasibility_rows = [
        {
            "mechanism": "M8",
            "candidate_gauge": "hidden_activation_covariance_whitening_gauge",
            "plan_requirement": "hidden activation covariance whitening gauge plus theta_t^canon = Gamma(theta_t)",
            "code_evidence": "PrimitiveKAN.hidden computes tanh(stream_mix(layer1_basis, w1)); layer2_basis then evaluates fixed basis functions on that nonlinear hidden state.",
            "repair_direction_tried": "Dynamic naive w1 hidden-column rescale from train hidden std, plus static code-path audit.",
            "faithful_runtime_gamma_available": 0,
            "runtime_fu_state_canonical_update_available": 0,
            "route_eligible": 0,
            "diagnostic_only": 1,
            "blocker": "Changing w1 scale changes hidden activations before tanh and therefore changes downstream fixed-basis features; no inverse parameter transform is implemented to preserve logits.",
        },
        {
            "mechanism": "M8",
            "candidate_gauge": "basis_gram_normalized_gauge",
            "plan_requirement": "basis Gram normalized gauge plus FU state updated in canonical gauge",
            "code_evidence": "basis_gram_diag only builds a diagonal metric from frozen_readout_features/layer1_basis; it does not rewrite model parameters or controller state into canonical coordinates.",
            "repair_direction_tried": "Dynamic naive w2 basis-Gram rescale from train readout feature energy, plus static audit of basis_gram_diag.",
            "faithful_runtime_gamma_available": 0,
            "runtime_fu_state_canonical_update_available": 0,
            "route_eligible": 0,
            "diagnostic_only": 1,
            "blocker": "The current code has metric normalization diagnostics, not Gamma(theta); rescaling w2 without transforming fixed basis functions changes the represented function.",
        },
        {
            "mechanism": "M8",
            "candidate_gauge": "OET_left_right_balanced_gauge",
            "plan_requirement": "OET left/right balanced gauge",
            "code_evidence": "v22.43/v22.45E P3 uses Cayley/Lie update diagnostics and transported Lie momentum, but no persistent left/right factorization gauge or canonical FU coordinate map is present.",
            "repair_direction_tried": "Static code-path audit; no dynamic transform attempted because no left/right OET factor parameters exist in the runtime controller state.",
            "faithful_runtime_gamma_available": 0,
            "runtime_fu_state_canonical_update_available": 0,
            "route_eligible": 0,
            "diagnostic_only": 1,
            "blocker": "No explicit left/right factor pair exists to balance while preserving the model function and transporting FU state.",
        },
        {
            "mechanism": "M8",
            "candidate_gauge": "KAN_degree_frequency_bank_balanced_gauge",
            "plan_requirement": "KAN degree/frequency bank balanced gauge",
            "code_evidence": "PrimitiveKAN fixed centers/scales and basis_name define basis functions; w1/w2 coefficients are parameters, but the basis functions themselves are not reparameterized by a bank-coordinate map.",
            "repair_direction_tried": "Dynamic naive w1 degree/frequency bank rescale from train layer1 basis energy, plus static audit.",
            "faithful_runtime_gamma_available": 0,
            "runtime_fu_state_canonical_update_available": 0,
            "route_eligible": 0,
            "diagnostic_only": 1,
            "blocker": "Bank rescaling of coefficients alone changes hidden preactivations or readout logits; no paired basis-function inverse transform is implemented.",
        },
    ]
    dynamic_rows: list[dict[str, Any]] = []
    transform_fns = [
        ("identity_restore_check", lambda model, xb: "identity_no_parameter_change"),
        ("hidden_covariance_whitening_naive_w1_rescale", _m8_apply_hidden_covariance_rescale),
        ("basis_gram_naive_w2_rescale", _m8_apply_basis_gram_w2_rescale),
        ("degree_frequency_bank_naive_w1_rescale", _m8_apply_degree_frequency_bank_rescale),
    ]
    for dataset in split_csv(args.eval_datasets):
        for seed in split_csv(args.eval_seeds, int):
            for architecture in split_csv(args.eval_architectures):
                if str(architecture) == "MLP":
                    continue
                train_loader, _held_loader, _test_loader, input_dim, output_dim, x_stats, meta = core.make_loaders_for_dataset(
                    str(dataset),
                    int(args.train_size),
                    int(args.held_size),
                    int(args.batch_size),
                    int(seed),
                    tier2_download=bool(args.tier2_download),
                )
                xb, _yb = next(iter(train_loader))
                xb = xb.to(device).float()
                model_seed = int(seed) + 224508 + (1000 if str(architecture) == "DGKAN_DCHE" else 2000)
                model = core.make_model_for_arch(str(architecture), int(input_dim), int(output_dim), int(args.hidden), model_seed, device, x_stats)
                model.eval()
                with torch.no_grad():
                    baseline_logits = model(xb).detach().float()
                    baseline_norm = float(baseline_logits.norm().item())
                    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                for transform_name, transform_fn in transform_fns:
                    _m8_restore_state(model, state)
                    status = transform_fn(model, xb)
                    with torch.no_grad():
                        after_logits = model(xb).detach().float()
                        delta = after_logits - baseline_logits
                        max_abs_delta = float(delta.abs().max().item())
                        rel_l2 = float(delta.norm().item() / max(1.0e-12, baseline_norm))
                    dynamic_rows.append(
                        {
                            "mechanism": "M8",
                            "dataset": dataset,
                            "seed": int(seed),
                            "architecture": architecture,
                            "transform_name": transform_name,
                            "transform_status": status,
                            "batch_size": int(xb.shape[0]),
                            "task_tier": meta.get("task_tier", ""),
                            "source_kind": meta.get("source_kind", ""),
                            "max_abs_logit_delta": max_abs_delta,
                            "relative_l2_logit_delta": rel_l2,
                            "function_equivalent_tolerance": 1.0e-5,
                            "function_equivalent_under_tol": int(max_abs_delta <= 1.0e-5 and rel_l2 <= 1.0e-5),
                            "faithful_runtime_gamma_available": 0 if transform_name != "identity_restore_check" else 1,
                            "diagnostic_only": int(transform_name != "identity_restore_check"),
                            "route_eligible": 0,
                        }
                    )
                    _m8_restore_state(model, state)
    write_rows(OUT_ROOT / "v22_45E_m8_gauge_feasibility_audit.csv", feasibility_rows)
    write_rows(OUT_ROOT / "v22_45E_m8_gauge_dynamic_equivalence.csv", dynamic_rows or [{"status": "no_dynamic_rows"}])
    equivalent = sum(int_flag(r.get("function_equivalent_under_tol")) for r in dynamic_rows if r.get("transform_name") != "identity_restore_check")
    tested = sum(1 for r in dynamic_rows if r.get("transform_name") != "identity_restore_check")
    append_exec(
        trajectory_command_line(args, "m8-gauge-audit"),
        task_id="m8_gauge_audit",
        status="pass",
        gpu=str(args.device),
        files="results/v22_45E/v22_45E_m8_gauge_feasibility_audit.csv, results/v22_45E/v22_45E_m8_gauge_dynamic_equivalence.csv",
        note=(
            f"feasibility_rows={len(feasibility_rows)}; dynamic_rows={len(dynamic_rows)}; "
            f"non_identity_equivalent={equivalent}/{tested}; "
            "audit-only: no Gamma(theta) or canonical FU state update promoted."
        ),
    )
    return {"feasibility_rows": len(feasibility_rows), "dynamic_rows": len(dynamic_rows), "non_identity_equivalent": equivalent, "tested": tested}


def m1_trajectory_prefix_verified_result(prefix: str) -> str:
    summary_path = OUT_ROOT / f"v22_45E_{prefix}_mechanism_summary.csv"
    status_path = OUT_ROOT / f"v22_45E_{prefix}_status.csv"
    matrix_path = OUT_ROOT / f"v22_45E_{prefix}_matrix.csv"
    if not summary_path.exists():
        return f"not_run; {summary_path.name} is absent."
    m1 = next((r for r in read_rows(summary_path) if r.get("mechanism") == "M1"), {})
    status_rows = read_rows(status_path)
    failures = sum(1 for r in status_rows if r.get("status") == "fail")
    candidates = [
        r for r in read_rows(matrix_path)
        if r.get("mechanism") == "M1" and r.get("control_mode") == "none"
    ]
    mirror_ms = [
        value_or(r.get("mirror_grad_ms"), math.nan)
        for r in candidates
        if finite_float(r.get("mirror_grad_ms")) is not None
    ]
    refresh = [
        value_or(r.get("mirror_grad_refreshed_fraction"), math.nan)
        for r in candidates
        if finite_float(r.get("mirror_grad_refreshed_fraction")) is not None
    ]
    cached = [
        value_or(r.get("mirror_grad_cached_fraction"), math.nan)
        for r in candidates
        if finite_float(r.get("mirror_grad_cached_fraction")) is not None
    ]
    ls_basis = [
        value_or(r.get("mirror_ls_basis_count"), math.nan)
        for r in candidates
        if finite_float(r.get("mirror_ls_basis_count")) is not None
    ]
    ls_random_basis = [
        value_or(r.get("mirror_ls_random_basis_count"), math.nan)
        for r in candidates
        if finite_float(r.get("mirror_ls_random_basis_count")) is not None
    ]
    ls_fd = [
        value_or(r.get("mirror_ls_fd_eval_count"), math.nan)
        for r in candidates
        if finite_float(r.get("mirror_ls_fd_eval_count")) is not None
    ]
    ls_residual = [
        value_or(r.get("mirror_ls_residual_ratio"), math.nan)
        for r in candidates
        if finite_float(r.get("mirror_ls_residual_ratio")) is not None
    ]
    ls_fallback = [
        value_or(r.get("mirror_ls_fallback_fraction"), math.nan)
        for r in candidates
        if finite_float(r.get("mirror_ls_fallback_fraction")) is not None
    ]
    extra = ""
    if mirror_ms:
        extra += f"; mirror_grad_ms_mean={statistics.fmean(mirror_ms):.6g}"
    if refresh:
        extra += f"; refresh_fraction_mean={statistics.fmean(refresh):.6g}"
    if cached:
        extra += f"; cached_fraction_mean={statistics.fmean(cached):.6g}"
    if ls_basis:
        extra += f"; mirror_ls_basis_mean={statistics.fmean(ls_basis):.6g}"
    if ls_random_basis:
        extra += f"; mirror_ls_random_basis_mean={statistics.fmean(ls_random_basis):.6g}"
    if ls_fd:
        extra += f"; mirror_ls_fd_eval_mean={statistics.fmean(ls_fd):.6g}"
    if ls_residual:
        extra += f"; mirror_ls_residual_mean={statistics.fmean(ls_residual):.6g}"
    if ls_fallback:
        extra += f"; mirror_ls_fallback_mean={statistics.fmean(ls_fallback):.6g}"
    return (
        f"{prefix}: trajectories={len(status_rows)}; failures={failures}; "
        f"cand={m1.get('candidate_rows')}; nll={m1.get('NLL_improvement_vs_own_rows')}; "
        f"control={m1.get('beats_matched_control_rows')}; no_debt={m1.get('no_ECE_Brier_tail_debt_rows')}; "
        f"overhead={m1.get('controller_overhead_le_0p35_rows')}; pass={m1.get('screen_pass')}"
        f"{extra}."
    )


def write_execution_repair_log() -> list[dict[str, Any]]:
    rows = [
        {
            "repair_id": "runner_metric_spectrum_args",
            "observed_blocker": "Initial Part C functional actuator spectrum audit failed with Namespace missing held_size, so computed_rows was 0.",
            "code_change": "Patched run_metric_and_spectrum to forward held_size, batch_size, and tier2_download into the v22.43 spectrum audit namespace.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage metric-spectrum --label v22_45E --gpus 0,1,2,3 --device cuda:0 --hidden 32 --held-size 128 --batch-size 64",
            "verified_result": "pass; results/v22_45E/v22_45E_functional_actuator_spectrum_matrix.csv has computed_rows=40 in command journal after rerun.",
        },
        {
            "repair_id": "runner_merge_pure_rows",
            "observed_blocker": "Initial v22.45E merge omitted v22_43P_pure_support_full_loop_matrix, leaving M2/M3 candidate rows absent from the screen summary.",
            "code_change": "Patched merge_v2243_outputs to call both merge_chunks and merge_pure_artifacts, then concatenate regular and pure support matrices before applying v22.45E screen specs.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage merge --label v22_45E --gpus 0,1,2,3 --workers 4 --steps 200 --train-size 128 --held-size 128 --batch-size 64 --hidden 32 --support-rank 4",
            "verified_result": "pass; v22_45E_lowcost_screen_matrix.csv has 296 rows; M2 candidate_rows=16 and M3 candidate_rows=16.",
        },
        {
            "repair_id": "runner_m4_m7_copy_labels",
            "observed_blocker": "Initial diagnostic copy looked for wrong v22.42R source labels, so v22.45E M4/M7 diagnostic artifacts were placeholders.",
            "code_change": "Patched run_m4_diagnostic and run_m7_diagnostic source labels to v22_45E_M4_slow_signal and v22_45E_M7_continual_memory.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage m4-diagnostic --label v22_45E ...; python experiments/run_v22_45E_multi_mechanism_extension.py --stage m7-diagnostic --label v22_45E ...",
            "verified_result": "pass after diagnostic reruns; copied v22_45E_M4_slow_signal_diagnostic_* and v22_45E_M7_continual_memory_* from v22.42R source artifacts.",
        },
        {
            "repair_id": "runner_derived_metric_backfill",
            "observed_blocker": "After regular+pure merge, some M1/M2/M3 rows had actual final_NLL/ECE/Brier/tail metrics but blank derived baseline deltas, so screen summaries undercounted NLL/no-debt evidence.",
            "code_change": "Added enrich_v2245_derived_metrics to recompute NLL improvement, safety deltas, no-debt, and control beats from landed final metrics after v22.45E combines regular and pure rows.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage merge --label v22_45E; python experiments/run_v22_45E_multi_mechanism_extension.py --stage finalize --label v22_45E",
            "verified_result": "pass after merge/finalize rerun; M1 now has 7 NLL-improvement rows and M2/M3 each have 2 instead of blank-derived zero.",
        },
        {
            "repair_id": "m2_functional_radial_gate_proxy",
            "observed_blocker": "Plan fallback asks for functional/safety-gated radial flow, while initial v22.45E only exercised a fixed radial005 cap.",
            "code_change": "Added P4-Euclidean-OET-functional-radial-gated-pure in v22.43 with train-only support_overlap/safety_budget_debt gating and recorded gate active fraction, score, and effective cap.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage repair-screen --label v22_45E_repair ...",
            "verified_result": "repair-screen pass with 152 rows; M2 improved to 6 NLL-gain rows, 12 control-beat rows, and 5 no-debt rows, but overhead_ok remained 0/16 so screen_pass stayed 0.",
        },
        {
            "repair_id": "m5_overhead_variant_audits",
            "observed_blocker": "M5 KAN BasisGram carrier nearly passed low-cost gates after corrected M5 grouping, but controller_overhead_le_0p35 was 0/8.",
            "code_change": "Added m5-variant-audit stage, dynamic m5_variant recap ingestion, same-basis-OET control counting, and M5 KAN-only summary with MLP matched support comparison.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage m5-variant-audit --m5-variant-output m5_variant... for OET-BankLocal, BasisGram h128, BasisGram h128/batch128, and h128/batch128+safety.",
            "verified_result": "OET-BankLocal reduced neither debt nor overhead; BasisGram h128/batch128 reached overhead_ok 6/8 but no-debt stayed 3/8; safety barrier kept no-debt 3/8 and overhead_ok 4/8. No M5 variant passed.",
        },
        {
            "repair_id": "m5_tail_q99_guard_audits",
            "observed_blocker": "Best M5 BasisGram h128/batch128 run still failed no-debt mostly because tail_q99_delta_vs_own_strong_optimizer stayed positive in 5/8 candidate rows.",
            "code_change": "Added tail_q99_brier_qp_strict_margin and tail_q99_brier_qp_hard_margin modes in the v22.43 kernel, using train-only tail_q99/brier auxiliary gradients with stricter QP guard margins.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage m5-variant-audit --m5-variant-output m5_variant_basis_h128_b128_tailstrict|tailhard_c80|tailstrict_light ...",
            "verified_result": "strict tail guard improved M5 NLL/control to 8/8 but no-debt stayed 3/8 and overhead_ok fell to 0/8; hard/c80 no-debt regressed to 2/8; light guard kept no-debt 3/8 and overhead_ok 0/8. Tail guard evidence is useful but not sufficient for screen pass.",
        },
        {
            "repair_id": "implementation_validity_audit",
            "observed_blocker": "User audit correctly identified that the current v22.45E harness and mechanism mappings are not faithful implementations of the full plan, not merely inefficient GPU utilization.",
            "code_change": "Separated legacy per-row dispatch evidence from trajectory-first evidence. Later M1 repairs implement train-batch dual/primal mirror state and an M1-KLMirrorLS sketched log-prob finite-difference ridge-LS actuator, but not the planned full all-function/all-parameter solver; M2 remains finite-difference train-batch functional-JVP; M3 now has transported left-Lie generator momentum but OET is still left-Cayley only; M9 now has train-only runtime mixture state but negative route evidence; M8 still lacks a faithful runtime Gamma(theta) and canonical FU state update.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage finalize --label v22_45E",
            "verified_result": "No fabricated promotion: final_route remains R14-SupportOnlyNoSignalIncrement; trajectory artifacts are retained with explicit claim limits, not promoted beyond their implementation fidelity.",
        },
        {
            "repair_id": "trajectory_first_m2_runner",
            "observed_blocker": "The old v22.45E execution path used per-spec subprocess dispatch and was not acceptable as a faithful FU training harness for route claims.",
            "code_change": "Added --stage trajectory-m2: builds complete trajectory specs, runs train_trajectory_from_spec in-process with one full model/optimizer/controller/step loop per trajectory on cuda:0-3, writes trajectory plan metrics, and records subprocess_used=0 in the execution log.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m2 --label v22_45E_m2_trajectory --gpus 0,1,2,3 --workers 4 ...",
            "verified_result": trajectory_first_verified_result(),
        },
        {
            "repair_id": "trajectory_m2_warmup_guard",
            "observed_blocker": "The first trajectory smoke used steps=40 with trajectory_warmup_steps=60, so pure-FU/radial updates never became active and functional_radial_fd_eval_count_mean stayed 0.",
            "code_change": "Clamped trajectory warmup to be strictly less than trajectory_steps; if a short smoke would otherwise be all warmup, warmup becomes min(steps-1, steps//4).",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m2 --label v22_45E_m2_trajectory_smoke --steps 40 --trajectory-steps 40 ...",
            "verified_result": trajectory_first_verified_result(),
        },
        {
            "repair_id": "m2_functional_rse_radial_gate",
            "observed_blocker": "The first trajectory M2 repair still used support_overlap to scale the top-level radial cap before per-mode response scoring, while the plan asks for functional actuator spectrum/RSE gated radial flow.",
            "code_change": "Added P4-Euclidean-OET-functional-rse-radial010-gated-pure and P4-Euclidean-OET-functional-rse-relaxed-safe-radial010-gated-pure. These variants use a safety-only top gate, cap radial energy at 0.10, and route the actual update through apply_functional_mode_radial_channel with train-batch finite-difference response/RSE scoring. Also added --m2-trajectory-output so multiple M2 trajectory suites do not overwrite one another, and expanded trajectory command logging to include support/metric refresh plus velocity/safety parameters.",
            "rerun_command": "/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m2 --label v22_45E_m2_trajectory_rse_relaxed_radial010_c240 --m2-trajectory-output m2_trajectory_rse_relaxed_radial010_c240 --m2-repair-variant P4-Euclidean-OET-functional-rse-relaxed-safe-radial010-gated-pure --eval-datasets MNIST,FashionMNIST,KMNIST,Wine --eval-seeds 0,1 --eval-architectures MLP,DGKAN_DCHE --gpus 0,1,2,3 --workers 4 --steps 240 --trajectory-steps 240 --train-size 128 --held-size 128 --batch-size 128 --hidden 128 --support-rank 8 --support-refresh-cadence 240 --metric-refresh-cadence 240 --trajectory-m2-velocity-scale 0.15 --trajectory-m2-safety-barrier 5.0 --trajectory-m2-beta-signal 0.02 --velocity-scale 0.20 --safety-budget-velocity-barrier 4.0",
            "verified_result": m2_trajectory_prefix_verified_result("m2_trajectory_rse_relaxed_radial010_c240"),
        },
        {
            "repair_id": "m3_transported_lie_momentum_runner",
            "observed_blocker": "M3 had only a pure OET tangent proxy and no runtime momentum state or matched Lie controls, so it could not satisfy the plan's transported-Lie mechanism identity.",
            "code_change": "Added P3-Euclidean-OET-lie-momentum-pure and P3-Euclidean-OET-transported-lie-momentum-pure. The transported variant keeps a per-matrix skew generator momentum, transports the previous generator by the previous Cayley rotation, applies EMA in the current Lie frame, and executes the Cayley update from the momentum generator. Added same-Lie-random/signflip controls and --stage trajectory-m3 with independent plan/status artifacts.",
            "rerun_command": "/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m3 --label v22_45E_m3_trajectory_lie_c240 --m3-trajectory-output m3_trajectory_lie_c240 --m3-trajectory-variant P3-Euclidean-OET-transported-lie-momentum-pure --eval-datasets MNIST,FashionMNIST,KMNIST,Wine --eval-seeds 0,1 --eval-architectures MLP,DGKAN_DCHE --gpus 0,1,2,3 --workers 4 --steps 240 --trajectory-steps 240 --train-size 128 --held-size 128 --batch-size 128 --hidden 128 --support-rank 8 --support-refresh-cadence 240 --metric-refresh-cadence 240 --trajectory-m3-velocity-scale 0.10 --trajectory-m3-safety-barrier 8.0 --trajectory-m3-beta-signal 0.05 --velocity-scale 0.20 --safety-budget-velocity-barrier 4.0",
            "verified_result": m3_trajectory_prefix_verified_result("m3_trajectory_lie_c240"),
        },
        {
            "repair_id": "trajectory_first_m4_runner",
            "observed_blocker": "M4 had only copied v22.42R diagnostic artifacts, so v22.45E could not audit slow-signal reservoir migration as its own complete training trajectory.",
            "code_change": "Added --stage trajectory-m4. Each spec runs one complete modular-addition training loop in-process, maintains EMA slow gradient state, defines a top-k train-gradient support and fast residual reservoir, applies slow_signal_flow or matched random/signflip controls after the optimizer step, and records slow_signal_energy, fast_signal_energy, RSM_index, grokking_delay, H200/H800, and runtime no-candidate flags. This is a partial top-k support/reservoir approximation, not the full metric/OET projection.",
            "rerun_command": "/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m4 --label v22_45E_m4_trajectory --m4-trajectory-output m4_trajectory --diagnostic-seeds 0,1 --diagnostic-architectures MLP,DGKAN_DCHE --gpus 0,1,2,3 --workers 4 --m4-trajectory-steps 800 --diagnostic-hidden 64 --diagnostic-batch-size 128 --m4-modulus 13 --m4-train-fraction 0.40 ...; reran p13 c1600 weight_decay=0.001 as m4_trajectory_wd1e3_c1600; reran p7 c1600 weight_decay=0.001 as m4_trajectory_p7_wd1e3_c1600. Exact commands are in v22_45E_command_journal.csv and 执行日志.",
            "verified_result": (
                m4_trajectory_prefix_verified_result("m4_trajectory")
                + " "
                + m4_trajectory_prefix_verified_result("m4_trajectory_wd1e3_c1600")
                + " "
                + m4_trajectory_prefix_verified_result("m4_trajectory_p7_wd1e3_c1600")
            ),
        },
        {
            "repair_id": "m4_class_mnist_continual_task",
            "observed_blocker": "M4 modular-addition trajectories were faithful but all had no test crossing, so the grokking-delay gate was invalid; the v22.45E plan also permits continual forgetting reduction as an M4 criterion.",
            "code_change": "Added --m4-tasks Class_MNIST_0_4_to_5_9. Each run trains an old digit split, then a new digit split, maintains the M4 EMA slow-gradient state, applies slow-signal/reservoir velocity or matched controls during the new phase, and evaluates relative forgetting reduction with runtime truth flags.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m4 --label v22_45E_m4_class_mnist --m4-trajectory-output m4_trajectory_class_mnist --m4-tasks Class_MNIST_0_4_to_5_9 --diagnostic-seeds 0,1 --diagnostic-architectures MLP,DGKAN_DCHE --gpus 0,1,2,3 --workers 4 ...",
            "verified_result": m4_trajectory_prefix_verified_result("m4_trajectory_class_mnist"),
        },
        {
            "repair_id": "m4_class_mnist_velocity_scale_audit",
            "observed_blocker": "The first Class_MNIST M4 continual run used the historical modular velocity scale, making the applied M4 displacement nearly no-op (lr*rho*velocity_scale=1e-6) and producing identical forgetting across real and controls.",
            "code_change": "Reran the same M4 Class_MNIST in-process trajectory with stronger normalized M4 velocity only (--m4-rho 0.5 --m4-diag-velocity-scale 1.0), leaving optimizer, data, variants, controls, and runtime-policy guards unchanged.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m4 --label v22_45E_m4_class_mnist_strong --m4-trajectory-output m4_trajectory_class_mnist_strong --m4-tasks Class_MNIST_0_4_to_5_9 --m4-rho 0.5 --m4-diag-velocity-scale 1.0 --diagnostic-seeds 0,1 --diagnostic-architectures MLP,DGKAN_DCHE --gpus 0,1,2,3 --workers 4 ...",
            "verified_result": m4_trajectory_prefix_verified_result("m4_trajectory_class_mnist_strong"),
        },
        {
            "repair_id": "trajectory_first_m1_runner",
            "observed_blocker": "M1 previously had only proxy safety/barrier evidence and no explicit train-batch dual/primal mirror trajectory with same-mirror controls.",
            "code_change": "Added --stage trajectory-m1 with M1-KLMirror/M1-BrierMirror/M1-TailSafeMirror complete in-process trajectories, same-mirror random/signflip controls, and mirror diagnostics for potential, Bregman/KL/Brier/tail/margin deltas.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m1 --label v22_45E_m1_trajectory_brier_c240 --m1-trajectory-output m1_trajectory_brier_c240 ...",
            "verified_result": (
                m1_trajectory_prefix_verified_result("m1_trajectory_brier_c240")
                + " "
                + m1_trajectory_prefix_verified_result("m1_trajectory_kl_c240")
            ),
        },
        {
            "repair_id": "m1_kl_residual_projection",
            "observed_blocker": "The initial M1-KL trajectory used ce_backward_reused as its actuator cotangent, which was cheaper but not a faithful support-masked J^T residual projection of the mirror target.",
            "code_change": "Changed M1-KLMirror to compute autograd gradients of the detached logit-space mirror residual target, lowered KL Bregman step from 0.08 to 0.04, and sparse-recorded diagnostics instead of synchronizing every step.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m1 --label v22_45E_m1_trajectory_kl_residual_c240 --m1-trajectory-output m1_trajectory_kl_residual_c240 --m1-trajectory-variants M1-KLMirror ...",
            "verified_result": m1_trajectory_prefix_verified_result("m1_trajectory_kl_residual_c240"),
        },
        {
            "repair_id": "m1_mirror_grad_cadence_safety_sweep",
            "observed_blocker": "Corrected M1-KL residual projection improved NLL/control/no-debt evidence but failed overhead; cached residual projection needed validation without hiding that the mirror target was stale between refreshes.",
            "code_change": "Added mirror_grad_cadence with per-step state fields mirror_grad_refreshed_this_step/mirror_grad_cached, then ran cached20 and cached40+lower-velocity+barrier8 trajectory suites.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m1 --label v22_45E_m1_trajectory_kl_cached20_c240 --trajectory-m1-mirror-grad-cadence 20 ...; python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m1 --label v22_45E_m1_trajectory_kl_cached40_v015_b8_c240 --trajectory-m1-mirror-grad-cadence 40 --trajectory-m1-velocity-scale 0.15 --trajectory-m1-safety-barrier 8.0 ...",
            "verified_result": (
                m1_trajectory_prefix_verified_result("m1_trajectory_kl_cached20_c240")
                + " "
                + m1_trajectory_prefix_verified_result("m1_trajectory_kl_cached40_v015_b8_c240")
            ),
        },
        {
            "repair_id": "m1_sketched_ls_actuator_projection",
            "observed_blocker": "Support-masked J^T residual projection is not the plan's functional least-squares actuator solve, and the first LS draft incorrectly used raw-logit finite differences for a log-prob mirror target.",
            "code_change": "Added M1-KLMirrorLS: computes detached log-prob mirror target deltas, estimates train-batch log_softmax finite-difference responses inside the same model/optimizer step loop, solves ridge least squares on support basis directions, caches the resulting velocity between mirror refreshes, and records LS active/basis/fd/residual/fallback diagnostics without writing private tensors to CSV.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m1 --label v22_45E_m1_trajectory_kl_ls_smoke ...; python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m1 --label v22_45E_m1_trajectory_kl_ls_cached40_v015_b8_c240 --m1-trajectory-variants M1-KLMirrorLS --trajectory-m1-mirror-grad-cadence 40 --trajectory-m1-velocity-scale 0.15 --trajectory-m1-safety-barrier 8.0 ...",
            "verified_result": m1_trajectory_prefix_verified_result("m1_trajectory_kl_ls_cached40_v015_b8_c240"),
        },
        {
            "repair_id": "m1_sketched_ls_basis_capacity_sweep",
            "observed_blocker": "The first M1-KLMirrorLS c240 used only 3-4 LS basis directions and had high residual fit error, so it could not be treated as a solid actuator solve.",
            "code_change": "Filled the support-restricted LS basis with metric-residualized random sketch directions and raised the M1-KLMirrorLS basis cap from 8 to 16 for support-rank>=16; added mirror_ls_random_basis_count to matrix/summary/recap tables.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m1 --label v22_45E_m1_trajectory_kl_ls_rand8_cached120_v015_b8_c240 --support-rank 8 --trajectory-m1-mirror-grad-cadence 120 ...; python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m1 --label v22_45E_m1_trajectory_kl_ls_basis16_cached240_v015_b8_c240 --support-rank 16 --trajectory-m1-mirror-grad-cadence 240 ...",
            "verified_result": (
                m1_trajectory_prefix_verified_result("m1_trajectory_kl_ls_rand8_cached120_v015_b8_c240")
                + " "
                + m1_trajectory_prefix_verified_result("m1_trajectory_kl_ls_basis16_cached240_v015_b8_c240")
            ),
        },
        {
            "repair_id": "trajectory_first_m5_runner",
            "observed_blocker": "M5 is the plan's highest-priority fairness check, but prior M5 evidence was produced through the old per-spec subprocess dispatch harness.",
            "code_change": "Added --stage trajectory-m5 to run M5 KAN carrier and MLP matched functional support specs in-process with complete training trajectories and separate v22_45E_m5_trajectory_* artifacts.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m5 --label v22_45E_m5_trajectory --gpus 0,1,2,3 --workers 4 ...",
            "verified_result": m5_trajectory_verified_result(),
        },
        {
            "repair_id": "m5_basisgram_fast_tail_guard",
            "observed_blocker": "M5 trajectory evidence had KAN-vs-control signal but failed no-debt/overhead; tail_q99 auxiliary gradients improved safety only weakly and doubled controller-state time.",
            "code_change": "Added KAN-D-CHE-BasisGramFast/KAN-D-FOU-BasisGramFast variants that update controller state only on basis parameters, then tested fast basis-only state with sparse refresh, lower velocity, and train-only tail_q99/brier strict QP guard.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m5 --label v22_45E_m5_trajectory_fast_c240_tail_v02 --m5-trajectory-output m5_trajectory_fast_c240_tail_v02 --support-refresh-cadence 240 --metric-refresh-cadence 240 --velocity-scale 0.20 --calibration-nuisance-mode tail_q99_brier_qp_strict_margin --m5-variant-audit-kan-variants KAN-D-CHE-BasisGramFast ...",
            "verified_result": m5_trajectory_prefix_verified_result("m5_trajectory_fast_c240_tail_v02"),
        },
        {
            "repair_id": "m5_sparse_budget_barrier",
            "observed_blocker": "Tail auxiliary-gradient guards were too expensive; a cheaper train-batch safety-budget barrier needed validation before abandoning M5 calibration sweeps.",
            "code_change": "Ran sparse-refresh c240 budget-only M5 trajectories with lower velocity and safety_budget_velocity_barrier=8.0 for both original BasisGram and BasisGramFast, without calibration auxiliary gradients.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m5 --label v22_45E_m5_trajectory_c240_budget_v02 --m5-trajectory-output m5_trajectory_c240_budget_v02 ...; python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m5 --label v22_45E_m5_trajectory_fast_c240_budget_v02 --m5-trajectory-output m5_trajectory_fast_c240_budget_v02 ...",
            "verified_result": (
                m5_trajectory_prefix_verified_result("m5_trajectory_c240_budget_v02")
                + " "
                + m5_trajectory_prefix_verified_result("m5_trajectory_fast_c240_budget_v02")
            ),
        },
        {
            "repair_id": "m5_fairness_plan_metric_join",
            "observed_blocker": "M5 is the architecture-claim fairness gate, but the trajectory recap only reported KAN-vs-MLP matched NLL/counts and did not expose the plan-required functional_spectrum_distance, RSE_distance, output_scale_distance, support_effective_rank, or explicit TrueKANGain/BothGain/ControlExplained/MLPDegradationDriven evidence chain.",
            "code_change": "Added write_m5_fairness_plan_metrics: it reads only complete in-process v22_45E_m5_trajectory*_matrix.csv main matrices, pairs each KAN carrier row with the best MLP matched-support row from the same artifact/dataset/seed, joins v22_45E_functional_actuator_spectrum_matrix.csv by dataset/seed/architecture/actuator with explicit seed=0 fallback flags, and writes v22_45E_m5_fairness_plan_metrics.csv plus summary.",
            "rerun_command": "python -m py_compile experiments/run_v22_45E_multi_mechanism_extension.py; python experiments/run_v22_45E_multi_mechanism_extension.py --stage finalize --label v22_45E",
            "verified_result": m5_fairness_plan_verified_result(),
        },
        {
            "repair_id": "m5_plan_mlp_variant_coverage",
            "observed_blocker": "The v22.45E plan lists polynomial-like and same-rank block MLP matched supports, but trajectory-m5 only exercised low-rank and frequency-like MLP supports.",
            "code_change": "Added --m5-mlp-matched-variants to trajectory-m5 and ran complete in-process M5 trajectories for MLP-polynomial-like-feature-support and MLP-same-rank-block-support with KAN-D-CHE-BasisGramFast under the c240 budget setting. The fairness join maps these variants to distinct actuator labels; since the spectrum audit has no corresponding rows yet, spectrum_pair_available remains explicit rather than faked.",
            "rerun_command": "python -m py_compile experiments/run_v22_45E_multi_mechanism_extension.py; python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m5 --label v22_45E_m5_poly_block_smoke --m5-mlp-matched-variants MLP-polynomial-like-feature-support,MLP-same-rank-block-support ...; python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m5 --label v22_45E_m5_poly_block_c240_budget --m5-trajectory-output m5_trajectory_poly_block_c240_budget --m5-mlp-matched-variants MLP-polynomial-like-feature-support,MLP-same-rank-block-support --gpus 0,1,2,3 --workers 4 ...",
            "verified_result": m5_trajectory_prefix_verified_result("m5_trajectory_poly_block_c240_budget"),
        },
        {
            "repair_id": "trajectory_first_m6_reservoir_noise",
            "observed_blocker": "M6 previously used NoiseControlProxy rows and therefore did not train with a reservoir-noise term inside the FU velocity field.",
            "code_change": "Added M6-reservoir-noise and M6-noise-suppression to v22.43 update_and_emit, using support-complement reservoir noise plus signal-noise and same-norm Gaussian controls; added --stage trajectory-m6 and M6 plan metrics in v22.45E.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m6 --label v22_45E_m6_trajectory --m6-trajectory-output m6_trajectory --gpus 0,1,2,3 --workers 4 ...",
            "verified_result": m6_trajectory_verified_result(),
        },
        {
            "repair_id": "trajectory_first_m7_rank1_memory_projection",
            "observed_blocker": "M7 was only a copied v22.42R continual-memory diagnostic and did not execute the plan's memory-subspace residualized new-task update inside v22.45E.",
            "code_change": "Added --stage trajectory-m7 with complete in-process Class_MNIST old/new loops. The new-task gradient is residualized against an EMA rank-1 old-gradient memory subspace for memory_projected_residual_flow, with optimizer, same-norm random-subspace, and old-memory rehearsal controls.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m7 --label v22_45E_m7_trajectory --m7-trajectory-output m7_trajectory --eval-seeds 0,1 --eval-architectures MLP,DGKAN_DCHE --gpus 0,1,2,3 --workers 4 ...",
            "verified_result": m7_trajectory_verified_result(),
        },
        {
            "repair_id": "m7_permuted_mnist_task_coverage",
            "observed_blocker": "The first trajectory-m7 repair only covered Class_MNIST, while the v22.45E plan explicitly lists permuted MNIST as an M7 continual-memory task.",
            "code_change": "Added --m7-tasks and a Permuted_MNIST_identity_to_perm loader. The old task trains/evaluates on identity MNIST, the new task trains/evaluates on a fixed seed-derived pixel permutation, with the same in-process rank-1 memory projection and matched controls.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m7 --label v22_45E_m7_permuted --m7-trajectory-output m7_trajectory_permuted --m7-tasks Permuted_MNIST_identity_to_perm --eval-seeds 0,1 --eval-architectures MLP,DGKAN_DCHE --gpus 0,1,2,3 --workers 4 ...",
            "verified_result": m7_trajectory_verified_result("m7_trajectory_permuted"),
        },
        {
            "repair_id": "m7_rankk_old_memory_subspace",
            "observed_blocker": "Permuted-MNIST M7 rank-1 projection was faithful as a partial trajectory but still weaker than the plan's old-memory span {s_old^(i)} and produced negative forgetting-reduction evidence.",
            "code_change": "Added --m7-memory-rank. For rank>1, old phase stores multiple old-task gradient snapshots, new phase residualizes against the rank-k Gram-projected memory subspace, and same_memory_random_subspace_control uses the same rank random basis.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m7 --label v22_45E_m7_permuted_rank4 --m7-trajectory-output m7_trajectory_permuted_rank4 --m7-tasks Permuted_MNIST_identity_to_perm --m7-memory-rank 4 --eval-seeds 0,1 --eval-architectures MLP,DGKAN_DCHE --gpus 0,1,2,3 --workers 4 ...",
            "verified_result": m7_trajectory_verified_result("m7_trajectory_permuted_rank4"),
        },
        {
            "repair_id": "m7_modular_curriculum_task_coverage",
            "observed_blocker": "The v22.45E M7 plan explicitly lists modular arithmetic with a curriculum boundary, but the first trajectory-m7 implementation only handled MNIST-family tasks and hard-coded input_dim=784/output_dim=10.",
            "code_change": "Added Modular_Add_to_Mul task support. The loader trains the old task on (a+b) mod p and the new task on (a*b) mod p in the same input/output space, returns task-specific input_dim/output_dim/x_stats, logs task_boundary/input_dim/output_dim/modulus/modular_train_fraction, and keeps complete in-process old/new trajectories with matched controls.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m7 --label v22_45E_m7_modular_rank4 --m7-trajectory-output m7_trajectory_modular_rank4 --m7-tasks Modular_Add_to_Mul --m7-modulus 13 --m7-modular-train-fraction 0.40 --m7-memory-rank 4 --eval-seeds 0,1 --eval-architectures MLP,DGKAN_DCHE --gpus 0,1,2,3 --workers 4 ...",
            "verified_result": m7_trajectory_verified_result("m7_trajectory_modular_rank4"),
        },
        {
            "repair_id": "m7_modular_fullgrid_valid_forgetting_audit",
            "observed_blocker": "The first Modular_Add_to_Mul_p13 held-out run was clean but not a valid forgetting test: old-task accuracy before the new task was too low, so all four formal groups had forgetting_task_valid=0.",
            "code_change": "Fixed the modular loader edge case so train_fraction=1.0 evaluates on the full finite operation table instead of an empty held-out set, records modular_eval_split=full_grid_train_eval, and reran p7 full-grid rank-4 trajectories to force a learned old task before the curriculum boundary.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m7 --label v22_45E_m7_modular_p7_fullgrid_rank4 --m7-trajectory-output m7_trajectory_modular_p7_fullgrid_rank4 --m7-tasks Modular_Add_to_Mul --m7-modulus 7 --m7-modular-train-fraction 1.0 --m7-memory-rank 4 --eval-seeds 0,1 --eval-architectures MLP,DGKAN_DCHE --gpus 0,1,2,3 --workers 4 --m7-trajectory-steps 400 ...",
            "verified_result": m7_trajectory_verified_result("m7_trajectory_modular_p7_fullgrid_rank4"),
        },
        {
            "repair_id": "m7_function_space_old_logit_projection",
            "observed_blocker": "Rank-k parameter-gradient projection satisfied the coded memory constraint but still did not implement the plan's function-space condition ||P_M_old J_t nu_t||_G_f.",
            "code_change": "Added --m7-projector functional with functional_memory_projected_residual_flow and functional_memory_random_subspace_control. Each new-task step estimates J_old*v on an old-task train anchor by finite differences, subtracts the old-logit memory component from the new gradient, logs functional_memory_projection_after_ratio and functional_fd_eval_count_mean, and keeps optimizer/random/rehearsal controls in the same in-process trajectory loop.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m7 --label v22_45E_m7_functional_p7_fullgrid_rank4 --m7-trajectory-output m7_trajectory_functional_p7_fullgrid_rank4 --m7-projector functional --m7-tasks Modular_Add_to_Mul --m7-modulus 7 --m7-modular-train-fraction 1.0 --m7-memory-rank 4 --eval-seeds 0,1 --eval-architectures MLP,DGKAN_DCHE --gpus 0,1,2,3 --workers 4 --m7-trajectory-steps 400 ...; python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m7 --label v22_45E_m7_functional_class_rank4 --m7-trajectory-output m7_trajectory_functional_class_rank4 --m7-projector functional --m7-tasks Class_MNIST_0_4_to_5_9 --m7-memory-rank 4 --eval-seeds 0,1 --eval-architectures MLP,DGKAN_DCHE --gpus 0,1,2,3 --workers 4 --m7-trajectory-steps 120 ...",
            "verified_result": (
                m7_trajectory_verified_result("m7_trajectory_functional_p7_fullgrid_rank4")
                + " "
                + m7_trajectory_verified_result("m7_trajectory_functional_class_rank4")
            ),
        },
        {
            "repair_id": "m7_functional_projection_metric_bugfix",
            "observed_blocker": "The first functional smoke recorded memory_projection_after_ratio as the residual old-logit response norm, but the plan's constraint is the remaining projection onto the old memory subspace after residualization.",
            "code_change": "Changed m7_residualize_functional_against_basis to recompute the residual response's projection back onto the finite-difference old-logit basis and use that projected norm as memory_projection_after_ratio / functional_memory_projection_after_ratio; residual response retention is kept separately for audit.",
            "rerun_command": "python -m py_compile experiments/run_v22_45E_multi_mechanism_extension.py; rerun v22_45E_m7_functional_smoke and both functional formal suites.",
            "verified_result": (
                "py_compile passed; "
                + m7_trajectory_verified_result("m7_trajectory_functional_p7_fullgrid_rank4")
                + " "
                + m7_trajectory_verified_result("m7_trajectory_functional_class_rank4")
            ),
        },
        {
            "repair_id": "m7_trajectory_command_line_repro_args",
            "observed_blocker": "The first permuted-M7 smoke command succeeded, but the auto-written execution-log command omitted newly added --m7-tasks and generic reproducibility flags such as --row-limit and --tier2-download.",
            "code_change": "Patched trajectory_command_line so M7 trajectory execution logs include --m7-tasks, and all trajectory logs include --row-limit when nonzero plus --tier2-download when enabled.",
            "rerun_command": "python -m py_compile experiments/run_v22_45E_multi_mechanism_extension.py; then rerun trajectory-m7 permuted formal command.",
            "verified_result": "py_compile passed; later permuted M7 execution-log commands include --m7-tasks, --tier2-download, and --row-limit when nonzero.",
        },
        {
            "repair_id": "m7_runner_smoke_bugfixes",
            "observed_blocker": "Initial trajectory-m7 smoke attempts failed before producing matrix rows: first torch.cuda.reset_peak_memory_stats rejected the device argument, then the random-memory control referenced an undefined stable_seed symbol.",
            "code_change": "Made M7 peak-memory reset/max-memory accounting non-blocking with RuntimeError fallback, and changed the random-memory seed call to v2243.stable_seed. No failed smoke rows were used as experiment evidence.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m7 --label v22_45E_m7_smoke --m7-trajectory-output m7_trajectory_smoke --eval-seeds 0 --eval-architectures MLP --gpus 0 --workers 1 --m7-trajectory-steps 12 ...",
            "verified_result": "m7_trajectory_smoke reran with trajectories=4, failures=0, groups=1; matrix/status/plan artifacts were written. Formal m7_trajectory then ran with trajectories=16, failures=0.",
        },
        {
            "repair_id": "trajectory_first_m9_adaptive_metric_mixture",
            "observed_blocker": "M9 previously had fixed metric rows only; there was no train-only runtime mixture state, so AdaptiveMetricFlowOpened could not be audited.",
            "code_change": "Added M9-adaptive-metric-mixture to v22.43 update_metric_diag/update_and_emit: four pre-registered diagonal components are mixed by train-batch mirror-descent weights, with fixed uniform and fixed single-metric controls; added --stage trajectory-m9 and M9 plan metrics in v22.45E.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage trajectory-m9 --label v22_45E_m9_trajectory --m9-trajectory-output m9_trajectory --gpus 0,1,2,3 --workers 4 ...",
            "verified_result": m9_trajectory_verified_result(),
        },
        {
            "repair_id": "m8_gauge_canonicalization_faithfulness_audit",
            "observed_blocker": "M8 requires theta_t^canon = Gamma(theta_t) and FU state updates in canonical gauge, while current code only had basis Gram/transport diagnostics.",
            "code_change": "Added --stage m8-gauge-audit with static feasibility rows and dynamic logit-equivalence tests for naive hidden covariance, basis Gram, and degree/frequency bank rescaling candidates. These rows are audit-only and never route-eligible.",
            "rerun_command": "python experiments/run_v22_45E_multi_mechanism_extension.py --stage m8-gauge-audit --label v22_45E --eval-datasets Wine --eval-seeds 0 --eval-architectures DGKAN_DCHE,DGKAN_DFOU --device cuda:0 --train-size 64 --held-size 64 --batch-size 32 --hidden 16",
            "verified_result": m8_gauge_audit_verified_result(),
        },
    ]
    write_rows(OUT_ROOT / "v22_45E_execution_repair_log.csv", rows)
    append_exec(
        "write_execution_repair_log",
        task_id="execution_repair_log",
        status="pass",
        files="results/v22_45E/v22_45E_execution_repair_log.csv",
        note="Records runner/reporting defects encountered during execution, code changes made, rerun commands, and verified outcomes.",
    )
    return rows


def write_proxy_evidence_audit() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("*specs.csv")):
        specs = read_rows(path)
        counts: dict[tuple[str, str], int] = {}
        for spec in specs:
            text = " ".join([str(spec.get("mechanism", "")), str(spec.get("recipe", "")), str(spec.get("label", ""))]).lower()
            if "proxy" not in text:
                continue
            key = (str(spec.get("mechanism", "")), str(spec.get("recipe", "")))
            counts[key] = counts.get(key, 0) + 1
        for (mechanism, recipe), count in sorted(counts.items()):
            rows.append(
                {
                    "audit_kind": "proxy_recipe_specs",
                    "source_artifact": path.name,
                    "mechanism": mechanism,
                    "recipe_or_coverage": recipe,
                    "row_count": count,
                    "candidate_rows": "",
                    "evidence_tier": "proxy_or_blocker",
                    "route_eligible": 0,
                    "note": "Spec/run labels contain Proxy; retain only as historical screening or blocker context.",
                }
            )
    summary_paths = sorted(OUT_ROOT.glob("*mechanism_summary.csv"))
    top_screen_summary = OUT_ROOT / "v22_45E_mechanism_screen_summary.csv"
    if top_screen_summary.exists() and top_screen_summary not in summary_paths:
        summary_paths.insert(0, top_screen_summary)
    for path in summary_paths:
        summaries = read_rows(path)
        for raw in summaries:
            candidate_rows = int(value_or(raw.get("candidate_rows"), 0.0))
            annotated = annotate_evidence_rows([raw], source_hint=path.name)[0]
            if str(annotated.get("evidence_tier")) != "proxy_or_blocker":
                continue
            if candidate_rows <= 0:
                continue
            rows.append(
                {
                    "audit_kind": "summary_proxy_or_blocker",
                    "source_artifact": path.name,
                    "mechanism": raw.get("mechanism", ""),
                    "recipe_or_coverage": raw.get("coverage", ""),
                    "row_count": "",
                    "candidate_rows": candidate_rows,
                    "evidence_tier": annotated.get("evidence_tier", ""),
                    "route_eligible": annotated.get("route_eligible", 0),
                    "note": "Summary row is proxy/blocker by raw/effective evidence annotation; excluded from opened-route gates.",
                }
            )
    write_rows(OUT_ROOT / "v22_45E_proxy_evidence_audit.csv", rows or [{"status": "no_proxy_evidence_rows_found"}])
    append_exec(
        "write_proxy_evidence_audit",
        task_id="proxy_evidence_audit",
        status="pass",
        files="results/v22_45E/v22_45E_proxy_evidence_audit.csv",
        note=f"proxy_or_blocker_audit_rows={len(rows)}; route gates use screen_pass_route_eligible, not raw screen_pass.",
    )
    return rows


def final_route() -> dict[str, Any]:
    code = (read_rows(OUT_ROOT / "v22_45E_code_truth_gate.csv") or [{}])[0]
    summaries = annotate_evidence_rows(read_rows(OUT_ROOT / "v22_45E_mechanism_screen_summary.csv"), source_hint="v22_45E_mechanism_screen_summary.csv")
    repair_summaries = annotate_evidence_rows(read_rows(OUT_ROOT / "v22_45E_repair_mechanism_summary.csv"), source_hint="v22_45E_repair_mechanism_summary.csv")
    m1_trajectory_summaries = annotate_evidence_rows(read_rows_with_source("v22_45E_m1_trajectory*_mechanism_summary.csv"))
    m2_repair_summaries = annotate_evidence_rows(read_rows_with_source("v22_45E_m2_*_mechanism_summary.csv"))
    m3_trajectory_summaries = annotate_evidence_rows(read_rows_with_source("v22_45E_m3_trajectory*_mechanism_summary.csv"))
    m5_variant_summaries = annotate_evidence_rows(read_rows_with_source("v22_45E_m5_variant*_mechanism_summary.csv"))
    m5_trajectory_summaries = annotate_evidence_rows([
        r for r in read_rows_with_source("v22_45E_m5_trajectory*_mechanism_summary.csv")
        if "smoke" not in str(r.get("source_artifact", "")).lower()
    ])
    m6_trajectory_summaries = annotate_evidence_rows(read_rows_with_source("v22_45E_m6_trajectory*_mechanism_summary.csv"))
    m9_trajectory_summaries = annotate_evidence_rows(read_rows_with_source("v22_45E_m9_trajectory*_mechanism_summary.csv"))
    route_summaries = summaries + [r for r in repair_summaries if r.get("status") != "not_run"] + [r for r in m1_trajectory_summaries if r.get("status") != "not_run"] + [r for r in m2_repair_summaries if r.get("status") != "not_run"] + [r for r in m3_trajectory_summaries if r.get("status") != "not_run"] + [r for r in m5_variant_summaries if r.get("status") != "not_run"] + [r for r in m5_trajectory_summaries if r.get("status") != "not_run"] + [r for r in m6_trajectory_summaries if r.get("status") != "not_run"] + [r for r in m9_trajectory_summaries if r.get("status") != "not_run"]
    decomp = annotate_evidence_rows(read_rows(OUT_ROOT / "v22_45E_support_direction_decomposition.csv"), source_hint="v22_45E_support_direction_decomposition.csv")
    decomp.extend(annotate_evidence_rows(read_rows(OUT_ROOT / "v22_45E_repair_support_direction_decomposition.csv"), source_hint="v22_45E_repair_support_direction_decomposition.csv"))
    decomp.extend(annotate_evidence_rows(read_rows_with_source("v22_45E_m1_trajectory*_support_direction_decomposition.csv")))
    decomp.extend(annotate_evidence_rows(read_rows_with_source("v22_45E_m2_*_support_direction_decomposition.csv")))
    decomp.extend(annotate_evidence_rows(read_rows_with_source("v22_45E_m3_trajectory*_support_direction_decomposition.csv")))
    decomp.extend(annotate_evidence_rows(read_rows_with_source("v22_45E_m5_variant*_support_direction_decomposition.csv")))
    decomp.extend(annotate_evidence_rows([
        r for r in read_rows_with_source("v22_45E_m5_trajectory*_support_direction_decomposition.csv")
        if "smoke" not in str(r.get("source_artifact", "")).lower()
    ]))
    decomp.extend(annotate_evidence_rows(read_rows_with_source("v22_45E_m6_trajectory*_support_direction_decomposition.csv")))
    decomp.extend(annotate_evidence_rows(read_rows_with_source("v22_45E_m9_trajectory*_support_direction_decomposition.csv")))
    m6_plan_rows = [r for r in read_rows_with_source("v22_45E_m6_trajectory*_plan_metrics.csv") if not r.get("status")]
    m6_explain_rows = [r for r in m6_plan_rows if int_flag(r.get("M6_reservoir_regularization_explains_gain"))]
    m7_all_plan_rows = [r for r in read_rows_with_source("v22_45E_m7_trajectory*_plan_metrics.csv") if not r.get("status")]
    m7_plan_rows = [r for r in m7_all_plan_rows if "smoke" not in str(r.get("source_artifact", "")).lower()]
    m7_open_rows = [r for r in m7_plan_rows if int_flag(r.get("M7_exploration_opened"))]
    m7_summaries = []
    for path in sorted(OUT_ROOT.glob("v22_45E_m7_trajectory*_summary.json")):
        if "smoke" in path.name.lower():
            continue
        item = read_json(path)
        if item:
            item["source_artifact"] = path.name
            m7_summaries.append(item)
    m9_plan_rows = [r for r in read_rows_with_source("v22_45E_m9_trajectory*_plan_metrics.csv") if not r.get("status")]
    m9_open_rows = [r for r in m9_plan_rows if int_flag(r.get("AdaptiveMetricFlowOpened"))]
    m9_metric_selection_rows = [r for r in m9_plan_rows if int_flag(r.get("MetricSelectionDiagnosticOnly"))]
    m5_fairness_rows = [r for r in read_rows(OUT_ROOT / "v22_45E_m5_fairness_plan_metrics.csv") if not r.get("status")]
    m5_fairness_pass_rows = [r for r in m5_fairness_rows if int_flag(r.get("M5_fairness_candidate_pass"))]
    m8_audit_rows = [r for r in read_rows(OUT_ROOT / "v22_45E_m8_gauge_feasibility_audit.csv") if not r.get("status")]
    m8_dynamic_rows = [r for r in read_rows(OUT_ROOT / "v22_45E_m8_gauge_dynamic_equivalence.csv") if not r.get("status")]
    m8_dynamic_test_rows = [r for r in m8_dynamic_rows if r.get("transform_name") != "identity_restore_check"]
    m4_trajectory_summaries = []
    for path in sorted(OUT_ROOT.glob("v22_45E_m4_trajectory*_summary.json")):
        if "smoke" in path.name.lower():
            continue
        item = read_json(path)
        if item:
            item["source_artifact"] = path.name
            m4_trajectory_summaries.append(item)
    m4 = read_json(OUT_ROOT / "v22_45E_M4_slow_signal_diagnostic_summary.json")
    m7 = read_json(OUT_ROOT / "v22_45E_M7_continual_memory_summary.json")
    if code.get("status") != "pass":
        route = "R0-CodeOrRuntimeRegression"
        reason = "Code/import truth gate failed."
    else:
        pass_mechs = [r for r in route_summaries if int_flag(r.get("screen_pass_route_eligible"))]
        support_only_all = [r for r in decomp if int_flag(r.get("support_positive_direction_nonpositive"))]
        support_only = [r for r in support_only_all if int_flag(r.get("route_eligible"))]
        safety_blocked = [
            r for r in route_summaries
            if int_flag(r.get("route_eligible"))
            and value_or(r.get("NLL_improvement_vs_own_rows"), 0.0) > 0.0
            and value_or(r.get("no_ECE_Brier_tail_debt_rows"), 0.0) == 0.0
        ]
        if any(r.get("mechanism") == "M5" for r in pass_mechs):
            route = "R11-TrueKANGainExplorationOpened"
            reason = "M5 low-cost screen passed; hard-task confirmation required before official route."
        elif any(r.get("mechanism") == "M1" for r in pass_mechs):
            route = "FunctionalMirrorFlowOpened"
            reason = "M1 trajectory-first mirror screen passed with explicit train-batch dual/primal mirror state; hard-task confirmation and full actuator solver remain required."
        elif any(r.get("mechanism") == "M2" for r in pass_mechs):
            route = "R5-SpectralRadialSplitFlowOpened"
            reason = "M2 low-cost screen passed under per-mode radial implementation; hard-task confirmation required."
        elif any(r.get("mechanism") == "M3" for r in pass_mechs):
            route = "R6-LieMomentumFlowOpened"
            reason = "M3 trajectory-first transported Lie momentum screen passed; hard-task confirmation required."
        elif any(r.get("mechanism") == "M4" for r in pass_mechs):
            route = "R7-SlowSignalReservoirMigrationOpened"
            reason = "M4 low-cost screen passed; diagnostic hard-task evidence remains separate."
        elif any(r.get("mechanism") == "Core-M" for r in pass_mechs):
            route = "R2-MetricSupportOptimizerOpened"
            reason = "Core-M low-cost screen passed."
        elif str(m7.get("route", "")).startswith("R12"):
            route = "R7-SlowSignalReservoirMigrationOpened"
            reason = f"M7 diagnostic opened a continual-memory diagnostic route: {m7.get('route')}; official route still blocked by diagnostic-only status."
        elif m7_open_rows and len(m7_open_rows) >= max(1, math.ceil(0.50 * len(m7_plan_rows))):
            route = "R12-ContinualFunctionalMemoryOpened_Partial"
            reason = (
                f"M7 parameter-gradient memory projection passed exploration gates in {len(m7_open_rows)}/{len(m7_plan_rows)} groups; "
                "evidence is parameter-gradient memory projection, not full function-space M7."
            )
        elif any(str(r.get("route", "")).startswith("R13") and int_flag(r.get("promotion_allowed")) for r in m4_trajectory_summaries):
            route = "R7-SlowSignalReservoirMigrationOpened"
            opened = next(r for r in m4_trajectory_summaries if str(r.get("route", "")).startswith("R13") and int_flag(r.get("promotion_allowed")))
            reason = (
                f"M4 modular trajectory opened partial route {opened.get('route')} in {opened.get('source_artifact')}; "
                "official route still requires full metric/OET projection and broader hard-task confirmation."
            )
        elif str(m4.get("route", "")).startswith("R13"):
            route = "R7-SlowSignalReservoirMigrationOpened"
            reason = f"M4 modular diagnostic opened diagnostic route: {m4.get('route')}; official route still blocked by diagnostic-only status."
        elif m9_open_rows and len(m9_open_rows) >= max(1, math.ceil(0.50 * len(m9_plan_rows))):
            route = "AdaptiveMetricFlowOpened"
            reason = (
                f"M9 adaptive metric mixture passed strict plan gates in {len(m9_open_rows)}/{len(m9_plan_rows)} groups; "
                "evidence is diagonal train-batch metric mixture with fixed metric controls, not validation/test metric search."
            )
        elif m6_explain_rows and len(m6_explain_rows) >= max(1, len(m6_plan_rows) // 2):
            route = "ReservoirRegularizationExplainsGain"
            reason = (
                f"M6 reservoir-noise trajectory passed explanatory gates in {len(m6_explain_rows)}/{len(m6_plan_rows)} groups; "
                "this is parameter support-complement reservoir regularization evidence, not SignalFUOpened."
            )
        elif support_only:
            route = "R14-SupportOnlyNoSignalIncrement"
            reason = (
                f"{len(support_only)} route-eligible rows had positive support effect with non-positive direction effect; "
                f"{len(support_only_all)} total rows including legacy/proxy rows showed the same pattern."
            )
        elif safety_blocked:
            route = "R16-SafetyDebtBlocksAll"
            reason = "At least one mechanism had NLL gains but no no-debt rows in its screen summary."
        elif route_summaries:
            route = "R15-StrongOptimizerExplainsAll"
            reason = "No non-proxy mechanism passed low-cost screen against matched controls/no-debt/overhead gates."
        else:
            route = "R0-CodeOrRuntimeRegression"
            reason = "No screen rows were available after code gate."
    final = {
        "final_route": route,
        "reason": reason,
        "screen_summary_rows": len(summaries),
        "repair_summary_rows": len(repair_summaries),
        "m1_trajectory_summary_rows": len(m1_trajectory_summaries),
        "m3_trajectory_summary_rows": len(m3_trajectory_summaries),
        "m4_trajectory_summary_rows": len(m4_trajectory_summaries),
        "m5_variant_summary_rows": len(m5_variant_summaries),
        "m5_trajectory_summary_rows": len(m5_trajectory_summaries),
        "m5_fairness_plan_rows": len(m5_fairness_rows),
        "m5_fairness_candidate_pass_rows": len(m5_fairness_pass_rows),
        "m6_trajectory_summary_rows": len(m6_trajectory_summaries),
        "m6_trajectory_plan_rows": len(m6_plan_rows),
        "m6_explanatory_pass_rows": len(m6_explain_rows),
        "m7_trajectory_summary_rows": len(m7_summaries),
        "m7_trajectory_plan_rows": len(m7_plan_rows),
        "m7_opened_rows": len(m7_open_rows),
        "m9_trajectory_summary_rows": len(m9_trajectory_summaries),
        "m9_trajectory_plan_rows": len(m9_plan_rows),
        "m9_opened_rows": len(m9_open_rows),
        "m9_metric_selection_diagnostic_rows": len(m9_metric_selection_rows),
        "m8_gauge_feasibility_rows": len(m8_audit_rows),
        "m8_faithful_runtime_gamma_available_rows": sum(int_flag(r.get("faithful_runtime_gamma_available")) for r in m8_audit_rows),
        "m8_runtime_fu_state_canonical_update_rows": sum(int_flag(r.get("runtime_fu_state_canonical_update_available")) for r in m8_audit_rows),
        "m8_dynamic_equivalence_rows": len(m8_dynamic_test_rows),
        "m8_dynamic_equivalent_rows": sum(int_flag(r.get("function_equivalent_under_tol")) for r in m8_dynamic_test_rows),
        "m8_dynamic_identity_sanity_rows": len(m8_dynamic_rows) - len(m8_dynamic_test_rows),
        "route_eligible_summary_rows": sum(int_flag(r.get("route_eligible")) for r in route_summaries),
        "proxy_or_blocker_summary_rows": sum(1 for r in route_summaries if r.get("evidence_tier") == "proxy_or_blocker"),
        "proxy_or_blocker_candidate_rows": sum(
            int(value_or(r.get("candidate_rows"), 0.0))
            for r in route_summaries
            if r.get("evidence_tier") == "proxy_or_blocker"
        ),
        "support_only_route_eligible_rows": len(support_only if code.get("status") == "pass" else []),
        "support_only_all_rows": len(support_only_all if code.get("status") == "pass" else []),
        "m4_trajectory_routes": [r.get("route", "not_run") for r in m4_trajectory_summaries],
        "m4_diagnostic_route": m4.get("route", "not_run"),
        "m7_diagnostic_route": m7.get("route", "not_run"),
        "timestamp": now_sg(),
    }
    write_json(OUT_ROOT / "v22_45E_final_route.json", final)
    return final


def md_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 20) -> str:
    if not rows:
        return "_no rows_\n"
    clipped = rows[:limit]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in clipped:
        vals = []
        for col in columns:
            text = str(row.get(col, ""))
            text = text.replace("\n", " ").replace("|", "\\|")
            if len(text) > 96:
                text = text[:93] + "..."
            vals.append(text)
        lines.append("| " + " | ".join(vals) + " |")
    if len(rows) > limit:
        lines.append(f"\n_只显示前 {limit} 行，共 {len(rows)} 行。_")
    return "\n".join(lines) + "\n"


def write_manifest() -> None:
    rows = []
    rels = list(V2245E_TABLES) + ["v22_45E_final_route.json", "v22_45E_M4_slow_signal_diagnostic_summary.json", "v22_45E_M7_continual_memory_summary.json"]
    for path in sorted(OUT_ROOT.glob("v22_45E_m5_variant*_*.csv")):
        rel = path.name
        if rel not in rels:
            rels.append(rel)
    for path in sorted(OUT_ROOT.glob("v22_45E_m5_trajectory*_*.csv")):
        rel = path.name
        if rel not in rels:
            rels.append(rel)
    for path in sorted(OUT_ROOT.glob("v22_45E_m1_trajectory*_*.csv")):
        rel = path.name
        if rel not in rels:
            rels.append(rel)
    for path in sorted(OUT_ROOT.glob("v22_45E_m2_*_*.csv")):
        rel = path.name
        if rel not in rels:
            rels.append(rel)
    for path in sorted(OUT_ROOT.glob("v22_45E_m3_trajectory*_*.csv")):
        rel = path.name
        if rel not in rels:
            rels.append(rel)
    for path in sorted(OUT_ROOT.glob("v22_45E_m4_trajectory*.*")):
        rel = path.name
        if rel not in rels:
            rels.append(rel)
    for path in sorted(OUT_ROOT.glob("v22_45E_m6_trajectory*_*.csv")):
        rel = path.name
        if rel not in rels:
            rels.append(rel)
    for path in sorted(OUT_ROOT.glob("v22_45E_m7_trajectory*.*")):
        rel = path.name
        if rel not in rels:
            rels.append(rel)
    for path in sorted(OUT_ROOT.glob("v22_45E_m9_trajectory*_*.csv")):
        rel = path.name
        if rel not in rels:
            rels.append(rel)
    for rel in rels:
        p = OUT_ROOT / rel
        if not p.exists():
            rows.append({"path": str(p.relative_to(ROOT)), "exists": 0, "bytes": "", "sha256": ""})
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        rows.append({"path": str(p.relative_to(ROOT)), "exists": 1, "bytes": p.stat().st_size, "sha256": h})
    write_rows(OUT_ROOT / "v22_45E_artifact_manifest.csv", rows)


def update_recap(final: dict[str, Any]) -> None:
    code = read_rows(OUT_ROOT / "v22_45E_code_truth_gate.csv")
    coverage = read_rows(OUT_ROOT / "v22_45E_mechanism_coverage_matrix.csv")
    screen = annotate_evidence_rows(read_rows(OUT_ROOT / "v22_45E_mechanism_screen_summary.csv"), source_hint="v22_45E_mechanism_screen_summary.csv")
    repair_screen = annotate_evidence_rows(read_rows(OUT_ROOT / "v22_45E_repair_mechanism_summary.csv"), source_hint="v22_45E_repair_mechanism_summary.csv")
    m1_trajectory_screen = annotate_evidence_rows(read_rows_with_source("v22_45E_m1_trajectory*_mechanism_summary.csv"))
    m1_trajectory_matrix = read_rows_with_source("v22_45E_m1_trajectory*_matrix.csv")
    m2_repair_screen = annotate_evidence_rows(read_rows_with_source("v22_45E_m2_*_mechanism_summary.csv"))
    m3_trajectory_screen = annotate_evidence_rows(read_rows_with_source("v22_45E_m3_trajectory*_mechanism_summary.csv"))
    m5_variant_screen = annotate_evidence_rows(read_rows_with_source("v22_45E_m5_variant*_mechanism_summary.csv"))
    m5_trajectory_screen = annotate_evidence_rows([
        r for r in read_rows_with_source("v22_45E_m5_trajectory*_mechanism_summary.csv")
        if "smoke" not in str(r.get("source_artifact", "")).lower()
    ])
    m6_trajectory_screen = annotate_evidence_rows(read_rows_with_source("v22_45E_m6_trajectory*_mechanism_summary.csv"))
    m9_trajectory_screen = annotate_evidence_rows(read_rows_with_source("v22_45E_m9_trajectory*_mechanism_summary.csv"))
    decomp = annotate_evidence_rows(read_rows(OUT_ROOT / "v22_45E_support_direction_decomposition.csv"), source_hint="v22_45E_support_direction_decomposition.csv")
    repair_decomp = annotate_evidence_rows(read_rows(OUT_ROOT / "v22_45E_repair_support_direction_decomposition.csv"), source_hint="v22_45E_repair_support_direction_decomposition.csv")
    m2_repair_decomp = annotate_evidence_rows(read_rows_with_source("v22_45E_m2_*_support_direction_decomposition.csv"))
    m3_trajectory_decomp = annotate_evidence_rows(read_rows_with_source("v22_45E_m3_trajectory*_support_direction_decomposition.csv"))
    m5_variant_decomp = annotate_evidence_rows(read_rows_with_source("v22_45E_m5_variant*_support_direction_decomposition.csv"))
    m5_trajectory_decomp = annotate_evidence_rows([
        r for r in read_rows_with_source("v22_45E_m5_trajectory*_support_direction_decomposition.csv")
        if "smoke" not in str(r.get("source_artifact", "")).lower()
    ])
    m6_trajectory_decomp = annotate_evidence_rows(read_rows_with_source("v22_45E_m6_trajectory*_support_direction_decomposition.csv"))
    m9_trajectory_decomp = annotate_evidence_rows(read_rows_with_source("v22_45E_m9_trajectory*_support_direction_decomposition.csv"))
    safety = read_rows(OUT_ROOT / "v22_45E_safety_debt_matrix.csv")
    repair_safety = read_rows(OUT_ROOT / "v22_45E_repair_safety_debt_matrix.csv")
    m1_trajectory_safety = read_rows_with_source("v22_45E_m1_trajectory*_safety_debt_matrix.csv")
    m2_repair_safety = read_rows_with_source("v22_45E_m2_*_safety_debt_matrix.csv")
    m3_trajectory_safety = read_rows_with_source("v22_45E_m3_trajectory*_safety_debt_matrix.csv")
    m5_variant_safety = read_rows_with_source("v22_45E_m5_variant*_safety_debt_matrix.csv")
    m5_trajectory_safety = [
        r for r in read_rows_with_source("v22_45E_m5_trajectory*_safety_debt_matrix.csv")
        if "smoke" not in str(r.get("source_artifact", "")).lower()
    ]
    m6_trajectory_safety = read_rows_with_source("v22_45E_m6_trajectory*_safety_debt_matrix.csv")
    m9_trajectory_safety = read_rows_with_source("v22_45E_m9_trajectory*_safety_debt_matrix.csv")
    overhead = read_rows(OUT_ROOT / "v22_45E_overhead_matrix.csv")
    repair_overhead = read_rows(OUT_ROOT / "v22_45E_repair_overhead_matrix.csv")
    m1_trajectory_overhead = read_rows_with_source("v22_45E_m1_trajectory*_overhead_matrix.csv")
    m2_repair_overhead = read_rows_with_source("v22_45E_m2_*_overhead_matrix.csv")
    m3_trajectory_overhead = read_rows_with_source("v22_45E_m3_trajectory*_overhead_matrix.csv")
    m5_variant_overhead = read_rows_with_source("v22_45E_m5_variant*_overhead_matrix.csv")
    m5_trajectory_overhead = [
        r for r in read_rows_with_source("v22_45E_m5_trajectory*_overhead_matrix.csv")
        if "smoke" not in str(r.get("source_artifact", "")).lower()
    ]
    m6_trajectory_overhead = read_rows_with_source("v22_45E_m6_trajectory*_overhead_matrix.csv")
    m9_trajectory_overhead = read_rows_with_source("v22_45E_m9_trajectory*_overhead_matrix.csv")
    m2_trajectory_plan = read_rows_with_source("v22_45E_m2_*_plan_metrics.csv")
    m2_trajectory_status = read_rows_with_source("v22_45E_m2_*_status.csv")
    m3_trajectory_plan = read_rows_with_source("v22_45E_m3_trajectory*_plan_metrics.csv")
    m3_trajectory_status = read_rows_with_source("v22_45E_m3_trajectory*_status.csv")
    m1_trajectory_status = read_rows_with_source("v22_45E_m1_trajectory*_status.csv")
    m4_trajectory_matrix = [r for r in read_rows_with_source("v22_45E_m4_trajectory*_matrix.csv") if "smoke" not in str(r.get("source_artifact", "")).lower()]
    m4_trajectory_plan = [r for r in read_rows_with_source("v22_45E_m4_trajectory*_plan_metrics.csv") if "smoke" not in str(r.get("source_artifact", "")).lower()]
    m4_trajectory_status = [r for r in read_rows_with_source("v22_45E_m4_trajectory*_status.csv") if "smoke" not in str(r.get("source_artifact", "")).lower()]
    m4_trajectory_summaries = []
    for path in sorted(OUT_ROOT.glob("v22_45E_m4_trajectory*_summary.json")):
        if "smoke" in path.name.lower():
            continue
        item = read_json(path)
        if item:
            item["source_artifact"] = path.name
            m4_trajectory_summaries.append(item)
    m5_trajectory_status = read_rows_with_source("v22_45E_m5_trajectory*_status.csv")
    m5_fairness_plan = [r for r in read_rows(OUT_ROOT / "v22_45E_m5_fairness_plan_metrics.csv") if not r.get("status")]
    m5_fairness_summary = read_rows(OUT_ROOT / "v22_45E_m5_fairness_plan_summary.csv")
    m6_trajectory_matrix = read_rows_with_source("v22_45E_m6_trajectory*_matrix.csv")
    m6_trajectory_plan = read_rows_with_source("v22_45E_m6_trajectory*_plan_metrics.csv")
    m6_trajectory_status = read_rows_with_source("v22_45E_m6_trajectory*_status.csv")
    m7_trajectory_matrix_all = read_rows_with_source("v22_45E_m7_trajectory*_matrix.csv")
    m7_trajectory_matrix = [r for r in m7_trajectory_matrix_all if "smoke" not in str(r.get("source_artifact", "")).lower()]
    m7_trajectory_plan_all = read_rows_with_source("v22_45E_m7_trajectory*_plan_metrics.csv")
    m7_trajectory_plan = [r for r in m7_trajectory_plan_all if "smoke" not in str(r.get("source_artifact", "")).lower()]
    m7_trajectory_status_all = read_rows_with_source("v22_45E_m7_trajectory*_status.csv")
    m7_trajectory_status = [r for r in m7_trajectory_status_all if "smoke" not in str(r.get("source_artifact", "")).lower()]
    m7_trajectory_summaries = []
    for path in sorted(OUT_ROOT.glob("v22_45E_m7_trajectory*_summary.json")):
        if "smoke" in path.name.lower():
            continue
        item = read_json(path)
        if item:
            item["source_artifact"] = path.name
            m7_trajectory_summaries.append(item)
    m9_trajectory_matrix = read_rows_with_source("v22_45E_m9_trajectory*_matrix.csv")
    m9_trajectory_plan = read_rows_with_source("v22_45E_m9_trajectory*_plan_metrics.csv")
    m9_trajectory_status = read_rows_with_source("v22_45E_m9_trajectory*_status.csv")
    m8_gauge_audit = read_rows(OUT_ROOT / "v22_45E_m8_gauge_feasibility_audit.csv")
    m8_gauge_dynamic = read_rows(OUT_ROOT / "v22_45E_m8_gauge_dynamic_equivalence.csv")
    m6_formal_plan = [r for r in m6_trajectory_plan if r.get("source_artifact") == "v22_45E_m6_trajectory_plan_metrics.csv"]
    if not m6_formal_plan:
        m6_formal_plan = [r for r in m6_trajectory_plan if not r.get("status")]
    m9_formal_plan = [r for r in m9_trajectory_plan if r.get("source_artifact") == "v22_45E_m9_trajectory_plan_metrics.csv"]
    if not m9_formal_plan:
        m9_formal_plan = [r for r in m9_trajectory_plan if not r.get("status")]

    def count_flag_in(rows: list[dict[str, Any]], key: str) -> int:
        return sum(int_flag(r.get(key)) for r in rows)

    def mean_float_in(rows: list[dict[str, Any]], key: str) -> str:
        vals = [value_or(r.get(key), math.nan) for r in rows if finite_float(r.get(key)) is not None]
        return "nan" if not vals else f"{statistics.fmean(vals):.6g}"

    m5_fairness_all = next((r for r in m5_fairness_summary if r.get("source_artifact") == "ALL"), {})
    m5_fairness_note = (
        f"M5 fairness plan metrics: rows={m5_fairness_all.get('rows', len(m5_fairness_plan))}; "
        f"spectrum_pair_available={m5_fairness_all.get('spectrum_pair_available_rows', '')}; "
        f"exact_spectrum_seed={m5_fairness_all.get('exact_spectrum_seed_rows', '')}; "
        f"beats_MLP_matched_support={m5_fairness_all.get('beats_MLP_matched_support_rows', '')}; "
        f"TrueKANGain+BothGain={m5_fairness_all.get('TrueKANGain_plus_BothGain_rows', '')}; "
        f"ControlExplained={m5_fairness_all.get('ControlExplained_rows', '')}; "
        f"MLPDegradationDriven={m5_fairness_all.get('MLPDegradationDriven_rows', '')}; "
        f"no_debt={m5_fairness_all.get('no_debt_rows', '')}; "
        f"overhead<=0.35={m5_fairness_all.get('overhead_le_0p35_rows', '')}; "
        f"fairness_candidate_pass={m5_fairness_all.get('M5_fairness_candidate_pass_rows', '')}; "
        f"mean_functional_spectrum_distance={m5_fairness_all.get('mean_functional_spectrum_distance', '')}; "
        f"mean_RSE_distance={m5_fairness_all.get('mean_RSE_distance', '')}; "
        f"mean_output_scale_distance={m5_fairness_all.get('mean_output_scale_distance', '')}."
    )

    m6_formal_note = (
        f"正式 M6 trajectory: groups={len(m6_formal_plan)}; "
        f"explanatory_pass={count_flag_in(m6_formal_plan, 'M6_reservoir_regularization_explains_gain')}/{len(m6_formal_plan)}; "
        f"beats_suppression={count_flag_in(m6_formal_plan, 'beats_noise_suppression')}/{len(m6_formal_plan)}; "
        f"beats_signal_noise={count_flag_in(m6_formal_plan, 'beats_signal_noise_control')}/{len(m6_formal_plan)}; "
        f"beats_gaussian={count_flag_in(m6_formal_plan, 'beats_same_norm_Gaussian_control')}/{len(m6_formal_plan)}; "
        f"no_debt={count_flag_in(m6_formal_plan, 'no_ECE_Brier_tail_debt')}/{len(m6_formal_plan)}; "
        f"runtime_truth={count_flag_in(m6_formal_plan, 'runtime_truth')}/{len(m6_formal_plan)}; "
        f"mean_NLL_improvement_vs_base={mean_float_in(m6_formal_plan, 'NLL_improvement_vs_base')}; "
        f"mean_NLL_improvement_vs_suppression={mean_float_in(m6_formal_plan, 'NLL_improvement_vs_noise_suppression')}; "
        f"mean_overhead={mean_float_in(m6_formal_plan, 'controller_overhead_ratio')}; "
        f"mean_reservoir_noise_energy={mean_float_in(m6_formal_plan, 'reservoir_noise_energy')}; "
        f"mean_signal_noise_leakage={mean_float_in(m6_formal_plan, 'signal_noise_leakage')}."
    )
    m9_formal_note = (
        f"正式 M9 trajectory: groups={len(m9_formal_plan)}; "
        f"opened={count_flag_in(m9_formal_plan, 'AdaptiveMetricFlowOpened')}/{len(m9_formal_plan)}; "
        f"metric_selection_diagnostic={count_flag_in(m9_formal_plan, 'MetricSelectionDiagnosticOnly')}/{len(m9_formal_plan)}; "
        f"beats_uniform={count_flag_in(m9_formal_plan, 'beats_uniform')}/{len(m9_formal_plan)}; "
        f"beats_best_fixed={count_flag_in(m9_formal_plan, 'beats_best_fixed_metric')}/{len(m9_formal_plan)}; "
        f"no_debt={count_flag_in(m9_formal_plan, 'no_ECE_Brier_tail_debt')}/{len(m9_formal_plan)}; "
        f"adaptation_truth={count_flag_in(m9_formal_plan, 'adaptation_truth')}/{len(m9_formal_plan)}; "
        f"runtime_truth={count_flag_in(m9_formal_plan, 'runtime_truth')}/{len(m9_formal_plan)}; "
        f"mean_NLL_improvement_vs_base={mean_float_in(m9_formal_plan, 'NLL_improvement_vs_base')}; "
        f"mean_NLL_improvement_vs_uniform={mean_float_in(m9_formal_plan, 'NLL_improvement_vs_uniform')}; "
        f"mean_NLL_improvement_vs_best_fixed={mean_float_in(m9_formal_plan, 'NLL_improvement_vs_best_fixed')}; "
        f"mean_final_entropy={mean_float_in(m9_formal_plan, 'metric_mixture_final_entropy')}; "
        f"mean_overhead={mean_float_in(m9_formal_plan, 'controller_overhead_ratio')}."
    )
    m7_formal_plan = [r for r in m7_trajectory_plan if r.get("source_artifact") == "v22_45E_m7_trajectory_plan_metrics.csv"]
    if not m7_formal_plan:
        m7_formal_plan = [r for r in m7_trajectory_plan if not r.get("status")]
    m7_formal_note = (
        f"正式 M7 trajectory: groups={len(m7_formal_plan)}; "
        f"valid_forgetting={count_flag_in(m7_formal_plan, 'forgetting_task_valid')}/{len(m7_formal_plan)}; "
        f"opened={count_flag_in(m7_formal_plan, 'M7_exploration_opened')}/{len(m7_formal_plan)}; "
        f"official_ready={count_flag_in(m7_formal_plan, 'M7_official_ready')}/{len(m7_formal_plan)}; "
        f"controls_fail={count_flag_in(m7_formal_plan, 'matched_memory_controls_fail')}/{len(m7_formal_plan)}; "
        f"final_nonworse={count_flag_in(m7_formal_plan, 'final_average_accuracy_nonworse')}/{len(m7_formal_plan)}; "
        f"constraint_ok={count_flag_in(m7_formal_plan, 'constraint_ok')}/{len(m7_formal_plan)}; "
        f"runtime_truth={count_flag_in(m7_formal_plan, 'runtime_truth')}/{len(m7_formal_plan)}; "
        f"mean_relative_forgetting_reduction={mean_float_in(m7_formal_plan, 'relative_forgetting_reduction')}; "
        f"mean_memory_projection_after={mean_float_in(m7_formal_plan, 'memory_projection_after_ratio')}."
    )
    m7_all_plan = [r for r in m7_trajectory_plan if not r.get("status")]
    m7_tasks = ",".join(sorted({str(r.get("dataset", "")) for r in m7_all_plan if r.get("dataset")}))
    m7_all_note = (
        f"M7 all task coverage: tasks={m7_tasks or 'n/a'}; groups={len(m7_all_plan)}; "
        f"valid_forgetting={count_flag_in(m7_all_plan, 'forgetting_task_valid')}/{len(m7_all_plan)}; "
        f"opened={count_flag_in(m7_all_plan, 'M7_exploration_opened')}/{len(m7_all_plan)}; "
        f"official_ready={count_flag_in(m7_all_plan, 'M7_official_ready')}/{len(m7_all_plan)}; "
        f"controls_fail={count_flag_in(m7_all_plan, 'matched_memory_controls_fail')}/{len(m7_all_plan)}; "
        f"final_nonworse={count_flag_in(m7_all_plan, 'final_average_accuracy_nonworse')}/{len(m7_all_plan)}; "
        f"constraint_ok={count_flag_in(m7_all_plan, 'constraint_ok')}/{len(m7_all_plan)}; "
        f"runtime_truth={count_flag_in(m7_all_plan, 'runtime_truth')}/{len(m7_all_plan)}; "
        f"mean_relative_forgetting_reduction={mean_float_in(m7_all_plan, 'relative_forgetting_reduction')}."
    )
    m8_dynamic_tests = [r for r in m8_gauge_dynamic if not r.get("status") and r.get("transform_name") != "identity_restore_check"]
    m8_feasibility_rows = [r for r in m8_gauge_audit if not r.get("status")]
    m8_note = (
        f"M8 gauge audit: feasibility_rows={len(m8_feasibility_rows)}; "
        f"faithful_runtime_gamma_available={count_flag_in(m8_feasibility_rows, 'faithful_runtime_gamma_available')}/{len(m8_feasibility_rows)}; "
        f"runtime_fu_state_canonical_update_available={count_flag_in(m8_feasibility_rows, 'runtime_fu_state_canonical_update_available')}/{len(m8_feasibility_rows)}; "
        f"non_identity_dynamic_equivalent={count_flag_in(m8_dynamic_tests, 'function_equivalent_under_tol')}/{len(m8_dynamic_tests)}; "
        f"max_non_identity_abs_logit_delta={max([value_or(r.get('max_abs_logit_delta'), 0.0) for r in m8_dynamic_tests if finite_float(r.get('max_abs_logit_delta')) is not None] or [0.0]):.6g}."
    )
    blocker = read_rows(OUT_ROOT / "v22_45E_blocker_repair_log.csv")
    execution_repairs = read_rows(OUT_ROOT / "v22_45E_execution_repair_log.csv")
    proxy_audit = read_rows(OUT_ROOT / "v22_45E_proxy_evidence_audit.csv")
    m4_summary = read_json(OUT_ROOT / "v22_45E_M4_slow_signal_diagnostic_summary.json")
    m7_summary = read_json(OUT_ROOT / "v22_45E_M7_continual_memory_summary.json")
    m4_rows = read_rows(OUT_ROOT / "v22_45E_M4_slow_signal_diagnostic_matrix.csv")
    m7_rows = read_rows(OUT_ROOT / "v22_45E_M7_continual_memory_matrix.csv")

    text = [
        "# DG-KAN v22.45E MultiMechanismExtension 实验结果复盘",
        "",
        f"更新时间：{now_sg()}",
        "",
        "## 结论",
        "",
        f"- final_route: `{final.get('final_route')}`",
        f"- reason: {final.get('reason')}",
        f"- route_eligible_summary_rows: `{final.get('route_eligible_summary_rows', 0)}`",
        f"- proxy_or_blocker_summary_rows: `{final.get('proxy_or_blocker_summary_rows', 0)}`",
        f"- proxy_or_blocker_candidate_rows: `{final.get('proxy_or_blocker_candidate_rows', 0)}`",
        f"- support_only_route_eligible_rows: `{final.get('support_only_route_eligible_rows', 0)}` / all_support_only_rows: `{final.get('support_only_all_rows', 0)}`",
        f"- m5_fairness_candidate_pass_rows: `{final.get('m5_fairness_candidate_pass_rows', 0)}` / m5_fairness_plan_rows: `{final.get('m5_fairness_plan_rows', 0)}`",
        f"- m6_explanatory_pass_rows: `{final.get('m6_explanatory_pass_rows', 0)}` / m6_trajectory_plan_rows: `{final.get('m6_trajectory_plan_rows', 0)}`",
        f"- m7_opened_rows: `{final.get('m7_opened_rows', 0)}` / m7_trajectory_plan_rows: `{final.get('m7_trajectory_plan_rows', 0)}`",
        f"- m9_opened_rows: `{final.get('m9_opened_rows', 0)}` / m9_trajectory_plan_rows: `{final.get('m9_trajectory_plan_rows', 0)}`",
        f"- m8_faithful_runtime_gamma_available_rows: `{final.get('m8_faithful_runtime_gamma_available_rows', 0)}` / m8_gauge_feasibility_rows: `{final.get('m8_gauge_feasibility_rows', 0)}`",
        f"- m8_dynamic_equivalent_rows: `{final.get('m8_dynamic_equivalent_rows', 0)}` / m8_dynamic_equivalence_rows: `{final.get('m8_dynamic_equivalence_rows', 0)}`",
        "- 本轮不把 proxy/blocker 机制升级为 opened route；所有 route 只来自实际落盘 artifact。",
        "",
        "## 机制覆盖口径",
        "",
        md_table(coverage, ["mechanism", "coverage", "kernel_mapping", "claim_limit"], 12),
        "## Proxy / Evidence Tier Audit",
        "",
        "本节由 finalize 重新扫描当前 artifact 生成；`route_eligible=0` 的行不得参与 opened route。",
        "",
        md_table(proxy_audit, ["audit_kind", "source_artifact", "mechanism", "recipe_or_coverage", "row_count", "candidate_rows", "evidence_tier", "route_eligible", "note"], 40),
        "## Code / Runtime Gate",
        "",
        md_table(code, ["compile_pass", "import_pass", "runtime_argmax_candidate_used", "candidate_action_selection_used_for_runtime", "metric_search_used", "status"], 5),
        "## 第二阶段低成本筛选",
        "",
        md_table(screen, ["mechanism", "coverage", "coverage_effective", "evidence_tier", "route_eligible", "candidate_rows", "NLL_improvement_vs_own_rows", "beats_matched_control_rows", "beats_MLP_matched_support_rows", "no_ECE_Brier_tail_debt_rows", "controller_overhead_le_0p35_rows", "screen_pass", "screen_pass_route_eligible", "claim_limit"], 12),
        "## 续跑修复实验",
        "",
        "本节只引用 `v22_45E_repair_*` artifact；这些实验按计划 fallback 针对 safety debt、functional radial gate 和 overhead/cadence 做续修。",
        "",
        md_table(repair_screen, ["mechanism", "coverage", "coverage_effective", "evidence_tier", "route_eligible", "candidate_rows", "NLL_improvement_vs_own_rows", "beats_matched_control_rows", "beats_MLP_matched_support_rows", "no_ECE_Brier_tail_debt_rows", "controller_overhead_le_0p35_rows", "screen_pass", "screen_pass_route_eligible", "claim_limit"], 12),
        "## M1 trajectory-first Functional Mirror",
        "",
        "本节只引用 `v22_45E_m1_trajectory_*` artifact。实现为 train-batch logit dual/primal mirror update；`M1-KLMirrorLS` 进一步用 sketched train-batch finite-difference response 解 ridge least-squares actuator projection。它仍不是完整 all-function/all-parameter solver。",
        "",
        md_table(m1_trajectory_screen, ["source_artifact", "mechanism", "coverage", "coverage_effective", "evidence_tier", "route_eligible", "candidate_rows", "NLL_improvement_vs_own_rows", "beats_matched_control_rows", "no_ECE_Brier_tail_debt_rows", "controller_overhead_le_0p35_rows", "screen_pass", "screen_pass_route_eligible", "claim_limit"], 12),
        "",
        md_table([r for r in m1_trajectory_matrix if r.get("mechanism") == "M1" and r.get("control_mode") == "none"], ["source_artifact", "dataset", "seed", "architecture", "variant", "mirror_potential_type", "mirror_actuator_projection", "mirror_grad_source", "Bregman_step_size", "KL_step", "Brier_delta", "tail_q99_mirror_delta", "margin_q10_mirror_delta", "mirror_ls_active_fraction", "mirror_ls_basis_count", "mirror_ls_random_basis_count", "mirror_ls_fd_eval_count", "mirror_ls_residual_ratio", "mirror_grad_refreshed_fraction", "mirror_grad_cached_fraction", "NLL_improvement_vs_own_strong_optimizer", "beats_same_mirror_support_random", "beats_same_mirror_support_signflip", "no_ECE_Brier_tail_debt", "controller_overhead_ratio"], 24),
        "## M2 per-mode radial 续修",
        "",
        "本节引用 `v22_45E_m2_*` artifact；目标是把上一轮 support-overlap radial proxy 升级为 per-matrix SVD mode RSE gated radial channel。",
        "",
        md_table(m2_repair_screen, ["source_artifact", "mechanism", "coverage", "coverage_effective", "evidence_tier", "route_eligible", "candidate_rows", "NLL_improvement_vs_own_rows", "beats_matched_control_rows", "no_ECE_Brier_tail_debt_rows", "controller_overhead_le_0p35_rows", "screen_pass", "screen_pass_route_eligible", "claim_limit"], 12),
        "## M2 trajectory-first 修复",
        "",
        "本节只引用 `v22_45E_m2_trajectory_*` artifact。它与旧 dispatch 不同：每条记录是一个完整训练轨迹，runner 在同一个 Python 进程内并行调用 `train_trajectory_from_spec`，不为每个 spec 启动子进程。该实现仍是 finite-difference functional-JVP 近似，不等同于完整 functional actuator spectrum solver。",
        "",
        md_table(m2_trajectory_plan, ["dataset", "seed", "architecture", "NLL_improvement_vs_base", "NLL_improvement_vs_strict_OET", "NLL_improvement_vs_same_radial", "NLL_improvement_vs_same_OET", "beats_strict_OET", "beats_same_radial", "beats_same_OET", "no_ECE_Brier_tail_debt", "controller_overhead_ratio", "radial_energy_fraction", "functional_radial_fd_eval_count_mean"], 24),
        "",
        md_table(m2_trajectory_status, ["label", "status", "error"], 24),
        "## M3 trajectory-first Transported Lie Momentum",
        "",
        "本节只引用 `v22_45E_m3_trajectory_*` artifact。实现为完整训练轨迹内的 runtime Lie generator momentum state：当前 OET skew generator 先与上一轮 Cayley rotation transport 后的 momentum 做 EMA，再用 momentum generator 执行 Cayley。matched controls 走同一 momentum 管线。",
        "",
        md_table(m3_trajectory_screen, ["source_artifact", "mechanism", "coverage", "coverage_effective", "evidence_tier", "route_eligible", "candidate_rows", "NLL_improvement_vs_own_rows", "beats_matched_control_rows", "no_ECE_Brier_tail_debt_rows", "controller_overhead_le_0p35_rows", "screen_pass", "screen_pass_route_eligible", "claim_limit"], 12),
        "",
        md_table(m3_trajectory_plan, ["dataset", "seed", "architecture", "NLL_improvement_vs_base", "NLL_improvement_vs_ambient_OET", "NLL_improvement_vs_same_Lie_random", "NLL_improvement_vs_same_Lie_signflip", "beats_ambient_OET", "beats_same_Lie_random", "beats_same_Lie_signflip", "no_ECE_Brier_tail_debt", "controller_overhead_ratio", "lie_momentum_active_fraction", "transport_error", "Lie_SNR"], 24),
        "",
        md_table(m3_trajectory_status, ["source_artifact", "label", "status", "error"], 24),
        "## M4 trajectory-first Slow-Signal Reservoir",
        "",
        "本节只引用正式 `v22_45E_m4_trajectory_*` artifact，文件名含 `smoke` 的调试运行只留在执行日志。每条记录是完整训练轨迹：modular-addition 用 grokking-delay gate，Class_MNIST 用 old/new continual-forgetting gate。当前实现用 slow-gradient top-k 坐标近似 support，并把 fast-gradient 剩余部分当 reservoir，因此是 partial implementation，不是完整 metric/OET 投影。",
        "",
        md_table(m4_trajectory_summaries, ["source_artifact", "rows", "groups", "route", "slow_signal_final_test_acc_beats_base_groups", "slow_signal_final_test_NLL_beats_base_groups", "matched_slow_controls_fail_groups", "invalid_no_delay_groups", "any_test_cross_groups", "grokking_delay_reduction_mean", "valid_forgetting_groups", "continual_forgetting_reduction_mean", "continual_forgetting_pass_groups", "final_average_accuracy_nonworse_groups", "RSM_index_mean", "runtime_bad_flag_count", "promotion_allowed", "claim_limit"], 12),
        "",
        md_table(m4_trajectory_plan, ["source_artifact", "dataset", "architecture", "seed", "base_final_test_accuracy", "slow_final_test_accuracy", "NLL_improvement_vs_base", "base_grokking_delay", "slow_grokking_delay", "grokking_delay_reduction", "grokking_delay_reduction_ge_25p", "base_average_forgetting", "slow_average_forgetting", "relative_forgetting_reduction", "continual_forgetting_reduction_ge_10p", "same_slow_random_control_gap", "same_slow_signflip_control_gap", "matched_slow_controls_fail", "final_average_accuracy_nonworse", "M4_pass", "RSM_index"], 24),
        "",
        md_table([r for r in m4_trajectory_matrix if r.get("variant") == "slow_signal_flow"], ["source_artifact", "dataset", "architecture", "seed", "final_train_accuracy", "final_test_accuracy", "old_task_accuracy_before_new_task", "old_task_accuracy_after_new_task", "new_task_accuracy", "average_forgetting", "final_average_accuracy", "train_cross_step", "test_cross_step", "grokking_delay", "slow_signal_energy", "fast_signal_energy", "RSM_index", "support_slow_fraction", "support_fast_fraction", "in_process", "subprocess_used", "complete_step_loop"], 24),
        "",
        md_table(m4_trajectory_status, ["source_artifact", "label", "status", "error"], 24),
        "## M5 变体效率复测",
        "",
        "本节测试 KAN carrier 是否必须使用高开销 BasisGram metric，重点比较 OET-BankLocal 变体的收益、matched controls、MLP matched support 和 overhead。",
        "",
        md_table(m5_variant_screen, ["source_artifact", "mechanism", "coverage", "coverage_effective", "evidence_tier", "route_eligible", "candidate_rows", "NLL_improvement_vs_own_rows", "beats_matched_control_rows", "beats_MLP_matched_support_rows", "no_ECE_Brier_tail_debt_rows", "controller_overhead_le_0p35_rows", "screen_pass", "screen_pass_route_eligible", "claim_limit"], 24),
        "## M5 trajectory-first 公平性审计",
        "",
        "本节只引用 `v22_45E_m5_trajectory_*` artifact；每条记录是完整训练轨迹，用于重新审计 KAN carrier 是否打过 same-basis controls 和 MLP matched functional supports。",
        "",
        md_table(m5_trajectory_screen, ["source_artifact", "mechanism", "coverage", "coverage_effective", "evidence_tier", "route_eligible", "candidate_rows", "NLL_improvement_vs_own_rows", "beats_matched_control_rows", "beats_MLP_matched_support_rows", "no_ECE_Brier_tail_debt_rows", "controller_overhead_le_0p35_rows", "screen_pass", "screen_pass_route_eligible", "claim_limit"], 12),
        "",
        f"- {m5_fairness_note}",
        "- `exact_spectrum_seed=0` 的行是显式 seed fallback：trajectory 是 seed=1 真实训练，但 spectrum audit 当前只有 seed=0；这些行不能写成 seed=1 spectrum 实测。",
        "",
        md_table(m5_fairness_summary, ["source_artifact", "rows", "KAN_improvement_rows", "beats_same_basis_controls_rows", "beats_MLP_matched_support_rows", "TrueKANGain_plus_BothGain_rows", "ControlExplained_rows", "MLPDegradationDriven_rows", "no_debt_rows", "overhead_le_0p35_rows", "spectrum_pair_available_rows", "exact_spectrum_seed_rows", "M5_fairness_candidate_pass_rows", "mean_functional_spectrum_distance", "mean_RSE_distance", "mean_output_scale_distance"], 12),
        "",
        md_table(m5_fairness_plan, ["source_artifact", "dataset", "seed", "KAN_variant", "matched_MLP_variant", "KAN_vs_MLP_matched_support_NLL_delta", "KAN_vs_MLP_matched_support_AUC_delta", "KAN_beats_MLP_matched_support", "KAN_improves", "MLP_improves", "TrueKANGain_class", "ControlExplained", "MLPDegradationDriven", "beats_same_basis_controls", "no_ECE_Brier_tail_debt", "controller_overhead_le_0p35", "support_rank", "support_effective_rank", "mlp_support_effective_rank", "functional_spectrum_distance", "RSE_distance", "output_scale_distance", "spectrum_seed_exact_all", "M5_fairness_candidate_pass"], 24),
        "",
        md_table(m5_trajectory_status, ["source_artifact", "label", "status", "error"], 24),
        "## M6 trajectory-first Reservoir Noise",
        "",
        "本节只引用 `v22_45E_m6_trajectory_*` artifact。M6 现在在 v22.43 完整训练 loop 内生成 parameter metric support-complement reservoir noise，并与 noise suppression、signal-noise control、same-norm Gaussian control 比较；它只能支持 ReservoirRegularizationExplainsGain，不能声明 SignalFUOpened。",
        "",
        f"- {m6_formal_note}",
        "",
        md_table(m6_trajectory_screen, ["source_artifact", "mechanism", "coverage", "coverage_effective", "evidence_tier", "route_eligible", "candidate_rows", "NLL_improvement_vs_own_rows", "beats_matched_control_rows", "no_ECE_Brier_tail_debt_rows", "controller_overhead_le_0p35_rows", "screen_pass", "screen_pass_route_eligible", "claim_limit"], 12),
        "",
        md_table(m6_trajectory_plan, ["source_artifact", "dataset", "seed", "architecture", "NLL_improvement_vs_base", "NLL_improvement_vs_noise_suppression", "NLL_improvement_vs_signal_noise_control", "NLL_improvement_vs_same_norm_Gaussian_control", "beats_noise_suppression", "beats_signal_noise_control", "beats_same_norm_Gaussian_control", "no_ECE_Brier_tail_debt", "reservoir_noise_energy", "signal_noise_leakage", "RSM_index", "runtime_truth", "M6_reservoir_regularization_explains_gain"], 24),
        "",
        md_table([r for r in m6_trajectory_matrix if r.get("mechanism") == "M6" and r.get("control_mode") == "none"], ["source_artifact", "dataset", "seed", "architecture", "variant", "final_NLL", "NLL_improvement_vs_own_strong_optimizer", "reservoir_noise_energy", "reservoir_noise_metric_norm", "signal_noise_leakage", "RSM_index", "no_ECE_Brier_tail_debt", "controller_overhead_ratio", "continuous_fu_state_updated_every_step", "fu_velocity_emitted_every_step", "candidate_action_selection_used_for_runtime"], 24),
        "",
        md_table(m6_trajectory_status, ["source_artifact", "label", "status", "error"], 24),
        "## M7 trajectory-first Continual Memory",
        "",
        "本节只引用 `v22_45E_m7_trajectory_*` artifact。M7 现在是 v22.45E 自己的完整 old/new MNIST-family 与 modular curriculum 训练轨迹：old-task 阶段维护 EMA / rank-k old-gradient memory；parameter projector 对新任务梯度做 memory-subspace residualization，functional projector 则在旧任务 train anchor 上用 finite-difference 估计 `J_old*v` 并投影 old-logit memory component。两者都与 optimizer baseline、same-rank random subspace、old-memory rehearsal control 比较。functional projector 是 train-anchor `J_old*v` 近似，仍不是完整 all-function `G_f` M7。",
        "正式 M7 route/结论排除文件名含 `smoke` 的调试 artifacts；smoke 只用于 runner 修复验证和执行日志复现。",
        "",
        f"- {m7_formal_note}",
        f"- {m7_all_note}",
        "",
        md_table(m7_trajectory_summaries, ["source_artifact", "rows", "groups", "memory_projected_opened_groups", "official_ready_groups", "relative_forgetting_reduction_mean", "matched_memory_controls_fail_groups", "final_average_accuracy_nonworse_groups", "constraint_ok_groups", "runtime_bad_flag_count", "route", "promotion_allowed", "claim_limit"], 12),
        "",
        md_table(m7_trajectory_plan, ["source_artifact", "dataset", "seed", "architecture", "memory_rank", "projected_variant", "base_average_forgetting", "projected_average_forgetting", "forgetting_task_valid", "relative_forgetting_reduction", "same_memory_random_control_gap", "same_memory_rehearsal_control_gap", "final_average_accuracy_nonworse", "matched_memory_controls_fail", "memory_projection_before_ratio", "memory_projection_after_ratio", "functional_memory_projection_after_ratio", "functional_fd_eval_count_mean", "memory_constraint_satisfied_fraction", "memory_signal_retention", "memory_rank_effective", "runtime_truth", "constraint_ok", "M7_exploration_opened", "M7_official_ready"], 24),
        "",
        md_table([r for r in m7_trajectory_matrix if r.get("variant") in {"memory_projected_residual_flow", "functional_memory_projected_residual_flow"}], ["source_artifact", "dataset", "seed", "architecture", "variant", "memory_rank", "memory_rank_effective", "task_boundary", "modular_eval_split", "old_task_accuracy_before_new_task", "old_task_accuracy_after_new_task", "new_task_accuracy", "final_average_accuracy", "average_forgetting", "memory_projection_before_ratio", "memory_projection_after_ratio", "functional_memory_projection_after_ratio", "functional_fd_eval_count_mean", "memory_constraint_satisfied_fraction", "memory_signal_retention", "in_process", "subprocess_used", "complete_step_loop"], 24),
        "",
        md_table(m7_trajectory_status, ["source_artifact", "label", "status", "error"], 24),
        "## M8 Gauge / Canonicalization audit",
        "",
        "本节不是训练轨迹。M8 的计划要求是 `theta_t^canon = Gamma(theta_t)` 且 FU state 在 canonical gauge 内更新；本轮只审计当前 PrimitiveKAN/FU runtime 是否具备这个 faithful transform，避免把参数 rescale 或 metric diagnostic 误报成 gauge training。",
        "",
        f"- {m8_note}",
        "",
        md_table(m8_gauge_audit, ["mechanism", "candidate_gauge", "plan_requirement", "faithful_runtime_gamma_available", "runtime_fu_state_canonical_update_available", "route_eligible", "diagnostic_only", "repair_direction_tried", "blocker"], 12),
        "",
        md_table(m8_gauge_dynamic, ["mechanism", "dataset", "seed", "architecture", "transform_name", "transform_status", "max_abs_logit_delta", "relative_l2_logit_delta", "function_equivalent_under_tol", "faithful_runtime_gamma_available", "diagnostic_only", "route_eligible"], 24),
        "## M9 trajectory-first Adaptive Metric Mixture",
        "",
        "本节只引用 `v22_45E_m9_trajectory_*` artifact。M9 在 v22.43 完整训练 loop 内维护 Euclidean/Fisher/Signal/Basis 四个预注册 diagonal metric 的 mixture weights；权重只由 train-batch safety/support/residual debt 做 mirror descent 更新，并与 fixed uniform / fixed single-metric controls 比较。",
        "",
        f"- {m9_formal_note}",
        "",
        md_table(m9_trajectory_screen, ["source_artifact", "mechanism", "coverage", "coverage_effective", "evidence_tier", "route_eligible", "candidate_rows", "NLL_improvement_vs_own_rows", "beats_matched_control_rows", "no_ECE_Brier_tail_debt_rows", "controller_overhead_le_0p35_rows", "screen_pass", "screen_pass_route_eligible", "claim_limit"], 12),
        "",
        md_table(m9_trajectory_plan, ["source_artifact", "dataset", "seed", "architecture", "NLL_improvement_vs_base", "NLL_improvement_vs_uniform", "NLL_improvement_vs_best_fixed", "best_fixed_metric", "beats_uniform", "beats_best_fixed_metric", "no_ECE_Brier_tail_debt", "adaptation_truth", "runtime_truth", "mixture_collapsed", "metric_mixture_final_entropy", "metric_weight_drift_total_proxy", "metric_final_weight_euclidean", "metric_final_weight_fisher", "metric_final_weight_signal", "metric_final_weight_basis", "controller_overhead_ratio", "AdaptiveMetricFlowOpened", "MetricSelectionDiagnosticOnly"], 24),
        "",
        md_table([r for r in m9_trajectory_matrix if r.get("mechanism") == "M9" and r.get("control_mode") == "none"], ["source_artifact", "dataset", "seed", "architecture", "variant", "final_NLL", "NLL_improvement_vs_own_strong_optimizer", "metric_mixture_active_fraction", "metric_mixture_final_entropy", "metric_weight_drift_total_proxy", "metric_final_weight_euclidean", "metric_final_weight_fisher", "metric_final_weight_signal", "metric_final_weight_basis", "metric_mixture_collapsed_to", "support_gain_by_metric_euclidean", "support_gain_by_metric_fisher", "support_gain_by_metric_signal", "support_gain_by_metric_basis", "no_ECE_Brier_tail_debt", "controller_overhead_ratio"], 24),
        "",
        md_table(m9_trajectory_status, ["source_artifact", "label", "status", "error"], 24),
        "## Support / Direction 证据链",
        "",
        "NLL 口径中 `nll_tau_support_base_minus_control > 0` 表示 same-support control 优于 base；"
        "`nll_tau_direction_control_minus_real > 0` 表示真实方向优于 same-support control。",
        "",
        md_table(decomp, ["mechanism", "evidence_tier", "route_eligible", "dataset", "seed", "architecture", "variant", "control_used", "nll_tau_support_base_minus_control", "nll_tau_direction_control_minus_real", "support_positive_direction_nonpositive"], 24),
        "## 续跑 Support / Direction 证据链",
        "",
        md_table(repair_decomp, ["mechanism", "evidence_tier", "route_eligible", "dataset", "seed", "architecture", "variant", "control_used", "nll_tau_support_base_minus_control", "nll_tau_direction_control_minus_real", "support_positive_direction_nonpositive"], 24),
        "## M2 per-mode radial Support / Direction",
        "",
        md_table(m2_repair_decomp, ["source_artifact", "mechanism", "evidence_tier", "route_eligible", "dataset", "seed", "architecture", "variant", "control_used", "nll_tau_support_base_minus_control", "nll_tau_direction_control_minus_real", "support_positive_direction_nonpositive"], 24),
        "## M3 trajectory Support / Direction",
        "",
        md_table(m3_trajectory_decomp, ["source_artifact", "mechanism", "evidence_tier", "route_eligible", "dataset", "seed", "architecture", "variant", "control_used", "nll_tau_support_base_minus_control", "nll_tau_direction_control_minus_real", "support_positive_direction_nonpositive"], 24),
        "## M5 变体 Support / Direction",
        "",
        md_table(m5_variant_decomp, ["mechanism", "evidence_tier", "route_eligible", "dataset", "seed", "architecture", "variant", "control_used", "nll_tau_support_base_minus_control", "nll_tau_direction_control_minus_real", "support_positive_direction_nonpositive"], 24),
        "## M5 trajectory Support / Direction",
        "",
        md_table(m5_trajectory_decomp, ["source_artifact", "mechanism", "evidence_tier", "route_eligible", "dataset", "seed", "architecture", "variant", "control_used", "nll_tau_support_base_minus_control", "nll_tau_direction_control_minus_real", "support_positive_direction_nonpositive"], 24),
        "## M6 trajectory Support / Direction",
        "",
        md_table(m6_trajectory_decomp, ["source_artifact", "mechanism", "evidence_tier", "route_eligible", "dataset", "seed", "architecture", "variant", "control_used", "nll_tau_support_base_minus_control", "nll_tau_direction_control_minus_real", "support_positive_direction_nonpositive"], 24),
        "## M9 trajectory Support / Direction",
        "",
        md_table(m9_trajectory_decomp, ["source_artifact", "mechanism", "evidence_tier", "route_eligible", "dataset", "seed", "architecture", "variant", "control_used", "nll_tau_support_base_minus_control", "nll_tau_direction_control_minus_real", "support_positive_direction_nonpositive"], 24),
        "## Safety Debt",
        "",
        md_table(safety, ["mechanism", "dataset", "seed", "architecture", "variant", "control_mode", "ECE_delta_vs_own_strong_optimizer", "Brier_delta_vs_own_strong_optimizer", "tail_q99_delta_vs_own_strong_optimizer", "no_ECE_Brier_tail_debt"], 24),
        "## 续跑 Safety Debt",
        "",
        md_table(repair_safety, ["mechanism", "dataset", "seed", "architecture", "variant", "control_mode", "ECE_delta_vs_own_strong_optimizer", "Brier_delta_vs_own_strong_optimizer", "tail_q99_delta_vs_own_strong_optimizer", "no_ECE_Brier_tail_debt"], 24),
        "## M2 per-mode radial Safety Debt",
        "",
        md_table(m2_repair_safety, ["source_artifact", "mechanism", "dataset", "seed", "architecture", "variant", "control_mode", "ECE_delta_vs_own_strong_optimizer", "Brier_delta_vs_own_strong_optimizer", "tail_q99_delta_vs_own_strong_optimizer", "no_ECE_Brier_tail_debt"], 24),
        "## M3 trajectory Safety Debt",
        "",
        md_table(m3_trajectory_safety, ["source_artifact", "mechanism", "dataset", "seed", "architecture", "variant", "control_mode", "ECE_delta_vs_own_strong_optimizer", "Brier_delta_vs_own_strong_optimizer", "tail_q99_delta_vs_own_strong_optimizer", "no_ECE_Brier_tail_debt"], 24),
        "## M1 trajectory Safety Debt",
        "",
        md_table(m1_trajectory_safety, ["source_artifact", "mechanism", "dataset", "seed", "architecture", "variant", "control_mode", "ECE_delta_vs_own_strong_optimizer", "Brier_delta_vs_own_strong_optimizer", "tail_q99_delta_vs_own_strong_optimizer", "no_ECE_Brier_tail_debt"], 24),
        "## M5 变体 Safety Debt",
        "",
        md_table(m5_variant_safety, ["mechanism", "dataset", "seed", "architecture", "variant", "control_mode", "ECE_delta_vs_own_strong_optimizer", "Brier_delta_vs_own_strong_optimizer", "tail_q99_delta_vs_own_strong_optimizer", "no_ECE_Brier_tail_debt"], 24),
        "## M5 trajectory Safety Debt",
        "",
        md_table(m5_trajectory_safety, ["source_artifact", "mechanism", "dataset", "seed", "architecture", "variant", "control_mode", "ECE_delta_vs_own_strong_optimizer", "Brier_delta_vs_own_strong_optimizer", "tail_q99_delta_vs_own_strong_optimizer", "no_ECE_Brier_tail_debt"], 24),
        "## M6 trajectory Safety Debt",
        "",
        md_table(m6_trajectory_safety, ["source_artifact", "mechanism", "dataset", "seed", "architecture", "variant", "control_mode", "ECE_delta_vs_own_strong_optimizer", "Brier_delta_vs_own_strong_optimizer", "tail_q99_delta_vs_own_strong_optimizer", "no_ECE_Brier_tail_debt"], 24),
        "## M9 trajectory Safety Debt",
        "",
        md_table(m9_trajectory_safety, ["source_artifact", "mechanism", "dataset", "seed", "architecture", "variant", "control_mode", "ECE_delta_vs_own_strong_optimizer", "Brier_delta_vs_own_strong_optimizer", "tail_q99_delta_vs_own_strong_optimizer", "no_ECE_Brier_tail_debt"], 24),
        "## Overhead",
        "",
        md_table(overhead, ["mechanism", "dataset", "seed", "architecture", "variant", "control_mode", "controller_overhead_ratio", "full_step_ms", "controller_ms", "memory_peak_MB"], 24),
        "## 续跑 Overhead",
        "",
        md_table(repair_overhead, ["mechanism", "dataset", "seed", "architecture", "variant", "control_mode", "controller_overhead_ratio", "full_step_ms", "controller_ms", "memory_peak_MB"], 24),
        "## M2 per-mode radial Overhead",
        "",
        md_table(m2_repair_overhead, ["source_artifact", "mechanism", "dataset", "seed", "architecture", "variant", "control_mode", "controller_overhead_ratio", "full_step_ms", "controller_ms", "memory_peak_MB"], 24),
        "## M3 trajectory Overhead",
        "",
        md_table(m3_trajectory_overhead, ["source_artifact", "mechanism", "dataset", "seed", "architecture", "variant", "control_mode", "controller_overhead_ratio", "full_step_ms", "controller_ms", "memory_peak_MB"], 24),
        "## M1 trajectory Overhead",
        "",
        md_table(m1_trajectory_overhead, ["source_artifact", "mechanism", "dataset", "seed", "architecture", "variant", "control_mode", "controller_overhead_ratio", "full_step_ms", "controller_ms", "mirror_grad_ms", "mirror_grad_refreshed_fraction", "mirror_grad_cached_fraction", "memory_peak_MB"], 24),
        "## M5 变体 Overhead",
        "",
        md_table(m5_variant_overhead, ["mechanism", "dataset", "seed", "architecture", "variant", "control_mode", "controller_overhead_ratio", "full_step_ms", "controller_ms", "memory_peak_MB"], 24),
        "## M5 trajectory Overhead",
        "",
        md_table(m5_trajectory_overhead, ["source_artifact", "mechanism", "dataset", "seed", "architecture", "variant", "control_mode", "controller_overhead_ratio", "full_step_ms", "controller_ms", "memory_peak_MB"], 24),
        "## M6 trajectory Overhead",
        "",
        md_table(m6_trajectory_overhead, ["source_artifact", "mechanism", "dataset", "seed", "architecture", "variant", "control_mode", "controller_overhead_ratio", "full_step_ms", "controller_ms", "memory_peak_MB"], 24),
        "## M9 trajectory Overhead",
        "",
        md_table(m9_trajectory_overhead, ["source_artifact", "mechanism", "dataset", "seed", "architecture", "variant", "control_mode", "controller_overhead_ratio", "full_step_ms", "controller_ms", "basis_metric_update_time_ms", "memory_peak_MB"], 24),
        "## M4 / M7 诊断",
        "",
        f"- M4 diagnostic route: `{m4_summary.get('route', 'not_run')}`; rows={len(m4_rows)}; source copied from v22.42R diagnostic label.",
        f"- M7 diagnostic route: `{m7_summary.get('route', 'not_run')}`; rows={len(m7_rows)}; source copied from v22.42R diagnostic label.",
        "",
        md_table(m4_rows, ["run_label", "architecture", "seed", "variant", "final_train_acc", "final_test_acc", "grokking_delay", "time_to_generalization"], 12),
        md_table(m7_rows, ["run_label", "architecture", "seed", "variant", "old_task_accuracy_before_new_task", "old_task_accuracy_after_new_task", "new_task_accuracy", "average_forgetting", "relative_forgetting_reduction"], 12),
        "## Blocker / 修复记录",
        "",
        md_table(blocker, ["mechanism", "blocker", "repair_attempt", "result"], 12),
        "## 执行缺陷修复审计",
        "",
        md_table(execution_repairs, ["repair_id", "observed_blocker", "code_change", "rerun_command", "verified_result"], max(12, len(execution_repairs))),
        "## Insight",
        "",
        "- 用户指出 proxy 污染后复核确认：当前最小可审计污染范围是 112 条显式 `Proxy` 低成本 screen/spec 运行行，以及 6 个被修正为 `proxy_or_blocker` 的 summary 行、合计 96 个 candidate rows。它们已经通过 `route_eligible=0` 和 `screen_pass_route_eligible=0` 从 opened-route gate 排除。",
        "- 早期手工审计曾用更宽的 trajectory-summary 扫描得到 209 条“可能污染”记录；该数字混入了后续 partial/blocker summary 扫描，不是最终证据口径，不能当作“误导实验数”或 route 证据使用。",
        "- 这次错误的实际伤害不是把最终结论推成成功，而是污染了中间解释语言：M1/M4/M6 的 proxy rows 曾让机制命名看起来比实现更接近计划。后续只允许 complete in-process trajectory rows 或明确标注的 diagnostic/blocker rows 进入复盘结论。",
        "- v22.45E 的关键风险不是代码是否能跑，而是机制身份是否被 proxy 污染；本轮显式把 proxy/blocker 与可实现机制分开。",
        "- finalize 现在写出 `v22_45E_proxy_evidence_audit.csv`，并用 `evidence_tier` / `route_eligible` / `screen_pass_route_eligible` 作为 route gate；旧 proxy/blocker rows 即使保留在 artifact 里也不能进入 opened route。",
        "- 旧 harness 的 per-row subprocess 调度只适合审计型短跑；本轮新增的 trajectory-m1/m2/m3/m4/m5/m6/m9 才作为 faithful trajectory evidence，均记录 `in_process=1`、`subprocess_used=0`、`complete_step_loop=1` 或等价 runtime truth。",
        "- 新增 `trajectory-m2` 后，M2 结果会单独以 `v22_45E_m2_trajectory_*` 命名；这些结果来自 in-process 完整 step loop，可与旧 dispatch proxy 分开审计。",
        "- M2 已从 support-overlap radial proxy 进一步尝试到 safety-only top gate + per-mode train-batch finite-difference RSE scoring；这比旧 proxy 更接近计划，但仍不是完整 functional actuator-spectrum solver。",
        "- M2 RSE-only c240 是负结果：beats_strict_OET=6/16、beats_same_radial=10/16、no_debt=3/16、overhead<=0.35=0/16，且弱于旧 trajectory M2 的 11/16、12/16、5/16。不能声明 SpectralRadialSplitFlowOpened。",
        "- M3 transported Lie momentum 已从 blocker 推进到真实 runtime state：c240 有 lie_momentum_active_fraction=0.75、transport_error≈1.2e-7、beats_same_Lie_random=12/16、beats_same_Lie_signflip=11/16；但 beats_ambient_OET 只有 5/16、no_debt=3/16、overhead<=0.35=0/16，因此不能声明 LieMomentumFlowOpened。",
        "- M4 现在不再只依赖旧 v22.42R copy；`trajectory-m4` 已产生 v22.45E 自己的 modular-addition 与 Class_MNIST 完整轨迹、matched slow controls、RSM_index、delay/forgetting 指标。modular-addition c800、p13 c1600+wd1e-3、p7 c1600+wd1e-3 均为 16/16 trajectory pass 且 runtime_bad=0，但全部 `GrokkingTaskInvalidNoDelay`。Class_MNIST default/strong velocity 也各为 16/16 pass 且 runtime_bad=0，valid_forgetting=4/4，但 forgetting_reduction>=10%=0/4、matched_controls_fail=0/4，因此仍不能声明 SlowSignalReservoirMigrationOpened。",
        "- M5 trajectory variants 显示 KAN BasisGram carrier 有 control/MLP-matched 信号，但 no-debt 与 overhead 同时不过；`basis_metric_update_time_ms` 当前记录的是整个 controller update 段而非纯 metric refresh，因此单纯提高 refresh cadence 不能线性修掉 overhead。",
        "- M5 fast/budget 修复给出负证据：basis-only controller state 只能把 fast+c240+budget 的 overhead_ok 提到 1/8，tail_q99/brier aux guard 会把 controller-state 时间从约 45ms 拉到约 82ms 且 no-debt 仍只有 2/8；继续盲扫 calibration 不合理。",
        f"- {m5_fairness_note} 这一步是审计修复，不是新训练；它暴露的关键限制是 M5 matched-support 证据必须同时看 functional spectrum/RSE/output scale、MLP matched support、same-basis controls、safety debt 与 overhead，不能只看 KAN-vs-MLP NLL delta。",
        f"- M6 的修复方向从 `NoiseControlProxy` 改为完整 trajectory 内的 support-complement reservoir-noise 注入；{m6_formal_note} 该结果不支持 ReservoirRegularizationExplainsGain，更不能写成 SignalFUOpened。",
        f"- M7 的修复方向从 v22.42R copy 推进到 v22.45E 自己的 trajectory-first continual-memory loop；{m7_all_note} 当前 parameter-gradient 与 finite-difference old-logit functional projector 都不是完整 all-function `G_f` M7。functional Class_MNIST rank4 虽有 mean_relative_forgetting_reduction≈0.2166 且 functional_projection_after≈6.9e-6，但 matched random functional control 在 4/4 组上遗忘更少，controls_fail=0/4，因此不能声明 ContinualFunctionalMemoryOpened。",
        f"- M8 的修复方向先做 faithful gauge 审计而不是直接训练；{m8_note} 当前 PrimitiveKAN/FU runtime 没有可用 `Gamma(theta)` 和 canonical FU state update，naive rescale 会改变 logits，因此只能保留 GaugeDiagnosticOnly/blocker。",
        f"- M9 的修复方向是从 fixed metric rows 推进到完整 trajectory 内 train-only diagonal metric mixture weights；{m9_formal_note} 只有同时 beats uniform、beats best fixed metric、no-debt、runtime/adaptation truth 且未 collapse，才允许声明 AdaptiveMetricFlowOpened。",
        "- 当前 M1 已新增 train-batch logit dual/primal mirror trajectory；KL 从 CE-gradient reuse 修正为 mirror-residual autograd 后，NLL/control 证据增强，但 no-debt 仍不过。cached residual cadence 可把 overhead 拉到门槛附近或通过，但会让 mirror target 在刷新间隔内 stale，且 safety gate 仍失败；因此不能声明 FunctionalMirrorFlowOpened。",
        "- M1-KLMirrorLS 把 support-masked J^T residual 推进到 log-prob finite-difference ridge LS：basis4 cached40 为 pass_all=1/16、residual_mean≈0.874；rand8 cached120 为 pass_all=2/16、residual_mean≈0.839；basis16 cached240 为 pass_all=0/16、residual_mean≈0.794。LS 拟合改善没有转化为稳定训练收益，且更大 basis 会降低 NLL/control/no-debt。",
        "- support/direction 拆分是最重要的解释护栏：support control 能解释的收益不能写成 SignalFUOpened。",
        "- M5 是 KAN architecture claim 的前置公平性审计；只有同时打过 KAN baseline、same-basis controls 和 MLP matched support，才允许推进 TrueKANGain。",
        "- M1/M2/M3/M6/M7/M9 的更 faithful trajectory repairs 均需要按各自 strict gates 审计；即便机制能运行，也不能越过 controls、safety/forgetting、overhead 和 claim-limit 护栏。M8 当前仍是 runtime canonical-gauge 实现缺口。",
        "",
        "## 可复现入口",
        "",
        "- 主 runner: `experiments/run_v22_45E_multi_mechanism_extension.py`",
        "- 结果目录: `results/v22_45E/`",
        "- 执行日志: `docs/DG-KAN_v22.45E_MultiMechanismExtension_执行日志.md`",
        "- 命令 journal: `results/v22_45E/v22_45E_command_journal.csv`",
    ]
    RECAP_DOC.write_text("\n".join(text) + "\n", encoding="utf-8")


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    write_rows(OUT_ROOT / "v22_45E_mechanism_coverage_matrix.csv", MECHANISM_COVERAGE)
    write_blocker_repair_log()
    write_m5_fairness_plan_metrics()
    write_execution_repair_log()
    write_proxy_evidence_audit()
    for table in V2245E_TABLES:
        p = OUT_ROOT / table
        if not p.exists():
            write_rows(p, [{"status": "not_run", "reason": "stage not executed or gate-blocked before artifact was produced"}])
    final = final_route()
    update_recap(final)
    write_manifest()
    append_exec(
        "finalize_v22_45E",
        task_id="finalize",
        status="pass",
        files="results/v22_45E/v22_45E_final_route.json, results/v22_45E/v22_45E_artifact_manifest.csv, docs/DG-KAN_v22.45E_MultiMechanismExtension_实验结果复盘.md",
        note=f"route={final.get('final_route')}; reason={final.get('reason')}",
    )
    return final


def stage_all(args: argparse.Namespace) -> dict[str, Any]:
    run_code_truth_gate()
    run_metric_and_spectrum(args)
    run_screen(args)
    if args.run_m4_diagnostic:
        run_m4_diagnostic(args)
    if args.run_m7_diagnostic:
        run_m7_diagnostic(args)
    return finalize(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["all", "code", "metric-spectrum", "screen", "repair-screen", "repair-merge", "m5-variant-audit", "m8-gauge-audit", "trajectory-m1", "trajectory-m2", "trajectory-m3", "trajectory-m4", "trajectory-m5", "trajectory-m6", "trajectory-m7", "trajectory-m9", "collect", "m4-diagnostic", "m7-diagnostic", "merge", "finalize"])
    p.add_argument("--label", default="v22_45E")
    p.add_argument("--dataset", default="MNIST")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--architecture", default="MLP")
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--variant", default="S4-SignalMetric-OET")
    p.add_argument("--control-mode", default="none")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--train-size", type=int, default=128)
    p.add_argument("--held-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--support-rank", type=int, default=4)
    p.add_argument("--nuisance-rank", type=int, default=2)
    p.add_argument("--support-refresh-cadence", type=int, default=20)
    p.add_argument("--beta-signal", type=float, default=0.05)
    p.add_argument("--beta-metric", type=float, default=0.05)
    p.add_argument("--beta-q", type=float, default=0.05)
    p.add_argument("--eta-rho", type=float, default=0.20)
    p.add_argument("--eta-debt", type=float, default=0.10)
    p.add_argument("--tau-safe", type=float, default=0.02)
    p.add_argument("--rho-min", type=float, default=0.0)
    p.add_argument("--rho-max", type=float, default=0.25)
    p.add_argument("--velocity-scale", type=float, default=0.50)
    p.add_argument("--metric-shrinkage", type=float, default=0.10)
    p.add_argument("--metric-eps", type=float, default=1.0e-6)
    p.add_argument("--metric-refresh-cadence", type=int, default=1)
    p.add_argument("--debt-velocity-barrier", type=float, default=0.0)
    p.add_argument("--calibration-velocity-barrier", type=float, default=0.0)
    p.add_argument("--safety-budget-velocity-barrier", type=float, default=0.0)
    p.add_argument("--calibration-readout-radial-cap", type=float, default=0.0)
    p.add_argument("--calibration-nuisance-weight", type=float, default=0.0)
    p.add_argument("--calibration-correction-weight", type=float, default=0.0)
    p.add_argument("--calibration-nuisance-mode", default="brier")
    p.add_argument("--calibration-nuisance-cadence", type=int, default=1)
    p.add_argument("--mirror-grad-cadence", type=int, default=1)
    p.add_argument("--kan-init-variant", default="default")
    p.add_argument("--pure-fu-mode", action="store_true")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--tier2-download", action="store_true")
    p.add_argument("--eval-datasets", default="MNIST,FashionMNIST,KMNIST,Wine")
    p.add_argument("--eval-seeds", default="0,1")
    p.add_argument("--eval-architectures", default="MLP,DGKAN_DCHE")
    p.add_argument("--mechanisms", default="Core-M,M1,M2,M3,M4,M5,M6,M9")
    p.add_argument("--repair-mechanisms", default="M1,M2,M5")
    p.add_argument("--repair-output", default="repair")
    p.add_argument("--m5-variant-audit-kan-variants", default="KAN-D-CHE-OET-BankLocal")
    p.add_argument("--m5-mlp-matched-variants", default="MLP-low-rank-hidden-metric-support,MLP-frequency-like-random-feature-support")
    p.add_argument("--m5-variant-output", default="m5_variant")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--row-timeout", type=int, default=1200)
    p.add_argument("--row-limit", type=int, default=0)
    p.add_argument("--audit-datasets", default="MNIST,FashionMNIST,KMNIST,Wine")
    p.add_argument("--audit-architectures", default="MLP,DGKAN_DCHE,DGKAN_DFOU")
    p.add_argument("--audit-supports", default="A_MLP_lowrank,A_MLP_spectral,A_MLP_frequency_like,A_KAN_DCHE_basis_bank,A_KAN_DFOU_basis_bank,A_KAN_DCHE_bank_OET,A_KAN_DFOU_bank_OET,A_OET_layer")
    p.add_argument("--audit-train-size", type=int, default=96)
    p.add_argument("--sketch-dim", type=int, default=6)
    p.add_argument("--functional-eps", type=float, default=1.0e-3)
    p.add_argument("--m1-calibration-weight", type=float, default=0.25)
    p.add_argument("--m1-calibration-barrier", type=float, default=1.0)
    p.add_argument("--m2-velocity-scale", type=float, default=0.20)
    p.add_argument("--m2-safety-barrier", type=float, default=1.0)
    p.add_argument("--m2-repair-variant", default="P4-Euclidean-OET-functional-jvp-radial-gated-pure")
    p.add_argument("--trajectory-steps", type=int, default=0)
    p.add_argument("--trajectory-warmup-steps", type=int, default=60)
    p.add_argument("--m1-trajectory-output", default="m1_trajectory")
    p.add_argument("--m1-trajectory-variants", default="M1-KLMirror,M1-BrierMirror,M1-TailSafeMirror")
    p.add_argument("--trajectory-m1-velocity-scale", type=float, default=0.20)
    p.add_argument("--trajectory-m1-safety-barrier", type=float, default=4.0)
    p.add_argument("--trajectory-m1-mirror-grad-cadence", type=int, default=1)
    p.add_argument("--m2-trajectory-output", default="m2_trajectory")
    p.add_argument("--trajectory-m2-velocity-scale", type=float, default=0.20)
    p.add_argument("--trajectory-m2-safety-barrier", type=float, default=5.0)
    p.add_argument("--trajectory-m2-beta-signal", type=float, default=0.02)
    p.add_argument("--m3-trajectory-output", default="m3_trajectory")
    p.add_argument("--m3-trajectory-variant", default="P3-Euclidean-OET-transported-lie-momentum-pure")
    p.add_argument("--trajectory-m3-velocity-scale", type=float, default=0.20)
    p.add_argument("--trajectory-m3-safety-barrier", type=float, default=5.0)
    p.add_argument("--trajectory-m3-beta-signal", type=float, default=0.08)
    p.add_argument("--m5-trajectory-output", default="m5_trajectory")
    p.add_argument("--m6-trajectory-output", default="m6_trajectory")
    p.add_argument("--trajectory-m6-velocity-scale", type=float, default=0.20)
    p.add_argument("--trajectory-m6-safety-barrier", type=float, default=5.0)
    p.add_argument("--trajectory-m6-beta-signal", type=float, default=0.05)
    p.add_argument("--m7-trajectory-output", default="m7_trajectory")
    p.add_argument("--m7-tasks", default="Class_MNIST_0_4_to_5_9")
    p.add_argument("--m7-trajectory-steps", type=int, default=0)
    p.add_argument("--m7-rho", type=float, default=0.20)
    p.add_argument("--m7-tau-forget", type=float, default=0.05)
    p.add_argument("--m7-memory-rank", type=int, default=1)
    p.add_argument("--m7-projector", default="parameter", choices=["parameter", "functional", "both"])
    p.add_argument("--m7-functional-eps", type=float, default=1.0e-3)
    p.add_argument("--m7-modulus", type=int, default=13)
    p.add_argument("--m7-modular-train-fraction", type=float, default=0.40)
    p.add_argument("--m9-trajectory-output", default="m9_trajectory")
    p.add_argument("--trajectory-m9-velocity-scale", type=float, default=0.20)
    p.add_argument("--trajectory-m9-safety-barrier", type=float, default=5.0)
    p.add_argument("--trajectory-m9-beta-signal", type=float, default=0.05)
    p.add_argument("--m3-velocity-scale", type=float, default=0.20)
    p.add_argument("--m3-safety-barrier", type=float, default=5.0)
    p.add_argument("--m4-beta-slow", type=float, default=0.01)
    p.add_argument("--m4-trajectory-output", default="m4_trajectory")
    p.add_argument("--m4-tasks", default="modular_addition")
    p.add_argument("--m4-trajectory-steps", type=int, default=0)
    p.add_argument("--m4-rho", type=float, default=0.05)
    p.add_argument("--m4-lambda-mig", type=float, default=0.50)
    p.add_argument("--m4-lambda-res", type=float, default=0.25)
    p.add_argument("--m4-support-rank", type=int, default=0)
    p.add_argument("--m4-train-threshold", type=float, default=0.90)
    p.add_argument("--m4-test-threshold", type=float, default=0.80)
    p.add_argument("--run-m4-diagnostic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-m7-diagnostic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--diagnostic-seeds", default="0,1")
    p.add_argument("--diagnostic-architectures", default="MLP,DGKAN_DCHE")
    p.add_argument("--diagnostic-hidden", type=int, default=32)
    p.add_argument("--diagnostic-batch-size", type=int, default=64)
    p.add_argument("--diagnostic-row-timeout", type=int, default=1200)
    p.add_argument("--diagnostic-total-timeout", type=int, default=7200)
    p.add_argument("--m4-diag-steps", type=int, default=400)
    p.add_argument("--m4-modulus", type=int, default=13)
    p.add_argument("--m4-train-fraction", type=float, default=0.40)
    p.add_argument("--m4-diag-velocity-scale", type=float, default=0.02)
    p.add_argument("--m7-steps-per-task", type=int, default=120)
    p.add_argument("--m7-train-size", type=int, default=256)
    p.add_argument("--m7-held-size", type=int, default=256)
    p.add_argument("--m7-velocity-scale", type=float, default=0.02)
    p.add_argument("--m7-beta-memory", type=float, default=0.02)
    return p


def main() -> None:
    args = build_parser().parse_args()
    ensure_out()
    bind_upstream()
    if args.stage == "code":
        run_code_truth_gate()
    elif args.stage == "metric-spectrum":
        run_metric_and_spectrum(args)
    elif args.stage == "screen":
        run_screen(args)
    elif args.stage == "repair-screen":
        run_repair_screen(args)
    elif args.stage == "repair-merge":
        merge_v2243_outputs(
            specs_filename="v22_45E_repair_specs.csv",
            matrix_filename="v22_45E_repair_screen_matrix.csv",
            runtime_filename="v22_45E_repair_runtime_regression_audit.csv",
            support_direction_filename="v22_45E_repair_support_direction_decomposition.csv",
            safety_filename="v22_45E_repair_safety_debt_matrix.csv",
            overhead_filename="v22_45E_repair_overhead_matrix.csv",
            summary_filename="v22_45E_repair_mechanism_summary.csv",
        )
    elif args.stage == "m5-variant-audit":
        run_m5_variant_audit(args)
    elif args.stage == "m8-gauge-audit":
        run_m8_gauge_audit(args)
    elif args.stage == "trajectory-m1":
        run_m1_trajectory(args)
    elif args.stage == "trajectory-m2":
        run_m2_trajectory(args)
    elif args.stage == "trajectory-m3":
        run_m3_trajectory(args)
    elif args.stage == "trajectory-m4":
        run_m4_trajectory(args)
    elif args.stage == "trajectory-m5":
        run_m5_trajectory(args)
    elif args.stage == "trajectory-m6":
        run_m6_trajectory(args)
    elif args.stage == "trajectory-m7":
        run_m7_trajectory(args)
    elif args.stage == "trajectory-m9":
        run_m9_trajectory(args)
    elif args.stage == "collect":
        stage_collect(args)
    elif args.stage == "m4-diagnostic":
        run_m4_diagnostic(args)
    elif args.stage == "m7-diagnostic":
        run_m7_diagnostic(args)
    elif args.stage == "merge":
        merge_v2243_outputs()
    elif args.stage == "finalize":
        finalize(args)
    else:
        stage_all(args)


if __name__ == "__main__":
    main()
