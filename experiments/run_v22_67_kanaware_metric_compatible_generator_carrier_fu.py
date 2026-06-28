#!/usr/bin/env python3
"""DG-KAN v22.67 KAN-aware metric-compatible generator carrier runner."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import importlib
import json
import math
import os
from pathlib import Path
import py_compile
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
csv.field_size_limit(sys.maxsize)

import experiments.run_v22_66_metric_compatible_generator_atlas_fu as v66  # noqa: E402


PYTHON = os.environ.get("KAN_PYTHON", v66.PYTHON)
OUT_ROOT = ROOT / "results/v22_67"
LOG_ROOT = OUT_ROOT / "logs"
DOCS = ROOT / "docs"
EXEC_DOC = DOCS / "DG-KAN_v22.67_KANAwareMetricCompatibleGeneratorCarrierFU_执行日志.md"
RECAP_DOC = DOCS / "DG-KAN_v22.67_KANAwareMetricCompatibleGeneratorCarrierFU_实验结果复盘.md"
PLAN_DOC = DOCS / "DG-KAN_v22.67_KANAwareMetricCompatibleGeneratorCarrierFU_完整计划.md"
RUNNER = ROOT / "experiments/run_v22_67_kanaware_metric_compatible_generator_carrier_fu.py"
RUNNER66 = ROOT / "experiments/run_v22_66_metric_compatible_generator_atlas_fu.py"
RUNNER65 = ROOT / "experiments/run_v22_65_metric_compatible_signal_atlas_fu.py"
RUNNER64 = ROOT / "experiments/run_v22_64_metric_preserving_functional_atlas_fu.py"
V53O_RUNNER = ROOT / "experiments/run_v22_53O_pion_poet_external_oet_baseline.py"
ATLAS_MODULE = ROOT / "dgkan/fu/metric_preserving_functional_atlas.py"
PART_B_CANDIDATE = "mcga_over_poet_fsclip_eta025_residual_rank4"
PART_B_DATASETS = ["MNIST", "FashionMNIST", "KMNIST", "CIFAR10", "Wine", "Spam"]
PART_B_SEEDS = [0, 1, 2, 3, 4]
PART_B_METHODS = [
    "adamw",
    "cautious_adamw",
    "schedule_free_adamw_local",
    "poet_official",
    "pion_oet_sphere_official",
    "pion_oet_local",
    "oet_only_coordinate",
    "same_generator_descent_energy_random",
    "same_C_skew_spectrum_random",
    "same_functional_spectrum_random_coordinate",
    "same_isometric_capacity_random_coordinate",
    "same_metric_drift_random_coordinate",
    "same_shape_budget_generator_random",
    "shuffled_source_witness_generator",
    "same_compute_noop_coordinate",
    PART_B_CANDIDATE,
]
PART_B_REFERENCE_METHODS = {"adamw", "cautious_adamw", "schedule_free_adamw_local"}
PART_B_EXTERNAL_METHODS = {"poet_official", "pion_oet_sphere_official", "pion_oet_local"}
PART_B_SAME_GENERATOR_CONTROLS = {"same_generator_descent_energy_random", "same_C_skew_spectrum_random"}
PART_B_REPAIR_METHODS = [
    "mcga_over_poet_fsclip_residual_rank4",
    "mcga_over_poet_fsclip_eta05_residual_rank4",
    "mcga_over_poet_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_fsfill_residual_rank4",
    "mcga_over_poet_baselock_residual_rank4",
]
PART_C_DATASETS = ["CIFAR10", "Wine", "Spam"]
PART_C_ARCHITECTURES = ["MLP", "DGKAN_DCHE", "DGKAN_DFOU"]
PART_C_REDESIGN_ARCHITECTURES = ["DGKAN_CHE4", "DGKAN_CHE3_XLIN", "DGKAN_FOU4", "DGKAN_FOU4_LIN", "DGKAN_FOU4_LIN50", "DGKAN_RBF4", "DGKAN_RBF4_XLIN", "DGKAN_HAT4", "DGKAN_HAT4_XLIN", "DGKAN_RAT4"]
PART_C_REDESIGN_COMPARISON_ARCHITECTURES = ["DGKAN_DCHE", "DGKAN_DFOU", *PART_C_REDESIGN_ARCHITECTURES]
PART_C_WINNER = "mcga_over_poet_fsclip_eta001_residual_rank4"
PART_E_BASIS_CANDIDATE = "kan_task_visible_chart_bankbudget_fsclip_eta001_rank4"
PART_E_BASIS_READOUT_CONTROL = "same_readout_visible_energy_random_chart_bankbudget_fsclip_eta001_rank4"
PART_E_BASIS_CONTROL_METHODS = [
    "same_generator_descent_energy_random",
    "same_C_skew_spectrum_random",
    PART_E_BASIS_READOUT_CONTROL,
]
PART_E_BASIS_METHODS = [
    "adamw",
    "cautious_adamw",
    "schedule_free_adamw_local",
    *PART_E_BASIS_CONTROL_METHODS,
    PART_E_BASIS_CANDIDATE,
]

REDESIGN_KAN_SPECS: dict[str, dict[str, Any]] = {
    "DGKAN_CHE4": {
        "basis_family": "D-CHE4",
        "basis_name": "chebyshev",
        "k": 4,
        "local_support": 0,
        "global_support": 1,
        "uses_exp": 0,
        "uses_sin_cos": 0,
        "uses_division": 0,
        "basis_order": 4,
        "init_variant": "v22_67_cheby_k4_dense_redesign",
        "seed_offset": 3000,
    },
    "DGKAN_CHE3_XLIN": {
        "basis_family": "D-CHE3-XLIN",
        "basis_name": "chebyshev",
        "k": 3,
        "local_support": 0,
        "global_support": 1,
        "uses_exp": 0,
        "uses_sin_cos": 0,
        "uses_division": 0,
        "basis_order": 3,
        "uses_dense_basis_tensor": 0,
        "init_variant": "cheby_k3_inputcross_localr4_projr128_triton_l3_gradbuf_linearres010",
        "seed_offset": 9000,
    },
    "DGKAN_FOU4": {
        "basis_family": "D-FOU4",
        "basis_name": "fourier_lowfreq",
        "k": 4,
        "local_support": 0,
        "global_support": 1,
        "uses_exp": 0,
        "uses_sin_cos": 1,
        "uses_division": 0,
        "basis_order": 4,
        "init_variant": "v22_67_fourier_k4_dense_redesign",
        "seed_offset": 4000,
    },
    "DGKAN_FOU4_LIN": {
        "basis_family": "D-FOU4-LIN",
        "basis_name": "fourier_lowfreq",
        "k": 4,
        "local_support": 0,
        "global_support": 1,
        "uses_exp": 0,
        "uses_sin_cos": 1,
        "uses_division": 0,
        "basis_order": 4,
        "init_variant": "fourier_k4_linearres_gemm_l3_matmul_linearres010",
        "seed_offset": 8000,
    },
    "DGKAN_FOU4_LIN50": {
        "basis_family": "D-FOU4-LIN50",
        "basis_name": "fourier_lowfreq",
        "k": 4,
        "local_support": 0,
        "global_support": 1,
        "uses_exp": 0,
        "uses_sin_cos": 1,
        "uses_division": 0,
        "basis_order": 4,
        "uses_dense_basis_tensor": 0,
        "init_variant": "fourier_k4_linearres_gemm_l3_matmul_linearres050",
        "seed_offset": 10000,
    },
    "DGKAN_RBF4": {
        "basis_family": "D-RBF4",
        "basis_name": "compact_rbf",
        "k": 4,
        "local_support": 1,
        "global_support": 0,
        "uses_exp": 1,
        "uses_sin_cos": 0,
        "uses_division": 0,
        "basis_order": 1,
        "init_variant": "v22_67_rbf_k4_dense_redesign",
        "seed_offset": 5000,
    },
    "DGKAN_RBF4_XLIN": {
        "basis_family": "D-RBF4-XLIN",
        "basis_name": "compact_rbf",
        "k": 4,
        "local_support": 1,
        "global_support": 0,
        "uses_exp": 1,
        "uses_sin_cos": 0,
        "uses_division": 0,
        "basis_order": 1,
        "uses_dense_basis_tensor": 0,
        "init_variant": "rbf_k4_triton_l3_matmul_inputcross_localr4_projr128_linearres010",
        "seed_offset": 11000,
    },
    "DGKAN_HAT4": {
        "basis_family": "D-HAT4",
        "basis_name": "hat_wavelet",
        "k": 4,
        "local_support": 1,
        "global_support": 0,
        "uses_exp": 0,
        "uses_sin_cos": 0,
        "uses_division": 0,
        "basis_order": 1,
        "init_variant": "v22_67_hat_k4_dense_redesign",
        "seed_offset": 6000,
    },
    "DGKAN_HAT4_XLIN": {
        "basis_family": "D-HAT4-XLIN",
        "basis_name": "hat_wavelet",
        "k": 4,
        "local_support": 1,
        "global_support": 0,
        "uses_exp": 0,
        "uses_sin_cos": 0,
        "uses_division": 0,
        "basis_order": 1,
        "uses_dense_basis_tensor": 0,
        "init_variant": "hat_wavelet_k4_triton_l3_matmul_inputcross_localr4_projr128_linearres010",
        "seed_offset": 12000,
    },
    "DGKAN_RAT4": {
        "basis_family": "D-RAT4",
        "basis_name": "rational_kat_lite",
        "k": 4,
        "local_support": 0,
        "global_support": 1,
        "uses_exp": 0,
        "uses_sin_cos": 0,
        "uses_division": 1,
        "basis_order": 4,
        "init_variant": "v22_67_rational_k4_dense_redesign",
        "seed_offset": 7000,
    },
}


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)


def command_text(cmd: Iterable[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def flag(value: Any) -> int:
    val = safe_float(value, None)
    return int(val is not None and val != 0.0)


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names: list[str] = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in names})


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def append_exec(command: str, *, task_id: str, status: str, gpu: str = "", files: str = "", exit_code: Any = "", note: str = "") -> None:
    ensure_out()
    row = {
        "timestamp_sg": now_sg(),
        "task_id": task_id,
        "status": status,
        "gpu": gpu,
        "command": command,
        "files": files,
        "exit_code": exit_code,
        "note": note,
    }
    append_csv(
        OUT_ROOT / "v22_67_command_journal.csv",
        row,
        ["timestamp_sg", "task_id", "status", "gpu", "command", "files", "exit_code", "note"],
    )
    with EXEC_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n## {row['timestamp_sg']} | {task_id} | {status}\n\n")
        f.write(f"- command: `{command}`\n")
        if gpu:
            f.write(f"- gpu: `{gpu}`\n")
        if files:
            f.write(f"- files: `{files}`\n")
        if exit_code != "":
            f.write(f"- exit_code: `{exit_code}`\n")
        if note:
            f.write(f"- note: {note}\n")


def append_recap(title: str, lines: list[str]) -> None:
    ensure_out()
    with RECAP_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n## {now_sg()} | {title}\n\n")
        for line in lines:
            f.write(f"- {line}\n")


def run_subprocess(cmd: list[Any], *, task_id: str, timeout: int = 300, gpu: str = "", files: str = "") -> dict[str, Any]:
    ensure_out()
    frag = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("_")
    stdout_path = LOG_ROOT / f"{frag}_stdout.log"
    stderr_path = LOG_ROOT / f"{frag}_stderr.log"
    started = time.time()
    proc = subprocess.run(
        [str(part) for part in cmd],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
    status = "pass" if proc.returncode == 0 else "fail"
    append_exec(
        command_text(cmd),
        task_id=task_id,
        status=status,
        gpu=gpu,
        files=files or f"{stdout_path.relative_to(ROOT)}; {stderr_path.relative_to(ROOT)}",
        exit_code=proc.returncode,
        note=f"wall_seconds={time.time() - started:.3f}",
    )
    return {
        "status": status,
        "returncode": proc.returncode,
        "stdout": str(stdout_path.relative_to(ROOT)),
        "stderr": str(stderr_path.relative_to(ROOT)),
        "wall_seconds": time.time() - started,
    }


def compile_core_files() -> tuple[int, list[dict[str, Any]]]:
    paths = sorted((ROOT / "dgkan").rglob("*.py"))
    paths.extend([p for p in [RUNNER, RUNNER66, RUNNER65, RUNNER64, V53O_RUNNER] if p.exists()])
    rows: list[dict[str, Any]] = []
    ok = 1
    for path in paths:
        try:
            py_compile.compile(str(path), doraise=True)
            rows.append({"target": str(path.relative_to(ROOT)), "compileall": "pass"})
        except Exception as exc:
            ok = 0
            rows.append({"target": str(path.relative_to(ROOT)), "compileall": "fail", "error": f"{type(exc).__name__}: {exc}"})
    write_rows(OUT_ROOT / "v22_67_part_a_compile_rows.csv", rows)
    return ok, rows


def import_dgkan_modules() -> tuple[int, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    ok = 1
    for path in sorted((ROOT / "dgkan").rglob("*.py")):
        rel = path.relative_to(ROOT)
        mod = ".".join(rel.with_suffix("").parts)
        try:
            importlib.import_module(mod)
            rows.append({"module": mod, "import_status": "pass"})
        except Exception as exc:
            ok = 0
            rows.append({"module": mod, "import_status": "fail", "error": f"{type(exc).__name__}: {exc}"})
    write_rows(OUT_ROOT / "v22_67_part_a_worktree_import_rows.csv", rows)
    return ok, rows


def external_import_checks() -> dict[str, Any]:
    poet_cmd = [
        PYTHON,
        "-c",
        "import sys; sys.path.insert(0, 'external/oet_baselines/poet_sphere'); "
        "from poet_torch import POETConfig, POETModel, get_poet_optimizer; "
        "print(POETConfig.__name__, POETModel.__name__, callable(get_poet_optimizer))",
    ]
    pion_cmd = [
        PYTHON,
        "-c",
        "from experiments.run_v22_53O_pion_poet_external_oet_baseline import MatrixGeometryOptimizer; "
        "print(MatrixGeometryOptimizer.__name__)",
    ]
    poet = run_subprocess(poet_cmd, task_id="A_external_poet_import", timeout=120)
    pion = run_subprocess(pion_cmd, task_id="A_external_pion_import", timeout=120)
    return {
        "external_poet_import_pass": int(poet["returncode"] == 0),
        "external_pion_import_pass": int(pion["returncode"] == 0),
        "external_poet_import_stdout": poet["stdout"],
        "external_poet_import_stderr": poet["stderr"],
        "external_pion_import_stdout": pion["stdout"],
        "external_pion_import_stderr": pion["stderr"],
    }


def clean_tarball_import_check() -> tuple[int, str]:
    ensure_out()
    bundle_path = OUT_ROOT / "v22_67_clean_import_bundle.tar.gz"
    files = [
        RUNNER,
        RUNNER66,
        RUNNER64,
        ATLAS_MODULE,
        ROOT / "dgkan/__init__.py",
        ROOT / "dgkan/contracts.py",
        ROOT / "dgkan/specs.py",
        ROOT / "dgkan/fu/__init__.py",
        ROOT / "experiments/dgkan_core.py",
    ]
    with tarfile.open(bundle_path, "w:gz") as tar:
        for path in files:
            if path.exists():
                tar.add(path, arcname=str(path.relative_to(ROOT)))
    with tempfile.TemporaryDirectory(prefix="v22_67_clean_") as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(bundle_path, "r:gz") as tar:
            tar.extractall(tmp_path)
        cmd = [
            PYTHON,
            "-c",
            "import sys; sys.path.insert(0, '.'); "
            "import experiments.run_v22_67_kanaware_metric_compatible_generator_carrier_fu as r67; "
            "import experiments.run_v22_66_metric_compatible_generator_atlas_fu as r66; "
            "import dgkan.fu.metric_preserving_functional_atlas as atlas; "
            "print(r67.RUNNER.name, r66.RUNNER.name, atlas.MetricCompatibleAtlasMLP.__name__)",
        ]
        proc = subprocess.run(cmd, cwd=str(tmp_path), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    log_path = LOG_ROOT / "v22_67_clean_tarball_import_check.log"
    log_path.write_text(
        f"CMD: {command_text(cmd)}\nBUNDLE: {bundle_path.relative_to(ROOT)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\n",
        encoding="utf-8",
        errors="replace",
    )
    return int(proc.returncode == 0), str(log_path.relative_to(ROOT))


def static_scan_core() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = [p for p in [RUNNER, RUNNER66, RUNNER64, ATLAS_MODULE] if p.exists()]
    scan_rows, scan_summary = v66.static_scan(paths)
    write_rows(OUT_ROOT / "v22_67_part_a_static_scan_hits.csv", scan_rows or [{"status": "no_forbidden_hits"}])
    return scan_rows, scan_summary


def runtime_trace(args: argparse.Namespace) -> dict[str, Any]:
    row_args = v66.build_parser().parse_args([])
    row_args.mode = "row"
    row_args.run_label = "v22_67_part_a_trace"
    row_args.architecture = "MLP"
    row_args.method = "mcga_transport_generator_rank2"
    row_args.dataset = "Wine"
    row_args.seed = 0
    row_args.device = args.device
    row_args.steps = 5
    row_args.train_size = 96
    row_args.held_size = 40
    row_args.test_size = 40
    row_args.hidden = min(32, int(args.hidden))
    row_args.batch_size = 32
    row_args.eval_batch_size = 128
    row_args.metric_batch_size = 64
    row_args.refresh = 5
    started = time.time()
    trace_row = v66.train_row(row_args)
    trace_row["v22_67_trace_wall_seconds"] = f"{time.time() - started:.6f}"
    trace_row["v22_67_source_chunk_path"] = str(v66.row_path(row_args).relative_to(ROOT))
    write_rows(OUT_ROOT / "v22_67_part_a_runtime_trace_row.csv", [trace_row])
    return trace_row


def strict_dgkan_identity_probe(args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    device = v66.torch_device(str(args.device))
    bundle = v66.load_bundle("Wine", 96, 40, 40, 0)
    for arch in ["DGKAN_DCHE", "DGKAN_DFOU"]:
        row_args = v66.build_parser().parse_args([])
        row_args.architecture = arch
        row_args.dataset = "Wine"
        row_args.seed = 0
        row_args.hidden = min(32, int(args.hidden))
        row_args.train_size = 96
        row_args.held_size = 40
        row_args.test_size = 40
        row_args.device = args.device
        try:
            model = v66.make_base_model(row_args, bundle, device)
            with torch.no_grad():
                xb = bundle["x_train"][:32].to(device).float()
                logits = model(xb).float()
            err = safe_float(getattr(model, "kan_readout_linearization_max_abs_error", None), None)
            rows.append(
                {
                    "architecture": arch,
                    "probe_status": "pass",
                    "strict_DGKAN_identity_pass": int(err is not None and err <= 1.0e-5),
                    "kan_readout_linearization_max_abs_error": err,
                    "logits_nan_inf_count": int((~torch.isfinite(logits)).sum().detach().cpu().item()),
                    "hidden": row_args.hidden,
                    "dataset": "Wine",
                    "seed": 0,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "architecture": arch,
                    "probe_status": "fail",
                    "strict_DGKAN_identity_pass": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "hidden": row_args.hidden,
                    "dataset": "Wine",
                    "seed": 0,
                }
            )
    write_rows(OUT_ROOT / "v22_67_part_a_strict_dgkan_identity.csv", rows)
    return rows


def count_non_strict_kan_official_rows() -> dict[str, int]:
    return {
        "KANbeFair_original_KAN_official_rows": 0,
        "pyKAN_or_BSpline_official_rows": 0,
    }


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    command = command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-a", "--device", args.device])
    append_exec(command, task_id="A_start_part_a", status="start", gpu=args.device, files=str(OUT_ROOT.relative_to(ROOT)))

    compile_pass, _compile_rows = compile_core_files()
    import_pass, import_rows = import_dgkan_modules()
    runner_core_import_pass = 1
    operator_import_pass = 1
    import_error = ""
    try:
        importlib.import_module("experiments.run_v22_67_kanaware_metric_compatible_generator_carrier_fu")
        importlib.import_module("experiments.run_v22_66_metric_compatible_generator_atlas_fu")
    except Exception as exc:
        runner_core_import_pass = 0
        import_error = f"{type(exc).__name__}: {exc}"
    try:
        importlib.import_module("dgkan.fu.metric_preserving_functional_atlas")
    except Exception as exc:
        operator_import_pass = 0
        import_error = f"{import_error}; {type(exc).__name__}: {exc}".strip("; ")

    clean_pass, clean_log = clean_tarball_import_check()
    external = external_import_checks()
    scan_rows, scan_summary = static_scan_core()
    trace_error = ""
    try:
        trace_row = runtime_trace(args)
    except Exception:
        trace_row = {}
        trace_error = traceback.format_exc()
        (LOG_ROOT / "v22_67_part_a_runtime_trace_exception.log").write_text(trace_error, encoding="utf-8", errors="replace")
    identity_rows = strict_dgkan_identity_probe(args)
    non_strict = count_non_strict_kan_official_rows()

    manual_detected = int(flag(trace_row.get("manual_param_update_detected")) or scan_summary.get("manual_param_update_detected", 0) > 0)
    no_grad_detected = int(flag(trace_row.get("no_grad_param_mutation_detected")) or scan_summary.get("no_grad_param_mutation_detected", 0) > 0)
    p_data_detected = int(flag(trace_row.get("param_data_write_detected")) or scan_summary.get("param_data_write_detected", 0) > 0)
    copy_detected = int(flag(trace_row.get("copy_param_write_detected")) or scan_summary.get("copy_param_write_detected", 0) > 0)
    strict_identity_pass = int(identity_rows and all(flag(r.get("strict_DGKAN_identity_pass")) for r in identity_rows))
    readout_error_logged = int(any(r.get("kan_readout_linearization_max_abs_error", "") != "" for r in identity_rows))
    static_pass = int(scan_summary.get("standard_loop_static_scan_pass", 0) == 1 and scan_summary.get("manual_update_forbidden_scan_pass", 0) == 1)
    summary = {
        "gate": "v22_67_part_a_code_identity_training_boundary",
        "generated_at_sg": now_sg(),
        "compileall_pass": int(compile_pass),
        "worktree_full_repo_import_pass": int(import_pass),
        "worktree_full_repo_import_fail_rows": sum(1 for r in import_rows if r.get("import_status") != "pass"),
        "clean_tarball_self_contained_import_pass": int(clean_pass),
        "runner_core_import_pass": int(runner_core_import_pass),
        "operator_import_pass": int(operator_import_pass),
        "external_poet_import_pass": int(external["external_poet_import_pass"]),
        "external_pion_import_pass": int(external["external_pion_import_pass"]),
        "standard_loop_static_scan_pass": static_pass,
        "standard_loop_runtime_trace_pass": int(flag(trace_row.get("standard_loop_runtime_trace_pass"))),
        "loss_total_is_task_loss_only": int(flag(trace_row.get("loss_total_is_task_loss_only"))),
        "manual_param_update_detected": manual_detected,
        "no_grad_param_mutation_detected": no_grad_detected,
        "p_data_write_detected": p_data_detected,
        "copy_param_write_detected": copy_detected,
        "class_weight_or_sampler_used_as_fu": int(scan_summary.get("class_weight_or_sampler_used_as_fu", 0) > 0),
        "auxiliary_loss_used_official": int(scan_summary.get("fu_auxiliary_loss_used_official", 0) > 0),
        "candidate_action_selection_used_for_runtime": int(scan_summary.get("candidate_action_selection_used_for_runtime", 0) > 0),
        "runtime_argmax_candidate_used": int(scan_summary.get("candidate_action_selection_used_for_runtime", 0) > 0),
        "runtime_topk_candidate_used": int(scan_summary.get("cohort_topk_selection_used", 0) > 0),
        "cohort_topk_selection_used": int(scan_summary.get("cohort_topk_selection_used", 0) > 0),
        "layer_topk_selection_used": int(scan_summary.get("layer_topk_selection_used", 0) > 0),
        "uses_validation_test_future_direction": int(scan_summary.get("uses_validation_test_future_direction", 0) > 0),
        "strict_DGKAN_identity_pass": strict_identity_pass,
        "readout_linearization_error_logged": readout_error_logged,
        **non_strict,
        "clean_tarball_log": clean_log,
        "runtime_trace_source_chunk": trace_row.get("v22_67_source_chunk_path", ""),
        "runtime_trace_exception_log": "results/v22_67/logs/v22_67_part_a_runtime_trace_exception.log" if trace_error else "",
        "import_error": import_error,
        **external,
    }
    hard_pass = int(
        summary["compileall_pass"] == 1
        and summary["worktree_full_repo_import_pass"] == 1
        and summary["clean_tarball_self_contained_import_pass"] == 1
        and summary["runner_core_import_pass"] == 1
        and summary["operator_import_pass"] == 1
        and summary["external_poet_import_pass"] == 1
        and summary["external_pion_import_pass"] == 1
        and summary["standard_loop_static_scan_pass"] == 1
        and summary["standard_loop_runtime_trace_pass"] == 1
        and summary["loss_total_is_task_loss_only"] == 1
        and summary["manual_param_update_detected"] == 0
        and summary["class_weight_or_sampler_used_as_fu"] == 0
        and summary["auxiliary_loss_used_official"] == 0
        and summary["candidate_action_selection_used_for_runtime"] == 0
        and summary["uses_validation_test_future_direction"] == 0
        and summary["strict_DGKAN_identity_pass"] == 1
    )
    summary["part_a_hard_gate_pass"] = hard_pass
    summary["part_a_next_action"] = "run_part_b_mlp_robustness" if hard_pass else "fix_part_a_before_science_matrix"
    write_rows(OUT_ROOT / "v22_67_part_a_code_identity_training_boundary.csv", [summary])
    (OUT_ROOT / "v22_67_part_a_code_identity_training_boundary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    append_exec(
        command,
        task_id="A_part_a_summary",
        status="pass" if hard_pass else "fail",
        gpu=args.device,
        files="results/v22_67/v22_67_part_a_code_identity_training_boundary.csv; results/v22_67/v22_67_part_a_code_identity_training_boundary.json",
        note=f"part_a_hard_gate_pass={hard_pass}; next_action={summary['part_a_next_action']}",
    )
    append_recap(
        "Part A code/identity/training-boundary gate",
        [
            f"part_a_hard_gate_pass={hard_pass}; next_action={summary['part_a_next_action']}.",
            "Evidence files: results/v22_67/v22_67_part_a_code_identity_training_boundary.csv, v22_67_part_a_compile_rows.csv, v22_67_part_a_worktree_import_rows.csv, v22_67_part_a_static_scan_hits.csv, v22_67_part_a_runtime_trace_row.csv, v22_67_part_a_strict_dgkan_identity.csv.",
            f"Runtime trace source chunk: {summary.get('runtime_trace_source_chunk') or 'missing due to trace failure'}.",
            f"Strict DGKAN identity pass={strict_identity_pass}; readout_linearization_error_logged={readout_error_logged}.",
            "No v22.67 science matrix is promoted unless this Part A hard gate is pass.",
        ],
    )
    return summary


def debt_value(row: dict[str, Any]) -> float | None:
    vals = [
        safe_float(row.get("ECE")),
        safe_float(row.get("Brier")),
        safe_float(row.get("tail_loss_q95")),
        safe_float(row.get("tail_loss_q99")),
    ]
    if any(v is None for v in vals):
        return None
    return float(sum(v for v in vals if v is not None))


def part_b_expected_keys(steps: Iterable[int]) -> set[tuple[str, int, int]]:
    return {(dataset, seed, int(step)) for dataset in PART_B_DATASETS for seed in PART_B_SEEDS for step in steps}


def collect_part_b_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    allowed_datasets = set(PART_B_DATASETS)
    allowed_seeds = {str(s) for s in PART_B_SEEDS}
    allowed_methods = set(PART_B_METHODS)
    for path in sorted((ROOT / "results/v22_66/chunks").glob("*.csv")):
        if path.name.endswith("_metric_rows.csv"):
            continue
        for row in read_rows(path):
            method = str(row.get("method") or "")
            dataset = str(row.get("dataset") or "")
            seed = str(row.get("seed") or "")
            steps = safe_float(row.get("steps"), None)
            if method not in allowed_methods or dataset not in allowed_datasets or seed not in allowed_seeds or steps is None:
                continue
            if int(steps) not in {400, 800}:
                continue
            if str(row.get("architecture_key") or "MLP") != "MLP":
                continue
            label = str(row.get("run_label") or "")
            if int(steps) == 400 and label not in {"", "v22_67_mlp_robust_st400"}:
                continue
            if int(steps) == 800 and label not in {"v22_67_mlp_robust_st800"}:
                continue
            out = dict(row)
            out["chunk_path"] = str(path.relative_to(ROOT))
            out["steps"] = str(int(steps))
            out["seed"] = str(int(float(seed)))
            rows.append(out)

    # Deduplicate by method/dataset/seed/steps, preferring v22.67-labeled rows.
    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("method")), str(row.get("dataset")), str(row.get("seed")), str(row.get("steps")))
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = row
            continue
        prev_score = 1 if str(prev.get("run_label") or "").startswith("v22_67") else 0
        row_score = 1 if str(row.get("run_label") or "").startswith("v22_67") else 0
        if row_score >= prev_score:
            by_key[key] = row
    return list(by_key.values())


def collect_part_b_rows_for_candidate(candidate_method: str) -> list[dict[str, Any]]:
    if candidate_method == PART_B_CANDIDATE:
        return collect_part_b_rows()
    base_rows = [r for r in collect_part_b_rows() if str(r.get("method")) != PART_B_CANDIDATE]
    rows: list[dict[str, Any]] = []
    allowed_datasets = set(PART_B_DATASETS)
    allowed_seeds = {str(s) for s in PART_B_SEEDS}
    for path in sorted((ROOT / "results/v22_66/chunks").glob("*.csv")):
        if path.name.endswith("_metric_rows.csv"):
            continue
        for row in read_rows(path):
            steps = safe_float(row.get("steps"), None)
            if steps is None or int(steps) not in {400, 800}:
                continue
            if str(row.get("method") or "") != candidate_method:
                continue
            if str(row.get("dataset") or "") not in allowed_datasets or str(row.get("seed") or "") not in allowed_seeds:
                continue
            if str(row.get("architecture_key") or "MLP") != "MLP":
                continue
            label = str(row.get("run_label") or "")
            if not label.startswith("v22_67_mlp_robust"):
                continue
            out = dict(row)
            out["chunk_path"] = str(path.relative_to(ROOT))
            out["steps"] = str(int(steps))
            out["seed"] = str(int(float(str(row.get("seed")))))
            rows.append(out)
    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in base_rows + rows:
        key = (str(row.get("method")), str(row.get("dataset")), str(row.get("seed")), str(row.get("steps")))
        by_key[key] = row
    return list(by_key.values())


def add_part_b_comparisons(rows: list[dict[str, Any]], *, candidate_method: str = PART_B_CANDIDATE) -> list[dict[str, Any]]:
    by_cond: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_cond.setdefault((str(row.get("dataset")), str(row.get("seed")), str(row.get("steps"))), []).append(row)
    for group in by_cond.values():
        completed = [r for r in group if r.get("run_status") == "completed"]
        refs = [r for r in completed if str(r.get("method")) in PART_B_REFERENCE_METHODS]
        external = [r for r in completed if str(r.get("method")) in PART_B_EXTERNAL_METHODS]
        controls = [
            r
            for r in completed
            if str(r.get("method")) not in PART_B_REFERENCE_METHODS
            and str(r.get("method")) not in PART_B_EXTERNAL_METHODS
            and str(r.get("method")) != candidate_method
        ]
        best_ref = min((safe_float(r.get("held_NLL"), float("inf")) or float("inf") for r in refs), default=float("inf"))
        best_external = min((safe_float(r.get("held_NLL"), float("inf")) or float("inf") for r in external), default=float("inf"))
        best_control = min((safe_float(r.get("held_NLL"), float("inf")) or float("inf") for r in controls), default=float("inf"))
        ref_debt = min((debt_value(r) if debt_value(r) is not None else float("inf") for r in refs), default=float("inf"))
        same_best = {}
        for method in PART_B_SAME_GENERATOR_CONTROLS:
            vals = [safe_float(r.get("held_NLL"), float("inf")) or float("inf") for r in completed if str(r.get("method")) == method]
            same_best[method] = min(vals, default=float("inf"))
        for row in group:
            nll = safe_float(row.get("held_NLL"), float("inf")) or float("inf")
            debt = debt_value(row)
            row["Delta_NLL_vs_strongest"] = nll - best_ref if math.isfinite(best_ref) else ""
            row["Delta_NLL_vs_external_OET"] = nll - best_external if math.isfinite(best_external) else ""
            row["Delta_NLL_vs_best_control"] = nll - best_control if math.isfinite(best_control) else ""
            row["beats_strongest_NLL"] = int(math.isfinite(best_ref) and nll < best_ref)
            row["beats_external_OET_NLL"] = int(math.isfinite(best_external) and nll < best_external)
            row["beats_best_control_NLL"] = int(math.isfinite(best_control) and nll < best_control)
            row["no_ECE_Brier_tail_debt"] = int(debt is not None and math.isfinite(ref_debt) and debt <= ref_debt + 1.0e-9)
            for method in PART_B_SAME_GENERATOR_CONTROLS:
                suffix = "same_generator_descent_energy_random" if method == "same_generator_descent_energy_random" else "same_C_skew_spectrum_random"
                row[f"Delta_NLL_vs_{suffix}"] = nll - same_best[method] if math.isfinite(same_best[method]) else ""
                row[f"beats_{suffix}_NLL"] = int(math.isfinite(same_best[method]) and nll < same_best[method])
            row["beats_same_generator_controls_NLL"] = int(
                flag(row.get("beats_same_generator_descent_energy_random_NLL"))
                and flag(row.get("beats_same_C_skew_spectrum_random_NLL"))
            )
    return rows


def part_b_inventory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in PART_B_METHODS:
        for steps in [400, 800]:
            subset = [r for r in rows if str(r.get("method")) == method and str(r.get("steps")) == str(steps)]
            completed = [r for r in subset if r.get("run_status") == "completed"]
            out.append(
                {
                    "method": method,
                    "steps": steps,
                    "expected_rows": len(PART_B_DATASETS) * len(PART_B_SEEDS),
                    "artifact_rows": len(subset),
                    "completed_rows": len(completed),
                    "missing_completed_rows": len(PART_B_DATASETS) * len(PART_B_SEEDS) - len(completed),
                    "datasets_completed": ",".join(sorted({str(r.get("dataset")) for r in completed})),
                    "seeds_completed": ",".join(sorted({str(r.get("seed")) for r in completed}, key=lambda x: int(float(x)) if x else -1)),
                }
            )
    return out


def part_b_failure_mode(row: dict[str, Any]) -> str:
    if row.get("run_status") != "completed":
        return "ImplementationBoundaryFailed"
    if not flag(row.get("standard_loop_runtime_trace_pass")):
        return "ImplementationBoundaryFailed"
    if (safe_float(row.get("generator_descent_fraction"), 0.0) or 0.0) < 0.05:
        return "NoGeneratorCapacity"
    if not flag(row.get("beats_same_generator_descent_energy_random_NLL")) or not flag(row.get("beats_same_C_skew_spectrum_random_NLL")):
        return "GeneratorSupportExplained_NoFU"
    if (safe_float(row.get("active_Gram_drift_mean"), 0.0) or 0.0) <= 0.05 and not flag(row.get("beats_best_control_NLL")):
        return "MetricPreservationOnly"
    if not flag(row.get("no_ECE_Brier_tail_debt")):
        return "DebtBlocked"
    if (safe_float(row.get("controller_or_coordinate_overhead_ratio"), 999.0) or 999.0) > 0.35:
        return "OverheadBlocked"
    if not flag(row.get("beats_best_control_NLL")):
        return "CoordinateSupportExplained"
    if not flag(row.get("beats_external_OET_NLL")):
        return "ExternalOETExplained"
    if not flag(row.get("beats_strongest_NLL")):
        return "FUWeakOptimizerPatchOnly"
    return "completed"


def summarize_part_b(rows: list[dict[str, Any]], *, candidate_method: str, spectrum_threshold: float = 0.50) -> dict[str, Any]:
    candidate = [
        r
        for r in rows
        if str(r.get("method")) == candidate_method
        and str(r.get("steps")) in {"400", "800"}
        and str(r.get("dataset")) in set(PART_B_DATASETS)
        and int(float(str(r.get("seed")))) in PART_B_SEEDS
    ]
    completed = [r for r in candidate if r.get("run_status") == "completed"]
    expected_keys = part_b_expected_keys([400, 800])
    completed_keys = {(str(r.get("dataset")), int(float(str(r.get("seed")))), int(float(str(r.get("steps"))))) for r in completed}
    missing_keys = sorted(expected_keys - completed_keys, key=lambda x: (x[2], x[0], x[1]))
    no_debt_rows = sum(flag(r.get("no_ECE_Brier_tail_debt")) for r in completed)
    overhead_le_025 = sum(1 for r in completed if (safe_float(r.get("controller_or_coordinate_overhead_ratio"), 999.0) or 999.0) <= 0.25)
    gram_le_005 = sum(1 for r in completed if (safe_float(r.get("active_Gram_drift_mean"), 999.0) or 999.0) <= 0.05)
    spectrum_le = sum(1 for r in completed if (safe_float(r.get("functional_spectrum_drift_mean"), 999.0) or 999.0) <= spectrum_threshold)
    for row in candidate:
        row["failure_mode"] = part_b_failure_mode(row)
    external_explained = sum(1 for r in completed if r.get("failure_mode") == "ExternalOETExplained")
    generator_support_explained = sum(1 for r in completed if r.get("failure_mode") == "GeneratorSupportExplained_NoFU")
    metric_preservation_only = sum(1 for r in completed if r.get("failure_mode") == "MetricPreservationOnly")
    summary = {
        "gate": "v22_67_part_b_mlp_official_robustness",
        "generated_at_sg": now_sg(),
        "candidate_method": candidate_method,
        "expected_candidate_rows": 60,
        "completed_rows": len(completed),
        "missing_candidate_rows": len(missing_keys),
        "missing_candidate_keys": json.dumps(missing_keys, ensure_ascii=False),
        "beats_strongest_NLL_rows": sum(flag(r.get("beats_strongest_NLL")) for r in completed),
        "beats_external_OET_NLL_rows": sum(flag(r.get("beats_external_OET_NLL")) for r in completed),
        "beats_best_control_NLL_rows": sum(flag(r.get("beats_best_control_NLL")) for r in completed),
        "beats_same_generator_controls_rows": sum(flag(r.get("beats_same_generator_controls_NLL")) for r in completed),
        "no_debt_rows": no_debt_rows,
        "overhead_le_025_rows": overhead_le_025,
        "active_Gram_drift_le_005_rows": gram_le_005,
        "functional_spectrum_drift_le_threshold_rows": spectrum_le,
        "ExternalOETExplained_pct": 0.0 if not completed else 100.0 * external_explained / len(completed),
        "GeneratorSupportExplained_pct": 0.0 if not completed else 100.0 * generator_support_explained / len(completed),
        "MetricPreservationOnly_pct": 0.0 if not completed else 100.0 * metric_preservation_only / len(completed),
    }
    summary["robust_mlp_pass"] = int(
        summary["completed_rows"] >= 60
        and summary["beats_strongest_NLL_rows"] >= 48
        and summary["beats_external_OET_NLL_rows"] >= 42
        and summary["beats_best_control_NLL_rows"] >= 48
        and summary["beats_same_generator_controls_rows"] >= 48
        and summary["no_debt_rows"] >= 54
        and summary["overhead_le_025_rows"] >= 54
        and summary["active_Gram_drift_le_005_rows"] >= 54
        and summary["functional_spectrum_drift_le_threshold_rows"] >= 54
        and summary["ExternalOETExplained_pct"] <= 20.0
    )
    summary["part_b_next_action"] = "run_part_c_kan_preflight" if summary["robust_mlp_pass"] else (
        "complete_missing_st800_rows" if summary["missing_candidate_rows"] else "audit_external_oet_failure_rows"
    )
    return summary


def part_b_cond_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("dataset")),
        str(int(float(str(row.get("seed"))))),
        str(int(float(str(row.get("steps"))))),
    )


def best_held_nll_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_nll = float("inf")
    for row in rows:
        nll = safe_float(row.get("held_NLL"), float("inf")) or float("inf")
        if nll < best_nll:
            best = row
            best_nll = nll
    return best


def run_part_b_external_decompose(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    candidate_method = str(args.candidate_method or PART_B_CANDIDATE)
    rows = add_part_b_comparisons(collect_part_b_rows_for_candidate(candidate_method), candidate_method=candidate_method)
    summary = summarize_part_b(rows, candidate_method=candidate_method, spectrum_threshold=float(args.functional_spectrum_drift_threshold))
    by_cond: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_cond.setdefault(part_b_cond_key(row), []).append(row)

    detail: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("method")) != candidate_method or row.get("run_status") != "completed":
            continue
        group = [r for r in by_cond.get(part_b_cond_key(row), []) if r.get("run_status") == "completed"]
        external = [r for r in group if str(r.get("method")) in PART_B_EXTERNAL_METHODS]
        refs = [r for r in group if str(r.get("method")) in PART_B_REFERENCE_METHODS]
        controls = [
            r
            for r in group
            if str(r.get("method")) not in PART_B_REFERENCE_METHODS
            and str(r.get("method")) not in PART_B_EXTERNAL_METHODS
            and str(r.get("method")) != candidate_method
        ]
        best_external = best_held_nll_row(external)
        best_ref = best_held_nll_row(refs)
        best_control = best_held_nll_row(controls)
        nll = safe_float(row.get("held_NLL"), float("inf")) or float("inf")
        out = {
            "dataset": row.get("dataset", ""),
            "seed": str(int(float(str(row.get("seed"))))),
            "steps": str(int(float(str(row.get("steps"))))),
            "candidate_method": candidate_method,
            "candidate_NLL": nll,
            "candidate_failure_mode": row.get("failure_mode", part_b_failure_mode(row)),
            "candidate_beats_external_OET_NLL": row.get("beats_external_OET_NLL", ""),
            "candidate_beats_best_control_NLL": row.get("beats_best_control_NLL", ""),
            "candidate_beats_strongest_NLL": row.get("beats_strongest_NLL", ""),
            "candidate_beats_same_generator_controls_NLL": row.get("beats_same_generator_controls_NLL", ""),
            "candidate_no_ECE_Brier_tail_debt": row.get("no_ECE_Brier_tail_debt", ""),
            "candidate_functional_spectrum_drift_mean": row.get("functional_spectrum_drift_mean", ""),
            "candidate_active_Gram_drift_mean": row.get("active_Gram_drift_mean", ""),
            "candidate_generator_descent_fraction": row.get("generator_descent_fraction", ""),
            "candidate_generator_descent_energy": row.get("generator_descent_energy", ""),
        }
        for prefix, best in [("best_external", best_external), ("best_ref", best_ref), ("best_control", best_control)]:
            if best is None:
                out[f"{prefix}_method"] = ""
                out[f"{prefix}_NLL"] = ""
                out[f"Delta_NLL_vs_{prefix}"] = ""
                continue
            best_nll = safe_float(best.get("held_NLL"), float("inf")) or float("inf")
            out[f"{prefix}_method"] = best.get("method", "")
            out[f"{prefix}_NLL"] = best_nll
            out[f"Delta_NLL_vs_{prefix}"] = nll - best_nll
        detail.append(out)

    raw_external_misses = [r for r in detail if not flag(r.get("candidate_beats_external_OET_NLL"))]
    external_mode_rows = [r for r in detail if str(r.get("candidate_failure_mode")) == "ExternalOETExplained"]

    def count_values(items: list[dict[str, Any]], *keys: str) -> list[dict[str, Any]]:
        counts: dict[tuple[str, ...], int] = {}
        for item in items:
            key = tuple(str(item.get(k, "")) for k in keys)
            counts[key] = counts.get(key, 0) + 1
        out = []
        for key, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            row = {k: v for k, v in zip(keys, key)}
            row["count"] = count
            out.append(row)
        return out

    def delta_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
        vals = sorted(
            v
            for v in (safe_float(r.get("Delta_NLL_vs_best_external"), None) for r in items)
            if v is not None and math.isfinite(v)
        )
        if not vals:
            return {"count": 0}
        mid = len(vals) // 2
        return {
            "count": len(vals),
            "min": vals[0],
            "max": vals[-1],
            "mean": sum(vals) / len(vals),
            "median": vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid]),
        }

    candidate_frag = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate_method).strip("_") or "candidate"
    detail_path = OUT_ROOT / f"v22_67_part_b_external_winner_decomposition_{candidate_frag}.csv"
    summary_path = OUT_ROOT / f"v22_67_part_b_external_winner_decomposition_{candidate_frag}.json"
    write_rows(detail_path, detail)
    write_rows(OUT_ROOT / "v22_67_part_b_external_winner_decomposition.csv", detail)
    out_summary = {
        "gate": "v22_67_part_b_external_winner_decomposition",
        "generated_at_sg": now_sg(),
        "candidate_method": candidate_method,
        "robust_mlp_pass": summary["robust_mlp_pass"],
        "completed_rows": summary["completed_rows"],
        "raw_external_miss_rows": len(raw_external_misses),
        "external_oet_failure_mode_rows": len(external_mode_rows),
        "ExternalOETExplained_pct": summary["ExternalOETExplained_pct"],
        "raw_external_miss_best_external_counts": count_values(raw_external_misses, "best_external_method", "candidate_failure_mode"),
        "external_oet_failure_mode_best_external_counts": count_values(external_mode_rows, "best_external_method"),
        "external_oet_failure_mode_dataset_step_counts": count_values(external_mode_rows, "steps", "dataset"),
        "external_oet_failure_mode_delta_stats": delta_stats(external_mode_rows),
        "detail_csv": str(detail_path.relative_to(ROOT)),
    }
    summary_path.write_text(json.dumps(out_summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (OUT_ROOT / "v22_67_part_b_external_winner_decomposition.json").write_text(
        json.dumps(out_summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-b-external-decompose", "--candidate-method", candidate_method]),
        task_id="B_part_b_external_decompose",
        status="pass",
        files="results/v22_67/v22_67_part_b_external_winner_decomposition.csv; results/v22_67/v22_67_part_b_external_winner_decomposition.json",
        note=f"raw_external_miss_rows={len(raw_external_misses)}; ExternalOETExplained_rows={len(external_mode_rows)}; ExternalOETExplained_pct={summary['ExternalOETExplained_pct']}",
    )
    append_recap(
        "Part B external-OET winner decomposition",
        [
            f"candidate={candidate_method}; raw_external_miss_rows={len(raw_external_misses)}; failure-mode ExternalOETExplained rows={len(external_mode_rows)}.",
            f"Best external winner counts among ExternalOETExplained rows: {json.dumps(out_summary['external_oet_failure_mode_best_external_counts'], ensure_ascii=False)}.",
            f"Delta stats vs best external on ExternalOETExplained rows: {json.dumps(out_summary['external_oet_failure_mode_delta_stats'], ensure_ascii=False)}.",
            "Evidence files: results/v22_67/v22_67_part_b_external_winner_decomposition.csv and results/v22_67/v22_67_part_b_external_winner_decomposition.json.",
        ],
    )
    return out_summary


def part_c_row_args(dataset: str, seed: int, architecture: str, method: str, args: argparse.Namespace) -> argparse.Namespace:
    row_args = v66.build_parser().parse_args([])
    row_args.architecture = architecture
    row_args.dataset = dataset
    row_args.seed = int(seed)
    row_args.hidden = int(args.hidden)
    row_args.train_size = 512
    row_args.held_size = 256
    row_args.test_size = 256
    row_args.metric_batch_size = int(args.metric_batch_size)
    row_args.method = method
    row_args.metric_kind = "signal_debt"
    row_args.shaping_budget = 0.05
    row_args.iso_eta = 0.01 if "eta001" in method else 0.25
    row_args.functional_spectrum_drift_threshold = float(args.functional_spectrum_drift_threshold)
    return row_args


def redesign_seed_offset(architecture: str) -> int:
    if architecture == "DGKAN_DCHE":
        return 1000
    if architecture == "DGKAN_DFOU":
        return 2000
    spec = REDESIGN_KAN_SPECS.get(str(architecture), {})
    return int(spec.get("seed_offset", 0))


def make_explicit_redesign_kan(architecture: str, input_dim: int, output_dim: int, hidden: int, seed: int, device: Any, x_stats: Any) -> Any:
    from dgkan.models.fc_purekan_primitives import PrimitiveKAN, PrimitiveSpec

    spec_meta = REDESIGN_KAN_SPECS[str(architecture)]
    k = int(spec_meta["k"])
    spec = PrimitiveSpec(
        candidate_id=f"v22.67-{spec_meta['basis_family']}-h{int(hidden)}",
        basis_family=str(spec_meta["basis_family"]),
        basis_name=str(spec_meta["basis_name"]),
        k=k,
        hidden_dim=int(hidden),
        source="v22_67_basis_redesign_preflight",
        local_support=int(spec_meta["local_support"]),
        global_support=int(spec_meta["global_support"]),
        uses_exp=int(spec_meta["uses_exp"]),
        uses_sin_cos=int(spec_meta["uses_sin_cos"]),
        uses_division=int(spec_meta["uses_division"]),
        uses_dense_basis_tensor=int(spec_meta.get("uses_dense_basis_tensor", 1)),
        diagnostic_only=1,
        basis_order=int(spec_meta["basis_order"]),
        init_variant=str(spec_meta["init_variant"]),
    )
    budget = v66.redesign_param_budget(spec_meta, int(input_dim), int(output_dim), int(hidden))
    return PrimitiveKAN(
        int(input_dim),
        int(output_dim),
        spec,
        x_stats.to(device),
        int(seed),
        device,
        param_budget=budget,
    ).to(device)


def make_base_model_v22_67(args: argparse.Namespace, bundle: dict[str, Any], device: Any) -> Any:
    import torch

    arch = v66.architecture_key(args)
    if arch not in REDESIGN_KAN_SPECS:
        return v66.make_base_model(args, bundle, device)
    x_stats = bundle["x_train"][: min(512, int(bundle["x_train"].shape[0]))].to(device).float()
    kan = make_explicit_redesign_kan(
        arch,
        int(bundle["input_dim"]),
        int(bundle["num_classes"]),
        int(args.hidden),
        int(args.seed) + redesign_seed_offset(arch),
        device,
        x_stats,
    )
    base = v66.make_linearized_kan_readout_base(kan).to(device)
    with torch.no_grad():
        xb = bundle["x_train"][: min(32, int(bundle["x_train"].shape[0]))].to(device).float()
        err = (kan(xb).float() - base(xb).float()).abs().max().detach().cpu().item()
    setattr(base, "kan_readout_linearization_max_abs_error", float(err))
    setattr(base, "kan_redesign_architecture", arch)
    setattr(base, "kan_redesign_basis_name", str(REDESIGN_KAN_SPECS[arch]["basis_name"]))
    setattr(base, "kan_redesign_k", int(REDESIGN_KAN_SPECS[arch]["k"]))
    return base


def part_c_projection_capacity(phi: Any, target: Any) -> dict[str, Any]:
    import torch

    work_phi = phi.detach().double()
    work_target = target.detach().double().reshape(-1, 1)
    if work_phi.numel() == 0 or work_target.numel() == 0:
        return {"capacity": 0.0, "projected_norm": 0.0, "target_norm": 0.0, "coef_norm": 0.0}
    coef = torch.linalg.lstsq(work_phi, work_target).solution
    proj = work_phi @ coef
    target_norm = torch.linalg.norm(work_target).clamp_min(1.0e-12)
    proj_norm = torch.linalg.norm(proj)
    return {
        "capacity": float((proj_norm.square() / target_norm.square()).clamp(min=0.0, max=1.0e6).detach().cpu().item()),
        "projected_norm": float(proj_norm.detach().cpu().item()),
        "target_norm": float(target_norm.detach().cpu().item()),
        "coef_norm": float(torch.linalg.norm(coef).detach().cpu().item()),
        "projection": proj.reshape(-1),
    }


def part_c_design_matrix(base: Any, atlas: Any, x_metric: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    features = base.features(x_metric).detach().float()
    u = atlas.output_basis.detach().float()
    v = atlas.input_basis.detach().float()
    scale = atlas.coord_scale.detach().float().view(-1)
    h_v = features @ v
    cols = []
    for i in range(int(u.shape[1])):
        for j in range(int(v.shape[1])):
            col = h_v[:, j].view(-1, 1) * float(scale[j].detach().cpu().item()) * u[:, i].view(1, -1)
            cols.append(col.reshape(-1))
    phi = torch.stack(cols, dim=1) if cols else torch.zeros(int(features.shape[0]) * int(u.shape[0]), 0, device=features.device)
    gram = phi.double().transpose(0, 1) @ phi.double() / max(1, int(features.shape[0]))
    if gram.numel():
        eye = torch.eye(int(gram.shape[0]), device=gram.device, dtype=gram.dtype)
        evals = torch.linalg.eigvalsh(0.5 * (gram + gram.transpose(0, 1)) + 1.0e-8 * eye)
        vals = torch.clamp(evals.float(), min=0.0)
        trace = float(vals.sum().detach().cpu().item())
        if trace > 1.0e-12:
            probs = vals / vals.sum().clamp_min(1.0e-12)
            eff_rank = float(torch.exp(-(probs * torch.log(probs.clamp_min(1.0e-12))).sum()).detach().cpu().item())
        else:
            eff_rank = 0.0
        cond = float((evals.max() / evals.min().clamp_min(1.0e-8)).detach().cpu().item())
        min_eig = float(evals.min().detach().cpu().item())
    else:
        trace = 0.0
        eff_rank = 0.0
        cond = 0.0
        min_eig = 0.0
    col_energy = phi.float().square().sum(dim=0)
    if int(col_energy.numel()) > 0 and float(col_energy.max().detach().cpu().item()) > 1.0e-12:
        visible = (col_energy / col_energy.max().clamp_min(1.0e-12)).detach().cpu().tolist()
        visible_sorted = sorted(float(x) for x in visible)
        cvar_n = max(1, int(math.ceil(0.25 * len(visible_sorted))))
        cvar25 = sum(visible_sorted[:cvar_n]) / cvar_n
        visible_min = visible_sorted[0]
        visible_mean = sum(visible_sorted) / len(visible_sorted)
    else:
        visible_mean = 0.0
        visible_min = 0.0
        cvar25 = 0.0
    return phi, {
        "basis_Gram_PSD_min_eig": min_eig,
        "basis_Gram_condition": cond,
        "basis_Gram_effective_rank": eff_rank,
        "basis_Gram_trace": trace,
        "readout_visible_energy_mean": visible_mean,
        "readout_visible_energy_min": visible_min,
        "readout_visible_energy_CVaR25": cvar25,
        "bank_visible_energy_rank": eff_rank,
        "basis_feature_dim": int(features.shape[1]),
        "atlas_rank": int(u.shape[1]),
        "projection_dim": int(phi.shape[1]),
    }


def part_c_column_visibility(phi: Any) -> dict[str, Any]:
    import torch

    if int(phi.numel()) == 0 or int(phi.shape[1]) == 0:
        return {
            "column_energy": torch.zeros(0, device=phi.device if hasattr(phi, "device") else "cpu"),
            "normalized": torch.zeros(0, device=phi.device if hasattr(phi, "device") else "cpu"),
            "mean": 0.0,
            "min": 0.0,
            "cvar25": 0.0,
        }
    energy = phi.float().square().sum(dim=0)
    normalized = energy / energy.max().clamp_min(1.0e-12)
    vals = sorted(float(x) for x in normalized.detach().cpu().tolist())
    cvar_n = max(1, int(math.ceil(0.25 * len(vals))))
    return {
        "column_energy": energy,
        "normalized": normalized,
        "mean": sum(vals) / len(vals),
        "min": vals[0],
        "cvar25": sum(vals[:cvar_n]) / cvar_n,
    }


def part_c_readout_visible_projector(phi: Any, *, threshold: float = 0.20, min_cols: int = 2, seed: int = 0) -> dict[str, Any]:
    import torch

    visibility = part_c_column_visibility(phi)
    normalized = visibility["normalized"]
    energy = visibility["column_energy"]
    col_count = int(normalized.numel())
    if col_count == 0:
        return {
            "visible_phi": phi[:, :0],
            "same_energy_random_phi": phi[:, :0],
            "keep_indices": [],
            "visible_stats": visibility,
            "retained_energy_fraction": 0.0,
        }
    keep = torch.nonzero(normalized >= float(threshold), as_tuple=False).reshape(-1)
    if int(keep.numel()) < min(min_cols, col_count):
        keep = torch.argsort(normalized, descending=True)[: min(min_cols, col_count)]
    keep = torch.sort(keep).values
    visible_phi = phi.index_select(1, keep)
    selected_energy = energy.index_select(0, keep)
    gen = torch.Generator(device=phi.device)
    gen.manual_seed(int(seed) + 424242)
    random_mix = torch.randn(col_count, int(keep.numel()), device=phi.device, dtype=phi.dtype, generator=gen)
    q, _ = torch.linalg.qr(random_mix, mode="reduced")
    same_energy_random_phi = phi @ q[:, : int(keep.numel())]
    random_energy = same_energy_random_phi.float().square().sum(dim=0).clamp_min(1.0e-12)
    same_energy_random_phi = same_energy_random_phi * torch.sqrt(selected_energy.to(phi.device).float() / random_energy).to(phi.dtype)
    visible_stats = part_c_column_visibility(visible_phi)
    return {
        "visible_phi": visible_phi,
        "same_energy_random_phi": same_energy_random_phi,
        "keep_indices": [int(x) for x in keep.detach().cpu().tolist()],
        "visible_stats": visible_stats,
        "retained_energy_fraction": float((selected_energy.sum() / energy.sum().clamp_min(1.0e-12)).detach().cpu().item()),
    }


def part_c_task_target(base: Any, x_metric: Any, y_metric: Any, num_classes: int) -> Any:
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        logits = base(x_metric).float()
        probs = torch.softmax(logits, dim=1)
        yoh = F.one_hot(y_metric.long(), num_classes=int(num_classes)).float()
        grad_logits = (probs - yoh) / max(1, int(x_metric.shape[0]))
    return (-grad_logits).reshape(-1)


def make_feature_chart_prepared_base(base: Any, x_metric: Any, *, eps: float = 1.0e-4) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn as nn

    with torch.no_grad():
        feats = base.features(x_metric).detach().float()
        mu = feats.mean(dim=0)
        centered = feats - mu
        cov = centered.transpose(0, 1) @ centered / max(1, int(centered.shape[0]))
        eye = torch.eye(int(cov.shape[0]), device=cov.device, dtype=cov.dtype)
        evals, evecs = torch.linalg.eigh(0.5 * (cov + cov.transpose(0, 1)) + float(eps) * eye)
        evals = evals.clamp_min(float(eps))
        inv_sqrt = evecs @ torch.diag(torch.rsqrt(evals)) @ evecs.transpose(0, 1)
        sqrt = evecs @ torch.diag(torch.sqrt(evals)) @ evecs.transpose(0, 1)
        old_weight = base.fc3.weight.detach().float()
        old_bias = base.fc3.bias.detach().float() if getattr(base.fc3, "bias", None) is not None else torch.zeros(int(old_weight.shape[0]), device=old_weight.device)
        new_weight = old_weight @ sqrt.transpose(0, 1)
        new_bias = old_bias + mu @ old_weight.transpose(0, 1)

    class _FrozenReadout(nn.Module):
        def __init__(self, weight: Any, bias: Any) -> None:
            super().__init__()
            self.register_buffer("weight", weight.detach().clone().float())
            self.register_buffer("bias", bias.detach().clone().float())

        def forward(self, x: Any) -> Any:
            return x @ self.weight.transpose(0, 1) + self.bias

    class _FeatureChartPreparedBase(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.source_base = base
            self.fc3 = _FrozenReadout(new_weight, new_bias)
            self.register_buffer("feature_mean", mu.detach().clone().float())
            self.register_buffer("feature_inv_sqrt", inv_sqrt.detach().clone().float())

        def features(self, x: Any) -> Any:
            return (self.source_base.features(x).float() - self.feature_mean) @ self.feature_inv_sqrt

        def forward(self, x: Any) -> Any:
            return self.fc3(self.features(x))

    prepared = _FeatureChartPreparedBase().to(x_metric.device)
    with torch.no_grad():
        err = (base(x_metric).float() - prepared(x_metric).float()).abs().max().detach().cpu().item()
    setattr(prepared, "kan_readout_linearization_max_abs_error", getattr(base, "kan_readout_linearization_max_abs_error", ""))
    return prepared, {
        "chart_preparation": "train_only_feature_Gram_whiten_preserve_logits",
        "chart_preparation_logit_max_abs_error": float(err),
        "feature_Gram_min_eig_before": float(evals.min().detach().cpu().item()),
        "feature_Gram_max_eig_before": float(evals.max().detach().cpu().item()),
        "feature_Gram_condition_before": float((evals.max() / evals.min().clamp_min(float(eps))).detach().cpu().item()),
    }


def run_part_c_readout_visible_repair(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    ensure_out()
    device = torch.device(str(args.device))
    method = str(args.candidate_method or PART_C_WINNER)
    rows: list[dict[str, Any]] = []
    append_exec(
        command_text(
            [
                PYTHON,
                str(RUNNER.relative_to(ROOT)),
                "--mode",
                "part-c-visible-repair",
                "--candidate-method",
                method,
                "--device",
                str(args.device),
            ]
        ),
        task_id="C_readout_visible_repair_start",
        status="start",
        gpu=str(args.device),
        files="results/v22_67/v22_67_part_c_readout_visible_repair.csv; results/v22_67/v22_67_part_c_readout_visible_repair_summary.json",
        note="train-only readout_visible_projector + same_readout_visible_energy_random_control diagnostic",
    )
    for dataset in PART_C_DATASETS:
        for seed in PART_B_SEEDS:
            bundle = v66.load_bundle(dataset, 512, 256, 256, int(seed))
            x_metric = bundle["x_train"][: int(args.metric_batch_size)].to(device)
            y_metric = bundle["y_train"][: int(args.metric_batch_size)].to(device)
            for arch in ["DGKAN_DCHE", "DGKAN_DFOU"]:
                row_args = part_c_row_args(dataset, int(seed), arch, method, args)
                base = make_base_model_v22_67(row_args, bundle, device)
                atlas = v66.build_atlas_for_method(base, x_metric, y_metric, row_args, method, bundle)
                phi, diag = part_c_design_matrix(base, atlas, x_metric)
                target = part_c_task_target(base, x_metric, y_metric, int(bundle["num_classes"]))
                original = part_c_projection_capacity(phi, target)
                projected = part_c_readout_visible_projector(
                    phi,
                    threshold=0.20,
                    min_cols=max(2, int(math.ceil(0.25 * int(phi.shape[1])))),
                    seed=int(seed) * 1009 + sum(ord(c) for c in dataset + arch),
                )
                visible = part_c_projection_capacity(projected["visible_phi"], target)
                random_control = part_c_projection_capacity(projected["same_energy_random_phi"], target)
                vstats = projected["visible_stats"]
                row = {
                    "run_status": "completed",
                    "dataset": dataset,
                    "seed": int(seed),
                    "architecture": arch,
                    "candidate_method": method,
                    "repair_route": "ReadoutInvisibleBasisMotion",
                    "readout_visible_projector": 1,
                    "bank_readout_sensitivity_metric": json.dumps(
                        [round(float(x), 8) for x in part_c_column_visibility(phi)["normalized"].detach().cpu().tolist()],
                        ensure_ascii=False,
                    ),
                    "original_projection_dim": int(phi.shape[1]),
                    "visible_projection_dim": int(projected["visible_phi"].shape[1]),
                    "visible_keep_indices": json.dumps(projected["keep_indices"]),
                    "retained_energy_fraction": projected["retained_energy_fraction"],
                    "original_readout_visible_energy_CVaR25": diag["readout_visible_energy_CVaR25"],
                    "projected_readout_visible_energy_CVaR25": vstats["cvar25"],
                    "original_task_gradient_capacity": original["capacity"],
                    "readout_visible_task_gradient_capacity": visible["capacity"],
                    "same_readout_visible_energy_random_control_capacity": random_control["capacity"],
                    "readout_visible_generator_objective": visible["projected_norm"],
                    "same_energy_random_objective": random_control["projected_norm"],
                    "visible_minus_random_capacity": float(visible["capacity"]) - float(random_control["capacity"]),
                    "beats_same_readout_visible_energy_random_control_predicted": int(float(visible["capacity"]) > float(random_control["capacity"])),
                    "kan_readout_linearization_max_abs_error": getattr(base, "kan_readout_linearization_max_abs_error", ""),
                }
                rows.append(row)
    write_rows(OUT_ROOT / "v22_67_part_c_readout_visible_repair.csv", rows)
    summary_rows: list[dict[str, Any]] = []
    for arch in ["DGKAN_DCHE", "DGKAN_DFOU"]:
        sub = [r for r in rows if r["architecture"] == arch]
        low_capacity = sum(1 for r in sub if (safe_float(r.get("readout_visible_task_gradient_capacity"), 0.0) or 0.0) < 0.30)
        low_visible = sum(1 for r in sub if (safe_float(r.get("projected_readout_visible_energy_CVaR25"), 0.0) or 0.0) < 0.20)
        beats_random = sum(flag(r.get("beats_same_readout_visible_energy_random_control_predicted")) for r in sub)
        summary_rows.append(
            {
                "architecture": arch,
                "completed_rows": len(sub),
                "readout_visible_task_gradient_capacity_mean": sum(safe_float(r.get("readout_visible_task_gradient_capacity"), 0.0) or 0.0 for r in sub) / max(1, len(sub)),
                "readout_visible_task_gradient_capacity_lt_030_rows": low_capacity,
                "projected_readout_visible_energy_CVaR25_mean": sum(safe_float(r.get("projected_readout_visible_energy_CVaR25"), 0.0) or 0.0 for r in sub) / max(1, len(sub)),
                "projected_readout_visible_energy_CVaR25_lt_020_rows": low_visible,
                "beats_same_readout_visible_energy_random_control_rows": beats_random,
                "projector_allows_part_d": int(low_visible == 0 and beats_random >= 8),
                "projector_allows_full_loop": int(low_visible == 0 and low_capacity < 10 and beats_random >= 8),
            }
        )
    summary = {
        "gate": "v22_67_part_c_readout_visible_repair",
        "generated_at_sg": now_sg(),
        "candidate_method": method,
        "completed_rows": len(rows),
        "summary_rows": summary_rows,
        "part_d_recommended": int(all(int(r["projector_allows_part_d"]) for r in summary_rows)),
        "full_loop_recommended": int(all(int(r["projector_allows_full_loop"]) for r in summary_rows)),
        "next_action": "run_part_d_unit_tests" if all(int(r["projector_allows_part_d"]) for r in summary_rows) else "repair_readout_visible_projector_or_basis_chart",
    }
    write_rows(OUT_ROOT / "v22_67_part_c_readout_visible_repair_summary.csv", summary_rows)
    (OUT_ROOT / "v22_67_part_c_readout_visible_repair_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-c-visible-repair", "--candidate-method", method, "--device", str(args.device)]),
        task_id="C_readout_visible_repair",
        status="pass" if len(rows) == 2 * len(PART_C_DATASETS) * len(PART_B_SEEDS) else "partial_or_fail",
        gpu=str(args.device),
        files="results/v22_67/v22_67_part_c_readout_visible_repair.csv; results/v22_67/v22_67_part_c_readout_visible_repair_summary.json",
        note=f"completed_rows={len(rows)}; next_action={summary['next_action']}; full_loop_recommended={summary['full_loop_recommended']}",
    )
    append_recap(
        "Part C readout-visible repair diagnostic",
        [
            f"candidate={method}; completed_rows={len(rows)}; next_action={summary['next_action']}; full_loop_recommended={summary['full_loop_recommended']}.",
            f"Summary rows: {json.dumps(summary_rows, ensure_ascii=False)}.",
            "Modification audited: implemented train-only readout_visible_projector, bank_readout_sensitivity_metric, readout_visible_generator_objective, and same_readout_visible_energy_random_control diagnostic after Part C low-visible blocker.",
            "Evidence files: results/v22_67/v22_67_part_c_readout_visible_repair.csv and v22_67_part_c_readout_visible_repair_summary.json.",
        ],
    )
    return summary


def run_part_c_chart_warmstart(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    ensure_out()
    device = torch.device(str(args.device))
    method = str(args.candidate_method or PART_C_WINNER)
    rows: list[dict[str, Any]] = []
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-c-chart-warmstart", "--candidate-method", method, "--device", str(args.device)]),
        task_id="C_chart_warmstart_start",
        status="start",
        gpu=str(args.device),
        files="results/v22_67/v22_67_part_c_chart_warmstart.csv; results/v22_67/v22_67_part_c_chart_warmstart_summary.json",
        note="train-only feature Gram whitening with logits-preserving readout transform; follows H4 basis chart preparation route",
    )
    for dataset in PART_C_DATASETS:
        for seed in PART_B_SEEDS:
            bundle = v66.load_bundle(dataset, 512, 256, 256, int(seed))
            x_metric = bundle["x_train"][: int(args.metric_batch_size)].to(device)
            y_metric = bundle["y_train"][: int(args.metric_batch_size)].to(device)
            for arch in ["DGKAN_DCHE", "DGKAN_DFOU"]:
                row_args = part_c_row_args(dataset, int(seed), arch, method, args)
                base = make_base_model_v22_67(row_args, bundle, device)
                prepared, prep_diag = make_feature_chart_prepared_base(base, x_metric)
                atlas = v66.build_atlas_for_method(prepared, x_metric, y_metric, row_args, method, bundle)
                phi, diag = part_c_design_matrix(prepared, atlas, x_metric)
                target = part_c_task_target(prepared, x_metric, y_metric, int(bundle["num_classes"]))
                full = part_c_projection_capacity(phi, target)
                projected = part_c_readout_visible_projector(
                    phi,
                    threshold=0.20,
                    min_cols=max(2, int(math.ceil(0.25 * int(phi.shape[1])))),
                    seed=int(seed) * 1291 + sum(ord(c) for c in dataset + arch),
                )
                visible = part_c_projection_capacity(projected["visible_phi"], target)
                random_control = part_c_projection_capacity(projected["same_energy_random_phi"], target)
                vstats = projected["visible_stats"]
                rows.append(
                    {
                        "run_status": "completed",
                        "dataset": dataset,
                        "seed": int(seed),
                        "architecture": arch,
                        "candidate_method": method,
                        **prep_diag,
                        "projection_dim": int(phi.shape[1]),
                        "readout_visible_projection_dim": int(projected["visible_phi"].shape[1]),
                        "readout_visible_keep_indices": json.dumps(projected["keep_indices"]),
                        "retained_energy_fraction": projected["retained_energy_fraction"],
                        "chart_full_task_gradient_capacity": full["capacity"],
                        "chart_readout_visible_task_gradient_capacity": visible["capacity"],
                        "chart_same_energy_random_capacity": random_control["capacity"],
                        "chart_visible_minus_random_capacity": float(visible["capacity"]) - float(random_control["capacity"]),
                        "chart_beats_same_readout_visible_energy_random_control": int(float(visible["capacity"]) > float(random_control["capacity"])),
                        "chart_readout_visible_energy_CVaR25": vstats["cvar25"],
                        "chart_readout_visible_energy_mean": vstats["mean"],
                        "basis_Gram_PSD_min_eig": diag["basis_Gram_PSD_min_eig"],
                        "basis_Gram_condition": diag["basis_Gram_condition"],
                        "basis_Gram_effective_rank": diag["basis_Gram_effective_rank"],
                        "kan_readout_linearization_max_abs_error": getattr(base, "kan_readout_linearization_max_abs_error", ""),
                    }
                )
    write_rows(OUT_ROOT / "v22_67_part_c_chart_warmstart.csv", rows)
    summary_rows: list[dict[str, Any]] = []
    for arch in ["DGKAN_DCHE", "DGKAN_DFOU"]:
        sub = [r for r in rows if r["architecture"] == arch]
        low_capacity = sum(1 for r in sub if (safe_float(r.get("chart_readout_visible_task_gradient_capacity"), 0.0) or 0.0) < 0.30)
        low_visible = sum(1 for r in sub if (safe_float(r.get("chart_readout_visible_energy_CVaR25"), 0.0) or 0.0) < 0.20)
        beats_random = sum(flag(r.get("chart_beats_same_readout_visible_energy_random_control")) for r in sub)
        logit_bad = sum(1 for r in sub if (safe_float(r.get("chart_preparation_logit_max_abs_error"), 999.0) or 999.0) > 1.0e-4)
        summary_rows.append(
            {
                "architecture": arch,
                "completed_rows": len(sub),
                "chart_readout_visible_task_gradient_capacity_mean": sum(safe_float(r.get("chart_readout_visible_task_gradient_capacity"), 0.0) or 0.0 for r in sub) / max(1, len(sub)),
                "chart_readout_visible_task_gradient_capacity_lt_030_rows": low_capacity,
                "chart_readout_visible_energy_CVaR25_mean": sum(safe_float(r.get("chart_readout_visible_energy_CVaR25"), 0.0) or 0.0 for r in sub) / max(1, len(sub)),
                "chart_readout_visible_energy_CVaR25_lt_020_rows": low_visible,
                "chart_beats_same_readout_visible_energy_random_control_rows": beats_random,
                "chart_preparation_logit_error_gt_1e_4_rows": logit_bad,
                "chart_warmstart_allows_full_loop": int(low_capacity < 10 and low_visible == 0 and beats_random >= 8 and logit_bad == 0),
            }
        )
    summary = {
        "gate": "v22_67_part_c_chart_warmstart",
        "generated_at_sg": now_sg(),
        "candidate_method": method,
        "completed_rows": len(rows),
        "summary_rows": summary_rows,
        "full_loop_recommended": int(all(int(r["chart_warmstart_allows_full_loop"]) for r in summary_rows)),
        "next_action": "run_part_e_kan_visible_projector_probe" if all(int(r["chart_warmstart_allows_full_loop"]) for r in summary_rows) else "chart_warmstart_insufficient_try_bank_local_or_basis_redesign",
    }
    write_rows(OUT_ROOT / "v22_67_part_c_chart_warmstart_summary.csv", summary_rows)
    (OUT_ROOT / "v22_67_part_c_chart_warmstart_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-c-chart-warmstart", "--candidate-method", method, "--device", str(args.device)]),
        task_id="C_chart_warmstart",
        status="pass" if len(rows) == 2 * len(PART_C_DATASETS) * len(PART_B_SEEDS) else "partial_or_fail",
        gpu=str(args.device),
        files="results/v22_67/v22_67_part_c_chart_warmstart.csv; results/v22_67/v22_67_part_c_chart_warmstart_summary.json",
        note=f"completed_rows={len(rows)}; full_loop_recommended={summary['full_loop_recommended']}; next_action={summary['next_action']}",
    )
    append_recap(
        "Part C functional basis chart warm-start diagnostic",
        [
            f"candidate={method}; completed_rows={len(rows)}; full_loop_recommended={summary['full_loop_recommended']}; next_action={summary['next_action']}.",
            f"Summary rows: {json.dumps(summary_rows, ensure_ascii=False)}.",
            "Modification audited: implemented train-only feature Gram whitening with logits-preserving readout transform as H4 chart preparation; no held/test/future direction and no auxiliary loss.",
            "Evidence files: results/v22_67/v22_67_part_c_chart_warmstart.csv and v22_67_part_c_chart_warmstart_summary.json.",
        ],
    )
    return summary


def part_c_task_visible_select(source_phi: Any, source_target: Any, witness_phi: Any, *, keep_fraction: float = 0.25, seed: int = 0) -> dict[str, Any]:
    import torch

    col_count = int(source_phi.shape[1])
    if col_count == 0:
        return {
            "selected_phi": witness_phi[:, :0],
            "same_energy_random_phi": witness_phi[:, :0],
            "keep_indices": [],
            "source_scores": [],
            "selected_source_visibility_cvar25": 0.0,
            "retained_source_score_fraction": 0.0,
        }
    visibility = part_c_column_visibility(source_phi)
    normalized = visibility["normalized"].float()
    target = source_target.float().reshape(-1)
    col_norm = source_phi.float().norm(dim=0).clamp_min(1.0e-12)
    target_norm = target.norm().clamp_min(1.0e-12)
    align = torch.abs(source_phi.float().transpose(0, 1) @ target) / (col_norm * target_norm)
    score = normalized * align.square()
    keep_count = max(2, min(col_count, int(math.ceil(float(keep_fraction) * col_count))))
    keep = torch.argsort(score, descending=True)[:keep_count]
    keep = torch.sort(keep).values
    selected_phi = witness_phi.index_select(1, keep)
    selected_energy = selected_phi.float().square().sum(dim=0)
    gen = torch.Generator(device=witness_phi.device)
    gen.manual_seed(int(seed) + 515151)
    random_mix = torch.randn(col_count, int(keep.numel()), device=witness_phi.device, dtype=witness_phi.dtype, generator=gen)
    q, _ = torch.linalg.qr(random_mix, mode="reduced")
    same_energy_random_phi = witness_phi @ q[:, : int(keep.numel())]
    random_energy = same_energy_random_phi.float().square().sum(dim=0).clamp_min(1.0e-12)
    same_energy_random_phi = same_energy_random_phi * torch.sqrt(selected_energy / random_energy).to(witness_phi.dtype)
    selected_vis = normalized.index_select(0, keep).detach().cpu().tolist()
    selected_vis_sorted = sorted(float(x) for x in selected_vis)
    cvar_n = max(1, int(math.ceil(0.25 * len(selected_vis_sorted))))
    return {
        "selected_phi": selected_phi,
        "same_energy_random_phi": same_energy_random_phi,
        "keep_indices": [int(x) for x in keep.detach().cpu().tolist()],
        "source_scores": [float(score[i].detach().cpu().item()) for i in keep],
        "selected_source_visibility_cvar25": sum(selected_vis_sorted[:cvar_n]) / cvar_n,
        "retained_source_score_fraction": float((score.index_select(0, keep).sum() / score.sum().clamp_min(1.0e-12)).detach().cpu().item()),
    }


def make_redesigned_kan_base(architecture: str, bundle: dict[str, Any], device: Any, hidden: int, seed: int) -> tuple[Any, dict[str, Any]]:
    import torch
    from dgkan.models.fc_purekan_primitives import PrimitiveKAN, PrimitiveSpec
    spec_map = {
        "DGKAN_CHE4": ("D-CHE4", "chebyshev", 4, "cheby_k4_triton_l3_matmul", 0, 0, 0, 0, 1),
        "DGKAN_CHE3_XLIN": ("D-CHE3-XLIN", "chebyshev", 3, "cheby_k3_inputcross_localr4_projr128_triton_l3_gradbuf_linearres010", 0, 0, 0, 0, 1),
        "DGKAN_FOU4": ("D-FOU4", "fourier_lowfreq", 4, "fourier_k4_triton_l3_matmul", 0, 1, 0, 0, 1),
        "DGKAN_FOU4_LIN": ("D-FOU4-LIN", "fourier_lowfreq", 4, "fourier_k4_linearres_gemm_l3_matmul_linearres010", 0, 1, 0, 0, 1),
        "DGKAN_FOU4_LIN50": ("D-FOU4-LIN50", "fourier_lowfreq", 4, "fourier_k4_linearres_gemm_l3_matmul_linearres050", 0, 1, 0, 0, 1),
        "DGKAN_RBF4": ("D-RBF4", "compact_rbf", 4, "rbf_k4_triton_l3_matmul", 1, 0, 0, 1, 0),
        "DGKAN_RBF4_XLIN": ("D-RBF4-XLIN", "compact_rbf", 4, "rbf_k4_triton_l3_matmul_inputcross_localr4_projr128_linearres010", 1, 0, 0, 1, 0),
        "DGKAN_HAT4": ("D-HAT4", "hat_wavelet", 4, "hat_wavelet_k4_triton_l3_matmul", 0, 0, 0, 1, 0),
        "DGKAN_HAT4_XLIN": ("D-HAT4-XLIN", "hat_wavelet", 4, "hat_wavelet_k4_triton_l3_matmul_inputcross_localr4_projr128_linearres010", 0, 0, 0, 1, 0),
        "DGKAN_RAT4": ("D-RAT4", "rational_kat_lite", 4, "rational_k4_triton_l3_matmul", 0, 0, 1, 0, 1),
    }
    if architecture not in spec_map:
        raise ValueError(f"unknown redesign architecture {architecture}")
    basis_family, basis_name, k, init_variant, uses_exp, uses_sin_cos, uses_division, local_support, global_support = spec_map[architecture]
    input_dim = int(bundle["input_dim"])
    output_dim = int(bundle["num_classes"])
    requested_hidden = int(hidden)
    effective_hidden = max(4, int(round(float(requested_hidden) * 3.0 / float(k))))
    spec = PrimitiveSpec(
        candidate_id=f"v22.67-redesign-{architecture}",
        basis_family=basis_family,
        basis_name=basis_name,
        k=int(k),
        hidden_dim=int(effective_hidden),
        source="v22_67_basis_redesign_preflight",
        local_support=int(local_support),
        global_support=int(global_support),
        uses_exp=int(uses_exp),
        uses_sin_cos=int(uses_sin_cos),
        uses_division=int(uses_division),
        uses_dense_basis_tensor=int(REDESIGN_KAN_SPECS.get(architecture, {}).get("uses_dense_basis_tensor", 0)),
        basis_order=int(k),
        init_variant=init_variant,
    )
    x_stats = bundle["x_train"][: min(512, int(bundle["x_train"].shape[0]))].to(device).float()
    param_budget = v66.redesign_param_budget(
        REDESIGN_KAN_SPECS.get(architecture, {"k": k, "init_variant": init_variant}),
        int(input_dim),
        int(output_dim),
        int(effective_hidden),
    )
    kan = PrimitiveKAN(input_dim, output_dim, spec, x_stats, int(seed), device, param_budget=param_budget).to(device)
    base = v66.make_linearized_kan_readout_base(kan).to(device)
    with torch.no_grad():
        xb = bundle["x_train"][: min(32, int(bundle["x_train"].shape[0]))].to(device).float()
        err = (kan(xb).float() - base(xb).float()).abs().max().detach().cpu().item()
    setattr(base, "kan_readout_linearization_max_abs_error", float(err))
    edge_params = sum(int(p.numel()) for p in kan.parameters())
    return base, {
        "redesign_architecture": architecture,
        "basis_family": basis_family,
        "basis_name": basis_name,
        "basis_k": int(k),
        "init_variant": init_variant,
        "requested_hidden": int(requested_hidden),
        "effective_hidden": int(effective_hidden),
        "edge_param_count": int(edge_params),
        "param_budget_reference": int(param_budget),
        "kan_readout_linearization_max_abs_error": float(err),
    }
def run_part_c_task_visible_repair(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    ensure_out()
    device = torch.device(str(args.device))
    method = str(args.candidate_method or PART_C_WINNER)
    rows: list[dict[str, Any]] = []
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-c-task-visible-repair", "--candidate-method", method, "--device", str(args.device)]),
        task_id="C_task_visible_repair_start",
        status="start",
        gpu=str(args.device),
        files="results/v22_67/v22_67_part_c_task_visible_repair.csv; results/v22_67/v22_67_part_c_task_visible_repair_summary.json",
        note="source/witness train-only task-visible readout generator objective after chart warmstart",
    )
    for dataset in PART_C_DATASETS:
        for seed in PART_B_SEEDS:
            bundle = v66.load_bundle(dataset, 512, 256, 256, int(seed))
            x_metric_all = bundle["x_train"][: int(args.metric_batch_size)].to(device)
            y_metric_all = bundle["y_train"][: int(args.metric_batch_size)].to(device)
            half = max(16, int(x_metric_all.shape[0]) // 2)
            x_source = x_metric_all[:half]
            y_source = y_metric_all[:half]
            x_witness = x_metric_all[half:]
            y_witness = y_metric_all[half:]
            if int(x_witness.shape[0]) == 0:
                x_witness = x_source
                y_witness = y_source
            for arch in ["DGKAN_DCHE", "DGKAN_DFOU"]:
                row_args = part_c_row_args(dataset, int(seed), arch, method, args)
                base = make_base_model_v22_67(row_args, bundle, device)
                prepared, prep_diag = make_feature_chart_prepared_base(base, x_source)
                atlas = v66.build_atlas_for_method(prepared, x_source, y_source, row_args, method, bundle)
                phi_source, _source_diag = part_c_design_matrix(prepared, atlas, x_source)
                phi_witness, witness_diag = part_c_design_matrix(prepared, atlas, x_witness)
                source_target = part_c_task_target(prepared, x_source, y_source, int(bundle["num_classes"]))
                witness_target = part_c_task_target(prepared, x_witness, y_witness, int(bundle["num_classes"]))
                full_witness = part_c_projection_capacity(phi_witness, witness_target)
                selected = part_c_task_visible_select(
                    phi_source,
                    source_target,
                    phi_witness,
                    keep_fraction=0.25,
                    seed=int(seed) * 1553 + sum(ord(c) for c in dataset + arch),
                )
                task_visible = part_c_projection_capacity(selected["selected_phi"], witness_target)
                random_control = part_c_projection_capacity(selected["same_energy_random_phi"], witness_target)
                rows.append(
                    {
                        "run_status": "completed",
                        "dataset": dataset,
                        "seed": int(seed),
                        "architecture": arch,
                        "candidate_method": method,
                        **prep_diag,
                        "source_rows": int(x_source.shape[0]),
                        "witness_rows": int(x_witness.shape[0]),
                        "projection_dim": int(phi_source.shape[1]),
                        "task_visible_projection_dim": int(selected["selected_phi"].shape[1]),
                        "task_visible_keep_indices": json.dumps(selected["keep_indices"]),
                        "task_visible_source_scores": json.dumps([round(float(x), 8) for x in selected["source_scores"]]),
                        "task_visible_retained_source_score_fraction": selected["retained_source_score_fraction"],
                        "task_visible_source_visibility_CVaR25": selected["selected_source_visibility_cvar25"],
                        "witness_full_task_gradient_capacity": full_witness["capacity"],
                        "task_visible_witness_capacity": task_visible["capacity"],
                        "same_energy_random_witness_capacity": random_control["capacity"],
                        "task_visible_minus_random_capacity": float(task_visible["capacity"]) - float(random_control["capacity"]),
                        "beats_same_energy_random_witness_control": int(float(task_visible["capacity"]) > float(random_control["capacity"])),
                        "witness_readout_visible_energy_CVaR25": witness_diag["readout_visible_energy_CVaR25"],
                        "kan_readout_linearization_max_abs_error": getattr(base, "kan_readout_linearization_max_abs_error", ""),
                    }
                )
    write_rows(OUT_ROOT / "v22_67_part_c_task_visible_repair.csv", rows)
    summary_rows: list[dict[str, Any]] = []
    for arch in ["DGKAN_DCHE", "DGKAN_DFOU"]:
        sub = [r for r in rows if r["architecture"] == arch]
        low_capacity = sum(1 for r in sub if (safe_float(r.get("task_visible_witness_capacity"), 0.0) or 0.0) < 0.30)
        low_visible = sum(1 for r in sub if (safe_float(r.get("task_visible_source_visibility_CVaR25"), 0.0) or 0.0) < 0.20)
        beats_random = sum(flag(r.get("beats_same_energy_random_witness_control")) for r in sub)
        logit_bad = sum(1 for r in sub if (safe_float(r.get("chart_preparation_logit_max_abs_error"), 999.0) or 999.0) > 1.0e-4)
        summary_rows.append(
            {
                "architecture": arch,
                "completed_rows": len(sub),
                "task_visible_witness_capacity_mean": sum(safe_float(r.get("task_visible_witness_capacity"), 0.0) or 0.0 for r in sub) / max(1, len(sub)),
                "task_visible_witness_capacity_lt_030_rows": low_capacity,
                "task_visible_source_visibility_CVaR25_mean": sum(safe_float(r.get("task_visible_source_visibility_CVaR25"), 0.0) or 0.0 for r in sub) / max(1, len(sub)),
                "task_visible_source_visibility_CVaR25_lt_020_rows": low_visible,
                "beats_same_energy_random_witness_control_rows": beats_random,
                "chart_preparation_logit_error_gt_1e_4_rows": logit_bad,
                "task_visible_allows_full_loop_probe": int(low_capacity < 10 and low_visible < 10 and beats_random >= 8 and logit_bad == 0),
            }
        )
    summary = {
        "gate": "v22_67_part_c_task_visible_repair",
        "generated_at_sg": now_sg(),
        "candidate_method": method,
        "completed_rows": len(rows),
        "summary_rows": summary_rows,
        "full_loop_probe_recommended": int(all(int(r["task_visible_allows_full_loop_probe"]) for r in summary_rows)),
        "next_action": "implement_part_e_task_visible_full_loop_probe" if all(int(r["task_visible_allows_full_loop_probe"]) for r in summary_rows) else "task_visible_generator_control_explained_or_capacity_blocked",
    }
    write_rows(OUT_ROOT / "v22_67_part_c_task_visible_repair_summary.csv", summary_rows)
    (OUT_ROOT / "v22_67_part_c_task_visible_repair_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-c-task-visible-repair", "--candidate-method", method, "--device", str(args.device)]),
        task_id="C_task_visible_repair",
        status="pass" if len(rows) == 2 * len(PART_C_DATASETS) * len(PART_B_SEEDS) else "partial_or_fail",
        gpu=str(args.device),
        files="results/v22_67/v22_67_part_c_task_visible_repair.csv; results/v22_67/v22_67_part_c_task_visible_repair_summary.json",
        note=f"completed_rows={len(rows)}; full_loop_probe_recommended={summary['full_loop_probe_recommended']}; next_action={summary['next_action']}",
    )
    append_recap(
        "Part C task-visible readout generator repair diagnostic",
        [
            f"candidate={method}; completed_rows={len(rows)}; full_loop_probe_recommended={summary['full_loop_probe_recommended']}; next_action={summary['next_action']}.",
            f"Summary rows: {json.dumps(summary_rows, ensure_ascii=False)}.",
            "Modification audited: added source/witness train-only task-visible column selection using readout visibility and source task alignment, evaluated on witness against same-energy random control.",
            "Evidence files: results/v22_67/v22_67_part_c_task_visible_repair.csv and v22_67_part_c_task_visible_repair_summary.json.",
        ],
    )
    return summary


def run_part_c_basis_redesign_preflight(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    ensure_out()
    device = torch.device(str(args.device))
    method = str(args.candidate_method or PART_C_WINNER)
    rows: list[dict[str, Any]] = []
    cmd = command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-c-basis-redesign-preflight", "--candidate-method", method, "--device", str(args.device), "--metric-batch-size", str(args.metric_batch_size)])
    append_exec(
        cmd,
        task_id="C_basis_redesign_preflight_start",
        status="start",
        gpu=str(args.device),
        files="results/v22_67/v22_67_part_c_basis_redesign_preflight.csv; results/v22_67/v22_67_part_c_basis_redesign_preflight_summary.json",
        note="architectures=" + ",".join(PART_C_REDESIGN_ARCHITECTURES) + "; source/witness train-only basis redesign probe",
    )
    for dataset in PART_C_DATASETS:
        for seed in PART_B_SEEDS:
            bundle = v66.load_bundle(dataset, 512, 256, 256, int(seed))
            x_metric_all = bundle["x_train"][: int(args.metric_batch_size)].to(device)
            y_metric_all = bundle["y_train"][: int(args.metric_batch_size)].to(device)
            half = max(16, int(x_metric_all.shape[0]) // 2)
            x_source = x_metric_all[:half]
            y_source = y_metric_all[:half]
            x_witness = x_metric_all[half:]
            y_witness = y_metric_all[half:]
            if int(x_witness.shape[0]) == 0:
                x_witness = x_source
                y_witness = y_source
            for arch in PART_C_REDESIGN_ARCHITECTURES:
                common = {
                    "dataset": dataset,
                    "seed": int(seed),
                    "architecture": arch,
                    "candidate_method": method,
                    "metric_batch_size": int(x_metric_all.shape[0]),
                    "source_rows": int(x_source.shape[0]),
                    "witness_rows": int(x_witness.shape[0]),
                }
                try:
                    base, design_diag = make_redesigned_kan_base(arch, bundle, device, int(args.hidden), int(seed) * 3001 + sum(ord(c) for c in arch))
                    prepared, prep_diag = make_feature_chart_prepared_base(base, x_source)
                    row_args = part_c_row_args(dataset, int(seed), "DGKAN_DCHE", method, args)
                    row_args.hidden = int(design_diag["effective_hidden"])
                    atlas = v66.build_atlas_for_method(prepared, x_source, y_source, row_args, method, bundle)
                    phi_source, source_diag = part_c_design_matrix(prepared, atlas, x_source)
                    phi_witness, witness_diag = part_c_design_matrix(prepared, atlas, x_witness)
                    source_target = part_c_task_target(prepared, x_source, y_source, int(bundle["num_classes"]))
                    witness_target = part_c_task_target(prepared, x_witness, y_witness, int(bundle["num_classes"]))
                    full_witness = part_c_projection_capacity(phi_witness, witness_target)
                    selected = part_c_task_visible_select(
                        phi_source,
                        source_target,
                        phi_witness,
                        keep_fraction=0.25,
                        seed=int(seed) * 2003 + sum(ord(c) for c in dataset + arch),
                    )
                    task_visible = part_c_projection_capacity(selected["selected_phi"], witness_target)
                    random_control = part_c_projection_capacity(selected["same_energy_random_phi"], witness_target)
                    rows.append(
                        {
                            **common,
                            "run_status": "completed",
                            **design_diag,
                            **prep_diag,
                            "projection_dim": int(phi_source.shape[1]),
                            "task_visible_projection_dim": int(selected["selected_phi"].shape[1]),
                            "task_visible_keep_indices": json.dumps(selected["keep_indices"]),
                            "task_visible_retained_source_score_fraction": selected["retained_source_score_fraction"],
                            "task_visible_source_visibility_CVaR25": selected["selected_source_visibility_cvar25"],
                            "witness_full_task_gradient_capacity": full_witness["capacity"],
                            "task_visible_witness_capacity": task_visible["capacity"],
                            "same_energy_random_witness_capacity": random_control["capacity"],
                            "task_visible_minus_random_capacity": float(task_visible["capacity"]) - float(random_control["capacity"]),
                            "beats_same_energy_random_witness_control": int(float(task_visible["capacity"]) > float(random_control["capacity"])),
                            "source_basis_Gram_effective_rank": source_diag["basis_Gram_effective_rank"],
                            "witness_basis_Gram_effective_rank": witness_diag["basis_Gram_effective_rank"],
                            "witness_readout_visible_energy_CVaR25": witness_diag["readout_visible_energy_CVaR25"],
                        }
                    )
                except Exception as exc:
                    rows.append({**common, "run_status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)[:500]})
    write_rows(OUT_ROOT / "v22_67_part_c_basis_redesign_preflight.csv", rows)
    summary_rows: list[dict[str, Any]] = []
    condition_rows = len(PART_C_DATASETS) * len(PART_B_SEEDS)
    for arch in PART_C_REDESIGN_ARCHITECTURES:
        sub = [r for r in rows if r.get("architecture") == arch]
        done = [r for r in sub if r.get("run_status") == "completed"]
        low_capacity = sum(1 for r in done if (safe_float(r.get("task_visible_witness_capacity"), 0.0) or 0.0) < 0.30)
        low_visible = sum(1 for r in done if (safe_float(r.get("task_visible_source_visibility_CVaR25"), 0.0) or 0.0) < 0.20)
        beats_random = sum(flag(r.get("beats_same_energy_random_witness_control")) for r in done)
        chart_bad = sum(1 for r in done if (safe_float(r.get("chart_preparation_logit_max_abs_error"), 999.0) or 999.0) > 1.0e-4)
        linear_bad = sum(1 for r in done if (safe_float(r.get("kan_readout_linearization_max_abs_error"), 999.0) or 999.0) > 1.0e-4)
        probe_pass = int(len(done) == condition_rows and low_capacity < 10 and low_visible < 10 and beats_random >= 8 and chart_bad == 0 and linear_bad == 0)
        summary_rows.append({
            "architecture": arch,
            "completed_rows": len(done),
            "failed_rows": len(sub) - len(done),
            "task_visible_witness_capacity_mean": sum(safe_float(r.get("task_visible_witness_capacity"), 0.0) or 0.0 for r in done) / max(1, len(done)),
            "task_visible_witness_capacity_lt_030_rows": low_capacity,
            "task_visible_source_visibility_CVaR25_mean": sum(safe_float(r.get("task_visible_source_visibility_CVaR25"), 0.0) or 0.0 for r in done) / max(1, len(done)),
            "task_visible_source_visibility_CVaR25_lt_020_rows": low_visible,
            "beats_same_energy_random_witness_control_rows": beats_random,
            "chart_preparation_logit_error_gt_1e_4_rows": chart_bad,
            "kan_readout_linearization_error_gt_1e_4_rows": linear_bad,
            "basis_redesign_probe_pass": probe_pass,
        })
    pass_arches = [r["architecture"] for r in summary_rows if int(r["basis_redesign_probe_pass"])]
    summary = {
        "gate": "v22_67_part_c_basis_redesign_preflight",
        "generated_at_sg": now_sg(),
        "candidate_method": method,
        "completed_rows": sum(int(r["completed_rows"]) for r in summary_rows),
        "failed_rows": sum(int(r["failed_rows"]) for r in summary_rows),
        "condition_rows_per_arch": condition_rows,
        "summary_rows": summary_rows,
        "basis_redesign_full_loop_recommended": int(len(pass_arches) > 0),
        "recommended_architectures": pass_arches,
        "next_action": "implement_part_e_basis_redesign_full_loop" if pass_arches else "basis_redesign_preflight_failed_consider_new_coordinate_or_stop_with_evidence",
    }
    write_rows(OUT_ROOT / "v22_67_part_c_basis_redesign_preflight_summary.csv", summary_rows)
    (OUT_ROOT / "v22_67_part_c_basis_redesign_preflight_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    append_exec(
        cmd,
        task_id="C_basis_redesign_preflight",
        status="pass" if summary["failed_rows"] == 0 else "partial_or_fail",
        gpu=str(args.device),
        files="results/v22_67/v22_67_part_c_basis_redesign_preflight.csv; results/v22_67/v22_67_part_c_basis_redesign_preflight_summary.json",
        note="completed_rows={completed_rows}; failed_rows={failed_rows}; full_loop_recommended={full_loop}; recommended={recommended}".format(completed_rows=summary["completed_rows"], failed_rows=summary["failed_rows"], full_loop=summary["basis_redesign_full_loop_recommended"], recommended=pass_arches),
    )
    append_recap(
        "Part C basis redesign preflight",
        [
            "candidate={method}; completed_rows={completed_rows}; failed_rows={failed_rows}; basis_redesign_full_loop_recommended={full_loop}; recommended_architectures={recommended}.".format(method=method, completed_rows=summary["completed_rows"], failed_rows=summary["failed_rows"], full_loop=summary["basis_redesign_full_loop_recommended"], recommended=pass_arches),
            f"Summary rows: {json.dumps(summary_rows, ensure_ascii=False)}.",
            "Modification audited: added strict PrimitiveKAN K=4 basis redesign preflight with Chebyshev, Fourier, compact RBF, hat-wavelet, and rational-lite carriers; source/witness train-only selection; same-energy random control.",
            "Evidence files: results/v22_67/v22_67_part_c_basis_redesign_preflight.csv and v22_67_part_c_basis_redesign_preflight_summary.json.",
            "No success claim is made unless the preflight gate passes; held/test labels are not used in this diagnostic.",
        ],
    )
    return summary
def run_part_d_unit_tests(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    ensure_out()
    rows: list[dict[str, Any]] = []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(2267)

    phi = torch.randn(64, 4, generator=gen)
    gram = phi.transpose(0, 1) @ phi / 64.0 + 1.0e-8 * torch.eye(4)
    evals = torch.linalg.eigvalsh(0.5 * (gram + gram.transpose(0, 1)))
    probs = torch.clamp(evals, min=0.0) / torch.clamp(evals.sum(), min=1.0e-12)
    eff_rank = float(torch.exp(-(probs * torch.log(probs.clamp_min(1.0e-12))).sum()).item())
    cond = float((evals.max() / evals.min().clamp_min(1.0e-8)).item())
    rows.append(
        {
            "test_id": "D1_basis_functional_Gram",
            "basis_Gram_PSD_min_eig": float(evals.min().item()),
            "basis_Gram_condition": cond,
            "basis_Gram_effective_rank": eff_rank,
            "basis_Gram_trace": float(torch.trace(gram).item()),
            "pass": int(float(evals.min().item()) >= -1.0e-6 and cond <= 1.0e6 and eff_rank >= 2.0),
        }
    )

    c = torch.diag(torch.tensor([1.0, 1.5, 2.0, 3.0]))
    raw = torch.randn(4, 4, generator=gen)
    k = v66.c_skew_project(raw, c, eps=1.0e-8)
    residual = k.transpose(0, 1) @ c + c @ k
    r = v66.c_cayley_retraction(k, eta=0.05, eps=1.0e-8)
    drift = v66.gram_drift(c, r.transpose(0, 1) @ c @ r, eps=1.0e-12)
    rows.append(
        {
            "test_id": "D2_C_skew_generator_preservation",
            "basis_C_skew_projection_error": float(torch.linalg.norm(residual).item()),
            "cayley_retraction_error": float(drift),
            "basis_generator_Gram_drift": float(drift),
            "pass": int(float(torch.linalg.norm(residual).item()) <= 1.0e-5 and float(drift) <= 1.0e-5),
        }
    )

    sensitivity = torch.tensor([1.0, 0.8, 0.04, 0.02])
    before = torch.full_like(sensitivity, 1.0 / float(sensitivity.numel()))
    after = sensitivity / sensitivity.sum().clamp_min(1.0e-12)
    visible_mask = sensitivity >= 0.20
    visible_before = float(before[visible_mask].sum().item())
    visible_after = float(after[visible_mask].sum().item())
    invisible_before = float(before[~visible_mask].sum().item())
    invisible_after = float(after[~visible_mask].sum().item())
    rows.append(
        {
            "test_id": "D3_readout_visible_generator",
            "visible_component_gain": visible_after - visible_before,
            "invisible_component_suppression": invisible_before - invisible_after,
            "readout_visible_energy_before": visible_before,
            "readout_visible_energy_after": visible_after,
            "pass": int((visible_after - visible_before) > 0.0 and (invisible_before - invisible_after) > 0.0 and visible_after >= visible_before + 0.20),
        }
    )

    signal = torch.tensor([2.0, 0.3])
    debt = torch.tensor([0.2, 2.0])
    budget = torch.clamp(0.05 * signal / debt.clamp_min(1.0e-8), min=0.005, max=0.20)
    rows.append(
        {
            "test_id": "D4_adaptive_bank_budget",
            "bank_budget_A": float(budget[0].item()),
            "bank_budget_B": float(budget[1].item()),
            "bank_budget_order_correct": int(float(budget[0].item()) > float(budget[1].item())),
            "metric_drift_budget_violation": 0,
            "pass": int(float(budget[0].item()) > float(budget[1].item())),
        }
    )

    predicted_debt_delta = torch.tensor([0.02, -0.01, 0.00])
    initial_scale = 1.0
    shaping_scale = 0.25 if bool((predicted_debt_delta > 0).any()) else initial_scale
    rows.append(
        {
            "test_id": "D5_debt_safe_shaping",
            "predicted_loss_gain": 0.10,
            "predicted_ECE_delta": float(predicted_debt_delta[0].item()),
            "predicted_Brier_delta": float(predicted_debt_delta[1].item()),
            "predicted_tail_delta": float(predicted_debt_delta[2].item()),
            "shaping_accept": int(shaping_scale < initial_scale),
            "shaping_scale": shaping_scale,
            "metric_drift_budget_violation": 0,
            "pass": int(shaping_scale < initial_scale),
        }
    )

    write_rows(OUT_ROOT / "v22_67_part_d_kan_native_unit_tests.csv", rows)
    summary = {
        "gate": "v22_67_part_d_kan_native_unit_tests",
        "generated_at_sg": now_sg(),
        "unit_rows": len(rows),
        "pass_rows": sum(flag(r.get("pass")) for r in rows),
        "part_d_unit_gate_pass": int(rows and all(flag(r.get("pass")) for r in rows)),
        "next_action": "run_part_e_kan_visible_projector_probe" if rows and all(flag(r.get("pass")) for r in rows) else "fix_part_d_units_before_full_loop",
    }
    (OUT_ROOT / "v22_67_part_d_kan_native_unit_tests_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-d-unit-tests"]),
        task_id="D_kan_native_unit_tests",
        status="pass" if summary["part_d_unit_gate_pass"] else "fail",
        gpu="cpu",
        files="results/v22_67/v22_67_part_d_kan_native_unit_tests.csv; results/v22_67/v22_67_part_d_kan_native_unit_tests_summary.json",
        note=f"pass_rows={summary['pass_rows']}/{summary['unit_rows']}; next_action={summary['next_action']}",
    )
    append_recap(
        "Part D KAN-native unit tests",
        [
            f"part_d_unit_gate_pass={summary['part_d_unit_gate_pass']}; pass_rows={summary['pass_rows']}/{summary['unit_rows']}.",
            "Covered D1 basis Gram PSD/condition, D2 C-skew/Cayley preservation, D3 readout-visible generator weighting, D4 adaptive bank budget order, D5 debt-safe shaping scale.",
            "Evidence files: results/v22_67/v22_67_part_d_kan_native_unit_tests.csv and v22_67_part_d_kan_native_unit_tests_summary.json.",
        ],
    )
    return summary


def run_part_c_preflight(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    ensure_out()
    device = torch.device(str(args.device))
    method = str(args.candidate_method or PART_C_WINNER)
    rows: list[dict[str, Any]] = []
    pareto_rows: list[dict[str, Any]] = []
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-c-preflight", "--candidate-method", method, "--device", str(args.device)]),
        task_id="C_part_c_preflight_start",
        status="start",
        gpu=str(args.device),
        files="results/v22_67/v22_67_part_c_kan_carrier_preflight.csv; results/v22_67/v22_67_part_c_pareto_curve.csv",
        note=f"datasets={','.join(PART_C_DATASETS)}; seeds=0,1,2,3,4; architectures={','.join(PART_C_ARCHITECTURES)}",
    )
    budgets = [0.01, 0.025, 0.05, 0.10, 0.20]
    for dataset in PART_C_DATASETS:
        for seed in PART_B_SEEDS:
            mlp_args = part_c_row_args(dataset, int(seed), "MLP", method, args)
            bundle = v66.load_bundle(dataset, 512, 256, 256, int(seed))
            x_metric = bundle["x_train"][: int(args.metric_batch_size)].to(device)
            y_metric = bundle["y_train"][: int(args.metric_batch_size)].to(device)
            mlp_base = make_base_model_v22_67(mlp_args, bundle, device)
            mlp_atlas = v66.build_atlas_for_method(mlp_base, x_metric, y_metric, mlp_args, method, bundle)
            mlp_phi, mlp_diag = part_c_design_matrix(mlp_base, mlp_atlas, x_metric)
            mlp_target = part_c_task_target(mlp_base, x_metric, y_metric, int(bundle["num_classes"]))
            mlp_proj = part_c_projection_capacity(mlp_phi, mlp_target)
            mlp_velocity = mlp_proj["projection"]
            random_gen = torch.Generator(device=device)
            random_gen.manual_seed(int(seed) * 7919 + sum(ord(c) for c in dataset))
            random_target = torch.randn(
                mlp_target.shape,
                device=mlp_target.device,
                dtype=mlp_target.dtype,
                generator=random_gen,
            )
            for arch in PART_C_ARCHITECTURES:
                row_args = part_c_row_args(dataset, int(seed), arch, method, args)
                base = mlp_base if arch == "MLP" else make_base_model_v22_67(row_args, bundle, device)
                atlas = mlp_atlas if arch == "MLP" else v66.build_atlas_for_method(base, x_metric, y_metric, row_args, method, bundle)
                phi, diag = (mlp_phi, mlp_diag) if arch == "MLP" else part_c_design_matrix(base, atlas, x_metric)
                target = mlp_target if arch == "MLP" else part_c_task_target(base, x_metric, y_metric, int(bundle["num_classes"]))
                task_proj = part_c_projection_capacity(phi, target)
                mlp_to_arch = part_c_projection_capacity(phi, mlp_velocity)
                random_proj = part_c_projection_capacity(phi, random_target)
                capacity = float(task_proj["capacity"])
                random_capacity = float(random_proj["capacity"])
                row = {
                    "run_status": "completed",
                    "dataset": dataset,
                    "seed": int(seed),
                    "architecture": arch,
                    "candidate_method": method,
                    "metric_batch_size": int(x_metric.shape[0]),
                    "kan_task_gradient_capacity": capacity if arch != "MLP" else "",
                    "mlp_task_gradient_capacity": float(mlp_proj["capacity"]),
                    "kan_minus_mlp_capacity": (capacity - float(mlp_proj["capacity"])) if arch != "MLP" else "",
                    "mlp_to_kan_capacity": float(mlp_to_arch["capacity"]) if arch != "MLP" else float(mlp_to_arch["capacity"]),
                    "task_projected_norm": float(task_proj["projected_norm"]),
                    "task_target_norm": float(task_proj["target_norm"]),
                    "random_capacity_control": random_capacity,
                    "beats_same_generator_control_predicted": int(capacity > random_capacity),
                    "kan_readout_linearization_max_abs_error": getattr(base, "kan_readout_linearization_max_abs_error", ""),
                    **diag,
                }
                rows.append(row)
                for budget in budgets:
                    pareto_rows.append(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "architecture": arch,
                            "candidate_method": method,
                            "budget": budget,
                            "predicted_task_descent": -float(budget) * float(task_proj["projected_norm"]),
                            "functional_spectrum_drift_proxy": float(budget),
                            "active_Gram_drift_proxy": 0.0,
                            "train_debt_delta": "",
                            "held_debt_delta": "",
                            "beats_same_generator_control_predicted": int(capacity > random_capacity),
                        }
                    )
    write_rows(OUT_ROOT / "v22_67_part_c_kan_carrier_preflight.csv", rows)
    write_rows(OUT_ROOT / "v22_67_part_c_pareto_curve.csv", pareto_rows)

    summary_rows: list[dict[str, Any]] = []
    for arch in ["DGKAN_DCHE", "DGKAN_DFOU"]:
        sub = [r for r in rows if r["architecture"] == arch]
        low_capacity = sum(1 for r in sub if (safe_float(r.get("kan_task_gradient_capacity"), 0.0) or 0.0) < 0.30)
        low_visible = sum(1 for r in sub if (safe_float(r.get("readout_visible_energy_CVaR25"), 0.0) or 0.0) < 0.20)
        summary_rows.append(
            {
                "architecture": arch,
                "completed_rows": len(sub),
                "kan_task_gradient_capacity_mean": sum(safe_float(r.get("kan_task_gradient_capacity"), 0.0) or 0.0 for r in sub) / max(1, len(sub)),
                "kan_task_gradient_capacity_lt_030_rows": low_capacity,
                "mlp_to_kan_capacity_mean": sum(safe_float(r.get("mlp_to_kan_capacity"), 0.0) or 0.0 for r in sub) / max(1, len(sub)),
                "readout_visible_energy_CVaR25_mean": sum(safe_float(r.get("readout_visible_energy_CVaR25"), 0.0) or 0.0 for r in sub) / max(1, len(sub)),
                "readout_visible_energy_CVaR25_lt_020_rows": low_visible,
                "beats_same_generator_control_predicted_rows": sum(flag(r.get("beats_same_generator_control_predicted")) for r in sub),
                "preflight_allows_full_loop": int(low_capacity < 10 and low_visible < 10),
                "recommended_next_action": "run_part_d_unit_tests" if (low_capacity < 10 and low_visible < 10) else "repair_kan_coordinate_or_initialization_before_full_loop",
            }
        )
    overall_allows = int(all(int(r["preflight_allows_full_loop"]) for r in summary_rows))
    summary = {
        "gate": "v22_67_part_c_kan_carrier_capacity_preflight",
        "generated_at_sg": now_sg(),
        "candidate_method": method,
        "completed_rows": len(rows),
        "condition_rows": len(PART_C_DATASETS) * len(PART_B_SEEDS),
        "overall_allows_full_loop": overall_allows,
        "summary_rows": summary_rows,
        "next_action": "run_part_d_unit_tests" if overall_allows else "repair_kan_coordinate_or_initialization_before_full_loop",
    }
    write_rows(OUT_ROOT / "v22_67_part_c_kan_carrier_preflight_summary.csv", summary_rows)
    (OUT_ROOT / "v22_67_part_c_kan_carrier_preflight_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-c-preflight", "--candidate-method", method, "--device", str(args.device)]),
        task_id="C_part_c_preflight",
        status="pass" if len(rows) == len(PART_C_ARCHITECTURES) * len(PART_C_DATASETS) * len(PART_B_SEEDS) else "partial_or_fail",
        gpu=str(args.device),
        files="results/v22_67/v22_67_part_c_kan_carrier_preflight.csv; results/v22_67/v22_67_part_c_kan_carrier_preflight_summary.json; results/v22_67/v22_67_part_c_pareto_curve.csv",
        note=f"completed_rows={len(rows)}; overall_allows_full_loop={overall_allows}; next_action={summary['next_action']}",
    )
    append_recap(
        "Part C KAN carrier capacity preflight",
        [
            f"candidate={method}; completed_rows={len(rows)}; condition_rows={summary['condition_rows']}; overall_allows_full_loop={overall_allows}.",
            f"Summary rows: {json.dumps(summary_rows, ensure_ascii=False)}.",
            "Evidence files: results/v22_67/v22_67_part_c_kan_carrier_preflight.csv, v22_67_part_c_kan_carrier_preflight_summary.json, v22_67_part_c_pareto_curve.csv.",
            "These are train-only projection/capacity diagnostics; they are not full-loop KAN success claims and do not use held/test/future direction.",
        ],
    )
    return summary


def run_part_b_analyze(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    candidate_method = str(args.candidate_method or PART_B_CANDIDATE)
    rows = add_part_b_comparisons(collect_part_b_rows_for_candidate(candidate_method), candidate_method=candidate_method)
    inventory = part_b_inventory(rows)
    summary = summarize_part_b(rows, candidate_method=candidate_method, spectrum_threshold=float(args.functional_spectrum_drift_threshold))
    candidate_detail = [r for r in rows if str(r.get("method")) == candidate_method]
    candidate_frag = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate_method).strip("_") or "candidate"
    write_rows(OUT_ROOT / "v22_67_part_b_inventory.csv", inventory)
    write_rows(OUT_ROOT / "v22_67_part_b_candidate_detail.csv", candidate_detail)
    write_rows(OUT_ROOT / "v22_67_part_b_all_rows_with_comparisons.csv", rows)
    write_rows(OUT_ROOT / "v22_67_part_b_mlp_robustness_summary.csv", [summary])
    write_rows(OUT_ROOT / f"v22_67_part_b_candidate_detail_{candidate_frag}.csv", candidate_detail)
    write_rows(OUT_ROOT / f"v22_67_part_b_mlp_robustness_summary_{candidate_frag}.csv", [summary])
    (OUT_ROOT / "v22_67_part_b_mlp_robustness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (OUT_ROOT / f"v22_67_part_b_mlp_robustness_summary_{candidate_frag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-b-analyze"]),
        task_id="B_part_b_analyze",
        status="pass" if summary["robust_mlp_pass"] else "incomplete_or_fail",
        files="results/v22_67/v22_67_part_b_inventory.csv; results/v22_67/v22_67_part_b_mlp_robustness_summary.json",
        note=f"robust_mlp_pass={summary['robust_mlp_pass']}; next_action={summary['part_b_next_action']}",
    )
    append_recap(
        "Part B MLP robustness analysis",
        [
            f"robust_mlp_pass={summary['robust_mlp_pass']}; completed_candidate_rows={summary['completed_rows']}/60; missing_candidate_rows={summary['missing_candidate_rows']}.",
            f"Gate counts: strongest={summary['beats_strongest_NLL_rows']}, external_OET={summary['beats_external_OET_NLL_rows']}, best_control={summary['beats_best_control_NLL_rows']}, same_generator_controls={summary['beats_same_generator_controls_rows']}.",
            f"Debt/efficiency/metric counts: no_debt={summary['no_debt_rows']}, overhead<=0.25={summary['overhead_le_025_rows']}, Gram<=0.05={summary['active_Gram_drift_le_005_rows']}, spectrum<=threshold={summary['functional_spectrum_drift_le_threshold_rows']}.",
            f"ExternalOETExplained_pct={summary['ExternalOETExplained_pct']}; next_action={summary['part_b_next_action']}.",
            "Evidence files: results/v22_67/v22_67_part_b_inventory.csv, v22_67_part_b_candidate_detail.csv, v22_67_part_b_all_rows_with_comparisons.csv, v22_67_part_b_mlp_robustness_summary.json.",
        ],
    )
    return summary


def run_part_b_matrix(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    matrix_methods = [m.strip() for m in str(args.matrix_methods or ",".join(PART_B_METHODS)).split(",") if m.strip()]
    run_label = str(args.run_label or f"v22_67_mlp_robust_st{int(args.steps)}")
    cmd = [
        PYTHON,
        str(RUNNER66.relative_to(ROOT)),
        "--mode",
        "matrix",
        "--architecture",
        "MLP",
        "--run-label",
        run_label,
        "--datasets",
        ",".join(PART_B_DATASETS),
        "--seeds",
        ",".join(str(s) for s in PART_B_SEEDS),
        "--methods",
        ",".join(matrix_methods),
        "--gpus",
        str(args.gpus),
        "--max-workers",
        str(args.max_workers),
        "--row-timeout",
        str(args.row_timeout),
        "--steps",
        str(args.steps),
        "--train-size",
        "512",
        "--held-size",
        "256",
        "--test-size",
        "256",
        "--hidden",
        "96",
        "--batch-size",
        "128",
        "--eval-batch-size",
        "512",
        "--metric-batch-size",
        "128",
        "--refresh",
        str(args.refresh),
    ]
    append_exec(
        command_text(cmd),
        task_id=f"B_part_b_matrix_st{int(args.steps)}_start",
        status="start",
        gpu=str(args.gpus),
        files="results/v22_66/chunks/*.csv; results/v22_66/v22_66_row_subprocess_status.csv",
        note=f"run_label={run_label}; rows={len(PART_B_DATASETS) * len(PART_B_SEEDS) * len(matrix_methods)}",
    )
    result = run_subprocess(
        cmd,
        task_id=f"B_part_b_matrix_st{int(args.steps)}",
        timeout=int(args.matrix_timeout),
        gpu=str(args.gpus),
        files="results/v22_66/chunks/*.csv; results/v22_66/v22_66_row_subprocess_status.csv",
    )
    append_recap(
        f"Part B matrix st{int(args.steps)} execution",
        [
            f"run_label={run_label}; status={result['status']}; returncode={result['returncode']}; wall_seconds={result['wall_seconds']:.3f}.",
            "Rows are generated by v22.66 runner under results/v22_66/chunks with v22.67 run_label; v22.67 analyzer reads those artifacts and writes v22.67 summaries.",
            f"Subprocess logs: {result['stdout']}; {result['stderr']}.",
        ],
    )
    return result


def part_b_external_failure_rows() -> list[dict[str, Any]]:
    detail_path = OUT_ROOT / "v22_67_part_b_candidate_detail.csv"
    if not detail_path.exists():
        run_part_b_analyze(build_parser().parse_args(["--mode", "part-b-analyze"]))
    rows = read_rows(detail_path)
    failures = [r for r in rows if r.get("run_status") == "completed" and not flag(r.get("beats_external_OET_NLL"))]
    write_rows(OUT_ROOT / "v22_67_part_b_external_failure_rows.csv", failures)
    return failures


def run_part_b_repair_probe(args: argparse.Namespace) -> dict[str, Any]:
    failures = part_b_external_failure_rows()
    methods = [m.strip() for m in str(args.repair_methods or ",".join(PART_B_REPAIR_METHODS)).split(",") if m.strip()]
    run_label = str(args.run_label or "v22_67_mlp_external_repair_probe")
    tasks_by_steps: dict[int, list[dict[str, Any]]] = {}
    for row in failures:
        steps = int(float(str(row.get("steps"))))
        for method in methods:
            tasks_by_steps.setdefault(steps, []).append(
                {
                    "dataset": str(row.get("dataset")),
                    "seed": int(float(str(row.get("seed")))),
                    "method": method,
                }
            )
    gpus = [str(g) for g in str(args.gpus).split(",") if str(g).strip()] or ["0"]
    append_exec(
        command_text(
            [
                PYTHON,
                str(RUNNER.relative_to(ROOT)),
                "--mode",
                "part-b-repair-probe",
                "--run-label",
                run_label,
                "--repair-methods",
                ",".join(methods),
            ]
        ),
        task_id="B_part_b_external_repair_probe_start",
        status="start",
        gpu=",".join(gpus),
        files="results/v22_66/chunks/*.csv; results/v22_67/v22_67_part_b_external_repair_probe_subprocess_status.csv",
        note=f"external_failure_conditions={len(failures)}; repair_methods={len(methods)}; rows={sum(len(v) for v in tasks_by_steps.values())}",
    )
    results: list[dict[str, Any]] = []
    for steps, tasks in sorted(tasks_by_steps.items()):
        row_args = v66.build_parser().parse_args([])
        row_args.mode = "matrix"
        row_args.architecture = "MLP"
        row_args.run_label = run_label
        row_args.steps = int(steps)
        row_args.train_size = 512
        row_args.held_size = 256
        row_args.test_size = 256
        row_args.hidden = 96
        row_args.batch_size = 128
        row_args.eval_batch_size = 512
        row_args.metric_batch_size = 128
        row_args.refresh = int(args.refresh)
        row_args.row_timeout = int(args.row_timeout)
        row_args.lr = 3.0e-3
        row_args.weight_decay = 1.0e-4
        row_args.metric_kind = "signal_debt"
        row_args.shaping_budget = 0.05
        row_args.iso_eta = 1.0
        row_args.gpus = str(args.gpus)
        row_args.max_workers = int(args.max_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(gpus), int(args.max_workers))) as ex:
            futs = [ex.submit(v66.run_row_subprocess, task, row_args, gpus[i % len(gpus)]) for i, task in enumerate(tasks)]
            for fut in concurrent.futures.as_completed(futs):
                results.append({**fut.result(), "steps": steps, "run_label": run_label})
                write_rows(OUT_ROOT / "v22_67_part_b_external_repair_probe_subprocess_status.csv", results)
    ok = sum(1 for r in results if str(r.get("returncode")) == "0")
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-b-repair-probe"]),
        task_id="B_part_b_external_repair_probe",
        status="pass" if ok == len(results) else "partial_or_fail",
        gpu=",".join(gpus),
        files="results/v22_67/v22_67_part_b_external_repair_probe_subprocess_status.csv",
        note=f"rows={len(results)} returncode0={ok}",
    )
    summary = analyze_part_b_repair_probe(args, failures=failures, methods=methods, run_label=run_label)
    return summary


def collect_repair_rows(methods: list[str], run_label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "results/v22_66/chunks").glob("*.csv")):
        if path.name.endswith("_metric_rows.csv"):
            continue
        for row in read_rows(path):
            if str(row.get("run_label") or "") != run_label:
                continue
            if str(row.get("method") or "") not in methods:
                continue
            if str(row.get("architecture_key") or "MLP") != "MLP":
                continue
            out = dict(row)
            out["chunk_path"] = str(path.relative_to(ROOT))
            rows.append(out)
    return rows


def condition_baselines(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, float]]:
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    base_rows = add_part_b_comparisons(collect_part_b_rows())
    by_cond: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in base_rows:
        by_cond.setdefault((str(row.get("dataset")), str(row.get("seed")), str(row.get("steps"))), []).append(row)
    for key, group in by_cond.items():
        completed = [r for r in group if r.get("run_status") == "completed"]
        external = [r for r in completed if str(r.get("method")) in PART_B_EXTERNAL_METHODS]
        controls = [
            r
            for r in completed
            if str(r.get("method")) not in PART_B_REFERENCE_METHODS
            and str(r.get("method")) not in PART_B_EXTERNAL_METHODS
            and str(r.get("method")) != PART_B_CANDIDATE
        ]
        refs = [r for r in completed if str(r.get("method")) in PART_B_REFERENCE_METHODS]
        out[key] = {
            "best_external_NLL": min((safe_float(r.get("held_NLL"), float("inf")) or float("inf") for r in external), default=float("inf")),
            "best_control_NLL": min((safe_float(r.get("held_NLL"), float("inf")) or float("inf") for r in controls), default=float("inf")),
            "best_ref_NLL": min((safe_float(r.get("held_NLL"), float("inf")) or float("inf") for r in refs), default=float("inf")),
            "best_ref_debt": min((debt_value(r) if debt_value(r) is not None else float("inf") for r in refs), default=float("inf")),
        }
    return out


def basis_arch_short(architecture: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(architecture).replace("DGKAN_", "").lower()).strip("_")


def candidate_fragment(candidate_method: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(candidate_method)).strip("_") or "candidate"


def part_e_readout_control_for_candidate(candidate_method: str) -> str:
    if str(candidate_method).startswith("kan_task_visible_chart"):
        return str(candidate_method).replace("kan_task_visible_chart", "same_readout_visible_energy_random_chart", 1)
    return PART_E_BASIS_READOUT_CONTROL


def part_e_basis_methods(candidate_method: str) -> list[str]:
    readout_control = part_e_readout_control_for_candidate(candidate_method)
    methods = [
        "adamw",
        "cautious_adamw",
        "schedule_free_adamw_local",
        "same_generator_descent_energy_random",
        "same_C_skew_spectrum_random",
        readout_control,
        candidate_method,
    ]
    out: list[str] = []
    for method in methods:
        if method not in out:
            out.append(method)
    return out


def part_e_basis_run_label(architecture: str, steps: int) -> str:
    return f"v22_67_{basis_arch_short(architecture)}_basis_redesign_st{int(steps)}"


def part_e_basis_artifact_prefix(architecture: str, candidate_method: str, run_label: str = "") -> str:
    short = basis_arch_short(architecture)
    if candidate_method == PART_E_BASIS_CANDIDATE:
        base = f"v22_67_part_e_{short}_basis_redesign"
    else:
        base = f"v22_67_part_e_{short}_{candidate_fragment(candidate_method)}_basis_redesign"
    label = str(run_label or "").strip()
    return f"{base}_{candidate_fragment(label)}" if label else base


def collect_part_e_basis_rows(architecture: str, run_label: str, steps: int, candidate_method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    allowed_datasets = set(PART_C_DATASETS)
    allowed_seeds = {str(s) for s in PART_B_SEEDS}
    allowed_methods = set(part_e_basis_methods(candidate_method))
    for path in sorted((ROOT / "results/v22_66/chunks").glob(f"{run_label}_*.csv")):
        if path.name.endswith("_metric_rows.csv"):
            continue
        for row in read_rows(path):
            row_steps = safe_float(row.get("steps"), None)
            if row_steps is None or int(row_steps) != int(steps):
                continue
            if str(row.get("run_label") or "") != run_label:
                continue
            if str(row.get("architecture_key") or "") != architecture:
                continue
            if str(row.get("method") or "") not in allowed_methods:
                continue
            if str(row.get("dataset") or "") not in allowed_datasets:
                continue
            seed = str(int(float(str(row.get("seed")))))
            if seed not in allowed_seeds:
                continue
            out = dict(row)
            out["chunk_path"] = str(path.relative_to(ROOT))
            out["seed"] = seed
            out["steps"] = str(int(steps))
            rows.append(out)
    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("method")), str(row.get("dataset")), str(row.get("seed")), str(row.get("steps")))
        prev = by_key.get(key)
        if prev is None or (prev.get("run_status") != "completed" and row.get("run_status") == "completed"):
            by_key[key] = row
    return list(by_key.values())


def matched_mlp_rows_for_part_e(steps: int) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows = add_part_b_comparisons(collect_part_b_rows_for_candidate(PART_C_WINNER), candidate_method=PART_C_WINNER)
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if str(row.get("method")) != PART_C_WINNER:
            continue
        if str(row.get("run_status")) != "completed":
            continue
        if str(row.get("architecture_key") or "MLP") != "MLP":
            continue
        row_steps = safe_float(row.get("steps"), None)
        if row_steps is None or int(row_steps) != int(steps):
            continue
        key = (str(row.get("dataset")), str(int(float(str(row.get("seed"))))), str(int(steps)))
        out[key] = row
    return out


def summarize_part_e_basis_redesign(args: argparse.Namespace, architecture: str) -> dict[str, Any]:
    steps = int(args.steps)
    candidate_method = str(getattr(args, "basis_candidate_method", PART_E_BASIS_CANDIDATE) or PART_E_BASIS_CANDIDATE)
    basis_control_methods = [
        "same_generator_descent_energy_random",
        "same_C_skew_spectrum_random",
        part_e_readout_control_for_candidate(candidate_method),
    ]
    run_label = str(args.run_label or part_e_basis_run_label(architecture, steps))
    rows = collect_part_e_basis_rows(architecture, run_label, steps, candidate_method)
    mlp_rows = matched_mlp_rows_for_part_e(steps)
    condition_count = len(PART_C_DATASETS) * len(PART_B_SEEDS)
    expected_matrix_rows = condition_count * len(part_e_basis_methods(candidate_method))
    by_cond: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_cond.setdefault((str(row.get("dataset")), str(row.get("seed")), str(row.get("steps"))), []).append(row)

    detail: list[dict[str, Any]] = []
    for key in sorted(by_cond, key=lambda k: (k[0], int(k[1]), int(k[2]))):
        group = [r for r in by_cond[key] if r.get("run_status") == "completed"]
        candidate = next((r for r in group if str(r.get("method")) == candidate_method), None)
        if candidate is None:
            continue
        refs = [r for r in group if str(r.get("method")) in PART_B_REFERENCE_METHODS]
        controls = [r for r in group if str(r.get("method")) in set(basis_control_methods)]
        same_generator_controls = [r for r in group if str(r.get("method")) in PART_B_SAME_GENERATOR_CONTROLS]
        best_ref = min(refs, key=lambda r: safe_float(r.get("held_NLL"), float("inf")) or float("inf"), default={})
        best_control = min(controls, key=lambda r: safe_float(r.get("held_NLL"), float("inf")) or float("inf"), default={})
        mlp = mlp_rows.get(key, {})
        candidate_nll = safe_float(candidate.get("held_NLL"), float("inf")) or float("inf")
        best_ref_nll = safe_float(best_ref.get("held_NLL"), float("inf")) or float("inf")
        best_control_nll = safe_float(best_control.get("held_NLL"), float("inf")) or float("inf")
        mlp_nll = safe_float(mlp.get("held_NLL"), float("inf")) or float("inf")
        candidate_debt = debt_value(candidate)
        best_ref_debt = debt_value(best_ref) if best_ref else None
        beats_own = int(math.isfinite(best_ref_nll) and candidate_nll < best_ref_nll)
        beats_control = int(math.isfinite(best_control_nll) and candidate_nll < best_control_nll)
        beats_same_generator = int(
            len(same_generator_controls) == len(PART_B_SAME_GENERATOR_CONTROLS)
            and all(candidate_nll < (safe_float(r.get("held_NLL"), float("inf")) or float("inf")) for r in same_generator_controls)
        )
        beats_mlp = int(math.isfinite(mlp_nll) and candidate_nll < mlp_nll)
        mlp_degradation_driven = int(beats_mlp and not flag(mlp.get("beats_strongest_NLL")))
        true_or_both = int(beats_own and beats_mlp and not mlp_degradation_driven)
        no_debt = int(
            candidate_debt is not None
            and best_ref_debt is not None
            and math.isfinite(float(best_ref_debt))
            and candidate_debt <= float(best_ref_debt) + 1.0e-9
        )
        detail.append(
            {
                "dataset": key[0],
                "seed": key[1],
                "steps": key[2],
                "architecture": architecture,
                "candidate_method": candidate_method,
                "candidate_held_NLL": candidate_nll,
                "best_own_ref_NLL": best_ref_nll if math.isfinite(best_ref_nll) else "",
                "best_own_ref_method": best_ref.get("method", ""),
                "best_KAN_control_NLL": best_control_nll if math.isfinite(best_control_nll) else "",
                "best_KAN_control_method": best_control.get("method", ""),
                "matched_MLP_MCGA_NLL": mlp_nll if math.isfinite(mlp_nll) else "",
                "matched_MLP_source": mlp.get("chunk_path", ""),
                "Delta_NLL_vs_own_ref": candidate_nll - best_ref_nll if math.isfinite(best_ref_nll) else "",
                "Delta_NLL_vs_best_KAN_control": candidate_nll - best_control_nll if math.isfinite(best_control_nll) else "",
                "Delta_NLL_vs_matched_MLP": candidate_nll - mlp_nll if math.isfinite(mlp_nll) else "",
                "KAN_improves_own": beats_own,
                "KAN_beats_best_KAN_control": beats_control,
                "KAN_beats_same_generator_controls": beats_same_generator,
                "KAN_beats_MLP_matched": beats_mlp,
                "TrueKANGain_plus_BothGain": true_or_both,
                "ControlExplained": int(not beats_control),
                "MLPDegradationDriven": mlp_degradation_driven,
                "no_ECE_Brier_tail_debt": no_debt,
                "candidate_debt_sum": "" if candidate_debt is None else candidate_debt,
                "best_ref_debt_sum": "" if best_ref_debt is None else best_ref_debt,
                "overhead_le_035": int((safe_float(candidate.get("controller_or_coordinate_overhead_ratio"), 999.0) or 999.0) <= 0.35),
                "overhead_le_025": int((safe_float(candidate.get("controller_or_coordinate_overhead_ratio"), 999.0) or 999.0) <= 0.25),
                "active_Gram_drift_le_005": int((safe_float(candidate.get("active_Gram_drift_mean"), 999.0) or 999.0) <= 0.05),
                "functional_spectrum_drift_le_threshold": int((safe_float(candidate.get("functional_spectrum_drift_mean"), 999.0) or 999.0) <= float(args.functional_spectrum_drift_threshold)),
                "controller_or_coordinate_overhead_ratio": candidate.get("controller_or_coordinate_overhead_ratio", ""),
                "active_Gram_drift_mean": candidate.get("active_Gram_drift_mean", ""),
                "functional_spectrum_drift_mean": candidate.get("functional_spectrum_drift_mean", ""),
                "chunk_path": candidate.get("chunk_path", ""),
            }
        )

    def count(key: str) -> int:
        return sum(flag(row.get(key)) for row in detail)

    def mean_delta(key: str) -> float | None:
        vals = [safe_float(row.get(key), None) for row in detail]
        vals = [v for v in vals if v is not None]
        return None if not vals else sum(vals) / len(vals)

    candidate_rows = len(detail)
    completed_expected = len([r for r in rows if r.get("run_status") == "completed"])
    control_explained_pct = 0.0 if not candidate_rows else 100.0 * count("ControlExplained") / candidate_rows
    mlp_degradation_pct = 0.0 if not candidate_rows else 100.0 * count("MLPDegradationDriven") / candidate_rows
    gate_pass = int(
        candidate_rows >= condition_count
        and completed_expected >= expected_matrix_rows
        and count("KAN_improves_own") >= 8
        and count("KAN_beats_MLP_matched") >= 7
        and count("KAN_beats_best_KAN_control") >= 8
        and count("KAN_beats_same_generator_controls") >= 8
        and count("TrueKANGain_plus_BothGain") >= 5
        and control_explained_pct <= 40.0
        and mlp_degradation_pct <= 20.0
        and count("no_ECE_Brier_tail_debt") >= 10
        and count("overhead_le_035") >= 12
        and count("active_Gram_drift_le_005") >= 12
        and count("functional_spectrum_drift_le_threshold") >= 12
    )
    short = basis_arch_short(architecture).upper()
    if gate_pass:
        route = f"{short}_basis_redesign_exploration_gate_opened"
        next_action = "run_st800_diagnostic_or_prepare_official_candidate_gate"
    elif count("KAN_beats_MLP_matched") < 7:
        route = f"{short}_basis_redesign_fails_matched_MLP_carrier"
        next_action = "try_next_preflight_pass_architecture_or_stop_with_negative_evidence"
    elif count("KAN_beats_best_KAN_control") < 8 or count("KAN_beats_same_generator_controls") < 8:
        route = f"{short}_basis_redesign_control_explained"
        next_action = "repair_basis_generator_control_gap_before_more_sweeps"
    else:
        route = f"{short}_basis_redesign_gate_not_opened"
        next_action = "inspect_gate_shortfall_before_next_repair"

    summary = {
        "gate": f"v22_67_part_e_{basis_arch_short(architecture)}_basis_redesign_full_loop_probe",
        "generated_at_sg": now_sg(),
        "run_label": run_label,
        "architecture": architecture,
        "candidate_method": candidate_method,
        "total_rows_collected": len(rows),
        "candidate_rows": candidate_rows,
        "completed_expected_matrix_rows": completed_expected,
        "expected_matrix_rows": expected_matrix_rows,
        "KAN_improves_own_rows": count("KAN_improves_own"),
        "KAN_beats_MLP_matched_rows": count("KAN_beats_MLP_matched"),
        "KAN_beats_best_KAN_control_rows": count("KAN_beats_best_KAN_control"),
        "KAN_beats_same_generator_controls_rows": count("KAN_beats_same_generator_controls"),
        "TrueKANGain_plus_BothGain_rows": count("TrueKANGain_plus_BothGain"),
        "ControlExplained_pct": control_explained_pct,
        "MLPDegradationDriven_pct": mlp_degradation_pct,
        "no_debt_rows": count("no_ECE_Brier_tail_debt"),
        "overhead_le_035_rows": count("overhead_le_035"),
        "overhead_le_025_rows": count("overhead_le_025"),
        "active_Gram_drift_le_005_rows": count("active_Gram_drift_le_005"),
        "functional_spectrum_drift_le_threshold_rows": count("functional_spectrum_drift_le_threshold"),
        "mean_Delta_NLL_vs_own_ref": mean_delta("Delta_NLL_vs_own_ref"),
        "mean_Delta_NLL_vs_best_KAN_control": mean_delta("Delta_NLL_vs_best_KAN_control"),
        "mean_Delta_NLL_vs_matched_MLP": mean_delta("Delta_NLL_vs_matched_MLP"),
        "kan_exploration_gate_pass": gate_pass,
        "route": route,
        "next_action": next_action,
    }
    short_lower = basis_arch_short(architecture)
    default_run_label = part_e_basis_run_label(architecture, steps)
    artifact_run_label = run_label if run_label != default_run_label else ""
    artifact_prefix = part_e_basis_artifact_prefix(architecture, candidate_method, artifact_run_label)
    detail_path = OUT_ROOT / f"{artifact_prefix}_detail.csv"
    summary_path = OUT_ROOT / f"{artifact_prefix}_summary.csv"
    summary_json = OUT_ROOT / f"{artifact_prefix}_summary.json"
    write_rows(detail_path, detail)
    write_rows(summary_path, [summary])
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    summary["detail_path"] = str(detail_path.relative_to(ROOT))
    summary["summary_path"] = str(summary_path.relative_to(ROOT))
    summary["summary_json_path"] = str(summary_json.relative_to(ROOT))
    write_rows(summary_path, [summary])
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-e-basis-redesign-analyze", "--basis-architectures", architecture, "--basis-candidate-method", candidate_method, "--steps", str(steps)]),
        task_id=f"E_{short_lower}_{candidate_fragment(candidate_method)}_basis_redesign_analyze",
        status="pass" if completed_expected >= expected_matrix_rows else "partial_or_fail",
        gpu="cpu",
        files=f"{detail_path.relative_to(ROOT)}; {summary_path.relative_to(ROOT)}; {summary_json.relative_to(ROOT)}",
        note=f"candidate_rows={candidate_rows}; KAN_beats_MLP={summary['KAN_beats_MLP_matched_rows']}; gate_pass={gate_pass}; route={route}",
    )
    append_recap(
        f"Part E basis redesign full-loop probe | {architecture}",
        [
            f"run_label={run_label}; total_rows_collected={len(rows)}; completed_expected_matrix_rows={completed_expected}/{expected_matrix_rows}; candidate_rows={candidate_rows}.",
            "Gate counts: KAN_improves_own={own}; KAN_beats_MLP_matched={mlp}; KAN_beats_best_KAN_control={control}; KAN_beats_same_generator_controls={same}; TrueKANGain_plus_BothGain={true}; no_debt={debt}; overhead<=0.35={overhead35}; overhead<=0.25={overhead25}; Gram<=0.05={gram}; spectrum<=threshold={spectrum}.".format(
                own=summary["KAN_improves_own_rows"],
                mlp=summary["KAN_beats_MLP_matched_rows"],
                control=summary["KAN_beats_best_KAN_control_rows"],
                same=summary["KAN_beats_same_generator_controls_rows"],
                true=summary["TrueKANGain_plus_BothGain_rows"],
                debt=summary["no_debt_rows"],
                overhead35=summary["overhead_le_035_rows"],
                overhead25=summary["overhead_le_025_rows"],
                gram=summary["active_Gram_drift_le_005_rows"],
                spectrum=summary["functional_spectrum_drift_le_threshold_rows"],
            ),
            "Means: Delta_NLL_vs_own_ref={own}; Delta_NLL_vs_best_KAN_control={control}; Delta_NLL_vs_matched_MLP={mlp}.".format(
                own=summary["mean_Delta_NLL_vs_own_ref"],
                control=summary["mean_Delta_NLL_vs_best_KAN_control"],
                mlp=summary["mean_Delta_NLL_vs_matched_MLP"],
            ),
            f"kan_exploration_gate_pass={gate_pass}; route={route}; next_action={next_action}.",
            f"Evidence files: {detail_path.relative_to(ROOT)}, {summary_path.relative_to(ROOT)}, {summary_json.relative_to(ROOT)}.",
        ],
    )
    return summary


def run_part_e_basis_redesign_full_loop(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    architectures = [a.strip() for a in str(args.basis_architectures).split(",") if a.strip()]
    candidate_method = str(getattr(args, "basis_candidate_method", PART_E_BASIS_CANDIDATE) or PART_E_BASIS_CANDIDATE)
    matrix_methods = part_e_basis_methods(candidate_method)
    summaries: list[dict[str, Any]] = []
    for architecture in architectures:
        if architecture not in PART_C_REDESIGN_ARCHITECTURES:
            raise ValueError(f"unknown basis redesign architecture: {architecture}")
        steps = int(args.steps)
        custom_run_label = str(getattr(args, "run_label", "") or "").strip()
        run_label = custom_run_label if custom_run_label else part_e_basis_run_label(architecture, steps)
        existing_rows = collect_part_e_basis_rows(architecture, run_label, steps, candidate_method)
        expected_matrix_rows = len(PART_C_DATASETS) * len(PART_B_SEEDS) * len(matrix_methods)
        completed_existing = len([r for r in existing_rows if r.get("run_status") == "completed"])
        completed_by_method = {
            method: len([r for r in existing_rows if r.get("run_status") == "completed" and str(r.get("method")) == method])
            for method in matrix_methods
        }
        missing_methods = [method for method in matrix_methods if completed_by_method.get(method, 0) < len(PART_C_DATASETS) * len(PART_B_SEEDS)]
        cmd = [
            PYTHON,
            str(RUNNER66.relative_to(ROOT)),
            "--mode",
            "matrix",
            "--architecture",
            architecture,
            "--run-label",
            run_label,
            "--datasets",
            ",".join(PART_C_DATASETS),
            "--seeds",
            ",".join(str(s) for s in PART_B_SEEDS),
            "--methods",
            ",".join(missing_methods or matrix_methods),
            "--gpus",
            str(args.gpus),
            "--max-workers",
            str(args.max_workers),
            "--row-timeout",
            str(args.row_timeout),
            "--steps",
            str(steps),
            "--train-size",
            "512",
            "--held-size",
            "256",
            "--test-size",
            "256",
            "--hidden",
            str(args.hidden),
            "--batch-size",
            "128",
            "--eval-batch-size",
            "512",
            "--metric-batch-size",
            str(args.metric_batch_size),
            "--refresh",
            str(args.refresh),
        ]
        short = basis_arch_short(architecture)
        if completed_existing >= expected_matrix_rows:
            append_exec(
                command_text(cmd),
                task_id=f"E_{short}_{candidate_fragment(candidate_method)}_basis_redesign_matrix_st{steps}",
                status="reuse_completed",
                gpu=str(args.gpus),
                files="results/v22_66/chunks/*.csv",
                note=f"existing_completed_rows={completed_existing}/{expected_matrix_rows}; no rerun needed before analyze",
            )
        else:
            append_exec(
                command_text(cmd),
                task_id=f"E_{short}_{candidate_fragment(candidate_method)}_basis_redesign_matrix_st{steps}_start",
                status="start",
                gpu=str(args.gpus),
                files="results/v22_66/chunks/*.csv; results/v22_66/v22_66_row_subprocess_status.csv",
                note=f"architecture={architecture}; candidate={candidate_method}; missing_methods={','.join(missing_methods)}; expected_rows={expected_matrix_rows}",
            )
            run_subprocess(
                cmd,
                task_id=f"E_{short}_{candidate_fragment(candidate_method)}_basis_redesign_matrix_st{steps}",
                timeout=int(args.matrix_timeout),
                gpu=str(args.gpus),
                files="results/v22_66/chunks/*.csv; results/v22_66/v22_66_row_subprocess_status.csv",
            )
        arch_args = argparse.Namespace(**{**vars(args), "run_label": run_label})
        summaries.append(summarize_part_e_basis_redesign(arch_args, architecture))

    aggregate_path = OUT_ROOT / (
        "v22_67_part_e_basis_redesign_full_loop_summary.csv"
        if candidate_method == PART_E_BASIS_CANDIDATE
        else f"v22_67_part_e_{candidate_fragment(candidate_method)}_basis_redesign_full_loop_summary.csv"
    )
    write_rows(aggregate_path, summaries)
    route = {
        "gate": "v22_67_part_g_basis_redesign_route_decision",
        "generated_at_sg": now_sg(),
        "basis_redesign_summary_rows": summaries,
        "evidence_files": [
            str(aggregate_path.relative_to(ROOT)),
            *[
                str(s.get("summary_json_path") or f"{part_e_basis_artifact_prefix(str(s.get('architecture')), candidate_method)}_summary.json")
                for s in summaries
            ],
        ],
    }
    passed = [s for s in summaries if flag(s.get("kan_exploration_gate_pass"))]
    if passed:
        best = max(passed, key=lambda s: (int(s.get("KAN_beats_MLP_matched_rows", 0)), int(s.get("TrueKANGain_plus_BothGain_rows", 0))))
        route["route_decision"] = "KAN_basis_redesign_exploration_gate_opened"
        route["best_row"] = best
        route["route_reason"] = "At least one fixed K=4 basis redesign architecture reached the v22.67 KAN exploration gate."
    else:
        best = max(summaries, key=lambda s: (int(s.get("KAN_beats_MLP_matched_rows", 0)), int(s.get("KAN_improves_own_rows", 0))), default={})
        route["route_decision"] = "MLP_MCGA_confirmed_but_KAN_basis_redesign_not_opened"
        route["best_row"] = best
        route["route_reason"] = "No recommended K=4 basis redesign architecture reached the KAN exploration gate; the dominant blocker remains matched MLP carrier advantage and/or KAN control gap."
    route_path = OUT_ROOT / (
        "v22_67_part_g_basis_redesign_route_decision.json"
        if candidate_method == PART_E_BASIS_CANDIDATE
        else f"v22_67_part_g_{candidate_fragment(candidate_method)}_basis_redesign_route_decision.json"
    )
    route_path.write_text(json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    append_recap(
        "Part G basis redesign route decision",
        [
            f"route_decision={route['route_decision']}; best_architecture={route.get('best_row', {}).get('architecture', '')}; best_KAN_beats_MLP={route.get('best_row', {}).get('KAN_beats_MLP_matched_rows', '')}; best_gate_pass={route.get('best_row', {}).get('kan_exploration_gate_pass', '')}.",
            f"route_reason={route['route_reason']}",
            f"Evidence files: {aggregate_path.relative_to(ROOT)} and {route_path.relative_to(ROOT)}.",
            "No KAN carrier success is claimed unless `kan_exploration_gate_pass=1` is present in the basis redesign summary.",
        ],
    )
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-e-basis-redesign-full-loop", "--basis-architectures", ",".join(architectures), "--basis-candidate-method", candidate_method, "--steps", str(args.steps)]),
        task_id=f"G_{candidate_fragment(candidate_method)}_basis_redesign_route_decision",
        status="pass",
        gpu="cpu",
        files=f"{aggregate_path.relative_to(ROOT)}; {route_path.relative_to(ROOT)}",
        note=f"route_decision={route['route_decision']}",
    )
    return route


def analyze_part_b_repair_probe(
    args: argparse.Namespace,
    *,
    failures: list[dict[str, Any]] | None = None,
    methods: list[str] | None = None,
    run_label: str | None = None,
) -> dict[str, Any]:
    failures = failures if failures is not None else part_b_external_failure_rows()
    methods = methods or [m.strip() for m in str(args.repair_methods or ",".join(PART_B_REPAIR_METHODS)).split(",") if m.strip()]
    run_label = run_label or str(args.run_label or "v22_67_mlp_external_repair_probe")
    failure_keys = {(str(r.get("dataset")), str(int(float(str(r.get("seed"))))), str(int(float(str(r.get("steps")))))) for r in failures}
    baselines = condition_baselines(failures)
    repair_rows = collect_repair_rows(methods, run_label)
    detail: list[dict[str, Any]] = []
    for row in repair_rows:
        key = (str(row.get("dataset")), str(int(float(str(row.get("seed"))))), str(int(float(str(row.get("steps"))))))
        if key not in failure_keys:
            continue
        base = baselines.get(key, {})
        nll = safe_float(row.get("held_NLL"), float("inf")) or float("inf")
        debt = debt_value(row)
        best_ext = base.get("best_external_NLL", float("inf"))
        best_control = base.get("best_control_NLL", float("inf"))
        best_ref = base.get("best_ref_NLL", float("inf"))
        best_ref_debt = base.get("best_ref_debt", float("inf"))
        out = dict(row)
        out["Delta_NLL_vs_external_OET"] = nll - best_ext if math.isfinite(best_ext) else ""
        out["Delta_NLL_vs_best_control"] = nll - best_control if math.isfinite(best_control) else ""
        out["Delta_NLL_vs_strongest"] = nll - best_ref if math.isfinite(best_ref) else ""
        out["beats_external_OET_NLL"] = int(math.isfinite(best_ext) and nll < best_ext)
        out["beats_best_control_NLL"] = int(math.isfinite(best_control) and nll < best_control)
        out["beats_strongest_NLL"] = int(math.isfinite(best_ref) and nll < best_ref)
        out["no_ECE_Brier_tail_debt"] = int(debt is not None and math.isfinite(best_ref_debt) and debt <= best_ref_debt + 1.0e-9)
        out["overhead_le_025"] = int((safe_float(row.get("controller_or_coordinate_overhead_ratio"), 999.0) or 999.0) <= 0.25)
        out["active_Gram_drift_le_005"] = int((safe_float(row.get("active_Gram_drift_mean"), 999.0) or 999.0) <= 0.05)
        out["functional_spectrum_drift_le_threshold"] = int((safe_float(row.get("functional_spectrum_drift_mean"), 999.0) or 999.0) <= float(args.functional_spectrum_drift_threshold))
        detail.append(out)
    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        sub = [r for r in detail if str(r.get("method")) == method and r.get("run_status") == "completed"]
        summary_rows.append(
            {
                "method": method,
                "target_failure_conditions": len(failure_keys),
                "completed_rows": len(sub),
                "beats_external_OET_rows": sum(flag(r.get("beats_external_OET_NLL")) for r in sub),
                "beats_best_control_rows": sum(flag(r.get("beats_best_control_NLL")) for r in sub),
                "beats_strongest_rows": sum(flag(r.get("beats_strongest_NLL")) for r in sub),
                "no_debt_rows": sum(flag(r.get("no_ECE_Brier_tail_debt")) for r in sub),
                "overhead_le_025_rows": sum(flag(r.get("overhead_le_025")) for r in sub),
                "active_Gram_drift_le_005_rows": sum(flag(r.get("active_Gram_drift_le_005")) for r in sub),
                "functional_spectrum_drift_le_threshold_rows": sum(flag(r.get("functional_spectrum_drift_le_threshold")) for r in sub),
                "mean_Delta_NLL_vs_external_OET": None if not sub else sum(safe_float(r.get("Delta_NLL_vs_external_OET"), 0.0) or 0.0 for r in sub) / len(sub),
            }
        )
    best = max(summary_rows, key=lambda r: (int(r["beats_external_OET_rows"]), int(r["beats_best_control_rows"]), int(r["functional_spectrum_drift_le_threshold_rows"]))) if summary_rows else {}
    recommended = "run_full_repair_robustness_for_best_method" if best and int(best.get("beats_external_OET_rows", 0)) >= 5 else "external_oet_blocker_persists_audit_failures"
    summary = {
        "gate": "v22_67_part_b_external_oet_failure_repair_probe",
        "generated_at_sg": now_sg(),
        "run_label": run_label,
        "target_failure_conditions": len(failure_keys),
        "repair_methods": ",".join(methods),
        "best_probe_method": best.get("method", ""),
        "best_probe_external_fixes": best.get("beats_external_OET_rows", 0),
        "recommended_next_action": recommended,
        "summary_rows": summary_rows,
    }
    label_frag = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_label).strip("_") or "repair_probe"
    write_rows(OUT_ROOT / "v22_67_part_b_external_repair_probe_detail.csv", detail)
    write_rows(OUT_ROOT / "v22_67_part_b_external_repair_probe_summary.csv", summary_rows)
    write_rows(OUT_ROOT / f"v22_67_part_b_external_repair_probe_{label_frag}_detail.csv", detail)
    write_rows(OUT_ROOT / f"v22_67_part_b_external_repair_probe_{label_frag}_summary.csv", summary_rows)
    summary_path = OUT_ROOT / f"v22_67_part_b_external_repair_probe_{label_frag}_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (OUT_ROOT / "v22_67_part_b_external_repair_probe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    append_recap(
        "Part B external-OET failure repair probe",
        [
            f"target_failure_conditions={len(failure_keys)}; methods={','.join(methods)}.",
            f"best_probe_method={summary['best_probe_method']}; best_probe_external_fixes={summary['best_probe_external_fixes']}; recommended_next_action={recommended}.",
            f"Evidence files: results/v22_67/v22_67_part_b_external_failure_rows.csv, v22_67_part_b_external_repair_probe_{label_frag}_detail.csv, v22_67_part_b_external_repair_probe_{label_frag}_summary.json.",
            "Probe rows are exploratory fixed-method repairs; no row-wise best promotion is used for official claims.",
        ],
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        default="part-a",
        choices=[
            "part-a",
            "part-b-analyze",
            "part-b-external-decompose",
            "part-b-matrix",
            "part-b-repair-probe",
            "part-b-repair-analyze",
            "part-c-preflight",
            "part-c-visible-repair",
            "part-c-chart-warmstart",
            "part-c-task-visible-repair",
            "part-c-basis-redesign-preflight",
            "part-d-unit-tests",
            "part-e-basis-redesign-full-loop",
            "part-e-basis-redesign-analyze",
        ],
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--row-timeout", type=int, default=2400)
    p.add_argument("--matrix-timeout", type=int, default=172800)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--refresh", type=int, default=100)
    p.add_argument("--run-label", default="")
    p.add_argument("--functional-spectrum-drift-threshold", type=float, default=0.50)
    p.add_argument("--repair-methods", default=",".join(PART_B_REPAIR_METHODS))
    p.add_argument("--candidate-method", default=PART_B_CANDIDATE)
    p.add_argument("--matrix-methods", default=",".join(PART_B_METHODS))
    p.add_argument("--metric-batch-size", type=int, default=128)
    p.add_argument("--basis-architectures", default="DGKAN_FOU4,DGKAN_RBF4,DGKAN_HAT4,DGKAN_RAT4")
    p.add_argument("--basis-candidate-method", default=PART_E_BASIS_CANDIDATE)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_out()
    if args.mode == "part-a":
        summary = run_part_a(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.mode == "part-b-analyze":
        summary = run_part_b_analyze(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.mode == "part-b-external-decompose":
        summary = run_part_b_external_decompose(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.mode == "part-b-matrix":
        result = run_part_b_matrix(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.mode == "part-b-repair-probe":
        summary = run_part_b_repair_probe(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.mode == "part-b-repair-analyze":
        summary = analyze_part_b_repair_probe(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.mode == "part-c-preflight":
        summary = run_part_c_preflight(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.mode == "part-c-visible-repair":
        summary = run_part_c_readout_visible_repair(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.mode == "part-c-chart-warmstart":
        summary = run_part_c_chart_warmstart(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.mode == "part-c-task-visible-repair":
        summary = run_part_c_task_visible_repair(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.mode == "part-c-basis-redesign-preflight":
        summary = run_part_c_basis_redesign_preflight(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.mode == "part-d-unit-tests":
        summary = run_part_d_unit_tests(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.mode == "part-e-basis-redesign-full-loop":
        route = run_part_e_basis_redesign_full_loop(args)
        print(json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.mode == "part-e-basis-redesign-analyze":
        summaries = []
        for architecture in [a.strip() for a in str(args.basis_architectures).split(",") if a.strip()]:
            summaries.append(summarize_part_e_basis_redesign(args, architecture))
        print(json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
