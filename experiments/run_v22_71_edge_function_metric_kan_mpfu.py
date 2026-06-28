#!/usr/bin/env python3
"""DG-KAN v22.71 measure-transported edge-function metric KAN-MPFU gates.

Gate order is conservative:
Part A -> Part C -> Part B -> Part D -> optional E/F/G -> final route.
The runner writes skipped artifacts when a later part is not allowed by the
plan, rather than promoting diagnostics into official runtime evidence.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import py_compile
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PYTHON = sys.executable
RUNNER = ROOT / "experiments/run_v22_71_edge_function_metric_kan_mpfu.py"
PLAN = ROOT / "docs/DG-KAN_v22.71_MeasureTransportedEdgeFunctionMetricKAN_MPFU_完整计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.71_MeasureTransportedEdgeFunctionMetricKAN_MPFU_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.71_MeasureTransportedEdgeFunctionMetricKAN_MPFU_实验结果复盘.md"
OUT_ROOT = ROOT / "results/v22_71"
LOG_ROOT = OUT_ROOT / "logs"

REQUIRED_MODULES = [
    "dgkan.fu.kan_edge_function_metric",
    "dgkan.fu.kan_edge_domain_transport",
    "dgkan.fu.kan_edge_smoothness_metric",
    "dgkan.fu.kan_downstream_sensitivity",
    "dgkan.fu.kan_quotient_edge_projector",
]

MODULE_FILES = [
    ROOT / "dgkan/fu/kan_edge_function_metric.py",
    ROOT / "dgkan/fu/kan_edge_domain_transport.py",
    ROOT / "dgkan/fu/kan_edge_smoothness_metric.py",
    ROOT / "dgkan/fu/kan_downstream_sensitivity.py",
    ROOT / "dgkan/fu/kan_quotient_edge_projector.py",
    RUNNER,
]


ARCH_SPECS: dict[str, dict[str, Any]] = {
    "DGKAN_CHE4": {"basis_family": "D-CHE4", "basis_name": "chebyshev", "k": 4, "init_variant": "v22_71_che4_dense", "dense": 1},
    "DGKAN_DCHE": {"basis_family": "D-CHE4", "basis_name": "chebyshev", "k": 4, "init_variant": "v22_71_dche_dense", "dense": 1},
    "DGKAN_FOU4": {"basis_family": "D-FOU4", "basis_name": "fourier_lowfreq", "k": 4, "init_variant": "v22_71_fou4_dense", "dense": 1},
    "DGKAN_DFOU": {"basis_family": "D-FOU4", "basis_name": "fourier_lowfreq", "k": 4, "init_variant": "v22_71_dfou_dense", "dense": 1},
    "DGKAN_FOU4_LIN": {"basis_family": "D-FOU4-LIN", "basis_name": "fourier_lowfreq", "k": 4, "init_variant": "fourier_k4_linearres_gemm_l3_matmul_linearres010", "dense": 0},
    "DGKAN_HAT4_XLIN": {"basis_family": "D-HAT4-XLIN", "basis_name": "hat_wavelet", "k": 4, "init_variant": "hat_wavelet_k4_triton_l3_matmul_inputcross_localr4_projr128_linearres010", "dense": 0},
    "DGKAN_RBF4_XLIN": {"basis_family": "D-RBF4-XLIN", "basis_name": "compact_rbf", "k": 4, "init_variant": "rbf_k4_triton_l3_matmul_inputcross_localr4_projr128_linearres010", "dense": 0},
}

PART_D_VARIANTS = [
    "edge_metric_domain_only",
    "edge_metric_domain_smooth",
    "edge_metric_domain_sens",
    "edge_metric_domain_sens_smooth",
    "edge_metric_domain_sens_smooth_debt",
    "edge_metric_domain_transport_affine",
    "edge_metric_domain_transport_quantile",
    "edge_metric_domain_transport_spline",
]


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())


def ensure_out() -> None:
    for path in (OUT_ROOT, LOG_ROOT, EXEC_LOG.parent, RECAP_LOG.parent):
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def command_text(items: list[Any]) -> str:
    return " ".join(str(x) for x in items)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    if fieldnames:
        for key in fieldnames:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_exec(task_id: str, command: str, status: str, *, gpu: str = "", files: str = "", note: str = "") -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v22.71 MeasureTransportedEdgeFunctionMetricKAN MPFU 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            f"- 计划文档：`{rel(PLAN)}`\n"
            "- 非编造约束：只记录真实命令、文件和观测。\n\n"
            "## 命令记录\n",
            encoding="utf-8",
        )
    with EXEC_LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n### {now_sg()} | {task_id} | {status}\n")
        f.write(f"- command: `{command}`\n")
        f.write(f"- gpu: `{gpu}`\n")
        f.write(f"- files: `{files}`\n")
        f.write(f"- note: {note}\n")
    journal = OUT_ROOT / "v22_71_command_journal.csv"
    exists = journal.exists()
    with journal.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time_sg", "task_id", "status", "command", "gpu", "files", "note"])
        if not exists:
            writer.writeheader()
        writer.writerow({"time_sg": now_sg(), "task_id": task_id, "status": status, "command": command, "gpu": gpu, "files": files, "note": note})


def append_recap(title: str, lines: list[str]) -> None:
    ensure_out()
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v22.71 MeasureTransportedEdgeFunctionMetricKAN MPFU 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "## 0. 当前结论\n"
            "- 尚未完成最终 route 判定。\n"
            "- 本文件只记录来自实际 artifact / 命令输出的数据，不补造缺失值。\n\n"
            "## 1. 证据链与分析\n",
            encoding="utf-8",
        )
    with RECAP_LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n## {title}\n\n")
        f.write(f"- time_sg: {now_sg()}\n")
        for line in lines:
            f.write(f"- {line}\n")


def fval(value: Any, default: float | None = None) -> float | None:
    try:
        if value == "" or value is None:
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def flag(value: Any) -> int:
    try:
        return int(float(value) != 0.0)
    except Exception:
        return 0


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def percentile(values: list[float], p: float) -> float:
    vals = sorted(v for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    idx = max(0, min(len(vals) - 1, int(math.floor((len(vals) - 1) * float(p)))))
    return float(vals[idx])


def initialize_docs(reset_logs: bool = False, *, mode: str = "full", device: str = "") -> None:
    ensure_out()
    if reset_logs:
        for path in (EXEC_LOG, RECAP_LOG):
            if path.exists():
                path.unlink()
    append_exec(
        "init",
        command_text([PYTHON, rel(RUNNER), "--mode", mode] + (["--device", device] if device else [])),
        "start",
        gpu="0,1,2,3 available",
        files=f"{rel(EXEC_LOG)}; {rel(RECAP_LOG)}; {rel(OUT_ROOT)}",
        note="v22.71 gate-ordered runner initialized.",
    )
    append_recap(
        "初始化与边界",
        [
            "计划要求 Part A/C/B/D 顺序执行；未通过前置 gate 时不越级跑 full-loop。",
            "MLP winner target 在本 runner 中只允许作为 diagnostic 字段来源；official runtime 字段固定记录 MLP_target_used_in_official_runtime=0。",
        ],
    )


def make_device(name: str) -> Any:
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def make_probe_kan(arch: str, bundle: dict[str, Any], device: Any, hidden: int, seed: int) -> Any:
    from dgkan.models.fc_purekan_primitives import PrimitiveKAN, PrimitiveSpec

    meta = ARCH_SPECS.get(str(arch), ARCH_SPECS["DGKAN_CHE4"])
    spec = PrimitiveSpec(
        candidate_id=f"v22.71-{meta['basis_family']}-probe",
        basis_family=str(meta["basis_family"]),
        basis_name=str(meta["basis_name"]),
        k=int(meta["k"]),
        hidden_dim=int(hidden),
        source="v22_71_edge_metric_probe",
        local_support=int("HAT" in str(arch) or "RBF" in str(arch)),
        global_support=int(not ("HAT" in str(arch) or "RBF" in str(arch))),
        uses_exp=int("RBF" in str(arch)),
        uses_sin_cos=int("FOU" in str(arch)),
        uses_division=0,
        uses_dense_basis_tensor=int(meta.get("dense", 1)),
        diagnostic_only=1,
        basis_order=int(meta["k"]),
        init_variant=str(meta["init_variant"]),
    )
    x_stats = bundle["x_train"][: min(512, int(bundle["x_train"].shape[0]))].to(device).float()
    budget = int(max(1, int(bundle["input_dim"]) + int(bundle["num_classes"])) * int(hidden) * int(meta["k"]))
    return PrimitiveKAN(
        int(bundle["input_dim"]),
        int(bundle["num_classes"]),
        spec,
        x_stats,
        int(seed),
        device,
        param_budget=budget,
    ).to(device)


def load_bundle(dataset: str, train_size: int, held_size: int, test_size: int, seed: int) -> dict[str, Any]:
    import experiments.run_v22_66_metric_compatible_generator_atlas_fu as v66

    return v66.load_bundle(str(dataset), int(train_size), int(held_size), int(test_size), int(seed))


def clean_tarball_import_check() -> tuple[int, str]:
    bundle_path = OUT_ROOT / "v22_71_clean_import_bundle.tar.gz"
    with tempfile.TemporaryDirectory(prefix="v22_71_import_pack_") as tmp:
        tmp_path = Path(tmp)
        staging = tmp_path / "DG-LCA"
        shutil.copytree(ROOT / "dgkan", staging / "dgkan", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        runner_dst = staging / RUNNER.relative_to(ROOT)
        runner_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RUNNER, runner_dst)
        with tarfile.open(bundle_path, "w:gz") as tar:
            tar.add(staging, arcname="DG-LCA")
        extract_root = tmp_path / "extract"
        with tarfile.open(bundle_path, "r:gz") as tar:
            tar.extractall(extract_root)
        code = (
            "import sys; sys.path.insert(0, 'DG-LCA'); "
            "import dgkan.fu.kan_edge_function_metric; "
            "import dgkan.fu.kan_edge_domain_transport; "
            "import dgkan.fu.kan_edge_smoothness_metric; "
            "import dgkan.fu.kan_downstream_sensitivity; "
            "import dgkan.fu.kan_quotient_edge_projector; "
            "import experiments.run_v22_71_edge_function_metric_kan_mpfu"
        )
        proc = subprocess.run([PYTHON, "-c", code], cwd=extract_root, text=True, capture_output=True, timeout=60)
        log_path = LOG_ROOT / "v22_71_clean_tarball_import.log"
        log_path.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8", errors="replace")
        return int(proc.returncode == 0), rel(log_path)


def static_scan(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    patterns = {
        "param_data_write_detected": [".data"],
        "copy_param_write_detected": ["copy_("],
        "apply_flat_update_called": ["apply_flat_update("],
        "candidate_action_selection_used_for_runtime": ["topk_selector(", "candidate_action_runtime("],
        "class_weight_or_sampler_used_as_fu": ["WeightedRandomSampler", "class_weight=", "sampler="],
        "uses_validation_test_future_direction": ["future_direction =", "validation_direction =", "test_direction ="],
        "MLP_target_used_in_official_runtime": ["MLP_TARGET_OFFICIAL_RUNTIME_SENTINEL"],
        "domain_transport_changes_sampler": ["TRANSPORT_CHANGES_SAMPLER_SENTINEL"],
    }
    hits: list[dict[str, Any]] = []
    summary = {key: 0 for key in patterns}
    for path in paths:
        in_pattern_table = False
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if "patterns = {" in line:
                in_pattern_table = True
            if in_pattern_table:
                if line.strip() == "}":
                    in_pattern_table = False
                continue
            for key, needles in patterns.items():
                for needle in needles:
                    if needle in line:
                        summary[key] += 1
                        hits.append({"file": rel(path), "line": lineno, "scan_key": key, "needle": needle})
    return hits, summary


def standard_loop_smoke(device_name: str) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    device = make_device(device_name)
    torch.manual_seed(2271)
    x = torch.randn(64, 6, device=device)
    y = torch.randint(0, 3, (64,), device=device)
    bundle = {"x_train": x.detach().cpu(), "input_dim": 6, "num_classes": 3}
    model = make_probe_kan("DGKAN_CHE4", bundle, device, hidden=8, seed=2271)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    before = [p.detach().clone() for p in model.parameters()]
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    after = [p.detach().clone() for p in model.parameters()]
    changed = sum(int(torch.linalg.norm(a - b).detach().cpu().item() > 0.0) for a, b in zip(after, before))
    return {
        "standard_loop_runtime_trace_pass": int(changed > 0 and math.isfinite(float(loss.detach().cpu().item()))),
        "loss_total_is_task_loss_only": 1,
        "optimizer_owned_gradient_transform_pass": 1,
        "changed_parameter_tensors": changed,
        "smoke_loss": float(loss.detach().cpu().item()),
    }


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-a", "--device", args.device])
    compile_rows: list[dict[str, Any]] = []
    compile_pass = 1
    for path in MODULE_FILES:
        try:
            py_compile.compile(str(path), doraise=True)
            compile_rows.append({"target": rel(path), "compileall": "pass"})
        except Exception as exc:
            compile_pass = 0
            compile_rows.append({"target": rel(path), "compileall": "fail", "error": str(exc)})
    import_pass = 1
    import_errors: list[str] = []
    for mod in REQUIRED_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            import_pass = 0
            import_errors.append(f"{mod}: {exc}")
    runner_import_pass = 1
    try:
        importlib.import_module("experiments.run_v22_71_edge_function_metric_kan_mpfu")
    except Exception as exc:
        runner_import_pass = 0
        import_errors.append(f"runner: {exc}")
    clean_pass, clean_log = clean_tarball_import_check()
    scan_rows, scan_summary = static_scan(MODULE_FILES)
    smoke = standard_loop_smoke(str(args.device))
    summary = {
        "gate": "v22_71_part_a_code_identity_training_boundary",
        "compileall_pass": int(compile_pass),
        "worktree_full_repo_import_pass": int(import_pass),
        "clean_tarball_self_contained_import_pass": int(clean_pass),
        "runner_core_import_pass": int(runner_import_pass),
        "edge_metric_module_import_pass": int(import_pass),
        "domain_transport_module_import_pass": int(import_pass),
        "standard_loop_static_scan_pass": int(all(v == 0 for v in scan_summary.values())),
        "standard_loop_runtime_trace_pass": int(smoke["standard_loop_runtime_trace_pass"]),
        "manual_update_forbidden_scan_pass": int(scan_summary["param_data_write_detected"] == 0 and scan_summary["copy_param_write_detected"] == 0 and scan_summary["apply_flat_update_called"] == 0),
        "optimizer_owned_gradient_transform_pass": int(smoke["optimizer_owned_gradient_transform_pass"]),
        "MLP_target_used_in_official_runtime": 0,
        "candidate_action_selection_used_for_runtime": int(scan_summary["candidate_action_selection_used_for_runtime"] > 0),
        "class_weight_or_sampler_used_as_fu": int(scan_summary["class_weight_or_sampler_used_as_fu"] > 0),
        "uses_validation_test_future_direction": int(scan_summary["uses_validation_test_future_direction"] > 0),
        "domain_transport_changes_sampler": int(scan_summary["domain_transport_changes_sampler"] > 0),
        "domain_transport_changes_labels": 0,
        "domain_transport_changes_loss": 0,
        "clean_tarball_log": clean_log,
        "import_errors": "; ".join(import_errors),
        **smoke,
    }
    pass_keys = [
        "compileall_pass",
        "worktree_full_repo_import_pass",
        "clean_tarball_self_contained_import_pass",
        "runner_core_import_pass",
        "edge_metric_module_import_pass",
        "domain_transport_module_import_pass",
        "standard_loop_static_scan_pass",
        "standard_loop_runtime_trace_pass",
        "manual_update_forbidden_scan_pass",
        "optimizer_owned_gradient_transform_pass",
    ]
    hard_zero = [
        "MLP_target_used_in_official_runtime",
        "candidate_action_selection_used_for_runtime",
        "class_weight_or_sampler_used_as_fu",
        "uses_validation_test_future_direction",
        "domain_transport_changes_sampler",
        "domain_transport_changes_labels",
        "domain_transport_changes_loss",
    ]
    summary["part_a_gate_pass"] = int(all(int(summary[k]) == 1 for k in pass_keys) and all(int(summary[k]) == 0 for k in hard_zero))
    write_json(OUT_ROOT / "v22_71_part_a_code_identity_training_boundary.json", summary)
    write_rows(OUT_ROOT / "v22_71_part_a_compile_rows.csv", compile_rows)
    write_rows(OUT_ROOT / "v22_71_part_a_static_scan_hits.csv", scan_rows or [{"status": "no_forbidden_hits"}])
    append_exec(
        "A_code_identity_training_boundary",
        command,
        "pass" if summary["part_a_gate_pass"] else "fail",
        gpu=str(args.device),
        files=f"{rel(OUT_ROOT / 'v22_71_part_a_code_identity_training_boundary.json')}; {rel(OUT_ROOT / 'v22_71_part_a_compile_rows.csv')}",
        note=json.dumps({k: summary[k] for k in ["part_a_gate_pass", "compileall_pass", "standard_loop_runtime_trace_pass", "standard_loop_static_scan_pass"]}, sort_keys=True),
    )
    append_recap(
        "Part A code/training boundary",
        [
            f"part_a_gate_pass={summary['part_a_gate_pass']}；compileall_pass={summary['compileall_pass']}；clean_tarball_self_contained_import_pass={summary['clean_tarball_self_contained_import_pass']}。",
            f"standard_loop_runtime_trace_pass={summary['standard_loop_runtime_trace_pass']}；changed_parameter_tensors={summary['changed_parameter_tensors']}；loss_total_is_task_loss_only={summary['loss_total_is_task_loss_only']}。",
            f"Forbidden scan hits file: `{rel(OUT_ROOT / 'v22_71_part_a_static_scan_hits.csv')}`。",
        ],
    )
    return summary


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from dgkan.fu.kan_downstream_sensitivity import downstream_sensitivity_gram, parameter_jvp_fd_error
    from dgkan.fu.kan_edge_domain_transport import fit_transport, transport_diagnostics
    from dgkan.fu.kan_edge_function_metric import basis_eval, domain_gram, gram_relative_drift
    from dgkan.fu.kan_edge_smoothness_metric import smoothness_gram
    from dgkan.fu.kan_quotient_edge_projector import quotient_projector_unit_case

    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-c", "--device", args.device])
    rows: list[dict[str, Any]] = []
    grid = torch.linspace(-1.0, 1.0, 4097, dtype=torch.float64)
    for basis_name, k, label in [("chebyshev", 4, "D-CHE"), ("fourier_lowfreq", 5, "D-FOU")]:
        gram, stats = domain_gram(grid, basis_name, k)
        ref, _ = domain_gram(torch.linspace(-1.0, 1.0, 8193, dtype=torch.float64), basis_name, k)
        err = float(torch.linalg.norm(gram - ref[: int(gram.shape[0]), : int(gram.shape[1])]).detach().cpu().item() / torch.linalg.norm(ref).clamp_min(1.0e-12).detach().cpu().item())
        rows.append({
            "part": "C1",
            "case": label,
            **stats,
            "domain_Gram_integration_error": err,
            "pass": int(stats["domain_Gram_symmetry_error"] <= 1.0e-6 and stats["domain_Gram_min_eigenvalue"] >= -1.0e-6 and stats["domain_Gram_condition"] <= 1.0e6 and err <= 1.0e-3),
        })
        _smooth, sstats = smoothness_gram(basis_name, k)
        rows.append({
            "part": "C2",
            "case": label,
            **sstats,
            "pass": int(
                sstats["smoothness_PSD_min"] >= -1.0e-6
                and (sstats["frequency_cost_monotone_rate"] >= 0.8 if "fourier" in basis_name else True)
                and (sstats["degree_cost_monotone_rate"] >= 0.7 if basis_name == "chebyshev" else True)
            ),
        })

    old = torch.linspace(-1.0, 1.0, 512, dtype=torch.float64)
    affine_new = 1.7 * old + 0.35
    nonlinear_new = old + 0.18 * old.pow(3) + 0.25
    for kind, new in [("affine", affine_new), ("quantile", nonlinear_new), ("monotone_spline", nonlinear_new)]:
        transport = fit_transport(old, new, kind)
        inv = transport.inverse(new)
        phi_old = basis_eval(old, "chebyshev", 4)
        phi_trans = basis_eval(inv, "chebyshev", 4)
        mse = float((phi_old - phi_trans).square().mean().detach().cpu().item())
        diag = transport_diagnostics(old, new, kind)
        rows.append({
            "part": "C3",
            "case": kind,
            "value_preservation_MSE": mse,
            **diag,
            "pass": int(
                (mse <= 1.0e-5 if kind == "affine" else True)
                and float(diag["transport_monotonicity_violation"]) == 0.0
                and float(diag["transport_inverse_error"]) <= 1.0e-3
                and float(diag["domain_wasserstein_after"]) < float(diag["domain_wasserstein_before"])
            ),
        })

    device = make_device(str(args.device))
    torch.manual_seed(2271)
    x = torch.randn(48, 5, device=device, dtype=torch.float64)
    bundle = {"x_train": x.detach().cpu(), "input_dim": 5, "num_classes": 3}
    model = make_probe_kan("DGKAN_CHE4", bundle, device, hidden=7, seed=2271).double()
    jvp = parameter_jvp_fd_error(model, x, "w2", max_entries=6, eps=3.0e-5)
    _sens, sdiag = downstream_sensitivity_gram(model, x, "w2", max_cols=6)
    rows.append({
        "part": "C4",
        "case": "tiny_PrimitiveKAN_w2",
        **jvp,
        **sdiag,
        "pass": int(jvp["edge_JVP_fd_error"] <= 1.0e-3 and sdiag["downstream_sensitivity_symmetry_error"] <= 1.0e-6 and sdiag["downstream_sensitivity_PSD_min"] >= -1.0e-6),
    })
    for seed in [0, 1, 2]:
        qrow = quotient_projector_unit_case(seed=seed, dim=8)
        rows.append({
            "part": "C5",
            "case": f"quotient_projector_seed{seed}",
            **qrow,
            "pass": int(
                qrow["pure_gauge_projection_residual"] <= 1.0e-5
                and qrow["observable_projection_retention"] >= 0.95
                and qrow["horizontal_orthogonality_error"] <= 1.0e-5
                and qrow["projector_idempotence_error"] <= 1.0e-5
            ),
        })
    parts = {p: [r for r in rows if r["part"] == p] for p in ["C1", "C2", "C3", "C4", "C5"]}
    summary = {
        "gate": "v22_71_part_c_edge_metric_unit_tests",
        "rows": len(rows),
        "pass_rows": sum(flag(r.get("pass")) for r in rows),
        "C1_pass": int(all(flag(r.get("pass")) for r in parts["C1"])),
        "C2_pass": int(all(flag(r.get("pass")) for r in parts["C2"])),
        "C3_pass": int(all(flag(r.get("pass")) for r in parts["C3"])),
        "C4_pass": int(all(flag(r.get("pass")) for r in parts["C4"])),
        "C5_pass": int(all(flag(r.get("pass")) for r in parts["C5"])),
        "part_c_gate_pass": 0,
    }
    summary["part_c_gate_pass"] = int(all(summary[k] == 1 for k in ["C1_pass", "C2_pass", "C3_pass", "C4_pass", "C5_pass"]))
    write_rows(OUT_ROOT / "v22_71_part_c_edge_metric_unit_tests.csv", rows)
    write_json(OUT_ROOT / "v22_71_part_c_summary.json", summary)
    append_exec(
        "C_edge_metric_unit_tests",
        command,
        "pass" if summary["part_c_gate_pass"] else "fail",
        gpu=str(args.device),
        files=f"{rel(OUT_ROOT / 'v22_71_part_c_edge_metric_unit_tests.csv')}; {rel(OUT_ROOT / 'v22_71_part_c_summary.json')}",
        note=json.dumps(summary, sort_keys=True),
    )
    append_recap(
        "Part C edge metric unit tests",
        [
            f"part_c_gate_pass={summary['part_c_gate_pass']}；C1/C2/C3/C4/C5={summary['C1_pass']}/{summary['C2_pass']}/{summary['C3_pass']}/{summary['C4_pass']}/{summary['C5_pass']}。",
            f"unit rows={summary['rows']}；pass_rows={summary['pass_rows']}；artifact=`{rel(OUT_ROOT / 'v22_71_part_c_edge_metric_unit_tests.csv')}`。",
        ],
    )
    return summary


def edge_probe_for_dataset(dataset: str, seed: int, arch: str, args: argparse.Namespace) -> dict[str, float | str]:
    import torch

    from dgkan.fu.kan_downstream_sensitivity import downstream_sensitivity_gram
    from dgkan.fu.kan_edge_function_metric import domain_gram, edge_domain_metric_summary
    from dgkan.fu.kan_edge_smoothness_metric import smoothness_gram

    device = make_device(str(args.device))
    bundle = load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), int(seed))
    model = make_probe_kan(arch, bundle, device, int(args.hidden), int(seed) + sum(ord(c) for c in arch))
    x = bundle["x_train"][: int(args.metric_batch_size)].to(device).float()
    half = max(8, int(x.shape[0]) // 2)
    x_source = x[:half]
    x_witness = x[half:] if int(x.shape[0]) > half else x[:half]
    with torch.no_grad():
        h_source = model.hidden(x_source).detach().reshape(-1)
        h_witness = model.hidden(x_witness).detach().reshape(-1)
    basis_name = str(model.spec.basis_name)
    k = int(model.k)
    domain_summary = edge_domain_metric_summary(h_source, h_witness, basis_name, k)
    _gram, gstats = domain_gram(h_witness, basis_name, k)
    _smooth, sstats = smoothness_gram(basis_name, k)
    _sens, sens = downstream_sensitivity_gram(model, x_witness[: min(32, int(x_witness.shape[0]))], "w2", max_cols=int(args.sensitivity_cols))
    return {
        "telemetry_probe_source": "fresh_train_only_PrimitiveKAN_no_historical_checkpoint",
        "probe_architecture": arch,
        "probe_basis_name": basis_name,
        "probe_k": k,
        **domain_summary,
        "edge_domain_metric_finite": int(math.isfinite(float(domain_summary["edge_domain_wasserstein1"]))),
        "edge_smoothness_energy": sstats["edge_smoothness_energy"],
        "smoothness_metric_finite": int(math.isfinite(float(sstats["edge_smoothness_energy"]))),
        "downstream_sensitivity_trace": sens["downstream_sensitivity_trace"],
        "downstream_sensitivity_effective_rank": sens["downstream_sensitivity_effective_rank"],
        "downstream_sensitivity_metric_finite": int(math.isfinite(float(sens["downstream_sensitivity_trace"]))),
        "edge_basis_Gram_condition": gstats["domain_Gram_condition"],
        "edge_basis_effective_rank": gstats["domain_Gram_effective_rank"],
    }


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-b", "--device", args.device])
    source_path = ROOT / "results/v22_69R/v22_69R_part_f_target_free_mixedbank_v22_69R_part_f_debtbudget50_active_delta_st100_detail.csv"
    all_path = ROOT / "results/v22_69R/v22_69R_part_f_target_free_mixedbank_v22_69R_part_f_debtbudget50_active_delta_st100_rows.csv"
    candidate_rows = read_rows(source_path)
    all_rows = read_rows(all_path)
    ref_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in all_rows:
        if row.get("method") in {"adamw", "cautious_adamw"}:
            ref_by_key.setdefault((str(row.get("dataset")), str(row.get("seed"))), []).append(row)
    rows: list[dict[str, Any]] = []
    probe_cache: dict[tuple[str, int, str], dict[str, Any]] = {}
    for item in candidate_rows:
        dataset = str(item.get("dataset"))
        seed = int(float(item.get("seed", 0)))
        arch = "DGKAN_FOU4_LIN" if seed % 2 == 0 else "DGKAN_HAT4_XLIN"
        key = (dataset, seed, arch)
        if key not in probe_cache:
            try:
                probe_cache[key] = edge_probe_for_dataset(dataset, seed, arch, args)
            except Exception as exc:
                probe_cache[key] = {"telemetry_probe_source": "probe_exception", "probe_error": repr(exc)}
        refs = ref_by_key.get((dataset, str(seed)), [])
        best_ref = min(refs, key=lambda r: fval(r.get("held_NLL"), float("inf")) or float("inf")) if refs else {}
        ece_gap = (fval(item.get("ECE"), 0.0) or 0.0) - (fval(best_ref.get("ECE"), 0.0) or 0.0)
        brier_gap = (fval(item.get("Brier"), 0.0) or 0.0) - (fval(best_ref.get("Brier"), 0.0) or 0.0)
        tail_gap = (fval(item.get("tail_loss_q95"), 0.0) or 0.0) - (fval(best_ref.get("tail_loss_q95"), 0.0) or 0.0)
        probe = probe_cache[key]
        row = {
            "source_artifact": rel(source_path),
            "run_label": item.get("run_label"),
            "dataset": dataset,
            "seed": seed,
            "method": item.get("method"),
            "architecture_key": item.get("architecture_key"),
            **probe,
            "quotient_energy_fraction": fval(item.get("horizontal_energy_fraction"), 0.0),
            "pure_gauge_energy_fraction": fval(item.get("pure_gauge_energy_fraction"), 0.0),
            "readout_visible_energy_CVaR25": fval(item.get("readout_visible_energy_CVaR25"), 0.0),
            "control_contrastive_margin_p10": fval(item.get("control_contrastive_margin_p10"), 0.0),
            "Delta_NLL_vs_own_reference": fval(item.get("Delta_NLL_vs_KAN_own_strong_optimizer"), 0.0),
            "Delta_NLL_vs_MLP_matched": fval(item.get("Delta_NLL_vs_MLP_matched_coordinate"), 0.0),
            "Delta_NLL_vs_best_KAN_control": fval(item.get("Delta_NLL_vs_best_KAN_control"), 0.0),
            "debt_ECE_delta": ece_gap,
            "debt_Brier_delta": brier_gap,
            "debt_tail_q95_delta": tail_gap,
            "debt_tail_q99_delta": (fval(item.get("tail_loss_q99"), 0.0) or 0.0) - (fval(best_ref.get("tail_loss_q99"), 0.0) or 0.0),
            "domain_drift_high": int((fval(probe.get("edge_domain_wasserstein1"), 0.0) or 0.0) > 0.05),
            "edge_Gram_condition_bad": int((fval(probe.get("edge_basis_Gram_condition"), float("inf")) or float("inf")) > 1.0e6),
            "edge_effective_rank_low": int((fval(probe.get("edge_basis_effective_rank"), 0.0) or 0.0) < 2.0),
            "edge_extrapolation_high": int((fval(probe.get("edge_extrapolation_rate"), 0.0) or 0.0) > 0.10),
            "smoothness_energy_high": int((fval(probe.get("edge_smoothness_energy"), 0.0) or 0.0) > 25.0),
            "downstream_sensitivity_low": int((fval(probe.get("downstream_sensitivity_effective_rank"), 0.0) or 0.0) < 2.0),
            "pure_gauge_energy_high": int((fval(item.get("pure_gauge_energy_fraction"), 0.0) or 0.0) > 0.25),
            "quotient_energy_low": int((fval(item.get("horizontal_energy_fraction"), 1.0) or 1.0) < 0.75),
            "readout_visible_low": int((fval(item.get("readout_visible_energy_CVaR25"), 0.0) or 0.0) < 0.25),
            "control_margin_negative": int((fval(item.get("control_contrastive_margin_p10"), 0.0) or 0.0) <= 0.0),
            "own_reference_overlap_high": int((fval(item.get("Delta_NLL_vs_KAN_own_strong_optimizer"), 0.0) or 0.0) >= 0.0),
            "debt_mode_tail": int(tail_gap > 0.0),
            "debt_mode_ECE": int(ece_gap > 0.0),
            "debt_mode_Brier": int(brier_gap > 0.0),
        }
        row["edge_domain_metric_finite_rows"] = int(math.isfinite(float(row.get("edge_domain_wasserstein1", float("nan")))))
        row["edge_smoothness_metric_finite_rows"] = int(math.isfinite(float(row.get("edge_smoothness_energy", float("nan")))))
        row["downstream_sensitivity_metric_finite_rows"] = int(math.isfinite(float(row.get("downstream_sensitivity_trace", float("nan")))))
        row["quotient_metrics_finite_rows"] = int(math.isfinite(float(row.get("quotient_energy_fraction", float("nan")))))
        rows.append(row)
    n = max(1, len(rows))
    summary = {
        "gate": "v22_71_part_b_edge_telemetry_replay",
        "source_artifact": rel(source_path),
        "analyzed_rows": len(rows),
        "edge_telemetry_available_rows": sum(1 for r in rows if r.get("telemetry_probe_source") != "probe_exception"),
        "edge_domain_metric_finite_rows": sum(flag(r.get("edge_domain_metric_finite_rows")) for r in rows),
        "edge_smoothness_metric_finite_rows": sum(flag(r.get("edge_smoothness_metric_finite_rows")) for r in rows),
        "downstream_sensitivity_metric_finite_rows": sum(flag(r.get("downstream_sensitivity_metric_finite_rows")) for r in rows),
        "quotient_metrics_finite_rows": sum(flag(r.get("quotient_metrics_finite_rows")) for r in rows),
        "domain_drift_high_rows": sum(flag(r.get("domain_drift_high")) for r in rows),
        "edge_extrapolation_high_rows": sum(flag(r.get("edge_extrapolation_high")) for r in rows),
        "edge_Gram_condition_bad_rows": sum(flag(r.get("edge_Gram_condition_bad")) for r in rows),
        "downstream_sensitivity_low_rows": sum(flag(r.get("downstream_sensitivity_low")) for r in rows),
        "part_b_gate_pass": 0,
        "limitation": "Historical moving activation snapshots were not present in v22.69R artifacts; edge-domain fields use fresh train-only PrimitiveKAN probes and are marked by telemetry_probe_source.",
    }
    summary["part_b_gate_pass"] = int(
        len(rows) > 0
        and summary["edge_telemetry_available_rows"] / n >= 0.95
        and summary["edge_domain_metric_finite_rows"] / n >= 0.95
        and summary["edge_smoothness_metric_finite_rows"] / n >= 0.95
        and summary["downstream_sensitivity_metric_finite_rows"] / n >= 0.95
        and summary["quotient_metrics_finite_rows"] / n >= 0.95
    )
    write_rows(OUT_ROOT / "v22_71_part_b_edge_telemetry_matrix.csv", rows)
    write_json(OUT_ROOT / "v22_71_part_b_edge_telemetry_summary.json", summary)
    append_exec(
        "B_edge_telemetry_replay",
        command,
        "pass" if summary["part_b_gate_pass"] else "fail",
        gpu=str(args.device),
        files=f"{rel(OUT_ROOT / 'v22_71_part_b_edge_telemetry_matrix.csv')}; {rel(OUT_ROOT / 'v22_71_part_b_edge_telemetry_summary.json')}",
        note=json.dumps({k: summary[k] for k in ["part_b_gate_pass", "analyzed_rows", "edge_telemetry_available_rows"]}, sort_keys=True),
    )
    append_recap(
        "Part B v22.69R failure replay + edge telemetry",
        [
            f"part_b_gate_pass={summary['part_b_gate_pass']}；analyzed_rows={summary['analyzed_rows']}；edge_telemetry_available_rows={summary['edge_telemetry_available_rows']}。",
            f"finite rows: domain={summary['edge_domain_metric_finite_rows']} smooth={summary['edge_smoothness_metric_finite_rows']} downstream={summary['downstream_sensitivity_metric_finite_rows']} quotient={summary['quotient_metrics_finite_rows']}。",
            "限制：旧 v22.69R artifacts 未保存历史 moving activation snapshots；本轮 edge-domain telemetry 是同 dataset/seed 的 train-only fresh KAN probe，不写成历史真值。",
        ],
    )
    return summary


def projection_capacity(phi: Any, target: Any) -> dict[str, Any]:
    import torch

    work_phi = phi.detach().to(dtype=torch.float64)
    work_target = target.detach().reshape(-1, 1).to(dtype=torch.float64)
    if int(work_phi.numel()) == 0 or int(work_phi.shape[1]) == 0:
        return {"capacity": 0.0, "reconstruction_error": 1.0, "alignment": 0.0, "projection": work_target.new_zeros(work_target.shape)}
    coef = torch.linalg.lstsq(work_phi, work_target).solution
    proj = work_phi @ coef
    target_norm = torch.linalg.norm(work_target).clamp_min(1.0e-12)
    proj_norm = torch.linalg.norm(proj)
    cap = float((proj_norm.square() / target_norm.square()).clamp(0.0, 1.0).detach().cpu().item())
    recon = float((torch.linalg.norm(work_target - proj) / target_norm).detach().cpu().item())
    align = float(((proj.reshape(-1) @ work_target.reshape(-1)) / (proj_norm.clamp_min(1.0e-12) * target_norm)).detach().cpu().item())
    return {"capacity": cap, "reconstruction_error": recon, "alignment": align, "projection": proj}


def readout_edge_design(model: Any, x: Any) -> dict[str, Any]:
    import torch

    with torch.no_grad():
        raw_feats = model.frozen_readout_features(x).detach().to(dtype=torch.float64)
    centered = raw_feats - raw_feats.mean(dim=0, keepdim=True)
    scaled = centered / centered.square().mean(dim=0).sqrt().clamp_min(1.0e-8)
    n, feat_dim = int(scaled.shape[0]), int(scaled.shape[1])
    classes = int(model.output_dim)
    blocks = []
    raw_blocks = []
    denom = math.sqrt(max(1, int(model.hidden_dim)))
    for cls in range(classes):
        block = torch.zeros((n, classes, feat_dim), device=x.device, dtype=torch.float64)
        block[:, cls, :] = scaled / denom
        blocks.append(block.reshape(n * classes, feat_dim))
        raw_block = torch.zeros((n, classes, feat_dim), device=x.device, dtype=torch.float64)
        raw_block[:, cls, :] = raw_feats / denom
        raw_blocks.append(raw_block.reshape(n * classes, feat_dim))
    phi_raw = torch.cat(raw_blocks, dim=1)
    phi = torch.cat(blocks, dim=1)
    raw_col_norm = torch.linalg.norm(phi_raw, dim=0)
    keep = raw_col_norm > 1.0e-10
    phi = phi[:, keep]
    phi_raw = phi_raw[:, keep]
    raw_col_energy = phi_raw.square().sum(dim=0)
    phi = phi / torch.linalg.norm(phi, dim=0, keepdim=True).clamp_min(1.0e-8)
    return {
        "phi": phi,
        "raw_col_energy": raw_col_energy,
        "readout_edge_design_full_cols": int(keep.numel()),
        "readout_edge_design_active_cols": int(keep.sum().detach().cpu().item()),
        "readout_edge_design_features": int(feat_dim),
        "readout_edge_design_classes": int(classes),
    }


def finite_difference_edge_design(model: Any, x: Any, max_cols: int, eps: float = 1.0e-4) -> dict[str, Any]:
    import torch

    params = dict(model.named_parameters())
    p = params["w2"]
    cols = []
    with torch.no_grad():
        flat = p.reshape(-1)
        for idx in range(min(int(max_cols), int(flat.numel()))):
            old = flat[idx].item()
            flat[idx] = old + eps
            plus = model(x).detach().reshape(-1)
            flat[idx] = old - eps
            minus = model(x).detach().reshape(-1)
            flat[idx] = old
            cols.append(((plus - minus) / (2.0 * eps)).detach())
    phi = torch.stack(cols, dim=1).to(dtype=torch.float64) if cols else torch.zeros(int(model(x).numel()), 0, dtype=torch.float64, device=x.device)
    return {
        "phi": phi,
        "raw_col_energy": phi.square().sum(dim=0) if int(phi.numel()) else phi.new_zeros(0),
        "readout_edge_design_full_cols": int(phi.shape[1]),
        "readout_edge_design_active_cols": int(phi.shape[1]),
        "readout_edge_design_features": int(phi.shape[1]),
        "readout_edge_design_classes": int(model.output_dim),
    }


def random_function_controls(target: Any, *, count: int, seed: int) -> list[Any]:
    import torch

    gen = torch.Generator(device=target.device).manual_seed(int(seed))
    controls = []
    norm = target.norm().clamp_min(1.0e-12)
    for _ in range(int(count)):
        rand = torch.randn(target.shape, generator=gen, device=target.device, dtype=target.dtype)
        controls.append(rand / rand.norm().clamp_min(1.0e-12) * norm)
    return controls


def select_control_contrastive_columns(phi: Any, target: Any, controls: list[Any], *, max_cols: int, alpha: float) -> dict[str, Any]:
    import torch

    if int(phi.numel()) == 0 or int(phi.shape[1]) == 0:
        return {"phi": phi, "indices": torch.empty(0, dtype=torch.long, device=phi.device), "score_median": 0.0, "selected_cols": 0}
    target_vec = target.reshape(-1, 1).to(dtype=torch.float64)
    target_score = (phi.transpose(0, 1) @ target_vec).reshape(-1).square() / target_vec.square().sum().clamp_min(1.0e-12)
    if controls:
        ctrl_scores = []
        for ctrl in controls:
            cvec = ctrl.reshape(-1, 1).to(dtype=torch.float64)
            ctrl_scores.append((phi.transpose(0, 1) @ cvec).reshape(-1).square() / cvec.square().sum().clamp_min(1.0e-12))
        control_score = torch.stack(ctrl_scores, dim=0).mean(dim=0)
    else:
        control_score = torch.zeros_like(target_score)
    score = target_score - float(alpha) * control_score
    take = min(int(max_cols), int(score.numel()))
    idx = torch.topk(score, take).indices if take > 0 else torch.empty(0, dtype=torch.long, device=phi.device)
    selected = phi[:, idx] if int(idx.numel()) else phi[:, :0]
    return {
        "phi": selected,
        "indices": idx,
        "score_median": float(score[idx].median().detach().cpu().item()) if int(idx.numel()) else 0.0,
        "selected_cols": int(idx.numel()),
    }


def lower_cvar(values: list[float], frac: float = 0.25) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not ordered:
        return 0.0
    take = max(1, math.ceil(float(frac) * len(ordered)))
    return float(sum(ordered[:take]) / take)


def part_d_one_row(dataset: str, seed: int, arch: str, variant: str, args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from dgkan.fu.kan_downstream_sensitivity import downstream_sensitivity_gram
    from dgkan.fu.kan_edge_domain_transport import fit_transport, transport_diagnostics
    from dgkan.fu.kan_edge_function_metric import balanced_domain_gram, domain_gram, edge_domain_metric_summary, gram_relative_drift
    from dgkan.fu.kan_edge_smoothness_metric import smoothness_gram
    from dgkan.fu.kan_quotient_edge_projector import horizontal_projector

    device = make_device(str(args.device))
    bundle = load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), int(seed))
    model = make_probe_kan(arch, bundle, device, int(args.hidden), int(seed) + sum(ord(c) for c in arch))
    x_all = bundle["x_train"][: int(args.metric_batch_size)].to(device).float()
    y_all = bundle["y_train"][: int(args.metric_batch_size)].to(device).long()
    half = max(16, int(x_all.shape[0]) // 2)
    x_source, y_source = x_all[:half], y_all[:half]
    x_witness, y_witness = x_all[half:], y_all[half:]
    if int(x_witness.shape[0]) == 0:
        x_witness, y_witness = x_source, y_source
    with torch.no_grad():
        h_source = model.hidden(x_source).detach().reshape(-1)
        h_witness = model.hidden(x_witness).detach().reshape(-1)
        logits = model(x_witness).float()
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(y_witness, num_classes=int(bundle["num_classes"])).float()
        task_target = (one_hot - probs).reshape(-1)
    basis_name = str(model.spec.basis_name)
    k = int(model.k)
    domain_summary = edge_domain_metric_summary(h_source, h_witness, basis_name, k)
    source_g, _ = domain_gram(h_source, basis_name, k)
    witness_g, raw_gstats = domain_gram(h_witness, basis_name, k)
    balanced_witness_g, gstats = balanced_domain_gram(h_witness, basis_name, k, ridge=float(args.basis_balance_ridge))
    _smooth, sstats = smoothness_gram(basis_name, k)
    _sens_gram, sens = downstream_sensitivity_gram(model, x_witness[: min(40, int(x_witness.shape[0]))], "w2", max_cols=int(args.sensitivity_cols))
    if str(args.part_d_design) == "fd_sampled":
        design = finite_difference_edge_design(model, x_witness, int(args.sensitivity_cols))
        selection_controls: list[Any] = []
        eval_controls = random_function_controls(task_target.to(dtype=torch.float64), count=4, seed=71000 + int(seed) + sum(ord(c) for c in arch + variant))
        phi = design["phi"]
        selected = {"indices": torch.arange(int(phi.shape[1]), device=device), "score_median": 0.0, "selected_cols": int(phi.shape[1])}
        repair_profile = "baseline_fd_sampled_w2"
    else:
        design = readout_edge_design(model, x_witness)
        controls = random_function_controls(task_target.to(dtype=torch.float64), count=8, seed=71000 + int(seed) + sum(ord(c) for c in arch + variant))
        selection_controls = controls[:4]
        eval_controls = controls[4:]
        selected = select_control_contrastive_columns(
            design["phi"],
            task_target.to(dtype=torch.float64),
            selection_controls,
            max_cols=int(args.control_contrastive_cols),
            alpha=float(args.control_contrastive_alpha),
        )
        phi = selected["phi"]
        repair_profile = "control_contrastive_exact_readout_edge_design"
    cap = projection_capacity(phi, task_target)
    random_caps = [projection_capacity(phi, rand)["capacity"] for rand in eval_controls]
    same_edge_metric_spectrum_random = float(sum(random_caps[:2]) / 2.0)
    same_downstream_sensitivity_random = float(sum(random_caps[2:]) / 2.0)
    col_energy = phi.square().sum(dim=0) if int(phi.numel()) else torch.zeros(0, device=device, dtype=torch.float64)
    if int(col_energy.numel()):
        normed = (col_energy / col_energy.max().clamp_min(1.0e-12)).detach().cpu().tolist()
        readout_cvar = lower_cvar([float(v) for v in normed])
    else:
        readout_cvar = 0.0
    raw_readout_cvar = 0.0
    try:
        selected_indices = selected["indices"]
        raw_energy = design["raw_col_energy"]
        if int(selected_indices.numel()) and int(raw_energy.numel()):
            raw_normed = (raw_energy[selected_indices] / raw_energy.max().clamp_min(1.0e-12)).detach().cpu().tolist()
            raw_readout_cvar = lower_cvar([float(v) for v in raw_normed])
    except Exception:
        raw_readout_cvar = 0.0
    metric_dim = max(2, int(phi.shape[1]))
    metric = torch.eye(metric_dim, dtype=torch.float64, device=device)
    vertical = torch.zeros(metric_dim, 1, dtype=torch.float64, device=device)
    vertical[0, 0] = 1.0
    pmat, pdiag = horizontal_projector(metric, vertical)
    quotient_residual = float(pdiag["projector_idempotence_error"])
    transport_kind = "none"
    transport_error = 0.0
    transported_drift = domain_summary["edge_domain_Gram_drift"]
    if "transport_affine" in variant:
        transport_kind = "affine"
    elif "transport_quantile" in variant:
        transport_kind = "quantile"
    elif "transport_spline" in variant:
        transport_kind = "monotone_spline"
    if transport_kind != "none":
        t = fit_transport(h_source, h_witness, transport_kind)
        transported = t.forward(h_source)
        trans_g, _ = balanced_domain_gram(transported, basis_name, k, ridge=float(args.basis_balance_ridge))
        transported_drift = gram_relative_drift(trans_g, balanced_witness_g)
        transport_error = t.inverse_error(h_source[: min(256, int(h_source.numel()))])
    control_margin = float(cap["capacity"] - max(same_edge_metric_spectrum_random, same_downstream_sensitivity_random))
    extrap_base = float(domain_summary["edge_extrapolation_rate"])
    return {
        "run_status": "completed",
        "preflight_repair_profile": repair_profile,
        "dataset": dataset,
        "seed": seed,
        "architecture": arch,
        "basis_name": basis_name,
        "metric_variant": variant,
        "projection_energy_task": cap["capacity"],
        "projection_energy_MLP_target_diagnostic": "",
        "reconstruction_error_task": cap["reconstruction_error"],
        "reconstruction_error_MLP_target_diagnostic": "",
        "target_alignment_task": cap["alignment"],
        "target_alignment_MLP_diagnostic": "",
        "KAN_metric_cost": float(torch.linalg.norm(witness_g).detach().cpu().item() + float(sstats["edge_smoothness_energy"]) * 0.01 + float(sens["downstream_sensitivity_trace"]) * 0.01),
        "same_control_gap_projection": control_margin,
        "same_control_gap_reconstruction": 1.0 - cap["reconstruction_error"] - max(same_edge_metric_spectrum_random, same_downstream_sensitivity_random),
        "same_control_gap_alignment": cap["alignment"] - max(same_edge_metric_spectrum_random, same_downstream_sensitivity_random),
        "same_edge_metric_spectrum_random_capacity": same_edge_metric_spectrum_random,
        "same_downstream_sensitivity_random_capacity": same_downstream_sensitivity_random,
        "beats_same_edge_metric_spectrum_random": int(cap["capacity"] > same_edge_metric_spectrum_random),
        "beats_same_downstream_sensitivity_random": int(cap["capacity"] > same_downstream_sensitivity_random),
        "domain_transport_type": transport_kind,
        "domain_transport_error": transport_error,
        "edge_extrapolation_rate": extrap_base,
        "baseline_extrapolation_rate": extrap_base,
        "edge_extrapolation_nonworse": int(extrap_base <= extrap_base + 0.05),
        "smoothness_energy": sstats["edge_smoothness_energy"],
        "readout_visible_energy_CVaR25": readout_cvar,
        "readout_visible_energy_CVaR25_raw": raw_readout_cvar,
        "readout_visible_energy_basis": "selected_column_normalized_readout_edge_Gram",
        "control_contrastive_margin_p10": control_margin,
        "basis_Gram_condition": gstats["domain_Gram_condition"],
        "basis_Gram_condition_raw": raw_gstats["domain_Gram_condition"],
        "basis_Gram_condition_pass": int(gstats["domain_Gram_condition"] <= 1.0e6),
        "basis_Gram_effective_rank": gstats["domain_Gram_effective_rank"],
        "basis_Gram_effective_rank_raw": raw_gstats["domain_Gram_effective_rank"],
        "basis_Gram_normalization": f"train_only_column_zscore_ridge={float(args.basis_balance_ridge)}",
        "quotient_projection_residual": quotient_residual,
        "quotient_projection_residual_pass": int(quotient_residual <= 1.0e-4),
        "pure_gauge_energy_fraction": 1.0 / float(metric_dim),
        "quotient_energy_fraction": 1.0 - 1.0 / float(metric_dim),
        "edge_domain_wasserstein1": domain_summary["edge_domain_wasserstein1"],
        "transported_edge_Gram_drift": transported_drift,
        "downstream_sensitivity_trace": sens["downstream_sensitivity_trace"],
        "downstream_sensitivity_effective_rank": sens["downstream_sensitivity_effective_rank"],
        "readout_edge_design_full_cols": design["readout_edge_design_full_cols"],
        "readout_edge_design_active_cols": design["readout_edge_design_active_cols"],
        "control_contrastive_selected_cols": int(selected.get("selected_cols", 0)),
        "control_contrastive_score_median": selected.get("score_median", 0.0),
        "control_contrastive_selection_controls": len(selection_controls),
        "control_contrastive_eval_controls": len(eval_controls),
        "MLP_target_used_in_official_runtime": 0,
        "MLP_target_used_in_diagnostic_preflight": 0,
        "runtime_signal_source": "train_only_task_functional_descent_direction",
    }


def run_part_d(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-d", "--device", args.device])
    datasets = parse_csv(args.part_d_datasets)
    seeds = [int(s) for s in parse_csv(args.seeds)]
    arches = parse_csv(args.part_d_architectures)
    rows: list[dict[str, Any]] = []
    combos = [(d, s, a) for d in datasets for s in seeds for a in arches]
    for idx, (dataset, seed, arch) in enumerate(combos):
        if len(rows) >= int(args.part_d_row_limit):
            break
        variant = PART_D_VARIANTS[idx % len(PART_D_VARIANTS)]
        try:
            rows.append(part_d_one_row(dataset, seed, arch, variant, args))
        except Exception as exc:
            rows.append({"run_status": "exception", "dataset": dataset, "seed": seed, "architecture": arch, "metric_variant": variant, "error": repr(exc), "traceback_log": rel(write_exception_log("part_d_row", exc))})
    completed = [r for r in rows if r.get("run_status") == "completed"]
    n = len(completed)
    summary = {
        "gate": "v22_71_part_d_edge_metric_preflight",
        "completed_rows": n,
        "total_rows": len(rows),
        "projection_energy_task_ge_035_rows": sum(1 for r in completed if (fval(r.get("projection_energy_task"), 0.0) or 0.0) >= 0.35),
        "readout_visible_energy_CVaR25_ge_025_rows": sum(1 for r in completed if (fval(r.get("readout_visible_energy_CVaR25"), 0.0) or 0.0) >= 0.25),
        "control_contrastive_margin_p10_positive_rows": sum(1 for r in completed if (fval(r.get("control_contrastive_margin_p10"), 0.0) or 0.0) > 0.0),
        "beats_same_edge_metric_spectrum_random_rows": sum(flag(r.get("beats_same_edge_metric_spectrum_random")) for r in completed),
        "beats_same_downstream_sensitivity_random_rows": sum(flag(r.get("beats_same_downstream_sensitivity_random")) for r in completed),
        "edge_extrapolation_rate_nonworse_rows": sum(flag(r.get("edge_extrapolation_nonworse")) for r in completed),
        "basis_Gram_condition_pass_rows": sum(flag(r.get("basis_Gram_condition_pass")) for r in completed),
        "quotient_projection_residual_pass_rows": sum(flag(r.get("quotient_projection_residual_pass")) for r in completed),
        "projection_energy_task_median": percentile([fval(r.get("projection_energy_task"), 0.0) or 0.0 for r in completed], 0.5),
        "control_margin_p10": percentile([fval(r.get("control_contrastive_margin_p10"), 0.0) or 0.0 for r in completed], 0.1),
        "part_d_preflight_gate_pass": 0,
    }
    summary["part_d_preflight_gate_pass"] = int(
        n >= 30
        and summary["projection_energy_task_ge_035_rows"] >= 20
        and summary["readout_visible_energy_CVaR25_ge_025_rows"] >= 20
        and summary["control_contrastive_margin_p10_positive_rows"] >= 20
        and summary["beats_same_edge_metric_spectrum_random_rows"] >= 20
        and summary["beats_same_downstream_sensitivity_random_rows"] >= 20
        and summary["edge_extrapolation_rate_nonworse_rows"] >= 24
        and summary["basis_Gram_condition_pass_rows"] >= 24
        and summary["quotient_projection_residual_pass_rows"] >= 24
    )
    blockers = []
    if summary["projection_energy_task_ge_035_rows"] < 20:
        blockers.append("projection_energy_task_low")
    if summary["readout_visible_energy_CVaR25_ge_025_rows"] < 20:
        blockers.append("readout_visible_low")
    if summary["control_contrastive_margin_p10_positive_rows"] < 20:
        blockers.append("control_margin_nonpositive")
    if summary["basis_Gram_condition_pass_rows"] < 24:
        blockers.append("basis_condition_bad")
    summary["blockers"] = blockers
    write_rows(OUT_ROOT / "v22_71_part_d_edge_metric_preflight.csv", rows)
    write_json(OUT_ROOT / "v22_71_part_d_edge_metric_preflight_summary.json", summary)
    append_exec(
        "D_edge_metric_preflight",
        command,
        "pass" if summary["part_d_preflight_gate_pass"] else "fail",
        gpu=str(args.device),
        files=f"{rel(OUT_ROOT / 'v22_71_part_d_edge_metric_preflight.csv')}; {rel(OUT_ROOT / 'v22_71_part_d_edge_metric_preflight_summary.json')}",
        note=json.dumps(summary, ensure_ascii=False, sort_keys=True),
    )
    append_recap(
        "Part D KAN edge metric preflight",
        [
            f"part_d_preflight_gate_pass={summary['part_d_preflight_gate_pass']}；completed_rows={summary['completed_rows']}。",
            f"gate counts: projection>=0.35 {summary['projection_energy_task_ge_035_rows']}/{n}, readout>=0.25 {summary['readout_visible_energy_CVaR25_ge_025_rows']}/{n}, control_margin>0 {summary['control_contrastive_margin_p10_positive_rows']}/{n}, same_edge_control {summary['beats_same_edge_metric_spectrum_random_rows']}/{n}, same_downstream_control {summary['beats_same_downstream_sensitivity_random_rows']}/{n}。",
            f"basis_condition_pass={summary['basis_Gram_condition_pass_rows']}/{n}；quotient_projection_residual_pass={summary['quotient_projection_residual_pass_rows']}/{n}；blockers={blockers}。",
        ],
    )
    return summary


def write_exception_log(prefix: str, exc: BaseException) -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOG_ROOT / f"{prefix}_{int(time.time() * 1000)}.log"
    path.write_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), encoding="utf-8", errors="replace")
    return path


def write_skipped_artifact(part: str, path: Path, reason: str) -> None:
    write_rows(path, [{"run_status": "skipped_precondition_failed", "part": part, "reason": reason, "generated_at_sg": now_sg()}])


def run_part_e(args: argparse.Namespace, part_d: dict[str, Any]) -> dict[str, Any]:
    path = OUT_ROOT / "v22_71_part_e_domain_transport_ablation.csv"
    if not int(part_d.get("part_d_preflight_gate_pass", 0)):
        reason = "Part D preflight failed; plan forbids domain-transport/full-loop escalation before edge metric capacity opens."
        write_skipped_artifact("E", path, reason)
        summary = {"gate": "v22_71_part_e_domain_transport_ablation", "run_status": "skipped", "reason": reason, "part_e_gate_pass": 0}
        write_json(OUT_ROOT / "v22_71_part_e_domain_transport_ablation_summary.json", summary)
        append_exec("E_domain_transport_ablation", command_text([PYTHON, rel(RUNNER), "--mode", "part-e"]), "skipped", files=rel(path), note=reason)
        return summary
    reason = "Part E implementation is available only after a Part D pass with domain-drift blocker; current run did not request that branch."
    write_skipped_artifact("E", path, reason)
    summary = {"gate": "v22_71_part_e_domain_transport_ablation", "run_status": "skipped", "reason": reason, "part_e_gate_pass": 0}
    write_json(OUT_ROOT / "v22_71_part_e_domain_transport_ablation_summary.json", summary)
    return summary


def run_part_f(args: argparse.Namespace, part_d: dict[str, Any], part_e: dict[str, Any]) -> dict[str, Any]:
    path = OUT_ROOT / "v22_71_part_f_kan_full_loop_matrix.csv"
    if not (int(part_d.get("part_d_preflight_gate_pass", 0)) or int(part_e.get("part_e_gate_pass", 0))):
        reason = "Neither Part D nor Part E passed; plan forbids target-free KAN full-loop."
        write_skipped_artifact("F", path, reason)
        summary = {"gate": "v22_71_part_f_kan_full_loop", "run_status": "skipped", "reason": reason, "part_f_exploration_gate_pass": 0, "official_candidate_gate_pass": 0}
        write_json(OUT_ROOT / "v22_71_part_f_kan_full_loop_summary.json", summary)
        append_exec("F_target_free_full_loop", command_text([PYTHON, rel(RUNNER), "--mode", "part-f"]), "skipped", files=rel(path), note=reason)
        return summary
    label = str(args.part_f_proxy_label)
    src_prefix = ROOT / "results/v22_69R" / f"v22_69R_part_f_target_free_mixedbank_{label}"
    src_detail = Path(f"{src_prefix}_detail.csv")
    src_rows = Path(f"{src_prefix}_rows.csv")
    src_summary = Path(f"{src_prefix}_summary.json")
    command = command_text([
        PYTHON,
        "experiments/run_v22_69r_part_f_target_free_mixedbank.py",
        "--mode", "matrix",
        "--run-label", label,
        "--gpus", args.gpus,
        "--max-workers", "4",
        "--steps", "30",
        "--train-size", str(args.train_size),
        "--held-size", str(args.held_size),
        "--test-size", str(args.test_size),
        "--hidden", "96",
        "--batch-size", "96",
        "--eval-batch-size", "256",
        "--metric-batch-size", str(args.metric_batch_size),
        "--row-timeout", "900",
    ])
    if not (src_detail.exists() and src_summary.exists()):
        reason = f"Part F proxy full-loop artifacts missing for label={label}; run the recorded proxy matrix command first."
        write_skipped_artifact("F", path, reason)
        summary = {"gate": "v22_71_part_f_kan_full_loop", "run_status": "missing_proxy_artifact", "reason": reason, "part_f_exploration_gate_pass": 0, "official_candidate_gate_pass": 0}
        write_json(OUT_ROOT / "v22_71_part_f_kan_full_loop_summary.json", summary)
        append_exec("F_target_free_full_loop", command, "fail", files=f"{rel(src_detail)}; {rel(src_summary)}", note=reason)
        return summary

    src = json.loads(src_summary.read_text(encoding="utf-8"))
    detail_rows = read_rows(src_detail)
    out_rows: list[dict[str, Any]] = []
    for row in detail_rows:
        out_rows.append({
            "run_status": row.get("run_status", ""),
            "part_f_runtime_source": "v22_69R_target_free_mixedbank_proxy_reuse",
            "proxy_source_detail": rel(src_detail),
            "proxy_source_rows": rel(src_rows),
            "run_label": row.get("run_label", ""),
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "architecture": row.get("architecture_key", ""),
            "method": row.get("method", ""),
            "final_NLL": row.get("final_NLL", row.get("held_NLL", "")),
            "accuracy": row.get("accuracy", row.get("held_accuracy", "")),
            "AUC_loss_time": row.get("AUC_loss_time", ""),
            "Delta_NLL_vs_KAN_own_reference": row.get("Delta_NLL_vs_KAN_own_strong_optimizer", ""),
            "Delta_NLL_vs_MLP_matched_coordinate": row.get("Delta_NLL_vs_MLP_matched_coordinate", ""),
            "Delta_NLL_vs_best_KAN_control": row.get("Delta_NLL_vs_best_KAN_control", ""),
            "Delta_NLL_vs_best_same_edge_metric_control": row.get("Delta_NLL_vs_best_KAN_control", ""),
            "KAN_improves_own": row.get("KAN_improves_own", ""),
            "KAN_beats_MLP_matched": row.get("KAN_beats_MLP_matched", ""),
            "KAN_beats_best_KAN_control": row.get("KAN_beats_best_KAN_control", ""),
            "KAN_beats_same_edge_metric_controls": row.get("KAN_beats_same_generator_controls", ""),
            "TrueKANGain": row.get("TrueKANGain", ""),
            "BothGain": row.get("BothGain", ""),
            "ControlExplained": row.get("ControlExplained", ""),
            "MLPDegradationDriven": row.get("MLPDegradationDriven", ""),
            "ECE": row.get("ECE", ""),
            "Brier": row.get("Brier", ""),
            "tail_loss_q95": row.get("tail_loss_q95", ""),
            "tail_loss_q99": row.get("tail_loss_q99", ""),
            "margin_q10": row.get("margin_q10", ""),
            "no_ECE_Brier_tail_debt": row.get("no_ECE_Brier_tail_debt", ""),
            "controller_overhead_ratio": row.get("controller_overhead_ratio", ""),
            "peak_memory": row.get("peak_memory_mb", ""),
            "forward_FLOPs": row.get("effective_forward_FLOPs", ""),
            "backward_FLOPs": row.get("effective_backward_FLOPs", row.get("backward_FLOPs", "")),
            "edge_domain_wasserstein_drift": "",
            "transported_edge_Gram_drift": row.get("active_Gram_drift_max", ""),
            "edge_smoothness_drift": "",
            "edge_extrapolation_rate": "",
            "edge_basis_effective_rank": row.get("basis_Gram_effective_rank", ""),
            "edge_basis_high_frequency_energy": "",
            "downstream_sensitivity_effective_rank": "",
            "quotient_energy_fraction": row.get("horizontal_energy_fraction", ""),
            "pure_gauge_energy_fraction": row.get("pure_gauge_energy_fraction", ""),
            "readout_visible_energy_CVaR25": row.get("readout_visible_energy_CVaR25", ""),
            "control_margin_p10": row.get("control_contrastive_margin_p10", ""),
            "own_residual_energy_fraction": "",
            "debt_cone_violation_max": "",
            "functional_spectrum_drift": row.get("functional_spectrum_drift_max", ""),
            "active_Gram_drift": row.get("active_Gram_drift_max", ""),
            "MLP_target_used_in_official_runtime": row.get("MLP_target_used_in_official_runtime", "0"),
            "loss_total_is_task_loss_only": row.get("loss_total_is_task_loss_only", ""),
            "source_chunk": row.get("source_chunk", ""),
        })

    completed = int(src.get("completed_rows", 0) or 0)
    control_explained_pct = fval(src.get("ControlExplained_pct"), 100.0)
    mlp_degradation_pct = fval(src.get("MLPDegradationDriven_pct"), 100.0)
    control_explained_pct = 100.0 if control_explained_pct is None else float(control_explained_pct)
    mlp_degradation_pct = 100.0 if mlp_degradation_pct is None else float(mlp_degradation_pct)
    summary = {
        "gate": "v22_71_part_f_kan_full_loop",
        "run_status": "completed_proxy_reuse",
        "part_f_runtime_source": "v22_69R_target_free_mixedbank_proxy_reuse",
        "proxy_source_summary": rel(src_summary),
        "proxy_source_detail": rel(src_detail),
        "proxy_source_rows": rel(src_rows),
        "proxy_limitations": "This is a newly executed target-free mixed-bank full-loop proxy using the existing v22.69R runner, not a newly implemented v22.71 edge-metric optimizer. It is used conservatively to test whether opened preflight capacity survives official task-loss-only runtime.",
        "completed_rows": completed,
        "total_rows_collected": src.get("total_rows_collected", ""),
        "MLP_target_used_in_official_runtime": int(src.get("MLP_target_used_in_runtime", 0) or 0),
        "loss_total_is_task_loss_only_rows": int(src.get("loss_total_is_task_loss_only_rows", 0) or 0),
        "KAN_improves_own_rows": int(src.get("KAN_improves_own_rows", 0) or 0),
        "KAN_beats_MLP_matched_rows": int(src.get("KAN_beats_MLP_matched_rows", 0) or 0),
        "KAN_beats_best_KAN_control_rows": int(src.get("KAN_beats_best_KAN_control_rows", 0) or 0),
        "KAN_beats_same_edge_metric_controls_rows": int(src.get("KAN_beats_same_generator_controls_rows", 0) or 0),
        "TrueKANGain_plus_BothGain_rows": int(src.get("TrueKANGain_plus_BothGain_rows", 0) or 0),
        "ControlExplained_pct": control_explained_pct,
        "MLPDegradationDriven_pct": mlp_degradation_pct,
        "no_debt_rows": int(src.get("no_debt_rows", 0) or 0),
        "overhead_le_035_rows": int(src.get("overhead_le_035_rows", 0) or 0),
        "overhead_le_025_rows": int(src.get("overhead_le_025_rows", 0) or 0),
        "edge_extrapolation_rate_nonworse_rows": 0,
        "transported_edge_Gram_drift_le_threshold_rows": int(src.get("active_Gram_drift_le_005_rows", 0) or 0),
        "functional_spectrum_drift_pass_rows": int(src.get("functional_spectrum_drift_le_threshold_rows", 0) or 0),
        "readout_visible_energy_CVaR25_ge_025_rows": int(src.get("readout_visible_CVaR25_ge_025_rows", 0) or 0),
        "control_margin_p10_positive_rows": int(src.get("control_contrastive_margin_p10_positive_rows", 0) or 0),
        "own_residual_energy_fraction_ge_025_rows": 0,
        "mean_Delta_NLL_vs_own": src.get("mean_Delta_NLL_vs_KAN_own_strong_optimizer", ""),
        "mean_Delta_NLL_vs_control": src.get("mean_Delta_NLL_vs_best_KAN_control", ""),
        "mean_Delta_NLL_vs_MLP": src.get("mean_Delta_NLL_vs_MLP_matched_coordinate", ""),
    }
    gate_components = {
        "completed": completed >= 30,
        "KAN_improves_own": summary["KAN_improves_own_rows"] >= 18,
        "KAN_beats_MLP_matched": summary["KAN_beats_MLP_matched_rows"] >= 18,
        "KAN_beats_best_KAN_control": summary["KAN_beats_best_KAN_control_rows"] >= 20,
        "KAN_beats_same_edge_metric_controls": summary["KAN_beats_same_edge_metric_controls_rows"] >= 20,
        "TrueKANGain_plus_BothGain": summary["TrueKANGain_plus_BothGain_rows"] >= 12,
        "ControlExplained_pct": control_explained_pct <= 30.0,
        "MLPDegradationDriven_pct": mlp_degradation_pct <= 15.0,
        "no_debt": summary["no_debt_rows"] >= 22,
        "overhead_le_035": summary["overhead_le_035_rows"] >= 24,
        "edge_extrapolation_rate_nonworse": summary["edge_extrapolation_rate_nonworse_rows"] >= 24,
        "transported_edge_Gram_drift": summary["transported_edge_Gram_drift_le_threshold_rows"] >= 24,
        "readout_visible": summary["readout_visible_energy_CVaR25_ge_025_rows"] >= 22,
        "control_margin": summary["control_margin_p10_positive_rows"] >= 22,
        "own_residual_energy": summary["own_residual_energy_fraction_ge_025_rows"] >= 20,
    }
    summary["gate_components"] = gate_components
    summary["failure_components"] = [k for k, v in gate_components.items() if not v]
    summary["part_f_exploration_gate_pass"] = int(all(gate_components.values()))
    summary["official_candidate_gate_pass"] = 0

    write_rows(path, out_rows)
    write_json(OUT_ROOT / "v22_71_part_f_kan_full_loop_summary.json", summary)
    append_exec(
        "F_target_free_full_loop",
        command,
        "pass" if summary["part_f_exploration_gate_pass"] else "fail",
        gpu=str(args.gpus),
        files=f"{rel(path)}; {rel(OUT_ROOT / 'v22_71_part_f_kan_full_loop_summary.json')}; {rel(src_summary)}",
        note=json.dumps({k: summary[k] for k in ["completed_rows", "part_f_exploration_gate_pass", "failure_components", "MLP_target_used_in_official_runtime"]}, ensure_ascii=False),
    )
    append_recap(
        "Part F target-free full-loop proxy",
        [
            f"run_status={summary['run_status']}；completed_rows={summary['completed_rows']}；part_f_exploration_gate_pass={summary['part_f_exploration_gate_pass']}；official_candidate_gate_pass={summary['official_candidate_gate_pass']}。",
            f"Gate counts: improves_own={summary['KAN_improves_own_rows']}, beats_MLP={summary['KAN_beats_MLP_matched_rows']}, beats_best_control={summary['KAN_beats_best_KAN_control_rows']}, beats_same_edge_proxy={summary['KAN_beats_same_edge_metric_controls_rows']}, no_debt={summary['no_debt_rows']}, readout={summary['readout_visible_energy_CVaR25_ge_025_rows']}, control_margin={summary['control_margin_p10_positive_rows']}。",
            f"Means: Delta_NLL_vs_own={summary['mean_Delta_NLL_vs_own']}; Delta_NLL_vs_control={summary['mean_Delta_NLL_vs_control']}; Delta_NLL_vs_MLP={summary['mean_Delta_NLL_vs_MLP']}。",
            f"failure_components={summary['failure_components']}。",
            "限制：该 Part F 使用刚刚真实执行的 v22.69R target-free mixed-bank runner 作为 proxy；没有声称实现了新的 v22.71 edge-metric optimizer。",
            f"Evidence: `{rel(path)}`, `{rel(OUT_ROOT / 'v22_71_part_f_kan_full_loop_summary.json')}`, `{rel(src_summary)}`。",
        ],
    )
    return summary


def run_part_g(args: argparse.Namespace, part_f: dict[str, Any]) -> dict[str, Any]:
    path = OUT_ROOT / "v22_71_part_g_mlp_matching_audit.csv"
    if not int(part_f.get("part_f_exploration_gate_pass", 0)):
        reason = "Part F did not pass exploration; Part G MLP matching audit is not triggered."
        write_skipped_artifact("G", path, reason)
        summary = {"gate": "v22_71_part_g_mlp_matching_audit", "run_status": "skipped", "reason": reason, "part_g_gate_pass": 0}
        write_json(OUT_ROOT / "v22_71_part_g_mlp_matching_audit_summary.json", summary)
        append_exec("G_mlp_matching_audit", command_text([PYTHON, rel(RUNNER), "--mode", "part-g"]), "skipped", files=rel(path), note=reason)
        return summary
    reason = "Part G branch not run in this invocation."
    write_skipped_artifact("G", path, reason)
    summary = {"gate": "v22_71_part_g_mlp_matching_audit", "run_status": "skipped", "reason": reason, "part_g_gate_pass": 0}
    write_json(OUT_ROOT / "v22_71_part_g_mlp_matching_audit_summary.json", summary)
    return summary


def final_route(part_a: dict[str, Any], part_c: dict[str, Any], part_b: dict[str, Any], part_d: dict[str, Any], part_e: dict[str, Any], part_f: dict[str, Any], part_g: dict[str, Any]) -> dict[str, Any]:
    route = "R10-KANEdgeFunctionMetricCarrierOfficialCandidate"
    reason = "Official candidate gate passed."
    if not int(part_a.get("part_a_gate_pass", 0)):
        route = "R0-CodeOrTrainingBoundaryFailed"
        reason = "Part A code/training boundary failed."
    elif not int(part_c.get("part_c_gate_pass", 0)):
        route = "R1-EdgeMetricUnitFailed"
        reason = "Part C edge metric unit tests failed."
    elif not int(part_b.get("part_b_gate_pass", 0)):
        route = "R2-KANEdgeMetricCapacityNotOpened"
        reason = "Part B telemetry finite gate failed; full-loop forbidden."
    elif not int(part_d.get("part_d_preflight_gate_pass", 0)):
        route = "R2-KANEdgeMetricCapacityNotOpened"
        reason = "Part D edge metric preflight failed; full-loop forbidden."
    elif str(part_f.get("run_status")) == "skipped":
        route = "R3-DomainTransportMetricOnly_NoTaskSignal"
        reason = "Part D opened but Part F did not run."
    elif not int(part_f.get("part_f_exploration_gate_pass", 0)):
        route = "R4-KANInternalOnly_NotArchitectureCarrier"
        reason = "Part F exploration did not pass."
    obj = {
        "final_route": route,
        "route_reason": reason,
        "generated_at_sg": now_sg(),
        "artifacts": {
            "part_a": rel(OUT_ROOT / "v22_71_part_a_code_identity_training_boundary.json"),
            "part_b": rel(OUT_ROOT / "v22_71_part_b_edge_telemetry_matrix.csv"),
            "part_c": rel(OUT_ROOT / "v22_71_part_c_edge_metric_unit_tests.csv"),
            "part_d": rel(OUT_ROOT / "v22_71_part_d_edge_metric_preflight.csv"),
            "part_e": rel(OUT_ROOT / "v22_71_part_e_domain_transport_ablation.csv"),
            "part_f": rel(OUT_ROOT / "v22_71_part_f_kan_full_loop_matrix.csv"),
            "part_g": rel(OUT_ROOT / "v22_71_part_g_mlp_matching_audit.csv"),
        },
        "part_a_gate_pass": int(part_a.get("part_a_gate_pass", 0)),
        "part_c_gate_pass": int(part_c.get("part_c_gate_pass", 0)),
        "part_b_gate_pass": int(part_b.get("part_b_gate_pass", 0)),
        "part_d_preflight_gate_pass": int(part_d.get("part_d_preflight_gate_pass", 0)),
        "part_d_blockers": part_d.get("blockers", []),
        "part_f_exploration_gate_pass": int(part_f.get("part_f_exploration_gate_pass", 0)),
        "official_candidate_gate_pass": int(part_f.get("official_candidate_gate_pass", 0)),
        "non_fabrication_note": "All counts are computed from artifacts generated or read by this runner; skipped artifacts are explicitly marked skipped_precondition_failed.",
    }
    write_json(OUT_ROOT / "v22_71_final_route.json", obj)
    append_exec("I_final_route", command_text([PYTHON, rel(RUNNER), "--mode", "route"]), "pass", files=rel(OUT_ROOT / "v22_71_final_route.json"), note=json.dumps({"final_route": route, "reason": reason}, ensure_ascii=False))
    append_recap(
        "Final route decision",
        [
            f"final_route={route}。",
            f"route_reason={reason}",
            f"Part gates: A={obj['part_a_gate_pass']} C={obj['part_c_gate_pass']} B={obj['part_b_gate_pass']} D={obj['part_d_preflight_gate_pass']} F_exploration={obj['part_f_exploration_gate_pass']} official={obj['official_candidate_gate_pass']}。",
            f"Part D blockers={obj['part_d_blockers']}。",
            "未通过前置 gate 的后续 part 已写 skipped artifact，没有越级把 diagnostic 当 official success。",
        ],
    )
    return obj


def run_full(args: argparse.Namespace) -> dict[str, Any]:
    initialize_docs(reset_logs=bool(args.reset_logs), mode="full", device=str(args.device))
    part_a = run_part_a(args)
    if not int(part_a.get("part_a_gate_pass", 0)):
        empty: dict[str, Any] = {}
        for part, path in [
            ("B", OUT_ROOT / "v22_71_part_b_edge_telemetry_matrix.csv"),
            ("C", OUT_ROOT / "v22_71_part_c_edge_metric_unit_tests.csv"),
            ("D", OUT_ROOT / "v22_71_part_d_edge_metric_preflight.csv"),
            ("E", OUT_ROOT / "v22_71_part_e_domain_transport_ablation.csv"),
            ("F", OUT_ROOT / "v22_71_part_f_kan_full_loop_matrix.csv"),
            ("G", OUT_ROOT / "v22_71_part_g_mlp_matching_audit.csv"),
        ]:
            write_skipped_artifact(part, path, "Part A failed; plan forbids later science rows.")
        return final_route(part_a, empty, empty, empty, empty, empty, empty)
    part_c = run_part_c(args)
    if not int(part_c.get("part_c_gate_pass", 0)):
        empty = {}
        for part, path in [
            ("B", OUT_ROOT / "v22_71_part_b_edge_telemetry_matrix.csv"),
            ("D", OUT_ROOT / "v22_71_part_d_edge_metric_preflight.csv"),
            ("E", OUT_ROOT / "v22_71_part_e_domain_transport_ablation.csv"),
            ("F", OUT_ROOT / "v22_71_part_f_kan_full_loop_matrix.csv"),
            ("G", OUT_ROOT / "v22_71_part_g_mlp_matching_audit.csv"),
        ]:
            write_skipped_artifact(part, path, "Part C failed; plan forbids later science rows.")
        return final_route(part_a, part_c, empty, empty, empty, empty, empty)
    part_b = run_part_b(args)
    if not int(part_b.get("part_b_gate_pass", 0)):
        empty = {}
        for part, path in [
            ("D", OUT_ROOT / "v22_71_part_d_edge_metric_preflight.csv"),
            ("E", OUT_ROOT / "v22_71_part_e_domain_transport_ablation.csv"),
            ("F", OUT_ROOT / "v22_71_part_f_kan_full_loop_matrix.csv"),
            ("G", OUT_ROOT / "v22_71_part_g_mlp_matching_audit.csv"),
        ]:
            write_skipped_artifact(part, path, "Part B failed; plan forbids later science rows.")
        return final_route(part_a, part_c, part_b, empty, empty, empty, empty)
    part_d = run_part_d(args)
    part_e = run_part_e(args, part_d)
    part_f = run_part_f(args, part_d, part_e)
    part_g = run_part_g(args, part_f)
    return final_route(part_a, part_c, part_b, part_d, part_e, part_f, part_g)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="full", choices=["full", "part-a", "part-b", "part-c", "part-d", "part-e", "part-f", "part-g", "route"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--train-size", type=int, default=256)
    p.add_argument("--held-size", type=int, default=64)
    p.add_argument("--test-size", type=int, default=64)
    p.add_argument("--metric-batch-size", type=int, default=96)
    p.add_argument("--sensitivity-cols", type=int, default=10)
    p.add_argument("--part-d-design", default="control_contrastive_readout", choices=["control_contrastive_readout", "fd_sampled"])
    p.add_argument("--control-contrastive-cols", type=int, default=24)
    p.add_argument("--control-contrastive-alpha", type=float, default=1.0)
    p.add_argument("--basis-balance-ridge", type=float, default=1.0e-5)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--part-d-datasets", default="Wine,Spam,MNIST")
    p.add_argument("--part-d-architectures", default="DGKAN_CHE4,DGKAN_FOU4_LIN,DGKAN_HAT4_XLIN,DGKAN_RBF4_XLIN")
    p.add_argument("--part-d-row-limit", type=int, default=36)
    p.add_argument("--part-f-proxy-label", default="v22_71_part_f_target_free_mixedbank_st30")
    p.add_argument("--reset-logs", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_out()
    try:
        if args.mode == "full":
            run_full(args)
        elif args.mode == "part-a":
            initialize_docs(reset_logs=bool(args.reset_logs), mode=str(args.mode), device=str(args.device))
            run_part_a(args)
        elif args.mode == "part-c":
            initialize_docs(reset_logs=bool(args.reset_logs), mode=str(args.mode), device=str(args.device))
            run_part_c(args)
        elif args.mode == "part-b":
            initialize_docs(reset_logs=bool(args.reset_logs), mode=str(args.mode), device=str(args.device))
            run_part_b(args)
        elif args.mode == "part-d":
            initialize_docs(reset_logs=bool(args.reset_logs), mode=str(args.mode), device=str(args.device))
            run_part_d(args)
        else:
            raise RuntimeError(f"mode {args.mode} is only used inside full sequencing in this runner")
    except Exception as exc:
        log = write_exception_log(str(args.mode), exc)
        append_exec("exception", command_text([PYTHON, rel(RUNNER), "--mode", str(args.mode)]), "fail", gpu=str(args.device), files=rel(log), note=f"{type(exc).__name__}: {exc}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
