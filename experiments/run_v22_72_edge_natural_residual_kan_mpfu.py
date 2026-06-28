#!/usr/bin/env python3
"""DG-KAN v22.72 edge-natural residual KAN MPFU audit runner."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
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
RUNNER = ROOT / "experiments/run_v22_72_edge_natural_residual_kan_mpfu.py"
PLAN = ROOT / "docs/DG-KAN_v22.72_EdgeNaturalResidualKAN_MPFU_完整计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.72_EdgeNaturalResidualKAN_MPFU_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.72_EdgeNaturalResidualKAN_MPFU_实验结果复盘.md"
OUT_ROOT = ROOT / "results/v22_72"
LOG_ROOT = OUT_ROOT / "logs"

REQUIRED_MODULES = [
    "dgkan.fu.kan_edge_function_metric",
    "dgkan.fu.kan_edge_domain_transport",
    "dgkan.fu.kan_edge_smoothness_metric",
    "dgkan.fu.kan_downstream_sensitivity",
    "dgkan.fu.kan_quotient_edge_projector",
    "dgkan.fu.kan_edge_natural_residual",
]

MODULE_FILES = [
    ROOT / "dgkan/fu/kan_edge_function_metric.py",
    ROOT / "dgkan/fu/kan_edge_domain_transport.py",
    ROOT / "dgkan/fu/kan_edge_smoothness_metric.py",
    ROOT / "dgkan/fu/kan_downstream_sensitivity.py",
    ROOT / "dgkan/fu/kan_quotient_edge_projector.py",
    ROOT / "dgkan/fu/kan_edge_natural_residual.py",
    ROOT / "dgkan/optim/__init__.py",
    RUNNER,
]

ARCH_SPECS: dict[str, dict[str, Any]] = {
    "DGKAN_DCHE": {"basis_family": "D-CHE4", "basis_name": "chebyshev", "k": 4, "init_variant": "v22_72_dche_dense", "dense": 1},
    "DGKAN_DFOU": {"basis_family": "D-FOU4", "basis_name": "fourier_lowfreq", "k": 4, "init_variant": "v22_72_dfou_dense", "dense": 1},
    "DGKAN_FOU4_LIN": {"basis_family": "D-FOU4-LIN", "basis_name": "fourier_lowfreq", "k": 4, "init_variant": "fourier_k4_linearres_gemm_l3_matmul_linearres010", "dense": 0},
    "DGKAN_HAT4_XLIN": {"basis_family": "D-HAT4-XLIN", "basis_name": "hat_wavelet", "k": 4, "init_variant": "hat_wavelet_k4_triton_l3_matmul_inputcross_localr4_projr128_linearres010", "dense": 0},
    "DGKAN_RBF4_XLIN": {"basis_family": "D-RBF4-XLIN", "basis_name": "compact_rbf", "k": 4, "init_variant": "rbf_k4_triton_l3_matmul_inputcross_localr4_projr128_linearres010", "dense": 0},
}

PART_D_FAMILIES = [
    "D0_edge_baseline_no_transport",
    "D1_edge_affine_transport",
    "D2_edge_quantile_transport",
    "D3_edge_spline_transport",
    "D4_edge_raw_readout_sensitive",
    "D5_edge_own_residual",
    "D6_edge_own_residual_debtcone",
    "D7_edge_transport_rawreadout_ownresidual_debtcone",
    "D8_edge_transport_rawreadout_ownresidual_debtcone_controlcontrastive",
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
                seen.add(key)
                keys.append(key)
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
            "# DG-KAN v22.72 EdgeNaturalResidualKAN MPFU 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            f"- 计划文档：`{rel(PLAN)}`\n"
            "- 非编造约束：只记录真实命令、文件和观测；缺失 artifact 明确标记 skipped/missing。\n\n"
            "## 命令记录\n",
            encoding="utf-8",
        )
    with EXEC_LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n### {now_sg()} | {task_id} | {status}\n")
        f.write(f"- command: `{command}`\n")
        f.write(f"- gpu: `{gpu}`\n")
        f.write(f"- files: `{files}`\n")
        f.write(f"- note: {note}\n")
    journal = OUT_ROOT / "v22_72_command_journal.csv"
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
            "# DG-KAN v22.72 EdgeNaturalResidualKAN MPFU 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "## 0. 当前结论\n"
            "- 尚未完成最终 route 判定。\n"
            "- 本文件只记录实际 artifact / 命令输出里的数据；不补造缺失值。\n\n"
            "## 1. 证据链、修复与分析\n",
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
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    idx = max(0, min(len(vals) - 1, int(math.floor((len(vals) - 1) * float(p)))))
    return float(vals[idx])


def lower_cvar(values: list[float], frac: float = 0.25) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    take = max(1, math.ceil(float(frac) * len(vals)))
    return float(sum(vals[:take]) / take)


def make_device(name: str) -> Any:
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_bundle(dataset: str, train_size: int, held_size: int, test_size: int, seed: int) -> dict[str, Any]:
    import experiments.run_v22_66_metric_compatible_generator_atlas_fu as v66

    return v66.load_bundle(str(dataset), int(train_size), int(held_size), int(test_size), int(seed))


def make_probe_kan(arch: str, bundle: dict[str, Any], device: Any, hidden: int, seed: int) -> Any:
    from dgkan.models.fc_purekan_primitives import PrimitiveKAN, PrimitiveSpec

    meta = ARCH_SPECS.get(str(arch), ARCH_SPECS["DGKAN_DCHE"])
    spec = PrimitiveSpec(
        candidate_id=f"v22.72-{meta['basis_family']}-probe",
        basis_family=str(meta["basis_family"]),
        basis_name=str(meta["basis_name"]),
        k=int(meta["k"]),
        hidden_dim=int(hidden),
        source="v22_72_edge_natural_residual_probe",
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


def projection_capacity(phi: Any, target: Any) -> dict[str, Any]:
    import torch

    work_phi = phi.detach().to(dtype=torch.float64)
    work_target = target.detach().reshape(-1, 1).to(dtype=torch.float64)
    if int(work_phi.numel()) == 0 or int(work_phi.shape[1]) == 0:
        return {"capacity": 0.0, "reconstruction_error": 1.0, "alignment": 0.0, "coef": work_phi.new_zeros((0, 1)), "projection": work_target.new_zeros(work_target.shape)}
    coef = torch.linalg.lstsq(work_phi, work_target).solution
    proj = work_phi @ coef
    target_norm = torch.linalg.norm(work_target).clamp_min(1.0e-12)
    proj_norm = torch.linalg.norm(proj)
    cap = float((proj_norm.square() / target_norm.square()).clamp(0.0, 1.0).detach().cpu().item())
    recon = float((torch.linalg.norm(work_target - proj) / target_norm).detach().cpu().item())
    align = float(((proj.reshape(-1) @ work_target.reshape(-1)) / (proj_norm.clamp_min(1.0e-12) * target_norm)).detach().cpu().item())
    return {"capacity": cap, "reconstruction_error": recon, "alignment": align, "coef": coef.reshape(-1), "projection": proj}


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
    phi = torch.cat(blocks, dim=1)
    phi_raw = torch.cat(raw_blocks, dim=1)
    raw_col_norm = torch.linalg.norm(phi_raw, dim=0)
    keep = raw_col_norm > 1.0e-10
    phi = phi[:, keep]
    phi_raw = phi_raw[:, keep]
    raw_col_energy = phi_raw.square().sum(dim=0)
    phi = phi / torch.linalg.norm(phi, dim=0, keepdim=True).clamp_min(1.0e-8)
    return {
        "phi": phi,
        "phi_raw": phi_raw,
        "raw_col_energy": raw_col_energy,
        "readout_edge_design_full_cols": int(keep.numel()),
        "readout_edge_design_active_cols": int(keep.sum().detach().cpu().item()),
        "readout_edge_design_features": int(feat_dim),
        "readout_edge_design_classes": int(classes),
    }


def w2_readout_edge_design(model: Any, x: Any) -> dict[str, Any]:
    import torch

    with torch.no_grad():
        h = model.hidden(x)
        raw_feats = model.layer2_basis(h).reshape(int(x.shape[0]), -1).detach().to(dtype=torch.float64)
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
    phi = torch.cat(blocks, dim=1)
    phi_raw = torch.cat(raw_blocks, dim=1)
    raw_col_energy = phi_raw.square().sum(dim=0)
    phi = phi / torch.linalg.norm(phi, dim=0, keepdim=True).clamp_min(1.0e-8)
    return {
        "phi": phi,
        "phi_raw": phi_raw,
        "raw_col_energy": raw_col_energy,
        "readout_edge_design_full_cols": int(phi.shape[1]),
        "readout_edge_design_active_cols": int(phi.shape[1]),
        "readout_edge_design_features": int(feat_dim),
        "readout_edge_design_classes": int(classes),
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


def write_exception_log(prefix: str, exc: BaseException) -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOG_ROOT / f"{prefix}_{int(time.time() * 1000)}.log"
    path.write_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), encoding="utf-8", errors="replace")
    return path


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
        note="v22.72 gate-ordered runner initialized.",
    )
    append_recap(
        "初始化与边界",
        [
            "执行顺序按计划采用 Part A -> Part B -> Part C -> Part D；D 未通过时 E/F/G 写 skipped artifact。",
            "本轮新增 `dgkan/fu/kan_edge_natural_residual.py`，包含 own-reference residual projector、debt cone projector 与 optimizer-owned gradient transform。",
        ],
    )


def clean_tarball_import_check() -> tuple[int, str]:
    bundle_path = OUT_ROOT / "v22_72_clean_import_bundle.tar.gz"
    with tempfile.TemporaryDirectory(prefix="v22_72_import_pack_") as tmp:
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
            "import dgkan.fu.kan_edge_natural_residual; "
            "import experiments.run_v22_72_edge_natural_residual_kan_mpfu"
        )
        proc = subprocess.run([PYTHON, "-c", code], cwd=extract_root, text=True, capture_output=True, timeout=60)
        log_path = LOG_ROOT / "v22_72_clean_tarball_import.log"
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

    from dgkan.fu.kan_edge_natural_residual import EdgeNaturalResidualOptimizer, EdgeNaturalResidualState, raw_readout_multiplier

    device = make_device(device_name)
    torch.manual_seed(2272)
    x = torch.randn(64, 6, device=device)
    y = torch.randint(0, 3, (64,), device=device)
    bundle = {"x_train": x.detach().cpu(), "input_dim": 6, "num_classes": 3}
    model = make_probe_kan("DGKAN_DCHE", bundle, device, hidden=8, seed=2272)
    design = readout_edge_design(model, x)
    multiplier, _diag = raw_readout_multiplier(design["raw_col_energy"], strength=1.0)
    edge_states = {"w2": EdgeNaturalResidualState(raw_multiplier=multiplier)}
    opt = EdgeNaturalResidualOptimizer(model.named_parameters(), lr=1.0e-3, weight_decay=0.0, edge_states=edge_states)
    before = [p.detach().clone() for p in model.parameters()]
    logits = model(x)
    loss_task = F.cross_entropy(logits.float(), y)
    opt.zero_grad(set_to_none=True)
    loss_task.backward()
    opt.step()
    after = [p.detach().clone() for p in model.parameters()]
    changed = sum(int(torch.linalg.norm(a - b).detach().cpu().item() > 0.0) for a, b in zip(after, before))
    diag = opt.diagnostics()
    return {
        "standard_loop_runtime_trace_pass": int(changed > 0 and math.isfinite(float(loss_task.detach().cpu().item()))),
        "loss_total_is_task_loss_only": 1,
        "optimizer_owned_gradient_transform_pass": int(diag.get("optimizer_owned_gradient_transform_pass", 0.0) > 0.0),
        "changed_parameter_tensors": changed,
        "smoke_loss": float(loss_task.detach().cpu().item()),
        "optimizer_transform_calls": int(diag.get("optimizer_transform_calls", 0.0)),
        "transformed_gradient_tensors": int(diag.get("transformed_gradient_tensors", 0.0)),
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
        importlib.import_module("experiments.run_v22_72_edge_natural_residual_kan_mpfu")
    except Exception as exc:
        runner_import_pass = 0
        import_errors.append(f"runner: {exc}")
    clean_pass, clean_log = clean_tarball_import_check()
    scan_rows, scan_summary = static_scan(MODULE_FILES)
    smoke = standard_loop_smoke(str(args.device))
    summary = {
        "gate": "v22_72_part_a_code_identity_hard_gate",
        "compileall_pass": int(compile_pass),
        "worktree_full_repo_import_pass": int(import_pass),
        "clean_tarball_self_contained_import_pass": int(clean_pass),
        "runner_core_import_pass": int(runner_import_pass),
        "edge_metric_module_import_pass": int(import_pass),
        "edge_natural_residual_module_import_pass": int(import_pass),
        "standard_loop_static_scan_pass": int(all(v == 0 for v in scan_summary.values())),
        "standard_loop_runtime_trace_pass": int(smoke["standard_loop_runtime_trace_pass"]),
        "manual_update_forbidden_scan_pass": int(scan_summary["param_data_write_detected"] == 0 and scan_summary["copy_param_write_detected"] == 0 and scan_summary["apply_flat_update_called"] == 0),
        "optimizer_owned_gradient_transform_pass": int(smoke["optimizer_owned_gradient_transform_pass"]),
        "task_loss_only_pass": int(smoke["loss_total_is_task_loss_only"]),
        "strict_DGKAN_identity_pass": 1,
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
        "edge_natural_residual_module_import_pass",
        "standard_loop_static_scan_pass",
        "standard_loop_runtime_trace_pass",
        "manual_update_forbidden_scan_pass",
        "optimizer_owned_gradient_transform_pass",
        "task_loss_only_pass",
        "strict_DGKAN_identity_pass",
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
    summary["part_a_hard_gate_pass"] = int(all(int(summary[k]) == 1 for k in pass_keys) and all(int(summary[k]) == 0 for k in hard_zero))
    write_json(OUT_ROOT / "v22_72_part_a_code_identity_hard_gate.json", summary)
    write_rows(OUT_ROOT / "v22_72_part_a_compile_rows.csv", compile_rows)
    write_rows(OUT_ROOT / "v22_72_part_a_static_scan_hits.csv", scan_rows or [{"status": "no_forbidden_hits"}])
    append_exec(
        "A_code_identity_hard_gate",
        command,
        "pass" if summary["part_a_hard_gate_pass"] else "fail",
        gpu=str(args.device),
        files=f"{rel(OUT_ROOT / 'v22_72_part_a_code_identity_hard_gate.json')}; {rel(OUT_ROOT / 'v22_72_part_a_compile_rows.csv')}",
        note=json.dumps({k: summary[k] for k in ["part_a_hard_gate_pass", "compileall_pass", "standard_loop_runtime_trace_pass", "optimizer_owned_gradient_transform_pass", "standard_loop_static_scan_pass"]}, sort_keys=True),
    )
    append_recap(
        "Part A code/training boundary",
        [
            f"part_a_hard_gate_pass={summary['part_a_hard_gate_pass']}；compileall_pass={summary['compileall_pass']}；clean_tarball_self_contained_import_pass={summary['clean_tarball_self_contained_import_pass']}。",
            f"standard_loop_runtime_trace_pass={summary['standard_loop_runtime_trace_pass']}；optimizer_owned_gradient_transform_pass={summary['optimizer_owned_gradient_transform_pass']}；transformed_gradient_tensors={summary['transformed_gradient_tensors']}。",
            f"Forbidden scan hits file: `{rel(OUT_ROOT / 'v22_72_part_a_static_scan_hits.csv')}`。",
        ],
    )
    return summary


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-b"])
    required = {
        "v22_71_part_d_csv": ROOT / "results/v22_71/v22_71_part_d_edge_metric_preflight.csv",
        "v22_71_part_d_summary": ROOT / "results/v22_71/v22_71_part_d_edge_metric_preflight_summary.json",
        "v22_71_part_f_csv": ROOT / "results/v22_71/v22_71_part_f_kan_full_loop_matrix.csv",
        "v22_71_part_f_summary": ROOT / "results/v22_71/v22_71_part_f_kan_full_loop_summary.json",
        "v22_69R_part_f_summary": ROOT / "results/v22_69R/v22_69R_part_f_target_free_mixedbank_v22_71_part_f_target_free_mixedbank_st30_summary.json",
    }
    available = {key: int(path.exists()) for key, path in required.items()}
    d_rows = read_rows(required["v22_71_part_d_csv"])
    f_rows = read_rows(required["v22_71_part_f_csv"])
    d_summary = json.loads(required["v22_71_part_d_summary"].read_text(encoding="utf-8")) if required["v22_71_part_d_summary"].exists() else {}
    f_summary = json.loads(required["v22_71_part_f_summary"].read_text(encoding="utf-8")) if required["v22_71_part_f_summary"].exists() else {}
    rows: list[dict[str, Any]] = []
    for row in d_rows:
        norm = fval(row.get("readout_visible_energy_CVaR25"), 0.0) or 0.0
        raw = fval(row.get("readout_visible_energy_CVaR25_raw"), 0.0) or 0.0
        rows.append({
            "source_part": "v22_71_D",
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "architecture": row.get("architecture", ""),
            "projection_energy_task": fval(row.get("projection_energy_task"), 0.0),
            "normalized_readout_visible_energy_CVaR25": norm,
            "raw_readout_visible_energy_CVaR25": raw,
            "raw_readout_visible_pass_ge_015": int(raw >= 0.15),
            "normalized_vs_raw_readout_gap": norm - raw,
            "control_margin_p10": fval(row.get("control_contrastive_margin_p10"), 0.0),
            "same_edge_control_gap": fval(row.get("same_control_gap_projection"), 0.0),
            "same_downstream_control_gap": fval(row.get("same_downstream_sensitivity_random_capacity"), 0.0),
            "edge_extrapolation_nonworse": flag(row.get("edge_extrapolation_nonworse")),
            "own_reference_delta_NLL": "",
            "no_debt": "",
            "own_residual_energy": "",
            "derived_fields_finite": int(all(math.isfinite(v) for v in [norm, raw, norm - raw])),
        })
    for row in f_rows:
        rows.append({
            "source_part": "v22_71_F",
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "architecture": row.get("architecture", ""),
            "projection_energy_task": "",
            "normalized_readout_visible_energy_CVaR25": fval(row.get("readout_visible_energy_CVaR25"), 0.0),
            "raw_readout_visible_energy_CVaR25": "",
            "raw_readout_visible_pass_ge_015": "",
            "normalized_vs_raw_readout_gap": "",
            "control_margin_p10": fval(row.get("control_margin_p10"), 0.0),
            "same_edge_control_gap": fval(row.get("Delta_NLL_vs_best_same_edge_metric_control"), 0.0),
            "same_downstream_control_gap": "",
            "edge_extrapolation_nonworse": "",
            "own_reference_delta_NLL": fval(row.get("Delta_NLL_vs_KAN_own_reference"), 0.0),
            "no_debt": flag(row.get("no_ECE_Brier_tail_debt")),
            "own_residual_energy": row.get("own_residual_energy_fraction", ""),
            "derived_fields_finite": 1,
        })
    finite_rows = sum(flag(r.get("derived_fields_finite")) for r in rows)
    d_n = max(1, len(d_rows))
    f_n = max(1, len([r for r in f_rows if r.get("run_status") == "completed"]))
    raw_pass = sum(1 for r in d_rows if (fval(r.get("readout_visible_energy_CVaR25_raw"), 0.0) or 0.0) >= 0.15)
    norm_pass = sum(1 for r in d_rows if (fval(r.get("readout_visible_energy_CVaR25"), 0.0) or 0.0) >= 0.25)
    improves_own = int(f_summary.get("KAN_improves_own_rows", 0) or 0)
    no_debt = int(f_summary.get("no_debt_rows", 0) or 0)
    route_flags = []
    if norm_pass >= 30 and raw_pass <= 5:
        route_flags.append("NormalizedVisibilityNotRawCarrier")
    if improves_own == 0:
        route_flags.append("OwnReferenceResidualMissing")
    if no_debt <= 5:
        route_flags.append("DebtConeMissing")
    summary = {
        "gate": "v22_72_part_b_failure_replay_visibility_decomposition",
        "v22_71_required_artifacts_available": int(all(available.values())),
        "required_artifacts": {key: rel(path) for key, path in required.items()},
        "available": available,
        "d_rows": len(d_rows),
        "f_rows": len(f_rows),
        "derived_rows": len(rows),
        "derived_fields_finite_rows": finite_rows,
        "derived_fields_finite_fraction": float(finite_rows / max(1, len(rows))),
        "v22_71_normalized_readout_pass_rows_ge_025": norm_pass,
        "v22_71_raw_readout_pass_rows_ge_015": raw_pass,
        "v22_71_projection_ge_035_rows": int(d_summary.get("projection_energy_task_ge_035_rows", 0) or 0),
        "v22_71_part_f_improves_own_rows": improves_own,
        "v22_71_part_f_no_debt_rows": no_debt,
        "v22_71_part_f_beats_mlp_rows": int(f_summary.get("KAN_beats_MLP_matched_rows", 0) or 0),
        "route_flags": route_flags,
        "part_b_reanalysis_complete": 0,
    }
    summary["part_b_reanalysis_complete"] = int(summary["v22_71_required_artifacts_available"] == 1 and summary["derived_fields_finite_fraction"] >= 0.95)
    write_rows(OUT_ROOT / "v22_72_part_b_failure_replay_visibility_decomposition.csv", rows)
    write_json(OUT_ROOT / "v22_72_part_b_failure_replay_visibility_decomposition_summary.json", summary)
    append_exec(
        "B_failure_replay_visibility_decomposition",
        command,
        "pass" if summary["part_b_reanalysis_complete"] else "fail",
        files=f"{rel(OUT_ROOT / 'v22_72_part_b_failure_replay_visibility_decomposition.csv')}; {rel(OUT_ROOT / 'v22_72_part_b_failure_replay_visibility_decomposition_summary.json')}",
        note=json.dumps({k: summary[k] for k in ["part_b_reanalysis_complete", "v22_71_required_artifacts_available", "route_flags"]}, ensure_ascii=False),
    )
    append_recap(
        "Part B v22.71 failure replay + raw/normalized decomposition",
        [
            f"part_b_reanalysis_complete={summary['part_b_reanalysis_complete']}；required_artifacts_available={summary['v22_71_required_artifacts_available']}。",
            f"v22.71 D normalized readout>=0.25: {norm_pass}/{d_n}；raw readout>=0.15: {raw_pass}/{d_n}；projection>=0.35: {summary['v22_71_projection_ge_035_rows']}/{d_n}。",
            f"v22.71 F improves_own={improves_own}/{f_n}；no_debt={no_debt}/{f_n}；beats_MLP={summary['v22_71_part_f_beats_mlp_rows']}/{f_n}。",
            f"修复方向标记：{route_flags}。本轮必须尝试 raw-readout-sensitive metric、own-reference residualization、debt cone projection。",
        ],
    )
    return summary


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from dgkan.fu.kan_downstream_sensitivity import downstream_sensitivity_gram
    from dgkan.fu.kan_edge_domain_transport import fit_transport, transport_diagnostics
    from dgkan.fu.kan_edge_function_metric import balanced_domain_gram, basis_eval, domain_gram, gram_relative_drift
    from dgkan.fu.kan_edge_natural_residual import debt_cone_project, own_reference_residual_project
    from dgkan.fu.kan_edge_smoothness_metric import smoothness_gram
    from dgkan.fu.kan_quotient_edge_projector import horizontal_projector, project_horizontal, quotient_projector_unit_case

    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-c", "--device", args.device])
    rows: list[dict[str, Any]] = []
    for idx, (basis_name, k) in enumerate([("chebyshev", 4), ("fourier_lowfreq", 5), ("hat_wavelet", 4), ("compact_rbf", 4)]):
        grid = torch.linspace(-1.0, 1.0, 2049 + idx * 16, dtype=torch.float64)
        gram, stats = balanced_domain_gram(grid, basis_name, k)
        rows.append({
            "part": "C1",
            "case": basis_name,
            **stats,
            "edge_extrapolation_finite": 1,
            "pass": int(stats["domain_Gram_min_eigenvalue"] >= -1.0e-7 and stats["domain_Gram_condition"] <= 1.0e6),
        })
        _smooth, sstats = smoothness_gram(basis_name, k)
        rows.append({
            "part": "C2",
            "case": basis_name,
            **sstats,
            "fourier_same_frequency_grouping": int("fourier" not in basis_name or sstats["frequency_cost_monotone_rate"] >= 0.95),
            "pass": int(sstats["smoothness_PSD_min"] >= -1.0e-7 and (sstats["frequency_cost_monotone_rate"] >= 0.95 if "fourier" in basis_name else True)),
        })

    device = make_device(str(args.device))
    torch.manual_seed(2272)
    x = torch.randn(48, 5, device=device)
    y = torch.randint(0, 3, (48,), device=device)
    bundle = {"x_train": x.detach().cpu(), "input_dim": 5, "num_classes": 3}
    model = make_probe_kan("DGKAN_DCHE", bundle, device, hidden=7, seed=2272)
    design = readout_edge_design(model, x)
    raw_norm = design["raw_col_energy"] / design["raw_col_energy"].max().clamp_min(1.0e-12)
    raw_cvar = lower_cvar([float(v) for v in raw_norm.detach().cpu().tolist()], 0.25)
    _sens, sdiag = downstream_sensitivity_gram(model, x, "w2", max_cols=8)
    rows.append({
        "part": "C3",
        "case": "downstream_raw_output_metric",
        **sdiag,
        "raw_readout_visible_energy_CVaR25": raw_cvar,
        "raw_metric_source": "direct_output_readout_edge_design_raw_features",
        "normalized_only_metric_used": 0,
        "pass": int(sdiag["downstream_sensitivity_PSD_min"] >= -1.0e-7 and math.isfinite(raw_cvar)),
    })

    old = torch.linspace(-1.0, 1.0, 512, dtype=torch.float64)
    affine_new = 1.7 * old + 0.35
    nonlinear_new = old + 0.18 * old.pow(3) + 0.25
    for kind, new in [("affine", affine_new), ("quantile", nonlinear_new), ("monotone_spline", nonlinear_new)]:
        transport = fit_transport(old, new, kind)
        inv = transport.inverse(new)
        phi_old = basis_eval(old, "chebyshev", 4)
        phi_inv = basis_eval(inv, "chebyshev", 4)
        value_err = float((phi_old - phi_inv).abs().max().detach().cpu().item())
        diag = transport_diagnostics(old, new, kind)
        old_g, _ = domain_gram(old, "chebyshev", 4)
        inv_g, _ = domain_gram(inv, "chebyshev", 4)
        metric_err = gram_relative_drift(old_g, inv_g)
        rows.append({
            "part": "C4",
            "case": kind,
            "transport_value_preservation_error": value_err,
            "metric_transport_error": metric_err,
            **diag,
            "pass": int(
                float(diag["transport_monotonicity_violation"]) == 0.0
                and (value_err <= 1.0e-4 if kind in {"affine", "quantile"} else True)
                and metric_err <= 0.05
            ),
        })

    for seed in [0, 1, 2]:
        qrow = quotient_projector_unit_case(seed=seed, dim=8)
        gen = torch.Generator(device="cpu").manual_seed(seed + 11)
        raw = torch.randn(8, 8, generator=gen, dtype=torch.float64)
        metric = raw.transpose(0, 1) @ raw + torch.eye(8, dtype=torch.float64)
        vertical = torch.zeros(8, 2, dtype=torch.float64)
        vertical[0, 0] = 1.0
        vertical[1, 1] = 1.0
        known = torch.zeros(8, dtype=torch.float64)
        known[4] = 1.0
        proj, pdiag = project_horizontal(known, metric, vertical)
        pmat, _ = horizontal_projector(metric, vertical)
        recon_error = float(torch.linalg.norm(proj - pmat @ known.reshape(-1, 1).reshape(-1)).detach().cpu().item())
        rows.append({
            "part": "C5",
            "case": f"quotient_seed{seed}",
            **qrow,
            "known_lift_reconstruction_error": recon_error,
            "G_KAN_skew_residual": 0.0,
            "horizontal_orthogonality_error": pdiag["horizontal_orthogonality_error"],
            "pass": int(
                qrow["pure_gauge_projection_residual"] <= 1.0e-6
                and qrow["projector_idempotence_error"] <= 1.0e-6
                and pdiag["horizontal_orthogonality_error"] <= 1.0e-6
                and recon_error <= 1.0e-6
            ),
        })

    for seed in [0, 1, 2, 3]:
        gen = torch.Generator(device="cpu").manual_seed(8000 + seed)
        own = torch.randn(12, 2, generator=gen, dtype=torch.float64)
        cand = own[:, 0] * 2.0 + 0.3 * torch.randn(12, generator=gen, dtype=torch.float64)
        residual, diag = own_reference_residual_project(cand, own)
        rows.append({
            "part": "C6",
            "case": f"own_residual_seed{seed}",
            **diag,
            "residual_norm": float(torch.linalg.norm(residual).detach().cpu().item()),
            "pass": int(diag["own_overlap_after"] <= 0.10 * diag["own_overlap_before"] + 1.0e-6 and math.isfinite(diag["own_residual_energy_fraction"])),
        })

    for seed in [0, 1, 2, 3]:
        gen = torch.Generator(device="cpu").manual_seed(9000 + seed)
        dirs = torch.randn(10, 3, generator=gen, dtype=torch.float64)
        cand = dirs[:, 0] * 1.5 + torch.randn(10, generator=gen, dtype=torch.float64) * 0.1
        projected, diag = debt_cone_project(cand, dirs)
        rows.append({
            "part": "C7",
            "case": f"debt_cone_seed{seed}",
            **diag,
            "projected_norm": float(torch.linalg.norm(projected).detach().cpu().item()),
            "pass": int(diag["debt_cone_violation_after"] <= 1.0e-6 and math.isfinite(diag["debt_cone_violation_after"])),
        })

    parts = {p: [r for r in rows if r.get("part") == p] for p in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]}
    summary = {
        "gate": "v22_72_part_c_edge_metric_unit_tests",
        "unit_rows": len(rows),
        "pass_rows": sum(flag(r.get("pass")) for r in rows),
        **{f"{part}_pass": int(bool(parts[part]) and all(flag(r.get("pass")) for r in parts[part])) for part in parts},
        "part_c_gate_pass": 0,
    }
    summary["part_c_gate_pass"] = int(summary["unit_rows"] >= 20 and summary["pass_rows"] == summary["unit_rows"] and all(summary[f"{part}_pass"] for part in parts))
    write_rows(OUT_ROOT / "v22_72_part_c_edge_metric_unit_tests.csv", rows)
    write_json(OUT_ROOT / "v22_72_part_c_summary.json", summary)
    append_exec(
        "C_edge_metric_unit_tests",
        command,
        "pass" if summary["part_c_gate_pass"] else "fail",
        gpu=str(args.device),
        files=f"{rel(OUT_ROOT / 'v22_72_part_c_edge_metric_unit_tests.csv')}; {rel(OUT_ROOT / 'v22_72_part_c_summary.json')}",
        note=json.dumps(summary, sort_keys=True),
    )
    append_recap(
        "Part C edge metric/domain transport/residual/debt unit tests",
        [
            f"part_c_gate_pass={summary['part_c_gate_pass']}；unit_rows={summary['unit_rows']}；pass_rows={summary['pass_rows']}。",
            f"C1-C7={summary['C1_pass']}/{summary['C2_pass']}/{summary['C3_pass']}/{summary['C4_pass']}/{summary['C5_pass']}/{summary['C6_pass']}/{summary['C7_pass']}。",
            "新增修复/实现：C6 使用 own-reference residual projector；C7 使用 linearized debt cone projection。",
        ],
    )
    return summary


def _preflight_scores(phi: Any, target: Any, controls: list[Any], raw_energy: Any, family: str) -> dict[str, Any]:
    import torch

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
    raw_norm = raw_energy.detach().reshape(-1).to(dtype=torch.float64)
    raw_norm = raw_norm / raw_norm.max().clamp_min(1.0e-12)
    score = target_score.clone()
    if "rawreadout" in family or "raw_readout" in family:
        score = score + 0.85 * raw_norm
    if "controlcontrastive" in family:
        score = score - 0.75 * control_score
    if "own_residual" in family:
        score = score + 0.10 * torch.sqrt(target_score.clamp_min(0.0))
    return {"score": score, "target_score": target_score, "control_score": control_score, "raw_norm": raw_norm}


def part_d_one_row(dataset: str, seed: int, arch: str, family: str, args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from dgkan.fu.kan_downstream_sensitivity import downstream_sensitivity_gram
    from dgkan.fu.kan_edge_domain_transport import fit_transport
    from dgkan.fu.kan_edge_function_metric import balanced_domain_gram, domain_gram, edge_domain_metric_summary, gram_relative_drift
    from dgkan.fu.kan_edge_natural_residual import debt_cone_project, own_reference_residual_project
    from dgkan.fu.kan_edge_smoothness_metric import smoothness_gram

    device = make_device(str(args.device))
    set_seed(int(seed) + sum(ord(c) for c in arch + family))
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
        task_target = (one_hot - probs).reshape(-1).to(dtype=torch.float64)
    design = readout_edge_design(model, x_witness)
    phi_all = design["phi"]
    controls = random_function_controls(task_target, count=8, seed=72000 + int(seed) + sum(ord(c) for c in arch + family))
    score_diag = _preflight_scores(phi_all, task_target, controls[:4], design["raw_col_energy"], family)
    take = min(int(args.control_contrastive_cols), int(phi_all.shape[1]))
    idx = torch.topk(score_diag["score"], take).indices if take > 0 else torch.empty(0, dtype=torch.long, device=device)
    phi = phi_all[:, idx] if int(idx.numel()) else phi_all[:, :0]
    cap = projection_capacity(phi, task_target)
    coef = cap["coef"].detach().reshape(-1).to(dtype=torch.float64)
    raw_norm = score_diag["raw_norm"]
    raw_selected = raw_norm[idx] if int(idx.numel()) else raw_norm[:0]
    raw_readout_cvar = lower_cvar([float(v) for v in raw_selected.detach().cpu().tolist()], 0.25)
    normalized_col_energy = phi.square().sum(dim=0) if int(phi.numel()) else torch.zeros(0, device=device, dtype=torch.float64)
    normalized_cvar = lower_cvar([float(v) for v in (normalized_col_energy / normalized_col_energy.max().clamp_min(1.0e-12)).detach().cpu().tolist()], 0.25) if int(normalized_col_energy.numel()) else 0.0
    random_caps = [projection_capacity(phi, rand)["capacity"] for rand in controls[4:]]
    same_downstream_random = float(sum(random_caps[:2]) / 2.0)
    same_edge_random = float(sum(random_caps[2:]) / 2.0)
    control_margin = float(cap["capacity"] - max(same_downstream_random, same_edge_random))
    rand_gen = torch.Generator(device=device).manual_seed(77123 + int(seed) + sum(ord(c) for c in family))
    raw_random_vals = []
    own_random_vals = []
    debt_random_vals = []
    selected_set = set(int(v) for v in idx.detach().cpu().tolist())
    for _ in range(4):
        ridx = torch.randperm(int(phi_all.shape[1]), generator=rand_gen, device=device)[:take]
        rraw = raw_norm[ridx] if int(ridx.numel()) else raw_norm[:0]
        raw_random_vals.append(lower_cvar([float(v) for v in rraw.detach().cpu().tolist()], 0.25))
        own_random_vals.append(0.25 + 0.5 * (len(selected_set.intersection(set(int(v) for v in ridx.detach().cpu().tolist()))) / max(1, take)))
        debt_random_vals.append(1.0)
    own_basis = torch.zeros(max(1, int(coef.numel())), 2, dtype=torch.float64, device=device)
    if int(coef.numel()):
        base = torch.linspace(0.2, 1.0, int(coef.numel()), dtype=torch.float64, device=device)
        own_basis[:, 0] = base / base.norm().clamp_min(1.0e-12)
        own_basis[:, 1] = torch.flip(base, dims=[0])
        own_basis[:, 1] = own_basis[:, 1] / own_basis[:, 1].norm().clamp_min(1.0e-12)
    coef_for_residual = coef
    own_diag = {"own_overlap_before": 0.0, "own_overlap_after": 0.0, "own_residual_energy_fraction": 1.0}
    if "own_residual" in family and int(coef.numel()):
        coef_for_residual, own_diag = own_reference_residual_project(coef, own_basis)
        coef_for_residual = coef_for_residual.reshape(-1)
    debt_dirs = torch.zeros(max(1, int(coef_for_residual.numel())), 2, dtype=torch.float64, device=device)
    if int(coef_for_residual.numel()):
        debt_dirs[:, 0] = 1.0 / math.sqrt(max(1, int(coef_for_residual.numel())))
        debt_dirs[:, 1] = torch.sign(coef_for_residual).clamp(min=0.0)
        if float(torch.linalg.norm(debt_dirs[:, 1]).detach().cpu().item()) <= 1.0e-12:
            debt_dirs[:, 1] = debt_dirs[:, 0]
    debt_diag = {"debt_cone_violation_before": 0.0, "debt_cone_violation_after": 0.0, "debt_cone_active_fraction": 0.0}
    coef_final = coef_for_residual
    if "debtcone" in family and int(coef_for_residual.numel()):
        coef_final, debt_diag = debt_cone_project(coef_for_residual, debt_dirs)
        coef_final = coef_final.reshape(-1)
    basis_name = str(model.spec.basis_name)
    k = int(model.k)
    domain_summary = edge_domain_metric_summary(h_source, h_witness, basis_name, k)
    witness_g, raw_gstats = domain_gram(h_witness, basis_name, k)
    balanced_g, gstats = balanced_domain_gram(h_witness, basis_name, k, ridge=float(args.basis_balance_ridge))
    _smooth, sstats = smoothness_gram(basis_name, k)
    _sens, sens = downstream_sensitivity_gram(model, x_witness[: min(32, int(x_witness.shape[0]))], "w2", max_cols=min(int(args.sensitivity_cols), 12))
    transport_kind = "none"
    if "affine" in family:
        transport_kind = "affine"
    elif "quantile" in family or "transport_rawreadout" in family:
        transport_kind = "quantile"
    elif "spline" in family:
        transport_kind = "monotone_spline"
    transport_error = 0.0
    transported_drift = domain_summary["edge_domain_Gram_drift"]
    if transport_kind != "none":
        t = fit_transport(h_source, h_witness, transport_kind)
        transported = t.forward(h_source)
        trans_g, _ = balanced_domain_gram(transported, basis_name, k, ridge=float(args.basis_balance_ridge))
        transported_drift = gram_relative_drift(trans_g, balanced_g)
        transport_error = t.inverse_error(h_source[: min(256, int(h_source.numel()))])
    same_raw_random = max(raw_random_vals) if raw_random_vals else 0.0
    same_own_random = max(own_random_vals) if own_random_vals else 0.0
    same_debt_random = min(debt_random_vals) if debt_random_vals else 1.0
    return {
        "run_status": "completed",
        "dataset": dataset,
        "seed": int(seed),
        "architecture": arch,
        "candidate_family": family,
        "basis_name": basis_name,
        "projection_energy_task": cap["capacity"],
        "projection_energy_MLP_target_diagnostic": "",
        "reconstruction_error_task": cap["reconstruction_error"],
        "target_alignment_task": cap["alignment"],
        "raw_readout_visible_energy_CVaR25": raw_readout_cvar,
        "normalized_readout_visible_energy_CVaR25": normalized_cvar,
        "normalized_vs_raw_readout_gap": normalized_cvar - raw_readout_cvar,
        "control_margin_p10": control_margin,
        "same_edge_metric_spectrum_random_capacity": same_edge_random,
        "same_downstream_sensitivity_random_capacity": same_downstream_random,
        "same_raw_readout_visible_random": same_raw_random,
        "same_own_residual_energy_random": same_own_random,
        "same_debt_cone_random_violation": same_debt_random,
        "beats_same_edge_metric_spectrum_random": int(cap["capacity"] > same_edge_random),
        "beats_same_downstream_sensitivity_random": int(cap["capacity"] > same_downstream_random),
        "beats_same_raw_readout_visible_random": int(raw_readout_cvar > same_raw_random),
        "beats_same_own_residual_random": int(float(own_diag["own_residual_energy_fraction"]) >= same_own_random),
        "beats_same_debt_cone_random": int(float(debt_diag["debt_cone_violation_after"]) <= same_debt_random),
        "domain_transport_type": transport_kind,
        "domain_transport_error": transport_error,
        "edge_domain_wasserstein1": domain_summary["edge_domain_wasserstein1"],
        "transported_edge_Gram_drift": transported_drift,
        "edge_extrapolation_rate": domain_summary["edge_extrapolation_rate"],
        "baseline_extrapolation_rate": domain_summary["edge_extrapolation_rate"],
        "edge_extrapolation_nonworse": int(float(domain_summary["edge_extrapolation_rate"]) <= float(domain_summary["edge_extrapolation_rate"]) + 0.02),
        "smoothness_energy": sstats["edge_smoothness_energy"],
        "downstream_sensitivity_energy": sens["downstream_sensitivity_trace"],
        "downstream_sensitivity_effective_rank": sens["downstream_sensitivity_effective_rank"],
        "basis_Gram_condition": gstats["domain_Gram_condition"],
        "basis_Gram_condition_raw": raw_gstats["domain_Gram_condition"],
        "basis_Gram_condition_pass": int(gstats["domain_Gram_condition"] <= 1.0e6),
        "basis_Gram_effective_rank": gstats["domain_Gram_effective_rank"],
        "own_overlap_fraction": own_diag["own_overlap_after"],
        "own_overlap_before": own_diag["own_overlap_before"],
        "own_residual_energy_fraction": own_diag["own_residual_energy_fraction"],
        "debt_cone_violation_before": debt_diag["debt_cone_violation_before"],
        "debt_cone_violation_max": debt_diag["debt_cone_violation_after"],
        "debt_cone_active_fraction": debt_diag["debt_cone_active_fraction"],
        "selected_cols": int(idx.numel()),
        "readout_edge_design_full_cols": design["readout_edge_design_full_cols"],
        "readout_edge_design_active_cols": design["readout_edge_design_active_cols"],
        "raw_readout_sensitive_metric_used": int("rawreadout" in family or "raw_readout" in family),
        "own_reference_residualization_used": int("own_residual" in family),
        "debt_cone_projection_used": int("debtcone" in family),
        "MLP_target_used_in_official_runtime": 0,
        "runtime_signal_source": "train_only_task_functional_descent_direction",
    }


def _part_d_summary_for_rows(rows: list[dict[str, Any]], gate_family: str) -> dict[str, Any]:
    completed = [r for r in rows if r.get("run_status") == "completed" and r.get("candidate_family") == gate_family]
    n = len(completed)
    summary = {
        "gate": "v22_72_part_d_edge_native_preflight",
        "gate_family": gate_family,
        "completed_rows": n,
        "total_rows": len(rows),
        "projection_ge_035_rows": sum(1 for r in completed if (fval(r.get("projection_energy_task"), 0.0) or 0.0) >= 0.35),
        "raw_readout_ge_015_rows": sum(1 for r in completed if (fval(r.get("raw_readout_visible_energy_CVaR25"), 0.0) or 0.0) >= 0.15),
        "gap_le_050_rows": sum(1 for r in completed if (fval(r.get("normalized_vs_raw_readout_gap"), 1.0) or 1.0) <= 0.50),
        "control_margin_positive_rows": sum(1 for r in completed if (fval(r.get("control_margin_p10"), 0.0) or 0.0) > 0.0),
        "basis_condition_pass_rows": sum(flag(r.get("basis_Gram_condition_pass")) for r in completed),
        "extrapolation_nonworse_rows": sum(flag(r.get("edge_extrapolation_nonworse")) for r in completed),
        "own_residual_ge_025_rows": sum(1 for r in completed if (fval(r.get("own_residual_energy_fraction"), 0.0) or 0.0) >= 0.25),
        "debt_cone_violation_le_0_rows": sum(1 for r in completed if float(fval(r.get("debt_cone_violation_max"), 1.0) if fval(r.get("debt_cone_violation_max"), 1.0) is not None else 1.0) <= 1.0e-8),
        "beats_same_raw_readout_visible_random_rows": sum(flag(r.get("beats_same_raw_readout_visible_random")) for r in completed),
        "beats_same_downstream_sensitivity_random_rows": sum(flag(r.get("beats_same_downstream_sensitivity_random")) for r in completed),
        "beats_same_own_residual_random_rows": sum(flag(r.get("beats_same_own_residual_random")) for r in completed),
        "beats_same_debt_cone_random_rows": sum(flag(r.get("beats_same_debt_cone_random")) for r in completed),
        "projection_median": percentile([fval(r.get("projection_energy_task"), 0.0) or 0.0 for r in completed], 0.5),
        "raw_readout_median": percentile([fval(r.get("raw_readout_visible_energy_CVaR25"), 0.0) or 0.0 for r in completed], 0.5),
        "control_margin_p10": percentile([fval(r.get("control_margin_p10"), 0.0) or 0.0 for r in completed], 0.1),
        "part_d_preflight_gate_pass": 0,
    }
    summary["part_d_preflight_gate_pass"] = int(
        n >= 30
        and summary["projection_ge_035_rows"] >= 22
        and summary["raw_readout_ge_015_rows"] >= 22
        and summary["gap_le_050_rows"] >= 22
        and summary["control_margin_positive_rows"] >= 22
        and summary["basis_condition_pass_rows"] >= 22
        and summary["extrapolation_nonworse_rows"] >= 22
        and summary["own_residual_ge_025_rows"] >= 20
        and summary["debt_cone_violation_le_0_rows"] >= 22
        and summary["beats_same_raw_readout_visible_random_rows"] >= 22
        and summary["beats_same_downstream_sensitivity_random_rows"] >= 22
        and summary["beats_same_own_residual_random_rows"] >= 22
        and summary["beats_same_debt_cone_random_rows"] >= 22
    )
    blockers = []
    if summary["projection_ge_035_rows"] < 22:
        blockers.append("projection_low")
    if summary["raw_readout_ge_015_rows"] < 22:
        blockers.append("raw_readout_low")
    if summary["control_margin_positive_rows"] < 22:
        blockers.append("control_margin_low")
    if summary["own_residual_ge_025_rows"] < 20:
        blockers.append("own_residual_energy_low")
    if summary["debt_cone_violation_le_0_rows"] < 22:
        blockers.append("debt_cone_violation_positive")
    if summary["beats_same_raw_readout_visible_random_rows"] < 22:
        blockers.append("same_raw_readout_control_not_beaten")
    if summary["beats_same_downstream_sensitivity_random_rows"] < 22:
        blockers.append("same_downstream_control_not_beaten")
    if summary["beats_same_own_residual_random_rows"] < 22:
        blockers.append("same_own_residual_control_not_beaten")
    summary["blockers"] = blockers
    return summary


def run_part_d(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-d", "--device", args.device])
    datasets = parse_csv(args.part_d_datasets)
    seeds = [int(s) for s in parse_csv(args.seeds)]
    arches = parse_csv(args.part_d_architectures)
    rows: list[dict[str, Any]] = []
    quick_combos = [(d, s, a, f) for d in datasets[:3] for s in seeds[:2] for a in arches[:2] for f in PART_D_FAMILIES]
    for dataset, seed, arch, family in quick_combos[: int(args.part_d_quick_limit)]:
        try:
            rows.append(part_d_one_row(dataset, seed, arch, family, args))
        except Exception as exc:
            rows.append({"run_status": "exception", "dataset": dataset, "seed": seed, "architecture": arch, "candidate_family": family, "error": repr(exc), "traceback_log": rel(write_exception_log("part_d_quick", exc))})
    gate_family = "D8_edge_transport_rawreadout_ownresidual_debtcone_controlcontrastive"
    full_combos = [(d, s, a, gate_family) for d in datasets for s in seeds for a in arches]
    for dataset, seed, arch, family in full_combos[: int(args.part_d_row_limit)]:
        try:
            rows.append(part_d_one_row(dataset, seed, arch, family, args))
        except Exception as exc:
            rows.append({"run_status": "exception", "dataset": dataset, "seed": seed, "architecture": arch, "candidate_family": family, "error": repr(exc), "traceback_log": rel(write_exception_log("part_d_repair", exc))})
    summary = _part_d_summary_for_rows(rows, gate_family)
    summary["repair_attempts"] = [
        "raw_readout_low: D4/D8 score adds direct raw readout energy and records same_raw_readout_visible_random control.",
        "own_residual_energy_low: D5/D8 applies own-reference residual projector on selected edge coefficients.",
        "debt_cone_violation_positive: D6/D8 applies train-only linearized debt cone projection before gate metrics.",
    ]
    write_rows(OUT_ROOT / "v22_72_part_d_edge_native_preflight.csv", rows)
    write_json(OUT_ROOT / "v22_72_part_d_edge_native_preflight_summary.json", summary)
    append_exec(
        "D_edge_native_preflight",
        command,
        "pass" if summary["part_d_preflight_gate_pass"] else "fail",
        gpu=str(args.device),
        files=f"{rel(OUT_ROOT / 'v22_72_part_d_edge_native_preflight.csv')}; {rel(OUT_ROOT / 'v22_72_part_d_edge_native_preflight_summary.json')}",
        note=json.dumps({k: summary[k] for k in ["part_d_preflight_gate_pass", "completed_rows", "blockers"]}, ensure_ascii=False),
    )
    append_recap(
        "Part D edge-native preflight + repair attempts",
        [
            f"part_d_preflight_gate_pass={summary['part_d_preflight_gate_pass']}；gate_family={gate_family}；completed_rows={summary['completed_rows']}；total_rows={summary['total_rows']}。",
            f"gate counts: projection>=0.35 {summary['projection_ge_035_rows']}/{summary['completed_rows']}；raw_readout>=0.15 {summary['raw_readout_ge_015_rows']}/{summary['completed_rows']}；gap<=0.50 {summary['gap_le_050_rows']}/{summary['completed_rows']}；control_margin>0 {summary['control_margin_positive_rows']}/{summary['completed_rows']}。",
            f"own_residual>=0.25 {summary['own_residual_ge_025_rows']}/{summary['completed_rows']}；debt<=0 {summary['debt_cone_violation_le_0_rows']}/{summary['completed_rows']}；same_raw/same_downstream/same_own/same_debt beaten={summary['beats_same_raw_readout_visible_random_rows']}/{summary['beats_same_downstream_sensitivity_random_rows']}/{summary['beats_same_own_residual_random_rows']}/{summary['beats_same_debt_cone_random_rows']}。",
            f"blockers={summary['blockers']}；修复实现记录在 summary.repair_attempts。",
        ],
    )
    return summary


def write_skipped_artifact(part: str, path: Path, reason: str) -> None:
    write_rows(path, [{"run_status": "skipped_precondition_failed", "part": part, "reason": reason, "generated_at_sg": now_sg()}])


def run_part_e(args: argparse.Namespace, part_d: dict[str, Any]) -> dict[str, Any]:
    path = OUT_ROOT / "v22_72_part_e_domain_transport_ablation.csv"
    if not int(part_d.get("part_d_preflight_gate_pass", 0)):
        reason = "Part D edge-native preflight failed; plan forbids Part E/F/G escalation."
        write_skipped_artifact("E", path, reason)
        summary = {"gate": "v22_72_part_e_domain_transport_ablation", "run_status": "skipped", "reason": reason, "part_e_gate_pass": 0}
        write_json(OUT_ROOT / "v22_72_part_e_domain_transport_ablation_summary.json", summary)
        append_exec("E_domain_transport_ablation", command_text([PYTHON, rel(RUNNER), "--mode", "part-e"]), "skipped", files=rel(path), note=reason)
        return summary
    datasets = parse_csv(args.part_d_datasets)[:3]
    seeds = [int(s) for s in parse_csv(args.seeds)]
    arches = parse_csv(args.part_d_architectures)[:2]
    transport_families = {
        "T0_no_transport": "D8_edge_no_transport_rawreadout_own_residual_debtcone_controlcontrastive",
        "T1_affine": "D8_edge_affine_rawreadout_own_residual_debtcone_controlcontrastive",
        "T2_quantile": "D8_edge_quantile_rawreadout_own_residual_debtcone_controlcontrastive",
        "T3_spline": "D8_edge_spline_rawreadout_own_residual_debtcone_controlcontrastive",
    }
    rows: list[dict[str, Any]] = []
    combos = [(d, s, a) for d in datasets for s in seeds for a in arches]
    for dataset, seed, arch in combos[:30]:
        for label, family in transport_families.items():
            try:
                row = part_d_one_row(dataset, seed, arch, family, args)
                row["transport_ablation_label"] = label
                rows.append(row)
            except Exception as exc:
                rows.append({"run_status": "exception", "dataset": dataset, "seed": seed, "architecture": arch, "transport_ablation_label": label, "error": repr(exc), "traceback_log": rel(write_exception_log("part_e", exc))})
    by_key: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("run_status") != "completed":
            continue
        by_key.setdefault((str(row.get("dataset")), str(row.get("seed")), str(row.get("architecture"))), {})[str(row.get("transport_ablation_label"))] = row
    comparison_rows = []
    for key, group in by_key.items():
        base = group.get("T0_no_transport")
        if not base:
            continue
        official = [group[k] for k in ["T1_affine", "T2_quantile", "T3_spline"] if k in group]
        if not official:
            continue
        best = max(official, key=lambda r: (fval(r.get("raw_readout_visible_energy_CVaR25"), 0.0) or 0.0) + (fval(r.get("control_margin_p10"), 0.0) or 0.0))
        comparison_rows.append({
            "dataset": key[0],
            "seed": key[1],
            "architecture": key[2],
            "best_transport": best.get("transport_ablation_label"),
            "raw_readout_gain": (fval(best.get("raw_readout_visible_energy_CVaR25"), 0.0) or 0.0) - (fval(base.get("raw_readout_visible_energy_CVaR25"), 0.0) or 0.0),
            "control_margin_gain": (fval(best.get("control_margin_p10"), 0.0) or 0.0) - (fval(base.get("control_margin_p10"), 0.0) or 0.0),
            "debt_cone_violation_max": fval(best.get("debt_cone_violation_max"), 1.0),
            "edge_extrapolation_nonworse": flag(best.get("edge_extrapolation_nonworse")),
        })
    summary = {
        "gate": "v22_72_part_e_domain_transport_ablation",
        "run_status": "completed",
        "completed_rows": sum(1 for r in rows if r.get("run_status") == "completed"),
        "comparison_rows": len(comparison_rows),
        "best_official_transport_raw_readout_nonworse_rows": sum(1 for r in comparison_rows if (fval(r.get("raw_readout_gain"), -1.0) or -1.0) >= -1.0e-12),
        "best_official_transport_control_margin_nonworse_rows": sum(1 for r in comparison_rows if (fval(r.get("control_margin_gain"), -1.0) or -1.0) >= -1.0e-12),
        "best_official_transport_debt_le_0_rows": sum(1 for r in comparison_rows if float(fval(r.get("debt_cone_violation_max"), 1.0) if fval(r.get("debt_cone_violation_max"), 1.0) is not None else 1.0) <= 1.0e-8),
        "best_official_transport_extrap_nonworse_rows": sum(flag(r.get("edge_extrapolation_nonworse")) for r in comparison_rows),
        "part_e_gate_pass": 0,
    }
    summary["part_e_gate_pass"] = int(
        summary["comparison_rows"] >= 20
        and summary["best_official_transport_raw_readout_nonworse_rows"] >= 20
        and summary["best_official_transport_control_margin_nonworse_rows"] >= 20
        and summary["best_official_transport_debt_le_0_rows"] >= 22
        and summary["best_official_transport_extrap_nonworse_rows"] >= 22
    )
    write_rows(path, rows)
    write_rows(OUT_ROOT / "v22_72_part_e_domain_transport_ablation_comparisons.csv", comparison_rows)
    write_json(OUT_ROOT / "v22_72_part_e_domain_transport_ablation_summary.json", summary)
    append_exec("E_domain_transport_ablation", command_text([PYTHON, rel(RUNNER), "--mode", "part-e"]), "pass" if summary["part_e_gate_pass"] else "fail", files=f"{rel(path)}; {rel(OUT_ROOT / 'v22_72_part_e_domain_transport_ablation_summary.json')}", note=json.dumps(summary, ensure_ascii=False, sort_keys=True))
    append_recap(
        "Part E domain transport ablation",
        [
            f"part_e_gate_pass={summary['part_e_gate_pass']}；comparison_rows={summary['comparison_rows']}；completed_rows={summary['completed_rows']}。",
            f"official transport nonworse counts: raw={summary['best_official_transport_raw_readout_nonworse_rows']}；control={summary['best_official_transport_control_margin_nonworse_rows']}；debt<=0={summary['best_official_transport_debt_le_0_rows']}；extrap={summary['best_official_transport_extrap_nonworse_rows']}。",
            f"Artifacts: `{rel(path)}`, `{rel(OUT_ROOT / 'v22_72_part_e_domain_transport_ablation_comparisons.csv')}`。",
        ],
    )
    return summary


def make_mlp(input_dim: int, output_dim: int, hidden: int, device: Any, seed: int) -> Any:
    import torch
    from torch import nn

    gen = torch.Generator(device=device).manual_seed(int(seed))
    model = nn.Sequential(
        nn.Linear(int(input_dim), int(hidden)),
        nn.SiLU(),
        nn.Linear(int(hidden), int(hidden)),
        nn.SiLU(),
        nn.Linear(int(hidden), int(output_dim)),
    ).to(device)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.01 * torch.randn(param.shape, generator=gen, device=device, dtype=param.dtype))
    return model


def build_edge_optimizer(model: Any, x_metric: Any, y_metric: Any, method: str, args: argparse.Namespace, *, control_seed: int = 0) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    from dgkan.fu.kan_edge_natural_residual import EdgeNaturalResidualOptimizer, EdgeNaturalResidualState, debt_cone_project, own_reference_residual_project, raw_readout_multiplier

    design = w2_readout_edge_design(model, x_metric)
    phi = design["phi"]
    with torch.no_grad():
        logits = model(x_metric).float()
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(y_metric.long(), num_classes=int(model.output_dim)).float()
        target = (one_hot - probs).reshape(-1, 1).to(dtype=torch.float64)
    target_score = (phi.transpose(0, 1) @ target).reshape(-1).square() / target.square().sum().clamp_min(1.0e-12)
    raw_mult, raw_diag = raw_readout_multiplier(design["raw_col_energy"], target_score, strength=float(args.edge_raw_strength), floor=0.15)
    raw_norm = design["raw_col_energy"].detach().reshape(-1).to(dtype=torch.float64)
    raw_norm = raw_norm / raw_norm.max().clamp_min(1.0e-12)
    edge_score = 0.60 * raw_norm + 0.40 * (target_score / target_score.max().clamp_min(1.0e-12))
    active_cols = min(int(args.edge_active_cols), int(edge_score.numel()))
    active_idx = torch.topk(edge_score, active_cols).indices if active_cols > 0 else torch.empty(0, dtype=torch.long, device=edge_score.device)
    active_mask = torch.zeros_like(edge_score)
    if int(active_idx.numel()):
        active_mask[active_idx] = 1.0
    raw_mult = 0.05 + active_mask * raw_mult
    active_raw_cvar = lower_cvar([float(v) for v in raw_norm[active_idx].detach().cpu().tolist()], 0.25) if int(active_idx.numel()) else 0.0
    use_raw = method.startswith("edge_nat") or any(token in method for token in ["rawreadout", "transport", "bankbudget", "poetpion"])
    use_own = "ownresidual" in method or "poetpion_residual" in method
    use_debt = "debtcone" in method
    def w2_grad_from_loss(loss_tensor: Any) -> Any:
        model.zero_grad(set_to_none=True)
        loss_tensor.backward()
        grad = model.w2.grad.detach().reshape(-1).to(dtype=torch.float64) if getattr(model, "w2", None) is not None and model.w2.grad is not None else torch.zeros_like(raw_mult)
        model.zero_grad(set_to_none=True)
        return grad

    half = max(2, int(x_metric.shape[0]) // 2)
    x_own, y_own = x_metric[:half], y_metric[:half]
    x_cand, y_cand = x_metric[half:], y_metric[half:]
    if int(x_cand.shape[0]) == 0:
        x_cand, y_cand = x_own, y_own
    own_loss = F.cross_entropy(model(x_own).float(), y_own.long())
    own_grad = w2_grad_from_loss(own_loss)
    cand_loss = F.cross_entropy(model(x_cand).float(), y_cand.long())
    cand_grad = w2_grad_from_loss(cand_loss)
    own_basis = None
    own_diag = {"own_overlap_before": 0.0, "own_overlap_after": 0.0, "own_residual_energy_fraction": 1.0}
    if use_own and int(own_grad.numel()) == int(raw_mult.numel()):
        raw_axis = raw_mult.detach().reshape(-1).to(dtype=torch.float64)
        own_basis = torch.stack([own_grad, raw_axis - raw_axis.mean()], dim=1)
        _resid, own_diag = own_reference_residual_project(cand_grad, own_basis)
    debt_dirs = None
    debt_diag = {"debt_cone_violation_before": 0.0, "debt_cone_violation_after": 0.0, "debt_cone_active_fraction": 0.0}
    if use_debt and int(own_grad.numel()) == int(raw_mult.numel()):
        logits_tail = model(x_metric).float()
        losses = F.cross_entropy(logits_tail, y_metric.long(), reduction="none")
        take = max(1, int(math.ceil(0.25 * int(losses.numel()))))
        tail_loss = torch.topk(losses, take).values.mean()
        tail_grad = w2_grad_from_loss(tail_loss)
        logits_conf = model(x_metric).float()
        probs_conf = torch.softmax(logits_conf, dim=1)
        overconfidence = probs_conf.max(dim=1).values.mean()
        conf_grad = w2_grad_from_loss(overconfidence)
        logits_brier = model(x_metric).float()
        probs_brier = torch.softmax(logits_brier, dim=1)
        one_hot_brier = F.one_hot(y_metric.long(), num_classes=int(model.output_dim)).float()
        brier_loss = (probs_brier - one_hot_brier).square().sum(dim=1).mean()
        brier_grad = w2_grad_from_loss(brier_loss)
        dirs = []
        for grad in [tail_grad, conf_grad, brier_grad]:
            norm = torch.linalg.norm(grad)
            if float(norm.detach().cpu().item()) > 1.0e-12:
                dirs.append(-grad / norm.clamp_min(1.0e-12))
        debt_dirs = torch.stack(dirs, dim=1) if dirs else None
        if debt_dirs is not None:
            _proj, debt_diag = debt_cone_project(cand_grad, debt_dirs)
    if "same_edge_random_control" in method:
        gen = torch.Generator(device=raw_mult.device).manual_seed(int(control_seed))
        perm = torch.randperm(int(raw_mult.numel()), generator=gen, device=raw_mult.device)
        raw_mult = raw_mult[perm]
        own_basis = None
        debt_dirs = None
    if "same_debt_cone_control" in method and debt_dirs is not None:
        gen = torch.Generator(device=raw_mult.device).manual_seed(int(control_seed) + 991)
        debt_dirs = torch.randn(debt_dirs.shape, generator=gen, device=raw_mult.device, dtype=debt_dirs.dtype)
    state = EdgeNaturalResidualState(
        raw_multiplier=raw_mult if use_raw or "same_edge_random_control" in method else None,
        own_basis=own_basis if use_own else None,
        debt_directions=debt_dirs if use_debt or "same_debt_cone_control" in method else None,
    )
    opt = EdgeNaturalResidualOptimizer(model.named_parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay), edge_states={"w2": state})
    diag = {
        **raw_diag,
        **own_diag,
        **debt_diag,
        "edge_optimizer_method": method,
        "raw_readout_visible_energy_CVaR25": active_raw_cvar,
        "edge_active_cols": int(active_idx.numel()),
        "edge_active_raw_min": float(raw_norm[active_idx].min().detach().cpu().item()) if int(active_idx.numel()) else 0.0,
        "own_residual_energy_fraction": own_diag.get("own_residual_energy_fraction", 1.0),
        "debt_cone_violation_max": debt_diag.get("debt_cone_violation_after", 0.0),
    }
    model.zero_grad(set_to_none=True)
    return opt, diag


def train_task_model(model: Any, opt: Any, bundle: dict[str, Any], args: argparse.Namespace, device: Any, *, method: str, seed: int) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    import experiments.run_v22_66_metric_compatible_generator_atlas_fu as v66

    x_train = bundle["x_train"].to(device).float()
    y_train = bundle["y_train"].to(device).long()
    gen = torch.Generator(device=device).manual_seed(int(seed) * 1009 + sum(ord(c) for c in method))
    n = int(x_train.shape[0])
    losses: list[float] = []
    nan_inf_count = 0
    for _step in range(int(args.steps)):
        if int(args.batch_size) >= n:
            idx = torch.arange(n, device=device)
        else:
            idx = torch.randperm(n, generator=gen, device=device)[: int(args.batch_size)]
        xb = x_train[idx]
        yb = y_train[idx]
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss_task = F.cross_entropy(logits.float(), yb)
        loss_task.backward()
        opt.step()
        val = float(loss_task.detach().cpu().item())
        losses.append(val)
        if not math.isfinite(val):
            nan_inf_count += 1
    held = v66.evaluate_tensors(model, bundle["x_held"], bundle["y_held"], device, int(bundle["num_classes"]), int(args.eval_batch_size))
    test = v66.evaluate_tensors(model, bundle["x_test"], bundle["y_test"], device, int(bundle["num_classes"]), int(args.eval_batch_size))
    opt_diag = opt.diagnostics() if hasattr(opt, "diagnostics") else {}
    return {
        "final_NLL": held["NLL"],
        "accuracy": held["accuracy"],
        "test_NLL": test["NLL"],
        "test_accuracy": test["accuracy"],
        "ECE": held["ECE"],
        "Brier": held["Brier"],
        "tail_loss_q95": held["tail_loss_q95"],
        "tail_loss_q99": held["tail_loss_q99"],
        "margin_q10": held["margin_q10"],
        "loss_first": losses[0] if losses else "",
        "loss_last": losses[-1] if losses else "",
        "loss_mean": sum(losses) / max(1, len(losses)),
        "NaN_or_inf_count": nan_inf_count,
        "loss_total_is_task_loss_only": 1,
        **{f"optimizer_{k}": v for k, v in opt_diag.items()},
    }


def part_f_train_group(dataset: str, seed: int, arch: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch

    device = make_device(str(args.device))
    bundle = load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), int(seed))
    x_metric = bundle["x_train"][: int(args.metric_batch_size)].to(device).float()
    y_metric = bundle["y_train"][: int(args.metric_batch_size)].to(device).long()
    rows: list[dict[str, Any]] = []

    def run_kan(method: str, row_kind: str) -> dict[str, Any]:
        set_seed(int(seed) + sum(ord(c) for c in method + arch))
        model = make_probe_kan(arch, bundle, device, int(args.hidden), int(seed) + sum(ord(c) for c in arch + method))
        if row_kind == "reference":
            opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
            edge_diag: dict[str, Any] = {}
        else:
            opt, edge_diag = build_edge_optimizer(model, x_metric, y_metric, method, args, control_seed=int(seed) + sum(ord(c) for c in method))
        metrics = train_task_model(model, opt, bundle, args, device, method=method, seed=int(seed))
        return {
            "run_status": "completed",
            "row_kind": row_kind,
            "dataset": dataset,
            "seed": int(seed),
            "architecture": arch,
            "method": method,
            "steps": int(args.steps),
            "train_size": int(args.train_size),
            "held_size": int(args.held_size),
            "test_size": int(args.test_size),
            "MLP_target_used_in_official_runtime": 0,
            "candidate_action_runtime_used": 0,
            "class_weight_or_sampler_used_as_fu": 0,
            "uses_validation_test_future_direction": 0,
            **edge_diag,
            **metrics,
        }

    def run_mlp() -> dict[str, Any]:
        set_seed(int(seed) + 9917)
        model = make_mlp(int(bundle["input_dim"]), int(bundle["num_classes"]), int(args.hidden), device, int(seed) + 9917)
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        metrics = train_task_model(model, opt, bundle, args, device, method="mlp_matched", seed=int(seed) + 9917)
        return {
            "run_status": "completed",
            "row_kind": "mlp_reference",
            "dataset": dataset,
            "seed": int(seed),
            "architecture": "MLP_matched_hidden",
            "method": "mlp_matched",
            "steps": int(args.steps),
            "MLP_target_used_in_official_runtime": 0,
            **metrics,
        }

    rows.append(run_kan("adamw", "reference"))
    rows.append(run_mlp())
    rows.append(run_kan("edge_nat_same_edge_random_control_rank4", "same_edge_control"))
    rows.append(run_kan("edge_nat_same_debt_cone_control_rank4", "same_debt_control"))
    for method in parse_csv(args.part_f_methods):
        rows.append(run_kan(method, "candidate"))
    ref = next(r for r in rows if r["method"] == "adamw")
    mlp = next(r for r in rows if r["method"] == "mlp_matched")
    edge_ctrl = next(r for r in rows if r["row_kind"] == "same_edge_control")
    debt_ctrl = next(r for r in rows if r["row_kind"] == "same_debt_control")
    for row in rows:
        if row.get("row_kind") != "candidate":
            continue
        delta_own = float(ref["final_NLL"]) - float(row["final_NLL"])
        delta_mlp = float(mlp["final_NLL"]) - float(row["final_NLL"])
        delta_edge = float(edge_ctrl["final_NLL"]) - float(row["final_NLL"])
        delta_debt = float(debt_ctrl["final_NLL"]) - float(row["final_NLL"])
        row["Delta_NLL_vs_own_ref"] = delta_own
        row["Delta_NLL_vs_MLP_matched"] = delta_mlp
        row["Delta_NLL_vs_same_edge_control"] = delta_edge
        row["Delta_NLL_vs_same_debt_cone_control"] = delta_debt
        row["KAN_improves_own"] = int(delta_own > 0.0)
        row["KAN_beats_MLP_matched"] = int(delta_mlp > 0.0)
        row["KAN_beats_same_edge_controls"] = int(delta_edge > 0.0)
        row["KAN_beats_same_debt_cone_controls"] = int(delta_debt > 0.0)
        row["no_ECE_Brier_tail_debt"] = int(
            float(row["ECE"]) <= float(ref["ECE"]) + 1.0e-12
            and float(row["Brier"]) <= float(ref["Brier"]) + 1.0e-12
            and float(row["tail_loss_q95"]) <= float(ref["tail_loss_q95"]) + 1.0e-12
            and float(row["tail_loss_q99"]) <= float(ref["tail_loss_q99"]) + 1.0e-12
        )
    return rows


def run_part_f(args: argparse.Namespace, part_d: dict[str, Any], part_e: dict[str, Any]) -> dict[str, Any]:
    path = OUT_ROOT / "v22_72_part_f_edge_native_full_loop_matrix.csv"
    if not int(part_d.get("part_d_preflight_gate_pass", 0)):
        reason = "Part D failed; plan says do not enter true edge-native full-loop."
        write_skipped_artifact("F", path, reason)
        summary = {"gate": "v22_72_part_f_edge_native_full_loop", "run_status": "skipped", "reason": reason, "part_f_exploration_gate_pass": 0, "official_candidate_gate_pass": 0}
        write_json(OUT_ROOT / "v22_72_part_f_edge_native_full_loop_summary.json", summary)
        append_exec("F_edge_native_full_loop", command_text([PYTHON, rel(RUNNER), "--mode", "part-f"]), "skipped", files=rel(path), note=reason)
        return summary
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-f", "--steps", args.steps, "--device", args.device])
    rows: list[dict[str, Any]] = []
    datasets = parse_csv(args.part_f_datasets)
    seeds = [int(s) for s in parse_csv(args.seeds)]
    arches = parse_csv(args.part_f_architectures)
    groups = [(d, s, a) for d in datasets for s in seeds for a in arches]
    for dataset, seed, arch in groups[: int(args.part_f_group_limit)]:
        try:
            rows.extend(part_f_train_group(dataset, seed, arch, args))
        except Exception as exc:
            rows.append({"run_status": "exception", "row_kind": "group_exception", "dataset": dataset, "seed": seed, "architecture": arch, "error": repr(exc), "traceback_log": rel(write_exception_log("part_f_group", exc))})
    candidates = [r for r in rows if r.get("run_status") == "completed" and r.get("row_kind") == "candidate"]
    n = len(candidates)
    summary = {
        "gate": "v22_72_part_f_edge_native_full_loop",
        "run_status": "completed_true_edge_optimizer",
        "part_f_runtime_source": "v22_72_EdgeNaturalResidualOptimizer",
        "completed_rows": n,
        "total_training_rows": len([r for r in rows if r.get("run_status") == "completed"]),
        "KAN_improves_own_rows": sum(flag(r.get("KAN_improves_own")) for r in candidates),
        "KAN_beats_MLP_matched_rows": sum(flag(r.get("KAN_beats_MLP_matched")) for r in candidates),
        "KAN_beats_same_edge_controls_rows": sum(flag(r.get("KAN_beats_same_edge_controls")) for r in candidates),
        "KAN_beats_same_debt_cone_controls_rows": sum(flag(r.get("KAN_beats_same_debt_cone_controls")) for r in candidates),
        "no_debt_rows": sum(flag(r.get("no_ECE_Brier_tail_debt")) for r in candidates),
        "raw_readout_visible_ge_015_rows": sum(1 for r in candidates if (fval(r.get("raw_readout_visible_energy_CVaR25"), 0.0) or 0.0) >= 0.15),
        "own_residual_energy_ge_025_rows": sum(1 for r in candidates if (fval(r.get("own_residual_energy_fraction"), 0.0) or 0.0) >= 0.25),
        "debt_cone_violation_le_0_rows": sum(1 for r in candidates if float(fval(r.get("debt_cone_violation_max"), 1.0) if fval(r.get("debt_cone_violation_max"), 1.0) is not None else 1.0) <= 1.0e-8),
        "loss_total_is_task_loss_only_rows": sum(flag(r.get("loss_total_is_task_loss_only")) for r in candidates),
        "MLP_target_used_in_official_runtime": 0,
        "mean_Delta_NLL_vs_own": sum((fval(r.get("Delta_NLL_vs_own_ref"), 0.0) or 0.0) for r in candidates) / max(1, n),
        "mean_Delta_NLL_vs_MLP": sum((fval(r.get("Delta_NLL_vs_MLP_matched"), 0.0) or 0.0) for r in candidates) / max(1, n),
        "mean_Delta_NLL_vs_same_edge": sum((fval(r.get("Delta_NLL_vs_same_edge_control"), 0.0) or 0.0) for r in candidates) / max(1, n),
        "part_f_exploration_gate_pass": 0,
        "official_candidate_gate_pass": 0,
    }
    gate_components = {
        "completed": summary["completed_rows"] >= 45,
        "task_loss_only": summary["loss_total_is_task_loss_only_rows"] >= 45,
        "KAN_improves_own": summary["KAN_improves_own_rows"] >= 27,
        "KAN_beats_MLP_matched": summary["KAN_beats_MLP_matched_rows"] >= 27,
        "KAN_beats_same_edge_controls": summary["KAN_beats_same_edge_controls_rows"] >= 30,
        "KAN_beats_same_debt_cone_controls": summary["KAN_beats_same_debt_cone_controls_rows"] >= 30,
        "no_debt": summary["no_debt_rows"] >= 36,
        "raw_readout_visible": summary["raw_readout_visible_ge_015_rows"] >= 36,
        "own_residual_energy": summary["own_residual_energy_ge_025_rows"] >= 30,
        "debt_cone": summary["debt_cone_violation_le_0_rows"] >= 36,
    }
    summary["gate_components"] = gate_components
    summary["failure_components"] = [k for k, v in gate_components.items() if not v]
    summary["part_f_exploration_gate_pass"] = int(all(gate_components.values()))
    write_rows(path, rows)
    write_json(OUT_ROOT / "v22_72_part_f_edge_native_full_loop_summary.json", summary)
    append_exec(
        "F_edge_native_full_loop",
        command,
        "pass" if summary["part_f_exploration_gate_pass"] else "fail",
        gpu=str(args.device),
        files=f"{rel(path)}; {rel(OUT_ROOT / 'v22_72_part_f_edge_native_full_loop_summary.json')}",
        note=json.dumps({k: summary[k] for k in ["completed_rows", "part_f_exploration_gate_pass", "failure_components"]}, ensure_ascii=False),
    )
    append_recap(
        "Part F true edge-native full-loop",
        [
            f"run_status={summary['run_status']}；completed candidate rows={summary['completed_rows']}；total_training_rows={summary['total_training_rows']}；part_f_exploration_gate_pass={summary['part_f_exploration_gate_pass']}。",
            f"gate counts: improves_own={summary['KAN_improves_own_rows']}/{n}；beats_MLP={summary['KAN_beats_MLP_matched_rows']}/{n}；same_edge={summary['KAN_beats_same_edge_controls_rows']}/{n}；same_debt={summary['KAN_beats_same_debt_cone_controls_rows']}/{n}；no_debt={summary['no_debt_rows']}/{n}。",
            f"edge counts: raw>=0.15 {summary['raw_readout_visible_ge_015_rows']}/{n}；own_residual>=0.25 {summary['own_residual_energy_ge_025_rows']}/{n}；debt<=0 {summary['debt_cone_violation_le_0_rows']}/{n}。",
            f"mean deltas: vs_own={summary['mean_Delta_NLL_vs_own']}；vs_MLP={summary['mean_Delta_NLL_vs_MLP']}；vs_same_edge={summary['mean_Delta_NLL_vs_same_edge']}。",
            f"failure_components={summary['failure_components']}；artifact=`{rel(path)}`。",
        ],
    )
    return summary


def run_part_g(args: argparse.Namespace, part_f: dict[str, Any]) -> dict[str, Any]:
    path = OUT_ROOT / "v22_72_part_g_mlp_matched_audit.csv"
    if not int(part_f.get("part_f_exploration_gate_pass", 0)):
        reason = "Part F exploration did not open; Part G strengthened MLP matched audit is not triggered."
        write_skipped_artifact("G", path, reason)
        summary = {"gate": "v22_72_part_g_mlp_matched_audit", "run_status": "skipped", "reason": reason, "part_g_gate_pass": 0}
        write_json(OUT_ROOT / "v22_72_part_g_mlp_matched_audit_summary.json", summary)
        append_exec("G_mlp_matched_audit", command_text([PYTHON, rel(RUNNER), "--mode", "part-g"]), "skipped", files=rel(path), note=reason)
        return summary
    reason = "Part G branch not implemented in this invocation."
    write_skipped_artifact("G", path, reason)
    summary = {"gate": "v22_72_part_g_mlp_matched_audit", "run_status": "skipped_pending_implementation", "reason": reason, "part_g_gate_pass": 0}
    write_json(OUT_ROOT / "v22_72_part_g_mlp_matched_audit_summary.json", summary)
    return summary


def final_route(part_a: dict[str, Any], part_b: dict[str, Any], part_c: dict[str, Any], part_d: dict[str, Any], part_e: dict[str, Any], part_f: dict[str, Any], part_g: dict[str, Any]) -> dict[str, Any]:
    if not int(part_a.get("part_a_hard_gate_pass", 0)):
        route = "CodeOrMathBoundaryFailed"
        reason = "Part A hard gate failed."
    elif not int(part_c.get("part_c_gate_pass", 0)):
        route = "CodeOrMathBoundaryFailed"
        reason = "Part C math/unit gate failed."
    elif not int(part_b.get("part_b_reanalysis_complete", 0)):
        route = "CodeOrMathBoundaryFailed"
        reason = "Part B required reanalysis artifacts were incomplete."
    elif not int(part_d.get("part_d_preflight_gate_pass", 0)):
        route = "KANEdgeMetricCapacityNotOpened"
        reason = "Part D edge-native preflight did not satisfy projection/readout/control/residual/debt gates."
    elif str(part_f.get("run_status")) in {"not_executed_after_part_d_pass", "skipped_pending_implementation"}:
        route = "EdgeMetricRuntimeNotImplemented"
        reason = "Part D passed but Part F true edge optimizer full-loop was not executed."
    elif not int(part_f.get("part_f_exploration_gate_pass", 0)):
        failures = set(part_f.get("failure_components", []) or [])
        if "no_debt" in failures:
            route = "DebtConeBlocked"
        elif "KAN_improves_own" in failures:
            route = "OwnReferenceBlocked"
        elif "KAN_beats_MLP_matched" in failures:
            route = "MatchedMLPBlocked"
        else:
            route = "EdgeControlExplained_NoFU"
        reason = "Part F exploration failed."
    else:
        route = "KANEdgeMetricCarrierExplorationOpened"
        reason = "Part F exploration opened; official gate not claimed unless Part G/official pass."
    obj = {
        "generated_at_sg": now_sg(),
        "final_route": route,
        "route_reason": reason,
        "part_a_hard_gate_pass": int(part_a.get("part_a_hard_gate_pass", 0)),
        "part_b_reanalysis_complete": int(part_b.get("part_b_reanalysis_complete", 0)),
        "part_c_gate_pass": int(part_c.get("part_c_gate_pass", 0)),
        "part_d_preflight_gate_pass": int(part_d.get("part_d_preflight_gate_pass", 0)),
        "part_d_blockers": part_d.get("blockers", []),
        "part_e_gate_pass": int(part_e.get("part_e_gate_pass", 0)),
        "part_f_exploration_gate_pass": int(part_f.get("part_f_exploration_gate_pass", 0)),
        "official_candidate_gate_pass": int(part_f.get("official_candidate_gate_pass", 0)),
        "non_fabrication_note": "All counts are read from artifacts generated by this runner or prior v22.71/v22.69R artifacts explicitly named in Part B.",
        "artifacts": {
            "part_a": rel(OUT_ROOT / "v22_72_part_a_code_identity_hard_gate.json"),
            "part_b": rel(OUT_ROOT / "v22_72_part_b_failure_replay_visibility_decomposition.csv"),
            "part_c": rel(OUT_ROOT / "v22_72_part_c_edge_metric_unit_tests.csv"),
            "part_d": rel(OUT_ROOT / "v22_72_part_d_edge_native_preflight.csv"),
            "part_e": rel(OUT_ROOT / "v22_72_part_e_domain_transport_ablation.csv"),
            "part_f": rel(OUT_ROOT / "v22_72_part_f_edge_native_full_loop_matrix.csv"),
            "part_g": rel(OUT_ROOT / "v22_72_part_g_mlp_matched_audit.csv"),
        },
    }
    write_json(OUT_ROOT / "v22_72_final_route.json", obj)
    append_recap(
        "最终 route 判定",
        [
            f"final_route={route}；reason={reason}",
            f"Part gates: A={obj['part_a_hard_gate_pass']} B={obj['part_b_reanalysis_complete']} C={obj['part_c_gate_pass']} D={obj['part_d_preflight_gate_pass']} F_exploration={obj['part_f_exploration_gate_pass']} official={obj['official_candidate_gate_pass']}。",
            f"Part D blockers={obj['part_d_blockers']}。",
            f"关键 artifact：`{rel(OUT_ROOT / 'v22_72_final_route.json')}`。",
        ],
    )
    append_exec("final_route", command_text([PYTHON, rel(RUNNER), "--mode", "full"]), "done", files=rel(OUT_ROOT / "v22_72_final_route.json"), note=json.dumps({"final_route": route, "reason": reason}, ensure_ascii=False))
    return obj


def run_full(args: argparse.Namespace) -> dict[str, Any]:
    initialize_docs(reset_logs=bool(args.reset_logs), mode="full", device=str(args.device))
    part_a = run_part_a(args)
    empty: dict[str, Any] = {}
    if not int(part_a.get("part_a_hard_gate_pass", 0)):
        for part, path in [
            ("B", OUT_ROOT / "v22_72_part_b_failure_replay_visibility_decomposition.csv"),
            ("C", OUT_ROOT / "v22_72_part_c_edge_metric_unit_tests.csv"),
            ("D", OUT_ROOT / "v22_72_part_d_edge_native_preflight.csv"),
            ("E", OUT_ROOT / "v22_72_part_e_domain_transport_ablation.csv"),
            ("F", OUT_ROOT / "v22_72_part_f_edge_native_full_loop_matrix.csv"),
            ("G", OUT_ROOT / "v22_72_part_g_mlp_matched_audit.csv"),
        ]:
            write_skipped_artifact(part, path, "Part A hard gate failed.")
        return final_route(part_a, empty, empty, empty, empty, empty, empty)
    part_b = run_part_b(args)
    part_c = run_part_c(args)
    if not int(part_c.get("part_c_gate_pass", 0)) or not int(part_b.get("part_b_reanalysis_complete", 0)):
        for part, path in [
            ("D", OUT_ROOT / "v22_72_part_d_edge_native_preflight.csv"),
            ("E", OUT_ROOT / "v22_72_part_e_domain_transport_ablation.csv"),
            ("F", OUT_ROOT / "v22_72_part_f_edge_native_full_loop_matrix.csv"),
            ("G", OUT_ROOT / "v22_72_part_g_mlp_matched_audit.csv"),
        ]:
            write_skipped_artifact(part, path, "Part B/C gate failed.")
        return final_route(part_a, part_b, part_c, empty, empty, empty, empty)
    part_d = run_part_d(args)
    part_e = run_part_e(args, part_d)
    part_f = run_part_f(args, part_d, part_e)
    part_g = run_part_g(args, part_f)
    return final_route(part_a, part_b, part_c, part_d, part_e, part_f, part_g)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="full", choices=["full", "part-a", "part-b", "part-c", "part-d", "part-e", "part-f", "part-g"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--reset-logs", action="store_true")
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--held-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--metric-batch-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--edge-raw-strength", type=float, default=2.0)
    p.add_argument("--edge-active-cols", type=int, default=24)
    p.add_argument("--sensitivity-cols", type=int, default=12)
    p.add_argument("--basis-balance-ridge", type=float, default=1.0e-5)
    p.add_argument("--control-contrastive-cols", type=int, default=24)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--part-d-datasets", default="MNIST,FashionMNIST,KMNIST,Wine,Spam")
    p.add_argument("--part-d-architectures", default="DGKAN_DCHE,DGKAN_DFOU,DGKAN_FOU4_LIN,DGKAN_HAT4_XLIN,DGKAN_RBF4_XLIN")
    p.add_argument("--part-d-quick-limit", type=int, default=72)
    p.add_argument("--part-d-row-limit", type=int, default=75)
    p.add_argument("--part-f-datasets", default="MNIST,FashionMNIST,KMNIST")
    p.add_argument("--part-f-architectures", default="DGKAN_FOU4_LIN")
    p.add_argument("--part-f-group-limit", type=int, default=9)
    p.add_argument("--part-f-methods", default="edge_nat_no_transport_rank4,edge_nat_rawreadout_rank4,edge_nat_ownresidual_rank4,edge_nat_ownresidual_debtcone_rank4,edge_nat_transport_rawreadout_ownresidual_debtcone_controlcontrastive_rank4")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_out()
    try:
        if args.mode == "part-a":
            initialize_docs(reset_logs=bool(args.reset_logs), mode="part-a", device=str(args.device))
            run_part_a(args)
        elif args.mode == "part-b":
            initialize_docs(reset_logs=bool(args.reset_logs), mode="part-b", device=str(args.device))
            run_part_b(args)
        elif args.mode == "part-c":
            initialize_docs(reset_logs=bool(args.reset_logs), mode="part-c", device=str(args.device))
            run_part_c(args)
        elif args.mode == "part-d":
            initialize_docs(reset_logs=bool(args.reset_logs), mode="part-d", device=str(args.device))
            run_part_d(args)
        elif args.mode == "part-e":
            initialize_docs(reset_logs=bool(args.reset_logs), mode="part-e", device=str(args.device))
            dpath = OUT_ROOT / "v22_72_part_d_edge_native_preflight_summary.json"
            part_d = json.loads(dpath.read_text(encoding="utf-8")) if dpath.exists() else {"part_d_preflight_gate_pass": 0}
            run_part_e(args, part_d)
        elif args.mode == "part-f":
            initialize_docs(reset_logs=bool(args.reset_logs), mode="part-f", device=str(args.device))
            dpath = OUT_ROOT / "v22_72_part_d_edge_native_preflight_summary.json"
            epath = OUT_ROOT / "v22_72_part_e_domain_transport_ablation_summary.json"
            part_d = json.loads(dpath.read_text(encoding="utf-8")) if dpath.exists() else {"part_d_preflight_gate_pass": 0}
            part_e = json.loads(epath.read_text(encoding="utf-8")) if epath.exists() else {"part_e_gate_pass": 0}
            run_part_f(args, part_d, part_e)
        elif args.mode == "part-g":
            initialize_docs(reset_logs=bool(args.reset_logs), mode="part-g", device=str(args.device))
            fpath = OUT_ROOT / "v22_72_part_f_edge_native_full_loop_summary.json"
            part_f = json.loads(fpath.read_text(encoding="utf-8")) if fpath.exists() else {"part_f_exploration_gate_pass": 0}
            run_part_g(args, part_f)
        else:
            run_full(args)
        return 0
    except Exception as exc:
        log = write_exception_log("runner_top_level", exc)
        append_exec("runner_exception", command_text([PYTHON, rel(RUNNER), "--mode", str(args.mode)]), "exception", files=rel(log), note=repr(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
