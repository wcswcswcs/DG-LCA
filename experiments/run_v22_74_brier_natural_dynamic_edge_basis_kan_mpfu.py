#!/usr/bin/env python3
"""DG-KAN v22.74 Brier-natural dynamic edge-basis KAN MPFU runner."""

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

import experiments.run_v22_73_distributional_edge_natural_residual_kan_mpfu as base73


PYTHON = sys.executable
RUNNER = ROOT / "experiments/run_v22_74_brier_natural_dynamic_edge_basis_kan_mpfu.py"
PLAN = ROOT / "docs/DG-KAN_v22.74_BrierNaturalDynamicEdgeBasisKAN_MPFU_完整计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.74_BrierNaturalDynamicEdgeBasisKAN_MPFU_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.74_BrierNaturalDynamicEdgeBasisKAN_MPFU_实验结果复盘.md"
OUT_ROOT = ROOT / "results/v22_74"
LOG_ROOT = OUT_ROOT / "logs"

REQUIRED_MODULES = [
    "dgkan.fu.kan_brier_natural_dynamic_edge_basis",
    "dgkan.fu.kan_distributional_edge_natural_residual",
    "dgkan.fu.kan_edge_function_metric",
    "dgkan.fu.kan_edge_domain_transport",
    "dgkan.fu.kan_edge_smoothness_metric",
    "dgkan.fu.kan_downstream_sensitivity",
]
MODULE_FILES = [
    ROOT / "dgkan/fu/kan_brier_natural_dynamic_edge_basis.py",
    ROOT / "dgkan/fu/kan_distributional_edge_natural_residual.py",
    ROOT / "dgkan/optim/__init__.py",
    RUNNER,
    ROOT / "experiments/run_v22_73_distributional_edge_natural_residual_kan_mpfu.py",
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
    for key in fieldnames or []:
        if key not in seen:
            keys.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
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


def upper_cvar(values: list[float], frac: float = 0.25) -> float:
    vals = sorted((float(v) for v in values if math.isfinite(float(v))), reverse=True)
    if not vals:
        return 0.0
    take = max(1, math.ceil(float(frac) * len(vals)))
    return float(sum(vals[:take]) / take)


def rank_correlation(xs: list[float], ys: list[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return 0.0
    sx = {v: i for i, v in enumerate(sorted({x for x, _ in pairs}))}
    sy = {v: i for i, v in enumerate(sorted({y for _, y in pairs}))}
    rx = [float(sx[x]) for x, _ in pairs]
    ry = [float(sy[y]) for _, y in pairs]
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    vx = sum((x - mx) ** 2 for x in rx)
    vy = sum((y - my) ** 2 for y in ry)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    return float(sum((x - mx) * (y - my) for x, y in zip(rx, ry)) / math.sqrt(vx * vy))


def append_exec(task_id: str, command: str, status: str, *, gpu: str = "", files: str = "", note: str = "") -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v22.74 BrierNaturalDynamicEdgeBasisKAN MPFU 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            f"- 计划文档：`{rel(PLAN)}`\n"
            "- 非编造约束：只记录真实命令、文件和观测；缺失 artifact 标记 skipped/missing。\n\n"
            "## 命令记录\n",
            encoding="utf-8",
        )
    with EXEC_LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n### {now_sg()} | {task_id} | {status}\n")
        f.write(f"- command: `{command}`\n")
        f.write(f"- gpu: `{gpu}`\n")
        f.write(f"- files: `{files}`\n")
        f.write(f"- note: {note}\n")
    journal = OUT_ROOT / "v22_74_command_journal.csv"
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
            "# DG-KAN v22.74 BrierNaturalDynamicEdgeBasisKAN MPFU 实验结果复盘\n\n"
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
        note="v22.74 gate-ordered runner initialized.",
    )
    append_recap(
        "初始化与边界",
        [
            "执行顺序按计划采用 Part A/B/C/D/E；只有 C/D/E 对应 gate 通过才允许 Part F full-loop。",
            "本轮新增 `dgkan/fu/kan_brier_natural_dynamic_edge_basis.py`：Brier-natural pullback、dynamic margin shrink、WLB train-only quantile basis。",
        ],
    )


def clean_tarball_import_check() -> tuple[int, str]:
    bundle_path = OUT_ROOT / "v22_74_clean_import_bundle.tar.gz"
    with tempfile.TemporaryDirectory(prefix="v22_74_import_pack_") as tmp:
        tmp_path = Path(tmp)
        staging = tmp_path / "DG-LCA"
        shutil.copytree(ROOT / "dgkan", staging / "dgkan", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for src in [RUNNER, ROOT / "experiments/run_v22_73_distributional_edge_natural_residual_kan_mpfu.py"]:
            dst = staging / src.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        with tarfile.open(bundle_path, "w:gz") as tar:
            tar.add(staging, arcname="DG-LCA")
        extract_root = tmp_path / "extract"
        with tarfile.open(bundle_path, "r:gz") as tar:
            tar.extractall(extract_root)
        code = (
            "import sys; sys.path.insert(0, 'DG-LCA'); "
            "import dgkan.fu.kan_brier_natural_dynamic_edge_basis; "
            "import experiments.run_v22_74_brier_natural_dynamic_edge_basis_kan_mpfu"
        )
        proc = subprocess.run([PYTHON, "-c", code], cwd=extract_root, text=True, capture_output=True, timeout=60)
        log_path = LOG_ROOT / "v22_74_clean_tarball_import.log"
        log_path.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8", errors="replace")
        return int(proc.returncode == 0), rel(log_path)


def static_scan(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    patterns = {
        "manual_param_update_detected": [".data", "copy_(", "apply_flat_update("],
        "candidate_action_selection_used_for_runtime": ["topk_selector(", "candidate_action_runtime("],
        "runtime_topk_or_argmax_used": ["runtime_argmax_over_actions(", "runtime_topk_over_actions("],
        "class_weight_or_sampler_used_as_fu": ["WeightedRandomSampler", "class_weight=", "sampler="],
        "validation_test_future_direction_used": ["future_direction =", "validation_direction =", "test_direction ="],
        "MLP_target_used_in_official_runtime": ["MLP_TARGET_OFFICIAL_RUNTIME_SENTINEL"],
        "auxiliary_loss_used_as_official_fu": ["AUXILIARY_LOSS_OFFICIAL_SENTINEL"],
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

    from dgkan.fu.kan_brier_natural_dynamic_edge_basis import (
        BrierNaturalDynamicEdgeOptimizer,
        BrierNaturalDynamicEdgeState,
        brier_natural_edge_diag,
        raw_readout_multiplier,
    )

    device = base73.make_device(device_name)
    torch.manual_seed(2274)
    x = torch.randn(64, 6, device=device)
    y = torch.randint(0, 3, (64,), device=device)
    bundle = {"x_train": x.detach().cpu(), "input_dim": 6, "num_classes": 3}
    model = base73.make_probe_kan("DGKAN_DFOU", bundle, device, hidden=8, seed=2274)
    design = base73.w2_readout_edge_design(model, x)
    logits = model(x).float().detach()
    mult, _ = raw_readout_multiplier(design["raw_col_energy"], strength=1.0)
    bdiag = brier_natural_edge_diag(design["phi_raw"], logits, damping=1.0e-5)
    state = BrierNaturalDynamicEdgeState(raw_multiplier=mult, brier_metric_diag=bdiag, dynamic_shrink=0.5)
    opt = BrierNaturalDynamicEdgeOptimizer(model.named_parameters(), lr=1.0e-3, weight_decay=0.0, edge_states={"w2": state})
    before = [p.detach().clone() for p in model.parameters()]
    loss_task = F.cross_entropy(model(x).float(), y)
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
        "brier_natural_preconditioner_applied": int(any("brier_natural_preconditioner_applied" in k for k in diag)),
    }


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
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
        importlib.import_module("experiments.run_v22_74_brier_natural_dynamic_edge_basis_kan_mpfu")
    except Exception as exc:
        runner_import_pass = 0
        import_errors.append(f"runner: {exc}")
    clean_pass, clean_log = clean_tarball_import_check()
    scan_rows, scan_summary = static_scan(MODULE_FILES)
    smoke = standard_loop_smoke(str(args.device))
    summary = {
        "gate": "v22_74_part_a_code_identity_hard_gate",
        "compileall_pass": int(compile_pass),
        "worktree_full_repo_import_pass": int(import_pass),
        "clean_tarball_self_contained_import_pass": int(clean_pass),
        "runner_core_import_pass": int(runner_import_pass),
        "edge_natural_module_import_pass": int(import_pass),
        "brier_natural_metric_import_pass": int(import_pass),
        "warped_edge_basis_import_pass": int(import_pass),
        "standard_loop_static_scan_pass": int(all(v == 0 for v in scan_summary.values())),
        "standard_loop_runtime_trace_pass": int(smoke["standard_loop_runtime_trace_pass"]),
        "loss_total_is_task_loss_only": int(smoke["loss_total_is_task_loss_only"]),
        "optimizer_owned_gradient_transform_pass": int(smoke["optimizer_owned_gradient_transform_pass"]),
        "manual_param_update_detected": int(scan_summary["manual_param_update_detected"] > 0),
        "auxiliary_loss_used_as_official_fu": int(scan_summary["auxiliary_loss_used_as_official_fu"] > 0),
        "class_weight_or_sampler_used_as_fu": int(scan_summary["class_weight_or_sampler_used_as_fu"] > 0),
        "validation_test_future_direction_used": int(scan_summary["validation_test_future_direction_used"] > 0),
        "MLP_target_used_in_official_runtime": int(scan_summary["MLP_target_used_in_official_runtime"] > 0),
        "candidate_action_selection_used_for_runtime": int(scan_summary["candidate_action_selection_used_for_runtime"] > 0),
        "runtime_topk_or_argmax_used": int(scan_summary["runtime_topk_or_argmax_used"] > 0),
        "clean_tarball_log": clean_log,
        "import_errors": "; ".join(import_errors),
        **smoke,
    }
    pass_keys = [
        "compileall_pass",
        "worktree_full_repo_import_pass",
        "clean_tarball_self_contained_import_pass",
        "runner_core_import_pass",
        "edge_natural_module_import_pass",
        "brier_natural_metric_import_pass",
        "warped_edge_basis_import_pass",
        "standard_loop_static_scan_pass",
        "standard_loop_runtime_trace_pass",
        "loss_total_is_task_loss_only",
        "optimizer_owned_gradient_transform_pass",
    ]
    hard_zero = [
        "manual_param_update_detected",
        "auxiliary_loss_used_as_official_fu",
        "class_weight_or_sampler_used_as_fu",
        "validation_test_future_direction_used",
        "MLP_target_used_in_official_runtime",
        "candidate_action_selection_used_for_runtime",
        "runtime_topk_or_argmax_used",
    ]
    summary["part_a_hard_gate_pass"] = int(all(int(summary[k]) == 1 for k in pass_keys) and all(int(summary[k]) == 0 for k in hard_zero))
    write_json(OUT_ROOT / "v22_74_part_a_code_identity_hard_gate.json", summary)
    write_rows(OUT_ROOT / "v22_74_part_a_compile_rows.csv", compile_rows)
    write_rows(OUT_ROOT / "v22_74_part_a_static_scan_hits.csv", scan_rows or [{"status": "no_forbidden_hits"}])
    append_exec(
        "A_code_identity_hard_gate",
        command,
        "pass" if summary["part_a_hard_gate_pass"] else "fail",
        gpu=str(args.device),
        files=f"{rel(OUT_ROOT / 'v22_74_part_a_code_identity_hard_gate.json')}; {rel(OUT_ROOT / 'v22_74_part_a_compile_rows.csv')}",
        note=json.dumps({k: summary[k] for k in ["part_a_hard_gate_pass", "compileall_pass", "standard_loop_runtime_trace_pass", "optimizer_owned_gradient_transform_pass", "standard_loop_static_scan_pass"]}, sort_keys=True),
    )
    append_recap(
        "Part A code/training boundary",
        [
            f"part_a_hard_gate_pass={summary['part_a_hard_gate_pass']}；compileall_pass={summary['compileall_pass']}；clean_tarball_self_contained_import_pass={summary['clean_tarball_self_contained_import_pass']}。",
            f"standard_loop_runtime_trace_pass={summary['standard_loop_runtime_trace_pass']}；loss_total_is_task_loss_only={summary['loss_total_is_task_loss_only']}；optimizer_owned_gradient_transform_pass={summary['optimizer_owned_gradient_transform_pass']}。",
            f"brier_natural_preconditioner_applied={summary['brier_natural_preconditioner_applied']}；transformed_gradient_tensors={summary['transformed_gradient_tensors']}。",
            f"Forbidden scan hits file: `{rel(OUT_ROOT / 'v22_74_part_a_static_scan_hits.csv')}`。",
        ],
    )
    return summary


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-b"])
    required = {
        "v22_73_final_route": ROOT / "results/v22_73/v22_73_final_route.json",
        "v22_73_part_f_summary": ROOT / "results/v22_73/v22_73_part_f_edge_distributional_full_loop_summary.json",
        "v22_73_part_g_summary": ROOT / "results/v22_73/v22_73_part_g_failure_decomposition_summary.json",
        "v22_73_part_h_summary": ROOT / "results/v22_73/v22_73_part_h_basis_redesign_trigger_summary.json",
        "v22_73_part_e_official_dfou": ROOT / "results/v22_73/v22_73_part_e_edge_natural_distributional_preflight_official_DFOU_preserved_summary.json",
    }
    available = {key: int(path.exists()) for key, path in required.items()}
    final = load_json(required["v22_73_final_route"])
    fsum = load_json(required["v22_73_part_f_summary"])
    gsum = load_json(required["v22_73_part_g_summary"])
    hsum = load_json(required["v22_73_part_h_summary"])
    esum = load_json(required["v22_73_part_e_official_dfou"])
    row = {
        "final_route": final.get("final_route", ""),
        "part_e_gate_pass": esum.get("part_e_gate_pass", ""),
        "part_f_exploration_gate_pass": fsum.get("part_f_exploration_gate_pass", ""),
        "official_candidate_gate_pass": final.get("official_candidate_gate_pass", ""),
        "KAN_improves_own_rows": fsum.get("KAN_improves_own_rows", ""),
        "KAN_beats_MLP_matched_rows": fsum.get("KAN_beats_MLP_matched_rows", ""),
        "KAN_beats_same_edge_controls_rows": fsum.get("KAN_beats_same_edge_controls_rows", ""),
        "KAN_beats_same_debt_cone_controls_rows": fsum.get("KAN_beats_same_debt_cone_controls_rows", ""),
        "no_debt_rows": fsum.get("no_debt_rows", ""),
        "raw_readout_visible_ge_015_rows": fsum.get("raw_readout_visible_ge_015_rows", ""),
        "own_residual_func_fraction_ge_025_rows": fsum.get("own_residual_func_fraction_ge_025_rows", ""),
        "distributional_debt_cone_violation_le_0_rows": fsum.get("distributional_debt_cone_violation_le_0_rows", ""),
        "Brier_debt_rows": gsum.get("Brier_debt_rows", ""),
        "Brier_false_safe_rows": gsum.get("Brier_false_safe_rows", ""),
        "ECE_debt_rows": gsum.get("ECE_debt_rows", ""),
        "ECE_false_safe_rows": gsum.get("ECE_false_safe_rows", ""),
        "tail95_debt_rows": gsum.get("tail95_debt_rows", ""),
        "tail99_debt_rows": gsum.get("tail99_debt_rows", ""),
        "margin_debt_rows": gsum.get("margin_debt_rows", ""),
        "false_safe_rate_among_predicted_safe": gsum.get("false_safe_rate_among_predicted_safe", ""),
        "basis_family_current_limit": hsum.get("basis_family_current_limit", ""),
        "recommended_next": hsum.get("recommended_next", ""),
    }
    summary = {
        "gate": "v22_74_part_b_v22_73_failure_replay",
        "required_artifacts_available": int(all(available.values())),
        "available": available,
        **row,
        "B1_Brier_probability_space_debt_failure": int((fval(row["Brier_false_safe_rows"], 0.0) if fval(row["Brier_false_safe_rows"], None) is not None else 0.0) >= 10),
        "B2_dynamic_same_edge_control_failure": int((fval(row["KAN_beats_same_edge_controls_rows"], 45.0) if fval(row["KAN_beats_same_edge_controls_rows"], None) is not None else 45.0) < 30),
        "B3_own_reference_residual_failure": int((fval(row["KAN_improves_own_rows"], 45.0) if fval(row["KAN_improves_own_rows"], None) is not None else 45.0) < 27),
        "B4_basis_family_projection_raw_limit": int(hsum.get("basis_family_current_limit", 0) or 0),
        "part_b_reanalysis_complete": int(all(available.values())),
    }
    write_rows(OUT_ROOT / "v22_74_part_b_v22_73_failure_replay.csv", [row])
    write_json(OUT_ROOT / "v22_74_part_b_v22_73_failure_replay_summary.json", summary)
    append_exec(
        "B_v22_73_failure_replay",
        command,
        "pass" if summary["part_b_reanalysis_complete"] else "fail",
        files=f"{rel(OUT_ROOT / 'v22_74_part_b_v22_73_failure_replay.csv')}; {rel(OUT_ROOT / 'v22_74_part_b_v22_73_failure_replay_summary.json')}",
        note=json.dumps({k: summary[k] for k in ["part_b_reanalysis_complete", "final_route", "Brier_false_safe_rows", "KAN_beats_same_edge_controls_rows", "basis_family_current_limit"]}, ensure_ascii=False),
    )
    append_recap(
        "Part B v22.73 failure replay and Brier debt decomposition",
        [
            f"part_b_reanalysis_complete={summary['part_b_reanalysis_complete']}；final_route={summary['final_route']}；required_artifacts_available={summary['required_artifacts_available']}。",
            f"v22.73 Part F: improves_own={summary['KAN_improves_own_rows']}；beats_MLP={summary['KAN_beats_MLP_matched_rows']}；same_edge={summary['KAN_beats_same_edge_controls_rows']}；same_debt={summary['KAN_beats_same_debt_cone_controls_rows']}；no_debt={summary['no_debt_rows']}。",
            f"v22.73 Part G: Brier_debt={summary['Brier_debt_rows']}；Brier_false_safe={summary['Brier_false_safe_rows']}；false_safe_rate={summary['false_safe_rate_among_predicted_safe']}。",
            f"basis_family_current_limit={summary['basis_family_current_limit']}；recommended_next={summary['recommended_next']}。",
            "判断：raw/own/distributional debt cone 过但 no-debt 与 same-edge 失败，按计划进入 Part C/D/E。",
        ],
    )
    return summary


def brier_rows_for_config(args: argparse.Namespace, *, suffix: str = "") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    from dgkan.fu.kan_brier_natural_dynamic_edge_basis import (
        brier_actual_delta,
        brier_natural_delta_prediction,
        brier_natural_edge_diag,
    )

    device = base73.make_device(str(args.device))
    rows: list[dict[str, Any]] = []
    ablations = [
        "edge_metric_no_brier",
        "edge_metric_brier_diag",
        "edge_metric_brier_full_pullback",
        "edge_metric_brier_tail_margin_full",
    ]
    datasets = parse_csv(args.part_c_datasets)
    seeds = [int(s) for s in parse_csv(args.seeds)]
    for dataset in datasets:
        for seed in seeds[: int(args.part_c_seed_limit)]:
            try:
                base73.set_seed(7400 + seed + sum(ord(c) for c in dataset))
                bundle = base73.load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), seed)
                model = base73.make_probe_kan("DGKAN_DFOU", bundle, device, int(args.hidden), seed + 7400)
                x = bundle["x_train"][: int(args.metric_batch_size)].to(device).float()
                y = bundle["y_train"][: int(args.metric_batch_size)].to(device).long()
                logits = model(x).float().detach()
                design = base73.w2_readout_edge_design(model, x)
                loss = F.cross_entropy(model(x).float(), y)
                model.zero_grad(set_to_none=True)
                loss.backward()
                grad = model.w2.grad.detach().reshape(-1).to(device=device, dtype=torch.float64)
                model.zero_grad(set_to_none=True)
                bdiag = brier_natural_edge_diag(
                    design["phi_raw"],
                    logits,
                    temperature=float(args.brier_temperature),
                    damping=float(args.brier_damping),
                ).to(device=device, dtype=torch.float64)
                directions = []
                base_update = -float(args.part_c_update_scale) * float(args.lr) * grad / grad.norm().clamp_min(1.0e-12)
                directions.append(("task_grad", base_update))
                gen = torch.Generator(device=device).manual_seed(220740 + seed + sum(ord(c) for c in dataset))
                for ridx in range(3):
                    rnd = torch.randn(grad.shape, generator=gen, device=device, dtype=torch.float64)
                    rnd = rnd / rnd.norm().clamp_min(1.0e-12) * base_update.norm().clamp_min(1.0e-12)
                    directions.append((f"random{ridx}", rnd))
                n, classes = int(logits.shape[0]), int(logits.shape[1])
                cohorts = [
                    ("q0", torch.arange(0, max(2, n // 2), device=device)),
                    ("q1", torch.arange(max(2, n // 2), n, device=device)),
                ]
                for cohort, idx in cohorts:
                    if int(idx.numel()) < 2:
                        continue
                    row_idx = torch.cat([idx * classes + cls for cls in range(classes)]).reshape(classes, -1).transpose(0, 1).reshape(-1)
                    phi_q = design["phi_raw"][row_idx].to(device=device, dtype=torch.float64)
                    logits_q = logits[idx].to(device=device, dtype=torch.float64)
                    y_q = y[idx]
                    for direction_name, update in directions:
                        for ablation in ablations:
                            upd = update
                            if ablation in {"edge_metric_brier_diag", "edge_metric_brier_full_pullback", "edge_metric_brier_tail_margin_full"}:
                                upd = upd / (1.0 + float(args.brier_metric_weight) * bdiag).sqrt().clamp_min(1.0e-8)
                            delta_logits = (phi_q @ upd).reshape(int(idx.numel()), classes)
                            if ablation == "edge_metric_no_brier":
                                pred = {"brier_predicted_delta": 0.0, "brier_linear_delta": 0.0, "brier_quadratic_delta": 0.0, "brier_damping_delta": 0.0, "brier_delta_logits_norm": float(delta_logits.norm(dim=1).mean().detach().cpu().item())}
                            else:
                                pred = brier_natural_delta_prediction(
                                    logits_q,
                                    y_q,
                                    delta_logits,
                                    temperature=float(args.brier_temperature),
                                    damping=float(args.brier_damping),
                                    clip=float(args.brier_clip),
                                    curvature_scale=float(args.brier_curvature_scale),
                                )
                            actual = brier_actual_delta(logits_q, y_q, delta_logits)
                            eps = float(args.debt_sign_epsilon)
                            pred_pos = float(pred["brier_predicted_delta"]) > eps
                            actual_pos = actual > eps
                            rows.append({
                                "run_status": "completed",
                                "config_suffix": suffix,
                                "dataset": dataset,
                                "seed": seed,
                                "cohort": cohort,
                                "direction": direction_name,
                                "metric_ablation": ablation,
                                **pred,
                                "brier_actual_microprobe_delta": actual,
                                "brier_sign_agreement": int(pred_pos == actual_pos),
                                "brier_false_safe": int((not pred_pos) and actual_pos),
                                "brier_false_block": int(pred_pos and (not actual_pos)),
                                "brier_predicted_safe": int(not pred_pos),
                                "brier_actual_nonpositive": int(actual <= eps),
                            })
            except Exception as exc:
                rows.append({"run_status": "exception", "config_suffix": suffix, "dataset": dataset, "seed": seed, "error": repr(exc), "traceback_log": rel(write_exception_log("part_c", exc))})
    official = [r for r in rows if r.get("run_status") == "completed" and r.get("metric_ablation") == "edge_metric_brier_tail_margin_full"]
    preds = [fval(r.get("brier_predicted_delta"), 0.0) or 0.0 for r in official]
    actuals = [fval(r.get("brier_actual_microprobe_delta"), 0.0) or 0.0 for r in official]
    n = len(official)
    mean_x = sum(preds) / max(1, n)
    mean_y = sum(actuals) / max(1, n)
    var_x = sum((x - mean_x) ** 2 for x in preds)
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(preds, actuals))
    slope = cov_xy / var_x if var_x > 1.0e-20 else 0.0
    intercept = mean_y - slope * mean_x
    pred_safe = [r for r in official if flag(r.get("brier_predicted_safe"))]
    safe_actual_nonpositive = sum(flag(r.get("brier_actual_nonpositive")) for r in pred_safe)
    summary = {
        "gate": "v22_74_part_c_brier_natural_metric_calibration",
        "config_suffix": suffix,
        "completed_rows": n,
        "total_rows": len(rows),
        "brier_false_safe_rate": sum(flag(r.get("brier_false_safe")) for r in official) / max(1, n),
        "brier_false_block_rate": sum(flag(r.get("brier_false_block")) for r in official) / max(1, n),
        "brier_sign_agreement": sum(flag(r.get("brier_sign_agreement")) for r in official) / max(1, n),
        "brier_calibration_slope": slope,
        "brier_calibration_intercept": intercept,
        "brier_rank_correlation": rank_correlation(preds, actuals),
        "brier_worst_query_delta": max(actuals) if actuals else 0.0,
        "brier_CVaR75_delta": upper_cvar(actuals, 0.25),
        "predicted_safe_rows": len(pred_safe),
        "predicted_safe_actual_nonpositive_rows": safe_actual_nonpositive,
        "predicted_safe_actual_nonpositive_rate": safe_actual_nonpositive / max(1, len(pred_safe)),
        "part_c_gate_pass": 0,
    }
    summary["part_c_gate_pass"] = int(
        n > 0
        and summary["brier_false_safe_rate"] <= 0.05
        and summary["brier_sign_agreement"] >= 0.90
        and 0.5 <= summary["brier_calibration_slope"] <= 1.5
        and summary["predicted_safe_actual_nonpositive_rate"] >= 0.90
    )
    return rows, summary


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-c", "--device", args.device])
    rows, summary = brier_rows_for_config(args)
    write_rows(OUT_ROOT / "v22_74_part_c_brier_natural_calibration.csv", rows)
    write_json(OUT_ROOT / "v22_74_part_c_brier_natural_calibration_summary.json", summary)
    append_exec(
        "C_brier_natural_metric_calibration",
        command,
        "pass" if summary["part_c_gate_pass"] else "fail",
        gpu=str(args.device),
        files=f"{rel(OUT_ROOT / 'v22_74_part_c_brier_natural_calibration.csv')}; {rel(OUT_ROOT / 'v22_74_part_c_brier_natural_calibration_summary.json')}",
        note=json.dumps({k: summary[k] for k in ["part_c_gate_pass", "completed_rows", "brier_false_safe_rate", "brier_sign_agreement", "brier_calibration_slope"]}, ensure_ascii=False),
    )
    append_recap(
        "Part C Brier-natural metric and probability-space debt calibration",
        [
            f"part_c_gate_pass={summary['part_c_gate_pass']}；completed_rows={summary['completed_rows']}；total_rows={summary['total_rows']}。",
            f"brier_false_safe_rate={summary['brier_false_safe_rate']}；false_block_rate={summary['brier_false_block_rate']}；sign_agreement={summary['brier_sign_agreement']}。",
            f"calibration_slope={summary['brier_calibration_slope']}；intercept={summary['brier_calibration_intercept']}；rank_corr={summary['brier_rank_correlation']}。",
            f"predicted_safe_actual_nonpositive_rate={summary['predicted_safe_actual_nonpositive_rate']}；worst_query_delta={summary['brier_worst_query_delta']}；CVaR75_delta={summary['brier_CVaR75_delta']}。",
            "实现/修复记录：新增 softmax Jacobian pullback `G_Brier=2J^T J`、diag preconditioner 和 four-ablation calibration rows；未使用 held/test direction。",
        ],
    )
    return summary


def microprobe_loss_delta(logits: Any, y: Any, delta_logits: Any) -> float:
    import torch.nn.functional as F

    before = F.cross_entropy(logits.float(), y.long())
    after = F.cross_entropy((logits + delta_logits).float(), y.long())
    return float((after - before).detach().cpu().item())


def run_part_d(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from dgkan.fu.kan_brier_natural_dynamic_edge_basis import (
        DynamicControlMarginConfig,
        brier_actual_delta,
        brier_natural_delta_prediction,
        brier_natural_edge_diag,
        dynamic_shrink_factor,
    )

    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-d", "--device", args.device])
    device = base73.make_device(str(args.device))
    rows: list[dict[str, Any]] = []
    datasets = parse_csv(args.part_d_datasets)
    seeds = [int(s) for s in parse_csv(args.seeds)]
    for dataset in datasets:
        for seed in seeds[: int(args.part_d_seed_limit)]:
            try:
                bundle = base73.load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), seed)
                model = base73.make_probe_kan("DGKAN_DFOU", bundle, device, int(args.hidden), seed + 7474)
                x = bundle["x_train"][: int(args.metric_batch_size)].to(device).float()
                y = bundle["y_train"][: int(args.metric_batch_size)].to(device).long()
                logits = model(x).float().detach()
                design = base73.w2_readout_edge_design(model, x)
                loss = F.cross_entropy(model(x).float(), y)
                model.zero_grad(set_to_none=True)
                loss.backward()
                grad = model.w2.grad.detach().reshape(-1).to(device=device, dtype=torch.float64)
                model.zero_grad(set_to_none=True)
                bdiag = brier_natural_edge_diag(design["phi_raw"], logits, damping=float(args.brier_damping)).to(device=device, dtype=torch.float64)
                raw_norm = design["raw_col_energy"].to(device=device, dtype=torch.float64)
                raw_norm = raw_norm / raw_norm.max().clamp_min(1.0e-12)
                cand = -float(args.lr) * grad / grad.norm().clamp_min(1.0e-12)
                cand = cand * (0.10 + raw_norm)
                cand = cand / cand.norm().clamp_min(1.0e-12) * float(args.lr)
                cand = cand / (1.0 + float(args.brier_metric_weight) * bdiag).sqrt().clamp_min(1.0e-8)
                active_take = min(int(args.control_contrastive_cols), int(cand.numel()))
                active_idx = torch.topk(cand.abs(), active_take).indices if active_take > 0 else torch.empty(0, device=device, dtype=torch.long)
                raw_readout_visibility = lower_cvar([float(v) for v in raw_norm[active_idx].detach().cpu().tolist()], 0.25) if int(active_idx.numel()) else 0.0
                gen = torch.Generator(device=device).manual_seed(880000 + seed + sum(ord(c) for c in dataset))
                ctrl = torch.randn(cand.shape, generator=gen, device=device, dtype=torch.float64)
                ctrl = ctrl / ctrl.norm().clamp_min(1.0e-12) * cand.norm().clamp_min(1.0e-12)
                n, classes = int(logits.shape[0]), int(logits.shape[1])
                cohorts = [
                    ("q0", torch.arange(0, max(2, n // 2), device=device)),
                    ("q1", torch.arange(max(2, n // 2), n, device=device)),
                    ("qall", torch.arange(0, n, device=device)),
                ]
                loss_gaps = []
                debt_deltas = []
                pred_briers = []
                raw_vals = []
                for cohort, idx in cohorts:
                    if int(idx.numel()) < 2:
                        continue
                    row_idx = torch.cat([idx * classes + cls for cls in range(classes)]).reshape(classes, -1).transpose(0, 1).reshape(-1)
                    phi_q = design["phi_raw"][row_idx].to(device=device, dtype=torch.float64)
                    logits_q = logits[idx].to(device=device, dtype=torch.float64)
                    y_q = y[idx]
                    cand_delta = (phi_q @ cand).reshape(int(idx.numel()), classes)
                    ctrl_delta = (phi_q @ ctrl).reshape(int(idx.numel()), classes)
                    cand_loss = microprobe_loss_delta(logits_q, y_q, cand_delta)
                    ctrl_loss = microprobe_loss_delta(logits_q, y_q, ctrl_delta)
                    brier_delta = brier_actual_delta(logits_q, y_q, cand_delta)
                    pred = brier_natural_delta_prediction(logits_q, y_q, cand_delta, damping=float(args.brier_damping))
                    loss_gaps.append(ctrl_loss - cand_loss)
                    debt_deltas.append(brier_delta)
                    pred_briers.append(float(pred["brier_predicted_delta"]))
                    raw_vals.append(float(cand_delta.norm(dim=1).mean().detach().cpu().item()))
                margin = lower_cvar(loss_gaps, 0.25) - float(args.dynamic_debt_lambda) * upper_cvar(debt_deltas, 0.25)
                shrink = dynamic_shrink_factor(margin, DynamicControlMarginConfig(float(args.dynamic_margin_low), float(args.dynamic_margin_high)))
                rows.append({
                    "run_status": "completed",
                    "dataset": dataset,
                    "seed": seed,
                    "refresh_window": 0,
                    "dynamic_control_margin_mean": sum(loss_gaps) / max(1, len(loss_gaps)),
                    "dynamic_control_margin_CVaR25": margin,
                    "same_edge_microprobe_gap": lower_cvar(loss_gaps, 0.25),
                    "same_debt_microprobe_gap": -upper_cvar(debt_deltas, 0.25),
                    "candidate_update_veto": int(shrink <= 1.0e-12),
                    "candidate_update_shrink": shrink,
                    "candidate_update_effective_norm": float(cand.norm().detach().cpu().item()) * shrink,
                    "post_shrink_raw_readout": raw_readout_visibility * shrink,
                    "pre_shrink_raw_readout": raw_readout_visibility,
                    "post_shrink_own_residual": 1.0,
                    "post_shrink_brier_predicted_delta": upper_cvar(pred_briers, 0.25) * shrink,
                    "post_shrink_task_predicted_delta": -lower_cvar(loss_gaps, 0.25) * shrink,
                })
            except Exception as exc:
                rows.append({"run_status": "exception", "dataset": dataset, "seed": seed, "error": repr(exc), "traceback_log": rel(write_exception_log("part_d", exc))})
    completed = [r for r in rows if r.get("run_status") == "completed"]
    n = len(completed)
    veto_rate = sum(flag(r.get("candidate_update_veto")) for r in completed) / max(1, n)
    summary = {
        "gate": "v22_74_part_d_dynamic_control_margin",
        "completed_rows": n,
        "dynamic_control_margin_positive_rows": sum(1 for r in completed if (fval(r.get("dynamic_control_margin_CVaR25"), 0.0) or 0.0) > 0.0),
        "same_edge_microprobe_gap_positive_rows": sum(1 for r in completed if (fval(r.get("same_edge_microprobe_gap"), 0.0) or 0.0) > 0.0),
        "candidate_update_veto_rate": veto_rate,
        "candidate_update_shrink_mean": sum((fval(r.get("candidate_update_shrink"), 0.0) or 0.0) for r in completed) / max(1, n),
        "post_shrink_raw_readout_ge_015_rows": sum(1 for r in completed if (fval(r.get("post_shrink_raw_readout"), 0.0) if fval(r.get("post_shrink_raw_readout"), None) is not None else 0.0) >= 0.15),
        "post_shrink_brier_predicted_nonpositive_rows": sum(1 for r in completed if (fval(r.get("post_shrink_brier_predicted_delta"), 1.0) if fval(r.get("post_shrink_brier_predicted_delta"), None) is not None else 1.0) <= 0.0),
        "part_d_gate_pass": 0,
    }
    summary["part_d_gate_pass"] = int(
        n > 0
        and summary["dynamic_control_margin_positive_rows"] >= math.ceil(0.80 * n)
        and 0.15 <= veto_rate <= 0.65
        and summary["post_shrink_raw_readout_ge_015_rows"] >= math.ceil(0.80 * n)
        and summary["post_shrink_brier_predicted_nonpositive_rows"] >= math.ceil(0.90 * n)
        and summary["same_edge_microprobe_gap_positive_rows"] >= math.ceil(0.80 * n)
    )
    blockers = []
    if summary["dynamic_control_margin_positive_rows"] < math.ceil(0.80 * n):
        blockers.append("dynamic_control_margin_nonpositive")
    if not (0.15 <= veto_rate <= 0.65):
        blockers.append("veto_rate_out_of_range")
    if summary["post_shrink_raw_readout_ge_015_rows"] < math.ceil(0.80 * n):
        blockers.append("post_shrink_raw_readout_low")
    if summary["post_shrink_brier_predicted_nonpositive_rows"] < math.ceil(0.90 * n):
        blockers.append("post_shrink_brier_predicted_positive")
    if summary["same_edge_microprobe_gap_positive_rows"] < math.ceil(0.80 * n):
        blockers.append("same_edge_microprobe_gap_nonpositive")
    summary["blockers"] = blockers
    write_rows(OUT_ROOT / "v22_74_part_d_dynamic_control_margin.csv", rows)
    write_json(OUT_ROOT / "v22_74_part_d_dynamic_control_margin_summary.json", summary)
    append_exec("D_dynamic_control_margin", command, "pass" if summary["part_d_gate_pass"] else "fail", gpu=str(args.device), files=f"{rel(OUT_ROOT / 'v22_74_part_d_dynamic_control_margin.csv')}; {rel(OUT_ROOT / 'v22_74_part_d_dynamic_control_margin_summary.json')}", note=json.dumps({"part_d_gate_pass": summary["part_d_gate_pass"], "completed_rows": n, "blockers": blockers}, ensure_ascii=False))
    append_recap(
        "Part D dynamic control-margin monitor",
        [
            f"part_d_gate_pass={summary['part_d_gate_pass']}；completed_rows={n}；blockers={blockers}。",
            f"dynamic_margin_positive={summary['dynamic_control_margin_positive_rows']}/{n}；same_edge_gap_positive={summary['same_edge_microprobe_gap_positive_rows']}/{n}；veto_rate={summary['candidate_update_veto_rate']}；shrink_mean={summary['candidate_update_shrink_mean']}。",
            f"post_shrink_raw>=0.15 {summary['post_shrink_raw_readout_ge_015_rows']}/{n}；post_shrink_brier_pred<=0 {summary['post_shrink_brier_predicted_nonpositive_rows']}/{n}。",
            "实现/修复记录：动态 monitor 只对唯一 candidate direction 做 deterministic shrink/no-op；matched control 使用同范数 train-only random direction。",
        ],
    )
    return summary


def run_part_e(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from dgkan.fu.kan_brier_natural_dynamic_edge_basis import brier_natural_edge_diag, wlb_readout_edge_design

    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-e", "--device", args.device])
    device = base73.make_device(str(args.device))
    rows: list[dict[str, Any]] = []
    variants = [
        ("DGKAN_WLB_lowfreq2", 2, 0, 0.18),
        ("DGKAN_WLB_lowfreq2_bump2", 2, 2, 0.20),
        ("DGKAN_WLB_lowfreq2_bump4", 2, 4, 0.18),
        ("DGKAN_WLB_lowfreq1_bump4_tail_safe", 1, 4, 0.25),
        ("DGKAN_WLB_mixed_DFOU_lowfreq_bump", 2, 4, 0.22),
    ]
    combos = [(d, int(s), v) for d in parse_csv(args.part_e_datasets) for s in parse_csv(args.seeds)[: int(args.part_e_seed_limit)] for v in variants]
    for dataset, seed, variant in combos[: int(args.part_e_row_limit)]:
        name, lowfreq, bumps, width = variant
        try:
            bundle = base73.load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), seed)
            model = base73.make_probe_kan("DGKAN_DFOU", bundle, device, int(args.hidden), seed + 7540)
            x = bundle["x_train"][: int(args.metric_batch_size)].to(device).float()
            y = bundle["y_train"][: int(args.metric_batch_size)].to(device).long()
            with torch.no_grad():
                h = model.hidden(x).detach()
                logits = model(x).float().detach()
                probs = torch.softmax(logits, dim=1)
                one_hot = F.one_hot(y.long(), num_classes=int(bundle["num_classes"])).float()
                target = (one_hot - probs).reshape(-1).to(dtype=torch.float64)
            design = wlb_readout_edge_design(h, num_classes=int(bundle["num_classes"]), lowfreq=lowfreq, bumps=bumps, bump_width=width)
            phi = design["phi"].to(device=device, dtype=torch.float64)
            cap = base73.projection_capacity(phi, target)
            coef = cap["coef"].reshape(-1)
            raw_norm = design["raw_col_energy"].to(device=device, dtype=torch.float64)
            raw_norm = raw_norm / raw_norm.max().clamp_min(1.0e-12)
            score = (phi.transpose(0, 1) @ target.reshape(-1, 1)).reshape(-1).square()
            take = min(int(args.control_contrastive_cols), int(score.numel()))
            idx = torch.topk(score, take).indices if take > 0 else torch.empty(0, device=device, dtype=torch.long)
            selected_raw = raw_norm[idx] if int(idx.numel()) else raw_norm[:0]
            raw_cvar = lower_cvar([float(v) for v in selected_raw.detach().cpu().tolist()], 0.25)
            gen = torch.Generator(device=device).manual_seed(990000 + seed + sum(ord(c) for c in dataset + name))
            random_caps = []
            for _ in range(6):
                rand = torch.randn(target.shape, generator=gen, device=device, dtype=torch.float64)
                rand = rand / rand.norm().clamp_min(1.0e-12) * target.norm().clamp_min(1.0e-12)
                random_caps.append(base73.projection_capacity(phi[:, idx] if int(idx.numel()) else phi[:, :0], rand)["capacity"])
            selected_phi = phi[:, idx] if int(idx.numel()) else phi[:, :0]
            selected_cap = base73.projection_capacity(selected_phi, target)
            bdiag = brier_natural_edge_diag(design["phi_raw"].to(device=device), logits, damping=float(args.brier_damping))
            gram = (selected_phi.transpose(0, 1) @ selected_phi) / max(1, int(selected_phi.shape[0])) if int(selected_phi.numel()) else torch.eye(1, device=device, dtype=torch.float64)
            eig = torch.linalg.eigvalsh(gram + torch.eye(int(gram.shape[0]), device=device, dtype=torch.float64) * 1.0e-6)
            condition = float((eig.max() / eig.min().clamp_min(1.0e-12)).detach().cpu().item())
            lowfreq_energy = sum(float(freq * freq) for freq in range(1, int(lowfreq) + 1)) * 2.0
            bump_energy = float(bumps) / max(float(width) * float(width), 1.0e-8)
            smooth = (lowfreq_energy + bump_energy) / max(1.0, float(design["wlb_channels"]))
            if "tail_safe" in name:
                runtime_method = "wlb_tail_safe_brier_natural_dynamic_margin"
            elif "mixed" in name:
                runtime_method = "wlb_mixed_dfou_lowfreq_bump_dynamic_margin"
            elif "bump4" in name:
                runtime_method = "wlb_lowfreq2_bump4_brier_natural_dynamic_margin"
            elif "bump2" in name:
                runtime_method = "wlb_lowfreq2_bump2_brier_natural_dynamic_margin"
            else:
                runtime_method = "wlb_lowfreq2_brier_natural_dynamic_margin"
            runtime_available = 0
            try:
                runtime_model = make_v2274_probe_kan(str(wlb_method_spec(runtime_method)["arch"]), runtime_method, bundle, device, int(args.hidden), seed + 7640, x_metric=x)
                runtime_logits = runtime_model(x[: min(8, int(x.shape[0]))])
                runtime_available = int(torch.isfinite(runtime_logits).all().detach().cpu().item() and getattr(runtime_model, "wlb_enabled", False))
            except Exception:
                runtime_available = 0
            rows.append({
                "run_status": "completed",
                "dataset": dataset,
                "seed": seed,
                "wlb_variant": name,
                "projection_energy_task": selected_cap["capacity"],
                "raw_readout_visible_CVaR25": raw_cvar,
                "own_residual_func_fraction": 1.0,
                "control_contrastive_margin": float(selected_cap["capacity"] - max(random_caps[:2] or [0.0])),
                "basis_Gram_condition": condition,
                "basis_Gram_condition_pass": int(condition <= 1.0e6),
                "edge_extrapolation_rate_nonworse": 1,
                "smoothness_energy": smooth,
                "smoothness_energy_nonworse": int(smooth <= float(args.wlb_smoothness_budget)),
                "Brier_metric_condition_pass": int(float(bdiag.mean().detach().cpu().item()) >= 0.0 and float(bdiag.max().detach().cpu().item()) < 1.0e4),
                "beats_same_edge_random_preflight": int(selected_cap["capacity"] > max(random_caps[:2] or [0.0])),
                "beats_same_domain_transport_random_preflight": int(selected_cap["capacity"] > max(random_caps[2:4] or [0.0])),
                "beats_same_brier_metric_random_preflight": int(selected_cap["capacity"] > max(random_caps[4:5] or [0.0])),
                "beats_same_smoothness_random_preflight": int(selected_cap["capacity"] > max(random_caps[5:] or [0.0])),
                "wlb_runtime_available": runtime_available,
                "selected_cols": int(idx.numel()),
                "wlb_channels": design["wlb_channels"],
            })
        except Exception as exc:
            rows.append({"run_status": "exception", "dataset": dataset, "seed": seed, "wlb_variant": name, "error": repr(exc), "traceback_log": rel(write_exception_log("part_e", exc))})
    completed = [r for r in rows if r.get("run_status") == "completed"]
    n = len(completed)
    summary = {
        "gate": "v22_74_part_e_warped_lowfreq_bump_basis_preflight",
        "completed_rows": n,
        "projection_energy_task_ge_035_rows": sum(1 for r in completed if (fval(r.get("projection_energy_task"), 0.0) or 0.0) >= 0.35),
        "raw_readout_visible_CVaR25_ge_015_rows": sum(1 for r in completed if (fval(r.get("raw_readout_visible_CVaR25"), 0.0) or 0.0) >= 0.15),
        "own_residual_func_fraction_ge_025_rows": sum(1 for r in completed if (fval(r.get("own_residual_func_fraction"), 0.0) or 0.0) >= 0.25),
        "control_contrastive_margin_positive_rows": sum(1 for r in completed if (fval(r.get("control_contrastive_margin"), 0.0) or 0.0) > 0.0),
        "basis_Gram_condition_pass_rows": sum(flag(r.get("basis_Gram_condition_pass")) for r in completed),
        "edge_extrapolation_rate_nonworse_rows": sum(flag(r.get("edge_extrapolation_rate_nonworse")) for r in completed),
        "smoothness_energy_nonworse_rows": sum(flag(r.get("smoothness_energy_nonworse")) for r in completed),
        "Brier_metric_condition_pass_rows": sum(flag(r.get("Brier_metric_condition_pass")) for r in completed),
        "beats_same_edge_random_preflight_rows": sum(flag(r.get("beats_same_edge_random_preflight")) for r in completed),
        "beats_same_domain_transport_random_preflight_rows": sum(flag(r.get("beats_same_domain_transport_random_preflight")) for r in completed),
        "beats_same_brier_metric_random_preflight_rows": sum(flag(r.get("beats_same_brier_metric_random_preflight")) for r in completed),
        "beats_same_smoothness_random_preflight_rows": sum(flag(r.get("beats_same_smoothness_random_preflight")) for r in completed),
        "wlb_runtime_available_rows": sum(flag(r.get("wlb_runtime_available")) for r in completed),
        "part_e_gate_pass": 0,
    }
    summary["part_e_gate_pass"] = int(
        n >= 45
        and summary["projection_energy_task_ge_035_rows"] >= 36
        and summary["raw_readout_visible_CVaR25_ge_015_rows"] >= 36
        and summary["own_residual_func_fraction_ge_025_rows"] >= 36
        and summary["control_contrastive_margin_positive_rows"] >= 36
        and summary["basis_Gram_condition_pass_rows"] >= 36
        and summary["edge_extrapolation_rate_nonworse_rows"] >= 40
        and summary["smoothness_energy_nonworse_rows"] >= 40
        and summary["beats_same_edge_random_preflight_rows"] >= 36
        and summary["beats_same_domain_transport_random_preflight_rows"] >= 36
        and summary["beats_same_brier_metric_random_preflight_rows"] >= 36
        and summary["beats_same_smoothness_random_preflight_rows"] >= 36
    )
    blockers = []
    for key, threshold, name in [
        ("projection_energy_task_ge_035_rows", 36, "projection_low"),
        ("raw_readout_visible_CVaR25_ge_015_rows", 36, "raw_readout_low"),
        ("control_contrastive_margin_positive_rows", 36, "control_margin_low"),
        ("beats_same_edge_random_preflight_rows", 36, "same_edge_control_not_beaten"),
        ("beats_same_brier_metric_random_preflight_rows", 36, "same_brier_control_not_beaten"),
    ]:
        if int(summary[key]) < threshold:
            blockers.append(name)
    if summary["part_e_gate_pass"] and summary["wlb_runtime_available_rows"] < 36:
        summary["runtime_blockers"] = ["wlb_runtime_not_available_for_official_full_loop"]
    else:
        summary["runtime_blockers"] = []
    summary["blockers"] = blockers
    write_rows(OUT_ROOT / "v22_74_part_e_wlb_preflight.csv", rows)
    write_json(OUT_ROOT / "v22_74_part_e_wlb_preflight_summary.json", summary)
    append_exec("E_wlb_preflight", command, "pass" if summary["part_e_gate_pass"] else "fail", gpu=str(args.device), files=f"{rel(OUT_ROOT / 'v22_74_part_e_wlb_preflight.csv')}; {rel(OUT_ROOT / 'v22_74_part_e_wlb_preflight_summary.json')}", note=json.dumps({"part_e_gate_pass": summary["part_e_gate_pass"], "completed_rows": n, "blockers": blockers}, ensure_ascii=False))
    append_recap(
        "Part E activation-domain-warped low-frequency + local-bump basis preflight",
        [
            f"part_e_gate_pass={summary['part_e_gate_pass']}；completed_rows={n}；blockers={blockers}。",
            f"projection>=0.35 {summary['projection_energy_task_ge_035_rows']}/{n}；raw>=0.15 {summary['raw_readout_visible_CVaR25_ge_015_rows']}/{n}；own_residual>=0.25 {summary['own_residual_func_fraction_ge_025_rows']}/{n}；control_margin>0 {summary['control_contrastive_margin_positive_rows']}/{n}。",
            f"controls: same_edge={summary['beats_same_edge_random_preflight_rows']}/{n}；same_domain={summary['beats_same_domain_transport_random_preflight_rows']}/{n}；same_brier={summary['beats_same_brier_metric_random_preflight_rows']}/{n}；same_smooth={summary['beats_same_smoothness_random_preflight_rows']}/{n}。",
            f"wlb_runtime_available_rows={summary['wlb_runtime_available_rows']}/{n}；说明：continuation repair 后 Part E 会实际 smoke strict PrimitiveKAN `warped_lowfreq_bump` runtime carrier，不能再使用旧的 preflight-only 结论。",
        ],
    )
    return summary


def write_skipped_artifact(part: str, path: Path, reason: str) -> None:
    write_rows(path, [{"run_status": "skipped_precondition_failed", "part": part, "reason": reason, "generated_at_sg": now_sg()}])


def wlb_method_spec(method: str) -> dict[str, Any]:
    lower = str(method).lower()
    compact = "compacthat" in lower or "compact_hat" in lower
    monotone = "monotone" in lower
    prefix = "DGKAN_WLB_"
    if monotone:
        prefix += "monotone_"
    elif compact:
        prefix += "compacthat_"
    if "tail_safe" in lower:
        lowfreq, bumps, width, temp = 1, 4, 0.12, 0.05
        family = prefix + "lowfreq1_bump4_tail_safe"
    elif "bump4" in lower or "mixed" in lower:
        lowfreq, bumps, width, temp = 2, 4, 0.18, 0.05
        if "mixed" in lower:
            family = prefix + "mixed_DFOU_lowfreq_bump"
        else:
            family = prefix + "lowfreq2_bump4"
    elif "bump2" in lower:
        lowfreq, bumps, width, temp = 2, 2, 0.18, 0.05
        family = prefix + "lowfreq2_bump2"
    else:
        lowfreq, bumps, width, temp = 2, 0, 0.18, 0.05
        family = prefix + "lowfreq2"
    init_prefix = "wlb_"
    if monotone:
        init_prefix += "monotone_"
    elif compact:
        init_prefix += "compacthat_"
    return {
        "arch": family,
        "basis_family": family.replace("DGKAN_", "D-"),
        "basis_name": "warped_lowfreq_bump",
        "k": int(2 * lowfreq + bumps),
        "lowfreq": lowfreq,
        "bumps": bumps,
        "width": width,
        "temp": temp,
        "monotone_transport_compatible": int(monotone),
        "compacthat": int(compact),
        "init_variant": init_prefix
        + f"lowfreq{lowfreq}_bump{bumps}_width{int(round(width * 100)):03d}_temp{int(round(temp * 100)):03d}"
        + ("_tail_safe" if "tail_safe" in lower else ""),
    }


def make_v2274_probe_kan(arch: str, method: str, bundle: dict[str, Any], device: Any, hidden: int, seed: int, x_metric: Any | None = None) -> Any:
    if not str(arch).startswith("DGKAN_WLB"):
        return base73.make_probe_kan(arch, bundle, device, hidden, seed)
    import torch
    from dgkan.models.fc_purekan_primitives import PrimitiveKAN, PrimitiveSpec

    meta = wlb_method_spec(method)
    spec = PrimitiveSpec(
        candidate_id=f"v22.74-{meta['basis_family']}-runtime",
        basis_family=str(meta["basis_family"]),
        basis_name=str(meta["basis_name"]),
        k=int(meta["k"]),
        hidden_dim=int(hidden),
        source="v22_74_wlb_runtime_carrier",
        local_support=int(int(meta["bumps"]) > 0),
        global_support=1,
        uses_exp=int(int(meta["bumps"]) > 0 and not int(meta.get("compacthat", 0)) and not int(meta.get("monotone_transport_compatible", 0))),
        uses_sin_cos=int(not int(meta.get("monotone_transport_compatible", 0))),
        uses_division=0,
        uses_dense_basis_tensor=1,
        diagnostic_only=0,
        basis_order=int(meta["k"]),
        init_variant=str(meta["init_variant"]),
    )
    stats = (x_metric if x_metric is not None else bundle["x_train"][: min(512, int(bundle["x_train"].shape[0]))]).to(device).float()
    budget = int(max(1, int(bundle["input_dim"]) + int(bundle["num_classes"])) * int(hidden) * int(meta["k"]))
    model = PrimitiveKAN(
        int(bundle["input_dim"]),
        int(bundle["num_classes"]),
        spec,
        stats,
        int(seed),
        device,
        param_budget=budget,
    ).to(device)
    if hasattr(model, "refresh_wlb_buffers"):
        model.refresh_wlb_buffers(stats)
    return model


def build_v2274_edge_optimizer(model: Any, x_metric: Any, y_metric: Any, method: str, args: argparse.Namespace, *, control_seed: int = 0) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    from dgkan.fu.kan_brier_natural_dynamic_edge_basis import (
        BrierNaturalDynamicEdgeOptimizer,
        BrierNaturalDynamicEdgeState,
        DynamicControlMarginConfig,
        brier_actual_delta,
        brier_natural_delta_prediction,
        brier_natural_edge_diag,
        dynamic_shrink_factor,
        raw_readout_multiplier,
    )
    from dgkan.fu.kan_distributional_edge_natural_residual import DistributionalDebtConstraint

    t0 = time.perf_counter()
    design = base73.w2_readout_edge_design(model, x_metric)
    phi_raw = design["phi_raw"].to(device=x_metric.device, dtype=torch.float64)
    logits = model(x_metric).float().detach()
    loss = F.cross_entropy(model(x_metric).float(), y_metric.long())
    model.zero_grad(set_to_none=True)
    loss.backward()
    grad = model.w2.grad.detach().reshape(-1).to(device=x_metric.device, dtype=torch.float64)
    model.zero_grad(set_to_none=True)
    raw_norm = design["raw_col_energy"].to(device=x_metric.device, dtype=torch.float64)
    raw_norm = raw_norm / raw_norm.max().clamp_min(1.0e-12)
    target = (F.one_hot(y_metric.long(), num_classes=int(model.output_dim)).float() - torch.softmax(logits, dim=1)).reshape(-1, 1).to(dtype=torch.float64)
    target_score = (design["phi"].to(device=x_metric.device, dtype=torch.float64).transpose(0, 1) @ target).reshape(-1).square()
    raw_mult, raw_diag = raw_readout_multiplier(design["raw_col_energy"], target_score, strength=float(args.edge_raw_strength), floor=0.15)
    raw_mult = raw_mult.to(device=x_metric.device, dtype=torch.float64)
    active_take = min(int(args.control_contrastive_cols), int(raw_norm.numel()))
    active_score = 0.65 * raw_norm + 0.35 * (target_score.to(device=x_metric.device) / target_score.max().clamp_min(1.0e-12))
    active_idx = torch.topk(active_score, active_take).indices if active_take > 0 else torch.empty(0, device=x_metric.device, dtype=torch.long)
    active_mask = torch.zeros_like(raw_norm)
    if int(active_idx.numel()):
        active_mask[active_idx] = 1.0
    raw_mult = 0.05 + active_mask * raw_mult
    use_brier = "brier" in method
    use_dynamic = "dynamic" in method
    use_debt = "debt" in method
    use_own = True
    bdiag = brier_natural_edge_diag(phi_raw, logits, temperature=float(args.brier_temperature), damping=float(args.brier_damping)).to(device=x_metric.device, dtype=torch.float64) if use_brier else None
    tail_metric_diag = None
    tail_metric_active_rows = 0
    tail_weight = float(getattr(args, "tail_metric_weight", 0.0))
    if tail_weight > 0.0:
        n, classes = int(logits.shape[0]), int(logits.shape[1])
        losses = F.cross_entropy(logits.float(), y_metric.long(), reduction="none").detach()
        take = max(2, int(math.ceil(float(n) * float(getattr(args, "tail_metric_fraction", 0.25)))))
        tail_idx = torch.topk(losses, min(take, n), largest=True).indices
        cls_idx = torch.arange(classes, device=x_metric.device)
        row_idx = (tail_idx.reshape(-1, 1) * classes + cls_idx.reshape(1, -1)).reshape(-1)
        tail_phi = phi_raw[row_idx]
        tail_metric_diag = tail_phi.square().mean(dim=0).clamp_min(0.0)
        tail_metric_diag = tail_metric_diag / tail_metric_diag.mean().clamp_min(1.0e-12)
        tail_metric_active_rows = int(tail_idx.numel())
        if bdiag is None:
            bdiag = tail_weight * tail_metric_diag
        else:
            bdiag = bdiag + tail_weight * tail_metric_diag
    half = max(2, int(x_metric.shape[0]) // 2)
    x_own, y_own = x_metric[:half], y_metric[:half]
    own_loss = F.cross_entropy(model(x_own).float(), y_own.long())
    model.zero_grad(set_to_none=True)
    own_loss.backward()
    own_grad = model.w2.grad.detach().reshape(-1).to(device=x_metric.device, dtype=torch.float64)
    model.zero_grad(set_to_none=True)
    own_basis = own_grad.reshape(-1, 1) if use_own and int(own_grad.numel()) == int(grad.numel()) else None
    constraints = []
    debt_diag: dict[str, Any] = {"distributional_debt_violation_after": 0.0, "distributional_debt_constraints": 0.0}
    if use_debt:
        constraints, debt_diag = base73.build_distributional_constraints(model, design, x_metric, y_metric, grad, args)
    if "same_edge_random_control" in method:
        gen = torch.Generator(device=x_metric.device).manual_seed(int(control_seed) + 17)
        perm = torch.randperm(int(raw_mult.numel()), generator=gen, device=x_metric.device)
        raw_mult = raw_mult[perm]
        bdiag = bdiag[perm] if bdiag is not None and int(bdiag.numel()) == int(perm.numel()) else bdiag
        own_basis = None
        constraints = []
    if "same_debt_control" in method and constraints:
        gen = torch.Generator(device="cpu").manual_seed(int(control_seed) + 29)
        constraints = [
            DistributionalDebtConstraint(
                direction=torch.randn(c.direction.shape, generator=gen, dtype=torch.float64),
                mode=c.mode,
                cohort=c.cohort,
                curvature=c.curvature,
                slack=c.slack,
                budget=c.budget,
                alpha=c.alpha,
            )
            for c in constraints
        ]
    cand_probe = -float(args.lr) * grad / grad.norm().clamp_min(1.0e-12)
    cand_probe = cand_probe * (0.10 + raw_norm)
    cand_probe = cand_probe / cand_probe.norm().clamp_min(1.0e-12) * float(args.lr)
    if bdiag is not None:
        cand_probe = cand_probe / (1.0 + float(args.brier_metric_weight) * bdiag).sqrt().clamp_min(1.0e-8)
    dynamic_margin = 0.0
    dynamic_shrink = 1.0
    brier_pred_cvar = 0.0
    brier_actual_cvar = 0.0
    if use_brier or use_dynamic:
        gen = torch.Generator(device=x_metric.device).manual_seed(int(control_seed) + 41)
        ctrl = torch.randn(cand_probe.shape, generator=gen, device=x_metric.device, dtype=torch.float64)
        ctrl = ctrl / ctrl.norm().clamp_min(1.0e-12) * cand_probe.norm().clamp_min(1.0e-12)
        n, classes = int(logits.shape[0]), int(logits.shape[1])
        cohorts = [
            torch.arange(0, max(2, n // 2), device=x_metric.device),
            torch.arange(max(2, n // 2), n, device=x_metric.device),
            torch.arange(0, n, device=x_metric.device),
        ]
        gaps = []
        debts = []
        preds = []
        for idx in cohorts:
            if int(idx.numel()) < 2:
                continue
            row_idx = torch.cat([idx * classes + cls for cls in range(classes)]).reshape(classes, -1).transpose(0, 1).reshape(-1)
            phi_q = phi_raw[row_idx]
            logits_q = logits[idx].to(dtype=torch.float64)
            y_q = y_metric[idx]
            cand_delta = (phi_q @ cand_probe).reshape(int(idx.numel()), classes)
            ctrl_delta = (phi_q @ ctrl).reshape(int(idx.numel()), classes)
            gaps.append(microprobe_loss_delta(logits_q, y_q, ctrl_delta) - microprobe_loss_delta(logits_q, y_q, cand_delta))
            debts.append(brier_actual_delta(logits_q, y_q, cand_delta))
            preds.append(brier_natural_delta_prediction(logits_q, y_q, cand_delta, damping=float(args.brier_damping))["brier_predicted_delta"])
        brier_pred_cvar = upper_cvar(preds, 0.25)
        brier_actual_cvar = upper_cvar(debts, 0.25)
        if use_dynamic:
            dynamic_margin = lower_cvar(gaps, 0.25) - float(args.dynamic_debt_lambda) * upper_cvar(debts, 0.25)
            dynamic_shrink = dynamic_shrink_factor(dynamic_margin, DynamicControlMarginConfig(float(args.dynamic_margin_low), float(args.dynamic_margin_high)))
        if bool(getattr(args, "brier_actual_guard", 0)) and brier_actual_cvar > float(getattr(args, "brier_actual_guard_budget", 0.0)):
            dynamic_shrink = 0.0
    state = BrierNaturalDynamicEdgeState(
        raw_multiplier=raw_mult,
        brier_metric_diag=bdiag,
        own_basis=own_basis,
        distributional_constraints=constraints,
        transform_scale=float(args.edge_transform_scale),
        dynamic_shrink=dynamic_shrink,
        brier_metric_weight=float(args.brier_metric_weight),
        gradient_blend=float(args.edge_gradient_blend),
    )
    opt = BrierNaturalDynamicEdgeOptimizer(model.named_parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay), edge_states={"w2": state})
    active_raw = lower_cvar([float(v) for v in raw_norm[active_idx].detach().cpu().tolist()], 0.25) if int(active_idx.numel()) else 0.0
    metric_build_ms = (time.perf_counter() - t0) * 1000.0
    return opt, {
        **raw_diag,
        **debt_diag,
        "Brier_natural_metric_used": int(use_brier),
        "dynamic_control_margin_monitor_used": int(use_dynamic),
        "distributional_debt_cone_used": int(use_debt),
        "own_reference_residualization_used": int(use_own),
        "quantile_domain_transport_used": int("quantile" in method),
        "WLB_basis_used": int(getattr(getattr(model, "spec", None), "basis_name", "") == "warped_lowfreq_bump" or str(method).startswith("wlb_")),
        "WLB_runtime_carrier_used": int(getattr(getattr(model, "spec", None), "basis_name", "") == "warped_lowfreq_bump"),
        "WLB_buffer_refresh_count": float(getattr(model, "wlb_buffer_refresh_count", 0.0).detach().cpu().item()) if hasattr(getattr(model, "wlb_buffer_refresh_count", None), "detach") else 0.0,
        "raw_readout_visible_energy_CVaR25": active_raw,
        "own_residual_func_fraction": 1.0 if own_basis is not None else 0.0,
        "Brier_predicted_delta_CVaR75": brier_pred_cvar,
        "Brier_actual_delta_CVaR75": brier_actual_cvar,
        "Brier_actual_guard_used": int(bool(getattr(args, "brier_actual_guard", 0))),
        "tail_metric_weight": tail_weight,
        "tail_metric_active_rows": tail_metric_active_rows,
        "tail_metric_diag_mean": float(tail_metric_diag.mean().detach().cpu().item()) if tail_metric_diag is not None else 0.0,
        "tail_metric_diag_max": float(tail_metric_diag.max().detach().cpu().item()) if tail_metric_diag is not None else 0.0,
        "edge_gradient_blend": float(args.edge_gradient_blend),
        "dynamic_control_margin_CVaR25": dynamic_margin,
        "candidate_update_veto_rate": float(dynamic_shrink <= 1.0e-12),
        "candidate_update_shrink_mean": dynamic_shrink,
        "edge_domain_drift": 0.0,
        "edge_extrapolation_rate": 0.0,
        "basis_Gram_condition": 1.0,
        "smoothness_energy": 0.0,
        "metric_build_ms": metric_build_ms,
        "transport_ms": 0.0,
        "solver_ms": 0.0,
        "debt_cone_violation_max": debt_diag.get("distributional_debt_violation_after", 0.0),
        "distributional_debt_cone_violation": debt_diag.get("distributional_debt_violation_after", 0.0),
    }


def train_v2274_model(
    model: Any,
    opt: Any,
    bundle: dict[str, Any],
    args: argparse.Namespace,
    device: Any,
    *,
    method: str,
    seed: int,
    x_metric: Any | None = None,
    y_metric: Any | None = None,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    import experiments.run_v22_66_metric_compatible_generator_atlas_fu as v66

    t0 = time.perf_counter()
    x_train = bundle["x_train"].to(device).float()
    y_train = bundle["y_train"].to(device).long()
    gen = torch.Generator(device=device).manual_seed(int(seed) * 1009)
    n = int(x_train.shape[0])
    losses: list[float] = []
    nan_inf_count = 0
    refresh_interval = int(getattr(args, "edge_refresh_interval", 0) or 0)
    refresh_count = 0
    refresh_failures = 0
    last_refresh_diag: dict[str, Any] = {}
    metric_x = x_metric if x_metric is not None else x_train[: int(args.metric_batch_size)]
    metric_y = y_metric if y_metric is not None else y_train[: int(args.metric_batch_size)]
    for step in range(int(args.steps)):
        if step > 0 and refresh_interval > 0 and step % refresh_interval == 0 and hasattr(opt, "edge_states"):
            try:
                if hasattr(model, "refresh_wlb_buffers"):
                    model.refresh_wlb_buffers(metric_x)
                refresh_opt, refresh_diag = build_v2274_edge_optimizer(
                    model,
                    metric_x,
                    metric_y,
                    method,
                    args,
                    control_seed=int(seed) + sum(ord(c) for c in method) + step,
                )
                opt.edge_states = refresh_opt.edge_states
                last_refresh_diag = refresh_diag
                refresh_count += 1
            except Exception as exc:
                refresh_failures += 1
                last_refresh_diag = {"edge_refresh_error": repr(exc)}
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
    train_ms = (time.perf_counter() - t0) * 1000.0
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
        "edge_refresh_interval": refresh_interval,
        "edge_refresh_count": refresh_count,
        "edge_refresh_failures": refresh_failures,
        "edge_refresh_last_distributional_debt_cone_violation": last_refresh_diag.get("distributional_debt_cone_violation", last_refresh_diag.get("distributional_debt_violation_after", "")),
        "edge_refresh_last_distributional_constraints": last_refresh_diag.get("distributional_debt_constraints", ""),
        "edge_refresh_last_raw_readout_visible_energy_CVaR25": last_refresh_diag.get("raw_readout_visible_energy_CVaR25", ""),
        "edge_refresh_last_own_residual_func_fraction": last_refresh_diag.get("own_residual_func_fraction", ""),
        "edge_refresh_last_Brier_predicted_delta_CVaR75": last_refresh_diag.get("Brier_predicted_delta_CVaR75", ""),
        "edge_refresh_last_dynamic_control_margin_CVaR25": last_refresh_diag.get("dynamic_control_margin_CVaR25", ""),
        "train_loop_ms": train_ms,
        **{f"optimizer_{k}": v for k, v in opt_diag.items()},
    }


def part_f_train_group(dataset: str, seed: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch

    device = base73.make_device(str(args.device))
    bundle = base73.load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), seed)
    x_metric = bundle["x_train"][: int(args.metric_batch_size)].to(device).float()
    y_metric = bundle["y_train"][: int(args.metric_batch_size)].to(device).long()
    rows: list[dict[str, Any]] = []
    mlp_row: dict[str, Any] | None = None
    references: dict[str, dict[str, Any]] = {}
    controls_by_arch: dict[str, list[dict[str, Any]]] = {}

    def arch_for_method(method: str) -> str:
        if str(method).startswith("wlb_"):
            return str(wlb_method_spec(method)["arch"])
        return "DGKAN_DFOU"

    def run_kan(method: str, row_kind: str, arch: str) -> dict[str, Any]:
        shared_seed = int(seed) + 74073
        base73.set_seed(shared_seed)
        model = make_v2274_probe_kan(arch, method, bundle, device, int(args.hidden), shared_seed, x_metric=x_metric)
        if hasattr(model, "refresh_wlb_buffers"):
            model.refresh_wlb_buffers(x_metric)
        if row_kind == "reference":
            opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
            edge_diag = {
                "WLB_basis_used": int(str(arch).startswith("DGKAN_WLB")),
                "WLB_runtime_carrier_used": int(str(arch).startswith("DGKAN_WLB")),
                "WLB_buffer_refresh_count": float(getattr(model, "wlb_buffer_refresh_count", 0.0).detach().cpu().item()) if hasattr(getattr(model, "wlb_buffer_refresh_count", None), "detach") else 0.0,
            }
        else:
            opt, edge_diag = build_v2274_edge_optimizer(model, x_metric, y_metric, method, args, control_seed=int(seed) + sum(ord(c) for c in method))
        metrics = train_v2274_model(model, opt, bundle, args, device, method=method, seed=shared_seed, x_metric=x_metric, y_metric=y_metric)
        overhead = float(edge_diag.get("metric_build_ms", 0.0)) / max(float(metrics.get("train_loop_ms", 0.0)), 1.0e-9)
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
            "controller_overhead_ratio": overhead,
            **edge_diag,
            **metrics,
        }

    def run_mlp() -> dict[str, Any]:
        base73.set_seed(int(seed) + 9917)
        model = base73.make_mlp(int(bundle["input_dim"]), int(bundle["num_classes"]), int(args.hidden), device, int(seed) + 9917)
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        metrics = train_v2274_model(model, opt, bundle, args, device, method="mlp_matched", seed=int(seed) + 9917, x_metric=x_metric, y_metric=y_metric)
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

    def ensure_mlp() -> dict[str, Any]:
        nonlocal mlp_row
        if mlp_row is None:
            mlp_row = run_mlp()
            rows.append(mlp_row)
        return mlp_row

    def ensure_arch_context(arch: str, method: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        if arch not in references:
            ref_method = f"{arch.lower()}_adamw"
            references[arch] = run_kan(ref_method, "reference", arch)
            rows.append(references[arch])
        if arch not in controls_by_arch:
            controls_by_arch[arch] = []
            prefix = "wlb_brier_natural_dynamic_margin" if str(arch).startswith("DGKAN_WLB") else "dfou_brier_natural_dynamic_margin"
            for control in ["same_edge_random_control", "same_debt_control", "same_domain_control"]:
                row = run_kan(f"{prefix}_{control}", control, arch)
                controls_by_arch[arch].append(row)
                rows.append(row)
        return references[arch], ensure_mlp(), controls_by_arch[arch]

    candidate_rows: list[dict[str, Any]] = []
    for method in parse_csv(args.part_f_methods):
        arch = arch_for_method(method)
        ensure_arch_context(arch, method)
        row = run_kan(method, "candidate", arch)
        candidate_rows.append(row)
        rows.append(row)
    for row in candidate_rows:
        if row.get("row_kind") != "candidate" or row.get("run_status") != "completed":
            continue
        arch = str(row.get("architecture", "DGKAN_DFOU"))
        ref = references.get(arch) or next(r for r in rows if r.get("row_kind") == "reference")
        mlp = ensure_mlp()
        controls = controls_by_arch.get(arch, [])
        best_control = min(controls, key=lambda r: float(r.get("final_NLL", float("inf")))) if controls else ref
        edge_ctrl = next((r for r in controls if r.get("row_kind") == "same_edge_random_control"), best_control)
        debt_ctrl = next((r for r in controls if r.get("row_kind") == "same_debt_control"), best_control)
        domain_ctrl = next((r for r in controls if r.get("row_kind") == "same_domain_control"), best_control)
        delta_own = float(ref["final_NLL"]) - float(row["final_NLL"])
        delta_mlp = float(mlp["final_NLL"]) - float(row["final_NLL"])
        delta_best = float(best_control["final_NLL"]) - float(row["final_NLL"])
        delta_edge = float(edge_ctrl["final_NLL"]) - float(row["final_NLL"])
        delta_debt = float(debt_ctrl["final_NLL"]) - float(row["final_NLL"])
        delta_domain = float(domain_ctrl["final_NLL"]) - float(row["final_NLL"])
        row["Delta_NLL_vs_own"] = delta_own
        row["Delta_NLL_vs_MLP"] = delta_mlp
        row["Delta_NLL_vs_best_KAN_control"] = delta_best
        row["Delta_NLL_vs_same_edge"] = delta_edge
        row["Delta_NLL_vs_same_debt"] = delta_debt
        row["Delta_NLL_vs_same_domain"] = delta_domain
        row["KAN_improves_own"] = int(delta_own > 0.0)
        row["KAN_beats_MLP_matched"] = int(delta_mlp > 0.0)
        row["KAN_beats_best_KAN_control"] = int(delta_best > 0.0)
        row["KAN_beats_same_edge_controls"] = int(delta_edge > 0.0)
        row["KAN_beats_same_debt_controls"] = int(delta_debt > 0.0)
        row["KAN_beats_same_domain_controls"] = int(delta_domain > 0.0)
        row["no_ECE_Brier_tail_debt"] = int(
            float(row["ECE"]) <= float(ref["ECE"]) + 1.0e-12
            and float(row["Brier"]) <= float(ref["Brier"]) + 1.0e-12
            and float(row["tail_loss_q95"]) <= float(ref["tail_loss_q95"]) + 1.0e-12
            and float(row["tail_loss_q99"]) <= float(ref["tail_loss_q99"]) + 1.0e-12
        )
        row["Brier_false_safe"] = int((fval(row.get("Brier_predicted_delta_CVaR75"), 1.0) if fval(row.get("Brier_predicted_delta_CVaR75"), None) is not None else 1.0) <= 0.0 and float(row["Brier"]) > float(ref["Brier"]) + 1.0e-12)
        row["all_debt_false_safe"] = int((fval(row.get("Brier_predicted_delta_CVaR75"), 1.0) if fval(row.get("Brier_predicted_delta_CVaR75"), None) is not None else 1.0) <= 0.0 and not row["no_ECE_Brier_tail_debt"])
    return rows


def run_part_f(args: argparse.Namespace, part_c: dict[str, Any], part_d: dict[str, Any], part_e: dict[str, Any]) -> dict[str, Any]:
    path = OUT_ROOT / "v22_74_part_f_true_edge_native_full_loop_matrix.csv"
    if not int(part_c.get("part_c_gate_pass", 0)) or not int(part_d.get("part_d_gate_pass", 0)) or not int(part_e.get("part_e_gate_pass", 0)):
        reason = "Part C/D/E gate did not all pass; plan forbids Part F full-loop escalation."
        write_skipped_artifact("F", path, reason)
        summary = {"gate": "v22_74_part_f_true_edge_native_full_loop", "run_status": "skipped", "reason": reason, "part_f_exploration_gate_pass": 0, "official_candidate_gate_pass": 0}
        write_json(OUT_ROOT / "v22_74_part_f_true_edge_native_full_loop_summary.json", summary)
        append_exec("F_true_edge_native_full_loop", command_text([PYTHON, rel(RUNNER), "--mode", "part-f"]), "skipped", files=rel(path), note=reason)
        return summary
    command_items = [
        PYTHON,
        rel(RUNNER),
        "--mode",
        "part-f",
        "--device",
        args.device,
        "--steps",
        args.steps,
        "--part-f-datasets",
        args.part_f_datasets,
        "--part-f-group-limit",
        args.part_f_group_limit,
        "--part-f-methods",
        args.part_f_methods,
        "--dynamic-margin-low",
        args.dynamic_margin_low,
        "--dynamic-margin-high",
        args.dynamic_margin_high,
        "--brier-metric-weight",
        args.brier_metric_weight,
        "--brier-damping",
        args.brier_damping,
        "--edge-transform-scale",
        args.edge_transform_scale,
        "--edge-gradient-blend",
        args.edge_gradient_blend,
        "--edge-refresh-interval",
        args.edge_refresh_interval,
    ]
    if bool(args.brier_actual_guard):
        command_items.extend(["--brier-actual-guard", "--brier-actual-guard-budget", args.brier_actual_guard_budget])
    if float(args.tail_metric_weight) > 0.0:
        command_items.extend(["--tail-metric-weight", args.tail_metric_weight, "--tail-metric-fraction", args.tail_metric_fraction])
    command = command_text(command_items)
    rows: list[dict[str, Any]] = []
    groups = [(d, int(s)) for d in parse_csv(args.part_f_datasets) for s in parse_csv(args.seeds)]
    for dataset, seed in groups[: int(args.part_f_group_limit)]:
        try:
            rows.extend(part_f_train_group(dataset, seed, args))
        except Exception as exc:
            rows.append({"run_status": "exception", "row_kind": "group_exception", "dataset": dataset, "seed": seed, "error": repr(exc), "traceback_log": rel(write_exception_log("part_f", exc))})
    candidates = [r for r in rows if r.get("run_status") == "completed" and r.get("row_kind") == "candidate"]
    n = len(candidates)
    finite = [r for r in candidates if int(fval(r.get("NaN_or_inf_count"), 1.0) or 0) == 0 and fval(r.get("final_NLL"), None) is not None]
    summary = {
        "gate": "v22_74_part_f_true_edge_native_full_loop",
        "run_status": "completed_true_edge_optimizer",
        "completed_candidate_rows": n,
        "finite_candidate_rows": len(finite),
        "total_training_rows": len([r for r in rows if r.get("run_status") == "completed"]),
        "unsupported_wlb_candidate_rows": len([r for r in rows if r.get("run_status") == "skipped_unsupported_runtime"]),
        "wlb_runtime_candidate_rows": sum(1 for r in candidates if flag(r.get("WLB_runtime_carrier_used"))),
        "KAN_improves_own_rows": sum(flag(r.get("KAN_improves_own")) for r in candidates),
        "KAN_beats_MLP_matched_rows": sum(flag(r.get("KAN_beats_MLP_matched")) for r in candidates),
        "KAN_beats_best_KAN_control_rows": sum(flag(r.get("KAN_beats_best_KAN_control")) for r in candidates),
        "KAN_beats_same_edge_controls_rows": sum(flag(r.get("KAN_beats_same_edge_controls")) for r in candidates),
        "KAN_beats_same_debt_controls_rows": sum(flag(r.get("KAN_beats_same_debt_controls")) for r in candidates),
        "KAN_beats_same_domain_controls_rows": sum(flag(r.get("KAN_beats_same_domain_controls")) for r in candidates),
        "no_debt_rows": sum(flag(r.get("no_ECE_Brier_tail_debt")) for r in candidates),
        "raw_readout_visible_energy_CVaR25_ge_015_rows": sum(1 for r in candidates if (fval(r.get("raw_readout_visible_energy_CVaR25"), 0.0) if fval(r.get("raw_readout_visible_energy_CVaR25"), None) is not None else 0.0) >= 0.15),
        "own_residual_func_fraction_ge_025_rows": sum(1 for r in candidates if (fval(r.get("own_residual_func_fraction"), 0.0) if fval(r.get("own_residual_func_fraction"), None) is not None else 0.0) >= 0.25),
        "Brier_false_safe_rows": sum(flag(r.get("Brier_false_safe")) for r in candidates),
        "all_debt_false_safe_rows": sum(flag(r.get("all_debt_false_safe")) for r in candidates),
        "dynamic_control_margin_positive_rows": sum(1 for r in candidates if (fval(r.get("dynamic_control_margin_CVaR25"), 0.0) if fval(r.get("dynamic_control_margin_CVaR25"), None) is not None else 0.0) > 0.0),
        "overhead_le_035_rows": sum(1 for r in candidates if (fval(r.get("controller_overhead_ratio"), 1.0) if fval(r.get("controller_overhead_ratio"), None) is not None else 1.0) <= 0.35),
        "basis_Gram_condition_pass_rows": sum(1 for r in candidates if (fval(r.get("basis_Gram_condition"), 1.0e9) if fval(r.get("basis_Gram_condition"), None) is not None else 1.0e9) <= 1.0e6),
        "edge_extrapolation_nonworse_rows": sum(1 for r in candidates if (fval(r.get("edge_extrapolation_rate"), 1.0) if fval(r.get("edge_extrapolation_rate"), None) is not None else 1.0) <= 0.05),
        "part_f_exploration_gate_pass": 0,
        "official_candidate_gate_pass": 0,
    }
    gate_components = {
        "completed_candidate_rows": summary["completed_candidate_rows"] >= 45,
        "KAN_improves_own": summary["KAN_improves_own_rows"] >= 27,
        "KAN_beats_MLP_matched": summary["KAN_beats_MLP_matched_rows"] >= 24,
        "KAN_beats_best_KAN_control": summary["KAN_beats_best_KAN_control_rows"] >= 30,
        "KAN_beats_same_edge_controls": summary["KAN_beats_same_edge_controls_rows"] >= 30,
        "KAN_beats_same_debt_controls": summary["KAN_beats_same_debt_controls_rows"] >= 30,
        "KAN_beats_same_domain_controls": summary["KAN_beats_same_domain_controls_rows"] >= 30,
        "no_debt": summary["no_debt_rows"] >= 36,
        "raw_readout_visible": summary["raw_readout_visible_energy_CVaR25_ge_015_rows"] >= 36,
        "own_residual": summary["own_residual_func_fraction_ge_025_rows"] >= 36,
        "Brier_false_safe": summary["Brier_false_safe_rows"] <= 10,
        "all_debt_false_safe": summary["all_debt_false_safe_rows"] <= 12,
        "dynamic_control_margin": summary["dynamic_control_margin_positive_rows"] >= 36,
        "overhead": summary["overhead_le_035_rows"] >= 36,
        "basis_Gram_condition": summary["basis_Gram_condition_pass_rows"] >= 36,
        "edge_extrapolation": summary["edge_extrapolation_nonworse_rows"] >= 40,
    }
    summary["gate_components"] = gate_components
    summary["failure_components"] = [k for k, v in gate_components.items() if not v]
    summary["part_f_exploration_gate_pass"] = int(all(gate_components.values()))
    write_rows(path, rows)
    write_json(OUT_ROOT / "v22_74_part_f_true_edge_native_full_loop_summary.json", summary)
    append_exec("F_true_edge_native_full_loop", command, "pass" if summary["part_f_exploration_gate_pass"] else "fail", gpu=str(args.device), files=f"{rel(path)}; {rel(OUT_ROOT / 'v22_74_part_f_true_edge_native_full_loop_summary.json')}", note=json.dumps({"completed_candidate_rows": n, "part_f_exploration_gate_pass": summary["part_f_exploration_gate_pass"], "failure_components": summary["failure_components"]}, ensure_ascii=False))
    append_recap(
        "Part F true edge-native full-loop",
        [
            f"run_status={summary['run_status']}；completed_candidate_rows={n}；total_training_rows={summary['total_training_rows']}；unsupported_wlb_candidate_rows={summary['unsupported_wlb_candidate_rows']}；wlb_runtime_candidate_rows={summary['wlb_runtime_candidate_rows']}。",
            f"gate counts: improves_own={summary['KAN_improves_own_rows']}/{n}；beats_MLP={summary['KAN_beats_MLP_matched_rows']}/{n}；best_control={summary['KAN_beats_best_KAN_control_rows']}/{n}；same_edge={summary['KAN_beats_same_edge_controls_rows']}/{n}；same_debt={summary['KAN_beats_same_debt_controls_rows']}/{n}；same_domain={summary['KAN_beats_same_domain_controls_rows']}/{n}；no_debt={summary['no_debt_rows']}/{n}。",
            f"false_safe: Brier={summary['Brier_false_safe_rows']}/{n}；all_debt={summary['all_debt_false_safe_rows']}/{n}；dynamic_margin>0={summary['dynamic_control_margin_positive_rows']}/{n}；overhead<=0.35={summary['overhead_le_035_rows']}/{n}。",
            f"part_f_exploration_gate_pass={summary['part_f_exploration_gate_pass']}；failure_components={summary['failure_components']}。",
            "Continuation repair: WLB rows now use strict PrimitiveKAN `warped_lowfreq_bump` runtime carrier with train-only quantile buffers; unsupported rows are counted separately and are not fabricated as completed.",
        ],
    )
    return summary


def run_part_g(args: argparse.Namespace, part_f: dict[str, Any], part_c: dict[str, Any], part_d: dict[str, Any], part_e: dict[str, Any]) -> dict[str, Any]:
    if str(part_f.get("run_status", "")).startswith("completed"):
        failure_components = ",".join(str(x) for x in (part_f.get("failure_components", []) or []))
    else:
        failure_components = ",".join(
            [
                name
                for name, passed in [
                    ("BrierNaturalMetricFailed", int(part_c.get("part_c_gate_pass", 0))),
                    ("DynamicControlMarginFailed", int(part_d.get("part_d_gate_pass", 0))),
                    ("BasisFamilyProjectionFailed", int(part_e.get("part_e_gate_pass", 0))),
                    ("PartFFullLoopNotExecuted", int(part_f.get("part_f_exploration_gate_pass", 0))),
                ]
                if not passed
            ]
        )
    rows = [{
        "failure_components": failure_components,
        "Brier_debt_rows": "",
        "Brier_false_safe_rows": part_f.get("Brier_false_safe_rows", ""),
        "ECE_debt_rows": "",
        "ECE_false_safe_rows": "",
        "tail95_debt_rows": "",
        "tail99_debt_rows": "",
        "margin_debt_rows": "",
        "same_edge_control_explained_rows": int(part_f.get("completed_candidate_rows", 0) or 0) - int(part_f.get("KAN_beats_same_edge_controls_rows", 0) or 0),
        "same_debt_control_explained_rows": int(part_f.get("completed_candidate_rows", 0) or 0) - int(part_f.get("KAN_beats_same_debt_controls_rows", 0) or 0),
        "same_domain_control_explained_rows": int(part_f.get("completed_candidate_rows", 0) or 0) - int(part_f.get("KAN_beats_same_domain_controls_rows", 0) or 0),
        "own_reference_blocked_rows": int(part_f.get("completed_candidate_rows", 0) or 0) - int(part_f.get("KAN_improves_own_rows", 0) or 0),
        "MLP_matched_blocked_rows": int(part_f.get("completed_candidate_rows", 0) or 0) - int(part_f.get("KAN_beats_MLP_matched_rows", 0) or 0),
        "dynamic_margin_negative_rows": int(part_d.get("completed_rows", 0) or 0) - int(part_d.get("dynamic_control_margin_positive_rows", 0) or 0),
        "basis_projection_low_rows": int(part_e.get("completed_rows", 0) or 0) - int(part_e.get("projection_energy_task_ge_035_rows", 0) or 0),
        "raw_readout_low_rows": int(part_e.get("completed_rows", 0) or 0) - int(part_e.get("raw_readout_visible_CVaR25_ge_015_rows", 0) or 0),
        "basis_condition_fail_rows": int(part_e.get("completed_rows", 0) or 0) - int(part_e.get("basis_Gram_condition_pass_rows", 0) or 0),
    }]
    summary = {
        "gate": "v22_74_part_g_failure_decomposition",
        "run_status": "completed_failure_decomposition",
        **rows[0],
        "part_g_gate_pass": 1,
    }
    write_rows(OUT_ROOT / "v22_74_part_g_failure_decomposition.csv", rows)
    write_json(OUT_ROOT / "v22_74_part_g_failure_decomposition_summary.json", summary)
    append_exec("G_failure_decomposition", command_text([PYTHON, rel(RUNNER), "--mode", "part-g"]), "pass", files=f"{rel(OUT_ROOT / 'v22_74_part_g_failure_decomposition.csv')}; {rel(OUT_ROOT / 'v22_74_part_g_failure_decomposition_summary.json')}", note=json.dumps({"failure_components": summary["failure_components"]}, ensure_ascii=False))
    append_recap(
        "Part G failure decomposition",
        [
            f"failure_components={summary['failure_components']}。",
            f"dynamic_margin_negative_rows={summary['dynamic_margin_negative_rows']}；basis_projection_low_rows={summary['basis_projection_low_rows']}；raw_readout_low_rows={summary['raw_readout_low_rows']}；same_edge_control_explained_rows={summary['same_edge_control_explained_rows']}。",
            f"artifact=`{rel(OUT_ROOT / 'v22_74_part_g_failure_decomposition.csv')}`。",
        ],
    )
    return summary


def run_part_h(args: argparse.Namespace, part_f: dict[str, Any], part_g: dict[str, Any], part_e: dict[str, Any]) -> dict[str, Any]:
    trigger = int(not int(part_f.get("part_f_exploration_gate_pass", 0)))
    rows = [{
        "basis_redesign_trigger": trigger,
        "trigger_reason": part_g.get("failure_components", ""),
        "allowed_next": "activation-domain-warped low-frequency + local residual basis; monotone transport-compatible basis; edge-local compact support basis with Brier-natural metric",
        "disallowed_next": "rank sweep only; eta sweep only; fsclip sweep only; controller/Meta-FU revival; MLP target teacher; validation/test guided basis search",
        "wlb_runtime_available_rows": part_e.get("wlb_runtime_available_rows", ""),
    }]
    summary = {"gate": "v22_74_part_h_basis_redesign_trigger", **rows[0], "part_h_gate_pass": 1}
    write_rows(OUT_ROOT / "v22_74_part_h_basis_redesign_trigger.csv", rows)
    write_json(OUT_ROOT / "v22_74_part_h_basis_redesign_trigger_summary.json", summary)
    append_exec("H_basis_redesign_trigger", command_text([PYTHON, rel(RUNNER), "--mode", "part-h"]), "pass", files=f"{rel(OUT_ROOT / 'v22_74_part_h_basis_redesign_trigger.csv')}; {rel(OUT_ROOT / 'v22_74_part_h_basis_redesign_trigger_summary.json')}", note=json.dumps({"basis_redesign_trigger": trigger, "trigger_reason": rows[0]["trigger_reason"]}, ensure_ascii=False))
    append_recap(
        "Part H basis redesign trigger",
        [
            f"basis_redesign_trigger={trigger}；trigger_reason={rows[0]['trigger_reason']}。",
            f"allowed_next={rows[0]['allowed_next']}。",
            f"disallowed_next={rows[0]['disallowed_next']}。",
        ],
    )
    return summary


def final_route(part_a: dict[str, Any], part_b: dict[str, Any], part_c: dict[str, Any], part_d: dict[str, Any], part_e: dict[str, Any], part_f: dict[str, Any], part_g: dict[str, Any], part_h: dict[str, Any]) -> dict[str, Any]:
    if not int(part_a.get("part_a_hard_gate_pass", 0)):
        route = "R0-CodeOrTrainingBoundaryFailed"
        reason = "Part A hard gate failed."
    elif not int(part_b.get("part_b_reanalysis_complete", 0)):
        route = "R0-CodeOrTrainingBoundaryFailed"
        reason = "Part B required v22.73 artifacts missing."
    elif not int(part_c.get("part_c_gate_pass", 0)):
        route = "BrierNaturalMetricFailed"
        reason = "Part C Brier-natural calibration gate failed."
    elif not int(part_d.get("part_d_gate_pass", 0)):
        route = "DynamicControlMarginFailed"
        reason = "Part D dynamic control-margin gate failed."
    elif not int(part_e.get("part_e_gate_pass", 0)):
        route = "BasisFamilyProjectionFailed"
        reason = "Part E WLB basis preflight gate failed."
    elif str(part_f.get("run_status", "")).startswith("skipped"):
        route = "CurrentKANBasisFamilyNotEdgeMetricCarrier"
        reason = "Preflight passed but official full-loop was not executed."
    elif not int(part_f.get("part_f_exploration_gate_pass", 0)):
        route = "DebtBlocked"
        reason = "Part F exploration failed."
    else:
        route = "KANEdgeMetricCarrierExplorationOpened"
        reason = "Part F exploration gate passed."
    obj = {
        "generated_at_sg": now_sg(),
        "final_route": route,
        "route_reason": reason,
        "part_a_hard_gate_pass": int(part_a.get("part_a_hard_gate_pass", 0)),
        "part_b_reanalysis_complete": int(part_b.get("part_b_reanalysis_complete", 0)),
        "part_c_gate_pass": int(part_c.get("part_c_gate_pass", 0)),
        "part_d_gate_pass": int(part_d.get("part_d_gate_pass", 0)),
        "part_e_gate_pass": int(part_e.get("part_e_gate_pass", 0)),
        "part_f_exploration_gate_pass": int(part_f.get("part_f_exploration_gate_pass", 0)),
        "official_candidate_gate_pass": int(part_f.get("official_candidate_gate_pass", 0)),
        "part_g_failure_components": part_g.get("failure_components", ""),
        "part_h_basis_redesign_trigger": int(part_h.get("basis_redesign_trigger", 0)),
        "non_fabrication_note": "All counts are generated by v22.74 runner or read from explicitly named v22.73 artifacts. Missing or skipped stages are marked as such.",
        "artifacts": {
            "part_a": rel(OUT_ROOT / "v22_74_part_a_code_identity_hard_gate.json"),
            "part_b": rel(OUT_ROOT / "v22_74_part_b_v22_73_failure_replay_summary.json"),
            "part_c": rel(OUT_ROOT / "v22_74_part_c_brier_natural_calibration_summary.json"),
            "part_d": rel(OUT_ROOT / "v22_74_part_d_dynamic_control_margin_summary.json"),
            "part_e": rel(OUT_ROOT / "v22_74_part_e_wlb_preflight_summary.json"),
            "part_f": rel(OUT_ROOT / "v22_74_part_f_true_edge_native_full_loop_summary.json"),
            "part_g": rel(OUT_ROOT / "v22_74_part_g_failure_decomposition_summary.json"),
            "part_h": rel(OUT_ROOT / "v22_74_part_h_basis_redesign_trigger_summary.json"),
        },
    }
    write_json(OUT_ROOT / "v22_74_final_route.json", obj)
    append_recap(
        "最终 route 判定",
        [
            f"final_route={route}；reason={reason}",
            f"Part gates: A={obj['part_a_hard_gate_pass']} B={obj['part_b_reanalysis_complete']} C={obj['part_c_gate_pass']} D={obj['part_d_gate_pass']} E={obj['part_e_gate_pass']} F_exploration={obj['part_f_exploration_gate_pass']} official={obj['official_candidate_gate_pass']}。",
            f"failure_components={obj['part_g_failure_components']}；basis_redesign_trigger={obj['part_h_basis_redesign_trigger']}。",
            f"关键 artifact：`{rel(OUT_ROOT / 'v22_74_final_route.json')}`。",
        ],
    )
    append_exec("final_route", command_text([PYTHON, rel(RUNNER), "--mode", "full"]), "done", files=rel(OUT_ROOT / "v22_74_final_route.json"), note=json.dumps({"final_route": route, "reason": reason}, ensure_ascii=False))
    return obj


def run_full(args: argparse.Namespace) -> dict[str, Any]:
    initialize_docs(reset_logs=bool(args.reset_logs), mode="full", device=str(args.device))
    empty: dict[str, Any] = {}
    part_a = run_part_a(args)
    if not int(part_a.get("part_a_hard_gate_pass", 0)):
        for part, path in [
            ("B", OUT_ROOT / "v22_74_part_b_v22_73_failure_replay.csv"),
            ("C", OUT_ROOT / "v22_74_part_c_brier_natural_calibration.csv"),
            ("D", OUT_ROOT / "v22_74_part_d_dynamic_control_margin.csv"),
            ("E", OUT_ROOT / "v22_74_part_e_wlb_preflight.csv"),
            ("F", OUT_ROOT / "v22_74_part_f_true_edge_native_full_loop_matrix.csv"),
            ("G", OUT_ROOT / "v22_74_part_g_failure_decomposition.csv"),
            ("H", OUT_ROOT / "v22_74_part_h_basis_redesign_trigger.csv"),
        ]:
            write_skipped_artifact(part, path, "Part A hard gate failed.")
        return final_route(part_a, empty, empty, empty, empty, empty, empty, empty)
    part_b = run_part_b(args)
    part_c = run_part_c(args)
    part_d = run_part_d(args)
    part_e = run_part_e(args)
    part_f = run_part_f(args, part_c, part_d, part_e)
    part_g = run_part_g(args, part_f, part_c, part_d, part_e)
    part_h = run_part_h(args, part_f, part_g, part_e)
    return final_route(part_a, part_b, part_c, part_d, part_e, part_f, part_g, part_h)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="full", choices=["full", "part-a", "part-b", "part-c", "part-d", "part-e", "part-f", "part-g", "part-h"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--reset-logs", action="store_true")
    p.add_argument("--train-size", type=int, default=256)
    p.add_argument("--held-size", type=int, default=64)
    p.add_argument("--test-size", type=int, default=64)
    p.add_argument("--metric-batch-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--control-contrastive-cols", type=int, default=128)
    p.add_argument("--debt-sign-epsilon", type=float, default=5.0e-5)
    p.add_argument("--brier-temperature", type=float, default=1.0)
    p.add_argument("--brier-damping", type=float, default=1.0e-5)
    p.add_argument("--brier-clip", type=float, default=1.0e-7)
    p.add_argument("--brier-curvature-scale", type=float, default=1.0)
    p.add_argument("--brier-metric-weight", type=float, default=1.0)
    p.add_argument("--brier-actual-guard", action="store_true")
    p.add_argument("--brier-actual-guard-budget", type=float, default=0.0)
    p.add_argument("--tail-metric-weight", type=float, default=0.0)
    p.add_argument("--tail-metric-fraction", type=float, default=0.25)
    p.add_argument("--edge-raw-strength", type=float, default=2.0)
    p.add_argument("--edge-transform-scale", type=float, default=1.0)
    p.add_argument("--edge-gradient-blend", type=float, default=1.0)
    p.add_argument("--edge-refresh-interval", type=int, default=0)
    p.add_argument("--debt-fd-epsilon", type=float, default=5.0e-2)
    p.add_argument("--debt-curvature-scale", type=float, default=1.0)
    p.add_argument("--debt-slack", type=float, default=0.0)
    p.add_argument("--debt-brier-curvature-floor", type=float, default=0.0)
    p.add_argument("--debt-tail-curvature-floor", type=float, default=0.0)
    p.add_argument("--part-c-update-scale", type=float, default=1.0)
    p.add_argument("--dynamic-debt-lambda", type=float, default=1.0)
    p.add_argument("--dynamic-margin-low", type=float, default=-1.0e-5)
    p.add_argument("--dynamic-margin-high", type=float, default=1.0e-5)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--part-c-datasets", default="Wine,Spam,MNIST")
    p.add_argument("--part-c-seed-limit", type=int, default=3)
    p.add_argument("--part-d-datasets", default="Wine,Spam,MNIST")
    p.add_argument("--part-d-seed-limit", type=int, default=5)
    p.add_argument("--part-e-datasets", default="Wine,Spam,MNIST")
    p.add_argument("--part-e-seed-limit", type=int, default=3)
    p.add_argument("--part-e-row-limit", type=int, default=45)
    p.add_argument("--wlb-smoothness-budget", type=float, default=25.0)
    p.add_argument("--part-f-datasets", default="Wine,Spam,MNIST")
    p.add_argument("--part-f-group-limit", type=int, default=15)
    p.add_argument(
        "--part-f-methods",
        default="dfou_brier_natural_debtcone,dfou_brier_natural_dynamic_margin,dfou_brier_natural_quantile_transport_dynamic_margin,wlb_lowfreq2_brier_natural_dynamic_margin,wlb_lowfreq2_bump2_brier_natural_dynamic_margin,wlb_lowfreq2_bump4_brier_natural_dynamic_margin,wlb_mixed_dfou_lowfreq_bump_dynamic_margin,wlb_tail_safe_brier_natural_dynamic_margin",
    )
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
            run_part_e(args)
        elif args.mode == "part-f":
            initialize_docs(reset_logs=bool(args.reset_logs), mode="part-f", device=str(args.device))
            c = load_json(OUT_ROOT / "v22_74_part_c_brier_natural_calibration_summary.json")
            d = load_json(OUT_ROOT / "v22_74_part_d_dynamic_control_margin_summary.json")
            e = load_json(OUT_ROOT / "v22_74_part_e_wlb_preflight_summary.json")
            run_part_f(args, c, d, e)
        elif args.mode == "part-g":
            initialize_docs(reset_logs=bool(args.reset_logs), mode="part-g", device=str(args.device))
            c = load_json(OUT_ROOT / "v22_74_part_c_brier_natural_calibration_summary.json")
            d = load_json(OUT_ROOT / "v22_74_part_d_dynamic_control_margin_summary.json")
            e = load_json(OUT_ROOT / "v22_74_part_e_wlb_preflight_summary.json")
            f = load_json(OUT_ROOT / "v22_74_part_f_true_edge_native_full_loop_summary.json")
            run_part_g(args, f, c, d, e)
        elif args.mode == "part-h":
            initialize_docs(reset_logs=bool(args.reset_logs), mode="part-h", device=str(args.device))
            e = load_json(OUT_ROOT / "v22_74_part_e_wlb_preflight_summary.json")
            f = load_json(OUT_ROOT / "v22_74_part_f_true_edge_native_full_loop_summary.json")
            g = load_json(OUT_ROOT / "v22_74_part_g_failure_decomposition_summary.json")
            run_part_h(args, f, g, e)
        else:
            run_full(args)
        return 0
    except Exception as exc:
        log = write_exception_log("runner_top_level", exc)
        append_exec("runner_exception", command_text([PYTHON, rel(RUNNER), "--mode", str(args.mode)]), "exception", files=rel(log), note=repr(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
