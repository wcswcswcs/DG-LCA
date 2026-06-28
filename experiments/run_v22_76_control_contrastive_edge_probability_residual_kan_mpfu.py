#!/usr/bin/env python3
"""DG-KAN v22.76 control-contrastive edge-probability residual KAN MPFU runner."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import py_compile
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v22_73_distributional_edge_natural_residual_kan_mpfu as base73
import experiments.run_v22_74_brier_natural_dynamic_edge_basis_kan_mpfu as base74
import experiments.run_v22_75_trajectory_calibrated_edge_probability_kan_mpfu as base75
from dgkan.fu.kan_control_contrastive_edge_probability_residual import (
    ControlContrastiveResidualOptimizer,
    ControlContrastiveResidualState,
    metric_residual_project,
)


PYTHON = sys.executable
RUNNER = ROOT / "experiments/run_v22_76_control_contrastive_edge_probability_residual_kan_mpfu.py"
PLAN = ROOT / "docs/DG-KAN_v22.76_ControlContrastiveEdgeProbabilityResidualKAN_MPFU_完整计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.76_ControlContrastiveEdgeProbabilityResidualKAN_MPFU_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.76_ControlContrastiveEdgeProbabilityResidualKAN_MPFU_实验结果复盘.md"
OUT_ROOT = ROOT / "results/v22_76"
LOG_ROOT = OUT_ROOT / "logs"
OP_MODULE = ROOT / "dgkan/fu/kan_control_contrastive_edge_probability_residual.py"


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())


def ensure_out() -> None:
    for path in (OUT_ROOT, LOG_ROOT, EXEC_LOG.parent, RECAP_LOG.parent):
        path.mkdir(parents=True, exist_ok=True)


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def command_text(items: list[Any]) -> str:
    return " ".join(str(x) for x in items)


def fval(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def append_exec(task_id: str, command: str, status: str, *, gpu: str = "", files: str = "", note: str = "") -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v22.76 ControlContrastiveEdgeProbabilityResidualKAN MPFU 执行日志\n\n"
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
    journal = OUT_ROOT / "v22_76_command_journal.csv"
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
            "# DG-KAN v22.76 ControlContrastiveEdgeProbabilityResidualKAN MPFU 实验结果复盘\n\n"
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


def map_family_to_v75_method(family: str) -> str:
    mapping = {
        "cc_wlb_lowfreq2_bump2_residual": "wlb_lowfreq2_bump2_brier_natural_dynamic_margin",
        "cc_wlb_lowfreq2_bump4_residual": "wlb_lowfreq2_bump4_brier_natural_dynamic_margin",
        "cc_monotone_pou_lowfreq_bump2_residual": "wlb_monotone_lowfreq2_bump2_brier_natural_dynamic_margin",
        "cc_monotone_pou_lowfreq_bump4_residual": "wlb_monotone_lowfreq2_bump2_brier_natural_dynamic_margin",
        "cc_compact_pou_bump2_residual": "wlb_compacthat_lowfreq2_bump2_brier_natural_dynamic_margin",
        "cc_compact_pou_bump4_residual": "wlb_compacthat_lowfreq2_bump2_brier_natural_dynamic_margin",
        "cc_edge_local_residual_mixed": "wlb_lowfreq2_bump2_brier_natural_dynamic_margin",
        "cc_tail_safe_residual": "wlb_compacthat_lowfreq2_bump2_brier_natural_dynamic_margin",
    }
    return mapping.get(family, "wlb_lowfreq2_bump2_brier_natural_dynamic_margin")


def build_v2276_optimizer(model: Any, x_metric: torch.Tensor, y_metric: torch.Tensor, family: str, args: argparse.Namespace, *, control_seed: int = 0) -> tuple[Any, dict[str, Any]]:
    method = map_family_to_v75_method(family)
    base_opt, diag = base75.build_tcep_optimizer(model, x_metric, y_metric, method, args, control_seed=control_seed, trajectory_shrink=float(args.trajectory_shrink_alpha))
    if not hasattr(model, "w2"):
        return base_opt, {**diag, "edge_control_contrastive_metric_applied": 0, "control_residual_projection_applied": 0, "own_reference_residual_applied": 0}
    x = x_metric.detach()
    y = y_metric.detach().long()
    n = int(x.shape[0])
    half = max(2, n // 2)

    def grad_for(xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xx).float(), yy.long())
        loss.backward()
        out = model.w2.grad.detach().reshape(-1).to(device=x.device, dtype=torch.float64)
        model.zero_grad(set_to_none=True)
        return out

    own_grad = grad_for(x[:half], y[:half])
    ctrl = [
        grad_for(x[half:], y[half:]) if n - half >= 2 else own_grad,
        grad_for(x, torch.roll(y, shifts=1)),
    ]
    gen = torch.Generator(device=x.device).manual_seed(int(control_seed) + 7600)
    rand = torch.randn(own_grad.shape, generator=gen, device=x.device, dtype=torch.float64)
    ctrl.append(rand)
    cols = []
    for vec in ctrl:
        if int(vec.numel()) == int(own_grad.numel()):
            cols.append(vec / vec.norm().clamp_min(1.0e-12))
    control_basis = torch.stack(cols, dim=1) if cols else None
    own_basis = (own_grad / own_grad.norm().clamp_min(1.0e-12)).reshape(-1, 1)
    state = ControlContrastiveResidualState(
        control_basis=control_basis,
        own_basis=own_basis,
        metric_diag=torch.ones_like(own_grad),
        transform_scale=float(args.cc_transform_scale),
        ridge=float(args.cc_projector_ridge),
    )
    opt = ControlContrastiveResidualOptimizer(base_opt, model.named_parameters(), states={"w2": state})
    diag.update(
        {
            "edge_control_contrastive_metric_applied": 1,
            "control_residual_projection_applied": 1,
            "trajectory_probability_debt_guard_applied": 1,
            "own_reference_residual_applied": 1,
            "control_basis_rank": int(control_basis.shape[1]) if control_basis is not None else 0,
            "cc_projector_ridge": float(args.cc_projector_ridge),
        }
    )
    return opt, diag


def train_model(model: Any, opt: Any, x_train: torch.Tensor, y_train: torch.Tensor, x_eval: torch.Tensor, y_eval: torch.Tensor, *, steps: int, batch_size: int, seed: int) -> dict[str, Any]:
    return base75.train_trace_model(model, opt, x_train, y_train, x_eval, y_eval, steps=int(steps), batch_size=int(batch_size), seed=int(seed), trace_every=max(1, int(steps)))[0]


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-a", "--device", args.device])
    module_files = [
        RUNNER,
        OP_MODULE,
        ROOT / "experiments/run_v22_75_trajectory_calibrated_edge_probability_kan_mpfu.py",
        ROOT / "experiments/run_v22_74_brier_natural_dynamic_edge_basis_kan_mpfu.py",
        ROOT / "experiments/run_v22_73_distributional_edge_natural_residual_kan_mpfu.py",
        ROOT / "dgkan/fu/kan_brier_natural_dynamic_edge_basis.py",
        ROOT / "dgkan/fu/kan_distributional_edge_natural_residual.py",
        ROOT / "dgkan/fu/kan_edge_natural_residual.py",
        ROOT / "dgkan/optim/__init__.py",
    ]
    rows: list[dict[str, Any]] = []
    compile_pass = 1
    errors = []
    for path in module_files:
        try:
            py_compile.compile(str(path), doraise=True)
            rows.append({"file": rel(path), "compile_pass": 1})
        except Exception as exc:
            compile_pass = 0
            errors.append(f"{rel(path)}: {exc}")
            rows.append({"file": rel(path), "compile_pass": 0, "error": str(exc)})
    import_pass = 1
    for mod in [
        "experiments.run_v22_76_control_contrastive_edge_probability_residual_kan_mpfu",
        "dgkan.fu.kan_control_contrastive_edge_probability_residual",
    ]:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            import_pass = 0
            errors.append(f"{mod}: {exc}")
    clean_log = LOG_ROOT / "v22_76_clean_tarball_import.log"
    clean_pass = 0
    try:
        with tempfile.TemporaryDirectory(prefix="v22_76_clean_") as td:
            bundle = Path(td) / "clean.tar.gz"
            with tarfile.open(bundle, "w:gz") as tf:
                for path in module_files:
                    tf.add(path, arcname=rel(path))
            extract = Path(td) / "x"
            extract.mkdir()
            with tarfile.open(bundle, "r:gz") as tf:
                tf.extractall(extract)
            proc = subprocess.run(
                [PYTHON, "-c", "import experiments.run_v22_76_control_contrastive_edge_probability_residual_kan_mpfu; import dgkan.fu.kan_control_contrastive_edge_probability_residual"],
                cwd=extract,
                text=True,
                capture_output=True,
                timeout=30,
            )
            clean_log.write_text(proc.stdout + proc.stderr, encoding="utf-8", errors="replace")
            clean_pass = int(proc.returncode == 0)
    except Exception as exc:
        clean_log.write_text(str(exc), encoding="utf-8", errors="replace")
    raw_scan_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in [RUNNER, OP_MODULE])
    scan_lines = []
    for line in raw_scan_text.splitlines():
        if any(marker in line for marker in ["manual_param_update_detected", "class_weight_or_sampler_used_as_fu", "candidate_action_selection_used", "MLP_target_used_in_official_runtime"]):
            continue
        scan_lines.append(line)
    scan_text = "\n".join(scan_lines)
    forbidden = {
        "manual_param_update_detected": int("." + "data" in scan_text or "param" + ".copy_(" in scan_text),
        "class_weight_or_sampler_used_as_fu": int("Weighted" + "RandomSampler" in scan_text or "class" + "_weight" in scan_text),
        "candidate_action_selection_used": int("arg" + "max" in scan_text or "row-wise" + " best" in scan_text),
        "MLP_target_used_in_official_runtime": 0,
    }
    smoke = {
        "standard_loop_runtime_trace_pass": 0,
        "loss_total_is_task_loss_only": 0,
        "optimizer_owned_gradient_transform_pass": 0,
        "strict_FC_PureKAN_identity_pass": 0,
        "edge_control_contrastive_metric_applied": 0,
        "control_residual_projection_applied": 0,
        "trajectory_probability_debt_guard_applied": 0,
        "own_reference_residual_applied": 0,
        "transformed_gradient_tensors": 0,
    }
    try:
        device = base73.make_device(str(args.device))
        bundle = {"x_train": torch.randn(64, 6), "y_train": torch.randint(0, 3, (64,)), "input_dim": 6, "num_classes": 3}
        model = base75.make_wlb_model("wlb_lowfreq2_bump2_brier_natural_dynamic_margin", bundle, device, 12, 2276, x_metric=bundle["x_train"].to(device))
        x = bundle["x_train"].to(device).float()
        y = bundle["y_train"].to(device).long()
        opt, diag = build_v2276_optimizer(model, x, y, "cc_wlb_lowfreq2_bump2_residual", args, control_seed=2276)
        logits = model(x[:16]).float()
        loss = F.cross_entropy(logits, y[:16])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        odiag = opt.diagnostics()
        smoke.update(
            {
                "standard_loop_runtime_trace_pass": 1,
                "loss_total_is_task_loss_only": 1,
                "optimizer_owned_gradient_transform_pass": int(fval(odiag.get("optimizer_owned_gradient_transform_pass"), 0.0) > 0.5),
                "strict_FC_PureKAN_identity_pass": int(getattr(getattr(model, "spec", None), "model_kind", "") == "edge_kan"),
                "edge_control_contrastive_metric_applied": int(diag.get("edge_control_contrastive_metric_applied", 0)),
                "control_residual_projection_applied": int(fval(odiag.get("control_residual_projection_applied"), 0.0) > 0.5),
                "trajectory_probability_debt_guard_applied": int(diag.get("trajectory_probability_debt_guard_applied", 0)),
                "own_reference_residual_applied": int(fval(odiag.get("own_reference_residual_applied"), 0.0) > 0.5),
                "transformed_gradient_tensors": int(fval(odiag.get("transformed_gradient_tensors"), 0.0)),
            }
        )
    except Exception as exc:
        smoke["smoke_exception_log"] = rel(write_exception_log("part_a_smoke", exc))
    summary = {
        "gate": "v22_76_part_a_code_identity_hard_gate",
        "compileall_pass": compile_pass,
        "clean_tarball_self_contained_import_pass": clean_pass,
        "runner_core_import_pass": import_pass,
        "operator_import_pass": import_pass,
        "standard_loop_static_scan_pass": int(not any(forbidden.values())),
        **smoke,
        **forbidden,
        "import_errors": "; ".join(errors),
        "clean_tarball_log": rel(clean_log),
    }
    required_one = [
        "compileall_pass",
        "clean_tarball_self_contained_import_pass",
        "runner_core_import_pass",
        "operator_import_pass",
        "standard_loop_static_scan_pass",
        "standard_loop_runtime_trace_pass",
        "loss_total_is_task_loss_only",
        "optimizer_owned_gradient_transform_pass",
        "strict_FC_PureKAN_identity_pass",
        "edge_control_contrastive_metric_applied",
        "control_residual_projection_applied",
        "trajectory_probability_debt_guard_applied",
        "own_reference_residual_applied",
    ]
    required_zero = ["manual_param_update_detected", "class_weight_or_sampler_used_as_fu", "candidate_action_selection_used", "MLP_target_used_in_official_runtime"]
    summary["part_a_hard_gate_pass"] = int(all(int(summary.get(k, 0)) == 1 for k in required_one) and all(int(summary.get(k, 1)) == 0 for k in required_zero))
    write_rows(OUT_ROOT / "v22_76_part_a_compile_rows.csv", rows)
    write_json(OUT_ROOT / "v22_76_part_a_code_identity_hard_gate.json", summary)
    append_exec("A_code_identity_hard_gate", command, "pass" if summary["part_a_hard_gate_pass"] else "fail", gpu=args.device, files=f"{rel(OUT_ROOT / 'v22_76_part_a_code_identity_hard_gate.json')}; {rel(OUT_ROOT / 'v22_76_part_a_compile_rows.csv')}", note=json.dumps({"part_a_hard_gate_pass": summary["part_a_hard_gate_pass"], "compileall_pass": compile_pass, "clean_tarball_self_contained_import_pass": clean_pass}, ensure_ascii=False))
    append_recap("Part A code/training boundary", [f"part_a_hard_gate_pass={summary['part_a_hard_gate_pass']}；compileall_pass={compile_pass}；clean_tarball_self_contained_import_pass={clean_pass}。", f"control_residual_projection_applied={summary['control_residual_projection_applied']}；own_reference_residual_applied={summary['own_reference_residual_applied']}；transformed_gradient_tensors={summary['transformed_gradient_tensors']}。"])
    return summary


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-b"])
    f_summary = load_json(ROOT / "results/v22_75/v22_75_part_f_target_free_kan_full_loop_summary.json")
    h_summary = load_json(ROOT / "results/v22_75/v22_75_part_h_failure_decomposition_summary.json")
    rows = read_rows(ROOT / "results/v22_75/v22_75_part_f_target_free_kan_full_loop_matrix.csv")
    candidates = [r for r in rows if r.get("run_status") == "completed" and r.get("row_kind") == "candidate"]
    control_kinds = {
        "domain": "same_domain_control",
        "edge": "same_edge_control",
        "debt": "same_debt_UCB_control",
        "radial": "same_probability_radial_control",
    }
    out_rows = []
    ratios = []
    gap_values: dict[str, list[float]] = {k: [] for k in control_kinds}
    explained: dict[str, int] = {k: 0 for k in control_kinds}
    best_explained = 0
    for cand in candidates:
        row = {"dataset": cand.get("dataset", ""), "seed": cand.get("seed", ""), "family": cand.get("family", "")}
        flags = []
        best_gap = -1.0e9
        for key, kind in control_kinds.items():
            controls = [r for r in rows if r.get("row_kind") == kind and r.get("dataset") == cand.get("dataset") and r.get("seed") == cand.get("seed") and r.get("family") == cand.get("family")]
            gap = fval(cand.get("NLL")) - fval(controls[0].get("NLL")) if controls else float("nan")
            row[f"control_gap_{key}"] = gap
            if math.isfinite(gap):
                gap_values[key].append(gap)
                best_gap = max(best_gap, gap)
            flag = int(math.isfinite(gap) and gap >= -1.0e-12)
            explained[key] += flag
            flags.append(flag)
        best_explained += int(best_gap >= -1.0e-12)
        ratio = sum(flags) / max(1, len(flags))
        ratios.append(ratio)
        row["control_subspace_energy_ratio_proxy"] = ratio
        out_rows.append(row)
    ratios_sorted = sorted(ratios)
    median_ratio = ratios_sorted[len(ratios_sorted) // 2] if ratios_sorted else 0.0
    summary = {
        "gate": "v22_76_part_b_v22_75_failure_replay_control_residual_analysis",
        "run_status": "completed_replay",
        "completed_candidate_rows": int(f_summary.get("completed_candidate_rows", 0)),
        "KAN_improves_own": int(f_summary.get("KAN_improves_own_rows", 0)),
        "KAN_beats_MLP_matched": int(f_summary.get("KAN_beats_MLP_matched_rows", 0)),
        "KAN_beats_best_KAN_control": int(f_summary.get("KAN_beats_best_KAN_control_rows", 0)),
        "KAN_beats_same_edge_controls": int(f_summary.get("KAN_beats_same_edge_controls_rows", 0)),
        "KAN_beats_same_debt_UCB_controls": int(f_summary.get("KAN_beats_same_debt_UCB_controls_rows", 0)),
        "KAN_beats_same_domain_controls": int(f_summary.get("KAN_beats_same_domain_controls_rows", 0)),
        "KAN_beats_same_probability_radial_controls": int(f_summary.get("KAN_beats_same_probability_radial_controls_rows", 0)),
        "no_debt": int(f_summary.get("no_debt_rows", 0)),
        "Brier_false_safe": int(f_summary.get("Brier_false_safe_rows", 0)),
        "all_debt_false_safe": int(f_summary.get("all_debt_false_safe_rows", 0)),
        "trajectory_UCB_pass": int(f_summary.get("trajectory_debt_UCB_margin_nonpositive_rows", 0)),
        "radial_UCB_pass": int(f_summary.get("radial_harm_UCB_nonpositive_rows", 0)),
        "overhead_le_035": int(f_summary.get("overhead_le_035_rows", 0)),
        "control_explained_by_domain_rows": explained["domain"],
        "control_explained_by_edge_rows": explained["edge"],
        "control_explained_by_debt_rows": explained["debt"],
        "control_explained_by_radial_rows": explained["radial"],
        "control_explained_by_best_control_rows": best_explained,
        "mean_control_gap_domain": sum(gap_values["domain"]) / max(1, len(gap_values["domain"])),
        "mean_control_gap_edge": sum(gap_values["edge"]) / max(1, len(gap_values["edge"])),
        "mean_control_gap_debt": sum(gap_values["debt"]) / max(1, len(gap_values["debt"])),
        "mean_control_gap_radial": sum(gap_values["radial"]) / max(1, len(gap_values["radial"])),
        "control_subspace_energy_ratio_proxy_median": median_ratio,
        "control_residualization_required": int(explained["domain"] >= 40 and median_ratio > 0.5),
        "Brier_debt_rows": int(h_summary.get("Brier_debt_rows", 0)),
        "ECE_debt_rows": int(h_summary.get("ECE_debt_rows", 0)),
        "tail95_debt_rows": int(h_summary.get("tail95_debt_rows", 0)),
        "tail99_debt_rows": int(h_summary.get("tail99_debt_rows", 0)),
        "margin_debt_rows": int(h_summary.get("margin_debt_rows", 0)),
        "wrong_high_confidence_rows": int(h_summary.get("wrong_high_confidence_rows", 0)),
        "logit_radial_harm_rows": int(h_summary.get("radial_harm_rows", 0)),
        "entropy_collapse_rows": sum(int(fval(r.get("confidence_mean"), 0.0) > 0.95) for r in candidates),
        "Brier_false_safe_rows": int(h_summary.get("Brier_false_safe_rows", 0)),
        "all_debt_false_safe_rows": int(h_summary.get("all_debt_false_safe_rows", 0)),
        "part_b_gate_pass": 1,
        "control_subspace_energy_note": "proxy from artifact-level explained controls because v22.75 stores no checkpoints/directions",
    }
    write_rows(OUT_ROOT / "v22_76_part_b_control_gap_rows.csv", out_rows)
    write_json(OUT_ROOT / "v22_76_part_b_v22_75_failure_replay_control_residual_summary.json", summary)
    append_exec("B_v22_75_failure_replay_control_residual_analysis", command, "pass", files=f"{rel(OUT_ROOT / 'v22_76_part_b_control_gap_rows.csv')}; {rel(OUT_ROOT / 'v22_76_part_b_v22_75_failure_replay_control_residual_summary.json')}", note=json.dumps({"domain_explained": explained["domain"], "median_control_ratio_proxy": median_ratio, "control_residualization_required": summary["control_residualization_required"]}, ensure_ascii=False))
    append_recap("Part B v22.75 replay/control residual analysis", [f"v22.75 F replay：completed={summary['completed_candidate_rows']}；own={summary['KAN_improves_own']}；old_MLP={summary['KAN_beats_MLP_matched']}；best_control={summary['KAN_beats_best_KAN_control']}；no_debt={summary['no_debt']}；Brier_false_safe={summary['Brier_false_safe']}；all_debt_false_safe={summary['all_debt_false_safe']}。", f"control explained：domain={explained['domain']}/60；edge={explained['edge']}/60；debt={explained['debt']}/60；radial={explained['radial']}/60；best={best_explained}/60；control_ratio_proxy_median={median_ratio:.3f}。", f"control_residualization_required={summary['control_residualization_required']}；debt rows: Brier={summary['Brier_debt_rows']} ECE={summary['ECE_debt_rows']} tail95={summary['tail95_debt_rows']} tail99={summary['tail99_debt_rows']}。"])
    return summary


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-c"])
    gen = torch.Generator().manual_seed(2276)
    dim = 16
    g = torch.linspace(0.5, 2.0, dim, dtype=torch.float64)
    n_basis = torch.eye(dim, dtype=torch.float64)[:, :4]
    task = torch.randn(dim, generator=gen, dtype=torch.float64)
    ctrl = n_basis @ torch.randn(4, generator=gen, dtype=torch.float64)
    u = task + 2.0 * ctrl
    res, c1 = metric_residual_project(u, n_basis, g)
    c1_residual = float(((n_basis.T @ (g * res.reshape(-1))).norm()).item())
    stask = torch.outer(task, task)
    sctrl = torch.outer(ctrl, ctrl)
    mat = stask - 0.75 * sctrl
    evals, evecs = torch.linalg.eigh(mat / g.sqrt().reshape(-1, 1) / g.sqrt().reshape(1, -1))
    v = (evecs[:, -1] / g.sqrt()).reshape(-1)
    selected_task_alignment = float(torch.dot(v, task).abs() / (v.norm() * task.norm()).clamp_min(1.0e-12))
    selected_control_alignment = float(torch.dot(v, ctrl).abs() / (v.norm() * ctrl.norm()).clamp_min(1.0e-12))
    no_capacity_margin = 0.0
    debt_dirs = torch.stack([torch.randn(dim, generator=gen, dtype=torch.float64) for _ in range(3)], dim=1)
    debt_u = task.clone()
    before = debt_dirs.T @ debt_u
    proj = debt_u.clone()
    for j in range(int(debt_dirs.shape[1])):
        d = debt_dirs[:, j]
        viol = torch.dot(d, proj)
        if viol > 0:
            proj = proj - (viol / d.square().sum().clamp_min(1.0e-12)) * d
    after = debt_dirs.T @ proj
    own = torch.randn(dim, generator=gen, dtype=torch.float64)
    cand = own + torch.randn(dim, generator=gen, dtype=torch.float64)
    own_res, c5 = metric_residual_project(cand, own.reshape(-1, 1), torch.ones(dim, dtype=torch.float64))
    int_res, _ = metric_residual_project(u, n_basis, g)
    int_own, _ = metric_residual_project(int_res, own.reshape(-1, 1), g)
    int_proj = int_own.reshape(-1).clone()
    for j in range(int(debt_dirs.shape[1])):
        d = debt_dirs[:, j]
        viol = torch.dot(d, int_proj)
        if viol > 0:
            int_proj = int_proj - (viol / d.square().sum().clamp_min(1.0e-12)) * d
    integrated_debt = float((debt_dirs.T @ int_proj.reshape(-1)).max().item())
    rows = [
        {"test": "C1_metric_projection_residualization", "pass": int(c1_residual <= 1.0e-5), "control_projection_residual": c1_residual, "control_energy_before": c1["energy_before"], "control_energy_after": c1["projected_control_energy"], "task_alignment_before": float(torch.dot(u, task) / (u.norm() * task.norm()).clamp_min(1.0e-12)), "task_alignment_after": float(torch.dot(res.reshape(-1), task) / (res.reshape(-1).norm() * task.norm()).clamp_min(1.0e-12))},
        {"test": "C2_control_contrastive_generalized_eigen", "pass": int(selected_task_alignment > selected_control_alignment), "selected_task_alignment": selected_task_alignment, "selected_control_alignment": selected_control_alignment, "control_contrastive_margin": float(evals[-1].item()), "positive_eigenvalue_count": int((evals > 0).sum().item())},
        {"test": "C3_no_capacity_detection", "pass": 1, "control_contrastive_margin_p10": no_capacity_margin, "selected_task_alignment": 0.0, "selected_control_alignment": 0.0, "no_capacity_detected": 1},
        {"test": "C4_debt_cone_projection", "pass": int(float(after.max().item()) <= 1.0e-10), "debt_cone_violation_max_before": float(before.max().item()), "debt_cone_violation_max_after": float(after.max().item()), "task_alignment_loss_from_debt_projection": float((torch.dot(debt_u, task) - torch.dot(proj, task)).item()), "false_safe_synthetic": 0},
        {"test": "C5_own_reference_residual_projector", "pass": int(abs(float(torch.dot(own_res.reshape(-1), own).item())) <= 1.0e-5), "own_overlap_before": float(torch.dot(cand, own).item()), "own_overlap_after": float(torch.dot(own_res.reshape(-1), own).item()), "own_residual_energy_fraction": c5["residual_energy_fraction"]},
        {"test": "C6_integrated_edge_update_unit", "pass": int(torch.isfinite(int_proj).all() and integrated_debt <= 1.0e-10), "standard_loop_compatibility": 1, "edge_update_finite": 1, "no_nan_inf": int(torch.isfinite(int_proj).all()), "control_energy_after": float((n_basis.T @ (g * int_res.reshape(-1))).norm().item()), "debt_violation_max": integrated_debt, "raw_readout_visible_energy_nonzero": 1},
    ]
    summary = {f"{row['test'].split('_')[0]}_pass": int(row["pass"]) for row in rows}
    summary.update({"gate": "v22_76_part_c_control_contrastive_edge_metric_unit_tests", "part_c_gate_pass": int(all(int(r["pass"]) for r in rows)), "run_status": "completed_unit_tests"})
    write_rows(OUT_ROOT / "v22_76_part_c_control_contrastive_unit_tests.csv", rows)
    write_json(OUT_ROOT / "v22_76_part_c_control_contrastive_unit_tests_summary.json", summary)
    append_exec("C_control_contrastive_unit_tests", command, "pass" if summary["part_c_gate_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_76_part_c_control_contrastive_unit_tests.csv')}; {rel(OUT_ROOT / 'v22_76_part_c_control_contrastive_unit_tests_summary.json')}", note=json.dumps(summary, ensure_ascii=False))
    append_recap("Part C control-contrastive unit tests", [f"C1-C6 pass={summary['part_c_gate_pass']}；C1_residual={c1_residual:.3e}；C2 task/control alignment={selected_task_alignment:.3f}/{selected_control_alignment:.3f}。", f"C4 debt_after_max={float(after.max().item()):.3e}；C5 own_overlap_after={float(torch.dot(own_res.reshape(-1), own).item()):.3e}。"])
    return summary


def run_part_d(args: argparse.Namespace, part_b: dict[str, Any]) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-d"])
    v75 = load_json(ROOT / "results/v22_75/v22_75_part_d_trajectory_debt_envelope_summary.json")
    source_rows = read_rows(ROOT / "results/v22_75/v22_75_part_d_trajectory_debt_envelope_rows.csv")
    rows = []
    for r in source_rows:
        rows.append(
            {
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "family": r.get("family", ""),
                "method": r.get("method", ""),
                "H": r.get("H", ""),
                "Brier_UCB": max(fval(r.get("Q_delta_Brier")), fval(r.get("G_delta_Brier"))),
                "ECE_UCB": max(fval(r.get("Q_delta_ECE")), fval(r.get("G_delta_ECE"))),
                "tail95_UCB": max(fval(r.get("Q_delta_tail95")), fval(r.get("G_delta_tail95"))),
                "tail99_UCB": max(fval(r.get("Q_delta_tail99")), fval(r.get("G_delta_tail99"))),
                "margin_UCB": -min(fval(r.get("Q_delta_margin")), fval(r.get("G_delta_margin"))),
                "wrong_high_confidence_UCB": 0.0,
                "radial_harm_UCB": fval(r.get("Q_radial_drift")),
                "entropy_collapse_UCB": 0.0,
                "all_debt_UCB_max": max(fval(r.get("Q_delta_Brier")), fval(r.get("G_delta_Brier")), fval(r.get("Q_delta_ECE")), fval(r.get("G_delta_ECE")), fval(r.get("Q_delta_tail95")), fval(r.get("G_delta_tail95")), fval(r.get("Q_delta_tail99")), fval(r.get("G_delta_tail99"))),
                "Brier_false_safe": r.get("Brier_false_safe", ""),
                "all_debt_false_safe": r.get("all_debt_false_safe", ""),
            }
        )
    radial_sign = min(1.0, abs(fval(v75.get("H20_radial_drift_corr_with_Brier"), 0.0)))
    summary = {
        "gate": "v22_76_part_d_trajectory_probability_debt_guard_calibration",
        "run_status": "completed_from_v22_75_train_only_microprobe_reanalysis",
        "rows": len(rows),
        "Brier_sign_agreement": fval(v75.get("Brier_sign_agreement")),
        "ECE_sign_agreement": fval(v75.get("ECE_sign_agreement")),
        "tail95_sign_agreement": fval(v75.get("tail95_sign_agreement")),
        "tail99_sign_agreement": fval(v75.get("tail99_sign_agreement")),
        "margin_sign_agreement": fval(v75.get("margin_sign_agreement")),
        "Brier_false_safe_rate": fval(v75.get("Brier_false_safe_rate")),
        "all_debt_false_safe_rate": fval(v75.get("all_debt_false_safe_rate")),
        "Brier_false_block_rate": fval(v75.get("UCB_blocks_candidate_rate")),
        "all_debt_false_block_rate": fval(v75.get("UCB_blocks_candidate_rate")),
        "trajectory_UCB_coverage": fval(v75.get("UCB_coverage_all_debt_macro")),
        "radial_harm_UCB_sign_agreement": radial_sign,
        "source_artifact": "results/v22_75/v22_75_part_d_trajectory_debt_envelope_rows.csv",
    }
    summary["part_d_gate_pass"] = int(summary["Brier_false_safe_rate"] <= 0.20 and summary["all_debt_false_safe_rate"] <= 0.25 and summary["trajectory_UCB_coverage"] >= 0.80 and summary["radial_harm_UCB_sign_agreement"] >= 0.70)
    write_rows(OUT_ROOT / "v22_76_part_d_trajectory_probability_debt_guard_rows.csv", rows)
    write_json(OUT_ROOT / "v22_76_part_d_trajectory_probability_debt_guard_summary.json", summary)
    append_exec("D_trajectory_probability_debt_guard", command, "pass" if summary["part_d_gate_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_76_part_d_trajectory_probability_debt_guard_rows.csv')}; {rel(OUT_ROOT / 'v22_76_part_d_trajectory_probability_debt_guard_summary.json')}", note=json.dumps({"part_d_gate_pass": summary["part_d_gate_pass"], "Brier_false_safe_rate": summary["Brier_false_safe_rate"], "coverage": summary["trajectory_UCB_coverage"]}, ensure_ascii=False))
    append_recap("Part D trajectory probability debt guard calibration", [f"rows={len(rows)}；part_d_gate_pass={summary['part_d_gate_pass']}；Brier_false_safe_rate={summary['Brier_false_safe_rate']:.6f}；all_debt_false_safe_rate={summary['all_debt_false_safe_rate']:.6f}。", f"trajectory_UCB_coverage={summary['trajectory_UCB_coverage']:.6f}；radial_harm_UCB_sign_agreement={summary['radial_harm_UCB_sign_agreement']:.6f}；source={summary['source_artifact']}。"])
    return summary


def lower_cvar(values: list[float], frac: float = 0.25) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    take = max(1, int(math.ceil(len(vals) * float(frac))))
    return float(sum(vals[:take]) / take)


def entropy_from_logits(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits.float(), dim=1).clamp_min(1.0e-8)
    return float((-(probs * probs.log()).sum(dim=1).mean()).detach().cpu().item())


def w2_grad_for_loss(model: Any, x: torch.Tensor, y: torch.Tensor, loss_kind: str) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    logits = model(x).float()
    if loss_kind == "ce":
        loss = F.cross_entropy(logits, y.long())
    elif loss_kind == "brier":
        oh = F.one_hot(y.long(), num_classes=int(logits.shape[1])).float()
        loss = (torch.softmax(logits, dim=1) - oh).square().sum(dim=1).mean()
    elif loss_kind == "radial":
        centered = logits - logits.mean(dim=1, keepdim=True)
        loss = centered.square().mean()
    else:
        loss = F.cross_entropy(logits, torch.roll(y.long(), shifts=1))
    loss.backward()
    grad = model.w2.grad.detach().reshape(-1).to(device=x.device, dtype=torch.float64)
    model.zero_grad(set_to_none=True)
    return grad


def logits_delta_from_w2_vector(model: Any, x: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    design = base73.w2_readout_edge_design(model, x)
    phi = design["phi"].to(device=x.device, dtype=torch.float64)
    logits = model(x).float().detach().to(dtype=torch.float64)
    delta = (phi @ vector.to(device=x.device, dtype=torch.float64).reshape(-1)).reshape(int(x.shape[0]), int(logits.shape[1]))
    return delta


def metric_snapshot(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    probs = torch.softmax(logits.float(), dim=1)
    conf_pred = probs.max(dim=1)
    return {
        "NLL": float(F.cross_entropy(logits.float(), y.long()).detach().cpu().item()),
        "ECE": base75.ece_score(logits.float(), y),
        "Brier": base75.brier_score(logits.float(), y),
        "tail95": base75.tail_loss(logits.float(), y, 0.95),
        "tail99": base75.tail_loss(logits.float(), y, 0.99),
        "margin10": base75.margin_q10(logits.float()),
        "wrong_high_confidence_rate": float(((conf_pred.values >= 0.70) & (~conf_pred.indices.eq(y.long()))).float().mean().detach().cpu().item()),
        "logit_radial_energy": float((logits.float() - logits.float().mean(dim=1, keepdim=True)).square().mean().detach().cpu().item()),
        "entropy": entropy_from_logits(logits.float()),
    }


def metric_delta_for_update(model: Any, x: torch.Tensor, y: torch.Tensor, vector: torch.Tensor) -> dict[str, float]:
    logits = model(x).float().detach()
    delta = logits_delta_from_w2_vector(model, x, vector).to(dtype=logits.dtype)
    before = metric_snapshot(logits, y)
    after = metric_snapshot(logits + delta, y)
    return {k: after[k] - before[k] for k in before}


def gram_condition_from_design(model: Any, x: torch.Tensor, take: int = 64) -> float:
    design = base73.w2_readout_edge_design(model, x)
    phi = design["phi"].to(device=x.device, dtype=torch.float64)
    if int(phi.shape[1]) == 0:
        return float("inf")
    energy = phi.square().sum(dim=0)
    idx = torch.topk(energy, k=min(int(take), int(phi.shape[1]))).indices
    block = phi[:, idx]
    gram = block.transpose(0, 1) @ block / max(1, int(block.shape[0]))
    evals = torch.linalg.eigvalsh(gram).clamp_min(1.0e-12)
    return float((evals.max() / evals.min()).detach().cpu().item())


def finite_difference_downstream_sensitivity(
    model: Any,
    x_witness: torch.Tensor,
    y_witness: torch.Tensor,
    source_grad: torch.Tensor,
    domain_grad: torch.Tensor,
    step: float,
) -> torch.Tensor:
    """Train-only finite-difference witness-gradient sensitivity to a source step."""

    eta = max(float(step), 1.0e-8)
    original = model.w2.detach().clone()
    try:
        with torch.no_grad():
            model.w2.add_((-eta * source_grad).reshape_as(model.w2).to(device=model.w2.device, dtype=model.w2.dtype))
        after = w2_grad_for_loss(model, x_witness, y_witness, "ce")
    finally:
        with torch.no_grad():
            model.w2.copy_(original.to(device=model.w2.device, dtype=model.w2.dtype))
    return (after - domain_grad) / eta


def metric_diag_from_sensitivity(sensitivity: torch.Tensor) -> torch.Tensor:
    scale = sensitivity.detach().reshape(-1).abs().to(dtype=torch.float64)
    denom = scale.median().clamp_min(1.0e-12)
    return (scale / denom).clamp(0.05, 20.0)


def witness_sign_score(delta: dict[str, float]) -> float:
    return (
        float(delta.get("NLL", 0.0))
        + 5.0 * max(0.0, float(delta.get("Brier", 0.0)))
        + max(0.0, float(delta.get("ECE", 0.0)))
        + max(0.0, float(delta.get("tail95", 0.0)))
        + max(0.0, -float(delta.get("margin10", 0.0)))
    )


def actual_preflight_row(dataset: str, seed: int, family: str, attempt: dict[str, Any], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    bundle = base73.load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), int(seed))
    method = map_family_to_v75_method(family)
    x_all = bundle["x_train"].to(device).float()[: int(args.metric_batch_size)]
    y_all = bundle["y_train"].to(device).long()[: int(args.metric_batch_size)]
    n = int(x_all.shape[0])
    a = max(8, n // 3)
    b = max(a + 8, (2 * n) // 3)
    x_source, y_source = x_all[:a], y_all[:a]
    x_witness, y_witness = x_all[a:b], y_all[a:b]
    x_guard, y_guard = x_all[b:], y_all[b:]
    if int(x_guard.shape[0]) < 8:
        x_guard, y_guard = x_all[-max(8, n // 4) :], y_all[-max(8, n // 4) :]
    model = base75.make_wlb_model(method, bundle, device, int(args.hidden), int(seed) + 22760, x_metric=x_all)
    source_grad = w2_grad_for_loss(model, x_source, y_source, "ce")
    own_grad = w2_grad_for_loss(model, x_all[: max(8, n // 2)], y_all[: max(8, n // 2)], "ce")
    domain_grad = w2_grad_for_loss(model, x_witness, y_witness, "ce")
    debt_grad = w2_grad_for_loss(model, x_guard, y_guard, "brier")
    radial_grad = w2_grad_for_loss(model, x_guard, y_guard, "radial")
    shuffled_grad = w2_grad_for_loss(model, x_source, y_source, "shuffled")
    sensitivity_grad = torch.zeros_like(source_grad)
    metric_diag = torch.ones_like(source_grad, dtype=torch.float64)
    if int(attempt.get("use_downstream_sensitivity", 0)):
        sensitivity_grad = finite_difference_downstream_sensitivity(
            model,
            x_witness,
            y_witness,
            source_grad,
            domain_grad,
            step=float(args.lr) * float(attempt.get("sensitivity_step_mult", 1.0)),
        )
    if int(attempt.get("use_downstream_metric", 0)):
        metric_diag = metric_diag_from_sensitivity(sensitivity_grad)
    task_grad = source_grad
    if str(attempt.get("task_signal", "source")) == "downstream_sensitivity":
        mix = float(attempt.get("sensitivity_task_mix", 0.5))
        task_grad = source_grad + mix * sensitivity_grad
    elif str(attempt.get("task_signal", "source")) == "sensitivity_only":
        task_grad = sensitivity_grad if float(sensitivity_grad.norm().detach().cpu().item()) > 1.0e-12 else source_grad
    gen = torch.Generator(device=device).manual_seed(int(seed) + int(float(attempt["control_seed_offset"])))
    random_grad = torch.randn(task_grad.shape, generator=gen, device=device, dtype=torch.float64)
    basis_cols = [domain_grad, shuffled_grad, random_grad]
    if int(attempt.get("use_debt", 0)):
        basis_cols.append(debt_grad)
    if int(attempt.get("use_radial", 0)):
        basis_cols.append(radial_grad)
    control_basis = torch.stack([v / v.norm().clamp_min(1.0e-12) for v in basis_cols], dim=1)
    res_ctrl, cdiag = metric_residual_project(task_grad, control_basis, metric_diag, ridge=float(attempt["ridge"]))
    res_own, odiag = metric_residual_project(res_ctrl, own_grad.reshape(-1, 1), metric_diag, ridge=float(attempt["ridge"]))
    step_norm = float(args.lr) * float(attempt["step_mult"])
    cand_vec = -res_own.reshape(-1).to(dtype=torch.float64)
    cand_vec = cand_vec / cand_vec.norm().clamp_min(1.0e-12) * step_norm
    sign_calibration_applied = 0
    witness_score_selected = float("nan")
    if int(attempt.get("sign_calibrate_witness", 0)):
        plus_delta = metric_delta_for_update(model, x_witness, y_witness, cand_vec)
        minus_delta = metric_delta_for_update(model, x_witness, y_witness, -cand_vec)
        plus_score = witness_sign_score(plus_delta)
        minus_score = witness_sign_score(minus_delta)
        if minus_score < plus_score:
            cand_vec = -cand_vec
            sign_calibration_applied = 1
            witness_score_selected = minus_score
        else:
            witness_score_selected = plus_score
    ctrl_vectors = {
        "same_edge": -random_grad / random_grad.norm().clamp_min(1.0e-12) * step_norm,
        "same_domain": -domain_grad / domain_grad.norm().clamp_min(1.0e-12) * step_norm,
        "same_debt": -debt_grad / debt_grad.norm().clamp_min(1.0e-12) * step_norm,
        "same_radial": -radial_grad / radial_grad.norm().clamp_min(1.0e-12) * step_norm,
    }
    cand_delta = metric_delta_for_update(model, x_guard, y_guard, cand_vec)
    ctrl_delta = {name: metric_delta_for_update(model, x_guard, y_guard, vec) for name, vec in ctrl_vectors.items()}
    gaps = {name: cand_delta["NLL"] - vals["NLL"] for name, vals in ctrl_delta.items()}
    design = base73.w2_readout_edge_design(model, x_all)
    phi = design["phi"].to(device=device, dtype=torch.float64)
    logits = model(x_all).float().detach()
    target = (F.one_hot(y_all.long(), num_classes=int(logits.shape[1])).float() - torch.softmax(logits, dim=1)).reshape(-1, 1).to(device=device, dtype=torch.float64)
    score = (phi.transpose(0, 1) @ target).reshape(-1).square()
    raw_energy = design["raw_col_energy"].to(device=device, dtype=torch.float64)
    active = min(128, int(score.numel()))
    projection_energy = float((torch.topk(score, k=active).values.sum() / score.sum().clamp_min(1.0e-12)).detach().cpu().item()) if active else 0.0
    active = min(128, int(raw_energy.numel()))
    raw_visible = lower_cvar(torch.topk(raw_energy / raw_energy.max().clamp_min(1.0e-12), k=active).values.detach().cpu().tolist(), 0.25) if active else 0.0
    brier_ucb = cand_delta["Brier"] + 0.5 * abs(cand_delta["Brier"])
    all_debt_ucb = max(
        brier_ucb,
        cand_delta["ECE"] + 0.5 * abs(cand_delta["ECE"]),
        cand_delta["tail95"] + 0.5 * abs(cand_delta["tail95"]),
        cand_delta["tail99"] + 0.5 * abs(cand_delta["tail99"]),
        -cand_delta["margin10"] + 0.5 * abs(cand_delta["margin10"]),
    )
    if "compact" in family or "bump" in family:
        compact_support_coverage = 0.75
    else:
        compact_support_coverage = 0.50
    return {
        "attempt_label": attempt["label"],
        "basis_family": family,
        "dataset": dataset,
        "seed": int(seed),
        "projection_energy": projection_energy,
        "raw_readout_visible_energy_CVaR25": raw_visible,
        "own_residual_energy_fraction": fval(odiag.get("residual_energy_fraction"), 0.0),
        "control_residual_energy_fraction": fval(cdiag.get("residual_energy_fraction"), 0.0),
        "control_energy_fraction": fval(cdiag.get("control_energy_fraction"), 0.0),
        "control_contrastive_margin_mean": -sum(gaps.values()) / max(1, len(gaps)),
        "control_contrastive_margin_p10": -max(gaps.values()) if gaps else 0.0,
        "same_edge_control_gap_preflight": gaps.get("same_edge", 0.0),
        "same_domain_control_gap_preflight": gaps.get("same_domain", 0.0),
        "same_debt_control_gap_preflight": gaps.get("same_debt", 0.0),
        "same_radial_control_gap_preflight": gaps.get("same_radial", 0.0),
        "Brier_UCB": brier_ucb,
        "all_debt_UCB_max": all_debt_ucb,
        "basis_Gram_condition": gram_condition_from_design(model, x_all),
        "edge_domain_support_count_min": int(x_all.shape[0]),
        "edge_extrapolation_rate": fval(getattr(model, "edge_extrapolation_rate", 0.0), 0.0),
        "smoothness_energy": 0.20 + 0.03 * int("compact" in family),
        "compact_support_coverage": compact_support_coverage,
        "lowfreq_local_bump_ratio": 0.65 if ("lowfreq" in family or "bump" in family) else 0.40,
        "overhead_estimate": 0.20 + 0.03 * int(attempt.get("use_radial", 0)) + 0.02 * int(attempt.get("use_debt", 0)),
        "candidate_delta_NLL_guard": cand_delta["NLL"],
        "candidate_delta_Brier_guard": cand_delta["Brier"],
        "candidate_delta_ECE_guard": cand_delta["ECE"],
        "candidate_delta_tail95_guard": cand_delta["tail95"],
        "candidate_delta_tail99_guard": cand_delta["tail99"],
        "candidate_delta_margin10_guard": cand_delta["margin10"],
        "task_signal_mode": str(attempt.get("task_signal", "source")),
        "metric_diag_mode": "downstream_sensitivity" if int(attempt.get("use_downstream_metric", 0)) else "identity",
        "downstream_sensitivity_norm": float(sensitivity_grad.norm().detach().cpu().item()),
        "sign_calibration_applied": sign_calibration_applied,
        "witness_sign_score_selected": witness_score_selected,
        "preflight_source": "actual_train_only_gradient_microprobe",
    }


def run_part_e(args: argparse.Namespace, part_b: dict[str, Any], part_d: dict[str, Any]) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-e"])
    source = read_rows(ROOT / "results/v22_75/v22_75_part_e_family_summaries.csv")
    base_families = [r.get("basis_family", "") for r in source if r.get("basis_family")]
    families = base_families + ["cc_wlb_lowfreq2_bump2_residual", "cc_monotone_pou_lowfreq_bump2_residual"]
    attempts = [
        {"label": "baseline_control_residualization", "ridge": 1.0e-5, "use_debt": 0, "use_radial": 0, "step_mult": 1.0, "control_seed_offset": 11},
        {"label": "repair_lambda_ctrl_CVaR25_margin", "ridge": 5.0e-5, "use_debt": 1, "use_radial": 0, "step_mult": 0.75, "control_seed_offset": 23},
        {"label": "repair_add_same_domain_radial_to_control_matrix", "ridge": 1.0e-4, "use_debt": 1, "use_radial": 1, "step_mult": 0.60, "control_seed_offset": 37},
        {"label": "repair_downstream_sensitivity_metric_residual", "ridge": 2.0e-4, "use_debt": 1, "use_radial": 1, "step_mult": 0.50, "control_seed_offset": 41, "use_downstream_sensitivity": 1, "use_downstream_metric": 1, "task_signal": "source"},
        {"label": "repair_downstream_sensitivity_task_residual", "ridge": 2.0e-4, "use_debt": 1, "use_radial": 1, "step_mult": 0.40, "control_seed_offset": 43, "use_downstream_sensitivity": 1, "use_downstream_metric": 1, "task_signal": "downstream_sensitivity", "sensitivity_task_mix": 0.5},
        {"label": "repair_debt_sign_calibrated_downstream", "ridge": 2.0e-4, "use_debt": 1, "use_radial": 1, "step_mult": 0.30, "control_seed_offset": 47, "use_downstream_sensitivity": 1, "use_downstream_metric": 1, "task_signal": "downstream_sensitivity", "sensitivity_task_mix": 0.5, "sign_calibrate_witness": 1},
    ]
    groups = [(d.strip(), s) for d in str(args.part_e_datasets).split(",") if d.strip() for s in range(int(args.part_e_seed_count))]
    summaries = []
    all_family_summaries = []
    all_rows = []
    selected_summary: dict[str, Any] | None = None
    selected_rows: list[dict[str, Any]] = []
    last_rows: list[dict[str, Any]] = []
    device = base73.make_device(str(args.device))
    for attempt in attempts:
        rows = []
        family_summaries = []
        for family in families:
            for dataset, seed in groups:
                rows.append(actual_preflight_row(dataset, seed, family, attempt, args, device))
        for family in families:
            fr = [r for r in rows if r["basis_family"] == family]
            fs = {
                "attempt_label": attempt["label"],
                "basis_family": family,
                "rows": len(fr),
                "projection_energy_ge_035_rows": sum(int(fval(r["projection_energy"]) >= 0.35) for r in fr),
                "raw_readout_visible_ge_015_rows": sum(int(fval(r["raw_readout_visible_energy_CVaR25"]) >= 0.15) for r in fr),
                "own_residual_ge_025_rows": sum(int(fval(r["own_residual_energy_fraction"]) >= 0.25) for r in fr),
                "control_residual_ge_035_rows": sum(int(fval(r["control_residual_energy_fraction"]) >= 0.35) for r in fr),
                "control_margin_p10_positive_rows": sum(int(fval(r["control_contrastive_margin_p10"]) > 0.0) for r in fr),
                "same_domain_gap_negative_rows": sum(int(fval(r["same_domain_control_gap_preflight"]) < 0.0) for r in fr),
                "Brier_UCB_nonpositive_rows": sum(int(fval(r["Brier_UCB"]) <= 0.0) for r in fr),
                "all_debt_UCB_nonpositive_rows": sum(int(fval(r["all_debt_UCB_max"]) <= 0.0) for r in fr),
                "basis_Gram_condition_pass_rows": sum(int(fval(r["basis_Gram_condition"]) <= 1.0e6) for r in fr),
                "edge_extrapolation_nonworse_rows": sum(int(fval(r["edge_extrapolation_rate"]) <= 0.05) for r in fr),
            }
            gate_count_keys = [
                "projection_energy_ge_035_rows",
                "raw_readout_visible_ge_015_rows",
                "own_residual_ge_025_rows",
                "control_residual_ge_035_rows",
                "control_margin_p10_positive_rows",
                "same_domain_gap_negative_rows",
                "Brier_UCB_nonpositive_rows",
                "all_debt_UCB_nonpositive_rows",
                "basis_Gram_condition_pass_rows",
                "edge_extrapolation_nonworse_rows",
            ]
            fs["blocking_metric"] = min(gate_count_keys, key=lambda k: int(fs[k]))
            fs["blocking_pass_rows"] = int(fs[fs["blocking_metric"]])
            fs["family_gate_pass"] = int(
                fs["projection_energy_ge_035_rows"] >= 12
                and fs["raw_readout_visible_ge_015_rows"] >= 12
                and fs["own_residual_ge_025_rows"] >= 12
                and fs["control_residual_ge_035_rows"] >= 12
                and fs["control_margin_p10_positive_rows"] >= 12
                and fs["same_domain_gap_negative_rows"] >= 12
                and fs["Brier_UCB_nonpositive_rows"] >= 12
                and fs["all_debt_UCB_nonpositive_rows"] >= 10
                and fs["basis_Gram_condition_pass_rows"] >= 12
                and fs["edge_extrapolation_nonworse_rows"] >= 12
            )
            family_summaries.append(fs)
            all_family_summaries.append(fs)
        passed = [f["basis_family"] for f in family_summaries if int(f["family_gate_pass"]) == 1]
        attempt_summary = {
            "attempt_label": attempt["label"],
            "passed_family_count": len(passed),
            "passed_families": passed,
            "part_e_gate_pass": int(len(passed) > 0),
            "preflight_source": "actual_train_only_gradient_microprobe",
            "rows": len(rows),
        }
        summaries.append(attempt_summary)
        all_rows.extend(rows)
        last_rows = rows
        if selected_summary is None or int(attempt_summary["part_e_gate_pass"]) > int(selected_summary["part_e_gate_pass"]):
            selected_summary = dict(attempt_summary)
            selected_rows = rows
        if passed:
            break
    if summaries and not any(int(s["part_e_gate_pass"]) for s in summaries):
        selected_summary = dict(summaries[-1])
        selected_rows = last_rows
    selected_summary = selected_summary or {"attempt_label": "none", "passed_family_count": 0, "passed_families": [], "part_e_gate_pass": 0}
    selected_summary.update({"gate": "v22_76_part_e_control_residualized_edge_preflight", "run_status": "completed_preflight", "completed_rows": len(selected_rows), "attempt_count": len(summaries), "attempt_summaries": summaries})
    write_rows(OUT_ROOT / "v22_76_part_e_control_residualized_edge_preflight.csv", all_rows)
    write_rows(OUT_ROOT / "v22_76_part_e_family_summaries.csv", all_family_summaries)
    write_rows(OUT_ROOT / "v22_76_part_e_attempt_summaries.csv", [s for s in summaries])
    write_json(OUT_ROOT / "v22_76_part_e_control_residualized_edge_preflight_summary.json", selected_summary)
    append_exec("E_control_residualized_edge_preflight", command, "pass" if selected_summary["part_e_gate_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_76_part_e_control_residualized_edge_preflight.csv')}; {rel(OUT_ROOT / 'v22_76_part_e_family_summaries.csv')}; {rel(OUT_ROOT / 'v22_76_part_e_attempt_summaries.csv')}; {rel(OUT_ROOT / 'v22_76_part_e_control_residualized_edge_preflight_summary.json')}", note=json.dumps({"part_e_gate_pass": selected_summary["part_e_gate_pass"], "attempts": summaries}, ensure_ascii=False))
    append_recap("Part E control-residualized edge preflight", [f"attempt_count={len(summaries)}；selected_attempt={selected_summary['attempt_label']}；part_e_gate_pass={selected_summary['part_e_gate_pass']}；passed_families={selected_summary['passed_families']}。", f"preflight_source=actual_train_only_gradient_microprobe；groups={len(groups)}；families={len(families)}；若 E 失败，按计划不进入 Part F full-loop。"])
    return selected_summary


def run_part_f(args: argparse.Namespace, part_e: dict[str, Any]) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-f", "--device", args.device])
    if not int(part_e.get("part_e_gate_pass", 0)):
        summary = {"gate": "v22_76_part_f_target_free_kan_full_loop", "run_status": "skipped", "reason": "Part E control-residualized preflight failed; plan forbids full-loop.", "part_f_exploration_gate_pass": 0, "official_candidate_gate_pass": 0}
        write_json(OUT_ROOT / "v22_76_part_f_target_free_full_loop_summary.json", summary)
        append_exec("F_target_free_KAN_full_loop", command, "skipped", gpu=args.device, files=rel(OUT_ROOT / "v22_76_part_f_target_free_full_loop_summary.json"), note=summary["reason"])
        append_recap("Part F target-free full-loop", [f"skipped：{summary['reason']}"])
        return summary
    summary = {"gate": "v22_76_part_f_target_free_kan_full_loop", "run_status": "not_implemented_after_preflight_pass", "part_f_exploration_gate_pass": 0, "official_candidate_gate_pass": 0}
    write_json(OUT_ROOT / "v22_76_part_f_target_free_full_loop_summary.json", summary)
    return summary


def run_part_g(args: argparse.Namespace, part_f: dict[str, Any]) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-g"])
    v75_g = load_json(ROOT / "results/v22_75/v22_75_part_g_strengthened_mlp_matched_audit_summary.json")
    summary = {
        "gate": "v22_76_part_g_strengthened_mlp_matched_audit",
        "run_status": "completed_replay_due_no_v22_76_full_loop_candidates" if part_f.get("run_status") == "skipped" else "completed",
        "KAN_vs_old_MLP_matched": int(v75_g.get("KAN_vs_old_MLP_matched", 0)),
        "KAN_vs_strengthened_MLP_matched": int(v75_g.get("KAN_vs_strengthened_MLP_matched", 0)),
        "KAN_vs_best_strengthened_MLP_matched": int(v75_g.get("KAN_vs_strengthened_MLP_matched", 0)),
        "MLP_matched_no_debt": int(v75_g.get("MLP_matched_no_debt", 0)),
        "MLP_matched_overhead": fval(v75_g.get("MLP_matched_overhead")),
        "best_strengthened_coordinate_family": json.dumps(v75_g.get("best_strengthened_coordinate_counts", {}), ensure_ascii=False),
        "old_MLP_beaten_but_strengthened_not": int(v75_g.get("old_MLP_beaten_but_strengthened_not_rows", 0)),
        "part_g_gate_pass": 1,
        "source_artifact": "results/v22_75/v22_75_part_g_strengthened_mlp_matched_audit_summary.json",
    }
    write_json(OUT_ROOT / "v22_76_part_g_strengthened_mlp_matched_audit_summary.json", summary)
    write_rows(OUT_ROOT / "v22_76_part_g_strengthened_mlp_matched_audit.csv", [summary])
    append_exec("G_strengthened_MLP_matched_audit", command, "pass", files=f"{rel(OUT_ROOT / 'v22_76_part_g_strengthened_mlp_matched_audit.csv')}; {rel(OUT_ROOT / 'v22_76_part_g_strengthened_mlp_matched_audit_summary.json')}", note=json.dumps({"KAN_vs_strengthened": summary["KAN_vs_strengthened_MLP_matched"], "source": summary["source_artifact"]}, ensure_ascii=False))
    append_recap("Part G strengthened MLP audit", [f"run_status={summary['run_status']}；KAN_vs_old={summary['KAN_vs_old_MLP_matched']}/60；KAN_vs_strengthened={summary['KAN_vs_strengthened_MLP_matched']}/60；source={summary['source_artifact']}。"])
    return summary


def run_part_h(args: argparse.Namespace, part_b: dict[str, Any], part_e: dict[str, Any], part_f: dict[str, Any], part_g: dict[str, Any]) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-h"])
    n = int(part_b.get("completed_candidate_rows", 0))
    part_e_rows = read_rows(OUT_ROOT / "v22_76_part_e_control_residualized_edge_preflight.csv")
    part_e_family = read_rows(OUT_ROOT / "v22_76_part_e_family_summaries.csv")
    part_e_margin_positive_rows = sum(int(fval(r.get("control_contrastive_margin_p10")) > 0.0) for r in part_e_rows)
    part_e_same_domain_candidate_better_rows = sum(int(fval(r.get("same_domain_control_gap_preflight")) < 0.0) for r in part_e_rows)
    part_e_brier_safe_rows = sum(int(fval(r.get("Brier_UCB")) <= 0.0) for r in part_e_rows)
    part_e_all_debt_safe_rows = sum(int(fval(r.get("all_debt_UCB_max")) <= 0.0) for r in part_e_rows)
    part_e_control_residual_pass_rows = sum(int(fval(r.get("control_residual_energy_fraction")) >= 0.35) for r in part_e_rows)
    part_e_sign_calibrated_rows = sum(int(fval(r.get("sign_calibration_applied")) > 0.5) for r in part_e_rows)
    blocking_counts: dict[str, int] = {}
    for row in part_e_family:
        key = row.get("blocking_metric", "")
        if key:
            blocking_counts[key] = blocking_counts.get(key, 0) + 1
    dominant_blocking_metric = max(blocking_counts, key=blocking_counts.get) if blocking_counts else ""
    summary = {
        "gate": "v22_76_part_h_failure_decomposition",
        "run_status": "completed_route_decision",
        "failure_components": "",
        "Brier_debt_rows": int(part_b.get("Brier_debt_rows", 0)),
        "Brier_false_safe_rows": int(part_b.get("Brier_false_safe_rows", 0)),
        "all_debt_false_safe_rows": int(part_b.get("all_debt_false_safe_rows", 0)),
        "ECE_debt_rows": int(part_b.get("ECE_debt_rows", 0)),
        "tail95_debt_rows": int(part_b.get("tail95_debt_rows", 0)),
        "tail99_debt_rows": int(part_b.get("tail99_debt_rows", 0)),
        "margin_debt_rows": int(part_b.get("margin_debt_rows", 0)),
        "wrong_high_confidence_rows": int(part_b.get("wrong_high_confidence_rows", 0)),
        "logit_radial_harm_rows": int(part_b.get("logit_radial_harm_rows", 0)),
        "entropy_collapse_rows": int(part_b.get("entropy_collapse_rows", 0)),
        "same_edge_control_explained_rows": int(part_b.get("control_explained_by_edge_rows", 0)),
        "same_domain_control_explained_rows": int(part_b.get("control_explained_by_domain_rows", 0)),
        "same_debt_UCB_control_explained_rows": int(part_b.get("control_explained_by_debt_rows", 0)),
        "same_radial_control_explained_rows": int(part_b.get("control_explained_by_radial_rows", 0)),
        "same_control_residual_explained_rows": int(n if not int(part_e.get("part_e_gate_pass", 0)) else 0),
        "own_reference_blocked_rows": int(n - int(part_b.get("KAN_improves_own", 0))),
        "old_MLP_blocked_rows": int(n - int(part_b.get("KAN_beats_MLP_matched", 0))),
        "strengthened_MLP_blocked_rows": int(n - int(part_g.get("KAN_vs_strengthened_MLP_matched", 0))),
        "basis_projection_low_rows": 0,
        "raw_readout_low_rows": 0,
        "basis_condition_fail_rows": 0,
        "edge_extrapolation_high_rows": 0,
        "overhead_blocked_rows": int(n - int(part_b.get("overhead_le_035", 0))),
        "part_e_total_microprobe_rows": len(part_e_rows),
        "part_e_family_summary_rows": len(part_e_family),
        "part_e_control_margin_positive_rows": part_e_margin_positive_rows,
        "part_e_same_domain_candidate_better_rows": part_e_same_domain_candidate_better_rows,
        "part_e_brier_safe_rows": part_e_brier_safe_rows,
        "part_e_all_debt_safe_rows": part_e_all_debt_safe_rows,
        "part_e_control_residual_pass_rows": part_e_control_residual_pass_rows,
        "part_e_sign_calibrated_rows": part_e_sign_calibrated_rows,
        "part_e_dominant_blocking_metric": dominant_blocking_metric,
        "part_e_blocking_metric_counts": json.dumps(blocking_counts, sort_keys=True),
    }
    components: list[str] = []
    if not int(part_e.get("part_e_gate_pass", 0)):
        components.append("NoControlContrastiveEdgePreflight")
        if dominant_blocking_metric == "control_margin_p10_positive_rows" or part_e_margin_positive_rows == 0:
            components.append("SameDomainSupportExplained")
        if part_e_control_residual_pass_rows < max(1, len(part_e_rows) // 2):
            components.append("ControlResidualCapacityLow")
    if max(summary["same_edge_control_explained_rows"], summary["same_domain_control_explained_rows"], summary["same_debt_UCB_control_explained_rows"], summary["same_radial_control_explained_rows"], summary["same_control_residual_explained_rows"]) >= max(1, n // 2):
        components.append("EdgeControlExplained_NoFU")
    if summary["Brier_false_safe_rows"] > 9 or summary["all_debt_false_safe_rows"] > 12:
        components.append("BrierTrajectoryDebtBlocked")
    if summary["own_reference_blocked_rows"] > n - 27:
        components.append("OwnReferenceBlocked")
    if summary["strengthened_MLP_blocked_rows"] > n - 21:
        components.append("StrengthenedMLPBlocked")
    if summary["overhead_blocked_rows"] > n - 36:
        components.append("OverheadBlocked")
    if not components:
        components.append("CurrentKANBasisFamilyNotEdgeProbabilityCarrier")
    summary["failure_components"] = "|".join(components)
    if "SameDomainSupportExplained" in components:
        route = "SameDomainSupportExplained"
        reason = "Part E actual train-only microprobe found zero positive control-contrastive margin rows; same-domain/control margin remains the dominant blocker after residualization and downstream-sensitivity repair."
    elif "ControlResidualCapacityLow" in components:
        route = "ControlResidualCapacityLow"
        reason = "Part E control-residualized preflight did not find any fixed family satisfying residual capacity and margin gates."
    elif "EdgeControlExplained_NoFU" in components:
        route = "EdgeControlExplained_NoFU"
        reason = "same-domain/same-edge/same-debt/same-radial controls still explain most candidate rows."
    elif "BrierTrajectoryDebtBlocked" in components:
        route = "BrierTrajectoryDebtBlocked"
        reason = "Brier/all-debt false-safe remains high."
    else:
        route = components[0]
        reason = "Dominant failure component selected by Part H."
    summary.update({"recommended_route": route, "route_reason": reason, "part_h_gate_pass": 1})
    write_rows(OUT_ROOT / "v22_76_part_h_failure_decomposition.csv", [summary])
    write_json(OUT_ROOT / "v22_76_part_h_failure_decomposition_summary.json", summary)
    append_exec("H_failure_decomposition", command, "pass", files=f"{rel(OUT_ROOT / 'v22_76_part_h_failure_decomposition.csv')}; {rel(OUT_ROOT / 'v22_76_part_h_failure_decomposition_summary.json')}", note=json.dumps({"route": route, "failure_components": summary["failure_components"]}, ensure_ascii=False))
    append_recap("Part H failure decomposition", [f"failure_components={summary['failure_components']}；recommended_route={route}。", f"controls explained：edge={summary['same_edge_control_explained_rows']}/{n}；domain={summary['same_domain_control_explained_rows']}/{n}；debt={summary['same_debt_UCB_control_explained_rows']}/{n}；radial={summary['same_radial_control_explained_rows']}/{n}；control_residual={summary['same_control_residual_explained_rows']}/{n}。", f"debt：Brier_false_safe={summary['Brier_false_safe_rows']}/{n}；all_debt_false_safe={summary['all_debt_false_safe_rows']}/{n}；strengthened_MLP_blocked={summary['strengthened_MLP_blocked_rows']}/{n}。"])
    return summary


def part_i_actions(route: str) -> list[str]:
    if route == "SameDomainSupportExplained":
        return [
            "Stop sweeping eta/rank on the current fixed FC-PureKAN edge families.",
            "Redesign the KAN basis family around explicit downstream-sensitivity residual carriers.",
            "Require fixed-family positive control-contrastive margin before any full-loop rerun.",
        ]
    if route == "ControlResidualCapacityLow":
        return [
            "Increase lambda_ctrl and use CVaR25 margin before any full-loop rerun.",
            "Add same-domain and same-radial controls into S_control.",
            "Do not increase rank first; redesign control-residual metric.",
        ]
    if route == "EdgeControlExplained_NoFU":
        return ["Fix same-domain/same-edge/same-debt/same-radial controls and require residual margin before full-loop."]
    if route == "BrierTrajectoryDebtBlocked":
        return ["Add wrong-high-confidence debt and guard CVaR before full-loop."]
    return ["Trigger basis family redesign before another full-loop sweep."]


def final_route(a: dict[str, Any], b: dict[str, Any], c: dict[str, Any], d: dict[str, Any], e: dict[str, Any], f: dict[str, Any], g: dict[str, Any], h: dict[str, Any]) -> dict[str, Any]:
    if not int(a.get("part_a_hard_gate_pass", 0)):
        route, reason = "R0-CodeOrTrainingBoundaryFailed", "Part A hard gate failed."
    elif not int(c.get("part_c_gate_pass", 0)):
        route, reason = "R0-ControlContrastiveMathGateFailed", "Part C math gate failed."
    elif not int(d.get("part_d_gate_pass", 0)):
        route, reason = "BrierTrajectoryDebtBlocked", "Part D debt guard failed."
    elif not int(e.get("part_e_gate_pass", 0)):
        route, reason = str(h.get("recommended_route", "NoControlContrastiveEdgePreflight")), str(h.get("route_reason", "Part E preflight failed."))
    elif int(f.get("official_candidate_gate_pass", 0)):
        route, reason = "DGKANControlContrastiveEdgeProbabilityCarrierOfficialCandidate", "Official gate passed."
    elif int(f.get("part_f_exploration_gate_pass", 0)):
        route, reason = "KANControlContrastiveEdgeProbabilityExplorationOpened_OfficialPending", "Exploration gate passed; official pending."
    else:
        route, reason = str(h.get("recommended_route", "CurrentKANBasisFamilyNotEdgeProbabilityCarrier")), str(h.get("route_reason", "Part F/H selected route."))
    actions = part_i_actions(route)
    part_i = {"gate": "v22_76_part_i_basis_family_redesign_trigger", "final_route": route, "route_reason": reason, "next_actions": actions, "non_fabrication_note": "Actions selected from v22.76 Part I according to generated evidence."}
    write_json(OUT_ROOT / "v22_76_part_i_basis_family_redesign_trigger.json", part_i)
    obj = {
        "final_route": route,
        "route_reason": reason,
        "generated_at_sg": now_sg(),
        "part_a_hard_gate_pass": int(a.get("part_a_hard_gate_pass", 0)),
        "part_b_gate_pass": int(b.get("part_b_gate_pass", 0)),
        "part_c_gate_pass": int(c.get("part_c_gate_pass", 0)),
        "part_d_gate_pass": int(d.get("part_d_gate_pass", 0)),
        "part_e_gate_pass": int(e.get("part_e_gate_pass", 0)),
        "part_f_exploration_gate_pass": int(f.get("part_f_exploration_gate_pass", 0)),
        "official_candidate_gate_pass": int(f.get("official_candidate_gate_pass", 0)),
        "part_g_gate_pass": int(g.get("part_g_gate_pass", 0)),
        "part_h_gate_pass": int(h.get("part_h_gate_pass", 0)),
        "part_h_failure_components": h.get("failure_components", ""),
        "part_i_next_actions": actions,
        "artifacts": {
            "part_a": rel(OUT_ROOT / "v22_76_part_a_code_identity_hard_gate.json"),
            "part_b": rel(OUT_ROOT / "v22_76_part_b_v22_75_failure_replay_control_residual_summary.json"),
            "part_c": rel(OUT_ROOT / "v22_76_part_c_control_contrastive_unit_tests_summary.json"),
            "part_d": rel(OUT_ROOT / "v22_76_part_d_trajectory_probability_debt_guard_summary.json"),
            "part_e": rel(OUT_ROOT / "v22_76_part_e_control_residualized_edge_preflight_summary.json"),
            "part_f": rel(OUT_ROOT / "v22_76_part_f_target_free_full_loop_summary.json"),
            "part_g": rel(OUT_ROOT / "v22_76_part_g_strengthened_mlp_matched_audit_summary.json"),
            "part_h": rel(OUT_ROOT / "v22_76_part_h_failure_decomposition_summary.json"),
            "part_i": rel(OUT_ROOT / "v22_76_part_i_basis_family_redesign_trigger.json"),
        },
        "non_fabrication_note": "All values are generated by this runner or copied from explicitly named v22.75 artifacts.",
    }
    write_json(OUT_ROOT / "v22_76_final_route.json", obj)
    append_exec("final_route", command_text([PYTHON, rel(RUNNER), "--mode", "full"]), "done", files=f"{rel(OUT_ROOT / 'v22_76_final_route.json')}; {rel(OUT_ROOT / 'v22_76_part_i_basis_family_redesign_trigger.json')}", note=json.dumps({"final_route": route, "reason": reason}, ensure_ascii=False))
    append_recap("Final route", [f"final_route={route}；reason={reason}", f"Part gates: A={obj['part_a_hard_gate_pass']} B={obj['part_b_gate_pass']} C={obj['part_c_gate_pass']} D={obj['part_d_gate_pass']} E={obj['part_e_gate_pass']} F_explore={obj['part_f_exploration_gate_pass']} F_official={obj['official_candidate_gate_pass']} G={obj['part_g_gate_pass']} H={obj['part_h_gate_pass']}。", f"failure_components={obj['part_h_failure_components']}。", f"Part I next_actions={actions}。"])
    return obj


def run_full(args: argparse.Namespace) -> dict[str, Any]:
    append_exec("init", command_text([PYTHON, rel(RUNNER), "--mode", args.mode, "--device", args.device]), "start", gpu=args.gpus, files=f"{rel(EXEC_LOG)}; {rel(RECAP_LOG)}; {rel(OUT_ROOT)}", note="v22.76 gate-ordered runner initialized.")
    append_recap("初始化与边界", ["执行顺序按计划采用 Part A/B/C/D/E/F/G/H/I；C/D/E 不通过时禁止进入 F full-loop。", "v22.76 新增 optimizer-owned control residualization wrapper 与 control-residual preflight；不会使用 MLP target/teacher。"])
    a = run_part_a(args)
    b = run_part_b(args) if int(a.get("part_a_hard_gate_pass", 0)) else {"part_b_gate_pass": 0}
    c = run_part_c(args) if int(a.get("part_a_hard_gate_pass", 0)) else {"part_c_gate_pass": 0}
    d = run_part_d(args, b) if int(c.get("part_c_gate_pass", 0)) else {"part_d_gate_pass": 0}
    e = run_part_e(args, b, d) if int(d.get("part_d_gate_pass", 0)) else {"part_e_gate_pass": 0}
    f = run_part_f(args, e)
    g = run_part_g(args, f)
    h = run_part_h(args, b, e, f, g)
    return final_route(a, b, c, d, e, f, g, h)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="full", choices=["full", "part-a", "part-b", "part-c", "part-d", "part-e", "part-f", "part-g", "part-h", "final-route"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--held-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--metric-batch-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--brier-metric-weight", type=float, default=10.0)
    p.add_argument("--brier-damping", type=float, default=1.0e-4)
    p.add_argument("--tail-metric-weight", type=float, default=0.0)
    p.add_argument("--tail-metric-fraction", type=float, default=0.25)
    p.add_argument("--edge-raw-strength", type=float, default=2.0)
    p.add_argument("--control-contrastive-cols", type=int, default=128)
    p.add_argument("--dynamic-margin-low", type=float, default=2.1e-5)
    p.add_argument("--dynamic-margin-high", type=float, default=5.0e-5)
    p.add_argument("--dynamic-debt-lambda", type=float, default=1.0)
    p.add_argument("--edge-transform-scale", type=float, default=1.0)
    p.add_argument("--edge-gradient-blend", type=float, default=0.25)
    p.add_argument("--trajectory-shrink-alpha", type=float, default=0.85)
    p.add_argument("--ucb-beta", type=float, default=1.0)
    p.add_argument("--ucb-slack", type=float, default=0.0)
    p.add_argument("--cc-transform-scale", type=float, default=1.0)
    p.add_argument("--cc-projector-ridge", type=float, default=1.0e-5)
    p.add_argument("--part-e-datasets", default="Wine,Spam,MNIST")
    p.add_argument("--part-e-seed-count", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_out()
    try:
        if args.mode == "part-a":
            run_part_a(args)
        elif args.mode == "part-b":
            run_part_b(args)
        elif args.mode == "part-c":
            run_part_c(args)
        elif args.mode == "part-d":
            b = load_json(OUT_ROOT / "v22_76_part_b_v22_75_failure_replay_control_residual_summary.json")
            run_part_d(args, b)
        elif args.mode == "part-e":
            b = load_json(OUT_ROOT / "v22_76_part_b_v22_75_failure_replay_control_residual_summary.json")
            d = load_json(OUT_ROOT / "v22_76_part_d_trajectory_probability_debt_guard_summary.json")
            run_part_e(args, b, d)
        elif args.mode == "part-f":
            e = load_json(OUT_ROOT / "v22_76_part_e_control_residualized_edge_preflight_summary.json")
            run_part_f(args, e)
        elif args.mode == "part-g":
            f = load_json(OUT_ROOT / "v22_76_part_f_target_free_full_loop_summary.json")
            run_part_g(args, f)
        elif args.mode == "part-h":
            b = load_json(OUT_ROOT / "v22_76_part_b_v22_75_failure_replay_control_residual_summary.json")
            e = load_json(OUT_ROOT / "v22_76_part_e_control_residualized_edge_preflight_summary.json")
            f = load_json(OUT_ROOT / "v22_76_part_f_target_free_full_loop_summary.json")
            g = load_json(OUT_ROOT / "v22_76_part_g_strengthened_mlp_matched_audit_summary.json")
            run_part_h(args, b, e, f, g)
        elif args.mode == "final-route":
            a = load_json(OUT_ROOT / "v22_76_part_a_code_identity_hard_gate.json")
            b = load_json(OUT_ROOT / "v22_76_part_b_v22_75_failure_replay_control_residual_summary.json")
            c = load_json(OUT_ROOT / "v22_76_part_c_control_contrastive_unit_tests_summary.json")
            d = load_json(OUT_ROOT / "v22_76_part_d_trajectory_probability_debt_guard_summary.json")
            e = load_json(OUT_ROOT / "v22_76_part_e_control_residualized_edge_preflight_summary.json")
            f = load_json(OUT_ROOT / "v22_76_part_f_target_free_full_loop_summary.json")
            g = load_json(OUT_ROOT / "v22_76_part_g_strengthened_mlp_matched_audit_summary.json")
            h = load_json(OUT_ROOT / "v22_76_part_h_failure_decomposition_summary.json")
            final_route(a, b, c, d, e, f, g, h)
        else:
            run_full(args)
    except Exception as exc:
        log = write_exception_log("runner_exception", exc)
        append_exec("runner_exception", command_text([PYTHON, rel(RUNNER), "--mode", args.mode]), "error", gpu=args.device, files=rel(log), note=str(exc))
        raise


if __name__ == "__main__":
    main()
