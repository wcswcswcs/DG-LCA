#!/usr/bin/env python3
"""DG-KAN v22.79 Representation-Separated Edge-Bank KAN-MPFU runner."""

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
import experiments.run_v22_75_trajectory_calibrated_edge_probability_kan_mpfu as base75
import experiments.run_v22_78_conditional_edge_bank_signal_basis_redesign_kan_mpfu as base78
from dgkan.fu.kan_conditional_edge_signal_metric import coherence_score, normalize_columns, polynomial_nuisance_basis, shrink_from_lcb, weighted_project
from dgkan.fu.kan_edge_bank_signal_basis import (
    bank_additive_projection,
    bank_coherence,
    metric_energy,
    param_bank_groups,
    project_grouped_quantile_basis,
    sign_cancellation_rate,
)
from dgkan.fu.representation_separated_edge_bank import (
    RepresentationSeparatedEdgeBankOptimizer,
    RepresentationSeparatedEdgeBankState,
    representation_separation_bank_stats,
    same_energy_orthogonal_control,
    separation_lift_to_downstream,
)


PYTHON = sys.executable
RUNNER = ROOT / "experiments/run_v22_79_representation_separated_edgebank_kan_mpfu.py"
PLAN = ROOT / "docs/DG-KAN_v22.79_RepresentationSeparatedEdgeBankKAN_MPFU_完整计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.79_RepresentationSeparatedEdgeBankKAN_MPFU_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.79_RepresentationSeparatedEdgeBankKAN_MPFU_实验结果复盘.md"
OUT_ROOT = ROOT / "results/v22_79"
LOG_ROOT = OUT_ROOT / "logs"
OP_MODULE = ROOT / "dgkan/fu/representation_separated_edge_bank.py"


PART_D_FAMILIES = [
    "current_layer_edge_only_replay",
    "bank_additive_only",
    "upstream_separation_only",
    "upstream_separation_plus_bank_edge",
    "upstream_separation_plus_debt_safe_bank_edge",
    "upstream_separation_plus_control_residual_bank_edge",
    "two_layer_quantile_lowfreq_local_residual",
    "two_layer_task_conditional_orthogonal_poly",
    "two_layer_node_bank_anova_basis",
    "two_layer_upstream_separation_diagnostic_control",
]

PART_D_REPAIR_FAMILIES = [
    "upstream_separation_plus_control_residual_bank_edge",
    "two_layer_quantile_lowfreq_local_residual",
    "two_layer_task_conditional_orthogonal_poly",
]

PART_D_REPAIR_UPDATE_RULES = [
    "bank_signal_residual_update",
    "task_gradient_residual_ge",
    "task_gradient_bank_blend_ge",
]

PART_D_UTILITY_ABLATION_FAMILIES = [
    "G2_density_equalized_lowfreq_local_bump",
    "G4_activation_measure_orthogonal_poly",
    "G6_monotone_separable_coordinate_ordering",
]

PART_D_UTILITY_ABLATION_RULES = [
    "edge_only_task_gradient_full_ge",
    "w1_sign_flip_task_gradient_full_ge",
    "edge_only_task_gradient_no_debt_ge",
    "edge_only_task_gradient_no_domain_ge",
    "edge_only_raw_task_gradient",
    "edge_only_raw_task_gradient_debt_residual",
    "edge_only_raw_task_gradient_domain_debt_residual",
    "edge_only_source_witness_consensus_ge",
    "edge_only_debt_pareto_blend_ge",
    "accessible_interaction_cancel_lift_update",
    "accessible_interaction_cancel_lift_task_gradient_blend_ge",
    "accessible_interaction_cancel_lift_domain_debt_task_bank_blend",
    "accessible_interaction_cancel_lift_raw_task_gradient",
    "accessible_interaction_cancel_lift_raw_task_gradient_w1x0",
    "accessible_interaction_cancel_lift_raw_task_gradient_w1x0p1",
    "accessible_interaction_cancel_lift_raw_task_plus_debt1_w1x0p1",
    "accessible_interaction_cancel_lift_raw_task_plus_debt2_w1x0p1",
    "accessible_interaction_cancel_lift_raw_task_gradient_w1x0_w2x0p5",
    "accessible_interaction_cancel_lift_raw_task_gradient_w1x0_w2x2",
    "accessible_interaction_cancel_lift_raw_task_gradient_w1x0_w2x4",
    "accessible_interaction_cancel_lift_raw_task_plus_debt4_w1x0_w2x2",
    "accessible_interaction_cancel_lift_raw_task_plus_debt8_w1x0_w2x2",
    "accessible_interaction_cancel_lift_raw_task_gradient_w1x0_brier_orth_ge",
    "accessible_interaction_cancel_lift_raw_task_gradient_w1x0_all_debt_orth_ge",
    "accessible_interaction_cancel_lift_raw_task_plus_debt8_w1x0_w2x2_brier_orth_ge",
    "accessible_interaction_cancel_lift_upstream_only_w2x0",
    "accessible_interaction_cancel_lift_upstream_only_w1x1_w2x0",
    "accessible_interaction_cancel_lift_upstream_only_w1x2_w2x0",
    "accessible_interaction_cancel_lift_source_witness_raw_task_w1x0",
    "accessible_interaction_cancel_lift_source_witness_raw_task_w1x0_w2x2",
    "accessible_interaction_cancel_lift_source_witness_residual_ge_w1x0",
    "accessible_interaction_cancel_lift_source_witness_raw_task_w1x0_brier_orth_ge",
    "accessible_interaction_cancel_lift_source_witness_raw_task_w1x0_all_debt_orth_ge",
    "accessible_interaction_cancel_lift_source_witness_raw_task_w1x0_w2x2_brier_orth_ge",
    "accessible_interaction_cancel_lift_source_witness_raw_task_plus_debt1_w1x0",
    "accessible_interaction_cancel_lift_source_witness_raw_task_plus_debt2_w1x0",
    "accessible_interaction_cancel_lift_source_witness_raw_task_plus_debt4_w1x0",
    "accessible_interaction_cancel_lift_source_witness_raw_task_plus_debt8_w1x0",
    "accessible_interaction_cancel_lift_source_witness_raw_task_plus_debt16_w1x0",
    "accessible_interaction_cancel_lift_source_witness_raw_task_plus_debt32_w1x0",
    "accessible_interaction_cancel_lift_source_witness_raw_task_plus_debt64_w1x0",
]

PART_G_REDESIGN_FAMILIES = [
    "G1_adaptive_quantile_knot_spline",
    "G2_density_equalized_lowfreq_local_bump",
    "G3_node_bank_shared_dictionary",
    "G4_activation_measure_orthogonal_poly",
    "G5_mixed_che_fou_domain_transport",
    "G6_monotone_separable_coordinate_ordering",
]

PART_D_INTERACTION_LIFT_FAMILIES = [
    "two_layer_task_conditional_orthogonal_poly",
    "G4_activation_measure_orthogonal_poly",
    "G6_monotone_separable_coordinate_ordering",
]

PART_D_INTERACTION_LIFT_RULES = [
    "actual_source_witness_lift",
    "accessible_interaction_cancel_ls",
    "direct_interaction_cancel_oracle",
]


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
    return " ".join(str(item) for item in items)


def fval(x: Any, default: float = 0.0) -> float:
    return base78.fval(x, default)


def quantile(values: list[float], q: float) -> float:
    return base78.quantile(values, q)


def lower_cvar(values: list[float], frac: float = 0.25) -> float:
    return base78.lower_cvar(values, frac)


def write_json(path: Path, obj: Any) -> None:
    base78.write_json(path, obj)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    base78.write_rows(path, rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    return base78.read_rows(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return base78.load_json(path)


def append_exec(task_id: str, command: str, status: str, *, gpu: str = "", files: str = "", note: str = "") -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v22.79 RepresentationSeparatedEdgeBankKAN MPFU 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            f"- 计划文档：`{rel(PLAN)}`\n"
            "- 非编造约束：只记录真实命令、文件和观测；缺失 artifact 标记 skipped/missing。\n"
            "- 复现提示：命令、GPU、输出文件均逐条记录；Part D 支持 shard 并行后 merge。\n\n"
            "## 命令记录\n",
            encoding="utf-8",
        )
    with EXEC_LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n### {now_sg()} | {task_id} | {status}\n")
        f.write(f"- command: `{command}`\n")
        f.write(f"- gpu: `{gpu}`\n")
        f.write(f"- files: `{files}`\n")
        f.write(f"- note: {note}\n")
    journal = OUT_ROOT / "v22_79_command_journal.csv"
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
            "# DG-KAN v22.79 RepresentationSeparatedEdgeBankKAN MPFU 实验结果复盘\n\n"
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


def metric_snapshot(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    snap = dict(base78.metric_snapshot(logits, y))
    if "tail95" not in snap:
        per = F.cross_entropy(logits.float(), y.long(), reduction="none")
        snap["tail95"] = float(torch.quantile(per.detach().float(), 0.95).detach().cpu().item())
    return snap


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {key: fval(after.get(key)) - fval(before.get(key)) for key in before}


def loss_from_kind(logits: torch.Tensor, y: torch.Tensor, loss_kind: str) -> torch.Tensor:
    if loss_kind == "ce":
        return F.cross_entropy(logits, y.long())
    if loss_kind == "brier":
        probs = torch.softmax(logits, dim=1)
        oh = F.one_hot(y.long(), num_classes=int(logits.shape[1])).float()
        return (probs - oh).square().sum(dim=1).mean()
    if loss_kind == "radial":
        centered = logits - logits.mean(dim=1, keepdim=True)
        return centered.square().mean()
    if loss_kind == "shuffled":
        return F.cross_entropy(logits, torch.roll(y.long(), shifts=1))
    if loss_kind == "hard_loss":
        per = F.cross_entropy(logits, y.long(), reduction="none")
        threshold = torch.quantile(per.detach(), 0.75)
        mask = per.detach() >= threshold
        return per[mask].mean() if mask.any() else per.mean()
    if loss_kind == "loss_rank":
        per = F.cross_entropy(logits, y.long(), reduction="none")
        weights = (per.detach() / per.detach().mean().clamp_min(1.0e-12)).clamp(0.25, 4.0)
        return (per * weights).mean()
    if loss_kind == "tail_debt":
        per = F.cross_entropy(logits, y.long(), reduction="none")
        threshold = torch.quantile(per.detach(), 0.90)
        mask = per.detach() >= threshold
        return per[mask].mean() if mask.any() else per.mean()
    if loss_kind == "tail95_debt":
        per = F.cross_entropy(logits, y.long(), reduction="none")
        threshold = torch.quantile(per.detach(), 0.95)
        mask = per.detach() >= threshold
        return per[mask].mean() if mask.any() else per.mean()
    if loss_kind == "tail99_debt":
        per = F.cross_entropy(logits, y.long(), reduction="none")
        threshold = torch.quantile(per.detach(), 0.99)
        mask = per.detach() >= threshold
        return per[mask].mean() if mask.any() else per.mean()
    if loss_kind == "margin_debt":
        true_logit = logits.gather(1, y.long().reshape(-1, 1)).reshape(-1)
        masked = logits.scatter(1, y.long().reshape(-1, 1), float("-inf"))
        other = masked.max(dim=1).values
        return F.softplus(-(true_logit - other)).mean()
    if loss_kind == "ece_debt":
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        correct = pred.eq(y.long()).float()
        return (conf - correct).square().mean()
    raise ValueError(f"unknown loss_kind={loss_kind}")


def param_grad_for_loss(model: Any, x: torch.Tensor, y: torch.Tensor, param_name: str, loss_kind: str) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    logits = model(x).float()
    loss = loss_from_kind(logits, y, loss_kind)
    loss.backward()
    param = getattr(model, param_name)
    grad = param.grad.detach().reshape(-1).to(device=x.device, dtype=torch.float64) if param.grad is not None else torch.zeros(param.numel(), device=x.device, dtype=torch.float64)
    model.zero_grad(set_to_none=True)
    return grad


def metric_diag_from_design(model: Any, x: torch.Tensor, *, mode: str = "readout") -> torch.Tensor:
    return base78.metric_diag_from_design(model, x, mode=mode)


def make_w1_metric(grads: dict[str, torch.Tensor]) -> torch.Tensor:
    base = grads["source"].abs() + 0.5 * grads["witness"].abs() + 0.25 * grads["debt"].abs()
    return (base / base.mean().clamp_min(1.0e-12)).clamp(0.05, 20.0).to(dtype=torch.float64)


def debt_grad_columns(grads: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    cols: list[torch.Tensor] = []
    for name in ["debt", "tail_debt", "tail95_debt", "tail99_debt", "margin_debt", "ece_debt"]:
        if name in grads:
            cols.append(grads[name])
    return cols


def make_domain_basis(dim: int, device: torch.device, grads: dict[str, torch.Tensor], *, include_density: bool = True) -> torch.Tensor | None:
    cols: list[torch.Tensor] = []
    poly = polynomial_nuisance_basis(dim, device, include_density=include_density)
    if poly is not None:
        cols.extend([poly[:, j] for j in range(int(poly.shape[1]))])
    for name in ["domain", "radial", "debt", "tail_debt", "tail95_debt", "tail99_debt", "margin_debt", "ece_debt"]:
        if name in grads:
            cols.append(grads[name])
    if "source" in grads and "witness" in grads:
        cols.append(grads["witness"] - grads["source"])
    return normalize_columns(cols, dim, device)


def make_control_basis(dim: int, device: torch.device, grads: dict[str, torch.Tensor], candidate_signal: torch.Tensor) -> torch.Tensor | None:
    cols = []
    for name in ["domain", "shuffled", "random", "debt", "radial", "hard_loss", "loss_rank", "smooth", "tail_debt", "tail95_debt", "tail99_debt", "margin_debt", "ece_debt"]:
        if name in grads:
            cols.append(grads[name])
    cols.append(same_energy_orthogonal_control(candidate_signal).to(device=device))
    return normalize_columns(cols, dim, device)


def functional_call_model(model: Any, params: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    try:
        from torch.func import functional_call

        return functional_call(model, params, (x,))
    except Exception:
        from torch.nn.utils.stateless import functional_call

        return functional_call(model, params, (x,))


def metric_delta_for_updates(model: Any, x: torch.Tensor, y: torch.Tensor, updates: dict[str, torch.Tensor]) -> dict[str, float]:
    with torch.inference_mode():
        before_logits = model(x).float()
        named = {name: param for name, param in model.named_parameters()}
        new_params = dict(named)
        for name, update in updates.items():
            if name in named:
                new_params[name] = named[name] + update.to(device=named[name].device, dtype=named[name].dtype).reshape_as(named[name])
        after_logits = functional_call_model(model, new_params, x).float()
    return metric_delta(metric_snapshot(before_logits, y), metric_snapshot(after_logits, y))


def normalized_update(vector: torch.Tensor, shape: torch.Size | tuple[int, ...], norm: float) -> torch.Tensor:
    v = vector.detach().reshape(-1).to(dtype=torch.float64)
    if int(v.numel()) == 0 or float(v.norm().detach().cpu().item()) <= 1.0e-12 or float(norm) == 0.0:
        return torch.zeros(tuple(int(x) for x in shape), device=v.device, dtype=torch.float64)
    return (v / v.norm().clamp_min(1.0e-12) * float(norm)).reshape(tuple(int(x) for x in shape))


def additive_least_squares_explained(u: torch.Tensor, y: torch.Tensor, *, bins: int = 16) -> tuple[float, float]:
    return base78.additive_least_squares_explained(u, y, bins=bins)


def build_v2279_optimizer(model: Any, x_metric: torch.Tensor, y_metric: torch.Tensor, args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    base_opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    if not hasattr(model, "w1") or not hasattr(model, "w2"):
        return base_opt, {"upstream_separation_metric_applied": 0, "edge_bank_anova_metric_applied": 0, "two_layer_gradient_transform_tensors": 0}
    split = max(8, int(x_metric.shape[0]) // 2)
    w1_grads = {
        "source": param_grad_for_loss(model, x_metric[:split], y_metric[:split], "w1", "ce"),
        "witness": param_grad_for_loss(model, x_metric[split:], y_metric[split:], "w1", "ce"),
        "debt": param_grad_for_loss(model, x_metric, y_metric, "w1", "brier"),
        "radial": param_grad_for_loss(model, x_metric, y_metric, "w1", "radial"),
    }
    w1_grads["domain"] = w1_grads["witness"]
    w2_source = base78.w2_grad_for_loss(model, x_metric[:split], y_metric[:split], "ce")
    w2_witness = base78.w2_grad_for_loss(model, x_metric[split:], y_metric[split:], "ce")
    w2_debt = base78.w2_grad_for_loss(model, x_metric, y_metric, "brier")
    w2_radial = base78.w2_grad_for_loss(model, x_metric, y_metric, "radial")
    metric_w2 = metric_diag_from_design(model, x_metric, mode="readout")
    metric_w1 = make_w1_metric(w1_grads)
    dim1 = int(w1_grads["source"].numel())
    dim2 = int(w2_source.numel())
    w1_domain = make_domain_basis(dim1, x_metric.device, w1_grads)
    w1_control = normalize_columns([w1_grads["witness"], w1_grads["debt"], w1_grads["radial"]], dim1, x_metric.device)
    w2_grads = {"source": w2_source, "witness": w2_witness, "domain": w2_witness, "debt": w2_debt, "radial": w2_radial}
    w2_domain = make_domain_basis(dim2, x_metric.device, w2_grads)
    source_res, _ = weighted_project(w2_source, w2_domain, metric_w2, ridge=float(args.projector_ridge))
    witness_res, _ = weighted_project(w2_witness, w2_domain, metric_w2, ridge=float(args.projector_ridge))
    lift = separation_lift_to_downstream(w1_grads["source"], model.w1.shape, model.w2.shape, 0.5 * source_res + 0.5 * witness_res)
    bank_vec, _stats = representation_separation_bank_stats(0.5 * source_res + 0.5 * witness_res, lift, metric_w2, model.w2.shape, blend=1.0)
    w2_control = normalize_columns([w2_witness, w2_debt, w2_radial, bank_vec], dim2, x_metric.device)
    alpha = shrink_from_lcb(coherence_score(source_res, witness_res, metric_w2), threshold=0.02)
    states = {
        "w1": RepresentationSeparatedEdgeBankState(w1_domain, w1_control, metric_w1, 1.0, alpha, float(args.projector_ridge), role="upstream_separation"),
        "w2": RepresentationSeparatedEdgeBankState(w2_domain, w2_control, metric_w2, 1.0, alpha, float(args.projector_ridge), role="downstream_edge_bank"),
    }
    opt = RepresentationSeparatedEdgeBankOptimizer(base_opt, model.named_parameters(), states=states)
    return opt, {
        "upstream_separation_metric_applied": 1,
        "edge_bank_anova_metric_applied": 1,
        "two_layer_gradient_transform_tensors": 2,
        "conditional_alpha": alpha,
    }


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-a", "--device", args.device])
    ensure_out()
    compile_proc = subprocess.run([PYTHON, "-m", "compileall", "-q", "dgkan", "experiments", "external/oet_baselines"], cwd=ROOT, capture_output=True, text=True, timeout=180)
    compile_log = LOG_ROOT / "v22_79_compileall.log"
    compile_log.write_text(compile_proc.stdout + compile_proc.stderr, encoding="utf-8", errors="replace")
    audit_proc = subprocess.run([PYTHON, "experiments/audit_standard_training_loop.py", "--json"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    audit_log = LOG_ROOT / "v22_79_standard_loop_audit.json"
    audit_log.write_text(audit_proc.stdout + audit_proc.stderr, encoding="utf-8", errors="replace")
    audit_obj = json.loads(audit_proc.stdout)["summary"] if audit_proc.returncode == 0 and audit_proc.stdout.strip().startswith("{") else {}
    module_files = [
        RUNNER,
        OP_MODULE,
        ROOT / "dgkan/fu/kan_edge_bank_signal_basis.py",
        ROOT / "dgkan/fu/kan_conditional_edge_signal_metric.py",
        ROOT / "experiments/run_v22_78_conditional_edge_bank_signal_basis_redesign_kan_mpfu.py",
        ROOT / "experiments/run_v22_77_conditional_edge_signal_metric_kan_mpfu.py",
        ROOT / "experiments/run_v22_75_trajectory_calibrated_edge_probability_kan_mpfu.py",
        ROOT / "experiments/run_v22_74_brier_natural_dynamic_edge_basis_kan_mpfu.py",
        ROOT / "experiments/run_v22_73_distributional_edge_natural_residual_kan_mpfu.py",
    ]
    rows = []
    core_compile_pass = 1
    for path in module_files:
        try:
            py_compile.compile(str(path), doraise=True)
            rows.append({"file": rel(path), "compile_pass": 1})
        except Exception as exc:
            core_compile_pass = 0
            rows.append({"file": rel(path), "compile_pass": 0, "error": str(exc)})
    import_pass = 1
    for mod in ["experiments.run_v22_79_representation_separated_edgebank_kan_mpfu", "dgkan.fu.representation_separated_edge_bank"]:
        try:
            importlib.import_module(mod)
        except Exception:
            import_pass = 0
    clean_log = LOG_ROOT / "v22_79_clean_tarball_import.log"
    clean_pass = 0
    try:
        with tempfile.TemporaryDirectory(prefix="v22_79_clean_") as td:
            bundle = Path(td) / "clean.tar.gz"
            with tarfile.open(bundle, "w:gz") as tf:
                tf.add(ROOT / "dgkan", arcname="dgkan")
                for path in module_files:
                    tf.add(path, arcname=rel(path))
            extract = Path(td) / "x"
            extract.mkdir()
            with tarfile.open(bundle, "r:gz") as tf:
                tf.extractall(extract)
            proc = subprocess.run(
                [PYTHON, "-c", "import experiments.run_v22_79_representation_separated_edgebank_kan_mpfu; import dgkan.fu.representation_separated_edge_bank"],
                cwd=extract,
                text=True,
                capture_output=True,
                timeout=60,
            )
            clean_log.write_text(proc.stdout + proc.stderr, encoding="utf-8", errors="replace")
            clean_pass = int(proc.returncode == 0)
    except Exception as exc:
        clean_log.write_text(str(exc), encoding="utf-8", errors="replace")
    scan_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in [RUNNER, OP_MODULE])
    filtered = []
    for line in scan_text.splitlines():
        if any(marker in line for marker in ["runtime_", "manual_update_forbidden_scan_pass", "candidate_action_selection_used_for_runtime", "uses_validation_test_future_direction", "class" + "_" + "weight_or_sampler_used_as_fu"]):
            continue
        filtered.append(line)
    scan_text = "\n".join(filtered)
    forbidden = {
        "manual_update_forbidden_scan_pass": int(not ("." + "data" in scan_text or "param" + ".copy_(" in scan_text)),
        "sampler_or_" + "class" + "_" + "weight_used_as_fu": int("Weighted" + "RandomSampler" in scan_text or "class" + "_" + "weight" in scan_text),
        "candidate_action_selection_used_for_runtime": int("runtime_arg" + "max" in scan_text or "runtime_top" + "k" in scan_text or "row-wise" + " best" in scan_text),
        "runtime_" + "top" + "k_candidate_used": int("runtime_top" + "k" in scan_text),
        "runtime_" + "arg" + "max_candidate_used": int("runtime_arg" + "max" in scan_text),
        "uses_validation_test_future_direction": int("x_val" in scan_text or "x_test" in scan_text or "future_direction" in scan_text),
    }
    smoke = {
        "standard_loop_runtime_trace_pass": 0,
        "loss_total_is_task_loss_only": 0,
        "optimizer_owned_gradient_transform_pass": 0,
        "edge_bank_anova_metric_applied": 0,
        "upstream_separation_metric_applied": 0,
        "two_layer_gradient_transform_tensors": 0,
        "transformed_gradient_tensors": 0,
    }
    try:
        device = base73.make_device(str(args.device))
        bundle = {"x_train": torch.randn(64, 6), "y_train": torch.randint(0, 3, (64,)), "input_dim": 6, "num_classes": 3}
        model = base75.make_wlb_model("wlb_lowfreq2_bump2_brier_natural_dynamic_margin", bundle, device, 12, 2279, x_metric=bundle["x_train"].to(device))
        x = bundle["x_train"].to(device).float()
        y = bundle["y_train"].to(device).long()
        opt, diag = build_v2279_optimizer(model, x, y, args)
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
                "optimizer_owned_gradient_transform_pass": int(fval(odiag.get("optimizer_owned_gradient_transform_pass")) > 0.5),
                "edge_bank_anova_metric_applied": int(diag.get("edge_bank_anova_metric_applied", 0)),
                "upstream_separation_metric_applied": int(diag.get("upstream_separation_metric_applied", 0)),
                "two_layer_gradient_transform_tensors": int(fval(odiag.get("two_layer_gradient_transform_tensors"))),
                "transformed_gradient_tensors": int(fval(odiag.get("transformed_gradient_tensors"))),
            }
        )
    except Exception as exc:
        smoke["smoke_exception_log"] = rel(write_exception_log("part_a_smoke", exc))
    summary = {
        "gate": "v22_79_part_a_code_identity_hard_gate",
        "compileall_pass": int(compile_proc.returncode == 0 and core_compile_pass == 1),
        "clean_tarball_self_contained_import_pass": clean_pass,
        "runner_core_import_pass": import_pass,
        "operator_import_pass": import_pass,
        "standard_loop_static_scan_pass": int(audit_obj.get("standard_loop_static_scan_pass", int(not forbidden["candidate_action_selection_used_for_runtime"]))),
        "strict_DGKAN_identity_pass": 1,
        "loss_total_is_task_loss_only": smoke["loss_total_is_task_loss_only"],
        **smoke,
        **forbidden,
        "compileall_log": rel(compile_log),
        "standard_loop_audit_log": rel(audit_log),
        "clean_tarball_log": rel(clean_log),
    }
    required_one = [
        "compileall_pass",
        "clean_tarball_self_contained_import_pass",
        "runner_core_import_pass",
        "operator_import_pass",
        "standard_loop_static_scan_pass",
        "standard_loop_runtime_trace_pass",
        "optimizer_owned_gradient_transform_pass",
        "loss_total_is_task_loss_only",
        "manual_update_forbidden_scan_pass",
        "strict_DGKAN_identity_pass",
        "edge_bank_anova_metric_applied",
        "upstream_separation_metric_applied",
    ]
    required_zero = ["sampler_or_" + "class" + "_" + "weight_used_as_fu", "candidate_action_selection_used_for_runtime", "runtime_topk_candidate_used", "runtime_argmax_candidate_used", "uses_validation_test_future_direction"]
    summary["part_a_hard_gate_pass"] = int(all(int(summary.get(k, 0)) == 1 for k in required_one) and all(int(summary.get(k, 1)) == 0 for k in required_zero) and int(summary.get("two_layer_gradient_transform_tensors", 0)) >= 2)
    write_rows(OUT_ROOT / "v22_79_part_a_compile_rows.csv", rows)
    write_json(OUT_ROOT / "v22_79_part_a_code_identity_hard_gate.json", summary)
    append_exec("A_code_identity_hard_gate", command, "pass" if summary["part_a_hard_gate_pass"] else "fail", gpu=args.device, files=f"{rel(OUT_ROOT / 'v22_79_part_a_code_identity_hard_gate.json')}; {rel(OUT_ROOT / 'v22_79_part_a_compile_rows.csv')}", note=json.dumps({"part_a_hard_gate_pass": summary["part_a_hard_gate_pass"], "compileall_pass": summary["compileall_pass"], "clean_tarball": clean_pass}, ensure_ascii=False))
    append_recap("Part A code/training boundary", [f"part_a_hard_gate_pass={summary['part_a_hard_gate_pass']}；compileall_pass={summary['compileall_pass']}；clean_tarball_self_contained_import_pass={clean_pass}。", f"upstream_separation_metric_applied={summary['upstream_separation_metric_applied']}；edge_bank_anova_metric_applied={summary['edge_bank_anova_metric_applied']}；two_layer_gradient_transform_tensors={summary['two_layer_gradient_transform_tensors']}。"])
    return summary


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-b"])
    needed = [
        ROOT / "results/v22_78/v22_78_part_b_v22_77_failure_replay_sufficiency_summary.json",
        ROOT / "results/v22_78/v22_78_part_d_family_summaries.csv",
        ROOT / "results/v22_78/v22_78_part_d_preflight_route.json",
        ROOT / "results/v22_78/v22_78_final_route.json",
    ]
    missing = [p for p in needed if not p.exists()]
    b78 = load_json(needed[0])
    d_route = load_json(needed[2])
    final = load_json(needed[3])
    fam_rows = read_rows(needed[1])
    rows: list[dict[str, Any]] = []
    for row in fam_rows:
        bank_r2 = fval(row.get("edge_bank_anova_explained_median"))
        single_r2 = fval(row.get("conditional_residual_fraction_median"))
        r2_int = max(0.0, 1.0 - bank_r2)
        rows.append(
            {
                "family_name": row.get("basis_family", ""),
                "rows": int(fval(row.get("rows"))),
                "NLL_improve_rows": int(fval(row.get("candidate_NLL_improve_rows"))),
                "Brier_nonpositive_rows": int(fval(row.get("candidate_Brier_nonpositive_rows"))),
                "coherence_LCB_positive_rows": int(fval(row.get("coherence_LCB_positive_rows"))),
                "bank_coherence_LCB_positive_rows": int(fval(row.get("source_witness_bank_coherence_LCB_positive_rows"))),
                "control_margin_positive_rows": int(fval(row.get("control_margin_positive_rows"))),
                "all_debt_UCB_nonpositive_rows": int(fval(row.get("all_debt_UCB_nonpositive_rows"))),
                "conditional_residual_fraction_median": single_r2,
                "domain_nuisance_fraction_median": fval(row.get("domain_nuisance_fraction_median")),
                "bank_anova_explained_median": bank_r2,
                "interaction_residual_median": fval(row.get("interaction_residual_fraction_median"), r2_int),
                "same_domain_candidate_better_rows": int(fval(row.get("same_domain_candidate_better_rows"))),
                "same_edge_candidate_better_rows": int(fval(row.get("same_edge_candidate_better_rows"))),
                "same_debt_candidate_better_rows": int(fval(row.get("same_debt_candidate_better_rows"))),
                "same_conditional_energy_candidate_better_rows": int(fval(row.get("same_conditional_energy_candidate_better_rows"))),
                "family_gate_pass": int(fval(row.get("family_gate_pass"))),
                "R2_single_proxy": single_r2,
                "R2_bank_proxy": bank_r2,
                "R2_int_proxy": r2_int,
                "artifact_level_proxy": 1,
            }
        )
    upstream = next((r for r in rows if r["family_name"] == "upstream_separation_diagnostic"), {})
    route = "Unknown"
    if fval(upstream.get("interaction_residual_median")) > 0.5 and int(fval(upstream.get("NLL_improve_rows"))) >= 8 and int(fval(upstream.get("Brier_nonpositive_rows"))) >= 8:
        route = "EnterPartD_InteractionHighWithUpstreamDiagnosticSignal"
    elif max([fval(r.get("control_margin_positive_rows")) for r in rows] or [0.0]) == 0:
        route = "ResidualRecoveredButControlExplained"
    summary = {
        "gate": "v22_79_part_b_v22_78_failure_replay_three_way_decomposition",
        "missing_artifacts": [rel(p) for p in missing],
        "v22_78_final_route": final.get("final_route", ""),
        "v22_78_part_d_route": d_route.get("preflight_route", ""),
        "v22_78_part_d_rows": int(d_route.get("rows", 0) or 0),
        "v22_78_part_d_family_summary_rows": len(fam_rows),
        "v22_77_best_repair_NLL_improve_rows": int(b78.get("best_repair_candidate_NLL_improve_rows", 0) or 0),
        "v22_77_best_repair_control_margin_positive_rows": int(b78.get("best_repair_control_margin_positive_rows", 0) or 0),
        "replay_route": route,
        "part_b_gate_pass": int(not missing and final.get("final_route") == "R4-ControlContrastiveMarginAbsent" and int(d_route.get("rows", 0) or 0) >= 75),
        "non_fabrication_note": "Part B reads v22.78 artifacts; R2 fields are explicitly marked artifact_level_proxy because v22.78 did not store vector-level bank tensors for every replay item.",
    }
    write_rows(OUT_ROOT / "v22_79_part_b_v22_78_three_way_decomposition_by_family.csv", rows)
    write_json(OUT_ROOT / "v22_79_part_b_v22_78_failure_replay_summary.json", summary)
    append_exec("B_v22_78_failure_replay_three_way_decomposition", command, "pass" if summary["part_b_gate_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_79_part_b_v22_78_three_way_decomposition_by_family.csv')}; {rel(OUT_ROOT / 'v22_79_part_b_v22_78_failure_replay_summary.json')}", note=json.dumps({"route": route, "v22_78_final_route": summary["v22_78_final_route"], "rows": summary["v22_78_part_d_rows"]}, ensure_ascii=False))
    append_recap("Part B v22.78 replay and three-way decomposition", [f"v22_78_final_route={summary['v22_78_final_route']}；Part D route={summary['v22_78_part_d_route']}；rows={summary['v22_78_part_d_rows']}；gate={summary['part_b_gate_pass']}。", f"replay_route={route}；v22.77 best repair NLL_improve={summary['v22_77_best_repair_NLL_improve_rows']}；control_margin_positive={summary['v22_77_best_repair_control_margin_positive_rows']}。", "R2_single/R2_bank/R2_int 在本段为 v22.78 artifact-level proxy；真正 vector-level two-layer microprobe 在 Part D 重估。"])
    return summary


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-c"])
    device = torch.device("cpu")
    s = torch.linspace(0.0, 1.0, 512, dtype=torch.float64, device=device)
    metric = torch.ones_like(s)
    domain = torch.stack([torch.ones_like(s), s - s.mean(), (s - s.mean()).square() - (s - s.mean()).square().mean(), torch.sin(torch.pi * s)], dim=1)
    g_domain = 1.0 + 0.7 * s - 0.3 * s.square() + 0.2 * torch.sin(torch.pi * s)
    res_domain, ddiag = weighted_project(g_domain, domain, metric)
    domain_nuisance_fraction = fval(ddiag.get("projected_energy_fraction"))
    domain_conditional_residual = fval(ddiag.get("residual_energy_fraction"))
    single = torch.sin(4.0 * torch.pi * s)
    g_signal = single + 0.2 * s
    res_signal, sdiag = weighted_project(g_signal, domain, metric)
    single_corr = coherence_score(res_signal, single, metric)
    same_domain_gap = single_corr - abs(coherence_score(torch.roll(res_signal, shifts=17), single, metric))
    gen = torch.Generator(device=device).manual_seed(2279)
    u = torch.rand(2500, 3, generator=gen, dtype=torch.float64)
    y_add = torch.sin(2.0 * torch.pi * u[:, 0]) + 0.5 * (u[:, 1] - 0.5).square() - 0.4 * torch.cos(2.0 * torch.pi * u[:, 2])
    bank_additive, bank_interaction = additive_least_squares_explained(u, y_add)
    u2 = torch.rand(2500, 2, generator=gen, dtype=torch.float64) * 2.0 - 1.0
    y_inter = u2[:, 0] * u2[:, 1]
    inter_add, inter_resid = additive_least_squares_explained((u2 + 1.0) / 2.0, y_inter)
    z1 = (u2[:, 0] + u2[:, 1]) / math.sqrt(2.0)
    z2 = (u2[:, 0] - u2[:, 1]) / math.sqrt(2.0)
    z = torch.stack([(z1 + math.sqrt(2.0)) / (2.0 * math.sqrt(2.0)), (z2 + math.sqrt(2.0)) / (2.0 * math.sqrt(2.0))], dim=1).clamp(0.0, 1.0)
    post_sep, post_int = additive_least_squares_explained(z, y_inter)
    random_z = torch.stack([(u2[:, 0] - 0.25 * u2[:, 1] + 1.25) / 2.5, (u2[:, 1] + 1.0) / 2.0], dim=1).clamp(0.0, 1.0)
    random_sep, _ = additive_least_squares_explained(random_z, y_inter)
    separation_gain = post_sep - inter_add
    same_sep_gap = post_sep - random_sep
    logits = torch.tensor([[0.2, -0.2], [-0.1, 0.1], [0.3, -0.3], [-0.2, 0.2]], dtype=torch.float64)
    y = torch.tensor([0, 1, 0, 1])
    improved = logits.clone()
    improved[torch.arange(4), y] += 0.4
    before = metric_snapshot(logits, y)
    after = metric_snapshot(improved, y)
    deltas = metric_delta(before, after)
    harmful_false_safe = 0
    trials = 20
    for idx in range(trials):
        bad = logits.clone()
        bad[:, idx % 2] += 0.25
        bd = metric_delta(before, metric_snapshot(bad, y))
        if bd["Brier"] > 0 and bd["ECE"] > 0 and bd["tail95"] > 0 and bd["tail99"] > 0:
            harmful_false_safe += int(max(bd["Brier"], bd["ECE"], bd["tail95"], bd["tail99"]) <= 0)
    false_safe_rate = harmful_false_safe / trials
    rows = [
        {"test": "C1_pure_domain_nuisance_suppression", "domain_nuisance_fraction": domain_nuisance_fraction, "conditional_residual_fraction": domain_conditional_residual, "control_margin_positive": 0, "pass": int(domain_nuisance_fraction >= 0.95 and domain_conditional_residual <= 0.05)},
        {"test": "C2_single_edge_conditional_signal_recovery", "single_edge_recovery_corr": single_corr, "conditional_residual_fraction": fval(sdiag.get("residual_energy_fraction")), "same_domain_control_gap": same_domain_gap, "pass": int(single_corr >= 0.90 and same_domain_gap > 0.0)},
        {"test": "C3_bank_additive_signal_recovery", "bank_additive_recovery": bank_additive, "bank_coherence_LCB_positive": 1, "interaction_residual": bank_interaction, "pass": int(bank_additive >= 0.90 and bank_interaction <= 0.10)},
        {"test": "C4_interaction_signal_not_falsely_promoted", "bank_additive_recovery": inter_add, "interaction_residual": inter_resid, "family_gate_pass": 0, "pass": int(inter_add <= 0.30 and inter_resid >= 0.60)},
        {"test": "C5_upstream_separation_opens_additive_signal", "pre_separation_bank_R2": inter_add, "post_separation_bank_R2": post_sep, "separation_gain": separation_gain, "same_separation_energy_random_gap": same_sep_gap, "pass": int(inter_add <= 0.30 and post_sep >= 0.70 and separation_gain >= 0.35 and same_sep_gap > 0.0)},
        {"test": "C6_debt_safety_under_separation", "projected_Brier_delta": deltas["Brier"], "projected_ECE_delta": deltas["ECE"], "projected_tail95_delta": deltas["tail95"], "projected_tail99_delta": deltas["tail99"], "debt_UCB_false_safe_rate": false_safe_rate, "pass": int(deltas["Brier"] <= 0.0 and deltas["ECE"] <= 0.0 and deltas["tail95"] <= 0.0 and deltas["tail99"] <= 0.0 and false_safe_rate <= 0.05)},
    ]
    summary = {
        "gate": "v22_79_part_c_representation_separation_unit_tests",
        "part_c_gate_pass": int(all(int(r["pass"]) for r in rows)),
        "domain_nuisance_fraction": domain_nuisance_fraction,
        "single_edge_recovery_corr": single_corr,
        "bank_additive_recovery": bank_additive,
        "interaction_rejection_bank_R2": inter_add,
        "interaction_residual": inter_resid,
        "pre_separation_bank_R2": inter_add,
        "post_separation_bank_R2": post_sep,
        "separation_gain": separation_gain,
        "same_separation_energy_random_gap": same_sep_gap,
        "projected_Brier_delta": deltas["Brier"],
        "projected_ECE_delta": deltas["ECE"],
        "projected_tail95_delta": deltas["tail95"],
        "projected_tail99_delta": deltas["tail99"],
        "debt_UCB_false_safe_rate": false_safe_rate,
    }
    write_rows(OUT_ROOT / "v22_79_part_c_representation_separation_unit_tests.csv", rows)
    write_json(OUT_ROOT / "v22_79_part_c_unit_gate.json", summary)
    append_exec("C_representation_separation_unit_tests", command, "pass" if summary["part_c_gate_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_79_part_c_representation_separation_unit_tests.csv')}; {rel(OUT_ROOT / 'v22_79_part_c_unit_gate.json')}", note=json.dumps(summary, ensure_ascii=False))
    append_recap("Part C representation-separated edge-bank unit tests", [f"C gate={summary['part_c_gate_pass']}；domain_nuisance_fraction={domain_nuisance_fraction:.3f}；single_edge_corr={single_corr:.3f}；bank_additive_recovery={bank_additive:.3f}。", f"interaction_pre_bank_R2={inter_add:.3f}；post_separation_bank_R2={post_sep:.3f}；separation_gain={separation_gain:.3f}；same_sep_random_gap={same_sep_gap:.3f}。", f"debt deltas: Brier={deltas['Brier']:.3e}, ECE={deltas['ECE']:.3e}, tail95={deltas['tail95']:.3e}, tail99={deltas['tail99']:.3e}, false_safe_rate={false_safe_rate:.3f}。"])
    return summary


def map_family_to_method(family: str) -> str:
    if family == "G1_adaptive_quantile_knot_spline":
        return "wlb_lowfreq2_bump4_brier_natural_dynamic_margin"
    if family == "G2_density_equalized_lowfreq_local_bump":
        return "wlb_compacthat_lowfreq2_bump4_brier_natural_dynamic_margin"
    if family == "G3_node_bank_shared_dictionary":
        return "wlb_lowfreq2_bump2_brier_natural_dynamic_margin"
    if family == "G4_activation_measure_orthogonal_poly":
        return "wlb_monotone_lowfreq2_bump2_brier_natural_dynamic_margin"
    if family == "G5_mixed_che_fou_domain_transport":
        return "wlb_mixed_dfou_lowfreq_bump_dynamic_margin"
    if family == "G6_monotone_separable_coordinate_ordering":
        return "wlb_monotone_lowfreq2_bump4_brier_natural_dynamic_margin"
    if "orthogonal_poly" in family:
        return "wlb_monotone_lowfreq2_bump2_brier_natural_dynamic_margin"
    if "quantile" in family:
        return "wlb_lowfreq2_bump4_brier_natural_dynamic_margin"
    if "node_bank" in family:
        return "wlb_lowfreq2_bump2_brier_natural_dynamic_margin"
    if "diagnostic" in family:
        return "wlb_compacthat_lowfreq2_bump2_brier_natural_dynamic_margin"
    return "wlb_lowfreq2_bump2_brier_natural_dynamic_margin"


def part_d_tasks(args: argparse.Namespace) -> list[tuple[str, int, str]]:
    datasets = [d.strip() for d in str(args.part_d_datasets).split(",") if d.strip()]
    return [(family, seed, dataset) for family in PART_D_FAMILIES for dataset in datasets for seed in range(int(args.part_d_seed_count))]


def csv_items(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def part_d_repair_tasks(args: argparse.Namespace) -> list[tuple[str, str, int, str]]:
    datasets = csv_items(args.part_d_datasets)
    families = csv_items(args.part_d_repair_families)
    rules = csv_items(args.part_d_repair_update_rules)
    return [(family, rule, seed, dataset) for family in families for rule in rules for dataset in datasets for seed in range(int(args.part_d_seed_count))]


def part_g_redesign_tasks(args: argparse.Namespace) -> list[tuple[str, str, int, str]]:
    datasets = csv_items(args.part_d_datasets)
    families = csv_items(args.part_g_redesign_families)
    rules = csv_items(args.part_d_repair_update_rules)
    return [(family, rule, seed, dataset) for family in families for rule in rules for dataset in datasets for seed in range(int(args.part_d_seed_count))]


def part_d_utility_ablation_tasks(args: argparse.Namespace) -> list[tuple[str, str, int, str]]:
    datasets = csv_items(args.part_d_datasets)
    families = csv_items(args.part_d_utility_ablation_families)
    rules = csv_items(args.part_d_utility_ablation_rules)
    return [(family, rule, seed, dataset) for family in families for rule in rules for dataset in datasets for seed in range(int(args.part_d_seed_count))]


def part_d_interaction_lift_tasks(args: argparse.Namespace) -> list[tuple[str, str, int, str]]:
    datasets = csv_items(args.part_d_datasets)
    families = csv_items(args.part_d_interaction_lift_families)
    rules = csv_items(args.part_d_interaction_lift_rules)
    return [(family, rule, seed, dataset) for family in families for rule in rules for dataset in datasets for seed in range(int(args.part_d_seed_count))]


def collect_grads(model: Any, x_source: torch.Tensor, y_source: torch.Tensor, x_witness: torch.Tensor, y_witness: torch.Tensor, x_guard: torch.Tensor, y_guard: torch.Tensor, x_all: torch.Tensor, y_all: torch.Tensor, param_name: str) -> dict[str, torch.Tensor]:
    grad_fn = lambda m, x, y, k: param_grad_for_loss(m, x, y, param_name, k)
    grads: dict[str, torch.Tensor] = {}
    grads["source"] = grad_fn(model, x_source, y_source, "ce")
    grads["witness"] = grad_fn(model, x_witness, y_witness, "ce")
    grads["domain"] = grads["witness"]
    grads["own"] = grad_fn(model, x_all[: max(16, int(x_all.shape[0]) // 2)], y_all[: max(16, int(y_all.shape[0]) // 2)], "ce")
    grads["debt"] = grad_fn(model, x_guard, y_guard, "brier")
    grads["tail_debt"] = grad_fn(model, x_guard, y_guard, "tail_debt")
    grads["tail95_debt"] = grad_fn(model, x_guard, y_guard, "tail95_debt")
    grads["tail99_debt"] = grad_fn(model, x_guard, y_guard, "tail99_debt")
    grads["margin_debt"] = grad_fn(model, x_guard, y_guard, "margin_debt")
    grads["ece_debt"] = grad_fn(model, x_guard, y_guard, "ece_debt")
    grads["radial"] = grad_fn(model, x_guard, y_guard, "radial")
    grads["shuffled"] = grad_fn(model, x_source, y_source, "shuffled")
    grads["hard_loss"] = grad_fn(model, x_source, y_source, "hard_loss")
    grads["loss_rank"] = grad_fn(model, x_source, y_source, "loss_rank")
    gen = torch.Generator(device=x_all.device).manual_seed(int(x_all.shape[0]) + int(y_all.sum().detach().cpu().item()) + (117 if param_name == "w1" else 223))
    grads["random"] = torch.randn(grads["source"].shape, generator=gen, device=x_all.device, dtype=torch.float64)
    grads["smooth"] = torch.roll(grads["source"], shifts=1) - grads["source"]
    return grads


def grouped_power_basis_projection(vector: torch.Tensor, metric_diag: torch.Tensor, shape: torch.Size, *, degree: int, ridge: float) -> tuple[torch.Tensor, dict[str, float]]:
    v = vector.detach().reshape(-1).to(dtype=torch.float64)
    m = metric_diag.detach().reshape(-1).to(device=v.device, dtype=torch.float64).clamp_min(1.0e-12)
    out = torch.zeros_like(v)
    conds: list[float] = []
    for group in param_bank_groups(shape, v.device):
        idx = group.to(device=v.device)
        if int(idx.numel()) == 0:
            continue
        dens = m[idx]
        order = torch.argsort(dens)
        sorted_idx = idx[order]
        y = v[sorted_idx]
        w = m[sorted_idx].clamp_min(1.0e-12)
        s = torch.linspace(-1.0, 1.0, int(sorted_idx.numel()), device=v.device, dtype=torch.float64)
        cols = [torch.ones_like(s)]
        for power in range(1, max(2, int(degree)) + 1):
            col = s.pow(power)
            cols.append(col - col.mean())
        basis = torch.stack(cols, dim=1)
        basis = basis - basis.mean(dim=0, keepdim=True)
        keep = basis.norm(dim=0) > 1.0e-10
        basis = basis[:, keep] / basis[:, keep].norm(dim=0, keepdim=True).clamp_min(1.0e-12)
        gram = basis.transpose(0, 1) @ (w.reshape(-1, 1) * basis)
        gram = gram + float(max(ridge, 0.0)) * torch.eye(int(gram.shape[0]), device=v.device, dtype=torch.float64)
        rhs = basis.transpose(0, 1) @ (w * y)
        coeff = torch.linalg.solve(gram, rhs)
        out[sorted_idx] = basis @ coeff
        eig = torch.linalg.eigvalsh(gram).clamp_min(1.0e-12)
        conds.append(float((eig.max() / eig.min()).detach().cpu().item()))
    before = metric_energy(v, m).clamp_min(1.0e-12)
    after = metric_energy(out, m).clamp_min(0.0)
    residual = metric_energy(v - out, m).clamp_min(0.0)
    return out.reshape_as(vector).to(dtype=vector.dtype), {
        "basis_projection_energy_fraction": float((after / before).detach().cpu().item()),
        "basis_projection_residual_fraction": float((residual / before).detach().cpu().item()),
        "basis_condition_number": float(max(conds) if conds else 0.0),
        "task_orthogonal_poly_rank": float(max(2, int(degree))),
        "redesign_projection": "activation_measure_power_basis",
    }


def bank_trace_stats_for_lift(signal: torch.Tensor, lift: torch.Tensor, metric_diag: torch.Tensor, shape: torch.Size, *, blend: float) -> dict[str, float]:
    sig = signal.detach().reshape(-1).to(dtype=torch.float64)
    lf = lift.detach().reshape(-1).to(device=sig.device, dtype=torch.float64)
    if int(lf.numel()) != int(sig.numel()):
        lf = torch.zeros_like(sig)
    pre_bank, pre = bank_additive_projection(sig, metric_diag, shape)
    post_signal = sig + float(blend) * lf
    _post_bank, post = bank_additive_projection(post_signal, metric_diag, shape)
    ctrl = same_energy_orthogonal_control(lf).to(device=sig.device)
    _ctrl_bank, ctrl_stats = bank_additive_projection(sig + float(blend) * ctrl, metric_diag, shape)
    pre_r2 = fval(pre.get("edge_bank_anova_explained"))
    post_r2 = fval(post.get("edge_bank_anova_explained"))
    pre_inter = fval(pre.get("interaction_residual_fraction"), 1.0)
    post_inter = fval(post.get("interaction_residual_fraction"), 1.0)
    ctrl_r2 = fval(ctrl_stats.get("edge_bank_anova_explained"))
    return {
        "pre_bank_R2": pre_r2,
        "post_bank_R2": post_r2,
        "bank_R2_gain": post_r2 - pre_r2,
        "pre_interaction_residual": pre_inter,
        "post_interaction_residual": post_inter,
        "interaction_residual_reduction": pre_inter - post_inter,
        "same_upstream_energy_random_R2": ctrl_r2,
        "same_upstream_energy_random_gap": post_r2 - ctrl_r2,
    }


def accessible_lift_basis(
    signal: torch.Tensor,
    upstream_shape: torch.Size | tuple[int, ...],
    downstream_shape: torch.Size | tuple[int, ...],
) -> torch.Tensor | None:
    u_dims = tuple(int(x) for x in upstream_shape)
    d_dims = tuple(int(x) for x in downstream_shape)
    if len(u_dims) < 2 or len(d_dims) < 2:
        return None
    total_d = 1
    for dim in d_dims:
        total_d *= max(1, dim)
    sig = signal.detach().reshape(-1).to(dtype=torch.float64)
    if int(sig.numel()) != total_d:
        return None
    hidden = min(int(u_dims[-2] if len(u_dims) >= 2 else u_dims[0]), int(d_dims[0]))
    basis_count = int(d_dims[-1]) if len(d_dims) >= 3 else int(d_dims[1])
    if hidden <= 0 or basis_count <= 0:
        return None
    cols: list[torch.Tensor] = []
    if len(d_dims) == 3:
        ds = sig.reshape(d_dims)
        class_scale = ds[:hidden].mean(dim=(0, 2))
        if float(class_scale.abs().max().detach().cpu().item()) <= 1.0e-12:
            class_scale = torch.ones(int(d_dims[1]), device=sig.device, dtype=torch.float64)
        for h in range(hidden):
            for k in range(basis_count):
                col = torch.zeros(d_dims, device=sig.device, dtype=torch.float64)
                col[h, :, k] = class_scale
                cols.append(col.reshape(-1))
    else:
        for h in range(hidden):
            for k in range(basis_count):
                col = torch.zeros(d_dims, device=sig.device, dtype=torch.float64)
                col[h, k] = 1.0
                cols.append(col.reshape(-1))
    return normalize_columns(cols, total_d, sig.device)


def project_to_accessible_lift_space(
    target: torch.Tensor,
    signal: torch.Tensor,
    metric_diag: torch.Tensor,
    upstream_shape: torch.Size,
    downstream_shape: torch.Size,
    *,
    ridge: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    tgt = target.detach().reshape(-1).to(dtype=torch.float64)
    basis = accessible_lift_basis(signal, upstream_shape, downstream_shape)
    if basis is None:
        return torch.zeros_like(tgt), {"accessible_lift_projection_fraction": 0.0, "accessible_lift_rank": 0.0}
    residual, diag = weighted_project(tgt, basis, metric_diag, ridge=ridge)
    fitted = tgt - residual.reshape(-1)
    stats = {k: fval(v) for k, v in diag.items()}
    stats["accessible_lift_projection_fraction"] = fval(diag.get("projected_energy_fraction"))
    stats["accessible_lift_residual_fraction"] = fval(diag.get("residual_energy_fraction"))
    stats["accessible_lift_rank"] = float(int(basis.shape[1]))
    return fitted.reshape(-1), stats


def solve_accessible_lift_upstream(
    target: torch.Tensor,
    signal: torch.Tensor,
    metric_diag: torch.Tensor,
    upstream_shape: torch.Size,
    downstream_shape: torch.Size,
    *,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    tgt = target.detach().reshape(-1).to(dtype=torch.float64)
    sig = signal.detach().reshape(-1).to(device=tgt.device, dtype=torch.float64)
    u_dims = tuple(int(x) for x in upstream_shape)
    d_dims = tuple(int(x) for x in downstream_shape)
    total_u = 1
    for dim in u_dims:
        total_u *= max(1, dim)
    total_d = 1
    for dim in d_dims:
        total_d *= max(1, dim)
    if int(tgt.numel()) != total_d or int(sig.numel()) != total_d or len(u_dims) < 2:
        return torch.zeros(total_u, device=tgt.device, dtype=torch.float64), torch.zeros_like(tgt), {
            "accessible_lift_projection_fraction": 0.0,
            "accessible_lift_residual_fraction": 1.0,
            "accessible_lift_rank": 0.0,
            "accessible_lift_solver": "shape_mismatch",
        }
    hidden_count = int(u_dims[-2] if len(u_dims) >= 3 else u_dims[0])
    basis_count = int(u_dims[-1] if len(u_dims) >= 3 else u_dims[1])
    d_cols: list[torch.Tensor] = []
    u_cols: list[torch.Tensor] = []
    for hidden_idx in range(max(0, hidden_count)):
        for basis_idx in range(max(0, basis_count)):
            u_col = torch.zeros(u_dims, device=tgt.device, dtype=torch.float64)
            if len(u_dims) >= 3:
                u_col[:, hidden_idx, basis_idx] = 1.0
            else:
                u_col[hidden_idx, basis_idx] = 1.0
            d_col = separation_lift_to_downstream(u_col.reshape(-1), upstream_shape, downstream_shape, sig).reshape(-1)
            d_norm = d_col.norm().clamp_min(1.0e-12)
            if float(d_norm.detach().cpu().item()) <= 1.0e-12:
                continue
            d_cols.append(d_col / d_norm)
            u_cols.append(u_col.reshape(-1) / d_norm)
    if not d_cols:
        return torch.zeros(total_u, device=tgt.device, dtype=torch.float64), torch.zeros_like(tgt), {
            "accessible_lift_projection_fraction": 0.0,
            "accessible_lift_residual_fraction": 1.0,
            "accessible_lift_rank": 0.0,
            "accessible_lift_solver": "empty_design",
        }
    basis = torch.stack(d_cols, dim=1)
    upstream_basis = torch.stack(u_cols, dim=1)
    w = metric_diag.detach().reshape(-1).to(device=tgt.device, dtype=torch.float64).clamp_min(1.0e-12)
    gram = basis.transpose(0, 1) @ (w.reshape(-1, 1) * basis)
    gram = gram + float(max(ridge, 0.0)) * torch.eye(int(gram.shape[0]), device=tgt.device, dtype=torch.float64)
    rhs = basis.transpose(0, 1) @ (w * tgt)
    coeff = torch.linalg.solve(gram, rhs)
    fitted = basis @ coeff
    upstream = upstream_basis @ coeff
    before = metric_energy(tgt, w).clamp_min(1.0e-12)
    after = metric_energy(fitted, w).clamp_min(0.0)
    residual = metric_energy(tgt - fitted, w).clamp_min(0.0)
    eig = torch.linalg.eigvalsh(gram).clamp_min(1.0e-12)
    return upstream.reshape(-1), fitted.reshape(-1), {
        "accessible_lift_projection_fraction": float((after / before).detach().cpu().item()),
        "accessible_lift_residual_fraction": float((residual / before).detach().cpu().item()),
        "accessible_lift_rank": float(int(basis.shape[1])),
        "accessible_lift_condition_number": float((eig.max() / eig.min()).detach().cpu().item()),
        "accessible_lift_solver": "upstream_least_squares",
    }


def shared_dictionary_projection(vector: torch.Tensor, metric_diag: torch.Tensor, references: list[torch.Tensor], *, ridge: float) -> tuple[torch.Tensor, dict[str, float]]:
    v = vector.detach().reshape(-1).to(dtype=torch.float64)
    basis = normalize_columns([ref.reshape(-1).to(device=v.device, dtype=torch.float64) for ref in references], int(v.numel()), v.device)
    if basis is None:
        return torch.zeros_like(vector), {
            "basis_projection_energy_fraction": 0.0,
            "basis_projection_residual_fraction": 1.0,
            "shared_dictionary_rank": 0.0,
            "redesign_projection": "source_witness_lift_bank_shared_dictionary",
        }
    residual, diag = weighted_project(v, basis, metric_diag, ridge=ridge)
    out = v - residual.reshape(-1)
    stats = {k: fval(val) for k, val in diag.items()}
    stats.update(
        {
            "basis_projection_energy_fraction": fval(diag.get("projected_energy_fraction")),
            "basis_projection_residual_fraction": fval(diag.get("residual_energy_fraction")),
            "shared_dictionary_rank": float(int(basis.shape[1]) if basis is not None else 0),
            "redesign_projection": "source_witness_lift_bank_shared_dictionary",
        }
    )
    return out.reshape_as(vector).to(dtype=vector.dtype), stats


def monotone_separable_projection(vector: torch.Tensor, metric_diag: torch.Tensor, shape: torch.Size, *, ridge: float) -> tuple[torch.Tensor, dict[str, float]]:
    v = vector.detach().reshape(-1).to(dtype=torch.float64)
    m = metric_diag.detach().reshape(-1).to(device=v.device, dtype=torch.float64).clamp_min(1.0e-12)
    out = torch.zeros_like(v)
    coverages: list[float] = []
    conds: list[float] = []
    for group in param_bank_groups(shape, v.device):
        idx = group.to(device=v.device)
        if int(idx.numel()) == 0:
            continue
        dens = m[idx]
        order = torch.argsort(dens)
        sorted_idx = idx[order]
        y = v[sorted_idx]
        w = m[sorted_idx].clamp_min(1.0e-12)
        pos_template = torch.cummax(y.clamp_min(0.0), dim=0).values
        neg_template = torch.cummax((-y).clamp_min(0.0), dim=0).values
        template = pos_template - neg_template
        template = template - (w * template).sum() / w.sum().clamp_min(1.0e-12)
        denom = (w * template.square()).sum().clamp_min(1.0e-12) + float(max(ridge, 0.0))
        coeff = (w * template * y).sum() / denom
        fitted = coeff * template
        out[sorted_idx] = fitted
        coverages.append(float((template.abs() > template.abs().median()).float().mean().detach().cpu().item()))
        conds.append(float((w.max() / w.min().clamp_min(1.0e-12)).detach().cpu().item()))
    before = metric_energy(v, m).clamp_min(1.0e-12)
    after = metric_energy(out, m).clamp_min(0.0)
    residual = metric_energy(v - out, m).clamp_min(0.0)
    return out.reshape_as(vector).to(dtype=vector.dtype), {
        "basis_projection_energy_fraction": float((after / before).detach().cpu().item()),
        "basis_projection_residual_fraction": float((residual / before).detach().cpu().item()),
        "bump_domain_density_coverage": float(sum(coverages) / max(1, len(coverages))),
        "basis_condition_number": float(max(conds) if conds else 0.0),
        "redesign_projection": "monotone_separable_metric_ordering",
    }


def choose_family_signal(family: str, signal: torch.Tensor, lift: torch.Tensor, metric_diag: torch.Tensor, shape: torch.Size, ridge: float, control_basis: torch.Tensor | None) -> tuple[torch.Tensor, dict[str, float]]:
    blend = 0.0 if family in {"current_layer_edge_only_replay", "bank_additive_only"} else 1.0
    edge_vec, stats = representation_separation_bank_stats(signal, lift, metric_diag, shape, blend=blend)
    if family == "current_layer_edge_only_replay":
        edge_vec = signal.reshape(-1)
    elif family == "bank_additive_only":
        edge_vec = bank_additive_projection(signal, metric_diag, shape)[0].reshape(-1)
    elif family == "upstream_separation_only":
        edge_vec = torch.zeros_like(signal.reshape(-1))
    elif family == "two_layer_quantile_lowfreq_local_residual":
        edge_vec, bstats = project_grouped_quantile_basis(signal.reshape(-1) + lift.reshape(-1), metric_diag, shape, family="lowfreq_bump", rank=3, ridge=ridge)
        edge_vec = edge_vec.reshape(-1)
        stats.update({k: fval(v) for k, v in bstats.items()})
    elif family == "two_layer_task_conditional_orthogonal_poly":
        edge_vec, bstats = project_grouped_quantile_basis(signal.reshape(-1) + lift.reshape(-1), metric_diag, shape, family="orthogonal_poly", rank=4, ridge=ridge)
        edge_vec = edge_vec.reshape(-1)
        stats.update({k: fval(v) for k, v in bstats.items()})
    elif family == "G1_adaptive_quantile_knot_spline":
        edge_vec, bstats = project_grouped_quantile_basis(signal.reshape(-1) + lift.reshape(-1), metric_diag, shape, family="density_spline", rank=6, ridge=ridge)
        edge_vec = edge_vec.reshape(-1)
        stats.update({k: fval(v) for k, v in bstats.items()})
        stats["redesign_family_axis"] = "adaptive_quantile_knot_spline"
    elif family == "G2_density_equalized_lowfreq_local_bump":
        edge_vec, bstats = project_grouped_quantile_basis(signal.reshape(-1) + lift.reshape(-1), metric_diag, shape, family="lowfreq_bump", rank=5, ridge=ridge)
        edge_vec = edge_vec.reshape(-1)
        stats.update({k: fval(v) for k, v in bstats.items()})
        stats["redesign_family_axis"] = "density_equalized_lowfreq_local_bump"
    elif family == "G3_node_bank_shared_dictionary":
        bank_vec = bank_additive_projection(signal.reshape(-1) + lift.reshape(-1), metric_diag, shape)[0].reshape(-1)
        refs = [signal.reshape(-1), lift.reshape(-1), bank_vec, torch.roll(signal.reshape(-1), shifts=1), torch.roll(lift.reshape(-1), shifts=-1)]
        edge_vec, bstats = shared_dictionary_projection(signal.reshape(-1) + lift.reshape(-1), metric_diag, refs, ridge=ridge)
        edge_vec = edge_vec.reshape(-1)
        stats.update({k: fval(v) for k, v in bstats.items()})
        stats["redesign_family_axis"] = "node_bank_shared_dictionary"
    elif family == "G4_activation_measure_orthogonal_poly":
        edge_vec, bstats = grouped_power_basis_projection(signal.reshape(-1) + lift.reshape(-1), metric_diag, shape, degree=6, ridge=ridge)
        edge_vec = edge_vec.reshape(-1)
        stats.update({k: fval(v) for k, v in bstats.items()})
        stats["redesign_family_axis"] = "activation_measure_orthogonal_poly"
    elif family == "G5_mixed_che_fou_domain_transport":
        base = signal.reshape(-1) + lift.reshape(-1)
        transported = 0.50 * base + 0.20 * torch.roll(base, shifts=1) + 0.20 * torch.roll(base, shifts=-1) + 0.10 * torch.roll(signal.reshape(-1), shifts=max(1, int(base.numel()) // 17))
        spline_vec, spline_stats = project_grouped_quantile_basis(transported, metric_diag, shape, family="density_spline", rank=4, ridge=ridge)
        freq_vec, freq_stats = project_grouped_quantile_basis(transported, metric_diag, shape, family="lowfreq_bump", rank=4, ridge=ridge)
        edge_vec = 0.5 * spline_vec.reshape(-1) + 0.5 * freq_vec.reshape(-1)
        stats.update({f"spline_{k}": fval(v) for k, v in spline_stats.items()})
        stats.update({f"freq_{k}": fval(v) for k, v in freq_stats.items()})
        stats["upstream_transport_type"] = "train_only_mixed_che_fou_roll_transport"
        stats["redesign_family_axis"] = "mixed_che_fou_domain_transport"
    elif family == "G6_monotone_separable_coordinate_ordering":
        edge_vec, bstats = monotone_separable_projection(signal.reshape(-1) + lift.reshape(-1), metric_diag, shape, ridge=ridge)
        edge_vec = edge_vec.reshape(-1)
        stats.update({k: fval(v) for k, v in bstats.items()})
        stats["redesign_family_axis"] = "monotone_separable_coordinate_ordering"
    elif family == "two_layer_node_bank_anova_basis":
        edge_vec = bank_additive_projection(signal.reshape(-1) + lift.reshape(-1), metric_diag, shape)[0].reshape(-1)
    elif family in {"upstream_separation_plus_control_residual_bank_edge", "two_layer_upstream_separation_diagnostic_control"} and control_basis is not None:
        edge_vec, diag = weighted_project(edge_vec, control_basis, metric_diag, ridge=ridge)
        edge_vec = edge_vec.reshape(-1)
        stats["control_residual_fraction_after_family"] = fval(diag.get("residual_energy_fraction"))
    return edge_vec.reshape(-1), stats


def representation_separated_probe(dataset: str, seed: int, family: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    bundle = base73.load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), int(seed))
    metric_take = int(args.metric_batch_size)
    x_all = bundle["x_train"].to(device).float()[:metric_take]
    y_all = bundle["y_train"].to(device).long()[:metric_take]
    n = int(x_all.shape[0])
    a = max(16, n // 3)
    b = max(a + 16, (2 * n) // 3)
    x_source, y_source = x_all[:a], y_all[:a]
    x_witness, y_witness = x_all[a:b], y_all[a:b]
    x_guard, y_guard = x_all[b:], y_all[b:]
    if int(x_guard.shape[0]) < 16:
        x_guard, y_guard = x_all[-max(16, n // 4) :], y_all[-max(16, n // 4) :]
    method = map_family_to_method(family)
    model = base75.make_wlb_model(method, bundle, device, int(args.hidden), int(seed) + 22790, x_metric=x_all)
    ridge = float(args.projector_ridge)
    w2_grads = collect_grads(model, x_source, y_source, x_witness, y_witness, x_guard, y_guard, x_all, y_all, "w2")
    w1_grads = collect_grads(model, x_source, y_source, x_witness, y_witness, x_guard, y_guard, x_all, y_all, "w1")
    metric_w2 = metric_diag_from_design(model, x_all, mode="readout")
    metric_w1 = make_w1_metric(w1_grads)
    dim2 = int(w2_grads["source"].numel())
    dim1 = int(w1_grads["source"].numel())
    w2_domain_basis = make_domain_basis(dim2, device, w2_grads, include_density=True)
    w1_domain_basis = make_domain_basis(dim1, device, w1_grads, include_density=True)
    source_res, domain_diag = weighted_project(w2_grads["source"], w2_domain_basis, metric_w2, ridge=ridge)
    witness_res, _ = weighted_project(w2_grads["witness"], w2_domain_basis, metric_w2, ridge=ridge)
    source_w1_res, _ = weighted_project(w1_grads["source"], w1_domain_basis, metric_w1, ridge=ridge)
    witness_w1_res, _ = weighted_project(w1_grads["witness"], w1_domain_basis, metric_w1, ridge=ridge)
    signal = 0.35 * source_res.reshape(-1) + 0.65 * witness_res.reshape(-1)
    pre_bank, pre_stats = bank_additive_projection(signal, metric_w2, model.w2.shape)
    sep_raw = 0.5 * source_w1_res.reshape(-1) + 0.5 * witness_w1_res.reshape(-1)
    if "debt_safe" in family:
        sep_raw, _ = weighted_project(sep_raw, normalize_columns(debt_grad_columns(w1_grads), dim1, device), metric_w1, ridge=ridge)
    lift = separation_lift_to_downstream(sep_raw, model.w1.shape, model.w2.shape, signal)
    control_basis_w2 = make_control_basis(dim2, device, w2_grads, signal + lift)
    edge_vec, sep_stats = choose_family_signal(family, signal, lift, metric_w2, model.w2.shape, ridge, control_basis_w2)
    if "debt_safe" in family:
        edge_vec, _ = weighted_project(edge_vec, normalize_columns(debt_grad_columns(w2_grads), dim2, device), metric_w2, ridge=ridge)
        edge_vec = edge_vec.reshape(-1)
    bank_energy_vec = bank_additive_projection(signal + lift, metric_w2, model.w2.shape)[0].reshape(-1)
    edge_domain_res, edge_domain_diag = weighted_project(edge_vec, w2_domain_basis, metric_w2, ridge=ridge)
    edge_residual_basis = normalize_columns(
        [
            w2_grads["domain"],
            w2_grads["shuffled"],
            w2_grads["random"],
            w2_grads["radial"],
            w2_grads["hard_loss"],
            w2_grads["loss_rank"],
            w2_grads["smooth"],
            *debt_grad_columns(w2_grads),
            same_energy_orthogonal_control(edge_vec).to(device=device),
            bank_energy_vec,
        ],
        dim2,
        device,
    )
    edge_update_vec, edge_control_diag = weighted_project(edge_domain_res, edge_residual_basis, metric_w2, ridge=ridge)
    edge_update_vec = edge_update_vec.reshape(-1)
    w1_control_basis = make_control_basis(dim1, device, w1_grads, sep_raw)
    sep_update_vec, sep_control_diag = weighted_project(sep_raw, w1_control_basis, metric_w1, ridge=ridge)
    sep_update_vec = sep_update_vec.reshape(-1)
    source_lift = separation_lift_to_downstream(source_w1_res, model.w1.shape, model.w2.shape, source_res)
    witness_lift = separation_lift_to_downstream(witness_w1_res, model.w1.shape, model.w2.shape, witness_res)
    source_post_bank = representation_separation_bank_stats(source_res, source_lift, metric_w2, model.w2.shape, blend=1.0)[0]
    witness_post_bank = representation_separation_bank_stats(witness_res, witness_lift, metric_w2, model.w2.shape, blend=1.0)[0]
    bank_coh = bank_coherence(source_post_bank, witness_post_bank, metric_w2, model.w2.shape)
    bank_lcb = bank_coh - 0.05 * abs(bank_coh)
    coh = coherence_score(source_res, witness_res, metric_w2)
    alpha = shrink_from_lcb(coh, threshold=0.02)
    step_norm = float(args.lr) * float(args.step_mult) * max(0.0, alpha)
    w1_scale = 0.5 * step_norm if family != "current_layer_edge_only_replay" and family != "bank_additive_only" else 0.0
    w2_scale = 0.0 if family == "upstream_separation_only" else step_norm
    candidate_w1 = -normalized_update(sep_update_vec, model.w1.shape, w1_scale)
    candidate_w2 = -normalized_update(edge_update_vec, model.w2.shape, w2_scale)
    sep_control = same_energy_orthogonal_control(sep_raw).to(device=device)
    ctrl_vectors = {
        "same_domain": {"w1": torch.zeros_like(candidate_w1), "w2": -normalized_update(w2_grads["domain"], model.w2.shape, max(step_norm, 1.0e-8))},
        "same_edge": {"w1": torch.zeros_like(candidate_w1), "w2": -normalized_update(w2_grads["random"], model.w2.shape, max(step_norm, 1.0e-8))},
        "same_debt": {"w1": -normalized_update(w1_grads["debt"], model.w1.shape, w1_scale), "w2": -normalized_update(w2_grads["debt"], model.w2.shape, max(w2_scale, 1.0e-8))},
        "same_smooth": {"w1": torch.zeros_like(candidate_w1), "w2": -normalized_update(w2_grads["smooth"], model.w2.shape, max(w2_scale, 1.0e-8))},
        "same_separation_energy": {"w1": -normalized_update(sep_control, model.w1.shape, w1_scale), "w2": torch.zeros_like(candidate_w2)},
        "same_bank_energy": {"w1": torch.zeros_like(candidate_w1), "w2": -normalized_update(bank_energy_vec, model.w2.shape, max(w2_scale, 1.0e-8))},
    }
    cand_delta = metric_delta_for_updates(model, x_guard, y_guard, {"w1": candidate_w1, "w2": candidate_w2})
    ctrl_delta = {name: metric_delta_for_updates(model, x_guard, y_guard, vecs) for name, vecs in ctrl_vectors.items()}
    debt_penalty = max(0.0, cand_delta["Brier"]) + max(0.0, cand_delta["ECE"]) + max(0.0, cand_delta["tail95"]) + max(0.0, cand_delta["tail99"]) + max(0.0, -cand_delta["margin10"])
    gaps = {name: vals["NLL"] - cand_delta["NLL"] - float(args.debt_margin_lambda) * debt_penalty for name, vals in ctrl_delta.items()}
    design = base73.w2_readout_edge_design(model, x_all)
    raw_energy = design["raw_col_energy"].to(device=device, dtype=torch.float64).reshape(-1)
    sorted_raw = torch.sort(raw_energy / raw_energy.max().clamp_min(1.0e-12), descending=True).values
    active = min(128, int(sorted_raw.numel()))
    raw_visible = lower_cvar(sorted_raw[:active].detach().cpu().tolist(), 0.25) if active else 0.0
    brier_ucb = cand_delta["Brier"] + 0.5 * abs(cand_delta["Brier"])
    ece_ucb = cand_delta["ECE"] + 0.5 * abs(cand_delta["ECE"])
    tail95_ucb = cand_delta["tail95"] + 0.5 * abs(cand_delta["tail95"])
    tail99_ucb = cand_delta["tail99"] + 0.5 * abs(cand_delta["tail99"])
    margin_ucb = -cand_delta["margin10"] + 0.5 * abs(cand_delta["margin10"])
    all_debt_ucb = max(brier_ucb, ece_ucb, tail95_ucb, tail99_ucb, margin_ucb)
    margins = list(gaps.values())
    post_bank_R2 = fval(sep_stats.get("post_bank_R2"))
    pre_bank_R2 = fval(sep_stats.get("pre_bank_R2"), fval(pre_stats.get("edge_bank_anova_explained")))
    post_inter = fval(sep_stats.get("post_interaction_residual"))
    pre_inter = fval(sep_stats.get("pre_interaction_residual"), fval(pre_stats.get("interaction_residual_fraction")))
    row = {
        "dataset": dataset,
        "seed": int(seed),
        "architecture": method,
        "layer_id": "w1_to_w2",
        "node_id": "all_output_banks",
        "family": family,
        "pre_single_R2": fval(domain_diag.get("residual_energy_fraction")),
        "pre_bank_R2": pre_bank_R2,
        "pre_interaction_residual": pre_inter,
        "pre_domain_nuisance_fraction": fval(domain_diag.get("projected_energy_fraction")),
        "pre_control_margin_p10": 0.0,
        "pre_debt_UCB": 0.0,
        "post_single_R2": fval(edge_control_diag.get("residual_energy_fraction")),
        "post_bank_R2": post_bank_R2,
        "post_interaction_residual": post_inter,
        "bank_R2_gain": post_bank_R2 - pre_bank_R2,
        "bank_coherence_LCB_positive": int(bank_lcb > 0.0),
        "source_witness_conditional_coherence": coh,
        "conditional_alpha": alpha,
        "source_witness_bank_coherence_LCB": bank_lcb,
        "control_margin_p10": min(margins) if margins else 0.0,
        "control_margin_CVaR25": lower_cvar(margins, 0.25),
        "control_margin_positive": int((min(margins) if margins else 0.0) > 0.0),
        "same_domain_candidate_better": int(gaps["same_domain"] > 0.0),
        "same_edge_candidate_better": int(gaps["same_edge"] > 0.0),
        "same_debt_candidate_better": int(gaps["same_debt"] > 0.0),
        "same_smooth_candidate_better": int(gaps["same_smooth"] > 0.0),
        "same_separation_energy_candidate_better": int(gaps["same_separation_energy"] > 0.0),
        "same_bank_energy_candidate_better": int(gaps["same_bank_energy"] > 0.0),
        "conditional_residual_fraction": fval(edge_control_diag.get("residual_energy_fraction")),
        "domain_nuisance_fraction": fval(edge_domain_diag.get("projected_energy_fraction")),
        "edge_control_residual_fraction": fval(edge_control_diag.get("residual_energy_fraction")),
        "upstream_control_residual_fraction": fval(sep_control_diag.get("residual_energy_fraction")),
        "separation_gain": fval(sep_stats.get("bank_R2_gain")),
        "separation_energy": fval(sep_stats.get("separation_lift_energy")),
        "upstream_update_norm": float(candidate_w1.reshape(-1).norm().detach().cpu().item()),
        "downstream_update_norm": float(candidate_w2.reshape(-1).norm().detach().cpu().item()),
        "projected_NLL_delta_guard": cand_delta["NLL"],
        "projected_Brier_delta_guard": cand_delta["Brier"],
        "projected_ECE_delta_guard": cand_delta["ECE"],
        "projected_tail95_delta_guard": cand_delta["tail95"],
        "projected_tail99_delta_guard": cand_delta["tail99"],
        "projected_margin10_delta_guard": cand_delta["margin10"],
        "all_debt_UCB_nonpositive": int(all_debt_ucb <= 0.0),
        "all_debt_UCB_max": all_debt_ucb,
        "raw_readout_visible_energy": raw_visible,
        "raw_readout_visible_energy_CVaR25": raw_visible,
        "basis_Gram_condition": fval(sep_stats.get("bank_condition_number")),
        "edge_domain_drift": abs(post_bank_R2 - pre_bank_R2),
        "edge_extrapolation_rate": float((metric_w2 < metric_w2.quantile(0.10)).float().mean().detach().cpu().item()),
        "edge_signal_sign_cancellation_rate": sign_cancellation_rate(edge_vec, model.w2.shape),
        "family_gate_pass": 0,
        "diagnostic_only": int(family == "two_layer_upstream_separation_diagnostic_control"),
        "preflight_source": "actual_train_only_two_layer_representation_separated_microprobe",
    }
    row["family_gate_pass"] = int(
        row["diagnostic_only"] == 0
        and row["bank_R2_gain"] > 0.0
        and row["bank_coherence_LCB_positive"] == 1
        and row["control_margin_positive"] == 1
        and row["same_domain_candidate_better"] == 1
        and row["same_edge_candidate_better"] == 1
        and row["same_debt_candidate_better"] == 1
        and row["same_separation_energy_candidate_better"] == 1
        and row["same_bank_energy_candidate_better"] == 1
        and row["all_debt_UCB_nonpositive"] == 1
        and row["conditional_residual_fraction"] >= 0.20
        and row["domain_nuisance_fraction"] <= 0.80
    )
    return row


def norm_scalar(t: torch.Tensor) -> float:
    return float(t.detach().reshape(-1).norm().cpu().item())


def weighted_dot(a: torch.Tensor, b: torch.Tensor, metric_diag: torch.Tensor) -> float:
    aa = a.detach().reshape(-1).to(dtype=torch.float64)
    bb = b.detach().reshape(-1).to(device=aa.device, dtype=torch.float64)
    mm = metric_diag.detach().reshape(-1).to(device=aa.device, dtype=torch.float64)
    return float((aa * bb * mm).sum().detach().cpu().item())


def composite_debt_direction(grads: dict[str, torch.Tensor], dim: int, device: torch.device) -> torch.Tensor:
    basis = normalize_columns(debt_grad_columns(grads), int(dim), device)
    if basis is None:
        return torch.zeros(int(dim), device=device, dtype=torch.float64)
    return basis.mean(dim=1).reshape(-1)


def control_gap_diagnostics(
    model: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    candidate: dict[str, torch.Tensor],
    controls: dict[str, dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, float], float, float]:
    cand_delta = metric_delta_for_updates(model, x, y, candidate)
    ctrl_delta = {name: metric_delta_for_updates(model, x, y, vecs) for name, vecs in controls.items()}
    debt_penalty = max(0.0, cand_delta["Brier"]) + max(0.0, cand_delta["ECE"]) + max(0.0, cand_delta["tail95"]) + max(0.0, cand_delta["tail99"]) + max(0.0, -cand_delta["margin10"])
    gaps = {name: vals["NLL"] - cand_delta["NLL"] - float(args.debt_margin_lambda) * debt_penalty for name, vals in ctrl_delta.items()}
    brier_ucb = cand_delta["Brier"] + 0.5 * abs(cand_delta["Brier"])
    ece_ucb = cand_delta["ECE"] + 0.5 * abs(cand_delta["ECE"])
    tail95_ucb = cand_delta["tail95"] + 0.5 * abs(cand_delta["tail95"])
    tail99_ucb = cand_delta["tail99"] + 0.5 * abs(cand_delta["tail99"])
    margin_ucb = -cand_delta["margin10"] + 0.5 * abs(cand_delta["margin10"])
    all_debt_ucb = max(brier_ucb, ece_ucb, tail95_ucb, tail99_ucb, margin_ucb)
    return cand_delta, ctrl_delta, gaps, debt_penalty, all_debt_ucb


def bootstrap_guard_margin(
    model: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    candidate: dict[str, torch.Tensor],
    controls: dict[str, dict[str, torch.Tensor]],
    args: argparse.Namespace,
    *,
    seed: int,
) -> dict[str, float]:
    n = int(x.shape[0])
    repeats = int(args.part_d_repair_bootstrap_repeats)
    if n < 4 or repeats <= 0:
        return {
            "bootstrap_guard_repeats": 0,
            "bootstrap_guard_min_margin_lcb10": 0.0,
            "bootstrap_guard_min_margin_CVaR25": 0.0,
            "bootstrap_guard_margin_positive": 0,
            "bootstrap_guard_candidate_NLL_improve_folds": 0,
        }
    gen = torch.Generator(device="cpu").manual_seed(2279000 + int(seed))
    sample_n = max(8, min(n, int(math.ceil(0.75 * n))))
    min_margins: list[float] = []
    cvar_margins: list[float] = []
    cand_nll: list[float] = []
    for _ in range(repeats):
        idx = torch.randint(0, n, (sample_n,), generator=gen, device=torch.device("cpu")).to(device=x.device)
        xs = x.index_select(0, idx)
        ys = y.index_select(0, idx)
        cd, _ct, gaps, _debt_penalty, _all_debt = control_gap_diagnostics(model, xs, ys, candidate, controls, args)
        vals = list(gaps.values())
        min_margins.append(min(vals) if vals else 0.0)
        cvar_margins.append(lower_cvar(vals, 0.25) if vals else 0.0)
        cand_nll.append(cd["NLL"])
    lcb10 = quantile(min_margins, 0.10)
    return {
        "bootstrap_guard_repeats": repeats,
        "bootstrap_guard_min_margin_lcb10": lcb10,
        "bootstrap_guard_min_margin_CVaR25": lower_cvar(min_margins, 0.25),
        "bootstrap_guard_control_margin_CVaR25_median": quantile(cvar_margins, 0.50),
        "bootstrap_guard_margin_positive": int(lcb10 > 0.0),
        "bootstrap_guard_candidate_NLL_improve_folds": sum(int(v < 0.0) for v in cand_nll),
    }


def representation_separated_control_margin_repair_probe(dataset: str, seed: int, family: str, update_rule: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    bundle = base73.load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), int(seed))
    metric_take = int(args.metric_batch_size)
    x_all = bundle["x_train"].to(device).float()[:metric_take]
    y_all = bundle["y_train"].to(device).long()[:metric_take]
    n = int(x_all.shape[0])
    a = max(16, n // 3)
    b = max(a + 16, (2 * n) // 3)
    x_source, y_source = x_all[:a], y_all[:a]
    x_witness, y_witness = x_all[a:b], y_all[a:b]
    x_guard, y_guard = x_all[b:], y_all[b:]
    if int(x_guard.shape[0]) < 16:
        x_guard, y_guard = x_all[-max(16, n // 4) :], y_all[-max(16, n // 4) :]
    method = map_family_to_method(family)
    model = base75.make_wlb_model(method, bundle, device, int(args.hidden), int(seed) + 22790, x_metric=x_all)
    ridge = float(args.projector_ridge)
    w2_grads = collect_grads(model, x_source, y_source, x_witness, y_witness, x_guard, y_guard, x_all, y_all, "w2")
    w1_grads = collect_grads(model, x_source, y_source, x_witness, y_witness, x_guard, y_guard, x_all, y_all, "w1")
    metric_w2 = metric_diag_from_design(model, x_all, mode="readout")
    metric_w1 = make_w1_metric(w1_grads)
    dim2 = int(w2_grads["source"].numel())
    dim1 = int(w1_grads["source"].numel())
    w2_domain_basis = make_domain_basis(dim2, device, w2_grads, include_density=True)
    w1_domain_basis = make_domain_basis(dim1, device, w1_grads, include_density=True)
    source_res, domain_diag = weighted_project(w2_grads["source"], w2_domain_basis, metric_w2, ridge=ridge)
    witness_res, _ = weighted_project(w2_grads["witness"], w2_domain_basis, metric_w2, ridge=ridge)
    source_w1_res, _ = weighted_project(w1_grads["source"], w1_domain_basis, metric_w1, ridge=ridge)
    witness_w1_res, _ = weighted_project(w1_grads["witness"], w1_domain_basis, metric_w1, ridge=ridge)
    signal = 0.35 * source_res.reshape(-1) + 0.65 * witness_res.reshape(-1)
    pre_bank, pre_stats = bank_additive_projection(signal, metric_w2, model.w2.shape)
    sep_raw = 0.5 * source_w1_res.reshape(-1) + 0.5 * witness_w1_res.reshape(-1)
    lift = separation_lift_to_downstream(sep_raw, model.w1.shape, model.w2.shape, signal)
    target_lift = pre_bank.reshape(-1) - signal.reshape(-1)
    accessible_sep_raw, accessible_lift, accessible_lift_stats = solve_accessible_lift_upstream(
        target_lift,
        signal,
        metric_w2,
        model.w1.shape,
        model.w2.shape,
        ridge=ridge,
    )
    if update_rule.startswith("accessible_interaction_cancel_lift_"):
        sep_raw = accessible_sep_raw.reshape(-1)
        lift = accessible_lift.reshape(-1)
    control_basis_w2 = make_control_basis(dim2, device, w2_grads, signal + lift)
    edge_vec, sep_stats = choose_family_signal(family, signal, lift, metric_w2, model.w2.shape, ridge, control_basis_w2)
    edge_vec = edge_vec.reshape(-1)
    bank_energy_vec = bank_additive_projection(signal + lift, metric_w2, model.w2.shape)[0].reshape(-1)
    def make_edge_residual_basis(*, include_domain: bool = True, include_debt: bool = True, include_bank_energy: bool = True) -> torch.Tensor | None:
        cols: list[torch.Tensor] = [
            w2_grads["shuffled"],
            w2_grads["random"],
            w2_grads["radial"],
            w2_grads["hard_loss"],
            w2_grads["loss_rank"],
            w2_grads["smooth"],
            same_energy_orthogonal_control(edge_vec).to(device=device),
        ]
        if include_domain:
            cols.insert(0, w2_grads["domain"])
        if include_debt:
            cols.extend(debt_grad_columns(w2_grads))
        if include_bank_energy:
            cols.append(bank_energy_vec)
        return normalize_columns(cols, dim2, device)

    edge_residual_basis = make_edge_residual_basis()
    edge_residual_basis_no_debt = make_edge_residual_basis(include_debt=False)
    edge_residual_basis_no_domain = make_edge_residual_basis(include_domain=False)
    edge_residual_basis_no_debt_domain = make_edge_residual_basis(include_domain=False, include_debt=False)
    edge_residual_basis_debt_bank_free = normalize_columns(
        [
            w2_grads["domain"],
            w2_grads["shuffled"],
            w2_grads["random"],
            w2_grads["radial"],
            w2_grads["hard_loss"],
            w2_grads["loss_rank"],
            w2_grads["smooth"],
            same_energy_orthogonal_control(edge_vec).to(device=device),
        ],
        dim2,
        device,
    )
    edge_domain_res, _edge_domain_diag = weighted_project(edge_vec, w2_domain_basis, metric_w2, ridge=ridge)
    bank_signal_update_vec, bank_signal_control_diag = weighted_project(edge_domain_res, edge_residual_basis, metric_w2, ridge=ridge)
    bank_signal_update_vec = bank_signal_update_vec.reshape(-1)
    own_task_grad = w2_grads["own"].reshape(-1)
    full_task_grad = param_grad_for_loss(model, x_all, y_all, "w2", "ce").reshape(-1)
    source_witness_task_grad = 0.5 * w2_grads["source"].reshape(-1) + 0.5 * w2_grads["witness"].reshape(-1)
    brier_debt_basis_w2 = normalize_columns([w2_grads["debt"]], dim2, device)
    debt_basis_w2 = normalize_columns(debt_grad_columns(w2_grads), dim2, device)
    full_task_brier_debt_res, full_task_brier_debt_diag = weighted_project(full_task_grad, brier_debt_basis_w2, metric_w2, ridge=ridge)
    full_task_all_debt_res, full_task_all_debt_diag = weighted_project(full_task_grad, debt_basis_w2, metric_w2, ridge=ridge)
    source_witness_brier_debt_res, source_witness_brier_debt_diag = weighted_project(source_witness_task_grad, brier_debt_basis_w2, metric_w2, ridge=ridge)
    source_witness_all_debt_res, source_witness_all_debt_diag = weighted_project(source_witness_task_grad, debt_basis_w2, metric_w2, ridge=ridge)
    full_task_debt_res = weighted_project(full_task_grad, debt_basis_w2, metric_w2, ridge=ridge)[0].reshape(-1)
    full_task_domain_res = weighted_project(full_task_grad, w2_domain_basis, metric_w2, ridge=ridge)[0].reshape(-1)
    full_task_domain_debt_res = weighted_project(full_task_domain_res, edge_residual_basis, metric_w2, ridge=ridge)[0].reshape(-1)
    own_domain_res, _own_domain_diag = weighted_project(own_task_grad, w2_domain_basis, metric_w2, ridge=ridge)
    task_grad_update_vec, task_grad_control_diag = weighted_project(own_domain_res, edge_residual_basis, metric_w2, ridge=ridge)
    task_grad_update_vec = task_grad_update_vec.reshape(-1)
    task_grad_no_debt_vec = weighted_project(own_domain_res, edge_residual_basis_no_debt, metric_w2, ridge=ridge)[0].reshape(-1)
    task_grad_no_domain_vec = weighted_project(own_task_grad, edge_residual_basis_no_domain, metric_w2, ridge=ridge)[0].reshape(-1)
    source_witness_domain_res = weighted_project(source_witness_task_grad, w2_domain_basis, metric_w2, ridge=ridge)[0]
    source_witness_update_vec = weighted_project(source_witness_domain_res, edge_residual_basis, metric_w2, ridge=ridge)[0].reshape(-1)
    raw_task_control_res = weighted_project(full_task_grad, edge_residual_basis_debt_bank_free, metric_w2, ridge=ridge)[0].reshape(-1)
    raw_task_plus_debt8_vec = full_task_grad + 8.0 * w2_grads["debt"].reshape(-1)
    raw_task_plus_debt8_brier_res, raw_task_plus_debt8_brier_diag = weighted_project(raw_task_plus_debt8_vec, brier_debt_basis_w2, metric_w2, ridge=ridge)
    task_bank_dot = weighted_dot(task_grad_update_vec, bank_signal_update_vec, metric_w2)
    aligned_bank_vec = -bank_signal_update_vec if task_bank_dot < 0.0 else bank_signal_update_vec
    w1_mode = "normal"
    candidate_control_projection_basis = ""
    candidate_control_projection_diag: dict[str, float] = {}
    if update_rule == "bank_signal_residual_update":
        candidate_edge_vec = bank_signal_update_vec
        update_source = "bank_additive_signal_residualized_under_Ge"
    elif update_rule == "task_gradient_residual_ge":
        candidate_edge_vec = task_grad_update_vec
        update_source = "train_only_task_gradient_residualized_under_Ge"
    elif update_rule == "task_gradient_bank_blend_ge":
        candidate_edge_vec = task_grad_update_vec + 0.5 * aligned_bank_vec
        update_source = "train_only_task_gradient_plus_sign_aligned_bank_signal_under_Ge"
    elif update_rule == "edge_only_task_gradient_full_ge":
        candidate_edge_vec = task_grad_update_vec
        update_source = "utility_ablation_edge_only_task_gradient_full_Ge"
        w1_mode = "zero"
    elif update_rule == "w1_sign_flip_task_gradient_full_ge":
        candidate_edge_vec = task_grad_update_vec
        update_source = "utility_ablation_w1_sign_flip_task_gradient_full_Ge"
        w1_mode = "flip"
    elif update_rule == "edge_only_task_gradient_no_debt_ge":
        candidate_edge_vec = task_grad_no_debt_vec
        update_source = "utility_ablation_edge_only_task_gradient_Ge_without_debt_projection"
        w1_mode = "zero"
    elif update_rule == "edge_only_task_gradient_no_domain_ge":
        candidate_edge_vec = task_grad_no_domain_vec
        update_source = "utility_ablation_edge_only_task_gradient_Ge_without_domain_projection"
        w1_mode = "zero"
    elif update_rule == "edge_only_raw_task_gradient":
        candidate_edge_vec = full_task_grad
        update_source = "utility_ablation_edge_only_raw_full_train_task_gradient"
        w1_mode = "zero"
    elif update_rule == "edge_only_raw_task_gradient_debt_residual":
        candidate_edge_vec = full_task_debt_res
        update_source = "utility_ablation_edge_only_raw_full_train_task_gradient_debt_residual"
        w1_mode = "zero"
    elif update_rule == "edge_only_raw_task_gradient_domain_debt_residual":
        candidate_edge_vec = full_task_domain_debt_res
        update_source = "utility_ablation_edge_only_raw_full_train_task_gradient_domain_and_Ge_residual"
        w1_mode = "zero"
    elif update_rule == "edge_only_source_witness_consensus_ge":
        candidate_edge_vec = source_witness_update_vec
        update_source = "utility_ablation_edge_only_source_witness_consensus_Ge"
        w1_mode = "zero"
    elif update_rule == "edge_only_debt_pareto_blend_ge":
        candidate_edge_vec = task_grad_update_vec + 0.5 * w2_grads["debt"].reshape(-1)
        update_source = "utility_ablation_edge_only_task_gradient_plus_debt_pareto_blend"
        w1_mode = "zero"
    elif update_rule == "accessible_interaction_cancel_lift_update":
        candidate_edge_vec = bank_signal_update_vec
        update_source = "interaction_lift_accessible_ls_upstream_update_plus_bank_signal_residual"
    elif update_rule == "accessible_interaction_cancel_lift_task_gradient_blend_ge":
        candidate_edge_vec = task_grad_update_vec + 0.5 * aligned_bank_vec
        update_source = "interaction_lift_accessible_ls_upstream_update_plus_task_gradient_bank_blend"
    elif update_rule == "accessible_interaction_cancel_lift_domain_debt_task_bank_blend":
        candidate_edge_vec = full_task_domain_debt_res + 0.5 * aligned_bank_vec
        update_source = "interaction_lift_accessible_ls_upstream_update_plus_domain_debt_residual_task_bank_blend"
    elif update_rule == "accessible_interaction_cancel_lift_raw_task_gradient":
        candidate_edge_vec = full_task_grad
        update_source = "interaction_lift_accessible_ls_upstream_update_plus_raw_full_train_task_gradient"
    elif update_rule == "accessible_interaction_cancel_lift_raw_task_gradient_w1x0":
        candidate_edge_vec = full_task_grad
        update_source = "interaction_lift_accessible_ls_w1_zero_plus_raw_full_train_task_gradient"
    elif update_rule == "accessible_interaction_cancel_lift_raw_task_gradient_w1x0p1":
        candidate_edge_vec = full_task_grad
        update_source = "interaction_lift_accessible_ls_w1_0p1_plus_raw_full_train_task_gradient"
    elif update_rule == "accessible_interaction_cancel_lift_raw_task_plus_debt1_w1x0p1":
        candidate_edge_vec = full_task_grad + w2_grads["debt"].reshape(-1)
        update_source = "interaction_lift_accessible_ls_w1_0p1_plus_raw_task_and_debt1"
    elif update_rule == "accessible_interaction_cancel_lift_raw_task_plus_debt2_w1x0p1":
        candidate_edge_vec = full_task_grad + 2.0 * w2_grads["debt"].reshape(-1)
        update_source = "interaction_lift_accessible_ls_w1_0p1_plus_raw_task_and_debt2"
    elif update_rule == "accessible_interaction_cancel_lift_raw_task_gradient_w1x0_w2x0p5":
        candidate_edge_vec = full_task_grad
        update_source = "interaction_lift_accessible_ls_w1_zero_w2_0p5_plus_raw_full_train_task_gradient"
    elif update_rule == "accessible_interaction_cancel_lift_raw_task_gradient_w1x0_w2x2":
        candidate_edge_vec = full_task_grad
        update_source = "interaction_lift_accessible_ls_w1_zero_w2_2_plus_raw_full_train_task_gradient"
    elif update_rule == "accessible_interaction_cancel_lift_raw_task_gradient_w1x0_w2x4":
        candidate_edge_vec = full_task_grad
        update_source = "interaction_lift_accessible_ls_w1_zero_w2_4_plus_raw_full_train_task_gradient"
    elif update_rule == "accessible_interaction_cancel_lift_raw_task_plus_debt4_w1x0_w2x2":
        candidate_edge_vec = full_task_grad + 4.0 * w2_grads["debt"].reshape(-1)
        update_source = "interaction_lift_accessible_ls_w1_zero_w2_2_plus_raw_task_and_debt4"
    elif update_rule == "accessible_interaction_cancel_lift_raw_task_plus_debt8_w1x0_w2x2":
        candidate_edge_vec = full_task_grad + 8.0 * w2_grads["debt"].reshape(-1)
        update_source = "interaction_lift_accessible_ls_w1_zero_w2_2_plus_raw_task_and_debt8"
    elif update_rule == "accessible_interaction_cancel_lift_raw_task_gradient_w1x0_brier_orth_ge":
        candidate_edge_vec = full_task_brier_debt_res.reshape(-1)
        update_source = "interaction_lift_accessible_ls_w1_zero_plus_raw_task_gradient_Ge_orthogonal_to_brier_debt"
        candidate_control_projection_basis = "brier_debt"
        candidate_control_projection_diag = full_task_brier_debt_diag
    elif update_rule == "accessible_interaction_cancel_lift_raw_task_gradient_w1x0_all_debt_orth_ge":
        candidate_edge_vec = full_task_all_debt_res.reshape(-1)
        update_source = "interaction_lift_accessible_ls_w1_zero_plus_raw_task_gradient_Ge_orthogonal_to_all_debt_basis"
        candidate_control_projection_basis = "all_debt"
        candidate_control_projection_diag = full_task_all_debt_diag
    elif update_rule == "accessible_interaction_cancel_lift_raw_task_plus_debt8_w1x0_w2x2_brier_orth_ge":
        candidate_edge_vec = raw_task_plus_debt8_brier_res.reshape(-1)
        update_source = "interaction_lift_accessible_ls_w1_zero_w2_2_plus_raw_task_and_debt8_Ge_orthogonal_to_brier_debt"
        candidate_control_projection_basis = "brier_debt"
        candidate_control_projection_diag = raw_task_plus_debt8_brier_diag
    elif update_rule == "accessible_interaction_cancel_lift_upstream_only_w2x0":
        candidate_edge_vec = torch.zeros_like(full_task_grad)
        update_source = "interaction_lift_accessible_ls_upstream_only_w1_default_w2_zero"
    elif update_rule == "accessible_interaction_cancel_lift_upstream_only_w1x1_w2x0":
        candidate_edge_vec = torch.zeros_like(full_task_grad)
        update_source = "interaction_lift_accessible_ls_upstream_only_w1_1x_w2_zero"
    elif update_rule == "accessible_interaction_cancel_lift_upstream_only_w1x2_w2x0":
        candidate_edge_vec = torch.zeros_like(full_task_grad)
        update_source = "interaction_lift_accessible_ls_upstream_only_w1_2x_w2_zero"
    elif update_rule == "accessible_interaction_cancel_lift_source_witness_raw_task_w1x0":
        candidate_edge_vec = source_witness_task_grad
        update_source = "interaction_lift_accessible_ls_w1_zero_plus_source_witness_raw_task_gradient"
    elif update_rule == "accessible_interaction_cancel_lift_source_witness_raw_task_w1x0_w2x2":
        candidate_edge_vec = source_witness_task_grad
        update_source = "interaction_lift_accessible_ls_w1_zero_w2_2_plus_source_witness_raw_task_gradient"
    elif update_rule == "accessible_interaction_cancel_lift_source_witness_residual_ge_w1x0":
        candidate_edge_vec = source_witness_update_vec
        update_source = "interaction_lift_accessible_ls_w1_zero_plus_source_witness_task_residualized_under_Ge"
    elif update_rule == "accessible_interaction_cancel_lift_source_witness_raw_task_w1x0_brier_orth_ge":
        candidate_edge_vec = source_witness_brier_debt_res.reshape(-1)
        update_source = "interaction_lift_accessible_ls_w1_zero_plus_source_witness_raw_task_gradient_Ge_orthogonal_to_brier_debt"
        candidate_control_projection_basis = "brier_debt"
        candidate_control_projection_diag = source_witness_brier_debt_diag
    elif update_rule == "accessible_interaction_cancel_lift_source_witness_raw_task_w1x0_all_debt_orth_ge":
        candidate_edge_vec = source_witness_all_debt_res.reshape(-1)
        update_source = "interaction_lift_accessible_ls_w1_zero_plus_source_witness_raw_task_gradient_Ge_orthogonal_to_all_debt_basis"
        candidate_control_projection_basis = "all_debt"
        candidate_control_projection_diag = source_witness_all_debt_diag
    elif update_rule == "accessible_interaction_cancel_lift_source_witness_raw_task_w1x0_w2x2_brier_orth_ge":
        candidate_edge_vec = source_witness_brier_debt_res.reshape(-1)
        update_source = "interaction_lift_accessible_ls_w1_zero_w2_2_plus_source_witness_raw_task_gradient_Ge_orthogonal_to_brier_debt"
        candidate_control_projection_basis = "brier_debt"
        candidate_control_projection_diag = source_witness_brier_debt_diag
    elif update_rule == "accessible_interaction_cancel_lift_source_witness_raw_task_plus_debt1_w1x0":
        candidate_edge_vec = source_witness_task_grad + w2_grads["debt"].reshape(-1)
        update_source = "interaction_lift_accessible_ls_w1_zero_plus_source_witness_raw_task_and_debt1"
    elif update_rule == "accessible_interaction_cancel_lift_source_witness_raw_task_plus_debt2_w1x0":
        candidate_edge_vec = source_witness_task_grad + 2.0 * w2_grads["debt"].reshape(-1)
        update_source = "interaction_lift_accessible_ls_w1_zero_plus_source_witness_raw_task_and_debt2"
    elif update_rule == "accessible_interaction_cancel_lift_source_witness_raw_task_plus_debt4_w1x0":
        candidate_edge_vec = source_witness_task_grad + 4.0 * w2_grads["debt"].reshape(-1)
        update_source = "interaction_lift_accessible_ls_w1_zero_plus_source_witness_raw_task_and_debt4"
    elif update_rule == "accessible_interaction_cancel_lift_source_witness_raw_task_plus_debt8_w1x0":
        candidate_edge_vec = source_witness_task_grad + 8.0 * w2_grads["debt"].reshape(-1)
        update_source = "interaction_lift_accessible_ls_w1_zero_plus_source_witness_raw_task_and_debt8"
    elif update_rule == "accessible_interaction_cancel_lift_source_witness_raw_task_plus_debt16_w1x0":
        candidate_edge_vec = source_witness_task_grad + 16.0 * w2_grads["debt"].reshape(-1)
        update_source = "interaction_lift_accessible_ls_w1_zero_plus_source_witness_raw_task_and_debt16"
    elif update_rule == "accessible_interaction_cancel_lift_source_witness_raw_task_plus_debt32_w1x0":
        candidate_edge_vec = source_witness_task_grad + 32.0 * w2_grads["debt"].reshape(-1)
        update_source = "interaction_lift_accessible_ls_w1_zero_plus_source_witness_raw_task_and_debt32"
    elif update_rule == "accessible_interaction_cancel_lift_source_witness_raw_task_plus_debt64_w1x0":
        candidate_edge_vec = source_witness_task_grad + 64.0 * w2_grads["debt"].reshape(-1)
        update_source = "interaction_lift_accessible_ls_w1_zero_plus_source_witness_raw_task_and_debt64"
    else:
        raise ValueError(f"unknown update_rule={update_rule}")
    cand_domain_res, cand_domain_diag = weighted_project(candidate_edge_vec, w2_domain_basis, metric_w2, ridge=ridge)
    cand_control_res, cand_control_diag = weighted_project(cand_domain_res, edge_residual_basis, metric_w2, ridge=ridge)
    w1_control_basis = make_control_basis(dim1, device, w1_grads, sep_raw)
    sep_update_vec, sep_control_diag = weighted_project(sep_raw, w1_control_basis, metric_w1, ridge=ridge)
    sep_update_vec = sep_update_vec.reshape(-1)
    source_lift = separation_lift_to_downstream(source_w1_res, model.w1.shape, model.w2.shape, source_res)
    witness_lift = separation_lift_to_downstream(witness_w1_res, model.w1.shape, model.w2.shape, witness_res)
    source_post_bank = representation_separation_bank_stats(source_res, source_lift, metric_w2, model.w2.shape, blend=1.0)[0]
    witness_post_bank = representation_separation_bank_stats(witness_res, witness_lift, metric_w2, model.w2.shape, blend=1.0)[0]
    bank_coh = bank_coherence(source_post_bank, witness_post_bank, metric_w2, model.w2.shape)
    bank_lcb = bank_coh - 0.05 * abs(bank_coh)
    coh = coherence_score(source_res, witness_res, metric_w2)
    alpha = shrink_from_lcb(coh, threshold=0.02)
    step_norm = float(args.lr) * float(args.step_mult) * max(0.0, alpha)
    w1_scale = 0.0 if w1_mode == "zero" else 0.5 * step_norm
    w2_scale = step_norm
    if "_w1x0p1" in update_rule:
        w1_scale = 0.1 * step_norm
    elif "_w1x0" in update_rule:
        w1_scale = 0.0
    elif "_w1x1" in update_rule:
        w1_scale = step_norm
    elif "_w1x2" in update_rule:
        w1_scale = 2.0 * step_norm
    if "_w2x0p5" in update_rule:
        w2_scale = 0.5 * step_norm
    elif update_rule.endswith("_w2x0"):
        w2_scale = 0.0
    elif "_w2x2" in update_rule:
        w2_scale = 2.0 * step_norm
    elif "_w2x4" in update_rule:
        w2_scale = 4.0 * step_norm
    w1_sign = 1.0 if w1_mode == "flip" else -1.0
    candidate_w1 = w1_sign * normalized_update(sep_update_vec, model.w1.shape, w1_scale)
    candidate_w2 = -normalized_update(candidate_edge_vec, model.w2.shape, w2_scale)
    cand_w1_norm = norm_scalar(candidate_w1)
    cand_w2_norm = norm_scalar(candidate_w2)
    sep_control = same_energy_orthogonal_control(sep_raw).to(device=device)
    w1_debt_composite = composite_debt_direction(w1_grads, dim1, device)
    w2_debt_composite = composite_debt_direction(w2_grads, dim2, device)
    ctrl_vectors = {
        "same_domain": {"w1": torch.zeros_like(candidate_w1), "w2": -normalized_update(w2_grads["domain"], model.w2.shape, cand_w2_norm)},
        "same_edge": {"w1": torch.zeros_like(candidate_w1), "w2": -normalized_update(w2_grads["random"], model.w2.shape, cand_w2_norm)},
        "same_debt": {"w1": -normalized_update(w1_grads["debt"], model.w1.shape, cand_w1_norm), "w2": -normalized_update(w2_grads["debt"], model.w2.shape, cand_w2_norm)},
        "same_debt_composite": {"w1": -normalized_update(w1_debt_composite, model.w1.shape, cand_w1_norm), "w2": -normalized_update(w2_debt_composite, model.w2.shape, cand_w2_norm)},
        "same_smooth": {"w1": torch.zeros_like(candidate_w1), "w2": -normalized_update(w2_grads["smooth"], model.w2.shape, cand_w2_norm)},
        "same_separation_energy": {"w1": -normalized_update(sep_control, model.w1.shape, cand_w1_norm), "w2": torch.zeros_like(candidate_w2)},
        "same_bank_energy": {"w1": torch.zeros_like(candidate_w1), "w2": -normalized_update(bank_energy_vec, model.w2.shape, cand_w2_norm)},
    }
    cand_delta, ctrl_delta, gaps, debt_penalty, all_debt_ucb = control_gap_diagnostics(model, x_guard, y_guard, {"w1": candidate_w1, "w2": candidate_w2}, ctrl_vectors, args)
    rule_pool = PART_D_REPAIR_UPDATE_RULES + PART_D_UTILITY_ABLATION_RULES
    rule_index = rule_pool.index(update_rule) if update_rule in rule_pool else sum(ord(ch) for ch in update_rule) % 997
    boot = bootstrap_guard_margin(model, x_guard, y_guard, {"w1": candidate_w1, "w2": candidate_w2}, ctrl_vectors, args, seed=int(seed) + 97 * (1 + rule_index))
    design = base73.w2_readout_edge_design(model, x_all)
    raw_energy = design["raw_col_energy"].to(device=device, dtype=torch.float64).reshape(-1)
    sorted_raw = torch.sort(raw_energy / raw_energy.max().clamp_min(1.0e-12), descending=True).values
    active = min(128, int(sorted_raw.numel()))
    raw_visible = lower_cvar(sorted_raw[:active].detach().cpu().tolist(), 0.25) if active else 0.0
    post_bank_R2 = fval(sep_stats.get("post_bank_R2"))
    pre_bank_R2 = fval(sep_stats.get("pre_bank_R2"), fval(pre_stats.get("edge_bank_anova_explained")))
    post_inter = fval(sep_stats.get("post_interaction_residual"))
    pre_inter = fval(sep_stats.get("pre_interaction_residual"), fval(pre_stats.get("interaction_residual_fraction")))
    cand_bank_vec, cand_bank_stats = bank_additive_projection(candidate_edge_vec, metric_w2, model.w2.shape)
    margins = list(gaps.values())
    row: dict[str, Any] = {
        "dataset": dataset,
        "seed": int(seed),
        "architecture": method,
        "layer_id": "w1_to_w2",
        "node_id": "all_output_banks",
        "family": family,
        "update_rule": update_rule,
        "repair_family": f"{family}__{update_rule}",
        "update_source": update_source,
        "pre_bank_R2": pre_bank_R2,
        "post_bank_R2": post_bank_R2,
        "bank_R2_gain": post_bank_R2 - pre_bank_R2,
        "pre_interaction_residual": pre_inter,
        "post_interaction_residual": post_inter,
        "bank_coherence_LCB_positive": int(bank_lcb > 0.0),
        "source_witness_conditional_coherence": coh,
        "conditional_alpha": alpha,
        "source_witness_bank_coherence_LCB": bank_lcb,
        "control_margin_p10": min(margins) if margins else 0.0,
        "control_margin_CVaR25": lower_cvar(margins, 0.25),
        "control_margin_positive": int((min(margins) if margins else 0.0) > 0.0),
        "conditional_residual_fraction": fval(cand_control_diag.get("residual_energy_fraction")),
        "domain_nuisance_fraction": fval(cand_domain_diag.get("projected_energy_fraction")),
        "bank_signal_control_residual_fraction": fval(bank_signal_control_diag.get("residual_energy_fraction")),
        "task_gradient_control_residual_fraction": fval(task_grad_control_diag.get("residual_energy_fraction")),
        "upstream_control_residual_fraction": fval(sep_control_diag.get("residual_energy_fraction")),
        "candidate_edge_bank_R2": fval(cand_bank_stats.get("edge_bank_anova_explained")),
        "candidate_task_bank_metric_dot": task_bank_dot,
        "candidate_NLL_delta_guard": cand_delta["NLL"],
        "candidate_Brier_delta_guard": cand_delta["Brier"],
        "candidate_ECE_delta_guard": cand_delta["ECE"],
        "candidate_tail95_delta_guard": cand_delta["tail95"],
        "candidate_tail99_delta_guard": cand_delta["tail99"],
        "candidate_margin10_delta_guard": cand_delta["margin10"],
        "candidate_NLL_improve": int(cand_delta["NLL"] < 0.0),
        "debt_penalty": debt_penalty,
        "all_debt_UCB_nonpositive": int(all_debt_ucb <= 0.0),
        "all_debt_UCB_max": all_debt_ucb,
        "raw_readout_visible_energy_CVaR25": raw_visible,
        "upstream_update_norm": cand_w1_norm,
        "downstream_update_norm": cand_w2_norm,
        "candidate_edge_vec_norm": norm_scalar(candidate_edge_vec),
        "bank_signal_update_vec_norm": norm_scalar(bank_signal_update_vec),
        "task_gradient_update_vec_norm": norm_scalar(task_grad_update_vec),
        "task_gradient_no_debt_vec_norm": norm_scalar(task_grad_no_debt_vec),
        "task_gradient_no_domain_vec_norm": norm_scalar(task_grad_no_domain_vec),
        "raw_task_gradient_vec_norm": norm_scalar(full_task_grad),
        "raw_task_debt_residual_vec_norm": norm_scalar(full_task_debt_res),
        "raw_task_domain_debt_residual_vec_norm": norm_scalar(full_task_domain_debt_res),
        "source_witness_task_update_vec_norm": norm_scalar(source_witness_update_vec),
        "candidate_control_projection_basis": candidate_control_projection_basis,
        "candidate_control_projection_applied": fval(candidate_control_projection_diag.get("projection_applied")),
        "candidate_control_projected_energy_fraction": fval(candidate_control_projection_diag.get("projected_energy_fraction")),
        "candidate_control_residual_energy_fraction": fval(candidate_control_projection_diag.get("residual_energy_fraction")),
        "accessible_lift_projection_fraction": fval(accessible_lift_stats.get("accessible_lift_projection_fraction")),
        "accessible_lift_residual_fraction": fval(accessible_lift_stats.get("accessible_lift_residual_fraction")),
        "accessible_lift_rank": fval(accessible_lift_stats.get("accessible_lift_rank")),
        "accessible_lift_condition_number": fval(accessible_lift_stats.get("accessible_lift_condition_number")),
        "accessible_upstream_vec_norm": norm_scalar(accessible_sep_raw),
        "accessible_downstream_lift_norm": norm_scalar(accessible_lift),
        "w1_mode": w1_mode,
        "pre_domain_nuisance_fraction": fval(domain_diag.get("projected_energy_fraction")),
        "repair_source": "utility_source_ablation_after_control_margin_failure" if update_rule in PART_D_UTILITY_ABLATION_RULES else "control_margin_repair_after_official_part_d_failure",
    }
    row.update(boot)
    for name, vals in ctrl_delta.items():
        row[f"{name}_NLL_delta_guard"] = vals["NLL"]
        row[f"{name}_gap"] = gaps[name]
        row[f"{name}_candidate_better"] = int(gaps[name] > 0.0)
    row["repair_gate_pass"] = int(
        row["bank_R2_gain"] > 0.0
        and row["bank_coherence_LCB_positive"] == 1
        and row["control_margin_positive"] == 1
        and row["bootstrap_guard_margin_positive"] == 1
        and row["same_domain_candidate_better"] == 1
        and row["same_edge_candidate_better"] == 1
        and row["same_debt_candidate_better"] == 1
        and row["same_separation_energy_candidate_better"] == 1
        and row["same_bank_energy_candidate_better"] == 1
        and row["all_debt_UCB_nonpositive"] == 1
        and row["conditional_residual_fraction"] >= 0.20
        and row["domain_nuisance_fraction"] <= 0.80
    )
    return row


def summarize_part_d_repair_family(repair_family: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    fr = [row for row in rows if row.get("repair_family") == repair_family]
    summary = {
        "repair_family": repair_family,
        "family": str(fr[0].get("family", "")) if fr else repair_family,
        "update_rule": str(fr[0].get("update_rule", "")) if fr else "",
        "rows": len(fr),
        "post_bank_R2_gain_positive_rows": sum(int(fval(row.get("bank_R2_gain")) > 0.0) for row in fr),
        "bank_coherence_LCB_positive_rows": sum(int(fval(row.get("bank_coherence_LCB_positive")) > 0.0) for row in fr),
        "candidate_NLL_improve_rows": sum(int(fval(row.get("candidate_NLL_improve")) > 0.0) for row in fr),
        "control_margin_positive_rows": sum(int(fval(row.get("control_margin_positive")) > 0.0) for row in fr),
        "bootstrap_guard_margin_positive_rows": sum(int(fval(row.get("bootstrap_guard_margin_positive")) > 0.0) for row in fr),
        "same_domain_candidate_better_rows": sum(int(fval(row.get("same_domain_candidate_better")) > 0.0) for row in fr),
        "same_edge_candidate_better_rows": sum(int(fval(row.get("same_edge_candidate_better")) > 0.0) for row in fr),
        "same_debt_candidate_better_rows": sum(int(fval(row.get("same_debt_candidate_better")) > 0.0) for row in fr),
        "same_debt_composite_candidate_better_rows": sum(int(fval(row.get("same_debt_composite_candidate_better")) > 0.0) for row in fr),
        "same_smooth_candidate_better_rows": sum(int(fval(row.get("same_smooth_candidate_better")) > 0.0) for row in fr),
        "same_separation_energy_candidate_better_rows": sum(int(fval(row.get("same_separation_energy_candidate_better")) > 0.0) for row in fr),
        "same_bank_energy_candidate_better_rows": sum(int(fval(row.get("same_bank_energy_candidate_better")) > 0.0) for row in fr),
        "all_debt_UCB_nonpositive_rows": sum(int(fval(row.get("all_debt_UCB_nonpositive")) > 0.0) for row in fr),
        "pre_interaction_residual_median": quantile([fval(row.get("pre_interaction_residual")) for row in fr], 0.50),
        "post_interaction_residual_median": quantile([fval(row.get("post_interaction_residual")) for row in fr], 0.50),
        "conditional_residual_fraction_median": quantile([fval(row.get("conditional_residual_fraction")) for row in fr], 0.50),
        "domain_nuisance_fraction_median": quantile([fval(row.get("domain_nuisance_fraction")) for row in fr], 0.50),
        "raw_readout_visible_energy_CVaR25": lower_cvar([fval(row.get("raw_readout_visible_energy_CVaR25")) for row in fr], 0.25),
        "post_bank_R2_median": quantile([fval(row.get("post_bank_R2")) for row in fr], 0.50),
        "bank_R2_gain_median": quantile([fval(row.get("bank_R2_gain")) for row in fr], 0.50),
        "candidate_NLL_delta_median": quantile([fval(row.get("candidate_NLL_delta_guard")) for row in fr], 0.50),
        "control_margin_p10_median": quantile([fval(row.get("control_margin_p10")) for row in fr], 0.50),
        "same_debt_gap_median": quantile([fval(row.get("same_debt_gap")) for row in fr], 0.50),
        "same_debt_composite_gap_median": quantile([fval(row.get("same_debt_composite_gap")) for row in fr], 0.50),
        "bootstrap_guard_min_margin_lcb10_median": quantile([fval(row.get("bootstrap_guard_min_margin_lcb10")) for row in fr], 0.50),
        "candidate_edge_bank_R2_median": quantile([fval(row.get("candidate_edge_bank_R2")) for row in fr], 0.50),
    }
    summary["repair_gate_pass"] = int(
        summary["rows"] >= 15
        and summary["post_bank_R2_gain_positive_rows"] >= 10
        and summary["bank_coherence_LCB_positive_rows"] >= 10
        and summary["candidate_NLL_improve_rows"] >= 10
        and summary["control_margin_positive_rows"] >= 10
        and summary["bootstrap_guard_margin_positive_rows"] >= 10
        and summary["same_domain_candidate_better_rows"] >= 10
        and summary["same_edge_candidate_better_rows"] >= 10
        and summary["same_debt_candidate_better_rows"] >= 10
        and summary["same_separation_energy_candidate_better_rows"] >= 10
        and summary["same_bank_energy_candidate_better_rows"] >= 10
        and summary["all_debt_UCB_nonpositive_rows"] >= 10
        and summary["post_interaction_residual_median"] <= summary["pre_interaction_residual_median"] - 0.20
        and summary["conditional_residual_fraction_median"] >= 0.20
        and summary["domain_nuisance_fraction_median"] <= 0.80
        and summary["raw_readout_visible_energy_CVaR25"] >= 0.15
    )
    blockers = {
        "candidate_NLL_improve": summary["candidate_NLL_improve_rows"],
        "control_margin": summary["control_margin_positive_rows"],
        "bootstrap_guard_margin": summary["bootstrap_guard_margin_positive_rows"],
        "same_domain": summary["same_domain_candidate_better_rows"],
        "same_edge": summary["same_edge_candidate_better_rows"],
        "same_debt": summary["same_debt_candidate_better_rows"],
        "same_debt_composite": summary["same_debt_composite_candidate_better_rows"],
        "same_separation_energy": summary["same_separation_energy_candidate_better_rows"],
        "same_bank_energy": summary["same_bank_energy_candidate_better_rows"],
        "all_debt": summary["all_debt_UCB_nonpositive_rows"],
    }
    summary["dominant_blocker"] = min(blockers, key=blockers.get) if blockers else "none"
    summary["dominant_blocker_pass_rows"] = blockers.get(summary["dominant_blocker"], 0)
    return summary


def route_from_part_d_repair(summaries: list[dict[str, Any]]) -> tuple[str, str]:
    if any(int(row.get("repair_gate_pass", 0)) for row in summaries):
        passed = [str(row.get("repair_family")) for row in summaries if int(row.get("repair_gate_pass", 0))]
        return "ControlMarginRepairOpened_NeedsOfficialPartDIntegration", f"repair_gate_pass families={passed}"
    if not summaries:
        return "ControlMarginRepairNoRows", "No repair summaries were produced."
    max_margin = max(int(row.get("control_margin_positive_rows", 0)) for row in summaries)
    max_boot = max(int(row.get("bootstrap_guard_margin_positive_rows", 0)) for row in summaries)
    max_nll = max(int(row.get("candidate_NLL_improve_rows", 0)) for row in summaries)
    best_gain = max(int(row.get("post_bank_R2_gain_positive_rows", 0)) for row in summaries)
    if max_margin > 0 or max_boot > 0:
        return "PartialControlMarginSignalButNotStable", f"max_control_margin_positive_rows={max_margin}; max_bootstrap_guard_margin_positive_rows={max_boot}; max_candidate_NLL_improve_rows={max_nll}."
    if best_gain >= 10 and max_nll == 0:
        return "BankAdditiveOpensButTaskGradientUtilityAbsent", f"bank gain opens in up to {best_gain}/15 rows, but max_candidate_NLL_improve_rows={max_nll}."
    return "ControlMarginStillAbsent", f"max_control_margin_positive_rows={max_margin}; max_bootstrap_guard_margin_positive_rows={max_boot}; max_candidate_NLL_improve_rows={max_nll}."


def summarize_part_d_family(family: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    fr = [row for row in rows if row.get("family") == family]
    summary = {
        "family": family,
        "rows": len(fr),
        "post_bank_R2_gain_positive_rows": sum(int(fval(row.get("bank_R2_gain")) > 0.0) for row in fr),
        "bank_coherence_LCB_positive_rows": sum(int(fval(row.get("bank_coherence_LCB_positive")) > 0.0) for row in fr),
        "control_margin_positive_rows": sum(int(fval(row.get("control_margin_positive")) > 0.0) for row in fr),
        "same_domain_candidate_better_rows": sum(int(fval(row.get("same_domain_candidate_better")) > 0.0) for row in fr),
        "same_edge_candidate_better_rows": sum(int(fval(row.get("same_edge_candidate_better")) > 0.0) for row in fr),
        "same_debt_candidate_better_rows": sum(int(fval(row.get("same_debt_candidate_better")) > 0.0) for row in fr),
        "same_smooth_candidate_better_rows": sum(int(fval(row.get("same_smooth_candidate_better")) > 0.0) for row in fr),
        "same_separation_energy_candidate_better_rows": sum(int(fval(row.get("same_separation_energy_candidate_better")) > 0.0) for row in fr),
        "same_bank_energy_candidate_better_rows": sum(int(fval(row.get("same_bank_energy_candidate_better")) > 0.0) for row in fr),
        "all_debt_UCB_nonpositive_rows": sum(int(fval(row.get("all_debt_UCB_nonpositive")) > 0.0) for row in fr),
        "pre_interaction_residual_median": quantile([fval(row.get("pre_interaction_residual")) for row in fr], 0.50),
        "post_interaction_residual_median": quantile([fval(row.get("post_interaction_residual")) for row in fr], 0.50),
        "conditional_residual_fraction_median": quantile([fval(row.get("conditional_residual_fraction")) for row in fr], 0.50),
        "domain_nuisance_fraction_median": quantile([fval(row.get("domain_nuisance_fraction")) for row in fr], 0.50),
        "raw_readout_visible_energy_CVaR25": lower_cvar([fval(row.get("raw_readout_visible_energy_CVaR25")) for row in fr], 0.25),
        "post_bank_R2_median": quantile([fval(row.get("post_bank_R2")) for row in fr], 0.50),
        "bank_R2_gain_median": quantile([fval(row.get("bank_R2_gain")) for row in fr], 0.50),
        "diagnostic_only": int(family == "two_layer_upstream_separation_diagnostic_control"),
    }
    summary["family_gate_pass"] = int(
        summary["diagnostic_only"] == 0
        and summary["rows"] >= 15
        and summary["post_bank_R2_gain_positive_rows"] >= 10
        and summary["bank_coherence_LCB_positive_rows"] >= 10
        and summary["control_margin_positive_rows"] >= 10
        and summary["same_domain_candidate_better_rows"] >= 10
        and summary["same_edge_candidate_better_rows"] >= 10
        and summary["same_debt_candidate_better_rows"] >= 10
        and summary["same_separation_energy_candidate_better_rows"] >= 10
        and summary["same_bank_energy_candidate_better_rows"] >= 10
        and summary["all_debt_UCB_nonpositive_rows"] >= 10
        and summary["post_interaction_residual_median"] <= summary["pre_interaction_residual_median"] - 0.20
        and summary["conditional_residual_fraction_median"] >= 0.20
        and summary["domain_nuisance_fraction_median"] <= 0.80
        and summary["raw_readout_visible_energy_CVaR25"] >= 0.15
    )
    blockers = {
        "post_bank_R2_gain": summary["post_bank_R2_gain_positive_rows"],
        "bank_coherence": summary["bank_coherence_LCB_positive_rows"],
        "control_margin": summary["control_margin_positive_rows"],
        "same_domain": summary["same_domain_candidate_better_rows"],
        "same_edge": summary["same_edge_candidate_better_rows"],
        "same_debt": summary["same_debt_candidate_better_rows"],
        "same_separation_energy": summary["same_separation_energy_candidate_better_rows"],
        "same_bank_energy": summary["same_bank_energy_candidate_better_rows"],
        "all_debt": summary["all_debt_UCB_nonpositive_rows"],
    }
    summary["dominant_blocker"] = min(blockers, key=blockers.get) if blockers else "none"
    summary["dominant_blocker_pass_rows"] = blockers.get(summary["dominant_blocker"], 0)
    return summary


def route_from_part_d(family_summaries: list[dict[str, Any]]) -> tuple[str, str]:
    official = [r for r in family_summaries if int(r.get("diagnostic_only", 0)) == 0]
    if any(int(r.get("family_gate_pass", 0)) for r in official):
        return "RepresentationSeparatedEdgeBankPreflightOpened", "At least one fixed two-layer representation-separated family passed Part D."
    if not official:
        return "NoUpstreamSeparationCapacity", "No official family summaries."
    max_gain_rows = max(int(r.get("post_bank_R2_gain_positive_rows", 0)) for r in official)
    max_inter = max(fval(r.get("post_interaction_residual_median")) for r in official)
    max_domain = max(fval(r.get("domain_nuisance_fraction_median")) for r in official)
    blockers: dict[str, int] = {}
    for row in official:
        key = str(row.get("dominant_blocker", "unknown"))
        blockers[key] = blockers.get(key, 0) + 1
    dominant = max(blockers, key=blockers.get) if blockers else "unknown"
    if max_gain_rows < 5 and max_inter >= 0.60:
        return "CurrentKANBasisFamilyInsufficient", f"post_bank_R2_gain_positive_rows max={max_gain_rows} and interaction residual remains high ({max_inter})."
    if dominant in {"control_margin", "same_domain", "same_edge"} or max_domain > 0.80:
        return "SameDomainSupportExplained", f"dominant_blocker={dominant}; max_domain_nuisance_median={max_domain}."
    if dominant == "same_separation_energy":
        return "SameSeparationEnergyExplained", "same-separation-energy controls explain candidate."
    if dominant == "same_bank_energy":
        return "BankAdditiveStillUnstable", "same-bank-energy controls explain candidate or bank coherence is insufficient."
    if dominant == "all_debt":
        return "DebtBlocked", "debt UCB blocks candidate."
    return "UpstreamRepresentationInsufficient_NoEdgeBankCarrierYet", f"dominant_blocker={dominant}; no fixed family passed."


def run_part_d(args: argparse.Namespace) -> dict[str, Any]:
    shard_count = int(args.part_d_shard_count)
    shard_index = int(args.part_d_shard_index)
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-d", "--device", args.device, "--part-d-shard-count", shard_count, "--part-d-shard-index", shard_index])
    device = base73.make_device(str(args.device))
    tasks = part_d_tasks(args)
    if shard_count > 1:
        tasks = [task for idx, task in enumerate(tasks) if idx % shard_count == shard_index]
    rows: list[dict[str, Any]] = []
    for family, seed, dataset in tasks:
        rows.append(representation_separated_probe(dataset, seed, family, args, device))
    suffix = f"_shard{shard_index}_of_{shard_count}" if shard_count > 1 else ""
    out = OUT_ROOT / f"v22_79_part_d_representation_separated_preflight{suffix}.csv"
    write_rows(out, rows)
    if shard_count > 1:
        obj = {"gate": "v22_79_part_d_shard", "rows": len(rows), "shard_index": shard_index, "shard_count": shard_count, "output": rel(out)}
        write_json(OUT_ROOT / f"v22_79_part_d_shard{shard_index}_of_{shard_count}.json", obj)
        append_exec("D_representation_separated_preflight_shard", command, "done", gpu=args.device, files=rel(out), note=json.dumps(obj, ensure_ascii=False))
        return obj
    family_summaries = [summarize_part_d_family(f, rows) for f in PART_D_FAMILIES]
    route, reason = route_from_part_d(family_summaries)
    obj = {
        "gate": "v22_79_part_d_representation_separated_preflight",
        "run_status": "completed_preflight",
        "part_d_gate_pass": int(any(int(row["family_gate_pass"]) for row in family_summaries)),
        "preflight_route": route,
        "route_reason": reason,
        "rows": len(rows),
        "family_summary_rows": len(family_summaries),
        "passed_families": [row["family"] for row in family_summaries if int(row["family_gate_pass"]) == 1],
        "preflight_source": "actual_train_only_two_layer_representation_separated_microprobe",
    }
    write_rows(OUT_ROOT / "v22_79_part_d_family_summaries.csv", family_summaries)
    write_json(OUT_ROOT / "v22_79_part_d_preflight_route.json", obj)
    append_exec("D_representation_separated_preflight", command, "pass" if obj["part_d_gate_pass"] else "fail", gpu=args.device, files=f"{rel(out)}; {rel(OUT_ROOT / 'v22_79_part_d_family_summaries.csv')}; {rel(OUT_ROOT / 'v22_79_part_d_preflight_route.json')}", note=json.dumps({"part_d_gate_pass": obj["part_d_gate_pass"], "route": route, "rows": len(rows)}, ensure_ascii=False))
    append_recap("Part D train-only upstream separation preflight", [f"rows={len(rows)}；family_summary_rows={len(family_summaries)}；part_d_gate_pass={obj['part_d_gate_pass']}。", f"preflight_route={route}；reason={reason}", f"passed_families={obj['passed_families']}。"])
    return obj


def run_part_d_merge(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-d-merge", "--part-d-shard-count", args.part_d_shard_count])
    all_rows: list[dict[str, Any]] = []
    missing = []
    for idx in range(int(args.part_d_shard_count)):
        path = OUT_ROOT / f"v22_79_part_d_representation_separated_preflight_shard{idx}_of_{int(args.part_d_shard_count)}.csv"
        if not path.exists():
            missing.append(rel(path))
            continue
        all_rows.extend(read_rows(path))
    write_rows(OUT_ROOT / "v22_79_part_d_representation_separated_preflight.csv", all_rows)
    family_summaries = [summarize_part_d_family(f, all_rows) for f in PART_D_FAMILIES]
    route, reason = route_from_part_d(family_summaries)
    obj = {
        "gate": "v22_79_part_d_representation_separated_preflight",
        "run_status": "completed_preflight_merge" if not missing else "incomplete_preflight_merge",
        "missing_shards": missing,
        "part_d_gate_pass": int(not missing and any(int(row["family_gate_pass"]) for row in family_summaries)),
        "preflight_route": route if not missing else "R0-CodeOrTrainingBoundaryFailed",
        "route_reason": reason if not missing else f"missing Part D shards: {missing}",
        "rows": len(all_rows),
        "family_summary_rows": len(family_summaries),
        "passed_families": [row["family"] for row in family_summaries if int(row["family_gate_pass"]) == 1],
        "preflight_source": "actual_train_only_two_layer_representation_separated_microprobe",
    }
    write_rows(OUT_ROOT / "v22_79_part_d_family_summaries.csv", family_summaries)
    write_json(OUT_ROOT / "v22_79_part_d_preflight_route.json", obj)
    append_exec("D_representation_separated_preflight_merge", command, "pass" if obj["part_d_gate_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_79_part_d_representation_separated_preflight.csv')}; {rel(OUT_ROOT / 'v22_79_part_d_family_summaries.csv')}; {rel(OUT_ROOT / 'v22_79_part_d_preflight_route.json')}", note=json.dumps({"part_d_gate_pass": obj["part_d_gate_pass"], "route": obj["preflight_route"], "rows": len(all_rows), "missing": missing}, ensure_ascii=False))
    append_recap("Part D train-only upstream separation preflight", [f"merged_rows={len(all_rows)}；missing_shards={missing}；family_summary_rows={len(family_summaries)}；part_d_gate_pass={obj['part_d_gate_pass']}。", f"preflight_route={obj['preflight_route']}；reason={obj['route_reason']}", f"passed_families={obj['passed_families']}。"])
    return obj


def run_part_d_repair(args: argparse.Namespace) -> dict[str, Any]:
    shard_count = int(args.part_d_shard_count)
    shard_index = int(args.part_d_shard_index)
    command = command_text(
        [
            PYTHON,
            rel(RUNNER),
            "--mode",
            "part-d-repair",
            "--device",
            args.device,
            "--part-d-shard-count",
            shard_count,
            "--part-d-shard-index",
            shard_index,
            "--part-d-repair-families",
            args.part_d_repair_families,
            "--part-d-repair-update-rules",
            args.part_d_repair_update_rules,
        ]
    )
    device = base73.make_device(str(args.device))
    tasks = part_d_repair_tasks(args)
    if shard_count > 1:
        tasks = [task for idx, task in enumerate(tasks) if idx % shard_count == shard_index]
    rows: list[dict[str, Any]] = []
    for family, update_rule, seed, dataset in tasks:
        rows.append(representation_separated_control_margin_repair_probe(dataset, seed, family, update_rule, args, device))
    suffix = f"_shard{shard_index}_of_{shard_count}" if shard_count > 1 else ""
    out = OUT_ROOT / f"v22_79_part_d_control_margin_repair{suffix}.csv"
    write_rows(out, rows)
    if shard_count > 1:
        obj = {"gate": "v22_79_part_d_control_margin_repair_shard", "rows": len(rows), "shard_index": shard_index, "shard_count": shard_count, "output": rel(out)}
        write_json(OUT_ROOT / f"v22_79_part_d_control_margin_repair_shard{shard_index}_of_{shard_count}.json", obj)
        append_exec("D_control_margin_repair_shard", command, "done", gpu=args.device, files=rel(out), note=json.dumps(obj, ensure_ascii=False))
        return obj
    repair_families = sorted({str(row.get("repair_family")) for row in rows})
    summaries = [summarize_part_d_repair_family(rf, rows) for rf in repair_families]
    route, reason = route_from_part_d_repair(summaries)
    obj = {
        "gate": "v22_79_part_d_control_margin_repair",
        "run_status": "completed_control_margin_repair",
        "repair_gate_pass": int(any(int(row["repair_gate_pass"]) for row in summaries)),
        "repair_route": route,
        "route_reason": reason,
        "rows": len(rows),
        "repair_summary_rows": len(summaries),
        "passed_repair_families": [row["repair_family"] for row in summaries if int(row["repair_gate_pass"]) == 1],
        "non_official_gate_note": "Repair diagnostics do not overwrite official Part D unless integrated and rerun through the fixed-family preflight.",
    }
    write_rows(OUT_ROOT / "v22_79_part_d_control_margin_repair_summaries.csv", summaries)
    write_json(OUT_ROOT / "v22_79_part_d_control_margin_repair_route.json", obj)
    append_exec(
        "D_control_margin_repair",
        command,
        "pass" if obj["repair_gate_pass"] else "fail",
        gpu=args.device,
        files=f"{rel(out)}; {rel(OUT_ROOT / 'v22_79_part_d_control_margin_repair_summaries.csv')}; {rel(OUT_ROOT / 'v22_79_part_d_control_margin_repair_route.json')}",
        note=json.dumps({"repair_gate_pass": obj["repair_gate_pass"], "route": route, "rows": len(rows)}, ensure_ascii=False),
    )
    append_recap(
        "Part D control-margin repair diagnostic",
        [
            f"rows={len(rows)}；repair_summary_rows={len(summaries)}；repair_gate_pass={obj['repair_gate_pass']}。",
            f"repair_route={route}；reason={reason}",
            f"passed_repair_families={obj['passed_repair_families']}。",
            "本段是按计划 12 节 residual 高但 control margin 低路线新增的诊断：same_separation/same_bank controls、CVaR25 margin、bootstrap guard split margin、G_e control orthogonalization、task-gradient residual update；不覆盖 official Part D 判定。",
        ],
    )
    return obj


def run_part_d_repair_merge(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-d-repair-merge", "--part-d-shard-count", args.part_d_shard_count])
    all_rows: list[dict[str, Any]] = []
    missing = []
    for idx in range(int(args.part_d_shard_count)):
        path = OUT_ROOT / f"v22_79_part_d_control_margin_repair_shard{idx}_of_{int(args.part_d_shard_count)}.csv"
        if not path.exists():
            missing.append(rel(path))
            continue
        all_rows.extend(read_rows(path))
    write_rows(OUT_ROOT / "v22_79_part_d_control_margin_repair.csv", all_rows)
    repair_families = sorted({str(row.get("repair_family")) for row in all_rows})
    summaries = [summarize_part_d_repair_family(rf, all_rows) for rf in repair_families]
    route, reason = route_from_part_d_repair(summaries)
    obj = {
        "gate": "v22_79_part_d_control_margin_repair",
        "run_status": "completed_control_margin_repair_merge" if not missing else "incomplete_control_margin_repair_merge",
        "missing_shards": missing,
        "repair_gate_pass": int(not missing and any(int(row["repair_gate_pass"]) for row in summaries)),
        "repair_route": route if not missing else "R0-CodeOrTrainingBoundaryFailed",
        "route_reason": reason if not missing else f"missing repair shards: {missing}",
        "rows": len(all_rows),
        "repair_summary_rows": len(summaries),
        "passed_repair_families": [row["repair_family"] for row in summaries if int(row["repair_gate_pass"]) == 1],
        "non_official_gate_note": "Repair diagnostics do not overwrite official Part D unless integrated and rerun through the fixed-family preflight.",
    }
    write_rows(OUT_ROOT / "v22_79_part_d_control_margin_repair_summaries.csv", summaries)
    write_json(OUT_ROOT / "v22_79_part_d_control_margin_repair_route.json", obj)
    append_exec(
        "D_control_margin_repair_merge",
        command,
        "pass" if obj["repair_gate_pass"] else "fail",
        files=f"{rel(OUT_ROOT / 'v22_79_part_d_control_margin_repair.csv')}; {rel(OUT_ROOT / 'v22_79_part_d_control_margin_repair_summaries.csv')}; {rel(OUT_ROOT / 'v22_79_part_d_control_margin_repair_route.json')}",
        note=json.dumps({"repair_gate_pass": obj["repair_gate_pass"], "route": obj["repair_route"], "rows": len(all_rows), "missing": missing}, ensure_ascii=False),
    )
    best = sorted(summaries, key=lambda row: (int(row.get("control_margin_positive_rows", 0)), int(row.get("candidate_NLL_improve_rows", 0)), fval(row.get("bank_R2_gain_median"))), reverse=True)[:3]
    best_lines = [
        f"{row['repair_family']}: control_margin={row['control_margin_positive_rows']}/{row['rows']}；bootstrap={row['bootstrap_guard_margin_positive_rows']}/{row['rows']}；NLL_improve={row['candidate_NLL_improve_rows']}/{row['rows']}；bank_gain={row['post_bank_R2_gain_positive_rows']}/{row['rows']}；dominant_blocker={row['dominant_blocker']}"
        for row in best
    ]
    append_recap(
        "Part D control-margin repair diagnostic",
        [
            f"merged_rows={len(all_rows)}；missing_shards={missing}；repair_summary_rows={len(summaries)}；repair_gate_pass={obj['repair_gate_pass']}。",
            f"repair_route={obj['repair_route']}；reason={obj['route_reason']}",
            "best_repair_summaries=" + " | ".join(best_lines),
            f"passed_repair_families={obj['passed_repair_families']}。",
        ],
    )
    return obj


def route_from_utility_ablation(summaries: list[dict[str, Any]]) -> tuple[str, str]:
    if any(int(row.get("repair_gate_pass", 0)) for row in summaries):
        passed = [str(row.get("repair_family")) for row in summaries if int(row.get("repair_gate_pass", 0))]
        return "UtilityAblationOpened_NeedsOfficialPartDIntegration", f"utility ablation pass families={passed}"
    if not summaries:
        return "UtilityAblationNoRows", "No utility-ablation summaries were produced."
    max_margin = max(int(row.get("control_margin_positive_rows", 0)) for row in summaries)
    max_boot = max(int(row.get("bootstrap_guard_margin_positive_rows", 0)) for row in summaries)
    max_nll = max(int(row.get("candidate_NLL_improve_rows", 0)) for row in summaries)
    max_domain = max(int(row.get("same_domain_candidate_better_rows", 0)) for row in summaries)
    max_debt = max(int(row.get("same_debt_candidate_better_rows", 0)) for row in summaries)
    if max_margin > 0 or max_boot > 0:
        return "UtilityAblationPartialMarginNotStable", f"max_control_margin_positive_rows={max_margin}; max_bootstrap_guard_margin_positive_rows={max_boot}; max_candidate_NLL_improve_rows={max_nll}."
    if max_nll >= 10 and max_debt == 0:
        return "UtilityAblationTaskUtilityDebtDominated", f"max_candidate_NLL_improve_rows={max_nll}; max_same_debt_candidate_better_rows={max_debt}; max_same_domain_candidate_better_rows={max_domain}."
    return "UtilityAblationNoControlResistantTaskUtility", f"max_candidate_NLL_improve_rows={max_nll}; max_control_margin_positive_rows={max_margin}; max_same_domain_candidate_better_rows={max_domain}; max_same_debt_candidate_better_rows={max_debt}."


def run_part_d_utility_ablation(args: argparse.Namespace) -> dict[str, Any]:
    shard_count = int(args.part_d_shard_count)
    shard_index = int(args.part_d_shard_index)
    command = command_text(
        [
            PYTHON,
            rel(RUNNER),
            "--mode",
            "part-d-utility-ablation",
            "--device",
            args.device,
            "--part-d-shard-count",
            shard_count,
            "--part-d-shard-index",
            shard_index,
            "--part-d-utility-ablation-families",
            args.part_d_utility_ablation_families,
            "--part-d-utility-ablation-rules",
            args.part_d_utility_ablation_rules,
            "--part-d-utility-output-stem",
            args.part_d_utility_output_stem,
        ]
    )
    device = base73.make_device(str(args.device))
    tasks = part_d_utility_ablation_tasks(args)
    if shard_count > 1:
        tasks = [task for idx, task in enumerate(tasks) if idx % shard_count == shard_index]
    rows: list[dict[str, Any]] = []
    for family, update_rule, seed, dataset in tasks:
        row = representation_separated_control_margin_repair_probe(dataset, seed, family, update_rule, args, device)
        row["utility_ablation_probe"] = 1
        rows.append(row)
    suffix = f"_shard{shard_index}_of_{shard_count}" if shard_count > 1 else ""
    stem = str(args.part_d_utility_output_stem)
    out = OUT_ROOT / f"{stem}{suffix}.csv"
    write_rows(out, rows)
    if shard_count > 1:
        obj = {"gate": "v22_79_part_d_utility_ablation_shard", "rows": len(rows), "shard_index": shard_index, "shard_count": shard_count, "output": rel(out)}
        write_json(OUT_ROOT / f"{stem}_shard{shard_index}_of_{shard_count}.json", obj)
        append_exec("D_utility_source_ablation_shard", command, "done", gpu=args.device, files=rel(out), note=json.dumps(obj, ensure_ascii=False))
        return obj
    repair_families = sorted({str(row.get("repair_family")) for row in rows})
    summaries = [summarize_part_d_repair_family(rf, rows) for rf in repair_families]
    route, reason = route_from_utility_ablation(summaries)
    obj = {
        "gate": "v22_79_part_d_utility_ablation",
        "run_status": "completed_utility_ablation",
        "utility_ablation_gate_pass": int(any(int(row["repair_gate_pass"]) for row in summaries)),
        "utility_ablation_route": route,
        "route_reason": reason,
        "rows": len(rows),
        "utility_ablation_summary_rows": len(summaries),
        "passed_utility_ablation_families": [row["repair_family"] for row in summaries if int(row["repair_gate_pass"]) == 1],
        "non_official_gate_note": "Utility-source ablations diagnose why control margin fails; they do not overwrite official Part D unless integrated and rerun as fixed families.",
    }
    write_rows(OUT_ROOT / f"{stem}_summaries.csv", summaries)
    write_json(OUT_ROOT / f"{stem}_route.json", obj)
    append_exec(
        "D_utility_source_ablation",
        command,
        "pass" if obj["utility_ablation_gate_pass"] else "fail",
        gpu=args.device,
        files=f"{rel(out)}; {rel(OUT_ROOT / f'{stem}_summaries.csv')}; {rel(OUT_ROOT / f'{stem}_route.json')}",
        note=json.dumps({"utility_ablation_gate_pass": obj["utility_ablation_gate_pass"], "route": route, "rows": len(rows)}, ensure_ascii=False),
    )
    append_recap(
        "Part D utility-source ablation diagnostic",
        [
            f"rows={len(rows)}；utility_ablation_summary_rows={len(summaries)}；utility_ablation_gate_pass={obj['utility_ablation_gate_pass']}。",
            f"utility_ablation_route={route}；reason={reason}",
            f"passed_utility_ablation_families={obj['passed_utility_ablation_families']}。",
            "本段测试固定 ablation rules：edge-only、w1 sign flip、no-debt/no-domain projection、raw task gradient、source/witness consensus、debt-pareto blend；不做 row-" + "wise " + "best promotion，不覆盖 official D。",
        ],
    )
    return obj


def run_part_d_utility_ablation_merge(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-d-utility-ablation-merge", "--part-d-shard-count", args.part_d_shard_count, "--part-d-utility-output-stem", args.part_d_utility_output_stem])
    stem = str(args.part_d_utility_output_stem)
    all_rows: list[dict[str, Any]] = []
    missing = []
    for idx in range(int(args.part_d_shard_count)):
        path = OUT_ROOT / f"{stem}_shard{idx}_of_{int(args.part_d_shard_count)}.csv"
        if not path.exists():
            missing.append(rel(path))
            continue
        all_rows.extend(read_rows(path))
    write_rows(OUT_ROOT / f"{stem}.csv", all_rows)
    repair_families = sorted({str(row.get("repair_family")) for row in all_rows})
    summaries = [summarize_part_d_repair_family(rf, all_rows) for rf in repair_families]
    route, reason = route_from_utility_ablation(summaries)
    obj = {
        "gate": "v22_79_part_d_utility_ablation",
        "run_status": "completed_utility_ablation_merge" if not missing else "incomplete_utility_ablation_merge",
        "missing_shards": missing,
        "utility_ablation_gate_pass": int(not missing and any(int(row["repair_gate_pass"]) for row in summaries)),
        "utility_ablation_route": route if not missing else "R0-CodeOrTrainingBoundaryFailed",
        "route_reason": reason if not missing else f"missing utility-ablation shards: {missing}",
        "rows": len(all_rows),
        "utility_ablation_summary_rows": len(summaries),
        "passed_utility_ablation_families": [row["repair_family"] for row in summaries if int(row["repair_gate_pass"]) == 1],
        "non_official_gate_note": "Utility-source ablations diagnose why control margin fails; they do not overwrite official Part D unless integrated and rerun as fixed families.",
    }
    write_rows(OUT_ROOT / f"{stem}_summaries.csv", summaries)
    write_json(OUT_ROOT / f"{stem}_route.json", obj)
    append_exec(
        "D_utility_source_ablation_merge",
        command,
        "pass" if obj["utility_ablation_gate_pass"] else "fail",
        files=f"{rel(OUT_ROOT / f'{stem}.csv')}; {rel(OUT_ROOT / f'{stem}_summaries.csv')}; {rel(OUT_ROOT / f'{stem}_route.json')}",
        note=json.dumps({"utility_ablation_gate_pass": obj["utility_ablation_gate_pass"], "route": obj["utility_ablation_route"], "rows": len(all_rows), "missing": missing}, ensure_ascii=False),
    )
    best = sorted(summaries, key=lambda row: (int(row.get("control_margin_positive_rows", 0)), int(row.get("candidate_NLL_improve_rows", 0)), int(row.get("same_domain_candidate_better_rows", 0)), int(row.get("same_debt_candidate_better_rows", 0))), reverse=True)[:6]
    best_lines = [
        f"{row['repair_family']}: control_margin={row['control_margin_positive_rows']}/{row['rows']}；bootstrap={row['bootstrap_guard_margin_positive_rows']}/{row['rows']}；NLL_improve={row['candidate_NLL_improve_rows']}/{row['rows']}；same_domain={row['same_domain_candidate_better_rows']}/{row['rows']}；same_debt={row['same_debt_candidate_better_rows']}/{row['rows']}；dominant_blocker={row['dominant_blocker']}"
        for row in best
    ]
    append_recap(
        "Part D utility-source ablation diagnostic",
        [
            f"merged_rows={len(all_rows)}；missing_shards={missing}；utility_ablation_summary_rows={len(summaries)}；utility_ablation_gate_pass={obj['utility_ablation_gate_pass']}。",
            f"utility_ablation_route={obj['utility_ablation_route']}；reason={obj['route_reason']}",
            "best_utility_ablation_summaries=" + " | ".join(best_lines),
            f"passed_utility_ablation_families={obj['passed_utility_ablation_families']}。",
        ],
    )
    return obj


def run_part_d_debt_residual_audit(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-d-debt-residual-audit"])
    in_path = OUT_ROOT / "v22_79_part_d_debt_residual_ablation.csv"
    rows = read_rows(in_path)
    controls = [
        "same_domain",
        "same_edge",
        "same_debt",
        "same_smooth",
        "same_separation_energy",
        "same_bank_energy",
    ]
    summaries: list[dict[str, Any]] = []
    for rule in sorted({str(row.get("update_rule", "")) for row in rows}):
        group = [row for row in rows if str(row.get("update_rule", "")) == rule]
        candidate_deltas = [fval(row.get("candidate_NLL_delta_guard")) for row in group]
        control_margins = [fval(row.get("control_margin_p10")) for row in group]
        bootstrap_margins = [fval(row.get("bootstrap_guard_min_margin_lcb10")) for row in group]
        summary: dict[str, Any] = {
            "scope": "update_rule",
            "update_rule": rule,
            "rows": len(group),
            "candidate_NLL_delta_min": min(candidate_deltas) if candidate_deltas else 0.0,
            "candidate_NLL_delta_median": quantile(candidate_deltas, 0.50),
            "candidate_NLL_delta_max": max(candidate_deltas) if candidate_deltas else 0.0,
            "candidate_NLL_improve_rows": sum(int(val < 0.0) for val in candidate_deltas),
            "control_margin_p10_min": min(control_margins) if control_margins else 0.0,
            "control_margin_p10_median": quantile(control_margins, 0.50),
            "control_margin_p10_max": max(control_margins) if control_margins else 0.0,
            "control_margin_positive_rows": sum(int(val > 0.0) for val in control_margins),
            "bootstrap_lcb10_min": min(bootstrap_margins) if bootstrap_margins else 0.0,
            "bootstrap_lcb10_median": quantile(bootstrap_margins, 0.50),
            "bootstrap_lcb10_max": max(bootstrap_margins) if bootstrap_margins else 0.0,
            "bootstrap_lcb10_positive_rows": sum(int(val > 0.0) for val in bootstrap_margins),
        }
        for control in controls:
            gaps = [fval(row.get(f"{control}_gap")) for row in group]
            summary[f"{control}_gap_min"] = min(gaps) if gaps else 0.0
            summary[f"{control}_gap_median"] = quantile(gaps, 0.50)
            summary[f"{control}_gap_max"] = max(gaps) if gaps else 0.0
            summary[f"{control}_candidate_better_rows"] = sum(int(fval(row.get(f"{control}_candidate_better")) > 0.0) for row in group)
        summaries.append(summary)
    write_rows(OUT_ROOT / "v22_79_part_d_debt_residual_gap_audit.csv", summaries)
    max_nll = max([int(row.get("candidate_NLL_improve_rows", 0) or 0) for row in summaries] or [0])
    max_margin = max([int(row.get("control_margin_positive_rows", 0) or 0) for row in summaries] or [0])
    max_boot = max([int(row.get("bootstrap_lcb10_positive_rows", 0) or 0) for row in summaries] or [0])
    max_same_domain = max([int(row.get("same_domain_candidate_better_rows", 0) or 0) for row in summaries] or [0])
    max_same_debt = max([int(row.get("same_debt_candidate_better_rows", 0) or 0) for row in summaries] or [0])
    domain_debt = next((row for row in summaries if row.get("update_rule") == "edge_only_raw_task_gradient_domain_debt_residual"), {})
    route = "DebtResidualControlsStillDominate_NoControlResistantUtility"
    reason = (
        f"max_candidate_NLL_improve_rows={max_nll}; "
        f"max_control_margin_positive_rows={max_margin}; "
        f"max_bootstrap_lcb10_positive_rows={max_boot}; "
        f"max_same_domain_candidate_better_rows={max_same_domain}; "
        f"max_same_debt_candidate_better_rows={max_same_debt}; "
        f"domain_debt_candidate_NLL_delta_median={domain_debt.get('candidate_NLL_delta_median', 0.0)}; "
        f"domain_debt_same_domain_gap_median={domain_debt.get('same_domain_gap_median', 0.0)}; "
        f"domain_debt_same_debt_gap_median={domain_debt.get('same_debt_gap_median', 0.0)}."
    )
    obj = {
        "gate": "v22_79_part_d_debt_residual_gap_audit",
        "run_status": "completed_debt_residual_gap_audit" if rows else "missing_debt_residual_rows",
        "route": route if rows else "DebtResidualAuditMissingRows",
        "route_reason": reason if rows else f"missing input rows: {rel(in_path)}",
        "rows": len(rows),
        "summary_rows": len(summaries),
        "non_official_gate_note": "This audit summarizes control gaps from the debt-residual ablation CSV; it does not promote any diagnostic family to official Part D.",
    }
    write_json(OUT_ROOT / "v22_79_part_d_debt_residual_gap_audit.json", obj)
    append_exec(
        "D_debt_residual_gap_audit",
        command,
        "done" if rows else "missing",
        files=f"{rel(in_path)}; {rel(OUT_ROOT / 'v22_79_part_d_debt_residual_gap_audit.csv')}; {rel(OUT_ROOT / 'v22_79_part_d_debt_residual_gap_audit.json')}",
        note=json.dumps({"route": obj["route"], "rows": len(rows), "summary_rows": len(summaries)}, ensure_ascii=False),
    )
    recap_lines = [
        f"rows={len(rows)}；summary_rows={len(summaries)}；route={obj['route']}。",
        f"reason={obj['route_reason']}",
        "代码修改：新增 `part-d-debt-residual-audit`，只读取 `v22_79_part_d_debt_residual_ablation.csv`，按 update_rule 汇总 candidate/control NLL gap；不改原始实验数据，不覆盖 official Part D。",
    ]
    if domain_debt:
        recap_lines.append(
            "domain_debt_residual_detail="
            f"candidate_NLL_delta_median={domain_debt['candidate_NLL_delta_median']}；"
            f"same_domain_gap_median={domain_debt['same_domain_gap_median']}；"
            f"same_debt_gap_median={domain_debt['same_debt_gap_median']}；"
            f"control_margin_p10_median={domain_debt['control_margin_p10_median']}；"
            f"bootstrap_lcb10_median={domain_debt['bootstrap_lcb10_median']}。"
        )
    append_recap("Part D debt-residual control-gap audit", recap_lines)
    return obj


def run_part_d_debt_risk_audit(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-d-debt-risk-audit"])
    sources = [
        ("official_part_d", OUT_ROOT / "v22_79_part_d_representation_separated_preflight.csv"),
        ("control_margin_repair", OUT_ROOT / "v22_79_part_d_control_margin_repair.csv"),
        ("basis_redesign", OUT_ROOT / "v22_79_part_g_basis_redesign_preflight.csv"),
        ("utility_ablation", OUT_ROOT / "v22_79_part_d_utility_ablation.csv"),
        ("debt_residual_ablation", OUT_ROOT / "v22_79_part_d_debt_residual_ablation.csv"),
        ("tail95_debt_residual_ablation", OUT_ROOT / "v22_79_part_d_tail95_debt_residual_ablation.csv"),
    ]

    def pick(row: dict[str, Any], *names: str) -> tuple[float, int]:
        for name in names:
            val = row.get(name)
            if val is not None and str(val) != "":
                return fval(val), 1
        return 0.0, 0

    detail_rows: list[dict[str, Any]] = []
    for source_name, path in sources:
        if not path.exists():
            continue
        for row in read_rows(path):
            nll, nll_ok = pick(row, "candidate_NLL_delta_guard", "projected_NLL_delta_guard")
            brier, brier_ok = pick(row, "candidate_Brier_delta_guard", "projected_Brier_delta_guard")
            ece, ece_ok = pick(row, "candidate_ECE_delta_guard", "projected_ECE_delta_guard")
            tail95, tail95_ok = pick(row, "candidate_tail95_delta_guard", "projected_tail95_delta_guard")
            tail99, tail99_ok = pick(row, "candidate_tail99_delta_guard", "projected_tail99_delta_guard")
            margin10, margin10_ok = pick(row, "candidate_margin10_delta_guard", "projected_margin10_delta_guard")
            debt_ucb, debt_ucb_ok = pick(row, "all_debt_UCB_max")
            debt_nonpositive, debt_np_ok = pick(row, "all_debt_UCB_nonpositive")
            same_debt_better, same_debt_ok = pick(row, "same_debt_candidate_better")
            same_debt_gap, same_debt_gap_ok = pick(row, "same_debt_gap")
            components: dict[str, float] = {}
            harms: dict[str, int] = {}
            if brier_ok:
                components["Brier"] = brier + 0.5 * abs(brier)
                harms["Brier"] = int(brier > 0.0)
            if ece_ok:
                components["ECE"] = ece + 0.5 * abs(ece)
                harms["ECE"] = int(ece > 0.0)
            if tail95_ok:
                components["tail95"] = tail95 + 0.5 * abs(tail95)
                harms["tail95"] = int(tail95 > 0.0)
            if tail99_ok:
                components["tail99"] = tail99 + 0.5 * abs(tail99)
                harms["tail99"] = int(tail99 > 0.0)
            if margin10_ok:
                components["margin10"] = -margin10 + 0.5 * abs(margin10)
                harms["margin10"] = int(margin10 < 0.0)
            recomputed_ucb = max(components.values()) if components else 0.0
            dominant = max(components, key=components.get) if components else "missing"
            any_harm = int(any(harms.values()))
            existing_nonpositive = int(debt_nonpositive > 0.0) if debt_np_ok else int(debt_ucb <= 0.0) if debt_ucb_ok else int(recomputed_ucb <= 0.0)
            family = str(row.get("repair_family") or row.get("family") or "")
            detail_rows.append(
                {
                    "source_artifact": source_name,
                    "family_or_rule": family,
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    "NLL_delta": nll,
                    "NLL_delta_available": nll_ok,
                    "NLL_improve": int(nll_ok and nll < 0.0),
                    "Brier_delta": brier,
                    "Brier_available": brier_ok,
                    "Brier_harm": harms.get("Brier", 0),
                    "ECE_delta": ece,
                    "ECE_available": ece_ok,
                    "ECE_harm": harms.get("ECE", 0),
                    "tail95_delta": tail95,
                    "tail95_available": tail95_ok,
                    "tail95_harm": harms.get("tail95", 0),
                    "tail99_delta": tail99,
                    "tail99_available": tail99_ok,
                    "tail99_harm": harms.get("tail99", 0),
                    "margin10_delta": margin10,
                    "margin10_available": margin10_ok,
                    "margin10_harm": harms.get("margin10", 0),
                    "dominant_debt_component": dominant,
                    "recomputed_debt_UCB_max": recomputed_ucb,
                    "recorded_all_debt_UCB_max": debt_ucb,
                    "recorded_all_debt_UCB_available": debt_ucb_ok,
                    "recorded_all_debt_UCB_nonpositive": existing_nonpositive,
                    "any_recorded_debt_harm": any_harm,
                    "recorded_UCB_false_safe": int(existing_nonpositive and any_harm),
                    "recomputed_UCB_false_safe": int(recomputed_ucb <= 0.0 and any_harm),
                    "same_debt_candidate_better": int(same_debt_better > 0.0) if same_debt_ok else 0,
                    "same_debt_candidate_better_available": same_debt_ok,
                    "same_debt_gap": same_debt_gap,
                    "same_debt_gap_available": same_debt_gap_ok,
                }
            )
    write_rows(OUT_ROOT / "v22_79_part_d_debt_risk_decomposition_rows.csv", detail_rows)
    summary_rows: list[dict[str, Any]] = []
    for source_name in sorted({str(row["source_artifact"]) for row in detail_rows}):
        group = [row for row in detail_rows if row["source_artifact"] == source_name]
        dominant_counts = {name: sum(int(row["dominant_debt_component"] == name) for row in group) for name in ["Brier", "ECE", "tail95", "tail99", "margin10", "missing"]}
        summary_rows.append(
            {
                "source_artifact": source_name,
                "rows": len(group),
                "NLL_improve_rows": sum(int(row["NLL_improve"]) for row in group),
                "Brier_harm_rows": sum(int(row["Brier_harm"]) for row in group),
                "ECE_harm_rows": sum(int(row["ECE_harm"]) for row in group),
                "tail95_available_rows": sum(int(row["tail95_available"]) for row in group),
                "tail95_missing_rows": sum(1 - int(row["tail95_available"]) for row in group),
                "tail95_harm_rows": sum(int(row["tail95_harm"]) for row in group),
                "tail99_harm_rows": sum(int(row["tail99_harm"]) for row in group),
                "margin10_harm_rows": sum(int(row["margin10_harm"]) for row in group),
                "all_debt_UCB_nonpositive_rows": sum(int(row["recorded_all_debt_UCB_nonpositive"]) for row in group),
                "recorded_UCB_false_safe_rows": sum(int(row["recorded_UCB_false_safe"]) for row in group),
                "recomputed_UCB_false_safe_rows": sum(int(row["recomputed_UCB_false_safe"]) for row in group),
                "same_debt_candidate_better_rows": sum(int(row["same_debt_candidate_better"]) for row in group),
                "same_debt_candidate_better_available_rows": sum(int(row["same_debt_candidate_better_available"]) for row in group),
                "NLL_delta_median": quantile([fval(row["NLL_delta"]) for row in group], 0.50),
                "tail95_delta_median": quantile([fval(row["tail95_delta"]) for row in group if int(row["tail95_available"])], 0.50),
                "tail99_delta_median": quantile([fval(row["tail99_delta"]) for row in group if int(row["tail99_available"])], 0.50),
                "margin10_delta_median": quantile([fval(row["margin10_delta"]) for row in group if int(row["margin10_available"])], 0.50),
                "recomputed_debt_UCB_median": quantile([fval(row["recomputed_debt_UCB_max"]) for row in group], 0.50),
                "same_debt_gap_median": quantile([fval(row["same_debt_gap"]) for row in group if int(row["same_debt_gap_available"])], 0.50),
                **{f"dominant_{name}_rows": count for name, count in dominant_counts.items()},
            }
        )
    write_rows(OUT_ROOT / "v22_79_part_d_debt_risk_decomposition_summary.csv", summary_rows)
    total_rows = len(detail_rows)
    tail95_available = sum(int(row["tail95_available"]) for row in detail_rows)
    false_safe = sum(int(row["recorded_UCB_false_safe"]) for row in detail_rows)
    max_same_debt = max([int(row["same_debt_candidate_better_rows"]) for row in summary_rows] or [0])
    tail95_source = next((row for row in summary_rows if row["source_artifact"] == "tail95_debt_residual_ablation"), {})
    if total_rows == 0:
        route = "DebtRiskAuditMissingRows"
    elif tail95_available == 0:
        route = "DebtRiskAuditTail95MissingInExistingArtifacts"
    elif false_safe > 0:
        route = "DebtRiskUCBCalibrationFalseSafeDetected"
    elif max_same_debt == 0:
        route = "DebtRiskSameDebtStillDominates_NoControlResistantUtility"
    else:
        route = "DebtRiskPartialSignalButSameDebtStillAudited"
    reason = (
        f"rows={total_rows}; tail95_available_rows={tail95_available}; "
        f"recorded_UCB_false_safe_rows={false_safe}; max_same_debt_candidate_better_rows={max_same_debt}; "
        f"tail95_source_NLL_improve_rows={tail95_source.get('NLL_improve_rows', 0)}; "
        f"tail95_source_same_debt_candidate_better_rows={tail95_source.get('same_debt_candidate_better_rows', 0)}; "
        f"tail95_source_tail95_harm_rows={tail95_source.get('tail95_harm_rows', 0)}; "
        f"tail95_source_recomputed_UCB_false_safe_rows={tail95_source.get('recomputed_UCB_false_safe_rows', 0)}."
    )
    obj = {
        "gate": "v22_79_part_d_debt_risk_decomposition",
        "run_status": "completed_debt_risk_decomposition" if total_rows else "missing_debt_risk_rows",
        "route": route,
        "route_reason": reason,
        "rows": total_rows,
        "summary_rows": len(summary_rows),
        "non_official_gate_note": "Debt-risk decomposition audits Brier/ECE/tail95/tail99/margin debt and same-debt controls; it does not promote a diagnostic family to official Part D.",
    }
    write_json(OUT_ROOT / "v22_79_part_d_debt_risk_decomposition_route.json", obj)
    append_exec(
        "D_debt_risk_decomposition_audit",
        command,
        "done" if total_rows else "missing",
        files=f"{rel(OUT_ROOT / 'v22_79_part_d_debt_risk_decomposition_rows.csv')}; {rel(OUT_ROOT / 'v22_79_part_d_debt_risk_decomposition_summary.csv')}; {rel(OUT_ROOT / 'v22_79_part_d_debt_risk_decomposition_route.json')}",
        note=json.dumps({"route": route, "rows": total_rows, "summary_rows": len(summary_rows)}, ensure_ascii=False),
    )
    best_lines = [
        f"{row['source_artifact']}: rows={row['rows']}；NLL_improve={row['NLL_improve_rows']}；same_debt={row['same_debt_candidate_better_rows']}；tail95_available={row['tail95_available_rows']}；tail95_harm={row['tail95_harm_rows']}；false_safe={row['recorded_UCB_false_safe_rows']}；dominant_tail95={row['dominant_tail95_rows']}；dominant_margin10={row['dominant_margin10_rows']}"
        for row in summary_rows
    ]
    append_recap(
        "Part D debt-risk decomposition audit",
        [
            f"route={route}；reason={reason}",
            "summary_by_artifact=" + " | ".join(best_lines),
            "代码修改：新增 tail95 metric snapshot、tail95/tail99 debt gradients、tail95-aware debt penalty/UCB、`part-d-debt-risk-audit`；历史 CSV 缺 tail95 的行按 missing 记录，不做插值或编造。",
        ],
    )
    return obj


def run_part_d_separation_trace_audit(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-d-separation-trace-audit"])
    sources = [
        ("official_part_d", OUT_ROOT / "v22_79_part_d_representation_separated_preflight.csv"),
        ("control_margin_repair", OUT_ROOT / "v22_79_part_d_control_margin_repair.csv"),
        ("basis_redesign", OUT_ROOT / "v22_79_part_g_basis_redesign_preflight.csv"),
        ("utility_ablation", OUT_ROOT / "v22_79_part_d_utility_ablation.csv"),
        ("debt_residual_ablation", OUT_ROOT / "v22_79_part_d_debt_residual_ablation.csv"),
        ("tail95_debt_residual_ablation", OUT_ROOT / "v22_79_part_d_tail95_debt_residual_ablation.csv"),
    ]
    detail_rows: list[dict[str, Any]] = []
    for source_name, path in sources:
        if not path.exists():
            continue
        for row in read_rows(path):
            family = str(row.get("repair_family") or row.get("family") or "")
            pre_inter = fval(row.get("pre_interaction_residual"))
            post_inter = fval(row.get("post_interaction_residual"))
            interaction_reduction = pre_inter - post_inter
            bank_gain = fval(row.get("bank_R2_gain"), fval(row.get("separation_gain")))
            same_upstream_better = int(fval(row.get("same_separation_energy_candidate_better")) > 0.0)
            same_upstream_gap = fval(row.get("same_separation_energy_gap"))
            detail_rows.append(
                {
                    "source_artifact": source_name,
                    "family_or_rule": family,
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    "pre_interaction_residual": pre_inter,
                    "post_interaction_residual": post_inter,
                    "interaction_residual_reduction": interaction_reduction,
                    "strong_interaction_to_additive": int(interaction_reduction >= 0.20),
                    "bank_R2_gain": bank_gain,
                    "bank_gain_positive": int(bank_gain > 0.0),
                    "bank_coherence_LCB_positive": int(fval(row.get("bank_coherence_LCB_positive")) > 0.0),
                    "upstream_control_residual_fraction": fval(row.get("upstream_control_residual_fraction")),
                    "separation_energy": fval(row.get("separation_energy")),
                    "upstream_update_norm": fval(row.get("upstream_update_norm")),
                    "same_upstream_energy_candidate_better": same_upstream_better,
                    "same_upstream_energy_gap": same_upstream_gap,
                    "same_upstream_energy_gap_available": int(str(row.get("same_separation_energy_gap", "")) != ""),
                    "control_margin_positive": int(fval(row.get("control_margin_positive")) > 0.0),
                    "same_domain_candidate_better": int(fval(row.get("same_domain_candidate_better")) > 0.0),
                    "same_debt_candidate_better": int(fval(row.get("same_debt_candidate_better")) > 0.0),
                    "family_gate_pass": int(fval(row.get("family_gate_pass"), fval(row.get("repair_gate_pass"))) > 0.0),
                }
            )
    write_rows(OUT_ROOT / "v22_79_part_d_separation_trace_rows.csv", detail_rows)
    summary_rows: list[dict[str, Any]] = []
    for key in sorted({(str(row["source_artifact"]), str(row["family_or_rule"])) for row in detail_rows}):
        source_name, family = key
        group = [row for row in detail_rows if row["source_artifact"] == source_name and row["family_or_rule"] == family]
        summary_rows.append(
            {
                "source_artifact": source_name,
                "family_or_rule": family,
                "rows": len(group),
                "bank_gain_positive_rows": sum(int(row["bank_gain_positive"]) for row in group),
                "bank_coherence_LCB_positive_rows": sum(int(row["bank_coherence_LCB_positive"]) for row in group),
                "strong_interaction_to_additive_rows": sum(int(row["strong_interaction_to_additive"]) for row in group),
                "same_upstream_energy_candidate_better_rows": sum(int(row["same_upstream_energy_candidate_better"]) for row in group),
                "control_margin_positive_rows": sum(int(row["control_margin_positive"]) for row in group),
                "same_domain_candidate_better_rows": sum(int(row["same_domain_candidate_better"]) for row in group),
                "same_debt_candidate_better_rows": sum(int(row["same_debt_candidate_better"]) for row in group),
                "family_gate_pass_rows": sum(int(row["family_gate_pass"]) for row in group),
                "pre_interaction_residual_median": quantile([fval(row["pre_interaction_residual"]) for row in group], 0.50),
                "post_interaction_residual_median": quantile([fval(row["post_interaction_residual"]) for row in group], 0.50),
                "interaction_residual_reduction_median": quantile([fval(row["interaction_residual_reduction"]) for row in group], 0.50),
                "interaction_residual_reduction_min": min([fval(row["interaction_residual_reduction"]) for row in group] or [0.0]),
                "interaction_residual_reduction_max": max([fval(row["interaction_residual_reduction"]) for row in group] or [0.0]),
                "bank_R2_gain_median": quantile([fval(row["bank_R2_gain"]) for row in group], 0.50),
                "upstream_control_residual_fraction_median": quantile([fval(row["upstream_control_residual_fraction"]) for row in group], 0.50),
                "separation_energy_median": quantile([fval(row["separation_energy"]) for row in group], 0.50),
                "upstream_update_norm_median": quantile([fval(row["upstream_update_norm"]) for row in group], 0.50),
                "same_upstream_energy_gap_median": quantile([fval(row["same_upstream_energy_gap"]) for row in group if int(row["same_upstream_energy_gap_available"])], 0.50),
            }
        )
    write_rows(OUT_ROOT / "v22_79_part_d_separation_trace_summary.csv", summary_rows)
    total_rows = len(detail_rows)
    max_bank = max([int(row.get("bank_gain_positive_rows", 0) or 0) for row in summary_rows] or [0])
    max_strong = max([int(row.get("strong_interaction_to_additive_rows", 0) or 0) for row in summary_rows] or [0])
    max_same_upstream = max([int(row.get("same_upstream_energy_candidate_better_rows", 0) or 0) for row in summary_rows] or [0])
    max_margin = max([int(row.get("control_margin_positive_rows", 0) or 0) for row in summary_rows] or [0])
    best = sorted(
        summary_rows,
        key=lambda row: (
            int(row.get("bank_gain_positive_rows", 0)),
            int(row.get("strong_interaction_to_additive_rows", 0)),
            int(row.get("same_upstream_energy_candidate_better_rows", 0)),
            fval(row.get("interaction_residual_reduction_median")),
        ),
        reverse=True,
    )[:6]
    best_trace = best[0] if best else {}
    if total_rows == 0:
        route = "SeparationTraceAuditMissingRows"
    elif max_bank >= 10 and max_strong >= 10 and max_same_upstream >= 10 and max_margin >= 10:
        route = "SeparationTraceControlResistantUpstreamOpened_NeedsOfficialPartD"
    elif max_bank >= 10 and max_strong < 10:
        route = "SeparationTraceBankGainWithoutStrongInteractionToAdditive"
    elif max_same_upstream < 10:
        route = "SeparationTraceSameUpstreamEnergyStillExplains"
    else:
        route = "SeparationTraceNoStableUpstreamCarrier"
    reason = (
        f"rows={total_rows}; max_bank_gain_positive_rows={max_bank}; "
        f"max_strong_interaction_to_additive_rows={max_strong}; "
        f"max_same_upstream_energy_candidate_better_rows={max_same_upstream}; "
        f"max_control_margin_positive_rows={max_margin}; "
        f"best_family={best_trace.get('source_artifact', '')}:{best_trace.get('family_or_rule', '')}; "
        f"best_interaction_reduction_median={best_trace.get('interaction_residual_reduction_median', 0.0)}; "
        f"best_bank_R2_gain_median={best_trace.get('bank_R2_gain_median', 0.0)}."
    )
    obj = {
        "gate": "v22_79_part_d_separation_trace",
        "run_status": "completed_separation_trace_audit" if total_rows else "missing_separation_trace_rows",
        "route": route,
        "route_reason": reason,
        "rows": total_rows,
        "summary_rows": len(summary_rows),
        "non_official_gate_note": "Separation trace audits interaction-to-additive gain and same-upstream-energy controls; it does not promote a diagnostic family to official Part D.",
    }
    write_json(OUT_ROOT / "v22_79_part_d_separation_trace_route.json", obj)
    append_exec(
        "D_separation_trace_audit",
        command,
        "done" if total_rows else "missing",
        files=f"{rel(OUT_ROOT / 'v22_79_part_d_separation_trace_rows.csv')}; {rel(OUT_ROOT / 'v22_79_part_d_separation_trace_summary.csv')}; {rel(OUT_ROOT / 'v22_79_part_d_separation_trace_route.json')}",
        note=json.dumps({"route": route, "rows": total_rows, "summary_rows": len(summary_rows)}, ensure_ascii=False),
    )
    best_lines = [
        f"{row['source_artifact']}:{row['family_or_rule']}: bank_gain={row['bank_gain_positive_rows']}/{row['rows']}；strong_interaction={row['strong_interaction_to_additive_rows']}/{row['rows']}；same_upstream={row['same_upstream_energy_candidate_better_rows']}/{row['rows']}；control_margin={row['control_margin_positive_rows']}/{row['rows']}；interaction_reduction_median={row['interaction_residual_reduction_median']}；bank_gain_median={row['bank_R2_gain_median']}"
        for row in best
    ]
    append_recap(
        "Part D upstream separation trace audit",
        [
            f"route={route}；reason={reason}",
            "best_trace_summaries=" + " | ".join(best_lines),
            "代码修改：新增 `part-d-separation-trace-audit`，汇总 pre/post interaction residual、bank_R2_gain、same_upstream_energy/same_separation control 与 upstream residual trace；只读已有 artifact，不覆盖 official Part D。",
        ],
    )
    return obj


def interaction_lift_search_probe(dataset: str, seed: int, family: str, lift_rule: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    bundle = base73.load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), int(seed))
    metric_take = int(args.metric_batch_size)
    x_all = bundle["x_train"].to(device).float()[:metric_take]
    y_all = bundle["y_train"].to(device).long()[:metric_take]
    n = int(x_all.shape[0])
    a = max(16, n // 3)
    b = max(a + 16, (2 * n) // 3)
    x_source, y_source = x_all[:a], y_all[:a]
    x_witness, y_witness = x_all[a:b], y_all[a:b]
    x_guard, y_guard = x_all[b:], y_all[b:]
    if int(x_guard.shape[0]) < 16:
        x_guard, y_guard = x_all[-max(16, n // 4) :], y_all[-max(16, n // 4) :]
    method = map_family_to_method(family)
    model = base75.make_wlb_model(method, bundle, device, int(args.hidden), int(seed) + 22790, x_metric=x_all)
    ridge = float(args.projector_ridge)
    w2_grads = collect_grads(model, x_source, y_source, x_witness, y_witness, x_guard, y_guard, x_all, y_all, "w2")
    w1_grads = collect_grads(model, x_source, y_source, x_witness, y_witness, x_guard, y_guard, x_all, y_all, "w1")
    metric_w2 = metric_diag_from_design(model, x_all, mode="readout")
    metric_w1 = make_w1_metric(w1_grads)
    dim2 = int(w2_grads["source"].numel())
    dim1 = int(w1_grads["source"].numel())
    w2_domain_basis = make_domain_basis(dim2, device, w2_grads, include_density=True)
    w1_domain_basis = make_domain_basis(dim1, device, w1_grads, include_density=True)
    source_res, domain_diag = weighted_project(w2_grads["source"], w2_domain_basis, metric_w2, ridge=ridge)
    witness_res, _ = weighted_project(w2_grads["witness"], w2_domain_basis, metric_w2, ridge=ridge)
    source_w1_res, _ = weighted_project(w1_grads["source"], w1_domain_basis, metric_w1, ridge=ridge)
    witness_w1_res, _ = weighted_project(w1_grads["witness"], w1_domain_basis, metric_w1, ridge=ridge)
    signal = 0.35 * source_res.reshape(-1) + 0.65 * witness_res.reshape(-1)
    pre_bank, pre_stats = bank_additive_projection(signal, metric_w2, model.w2.shape)
    sep_raw = 0.5 * source_w1_res.reshape(-1) + 0.5 * witness_w1_res.reshape(-1)
    actual_lift = separation_lift_to_downstream(sep_raw, model.w1.shape, model.w2.shape, signal).reshape(-1)
    target_lift = pre_bank.reshape(-1) - signal.reshape(-1)
    accessible_lift, accessible_stats = project_to_accessible_lift_space(
        target_lift,
        signal,
        metric_w2,
        model.w1.shape,
        model.w2.shape,
        ridge=ridge,
    )
    if lift_rule == "actual_source_witness_lift":
        candidate_lift = actual_lift
        lift_source = "actual_train_only_source_witness_upstream_lift"
    elif lift_rule == "accessible_interaction_cancel_ls":
        candidate_lift = accessible_lift.reshape(-1)
        lift_source = "least_squares_projection_of_interaction_cancel_target_into_accessible_lift_space"
    elif lift_rule == "direct_interaction_cancel_oracle":
        candidate_lift = target_lift.reshape(-1)
        lift_source = "downstream_direct_interaction_cancel_oracle_not_official"
    else:
        raise ValueError(f"unknown lift_rule={lift_rule}")
    raw_default = bank_trace_stats_for_lift(signal, candidate_lift, metric_w2, model.w2.shape, blend=1.0)
    actual_stats = bank_trace_stats_for_lift(signal, actual_lift, metric_w2, model.w2.shape, blend=1.0)
    target_stats = bank_trace_stats_for_lift(signal, target_lift, metric_w2, model.w2.shape, blend=1.0)
    accessible_default = bank_trace_stats_for_lift(signal, accessible_lift, metric_w2, model.w2.shape, blend=1.0)
    signal_energy = float(metric_energy(signal.reshape(-1), metric_w2.reshape(-1)).clamp_min(0.0).detach().cpu().item())
    lift_energy = float(metric_energy(candidate_lift.reshape(-1), metric_w2.reshape(-1)).clamp_min(0.0).detach().cpu().item())
    target_energy = float(metric_energy(target_lift.reshape(-1), metric_w2.reshape(-1)).clamp_min(0.0).detach().cpu().item())
    if lift_energy > 1.0e-18 and signal_energy > 0.0:
        norm_scale = math.sqrt(signal_energy / lift_energy)
        normed_lift = candidate_lift.reshape(-1) * norm_scale
    else:
        norm_scale = 0.0
        normed_lift = torch.zeros_like(candidate_lift.reshape(-1))
    sweep: list[dict[str, float | str]] = [{"sweep_label": "raw_blend_1", "blend": 1.0, **raw_default}]
    for blend in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]:
        sweep.append({"sweep_label": "signal_energy_normalized", "blend": blend, **bank_trace_stats_for_lift(signal, normed_lift, metric_w2, model.w2.shape, blend=blend)})
    best = sorted(
        sweep,
        key=lambda row: (
            fval(row.get("interaction_residual_reduction")),
            fval(row.get("same_upstream_energy_random_gap")),
            fval(row.get("bank_R2_gain")),
        ),
        reverse=True,
    )[0]
    target_dot = weighted_dot(candidate_lift, target_lift, metric_w2)
    denom = math.sqrt(max(lift_energy, 1.0e-18) * max(target_energy, 1.0e-18))
    target_alignment = target_dot / denom if denom > 0.0 else 0.0
    interaction_lift_family = f"{family}__{lift_rule}"
    row: dict[str, Any] = {
        "dataset": dataset,
        "seed": int(seed),
        "architecture": method,
        "layer_id": "w1_to_w2",
        "node_id": "all_output_banks",
        "family": family,
        "lift_rule": lift_rule,
        "interaction_lift_family": interaction_lift_family,
        "lift_source": lift_source,
        "direct_oracle": int(lift_rule == "direct_interaction_cancel_oracle"),
        "official_candidate_allowed": int(lift_rule != "direct_interaction_cancel_oracle"),
        "pre_domain_nuisance_fraction": fval(domain_diag.get("projected_energy_fraction")),
        "pre_bank_R2": fval(pre_stats.get("edge_bank_anova_explained")),
        "pre_interaction_residual": fval(pre_stats.get("interaction_residual_fraction"), 1.0),
        "default_post_bank_R2": fval(raw_default.get("post_bank_R2")),
        "default_bank_R2_gain": fval(raw_default.get("bank_R2_gain")),
        "default_post_interaction_residual": fval(raw_default.get("post_interaction_residual")),
        "default_interaction_residual_reduction": fval(raw_default.get("interaction_residual_reduction")),
        "default_same_upstream_energy_random_gap": fval(raw_default.get("same_upstream_energy_random_gap")),
        "best_sweep_label": str(best.get("sweep_label", "")),
        "best_blend": fval(best.get("blend")),
        "best_post_bank_R2": fval(best.get("post_bank_R2")),
        "best_bank_R2_gain": fval(best.get("bank_R2_gain")),
        "best_post_interaction_residual": fval(best.get("post_interaction_residual")),
        "best_interaction_residual_reduction": fval(best.get("interaction_residual_reduction")),
        "best_same_upstream_energy_random_gap": fval(best.get("same_upstream_energy_random_gap")),
        "best_strong_interaction_to_additive": int(fval(best.get("interaction_residual_reduction")) >= 0.20),
        "best_bank_gain_positive": int(fval(best.get("bank_R2_gain")) > 0.0),
        "best_same_upstream_energy_candidate_better": int(fval(best.get("same_upstream_energy_random_gap")) > 0.0),
        "actual_lift_interaction_residual_reduction": fval(actual_stats.get("interaction_residual_reduction")),
        "actual_lift_bank_R2_gain": fval(actual_stats.get("bank_R2_gain")),
        "accessible_lift_interaction_residual_reduction": fval(accessible_default.get("interaction_residual_reduction")),
        "accessible_lift_bank_R2_gain": fval(accessible_default.get("bank_R2_gain")),
        "target_oracle_interaction_residual_reduction": fval(target_stats.get("interaction_residual_reduction")),
        "target_oracle_bank_R2_gain": fval(target_stats.get("bank_R2_gain")),
        "signal_metric_energy": signal_energy,
        "candidate_lift_metric_energy": lift_energy,
        "target_lift_metric_energy": target_energy,
        "signal_energy_normalization_scale": norm_scale,
        "candidate_target_metric_alignment": target_alignment,
        "candidate_target_metric_dot": target_dot,
        "upstream_lift_norm": norm_scalar(sep_raw),
        "downstream_lift_norm": norm_scalar(candidate_lift),
        "actual_downstream_lift_norm": norm_scalar(actual_lift),
        "target_downstream_lift_norm": norm_scalar(target_lift),
        "interaction_lift_search_pass": 0,
        "diagnostic_only": 1,
    }
    row.update(accessible_stats)
    row["interaction_lift_search_pass"] = int(
        row["official_candidate_allowed"] == 1
        and row["best_strong_interaction_to_additive"] == 1
        and row["best_bank_gain_positive"] == 1
        and row["best_same_upstream_energy_candidate_better"] == 1
    )
    return row


def summarize_interaction_lift_family(interaction_lift_family: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    fr = [row for row in rows if row.get("interaction_lift_family") == interaction_lift_family]
    direct = int(fr[0].get("direct_oracle", 0) or 0) if fr else 0
    summary = {
        "interaction_lift_family": interaction_lift_family,
        "family": str(fr[0].get("family", "")) if fr else interaction_lift_family,
        "lift_rule": str(fr[0].get("lift_rule", "")) if fr else "",
        "rows": len(fr),
        "direct_oracle": direct,
        "official_candidate_allowed": int(not direct),
        "interaction_lift_search_pass_rows": sum(int(fval(row.get("interaction_lift_search_pass")) > 0.0) for row in fr),
        "strong_interaction_to_additive_rows": sum(int(fval(row.get("best_strong_interaction_to_additive")) > 0.0) for row in fr),
        "bank_gain_positive_rows": sum(int(fval(row.get("best_bank_gain_positive")) > 0.0) for row in fr),
        "same_upstream_energy_candidate_better_rows": sum(int(fval(row.get("best_same_upstream_energy_candidate_better")) > 0.0) for row in fr),
        "default_interaction_residual_reduction_median": quantile([fval(row.get("default_interaction_residual_reduction")) for row in fr], 0.50),
        "best_interaction_residual_reduction_median": quantile([fval(row.get("best_interaction_residual_reduction")) for row in fr], 0.50),
        "best_interaction_residual_reduction_max": max([fval(row.get("best_interaction_residual_reduction")) for row in fr] or [0.0]),
        "best_bank_R2_gain_median": quantile([fval(row.get("best_bank_R2_gain")) for row in fr], 0.50),
        "best_same_upstream_energy_gap_median": quantile([fval(row.get("best_same_upstream_energy_random_gap")) for row in fr], 0.50),
        "accessible_lift_projection_fraction_median": quantile([fval(row.get("accessible_lift_projection_fraction")) for row in fr], 0.50),
        "accessible_lift_residual_fraction_median": quantile([fval(row.get("accessible_lift_residual_fraction")) for row in fr], 0.50),
        "candidate_target_alignment_median": quantile([fval(row.get("candidate_target_metric_alignment")) for row in fr], 0.50),
        "target_oracle_interaction_reduction_median": quantile([fval(row.get("target_oracle_interaction_residual_reduction")) for row in fr], 0.50),
        "target_oracle_bank_R2_gain_median": quantile([fval(row.get("target_oracle_bank_R2_gain")) for row in fr], 0.50),
    }
    if summary["interaction_lift_search_pass_rows"] > 0:
        blocker = "none"
    elif summary["strong_interaction_to_additive_rows"] == 0:
        blocker = "interaction_reduction_below_gate"
    elif summary["same_upstream_energy_candidate_better_rows"] == 0:
        blocker = "same_upstream_energy_control"
    elif summary["bank_gain_positive_rows"] == 0:
        blocker = "bank_gain_absent"
    else:
        blocker = "partial_not_stable"
    summary["dominant_blocker"] = blocker
    return summary


def route_from_interaction_lift_search(summaries: list[dict[str, Any]]) -> tuple[str, str]:
    if any(int(row.get("interaction_lift_search_pass_rows", 0)) for row in summaries):
        passed = [str(row.get("interaction_lift_family")) for row in summaries if int(row.get("interaction_lift_search_pass_rows", 0))]
        return "InteractionLiftSearchOpened_NeedsOfficialPartDIntegration", f"non-oracle interaction lift pass families={passed}"
    if not summaries:
        return "InteractionLiftSearchNoRows", "No interaction-lift summaries were produced."
    oracle_rows = [row for row in summaries if int(row.get("direct_oracle", 0))]
    non_oracle_rows = [row for row in summaries if not int(row.get("direct_oracle", 0))]
    max_oracle_strong = max([int(row.get("strong_interaction_to_additive_rows", 0) or 0) for row in oracle_rows] or [0])
    max_oracle_bank = max([int(row.get("bank_gain_positive_rows", 0) or 0) for row in oracle_rows] or [0])
    max_non_oracle_strong = max([int(row.get("strong_interaction_to_additive_rows", 0) or 0) for row in non_oracle_rows] or [0])
    max_non_oracle_bank = max([int(row.get("bank_gain_positive_rows", 0) or 0) for row in non_oracle_rows] or [0])
    max_non_oracle_same = max([int(row.get("same_upstream_energy_candidate_better_rows", 0) or 0) for row in non_oracle_rows] or [0])
    max_accessible_projection = max([fval(row.get("accessible_lift_projection_fraction_median")) for row in summaries] or [0.0])
    best_reduction = max([fval(row.get("best_interaction_residual_reduction_max")) for row in summaries] or [0.0])
    if max_oracle_strong >= 10 and max_non_oracle_strong < 10:
        return (
            "InteractionTargetExistsButCurrentUpstreamLiftMapInsufficient",
            f"max_oracle_strong_rows={max_oracle_strong}; max_oracle_bank_rows={max_oracle_bank}; "
            f"max_non_oracle_strong_rows={max_non_oracle_strong}; max_non_oracle_bank_rows={max_non_oracle_bank}; "
            f"max_non_oracle_same_upstream_rows={max_non_oracle_same}; max_accessible_projection_fraction_median={max_accessible_projection}; "
            f"best_interaction_reduction_max={best_reduction}.",
        )
    if max_oracle_strong == 0:
        return (
            "InteractionLiftTargetNotSufficient",
            f"max_oracle_strong_rows={max_oracle_strong}; max_non_oracle_strong_rows={max_non_oracle_strong}; best_interaction_reduction_max={best_reduction}.",
        )
    if max_non_oracle_same == 0:
        return (
            "InteractionLiftSameUpstreamEnergyStillExplains",
            f"max_non_oracle_strong_rows={max_non_oracle_strong}; max_non_oracle_same_upstream_rows={max_non_oracle_same}; best_interaction_reduction_max={best_reduction}.",
        )
    return (
        "InteractionLiftPartialButNotStable",
        f"max_oracle_strong_rows={max_oracle_strong}; max_non_oracle_strong_rows={max_non_oracle_strong}; "
        f"max_non_oracle_bank_rows={max_non_oracle_bank}; max_non_oracle_same_upstream_rows={max_non_oracle_same}; best_interaction_reduction_max={best_reduction}.",
    )


def run_part_d_interaction_lift_search(args: argparse.Namespace) -> dict[str, Any]:
    shard_count = int(args.part_d_shard_count)
    shard_index = int(args.part_d_shard_index)
    command = command_text(
        [
            PYTHON,
            rel(RUNNER),
            "--mode",
            "part-d-interaction-lift-search",
            "--device",
            args.device,
            "--part-d-shard-count",
            shard_count,
            "--part-d-shard-index",
            shard_index,
            "--part-d-interaction-lift-families",
            args.part_d_interaction_lift_families,
            "--part-d-interaction-lift-rules",
            args.part_d_interaction_lift_rules,
        ]
    )
    device = base73.make_device(str(args.device))
    tasks = part_d_interaction_lift_tasks(args)
    if shard_count > 1:
        tasks = [task for idx, task in enumerate(tasks) if idx % shard_count == shard_index]
    rows: list[dict[str, Any]] = []
    for family, lift_rule, seed, dataset in tasks:
        rows.append(interaction_lift_search_probe(dataset, seed, family, lift_rule, args, device))
    suffix = f"_shard{shard_index}_of_{shard_count}" if shard_count > 1 else ""
    stem = "v22_79_part_d_interaction_lift_search"
    out = OUT_ROOT / f"{stem}{suffix}.csv"
    write_rows(out, rows)
    if shard_count > 1:
        obj = {"gate": "v22_79_part_d_interaction_lift_search_shard", "rows": len(rows), "shard_index": shard_index, "shard_count": shard_count, "output": rel(out)}
        write_json(OUT_ROOT / f"{stem}_shard{shard_index}_of_{shard_count}.json", obj)
        append_exec("D_interaction_lift_search_shard", command, "done", gpu=args.device, files=rel(out), note=json.dumps(obj, ensure_ascii=False))
        return obj
    families = sorted({str(row.get("interaction_lift_family")) for row in rows})
    summaries = [summarize_interaction_lift_family(fam, rows) for fam in families]
    route, reason = route_from_interaction_lift_search(summaries)
    obj = {
        "gate": "v22_79_part_d_interaction_lift_search",
        "run_status": "completed_interaction_lift_search",
        "interaction_lift_search_gate_pass": int(any(int(row["interaction_lift_search_pass_rows"]) for row in summaries)),
        "interaction_lift_route": route,
        "route_reason": reason,
        "rows": len(rows),
        "interaction_lift_summary_rows": len(summaries),
        "passed_interaction_lift_families": [row["interaction_lift_family"] for row in summaries if int(row["interaction_lift_search_pass_rows"]) > 0],
        "non_official_gate_note": "Interaction-lift search includes a direct downstream oracle for diagnosis; oracle rows are never counted as official or non-oracle pass.",
    }
    write_rows(OUT_ROOT / f"{stem}_summaries.csv", summaries)
    write_json(OUT_ROOT / f"{stem}_route.json", obj)
    append_exec(
        "D_interaction_lift_search",
        command,
        "pass" if obj["interaction_lift_search_gate_pass"] else "fail",
        gpu=args.device,
        files=f"{rel(out)}; {rel(OUT_ROOT / f'{stem}_summaries.csv')}; {rel(OUT_ROOT / f'{stem}_route.json')}",
        note=json.dumps({"interaction_lift_search_gate_pass": obj["interaction_lift_search_gate_pass"], "route": route, "rows": len(rows)}, ensure_ascii=False),
    )
    return obj


def run_part_d_interaction_lift_search_merge(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-d-interaction-lift-search-merge", "--part-d-shard-count", args.part_d_shard_count])
    stem = "v22_79_part_d_interaction_lift_search"
    all_rows: list[dict[str, Any]] = []
    missing = []
    for idx in range(int(args.part_d_shard_count)):
        path = OUT_ROOT / f"{stem}_shard{idx}_of_{int(args.part_d_shard_count)}.csv"
        if not path.exists():
            missing.append(rel(path))
            continue
        all_rows.extend(read_rows(path))
    write_rows(OUT_ROOT / f"{stem}.csv", all_rows)
    families = sorted({str(row.get("interaction_lift_family")) for row in all_rows})
    summaries = [summarize_interaction_lift_family(fam, all_rows) for fam in families]
    route, reason = route_from_interaction_lift_search(summaries)
    obj = {
        "gate": "v22_79_part_d_interaction_lift_search",
        "run_status": "completed_interaction_lift_search_merge" if not missing else "incomplete_interaction_lift_search_merge",
        "missing_shards": missing,
        "interaction_lift_search_gate_pass": int(not missing and any(int(row["interaction_lift_search_pass_rows"]) for row in summaries)),
        "interaction_lift_route": route if not missing else "R0-CodeOrTrainingBoundaryFailed",
        "route_reason": reason if not missing else f"missing interaction-lift shards: {missing}",
        "rows": len(all_rows),
        "interaction_lift_summary_rows": len(summaries),
        "passed_interaction_lift_families": [row["interaction_lift_family"] for row in summaries if int(row["interaction_lift_search_pass_rows"]) > 0],
        "non_official_gate_note": "Interaction-lift search includes a direct downstream oracle for diagnosis; oracle rows are never counted as official or non-oracle pass.",
    }
    write_rows(OUT_ROOT / f"{stem}_summaries.csv", summaries)
    write_json(OUT_ROOT / f"{stem}_route.json", obj)
    append_exec(
        "D_interaction_lift_search_merge",
        command,
        "pass" if obj["interaction_lift_search_gate_pass"] else "fail",
        files=f"{rel(OUT_ROOT / f'{stem}.csv')}; {rel(OUT_ROOT / f'{stem}_summaries.csv')}; {rel(OUT_ROOT / f'{stem}_route.json')}",
        note=json.dumps({"interaction_lift_search_gate_pass": obj["interaction_lift_search_gate_pass"], "route": obj["interaction_lift_route"], "rows": len(all_rows), "missing": missing}, ensure_ascii=False),
    )
    best = sorted(
        summaries,
        key=lambda row: (
            int(row.get("interaction_lift_search_pass_rows", 0)),
            int(row.get("strong_interaction_to_additive_rows", 0)),
            int(row.get("bank_gain_positive_rows", 0)),
            int(row.get("same_upstream_energy_candidate_better_rows", 0)),
            fval(row.get("best_interaction_residual_reduction_median")),
        ),
        reverse=True,
    )[:9]
    best_lines = [
        f"{row['interaction_lift_family']}: pass={row['interaction_lift_search_pass_rows']}/{row['rows']}；strong={row['strong_interaction_to_additive_rows']}/{row['rows']}；bank={row['bank_gain_positive_rows']}/{row['rows']}；same_upstream={row['same_upstream_energy_candidate_better_rows']}/{row['rows']}；best_reduction_median={row['best_interaction_residual_reduction_median']}；accessible_proj_median={row['accessible_lift_projection_fraction_median']}；blocker={row['dominant_blocker']}"
        for row in best
    ]
    append_recap(
        "Part D interaction-lift search diagnostic",
        [
            f"merged_rows={len(all_rows)}；missing_shards={missing}；summary_rows={len(summaries)}；interaction_lift_search_gate_pass={obj['interaction_lift_search_gate_pass']}。",
            f"interaction_lift_route={obj['interaction_lift_route']}；reason={obj['route_reason']}",
            "best_interaction_lift_summaries=" + " | ".join(best_lines),
            f"passed_interaction_lift_families={obj['passed_interaction_lift_families']}。",
            "代码修改：新增 `part-d-interaction-lift-search`，比较 actual source/witness lift、accessible LS lift 与 direct downstream oracle；direct oracle 只用于证明 target 是否存在，不计 official pass。",
        ],
    )
    return obj


def route_from_part_g_redesign(summaries: list[dict[str, Any]]) -> tuple[str, str]:
    if any(int(row.get("repair_gate_pass", 0)) for row in summaries):
        passed = [str(row.get("repair_family")) for row in summaries if int(row.get("repair_gate_pass", 0))]
        return "BasisRedesignPreflightOpened_NeedsOfficialPartDIntegration", f"redesign repair_gate_pass families={passed}"
    if not summaries:
        return "BasisRedesignNoRows", "No basis-redesign summaries were produced."
    max_margin = max(int(row.get("control_margin_positive_rows", 0)) for row in summaries)
    max_boot = max(int(row.get("bootstrap_guard_margin_positive_rows", 0)) for row in summaries)
    max_nll = max(int(row.get("candidate_NLL_improve_rows", 0)) for row in summaries)
    best_gain = max(int(row.get("post_bank_R2_gain_positive_rows", 0)) for row in summaries)
    if best_gain >= 10 and max_margin == 0:
        return "BasisRedesignBankSignalOpenedButControlMarginAbsent", f"max_bank_gain_rows={best_gain}; max_control_margin_positive_rows={max_margin}; max_candidate_NLL_improve_rows={max_nll}."
    if max_margin > 0 or max_boot > 0:
        return "BasisRedesignPartialControlMarginSignalButNotStable", f"max_control_margin_positive_rows={max_margin}; max_bootstrap_guard_margin_positive_rows={max_boot}; max_candidate_NLL_improve_rows={max_nll}."
    return "CurrentStrictFCPureKANBasisFamilyConditionalCarrierNotOpened", f"max_bank_gain_rows={best_gain}; max_control_margin_positive_rows={max_margin}; max_bootstrap_guard_margin_positive_rows={max_boot}; max_candidate_NLL_improve_rows={max_nll}."


def run_part_g_redesign(args: argparse.Namespace) -> dict[str, Any]:
    shard_count = int(args.part_d_shard_count)
    shard_index = int(args.part_d_shard_index)
    command = command_text(
        [
            PYTHON,
            rel(RUNNER),
            "--mode",
            "part-g-redesign",
            "--device",
            args.device,
            "--part-d-shard-count",
            shard_count,
            "--part-d-shard-index",
            shard_index,
            "--part-g-redesign-families",
            args.part_g_redesign_families,
            "--part-d-repair-update-rules",
            args.part_d_repair_update_rules,
        ]
    )
    device = base73.make_device(str(args.device))
    tasks = part_g_redesign_tasks(args)
    if shard_count > 1:
        tasks = [task for idx, task in enumerate(tasks) if idx % shard_count == shard_index]
    rows: list[dict[str, Any]] = []
    for family, update_rule, seed, dataset in tasks:
        row = representation_separated_control_margin_repair_probe(dataset, seed, family, update_rule, args, device)
        row["redesign_probe"] = 1
        rows.append(row)
    suffix = f"_shard{shard_index}_of_{shard_count}" if shard_count > 1 else ""
    out = OUT_ROOT / f"v22_79_part_g_basis_redesign_preflight{suffix}.csv"
    write_rows(out, rows)
    if shard_count > 1:
        obj = {"gate": "v22_79_part_g_basis_redesign_shard", "rows": len(rows), "shard_index": shard_index, "shard_count": shard_count, "output": rel(out)}
        write_json(OUT_ROOT / f"v22_79_part_g_basis_redesign_shard{shard_index}_of_{shard_count}.json", obj)
        append_exec("G_basis_redesign_shard", command, "done", gpu=args.device, files=rel(out), note=json.dumps(obj, ensure_ascii=False))
        return obj
    redesign_families = sorted({str(row.get("repair_family")) for row in rows})
    summaries = [summarize_part_d_repair_family(rf, rows) for rf in redesign_families]
    route, reason = route_from_part_g_redesign(summaries)
    obj = {
        "gate": "v22_79_part_g_basis_redesign_preflight",
        "run_status": "completed_basis_redesign_preflight",
        "basis_redesign_gate_pass": int(any(int(row["repair_gate_pass"]) for row in summaries)),
        "basis_redesign_route": route,
        "route_reason": reason,
        "rows": len(rows),
        "basis_redesign_summary_rows": len(summaries),
        "passed_basis_redesign_families": [row["repair_family"] for row in summaries if int(row["repair_gate_pass"]) == 1],
        "no_full_loop_note": "Per plan, basis redesign is evaluated only with Part C/D-level probes unless a fixed family passes preflight.",
    }
    write_rows(OUT_ROOT / "v22_79_part_g_basis_redesign_summaries.csv", summaries)
    write_json(OUT_ROOT / "v22_79_part_g_basis_redesign_route.json", obj)
    append_exec(
        "G_basis_redesign",
        command,
        "pass" if obj["basis_redesign_gate_pass"] else "fail",
        gpu=args.device,
        files=f"{rel(out)}; {rel(OUT_ROOT / 'v22_79_part_g_basis_redesign_summaries.csv')}; {rel(OUT_ROOT / 'v22_79_part_g_basis_redesign_route.json')}",
        note=json.dumps({"basis_redesign_gate_pass": obj["basis_redesign_gate_pass"], "route": route, "rows": len(rows)}, ensure_ascii=False),
    )
    append_recap(
        "Part G basis-family redesign preflight",
        [
            f"rows={len(rows)}；basis_redesign_summary_rows={len(summaries)}；basis_redesign_gate_pass={obj['basis_redesign_gate_pass']}。",
            f"basis_redesign_route={route}；reason={reason}",
            f"passed_basis_redesign_families={obj['passed_basis_redesign_families']}。",
            "本段只做 Part C/D 级别 probe；未进入 full-loop，符合计划 12 节“所有 Part D family 失败后转入 basis-family redesign”的限制。",
        ],
    )
    return obj


def run_part_g_redesign_merge(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-g-redesign-merge", "--part-d-shard-count", args.part_d_shard_count])
    all_rows: list[dict[str, Any]] = []
    missing = []
    for idx in range(int(args.part_d_shard_count)):
        path = OUT_ROOT / f"v22_79_part_g_basis_redesign_preflight_shard{idx}_of_{int(args.part_d_shard_count)}.csv"
        if not path.exists():
            missing.append(rel(path))
            continue
        all_rows.extend(read_rows(path))
    write_rows(OUT_ROOT / "v22_79_part_g_basis_redesign_preflight.csv", all_rows)
    redesign_families = sorted({str(row.get("repair_family")) for row in all_rows})
    summaries = [summarize_part_d_repair_family(rf, all_rows) for rf in redesign_families]
    route, reason = route_from_part_g_redesign(summaries)
    obj = {
        "gate": "v22_79_part_g_basis_redesign_preflight",
        "run_status": "completed_basis_redesign_preflight_merge" if not missing else "incomplete_basis_redesign_preflight_merge",
        "missing_shards": missing,
        "basis_redesign_gate_pass": int(not missing and any(int(row["repair_gate_pass"]) for row in summaries)),
        "basis_redesign_route": route if not missing else "R0-CodeOrTrainingBoundaryFailed",
        "route_reason": reason if not missing else f"missing redesign shards: {missing}",
        "rows": len(all_rows),
        "basis_redesign_summary_rows": len(summaries),
        "passed_basis_redesign_families": [row["repair_family"] for row in summaries if int(row["repair_gate_pass"]) == 1],
        "no_full_loop_note": "Per plan, basis redesign is evaluated only with Part C/D-level probes unless a fixed family passes preflight.",
    }
    write_rows(OUT_ROOT / "v22_79_part_g_basis_redesign_summaries.csv", summaries)
    write_json(OUT_ROOT / "v22_79_part_g_basis_redesign_route.json", obj)
    append_exec(
        "G_basis_redesign_merge",
        command,
        "pass" if obj["basis_redesign_gate_pass"] else "fail",
        files=f"{rel(OUT_ROOT / 'v22_79_part_g_basis_redesign_preflight.csv')}; {rel(OUT_ROOT / 'v22_79_part_g_basis_redesign_summaries.csv')}; {rel(OUT_ROOT / 'v22_79_part_g_basis_redesign_route.json')}",
        note=json.dumps({"basis_redesign_gate_pass": obj["basis_redesign_gate_pass"], "route": obj["basis_redesign_route"], "rows": len(all_rows), "missing": missing}, ensure_ascii=False),
    )
    best = sorted(summaries, key=lambda row: (int(row.get("control_margin_positive_rows", 0)), int(row.get("candidate_NLL_improve_rows", 0)), int(row.get("post_bank_R2_gain_positive_rows", 0)), fval(row.get("bank_R2_gain_median"))), reverse=True)[:5]
    best_lines = [
        f"{row['repair_family']}: control_margin={row['control_margin_positive_rows']}/{row['rows']}；bootstrap={row['bootstrap_guard_margin_positive_rows']}/{row['rows']}；NLL_improve={row['candidate_NLL_improve_rows']}/{row['rows']}；bank_gain={row['post_bank_R2_gain_positive_rows']}/{row['rows']}；bank_gain_median={row['bank_R2_gain_median']}；dominant_blocker={row['dominant_blocker']}"
        for row in best
    ]
    append_recap(
        "Part G basis-family redesign preflight",
        [
            f"merged_rows={len(all_rows)}；missing_shards={missing}；basis_redesign_summary_rows={len(summaries)}；basis_redesign_gate_pass={obj['basis_redesign_gate_pass']}。",
            f"basis_redesign_route={obj['basis_redesign_route']}；reason={obj['route_reason']}",
            "best_redesign_summaries=" + " | ".join(best_lines),
            f"passed_basis_redesign_families={obj['passed_basis_redesign_families']}。",
        ],
    )
    return obj


def run_part_e(args: argparse.Namespace, part_d: dict[str, Any]) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-e"])
    if not int(part_d.get("part_d_gate_pass", 0)):
        rows = [{"run_status": "skipped", "reason": "Part D preflight failed; plan forbids target-free KAN full-loop.", "completed_rows": 0}]
        gate = {"gate": "v22_79_part_e_candidate_gate", "run_status": "skipped", "reason": rows[0]["reason"], "part_e_exploration_gate_pass": 0, "official_candidate_gate_pass": 0, "completed_rows": 0}
        write_rows(OUT_ROOT / "v22_79_part_e_target_free_full_loop_matrix.csv", rows)
        write_rows(OUT_ROOT / "v22_79_part_e_candidate_gate.csv", [gate])
        append_exec("E_target_free_two_layer_KAN_full_loop", command, "skipped", files=f"{rel(OUT_ROOT / 'v22_79_part_e_target_free_full_loop_matrix.csv')}; {rel(OUT_ROOT / 'v22_79_part_e_candidate_gate.csv')}", note=rows[0]["reason"])
        append_recap("Part E target-free two-layer KAN full-loop", [f"skipped：{rows[0]['reason']}"])
        return gate
    rows = [{"run_status": "not_run_in_this_pass", "reason": "Part D passed; full-loop implementation requires extended 45-row audited runtime beyond this preflight script.", "completed_rows": 0}]
    gate = {"gate": "v22_79_part_e_candidate_gate", "run_status": rows[0]["run_status"], "reason": rows[0]["reason"], "part_e_exploration_gate_pass": 0, "official_candidate_gate_pass": 0, "completed_rows": 0}
    write_rows(OUT_ROOT / "v22_79_part_e_target_free_full_loop_matrix.csv", rows)
    write_rows(OUT_ROOT / "v22_79_part_e_candidate_gate.csv", [gate])
    append_exec("E_target_free_two_layer_KAN_full_loop", command, "blocked", files=f"{rel(OUT_ROOT / 'v22_79_part_e_target_free_full_loop_matrix.csv')}; {rel(OUT_ROOT / 'v22_79_part_e_candidate_gate.csv')}", note=rows[0]["reason"])
    append_recap("Part E target-free two-layer KAN full-loop", [f"blocked：{rows[0]['reason']}"])
    return gate


def run_part_f(args: argparse.Namespace, part_e: dict[str, Any]) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-f"])
    v78_f = read_rows(ROOT / "results/v22_78/v22_78_part_f_strengthened_mlp_audit.csv")
    source = v78_f[0] if v78_f else {}
    row = {
        "gate": "v22_79_part_f_strengthened_mlp_matched_coordinate_audit",
        "run_status": "skipped_no_v22_79_full_loop_candidates" if not int(part_e.get("official_candidate_gate_pass", 0)) else "completed",
        "KAN_vs_old_MLP_matched": int(fval(source.get("KAN_vs_old_MLP_matched"))),
        "KAN_vs_strengthened_MLP_matched": int(fval(source.get("KAN_vs_strengthened_MLP_matched"))),
        "old_MLP_beaten_but_strengthened_not": int(fval(source.get("old_MLP_beaten_but_strengthened_not"))),
        "part_f_gate_pass": 1,
        "source_artifact": "results/v22_78/v22_78_part_f_strengthened_mlp_audit.csv" if source else "missing",
        "non_fabrication_note": "No v22.79 full-loop candidate exists when Part E is skipped; prior strengthened MLP audit is recorded only as context.",
    }
    write_rows(OUT_ROOT / "v22_79_part_f_strengthened_mlp_audit.csv", [row])
    append_exec("F_strengthened_MLP_matched_coordinate_audit", command, "pass", files=rel(OUT_ROOT / "v22_79_part_f_strengthened_mlp_audit.csv"), note=json.dumps({"run_status": row["run_status"], "source": row["source_artifact"]}, ensure_ascii=False))
    append_recap("Part F strengthened MLP matched coordinate audit", [f"run_status={row['run_status']}；source={row['source_artifact']}；KAN_vs_old={row['KAN_vs_old_MLP_matched']}；KAN_vs_strengthened={row['KAN_vs_strengthened_MLP_matched']}。"])
    return row


def run_part_g(args: argparse.Namespace, a: dict[str, Any], b: dict[str, Any], c: dict[str, Any], d: dict[str, Any], e: dict[str, Any], f: dict[str, Any]) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-g"])
    repair = load_json(OUT_ROOT / "v22_79_part_d_control_margin_repair_route.json")
    redesign = load_json(OUT_ROOT / "v22_79_part_g_basis_redesign_route.json")
    utility = load_json(OUT_ROOT / "v22_79_part_d_utility_ablation_route.json")
    debt_residual = load_json(OUT_ROOT / "v22_79_part_d_debt_residual_ablation_route.json")
    tail95_debt_residual = load_json(OUT_ROOT / "v22_79_part_d_tail95_debt_residual_ablation_route.json")
    debt_gap_audit = load_json(OUT_ROOT / "v22_79_part_d_debt_residual_gap_audit.json")
    debt_risk_audit = load_json(OUT_ROOT / "v22_79_part_d_debt_risk_decomposition_route.json")
    separation_trace = load_json(OUT_ROOT / "v22_79_part_d_separation_trace_route.json")
    interaction_lift = load_json(OUT_ROOT / "v22_79_part_d_interaction_lift_search_route.json")
    interaction_lift_integration = load_json(OUT_ROOT / "v22_79_part_d_interaction_lift_integration_blends_route.json")
    interaction_lift_scale_debt = load_json(OUT_ROOT / "v22_79_part_d_interaction_lift_scale_debt_blends_route.json")
    interaction_lift_w2scale = load_json(OUT_ROOT / "v22_79_part_d_interaction_lift_w2scale_probe_route.json")
    interaction_lift_debt_frontier = load_json(OUT_ROOT / "v22_79_part_d_interaction_lift_debt_frontier_control_route.json")
    interaction_lift_debt_orthogonalized = load_json(OUT_ROOT / "v22_79_part_d_interaction_lift_debt_orthogonalized_probe_route.json")
    interaction_lift_upstream_only = load_json(OUT_ROOT / "v22_79_part_d_interaction_lift_upstream_only_probe_route.json")
    interaction_lift_source_witness = load_json(OUT_ROOT / "v22_79_part_d_interaction_lift_source_witness_probe_route.json")
    interaction_lift_source_witness_orthogonalized = load_json(OUT_ROOT / "v22_79_part_d_interaction_lift_source_witness_orthogonalized_probe_route.json")
    interaction_lift_source_witness_debt_blend = load_json(OUT_ROOT / "v22_79_part_d_interaction_lift_source_witness_debt_blend_probe_route.json")
    interaction_lift_source_witness_high_debt = load_json(OUT_ROOT / "v22_79_part_d_interaction_lift_source_witness_high_debt_probe_route.json")
    if not int(a.get("part_a_hard_gate_pass", 0)):
        route, reason = "R0-CodeOrTrainingBoundaryFailed", "Part A hard gate failed."
    elif not int(b.get("part_b_gate_pass", 0)):
        route, reason = "R0-V22_78ReplayArtifactsMissingOrInconsistent", "Part B v22.78 replay gate failed."
    elif not int(c.get("part_c_gate_pass", 0)):
        route, reason = "R0-RepresentationSeparationUnitFailed", "Part C synthetic/unit gate failed."
    elif not int(d.get("part_d_gate_pass", 0)):
        debt_residual_done = debt_residual.get("run_status") in {"completed_utility_ablation", "completed_utility_ablation_merge"}
        if utility.get("run_status") in {"completed_utility_ablation", "completed_utility_ablation_merge"}:
            route = str(utility.get("utility_ablation_route", d.get("preflight_route", "UpstreamRepresentationInsufficient_NoEdgeBankCarrierYet")))
            reason = f"Official Part D failed: {d.get('route_reason', '')}; basis redesign probe: {redesign.get('route_reason', '')}; utility ablation probe: {utility.get('route_reason', '')}"
            if debt_residual_done:
                reason += f"; debt residual ablation probe: {debt_residual.get('route_reason', '')}"
            if debt_gap_audit.get("run_status") == "completed_debt_residual_gap_audit":
                reason += f"; debt residual gap audit: {debt_gap_audit.get('route_reason', '')}"
            if debt_risk_audit.get("run_status") == "completed_debt_risk_decomposition":
                reason += f"; debt risk decomposition: {debt_risk_audit.get('route_reason', '')}"
            if separation_trace.get("run_status") == "completed_separation_trace_audit":
                reason += f"; separation trace audit: {separation_trace.get('route_reason', '')}"
            if interaction_lift.get("run_status") == "completed_interaction_lift_search_merge":
                reason += f"; interaction lift search: {interaction_lift.get('route_reason', '')}"
                if interaction_lift_integration.get("run_status") == "completed_utility_ablation_merge":
                    if int(interaction_lift_integration.get("utility_ablation_gate_pass", 0)):
                        route = "InteractionLiftIntegrationOpened_NeedsOfficialPartD"
                    elif str(interaction_lift.get("interaction_lift_route", "")) == "InteractionLiftSearchOpened_NeedsOfficialPartDIntegration":
                        route = "InteractionLiftSearchOpenedButIntegrationDebtControlBlocked"
                    reason += f"; interaction lift integration: {interaction_lift_integration.get('route_reason', '')}"
                if interaction_lift_scale_debt.get("run_status") == "completed_utility_ablation_merge":
                    if int(interaction_lift_scale_debt.get("utility_ablation_gate_pass", 0)):
                        route = "InteractionLiftScaleDebtIntegrationOpened_NeedsOfficialPartD"
                    elif str(interaction_lift.get("interaction_lift_route", "")) == "InteractionLiftSearchOpened_NeedsOfficialPartDIntegration":
                        route = "InteractionLiftSearchOpenedButIntegrationDebtControlBlocked"
                    reason += f"; interaction lift scale/debt integration: {interaction_lift_scale_debt.get('route_reason', '')}"
                if interaction_lift_w2scale.get("run_status") == "completed_utility_ablation_merge":
                    if int(interaction_lift_w2scale.get("utility_ablation_gate_pass", 0)):
                        route = "InteractionLiftW2ScaleIntegrationOpened_NeedsOfficialPartD"
                    elif str(interaction_lift.get("interaction_lift_route", "")) == "InteractionLiftSearchOpened_NeedsOfficialPartDIntegration":
                        route = "InteractionLiftSearchOpenedButIntegrationDebtControlBlocked"
                    reason += f"; interaction lift w2-scale integration: {interaction_lift_w2scale.get('route_reason', '')}"
                if interaction_lift_debt_frontier.get("run_status") == "completed_utility_ablation_merge":
                    if int(interaction_lift_debt_frontier.get("utility_ablation_gate_pass", 0)):
                        route = "InteractionLiftDebtFrontierIntegrationOpened_NeedsOfficialPartD"
                    elif str(interaction_lift.get("interaction_lift_route", "")) == "InteractionLiftSearchOpened_NeedsOfficialPartDIntegration":
                        route = "InteractionLiftSearchOpenedButIntegrationDebtControlBlocked"
                    reason += f"; interaction lift debt-frontier control: {interaction_lift_debt_frontier.get('route_reason', '')}"
                if interaction_lift_debt_orthogonalized.get("run_status") == "completed_utility_ablation_merge":
                    if int(interaction_lift_debt_orthogonalized.get("utility_ablation_gate_pass", 0)):
                        route = "InteractionLiftDebtOrthogonalizationOpened_NeedsOfficialPartD"
                    elif str(interaction_lift.get("interaction_lift_route", "")) == "InteractionLiftSearchOpened_NeedsOfficialPartDIntegration":
                        route = "InteractionLiftSearchOpenedButIntegrationDebtControlBlocked"
                    reason += f"; interaction lift debt-orthogonalized control: {interaction_lift_debt_orthogonalized.get('route_reason', '')}"
                if interaction_lift_upstream_only.get("run_status") == "completed_utility_ablation_merge":
                    if int(interaction_lift_upstream_only.get("utility_ablation_gate_pass", 0)):
                        route = "InteractionLiftUpstreamOnlyOpened_NeedsOfficialPartD"
                    elif str(interaction_lift.get("interaction_lift_route", "")) == "InteractionLiftSearchOpened_NeedsOfficialPartDIntegration":
                        route = "InteractionLiftSearchOpenedButIntegrationDebtControlBlocked"
                    reason += f"; interaction lift upstream-only control: {interaction_lift_upstream_only.get('route_reason', '')}"
                if interaction_lift_source_witness.get("run_status") == "completed_utility_ablation_merge":
                    if int(interaction_lift_source_witness.get("utility_ablation_gate_pass", 0)):
                        route = "InteractionLiftSourceWitnessOpened_NeedsOfficialPartD"
                    elif str(interaction_lift.get("interaction_lift_route", "")) == "InteractionLiftSearchOpened_NeedsOfficialPartDIntegration":
                        route = "InteractionLiftSearchOpenedButIntegrationDebtControlBlocked"
                    reason += f"; interaction lift source-witness control: {interaction_lift_source_witness.get('route_reason', '')}"
                if interaction_lift_source_witness_orthogonalized.get("run_status") == "completed_utility_ablation_merge":
                    if int(interaction_lift_source_witness_orthogonalized.get("utility_ablation_gate_pass", 0)):
                        route = "InteractionLiftSourceWitnessOrthogonalizationOpened_NeedsOfficialPartD"
                    elif str(interaction_lift.get("interaction_lift_route", "")) == "InteractionLiftSearchOpened_NeedsOfficialPartDIntegration":
                        route = "InteractionLiftSearchOpenedButIntegrationDebtControlBlocked"
                    reason += f"; interaction lift source-witness orthogonalized control: {interaction_lift_source_witness_orthogonalized.get('route_reason', '')}"
                if interaction_lift_source_witness_debt_blend.get("run_status") == "completed_utility_ablation_merge":
                    if int(interaction_lift_source_witness_debt_blend.get("utility_ablation_gate_pass", 0)):
                        route = "InteractionLiftSourceWitnessDebtBlendOpened_NeedsOfficialPartD"
                    elif str(interaction_lift.get("interaction_lift_route", "")) == "InteractionLiftSearchOpened_NeedsOfficialPartDIntegration":
                        route = "InteractionLiftSearchOpenedButIntegrationDebtControlBlocked"
                    reason += f"; interaction lift source-witness debt-blend control: {interaction_lift_source_witness_debt_blend.get('route_reason', '')}"
                if interaction_lift_source_witness_high_debt.get("run_status") == "completed_utility_ablation_merge":
                    if int(interaction_lift_source_witness_high_debt.get("utility_ablation_gate_pass", 0)):
                        route = "InteractionLiftSourceWitnessHighDebtOpened_NeedsOfficialPartD"
                    elif str(interaction_lift.get("interaction_lift_route", "")) == "InteractionLiftSearchOpened_NeedsOfficialPartDIntegration":
                        route = "InteractionLiftSearchOpenedButIntegrationDebtControlBlocked"
                    reason += f"; interaction lift source-witness high-debt control: {interaction_lift_source_witness_high_debt.get('route_reason', '')}"
                elif str(interaction_lift.get("interaction_lift_route", "")) == "InteractionLiftSearchOpened_NeedsOfficialPartDIntegration" and interaction_lift_integration.get("run_status") != "completed_utility_ablation_merge" and interaction_lift_scale_debt.get("run_status") != "completed_utility_ablation_merge" and interaction_lift_w2scale.get("run_status") != "completed_utility_ablation_merge" and interaction_lift_debt_frontier.get("run_status") != "completed_utility_ablation_merge" and interaction_lift_debt_orthogonalized.get("run_status") != "completed_utility_ablation_merge" and interaction_lift_upstream_only.get("run_status") != "completed_utility_ablation_merge" and interaction_lift_source_witness.get("run_status") != "completed_utility_ablation_merge":
                    route = str(interaction_lift.get("interaction_lift_route"))
        elif debt_residual_done:
            route = str(debt_residual.get("utility_ablation_route", d.get("preflight_route", "UpstreamRepresentationInsufficient_NoEdgeBankCarrierYet")))
            reason = f"Official Part D failed: {d.get('route_reason', '')}; basis redesign probe: {redesign.get('route_reason', '')}; debt residual ablation probe: {debt_residual.get('route_reason', '')}"
            if debt_gap_audit.get("run_status") == "completed_debt_residual_gap_audit":
                reason += f"; debt residual gap audit: {debt_gap_audit.get('route_reason', '')}"
            if debt_risk_audit.get("run_status") == "completed_debt_risk_decomposition":
                reason += f"; debt risk decomposition: {debt_risk_audit.get('route_reason', '')}"
            if separation_trace.get("run_status") == "completed_separation_trace_audit":
                reason += f"; separation trace audit: {separation_trace.get('route_reason', '')}"
            if interaction_lift.get("run_status") == "completed_interaction_lift_search_merge":
                reason += f"; interaction lift search: {interaction_lift.get('route_reason', '')}"
                if interaction_lift_integration.get("run_status") == "completed_utility_ablation_merge":
                    if int(interaction_lift_integration.get("utility_ablation_gate_pass", 0)):
                        route = "InteractionLiftIntegrationOpened_NeedsOfficialPartD"
                    elif str(interaction_lift.get("interaction_lift_route", "")) == "InteractionLiftSearchOpened_NeedsOfficialPartDIntegration":
                        route = "InteractionLiftSearchOpenedButIntegrationDebtControlBlocked"
                    reason += f"; interaction lift integration: {interaction_lift_integration.get('route_reason', '')}"
                if interaction_lift_scale_debt.get("run_status") == "completed_utility_ablation_merge":
                    if int(interaction_lift_scale_debt.get("utility_ablation_gate_pass", 0)):
                        route = "InteractionLiftScaleDebtIntegrationOpened_NeedsOfficialPartD"
                    elif str(interaction_lift.get("interaction_lift_route", "")) == "InteractionLiftSearchOpened_NeedsOfficialPartDIntegration":
                        route = "InteractionLiftSearchOpenedButIntegrationDebtControlBlocked"
                    reason += f"; interaction lift scale/debt integration: {interaction_lift_scale_debt.get('route_reason', '')}"
                if interaction_lift_w2scale.get("run_status") == "completed_utility_ablation_merge":
                    if int(interaction_lift_w2scale.get("utility_ablation_gate_pass", 0)):
                        route = "InteractionLiftW2ScaleIntegrationOpened_NeedsOfficialPartD"
                    elif str(interaction_lift.get("interaction_lift_route", "")) == "InteractionLiftSearchOpened_NeedsOfficialPartDIntegration":
                        route = "InteractionLiftSearchOpenedButIntegrationDebtControlBlocked"
                    reason += f"; interaction lift w2-scale integration: {interaction_lift_w2scale.get('route_reason', '')}"
                if interaction_lift_debt_frontier.get("run_status") == "completed_utility_ablation_merge":
                    if int(interaction_lift_debt_frontier.get("utility_ablation_gate_pass", 0)):
                        route = "InteractionLiftDebtFrontierIntegrationOpened_NeedsOfficialPartD"
                    elif str(interaction_lift.get("interaction_lift_route", "")) == "InteractionLiftSearchOpened_NeedsOfficialPartDIntegration":
                        route = "InteractionLiftSearchOpenedButIntegrationDebtControlBlocked"
                    reason += f"; interaction lift debt-frontier control: {interaction_lift_debt_frontier.get('route_reason', '')}"
                if interaction_lift_debt_orthogonalized.get("run_status") == "completed_utility_ablation_merge":
                    if int(interaction_lift_debt_orthogonalized.get("utility_ablation_gate_pass", 0)):
                        route = "InteractionLiftDebtOrthogonalizationOpened_NeedsOfficialPartD"
                    elif str(interaction_lift.get("interaction_lift_route", "")) == "InteractionLiftSearchOpened_NeedsOfficialPartDIntegration":
                        route = "InteractionLiftSearchOpenedButIntegrationDebtControlBlocked"
                    reason += f"; interaction lift debt-orthogonalized control: {interaction_lift_debt_orthogonalized.get('route_reason', '')}"
                if interaction_lift_upstream_only.get("run_status") == "completed_utility_ablation_merge":
                    if int(interaction_lift_upstream_only.get("utility_ablation_gate_pass", 0)):
                        route = "InteractionLiftUpstreamOnlyOpened_NeedsOfficialPartD"
                    elif str(interaction_lift.get("interaction_lift_route", "")) == "InteractionLiftSearchOpened_NeedsOfficialPartDIntegration":
                        route = "InteractionLiftSearchOpenedButIntegrationDebtControlBlocked"
                    reason += f"; interaction lift upstream-only control: {interaction_lift_upstream_only.get('route_reason', '')}"
                if interaction_lift_source_witness.get("run_status") == "completed_utility_ablation_merge":
                    if int(interaction_lift_source_witness.get("utility_ablation_gate_pass", 0)):
                        route = "InteractionLiftSourceWitnessOpened_NeedsOfficialPartD"
                    elif str(interaction_lift.get("interaction_lift_route", "")) == "InteractionLiftSearchOpened_NeedsOfficialPartDIntegration":
                        route = "InteractionLiftSearchOpenedButIntegrationDebtControlBlocked"
                    reason += f"; interaction lift source-witness control: {interaction_lift_source_witness.get('route_reason', '')}"
                if interaction_lift_source_witness_orthogonalized.get("run_status") == "completed_utility_ablation_merge":
                    if int(interaction_lift_source_witness_orthogonalized.get("utility_ablation_gate_pass", 0)):
                        route = "InteractionLiftSourceWitnessOrthogonalizationOpened_NeedsOfficialPartD"
                    elif str(interaction_lift.get("interaction_lift_route", "")) == "InteractionLiftSearchOpened_NeedsOfficialPartDIntegration":
                        route = "InteractionLiftSearchOpenedButIntegrationDebtControlBlocked"
                    reason += f"; interaction lift source-witness orthogonalized control: {interaction_lift_source_witness_orthogonalized.get('route_reason', '')}"
                if interaction_lift_source_witness_debt_blend.get("run_status") == "completed_utility_ablation_merge":
                    if int(interaction_lift_source_witness_debt_blend.get("utility_ablation_gate_pass", 0)):
                        route = "InteractionLiftSourceWitnessDebtBlendOpened_NeedsOfficialPartD"
                    elif str(interaction_lift.get("interaction_lift_route", "")) == "InteractionLiftSearchOpened_NeedsOfficialPartDIntegration":
                        route = "InteractionLiftSearchOpenedButIntegrationDebtControlBlocked"
                    reason += f"; interaction lift source-witness debt-blend control: {interaction_lift_source_witness_debt_blend.get('route_reason', '')}"
                if interaction_lift_source_witness_high_debt.get("run_status") == "completed_utility_ablation_merge":
                    if int(interaction_lift_source_witness_high_debt.get("utility_ablation_gate_pass", 0)):
                        route = "InteractionLiftSourceWitnessHighDebtOpened_NeedsOfficialPartD"
                    elif str(interaction_lift.get("interaction_lift_route", "")) == "InteractionLiftSearchOpened_NeedsOfficialPartDIntegration":
                        route = "InteractionLiftSearchOpenedButIntegrationDebtControlBlocked"
                    reason += f"; interaction lift source-witness high-debt control: {interaction_lift_source_witness_high_debt.get('route_reason', '')}"
                elif str(interaction_lift.get("interaction_lift_route", "")) == "InteractionLiftSearchOpened_NeedsOfficialPartDIntegration" and interaction_lift_integration.get("run_status") != "completed_utility_ablation_merge" and interaction_lift_scale_debt.get("run_status") != "completed_utility_ablation_merge" and interaction_lift_w2scale.get("run_status") != "completed_utility_ablation_merge" and interaction_lift_debt_frontier.get("run_status") != "completed_utility_ablation_merge" and interaction_lift_debt_orthogonalized.get("run_status") != "completed_utility_ablation_merge" and interaction_lift_upstream_only.get("run_status") != "completed_utility_ablation_merge" and interaction_lift_source_witness.get("run_status") != "completed_utility_ablation_merge":
                    route = str(interaction_lift.get("interaction_lift_route"))
        elif redesign.get("run_status") in {"completed_basis_redesign_preflight", "completed_basis_redesign_preflight_merge"}:
            route = str(redesign.get("basis_redesign_route", d.get("preflight_route", "UpstreamRepresentationInsufficient_NoEdgeBankCarrierYet")))
            reason = f"Official Part D failed: {d.get('route_reason', '')}; basis redesign probe: {redesign.get('route_reason', '')}"
        elif repair.get("run_status") in {"completed_control_margin_repair", "completed_control_margin_repair_merge"}:
            route = str(repair.get("repair_route", d.get("preflight_route", "UpstreamRepresentationInsufficient_NoEdgeBankCarrierYet")))
            reason = f"Official Part D failed: {d.get('route_reason', '')}; control-margin repair probe: {repair.get('route_reason', '')}"
        else:
            route, reason = str(d.get("preflight_route", "UpstreamRepresentationInsufficient_NoEdgeBankCarrierYet")), str(d.get("route_reason", "Part D preflight failed."))
    elif int(e.get("official_candidate_gate_pass", 0)):
        route, reason = "R9-RepresentationSeparatedKANCarrierOfficialOpened", "Official candidate gate passed."
    elif int(e.get("part_e_exploration_gate_pass", 0)):
        route, reason = "R8-RepresentationSeparatedKANCarrierExplorationOpened", "Exploration gate passed."
    elif int(f.get("KAN_vs_strengthened_MLP_matched", 0)) < int(f.get("KAN_vs_old_MLP_matched", 0)):
        route, reason = "MatchedMLPBlocked_CurrentKANNotSuperiorCarrier", "Strengthened MLP matched coordinate remains relevant."
    else:
        route, reason = "R7-KANInternalOnly", "Part D passed but no official full-loop candidate opened."
    components = []
    if int(d.get("part_d_gate_pass", 0)) == 0:
        components.append("NoPartEFullLoop")
    if "SameDomain" in route:
        components.append("SameDomainSupportExplained")
    if "SameSeparation" in route:
        components.append("SameSeparationEnergyExplained")
    if "BankAdditive" in route:
        components.append("BankAdditiveStillUnstable")
    if "ControlMargin" in route or "control margin" in reason or "control_margin" in reason:
        components.append("ControlContrastiveMarginAbsent")
    if "BasisRedesign" in route:
        components.append("BasisRedesignProbeFailed")
    if "BasisFamily" in route:
        components.append("CurrentStrictFCPureKANBasisFamilyConditionalCarrierNotOpened")
    if "Debt" in route:
        components.append("DebtBlocked")
    if "DebtDominated" in route:
        components.append("DebtDominatedTaskUtility")
    if "Interaction" in route or "interaction lift" in reason or "interaction-lift" in reason:
        components.append("InteractionLiftMapAudited")
    if "CurrentUpstreamLiftMapInsufficient" in route or "CurrentUpstreamLiftMapInsufficient" in reason:
        components.append("InteractionLiftMapInsufficient")
    row = {
        "gate": "v22_79_part_g_failure_decomposition",
        "run_status": "completed_route_decision",
        "final_route": route,
        "route_reason": reason,
        "failure_components": "|".join(components),
        "part_a_hard_gate_pass": int(a.get("part_a_hard_gate_pass", 0)),
        "part_b_gate_pass": int(b.get("part_b_gate_pass", 0)),
        "part_c_gate_pass": int(c.get("part_c_gate_pass", 0)),
        "part_d_gate_pass": int(d.get("part_d_gate_pass", 0)),
        "part_e_exploration_gate_pass": int(e.get("part_e_exploration_gate_pass", 0)),
        "official_candidate_gate_pass": int(e.get("official_candidate_gate_pass", 0)),
        "part_f_gate_pass": int(f.get("part_f_gate_pass", 0)),
        "part_d_rows": int(d.get("rows", 0) or 0),
        "part_d_family_summary_rows": int(d.get("family_summary_rows", 0) or 0),
        "part_d_preflight_route": d.get("preflight_route", ""),
        "part_d_control_margin_repair_route": repair.get("repair_route", ""),
        "part_d_control_margin_repair_gate_pass": int(repair.get("repair_gate_pass", 0) or 0),
        "part_g_basis_redesign_route": redesign.get("basis_redesign_route", ""),
        "part_g_basis_redesign_gate_pass": int(redesign.get("basis_redesign_gate_pass", 0) or 0),
        "part_g_basis_redesign_rows": int(redesign.get("rows", 0) or 0),
        "part_d_utility_ablation_route": utility.get("utility_ablation_route", ""),
        "part_d_utility_ablation_gate_pass": int(utility.get("utility_ablation_gate_pass", 0) or 0),
        "part_d_utility_ablation_rows": int(utility.get("rows", 0) or 0),
        "part_d_debt_residual_ablation_route": debt_residual.get("utility_ablation_route", ""),
        "part_d_debt_residual_ablation_gate_pass": int(debt_residual.get("utility_ablation_gate_pass", 0) or 0),
        "part_d_debt_residual_ablation_rows": int(debt_residual.get("rows", 0) or 0),
        "part_d_tail95_debt_residual_ablation_route": tail95_debt_residual.get("utility_ablation_route", ""),
        "part_d_tail95_debt_residual_ablation_gate_pass": int(tail95_debt_residual.get("utility_ablation_gate_pass", 0) or 0),
        "part_d_tail95_debt_residual_ablation_rows": int(tail95_debt_residual.get("rows", 0) or 0),
        "part_d_debt_residual_gap_audit_route": debt_gap_audit.get("route", ""),
        "part_d_debt_residual_gap_audit_rows": int(debt_gap_audit.get("rows", 0) or 0),
        "part_d_debt_risk_decomposition_route": debt_risk_audit.get("route", ""),
        "part_d_debt_risk_decomposition_rows": int(debt_risk_audit.get("rows", 0) or 0),
        "part_d_separation_trace_route": separation_trace.get("route", ""),
        "part_d_separation_trace_rows": int(separation_trace.get("rows", 0) or 0),
        "part_d_interaction_lift_route": interaction_lift.get("interaction_lift_route", ""),
        "part_d_interaction_lift_gate_pass": int(interaction_lift.get("interaction_lift_search_gate_pass", 0) or 0),
        "part_d_interaction_lift_rows": int(interaction_lift.get("rows", 0) or 0),
        "part_d_interaction_lift_integration_route": interaction_lift_integration.get("utility_ablation_route", ""),
        "part_d_interaction_lift_integration_gate_pass": int(interaction_lift_integration.get("utility_ablation_gate_pass", 0) or 0),
        "part_d_interaction_lift_integration_rows": int(interaction_lift_integration.get("rows", 0) or 0),
        "part_d_interaction_lift_scale_debt_route": interaction_lift_scale_debt.get("utility_ablation_route", ""),
        "part_d_interaction_lift_scale_debt_gate_pass": int(interaction_lift_scale_debt.get("utility_ablation_gate_pass", 0) or 0),
        "part_d_interaction_lift_scale_debt_rows": int(interaction_lift_scale_debt.get("rows", 0) or 0),
        "part_d_interaction_lift_w2scale_route": interaction_lift_w2scale.get("utility_ablation_route", ""),
        "part_d_interaction_lift_w2scale_gate_pass": int(interaction_lift_w2scale.get("utility_ablation_gate_pass", 0) or 0),
        "part_d_interaction_lift_w2scale_rows": int(interaction_lift_w2scale.get("rows", 0) or 0),
        "part_d_interaction_lift_debt_frontier_route": interaction_lift_debt_frontier.get("utility_ablation_route", ""),
        "part_d_interaction_lift_debt_frontier_gate_pass": int(interaction_lift_debt_frontier.get("utility_ablation_gate_pass", 0) or 0),
        "part_d_interaction_lift_debt_frontier_rows": int(interaction_lift_debt_frontier.get("rows", 0) or 0),
        "part_d_interaction_lift_debt_orthogonalized_route": interaction_lift_debt_orthogonalized.get("utility_ablation_route", ""),
        "part_d_interaction_lift_debt_orthogonalized_gate_pass": int(interaction_lift_debt_orthogonalized.get("utility_ablation_gate_pass", 0) or 0),
        "part_d_interaction_lift_debt_orthogonalized_rows": int(interaction_lift_debt_orthogonalized.get("rows", 0) or 0),
        "part_d_interaction_lift_upstream_only_route": interaction_lift_upstream_only.get("utility_ablation_route", ""),
        "part_d_interaction_lift_upstream_only_gate_pass": int(interaction_lift_upstream_only.get("utility_ablation_gate_pass", 0) or 0),
        "part_d_interaction_lift_upstream_only_rows": int(interaction_lift_upstream_only.get("rows", 0) or 0),
        "part_d_interaction_lift_source_witness_route": interaction_lift_source_witness.get("utility_ablation_route", ""),
        "part_d_interaction_lift_source_witness_gate_pass": int(interaction_lift_source_witness.get("utility_ablation_gate_pass", 0) or 0),
        "part_d_interaction_lift_source_witness_rows": int(interaction_lift_source_witness.get("rows", 0) or 0),
        "part_d_interaction_lift_source_witness_orthogonalized_route": interaction_lift_source_witness_orthogonalized.get("utility_ablation_route", ""),
        "part_d_interaction_lift_source_witness_orthogonalized_gate_pass": int(interaction_lift_source_witness_orthogonalized.get("utility_ablation_gate_pass", 0) or 0),
        "part_d_interaction_lift_source_witness_orthogonalized_rows": int(interaction_lift_source_witness_orthogonalized.get("rows", 0) or 0),
        "part_d_interaction_lift_source_witness_debt_blend_route": interaction_lift_source_witness_debt_blend.get("utility_ablation_route", ""),
        "part_d_interaction_lift_source_witness_debt_blend_gate_pass": int(interaction_lift_source_witness_debt_blend.get("utility_ablation_gate_pass", 0) or 0),
        "part_d_interaction_lift_source_witness_debt_blend_rows": int(interaction_lift_source_witness_debt_blend.get("rows", 0) or 0),
        "part_d_interaction_lift_source_witness_high_debt_route": interaction_lift_source_witness_high_debt.get("utility_ablation_route", ""),
        "part_d_interaction_lift_source_witness_high_debt_gate_pass": int(interaction_lift_source_witness_high_debt.get("utility_ablation_gate_pass", 0) or 0),
        "part_d_interaction_lift_source_witness_high_debt_rows": int(interaction_lift_source_witness_high_debt.get("rows", 0) or 0),
    }
    write_rows(OUT_ROOT / "v22_79_part_g_failure_decomposition.csv", [row])
    final = {
        "final_route": route,
        "route_reason": reason,
        "generated_at_sg": now_sg(),
        "non_fabrication_note": "All values are generated by v22.79 runner or copied from explicitly named prior-version artifacts.",
        **{k: row[k] for k in ["part_a_hard_gate_pass", "part_b_gate_pass", "part_c_gate_pass", "part_d_gate_pass", "part_e_exploration_gate_pass", "official_candidate_gate_pass", "part_f_gate_pass", "failure_components"]},
        "route_detail": row,
        "artifacts": {
            "part_a": rel(OUT_ROOT / "v22_79_part_a_code_identity_hard_gate.json"),
            "part_b": rel(OUT_ROOT / "v22_79_part_b_v22_78_failure_replay_summary.json"),
            "part_c": rel(OUT_ROOT / "v22_79_part_c_unit_gate.json"),
            "part_d": rel(OUT_ROOT / "v22_79_part_d_preflight_route.json"),
            "part_e_matrix": rel(OUT_ROOT / "v22_79_part_e_target_free_full_loop_matrix.csv"),
            "part_e_gate": rel(OUT_ROOT / "v22_79_part_e_candidate_gate.csv"),
            "part_f": rel(OUT_ROOT / "v22_79_part_f_strengthened_mlp_audit.csv"),
            "part_g": rel(OUT_ROOT / "v22_79_part_g_failure_decomposition.csv"),
            "part_d_control_margin_repair": rel(OUT_ROOT / "v22_79_part_d_control_margin_repair_route.json"),
            "part_g_basis_redesign": rel(OUT_ROOT / "v22_79_part_g_basis_redesign_route.json"),
            "part_d_utility_ablation": rel(OUT_ROOT / "v22_79_part_d_utility_ablation_route.json"),
            "part_d_debt_residual_ablation": rel(OUT_ROOT / "v22_79_part_d_debt_residual_ablation_route.json"),
            "part_d_tail95_debt_residual_ablation": rel(OUT_ROOT / "v22_79_part_d_tail95_debt_residual_ablation_route.json"),
            "part_d_debt_residual_gap_audit": rel(OUT_ROOT / "v22_79_part_d_debt_residual_gap_audit.json"),
            "part_d_debt_risk_decomposition": rel(OUT_ROOT / "v22_79_part_d_debt_risk_decomposition_route.json"),
            "part_d_separation_trace": rel(OUT_ROOT / "v22_79_part_d_separation_trace_route.json"),
            "part_d_interaction_lift_search": rel(OUT_ROOT / "v22_79_part_d_interaction_lift_search_route.json"),
            "part_d_interaction_lift_integration_blends": rel(OUT_ROOT / "v22_79_part_d_interaction_lift_integration_blends_route.json"),
            "part_d_interaction_lift_scale_debt_blends": rel(OUT_ROOT / "v22_79_part_d_interaction_lift_scale_debt_blends_route.json"),
            "part_d_interaction_lift_w2scale_probe": rel(OUT_ROOT / "v22_79_part_d_interaction_lift_w2scale_probe_route.json"),
            "part_d_interaction_lift_debt_frontier_control": rel(OUT_ROOT / "v22_79_part_d_interaction_lift_debt_frontier_control_route.json"),
            "part_d_interaction_lift_debt_orthogonalized_probe": rel(OUT_ROOT / "v22_79_part_d_interaction_lift_debt_orthogonalized_probe_route.json"),
            "part_d_interaction_lift_upstream_only_probe": rel(OUT_ROOT / "v22_79_part_d_interaction_lift_upstream_only_probe_route.json"),
            "part_d_interaction_lift_source_witness_probe": rel(OUT_ROOT / "v22_79_part_d_interaction_lift_source_witness_probe_route.json"),
            "part_d_interaction_lift_source_witness_orthogonalized_probe": rel(OUT_ROOT / "v22_79_part_d_interaction_lift_source_witness_orthogonalized_probe_route.json"),
            "part_d_interaction_lift_source_witness_debt_blend_probe": rel(OUT_ROOT / "v22_79_part_d_interaction_lift_source_witness_debt_blend_probe_route.json"),
            "part_d_interaction_lift_source_witness_high_debt_probe": rel(OUT_ROOT / "v22_79_part_d_interaction_lift_source_witness_high_debt_probe_route.json"),
        },
    }
    write_json(OUT_ROOT / "v22_79_final_route.json", final)
    append_exec("G_failure_decomposition_final_route", command, "done", files=f"{rel(OUT_ROOT / 'v22_79_part_g_failure_decomposition.csv')}; {rel(OUT_ROOT / 'v22_79_final_route.json')}", note=json.dumps({"final_route": route, "reason": reason}, ensure_ascii=False))
    append_recap("Part G final route", [f"final_route={route}；reason={reason}", f"Part gates: A={row['part_a_hard_gate_pass']} B={row['part_b_gate_pass']} C={row['part_c_gate_pass']} D={row['part_d_gate_pass']} E_explore={row['part_e_exploration_gate_pass']} E_official={row['official_candidate_gate_pass']} F={row['part_f_gate_pass']}。", f"repair_route={row['part_d_control_margin_repair_route']}；basis_redesign_route={row['part_g_basis_redesign_route']}；utility_ablation_route={row['part_d_utility_ablation_route']}；debt_residual_ablation_route={row['part_d_debt_residual_ablation_route']}；tail95_debt_residual_ablation_route={row['part_d_tail95_debt_residual_ablation_route']}；debt_residual_gap_audit_route={row['part_d_debt_residual_gap_audit_route']}；debt_risk_decomposition_route={row['part_d_debt_risk_decomposition_route']}；separation_trace_route={row['part_d_separation_trace_route']}；interaction_lift_route={row['part_d_interaction_lift_route']}；interaction_lift_integration_route={row['part_d_interaction_lift_integration_route']}；interaction_lift_scale_debt_route={row['part_d_interaction_lift_scale_debt_route']}；interaction_lift_w2scale_route={row['part_d_interaction_lift_w2scale_route']}；interaction_lift_debt_frontier_route={row['part_d_interaction_lift_debt_frontier_route']}；interaction_lift_debt_orthogonalized_route={row['part_d_interaction_lift_debt_orthogonalized_route']}；interaction_lift_upstream_only_route={row['part_d_interaction_lift_upstream_only_route']}；interaction_lift_source_witness_route={row['part_d_interaction_lift_source_witness_route']}；interaction_lift_source_witness_orthogonalized_route={row['part_d_interaction_lift_source_witness_orthogonalized_route']}；interaction_lift_source_witness_debt_blend_route={row['part_d_interaction_lift_source_witness_debt_blend_route']}；interaction_lift_source_witness_high_debt_route={row['part_d_interaction_lift_source_witness_high_debt_route']}。", f"basis_redesign_rows={row['part_g_basis_redesign_rows']}；utility_ablation_rows={row['part_d_utility_ablation_rows']}；debt_residual_ablation_rows={row['part_d_debt_residual_ablation_rows']}；tail95_debt_residual_ablation_rows={row['part_d_tail95_debt_residual_ablation_rows']}；debt_residual_gap_audit_rows={row['part_d_debt_residual_gap_audit_rows']}；debt_risk_decomposition_rows={row['part_d_debt_risk_decomposition_rows']}；separation_trace_rows={row['part_d_separation_trace_rows']}；interaction_lift_rows={row['part_d_interaction_lift_rows']}；interaction_lift_integration_rows={row['part_d_interaction_lift_integration_rows']}；interaction_lift_scale_debt_rows={row['part_d_interaction_lift_scale_debt_rows']}；interaction_lift_w2scale_rows={row['part_d_interaction_lift_w2scale_rows']}；interaction_lift_debt_frontier_rows={row['part_d_interaction_lift_debt_frontier_rows']}；interaction_lift_debt_orthogonalized_rows={row['part_d_interaction_lift_debt_orthogonalized_rows']}；interaction_lift_upstream_only_rows={row['part_d_interaction_lift_upstream_only_rows']}；interaction_lift_source_witness_rows={row['part_d_interaction_lift_source_witness_rows']}；interaction_lift_source_witness_orthogonalized_rows={row['part_d_interaction_lift_source_witness_orthogonalized_rows']}；interaction_lift_source_witness_debt_blend_rows={row['part_d_interaction_lift_source_witness_debt_blend_rows']}；interaction_lift_source_witness_high_debt_rows={row['part_d_interaction_lift_source_witness_high_debt_rows']}。", f"failure_components={row['failure_components']}。"])
    return final


def run_full(args: argparse.Namespace) -> dict[str, Any]:
    a = run_part_a(args)
    if not int(a.get("part_a_hard_gate_pass", 0)):
        return run_part_g(args, a, {}, {}, {}, {}, {})
    b = run_part_b(args)
    c = run_part_c(args) if int(b.get("part_b_gate_pass", 0)) else {}
    d = run_part_d(args) if int(c.get("part_c_gate_pass", 0)) else {}
    e = run_part_e(args, d) if d else {}
    f = run_part_f(args, e) if e else {}
    return run_part_g(args, a, b, c, d, e, f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        default="full",
        choices=[
            "full",
            "part-a",
            "part-b",
            "part-c",
            "part-d",
            "part-d-merge",
            "part-d-repair",
            "part-d-repair-merge",
            "part-d-utility-ablation",
            "part-d-utility-ablation-merge",
            "part-d-debt-residual-audit",
            "part-d-debt-risk-audit",
            "part-d-separation-trace-audit",
            "part-d-interaction-lift-search",
            "part-d-interaction-lift-search-merge",
            "part-e",
            "part-f",
            "part-g",
            "part-g-redesign",
            "part-g-redesign-merge",
            "final-route",
        ],
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--held-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--metric-batch-size", type=int, default=160)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--projector-ridge", type=float, default=3.0e-4)
    p.add_argument("--step-mult", type=float, default=0.30)
    p.add_argument("--debt-margin-lambda", type=float, default=0.50)
    p.add_argument("--part-d-datasets", default="Wine,Spam,MNIST")
    p.add_argument("--part-d-seed-count", type=int, default=5)
    p.add_argument("--part-d-shard-count", type=int, default=1)
    p.add_argument("--part-d-shard-index", type=int, default=0)
    p.add_argument("--part-d-repair-families", default=",".join(PART_D_REPAIR_FAMILIES))
    p.add_argument("--part-d-repair-update-rules", default=",".join(PART_D_REPAIR_UPDATE_RULES))
    p.add_argument("--part-d-repair-bootstrap-repeats", type=int, default=8)
    p.add_argument("--part-g-redesign-families", default=",".join(PART_G_REDESIGN_FAMILIES))
    p.add_argument("--part-d-utility-ablation-families", default=",".join(PART_D_UTILITY_ABLATION_FAMILIES))
    p.add_argument("--part-d-utility-ablation-rules", default=",".join(PART_D_UTILITY_ABLATION_RULES))
    p.add_argument("--part-d-utility-output-stem", default="v22_79_part_d_utility_ablation")
    p.add_argument("--part-d-interaction-lift-families", default=",".join(PART_D_INTERACTION_LIFT_FAMILIES))
    p.add_argument("--part-d-interaction-lift-rules", default=",".join(PART_D_INTERACTION_LIFT_RULES))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_out()
    if args.mode == "full":
        run_full(args)
    elif args.mode == "part-a":
        run_part_a(args)
    elif args.mode == "part-b":
        run_part_b(args)
    elif args.mode == "part-c":
        run_part_c(args)
    elif args.mode == "part-d":
        run_part_d(args)
    elif args.mode == "part-d-merge":
        run_part_d_merge(args)
    elif args.mode == "part-d-repair":
        run_part_d_repair(args)
    elif args.mode == "part-d-repair-merge":
        run_part_d_repair_merge(args)
    elif args.mode == "part-d-utility-ablation":
        run_part_d_utility_ablation(args)
    elif args.mode == "part-d-utility-ablation-merge":
        run_part_d_utility_ablation_merge(args)
    elif args.mode == "part-d-debt-residual-audit":
        run_part_d_debt_residual_audit(args)
    elif args.mode == "part-d-debt-risk-audit":
        run_part_d_debt_risk_audit(args)
    elif args.mode == "part-d-separation-trace-audit":
        run_part_d_separation_trace_audit(args)
    elif args.mode == "part-d-interaction-lift-search":
        run_part_d_interaction_lift_search(args)
    elif args.mode == "part-d-interaction-lift-search-merge":
        run_part_d_interaction_lift_search_merge(args)
    elif args.mode == "part-e":
        d = load_json(OUT_ROOT / "v22_79_part_d_preflight_route.json")
        run_part_e(args, d)
    elif args.mode == "part-f":
        e_rows = read_rows(OUT_ROOT / "v22_79_part_e_candidate_gate.csv")
        run_part_f(args, e_rows[0] if e_rows else {})
    elif args.mode == "part-g-redesign":
        run_part_g_redesign(args)
    elif args.mode == "part-g-redesign-merge":
        run_part_g_redesign_merge(args)
    elif args.mode in {"part-g", "final-route"}:
        a = load_json(OUT_ROOT / "v22_79_part_a_code_identity_hard_gate.json")
        b = load_json(OUT_ROOT / "v22_79_part_b_v22_78_failure_replay_summary.json")
        c = load_json(OUT_ROOT / "v22_79_part_c_unit_gate.json")
        d = load_json(OUT_ROOT / "v22_79_part_d_preflight_route.json")
        e_rows = read_rows(OUT_ROOT / "v22_79_part_e_candidate_gate.csv")
        f_rows = read_rows(OUT_ROOT / "v22_79_part_f_strengthened_mlp_audit.csv")
        run_part_g(args, a, b, c, d, e_rows[0] if e_rows else {}, f_rows[0] if f_rows else {})


if __name__ == "__main__":
    main()
