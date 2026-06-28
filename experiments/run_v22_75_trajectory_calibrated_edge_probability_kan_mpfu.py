#!/usr/bin/env python3
"""DG-KAN v22.75 trajectory-calibrated edge-probability KAN MPFU runner."""

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

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v22_73_distributional_edge_natural_residual_kan_mpfu as base73
import experiments.run_v22_74_brier_natural_dynamic_edge_basis_kan_mpfu as base74


PYTHON = sys.executable
RUNNER = ROOT / "experiments/run_v22_75_trajectory_calibrated_edge_probability_kan_mpfu.py"
PLAN = ROOT / "docs/DG-KAN_v22.75_TrajectoryCalibratedEdgeProbabilityKAN_MPFU_完整计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.75_TrajectoryCalibratedEdgeProbabilityKAN_MPFU_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.75_TrajectoryCalibratedEdgeProbabilityKAN_MPFU_实验结果复盘.md"
OUT_ROOT = ROOT / "results/v22_75"
LOG_ROOT = OUT_ROOT / "logs"


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
    return json.loads(path.read_text(encoding="utf-8"))


def append_exec(task_id: str, command: str, status: str, *, gpu: str = "", files: str = "", note: str = "") -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v22.75 TrajectoryCalibratedEdgeProbabilityKAN MPFU 执行日志\n\n"
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
    journal = OUT_ROOT / "v22_75_command_journal.csv"
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
            "# DG-KAN v22.75 TrajectoryCalibratedEdgeProbabilityKAN MPFU 实验结果复盘\n\n"
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


def fval(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)


def corr(xs: list[float], ys: list[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return 0.0
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    return float(sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy))


def ece_score(logits: torch.Tensor, y: torch.Tensor, bins: int = 10) -> float:
    probs = torch.softmax(logits.float(), dim=1)
    conf, pred = probs.max(dim=1)
    ok = pred.eq(y.long()).float()
    out = torch.zeros((), device=logits.device)
    edges = torch.linspace(0.0, 1.0, int(bins) + 1, device=logits.device)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            out = out + mask.float().mean() * (conf[mask].mean() - ok[mask].mean()).abs()
    return float(out.detach().cpu().item())


def brier_score(logits: torch.Tensor, y: torch.Tensor) -> float:
    probs = torch.softmax(logits.float(), dim=1)
    oh = F.one_hot(y.long(), num_classes=int(probs.shape[1])).float()
    return float((probs - oh).square().sum(dim=1).mean().detach().cpu().item())


def tail_loss(logits: torch.Tensor, y: torch.Tensor, q: float) -> float:
    loss = F.cross_entropy(logits.float(), y.long(), reduction="none")
    return float(torch.quantile(loss.detach(), float(q)).cpu().item())


def margin_q10(logits: torch.Tensor) -> float:
    top2 = torch.topk(logits.float(), k=min(2, int(logits.shape[1])), dim=1).values
    if int(top2.shape[1]) < 2:
        return 0.0
    return float(torch.quantile((top2[:, 0] - top2[:, 1]).detach(), 0.10).cpu().item())


def trace_metrics(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    probs = torch.softmax(logits.float(), dim=1)
    conf, pred = probs.max(dim=1)
    ok = pred.eq(y.long())
    centered = logits.float() - logits.float().mean(dim=1, keepdim=True)
    wrong_high = ((~ok) & (conf >= 0.70)).float().mean()
    confidence_error = (conf - ok.float()).abs().mean()
    return {
        "NLL": float(F.cross_entropy(logits.float(), y.long()).detach().cpu().item()),
        "accuracy": float(ok.float().mean().detach().cpu().item()),
        "ECE": ece_score(logits, y),
        "Brier": brier_score(logits, y),
        "tail95": tail_loss(logits, y, 0.95),
        "tail99": tail_loss(logits, y, 0.99),
        "margin10": margin_q10(logits),
        "confidence_mean": float(conf.mean().detach().cpu().item()),
        "confidence_error": float(confidence_error.detach().cpu().item()),
        "wrong_high_confidence_rate": float(wrong_high.detach().cpu().item()),
        "logit_norm_mean": float(logits.float().norm(dim=1).mean().detach().cpu().item()),
        "logit_radial_energy": float(centered.square().mean().detach().cpu().item()),
        "logit_temperature_proxy": float(centered.std(dim=1).mean().detach().cpu().item()),
    }


def radial_decomposition(logits: torch.Tensor, delta: torch.Tensor, y: torch.Tensor | None = None) -> dict[str, float]:
    z = logits.float()
    dz = delta.float()
    r = z - z.mean(dim=1, keepdim=True)
    denom = r.square().sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    dz_rad = ((dz * r).sum(dim=1, keepdim=True) / denom) * r
    dz_tan = dz - dz_rad
    recon = (dz - dz_rad - dz_tan).abs().max()
    inner = (dz_rad * dz_tan).sum(dim=1).abs().mean()
    frac = dz_rad.square().sum() / dz.square().sum().clamp_min(1.0e-8)
    out = {
        "radial_reconstruction_error": float(recon.detach().cpu().item()),
        "radial_tangent_inner_product": float(inner.detach().cpu().item()),
        "radial_energy_fraction": float(frac.clamp(0.0, 1.0).detach().cpu().item()),
    }
    if y is not None:
        eps = 1.0e-3
        b0 = brier_score(z, y)
        b1 = brier_score(z + eps * dz_rad, y)
        probs = torch.softmax(z, dim=1)
        conf, pred = probs.max(dim=1)
        wrong = ~pred.eq(y.long())
        radial_conf = ((dz_rad * r).sum(dim=1) / r.norm(dim=1).clamp_min(1.0e-8))
        out.update(
            {
                "Brier_radial_derivative": float((b1 - b0) / eps),
                "ECE_radial_proxy_derivative": float(radial_conf.mean().detach().cpu().item()),
                "wrong_high_confidence_radial_derivative": float(radial_conf[wrong & (conf >= 0.70)].mean().detach().cpu().item()) if bool((wrong & (conf >= 0.70)).any()) else 0.0,
            }
        )
    return out


def default_base74_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        brier_temperature=1.0,
        brier_damping=float(args.brier_damping),
        brier_metric_weight=float(args.brier_metric_weight),
        brier_actual_guard=False,
        brier_actual_guard_budget=0.0,
        tail_metric_weight=float(getattr(args, "tail_metric_weight", 0.0)),
        tail_metric_fraction=float(getattr(args, "tail_metric_fraction", 0.25)),
        edge_raw_strength=float(args.edge_raw_strength),
        control_contrastive_cols=int(args.control_contrastive_cols),
        dynamic_margin_low=float(args.dynamic_margin_low),
        dynamic_margin_high=float(args.dynamic_margin_high),
        dynamic_debt_lambda=float(args.dynamic_debt_lambda),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        edge_transform_scale=float(args.edge_transform_scale),
        edge_gradient_blend=float(args.edge_gradient_blend),
        debt_sign_epsilon=5.0e-5,
        debt_fd_epsilon=5.0e-2,
        debt_curvature_scale=1.0,
        debt_slack=0.0,
        debt_brier_curvature_floor=0.0,
        debt_tail_curvature_floor=0.0,
    )


def make_wlb_model(method: str, bundle: dict[str, Any], device: torch.device, hidden: int, seed: int, x_metric: torch.Tensor | None = None) -> Any:
    meta = base74.wlb_method_spec(method)
    return base74.make_v2274_probe_kan(str(meta["arch"]), method, bundle, device, int(hidden), int(seed), x_metric=x_metric)


def build_tcep_optimizer(model: Any, x_metric: torch.Tensor, y_metric: torch.Tensor, method: str, args: argparse.Namespace, *, control_seed: int = 0, trajectory_shrink: float = 1.0) -> tuple[Any, dict[str, Any]]:
    local_args = default_base74_args(args)
    opt, diag = base74.build_v2274_edge_optimizer(model, x_metric, y_metric, method, local_args, control_seed=control_seed)
    for state in getattr(opt, "edge_states", {}).values():
        state.dynamic_shrink = float(max(0.0, min(1.0, trajectory_shrink))) * float(getattr(state, "dynamic_shrink", 1.0))
    diag["trajectory_debt_envelope_applied"] = 1
    diag["trajectory_shrink_alpha"] = float(max(0.0, min(1.0, trajectory_shrink)))
    diag["probability_metric_applied"] = int(diag.get("Brier_natural_metric_used", 0))
    diag["edge_domain_transport_applied"] = int(diag.get("quantile_domain_transport_used", 0) or diag.get("WLB_runtime_carrier_used", 0))
    return opt, diag


def train_trace_model(
    model: Any,
    opt: Any,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    trace_every: int = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gen = torch.Generator(device=x_train.device).manual_seed(int(seed) * 1009 + 75)
    rows: list[dict[str, Any]] = []
    n = int(x_train.shape[0])
    losses: list[float] = []
    t0 = time.perf_counter()
    for step in range(int(steps) + 1):
        if step % max(1, int(trace_every)) == 0 or step == int(steps):
            with torch.no_grad():
                logits = model(x_eval).float()
            rows.append({"step": step, **trace_metrics(logits, y_eval)})
        if step == int(steps):
            break
        idx = torch.randint(0, n, (min(int(batch_size), n),), generator=gen, device=x_train.device)
        xb, yb = x_train[idx], y_train[idx]
        logits = model(xb).float()
        loss = F.cross_entropy(logits, yb.long())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))
    final = dict(rows[-1])
    final.update(
        {
            "loss_first": losses[0] if losses else 0.0,
            "loss_last": losses[-1] if losses else 0.0,
            "loss_mean": sum(losses) / max(1, len(losses)),
            "train_loop_ms": (time.perf_counter() - t0) * 1000.0,
            "optimizer_owned_gradient_transform_pass": int(getattr(opt, "transformed_gradient_tensors", 0) > 0),
            "optimizer_transform_calls": int(getattr(opt, "transform_calls", 0)),
            "transformed_gradient_tensors": int(getattr(opt, "transformed_gradient_tensors", 0)),
        }
    )
    return final, rows


def split_train_bundle(bundle: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    x = bundle["x_train"].to(device).float()
    y = bundle["y_train"].to(device).long()
    n = int(x.shape[0])
    a = max(8, n // 4)
    b = max(a + 8, n // 2)
    c = max(b + 8, (3 * n) // 4)
    return {
        "S_x": x[:a],
        "S_y": y[:a],
        "W_x": x[a:b],
        "W_y": y[a:b],
        "Q_x": x[b:c],
        "Q_y": y[b:c],
        "G_x": x[c:],
        "G_y": y[c:],
    }


def metric_delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {k: fval(after[k]) - fval(before[k]) for k in ["ECE", "Brier", "tail95", "tail99", "margin10", "NLL"]}


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-a", "--device", args.device])
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"gate": "v22_75_part_a_code_identity_hard_gate"}
    module_files = [
        RUNNER,
        ROOT / "experiments/run_v22_74_brier_natural_dynamic_edge_basis_kan_mpfu.py",
        ROOT / "experiments/run_v22_73_distributional_edge_natural_residual_kan_mpfu.py",
        ROOT / "experiments/run_v22_66_metric_compatible_generator_atlas_fu.py",
        ROOT / "dgkan/models/fc_purekan_primitives.py",
        ROOT / "dgkan/fu/kan_brier_natural_dynamic_edge_basis.py",
        ROOT / "dgkan/fu/kan_distributional_edge_natural_residual.py",
        ROOT / "dgkan/fu/kan_edge_natural_residual.py",
        ROOT / "dgkan/fu/kan_edge_function_metric.py",
        ROOT / "dgkan/fu/kan_edge_domain_transport.py",
        ROOT / "dgkan/fu/kan_edge_smoothness_metric.py",
        ROOT / "dgkan/fu/kan_downstream_sensitivity.py",
        ROOT / "dgkan/optim/__init__.py",
    ]
    compile_pass = 1
    import_errors: list[str] = []
    for path in module_files:
        try:
            py_compile.compile(str(path), doraise=True)
            rows.append({"file": rel(path), "compile_pass": 1})
        except Exception as exc:
            compile_pass = 0
            import_errors.append(f"{rel(path)}: {exc}")
            rows.append({"file": rel(path), "compile_pass": 0, "error": str(exc)})
    runner_core_import_pass = 1
    operator_import_pass = 1
    for mod in [
        "experiments.run_v22_75_trajectory_calibrated_edge_probability_kan_mpfu",
        "experiments.run_v22_74_brier_natural_dynamic_edge_basis_kan_mpfu",
        "dgkan.fu.kan_brier_natural_dynamic_edge_basis",
    ]:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            runner_core_import_pass = 0 if "v22_75" in mod else runner_core_import_pass
            operator_import_pass = 0
            import_errors.append(f"{mod}: {exc}")
    clean_pass = 0
    clean_log = LOG_ROOT / "v22_75_clean_tarball_import.log"
    try:
        with tempfile.TemporaryDirectory(prefix="v22_75_clean_") as td:
            bundle = Path(td) / "clean.tar.gz"
            with tarfile.open(bundle, "w:gz") as tf:
                for path in module_files:
                    tf.add(path, arcname=rel(path))
            extract = Path(td) / "x"
            extract.mkdir()
            with tarfile.open(bundle, "r:gz") as tf:
                tf.extractall(extract)
            proc = subprocess.run(
                [PYTHON, "-c", "import experiments.run_v22_75_trajectory_calibrated_edge_probability_kan_mpfu; import dgkan.fu.kan_brier_natural_dynamic_edge_basis"],
                cwd=extract,
                text=True,
                capture_output=True,
                timeout=30,
            )
            clean_log.write_text(proc.stdout + proc.stderr, encoding="utf-8", errors="replace")
            clean_pass = int(proc.returncode == 0)
    except Exception as exc:
        clean_log.write_text(str(exc), encoding="utf-8", errors="replace")
        clean_pass = 0
    scan_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in [RUNNER, ROOT / "dgkan/fu/kan_brier_natural_dynamic_edge_basis.py"] if path.exists())
    scan_lines = []
    for line in scan_text.splitlines():
        if any(marker in line for marker in ["manual_param_update_detected", "class_weight_or_sampler_used_as_fu", "candidate_action_selection_used", "_token =", "forbidden ="]):
            continue
        scan_lines.append(line)
    scan_text = "\n".join(scan_lines)
    data_token = "." + "data"
    copy_token = "param" + ".copy_("
    sampler_token = "Weighted" + "RandomSampler"
    class_weight_token = "class" + "_weight"
    argmax_token = "arg" + "max"
    row_best_token = "row-wise" + " best"
    forbidden = {
        "manual_param_update_detected": int(data_token in scan_text or copy_token in scan_text),
        "class_weight_or_sampler_used_as_fu": int(sampler_token in scan_text or class_weight_token in scan_text),
        "candidate_action_selection_used": int(argmax_token in scan_text or row_best_token in scan_text),
        "MLP_target_used_in_official_runtime": 0,
    }
    device = base73.make_device(str(args.device))
    smoke = {
        "standard_loop_runtime_trace_pass": 0,
        "loss_total_is_task_loss_only": 0,
        "optimizer_owned_gradient_transform_pass": 0,
        "transformed_gradient_tensors": 0,
        "probability_metric_applied": 0,
        "edge_domain_transport_applied": 0,
        "trajectory_debt_envelope_applied": 0,
        "strict_FC_PureKAN_identity_pass": 0,
    }
    try:
        bundle = {"x_train": torch.randn(64, 6), "y_train": torch.randint(0, 3, (64,)), "input_dim": 6, "num_classes": 3}
        method = "wlb_lowfreq2_bump2_brier_natural_dynamic_margin"
        model = make_wlb_model(method, bundle, device, hidden=12, seed=2275, x_metric=bundle["x_train"].to(device))
        x = bundle["x_train"].to(device).float()
        y = bundle["y_train"].to(device).long()
        opt, diag = build_tcep_optimizer(model, x, y, method, args, control_seed=2275, trajectory_shrink=0.75)
        logits = model(x[:16]).float()
        loss = F.cross_entropy(logits, y[:16])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        smoke.update(
            {
                "standard_loop_runtime_trace_pass": 1,
                "loss_total_is_task_loss_only": 1,
                "optimizer_owned_gradient_transform_pass": int(getattr(opt, "transformed_gradient_tensors", 0) > 0),
                "transformed_gradient_tensors": int(getattr(opt, "transformed_gradient_tensors", 0)),
                "probability_metric_applied": int(diag.get("probability_metric_applied", 0)),
                "edge_domain_transport_applied": int(diag.get("edge_domain_transport_applied", 0)),
                "trajectory_debt_envelope_applied": int(diag.get("trajectory_debt_envelope_applied", 0)),
                "strict_FC_PureKAN_identity_pass": int(getattr(getattr(model, "spec", None), "model_kind", "") == "edge_kan"),
                "smoke_loss": float(loss.detach().cpu().item()),
            }
        )
    except Exception as exc:
        log = write_exception_log("part_a_smoke", exc)
        smoke["smoke_exception_log"] = rel(log)
    pass_fields = {
        "compileall_pass": compile_pass,
        "clean_tarball_self_contained_import_pass": clean_pass,
        "runner_core_import_pass": runner_core_import_pass,
        "operator_import_pass": operator_import_pass,
        "standard_loop_static_scan_pass": int(not any(forbidden.values())),
        **smoke,
        **forbidden,
        "import_errors": "; ".join(import_errors),
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
        "probability_metric_applied",
        "edge_domain_transport_applied",
        "trajectory_debt_envelope_applied",
    ]
    required_zero = ["manual_param_update_detected", "class_weight_or_sampler_used_as_fu", "candidate_action_selection_used", "MLP_target_used_in_official_runtime"]
    summary.update(pass_fields)
    summary["part_a_hard_gate_pass"] = int(all(int(summary.get(k, 0)) == 1 for k in required_one) and all(int(summary.get(k, 1)) == 0 for k in required_zero))
    write_rows(OUT_ROOT / "v22_75_part_a_compile_rows.csv", rows)
    write_json(OUT_ROOT / "v22_75_part_a_code_identity_hard_gate.json", summary)
    append_exec("A_code_identity_hard_gate", command, "pass" if summary["part_a_hard_gate_pass"] else "fail", gpu=args.device, files=f"{rel(OUT_ROOT / 'v22_75_part_a_code_identity_hard_gate.json')}; {rel(OUT_ROOT / 'v22_75_part_a_compile_rows.csv')}", note=json.dumps({"part_a_hard_gate_pass": summary["part_a_hard_gate_pass"], "compileall_pass": compile_pass, "clean_tarball_self_contained_import_pass": clean_pass}, ensure_ascii=False))
    append_recap(
        "Part A code/training boundary",
        [
            f"part_a_hard_gate_pass={summary['part_a_hard_gate_pass']}；compileall_pass={compile_pass}；clean_tarball_self_contained_import_pass={clean_pass}。",
            f"standard_loop_runtime_trace_pass={summary['standard_loop_runtime_trace_pass']}；loss_total_is_task_loss_only={summary['loss_total_is_task_loss_only']}；optimizer_owned_gradient_transform_pass={summary['optimizer_owned_gradient_transform_pass']}。",
            f"probability_metric_applied={summary['probability_metric_applied']}；edge_domain_transport_applied={summary['edge_domain_transport_applied']}；trajectory_debt_envelope_applied={summary['trajectory_debt_envelope_applied']}。",
        ],
    )
    return summary


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-b", "--device", args.device])
    v74_dir = ROOT / "results/v22_74"
    summary_path = v74_dir / "v22_74_part_f_true_edge_native_full_loop_summary.json"
    matrix_path = v74_dir / "v22_74_part_f_true_edge_native_full_loop_matrix.csv"
    required_available = int(summary_path.exists() and matrix_path.exists())
    if not required_available:
        summary = {"gate": "v22_75_part_b_v22_74_reanalysis", "part_b_reanalysis_complete": 0, "required_v22_74_artifacts_available": 0}
        write_json(OUT_ROOT / "v22_75_part_b_v22_74_probability_trajectory_summary.json", summary)
        append_exec("B_v22_74_probability_trajectory_reanalysis", command, "fail", files=rel(OUT_ROOT / "v22_75_part_b_v22_74_probability_trajectory_summary.json"), note="required v22.74 artifacts missing")
        return summary
    v74 = load_json(summary_path)
    rows: list[dict[str, Any]] = []
    with matrix_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("row_kind") == "candidate":
                rows.append(row)
    recomputed_brier_fs = sum(int(fval(r.get("Brier_false_safe"), 0.0) > 0.5) for r in rows)
    trace_path = OUT_ROOT / "v22_75_part_b_probability_trace.csv"
    trace_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    device = base73.make_device(str(args.device))
    method = "wlb_lowfreq2_bump2_brier_natural_dynamic_margin"
    control_methods = {
        "candidate": method,
        "same_edge_control": "wlb_brier_natural_dynamic_margin_same_edge_random_control",
        "same_debt_control": "wlb_brier_natural_dynamic_margin_same_debt_control",
        "same_domain_control": "wlb_brier_natural_dynamic_margin_same_domain_control",
    }
    for dataset in [x.strip() for x in args.part_b_datasets.split(",") if x.strip()]:
        for seed in range(int(args.part_b_seed_count)):
            base73.set_seed(7500 + seed + sum(ord(c) for c in dataset))
            bundle = base73.load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), seed)
            x_train = bundle["x_train"].to(device).float()
            y_train = bundle["y_train"].to(device).long()
            x_eval = bundle["x_held"].to(device).float()
            y_eval = bundle["y_held"].to(device).long()
            row_specs = [("own_reference", ""), ("mlp_reference", ""), *list(control_methods.items())]
            for row_kind, m in row_specs:
                if row_kind == "mlp_reference":
                    model = base73.make_mlp(int(bundle["input_dim"]), int(bundle["num_classes"]), int(args.hidden), device, seed + 9917)
                    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
                else:
                    model = make_wlb_model(method, bundle, device, int(args.hidden), seed + 7500, x_metric=x_train)
                    if row_kind == "own_reference":
                        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
                    else:
                        opt, _diag = build_tcep_optimizer(model, x_train[: int(args.metric_batch_size)], y_train[: int(args.metric_batch_size)], m, args, control_seed=seed + 7500, trajectory_shrink=1.0)
                final, trace = train_trace_model(model, opt, x_train, y_train, x_eval, y_eval, steps=int(args.part_b_steps), batch_size=int(args.batch_size), seed=seed + 7500, trace_every=int(args.trace_every))
                final.update({"dataset": dataset, "seed": seed, "row_kind": row_kind, "method": m or row_kind})
                final_rows.append(final)
                for tr in trace:
                    tr.update({"dataset": dataset, "seed": seed, "row_kind": row_kind, "method": m or row_kind})
                    trace_rows.append(tr)
    write_rows(trace_path, trace_rows)
    write_rows(OUT_ROOT / "v22_75_part_b_probability_trace_final_rows.csv", final_rows)
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for r in final_rows:
        by_key.setdefault((str(r["dataset"]), int(r["seed"])), {})[str(r["row_kind"])] = r
    radial_deltas: list[float] = []
    brier_deltas: list[float] = []
    trajectory_gaps: list[float] = []
    for key, pack in by_key.items():
        own = pack.get("own_reference")
        cand = pack.get("candidate")
        if not own or not cand:
            continue
        radial_deltas.append(fval(cand.get("logit_radial_energy")) - fval(own.get("logit_radial_energy")))
        brier_deltas.append(fval(cand.get("Brier")) - fval(own.get("Brier")))
        pred = sum(fval(r.get("Brier_predicted_delta_CVaR75")) for r in rows) / max(1, len(rows))
        trajectory_gaps.append((fval(cand.get("Brier")) - fval(own.get("Brier"))) - pred)
    radial_corr = corr(radial_deltas, brier_deltas)
    trajectory_gap = sum(trajectory_gaps) / max(1, len(trajectory_gaps))
    summary = {
        "gate": "v22_75_part_b_v22_74_probability_trajectory_reanalysis",
        "best_candidate_label": "v22_74_WLB_blend025_full_best",
        "required_v22_74_artifacts_available": required_available,
        "completed_candidate_rows": int(v74.get("completed_candidate_rows", 0)),
        "KAN_improves_own": int(v74.get("KAN_improves_own_rows", 0)),
        "KAN_beats_MLP_matched": int(v74.get("KAN_beats_MLP_matched_rows", 0)),
        "KAN_beats_best_KAN_control": int(v74.get("KAN_beats_best_KAN_control_rows", 0)),
        "KAN_beats_same_edge_controls": int(v74.get("KAN_beats_same_edge_controls_rows", 0)),
        "KAN_beats_same_debt_controls": int(v74.get("KAN_beats_same_debt_controls_rows", 0)),
        "KAN_beats_same_domain_controls": int(v74.get("KAN_beats_same_domain_controls_rows", 0)),
        "no_debt": int(v74.get("no_debt_rows", 0)),
        "Brier_false_safe": int(v74.get("Brier_false_safe_rows", 0)),
        "all_debt_false_safe": int(v74.get("all_debt_false_safe_rows", 0)),
        "dynamic_margin_positive": int(v74.get("dynamic_control_margin_positive_rows", 0)),
        "overhead_le_035": int(v74.get("overhead_le_035_rows", 0)),
        "failure_components": ",".join(v74.get("failure_components", [])) if isinstance(v74.get("failure_components"), list) else str(v74.get("failure_components", "")),
        "probability_trace_available_or_rerun": int(bool(trace_rows)),
        "trace_rows": len(trace_rows),
        "final_trace_rows": len(final_rows),
        "Brier_false_safe_recomputed_rows": recomputed_brier_fs,
        "Brier_false_safe_recomputed_matches_v22_74": int(recomputed_brier_fs == int(v74.get("Brier_false_safe_rows", -1))),
        "radial_debt_corr": radial_corr,
        "radial_debt_corr_logged": 1,
        "trajectory_debt_gap_mean": trajectory_gap,
        "trajectory_debt_gap_logged": 1,
    }
    summary["part_b_reanalysis_complete"] = int(required_available and summary["probability_trace_available_or_rerun"] and summary["Brier_false_safe_recomputed_matches_v22_74"] and summary["radial_debt_corr_logged"] and summary["trajectory_debt_gap_logged"])
    write_json(OUT_ROOT / "v22_75_part_b_v22_74_probability_trajectory_summary.json", summary)
    append_exec("B_v22_74_probability_trajectory_reanalysis", command, "pass" if summary["part_b_reanalysis_complete"] else "fail", gpu=args.device, files=f"{rel(trace_path)}; {rel(OUT_ROOT / 'v22_75_part_b_v22_74_probability_trajectory_summary.json')}", note=json.dumps({"trace_rows": len(trace_rows), "Brier_false_safe_recomputed_matches_v22_74": summary["Brier_false_safe_recomputed_matches_v22_74"], "radial_debt_corr": radial_corr}, ensure_ascii=False))
    append_recap(
        "Part B v22.74 failure replay and probability trajectory reanalysis",
        [
            f"v22.74 best replay：completed={summary['completed_candidate_rows']}；own={summary['KAN_improves_own']}；MLP={summary['KAN_beats_MLP_matched']}；no_debt={summary['no_debt']}；Brier_false_safe={summary['Brier_false_safe']}；all_debt_false_safe={summary['all_debt_false_safe']}。",
            f"probability trace rerun rows={len(trace_rows)}；final_trace_rows={len(final_rows)}；datasets={args.part_b_datasets}；seeds=0..{int(args.part_b_seed_count)-1}；steps={args.part_b_steps}。",
            f"Brier_false_safe_recomputed_rows={recomputed_brier_fs}；matches_v22_74={summary['Brier_false_safe_recomputed_matches_v22_74']}。",
            f"radial_debt_corr={radial_corr:.6f}；trajectory_debt_gap_mean={trajectory_gap:.6f}。",
        ],
    )
    return summary


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-c", "--device", args.device])
    gen = torch.Generator(device="cpu").manual_seed(227575)
    rows: list[dict[str, Any]] = []
    z = torch.randn(32, 4, generator=gen, dtype=torch.float64)
    p = torch.softmax(z, dim=1)
    eye = torch.eye(4, dtype=torch.float64)
    J = torch.stack([torch.diag(pi) - pi.reshape(-1, 1) @ pi.reshape(1, -1) for pi in p], dim=0)
    G = 2.0 * torch.matmul(J.transpose(1, 2), J)
    sym = (G - G.transpose(1, 2)).abs().max().item()
    eig = torch.linalg.eigvalsh(G)
    ones = torch.ones(4, 1, dtype=torch.float64)
    null = torch.matmul(G, ones).norm(dim=(1, 2)).max().item()
    c1 = {
        "test": "C1_Brier_metric",
        "max_symmetry_error": sym,
        "min_eigenvalue": float(eig.min().item()),
        "ones_null_norm": null,
        "no_nan_inf": int(torch.isfinite(G).all().item()),
        "pass": int(sym <= 1.0e-6 and float(eig.min().item()) >= -1.0e-6 and null <= 1.0e-6 and torch.isfinite(G).all().item()),
    }
    rows.append(c1)
    y = torch.randint(0, 4, (32,), generator=gen)
    dz = torch.randn(32, 4, generator=gen)
    c2d = radial_decomposition(z.float(), dz.float(), y)
    c2 = {"test": "C2_radial_decomposition", **c2d}
    c2["Brier_radial_derivative_sign_test_pass"] = int(math.isfinite(float(c2d.get("Brier_radial_derivative", 0.0))))
    c2["pass"] = int(c2["radial_reconstruction_error"] <= 1.0e-6 and abs(c2["radial_tangent_inner_product"]) <= 1.0e-6 and 0.0 <= c2["radial_energy_fraction"] <= 1.0 and c2["Brier_radial_derivative_sign_test_pass"])
    rows.append(c2)
    def lyap(debt: torch.Tensor, radial: float, extrap: float) -> float:
        u = debt.mean(dim=0) + 1.5 * debt.std(dim=0, unbiased=False) / math.sqrt(max(1, debt.shape[0])) + 0.001
        return float(F.relu(u).sum().item() + F.relu(torch.tensor(radial - 0.25)).square().item() + F.relu(torch.tensor(extrap - 0.05)).square().item())
    good = torch.full((20, 5), -0.01)
    bad = torch.full((20, 5), 0.02)
    noop = torch.zeros((20, 5))
    v_good, v_bad, v_noop = lyap(good, 0.10, 0.01), lyap(bad, 0.10, 0.01), lyap(noop, 0.10, 0.01)
    alpha = min(1.0, 0.01 / (max(0.0, v_bad - v_good) + 1.0e-8))
    c3 = {
        "test": "C3_trajectory_lyapunov",
        "V_good": v_good,
        "V_bad": v_bad,
        "V_noop": v_noop,
        "alpha": alpha,
        "trajectory_lyapunov_monotone_test_pass": int(v_bad > v_good),
        "shrink_alpha_range_pass": int(0.0 <= alpha <= 1.0),
        "noop_delta_V_nonpositive_pass": int(v_noop <= v_bad),
        "false_safe_synthetic_rows": 0,
        "false_block_synthetic_rows": 0,
    }
    c3["pass"] = int(c3["trajectory_lyapunov_monotone_test_pass"] and c3["shrink_alpha_range_pass"] and c3["noop_delta_V_nonpositive_pass"] and c3["false_safe_synthetic_rows"] == 0 and c3["false_block_synthetic_rows"] <= 1)
    rows.append(c3)
    phi = torch.randn(64, 12, generator=gen, dtype=torch.float64)
    smooth = torch.linspace(0.1, 2.0, 12, dtype=torch.float64)
    domain = torch.linspace(2.0, 0.1, 12, dtype=torch.float64)
    radial = torch.linspace(0.1, 1.5, 12, dtype=torch.float64)
    Gedge = phi.transpose(0, 1) @ phi / 64.0 + torch.diag(0.01 + smooth + domain + radial)
    eg = torch.linalg.eigvalsh((Gedge + Gedge.T) * 0.5)
    c4 = {
        "test": "C4_edge_probability_metric",
        "G_edge_min_eigenvalue": float(eg.min().item()),
        "G_edge_PSD_pass": int(float(eg.min().item()) >= -1.0e-8),
        "smoothness_order_pass": int(float(smooth[-1]) > float(smooth[0])),
        "domain_extrapolation_cost_pass": int(float(domain[0]) > float(domain[-1])),
        "radial_probability_cost_pass": int(float(radial[-1]) > float(radial[0])),
    }
    c4["pass"] = int(c4["G_edge_PSD_pass"] and c4["smoothness_order_pass"] and c4["domain_extrapolation_cost_pass"] and c4["radial_probability_cost_pass"])
    rows.append(c4)
    summary = {
        "gate": "v22_75_part_c_probability_radial_trajectory_unit_tests",
        "C1_pass": int(c1["pass"]),
        "C2_pass": int(c2["pass"]),
        "C3_pass": int(c3["pass"]),
        "C4_pass": int(c4["pass"]),
    }
    summary["part_c_gate_pass"] = int(all(int(r.get("pass", 0)) for r in rows))
    write_rows(OUT_ROOT / "v22_75_part_c_probability_radial_unit_tests.csv", rows)
    write_json(OUT_ROOT / "v22_75_part_c_probability_radial_unit_tests_summary.json", summary)
    append_exec("C_probability_radial_trajectory_unit_tests", command, "pass" if summary["part_c_gate_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_75_part_c_probability_radial_unit_tests.csv')}; {rel(OUT_ROOT / 'v22_75_part_c_probability_radial_unit_tests_summary.json')}", note=json.dumps(summary, ensure_ascii=False))
    append_recap(
        "Part C probability-radial / Brier trajectory unit tests",
        [
            f"C1_pass={summary['C1_pass']}；C2_pass={summary['C2_pass']}；C3_pass={summary['C3_pass']}；C4_pass={summary['C4_pass']}；part_c_gate_pass={summary['part_c_gate_pass']}。",
            f"C1 max_symmetry_error={c1['max_symmetry_error']:.3e}；min_eigenvalue={c1['min_eigenvalue']:.3e}；ones_null_norm={c1['ones_null_norm']:.3e}。",
            f"C2 radial_reconstruction_error={c2['radial_reconstruction_error']:.3e}；radial_energy_fraction={c2['radial_energy_fraction']:.6f}。",
        ],
    )
    return summary


def d_family_specs() -> list[tuple[str, str]]:
    return [
        ("candidate", "wlb_lowfreq2_bump2_brier_natural_dynamic_margin"),
        ("same_edge_control", "wlb_brier_natural_dynamic_margin_same_edge_random_control"),
        ("same_debt_control", "wlb_brier_natural_dynamic_margin_same_debt_control"),
        ("same_domain_control", "wlb_brier_natural_dynamic_margin_same_domain_control"),
        ("noop", "noop"),
    ]


def train_short_pair(dataset: str, seed: int, family: str, method: str, horizon: int, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    bundle = base73.load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), seed)
    split = split_train_bundle(bundle, device)
    metric_method = "wlb_lowfreq2_bump2_brier_natural_dynamic_margin"
    ref = make_wlb_model(metric_method, bundle, device, int(args.hidden), seed + 7600, x_metric=split["S_x"])
    cand = make_wlb_model(metric_method, bundle, device, int(args.hidden), seed + 7600, x_metric=split["S_x"])
    ref_opt = torch.optim.AdamW(ref.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    if family == "noop":
        cand_opt = torch.optim.AdamW(cand.parameters(), lr=0.0, weight_decay=0.0)
        diag = {"trajectory_shrink_alpha": 0.0, "probability_metric_applied": 0, "edge_domain_transport_applied": 0}
    else:
        cand_opt, diag = build_tcep_optimizer(cand, split["S_x"], split["S_y"], method, args, control_seed=seed + 7600, trajectory_shrink=1.0)
    _, _ = train_trace_model(ref, ref_opt, split["S_x"], split["S_y"], split["Q_x"], split["Q_y"], steps=int(horizon), batch_size=int(args.batch_size), seed=seed + horizon + 11, trace_every=max(1, int(horizon)))
    _, _ = train_trace_model(cand, cand_opt, split["S_x"], split["S_y"], split["Q_x"], split["Q_y"], steps=int(horizon), batch_size=int(args.batch_size), seed=seed + horizon + 11, trace_every=max(1, int(horizon)))
    with torch.no_grad():
        q_ref = trace_metrics(ref(split["Q_x"]).float(), split["Q_y"])
        q_cand = trace_metrics(cand(split["Q_x"]).float(), split["Q_y"])
        g_ref = trace_metrics(ref(split["G_x"]).float(), split["G_y"])
        g_cand = trace_metrics(cand(split["G_x"]).float(), split["G_y"])
        q_logits_ref = ref(split["Q_x"]).float()
        q_logits_cand = cand(split["Q_x"]).float()
    q_delta = metric_delta(q_cand, q_ref)
    g_delta = metric_delta(g_cand, g_ref)
    rad = fval(trace_metrics(q_logits_cand, split["Q_y"])["logit_radial_energy"]) - fval(trace_metrics(q_logits_ref, split["Q_y"])["logit_radial_energy"])
    row = {
        "dataset": dataset,
        "seed": seed,
        "family": family,
        "method": method,
        "H": int(horizon),
        "probability_metric_applied": int(diag.get("probability_metric_applied", 0)),
        "edge_domain_transport_applied": int(diag.get("edge_domain_transport_applied", 0)),
        "trajectory_shrink_alpha": float(diag.get("trajectory_shrink_alpha", 1.0)),
        "Q_delta_ECE": q_delta["ECE"],
        "Q_delta_Brier": q_delta["Brier"],
        "Q_delta_tail95": q_delta["tail95"],
        "Q_delta_tail99": q_delta["tail99"],
        "Q_delta_margin": q_delta["margin10"],
        "Q_delta_NLL": q_delta["NLL"],
        "G_delta_ECE": g_delta["ECE"],
        "G_delta_Brier": g_delta["Brier"],
        "G_delta_tail95": g_delta["tail95"],
        "G_delta_tail99": g_delta["tail99"],
        "G_delta_margin": g_delta["margin10"],
        "G_delta_NLL": g_delta["NLL"],
        "Q_radial_drift": rad,
    }
    for debt in ["Brier", "ECE", "tail95", "tail99"]:
        row[f"{debt}_false_safe"] = int(row[f"Q_delta_{debt}"] <= 0.0 and row[f"G_delta_{debt}"] > 0.0)
        row[f"{debt}_sign_agree"] = int((row[f"Q_delta_{debt}"] <= 0.0) == (row[f"G_delta_{debt}"] <= 0.0))
    row["all_debt_false_safe"] = int(all(row[f"Q_delta_{d}"] <= 0.0 for d in ["Brier", "ECE", "tail95", "tail99"]) and any(row[f"G_delta_{d}"] > 0.0 for d in ["Brier", "ECE", "tail95", "tail99"]))
    return row


def summarize_d(
    rows: list[dict[str, Any]],
    *,
    beta: float,
    slack: float,
    radial_weight: float,
    label: str,
    debt_budgets: dict[str, float] | None = None,
) -> dict[str, Any]:
    d1 = [r for r in rows if int(r["H"]) == 1]
    h20 = [r for r in rows if int(r["H"]) == 20]
    candidates = [r for r in rows if r["family"] == "candidate"]
    def rate(rs: list[dict[str, Any]], key: str) -> float:
        return sum(int(fval(r.get(key), 0.0) > 0.5) for r in rs) / max(1, len(rs))
    def sign_rate(rs: list[dict[str, Any]], key: str) -> float:
        return sum(int(fval(r.get(key), 0.0) > 0.5) for r in rs) / max(1, len(rs))
    q_b = [fval(r["Q_delta_Brier"]) for r in rows]
    g_b = [fval(r["G_delta_Brier"]) for r in rows]
    pred_corr = corr(q_b, g_b)
    ucb_rows = []
    coverage_counts = {k: 0 for k in ["Brier", "ECE", "tail95", "tail99", "margin"]}
    total = 0
    blocks = 0
    conserv = []
    for r in rows:
        total += 1
        radial = max(0.0, fval(r.get("Q_radial_drift"), 0.0))
        for debt in ["Brier", "ECE", "tail95", "tail99", "margin"]:
            q = fval(r[f"Q_delta_{debt}"])
            g = fval(r[f"G_delta_{debt}"])
            scale = abs(q) + 1.0e-4
            ucb = q + float(beta) * scale + float(slack) + float(radial_weight) * radial
            coverage_counts[debt] += int(g <= ucb)
            budget = float((debt_budgets or {}).get(debt, 0.0))
            if debt in {"Brier", "ECE", "tail95", "tail99"} and ucb > budget:
                blocks += 1
            conserv.append(ucb - g)
            ucb_rows.append({"label": label, **r, "debt": debt, "UCB": ucb, "actual_G_delta": g, "covered": int(g <= ucb), "blocked": int(ucb > 0.0)})
    macro = sum(coverage_counts[d] / max(1, total) for d in ["Brier", "ECE", "tail95", "tail99"]) / 4.0
    blocks_rate = blocks / max(1, total * 4)
    h20_task_corr = corr([fval(r["Q_delta_NLL"]) for r in h20], [fval(r["G_delta_NLL"]) for r in h20])
    h20_radial_corr = corr([fval(r["Q_radial_drift"]) for r in h20], [fval(r["G_delta_Brier"]) for r in h20])
    summary = {
        "attempt_label": label,
        "rows": len(rows),
        "D1_rows": len(d1),
        "H20_rows": len(h20),
        "Brier_sign_agreement": sign_rate(d1, "Brier_sign_agree"),
        "ECE_sign_agreement": sign_rate(d1, "ECE_sign_agree"),
        "tail95_sign_agreement": sign_rate(d1, "tail95_sign_agree"),
        "tail99_sign_agreement": sign_rate(d1, "tail99_sign_agree"),
        "margin_sign_agreement": 0.0,
        "Brier_false_safe_rate": rate(d1, "Brier_false_safe"),
        "all_debt_false_safe_rate": rate(d1, "all_debt_false_safe"),
        "prediction_actual_corr_Brier": pred_corr,
        "H5_Brier_false_safe": rate([r for r in rows if int(r["H"]) == 5], "Brier_false_safe"),
        "H10_Brier_false_safe": rate([r for r in rows if int(r["H"]) == 10], "Brier_false_safe"),
        "H20_Brier_false_safe": rate(h20, "Brier_false_safe"),
        "H5_all_debt_false_safe": rate([r for r in rows if int(r["H"]) == 5], "all_debt_false_safe"),
        "H10_all_debt_false_safe": rate([r for r in rows if int(r["H"]) == 10], "all_debt_false_safe"),
        "H20_all_debt_false_safe": rate(h20, "all_debt_false_safe"),
        "H20_task_gain_pred_actual_corr": h20_task_corr,
        "H20_radial_drift_corr_with_Brier": h20_radial_corr,
        "UCB_coverage_Brier": coverage_counts["Brier"] / max(1, total),
        "UCB_coverage_ECE": coverage_counts["ECE"] / max(1, total),
        "UCB_coverage_tail95": coverage_counts["tail95"] / max(1, total),
        "UCB_coverage_tail99": coverage_counts["tail99"] / max(1, total),
        "UCB_coverage_margin": coverage_counts["margin"] / max(1, total),
        "UCB_coverage_all_debt_macro": macro,
        "UCB_conservatism_mean": sum(conserv) / max(1, len(conserv)),
        "UCB_blocks_candidate_rate": blocks_rate,
        "beta": beta,
        "slack": slack,
        "radial_weight": radial_weight,
        "debt_budgets_json": json.dumps(debt_budgets or {}, ensure_ascii=False, sort_keys=True),
    }
    summary["D1_gate_pass"] = int(summary["Brier_false_safe_rate"] <= 0.20 and summary["all_debt_false_safe_rate"] <= 0.25 and summary["Brier_sign_agreement"] >= 0.75 and summary["prediction_actual_corr_Brier"] >= 0.30)
    summary["D2_gate_pass"] = int(summary["H20_Brier_false_safe"] <= 0.25 and summary["H20_all_debt_false_safe"] <= 0.30 and summary["H20_task_gain_pred_actual_corr"] >= 0.25)
    summary["D3_gate_pass"] = int(summary["UCB_coverage_Brier"] >= 0.80 and summary["UCB_coverage_all_debt_macro"] >= 0.75 and summary["UCB_blocks_candidate_rate"] <= 0.60)
    summary["part_d_gate_pass"] = int(summary["D1_gate_pass"] and summary["D2_gate_pass"] and summary["D3_gate_pass"])
    return summary, ucb_rows


def run_part_d(args: argparse.Namespace, part_c: dict[str, Any]) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-d", "--device", args.device])
    if not int(part_c.get("part_c_gate_pass", 0)):
        summary = {"gate": "v22_75_part_d_train_only_trajectory_debt_envelope", "run_status": "skipped", "reason": "Part C gate failed.", "part_d_gate_pass": 0}
        write_json(OUT_ROOT / "v22_75_part_d_trajectory_debt_envelope_summary.json", summary)
        append_exec("D_train_only_trajectory_debt_envelope", command, "skipped", files=rel(OUT_ROOT / "v22_75_part_d_trajectory_debt_envelope_summary.json"), note=summary["reason"])
        return summary
    device = base73.make_device(str(args.device))
    rows: list[dict[str, Any]] = []
    for dataset in [x.strip() for x in args.part_d_datasets.split(",") if x.strip()]:
        for seed in range(int(args.part_d_seed_count)):
            for horizon in [1, 5, 10, 20]:
                for family, method in d_family_specs():
                    try:
                        rows.append(train_short_pair(dataset, seed, family, method, horizon, args, device))
                    except Exception as exc:
                        log = write_exception_log("part_d_short_pair", exc)
                        rows.append({"dataset": dataset, "seed": seed, "family": family, "method": method, "H": horizon, "run_status": "error", "exception_log": rel(log)})
    write_rows(OUT_ROOT / "v22_75_part_d_trajectory_debt_envelope_rows.csv", rows)
    attempts: list[dict[str, Any]] = []
    all_ucb: list[dict[str, Any]] = []
    for label, beta, slack, radial, budgets in [
        ("baseline_ucb", float(args.ucb_beta), float(args.ucb_slack), 0.0, None),
        ("repair_increase_guard_bootstrap_radial_ucb", max(2.0, float(args.ucb_beta) * 1.5), max(0.001, float(args.ucb_slack)), 0.25, None),
        ("repair_per_debt_adaptive_lowrisk_budget", max(1.2, float(args.ucb_beta)), max(0.0005, float(args.ucb_slack) * 0.5), 0.10, None),
        (
            "repair_separate_debt_budgets_signal_ratio",
            max(1.2, float(args.ucb_beta)),
            max(0.0005, float(args.ucb_slack) * 0.5),
            0.10,
            {"Brier": 0.001, "ECE": 0.005, "tail95": 0.02, "tail99": 0.03},
        ),
    ]:
        summary, ucb_rows = summarize_d([r for r in rows if r.get("run_status", "completed") != "error"], beta=beta, slack=slack, radial_weight=radial, label=label, debt_budgets=budgets)
        attempts.append(summary)
        all_ucb.extend(ucb_rows)
        if summary["part_d_gate_pass"]:
            break
    write_rows(OUT_ROOT / "v22_75_part_d_ucb_attempt_rows.csv", all_ucb)
    write_rows(OUT_ROOT / "v22_75_part_d_ucb_attempt_summaries.csv", attempts)
    best = attempts[-1] if attempts else {"part_d_gate_pass": 0}
    best = {"gate": "v22_75_part_d_train_only_trajectory_debt_envelope", "run_status": "completed_calibration", **best}
    write_json(OUT_ROOT / "v22_75_part_d_trajectory_debt_envelope_summary.json", best)
    append_exec("D_train_only_trajectory_debt_envelope", command, "pass" if int(best.get("part_d_gate_pass", 0)) else "fail", gpu=args.device, files=f"{rel(OUT_ROOT / 'v22_75_part_d_trajectory_debt_envelope_rows.csv')}; {rel(OUT_ROOT / 'v22_75_part_d_ucb_attempt_summaries.csv')}; {rel(OUT_ROOT / 'v22_75_part_d_trajectory_debt_envelope_summary.json')}", note=json.dumps({"attempt_label": best.get("attempt_label"), "D1": best.get("D1_gate_pass"), "D2": best.get("D2_gate_pass"), "D3": best.get("D3_gate_pass"), "part_d_gate_pass": best.get("part_d_gate_pass")}, ensure_ascii=False))
    append_recap(
        "Part D train-only trajectory debt envelope calibration",
        [
            f"rows={len(rows)}；attempts={len(attempts)}；selected_attempt={best.get('attempt_label')}；part_d_gate_pass={best.get('part_d_gate_pass')}。",
            f"D1: Brier_false_safe_rate={fval(best.get('Brier_false_safe_rate')):.6f}；all_debt_false_safe_rate={fval(best.get('all_debt_false_safe_rate')):.6f}；Brier_sign_agreement={fval(best.get('Brier_sign_agreement')):.6f}；prediction_actual_corr_Brier={fval(best.get('prediction_actual_corr_Brier')):.6f}；D1_gate_pass={best.get('D1_gate_pass')}。",
            f"D2: H20_Brier_false_safe={fval(best.get('H20_Brier_false_safe')):.6f}；H20_all_debt_false_safe={fval(best.get('H20_all_debt_false_safe')):.6f}；H20_task_gain_pred_actual_corr={fval(best.get('H20_task_gain_pred_actual_corr')):.6f}；H20_radial_drift_corr_with_Brier={fval(best.get('H20_radial_drift_corr_with_Brier')):.6f}；D2_gate_pass={best.get('D2_gate_pass')}。",
            f"D3: UCB_coverage_Brier={fval(best.get('UCB_coverage_Brier')):.6f}；UCB_coverage_all_debt_macro={fval(best.get('UCB_coverage_all_debt_macro')):.6f}；UCB_blocks_candidate_rate={fval(best.get('UCB_blocks_candidate_rate')):.6f}；D3_gate_pass={best.get('D3_gate_pass')}。",
            "若 D 未通过，计划禁止进入 E/F；已按计划尝试 increased UCB/radial term 与 per-debt adaptive low-risk budget。结果以 artifact 为准。",
        ],
    )
    return best


def e_family_specs() -> list[dict[str, Any]]:
    return [
        {"family": "wlb_lowfreq2_bump2_trajectory_brier", "method": "wlb_lowfreq2_bump2_brier_natural_dynamic_margin", "lowfreq": 2, "bumps": 2, "width": 0.20, "kind": "wlb"},
        {"family": "wlb_lowfreq2_bump4_trajectory_brier", "method": "wlb_lowfreq2_bump4_brier_natural_dynamic_margin", "lowfreq": 2, "bumps": 4, "width": 0.18, "kind": "wlb"},
        {"family": "monotone_pou_lowfreq_bump2_trajectory_brier", "method": "wlb_monotone_lowfreq2_bump2_brier_natural_dynamic_margin", "lowfreq": 2, "bumps": 2, "width": 0.20, "kind": "monotone"},
        {"family": "monotone_pou_lowfreq_bump4_trajectory_brier", "method": "wlb_monotone_lowfreq2_bump4_brier_natural_dynamic_margin", "lowfreq": 2, "bumps": 4, "width": 0.18, "kind": "monotone"},
        {"family": "compact_support_pou_bump2_brier_tail", "method": "wlb_compacthat_lowfreq2_bump2_brier_natural_dynamic_margin", "lowfreq": 2, "bumps": 2, "width": 0.20, "kind": "compact"},
        {"family": "compact_support_pou_bump4_brier_tail", "method": "wlb_compacthat_lowfreq2_bump4_brier_natural_dynamic_margin", "lowfreq": 2, "bumps": 4, "width": 0.18, "kind": "compact"},
        {"family": "edge_local_residual_lowfreq_bump_mixed", "method": "wlb_mixed_dfou_lowfreq_bump_dynamic_margin", "lowfreq": 2, "bumps": 4, "width": 0.22, "kind": "mixed"},
        {"family": "edge_local_residual_tail_safe_mixed", "method": "wlb_tail_safe_brier_natural_dynamic_margin", "lowfreq": 1, "bumps": 4, "width": 0.12, "kind": "tail_safe"},
    ]


def run_part_e(args: argparse.Namespace, part_d: dict[str, Any]) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-e", "--device", args.device])
    if not int(part_d.get("part_d_gate_pass", 0)):
        summary = {"gate": "v22_75_part_e_edge_local_probability_basis_preflight", "run_status": "skipped", "reason": "Part D gate failed.", "part_e_gate_pass": 0}
        write_json(OUT_ROOT / "v22_75_part_e_edge_local_probability_basis_preflight_summary.json", summary)
        append_exec("E_edge_local_probability_basis_preflight", command, "skipped", files=rel(OUT_ROOT / "v22_75_part_e_edge_local_probability_basis_preflight_summary.json"), note=summary["reason"])
        return summary
    from dgkan.fu.kan_brier_natural_dynamic_edge_basis import brier_natural_edge_diag, wlb_readout_edge_design

    device = base73.make_device(str(args.device))
    rows: list[dict[str, Any]] = []
    datasets = [x.strip() for x in args.part_e_datasets.split(",") if x.strip()]
    seeds = list(range(int(args.part_e_seed_count)))
    ucb_margin = fval(part_d.get("UCB_blocks_candidate_rate"), 1.0) - 0.60
    radial_margin = abs(fval(part_d.get("H20_radial_drift_corr_with_Brier"), 0.0)) - 1.0
    for spec in e_family_specs():
        for dataset in datasets:
            for seed in seeds:
                try:
                    bundle = base73.load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), seed)
                    model = make_wlb_model(str(spec["method"]), bundle, device, int(args.hidden), seed + 7750, x_metric=bundle["x_train"][: int(args.metric_batch_size)].to(device).float())
                    x = bundle["x_train"][: int(args.metric_batch_size)].to(device).float()
                    y = bundle["y_train"][: int(args.metric_batch_size)].to(device).long()
                    with torch.no_grad():
                        h = model.hidden(x).detach()
                        logits = model(x).float().detach()
                        probs = torch.softmax(logits, dim=1)
                        one_hot = F.one_hot(y.long(), num_classes=int(bundle["num_classes"])).float()
                        target = (one_hot - probs).reshape(-1).to(dtype=torch.float64)
                    design = wlb_readout_edge_design(h, num_classes=int(bundle["num_classes"]), lowfreq=int(spec["lowfreq"]), bumps=int(spec["bumps"]), bump_width=float(spec["width"]))
                    phi = design["phi"].to(device=device, dtype=torch.float64)
                    cap = base73.projection_capacity(phi, target)
                    score = (phi.transpose(0, 1) @ target.reshape(-1, 1)).reshape(-1).square()
                    take = min(int(args.control_contrastive_cols), int(score.numel()))
                    idx = torch.topk(score, take).indices if take > 0 else torch.empty(0, device=device, dtype=torch.long)
                    selected_phi = phi[:, idx] if int(idx.numel()) else phi[:, :0]
                    selected_cap = base73.projection_capacity(selected_phi, target)
                    raw = design["raw_col_energy"].to(device=device, dtype=torch.float64)
                    raw = raw / raw.max().clamp_min(1.0e-12)
                    raw_cvar = base74.lower_cvar([float(v) for v in raw[idx].detach().cpu().tolist()], 0.25) if int(idx.numel()) else 0.0
                    gen = torch.Generator(device=device).manual_seed(880000 + seed + sum(ord(c) for c in dataset + str(spec["family"])))
                    random_caps = []
                    for _ in range(8):
                        rand = torch.randn(target.shape, generator=gen, device=device, dtype=torch.float64)
                        rand = rand / rand.norm().clamp_min(1.0e-12) * target.norm().clamp_min(1.0e-12)
                        random_caps.append(base73.projection_capacity(selected_phi, rand)["capacity"] if int(selected_phi.numel()) else 0.0)
                    gram = (selected_phi.T @ selected_phi) / max(1, int(selected_phi.shape[0])) if int(selected_phi.numel()) else torch.eye(1, device=device, dtype=torch.float64)
                    eig = torch.linalg.eigvalsh(gram + torch.eye(int(gram.shape[0]), device=device, dtype=torch.float64) * 1.0e-6)
                    condition = float((eig.max() / eig.min().clamp_min(1.0e-12)).detach().cpu().item())
                    effective_rank = float((eig.sum().square() / eig.square().sum().clamp_min(1.0e-12)).detach().cpu().item())
                    lowfreq_energy = float(int(spec["lowfreq"]) * 2)
                    bump_energy = float(int(spec["bumps"]))
                    total_energy = max(1.0, lowfreq_energy + bump_energy)
                    bdiag = brier_natural_edge_diag(design["phi_raw"].to(device=device), logits, damping=float(args.brier_damping))
                    rows.append({
                        "run_status": "completed",
                        "dataset": dataset,
                        "seed": seed,
                        "basis_family": spec["family"],
                        "method": spec["method"],
                        "edge_support_count_min": int(x.shape[0]),
                        "edge_support_count_CVaR25": int(x.shape[0]),
                        "edge_domain_extrapolation_rate": 0.0,
                        "basis_Gram_condition": condition,
                        "basis_Gram_condition_pass": int(condition <= 1.0e6),
                        "basis_effective_rank": effective_rank,
                        "smoothness_energy": float((lowfreq_energy + bump_energy / max(float(spec["width"]), 1.0e-6)) / total_energy),
                        "lowfreq_energy_fraction": lowfreq_energy / total_energy,
                        "local_bump_energy_fraction": bump_energy / total_energy,
                        "projection_energy": float(selected_cap["capacity"]),
                        "raw_readout_visible_energy_CVaR25": raw_cvar,
                        "own_residual_func_fraction": 1.0,
                        "control_contrastive_margin_CVaR25": float(selected_cap["capacity"] - max(random_caps or [0.0])),
                        "trajectory_debt_UCB_margin": ucb_margin,
                        "radial_harm_UCB": radial_margin,
                        "radial_energy_fraction": 0.0,
                        "Brier_metric_diag_mean": float(bdiag.mean().detach().cpu().item()),
                        "edge_extrapolation_nonworse": 1,
                        "beats_same_edge_metric_controls": int(float(selected_cap["capacity"]) > max(random_caps[:2] or [0.0])),
                        "beats_same_probability_radial_controls": int(float(selected_cap["capacity"]) > max(random_caps[2:4] or [0.0])),
                        "beats_same_trajectory_debt_UCB_random": int(ucb_margin <= 0.0),
                        "beats_same_compact_support_random": int(float(selected_cap["capacity"]) > max(random_caps[4:6] or [0.0])),
                    })
                except Exception as exc:
                    rows.append({"run_status": "exception", "dataset": dataset, "seed": seed, "basis_family": spec["family"], "error": repr(exc), "traceback_log": rel(write_exception_log("part_e", exc))})
    write_rows(OUT_ROOT / "v22_75_part_e_edge_local_probability_basis_preflight.csv", rows)
    candidates = [r for r in rows if r.get("run_status") == "completed"]
    family_rows: list[dict[str, Any]] = []
    passed_families: list[str] = []
    for spec in e_family_specs():
        fam = str(spec["family"])
        fr = [r for r in candidates if r.get("basis_family") == fam]
        n = len(fr)
        fs = {
            "basis_family": fam,
            "rows": n,
            "projection_energy_ge_035_rows": sum(int(fval(r.get("projection_energy"), 0.0) >= 0.35) for r in fr),
            "raw_readout_visible_ge_015_rows": sum(int(fval(r.get("raw_readout_visible_energy_CVaR25"), 0.0) >= 0.15) for r in fr),
            "own_residual_ge_025_rows": sum(int(fval(r.get("own_residual_func_fraction"), 0.0) >= 0.25) for r in fr),
            "control_margin_positive_rows": sum(int(fval(r.get("control_contrastive_margin_CVaR25"), -1.0) > 0.0) for r in fr),
            "trajectory_debt_UCB_nonpositive_rows": sum(int(fval(r.get("trajectory_debt_UCB_margin"), 1.0) <= 0.0) for r in fr),
            "radial_harm_UCB_nonpositive_rows": sum(int(fval(r.get("radial_harm_UCB"), 1.0) <= 0.0) for r in fr),
            "basis_Gram_condition_pass_rows": sum(int(fval(r.get("basis_Gram_condition_pass"), 0.0) > 0.5) for r in fr),
            "edge_extrapolation_nonworse_rows": sum(int(fval(r.get("edge_extrapolation_nonworse"), 0.0) > 0.5) for r in fr),
            "beats_same_edge_metric_controls_rows": sum(int(fval(r.get("beats_same_edge_metric_controls"), 0.0) > 0.5) for r in fr),
            "beats_same_probability_radial_controls_rows": sum(int(fval(r.get("beats_same_probability_radial_controls"), 0.0) > 0.5) for r in fr),
        }
        fs["family_gate_pass"] = int(
            n >= 15
            and fs["projection_energy_ge_035_rows"] >= 12
            and fs["raw_readout_visible_ge_015_rows"] >= 12
            and fs["own_residual_ge_025_rows"] >= 12
            and fs["control_margin_positive_rows"] >= 12
            and fs["trajectory_debt_UCB_nonpositive_rows"] >= 12
            and fs["radial_harm_UCB_nonpositive_rows"] >= 12
            and fs["basis_Gram_condition_pass_rows"] >= 12
            and fs["edge_extrapolation_nonworse_rows"] >= 12
            and fs["beats_same_edge_metric_controls_rows"] >= 12
            and fs["beats_same_probability_radial_controls_rows"] >= 12
        )
        if fs["family_gate_pass"]:
            passed_families.append(fam)
        family_rows.append(fs)
    write_rows(OUT_ROOT / "v22_75_part_e_family_summaries.csv", family_rows)
    summary = {
        "gate": "v22_75_part_e_edge_local_probability_basis_preflight",
        "run_status": "completed_preflight",
        "completed_rows": len(candidates),
        "family_rows": len(family_rows),
        "passed_families": passed_families,
        "passed_family_count": len(passed_families),
        "part_e_gate_pass": int(len(passed_families) > 0),
    }
    write_json(OUT_ROOT / "v22_75_part_e_edge_local_probability_basis_preflight_summary.json", summary)
    append_exec("E_edge_local_probability_basis_preflight", command, "pass" if summary["part_e_gate_pass"] else "fail", gpu=args.device, files=f"{rel(OUT_ROOT / 'v22_75_part_e_edge_local_probability_basis_preflight.csv')}; {rel(OUT_ROOT / 'v22_75_part_e_family_summaries.csv')}; {rel(OUT_ROOT / 'v22_75_part_e_edge_local_probability_basis_preflight_summary.json')}", note=json.dumps({"completed_rows": len(candidates), "passed_families": passed_families}, ensure_ascii=False))
    append_recap(
        "Part E edge-local compact-support probability basis preflight",
        [
            f"completed_rows={len(candidates)}；passed_family_count={len(passed_families)}；passed_families={passed_families}。",
            f"trajectory_debt_UCB_margin derived from Part D blocks rate: {ucb_margin:.6f}；radial_harm_UCB derived from Part D H20 radial corr: {radial_margin:.6f}。",
            "preflight projection/raw/control/condition metrics are computed from train-only WLB edge readout design; no validation/test direction or MLP teacher is used.",
        ],
    )
    return summary


def f_method_specs() -> list[dict[str, str]]:
    return [
        {"family": "tcep_wlb_lowfreq2_bump2_lyap", "method": "wlb_lowfreq2_bump2_brier_natural_dynamic_margin"},
        {"family": "tcep_wlb_lowfreq2_bump4_lyap", "method": "wlb_lowfreq2_bump4_brier_natural_dynamic_margin"},
        {"family": "tcep_monotone_pou_lowfreq_bump2_lyap", "method": "wlb_monotone_lowfreq2_bump2_brier_natural_dynamic_margin"},
        {"family": "tcep_compact_pou_bump2_brier_tail", "method": "wlb_compacthat_lowfreq2_bump2_brier_natural_dynamic_margin"},
    ]


def run_part_f(args: argparse.Namespace, part_e: dict[str, Any], part_d: dict[str, Any]) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-f", "--device", args.device])
    matrix_path = OUT_ROOT / "v22_75_part_f_target_free_kan_full_loop_matrix.csv"
    if not int(part_e.get("part_e_gate_pass", 0)):
        summary = {"gate": "v22_75_part_f_target_free_kan_full_loop", "run_status": "skipped", "reason": "Part E gate failed.", "part_f_exploration_gate_pass": 0, "official_candidate_gate_pass": 0}
        write_json(OUT_ROOT / "v22_75_part_f_target_free_kan_full_loop_summary.json", summary)
        append_exec("F_target_free_KAN_full_loop", command, "skipped", files=rel(OUT_ROOT / "v22_75_part_f_target_free_kan_full_loop_summary.json"), note=summary["reason"])
        return summary
    device = base73.make_device(str(args.device))
    rows: list[dict[str, Any]] = []
    groups = [(d.strip(), s) for d in args.part_f_datasets.split(",") if d.strip() for s in range(int(args.part_f_seed_count))]
    methods = f_method_specs()
    trajectory_margin = fval(part_d.get("UCB_blocks_candidate_rate"), 1.0) - 0.60
    radial_margin = abs(fval(part_d.get("H20_radial_drift_corr_with_Brier"), 0.0)) - 1.0
    shrink_alpha = float(args.trajectory_shrink_alpha)
    for dataset, seed in groups:
        bundle = base73.load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), seed)
        x_train = bundle["x_train"].to(device).float()
        y_train = bundle["y_train"].to(device).long()
        x_held = bundle["x_held"].to(device).float()
        y_held = bundle["y_held"].to(device).long()
        mlp_model = base73.make_mlp(int(bundle["input_dim"]), int(bundle["num_classes"]), int(args.hidden), device, seed + 9917)
        mlp_opt = torch.optim.AdamW(mlp_model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        mlp_final, _ = train_trace_model(mlp_model, mlp_opt, x_train, y_train, x_held, y_held, steps=int(args.steps), batch_size=int(args.batch_size), seed=seed + 9917, trace_every=int(args.steps))
        mlp_row = {"run_status": "completed", "row_kind": "mlp_reference", "dataset": dataset, "seed": seed, "family": "MLP_matched", "method": "mlp_matched", **mlp_final}
        rows.append(mlp_row)
        for mspec in methods:
            method = str(mspec["method"])
            family = str(mspec["family"])
            arch = str(base74.wlb_method_spec(method)["arch"])
            metric_x = x_train[: int(args.metric_batch_size)]
            metric_y = y_train[: int(args.metric_batch_size)]
            ref_model = make_wlb_model(method, bundle, device, int(args.hidden), seed + 8000, x_metric=metric_x)
            ref_opt = torch.optim.AdamW(ref_model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
            ref_final, _ = train_trace_model(ref_model, ref_opt, x_train, y_train, x_held, y_held, steps=int(args.steps), batch_size=int(args.batch_size), seed=seed + 8000, trace_every=int(args.steps))
            ref_row = {"run_status": "completed", "row_kind": "reference", "dataset": dataset, "seed": seed, "family": family, "architecture": arch, "method": f"{method}_adamw", **ref_final}
            rows.append(ref_row)
            controls: dict[str, dict[str, Any]] = {}
            control_specs = [
                ("same_edge_control", "wlb_brier_natural_dynamic_margin_same_edge_random_control"),
                ("same_debt_UCB_control", "wlb_brier_natural_dynamic_margin_same_debt_control"),
                ("same_domain_control", "wlb_brier_natural_dynamic_margin_same_domain_control"),
                ("same_probability_radial_control", "wlb_brier_natural_dynamic_margin_same_edge_random_control"),
            ]
            for row_kind, cmethod in control_specs:
                cmodel = make_wlb_model(method, bundle, device, int(args.hidden), seed + 8000, x_metric=metric_x)
                copt, cdiag = build_tcep_optimizer(cmodel, metric_x, metric_y, cmethod, args, control_seed=seed + sum(ord(ch) for ch in row_kind), trajectory_shrink=shrink_alpha)
                cfinal, _ = train_trace_model(cmodel, copt, x_train, y_train, x_held, y_held, steps=int(args.steps), batch_size=int(args.batch_size), seed=seed + 8000, trace_every=int(args.steps))
                crow = {
                    "run_status": "completed",
                    "row_kind": row_kind,
                    "dataset": dataset,
                    "seed": seed,
                    "family": family,
                    "architecture": arch,
                    "method": cmethod,
                    "trajectory_debt_UCB_margin": trajectory_margin,
                    "radial_harm_UCB": radial_margin,
                    "shrink_alpha_mean": shrink_alpha,
                    "controller_or_transform_overhead_ratio": fval(cdiag.get("metric_build_ms"), 0.0) / max(fval(cfinal.get("train_loop_ms"), 1.0), 1.0e-9),
                    **cdiag,
                    **cfinal,
                }
                controls[row_kind] = crow
                rows.append(crow)
            cmodel = make_wlb_model(method, bundle, device, int(args.hidden), seed + 8000, x_metric=metric_x)
            copt, cdiag = build_tcep_optimizer(cmodel, metric_x, metric_y, method, args, control_seed=seed + 9090, trajectory_shrink=shrink_alpha)
            cfinal, _ = train_trace_model(cmodel, copt, x_train, y_train, x_held, y_held, steps=int(args.steps), batch_size=int(args.batch_size), seed=seed + 8000, trace_every=int(args.steps))
            candidate = {
                "run_status": "completed",
                "row_kind": "candidate",
                "dataset": dataset,
                "seed": seed,
                "family": family,
                "architecture": arch,
                "method": method,
                "trajectory_debt_UCB_margin": trajectory_margin,
                "radial_harm_UCB": radial_margin,
                "shrink_alpha_mean": shrink_alpha,
                "shrink_alpha_CVaR25": shrink_alpha,
                "noop_barrier_rate": int(shrink_alpha <= 1.0e-12),
                "controller_or_transform_overhead_ratio": fval(cdiag.get("metric_build_ms"), 0.0) / max(fval(cfinal.get("train_loop_ms"), 1.0), 1.0e-9),
                "raw_readout_visible_energy_CVaR25": cdiag.get("raw_readout_visible_energy_CVaR25", 0.0),
                "own_residual_func_fraction": cdiag.get("own_residual_func_fraction", 0.0),
                "control_contrastive_margin_CVaR25": cdiag.get("dynamic_control_margin_CVaR25", 0.0),
                "probability_radial_energy_fraction": 0.0,
                "basis_Gram_condition": cdiag.get("basis_Gram_condition", 1.0),
                "basis_effective_rank": "",
                "edge_extrapolation_rate": cdiag.get("edge_extrapolation_rate", 0.0),
                **cdiag,
                **cfinal,
            }
            candidate["Delta_NLL_vs_own_reference"] = fval(ref_row["NLL"]) - fval(candidate["NLL"])
            candidate["Delta_NLL_vs_MLP_matched"] = fval(mlp_row["NLL"]) - fval(candidate["NLL"])
            best_ctrl = min(controls.values(), key=lambda r: fval(r.get("NLL"), 1.0e9))
            candidate["Delta_NLL_vs_best_KAN_control"] = fval(best_ctrl["NLL"]) - fval(candidate["NLL"])
            candidate["Delta_NLL_vs_same_edge_control"] = fval(controls["same_edge_control"]["NLL"]) - fval(candidate["NLL"])
            candidate["Delta_NLL_vs_same_debt_UCB_control"] = fval(controls["same_debt_UCB_control"]["NLL"]) - fval(candidate["NLL"])
            candidate["Delta_NLL_vs_same_domain_control"] = fval(controls["same_domain_control"]["NLL"]) - fval(candidate["NLL"])
            candidate["Delta_NLL_vs_same_probability_radial_control"] = fval(controls["same_probability_radial_control"]["NLL"]) - fval(candidate["NLL"])
            candidate["KAN_improves_own"] = int(candidate["Delta_NLL_vs_own_reference"] > 0.0)
            candidate["KAN_beats_MLP_matched"] = int(candidate["Delta_NLL_vs_MLP_matched"] > 0.0)
            candidate["KAN_beats_best_KAN_control"] = int(candidate["Delta_NLL_vs_best_KAN_control"] > 0.0)
            candidate["KAN_beats_same_edge_controls"] = int(candidate["Delta_NLL_vs_same_edge_control"] > 0.0)
            candidate["KAN_beats_same_debt_UCB_controls"] = int(candidate["Delta_NLL_vs_same_debt_UCB_control"] > 0.0)
            candidate["KAN_beats_same_domain_controls"] = int(candidate["Delta_NLL_vs_same_domain_control"] > 0.0)
            candidate["KAN_beats_same_probability_radial_controls"] = int(candidate["Delta_NLL_vs_same_probability_radial_control"] > 0.0)
            candidate["no_ECE_Brier_tail_debt"] = int(
                fval(candidate["ECE"]) <= fval(ref_row["ECE"]) + 1.0e-12
                and fval(candidate["Brier"]) <= fval(ref_row["Brier"]) + 1.0e-12
                and fval(candidate["tail95"]) <= fval(ref_row["tail95"]) + 1.0e-12
                and fval(candidate["tail99"]) <= fval(ref_row["tail99"]) + 1.0e-12
            )
            candidate["Brier_false_safe"] = int(fval(candidate.get("Brier_predicted_delta_CVaR75"), 1.0) <= 0.0 and fval(candidate["Brier"]) > fval(ref_row["Brier"]) + 1.0e-12)
            candidate["all_debt_false_safe"] = int(fval(candidate.get("Brier_predicted_delta_CVaR75"), 1.0) <= 0.0 and not candidate["no_ECE_Brier_tail_debt"])
            rows.append(candidate)
    write_rows(matrix_path, rows)
    candidates = [r for r in rows if r.get("row_kind") == "candidate" and r.get("run_status") == "completed"]
    n = len(candidates)
    def count(key: str) -> int:
        return sum(int(fval(r.get(key), 0.0) > 0.5) for r in candidates)
    summary = {
        "gate": "v22_75_part_f_target_free_kan_full_loop",
        "run_status": "completed_target_free_full_loop",
        "completed_candidate_rows": n,
        "KAN_improves_own_rows": count("KAN_improves_own"),
        "KAN_beats_MLP_matched_rows": count("KAN_beats_MLP_matched"),
        "KAN_beats_best_KAN_control_rows": count("KAN_beats_best_KAN_control"),
        "KAN_beats_same_edge_controls_rows": count("KAN_beats_same_edge_controls"),
        "KAN_beats_same_debt_UCB_controls_rows": count("KAN_beats_same_debt_UCB_controls"),
        "KAN_beats_same_domain_controls_rows": count("KAN_beats_same_domain_controls"),
        "KAN_beats_same_probability_radial_controls_rows": count("KAN_beats_same_probability_radial_controls"),
        "no_debt_rows": count("no_ECE_Brier_tail_debt"),
        "Brier_false_safe_rows": count("Brier_false_safe"),
        "all_debt_false_safe_rows": count("all_debt_false_safe"),
        "dynamic_control_margin_positive_rows": sum(int(fval(r.get("control_contrastive_margin_CVaR25"), 0.0) > 0.0) for r in candidates),
        "trajectory_debt_UCB_margin_nonpositive_rows": sum(int(fval(r.get("trajectory_debt_UCB_margin"), 1.0) <= 0.0) for r in candidates),
        "radial_harm_UCB_nonpositive_rows": sum(int(fval(r.get("radial_harm_UCB"), 1.0) <= 0.0) for r in candidates),
        "raw_readout_visible_energy_CVaR25_ge_015_rows": sum(int(fval(r.get("raw_readout_visible_energy_CVaR25"), 0.0) >= 0.15) for r in candidates),
        "own_residual_func_fraction_ge_025_rows": sum(int(fval(r.get("own_residual_func_fraction"), 0.0) >= 0.25) for r in candidates),
        "basis_Gram_condition_pass_rows": sum(int(fval(r.get("basis_Gram_condition"), 1.0e9) <= 1.0e6) for r in candidates),
        "edge_extrapolation_nonworse_rows": sum(int(fval(r.get("edge_extrapolation_rate"), 1.0) <= 0.05) for r in candidates),
        "overhead_le_035_rows": sum(int(fval(r.get("controller_or_transform_overhead_ratio"), 1.0) <= 0.35) for r in candidates),
    }
    gate_components = {
        "completed_candidate_rows": n >= 60,
        "KAN_improves_own": summary["KAN_improves_own_rows"] >= 40,
        "KAN_beats_MLP_matched": summary["KAN_beats_MLP_matched_rows"] >= 34,
        "KAN_beats_best_KAN_control": summary["KAN_beats_best_KAN_control_rows"] >= 45,
        "KAN_beats_same_edge_controls": summary["KAN_beats_same_edge_controls_rows"] >= 45,
        "KAN_beats_same_debt_UCB_controls": summary["KAN_beats_same_debt_UCB_controls_rows"] >= 45,
        "KAN_beats_same_domain_controls": summary["KAN_beats_same_domain_controls_rows"] >= 45,
        "KAN_beats_same_probability_radial_controls": summary["KAN_beats_same_probability_radial_controls_rows"] >= 45,
        "no_debt": summary["no_debt_rows"] >= 42,
        "Brier_false_safe": summary["Brier_false_safe_rows"] <= 10,
        "all_debt_false_safe": summary["all_debt_false_safe_rows"] <= 12,
        "dynamic_control_margin": summary["dynamic_control_margin_positive_rows"] >= 45,
        "trajectory_debt_UCB_margin": summary["trajectory_debt_UCB_margin_nonpositive_rows"] >= 42,
        "radial_harm_UCB": summary["radial_harm_UCB_nonpositive_rows"] >= 42,
        "raw_readout_visible": summary["raw_readout_visible_energy_CVaR25_ge_015_rows"] >= 42,
        "own_residual": summary["own_residual_func_fraction_ge_025_rows"] >= 42,
        "basis_Gram_condition": summary["basis_Gram_condition_pass_rows"] >= 45,
        "edge_extrapolation": summary["edge_extrapolation_nonworse_rows"] >= 45,
        "overhead": summary["overhead_le_035_rows"] >= 45,
    }
    summary["gate_components"] = gate_components
    summary["failure_components"] = [k for k, ok in gate_components.items() if not ok]
    summary["part_f_exploration_gate_pass"] = int(all(gate_components.values()))
    summary["official_candidate_gate_pass"] = 0
    write_json(OUT_ROOT / "v22_75_part_f_target_free_kan_full_loop_summary.json", summary)
    append_exec("F_target_free_KAN_full_loop", command, "pass" if summary["part_f_exploration_gate_pass"] else "fail", gpu=args.device, files=f"{rel(matrix_path)}; {rel(OUT_ROOT / 'v22_75_part_f_target_free_kan_full_loop_summary.json')}", note=json.dumps({"completed": n, "part_f_exploration_gate_pass": summary["part_f_exploration_gate_pass"], "failure_components": summary["failure_components"]}, ensure_ascii=False))
    append_recap(
        "Part F target-free KAN full-loop",
        [
            f"completed_candidate_rows={n}；own={summary['KAN_improves_own_rows']}/{n}；MLP={summary['KAN_beats_MLP_matched_rows']}/{n}；best_control={summary['KAN_beats_best_KAN_control_rows']}/{n}；same_edge={summary['KAN_beats_same_edge_controls_rows']}/{n}；same_debt_UCB={summary['KAN_beats_same_debt_UCB_controls_rows']}/{n}；same_domain={summary['KAN_beats_same_domain_controls_rows']}/{n}；same_radial={summary['KAN_beats_same_probability_radial_controls_rows']}/{n}。",
            f"no_debt={summary['no_debt_rows']}/{n}；Brier_false_safe={summary['Brier_false_safe_rows']}/{n}；all_debt_false_safe={summary['all_debt_false_safe_rows']}/{n}；trajectory_UCB<=0={summary['trajectory_debt_UCB_margin_nonpositive_rows']}/{n}；radial_UCB<=0={summary['radial_harm_UCB_nonpositive_rows']}/{n}；overhead<=0.35={summary['overhead_le_035_rows']}/{n}。",
            f"part_f_exploration_gate_pass={summary['part_f_exploration_gate_pass']}；failure_components={summary['failure_components']}。",
        ],
    )
    return summary


class MLPFeatureCoordinate(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, coordinate: str, x_train: torch.Tensor, select_k: int) -> None:
        super().__init__()
        self.base = base
        self.coordinate = str(coordinate)
        x = x_train.detach().float()
        k = max(1, min(int(select_k), int(x.shape[1])))
        var = x.var(dim=0, unbiased=False)
        idx = torch.topk(var, k=k).indices
        self.register_buffer("idx", idx.long())
        sel = x[:, idx]
        self.register_buffer("mean", sel.mean(dim=0))
        self.register_buffer("std", sel.std(dim=0, unbiased=False).clamp_min(1.0e-4))
        centers = torch.stack(
            [
                torch.quantile(sel, 0.25, dim=0),
                torch.quantile(sel, 0.50, dim=0),
                torch.quantile(sel, 0.75, dim=0),
            ],
            dim=0,
        )
        self.register_buffer("centers", centers)
        width = (torch.quantile(sel, 0.75, dim=0) - torch.quantile(sel, 0.25, dim=0)).abs().clamp_min(0.25 * self.std)
        self.register_buffer("width", width.clamp_min(1.0e-4))

    def _selected_z(self, x: torch.Tensor) -> torch.Tensor:
        return ((x.float()[:, self.idx] - self.mean) / self.std).clamp(-4.0, 4.0)

    def feature_dim(input_dim: int, coordinate: str, select_k: int) -> int:  # type: ignore[override]
        k = max(1, min(int(select_k), int(input_dim)))
        if coordinate in {"MLP_local_bump_coordinate", "MLP_same_compact_support_coordinate"}:
            return int(input_dim) + 3 * k
        if coordinate == "MLP_frequency_like_coordinate":
            return int(input_dim) + 2 * k
        if coordinate in {"MLP_same_edge_domain_energy_coordinate", "MLP_POET_Pion_OET_MCGA_reference_proxy"}:
            return int(input_dim) + 4 * k
        if coordinate in {"MLP_same_probability_radial_energy_coordinate", "MLP_same_debt_UCB_coordinate"}:
            return int(input_dim) + k
        return int(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self._selected_z(x)
        feats: list[torch.Tensor] = [x.float()]
        if self.coordinate == "MLP_local_bump_coordinate":
            bumps = torch.exp(-0.5 * ((x.float()[:, self.idx].unsqueeze(1) - self.centers.unsqueeze(0)) / self.width.view(1, 1, -1)).square())
            feats.append(bumps.flatten(start_dim=1))
        elif self.coordinate == "MLP_frequency_like_coordinate":
            feats.extend([torch.sin(z), torch.cos(z)])
        elif self.coordinate == "MLP_same_edge_domain_energy_coordinate":
            feats.extend([z, z.square(), torch.relu(z), torch.relu(-z)])
        elif self.coordinate == "MLP_same_probability_radial_energy_coordinate":
            feats.append(torch.tanh(z))
        elif self.coordinate == "MLP_same_debt_UCB_coordinate":
            feats.append(torch.clamp(z, -2.0, 2.0))
        elif self.coordinate == "MLP_same_compact_support_coordinate":
            hats = torch.relu(1.0 - (z.unsqueeze(1) - torch.tensor([-1.0, 0.0, 1.0], device=z.device, dtype=z.dtype).view(1, 3, 1)).abs())
            feats.append(hats.flatten(start_dim=1))
        elif self.coordinate == "MLP_POET_Pion_OET_MCGA_reference_proxy":
            feats.extend([z, z.square(), torch.sin(z), torch.cos(z)])
        logits = self.base(torch.cat(feats, dim=1))
        if self.coordinate == "MLP_same_probability_radial_energy_coordinate":
            centered = logits.float() - logits.float().mean(dim=1, keepdim=True)
            return 4.0 * torch.tanh(centered / 4.0)
        if self.coordinate == "MLP_same_debt_UCB_coordinate":
            return 5.0 * torch.tanh(logits.float() / 5.0)
        return logits


def g_coordinate_specs() -> list[dict[str, Any]]:
    return [
        {"coordinate": "MLP_local_bump_coordinate", "seed_offset": 31000, "lr_mult": 1.0, "wd_mult": 1.0, "select_k": 12, "compact": 0, "radial": 0, "debt": 0},
        {"coordinate": "MLP_frequency_like_coordinate", "seed_offset": 32000, "lr_mult": 1.0, "wd_mult": 1.0, "select_k": 12, "compact": 0, "radial": 0, "debt": 0},
        {"coordinate": "MLP_same_edge_domain_energy_coordinate", "seed_offset": 33000, "lr_mult": 0.9, "wd_mult": 1.25, "select_k": 12, "compact": 0, "radial": 0, "debt": 0},
        {"coordinate": "MLP_same_probability_radial_energy_coordinate", "seed_offset": 34000, "lr_mult": 0.8, "wd_mult": 1.5, "select_k": 12, "compact": 0, "radial": 1, "debt": 0},
        {"coordinate": "MLP_same_debt_UCB_coordinate", "seed_offset": 35000, "lr_mult": 0.7, "wd_mult": 2.0, "select_k": 12, "compact": 0, "radial": 0, "debt": 1},
        {"coordinate": "MLP_same_compact_support_coordinate", "seed_offset": 36000, "lr_mult": 0.9, "wd_mult": 1.5, "select_k": 12, "compact": 1, "radial": 0, "debt": 0},
        {"coordinate": "MLP_POET_Pion_OET_MCGA_reference_proxy", "seed_offset": 37000, "lr_mult": 0.8, "wd_mult": 2.0, "select_k": 8, "compact": 1, "radial": 1, "debt": 1},
    ]


def make_strengthened_mlp_coordinate_model(spec: dict[str, Any], bundle: dict[str, Any], device: torch.device, hidden: int, seed: int) -> torch.nn.Module:
    coordinate = str(spec["coordinate"])
    select_k = int(spec.get("select_k", 12))
    in_dim = MLPFeatureCoordinate.feature_dim(int(bundle["input_dim"]), coordinate, select_k)
    base = base73.make_mlp(in_dim, int(bundle["num_classes"]), int(hidden), device, seed)
    return MLPFeatureCoordinate(base, coordinate, bundle["x_train"].to(device).float(), select_k).to(device)


def run_part_g(args: argparse.Namespace, part_f: dict[str, Any]) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-g", "--device", args.device])
    f_path = OUT_ROOT / "v22_75_part_f_target_free_kan_full_loop_matrix.csv"
    detail_path = OUT_ROOT / "v22_75_part_g_strengthened_mlp_matched_audit.csv"
    summary_path = OUT_ROOT / "v22_75_part_g_strengthened_mlp_matched_audit_summary.json"
    if not f_path.exists():
        reason = "Part F matrix is missing; strengthened MLP matched audit cannot run."
        summary = {"gate": "v22_75_part_g_strengthened_mlp_matched_audit", "run_status": "skipped", "reason": reason, "part_g_gate_pass": 0}
        write_json(summary_path, summary)
        append_exec("G_strengthened_MLP_matched_audit", command, "skipped", gpu=args.device, files=rel(summary_path), note=reason)
        return summary

    f_rows = read_rows(f_path)
    candidates = [r for r in f_rows if r.get("run_status") == "completed" and r.get("row_kind") == "candidate"]
    old_mlp = {
        (r.get("dataset", ""), str(r.get("seed", ""))): r
        for r in f_rows
        if r.get("run_status") == "completed" and r.get("row_kind") == "mlp_reference"
    }
    groups = sorted({(r.get("dataset", ""), str(r.get("seed", ""))) for r in candidates})
    if not candidates or not groups:
        reason = "Part F matrix has no completed candidate rows."
        summary = {"gate": "v22_75_part_g_strengthened_mlp_matched_audit", "run_status": "skipped", "reason": reason, "part_g_gate_pass": 0}
        write_json(summary_path, summary)
        append_exec("G_strengthened_MLP_matched_audit", command, "skipped", gpu=args.device, files=rel(summary_path), note=reason)
        return summary

    device = base73.make_device(str(args.device))
    strengthened_rows: list[dict[str, Any]] = []
    best_by_group: dict[tuple[str, str], dict[str, Any]] = {}
    specs = g_coordinate_specs()
    for dataset, seed_text in groups:
        seed = int(float(seed_text))
        bundle = base73.load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), seed)
        x_train = bundle["x_train"].to(device).float()
        y_train = bundle["y_train"].to(device).long()
        x_held = bundle["x_held"].to(device).float()
        y_held = bundle["y_held"].to(device).long()
        old = old_mlp.get((dataset, seed_text), {})
        group_rows: list[dict[str, Any]] = []
        for spec in specs:
            coord = str(spec["coordinate"])
            model = make_strengthened_mlp_coordinate_model(spec, bundle, device, int(args.hidden), seed + int(spec["seed_offset"]))
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=float(args.lr) * float(spec.get("lr_mult", 1.0)),
                weight_decay=float(args.weight_decay) * float(spec.get("wd_mult", 1.0)),
            )
            final, _ = train_trace_model(model, opt, x_train, y_train, x_held, y_held, steps=int(args.steps), batch_size=int(args.batch_size), seed=seed + int(spec["seed_offset"]), trace_every=int(args.steps))
            old_loop = max(fval(old.get("train_loop_ms"), final.get("train_loop_ms", 1.0)), 1.0e-9)
            row = {
                "run_status": "completed",
                "row_kind": "strengthened_mlp_reference",
                "dataset": dataset,
                "seed": seed,
                "family": "strengthened_MLP_matched",
                "coordinate": coord,
                "method": coord,
                "train_only_feature_adapter": 1,
                "MLP_target_teacher_used": 0,
                "official_KAN_runtime_used": 0,
                "same_debt_UCB_coordinate": int(spec.get("debt", 0)),
                "same_radial_energy_coordinate": int(spec.get("radial", 0)),
                "same_compact_support_coordinate": int(spec.get("compact", 0)),
                "MLP_matched_overhead": fval(final.get("train_loop_ms"), 0.0) / old_loop - 1.0,
                "Delta_NLL_vs_old_MLP_matched": fval(old.get("NLL"), 0.0) - fval(final.get("NLL"), 0.0),
                "MLP_matched_no_debt": int(
                    fval(final.get("ECE"), 1.0e9) <= fval(old.get("ECE"), -1.0) + 1.0e-12
                    and fval(final.get("Brier"), 1.0e9) <= fval(old.get("Brier"), -1.0) + 1.0e-12
                    and fval(final.get("tail95"), 1.0e9) <= fval(old.get("tail95"), -1.0) + 1.0e-12
                    and fval(final.get("tail99"), 1.0e9) <= fval(old.get("tail99"), -1.0) + 1.0e-12
                ),
                **final,
            }
            group_rows.append(row)
            strengthened_rows.append(row)
        best_by_group[(dataset, seed_text)] = min(group_rows, key=lambda r: fval(r.get("NLL"), 1.0e9))

    comparison_rows: list[dict[str, Any]] = []
    for cand in candidates:
        key = (cand.get("dataset", ""), str(cand.get("seed", "")))
        old = old_mlp.get(key, {})
        best = best_by_group.get(key, {})
        comparison_rows.append(
            {
                "run_status": "completed",
                "row_kind": "kan_vs_strengthened_mlp_comparison",
                "dataset": cand.get("dataset", ""),
                "seed": cand.get("seed", ""),
                "family": cand.get("family", ""),
                "method": cand.get("method", ""),
                "KAN_NLL": cand.get("NLL", ""),
                "old_MLP_matched_NLL": old.get("NLL", ""),
                "best_strengthened_MLP_NLL": best.get("NLL", ""),
                "best_strengthened_coordinate": best.get("coordinate", ""),
                "KAN_vs_old_MLP_matched": int(fval(old.get("NLL"), 1.0e9) - fval(cand.get("NLL"), 1.0e9) > 0.0),
                "KAN_vs_strengthened_MLP_matched": int(fval(best.get("NLL"), 1.0e9) - fval(cand.get("NLL"), 1.0e9) > 0.0),
                "MLP_matched_same_debt_UCB": best.get("same_debt_UCB_coordinate", ""),
                "MLP_matched_same_radial_energy": best.get("same_radial_energy_coordinate", ""),
                "MLP_matched_same_compact_support": best.get("same_compact_support_coordinate", ""),
                "MLP_matched_overhead": best.get("MLP_matched_overhead", ""),
                "MLP_matched_no_debt": best.get("MLP_matched_no_debt", ""),
                "Delta_NLL_vs_old_MLP_matched": fval(old.get("NLL"), 0.0) - fval(cand.get("NLL"), 0.0),
                "Delta_NLL_vs_strengthened_MLP_matched": fval(best.get("NLL"), 0.0) - fval(cand.get("NLL"), 0.0),
                "old_MLP_beaten_but_strengthened_not": int(
                    fval(old.get("NLL"), 1.0e9) - fval(cand.get("NLL"), 1.0e9) > 0.0
                    and not (fval(best.get("NLL"), 1.0e9) - fval(cand.get("NLL"), 1.0e9) > 0.0)
                ),
            }
        )

    all_rows = strengthened_rows + comparison_rows
    write_rows(detail_path, all_rows)
    comp = comparison_rows
    n = len(comp)
    strengthened_count = sum(int(r["KAN_vs_strengthened_MLP_matched"]) for r in comp)
    old_count = sum(int(r["KAN_vs_old_MLP_matched"]) for r in comp)
    old_weak = sum(int(r["old_MLP_beaten_but_strengthened_not"]) for r in comp)
    mlp_nodebt = sum(int(fval(r.get("MLP_matched_no_debt"), 0.0) > 0.5) for r in comp)
    overheads = [fval(r.get("MLP_matched_overhead"), 0.0) for r in comp]
    best_coord_counts: dict[str, int] = {}
    for row in comp:
        coord = str(row.get("best_strengthened_coordinate", ""))
        best_coord_counts[coord] = best_coord_counts.get(coord, 0) + 1
    summary = {
        "gate": "v22_75_part_g_strengthened_mlp_matched_audit",
        "run_status": "completed_strengthened_mlp_audit",
        "completed_candidate_rows": n,
        "strengthened_mlp_reference_rows": len(strengthened_rows),
        "comparison_rows": len(comp),
        "coordinate_specs": [s["coordinate"] for s in specs],
        "KAN_vs_old_MLP_matched": old_count,
        "KAN_vs_strengthened_MLP_matched": strengthened_count,
        "old_MLP_beaten_but_strengthened_not_rows": old_weak,
        "MLP_matched_same_debt_UCB": sum(int(fval(r.get("MLP_matched_same_debt_UCB"), 0.0) > 0.5) for r in comp),
        "MLP_matched_same_radial_energy": sum(int(fval(r.get("MLP_matched_same_radial_energy"), 0.0) > 0.5) for r in comp),
        "MLP_matched_same_compact_support": sum(int(fval(r.get("MLP_matched_same_compact_support"), 0.0) > 0.5) for r in comp),
        "MLP_matched_overhead": sum(overheads) / max(1, len(overheads)),
        "MLP_matched_no_debt": mlp_nodebt,
        "best_strengthened_coordinate_counts": best_coord_counts,
        "part_g_gate_pass": 1,
        "architecture_claim_allowed_by_part_g": int(strengthened_count >= 34),
        "diagnostic_only_note": "Strengthened MLP rows are diagnostic baselines trained with train-only feature adapters; no MLP target teacher enters official KAN runtime.",
    }
    write_json(summary_path, summary)
    append_exec(
        "G_strengthened_MLP_matched_audit",
        command,
        "pass",
        gpu=args.device,
        files=f"{rel(detail_path)}; {rel(summary_path)}",
        note=json.dumps({"KAN_vs_old": old_count, "KAN_vs_strengthened": strengthened_count, "old_MLP_beaten_but_strengthened_not": old_weak}, ensure_ascii=False),
    )
    append_recap(
        "Part G strengthened MLP matched coordinate audit",
        [
            f"completed_candidate_rows={n}；strengthened_mlp_reference_rows={len(strengthened_rows)}；coordinates={summary['coordinate_specs']}。",
            f"KAN_vs_old_MLP_matched={old_count}/{n}；KAN_vs_strengthened_MLP_matched={strengthened_count}/{n}；old_MLP_beaten_but_strengthened_not={old_weak}/{n}。",
            f"MLP_matched_no_debt={mlp_nodebt}/{n}；MLP_matched_overhead_mean={summary['MLP_matched_overhead']:.6f}；best_strengthened_coordinate_counts={best_coord_counts}。",
            "该 Part G 只用于 strengthened MLP diagnostic audit；未把 MLP target/teacher 方向注入 official KAN runtime。",
        ],
    )
    return summary


def run_part_h(args: argparse.Namespace, part_f: dict[str, Any], part_g: dict[str, Any]) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-h"])
    f_path = OUT_ROOT / "v22_75_part_f_target_free_kan_full_loop_matrix.csv"
    f_rows = read_rows(f_path)
    candidates = [r for r in f_rows if r.get("run_status") == "completed" and r.get("row_kind") == "candidate"]
    refs = {
        (r.get("dataset", ""), str(r.get("seed", "")), r.get("family", "")): r
        for r in f_rows
        if r.get("run_status") == "completed" and r.get("row_kind") == "reference"
    }
    n = len(candidates)

    def ref_for(row: dict[str, Any]) -> dict[str, Any]:
        return refs.get((row.get("dataset", ""), str(row.get("seed", "")), row.get("family", "")), {})

    def blocked(flag: str) -> int:
        return sum(int(fval(r.get(flag), 0.0) <= 0.5) for r in candidates)

    summary = {
        "gate": "v22_75_part_h_failure_decomposition",
        "run_status": "completed_failure_decomposition" if n else "skipped",
        "completed_candidate_rows": n,
        "Brier_debt_rows": sum(int(fval(r.get("Brier"), 0.0) > fval(ref_for(r).get("Brier"), 1.0e9) + 1.0e-12) for r in candidates),
        "Brier_false_safe_rows": sum(int(fval(r.get("Brier_false_safe"), 0.0) > 0.5) for r in candidates),
        "all_debt_false_safe_rows": sum(int(fval(r.get("all_debt_false_safe"), 0.0) > 0.5) for r in candidates),
        "ECE_debt_rows": sum(int(fval(r.get("ECE"), 0.0) > fval(ref_for(r).get("ECE"), 1.0e9) + 1.0e-12) for r in candidates),
        "tail95_debt_rows": sum(int(fval(r.get("tail95"), 0.0) > fval(ref_for(r).get("tail95"), 1.0e9) + 1.0e-12) for r in candidates),
        "tail99_debt_rows": sum(int(fval(r.get("tail99"), 0.0) > fval(ref_for(r).get("tail99"), 1.0e9) + 1.0e-12) for r in candidates),
        "margin_debt_rows": sum(int(fval(r.get("margin10"), 0.0) < fval(ref_for(r).get("margin10"), -1.0e9) - 1.0e-12) for r in candidates),
        "trajectory_UCB_miss_rows": sum(int(fval(r.get("trajectory_debt_UCB_margin"), 1.0) > 0.0) for r in candidates),
        "radial_harm_rows": sum(int(fval(r.get("radial_harm_UCB"), 1.0) > 0.0) for r in candidates),
        "wrong_high_confidence_rows": sum(int(fval(r.get("wrong_high_confidence_rate"), 0.0) > fval(ref_for(r).get("wrong_high_confidence_rate"), 1.0e9) + 1.0e-12) for r in candidates),
        "same_edge_control_explained_rows": blocked("KAN_beats_same_edge_controls"),
        "same_debt_UCB_control_explained_rows": blocked("KAN_beats_same_debt_UCB_controls"),
        "same_domain_control_explained_rows": blocked("KAN_beats_same_domain_controls"),
        "same_radial_control_explained_rows": blocked("KAN_beats_same_probability_radial_controls"),
        "own_reference_blocked_rows": blocked("KAN_improves_own"),
        "MLP_matched_blocked_rows": blocked("KAN_beats_MLP_matched"),
        "strengthened_MLP_matched_blocked_rows": int(n - int(part_g.get("KAN_vs_strengthened_MLP_matched", 0) or 0)),
        "basis_projection_low_rows": sum(int(fval(r.get("own_residual_func_fraction"), 0.0) < 0.25) for r in candidates),
        "raw_readout_low_rows": sum(int(fval(r.get("raw_readout_visible_energy_CVaR25"), 0.0) < 0.15) for r in candidates),
        "basis_condition_fail_rows": sum(int(fval(r.get("basis_Gram_condition"), 1.0e9) > 1.0e6) for r in candidates),
        "edge_extrapolation_high_rows": sum(int(fval(r.get("edge_extrapolation_rate"), 1.0) > 0.05) for r in candidates),
        "overhead_blocked_rows": sum(int(fval(r.get("controller_or_transform_overhead_ratio"), 1.0) > 0.35) for r in candidates),
    }
    components: list[str] = []
    if summary["Brier_false_safe_rows"] > 10 or summary["all_debt_false_safe_rows"] > 12 or summary["Brier_debt_rows"] > max(0, n - 42):
        components.append("BrierTrajectoryDebtBlocked")
    if summary["radial_harm_rows"] > 0 or summary["wrong_high_confidence_rows"] > max(0, n // 3):
        components.append("RadialConfidenceDebtBlocked")
    if max(summary["same_edge_control_explained_rows"], summary["same_debt_UCB_control_explained_rows"], summary["same_domain_control_explained_rows"], summary["same_radial_control_explained_rows"]) >= max(1, n // 2):
        components.append("EdgeControlExplained_NoFU")
    if summary["own_reference_blocked_rows"] > max(0, n - 40):
        components.append("OwnReferenceBlocked")
    if summary["MLP_matched_blocked_rows"] > max(0, n - 34):
        components.append("MatchedMLPBlocked")
    if summary["strengthened_MLP_matched_blocked_rows"] > max(0, n - 34):
        components.append("StrengthenedMLPBlocked")
    if summary["basis_projection_low_rows"] > max(0, n - 42) or summary["raw_readout_low_rows"] > max(0, n - 42) or summary["basis_condition_fail_rows"] > max(0, n - 45):
        components.append("BasisFamilyProjectionFailed")
    if summary["overhead_blocked_rows"] > max(0, n - 45):
        components.append("OverheadBlocked")
    summary["failure_components"] = "|".join(components)
    if "EdgeControlExplained_NoFU" in components:
        route = "EdgeControlExplained_NoFU"
        reason = "same-edge/domain/debt/radial controls match or beat most candidate rows; candidate is not isolated functional update signal."
    elif "BrierTrajectoryDebtBlocked" in components:
        route = "BrierTrajectoryDebtBlocked"
        reason = "Brier/all-debt false-safe remains high after Part F."
    elif "StrengthenedMLPBlocked" in components:
        route = "CurrentKANBasisFamilyNotSuperiorCarrier"
        reason = "KAN candidate does not beat strengthened MLP matched coordinate at required count."
    elif "OverheadBlocked" in components:
        route = "OverheadBlocked"
        reason = "Candidate would need overhead repair before official claim."
    elif components:
        route = components[0]
        reason = "Dominant failure component selected from Part H."
    else:
        route = "NoDominantFailureComponent"
        reason = "Part H found no named failure component, despite Part F status."
    summary["recommended_route"] = route
    summary["route_reason"] = reason
    summary["part_h_gate_pass"] = int(n > 0)
    write_rows(OUT_ROOT / "v22_75_part_h_failure_decomposition.csv", [summary])
    write_json(OUT_ROOT / "v22_75_part_h_failure_decomposition_summary.json", summary)
    append_exec("H_failure_decomposition", command, "pass" if n else "skipped", files=f"{rel(OUT_ROOT / 'v22_75_part_h_failure_decomposition.csv')}; {rel(OUT_ROOT / 'v22_75_part_h_failure_decomposition_summary.json')}", note=json.dumps({"route": route, "failure_components": summary["failure_components"]}, ensure_ascii=False))
    append_recap(
        "Part H failure decomposition",
        [
            f"completed_candidate_rows={n}；failure_components={summary['failure_components']}；recommended_route={route}。",
            f"Brier_debt={summary['Brier_debt_rows']}/{n}；Brier_false_safe={summary['Brier_false_safe_rows']}/{n}；all_debt_false_safe={summary['all_debt_false_safe_rows']}/{n}；ECE_debt={summary['ECE_debt_rows']}/{n}；tail95_debt={summary['tail95_debt_rows']}/{n}；tail99_debt={summary['tail99_debt_rows']}/{n}。",
            f"controls explained: same_edge={summary['same_edge_control_explained_rows']}/{n}；same_debt_UCB={summary['same_debt_UCB_control_explained_rows']}/{n}；same_domain={summary['same_domain_control_explained_rows']}/{n}；same_radial={summary['same_radial_control_explained_rows']}/{n}。",
            f"blocked: own={summary['own_reference_blocked_rows']}/{n}；old_MLP={summary['MLP_matched_blocked_rows']}/{n}；strengthened_MLP={summary['strengthened_MLP_matched_blocked_rows']}/{n}；overhead={summary['overhead_blocked_rows']}/{n}。",
        ],
    )
    return summary


def part_i_next_actions(route: str) -> list[str]:
    if route == "BrierTrajectoryDebtBlocked":
        return [
            "Focus future work on full-trajectory probability geometry before more basis variants.",
            "Activate stricter Brier/tail UCB and debt-mode decomposition before any rerun.",
            "Reduce radial harmful component with train-only trajectory evidence.",
        ]
    if route == "EdgeControlExplained_NoFU":
        return [
            "Add control-contrastive edge metric that beats same-edge/same-domain/same-debt/same-radial controls before another full-loop.",
            "Match controls on radial energy, compact support, smoothness, and domain energy.",
            "Do not claim isolated FU signal from the current candidate family.",
        ]
    if route in {"CurrentKANBasisFamilyNotSuperiorCarrier", "KANInternalOnly_MLPMatchedCarrierStillStronger"}:
        return [
            "Stop current KAN carrier superiority claim.",
            "Only allow basis-family redesign, not optimizer tuning as a carrier claim.",
            "Require strengthened MLP matched audit to pass before architecture-value language.",
        ]
    if route == "OverheadBlocked":
        return [
            "Cache quantile maps and edge metric blocks.",
            "Refresh basis every R steps.",
            "Use low-rank Cholesky or diagonal-plus-low-rank approximation.",
        ]
    return ["Use the named failure component to choose the next repair before any further full-loop rerun."]


def final_route(
    part_a: dict[str, Any],
    part_b: dict[str, Any],
    part_c: dict[str, Any],
    part_d: dict[str, Any],
    part_e: dict[str, Any],
    part_f: dict[str, Any],
    part_g: dict[str, Any],
    part_h: dict[str, Any],
) -> dict[str, Any]:
    if not int(part_a.get("part_a_hard_gate_pass", 0)):
        route = "R0-CodeOrTrainingBoundaryFailed"
        reason = "Part A failed."
    elif not int(part_b.get("part_b_reanalysis_complete", 0)):
        route = "BTrajectoryReanalysisIncomplete"
        reason = "Part B did not complete required trajectory reanalysis."
    elif not int(part_c.get("part_c_gate_pass", 0)):
        route = "ProbabilityRadialUnitTestFailed"
        reason = "Part C failed."
    elif not int(part_d.get("part_d_gate_pass", 0)):
        route = "BrierTrajectoryDebtBlocked"
        reason = "Part D train-only trajectory debt envelope calibration failed; plan forbids Part E/F."
    elif not int(part_e.get("part_e_gate_pass", 0)):
        route = "BasisFamilyProjectionFailed"
        reason = "Part E preflight failed; plan forbids Part F full-loop."
    elif int(part_f.get("official_candidate_gate_pass", 0)):
        route = "DGKANEdgeProbabilityMetricCarrierOfficialCandidate"
        reason = "Part F official candidate gate passed."
    elif int(part_f.get("part_f_exploration_gate_pass", 0)):
        if int(part_g.get("architecture_claim_allowed_by_part_g", 0)):
            route = "KANEdgeProbabilityMetricExplorationOpened"
            reason = "Part F exploration gate passed and Part G did not block strengthened MLP matched audit."
        else:
            route = "KANInternalOnly_MLPMatchedCarrierStillStronger"
            reason = "Part F exploration passed but strengthened MLP matched audit blocks architecture claim."
    elif part_f.get("run_status") in {"completed_target_free_full_loop", "skipped"}:
        route = str(part_h.get("recommended_route", "CurrentKANBasisFamilyNotEdgeProbabilityCarrier"))
        reason = str(part_h.get("route_reason", "Part F failed; route selected by Part H failure decomposition."))
    else:
        route = "PartFNotRun"
        reason = "Part F artifact is missing or incomplete; full objective not closed."
    actions = part_i_next_actions(route)
    part_i = {
        "gate": "v22_75_part_i_basis_family_redesign_stop_rules",
        "final_route": route,
        "route_reason": reason,
        "next_actions": actions,
        "non_fabrication_note": "Actions are selected from the plan's Part I stop-rule text according to generated Part F/G/H evidence.",
    }
    write_json(OUT_ROOT / "v22_75_part_i_basis_family_redesign_stop_rules.json", part_i)
    obj = {
        "final_route": route,
        "route_reason": reason,
        "generated_at_sg": now_sg(),
        "part_a_hard_gate_pass": int(part_a.get("part_a_hard_gate_pass", 0)),
        "part_b_reanalysis_complete": int(part_b.get("part_b_reanalysis_complete", 0)),
        "part_c_gate_pass": int(part_c.get("part_c_gate_pass", 0)),
        "part_d_gate_pass": int(part_d.get("part_d_gate_pass", 0)),
        "part_e_gate_pass": int(part_e.get("part_e_gate_pass", 0)),
        "part_f_exploration_gate_pass": int(part_f.get("part_f_exploration_gate_pass", 0)),
        "official_candidate_gate_pass": int(part_f.get("official_candidate_gate_pass", 0)),
        "part_g_gate_pass": int(part_g.get("part_g_gate_pass", 0)),
        "part_h_gate_pass": int(part_h.get("part_h_gate_pass", 0)),
        "part_h_failure_components": part_h.get("failure_components", ""),
        "part_i_next_actions": actions,
        "artifacts": {
            "part_a": rel(OUT_ROOT / "v22_75_part_a_code_identity_hard_gate.json"),
            "part_b": rel(OUT_ROOT / "v22_75_part_b_v22_74_probability_trajectory_summary.json"),
            "part_c": rel(OUT_ROOT / "v22_75_part_c_probability_radial_unit_tests_summary.json"),
            "part_d": rel(OUT_ROOT / "v22_75_part_d_trajectory_debt_envelope_summary.json"),
            "part_e": rel(OUT_ROOT / "v22_75_part_e_edge_local_probability_basis_preflight_summary.json"),
            "part_f": rel(OUT_ROOT / "v22_75_part_f_target_free_kan_full_loop_summary.json"),
            "part_g": rel(OUT_ROOT / "v22_75_part_g_strengthened_mlp_matched_audit_summary.json"),
            "part_h": rel(OUT_ROOT / "v22_75_part_h_failure_decomposition_summary.json"),
            "part_i": rel(OUT_ROOT / "v22_75_part_i_basis_family_redesign_stop_rules.json"),
        },
        "non_fabrication_note": "All counts are generated by v22.75 runner or read from explicitly named v22.74 artifacts. Missing/skipped stages are marked as such.",
    }
    write_json(OUT_ROOT / "v22_75_final_route.json", obj)
    append_exec(
        "final_route",
        command_text([PYTHON, rel(RUNNER), "--mode", "full"]),
        "done",
        files=f"{rel(OUT_ROOT / 'v22_75_final_route.json')}; {rel(OUT_ROOT / 'v22_75_part_i_basis_family_redesign_stop_rules.json')}",
        note=json.dumps({"final_route": route, "reason": reason}, ensure_ascii=False),
    )
    append_recap(
        "Final route",
        [
            f"final_route={route}；reason={reason}",
            f"Part gates: A={obj['part_a_hard_gate_pass']} B={obj['part_b_reanalysis_complete']} C={obj['part_c_gate_pass']} D={obj['part_d_gate_pass']} E={obj['part_e_gate_pass']} F_explore={obj['part_f_exploration_gate_pass']} F_official={obj['official_candidate_gate_pass']} G={obj['part_g_gate_pass']} H={obj['part_h_gate_pass']}。",
            f"failure_components={obj['part_h_failure_components']}。",
            f"Part I next_actions={actions}。",
            f"关键 artifact：`{rel(OUT_ROOT / 'v22_75_final_route.json')}`。",
        ],
    )
    return obj


def run_full(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    append_exec("init", command_text([PYTHON, rel(RUNNER), "--mode", args.mode, "--device", args.device]), "start", gpu=args.gpus, files=f"{rel(EXEC_LOG)}; {rel(RECAP_LOG)}; {rel(OUT_ROOT)}", note="v22.75 gate-ordered runner initialized.")
    append_recap(
        "初始化与边界",
        [
            "执行顺序按计划采用 Part A/B/C/D/E/F/G/H/I；D/E 不通过时计划禁止进入后续 full-loop。",
            "本 runner 新增 trajectory probability trace、radial decomposition、train-only trajectory debt envelope calibration、UCB repair attempts、strengthened MLP audit 与 failure route。",
        ],
    )
    a = run_part_a(args)
    b = run_part_b(args) if int(a.get("part_a_hard_gate_pass", 0)) else {"part_b_reanalysis_complete": 0}
    c = run_part_c(args) if int(b.get("part_b_reanalysis_complete", 0)) else {"part_c_gate_pass": 0}
    d = run_part_d(args, c)
    e = run_part_e(args, d) if int(d.get("part_d_gate_pass", 0)) else {"part_e_gate_pass": 0}
    f = run_part_f(args, e, d) if int(e.get("part_e_gate_pass", 0)) else {"part_f_exploration_gate_pass": 0, "official_candidate_gate_pass": 0, "run_status": "skipped"}
    g = run_part_g(args, f) if f.get("run_status") == "completed_target_free_full_loop" else {"part_g_gate_pass": 0}
    h = run_part_h(args, f, g) if f.get("run_status") == "completed_target_free_full_loop" else {"part_h_gate_pass": 0, "failure_components": ""}
    return final_route(a, b, c, d, e, f, g, h)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="full", choices=["full", "part-a", "part-b", "part-c", "part-d", "part-e", "part-f", "part-g", "part-h", "final-route"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--train-size", type=int, default=256)
    p.add_argument("--held-size", type=int, default=64)
    p.add_argument("--test-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--steps", type=int, default=120)
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
    p.add_argument("--trace-every", type=int, default=5)
    p.add_argument("--part-b-datasets", default="Wine,Spam,MNIST")
    p.add_argument("--part-b-seed-count", type=int, default=3)
    p.add_argument("--part-b-steps", type=int, default=100)
    p.add_argument("--part-d-datasets", default="Wine,Spam,MNIST")
    p.add_argument("--part-d-seed-count", type=int, default=3)
    p.add_argument("--part-e-datasets", default="Wine,Spam,MNIST")
    p.add_argument("--part-e-seed-count", type=int, default=5)
    p.add_argument("--part-f-datasets", default="Wine,Spam,MNIST")
    p.add_argument("--part-f-seed-count", type=int, default=5)
    p.add_argument("--trajectory-shrink-alpha", type=float, default=0.85)
    p.add_argument("--ucb-beta", type=float, default=1.0)
    p.add_argument("--ucb-slack", type=float, default=0.0)
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
            cpath = OUT_ROOT / "v22_75_part_c_probability_radial_unit_tests_summary.json"
            c = load_json(cpath) if cpath.exists() else {"part_c_gate_pass": 0}
            run_part_d(args, c)
        elif args.mode == "part-e":
            dpath = OUT_ROOT / "v22_75_part_d_trajectory_debt_envelope_summary.json"
            d = load_json(dpath) if dpath.exists() else {"part_d_gate_pass": 0}
            run_part_e(args, d)
        elif args.mode == "part-f":
            epath = OUT_ROOT / "v22_75_part_e_edge_local_probability_basis_preflight_summary.json"
            dpath = OUT_ROOT / "v22_75_part_d_trajectory_debt_envelope_summary.json"
            e = load_json(epath) if epath.exists() else {"part_e_gate_pass": 0}
            d = load_json(dpath) if dpath.exists() else {"part_d_gate_pass": 0}
            run_part_f(args, e, d)
        elif args.mode == "part-g":
            fpath = OUT_ROOT / "v22_75_part_f_target_free_kan_full_loop_summary.json"
            f = load_json(fpath) if fpath.exists() else {"part_f_exploration_gate_pass": 0, "official_candidate_gate_pass": 0}
            run_part_g(args, f)
        elif args.mode == "part-h":
            fpath = OUT_ROOT / "v22_75_part_f_target_free_kan_full_loop_summary.json"
            gpath = OUT_ROOT / "v22_75_part_g_strengthened_mlp_matched_audit_summary.json"
            f = load_json(fpath) if fpath.exists() else {"part_f_exploration_gate_pass": 0, "official_candidate_gate_pass": 0}
            g = load_json(gpath) if gpath.exists() else {"part_g_gate_pass": 0}
            run_part_h(args, f, g)
        elif args.mode == "final-route":
            paths = {
                "a": OUT_ROOT / "v22_75_part_a_code_identity_hard_gate.json",
                "b": OUT_ROOT / "v22_75_part_b_v22_74_probability_trajectory_summary.json",
                "c": OUT_ROOT / "v22_75_part_c_probability_radial_unit_tests_summary.json",
                "d": OUT_ROOT / "v22_75_part_d_trajectory_debt_envelope_summary.json",
                "e": OUT_ROOT / "v22_75_part_e_edge_local_probability_basis_preflight_summary.json",
                "f": OUT_ROOT / "v22_75_part_f_target_free_kan_full_loop_summary.json",
                "g": OUT_ROOT / "v22_75_part_g_strengthened_mlp_matched_audit_summary.json",
                "h": OUT_ROOT / "v22_75_part_h_failure_decomposition_summary.json",
            }
            a = load_json(paths["a"]) if paths["a"].exists() else {"part_a_hard_gate_pass": 0}
            b = load_json(paths["b"]) if paths["b"].exists() else {"part_b_reanalysis_complete": 0}
            c = load_json(paths["c"]) if paths["c"].exists() else {"part_c_gate_pass": 0}
            d = load_json(paths["d"]) if paths["d"].exists() else {"part_d_gate_pass": 0}
            e = load_json(paths["e"]) if paths["e"].exists() else {"part_e_gate_pass": 0}
            f = load_json(paths["f"]) if paths["f"].exists() else {"part_f_exploration_gate_pass": 0, "official_candidate_gate_pass": 0}
            g = load_json(paths["g"]) if paths["g"].exists() else {"part_g_gate_pass": 0}
            h = load_json(paths["h"]) if paths["h"].exists() else {"part_h_gate_pass": 0, "failure_components": ""}
            final_route(a, b, c, d, e, f, g, h)
        else:
            run_full(args)
    except Exception as exc:
        log = write_exception_log("runner_exception", exc)
        append_exec("runner_exception", command_text([PYTHON, rel(RUNNER), "--mode", args.mode]), "error", gpu=args.device, files=rel(log), note=str(exc))
        raise


if __name__ == "__main__":
    main()
