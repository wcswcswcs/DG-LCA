#!/usr/bin/env python3
"""DG-KAN v22.62 de-risked Functional Geometry + Meta-FU triage runner."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
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

from dgkan.fu.population_snr_operator import PopulationSNRConfig, PopulationSNRPreconditioner  # noqa: E402
from dgkan.optim.ccsg_optimizer_wrapper import CCSGOptimizerWrapper  # noqa: E402
from experiments import audit_v22_61_standard_loop as loop_audit  # noqa: E402
from experiments.run_v22_56_task_compatible_witnessed_preconditioner_fu import (  # noqa: E402
    ExternalUnavailable,
    batch_indices,
    cohort_positions,
    compute_cohort_grads,
    compute_last_layer_closed_form_cohort_grads,
    continual_device_views,
    continual_metric_fields,
    continual_phase_boundary,
    continual_source_for_step,
    evaluate_tensors,
    evaluate_continual_views,
    fval,
    finalize_continual_metrics,
    iflag,
    load_bundle_tensors,
    make_base_optimizer,
    make_model,
    mean,
    margins_from_logits,
    optimizer_state_count,
    pressure_metadata_for_row,
    split_csv,
    stratified_cohort_positions,
    torch_device,
    trainable_param_count,
)


KAN_ENV_PYTHON = ROOT.parent / "miniconda3/envs/kan/bin/python"
PYTHON = os.environ.get("KAN_PYTHON", str(KAN_ENV_PYTHON if KAN_ENV_PYTHON.exists() else sys.executable))
OUT_ROOT = ROOT / "results/v22_62"
CHUNK_ROOT = OUT_ROOT / "chunks"
LOG_ROOT = OUT_ROOT / "logs"
FIG_ROOT = OUT_ROOT / "figures"
DOCS = ROOT / "docs"
EXEC_DOC = DOCS / "DG-KAN_v22.62_DeRiskedFunctionalGeometryMetaFUTriage_执行日志.md"
RECAP_DOC = DOCS / "DG-KAN_v22.62_DeRiskedFunctionalGeometryMetaFUTriage_实验结果复盘.md"
PLAN_DOC = DOCS / "DG-KAN_v22.62_DeRiskedFunctionalGeometryMetaFUTriage_完整计划.md"
RUNNER = ROOT / "experiments/run_v22_62_derisked_functional_geometry_meta_fu_triage.py"

REFERENCE_METHODS = {"adamw", "cautious_adamw", "schedule_free_adamw_local", "poet_official", "pion_oet_sphere_official", "pion_oet_local"}
PSNR_CANDIDATES = {"psnr_last_layer", "psnr_ema_last_layer", "psnr_rank2_last_layer"}
PSNR_CONTROLS = {
    "source_only_snr",
    "witness_only_snr",
    "shuffled_cohort_snr",
    "same_spectrum_random_psd",
    "same_rank_random_psd",
    "same_compute_noop",
    "hard_loss_gate",
    "loss_rank_gate",
    "inverse_class_count_gate",
}
PSNR_METHODS = PSNR_CANDIDATES | PSNR_CONTROLS
PSNR_REWEIGHTING_CONTROLS = {"hard_loss_gate", "loss_rank_gate", "inverse_class_count_gate"}
DEFAULT_REFERENCE_METHODS = "adamw,cautious_adamw,schedule_free_adamw_local"
DEFAULT_PSNR_METHODS = (
    "psnr_last_layer,psnr_ema_last_layer,psnr_rank2_last_layer,"
    "source_only_snr,witness_only_snr,shuffled_cohort_snr,"
    "same_spectrum_random_psd,same_rank_random_psd,same_compute_noop,"
    "hard_loss_gate,loss_rank_gate,inverse_class_count_gate"
)
DEFAULT_CCSG_METHODS = (
    "ccsg_last_layer,ccsg_cotangent_only,ccsg_activation_only,"
    "credit_only_operator,self_only_operator,same_spectrum_random_out_in_factor,"
    "same_rank_random_out_in_factor,source_only_crossfit_factor,witness_only_crossfit_factor,"
    "shuffled_source_witness_pairing,signflip_cotangent_factor,random_activation_factor,"
    "kfac_style_train_covariance_only,oet_self_geometry_only,same_compute_noop,hard_loss_gate,loss_rank_gate"
)
DEFAULT_SCFG_METHODS = (
    "scfg-state-transition,scfg-state-transition-tail-aware,scfg-state-transition-oet-compatible,"
    "scfg-state-transition-low-overhead,static_credit_self_operator,self_only_operator,"
    "same_generalized_spectrum_random_basis,same_compute_noop,hard_loss_gate,loss_rank_gate"
)
CCSG_CANDIDATES = {"ccsg_last_layer", "ccsg_cotangent_only", "ccsg_activation_only"}
CCSG_CONTROLS = {
    "credit_only_operator",
    "self_only_operator",
    "same_spectrum_random_out_in_factor",
    "same_rank_random_out_in_factor",
    "source_only_crossfit_factor",
    "witness_only_crossfit_factor",
    "shuffled_source_witness_pairing",
    "signflip_cotangent_factor",
    "random_activation_factor",
    "kfac_style_train_covariance_only",
    "oet_self_geometry_only",
    "same_compute_noop",
    "hard_loss_gate",
    "loss_rank_gate",
}
SCFG_ANALYTIC = {
    "scfg-state-transition",
    "scfg-state-transition-tail-aware",
    "scfg-state-transition-oet-compatible",
    "scfg-state-transition-low-overhead",
}
SCFG_CONTROLS = {"static_credit_self_operator", "self_only_operator", "same_generalized_spectrum_random_basis", "same_compute_noop", "hard_loss_gate", "loss_rank_gate"}


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.62 DeRiskedFunctionalGeometryMetaFUTriage 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只记录真实命令、解释器、GPU、输入输出文件、状态、失败与修复尝试；不写虚构数据。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.62 DeRiskedFunctionalGeometryMetaFUTriage 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只引用真实 artifact；派生指标必须说明来源；缺失指标标注未计算，不补造。\n",
            encoding="utf-8",
        )


def safe_fragment(value: Any) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))[:180]


def read_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def write_rows(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = [dict(r) for r in rows]
    if fieldnames is None:
        seen: set[str] = set()
        fieldnames = []
        for row in data:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        if not fieldnames:
            fieldnames = ["status"]
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in data:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    return json.loads(p.read_text(encoding="utf-8", errors="replace"))


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def md_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 40) -> str:
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows[:limit]:
        cells = []
        for col in columns:
            text = str(row.get(col, ""))
            if len(text) > 90:
                text = text[:87] + "..."
            cells.append(text.replace("|", "\\|").replace("\n", " "))
        out.append("| " + " | ".join(cells) + " |")
    if len(rows) > limit:
        out.append(f"\n_... {len(rows) - limit} more rows omitted_")
    return "\n".join(out) + "\n"


def append_exec(command: str, *, task_id: str, status: str, gpu: str = "", files: str = "", note: str = "", exit_code: Any = "n/a") -> None:
    ensure_out()
    row = {
        "timestamp": now_sg(),
        "task_id": task_id,
        "gpu": gpu,
        "command": command,
        "status": status,
        "exit_code": exit_code,
        "files": files,
        "note": note,
    }
    journal = read_rows(OUT_ROOT / "v22_62_command_journal.csv")
    journal.append({k: str(v) for k, v in row.items()})
    write_rows(OUT_ROOT / "v22_62_command_journal.csv", journal, ["timestamp", "task_id", "gpu", "command", "status", "exit_code", "files", "note"])
    with EXEC_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n## {row['timestamp']} {task_id}\n\n")
        f.write("```bash\n" + command + "\n```\n\n")
        f.write(f"- gpu: {gpu or 'n/a'}\n- status: {status}\n- exit_code: {exit_code}\n")
        if files:
            f.write(f"- files: {files}\n")
        if note:
            f.write(f"- note: {note}\n")


def append_recap(title: str, body: str) -> None:
    ensure_out()
    with RECAP_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n## {now_sg()} {title}\n\n{body.rstrip()}\n")


def run_cmd(cmd: list[str], *, task_id: str, files: str = "", gpu: str = "cpu", timeout: int | None = None, cwd: Path | None = None) -> dict[str, Any]:
    ensure_out()
    stdout_path = LOG_ROOT / f"{safe_fragment(task_id)}_stdout.log"
    stderr_path = LOG_ROOT / f"{safe_fragment(task_id)}_stderr.log"
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd or ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
        stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
        status = "pass" if proc.returncode == 0 else "fail"
        append_exec(
            " ".join(shlex.quote(str(x)) for x in cmd),
            task_id=task_id,
            status=status,
            gpu=gpu,
            files=files,
            exit_code=proc.returncode,
            note=f"cwd={cwd or ROOT}; stdout={stdout_path}; stderr={stderr_path}; wall_seconds={time.time() - start:.3f}",
        )
        return {"status": status, "returncode": proc.returncode, "stdout": str(stdout_path), "stderr": str(stderr_path), "wall_seconds": time.time() - start}
    except Exception as exc:
        stderr_path.write_text(traceback.format_exc(), encoding="utf-8", errors="replace")
        append_exec(
            " ".join(shlex.quote(str(x)) for x in cmd),
            task_id=task_id,
            status="exception",
            gpu=gpu,
            files=files,
            exit_code="exception",
            note=f"{type(exc).__name__}: {exc}; stderr={stderr_path}",
        )
        return {"status": "exception", "returncode": -999, "stdout": str(stdout_path), "stderr": str(stderr_path), "wall_seconds": time.time() - start}


def _need(rows: int, numerator: int, denominator: int) -> int:
    return int(math.ceil(rows * numerator / denominator)) if rows > 0 else 0


def select_psnr_params(model: Any, scope: str) -> list[Any]:
    named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if str(scope) == "all":
        return [param for _, param in named]
    if str(scope) == "last_layer":
        fc3 = getattr(model, "fc3", None)
        if fc3 is not None:
            selected = [param for param in fc3.parameters() if param.requires_grad]
            if selected:
                return selected
        if not named:
            return []
        prefix = named[-1][0].rsplit(".", 1)[0]
        return [param for name, param in named if name.startswith(prefix + ".")] or [named[-1][1]]
    raise ValueError(f"unknown PSNR parameter scope: {scope}")


def reference_method_for(method: str, args: argparse.Namespace) -> str:
    return str(args.reference_method) if method.lower() in PSNR_METHODS else method.lower()


def psnr_config_for(method: str, args: argparse.Namespace) -> PopulationSNRConfig:
    m = method.lower()
    rank = int(args.psnr_rank)
    alpha = float(args.psnr_alpha)
    ema_beta = float(args.psnr_ema_beta)
    variant = "population_snr"
    control = "none"
    if m == "psnr_rank2_last_layer":
        rank = 2
    elif m == "psnr_ema_last_layer":
        variant = "population_snr_ema"
        ema_beta = max(0.0, min(0.99, ema_beta))
        alpha = min(alpha, 0.14)
    elif m in PSNR_CONTROLS:
        control = m
        variant = "population_snr_control"
        if m in {"hard_loss_gate", "loss_rank_gate", "inverse_class_count_gate"}:
            alpha = min(alpha, 0.10)
    elif m != "psnr_last_layer":
        raise ValueError(f"unknown PSNR method: {method}")
    return PopulationSNRConfig(
        variant=variant,
        control=control,
        rank=rank,
        alpha=alpha,
        rho=float(args.psnr_rho),
        eta=float(args.psnr_eta),
        ema_beta=ema_beta,
        random_seed=int(args.seed) + int(hashlib.sha256(m.encode("utf-8")).hexdigest()[:8], 16),
        max_norm_ratio=float(args.psnr_max_norm_ratio),
    )


def write_blocked_collect(args: argparse.Namespace, reason: str, method: str, ref_method: str) -> dict[str, Any]:
    row = {
        "run_label": str(args.label),
        "run_status": "blocked",
        "blocker": reason,
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "method": method,
        "phase": "PSNR" if method in PSNR_METHODS else "reference",
        "reference_method": ref_method if method in PSNR_METHODS else "",
        "final_NLL": "",
        "standard_loop_hard_gate_pass": 0,
    }
    out = CHUNK_ROOT / f"v22_62_{safe_fragment(str(args.label))}_summary.csv"
    write_rows(out, [row])
    return row


def run_collect(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    ensure_out()
    method = str(args.method).lower()
    ref_method = reference_method_for(method, args)
    if method not in REFERENCE_METHODS and method not in PSNR_METHODS:
        raise ValueError(f"unknown method: {method}")
    device = torch_device(str(args.device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(args.seed) + 226200)
    try:
        tensors = load_bundle_tensors(str(args.dataset), int(args.train_size), int(args.held_size), int(args.test_size), int(args.seed))
    except Exception as exc:
        return write_blocked_collect(args, f"DatasetUnavailable: {exc}", method, ref_method)
    if tensors["used_fake_data"]:
        return write_blocked_collect(args, "FakeDataRejected: loader reported used_fake_data=1", method, ref_method)
    base_model = make_model(int(tensors["input_dim"]), int(tensors["num_classes"]), int(args.hidden), int(args.seed) + 226200, device)
    try:
        model, base_opt, opt_audit = make_base_optimizer(ref_method, base_model, args, device)
    except ExternalUnavailable as exc:
        return write_blocked_collect(args, str(exc), method, ref_method)
    params = [p for p in model.parameters() if p.requires_grad]
    psnr_params = select_psnr_params(model, str(args.psnr_param_scope))
    psnr_state = None
    opt = base_opt
    if method in PSNR_METHODS:
        psnr_state = PopulationSNRPreconditioner(psnr_params, psnr_config_for(method, args))
        opt = CCSGOptimizerWrapper(base_opt, psnr_state)
    x_train = tensors["x_train"].to(device)
    y_train = tensors["y_train"].to(device)
    x_held = tensors["x_held"].to(device)
    y_held = tensors["y_held"].to(device)
    x_test = tensors["x_test"].to(device)
    y_test = tensors["y_test"].to(device)
    continual_views = continual_device_views(tensors, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + int(hashlib.sha256(method.encode("utf-8")).hexdigest()[:8], 16))
    pre_train = evaluate_tensors(model, x_train, y_train, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    pre_held = evaluate_tensors(model, x_held, y_held, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    pre_test = evaluate_tensors(model, x_test, y_test, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    train_losses: list[float] = []
    step_ms: list[float] = []
    opt_ms: list[float] = []
    stats_ms: list[float] = []
    trace_events: list[str] = []
    forward_called = 0
    loss_task_backward_called = 0
    optimizer_step_called = 0
    runtime_audit = loop_audit.RuntimeTrainingLoopAudit(params)
    feature_cache: dict[str, Any] = {}
    hook_handle = None
    if method in PSNR_METHODS and str(args.psnr_param_scope) == "last_layer" and hasattr(model, "fc3"):
        def _capture_last_layer_input(_module: Any, inputs: tuple[Any, ...], _output: Any) -> None:
            if inputs:
                feature_cache["last_layer_input"] = inputs[0].detach()

        hook_handle = model.fc3.register_forward_hook(_capture_last_layer_input)
    after_old_metrics: dict[str, Any] | None = None
    phase_counts = {"old": 0, "new": 0}
    total_start = time.perf_counter()
    try:
        for step in range(int(args.steps)):
            if continual_views and step == continual_phase_boundary(int(args.steps)) and after_old_metrics is None:
                after_old_metrics = evaluate_continual_views(model, continual_views, device, int(tensors["num_classes"]), int(args.eval_batch_size), "continual_after_old")
            if continual_views:
                source_x, source_y, phase_name = continual_source_for_step(continual_views, step, int(args.steps))
                phase_counts[phase_name] = phase_counts.get(phase_name, 0) + 1
                idx = batch_indices(int(source_x.shape[0]), int(args.batch_size), generator, device)
                xb = source_x[idx]
                yb = source_y[idx].long()
            else:
                idx = batch_indices(int(x_train.shape[0]), int(args.batch_size), generator, device)
                xb = x_train[idx]
                yb = y_train[idx].long()
            opt.zero_grad(set_to_none=True)
            step_start = time.perf_counter()
            logits = model(xb).float()
            forward_called = 1
            trace_events.append("forward")
            loss_task = F.cross_entropy(logits, yb)
            if method in PSNR_METHODS and psnr_state is not None:
                stat_start = time.perf_counter()
                should_observe = step == 0 or step % max(1, int(args.psnr_refresh_interval)) == 0
                if should_observe:
                    if str(args.cohort_mode) == "label_loss_stratified":
                        per_sample = F.cross_entropy(logits.detach().float(), yb, reduction="none")
                        positions = stratified_cohort_positions(yb, per_sample, int(args.cohorts), generator, device)
                    else:
                        positions = cohort_positions(int(idx.numel()), int(args.cohorts), generator, device)
                    if str(args.psnr_param_scope) == "last_layer" and "last_layer_input" in feature_cache:
                        cohort_grads, cohort_labels, cohort_losses, cohort_margins = compute_last_layer_closed_form_cohort_grads(
                            method, logits, feature_cache["last_layer_input"], yb, positions, psnr_params, generator
                        )
                    else:
                        cohort_grads, cohort_labels, cohort_losses, cohort_margins = compute_cohort_grads(method, logits, yb, positions, psnr_params, generator)
                    psnr_state.observe(
                        cohort_grads,
                        cohort_labels=cohort_labels,
                        cohort_losses=cohort_losses,
                        cohort_margins=cohort_margins,
                        optimizer_state=getattr(base_opt, "state", {}),
                    )
                    stats_ms.append((time.perf_counter() - stat_start) * 1000.0)
                else:
                    stats_ms.append(0.0)
            runtime_audit.snapshot_before_backward()
            loss_task.backward()
            loss_task_backward_called = 1
            trace_events.append("loss_task_backward")
            runtime_audit.check_before_optimizer_step()
            opt_start = time.perf_counter()
            opt.step()
            optimizer_step_called = 1
            trace_events.append("optimizer_step")
            if ref_method == "poet_official":
                try:
                    model.merge_if_needed(step + 1)
                    trace_events.append("poet_merge_if_needed")
                except Exception:
                    trace_events.append("poet_merge_if_needed_exception")
            opt_ms.append((time.perf_counter() - opt_start) * 1000.0)
            train_losses.append(float(loss_task.detach().item()))
            step_ms.append((time.perf_counter() - step_start) * 1000.0)
    finally:
        if hook_handle is not None:
            hook_handle.remove()
    wall = time.perf_counter() - total_start
    post_train = evaluate_tensors(model, x_train, y_train, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    post_held = evaluate_tensors(model, x_held, y_held, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    post_test = evaluate_tensors(model, x_test, y_test, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    pressure_metrics: dict[str, Any] = {
        **pressure_metadata_for_row(tensors, str(args.dataset)),
        "noise_memorization_rate": "",
        "clean_label_accuracy_on_corrupted_subset": "",
        "clean_subset_NLL": "",
        "noisy_subset_NLL": "",
        "continual_old_phase_steps": phase_counts.get("old", 0),
        "continual_new_phase_steps": phase_counts.get("new", 0),
        **continual_metric_fields(),
    }
    if "train_label_corruption_mask" in tensors and int(tensors.get("pressure_train_corrupted_count", 0) or 0) > 0:
        mask = tensors["train_label_corruption_mask"].to(device).bool()
        if bool(mask.any().item()):
            x_corr = x_train[mask]
            y_noisy_corr = y_train[mask].long()
            y_clean_corr = tensors["y_train_clean"].to(device)[mask].long()
            with torch.no_grad():
                logits_corr = model(x_corr).float()
                pred_corr = logits_corr.argmax(dim=1)
                pressure_metrics["noise_memorization_rate"] = float((pred_corr == y_noisy_corr).float().mean().item())
                pressure_metrics["clean_label_accuracy_on_corrupted_subset"] = float((pred_corr == y_clean_corr).float().mean().item())
                pressure_metrics["clean_subset_NLL"] = float(F.cross_entropy(logits_corr, y_clean_corr).item())
                pressure_metrics["noisy_subset_NLL"] = float(F.cross_entropy(logits_corr, y_noisy_corr).item())
    if continual_views:
        pressure_metrics.update(finalize_continual_metrics(model, continual_views, device, int(tensors["num_classes"]), int(args.eval_batch_size), after_old_metrics))
    peak_memory = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == "cuda" else 0.0
    psnr_diag = opt.diagnostics() if method in PSNR_METHODS else {}
    psnr_observe_ms = float(psnr_diag.get("psnr_stats_ms", 0.0) or 0.0)
    psnr_transform_ms = float(psnr_diag.get("psnr_transform_ms", 0.0) or 0.0)
    psnr_diag = dict(psnr_diag)
    psnr_diag.pop("psnr_stats_ms", None)
    psnr_diag.pop("psnr_transform_ms", None)
    psnr_stats_step_ms = mean(stats_ms) if stats_ms else 0.0
    psnr_transform_step_ms = psnr_transform_ms / max(1, int(args.steps))
    full_step_ms = mean(step_ms) or 0.0
    audit_flags = runtime_audit.as_dict()
    optimizer_owned = int(psnr_diag.get("optimizer_owned_gradient_transform_pass", 0)) if method in PSNR_METHODS else 1
    no_debt = int(
        (post_held["ECE"] - pre_held["ECE"]) <= float(args.debt_tolerance)
        and (post_held["Brier"] - pre_held["Brier"]) <= float(args.debt_tolerance)
        and (post_held["tail_q95"] - pre_held["tail_q95"]) <= float(args.tail_debt_tolerance)
    )
    standard_loop_hard_gate_pass = int(
        forward_called == 1
        and loss_task_backward_called == 1
        and optimizer_step_called == 1
        and optimizer_owned == 1
        and sum(audit_flags.values()) == 0
    )
    cfg = psnr_config_for(method, args) if method in PSNR_METHODS else None
    row = {
        "run_label": str(args.label),
        "run_status": "completed",
        "phase": "PSNR" if method in PSNR_METHODS else "reference",
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "method": method,
        "objective_family": "PopulationSNR",
        "control_kind": method if method in PSNR_CONTROLS else ("candidate" if method in PSNR_CANDIDATES else "reference"),
        "reference_method": ref_method if method in PSNR_METHODS else "",
        "model_family": f"MLP_hidden{int(args.hidden)}",
        "device": str(args.device),
        "python": PYTHON,
        "dataset_source_kind": tensors.get("source_kind", ""),
        "train_size": int(x_train.shape[0]),
        "held_size": int(x_held.shape[0]),
        "test_size": int(x_test.shape[0]),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "cohorts": int(args.cohorts),
        "cohort_mode": str(args.cohort_mode),
        "psnr_refresh_interval": int(args.psnr_refresh_interval),
        "psnr_param_scope": str(args.psnr_param_scope) if method in PSNR_METHODS else "",
        "rank": int(cfg.rank) if cfg else 0,
        "operator_variant": cfg.variant if cfg else "",
        "operator_control": cfg.control if cfg else "",
        "final_NLL": post_held["NLL"],
        "accuracy": post_held["accuracy"],
        "final_accuracy": post_held["accuracy"],
        "AUC_loss_time": mean(train_losses),
        "held_train_NLL_gain": pre_train["NLL"] - post_train["NLL"],
        "test_NLL_gain": pre_test["NLL"] - post_test["NLL"],
        "test_NLL": post_test["NLL"],
        "test_accuracy": post_test["accuracy"],
        "ECE": post_held["ECE"],
        "Brier": post_held["Brier"],
        "tail_loss_q95": post_held["tail_q95"],
        "tail_loss_q99": post_held["tail_q99"],
        "margin_q10": post_held["margin_q10"],
        "pre_NLL": pre_held["NLL"],
        "held_ECE_delta": post_held["ECE"] - pre_held["ECE"],
        "held_Brier_delta": post_held["Brier"] - pre_held["Brier"],
        "held_tail_q95_delta": post_held["tail_q95"] - pre_held["tail_q95"],
        "held_tail_q99_delta": post_held["tail_q99"] - pre_held["tail_q99"],
        "no_ECE_Brier_tail_debt": no_debt,
        "train_time_sec": wall,
        "optimizer_step_ms": mean(opt_ms),
        "full_step_ms": full_step_ms,
        "psnr_stats_step_ms": psnr_stats_step_ms,
        "psnr_transform_step_ms": psnr_transform_step_ms,
        "psnr_observe_cumulative_ms": psnr_observe_ms,
        "psnr_transform_cumulative_ms": psnr_transform_ms,
        "controller_overhead_ratio": float((psnr_stats_step_ms + psnr_transform_step_ms) / max(full_step_ms, 1.0e-12)),
        "peak_memory_mb": peak_memory,
        "trainable_parameter_count": trainable_param_count(model),
        "optimizer_state_count": optimizer_state_count(opt),
        "loss_total_is_task_loss_only": 1,
        "loss_total_is_task_loss_only_for_official_psnr": 1,
        "fu_auxiliary_loss_used_official": 0,
        "forward_called": forward_called,
        "task_loss_backward_called": loss_task_backward_called,
        "loss_task_backward_called": loss_task_backward_called,
        "optimizer_step_called": optimizer_step_called,
        "standard_loop_hard_gate_pass": standard_loop_hard_gate_pass,
        "standard_loop_runtime_trace_pass": standard_loop_hard_gate_pass,
        "branch_replay_used_as_training": 0,
        "candidate_action_selection_used_for_runtime": 0,
        "cohort_topk_selection_used": 0,
        "layer_topk_selection_used": 0,
        "score_selector_used": 0,
        "class_weight_or_sampler_used_as_fu": 0,
        "uses_validation_test_future_direction": 0,
        "training_loop_trace_hash": hashlib.sha256("|".join(trace_events).encode("utf-8")).hexdigest()[:16],
        **pressure_metrics,
        **audit_flags,
        **psnr_diag,
        **opt_audit,
    }
    out = CHUNK_ROOT / f"v22_62_{safe_fragment(str(args.label))}_summary.csv"
    write_rows(out, [row])
    return row


def collect_command(spec: dict[str, Any], args: argparse.Namespace, device: str) -> list[str]:
    return [
        PYTHON,
        "experiments/run_v22_62_derisked_functional_geometry_meta_fu_triage.py",
        "--stage",
        "collect",
        "--dataset",
        str(spec["dataset"]),
        "--seed",
        str(spec["seed"]),
        "--method",
        str(spec["method"]),
        "--reference-method",
        str(spec.get("reference_method", args.reference_method)),
        "--label",
        str(spec["label"]),
        "--device",
        f"cuda:{device}" if str(device).isdigit() else str(device),
        "--train-size",
        str(args.train_size),
        "--held-size",
        str(args.held_size),
        "--test-size",
        str(args.test_size),
        "--hidden",
        str(args.hidden),
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--cohorts",
        str(args.cohorts),
        "--cohort-mode",
        str(args.cohort_mode),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--psnr-rank",
        str(args.psnr_rank),
        "--psnr-alpha",
        str(args.psnr_alpha),
        "--psnr-rho",
        str(args.psnr_rho),
        "--psnr-eta",
        str(args.psnr_eta),
        "--psnr-ema-beta",
        str(args.psnr_ema_beta),
        "--psnr-refresh-interval",
        str(args.psnr_refresh_interval),
        "--psnr-param-scope",
        str(args.psnr_param_scope),
        "--psnr-max-norm-ratio",
        str(args.psnr_max_norm_ratio),
        "--debt-tolerance",
        str(args.debt_tolerance),
        "--tail-debt-tolerance",
        str(args.tail_debt_tolerance),
    ]


def run_psnr_matrix(args: argparse.Namespace, *, phase: str) -> list[dict[str, Any]]:
    ensure_out()
    datasets = split_csv(str(args.datasets), str)
    seeds = split_csv(str(args.seeds), int)
    methods = split_csv(str(args.methods or (DEFAULT_REFERENCE_METHODS if phase == "reference" else DEFAULT_PSNR_METHODS)), str)
    specs = []
    for dataset in datasets:
        for seed in seeds:
            for method in methods:
                method_l = str(method).lower()
                label = f"{args.label_prefix}_{method_l}_{dataset}_s{seed}"
                spec = {"dataset": dataset, "seed": seed, "method": method_l, "label": label}
                if method_l in PSNR_METHODS:
                    spec["reference_method"] = reference_method_for(method_l, args)
                specs.append(spec)
    specs_path = OUT_ROOT / f"{safe_fragment(args.label_prefix)}_specs.csv"
    dispatch_path = OUT_ROOT / f"{safe_fragment(args.label_prefix)}_dispatch_status.csv"
    write_rows(specs_path, specs)
    gpus = split_csv(str(args.gpus), str) or ["cpu"]
    dispatch_rows: list[dict[str, Any]] = []

    def launch(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        i, spec = item
        gpu = gpus[i % len(gpus)]
        cmd = collect_command(spec, args, gpu)
        summary = f"results/v22_62/chunks/v22_62_{safe_fragment(str(spec['label']))}_summary.csv"
        result = run_cmd(cmd, task_id=str(spec["label"]), files=summary, gpu=str(gpu), timeout=int(args.collect_timeout))
        return {**spec, "gpu": gpu, "command": " ".join(shlex.quote(str(x)) for x in cmd), **result, "summary": summary}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futures = [ex.submit(launch, item) for item in enumerate(specs)]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            dispatch_rows.append(row)
            write_rows(dispatch_path, dispatch_rows)
    append_exec(
        f"{PYTHON} experiments/run_v22_62_derisked_functional_geometry_meta_fu_triage.py --stage {phase}",
        task_id=f"v22_62_psnr_{phase}_matrix",
        status="pass" if all(r.get("status") == "pass" for r in dispatch_rows) else "fail",
        files=f"{specs_path}; {dispatch_path}; results/v22_62/chunks/",
        note=f"rows={len(dispatch_rows)}; workers={args.workers}; gpus={args.gpus}",
    )
    return dispatch_rows


def external_collect_command(script: str, stage: str, spec: dict[str, Any], args: argparse.Namespace, device: str) -> list[str]:
    prefix = "ccsg" if "v22_57" in script else "scfg"
    param_scope = getattr(args, f"{prefix}_param_scope")
    rank = getattr(args, f"{prefix}_rank")
    alpha = getattr(args, f"{prefix}_alpha")
    beta = getattr(args, f"{prefix}_beta")
    refresh = getattr(args, f"{prefix}_refresh_interval")
    cmd = [
        PYTHON,
        script,
        "--stage",
        "collect",
        "--dataset",
        str(spec["dataset"]),
        "--seed",
        str(spec["seed"]),
        "--method",
        str(spec["method"]),
        "--reference-method",
        str(spec.get("reference_method", args.reference_method)),
        "--label",
        str(spec["label"]),
        "--device",
        f"cuda:{device}" if str(device).isdigit() else str(device),
        "--train-size",
        str(args.train_size),
        "--held-size",
        str(args.held_size),
        "--test-size",
        str(args.test_size),
        "--hidden",
        str(args.hidden),
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--cohorts",
        str(args.cohorts),
        "--cohort-mode",
        str(args.cohort_mode),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        f"--{prefix}-rank",
        str(rank),
        f"--{prefix}-alpha",
        str(alpha),
        f"--{prefix}-beta",
        str(beta),
        f"--{prefix}-refresh-interval",
        str(refresh),
        f"--{prefix}-param-scope",
        str(param_scope),
        "--debt-tolerance",
        str(args.debt_tolerance),
        "--tail-debt-tolerance",
        str(args.tail_debt_tolerance),
    ]
    if prefix == "scfg":
        cmd.extend(["--scfg-state-transition-weight", str(args.scfg_state_transition_weight), "--scfg-debt-weight", str(args.scfg_debt_weight), "--scfg-tail-weight", str(args.scfg_tail_weight)])
    return cmd


def run_external_matrix(args: argparse.Namespace, *, line: str) -> list[dict[str, Any]]:
    ensure_out()
    if line == "ccsg":
        script = "experiments/run_v22_57_credit_conditioned_self_geometry_fu.py"
        methods_default = DEFAULT_CCSG_METHODS
        candidates = CCSG_CANDIDATES
        chunk_prefix = "v22_57"
        result_dir = ROOT / "results/v22_57/chunks"
    elif line == "scfg":
        script = "experiments/run_v22_58_state_coupled_functional_geometry_fu.py"
        methods_default = DEFAULT_SCFG_METHODS
        candidates = SCFG_ANALYTIC
        chunk_prefix = "v22_58"
        result_dir = ROOT / "results/v22_58/chunks"
    else:
        raise ValueError(line)
    datasets = split_csv(str(args.datasets), str)
    seeds = split_csv(str(args.seeds), int)
    methods = split_csv(str(args.methods or methods_default), str)
    specs = []
    for dataset in datasets:
        for seed in seeds:
            for method in methods:
                method_l = str(method).lower()
                label = f"{args.label_prefix}_{method_l}_{dataset}_s{seed}"
                spec = {"dataset": dataset, "seed": seed, "method": method_l, "label": label}
                if method_l in candidates or method_l not in REFERENCE_METHODS:
                    spec["reference_method"] = args.reference_method
                specs.append(spec)
    specs_path = OUT_ROOT / f"{safe_fragment(args.label_prefix)}_specs.csv"
    dispatch_path = OUT_ROOT / f"{safe_fragment(args.label_prefix)}_dispatch_status.csv"
    write_rows(specs_path, specs)
    gpus = split_csv(str(args.gpus), str) or ["cpu"]
    dispatch_rows: list[dict[str, Any]] = []

    def launch(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        i, spec = item
        gpu = gpus[i % len(gpus)]
        cmd = external_collect_command(script, "collect", spec, args, gpu)
        summary = f"results/{chunk_prefix}/chunks/{chunk_prefix}_{safe_fragment(str(spec['label']))}_summary.csv"
        result = run_cmd(cmd, task_id=str(spec["label"]), files=summary, gpu=str(gpu), timeout=int(args.collect_timeout))
        return {**spec, "gpu": gpu, "command": " ".join(shlex.quote(str(x)) for x in cmd), **result, "summary": summary}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futures = [ex.submit(launch, item) for item in enumerate(specs)]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            dispatch_rows.append(row)
            write_rows(dispatch_path, dispatch_rows)
    append_exec(
        f"{PYTHON} {script} --stage collect matrix delegated by v22.62",
        task_id=f"v22_62_{line}_matrix",
        status="pass" if all(r.get("status") == "pass" for r in dispatch_rows) else "fail",
        files=f"{specs_path}; {dispatch_path}; {result_dir}",
        note=f"rows={len(dispatch_rows)}; workers={args.workers}; gpus={args.gpus}; no v22_57/v22_58 summarize overwrite",
    )
    return dispatch_rows


def collect_psnr_rows(include_prefixes: list[str] | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(CHUNK_ROOT.glob("v22_62_*_summary.csv")):
        for row in read_rows(path):
            label = str(row.get("run_label", ""))
            if include_prefixes and not any(label.startswith(prefix) for prefix in include_prefixes):
                continue
            rows.append(row)
    return rows


def summarize_psnr(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    include_prefixes = split_csv(str(args.summary_include_prefixes), str) if str(args.summary_include_prefixes).strip() else []
    rows = collect_psnr_rows(include_prefixes)
    route, pairwise, method_summary, gate_rows = summarize_family_rows(
        rows,
        family="PSNR",
        candidates=PSNR_CANDIDATES,
        controls=PSNR_CONTROLS,
        reweight_controls=PSNR_REWEIGHTING_CONTROLS,
        random_controls={"same_spectrum_random_psd", "same_rank_random_psd"},
        self_controls=set(),
        reference_phase="reference",
    )
    write_rows(OUT_ROOT / "v22_62_psnr_method_summary.csv", method_summary)
    write_rows(OUT_ROOT / "v22_62_psnr_pairwise.csv", pairwise)
    write_rows(OUT_ROOT / "v22_62_psnr_candidate_gate.csv", gate_rows)
    write_json(OUT_ROOT / "v22_62_psnr_route.json", route)
    append_exec(
        f"{PYTHON} experiments/run_v22_62_derisked_functional_geometry_meta_fu_triage.py --stage summarize-psnr",
        task_id="v22_62_summarize_psnr",
        status="pass",
        files="results/v22_62/v22_62_psnr_method_summary.csv; results/v22_62/v22_62_psnr_pairwise.csv; results/v22_62/v22_62_psnr_candidate_gate.csv; results/v22_62/v22_62_psnr_route.json",
        note=f"final_route={route.get('final_route')}",
    )
    append_recap(
        "Part C Population-SNR summary",
        "Method summary:\n\n"
        + md_table(method_summary, ["method", "phase", "rows", "completed_rows", "blocked_rows", "mean_final_NLL", "no_debt_rows", "standard_loop_pass_rows", "mean_overhead_ratio"], 40)
        + "\nCandidate pairwise:\n\n"
        + md_table(pairwise, ["dataset", "seed", "method", "Delta_NLL_vs_strongest", "beats_strongest", "beats_best_control", "beats_same_spectrum_random", "beats_all_reweighting_controls", "no_ECE_Brier_tail_debt", "population_SNR_positive_rank", "controller_overhead_ratio"], 80)
        + "\nGate:\n\n"
        + md_table(gate_rows, ["method", "rows", "method_gate_pass", "beats_strongest_rows", "beats_best_control_rows", "beats_same_spectrum_random_rows", "beats_all_reweighting_controls_rows", "no_debt_rows", "positive_rank_rows", "overhead_le_035_rows"], 20)
        + "\nRoute JSON:\n\n```json\n"
        + json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n",
    )
    return route


def summarize_family_rows(
    rows: list[dict[str, str]],
    *,
    family: str,
    candidates: set[str],
    controls: set[str],
    reweight_controls: set[str],
    random_controls: set[str],
    self_controls: set[str],
    reference_phase: str = "reference",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    reference_rows = [r for r in rows if r.get("phase") == reference_phase and r.get("run_status") == "completed"]
    candidate_rows = [r for r in rows if r.get("method") in candidates and r.get("run_status") == "completed"]
    method_summary = []
    for method in sorted({r.get("method", "") for r in rows}):
        group = [r for r in rows if r.get("method") == method]
        method_summary.append(
            {
                "method": method,
                "phase": group[0].get("phase", "") if group else "",
                "rows": len(group),
                "completed_rows": sum(int(r.get("run_status") == "completed") for r in group),
                "blocked_rows": sum(int(r.get("run_status") == "blocked") for r in group),
                "mean_final_NLL": mean(fval(r.get("final_NLL")) for r in group),
                "mean_accuracy": mean(fval(r.get("final_accuracy", r.get("accuracy"))) for r in group),
                "no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in group),
                "standard_loop_pass_rows": sum(iflag(r.get("standard_loop_hard_gate_pass")) for r in group),
                "mean_overhead_ratio": mean(fval(r.get("controller_overhead_ratio")) for r in group),
            }
        )
    ref_by_key = {(r.get("method"), r.get("dataset"), str(r.get("seed"))): r for r in reference_rows}
    strongest_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for r in reference_rows:
        key = (r.get("dataset", ""), str(r.get("seed", "")))
        current = strongest_by_key.get(key)
        if current is None or (fval(r.get("final_NLL"), float("inf")) or float("inf")) < (fval(current.get("final_NLL"), float("inf")) or float("inf")):
            strongest_by_key[key] = r
    nonref_rows = [r for r in rows if r.get("run_status") == "completed" and r.get("method") not in REFERENCE_METHODS]
    pairwise = []
    for r in candidate_rows:
        key = (r.get("dataset", ""), str(r.get("seed", "")))
        ref_method = r.get("reference_method", "")
        ref = ref_by_key.get((ref_method, r.get("dataset"), str(r.get("seed"))))
        strongest = strongest_by_key.get(key)
        related_controls = [x for x in nonref_rows if x.get("dataset") == r.get("dataset") and str(x.get("seed")) == str(r.get("seed")) and x.get("method") in controls and x.get("reference_method") == ref_method]
        best_control = min(related_controls, key=lambda x: fval(x.get("final_NLL"), float("inf")) or float("inf"), default=None)
        random_group = [x for x in related_controls if x.get("method") in random_controls]
        reweight_group = [x for x in related_controls if x.get("method") in reweight_controls]
        self_group = [x for x in related_controls if x.get("method") in self_controls]
        best_random = min(random_group, key=lambda x: fval(x.get("final_NLL"), float("inf")) or float("inf"), default=None)
        best_self = min(self_group, key=lambda x: fval(x.get("final_NLL"), float("inf")) or float("inf"), default=None)
        final_nll = fval(r.get("final_NLL"))
        ref_nll = fval(ref.get("final_NLL")) if ref else None
        strongest_nll = fval(strongest.get("final_NLL")) if strongest else None
        control_nll = fval(best_control.get("final_NLL")) if best_control else None
        random_nll = fval(best_random.get("final_NLL")) if best_random else None
        self_nll = fval(best_self.get("final_NLL")) if best_self else None
        pairwise.append(
            {
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "method": r.get("method", ""),
                "family": family,
                "pressure_regime": r.get("pressure_regime", "none"),
                "pressure_base_dataset": r.get("pressure_base_dataset", r.get("dataset", "")),
                "pressure_dataset_name": r.get("pressure_dataset_name", r.get("dataset", "")),
                "pressure_noise_rate": r.get("pressure_noise_rate", ""),
                "pressure_train_corrupted_count": r.get("pressure_train_corrupted_count", ""),
                "pressure_tail_imbalance_ratio": r.get("pressure_tail_imbalance_ratio", ""),
                "pressure_old_classes": r.get("pressure_old_classes", ""),
                "pressure_new_classes": r.get("pressure_new_classes", ""),
                "pressure_old_train_count": r.get("pressure_old_train_count", ""),
                "pressure_new_train_count": r.get("pressure_new_train_count", ""),
                "pressure_preference_noise_rate": r.get("pressure_preference_noise_rate", ""),
                "pressure_preference_train_flip_count": r.get("pressure_preference_train_flip_count", ""),
                "pressure_pairwise_feature_mode": r.get("pressure_pairwise_feature_mode", ""),
                "pressure_pairwise_label_rule": r.get("pressure_pairwise_label_rule", ""),
                "reference_method": ref_method,
                "reference_final_NLL": ref_nll,
                "strongest_method": strongest.get("method", "") if strongest else "",
                "strongest_final_NLL": strongest_nll,
                "final_NLL": final_nll,
                "Delta_NLL_vs_reference": (final_nll - ref_nll) if final_nll is not None and ref_nll is not None else "",
                "Delta_NLL_vs_strongest": (final_nll - strongest_nll) if final_nll is not None and strongest_nll is not None else "",
                "Delta_NLL_vs_best_control": (final_nll - control_nll) if final_nll is not None and control_nll is not None else "",
                "Delta_NLL_vs_same_spectrum_random": (final_nll - random_nll) if final_nll is not None and random_nll is not None else "",
                "Delta_NLL_vs_self_only": (final_nll - self_nll) if final_nll is not None and self_nll is not None else "",
                "beats_reference": int(final_nll is not None and ref_nll is not None and final_nll < ref_nll),
                "beats_strongest": int(final_nll is not None and strongest_nll is not None and final_nll < strongest_nll),
                "best_control_method": best_control.get("method", "") if best_control else "",
                "beats_best_control": int(final_nll is not None and control_nll is not None and final_nll < control_nll),
                "best_random_spectrum_control_method": best_random.get("method", "") if best_random else "",
                "beats_same_spectrum_random": int(final_nll is not None and random_nll is not None and final_nll < random_nll),
                "best_self_control_method": best_self.get("method", "") if best_self else "",
                "beats_self_only": int(final_nll is not None and self_nll is not None and final_nll < self_nll),
                "beats_all_reweighting_controls": int(final_nll is not None and bool(reweight_group) and all(final_nll < (fval(x.get("final_NLL"), float("inf")) or float("inf")) for x in reweight_group)),
                "no_ECE_Brier_tail_debt": r.get("no_ECE_Brier_tail_debt", ""),
                "AUC_loss_time_improvement": int((fval(r.get("AUC_loss_time")) or float("inf")) < (fval(ref.get("AUC_loss_time")) or float("inf"))) if ref else 0,
                "controller_overhead_ratio": r.get("controller_overhead_ratio", ""),
                "population_SNR_positive_rank": r.get("population_SNR_positive_rank", ""),
                "population_SNR_mean": r.get("population_SNR_mean", ""),
                "signal_energy_fraction": r.get("signal_energy_fraction", ""),
                "reservoir_energy_fraction": r.get("reservoir_energy_fraction", ""),
                "source_witness_gap": r.get("source_witness_gap", ""),
                "state_transition_prediction_error": r.get("state_transition_prediction_error", ""),
                "continual_old_phase_steps": r.get("continual_old_phase_steps", ""),
                "continual_new_phase_steps": r.get("continual_new_phase_steps", ""),
                "continual_after_old_old_NLL": r.get("continual_after_old_old_NLL", ""),
                "continual_after_old_old_accuracy": r.get("continual_after_old_old_accuracy", ""),
                "continual_final_old_NLL": r.get("continual_final_old_NLL", ""),
                "continual_final_old_accuracy": r.get("continual_final_old_accuracy", ""),
                "continual_final_new_NLL": r.get("continual_final_new_NLL", ""),
                "continual_final_new_accuracy": r.get("continual_final_new_accuracy", ""),
                "continual_old_task_forgetting_NLL": r.get("continual_old_task_forgetting_NLL", ""),
                "continual_old_task_forgetting_accuracy": r.get("continual_old_task_forgetting_accuracy", ""),
                "continual_new_task_accuracy": r.get("continual_new_task_accuracy", ""),
                "continual_worst_group_NLL": r.get("continual_worst_group_NLL", ""),
                "continual_composite_NLL": r.get("continual_composite_NLL", ""),
                "continual_memory_pass": r.get("continual_memory_pass", ""),
                "standard_loop_hard_gate_pass": r.get("standard_loop_hard_gate_pass", ""),
            }
        )
    gate_rows: list[dict[str, Any]] = []
    for method in sorted({r.get("method", "") for r in pairwise}):
        group = [r for r in pairwise if r.get("method") == method]
        n = len(group)
        row = {
            "method": method,
            "rows": n,
            "beats_strongest_rows": sum(iflag(r.get("beats_strongest")) for r in group),
            "beats_best_control_rows": sum(iflag(r.get("beats_best_control")) for r in group),
            "beats_same_spectrum_random_rows": sum(iflag(r.get("beats_same_spectrum_random")) for r in group),
            "beats_all_reweighting_controls_rows": sum(iflag(r.get("beats_all_reweighting_controls")) for r in group),
            "beats_self_only_rows": sum(iflag(r.get("beats_self_only")) for r in group),
            "no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in group),
            "AUC_loss_time_improvement_rows": sum(iflag(r.get("AUC_loss_time_improvement")) for r in group),
            "overhead_le_035_rows": sum(int((fval(r.get("controller_overhead_ratio"), float("inf")) or float("inf")) <= 0.35) for r in group),
            "positive_rank_rows": sum(int((fval(r.get("population_SNR_positive_rank"), 0.0) or 0.0) > 0.0) for r in group),
            "state_transition_error_le_025_rows": sum(int((fval(r.get("state_transition_prediction_error"), 0.0) or 0.0) <= 0.25) for r in group),
            "continual_memory_pass_rows": sum(iflag(r.get("continual_memory_pass")) for r in group),
            "standard_loop_pass_rows": sum(iflag(r.get("standard_loop_hard_gate_pass")) for r in group),
        }
        row["has_continual_rows"] = int(any(str(r.get("pressure_regime", "")) == "continual_class_incremental" for r in group))
        row["required_beats_strongest_rows"] = _need(n, 5, 9)
        row["required_best_control_rows"] = _need(n, 6, 9)
        row["required_same_spectrum_rows"] = _need(n, 6, 9)
        row["required_reweight_rows"] = _need(n, 6, 9)
        row["required_self_rows"] = _need(n, 6, 9)
        row["required_no_debt_rows"] = _need(n, 7, 9)
        row["required_overhead_rows"] = _need(n, 8, 9)
        row["required_positive_rank_rows"] = _need(n, 8, 9)
        row["required_continual_memory_rows"] = _need(n, 7, 9)
        if family == "PSNR":
            row["method_gate_pass"] = int(
                n >= 9
                and row["standard_loop_pass_rows"] == n
                and row["beats_strongest_rows"] >= row["required_beats_strongest_rows"]
                and row["beats_best_control_rows"] >= row["required_best_control_rows"]
                and row["beats_same_spectrum_random_rows"] >= row["required_same_spectrum_rows"]
                and row["beats_all_reweighting_controls_rows"] >= row["required_reweight_rows"]
                and row["no_debt_rows"] >= row["required_no_debt_rows"]
                and row["overhead_le_035_rows"] >= row["required_overhead_rows"]
                and row["positive_rank_rows"] >= row["required_positive_rank_rows"]
                and (not row["has_continual_rows"] or row["continual_memory_pass_rows"] >= row["required_continual_memory_rows"])
            )
        elif family == "CCSG":
            row["method_gate_pass"] = int(
                n >= 9
                and row["standard_loop_pass_rows"] == n
                and row["beats_strongest_rows"] >= row["required_beats_strongest_rows"]
                and row["beats_best_control_rows"] >= row["required_best_control_rows"]
                and row["beats_same_spectrum_random_rows"] >= row["required_same_spectrum_rows"]
                and row["beats_self_only_rows"] >= row["required_self_rows"]
                and row["no_debt_rows"] >= row["required_no_debt_rows"]
                and row["overhead_le_035_rows"] >= row["required_overhead_rows"]
                and (not row["has_continual_rows"] or row["continual_memory_pass_rows"] >= row["required_continual_memory_rows"])
            )
        else:
            row["method_gate_pass"] = int(
                n >= 9
                and row["standard_loop_pass_rows"] == n
                and row["beats_strongest_rows"] >= row["required_beats_strongest_rows"]
                and row["beats_best_control_rows"] >= row["required_best_control_rows"]
                and row["beats_same_spectrum_random_rows"] >= row["required_same_spectrum_rows"]
                and row["no_debt_rows"] >= row["required_no_debt_rows"]
                and row["overhead_le_035_rows"] >= row["required_overhead_rows"]
                and (not row["has_continual_rows"] or row["continual_memory_pass_rows"] >= row["required_continual_memory_rows"])
            )
        gate_rows.append(row)
    route = {
        "family": family,
        "rows": len(rows),
        "reference_rows": len(reference_rows),
        "candidate_rows": len(pairwise),
        "candidate_method_gate_rows": len(gate_rows),
        "candidate_method_gate_pass_rows": sum(iflag(r.get("method_gate_pass")) for r in gate_rows),
        "standard_loop_pass_candidate_rows": sum(iflag(r.get("standard_loop_hard_gate_pass")) for r in pairwise),
        "beats_strongest_rows": sum(iflag(r.get("beats_strongest")) for r in pairwise),
        "beats_best_control_rows": sum(iflag(r.get("beats_best_control")) for r in pairwise),
        "beats_same_spectrum_random_rows": sum(iflag(r.get("beats_same_spectrum_random")) for r in pairwise),
        "no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in pairwise),
        "overhead_le_035_rows": sum(int((fval(r.get("controller_overhead_ratio"), float("inf")) or float("inf")) <= 0.35) for r in pairwise),
        "mlp_exploration_pass": int(any(iflag(r.get("method_gate_pass")) for r in gate_rows)),
        "final_route": "",
    }
    if route["mlp_exploration_pass"]:
        route["final_route"] = f"R6-MLP{family}GeneralValueOpened"
    elif family == "PSNR" and pairwise and route["beats_same_spectrum_random_rows"] < _need(len(pairwise), 6, 9):
        route["final_route"] = "R3-SNRSupportOnly_NoFunctionalSignal"
    elif family == "CCSG" and pairwise and route["beats_same_spectrum_random_rows"] < _need(len(pairwise), 6, 9):
        route["final_route"] = "R4-FactorSupportExplained_NoFU"
    elif family == "SCFG" and pairwise and route["beats_best_control_rows"] > 0 and route["beats_strongest_rows"] == 0:
        route["final_route"] = "R5-SafetyOnlyController"
    else:
        route["final_route"] = f"{family}NoExplorationGate"
    return route, pairwise, method_summary, gate_rows


def summarize_external(args: argparse.Namespace, *, line: str) -> dict[str, Any]:
    ensure_out()
    if line == "ccsg":
        root = ROOT / "results/v22_57/chunks"
        prefix = "v22_57"
        family = "CCSG"
        candidates = CCSG_CANDIDATES
        controls = CCSG_CONTROLS
        reweight = {"hard_loss_gate", "loss_rank_gate"}
        randoms = {"same_spectrum_random_out_in_factor", "same_rank_random_out_in_factor"}
        selfs = {"self_only_operator", "kfac_style_train_covariance_only", "oet_self_geometry_only"}
    else:
        root = ROOT / "results/v22_58/chunks"
        prefix = "v22_58"
        family = "SCFG"
        candidates = SCFG_ANALYTIC
        controls = SCFG_CONTROLS
        reweight = {"hard_loss_gate", "loss_rank_gate"}
        randoms = {"same_generalized_spectrum_random_basis"}
        selfs = {"self_only_operator"}
    include_prefixes = split_csv(str(args.summary_include_prefixes), str) if str(args.summary_include_prefixes).strip() else []
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob(f"{prefix}_*_summary.csv")):
        for row in read_rows(path):
            label = str(row.get("run_label", ""))
            if include_prefixes and not any(label.startswith(p) for p in include_prefixes):
                continue
            rows.append(row)
    route, pairwise, method_summary, gate_rows = summarize_family_rows(
        rows,
        family=family,
        candidates=candidates,
        controls=controls,
        reweight_controls=reweight,
        random_controls=randoms,
        self_controls=selfs,
    )
    write_rows(OUT_ROOT / f"v22_62_{line}_method_summary.csv", method_summary)
    write_rows(OUT_ROOT / f"v22_62_{line}_pairwise.csv", pairwise)
    write_rows(OUT_ROOT / f"v22_62_{line}_candidate_gate.csv", gate_rows)
    write_json(OUT_ROOT / f"v22_62_{line}_route.json", route)
    append_exec(
        f"{PYTHON} experiments/run_v22_62_derisked_functional_geometry_meta_fu_triage.py --stage summarize-{line}",
        task_id=f"v22_62_summarize_{line}",
        status="pass",
        files=f"results/v22_62/v22_62_{line}_method_summary.csv; results/v22_62/v22_62_{line}_pairwise.csv; results/v22_62/v22_62_{line}_candidate_gate.csv; results/v22_62/v22_62_{line}_route.json",
        note=f"final_route={route.get('final_route')}",
    )
    title = "Part D Layer-Factorized CCSG summary" if line == "ccsg" else "Part E Analytic State-Constrained SCFG summary"
    append_recap(
        title,
        "Method summary:\n\n"
        + md_table(method_summary, ["method", "phase", "rows", "completed_rows", "blocked_rows", "mean_final_NLL", "no_debt_rows", "standard_loop_pass_rows", "mean_overhead_ratio"], 40)
        + "\nCandidate pairwise:\n\n"
        + md_table(pairwise, ["dataset", "seed", "method", "Delta_NLL_vs_strongest", "beats_strongest", "beats_best_control", "beats_same_spectrum_random", "beats_self_only", "no_ECE_Brier_tail_debt", "state_transition_prediction_error", "controller_overhead_ratio"], 80)
        + "\nGate:\n\n"
        + md_table(gate_rows, ["method", "rows", "method_gate_pass", "beats_strongest_rows", "beats_best_control_rows", "beats_same_spectrum_random_rows", "beats_self_only_rows", "no_debt_rows", "overhead_le_035_rows"], 20)
        + "\nRoute JSON:\n\n```json\n"
        + json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n",
    )
    return route


def _collect_external_rows_for_prefixes(root: Path, chunk_prefix: str, include_prefixes: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob(f"{chunk_prefix}_*_summary.csv")):
        for row in read_rows(path):
            label = str(row.get("run_label", ""))
            if include_prefixes and not any(label.startswith(prefix) for prefix in include_prefixes):
                continue
            rows.append(row)
    return rows


def _pressure_write_one(
    *,
    line: str,
    family: str,
    rows: list[dict[str, str]],
    candidates: set[str],
    controls: set[str],
    reweight_controls: set[str],
    random_controls: set[str],
    self_controls: set[str],
) -> dict[str, Any]:
    if not rows:
        route = {
            "family": family,
            "rows": 0,
            "candidate_method_gate_pass_rows": 0,
            "mlp_exploration_pass": 0,
            "final_route": f"{family}PressureNotRun",
        }
        write_rows(OUT_ROOT / f"v22_62_pressure_{line}_method_summary.csv", [])
        write_rows(OUT_ROOT / f"v22_62_pressure_{line}_pairwise.csv", [])
        write_rows(OUT_ROOT / f"v22_62_pressure_{line}_candidate_gate.csv", [])
        write_json(OUT_ROOT / f"v22_62_pressure_{line}_route.json", route)
        return route
    route, pairwise, method_summary, gate_rows = summarize_family_rows(
        rows,
        family=family,
        candidates=candidates,
        controls=controls,
        reweight_controls=reweight_controls,
        random_controls=random_controls,
        self_controls=self_controls,
    )
    route["pressure_summary"] = 1
    route["final_route"] = f"{route.get('final_route', family + 'NoExplorationGate')}_Pressure"
    write_rows(OUT_ROOT / f"v22_62_pressure_{line}_method_summary.csv", method_summary)
    write_rows(OUT_ROOT / f"v22_62_pressure_{line}_pairwise.csv", pairwise)
    write_rows(OUT_ROOT / f"v22_62_pressure_{line}_candidate_gate.csv", gate_rows)
    write_json(OUT_ROOT / f"v22_62_pressure_{line}_route.json", route)
    append_recap(
        f"Part F pressure summary - {family}",
        "Pressure rows are standard-loop rows on train-only stress protocols selected by label prefix. For continual rows, training is old-class phase followed by new-class phase; held/test labels remain clean.\n\n"
        "Method summary:\n\n"
        + md_table(method_summary, ["method", "phase", "rows", "completed_rows", "blocked_rows", "mean_final_NLL", "mean_accuracy", "no_debt_rows", "standard_loop_pass_rows", "mean_overhead_ratio"], 40)
        + "\nCandidate pairwise:\n\n"
        + md_table(pairwise, ["dataset", "pressure_regime", "pressure_base_dataset", "seed", "method", "Delta_NLL_vs_strongest", "beats_strongest", "beats_best_control", "beats_same_spectrum_random", "no_ECE_Brier_tail_debt", "continual_old_task_forgetting_NLL", "continual_new_task_accuracy", "continual_composite_NLL", "continual_memory_pass", "controller_overhead_ratio"], 80)
        + "\nGate:\n\n"
        + md_table(gate_rows, ["method", "rows", "method_gate_pass", "beats_strongest_rows", "beats_best_control_rows", "beats_same_spectrum_random_rows", "beats_self_only_rows", "no_debt_rows", "continual_memory_pass_rows", "overhead_le_035_rows"], 20)
        + "\nRoute JSON:\n\n```json\n"
        + json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n",
    )
    return route


def summarize_pressure(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    include_prefixes = split_csv(str(args.summary_include_prefixes), str) if str(args.summary_include_prefixes).strip() else ["v22_62_pressure"]
    psnr_rows = collect_psnr_rows(include_prefixes)
    ccsg_rows = _collect_external_rows_for_prefixes(ROOT / "results/v22_57/chunks", "v22_57", include_prefixes)
    scfg_rows = _collect_external_rows_for_prefixes(ROOT / "results/v22_58/chunks", "v22_58", include_prefixes)
    psnr_route = _pressure_write_one(
        line="psnr",
        family="PSNR",
        rows=psnr_rows,
        candidates=PSNR_CANDIDATES,
        controls=PSNR_CONTROLS,
        reweight_controls=PSNR_REWEIGHTING_CONTROLS,
        random_controls={"same_spectrum_random_psd", "same_rank_random_psd"},
        self_controls=set(),
    )
    ccsg_route = _pressure_write_one(
        line="ccsg",
        family="CCSG",
        rows=ccsg_rows,
        candidates=CCSG_CANDIDATES,
        controls=CCSG_CONTROLS,
        reweight_controls={"hard_loss_gate", "loss_rank_gate"},
        random_controls={"same_spectrum_random_out_in_factor", "same_rank_random_out_in_factor"},
        self_controls={"self_only_operator", "kfac_style_train_covariance_only", "oet_self_geometry_only"},
    )
    scfg_route = _pressure_write_one(
        line="scfg",
        family="SCFG",
        rows=scfg_rows,
        candidates=SCFG_ANALYTIC,
        controls=SCFG_CONTROLS,
        reweight_controls={"hard_loss_gate", "loss_rank_gate"},
        random_controls={"same_generalized_spectrum_random_basis"},
        self_controls={"self_only_operator"},
    )
    route = {
        "pressure_prefixes": include_prefixes,
        "psnr_pressure_route": psnr_route.get("final_route", ""),
        "ccsg_pressure_route": ccsg_route.get("final_route", ""),
        "scfg_pressure_route": scfg_route.get("final_route", ""),
        "pressure_mlp_exploration_pass": int(any(iflag(r.get("mlp_exploration_pass")) for r in [psnr_route, ccsg_route, scfg_route])),
        "pressure_rows": int(psnr_route.get("rows", 0) or 0) + int(ccsg_route.get("rows", 0) or 0) + int(scfg_route.get("rows", 0) or 0),
        "final_route": "",
    }
    route["final_route"] = "R6-MLPFUGeneralValueOpened_PressureRegime" if route["pressure_mlp_exploration_pass"] else "PartFPressureNoMLPFUGate"
    write_json(OUT_ROOT / "v22_62_pressure_route.json", route)
    append_exec(
        f"{PYTHON} experiments/run_v22_62_derisked_functional_geometry_meta_fu_triage.py --stage summarize-pressure",
        task_id="v22_62_summarize_pressure",
        status="pass",
        files="results/v22_62/v22_62_pressure_route.json; results/v22_62/v22_62_pressure_*_route.json",
        note=f"final_route={route['final_route']}; rows={route['pressure_rows']}",
    )
    append_recap(
        "Part F pressure global route",
        "Pressure route:\n\n```json\n"
        + json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n\n"
        "Boundary: pressure regimes are train-only stress protocols over real cached datasets. No sampler/class weighting, no auxiliary loss, no validation/test/future direction, no old diagnostic artifact promotion, no KAN launch unless this route opens MLP+FU gate.\n",
    )
    return route


def forbidden_patterns() -> dict[str, str]:
    return {
        "apply_flat_update_called": r"apply_" + r"flat_update\s*\(",
        "p_data_write_detected": r"\.data\s*(?:\[|\.add_|\.copy_|=)",
        "copy_param_write_detected": r"(?:Parameter|param|p)\.copy_\s*\(",
        "manual_param_update_detected": r"(?:param|p)\.add_\s*\(",
        "no_grad_param_mutation_detected": r"with\s+torch\.no_grad\s*\(\)\s*:\s*(?:\n|.){0,240}(?:param|p)\.",
        "branch_replay_used_as_training": r"branch[_ -]?replay\s*\(",
        "runtime_argmax_candidate_used": r"(?:candidate[^\n]{0,120}\.argmax\s*\(|\.argmax\s*\([^\n]{0,120}candidate)",
        "runtime_topk_candidate_used": r"(?:candidate[^\n]{0,120}\.topk\s*\(|\.topk\s*\([^\n]{0,120}candidate)",
        "class_weight_or_sampler_used_as_fu": r"(?:Weighted" + r"RandomSampler|class_" + r"weight\s*=)",
        "fu_auxiliary_loss_used_official": r"loss_total\s*=\s*loss_task\s*\+",
    }


def scan_static_files(paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    aggregate = {name: 0 for name in forbidden_patterns()}
    for path in paths:
        if not path.exists():
            rows.append({"file": str(path), "pattern": "missing", "hits": 1})
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in forbidden_patterns().items():
            hits = [m.start() for m in re.finditer(pattern, text, flags=re.MULTILINE)]
            if hits:
                aggregate[name] += len(hits)
                rows.append({"file": str(path.relative_to(ROOT)), "pattern": name, "hits": len(hits)})
    runner_text = RUNNER.read_text(encoding="utf-8", errors="replace")
    static_pass = int("logits = model(xb).float()" in runner_text and "loss_task.backward()" in runner_text and "opt.step()" in runner_text)
    return {
        "rows": rows,
        "summary": {
            **aggregate,
            "standard_loop_static_scan_pass": static_pass,
            "manual_update_forbidden_scan_pass": int(sum(aggregate.values()) == 0),
            "official_files_scanned": len(paths),
        },
    }


def clean_tarball_import_check(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="v22_62_clean_import_") as tmp:
        tmp_path = Path(tmp)
        tar_path = tmp_path / "repo_subset.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            for rel in ["dgkan", "experiments"]:
                tar.add(ROOT / rel, arcname=rel)
        extract = tmp_path / "extract"
        extract.mkdir()
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract)
        code = (
            "import experiments.run_v22_62_derisked_functional_geometry_meta_fu_triage as r; "
            "from dgkan.fu.population_snr_operator import PopulationSNRPreconditioner; "
            "print(r.__name__, PopulationSNRPreconditioner.__name__)"
        )
        res = run_cmd([PYTHON, "-c", code], task_id="v22_62_clean_tarball_self_contained_import", files="results/v22_62/v22_62_code_truth_gate.csv", timeout=int(args.compile_timeout), cwd=extract)
        return int(res.get("status") == "pass")


def run_o0(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    try:
        py_compile.compile(str(RUNNER), doraise=True)
        py_compile_runner_pass = 1
        py_compile_error = ""
    except Exception as exc:
        py_compile_runner_pass = 0
        py_compile_error = repr(exc)
    compile_result = run_cmd([PYTHON, "-m", "compileall", "-q", "dgkan", "experiments"], task_id="v22_62_compileall_repo", files="results/v22_62/v22_62_code_truth_gate.csv", timeout=int(args.compile_timeout))
    import_cmds = {
        "worktree_full_repo_import_pass": "import dgkan; import experiments.run_v22_62_derisked_functional_geometry_meta_fu_triage; print('pass')",
        "runner_core_import_pass": "import experiments.run_v22_62_derisked_functional_geometry_meta_fu_triage as r; print(r.__name__)",
        "operator_import_pass": "from dgkan.fu.population_snr_operator import PopulationSNRPreconditioner; print('pass')",
        "ccsg_runner_import_pass": "import experiments.run_v22_57_credit_conditioned_self_geometry_fu; print('pass')",
        "scfg_runner_import_pass": "import experiments.run_v22_58_state_coupled_functional_geometry_fu; print('pass')",
    }
    import_results: dict[str, int] = {}
    for key, code in import_cmds.items():
        res = run_cmd([PYTHON, "-c", code], task_id=f"v22_62_{key}", files="results/v22_62/v22_62_code_truth_gate.csv")
        import_results[key] = int(res.get("status") == "pass")
    static = scan_static_files([ROOT / "dgkan/fu/population_snr_operator.py", RUNNER, ROOT / "experiments/run_v22_57_credit_conditioned_self_geometry_fu.py", ROOT / "experiments/run_v22_58_state_coupled_functional_geometry_fu.py"])
    write_rows(OUT_ROOT / "v22_62_standard_loop_static_scan.csv", static["rows"] or [{"status": "no_forbidden_hits"}])
    runtime_args = argparse.Namespace(**vars(args))
    runtime_args.dataset = "Wine"
    runtime_args.seed = 0
    runtime_args.method = "psnr_last_layer"
    runtime_args.reference_method = "adamw"
    runtime_args.label = "v22_62_runtime_trace_psnr_smoke"
    runtime_args.device = str(args.smoke_device)
    runtime_args.steps = int(args.smoke_steps)
    runtime_args.train_size = min(int(args.train_size), 96)
    runtime_args.held_size = min(int(args.held_size), 40)
    runtime_args.test_size = min(int(args.test_size), 40)
    runtime_row = run_collect(runtime_args)
    write_rows(OUT_ROOT / "v22_62_standard_loop_runtime_trace.csv", [runtime_row])
    clean_import = clean_tarball_import_check(args)
    row = {
        "py_compile_runner_pass": py_compile_runner_pass,
        "py_compile_error": py_compile_error,
        "compileall_pass": int(compile_result.get("status") == "pass"),
        "clean_tarball_self_contained_import_pass": clean_import,
        **import_results,
        **static["summary"],
        "standard_loop_runtime_trace_pass": iflag(runtime_row.get("standard_loop_runtime_trace_pass")),
        "forward_called": iflag(runtime_row.get("forward_called")),
        "task_loss_backward_called": iflag(runtime_row.get("task_loss_backward_called")),
        "optimizer_step_called": iflag(runtime_row.get("optimizer_step_called")),
        "loss_total_is_task_loss_only": iflag(runtime_row.get("loss_total_is_task_loss_only")),
        "fu_auxiliary_loss_used_official": iflag(runtime_row.get("fu_auxiliary_loss_used_official")),
        "manual_param_update_detected": iflag(runtime_row.get("manual_param_update_detected")),
        "no_grad_param_mutation_detected": iflag(runtime_row.get("no_grad_param_mutation_detected")),
        "p_data_write_detected": iflag(runtime_row.get("p_data_write_detected")),
        "copy_param_write_detected": iflag(runtime_row.get("copy_param_write_detected")),
        "apply_flat_update_called": iflag(runtime_row.get("apply_flat_update_called")),
        "branch_replay_used_as_training": iflag(runtime_row.get("branch_replay_used_as_training")),
        "candidate_action_selection_used_for_runtime": iflag(runtime_row.get("candidate_action_selection_used_for_runtime")),
        "cohort_topk_selection_used": iflag(runtime_row.get("cohort_topk_selection_used")),
        "layer_topk_selection_used": iflag(runtime_row.get("layer_topk_selection_used")),
        "score_selector_used": iflag(runtime_row.get("score_selector_used")),
        "class_weight_or_sampler_used_as_fu": iflag(runtime_row.get("class_weight_or_sampler_used_as_fu")),
        "uses_validation_test_future_direction": iflag(runtime_row.get("uses_validation_test_future_direction")),
    }
    hard_gate_keys = [
        "compileall_pass",
        "worktree_full_repo_import_pass",
        "clean_tarball_self_contained_import_pass",
        "runner_core_import_pass",
        "operator_import_pass",
        "standard_loop_static_scan_pass",
        "standard_loop_runtime_trace_pass",
        "forward_called",
        "task_loss_backward_called",
        "optimizer_step_called",
        "loss_total_is_task_loss_only",
    ]
    forbidden_zero = [
        "fu_auxiliary_loss_used_official",
        "manual_param_update_detected",
        "no_grad_param_mutation_detected",
        "p_data_write_detected",
        "copy_param_write_detected",
        "apply_flat_update_called",
        "branch_replay_used_as_training",
        "candidate_action_selection_used_for_runtime",
        "cohort_topk_selection_used",
        "layer_topk_selection_used",
        "score_selector_used",
        "class_weight_or_sampler_used_as_fu",
        "uses_validation_test_future_direction",
    ]
    row["part_a_hard_gate_pass"] = int(all(iflag(row.get(k)) for k in hard_gate_keys) and all(iflag(row.get(k)) == 0 for k in forbidden_zero))
    write_rows(OUT_ROOT / "v22_62_code_truth_gate.csv", [row])
    append_exec(
        f"{PYTHON} experiments/run_v22_62_derisked_functional_geometry_meta_fu_triage.py --stage o0",
        task_id="v22_62_o0_code_evidence_gate",
        status="pass" if row["part_a_hard_gate_pass"] else "fail",
        files="results/v22_62/v22_62_code_truth_gate.csv; results/v22_62/v22_62_standard_loop_static_scan.csv; results/v22_62/v22_62_standard_loop_runtime_trace.csv",
        note=f"part_a_hard_gate_pass={row['part_a_hard_gate_pass']}",
    )
    append_recap(
        "O0 code/evidence gate",
        md_table([row], ["part_a_hard_gate_pass", "compileall_pass", "worktree_full_repo_import_pass", "clean_tarball_self_contained_import_pass", "standard_loop_static_scan_pass", "standard_loop_runtime_trace_pass", "manual_param_update_detected", "class_weight_or_sampler_used_as_fu"], 1),
    )
    return row


def reanalyse_v22_61(args: argparse.Namespace) -> dict[str, Any]:
    del args
    ensure_out()
    gate_rows = read_rows(ROOT / "results/v22_61/v22_61_m0o_oracle_gate.csv")
    summary_rows = read_rows(ROOT / "results/v22_61/v22_61_m0o_oracle_summary_rows.csv")
    dominance_rows = read_rows(ROOT / "results/v22_61/v22_61_operator_dominance_near_identity.csv")
    m0c_summary = read_rows(ROOT / "results/v22_61/v22_61_m0c_feature_imitation_summary.csv")
    m0c_phase = read_rows(ROOT / "results/v22_61/v22_61_m0c_phase_imitation_summary.csv")
    final_route = read_json(ROOT / "results/v22_61/v22_61_final_route.json")

    def group_for(prefix: str) -> list[dict[str, str]]:
        return [r for r in summary_rows if r.get("oracle_run_prefix") == prefix]

    main_prefix = "v22_61_m0o_objective_features"
    holdout_prefix = "v22_61_m0o_objective_features_holdout"
    main = group_for(main_prefix)
    holdout = group_for(holdout_prefix)
    analysis_rows = []
    for name, group in [("main_objective_features", main), ("holdout_objective_features", holdout)]:
        n = len(group)
        onehot = sum(int(str(r.get("oracle_best_mixture_name", "")).startswith("onehot_")) for r in group)
        a4 = sum(int("A4" in str(r.get("oracle_best_mixture_name", ""))) for r in group)
        a5 = sum(int("A5" in str(r.get("oracle_best_mixture_name", ""))) for r in group)
        a6 = sum(int("A6" in str(r.get("oracle_best_mixture_name", ""))) for r in group)
        analysis_rows.append(
            {
                "oracle_run": name,
                "rows": n,
                "oracle_best_beats_reference": sum(iflag(r.get("oracle_best_beats_reference")) for r in group),
                "oracle_best_beats_best_random_mixture": sum(iflag(r.get("oracle_best_beats_best_random_mixture")) for r in group),
                "oracle_best_no_debt": sum(iflag(r.get("oracle_best_no_debt")) for r in group),
                "oracle_operator_entropy_ge_0p4": sum(int((fval(r.get("oracle_best_operator_entropy"), 0.0) or 0.0) >= 0.4) for r in group),
                "oracle_train_query_gap_pass": sum(iflag(r.get("oracle_train_query_gap_pass")) for r in group),
                "oracle_holdout_gap_pass": sum(int(iflag(r.get("oracle_best_beats_reference")) and iflag(r.get("oracle_best_beats_best_random_mixture")) and iflag(r.get("oracle_best_no_debt"))) for r in group) if name.startswith("holdout") else "",
                "oracle_onehot_dominance_rate": onehot / max(1, n),
                "oracle_A4_state_transition_rate": a4 / max(1, n),
                "oracle_A5_oet_tangent_rate": a5 / max(1, n),
                "oracle_A6_tail_safe_rate": a6 / max(1, n),
                "oracle_time_only_gap": "artifact_missing_not_computed",
                "oracle_phase_only_gap": "artifact_missing_not_computed",
                "oracle_same_entropy_random_gap": "best_random_mixture_Delta_NLL available; same_entropy_random not separately materialized",
            }
        )
    main_row = analysis_rows[0] if analysis_rows else {}
    hold_row = analysis_rows[1] if len(analysis_rows) > 1 else {}
    strict_pass = int(
        int(main_row.get("rows", 0)) >= 15
        and int(main_row.get("oracle_best_beats_reference", 0)) >= 12
        and int(main_row.get("oracle_best_beats_best_random_mixture", 0)) >= 11
        and int(main_row.get("oracle_best_no_debt", 0)) >= 11
        and int(main_row.get("oracle_operator_entropy_ge_0p4", 0)) >= 8
        and int(hold_row.get("oracle_holdout_gap_pass", 0) or 0) >= 11
        and float(main_row.get("oracle_onehot_dominance_rate", 1.0) or 1.0) <= 0.65
    )
    m0c_best = sorted(m0c_summary + m0c_phase, key=lambda r: int(r.get("beats_random_frozen_controls_rows", 0) or 0), reverse=True)[:8]
    route = {
        "v22_61_final_route": final_route.get("final_route", ""),
        "v22_61_part_a_pass": final_route.get("part_a_pass", ""),
        "v22_62_oracle_strict_gate_pass": strict_pass,
        "oracle_route": "OracleRobustEnoughForMetaFU" if strict_pass else "R1-OracleNotRobust_MetaFUTriage",
        "m0c_controller_status": "MetaFUDeprioritized_ControllerNotIndependent",
        "evidence_note": "holdout_gap_pass is derived as beat_reference AND beat_random AND no_debt from v22.61 summary rows; time_only/same_entropy controls were not separately materialized in available artifacts.",
    }
    write_rows(OUT_ROOT / "v22_62_v22_61_oracle_reanalysis.csv", analysis_rows)
    write_rows(OUT_ROOT / "v22_62_v22_61_m0c_summary_import.csv", m0c_best)
    write_rows(OUT_ROOT / "v22_62_v22_61_operator_dominance_import.csv", dominance_rows)
    write_json(OUT_ROOT / "v22_62_meta_fu_triage_route.json", route)
    append_exec(
        "read results/v22_61 artifacts and derive v22.62 oracle robustness metrics",
        task_id="v22_62_part_a_v22_61_reanalysis",
        status="pass",
        files="results/v22_62/v22_62_v22_61_oracle_reanalysis.csv; results/v22_62/v22_62_v22_61_m0c_summary_import.csv; results/v22_62/v22_62_meta_fu_triage_route.json",
        note=f"oracle_route={route['oracle_route']}",
    )
    append_recap(
        "Part A v22.61 oracle robustness and Meta-FU triage",
        "Oracle reanalysis:\n\n"
        + md_table(analysis_rows, ["oracle_run", "rows", "oracle_best_beats_reference", "oracle_best_beats_best_random_mixture", "oracle_best_no_debt", "oracle_operator_entropy_ge_0p4", "oracle_holdout_gap_pass", "oracle_onehot_dominance_rate", "oracle_A4_state_transition_rate", "oracle_A5_oet_tangent_rate", "oracle_A6_tail_safe_rate"], 10)
        + "\nM0C imported summary, ordered by beats_random_frozen_controls_rows:\n\n"
        + md_table(m0c_best, ["method", "rows", "beats_strongest_reference_rows", "beats_random_frozen_controls_rows", "beats_credit_only_and_self_only_rows", "no_ECE_Brier_tail_debt_rows", "operator_mixture_entropy_mean"], 12)
        + "\nRoute JSON:\n\n```json\n"
        + json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n",
    )
    return route


def simple_svg(path: Path, title: str, rows: list[dict[str, Any]], x_key: str, y_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [(str(r.get(x_key, ""))[:24], fval(r.get(y_key), 0.0) or 0.0) for r in rows[:24]]
    if not data:
        data = [("no_data", 0.0)]
    vals = [v for _, v in data]
    lo = min(vals + [0.0])
    hi = max(vals + [0.0])
    span = max(1.0e-9, hi - lo)
    width = 960
    height = 420
    bar_w = max(8, int((width - 160) / max(1, len(data))))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#ffffff"/>', f'<text x="24" y="34" font-family="monospace" font-size="18" fill="#111">{title}</text>']
    zero_y = 350 - int((0.0 - lo) / span * 260)
    parts.append(f'<line x1="80" y1="{zero_y}" x2="{width-40}" y2="{zero_y}" stroke="#777" stroke-width="1"/>')
    for i, (label, val) in enumerate(data):
        x = 90 + i * bar_w
        y = 350 - int((val - lo) / span * 260)
        h = max(1, abs(zero_y - y))
        fill = "#2d6cdf" if val <= 0 else "#d04a3a"
        parts.append(f'<rect x="{x}" y="{min(y, zero_y)}" width="{max(4, bar_w-4)}" height="{h}" fill="{fill}" opacity="0.82"/>')
        parts.append(f'<text x="{x}" y="382" font-family="monospace" font-size="9" transform="rotate(45 {x},382)" fill="#222">{label}</text>')
    parts.append(f'<text x="24" y="405" font-family="monospace" font-size="12" fill="#333">{y_key}: min={lo:.6g} max={hi:.6g}</text>')
    parts.append("</svg>\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_figures(args: argparse.Namespace) -> dict[str, Any]:
    del args
    ensure_out()
    files = []
    for name, pair_path in [
        ("psnr", OUT_ROOT / "v22_62_psnr_pairwise.csv"),
        ("ccsg", OUT_ROOT / "v22_62_ccsg_pairwise.csv"),
        ("scfg", OUT_ROOT / "v22_62_scfg_pairwise.csv"),
    ]:
        rows = read_rows(pair_path)
        simple_svg(FIG_ROOT / f"v22_62_{name}_final_nll_delta.svg", f"v22.62 {name} Delta_NLL_vs_strongest", rows, "method", "Delta_NLL_vs_strongest")
        simple_svg(FIG_ROOT / f"v22_62_{name}_control_gap.svg", f"v22.62 {name} Delta_NLL_vs_best_control", rows, "method", "Delta_NLL_vs_best_control")
        simple_svg(FIG_ROOT / f"v22_62_{name}_overhead.svg", f"v22.62 {name} overhead", rows, "method", "controller_overhead_ratio")
        files.extend([str(FIG_ROOT / f"v22_62_{name}_final_nll_delta.svg"), str(FIG_ROOT / f"v22_62_{name}_control_gap.svg"), str(FIG_ROOT / f"v22_62_{name}_overhead.svg")])
    write_json(OUT_ROOT / "v22_62_figure_summary.json", {"files": files, "kan_plot": "not_generated_KAN_gate_not_opened_unless_global_route_says_otherwise"})
    append_exec(
        f"{PYTHON} experiments/run_v22_62_derisked_functional_geometry_meta_fu_triage.py --stage figures",
        task_id="v22_62_figures",
        status="pass",
        files="; ".join(files),
    )
    append_recap("Visualizations", "生成 SVG：\n\n" + "\n".join(f"- `{x}`" for x in files))
    return {"files": files}


def global_triage(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    o0 = read_rows(OUT_ROOT / "v22_62_code_truth_gate.csv")
    meta = read_json(OUT_ROOT / "v22_62_meta_fu_triage_route.json")
    psnr = read_json(OUT_ROOT / "v22_62_psnr_route.json")
    ccsg = read_json(OUT_ROOT / "v22_62_ccsg_route.json")
    scfg = read_json(OUT_ROOT / "v22_62_scfg_route.json")
    pressure = read_json(OUT_ROOT / "v22_62_pressure_route.json")
    part_a_pass = iflag(o0[0].get("part_a_hard_gate_pass")) if o0 else 0
    mlp_pass_lines = [r for r in [psnr, ccsg, scfg] if iflag(r.get("mlp_exploration_pass"))]
    pressure_pass = iflag(pressure.get("pressure_mlp_exploration_pass"))
    route = {
        "part_a_hard_gate_pass": part_a_pass,
        "meta_fu_oracle_route": meta.get("oracle_route", ""),
        "meta_fu_controller_status": meta.get("m0c_controller_status", ""),
        "psnr_route": psnr.get("final_route", ""),
        "ccsg_route": ccsg.get("final_route", ""),
        "scfg_route": scfg.get("final_route", ""),
        "pressure_route": pressure.get("final_route", "not_run"),
        "mlp_gate_opened_lines": [r.get("family") for r in mlp_pass_lines],
        "kan_gate_status": "eligible_not_run_yet" if (mlp_pass_lines or pressure_pass) else "skipped_MLP_FU_gate_not_opened",
        "final_route": "",
        "no_fake_data_claim": 1,
    }
    if not part_a_pass:
        route["final_route"] = "R0-TrainingLoopBoundaryFailed"
    elif mlp_pass_lines:
        route["final_route"] = "R6-MLPFUGeneralValueOpened"
    elif pressure_pass:
        route["final_route"] = "R6-MLPFUGeneralValueOpened_PressureRegime"
    elif meta.get("oracle_route") == "R1-OracleNotRobust_MetaFUTriage":
        suffix = "+PartFPressureNoGate" if pressure else ""
        route["final_route"] = "R1-OracleNotRobust_MetaFUTriage+NonMetaFUNoGate" + suffix
    elif scfg.get("final_route") == "R5-SafetyOnlyController":
        route["final_route"] = "R5-SafetyOnlyController"
    else:
        route["final_route"] = "NoMLPFUGate_AllControlsOrReferencesHold"
    write_json(OUT_ROOT / "v22_62_final_route.json", route)
    append_exec(
        f"{PYTHON} experiments/run_v22_62_derisked_functional_geometry_meta_fu_triage.py --stage triage",
        task_id="v22_62_global_triage",
        status="pass",
        files="results/v22_62/v22_62_final_route.json",
        note=f"final_route={route['final_route']}; kan_gate_status={route['kan_gate_status']}",
    )
    append_recap(
        "Final triage, conclusion, insight",
        "Global route:\n\n```json\n"
        + json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n\n"
        "Evidence chain:\n\n"
        "- O0 hard gate 决定实现是否可解释；若失败，科学解释停止。\n"
        "- Part A 使用 v22.61 raw artifacts 做 oracle robustness 派生分析；缺失的 time-only/same-entropy 字段没有补造。\n"
        "- Part C/D/E 只汇总 `run_status=completed` 且带 `standard_loop_hard_gate_pass` 的真实训练行。\n"
        "- KAN gate 只由 MLP+FU exploration gate 决定；未打开时不启动 KAN。\n",
    )
    return route


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="all", choices=["all", "o0", "reanalyse-v61", "collect", "psnr-reference", "psnr", "summarize-psnr", "ccsg", "summarize-ccsg", "scfg", "summarize-scfg", "summarize-pressure", "figures", "triage"])
    p.add_argument("--dataset", default="MNIST")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="adamw")
    p.add_argument("--reference-method", default="adamw")
    p.add_argument("--label", default="v22_62_collect")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--datasets", default="MNIST,FashionMNIST,Wine")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--methods", default="")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--label-prefix", default="v22_62_psnr")
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--held-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--smoke-steps", type=int, default=3)
    p.add_argument("--smoke-device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--cohorts", type=int, default=4)
    p.add_argument("--cohort-mode", default="random", choices=["random", "label_loss_stratified"])
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--psnr-rank", type=int, default=4)
    p.add_argument("--psnr-alpha", type=float, default=0.18)
    p.add_argument("--psnr-rho", type=float, default=0.25)
    p.add_argument("--psnr-eta", type=float, default=1.0e-3)
    p.add_argument("--psnr-ema-beta", type=float, default=0.90)
    p.add_argument("--psnr-refresh-interval", type=int, default=20)
    p.add_argument("--psnr-param-scope", default="last_layer", choices=["all", "last_layer"])
    p.add_argument("--psnr-max-norm-ratio", type=float, default=1.35)
    p.add_argument("--ccsg-rank", type=int, default=4)
    p.add_argument("--ccsg-alpha", type=float, default=0.18)
    p.add_argument("--ccsg-beta", type=float, default=0.18)
    p.add_argument("--ccsg-refresh-interval", type=int, default=20)
    p.add_argument("--ccsg-param-scope", default="last_layer", choices=["all", "last_layer"])
    p.add_argument("--scfg-rank", type=int, default=4)
    p.add_argument("--scfg-alpha", type=float, default=0.18)
    p.add_argument("--scfg-beta", type=float, default=0.18)
    p.add_argument("--scfg-refresh-interval", type=int, default=20)
    p.add_argument("--scfg-param-scope", default="last_layer", choices=["all", "last_layer"])
    p.add_argument("--scfg-state-transition-weight", type=float, default=0.55)
    p.add_argument("--scfg-debt-weight", type=float, default=0.35)
    p.add_argument("--scfg-tail-weight", type=float, default=0.35)
    p.add_argument("--debt-tolerance", type=float, default=1.0e-6)
    p.add_argument("--tail-debt-tolerance", type=float, default=1.0e-6)
    p.add_argument("--compile-timeout", type=int, default=600)
    p.add_argument("--collect-timeout", type=int, default=1800)
    p.add_argument("--summary-include-prefixes", default="")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_out()
    try:
        if args.stage == "o0":
            run_o0(args)
        elif args.stage == "reanalyse-v61":
            reanalyse_v22_61(args)
        elif args.stage == "collect":
            run_collect(args)
        elif args.stage == "psnr-reference":
            if not args.methods:
                args.methods = DEFAULT_REFERENCE_METHODS
            run_psnr_matrix(args, phase="reference")
        elif args.stage == "psnr":
            if not args.methods:
                args.methods = DEFAULT_PSNR_METHODS
            run_psnr_matrix(args, phase="psnr")
        elif args.stage == "summarize-psnr":
            summarize_psnr(args)
        elif args.stage == "ccsg":
            if not args.methods:
                args.methods = DEFAULT_REFERENCE_METHODS + "," + DEFAULT_CCSG_METHODS
            run_external_matrix(args, line="ccsg")
        elif args.stage == "summarize-ccsg":
            summarize_external(args, line="ccsg")
        elif args.stage == "scfg":
            if not args.methods:
                args.methods = DEFAULT_REFERENCE_METHODS + "," + DEFAULT_SCFG_METHODS
            run_external_matrix(args, line="scfg")
        elif args.stage == "summarize-scfg":
            summarize_external(args, line="scfg")
        elif args.stage == "summarize-pressure":
            summarize_pressure(args)
        elif args.stage == "figures":
            write_figures(args)
        elif args.stage == "triage":
            global_triage(args)
        elif args.stage == "all":
            run_o0(args)
            reanalyse_v22_61(args)
            psnr_ref_prefix = "v22_62_psnr_reference"
            psnr_prefix = "v22_62_psnr"
            args.label_prefix = psnr_ref_prefix
            args.methods = DEFAULT_REFERENCE_METHODS
            run_psnr_matrix(args, phase="reference")
            args.label_prefix = psnr_prefix
            args.methods = DEFAULT_PSNR_METHODS
            run_psnr_matrix(args, phase="psnr")
            args.summary_include_prefixes = f"{psnr_ref_prefix},{psnr_prefix}"
            summarize_psnr(args)
            ccsg_prefix = "v22_62_ccsg"
            args.label_prefix = ccsg_prefix
            args.methods = DEFAULT_REFERENCE_METHODS + "," + DEFAULT_CCSG_METHODS
            run_external_matrix(args, line="ccsg")
            args.summary_include_prefixes = ccsg_prefix
            summarize_external(args, line="ccsg")
            scfg_prefix = "v22_62_scfg"
            args.label_prefix = scfg_prefix
            args.methods = DEFAULT_REFERENCE_METHODS + "," + DEFAULT_SCFG_METHODS
            run_external_matrix(args, line="scfg")
            args.summary_include_prefixes = scfg_prefix
            summarize_external(args, line="scfg")
            write_figures(args)
            global_triage(args)
        return 0
    except Exception:
        err_path = LOG_ROOT / f"v22_62_stage_{safe_fragment(args.stage)}_exception.log"
        err_path.write_text(traceback.format_exc(), encoding="utf-8", errors="replace")
        append_exec(
            f"{PYTHON} experiments/run_v22_62_derisked_functional_geometry_meta_fu_triage.py --stage {args.stage}",
            task_id=f"v22_62_stage_{args.stage}_exception",
            status="exception",
            files=str(err_path),
            exit_code="exception",
        )
        append_recap("Exception / blocker", f"阶段 `{args.stage}` 出现异常，trace 已写入 `{err_path}`。本条不是科学结论，需按 blocker 修复后重跑。\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
